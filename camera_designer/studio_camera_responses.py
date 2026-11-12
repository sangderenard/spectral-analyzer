"""Copy canonical evaluated camera responses immediately before integration.

No response is reconstructed. Each RHS executes once, is copied unchanged, and
is returned to the original accumulator/multiplier. No additional RNG is drawn.
"""
import re
VERSION = 'native-camera-response-copies-v1'
RESPONSE_WORDS = 128
RESPONSE_BINDING = 24
CHANNELS = ('air_volume_add', 'area_emission_add', 'legacy_emission_add',
            'surface_field_add', 'direct_light_add', 'pbr_nee_add',
            'beer_multiplier', 'boundary_multiplier', 'legacy_transmit_multiplier',
            'surface_survival_multiplier', 'pbr_multiplier', 'air_multiplier')


def copy_camera_responses(source, *, enabled=False):
    if not enabled:
        return source
    if 'uint sci_row=' not in source:
        raise ValueError('response copies require canonical interval identity')
    if 'StudioCameraResponses' in source:
        raise ValueError('response copies already installed')
    if re.search(r'binding\s*=\s*24\s*\)', source):
        raise ValueError('camera response binding24 is occupied')
    # Capture actual NEE branch outcomes, without another shadow query/RNG.
    anchor = 'float sample_direct_light('
    if source.count(anchor) != 1:
        raise ValueError('canonical direct-light helper changed')
    source = source.replace(anchor, 'uint sci_nee_outcome=0u; int sci_nee_source=-1; int sci_nee_shadow=-1;\n'+anchor, 1)
    seams = [
        ('    src_freq_hz = 0.0;', '    sci_nee_outcome=1u;sci_nee_source=-1;sci_nee_shadow=-1;\n    src_freq_hz = 0.0;'),
        ('    SourceRec src = bdpt_sources[si];', '    sci_nee_source=si;sci_nee_outcome=2u;\n    SourceRec src = bdpt_sources[si];'),
        ('if (cos_surf <= 0.0) return 0.0;', 'if (cos_surf <= 0.0) {sci_nee_outcome=3u;return 0.0;}'),
        ('if(cos_light<=0.0)return 0.0;', 'if(cos_light<=0.0){sci_nee_outcome=4u;return 0.0;}'),
        ('if(shadow_hit>=0 && shadow_t<dist-2e-6)return 0.0;', 'sci_nee_shadow=shadow_hit;\n        if(shadow_hit>=0 && shadow_t<dist-2e-6){sci_nee_outcome=5u;return 0.0;}'),
        ('if (shadow_hit >= 0 && shadow_t < dist - 3e-4) return 0.0;', 'sci_nee_shadow=shadow_hit;\n    if (shadow_hit >= 0 && shadow_t < dist - 3e-4) {sci_nee_outcome=5u;return 0.0;}'),
        ('return direct_bands.x;', 'sci_nee_outcome=sci_nee_shadow< -1?7u:6u;return direct_bands.x;'),
        ('return energy * geometry;', 'sci_nee_outcome=sci_nee_shadow< -1?7u:6u;return energy * geometry;'),
        ('uint row=atomicAdd(sci_words[0],1u);', 'sci_nee_outcome=0u;sci_nee_source=-1;sci_nee_shadow=-1;\n    uint row=atomicAdd(sci_words[0],1u);')]
    for old,new in seams:
        if source.count(old) != 1:
            raise ValueError('canonical NEE outcome seam changed: '+old)
        source=source.replace(old,new,1)
    # Retain visitation at the actual existing air-march field read. Its alpha
    # depends on ALL active bands; presence of one colored deposit is not enough.
    has_visits = 'uint studio_scalar_visit_flags(' in source
    march = 'void march_volume_layers('
    if source.count(march)!=1:raise ValueError('original air march seam changed')
    source=source.replace(march,'bool sci_air_known=false;uint sci_air_queries=0u,sci_air_missing=0u,sci_air_flags=0u;\n'+march,1)
    air_init='    out_alpha = 0.0;'
    if source.count(air_init)!=1:raise ValueError('original air alpha initializer changed')
    source=source.replace(air_init,air_init+'\n    sci_air_queries=0u;sci_air_missing=0u;sci_air_flags=0u;\n    sci_air_known='+('true;' if has_visits else 'false;'),1)
    fetch='for (int i = 0; i < n; i++) { bv[i] = fwd_layer(i, q) * sc; tot += bv[i]; }'
    if source.count(fetch)!=1:raise ValueError('original air field query changed')
    if has_visits:
        source=source.replace(fetch,'for (int i = 0; i < n; i++) { uint flags=studio_scalar_visit_flags(pos,i);sci_air_queries++;sci_air_flags|=flags;if((flags&1u)==0u || (flags&126u)!=0u){sci_air_known=false;sci_air_missing++;} bv[i] = fwd_layer(i, q) * sc; tot += bv[i]; }',1)
    helper = '''
layout(std430,binding=24) buffer StudioCameraResponses { uint scr_words[]; };
float sci_response(uint interval,int kind,int band,float value,uint source_state) {
    if(interval==0xffffffffu)return value;
    uint row=(interval-2u)/128u;
    uint slot=2u+row*128u+uint(kind*8+band);
    if(slot>=uint(scr_words.length())){atomicOr(scr_words[1],1u);return value;}
    scr_words[slot]=floatBitsToUint(value);
    if(kind==4 || kind==5) {
        sci_words[interval+112u]=uint(sci_nee_source);
        sci_words[interval+113u]=sci_nee_outcome;
        sci_words[interval+114u]=uint(sci_nee_shadow);
    }
    if(kind==0 || kind==11){
        sci_words[interval+44u]=sci_air_queries;sci_words[interval+45u]=sci_air_missing;
        sci_words[interval+46u]=sci_air_flags;
        sci_words[interval+47u]=(uVolAlpha==0.0 || (uAirDiffuseScatter==0.0 && uAirSpecularScatter==0.0))?1u:0u;
    }
    if(isnan(value)||isinf(value))source_state=0u;
    // Separate actual-evaluation presence, including evaluated zero.
    atomicOr(scr_words[2u+row*128u+96u+uint(kind)],1u<<uint(band));
    // 2 bits per band: 0 unknown, 1 known absent, 3 known present.
    // This is source availability, independent of expression presence above.
    atomicOr(scr_words[2u+row*128u+108u+uint(kind)],source_state<<uint(2*band));
    return value;
}
'''
    if 'uint studio_scalar_visit_flags(' in source:
        helper += '''
float sci_scalar_response(uint interval,int band,float value,uint flags) {
    if(interval!=0xffffffffu){uint row=(interval-2u)/128u;
        scr_words[2u+row*128u+120u+uint(band)]=flags;}
    uint availability=(flags&126u)!=0u?0u:((flags&1u)!=0u?3u:0u);
    return sci_response(interval,3,band,value,availability);
}
'''
    source = source.replace('void main() {', helper+'\nvoid main() {', 1)
    # Exact existing expressions. A changed or absent mandatory source seam
    # raises instead of silently claiming that a response has been copied.
    expressions = [
        ('lum', '+=', r'throughput\[i\] \* vol_lum\[i\]', 0),
        ('lum', '+=', r'throughput\[i\]\*studio_emitter_radiance\(tri,f\)', 1),
        ('lum', '+=', r'throughput\[i\] \* source_radiance \* tri.albedo.w \* emission_weight', 2),
        ('lum', '+=', r'throughput\[i\] \* sample_layer_energy\(hp, i\) \* refl \* cos_in\s*\* clamp\(aw, 0.0, 1.0\) \* opacity', 3),
        ('lum', '+=', r'throughput\[i\] \* direct_layer \* refl\s*\* clamp\(aw, 0.0, 1.0\) \* opacity', 4),
        ('throughput', '*=', r'studio_medium_transmittance\(currentMedium,frequency,surf_t\)', 6),
        ('throughput', '*=', r'factor', 7),
        ('throughput', '*=', r'exp\(-abso \* surf_t \* \(1.0 - chan\)\)', 8),
        ('throughput', '*=', r'rho / survival', 9),
        ('throughput', '*=', r'\(1.0 - vol_alpha\)', 11),
    ]
    if 'bool studio_pbr_opaque(' in source:
        expressions += [
            ('lum', '+=', r'throughput\[i\]\*direct_bands\[i\]\*STUDIO_PBR_PI\*f', 5),
            ('throughput', '*=', r'studio_pbr_weight\(tri,incoming,outgoing,n,studio_band_frequency\(i\),pdf,delta\)', 10)]
    for array, op, rhs, kind in expressions:
        pattern = re.compile(r'('+array+r'\[i\]\s*'+re.escape(op)+r'\s*)('+rhs+r')(\s*;)')
        # Material/emission laws are actual evaluated inputs. Field/NEE/air
        # queries currently lack a canonical visitation/outcome bit: unknown.
        state = '(currentMedium==-2?0u:3u)' if kind == 6 else ('3u' if kind in (1, 2, 7, 8, 9, 10) else '0u')
        if kind in (4,5):
            state='(uSourceCount==0 || (sci_nee_outcome>=3u && sci_nee_outcome<=6u)?3u:0u)'
        if kind in (0,11):
            state='(uVolAlpha==0.0 || (uAirDiffuseScatter==0.0 && uAirSpecularScatter==0.0) || sci_air_known?3u:0u)'
        if kind==3 and 'uint studio_scalar_visit_flags(' in source:
            source,count=pattern.subn(lambda m:m[1]+'sci_scalar_response(sci_row,i,('+m[2]+'),studio_scalar_visit_flags(hp,i))'+m[3],source)
        else:
            source, count = pattern.subn(lambda m: m[1]+f'sci_response(sci_row,{kind},i,('+m[2]+f'),{state})'+m[3], source)
        if count != 1:
            raise ValueError('canonical camera response seam changed: '+CHANNELS[kind])
    return source


def scalar_source_state(flags):
    """Metadata policy, not convergence: lossy/invalid samples stay unknown."""
    if type(flags) is not int or not 0 <= flags <= 127:
        raise ValueError('invalid original scalar query flags')
    return 0 if flags & 126 else (3 if flags & 1 else 0)
