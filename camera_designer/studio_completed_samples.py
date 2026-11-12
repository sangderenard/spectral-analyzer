"""Copy an existing completed camera sample before film/pixel integration.

Native lum is copied verbatim. Component sums only mirror the SAME original
additions; no response or path factor is re-evaluated. No pixel averaging here.
"""
VERSION='native-camera-scene-sample-v1'
BINDING=25
WORDS=128


def _once(source,old,new):
    if source.count(old)!=1:raise ValueError('completed sample copy seam changed: '+old[:80])
    return source.replace(old,new,1)


def with_completed_camera_samples(source):
    if 'StudioCompletedSamples' in source:raise ValueError('completed sample copy already installed')
    if 'int studio_field_band=' in source:
        raise ValueError('complete scene sample copy currently preserves the original shared spectral path only')
    if 'float sci_response(' not in source:raise ValueError('original response-copy hooks required')
    declarations=r'''
layout(std430,binding=25) buffer StudioCompletedSamples {uint scs_words[];};
uniform uint uCompletedSampleCapacity;
uniform bool uCompletedSamplesEnabled;
const bool uCameraIntervalsEnabled=false;
float scs_volume[8],scs_surface[8];
uint scs_prefix_known[8],scs_volume_known[8],scs_surface_known[8];
uint scs_termination=0u;bool scs_has_recipient=false;bool scs_position_written=false;
void scs_add(int kind,int band,float value,uint known){
    bool valid=known==3u && !isnan(value) && !isinf(value);
    if(kind>=6){scs_prefix_known[band]&=valid?1u:0u;return;}
    uint available=valid && scs_prefix_known[band]!=0u?1u:0u;
    if(kind==0 && !scs_has_recipient){scs_volume[band]+=value;scs_volume_known[band]&=available;}
    else {scs_surface[band]+=value;scs_surface_known[band]&=available;}
}
uint scs_write(uint invocation,int sample_id,int hit,int kind,int owner,int stage,
               int bands,ivec2 pixel,vec3 position,vec3 direction,int medium,
               vec3 camera_origin,vec3 camera_direction,float prefix[8],
               float lum[8],uint regions[32],uvec4 material){
    if(!uCompletedSamplesEnabled)return 0xffffffffu;
    uint row=atomicAdd(scs_words[0],1u);
    uint capacity=min(uCompletedSampleCapacity,uint(max(scs_words.length()-2,0))/128u);
    if(row>=capacity){atomicOr(scs_words[1],1u);return 0xffffffffu;}
    uint o=2u+row*128u;for(uint j=0u;j<128u;++j)scs_words[o+j]=0u;
    scs_words[o]=invocation;scs_words[o+1u]=uint(sample_id);scs_words[o+2u]=uint(hit);
    scs_words[o+3u]=uint(kind);scs_words[o+4u]=uint(owner);scs_words[o+5u]=uint(stage);
    scs_words[o+6u]=uint(uFixed3Transport);scs_words[o+7u]=uint(bands);
    for(int j=0;j<3;++j){
        scs_words[o+8u+uint(j)]=floatBitsToUint(position[j]);
        scs_words[o+12u+uint(j)]=floatBitsToUint(direction[j]);
        scs_words[o+16u+uint(j)]=floatBitsToUint(camera_origin[j]);
        scs_words[o+116u+uint(j)]=floatBitsToUint(camera_direction[j]);
    }
    scs_words[o+11u]=scs_position_written?1u:0u;scs_words[o+15u]=uint(medium);
    __REGION_COUNT__
    for(int j=0;j<8;++j){
        scs_words[o+20u+uint(j)]=floatBitsToUint(prefix[j]);
        scs_words[o+28u+uint(j)]=floatBitsToUint(lum[j]);
        scs_words[o+72u+uint(j)]=floatBitsToUint(scs_volume[j]);
        scs_words[o+80u+uint(j)]=floatBitsToUint(scs_surface[j]);
        bool bounded=scs_termination==0u;
        scs_words[o+88u+uint(j)]=j<bands && scs_volume_known[j]!=0u?3u:0u;
        scs_words[o+96u+uint(j)]=j<bands && bounded && scs_surface_known[j]!=0u?3u:0u;
        scs_words[o+104u+uint(j)]=j<bands && bounded && scs_volume_known[j]!=0u && scs_surface_known[j]!=0u?3u:0u;
    }
    for(uint j=0u;j<32u;++j)scs_words[o+36u+j]=regions[j];
    for(int j=0;j<4;++j)scs_words[o+68u+uint(j)]=material[j];
    scs_words[o+112u]=scs_termination;scs_words[o+113u]=uint(pixel.x);scs_words[o+114u]=uint(pixel.y);
    scs_words[o+115u]=scs_position_written?1u:0u;
    return o;
}
'''
    regions='uint studio_active[32]' in source
    declarations=declarations.replace('__REGION_COUNT__','scs_words[o+19u]=uint(uStudioRegionCount);' if regions else '')
    source=_once(source,'uint sci_nee_outcome=0u;',declarations+'\nuint sci_nee_outcome=0u;')
    source=_once(source,'float sci_response(uint interval,int kind,int band,float value,uint source_state) {',
        'float sci_response(uint interval,int kind,int band,float value,uint source_state) {\n    scs_add(kind,band,value,source_state);')
    source=_once(source,'    sci_nee_outcome=0u;sci_nee_source=-1;sci_nee_shadow=-1;\n    uint row=atomicAdd(sci_words[0],1u);',
        '    sci_nee_outcome=0u;sci_nee_source=-1;sci_nee_shadow=-1;\n    if(!uCameraIntervalsEnabled)return 0xffffffffu;\n    uint row=atomicAdd(sci_words[0],1u);')
    reset='''
        vec3 scs_camera_origin=ro,scs_camera_direction=rd,scs_first_direction=vec3(0.0);
        int scs_first_medium=-2;float scs_first_prefix[8];uint scs_first_regions[32];
        for(int j=0;j<8;++j){scs_first_prefix[j]=0.0;scs_volume[j]=0.0;scs_surface[j]=0.0;
            scs_prefix_known[j]=1u;scs_volume_known[j]=1u;scs_surface_known[j]=1u;}
        for(int j=0;j<32;++j)scs_first_regions[j]=0u;
        scs_termination=0u;scs_has_recipient=false;scs_position_written=false;
'''
    source=_once(source,'int sci_first_hit=-1;',reset+'\n        int sci_first_hit=-1;')
    first='if(sci_first_interval)sci_first_position=hp;'
    save='''if(sci_first_interval){scs_has_recipient=true;scs_position_written=true;sci_first_position=hp;scs_first_direction=rd;scs_first_medium=currentMedium;
                for(int j=0;j<8;++j)scs_first_prefix[j]=throughput[j];
                __FIRST_REGIONS__
            }'''
    save=save.replace('__FIRST_REGIONS__','for(int j=0;j<32;++j)scs_first_regions[j]=studio_active[j];' if regions else '')
    source=_once(source,first,save)
    source=_once(source,'for (int bounce = 0; bounce <= uMaxBounces; ++bounce) {',
        'int bounce=0;for (; bounce <= uMaxBounces; ++bounce) {')
    # Record actual native unresolved/cutoff conditions; no additional query.
    source=_once(source,'int   hit_id = nearest_hit(ro, rd, surf_t);',
        'int   hit_id = nearest_hit(ro, rd, surf_t);\n            if(hit_id< -1 || currentMedium==-2){scs_termination=6u;if(!scs_has_recipient)for(int j=0;j<8;++j)scs_volume_known[j]=0u;}')
    source=_once(source,'if (hit_id < 0 || tp_max < 0.003) {',
        'if (hit_id < 0 || tp_max < 0.003) {if(hit_id==-1 && currentMedium!=-2)scs_termination=0u;else if(hit_id>=0)scs_termination=tp_max>0.0?4u:0u;')
    source=_once(source,'if (tp_after < 0.005) {',
        'if (tp_after < 0.005) {if(tp_after>0.0)scs_termination=7u;')
    source=_once(source,'if(!sci_ok) break;','if(!sci_ok){scs_termination=2u;break;}')
    marker='        for (int i = 0; i < nl; i++) {\n            float measured = lum[i];'
    call='''        if(bounce>uMaxBounces && scs_termination==0u)scs_termination=8u;
        uint scs_sample_row=scs_write(linear_id,s,sci_first_hit,sci_first_kind,sci_first_owner,sci_first_stage,nl,
            sensor_px,sci_first_position,scs_first_direction,scs_first_medium,scs_camera_origin,
            scs_camera_direction,scs_first_prefix,lum,scs_first_regions,sci_first_metadata);
'''
    source=_once(source,marker,call+marker)
    source=_once(source,'            float measured = lum[i];',
        '            float measured = lum[i];\n            float scs_measured_volume=scs_volume[i],scs_measured_surface=scs_surface[i];')
    source=_once(source,'                measured = 0.0;',
        '                measured = 0.0;scs_measured_volume=0.0;scs_measured_surface=0.0;')
    source=_once(source,'                    measured += lum[band] * exp(-0.5 * oct * oct / (width * width));',
        '                    float scs_original_filter=exp(-0.5 * oct * oct / (width * width));\n                    measured += lum[band] * scs_original_filter;\n                    scs_measured_volume+=scs_volume[band]*scs_original_filter;\n                    scs_measured_surface+=scs_surface[band]*scs_original_filter;')
    source=_once(source,'                measured *= uSensorLayerGains[i];',
        '                measured *= uSensorLayerGains[i];\n                scs_measured_volume*=uSensorLayerGains[i];scs_measured_surface*=uSensorLayerGains[i];')

    source=_once(source,'            layer_acc[i] += max(measured, 0.0);',
        '            if(scs_sample_row!=0xffffffffu){scs_words[scs_sample_row+120u+uint(i)]=floatBitsToUint(max(measured,0.0));scs_words[scs_sample_row+72u+uint(i)]=floatBitsToUint(scs_measured_volume);scs_words[scs_sample_row+80u+uint(i)]=floatBitsToUint(scs_measured_surface);}\n            layer_acc[i] += max(measured, 0.0);')
    # Completed-only mode never allocates or emits the optional raw histories.
    # Retain the original copy hooks for their local metadata, but eliminate
    # two inactive SSBO blocks instead of spending native per-stage bindings.
    # Every aliased raw access is guarded by the compile-time sentinel path.
    source=_once(source,'layout(std430,binding=22) buffer StudioCameraIntervals { uint sci_words[]; };','')
    source=_once(source,'layout(std430,binding=24) buffer StudioCameraResponses { uint scr_words[]; };','')
    source=source.replace('sci_words','scs_words').replace('scr_words','scs_words')
    return source
