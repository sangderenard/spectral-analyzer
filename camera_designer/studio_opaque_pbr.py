"""Explicit shared opaque dielectric BSDF for original Studio packet/camera paths.

A convex Lambert/GGX mixture, using authored native PBR values and the existing
compatibility diffusion lobe weight. No new emitter, geometry or medium law.
"""
VERSION='studio-pbr-mixture-v1'

def validate_surface_transport(config):
    value=config.get('surface_transport')
    if value is None:return None
    if value!=VERSION:raise ValueError('unsupported surface_transport')
    if config.get('transport')!='fixed:3' or 'fixtures' not in config:raise ValueError('corrected opaque PBR transport requires fixed:3 Studio fixtures')
    return value


GLSL=r'''
const float STUDIO_PBR_PI=3.14159265358979323846;
bool studio_pbr_opaque(Tri t) {
    uint flags=floatBitsToUint(t.normal.w);
    int m=int(floatBitsToUint(t.emissive.z))*16;
    return t.mat_out.w>=.999 && studio_native_pbr[m+7]>=.999
        && studio_native_pbr[m+4]==0.0 && studio_native_pbr[m+5]==0.0
        && (flags & (MAT_FLAG_EMISSIVE|MAT_FLAG_REACTIVE|4u))==0u;
}
// Same existing fixed3 Gaussian RGB spectral approximation, with defined
// signed squaring: GLSL pow(negative,2) is undefined on real drivers.
float studio_pbr_rho(Tri t,float frequency){
    vec3 centres=vec3(299792458.0/650e-9,299792458.0/550e-9,299792458.0/450e-9);
    vec3 x=log(vec3(max(frequency,1.0))/centres)/(log(2.0)*.12);
    vec3 w=exp(-.5*x*x);float den=w.x+w.y+w.z;
    return den>1e-6?clamp(dot(w,t.albedo.rgb)/den,0.0,1.0):1.0;
}
float studio_pbr_diffusion(Tri t){return clamp(t.mat_out.y,0.0,1.0);}
float studio_pbr_roughness(Tri t){return clamp(studio_native_pbr[int(floatBitsToUint(t.emissive.z))*16+3],0.0,1.0);}
float studio_pbr_fresnel(Tri t,float cosine){
    float eta=max(studio_native_pbr[int(floatBitsToUint(t.emissive.z))*16+6],1.0);
    float ci=clamp(cosine,0.0,1.0),ct=sqrt(max(0.0,1.0-(1.0-ci*ci)/(eta*eta)));
    if(eta==1.0)return 0.0;
    float rs=(ci-eta*ct)/max(ci+eta*ct,1e-30),rp=(eta*ci-ct)/max(eta*ci+ct,1e-30);
    return .5*(rs*rs+rp*rp);
}
float studio_pbr_eval(Tri t,vec3 wi,vec3 wo,vec3 n,float frequency,out float pdf){
    float ni=dot(n,wi),no=dot(n,wo);pdf=0.0;
    if(ni<=0.0||no<=0.0)return 0.0;
    float d=studio_pbr_diffusion(t),rough=studio_pbr_roughness(t);
    float value=d*studio_pbr_rho(t,frequency)/STUDIO_PBR_PI;
    pdf=d*no/STUDIO_PBR_PI;
    vec3 hs=wi+wo;float h2=dot(hs,hs);
    // Exactly smooth specular is a delta measure, not a finite angular PDF.
    if(d<1.0&&rough>0.0&&h2>1e-20){
        vec3 h=hs*inversesqrt(h2);float nh=max(dot(n,h),0.0),vh=max(dot(wi,h),0.0);
        if(nh>0.0&&vh>0.0){
            float alpha=rough*rough,a2=alpha*alpha;
            float den=(1.0-nh*nh)+a2*nh*nh;
            float D=a2/(STUDIO_PBR_PI*den*den);
            float G=studio_ggx_g1(ni,a2)*studio_ggx_g1(no,a2);
            value+=(1.0-d)*studio_pbr_fresnel(t,vh)*D*G/(4.0*ni*no);
            pdf+=(1.0-d)*D*nh/(4.0*vh);
        }
    }
    return value;
}
bool studio_pbr_sample(Tri t,vec3 wi,vec3 n,vec3 u,out vec3 wo,out float pdf,out bool delta){
    float d=studio_pbr_diffusion(t),rough=studio_pbr_roughness(t);delta=false;pdf=0.0;
    if(min(u.x,16777215.0/16777216.0)<d){wo=cosine_dir(u.y,u.z,n);}
    else{
        vec3 tangent=normalize(cross(abs(n.z)<.99?vec3(0,0,1):vec3(0,1,0),n));
        vec3 bitangent=cross(n,tangent),local=vec3(dot(wi,tangent),dot(wi,bitangent),dot(wi,n)),outlocal;
        float vh;float accepted=studio_sample_ggx(local,rough,u.yz,outlocal,vh);
        wo=normalize(tangent*outlocal.x+bitangent*outlocal.y+n*outlocal.z);
        if(accepted<=0.0)return false;
        if(rough==0.0){delta=true;pdf=1.0-d;return pdf>0.0;}
    }
    float unused=studio_pbr_eval(t,wi,wo,n,studio_band_frequency(0),pdf);
    return pdf>0.0;
}
float studio_pbr_weight(Tri t,vec3 wi,vec3 wo,vec3 n,float frequency,float pdf,bool delta){
    if(delta)return studio_pbr_fresnel(t,max(dot(n,wi),0.0));
    float evaluated_pdf;float f=studio_pbr_eval(t,wi,wo,n,frequency,evaluated_pdf);
    return pdf>0.0 ? f*max(dot(n,wo),0.0)/pdf : 0.0;
}
'''

FORWARD=r'''
        if(studio_pbr_opaque(tri)) {
            vec3 incoming=-rd,outgoing;float pdf;bool delta;
            if(!studio_pbr_sample(tri,incoming,n,vec3(rand01(rng),rand01(rng),rand01(rng)),outgoing,pdf,delta))break;
            packet.z*=studio_pbr_weight(tri,incoming,outgoing,n,packet.x,pdf,delta);
            if(!(packet.z>0.0)||isnan(packet.z)||isinf(packet.z))break;
            rd=outgoing;ro=hp+rd*(studio_media_enabled()?1e-6:1e-5);
            path_len+=best;packet.y+=best*packet.x*0.000021;
            continue;
        }
'''

CAMERA=r'''
            if(studio_pbr_opaque(tri)) {
                // Disjoint estimators: NEE owns direct continuous-lobe emitter
                // connections; sampled BSDF owns indirect paths and delta hits.
                float source_frequency;vec3 direct_bands,light_direction;
                float direct=sample_direct_light(hp,n,rng,source_frequency,direct_bands,light_direction);
                if(uNEEAreaSources!=0){
                    for(int i=0;i<nl;++i){
                        float pdf;float f=studio_pbr_eval(tri,-rd,light_direction,n,studio_band_frequency(i),pdf);
                        lum[i]+=throughput[i]*direct_bands[i]*STUDIO_PBR_PI*f;
                    }
                }
                vec3 incoming=-rd,outgoing;float pdf;bool delta;
                if(!studio_pbr_sample(tri,incoming,n,vec3(rand01(rng),rand01(rng),rand01(rng)),outgoing,pdf,delta))break;
                float largest=0.0;
                for(int i=0;i<nl;++i){
                    throughput[i]*=studio_pbr_weight(tri,incoming,outgoing,n,studio_band_frequency(i),pdf,delta);
                    largest=max(largest,throughput[i]);
                }
                if(!(largest>0.0)||isinf(largest)||isnan(largest))break;
                emission_via_bsdf=delta||uNEEAreaSources==0;
                rd=outgoing;ro=hp+rd*(studio_media_enabled()?1e-6:1e-5);
                continue;
            }
'''

def _once(source,old,new):
    if source.count(old)!=1:raise ValueError('opaque PBR shader seam changed: '+old[:80])
    return source.replace(old,new,1)

def with_opaque_pbr(source,*,camera=False):
    """Compose only on explicit corrected Studio jobs; old sources stay intact."""
    if 'bool studio_pbr_opaque(' in source:raise ValueError('opaque PBR helper already installed')
    source=_once(source,'void main() {',GLSL+'\nvoid main() {')
    if camera:
        old='out float src_freq_hz, out vec3 direct_bands) {'
        source=_once(source,old,'out float src_freq_hz, out vec3 direct_bands, out vec3 light_direction) {')
        source=_once(source,'    src_freq_hz = 0.0;','    light_direction=vec3(0.0);\n    src_freq_hz = 0.0;')
        source=_once(source,'    vec3  wi     = to_src / dist;','    vec3  wi     = to_src / dist;\n    light_direction=wi;')
        source=_once(source,'                vec3 direct_bands = vec3(0.0);','                vec3 direct_bands = vec3(0.0),direct_direction;')
        source=_once(source,'sample_direct_light(hp, n, rng, direct_freq, direct_bands)','sample_direct_light(hp, n, rng, direct_freq, direct_bands,direct_direction)')
        source=_once(source,'            vec3  surf_albedo = tri.albedo.rgb;','            vec3  surf_albedo = tri.albedo.rgb;\n'+CAMERA)
    else:
        anchor='        if(spectral_reflector)reflectivity=studio_reflector_rho(tri,packet.x,cos_i);'
        if source.count(anchor)!=2:raise ValueError('forward/reactive opaque material seam changed')
        source=source.replace(anchor,anchor+'\n'+FORWARD,1)
        location=source.index('void reactive_main(uint gid, inout uint rng) {')
        source=source[:location]+source[location:].replace(anchor,anchor+'\n'+FORWARD.replace('path_len+=best;',''),1)
        # A finite bounce budget is the work bound. No positive weighted packet
        # is discarded by an absolute energy threshold that changes with Nroots.
        source=source.replace('packet.z < 0.002','!(packet.z>0.0)')
        source=source.replace('react_pkt.z > 0.002','react_pkt.z > 0.0')
    return source
