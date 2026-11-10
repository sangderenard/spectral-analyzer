"""Artistic full-metal reflector projection using the existing Studio spectrum.

Normal reflectance is authored RGB interpreted in the existing fixed spectral
basis, not measured conductor n/k. Ideal specular directions remain unchanged.
"""
import numpy as np
from .spectral_sensor_basis import FIXED3_WAVELENGTHS_NM


def authored_reflector_bands(row,wavelengths_um):
    bands=[]
    for wl in wavelengths_um:
        weights=np.exp(-.5*(np.log2(np.asarray(FIXED3_WAVELENGTHS_NM)/(wl*1000.))/.12)**2)
        rho=float(np.clip(weights@np.asarray(row[:3])/weights.sum(),0,1))
        f=299792458./(wl*1e-6)
        bands.append(dict(center_hz=f,bandwidth_hz=f*.01,reflectance=rho,transmittance=0.,diffuse_frac=0.,emission=0.,reemission=0.,ior_real=float(row[6]),ior_imag=0.))
    return bands


from .studio_conductor_surface import CONDUCTOR_SURFACE_GLSL

STUDIO_REFLECTOR_GLSL=CONDUCTOR_SURFACE_GLSL+r'''
layout(std430,binding=STUDIO_NATIVE_PBR_BINDING) readonly buffer StudioNativePbr {float studio_native_pbr[];};
bool studio_rough_conductor(Tri t){return (floatBitsToUint(t.normal.w)&MAT_FLAG_ROUGH_CONDUCTOR)!=0u;}
float studio_conductor_sample(Tri t,vec3 incoming,vec3 normal,vec2 u,out vec3 outgoing,out float vh){
    int material=int(floatBitsToUint(t.emissive.z));
    float roughness=clamp(studio_native_pbr[material*16+3],0.0,1.0);
    vec3 tangent=normalize(cross(abs(normal.z)<.99?vec3(0,0,1):vec3(0,1,0),normal));
    vec3 bitangent=cross(normal,tangent),wi=-incoming;
    vec3 local=vec3(dot(wi,tangent),dot(wi,bitangent),dot(wi,normal)),wo;
    float weight=studio_sample_ggx(local,roughness,u,wo,vh);
    outgoing=normalize(tangent*wo.x+bitangent*wo.y+normal*wo.z);return weight;
}
bool studio_spectral_reflector(Tri t) {
    return (floatBitsToUint(t.normal.w)&(MAT_FLAG_SPECTRAL_REFLECTOR|MAT_FLAG_ROUGH_CONDUCTOR))!=0u;
}
float studio_reflector_rho(Tri t,float frequency,float cosine) {
    int base=int(floatBitsToUint(t.emissive.z))*32*12;
    int selected=0;float error=1e30;
    for(int band=0;band<32;++band){
        float f=mat_buf[base+band*12];if(f<=0.0)continue;
        float d=abs(log(max(frequency,1.0)/f));
        if(d<error){error=d;selected=band;}
    }
    float nk=mat_buf[base+selected*12+8];
    if((floatBitsToUint(t.normal.w)&MAT_FLAG_NATIVE_CONDUCTOR_NK)!=0u)
        return studio_conductor_fresnel(mat_buf[base+selected*12+7],nk,cosine);
    float F0=clamp(mat_buf[base+selected*12+2],0.0,1.0);
    return F0+(1.0-F0)*pow(1.0-clamp(cosine,0.0,1.0),5.0);
}
'''
