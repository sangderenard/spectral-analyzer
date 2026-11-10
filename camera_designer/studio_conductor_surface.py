"""Shared candidate conductor surface law; not bound to Studio transport yet.

Uses canonical per-wavelength n/k when available, or explicitly authored
reflectance with Schlick when n/k is not supplied. Roughness is a parameter,
never recovered from compatibility diffusion or a remission buffer slot.
"""
import numpy as np


def conductor_fresnel(n, k, cosine):
    n,k,c=np.broadcast_arrays(np.asarray(n,float),np.asarray(k,float),np.asarray(cosine,float))
    if np.any(~np.isfinite(n+k+c)) or np.any(n<0) or np.any(k<0):
        raise ValueError('conductor n/k must be finite and nonnegative')
    c=np.clip(c,0,1);c2=c*c;s2=1-c2;t=n*n-k*k-s2
    ab=np.sqrt(t*t+4*n*n*k*k);a=np.sqrt(np.maximum((ab+t)*.5,0))
    rs=(ab+c2-2*a*c)/np.maximum(ab+c2+2*a*c,1e-30)
    rp=rs*(c2*ab+s2*s2-2*a*c*s2)/np.maximum(c2*ab+s2*s2+2*a*c*s2,1e-30)
    return np.where(c==0,1,np.clip((rs+rp)*.5,0,1))


def sample_ggx(wi, roughness, samples):
    """Local normal +Z, wi points away from surface; null samples weight zero.

    NDF sampling (not VNDF): the f*cos/pdf geometric weight is returned,
    excluding Fresnel. Supports a delta at exactly zero roughness. Invalid
    outgoing hemisphere samples are absorption/null events, not resampled.
    """
    wi=np.asarray(wi,float);u=np.asarray(samples,float).reshape(-1,2)
    if wi.shape!=(3,) or not np.isfinite(wi).all() or abs(np.linalg.norm(wi)-1)>1e-6 or wi[2]<=0:
        raise ValueError('unit upper-hemisphere incident direction required')
    if not np.isfinite(roughness) or not 0<=roughness<=1 or np.any((u<0)|(u>1)):
        raise ValueError('invalid roughness or uniform samples')
    if roughness==0:
        return np.tile([-wi[0],-wi[1],wi[2]],(len(u),1)),np.ones(len(u)),np.full(len(u),wi[2])
    u=u*(16777215.0/16777216.0)  # production rand01 has an inclusive endpoint
    alpha=roughness*roughness;a2=alpha*alpha
    tan2=a2*u[:,0]/(1-u[:,0]);c=1/np.sqrt(1+tan2);s=np.sqrt(tan2)*c;phi=2*np.pi*u[:,1]
    h=np.column_stack([s*np.cos(phi),s*np.sin(phi),c]);vh=h@wi
    wo=2*vh[:,None]*h-wi;valid=(vh>0)&(wo[:,2]>0)
    def g1(x):return 2*x/(x+np.sqrt(a2+(1-a2)*x*x))
    weight=np.zeros(len(u));weight[valid]=g1(wi[2])*g1(wo[valid,2])*vh[valid]/(wi[2]*c[valid])
    return wo,weight,np.clip(vh,0,1)

CONDUCTOR_SURFACE_GLSL=r'''
float studio_conductor_fresnel(float eta,float k,float cosine){
    float c=clamp(cosine,0.0,1.0);if(c==0.0)return 1.0;
    float c2=c*c,s2=1.0-c2,t=eta*eta-k*k-s2;
    float ab=sqrt(t*t+4.0*eta*eta*k*k),a=sqrt(max(.5*(ab+t),0.0));
    float rs=(ab+c2-2.0*a*c)/max(ab+c2+2.0*a*c,1e-30);
    float rp=rs*(c2*ab+s2*s2-2.0*a*c*s2)/max(c2*ab+s2*s2+2.0*a*c*s2,1e-30);
    return clamp(.5*(rs+rp),0.0,1.0);
}
float studio_ggx_g1(float cosine,float alpha2){
    return 2.0*cosine/(cosine+sqrt(alpha2+(1.0-alpha2)*cosine*cosine));
}
// Inputs/outputs are in the local normal=+Z frame. Return zero for null events.
float studio_sample_ggx(vec3 wi,float roughness,vec2 u,out vec3 wo,out float vh){
    if(roughness==0.0){wo=vec3(-wi.xy,wi.z);vh=wi.z;return 1.0;}
    u*=16777215.0/16777216.0; // map inclusive 24-bit rand01 to half-open uniform
    float alpha=roughness*roughness,a2=alpha*alpha;
    float tan2=a2*u.x/(1.0-u.x),c=inversesqrt(1.0+tan2);
    float s=sqrt(tan2)*c,phi=6.283185307179586*u.y;
    vec3 h=vec3(s*cos(phi),s*sin(phi),c);vh=dot(wi,h);wo=2.0*vh*h-wi;
    if(vh<=0.0||wo.z<=0.0){vh=clamp(vh,0.0,1.0);return 0.0;}
    float weight=studio_ggx_g1(wi.z,a2)*studio_ggx_g1(wo.z,a2)*vh/(wi.z*c);
    vh=clamp(vh,0.0,1.0);return weight;
}
'''


def checked_native_pbr(data, geom, shade, material_count):
    """Validate existing native PBR ABI, never infer roughness from another field."""
    from mat_flags import MAT_FLAG_ROUGH_CONDUCTOR
    flags=np.asarray(geom,dtype=np.float32)[:,15].view(np.uint32)
    active=(flags & MAT_FLAG_ROUGH_CONDUCTOR)!=0
    if data is None:
        if np.any(active):raise ValueError('advanced conductor requires canonical native PBR buffer')
        return np.zeros((1,16),np.float32)
    pbr=np.asarray(data,dtype=np.float32)
    if pbr.ndim!=2 or pbr.shape[1]!=16 or not np.isfinite(pbr).all():
        raise ValueError('invalid canonical native PBR buffer')
    if np.any(active):
        ids=np.asarray(shade,dtype=np.float32)[active,14].view(np.uint32)
        if np.any(ids>=len(pbr)) or np.any(ids>=material_count):raise ValueError('conductor native material ID outside buffers')
        rows=pbr[ids]
        if np.any((rows[:,3]<0)|(rows[:,3]>1)) or np.any(rows[:,4]!=1) or np.any(rows[:,5]!=0) or np.any(rows[:,7]!=1):
            raise ValueError('advanced conductor requires opaque full-metal native PBR')
    return np.ascontiguousarray(pbr)
