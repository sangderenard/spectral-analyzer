"""Explicit Studio interface media, using the native +normal/-normal convention.

Only meshes with an immutable authored boundary map opt in. Material indices
address the existing MatBuf. Air is -1, as in the native tracer. Untagged
legacy meshes/shaders retain their old path; this module has no side effects.
"""
from __future__ import annotations
import numpy as np
from mat_flags import MAX_SPECTRAL_BANDS

STUDIO_BOUNDARY_GLSL = r'''
// TriGeom.v0.w / e1.w encode (positive / negative MatBuf ID + 2).
// Zero means a legacy untagged face, one means air (-1). The scene marker
// e2.w=4097 on its first triangle enables origin classification only.
bool studio_media_enabled() { return geom_tris.length()>0 && geom_tris[0].e2.w==4097.0; }
bool studio_boundary(Tri t) { return t.v0.w>=1.0 && t.e1.w>=1.0 && t.mat_out.w<0.999; }
int nearest_boundary(vec3 ro, vec3 rd, out float best);
int studio_origin_medium(vec3 ro, vec3 rd) {
    if(!studio_media_enabled()) return -1;
    float t; int hit=nearest_boundary(ro,rd,t);
    if(hit<0) return -1;
    Tri b=load_tri(hit);
    return int((dot(rd,b.normal.xyz)<0.0 ? b.v0.w : b.e1.w))-2;
}
vec2 studio_medium_optics(int medium, float frequency) {
    if(medium<0) return vec2(1.0,0.0);
    int base=medium*__STUDIO_MAX_BANDS__*12;
    int chosen=0; float best=1e30;
    for(int band=0;band<__STUDIO_MAX_BANDS__;++band) {
        float f=mat_buf[base+band*12];
        if(f<=0.0) continue;
        float error=abs(log(max(frequency,1.0)/f));
        if(error<best) {best=error;chosen=band;}
    }
    int row=base+chosen*12;
    return vec2(max(mat_buf[row+7],1e-6),max(mat_buf[row+8],0.0));
}
float studio_medium_transmittance(int medium,float frequency,float distance) {
    if(medium<0) return 1.0;
    float k=studio_medium_optics(medium,frequency).y;
    return exp(-4.0*3.14159265358979323846*k*frequency*max(distance,0.0)/299792458.0);
}
// Exact unpolarized dielectric Fresnel. The oriented normal faces the
// incident ray. Reflection/TIR keep medium; only successful transmission
// changes it. eta^2 is returned for camera-radiance transport (not photons).
bool studio_boundary_step(Tri t,inout vec3 rd,float frequency,float xi,
                          inout int medium,out float camera_factor) {
    bool front=dot(rd,t.normal.xyz)<0.0;
    int from=int(front?t.v0.w:t.e1.w)-2;
    int to=int(front?t.e1.w:t.v0.w)-2;
    camera_factor=1.0;
    if(medium!=from) return false; // invalid/overlapping authored topology
    vec3 n=normalize(front?t.normal.xyz:-t.normal.xyz);
    float ni=studio_medium_optics(from,frequency).x;
    float nt=studio_medium_optics(to,frequency).x;
    if(ni==nt) {medium=to;return true;} // no optical interface at equal index
    float eta=ni/nt,ci=clamp(dot(-rd,n),0.0,1.0);
    float sin2=eta*eta*max(0.0,1.0-ci*ci);
    if(sin2>=1.0) {rd=normalize(reflect(rd,n));return true;}
    float ct=sqrt(max(0.0,1.0-sin2));
    float rs=(ni*ci-nt*ct)/max(ni*ci+nt*ct,1e-20);
    float rp=(nt*ci-ni*ct)/max(nt*ci+ni*ct,1e-20);
    float F=clamp(0.5*(rs*rs+rp*rp),0.0,1.0);
    if(xi<F) rd=normalize(reflect(rd,n));
    else {rd=normalize(refract(rd,n,eta));medium=to;camera_factor=eta*eta;}
    return true;
}
'''.replace('__STUDIO_MAX_BANDS__',str(MAX_SPECTRAL_BANDS))


def validate_boundary_ranges(value, triangle_count, material_count):
    """Validate the stored, digest-covered ownership; never infer it from IDs."""
    if not isinstance(value,dict) or set(value)!={'version','ranges'} or value['version']!=1:
        raise ValueError('expected version 1 explicit boundary media')
    ranges=value['ranges']
    if not isinstance(ranges,list) or not ranges or len(ranges)>1024:
        raise ValueError('invalid explicit boundary range count')
    pairs=np.full((triangle_count,2),-2,np.int32)
    for item in ranges:
        if not isinstance(item,dict) or set(item)!={'start','count','positive','negative'}:
            raise ValueError('invalid explicit boundary range')
        if any(not isinstance(x,int) or isinstance(x,bool) for x in item.values()):
            raise ValueError('boundary indices must be integers')
        start,count,positive,negative=(item[k] for k in ('start','count','positive','negative'))
        if start<0 or count<1 or start+count>triangle_count or any(x< -1 or x>=material_count for x in (positive,negative)) :
            raise ValueError('boundary range/index outside saved geometry')
        if np.any(pairs[start:start+count]!=-2):raise ValueError('overlapping boundary membership')
        pairs[start:start+count]=positive,negative
    return pairs


def install_boundary_pairs(geom,pairs,valid,mapped_materials):
    """Retain explicit original membership through degenerate filtering."""
    retained=np.asarray(pairs,np.int32)[valid]
    for side,col in ((0,3),(1,7)):
        tagged=retained[:,side]>=-1
        encoded=np.zeros(len(retained),np.float32)
        encoded[tagged]=[1 if x==-1 else mapped_materials[int(x)]+2 for x in retained[tagged,side]]
        geom[:,col]=encoded
    return geom




def studio_medium_extinction(pbr_row, wavelength_um):
    """Existing authored RGB tint approximation, expressed as physical k.

    The canonical PBR-to-ray converter supplies its absorption coefficient.
    This is not a measured spectrum. k = sigma_a * lambda / (4*pi), so native
    MatBuf Beer propagation applies tint to the segment's actual medium.
    """
    from material_db import PBRBaseRecord, _pbr_to_ray_record
    from camera_designer.spectral_sensor_basis import FIXED3_WAVELENGTHS_NM
    pbr=np.asarray(pbr_row,np.float32)
    ray=_pbr_to_ray_record(PBRBaseRecord.from_buffer_copy(pbr.tobytes()))
    absorption=float(ray.refl_in[2])
    centers=np.asarray(FIXED3_WAVELENGTHS_NM,np.float64)
    weights=np.exp(-.5*(np.log2(centers/(float(wavelength_um)*1000.))/.12)**2)
    chroma=float(np.clip(weights @ pbr[:3]/weights.sum(),0.,1.)) if weights.sum()>1e-6 else 1.
    return absorption*(1.-chroma)*(float(wavelength_um)*1e-6)/(4.*np.pi)

