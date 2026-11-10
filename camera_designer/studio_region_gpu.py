"""Opt-in multi-region ownership for the existing forward/camera shader pair.

The legacy variants remain unchanged. Region records are [mode, material,0,0]
(host=0, rod=1, injected dye=2, air=3); record zero is the host. Tagged geometry
v0.w is -(4*region_id+mode+1). This module resolves membership; existing native n/k
and exact real-index dielectric transport remain optical authorities.
"""
from __future__ import annotations
import numpy as np
import re
from .studio_boundary_media import STUDIO_BOUNDARY_GLSL
from .studio_rod_bundle_gpu import DIVIDER_GLSL, BUNDLE_GLSL, region_record_count

MAX_REGIONS=1024
REGION_GLSL=r'''
layout(std430,binding=15) readonly buffer StudioRegionBuf { vec4 studio_regions[]; };
uniform int uStudioRegionCount;
uint studio_active[32];
vec3 studio_region_sample_position=vec3(0);
bool studio_has_bundle();
int studio_bundle_owner(vec3 world);
int studio_region_filter=-1;
vec3 studio_region_hit_position=vec3(0.0);
// Coordinate-relative float32 heuristic, not a propagated intersection-error
// bound. Cancellation and grazing rays still require numerical validation.
// Features below this threshold are unresolved; no fixed micrometre skip.
float studio_region_epsilon(vec3 p) {
    return max(1e-12,2.384185791015625e-7*max(max(abs(p.x),abs(p.y)),abs(p.z)));
}
int studio_region_id(float encoded) { return encoded<=-1.0 ? (int(-encoded)-1)/4 : -1; }
bool studio_matches_region(float tag,float negative,int id) {
    if(id<0)return true;
    if(int(studio_regions[id].x)==4)return tag==1.0 && negative==studio_regions[id].y+2.0;
    return studio_region_id(tag)==id;
}
bool studio_is_active(int id) { return (studio_active[id/32] & (1u<<uint(id%32)))!=0u; }
void studio_set_active(int id,bool value) {
    uint bit=1u<<uint(id%32);
    if(value)studio_active[id/32]|=bit;else studio_active[id/32]&=~bit;
}
int studio_resolve_regions() {
    if(uStudioRegionCount==0)return -1;
    int standalone=-1;
    for(int i=1;i<uStudioRegionCount;++i)if(studio_is_active(i) && int(studio_regions[i].x)==4) {
        if(standalone>=0)return -2;
        standalone=int(studio_regions[i].y);
    }
    if(standalone>=0)return studio_is_active(0)?-2:standalone;
    if(!studio_is_active(0))return -1;
    int rod=-1;bool dye=false;
    for(int i=1;i<uStudioRegionCount;++i) if(studio_is_active(i)) {
        int mode=int(studio_regions[i].x),material=int(studio_regions[i].y);
        if(mode==3)return -1;
        if(mode==1) {if(rod>=0 && rod!=material && !studio_has_bundle())return -2;rod=material;}
        if(mode==2)dye=true;
    }
    if(rod>=0 && dye)return -2;
    if(rod>=0 && studio_has_bundle()) {
        int owner=studio_bundle_owner(studio_region_sample_position);
        if(owner>=0)return int(studio_regions[owner].y);
    }
    return rod>=0?rod:int(studio_regions[0].y);
}
'''

REGION_WRAPPERS=r'''
int studio_classify_regions(vec3 ro,vec3 rd) {
    studio_region_sample_position=ro;
    for(int i=0;i<32;++i)studio_active[i]=0u;
    for(int i=0;i<uStudioRegionCount;++i) {
        studio_region_filter=i;
        float t;int hit=nearest_boundary(ro,rd,t);
        if(hit>=0) {
            Tri b=load_tri(hit);
            studio_set_active(i,dot(rd,b.normal.xyz)>0.0);
        }
    }
    studio_region_filter=-1;
    return studio_resolve_regions();
}
int studio_origin_medium(vec3 ro,vec3 rd) {
    return uStudioRegionCount==0?studio_legacy_origin_medium(ro,rd):studio_classify_regions(ro,rd);
}
float studio_medium_transmittance(int medium,float frequency,float distance) {
    if(uStudioRegionCount==0)return studio_legacy_medium_transmittance(medium,frequency,distance);
    if(medium==-2)return 0.0;
    if(medium<0)return 1.0;
    float k=studio_medium_optics(medium,frequency).y;
    for(int i=1;i<uStudioRegionCount;++i)
        if(studio_is_active(0) && studio_is_active(i) && int(studio_regions[i].x)==2)
            k+=studio_medium_optics(int(studio_regions[i].y),frequency).y;
    return exp(-4.0*3.14159265358979323846*k*frequency*max(distance,0.0)/299792458.0);
}
bool studio_boundary_step(Tri t,inout vec3 rd,float frequency,float xi,
                          inout int medium,out float camera_factor) {
    if(uStudioRegionCount==0)return studio_legacy_boundary_step(t,rd,frequency,xi,medium,camera_factor);
    camera_factor=1.0;
    int id=studio_region_id(t.v0.w);
    bool fixed_pair=t.v0.w==1.0 && t.e1.w>=2.0;
    if((!fixed_pair && (id<0 || id>=uStudioRegionCount)) || medium==-2)return false;
    bool entering=dot(rd,t.normal.xyz)<0.0;
    uint before[32];for(int i=0;i<32;++i)before[i]=studio_active[i];
    // Resolve all coincident crossings together from actual geometry. This
    // conservative reference is deliberately not a last-entered-material stack.
    int to=studio_classify_regions(studio_region_hit_position+rd*studio_region_epsilon(studio_region_hit_position),rd);
    if(to==-2) {for(int i=0;i<32;++i)studio_active[i]=before[i];return false;}
    // Resolve dynamic sides, then use the SAME existing Fresnel/Snell/TIR
    // kernel as the legacy explicit-boundary path. Do not fork optical laws.
    Tri resolved=t;
    resolved.v0.w=float((entering?medium:to)+2);
    resolved.e1.w=float((entering?to:medium)+2);
    int accepted=medium;
    bool ok=studio_legacy_boundary_step(resolved,rd,frequency,xi,accepted,camera_factor);
    if(!ok || accepted!=to)for(int i=0;i<32;++i)studio_active[i]=before[i];
    if(!ok)return false;
    medium=accepted;
    return true;
}
'''


def region_boundary_helpers():
    legacy=STUDIO_BOUNDARY_GLSL
    for name in ('origin_medium','medium_transmittance','boundary_step'):
        legacy=legacy.replace('studio_'+name,'studio_legacy_'+name)
    legacy=legacy.replace('t.v0.w>=1.0 && t.e1.w>=1.0', '(t.v0.w>=1.0 || t.v0.w<=-1.0) && t.e1.w>=1.0')
    legacy=legacy.replace('return geom_tris.length()>0 &&', 'return uStudioRegionCount>0 || geom_tris.length()>0 &&',1)
    # The final optional bundle row is an integer failure counter. Read-only
    # float records and diagnostic atomic writes share binding15 deliberately.
    source=REGION_GLSL+DIVIDER_GLSL+BUNDLE_GLSL+legacy+REGION_WRAPPERS
    source=source.replace('layout(std430,binding=15) readonly buffer StudioRegionBuf { vec4 studio_regions[]; };',
        'layout(std430,binding=15) buffer StudioRegionBuf { uvec4 studio_region_bits[]; };\nvec4 studio_region_record(int i){return uintBitsToFloat(studio_region_bits[i]);}')
    return re.sub(r'studio_regions\[([^\]]+)\]',r'studio_region_record(\1)',source)


def with_region_media(source):
    # Called on canonical source before exact-sphere adaptation. Legacy source
    # and shader variants remain byte-identical until explicitly selected.
    if source.count(STUDIO_BOUNDARY_GLSL)!=1:
        raise RuntimeError('Studio boundary helper seam changed')
    new=region_boundary_helpers()
    source=source.replace(STUDIO_BOUNDARY_GLSL,new)
    old='(!boundary_only || (geom_tris[ti].v0.w>=1.0 && shade_tris[ti].mat_out.w<0.999))'
    replacement='''(!boundary_only || ((geom_tris[ti].v0.w>=1.0 || geom_tris[ti].v0.w<=-1.0) && shade_tris[ti].mat_out.w<0.999)) && (studio_region_filter<0 || studio_matches_region(geom_tris[ti].v0.w,geom_tris[ti].e1.w,studio_region_filter))'''
    if source.count(old)!=1:raise RuntimeError('Studio BVH membership seam changed')
    source=source.replace(old,replacement)
    source=source.replace('if(!studio_boundary_step(tri,rd,', 'studio_region_hit_position=hp;\n            if(!studio_boundary_step(tri,rd,')
    # Invalid ownership never masquerades as exterior-air volume illumination.
    source=source.replace('currentMedium<0','currentMedium==-1')
    # Forward source declares hit_tri before the boundary-helper seam. GLSL
    # needs the shared precision helper prototype before that first call.
    source=source.replace('bool hit_tri(vec3 ro, vec3 rd, TriGeom t, out float hit_t) {',
        'float studio_region_epsilon(vec3 p);\nbool hit_tri(vec3 ro, vec3 rd, TriGeom t, out float hit_t) {',1)
    # Keep all region traversal/continuation thresholds on the same positional
    # precision budget. Legacy shader variants are assembled separately.
    source=source.replace('if (abs(a) < 1e-8) return false;',
        'if (abs(a) <= 2.384185791015625e-7 * length(t.e1.xyz) * length(t.e2.xyz) * length(rd)) return false;')
    source=source.replace('if (tt <= 1e-6) return false;',
        'if (tt <= studio_region_epsilon(ro)/max(length(rd),1e-30)) return false;')
    source=source.replace('ro=hp+rd*1e-6;', 'ro=hp+rd*studio_region_epsilon(hp);')
    source=source.replace('studio_media_enabled()?1e-6:1e-5',
        'uStudioRegionCount>0?studio_region_epsilon(hp):(studio_media_enabled()?1e-6:1e-5)')
    return source


def with_region_spheres(source):
    """Apply AFTER existing sphere adapter to retain exact region normals."""
    # Sphere helpers precede region helpers in generated source: declare the
    # same shared budget before its first use instead of duplicating constants.
    source=source.replace('bool studio_sphere_hit(', 'float studio_region_epsilon(vec3 p);\nbool studio_sphere_hit(',1)
    source=source.replace('1e-6*ray_length/sphere.w',
                          'studio_region_epsilon(ro)*ray_length/sphere.w')
    source=source.replace('Tri b=load_tri(hit);','Tri b=studio_hit_tri(hit,ro+rd*t);')
    old='if(boundary_only && !(geom_tris[ti].v0.w>=1.0 && shade_tris[ti].mat_out.w<0.999)) continue;'
    new='''if(boundary_only && !((geom_tris[ti].v0.w>=1.0 || geom_tris[ti].v0.w<=-1.0) && shade_tris[ti].mat_out.w<0.999)) continue;
        if(studio_region_filter>=0 && !studio_matches_region(geom_tris[ti].v0.w,geom_tris[ti].e1.w,studio_region_filter))continue;'''
    if source.count(old)!=1:raise RuntimeError('Studio sphere membership seam changed')
    source=source.replace(old,new)
    old_hit='int nearest_hit(vec3 ro,vec3 rd,out float best) {return nearest_filtered(ro,rd,best,false);}'
    new_hit='''int nearest_hit(vec3 ro,vec3 rd,out float best) {
        studio_partition_hit=false;
        int original=nearest_filtered(ro,rd,best,false);
        int cut_hit=studio_partition_nearest(ro,rd,best);
        if(cut_hit==-2){atomicAdd(studio_region_bits[uStudioRegionCount+4].x,1u);return -2;} // fail closed, never skip unresolved cut
        if(cut_hit>=0){studio_partition_hit=true;return cut_hit;}
        return original;
    }'''
    if source.count(old_hit)!=1:raise RuntimeError('Studio partition nearest-hit seam changed')
    source=source.replace(old_hit,new_hit)
    source=source.replace('Tri t=load_tri(ti);','''Tri t=load_tri(ti);
    if(studio_partition_hit && studio_region_filter<0 && ti==studio_partition_owner_triangle){
        t.normal.xyz=studio_partition_normal;return t;
    }''',1)
    return source


def validate_region_records(records,geometry,shade,material_count):
    records=np.asarray(records,np.float32)
    original=records
    bundled=records.ndim==2 and records.shape[1]==4 and len(records)>0 and records[0,2]==1
    if bundled:
        count=region_record_count(records)
        if not 1<=count<=MAX_REGIONS or len(records)!=count+5 or not np.isfinite(records).all():raise ValueError('invalid bundle record length')
        indices=records[1:count,2];selected=indices>0
        if np.any(records[1:count,3]!=0) or np.any(indices<0) or np.any(indices!=np.floor(indices)) or sorted(indices[selected])!=list(range(1,int(selected.sum())+1)) or not 2<=selected.sum()<=32:
            raise ValueError('invalid bundle rod order')
        if np.any(records[1:count,0][selected]!=1) or records[count+3,0]<=0:raise ValueError('invalid bundle rod/length')
        if abs(np.linalg.det(records[count:count+3,:3]))<1e-20:raise ValueError('invalid bundle affine frame')
        host=records[count+4,1]
        if host!=int(host) or not 0<=host<len(geometry) or np.asarray(geometry)[int(host),3]!=-1:raise ValueError('invalid host representative')
        records=records[:count].copy();records[:,2:]=0
    if records.ndim!=2 or records.shape[1]!=4 or not 1<=len(records)<=MAX_REGIONS or not np.isfinite(records).all():
        raise ValueError('invalid region GPU records')
    if np.any(records[:,2:]!=0) or np.any(records[:,:2]!=np.floor(records[:,:2])):
        raise ValueError('region records must use exact IDs and zero reserved fields')
    if records[0,0]!=0 or np.any(~np.isin(records[1:,0],[1,2,3,4])):
        raise ValueError('region zero must be the host')
    for mode,material,_,_ in records:
        if (mode==3 and material!=-1) or (mode!=3 and not 0<=material<material_count):
            raise ValueError('invalid region material index')
    tags=np.asarray(geometry)[:,3]
    selected=tags<=-1
    encoded=-tags[selected]-1
    ids=np.floor(encoded/4)
    if np.any(encoded!=np.floor(encoded)) or np.any(ids>=len(records)) or set(ids.astype(int))!=set(np.flatnonzero(records[:,0]!=4)):
        raise ValueError('region triangle identities do not cover GPU records')
    index=ids.astype(int)
    if np.any(records[index,0]!=(encoded%4)) or np.any(records[index,1]!=(np.asarray(geometry)[selected,7]-2)):
        raise ValueError('region metadata disagrees with tagged material ownership')
    fixed=(tags>=1)
    if np.any(fixed):
        pairs=np.asarray(geometry)[fixed][:,[3,7]]
        if np.any(pairs[:,0]!=1) or np.any(pairs[:,1]<2) or np.any(pairs!=np.floor(pairs)):
            raise ValueError('standalone region boundaries require outward air/material pairs')
        mats=pairs[:,1]-2
        if set(mats)!=set(records[records[:,0]==4,1]) or len(set(records[records[:,0]==4,1]))!=sum(records[:,0]==4):
            raise ValueError('standalone material identities do not cover GPU records')
        if np.any(np.asarray(shade)[fixed,7]!=0):
            raise ValueError('standalone boundaries require opacity zero')
    elif np.any(records[:,0]==4):raise ValueError('standalone records lack geometry')
    if np.any(np.asarray(shade)[selected,7]!=0):
        raise ValueError('region boundaries require opacity zero')
    return np.ascontiguousarray(original if bundled else records)


REGION_MODES={'host':0,'glass_rod':1,'injection_dye':2,'air_bubble':3}


def validate_region_ranges(value,triangle_count,material_count):
    if not isinstance(value,dict) or set(value) not in ({'version','regions'},{'version','regions','axial_bundle'}) or type(value['version']) is not int or value['version']!=1:
        raise ValueError('expected version 1 region media')
    regions=value['regions']
    if not isinstance(regions,list) or not 1<=len(regions)<=MAX_REGIONS:
        raise ValueError('invalid region count')
    owners=np.full(triangle_count,-1,np.int32)
    for ri,r in enumerate(regions):
        if not isinstance(r,dict) or set(r)!={'start','count','mode','material'} or r['mode'] not in REGION_MODES:
            raise ValueError('invalid region descriptor')
        start,count,mat=r['start'],r['count'],r['material']
        if any(type(x) is not int for x in (start,count,mat)) or start<0 or count<1 or start+count>triangle_count:
            raise ValueError('region membership outside saved mesh')
        if (ri==0)!=(r['mode']=='host'):
            raise ValueError('region zero must be the only host')
        if (r['mode']=='air_bubble' and mat!=-1) or (r['mode']!='air_bubble' and not 0<=mat<material_count):
            raise ValueError('region material outside saved table')
        if np.any(owners[start:start+count]>=0):raise ValueError('region triangle ranges overlap')
        owners[start:start+count]=ri
    if np.any(owners<0):raise ValueError('region map must cover the saved mesh')
    host=regions[0]['material']
    pairs=np.array([[-1,host] if ri==0 else [host,regions[ri]['material']] for ri in owners],np.int32)
    return regions,owners,pairs


def tag_region_geometry(geom,regions,owners,valid,mapped_materials):
    for row,ri in enumerate(owners[valid]):
        region=regions[int(ri)];mode=REGION_MODES[region['mode']]
        geom[row,3]=-(4*int(ri)+mode+1)
        mat=region['material']
        geom[row,7]=1 if mat<0 else mapped_materials[mat]+2
    return geom


def region_records_from_geometry(geometry):
    geom=np.asarray(geometry);selected=geom[:,3]<=-1
    if not np.any(selected):return None
    codes=-geom[selected,3]-1
    if np.any(codes!=np.floor(codes)):raise ValueError('invalid region geometry tag')
    ids=(codes//4).astype(int);modes=(codes%4).astype(int);materials=geom[selected,7]-2
    count=int(ids.max())+1
    if count>MAX_REGIONS or set(ids)!=set(range(count)):raise ValueError('incomplete region identities')
    records=np.zeros((count,4),np.float32)
    for ri in range(count):
        rows=ids==ri
        if len(set(modes[rows]))!=1 or len(set(materials[rows]))!=1:
            raise ValueError('region spans inconsistent material records')
        records[ri,:2]=modes[rows][0],materials[rows][0]
    fixed=geom[:,3]>=1
    if np.any(fixed):
        pairs=geom[fixed][:,[3,7]]
        if np.any(pairs[:,0]!=1) or np.any(pairs[:,1]<2) or np.any(pairs!=np.floor(pairs)):
            raise ValueError('standalone region boundaries require outward air/material pairs')
        materials=sorted(set(pairs[:,1]-2))
        if count+len(materials)>MAX_REGIONS:raise ValueError('too many combined optical regions')
        extra=np.zeros((len(materials),4),np.float32);extra[:,0]=4;extra[:,1]=materials
        records=np.concatenate((records,extra))
    return records


def refreshed_region_records(geometry, previous=None):
    """Camera-only rebuild keeps bundle frame, never stale triangle indices."""
    records=region_records_from_geometry(geometry)
    if records is not None and previous is not None and previous[0,2]==1:
        count=len(records)
        if count!=int(previous[0,3]):raise ValueError('bundle region topology changed; rebuild worker')
        records[:,2:]=previous[:count,2:]
        metadata=previous[count:].copy()
        hosts=np.flatnonzero(np.asarray(geometry)[:,3]==-1)
        if not len(hosts):raise ValueError('bundle host missing after geometry update')
        metadata[4]=[0,int(hosts[0]),0,0]
        records=np.concatenate((records,metadata))
    return records
