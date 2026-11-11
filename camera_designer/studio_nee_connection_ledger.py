"""Observe camera NEE candidates and actual visibility tests, never emit rays."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .studio_ray_arrival_ledger import ArrivalLedger, _once, _block

BINDING = 22
REASONS = {1:'surface_backface',2:'emitter_backface',3:'occluded',4:'visible_area_connection',5:'visible_point_connection'}

_GLSL = r'''
layout(std430,binding=22) buffer StudioNeeLedger { uint snl_words[]; };
uniform vec3 uNeeLedgerMin;
uniform vec3 uNeeLedgerMax;
uniform uint uNeeLedgerPopulation;
uint snl_ray=0u,snl_sample=0u;
int snl_bounce=0,snl_surface=-1,snl_slot=-1;
vec3 snl_beta=vec3(0.0);
void snl_u(int k,uint v){if(snl_slot>=0)snl_words[snl_slot+k]=v;}
void snl_f(int k,float v){snl_u(k,floatBitsToUint(v));}
void snl_v(int k,vec3 v){snl_f(k,v.x);snl_f(k+1,v.y);snl_f(k+2,v.z);}
void snl_begin(vec3 p,vec3 n,int si,SourceRec src,float dist,vec3 wi,float cosine){
    uint index=atomicAdd(snl_words[0],1u);snl_slot=-1;
    bool in_roi=all(greaterThanEqual(p,uNeeLedgerMin))&&all(lessThan(p,uNeeLedgerMax));
    if(in_roi)atomicAdd(snl_words[2],1u);
    if(index>=__CAP__u){atomicAdd(snl_words[1],1u);return;}
    snl_slot=256+int(index)*64;
    for(int i=0;i<64;++i)snl_words[snl_slot+i]=0u;
    snl_u(0,snl_ray);snl_u(1,uint(uSeed));snl_u(2,snl_sample);snl_u(3,uint(snl_bounce));
    snl_u(4,uint(snl_surface));snl_u(5,uint(si));snl_u(7,0xffffffffu);
    snl_v(8,p);snl_f(11,dist);snl_v(12,src.pos_weight.xyz);snl_f(15,cosine);
    snl_v(16,wi);snl_u(25,uint(uNEEAreaSources));snl_u(26,uint(uSourceCount));
    snl_f(27,src.packet.z);snl_v(28,src.dir_kind.xyz);snl_u(31,in_roi?1u:0u);
    snl_v(32,snl_beta);snl_u(35,uNeeLedgerPopulation);snl_v(36,n);snl_f(45,src.packet.x);
}
void snl_shadow(vec3 origin,int hit,float distance){
    snl_u(41,1u);snl_v(42,origin);snl_u(7,uint(hit));
    if(hit>=0){Tri t=load_tri(hit);snl_f(20,distance);snl_u(39,floatBitsToUint(t.emissive.z));snl_u(40,floatBitsToUint(t.normal.w));}
}
void snl_end(int reason,vec3 bands,float weight){
    atomicAdd(snl_words[16+reason],1u);snl_u(6,uint(reason));snl_v(21,bands);snl_f(24,weight);
}
'''


def connection_shader(source,capacity=131072):
    """Compose after camera_shader; fail closed if NEE geometry changes."""
    if type(capacity) is not int or not 1<=capacity<=1048576:raise ValueError('invalid NEE capacity')
    if 'snl_begin(' in source:raise ValueError('NEE ledger already installed')
    signature='float sample_direct_light(vec3 p, vec3 n, inout uint rng, out float src_freq_hz, out vec3 direct_bands) {'
    start,end=_block(source,signature);body=source[start:end]
    body=_once(body,'if (cos_surf <= 0.0) return 0.0;',
        'snl_begin(p,n,si,src,dist,wi,cos_surf);\n    if (cos_surf <= 0.0) {snl_end(1,vec3(0.0),0.0);return 0.0;}')
    body=_once(body,'if(cos_light<=0.0)return 0.0;',
        'snl_f(19,cos_light);if(cos_light<=0.0){snl_end(2,vec3(0.0),0.0);return 0.0;}')
    body=_once(body,'int shadow_hit=nearest_hit(p+wi*1e-6,wi,shadow_t);',
        'int shadow_hit=nearest_hit(p+wi*1e-6,wi,shadow_t);snl_shadow(p+wi*1e-6,shadow_hit,shadow_hit>=0?shadow_t:0.0);')
    body=_once(body,'if(shadow_hit>=0 && shadow_t<dist-2e-6)return 0.0;',
        'if(shadow_hit>=0 && shadow_t<dist-2e-6){snl_end(3,vec3(0.0),0.0);return 0.0;}')
    body=_once(body,'return direct_bands.x;','snl_end(4,direct_bands,weight);return direct_bands.x;')
    body=_once(body,'int shadow_hit = nearest_hit(p + wi * 1e-4, wi, shadow_t);',
        'int shadow_hit = nearest_hit(p + wi * 1e-4, wi, shadow_t);snl_shadow(p+wi*1e-4,shadow_hit,shadow_hit>=0?shadow_t:0.0);')
    body=_once(body,'if (shadow_hit >= 0 && shadow_t < dist - 3e-4) return 0.0;',
        'if (shadow_hit >= 0 && shadow_t < dist - 3e-4) {snl_end(3,vec3(0.0),0.0);return 0.0;}')
    body=_once(body,'return energy * geometry;','snl_end(5,direct_bands,geometry);return energy * geometry;')
    result=source[:start]+_GLSL.replace('__CAP__',str(capacity))+body+source[end:]
    result=_once(result,'sal_sample=uint(s);sal_count(16);',
        'sal_sample=uint(s);sal_count(16);snl_ray=sal_ray;snl_sample=sal_sample;')
    result=_once(result,'sal_bounce=bounce;sal_begin(',
        'snl_bounce=bounce;snl_surface=hit_id;sal_bounce=bounce;sal_begin(')
    result=_once(result,'float direct = (uSourceCount > 0) ? sample_direct_light',
        'snl_beta=vec3(throughput[0],throughput[1],throughput[2]);\n                float direct = (uSourceCount > 0) ? sample_direct_light')
    return result


class NeeLedger(ArrivalLedger):
    def __init__(self,bounds,capacity=131072):
        from OpenGL import GL
        if int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS))<=BINDING:raise RuntimeError('NEE audit requires binding22')
        self.capacity=capacity
        super().__init__(bounds,roi_capacity=capacity,other_capacity=0)

    def bind(self,program,population):
        from OpenGL import GL
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,BINDING,self.buffer)
        old=int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM));GL.glUseProgram(program)
        GL.glUniform3fv(GL.glGetUniformLocation(program,'uNeeLedgerMin'),1,self.bounds[0])
        GL.glUniform3fv(GL.glGetUniformLocation(program,'uNeeLedgerMax'),1,self.bounds[1])
        GL.glUniform1ui(GL.glGetUniformLocation(program,'uNeeLedgerPopulation'),int(population));GL.glUseProgram(old)

    def read(self):
        from OpenGL import GL
        GL.glFinish();GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,self.buffer)
        head=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,256*4),'<u4').copy()
        count=min(int(head[0]),self.capacity)
        rows=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,256*4,count*64*4),'<u4').reshape(-1,64).copy() if count else np.zeros((0,64),'<u4')
        return head,rows

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True);head,rows=self.read()
        np.savez_compressed(path/'connections.npz',header=head,records=rows)
        report=dict(schema='studio.camera-nee-connections.v1',meaning='Camera NEE candidates; rejected candidates are not photon arrivals. Actual shadow test only where word41==1. Never forward energy.',
            candidates=int(head[0]),records=len(rows),dropped=int(head[1]),receiving_points_in_roi=int(head[2]),
            reasons={name:int(head[16+key]) for key,name in REASONS.items()},
            roi_reasons={name:int(np.count_nonzero((rows[:,6]==key)&(rows[:,31]!=0))) for key,name in REASONS.items()},
            blocker_material_counts={str(int(k)):int(v) for k,v in zip(*np.unique(rows[rows[:,6]==3,39],return_counts=True))},
            complete=not bool(head[1]),sha256=hashlib.sha256((path/'connections.npz').read_bytes()).hexdigest())
        (path/'receipt.json').write_text(json.dumps(report,indent=2));(path/'README.txt').write_text(FORMAT)
        return report


FORMAT='''Camera NEE candidate/connection diagnostic, not forward emitted energy.
connections.npz: header256uint32; recordsNx64uint32, all little-endian.
Float fields are raw IEEE754 bits; view asfloat32, do not numerically cast.
0 pixelID,1 RNGseed,2 sample,3 bounce,4 receiving triangle,5 selected source row,
6 rejection/acceptance reason,7 signed first shadow hit (-1 none),
8:11 f receiving point,11 f emitter distance,12:15 f source position,
15 f surface cosine,16:19 f candidate direction,19 f emitter cosine,
20 f shadow-hit distance (only if hit>=0),21:24 f returned direct bands,
24 f geometry weight,25 area-mode flag,26 source count,27 f area-per-root,
28:31 f source spectral emission,31 receiving point in studied ROI,
32:35 f camera adjoint throughput,35 source-population seed,36:39 f normal,
39 blocker material,40 blocker flags,41 shadow-test-executed flag,
42:45 f actual tested shadow origin,45 f source frequency,46:64 reserved.
Reasons:1 surface backface,2 emitter backface,3 occluded,4 visible area,
5 visible point. No finite endpoint is asserted for a miss. A backfacing
candidate did not execute any shadow ray. Header0 candidates,1 dropped,
2 receiving points in ROI,16+reason outcome counts. No forced lights/seeds.
'''
