"""Opt-in evidence at ORIGINAL scalar deposits and identical sensor queries."""
import numpy as np
import re
BINDING=23
_VISIT_BYTES={}
QUERY_GLSL=r'''
layout(std430,binding=23) readonly buffer StudioScalarVisits { uint studio_scalar_visits[]; };
uint studio_scalar_visit_flags(vec3 p,int band){
 vec3 uvw=(p-uBoxMin)/max(uBoxMax-uBoxMin,vec3(1e-6));
 ivec3 q=ivec3(floor(uvw*vec3(uDims)));
 if(band<0||band>=8||any(lessThan(q,ivec3(0)))||any(greaterThanEqual(q,uDims)))return 0u;
 int index=((q.z*uDims.y+q.y)*uDims.x+q.x)*2;
 uint bit=1u<<uint(band),e=studio_scalar_visits[index+1];
 return ((studio_scalar_visits[index]&bit)!=0u?1u:0u)|((e&bit)!=0u?2u:0u)|((e&(bit<<8))!=0u?4u:0u)|((e&(bit<<16))!=0u?8u:0u)|((e&(bit<<24))!=0u?16u:0u)|((studio_scalar_visits[index]&(bit<<8))!=0u?32u:0u)|((studio_scalar_visits[index]&(bit<<16))!=0u?64u:0u);
}
'''
DEPOSIT_GLSL=r'''
layout(std430,binding=23) coherent buffer StudioScalarVisits { uint studio_scalar_visits[]; };
void studio_scalar_add_result(ivec3 q,int band,uint prior,uint value){
 if(prior>0xffffffffu-value){
  int index=((q.z*uDims.y+q.y)*uDims.x+q.x)*2;
  atomicOr(studio_scalar_visits[index+1],1u<<uint(band+24));
 }
}
void studio_scalar_visit(ivec3 q,int band,float raw){
 if(band<0||band>=8)return;
 int index=((q.z*uDims.y+q.y)*uDims.x+q.x)*2;
 uint bit=1u<<uint(band);
 if(isnan(raw)||isinf(raw)||raw<0.0){atomicOr(studio_scalar_visits[index+1],bit<<16);return;}
 atomicOr(studio_scalar_visits[index],bit);
 if(raw>0.0&&raw<1.0)atomicOr(studio_scalar_visits[index+1],bit);
 if(raw>1e6)atomicOr(studio_scalar_visits[index+1],bit<<8);
}
'''


def with_scalar_visits(source):
    if 'buffer StudioScalarVisits' in source:raise ValueError('scalar visit observer already installed')
    anchor='void layer_splat(int i, ivec3 q, uint val) {'
    if source.count(anchor)!=1:raise ValueError('original scalar layer seam changed')
    source=source.replace(anchor,DEPOSIT_GLSL+anchor)
    # Observe the return of each ORIGINAL atomic addition. No replay/readback
    # can safely reconstruct this under concurrent writers.
    atomic=re.compile(r'imageAtomicAdd\(uLayer([0-7]), q, val\);')
    bands=atomic.findall(source)
    if bands!=list('01234567'):raise ValueError('original scalar atomic seams changed')
    source=atomic.sub(lambda m:'{ uint studio_prior=imageAtomicAdd(uLayer'+m[1]+', q, val); studio_scalar_add_result(q,'+m[1]+',studio_prior,val); }',source)
    seams=[('uint val = uint(clamp(energy * coherence * 65535.0, 0.0, 1e6));','studio_scalar_visit(q,studio_band_index(freq),energy*coherence*65535.0);'),
           ('uint val = uint(clamp(activation * 65535.0, 0.0, 1e6));','studio_scalar_visit(q,i,activation*65535.0);'),
           ('uint val = uint(clamp(per_layer * 65535.0, 0.0, 1e6));','for(int i=0;i<n;++i)studio_scalar_visit(q,i,per_layer*65535.0);')]
    for old,new in seams:
        if source.count(old)!=1:raise ValueError('original scalar deposit seam changed')
        source=source.replace(old,new+'\n        '+old,1)
    return source


def with_scalar_visit_queries(source):
    anchor='float sample_layer_energy(vec3 p, int i) {'
    if source.count(anchor)!=1 or 'buffer StudioScalarVisits' in source:raise ValueError('original scalar query seam changed')
    return source.replace(anchor,QUERY_GLSL+anchor)


def allocate(dims, *, extra_bytes=0):
    from OpenGL import GL
    if int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS))<=BINDING:raise ValueError('scalar availability requires24 SSBO bindings')
    if type(extra_bytes) is not int or extra_bytes<0 or extra_bytes%4:raise ValueError("invalid shared auxiliary capacity")
    size=int(np.prod(dims))*8+extra_bytes
    if size>int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BLOCK_SIZE)):raise MemoryError("shared auxiliary storage exceeds SSBO limit")
    handle=int(GL.glGenBuffers(1));GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,handle)
    try:
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,size,None,GL.GL_DYNAMIC_DRAW)
        if int(GL.glGetBufferParameteriv(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_BUFFER_SIZE))!=size:
            raise MemoryError('scalar availability allocation incomplete')
        # Whole-buffer initialization belongs only to its allocation owner.
        GL.glClearBufferData(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_R32UI,GL.GL_RED_INTEGER,GL.GL_UNSIGNED_INT,np.zeros(1,np.uint32))
        _VISIT_BYTES[handle]=int(np.prod(dims))*8
        bind(handle)
    except BaseException:
        _VISIT_BYTES.pop(handle,None)
        GL.glDeleteBuffers(1,[handle]);raise
    return handle


def clear(handle):
    """Reset only the registered visit prefix, never borrowed adjacent views."""
    from OpenGL import GL
    if int(handle) not in _VISIT_BYTES:raise ValueError('unknown scalar visit view')
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,handle)
    GL.glClearBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_R32UI,0,_VISIT_BYTES[int(handle)],
        GL.GL_RED_INTEGER,GL.GL_UNSIGNED_INT,np.zeros(1,np.uint32))


def release(handle):
    """Only native field owner releases this allocation; views never do."""
    from OpenGL import GL
    try:GL.glDeleteBuffers(1,[handle])
    finally:_VISIT_BYTES.pop(int(handle),None)


def bind(handle):
    from OpenGL import GL
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,BINDING,handle)


def scalar_add_overflow(prior,value):
    """CPU reference for the exact unsigned return-value observer predicate."""
    if type(prior) is not int or type(value) is not int or not (0<=prior<=0xffffffff and 0<=value<=0xffffffff):
        raise ValueError('expected uint32 addition operands')
    return prior>0xffffffff-value


def prepare_sampling(handle):
    """Synchronize BOTH original scalar texels and their opt-in SSBO evidence."""
    from OpenGL import GL
    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT |
                       GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                       GL.GL_TEXTURE_FETCH_BARRIER_BIT)
    bind(handle)
