"""Opt-in native scalar integration columns, not an alternate volume estimator."""
import numpy as np
BINDING=25
STRIDE_WORDS=4

GLSL=r'''
layout(std430,binding=25) coherent buffer StudioScalarColumns { uint studio_scalar_columns[]; };
uint studio_scalar_column_index(ivec3 q,int band){return uint(((q.z*uDims.y+q.y)*uDims.x+q.x)*STUDIO_COLUMN_BANDS+band)*4u;}
void studio_scalar_column_attempt(ivec3 q,int band,float raw){
 if(band<0||band>=STUDIO_COLUMN_BANDS)return;
 uint b=studio_scalar_column_index(q,band);
 if(isnan(raw)||isinf(raw)||raw<0.0){atomicOr(studio_scalar_columns[b+3u],4u);return;}
 uint old=atomicAdd(studio_scalar_columns[b],1u);
 if(old==0xffffffffu){atomicOr(studio_scalar_columns[b+3u],16u);STUDIO_ATTEMPT_COUNTER_LOSS}
 if(raw>0.0&&raw<1.0)atomicOr(studio_scalar_columns[b+3u],1u);
 if(raw>1e6)atomicOr(studio_scalar_columns[b+3u],2u);
}
void studio_scalar_column_add(ivec3 q,int band,uint prior,uint value){
 if(band<0||band>=STUDIO_COLUMN_BANDS)return;
 uint b=studio_scalar_column_index(q,band);
 uint old=atomicAdd(studio_scalar_columns[b+1u],1u);
 if(old==0xffffffffu){atomicOr(studio_scalar_columns[b+3u],32u);STUDIO_ADD_COUNTER_LOSS}
 uint previous=atomicAdd(studio_scalar_columns[b+2u],value);
 if(previous>0xffffffffu-value||prior>0xffffffffu-value)atomicOr(studio_scalar_columns[b+3u],8u);
}
'''


def with_scalar_columns(source,bands,*,shared=False):
    if type(bands) is not int or not 1<=bands<=8:raise ValueError('actual native layer count required')
    if 'StudioScalarColumns' in source:raise ValueError('scalar columns already installed')
    visit='void studio_scalar_visit(ivec3 q,int band,float raw){'
    add='void studio_scalar_add_result(ivec3 q,int band,uint prior,uint value){'
    if source.count(visit)!=1 or source.count(add)!=1:raise ValueError('original scalar visit/result seams required')
    # Existing RHS is evaluated once by the original observer. Copy its value.
    source=source.replace(visit,visit+'studio_scalar_column_attempt(q,band,raw);',1)
    source=source.replace(add,add+'studio_scalar_column_add(q,band,prior,value);',1)
    if shared:
        code=GLSL.replace('layout(std430,binding=25) coherent buffer StudioScalarColumns { uint studio_scalar_columns[]; };','#define studio_scalar_columns studio_scalar_visits')
        code=code.replace('return uint(((q.z', 'return uint(uDims.x*uDims.y*uDims.z)*2u+uint(((q.z')
        code=code.replace('STUDIO_ATTEMPT_COUNTER_LOSS','atomicOr(studio_scalar_visits[((q.z*uDims.y+q.y)*uDims.x+q.x)*2],1u<<uint(band+8));')
        code=code.replace('STUDIO_ADD_COUNTER_LOSS','atomicOr(studio_scalar_visits[((q.z*uDims.y+q.y)*uDims.x+q.x)*2],1u<<uint(band+16));')
        declaration='layout(std430,binding=23) coherent buffer StudioScalarVisits { uint studio_scalar_visits[]; };'
        if source.count(declaration)!=1:raise ValueError('shared scalar storage declaration changed')
        return source.replace(declaration,declaration+'\n#define STUDIO_COLUMN_BANDS '+str(bands)+'\n'+code,1)
    return source.replace('layout(std430,binding=23) coherent buffer',
        '#define STUDIO_COLUMN_BANDS '+str(bands)+'\n'+GLSL.replace('STUDIO_ATTEMPT_COUNTER_LOSS','').replace('STUDIO_ADD_COUNTER_LOSS','')+'\nlayout(std430,binding=23) coherent buffer',1)


def descriptor(dims,bands):
    return dict(version='native-scalar-integration-columns-v1',dimensions=list(dims),bands=bands,
        stride_bytes=16,byte_length=int(np.prod(dims))*bands*16,
        channels=['evaluated_finite_nonnegative_attempts','accepted_native_atomic_adds','native_uint_val_sum','loss_flags'],
        loss_bits=dict(rounded_positive=1,clipped=2,invalid=4,sum_overflow=8,attempt_count_overflow=16,accepted_count_overflow=32),
        normalization='none; exact native scalar uint val sums since matching scalar clear',
        sampling='events, not independent paths; evaluated zero is not physical darkness or convergence',
        attribution='mixed native scalar inputs; no per-material, per-medium or angular separation',
        interior_scope='only deposits already made by the original scalar tracer; terminal interior packets are separate')


def allocate(dims,bands):
    from OpenGL import GL
    meta=descriptor(dims,bands);size=meta['byte_length']
    if int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS))<=BINDING or size>int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BLOCK_SIZE)):
        raise MemoryError('native scalar columns exceed SSBO hardware admission')
    handle=int(GL.glGenBuffers(1));GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,handle)
    try:
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,size,None,GL.GL_DYNAMIC_DRAW)
        if int(GL.glGetBufferParameteriv(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_BUFFER_SIZE))!=size:raise MemoryError('native scalar columns allocation incomplete')
        clear(handle);bind(handle)
    except BaseException:
        GL.glDeleteBuffers(1,[handle]);raise
    return dict(buffer=handle,metadata=meta)


def bind(handle):
    from OpenGL import GL
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,BINDING,handle)


def clear(handle):
    from OpenGL import GL
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,handle)
    GL.glClearBufferData(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_R32UI,GL.GL_RED_INTEGER,GL.GL_UNSIGNED_INT,np.zeros(1,np.uint32))


def consume(record,consumer,identity):
    """Read native cells in bounded z-slabs; no event or full-matrix history."""
    from OpenGL import GL
    if not callable(consumer):raise TypeError('native scalar cell consumer required')
    meta=record['metadata'];bands=meta['bands'];dims=meta['dimensions']
    total=int(np.prod(dims));slab=int(dims[0])*int(dims[1])
    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT|GL.GL_BUFFER_UPDATE_BARRIER_BIT)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,record['buffer'])
    for start in range(0,total,slab):
        count=min(slab,total-start)
        raw=GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,record.get('offset_bytes',0)+start*bands*16,count*bands*16)
        values=np.frombuffer(raw,np.uint32).reshape(count,bands,4)
        values.setflags(write=False)
        metadata=dict(meta,**identity,cell_offset=start,total_cells=total,final_slab=start+count==total)
        consumer(metadata,values)
        # The synchronous consumer is allowed to run native work/change bindings.
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,record['buffer'])


def clear_view(record):
    from OpenGL import GL
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,record['buffer'])
    GL.glClearBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_R32UI,record.get('offset_bytes',0),
        record['metadata']['byte_length'],GL.GL_RED_INTEGER,GL.GL_UNSIGNED_INT,np.zeros(1,np.uint32))
