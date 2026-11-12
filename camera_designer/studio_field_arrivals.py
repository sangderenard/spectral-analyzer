"""Lossless bounded CPU contract for actual forward arrival observations.

Counts are observations, not convergence or certificates of darkness. No GL
context or renderer is needed to validate or serialize this contract.
"""
import numpy as np

VERSION='arrival-record-v1'
UINT_MAX=2**32-1
CONTEXT=['medium_min','medium_max','bounce_mask','root_min','root_max',
         'endpoint_triangle_min','endpoint_triangle_max','path_kind_mask']
DTYPE=np.dtype([('index','<u4'),('spectral','<f4',(3,4)),('visits','<u4',(3,)),
                ('rounded_zero','<u4',(3,)),('excluded','<u4',(3,)),('context','<u4',(8,))])
assert DTYPE.itemsize==120


def checked_counts(value):
    a=np.asarray(value)
    if not np.all(np.isfinite(a)) or np.any(a<0) or np.any(a>UINT_MAX) or np.any(a!=np.floor(a)):
        raise ValueError('arrival counter overflow or invalid count')
    return a.astype('<u4')


def merge_context(left,right):
    """Merge encoded identity sets conservatively; min!=max means unresolved."""
    a=np.asarray(left,dtype=np.uint32);b=np.asarray(right,dtype=np.uint32)
    out=np.empty_like(a)
    for low,high in ((0,1),(3,4),(5,6)):
        out[...,low]=np.minimum(a[...,low],b[...,low])
        out[...,high]=np.maximum(a[...,high],b[...,high])
    out[...,2]=a[...,2]|b[...,2];out[...,7]=a[...,7]|b[...,7]
    return out


def encode_arrivals(energy,moments,visits,rounded_zero,excluded,context,*,
                    quantization_divisor,voxel_volume_m3,populations,offset_bytes,
                    payload_limit=256*1024*1024):
    """Serialize every visited/excluded bin, including measured quantized zero.

    Inputs retain raw accumulated integer values. Spectrum is650/550/450nm.
    Energy shape(N,3), moments(N,3,3), counts(N,3), context(N,8).
    No bin selection by energy and no truncation is permitted.
    """
    e=np.asarray(energy,dtype=np.float64);m=np.asarray(moments,dtype=np.float64)
    v=checked_counts(visits);z=checked_counts(rounded_zero);x=checked_counts(excluded)
    c=checked_counts(context)
    n=len(e)
    if e.shape!=(n,3) or m.shape!=(n,3,3) or any(a.shape!=(n,3) for a in (v,z,x)) or c.shape!=(n,8):
        raise ValueError('arrival array shape mismatch')
    if not np.isfinite(e).all() or not np.isfinite(m).all() or np.any(e<0):
        raise ValueError('invalid arrival energy or moments')
    if np.any(z>v):raise ValueError('rounded-zero count exceeds valid visits')
    if np.any((e>0)&(v==0)):raise ValueError('energy without arrival evidence')
    denominator=float(quantization_divisor)*float(voxel_volume_m3)*float(populations)
    if not np.isfinite(denominator) or denominator<=0 or populations!=int(populations):
        raise ValueError('invalid arrival normalization')
    selected=np.flatnonzero(np.any((v>0)|(x>0),axis=1))
    size=len(selected)*DTYPE.itemsize
    if offset_bytes+size>payload_limit:raise ValueError('complete arrival payload exceeds resource limit; no records omitted')
    rows=np.zeros(len(selected),DTYPE)
    rows['index']=selected;rows['spectral'][:,:,0]=e[selected]/denominator
    rows['spectral'][:,:,1:]=m[selected]/denominator
    if not np.isfinite(rows['spectral']).all():raise ValueError('arrival float32 overflow')
    for name,a in [('visits',v),('rounded_zero',z),('excluded',x),('context',c)]:rows[name]=a[selected]
    metadata=dict(offset_bytes=offset_bytes,byte_length=size,count=len(rows),stride_bytes=120,
        format=VERSION,layout='sparse-zyx-bin',wavelengths_nm=[650,550,450],
        spectral_channels=['energy','moment_x','moment_y','moment_z'],context_channels=CONTEXT,
        normalization='energy and vector moments divided by fixedpoint divisor, voxel volume and completed populations; counters unnormalized',
        quantization_divisor=quantization_divisor,voxel_volume_m3=voxel_volume_m3,populations=populations,
        index_count=n,medium_encoding='canonical medium + 2; 0 invalid, 1 exterior; unequal min/max unresolved',
        missing='no recorded arrival or exclusion; unknown, not black',
        sampling='counts include correlated root replays; not independent samples or a convergence certificate',
        contributors='actual forward/reactive track-length; camera adjoint and NEE are not summed into this estimator')
    return rows.tobytes(),metadata

# Collision-resolved full domain keys; no hash-only identities or dropped slots.
DOMAIN_VERSION='arrival-domain-record-v1'
DOMAIN_DTYPE=np.dtype(DTYPE.descr+[('domain_words','<u4',(32,))])
SLOTS=262144
STRIDE_WORDS=64
BINDINGS=(21,)


def observation_shader(source,grid,bins):
    import re
    match=re.search(r'float scaled=\(packet.z\*studio_record_root_weight\)\*meters\*([0-9]+)\.0;',source)
    if not match:raise ValueError('arrival observer requires root-normalized recorder')
    from camera_designer.studio_field_arrival_gpu import table_glsl
    declarations=table_glsl(grid,bins,int(match[1]))
    anchor='void studio_record_track(ivec3 q,vec3 d,vec4 packet,float meters) {'
    if source.count(anchor)!=1:raise ValueError('arrival track seam changed')
    source=source.replace(anchor,declarations+'\n'+anchor+'\n    arrival_observe(q,d,packet,meters);\n    if(arrival_medium < -1)return;')
    old='    if(medium < -1)return; // invalid ownership is not air or glass'
    if source.count(old)!=1:raise ValueError('arrival observation requires canonical internal observer')
    domain='for(int i=0;i<32;++i)arrival_domain[i]=studio_active[i];' if 'uint studio_active[32];' in source else 'for(int i=0;i<32;++i)arrival_domain[i]=0u;'
    source=source.replace(old,'    arrival_medium=medium;arrival_domain_id=-1;'+domain)
    source=source.replace('    if(isnan(studio_record_sigma) || studio_record_sigma<0.0)return;',
        '    if(isnan(studio_record_sigma) || studio_record_sigma<0.0){arrival_medium=-2;studio_record_sigma=0.0;}')
    call='studio_record_segment(ro,hp,packet,currentMedium);'
    if source.count(call)!=2:raise ValueError('arrival forward/reactive segment seams changed')
    source=source.replace(call,'arrival_bounce=uint(bounce);arrival_root=uint(_ray_idx);arrival_endpoint=uint(hit);arrival_kind=1u;'+call,1)
    needle='        '+call
    if source.count(needle)!=1:raise ValueError('reactive arrival seam ambiguous')
    source=source.replace(needle,'        arrival_bounce=uint(bounce);arrival_root=0xffffffffu;arrival_endpoint=uint(hit);arrival_kind=2u;'+call,1)
    # Escaping traced rays still carry their actual packet through the ROI.
    bounds=('uStudioRecordMin','uStudioRecordMax') if 'uniform vec3 uStudioRecordMin;' in source else ('uBoxMin','uBoxMax')
    escape="""void arrival_escape(vec3 ro,vec3 rd,vec4 packet,int medium){
 float end=1e30;
 for(int k=0;k<3;++k)if(rd[k]!=0.0)end=min(end,max((LOW[k]-ro[k])/rd[k],(HIGH[k]-ro[k])/rd[k]));
 if(end>0.0 && end<1e29)studio_record_segment(ro,ro+rd*end,packet,medium);
}
""".replace('LOW',bounds[0]).replace('HIGH',bounds[1])
    anchor='void splat_spectral(vec3 p, vec4 packet) {'
    if source.count(anchor)!=1:raise ValueError('arrival escape declaration seam changed')
    source=source.replace(anchor,escape+anchor)
    miss='int hit = nearest_hit(ro, rd, best);\n        if (hit < 0) break;'
    if source.count(miss)!=2:raise ValueError('canonical escape trace seams changed')
    for kind,root in ((1,'uint(_ray_idx)'),(2,'0xffffffffu')):
        source=source.replace(miss,'int hit = nearest_hit(ro, rd, best);\n        if (hit < 0) { arrival_bounce=uint(bounce);arrival_root='+root+';arrival_endpoint=0xffffffffu;arrival_kind='+str(kind)+'u;arrival_escape(ro,rd,packet,currentMedium);break;}',1)
    return source


def allocate_observations(record):
    from OpenGL import GL
    from camera_designer.studio_field_arrival_gpu import resource_plan
    plan=resource_plan(record['grid_size'],record['bin_count'])
    if int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS))<=25:raise ValueError('arrival information requires26 SSBO bindings')
    if int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BLOCK_SIZE))<max(plan['table_chunk_bytes']):raise ValueError('arrival chunk exceeds supported SSBO block size')
    record['arrival_resources']=plan
    handles=[]
    for binding,size in [*zip(range(21,25),plan['table_chunk_bytes']),(25,plan['domain_bytes'])]:
        h=int(GL.glGenBuffers(1));handles.append(h);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,h)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,size,None,GL.GL_DYNAMIC_DRAW)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,h)
    record['arrival_buffers']=handles;clear_observations(record)


def clear_observations(record,preserve=False):
    if not record.get('arrival_buffers'):return
    from OpenGL import GL
    for h in record['arrival_buffers']:
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,h)
        GL.glClearBufferData(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_R32UI,GL.GL_RED_INTEGER,GL.GL_UNSIGNED_INT,np.zeros(1,np.uint32))
    if not preserve:record.pop('arrival_history',None)


def bind_observations(record):
    from OpenGL import GL
    bindings=[*range(21,21+len(record['arrival_buffers'])-1),25]
    for b,h in zip(bindings,record['arrival_buffers']):GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,b,h)


def merge_domain_rows(history,rows,capacity=None):
    """Merge exact domain identities without pretending samples are independent."""
    for row in rows:
        if row[0]!=2:raise ValueError('arrival table contains incomplete key')
        key=tuple(map(int,row[1:35]))
        values=np.concatenate([row[35:38].astype(np.float64),row[38:47].view(np.int32).astype(np.float64)])
        if np.any(values[:3]<0):raise ValueError('arrival energy overflow')
        counts=row[47:56].astype(np.uint64);context=row[56:62].copy()
        if key not in history:
            if len(history)>=(capacity or SLOTS):raise ValueError('progressive arrival domain capacity exhausted; no domains discarded')
            history[key]=[values,counts,context]
        else:
            old=history[key]
            if np.max(np.abs(old[0]))+np.max(np.abs(values))>=2**53:raise ValueError('arrival moment precision exhausted')
            old[0]+=values;old[1]+=counts;checked_counts(old[1])
            old[2][0]|=context[0];old[2][5]|=context[5]
            for lo,hi in ((1,2),(3,4)):
                old[2][lo]=min(old[2][lo],context[lo]);old[2][hi]=max(old[2][hi],context[hi])
    return history


def commit_observations(record):
    if not record.get('arrival_buffers'):return
    from OpenGL import GL
    GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT|GL.GL_SHADER_STORAGE_BARRIER_BIT)
    plan=record['arrival_resources']
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,record['arrival_buffers'][-1])
    domains=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,plan['domain_bytes']),np.uint32).reshape(-1,36)
    if np.any((domains[:,0]!=0)&(domains[:,0]!=2)):raise ValueError('incomplete arrival domain dictionary')
    history=record.get('arrival_history',{})
    for handle,size in zip(record['arrival_buffers'][:-1],plan['table_chunk_bytes']):
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,handle)
        raw=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,size),np.uint32).reshape(-1,32)
        selected=raw[raw[:,0]!=0]
        if np.any(selected[:,0]!=2):raise ValueError('incomplete arrival record')
        if np.any(selected[:,2]>=len(domains)):raise ValueError('arrival domain outside dictionary')
        identities=domains[selected[:,2]]
        if np.any(identities[:,0]!=2):raise ValueError('arrival references incomplete domain')
        # CPU conversion preserves exact identities; it does not reconstruct missing samples.
        rows=np.zeros((len(selected),64),np.uint32)
        rows[:,0]=2;rows[:,1]=selected[:,1];rows[:,2:35]=identities[:,1:34]
        rows[:,35:38]=selected[:,3:6];rows[:,38:47]=selected[:,6:15]
        rows[:,47:56]=selected[:,15:24];rows[:,56:62]=selected[:,24:30]
        history=merge_domain_rows(history,rows,plan['record_capacity'])
    record['arrival_history']=history


def export_observations(record,offset):
    history=record['arrival_history'];size=len(history)*DOMAIN_DTYPE.itemsize
    if offset+size>256*1024*1024:raise ValueError('complete arrival payload exceeds resource limit; no records omitted')
    rows=np.zeros(len(history),DOMAIN_DTYPE)
    populations=record['progressive']['metadata']['count']
    denominator=record['quantization_divisor']*record['voxel_volume_m3']*populations
    for i,key in enumerate(sorted(history)):
        value,counts,context=history[key]
        rows[i]['index']=key[0];rows[i]['domain_words']=key[2:]
        rows[i]['spectral'][:,0]=value[:3]/denominator
        rows[i]['spectral'][:,1:]=value[3:].reshape(3,3)/denominator
        rows[i]['visits']=checked_counts(counts[:3]);rows[i]['rounded_zero']=checked_counts(counts[3:6]);rows[i]['excluded']=checked_counts(counts[6:])
        rows[i]['context']=[key[1],key[1],*context]
    if not np.isfinite(rows['spectral']).all():raise ValueError('arrival float32 overflow')
    return rows.tobytes(),dict(offset_bytes=offset,byte_length=size,count=len(rows),stride_bytes=248,
        format=DOMAIN_VERSION,layout='sparse-zyx-bin-domain',wavelengths_nm=[650,550,450],
        spectral_channels=['energy','moment_x','moment_y','moment_z'],context_channels=CONTEXT,
        domain_word_count=32,domain_identity='exact canonical medium + 2 and full active-region uint32 bitset; never hash-only',
        index_count=record['grid_size']**3*record['bin_count'],quantization_divisor=record['quantization_divisor'],
        voxel_volume_m3=record['voxel_volume_m3'],populations=populations,
        normalization='energy and vector moments divided by fixedpoint divisor, voxel volume and completed populations; counters unnormalized',
        capacity_records=SLOTS,overflow='reject complete publication; no saturation, dropping or partial table',
        missing='no recorded arrival or exclusion; unknown, not black',
        sampling='counts include correlated root replays; not independent samples or a convergence certificate',
        contributors='actual forward/reactive track-length; camera adjoint and NEE are not summed into this estimator')

WIRE_DTYPE=np.dtype(DTYPE.descr+[('domain_id','<u4')])
DOMAIN_KEY_DTYPE=np.dtype([('medium','<i4'),('active','<u4',(32,))])
assert WIRE_DTYPE.itemsize==124 and DOMAIN_KEY_DTYPE.itemsize==132


def chunk_sections(record):
    """Streaming CPU sections with deduplicated, collision-free optical domains."""
    history=record['arrival_history']
    keys=sorted(history)
    identities=sorted(set(key[1:] for key in keys))
    if len(identities)>4096:raise ValueError('progressive exact domain dictionary capacity exhausted')
    lookup={key:i for i,key in enumerate(identities)}
    domains=np.zeros(len(identities),DOMAIN_KEY_DTYPE)
    for i,key in enumerate(identities):domains[i]['medium']=key[0]-2;domains[i]['active']=key[1:]
    populations=record['progressive']['metadata']['count']
    denominator=record['quantization_divisor']*record['voxel_volume_m3']*populations
    if not np.isfinite(denominator) or denominator<=0:raise ValueError('invalid arrival normalization')
    def blocks():
        for start in range(0,len(keys),65536):
            selected=keys[start:start+65536];rows=np.zeros(len(selected),WIRE_DTYPE)
            for i,key in enumerate(selected):
                value,counts,context=history[key]
                if np.any(value[:3]<0) or not np.isfinite(value).all():raise ValueError('invalid arrival values')
                if np.any(counts[3:6]>counts[:3]) or np.any((value[:3]>0)&(counts[:3]==0)):raise ValueError('inconsistent arrival evidence')
                rows[i]['index']=key[0];rows[i]['domain_id']=lookup[key[1:]]
                rows[i]['spectral'][:,0]=value[:3]/denominator
                rows[i]['spectral'][:,1:]=value[3:].reshape(3,3)/denominator
                rows[i]['visits']=checked_counts(counts[:3]);rows[i]['rounded_zero']=checked_counts(counts[3:6]);rows[i]['excluded']=checked_counts(counts[6:])
                rows[i]['context']=[key[1],key[1],*context]
            if not np.isfinite(rows['spectral']).all():raise ValueError('arrival float32 overflow')
            yield rows.tobytes()
    meta=dict(wavelengths_nm=[650,550,450],spectral_channels=['energy','moment_x','moment_y','moment_z'],
        context_channels=CONTEXT,index_count=record['grid_size']**3*record['bin_count'],
        quantization_divisor=record['quantization_divisor'],voxel_volume_m3=record['voxel_volume_m3'],populations=populations,
        normalization='energy/moments divided by fixedpoint divisor, voxel volume and completed populations; counts unnormalized',
        sampling='correlated root replays; no convergence certificate',missing='unknown, not black',
        contributors={'forward':'actual exterior/interior and escaping segments','reactive':'queue ancestry not retained; no normalized reactive contribution claimed','camera_adjoint':'separate estimator, not in forward density','nee':'separate estimator, not in forward density'},
        resources=record.get('arrival_resources',{}),direction_space='world propagation',
        moment_quantization='per-line signed component sums, <=0.5 count rounding per deposited component; no coherence multiplier')
    return [dict(id='arrivals',format='arrival-domain-id-record-v1',stride_bytes=124,count=len(keys),byte_length=len(keys)*124,metadata=meta,blocks=blocks()),
        dict(id='domains',format='domain-key-record-v1',stride_bytes=132,count=len(domains),byte_length=domains.nbytes,
             metadata=dict(identity='row index; exact canonical signed medium plus32 active-region words',domain_word_count=32),blocks=iter([domains.tobytes()]))]
