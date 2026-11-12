"""Opt-in track-length moments of actual exterior ray segments; transport unchanged."""
import hashlib
import numpy as np

class DirectionalUnavailable(ValueError):
    def __init__(self,reason,message=None,*,recovery='next field clear'):
        self.reason=reason;self.recovery=recovery
        super().__init__(message or reason)


class DirectionalOverflow(DirectionalUnavailable):
    """The optional recorder overflowed; canonical transport may continue."""
    def __init__(self,message):
        super().__init__('signed-accumulator-overflow',message)


BINDING = 19
DIMS = (32, 32, 32)
DEFAULT_STENCIL = 12
LEGACY_ESTIMATOR = 'track-length-throughput-v1'
ROOT_ESTIMATOR = 'track-length-root-sweep-v1'


def recording_options(estimator=LEGACY_ESTIMATOR,grid_size=None):
    if estimator not in (LEGACY_ESTIMATOR,ROOT_ESTIMATOR):
        raise ValueError('unsupported light_field_estimator')
    grid=(64 if estimator==ROOT_ESTIMATOR else 32) if grid_size is None else grid_size
    if type(grid) is not int or grid not in (32,64) or (estimator==LEGACY_ESTIMATOR and grid!=32):
        raise ValueError('light_field_grid_size must be32 for legacy or32/64 for root-sweep estimator')
    return estimator,grid


def root_multiplicities(root_count,jobs):
    if type(root_count) is not int or root_count<=0 or type(jobs) is not int or jobs<0:
        raise ValueError('invalid source-root or dispatch count')
    result=np.full(root_count,jobs//root_count,np.int64)
    result[:jobs%root_count]+=1
    return result


def reset_root_accounting(record):
    if record.get('estimator')!=ROOT_ESTIMATOR:return
    n=record['source_root_count']
    record['root_sampling']=dict(version='per-complete-pump-inverse-replay-v1',
        source_root_count=n,completed_root_sweeps=0,recorded_root_jobs=0,
        normalization_divisor=0,excluded_partial_root_jobs=0,incomplete_dispatch=False)
    record['pending_dispatch']=None
    record['root_replay_counts']=np.zeros(n,np.int64)


def begin_root_dispatch(record,jobs):
    if record is None or record.get('estimator')!=ROOT_ESTIMATOR:return None
    if type(jobs) is not int or jobs<0:raise ValueError('invalid root dispatch size')
    if record.get('pending_dispatch') is not None:record['root_sampling']['incomplete_dispatch']=True
    eligible=record['source_root_count']>0 and jobs>=record['source_root_count']
    record['pending_dispatch']=dict(jobs=jobs,submitted=0,eligible=eligible)
    return eligible and not record['root_sampling']['incomplete_dispatch']


def submitted_root_chunk(record,jobs):
    if record is None or record.get('estimator')!=ROOT_ESTIMATOR:return
    pending=record['pending_dispatch']
    if pending is None or type(jobs) is not int or jobs<0 or pending['submitted']+jobs>pending['jobs']:
        record['root_sampling']['incomplete_dispatch']=True
        raise ValueError('root dispatch accounting mismatch')
    pending['submitted']+=jobs


def finish_root_dispatch(record):
    if record is None or record.get('estimator')!=ROOT_ESTIMATOR:return
    pending=record['pending_dispatch'];stats=record['root_sampling']
    if pending is None or pending['submitted']!=pending['jobs']:
        stats['incomplete_dispatch']=True
        return
    record['pending_dispatch']=None
    if stats['incomplete_dispatch']:return
    if not pending['eligible']:
        stats['excluded_partial_root_jobs']+=pending['submitted']
        return
    stats['completed_root_sweeps']+=1
    stats['normalization_divisor']=stats['completed_root_sweeps']
    stats['recorded_root_jobs']+=pending['submitted']
    record['root_replay_counts']+=root_multiplicities(record['source_root_count'],pending['submitted'])


def abort_root_dispatch(record):
    if record is not None and record.get('estimator')==ROOT_ESTIMATOR:
        record['root_sampling']['incomplete_dispatch']=True


def root_normalization(record):
    if record.get('estimator')!=ROOT_ESTIMATOR:return 1
    stats=record['root_sampling']
    if stats['incomplete_dispatch'] or record.get('pending_dispatch') is not None:
        raise DirectionalUnavailable('incomplete-root-dispatch')
    if not stats['completed_root_sweeps']:
        raise DirectionalUnavailable('incomplete-root-coverage',recovery='next completed full-coverage pump')
    return stats['completed_root_sweeps']


def prepare_root_dispatch(record,program,jobs):
    enabled=begin_root_dispatch(record,jobs)
    if enabled is not None:
        from OpenGL import GL
        GL.glUniform1i(GL.glGetUniformLocation(program,'uStudioRecordRootSweepEnabled'),int(enabled))



def validate_stencil(value):
    if type(value) is not int or value not in (6,12):
        raise ValueError('light_field_stencil must be integer6 or12')
    return value


def stencil_directions(stencil=DEFAULT_STENCIL):
    stencil=validate_stencil(stencil)
    if stencil==6:
        return np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],np.float64)
    phi=(1+np.sqrt(5.0))/2
    directions=np.array([[0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
        [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
        [phi,0,1],[phi,0,-1],[-phi,0,1],[-phi,0,-1]],np.float64)
    return directions/np.linalg.norm(directions,axis=1,keepdims=True)


def subject_recording_region(vertices, *, scale_m, center_m, cell_size_m, spheres=None, margin_m=.001):
    """Bounded exact-pitch ROI, using the canonical float32 world transform."""
    pitch=float(cell_size_m)
    if not np.isfinite(pitch) or not .00005<=pitch<=.01:raise ValueError('light_field_cell_size_m must be between0.00005 and0.01 metres')
    if not np.isfinite(scale_m) or scale_m<=0:raise ValueError('invalid subject scale')
    if np.asarray(center_m).shape!=(3,):raise ValueError('invalid subject center')
    world=np.ascontiguousarray(np.asarray(vertices,np.float32)*np.float32(scale_m)+np.asarray(center_m,np.float32),dtype='<f4')
    if world.size==0 or world.shape[-2:]!=(3,3) or not np.isfinite(world).all():raise ValueError('invalid subject geometry for recording ROI')
    points=world.reshape(-1,3).astype(np.float64)
    sphere=np.asarray(spheres if spheres is not None else [],dtype='<f4').reshape(-1,8)
    if len(sphere):points=np.concatenate((points,sphere[:,:3]-sphere[:,3,None],sphere[:,:3]+sphere[:,3,None]))
    lo=points.min(axis=0);hi=points.max(axis=0)
    margin=float(margin_m)
    if not np.isfinite(margin) or not 0<=margin<=.032:raise ValueError('light_field_margin_m must be between0 and0.032 metres')
    padding=max(margin,pitch)
    lower=np.floor((lo-padding)/pitch).astype(np.int64);upper=np.ceil((hi+padding)/pitch).astype(np.int64)
    required=np.maximum(upper-lower,1)
    grid=next((n for n in (32,64) if np.all(required<=n)),None)
    if grid is None:raise ValueError(f'subject recording ROI requires {required.tolist()} cells at {pitch} m; maximum is64 per axis; no coarsening applied')
    lower-=(grid-required)//2
    digest=hashlib.sha256(world.tobytes()+sphere.tobytes()).hexdigest()
    return dict(bounds_min_m=(lower*pitch).tolist(),bounds_max_m=((lower+grid)*pitch).tolist(),cell_size_m=pitch,
        dimensions=[grid]*3,margin_m=margin,region=dict(kind='subject',subject_geometry_sha256=digest,effective_margin_m=padding,
        padding_min_m=(lo-lower*pitch).tolist(),padding_max_m=((lower+grid)*pitch-hi).tolist(),
        margin_semantics='effective_margin_m is the minimum guard-adjusted padding before snapping; padding_min_m/max_m are actual final per-axis padding',
        actual_bounds_min_m=lo.tolist(),actual_bounds_max_m=hi.tolist()))


def subject_field_region(vertices,config):
    spheres=[]
    if config.get('sphere_coverage','mesh')=='exact':
        # Validation remains in canonical geometry preparation; this receipt uses
        # the same submitted analytic sphere transform, then checks finite bounds.
        for item in config.get('analytic_spheres',{}).get('spheres',[]):
            row=np.zeros(8,np.float32)
            row[:3]=np.asarray(config.get('object_center_m',[0,0,.18]),np.float64)+float(config.get('scale_m',.025))*np.asarray(item['center'])
            row[3]=float(config.get('scale_m',.025))*float(item['radius'])
            if not np.isfinite(row).all() or row[3]<=0:raise ValueError('invalid analytic sphere ROI')
            spheres.append(row)
    result=subject_recording_region(vertices,scale_m=config.get('scale_m',.025),center_m=config.get('object_center_m',[0,0,.18]),cell_size_m=config['light_field_cell_size_m'],margin_m=config.get('light_field_margin_m',.001),spheres=spheres)
    result['region']['world_geometry_sha256']=result['region']['subject_geometry_sha256']
    result['region']['subject_geometry_sha256']=config.get('geometry_sha256',result['region']['world_geometry_sha256'])
    return result


def word_count(stencil=DEFAULT_STENCIL,grid_size=32):
    _,grid=recording_options(ROOT_ESTIMATOR,grid_size)
    return grid**3*validate_stencil(stencil)*6


def weight_from_linear_rgb():
    from camera_designer.marble_color_profile import XYZ_TO_SRGB,_LINE_TO_XYZ
    return np.linalg.solve((XYZ_TO_SRGB@_LINE_TO_XYZ).T,np.ones(3))


WORDS = word_count()  # default12; runtime allocations use their explicit bin count

_GLSL_TEMPLATE = r'''
layout(std430,binding=19) buffer StudioDirectionalField { int studio_directional[]; };
const int STUDIO_DIRECTION_BIN_COUNT=__STUDIO_BIN_COUNT__;
const vec3 STUDIO_DIRECTION_BASIS[STUDIO_DIRECTION_BIN_COUNT]=vec3[](__STUDIO_DIRECTIONS__);
// Private to this shader invocation; refreshed once for every recorded segment.
int studio_record_bin=0;
void studio_direction_add(int index,int value) {
    int old=atomicAdd(studio_directional[index],value);
    if ((value>0 && old>2147483647-value) || (value<0 && old<(-2147483647-1)-value))
        atomicOr(studio_directional[__STUDIO_WORD_COUNT__],1);
}
void studio_record_track(ivec3 q,vec3 d,vec4 packet,float meters) {
    float scaled=packet.z*meters*65535.0;
    // Never silently clip high throughput into a purported measurement.
    if(scaled>=2147483000.0){atomicOr(studio_directional[__STUDIO_WORD_COUNT__],1);return;}
    int w=int(round(max(scaled,0.0)));
    if(w<=0)return;
    int base=(((q.z*__STUDIO_GRID__+q.y)*__STUDIO_GRID__+q.x)*STUDIO_DIRECTION_BIN_COUNT+studio_record_bin)*6;
    studio_direction_add(base+studio_band_index(packet.x),w);
    studio_direction_add(base+3,int(round(float(w)*d.x)));
    studio_direction_add(base+4,int(round(float(w)*d.y)));
    studio_direction_add(base+5,int(round(float(w)*d.z)));
}
void studio_record_segment(vec3 a,vec3 b,vec4 packet) {
    if(uFixed3Transport==0 || !(packet.z>0.0))return;
    vec3 delta=b-a;
    float length_m=length(delta);
    if(!(length_m>0.0))return;
    vec3 d=delta/length_m;
    float first=0.0,last=length_m;
    // Clip the existing segment, not a newly traced ray, to the field box.
    for(int k=0;k<3;++k){
        if(d[k]==0.0){if(a[k]<uBoxMin[k] || a[k]>=uBoxMax[k])return;}
        else{
            float ta=(uBoxMin[k]-a[k])/d[k],tb=(uBoxMax[k]-a[k])/d[k];
            first=max(first,min(ta,tb));last=min(last,max(ta,tb));
        }
    }
    if(!(last>first))return;
    // Basis vectors classify actual directions; never substitute them for
    // the measured propagation moments. Strict > gives lowest-index ties.
    studio_record_bin=0;
    float closest=dot(d,STUDIO_DIRECTION_BASIS[0]);
    for(int i=1;i<STUDIO_DIRECTION_BIN_COUNT;++i){
        float score=dot(d,STUDIO_DIRECTION_BASIS[i]);
        if(score>closest){closest=score;studio_record_bin=i;}
    }
    vec3 cell=(uBoxMax-uBoxMin)/float(__STUDIO_GRID__);
    vec3 grid=(a+d*first-uBoxMin)/cell;
    ivec3 q;
    for(int k=0;k<3;++k)q[k]=clamp(int(d[k]<0.0 ? ceil(grid[k])-1.0 : floor(grid[k])),0,__STUDIO_GRID_MAX__);
    float cursor=first;
    // At most grid-size crossings per axis plus the initial clipped cell.
    for(int iteration=0;iteration<__STUDIO_DDA_CAP__;++iteration){
        if(any(lessThan(q,ivec3(0)))||any(greaterThanEqual(q,ivec3(__STUDIO_GRID__))))return;
        vec3 crossing=vec3(1e30);
        for(int k=0;k<3;++k)if(d[k]!=0.0){
            float edge=uBoxMin[k]+float(q[k]+(d[k]>0.0 ? 1 : 0))*cell[k];
            crossing[k]=(edge-a[k])/d[k];
        }
        float next=max(cursor,min(last,min(crossing.x,min(crossing.y,crossing.z))));
        if(next>cursor)studio_record_track(q,d,packet,next-cursor);
        if(next>=last)return;
        for(int k=0;k<3;++k)if(crossing[k]<=next)q[k]+=(d[k]>0.0 ? 1 : -1);
        cursor=next;
    }
    // A changed traversal contract must not silently publish partial paths.
    atomicOr(studio_directional[__STUDIO_WORD_COUNT__],2);
}
'''


def directional_glsl(stencil=DEFAULT_STENCIL,grid_size=None,estimator=LEGACY_ESTIMATOR):
    estimator,grid=recording_options(estimator,grid_size)
    stencil=validate_stencil(stencil)
    directions=','.join('vec3('+','.join(format(float(v),'.17e') for v in row)+')'
                        for row in stencil_directions(stencil))
    source=(_GLSL_TEMPLATE.replace('__STUDIO_BIN_COUNT__',str(stencil))
            .replace('__STUDIO_WORD_COUNT__',str(word_count(stencil,grid)))
            .replace('__STUDIO_DIRECTIONS__',directions)
            .replace('__STUDIO_GRID__',str(grid)).replace('__STUDIO_GRID_MAX__',str(grid-1))
            .replace('__STUDIO_DDA_CAP__',str(3*grid+3)))
    if estimator==ROOT_ESTIMATOR:
        source='uniform int uStudioRecordRootSweepEnabled;\nfloat studio_record_root_weight=0.0;\n'+source
        source=source.replace('float scaled=packet.z*meters*65535.0;','float scaled=(packet.z*studio_record_root_weight)*meters*65535.0;')
        source=source.replace('if(uFixed3Transport==0 || !(packet.z>0.0))return;','if(uFixed3Transport==0 || !(packet.z>0.0) || studio_record_root_weight<=0.0)return;')
    return source


GLSL=directional_glsl()  # standalone GPU verification uses the same default


def recording_shader(source,stencil=DEFAULT_STENCIL,grid_size=None,estimator=LEGACY_ESTIMATOR,independent_bounds=False,quantization_divisor=65535):
    estimator,grid=recording_options(estimator,grid_size)
    anchor='void splat_spectral(vec3 p, vec4 packet) {'
    entries=[]
    for call in ('splat_segment_multiscale(ro, hp, packet, path_len);','splat_segment_spectral(ro, hp, packet);'):
        candidates=['if(currentMedium'+comparison+') '+call for comparison in ('<0','==-1')]
        if sum(source.count(item) for item in candidates)!=1:
            raise ValueError('canonical forward/reactive segment recording entries changed')
        entries.append(next(item for item in candidates if source.count(item)==1))
    if source.count(anchor)!=1:
        raise ValueError('canonical splat declaration changed')
    if quantization_divisor not in (65535,16777216):raise ValueError('unsupported directional fixedpoint divisor')
    recorder=directional_glsl(stencil,grid,estimator).replace('65535.0',f'{quantization_divisor}.0')
    if independent_bounds:
        recorder='uniform vec3 uStudioRecordMin;\nuniform vec3 uStudioRecordMax;\n'+recorder.replace('uBoxMin','uStudioRecordMin').replace('uBoxMax','uStudioRecordMax')
    source=source.replace(anchor,recorder+'\n'+anchor)
    for entry in entries:
        guard,call=entry.split(') ',1)
        source=source.replace(entry,guard+') { studio_record_segment(ro,hp,packet); '+call+' }')
    if estimator==ROOT_ESTIMATOR:
        root='    int _ray_idx = int(ray_id % uint(max(uSourceCount, 1)));'
        if source.count(root)!=1:raise ValueError('canonical source-root selection changed')
        source=source.replace(root,root+"\n    if(uStudioRecordRootSweepEnabled!=0 && uSourceCount>0 && uTotalRaysPerSource>=uSourceCount) {\n        int replay=uTotalRaysPerSource/uSourceCount+(_ray_idx<uTotalRaysPerSource%uSourceCount ? 1 : 0);\n        studio_record_root_weight=1.0/float(replay);\n    }",1)
    return source


def triangle_metadata(geom, shade, native_pbr=None):
    geom=np.asarray(geom,np.float32); shade=np.asarray(shade,np.float32)
    flags=geom[:,15].copy().view(np.uint32)
    # Original packed scene triangle ordering, never BVH ordering.
    result=np.zeros((len(geom),4),dtype='<u4')
    result[:,0]=np.arange(len(geom))
    result[:,1]=shade[:,14].copy().view(np.uint32)
    result[:,2]=flags
    # 1 volume domain, 2 reflective surface, 4 transmissive/glass surface.
    # Every physical triangle can receive the adjacent volume estimate.
    # The canonical nonabsorber surface has at least its clamped/interface
    # reflection, even when compatibility reflectivity is authored as zero.
    result[:,3]=1 | ((flags&4)==0).astype(np.uint32)*2
    result[:,3]|=((shade[:,7]<.999)|((flags&64)!=0)).astype(np.uint32)*4
    if native_pbr is not None:
        pbr=np.asarray(native_pbr,np.float32)
        if np.any(result[:,1]>=len(pbr)):raise ValueError('triangle material outside native PBR table')
        rows=pbr[result[:,1]]
        result[:,3]|=((rows[:,5]>0)|(rows[:,7]<.999)).astype(np.uint32)*4
    return result


def scene_digest(geom, shade, materials, sources, *extra):
    h=hashlib.sha256()
    for data in (geom,shade,materials,sources,*extra):
        a=np.ascontiguousarray(data,dtype="<f4")
        h.update(np.asarray(a.shape,dtype="<u4").tobytes());h.update(a.tobytes())
    return h.hexdigest()


def voxel_volume(bounds,grid_size=32):
    lo,hi=np.asarray(bounds,dtype=np.float64)
    volume=float(np.prod((hi-lo)/grid_size))
    if not np.isfinite(volume) or volume<=0 or np.any(hi<=lo):
        raise ValueError("invalid directional field bounds")
    return volume


def allocate(geom,shade,native_pbr=None,*,bounds,stencil=DEFAULT_STENCIL,grid_size=None,estimator=LEGACY_ESTIMATOR,source_root_count=0):
    from OpenGL import GL
    stencil=validate_stencil(stencil)
    estimator,grid=recording_options(estimator,grid_size)
    words=word_count(stencil,grid)
    triangles=triangle_metadata(geom,shade,native_pbr)
    volume=voxel_volume(bounds,grid)
    if int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS))<=BINDING:
        raise RuntimeError('directional field requires 20 SSBO bindings')
    handle=int(GL.glGenBuffers(1))
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,handle)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,(words+1)*4,np.zeros(words+1,np.int32),GL.GL_DYNAMIC_DRAW)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,BINDING,handle)
    record={'buffer':handle,'triangles':triangles,'voxel_volume_m3':volume,'bin_count':stencil,'grid_size':grid,'estimator':estimator,'source_root_count':int(source_root_count)}
    reset_root_accounting(record)
    return record


def clear(record, preserve_progressive=False):
    from OpenGL import GL
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,record['buffer'])
    GL.glClearBufferData(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_R32I,GL.GL_RED_INTEGER,GL.GL_INT,np.zeros(1,np.int32))
    reset_root_accounting(record)
    if not preserve_progressive:record.pop('progressive',None)


def _decode_directional_slices(raw, record, divisor):
    """Same native decode, bounded one-z-slab color/weight temporaries."""
    from camera_designer.marble_color_profile import XYZ_TO_SRGB, _LINE_TO_XYZ
    result=np.zeros(raw.shape,dtype='<f4')
    matrix=(XYZ_TO_SRGB@_LINE_TO_XYZ).T
    for z in range(raw.shape[0]):
        slab=raw[z];out=result[z]
        if np.any(slab[...,:3]<0):raise DirectionalOverflow('directional field accumulator overflow; clear more frequently')
        out[...,:3]=((slab[...,:3].astype(np.float64)/record.get('quantization_divisor',65535))@matrix)/(record['voxel_volume_m3']*divisor)
        weight=slab[...,:3].astype(np.float64).sum(axis=-1)
        np.divide(slab[...,3:],weight[...,None],out=out[...,3:],where=weight[...,None]>0)
        norm=np.linalg.norm(out[...,3:],axis=-1,keepdims=True)
        out[...,3:]/=np.maximum(norm,1)
    return result


def readback(record):
    from OpenGL import GL
    GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT|GL.GL_SHADER_STORAGE_BARRIER_BIT)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,record['buffer'])
    divisor=root_normalization(record)
    grid=record.get('grid_size',32)
    bins=validate_stencil(record['bin_count'])
    words=word_count(bins,grid)
    if record.get('progressive'):
        # A committed population has already been copied into the owned sum.
        # Still inspect the native loss flag; do not copy its unused full atlas.
        overflow=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,words*4,4),dtype=np.int32)
        if overflow[0]:raise DirectionalOverflow('directional field signed accumulator overflow; clear more frequently')
        raw=record['progressive']['sum']
        divisor=record['progressive']['metadata']['count']
    else:
        raw=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,(words+1)*4),dtype=np.int32)
        if raw[-1]:raise DirectionalOverflow('directional field signed accumulator overflow; clear more frequently')
        raw=raw[:-1].reshape(grid,grid,grid,bins,6)
    return _decode_directional_slices(raw,record,divisor)


def commit_source_population(record,sources,settings,index):
    """Merge one complete corrected population, before any fresh GPU clear."""
    from OpenGL import GL
    if root_normalization(record)!=1:raise ValueError('progressive population must contain exactly one complete pump')
    grid=record['grid_size'];bins=record['bin_count']
    GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT|GL.GL_SHADER_STORAGE_BARRIER_BIT)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,record['buffer'])
    raw=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,(word_count(bins,grid)+1)*4),np.int32)
    if raw[-1]:raise DirectionalOverflow('population recorder overflow')
    raw=raw[:-1].reshape(grid,grid,grid,bins,6)
    if np.any(raw[...,:3]<0):raise DirectionalOverflow('population recorder overflow')
    digest=hashlib.sha256(np.ascontiguousarray(sources,dtype='<f4').tobytes()).hexdigest()
    history=record.get('progressive')
    if history is None:
        history=dict(sum=np.zeros(raw.shape,np.float64),metadata=dict(version='native-angular-refresh-mean-v1',generator='native-emissive-triangle-roots-v1',ray_budget=settings['ray_budget'],base_seed=settings['seed'],first_index=index,last_index=index-1,count=0,rolling_sha256='0'*64),root_sampling=dict(record['root_sampling']))
        history['root_sampling']['recorded_root_jobs']=0
    meta=history['metadata']
    if index!=meta['last_index']+1 or history['root_sampling']['source_root_count']!=len(sources):raise ValueError('source population sequence or root count changed')
    # Float64 integer sums are exact while below2**53; reject before precision loss.
    if np.max(np.abs(history['sum']))+np.max(np.abs(raw.astype(np.float64)))>=2**53:raise DirectionalUnavailable('progressive-precision-budget-exhausted')
    history['sum']+=raw
    meta.update(last_index=index,count=meta['count']+1,latest_seed=(settings['seed']+index)%2147483648,latest_sources_sha256=digest,
        rolling_sha256=hashlib.sha256(bytes.fromhex(meta['rolling_sha256'])+index.to_bytes(8,'little')+bytes.fromhex(digest)).hexdigest())
    stats=history['root_sampling'];stats['completed_root_sweeps']=stats['normalization_divisor']=meta['count'];stats['recorded_root_jobs']+=record['root_sampling']['recorded_root_jobs']
    history['nonzero_cells']=int(np.count_nonzero(np.any(history['sum'][...,:3]>0,axis=(-1,-2))))
    record['progressive']=history


def partial_estimate_validity():
    """Information actually available, never a certification of darkness."""
    return dict(version='partial-track-estimate-v1',zero_semantics='unknown',
        nonzero_semantics='recorded-quantized-throughput',spatial_sampling='unreported',
        angular_sampling='coarse-bin-mean-only',spectral_sampling='three-fixed-lines',
        occlusion_certification='none',convergence_certification='none')


def bundle_sections(record, source_buf, scalar_bytes, *, streaming=False):
    atlas=readback(record); triangles=record['triangles']; sources=np.ascontiguousarray(source_buf,dtype='<f4')
    materials=np.ascontiguousarray(record['materials'],dtype='<f4')
    lineage=list(dict(((int(i),np.asarray(row,dtype='<f4').tobytes()),None) for i,row in record.get('material_lineage',[])))
    lineage_ids=[item[0] for item in lineage]
    lineage_bytes=b''.join(item[1] for item in lineage)
    parts=[(memoryview(np.ascontiguousarray(a)).cast('B') if a.size else b'') for a in (atlas,triangles,sources,materials)]+[lineage_bytes]
    bins=validate_stencil(record['bin_count'])
    offset=scalar_bytes
    directional=dict(offset_bytes=offset,byte_length=len(parts[0]),dimensions=[record.get('grid_size',32)]*3,
        stencil='axes6-v1' if bins==6 else 'icosahedron12-v1',bin_count=bins,
        bin_directions=stencil_directions(bins).tolist(),weight_from_linear_rgb=weight_from_linear_rgb().tolist(),
        bins=['+X','-X','+Y','-Y','+Z','-Z'] if bins==6 else [f'I{i}' for i in range(12)],channels=['R','G','B','mean_propagation_x','mean_propagation_y','mean_propagation_z'],
        format='rgb-direction-f32le',layout='zyx-bin-channel',direction_space='world',
        estimator=record.get('estimator',LEGACY_ESTIMATOR),units='transport-energy/m^2',
        voxel_volume_m3=record.get('voxel_volume_m3'),quantization_divisor=record.get('quantization_divisor',65535),
        normalization='divided by voxel volume; not divided by ray jobs or solid angle',
        direction_weight='sum of nonnegative fixed3 energy times in-voxel segment length',
        direction_quantization='nearest integer energy-length and moment per segment-voxel; <=0.5 count rounding each; mean clamped to unit ball',
        overflow_policy='signed atomic overflow publishes scalar-only v1 with explicit unavailable reason until field clear',
        semantics='Actual exterior segment entry-packet throughput times exact clipped in-voxel length, divided by voxel volume. No legacy fog attenuation, scattering coefficient or phase-coherence filter. Pre-wave-context entry throughput; not an interior-medium estimate, calibrated irradiance or full angular radiance. Surface emission without traced direction excluded.')
    directional['validity']=partial_estimate_validity()
    if record.get('estimator')==ROOT_ESTIMATOR:
        root_normalization(record)
        directional['root_sampling']=dict(record.get('progressive',{}).get('root_sampling',record['root_sampling']))
        if record.get('progressive'):
            directional['source_populations']=dict(record['progressive']['metadata'])
            directional['quantization_version']='fixed24-track-length-v1' if record.get('quantization_divisor',65535)==16777216 else 'fixed16-track-length-v1'
            directional['coverage']={'nonzero_cells':record['progressive']['nonzero_cells'],'total_cells':record['grid_size']**3,'meaning':'cells with recorded nonzero quantized throughput; zero is not proof of no light'}
        directional['normalization']='per root inverse actual replay multiplicity within each completed full-coverage pump, then mean of those pumps and divided by voxel volume; no solid-angle normalization'
    if record.get('volume_extinction') is not None:
        directional['volume_extinction']=record['volume_extinction']
        directional['direction_weight']='sum of nonnegative fixed3 entry energy times integral of supplied-volume transmittance over in-voxel segment length'
        directional['semantics']+=' Supplied piecewise-constant extinction is integrated along the actual segment; no ambient density is invented.'
    if record.get('estimator')==ROOT_ESTIMATOR:
        directional['direction_weight']+='; inverse actual within-pump root replay multiplicity applied to every recorded contribution'
        directional['direction_quantization']+='; root replay correction precedes quantization, so very small weighted contributions may round to zero'
    if record.get('internal_media'):
        directional['internal_media']=dict(version='canonical-beer-segment-integral-v1',scope='actual valid interior and exterior traced segments',entry_packet=True,integration='per-recording-cell integral of canonical Beer transmittance',optical_authority='installed medium/region/dye helper',invalid_ownership='excluded',external_extinction='unsupported; request rejected')
        directional['semantics']='Actual traced segment entry-packet throughput times the per-cell integral of installed canonical medium/region/dye Beer transmittance, divided by voxel volume and averaged over complete refreshed populations. Valid interior and exterior segments only; invalid ownership excluded. No added scattering, rays, fog or unvisited-cell reconstruction; not calibrated irradiance or complete angular radiance.'
        directional['direction_weight']='sum of nonnegative fixed3 entry energy times per-cell integral of canonical Beer transmittance, with inverse actual root replay multiplicity'
    if record.get('recording_region') is not None:
        directional.update(record['recording_region'])
    offset+=len(parts[0])
    tri=dict(offset_bytes=offset,byte_length=len(parts[1]),count=len(triangles),stride_bytes=16,format='u32le',
        channels=['triangle_id','material_id','flags','class_mask'],class_bits={'volumetric':1,'reflectance':2,'glass':4},ordering='original packed scene triangles, before BVH reorder')
    offset+=len(parts[1])
    src=dict(offset_bytes=offset,byte_length=len(parts[2]),count=len(sources),stride_bytes=48,format='f32le',layout='baked-source-records',
        channels=['position_x_m','position_y_m','position_z_m','amplitude_weight','direction_x','direction_y','direction_z','kind','frequency_hz','phase_radians','energy','packet_aux'],
        semantics='Actual pre-baked source rays, not emitter identities; atlas combines all sources and does not preserve per-source ancestry')
    if record.get('progressive'):src['semantics']='Latest completed source population only; preceding populations are reproducible from directional.source_populations generator and seed sequence, with rolling buffer hash'
    offset+=len(parts[2])
    mat=dict(offset_bytes=offset,byte_length=len(parts[3]),count=len(materials),stride_bytes=64,format='f32le',layout='native-pbr16')
    offset+=len(parts[3])
    authored=dict(offset_bytes=offset,byte_length=len(parts[4]),count=len(lineage),stride_bytes=64,format='f32le',layout='authored-pbr16',material_ids=lineage_ids)
    sections=dict(directional=directional,triangles=tri,sources=src,materials=mat,source_pbr16=authored,scene_sha256=record['scene_sha256'])
    return (parts if streaming else b''.join(parts)),sections
