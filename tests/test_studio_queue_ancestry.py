"""CPU-only native queue composition/ownership. No GL context or rendering."""
import copy
import runpy
from pathlib import Path
import numpy as np
import pytest

from camera_designer.studio_forward_responses import (
    ANCESTRY_VERSION,FORWARD_ANCESTRY_CONTRACT,QUEUE_ENTRY_CONTRACT,
    ForwardResponseCopies,begin_native_queue,finish_native_dispatch,
    require_queue_idle,queued_ray_storage_plan,with_queued_ray_ancestry,
    copy_forward_responses,validate_queue_rows,
)


@pytest.mark.parametrize('regions',[False,True])
@pytest.mark.parametrize('pbr',[False,True])
def test_actual_queue_composition_copies_parent_without_new_optics(regions,pbr):
    from camera_designer.studio_directional_field import recording_shader
    from camera_designer.studio_directional_media import with_internal_medium_recording
    from camera_designer.studio_region_gpu import with_region_media
    from camera_designer.studio_opaque_pbr import with_opaque_pbr
    source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    if regions:source=with_region_media(source)
    if pbr:source=with_opaque_pbr(source)
    source=with_internal_medium_recording(recording_shader(source,estimator='track-length-root-sweep-v1'))
    queued=with_queued_ray_ancestry(source)
    assert queued.count('rand01(rng)')==source.count('rand01(rng)')
    for expression in ('packet.z * clamp(reflectivity, 0.0, 1.0)',
                       'min(packet.z * sqrt(lambda_d / r_edge) * 0.3, packet.z * 0.5)',
                       'min(bounces_rem, uMaxBounces)','normalize(pr.dir_bounces.xyz)'):
        assert queued.count(expression)==source.count(expression)==1
    assert 'reactive_main(ray_id, rng);' in queued and 'reactive_main(gid, rng);' not in queued
    assert 'studio_record_root_weight=uintBitsToFloat(pr.studio_sampling.x);' in queued
    assert 'uvec4(uStudioParentDispatch,ray_id,uint(_ray_idx),0u)' in queued
    assert 'floatBitsToUint(studio_record_root_weight),studio_queue_cause,0u,0u' in queued
    copied,_=copy_forward_responses(queued,terminal_only=True,queue_ancestry=True)
    assert 'studio_response_copy(5u,2u,gid,studio_queue_parent.z' in copied
    assert copied.count('studio_response_words[base+125u]=studio_queue_sampling.x;')==1
    assert copied.count('imageAtomicAdd(')==source.count('imageAtomicAdd(')
    with pytest.raises(ValueError,match='already installed'):with_queued_ray_ancestry(queued)


def test_queue_admission_uses_native_enqueue_bound_without_shrinking_jobs():
    from camera_designer.studio_forward_responses import dispatch_storage_plan
    assert queued_ray_storage_plan(512,6,max_ssbo_bytes=128*1024*1024)==dict(
        version='native-queued-ancestry-v1',capacity=10240,row_words=24,buffer_bytes=983040)
    assert queued_ray_storage_plan(512,1,max_ssbo_bytes=96)['capacity']==0
    with pytest.raises(MemoryError):queued_ray_storage_plan(512,6,max_ssbo_bytes=100)
    with pytest.raises(ValueError):queued_ray_storage_plan(True,6,max_ssbo_bytes=100)
    large=queued_ray_storage_plan(8192,200,max_ssbo_bytes=1024*1024*1024)
    with pytest.raises(MemoryError,match='peak requires'):
        dispatch_storage_plan(8192,200,reserved_bytes=large['buffer_bytes'],max_ssbo_bytes=1024*1024*1024)


class QueueGL:
    GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS=1;GL_MAX_SHADER_STORAGE_BLOCK_SIZE=2
    GL_SHADER_STORAGE_BUFFER=3;GL_DYNAMIC_DRAW=4;GL_BUFFER_SIZE=5
    GL_SHADER_STORAGE_BARRIER_BIT=8;GL_BUFFER_UPDATE_BARRIER_BIT=16
    GL_SHADER_IMAGE_ACCESS_BARRIER_BIT=32;GL_R32UI=17;GL_RED_INTEGER=18;GL_UNSIGNED_INT=19
    def __init__(self):
        self.data={104:bytearray(20)};self.bound=0;self.uniforms={};self.calls=[]
        self.fail_dispatch=False;self.fail_reset=False;self.corrupt_root=False;self.parent=None
    def glGetIntegerv(self,key):return 32 if key==1 else 128*1024*1024
    def glGenBuffers(self,n):return 19
    def glBindBuffer(self,target,handle):self.bound=handle
    def glBufferData(self,target,size,data,usage):self.data[self.bound]=bytearray(size)
    def glGetBufferParameteriv(self,*a):return len(self.data[self.bound])
    def glDeleteBuffers(self,*a):pass
    def glClearBufferSubData(self,target,internal,offset,size,*a):self.data[self.bound][offset:offset+size]=b'\0'*size
    def glGetBufferSubData(self,target,offset,size):return bytes(self.data[self.bound][offset:offset+size])
    def glBufferSubData(self,target,offset,size,data):
        if self.fail_reset:raise RuntimeError('uncertain counter reset')
        assert self.bound==104 and offset==16 and size==4
        self.calls.append(('reset',offset,size));self.data[self.bound][offset:offset+size]=data.tobytes()
    def glBindBufferBase(self,*a):pass
    def glGetUniformLocation(self,program,name):return name
    def glUniform1ui(self,name,value):self.uniforms[name]=value
    glUniform1i=glUniform1ui
    def glMemoryBarrier(self,*a):pass
    def attempted(self,count):self.data[104][16:20]=np.uint32(count).tobytes()
    def glDispatchCompute(self,*groups):
        u=self.uniforms;offset=u['batch_off'];count=u['batch_size'];parent=self.parent
        self.calls.append(('dispatch',offset,count,u['seed'],u['uStudioQueueSequenceBase']))
        rows=np.zeros((count,128),np.uint32)
        rows[:,0]=5;rows[:,1]=2;rows[:,2]=np.arange(offset,offset+count)
        rows[:,102]=parent['ray_offset']+np.arange(count)%parent['ray_count']
        rows[:,3]=rows[:,102]%3+(1 if self.corrupt_root else 0)
        rows[:,101]=parent['dispatch_id'];rows[:,103]=1;rows[:,125]=np.float32(.5).view(np.uint32)
        rows[:,126]=1+np.arange(count)%2;rows[:,127]=17
        self.data[19][:8+rows.nbytes]=np.array([count,0],np.uint32).tobytes()+rows.tobytes()
        if self.fail_dispatch:raise RuntimeError('uncertain after submission')


def setup_queue(*,consume=None):
    gl=QueueGL();seen=[]
    owner=ForwardResponseCopies(ray_count=2,bounce_limit=3,reserved_bytes=0,
        consume=consume or (lambda m,r:seen.append((copy.deepcopy(m),r.copy()))),gl=gl,queue_ancestry=True)
    state=dict(prog=9,ssbo=[100,101,102,103,104,105,106],source_buf=np.zeros((3,12),np.float32),
        _pending_cap=16,forward_response_owner=owner,response_entry_contract=QUEUE_ENTRY_CONTRACT,
        response_queue_owner=dict(phase='idle',logical_ray_base=0),response_dispatch_serial=1,
        locs={k:k for k in ('mode','seed','batch_size','batch_off','total_rays')})
    return gl,owner,state,seen


def begin(gl,owner,state,*,serial=0):
    meta=dict(scene_revision='a'*64,population_index=4,source_sha256='b'*64,
        stream_id='real-dispatch',sequence=serial,dispatch_id=serial,path_family='forward',
        ray_offset=2,ray_count=2,total_rays=6,bounce_limit=3,seed=42,root_sampling={})
    meta['dispatch']={k:meta[k] for k in ('dispatch_id','population_index','ray_offset','ray_count','total_rays','bounce_limit','seed','root_sampling')}
    gl.parent=copy.deepcopy(meta)
    begin_native_queue(state,meta);owner.begin(state['prog'],meta)
    return meta


def test_real_owner_consumes_bounded_offsets_and_resets_only_after_success():
    gl,owner,state,seen=setup_queue();meta=begin(gl,owner,state)
    gl.attempted(5)
    assert finish_native_dispatch(owner,state,lambda:None)==0
    assert gl.calls==[('dispatch',0,2,31415926,0),('dispatch',2,2,31415926,0),
        ('dispatch',4,1,31415926,0),('reset',16,4)]
    assert state['response_reactive_queue']['complete']
    assert state['response_queue_owner']['logical_ray_base']==5
    assert state['response_queue_owner']['phase']=='idle'
    assert len(seen)==4 and all(m['version']==ANCESTRY_VERSION for m,r in seen)
    for m,rows in seen[1:]:
        assert m['source_sha256']==meta['source_sha256']
        assert m['dispatch']['native_queue']['parent_dispatch']['dispatch_id']==0
        assert m['forward_ancestry_contract']==FORWARD_ANCESTRY_CONTRACT
        assert np.all(rows[:,125]==np.float32(.5).view(np.uint32))
    assert gl.uniforms['mode']==0 and gl.uniforms['seed']==42
    begin(gl,owner,state,serial=state['response_dispatch_serial']);gl.attempted(1)
    finish_native_dispatch(owner,state,lambda:None)
    assert ('dispatch',0,1,31415926,5) in gl.calls
    assert state['response_queue_owner']['logical_ray_base']==6


@pytest.mark.parametrize('failure',['submission','consumer','reset','ancestry','overflow'])
def test_uncertain_or_failed_additive_work_cannot_be_retried_or_cleared(failure):
    def consume(m,r):
        if m['path_family']=='reactive' and failure=='consumer':raise RuntimeError('consumer failed after readback')
    gl,owner,state,_=setup_queue(consume=consume);begin(gl,owner,state);gl.attempted(17 if failure=='overflow' else 3)
    gl.fail_dispatch=failure=='submission';gl.fail_reset=failure=='reset';gl.corrupt_root=failure=='ancestry'
    with pytest.raises((RuntimeError,MemoryError,ValueError)):finish_native_dispatch(owner,state,lambda:None)
    assert state['response_queue_owner']['phase']=='uncertain'
    assert not state['response_reactive_queue']['complete']
    with pytest.raises(RuntimeError,match='destroy/recreate'):require_queue_idle(state)
    calls=gl.calls.copy()
    with pytest.raises(RuntimeError):begin(gl,owner,state,serial=50)
    assert gl.calls==calls
    assert not any(c[0]=='reset' for c in calls)


def test_unexpected_backlog_and_token_overflow_fail_before_new_submission():
    gl,owner,state,_=setup_queue();gl.attempted(1)
    with pytest.raises(RuntimeError,match='backlog'):begin(gl,owner,state)
    assert gl.calls==[]
    gl,owner,state,_=setup_queue()
    with pytest.raises(ValueError,match='generation'):begin(gl,owner,state,serial=2**32)
    assert gl.calls==[]


def test_no_queued_children_is_observed_complete_not_a_fabricated_dispatch():
    gl,owner,state,seen=setup_queue();begin(gl,owner,state)
    finish_native_dispatch(owner,state,lambda:None)
    assert gl.calls==[('reset',16,4)] and len(seen)==1
    assert state['response_reactive_queue']['complete']


def test_mutation_guards_precede_any_real_gl_access_or_identity_update():
    import ast
    from camera_designer.studio_native_emitters import install_source_population,install_timed_light_frame
    state={'response_entry_contract':QUEUE_ENTRY_CONTRACT,'response_queue_owner':{'phase':'uncertain'}}
    tree=ast.parse(Path('demo_pluck_gl.py').read_text(encoding='utf-8'))
    for name,argc in [('_clear_prebuilt',1),('_update_prebuilt_geometry',10)]:
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        env={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-owner>','exec'),env)
        with pytest.raises(RuntimeError,match='destroy/recreate'):env[name](state,*([None]*(argc-1)))
    with pytest.raises(RuntimeError):install_source_population(None,state,None,None,None,None,None,ray_jobs=1)
    with pytest.raises(RuntimeError):install_timed_light_frame(state,None,None,None)


@pytest.mark.parametrize('children',[0,5])
def test_actual_dispatch_metadata_and_owner_deliver_v3_to_real_store(children):
    import ast
    from camera_designer.studio_response_lineage import NativeResponseLineage
    from camera_designer.studio_scene_responses import SceneResponseStore,validate_scene_response_snapshot
    lineage=NativeResponseLineage(0)
    store=SceneResponseStore(scene_revision='a'*64,memory_budget_bytes=1048576,lineage_provider=lineage.snapshot)
    gl,owner,state,_=setup_queue(consume=store.consume_forward)
    if not children:
        owner.bounce_limit=1;state['_pending_cap']=0
    state.update(response_identity=dict(scene_revision='a'*64,population_index=4),
        response_resource_receipt={},response_stream_id='actual-owner',response_dispatch_serial=0,
        response_transport_scene_sha256='c'*64,response_program_sha256='d'*64,response_auxiliary_layout={},
        directional_record=dict(scene_sha256='e'*64,root_sampling={},grid_size=32),
        bmin=np.zeros(3),bmax=np.ones(3))
    tree=ast.parse(Path('demo_pluck_gl.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_begin_prebuilt_responses')
    env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-owner-begin>','exec'),env)
    env['_begin_prebuilt_responses'](state,42,2,2,6)
    gl.parent=copy.deepcopy(owner.pending)
    # One actual forward disposition fixture, followed by optional native
    # secondary fixtures through the real owner/consumer/serializer boundaries.
    row=np.zeros((1,128),np.uint32);row[0,:4]=[6,1,2,2];row[0,5]=0xffffffff
    gl.data[19][:520]=np.array([1,0],np.uint32).tobytes()+row.tobytes()
    gl.attempted(children)
    finish_native_dispatch(owner,state,lambda:None)
    manifest,payload=store.snapshot()
    assert manifest['schema']=='studio-scene-responses-v3'
    assert validate_scene_response_snapshot(manifest,payload)
    assert manifest['sections'][1]['count']==1+children
    assert state['response_reactive_queue']['complete']
    assert all(c['source_sha256']==gl.parent['source_sha256'] for c in manifest['contexts'])


def test_ancestry_validation_peak_is_reserved_and_chunks_are_bounded(monkeypatch):
    import camera_designer.studio_forward_responses as module
    gl,owner,state,_=setup_queue()
    assert owner.plan['consumer_bytes']==module.QUEUE_VALIDATION_BYTES
    meta=begin(gl,owner,state)
    seen=[];original=module._validate_queue_row_chunk
    monkeypatch.setattr(module,'_validate_queue_row_chunk',lambda m,r,q,p,e:(seen.append(len(r)),original(m,r,q,p,e)))
    rows=np.zeros((2050,128),np.uint32);rows[:,1]=1
    validate_queue_rows(meta,rows)
    assert seen==[512,512,512,512,2]


def test_ordinary_renderer_is_not_modified_by_opt_in_queue_owner():
    import ast
    tree=ast.parse(Path('demo_pluck_gl.py').read_text(encoding='utf-8'))
    ordinary=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gpu_ray_field')
    assert not any(isinstance(n,ast.Name) and n.id=='queue_ancestry' for n in ast.walk(ordinary))
