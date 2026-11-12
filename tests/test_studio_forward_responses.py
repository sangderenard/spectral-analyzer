"""CPU-only exact-RHS copy seam gates; no rendering or GL context."""
import runpy
from pathlib import Path
import pytest
from camera_designer.studio_forward_responses import copy_forward_responses
from camera_designer.studio_region_gpu import with_region_media
from camera_designer.studio_opaque_pbr import with_opaque_pbr

@pytest.mark.parametrize('region',[False,True])
@pytest.mark.parametrize('corrected',[False,True])
def test_exact_actual_response_is_evaluated_once(region,corrected):
    source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    if region:source=with_region_media(source)
    if corrected:source=with_opaque_pbr(source)
    assert 'StudioResponseCopies' not in source
    copied,seams=copy_forward_responses(source)
    assert sum(s['kind']==1 for s in seams)==2
    for item in seams:
        assert copied.count(item['expression'])==source.count(item['expression'])
    assert copied.count('studio_boundary_step(tri,rd,packet.x,rand01(rng),currentMedium,factor)')==2
    assert copied.count('apply_wave_context_physics(ro, rd, best, packet)')==source.count('apply_wave_context_physics(ro, rd, best, packet)')
    assert 'if(slot>=uStudioResponseCapacity)' in copied
    if region:assert 'studio_active[i]' in copied
    with pytest.raises(ValueError,match='already installed'):copy_forward_responses(copied)

@pytest.mark.parametrize('spheres',[False,True])
@pytest.mark.parametrize('terminal_only',[False,True])
@pytest.mark.parametrize('region',[False,True])
@pytest.mark.parametrize('queue_ancestry',[False,True])
def test_copied_response_compiles_offline_without_gpu(tmp_path,region,terminal_only,spheres,queue_ancestry):
    import shutil,subprocess
    from camera_designer.spectral_sensor_basis import FIXED3_SPECTRAL_GLSL
    from camera_designer.studio_diffuse_surface import STUDIO_DIFFUSE_GLSL
    from camera_designer.studio_spectral_reflector import STUDIO_REFLECTOR_GLSL
    from camera_designer.studio_boundary_media import STUDIO_BOUNDARY_GLSL
    import mat_flags as flags
    source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    prefix=''.join('#define '+key+' '+str(getattr(flags,key))+'u\n' for key in ('MAT_FLAG_SPECTRAL_REFLECTOR','MAT_FLAG_ROUGH_CONDUCTOR','MAT_FLAG_NATIVE_CONDUCTOR_NK'))
    prefix+='#define STUDIO_NATIVE_PBR_BINDING '+str(flags.BINDING_STUDIO_NATIVE_PBR)+'\n'
    source=source.replace('__FIXED3_SPECTRAL_HELPERS__',FIXED3_SPECTRAL_GLSL).replace(STUDIO_BOUNDARY_GLSL,STUDIO_DIFFUSE_GLSL+prefix+STUDIO_REFLECTOR_GLSL+STUDIO_BOUNDARY_GLSL)
    if region:source=with_region_media(source)
    if spheres:
        from camera_designer.studio_sphere_regions import with_sphere_regions
        source=with_sphere_regions(source)
    from camera_designer.studio_scalar_visits import with_scalar_visits
    from camera_designer.studio_scalar_columns import with_scalar_columns
    source=with_opaque_pbr(source)
    if queue_ancestry:
        from camera_designer.studio_directional_field import recording_shader
        from camera_designer.studio_directional_media import with_internal_medium_recording
        from camera_designer.studio_forward_responses import with_queued_ray_ancestry
        source=with_queued_ray_ancestry(with_internal_medium_recording(recording_shader(source,estimator='track-length-root-sweep-v1')))
    source,seams=copy_forward_responses(with_scalar_columns(with_scalar_visits(source),3,shared=True),terminal_only=terminal_only,shared_scalar_buffer=True,precleared_rows=True,queue_ancestry=queue_ancestry)
    path=tmp_path/'forward_responses.comp';path.write_text(source,encoding='utf-8')
    result=subprocess.run([shutil.which('glslangValidator'),str(path)],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr


def test_exact_dispatch_storage_does_not_shrink_original_schedule():
    from camera_designer.studio_forward_responses import dispatch_storage_plan
    plan=dispatch_storage_plan(8192,6,reserved_bytes=0)
    assert plan['capacity_rows']==49152
    assert plan['buffer_bytes']==25165832
    assert plan['peak_bytes']==50331664
    assert dispatch_storage_plan(512,6,reserved_bytes=0)['buffer_bytes']==1572872
    with pytest.raises(MemoryError,match='peak requires'):
        dispatch_storage_plan(8192,6,reserved_bytes=500*1024*1024)
    with pytest.raises(MemoryError,match='SSBO limit'):
        dispatch_storage_plan(8192,6,reserved_bytes=0,max_ssbo_bytes=1024)


def test_terminal_mode_keeps_intermediate_attributes_out_of_row_count():
    source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    copied,_=copy_forward_responses(with_opaque_pbr(source),terminal_only=True)
    assert not any('studio_response_copy('+str(k)+'u,' in copied for k in (1,2,3,4))
    assert 'studio_response_stage_factors[0]=(studio_copied_factor)' in copied
    assert copied.count('studio_response_stage_presence=0u;')==3 # declaration and two original loops
    assert copied.count('rand01(rng)')==with_opaque_pbr(source).count('rand01(rng)')


class FakeResponseGL:
    GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS=1;GL_MAX_SHADER_STORAGE_BLOCK_SIZE=2
    GL_SHADER_STORAGE_BUFFER=3;GL_DYNAMIC_DRAW=4;GL_BUFFER_SIZE=5
    GL_SHADER_STORAGE_BARRIER_BIT=8;GL_BUFFER_UPDATE_BARRIER_BIT=16
    GL_R32UI=17;GL_RED_INTEGER=18;GL_UNSIGNED_INT=19
    def __init__(self):self.size=0;self.header=b'\0'*8;self.deleted=[]
    def glGetIntegerv(self,key):return 32 if key==1 else 128*1024*1024
    def glGenBuffers(self,n):return 19
    def glBindBuffer(self,*a):pass
    def glBufferData(self,target,size,data,usage):assert data is None;self.size=size
    def glGetBufferParameteriv(self,*a):return self.size
    def glDeleteBuffers(self,n,handles):self.deleted+=handles
    def glBufferSubData(self,*a):self.header=a[-1].tobytes()
    def glClearBufferSubData(self,*a):self.header=b'\0'*8
    def glBindBufferBase(self,*a):pass
    def glUniform1ui(self,*a):pass
    def glGetUniformLocation(self,*a):return 7
    def glMemoryBarrier(self,*a):pass
    def glGetBufferSubData(self,target,offset,size):return self.header if offset==0 else b'\0'*size


def test_transient_owner_requires_actual_identity_and_consumer_no_history():
    import numpy as np
    from camera_designer.studio_forward_responses import ForwardResponseCopies
    gl=FakeResponseGL();seen=[]
    owner=ForwardResponseCopies(ray_count=512,bounce_limit=6,reserved_bytes=0,consume=lambda m,r:seen.append((m,r.shape)),gl=gl)
    with pytest.raises(ValueError,match='ancestry'):owner.begin(1,{})
    meta=dict(scene_revision='scene',population_index=2,source_sha256='source',dispatch_id=3,path_family='forward',ray_offset=512,ray_count=512,bounce_limit=6,seed=42,root_sampling={'source_count':4464})
    owner.begin(1,meta);gl.header=np.array([1,0],np.uint32).tobytes()
    assert owner.finish()==1
    assert seen[0][1]==(1,128) and seen[0][0]['independent_sample_count'] is None
    assert owner.pending is None
    owner.begin(1,meta);gl.header=np.array([3073,1],np.uint32).tobytes()
    with pytest.raises(MemoryError,match='overflow'):owner.finish()
    assert len(seen)==1
    owner.close();assert gl.deleted==[19]


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_prebuilt_pump_preserves_launches_and_rng(monkeypatch,enabled):
    import ast,math,sys,types
    import camera_designer.studio_directional_field as directional
    import camera_designer.studio_forward_responses as copies
    source=Path('demo_pluck_gl.py').read_text(encoding='utf-8')
    tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_gpu_pump_prebuilt')
    calls=[];gl=types.ModuleType('OpenGL.GL')
    for key in ('GL_SHADER_STORAGE_BARRIER_BIT','GL_SHADER_IMAGE_ACCESS_BARRIER_BIT','GL_SHADER_STORAGE_BUFFER','GL_R32UI','GL_READ_WRITE','GL_TEXTURE_3D'):setattr(gl,key,1)
    for key in ('glUseProgram','glBindBufferBase','glBindImageTexture','glMemoryBarrier'):setattr(gl,key,lambda *a:None)
    gl.glUniform1i=lambda *a:calls.append(('uniform',a))
    gl.glDispatchCompute=lambda *a:calls.append(('dispatch',a))
    monkeypatch.setitem(sys.modules,'OpenGL.GL',gl)
    for name in ('prepare_root_dispatch','submitted_root_chunk','finish_root_dispatch','abort_root_dispatch'):monkeypatch.setattr(directional,name,lambda *a:None)
    monkeypatch.setattr(copies,'finish_native_dispatch',lambda owner,state,restore:(calls.append(('finish',)),restore()))
    env=dict(math=math,GL_TRUE=1,BINDING_STUDIO_NATIVE_PBR=18,
        _begin_prebuilt_responses=lambda *a:calls.append(('begin',a[1:])),
        _restore_prebuilt_bindings=lambda state:calls.append(('restore',)),
        _consume_prebuilt_scalar_columns=lambda state:calls.append(('scalar_cells',)))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-pump>','exec'),env)
    state=dict(n_sources=4464,prog=9,locs={k:k for k in ('mode','seed','total_rays','batch_size','batch_off')},dispatch_batch=512,tex_bands=[],ssbo=[],native_pbr_ssbo=7,_pump_seed=42)
    if enabled:state['forward_response_owner']=object()
    env['_gpu_pump_prebuilt'](state,1100)
    assert [x for x in calls if x[0]=='dispatch']==[('dispatch',(4,1,1)),('dispatch',(4,1,1)),('dispatch',(1,1,1))]
    assert state['_pump_seed']==49
    assert len([x for x in calls if x[0]=='begin'])==(3 if enabled else 0)
    assert len([x for x in calls if x[0]=='restore'])==(3 if enabled else 0)


def test_native_queue_counts_do_not_claim_unprocessed_paths_are_calculated():
    from camera_designer.studio_forward_responses import native_queue_status
    q=native_queue_status(11,8)
    assert q['dropped_enqueue']==3 and q['pending_unprocessed']==8 and not q['complete']
    assert native_queue_status(0,8)['complete']


def test_scalar_columns_copy_native_attempt_and_add_not_new_energy():
    from camera_designer.studio_scalar_columns import descriptor,with_scalar_columns
    from camera_designer.studio_scalar_visits import with_scalar_visits
    source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    observed=with_scalar_columns(with_scalar_visits(source),3)
    assert descriptor((96,96,96),3)['byte_length']==42467328
    assert observed.count('imageAtomicAdd(')==source.count('imageAtomicAdd(')==8
    assert observed.count('energy * coherence * 65535.0')==source.count('energy * coherence * 65535.0')
    assert 'studio_scalar_column_attempt(q,band,raw)' in observed
    assert 'studio_scalar_column_add(q,band,prior,value)' in observed
    assert 'atomicAdd(studio_scalar_columns[b+2u],value)' in observed
    assert 'raw>0.0&&raw<1.0' in observed
    assert 'old==0xffffffffu' in observed


def test_combined_compute_block_count_is_explicit_hardware_admission():
    from camera_designer.studio_forward_responses import admit_shader_storage
    source='layout(std430,binding=1) buffer A {uint a[];}; layout(std430,binding=25) buffer B {uint b[];};'
    assert admit_shader_storage(source,2)['declared_compute_storage_blocks']==2
    assert not admit_shader_storage(source,2)['hardware_link_verified']
    with pytest.raises(MemoryError,match='hardware limit'):admit_shader_storage(source,1)


def test_shared_auxiliary_storage_adds_only_one_compute_block():
    from camera_designer.studio_scalar_visits import with_scalar_visits
    from camera_designer.studio_scalar_columns import with_scalar_columns
    from camera_designer.studio_forward_responses import shader_storage_requirements
    source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    base=shader_storage_requirements(source)['declared_compute_storage_blocks']
    observed=copy_forward_responses(with_scalar_columns(with_scalar_visits(source),3,shared=True),terminal_only=True,shared_scalar_buffer=True)[0]
    assert shader_storage_requirements(observed)['declared_compute_storage_blocks']==base+1
    assert 'uStudioResponseWordOffset+2u+slot*128u' in observed
    assert 'uint(uDims.x*uDims.y*uDims.z)*2u+' in observed
    assert observed.count('imageAtomicAdd(')==source.count('imageAtomicAdd(')


def test_borrowed_terminal_reset_and_close_never_touch_other_views():
    import numpy as np
    from camera_designer.studio_forward_responses import ForwardResponseCopies
    class SharedGL(FakeResponseGL):
        def __init__(self):super().__init__();self.data=bytearray([165])*(64+8+512);self.size=len(self.data)
        def glBufferSubData(self,target,offset,size,data):self.data[offset:offset+size]=data.tobytes()
        def glClearBufferSubData(self,target,internal,offset,size,format_,type_,data):self.data[offset:offset+size]=b'\0'*size
        def glGetBufferSubData(self,target,offset,size):return bytes(self.data[offset:offset+size])
    gl=SharedGL();seen=[]
    owner=ForwardResponseCopies(ray_count=1,bounce_limit=1,reserved_bytes=64,consume=lambda m,r:seen.append(r.copy()),gl=gl,borrowed_buffer=19,offset_bytes=64)
    m=dict(scene_revision='s',population_index=0,source_sha256='h',dispatch_id=0,path_family='forward',ray_offset=0,ray_count=1,bounce_limit=1,seed=1,root_sampling={})
    owner.begin(1,m)
    assert gl.data[:64]==bytes([165])*64
    assert gl.data[64:]==b'\0'*(8+512)
    assert owner.finish()==0
    owner.close();assert gl.deleted==[]

def test_queue_missing_work_stays_unknown_across_scalar_clears():
    from camera_designer.studio_forward_responses import record_queue_observation
    state={'_pending_cap':4,'scalar_clear_epoch':0}
    assert record_queue_observation(state,0)['complete']
    q=record_queue_observation(state,3)
    assert q['observed'] and q['pending_unprocessed']==3 and not q['complete']
    state['scalar_clear_epoch']=1
    q=record_queue_observation(state,7)
    assert q['dropped_enqueue']==3 and q['pending_unprocessed']==4 and q['incomplete_observed']
    q=record_queue_observation(state,0)
    assert q['counter_regression'] and not q['complete'] and q['incomplete_observed']
