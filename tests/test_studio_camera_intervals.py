import ast
from pathlib import Path
import pytest
from camera_designer.studio_camera_intervals import observe_camera_continuation


def canonical():
    root = ast.parse((Path(__file__).resolve().parents[1] / 'demo_pluck_gl.py').read_text(encoding='utf-8'))
    for node in root.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(isinstance(t, ast.Name) and t.id == '_GPU_SENSOR_CS' for t in node.targets):
                return node.value.value
    raise AssertionError('canonical sensor missing')


def test_absent_observer_exact_identity():
    source = canonical()
    assert observe_camera_continuation(source) is source


def test_observer_preserves_rng_and_original_optics():
    source = canonical()
    observed = observe_camera_continuation(source, enabled=True)
    assert observed.count('rand01(rng)') == source.count('rand01(rng)')
    assert observed.count('studio_boundary_step(tri,rd,frequency,rand01(rng),currentMedium,factor)') == 1
    assert 'throughput[i]*=studio_medium_transmittance(currentMedium,frequency,surf_t);' in observed
    assert observed.index('uint sci_row=') < observed.index('throughput[i]*=studio_medium_transmittance')
    assert 'sample_layer_energy' in observed
    assert 'uint cap=min(uCameraIntervalCapacity' in observed
    assert 'atomicOr(sci_words[1],1u)' in observed
    assert 'hit< -1?6u:(medium==-2?5u:(hit==-1?3u:0u))' in observed


def test_full_region_state_and_strict_seams():
    source = 'uint studio_active[32];\n' + canonical()
    observed = observe_camera_continuation(source, enabled=True)
    assert 'sci_words[o+48u+uint(j)]=studio_active[j]' in observed
    assert 'sci_words[o+80u+uint(j)]=studio_active[j]' in observed
    with pytest.raises(ValueError):
        observe_camera_continuation(observed, enabled=True)
    with pytest.raises(ValueError):
        observe_camera_continuation(source.replace('int   hit_id', 'int hit_id'), enabled=True)

def test_observer_configuration_is_cpu_only_and_identity_bound():
    from camera_designer.studio_camera_intervals import CameraIntervalObserver
    item = CameraIntervalObserver(1, 12, scene_revision='scene-hash', camera_revision='camera-hash')
    assert item.identity['required_bytes'] == 8 + 12 * 512
    assert item.identity['shared_band0_path'] is True
    with pytest.raises(ValueError):
        CameraIntervalObserver(1, 12, scene_revision='', camera_revision='camera')


def test_runtime_opt_in_is_explicit_and_replay_rejected():
    source = (Path(__file__).resolve().parents[1] / 'demo_pluck_gl.py').read_text(encoding='utf-8')
    assert 'surface_transport=None,camera_interval_observer=None,field_continuation=False,light_field_information=None):' in source
    assert 'if camera_interval_observer is not None:' in source
    assert 'sensor_shader_source = observe_camera_continuation(sensor_shader_source, enabled=True)' in source
    assert "camera interval observer does not yet support retained loan replay" in source

def test_dispatch_sizing_accounts_for_all_paths_and_response_buffers():
    from camera_designer.studio_camera_intervals import plan_interval_dispatch
    plan = plan_interval_dispatch(width=128, rows=16, samples=1, path_bands=3,
        max_bounces=5, memory_budget_bytes=80*1024*1024, reserved_bytes=0,
        row_granularity=16, response_copies=True)
    assert plan['rows'] == 16
    assert plan['capacity'] == 128*16*3*6
    assert plan['total_peak_bytes'] <= 80*1024*1024
    with pytest.raises(MemoryError):
        plan_interval_dispatch(width=128, rows=16, samples=1,path_bands=3,
            max_bounces=5,memory_budget_bytes=1024,reserved_bytes=0,row_granularity=16)


def test_response_copies_preserve_rhs_once_and_zero_presence():
    from camera_designer.studio_camera_responses import copy_camera_responses
    source = observe_camera_continuation(canonical(), enabled=True)
    copied = copy_camera_responses(source, enabled=True)
    assert copied.count('rand01(rng)') == source.count('rand01(rng)')
    assert copied.count('sample_layer_energy(hp, i)') == source.count('sample_layer_energy(hp, i)')
    assert 'sci_response(sci_row,3,i,(throughput[i] * sample_layer_energy' in copied
    assert 'atomicOr(scr_words[2u+row*128u+96u+uint(kind)],1u<<uint(band));' in copied
    assert 'return value;' in copied
    assert copy_camera_responses(source) is source


def test_response_evidence_lifecycle_with_cpu_fake_gl(monkeypatch):
    import struct
    import sys
    import types
    from camera_designer.studio_sensor_evidence import StudioResponseEvidence
    class FakeGL:
        GL_SHADER_STORAGE_BUFFER=1; GL_DYNAMIC_READ=2; GL_R32UI=3
        GL_RED_INTEGER=4; GL_UNSIGNED_INT=5; GL_BUFFER_SIZE=6; GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS=7; GL_MAX_SHADER_STORAGE_BLOCK_SIZE=8
        def __init__(self): self.buffers={}; self.current=0; self.next=1
        def glGetIntegerv(self,pname): return 10_000_000 if pname==8 else 32
        def glGenBuffers(self,n):
            result=self.next;self.next+=1;self.buffers[result]=bytearray();return result
        def glBindBuffer(self,target,handle): self.current=handle
        def glBufferData(self,target,size,data,usage): self.buffers[self.current]=bytearray(size)
        def glBufferSubData(self,target,offset,size,data): self.buffers[self.current][offset:offset+size]=data
        def glClearBufferData(self,*args): self.buffers[self.current][:]=bytes(len(self.buffers[self.current]))
        def glGetBufferSubData(self,target,offset,size): return bytes(self.buffers[self.current][offset:offset+size])
        def glGetBufferParameteriv(self,*args): return len(self.buffers[self.current])
        def glBindBufferBase(self,*args): pass
        def glGetUniformLocation(self,*args): return 1
        def glUniform1ui(self,*args): pass
        def glDeleteBuffers(self,n,handles):
            for handle in handles: del self.buffers[handle]
    gl=FakeGL(); monkeypatch.setitem(sys.modules,'OpenGL',types.SimpleNamespace(GL=gl))
    collected=[]
    evidence=StudioResponseEvidence(memory_budget_bytes=100000,reserved_bytes=1000,
        scene_revision='scene',camera_revision='camera',consume=lambda meta,rows,responses:collected.append((meta,len(rows),len(responses))))
    assert evidence.prepare(width=1,rows=2,samples=1,path_bands=1,max_bounces=1)==2
    evidence.bind(99)
    gl.buffers[evidence.buffer][0:8]=struct.pack('<II',1,0)
    evidence.finish(dispatch_identity={'seed':1})
    assert collected[0][1:]==(512,512)
    assert collected[0][0]['excludes'] == ['NEE sample lineage', 'untraced paths', 'field response evaluation']
    gl.buffers[evidence.buffer][0:8]=struct.pack('<II',5,1)
    with pytest.raises(RuntimeError,match='incomplete'):
        evidence.finish(dispatch_identity={'seed':2})
    assert len(collected)==1
    evidence.close()
    assert not gl.buffers


@pytest.mark.parametrize('flags,state', [(0,0),(1,3),(3,0),(5,0),(9,0),(17,0),(33,0),(65,0),(127,0)])
def test_scalar_source_loss_never_authorizes_black(flags,state):
    from camera_designer.studio_camera_responses import scalar_source_state
    assert scalar_source_state(flags)==state
    source=(Path(__file__).resolve().parents[1]/'camera_designer/studio_camera_responses.py').read_text()
    assert 'uint availability=(flags&126u)!=0u?0u:' in source


def test_copy_opt_in_never_changes_original_dispatch_size():
    from camera_designer.studio_camera_intervals import plan_interval_dispatch
    with pytest.raises(MemoryError,match='original dispatch'):
        plan_interval_dispatch(width=128,rows=72,samples=1,path_bands=3,max_bounces=5,
            memory_budget_bytes=80*1024*1024,response_copies=True)
    source=(Path(__file__).resolve().parents[1]/'demo_pluck_gl.py').read_text(encoding='utf-8')
    assert 'self._rows_per_frame = self._light_field_information.prepare' not in source


def test_actual_sensor_mutators_invalidate_response_identity_without_gl():
    import ast
    import math
    import types
    import numpy as np
    from camera_designer.studio_sensor_evidence import StudioResponseEvidence
    text=(Path(__file__).resolve().parents[1]/'demo_pluck_gl.py').read_text(encoding='utf-8')
    tree=ast.parse(text)
    wanted={}
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef):
            methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
            if all(name in methods for name in ('set_camera','set_lens','set_scene_geometry','clear_field')):
                wanted=methods;break
    assert wanted
    namespace={'np':np,'math':math}
    for name in ('set_camera','set_lens','set_scene_geometry'):
        exec(compile(ast.Module(body=[wanted[name]],type_ignores=[]),'native_mutator','exec'),namespace)
    owner=StudioResponseEvidence(memory_budget_bytes=100000,reserved_bytes=0,
        scene_revision='s',camera_revision='c',consume=lambda *args:None)
    obj=types.SimpleNamespace(_light_field_information=owner,_response_initializing=False)
    namespace['set_camera'](obj,[0,0,1],[0,0,0],fov_deg=40,
        basis_right=[1,0,0],basis_up=[0,1,0],basis_fwd=[0,0,-1])
    assert owner._pending_identity=={'camera'}
    with pytest.raises(ValueError,match='rebound'):owner.prepare(width=1,rows=1,samples=1,path_bands=1,max_bounces=1)
    owner.rebind_identity(camera_revision='c2')
    namespace['set_lens'](obj,focus_dist=2.)
    assert owner._pending_identity=={'camera'}
    owner.rebind_identity(camera_revision='c3')
    namespace['set_scene_geometry'](obj,n_tris=1,n_bvh_nodes=1,bmin=[0,0,0],bmax=[1,1,1],n_spheres=0)
    assert owner._pending_identity=={'scene'}
    owner.rebind_identity(scene_revision='s2')
    assert not owner._pending_identity
    owner.note_field_reset();assert owner.identity['field_generation']==1
    clear_source=ast.get_source_segment(text,wanted['clear_field'])
    assert 'clear_scalar_visits(self._light_field_information.scalar_visit_buffer)' in clear_source
    assert 'self._light_field_information.note_field_reset()' in clear_source


def test_air_availability_copies_actual_query_support_and_disabled_controls():
    source=(Path(__file__).resolve().parents[1]/'camera_designer/studio_camera_responses.py').read_text()
    assert 'uint flags=studio_scalar_visit_flags(pos,i);' in source
    assert '(flags&1u)==0u || (flags&126u)!=0u' in source
    assert 'uVolAlpha==0.0 || (uAirDiffuseScatter==0.0 && uAirSpecularScatter==0.0)' in source
    assert 'sci_words[interval+44u]=sci_air_queries' in source
    observer=(Path(__file__).resolve().parents[1]/'camera_designer/studio_camera_intervals.py').read_text()
    assert 'floatBitsToUint(tp_max)' in observer
    assert 'hit< -1?6u:(medium==-2?5u:' in observer


def test_both_original_positive_residual_cutoffs_are_copied():
    observed=observe_camera_continuation(canonical(),enabled=True)
    assert 'floatBitsToUint(tp_max)' in observed
    assert 'floatBitsToUint(tp_after)' in observed
    assert 'sci_words[sci_row+32u]=7u' in observed
    assert 'sci_words[sci_row+43u]=2u' in observed
