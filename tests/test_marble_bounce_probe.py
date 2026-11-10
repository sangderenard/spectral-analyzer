import ast
from pathlib import Path
import numpy as np
import pytest
from camera_designer.marble_bounce_probe import (scatter_sources, summarize_probe, probe_shader,
    PROBE_DEPTH, PROBE_GROUPS, RAYS_PER_GROUP)


def data():
    return np.zeros((PROBE_GROUPS, RAYS_PER_GROUP, PROBE_DEPTH+1))


def test_fast_decay_uses_whole_tail():
    e = data()
    e[:, :, :1000] = .5 ** np.arange(1000)
    result = summarize_probe(e)
    assert result['chosen_depth'] == 8
    assert result['probe_depth'] == 1000 and result['render_cap'] == 200
    assert not result['warning']


def test_delayed_return_is_not_mistaken_for_convergence():
    e=data(); e[:, :, 0]=1; e[:, :, 500]=1
    result=summarize_probe(e)
    assert result['chosen_depth']==200
    assert result['estimated_required_depth']==501
    assert result['tail_fraction_beyond_render_cap']==.5
    assert 'beyond the render cap' in result['warning']


def test_mirror_room_censoring_is_visible():
    e=data(); e[:]=1
    result=summarize_probe(e)
    assert result['paths_at_probe_limit']==256
    assert result['chosen_depth']==200 and result['estimated_required_depth'] is None
    assert 'after 1000' in result['warning']


def test_one_rare_long_path_prevents_false_confidence():
    e=data(); e[:, :, :4]=1; e[0,0,:]=1
    result=summarize_probe(e)
    assert result['paths_at_probe_limit']==1
    assert result['chosen_depth']==200


def test_no_hits_does_not_claim_convergence():
    result=summarize_probe(data())
    assert result['chosen_depth']==200
    assert result['confidence']=='insufficient spatial coverage'


def test_four_batches_and_finite_input_required():
    with pytest.raises(ValueError): summarize_probe(data()[:3])
    e=data(); e[0,0,0]=float('nan')
    with pytest.raises(ValueError): summarize_probe(e)


def test_scatter_spatial_coverage_and_reproducibility():
    lo=np.array([-3,-2,-1]); hi=np.array([2,4,5])
    a=scatter_sources((lo,hi),[.450,.550,.650]); b=scatter_sources((lo,hi),[.450,.550,.650])
    np.testing.assert_array_equal(a,b)
    assert a.shape==(4,64,12)
    np.testing.assert_allclose(np.linalg.norm(a[:,:,4:7],axis=2),1,atol=1e-7)
    for group in a:
        cells=np.floor((group[:,:3]-lo)/(hi-lo)*4).astype(int)
        assert len(np.unique(cells,axis=0))==64
    assert np.all(a[:,:,10]==1)


def test_canonical_shader_instrumentation_keeps_bounce_physics():
    tree=ast.parse(Path('demo_pluck_gl.py').read_text(encoding='utf-8'))
    literal=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_GPU_RAY_FIELD_CS' for t in n.targets) and isinstance(n.value,ast.Constant))
    source=probe_shader(literal)
    assert 'probe_energy[ray_id * uint(uMaxBounces + 1) + uint(bounce)]' in source
    assert source.count('if (packet.z < 0.002) break;') == literal.count('if (packet.z < 0.002) break;')
    assert 'packet.z *= reflectivity * band_decay * dist_falloff;' in source
    assert 'packet.z *= (1.0 - fr) * medium_decay;' in source
    assert 'layout(std430, binding = 11)' in source
    assert source.count('if (bounce + 1 == uMaxBounces)') == 3
    assert source.count('currentMedium') == literal.count('currentMedium')
    assert 'studio_boundary_step(tri,rd,packet.x,rand01(rng),currentMedium,factor)' in source
    with pytest.raises(RuntimeError): probe_shader(literal.replace('if(studio_boundary(tri)) {', 'if(false) {'))

    forward_end = source.index('void reactive_main(uint gid, inout uint rng) {')
    endpoint = 'if (bounce + 1 == uMaxBounces) probe_energy'
    assert source[:forward_end].count(endpoint) == 3
    forward_start = source.index('void main() {')
    boundary = source[source.index('if(studio_boundary(tri)) {', forward_start):]
    boundary = boundary[:boundary.index('continue;')]
    assert 'studio_boundary_step(tri,rd,packet.x,rand01(rng),currentMedium,factor)' in boundary
    assert endpoint in boundary
    assert source[forward_end:] == literal[literal.index('void reactive_main(uint gid, inout uint rng) {'):]
    with pytest.raises(RuntimeError, match='continuation sites'):
        probe_shader(literal.replace('if (packet.z < 0.002) break;', 'if (packet.z < 0.002) { break; }', 1))
    with pytest.raises(RuntimeError): probe_shader(literal.replace('uniform int   uMaxBounces;', 'uniform int changed;'))

def test_host_uses_independent_depth_and_restores_scene_bindings(monkeypatch):
    from types import SimpleNamespace
    from OpenGL import GL
    from camera_designer.marble_bounce_probe import run_bounce_probe
    calls=[]; data_bytes={}; bound=[None]
    constants=['GL_COMPUTE_SHADER','GL_LINK_STATUS','GL_SHADER_STORAGE_BUFFER','GL_STATIC_DRAW','GL_DYNAMIC_READ','GL_SHADER_STORAGE_BARRIER_BIT']
    for i,key in enumerate(constants): monkeypatch.setattr(GL,key,i+1,raising=False)
    monkeypatch.setattr(GL,'glGetProgramiv',lambda *a: True,raising=False)
    monkeypatch.setattr(GL,'glGenBuffers',lambda n: [701,702],raising=False)
    monkeypatch.setattr(GL,'glGetUniformLocation',lambda program,name: name,raising=False)
    monkeypatch.setattr(GL,'glUniform1i',lambda key,value: calls.append(('uniform',key,value)),raising=False)
    monkeypatch.setattr(GL,'glUniform1f',lambda *a: None,raising=False)
    monkeypatch.setattr(GL,'glUseProgram',lambda *a: None,raising=False)
    monkeypatch.setattr(GL,'glBindBufferBase',lambda target,index,buffer: calls.append(('bind',index,buffer)),raising=False)
    monkeypatch.setattr(GL,'glBindBuffer',lambda target,buffer: bound.__setitem__(0,buffer),raising=False)
    monkeypatch.setattr(GL,'glBufferData',lambda target,size,data,usage: data_bytes.__setitem__(bound[0],data.copy()),raising=False)
    def dispatch(*args):
        values=data_bytes[702]; values[:,:20]=.5**np.arange(20)
        calls.append(('dispatch',))
    monkeypatch.setattr(GL,'glDispatchCompute',dispatch,raising=False)
    monkeypatch.setattr(GL,'glMemoryBarrier',lambda *a: None,raising=False)
    monkeypatch.setattr(GL,'glFinish',lambda: None,raising=False)
    monkeypatch.setattr(GL,'glGetBufferSubData',lambda *args: data_bytes[702].tobytes(),raising=False)
    monkeypatch.setattr(GL,'glDeleteBuffers',lambda *a: calls.append(('destroy_buffers',)),raising=False)
    monkeypatch.setattr(GL,'glDeleteProgram',lambda *a: calls.append(('destroy_program',)),raising=False)
    tree=ast.parse(Path('demo_pluck_gl.py').read_text(encoding='utf-8'))
    source=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_GPU_RAY_FIELD_CS' for t in n.targets) and isinstance(n.value,ast.Constant))
    renderer=SimpleNamespace(_GPU_RAY_FIELD_CS=source,_prog=lambda args:51)
    state={'ssbo':list(range(101,112))}
    before=state.copy()
    result=run_bounce_probe(renderer,state,([-1,-1,-1],[1,1,1]),[.450,.550,.650],n_triangles=12,n_nodes=3)
    assert result['chosen_depth']==8 and state==before
    assert ('uniform',b'uMaxBounces',1000) in calls
    assert calls.count(('dispatch',))==4
    assert calls[-3:]==[('bind',11,0),('destroy_buffers',),('destroy_program',)]
    assert ('bind',5,106) in calls[-14:]
