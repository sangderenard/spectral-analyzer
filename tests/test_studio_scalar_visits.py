import runpy
from pathlib import Path
from camera_designer.studio_scalar_visits import with_scalar_visits,QUERY_GLSL

def test_original_scalar_assignments_unchanged_and_prequantization_visit():
    source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    output=with_scalar_visits(source)
    for old in ['uint val = uint(clamp(energy * coherence * 65535.0, 0.0, 1e6));','if (val > 0u) layer_splat(studio_band_index(freq), q, val);','if (val == 0u) return;']:
        assert source.count(old)==output.count(old)==1
    assert output.index('studio_scalar_visit(q,studio_band_index(freq),energy*coherence*65535.0);')<output.index('if (val > 0u) layer_splat(studio_band_index(freq), q, val);')
    assert 'StudioScalarVisits' not in source
    assert 'ivec3 q=ivec3(floor(uvw*vec3(uDims)))' in QUERY_GLSL

def test_actual_prebuilt_optin_block_allocates_only_scalar_evidence(monkeypatch):
    import ast
    import camera_designer.studio_scalar_visits as module
    tree=ast.parse((Path(__file__).resolve().parents[1]/'demo_pluck_gl.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gpu_ray_field_prebuilt')
    at=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='scalar_visit_buffer' for t in n.targets))
    optin=next(n for n in ast.walk(function) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=="light_field_information" and any(isinstance(x,ast.ImportFrom) and x.module=="camera_designer.studio_scalar_visits" for x in n.body))
    block=ast.Module(body=[function.body[at],optin],type_ignores=[])
    source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    import OpenGL.GL as gl
    monkeypatch.setattr(gl,'GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS',99,raising=False)
    calls=[];monkeypatch.setattr(module,'allocate',lambda dims,**kw:calls.append(tuple(dims)) or 71)
    for opt in (None,'evaluated-responses-v1'):
        state=dict(light_field_information=opt,shader_source=source,dims=(96,96,96),n_layers=3,dispatch_batch=512,max_bounces=6,np=__import__('numpy'),glGetIntegerv=lambda _:100)
        exec(compile(ast.fix_missing_locations(block),'<actual-prebuilt-optin>','exec'),state)
        if opt is None:
            assert calls==[] and state['scalar_visit_buffer'] is None and state['shader_source']==source
        else:
            assert calls==[(96,96,96)] and state['scalar_visit_buffer']==71
            assert state['shader_source'].count('rand01(')==source.count('rand01(')
            assert state['shader_source'].count('imageAtomicAdd(')==source.count('imageAtomicAdd(')
    actual=ast.get_source_segment((Path(__file__).resolve().parents[1]/'demo_pluck_gl.py').read_text(encoding='utf-8'),function)
    assert 'allocate_observations' not in actual and 'capture_scene_contract' not in actual


def test_exact_scalar_availability_buffer_size_and_clear(monkeypatch):
    import sys,types
    from camera_designer import studio_scalar_visits as module
    calls=[]
    gl=types.SimpleNamespace(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS=1,GL_MAX_SHADER_STORAGE_BLOCK_SIZE=8,GL_SHADER_STORAGE_BUFFER=2,GL_DYNAMIC_DRAW=3,GL_R32UI=4,GL_RED_INTEGER=5,GL_UNSIGNED_INT=6,GL_BUFFER_SIZE=7,glGetBufferParameteriv=lambda *a:7077888,glDeleteBuffers=lambda *a:calls.append(('delete',a)),
        glGetIntegerv=lambda key:128*1024*1024 if key==8 else 32,glGenBuffers=lambda _:73,
        glBindBuffer=lambda *a:calls.append(('bind',a)),glBufferData=lambda *a:calls.append(('allocate',a[1])),
        glBindBufferBase=lambda *a:calls.append(('binding',a)),glClearBufferSubData=lambda *a:calls.append(('clear_view',a[2],a[3])),glClearBufferData=lambda *a:calls.append(('clear',int(a[-1][0]))))
    monkeypatch.setitem(sys.modules,'OpenGL',types.SimpleNamespace(GL=gl))
    assert module.allocate((96,96,96))==73
    assert ('allocate',7077888) in calls and ('clear',0) in calls
    assert ('binding',(2,23,73)) in calls


def test_actual_uint_atomic_overflow_boundary_cases():
    from camera_designer.studio_scalar_visits import scalar_add_overflow
    assert not scalar_add_overflow(0,0)
    assert not scalar_add_overflow(0xffffffff,0)
    assert not scalar_add_overflow(0xfffffffe,1)
    assert not scalar_add_overflow(0,0xffffffff)
    assert scalar_add_overflow(0xffffffff,1)
    assert scalar_add_overflow(0xfffffffe,2)
    assert scalar_add_overflow(0xfffffff0,1000000)
    source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    observed=with_scalar_visits(source)
    assert source.count('imageAtomicAdd(')==observed.count('imageAtomicAdd(')==8
    for band in range(8):
        assert observed.count(f'uint studio_prior=imageAtomicAdd(uLayer{band}, q, val);')==1
        assert observed.count(f'studio_scalar_add_result(q,{band},studio_prior,val);')==1
    assert 'prior>0xffffffffu-value' in observed
    assert '(bit<<24)' in QUERY_GLSL and '?16u:0u' in QUERY_GLSL


def test_optin_sampling_barrier_covers_texels_and_availability(monkeypatch):
    import sys,types
    from camera_designer import studio_scalar_visits as module
    calls=[]
    gl=types.SimpleNamespace(GL_SHADER_STORAGE_BARRIER_BIT=1,GL_SHADER_IMAGE_ACCESS_BARRIER_BIT=2,GL_TEXTURE_FETCH_BARRIER_BIT=4,GL_SHADER_STORAGE_BUFFER=17,
        glMemoryBarrier=lambda bits:calls.append(('barrier',bits)),glBindBufferBase=lambda *args:calls.append(('bind',args)))
    monkeypatch.setitem(sys.modules,'OpenGL',types.SimpleNamespace(GL=gl))
    module.prepare_sampling(73)
    assert calls==[('barrier',7),('bind',(17,23,73))]


def test_shared_prefix_and_matrix_clear_isolation(monkeypatch):
    import sys,types,numpy as np
    from camera_designer import studio_scalar_visits as visits
    from camera_designer.studio_scalar_columns import clear_view
    data=bytearray([9])*80;calls=[]
    def clear_sub(target,internal,offset,size,format,type_,value):
        calls.append((offset,size));data[offset:offset+size]=b'\0'*size
    gl=types.SimpleNamespace(GL_SHADER_STORAGE_BUFFER=1,GL_R32UI=2,GL_RED_INTEGER=3,GL_UNSIGNED_INT=4,
        glBindBuffer=lambda *a:None,glClearBufferSubData=clear_sub)
    monkeypatch.setitem(sys.modules,'OpenGL',types.SimpleNamespace(GL=gl))
    visits._VISIT_BYTES[991]=16
    try:
        visits.clear(991)
        assert data[:16]==b'\0'*16 and data[16:]==bytes([9])*64
        clear_view({'buffer':991,'offset_bytes':16,'metadata':{'byte_length':32}})
        assert data[:48]==b'\0'*48 and data[48:]==bytes([9])*32
        assert calls==[(0,16),(16,32)]
    finally:visits._VISIT_BYTES.pop(991,None)
