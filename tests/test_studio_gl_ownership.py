"""Mock-only ownership checks: no GL context or shader execution."""
import ast
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def functions(*names):
    tree=ast.parse((ROOT/'demo_pluck_gl.py').read_text(encoding='utf-8'))
    return compile(ast.fix_missing_locations(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])),'<mock-GL-ownership>','exec')

def test_compile_exception_releases_created_shader():
    calls=[]
    def fail(*args):raise RuntimeError('original')
    scope=dict(glCreateShader=lambda _:17,glShaderSource=fail,glDeleteShader=lambda x:calls.append(x))
    exec(functions('_compile'),scope)
    with pytest.raises(RuntimeError,match='original'):scope['_compile']('source',1)
    assert calls==[17]

@pytest.mark.parametrize('failure',['compile','attach','link','status','log',None])
def test_program_ownership_on_every_failure_and_success(failure):
    calls=[]
    def op(name,value=None):
        def call(*args):
            calls.append((name,args))
            if name==failure:raise RuntimeError(name)
            return value
        return call
    scope=dict(glCreateProgram=lambda:19,_compile=op('compile',23),glAttachShader=op('attach'),glLinkProgram=op('link'),glGetProgramiv=op('status',0),glGetProgramInfoLog=op('log',b''),glDeleteProgram=op('delete_program'),glDeleteShader=op('delete_shader'),glDetachShader=op('detach'),GL_LINK_STATUS=1)
    exec(functions('_prog'),scope)
    if failure:
        with pytest.raises(RuntimeError,match=failure):scope['_prog'](('source',2))
    else:assert scope['_prog'](('source',2))==19
    assert sum(n=='delete_program' for n,_ in calls)==bool(failure)
    assert sum(n=='delete_shader' for n,_ in calls)==(failure!='compile')
    assert sum(n=='detach' for n,_ in calls)==(failure is None)

def test_optin_execution_hold_precedes_context_and_preserves_request():
    source=(ROOT/'camera_designer/marble_preview_worker.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    hold=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='response_requested' and n.body and isinstance(n.body[0],ast.Raise))
    assert hold.lineno<next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_ensure_worker_gl_context')
    code=compile(ast.fix_missing_locations(ast.Module(body=[hold],type_ignores=[])),'<execution-hold>','exec')
    exec(code,{'response_requested':False})
    with pytest.raises(RuntimeError,match='hardware compile resource validation pending'):exec(code,{'response_requested':True})

def test_prebuilt_auxiliary_allocation_and_compile_are_cleanup_protected():
    tree=ast.parse((ROOT/'demo_pluck_gl.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gpu_ray_field_prebuilt')
    protected=next(n for n in fn.body if isinstance(n,ast.Try) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='allocate_scalar_visits' for x in ast.walk(n)))
    assert any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='_prog' for x in ast.walk(protected))
    handler=protected.handlers[0]
    assert isinstance(handler.type,ast.Name) and handler.type.id=='BaseException'
    assert isinstance(handler.body[-1],ast.Raise)
    assert any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='release_failed_prebuilt' for x in ast.walk(handler))

def test_prebuilt_entry_specialization_preserves_forward_body_and_rng():
    import runpy
    from camera_designer.studio_forward_responses import specialize_prebuilt_forward,require_prebuilt_mode,PREBUILT_ENTRY_CONTRACT
    original=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    specialized=specialize_prebuilt_forward(original)
    before=original.index('    if (uMode == PASS_SENSOR) {')
    after=original.index('    }',original.index('        reactive_main(gid, rng);'))+5
    assert specialized[:before]==original[:before]
    marker='    if (uMode != PASS_FORWARD) return; // owner rejects unsupported mode'
    assert specialized[specialized.index(marker)+len(marker):]==original[after:]
    assert specialized.count('rand01(')==original.count('rand01(')
    assert specialized.count('imageAtomicAdd(')==original.count('imageAtomicAdd(')
    require_prebuilt_mode({'response_entry_contract':PREBUILT_ENTRY_CONTRACT},0)
    with pytest.raises(ValueError,match='PASS_FORWARD only'):require_prebuilt_mode({'response_entry_contract':PREBUILT_ENTRY_CONTRACT},2)
    require_prebuilt_mode({},2)
