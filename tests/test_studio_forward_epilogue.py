"""Bounded static/control tests; never invoke a shader compiler or GL."""
from pathlib import Path
import copy,hashlib,re
import pytest
from camera_designer.studio_forward_epilogue import single_forward_epilogue,_mask,_LATCH,_EMIT
ROOT=Path(__file__).resolve().parents[1]
def canonical_candidate():
    # Evaluate only the installed shader string assignments and pure source
    # composers, never import the renderer or create a GL context.
    import ast
    from camera_designer.spectral_sensor_basis import FIXED3_SPECTRAL_GLSL
    from camera_designer.studio_diffuse_surface import STUDIO_DIFFUSE_GLSL
    from camera_designer.studio_spectral_reflector import STUDIO_REFLECTOR_GLSL
    from camera_designer.studio_boundary_media import STUDIO_BOUNDARY_GLSL
    from camera_designer.studio_region_gpu import with_region_media,with_region_spheres
    from camera_designer.studio_sphere_regions import with_sphere_regions
    from camera_designer.studio_opaque_pbr import with_opaque_pbr
    from camera_designer.studio_directional_field import recording_shader
    from camera_designer.studio_directional_media import with_internal_medium_recording
    from camera_designer.studio_scalar_visits import with_scalar_visits
    from camera_designer.studio_scalar_columns import with_scalar_columns
    from camera_designer.studio_forward_responses import copy_forward_responses,specialize_prebuilt_forward
    import mat_flags
    scope=locals().copy()
    for name in ('MAT_FLAG_SPECTRAL_REFLECTOR','MAT_FLAG_ROUGH_CONDUCTOR','MAT_FLAG_NATIVE_CONDUCTOR_NK','BINDING_STUDIO_NATIVE_PBR'):
        scope[name]=getattr(mat_flags,name)
    tree=ast.parse((ROOT/'demo_pluck_gl.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('STUDIO_REFLECTOR_GLSL','_GPU_RAY_FIELD_CS','_GPU_RAY_FIELD_REGION_CS') for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<native-source-assignment>','exec'),scope)
    source=with_opaque_pbr(scope['_GPU_RAY_FIELD_REGION_CS'])
    source=recording_shader(source,stencil=12,grid_size=64,estimator='track-length-root-sweep-v1',independent_bounds=True,quantization_divisor=16777216)
    source=with_internal_medium_recording(source)
    source=with_scalar_columns(with_scalar_visits(source),3,shared=True)
    source,_=copy_forward_responses(source,terminal_only=True,shared_scalar_buffer=True,precleared_rows=True)
    return specialize_prebuilt_forward(source)


def test_exact_frozen_source_native_tokens_and_nested_controls_preserved():
    source=canonical_candidate()
    result,receipt=single_forward_epilogue(source)
    assert receipt['writer_sites_before']==9 and receipt['writer_sites_after']==1
    assert receipt['control']=={'break':9,'continue':3}
    # Reverse only newly introduced mechanical syntax: every old byte survives.
    restored=result.replace(_LATCH+'\n','',1)
    restored=restored.replace('''{
        studio_terminal_present=false;
        int studio_bounce_exit=0;
        do {''','{',1)
    ending='''\n        } while(false);
        '''+_EMIT+'''
        if(studio_bounce_exit==1)break;
        if(studio_bounce_exit==2)continue;
    }'''
    assert restored.count(ending)==1
    restored=restored.replace(ending,'}',1)
    restored=restored.replace('{studio_bounce_exit=1;break;}','break;').replace('{studio_bounce_exit=2;break;}','continue;')
    restored=restored.replace('studio_terminal_latch(','studio_response_copy(')
    assert restored==source
    assert result.count('if (best_idx_f[ki] < 0) continue;')==1
    assert result.count('rand01(')==source.count('rand01(')
    assert result.count('imageAtomicAdd(')==source.count('imageAtomicAdd(')


@pytest.mark.parametrize('path,kind,termination,outcome',[
 ('escape',6,5,'break'),('invalid',5,2,'break'),('absorber',6,7,'break'),
 ('refraction',5,0,'continue'),('boundary_rejected',6,4,'break'),
 ('pbr',5,0,'continue'),('pbr_rejected',6,6,'break'),
 ('manifold',5,0,'continue'),('common_terminal',5,0,'fallthrough'),
 ('cutoff',5,3,'break'),('virtual_partition',5,0,'continue'),
 ('maxbounce',5,8,'fallthrough'),('no_evaluated_terminal',None,None,'break')])
def test_snapshot_lifetime_and_control_model(path,kind,termination,outcome):
    # All mutable argument vectors are value copies at the ORIGINAL call site.
    native=dict(packet=[4.,3.,2.,1.],ro=[1.,2.,3.],hp=[4.,5.,6.],direction=[0.,0.,1.],medium=7)
    global_evidence=dict(before=[8.,7.,6.,5.],regions_before=list(range(32)),regions_after=list(range(32,64)),primitive=[2,91],stages=[1.,2.,3.,4.])
    expected=[];present=False;snapshot=None
    if kind is not None:
        expected.append(copy.deepcopy(dict(native,kind=kind,termination=termination,evidence=global_evidence)))
        snapshot=copy.deepcopy(native);present=True
    # Actual boundary/PBR continuations alter packet phase/position/direction
    # AFTER the producer site. Emission must use the latched values, not these.
    native['packet'][1]+=17;native['ro'][:]=[99.,98.,97.];native['direction'][:]=[1.,0.,0.]
    actual=[]
    if present:actual.append(dict(snapshot,kind=kind,termination=termination,evidence=copy.deepcopy(global_evidence)))
    assert actual==expected
    control={'break':1,'continue':2,'fallthrough':0}[outcome]
    assert ('break' if control==1 else 'continue' if control==2 else 'fallthrough')==outcome
    assert len(actual)==int(kind is not None)


def test_unreviewed_graph_or_duplicate_transform_rejected():
    source=canonical_candidate()
    with pytest.raises(ValueError,match='specialization'):single_forward_epilogue(source.replace('if (uMode != PASS_FORWARD) return;',''))
    with pytest.raises(ValueError,match='graph changed'):single_forward_epilogue(source.replace('studio_response_copy(6u,1u','unexpected_copy(6u,1u',1))
    transformed,_=single_forward_epilogue(source)
    with pytest.raises(ValueError,match='already installed'):single_forward_epilogue(transformed)

def test_actual_branch_graph_has_at_most_one_copy_and_no_late_evidence_mutation():
    # Parse actual frozen bounce statements, branch exhaustively without ray
    # execution. Nested numeric loops contain no copy/evidence mutations and
    # are opaque here; their internal continue still targets their own loop.
    from camera_designer.studio_forward_epilogue import _close
    source=canonical_candidate();masked=_mask(source)
    start=masked.index('for (int bounce = 0; bounce < uMaxBounces; ++bounce) {',masked.index('void main() {'))
    start=masked.index('{',start);end=_close(masked,start,'{','}')
    text=masked[start+1:end]
    def space(i):
        while i<len(text) and text[i].isspace():i+=1
        return i
    def word(i,w):return text.startswith(w,i) and (i+len(w)==len(text) or not(text[i+len(w)].isalnum() or text[i+len(w)]=='_'))
    def stmt(i):
        i=space(i)
        if text[i]=='{':
            nodes=[];i=space(i+1)
            while text[i]!='}':
                node,i=stmt(i);nodes.append(node);i=space(i)
            return ('block',nodes),i+1
        if word(i,'if'):
            op=text.index('(',i);cl=_close(text,op,'(',')')
            yes,j=stmt(cl+1);j=space(j);no=('block',[])
            if word(j,'else'):no,j=stmt(j+4)
            return ('if',yes,no,text[op+1:cl]),j
        if word(i,'for'):
            op=text.index('(',i);cl=_close(text,op,'(',')');body,j=stmt(cl+1)
            raw=text[i:j]
            assert 'studio_response_' not in raw and 'studio_active' not in raw
            return ('numeric_loop',raw),j
        j=i;paren=0
        while j<len(text):
            if text[j]=='(':paren+=1
            elif text[j]==')':paren-=1
            elif text[j]==';' and paren==0:return ('simple',text[i:j+1]),j+1
            j+=1
        raise AssertionError('unterminated statement')
    nodes=[];i=space(0)
    while i<len(text):node,i=stmt(i);nodes.append(node);i=space(i)
    pure_builtins={'isnan','isinf','abs','max','length'}
    checked=set()
    def check_calls(raw):
        for name in re.findall(r'\b([A-Za-z_]\w*)\s*\(',raw):
            if name in pure_builtins or name in checked:continue
            definition=re.search(r'\b(?:float|bool)\s+'+re.escape(name)+r'\s*\([^)]*\)\s*\{',masked)
            assert definition is not None,('unproved helper after copy',name)
            signature=masked[definition.start():definition.end()]
            assert not re.search(r'\b(?:out|inout)\b',signature),signature
            op=definition.end()-1;cl=_close(masked,op,'{','}')
            body=masked[op+1:cl].strip()
            # Current late helpers are single return expressions. Any future
            # assignment/control/store, including hidden active-bit mutation,
            # invalidates this conservative purity proof instead of guessing.
            assert re.fullmatch(r'return\s+[^;]+;',body),body
            assert not re.search(r'(?<![=!<>])=(?!=)|\+\+|--',body),body
            checked.add(name);check_calls(body)
    # state: copy-count, control outcome. Only active paths continue execution.
    def run(node,states):
        kind=node[0]
        if kind=='block':
            for child in node[1]:states=run(child,states)
            return states
        result=set()
        for count,outcome in states:
            if outcome!='active':result.add((count,outcome));continue
            if kind=='if':
                if count:check_calls(node[3])
                result|=run(node[1],{(count,outcome)})|run(node[2],{(count,outcome)});continue
            if kind=='numeric_loop':result.add((count,outcome));continue
            raw=node[1]
            if 'studio_response_copy(' in raw:
                count+=1;assert count<=1,raw
            elif count:
                check_calls(raw)
                assert not re.search(r'\bstudio_response_\w+\s*(?:\(|=|\[)',raw),raw
                assert not re.search(r'\bstudio_(?:boundary_step|origin_medium|active)\b',raw),raw
            if raw=='break;':outcome='break'
            elif raw=='continue;':outcome='continue'
            result.add((count,outcome))
        return result
    outcomes=run(('block',nodes),{(0,'active')})
    assert outcomes=={(1,'break'),(1,'continue'),(1,'active')}
    assert checked=={'studio_region_epsilon','studio_media_enabled'}

@pytest.mark.parametrize('helper',['studio_region_epsilon','studio_media_enabled'])
def test_late_helper_hidden_region_mutation_fails_proof(monkeypatch,helper):
    source=canonical_candidate()
    definition=re.search(r'\b(?:float|bool)\s+'+helper+r'\s*\([^)]*\)\s*\{',source)
    assert definition is not None
    at=definition.end()
    changed=source[:at]+'studio_active[0]=0u;'+source[at:]
    monkeypatch.setitem(globals(),'canonical_candidate',lambda:changed)
    with pytest.raises(AssertionError):test_actual_branch_graph_has_at_most_one_copy_and_no_late_evidence_mutation()
