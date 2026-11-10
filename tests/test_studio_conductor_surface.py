import numpy as np
import pytest
from camera_designer.studio_conductor_surface import conductor_fresnel,sample_ggx

def test_complex_fresnel_matches_normal_incidence_and_limits():
    for n,k in [(.2,3),(1.5,0),(.8,5),(2,1)]:
        assert conductor_fresnel(n,k,1)==pytest.approx(((n-1)**2+k*k)/((n+1)**2+k*k))
        values=conductor_fresnel(n,k,np.linspace(0,1,101))
        assert np.all((values>=0)&(values<=1));assert values[0]==1

def test_ggx_honors_roughness_and_white_furnace_bound():
    q=(np.arange(256)+.5)/256
    u=np.stack(np.meshgrid(q,q),axis=-1).reshape(-1,2)
    for theta in [0,.5,1.2]:
        wi=[np.sin(theta),0,np.cos(theta)]
        for roughness in [0,.1,.4,.8,1]:
            wo,w,c=sample_ggx(wi,roughness,u)
            assert np.isfinite(wo).all() and np.isfinite(w).all()
            assert np.all(w>=0);assert w.mean()<=1.002
            if roughness==0:assert np.all(w==1)
            else:assert np.std(wo[:,0])>0

def test_metal_colour_changes_weights_not_sample_directions():
    u=np.random.default_rng(42).random((1000,2));wi=[0,0,1]
    wo,w,c=sample_ggx(wi,.4,u)
    # Synthetic n/k fixtures, not measured silver/gold presets.
    a=w*conductor_fresnel(.2,3,c);b=w*conductor_fresnel(1.2,1,c)
    assert a.mean()>b.mean();assert np.all(a<=w) and np.all(b<=w)
    np.testing.assert_array_equal(wo,sample_ggx(wi,.4,u)[0])

def test_advanced_surface_is_per_triangle_and_native_roughness_is_required():
    from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
    from camera_designer.studio_conductor_surface import checked_native_pbr
    from material_db import MaterialDatabase
    from mat_flags import MAT_FLAG_ROUGH_CONDUCTOR
    p=np.zeros((1,16),np.float32);p[0,:3]=[1,.72,.29];p[0,3]=.4;p[0,4]=1;p[0,6]=1.5;p[0,7]=1
    empty=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.zeros(3),np.ones(3)),np.zeros((0,12),np.float32))
    v=np.array([[[0,0,0],[1,0,0],[0,1,0]],[[0,0,1],[1,0,1],[0,1,1]]],np.float32)
    out=append_saved_studio_mesh_to_gpu_scene(empty,v,[0,0],p,scale_m=1,center_m=[0,0,0],wavelengths_um=[.65,.55,.45],advanced_surface_mask=[True,False])
    flags=out[0][:,15].view(np.uint32)
    assert flags[0]&MAT_FLAG_ROUGH_CONDUCTOR and not flags[1]&MAT_FLAG_ROUGH_CONDUCTOR
    with pytest.raises(ValueError,match='canonical native PBR'):checked_native_pbr(None,out[0],out[1],len(out[2])//32)
    native=MaterialDatabase.instance().build_tensors()['pbr']
    checked_native_pbr(native,out[0],out[1],len(out[2])//32)
    mid=out[1][0,14:15].view(np.uint32)[0];assert native[mid,3]==pytest.approx(.4)
    assert p[0,3]==pytest.approx(.4)

@pytest.mark.parametrize("extinction",[.01,3.0])
def test_canonical_conductor_preserves_native_nk_and_roughness(extinction):
    from canonical_material_documents import MaterialDocument
    from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
    from material_db import MaterialDatabase
    from mat_flags import MAT_FLAG_ROUGH_CONDUCTOR
    source='''metallic: 1
roughness: 0.4
opacity: 1
transmission: 0
ior: 0.2
spectral_bands:
- center_hz: 499654096666666.7
  bandwidth_hz: 100000000000000
  reflectance: 0.2
  transmittance: 0
  ior_real: 0.2
  ior_imag: 3
'''
    source=source.replace('ior_imag: 3',f'ior_imag: {extinction}')
    doc=MaterialDocument('synthetic_conductor_test','test',source.encode())
    original=doc.snapshot()
    authoring={'version':1,'records':[{'material':0,'source':doc.snapshot(),'overrides':[]}]}
    empty=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.zeros(3),np.ones(3)),np.zeros((0,12),np.float32))
    v=np.array([[[0,0,0],[1,0,0],[0,1,0]]],np.float32);p=np.zeros((1,16),np.float32);p[:,6]=1.5;p[:,7]=1
    out=append_saved_studio_mesh_to_gpu_scene(empty,v,[0],p,scale_m=1,center_m=[0,0,0],wavelengths_um=[.6],material_authoring=authoring,advanced_surface_mask=[True])
    mid=out[1][0,14:15].view(np.uint32)[0];native=MaterialDatabase.instance().build_tensors()
    assert out[0][0,15:16].view(np.uint32)[0]&MAT_FLAG_ROUGH_CONDUCTOR
    assert native['pbr'][mid,3]==pytest.approx(.4)
    assert out[2].reshape(-1,32,12)[mid,0,7]==pytest.approx(.2)
    assert out[2].reshape(-1,32,12)[mid,0,8]==pytest.approx(extinction)
    from mat_flags import MAT_FLAG_NATIVE_CONDUCTOR_NK
    assert out[0][0,15:16].view(np.uint32)[0]&MAT_FLAG_NATIVE_CONDUCTOR_NK
    assert doc.snapshot()==original


def test_inclusive_rng_endpoint_is_finite_null_or_valid_event():
    for roughness in [.001,.4,1]:
        wo,w,vh=sample_ggx([0,0,1],roughness,[[0,0],[1,0],[1,1]])
        assert np.isfinite(wo).all() and np.isfinite(w).all() and np.isfinite(vh).all()
        assert np.all(w>=0)

def test_release_gate_and_engine_ranges_are_explicit(monkeypatch):
    import camera_designer.studio_scene_engines as engines
    fixtures={'material_ids':[0,0,0,0],'material_engines':{'groups':[
        {'name':'stage','engine':'dielectric','ranges':[{'start':0,'count':2}]},
        {'name':'room','engine':'legacy','ranges':[{'start':2,'count':2}]}]}}
    monkeypatch.setattr(engines,'ADVANCED_CONDUCTOR_RUNTIME_ENABLED',False)
    assert not engines.advanced_surface_triangles(fixtures).any()
    monkeypatch.setattr(engines,'ADVANCED_CONDUCTOR_RUNTIME_ENABLED',True)
    assert engines.advanced_surface_triangles(fixtures).tolist()==[True,True,False,False]
