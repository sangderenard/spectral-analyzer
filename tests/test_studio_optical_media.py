import copy
import math
import numpy as np
import pytest
from camera_designer.studio_optical_media import validate_optical_media, optical_extinction
from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene


def fixture():
    pbr=np.zeros((1,16),np.float32);pbr[0,:3]=[.1,.7,.3];pbr[0,5]=1;pbr[0,6]=1.54
    value={'version':1,'materials':[{'material':0,'wavelengths_nm':[400.,550.,700.],
       'transmittance':[.1,.7,.2],'reference_thickness_mm':1.,'concentration':1.}]}
    return value,pbr


def test_reference_thickness_concentration_and_clear_limit():
    value,pbr=fixture();row=validate_optical_media(value,pbr,[0])[0]
    for nm,t in zip(row['wavelengths_nm'],row['transmittance']):
        k=optical_extinction(row,nm/1000);sigma=4*math.pi*k/(nm*1e-9)
        assert math.exp(-sigma*.001)==pytest.approx(t)
        assert math.exp(-sigma*.002)==pytest.approx(t*t)
    double=copy.deepcopy(row);double['concentration']=2
    assert optical_extinction(double,.55)==pytest.approx(2*optical_extinction(row,.55))
    double['concentration']=0
    assert optical_extinction(double,.55)==0
    double['transmittance']=[1,1,1];double['concentration']=1
    assert optical_extinction(double,.55)==0


def test_sigma_interpolation_no_silent_extrapolation_or_roughness_coupling():
    v,p=fixture();r=validate_optical_media(v,p,[0])[0]
    p[0,3]=1
    assert validate_optical_media(v,p,[0])=={0:r}
    with pytest.raises(ValueError,match='outside'):optical_extinction(r,.8)
    assert optical_extinction(r,.8,outside_policy='game-transparent-outside-authored-v1')==0
    assert optical_extinction(r,.365,outside_policy='game-transparent-outside-authored-v1')==0
    sigma=lambda nm:optical_extinction(r,nm/1000)*4*math.pi/(nm*1e-9)
    assert sigma(475)==pytest.approx((sigma(400)+sigma(550))/2)


@pytest.mark.parametrize('key,value',[('reference_thickness_mm',0),('reference_thickness_mm',float('nan')),
 ('concentration',-1),('concentration',True),('reference_thickness_mm',1e-320),('concentration',1e308),('transmittance',[0,.7,.2]),
 ('transmittance',[1.1,.7,.2]),('wavelengths_nm',[550,400,700]),('material',True)])
def test_bad_authoring_rejected(key,value):
    v,p=fixture();v['materials'][0][key]=value
    with pytest.raises(ValueError):validate_optical_media(v,p,[0])


def test_exact_coverage_and_dielectric_contract():
    v,p=fixture()
    with pytest.raises(ValueError):validate_optical_media(v,p,[])
    p[0,7]=.1
    with pytest.raises(ValueError,match='opacity zero'):validate_optical_media(v,p,[0])


def test_native_matbuf_carries_authored_extinction_and_cache_identity():
    v,p=fixture()
    empty=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),
       np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},
       (np.full(3,-1,np.float32),np.full(3,1,np.float32)),np.zeros((0,12),np.float32))
    vertices=np.array([[[-1,-1,0],[1,-1,0],[0,1,0]]],np.float32)
    boundary={'version':1,'ranges':[{'start':0,'count':1,'positive':-1,'negative':0}]}
    def build(value):
        return append_saved_studio_mesh_to_gpu_scene(empty,vertices,np.array([0]),p,
          scale_m=.01,center_m=[0,0,0],wavelengths_um=[.55],boundary_media=boundary,optical_media=value)
    a=build(v);idx=int(a[0][0,7])-2
    k=float(a[2][idx*32,8]);sigma=4*math.pi*k/550e-9
    assert math.exp(-sigma*.001)==pytest.approx(.7,rel=1e-6)
    v2=copy.deepcopy(v);v2['materials'][0]['concentration']=2
    b=build(v2);idx2=int(b[0][0,7])-2
    assert idx2!=idx
    assert b[2][idx2*32,8]==pytest.approx(2*k,rel=1e-6)
    with pytest.raises(ValueError,match='ownership'):
        append_saved_studio_mesh_to_gpu_scene(empty,vertices,np.array([0]),p,
          scale_m=.01,center_m=[0,0,0],optical_media=v)


def test_unmeasured_game_bands_have_no_optical_interface():
    v,p=fixture()
    empty=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),
       np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},
       (np.full(3,-1,np.float32),np.full(3,1,np.float32)),np.zeros((0,12),np.float32))
    vertices=np.array([[[-1,-1,0],[1,-1,0],[0,1,0]]],np.float32)
    boundary={'version':1,'ranges':[{'start':0,'count':1,'positive':-1,'negative':0}]}
    wavelengths=[.365,.55,.85]
    with pytest.raises(ValueError,match='outside'):
        append_saved_studio_mesh_to_gpu_scene(empty,vertices,[0],p,scale_m=.01,
            center_m=[0,0,0],wavelengths_um=wavelengths,boundary_media=boundary,optical_media=v)
    scene=append_saved_studio_mesh_to_gpu_scene(empty,vertices,[0],p,scale_m=.01,
        center_m=[0,0,0],wavelengths_um=wavelengths,boundary_media=boundary,optical_media=v,
        optical_outside_policy='game-transparent-outside-authored-v1')
    idx=int(scene[0][0,7])-2
    bands=scene[2].reshape(-1,32,12)[idx,:3]
    for band in (bands[0],bands[2]):
        np.testing.assert_allclose(band[[2,3,4,7,8]],[0,1,0,1,0],atol=1e-7)
    assert bands[1,7]>1 and bands[1,8]>0


def test_strong_extinction_requires_complex_interface_solver():
    v,p=fixture();v['materials'][0]['reference_thickness_mm']=.0001
    v['materials'][0]['transmittance']=[.01,.01,.01]
    with pytest.raises(ValueError,match='complex-index'):validate_optical_media(v,p,[0])
