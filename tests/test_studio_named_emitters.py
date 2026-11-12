import copy
import numpy as np
import pytest
from camera_designer.studio_named_emitters import KEYS,named_emitter,validate_named_emitter_refs,transport_band_power
from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene


def ref(key,gain=1,material=0):
    d,_=named_emitter(key)
    return dict(material=material,profile=key,revision=d['revision'],gain=gain,filter=[1,1,1])


def scene(reference,wavelengths=None):
    wavelengths=wavelengths or ([.65,.55,.45,.59,.365,.395,.85,.94] if reference['profile']=='studio_calibration_nir850' else [.45,.55,.65])
    empty=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.zeros(3),np.ones(3)),np.zeros((0,12),np.float32))
    v=np.array([[[0,0,0],[.1,0,0],[0,.1,0]]],np.float32);p=np.zeros((1,16),np.float32);p[0,:3]=1;p[0,3]=1;p[0,6]=1.5;p[0,7]=1
    return append_saved_studio_mesh_to_gpu_scene(empty,v,[0],p,scale_m=1,center_m=[0,0,0],wavelengths_um=wavelengths,emitter_profiles={'version':1,'records':[reference]})


@pytest.mark.parametrize('key',KEYS)
def test_original_named_source_reaches_native_emission_bands_without_rgb_substitution(key):
    definition,_=named_emitter(key);source=ref(key);s=scene(source)
    wavelengths=[.65,.55,.45,.59,.365,.395,.85,.94] if key=='studio_calibration_nir850' else [.45,.55,.65]
    mid=s[1][0,14:15].view(np.uint32)[0];bands=s[2].reshape(-1,32,12)[mid,:len(wavelengths)]
    np.testing.assert_allclose(bands[:,5],transport_band_power(definition,wavelengths),rtol=1e-5,atol=1e-30)
    assert s[0][0,15:16].view(np.uint32)[0]&1
    off=scene(ref(key,0));offmid=off[1][0,14:15].view(np.uint32)[0]
    assert not off[0][0,15:16].view(np.uint32)[0]&1
    assert not off[2].reshape(-1,32,12)[offmid,:len(wavelengths),5].any()


def test_revision_and_duplicate_material_fail_closed():
    a=ref(KEYS[0]);a['revision']='0'*64
    with pytest.raises(ValueError,match='revision'):validate_named_emitter_refs({'version':1,'records':[a]},1)
    a=ref(KEYS[0])
    with pytest.raises(ValueError,match='repeated'):validate_named_emitter_refs({'version':1,'records':[a,a]},2)


def test_native_colour_is_not_peak_normalized():
    from material_db import profile_to_rgb
    for key in KEYS:
        d,native=named_emitter(key)
        np.testing.assert_allclose(d['preview_rgb'],np.maximum(profile_to_rgb(native),0))
    d,_=named_emitter(KEYS[0]);assert max(d['preview_rgb'])>1  # HDR is retained, never peak-clamped


def test_source_power_is_preserved_without_pdf_height_dimming():
    d={'wavelengths_nm':[380,750],'spectral_power':[1/370,1/370]}
    assert transport_band_power(d,[.45,.55,.65]).sum()==pytest.approx(3)
    for key in KEYS[:-1]:
        definition,_=named_emitter(key)
        assert transport_band_power(definition,[.45,.55,.65]).sum()==pytest.approx(3)
    definition,_=named_emitter(KEYS[-1])
    assert transport_band_power(definition,[.65,.55,.45,.59,.365,.395,.85,.94]).sum()==pytest.approx(8)

@pytest.mark.parametrize('key',KEYS)
def test_original_native_triangle_emitter_samples_named_power(key):
    from camera_designer.studio_native_emitters import native_emissive_sources
    wavelengths=[.65,.55,.45,.59,.365,.395,.85,.94] if key=='studio_calibration_nir850' else [.45,.55,.65]
    s=scene(ref(key));roots=native_emissive_sources(s,wavelengths,ray_budget=30)
    assert len(roots)>=3 and np.isfinite(roots).all()
    # Same authored triangle area and source gain for every profile; changing
    # spectrum redistributes original power instead of shrinking by PDF height.
    assert roots[:,10].sum()==pytest.approx(.005*len(wavelengths),rel=1e-5)
    assert len(native_emissive_sources(scene(ref(key,0)),wavelengths,ray_budget=30))==0

@pytest.mark.parametrize('bad',[10**400,float('inf'),float('nan')])
def test_oversized_gain_rejected_as_value_error(bad):
    value=ref(KEYS[0]);value['gain']=bad
    with pytest.raises(ValueError):validate_named_emitter_refs({'version':1,'records':[value]},1)


def test_oversized_filter_rejected_as_value_error():
    value=ref(KEYS[0]);value['filter']=[10**400,1,1]
    with pytest.raises(ValueError):validate_named_emitter_refs({'version':1,'records':[value]},1)


def test_filter_off_uses_unmodified_material_emission_profile():
    from camera_designer.studio_named_emitters import named_source_material
    from material_db import EmissionProfileDatabase,profile_to_rgb
    value=ref('led_cool_white');value['filter']=None
    validated=validate_named_emitter_refs({'version':1,'records':[value]},1)[0]
    definition,native=named_emitter(value['profile'])
    _,payload=named_source_material(validated,[.45,.55,.65],EmissionProfileDatabase())
    np.testing.assert_allclose(payload['emission_rgb'],np.maximum(profile_to_rgb(native),0),atol=2e-6)
    np.testing.assert_allclose(transport_band_power(definition,[.45,.55,.65],None).sum(),3,rtol=1e-6)


@pytest.mark.parametrize('filter_rgb',[[1,1,1],[1,0,0],[0,.4,1],[0,0,0]])
def test_filter_matrix_matches_native_colour_and_losses_stay_lost(filter_rgb):
    from camera_designer.studio_named_emitters import named_source_material
    from material_db import EmissionProfileDatabase,profile_to_rgb
    for key in KEYS:
        d,_=named_emitter(key);value=ref(key);value['filter']=filter_rgb
        db=EmissionProfileDatabase();name,payload=named_source_material(value,[.45,.55,.65],db)
        expected=np.maximum(np.asarray(d['filter_rgb_matrix'])@filter_rgb,0)
        np.testing.assert_allclose(payload['emission_rgb'],expected,atol=2e-6)
        filtered=transport_band_power(d,[.45,.55,.65],filter_rgb)
        assert filtered.sum()<=3+1e-10
        if filter_rgb==[0,0,0]:assert not filtered.any()
        if filter_rgb==[1,0,0]:assert filtered.sum()<3


def test_dark_named_emitter_retains_native_surface_coefficients():
    from camera_designer.studio_named_emitters import named_source_material
    from material_db import EmissionProfileDatabase,MaterialDatabase
    base=dict(albedo=[.2,.4,.6],roughness=.8,metallic=0,transmission=0,ior=1.5,opacity=1)
    db=MaterialDatabase();db.register('base',base)
    native=db.build_mat_buf(freq_hz=299792458./(np.array([.45,.55,.65])*1e-6))[:3]
    _,payload=named_source_material(ref(KEYS[0],0),[.45,.55,.65],EmissionProfileDatabase(),surface=base)
    assert payload['albedo']==base['albedo']
    for i,b in enumerate(payload['spectral_bands']):
        np.testing.assert_allclose([b['reflectance'],b['transmittance'],b['diffuse_frac'],b['ior_real'],b['ior_imag']],native[i,[2,3,4,7,8]])
        assert b['emission']==0
