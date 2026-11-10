import numpy as np
import pytest
from canonical_material_documents import MaterialDocument
from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
from material_db import MaterialDatabase


def fixture(source):
    doc=MaterialDocument('adapter_test','test',source.encode())
    authoring={'version':1,'records':[{'material':0,'source':doc.snapshot(),'overrides':[]}]}
    scene=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.zeros(3),np.ones(3)),np.zeros((0,12),np.float32))
    vertices=np.array([[[0,0,0],[1,0,0],[0,1,0]]],np.float32)
    pbr=np.zeros((1,16),np.float32);pbr[:,6]=1;pbr[:,7]=1
    return doc,authoring,scene,vertices,pbr


def test_native_source_ray_and_spectral_buffers():
    doc,a,scene,v,p=fixture('''reflectivity: 0.2
diffusion: 0.1
absorption: 0.7
ior: 1.6
spectral_bands:
- center_hz: 500000000000000
  bandwidth_hz: 100000000000000
  reflectance: 0.15
  transmittance: 0.75
  ior_real: 1.6
  ior_imag: 0.002
''')
    result=append_saved_studio_mesh_to_gpu_scene(scene,v,[0],p,scale_m=1,center_m=[0,0,0],wavelengths_um=[.6],material_authoring=a)
    db=MaterialDatabase.instance();t=db.build_tensors()
    index=int(result[1][0,14:15].view(np.uint32)[0])
    np.testing.assert_allclose(result[1][0,:3],[.2,.1,.7])
    np.testing.assert_array_equal(result[2],db.build_mat_buf(freq_hz=np.array([299792458/.6e-6])))
    assert t['spectral'][index,0,8]==pytest.approx(.002)
    assert t['pbr'][index,6]==pytest.approx(1.6)


@pytest.mark.parametrize('extra',['enamel: {thickness_nm: 10}', 'texture_stack: {layers: [1]}', 'emission_rgb: [1, 0, 0]', 'mystery_physics: 4'])
def test_unsupported_source_fails_without_fallback(extra):
    _,a,scene,v,p=fixture('ior: 1.5\n'+extra+'\n')
    with pytest.raises(ValueError,match='does not transport'):
        append_saved_studio_mesh_to_gpu_scene(scene,v,[0],p,scale_m=1,center_m=[0,0,0],material_authoring=a)


@pytest.mark.parametrize('source',['ior: .nan', 'a: '+ '['*50+'0'+']'*50, 'a: &a [*a]'])
def test_nonfinite_deep_recursive_source_rejected(source):
    with pytest.raises(ValueError): MaterialDocument('invalid','test',source.encode())

@pytest.mark.parametrize('source,field',[
    ('pbr: {anisotropy: 0.8}', 'pbr.anisotropy'),
    ('spectral_bands: [{ior_real: 1.5, mystery: 2}]', 'spectral_bands[0].mystery'),
    ('enamel: {custom_film: 2}', 'enamel.custom_film'),
    ('texture_stack: {unknown_map: 1}', 'texture_stack.unknown_map'),
])
def test_nested_capabilities_not_silently_consumed(source,field):
    from canonical_material_documents import native_payload
    doc=MaterialDocument('nested','test',source.encode())
    assert field in native_payload(doc).uncompiled_fields


@pytest.mark.parametrize('source',['ior: 1.0e100','pbr: {albedo: [1.0e100, 0, 0]}','spectral_bands: [{center_hz: 1.0e100}]'])
def test_finite_source_overflows_native_projection(source):
    from canonical_material_documents import native_payload
    with pytest.raises(ValueError,match='native|float32'):
        native_payload(MaterialDocument('overflow','test',source.encode()))


def test_pbr_route_reactive_shift_explicitly_uncompiled():
    from canonical_material_documents import native_payload
    doc,a,scene,v,p=fixture('ior: 1.5\nreactive_shift_hz: 123\n')
    assert any('reactive_shift_hz' in f for f in native_payload(doc).uncompiled_fields)
    with pytest.raises(ValueError,match='reactive_shift_hz'):
        append_saved_studio_mesh_to_gpu_scene(scene,v,[0],p,scale_m=1,center_m=[0,0,0],material_authoring=a)

@pytest.mark.parametrize('source',['date: 2026-11-09','a: !!binary SGVsbG8=','1: value','a: !!set {x: null}'])
def test_non_json_yaml_definition_rejected(source):
    with pytest.raises(ValueError): MaterialDocument('nonjson','test',source.encode())

@pytest.mark.parametrize('field',['color_profile_idx','emit_profile_idx','remit_profile_idx'])
def test_raw_profile_indices_are_not_portable_authoring(field):
    from canonical_material_documents import native_payload
    with pytest.raises(ValueError,match='stable named dependency'):
        native_payload(MaterialDocument('index_only','test',f'{field}: 0'.encode()))


def test_nested_smoothness_reported_uncompiled():
    from canonical_material_documents import native_payload
    compiled=native_payload(MaterialDocument('nested_smoothness','test',b'pbr: {smoothness: 0.9}'))
    assert 'pbr.smoothness' in compiled.uncompiled_fields


def test_canonical_volume_keeps_native_nk_and_uses_bulk_beer():
    from mat_flags import MAX_SPECTRAL_BANDS
    doc,a,scene,v,p=fixture('''ior: 1.53
opacity: 0.8
transmission: 0.2
spectral_bands:
- center_hz: 500000000000000
  bandwidth_hz: 100000000000000
  ior_real: 1.53
  ior_imag: 0.000002
''')
    marker={'version':1,'source':'native_spectral_bands'}
    boundary={'version':1,'ranges':[{'start':0,'count':1,'positive':-1,'negative':0}]}
    out=append_saved_studio_mesh_to_gpu_scene(scene,v,[0],p,scale_m=1,center_m=[0,0,0],wavelengths_um=[.6],material_authoring=a,canonical_volume_optics=marker,boundary_media=boundary)
    mid=int(out[1][0,14:15].view(np.uint32)[0]);band=out[2][mid*MAX_SPECTRAL_BANDS]
    assert band[7]==pytest.approx(1.53)
    assert band[8]==pytest.approx(.000002)
    assert out[1][0,7]==0  # boundary enabled despite authored surface opacity
    assert doc.definition['opacity']==.8  # source not rewritten
    trans=lambda d:np.exp(-4*np.pi*float(band[8])*float(band[0])*d/299792458)
    assert trans(.002)==pytest.approx(trans(.001)**2)
    with pytest.raises(ValueError,match='not both'):
        append_saved_studio_mesh_to_gpu_scene(scene,v,[0],p,scale_m=1,center_m=[0,0,0],material_authoring=a,canonical_volume_optics=marker,boundary_media=boundary,optical_media={})


def test_canonical_injection_bounds_count_each_owner():
    from canonical_material_documents import native_payload
    from camera_designer.studio_material_authoring import validate_canonical_volume_optics
    c=native_payload(MaterialDocument('nk','test',b'ior: 1.5\nspectral_bands: [{ior_real: 1.5, ior_imag: 0.0004}]'))
    marker={'version':1,'source':'native_spectral_bands'}
    regions=[{'material':0,'mode':'host'},{'material':0,'mode':'injection_dye'},{'material':0,'mode':'injection_dye'}]
    with pytest.raises(ValueError,match='combined'):
        validate_canonical_volume_optics(marker,{0:c},[0],regions)
    assert validate_canonical_volume_optics(marker,{0:c},[0],regions[:2])[0]==.0004


def test_native_volume_ignores_surface_only_data_without_losing_source():
    from canonical_material_documents import native_volume_payload
    from camera_designer.studio_material_authoring import compile_scene_authoring
    source='''ior: 1.5
color_profile_name: missing_surface_tint_only
enamel: {unknown_coating_metadata: 3}
texture_stack: {unavailable_texture: example}
spectral_bands: [{center_hz: 5.0e14, bandwidth_hz: 1.0e14, ior_real: 1.5, ior_imag: 0.000001}]
'''
    doc,a,scene,v,p=fixture(source)
    c=native_volume_payload(doc)
    assert c.revision==doc.revision
    assert c.payload['_canonical_source_snapshot']==doc.snapshot()
    assert set(c.payload['_volume_ignored_fields'])=={'color_profile_name','enamel','texture_stack'}
    assert len(c.warnings)==3
    with pytest.raises(ValueError,match='unresolved color_profile_name'):
        compile_scene_authoring(a,1,None)
    boundary={'version':1,'ranges':[{'start':0,'count':1,'positive':-1,'negative':0}]}
    out=append_saved_studio_mesh_to_gpu_scene(scene,v,[0],p,scale_m=1,center_m=[0,0,0],material_authoring=a,boundary_media=boundary,canonical_volume_optics={'version':1,'source':'native_spectral_bands'})
    mid=int(out[1][0,14:15].view(np.uint32)[0])
    assert out[2][mid*32,8]==pytest.approx(.000001)


def test_native_volume_does_not_ignore_construction_glass_emitter():
    from canonical_material_documents import read_material_document,native_volume_payload
    # Disk profile actually authors an emitter, so cannot truthfully be admitted
    # as a nonemissive dielectric merely because its colour name is missing.
    doc=read_material_document('construction_glass')
    from camera_designer.studio_material_authoring import compile_scene_authoring
    a={'version':1,'records':[{'material':0,'source':doc.snapshot(),'overrides':[]}]}
    with pytest.raises(ValueError,match='emit_profile|emission|mat_flags'):
        compile_scene_authoring(a,1,None,native_volume=True)
