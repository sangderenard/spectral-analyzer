import numpy as np
import pytest
from camera_designer.studio_forward_field import validate_receive_light_field
from camera_designer import studio_directional_field as d

def config():
    return dict(receive_light_field=True,transport='fixed:3',fixtures={},live={'ray_batch_size':8192},light_field_estimator=d.ROOT_ESTIMATOR,emitter_sampling=dict(version=1,ray_budget=4096,seed=42,refresh_each_sweep=True))

def test_internal_flag_requires_v6_and_strict_boolean():
    assert validate_receive_light_field(config())
    assert validate_receive_light_field({**config(),'light_field_internal_media':True})
    for bad in [1,'true',None]:
        with pytest.raises(ValueError,match='boolean'):validate_receive_light_field({**config(),'light_field_internal_media':bad})
    with pytest.raises(ValueError,match='refreshed v6'):
        validate_receive_light_field(dict(receive_light_field=True,transport='fixed:3',light_field_internal_media=True))

def test_internal_receipt_is_explicit_and_legacy_absent(monkeypatch):
    monkeypatch.setattr(d,'readback',lambda r:np.zeros((32,32,32,6,6),'<f4'))
    r=dict(triangles=np.zeros((1,4),'<u4'),materials=np.zeros((1,16),'<f4'),scene_sha256='a'*64,bin_count=6)
    _,old=d.bundle_sections(r,np.zeros((1,12),'<f4'),12)
    assert 'internal_media' not in old['directional']
    r['internal_media']=True
    _,new=d.bundle_sections(r,np.zeros((1,12),'<f4'),12)
    assert new['directional']['internal_media']['entry_packet'] is True
    assert 'not an interior-medium estimate' not in new['directional']['semantics']
    assert 'No added scattering' in new['directional']['semantics']
