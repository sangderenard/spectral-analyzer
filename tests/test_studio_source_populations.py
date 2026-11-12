import numpy as np
import pytest
from camera_designer.studio_native_emitters import sampling_settings
from camera_designer import studio_directional_field as f

def test_settings_are_existing_generator_knobs():
    assert sampling_settings({})['seed']==42
    config=dict(live={'ray_batch_size':8192},transport='fixed:3',fixtures={},receive_light_field=True,light_field_estimator=f.ROOT_ESTIMATOR,emitter_sampling=dict(version=1,ray_budget=4096,seed=42,refresh_each_sweep=True))
    assert sampling_settings(config)['refresh_each_sweep']
    for key,value in [('ray_budget',0),('seed',-1),('refresh_each_sweep',1)]:
        bad={**config,'emitter_sampling':{**config['emitter_sampling'],key:value}}
        with pytest.raises(ValueError):sampling_settings(bad)

def test_progressive_raw_moments_are_weighted_and_reset(monkeypatch):
    from OpenGL import GL
    for name in ('GL_BUFFER_UPDATE_BARRIER_BIT','GL_SHADER_STORAGE_BARRIER_BIT','GL_SHADER_STORAGE_BUFFER','GL_R32I','GL_RED_INTEGER','GL_INT'):monkeypatch.setattr(GL,name,1,raising=False)
    raw=np.zeros(f.word_count(6,32)+1,np.int32)
    raw[:6]=[100,0,0,100,0,0]
    monkeypatch.setattr(GL,'glMemoryBarrier',lambda *a:None,raising=False)
    monkeypatch.setattr(GL,'glBindBuffer',lambda *a:None,raising=False)
    monkeypatch.setattr(GL,'glGetBufferSubData',lambda target,offset,size:raw.tobytes()[offset:offset+size],raising=False)
    monkeypatch.setattr(GL,'glClearBufferData',lambda *a:None,raising=False)
    record=dict(estimator=f.ROOT_ESTIMATOR,source_root_count=3,grid_size=32,bin_count=6,buffer=1,voxel_volume_m3=1)
    settings=dict(ray_budget=128,seed=42)
    sources=np.zeros((3,12),np.float32)
    def pump():
        f.reset_root_accounting(record);f.begin_root_dispatch(record,3);f.submitted_root_chunk(record,3);f.finish_root_dispatch(record)
    pump();f.commit_source_population(record,sources,settings,0)
    f.clear(record,preserve_progressive=True)
    raw[:6]=[300,0,0,0,300,0]
    pump();f.commit_source_population(record,sources,settings,1)
    a=f.readback(record)
    np.testing.assert_allclose(a[0,0,0,0,3:],[.25,.75,0])
    assert record['progressive']['root_sampling']['normalization_divisor']==2
    record['quantization_divisor']=16777216
    finer=f.readback(record)
    np.testing.assert_allclose(finer[...,:3],a[...,:3]*(65535/16777216),rtol=2e-6)
    np.testing.assert_array_equal(finer[...,3:],a[...,3:])
    assert record['progressive']['metadata']['last_index']==1
    assert record['progressive']['nonzero_cells']==1
    with pytest.raises(ValueError):f.commit_source_population(record,sources,settings,1)
    f.clear(record)
    assert 'progressive' not in record

def test_half_mm_quantization_is_precision_not_gain():
    energy=.019037418198167233*.0005/2
    assert round(energy*65535)==0
    count=round(energy*16777216)
    assert count==80
    assert abs(count/16777216-energy)<=.5/16777216
