import json
import numpy as np
import pytest
from PIL import Image
from camera_designer.marble_tiles import tile_plan, tile_windows, TileAssembly, require_tile_disk
from camera_designer.marble_color_profile import sensor_to_linear_srgb, render_linear_srgb


def config(width=129,height=130,**extra):
    return {"camera":{"width":width,"height":height},"transport":"fixed:3","tiled":{"tile_size":128},**extra}


def test_plan_auto_and_bounds():
    assert tile_plan({"camera":{"resolution":4096},"transport":"fixed:3"}) is None
    plan=tile_plan({"camera":{"resolution":8192},"transport":"fixed:3","oversampling":2})
    assert (plan["width"],plan["output_width"],plan["total_tiles"])==(16384,8192,256)
    assert plan["requested_exposure_seconds"]==7680
    assert 8*1024**3 < plan["required_disk_bytes"] < 9*1024**3
    for change in ({"oversampling":True},{"oversampling":5},{"live":{}},{"orbit_frames":8},{"client_intersections":True},{"transport":"fixed:1"}):
        with pytest.raises(ValueError):tile_plan(config(**change))
    with pytest.raises(ValueError):tile_plan(config(32768,32768))
    with pytest.raises(ValueError):tile_plan(config(16384,16384,oversampling=3))


def test_windows_cover_partial_edges_once():
    plan=tile_plan(config(257,259))
    coverage=np.zeros((259,257),np.uint8)
    for w in tile_windows(plan):coverage[w['y']:w['y']+w['height'],w['x']:w['x']+w['width']]+=1
    assert np.all(coverage==1)
    assert list(tile_windows(plan))[-1]=={'x':256,'y':256,'width':1,'height':3}


@pytest.mark.parametrize('factor,passes,effective',[(1,64,64),(2,16,64),(3,8,72),(4,4,64)])
def test_uniform_output_sample_target(factor,passes,effective):
    plan=tile_plan({'camera':{'resolution':256},'transport':'fixed:3',
        'oversampling':factor,'samples_per_output_pixel':64,'exposure_seconds':.1})
    assert plan['target_samples_per_pixel']==passes
    assert plan['requested_samples_per_output_pixel']==64
    assert plan['target_samples_per_output_pixel']==effective
    assert plan['exposure_seconds_per_tile'] is None
    assert plan['requested_exposure_seconds'] is None


@pytest.mark.parametrize('value',[None,True,False,0,-1,1.5,float('inf'),'64',65537])
def test_invalid_uniform_sample_targets(value):
    with pytest.raises(ValueError,match='samples_per_output_pixel'):
        tile_plan(config(samples_per_output_pixel=value))


def test_uniform_assembly_rejects_undersampled_tile(tmp_path):
    plan=tile_plan(config(samples_per_output_pixel=8))
    assembly=TileAssembly(tmp_path,plan)
    try:
        with pytest.raises(ValueError,match='uniform sample target'):
            assembly.add(next(tile_windows(plan)),tmp_path/'missing.npz',7)
        assert assembly.completed==0
    finally:assembly.close()


@pytest.mark.parametrize('factor',[1,2,3,4])
@pytest.mark.parametrize('uniform',[False,True])
def test_assembly_matches_monolithic_linear_downsample(tmp_path,factor,uniform):
    plan=tile_plan(config(129,130,oversampling=factor,
        **({'samples_per_output_pixel':64} if uniform else {})))
    raw=np.random.default_rng(17).random((130*factor,129*factor,3),dtype=np.float32)
    assembly=TileAssembly(tmp_path,plan)
    try:
        for i,w in enumerate(tile_windows(plan)):
            samples=plan['target_samples_per_pixel'] if uniform else 2**(i%3)
            tile=raw[w['y']:w['y']+w['height'],w['x']:w['x']+w['width']]
            path=tmp_path/'input.npz';np.savez(path,linear_rgb_exposure=tile*samples)
            assembly.add(w,path,samples)
        result=assembly.finish(None,1.25)
        expected_raw=raw if factor==1 else raw.reshape(130,factor,129,factor,3).mean(axis=(1,3),dtype=np.float64).astype(np.float32)
        rgb,profile=sensor_to_linear_srgb(expected_raw,None)
        expected,white=render_linear_srgb(rgb,{'gamma':1.25})
        with Image.open(tmp_path/result['image']) as image:np.testing.assert_array_equal(np.asarray(image),expected)
        assert result['display_white']==white
        with np.load(tmp_path/result['hdr']) as data:
            np.testing.assert_array_equal(data['linear_rgb_exposure'],expected_raw)
            np.testing.assert_array_equal(data['linear_srgb_exposure'],rgb)
            assert int(data['oversampling'])==factor
            if uniform:
                assert np.all(data['tile_samples']==plan['target_samples_per_pixel'])
                assert int(data['target_samples_per_output_pixel'])==plan['target_samples_per_output_pixel']
        assert not list(tmp_path.glob('tile-*.npy'))
    finally:assembly.close()


def test_incomplete_and_duplicate_tiles_rejected(tmp_path):
    plan=tile_plan(config())
    assembly=TileAssembly(tmp_path,plan)
    try:
        with pytest.raises(ValueError,match='incomplete'):assembly.finish(None,1)
        w=next(tile_windows(plan))
        with pytest.raises(ValueError,match='complete sensor pass'):assembly.add(w,tmp_path/'missing',0)
        path=tmp_path/'input.npz';np.savez(path,linear_rgb_exposure=np.ones((128,128,3),np.float32))
        assembly.add(w,path,1)
        with pytest.raises(ValueError,match='next queued'):assembly.add(w,path,1)
    finally:assembly.close()


def test_disk_admission(monkeypatch,tmp_path):
    import camera_designer.marble_tiles as tiles
    class Usage:free=1
    monkeypatch.setattr(tiles.shutil,'disk_usage',lambda p:Usage())
    with pytest.raises(ValueError,match='free disk'):require_tile_disk(tmp_path,tile_plan(config()))
