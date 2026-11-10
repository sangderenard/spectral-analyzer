import sys
from pathlib import Path
import numpy as np
import pytest
from camera_designer.marble_color_profile import (render_linear_srgb, color_priority_map,
    normalize_display_params, display_transform_metadata, SRGB_LUMA,
    response_matrix, sensor_to_linear_srgb)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"turing/examples"))
from meowdoku.marble_photo import _retonemap_linear


def test_default_reference_bytes_are_unchanged():
    rgb=np.random.default_rng(4).uniform(-.2,4,(16,16,3))
    a,_=render_linear_srgb(rgb)
    b,_=render_linear_srgb(rgb,{"display_look":"reference"})
    np.testing.assert_array_equal(a,b)


def test_color_priority_preserves_neutral_ramps_and_raw():
    rgb=np.repeat(np.geomspace(1e-7,1e4,256)[None,:,None],3,axis=2)
    before=rgb.copy()
    for tone in ["exponential","linear"]:
        controls={"normalization":"manual","normalization_white":1,"tone_curve":tone}
        a,_=render_linear_srgb(rgb,controls)
        b,_=render_linear_srgb(rgb,{**controls,"display_look":"color_priority"})
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(rgb,before)


def test_positive_rgb_ratios_survive_exposure_and_shoulder():
    rgb=np.array([[[.1,.8,.2],[.7,.2,.1],[.1,.2,.9]]])
    controls=normalize_display_params({"display_look":"color_priority"})
    for scale in [.001,.1,1,10,1000]:
        mapped=color_priority_map(rgb*scale,controls,1,SRGB_LUMA)
        np.testing.assert_allclose(mapped/mapped.max(axis=-1,keepdims=True),rgb/rgb.max(axis=-1,keepdims=True),atol=2e-15)
        assert np.all((mapped>=0)&(mapped<=1))


def test_old_saturation_cancellation_reproduced_and_color_priority_keeps_highlight_color():
    lines=np.array([[[0,1.,0],[1.,.1,.1],[.1,.1,1.]]])
    rgb,_=sensor_to_linear_srgb(lines@response_matrix().T)
    base={"normalization":"manual","normalization_white":1}
    old,_=render_linear_srgb(rgb,base)
    old2,_=render_linear_srgb(rgb,{**base,"saturation":2})
    np.testing.assert_array_equal(old,old2)
    new,_=render_linear_srgb(rgb,{**base,"display_look":"color_priority"})
    assert int(new[0,0,0])<int(old[0,0,0]) and int(new[0,0,2])<int(old[0,0,2])
    assert int(new[0,0,1])-int(new[0,0,0]) > int(old[0,0,1])-int(old[0,0,0])


def test_color_priority_saturation_adjusts_mixed_color_without_modifying_neutral():
    rgb=np.array([[[.7,.4,.3],[.5,.5,.5]]])
    a,_=render_linear_srgb(rgb,{"display_look":"color_priority","saturation":1})
    b,_=render_linear_srgb(rgb,{"display_look":"color_priority","saturation":1.5})
    assert not np.array_equal(a[0,0],b[0,0])
    np.testing.assert_array_equal(a[0,1],b[0,1])


@pytest.mark.parametrize("tone",["linear","exponential"])
@pytest.mark.parametrize("black",[0,.15,.5])
def test_actual_saved_and_live_display_paths_match(tone,black):
    rgb=np.random.default_rng(10).normal(1,2,(31,27,3))
    rgb[0,0]=0
    params={"display_look":"color_priority","tone_curve":tone,"black_point":black,
        "exposure":-.7,"tint":.2,"whitebalance":-.3,"saturation":1.4,"vibrance":.3,
        "firefly_filter":True}
    a,_=render_linear_srgb(rgb,params)
    b=_retonemap_linear(rgb,params,colorimetric=True)
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all()


def test_specular_clipping_policy_unchanged_and_invalid_look_rejected():
    p=normalize_display_params({"normalization":"specular","display_look":"color_priority"})
    assert p["display_look"]=="reference"
    with pytest.raises(ValueError):normalize_display_params({"display_look":"Canon"})
    m=display_transform_metadata({"display_look":"color_priority"})
    assert 'Not Canon or ACES' in m['highlight_policy']
