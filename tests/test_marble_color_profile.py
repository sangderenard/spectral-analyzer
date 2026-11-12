"""Reference colorimetry, conditioning, and one-pass output regressions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "turing" / "examples"))

from camera_designer.marble_color_profile import (
    CIE_FIXED3_XYZ, D65_XYZ, SRGB_TO_XYZ, PROFILE_ID,
    build_output_profile, response_matrix, sensor_to_linear_srgb, render_linear_srgb,
    preview_sensor_samples, normalize_display_params, display_transform_metadata, suppress_fireflies,
    display_white_reference,
)
from camera_designer import marble_preview_worker as worker


def test_cie_reference_samples_and_srgb_white_are_standard_values():
    np.testing.assert_array_equal(CIE_FIXED3_XYZ[:, 0], [.2835, .107, 0.])
    np.testing.assert_array_equal(CIE_FIXED3_XYZ[:, 1], [.4334499, .9949501, .008749999])
    np.testing.assert_array_equal(CIE_FIXED3_XYZ[:, 2], [.3362, .038, 1.77211])
    np.testing.assert_allclose(SRGB_TO_XYZ @ np.ones(3), D65_XYZ, rtol=1e-14)


def test_single_lines_and_mixtures_transform_linearly_without_clipping():
    sensor = response_matrix()
    primaries, metadata = sensor_to_linear_srgb(sensor.T)
    mixture, _ = sensor_to_linear_srgb((sensor @ np.array([.7, .2, .4]))[None, :])
    np.testing.assert_allclose(mixture[0], np.array([.7, .2, .4]) @ primaries, atol=1e-14)
    assert np.any(primaries < 0)  # real spectral lines lie outside sRGB
    assert not metadata["neutral_fallback"]
    assert metadata["id"] == PROFILE_ID


def test_changing_valid_sensor_curves_preserves_profiled_color_and_raw():
    lines = np.random.default_rng(7).uniform(0, 4, (8, 9, 3))
    custom = {"version": 1, "centers_nm": [630, 540, 470], "sigmas_oct": [.24, .21, .18]}
    raw_default = lines @ response_matrix().T
    raw_custom = lines @ response_matrix(custom).T
    original = raw_custom.copy()
    a, _ = sensor_to_linear_srgb(raw_default)
    b, metadata = sensor_to_linear_srgb(raw_custom, custom)
    np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(raw_custom, original)
    assert not metadata["neutral_fallback"]


def test_fixed_reference_white_is_neutral_at_all_exposures():
    sensor_white = response_matrix() @ np.ones(3)
    for scale in [.001, 1., 1e6]:
        rgb, _ = sensor_to_linear_srgb((scale * sensor_white)[None, :])
        np.testing.assert_allclose(rgb, np.full((1, 3), scale), rtol=2e-14)


@pytest.mark.parametrize("response", [
    {"version": 1, "centers_nm": [550] * 3, "sigmas_oct": [.06] * 3},
    {"version": 1, "centers_nm": [380] * 3, "sigmas_oct": [.01] * 3},
])
def test_degenerate_or_insensitive_filters_use_explicit_finite_neutral_fallback(response):
    profile, metadata = build_output_profile(response)
    assert metadata["neutral_fallback"] and metadata["warning"]
    json.dumps(metadata, allow_nan=False)
    value = profile.apply_matrix(np.array([[1., .5, .1]]))
    assert np.all(np.isfinite(value))
    np.testing.assert_allclose(value[:, 0], value[:, 1], atol=1e-12)
    np.testing.assert_allclose(value[:, 1], value[:, 2], atol=1e-12)


def test_display_has_one_srgb_transfer_and_preserves_neutral():
    linear = np.full((2, 2, 3), .18)
    display, white = render_linear_srgb(linear)
    mapped = 1 - np.exp(-2.)
    expected = np.uint8((1.055 * mapped ** (1 / 2.4) - .055) * 255)
    np.testing.assert_array_equal(display, np.full((2, 2, 3), expected, np.uint8))
    assert white == pytest.approx(.18)
    black, _ = render_linear_srgb(np.zeros((2, 2, 3)))
    assert not black.any()


@pytest.mark.parametrize("live_display", [None,
    {"normalization": "specular", "normalization_percentile": 75.}])
def test_snapshot_preserves_raw_and_separately_labels_profiled_hdr(tmp_path, monkeypatch, live_display):
    from OpenGL import GL
    raw = np.array([[[2., 3., 4.], [.5, 1., .25]]], dtype=np.float32)
    selected = [0]
    monkeypatch.setattr(GL, "glFinish", lambda: None, raising=False)
    monkeypatch.setattr(GL, "glBindTexture", lambda target, texture: selected.__setitem__(0, texture), raising=False)
    def pixels(*args):
        rgba = np.ones((1, 2, 4), np.float32)
        rgba[..., :3] = raw[..., selected[0], None]
        return rgba.tobytes()
    monkeypatch.setattr(GL, "glGetTexImage", pixels, raising=False)
    for key in ("GL_TEXTURE_2D", "GL_RGBA", "GL_FLOAT"):
        monkeypatch.setattr(GL, key, 0, raising=False)
    acc = SimpleNamespace(_textures=[0, 1, 2], _w=2, _h=1)
    retouch = {"normalization": "specular", "normalization_percentile": 50.}
    (tmp_path / "retouch.json").write_text(json.dumps(retouch))
    status = worker._snapshot(acc, tmp_path, "test", 1, 1., 0., 1, 1,
                              finished=True, transport="fixed:3", display_params=live_display)
    assert status["color_profile"]["id"] == PROFILE_ID
    with np.load(tmp_path / status["hdr"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["linear_rgb_exposure"], raw)
        expected, _ = sensor_to_linear_srgb(raw)
        np.testing.assert_array_equal(saved["linear_srgb_exposure"], expected)
        assert saved["linear_srgb_exposure"].dtype == np.float64
        assert json.loads(str(saved["color_profile_json"]))["id"] == PROFILE_ID
        assert status["retouch_display_transform"] == display_transform_metadata(
            retouch, display_white_reference(expected, retouch))
        assert status["display_white"] == display_white_reference(expected, live_display)
        assert status["display_transform"] == display_transform_metadata(
            live_display, status["display_white"])


def test_saved_retouch_matches_live_recipe_and_legacy_default_is_unchanged():
    path = Path(__file__).parents[2] / "turing/examples/meowdoku/marble_photo.py"
    if not path.exists():
        pytest.skip("sibling Turing checkout is needed for cross-repository parity")
    spec = importlib.util.spec_from_file_location("meowdoku.marble_photo_color_parity", path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    rgb = np.random.default_rng(8).uniform(-.3, 4, (7, 9, 3))
    for params in ({}, {"exposure": 1.2, "gamma": 1.1, "red": .7,
                       "whitebalance": -.4, "tint": .2, "intensity": .8},
                   {"normalization": "manual", "normalization_white": 2.5},
                   {"tone_curve": "linear"},
                   {"normalization": "manual", "normalization_white": .02, "tone_curve": "linear"},
                   {"saturation": 0., "vibrance": -1., "black_point": .2},
                   {"saturation": 1.7, "vibrance": .8, "black_point": .05},
                   {"firefly_filter": True, "firefly_threshold": 2., "firefly_strength": .7},
                   {"normalization": "percentile", "normalization_percentile": 80.},
                   {"normalization": "specular", "normalization_percentile": 25.},
                   {"normalization": "manual", "normalization_white": 4., "tone_curve": "linear",
                    "saturation": 1.4, "vibrance": -.6, "black_point": .5}):
        live, _ = render_linear_srgb(rgb, params)
        np.testing.assert_array_equal(bridge._retonemap_linear(rgb, params, colorimetric=True), live)
        assert bridge._display_metadata(rgb, params) == display_transform_metadata(params, render_linear_srgb(rgb, params)[1])
    legacy = np.full((2, 2, 3), .18)
    old = bridge._retonemap_linear(legacy, {})
    expected = np.uint8((1.055 * (1 - np.exp(-2.)) ** (1 / 2.4) - .055) * 255)
    np.testing.assert_array_equal(old, np.full((2, 2, 3), expected, np.uint8))
    np.testing.assert_array_equal(bridge._retonemap_linear(legacy,
        {"normalization": "manual", "normalization_white": 99, "tone_curve": "linear"}), old)


@pytest.mark.parametrize("bad", [
    {"normalization": "bogus"}, {"tone_curve": "bogus"},
    {"normalization_white": 0}, {"normalization_white": 1e13},
    {"normalization_white": float("nan")}, {"normalization_white": "oops"},
    {"saturation": -1}, {"saturation": 2.01}, {"saturation": True},
    {"vibrance": -1.01}, {"vibrance": float("inf")},
    {"black_point": .51}, {"black_point": float("nan")},
    {"firefly_filter": 1}, {"firefly_filter": "false"},
    {"firefly_threshold": 1.9}, {"firefly_threshold": 65},
    {"firefly_strength": -.1}, {"firefly_strength": float("nan")},
    {"normalization_percentile": 0}, {"normalization_percentile": 100.1},
    {"normalization_percentile": True}, {"normalization_percentile": float("nan")},
])
def test_display_controls_reject_invalid_values(bad):
    with pytest.raises(ValueError):
        normalize_display_params(bad)


def test_manual_linear_maps_reference_values_once_and_does_not_auto_normalize():
    rgb = np.array([[[0, 0, 0], [.18, .18, .18], [1, 1, 1], [8, 8, 8]]], float)
    params = {"normalization": "manual", "normalization_white": 1., "tone_curve": "linear"}
    pixels, white = render_linear_srgb(rgb, params)
    np.testing.assert_array_equal(pixels, [[[0]*3, [118]*3, [255]*3, [255]*3]])
    assert white == 1.
    # Other patches and HDR highlights cannot silently re-expose this sample.
    isolated, _ = render_linear_srgb(rgb[:, 1:2], params)
    np.testing.assert_array_equal(isolated, pixels[:, 1:2])
    before = rgb.copy()
    preview = preview_sensor_samples(params=params)
    assert preview["display_transform"]["tone_formula"] == "clip(Y/white,0,1)"
    assert preview["display_white"] == 1.
    np.testing.assert_array_equal(rgb, before)


def test_artistic_chroma_controls_preserve_neutrals_and_vibrance_targets_weaker_color():
    params = {"normalization": "manual", "normalization_white": 1., "tone_curve": "linear"}
    rgb = np.array([[[.3, .3, .3], [.36, .3, .24], [.6, .3, 0.]]])
    original = rgb.copy()
    base, _ = render_linear_srgb(rgb, params)
    vivid, _ = render_linear_srgb(rgb, dict(params, vibrance=1.))
    np.testing.assert_array_equal(vivid[:, 0], base[:, 0])
    np.testing.assert_array_equal(vivid[:, 2], base[:, 2])
    assert np.ptp(vivid[0, 1].astype(int)) > np.ptp(base[0, 1].astype(int))
    gray, _ = render_linear_srgb(rgb, dict(params, saturation=0.))
    np.testing.assert_array_equal(gray[..., 0], gray[..., 1])
    np.testing.assert_array_equal(gray[..., 1], gray[..., 2])
    np.testing.assert_array_equal(rgb, original)


def test_black_point_threshold_and_white_anchor_are_explicit():
    rgb = np.repeat(np.array([0., .05, .1, .55, 1., 2.])[None, :, None], 3, axis=2)
    params = {"normalization": "manual", "normalization_white": 1.,
              "tone_curve": "linear", "black_point": .1}
    pixels, _ = render_linear_srgb(rgb, params)
    np.testing.assert_array_equal(pixels[0, :, 0], [0, 0, 0, 188, 255, 255])
    assert display_transform_metadata(params)["tone_formula"] == "clip(x,0,1)"


def test_creative_panel_live_png_and_saved_display_recipe_match(tmp_path):
    params = {"normalization": "manual", "normalization_white": 1.7,
              "saturation": 1.4, "vibrance": .7, "black_point": .08,
              "firefly_filter": True, "firefly_threshold": 8., "firefly_strength": .8}
    result = preview_sensor_samples(params=params)
    linear = np.asarray(result["linear_srgb"])[None, ...]
    png = worker._render_retouch(linear, params, tmp_path, colorimetric=True)
    from PIL import Image
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / png))[0], result["output_display_rgb8"])
    path = Path(__file__).parents[2] / "turing/examples/meowdoku/marble_photo.py"
    if not path.exists():
        pytest.skip("sibling Turing checkout needed for saved path comparison")
    spec = importlib.util.spec_from_file_location("meowdoku.marble_creative_parity", path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    np.testing.assert_array_equal(bridge._retonemap_linear(linear, params, colorimetric=True)[0],
                                  result["output_display_rgb8"])


def test_firefly_suppression_is_opt_in_and_preserves_raw_and_color_ratios():
    rgb = np.full((7, 7, 3), .01)
    rgb[3, 3] = [100., 50., 25.]
    original = rgb.copy()
    assert suppress_fireflies(rgb, normalize_display_params()) is rgb
    clean = suppress_fireflies(rgb, normalize_display_params({"firefly_filter": True}))
    np.testing.assert_array_equal(rgb, original)
    np.testing.assert_allclose(clean[3, 3] / clean[3, 3, 0], [1., .5, .25], rtol=1e-14)
    np.testing.assert_allclose(clean[3, 3] @ SRGB_TO_XYZ[1], .01, rtol=1e-14)
    np.testing.assert_array_equal(clean[2], rgb[2])
    # Display does not silently renormalize the unchanged surroundings.
    base, white = render_linear_srgb(rgb)
    output, filtered_white = render_linear_srgb(rgb, {"firefly_filter": True})
    assert white == filtered_white
    assert output[3, 3].max() < base[3, 3].max()
    np.testing.assert_array_equal(output[2], base[2])


def test_firefly_filter_preserves_small_connected_features_lines_and_borders():
    rgb = np.full((9, 9, 3), .01)
    rgb[3, 3] = rgb[3, 4] = 100.  # Two-pixel highlight.
    rgb[5, 5] = rgb[6, 6] = 50.   # Diagonal thin feature.
    rgb[1:8, 1] = 80.            # One-pixel-wide line.
    rgb[0, 7] = 1000.            # Border remains unfiltered.
    clean = suppress_fireflies(rgb, normalize_display_params({"firefly_filter": True}))
    np.testing.assert_array_equal(clean, rgb)


def test_firefly_threshold_strength_and_real_point_tradeoff_are_explicit():
    rgb = np.full((5, 5, 3), .01)
    rgb[2, 2] = .1
    base = {"firefly_filter": True}
    protected = suppress_fireflies(rgb, normalize_display_params(dict(base, firefly_threshold=16)))
    np.testing.assert_array_equal(protected, rgb)
    half = suppress_fireflies(rgb, normalize_display_params(dict(base, firefly_strength=.5)))
    np.testing.assert_allclose(half[2, 2], .055)
    disabled = suppress_fireflies(rgb, normalize_display_params(dict(base, firefly_strength=0)))
    np.testing.assert_array_equal(disabled, rgb)
    point = np.zeros((3, 3, 3)); point[1, 1] = 1.
    # A spatial-only filter cannot identify whether this is a real point light.
    assert not suppress_fireflies(point, normalize_display_params(base)).any()
    assert "genuine isolated glints" in display_transform_metadata(base)["firefly_warning"]


def test_actual_firefly_live_png_matches_saved_retouch_recipe(tmp_path):
    from PIL import Image
    rgb = np.full((9, 9, 3), .01); rgb[4, 4] = [100., 30., 1.]
    params = {"firefly_filter": True, "firefly_threshold": 4., "firefly_strength": .9}
    png = worker._render_retouch(rgb, params, tmp_path, colorimetric=True)
    expected, _ = render_linear_srgb(rgb, params)
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / png)), expected)
    path = Path(__file__).parents[2] / "turing/examples/meowdoku/marble_photo.py"
    if not path.exists():
        pytest.skip("sibling Turing checkout needed for saved path comparison")
    spec = importlib.util.spec_from_file_location("meowdoku.marble_firefly_parity", path)
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    np.testing.assert_array_equal(bridge._retonemap_linear(rgb, params, colorimetric=True), expected)


def test_direct_pattern_uses_actual_sensor_profile_and_display_path():
    result = preview_sensor_samples()
    assert result["input_display_rgb8"][0] == [255, 255, 255]
    assert len(result["patch_names"]) == 8
    raw = np.array(result["sensor_rgb"], float)[None, ...]
    linear, metadata = sensor_to_linear_srgb(raw)
    pixels, white = render_linear_srgb(linear)
    assert result["output_display_rgb8"] == pixels[0].tolist()
    assert result["linear_srgb"] == linear[0].tolist()
    assert result["profile"] == metadata and result["display_white"] == white


def test_percentile_specular_mode_clips_only_values_above_chosen_white():
    rgb = np.repeat(np.array([0., 1., 2., 3., 4., 10.])[None, :, None], 3, axis=2)
    original = rgb.copy()
    params = {"normalization": "specular", "normalization_percentile": 50., "tone_curve": "exponential"}
    pixels, white = render_linear_srgb(rgb, params)
    assert white == pytest.approx(3.)
    np.testing.assert_array_equal(pixels[0, :, 0], [0, 156, 213, 255, 255, 255])
    assert display_transform_metadata(params, white)["tone_curve"] == "linear"
    assert display_transform_metadata(params, white)["effective_percentile"] == 50.
    # Relative light level does not change this mode's normalized image.
    brighter, bright_white = render_linear_srgb(rgb * 128., params)
    np.testing.assert_array_equal(brighter, pixels)
    assert bright_white == pytest.approx(128. * white)
    less_clipped, white100 = render_linear_srgb(rgb, dict(params, normalization_percentile=100.))
    assert white100 == pytest.approx(10.)
    assert np.count_nonzero(less_clipped[..., 0] == 255) < np.count_nonzero(pixels[..., 0] == 255)
    np.testing.assert_array_equal(rgb, original)


def test_exposure_iso_and_top_quarter_linear_clip_preserve_raw_sensor_values():
    rgb = np.repeat(np.arange(1., 101.)[None, :, None], 3, axis=2)
    original = rgb.copy()
    params = {"normalization": "percentile", "normalization_percentile": 75.,
              "tone_curve": "linear", "exposure": 0.}
    base, white = render_linear_srgb(rgb, params)
    boosted, boosted_white = render_linear_srgb(rgb, dict(params, exposure=1.))
    assert white == pytest.approx(75.25)
    assert boosted_white == white
    assert 25 <= np.count_nonzero(base[0, :, 0] == 255) <= 26
    assert np.count_nonzero(boosted[0, :, 0] == 255) > 25
    np.testing.assert_array_equal(rgb, original)


def test_percentile_defaults_legacy_auto99_and_black_are_stable():
    assert normalize_display_params()["normalization_percentile"] == 99.
    assert normalize_display_params({"normalization": "specular"})["normalization_percentile"] == 95.
    rgb = np.repeat(np.arange(1., 101.)[None, :, None], 3, axis=2)
    original, white = render_linear_srgb(rgb)
    legacy, legacy_white = render_linear_srgb(rgb, {"normalization": "auto99", "normalization_percentile": 1.})
    np.testing.assert_array_equal(legacy, original)
    assert legacy_white == white
    equivalent, pwhite = render_linear_srgb(rgb, {"normalization": "percentile", "normalization_percentile": 99.})
    np.testing.assert_array_equal(equivalent, original)
    assert pwhite == white
    for mode in ("percentile", "specular"):
        black, bwhite = render_linear_srgb(np.zeros((4, 4, 3)), {"normalization": mode})
        assert not black.any() and bwhite == 1e-6


def test_specular_direct_panel_uses_exact_profile_and_display_recipe():
    params = {"normalization": "specular", "normalization_percentile": 80.}
    result = preview_sensor_samples(params=params)
    output, white = render_linear_srgb(np.asarray(result["linear_srgb"])[None, ...], params)
    assert result["output_display_rgb8"] == output[0].tolist()
    assert result["display_white"] == white
    assert result["display_transform"]["specular_clip_mode"] is True


def test_profiled_hdr_round_trip_preserves_retouch_pixels(tmp_path):
    raw = np.random.default_rng(30).uniform(0, 5, (512, 512, 3)).astype(np.float32)
    profiled, _ = sensor_to_linear_srgb(raw)
    expected, _ = render_linear_srgb(profiled)
    path = tmp_path / "exposure.npz"
    np.savez_compressed(path, linear_rgb_exposure=raw, linear_srgb_exposure=profiled)
    with np.load(path, allow_pickle=False) as saved:
        actual, _ = render_linear_srgb(saved["linear_srgb_exposure"])
    np.testing.assert_array_equal(actual, expected)
