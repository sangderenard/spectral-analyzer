"""Live PNG opacity follows actual sensor sampling, never brightness."""
import importlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from camera_designer.marble_preview_worker import _snapshot
from camera_designer.marble_color_profile import sensor_to_linear_srgb, render_linear_srgb


def readback(monkeypatch, raw, alpha, *, live=True):
    gl = importlib.import_module("OpenGL.GL")
    bound = [0]
    for name in ("GL_TEXTURE_2D", "GL_RGBA", "GL_FLOAT"):
        monkeypatch.setattr(gl, name, 0, raising=False)
    monkeypatch.setattr(gl, "glFinish", lambda: None, raising=False)
    monkeypatch.setattr(gl, "glBindTexture", lambda target, texture: bound.__setitem__(0, texture), raising=False)
    def get_image(*args):
        index = bound[0] - 1
        pixels = np.repeat(raw[:, :, index, None], 4, axis=2)
        pixels[:, :, 3] = alpha[:, :, index]
        return pixels.astype(np.float32).tobytes()
    monkeypatch.setattr(gl, "glGetTexImage", get_image, raising=False)
    return SimpleNamespace(_textures=[1, 2, 3], _w=raw.shape[1], _h=raw.shape[0],
                           _history_samples=4 if live else 0)


@pytest.mark.parametrize("normalization", ["auto99", "manual", "percentile", "specular"])
def test_live_coverage_preserves_display_rgb_and_sampled_black(tmp_path, monkeypatch, normalization):
    raw = np.array([[[0, 0, 0], [0, 0, 0]], [[1, 2, 3], [3, 1, 2]]], np.float32)
    alpha = np.ones((2, 2, 3), np.float32)
    alpha[0, 1] = 0  # Same black RGB, but this pixel has never been sampled.
    alpha[1, 1, 2] = 0  # Do not reveal a pixel missing a film-layer sample.
    acc = readback(monkeypatch, raw, alpha)
    params = dict(tint=1.25, normalization=normalization, normalization_percentile=94,
                  normalization_white=3)
    status = _snapshot(acc, tmp_path, "a" * 64, 1, 1, 0, 8192, 4,
                       transport="fixed:3", display_params=params)
    image = Image.open(tmp_path / status["image"])
    assert image.mode == "RGBA"
    pixels = np.asarray(image)
    expected, white = render_linear_srgb(sensor_to_linear_srgb(raw)[0], {"gamma": 1, **params})
    np.testing.assert_array_equal(pixels[:, :, :3], expected)
    np.testing.assert_array_equal(pixels[:, :, 3], [[255, 0], [255, 0]])
    assert status["sampled_pixels"] == 2 and status["coverage_fraction"] == .5
    assert status["display_white"] == white


def test_full_coverage_then_reset_removes_old_view_alpha(tmp_path, monkeypatch):
    raw = np.ones((2, 2, 3), np.float32)
    alpha = np.ones_like(raw)
    acc = readback(monkeypatch, raw, alpha)
    first = _snapshot(acc, tmp_path, "a" * 64, 1, 1, 0, 8192, 4, transport="fixed:3")
    assert first["coverage_fraction"] == 1
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / first["image"]))[:, :, 3], 255)
    # Real SensorAccumulator.clear() zeroes every RGBA texture. The next
    # camera has only one sampled black pixel; old bright pixels stay clear.
    raw[:] = 0
    alpha[:] = 0
    alpha[0, 0] = 1
    second = _snapshot(acc, tmp_path, "a" * 64, 2, 2, 0, 16384, 5, transport="fixed:3")
    assert second["sampled_pixels"] == 1 and second["coverage_fraction"] == .25
    pixels = np.asarray(Image.open(tmp_path / second["image"]))
    np.testing.assert_array_equal(pixels[:, :, :3], 0)
    np.testing.assert_array_equal(pixels[:, :, 3], [[255, 0], [0, 0]])


def test_still_png_remains_rgb_even_when_texture_alpha_is_zero(tmp_path, monkeypatch):
    raw = np.array([[[1, 2, 3], [0, 0, 0]]], np.float32)
    acc = readback(monkeypatch, raw, np.zeros_like(raw), live=False)
    status = _snapshot(acc, tmp_path, "a" * 64, 1, 1, 0, 8192, 2, transport="fixed:3")
    image = Image.open(tmp_path / status["image"])
    assert image.mode == "RGB" and "coverage_fraction" not in status
    expected, _ = render_linear_srgb(sensor_to_linear_srgb(raw)[0], {"gamma": 1})
    np.testing.assert_array_equal(np.asarray(image), expected)


def test_live_snapshot_can_leave_atomic_status_publication_to_owner(tmp_path, monkeypatch):
    raw = np.zeros((1, 1, 3), np.float32)
    acc = readback(monkeypatch, raw, np.zeros_like(raw))
    status = _snapshot(acc, tmp_path, "a" * 64, 1, 1, 0, 8192, 0,
                       transport="fixed:3", publish_status=False)
    assert not (tmp_path / "status.json").exists()
    assert (tmp_path / status["image"]).is_file()
    assert status["sampled_pixels"] == 0
    assert np.asarray(Image.open(tmp_path / status["image"]))[0, 0, 3] == 0
