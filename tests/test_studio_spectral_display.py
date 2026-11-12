"""Eight-band display projection leaves captured sensor data intact."""
import numpy as np
import pytest
from types import SimpleNamespace

from camera_designer.studio_spectral_display import project_spectral_exposure
from camera_designer import marble_preview_worker as worker


def test_compressed_view_uses_each_band_and_records_the_wavelength_map():
    raw = np.eye(8, dtype=np.float32)
    original = raw.copy()
    rgb, receipt = project_spectral_exposure(raw)
    assert rgb.shape == (8, 3)
    assert np.all(rgb.sum(axis=1) > 0)
    assert receipt["display_wavelengths_nm"][4] == 380.
    assert receipt["display_wavelengths_nm"][7] == 700.
    np.testing.assert_array_equal(raw, original)


def test_artist_views_remap_invisible_bands_without_changing_visible_reference():
    raw = np.eye(8, dtype=np.float32)
    visible, _ = project_spectral_exposure(raw, "visible_reference")
    ir, _ = project_spectral_exposure(raw, "nir_red")
    uv, _ = project_spectral_exposure(raw, "uv_violet")
    assert not visible[6:].any() and not visible[4:6].any()
    assert ir[6, 0] > 0 and not ir[6, 1:].any()
    assert uv[4, 2] > 0 and not uv[4, :2].any()
    with pytest.raises(ValueError):
        project_spectral_exposure(raw, "unknown")


def test_fixed8_snapshot_saves_all_planes_and_marks_false_colour(tmp_path, monkeypatch):
    from OpenGL import GL
    selected = [0]
    monkeypatch.setattr(GL, "glFinish", lambda: None, raising=False)
    monkeypatch.setattr(GL, "glBindTexture", lambda target, texture: selected.__setitem__(0, texture), raising=False)
    def pixels(*args):
        rgba = np.ones((1, 1, 4), np.float32)
        rgba[..., :3] = selected[0] + 1
        return rgba.tobytes()
    monkeypatch.setattr(GL, "glGetTexImage", pixels, raising=False)
    for key in ("GL_TEXTURE_2D", "GL_RGBA", "GL_FLOAT"):
        monkeypatch.setattr(GL, key, 0, raising=False)
    acc = SimpleNamespace(_textures=list(range(8)), _w=1, _h=1)
    status = worker._snapshot(acc, tmp_path, "test", 1, 1., 0., 1, 1,
                              finished=True, transport="fixed:8")
    assert status["color_profile"]["look"] == "compressed"
    assert "false-color" in status["hdr_channels"]
    with np.load(tmp_path / status["hdr"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["linear_spectral_exposure"][0, 0], np.arange(1, 9))
        expected, _ = project_spectral_exposure(saved["linear_spectral_exposure"])
        np.testing.assert_array_equal(saved["linear_srgb_exposure"], expected)
