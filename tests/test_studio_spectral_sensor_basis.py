"""Regression tests for separating material transport from camera response."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from camera_designer.spectral_sensor_basis import (
    FIXED3_WAVELENGTHS_NM, fixed3_response_matrix, uses_fixed3_transport,
)
from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene


def specs(width=.06, centers=(650., 550., 450.)):
    return [dict(center_hz=2.998e17 / c, width_oct=width, gain=1.) for c in centers]


def test_sensor_matrix_is_linear_and_overlap_is_not_a_material_property():
    incident = np.array([2., 3., 5.])
    absorption = np.array([.1, .4, .9])
    physical = incident * absorption
    narrow = fixed3_response_matrix(specs())
    broad = fixed3_response_matrix(specs(.3))
    np.testing.assert_allclose(narrow @ physical, sum(
        narrow[:, i] * physical[i] for i in range(3)))
    assert np.all(np.diag(narrow) == 1.)
    assert broad[0, 1] > narrow[0, 1]
    # The same absorption precedes either response matrix. Mixing first and
    # treating filter channels as wavelengths is demonstrably different.
    assert not np.allclose(broad @ physical, (broad @ incident) * absorption)


def test_transport_opt_in_does_not_depend_on_display_tones_or_film_centers():
    assert uses_fixed3_transport(SimpleNamespace(transport_wavelengths_nm=FIXED3_WAVELENGTHS_NM))
    assert not uses_fixed3_transport(SimpleNamespace())
    assert not uses_fixed3_transport(SimpleNamespace(transport_wavelengths_nm=(587.56,)))


def test_saved_emitter_preserves_source_spectrum_in_opt_in_mode():
    scene = (
        np.zeros((0, 16), np.float32), np.zeros((0, 16), np.float32),
        np.zeros((0, 12), np.float32), np.zeros((0, 3, 3), np.float32),
        np.zeros((0, 8), np.float32), {},
        (np.full(3, -.1, np.float32), np.full(3, .1, np.float32)),
        np.zeros((0, 12), np.float32),
    )
    vertices = np.array([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]], np.float32)
    pbr = np.zeros((1, 16), np.float32)
    pbr[0, 6:8] = [1., 1.]
    pbr[0, 8:11] = [4., 2., .5]
    output = append_saved_studio_mesh_to_gpu_scene(
        scene, vertices, np.array([0]), pbr, scale_m=.01, center_m=[0, 0, 0],
        wavelengths_um=[.450, .550, .650], spectral_emission_rgb=True,
    )
    shade = output[1]
    np.testing.assert_allclose(shade[0, 8:11] * shade[0, 12], pbr[0, 8:11])


def test_actual_shaders_share_material_spectrum_and_integrate_film_last():
    source = (Path(__file__).parents[1] / "demo_pluck_gl.py").read_text(encoding="utf8")
    assert "return studio_material_weight(albedo, freq_hz);" in source
    assert "return studio_material_weight(albedo, studio_band_frequency(i));" in source
    assert "layer_splat(studio_band_index(freq), q, val)" in source
    assert "measured += lum[band] * exp(" in source
    assert "studio_emission_weight(surf_albedo, studio_band_frequency(i))" in source
