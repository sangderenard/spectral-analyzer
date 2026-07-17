import numpy as np
import pytest

import live_spectral_text_demo as demo
import scene_orders
from camera_software.sensor_priority_network import (
    PARAMETER_COUNT,
    SensorWorkValueNet,
    _orthographic_scene_geometry,
    orthographic_scene_bounds,
    render_flat_orthographic,
)


pytestmark = pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False,
    reason="priority network contract requires CUDA",
)


def _job(text: str = "Clear text"):
    package = demo.build_paragraph_order(text, display_width=96, display_height=64)
    return scene_orders.resolved_jobs(package, demo.JOB_ID)[0]


def test_cuda_orthographic_reference_contains_flat_authored_colors():
    job = _job()
    image = render_flat_orthographic(job, 96, 64).cpu().numpy()
    assert image.shape == (64, 96, 3)
    for material in ("quiet_background", "text_surface"):
        color = np.asarray(job["materials"][material]["albedo_rgb"], np.float32)
        assert np.any(np.all(np.isclose(image, color, atol=1.0e-6), axis=2))


def test_orthographic_reference_uses_exact_scene_triangles_and_roi_not_text_box():
    job = _job()
    triangles, _colors, _camera, _forward, _axes = _orthographic_scene_geometry(job)
    plane = scene_orders._box_plane_triangles(job["planes"][0])
    glyph = scene_orders._glyph_triangles(job, job["planes"][0])
    assert np.array_equal(triangles, np.concatenate([plane, glyph]).astype(np.float32))

    left, right, bottom, top = orthographic_scene_bounds(job, 96, 64)
    assert np.isclose((right - left) / (top - bottom), 96.0 / 64.0)
    assert not np.isclose(right - left, job["geometry"]["text_box_m"][0])

def test_network_export_matches_fixed_glsl_abi():
    source = SensorWorkValueNet().cuda()
    params = source.export_glsl_parameters()
    assert params.shape == (PARAMETER_COUNT,)
    assert params.dtype == np.float32
    assert np.all(np.isfinite(params))
    restored = SensorWorkValueNet().cuda()
    restored.load_glsl_parameters(params)
    assert np.array_equal(restored.export_glsl_parameters(), params)
