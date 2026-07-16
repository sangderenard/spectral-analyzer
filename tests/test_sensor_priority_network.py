import numpy as np
import pytest

import live_spectral_text_demo as demo
import scene_orders
from camera_software.sensor_priority_network import (
    PARAMETER_COUNT,
    SensorWorkValueNet,
    render_flat_orthographic,
    train_scene_priority_network,
)


pytestmark = pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False,
    reason="priority network contract requires CUDA",
)


def _job(text: str = "Clear text"):
    package = demo.build_paragraph_order(text, display_width=96, display_height=64)
    return scene_orders.resolved_jobs(package, demo.JOB_ID)[0]


def test_cuda_orthographic_reference_contains_flat_authored_colors():
    image = render_flat_orthographic(_job(), 96, 64).cpu().numpy()
    assert image.shape == (64, 96, 3)
    assert np.unique(image.reshape(-1, 3), axis=0).shape[0] == 2
    assert np.any(np.linalg.norm(image - image[0, 0], axis=2) > 0.1)


def test_network_export_matches_fixed_glsl_abi():
    params = SensorWorkValueNet().cuda().export_glsl_parameters()
    assert params.shape == (PARAMETER_COUNT,)
    assert params.dtype == np.float32
    assert np.all(np.isfinite(params))


def test_short_training_writes_flat_reference_model_and_overlay(tmp_path):
    result = train_scene_priority_network(
        _job("AB BA"), str(tmp_path), width=48, height=32, steps=4,
    )
    for path in (
        result.model_path, result.flat_reference_path,
        result.priority_overlay_path, result.metadata_path,
    ):
        assert __import__("os").path.isfile(path)
    model = np.load(result.model_path)
    assert model["parameters"].shape == (PARAMETER_COUNT,)
    assert np.isfinite(result.final_loss)
