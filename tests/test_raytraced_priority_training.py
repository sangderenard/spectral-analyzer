import os
import inspect

import numpy as np
import pytest

from camera_software.raytraced_priority_training import (
    RaytracedSensorState,
    load_raytraced_states,
    measured_work_examples,
    normalized_camera_rgb,
    train_from_raytraced_examples,
)
import camera_software.raytraced_priority_training as ray_training
import train_raytraced_sensor_priority as trainer_cli


def _state(index: int, image: np.ndarray, weight: float) -> RaytracedSensorState:
    weights = np.full(image.shape[:2], weight, np.float32)
    return RaytracedSensorState(index, image.astype(np.float32) * weight, weights)


def test_normalized_camera_rgb_matches_runtime_sum_over_exposure():
    sums = np.asarray([[[4.0, 2.0, 1.0], [9.0, 6.0, 3.0]]], np.float32)
    weights = np.asarray([[2.0, 3.0]], np.float32)
    assert np.allclose(normalized_camera_rgb(sums, weights), [
        [[2.0, 1.0, 0.5], [3.0, 2.0, 1.0]],
    ])


def test_training_module_has_no_orthographic_preview_dependency():
    source = inspect.getsource(ray_training)
    assert "render_flat_orthographic" not in source
    assert "orthographic" not in source.lower()


def test_measured_target_marks_where_next_real_layer_improved():
    teacher = np.ones((8, 8, 3), np.float32)
    before = np.zeros_like(teacher)
    after = np.zeros_like(teacher)
    after[:, :4] = 1.0
    examples = measured_work_examples([
        _state(1, before, 1.0),
        _state(2, after, 2.0),
        _state(3, teacher, 3.0),
    ], smoothing=1)
    target = examples[0].measured_improvement
    assert target[:, :4].mean() > 0.99
    assert target[:, 4:].max() == 0.0
    assert np.array_equal(examples[0].camera_rgb, before)


def test_progress_loader_pairs_exact_sum_and_float_weight_artifacts(tmp_path):
    for index in (1, 2):
        suffix = f"{index:06d}_z00_s00.npy"
        np.save(tmp_path / f"sum_{suffix}", np.full((4, 5, 3), index, np.float32))
        np.save(tmp_path / f"weight_{suffix}", np.full((4, 5), index, np.float32))
    states = load_raytraced_states(str(tmp_path))
    assert [state.pass_index for state in states] == [1, 2]
    assert all(state.exposure_weight.dtype == np.float32 for state in states)


def test_trainer_resumes_completed_scene_without_launching_renderer(tmp_path):
    progress = tmp_path / "scene_0000" / "progress"
    progress.mkdir(parents=True)
    for index in (1, 2):
        suffix = f"{index:06d}_z00_s00.npy"
        np.save(progress / f"sum_{suffix}", np.full((4, 4, 3), index, np.float32))
        np.save(progress / f"weight_{suffix}", np.full((4, 4), index, np.float32))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    np.savez(
        model_dir / "raytraced_priority_network.npz",
        parameters=np.zeros(305, np.float32),
    )
    assert trainer_cli.main([
        "--out-dir", str(tmp_path), "--scenes", "1",
        "--width", "4", "--height", "4",
        "--min-epochs", "2", "--max-epochs", "2",
    ]) == 0


@pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False,
    reason="ray-traced priority training requires CUDA",
)
def test_short_raytraced_training_exports_runtime_model(tmp_path):
    teacher = np.ones((8, 8, 3), np.float32)
    examples = measured_work_examples([
        _state(1, np.zeros_like(teacher), 1.0),
        _state(2, teacher, 2.0),
    ], smoothing=1)
    result = train_from_raytraced_examples(examples, str(tmp_path), steps=4)
    assert os.path.isfile(result.model_path)
    archive = np.load(result.model_path)
    assert archive["parameters"].shape == (305,)
    assert np.isfinite(result.final_loss)
