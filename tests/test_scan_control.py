import numpy as np
import pytest

from camera_software.scan_control import (
    CameraScanController,
    CoverageBalanceController,
    FrameDeltaMetrics,
    FilmPlaneAdjustment,
    NextSiteScan,
    UvRefinementRequest,
    FocusCalibrationController,
    raytraced_focus_score,
)
from camera_software.progressive_exposure import SensorPixelSlice
from camera_software.sensor_mipmap import SensorUvBounds


def test_uv_request_rasterizes_region_without_quantizing_its_address():
    command = NextSiteScan(
        sequence=1,
        uv_requests=(UvRefinementRequest(
            SensorUvBounds(0.501, 0.247, 0.509, 0.261), 6, 4.0
        ),),
    )
    priority = command.priority_map(200, 200)
    assert priority.max() == 4.0
    assert np.count_nonzero(priority) > 0
    assert np.count_nonzero(priority) < priority.size


def test_film_plane_adjustment_is_bounded_and_sequence_ordered():
    controller = CameraScanController(
        maximum_film_travel_m=1.0e-3,
        maximum_film_tilt_deg=2.0,
    )
    command = NextSiteScan(
        sequence=4,
        film_plane_adjustment=FilmPlaneAdjustment(
            lens_distance_delta_m=0.2e-3,
            tilt_about_right_deg=1.0,
            tilt_about_up_deg=-0.5,
        ),
    )
    assert controller.validate(command) == command.film_plane_adjustment
    with pytest.raises(ValueError, match="strictly increasing"):
        controller.validate(command)


@pytest.mark.parametrize(
    "adjustment, message",
    [
        (FilmPlaneAdjustment(lens_distance_delta_m=1.1e-3), "film travel"),
        (FilmPlaneAdjustment(tilt_about_up_deg=2.1), "film tilt"),
    ],
)
def test_film_plane_adjustment_rejects_motion_over_camera_limit(adjustment, message):
    controller = CameraScanController(
        maximum_film_travel_m=1.0e-3,
        maximum_film_tilt_deg=2.0,
    )
    with pytest.raises(ValueError, match=message):
        controller.validate(NextSiteScan(sequence=1, film_plane_adjustment=adjustment))


def test_frame_delta_metrics_measure_change_without_becoming_training_reference():
    previous = np.zeros((8, 8, 3), dtype=np.float32)
    current = previous.copy()
    current[3:5, 3:5] = 1.0
    metrics = FrameDeltaMetrics.between(previous, current)
    assert metrics.root_mean_square > metrics.mean_absolute > 0.0
    assert 0.0 < metrics.changed_fraction < 1.0


def test_coverage_balance_moves_compute_to_coating_and_recovers_smoothly():
    controller = CoverageBalanceController(
        preferred_targeted=0.75, minimum_targeted=0.20, response=1.0
    )
    sparse = np.zeros((10, 10), dtype=np.float32)
    sparse[:2] = 100.0
    assert controller.update(sparse, 1.0) == pytest.approx(0.20)
    uniform = np.full((10, 10), 100.0, dtype=np.float32)
    assert controller.update(uniform, 0.1) == pytest.approx(0.75)


def test_next_scan_film_command_moves_only_sensor_as_a_rigid_plane():
    from exposure_render_demo import TracerScene, apply_next_site_scan

    sensor = np.asarray([[3.0, -1.0, -1.0], [3.0, 1.0, -1.0], [3.0, -1.0, 1.0]])
    subject = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    triangles = np.stack([sensor, subject])
    scene = TracerScene(
        verts=triangles.reshape(2, 9),
        normals=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        mat_idx=np.asarray([0, 0], np.int32),
        mat_buf=np.zeros((1, 12), np.float32),
        mat_n_mats=1,
        src_pos=np.empty((0, 3)), src_dir=np.empty((0, 3)),
        src_directivity=np.empty(0), src_area_m2=np.empty(0),
        src_emit_W=np.empty(0), src_emit_rgb_W=np.empty((0, 3)),
        src_tri_idx=np.empty(0, np.int32),
        bounds_min=triangles.min(axis=(0, 1)).astype(np.float32),
        bounds_max=triangles.max(axis=(0, 1)).astype(np.float32),
        camera_tri_groups={"sensor": np.asarray([0], np.int32)},
    )
    command = NextSiteScan(
        sequence=1,
        film_plane_adjustment=FilmPlaneAdjustment(
            lens_distance_delta_m=0.2e-3,
            tilt_about_right_deg=1.0,
            tilt_about_up_deg=-0.5,
        ),
    )
    result = apply_next_site_scan(scene, command)
    pose = result.film_plane_pose
    assert pose is not None
    assert pose.center[0] == pytest.approx(sensor.mean(axis=0)[0] + 0.2e-3)
    assert np.dot(pose.right, pose.up) == pytest.approx(0.0, abs=1.0e-12)
    assert np.linalg.norm(pose.right) == pytest.approx(1.0)
    assert np.linalg.norm(pose.up) == pytest.approx(1.0)
    assert np.allclose(result.verts.reshape(-1, 3, 3)[1], subject)
    film_delta = result.verts.reshape(-1, 3, 3)[0] - pose.center
    assert np.max(np.abs(film_delta @ pose.normal_to_scene)) < 1.0e-12
    assert np.all(np.isfinite(result.normals))


def test_focus_trials_are_full_coverage_absolute_commands():
    controller = FocusCalibrationController(first_sequence=7)
    adjustment = FilmPlaneAdjustment(lens_distance_delta_m=0.1e-3)
    command = controller.request(adjustment, metadata={"policy": "test-network"})
    assert command.sequence == 7
    assert command.targeted_fraction == 0.0
    assert command.film_plane_adjustment is adjustment
    assert command.metadata["film_pose_reference"] == "machined_zero"
    assert command.metadata["policy"] == "test-network"


def test_focus_objective_uses_only_raytraced_frame_sharpness():
    blurred = np.zeros((32, 32), np.float32)
    blurred[:, 13:19] = np.asarray([0.2, 0.5, 0.8, 0.8, 0.5, 0.2])
    sharp = np.zeros((32, 32), np.float32)
    sharp[:, 15:17] = 1.0
    blurred_score, _ = raytraced_focus_score(blurred)
    sharp_score, confidence = raytraced_focus_score(sharp)
    assert sharp_score > blurred_score
    assert confidence > 0.0
    controller = FocusCalibrationController()
    pose = FilmPlaneAdjustment()
    result = controller.observe(pose, sharp)
    assert result == controller.best
def test_next_scan_pixel_slice_marks_only_arbitrary_requested_sites():
    command = NextSiteScan.from_mapping({
        "sequence": 1,
        "pixel_slice_requests": [{
            "width": 4,
            "height": 3,
            "site_indices": [0, 6, 11],
            "target_level": 3,
            "work_value": 2.0,
        }],
    })
    priority = command.priority_map(3, 4)

    assert np.flatnonzero(priority).tolist() == [0, 6, 11]
    assert np.all(priority.reshape(-1)[[0, 6, 11]] == 2.0)


def test_active_work_bounds_union_regions_and_arbitrary_site_slices():
    command = NextSiteScan.from_mapping({
        "sequence": 1,
        "uv_requests": [{
            "uv_bounds": [0.0, 0.0, 0.25, 0.25],
            "target_level": 2,
        }],
        "pixel_slice_requests": [{
            "width": 8,
            "height": 8,
            "site_indices": [45, 63],
            "target_level": 3,
        }],
    })

    assert command.active_uv_bounds() == (0.0, 0.0, 1.0, 1.0)
