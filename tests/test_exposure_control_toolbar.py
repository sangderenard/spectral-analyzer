import copy

import pytest

from exposure_control_toolbar import ExposureControlSettings
from live_spectral_text_demo import build_calibration_render_order
from scene_orders import resolved_jobs, validate_order


def test_exposure_controls_apply_sensor_film_and_photographic_settings():
    order = build_calibration_render_order("prism-room")
    settings = ExposureControlSettings(
        sensor_id=1,
        film_id=1,
        iso=800,
        shutter_s=1.0 / 250.0,
        flash_mode="on",
        lighting_ev=2,
    )

    settings.apply_to_order(order)
    validate_order(order)
    job = resolved_jobs(order, "live_paragraph")[0]

    assert job["exposure"]["iso"] == 800.0
    assert job["exposure"]["time_s"] == pytest.approx(1.0 / 250.0)
    assert job["exposure"]["sensor_id"] == 1
    assert job["exposure"]["film_id"] == 1
    assert job["flash"]["enabled"] is True
    assert job["flash"]["intensity_scale"] == 4.0


def test_scene_flash_mode_preserves_authored_enable_state():
    order = build_calibration_render_order("prism-room")
    original = copy.deepcopy(order["defaults"]["flash"])

    ExposureControlSettings(
        flash_mode="scene", lighting_ev=-1
    ).apply_to_order(order)

    assert order["defaults"]["flash"]["enabled"] == original["enabled"]
    assert order["defaults"]["flash"]["intensity_scale"] == 0.5


def test_light_ev_scales_the_authored_prism_beam_spectrally():
    import numpy as np
    import exposure_render_demo as exposure
    import scene_orders as orders

    base = exposure._build_thick_lens_lab_tracer_scene()
    normal_order = build_calibration_render_order("prism-room")
    bright_order = copy.deepcopy(normal_order)
    ExposureControlSettings(lighting_ev=1).apply_to_order(bright_order)
    normal_job = resolved_jobs(normal_order, "live_paragraph")[0]
    bright_job = resolved_jobs(bright_order, "live_paragraph")[0]

    normal_scene, _ = orders.compile_job(base, normal_job)
    bright_scene, _ = orders.compile_job(base, bright_job)

    assert np.sum(bright_scene.src_emit_W) == pytest.approx(
        2.0 * np.sum(normal_scene.src_emit_W)
    )
    normal_rows = normal_scene.mat_buf.reshape(normal_scene.mat_n_mats, 32, 12)
    bright_rows = bright_scene.mat_buf.reshape(bright_scene.mat_n_mats, 32, 12)
    normal_source_mats = normal_scene.mat_idx[normal_scene.src_tri_idx]
    bright_source_mats = bright_scene.mat_idx[bright_scene.src_tri_idx]
    assert np.array_equal(normal_source_mats, bright_source_mats)
    assert np.allclose(
        bright_rows[bright_source_mats, :, 5],
        2.0 * normal_rows[normal_source_mats, :, 5],
    )


@pytest.mark.parametrize("field,value", (("sensor_id", 2), ("film_id", -1)))
def test_exposure_controls_reject_unknown_database_slots(field, value):
    with pytest.raises(ValueError):
        ExposureControlSettings(**{field: value}).validated()


def test_live_raster_controls_are_explicit_and_positive():
    settings = ExposureControlSettings().validated()
    assert (settings.work_width_px, settings.work_height_px) == (256, 256)
    assert settings.final_edge_px == 1024
    assert settings.allocation_mode == "even-sensor"
    assert settings.targeted_fraction() == 0.0

    with pytest.raises(ValueError):
        ExposureControlSettings(final_edge_px=0).validated()


def test_sensor_allocation_modes_resolve_to_native_scheduler_splits():
    adaptive = ExposureControlSettings(allocation_mode="focus-explore")
    even = ExposureControlSettings(allocation_mode="even-sensor")
    preview = ExposureControlSettings(allocation_mode="n-tree-preview")

    assert adaptive.targeted_fraction(0.7) == pytest.approx(0.7)
    assert even.targeted_fraction(0.7) == 0.0
    assert preview.targeted_fraction(0.7) == 1.0

    order = build_calibration_render_order("prism-room")
    even.apply_to_order(order)
    assert order["runtime"]["sensor_allocation_mode"] == "even-sensor"

    with pytest.raises(ValueError):
        ExposureControlSettings(allocation_mode="neural-magic").validated()


def test_lens_row_updates_focal_length_and_physical_aperture():
    order = build_calibration_render_order("prism-room")
    ExposureControlSettings(
        focal_length_mm=50.0, f_number=2.0
    ).apply_to_order(order)

    camera = order["defaults"]["camera"]
    assert camera["focal_mm"] == 50.0
    assert camera["aperture_mm"] == 25.0
    assert camera["manifest"]["lens"]["focal_length_mm"] == 50.0
    assert camera["manifest"]["lens"]["f_number"] == 2.0
    assert camera["manifest"]["lens"]["aperture_diameter_mm"] == 25.0


def test_camera_row_applies_local_translation_and_two_angle_offsets():
    import numpy as np

    order = build_calibration_render_order("prism-room")
    original = order["defaults"]["camera"]
    original_position = np.asarray(original["position_m"], np.float64)
    original_target = np.asarray(original["target_m"], np.float64)
    original_distance = np.linalg.norm(original_target - original_position)

    ExposureControlSettings(
        yaw_offset_deg=10.0,
        pitch_offset_deg=-5.0,
        height_offset_m=0.25,
        sideways_offset_m=-0.50,
    ).apply_to_order(order)

    camera = order["defaults"]["camera"]
    position = np.asarray(camera["position_m"], np.float64)
    target = np.asarray(camera["target_m"], np.float64)
    assert not np.allclose(position, original_position)
    assert not np.allclose(target - position, original_target - original_position)
    assert np.linalg.norm(target - position) == pytest.approx(original_distance)
    assert np.linalg.norm(np.asarray(camera["up"], np.float64)) == pytest.approx(1.0)
