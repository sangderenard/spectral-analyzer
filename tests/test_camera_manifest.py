import copy

import numpy as np

from camera_software.camera_manifest import (
    camera_compatibility,
    default_camera_manifest,
    resolve_camera_manifest,
    sample_spectral_model,
)
from camera_software.camera_preparation import CameraPreparationKey


def test_manifest_is_human_readable_and_deterministically_hashed():
    first = resolve_camera_manifest(
        {"manifest": default_camera_manifest()},
        {"width": 500, "height": 500, "region": {"x": 200, "y": 200, "width": 100, "height": 100}},
        {"intensity_scale": 1.0},
    )
    second = resolve_camera_manifest(
        {"manifest": copy.deepcopy(default_camera_manifest())},
        {"height": 500, "width": 500, "region": {"height": 100, "width": 100, "y": 200, "x": 200}},
        {"intensity_scale": 1.0},
    )

    assert first.key == "camera/6x6/four-group/default"
    assert first.hash == second.hash
    assert first.mapping()["lens"]["focal_length_mm"] == 82.5
    assert first.compatibility_key("optical_geometry")
    assert CameraPreparationKey.from_manifest(first).cache_key.startswith("sha256:")
    sensor = first.mapping()["sensor"]
    assert sensor["film_format_key"] == "120_6x6"
    assert sensor["mount_standard"] == "120_6x6"
    assert sensor["physical_width_mm"] == sensor["physical_height_mm"] == 56.0
    assert sensor["image_circle_diameter_mm"] == 80.0
    assert sensor["default_work_raster_px"] == [256, 256]
    assert sensor["default_final_raster_px"] == [1024, 1024]
    assert np.allclose(sensor["raster_sample_pitch_mm"], [0.112, 0.112])
    assert sensor["raster_pixel_aspect_ratio"] == 1.0
    assert np.allclose([
        sensor["crop_physical_mm"][key]
        for key in ("x", "y", "width", "height")
    ], [22.4, 22.4, 11.2, 11.2])


def test_six_by_six_gate_is_inside_the_authored_image_circle():
    sensor = resolve_camera_manifest().mapping()["sensor"]
    corner_diameter = float(np.hypot(
        sensor["physical_width_mm"], sensor["physical_height_mm"]
    ))

    assert sensor["image_circle_diameter_mm"] >= corner_diameter


def test_unspecified_custom_gate_grows_its_default_image_circle():
    sensor = resolve_camera_manifest({
        "sensor_w_mm": 96.0,
        "sensor_h_mm": 121.0,
    }).mapping()["sensor"]

    assert sensor["image_circle_diameter_mm"] >= np.hypot(96.0, 121.0)


def test_compatibility_facets_do_not_invalidate_unrelated_camera_parts():
    base = resolve_camera_manifest(
        {"manifest": default_camera_manifest()},
        {"width": 500, "height": 500, "region": {"x": 0, "y": 0, "width": 100, "height": 100}},
    )
    recrop = resolve_camera_manifest(
        {"manifest": default_camera_manifest()},
        {"width": 500, "height": 500, "region": {"x": 100, "y": 100, "width": 200, "height": 200}},
    )

    assert base.hash != recrop.hash
    assert base.compatibility_key("sensor_geometry") != recrop.compatibility_key("sensor_geometry")
    assert base.compatibility_key("optical_geometry") == recrop.compatibility_key("optical_geometry")
    assert base.compatibility_key("spectral_grid") == recrop.compatibility_key("spectral_grid")
    decision = camera_compatibility(base, recrop)
    assert decision.update_class == "film_stage_or_sensor_update"
    assert decision.reuse_bvh and decision.reuse_optical_payload


def test_flash_change_reuses_camera_geometry_and_optical_payload_cache():
    first = resolve_camera_manifest(
        {"manifest": default_camera_manifest()}, flash={"intensity_scale": 1.0}
    )
    second = resolve_camera_manifest(
        {"manifest": default_camera_manifest()}, flash={"intensity_scale": 3.0}
    )
    first_key = CameraPreparationKey.from_manifest(first)
    second_key = CameraPreparationKey.from_manifest(second)

    assert first.hash != second.hash
    assert first_key.cache_key == second_key.cache_key
    assert first_key.optical_payload_cache_key == second_key.optical_payload_cache_key


def test_focus_change_reuses_identity_but_not_stale_rear_group_geometry():
    first_camera = {"manifest": default_camera_manifest(), "focus_distance_m": 1.0}
    second_camera = {"manifest": default_camera_manifest(), "focus_distance_m": 2.0}
    first = resolve_camera_manifest(first_camera)
    second = resolve_camera_manifest(second_camera)

    decision = camera_compatibility(first, second)
    assert decision.update_class == "rear_group_focus_geometry_rebuild"
    assert not decision.reuse_bvh
    assert not decision.reuse_optical_payload


def test_legacy_camera_values_become_a_real_physical_lens_request():
    manifest = resolve_camera_manifest(
        {"focal_mm": 50.0, "aperture_mm": 25.0},
        {"width": 64, "height": 64},
    ).mapping()

    assert manifest["lens"]["focal_length_range_mm"] == [50.0, 50.0]
    assert manifest["lens"]["focal_length_mm"] == 50.0
    assert manifest["lens"]["f_number"] == 2.0


def test_continuous_blackbody_is_resolved_onto_transport_bands():
    wavelengths = [400.0, 500.0, 600.0, 700.0]
    values = sample_spectral_model(
        {"kind": "blackbody", "temperature_k": 5600.0}, wavelengths
    )

    assert len(values) == len(wavelengths)
    assert np.isclose(max(values), 1.0)
    assert all(np.isfinite(values))
    assert len(set(round(value, 8) for value in values)) > 1


def test_default_flash_records_resolved_band_samples_and_transport_truth():
    manifest = resolve_camera_manifest(
        {"manifest": default_camera_manifest()},
        wavelengths_nm=[700.0, 600.0, 500.0, 400.0],
    ).mapping()

    spectral = manifest["spectral"]
    samples = manifest["flash"]["resolved_spectral_samples"]
    assert spectral["transport_mode"] == "one_discrete_wavelength_per_path"
    assert spectral["lane_semantics"] == "fixed_spectral"
    assert spectral["transport_abi_version"] == 1
    assert "continuous_wavelength_transport" not in spectral
    assert spectral["active_band_count"] == 4
    assert samples["wavelengths_nm"] == [700.0, 600.0, 500.0, 400.0]
    assert len(samples["relative_power"]) == 4
