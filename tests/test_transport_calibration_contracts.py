from __future__ import annotations

import pytest

from camera_software.calibration_modes import (
    CALIBRATION_MODES,
    calibration_mode,
    run_calibration_validators,
    calibration_bootstrap_plan,
)
from camera_software.transport_contract import (
    SpectralLaneDescriptor,
    TransportDomain,
    TransportLaneTable,
    TransportWorkContract,
    achromatic_lane_table,
    rgb_lane_table,
    SpectralLookupCache,
    continuous_lut_lane_table,
    fixed_visible_lane_table,
    perceptual_visible_bins,
    perceptual_visible_wavelengths,
)


def test_lane_identity_is_separate_from_fixed_frequency_bin_semantics() -> None:
    table = continuous_lut_lane_table(
        "epoch-7", (4.0e14, 5.0e14, 7.0e14), (0.25, 1.0, 0.1), 1,
    )

    restored = TransportLaneTable.from_mapping(table.mapping())
    assert restored == table
    assert restored.compatibility_key == table.compatibility_key
    assert restored.lanes[0].lut_index == 0
    assert restored.lanes[0].frequency_hz is None


def test_compact_payload_variants_cover_scalar_rgb_and_spectral_work() -> None:
    assert TransportWorkContract(achromatic_lane_table(), payload_variant=1).payload_variant == 1
    assert TransportWorkContract(rgb_lane_table(), payload_variant=3).payload_variant == 3
    with pytest.raises(ValueError, match="cannot hold"):
        TransportWorkContract(rgb_lane_table(), payload_variant=1)


def test_continuous_lanes_signal_lut_and_resolution_is_cached_per_ray_state() -> None:
    kwargs = {
        "cohort_id": "exposure-17",
        "frequency_knots_hz": (4.0e14, 5.0e14, 7.0e14),
        "proposal_density": (0.25, 1.0, 0.10),
        "lane_count": 8,
        "seed": 91,
    }

    first = continuous_lut_lane_table(**kwargs)
    second = continuous_lut_lane_table(**kwargs)

    assert first == second
    assert first.domain is TransportDomain.CONTINUOUS_SPECTRAL_LUT
    assert all(lane.frequency_hz is None and lane.lut_index == 0 for lane in first.lanes)
    cache = SpectralLookupCache(first.lookup_tables)
    resolved = cache.resolve(first.lanes[0].lut_index, 0.37123456789)
    again = cache.resolve(first.lanes[0].lut_index, 0.37123456789)
    assert resolved is again
    assert 4.0e14 < resolved.frequency_hz < 7.0e14
    assert resolved.sampling_pdf > 0.0
    assert cache.cached_resolution_count == 1


@pytest.mark.parametrize("lane_count", (1, 3, 4, 8, 16, 32))
def test_fixed_visible_lanes_are_perceptual_and_have_coloured_sensor_weights(
    lane_count,
) -> None:
    table = fixed_visible_lane_table(f"fixed-{lane_count}", lane_count)
    wavelengths = perceptual_visible_wavelengths(lane_count)
    assert len(table.lanes) == lane_count
    assert tuple(sorted(wavelengths)) == wavelengths
    assert 380.0 < wavelengths[0] <= wavelengths[-1] < 700.0
    assert all(lane.frequency_hz and lane.frequency_hz > 0.0 for lane in table.lanes)
    assert all(lane.sensor_weight_xyz != (1.0, 1.0, 1.0) for lane in table.lanes)
    if lane_count >= 8:
        dominant_channels = {
            max(range(3), key=lane.sensor_weight_xyz.__getitem__)
            for lane in table.lanes
        }
        assert dominant_channels == {0, 1, 2}
    bins = perceptual_visible_bins(lane_count)
    assert bins[0][0] == pytest.approx(380.0)
    assert bins[-1][2] == pytest.approx(700.0)
    assert sum(lane.quadrature_weight for lane in table.lanes) == pytest.approx(
        float(lane_count)
    )
    assert TransportLaneTable.from_mapping(table.mapping()) == table


def test_calibration_modes_are_off_by_default_and_validators_pass() -> None:
    assert CALIBRATION_MODES[0].key == "off"
    assert all(not mode.enabled_by_default for mode in CALIBRATION_MODES)
    assert "continuous-8" not in {mode.key for mode in CALIBRATION_MODES}
    assert not any(mode.key.startswith("fixed-") for mode in CALIBRATION_MODES)
    for key in (
        "color-science", "glass", "focus-hall", "depth", "prism-room",
        "mirror-box",
        "single-lane-ui",
    ):
        mode = calibration_mode(key)
        results = run_calibration_validators(key)
        assert results
        assert all(result.passed for result in results), (key, results)
        assert mode.work_asset_manifest()["transport"] is not None
    depth_scene = calibration_mode("depth").scene_manifest
    assert depth_scene["gpu_required"] is True
    assert depth_scene["scene_bounces"] == 1
    assert depth_scene["camera_optics_traversed"] is True
    mirror = calibration_mode("mirror-box")
    assert mirror.transport.payload_variant == 32
    assert mirror.transport.lane_table.active_lane_count == 32
    assert mirror.scene_manifest["capacity_contract"]["constant_stride_records"] is True


def test_camera_bootstrap_orders_validation_before_ui_harvest() -> None:
    plan = calibration_bootstrap_plan()
    assert [stage.order for stage in plan] == list(range(1, len(plan) + 1))
    assert [stage.mode_key for stage in plan[:3]] == [
        "color-science", "glass", "focus-hall"
    ]
    assert plan[-1].enables_panel_harvest is True
    manifest = calibration_mode("camera-bootstrap").work_asset_manifest()
    assert manifest["transport"] is None
    assert len(manifest["scene"]["stages"]) == len(plan)
