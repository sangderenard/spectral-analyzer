from __future__ import annotations

import pytest

from camera_software.ray_trace_settings import (
    RayTraceSettings,
    resolve_ray_trace_settings,
)
from ray_trace_toolbar import RayTraceToolbar


def test_toolbar_defaults_are_balanced_single_band_continuous() -> None:
    settings = RayTraceSettings()
    assert settings.transport_mode == "continuous"
    assert settings.lane_count == 1
    assert settings.total_rays == 204_800
    assert settings.max_sensor_epochs == 1
    assert settings.epoch_bundle_count == 16
    assert settings.sensor_samples_per_node == 1_024
    assert settings.sensor_t5_pair_budget == 67_108_864
    assert settings.max_bounces == 16
    budget = settings.render_budget(sensor_top_k=200)
    assert budget["sensor_samples_per_node"] == 1_024
    assert budget["sensor_top_k"] * budget["sensor_samples_per_node"] == 204_800


def test_toolbar_exposes_high_upper_limits() -> None:
    toolbar = RayTraceToolbar()
    assert max(toolbar._ray_options) == 67_108_864
    assert max(toolbar._epoch_options) == 128
    assert max(toolbar._bundle_options) == 256
    assert max(toolbar._sample_options) == 4_096
    assert max(toolbar._t5_options) == 268_435_456
    assert max(toolbar._bounce_options) == 128


def test_depth_is_a_toolbar_mode_with_continuous_transport() -> None:
    settings = RayTraceSettings(transport_mode="depth", lane_count=1).validated()
    assert settings.transport_option == "continuous:1"
    toolbar = RayTraceToolbar(settings)
    assert toolbar._toggle_transport() == "ray-settings-changed"
    assert toolbar.settings.transport_mode == "continuous"


def test_user_toolbar_overrides_defaults() -> None:
    resolved = resolve_ray_trace_settings(RayTraceSettings(
        transport_mode="continuous",
        lane_count=1,
        total_rays=4096,
        max_sensor_epochs=2,
        sensor_samples_per_node=1,
        sensor_t5_pair_budget=65536,
        max_bounces=2,
    ))
    assert resolved.transport_option == "continuous:1"
    assert resolved.render_budget()["total_rays"] == 4096
    assert resolved.render_budget()["max_bounces"] == 2


def test_total_rays_count_camera_impact_samples_not_flash_support() -> None:
    settings = RayTraceSettings(
        total_rays=1_048_576,
        max_sensor_epochs=4,
        sensor_samples_per_node=2,
    )
    budget = settings.render_budget(sensor_top_k=200)
    assert budget["sensor_top_k"] == 200
    assert budget["sensor_samples_per_node"] == 656
    assert budget["max_sensor_epochs"] == 8
    camera_samples = (
        budget["sensor_top_k"]
        * budget["sensor_samples_per_node"]
        * budget["max_sensor_epochs"]
    )
    assert camera_samples >= settings.total_rays
    assert camera_samples - settings.total_rays < 200 * budget["max_sensor_epochs"]


def test_large_logical_epoch_uses_complete_mip_steps_within_lineage_arena() -> None:
    budget = RayTraceSettings(
        total_rays=4_194_304,
        max_sensor_epochs=2,
        sensor_samples_per_node=2_048,
    ).render_budget(sensor_top_k=1_024)

    assert budget["sensor_samples_per_node"] == 256
    assert budget["sensor_steps_per_layer"] == 1
    assert budget["max_sensor_epochs"] == 16
    assert budget["sensor_t5_pair_budget"] == 8_388_608
    assert (
        budget["sensor_top_k"]
        * budget["sensor_samples_per_node"]
        * budget["max_sensor_epochs"]
    ) == 4_194_304


def test_epoch_bundles_are_separate_from_inner_epochs() -> None:
    settings = RayTraceSettings(
        total_rays=4_096,
        max_sensor_epochs=1,
        epoch_bundle_count=32,
    )
    budget = settings.render_budget(sensor_top_k=64)
    assert budget["max_sensor_epochs"] == 1
    assert budget["epoch_bundle_count"] == 32
    assert budget["total_rays"] == 4_096


def test_high_ray_presets_page_flash_support_inside_native_record_caps() -> None:
    budget = RayTraceSettings(
        total_rays=16_777_216,
        max_sensor_epochs=1,
        epoch_bundle_count=16,
        max_bounces=32,
    ).render_budget(sensor_top_k=200)
    assert budget["total_rays"] == 16_777_216
    assert budget["max_sensor_epochs"] == 129
    assert budget["sensor_flash_rays"] == 8_129
    assert budget["sensor_flash_page_count"] == 1
    assert budget["sensor_flash_total_rays"] == 8_129
    assert budget["epoch_bundle_count"] == 16


@pytest.mark.parametrize("lane_count, expected_flash, expected_epochs", (
    (8, 24_386, 43), (16, 12_193, 86), (32, 6_133, 171),
))
def test_fixed_spectral_flash_pages_account_for_record_lane_width(
    lane_count, expected_flash, expected_epochs,
) -> None:
    budget = RayTraceSettings(
        transport_mode="fixed",
        lane_count=lane_count,
        total_rays=4_194_304,
        max_bounces=16,
    ).render_budget(sensor_top_k=256)
    assert budget["sensor_flash_rays"] == expected_flash
    assert budget["max_sensor_epochs"] == expected_epochs
    assert budget["sensor_flash_page_count"] == 1
    assert budget["sensor_flash_total_rays"] == expected_flash
    assert budget["epoch_bundle_count"] == 16


def test_explicit_manifest_fields_override_only_their_user_fields() -> None:
    user = RayTraceSettings(lane_count=3, total_rays=4096, max_bounces=2)
    resolved = resolve_ray_trace_settings(user, {
        "scene": {"ray_trace_settings": {"lane_count": 8, "max_bounces": 16}}
    })
    assert resolved.lane_count == 8
    assert resolved.max_bounces == 16
    assert resolved.total_rays == 4096


def test_scene_transport_metadata_is_not_an_implicit_override() -> None:
    user = RayTraceSettings(transport_mode="continuous", lane_count=1)
    resolved = resolve_ray_trace_settings(user, {
        "transport": {"domain": "fixed_spectral", "lane_count": 8}
    })
    assert resolved.transport_option == "continuous:1"


def test_invalid_mode_lane_pair_is_rejected() -> None:
    with pytest.raises(ValueError):
        RayTraceSettings(transport_mode="fixed", lane_count=2).validated()
