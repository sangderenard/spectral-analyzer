"""Manifest defaults and strict partial resolution for program equipment."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping


EQUIPMENT_SCHEMA_VERSION = 1

EQUIPMENT_FIELDS: dict[str, tuple[str, ...]] = {
    "camera": (
        "sensor_id", "shutter_s", "yaw_offset_deg", "pitch_offset_deg",
        "height_offset_m", "sideways_offset_m",
    ),
    "lens": ("focal_length_mm", "f_number"),
    "light": ("flash_mode", "lighting_ev"),
    "film": ("film_id", "iso"),
    "integrator": (
        "transport_mode", "lane_count", "total_rays", "max_sensor_epochs",
        "epoch_bundle_count", "sensor_samples_per_node",
        "sensor_t5_pair_budget", "max_bounces",
    ),
    "exposure": (
        "allocation_mode", "grid_mode", "subdivision_axis",
        "locked_grid_columns", "locked_grid_rows", "work_width_px",
        "work_height_px", "final_edge_px",
    ),
}


def default_equipment_manifest() -> dict[str, Any]:
    """Return the fully authored equipment state matching current behavior."""

    from camera_software.ray_trace_settings import RayTraceSettings
    from exposure_control_toolbar import ExposureControlSettings

    current = {
        **asdict(RayTraceSettings().validated()),
        **asdict(ExposureControlSettings().validated()),
    }
    return {
        "schema_version": EQUIPMENT_SCHEMA_VERSION,
        **{
            group: {field: current[field] for field in fields}
            for group, fields in EQUIPMENT_FIELDS.items()
        },
    }


def _root_payload(manifest: Any) -> Mapping[str, Any]:
    if manifest is None:
        return {}
    payload = getattr(manifest, "payload", manifest)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("program manifest payload must be a mapping")
    return payload


def authored_equipment(manifest: Any) -> dict[str, dict[str, Any]]:
    """Validate and return only fields actually authored by this manifest."""

    raw = _root_payload(manifest).get("equipment", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("manifest equipment must be a mapping")
    schema_version = int(raw.get("schema_version", EQUIPMENT_SCHEMA_VERSION))
    if schema_version != EQUIPMENT_SCHEMA_VERSION:
        raise ValueError(
            f"equipment schema_version must be {EQUIPMENT_SCHEMA_VERSION}"
        )
    unknown_groups = sorted(
        set(raw) - set(EQUIPMENT_FIELDS) - {"schema_version"}
    )
    if unknown_groups:
        raise ValueError(f"unknown equipment groups: {unknown_groups}")
    result: dict[str, dict[str, Any]] = {}
    for group, fields in EQUIPMENT_FIELDS.items():
        values = raw.get(group)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ValueError(f"equipment.{group} must be a mapping")
        unknown = sorted(set(values) - set(fields))
        if unknown:
            raise ValueError(
                f"unknown equipment.{group} settings: {unknown}"
            )
        result[group] = dict(values)
    return result


def resolve_equipment_settings(
    manifest: Any,
    *,
    ray_fallback: Any = None,
    exposure_fallback: Any = None,
) -> tuple[Any, Any]:
    """Resolve current defaults < partially authored manifest equipment."""

    from camera_software.ray_trace_settings import RayTraceSettings
    from exposure_control_toolbar import ExposureControlSettings

    ray = (ray_fallback or RayTraceSettings()).validated()
    exposure = (exposure_fallback or ExposureControlSettings()).validated()
    authored = authored_equipment(manifest)
    ray_values = asdict(ray)
    exposure_values = asdict(exposure)
    for group in ("camera", "lens", "light", "film"):
        exposure_values.update(authored.get(group, {}))
    integrator = authored.get("integrator", {})
    ray_values.update(integrator)
    exposure_group = authored.get("exposure", {})
    for key, value in exposure_group.items():
        if key in ray_values:
            ray_values[key] = value
        else:
            exposure_values[key] = value
    return (
        RayTraceSettings(**ray_values).validated(),
        ExposureControlSettings(**exposure_values).validated(),
    )


__all__ = [
    "EQUIPMENT_SCHEMA_VERSION", "EQUIPMENT_FIELDS",
    "authored_equipment", "default_equipment_manifest",
    "resolve_equipment_settings",
]
