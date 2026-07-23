from dataclasses import fields

import pytest

from camera_software.control_layout import program_ui_manifest
from camera_software.equipment_manifest import (
    EQUIPMENT_FIELDS,
    authored_equipment,
    resolve_equipment_settings,
)
from camera_software.ray_trace_settings import RayTraceSettings
from exposure_control_toolbar import ExposureControlSettings


def test_default_program_manifest_authors_every_equipment_setting():
    manifest = program_ui_manifest()
    authored = authored_equipment(manifest)
    authored_fields = {
        field for values in authored.values() for field in values
    }
    expected = {
        field.name for field in fields(RayTraceSettings)
    } | {
        field.name for field in fields(ExposureControlSettings)
    }

    assert authored_fields == expected
    ray, exposure = resolve_equipment_settings(manifest)
    assert ray == RayTraceSettings().validated()
    assert exposure == ExposureControlSettings().validated()


def test_partial_manifest_overlays_only_authored_equipment_details():
    ray, exposure = resolve_equipment_settings({
        "equipment": {
            "lens": {"f_number": 8.0},
            "arena": {"lane_count": 16, "wave_mode": True},
            "exposure": {
                "allocation_mode": "focus-explore",
                "final_edge_px": 2048,
            },
        }
    })

    assert exposure.f_number == 8.0
    assert exposure.focal_length_mm == 82.5
    assert exposure.allocation_mode == "focus-explore"
    assert exposure.final_edge_px == 2048
    assert exposure.iso == 100
    assert ray.lane_count == 16
    assert ray.wave_mode is True
    assert ray.transport_mode == "continuous"
    assert ray.total_rays == 204_800


def test_equipment_manifest_rejects_misspelled_groups_and_fields():
    with pytest.raises(ValueError, match="unknown equipment groups"):
        resolve_equipment_settings({"equipment": {"lenz": {}}})
    with pytest.raises(ValueError, match="unknown equipment.lens settings"):
        resolve_equipment_settings({
            "equipment": {"lens": {"fstop": 4.0}}
        })


def test_legacy_integrator_lane_fields_migrate_to_arena_settings():
    ray, _ = resolve_equipment_settings({
        "equipment": {"integrator": {"lane_count": 8}}
    })
    assert ray.lane_count == 8


def test_equipment_field_partition_has_no_duplicate_ownership():
    all_fields = [field for group in EQUIPMENT_FIELDS.values() for field in group]
    assert len(all_fields) == len(set(all_fields))
