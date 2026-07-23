from __future__ import annotations

import numpy as np
import pytest
from pxr import Usd

from camera_software.usd_material_bridge import author_material_library
from placed_object import PlacedDutyStation, PlacedEnclosure
from pluck_scene_usd import workspace_from_usd, write_workspace_usd_package
from room_workspace import RoomWorkspace


def test_authored_initial_map_retains_room_meta_authority():
    workspace = workspace_from_usd()

    assert workspace.room_authority_id in workspace.objects
    authority = workspace.objects[workspace.room_authority_id]
    assert authority.station_type == "room_control"
    assert authority.config_dir == "configs/duty_stations/room_control"
    assert workspace.movement_policy == "room-authority-bounded"
    assert workspace.movement_enforced is False
    assert len(workspace.duty_stations()) == 3
    assert len(workspace.enclosures()) == 1
    assert len(workspace.cameras()) == 1


def test_workspace_usd_roundtrip_references_components_and_material_authority(tmp_path):
    workspace = RoomWorkspace.blank()
    workspace.add_object(PlacedDutyStation(
        obj_id="room-authority",
        label="Room Authority",
        pos=np.asarray([0.0, 0.0, 0.0]),
        station_type="room_control",
        config_dir="configs/duty_stations/room_control",
        material_bindings={"body": "basic_paneling", "screen": "basic_led_display"},
    ))
    workspace.add_object(PlacedEnclosure(
        obj_id="test-jar",
        label="Test Jar",
        pos=np.asarray([1.0, 2.0, 0.5]),
        shape="rect",
        dims={"width_m": 1.0, "depth_m": 0.8, "height_m": 0.7},
        material_bindings={"glass": "borosilicate_glass"},
    ))

    path = write_workspace_usd_package(
        workspace, str(tmp_path / "initial_map.usda")
    )
    stage = Usd.Stage.Open(path)
    world = stage.GetPrimAtPath("/PluckWorld")
    assert str(world.GetAttribute("pluck:sceneRole").Get()) == "general-initial-map"
    assert world.GetRelationship("pluck:roomAuthority").GetTargets()
    jar = stage.GetPrimAtPath("/PluckWorld/Objects/test_jar")
    assert jar.GetMetadata("references") is not None

    material = stage.GetPrimAtPath("/PluckMaterials/Materials/borosilicate_glass")
    assert str(material.GetAttribute("pluck:authority").Get()) == (
        "canonical-project-material"
    )
    assert str(material.GetAttribute("pluck:projectionKind").Get()) == (
        "portable-preview"
    )
    assert material.GetAttribute("pluck:materialRevision").Get()

    restored = workspace_from_usd(path)
    assert restored.room_authority_id == "room-authority"
    assert restored.objects["test-jar"].material_bindings == {
        "glass": "borosilicate_glass"
    }
    np.testing.assert_allclose(restored.objects["test-jar"].pos, [1.0, 2.0, 0.5])


def test_structural_material_keeps_engine_requirement_beside_portable_projection(
    tmp_path,
):
    material_dir = tmp_path / "materials"
    material_dir.mkdir()
    (material_dir / "structural_blue.yaml").write_text(
        """
name: structural_blue
opacity: 1.0
pbr:
  albedo: [0.1, 0.25, 0.8]
  roughness: 0.18
  metallic: 0.0
maxwell_patch:
  asset: patches/structural_blue.usda
  solver_context: maxwell
""".strip(),
        encoding="utf-8",
    )
    stage = Usd.Stage.CreateInMemory()
    authored = author_material_library(
        stage,
        ["structural_blue"],
        material_directory=str(material_dir),
        asset_anchor_path=str(tmp_path / "scene.usda"),
    )
    prim = authored["structural_blue"].GetPrim()
    assert list(prim.GetAttribute("pluck:engineRequirements").Get()) == [
        "maxwell_patch"
    ]
    assert str(prim.GetAttribute("pluck:authority").Get()) == (
        "canonical-project-material"
    )
    assert stage.GetPrimAtPath(
        "/PluckMaterials/Materials/structural_blue/PreviewSurface"
    )


def test_usd_publication_rejects_physical_bodies_without_material_authority(
    tmp_path,
):
    workspace = RoomWorkspace.blank()
    workspace.add_object(PlacedDutyStation(
        obj_id="room-authority",
        label="Room Authority",
        station_type="room_control",
        config_dir="configs/duty_stations/room_control",
        material_bindings={"body": "basic_paneling", "screen": "basic_led_display"},
    ))
    workspace.add_object(PlacedEnclosure(
        obj_id="unbound-jar",
        label="Unbound Jar",
        shape="rect",
    ))

    with pytest.raises(ValueError, match="unbound-jar:glass"):
        write_workspace_usd_package(
            workspace, str(tmp_path / "invalid.usda")
        )
