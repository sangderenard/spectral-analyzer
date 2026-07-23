from __future__ import annotations

import numpy as np
import pytest

from camera_software.optical_bench import (
    BellJarWorkspace,
    BenchViewMode,
    OpticalBenchJob,
    OpticalTransportMode,
    WorkspaceMapStack,
    camera_bell_jar_scene_manifest,
)
from camera_software.camera_manifest import default_camera_manifest
from room_workspace import RoomWorkspace


def test_bell_jar_defaults_to_camera_scale_orthographic_room():
    workspace = BellJarWorkspace.create(
        bench_id="lens-table",
        scene_manifest={"apparatus": [{"kind": "lens"}]},
    )

    assert workspace.view.mode is BenchViewMode.ORTHOGRAPHIC_TOP_DOWN
    assert workspace.view.visible_layers["apparatus"] is True
    assert workspace.view.visible_layers["materials"] is True
    assert workspace.view.visible_layers["complex_field"] is False
    assert workspace.room_cfg["workspace_kind"] == "optical_bell_jar"
    assert workspace.room_cfg["boundary"]["material"] == "borosilicate_glass"
    assert workspace.player_eye_height_m == pytest.approx(0.14)
    assert workspace.job.material_library_key == "canonical-optical-materials"
    assert workspace.work_asset_manifest()["units"] == "metres"


def test_bell_jar_layers_are_presentation_only():
    workspace = BellJarWorkspace.create(bench_id="field-table")
    before = workspace.job.cache_identity

    assert workspace.view.toggle_layer("complex_field") is True
    workspace.view.mode = BenchViewMode.WALK

    assert workspace.job.cache_identity == before
    with pytest.raises(KeyError):
        workspace.view.toggle_layer("invented-layer")


def test_optical_job_cache_tracks_physics_not_mapping_order():
    a = OpticalBenchJob(
        bench_id="prism",
        scene_manifest={"b": 2, "a": 1},
        transport_mode=OpticalTransportMode.MIXED,
        contexts=({"id": "wave-1", "transport": "wave"},),
    )
    b = OpticalBenchJob(
        bench_id="prism",
        scene_manifest={"a": 1, "b": 2},
        transport_mode=OpticalTransportMode.MIXED,
        contexts=({"transport": "wave", "id": "wave-1"},),
    )
    changed = OpticalBenchJob(
        bench_id="prism",
        scene_manifest={"a": 1, "b": 3},
        transport_mode=OpticalTransportMode.MIXED,
        contexts=({"id": "wave-1", "transport": "wave"},),
    )

    assert a.cache_identity == b.cache_identity
    assert a.cache_identity != changed.cache_identity


def test_ray_only_job_rejects_wave_context():
    with pytest.raises(ValueError, match="ray-only"):
        OpticalBenchJob(
            bench_id="invalid",
            scene_manifest={},
            transport_mode=OpticalTransportMode.RAY,
            contexts=({"transport": "wave"},),
        )


def test_default_camera_is_opened_verbatim_as_wave_bench_apparatus():
    workspace = BellJarWorkspace.for_camera()
    apparatus = workspace.job.scene_manifest["apparatus"][0]
    camera = apparatus["camera_manifest"]

    assert workspace.job.transport_mode is OpticalTransportMode.WAVE
    assert apparatus["kind"] == "canonical_physical_camera"
    assert camera["asset_key"] == "camera/6x6/four-group/default"
    assert camera["sensor"]["physical_width_mm"] == 56.0
    assert camera["sensor"]["physical_height_mm"] == 56.0
    assert apparatus["camera_manifest_hash"] == camera["identity"]["manifest_hash"]


def test_editing_bell_jar_camera_re_resolves_and_invalidates_preparation():
    workspace = BellJarWorkspace.for_camera()
    old_job_key = workspace.job.cache_identity
    old_apparatus = workspace.job.scene_manifest["apparatus"][0]
    authored = default_camera_manifest()
    authored["lens"]["glass"] = "N-SF11"
    authored["focus"]["distance_m"] = 2.0

    resolved = workspace.set_camera_manifest(authored)
    new_apparatus = workspace.job.scene_manifest["apparatus"][0]

    assert resolved.mapping()["lens"]["glass"] == "N-SF11"
    assert resolved.mapping()["focus"]["distance_m"] == 2.0
    assert new_apparatus["preparation_cache_key"] != old_apparatus["preparation_cache_key"]
    assert workspace.job.cache_identity != old_job_key
    assert workspace._dirty is True


def test_pulled_resolved_camera_can_be_edited_without_stale_build_products():
    scene = camera_bell_jar_scene_manifest()
    pulled = scene["apparatus"][0]["camera_manifest"]
    pulled["lens"]["f_number"] = 8.0
    pulled["lens"]["aperture_diameter_mm"] = 10.3125

    rebuilt = camera_bell_jar_scene_manifest(pulled)
    camera = rebuilt["apparatus"][0]["camera_manifest"]

    assert camera["lens"]["f_number"] == 8.0
    assert camera["lens"]["aperture_diameter_mm"] == 10.3125
    assert rebuilt["apparatus"][0]["camera_manifest_hash"] != scene["apparatus"][0]["camera_manifest_hash"]


class _FakePlayer:
    def __init__(self):
        self._walk_pos = np.asarray([4.0, 5.0, 0.0], np.float64)
        self._walk_yaw = 73.0
        self._walk_pitch = -4.0
        self._eye_height = 1.65
        self._move_speed = 2.5
        self._floor_z = 0.0
        self._active_station = object()
        self._focus_target = object()
        self.state = "walk"
        self.applied_rooms = []
        self.camera_updates = 0

    def set_room_cfg(self, cfg):
        self.applied_rooms.append(cfg)

    def _update_walk_camera(self):
        self.camera_updates += 1


def test_map_stack_replaces_room_and_restores_player_exactly():
    root = RoomWorkspace.blank()
    root.room_cfg = {
        "workspace_kind": "game_room",
        "dimensions": {"width_m": 12.0, "depth_m": 10.0, "height_m": 4.0},
    }
    bell_jar = BellJarWorkspace.create(bench_id="camera-station")
    player = _FakePlayer()
    original_position = player._walk_pos.copy()
    stack = WorkspaceMapStack(root)

    stack.enter_bell_jar(bell_jar, player)

    assert stack.current_workspace is bell_jar
    assert stack.depth == 1
    assert np.array_equal(player._walk_pos, bell_jar.player_spawn_m)
    assert player._eye_height == bell_jar.player_eye_height_m
    assert player._move_speed == bell_jar.player_move_speed_m_s
    assert player._active_station is None
    assert player.applied_rooms[-1]["workspace_kind"] == "optical_bell_jar"

    restored = stack.exit_workspace(player)

    assert restored is root
    assert stack.depth == 0
    assert np.array_equal(player._walk_pos, original_position)
    assert player._walk_yaw == 73.0
    assert player._walk_pitch == -4.0
    assert player._eye_height == 1.65
    assert player._move_speed == 2.5
    assert player.state == "walk"
    assert player.applied_rooms[-1]["workspace_kind"] == "game_room"
    assert player.camera_updates == 1


def test_map_stack_cannot_exit_its_root():
    stack = WorkspaceMapStack(RoomWorkspace.blank())
    with pytest.raises(RuntimeError, match="already at its root"):
        stack.exit_workspace()
