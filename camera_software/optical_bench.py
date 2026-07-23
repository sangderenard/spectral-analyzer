"""Pure state contracts for the in-game bell-jar optical workbench.

The bell jar replaces the active room while it is entered.  It does not scale
the optical scene or create a private renderer: metre-valued apparatus geometry,
canonical material identities, transport contexts, and engine products retain
the same coordinates and contracts used by production rendering.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

import numpy as np

from room_workspace import RoomWorkspace

from .camera_manifest import CameraManifest, resolve_camera_manifest
from .camera_preparation import CameraPreparationKey


class BenchViewMode(str, Enum):
    ORTHOGRAPHIC_TOP_DOWN = "orthographic_top_down"
    WALK = "walk"


class OpticalTransportMode(str, Enum):
    RAY = "ray"
    WAVE = "wave"
    MIXED = "mixed"


DEFAULT_BENCH_LAYERS = (
    "apparatus",
    "materials",
    "emitters",
    "sensors",
    "ray_paths",
    "complex_field",
    "wave_contexts",
    "material_ids",
    "sampling_density",
    "processing_groups",
    "sensor_accumulation",
)


def resolve_bench_camera_manifest(
    camera: CameraManifest | Mapping[str, Any] | None = None,
) -> CameraManifest:
    """Resolve an editable camera mapping through the canonical camera schema."""

    if camera is None:
        return resolve_camera_manifest()
    if isinstance(camera, CameraManifest):
        return camera
    authored = copy.deepcopy(dict(camera))
    # These are derived build/cache products, never authoring inputs.  Removing
    # them makes editing a manifest pulled from a completed build deterministic.
    authored.pop("identity", None)
    authored.pop("resolved", None)
    return resolve_camera_manifest({"manifest": authored})


def camera_bell_jar_scene_manifest(
    camera: CameraManifest | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Place the real optical-engine camera in a metre-valued bench scene.

    This is an ownership wrapper, not another camera representation: the full
    canonical camera manifest is retained verbatim and remains the input to the
    normal thick-lens preparation path.
    """

    resolved = resolve_bench_camera_manifest(camera)
    preparation = CameraPreparationKey.from_manifest(resolved)
    return {
        "schema_version": 1,
        "scene_kind": "camera_bell_jar",
        "units": "metres",
        "apparatus": [{
            "id": "camera-under-test",
            "kind": "canonical_physical_camera",
            "transform": {
                "position_m": [0.0, 0.0, 0.0],
                "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "camera_manifest": resolved.mapping(),
            "camera_manifest_hash": resolved.hash,
            "preparation_cache_key": preparation.cache_key,
        }],
    }


@dataclass
class BellJarViewState:
    """Presentation state only; it never changes transport physics."""

    mode: BenchViewMode = BenchViewMode.ORTHOGRAPHIC_TOP_DOWN
    visible_layers: dict[str, bool] = field(default_factory=lambda: {
        "apparatus": True,
        "materials": True,
        "emitters": True,
        "sensors": True,
        "ray_paths": False,
        "complex_field": False,
        "wave_contexts": False,
        "material_ids": False,
        "sampling_density": False,
        "processing_groups": False,
        "sensor_accumulation": False,
    })

    def set_layer(self, layer: str, visible: bool) -> None:
        key = str(layer)
        if key not in DEFAULT_BENCH_LAYERS:
            raise KeyError(f"unknown optical-bench layer {key!r}")
        self.visible_layers[key] = bool(visible)

    def toggle_layer(self, layer: str) -> bool:
        key = str(layer)
        self.set_layer(key, not self.visible_layers.get(key, False))
        return self.visible_layers[key]


@dataclass(frozen=True)
class OpticalBenchJob:
    """Renderer-neutral request submitted by the bench to the optical engine."""

    bench_id: str
    scene_manifest: Mapping[str, Any]
    material_library_key: str = "canonical-optical-materials"
    transport_mode: OpticalTransportMode = OpticalTransportMode.RAY
    contexts: tuple[Mapping[str, Any], ...] = ()
    requested_products: tuple[str, ...] = (
        "surface_scan",
        "camera_geometry",
        "light_field",
        "field_accumulation",
        "sensor_accumulation",
        "transport_diagnostics",
    )
    execution_policy: str = "interactive_preview"

    def __post_init__(self) -> None:
        if not str(self.bench_id).strip():
            raise ValueError("optical bench job requires a bench_id")
        if not str(self.material_library_key).strip():
            raise ValueError("optical bench job requires a material library")
        if self.execution_policy not in {"interactive_preview", "scientific", "final"}:
            raise ValueError("unsupported optical bench execution policy")
        if self.transport_mode is OpticalTransportMode.RAY and any(
            str(ctx.get("transport", "ray")) == "wave" for ctx in self.contexts
        ):
            raise ValueError("ray-only bench job cannot contain a wave context")

    def mapping(self) -> dict[str, Any]:
        return {
            "bench_id": self.bench_id,
            "scene_manifest": copy.deepcopy(dict(self.scene_manifest)),
            "material_library_key": self.material_library_key,
            "transport_mode": self.transport_mode.value,
            "contexts": [copy.deepcopy(dict(value)) for value in self.contexts],
            "requested_products": list(self.requested_products),
            "execution_policy": self.execution_policy,
        }

    @property
    def cache_identity(self) -> str:
        canonical = json.dumps(
            self.mapping(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class BellJarWorkspace(RoomWorkspace):
    """A metre-based optical room which temporarily replaces the active map."""

    @classmethod
    def create(
        cls,
        *,
        bench_id: str,
        scene_manifest: Mapping[str, Any] | None = None,
        transport_mode: OpticalTransportMode = OpticalTransportMode.RAY,
        contexts: tuple[Mapping[str, Any], ...] = (),
        width_m: float = 2.4,
        depth_m: float = 1.8,
        height_m: float = 1.2,
        camera_body_height_m: float = 0.14,
    ) -> "BellJarWorkspace":
        if min(width_m, depth_m, height_m, camera_body_height_m) <= 0.0:
            raise ValueError("bell-jar dimensions and player height must be positive")

        ws = cls.__new__(cls)
        ws._config_dir = "configs/optical_bench"
        ws.room_cfg = {
            "workspace_kind": "optical_bell_jar",
            "dimensions": {
                "width_m": float(width_m),
                "depth_m": float(depth_m),
                "height_m": float(height_m),
                "wall_thickness_m": 0.012,
            },
            "boundary": {
                "shape": "bell_jar",
                "material": "borosilicate_glass",
                "transport_boundary": "scene-authored",
            },
            "table": {
                "height_m": 0.0,
                "material": "optical_bench_black",
            },
            "player_clip_surface_cast_enabled": True,
            "player_clip_surface_cast_distance_m": 0.08,
        }
        ws.physics_cfg = {}
        ws.station_cfg = {}
        ws._objects = {}
        ws.selected_id = None
        ws._dirty = False
        ws.bench_id = str(bench_id)
        ws.view = BellJarViewState()
        ws.player_eye_height_m = float(camera_body_height_m)
        ws.player_move_speed_m_s = max(0.05, float(camera_body_height_m) * 2.5)
        ws.player_spawn_m = np.asarray(
            [0.0, -0.42 * float(depth_m), 0.0], dtype=np.float64,
        )
        ws.job = OpticalBenchJob(
            bench_id=str(bench_id),
            scene_manifest=dict(scene_manifest or {}),
            transport_mode=transport_mode,
            contexts=tuple(contexts),
        )
        return ws

    def work_asset_manifest(self) -> dict[str, Any]:
        return {
            "object_kind": "optical_bell_jar_workspace",
            "units": "metres",
            "bench_id": self.bench_id,
            "room": copy.deepcopy(self.room_cfg),
            "view_mode": self.view.mode.value,
            "layers": dict(self.view.visible_layers),
            "player": {
                "scale_policy": "camera_body",
                "eye_height_m": self.player_eye_height_m,
                "move_speed_m_s": self.player_move_speed_m_s,
                "spawn_m": self.player_spawn_m.tolist(),
            },
            "optical_job": self.job.mapping(),
            "cache_identity": self.job.cache_identity,
        }

    @classmethod
    def for_camera(
        cls,
        *,
        bench_id: str = "camera-optical-bench",
        camera: CameraManifest | Mapping[str, Any] | None = None,
        transport_mode: OpticalTransportMode = OpticalTransportMode.WAVE,
        contexts: tuple[Mapping[str, Any], ...] = (),
        **dimensions: float,
    ) -> "BellJarWorkspace":
        """Open the canonical or edited camera as bell-jar apparatus."""

        return cls.create(
            bench_id=bench_id,
            scene_manifest=camera_bell_jar_scene_manifest(camera),
            transport_mode=transport_mode,
            contexts=contexts,
            **dimensions,
        )

    def set_camera_manifest(
        self, camera: CameraManifest | Mapping[str, Any]
    ) -> CameraManifest:
        """Replace the camera request and invalidate only its bench job identity."""

        resolved = resolve_bench_camera_manifest(camera)
        self.job = replace(
            self.job,
            scene_manifest=camera_bell_jar_scene_manifest(resolved),
        )
        self._dirty = True
        return resolved


@dataclass
class _PlayerWorkspaceSnapshot:
    walk_pos: np.ndarray
    walk_yaw: float
    walk_pitch: float
    eye_height: float
    move_speed: float
    floor_z: float
    state: Any
    room_cfg: Mapping[str, Any]


@dataclass
class _WorkspaceFrame:
    workspace: RoomWorkspace
    player: _PlayerWorkspaceSnapshot | None


class WorkspaceMapStack:
    """Push/pop map replacement with exact player-state restoration."""

    def __init__(self, workspace: RoomWorkspace):
        self.current_workspace = workspace
        self._frames: list[_WorkspaceFrame] = []

    @property
    def depth(self) -> int:
        return len(self._frames)

    @staticmethod
    def _snapshot_player(player: Any, workspace: RoomWorkspace) -> _PlayerWorkspaceSnapshot:
        return _PlayerWorkspaceSnapshot(
            walk_pos=np.asarray(player._walk_pos, np.float64).copy(),
            walk_yaw=float(player._walk_yaw),
            walk_pitch=float(player._walk_pitch),
            eye_height=float(player._eye_height),
            move_speed=float(player._move_speed),
            floor_z=float(player._floor_z),
            state=player.state,
            room_cfg=copy.deepcopy(workspace.room_cfg),
        )

    def enter_bell_jar(self, workspace: BellJarWorkspace, player: Any | None = None) -> None:
        snapshot = None if player is None else self._snapshot_player(
            player, self.current_workspace,
        )
        self._frames.append(_WorkspaceFrame(self.current_workspace, snapshot))
        self.current_workspace = workspace
        if player is not None:
            player._walk_pos = workspace.player_spawn_m.copy()
            player._walk_yaw = 0.0
            player._walk_pitch = 0.0
            player._eye_height = workspace.player_eye_height_m
            player._move_speed = workspace.player_move_speed_m_s
            player._floor_z = 0.0
            player._active_station = None
            player._focus_target = None
            player.set_room_cfg(workspace.room_cfg)

    def exit_workspace(self, player: Any | None = None) -> RoomWorkspace:
        if not self._frames:
            raise RuntimeError("workspace map stack is already at its root")
        frame = self._frames.pop()
        self.current_workspace = frame.workspace
        if player is not None and frame.player is not None:
            state = frame.player
            player._walk_pos = state.walk_pos.copy()
            player._walk_yaw = state.walk_yaw
            player._walk_pitch = state.walk_pitch
            player._eye_height = state.eye_height
            player._move_speed = state.move_speed
            player._floor_z = state.floor_z
            player.state = state.state
            player.set_room_cfg(dict(state.room_cfg))
            if hasattr(player, "_update_walk_camera"):
                player._update_walk_camera()
        return self.current_workspace


__all__ = [
    "BenchViewMode",
    "OpticalTransportMode",
    "DEFAULT_BENCH_LAYERS",
    "resolve_bench_camera_manifest",
    "camera_bell_jar_scene_manifest",
    "BellJarViewState",
    "OpticalBenchJob",
    "BellJarWorkspace",
    "WorkspaceMapStack",
]
