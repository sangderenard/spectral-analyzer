"""Duty-station entry coordinator for scene I/O, backend mode, and scene fitting.

This is the first integration surface for a center tab panel model driven by
knob-like descriptors: save/load controls, backend mode selection, and scene
insertion scaling into a bounded test rig volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .scene_io import SceneCoordinator, SceneDocument
from .station_specs import (
    StationControlNode,
    StationHierarchyTable,
    StationMaterialSlot,
    StationPrebakePlan,
    build_station_hierarchy,
    default_station_material_slots,
    make_prebake_plan,
)
from .unified import CameraSpec, CanonicalScene, FilmSpec


class BackendMode(str, Enum):
    CPU = "cpu"
    OPENGL = "opengl"


@dataclass(slots=True)
class SceneInsertSpec:
    test_rig_box_m: tuple[float, float, float] = (4.0, 2.5, 4.0)
    fit_mode: str = "uniform_fit"
    padding_ratio: float = 0.95

    @classmethod
    def knobs(cls) -> list[Any]:
        KnobSpec = _resolve_knob_spec()
        return [
            KnobSpec("test_rig_box_m.x", "Rig X", "float", 4.0, 0.1, 100.0, 0, "m", [], False, "Scene Inserter", ".3f"),
            KnobSpec("test_rig_box_m.y", "Rig Y", "float", 2.5, 0.1, 100.0, 0, "m", [], False, "Scene Inserter", ".3f"),
            KnobSpec("test_rig_box_m.z", "Rig Z", "float", 4.0, 0.1, 100.0, 0, "m", [], False, "Scene Inserter", ".3f"),
            KnobSpec("padding_ratio", "Padding", "float", 0.95, 0.1, 1.0, 0, "", [], False, "Scene Inserter", ".3f"),
            KnobSpec("fit_mode", "Fit Mode", "choice", "uniform_fit", 0, 1, 1, "", ["uniform_fit"], False, "Scene Inserter", ".0f"),
        ]


@dataclass(slots=True)
class LightSimSettings:
    backend: BackendMode = BackendMode.CPU
    enabled: bool = True

    @classmethod
    def knobs(cls) -> list[Any]:
        KnobSpec = _resolve_knob_spec()
        return [
            KnobSpec("backend", "Backend", "choice", BackendMode.CPU.value, 0, 1, 1, "", [BackendMode.CPU.value, BackendMode.OPENGL.value], False, "Light Sim", ".0f"),
            KnobSpec("enabled", "Enabled", "bool", True, 0, 1, 1, "", [], False, "Light Sim", ""),
        ]


@dataclass(slots=True)
class StationSceneState:
    default_scene_path: str
    active_scene_path: str | None = None
    active_scene: CanonicalScene | None = None
    active_camera: CameraSpec | None = None
    active_film: FilmSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    insert: SceneInsertSpec = field(default_factory=SceneInsertSpec)
    light_sim: LightSimSettings = field(default_factory=LightSimSettings)


class RayTracerStationCoordinator:
    """Coordinator used by center-panel Save/Load and test-rig insert controls."""

    def __init__(
        self,
        default_scene_path: str,
        *,
        scene_io: SceneCoordinator | None = None,
    ) -> None:
        self.scene_io = scene_io or SceneCoordinator()
        self.state = StationSceneState(default_scene_path=default_scene_path)

    def load_default_scene(self) -> StationSceneState:
        return self.load_scene(self.state.default_scene_path)

    def load_scene(self, path: str | Path) -> StationSceneState:
        decoded = self.scene_io.decode_file(path)
        self.state.active_scene_path = str(path)
        self.state.active_scene = decoded.scene
        self.state.active_camera = decoded.camera
        self.state.active_film = decoded.film
        self.state.metadata = dict(decoded.metadata)
        return self.state

    def save_scene(self, path: str | Path | None = None) -> SceneDocument:
        if self.state.active_scene is None:
            raise RuntimeError("No active scene loaded.")

        out_path = path or self.state.active_scene_path
        if out_path is None:
            raise ValueError("No output path provided and no active scene path set.")

        document = self.scene_io.encode_file(
            out_path,
            self.state.active_scene,
            camera=self.state.active_camera,
            film=self.state.active_film,
            metadata=self.state.metadata,
        )
        self.state.active_scene_path = str(out_path)
        return document

    def set_backend_mode(self, mode: BackendMode | str) -> None:
        self.state.light_sim.backend = BackendMode(mode)

    def set_insert_box(self, box_m: tuple[float, float, float], *, padding_ratio: float | None = None) -> None:
        self.state.insert.test_rig_box_m = box_m
        if padding_ratio is not None:
            self.state.insert.padding_ratio = padding_ratio

    def compute_scene_fit_scale(self) -> float:
        if self.state.active_scene is None:
            return 1.0

        bounds = self._read_scene_bounds(self.state.active_scene)
        if bounds is None:
            return 1.0

        bmin, bmax = bounds
        extents = [max(1e-9, bmax[i] - bmin[i]) for i in range(3)]
        rig = self.state.insert.test_rig_box_m
        ratios = [rig[i] / extents[i] for i in range(3)]
        scale = min(ratios) * float(self.state.insert.padding_ratio)
        return float(scale)

    def build_modular_subpanels(self) -> list[Any]:
        """Build bass_viewer-native ModularSubpanelSpec records.

        This replaces the custom tab payload schema and aligns with the
        existing modular panel system used in bass_viewer.
        """
        ModularSubpanelSpec = _resolve_modular_subpanel_spec()
        return [
            ModularSubpanelSpec(
                key="scene_io",
                title="Scene IO",
                summary_lines=[
                    f"path: {self.state.active_scene_path or self.state.default_scene_path}",
                    "actions: load_scene / save_scene",
                ],
                expanded=True,
                enabled=True,
                accent_rgb=(0, 71, 199),
                body_height=0,
                render_body=None,
                payload={
                    "knobs": self.scene_io_knobs(),
                    "actions": ["load_scene", "save_scene"],
                },
            ),
            ModularSubpanelSpec(
                key="light_sim",
                title="Light Sim",
                summary_lines=[
                    f"backend: {self.state.light_sim.backend.value}",
                    f"enabled: {self.state.light_sim.enabled}",
                ],
                expanded=True,
                enabled=True,
                accent_rgb=(110, 150, 210),
                body_height=0,
                render_body=None,
                payload={"knobs": self.light_sim_knobs(), "actions": []},
            ),
            ModularSubpanelSpec(
                key="scene_insert",
                title="Scene Inserter",
                summary_lines=[
                    f"box m: {self.state.insert.test_rig_box_m}",
                    f"fit scale: {self.compute_scene_fit_scale():.6g}",
                ],
                expanded=True,
                enabled=True,
                accent_rgb=(140, 185, 120),
                body_height=0,
                render_body=None,
                payload={"knobs": self.scene_insert_knobs(), "actions": []},
            ),
        ]

    def scene_io_knobs(self) -> list[Any]:
        KnobSpec = _resolve_knob_spec()
        return [
            KnobSpec("default_scene_path", "Default Scene", "str", self.state.default_scene_path, 0, 0, 0, "", [], False, "Scene IO", ""),
            KnobSpec("active_scene_path", "Active Scene", "str", self.state.active_scene_path or self.state.default_scene_path, 0, 0, 0, "", [], False, "Scene IO", ""),
        ]

    def light_sim_knobs(self) -> list[Any]:
        return LightSimSettings.knobs()

    def scene_insert_knobs(self) -> list[Any]:
        return SceneInsertSpec.knobs()

    def build_station_control_stack(self) -> list[StationControlNode]:
        """Build station controls as a hierarchy-ready stack with implicit z order."""
        sections: list[tuple[str, str, list[Any], str]] = [
            ("scene_io", "Scene IO", self.scene_io_knobs(), "left_physical"),
            ("light_sim", "Light Sim", self.light_sim_knobs(), "right_physical"),
            ("scene_insert", "Scene Inserter", self.scene_insert_knobs(), "center_screen"),
        ]

        nodes: list[StationControlNode] = []
        for key, label, knobs, material_slot in sections:
            children = [
                StationControlNode(
                    key=f"{key}.{getattr(k, 'name', i)}",
                    label=str(getattr(k, 'label', getattr(k, 'name', ''))),
                    knob=k,
                    children=[],
                    depth_mode="raise",
                    material_slot=material_slot,
                    payload={
                        "knob_name": getattr(k, 'name', ''),
                        "parent_key": key,
                    },
                )
                for i, k in enumerate(knobs)
            ]
            depth_mode = "raise" if material_slot != "center_screen" else "engrave"
            nodes.append(
                StationControlNode(
                    key=key,
                    label=label,
                    children=children,
                    depth_mode=depth_mode,
                    material_slot=material_slot,
                    payload={
                        "section": key,
                        "z_step_m": 0.0022 if material_slot != "center_screen" else 0.00045,
                    },
                )
            )

        return nodes

    def build_hierarchy_table(self) -> StationHierarchyTable:
        return build_station_hierarchy("duty_station.controls", self.build_station_control_stack())

    def default_material_slots(self) -> dict[str, StationMaterialSlot]:
        return default_station_material_slots()

    def build_prebake_plan(self, *, mode: str = "hud2d") -> StationPrebakePlan:
        hierarchy = self.build_hierarchy_table()
        mats = self.default_material_slots()
        return make_prebake_plan(hierarchy, mode=mode, material_slots=mats)

    @staticmethod
    def _read_scene_bounds(scene: CanonicalScene) -> tuple[list[float], list[float]] | None:
        geometry = scene.geometry
        bounds = geometry.get("bounds")
        if isinstance(bounds, dict):
            bmin = bounds.get("min")
            bmax = bounds.get("max")
            if (
                isinstance(bmin, list)
                and isinstance(bmax, list)
                and len(bmin) >= 3
                and len(bmax) >= 3
            ):
                return [float(bmin[0]), float(bmin[1]), float(bmin[2])], [
                    float(bmax[0]),
                    float(bmax[1]),
                    float(bmax[2]),
                ]

        return None


def _resolve_knob_spec() -> Any:
    try:
        from signal_generator_v2 import KnobSpec
        return KnobSpec
    except Exception:
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class KnobSpec:  # type: ignore[no-redef]
            name: str
            label: str
            dtype: str = "float"
            default: Any = None
            low: float = 0.0
            high: float = 1.0
            step: float = 0.0
            unit: str = ""
            choices: list[str] = field(default_factory=list)
            is_log: bool = False
            group: str = ""
            fmt: str = ".3g"
            source_class: str = ""
            rebuild_layout: bool = False
            visible_when: Optional[tuple[str, str]] = None

        return KnobSpec


def _resolve_modular_subpanel_spec() -> Any:
    try:
        from bass_viewer import ModularSubpanelSpec
        return ModularSubpanelSpec
    except Exception:
        from dataclasses import dataclass

        @dataclass
        class ModularSubpanelSpec:  # type: ignore[no-redef]
            key: str
            title: str
            summary_lines: list[str] = field(default_factory=list)
            expanded: bool = True
            enabled: bool = True
            accent_rgb: tuple[int, int, int] = (120, 140, 200)
            body_height: int = 0
            render_body: Any = None
            payload: Any = None

        return ModularSubpanelSpec
