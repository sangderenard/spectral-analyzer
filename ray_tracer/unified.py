"""Canonical transport model skeleton.

This module defines the shape of a unified view where CPU tracing and GLSL
tracing are treated as peer backends over the same job contract.

Associated repository files
---------------------------
Scene extraction and CPU bridge:
- ray_tracer_bridge.py
- acoustic_fdtd_bridge.py
- csrc/include/ray_tracer.h
- csrc/kernels/ray_tracer.cpp
- csrc/kernels/rt_field_solver.cpp

GLSL transport, volume, and sensor logic currently embedded in app code:
- demo_pluck_gl.py
- camera_designer_station.py
- opengl_widget.py

Camera systems and world-mounted camera meshes:
- camera_item.py
- camera_panel.py
- player_controller.py
- placed_object.py

Camera optics, backs, manifolds, and LUT/noodle tables:
- camera_software/base.py
- camera_software/lens_manifold.py
- camera_software/camera_back.py
- camera_software/eye_geometry.py
- camera_designer/camera_preset.py
- camera_designer/bake_worker.py

Material presets and profile databases:
- material_db.py
- spectral_material.py
- source_presets/
- presets/
- configs/

The canonical split intended here is:
1. Canonical scene and camera descriptors.
2. Backend capability and planning metadata.
3. Backend-specific preparation and execution hooks.
4. Normalized outputs returned to app-facing renderers and tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class BackendKind(str, Enum):
    CPU = "cpu"
    GLSL = "glsl"
    HYBRID = "hybrid"


class TraceMode(str, Enum):
    SEGMENTS = "segments"
    SURFACE_FLUX = "surface_flux"
    CAMERA_IMAGE = "camera_image"
    SENSOR_RENDER = "sensor_render"
    COHERENT_SENSOR = "coherent_sensor"
    VOLUME_FIELD = "volume_field"
    MULTISCALE = "multiscale"
    HYBRID_CROSSOVER = "hybrid_crossover"
    PREPARE_ONLY = "prepare_only"
    PREVIEW = "preview"


@dataclass(slots=True)
class MaterialTable:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceTable:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CameraSpec:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FilmSpec:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CanonicalScene:
    geometry: dict[str, Any] = field(default_factory=dict)
    materials: MaterialTable = field(default_factory=MaterialTable)
    sources: SourceTable = field(default_factory=SourceTable)
    media: dict[str, Any] = field(default_factory=dict)
    scale_contexts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutputRequest:
    segments: bool = False
    surface_flux: bool = False
    camera_image: bool = False
    sensor_field: bool = False
    volume_field: bool = False
    diagnostics: bool = True


@dataclass(slots=True)
class BackendCaps:
    backend: BackendKind
    supports_segments: bool = False
    supports_surface_flux: bool = False
    supports_camera_image: bool = False
    supports_sensor_render: bool = False
    supports_coherent_sensor: bool = False
    supports_volume_field: bool = False
    supports_multiscale: bool = False
    supports_lens_manifold: bool = False
    supports_nonplanar_back: bool = False
    supports_progressive: bool = False


@dataclass(slots=True)
class TraceJob:
    mode: TraceMode
    scene: CanonicalScene
    camera: CameraSpec | None = None
    film: FilmSpec | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    outputs: OutputRequest = field(default_factory=OutputRequest)
    schedule: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlanDecision:
    backend: BackendKind
    mode: TraceMode
    degraded_features: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class TransportBackend(Protocol):
    kind: BackendKind

    def capabilities(self) -> BackendCaps:
        ...

    def prepare(self, job: TraceJob) -> Any:
        ...

    def execute(self, job: TraceJob, prepared: Any) -> dict[str, Any]:
        ...


class TransportBridge:
    """Planner and dispatcher skeleton for unified transport jobs.

    Intended responsibilities:
    - canonicalize app-facing scene, camera, and film state into TraceJob
    - compare TraceJob requirements against backend capability tables
    - select CPU, GLSL, or a composed hybrid plan
    - prepare backend-owned resources
    - execute and normalize results into one output shape
    """

    def __init__(self, backends: list[TransportBackend] | None = None) -> None:
        self.backends = list(backends or [])

    def register_backend(self, backend: TransportBackend) -> None:
        self.backends.append(backend)

    def plan(self, job: TraceJob) -> PlanDecision:
        raise NotImplementedError

    def prepare(self, job: TraceJob, plan: PlanDecision | None = None) -> Any:
        raise NotImplementedError

    def dispatch(self, job: TraceJob, prepared: Any | None = None) -> dict[str, Any]:
        raise NotImplementedError
