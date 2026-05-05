"""Unified ray-tracing package skeleton.

This package is the intended landing zone for a backend-agnostic transport
layer that can dispatch the same canonical job model to either the CPU kernel
stack or the OpenGL/GLSL stack.

Associated repository files
---------------------------
Unified transport and bridge surfaces:
- ray_tracer_bridge.py
- acoustic_fdtd_bridge.py
- csrc/include/ray_tracer.h
- csrc/kernels/ray_tracer.cpp
- csrc/kernels/rt_field_solver.cpp
- csrc/bindings/pybind_kernels.cpp

Current OpenGL/GLSL ownership:
- demo_pluck_gl.py
- camera_designer_station.py
- csrc/shaders/coherent_accumulate.comp.glsl
- opengl_widget.py

Camera objects, meshes, big-world placement, and UI:
- camera_item.py
- camera_panel.py
- player_controller.py
- placed_object.py
- room_workspace.py
- room_geometry.py

Camera optics, backs, manifolds, and LUT-like noodle tables:
- camera_software/base.py
- camera_software/lens_manifold.py
- camera_software/camera_back.py
- camera_software/eye_geometry.py
- camera_designer/camera_preset.py
- camera_designer/bake_worker.py

Material and profile databases used by rasterizing and ray tracing:
- material_db.py
- spectral_material.py
- guitar_part.py
- camera_designer/scene_builder.py
- source_presets/
- presets/
- configs/

The initial modules are intentionally skeleton-only so the package can become
the single unifying bridge without changing runtime behavior yet.
"""

from .unified import (
    BackendCaps,
    BackendKind,
    CameraSpec,
    CanonicalScene,
    FilmSpec,
    MaterialTable,
    OutputRequest,
    PlanDecision,
    SourceTable,
    TraceJob,
    TraceMode,
    TransportBackend,
    TransportBridge,
)
from .scene_io import (
    SCENE_EXTENSION_KEY,
    SCENE_METADATA_NS,
    SceneCoordinator,
    SceneDecodeResult,
    SceneDocument,
)
from .preset_survey import (
    PresetFileRecord,
    PresetSurvey,
    PresetSurveyCoordinator,
)
from .station_entry import (
    BackendMode,
    LightSimSettings,
    RayTracerStationCoordinator,
    SceneInsertSpec,
    StationSceneState,
)
from .station_specs import (
    LayerMesh,
    StationControlNode,
    StationHierarchyRecord,
    StationHierarchyTable,
    StationMaterialSlot,
    StationPrebakePlan,
    build_paper_layer_meshes,
    build_station_hierarchy,
    default_station_material_slots,
    make_prebake_plan,
)
# Removed in favor of the root-level `controls` module, which is now the
# exclusive home for KnobSpec/Panel-based control hierarchy and the action
# registry/dispatcher. Import from `controls` directly.

__all__ = [
    "BackendCaps",
    "BackendKind",
    "CameraSpec",
    "CanonicalScene",
    "FilmSpec",
    "MaterialTable",
    "OutputRequest",
    "PlanDecision",
    "SourceTable",
    "TraceJob",
    "TraceMode",
    "TransportBackend",
    "TransportBridge",
    "SCENE_EXTENSION_KEY",
    "SCENE_METADATA_NS",
    "SceneCoordinator",
    "SceneDecodeResult",
    "SceneDocument",
    "PresetFileRecord",
    "PresetSurvey",
    "PresetSurveyCoordinator",
    "BackendMode",
    "LightSimSettings",
    "RayTracerStationCoordinator",
    "SceneInsertSpec",
    "StationSceneState",
    "LayerMesh",
    "StationControlNode",
    "StationHierarchyRecord",
    "StationHierarchyTable",
    "StationMaterialSlot",
    "StationPrebakePlan",
    "build_paper_layer_meshes",
    "build_station_hierarchy",
    "default_station_material_slots",
    "make_prebake_plan",
]