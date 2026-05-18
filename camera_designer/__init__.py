"""camera_designer — parametric optical design toolkit.

Submodules
----------
parametric_surfaces   Closed-form surface primitives (CPU + GLSL)
camera_preset         CameraPreset dataclass + built-in presets
bake_worker           64-bit ray tracer → LensManifold noodle LUT
emitter_profile       Physical emitter profiles (spectral, phase, directional,
                      polarization, UV texture)
"""
from .parametric_surfaces import (
    ParametricSurface,
    FlatSurface,
    SphericalSurface,
    ConicSurface,
    ApertureStop,
    PolynomialSurface,
    LensMountRing,
    SensorSurface,
    ToricSurface,
    surface_from_dict,
)
from .camera_preset import (
    GlassSpec,
    LensElement,
    LensGroup,
    BodySpec,
    EmitterSpec,
    CameraPreset,
    PRESET_REGISTRY,
    simple_doublet_preset,
    eye_model_preset,
    GLASS_CATALOG,
)
from .emitter_profile import (
    SpectralModel,
    SpectralDistribution,
    CoherenceModel,
    PhaseState,
    DirectionalModel,
    AngularDistribution,
    PolarizationMode,
    PolarizationState,
    EmissiveTexture,
    EmitterProfile,
    EMITTER_CATALOG,
    TEXTURE_CHANNEL_MAP,
    emitter_from_dict,
)
from .bake_worker import BakeWorker, trace_ray, trace_ray_backward
from .manifold_endpoint import ManifoldEndpoint
from .ray_order import TracerGap, SourceRecord, RayOrder

__all__ = [
    # surfaces
    "ParametricSurface", "FlatSurface", "SphericalSurface", "ConicSurface",
    "ApertureStop", "PolynomialSurface", "LensMountRing", "SensorSurface",
    "ToricSurface", "surface_from_dict",
    # preset
    "GlassSpec", "LensElement", "LensGroup", "BodySpec", "EmitterSpec",
    "CameraPreset", "PRESET_REGISTRY", "GLASS_CATALOG",
    "simple_doublet_preset", "eye_model_preset",
    # emitter profiles
    "SpectralModel", "SpectralDistribution",
    "CoherenceModel", "PhaseState",
    "DirectionalModel", "AngularDistribution",
    "PolarizationMode", "PolarizationState",
    "EmissiveTexture", "TEXTURE_CHANNEL_MAP",
    "EmitterProfile", "EMITTER_CATALOG", "emitter_from_dict",
    # bake
    "BakeWorker", "trace_ray", "trace_ray_backward", "ManifoldEndpoint",
    # ray order
    "TracerGap", "SourceRecord", "RayOrder",
]
