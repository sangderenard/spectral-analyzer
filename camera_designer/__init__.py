"""camera_designer — parametric optical design toolkit.

Submodules
----------
compound_optics       Exact algebraic compound lens model (CompoundLens)
parametric_surfaces   Closed-form surface primitives (CPU + GLSL)
camera_preset         CameraPreset dataclass + built-in presets
emitter_profile       Physical emitter profiles (spectral, phase, directional,
                      polarization, UV texture)
wave_tube             Retired standalone ADI-BPM validation reference; import
                      explicitly when comparing historical results
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
from .compound_optics import (
    CompoundLens,
    LensHood,
    RayBundle,
    BundleTraceResult,
    RayTraceResult,
    OpticalFace,
    FaceVignettingProfile,
    BoundaryTeleportProfile,
    AssemblyFieldProfile,
    FaceAngularLimit,
    AssemblyAngularLimits,
    PLENS_MAGIC,
    PLENS_HEADER,
    PLENS_SURF_STRIDE,
)
from .ray_order import TracerGap, SourceRecord, RayOrder
from .neural_assembly import (
    NeuralAssemblyMLP,
    NormStats,
    train as train_neural_assembly,
    train_from_array as train_neural_assembly_from_array,
    export_payload as export_neural_payload,
    load_training_data,
    infer_payload,
    MAGIC_NEURAL,
)


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
    # compound optics (element types accessed via camera_designer.compound_optics)
    "CompoundLens", "LensHood",
    "RayBundle", "BundleTraceResult", "RayTraceResult",
    "OpticalFace", "FaceVignettingProfile", "BoundaryTeleportProfile",
    "AssemblyFieldProfile",
    "FaceAngularLimit", "AssemblyAngularLimits",
    "PLENS_MAGIC", "PLENS_HEADER", "PLENS_SURF_STRIDE",
    # ray order
    "TracerGap", "SourceRecord", "RayOrder",
    # neural assembly
    "NeuralAssemblyMLP", "NormStats",
    "train_neural_assembly", "train_neural_assembly_from_array",
    "export_neural_payload",
    "load_training_data", "infer_payload",
    "MAGIC_NEURAL",
]
