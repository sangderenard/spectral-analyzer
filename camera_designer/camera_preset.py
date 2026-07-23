"""camera_designer/camera_preset.py
=====================================
Complete camera preset: body, mount, lens array, aperture plane, sensor.

A ``CameraPreset`` is the single serialisable object that owns:
  - the physical body geometry (dimensions, mount position)
  - the lens group: an ordered sequence of (surface, glass, thickness) elements
  - the aperture stop surface
  - the lens-mount ring (defines back-focal datum and mirror clearance)
  - the sensor surface (parametric, curved or flat)
  - camera-software metadata (focal mm, f-number, field of view)

Everything is stored in float64 and serialises to/from a plain dict that can
be written as JSON or YAML with no numpy dependencies.

The preset is the input to ``BakeWorker``, which uses the parametric surfaces'
``intersect()`` methods (64-bit precision) to trace rays and produce the noodle
LUT consumed by ``LensManifold``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .parametric_surfaces import (
    ParametricSurface,
    FlatSurface,
    ConicSurface,
    ApertureStop,
    LensMountRing,
    SensorSurface,
    surface_from_dict,
)

__all__ = [
    "GlassSpec",
    "LensElement",
    "LensGroup",
    "BodySpec",
    "EmitterSpec",
    "ProjectorBackSpec",
    "CameraPreset",
    "PRESET_REGISTRY",
    "simple_doublet_preset",
    "eye_model_preset",
]


# ─────────────────────────────────────────────────────────────────────────────
# Glass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GlassSpec:
    """Refractive index data for one glass type.

    Sellmeier coefficients B1-B3, C1-C3 (wavelength in micrometres).
    Falls back to a constant n when coefficients are all zero.
    """
    name:  str   = "air"
    n_d:   float = 1.0      # index at d-line (587.6 nm)
    V_d:   float = 0.0      # Abbe number (0 = not specified)
    B:     Tuple[float,float,float] = (0., 0., 0.)
    C:     Tuple[float,float,float] = (0., 0., 0.)

    def n_at(self, wavelength_um: float) -> float:
        """Sellmeier index; falls back to n_d when no coefficients given."""
        B1, B2, B3 = self.B
        C1, C2, C3 = self.C
        if B1 == B2 == B3 == 0.:
            return self.n_d
        l2 = wavelength_um**2
        n2 = 1. + B1*l2/(l2-C1) + B2*l2/(l2-C2) + B3*l2/(l2-C3)
        return math.sqrt(max(n2, 1.0))

    def to_dict(self) -> dict:
        return {"name": self.name, "n_d": self.n_d, "V_d": self.V_d,
                "B": list(self.B), "C": list(self.C)}

    @classmethod
    def from_dict(cls, d: dict) -> "GlassSpec":
        return cls(
            name=d.get("name", "air"),
            n_d=float(d.get("n_d", 1.0)),
            V_d=float(d.get("V_d", 0.0)),
            B=tuple(d.get("B", [0., 0., 0.])),
            C=tuple(d.get("C", [0., 0., 0.])),
        )

# Common presets
_AIR       = GlassSpec("air",  1.0,    0.)
_BK7       = GlassSpec("BK7",  1.5168, 64.17,
                        B=(1.03961212, 0.23179234, 1.01046945),
                        C=(0.00600069867, 0.0200179144, 103.560653))
_SF5       = GlassSpec("SF5",  1.6727, 32.21,
                        B=(1.46141885, 0.247713019, 0.949995832),
                        C=(0.0111826126, 0.0508191367, 112.041888))
_SILICA    = GlassSpec("SiO2", 1.4585, 67.8,
                        B=(0.6961663, 0.4079426, 0.8974794),
                        C=(0.0684043**2, 0.1162414**2, 9.896161**2))


GLASS_CATALOG: dict[str, GlassSpec] = {
    "air":    _AIR,
    "BK7":    _BK7,
    "SF5":    _SF5,
    "SiO2":   _SILICA,
}


# ─────────────────────────────────────────────────────────────────────────────
# Lens element
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LensElement:
    """One refracting surface + the glass on its transmission side.

    ``z_vertex`` is the Z position of the surface vertex in the global
    camera coordinate frame (optical axis = +Z, origin at mount face).
    The bake path transforms rays into the surface's local frame before
    calling ``surface.intersect()``.
    """
    surface:    ParametricSurface = field(default_factory=lambda: ConicSurface())
    glass_out:  GlassSpec         = field(default_factory=lambda: GlassSpec())
    z_vertex:   float             = 0.0     # metres
    label:      str               = ""
    # Rigid surface frame relative to the nominal optical axis.  These are
    # derived from CameraManifest.lens.surface_adjustments in the optical
    # engine frontend; both tessellation and exact preview transport consume
    # them, so alignment is never a display-only annotation.
    shift_xy_m: Tuple[float, float] = (0.0, 0.0)
    tilt_xy_deg: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "surface":   self.surface.to_dict(),
            "glass_out": self.glass_out.to_dict(),
            "z_vertex":  self.z_vertex,
            "label":     self.label,
            "shift_xy_m": list(self.shift_xy_m),
            "tilt_xy_deg": list(self.tilt_xy_deg),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LensElement":
        return cls(
            surface   = surface_from_dict(d["surface"]),
            glass_out = GlassSpec.from_dict(d.get("glass_out", {})),
            z_vertex  = float(d.get("z_vertex", 0.)),
            label     = d.get("label", ""),
            shift_xy_m=tuple(float(v) for v in d.get("shift_xy_m", (0.0, 0.0))),
            tilt_xy_deg=tuple(float(v) for v in d.get("tilt_xy_deg", (0.0, 0.0))),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Lens group
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LensGroup:
    """Ordered sequence of lens elements forming a complete optical design.

    Elements are listed front-to-back (increasing Z toward sensor).
    The bake path traces rays through them in order, applying Snell's law
    at each surface using ``glass_out.n_at(wavelength)``.
    """
    elements:  List[LensElement] = field(default_factory=list)
    label:     str               = "lens_group"

    def to_dict(self) -> dict:
        return {
            "label":    self.label,
            "elements": [e.to_dict() for e in self.elements],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LensGroup":
        return cls(
            label    = d.get("label", "lens_group"),
            elements = [LensElement.from_dict(e) for e in d.get("elements", [])],
        )


# ─────────────────────────────────────────────────────────────────────────────
# EmitterSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmitterSpec:
    """Geometric placement of an emissive surface inside the camera body.

    EmitterSpec owns the *where*; the *what* (spectral distribution, phase,
    directionality, polarization, optional UV texture) is carried by the
    ``EmitterProfile`` referenced by ``profile_name`` or supplied inline via
    ``profile_inline``.

    Placement fields
    ----------------
    pos            : 3D centre position in metres, [x, y, z] in camera local
                     coordinates (z-axis = optical axis, origin = mount face)
    normal         : outward emission direction unit vector (default −Z = into lens)
    radius         : emitter disc radius in metres
    spatial_samples: launch quadrature sites across the physical emitter disc
    enabled        : False = geometry present but emits no light

    Profile reference
    -----------------
    profile_name   : key into ``EMITTER_CATALOG`` (camera_designer.emitter_profile).
                     When non-empty the named profile is used for all emission
                     properties.  Takes precedence over ``profile_inline``.
    profile_inline : optional inline profile dict (the ``EmitterProfile.to_dict()``
                     output) for presets that embed the profile directly without
                     requiring a catalog entry.  Ignored when ``profile_name`` is set.

    Display
    -------
    label          : human-readable identifier shown in the designer UI
    """
    pos:            Tuple[float, float, float] = (0.0, 0.0, -0.010)
    normal:         Tuple[float, float, float] = (0.0, 0.0, -1.0)
    radius:         float                      = 0.005
    spatial_samples: int                       = 1
    enabled:        bool                       = True
    profile_name:   str                        = "led_warm_white"
    profile_inline: Optional[dict]             = field(default=None)
    label:          str                        = "emitter"

    def resolve_profile(self):
        """Return the EmitterProfile for this spec.

        Looks up ``profile_name`` in ``EMITTER_CATALOG`` first.  Falls back to
        deserialising ``profile_inline``.  Returns ``None`` if neither is set.
        """
        from .emitter_profile import EMITTER_CATALOG, EmitterProfile
        if self.profile_name:
            p = EMITTER_CATALOG.get(self.profile_name)
            if p is not None:
                return p
        if self.profile_inline:
            return EmitterProfile.from_dict(self.profile_inline)
        return None

    @property
    def display_tint(self) -> Tuple[float, float, float]:
        """RGB tint from the resolved profile, or a neutral default."""
        p = self.resolve_profile()
        if p is not None:
            return p.display_tint
        return (1.0, 0.92, 0.80)

    @property
    def color(self) -> Tuple[float, float, float]:
        """OpenGL-ready RGB derived from the profile's display_tint."""
        return self.display_tint

    @property
    def power(self) -> float:
        """Radiant exitance (W/m²) from the resolved profile's spectral spec.

        Falls back to 1.0 when no profile is available so legacy callers
        always get a finite, positive value for OpenGL light intensity.
        """
        p = self.resolve_profile()
        if p is not None:
            return float(p.spectral.radiant_exitance)
        return 1.0

    def to_dict(self) -> dict:
        d: dict = {
            "pos":          list(self.pos),
            "normal":       list(self.normal),
            "radius":       self.radius,
            "spatial_samples": self.spatial_samples,
            "enabled":      self.enabled,
            "profile_name": self.profile_name,
            "label":        self.label,
        }
        if self.profile_inline is not None:
            d["profile_inline"] = self.profile_inline
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EmitterSpec":
        return cls(
            pos            = tuple(float(v) for v in d.get("pos",    [0., 0., -0.010])),
            normal         = tuple(float(v) for v in d.get("normal", [0., 0., -1.])),
            radius         = float(d.get("radius", 0.005)),
            spatial_samples = max(1, int(d.get("spatial_samples", 1))),
            enabled        = bool(d.get("enabled", True)),
            profile_name   = str(d.get("profile_name", "led_warm_white")),
            profile_inline = d.get("profile_inline", None),
            label          = str(d.get("label", "emitter")),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ProjectorBackSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProjectorBackSpec:
    """A large backlit emitter panel mounted just behind the sensor plane.

    In a normal camera this position is occupied by the sensor die.  When
    the optics are used in reverse (e.g. a projector or structured-light
    emitter) this spec replaces or supplements the sensor with a powered
    emitter whose spectral output is shaped by the sensor’s own QE curve.

    Concept
    -------
    The back remains a sensor-sized physical source plane.  Its emission is
    described by the same ``EmitterProfile`` contract as every other source,
    so spectrum, coherence, phase, directionality, polarization, and an
    optional Jones-field texture survive lowering into the optical backend.
    Empty profile fields retain the legacy QE-weighted broadband panel.

    Fields
    ------
    enabled           : master on/off switch
    power             : total emitted power (scene units, ≥ 0)
    z_offset          : metres behind the sensor plane (default 2 mm)
    radius_scale      : multiplier on sensor.r_max for emitter disc size
                        (>1 over-illuminates the aperture, <1 is apodised)
    spectral_weights  : per-wavelength relative power fractions — empty list
                        means flat/white (all bands equal).
                        If non-empty, length must equal len(preset.wavelengths)
                        when the scene is built; extra entries are ignored,
                        missing entries are padded with 1.0.
    color             : GL display tint (r,g,b), 0–1 each
    profile_name      : optional key in ``EMITTER_CATALOG``.  A named profile
                        takes precedence over ``profile_inline``.
    profile_inline    : optional embedded ``EmitterProfile`` dictionary.
    spatial_samples   : source-plane sites used when ray packets are launched.
                        This changes source quadrature, never panel geometry.
    scrim_transmission: per-lane transmission of the sensor layer in projector
                        mode; empty reuses ``spectral_weights``.
    scrim_diffusion   : diffuse fraction of that transmissive sensor layer.
    label             : identifier string
    """
    enabled:          bool                        = False
    power:            float                       = 10.0
    z_offset:         float                       = 0.002   # 2 mm behind sensor
    radius_scale:     float                       = 1.0     # match sensor size
    spectral_weights: List[float]                 = field(default_factory=list)
    color:            Tuple[float, float, float]  = (0.95, 0.97, 1.0)  # cool white
    profile_name:     str                          = ""
    profile_inline:   Optional[dict]               = field(default=None)
    spatial_samples:  int                          = 64
    scrim_transmission: List[float]                 = field(default_factory=list)
    scrim_diffusion:  float                         = 0.02
    label:            str                         = "projector_back"

    # ── Helpers ─────────────────────────────────────────────────────────────

    def resolved_weights(self, n_bands: int) -> List[float]:
        """Return a list of length ``n_bands``, padding/truncating as needed.

        An empty ``spectral_weights`` list means flat white (all 1.0).
        """
        if not self.spectral_weights:
            return [1.0] * n_bands
        w = list(self.spectral_weights)
        # Pad short lists with 1.0, truncate long lists
        while len(w) < n_bands:
            w.append(1.0)
        return w[:n_bands]

    def resolved_scrim_transmission(self, n_bands: int) -> List[float]:
        authored = self.scrim_transmission or self.spectral_weights
        if not authored:
            return [1.0] * n_bands
        values = [min(1.0, max(0.0, float(v))) for v in authored]
        while len(values) < n_bands:
            values.append(1.0)
        return values[:n_bands]

    def resolve_profile(self, wavelengths_um: Sequence[float]):
        """Return an isolated, power-scaled physical emitter profile.

        Catalog profiles are cloned before their power is changed.  With no
        authored profile, the old per-lane panel weights become a histogram
        profile, preserving old manifests without retaining the legacy
        point-light launch.
        """
        from .emitter_profile import (
            AngularDistribution,
            CoherenceModel,
            DirectionalModel,
            EMITTER_CATALOG,
            EmitterProfile,
            PhaseState,
            PolarizationMode,
            PolarizationState,
            SpectralDistribution,
            SpectralModel,
        )

        profile = None
        if self.profile_name:
            catalog_profile = EMITTER_CATALOG.get(self.profile_name)
            if catalog_profile is None:
                raise ValueError(
                    f"unknown projector-back emitter profile {self.profile_name!r}"
                )
            profile = EmitterProfile.from_dict(catalog_profile.to_dict())
        elif self.profile_inline is not None:
            profile = EmitterProfile.from_dict(self.profile_inline)

        if profile is None:
            wavelengths = [float(v) for v in wavelengths_um]
            if not wavelengths:
                wavelengths = [0.550]
            profile = EmitterProfile(
                name="projector_back_legacy_panel",
                label="QE-weighted broadband projector back",
                spectral=SpectralDistribution(
                    model=SpectralModel.HISTOGRAM,
                    wavelengths_um=wavelengths,
                    weights=self.resolved_weights(len(wavelengths)),
                ),
                phase=PhaseState(model=CoherenceModel.INCOHERENT),
                directional=AngularDistribution(
                    model=DirectionalModel.LAMBERTIAN
                ),
                polarization=PolarizationState(
                    mode=PolarizationMode.UNPOLARIZED
                ),
                display_tint=self.color,
                notes="Legacy projector-back spectrum lowered to EmitterProfile.",
            )

        def _scale_exitance(node) -> None:
            if node.components:
                for _weight, child in node.components:
                    _scale_exitance(child)
            else:
                node.spectral.radiant_exitance *= max(0.0, float(self.power))

        _scale_exitance(profile)
        return profile

    def as_emitter_spec(
        self,
        *,
        sensor_z_pos: float,
        sensor_radius: float,
        wavelengths_um: Sequence[float],
    ) -> EmitterSpec:
        """Lower this panel into the common physical source-placement ABI."""
        profile = self.resolve_profile(wavelengths_um)
        return EmitterSpec(
            pos=(0.0, 0.0, float(sensor_z_pos) - float(self.z_offset)),
            normal=(0.0, 0.0, 1.0),
            radius=float(sensor_radius) * float(self.radius_scale),
            spatial_samples=max(1, int(self.spatial_samples)),
            enabled=bool(self.enabled),
            profile_name="",
            profile_inline=profile.to_dict(),
            label=str(self.label),
        )

    # ── Serialisation ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = {
            "enabled":          self.enabled,
            "power":            self.power,
            "z_offset":         self.z_offset,
            "radius_scale":     self.radius_scale,
            "spectral_weights": list(self.spectral_weights),
            "color":            list(self.color),
            "profile_name":     self.profile_name,
            "spatial_samples":  self.spatial_samples,
            "scrim_transmission": list(self.scrim_transmission),
            "scrim_diffusion":  self.scrim_diffusion,
            "label":            self.label,
        }
        if self.profile_inline is not None:
            d["profile_inline"] = self.profile_inline
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectorBackSpec":
        return cls(
            enabled          = bool(d.get("enabled",          False)),
            power            = float(d.get("power",           10.0)),
            z_offset         = float(d.get("z_offset",        0.002)),
            radius_scale     = float(d.get("radius_scale",    1.0)),
            spectral_weights = [float(v) for v in d.get("spectral_weights", [])],
            color            = tuple(float(v) for v in d.get("color", [0.95, 0.97, 1.0])),
            profile_name     = str(d.get("profile_name", "")),
            profile_inline   = d.get("profile_inline", None),
            spatial_samples  = max(1, int(d.get("spatial_samples", 64))),
            scrim_transmission = [
                float(v) for v in d.get("scrim_transmission", [])
            ],
            scrim_diffusion  = min(
                1.0, max(0.0, float(d.get("scrim_diffusion", 0.02)))
            ),
            label            = str(d.get("label", "projector_back")),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Body spec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BodySpec:
    """Physical camera body dimensions and mount position.

    All distances in metres.  The coordinate origin is the mount face
    (``LensMountRing.z_flange``).
    """
    width:         float = 0.139   # body width
    height:        float = 0.101   # body height
    depth:         float = 0.077   # body depth (front to back)
    grip_width:    float = 0.044   # grip side width
    label:         str   = "body"

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height,
                "depth": self.depth, "grip_width": self.grip_width,
                "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> "BodySpec":
        return cls(**{k: v for k, v in d.items()})


# ─────────────────────────────────────────────────────────────────────────────
# CameraPreset — the total package
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraPreset:
    """Complete packaged camera: body + mount + lens + aperture + sensor.

    This is the single object saved/loaded as a preset file (.camera.json).
    The ``BakeWorker`` reads it and produces a ``LensManifold`` noodle LUT.

    Key design datums
    -----------------
    ``mount_ring``       : defines z_flange (back-focal-distance reference)
                           and mirror clearance.
    ``aperture_stop``    : the physical aperture plane; this is ALSO the
                           manifold aperture — the (u,v) LUT key space is
                           normalised to [-1,1] over aperture_stop.r_outer.
    ``sensor``           : the photosensitive surface (flat or curved).
    ``lens_group``       : all refractive elements between world and sensor.
    """
    name:         str             = "untitled_camera"
    body:         BodySpec        = field(default_factory=BodySpec)
    mount_ring:   LensMountRing   = field(default_factory=LensMountRing)
    lens_group:   LensGroup       = field(default_factory=LensGroup)
    aperture_stop: ApertureStop   = field(default_factory=lambda: ApertureStop(
                                        z_pos=0.0,
                                        r_inner=0.0,
                                        r_outer=0.010))
    sensor:       SensorSurface   = field(default_factory=SensorSurface)

    # Bake metadata
    focal_mm:          float = 50.0
    f_number:          float = 1.8
    fov_deg:           float = 46.8   # full diagonal
    wavelengths:       list  = field(default_factory=lambda: [0.486, 0.587, 0.656])

    # Display / tonemap options (read by camera_designer_station)
    # When True the glow overlay uses log(1+bv) compression before gamma,
    # matching demo_pluck_gl.py’s uLogScale path.  Set to True for acoustic
    # presets (60+ stop DR) and leave False for photographic lenses.
    sensor_log_scale:  bool  = False
    # Emissive surfaces embedded in the camera body — LEDs, flash tubes,
    # projector elements.  Each EmitterSpec triangulates as a disc and
    # auto-injects a light source into the ray tracer when enabled.
    emitters: List["EmitterSpec"] = field(default_factory=list)

    # Projector back — a large backlit emitter panel behind the sensor plane.
    # When enabled the sensor plane is supplemented (or replaced in intent) by
    # a spectrally-filtered emitter whose output is shaped by the sensor QE
    # curve, turning the optical system into a reverse projector.
    projector_back: "ProjectorBackSpec" = field(
        default_factory=ProjectorBackSpec)
    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def aperture_radius(self) -> float:
        """Physical aperture radius at current f-number (metres)."""
        focal_m = self.focal_mm * 1e-3
        return focal_m / (2.0 * self.f_number)

    @property
    def aperture_plane_z(self) -> float:
        """Z of the aperture manifold plane (mount_ring flange + back_clearance)."""
        return self.mount_ring.aperture_plane_z

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "body":          self.body.to_dict(),
            "mount_ring":    self.mount_ring.to_dict(),
            "lens_group":    self.lens_group.to_dict(),
            "aperture_stop": self.aperture_stop.to_dict(),
            "sensor":        self.sensor.to_dict(),
            "focal_mm":         self.focal_mm,
            "f_number":         self.f_number,
            "fov_deg":          self.fov_deg,
            "wavelengths":      list(self.wavelengths),
            "sensor_log_scale": self.sensor_log_scale,
            "emitters":         [e.to_dict() for e in self.emitters],
            "projector_back":   self.projector_back.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraPreset":
        from .parametric_surfaces import surface_from_dict
        ap_d = d.get("aperture_stop", {})
        ap_surf = surface_from_dict(ap_d) if ap_d else ApertureStop()
        return cls(
            name          = d.get("name", "untitled_camera"),
            body          = BodySpec.from_dict(d.get("body", {})),
            mount_ring    = LensMountRing(**{k: v for k, v in
                                d.get("mount_ring", {}).items()
                                if k != "type"}),
            lens_group    = LensGroup.from_dict(d.get("lens_group", {})),
            aperture_stop = ap_surf,
            sensor        = SensorSurface(**{k: v for k, v in
                                d.get("sensor", {}).items()
                                if k != "type"}),
            focal_mm          = float(d.get("focal_mm", 50.)),
            f_number          = float(d.get("f_number", 1.8)),
            fov_deg           = float(d.get("fov_deg", 46.8)),
            wavelengths       = list(d.get("wavelengths", [0.486, 0.587, 0.656])),
            sensor_log_scale  = bool(d.get("sensor_log_scale", False)),
            emitters          = [EmitterSpec.from_dict(e)
                                 for e in d.get("emitters", [])],
            projector_back    = ProjectorBackSpec.from_dict(
                                 d.get("projector_back", {})),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CameraPreset":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ── Cross-section geometry for the designer UI ────────────────────────────

    def xz_cross_section(self) -> dict:
        """Return lists of (z, r) pairs for each optical element for 2-D drawing.

        Returned dict keys:
          'lens_edges'   : list of [(z0,r0),(z1,r1)] edge segments per element
          'aperture'     : (z, r_inner, r_outer) of aperture stop
          'sensor'       : (z, r) of sensor surface
          'mount_face'   : z of mount flange face
        """
        N = 64
        edges = []
        for el in self.lens_group.elements:
            surf = el.surface
            rs   = np.linspace(-surf.r_max, surf.r_max, N)
            pts  = []
            for r in rs:
                ro = np.array([r, 0., -1.])
                rd = np.array([0., 0., 1.])
                # Translate ray to surface local frame (vertex at z=0 local)
                ro_local = ro.copy()
                ro_local[2] -= el.z_vertex
                t, hit, _ = surf.intersect(ro_local, rd)
                if math.isfinite(t):
                    pts.append((hit[2] + el.z_vertex, r))
            edges.append(pts)

        return {
            "lens_edges":  edges,
            "aperture":    (self.aperture_stop.z_pos,
                            self.aperture_stop.r_inner,
                            self.aperture_stop.r_outer),
            "sensor":      (self.sensor.z_pos, self.sensor.r_max),
            "mount_face":  self.mount_ring.z_flange,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Built-in preset factory functions
# ─────────────────────────────────────────────────────────────────────────────

def simple_doublet_preset() -> CameraPreset:
    """A simple achromatic doublet (BK7 + SF5) at 50 mm f/1.8."""
    from .parametric_surfaces import ConicSurface

    # Front crown element: two spherical surfaces
    el1 = LensElement(
        surface   = ConicSurface(R=0.0305, K=0., r_max=0.018),
        glass_out = _BK7,
        z_vertex  = 0.060,
        label     = "doublet_front_1",
    )
    el2 = LensElement(
        surface   = ConicSurface(R=-0.0925, K=0., r_max=0.018),
        glass_out = _SF5,
        z_vertex  = 0.057,
        label     = "doublet_cemented",
    )
    el3 = LensElement(
        surface   = ConicSurface(R=-0.0215, K=0., r_max=0.018),
        glass_out = _AIR,
        z_vertex  = 0.054,
        label     = "doublet_rear",
    )

    mount = LensMountRing(z_flange=0.0, r_inner=0.021,
                          r_outer=0.031, back_clearance=0.004)
    aperture = ApertureStop(
        # Physical iris sits immediately behind the 54 mm rear surface.  The
        # former mount-derived 6 mm station left only 8.4 mm from the sensor;
        # center-pupil rays from almost the entire 43 mm format consequently
        # missed the 18 mm doublet before they ever reached the scene.
        z_pos=0.050,
        r_inner=0., r_outer=0.0139,
        n_blades=8,
    )  # material 8-blade iris, f/1.8 at 50mm
    sensor = SensorSurface(z_pos=-0.0024, r_max=0.0215)

    return CameraPreset(
        name          = "simple_doublet_50mm_f1.8",
        mount_ring    = mount,
        lens_group    = LensGroup(elements=[el1, el2, el3]),
        aperture_stop = aperture,
        sensor        = sensor,
        focal_mm      = 50.0,
        f_number      = 1.8,
        fov_deg       = 46.8,
    )


def eye_model_preset() -> CameraPreset:
    """Schematic eye model (Gullstrand simplified) as a CameraPreset.

    Surfaces in millimetres converted to metres:
      Cornea front  R=7.8 mm  glass=cornea (n≈1.376)
      Cornea back   R=6.5 mm  glass=aqueous (n≈1.336)
      Lens front    R=10.2 mm glass=lens (n≈1.413)
      Lens back     R=-6.0 mm glass=vitreous (n≈1.336)
      Retina        R=-12.0 mm sensor
    """
    cornea_glass   = GlassSpec("cornea",   1.376, 0.)
    aqueous_glass  = GlassSpec("aqueous",  1.336, 0.)
    lens_glass     = GlassSpec("lens",     1.413, 0.)
    vitreous_glass = GlassSpec("vitreous", 1.336, 0.)

    m = 1e-3  # mm → m

    els = [
        LensElement(ConicSurface(R=7.8*m,   r_max=5.*m), cornea_glass,   z_vertex=0.,      label="cornea_front"),
        LensElement(ConicSurface(R=6.5*m,   r_max=5.*m), aqueous_glass,  z_vertex=-0.5*m,  label="cornea_back"),
        LensElement(ConicSurface(R=10.2*m,  r_max=4.*m), lens_glass,     z_vertex=-3.6*m,  label="lens_front"),
        LensElement(ConicSurface(R=-6.0*m,  r_max=4.*m), vitreous_glass, z_vertex=-7.6*m,  label="lens_back"),
    ]

    mount = LensMountRing(z_flange=0., r_inner=0., r_outer=6.*m, back_clearance=0.)
    aperture = ApertureStop(z_pos=-3.6*m, r_inner=0., r_outer=4.*m)
    sensor = SensorSurface(z_pos=-24.*m, r_max=12.*m,
                           curvature_R=-12.*m)  # curved retina

    return CameraPreset(
        name          = "gullstrand_eye",
        mount_ring    = mount,
        lens_group    = LensGroup(elements=els),
        aperture_stop = aperture,
        sensor        = sensor,
        focal_mm      = 17.0,
        f_number      = 2.1,
        fov_deg       = 120.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Global preset registry
# ─────────────────────────────────────────────────────────────────────────────

PRESET_REGISTRY: dict[str, "CameraPreset"] = {}


def _register_builtins() -> None:
    PRESET_REGISTRY["simple_doublet_50mm_f1.8"] = simple_doublet_preset()
    PRESET_REGISTRY["gullstrand_eye"]            = eye_model_preset()


_register_builtins()
