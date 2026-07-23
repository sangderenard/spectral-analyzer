"""camera_designer/emitter_profile.py
========================================
Physical emitter profiles for the camera designer station.

An ``EmitterProfile`` describes the full radiometric and wave-optical
character of a light source — independent of its geometric placement
(that is handled by ``EmitterSpec`` in ``camera_preset.py``).

Design philosophy
-----------------
Real EM sources do not emit ideal cones of uniform density.  Only a
*programmatic* construct — a projective ray bundle used purely for tracing
geometry — may use an ideal cone (``DirectionalModel.PROJECTIVE``).  All
physically motivated emitters use distributions grounded in EM radiation
theory (Lambertian, Gaussian beam, dipole, etendue-limited, etc.).

Structure
---------
EmitterProfile
├── spectral      : SpectralDistribution   — power vs wavelength
├── phase         : PhaseState             — coherence model
├── directional   : AngularDistribution    — angular emission pattern
├── polarization  : PolarizationState      — Stokes / Jones description
└── texture       : EmissiveTexture | None — per-UV-point field override

Any surface emitter can carry an ``EmissiveTexture`` — a UV-indexed array
that records local spectral, phase, directional, and polarization parameters
at each point across the emitter face.  Think of it as the perfect film: the
recording medium that captures every quantum of passage at every UV point —
baked into a distribution map.  A 4-channel Jones field can also be stored,
making the texture a hologram-like representation of a coherent emitter's
field.  When a texture is present it takes precedence over the profile's
global fields for the corresponding sample.

All wavelengths are in **micrometres (μm)**.  Angles are in **radians** unless
a ``_deg`` suffix is present.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "SpectralModel",
    "SpectralDistribution",
    "CoherenceModel",
    "PhaseState",
    "DirectionalModel",
    "AngularDistribution",
    "PolarizationMode",
    "PolarizationState",
    "EmissiveTexture",
    "EmitterProfile",
    "EMITTER_CATALOG",
    "emitter_from_dict",
]


# ─────────────────────────────────────────────────────────────────────────────
# Spectral distribution
# ─────────────────────────────────────────────────────────────────────────────

class SpectralModel(str, Enum):
    """Functional family for the spectral power distribution."""
    BLACKBODY      = "blackbody"        # Planck: temperature_K required
    GAUSSIAN       = "gaussian"         # center_um + sigma_um
    LORENTZIAN     = "lorentzian"       # center_um + fwhm_um (Cauchy profile)
    FLAT           = "flat"             # spectrally flat (white)
    HISTOGRAM      = "histogram"        # arbitrary bins: wavelengths_um + weights
    D65            = "d65"              # CIE D65 daylight standard illuminant
    D50            = "d50"              # CIE D50 white-point
    LED_PHOSPHOR   = "led_phosphor"     # blue GaN peak + broad phosphor hump
    SODIUM_D       = "sodium_d"         # sodium D doublet (589.0 / 589.6 nm)
    MERCURY_ARC    = "mercury_arc"      # Hg arc lamp lines


@dataclass
class SpectralDistribution:
    """Spectral power distribution of an emitter.

    For HISTOGRAM mode ``wavelengths_um`` and ``weights`` must be equal-length
    lists.  For parametric modes only the relevant scalar parameters are used.
    ``radiant_exitance`` is the total power scale (arbitrary scene units).
    """
    model:            SpectralModel = SpectralModel.FLAT
    temperature_K:    float         = 6500.0      # BLACKBODY
    center_um:        float         = 0.550        # GAUSSIAN / LORENTZIAN
    sigma_um:         float         = 0.030        # GAUSSIAN half-width
    fwhm_um:          float         = 0.002        # LORENTZIAN full-width half-max
    wavelengths_um:   List[float]   = field(default_factory=list)
    weights:          List[float]   = field(default_factory=list)
    radiant_exitance: float         = 1.0

    # ── Evaluation ────────────────────────────────────────────────────────

    def power_at(self, lam_um: float) -> float:
        """Unnormalised spectral power density at *lam_um* micrometres."""
        m = self.model

        if m == SpectralModel.FLAT:
            return self.radiant_exitance

        if m == SpectralModel.BLACKBODY:
            lam_m = lam_um * 1e-6
            h, c, k = 6.626e-34, 2.998e8, 1.381e-23
            T = max(self.temperature_K, 1.0)
            exp_arg = (h * c) / (lam_m * k * T)
            if exp_arg > 700:
                return 0.0
            return self.radiant_exitance / (math.exp(exp_arg) - 1.0)

        if m == SpectralModel.GAUSSIAN:
            x = (lam_um - self.center_um) / max(self.sigma_um, 1e-9)
            return math.exp(-0.5 * x * x) * self.radiant_exitance

        if m == SpectralModel.LORENTZIAN:
            hwhm = self.fwhm_um * 0.5
            return (hwhm ** 2 / (hwhm ** 2 + (lam_um - self.center_um) ** 2)) * self.radiant_exitance

        if m == SpectralModel.HISTOGRAM:
            return self._histogram_interp(lam_um) * self.radiant_exitance

        if m == SpectralModel.D65:
            return self._cie_daylight(lam_um) * self.radiant_exitance

        if m == SpectralModel.D50:
            # D50 shape is very close to D65; shift peak slightly
            return self._cie_daylight(lam_um, shift_nm=0.0, scale=0.95) * self.radiant_exitance

        if m == SpectralModel.LED_PHOSPHOR:
            return self._led_phosphor(lam_um) * self.radiant_exitance

        if m == SpectralModel.SODIUM_D:
            return self._sodium_d(lam_um) * self.radiant_exitance

        if m == SpectralModel.MERCURY_ARC:
            return self._mercury_arc(lam_um) * self.radiant_exitance

        return self.radiant_exitance

    def sample_array(self, wavelengths_um) -> np.ndarray:
        """Evaluate over an array of wavelengths; input dtype is preserved."""
        wl = np.asarray(wavelengths_um)
        out = np.empty(wl.shape, dtype=wl.dtype)
        for idx in np.ndindex(wl.shape):
            out[idx] = self.power_at(float(wl[idx]))
        return out

    # ── Private spectral helpers ──────────────────────────────────────────

    def _histogram_interp(self, lam_um: float) -> float:
        wls, ws = self.wavelengths_um, self.weights
        if len(wls) < 2:
            return float(ws[0]) if ws else 1.0
        if lam_um <= wls[0]:
            return float(ws[0])
        if lam_um >= wls[-1]:
            return float(ws[-1])
        for i in range(len(wls) - 1):
            if wls[i] <= lam_um < wls[i + 1]:
                t = (lam_um - wls[i]) / (wls[i + 1] - wls[i])
                return float(ws[i] * (1 - t) + ws[i + 1] * t)
        return 1.0

    @staticmethod
    def _cie_daylight(lam_um: float, shift_nm: float = 0.0, scale: float = 1.0) -> float:
        """Smooth approximation of CIE D-series daylight (shape only)."""
        lam_nm = lam_um * 1000.0 + shift_nm
        if lam_nm < 300 or lam_nm > 830:
            return 0.0
        # Piecewise Gaussian blend centred on ~560 nm
        x = (lam_nm - 560.0) / 200.0
        v = max(0.0, 1.0 - 0.5 * x * x + 0.18 * x ** 4)
        return v * scale

    @staticmethod
    def _led_phosphor(lam_um: float) -> float:
        """Blue GaN peak (~450 nm) + broad phosphor hump (~555 nm)."""
        lam_nm = lam_um * 1000.0
        blue = math.exp(-0.5 * ((lam_nm - 450) / 20) ** 2)
        phos = math.exp(-0.5 * ((lam_nm - 555) / 60) ** 2) * 0.72
        return blue + phos

    @staticmethod
    def _sodium_d(lam_um: float) -> float:
        lam_nm = lam_um * 1000.0
        d1 = math.exp(-0.5 * ((lam_nm - 589.0) / 0.3) ** 2)
        d2 = math.exp(-0.5 * ((lam_nm - 589.6) / 0.3) ** 2) * 0.50
        return d1 + d2

    @staticmethod
    def _mercury_arc(lam_um: float) -> float:
        lam_nm = lam_um * 1000.0
        lines = [(404.7, 1.0), (435.8, 2.0), (546.1, 2.5),
                 (577.0, 1.5), (579.1, 1.5), (615.0, 0.5)]
        v = 0.0
        for ctr, amp in lines:
            v += amp * math.exp(-0.5 * ((lam_nm - ctr) / 0.5) ** 2)
        return v

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "model":            self.model.value,
            "temperature_K":    self.temperature_K,
            "center_um":        self.center_um,
            "sigma_um":         self.sigma_um,
            "fwhm_um":          self.fwhm_um,
            "wavelengths_um":   list(self.wavelengths_um),
            "weights":          list(self.weights),
            "radiant_exitance": self.radiant_exitance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpectralDistribution":
        return cls(
            model            = SpectralModel(d.get("model", "flat")),
            temperature_K    = float(d.get("temperature_K",    6500.0)),
            center_um        = float(d.get("center_um",        0.550)),
            sigma_um         = float(d.get("sigma_um",         0.030)),
            fwhm_um          = float(d.get("fwhm_um",          0.002)),
            wavelengths_um   = [float(v) for v in d.get("wavelengths_um", [])],
            weights          = [float(v) for v in d.get("weights", [])],
            radiant_exitance = float(d.get("radiant_exitance", 1.0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase / coherence state
# ─────────────────────────────────────────────────────────────────────────────

class CoherenceModel(str, Enum):
    INCOHERENT       = "incoherent"       # thermal / LED: random phase per emission
    COHERENT         = "coherent"         # laser: fixed phase relationship
    PARTIAL          = "partial"          # characterised by coherence_length_um
    SPATIAL_COHERENT = "spatial_coherent" # transverse coherence only


@dataclass
class PhaseState:
    """Optical coherence and phase distribution of the source.

    ``coherence_length_um`` is meaningful for PARTIAL and COHERENT modes.
    ``phase_offset_rad`` is the deterministic phase at the emitter centre;
    for COHERENT mode this sets the carrier phase of the wavefront.
    ``temporal_coherence_s`` overrides derived coherence time when non-zero;
    otherwise it is derived as lcoh / c.
    """
    model:                CoherenceModel = CoherenceModel.INCOHERENT
    coherence_length_um:  float          = 0.0   # 0 = not specified
    phase_offset_rad:     float          = 0.0
    temporal_coherence_s: float          = 0.0   # 0 = derived from lcoh

    @property
    def coherence_time_s(self) -> float:
        if self.temporal_coherence_s > 0.0:
            return self.temporal_coherence_s
        if self.coherence_length_um > 0.0:
            c_um_per_s = 2.998e14   # μm / s
            return self.coherence_length_um / c_um_per_s
        return 0.0

    def to_dict(self) -> dict:
        return {
            "model":                self.model.value,
            "coherence_length_um":  self.coherence_length_um,
            "phase_offset_rad":     self.phase_offset_rad,
            "temporal_coherence_s": self.temporal_coherence_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PhaseState":
        return cls(
            model                = CoherenceModel(d.get("model", "incoherent")),
            coherence_length_um  = float(d.get("coherence_length_um",  0.0)),
            phase_offset_rad     = float(d.get("phase_offset_rad",     0.0)),
            temporal_coherence_s = float(d.get("temporal_coherence_s", 0.0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Directional / angular emission model
# ─────────────────────────────────────────────────────────────────────────────

class DirectionalModel(str, Enum):
    """Angular emission pattern family.

    PROJECTIVE is the only model that may use an ideal geometric cone; it
    represents a programmatic ray bundle (e.g. structured-light projector),
    NOT a physical EM source.  All other models are grounded in radiometry
    or wave-optics.
    """
    LAMBERTIAN          = "lambertian"          # cosine-weighted hemisphere (real surface)
    GAUSSIAN_BEAM       = "gaussian_beam"       # TEM₀₀ Gaussian beam (laser / fibre)
    DIPOLE_ELECTRIC     = "dipole_electric"     # single oriented electric dipole: I ∝ sin²ψ
                                                # where ψ = angle between emission dir and
                                                # dipole_axis.  For z-axis dipole: I ∝ sin²θ
                                                # (toroidal — zero along normal).  Use
                                                # DIPOLE_INPLANE_MIXED for QW emitters.
    DIPOLE_MAGNETIC     = "dipole_magnetic"     # oscillating magnetic dipole (same angular
                                                # pattern as DIPOLE_ELECTRIC for given axis)
    DIPOLE_INPLANE_MIXED = "dipole_inplane_mixed"  # isotropic ensemble of in-plane (TE)
                                                   # electric dipoles — the correct model for
                                                   # InGaN/GaN quantum well emission.  φ-average
                                                   # of x̂ and ŷ dipoles gives:
                                                   #   I(θ) ∝ (1 + cos²θ) / 2
                                                   # Forward-peaked (2× on-axis vs 90°) with
                                                   # non-zero emission at 90° — unlike Lambertian
                                                   # (cos θ, zero at 90°) and unlike z-dipole
                                                   # (sin²θ, zero on-axis).
    ETENDUE_LIMITED     = "etendue_limited"     # NA-limited flat top (fibre bundle, LED+lens)
    HENYEY_GREENSTEIN   = "henyey_greenstein"   # forward/backward asymmetric scatter lobe
    MEASURED            = "measured"            # angular histogram stored in texture
    PROJECTIVE          = "projective"          # ideal cone — BACKTRACE / SENSOR PASS ONLY.
                                                # NOT a physical forward-emission model.
                                                # Use for sensor frustum confinement in the
                                                # backward visibility pass.  Never assign
                                                # to a real source (stage light, LED, laser).
                                                # Real etendue-limited sources → ETENDUE_LIMITED.


@dataclass
class AngularDistribution:
    """Parameterised angular emission distribution.

    For GAUSSIAN_BEAM: ``beam_waist_um`` is the 1/e² intensity radius at the
    waist.  ``divergence_half_angle_rad`` is the far-field half-angle; if 0 it
    is derived from beam_waist_um and center_wavelength_um via diffraction.

    For ETENDUE_LIMITED: ``numerical_aperture`` is the sine of the half-angle.

    For PROJECTIVE (programmatic only): ``half_angle_rad`` sets the ideal cone
    boundary.

    For HENYEY_GREENSTEIN: ``hg_g`` is the asymmetry parameter (−1 … +1).

    For DIPOLE_ELECTRIC / DIPOLE_MAGNETIC: ``dipole_axis`` is the unit vector
    along which the dipole oscillates (default z-axis = [0,0,1]).  The
    radiance weight is I ∝ sin²ψ where ψ is the angle between the emission
    direction and this axis.  For φ-integrated emission (``radiance_weight``
    only receives cos_theta) the result is exact only when dipole_axis == ẑ;
    for other orientations the φ-average is used automatically.
    ``DIPOLE_INPLANE_MIXED`` does not use dipole_axis — it is always the
    correct φ-averaged ensemble of x̂+ŷ dipoles.

    For MEASURED: the angular histogram lives in the owning EmissiveTexture or
    in ``angular_histogram`` as a (n_theta, n_phi) float array.
    """
    model:                    DirectionalModel     = DirectionalModel.LAMBERTIAN
    beam_waist_um:            float                = 50.0   # GAUSSIAN_BEAM
    divergence_half_angle_rad: float               = 0.0    # 0 = diffraction-limited
    center_wavelength_um:     float                = 0.550  # for diffraction calc
    numerical_aperture:       float                = 0.5    # ETENDUE_LIMITED
    half_angle_rad:           float                = math.radians(30.0)  # PROJECTIVE
    hg_g:                     float                = 0.0    # HENYEY_GREENSTEIN
    # dipole_axis: unit vector along which the dipole oscillates.
    # Stored as a tuple so the dataclass remains hashable and JSON-serialisable.
    # Default (0,0,1) = z-axis (out-of-plane).  For in-plane: (1,0,0) or (0,1,0).
    dipole_axis:              Tuple[float,float,float] = (0.0, 0.0, 1.0)
    angular_histogram:        Optional[np.ndarray] = field(default=None, compare=False)

    @property
    def effective_divergence_rad(self) -> float:
        """Far-field half-angle for GAUSSIAN_BEAM (radians)."""
        if self.divergence_half_angle_rad > 0.0:
            return self.divergence_half_angle_rad
        w0 = self.beam_waist_um
        lam = self.center_wavelength_um
        if w0 > 0:
            return lam / (math.pi * w0)
        return math.radians(1.0)

    def radiance_weight(self, cos_theta: float) -> float:
        """Relative emission weight at polar angle θ from the surface normal.

        ``cos_theta`` = cos(θ); returns an unnormalised value ≥ 0.
        """
        m = self.model

        if m == DirectionalModel.LAMBERTIAN:
            return max(0.0, cos_theta)

        if m == DirectionalModel.GAUSSIAN_BEAM:
            theta = math.acos(min(max(cos_theta, -1.0), 1.0))
            td = self.effective_divergence_rad
            return math.exp(-2.0 * theta * theta / max(td * td, 1e-30))

        if m in (DirectionalModel.DIPOLE_ELECTRIC, DirectionalModel.DIPOLE_MAGNETIC):
            # I(θ,φ) ∝ sin²ψ  where ψ = angle between emission direction and dipole axis.
            # For a dipole axis d̂ and emission direction r̂:
            #   sin²ψ = 1 - (d̂·r̂)²
            # radiance_weight receives only cos_theta (the angle from ẑ normal).
            # When dipole_axis == ẑ: d̂·r̂ = cos_theta exactly — no φ ambiguity.
            # When dipole_axis ∈ x̂/ŷ plane: average over φ analytically:
            #   <(d̂·r̂)²>_φ = sin²θ_d * sin²θ_r / 2   where θ_d = angle between d̂ and ẑ
            # For a general axis at polar angle θ_d from ẑ:
            #   <sin²ψ>_φ = 1 - (cos_theta*cos_θ_d)² - sin²θ_d * sin²θ_r / 2
            # Derivation: d̂ = (sin_θd, 0, cos_θd) without loss of generality.
            #   (d̂·r̂)² = (sin_θd*sin_θr*cosφ + cos_θd*cos_θr)²
            #   <...>_φ = sin²θd * sin²θr / 2 + cos²θd * cos²θr
            # So: <sin²ψ>_φ = 1 - cos²θd*cos²θr - sin²θd*sin²θr/2
            ax, ay, az = self.dipole_axis
            axis_len = math.sqrt(ax*ax + ay*ay + az*az)
            if axis_len < 1e-12:
                # degenerate axis — fall through to z-dipole default
                az = 1.0
                ax = ay = 0.0
            else:
                ax /= axis_len; ay /= axis_len; az /= axis_len
            cos_td = az   # cos of angle between dipole_axis and surface normal ẑ
            sin2_td = max(0.0, 1.0 - cos_td * cos_td)
            cos2_theta = cos_theta * cos_theta
            sin2_theta = max(0.0, 1.0 - cos2_theta)
            # φ-averaged sin²ψ:
            return max(0.0, 1.0 - cos_td * cos_td * cos2_theta - sin2_td * sin2_theta * 0.5)

        if m == DirectionalModel.DIPOLE_INPLANE_MIXED:
            # Isotropic in-plane (TE) dipole ensemble — InGaN/GaN QW.
            # Equal incoherent mixture of x̂ and ŷ dipoles.
            # <sin²ψ_x>_φ + <sin²ψ_y>_φ averaged:
            #   x-dipole: <sin²ψ_x>_φ = 1 - sin²θ/2      (cos_td=0, sin_td=1)
            #   y-dipole: identical by symmetry
            # Average: (1 - sin²θ/2 + 1 - sin²θ/2) / 2 = 1 - sin²θ/2
            #         = 1 - (1 - cos²θ)/2 = (1 + cos²θ)/2
            # Physical interpretation:
            #   θ=0  (normal):  I = 1.0  (maximum, along normal)
            #   θ=90°:          I = 0.5  (half-maximum at grazing)
            #   vs Lambertian:  I = cos θ → 0.0 at 90°  (wrong for QW)
            #   vs z-dipole:    I = sin²θ → 0.0 at 0°   (completely wrong)
            cos2 = cos_theta * cos_theta
            return 0.5 * (1.0 + cos2)

        if m == DirectionalModel.ETENDUE_LIMITED:
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
            return 1.0 if sin_theta <= self.numerical_aperture else 0.0

        if m == DirectionalModel.HENYEY_GREENSTEIN:
            g = self.hg_g
            if abs(g) < 1e-9:
                return 1.0 / (4.0 * math.pi)
            denom = (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5
            return (1.0 - g * g) / (4.0 * math.pi * max(denom, 1e-30))

        if m == DirectionalModel.PROJECTIVE:
            theta = math.acos(min(max(cos_theta, -1.0), 1.0))
            return 1.0 if theta <= self.half_angle_rad else 0.0

        # MEASURED: caller is responsible for querying texture directly
        return max(0.0, cos_theta)

    def to_dict(self) -> dict:
        d: dict = {
            "model":                     self.model.value,
            "beam_waist_um":             self.beam_waist_um,
            "divergence_half_angle_rad": self.divergence_half_angle_rad,
            "center_wavelength_um":      self.center_wavelength_um,
            "numerical_aperture":        self.numerical_aperture,
            "half_angle_rad":            self.half_angle_rad,
            "hg_g":                      self.hg_g,
            "dipole_axis":               list(self.dipole_axis),
        }
        if self.angular_histogram is not None:
            d["angular_histogram"] = self.angular_histogram.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AngularDistribution":
        hist: Optional[np.ndarray] = None
        if "angular_histogram" in d:
            hist = np.array(d["angular_histogram"], dtype=np.float64)
        raw_axis = d.get("dipole_axis", [0.0, 0.0, 1.0])
        axis = (float(raw_axis[0]), float(raw_axis[1]), float(raw_axis[2]))
        return cls(
            model                     = DirectionalModel(d.get("model", "lambertian")),
            beam_waist_um             = float(d.get("beam_waist_um",             50.0)),
            divergence_half_angle_rad = float(d.get("divergence_half_angle_rad", 0.0)),
            center_wavelength_um      = float(d.get("center_wavelength_um",      0.550)),
            numerical_aperture        = float(d.get("numerical_aperture",        0.5)),
            half_angle_rad            = float(d.get("half_angle_rad",            math.radians(30.0))),
            hg_g                      = float(d.get("hg_g",                      0.0)),
            dipole_axis               = axis,
            angular_histogram         = hist,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Polarization state
# ─────────────────────────────────────────────────────────────────────────────

class PolarizationMode(str, Enum):
    UNPOLARIZED = "unpolarized"   # natural / thermal emission
    LINEAR      = "linear"        # linear at angle_deg from local x-axis
    CIRCULAR    = "circular"      # left (+1) or right (−1) hand
    ELLIPTICAL  = "elliptical"    # full Jones vector description
    RADIAL      = "radial"        # radially polarised (cylindrical beam)
    AZIMUTHAL   = "azimuthal"     # azimuthally polarised


@dataclass
class PolarizationState:
    """Description of the polarization state of an emitter.

    For LINEAR: ``angle_deg`` is the polarization axis measured from the
    emitter's local x-axis in the plane of the emitter face.

    For CIRCULAR: ``handedness`` is +1 (left-hand, CCW looking into beam)
    or −1 (right-hand).

    For ELLIPTICAL: ``jones`` stores the normalised 2-component complex Jones
    vector as [re_x, im_x, re_y, im_y] for JSON/YAML compatibility.

    ``degree_of_polarization`` ∈ [0, 1]: 0 = fully unpolarised,
    1 = fully polarised; intermediate values describe partial polarisation.
    """
    mode:                   PolarizationMode = PolarizationMode.UNPOLARIZED
    angle_deg:              float            = 0.0
    handedness:             int              = 1     # +1 or -1 for CIRCULAR
    jones:                  List[float]      = field(default_factory=lambda: [1., 0., 0., 0.])
    degree_of_polarization: float            = 1.0

    @property
    def jones_vector(self) -> np.ndarray:
        """Normalised polarized component as a complex Jones vector.

        Unpolarized power is not represented by this one vector; use
        :meth:`coherent_mode_decomposition` for transport.
        """
        if self.mode == PolarizationMode.LINEAR:
            a = math.radians(self.angle_deg)
            value = np.array([math.cos(a), math.sin(a)], dtype=np.complex128)
        elif self.mode == PolarizationMode.CIRCULAR:
            s = 1.0 / math.sqrt(2.0)
            value = np.array(
                [s, 1j * (1 if self.handedness >= 0 else -1) * s],
                dtype=np.complex128,
            )
        elif self.mode == PolarizationMode.ELLIPTICAL:
            j = self.jones
            if len(j) < 4:
                raise ValueError("elliptical polarization requires four Jones values")
            value = np.array(
                [j[0] + 1j * j[1], j[2] + 1j * j[3]],
                dtype=np.complex128,
            )
        else:
            # UNPOLARIZED uses two incoherent modes below. RADIAL/AZIMUTHAL
            # use the zero-azimuth local value unless jones_vector_at is used.
            value = self.jones_vector_at(0.0)
        norm = float(np.linalg.norm(value))
        if not math.isfinite(norm) or norm <= 1.0e-15:
            raise ValueError("polarization Jones vector must be finite and non-zero")
        return value / norm

    def jones_vector_at(self, azimuth_rad: float) -> np.ndarray:
        """Return the local vector for spatially varying cylindrical modes."""
        angle = float(azimuth_rad)
        if self.mode == PolarizationMode.RADIAL:
            return np.asarray((math.cos(angle), math.sin(angle)), np.complex128)
        if self.mode == PolarizationMode.AZIMUTHAL:
            return np.asarray((-math.sin(angle), math.cos(angle)), np.complex128)
        if self.mode == PolarizationMode.UNPOLARIZED:
            return np.asarray((1.0, 0.0), np.complex128)
        return self.jones_vector

    def coherent_mode_decomposition(
        self,
        azimuth_rad: float = 0.0,
    ) -> Tuple[Tuple[float, np.ndarray], ...]:
        """Decompose partial/unpolarized light into incoherent Jones modes.

        Returned weights are power fractions. Modes must be propagated with
        distinct coherence identities and combined in intensity, never by
        adding their Jones amplitudes.
        """
        if self.mode == PolarizationMode.UNPOLARIZED:
            degree = 0.0
            primary = np.asarray((1.0, 0.0), np.complex128)
        else:
            degree = min(1.0, max(0.0, float(self.degree_of_polarization)))
            primary = (
                self.jones_vector_at(azimuth_rad)
                if self.mode in (PolarizationMode.RADIAL, PolarizationMode.AZIMUTHAL)
                else self.jones_vector
            )
        primary = primary / max(float(np.linalg.norm(primary)), 1.0e-30)
        orthogonal = np.asarray(
            (-np.conj(primary[1]), np.conj(primary[0])),
            np.complex128,
        )
        principal_weight = 0.5 * (1.0 + degree)
        orthogonal_weight = 0.5 * (1.0 - degree)
        modes = [(principal_weight, primary)]
        if orthogonal_weight > 1.0e-15:
            modes.append((orthogonal_weight, orthogonal))
        return tuple(modes)

    def to_dict(self) -> dict:
        return {
            "mode":                   self.mode.value,
            "angle_deg":              self.angle_deg,
            "handedness":             self.handedness,
            "jones":                  list(self.jones),
            "degree_of_polarization": self.degree_of_polarization,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolarizationState":
        return cls(
            mode                   = PolarizationMode(d.get("mode", "unpolarized")),
            angle_deg              = float(d.get("angle_deg",              0.0)),
            handedness             = int(d.get("handedness",               1)),
            jones                  = [float(v) for v in d.get("jones", [1., 0., 0., 0.])],
            degree_of_polarization = float(d.get("degree_of_polarization", 1.0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Emissive texture  (per-UV surface field map)
# ─────────────────────────────────────────────────────────────────────────────

#: Fixed channel index map — every EmissiveTexture uses this layout.
TEXTURE_CHANNEL_MAP: Dict[str, int] = {
    "intensity_weight":       0,   # relative local power scale factor
    "spectral_shift_um":      1,   # wavelength shift from profile centre (μm)
    "spectral_sigma_um":      2,   # local spectral width override; 0 = use profile
    "phase_offset_rad":       3,   # local phase offset added to profile phase
    "coherence_length_um":    4,   # local coherence length override; 0 = use profile
    "polarization_angle_deg": 5,   # local linear polarization angle override
    "jones_re_x":             6,   # Re(Ex) of local Jones vector override
    "jones_im_x":             7,   # Im(Ex)
    "jones_re_y":             8,   # Re(Ey)
    "jones_im_y":             9,   # Im(Ey)
}


@dataclass
class EmissiveTexture:
    """Per-UV-point override map for an emitter surface.

    Concept
    -------
    The perfect film — a recording medium that captures every quantum of
    passage at every point on the emitter face.  The EmissiveTexture is that
    recording baked into a 2-D array indexed by surface UV coordinates.
    When channels 6–9 are all populated the texture becomes a hologram-like
    representation of the emitter's coherent vector field.

    Channel layout (see TEXTURE_CHANNEL_MAP for indices 0–9)
    ---------------------------------------------------------
    0  intensity_weight       — relative local power scale (1.0 = neutral)
    1  spectral_shift_um      — wavelength offset from profile centre (μm)
    2  spectral_sigma_um      — local bandwidth override (μm); 0 = global
    3  phase_offset_rad       — local carrier phase added to global offset
    4  coherence_length_um    — local coherence override; 0 = global
    5  polarization_angle_deg — local linear polarization angle
    6  jones_re_x             — Re(Ex) of local Jones vector
    7  jones_im_x             — Im(Ex)
    8  jones_re_y             — Re(Ey)
    9  jones_im_y             — Im(Ey)

    Channels absent from the array (n_channels < index) are treated as
    "use global profile default".

    Storage
    -------
    ``data``   : ndarray (H, W, C)  float64 or float32, row-major UV order
    ``u_wrap`` : True → tile in U (periodic); False → clamp to [0,1]
    ``v_wrap`` : True → tile in V
    ``label``  : identifier string
    """
    data:   np.ndarray = field(default_factory=lambda: np.ones((1, 1, 1), np.float64))
    u_wrap: bool       = False
    v_wrap: bool       = False
    label:  str        = "emissive_texture"

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def n_channels(self) -> int:
        return self.data.shape[2] if self.data.ndim == 3 else 1

    def sample(self, u: float, v: float) -> np.ndarray:
        """Bilinear sample at (u, v) ∈ [0, 1]².  Returns full channel vector."""
        H, W = self.height, self.width
        u = (u % 1.0) if self.u_wrap else min(max(u, 0.0), 1.0)
        v = (v % 1.0) if self.v_wrap else min(max(v, 0.0), 1.0)
        x = u * (W - 1)
        y = v * (H - 1)
        x0, y0 = int(x), int(y)
        x1 = min(x0 + 1, W - 1)
        y1 = min(y0 + 1, H - 1)
        tx, ty = x - x0, y - y0
        d = self.data
        return (d[y0, x0] * (1 - tx) * (1 - ty) +
                d[y0, x1] *      tx  * (1 - ty) +
                d[y1, x0] * (1 - tx) *      ty  +
                d[y1, x1] *      tx  *      ty)

    def channel(self, name: str, u: float, v: float) -> Optional[float]:
        """Sample a named channel.  Returns None if channel not present."""
        idx = TEXTURE_CHANNEL_MAP.get(name)
        if idx is None or idx >= self.n_channels:
            return None
        return float(self.sample(u, v)[idx])

    def jones_at(self, u: float, v: float) -> Optional[np.ndarray]:
        """Return complex Jones vector [Ex, Ey] from channels 6–9, or None."""
        if self.n_channels < 10:
            return None
        ch = self.sample(u, v)
        return np.array([ch[6] + 1j * ch[7], ch[8] + 1j * ch[9]], dtype=np.complex128)

    def to_dict(self) -> dict:
        return {
            "data":   self.data.tolist(),
            "u_wrap": self.u_wrap,
            "v_wrap": self.v_wrap,
            "label":  self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmissiveTexture":
        arr = np.array(d["data"], dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return cls(
            data   = arr,
            u_wrap = bool(d.get("u_wrap", False)),
            v_wrap = bool(d.get("v_wrap", False)),
            label  = str(d.get("label",   "emissive_texture")),
        )

    @classmethod
    def uniform(cls, intensity: float = 1.0,
                width: int = 1, height: int = 1,
                label: str = "uniform") -> "EmissiveTexture":
        """Uniform single-channel intensity texture."""
        return cls(
            data  = np.full((height, width, 1), intensity, dtype=np.float64),
            label = label,
        )

    @classmethod
    def from_jones_field(cls, jones_field: np.ndarray,
                         label: str = "jones_field") -> "EmissiveTexture":
        """Build a 10-channel texture from a complex Jones field.

        ``jones_field`` : shape (H, W, 2) complex128 — Ex, Ey per UV sample.
        Channel 0 (intensity) is set to |Ex|² + |Ey|².
        """
        H, W = jones_field.shape[:2]
        Ex = jones_field[:, :, 0]
        Ey = jones_field[:, :, 1]
        intensity = (np.abs(Ex) ** 2 + np.abs(Ey) ** 2).real
        out = np.zeros((H, W, 10), dtype=np.float64)
        out[:, :, 0] = intensity
        out[:, :, 6] = Ex.real
        out[:, :, 7] = Ex.imag
        out[:, :, 8] = Ey.real
        out[:, :, 9] = Ey.imag
        return cls(data=out, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# EmitterProfile — the composite description
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmitterProfile:
    """Complete physical description of a light source's emission character.

    An EmitterProfile is position-independent: it describes *what* is emitted,
    not *where* the emitter sits.  Placement is handled by ``EmitterSpec`` in
    ``camera_preset.py``, which references a profile by name or carries an
    inline profile dict.

    Fields
    ------
    name          : identifier used in EMITTER_CATALOG and preset files
    spectral      : SpectralDistribution — spectral power distribution
    phase         : PhaseState — coherence and carrier phase model
    directional   : AngularDistribution — angular emission pattern
    polarization  : PolarizationState — Stokes / Jones polarization state
    texture       : EmissiveTexture | None — per-UV override map
    display_tint  : (r, g, b) 0–1 display colour for cross-section renders
    label         : human-readable long name
    notes         : free-form annotation string

    Composite construction
    ----------------------
    The camera duty station will support building composite profiles by
    assembling materials (shaping spectral / phase response) and geometries
    (contributing the directional pattern and texture field).  A composite is
    stored as a regular EmitterProfile whose texture carries the baked Jones
    field from the assembly simulation — making it a holographic record of the
    compound source's far-field emission.
    """
    name:         str                                = "unknown"
    spectral:     SpectralDistribution               = field(default_factory=SpectralDistribution)
    phase:        PhaseState                         = field(default_factory=PhaseState)
    directional:  AngularDistribution                = field(default_factory=AngularDistribution)
    polarization: PolarizationState                  = field(default_factory=PolarizationState)
    texture:      Optional[EmissiveTexture]          = None
    display_tint: Tuple[float, float, float]         = (1.0, 1.0, 0.9)
    label:        str                                = ""
    notes:        str                                = ""
    # Composite sub-sources.  When non-empty this profile is a weighted sum
    # of independently-specified sub-profiles, each with its own spectrum,
    # phase, directional model and polarization state.  The top-level
    # spectral/phase/directional/polarization fields are ignored for sampling
    # when components is populated — they serve only as a human-readable
    # summary.  Each entry is (weight, EmitterProfile).
    components:   List[Tuple[float, "EmitterProfile"]] = field(default_factory=list)

    # ── Sampling ──────────────────────────────────────────────────────────

    def power_at(self,
                 wavelength_um: float,
                 cos_theta:     float = 1.0,
                 u:             float = 0.5,
                 v:             float = 0.5) -> float:
        """Radiance weight at (wavelength, polar-cos, UV point).

        For composite profiles the result is the weighted sum over all
        sub-components; each component is evaluated independently so that
        different spectral, phase and directional models can coexist.

        For leaf profiles a texture overrides ``spectral_shift_um`` and
        ``intensity_weight`` when present.
        """
        if self.components:
            total = 0.0
            for w, sub in self.components:
                total += w * sub.power_at(wavelength_um, cos_theta, u, v)
            return total

        shift = 0.0
        intensity_scale = 1.0
        if self.texture is not None:
            s = self.texture.channel("spectral_shift_um", u, v)
            if s is not None:
                shift = s
            wt = self.texture.channel("intensity_weight", u, v)
            if wt is not None:
                intensity_scale = wt

        spec  = self.spectral.power_at(wavelength_um - shift)
        direc = self.directional.radiance_weight(cos_theta)
        return spec * direc * intensity_scale

    def sample_weights(self,
                       wavelengths_um,
                       cos_theta: float = 1.0,
                       u: float = 0.5,
                       v: float = 0.5) -> np.ndarray:
        """Evaluate ``power_at`` over an array of wavelengths; preserves dtype."""
        wl  = np.asarray(wavelengths_um)
        out = np.empty(wl.shape, dtype=wl.dtype)
        for idx in np.ndindex(wl.shape):
            out[idx] = self.power_at(float(wl[idx]), cos_theta, u, v)
        return out

    def component_spectral_arrays(self, wavelengths_um) -> List[Tuple[float, np.ndarray, "EmitterProfile"]]:
        """Return list of (weight, spectral_array, sub_profile) for each component.

        Useful for spawn loops that need per-band amplitude *and* the
        directional / phase model of each sub-source independently.
        For leaf profiles returns a single entry with weight=1.0.
        """
        wl = np.asarray(wavelengths_um, dtype=np.float64)
        if self.components:
            result = []
            for w, sub in self.components:
                arr = sub.spectral.sample_array(wl)
                result.append((float(w), arr, sub))
            return result
        arr = self.spectral.sample_array(wl)
        return [(1.0, arr, self)]

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d: dict = {
            "name":         self.name,
            "spectral":     self.spectral.to_dict(),
            "phase":        self.phase.to_dict(),
            "directional":  self.directional.to_dict(),
            "polarization": self.polarization.to_dict(),
            "display_tint": list(self.display_tint),
            "label":        self.label,
            "notes":        self.notes,
        }
        if self.texture is not None:
            d["texture"] = self.texture.to_dict()
        if self.components:
            d["components"] = [[float(w), sub.to_dict()] for w, sub in self.components]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EmitterProfile":
        tex: Optional[EmissiveTexture] = None
        if "texture" in d:
            tex = EmissiveTexture.from_dict(d["texture"])
        components: List[Tuple[float, "EmitterProfile"]] = []
        for entry in d.get("components", []):
            components.append((float(entry[0]), cls.from_dict(entry[1])))
        return cls(
            name         = str(d.get("name",   "unknown")),
            spectral     = SpectralDistribution.from_dict(d.get("spectral",     {})),
            phase        = PhaseState.from_dict(d.get("phase",            {})),
            directional  = AngularDistribution.from_dict(d.get("directional",   {})),
            polarization = PolarizationState.from_dict(d.get("polarization",  {})),
            texture      = tex,
            display_tint = tuple(float(v) for v in d.get("display_tint", [1., 1., .9])),
            label        = str(d.get("label",  "")),
            notes        = str(d.get("notes",  "")),
            components   = components,
        )

    # ── File I/O ──────────────────────────────────────────────────────────

    def save_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, path: str) -> "EmitterProfile":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def save_json(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "EmitterProfile":
        import json
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def emitter_from_dict(d: dict) -> "EmitterProfile":
    """Deserialise an EmitterProfile from a plain dict."""
    return EmitterProfile.from_dict(d)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in emitter catalog
# ─────────────────────────────────────────────────────────────────────────────

EMITTER_CATALOG: Dict[str, EmitterProfile] = {

    # ── Thermal / broadband ───────────────────────────────────────────────

    "thermal_6500K": EmitterProfile(
        name="thermal_6500K",
        label="Blackbody 6500 K (daylight)",
        spectral=SpectralDistribution(
            model=SpectralModel.BLACKBODY, temperature_K=6500.0),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(1.0, 0.98, 0.92),
        notes="Lambertian blackbody at 6500 K. Daylight / solar equivalent.",
    ),

    "thermal_3200K": EmitterProfile(
        name="thermal_3200K",
        label="Blackbody 3200 K (tungsten)",
        spectral=SpectralDistribution(
            model=SpectralModel.BLACKBODY, temperature_K=3200.0),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(1.0, 0.80, 0.55),
        notes="Tungsten / incandescent. Warm orange-white Planckian.",
    ),

    "thermal_2856K": EmitterProfile(
        name="thermal_2856K",
        label="CIE Illuminant A — tungsten filament 2856 K",
        spectral=SpectralDistribution(
            model=SpectralModel.BLACKBODY, temperature_K=2856.0),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(1.0, 0.72, 0.40),
        notes="CIE Standard Illuminant A: 2856 K blackbody.",
    ),

    # ── LED ───────────────────────────────────────────────────────────────

    # Warm-white phosphor-converted LED (~2700–3000 K).
    # Physical sub-sources modelled independently:
    #   A  — GaN die emission: blue photons that escape the package without
    #         hitting the phosphor layer.  Partial temporal coherence (L_c ≈ 5 µm
    #         from L_c = λ²/Δλ for a 42 nm FWHM GaN die).  Weight 0.20 because
    #         warm-white binning uses heavy phosphor loading.
    #   B  — YAG:Ce phosphor re-emission: Stokes-shifted broad Gaussian centred
    #         at 575 nm (amber-warm side).  Stokes shift and volume scattering
    #         completely destroy pump coherence (L_c ≈ 0.05 µm).  Weight 0.80.
    "led_warm_white": EmitterProfile(
        name="led_warm_white",
        label="LED warm white (~3000 K phosphor-converted)",
        # The top-level spectral/phase/directional fields summarise the aggregate
        # colour for quick lookup; power_at() delegates to components.
        spectral=SpectralDistribution(model=SpectralModel.LED_PHOSPHOR),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(1.0, 0.90, 0.70),
        notes=(
            "Composite: GaN die direct (20 %) + YAG:Ce phosphor (80 %). "
            "Die: Gaussian 450 nm σ=18 nm, L_c=5 µm partial coherence. "
            "Phosphor: Gaussian 575 nm σ=70 nm, fully incoherent L_c=0.05 µm."
        ),
        components=[
            # --- Sub-source A: GaN direct blue emission ---
            (0.20, EmitterProfile(
                name="led_warm_white.gan_die",
                label="GaN die direct (warm white)",
                spectral=SpectralDistribution(
                    model=SpectralModel.GAUSSIAN,
                    center_um=0.450,
                    sigma_um=0.018,       # 18 nm σ → ~42 nm FWHM, typical InGaN
                ),
                phase=PhaseState(
                    model=CoherenceModel.PARTIAL,
                    coherence_length_um=5.0,  # λ²/Δλ ≈ 0.450²/0.042 ≈ 4.8 µm
                    temporal_coherence_s=1.67e-14,
                ),
                # InGaN QW: dominant TE in-plane electric dipole transitions.
                # Isotropic x̂/ŷ ensemble → I(θ) ∝ (1+cos²θ)/2.
                # NOT Lambertian: QW emission is non-zero at 90° and actually
                # peaks on-axis at 2× the 90° value — measurably different.
                directional=AngularDistribution(
                    model=DirectionalModel.DIPOLE_INPLANE_MIXED,
                ),
                polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
                display_tint=(0.45, 0.45, 1.0),
                notes=(
                    "Blue InGaN QW die. In-plane mixed electric dipole emission "
                    "I(θ)∝(1+cos²θ)/2. Partial coherence L_c=5µm."
                ),
            )),
            # --- Sub-source B: YAG:Ce phosphor re-emission ---
            (0.80, EmitterProfile(
                name="led_warm_white.phosphor",
                label="YAG:Ce phosphor (warm white)",
                spectral=SpectralDistribution(
                    model=SpectralModel.GAUSSIAN,
                    center_um=0.575,      # 575 nm — warm/amber-biased YAG:Ce
                    sigma_um=0.070,       # 70 nm σ — broad phosphor hump
                ),
                phase=PhaseState(
                    model=CoherenceModel.INCOHERENT,
                    coherence_length_um=0.05,   # Stokes shift destroys coherence
                    temporal_coherence_s=1.67e-16,
                ),
                directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
                polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
                display_tint=(1.0, 0.85, 0.45),
                notes="YAG:Ce phosphor layer. Fully incoherent, true Lambertian diffuser.",
            )),
        ],
    ),

    # Cool-white phosphor-converted LED (~5700–6500 K).
    # Physical sub-sources:
    #   A  — GaN die direct blue, weight 0.30 (less phosphor loading than warm).
    #   B  — YAG:Ce phosphor re-emission centred at 545 nm (cooler/greener),
    #         weight 0.70.
    "led_cool_white": EmitterProfile(
        name="led_cool_white",
        label="LED cool white (~6000 K phosphor-converted)",
        spectral=SpectralDistribution(model=SpectralModel.LED_PHOSPHOR),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(0.92, 0.95, 1.0),
        notes=(
            "Composite: GaN die direct (30 %) + YAG:Ce phosphor (70 %). "
            "Die: Gaussian 450 nm σ=18 nm, L_c=5 µm partial coherence. "
            "Phosphor: Gaussian 545 nm σ=60 nm, fully incoherent L_c=0.05 µm."
        ),
        components=[
            # --- Sub-source A: GaN direct blue emission ---
            (0.30, EmitterProfile(
                name="led_cool_white.gan_die",
                label="GaN die direct (cool white)",
                spectral=SpectralDistribution(
                    model=SpectralModel.GAUSSIAN,
                    center_um=0.450,
                    sigma_um=0.018,
                ),
                phase=PhaseState(
                    model=CoherenceModel.PARTIAL,
                    coherence_length_um=5.0,
                    temporal_coherence_s=1.67e-14,
                ),
                # InGaN QW in-plane mixed dipole: I(θ) ∝ (1+cos²θ)/2.
                directional=AngularDistribution(
                    model=DirectionalModel.DIPOLE_INPLANE_MIXED,
                ),
                polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
                display_tint=(0.45, 0.45, 1.0),
                notes=(
                    "Blue InGaN QW die. In-plane mixed electric dipole emission "
                    "I(θ)∝(1+cos²θ)/2. Partial coherence L_c=5µm."
                ),
            )),
            # --- Sub-source B: YAG:Ce phosphor re-emission (cooler) ---
            (0.70, EmitterProfile(
                name="led_cool_white.phosphor",
                label="YAG:Ce phosphor (cool white)",
                spectral=SpectralDistribution(
                    model=SpectralModel.GAUSSIAN,
                    center_um=0.545,      # 545 nm — cooler/greener YAG:Ce
                    sigma_um=0.060,       # 60 nm σ
                ),
                phase=PhaseState(
                    model=CoherenceModel.INCOHERENT,
                    coherence_length_um=0.05,
                    temporal_coherence_s=1.67e-16,
                ),
                directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
                polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
                display_tint=(0.85, 1.0, 0.65),
                notes="YAG:Ce phosphor layer, cooler CCT. Fully incoherent, true Lambertian diffuser.",
            )),
        ],
    ),

    "led_uv_365nm": EmitterProfile(
        name="led_uv_365nm",
        label="UV LED 365 nm",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=0.365, sigma_um=0.008),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(0.5, 0.0, 1.0),
        notes="UV LED at 365 nm. ~8 nm sigma Gaussian.",
    ),

    "led_narrow_520nm": EmitterProfile(
        name="led_narrow_520nm",
        label="Narrow-band green LED 520 nm",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=0.520, sigma_um=0.012),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(0.2, 1.0, 0.2),
        notes="High-brightness narrow-band green LED. ~12 nm sigma.",
    ),

    # ── Standard illuminants ──────────────────────────────────────────────

    "d65_illuminant": EmitterProfile(
        name="d65_illuminant",
        label="CIE D65 standard daylight illuminant",
        spectral=SpectralDistribution(model=SpectralModel.D65),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(1.0, 1.0, 1.0),
        notes="CIE D65 daylight illuminant. Lambertian surface emitter.",
    ),

    "d50_illuminant": EmitterProfile(
        name="d50_illuminant",
        label="CIE D50 horizon daylight illuminant",
        spectral=SpectralDistribution(model=SpectralModel.D50),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(1.0, 0.97, 0.90),
        notes="CIE D50 illuminant, horizon/afternoon daylight approximation.",
    ),

    # ── Arc lamps ─────────────────────────────────────────────────────────

    "sodium_streetlamp": EmitterProfile(
        name="sodium_streetlamp",
        label="Low-pressure sodium D-line doublet (589 nm)",
        spectral=SpectralDistribution(model=SpectralModel.SODIUM_D),
        phase=PhaseState(model=CoherenceModel.PARTIAL, coherence_length_um=1200.0),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(1.0, 0.85, 0.05),
        notes="Low-pressure sodium vapour lamp. Near-monochromatic 589 nm doublet.",
    ),

    "mercury_arc": EmitterProfile(
        name="mercury_arc",
        label="Mercury arc lamp (Hg spectral lines)",
        spectral=SpectralDistribution(model=SpectralModel.MERCURY_ARC),
        phase=PhaseState(model=CoherenceModel.PARTIAL, coherence_length_um=50.0),
        directional=AngularDistribution(model=DirectionalModel.LAMBERTIAN),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(0.8, 0.8, 1.0),
        notes="Mercury arc lamp. Lines at 404.7, 435.8, 546.1, 577/579 nm.",
    ),

    # ── Lasers ────────────────────────────────────────────────────────────

    "laser_532nm_green": EmitterProfile(
        name="laser_532nm_green",
        label="Nd:YAG 532 nm CW laser (TEM₀₀, linear H)",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=0.532, sigma_um=0.0001),
        phase=PhaseState(model=CoherenceModel.COHERENT, coherence_length_um=1e8),
        directional=AngularDistribution(
            model=DirectionalModel.GAUSSIAN_BEAM,
            beam_waist_um=500.0, center_wavelength_um=0.532),
        polarization=PolarizationState(
            mode=PolarizationMode.LINEAR, angle_deg=0.0,
            degree_of_polarization=1.0),
        display_tint=(0.2, 1.0, 0.2),
        notes="532 nm DPSS laser, TEM00, linearly polarised horizontal. ~100 μm coherence length.",
    ),

    "laser_633nm_HeNe": EmitterProfile(
        name="laser_633nm_HeNe",
        label="He-Ne 632.8 nm laser (TEM₀₀)",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=0.6328, sigma_um=0.00005),
        phase=PhaseState(model=CoherenceModel.COHERENT, coherence_length_um=3e8),
        directional=AngularDistribution(
            model=DirectionalModel.GAUSSIAN_BEAM,
            beam_waist_um=350.0, center_wavelength_um=0.6328),
        polarization=PolarizationState(
            mode=PolarizationMode.LINEAR, angle_deg=0.0,
            degree_of_polarization=1.0),
        display_tint=(1.0, 0.1, 0.05),
        notes="He-Ne 632.8 nm CW laser. Excellent coherence (lcoh ~300 m).",
    ),

    "laser_405nm_violet": EmitterProfile(
        name="laser_405nm_violet",
        label="405 nm violet diode laser",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=0.405, sigma_um=0.0005),
        phase=PhaseState(model=CoherenceModel.COHERENT, coherence_length_um=1e6),
        directional=AngularDistribution(
            model=DirectionalModel.GAUSSIAN_BEAM,
            beam_waist_um=200.0, center_wavelength_um=0.405),
        polarization=PolarizationState(
            mode=PolarizationMode.LINEAR, angle_deg=0.0,
            degree_of_polarization=0.95),
        display_tint=(0.65, 0.1, 1.0),
        notes="405 nm violet diode laser. ~1 m coherence length.",
    ),

    "laser_1064nm_YAG": EmitterProfile(
        name="laser_1064nm_YAG",
        label="Nd:YAG 1064 nm fundamental CW",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=1.064, sigma_um=0.00005),
        phase=PhaseState(model=CoherenceModel.COHERENT, coherence_length_um=2e8),
        directional=AngularDistribution(
            model=DirectionalModel.GAUSSIAN_BEAM,
            beam_waist_um=800.0, center_wavelength_um=1.064),
        polarization=PolarizationState(
            mode=PolarizationMode.LINEAR, angle_deg=0.0,
            degree_of_polarization=1.0),
        display_tint=(0.7, 0.2, 0.0),
        notes="Nd:YAG 1064 nm fundamental. Near-IR, invisible.",
    ),

    "laser_532nm_green_lcp": EmitterProfile(
        name="laser_532nm_green_lcp",
        label="532 nm laser — left-circular polarisation",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=0.532, sigma_um=0.0001),
        phase=PhaseState(model=CoherenceModel.COHERENT, coherence_length_um=1e8),
        directional=AngularDistribution(
            model=DirectionalModel.GAUSSIAN_BEAM,
            beam_waist_um=500.0, center_wavelength_um=0.532),
        polarization=PolarizationState(
            mode=PolarizationMode.CIRCULAR, handedness=1,
            degree_of_polarization=1.0),
        display_tint=(0.2, 1.0, 0.3),
        notes="Left-circularly polarised 532 nm DPSS laser.",
    ),

    # ── Fibres ────────────────────────────────────────────────────────────

    "fiber_end_sm_1310nm": EmitterProfile(
        name="fiber_end_sm_1310nm",
        label="SMF-28 single-mode fibre end face 1310 nm",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=1.310, sigma_um=0.001),
        phase=PhaseState(model=CoherenceModel.COHERENT, coherence_length_um=5e7),
        directional=AngularDistribution(
            model=DirectionalModel.GAUSSIAN_BEAM,
            beam_waist_um=4.5, center_wavelength_um=1.310),
        polarization=PolarizationState(
            mode=PolarizationMode.LINEAR, angle_deg=0.0,
            degree_of_polarization=0.98),
        display_tint=(0.8, 0.4, 0.1),
        notes="SMF-28 end face at 1310 nm. Mode field ~9 μm diameter, 4.5 μm waist.",
    ),

    "fiber_end_sm_1550nm": EmitterProfile(
        name="fiber_end_sm_1550nm",
        label="SMF-28 single-mode fibre end face 1550 nm",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=1.550, sigma_um=0.001),
        phase=PhaseState(model=CoherenceModel.COHERENT, coherence_length_um=5e7),
        directional=AngularDistribution(
            model=DirectionalModel.GAUSSIAN_BEAM,
            beam_waist_um=5.25, center_wavelength_um=1.550),
        polarization=PolarizationState(
            mode=PolarizationMode.LINEAR, angle_deg=0.0,
            degree_of_polarization=0.98),
        display_tint=(0.6, 0.3, 0.0),
        notes="SMF-28 end face at 1550 nm (telecom C-band). Mode field ~10.5 μm.",
    ),

    "fiber_bundle_multimode": EmitterProfile(
        name="fiber_bundle_multimode",
        label="Multimode fibre bundle output (NA=0.22, D65)",
        spectral=SpectralDistribution(model=SpectralModel.D65),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(
            model=DirectionalModel.ETENDUE_LIMITED, numerical_aperture=0.22),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(1.0, 1.0, 0.95),
        notes="Multimode fibre bundle output, NA=0.22. Flat top within acceptance cone.",
    ),

    # ── Dipole / wave-optics ──────────────────────────────────────────────

    "dipole_electric_visible": EmitterProfile(
        name="dipole_electric_visible",
        label="Oscillating electric dipole (visible broadband)",
        spectral=SpectralDistribution(model=SpectralModel.FLAT),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.DIPOLE_ELECTRIC),
        polarization=PolarizationState(
            mode=PolarizationMode.LINEAR, angle_deg=0.0,
            degree_of_polarization=1.0),
        display_tint=(0.9, 0.9, 0.7),
        notes="Ideal oscillating electric dipole. sin²θ angular distribution.",
    ),

    "dipole_magnetic_visible": EmitterProfile(
        name="dipole_magnetic_visible",
        label="Oscillating magnetic dipole (visible broadband)",
        spectral=SpectralDistribution(model=SpectralModel.FLAT),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(model=DirectionalModel.DIPOLE_MAGNETIC),
        polarization=PolarizationState(
            mode=PolarizationMode.LINEAR, angle_deg=90.0,
            degree_of_polarization=1.0),
        display_tint=(0.7, 0.9, 0.9),
        notes="Oscillating magnetic dipole. Same geometry as electric dipole, rotated 90°.",
    ),

    # ── Scatter / diffuse ─────────────────────────────────────────────────

    "forward_scatter_hg08": EmitterProfile(
        name="forward_scatter_hg08",
        label="Forward-scattering Henyey-Greenstein g=0.8",
        spectral=SpectralDistribution(model=SpectralModel.FLAT),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(
            model=DirectionalModel.HENYEY_GREENSTEIN, hg_g=0.8),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(1.0, 0.95, 0.85),
        notes="Forward-scattering HG phase function g=0.8 (aerosol, tissue).",
    ),

    "isotropic_scatter": EmitterProfile(
        name="isotropic_scatter",
        label="Isotropic scatter g=0 (Rayleigh-like)",
        spectral=SpectralDistribution(model=SpectralModel.FLAT),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(
            model=DirectionalModel.HENYEY_GREENSTEIN, hg_g=0.0),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(0.85, 0.90, 1.0),
        notes="Isotropic HG scatterer (g=0). Equal in all directions.",
    ),

    # ── Projective (programmatic only) ────────────────────────────────────

    "projector_red_638nm": EmitterProfile(
        name="projector_red_638nm",
        label="DLP projector red primary ~638 nm [PROJECTIVE]",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=0.638, sigma_um=0.006),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(
            model=DirectionalModel.PROJECTIVE,
            half_angle_rad=math.radians(12.0)),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(1.0, 0.1, 0.1),
        notes="PROJECTIVE model — ideal geometric cone. Not physical EM. Use for ray-tracing geometry only.",
    ),

    "projector_green_520nm": EmitterProfile(
        name="projector_green_520nm",
        label="DLP projector green primary ~520 nm [PROJECTIVE]",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=0.520, sigma_um=0.007),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(
            model=DirectionalModel.PROJECTIVE,
            half_angle_rad=math.radians(12.0)),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(0.1, 1.0, 0.1),
        notes="PROJECTIVE model — ideal cone. Programmatic only.",
    ),

    "projector_blue_450nm": EmitterProfile(
        name="projector_blue_450nm",
        label="DLP projector blue primary ~450 nm [PROJECTIVE]",
        spectral=SpectralDistribution(
            model=SpectralModel.GAUSSIAN, center_um=0.450, sigma_um=0.008),
        phase=PhaseState(model=CoherenceModel.INCOHERENT),
        directional=AngularDistribution(
            model=DirectionalModel.PROJECTIVE,
            half_angle_rad=math.radians(12.0)),
        polarization=PolarizationState(mode=PolarizationMode.UNPOLARIZED),
        display_tint=(0.2, 0.4, 1.0),
        notes="PROJECTIVE model — ideal cone. Programmatic only.",
    ),
}
