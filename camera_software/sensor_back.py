"""camera_software/sensor_back.py
----------------------------------
Hook-based sensor/film back pipeline.

SensorBack owns the raw image handoff from the C++ ray tracer and mediates
it through a composable hook chain of independently replaceable parts.

Parts hierarchy
---------------
BackGeometrySpec     — physical frame geometry and mount standard.
                       Determines sensor/film physical dimensions, aspect,
                       and mount compatibility (Hasselblad V, 4×5 Graflock, …).

SensorChipSpec       — digital sensor electronics: QE, noise parameters, well
                       capacity, ADC bit depth, CFA layout.  Maps to a
                       SensorFilmDatabase sensor entry when present.

FilmEmulsionSpec     — chemical film: ISO, grain, halation radius, H&D curve
                       toe/shoulder, base+fog density.

ColorScienceProfile  — 3×3 color matrix, tone curve, output color space, and
                       white balance illuminant.

SensorBackProfile    — composes any combination of the above parts.
                       Missing parts leave the corresponding hook stage a no-op
                       so the pipeline degrades gracefully.

Hook chain
----------
Each hook is a callable registered on a SensorBack at a named stage.
Stages run in order; within a stage, lower priority int = earlier.

  Stage                    What runs here
  -----------------------  -------------------------------------------------------
  STAGE_RAW_RECEIVED       Flat-field correction, bad-pixel masking, hot-column
                           suppression, physical calibration.
                           Input: context.raw_image  (native dtype from C++).

  STAGE_DEVELOP            Photon/energy → electron conversion, QE weighting,
                           well-fill clamp, shot noise, dark current injection.
                           Input: context.electrons (float64, same spatial shape).

  STAGE_READOUT            Electron → ADU, read noise, gain staging, ADC
                           quantisation, black-level subtraction.
                           Input: context.adu (float64, same shape).

  STAGE_COLOR_SCIENCE      CFA demosaicing, color matrix application, tone curve
                           / H&D transfer, output color space transform.
                           Input: context.rgb  (H, W, 3) float64.

  STAGE_OUTPUT             Final sharpening, output rescale, per-format clamp,
                           format cast.
                           Input: context.output (any shape / dtype).

dtype contract
--------------
native dtype of raw_image is detected and propagated.  Individual hooks may
cast internally but must document their output dtype contract.  The pipeline
never silently coerces the chain input dtype between stages.

Built-in profiles
-----------------
  SensorBackProfile.medium_format_120_6x6()  — 120 6×6: 56mm square, 40mm radius
  SensorBackProfile.medium_format_120_6x45() — 120 6×4.5: 56×42mm
  SensorBackProfile.large_format_4x5()       — 4×5 in: 96×121mm
  SensorBackProfile.fullframe_35mm()         — 35mm full-frame: 24×36mm
  SensorBackProfile.digital_back_mf()        — generic 53.4×40.0mm digital back

SensorBack
----------
CameraBack subclass.  Install on ForwardCppLensBench as _sensor_back.
Call receive_raw(tracer) instead of tracer.get_sensor_image() directly.

The raw pull (tracer.get_sensor_image()) is issued inside receive_raw(), which
is the ownership boundary: the lab no longer calls get_sensor_image() directly.

Example
-------
    from camera_software.sensor_back import SensorBack, SensorBackProfile

    profile = SensorBackProfile.medium_format_120_6x6()
    bench._sensor_back = SensorBack.from_profile(profile, res=64)
    ...
    img = bench._sensor_back.receive_raw(bench.tracer)
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .camera_back import CameraBack, FlatBack

__all__ = [
    # Parts
    "BackGeometrySpec",
    "SensorChipSpec",
    "FilmEmulsionSpec",
    "ColorScienceProfile",
    "SensorBackProfile",
    # Hook machinery
    "HookStage",
    "RawHandoffContext",
    "RawHandoffHook",
    # Main class
    "SensorBack",
]


# ---------------------------------------------------------------------------
# Stage enumeration
# ---------------------------------------------------------------------------

class HookStage(enum.IntEnum):
    """Ordered processing stages in the raw → output pipeline.

    Comparison works naturally: STAGE_RAW_RECEIVED < STAGE_DEVELOP < … .
    """
    RAW_RECEIVED  = 0
    DEVELOP       = 1
    READOUT       = 2
    COLOR_SCIENCE = 3
    OUTPUT        = 4


# ---------------------------------------------------------------------------
# RawHandoffContext
# ---------------------------------------------------------------------------

class RawHandoffContext:
    """Mutable container passed through the hook chain.

    Each stage populates its output slot; downstream stages read it.
    The dtype of raw_image is the native dtype from the C++ tracer —
    it is stored as-is and never coerced by the pipeline itself.

    Attributes
    ----------
    raw_image    : ndarray  — as returned by tracer.get_sensor_image().
                              shape is typically (H, W) or (H, W, C).
    electrons    : ndarray or None — after STAGE_DEVELOP.  float64.
    adu          : ndarray or None — after STAGE_READOUT.
    rgb          : ndarray or None — (H, W, 3) after STAGE_COLOR_SCIENCE.
    output       : ndarray or None — final result after STAGE_OUTPUT.
    profile      : SensorBackProfile — the owning profile.
    extra        : dict — arbitrary hook-private scratch space.
    """

    __slots__ = ("raw_image", "electrons", "adu", "rgb", "output",
                 "profile", "extra")

    def __init__(self,
                 raw_image: np.ndarray,
                 profile: "SensorBackProfile") -> None:
        self.raw_image : np.ndarray             = raw_image
        self.electrons : Optional[np.ndarray]   = None
        self.adu       : Optional[np.ndarray]   = None
        self.rgb       : Optional[np.ndarray]   = None
        self.output    : Optional[np.ndarray]   = None
        self.profile   : SensorBackProfile      = profile
        self.extra     : Dict                   = {}

    def resolved_output(self) -> np.ndarray:
        """Return the best available result, falling back through stages."""
        for cand in (self.output, self.rgb, self.adu, self.electrons, self.raw_image):
            if cand is not None:
                return cand
        return self.raw_image


# ---------------------------------------------------------------------------
# RawHandoffHook
# ---------------------------------------------------------------------------

class RawHandoffHook:
    """Base class for a pipeline hook.

    Subclass and override ``__call__``, or pass a plain callable to
    ``SensorBack.register_hook(fn, stage, priority)``.

    Attributes
    ----------
    stage    : HookStage   — which stage this hook runs at.
    priority : int         — execution order within a stage (lower = earlier).
    name     : str         — human-readable label for diagnostics.
    enabled  : bool        — when False the hook is skipped (not removed).
    """

    def __init__(self,
                 stage: HookStage = HookStage.RAW_RECEIVED,
                 priority: int = 100,
                 name: str = "") -> None:
        self.stage    : HookStage = HookStage(int(stage))
        self.priority : int       = int(priority)
        self.name     : str       = name or type(self).__name__
        self.enabled  : bool      = True

    def __call__(self, ctx: RawHandoffContext) -> None:  # pragma: no cover
        """Run the hook.  Modify ctx in-place; no return value is used."""
        raise NotImplementedError


class _CallableHook(RawHandoffHook):
    """Wraps a plain callable as a RawHandoffHook."""

    def __init__(self,
                 fn: Callable[[RawHandoffContext], None],
                 stage: HookStage,
                 priority: int,
                 name: str = "") -> None:
        super().__init__(stage, priority, name or getattr(fn, "__name__", "hook"))
        self._fn = fn

    def __call__(self, ctx: RawHandoffContext) -> None:
        self._fn(ctx)


# ---------------------------------------------------------------------------
# Parts — BackGeometrySpec
# ---------------------------------------------------------------------------

# Standard mount/format identifiers used by BackGeometrySpec.mount_standard.
MOUNT_120_6x6    = "120_6x6"
MOUNT_120_6x7    = "120_6x7"
MOUNT_120_6x45   = "120_6x45"
MOUNT_120_6x9    = "120_6x9"
MOUNT_4x5_IN     = "4x5_large"
MOUNT_8x10_IN    = "8x10_large"
MOUNT_35MM_FULL  = "35mm_full"
MOUNT_APS_C      = "aps_c"
MOUNT_APS_H      = "aps_h"
MOUNT_MF_DIGITAL = "mf_digital"   # generic 53.4×40 mm digital back
MOUNT_CUSTOM     = "custom"


@dataclass
class BackGeometrySpec:
    """Physical geometry of a camera back or film gate.

    All dimensions in millimetres.  ``radius_m`` is the inscribed-circle half-
    diagonal used by the optical design to define the image circle requirement.

    Attributes
    ----------
    mount_standard  : str   — One of the MOUNT_* constants above.
    frame_w_mm      : float — Exposed frame width.
    frame_h_mm      : float — Exposed frame height.
    image_circle_mm : float — Minimum required image circle diameter.
    aspect_ratio    : float — frame_w_mm / frame_h_mm (computed if 0).
    pixel_pitch_um  : float — Pixel pitch in μm (0 = unresolved / film).
    res_x           : int   — Pixel columns (0 = unresolved).
    res_y           : int   — Pixel rows (0 = unresolved).

    Read-only properties
    --------------------
    radius_m        — half-diagonal in metres (image_circle_mm / 2 / 1000).
    """

    mount_standard  : str   = MOUNT_120_6x6
    frame_w_mm      : float = 56.0
    frame_h_mm      : float = 56.0
    image_circle_mm : float = 80.0
    aspect_ratio    : float = 0.0          # 0 → auto-compute
    pixel_pitch_um  : float = 0.0          # 0 → unresolved
    res_x           : int   = 0            # 0 → unresolved
    res_y           : int   = 0            # 0 → unresolved

    @property
    def radius_m(self) -> float:
        return self.image_circle_mm * 0.5e-3

    @property
    def effective_aspect(self) -> float:
        if self.aspect_ratio > 0.0:
            return self.aspect_ratio
        if self.frame_h_mm > 1e-6:
            return self.frame_w_mm / self.frame_h_mm
        return 1.0

    # ── Named constructors for standard formats ───────────────────────────────

    @classmethod
    def medium_format_120_6x6(cls) -> "BackGeometrySpec":
        """120 6×6: 56×56 mm square format, ~80 mm image circle."""
        return cls(mount_standard=MOUNT_120_6x6,
                   frame_w_mm=56.0, frame_h_mm=56.0, image_circle_mm=80.0)

    @classmethod
    def medium_format_120_6x45(cls) -> "BackGeometrySpec":
        """120 6×4.5: 56×42 mm portrait format, ~70 mm image circle."""
        return cls(mount_standard=MOUNT_120_6x45,
                   frame_w_mm=56.0, frame_h_mm=42.0, image_circle_mm=70.0)

    @classmethod
    def large_format_4x5(cls) -> "BackGeometrySpec":
        """4×5 inch large format: 96×121 mm, 155 mm image circle."""
        return cls(mount_standard=MOUNT_4x5_IN,
                   frame_w_mm=96.0, frame_h_mm=121.0, image_circle_mm=155.0)

    @classmethod
    def fullframe_35mm(cls) -> "BackGeometrySpec":
        """35 mm full-frame: 24×36 mm, 43 mm image circle."""
        return cls(mount_standard=MOUNT_35MM_FULL,
                   frame_w_mm=36.0, frame_h_mm=24.0, image_circle_mm=43.0)

    @classmethod
    def digital_back_mf(cls) -> "BackGeometrySpec":
        """Generic medium-format digital back: 53.4×40.0 mm, 66 mm image circle."""
        return cls(mount_standard=MOUNT_MF_DIGITAL,
                   frame_w_mm=53.4, frame_h_mm=40.0, image_circle_mm=66.0)


# ---------------------------------------------------------------------------
# Parts — SensorChipSpec
# ---------------------------------------------------------------------------

@dataclass
class SensorChipSpec:
    """Electronic sensor chip parameters.

    Used by the built-in DEVELOP and READOUT hooks.  All values are per-pixel
    unless noted otherwise.

    The ``sensor_db_name`` field optionally links this spec to an entry in
    ``SensorFilmDatabase`` so the lab can synchronise C++ SSBO parameters from
    the same source of truth.

    Photon-electron conversion model
    ---------------------------------
    electrons = photons × qe_peak × exposure_time_s × scale
    shot_noise ~ Poisson(electrons)               (approx via √electrons Gaussian)
    dark       ~ Poisson(dark_current_e_s × t)
    total_e    = clamp(shot + dark, 0, full_well_e)
    """

    # Link to SensorFilmDatabase entry (optional)
    sensor_db_name  : str   = ""

    # Physical
    pixel_pitch_um  : float = 5.3          # μm — medium format typical
    full_well_e     : float = 150_000.0    # electrons at saturation
    qe_peak         : float = 0.78         # peak quantum efficiency [0,1]

    # Noise
    read_noise_e    : float = 2.5          # e− RMS
    dark_current_e_s: float = 0.10         # e− / pixel / second

    # ADC
    adc_bits        : int   = 16           # bit depth
    black_level_adu : int   = 512          # output ADU at zero signal
    white_level_adu : int   = 65535        # output ADU at saturation

    # CFA — matching CFAPattern names (RGGB, XTRANS, MONO, …)
    cfa_layout      : str   = "RGGB"
    cfa_r_peak_nm   : float = 600.0
    cfa_r_fwhm_nm   : float = 85.0
    cfa_g_peak_nm   : float = 535.0
    cfa_g_fwhm_nm   : float = 90.0
    cfa_b_peak_nm   : float = 455.0
    cfa_b_fwhm_nm   : float = 80.0

    # Spectral sensitivity normalisation
    apply_cfa_weights: bool  = True

    @property
    def adu_range(self) -> int:
        return max(1, self.white_level_adu - self.black_level_adu)

    @property
    def electrons_per_adu(self) -> float:
        return self.full_well_e / max(1.0, float(self.adu_range))

    def rgb_sensitivity_weights(self) -> Tuple[float, float, float]:
        """Rough (R, G, B) relative sensitivity weights from CFA peak QE."""
        # Approximated as a Gaussian integral proportional to FWHM.
        # For a Bayer sensor, G has 2× area weight.
        g_area = 2.0 if self.cfa_layout in ("RGGB", "GRBG", "GBRG", "BGGR") else 1.0
        r_w = self.cfa_r_fwhm_nm * 0.25
        g_w = self.cfa_g_fwhm_nm * g_area * 0.5 / g_area   # normalise back
        b_w = self.cfa_b_fwhm_nm * 0.25
        peak = max(r_w, g_w, b_w, 1e-12)
        return (r_w / peak, g_w / peak, b_w / peak)


# ---------------------------------------------------------------------------
# Parts — FilmEmulsionSpec
# ---------------------------------------------------------------------------

@dataclass
class FilmEmulsionSpec:
    """Chemical film emulsion parameters.

    Used by the built-in DEVELOP hook when no ``SensorChipSpec`` is present
    (i.e. a purely film-based back).  Can also augment a digital sensor to
    model film-simulation modes.

    Attributes
    ----------
    film_db_name    : str   — optional link to SensorFilmDatabase film entry.
    iso             : float — emulsion speed.
    exposure_time_s : float — shutter speed (seconds).
    base_density    : float — D-min (film base + fog), D units.
    gamma_curve     : float — H&D straight-line slope (contrast index).
    toe_density     : float — D value at the toe/shoulder inflection, D units.
    shoulder_density: float — D value at the shoulder inflection, D units.
    halation_radius : float — halation blur σ in normalised sensor units [0,1].
                              0 = no halation.
    grain_sigma     : float — grain σ as fraction of signal level [0,1].
                              0 = no grain.
    """

    film_db_name      : str   = ""
    iso               : float = 100.0
    exposure_time_s   : float = 0.010
    base_density      : float = 0.05     # D-min
    gamma_curve       : float = 0.70     # characteristic curve slope
    toe_density       : float = 0.10     # toe threshold, D
    shoulder_density  : float = 2.80     # shoulder ceiling, D
    halation_radius   : float = 0.0      # 0 = disabled
    grain_sigma       : float = 0.0      # 0 = disabled

    # Processing
    quantum_efficiency : float = 0.95
    target_grey_point  : float = 0.18    # scene reflectance → 18% grey

    def hd_curve(self, log_h: float) -> float:
        """Evaluate the Hurter-Driffield density curve at log exposure log_H.

        Piecewise: toe → straight → shoulder (simple 3-segment approximation).
        Returns density D in [base_density, shoulder_density].
        """
        log_H = float(log_h)
        D_base = self.base_density
        D_toe  = self.toe_density
        D_sho  = self.shoulder_density
        gamma  = self.gamma_curve
        # Toe: compressed exponential below D_toe
        D_toe_log = D_toe / max(gamma, 1e-9)
        if log_H < D_toe_log:
            alpha = max(0.0, log_H / max(D_toe_log, 1e-12))
            return D_base + (D_toe - D_base) * alpha * alpha
        # Straight: linear from toe → shoulder
        D_straight = D_toe + gamma * (log_H - D_toe_log)
        D_sho_log  = D_toe_log + (D_sho - D_toe) / max(gamma, 1e-9)
        if D_straight <= D_sho:
            return min(D_straight, D_sho)
        # Shoulder: compressed exponential above D_sho
        over = log_H - D_sho_log
        shoulder_compress = 0.3   # density units of shoulder compression region
        if shoulder_compress < 1e-9:
            return D_sho
        alpha = min(1.0, over / shoulder_compress)
        return D_sho + (D_sho * 0.05) * alpha * (2.0 - alpha)   # asymptote


# ---------------------------------------------------------------------------
# Parts — ColorScienceProfile
# ---------------------------------------------------------------------------

@dataclass
class ColorScienceProfile:
    """Color matrix, tone curve, and output color space.

    The color_matrix transforms XYZ or camera-native (R,G,B) linear values
    into the target color space.  When None, an identity transform is used.

    tone_curve_lut is a (N,) float64 array sampled uniformly over [0,1].
    Linear interpolation is used.  When None, a linear passthrough is used.

    Attributes
    ----------
    color_matrix    : (3,3) float64 or None
    tone_curve_lut  : (N,) float64 or None
    white_balance   : (3,) float64  — per-channel multiplier (default all-1).
    output_space    : str — 'sRGB', 'linear_sRGB', 'acescg', 'display_p3'.
    gamma           : float — output gamma exponent (1.0 = linear, 2.2 = sRGB).
                      If output_space is 'sRGB' the full sRGB transfer function
                      is applied instead of a simple power.
    clip_output     : bool — when True, clamp output to [0, 1].
    """

    color_matrix    : Optional[np.ndarray] = None
    tone_curve_lut  : Optional[np.ndarray] = None
    white_balance   : Optional[np.ndarray] = None   # (3,) float64
    output_space    : str   = "sRGB"
    gamma           : float = 2.2
    clip_output     : bool  = False

    # ── Named constructors ────────────────────────────────────────────────────

    @classmethod
    def identity(cls) -> "ColorScienceProfile":
        """No color transform, linear output."""
        return cls(output_space="linear_sRGB", gamma=1.0, clip_output=False)

    @classmethod
    def srgb_standard(cls) -> "ColorScienceProfile":
        """sRGB transfer function, no matrix."""
        return cls(output_space="sRGB", gamma=2.2, clip_output=True)

    @classmethod
    def film_print(cls) -> "ColorScienceProfile":
        """Mild warm colour matrix + gentle S-curve tone — film print emulation."""
        # Slight warm lift on red, slight hue rotation toward orange.
        m = np.array([
            [1.05, -0.02,  0.00],
            [0.00,  1.00,  0.00],
            [0.00, -0.02,  0.98],
        ], dtype=np.float64)
        # Gentle S-curve: darkens shadows, lifts highlights slightly.
        t = np.linspace(0.0, 1.0, 256, dtype=np.float64)
        s_curve = t + 0.05 * np.sin(np.pi * t) * (1.0 - t)
        s_curve = np.clip(s_curve, 0.0, 1.0)
        return cls(color_matrix=m, tone_curve_lut=s_curve,
                   output_space="sRGB", gamma=2.2, clip_output=True)

    def apply_matrix(self, rgb: np.ndarray) -> np.ndarray:
        """Apply color_matrix to (H,W,3) or (N,3) float64 array."""
        img = np.asarray(rgb, np.float64)
        if self.color_matrix is None:
            return img
        shape = img.shape
        flat = img.reshape(-1, 3)
        out = flat @ self.color_matrix.T.astype(np.float64)
        return out.reshape(shape)

    def apply_white_balance(self, rgb: np.ndarray) -> np.ndarray:
        """Apply white_balance (3,) multiplier to (H,W,3) or (N,3) array."""
        if self.white_balance is None:
            return np.asarray(rgb, np.float64)
        wb = np.asarray(self.white_balance, np.float64).reshape(1, 1, 3)
        return np.asarray(rgb, np.float64) * wb

    def apply_tone_curve(self, rgb: np.ndarray) -> np.ndarray:
        """Apply tone_curve_lut to (H,W,3) or (N,3) float64 array via lerp."""
        img = np.asarray(rgb, np.float64)
        if self.tone_curve_lut is None:
            return img
        lut = np.asarray(self.tone_curve_lut, np.float64)
        n = max(2, len(lut))
        idx_f = np.clip(img.ravel() * (n - 1), 0.0, n - 1)
        idx0 = np.floor(idx_f).astype(np.intp)
        idx1 = np.minimum(idx0 + 1, n - 1)
        frac = idx_f - idx0.astype(np.float64)
        mapped = lut[idx0] + frac * (lut[idx1] - lut[idx0])
        return mapped.reshape(img.shape)

    def apply_gamma(self, rgb: np.ndarray) -> np.ndarray:
        """Apply output gamma / sRGB transfer to (H,W,3) or (N,3) float64 array."""
        img = np.asarray(rgb, np.float64)
        if self.output_space == "sRGB":
            lo = img <= 0.0031308
            out = np.where(lo, img * 12.92,
                           1.055 * np.power(np.maximum(img, 0.0), 1.0 / 2.4) - 0.055)
            return out
        if abs(self.gamma - 1.0) < 1e-6:
            return img
        return np.power(np.maximum(img, 0.0), 1.0 / max(self.gamma, 1e-4))


# ---------------------------------------------------------------------------
# SensorBackProfile
# ---------------------------------------------------------------------------

class SensorBackProfile:
    """Composable sensor/film back profile.

    Any combination of parts can be provided.  Missing parts leave the
    corresponding pipeline stage as a no-op.  Each profile is also a hook
    factory: call ``build_hooks()`` to get the built-in hook objects for
    the pipeline.

    Parameters
    ----------
    geometry        : BackGeometrySpec or None
    sensor_chip     : SensorChipSpec or None    — digital sensor params
    film_emulsion   : FilmEmulsionSpec or None  — chemical film params
    color_science   : ColorScienceProfile or None
    exposure_time_s : float — default integration time (overrides parts)
    energy_scale    : float — overall photon→signal scale factor
    label           : str   — human-readable profile name
    """

    def __init__(self,
                 geometry      : Optional[BackGeometrySpec]  = None,
                 sensor_chip   : Optional[SensorChipSpec]    = None,
                 film_emulsion : Optional[FilmEmulsionSpec]  = None,
                 color_science : Optional[ColorScienceProfile] = None,
                 *,
                 exposure_time_s : float = 0.010,
                 energy_scale    : float = 1.0,
                 label           : str   = "") -> None:
        self.geometry       : BackGeometrySpec    = geometry or BackGeometrySpec()
        self.sensor_chip    : Optional[SensorChipSpec]      = sensor_chip
        self.film_emulsion  : Optional[FilmEmulsionSpec]    = film_emulsion
        self.color_science  : Optional[ColorScienceProfile] = color_science
        self.exposure_time_s: float = float(exposure_time_s)
        self.energy_scale   : float = float(energy_scale)
        self.label          : str   = str(label)

    # ── Hook factory ──────────────────────────────────────────────────────────

    def build_hooks(self) -> List["RawHandoffHook"]:
        """Build the standard hook list from this profile's parts.

        Returns hooks in registration order.  Callers should call
        ``SensorBack.register_hook()`` for each returned hook.
        """
        hooks: List[RawHandoffHook] = []
        if self.sensor_chip is not None:
            hooks.append(_DevelopDigitalHook(self))
            hooks.append(_ReadoutDigitalHook(self))
        elif self.film_emulsion is not None:
            hooks.append(_DevelopFilmHook(self))
        if self.color_science is not None:
            hooks.append(_ColorScienceHook(self))
        return hooks

    # ── Named constructors ────────────────────────────────────────────────────

    @classmethod
    def medium_format_120_6x6(cls,
                               color_science: Optional[ColorScienceProfile] = None,
                               exposure_time_s: float = 0.010) -> "SensorBackProfile":
        """120 6×6 format: 56×56 mm, ~80 mm image circle, ~5.3 μm pitch digital back.

        Sensor parameters approximate a Phase One IQ-class back.
        """
        geom = BackGeometrySpec.medium_format_120_6x6()
        chip = SensorChipSpec(
            pixel_pitch_um=5.3,
            full_well_e=100_000.0,
            qe_peak=0.80,
            read_noise_e=2.2,
            dark_current_e_s=0.05,
            adc_bits=16,
            black_level_adu=256,
            white_level_adu=65535,
            cfa_layout="RGGB",
        )
        cs = color_science or ColorScienceProfile.film_print()
        return cls(geom, chip, None, cs,
                   exposure_time_s=exposure_time_s,
                   label="120_6x6_digital_back")

    @classmethod
    def medium_format_120_6x45(cls,
                                exposure_time_s: float = 0.010) -> "SensorBackProfile":
        """120 6×4.5 format: 56×42 mm portrait orientation."""
        geom = BackGeometrySpec.medium_format_120_6x45()
        chip = SensorChipSpec(
            pixel_pitch_um=5.3,
            full_well_e=100_000.0,
            qe_peak=0.78,
            read_noise_e=2.5,
            dark_current_e_s=0.06,
            adc_bits=16,
            cfa_layout="RGGB",
        )
        return cls(geom, chip, None, ColorScienceProfile.srgb_standard(),
                   exposure_time_s=exposure_time_s,
                   label="120_6x45_digital_back")

    @classmethod
    def large_format_4x5(cls,
                          exposure_time_s: float = 0.100) -> "SensorBackProfile":
        """4×5 large-format film back."""
        geom = BackGeometrySpec.large_format_4x5()
        film = FilmEmulsionSpec(
            iso=100.0,
            exposure_time_s=exposure_time_s,
            gamma_curve=0.65,
            halation_radius=0.002,
            grain_sigma=0.008,
        )
        return cls(geom, None, film, ColorScienceProfile.film_print(),
                   exposure_time_s=exposure_time_s,
                   label="4x5_film")

    @classmethod
    def fullframe_35mm(cls,
                        exposure_time_s: float = 0.010) -> "SensorBackProfile":
        """35 mm full-frame digital (Bayer RGGB, 24 MP class)."""
        geom = BackGeometrySpec.fullframe_35mm()
        chip = SensorChipSpec(
            pixel_pitch_um=4.4,
            full_well_e=150_000.0,
            qe_peak=0.78,
            read_noise_e=2.5,
            dark_current_e_s=0.10,
            adc_bits=14,
            black_level_adu=512,
            white_level_adu=16383,
        )
        return cls(geom, chip, None, ColorScienceProfile.srgb_standard(),
                   exposure_time_s=exposure_time_s,
                   label="35mm_ff_digital")

    @classmethod
    def raw_passthrough(cls) -> "SensorBackProfile":
        """No conversion, no color science — raw photon/energy values only."""
        return cls(label="raw_passthrough")

    def __repr__(self) -> str:
        return (f"SensorBackProfile(label={self.label!r}, "
                f"geometry={self.geometry.mount_standard!r}, "
                f"chip={self.sensor_chip is not None}, "
                f"film={self.film_emulsion is not None}, "
                f"cs={self.color_science is not None})")


# ---------------------------------------------------------------------------
# Built-in hooks (private)
# ---------------------------------------------------------------------------

class _DevelopDigitalHook(RawHandoffHook):
    """Photon → electron conversion for digital sensors.

    Model: electrons = raw_photons * qe_peak * exposure_time_s * energy_scale
    Shot noise: approx via Gaussian with σ = √(electrons) (Poisson approx).
    Dark current: added per-pixel, scaled by exposure_time_s.
    Well-fill: clamped to full_well_e.
    """

    def __init__(self, profile: SensorBackProfile) -> None:
        super().__init__(HookStage.DEVELOP, priority=50, name="develop_digital")
        self._profile = profile

    def __call__(self, ctx: RawHandoffContext) -> None:
        chip  = ctx.profile.sensor_chip
        if chip is None:
            return
        raw   = np.asarray(ctx.raw_image, np.float64)
        t     = ctx.profile.exposure_time_s
        scale = ctx.profile.energy_scale
        # Photon → electron
        electrons = raw * chip.qe_peak * max(t, 0.0) * max(scale, 0.0)
        # CFA sensitivity weighting (when enabled)
        if chip.apply_cfa_weights and raw.ndim >= 3 and raw.shape[-1] >= 3:
            rw, gw, bw = chip.rgb_sensitivity_weights()
            electrons[..., 0] *= rw
            electrons[..., 1] *= gw
            if raw.shape[-1] >= 3:
                electrons[..., 2] *= bw
        # Shot noise (Poisson approx): σ = sqrt(signal)
        shot_sigma = np.sqrt(np.maximum(electrons, 0.0))
        noise = np.random.default_rng().standard_normal(electrons.shape) * shot_sigma
        electrons = electrons + noise
        # Dark current
        dark = chip.dark_current_e_s * max(t, 0.0)
        electrons = electrons + dark
        # Well-fill clamp
        electrons = np.clip(electrons, 0.0, chip.full_well_e)
        ctx.electrons = electrons


class _ReadoutDigitalHook(RawHandoffHook):
    """Electron → ADU conversion (read noise, ADC quantisation, black level)."""

    def __init__(self, profile: SensorBackProfile) -> None:
        super().__init__(HookStage.READOUT, priority=50, name="readout_digital")
        self._profile = profile

    def __call__(self, ctx: RawHandoffContext) -> None:
        chip = ctx.profile.sensor_chip
        if chip is None:
            return
        src = ctx.electrons
        if src is None:
            src = np.asarray(ctx.raw_image, np.float64)
        electrons = np.asarray(src, np.float64)
        # Read noise
        read_sigma = chip.read_noise_e
        if read_sigma > 0.0:
            electrons = electrons + np.random.default_rng().standard_normal(
                electrons.shape) * read_sigma
        # Electron → ADU
        e_per_adu = chip.electrons_per_adu
        adu = electrons / max(e_per_adu, 1e-12) + chip.black_level_adu
        # Clamp to ADC range
        adu = np.clip(adu, 0.0, float(2 ** chip.adc_bits - 1))
        ctx.adu = adu


class _DevelopFilmHook(RawHandoffHook):
    """Photon → density conversion for chemical film emulsions.

    Uses the H&D curve from ``FilmEmulsionSpec.hd_curve``.  Output is
    stored in ``ctx.adu`` as a density map in [0, 1] (normalized).
    """

    def __init__(self, profile: SensorBackProfile) -> None:
        super().__init__(HookStage.DEVELOP, priority=50, name="develop_film")
        self._profile = profile

    def __call__(self, ctx: RawHandoffContext) -> None:
        film = ctx.profile.film_emulsion
        if film is None:
            return
        raw   = np.asarray(ctx.raw_image, np.float64)
        scale = ctx.profile.energy_scale * film.quantum_efficiency
        t     = ctx.profile.exposure_time_s
        # Exposure H (relative log units)
        H  = np.maximum(raw * scale * max(t, 0.0), 1e-30)
        logH = np.log10(H)
        # Vectorised H&D lookup via scalar map
        D = np.vectorize(film.hd_curve)(logH)
        D_range = max(film.shoulder_density - film.base_density, 1e-9)
        norm    = np.clip((D - film.base_density) / D_range, 0.0, 1.0)
        # Optional grain
        if film.grain_sigma > 0.0:
            grain = np.random.default_rng().standard_normal(norm.shape) * film.grain_sigma
            norm  = np.clip(norm + grain * norm, 0.0, 1.0)
        ctx.electrons = D          # raw density (physical)
        ctx.adu       = norm       # [0,1] normalised


class _ColorScienceHook(RawHandoffHook):
    """Apply white balance, color matrix, tone curve, and output gamma.

    Input is taken from ctx.adu when present, else ctx.electrons, else raw.
    For multi-channel (H,W,C) images, operates on the first 3 channels.
    Grayscale images are broadcast to (H,W,3) via CFA-weight tinting.
    """

    def __init__(self, profile: SensorBackProfile) -> None:
        super().__init__(HookStage.COLOR_SCIENCE, priority=50, name="color_science")
        self._profile = profile

    def __call__(self, ctx: RawHandoffContext) -> None:
        cs = ctx.profile.color_science
        if cs is None:
            return

        # Resolve input
        for cand in (ctx.adu, ctx.electrons, ctx.raw_image):
            if cand is not None:
                src = np.asarray(cand, np.float64)
                break

        # Normalise to [0, 1] for downstream color processing
        peak = float(np.max(np.abs(src)))
        if peak < 1e-30:
            # All-black — pass zeros through
            ctx.rgb = np.zeros(src.shape[:2] + (3,), np.float64)
            ctx.output = ctx.rgb.copy()
            return
        src_norm = src / peak   # dtype-neutral normalisation

        # Ensure (H, W, 3)
        if src_norm.ndim == 2:
            # Grayscale: replicate to RGB with sensitivity tinting
            chip = ctx.profile.sensor_chip
            if chip is not None:
                rw, gw, bw = chip.rgb_sensitivity_weights()
            else:
                rw = gw = bw = 1.0
            rgb = np.stack([src_norm * rw, src_norm * gw, src_norm * bw], axis=-1)
        elif src_norm.shape[-1] == 1:
            rgb = np.repeat(src_norm, 3, axis=-1)
        elif src_norm.shape[-1] >= 3:
            rgb = src_norm[..., :3].copy()
        else:
            rgb = np.concatenate([
                src_norm, np.zeros(src_norm.shape[:2] + (3 - src_norm.shape[-1],))
            ], axis=-1)

        rgb = cs.apply_white_balance(rgb)
        rgb = cs.apply_matrix(rgb)
        rgb = cs.apply_tone_curve(rgb)
        rgb = cs.apply_gamma(rgb)
        if cs.clip_output:
            rgb = np.clip(rgb, 0.0, 1.0)

        ctx.rgb = rgb
        ctx.output = rgb


# ---------------------------------------------------------------------------
# Frame crop helper
# ---------------------------------------------------------------------------

def _apply_frame_crop(raw: np.ndarray, geom: "BackGeometrySpec") -> np.ndarray:
    """Crop the square C++ sensor grid to match the physical frame aspect ratio.

    The C++ renders ``res × res`` pixels over the full image plane regardless
    of the film gate shape.  For a non-square format (e.g. 6×4.5 = 56×42 mm)
    the rows/columns that fall outside the film gate must be discarded.

    For a square format (6×6) the two sides are equal and this is a no-op.
    The crop is always centred.  The native dtype is preserved.

    Parameters
    ----------
    raw  : ndarray — shape (H, W) or (H, W, C); must be square (H == W).
    geom : BackGeometrySpec — provides frame_w_mm and frame_h_mm.

    Returns
    -------
    ndarray — cropped array; dtype unchanged.  If no crop is needed, the
    *same object* is returned (no copy).
    """
    if raw.ndim < 2:
        return raw
    res_h, res_w = raw.shape[0], raw.shape[1]
    if res_h != res_w:
        return raw   # already non-square — trust caller
    fw = float(geom.frame_w_mm)
    fh = float(geom.frame_h_mm)
    if fw <= 1e-6 or fh <= 1e-6:
        return raw
    if abs(fw - fh) < 1e-3 * max(fw, fh):
        return raw   # square format — no crop needed
    if fw > fh:
        # landscape: full width, trim height
        crop_w = res_w
        crop_h = max(1, round(res_h * fh / fw))
    else:
        # portrait: full height, trim width
        crop_h = res_h
        crop_w = max(1, round(res_w * fw / fh))
    cy, cx = res_h // 2, res_w // 2
    y0 = cy - crop_h // 2
    y1 = y0 + crop_h
    x0 = cx - crop_w // 2
    x1 = x0 + crop_w
    if raw.ndim == 2:
        return raw[y0:y1, x0:x1]
    return raw[y0:y1, x0:x1, ...]


# ---------------------------------------------------------------------------
# SensorBack
# ---------------------------------------------------------------------------

class SensorBack(FlatBack):
    """CameraBack subclass that owns the raw sensor image handoff pipeline.

    Install on ``ForwardCppLensBench`` as ``bench._sensor_back``.  Call
    ``receive_raw(tracer)`` in place of ``tracer.get_sensor_image()``.

    The C++ sensor image accumulator is still configured and owned by the
    tracer (``tracer.configure_sensor_image(...)``); ``SensorBack`` is
    responsible only for the readout boundary and the processing pipeline
    downstream of that boundary.

    Hook registration
    -----------------
    ``register_hook(hook_or_fn, stage, priority, name)``
        Accepts a RawHandoffHook instance, or a plain callable.  When a
        callable is passed, stage, priority, and name are forwarded.

    ``remove_hook(hook_or_name)``
        Remove by instance or by name string.

    ``clear_hooks(stage)``
        Remove all hooks at a given stage (or all stages when stage is None).

    Built-in hooks from profile
    ---------------------------
    Call ``SensorBack.from_profile(profile, res)`` to get a SensorBack with
    all profile-implied hooks pre-registered.
    """

    def __init__(self,
                 profile: Optional[SensorBackProfile] = None,
                 res_w: int = 64,
                 res_h: int = 64,
                 n_channels: int = 4) -> None:
        super().__init__(res_w=res_w, res_h=res_h, n_channels=n_channels)
        self._profile : SensorBackProfile = profile or SensorBackProfile.raw_passthrough()
        self._hooks   : List[RawHandoffHook] = []

    @classmethod
    def from_profile(cls,
                     profile: SensorBackProfile,
                     res: int = 64,
                     *,
                     res_w: Optional[int] = None,
                     res_h: Optional[int] = None) -> "SensorBack":
        """Construct a SensorBack and pre-register all profile-implied hooks."""
        geom = profile.geometry
        # Derive resolution from geometry if not explicit
        if res_w is None or res_h is None:
            if geom.res_x > 0 and geom.res_y > 0:
                rw = geom.res_x
                rh = geom.res_y
            elif geom.effective_aspect >= 1.0:
                rw = int(res)
                rh = max(1, int(round(res / geom.effective_aspect)))
            else:
                rh = int(res)
                rw = max(1, int(round(res * geom.effective_aspect)))
        else:
            rw = int(res_w)
            rh = int(res_h)

        back = cls(profile=profile, res_w=rw, res_h=rh)
        for hook in profile.build_hooks():
            back.register_hook(hook)
        return back

    # ── Hook management ───────────────────────────────────────────────────────

    def register_hook(self,
                      hook_or_fn,
                      stage: Optional[HookStage] = None,
                      priority: int = 100,
                      name: str = "") -> "RawHandoffHook":
        """Register a hook.

        Parameters
        ----------
        hook_or_fn  : RawHandoffHook or callable(RawHandoffContext)
        stage       : HookStage — ignored when hook_or_fn is a RawHandoffHook.
        priority    : int       — ignored when hook_or_fn is a RawHandoffHook.
        name        : str       — ignored when hook_or_fn is a RawHandoffHook.

        Returns the hook object that was registered.
        """
        if isinstance(hook_or_fn, RawHandoffHook):
            hook = hook_or_fn
        else:
            if stage is None:
                raise ValueError("stage must be provided when registering a callable.")
            hook = _CallableHook(hook_or_fn, HookStage(int(stage)), priority, name)
        self._hooks.append(hook)
        # Keep sorted by (stage, priority) so run() is O(N).
        self._hooks.sort(key=lambda h: (h.stage, h.priority))
        return hook

    def remove_hook(self, hook_or_name) -> bool:
        """Remove a hook by instance or by name.  Returns True if removed."""
        before = len(self._hooks)
        if isinstance(hook_or_name, str):
            self._hooks = [h for h in self._hooks if h.name != hook_or_name]
        else:
            self._hooks = [h for h in self._hooks if h is not hook_or_name]
        return len(self._hooks) < before

    def clear_hooks(self, stage: Optional[HookStage] = None) -> None:
        """Remove all hooks (or all hooks at a specific stage)."""
        if stage is None:
            self._hooks.clear()
        else:
            st = HookStage(int(stage))
            self._hooks = [h for h in self._hooks if h.stage != st]

    # ── Pipeline execution ────────────────────────────────────────────────────

    def run_pipeline(self, raw_image: np.ndarray) -> np.ndarray:
        """Run the full hook chain on a raw image and return the result.

        Parameters
        ----------
        raw_image : ndarray — as returned by tracer.get_sensor_image().
                              dtype is preserved as-is through the chain.

        Returns
        -------
        ndarray — resolved output from the chain.
        """
        ctx = RawHandoffContext(raw_image=raw_image, profile=self._profile)
        for hook in self._hooks:
            if hook.enabled:
                hook(ctx)
        return ctx.resolved_output()

    # ── Raw ownership boundary ────────────────────────────────────────────────

    def receive_raw(self, tracer) -> np.ndarray:
        """Pull the raw sensor image from *tracer* and run it through the pipeline.

        This is the ownership boundary: the lab calls this method and never
        calls ``tracer.get_sensor_image()`` directly.

        Parameters
        ----------
        tracer : _spectral_kernels.RayTracer — the C++ tracer instance.

        Returns
        -------
        ndarray — processed image.  dtype is whatever the pipeline produces.
                  Shape matches the C++ sensor image output (typically H×W×C).
        """
        raw = np.asarray(tracer.get_sensor_image())   # preserve native dtype
        raw = _apply_frame_crop(raw, self._profile.geometry)
        return self.run_pipeline(raw)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def hook_summary(self) -> str:
        """Return a one-line summary of registered hooks for logging."""
        if not self._hooks:
            return f"SensorBack(profile={self._profile.label!r}, hooks=[])"
        parts = [f"{h.stage.name}:{h.name}(p={h.priority})" for h in self._hooks]
        return f"SensorBack(profile={self._profile.label!r}, hooks=[{', '.join(parts)}])"

    @property
    def output_shape(self) -> Tuple[int, int]:
        """Expected (H, W) of the cropped output based on the profile geometry.

        Computed from ``res_w``, ``res_h`` and the frame aspect ratio so
        callers (GL renderer, display panel) can allocate buffers before the
        first frame arrives.
        """
        fw = self._profile.geometry.frame_w_mm
        fh = self._profile.geometry.frame_h_mm
        if fw <= 1e-6 or fh <= 1e-6 or abs(fw - fh) < 1e-3 * max(fw, fh):
            return (self.res_h, self.res_w)
        if fw > fh:
            return (max(1, round(self.res_h * fh / fw)), self.res_w)
        return (self.res_h, max(1, round(self.res_w * fw / fh)))

    @property
    def profile(self) -> SensorBackProfile:
        return self._profile

    @profile.setter
    def profile(self, new_profile: SensorBackProfile) -> None:
        """Replace profile and rebuild built-in hooks from the new profile.

        User-registered hooks (not from the previous profile) are preserved.
        Built-in hooks are identified by being instances of the private
        _Develop*, _Readout*, _ColorScience* classes.
        """
        _builtin_types = (_DevelopDigitalHook, _ReadoutDigitalHook,
                          _DevelopFilmHook, _ColorScienceHook)
        user_hooks = [h for h in self._hooks if not isinstance(h, _builtin_types)]
        self._profile = new_profile
        self._hooks = user_hooks
        for hook in new_profile.build_hooks():
            self.register_hook(hook)

    def __repr__(self) -> str:
        return (f"SensorBack(res={self.res_w}x{self.res_h}, "
                f"profile={self._profile!r}, hooks={len(self._hooks)})")
