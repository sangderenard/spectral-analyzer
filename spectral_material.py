"""spectral_material.py
======================
Unified material definition system shared by:
  - YAML config files (station.yaml, room YAML, instrument presets, …)
  - ray_tracer_bridge  — converts to (reflectivity, diffusion, absorption) per
                         spectral band, plus per-band emission weights
  - OpenGL shader path — packs into std140 uniform buffers and UBO blocks
  - DepthMesh / duty_station geometry pieces
  - Acoustic AND electromagnetic domains (same schema, different freq ranges)

Localized structural-colour and metasurface declarations may be carried in a
``maxwell_patch`` mapping.  This is cold authoring metadata for the future
localized Maxwell compiler described in ``MAXWELL_PATCH_CONTEXT.md``.  It is
round-tripped but is not evaluated by the current material or hot shader paths.

Domain awareness
----------------
Every material carries a ``domain`` field:

  "acoustic"     20 Hz – 20 kHz  (default; speed = 343 m/s)
  "em_optical"   380 – 780 nm    (visible light; speed = c)
  "em_ir"        700 nm – 1 mm   (near/mid/far IR)
  "em_rf"        1 mm – 1 m      (radio / microwave)
  "em_full"      DC – UV         (broadband EM; raytraced as geometric optics)

All frequency/wavelength values are stored in Hz internally.  Convenience
constructors accept nm or m and convert automatically.

Spectral response curve
-----------------------
Each material has a list of ``SpectralBand`` entries that together define
the full response function across its domain.  Each band is a Gaussian lobe:

  center_hz      peak frequency (Hz)
  bandwidth_hz   1σ bandwidth (Hz)
  reflectance    peak specular reflectance  [0, 1]
  transmittance  peak transmission           [0, 1]
                  (absorption = 1 - reflectance - transmittance, clamped ≥ 0)
  diffuse_frac   fraction of reflected energy that scatters diffusely [0, 1]
  emission       self-emission power at this band (W/sr/m² or arbitrary units)
  reemission     fraction of absorbed energy re-emitted at *this* band
                  (e.g. phosphorescence, Stokes-shifted fluorescence)
  ior_real       real part of refractive index (for Fresnel + refraction)
  ior_imag       imaginary part (extinction coeff; determines skin depth)

Parametric bandwidth
--------------------
``bandwidth_hz`` may be a scalar or a dict of the form::

  {type: "q_factor", q: 8.0, center_hz: <same as center_hz>}

This allows Q-based specification (for resonances) and octave-based
specification (for noise/diffuse bands)::

  {type: "octaves", octaves: 1.0}

The ``SpectralBand.resolve_bandwidth()`` method converts these to Hz.

GL packing
----------
``Material.to_gl_std140()`` returns a contiguous float32 array ready for
upload to a std140 UBO.  The layout is documented in the docstring of that
method and mirrored by the GLSL struct ``SpectralMaterial`` in
``SPECTRAL_MATERIAL_GLSL``.

ray_tracer_bridge integration
------------------------------
``Material.to_tracer_bands(freq_hz)`` returns ``(refl_re, refl_im, diffusion,
emission)`` arrays suitable for the existing ``_CRayTracer`` constructor and
for the emission pre-pass.

Named presets
-------------
``MATERIAL_PRESETS`` dict maps common names to ready-to-use Material objects.
All YAMLs can reference a preset by name as a single-key dict::

  material:
    preset: brushed_steel

and the loader will populate the full definition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

_C_LIGHT   = 2.997_924_58e8   # m/s
_C_SOUND   = 343.0            # m/s (standard air, 20°C)

# Domain frequency ranges (Hz)
_DOMAIN_RANGES: dict[str, tuple[float, float]] = {
    "acoustic":    (20.0,        20_000.0),
    "em_optical":  (3.84e14,     7.89e14),    # ~380–780 nm
    "em_ir":       (3.0e11,      4.3e14),     # ~700 nm down to 1 mm
    "em_rf":       (3.0e8,       3.0e11),     # 1 mm – 1 m
    "em_full":     (1.0,         3.0e17),     # near-DC to near-UV
}


def nm_to_hz(nm: float) -> float:
    """Convert wavelength in nanometres to frequency in Hz."""
    return _C_LIGHT / (nm * 1e-9)


def hz_to_nm(hz: float) -> float:
    """Convert frequency in Hz to wavelength in nanometres."""
    return _C_LIGHT / hz * 1e9


# ─────────────────────────────────────────────────────────────────────────────
# SpectralBand
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpectralBand:
    """One Gaussian lobe in the material's spectral response.

    Parameters
    ----------
    center_hz      : float  — peak frequency in Hz
    bandwidth_hz   : float | dict  — 1σ bandwidth (Hz) or parametric spec
    reflectance    : float  — peak specular reflectance [0, 1]
    transmittance  : float  — peak transmittance [0, 1]
    diffuse_frac   : float  — Lambertian fraction of reflected energy [0, 1]
    emission       : float  — self-emission power at this band
    reemission     : float  — fraction of absorbed energy re-emitted HERE
    ior_real       : float  — real refractive index
    ior_imag       : float  — extinction coefficient (imaginary ior)
    """
    center_hz:    float
    bandwidth_hz: Union[float, dict] = 1.0
    reflectance:  float = 0.5
    transmittance:float = 0.0
    diffuse_frac: float = 0.5
    emission:     float = 0.0
    reemission:   float = 0.0
    ior_real:     float = 1.5
    ior_imag:     float = 0.0

    def resolve_bandwidth(self) -> float:
        """Return bandwidth_hz as a float, resolving parametric forms."""
        bw = self.bandwidth_hz
        if isinstance(bw, (int, float)):
            return max(float(bw), 1e-6)
        if isinstance(bw, str):
            try:
                return max(float(bw), 1e-6)
            except ValueError:
                return 1.0
        if not isinstance(bw, dict):
            return 1.0
        t = bw.get("type", "")
        if t == "q_factor":
            q = float(bw.get("q", 8.0))
            return self.center_hz / max(q, 1e-6)
        if t == "octaves":
            oct_ = float(bw.get("octaves", 1.0))
            return self.center_hz * (2.0 ** (oct_ * 0.5) - 2.0 ** (-oct_ * 0.5))
        return 1.0

    @property
    def absorption(self) -> float:
        """Derived absorption = 1 - reflectance - transmittance, clamped ≥ 0."""
        return max(0.0, 1.0 - self.reflectance - self.transmittance)

    def response_at(self, freq_hz: float) -> tuple[float, float, float, float, float]:
        """Evaluate all response fields at freq_hz (Gaussian weighting).

        Returns (reflectance, transmittance, diffuse_frac, emission, reemission)
        each scaled by the Gaussian envelope exp(-0.5*(Δf/σ)²).
        """
        bw  = self.resolve_bandwidth()
        x   = (freq_hz - self.center_hz) / bw
        g   = math.exp(-0.5 * x * x)
        return (
            self.reflectance   * g,
            self.transmittance * g,
            self.diffuse_frac,   # diffuse fraction is NOT frequency-weighted
            self.emission      * g,
            self.reemission    * g,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SpectralBand":
        """Construct from a plain dict (e.g. from YAML)."""
        bw_raw = d.get("bandwidth_hz", d.get("bandwidth", 1.0))
        return cls(
            center_hz    = float(d.get("center_hz", d.get("center", 1000.0))),
            bandwidth_hz = bw_raw,
            reflectance  = float(d.get("reflectance",  d.get("reflectivity", 0.5))),
            transmittance= float(d.get("transmittance", 0.0)),
            diffuse_frac = float(d.get("diffuse_frac", d.get("diffusion", 0.5))),
            emission     = float(d.get("emission",  0.0)),
            reemission   = float(d.get("reemission", 0.0)),
            ior_real     = float(d.get("ior_real", d.get("ior", 1.5))),
            ior_imag     = float(d.get("ior_imag", 0.0)),
        )

    def to_dict(self) -> dict:
        return {
            "center_hz":    self.center_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "reflectance":  self.reflectance,
            "transmittance":self.transmittance,
            "diffuse_frac": self.diffuse_frac,
            "emission":     self.emission,
            "reemission":   self.reemission,
            "ior_real":     self.ior_real,
            "ior_imag":     self.ior_imag,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EnamelCoating  (thin dielectric film on a surface)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnamelCoating:
    """Thin dielectric coating applied over a base material surface.

    Models enamel, lacquer, clear-coat, glaze, or any thin dielectric film.
    Thickness governs Fabry–Pérot interference (iridescence in EM domain,
    surface impedance modification in acoustic domain).

    Parameters
    ----------
    thickness_m    : float — physical thickness in metres (50–200 nm typical)
    ior_real       : float — real refractive index of the coating (1.52 = glass)
    ior_imag       : float — extinction coefficient (0 = fully transparent)
    color_rgb      : list  — tint colour [r,g,b] linear sRGB (1,1,1 = colourless)
    roughness      : float — enamel surface roughness [0,1]
    spectral_bands : list[SpectralBand]  — per-band absorption/emission of the
                     coating itself (leave empty for a plain clear coat)
    """
    thickness_m:    float = 80e-9       # 80 nm: thin lacquer / clear-coat
    ior_real:       float = 1.52
    ior_imag:       float = 0.0         # 0 = fully transparent
    color_rgb:      list  = field(default_factory=lambda: [1.0, 1.0, 1.0])
    roughness:      float = 0.06
    spectral_bands: list  = field(default_factory=list)

    def fresnel_factor(self, freq_hz: np.ndarray, domain: str = "em_optical",
                       angle_cos: float = 1.0) -> np.ndarray:
        """Thin-film reflectance modulation factor at each frequency.

        Returns (n_bands,) float64.  Values oscillate around 1.0; deviation
        encodes constructive/destructive interference (iridescence).
        """
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        if self.thickness_m < 1e-13:
            return np.ones_like(freq_hz)
        if domain.startswith("em"):
            # Phase accumulation δ = 2π·n·d·cos(θ)·f / c
            delta = (2.0 * math.pi * self.ior_real * self.thickness_m
                     * float(angle_cos) * freq_hz / _C_LIGHT)
        else:
            # Acoustic: surface mass-law (m = ρ·d, Z_m = j·ω·m)
            rho_coat  = 1200.0    # kg/m³ — typical polymer/lacquer
            m_surface = rho_coat * self.thickness_m
            Z_air     = 415.0     # Pa·s/m
            omega     = 2.0 * math.pi * freq_hz
            delta     = np.arctan(omega * m_surface / Z_air)
        n    = self.ior_real
        r01  = ((n - 1.0) / (n + 1.0)) ** 2
        mod  = 1.0 + 2.0 * math.sqrt(r01) * np.cos(2.0 * delta)
        norm = 1.0 + 2.0 * math.sqrt(r01)
        return np.clip(mod / max(norm, 1e-9), 0.0, 1.5)

    @classmethod
    def from_dict(cls, d: dict) -> "EnamelCoating":
        t_m = d.get("thickness_m", d.get("thickness_nm", 80.0) * 1e-9)
        spb = [SpectralBand.from_dict(b) for b in d.get("spectral_bands", [])
               if "center_hz" in b]
        return cls(
            thickness_m    = float(t_m),
            ior_real       = float(d.get("ior_real", d.get("ior", 1.52))),
            ior_imag       = float(d.get("ior_imag", 0.0)),
            color_rgb      = [float(v) for v in
                              d.get("color_rgb", d.get("color", [1.0, 1.0, 1.0]))[:3]],
            roughness      = float(d.get("roughness", 0.06)),
            spectral_bands = spb,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "thickness_m": self.thickness_m,
            "ior_real":    self.ior_real,
            "ior_imag":    self.ior_imag,
            "color_rgb":   list(self.color_rgb),
            "roughness":   self.roughness,
        }
        if self.spectral_bands:
            d["spectral_bands"] = [b.to_dict() for b in self.spectral_bands]
        return d

    def to_gl_params(self) -> dict:
        """Dict of GL uniform name → value for shader upload."""
        return {
            "uEnamelThickness":  float(self.thickness_m),
            "uEnamelIOR":        float(self.ior_real),
            "uEnamelAbsorption": float(self.ior_imag),
            "uEnamelColor":      [float(v) for v in self.color_rgb[:3]],
        }


# ─────────────────────────────────────────────────────────────────────────────
# RadianceProfile  (photometric / radiometric emission characterisation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RadianceProfile:
    """Radiometric and photometric characterisation of an emissive material.

    Encodes luminance (cd/m²), correlated colour temperature, CRI, emission
    geometry, and an optional measured spectral power distribution (SPD).

    Parameters
    ----------
    luminance      : float — surface luminance in cd/m² (0 = non-emissive)
    cct_k          : float — correlated colour temperature in Kelvin
    cri            : float — colour rendering index (0–100)
    solid_angle_sr : float — emission solid angle; 2π = full hemisphere
    distribution   : str   — "lambertian" | "cosine_power" | "spot" | "ies"
    distribution_params : dict — e.g. {"exponent": 8.0} for cosine_power
    spectral_power_distribution : list[SpectralBand] — measured / modelled SPD
    """
    luminance:      float = 0.0
    cct_k:          float = 4000.0
    cri:            float = 80.0
    solid_angle_sr: float = math.pi * 2.0
    distribution:   str   = "lambertian"
    distribution_params: dict = field(default_factory=dict)
    spectral_power_distribution: list = field(default_factory=list)

    def srgb_color(self) -> tuple:
        """Approximate linear-sRGB colour from the CCT (Planckian locus).

        Uses the Kang et al. 2002 polynomial approximation.
        Returns (r, g, b) each in [0, 1].
        """
        T = max(1000.0, min(float(self.cct_k), 20000.0))
        if T <= 4000.0:
            x = -0.2661239e9/T**3 - 0.2343580e6/T**2 + 0.8776956e3/T + 0.179910
        else:
            x = -3.0258469e9/T**3 + 2.1070379e6/T**2 + 0.2226347e3/T + 0.240390
        if T <= 2222.0:
            y = -1.1063814*x**3 - 1.34811020*x**2 + 2.18555832*x - 0.20219683
        elif T <= 4000.0:
            y = -0.9549476*x**3 - 1.37418593*x**2 + 2.09137015*x - 0.16748867
        else:
            y =  3.0817580*x**3 - 5.8733867*x**2  + 3.75112997*x - 0.37001483
        X = (x / max(y, 1e-9))
        Z = ((1.0 - x - y) / max(y, 1e-9))
        # XYZ → linear sRGB (D65 primaries)
        r =  3.2406*X - 1.5372   - 0.4986*Z
        g = -0.9689*X + 1.8758   + 0.0415*Z
        b =  0.0557*X - 0.2040   + 1.0570*Z
        m = max(r, g, b, 1e-9)
        return (max(r, 0.0)/m, max(g, 0.0)/m, max(b, 0.0)/m)

    @classmethod
    def from_dict(cls, d: dict) -> "RadianceProfile":
        spd = [SpectralBand.from_dict(b) for b in
               d.get("spectral_power_distribution", d.get("spd", []))
               if "center_hz" in b]
        return cls(
            luminance      = float(d.get("luminance", 0.0)),
            cct_k          = float(d.get("cct_k", d.get("cct", 4000.0))),
            cri            = float(d.get("cri", 80.0)),
            solid_angle_sr = float(d.get("solid_angle_sr", math.pi * 2.0)),
            distribution   = str(d.get("distribution", "lambertian")),
            distribution_params = d.get("distribution_params", {}),
            spectral_power_distribution = spd,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "luminance":      self.luminance,
            "cct_k":          self.cct_k,
            "cri":            self.cri,
            "solid_angle_sr": self.solid_angle_sr,
            "distribution":   self.distribution,
        }
        if self.distribution_params:
            d["distribution_params"] = self.distribution_params
        if self.spectral_power_distribution:
            d["spectral_power_distribution"] = [b.to_dict()
                                                 for b in self.spectral_power_distribution]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# ReemissionMatrix  (spectrum-for-spectrum re-emission map)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReemissionMatrix:
    """Maps each incoming frequency band to an outgoing emission spectrum.

    Encodes processes where absorbed energy drives a *different* output spectrum:
      - Fluorescence / phosphorescence  (UV → visible, Stokes shift)
      - Acoustic mode conversion        (airborne → structure-borne)
      - Photoluminescence               (narrow excitation → broad emission)
      - Panel resonance re-radiation    (acoustic cavity re-emission)

    ``matrix[i, j]`` = fraction of energy absorbed in input band i that is
    re-emitted in output band j.  Rows sum to ≤ 1 (energy conservation).
    """
    in_bands:  list   = field(default_factory=list)   # list[SpectralBand]
    out_bands: list   = field(default_factory=list)   # list[SpectralBand]
    matrix:    object = None                          # (n_in, n_out) float64

    def __post_init__(self) -> None:
        if self.matrix is None and self.in_bands and self.out_bands:
            self.matrix = np.zeros(
                (len(self.in_bands), len(self.out_bands)), dtype=np.float64)
        elif self.matrix is not None:
            self.matrix = np.asarray(self.matrix, dtype=np.float64)

    def evaluate_out(self, in_freq_hz: float,
                     out_freq_hz: np.ndarray) -> np.ndarray:
        """Re-emission spectrum at out_freq_hz given excitation at in_freq_hz."""
        out_freq_hz = np.asarray(out_freq_hz, dtype=np.float64)
        if self.matrix is None or not self.in_bands or not self.out_bands:
            return np.zeros(len(out_freq_hz), np.float64)
        w_in  = np.array([b.response_at(float(in_freq_hz))[0]
                          for b in self.in_bands], dtype=np.float64)
        w_sum = w_in.sum()
        if w_sum < 1e-12:
            return np.zeros(len(out_freq_hz), np.float64)
        w_in /= w_sum
        out_weights = w_in @ self.matrix   # (n_out_bands,)
        result = np.zeros(len(out_freq_hz), np.float64)
        for b, ow in zip(self.out_bands, out_weights):
            if ow < 1e-12:
                continue
            bw = b.resolve_bandwidth()
            x  = (out_freq_hz - b.center_hz) / max(bw, 1e-9)
            result += ow * np.exp(-0.5 * x * x)
        return result

    def collapse_to_dominant(self, n_out: int = 4) -> list:
        """Approximate the matrix as dominant re-emission SpectralBand lobes."""
        if self.matrix is None or not self.out_bands:
            return []
        total_out = self.matrix.sum(axis=0)
        idx_top   = np.argsort(total_out)[::-1][:n_out]
        result    = []
        for i in idx_top:
            amp = float(total_out[i])
            if amp < 1e-9:
                continue
            b = self.out_bands[i]
            result.append(SpectralBand(
                center_hz    = b.center_hz,
                bandwidth_hz = b.resolve_bandwidth(),
                reemission   = amp,
                ior_real     = b.ior_real,
                ior_imag     = b.ior_imag,
            ))
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "ReemissionMatrix":
        in_bands  = [SpectralBand.from_dict(b) for b in d.get("in_bands",  [])
                     if "center_hz" in b]
        out_bands = [SpectralBand.from_dict(b) for b in d.get("out_bands", [])
                     if "center_hz" in b]
        mat_raw   = d.get("matrix")
        matrix    = np.array(mat_raw, dtype=np.float64) if mat_raw is not None else None
        return cls(in_bands=in_bands, out_bands=out_bands, matrix=matrix)

    def to_dict(self) -> dict:
        d: dict = {
            "in_bands":  [b.to_dict() for b in self.in_bands],
            "out_bands": [b.to_dict() for b in self.out_bands],
        }
        if self.matrix is not None:
            d["matrix"] = self.matrix.tolist()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# SpectralHistogram  (histogram-bin spectral identity)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpectralHistogram:
    """Histogram-bin based spectral identity.

    Alternative to Gaussian SpectralBand lists for materials with complex or
    tabulated spectral signatures (measured reflectance curves, IES data…).
    Each bin covers a frequency range and holds a single response value.

    ``quantity`` specifies what the values represent:
      "reflectance" | "transmittance" | "emission" | "reemission"

    ``to_spectral_bands()`` converts to SpectralBand Gaussians for use
    wherever a band list is expected.
    """
    freq_edges_hz: object = None   # np.ndarray (n_bins+1,) float64
    values:        object = None   # np.ndarray (n_bins,)   float64
    quantity:      str    = "reflectance"

    def __post_init__(self) -> None:
        if self.freq_edges_hz is not None:
            self.freq_edges_hz = np.asarray(self.freq_edges_hz, dtype=np.float64)
        if self.values is not None:
            self.values = np.asarray(self.values, dtype=np.float64)

    @classmethod
    def from_array(cls, freq_edges: np.ndarray, values: np.ndarray,
                   quantity: str = "reflectance") -> "SpectralHistogram":
        return cls(
            freq_edges_hz = np.asarray(freq_edges, np.float64),
            values        = np.asarray(values,     np.float64),
            quantity      = quantity,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SpectralHistogram":
        return cls(
            freq_edges_hz = np.array(d["freq_edges_hz"], dtype=np.float64),
            values        = np.array(d["values"],        dtype=np.float64),
            quantity      = str(d.get("quantity", "reflectance")),
        )

    def to_dict(self) -> dict:
        return {
            "freq_edges_hz": (self.freq_edges_hz.tolist()
                              if self.freq_edges_hz is not None else []),
            "values":        (self.values.tolist()
                              if self.values is not None else []),
            "quantity":      self.quantity,
        }

    def evaluate(self, freq_hz: np.ndarray) -> np.ndarray:
        """Piecewise-constant interpolation at probe frequencies."""
        freq_hz = np.asarray(freq_hz, np.float64)
        out = np.zeros_like(freq_hz)
        if self.freq_edges_hz is None or self.values is None:
            return out
        for i in range(len(self.values)):
            mask = ((freq_hz >= self.freq_edges_hz[i]) &
                    (freq_hz <  self.freq_edges_hz[i + 1]))
            out[mask] = float(self.values[i])
        return out

    def to_spectral_bands(self) -> list:
        """Convert histogram bins to SpectralBand Gaussians (one per bin)."""
        if self.freq_edges_hz is None or self.values is None:
            return []
        bands = []
        q = self.quantity
        for i in range(len(self.values)):
            v = float(self.values[i])
            if v < 1e-9:
                continue
            center = 0.5 * (float(self.freq_edges_hz[i])
                            + float(self.freq_edges_hz[i + 1]))
            bw     = 0.5 * abs(float(self.freq_edges_hz[i + 1])
                               - float(self.freq_edges_hz[i]))
            bands.append(SpectralBand(
                center_hz     = center,
                bandwidth_hz  = max(bw, center * 0.01),
                reflectance   = v if q == "reflectance"   else 0.0,
                transmittance = v if q == "transmittance" else 0.0,
                diffuse_frac  = 0.5,
                emission      = v if q == "emission"   else 0.0,
                reemission    = v if q == "reemission" else 0.0,
            ))
        return bands


# ─────────────────────────────────────────────────────────────────────────────
# Material
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Material:
    """Full spectral material description.

    Holds PBR base parameters (for OpenGL fast path) and a list of
    SpectralBand entries (for ray tracing, spectral rendering, audio).

    When ``spectral_bands`` is empty the class derives synthetic bands from
    the PBR parameters automatically via ``_synthetic_bands()``.

    Parameters
    ----------
    name           : str   — human-readable identifier
    domain         : str   — "acoustic" | "em_optical" | "em_ir" | "em_rf" | "em_full"

    PBR base parameters (OpenGL fast-path)
    ----------------------------------------
    albedo         : [r, g, b]  linear sRGB [0, 1]
    roughness      : float      0 = mirror, 1 = Lambertian
    metallic       : float      0 = dielectric, 1 = conductor
    emission_rgb   : [r, g, b]  self-emission colour × intensity
    ior            : float      bulk index of refraction
    transmission   : float      bulk transmission [0, 1]
    normal_map     : str | None path to tangent-space normal map
    albedo_texture : str | None path to albedo texture

    Spectral description
    --------------------
    spectral_bands : list[SpectralBand]
        Empty list → synthetic bands derived from PBR params at query time.
        Non-empty → used directly for ray-tracing and spectral evaluation.
    spectral_identity : SpectralHistogram | None
        Histogram-bin based spectral identity.  When set, its bins are
        converted to SpectralBand Gaussians and merged with spectral_bands.
    enamel : EnamelCoating | None
        Thin dielectric coating (adds Fabry-Pérot Fresnel / iridescence).
    radiance : RadianceProfile | None
        Photometric emission profile (for light-source materials).
    reemission_matrix : ReemissionMatrix | None
        Per-band input→output re-emission mapping (fluorescence, Stokes shift,
        acoustic mode conversion).
    maxwell_patch : dict | None
        Cold localized microstructure declaration. It is resolved into a
        versioned polarized scattering artifact by a scene-compile step; it is
        never packed into ordinary hot material rows or solved in a shader hit.
        See ``MAXWELL_PATCH_CONTEXT.md`` for the schema and runtime contract.
    """
    name:          str   = "unnamed"
    domain:        str   = "acoustic"

    # PBR base
    albedo:        list  = field(default_factory=lambda: [0.5, 0.5, 0.5])
    roughness:     float = 0.5
    metallic:      float = 0.0
    emission_rgb:        list  = field(default_factory=lambda: [0.0, 0.0, 0.0])
    reactive_shift_hz:   float = 0.0    # Stokes shift for re-emission (Hz); 0 = non-reactive
    ior:                 float = 1.5
    transmission:        float = 0.0
    normal_map:    Optional[str] = None
    albedo_texture:Optional[str] = None
    gl_opacity:    Optional[float] = None  # GL visual alpha; overrides 1-transmission

    # Spectral description
    spectral_bands:    list                        = field(default_factory=list)
    spectral_identity: Optional["SpectralHistogram"]  = None
    enamel:            Optional["EnamelCoating"]      = None
    radiance:          Optional["RadianceProfile"]    = None
    reemission_matrix: Optional["ReemissionMatrix"]   = None
    maxwell_patch:     Optional[dict]                 = None

    # ── Synthetic band fallback ───────────────────────────────────────────────

    def _synthetic_bands(self, n: int = 4) -> list:
        """Derive SpectralBand list from PBR params when spectral_bands is empty.

        Produces n log-spaced bands spanning the domain range.  PBR params
        govern the magnitude; emission_rgb luminance seeds emission.
        """
        lo, hi = _DOMAIN_RANGES.get(self.domain, (20.0, 20000.0))
        centers = np.geomspace(lo, hi, n).tolist()
        bands = []
        # Average albedo luminance drives reflectance.
        lum    = float(0.2126 * self.albedo[0] + 0.7152 * self.albedo[1]
                       + 0.0722 * self.albedo[2])
        refl   = lum * (1.0 - self.roughness * 0.4)
        diff_f = self.roughness
        trans  = self.transmission
        emit_lum = float(0.2126 * self.emission_rgb[0]
                         + 0.7152 * self.emission_rgb[1]
                         + 0.0722 * self.emission_rgb[2])
        for c in centers:
            bw = c * 0.5   # one-octave bandwidth per synthetic band
            bands.append(SpectralBand(
                center_hz    = c,
                bandwidth_hz = bw,
                reflectance  = refl,
                transmittance= trans,
                diffuse_frac = diff_f,
                emission     = emit_lum,
                reemission   = 0.0,
                ior_real     = self.ior,
                ior_imag     = self.metallic * 3.0,  # crude extinction for metals
            ))
        return bands

    def effective_bands(self, n_synthetic: int = 4) -> list:
        """Return the full band list: explicit bands + histogram bins + synthetics.

        Priority:
          1. self.spectral_bands (explicit, always included)
          2. self.spectral_identity histogram converted to Gaussian bands
          3. If still empty, synthesise from PBR params
        """
        bands = list(self.spectral_bands)
        if self.spectral_identity is not None:
            bands = bands + self.spectral_identity.to_spectral_bands()
        if not bands:
            bands = self._synthetic_bands(n_synthetic)
        return bands

    # ── Spectral evaluation ───────────────────────────────────────────────────

    def evaluate(self, freq_hz: float) -> dict:
        """Evaluate all response fields at a single frequency.

        Returns a dict with keys:
          reflectance, transmittance, absorption, diffuse_frac,
          emission, reemission, ior_real, ior_imag, specular_frac
        All values are sums over the band Gaussians.
        """
        bands = self.effective_bands()
        refl = trans = diff = emit = reemit = ior_r = ior_i = 0.0
        for b in bands:
            r, t, df, em, re = b.response_at(freq_hz)
            refl  += r
            trans += t
            diff  += df * r         # weighted by reflectance contribution
            emit  += em
            reemit+= re
            bw     = b.resolve_bandwidth()
            g      = math.exp(-0.5 * ((freq_hz - b.center_hz) / bw) ** 2)
            ior_r += b.ior_real * g
            ior_i += b.ior_imag * g

        refl  = min(refl,  1.0)
        trans = min(trans, max(0.0, 1.0 - refl))
        abso  = max(0.0,   1.0 - refl - trans)
        diff_f = min(diff / (refl + 1e-9), 1.0)
        n_b = len(bands) or 1
        return {
            "reflectance":  refl,
            "transmittance":trans,
            "absorption":   abso,
            "diffuse_frac": diff_f,
            "specular_frac":1.0 - diff_f,
            "emission":     emit,
            "reemission":   reemit,
            "ior_real":     ior_r / n_b,
            "ior_imag":     ior_i / n_b,
        }

    # ── ray_tracer_bridge interface ───────────────────────────────────────────

    def to_tracer_bands(
            self,
            freq_hz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays suitable for the _CRayTracer constructor.

        Parameters
        ----------
        freq_hz : (n_bands,) float64  — band centre frequencies in Hz

        Returns
        -------
        refl_re   : (n_bands,) float64  — real part of complex reflectance
        refl_im   : (n_bands,) float64  — imaginary part (KK phase)
        diffusion : (n_bands,) float64  — diffuse fraction at each band
        emission  : (n_bands,) float64  — self-emission power at each band
        reemission: (n_bands,) float64  — re-emission coefficient at each band
        """
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        n = len(freq_hz)
        refl_re   = np.zeros(n, np.float64)
        refl_im   = np.zeros(n, np.float64)
        diffusion = np.zeros(n, np.float64)
        emission  = np.zeros(n, np.float64)
        reemission= np.zeros(n, np.float64)

        for bi, b in enumerate(self.effective_bands()):
            bw   = b.resolve_bandwidth()
            x    = (freq_hz - b.center_hz) / bw
            g    = np.exp(-0.5 * x * x)
            r    = b.reflectance  * g

            # Kramers–Kronig–consistent imaginary part (same model as bridge)
            log_ratio = np.log(np.maximum(freq_hz / max(b.center_hz, 1e-9), 1e-12))
            phi       = np.arctan(b.absorption * log_ratio / math.pi)

            refl_re   += r * np.cos(phi)
            refl_im   += r * np.sin(phi)
            diffusion += b.diffuse_frac * g          # will be normalised below
            emission  += b.emission   * g
            reemission+= b.reemission * g

        # Normalise diffusion to [0,1] per band (sum of Gaussians can exceed 1)
        total_g = sum(
            np.exp(-0.5 * ((freq_hz - b.center_hz) / b.resolve_bandwidth()) ** 2)
            for b in self.effective_bands()
        )
        diffusion = np.where(total_g > 1e-9, diffusion / total_g, 0.0)
        diffusion = np.clip(diffusion, 0.0, 1.0)

        return refl_re, refl_im, diffusion, emission, reemission

    def to_tracer_mat_row(self, freq_hz: np.ndarray) -> np.ndarray:
        """Return a (3,) float64 row [mean_reflectance, mean_diffusion, mean_absorption].

        Convenience for legacy code that expects a single (N_tri, 3) mat_props array.
        Uses the mean over supplied frequencies.
        """
        refl_re, refl_im, diff, emit, reemit = self.to_tracer_bands(freq_hz)
        mag   = np.sqrt(refl_re**2 + refl_im**2)
        return np.array([mag.mean(), diff.mean(),
                         max(0.0, 1.0 - mag.mean() - diff.mean() * mag.mean())],
                        dtype=np.float64)

    # ── OpenGL packing ────────────────────────────────────────────────────────

    # Number of spectral bands packed into the std140 UBO.
    # Must match _MAX_SPEC_BANDS in SPECTRAL_MATERIAL_GLSL.
    MAX_GL_BANDS: int = 8

    def to_gl_std140(self) -> np.ndarray:
        """Pack into a contiguous float32 array matching GLSL SpectralMaterial std140.

        Layout (see SPECTRAL_MATERIAL_GLSL for the matching GLSL struct)
        ----------------------------------------------------------------
        Offset  Size  Name
           0     4    albedo.rgb + roughness          (vec4)
           4     4    emission_rgb + metallic         (vec4)
           8     4    ior, transmission, _pad×2        (vec4)
           12    4×MAX_GL_BANDS   band centre_hz[i]   (float[])
           12+M  4×M  band bw_hz[i]                  (float[])
           12+2M 4×M  band reflectance[i]             (float[])
           12+3M 4×M  band transmittance[i]           (float[])
           12+4M 4×M  band diffuse_frac[i]            (float[])
           12+5M 4×M  band emission[i]                (float[])
           12+6M 4×M  band reemission[i]              (float[])
           12+7M 4×M  band ior_real[i]                (float[])
           12+8M 4×M  band ior_imag[i]                (float[])
           12+9M 1    n_bands (as float)              (float)
           12+9M+1  3  _pad                           (vec3 padding to vec4)
        where M = MAX_GL_BANDS.
        Total floats = 12 + 9*M + 4 (padded to vec4 boundary) = 88 for M=8.
        """
        M = self.MAX_GL_BANDS
        out = np.zeros(12 + 9 * M + 4, dtype=np.float32)

        # Base PBR block
        out[0:3]  = np.array(self.albedo[:3], np.float32)
        out[3]    = float(self.roughness)
        out[4:7]  = np.array(self.emission_rgb[:3], np.float32)
        out[7]    = float(self.metallic)
        out[8]    = float(self.ior)
        out[9]    = float(self.transmission)
        # out[10], out[11] padding

        # Spectral bands
        bands = self.effective_bands(n_synthetic=M)[:M]
        n_packed = len(bands)
        for i, b in enumerate(bands):
            bw = b.resolve_bandwidth()
            base = 12 + i
            out[12 +       i] = float(b.center_hz)
            out[12 + M   + i] = float(bw)
            out[12 + 2*M + i] = float(b.reflectance)
            out[12 + 3*M + i] = float(b.transmittance)
            out[12 + 4*M + i] = float(b.diffuse_frac)
            out[12 + 5*M + i] = float(b.emission)
            out[12 + 6*M + i] = float(b.reemission)
            out[12 + 7*M + i] = float(b.ior_real)
            out[12 + 8*M + i] = float(b.ior_imag)

        out[12 + 9*M] = float(n_packed)
        return out

    def to_phong(self) -> dict:
        """Fast-path Phong dict for the existing ``_STATION_BODY_FS`` shader.

        Derives Phong params from the PBR albedo / roughness / metallic fields.
        Compatible with ``_pbr_to_phong()`` in duty_station.py.
        """
        lum  = float(0.2126 * self.albedo[0] + 0.7152 * self.albedo[1]
                     + 0.0722 * self.albedo[2])
        rough = float(self.roughness)
        metal = float(self.metallic)
        return {
            "albedo_rgb":    list(self.albedo[:3]),
            "ambient":       0.15 + (1.0 - rough) * 0.05,
            "spec_strength": metal * 0.85 + (1.0 - metal) * 0.04 * (1.0 - rough),
            "shininess":     max(4.0, 2.0 / max(rough ** 2, 0.01)),
            "grain":         rough * 0.06,
        }

    # ── YAML round-trip ───────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict, name: str = "unnamed") -> "Material":
        """Construct from a plain dict (e.g. from station.yaml piece_materials).

        Accepts both old-style dicts (albedo_rgb, spec_strength, …) and new
        PBR dicts (albedo, roughness, metallic, …).  Also accepts:
          ``preset: <name>`` — returns a copy of the named preset.
          ``spectral_bands:`` list — parsed into SpectralBand objects.
        """
        if "preset" in d:
            p = MATERIAL_PRESETS.get(d["preset"])
            if p is None:
                raise ValueError(f"Unknown material preset: {d['preset']!r}")
            return p

        # Colour — accept 'albedo' or 'albedo_rgb' (legacy)
        albedo = d.get("albedo", d.get("albedo_rgb", [0.5, 0.5, 0.5]))
        # Emission — accept 'emission' (new PBR list) or legacy 'emissive'
        emit_raw = d.get("emission", d.get("emissive", [0.0, 0.0, 0.0]))
        if isinstance(emit_raw, (int, float)):
            emit_raw = [float(emit_raw)] * 3

        # Spectral bands
        bands_raw = d.get("spectral_bands", d.get("bands", []))
        # Filter: bands_raw may be the YAML wall bands (which have a 'top' key);
        # those are WallBands, not SpectralBands — skip them silently.
        spec_bands = []
        for b in bands_raw:
            if isinstance(b, dict) and "center_hz" in b:
                spec_bands.append(SpectralBand.from_dict(b))

        enamel_raw = d.get("enamel")
        enamel = EnamelCoating.from_dict(enamel_raw) if enamel_raw else None

        rad_raw = d.get("radiance")
        radiance = RadianceProfile.from_dict(rad_raw) if rad_raw else None

        remit_raw = d.get("reemission_matrix")
        reemission_matrix = (ReemissionMatrix.from_dict(remit_raw)
                             if remit_raw else None)

        si_raw = d.get("spectral_identity", d.get("histogram"))
        spectral_identity = (SpectralHistogram.from_dict(si_raw)
                             if si_raw else None)

        maxwell_patch_raw = d.get("maxwell_patch")
        if maxwell_patch_raw is not None and not isinstance(maxwell_patch_raw, dict):
            raise ValueError("maxwell_patch must be a mapping")
        maxwell_patch = (dict(maxwell_patch_raw)
                         if maxwell_patch_raw is not None else None)

        return cls(
            name              = name,
            domain            = str(d.get("domain", "acoustic")),
            albedo            = [float(v) for v in albedo[:3]],
            roughness         = float(d.get("roughness",  d.get("smoothness", 0.5))),
            metallic          = float(d.get("metallic",   0.0)),
            emission_rgb      = [float(v) for v in emit_raw[:3]],
            reactive_shift_hz = float(d.get("reactive_shift_hz", 0.0)),
            ior               = float(d.get("ior",         1.5)),
            transmission      = float(d.get("transmission", 0.0)),
            normal_map        = d.get("normal_map"),
            albedo_texture    = d.get("albedo_texture"),
            spectral_bands    = spec_bands,
            spectral_identity = spectral_identity,
            enamel            = enamel,
            radiance          = radiance,
            reemission_matrix = reemission_matrix,
            maxwell_patch     = maxwell_patch,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "name":          self.name,
            "domain":        self.domain,
            "albedo":        list(self.albedo),
            "roughness":     self.roughness,
            "metallic":      self.metallic,
            "emission":           list(self.emission_rgb),
            "reactive_shift_hz":  self.reactive_shift_hz,
            "ior":                self.ior,
            "transmission":  self.transmission,
        }
        if self.normal_map:
            d["normal_map"] = self.normal_map
        if self.albedo_texture:
            d["albedo_texture"] = self.albedo_texture
        if self.spectral_bands:
            d["spectral_bands"] = [b.to_dict() for b in self.spectral_bands]
        if self.spectral_identity:
            d["spectral_identity"] = self.spectral_identity.to_dict()
        if self.enamel:
            d["enamel"] = self.enamel.to_dict()
        if self.radiance:
            d["radiance"] = self.radiance.to_dict()
        if self.reemission_matrix:
            d["reemission_matrix"] = self.reemission_matrix.to_dict()
        if self.maxwell_patch is not None:
            d["maxwell_patch"] = dict(self.maxwell_patch)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# LightSource
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LightSource:
    """Emissive light object for scene lighting.

    Carries radiometric properties (via ``radiance``) and an optional full
    spectral material (via ``material``) for ray-tracing.

    ``shape`` governs geometric extent:
      "point"   — zero-area; ``size_m`` ignored
      "disk"    — circular disk; ``size_m`` = radius
      "rect"    — rectangular panel; ``size_m`` = [width, height]
      "sphere"  — spherical; ``size_m`` = radius
      "linear"  — line / tube; ``size_m`` = length
      "strip"   — LED strip; ``size_m`` = [length, width]

    ``falloff`` controls intensity with distance:
      "inverse_square" — physical 1/r² (default)
      "linear"         — 1/r
      "constant"       — no falloff (ambient / sky)
    """
    name:      str   = "light"
    position:  list  = field(default_factory=lambda: [0.0, 0.0, 2.5])
    direction: list  = field(default_factory=lambda: [0.0, -1.0, 0.0])
    color_rgb: list  = field(default_factory=lambda: [1.0, 1.0, 1.0])
    intensity: float = 1.0
    shape:     str   = "point"
    size_m:    Union[float, list] = 0.1
    falloff:   str   = "inverse_square"
    radiance:  Optional[RadianceProfile] = None
    material:  Optional[Material]        = None

    @classmethod
    def from_dict(cls, d: dict, name: str = "light") -> "LightSource":
        rad = (RadianceProfile.from_dict(d["radiance"])
               if "radiance" in d else None)
        mat = (Material.from_dict(d["material"], name=name + "_mat")
               if "material" in d else None)
        return cls(
            name      = str(d.get("name", name)),
            position  = [float(v) for v in d.get("position", [0.0, 0.0, 2.5])[:3]],
            direction = [float(v) for v in d.get("direction", [0.0, -1.0, 0.0])[:3]],
            color_rgb = [float(v) for v in
                         d.get("color_rgb", d.get("color", [1.0, 1.0, 1.0]))[:3]],
            intensity = float(d.get("intensity", 1.0)),
            shape     = str(d.get("shape", "point")),
            size_m    = d.get("size_m", d.get("size", 0.1)),
            falloff   = str(d.get("falloff", "inverse_square")),
            radiance  = rad,
            material  = mat,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "name":      self.name,
            "position":  list(self.position),
            "direction": list(self.direction),
            "color_rgb": list(self.color_rgb),
            "intensity": self.intensity,
            "shape":     self.shape,
            "size_m":    self.size_m,
            "falloff":   self.falloff,
        }
        if self.radiance:  d["radiance"]  = self.radiance.to_dict()
        if self.material:  d["material"]  = self.material.to_dict()
        return d

    def emission_at(self, freq_hz: np.ndarray) -> np.ndarray:
        """Spectral emission power at the given frequency bands."""
        freq_hz = np.asarray(freq_hz, np.float64)
        if self.material is not None:
            _, _, _, em, _ = self.material.to_tracer_bands(freq_hz)
            lum = float(0.2126 * self.color_rgb[0] + 0.7152 * self.color_rgb[1]
                        + 0.0722 * self.color_rgb[2])
            return em * self.intensity * lum
        if self.radiance and self.radiance.spectral_power_distribution:
            m = Material(spectral_bands=self.radiance.spectral_power_distribution)
            _, _, _, em, _ = m.to_tracer_bands(freq_hz)
            return em * self.intensity
        return np.full(len(freq_hz), float(self.intensity), dtype=np.float64)

    def to_gl_source_vec(self) -> np.ndarray:
        """Compact (10,) float32: pos(3) dir(3) rgb×intensity(3) intensity(1)."""
        arr = np.zeros(10, np.float32)
        arr[0:3] = np.array(self.position[:3],  np.float32)
        arr[3:6] = np.array(self.direction[:3], np.float32)
        arr[6:9] = np.array(self.color_rgb[:3], np.float32) * float(self.intensity)
        arr[9]   = float(self.intensity)
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# LightDistributionPolicy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LightDistributionPolicy:
    """Defines how lights are distributed over a room surface.

    Used by room_control to auto-generate ceiling / wall / floor light arrays.

    ``symmetry`` applies spatial mirroring to the generated grid:
      "none"      — asymmetric placement
      "bilateral" — mirror across the primary-axis midpoint
      "quadrant"  — four-fold symmetry
      "radial"    — rotational (N-fold, use distribution_params)

    ``packing`` controls the layout algorithm:
      "uniform"   — regular grid
      "perimeter" — lights along the perimeter only
      "center"    — clustered at the centroid
      "strip"     — single row / column
      "manual"    — use manual_positions exactly
    """
    target:     str   = "ceiling"
    symmetry:   str   = "bilateral"
    packing:    str   = "uniform"
    count:      Optional[int]   = None
    spacing_m:  Optional[float] = None
    light_type: str   = "point"
    light_params: dict = field(default_factory=dict)
    manual_positions: list = field(default_factory=list)

    def generate_lights(self, bbox_min: np.ndarray,
                        bbox_max: np.ndarray) -> list:
        """Generate list[LightSource] within the room bounding box."""
        bbox_min = np.asarray(bbox_min, np.float64)
        bbox_max = np.asarray(bbox_max, np.float64)
        lights: list = []

        if self.target == "ceiling":
            fixed_y = float(bbox_max[1]) - 0.05
            normal  = [0.0, -1.0, 0.0]
            x_lo, x_hi = float(bbox_min[0]), float(bbox_max[0])
            z_lo, z_hi = float(bbox_min[2]), float(bbox_max[2])
        elif self.target == "floor":
            fixed_y = float(bbox_min[1]) + 0.05
            normal  = [0.0, 1.0, 0.0]
            x_lo, x_hi = float(bbox_min[0]), float(bbox_max[0])
            z_lo, z_hi = float(bbox_min[2]), float(bbox_max[2])
        else:
            # Wall — mount on back face (max-Z)
            fixed_y = (float(bbox_min[1]) + float(bbox_max[1])) * 0.5
            normal  = [0.0, 0.0, -1.0]
            x_lo, x_hi = float(bbox_min[0]), float(bbox_max[0])
            z_lo, z_hi = float(bbox_min[1]), float(bbox_max[1])

        if self.packing == "manual":
            for i, pos in enumerate(self.manual_positions):
                lights.append(LightSource.from_dict(
                    dict(position=list(pos), direction=normal, **self.light_params),
                    name=f"{self.target}_{i}"))
            return lights

        # Determine grid dimensions
        if self.spacing_m:
            nx = max(1, round((x_hi - x_lo) / self.spacing_m))
            nz = max(1, round((z_hi - z_lo) / self.spacing_m))
        elif self.count:
            nx = max(1, round(math.sqrt(float(self.count))))
            nz = max(1, int(math.ceil(float(self.count) / nx)))
        else:
            nx = nz = 2

        if self.packing == "strip":
            nx, nz = max(nx, nz), 1

        xs = np.linspace(x_lo, x_hi, nx + 2)[1:-1]
        zs = np.linspace(z_lo, z_hi, nz + 2)[1:-1]

        if self.symmetry in ("bilateral", "quadrant"):
            cx   = (x_lo + x_hi) * 0.5
            hi   = xs[xs >= cx]
            xs   = np.unique(np.concatenate([cx * 2 - hi[::-1], hi]))

        idx = 0
        for x in xs:
            for z in zs:
                if self.target in ("ceiling", "floor"):
                    pos = [float(x), fixed_y, float(z)]
                else:
                    pos = [float(x), float(z), float(bbox_max[2]) - 0.05]
                lights.append(LightSource.from_dict(
                    dict(position=pos, direction=normal, **self.light_params),
                    name=f"{self.target}_{idx}"))
                idx += 1
        return lights

    @classmethod
    def from_dict(cls, d: dict) -> "LightDistributionPolicy":
        return cls(
            target           = str(d.get("target",     "ceiling")),
            symmetry         = str(d.get("symmetry",   "bilateral")),
            packing          = str(d.get("packing",    "uniform")),
            count            = int(d["count"])       if "count"     in d else None,
            spacing_m        = float(d["spacing_m"]) if "spacing_m" in d else None,
            light_type       = str(d.get("light_type", "point")),
            light_params     = d.get("light_params", {}),
            manual_positions = [list(p) for p in d.get("manual_positions", [])],
        )

    def to_dict(self) -> dict:
        d: dict = {
            "target":     self.target,
            "symmetry":   self.symmetry,
            "packing":    self.packing,
            "light_type": self.light_type,
        }
        if self.count     is not None: d["count"]     = self.count
        if self.spacing_m is not None: d["spacing_m"] = self.spacing_m
        if self.light_params:          d["light_params"] = self.light_params
        if self.manual_positions:      d["manual_positions"] = self.manual_positions
        return d


# ─────────────────────────────────────────────────────────────────────────────
# WallBand  (material zone at a normalised height on a cornerstone wall)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WallBand:
    """One material zone of the cornerstone wall.

    ``top`` is a normalised wall height (0 = floor level, 1 = wall top).
    The zone covers wall heights from the previous band's top to this top.
    ``material`` is the full spectral Material for this zone.
    """
    top:      float
    material: Material

    @classmethod
    def from_dict(cls, d: dict, name: str = "wall_band") -> "WallBand":
        mat_d = d.get("material", {})
        return cls(
            top      = float(d.get("top", 1.0)),
            material = Material.from_dict(mat_d, name=name),
        )

    def to_dict(self) -> dict:
        return {"top": self.top, "material": self.material.to_dict()}


def parse_wall_bands(bands_list: list) -> list:
    """Parse a YAML bands list into a list of WallBand objects (sorted by top)."""
    result = [WallBand.from_dict(b, name=f"band_{i}")
              for i, b in enumerate(bands_list)]
    result.sort(key=lambda wb: wb.top)
    # Ensure the last band reaches 1.0
    if result and result[-1].top < 1.0 - 1e-6:
        result[-1] = WallBand(top=1.0, material=result[-1].material)
    return result


def wall_band_at(bands: list, height_norm: float) -> Optional[Material]:
    """Return the Material for the given normalised wall height (0..1)."""
    for wb in bands:
        if height_norm <= wb.top + 1e-9:
            return wb.material
    return bands[-1].material if bands else None


# ─────────────────────────────────────────────────────────────────────────────
# GLSL source for SpectralMaterial std140 UBO
# ─────────────────────────────────────────────────────────────────────────────

_MAX_SPEC_BANDS_DEFAULT = 8

SPECTRAL_MATERIAL_GLSL = f"""\
// ── SpectralMaterial UBO — generated by spectral_material.py ─────────────────
// Matches Material.to_gl_std140() layout exactly.
// M = {_MAX_SPEC_BANDS_DEFAULT}  (must equal Material.MAX_GL_BANDS in Python)
#define MAX_SPEC_BANDS {_MAX_SPEC_BANDS_DEFAULT}

struct SpectralBandGL {{
    float center_hz;
    float bw_hz;
    float reflectance;
    float transmittance;
    float diffuse_frac;
    float emission;
    float reemission;
    float ior_real;
    float ior_imag;
    float _pad0, _pad1, _pad2;  // align to vec4
}};

struct SpectralMaterial {{
    vec4  albedo_rough;      // .rgb = albedo, .a = roughness
    vec4  emission_metal;    // .rgb = emission_rgb, .a = metallic
    vec4  ior_trans;         // .x = ior, .y = transmission, .zw = pad
    // spectral bands (array-of-struct packed as float arrays for alignment)
    float band_center_hz   [MAX_SPEC_BANDS];
    float band_bw_hz       [MAX_SPEC_BANDS];
    float band_reflectance [MAX_SPEC_BANDS];
    float band_transmittance[MAX_SPEC_BANDS];
    float band_diffuse_frac[MAX_SPEC_BANDS];
    float band_emission    [MAX_SPEC_BANDS];
    float band_reemission  [MAX_SPEC_BANDS];
    float band_ior_real    [MAX_SPEC_BANDS];
    float band_ior_imag    [MAX_SPEC_BANDS];
    float n_bands;
    float _pad1, _pad2, _pad3;
}};

// Evaluate total reflectance at a normalised probe frequency [0,1] mapped
// linearly across the current band range.  Used in fragment shaders for
// spectral tinting and emission glow.
float spectral_reflectance(SpectralMaterial m, float t_norm) {{
    float total = 0.0;
    for (int i = 0; i < int(m.n_bands); i++) {{
        float c  = m.band_center_hz[i];
        float bw = max(m.band_bw_hz[i], 1e-6);
        // t_norm in [0,1] → probe_hz via log-spacing hint stored in center_hz
        // For GL shaders the caller passes probe_hz directly as a uniform.
        float x  = (c - t_norm) / bw;   // t_norm treated as Hz surrogate
        total   += m.band_reflectance[i] * exp(-0.5 * x * x);
    }}
    return clamp(total, 0.0, 1.0);
}}

// Self-emission power at t_norm (same frequency surrogate convention).
float spectral_emission(SpectralMaterial m, float t_norm) {{
    float total = 0.0;
    for (int i = 0; i < int(m.n_bands); i++) {{
        float c  = m.band_center_hz[i];
        float bw = max(m.band_bw_hz[i], 1e-6);
        float x  = (c - t_norm) / bw;
        total   += m.band_emission[i] * exp(-0.5 * x * x);
    }}
    return max(total, 0.0);
}}

// Re-emission power (Stokes shift / phosphorescence / acoustic re-radiation).
float spectral_reemission(SpectralMaterial m, float t_norm) {{
    float total = 0.0;
    for (int i = 0; i < int(m.n_bands); i++) {{
        float c  = m.band_center_hz[i];
        float bw = max(m.band_bw_hz[i], 1e-6);
        float x  = (c - t_norm) / bw;
        total   += m.band_reemission[i] * exp(-0.5 * x * x);
    }}
    return max(total, 0.0);
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Spectral-PBR fragment shader (replaces _STATION_BODY_FS in duty_station.py)
# ─────────────────────────────────────────────────────────────────────────────

SPECTRAL_PBR_BODY_FS = """\
#version 330 core
// ── Spectral-PBR fragment shader ─────────────────────────────────────────────
// Replaces the legacy Phong _STATION_BODY_FS.
// Supports per-fragment spectral tint, self-emission, and re-emission glow
// without requiring a full UBO upload — parameters are passed as scalar
// uniforms for compatibility with all hardware.
//
// Spectral uniforms
// -----------------
// uBandCenters[MAX_SPEC_BANDS]  — band centre frequencies (normalised 0..1
//                                  mapping lo..hi within the current domain)
// uBandBW[MAX_SPEC_BANDS]       — band half-widths (same normalised units)
// uBandReflect[MAX_SPEC_BANDS]  — per-band peak reflectance
// uBandEmit[MAX_SPEC_BANDS]     — per-band peak emission
// uBandReemit[MAX_SPEC_BANDS]   — per-band peak re-emission
// uNBands                       — number of active bands (float, cast to int)
// uProbeFreq                    — normalised probe frequency for spectral tint
// uSpecTintStrength             — blend weight of spectral tint vs albedo (0..1)
in  vec3 vNormV;
in  vec3 vPosV;
out vec4 FragColor;

// Base PBR uniforms (always required)
uniform vec4  uColor;           // albedo.rgb + alpha
uniform vec3  uLightV;          // view-space light direction
uniform float uAmbient;
uniform float uSpecStrength;
uniform float uShininess;
uniform float uGrain;
uniform float uRoughness;
uniform float uMetallic;
uniform vec3  uEmission;        // additive self-emission colour
// Enamel thin-film coating (Fabry-Pérot; set uEnamelThickness=0 to disable)
uniform float uEnamelThickness;   // metres
uniform float uEnamelIOR;         // refractive index of coating (1.5 = glass)
uniform float uEnamelAbsorption;  // extinction coefficient (ior_imag)
uniform vec3  uEnamelColor;       // tint colour [0,1]^3 (1,1,1 = colourless)

// Spectral band uniforms (optional; uNBands==0 falls back to pure Phong)
#define MAX_SPEC_BANDS 8
uniform float uBandCenters[MAX_SPEC_BANDS];
uniform float uBandBW[MAX_SPEC_BANDS];
uniform float uBandReflect[MAX_SPEC_BANDS];
uniform float uBandEmit[MAX_SPEC_BANDS];
uniform float uBandReemit[MAX_SPEC_BANDS];
uniform float uNBands;
uniform float uProbeFreq;
uniform float uSpecTintStrength;

// Scene-level spectral field uniforms (from SceneFieldIntegration)
// Injected as a flat float array; indices 0..3 = 4 GPU band powers.
uniform float uSceneBandPower[4];   // normalised 0..1

// ── Helpers ──────────────────────────────────────────────────────────────────
float gaussBand(float center, float bw, float probe) {
    float x = (center - probe) / max(bw, 1e-6);
    return exp(-0.5 * x * x);
}

vec3 spectralTint(float probe) {
    // Map probe [0,1] to a perceptual hue (red→violet, acoustic analogy).
    float h  = clamp(probe, 0.0, 1.0);
    float r  = clamp(cos(3.14159 * h) * 1.5,        0.0, 1.0);
    float g  = clamp(sin(3.14159 * h) * 1.2,        0.0, 1.0);
    float b  = clamp(cos(3.14159 * (h - 1.0)) * 1.5, 0.0, 1.0);
    return vec3(r, g, b);
}

// ── Main ──────────────────────────────────────────────────────────────────────
void main() {
    vec3  N    = normalize(gl_FrontFacing ? vNormV : -vNormV);
    vec3  L    = normalize(uLightV);
    vec3  V    = normalize(-vPosV);
    vec3  H    = normalize(L + V);
    float NdL  = max(dot(N, L), 0.0);
    float NdH  = max(dot(N, H), 0.0);
    float NdV  = max(dot(N, V), 0.0);

    // ── Base albedo (with grain) ──────────────────────────────────────────
    vec3  base = uColor.rgb;
    float g    = 0.5 + 0.5 * sin(vPosV.x * 60.0 + vPosV.z * 40.0);
    base *= mix(1.0, 0.85 + 0.30 * g, uGrain);

    // ── Spectral reflectance evaluation ──────────────────────────────────
    float specRefl  = 0.0;
    float specEmit  = 0.0;
    float specReemit= 0.0;
    int nBands = int(uNBands);
    if (nBands > 0) {
        for (int i = 0; i < nBands; i++) {
            float bv = gaussBand(uBandCenters[i], uBandBW[i], uProbeFreq);
            specRefl   += uBandReflect[i] * bv;
            specEmit   += uBandEmit[i]   * bv;
            specReemit += uBandReemit[i] * bv;
        }
        specRefl = clamp(specRefl, 0.0, 1.0);
    }

    // Blend spectral tint into albedo
    vec3 tint  = spectralTint(uProbeFreq);
    base = mix(base, base * tint * (1.0 + specRefl), uSpecTintStrength);

    // ── Phong-PBR lighting ────────────────────────────────────────────────
    // Adjust shininess by roughness and metallic
    float eff_shin  = uShininess * (1.0 - uRoughness * 0.7) * (1.0 + uMetallic * 2.0);
    float diff_comp = NdL;
    float spec_comp = pow(NdH, max(eff_shin, 1.0));

    // Metal tints specular highlight with albedo
    vec3 spec_col = mix(vec3(1.0, 0.92, 0.72), base, uMetallic);

    // Scene-level spectral field contribution (bounce light from tracer)
    float scene_energy = (uSceneBandPower[0] + uSceneBandPower[1]
                        + uSceneBandPower[2] + uSceneBandPower[3]) * 0.25;

    vec3 col = base * (uAmbient + scene_energy * 0.15 + 0.78 * diff_comp)
             + spec_col * uSpecStrength * spec_comp;

    // Rim light
    float rim = pow(1.0 - NdV, 3.0);
    col += base * rim * (0.10 + uMetallic * 0.12);

    // ── Emission + re-emission ────────────────────────────────────────────
    col += uEmission;
    col += base * specEmit   * 0.8;
    col += tint  * specReemit * 0.4;

    // ── Enamel thin-film (Fabry–Pérot iridescence) ────────────────────────
    if (uEnamelThickness > 1e-9) {
        // δ = 4π·n·d·cos(θ)·f/c — uProbeFreq [0,1] scaled to optical proxy
        float probe_hz = uProbeFreq * 4.0e14 + 3.8e14;   // ~380–780 nm
        float delta    = 4.0 * 3.14159265 * uEnamelIOR * uEnamelThickness
                         * NdV * probe_hz / 3.0e8;
        float fringe   = 0.5 + 0.5 * cos(delta);
        float F0_e = pow((uEnamelIOR - 1.0) / (uEnamelIOR + 1.0), 2.0);
        float Fe   = F0_e + (1.0 - F0_e) * pow(1.0 - NdV, 5.0);
        col  = mix(col, col * uEnamelColor, Fe * (1.0 - uRoughness) * 0.45);
        col += uEnamelColor * fringe * Fe * uSpecStrength * 0.18;
    }

    FragColor = vec4(col, uColor.a);
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# GL uniform upload helper
# ─────────────────────────────────────────────────────────────────────────────

def set_spectral_uniforms(prog: int, mat: "Material",
                          probe_freq_norm: float = 0.5,
                          spec_tint_strength: float = 0.0,
                          scene_band_power: Optional[np.ndarray] = None) -> None:
    """Upload spectral band uniforms for ``SPECTRAL_PBR_BODY_FS`` to program ``prog``.

    Must be called while the program is bound (glUseProgram already called).

    Parameters
    ----------
    prog               : int          — GL program handle
    mat                : Material     — material to upload
    probe_freq_norm    : float [0,1]  — normalised probe frequency for spectral tint
    spec_tint_strength : float [0,1]  — blend weight of spectral tint vs albedo
    scene_band_power   : (4,) float32 or None — from SceneFieldIntegration
    """
    try:
        from OpenGL.GL import (glUniform1f, glUniform1fv, glUniform3f,
                                glGetUniformLocation, glUniform4f)
    except ImportError:
        return

    M = Material.MAX_GL_BANDS
    bands = mat.effective_bands(n_synthetic=M)[:M]
    n = len(bands)

    centers = np.zeros(M, np.float32)
    bws     = np.zeros(M, np.float32)
    refls   = np.zeros(M, np.float32)
    emits   = np.zeros(M, np.float32)
    reemits = np.zeros(M, np.float32)
    for i, b in enumerate(bands):
        centers[i] = float(b.center_hz)
        bws[i]     = float(b.resolve_bandwidth())
        refls[i]   = float(b.reflectance)
        emits[i]   = float(b.emission)
        reemits[i] = float(b.reemission)

    def _loc(name: str) -> int:
        return glGetUniformLocation(prog, name.encode())

    def _set1fv(name: str, arr: np.ndarray) -> None:
        loc = _loc(name)
        if loc >= 0:
            glUniform1fv(loc, M, arr)

    _set1fv("uBandCenters", centers)
    _set1fv("uBandBW",      bws)
    _set1fv("uBandReflect", refls)
    _set1fv("uBandEmit",    emits)
    _set1fv("uBandReemit",  reemits)

    loc_nb = _loc("uNBands")
    if loc_nb >= 0:
        glUniform1f(loc_nb, float(n))

    loc_pf = _loc("uProbeFreq")
    if loc_pf >= 0:
        glUniform1f(loc_pf, float(probe_freq_norm))

    loc_ts = _loc("uSpecTintStrength")
    if loc_ts >= 0:
        glUniform1f(loc_ts, float(spec_tint_strength))

    # Scene band power
    sbp = scene_band_power if scene_band_power is not None else np.zeros(4, np.float32)
    sbp = np.asarray(sbp[:4], np.float32)
    loc_sbp = _loc("uSceneBandPower")
    if loc_sbp >= 0:
        glUniform1fv(loc_sbp, 4, sbp)

    # Base PBR extras
    loc_rough = _loc("uRoughness");  loc_metal = _loc("uMetallic")
    if loc_rough >= 0: glUniform1f(loc_rough, float(mat.roughness))
    if loc_metal >= 0: glUniform1f(loc_metal, float(mat.metallic))

    er, eg, eb = mat.emission_rgb[:3]
    loc_emit = _loc("uEmission")
    if loc_emit >= 0:
        glUniform3f(loc_emit, er, eg, eb)

    # Enamel coating uniforms
    en = getattr(mat, "enamel", None)
    loc_et = _loc("uEnamelThickness")
    if en is not None and loc_et >= 0:
        glUniform1f(loc_et, float(en.thickness_m))
        loc = _loc("uEnamelIOR");        
        if loc >= 0: glUniform1f(loc, float(en.ior_real))
        loc = _loc("uEnamelAbsorption")
        if loc >= 0: glUniform1f(loc, float(en.ior_imag))
        cr, cg, cb = en.color_rgb[:3]
        loc = _loc("uEnamelColor")
        if loc >= 0: glUniform3f(loc, float(cr), float(cg), float(cb))
    elif loc_et >= 0:
        glUniform1f(loc_et, 0.0)   # disable enamel for this material


# ─────────────────────────────────────────────────────────────────────────────
# Preset material library
# ─────────────────────────────────────────────────────────────────────────────

def _acoustic_wall_preset(name: str, albedo: list, rough: float, metal: float,
                           absorb_lo: float, absorb_hi: float,
                           n_bands: int = 4, domain_lo=20.0, domain_hi=20000.0) -> Material:
    """Build a Material with frequency-dependent absorption for acoustic walls."""
    centers = np.geomspace(domain_lo, domain_hi, n_bands).tolist()
    absorb  = np.linspace(absorb_lo, absorb_hi, n_bands).tolist()
    bands   = []
    for c, a in zip(centers, absorb):
        r = max(0.0, 1.0 - a) * (0.2 + 0.8 * (1.0 - rough))
        bw = c * 0.5
        bands.append(SpectralBand(
            center_hz=c, bandwidth_hz=bw,
            reflectance=r, transmittance=0.0,
            diffuse_frac=rough,
            emission=0.0, reemission=0.0,
            ior_real=1.5 + metal * 0.5, ior_imag=metal * 2.0,
        ))
    return Material(name=name, domain="acoustic",
                    albedo=albedo, roughness=rough, metallic=metal,
                    spectral_bands=bands)


def _em_material(name: str, albedo: list, rough: float, metal: float,
                 emission_rgb: list = None, ior: float = 1.5) -> Material:
    """Build a Material with EM-optical spectral bands (RGB analogy)."""
    # Three bands: R (~700nm/430THz), G (~550nm/545THz), B (~450nm/666THz)
    bands = []
    for center_hz, refl_mod, emit in [
        (nm_to_hz(700), albedo[0], (emission_rgb or [0,0,0])[0]),
        (nm_to_hz(550), albedo[1], (emission_rgb or [0,0,0])[1]),
        (nm_to_hz(450), albedo[2], (emission_rgb or [0,0,0])[2]),
    ]:
        bw = center_hz * 0.1   # ~10% bandwidth
        bands.append(SpectralBand(
            center_hz=center_hz, bandwidth_hz=bw,
            reflectance=float(refl_mod) * (1.0 - rough * 0.5),
            transmittance=0.0,
            diffuse_frac=rough,
            emission=float(emit),
            reemission=0.0,
            ior_real=ior, ior_imag=metal * 3.0,
        ))
    return Material(name=name, domain="em_optical",
                    albedo=albedo, roughness=rough, metallic=metal,
                    emission_rgb=emission_rgb or [0, 0, 0],
                    ior=ior, spectral_bands=bands)


MATERIAL_PRESETS: dict[str, Material] = {
    # ── Acoustic / room materials ─────────────────────────────────────────────
    "hard_concrete":
        _acoustic_wall_preset("hard_concrete", [0.14, 0.14, 0.16], 0.90, 0.02,
                              absorb_lo=0.02, absorb_hi=0.10),
    "painted_concrete":
        _acoustic_wall_preset("painted_concrete", [0.18, 0.18, 0.20], 0.82, 0.02,
                              absorb_lo=0.03, absorb_hi=0.08),
    "carpet":
        _acoustic_wall_preset("carpet", [0.30, 0.22, 0.18], 0.98, 0.0,
                              absorb_lo=0.05, absorb_hi=0.55),
    "glass_panel":
        _acoustic_wall_preset("glass_panel", [0.75, 0.85, 0.90], 0.05, 0.0,
                              absorb_lo=0.03, absorb_hi=0.06,
                              domain_lo=200.0, domain_hi=20000.0),
    "perforated_steel":
        _acoustic_wall_preset("perforated_steel", [0.50, 0.50, 0.52], 0.35, 0.80,
                              absorb_lo=0.10, absorb_hi=0.45),
    "brushed_steel":
        _acoustic_wall_preset("brushed_steel", [0.60, 0.60, 0.62], 0.20, 0.95,
                              absorb_lo=0.02, absorb_hi=0.05),
    "wood_panel":
        _acoustic_wall_preset("wood_panel", [0.45, 0.32, 0.18], 0.70, 0.04,
                              absorb_lo=0.10, absorb_hi=0.30),
    "acoustic_foam":
        _acoustic_wall_preset("acoustic_foam", [0.18, 0.16, 0.14], 0.98, 0.0,
                              absorb_lo=0.40, absorb_hi=0.95),
    "anechoic_wedge":
        _acoustic_wall_preset("anechoic_wedge", [0.05, 0.05, 0.05], 1.0, 0.0,
                              absorb_lo=0.90, absorb_hi=0.99),
    # ── Station body / LCARS materials ───────────────────────────────────────
    "lcars_body":
        _acoustic_wall_preset("lcars_body", [0.07, 0.09, 0.13], 0.45, 0.45,
                              absorb_lo=0.10, absorb_hi=0.18),
    "lcars_trim":
        Material(name="lcars_trim", domain="acoustic",
                 albedo=[0.02, 0.18, 0.42], roughness=0.15, metallic=0.70,
                 emission_rgb=[0.00, 0.28, 0.78],
                 spectral_bands=[
                     SpectralBand(center_hz=440.0, bandwidth_hz=220.0,
                                  reflectance=0.65, diffuse_frac=0.10,
                                  emission=0.6, reemission=0.05,
                                  ior_real=2.1, ior_imag=1.8),
                 ]),
    "lcars_screen":
        Material(name="lcars_screen", domain="em_optical",
                 albedo=[0.01, 0.01, 0.02], roughness=0.04, metallic=0.0,
                 emission_rgb=[0.05, 0.28, 0.82],
                 spectral_bands=[
                     SpectralBand(center_hz=nm_to_hz(470), bandwidth_hz=nm_to_hz(470)*0.08,
                                  reflectance=0.05, diffuse_frac=0.02,
                                  emission=1.0, reemission=0.0,
                                  ior_real=1.52, ior_imag=0.0),
                 ]),
    # ── Optical / EM materials ────────────────────────────────────────────────
    "polished_chrome":
        _em_material("polished_chrome", [0.92, 0.90, 0.88], 0.02, 1.0, ior=0.14),
    "matte_black":
        _em_material("matte_black", [0.02, 0.02, 0.02], 0.98, 0.0),
    "warm_white_emit":
        _em_material("warm_white_emit", [0.95, 0.90, 0.80], 0.30, 0.0,
                     emission_rgb=[0.95, 0.82, 0.60], ior=1.5),
    "pearl_white_tile": Material(
        name="pearl_white_tile",
        domain="em_optical",
        albedo=[0.96, 0.93, 0.86],
        roughness=0.11,
        metallic=0.0,
        emission_rgb=[0.0, 0.0, 0.0],
        ior=1.58,
        transmission=0.0,
        enamel=EnamelCoating(
            thickness_m=135e-9,
            ior_real=1.54,
            ior_imag=0.0,
            color_rgb=[1.0, 0.985, 0.94],
            roughness=0.025,
            spectral_bands=[
                SpectralBand(center_hz=nm_to_hz(450), bandwidth_hz=nm_to_hz(450) * 0.055,
                             reflectance=0.16, transmittance=0.0, diffuse_frac=0.04,
                             ior_real=1.54, ior_imag=0.0),
                SpectralBand(center_hz=nm_to_hz(560), bandwidth_hz=nm_to_hz(560) * 0.070,
                             reflectance=0.12, transmittance=0.0, diffuse_frac=0.05,
                             ior_real=1.54, ior_imag=0.0),
            ],
        ),
        spectral_identity=SpectralHistogram.from_array(
            np.array([nm_to_hz(nm) for nm in [780, 700, 620, 560, 500, 450, 400, 380]], dtype=np.float64),
            np.array([0.78, 0.84, 0.88, 0.86, 0.90, 0.94, 0.91], dtype=np.float64),
            quantity="reflectance",
        ),
        spectral_bands=[
            SpectralBand(center_hz=nm_to_hz(700), bandwidth_hz=nm_to_hz(700) * 0.080,
                         reflectance=0.82, transmittance=0.0, diffuse_frac=0.10,
                         ior_real=1.58, ior_imag=0.0),
            SpectralBand(center_hz=nm_to_hz(550), bandwidth_hz=nm_to_hz(550) * 0.075,
                         reflectance=0.88, transmittance=0.0, diffuse_frac=0.09,
                         ior_real=1.58, ior_imag=0.0),
            SpectralBand(center_hz=nm_to_hz(450), bandwidth_hz=nm_to_hz(450) * 0.065,
                         reflectance=0.94, transmittance=0.0, diffuse_frac=0.08,
                         ior_real=1.58, ior_imag=0.0),
        ],
    ),
    # ── Duty-station surface materials ───────────────────────────────────────
    "lcars_body_enamel": Material(
        name="lcars_body_enamel", domain="acoustic",
        albedo=[0.07, 0.09, 0.13], roughness=0.40, metallic=0.50,
        emission_rgb=[0.0, 0.0, 0.0],
        enamel=EnamelCoating(thickness_m=80e-9, ior_real=1.52, ior_imag=0.0,
                             color_rgb=[1.0, 1.0, 1.0], roughness=0.06),
        spectral_bands=[
            SpectralBand(center_hz=440.0,  bandwidth_hz=220.0,
                         reflectance=0.62, diffuse_frac=0.40),
            SpectralBand(center_hz=2000.0, bandwidth_hz=1000.0,
                         reflectance=0.55, diffuse_frac=0.45),
        ],
    ),
    "monitor_emissive_blue": Material(
        name="monitor_emissive_blue", domain="em_optical",
        albedo=[0.01, 0.02, 0.05], roughness=0.03, metallic=0.0,
        emission_rgb=[0.05, 0.28, 0.82],
        enamel=EnamelCoating(thickness_m=120e-9, ior_real=1.52, ior_imag=0.0,
                             color_rgb=[0.20, 0.55, 1.0], roughness=0.03),
        radiance=RadianceProfile(luminance=250.0, cct_k=8500.0, cri=72.0,
                                 solid_angle_sr=math.pi * 2.0,
                                 distribution="lambertian"),
        spectral_bands=[
            SpectralBand(center_hz=nm_to_hz(460), bandwidth_hz=nm_to_hz(460)*0.08,
                         reflectance=0.04, diffuse_frac=0.02,
                         emission=1.0, reemission=0.0,
                         ior_real=1.52, ior_imag=0.0),
            SpectralBand(center_hz=nm_to_hz(490), bandwidth_hz=nm_to_hz(490)*0.06,
                         reflectance=0.03, diffuse_frac=0.02,
                         emission=0.30, reemission=0.0,
                         ior_real=1.52, ior_imag=0.0),
        ],
    ),
    # ── Light source materials ────────────────────────────────────────────────
    "warm_panel_light": Material(
        name="warm_panel_light", domain="em_optical",
        albedo=[0.95, 0.88, 0.75], roughness=0.8, metallic=0.0,
        emission_rgb=[0.95, 0.82, 0.60],
        radiance=RadianceProfile(luminance=1200.0, cct_k=3000.0, cri=90.0,
                                 solid_angle_sr=math.pi * 2.0,
                                 distribution="lambertian"),
        spectral_bands=[
            SpectralBand(center_hz=nm_to_hz(620), bandwidth_hz=nm_to_hz(620)*0.15,
                         emission=0.60, reflectance=0.02),
            SpectralBand(center_hz=nm_to_hz(550), bandwidth_hz=nm_to_hz(550)*0.15,
                         emission=0.80, reflectance=0.02),
            SpectralBand(center_hz=nm_to_hz(460), bandwidth_hz=nm_to_hz(460)*0.12,
                         emission=0.45, reflectance=0.02),
        ],
    ),
    "cool_panel_light": Material(
        name="cool_panel_light", domain="em_optical",
        albedo=[0.90, 0.92, 0.98], roughness=0.8, metallic=0.0,
        emission_rgb=[0.82, 0.88, 1.00],
        radiance=RadianceProfile(luminance=1400.0, cct_k=5000.0, cri=85.0,
                                 solid_angle_sr=math.pi * 2.0,
                                 distribution="lambertian"),
        spectral_bands=[
            SpectralBand(center_hz=nm_to_hz(620), bandwidth_hz=nm_to_hz(620)*0.12,
                         emission=0.50, reflectance=0.02),
            SpectralBand(center_hz=nm_to_hz(550), bandwidth_hz=nm_to_hz(550)*0.12,
                         emission=0.85, reflectance=0.02),
            SpectralBand(center_hz=nm_to_hz(450), bandwidth_hz=nm_to_hz(450)*0.12,
                         emission=0.75, reflectance=0.02),
        ],
    ),
}

# Convenience dict of named enamel presets for direct use / YAML reference
ENAMEL_PRESETS: dict = {
    "clear_coat":       EnamelCoating(thickness_m=80e-9,  ior_real=1.52,
                                      color_rgb=[1.0, 1.0, 1.0], roughness=0.06),
    "gloss_white":      EnamelCoating(thickness_m=150e-9, ior_real=1.52,
                                      color_rgb=[0.98, 0.98, 0.96], roughness=0.04),
    "blue_tint":        EnamelCoating(thickness_m=120e-9, ior_real=1.52,
                                      color_rgb=[0.20, 0.55, 1.0],  roughness=0.03),
    "amber_iridescent": EnamelCoating(thickness_m=180e-9, ior_real=1.60,
                                      color_rgb=[1.0, 0.82, 0.30],  roughness=0.05),
    "metallic_sheen":   EnamelCoating(thickness_m=60e-9,  ior_real=1.8,
                                      ior_imag=0.8,
                                      color_rgb=[0.85, 0.85, 0.88], roughness=0.04),
}


# ─────────────────────────────────────────────────────────────────────────────
# ray_tracer_bridge compat helper
# ─────────────────────────────────────────────────────────────────────────────

def materials_to_tracer_mat_props(
        materials: list,        # list[Material], one per triangle
        freq_hz:   np.ndarray,  # (n_bands,) float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a per-triangle Material list to arrays for _CRayTracer.

    Returns
    -------
    refl_re   : (n_tri, n_bands) float64
    refl_im   : (n_tri, n_bands) float64
    diffusion : (n_tri, n_bands) float64
    emission  : (n_tri, n_bands) float64
    reemission: (n_tri, n_bands) float64
    """
    n_tri  = len(materials)
    n_bands = len(freq_hz)
    refl_re   = np.empty((n_tri, n_bands), np.float64)
    refl_im   = np.empty((n_tri, n_bands), np.float64)
    diffusion = np.empty((n_tri, n_bands), np.float64)
    emission  = np.empty((n_tri, n_bands), np.float64)
    reemission= np.empty((n_tri, n_bands), np.float64)

    for i, mat in enumerate(materials):
        rr, ri, df, em, re = mat.to_tracer_bands(freq_hz)
        refl_re[i]   = rr
        refl_im[i]   = ri
        diffusion[i] = df
        emission[i]  = em
        reemission[i]= re

    return refl_re, refl_im, diffusion, emission, reemission
