"""camera_designer/optical_material.py
========================================
Optical material definitions for volume fills between parametric surfaces.

An ``OpticalMaterial`` describes how light behaves *inside* a volume:
  - Refractive index n(λ) via Sellmeier coefficients or fixed n_d
  - Extinction coefficient k(λ) — imaginary part of complex refractive index
  - Volume scatter: single-scatter albedo + Henyey-Greenstein phase function
  - Interface micro-roughness for scattered transmission / BRDF
  - ``is_mirror``  — surfaces bounding this material reflect rather than refract
  - ``is_opaque``  — all incident rays absorbed at entry (flocking, blackening)

All wavelengths are in **micrometres (μm)**.  Distances are in metres.

Material catalog
----------------
air, BK7, SF5, SiO2, water, oil_immersion,
flocking_black, aluminum_mirror, chrome_mirror, blackened_steel

This catalog describes homogeneous bulk media used by camera geometry. It is
not the authoring home for structural-colour microgeometry. Such a surface is
declared by ``spectral_material.Material.maxwell_patch`` and compiled to a
localized scattering artifact; the resulting ordinary bulk fallback may still
reference an ``OpticalMaterial`` here. See ``MAXWELL_PATCH_CONTEXT.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

__all__ = [
    "OpticalMaterial",
    "MATERIAL_CATALOG",
    "sellmeier_n",
]


# ─────────────────────────────────────────────────────────────────────────────
# Sellmeier helper
# ─────────────────────────────────────────────────────────────────────────────

def sellmeier_n(B: Tuple[float,float,float],
                C: Tuple[float,float,float],
                lam_um: float) -> float:
    """Sellmeier refractive index from B/C coefficient 3-tuples.

    n² = 1 + Σ Bᵢ λ² / (λ² - Cᵢ)  where λ is in micrometres.
    """
    l2 = lam_um * lam_um
    B1, B2, B3 = B
    C1, C2, C3 = C
    n2 = 1.0 + B1*l2/(l2-C1) + B2*l2/(l2-C2) + B3*l2/(l2-C3)
    return math.sqrt(max(n2, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# OpticalMaterial dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpticalMaterial:
    """Full material description for an optical volume or surface interface.

    Parameters
    ----------
    name            : human-readable identifier
    n_d             : refractive index at d-line (587.6 nm); used as fallback
                      when Sellmeier coefficients are all zero
    B, C            : Sellmeier coefficients (3-tuples, λ in μm)
    k               : extinction coefficient (imaginary part of ñ = n + ik)
    scatter_albedo  : single-scatter albedo  0 = pure absorber, 1 = pure scatterer
    hg_g            : Henyey-Greenstein asymmetry parameter  -1..+1
                      0 = isotropic,  ~0.7 = forward-scattering glass impurity
    roughness       : interface micro-roughness σ (metres); 0 = optically smooth
    is_mirror       : surface reflects rather than refracts (metallic coating)
    is_opaque       : all rays absorbed on entry; overrides everything else
    tint            : RGBA tuple (0-1) for display colour in cross-section views
    """
    name:           str                    = "air"
    n_d:            float                  = 1.0
    B:              Tuple[float,float,float] = (0., 0., 0.)
    C:              Tuple[float,float,float] = (0., 0., 0.)
    k:              float                  = 0.0
    scatter_albedo: float                  = 0.0
    hg_g:           float                  = 0.0
    roughness:      float                  = 0.0
    is_mirror:      bool                   = False
    is_opaque:      bool                   = False
    tint:           Tuple[float,float,float,float] = (0.5, 0.5, 0.5, 0.0)

    # ── Index ──────────────────────────────────────────────────────────────

    def n_at(self, wavelength_um: float) -> float:
        """Refractive index at *wavelength_um* (micrometres)."""
        if self.B[0] == 0.0 and self.B[1] == 0.0 and self.B[2] == 0.0:
            return self.n_d
        return sellmeier_n(self.B, self.C, wavelength_um)

    # ── Radiometry helpers ─────────────────────────────────────────────────

    def transmittance(self, path_m: float, wavelength_um: float = 0.587) -> float:
        """Beer-Lambert transmittance over *path_m* metres of this medium."""
        if self.k <= 0.0:
            return 1.0
        lam_m = wavelength_um * 1e-6
        alpha = 4.0 * math.pi * self.k / lam_m
        return math.exp(-alpha * path_m)

    def hg_phase(self, cos_theta: float) -> float:
        """Henyey-Greenstein phase function value at cos(θ)."""
        g = self.hg_g
        if abs(g) < 1e-9:
            return 1.0 / (4.0 * math.pi)
        denom = (1.0 + g*g - 2.0*g*cos_theta) ** 1.5
        return (1.0 - g*g) / (4.0 * math.pi * max(denom, 1e-30))

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "n_d":            self.n_d,
            "B":              list(self.B),
            "C":              list(self.C),
            "k":              self.k,
            "scatter_albedo": self.scatter_albedo,
            "hg_g":           self.hg_g,
            "roughness":      self.roughness,
            "is_mirror":      self.is_mirror,
            "is_opaque":      self.is_opaque,
            "tint":           list(self.tint),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OpticalMaterial":
        return cls(
            name           = d.get("name", "air"),
            n_d            = float(d.get("n_d", 1.0)),
            B              = tuple(d.get("B", [0., 0., 0.])),
            C              = tuple(d.get("C", [0., 0., 0.])),
            k              = float(d.get("k", 0.0)),
            scatter_albedo = float(d.get("scatter_albedo", 0.0)),
            hg_g           = float(d.get("hg_g", 0.0)),
            roughness      = float(d.get("roughness", 0.0)),
            is_mirror      = bool(d.get("is_mirror", False)),
            is_opaque      = bool(d.get("is_opaque", False)),
            tint           = tuple(d.get("tint", [0.5, 0.5, 0.5, 0.0])),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Built-in material catalog
# ─────────────────────────────────────────────────────────────────────────────

MATERIAL_CATALOG: dict[str, OpticalMaterial] = {

    "air": OpticalMaterial(
        name="air", n_d=1.0003,
        tint=(0.7, 0.8, 1.0, 0.0)),

    "vacuum": OpticalMaterial(
        name="vacuum", n_d=1.0,
        tint=(0.5, 0.5, 0.8, 0.0)),

    "BK7": OpticalMaterial(
        name="BK7", n_d=1.5168,
        B=(1.03961212, 0.23179234, 1.01046945),
        C=(0.00600069867, 0.0200179144, 103.560653),
        tint=(0.4, 0.75, 1.0, 0.25)),

    "SF5": OpticalMaterial(
        name="SF5", n_d=1.6727,
        B=(1.46141885, 0.247713019, 0.949995832),
        C=(0.0111826126, 0.0508191367, 112.041888),
        tint=(0.5, 0.6, 1.0, 0.30)),

    "SiO2": OpticalMaterial(
        name="SiO2", n_d=1.4585,
        B=(0.6961663, 0.4079426, 0.8974794),
        C=(0.0684043**2, 0.1162414**2, 9.896161**2),
        tint=(0.6, 0.9, 1.0, 0.20)),

    "N-LAK22": OpticalMaterial(
        name="N-LAK22", n_d=1.6510,
        B=(1.14229781, 0.535138441, 1.04088385),
        C=(0.00585778594, 0.0198546147, 100.834017),
        tint=(0.55, 0.7, 1.0, 0.28)),

    "water": OpticalMaterial(
        name="water", n_d=1.333,
        B=(0.75831, 0.08495, 0.),
        C=(0.01007, 8.91377, 0.),
        tint=(0.3, 0.6, 0.9, 0.18)),

    "oil_immersion": OpticalMaterial(
        name="oil_immersion", n_d=1.515,
        B=(1.5, 0., 0.),
        C=(0.010, 0., 0.),
        tint=(0.7, 0.75, 0.5, 0.30)),

    "flocking_black": OpticalMaterial(
        name="flocking_black", n_d=1.6,
        k=0.5,
        scatter_albedo=0.02,
        hg_g=0.0,
        is_opaque=True,
        tint=(0.05, 0.05, 0.05, 0.9)),

    "aluminum_mirror": OpticalMaterial(
        name="aluminum_mirror", n_d=0.96,
        k=6.9,
        is_mirror=True,
        tint=(0.8, 0.85, 0.9, 0.8)),

    "chrome_mirror": OpticalMaterial(
        name="chrome_mirror", n_d=3.2,
        k=3.3,
        is_mirror=True,
        tint=(0.65, 0.7, 0.75, 0.85)),

    "blackened_steel": OpticalMaterial(
        name="blackened_steel", n_d=2.9,
        k=3.0,
        scatter_albedo=0.05,
        is_opaque=True,
        tint=(0.1, 0.1, 0.12, 0.9)),

    "black_anodize": OpticalMaterial(
        name="black_anodize", n_d=1.7,
        k=0.8,
        scatter_albedo=0.04,
        is_opaque=True,
        tint=(0.08, 0.08, 0.1, 0.9)),

    # ── Camera body / tube / sensor bay materials ──────────────────────────

    # "tube"  — black-anodised aluminium lens barrel (alias of black_anodize
    #            with a slightly warmer very-dark charcoal tint for display)
    "tube": OpticalMaterial(
        name="tube", n_d=1.7,
        k=0.8,
        scatter_albedo=0.03,
        is_opaque=True,
        tint=(0.07, 0.06, 0.05, 0.92)),

    # "body"  — matte-black polycarbonate / magnesium-alloy camera body shell.
    #           Near-zero reflectance; slightly lighter tint than pure flocking
    #           so it reads as a distinct region in the cross-section view.
    "body": OpticalMaterial(
        name="body", n_d=1.58,
        k=0.3,
        scatter_albedo=0.06,
        is_opaque=True,
        tint=(0.10, 0.09, 0.08, 0.88)),

    # "back"  — deep velvet-black sensor-bay lining / film backing.
    #           Essentially an ideal absorber; used on the rear inner wall of
    #           the camera box behind the sensor to kill any back-scattered light.
    "back": OpticalMaterial(
        name="back", n_d=1.6,
        k=1.2,
        scatter_albedo=0.01,
        is_opaque=True,
        tint=(0.03, 0.03, 0.04, 0.96)),
}
