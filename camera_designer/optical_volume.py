"""camera_designer/optical_volume.py
======================================
Optical volume — a closed region between two parametric surfaces that is
filled with a specific optical material.

This is the **bucket fill** tool: you pick a front surface, a back surface,
and a material.  The resulting ``OpticalVolume`` tells the ray tracer what
medium a ray is travelling through at each step along its path.

Usage
-----
    from camera_designer.optical_volume import OpticalVolume, fill_volumes
    from camera_designer.optical_material import MATERIAL_CATALOG

    # Manual construction
    vol = OpticalVolume(front_surface=lens_front, back_surface=lens_back,
                        material=MATERIAL_CATALOG["BK7"], label="doublet_front")

    # Bucket fill: pair N surfaces with N-1 materials
    surfaces = [front_lens, back_lens, aperture_plane, sensor]
    materials = [MATERIAL_CATALOG["BK7"], MATERIAL_CATALOG["air"],
                 MATERIAL_CATALOG["air"]]
    volumes = fill_volumes(surfaces, materials)

    # Build volumes from a CameraPreset automatically
    volumes = OpticalVolume.from_preset(preset)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from .parametric_surfaces import ParametricSurface, FlatSurface, surface_from_dict
from .optical_material import OpticalMaterial, MATERIAL_CATALOG

if TYPE_CHECKING:
    from .camera_preset import CameraPreset

__all__ = ["OpticalVolume", "fill_volumes"]


@dataclass
class OpticalVolume:
    """A closed optical region bounded by two parametric surfaces.

    Rays travel from ``front_surface`` toward ``back_surface`` through
    ``material``.  Either surface may be ``None`` to represent a semi-infinite
    half-space (e.g. open air before the first lens element).

    The ray tracer intersects each surface in Z order, applies the
    material's n(λ) and extinction, and hands off to the next volume at
    the shared boundary.

    Parameters
    ----------
    front_surface   : surface rays enter through (may be None)
    back_surface    : surface rays exit through (may be None)
    material        : medium filling this region
    label           : human-readable name
    z_sort_key      : optional explicit ordering key (metres); if None, derived
                      from front_surface position for auto-sort
    """
    front_surface:  Optional[ParametricSurface] = None
    back_surface:   Optional[ParametricSurface] = None
    material:       OpticalMaterial             = field(default_factory=OpticalMaterial)
    label:          str                         = "volume"
    z_sort_key:     Optional[float]             = None

    # ── Geometry helpers ───────────────────────────────────────────────────

    def z_front(self) -> float:
        """Best-guess Z position of the front surface, for ordering."""
        if self.z_sort_key is not None:
            return self.z_sort_key
        s = self.front_surface
        if s is None:
            return -math.inf
        for attr in ("z_pos", "z_vertex", "z_flange"):
            if hasattr(s, attr):
                return float(getattr(s, attr))
        return 0.0

    def z_back(self) -> float:
        """Best-guess Z position of the back surface."""
        s = self.back_surface
        if s is None:
            return math.inf
        for attr in ("z_pos", "z_vertex", "z_flange"):
            if hasattr(s, attr):
                return float(getattr(s, attr))
        return 0.0

    # ── Preset import ──────────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, preset: "CameraPreset") -> List["OpticalVolume"]:
        """Build an ordered volume list from a CameraPreset.

        Volumes are created for each inter-element air gap and each glass
        element.  An open air half-space is prepended before the first
        element.
        """
        from .camera_preset import GlassSpec
        from .parametric_surfaces import ApertureStop

        volumes: List[OpticalVolume] = []

        # Sort elements front-to-back (descending z_vertex; world +Z = sensor)
        elements = sorted(preset.lens_group.elements,
                          key=lambda e: e.z_vertex, reverse=True)

        prev_surface: Optional[ParametricSurface] = None
        prev_mat = MATERIAL_CATALOG.get("air", OpticalMaterial())

        for el in elements:
            # Air gap before this element (if not the first)
            if prev_surface is not None:
                air_vol = cls(
                    front_surface = prev_surface,
                    back_surface  = el.surface,
                    material      = prev_mat,
                    label         = f"air_before_{el.label or 'element'}",
                )
                volumes.append(air_vol)

            # Glass element volume (front = this surface, back = next element)
            glass_name = el.glass_out.name if el.glass_out else "air"
            glass_mat  = MATERIAL_CATALOG.get(
                glass_name,
                _glass_spec_to_material(el.glass_out),
            )
            # For now the glass volume spans until the next surface;
            # the back_surface is set to the next element's surface in the
            # next iteration.  We store current surface as prev.
            # Build glass volume with back_surface=None; fill in on next pass.
            glass_vol = cls(
                front_surface = el.surface,
                back_surface  = None,       # filled below
                material      = glass_mat,
                label         = f"glass_{el.label or 'element'}",
                z_sort_key    = getattr(el.surface, "z_pos",
                                getattr(el.surface, "z_vertex", 0.)),
            )
            volumes.append(glass_vol)
            prev_surface = el.surface
            prev_mat     = glass_mat

        # Wire back-surfaces: glass_vol[i].back = glass_vol[i+1].front
        glass_vols = [v for v in volumes if v.label.startswith("glass_")]
        for i, gv in enumerate(glass_vols[:-1]):
            gv.back_surface = glass_vols[i+1].front_surface

        # Last element: back = sensor
        if glass_vols:
            glass_vols[-1].back_surface = preset.sensor

        # Aperture stop volume — air gap at aperture plane
        ap = preset.aperture_stop
        volumes.append(cls(
            front_surface = ap,
            back_surface  = None,
            material      = MATERIAL_CATALOG.get("air", OpticalMaterial()),
            label         = "aperture_stop_plane",
            z_sort_key    = float(getattr(ap, "z_pos", 0.)),
        ))

        return volumes

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "front_surface": self.front_surface.to_dict() if self.front_surface else None,
            "back_surface":  self.back_surface.to_dict()  if self.back_surface  else None,
            "material":      self.material.to_dict(),
            "label":         self.label,
            "z_sort_key":    self.z_sort_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OpticalVolume":
        mat_d    = d.get("material") or {}
        mat_name = mat_d.get("name", "air")
        mat      = MATERIAL_CATALOG.get(mat_name) or OpticalMaterial.from_dict(mat_d)
        return cls(
            front_surface = surface_from_dict(d["front_surface"]) if d.get("front_surface") else None,
            back_surface  = surface_from_dict(d["back_surface"])  if d.get("back_surface")  else None,
            material      = mat,
            label         = d.get("label", "volume"),
            z_sort_key    = d.get("z_sort_key"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Bucket fill factory
# ─────────────────────────────────────────────────────────────────────────────

def fill_volumes(
    surfaces:  List[ParametricSurface],
    materials: List[OpticalMaterial],
    labels:    Optional[List[str]] = None,
) -> List[OpticalVolume]:
    """Create volumes by filling between consecutive surface pairs.

    The i-th volume spans ``surfaces[i]`` (front) → ``surfaces[i+1]`` (back),
    filled with ``materials[i]``.

    Requires ``len(materials) == len(surfaces) - 1``.
    """
    if len(materials) != len(surfaces) - 1:
        raise ValueError(
            f"Need len(surfaces)-1 = {len(surfaces)-1} materials, "
            f"got {len(materials)}"
        )
    labels = labels or [f"vol_{i}" for i in range(len(materials))]
    return [
        OpticalVolume(
            front_surface = surfaces[i],
            back_surface  = surfaces[i + 1],
            material      = materials[i],
            label         = labels[i] if i < len(labels) else f"vol_{i}",
        )
        for i in range(len(materials))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper — convert GlassSpec → OpticalMaterial
# ─────────────────────────────────────────────────────────────────────────────

def _glass_spec_to_material(gs) -> OpticalMaterial:
    """Convert a camera_preset.GlassSpec to an OpticalMaterial."""
    if gs is None:
        return MATERIAL_CATALOG.get("air", OpticalMaterial())
    return OpticalMaterial(
        name = gs.name,
        n_d  = gs.n_d,
        B    = tuple(gs.B) if hasattr(gs, "B") else (0., 0., 0.),
        C    = tuple(gs.C) if hasattr(gs, "C") else (0., 0., 0.),
        tint = (0.4, 0.75, 1.0, 0.25),   # generic glass tint
    )
