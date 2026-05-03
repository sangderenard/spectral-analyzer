"""camera_designer/shell_components.py
========================================
Physical shell containers for the camera designer.

Shell components are composite objects — they are NOT ParametricSurface
instances themselves.  Each shell encapsulates a set of parametric surfaces
that define its walls, plus an interior ``OpticalVolume``.

Available shells
----------------
BoxShell    rectangular prism (6 flat walls); body / housing / filter holder
TubeShell   cylindrical tube (2 flat end caps + 1 cylinder wall); lens barrel

Usage
-----
    from camera_designer.shell_components import BoxShell, TubeShell
    from camera_designer.optical_material import MATERIAL_CATALOG

    barrel = TubeShell(
        r_inner      = 0.023,
        r_outer      = 0.026,
        z_front      = 0.060,
        z_back       = -0.004,
        wall_material = MATERIAL_CATALOG["black_anodize"],
        fill_material = MATERIAL_CATALOG["air"],
    )
    # barrel.surfaces -> list of ParametricSurface
    # barrel.volumes  -> list of OpticalVolume

    box = BoxShell(
        x_half=0.070, y_half=0.050, z_half=0.040,
        z_center=0.0,
        wall_material=MATERIAL_CATALOG["blackened_steel"],
        fill_material=MATERIAL_CATALOG["air"],
    )
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .parametric_surfaces import (
    ParametricSurface,
    FlatSurface,
    CylinderSurface,
    ApertureStop,
)
from .optical_material import OpticalMaterial, MATERIAL_CATALOG
from .optical_volume import OpticalVolume

__all__ = ["BoxShell", "TubeShell"]


# ─────────────────────────────────────────────────────────────────────────────
# TubeShell — cylindrical lens barrel / filter ring
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TubeShell:
    """Cylindrical shell: outer tube wall + two annular end-cap planes.

    The tube is aligned along the optical axis (+Z).

    Parameters
    ----------
    r_inner       : clear bore radius (inner light path, metres)
    r_outer       : outer barrel radius (metres)
    z_front       : Z position of the front face (higher Z = scene side)
    z_back        : Z position of the back face  (lower Z  = sensor side)
    wall_material : material of the tube wall (usually black anodize)
    fill_material : material inside the clear bore (usually air)
    label         : human-readable name
    """
    r_inner:       float           = 0.023
    r_outer:       float           = 0.026
    z_front:       float           = 0.060
    z_back:        float           = -0.004
    wall_material: OpticalMaterial = field(
        default_factory=lambda: MATERIAL_CATALOG.get("black_anodize", OpticalMaterial()))
    fill_material:  OpticalMaterial = field(
        default_factory=lambda: MATERIAL_CATALOG.get("air", OpticalMaterial()))
    label:          str             = "tube_barrel"

    # ── Surface / volume accessors ─────────────────────────────────────────

    @property
    def surfaces(self) -> List[ParametricSurface]:
        """Three bounding surfaces: front cap, back cap, cylinder wall."""
        front_cap = FlatSurface(
            z_pos=self.z_front,
            r_max=self.r_outer,
            r_min=self.r_inner,
        )
        back_cap = FlatSurface(
            z_pos=self.z_back,
            r_max=self.r_outer,
            r_min=self.r_inner,
        )
        wall = CylinderSurface(
            r=self.r_outer,
            z_min=self.z_back,
            z_max=self.z_front,
        )
        bore_front = ApertureStop(
            z_pos=self.z_front,
            r_inner=0.0,
            r_outer=self.r_inner,
        )
        bore_back = ApertureStop(
            z_pos=self.z_back,
            r_inner=0.0,
            r_outer=self.r_inner,
        )
        return [front_cap, back_cap, wall, bore_front, bore_back]

    @property
    def volumes(self) -> List[OpticalVolume]:
        """Wall volume (annular region) + bore volume (clear interior)."""
        front_cap = FlatSurface(z_pos=self.z_front, r_max=self.r_outer,
                                r_min=self.r_inner)
        back_cap  = FlatSurface(z_pos=self.z_back,  r_max=self.r_outer,
                                r_min=self.r_inner)
        wall      = CylinderSurface(r=self.r_outer,
                                    z_min=self.z_back, z_max=self.z_front)
        bore_inner = CylinderSurface(r=self.r_inner,
                                     z_min=self.z_back, z_max=self.z_front)
        bore_front = FlatSurface(z_pos=self.z_front, r_max=self.r_inner)
        bore_back  = FlatSurface(z_pos=self.z_back,  r_max=self.r_inner)

        wall_vol = OpticalVolume(
            front_surface = front_cap,
            back_surface  = back_cap,
            material      = self.wall_material,
            label         = f"{self.label}_wall",
            z_sort_key    = self.z_front,
        )
        bore_vol = OpticalVolume(
            front_surface = bore_front,
            back_surface  = bore_back,
            material      = self.fill_material,
            label         = f"{self.label}_bore",
            z_sort_key    = self.z_front + 1e-6,
        )
        return [wall_vol, bore_vol]

    # ── Geometry helpers for 2-D display ──────────────────────────────────

    def xz_outline(self, n: int = 2) -> List[tuple]:
        """Return list of (z, r) polyline segments for XZ cross-section display.

        Returns outer wall outline and inner bore outline as separate lists.
        """
        # Outer wall rectangle
        outer = [
            (self.z_front, self.r_outer),
            (self.z_front, self.r_inner),
            (self.z_back,  self.r_inner),
            (self.z_back,  self.r_outer),
            (self.z_front, self.r_outer),
        ]
        # Mirror on negative side
        outer_neg = [(z, -r) for z, r in outer]
        return [outer, outer_neg]

    def to_dict(self) -> dict:
        return {
            "type":          "tube",
            "r_inner":       self.r_inner,
            "r_outer":       self.r_outer,
            "z_front":       self.z_front,
            "z_back":        self.z_back,
            "wall_material": self.wall_material.to_dict(),
            "fill_material": self.fill_material.to_dict(),
            "label":         self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TubeShell":
        wm = OpticalMaterial.from_dict(d.get("wall_material", {}))
        fm = OpticalMaterial.from_dict(d.get("fill_material", {}))
        return cls(
            r_inner=float(d.get("r_inner", 0.023)),
            r_outer=float(d.get("r_outer", 0.026)),
            z_front=float(d.get("z_front", 0.060)),
            z_back=float(d.get("z_back", -0.004)),
            wall_material=wm,
            fill_material=fm,
            label=d.get("label", "tube_barrel"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# BoxShell — rectangular prism housing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoxShell:
    """Rectangular prism shell (camera body, filter holder, enclosure).

    The box is axis-aligned: ±x_half × ±y_half × ±z_half centred at
    (0, 0, z_center) in camera-local coordinates.

    All six walls are represented as FlatSurfaces with annular holes set to
    ``r_min=0, r_max`` large enough to cover the wall.  In practice only the
    four sides (±X, ±Y) and the front/back faces (±Z) matter for ray tracing
    through a camera body cross-section.

    Parameters
    ----------
    x_half, y_half, z_half  : half-dimensions in metres
    z_center                 : Z centre of the box
    wall_material            : material of the six walls
    fill_material            : material inside the box interior
    label                    : human-readable name
    port_r                   : if > 0, a circular port is cut in the front face
                               (for the lens mount opening)
    """
    x_half:        float           = 0.070
    y_half:        float           = 0.051
    z_half:        float           = 0.040
    z_center:      float           = 0.0
    wall_material: OpticalMaterial = field(
        default_factory=lambda: MATERIAL_CATALOG.get("blackened_steel", OpticalMaterial()))
    fill_material:  OpticalMaterial = field(
        default_factory=lambda: MATERIAL_CATALOG.get("air", OpticalMaterial()))
    label:          str             = "camera_body"
    port_r:         float           = 0.0   # front-face lens-mount opening radius

    # ── Surface / volume accessors ─────────────────────────────────────────

    @property
    def z_front(self) -> float:
        return self.z_center + self.z_half

    @property
    def z_back(self) -> float:
        return self.z_center - self.z_half

    @property
    def surfaces(self) -> List[ParametricSurface]:
        """Six bounding faces as FlatSurfaces (front, back) and CylinderSurfaces
        approximating the side walls as oversized flat planes at ±X and ±Y."""
        r_large = math.hypot(self.x_half, self.y_half) * 1.5
        front = FlatSurface(
            z_pos=self.z_front,
            r_max=r_large,
            r_min=self.port_r if self.port_r > 0 else 0.0,
        )
        back = FlatSurface(
            z_pos=self.z_back,
            r_max=r_large,
        )
        return [front, back]

    @property
    def volumes(self) -> List[OpticalVolume]:
        """Interior volume (one OpticalVolume spanning the full box)."""
        r_large = math.hypot(self.x_half, self.y_half) * 1.5
        front = FlatSurface(
            z_pos=self.z_front, r_max=r_large,
            r_min=self.port_r if self.port_r > 0 else 0.0)
        back  = FlatSurface(z_pos=self.z_back, r_max=r_large)
        return [OpticalVolume(
            front_surface = front,
            back_surface  = back,
            material      = self.fill_material,
            label         = f"{self.label}_interior",
            z_sort_key    = self.z_front,
        )]

    # ── Cross-section outline for display ─────────────────────────────────

    def xz_outline(self) -> List[List[tuple]]:
        """Return (z, r) rectangle outlines for XZ cross-section display."""
        zf, zb = self.z_front, self.z_back
        yh = self.y_half   # use Y half as the radial extent in XZ cross-section
        outer = [(zf, yh), (zb, yh), (zb, -yh), (zf, -yh), (zf, yh)]
        return [outer]

    def to_dict(self) -> dict:
        return {
            "type":          "box",
            "x_half":        self.x_half,
            "y_half":        self.y_half,
            "z_half":        self.z_half,
            "z_center":      self.z_center,
            "wall_material": self.wall_material.to_dict(),
            "fill_material": self.fill_material.to_dict(),
            "label":         self.label,
            "port_r":        self.port_r,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoxShell":
        wm = OpticalMaterial.from_dict(d.get("wall_material", {}))
        fm = OpticalMaterial.from_dict(d.get("fill_material", {}))
        return cls(
            x_half=float(d.get("x_half", 0.070)),
            y_half=float(d.get("y_half", 0.051)),
            z_half=float(d.get("z_half", 0.040)),
            z_center=float(d.get("z_center", 0.0)),
            wall_material=wm,
            fill_material=fm,
            label=d.get("label", "camera_body"),
            port_r=float(d.get("port_r", 0.0)),
        )
