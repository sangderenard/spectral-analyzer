"""guitar_part.py
=================
Core data structures for guitar sub-components as modular game-item parts.

Every GuitarPart carries:
  • its geometry (vertices, normals, optional index buffer or line buffer)
  • a RenderMaterial for OpenGL Phong/heatmap rendering
  • a RayMaterial for physically correct ray-tracer scattering
  • a default PartVisibility (OPAQUE | TRANSPARENT | HIDDEN)
  • a sim_tag that names the FDTD sidecar data channel it drives or reads

No GL, no pygame, no physics imports — pure data.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Visibility
# ─────────────────────────────────────────────────────────────────────────────

class PartVisibility(enum.Enum):
    """Default draw mode for a guitar part.

    OPAQUE      — full Phong shading at opaque_alpha from RenderMaterial.
    TRANSPARENT — same shading at alpha_alpha (glass-like, depth-write off).
    HIDDEN      — geometry exists but is not rendered; still in ray scene.
    """
    OPAQUE      = "opaque"
    TRANSPARENT = "transparent"
    HIDDEN      = "hidden"


# ─────────────────────────────────────────────────────────────────────────────
# Material specs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RenderMaterial:
    """OpenGL Phong material properties for a guitar part.

    These map directly to the ``uColor``, ``uInnerColor``, ``uAmbient``,
    ``uSpecStrength``, ``uShininess``, ``uGrain``, and alpha uniforms used by
    the body/neck shader (``_BODY_VS`` / ``_BODY_FS`` in demo_pluck_gl.py).
    """
    color:          tuple[float, float, float]  # diffuse RGB (front face)
    inner_color:    tuple[float, float, float]  # diffuse RGB (back face)
    ambient:        float
    spec_strength:  float
    shininess:      float
    grain:          float
    opaque_alpha:   float = 1.0    # alpha when OPAQUE
    alpha_alpha:    float = 0.42   # alpha when TRANSPARENT

    def rgba(self, visibility: PartVisibility = PartVisibility.OPAQUE) -> tuple[float, float, float, float]:
        if visibility is PartVisibility.HIDDEN:
            a = 0.0
        elif visibility is PartVisibility.TRANSPARENT:
            a = self.alpha_alpha
        else:
            a = self.opaque_alpha
        return (*self.color, a)


@dataclass(frozen=True)
class RayMaterial:
    """Ray-tracer scattering properties for a guitar part.

    These populate the ``mat_in`` / ``mat_out`` vec4 fields of the BVH ``Tri``
    struct in the GPU compute shader (``_GPU_RAY_FIELD_CS``):

        mat_in  = vec4(reflectivity, diffusion, absorption, ior)
        mat_out = vec4(reflectivity, diffusion, absorption, opacity)

    For opaque surfaces ``opacity = 1.0`` and ``ior`` is unused.
    For transparent surfaces (glass, water) set opacity < 1.0 and ior > 1.0.
    """
    reflectivity: float   # fraction of incident energy reflected [0, 1]
    diffusion:    float   # Lambertian weight vs specular [0=specular, 1=diffuse]
    absorption:   float   # energy absorbed per bounce [0, 1]
    ior:          float = 1.5   # index of refraction (used when opacity < 1)
    opacity:      float = 1.0   # 1 = opaque; < 1 = transmissive (glass)

    def mat_in_vec(self) -> tuple[float, float, float, float]:
        return (self.reflectivity, self.diffusion, self.absorption, self.ior)

    def mat_out_vec(self) -> tuple[float, float, float, float]:
        return (self.reflectivity, self.diffusion, self.absorption, self.opacity)


# ─────────────────────────────────────────────────────────────────────────────
# Part
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuitarPart:
    """One named sub-component of a GuitarItem.

    Geometry
    --------
    triangle_mesh (verts, norms, indices)  — indexed triangle mesh for Phong
      rendering and BVH construction.  ``verts`` and ``norms`` are shaped
      (-1, 3) float32; ``indices`` is shaped (-1,) or (-1, 3) int32 or None
      (in which case vertices are already in triangle-list order).

    line_verts — optional shaped (-1, 3) float32 for GL_LINES rendering
      (strings, frets, pickup markers, …).  May be None.

    plate_verts — optional shaped (-1, 2) float32 plus ``plate_indices`` int32
      for the soundboard fan mesh driven by FDTD plate displacement.
      These are 2-D (X, Y in guitar frame); Z is added at render time from
      the displacement field.

    Metadata
    --------
    sim_tag — name of the FDTD output channel this part visualises.
      e.g. "plate", "string_0" … "string_5", "pressure".  Empty string = no sim.

    sim_sidecar — dict of part-specific FDTD parameters (per-string tension,
      gauge, grid indices, etc.).  Populated by GuitarItem after physics build.
    """
    name:           str
    render_mat:     RenderMaterial
    ray_mat:        RayMaterial
    visibility:     PartVisibility = PartVisibility.OPAQUE

    # Geometry — triangle mesh
    verts:          Optional[np.ndarray] = field(default=None, repr=False)  # (-1, 3) float32
    norms:          Optional[np.ndarray] = field(default=None, repr=False)  # (-1, 3) float32
    indices:        Optional[np.ndarray] = field(default=None, repr=False)  # (-1,) int32

    # Geometry — line primitives (strings, frets, wires)
    line_verts:     Optional[np.ndarray] = field(default=None, repr=False)  # (-1, 3) float32

    # Geometry — plate soundboard (updated per-frame by physics)
    plate_verts:    Optional[np.ndarray] = field(default=None, repr=False)  # (-1, 2) float32
    plate_indices:  Optional[np.ndarray] = field(default=None, repr=False)  # (-1,) int32

    # FDTD coupling
    sim_tag:        str = ""
    sim_sidecar:    dict = field(default_factory=dict)

    # ── Convenience ──────────────────────────────────────────────────────────

    def triangle_verts_flat(self) -> np.ndarray:
        """Return triangle-list verts shaped (-1, 3) from indexed or fan mesh."""
        if self.verts is None:
            return np.zeros((0, 3), np.float32)
        if self.indices is None:
            return self.verts
        idx = self.indices.reshape(-1)
        return self.verts[idx]

    def triangle_norms_flat(self) -> np.ndarray:
        if self.norms is None:
            return np.zeros((0, 3), np.float32)
        if self.indices is None:
            return self.norms
        idx = self.indices.reshape(-1)
        return self.norms[idx]

    def interleaved(self) -> np.ndarray:
        """Interleaved [x,y,z,nx,ny,nz] float32, shape (-1, 6), stride 24 bytes."""
        v = self.triangle_verts_flat()
        n = self.triangle_norms_flat()
        if len(v) == 0:
            return np.zeros((0, 6), np.float32)
        return np.ascontiguousarray(np.column_stack([v, n]), np.float32)

    def albedo_rgb(self) -> tuple[float, float, float]:
        return self.render_mat.color


# ─────────────────────────────────────────────────────────────────────────────
# Predefined materials
# ─────────────────────────────────────────────────────────────────────────────

# Render materials ─ match the *_MAT_* constants in demo_pluck_gl.py exactly.
MAT_BODY_BACK = RenderMaterial(
    color=(0.16, 0.035, 0.018), inner_color=(0.64, 0.38, 0.18),
    ambient=0.24, spec_strength=0.42, shininess=96.0, grain=0.25,
    opaque_alpha=0.98, alpha_alpha=0.48,
)
MAT_BODY_SIDES = RenderMaterial(
    color=(0.18, 0.035, 0.018), inner_color=(0.72, 0.44, 0.21),
    ambient=0.22, spec_strength=0.55, shininess=128.0, grain=0.65,
    opaque_alpha=1.0, alpha_alpha=0.36,
)
MAT_SOUNDBOARD = RenderMaterial(
    color=(0.70, 0.54, 0.28), inner_color=(0.82, 0.66, 0.38),
    ambient=0.20, spec_strength=0.32, shininess=80.0, grain=0.45,
    opaque_alpha=1.0, alpha_alpha=0.42,
)
MAT_NECK = RenderMaterial(
    color=(0.20, 0.085, 0.035), inner_color=(0.50, 0.28, 0.12),
    ambient=0.26, spec_strength=0.36, shininess=88.0, grain=0.55,
    opaque_alpha=0.95, alpha_alpha=0.52,
)
MAT_FRETS = RenderMaterial(
    color=(0.85, 0.82, 0.78), inner_color=(0.85, 0.82, 0.78),
    ambient=0.30, spec_strength=0.80, shininess=200.0, grain=0.04,
    opaque_alpha=1.0, alpha_alpha=0.55,
)
MAT_TUNERS = RenderMaterial(
    color=(0.72, 0.70, 0.65), inner_color=(0.72, 0.70, 0.65),
    ambient=0.28, spec_strength=0.75, shininess=160.0, grain=0.06,
    opaque_alpha=1.0, alpha_alpha=0.55,
)
MAT_PICKUP = RenderMaterial(
    color=(0.08, 0.08, 0.08), inner_color=(0.14, 0.12, 0.10),
    ambient=0.18, spec_strength=0.28, shininess=40.0, grain=0.12,
    opaque_alpha=1.0, alpha_alpha=0.60,
)

# Ray materials ─ physically grounded (Sabine absorption data for guitar tonewoods)
RAYMAT_SPRUCE = RayMaterial(   # Sitka spruce top
    reflectivity=0.28, diffusion=0.72, absorption=0.22, ior=1.52, opacity=1.0)
RAYMAT_ROSEWOOD = RayMaterial( # Indian rosewood back/sides
    reflectivity=0.42, diffusion=0.60, absorption=0.12, ior=1.55, opacity=1.0)
RAYMAT_MAHOGANY = RayMaterial( # Mahogany neck
    reflectivity=0.38, diffusion=0.64, absorption=0.15, ior=1.53, opacity=1.0)
RAYMAT_STEEL_PLAIN = RayMaterial(  # plain steel strings
    reflectivity=0.85, diffusion=0.22, absorption=0.04, ior=2.80, opacity=1.0)
RAYMAT_STEEL_WOUND = RayMaterial(  # wound steel strings (more diffuse)
    reflectivity=0.78, diffusion=0.38, absorption=0.06, ior=2.50, opacity=1.0)
RAYMAT_NICKEL = RayMaterial(       # nickel-silver frets / tuner buttons
    reflectivity=0.88, diffusion=0.18, absorption=0.04, ior=2.50, opacity=1.0)
RAYMAT_CERAMIC = RayMaterial(      # pickup bobbin / polymer
    reflectivity=0.18, diffusion=0.70, absorption=0.48, ior=1.58, opacity=1.0)
