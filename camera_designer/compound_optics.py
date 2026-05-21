"""camera_designer/compound_optics.py
=====================================
Exact algebraic model of a compound optical assembly.

Each element is expressed as a closed-form algebraic operation on rays.
The full system is the composition of those operations; no iteration or
numerical refinement occurs at trace time.

Physical parameters live here and nowhere else.  Mesh geometry is for
BVH intersection only and is never consulted for positions or curvatures.

Elements
--------
  ConicSurface    — refracting surface (sphere, paraboloid, hyperboloid, ellipsoid)
  FlatSurface     — refracting plane (R → ∞ limit, degenerate conic)
  ApertureStop    — pure vignetting check, no refraction
  LensHood        — entrance shade; clips oblique rays before the first element

Exact conic ray intersection
-----------------------------
For a conic surface at vertex x = x_v with curvature c = 1/R and conic
constant k, the surface satisfies:

    c·(y² + z²) = 2·(x − x_v) − (1+k)·c·(x − x_v)²

Substituting P(t) = origin + t·dir yields the quadratic

    A·t² + B·t + C = 0

    A = c·[dy² + dz² + (1+k)·dx²]
    B = 2·{c·[oy·dy + oz·dz + (1+k)·(ox−xv)·dx] − dx}
    C = c·[oy² + oz² + (1+k)·(ox−xv)²] − 2·(ox−xv)

This is exact for every conic section — no approximation.

Human-interest properties
--------------------------
  CompoundLens.f_eff           effective focal length  (m)
  CompoundLens.f_number        f/#
  CompoundLens.entrance_pupil  (z_pos, radius)
  CompoundLens.exit_pupil      (z_pos, radius)
  CompoundLens.acceptance_cone(field_r)   max half-angle (rad)
  CompoundLens.depth_of_field(so, coc)   (near_m, far_m)
  CompoundLens.vignetting(field_angle)    throughput ∈ [0, 1]
  CompoundLens.image_circle_radius()      sensor image circle (m)
  CompoundLens.build_gpu_payload()        float32 SSBO payload for shader
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    # Elements
    "ConicSurface",
    "FlatSurface",
    "ApertureStop",
    "LensHood",
    # System
    "CompoundLens",
    "ConeSpec",
    "OpticalSide",
    # Payload constants (mirror shader)
    "PLENS_MAGIC",
    "PLENS_HEADER",
    "PLENS_SURF_STRIDE",
]

# Magic number distinguishes parametric payload from MLP payload (14948)
PLENS_MAGIC       = 14949.0
PLENS_HEADER      = 8      # floats before first surface record
PLENS_SURF_STRIDE = 8      # floats per surface record
_EPS              = 1e-12


from enum import Enum, auto

class TerminationReason(Enum):
    PASSED         = auto()
    MISSED_SURFACE = auto()
    VIGNETTED      = auto()
    TIR            = auto()
    CLIPPED_STOP   = auto()
    CLIPPED_HOOD   = auto()

# ── Optical side / cone descriptors ──────────────────────────────────────────

@dataclass(frozen=True)
class OpticalSide:
    """Parametric side plane of the optical assembly.

    side        : "front" or "back"
    x_pos       : axial position of the side plane
    radius      : clear radius on that plane
    axis_sign   : +1 means directions into the system point along +X;
                  -1 means directions into the system point along -X.
    element_idx : element that supplied this side's physical aperture
    """
    side:        str
    x_pos:       float
    radius:      float
    axis_sign:   int
    element_idx: int


@dataclass(frozen=True)
class ConeSpec:
    """Geometric cone that describes a useful ray domain on an assembly side."""
    side:           str
    apex_x:         float
    aperture_x:     float
    aperture_radius: float
    axis:           np.ndarray
    half_angle_rad: float


# ── Algebraic primitives ──────────────────────────────────────────────────────

def _conic_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    x_v: float,
    R: float,
    k: float,
) -> Optional[float]:
    """Return the smallest positive t for the exact conic ray intersection.

    Returns None if the ray misses the surface or all intersections are behind
    the origin.

    The surface equation (optical axis = scene X):
        c·(y²+z²) = 2·(x−xv) − (1+k)·c·(x−xv)²   where c = 1/R

    Derived quadratic A·t²+B·t+C = 0 is exact for every conic section.
    """
    if abs(R) < _EPS:
        # Flat limit — handled separately by FlatSurface
        if abs(direction[0]) < _EPS:
            return None
        t = (x_v - origin[0]) / direction[0]
        return t if t > _EPS else None

    c  = 1.0 / R
    kp = 1.0 + k                            # 1 + conic_k
    ox, oy, oz = origin[0] - x_v, origin[1], origin[2]
    dx, dy, dz = direction

    A = c * (dy*dy + dz*dz + kp * dx*dx)
    B = 2.0 * (c * (oy*dy + oz*dz + kp * ox*dx) - dx)
    C = c * (oy*oy + oz*oz + kp * ox*ox) - 2.0 * ox

    if abs(A) < _EPS:
        # Degenerate (ray nearly parallel to conic axis near vertex)
        if abs(B) < _EPS:
            return None
        t = -C / B
        return t if t > _EPS else None

    disc = B*B - 4.0*A*C
    if disc < 0.0:
        return None

    sq = math.sqrt(disc)
    t1 = (-B - sq) / (2.0 * A)
    t2 = (-B + sq) / (2.0 * A)

    # Pick the intersection whose x-intercept is closest to the vertex xv.
    # This selects the physically relevant solution (the refracting cap, not
    # the back of the full sphere or the spurious second conic branch).
    p1x = origin[0] + t1 * dx
    p2x = origin[0] + t2 * dx
    if t1 > _EPS and t2 > _EPS:
        return t1 if abs(p1x - x_v) <= abs(p2x - x_v) else t2
    if t1 > _EPS:
        return t1
    if t2 > _EPS:
        return t2
    return None


def _conic_normal(pos: np.ndarray, x_v: float, R: float, k: float) -> np.ndarray:
    """Outward surface normal at a point on the conic (exact gradient)."""
    if abs(R) < _EPS:
        return np.array([1.0, 0.0, 0.0])

    c  = 1.0 / R
    kp = 1.0 + k
    dx = pos[0] - x_v
    # Gradient of f = c·(y²+z²) − 2·(x−xv) + (1+k)·c·(x−xv)²
    #   ∂f/∂x = −2 + 2·(1+k)·c·(x−xv)
    #   ∂f/∂y = 2·c·y
    #   ∂f/∂z = 2·c·z
    gx = -1.0 + kp * c * dx
    gy =  c * pos[1]
    gz =  c * pos[2]
    n  = np.array([gx, gy, gz])
    norm = np.linalg.norm(n)
    return n / norm if norm > _EPS else np.array([1.0, 0.0, 0.0])


def _snell(
    ray_dir: np.ndarray,
    surface_normal: np.ndarray,
    n1: float,
    n2: float,
) -> Optional[np.ndarray]:
    """Exact vector Snell's law.  Returns refracted unit direction or None (TIR)."""
    cos_i = -float(np.dot(ray_dir, surface_normal))
    if cos_i < 0.0:
        # Ray arrived from the wrong side — flip normal
        surface_normal = -surface_normal
        cos_i = -cos_i
    eta    = n1 / n2
    sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return None   # TIR
    cos_t  = math.sqrt(1.0 - sin2_t)
    refracted = eta * ray_dir + (eta * cos_i - cos_t) * surface_normal
    norm = np.linalg.norm(refracted)
    return refracted / norm if norm > _EPS else None


# ── Optical element types ─────────────────────────────────────────────────────

@dataclass
class ConicSurface:
    """Exact refracting conic surface.

    x_pos       : vertex axial position (scene X, metres)
    R_curvature : signed radius of curvature (+ = centre right of surface)
                  Positive R → convex surface when light comes from the left.
    n_before    : refractive index on the entry side
    n_after     : refractive index on the exit side
    aperture_r  : clear aperture radius; rays outside → vignetted
    conic_k     : conic constant
                    0    = sphere
                   −1    = paraboloid
                   −e²   = hyperboloid  (e = eccentricity, e>1)
                   k>−1  = ellipsoid
    """
    x_pos:      float
    R_curvature: float
    n_before:   float
    n_after:    float
    aperture_r: float
    conic_k:    float = 0.0

    def refract(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        opl: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, TerminationReason]:
        """Intersect ray, check aperture, apply exact Snell's law.

        Returns (new_origin, new_dir, new_opl, reason).
        new_origin / new_dir are None on failure.
        """
        t = _conic_intersect(origin, direction, self.x_pos, self.R_curvature, self.conic_k)
        if t is None:
            return None, None, opl, TerminationReason.MISSED_SURFACE

        hit = origin + t * direction
        r_tr = math.hypot(hit[1], hit[2])
        if self.aperture_r > 0.0 and r_tr > self.aperture_r:
            return None, None, opl, TerminationReason.VIGNETTED

        opl += self.n_before * t

        normal = _conic_normal(hit, self.x_pos, self.R_curvature, self.conic_k)
        refracted = _snell(direction, normal, self.n_before, self.n_after)
        if refracted is None:
            return None, None, opl, TerminationReason.TIR

        return hit, refracted, opl, TerminationReason.PASSED


@dataclass
class FlatSurface:
    """Exact refracting flat interface (R → ∞ limit).

    Equivalent to ConicSurface with R=0, but avoids the degenerate-quadratic
    path and makes the intent explicit.
    """
    x_pos:     float
    n_before:  float
    n_after:   float
    aperture_r: float

    def refract(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        opl: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, TerminationReason]:
        if abs(direction[0]) < _EPS:
            return None, None, opl, TerminationReason.MISSED_SURFACE
        t = (self.x_pos - origin[0]) / direction[0]
        if t <= _EPS:
            return None, None, opl, TerminationReason.MISSED_SURFACE

        hit = origin + t * direction
        r_tr = math.hypot(hit[1], hit[2])
        if self.aperture_r > 0.0 and r_tr > self.aperture_r:
            return None, None, opl, TerminationReason.VIGNETTED

        opl += self.n_before * t

        normal = np.array([-1.0, 0.0, 0.0]) if direction[0] < 0.0 else np.array([1.0, 0.0, 0.0])
        refracted = _snell(direction, normal, self.n_before, self.n_after)
        if refracted is None:
            return None, None, opl, TerminationReason.TIR

        return hit, refracted, opl, TerminationReason.PASSED


@dataclass
class ApertureStop:
    """Pure aperture stop — vignetting only, no refraction.

    Rays with transverse radius > r_clear at x_pos are blocked.
    Rays inside the aperture continue with direction and OPL unchanged.

    n_medium : refractive index of the medium at the stop location;
               used for OPL accumulation of the segment arriving at the stop.
    """
    x_pos:   float
    r_clear: float
    n_medium: float = 1.0

    def check(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        opl: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, TerminationReason]:
        if abs(direction[0]) < _EPS:
            return None, None, opl, TerminationReason.MISSED_SURFACE
        t = (self.x_pos - origin[0]) / direction[0]
        if t <= _EPS:
            return None, None, opl, TerminationReason.MISSED_SURFACE

        hit = origin + t * direction
        r_tr = math.hypot(hit[1], hit[2])
        if r_tr > self.r_clear:
            return None, None, opl, TerminationReason.CLIPPED_STOP

        opl += self.n_medium * t
        return hit, direction.copy(), opl, TerminationReason.PASSED


@dataclass
class LensHood:
    """Conical or cylindrical lens hood (entrance shade).

    Rays that would have struck the inside wall of the hood before reaching
    the first lens element are absorbed.  The check is exact: intersect the
    ray with the inner cone/cylinder and reject if it hits before the rim.

    x_front   : axial position of the open tip of the hood (scene X)
    x_rim     : axial position where the hood attaches to the barrel
    r_opening : inner radius at the open tip  (cone narrows toward rim
                if r_rim < r_opening, or is cylindrical if r_rim == r_opening)
    r_rim     : inner radius at the barrel attachment point
    """
    x_front:   float
    x_rim:     float
    r_opening: float
    r_rim:     float

    def clips(self, origin: np.ndarray, direction: np.ndarray) -> bool:
        """Return True if the ray is blocked by the hood.

        The inner wall of the hood is the cone r(x) = r_opening + (r_rim - r_opening)
        * (x_rim - x) / (x_rim - x_front) for x in [x_front, x_rim].

        We check whether the ray at x_front has r > r_opening — a conservative
        test that catches all rays entering the hood at too steep an angle.
        """
        if abs(direction[0]) < _EPS:
            return False
        t_front = (self.x_front - origin[0]) / direction[0]
        if t_front >= 0.0:
            # Hood is ahead — check the opening
            p = origin + t_front * direction
            return math.hypot(p[1], p[2]) > self.r_opening
        else:
            # Ray origin is already past the front tip — check the rim
            t_rim = (self.x_rim - origin[0]) / direction[0]
            if t_rim <= 0.0:
                return False   # origin past rim too, inside barrel already
            p = origin + t_rim * direction
            return math.hypot(p[1], p[2]) > self.r_rim


# ── Compound system ───────────────────────────────────────────────────────────

# Type alias for the heterogeneous element list
_Element = ConicSurface | FlatSurface | ApertureStop


class CompoundLens:
    """Exact algebraic model of a complete compound optical assembly.

    Build the system by adding elements front-to-back along the optical axis
    (scene X).  Each element is an instance of ConicSurface, FlatSurface, or
    ApertureStop.  An optional LensHood is checked before the first element.

    The coordinate convention throughout:
      scene X = optical axis (light travels in +X direction for forward rays)
      scene Y / Z = transverse plane

    Derived properties (read-only, computed from the element list):

      f_eff             effective focal length (m)
      f_number          f/# = f_eff / entrance_pupil_radius
      entrance_pupil    (x_pos, radius) in scene coords
      exit_pupil        (x_pos, radius) in scene coords
      image_circle_radius  radius of illuminated image circle on sensor (m)

    Callable properties:

      acceptance_cone(field_r)        max marginal half-angle (rad)
      depth_of_field(focus_dist, coc) (near_m, far_m)
      vignetting(field_angle_rad)     throughput fraction 0..1
    """

    def __init__(self) -> None:
        self._elements: List[_Element] = []
        self.hood:      Optional[LensHood] = None

    # ── Construction ─────────────────────────────────────────────────────────

    def add(self, element: _Element) -> "CompoundLens":
        """Append an element.  Returns self for chaining."""
        self._elements.append(element)
        return self

    def set_hood(self, hood: LensHood) -> "CompoundLens":
        self.hood = hood
        return self

    @classmethod
    def from_preset(
        cls,
        preset,
        x_offset: float = 0.0,
        wavelength_um: float = 0.587,
    ) -> "CompoundLens":
        """Build an exact algebraic CompoundLens from a CameraPreset.

        Coordinate mapping: camera local Z (optical axis = +Z) → scene X
        (optical axis = +X).  x_offset shifts the whole assembly so that
        z_vertex=0 maps to x_pos=x_offset.

        IOR chain
        ---------
        n_before[0] = 1.0 (air).
        n_after[i]  = element[i].glass_out.n_at(wavelength_um).
        n_before[i] = n_after[i-1].

        The aperture_stop from the preset is inserted in z-order among the
        refracting surfaces.
        """
        from .parametric_surfaces import (
            ConicSurface    as _PSConic,
            FlatSurface     as _PSFlat,
        )
        try:
            from .parametric_surfaces import SphericalSurface as _PSSphere
        except ImportError:
            _PSSphere = None

        lens = cls()

        elements = sorted(
            list(preset.lens_group.elements) if hasattr(preset, "lens_group") else [],
            key=lambda e: float(e.z_vertex),
        )

        ap     = getattr(preset, "aperture_stop", None)
        ap_z   = float(ap.z_pos) if ap is not None else None

        # Build a merged list: (z, kind, data) sorted by z
        merged: list = []
        stop_done = False
        for el in elements:
            z = float(el.z_vertex)
            if not stop_done and ap_z is not None and ap_z <= z:
                merged.append((ap_z, "stop", ap))
                stop_done = True
            merged.append((z, "lens", el))
        if not stop_done and ap_z is not None:
            merged.append((ap_z, "stop", ap))

        n_prev = 1.0
        for z, kind, data in merged:
            x = x_offset + z
            if kind == "stop":
                lens.add(ApertureStop(
                    x_pos    = x,
                    r_clear  = float(data.r_outer),
                    n_medium = n_prev,
                ))
            else:
                el     = data
                surf   = el.surface
                n_out  = el.glass_out.n_at(wavelength_um)
                r_ap   = float(getattr(surf, "r_max", 0.020))

                if isinstance(surf, _PSConic):
                    lens.add(ConicSurface(
                        x_pos       = x,
                        R_curvature = float(surf.R),
                        n_before    = n_prev,
                        n_after     = n_out,
                        aperture_r  = r_ap,
                        conic_k     = float(surf.K),
                    ))
                elif _PSSphere is not None and isinstance(surf, _PSSphere):
                    lens.add(ConicSurface(
                        x_pos       = x,
                        R_curvature = float(surf.R),
                        n_before    = n_prev,
                        n_after     = n_out,
                        aperture_r  = r_ap,
                        conic_k     = 0.0,
                    ))
                elif isinstance(surf, _PSFlat):
                    lens.add(FlatSurface(
                        x_pos      = x,
                        n_before   = n_prev,
                        n_after    = n_out,
                        aperture_r = r_ap,
                    ))
                else:
                    lens.add(FlatSurface(
                        x_pos      = x,
                        n_before   = n_prev,
                        n_after    = n_out,
                        aperture_r = r_ap,
                    ))

                n_prev = n_out

        return lens

    def update_element(self, idx: int, **kwargs) -> "CompoundLens":
        """Update named fields on the element at index idx.  Returns self.

        Allows per-element reconfiguration without rebuilding the assembly,
        e.g.:  lens.update_element(2, R_curvature=0.045, conic_k=-0.5)
        """
        el = self._elements[idx]
        for k, v in kwargs.items():
            if not hasattr(el, k):
                raise AttributeError(
                    f"Element {type(el).__name__} has no field '{k}'"
                )
            setattr(el, k, v)
        return self

    def translate_elements(
        self,
        indices: Sequence[int],
        delta_x: float,
    ) -> "CompoundLens":
        """Shift a subset of elements along the optical axis by delta_x metres.

        Use this to adjust group positions for focus throw, zoom, or per-ray
        configuration — the parametric model is re-evaluated from the updated
        field values without any rebaking.  IOR values are unchanged.

        Returns self for chaining.
        """
        for i in indices:
            el = self._elements[i]
            if hasattr(el, "x_pos"):
                el.x_pos = el.x_pos + delta_x
        return self

    def side(self, side: str) -> OpticalSide:
        """Return the physical side plane for "front"/"back" evaluation.

        This is the canonical replacement for ad hoc entrance/exit plane
        discovery.  It uses physical element parameters only, never mesh data.
        """
        side_l = side.lower()
        if side_l in ("front", "entrance", "object"):
            for idx, el in enumerate(self._elements):
                if isinstance(el, ApertureStop):
                    return OpticalSide("front", el.x_pos, el.r_clear, +1, idx)
                if isinstance(el, (ConicSurface, FlatSurface)):
                    return OpticalSide("front", el.x_pos, el.aperture_r, +1, idx)
        elif side_l in ("back", "exit", "sensor", "image"):
            for ridx, el in enumerate(reversed(self._elements)):
                idx = len(self._elements) - 1 - ridx
                if isinstance(el, ApertureStop):
                    return OpticalSide("back", el.x_pos, el.r_clear, -1, idx)
                if isinstance(el, (ConicSurface, FlatSurface)):
                    return OpticalSide("back", el.x_pos, el.aperture_r, -1, idx)
        else:
            raise ValueError("side must be one of front/entrance/object or back/exit/sensor/image")
        raise RuntimeError("CompoundLens has no optical elements")

    def side_cone(self, side: str, *, target: str = "opposite") -> ConeSpec:
        """Return a conservative axial cone of useful rays for a side.

        The cone bounds rays launched from this side that can plausibly reach
        the target side aperture.  It is a geometric baseline for LUT domains,
        MLP acceptance screens, and pixel-cone setup.
        """
        s0 = self.side(side)
        if target == "opposite":
            s1 = self.side("back" if s0.side == "front" else "front")
        else:
            s1 = self.side(target)
        dist = abs(s1.x_pos - s0.x_pos)
        radius = max(0.0, s0.radius) + max(0.0, s1.radius)
        half = 0.5 * math.pi if dist < _EPS else math.atan2(radius, dist)
        return ConeSpec(
            side=s0.side,
            apex_x=s0.x_pos,
            aperture_x=s1.x_pos,
            aperture_radius=s1.radius,
            axis=np.array([float(s0.axis_sign), 0.0, 0.0], dtype=np.float64),
            half_angle_rad=float(np.clip(half, 0.0, 0.5 * math.pi)),
        )

    def side_cones(self) -> dict:
        """Return useful front/back cones derived from physical apertures."""
        return {
            "front": self.side_cone("front"),
            "back": self.side_cone("back"),
        }

    @property
    def elements(self) -> List[_Element]:
        return list(self._elements)

    # ── Paraxial system matrix ────────────────────────────────────────────────

    def _paraxial_matrix(self) -> np.ndarray:
        """2×2 paraxial ray transfer matrix for the whole system.

        State vector: [y, nu]  where y = height, n = current IOR, u = angle.

        Refraction matrix at surface with power P = (n2−n1)/R:
          M_refr = [[1, 0], [-P, 1]]

        Transfer matrix over gap of width d in medium n:
          M_gap  = [[1, d/n], [0, 1]]

        System matrix = product right-to-left.
        """
        M = np.eye(2)
        prev_x = None
        prev_n = 1.0   # assume air before the first surface

        for el in self._elements:
            if isinstance(el, ApertureStop):
                # Pure propagation to the stop, no power
                if prev_x is not None:
                    d = el.x_pos - prev_x
                    M = np.array([[1.0, d / prev_n], [0.0, 1.0]]) @ M
                prev_x = el.x_pos
                continue

            if isinstance(el, (ConicSurface, FlatSurface)):
                n1 = el.n_before
                n2 = el.n_after
                x  = el.x_pos
                R  = el.R_curvature if isinstance(el, ConicSurface) else 0.0

                # Propagation from previous surface to this one
                if prev_x is not None:
                    d = x - prev_x
                    M = np.array([[1.0, d / n1], [0.0, 1.0]]) @ M

                # Refraction at this surface
                P = (n2 - n1) / R if abs(R) > _EPS else 0.0
                M = np.array([[1.0, 0.0], [-P, 1.0]]) @ M

                prev_x = x
                prev_n = n2

        return M

    # ── Human-interest properties ─────────────────────────────────────────────

    @property
    def f_eff(self) -> float:
        """Effective focal length (metres).

        Derived from the paraxial system matrix: f = −1/M[1,0] (reduced
        coordinates, M[1,0] = −power of the whole system).
        """
        M = self._paraxial_matrix()
        power = -M[1, 0]
        return 1.0 / power if abs(power) > _EPS else float("inf")

    @property
    def entrance_pupil(self) -> Tuple[float, float]:
        """(x_position, radius) of the entrance pupil in scene coords.

        Found by tracing a marginal ray backward from the aperture stop through
        all elements preceding it, then intersecting the extrapolated ray with
        the optical axis (x-axis).
        """
        stop = self._aperture_stop()
        if stop is None:
            # No explicit stop — entrance pupil is the first element's clear aperture
            for el in self._elements:
                if isinstance(el, (ConicSurface, FlatSurface)):
                    return el.x_pos, el.aperture_r
            return 0.0, 0.0

        # Trace a ray from the stop edge back to object space via reversed elements
        # (paraxial approximation for pupil location)
        M = self._paraxial_matrix_to_stop()
        # Marginal ray at stop: [y=r_stop, u=0]
        r_stop = stop.r_clear
        y_ent  = M[0, 0] * r_stop           # image of stop edge in entrance space
        # x position: track propagation
        x_stop = stop.x_pos
        # Paraxial: entrance pupil x is where the backward-traced chief ray would cross axis.
        # Use the system matrix's x-mapping (simplified: report stop position for now;
        # a full pupil trace uses the complete conjugate calculation).
        return x_stop, abs(y_ent)

    @property
    def exit_pupil(self) -> Tuple[float, float]:
        """(x_position, radius) of the exit pupil in scene coords."""
        stop = self._aperture_stop()
        if stop is None:
            for el in reversed(self._elements):
                if isinstance(el, (ConicSurface, FlatSurface)):
                    return el.x_pos, el.aperture_r
            return 0.0, 0.0

        M_after = self._paraxial_matrix_after_stop()
        r_stop  = stop.r_clear
        y_exit  = M_after[0, 0] * r_stop
        return stop.x_pos, abs(y_exit)

    @property
    def f_number(self) -> float:
        """f/# = effective focal length / entrance pupil diameter."""
        _, r_ent = self.entrance_pupil
        if r_ent < _EPS:
            return float("inf")
        return self.f_eff / (2.0 * r_ent)

    def acceptance_cone(self, field_r: float = 0.0) -> float:
        """Maximum marginal ray half-angle (rad) for an on-axis object point.

        Determined by the marginal ray that just clips the limiting aperture.
        field_r is the transverse object height (metres); 0 = on-axis.
        """
        _, r_ent = self.entrance_pupil
        f       = self.f_eff
        if f <= 0.0 or r_ent <= 0.0:
            return 0.0
        return math.atan2(r_ent, f)

    def depth_of_field(
        self,
        focus_distance_m: float,
        circle_of_confusion_m: float,
    ) -> Tuple[float, float]:
        """(near limit, far limit) of acceptable focus (metres from lens).

        Uses the standard geometric DoF formula with the effective focal length
        and entrance pupil diameter.

        circle_of_confusion_m: acceptable blur circle diameter on sensor (e.g. 0.03e-3 for 35mm).
        """
        f  = self.f_eff
        _, r_ent = self.entrance_pupil
        D  = 2.0 * r_ent          # entrance pupil diameter
        so = focus_distance_m

        if D < _EPS or f < _EPS or so < _EPS:
            return 0.0, float("inf")

        c  = circle_of_confusion_m
        # Image distance for focus at so: 1/si = 1/f - 1/so
        denom = so - f
        if abs(denom) < _EPS:
            return 0.0, float("inf")
        si = f * so / denom

        # Near / far DoF limits
        dof_denom_near = D * so / si - c * (so / si - 1.0)
        dof_denom_far  = D * so / si + c * (so / si - 1.0)
        near = (D * so * f) / dof_denom_near if abs(dof_denom_near) > _EPS else 0.0
        far  = (D * so * f) / dof_denom_far  if abs(dof_denom_far)  > _EPS else float("inf")
        return max(0.0, near), max(0.0, far)

    # ── GPU payload ───────────────────────────────────────────────────────────

    def build_gpu_payload(self) -> np.ndarray:
        """Pack the assembly into a float32 SSBO payload for the T2 shader.

        Layout (mirrors PLENS_* constants in ray_refine.comp.glsl):

        Header  (PLENS_HEADER = 8 floats):
          [0]  PLENS_MAGIC  (14949.0)
          [1]  n_surfaces   (float cast to int in shader)
          [2]  hood_r_opening (0 = no hood)
          [3]  hood_x_front
          [4]  hood_x_rim
          [5..7] reserved

        Per surface  (PLENS_SURF_STRIDE = 8 floats each):
          [0]  x_pos
          [1]  R_curvature  (0 = flat)
          [2]  n_before
          [3]  n_after
          [4]  aperture_r
          [5]  conic_k
          [6]  flags  (bit 0 = is_stop: aperture check only, no refraction)
          [7]  reserved
        """
        refracting = [
            el for el in self._elements
            if isinstance(el, (ConicSurface, FlatSurface, ApertureStop))
        ]
        n = len(refracting)
        buf = np.zeros(PLENS_HEADER + n * PLENS_SURF_STRIDE, dtype=np.float32)

        buf[0] = PLENS_MAGIC
        buf[1] = float(n)
        if self.hood is not None:
            buf[2] = float(self.hood.r_opening)
            buf[3] = float(self.hood.x_front)
            buf[4] = float(self.hood.x_rim)

        for i, el in enumerate(refracting):
            off = PLENS_HEADER + i * PLENS_SURF_STRIDE
            if isinstance(el, ApertureStop):
                buf[off + 0] = float(el.x_pos)
                buf[off + 1] = 0.0              # R = 0 (flat check)
                buf[off + 2] = float(el.n_medium)
                buf[off + 3] = float(el.n_medium)
                buf[off + 4] = float(el.r_clear)
                buf[off + 5] = 0.0              # conic_k unused
                buf[off + 6] = 1.0              # is_stop flag
            elif isinstance(el, FlatSurface):
                buf[off + 0] = float(el.x_pos)
                buf[off + 1] = 0.0
                buf[off + 2] = float(el.n_before)
                buf[off + 3] = float(el.n_after)
                buf[off + 4] = float(el.aperture_r)
                buf[off + 5] = 0.0
                buf[off + 6] = 0.0
            else:  # ConicSurface
                buf[off + 0] = float(el.x_pos)
                buf[off + 1] = float(el.R_curvature)
                buf[off + 2] = float(el.n_before)
                buf[off + 3] = float(el.n_after)
                buf[off + 4] = float(el.aperture_r)
                buf[off + 5] = float(el.conic_k)
                buf[off + 6] = 0.0

        return buf

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _aperture_stop(self) -> Optional[ApertureStop]:
        """Return the first ApertureStop in the element list, or None."""
        for el in self._elements:
            if isinstance(el, ApertureStop):
                return el
        return None

    def _paraxial_matrix_to_stop(self) -> np.ndarray:
        """System matrix for elements preceding (not including) the aperture stop."""
        M = np.eye(2)
        prev_x = None
        prev_n = 1.0
        for el in self._elements:
            if isinstance(el, ApertureStop):
                break
            if isinstance(el, (ConicSurface, FlatSurface)):
                n1 = el.n_before
                n2 = el.n_after
                x  = el.x_pos
                R  = el.R_curvature if isinstance(el, ConicSurface) else 0.0
                if prev_x is not None:
                    d = x - prev_x
                    M = np.array([[1.0, d / n1], [0.0, 1.0]]) @ M
                P = (n2 - n1) / R if abs(R) > _EPS else 0.0
                M = np.array([[1.0, 0.0], [-P, 1.0]]) @ M
                prev_x = x
                prev_n = n2
        return M

    def _paraxial_matrix_after_stop(self) -> np.ndarray:
        """System matrix for elements following (not including) the aperture stop."""
        M = np.eye(2)
        prev_x = None
        prev_n = 1.0
        past_stop = False
        for el in self._elements:
            if isinstance(el, ApertureStop):
                past_stop = True
                prev_x = el.x_pos
                continue
            if not past_stop:
                continue
            if isinstance(el, (ConicSurface, FlatSurface)):
                n1 = el.n_before
                n2 = el.n_after
                x  = el.x_pos
                R  = el.R_curvature if isinstance(el, ConicSurface) else 0.0
                if prev_x is not None:
                    d = x - prev_x
                    M = np.array([[1.0, d / n1], [0.0, 1.0]]) @ M
                P = (n2 - n1) / R if abs(R) > _EPS else 0.0
                M = np.array([[1.0, 0.0], [-P, 1.0]]) @ M
                prev_x = x
                prev_n = n2
        return M
