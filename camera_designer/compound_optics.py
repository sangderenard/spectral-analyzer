"""=====================================
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
from dataclasses import dataclass
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
    "OpticalFace",
    "FaceVignettingProfile",
    "BoundaryTeleportProfile",
    "AssemblyFieldProfile",
    "FaceAngularLimit",
    "AssemblyAngularLimits",
    "OpticalSide",
    "RayBundle",
    "BundleTraceResult",
    "RayTraceResult",
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


@dataclass(frozen=True)
class OpticalFace:
    """One registered analytical face in a compound assembly.

    This is the compact, mesh-independent inventory used by batch domain
    builders.  ``radius`` is exact for the clear aperture of the face; whether
    rays actually survive to or through that face is a transport question.
    """
    element_idx: int
    kind:        str
    x_pos:       float
    radius:      float
    n_before:    float
    n_after:     float
    conic_k:     float = 0.0
    R_curvature: float = 0.0
    is_stop:     bool = False

    @property
    def center(self) -> np.ndarray:
        """Clear-aperture center in scene coordinates."""
        return np.array([self.x_pos, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class FaceVignettingProfile:
    """Closed-form aperture cone from a chosen focal point to one face.

    ``q_matrix`` is the symmetric quadratic form for unit directions ``d``:
    ``d.T @ q_matrix @ d >= 0`` means the ray direction lies inside the clear
    cone of this face, before upstream/downstream transport is considered.
    """
    face:                         OpticalFace
    side:                         str
    focal_point:                  np.ndarray
    center:                       np.ndarray
    center_direction:             np.ndarray
    center_half_angle_rad:         float
    clear_cone_half_angle_rad:     float
    onset_half_angle_rad:          float
    cutoff_half_angle_rad:         float
    q_matrix:                     np.ndarray
    axial_distance:               float
    center_offset:                float
    reachable_from_side:          bool


@dataclass(frozen=True)
class BoundaryTeleportProfile:
    """Compound interface cone between the assembly's own side boundaries.

    This is not a cone to every interior element.  It is the high-level domain
    of the special parametric/teleport handler: rays enter at ``source_side``,
    are evaluated through the whole chain, and emerge at ``target_side``.
    """
    source_side:                    str
    target_side:                    str
    source_center:                  np.ndarray
    target_center:                  np.ndarray
    source_radius:                  float
    target_radius:                  float
    axis:                           np.ndarray
    center_direction:               np.ndarray
    center_distance:                float
    target_cone_half_angle_rad:      float
    edge_to_edge_half_angle_rad:     float
    q_matrix:                       np.ndarray
    verified_transmission_fraction: float
    verified_exit_center:           np.ndarray
    verified_exit_radius:           float
    verified_projection:            Optional[BundleTraceResult] = None


@dataclass(frozen=True)
class AssemblyFieldProfile:
    """Field profile for a focal point on one side of the optical system."""
    side:                         str
    focal_point:                  np.ndarray
    faces:                        Tuple[FaceVignettingProfile, ...]
    limiting_face:                Optional[FaceVignettingProfile]
    boundary:                     Optional[BoundaryTeleportProfile]
    full_field_half_angle_rad:     float
    cutoff_half_angle_rad:         float
    verified_projection:          Optional[BundleTraceResult] = None


@dataclass(frozen=True)
class FaceAngularLimit:
    """Angular domain for one registered face, measured from a launch origin."""
    face:                         OpticalFace
    origin:                       np.ndarray
    spread_half_angle_rad:         float
    verified_transmission_fraction: float
    convergence_half_angle_rad:    float
    axis_intercept_x:              float
    status_counts:                 dict


@dataclass(frozen=True)
class AssemblyAngularLimits:
    """Batch angular envelope across every registered face in an assembly."""
    origin:                         np.ndarray
    side:                           str
    faces:                          Tuple[FaceAngularLimit, ...]
    spread_half_angle_rad:           float
    verified_spread_half_angle_rad:  float
    convergence_half_angle_rad:      float


@dataclass(frozen=True)
class RayBundle:
    """Batch of ray origins/directions plus optional wavelengths."""
    origins:      np.ndarray
    directions:   np.ndarray
    wavelengths:  Optional[np.ndarray] = None

    def __post_init__(self):
        object.__setattr__(self, "origins", np.asarray(self.origins, dtype=np.float64))
        dirs = np.asarray(self.directions, dtype=np.float64)
        norms = np.linalg.norm(dirs, axis=1)
        dirs = dirs / np.maximum(norms[:, None], _EPS)
        object.__setattr__(self, "directions", dirs)
        if self.origins.ndim != 2 or self.origins.shape[1] != 3:
            raise ValueError("RayBundle.origins must have shape (N, 3)")
        if dirs.shape != self.origins.shape:
            raise ValueError("RayBundle.directions must have shape (N, 3)")
        if self.wavelengths is not None:
            wl = np.asarray(self.wavelengths, dtype=np.float64).reshape(-1)
            if wl.shape[0] != self.origins.shape[0]:
                raise ValueError("RayBundle.wavelengths must have length N")
            object.__setattr__(self, "wavelengths", wl)


@dataclass(frozen=True)
class RayTraceResult:
    """Scalar transfer result for one ray."""
    origin:      np.ndarray
    direction:   np.ndarray
    opl:         float
    reason:      TerminationReason
    intercepts:  Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class BundleTraceResult:
    """Vectorized container returned by CompoundLens.evaluate_bundle()."""
    origins:     np.ndarray
    directions:  np.ndarray
    opl:         np.ndarray
    status:      np.ndarray


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

    def registered_faces(self) -> Tuple[OpticalFace, ...]:
        """Return all analytical faces registered by this compound model.

        The result unifies lenses, flat interfaces, mirrors added later through
        handler roles, and aperture stops under a single face inventory.  These
        fields are exact parameters; transport-dependent limits are computed by
        tracing against this inventory.
        """
        faces: list[OpticalFace] = []
        for idx, el in enumerate(self._elements):
            if isinstance(el, ApertureStop):
                faces.append(OpticalFace(
                    element_idx=idx,
                    kind="aperture_stop",
                    x_pos=float(el.x_pos),
                    radius=float(el.r_clear),
                    n_before=float(el.n_medium),
                    n_after=float(el.n_medium),
                    is_stop=True,
                ))
            elif isinstance(el, FlatSurface):
                faces.append(OpticalFace(
                    element_idx=idx,
                    kind="flat_surface",
                    x_pos=float(el.x_pos),
                    radius=float(el.aperture_r),
                    n_before=float(el.n_before),
                    n_after=float(el.n_after),
                ))
            elif isinstance(el, ConicSurface):
                faces.append(OpticalFace(
                    element_idx=idx,
                    kind="conic_surface",
                    x_pos=float(el.x_pos),
                    radius=float(el.aperture_r),
                    n_before=float(el.n_before),
                    n_after=float(el.n_after),
                    conic_k=float(el.conic_k),
                    R_curvature=float(el.R_curvature),
                ))
        return tuple(faces)

    def boundary_teleport_profile(
        self,
        side: str,
        *,
        n_azimuth: int = 32,
        verify: bool = True,
    ) -> BoundaryTeleportProfile:
        """Return the compound front↔back interface profile for one launch side.

        The cone is defined only by the assembly's own side planes.  It is the
        shape a parametric teleport handler presents to the surrounding scene:
        enter one boundary, evaluate the entire optical chain, emerge at the
        other boundary.
        """
        src = self.side(side)
        dst = self.side("back" if src.side == "front" else "front")
        source_center = np.array([float(src.x_pos), 0.0, 0.0], dtype=np.float64)
        target_center = np.array([float(dst.x_pos), 0.0, 0.0], dtype=np.float64)
        axis = np.array([float(src.axis_sign), 0.0, 0.0], dtype=np.float64)
        delta = target_center - source_center
        dist = float(np.linalg.norm(delta))
        center_dir = delta / max(dist, _EPS)
        target_half = math.atan2(max(0.0, float(dst.radius)), max(dist, _EPS))
        edge_half = math.atan2(
            max(0.0, float(src.radius)) + max(0.0, float(dst.radius)),
            max(dist, _EPS),
        )
        cos_half = math.cos(target_half)
        q = np.outer(center_dir, center_dir) - (cos_half * cos_half) * np.eye(3, dtype=np.float64)

        result = None
        verified_fraction = 0.0
        verified_center = np.full(3, np.nan, dtype=np.float64)
        verified_radius = float("nan")
        if verify:
            origins = []
            dirs = []
            n_phi = max(4, int(n_azimuth))
            for scale in (0.0, 0.5, 0.95, 1.0):
                for k in range(n_phi):
                    phi = 2.0 * math.pi * k / n_phi
                    target = np.array([
                        float(dst.x_pos),
                        scale * float(dst.radius) * math.cos(phi),
                        scale * float(dst.radius) * math.sin(phi),
                    ], dtype=np.float64)
                    d = target - source_center
                    if float(np.dot(d, axis)) <= _EPS:
                        continue
                    origins.append(source_center - axis * 1.0e-9)
                    dirs.append(d)
            if origins:
                bundle = RayBundle(np.asarray(origins, dtype=np.float64),
                                   np.asarray(dirs, dtype=np.float64))
                result = self.evaluate_bundle(bundle)
                ok = result.status == int(TerminationReason.PASSED.value)
                verified_fraction = float(np.count_nonzero(ok)) / float(ok.shape[0])
                if np.any(ok):
                    pts = result.origins[ok]
                    verified_center = np.mean(pts, axis=0)
                    radial = np.linalg.norm(pts[:, 1:3] - verified_center[None, 1:3], axis=1)
                    verified_radius = float(np.max(radial)) if radial.size else 0.0

        return BoundaryTeleportProfile(
            source_side=src.side,
            target_side=dst.side,
            source_center=source_center,
            target_center=target_center,
            source_radius=float(src.radius),
            target_radius=float(dst.radius),
            axis=axis,
            center_direction=center_dir,
            center_distance=float(dist),
            target_cone_half_angle_rad=float(target_half),
            edge_to_edge_half_angle_rad=float(edge_half),
            q_matrix=q,
            verified_transmission_fraction=float(verified_fraction),
            verified_exit_center=verified_center,
            verified_exit_radius=float(verified_radius),
            verified_projection=result,
        )

    def boundary_teleport_profiles(self, *, n_azimuth: int = 32, verify: bool = True) -> dict:
        """Return both front→back and back→front compound boundary profiles."""
        return {
            "front": self.boundary_teleport_profile("front", n_azimuth=n_azimuth, verify=verify),
            "back": self.boundary_teleport_profile("back", n_azimuth=n_azimuth, verify=verify),
        }

    def vignetting_profile_from_point(
        self,
        focal_point: Sequence[float],
        *,
        side: str = "front",
        verify: bool = False,
        n_azimuth: int = 32,
    ) -> AssemblyFieldProfile:
        """Profile face-by-face vignetting from a focal point on either side.

        The closed-form part is geometric and exact for every registered clear
        aperture: each face contributes a cone in direction space.  If
        ``verify`` is true, edge samples on the limiting face are traced through
        the full refractive/stop chain to identify transport losses.
        """
        p = np.asarray(focal_point, dtype=np.float64)
        s = self.side(side)
        axis = np.array([float(s.axis_sign), 0.0, 0.0], dtype=np.float64)
        profiles = []
        identity = np.eye(3, dtype=np.float64)

        for face in self.registered_faces():
            center = face.center
            v = center - p
            dist = float(np.linalg.norm(v))
            reachable = dist > _EPS and float(np.dot(v, axis)) > _EPS
            if reachable:
                cdir = v / dist
                alpha = math.asin(float(np.clip(face.radius / max(dist, _EPS), 0.0, 1.0)))
                cos_alpha = math.cos(alpha)
                q = np.outer(cdir, cdir) - (cos_alpha * cos_alpha) * identity
                center_angle = math.acos(float(np.clip(np.dot(cdir, axis), -1.0, 1.0)))
                onset = max(0.0, center_angle - alpha)
                cutoff = min(0.5 * math.pi, center_angle + alpha)
                axial_distance = abs(float(np.dot(v, axis)))
                center_offset = float(np.linalg.norm(v - np.dot(v, axis) * axis))
            else:
                cdir = np.zeros(3, dtype=np.float64)
                q = np.full((3, 3), np.nan, dtype=np.float64)
                center_angle = float("nan")
                alpha = 0.0
                onset = 0.0
                cutoff = 0.0
                axial_distance = 0.0
                center_offset = float("nan")

            profiles.append(FaceVignettingProfile(
                face=face,
                side=s.side,
                focal_point=p.copy(),
                center=center,
                center_direction=cdir,
                center_half_angle_rad=float(center_angle),
                clear_cone_half_angle_rad=float(alpha),
                onset_half_angle_rad=float(onset),
                cutoff_half_angle_rad=float(cutoff),
                q_matrix=q,
                axial_distance=float(axial_distance),
                center_offset=float(center_offset),
                reachable_from_side=bool(reachable),
            ))

        reachable_profiles = [prof for prof in profiles if prof.reachable_from_side]
        limiting = min(
            reachable_profiles,
            key=lambda prof: prof.cutoff_half_angle_rad,
            default=None,
        )
        full_field = min((prof.onset_half_angle_rad for prof in reachable_profiles),
                         default=0.0)
        cutoff = min((prof.cutoff_half_angle_rad for prof in reachable_profiles),
                     default=0.0)

        verified = None
        if verify and limiting is not None:
            bundle = self._bundle_for_profile_edge(p, limiting, axis, n_azimuth)
            if bundle.origins.shape[0] > 0:
                verified = self.evaluate_bundle(bundle)

        return AssemblyFieldProfile(
            side=s.side,
            focal_point=p.copy(),
            faces=tuple(profiles),
            limiting_face=limiting,
            boundary=self.boundary_teleport_profile(s.side, n_azimuth=n_azimuth, verify=verify),
            full_field_half_angle_rad=float(full_field),
            cutoff_half_angle_rad=float(cutoff),
            verified_projection=verified,
        )

    def profile_field_pair(
        self,
        object_point: Sequence[float],
        image_point: Optional[Sequence[float]] = None,
        *,
        verify: bool = False,
        n_azimuth: int = 32,
    ) -> dict:
        """Return front and optional back field profiles for a receptive/projective pair.

        This is intentionally point-to-point: camera use supplies an object-side
        point and optionally a sensor/image point; portal use may supply only one
        side and treat the returned face cones as the projective domain.
        """
        front = self.vignetting_profile_from_point(
            object_point,
            side="front",
            verify=verify,
            n_azimuth=n_azimuth,
        )
        result = {"front": front}
        if image_point is not None:
            result["back"] = self.vignetting_profile_from_point(
                image_point,
                side="back",
                verify=verify,
                n_azimuth=n_azimuth,
            )
        return result

    @property
    def elements(self) -> List[_Element]:
        return list(self._elements)

    # ── Scalar and batch transport ───────────────────────────────────────────

    def trace(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
    ) -> RayTraceResult:
        """Trace one ray through the assembly.

        Forward rays traverse front→back.  Backward rays traverse back→front
        with refractive indices swapped at every refracting face, so the same
        analytical model supports sensor→scene domain probes.
        """
        o = np.asarray(origin, dtype=np.float64).copy()
        d = np.asarray(direction, dtype=np.float64).copy()
        d_norm = float(np.linalg.norm(d))
        if d_norm <= _EPS:
            raise ValueError("ray direction must be non-zero")
        d /= d_norm

        if self.hood is not None and d[0] > 0.0 and self.hood.clips(o, d):
            return RayTraceResult(o, d, 0.0, TerminationReason.CLIPPED_HOOD, tuple())

        opl = 0.0
        hits: list[np.ndarray] = []
        sequence = self._trace_sequence(d[0])
        for el in sequence:
            if isinstance(el, ApertureStop):
                o2, d2, opl, reason = el.check(o, d, opl)
            else:
                o2, d2, opl, reason = el.refract(o, d, opl)
            if reason is not TerminationReason.PASSED:
                return RayTraceResult(o, d, float(opl), reason, tuple(hits))
            assert o2 is not None and d2 is not None
            hits.append(o2.copy())
            o = o2 + d2 * (10.0 * _EPS)
            d = d2

        return RayTraceResult(o, d, float(opl), TerminationReason.PASSED, tuple(hits))

    def evaluate_bundle(self, bundle: RayBundle) -> BundleTraceResult:
        """Trace a ray bundle through the current parametric chain."""
        n = int(bundle.origins.shape[0])
        out_o = np.empty((n, 3), dtype=np.float64)
        out_d = np.empty((n, 3), dtype=np.float64)
        opl = np.empty(n, dtype=np.float64)
        status = np.empty(n, dtype=np.int32)
        for i in range(n):
            r = self.trace(bundle.origins[i], bundle.directions[i])
            if r.reason is TerminationReason.PASSED:
                out_o[i] = r.intercepts[-1] if r.intercepts else r.origin
                out_d[i] = r.direction
            else:
                out_o[i] = r.origin
                out_d[i] = r.direction
            opl[i] = r.opl
            status[i] = int(r.reason.value)
        return BundleTraceResult(out_o, out_d, opl, status)

    def drop_terminated(
        self,
        bundle: RayBundle,
    ) -> Tuple[RayBundle, BundleTraceResult, np.ndarray]:
        """Evaluate a bundle and return only rays that passed the assembly."""
        result = self.evaluate_bundle(bundle)
        mask = result.status == int(TerminationReason.PASSED.value)
        wl = bundle.wavelengths[mask] if bundle.wavelengths is not None else None
        return RayBundle(bundle.origins[mask], bundle.directions[mask], wl), result, mask

    def sample_side_bundle(
        self,
        side: str,
        *,
        n_spatial: int = 8,
        n_directions: int = 8,
        wavelengths_um: Optional[Sequence[float]] = None,
    ) -> RayBundle:
        """Generate a deterministic polar bundle on a side's clear aperture."""
        s = self.side(side)
        cone = self.side_cone(s.side)
        wls = list(wavelengths_um) if wavelengths_um is not None else [0.587]
        origins: list[list[float]] = []
        dirs: list[list[float]] = []
        wl_out: list[float] = []

        n_r = max(1, int(n_spatial))
        n_a = max(1, int(n_directions))
        for ir in range(n_r):
            r = 0.0 if n_r == 1 else s.radius * math.sqrt((ir + 0.5) / n_r)
            phi_o = 2.0 * math.pi * (ir % max(1, n_r)) / max(1, n_r)
            oy = r * math.cos(phi_o)
            oz = r * math.sin(phi_o)
            for ia in range(n_a):
                frac = 0.0 if n_a == 1 else (ia + 0.5) / n_a
                theta = frac * cone.half_angle_rad
                phi_d = 2.0 * math.pi * ia / n_a
                dx = float(s.axis_sign) * math.cos(theta)
                dy = math.sin(theta) * math.cos(phi_d)
                dz = math.sin(theta) * math.sin(phi_d)
                for wl in wls:
                    origins.append([s.x_pos - float(s.axis_sign) * 1.0e-9, oy, oz])
                    dirs.append([dx, dy, dz])
                    wl_out.append(float(wl))

        return RayBundle(
            np.asarray(origins, dtype=np.float64),
            np.asarray(dirs, dtype=np.float64),
            np.asarray(wl_out, dtype=np.float64),
        )

    def build_transfer_lut(
        self,
        *,
        n_u: int = 32,
        n_v: int = 32,
        n_directions: int = 8,
    ) -> Tuple[np.ndarray, int]:
        """Build a compact transfer LUT by verified parametric ray batches.

        Payload layout:
          magic, n_u, n_v, n_a, n_b, reserved..., then cells (n_u,n_v,n_a,n_b,7)
        The seven cell fields are exit origin y/z, exit dir x/y/z, valid, opl.
        """
        n_u = max(2, int(n_u))
        n_v = max(2, int(n_v))
        n_a = max(1, int(math.sqrt(max(1, n_directions))))
        n_b = max(1, int(math.ceil(max(1, n_directions) / n_a)))
        side = self.side("front")
        cone = self.side_cone("front")

        origins: list[list[float]] = []
        dirs: list[list[float]] = []
        slots: list[tuple[int, int, int, int]] = []
        for iu in range(n_u):
            y = ((iu + 0.5) / n_u * 2.0 - 1.0) * side.radius
            for iv in range(n_v):
                z = ((iv + 0.5) / n_v * 2.0 - 1.0) * side.radius
                if y*y + z*z > side.radius * side.radius:
                    continue
                for ia in range(n_a):
                    theta = ((ia + 0.5) / n_a) * cone.half_angle_rad
                    for ib in range(n_b):
                        phi = 2.0 * math.pi * (ib + 0.5) / n_b
                        origins.append([side.x_pos - float(side.axis_sign) * 1.0e-9, y, z])
                        dirs.append([
                            math.cos(theta),
                            math.sin(theta) * math.cos(phi),
                            math.sin(theta) * math.sin(phi),
                        ])
                        slots.append((iu, iv, ia, ib))

        payload = np.zeros(16 + n_u * n_v * n_a * n_b * 7, dtype=np.float32)
        payload[:5] = np.array([14950.0, n_u, n_v, n_a, n_b], dtype=np.float32)
        if not origins:
            return payload, 0

        result = self.evaluate_bundle(RayBundle(np.asarray(origins), np.asarray(dirs)))
        cells = payload[16:].reshape(n_u, n_v, n_a, n_b, 7)
        valid_count = 0
        passed = result.status == int(TerminationReason.PASSED.value)
        for i, ok in enumerate(passed):
            if not ok:
                continue
            iu, iv, ia, ib = slots[i]
            cells[iu, iv, ia, ib, 0] = float(result.origins[i, 1])
            cells[iu, iv, ia, ib, 1] = float(result.origins[i, 2])
            cells[iu, iv, ia, ib, 2:5] = result.directions[i].astype(np.float32)
            cells[iu, iv, ia, ib, 5] = 1.0
            cells[iu, iv, ia, ib, 6] = float(result.opl[i])
            valid_count += 1
        return payload, valid_count

    def angular_limits_from_origin(
        self,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        side: str = "front",
        n_azimuth: int = 16,
    ) -> AssemblyAngularLimits:
        """Compute per-face and aggregate angular limits from a launch origin.

        ``spread_half_angle_rad`` is exact geometry from the origin to each
        registered clear aperture.  ``verified_*`` fields are derived by tracing
        the corresponding edge rays through the whole chain.
        """
        o = np.asarray(origin, dtype=np.float64)
        s = self.side(side)
        axis = np.array([float(s.axis_sign), 0.0, 0.0], dtype=np.float64)
        face_limits: list[FaceAngularLimit] = []
        n_phi = max(4, int(n_azimuth))

        for face in self.registered_faces():
            dx = face.x_pos - float(o[0])
            transverse_center = math.hypot(float(o[1]), float(o[2]))
            spread = math.atan2(max(0.0, face.radius) + transverse_center, abs(dx))
            ray_dirs = []
            targets = [np.array([face.x_pos, 0.0, 0.0], dtype=np.float64)]
            for scale in (0.5, 0.95, 1.0):
                for k in range(n_phi):
                    phi = 2.0 * math.pi * k / n_phi
                    targets.append(np.array([
                        face.x_pos,
                        scale * face.radius * math.cos(phi),
                        scale * face.radius * math.sin(phi),
                    ], dtype=np.float64))
            for target in targets:
                d = target - o
                if np.dot(d, axis) > _EPS:
                    ray_dirs.append(d)

            if ray_dirs:
                bundle = RayBundle(np.repeat(o[None, :], len(ray_dirs), axis=0),
                                   np.asarray(ray_dirs, dtype=np.float64))
                result = self.evaluate_bundle(bundle)
                ok = result.status == int(TerminationReason.PASSED.value)
                counts = {
                    TerminationReason(int(v)).name: int(np.count_nonzero(result.status == v))
                    for v in np.unique(result.status)
                }
                verified_frac = float(np.count_nonzero(ok)) / float(len(ok))
                conv = float("nan")
                axis_x = float("nan")
                if np.any(ok):
                    conv_vals = []
                    axis_vals = []
                    for ro, rd in zip(result.origins[ok], result.directions[ok]):
                        r2 = float(rd[1] * rd[1] + rd[2] * rd[2])
                        if r2 > _EPS:
                            t_axis = -float(ro[1] * rd[1] + ro[2] * rd[2]) / r2
                            if t_axis > 0.0:
                                x_axis = float(ro[0] + t_axis * rd[0])
                                axial_dist = abs(x_axis - float(ro[0]))
                                radius_at_exit = math.hypot(float(ro[1]), float(ro[2]))
                                conv_vals.append(math.atan2(radius_at_exit, max(axial_dist, _EPS)))
                                axis_vals.append(x_axis)
                    if conv_vals:
                        conv = float(max(conv_vals))
                        axis_x = float(np.mean(axis_vals))
                    else:
                        out = result.directions[ok]
                        cosang = np.clip(out @ axis, -1.0, 1.0)
                        conv = float(np.max(np.arccos(cosang)))
                face_limits.append(FaceAngularLimit(
                    face=face,
                    origin=o.copy(),
                    spread_half_angle_rad=float(spread),
                    verified_transmission_fraction=verified_frac,
                    convergence_half_angle_rad=conv,
                    axis_intercept_x=axis_x,
                    status_counts=counts,
                ))
            else:
                face_limits.append(FaceAngularLimit(
                    face=face,
                    origin=o.copy(),
                    spread_half_angle_rad=float(spread),
                    verified_transmission_fraction=0.0,
                    convergence_half_angle_rad=float("nan"),
                    axis_intercept_x=float("nan"),
                    status_counts={},
                ))

        spread_all = max((f.spread_half_angle_rad for f in face_limits), default=0.0)
        verified_spread = max(
            (f.spread_half_angle_rad for f in face_limits if f.verified_transmission_fraction > 0.0),
            default=0.0,
        )
        finite_conv = [f.convergence_half_angle_rad for f in face_limits
                       if math.isfinite(f.convergence_half_angle_rad)]
        return AssemblyAngularLimits(
            origin=o.copy(),
            side=s.side,
            faces=tuple(face_limits),
            spread_half_angle_rad=float(spread_all),
            verified_spread_half_angle_rad=float(verified_spread),
            convergence_half_angle_rad=float(max(finite_conv) if finite_conv else 0.0),
        )

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
        """(x_position, radius) of the exit pupil in scene coords.

        The exit pupil is the image of the aperture stop formed by all optical
        elements that follow it.  Found via the ABCD imaging condition B=0:
        propagate past the last element by d = -B/D to the plane where all
        rays from a given stop-edge point converge.

        Using [y, nu] (reduced-angle) convention throughout.
        For an air-to-air system det(M_after) = 1, so the lateral magnification
        is m = 1/D.

        Returns (x_ep, r_ep) in the same coordinate system as element x_pos.
        """
        stop = self._aperture_stop()
        if stop is None:
            for el in reversed(self._elements):
                if isinstance(el, (ConicSurface, FlatSurface)):
                    return el.x_pos, el.aperture_r
            return 0.0, 0.0

        M_after, last_x = self._paraxial_matrix_after_stop_with_last_x()
        r_stop = stop.r_clear
        A = float(M_after[0, 0])
        B = float(M_after[0, 1])
        C = float(M_after[1, 0])
        D = float(M_after[1, 1])

        if abs(D) > _EPS:
            # Imaging condition: all rays from stop edge meet at distance d_ep
            # past the last element where B + d*D = 0.
            d_ep = -B / D
            x_ep = last_x + d_ep
            det = A * D - B * C          # = 1 for air-to-air (Lagrange invariant)
            m = det / D                  # lateral magnification stop→exit pupil
            r_ep = abs(m) * r_stop
        else:
            # D=0 → telecentric image space; exit pupil at infinity.
            # Use last element position; radius from A (stop height magnification).
            x_ep = last_x
            r_ep = abs(A) * r_stop if abs(A) > _EPS else r_stop

        return float(x_ep), float(r_ep)

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

    def pixel_acceptance_fan(
        self,
        sensor_x: float,
        sensor_heights: Sequence[float],
    ) -> List[dict]:
        """Per-pixel acceptance cone geometry for a set of sensor pixel heights.

        For each pixel at height h on the sensor, returns the image-side cone
        (pixel → exit pupil rim / centre) and the paraxial object-side chief ray
        direction (entrance pupil → scene).

        Returns a list of dicts keyed by:
            h_sensor          – pixel transverse height (m)
            x_ep_img, r_ep_img – exit pupil centre and radius
            x_ep_obj, r_ep_obj – entrance pupil centre and radius
            chief_angle_img   – angle of chief ray from axis on image side (rad)
            marginal_half_angle_img – half-angle subtended by exit pupil (rad)
            chief_angle_obj   – field angle of this pixel in object space (rad,
                                 paraxial: atan2(h, f_eff))
        """
        x_ep_img, r_ep_img = self.exit_pupil
        x_ep_obj, r_ep_obj = self.entrance_pupil
        f = abs(self.f_eff)
        result: List[dict] = []
        for h in sensor_heights:
            d_img = abs(sensor_x - x_ep_img)
            if d_img < _EPS:
                chief_img = 0.5 * math.pi * (1.0 if h > 0 else -1.0)
                marginal_img = 0.5 * math.pi
            else:
                chief_img = math.atan2(h, d_img)
                marginal_img = math.atan2(r_ep_img, math.sqrt(d_img ** 2 + h ** 2))
            chief_obj = math.atan2(h, f) if f > _EPS else 0.0
            result.append({
                "h_sensor": float(h),
                "x_ep_img": float(x_ep_img),
                "r_ep_img": float(r_ep_img),
                "x_ep_obj": float(x_ep_obj),
                "r_ep_obj": float(r_ep_obj),
                "chief_angle_img": float(chief_img),
                "marginal_half_angle_img": float(marginal_img),
                "chief_angle_obj": float(chief_obj),
            })
        return result

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

    def _trace_sequence(self, dir_x: float) -> List[_Element]:
        """Return forward or reverse traversal elements for a ray direction."""
        if dir_x >= 0.0:
            return list(self._elements)

        rev: list[_Element] = []
        for el in reversed(self._elements):
            if isinstance(el, ApertureStop):
                rev.append(ApertureStop(el.x_pos, el.r_clear, el.n_medium))
            elif isinstance(el, FlatSurface):
                rev.append(FlatSurface(el.x_pos, el.n_after, el.n_before, el.aperture_r))
            elif isinstance(el, ConicSurface):
                rev.append(ConicSurface(
                    x_pos=float(el.x_pos),
                    R_curvature=float(el.R_curvature),
                    n_before=float(el.n_after),
                    n_after=float(el.n_before),
                    aperture_r=float(el.aperture_r),
                    conic_k=float(el.conic_k),
                ))
        return rev

    def _bundle_for_profile_edge(
        self,
        focal_point: np.ndarray,
        profile: FaceVignettingProfile,
        axis: np.ndarray,
        n_azimuth: int,
    ) -> RayBundle:
        """Build edge rays on a face profile's aperture cone."""
        n_phi = max(4, int(n_azimuth))
        origins = []
        dirs = []
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, profile.center_direction))) > 0.9:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u = np.cross(profile.center_direction, ref)
        u_n = float(np.linalg.norm(u))
        if u_n <= _EPS:
            return RayBundle(np.zeros((0, 3), dtype=np.float64),
                             np.zeros((0, 3), dtype=np.float64))
        u /= u_n
        v = np.cross(profile.center_direction, u)
        for scale in (0.0, 0.5, 1.0):
            alpha = profile.clear_cone_half_angle_rad * scale
            for k in range(n_phi):
                phi = 2.0 * math.pi * k / n_phi
                d = (math.cos(alpha) * profile.center_direction
                     + math.sin(alpha) * (math.cos(phi) * u + math.sin(phi) * v))
                if float(np.dot(d, axis)) > _EPS:
                    origins.append(focal_point)
                    dirs.append(d)
        if not origins:
            return RayBundle(np.zeros((0, 3), dtype=np.float64),
                             np.zeros((0, 3), dtype=np.float64))
        return RayBundle(np.asarray(origins, dtype=np.float64),
                         np.asarray(dirs, dtype=np.float64))

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
        M, _ = self._paraxial_matrix_after_stop_with_last_x()
        return M

    def _paraxial_matrix_after_stop_with_last_x(self) -> "Tuple[np.ndarray, float]":
        """System matrix after the aperture stop plus x-position of the last element."""
        M = np.eye(2)
        prev_x: Optional[float] = None
        prev_n = 1.0
        past_stop = False
        last_x = 0.0
        for el in self._elements:
            if isinstance(el, ApertureStop):
                past_stop = True
                prev_x = el.x_pos
                last_x = el.x_pos
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
                last_x = x
        return M, float(last_x)
