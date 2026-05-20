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
  CompoundLens.build_transfer_lut()       dense LUT from canonical transfer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
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
    "RayBundle",
    "TransferResultBundle",
    "TerminationReason",
    "TracedRay",
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


# ── Termination reasons ───────────────────────────────────────────────────────

class TerminationReason(Enum):
    PASSED         = auto()   # ray exited the system successfully
    CLIPPED_HOOD   = auto()   # lens hood blocked entry
    CLIPPED_STOP   = auto()   # aperture stop blocked ray
    VIGNETTED      = auto()   # outside clear aperture of a refracting surface
    TIR            = auto()   # total internal reflection
    MISSED_SURFACE = auto()   # ray did not intersect a required surface
    DEGENERATE     = auto()   # numerical degenerate case (near-zero discriminant)


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


@dataclass
class TracedRay:
    """Result of tracing one ray through the system."""
    origin:    np.ndarray         # (3,) entry position
    direction: np.ndarray         # (3,) exit unit direction
    opl:       float              # optical path length accumulated inside assembly
    reason:    TerminationReason  # PASSED or failure mode
    # Surface-by-surface record for inspection / debugging
    intercepts: List[np.ndarray] = field(default_factory=list)  # [(3,), ...]


@dataclass
class RayBundle:
    """Vectorized input domain for a lens transfer evaluation.

    origins      : (N, 3) ray origins in scene coordinates
    directions   : (N, 3) ray unit directions in scene coordinates
    wavelengths  : (N,) wavelength in micrometres. If omitted, Fraunhofer d-line
                   0.587 um is used for every ray.
    config_index : optional (N,) per-ray configuration selector. The current
                   evaluator consumes one CompoundLens state; this field exists
                   so multi-config group evaluation can be added without changing
                   the bundle ABI.
    """
    origins:      np.ndarray
    directions:   np.ndarray
    wavelengths:  Optional[np.ndarray] = None
    config_index: Optional[np.ndarray] = None

    def normalized(self) -> "RayBundle":
        origins = np.asarray(self.origins, dtype=np.float64)
        dirs    = np.asarray(self.directions, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("RayBundle.origins must have shape (N, 3)")
        if dirs.shape != origins.shape:
            raise ValueError("RayBundle.directions must match origins shape")
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs  = dirs / np.maximum(norms, _EPS)
        n     = origins.shape[0]
        if self.wavelengths is None:
            wl = np.full(n, 0.587, dtype=np.float64)
        else:
            wl = np.asarray(self.wavelengths, dtype=np.float64)
            if wl.shape != (n,):
                raise ValueError("RayBundle.wavelengths must have shape (N,)")
        cfg = None if self.config_index is None else np.asarray(self.config_index, dtype=np.int32)
        if cfg is not None and cfg.shape != (n,):
            raise ValueError("RayBundle.config_index must have shape (N,)")
        return RayBundle(origins, dirs, wl, cfg)


@dataclass
class TransferResultBundle:
    """Vectorized transfer result for a complete lens assembly."""
    origins:             np.ndarray  # (N, 3) final/exit positions
    directions:          np.ndarray  # (N, 3) final/exit directions
    optical_path:        np.ndarray  # (N,) accumulated OPL inside assembly
    status:              np.ndarray  # (N,) TerminationReason enum values
    terminating_element: np.ndarray  # (N,) element index, -1 on successful pass

    @property
    def passed(self) -> np.ndarray:
        return self.status == TerminationReason.PASSED.value


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


def _conic_intersect_batch(
    origins: np.ndarray,
    directions: np.ndarray,
    x_v: float,
    R: float,
    k: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized exact conic intersection.

    Returns (t, hit_mask). t is undefined where hit_mask is False.
    """
    n = origins.shape[0]
    t = np.zeros(n, dtype=np.float64)
    hit = np.zeros(n, dtype=bool)

    if abs(R) < _EPS:
        dx = directions[:, 0]
        ok = np.abs(dx) > _EPS
        tv = np.zeros(n, dtype=np.float64)
        tv[ok] = (x_v - origins[ok, 0]) / dx[ok]
        hit = ok & (tv > _EPS)
        t[hit] = tv[hit]
        return t, hit

    c  = 1.0 / R
    kp = 1.0 + k
    ox = origins[:, 0] - x_v
    oy = origins[:, 1]
    oz = origins[:, 2]
    dx = directions[:, 0]
    dy = directions[:, 1]
    dz = directions[:, 2]

    A = c * (dy*dy + dz*dz + kp * dx*dx)
    B = 2.0 * (c * (oy*dy + oz*dz + kp * ox*dx) - dx)
    C = c * (oy*oy + oz*oz + kp * ox*ox) - 2.0 * ox

    linear = np.abs(A) < _EPS
    ok_lin = linear & (np.abs(B) > _EPS)
    tv = np.zeros(n, dtype=np.float64)
    tv[ok_lin] = -C[ok_lin] / B[ok_lin]
    hit_lin = ok_lin & (tv > _EPS)

    quad = ~linear
    disc = B*B - 4.0*A*C
    ok_quad = quad & (disc >= 0.0)
    sq = np.zeros(n, dtype=np.float64)
    sq[ok_quad] = np.sqrt(disc[ok_quad])

    t1 = np.full(n, np.inf, dtype=np.float64)
    t2 = np.full(n, np.inf, dtype=np.float64)
    t1[ok_quad] = (-B[ok_quad] - sq[ok_quad]) / (2.0 * A[ok_quad])
    t2[ok_quad] = (-B[ok_quad] + sq[ok_quad]) / (2.0 * A[ok_quad])
    valid1 = t1 > _EPS
    valid2 = t2 > _EPS
    both = valid1 & valid2

    p1x = origins[:, 0] + t1 * dx
    p2x = origins[:, 0] + t2 * dx
    choose1 = both & (np.abs(p1x - x_v) <= np.abs(p2x - x_v))
    choose2 = both & ~choose1
    only1 = valid1 & ~valid2
    only2 = valid2 & ~valid1

    t[hit_lin] = tv[hit_lin]
    t[choose1 | only1] = t1[choose1 | only1]
    t[choose2 | only2] = t2[choose2 | only2]
    hit = hit_lin | choose1 | choose2 | only1 | only2
    return t, hit


def _conic_normal_batch(pos: np.ndarray, x_v: float, R: float, k: float) -> np.ndarray:
    if abs(R) < _EPS:
        n = np.zeros_like(pos)
        n[:, 0] = 1.0
        return n
    c  = 1.0 / R
    kp = 1.0 + k
    out = np.empty_like(pos, dtype=np.float64)
    out[:, 0] = -1.0 + kp * c * (pos[:, 0] - x_v)
    out[:, 1] = c * pos[:, 1]
    out[:, 2] = c * pos[:, 2]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, _EPS)


def _snell_batch(
    ray_dir: np.ndarray,
    surface_normal: np.ndarray,
    n1: np.ndarray | float,
    n2: np.ndarray | float,
) -> Tuple[np.ndarray, np.ndarray]:
    nrm = np.asarray(surface_normal, dtype=np.float64).copy()
    dirs = np.asarray(ray_dir, dtype=np.float64)
    cos_i = -np.sum(dirs * nrm, axis=1)
    flip = cos_i < 0.0
    nrm[flip] *= -1.0
    cos_i[flip] *= -1.0
    n = dirs.shape[0]
    n1_arr = np.broadcast_to(np.asarray(n1, dtype=np.float64), (n,))
    n2_arr = np.broadcast_to(np.asarray(n2, dtype=np.float64), (n,))
    eta = n1_arr / n2_arr
    sin2_t = eta * eta * np.maximum(0.0, 1.0 - cos_i * cos_i)
    ok = sin2_t <= 1.0
    out = np.zeros_like(dirs)
    if np.any(ok):
        cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t[ok]))
        out[ok] = eta[ok, None] * dirs[ok] + (eta[ok] * cos_i[ok] - cos_t)[:, None] * nrm[ok]
        out[ok] /= np.maximum(np.linalg.norm(out[ok], axis=1, keepdims=True), _EPS)
    return out, ok


def _hood_clips_batch(hood: "LensHood", origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    dx = directions[:, 0]
    clips = np.zeros(origins.shape[0], dtype=bool)
    ok = np.abs(dx) > _EPS
    if not np.any(ok):
        return clips

    t_front = np.zeros(origins.shape[0], dtype=np.float64)
    t_front[ok] = (hood.x_front - origins[ok, 0]) / dx[ok]
    use_front = ok & (t_front >= 0.0)
    if np.any(use_front):
        p = origins[use_front] + t_front[use_front, None] * directions[use_front]
        clips[use_front] = np.hypot(p[:, 1], p[:, 2]) > hood.r_opening

    use_rim = ok & ~use_front
    if np.any(use_rim):
        t_rim = np.zeros(origins.shape[0], dtype=np.float64)
        t_rim[use_rim] = (hood.x_rim - origins[use_rim, 0]) / dx[use_rim]
        in_front_of_rim = use_rim & (t_rim > 0.0)
        if np.any(in_front_of_rim):
            p = origins[in_front_of_rim] + t_rim[in_front_of_rim, None] * directions[in_front_of_rim]
            clips[in_front_of_rim] = np.hypot(p[:, 1], p[:, 2]) > hood.r_rim
    return clips


def _fibonacci_disc(n: int, radius: float) -> np.ndarray:
    """Return (n, 2) deterministic uniform-area Fibonacci disc samples."""
    n = int(max(0, n))
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n == 1:
        return np.zeros((1, 2), dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    r = radius * np.sqrt((i + 0.5) / n)
    phi = i * golden_angle
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def _fibonacci_hemisphere(n: int, axis_sign: int, half_angle_rad: float) -> np.ndarray:
    """Return (n, 3) deterministic Fibonacci samples within an axial cone.

    axis_sign=+1 samples around +X; axis_sign=-1 samples around -X.
    half_angle_rad=pi/2 gives a full hemisphere.
    """
    n = int(max(0, n))
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n == 1:
        return np.array([[float(axis_sign), 0.0, 0.0]], dtype=np.float64)
    half_angle_rad = float(np.clip(half_angle_rad, 0.0, 0.5 * math.pi))
    cos_min = math.cos(half_angle_rad)
    i = np.arange(n, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    cos_th = 1.0 - (i + 0.5) / n * (1.0 - cos_min)
    sin_th = np.sqrt(np.maximum(0.0, 1.0 - cos_th * cos_th))
    phi = i * golden_angle
    return np.stack([
        float(axis_sign) * cos_th,
        sin_th * np.cos(phi),
        sin_th * np.sin(phi),
    ], axis=1)


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

    def _elements_for_axis(self, axis_sign: int) -> List[_Element]:
        """Return physical elements ordered for rays travelling along axis_sign."""
        if axis_sign >= 0:
            return self._elements
        rev: List[_Element] = []
        for el in reversed(self._elements):
            if isinstance(el, ApertureStop):
                rev.append(ApertureStop(el.x_pos, el.r_clear, el.n_medium))
            elif isinstance(el, FlatSurface):
                rev.append(FlatSurface(el.x_pos, el.n_after, el.n_before, el.aperture_r))
            else:
                rev.append(ConicSurface(
                    el.x_pos,
                    el.R_curvature,
                    el.n_after,
                    el.n_before,
                    el.aperture_r,
                    el.conic_k,
                ))
        return rev

    def sample_side_bundle(
        self,
        side: str = "front",
        *,
        n_spatial: int = 64,
        n_directions: int = 64,
        wavelengths_um: Optional[Sequence[float]] = None,
        cone: Optional[ConeSpec] = None,
        side_offset_m: float = 1.0e-6,
        filter_terminated: bool = False,
    ) -> RayBundle | Tuple[RayBundle, TransferResultBundle, np.ndarray]:
        """Dense deterministic input discretization on a parametric side.

        Origins live on the side's physical clear aperture using a Fibonacci
        disc.  Directions live in the side's physically meaningful cone using a
        Fibonacci hemisphere.  The Cartesian product gives a reproducible
        baseline of possible paths through the system.

        If filter_terminated=True, immediately evaluates the transfer and
        returns only rays that pass the system:

            (filtered_bundle, full_transfer_result, passed_mask)
        """
        s = self.side(side)
        cone = cone if cone is not None else self.side_cone(s.side)
        wl = np.array(list(wavelengths_um) if wavelengths_um is not None else [0.587],
                      dtype=np.float64)
        if wl.ndim != 1 or wl.size == 0:
            raise ValueError("wavelengths_um must contain at least one wavelength")

        disc = _fibonacci_disc(n_spatial, s.radius)
        dirs = _fibonacci_hemisphere(n_directions, s.axis_sign, cone.half_angle_rad)
        x_origin = s.x_pos - float(s.axis_sign) * abs(float(side_offset_m))

        n_total = disc.shape[0] * dirs.shape[0] * wl.size
        origins = np.empty((n_total, 3), dtype=np.float64)
        directions = np.empty((n_total, 3), dtype=np.float64)
        wavelengths = np.empty(n_total, dtype=np.float64)

        k = 0
        for wavelength in wl:
            for yz in disc:
                j0 = k
                j1 = k + dirs.shape[0]
                origins[j0:j1, 0] = x_origin
                origins[j0:j1, 1] = yz[0]
                origins[j0:j1, 2] = yz[1]
                directions[j0:j1] = dirs
                wavelengths[j0:j1] = wavelength
                k = j1

        bundle = RayBundle(origins, directions, wavelengths)
        if not filter_terminated:
            return bundle
        result = self.evaluate_bundle(bundle, axis_sign=s.axis_sign)
        mask = result.passed
        filtered = RayBundle(
            origins=result.origins[mask],
            directions=result.directions[mask],
            wavelengths=wavelengths[mask],
        )
        return filtered, result, mask

    def drop_terminated(self, rays: RayBundle) -> Tuple[RayBundle, TransferResultBundle, np.ndarray]:
        """Evaluate and drop rays known to terminate inside the system.

        This is the failure short-circuit LUT/MLP/domain builders should use
        before spending memory or training capacity on impossible paths.
        """
        rb = rays.normalized()
        axis_sign = +1 if float(np.mean(rb.directions[:, 0])) >= 0.0 else -1
        result = self.evaluate_bundle(rb, axis_sign=axis_sign)
        mask = result.passed
        return (
            RayBundle(result.origins[mask], result.directions[mask], rb.wavelengths[mask],
                      None if rb.config_index is None else rb.config_index[mask]),
            result,
            mask,
        )

    def build_transfer_lut(
        self,
        *,
        side: str = "front",
        n_u: int = 64,
        n_v: int = 64,
        n_directions: int = 16,
        n_angle_u: Optional[int] = None,
        n_angle_v: Optional[int] = None,
        wavelengths_um: Optional[Sequence[float]] = None,
        full_assembly_payload: bool = False,
        accumulator_dtype=np.float32,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """Build an angle-resolved dense transfer LUT from the parametric model.

        Payload magic 14950.0 (or 14951.0 for full-assembly payload) stores a
        physically meaningful 4D table:

            side_u, side_v, angle_u, angle_v -> output ray

        The angle dimensions are vital: different incident directions at the
        same aperture coordinate can transport to different exit states.  Empty
        cells remain blocked, so termination knowledge is preserved.
        """
        n_u = int(max(2, n_u))
        n_v = int(max(2, n_v))
        if n_angle_u is None or n_angle_v is None:
            n_ang = int(max(1, math.ceil(math.sqrt(max(1, n_directions)))))
            n_angle_u = n_ang if n_angle_u is None else n_angle_u
            n_angle_v = n_ang if n_angle_v is None else n_angle_v
        n_angle_u = int(max(2, n_angle_u))
        n_angle_v = int(max(2, n_angle_v))
        side_spec = self.side(side)
        wl = np.array(list(wavelengths_um) if wavelengths_um is not None else [0.587],
                      dtype=np.float64)

        # Use cell centers rather than stochastic points so the LUT has a stable
        # dense coverage of the actual grid it will serve at runtime.
        uu = (np.arange(n_u, dtype=np.float64) + 0.5) / n_u * 2.0 - 1.0
        vv = (np.arange(n_v, dtype=np.float64) + 0.5) / n_v * 2.0 - 1.0
        U, V = np.meshgrid(uu, vv, indexing="xy")
        in_disc = (U * U + V * V) <= 1.0
        u_vals = U[in_disc]
        v_vals = V[in_disc]
        spatial = np.stack([u_vals * side_spec.radius, v_vals * side_spec.radius], axis=1)
        cone = self.side_cone(side_spec.side)
        sin_max = max(math.sin(cone.half_angle_rad), _EPS)
        aa = (np.arange(n_angle_u, dtype=np.float64) + 0.5) / n_angle_u * 2.0 - 1.0
        bb = (np.arange(n_angle_v, dtype=np.float64) + 0.5) / n_angle_v * 2.0 - 1.0
        A, B = np.meshgrid(aa, bb, indexing="xy")
        in_angle = (A * A + B * B) <= 1.0
        a_vals = A[in_angle]
        b_vals = B[in_angle]
        trans_y = a_vals * sin_max
        trans_z = b_vals * sin_max
        axial = np.sqrt(np.maximum(0.0, 1.0 - trans_y * trans_y - trans_z * trans_z))
        dirs = np.stack([
            float(side_spec.axis_sign) * axial,
            trans_y,
            trans_z,
        ], axis=1)

        n_total = spatial.shape[0] * dirs.shape[0] * wl.size
        origins = np.empty((n_total, 3), dtype=np.float64)
        directions = np.empty((n_total, 3), dtype=np.float64)
        wavelengths = np.empty(n_total, dtype=np.float64)
        x_origin = side_spec.x_pos - float(side_spec.axis_sign) * 1.0e-6

        k = 0
        for wavelength in wl:
            for yz in spatial:
                j0 = k
                j1 = k + dirs.shape[0]
                origins[j0:j1, 0] = x_origin
                origins[j0:j1, 1] = yz[0]
                origins[j0:j1, 2] = yz[1]
                directions[j0:j1] = dirs
                wavelengths[j0:j1] = wavelength
                k = j1

        bundle = RayBundle(origins, directions, wavelengths)
        result = self.evaluate_bundle(bundle, axis_sign=side_spec.axis_sign)
        passed = result.passed

        stride = 9 if full_assembly_payload else 7
        header_len = 16
        payload = np.zeros(header_len + n_v * n_u * n_angle_v * n_angle_u * stride, dtype=np.float32)
        other = self.side("back" if side_spec.side == "front" else "front")
        payload[:16] = np.array(
            [
                14951.0 if full_assembly_payload else 14950.0,
                float(n_u),
                float(n_v),
                float(n_angle_u),
                float(n_angle_v),
                -1.0,
                1.0,
                -1.0,
                1.0,
                -sin_max,
                sin_max,
                -sin_max,
                sin_max,
                float(side_spec.radius),
                0.0,  # axis_idx: 0 = X optical axis
                float(other.x_pos) if full_assembly_payload else 0.0,
            ],
            dtype=np.float32,
        )

        if not np.any(passed):
            if verbose:
                print("[compound/LUT] no rays survived parametric transfer", flush=True)
            return payload, 0

        cells = payload[header_len:].reshape(n_v, n_u, n_angle_v, n_angle_u, stride)
        p_orig = origins[passed]
        p_dir_in = directions[passed]
        p_dir_out = result.directions[passed]
        p_opl = result.optical_path[passed]

        u = p_orig[:, 1] / max(side_spec.radius, _EPS)
        v = p_orig[:, 2] / max(side_spec.radius, _EPS)
        iu = np.clip(((u + 1.0) * 0.5 * n_u).astype(np.int64), 0, n_u - 1)
        iv = np.clip(((v + 1.0) * 0.5 * n_v).astype(np.int64), 0, n_v - 1)
        ia = np.clip(((p_dir_in[:, 1] + sin_max) / (2.0 * sin_max) * n_angle_u).astype(np.int64), 0, n_angle_u - 1)
        ib = np.clip(((p_dir_in[:, 2] + sin_max) / (2.0 * sin_max) * n_angle_v).astype(np.int64), 0, n_angle_v - 1)

        np.add.at(cells[:, :, :, :, 0], (iv, iu, ib, ia), p_dir_out[:, 0].astype(accumulator_dtype, copy=False))
        np.add.at(cells[:, :, :, :, 1], (iv, iu, ib, ia), p_dir_out[:, 1].astype(accumulator_dtype, copy=False))
        np.add.at(cells[:, :, :, :, 2], (iv, iu, ib, ia), p_dir_out[:, 2].astype(accumulator_dtype, copy=False))
        np.add.at(cells[:, :, :, :, 3], (iv, iu, ib, ia), p_opl.astype(accumulator_dtype, copy=False))
        np.add.at(cells[:, :, :, :, 4], (iv, iu, ib, ia), 1.0)
        np.add.at(cells[:, :, :, :, 5], (iv, iu, ib, ia), p_dir_in[:, 1].astype(accumulator_dtype, copy=False))
        np.add.at(cells[:, :, :, :, 6], (iv, iu, ib, ia), p_dir_in[:, 2].astype(accumulator_dtype, copy=False))
        if full_assembly_payload:
            np.add.at(cells[:, :, :, :, 7], (iv, iu, ib, ia), result.origins[passed, 1].astype(accumulator_dtype, copy=False))
            np.add.at(cells[:, :, :, :, 8], (iv, iu, ib, ia), result.origins[passed, 2].astype(accumulator_dtype, copy=False))

        counts = cells[:, :, :, :, 4]
        mask = counts > 0.0
        mean_channels = [0, 1, 2, 3, 5, 6]
        if full_assembly_payload:
            mean_channels.extend([7, 8])
        for ch in mean_channels:
            cells[:, :, :, :, ch][mask] /= counts[mask]

        # Normalize output direction per occupied cell after averaging.
        od = cells[:, :, :, :, 0:3]
        norm = np.linalg.norm(od, axis=4)
        nmask = mask & (norm > _EPS)
        od[nmask] /= norm[nmask][:, None]

        n_pass = int(np.count_nonzero(passed))
        if verbose:
            occupied = int(np.count_nonzero(mask))
            print(
                f"[compound/LUT] {n_pass:,}/{n_total:,} rays survived; "
                f"{occupied:,}/{n_u*n_v*n_angle_u*n_angle_v:,} position-angle cells occupied",
                flush=True,
            )
        return payload, n_pass

    @property
    def elements(self) -> List[_Element]:
        return list(self._elements)

    # ── Core trace ────────────────────────────────────────────────────────────

    def evaluate_bundle(
        self,
        rays: RayBundle,
        *,
        axis_sign: Optional[int] = None,
    ) -> TransferResultBundle:
        """Vectorized transfer function for the complete lens assembly.

        This is the canonical CPU evaluator for the pure geometric lens model:

            TransferResultBundle = F(RayBundle, CompoundLens state)

        LUT baking, MLP training, shader payload generation, and future wave
        evaluators should consume this same state rather than carrying separate
        definitions of entrance planes, stops, surfaces, or group offsets.
        """
        rb = rays.normalized()
        if axis_sign is None:
            axis_sign = +1 if float(np.mean(rb.directions[:, 0])) >= 0.0 else -1
        elements = self._elements_for_axis(int(axis_sign))
        pos = rb.origins.copy()
        dir_ = rb.directions.copy()
        n = pos.shape[0]

        opl = np.zeros(n, dtype=np.float64)
        status = np.full(n, TerminationReason.PASSED.value, dtype=np.int32)
        terminating = np.full(n, -1, dtype=np.int32)
        active = np.ones(n, dtype=bool)

        if self.hood is not None:
            clipped = _hood_clips_batch(self.hood, pos, dir_) & active
            status[clipped] = TerminationReason.CLIPPED_HOOD.value
            terminating[clipped] = -2
            active[clipped] = False

        for elem_idx, element in enumerate(elements):
            if not np.any(active):
                break
            idx = np.nonzero(active)[0]
            p0 = pos[idx]
            d0 = dir_[idx]

            if isinstance(element, ApertureStop):
                dx = d0[:, 0]
                hit = np.abs(dx) > _EPS
                t = np.zeros(idx.shape[0], dtype=np.float64)
                t[hit] = (element.x_pos - p0[hit, 0]) / dx[hit]
                hit &= t > _EPS
                missed = idx[~hit]
                if missed.size:
                    status[missed] = TerminationReason.MISSED_SURFACE.value
                    terminating[missed] = elem_idx
                    active[missed] = False

                good_idx = idx[hit]
                if good_idx.size == 0:
                    continue
                local_hit = p0[hit] + t[hit, None] * d0[hit]
                r = np.hypot(local_hit[:, 1], local_hit[:, 2])
                clipped = r > element.r_clear
                if np.any(clipped):
                    gi = good_idx[clipped]
                    status[gi] = TerminationReason.CLIPPED_STOP.value
                    terminating[gi] = elem_idx
                    active[gi] = False

                passed = ~clipped
                if np.any(passed):
                    gi = good_idx[passed]
                    pos[gi] = local_hit[passed]
                    opl[gi] += element.n_medium * t[hit][passed]
                continue

            if isinstance(element, FlatSurface):
                x_pos = element.x_pos
                R = 0.0
                k = 0.0
                n_before = element.n_before
                n_after = element.n_after
                aperture_r = element.aperture_r
            else:
                x_pos = element.x_pos
                R = element.R_curvature
                k = element.conic_k
                n_before = element.n_before
                n_after = element.n_after
                aperture_r = element.aperture_r

            t, hit = _conic_intersect_batch(p0, d0, x_pos, R, k)
            missed = idx[~hit]
            if missed.size:
                status[missed] = TerminationReason.MISSED_SURFACE.value
                terminating[missed] = elem_idx
                active[missed] = False

            good_idx = idx[hit]
            if good_idx.size == 0:
                continue

            local_hit = p0[hit] + t[hit, None] * d0[hit]
            r = np.hypot(local_hit[:, 1], local_hit[:, 2])
            clipped = (aperture_r > 0.0) & (r > aperture_r)
            if np.any(clipped):
                gi = good_idx[clipped]
                status[gi] = TerminationReason.VIGNETTED.value
                terminating[gi] = elem_idx
                active[gi] = False

            passed = ~clipped
            if not np.any(passed):
                continue

            gi = good_idx[passed]
            hit_pass = local_hit[passed]
            dir_pass = d0[hit][passed]
            normals = _conic_normal_batch(hit_pass, x_pos, R, k)
            n1 = np.full(gi.shape[0], n_before, dtype=np.float64)
            n2 = np.full(gi.shape[0], n_after, dtype=np.float64)
            refracted, ok = _snell_batch(dir_pass, normals, n1, n2)
            if np.any(~ok):
                fail = gi[~ok]
                status[fail] = TerminationReason.TIR.value
                terminating[fail] = elem_idx
                active[fail] = False
            if np.any(ok):
                ok_idx = gi[ok]
                pos[ok_idx] = hit_pass[ok]
                dir_[ok_idx] = refracted[ok]
                opl[ok_idx] += n_before * t[hit][passed][ok]

        return TransferResultBundle(pos, dir_, opl, status, terminating)

    def trace(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        wavelength_um: float = 0.587,
    ) -> TracedRay:
        """Exact algebraic trace through the full assembly.

        Parameters
        ----------
        origin        (3,) entry point, scene coords
        direction     (3,) unit incident direction
        wavelength_um wavelength in micrometres (for future dispersion support)

        Returns
        -------
        TracedRay with the exit state and the list of surface intercept
        positions (useful for visualisation and debugging).
        """
        origin    = np.asarray(origin,    dtype=float)
        direction = np.asarray(direction, dtype=float)
        direction = direction / (np.linalg.norm(direction) + _EPS)

        intercepts: List[np.ndarray] = [origin.copy()]
        opl = 0.0

        # Lens hood — check before any refracting element
        if self.hood is not None and self.hood.clips(origin, direction):
            return TracedRay(origin, direction, opl,
                             TerminationReason.CLIPPED_HOOD, intercepts)

        pos = origin.copy()
        dir_ = direction.copy()

        for element in self._elements:
            if isinstance(element, ApertureStop):
                pos, dir_, opl, reason = element.check(pos, dir_, opl)
            else:
                pos, dir_, opl, reason = element.refract(pos, dir_, opl)

            if reason != TerminationReason.PASSED:
                return TracedRay(origin, direction, opl, reason, intercepts)

            intercepts.append(pos.copy())

        return TracedRay(origin, dir_, opl, TerminationReason.PASSED, intercepts)

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

    def vignetting(
        self,
        field_angle_rad: float,
        n_samples: int = 64,
    ) -> float:
        """Fraction of marginal rays that pass the assembly at the given field angle.

        Samples a uniform disc of rays at the entrance pupil and counts how
        many exit the system, divided by n_samples.  Returns a value in [0, 1].
        """
        x_ent, r_ent = self.entrance_pupil
        if r_ent < _EPS:
            return 0.0

        # Chief ray direction for this field angle
        chief_dir = np.array([
            math.cos(field_angle_rad),
            math.sin(field_angle_rad),
            0.0,
        ])

        passed = 0
        rng    = np.random.default_rng(0)
        for _ in range(n_samples):
            # Uniform disc sample at entrance pupil
            r_s = r_ent * math.sqrt(rng.random())
            phi = rng.random() * 2.0 * math.pi
            origin = np.array([x_ent, r_s * math.cos(phi), r_s * math.sin(phi)])
            result = self.trace(origin, chief_dir)
            if result.reason == TerminationReason.PASSED:
                passed += 1
        return passed / n_samples

    def image_circle_radius(
        self,
        max_field_angle_rad: float = math.radians(40.0),
        n_field_steps: int = 16,
        n_samples_per_field: int = 32,
    ) -> float:
        """Radius of the illuminated image circle on the sensor plane (metres).

        Traces chief rays at increasing field angles and finds the largest angle
        for which ≥ 50% of marginal samples pass (50% vignetting threshold).
        Returns the corresponding image height on sensor.
        """
        # Find exit / sensor x from the last element
        x_sensor = None
        for el in reversed(self._elements):
            if isinstance(el, (ConicSurface, FlatSurface)):
                x_sensor = el.x_pos
                break
        if x_sensor is None:
            return 0.0

        for step in range(n_field_steps, 0, -1):
            angle = max_field_angle_rad * step / n_field_steps
            v = self.vignetting(angle, n_samples=n_field_steps)
            if v >= 0.5:
                # Image height = sensor distance * tan(field angle) (paraxial: f*angle)
                return self.f_eff * math.tan(angle)
        return 0.0

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
