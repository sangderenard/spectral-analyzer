"""parametric_surface.py — Parameterized optical surface geometry.

Each concrete surface provides:

  intersect_ray(origin, direction) → t > 0 or None
      Distance along the ray to the first valid intersection.

  normal_at_point(pos) → np.ndarray (3,) float64
      Outward surface normal at a world-space point.  The convention is that
      the normal points AWAY from the bulk material (toward the incoming ray).

  point_to_uv(pos) → (u, v) or None
      Map a world-space hit point to normalised (u, v) coordinates.  The
      aperture centre maps to (0, 0); the edge of the clear aperture maps to
      unit radius (circular surfaces) or (±1, ±1) (planar rectangular
      surfaces).  Returns None if pos is outside the clear aperture.

  uv_to_point(u, v) → np.ndarray (3,) float64
      Inverse of point_to_uv; maps (u, v) to the corresponding world
      position on the surface.

  project_ray_to_uv(origin, direction) → (u, v) or None
      Convenience: intersect then project.

Coordinate conventions
----------------------
* All positions and directions are in world-space metres / unit vectors.
* (u, v) are dimensionless.  For a circular clear aperture of radius R_ca
  the mapping is (u, v) = (x_obj/R_ca, y_obj/R_ca) so that the edge is at
  |(u,v)| = 1.  Callers may choose to clip to the unit disk or unit square
  depending on aperture shape.

Provided surfaces
-----------------
  PlaneSurface   — flat plane (aperture stops, sensor planes, flat mirrors)
  ConicSurface   — conic section of revolution about a given axis:
                     k = 0          sphere
                     k = -1         paraboloid
                     k < -1         hyperboloid
                    -1 < k < 0      prolate ellipsoid
                     k > 0          oblate ellipsoid
  SphericalSurface — shortcut constructor for ConicSurface(k=0)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class ParametricSurface(ABC):
    """Abstract interface shared by all parameterised optical surfaces."""

    # Subclasses may override to describe their natural (u,v) domain.
    uv_domain: str = "unit_disk"   # "unit_disk" | "unit_square"

    @abstractmethod
    def intersect_ray(
        self,
        origin: np.ndarray,     # (3,) float64
        direction: np.ndarray,  # (3,) float64, need not be unit
    ) -> float | None:
        """Return t > 0 such that origin + t*direction lies on the surface,
        or None if there is no valid intersection within the clear aperture."""

    @abstractmethod
    def normal_at_point(self, pos: np.ndarray) -> np.ndarray:
        """Outward unit surface normal at world-space pos."""

    @abstractmethod
    def point_to_uv(self, pos: np.ndarray) -> tuple[float, float] | None:
        """World-space pos → (u, v), or None if outside clear aperture."""

    @abstractmethod
    def uv_to_point(self, u: float, v: float) -> np.ndarray:
        """(u, v) → world-space position on the surface."""

    # ── Derived convenience ───────────────────────────────────────────────────

    def project_ray_to_uv(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[float, float] | None:
        """Find the surface intersection of origin + t*direction and return
        its (u, v), or None if the ray misses the clear aperture."""
        t = self.intersect_ray(origin, direction)
        if t is None:
            return None
        hit = np.asarray(origin, dtype=np.float64) + t * np.asarray(direction, dtype=np.float64)
        return self.point_to_uv(hit)

    def area_element(
        self,
        u: float,
        v: float,
        eps: float = 1e-4,
    ) -> float:
        """Numerical Jacobian |∂r/∂u × ∂r/∂v| at (u, v).

        Gives the area element for importance-sampling integrals over the
        surface.  Subclasses may override with an analytic version.
        """
        p0 = self.uv_to_point(u, v)
        pu = self.uv_to_point(u + eps, v)
        pv = self.uv_to_point(u, v + eps)
        return float(np.linalg.norm(np.cross(pu - p0, pv - p0))) / (eps * eps)


# ─────────────────────────────────────────────────────────────────────────────
# PlaneSurface
# ─────────────────────────────────────────────────────────────────────────────

class PlaneSurface(ParametricSurface):
    """Flat plane parameterised by two orthonormal in-plane axes.

    The natural coordinate frame has:
      centre → (u=0, v=0)
      u_axis direction at distance half_extent_u → u = 1
      v_axis direction at distance half_extent_v → v = 1

    So (u, v) ∈ [-1, 1]² when pos is within the rectangular extent.
    For a circular aperture of radius R, pass half_extent_u = half_extent_v = R
    and clip callers to |(u,v)| ≤ 1.

    Parameters
    ----------
    centre : (3,) world-space point at the aperture centre.
    normal : (3,) outward unit normal.
    u_axis : (3,) reference in-plane direction (will be orthogonalised).
    half_extent_u, half_extent_v : physical half-widths in metres.
    """

    uv_domain = "unit_square"

    def __init__(
        self,
        centre: np.ndarray,
        normal: np.ndarray,
        u_axis: np.ndarray,
        half_extent_u: float = 1.0,
        half_extent_v: float = 1.0,
    ) -> None:
        self._centre = np.asarray(centre, dtype=np.float64).copy()
        n = np.asarray(normal, dtype=np.float64)
        self._normal = n / np.linalg.norm(n)
        # Orthogonalise u_axis against normal.
        u = np.asarray(u_axis, dtype=np.float64)
        u = u - u.dot(self._normal) * self._normal
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            # u_axis is parallel to normal; pick an arbitrary perpendicular.
            perp = np.array([1.0, 0.0, 0.0])
            if abs(perp.dot(self._normal)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            perp -= perp.dot(self._normal) * self._normal
            u = perp / np.linalg.norm(perp)
        else:
            u = u / u_norm
        self._u_axis = u
        self._v_axis = np.cross(self._normal, self._u_axis)
        self._hu = float(half_extent_u)
        self._hv = float(half_extent_v)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def centre(self) -> np.ndarray:
        return self._centre.copy()

    @property
    def u_axis(self) -> np.ndarray:
        return self._u_axis.copy()

    @property
    def v_axis(self) -> np.ndarray:
        return self._v_axis.copy()

    @property
    def half_extent_u(self) -> float:
        return self._hu

    @property
    def half_extent_v(self) -> float:
        return self._hv

    # ── ParametricSurface interface ───────────────────────────────────────────

    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray) -> float | None:
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        denom = d.dot(self._normal)
        if abs(denom) < 1e-12:
            return None  # ray parallel to plane
        t = (self._centre - o).dot(self._normal) / denom
        if t <= 0.0:
            return None
        hit = o + t * d
        # Clip to rectangular extent.
        delta = hit - self._centre
        u = delta.dot(self._u_axis) / self._hu
        v = delta.dot(self._v_axis) / self._hv
        if abs(u) > 1.0 or abs(v) > 1.0:
            return None
        return float(t)

    def normal_at_point(self, pos: np.ndarray) -> np.ndarray:
        return self._normal.copy()

    def point_to_uv(self, pos: np.ndarray) -> tuple[float, float] | None:
        delta = np.asarray(pos, dtype=np.float64) - self._centre
        u = float(delta.dot(self._u_axis) / self._hu)
        v = float(delta.dot(self._v_axis) / self._hv)
        if abs(u) > 1.0 + 1e-6 or abs(v) > 1.0 + 1e-6:
            return None
        return (u, v)

    def uv_to_point(self, u: float, v: float) -> np.ndarray:
        return (
            self._centre
            + u * self._hu * self._u_axis
            + v * self._hv * self._v_axis
        )

    def area_element(self, u: float, v: float, eps: float = 1e-4) -> float:
        return float(self._hu * self._hv)   # constant; Jacobian is 1


# ─────────────────────────────────────────────────────────────────────────────
# ConicSurface
# ─────────────────────────────────────────────────────────────────────────────

class ConicSurface(ParametricSurface):
    """Conic section of revolution parameterised by (u, v) = (x/R_ca, y/R_ca).

    The surface is defined by the sag formula in object space
    (vertex at origin, optical axis along +z)::

        z = r² / (R * (1 + √(1 − (1+k)·r²/R²)))

    where r² = x² + y², R is the radius of curvature, and k is the conic
    constant.  The world transform is defined by *vertex*, *axis* and
    *u_ref_axis* (orthogonalised against *axis*).

    (u, v) are radial-Cartesian coordinates normalised to the clear aperture::

        u = x_obj / R_ca,   v = y_obj / R_ca

    so that |(u, v)| = 1 at the edge of the clear aperture.

    Parameters
    ----------
    vertex            : (3,) world-space apex of the conic.
    axis              : (3,) optical axis direction (from vertex outward).
    radius_of_curvature : R > 0: centre on +axis side; R < 0: on −axis side.
    conic_constant    : k.  k=0 → sphere.
    clear_aperture_radius : maximum radial extent in metres.
    u_ref_axis        : (3,) reference for φ=0 in the radial plane.
                        Auto-computed perpendicular to *axis* if None.
    """

    uv_domain = "unit_disk"

    def __init__(
        self,
        vertex: np.ndarray,
        axis: np.ndarray,
        radius_of_curvature: float,
        conic_constant: float = 0.0,
        clear_aperture_radius: float = 1.0,
        u_ref_axis: np.ndarray | None = None,
    ) -> None:
        self._vertex = np.asarray(vertex, dtype=np.float64).copy()
        ax = np.asarray(axis, dtype=np.float64)
        self._axis = ax / np.linalg.norm(ax)
        self._R = float(radius_of_curvature)
        self._k = float(conic_constant)
        self._R_ca = float(clear_aperture_radius)

        # Build orthonormal frame: (u_ref, v_ref, axis).
        if u_ref_axis is not None:
            ur = np.asarray(u_ref_axis, dtype=np.float64)
        else:
            ur = np.array([1.0, 0.0, 0.0])
            if abs(ur.dot(self._axis)) > 0.9:
                ur = np.array([0.0, 1.0, 0.0])
        ur -= ur.dot(self._axis) * self._axis
        self._u_ref = ur / np.linalg.norm(ur)
        self._v_ref = np.cross(self._axis, self._u_ref)

    # ── World ↔ object-space helpers ─────────────────────────────────────────

    def _to_obj(self, pos: np.ndarray) -> np.ndarray:
        """World → object (vertex at origin, axis = +z)."""
        d = np.asarray(pos, dtype=np.float64) - self._vertex
        return np.array([d.dot(self._u_ref), d.dot(self._v_ref), d.dot(self._axis)],
                        dtype=np.float64)

    def _to_world(self, x: float, y: float, z: float) -> np.ndarray:
        return (self._vertex
                + x * self._u_ref
                + y * self._v_ref
                + z * self._axis)

    def _ray_to_obj(self, origin: np.ndarray, direction: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform ray to object space."""
        d = np.asarray(direction, dtype=np.float64)
        o_obj = self._to_obj(origin)
        d_obj = np.array([d.dot(self._u_ref), d.dot(self._v_ref), d.dot(self._axis)],
                         dtype=np.float64)
        return o_obj, d_obj

    # ── ParametricSurface interface ───────────────────────────────────────────

    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray) -> float | None:
        o, d = self._ray_to_obj(origin, direction)
        ox, oy, oz = o
        dx, dy, dz = d

        k1 = 1.0 + self._k
        R  = self._R

        A = k1 * dz * dz + dx * dx + dy * dy
        B = 2.0 * (k1 * oz * dz - R * dz + ox * dx + oy * dy)
        C = k1 * oz * oz - 2.0 * R * oz + ox * ox + oy * oy

        t_best: float | None = None

        if abs(A) < 1e-15:
            # Degenerate quadratic (ray along axis for k=-1 paraboloid case).
            if abs(B) > 1e-15:
                t_best = -C / B
        else:
            disc = B * B - 4.0 * A * C
            if disc < 0.0:
                return None
            sq = math.sqrt(disc)
            for t in ((-B - sq) / (2.0 * A), (-B + sq) / (2.0 * A)):
                if t <= 1e-9:
                    continue
                hx = ox + t * dx
                hy = oy + t * dy
                if math.sqrt(hx * hx + hy * hy) > self._R_ca:
                    continue
                if t_best is None or t < t_best:
                    t_best = t

        return t_best

    def normal_at_point(self, pos: np.ndarray) -> np.ndarray:
        x, y, z = self._to_obj(pos)
        # Gradient of F(x,y,z) = (1+k)*z² - 2R*z + x² + y²
        gx = 2.0 * x
        gy = 2.0 * y
        gz = 2.0 * (1.0 + self._k) * z - 2.0 * self._R
        g_world = gx * self._u_ref + gy * self._v_ref + gz * self._axis
        nrm = np.linalg.norm(g_world)
        if nrm < 1e-15:
            return self._axis.copy()
        return g_world / nrm

    def point_to_uv(self, pos: np.ndarray) -> tuple[float, float] | None:
        x, y, _z = self._to_obj(pos)
        r = math.sqrt(x * x + y * y)
        if r > self._R_ca + 1e-9:
            return None
        u = x / self._R_ca
        v = y / self._R_ca
        return (u, v)

    def uv_to_point(self, u: float, v: float) -> np.ndarray:
        x = u * self._R_ca
        y = v * self._R_ca
        r2 = x * x + y * y
        R = self._R
        k1 = 1.0 + self._k
        disc = 1.0 - k1 * r2 / (R * R)
        if disc <= 0.0:
            disc = 0.0
        z = r2 / (R * (1.0 + math.sqrt(max(0.0, disc))))
        return self._to_world(x, y, z)

    def area_element(self, u: float, v: float, eps: float = 1e-4) -> float:
        # Analytic: dA = r * (1 + z'(r)²)^(1/2) * dφ * dr  integrated over bin.
        # Numerical fallback is precise enough for all downstream uses.
        p0 = self.uv_to_point(u, v)
        pu = self.uv_to_point(u + eps, v)
        pv = self.uv_to_point(u, v + eps)
        return float(np.linalg.norm(np.cross(pu - p0, pv - p0))) / (eps * eps)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: SphericalSurface
# ─────────────────────────────────────────────────────────────────────────────

def SphericalSurface(
    vertex: np.ndarray,
    axis: np.ndarray,
    radius_of_curvature: float,
    clear_aperture_radius: float = 1.0,
    u_ref_axis: np.ndarray | None = None,
) -> ConicSurface:
    """Return a ConicSurface with k=0 (spherical surface).

    This is a factory function, not a class, so that isinstance checks
    against ConicSurface still work.
    """
    return ConicSurface(
        vertex=vertex,
        axis=axis,
        radius_of_curvature=radius_of_curvature,
        conic_constant=0.0,
        clear_aperture_radius=clear_aperture_radius,
        u_ref_axis=u_ref_axis,
    )
