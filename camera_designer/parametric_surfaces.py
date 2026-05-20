"""camera_designer/parametric_surfaces.py
=========================================
Parametric surface primitives for camera optical design.

Every surface is defined by a closed-form equation — no mesh, no
triangles.  The GPU bake path calls ``glsl_intercept_fn()`` to get a
GLSL function string that can be inlined into any compute shader.
The CPU path calls ``intersect(ro, rd)`` for 64-bit bake precision.

Coordinate convention
---------------------
All surfaces live in their own **local frame** where the optical axis
is +Z.  The host is responsible for transforming world-space rays into
local frame before calling ``intersect``.

Surface catalogue
-----------------
FlatSurface       — plane perpendicular to Z at z=0 (aperture plane, sensor)
SphericalSurface  — sphere of radius R centred on Z axis
ConicSurface      — spherical + conic constant K (paraboloid, hyperboloid…)
ApertureStop      — annular mask: passes rays with r in [r_inner, r_outer]
ToricSurface      — torus cross-section (for cylindrical / anamorphic elements)
PolynomialSurface — Zernike / even-polynomial asphere

All ``intersect`` methods have signature::

    t, hit, normal = surface.intersect(ro, rd)
    # t      : float64 — ray parameter; np.inf if miss
    # hit    : (3,) float64 — world-space hit point
    # normal : (3,) float64 — outward unit normal at hit
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "ParametricSurface",
    "SensorProperties",
    "FlatSurface",
    "SphericalSurface",
    "ConicSurface",
    "ApertureStop",
    "ToricSurface",
    "PolynomialSurface",
    "LensMountRing",
    "SensorSurface",
    "MirrorSurface",
    "ShutterPlane",
    "CylinderSurface",
    "CatmullRomMesh",
    "surface_from_dict",
]

_INF = np.inf


# ─────────────────────────────────────────────────────────────────────────────
# Sensor properties (attached to any parametric surface)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorProperties:
    """Detector / sensor metadata that can be attached to any parametric surface.

    Set ``surface.sensor_props = SensorProperties(...)`` to mark a surface as
    the detector plane.  The ray tracer terminates rays here and records the
    hit position.  No specific surface type is required — flat, curved,
    polynomial, and toric sensor surfaces are all supported.

    Parameters
    ----------
    pixel_pitch    : pixel centre-to-centre spacing in metres
    sensor_w       : sensor active width  (metres)
    sensor_h       : sensor active height (metres)
    bit_depth      : ADC bit depth (informational)
    qe_peak        : peak quantum efficiency  0-1
    full_well      : full-well capacity in electrons
    read_noise_e   : read-noise in electrons RMS
    """
    pixel_pitch:   float = 3.76e-6    # e.g. 26 MP APS-C pixel
    sensor_w:      float = 0.02359    # APS-C width metres
    sensor_h:      float = 0.01576    # APS-C height metres
    bit_depth:     int   = 14
    qe_peak:       float = 0.60       # 60 % QE typical CMOS
    full_well:     int   = 50_000
    read_noise_e:  float = 2.5

    @property
    def pixel_count_h(self) -> int:
        return max(1, round(self.sensor_w / self.pixel_pitch))

    @property
    def pixel_count_v(self) -> int:
        return max(1, round(self.sensor_h / self.pixel_pitch))

    def to_dict(self) -> dict:
        return {
            "pixel_pitch":  self.pixel_pitch,
            "sensor_w":     self.sensor_w,
            "sensor_h":     self.sensor_h,
            "bit_depth":    self.bit_depth,
            "qe_peak":      self.qe_peak,
            "full_well":    self.full_well,
            "read_noise_e": self.read_noise_e,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SensorProperties":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class ParametricSurface(ABC):
    """Abstract base for all camera-designer parametric surfaces."""

    #: Short GLSL type tag used to dispatch the right intercept function.
    glsl_type: str = "unknown"

    #: Optional sensor metadata.  Set to a SensorProperties instance to mark
    #: this surface as the detector plane.  The ray tracer will terminate rays
    #: here and record the hit position.  Any surface type can be a sensor.
    sensor_props: Optional["SensorProperties"] = None

    @abstractmethod
    def intersect(
        self,
        ro: np.ndarray,
        rd: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Return (t, hit_pt, normal) in surface-local float64.

        Returns t=inf on miss.
        """

    @abstractmethod
    def glsl_intercept_fn(self, fn_name: str = "surf_intercept") -> str:
        """Return a GLSL 460 function with signature::

            bool <fn_name>(vec3 ro, vec3 rd, out float t, out vec3 normal);

        The caller inlines this into its compute shader source.
        """

    # ── Serialisation ─────────────────────────────────────────────────────────

    def intersect_batch(
        self,
        ro: np.ndarray,
        rd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised intersect: ro/rd are (N,3) float64 in surface-local frame.

        Returns (t, hit, normal) each shape (N,) or (N,3).
        t[i]=inf on miss.  Default: scalar loop — override for speed.
        """
        N = len(ro)
        t_arr   = np.empty(N, np.float64)
        hit_arr = np.empty((N, 3), np.float64)
        nrm_arr = np.empty((N, 3), np.float64)
        for i in range(N):
            t_arr[i], hit_arr[i], nrm_arr[i] = self.intersect(ro[i], rd[i])
        return t_arr, hit_arr, nrm_arr

    def to_dict(self) -> dict:
        """Return a plain-Python dict for JSON / YAML serialisation."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> "ParametricSurface":
        """Reconstruct from ``to_dict()`` output."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Flat (aperture / sensor) surface
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlatSurface(ParametricSurface):
    """Plane at z=``z_pos`` with finite semi-diameter ``r_max``.

    Optionally an inner hole ``r_min`` turns it into an annular ring
    (mirror clearance disk, mount ring face, etc.).
    """
    z_pos:  float = 0.0     # metres along optical axis
    r_max:  float = 0.020   # semi-diameter (metres)
    r_min:  float = 0.0     # inner hole radius (0 = solid)

    glsl_type = "flat"

    def intersect(self, ro, rd):
        ro = np.asarray(ro, np.float64)
        rd = np.asarray(rd, np.float64)
        dz = rd[2]
        if abs(dz) < 1e-12:
            return _INF, np.zeros(3), np.zeros(3)
        t = (self.z_pos - ro[2]) / dz
        if t <= 1e-9:
            return _INF, np.zeros(3), np.zeros(3)
        hit = ro + t * rd
        r2 = hit[0]**2 + hit[1]**2
        if r2 > self.r_max**2 or r2 < self.r_min**2:
            return _INF, np.zeros(3), np.zeros(3)
        n = np.array([0., 0., 1. if dz < 0 else -1.], np.float64)
        return t, hit, n

    def intersect_batch(self, ro, rd):
        dz   = rd[:, 2]
        safe = np.abs(dz) > 1e-12
        t    = np.where(safe, (self.z_pos - ro[:, 2]) / np.where(safe, dz, 1.0), np.inf)
        t    = np.where((t > 1e-9) & safe, t, np.inf)
        hit  = ro + t[:, None] * rd
        r2   = hit[:, 0] ** 2 + hit[:, 1] ** 2
        t    = np.where(np.isfinite(t) & (r2 <= self.r_max ** 2) & (r2 >= self.r_min ** 2), t, np.inf)
        nrm  = np.zeros_like(rd)
        nrm[:, 2] = np.where(dz < 0, 1.0, -1.0)
        return t, hit, nrm

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        return f"""
bool {fn_name}(vec3 ro, vec3 rd, out float t, out vec3 normal) {{
    float z_pos = {self.z_pos};
    float r_max = {self.r_max};
    float r_min = {self.r_min};
    if (abs(rd.z) < 1e-9) {{ t = 1e30; return false; }}
    float tt = (z_pos - ro.z) / rd.z;
    if (tt <= 1e-6) {{ t = 1e30; return false; }}
    vec3 h = ro + tt * rd;
    float r2 = h.x*h.x + h.y*h.y;
    if (r2 > r_max*r_max || r2 < r_min*r_min) {{ t = 1e30; return false; }}
    t = tt;
    normal = vec3(0.0, 0.0, rd.z < 0.0 ? 1.0 : -1.0);
    return true;
}}"""

    def to_dict(self):
        return {"type": "flat", "z_pos": self.z_pos,
                "r_max": self.r_max, "r_min": self.r_min}


# ─────────────────────────────────────────────────────────────────────────────
# Spherical surface
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SphericalSurface(ParametricSurface):
    """Spherical surface with radius of curvature R and semi-diameter r_max.

    Sign convention: R > 0 = centre of curvature on +Z side (convex facing -Z).
    The vertex is at z=0.
    """
    R:      float = 0.050   # radius of curvature (metres)
    r_max:  float = 0.020   # semi-diameter
    r_min:  float = 0.0

    glsl_type = "spherical"

    def intersect(self, ro, rd):
        ro = np.asarray(ro, np.float64)
        rd = np.asarray(rd, np.float64)
        # Centre of sphere at (0, 0, R)
        oc = ro - np.array([0., 0., self.R], np.float64)
        a = np.dot(rd, rd)
        b = 2.0 * np.dot(oc, rd)
        c = np.dot(oc, oc) - self.R**2
        disc = b*b - 4*a*c
        if disc < 0:
            return _INF, np.zeros(3), np.zeros(3)
        sq = math.sqrt(disc)
        t1 = (-b - sq) / (2*a)
        t2 = (-b + sq) / (2*a)
        t = None
        for tt in sorted([t1, t2]):
            if tt > 1e-9:
                hit = ro + tt * rd
                r2 = hit[0]**2 + hit[1]**2
                if self.r_min**2 <= r2 <= self.r_max**2:
                    t = tt; break
        if t is None:
            return _INF, np.zeros(3), np.zeros(3)
        hit = ro + t * rd
        # Normal points from centre to hit
        ctr = np.array([0., 0., self.R], np.float64)
        n = (hit - ctr) / abs(self.R)
        if np.dot(n, rd) > 0:
            n = -n
        return t, hit, n

    def intersect_batch(self, ro, rd):
        ctr  = np.array([0., 0., self.R], np.float64)
        oc   = ro - ctr[None, :]
        a    = np.sum(rd * rd, axis=1)
        b    = 2.0 * np.sum(oc * rd, axis=1)
        c    = np.sum(oc * oc, axis=1) - self.R ** 2
        disc = b * b - 4 * a * c
        sq   = np.sqrt(np.maximum(disc, 0.0))
        t1   = (-b - sq) / (2 * a)
        t2   = (-b + sq) / (2 * a)
        h1   = ro + t1[:, None] * rd
        h2   = ro + t2[:, None] * rd
        r2_1 = h1[:, 0] ** 2 + h1[:, 1] ** 2
        r2_2 = h2[:, 0] ** 2 + h2[:, 1] ** 2
        rim2 = self.r_min ** 2
        rox2 = self.r_max ** 2
        ok1  = (disc >= 0) & (t1 > 1e-9) & (r2_1 >= rim2) & (r2_1 <= rox2)
        ok2  = (disc >= 0) & (t2 > 1e-9) & (r2_2 >= rim2) & (r2_2 <= rox2)
        t    = np.where(ok1, t1, np.where(ok2, t2, np.inf))
        hit  = np.where(ok1[:, None], h1, np.where(ok2[:, None], h2, ro))
        nrm  = (hit - ctr[None, :]) / abs(self.R)
        flip = np.sum(nrm * rd, axis=1) > 0
        nrm  = np.where(flip[:, None], -nrm, nrm)
        return t, hit, nrm

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        return f"""
bool {fn_name}(vec3 ro, vec3 rd, out float t, out vec3 normal) {{
    float R = {self.R};
    float r_max = {self.r_max};
    float r_min = {self.r_min};
    vec3 ctr = vec3(0.0, 0.0, R);
    vec3 oc  = ro - ctr;
    float b  = 2.0 * dot(oc, rd);
    float c  = dot(oc, oc) - R*R;
    float disc = b*b - 4.0*c;
    if (disc < 0.0) {{ t = 1e30; return false; }}
    float sq = sqrt(disc);
    float t1 = (-b - sq) * 0.5;
    float t2 = (-b + sq) * 0.5;
    t = 1e30;
    for (int i = 0; i < 2; i++) {{
        float tt = (i == 0) ? t1 : t2;
        if (tt > 1e-6) {{
            vec3 h = ro + tt * rd;
            float r2 = h.x*h.x + h.y*h.y;
            if (r2 >= r_min*r_min && r2 <= r_max*r_max) {{
                t = tt; break;
            }}
        }}
    }}
    if (t >= 1e29) return false;
    vec3 h = ro + t * rd;
    normal  = normalize(h - ctr);
    if (dot(normal, rd) > 0.0) normal = -normal;
    return true;
}}"""

    def to_dict(self):
        return {"type": "spherical", "R": self.R,
                "r_max": self.r_max, "r_min": self.r_min}


# ─────────────────────────────────────────────────────────────────────────────
# Conic surface  (spherical + conic constant K)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConicSurface(ParametricSurface):
    """General conic of revolution: z = r²/(R(1+√(1-(1+K)r²/R²))).

    K = 0      sphere
    K = -1     paraboloid
    K < -1     hyperboloid
    K > 0      oblate ellipsoid

    Uses iterative Newton refinement for sub-micron 64-bit precision.
    GPU path uses the closed-form quadratic for the conic quadric.
    """
    R:      float = 0.050
    K:      float = 0.0
    r_max:  float = 0.020
    r_min:  float = 0.0

    glsl_type = "conic"

    # ---- sag formula (z as a function of r²) --------------------------------
    def _sag(self, r2: float) -> float:
        rR = r2 / (self.R**2)
        under = 1.0 - (1.0 + self.K) * rR
        if under < 0:
            return math.nan
        return r2 / (self.R * (1.0 + math.sqrt(under)))

    def intersect(self, ro, rd):
        ro = np.asarray(ro, np.float64)
        rd = np.asarray(rd, np.float64)
        # Conic quadric: (1+K)(x²+y²+z²) - 2Rz = 0  when K≠0
        # Use quadratic in t; iterate for precision
        K1 = 1.0 + self.K
        # F(t) = K1*(xt²+yt²+zt²) - 2R*zt  where xt = ro.x+t*rd.x etc.
        # Expand to At² + Bt + C = 0
        A = K1 * (rd[0]**2 + rd[1]**2) + K1 * rd[2]**2 - 0.0
        # Simplified: A = rd·rd * K1  (when K=0 reduces to sphere centre at R)
        # Actually the conic quadric is:
        #   (1+K)(x²+y²) + z² - 2Rz = 0
        A2 = K1*(rd[0]**2 + rd[1]**2) + rd[2]**2
        B2 = 2*(K1*(ro[0]*rd[0] + ro[1]*rd[1]) + ro[2]*rd[2] - self.R*rd[2])
        C2 = K1*(ro[0]**2 + ro[1]**2) + ro[2]**2 - 2*self.R*ro[2]
        disc = B2**2 - 4*A2*C2
        if disc < 0 or abs(A2) < 1e-30:
            return _INF, np.zeros(3), np.zeros(3)
        sq = math.sqrt(disc)
        t = None
        for tt in sorted([(-B2-sq)/(2*A2), (-B2+sq)/(2*A2)]):
            if tt > 1e-9:
                hit = ro + tt * rd
                r2 = hit[0]**2 + hit[1]**2
                if self.r_min**2 <= r2 <= self.r_max**2:
                    t = tt; break
        if t is None:
            return _INF, np.zeros(3), np.zeros(3)
        # Newton step for extra precision
        for _ in range(3):
            hit = ro + t * rd
            r2  = hit[0]**2 + hit[1]**2
            F   = K1*(hit[0]**2+hit[1]**2) + hit[2]**2 - 2*self.R*hit[2]
            dF  = 2*(K1*(hit[0]*rd[0]+hit[1]*rd[1]) + hit[2]*rd[2] - self.R*rd[2])
            if abs(dF) < 1e-30:
                break
            t -= F / dF
        hit = ro + t * rd
        # Gradient of implicit F = (1+K)(x²+y²) + z² - 2Rz
        n = np.array([
            2*K1*hit[0],
            2*K1*hit[1],
            2*hit[2] - 2*self.R,
        ], np.float64)
        n /= max(np.linalg.norm(n), 1e-30)
        if np.dot(n, rd) > 0:
            n = -n
        return t, hit, n

    def intersect_batch(self, ro, rd):
        K1   = 1.0 + self.K
        A2   = K1 * (rd[:, 0] ** 2 + rd[:, 1] ** 2) + rd[:, 2] ** 2
        B2   = 2 * (K1 * (ro[:, 0] * rd[:, 0] + ro[:, 1] * rd[:, 1])
                    + ro[:, 2] * rd[:, 2] - self.R * rd[:, 2])
        C2   = K1 * (ro[:, 0] ** 2 + ro[:, 1] ** 2) + ro[:, 2] ** 2 - 2 * self.R * ro[:, 2]
        disc = B2 ** 2 - 4 * A2 * C2
        ok_a = np.abs(A2) > 1e-30
        sq   = np.sqrt(np.maximum(disc, 0.0))
        denom = np.where(ok_a, 2 * A2, 1.0)
        t1   = np.where(ok_a, (-B2 - sq) / denom, np.inf)
        t2   = np.where(ok_a, (-B2 + sq) / denom, np.inf)
        h1   = ro + t1[:, None] * rd
        h2   = ro + t2[:, None] * rd
        r2_1 = h1[:, 0] ** 2 + h1[:, 1] ** 2
        r2_2 = h2[:, 0] ** 2 + h2[:, 1] ** 2
        rim2 = self.r_min ** 2
        rox2 = self.r_max ** 2
        ok1  = ok_a & (disc >= 0) & (t1 > 1e-9) & (r2_1 >= rim2) & (r2_1 <= rox2)
        ok2  = ok_a & (disc >= 0) & (t2 > 1e-9) & (r2_2 >= rim2) & (r2_2 <= rox2)
        t    = np.where(ok1, t1, np.where(ok2, t2, np.inf))
        # Newton refinement on finite-t rays
        with np.errstate(invalid='ignore', divide='ignore'):
            alive = np.isfinite(t)
            for _ in range(3):
                if not np.any(alive):
                    break
                h  = ro + t[:, None] * rd
                F  = K1 * (h[:, 0] ** 2 + h[:, 1] ** 2) + h[:, 2] ** 2 - 2 * self.R * h[:, 2]
                dF = 2 * (K1 * (h[:, 0] * rd[:, 0] + h[:, 1] * rd[:, 1])
                          + h[:, 2] * rd[:, 2] - self.R * rd[:, 2])
                safe_dF = np.abs(dF) > 1e-30
                t = np.where(alive & safe_dF, t - F / np.where(safe_dF, dF, 1.0), t)
            hit  = ro + t[:, None] * rd
            nrm  = np.stack([2 * K1 * hit[:, 0],
                             2 * K1 * hit[:, 1],
                             2 * hit[:, 2] - 2 * self.R], axis=1)
            nrm  = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
        flip = np.sum(nrm * rd, axis=1) > 0
        nrm  = np.where(flip[:, None], -nrm, nrm)
        r2   = hit[:, 0] ** 2 + hit[:, 1] ** 2
        t    = np.where(alive & (t > 1e-9) & (r2 >= rim2) & (r2 <= rox2), t, np.inf)
        return t, hit, nrm

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        return f"""
bool {fn_name}(vec3 ro, vec3 rd, out float t, out vec3 normal) {{
    float R   = {self.R};
    float K   = {self.K};
    float r_max = {self.r_max};
    float r_min = {self.r_min};
    float K1  = 1.0 + K;
    float A   = K1*(rd.x*rd.x + rd.y*rd.y) + rd.z*rd.z;
    float B   = 2.0*(K1*(ro.x*rd.x + ro.y*rd.y) + ro.z*rd.z - R*rd.z);
    float C   = K1*(ro.x*ro.x + ro.y*ro.y) + ro.z*ro.z - 2.0*R*ro.z;
    float disc = B*B - 4.0*A*C;
    if (disc < 0.0 || abs(A) < 1e-20) {{ t = 1e30; return false; }}
    float sq = sqrt(disc);
    float t1 = (-B - sq) / (2.0*A);
    float t2 = (-B + sq) / (2.0*A);
    t = 1e30;
    for (int i = 0; i < 2; i++) {{
        float tt = (i==0) ? t1 : t2;
        if (tt > 1e-6) {{
            vec3 h = ro + tt*rd;
            float r2 = h.x*h.x + h.y*h.y;
            if (r2 >= r_min*r_min && r2 <= r_max*r_max) {{
                t = tt; break;
            }}
        }}
    }}
    if (t >= 1e29) return false;
    // Newton refinement (2 steps)
    for (int it = 0; it < 2; it++) {{
        vec3 h = ro + t*rd;
        float F  = K1*(h.x*h.x+h.y*h.y) + h.z*h.z - 2.0*R*h.z;
        float dF = 2.0*(K1*(h.x*rd.x+h.y*rd.y) + h.z*rd.z - R*rd.z);
        if (abs(dF) < 1e-20) break;
        t -= F/dF;
    }}
    vec3 h  = ro + t*rd;
    normal  = normalize(vec3(2.0*K1*h.x, 2.0*K1*h.y, 2.0*h.z - 2.0*R));
    if (dot(normal, rd) > 0.0) normal = -normal;
    return true;
}}"""

    def to_dict(self):
        return {"type": "conic", "R": self.R, "K": self.K,
                "r_max": self.r_max, "r_min": self.r_min}


# ─────────────────────────────────────────────────────────────────────────────
# Aperture stop — annular mask (no geometry, just a gating plane)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ApertureStop(ParametricSurface):
    """Opaque plane at z=``z_pos`` with a clear aperture opening.

    Shape of the opening:
      n_blades == 0  →  circular annulus  r ∈ [r_inner, r_outer]
      n_blades >= 3  →  regular N-gon inscribed in r_outer, rotated by
                        aperture_rot radians (first-blade edge angle).

    Rays hitting the opaque region are terminated (return t=inf).
    Rays through the clear opening pass with t from the plane hit.

    ``blade_polygon_xy()`` returns the (n_blades, 2) float64 vertex array
    of the opening polygon in metres — suitable for passing directly to
    ``ray_tracer_apply_aperture_mask``.  For n_blades==0 a high-resolution
    circle approximation is returned instead.
    """
    z_pos:       float = 0.0
    r_inner:     float = 0.0
    r_outer:     float = 0.010
    n_blades:    int   = 0      # 0 = circle; >=3 = regular N-gon iris
    aperture_rot: float = 0.0   # first-blade edge angle, radians

    glsl_type = "aperture_stop"

    # ── Polygon helpers ───────────────────────────────────────────────────

    def blade_polygon_xy(self, circle_approx_n: int = 64) -> np.ndarray:
        """Return (N, 2) float64 polygon vertices for the clear opening.

        For a bladed iris (n_blades >= 3) these are the exact N-gon corners
        inscribed in r_outer, rotated by aperture_rot.
        For a circular aperture (n_blades < 3) a regular circle_approx_n-gon
        at radius r_outer is returned as a smooth approximation.
        """
        n = self.n_blades if self.n_blades >= 3 else circle_approx_n
        angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False) + self.aperture_rot
        return self.r_outer * np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def _in_opening(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the clear aperture opening."""
        r2 = x * x + y * y
        if r2 < self.r_inner ** 2 or r2 > self.r_outer ** 2 + 1e-15:
            return False
        if self.n_blades < 3:
            return True
        # Point-in-N-gon test: check that the point is inside every half-plane
        # defined by each blade edge (convex polygon).
        poly = self.blade_polygon_xy()
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            # Cross product of edge vector with point vector (inside = same sign)
            if (bx - ax) * (y - ay) - (by - ay) * (x - ax) < 0:
                return False
        return True

    # ── ParametricSurface interface ───────────────────────────────────────

    def intersect(self, ro, rd):
        ro = np.asarray(ro, np.float64)
        rd = np.asarray(rd, np.float64)
        if abs(rd[2]) < 1e-12:
            return _INF, np.zeros(3), np.zeros(3)
        t = (self.z_pos - ro[2]) / rd[2]
        if t <= 1e-9:
            return _INF, np.zeros(3), np.zeros(3)
        hit = ro + t * rd
        if not self._in_opening(float(hit[0]), float(hit[1])):
            return _INF, np.zeros(3), np.zeros(3)
        n = np.array([0., 0., 1. if rd[2] < 0 else -1.], np.float64)
        return t, hit, n

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        """Generate GLSL for the aperture test.

        For n_blades >= 3 the N-gon half-plane test is unrolled inline.
        For circles the original r² test is used.
        """
        if self.n_blades >= 3:
            poly = self.blade_polygon_xy()
            n    = len(poly)
            edge_tests = []
            for i in range(n):
                ax, ay = poly[i]
                bx, by = poly[(i + 1) % n]
                edge_tests.append(
                    f"    if (({bx-ax:.9f}*(h.y-({ay:.9f})) - ({by-ay:.9f})*(h.x-({ax:.9f}))) < 0.0) "
                    f"{{ t = 1e30; return false; }}"
                )
            edge_block = "\n".join(edge_tests)
            return f"""
bool {fn_name}(vec3 ro, vec3 rd, out float t, out vec3 normal) {{
    if (abs(rd.z) < 1e-9) {{ t = 1e30; return false; }}
    float tt = ({self.z_pos:.9f} - ro.z) / rd.z;
    if (tt <= 1e-6) {{ t = 1e30; return false; }}
    vec3 h = ro + tt * rd;
    float r2 = h.x*h.x + h.y*h.y;
    if (r2 < {self.r_inner*self.r_inner:.9e} || r2 > {self.r_outer*self.r_outer:.9e})
        {{ t = 1e30; return false; }}
{edge_block}
    t = tt;
    normal = vec3(0.0, 0.0, rd.z < 0.0 ? 1.0 : -1.0);
    return true;
}}"""
        else:
            return f"""
bool {fn_name}(vec3 ro, vec3 rd, out float t, out vec3 normal) {{
    if (abs(rd.z) < 1e-9) {{ t = 1e30; return false; }}
    float tt = ({self.z_pos:.9f} - ro.z) / rd.z;
    if (tt <= 1e-6) {{ t = 1e30; return false; }}
    vec3 h = ro + tt * rd;
    float r2 = h.x*h.x + h.y*h.y;
    if (r2 < {self.r_inner*self.r_inner:.9e} || r2 > {self.r_outer*self.r_outer:.9e})
        {{ t = 1e30; return false; }}
    t = tt;
    normal = vec3(0.0, 0.0, rd.z < 0.0 ? 1.0 : -1.0);
    return true;
}}"""

    def to_dict(self):
        return {
            "type":         "aperture_stop",
            "z_pos":        self.z_pos,
            "r_inner":      self.r_inner,
            "r_outer":      self.r_outer,
            "n_blades":     self.n_blades,
            "aperture_rot": self.aperture_rot,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial (Zernike / even-asphere) surface
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolynomialSurface(ParametricSurface):
    """Even-order polynomial asphere on top of a base conic.

    z(r) = r²/(R(1+√(1-(1+K)r²/R²))) + Σ Aₙ r^(2n)  for n=2..N

    ``coeffs`` is indexed from A2 (coefficient of r⁴) onward.
    GPU path evaluates Horner's method with a generated GLSL loop.
    """
    R:      float = 0.050
    K:      float = 0.0
    r_max:  float = 0.020
    r_min:  float = 0.0
    coeffs: list  = field(default_factory=list)  # [A2, A3, A4, ...]

    glsl_type = "polynomial"

    def _sag(self, r: float) -> float:
        r2 = r * r
        rR2 = r2 / (self.R**2)
        under = 1.0 - (1.0 + self.K) * rR2
        if under < 0:
            return math.nan
        z = r2 / (self.R * (1.0 + math.sqrt(under)))
        for n, a in enumerate(self.coeffs, start=2):
            z += a * r2**(n)
        return z

    def intersect(self, ro, rd):
        """Iterative ray-surface solver via bracketed Newton on sag residual."""
        ro = np.asarray(ro, np.float64)
        rd = np.asarray(rd, np.float64)
        # Fast bracket: march along ray in ~50 steps until sign change in
        # F(t) = z(t) - sag(sqrt(x(t)²+y(t)²))
        N = 64
        t_vals = np.linspace(1e-6, 0.5, N)
        prev_F = None
        t_bracket = None
        for tt in t_vals:
            h = ro + tt * rd
            r = math.sqrt(h[0]**2 + h[1]**2)
            if r > self.r_max:
                continue
            s = self._sag(r)
            if math.isnan(s):
                continue
            F = h[2] - s
            if prev_F is not None and prev_F * F < 0:
                t_bracket = (tt - (t_vals[1]-t_vals[0]), tt)
                break
            prev_F = F
        if t_bracket is None:
            return _INF, np.zeros(3), np.zeros(3)
        # Newton inside bracket
        t = 0.5 * (t_bracket[0] + t_bracket[1])
        for _ in range(16):
            h = ro + t * rd
            r = math.sqrt(h[0]**2 + h[1]**2)
            s = self._sag(r)
            if math.isnan(s):
                break
            F = h[2] - s
            # Numerical derivative
            eps = 1e-8
            h2 = ro + (t+eps) * rd
            r2 = math.sqrt(h2[0]**2 + h2[1]**2)
            s2 = self._sag(r2)
            dF = (h2[2] - s2 - F) / eps
            if abs(dF) < 1e-30:
                break
            t -= F / dF
        hit = ro + t * rd
        r = math.sqrt(hit[0]**2 + hit[1]**2)
        if not (self.r_min <= r <= self.r_max):
            return _INF, np.zeros(3), np.zeros(3)
        # Numerical normal
        eps = 1e-7
        def _F(p):
            rp = math.sqrt(p[0]**2 + p[1]**2)
            return p[2] - self._sag(rp)
        gx = (_F(hit + np.array([eps,0,0])) - _F(hit - np.array([eps,0,0]))) / (2*eps)
        gy = (_F(hit + np.array([0,eps,0])) - _F(hit - np.array([0,eps,0]))) / (2*eps)
        gz = 1.0
        n = np.array([gx, gy, gz], np.float64)
        n /= max(np.linalg.norm(n), 1e-30)
        if np.dot(n, rd) > 0:
            n = -n
        return t, hit, n

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        # Build Horner polynomial in r² for the aspheric add-on
        if self.coeffs:
            terms = " + ".join(
                f"({a}) * pow(r2, {n+2.0})"
                for n, a in enumerate(self.coeffs)
            )
            poly_expr = f"asph = {terms};"
        else:
            poly_expr = "asph = 0.0;"
        return f"""
bool {fn_name}(vec3 ro, vec3 rd, out float t, out vec3 normal) {{
    float R     = {self.R};
    float K     = {self.K};
    float r_max = {self.r_max};
    float r_min = {self.r_min};
    float K1    = 1.0 + K;
    // Implicit: z - sag(r) = 0; iterate with Newton in t
    t = 0.02;  // initial guess (metres along optical axis)
    for (int iter = 0; iter < 20; iter++) {{
        vec3  h   = ro + t * rd;
        float r2  = h.x*h.x + h.y*h.y;
        float rR2 = r2 / (R*R);
        float under = 1.0 - K1*rR2;
        if (under < 1e-9) {{ t = 1e30; return false; }}
        float base = r2 / (R * (1.0 + sqrt(under)));
        float asph;
        {poly_expr}
        float F   = h.z - (base + asph);
        float dFdt = rd.z - 0.0; // approx; next step adds radial deriv
        // Numerical dF/dt
        float eps = 1e-6;
        vec3 h2 = ro + (t+eps)*rd;
        float r2b = h2.x*h2.x + h2.y*h2.y;
        float underb = 1.0 - K1*(r2b/(R*R));
        float asph2; float rR2b = r2b/(R*R);
        {{float under_ = max(underb,1e-9); {poly_expr.replace('r2','r2b').replace('asph','asph2')}}}
        float baseb = r2b / (R*(1.0+sqrt(max(underb,1e-9))));
        float Fb  = h2.z - (baseb + asph2);
        float dF  = (Fb - F) / eps;
        if (abs(dF) < 1e-20) break;
        t -= F / dF;
    }}
    if (t < 1e-6 || t > 1e29) return false;
    vec3 h = ro + t*rd;
    float r2 = h.x*h.x + h.y*h.y;
    float r  = sqrt(r2);
    if (r < r_min || r > r_max) {{ t = 1e30; return false; }}
    // Numerical normal via finite difference
    float ep = 1e-5;
    float r2px = (h.x+ep)*(h.x+ep)+h.y*h.y;
    float r2py = h.x*h.x+(h.y+ep)*(h.y+ep);
    float K1_ = K1;
    float sag_h  = r2  / (R*(1.0+sqrt(max(1.0-K1_*r2/(R*R),  1e-9))));
    float sag_px = r2px/ (R*(1.0+sqrt(max(1.0-K1_*r2px/(R*R),1e-9))));
    float sag_py = r2py/ (R*(1.0+sqrt(max(1.0-K1_*r2py/(R*R),1e-9))));
    normal = normalize(vec3(-(sag_px-sag_h)/ep, -(sag_py-sag_h)/ep, 1.0));
    if (dot(normal, rd) > 0.0) normal = -normal;
    return true;
}}"""

    def to_dict(self):
        return {"type": "polynomial", "R": self.R, "K": self.K,
                "r_max": self.r_max, "r_min": self.r_min,
                "coeffs": list(self.coeffs)}


# ─────────────────────────────────────────────────────────────────────────────
# Lens mount ring  (the physical flange / back clearance marker)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LensMountRing(ParametricSurface):
    """Annular flange face at a fixed Z position.

    Used to mark the lens mount datum, back-focal-distance reference, and
    minimum mirror clearance.  Functionally an annular FlatSurface.

    Parameters
    ----------
    z_flange      : Z position of the mount face (metres)
    r_inner       : inner clear radius (mirror clearance circle)
    r_outer       : outer mount ring radius
    back_clearance: minimum distance from z_flange to first optic vertex
    """
    z_flange:       float = -0.0445   # e.g. Canon EF = 44.5 mm
    r_inner:        float = 0.021     # mirror clearance ~21 mm
    r_outer:        float = 0.031     # mount inner thread radius
    back_clearance: float = 0.004     # minimum back clearance metres

    glsl_type = "mount_ring"

    @property
    def aperture_plane_z(self) -> float:
        """Z position of the aperture manifold plane (flange + back_clearance)."""
        return self.z_flange + self.back_clearance

    def intersect(self, ro, rd):
        flat = FlatSurface(z_pos=self.z_flange, r_max=self.r_outer,
                           r_min=self.r_inner)
        return flat.intersect(ro, rd)

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        flat = FlatSurface(z_pos=self.z_flange, r_max=self.r_outer,
                           r_min=self.r_inner)
        return flat.glsl_intercept_fn(fn_name)

    def to_dict(self):
        return {"type": "mount_ring",
                "z_flange": self.z_flange,
                "r_inner": self.r_inner, "r_outer": self.r_outer,
                "back_clearance": self.back_clearance}


# ─────────────────────────────────────────────────────────────────────────────
# Sensor surface (flat, may be slightly curved for field curvature correction)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorSurface(ParametricSurface):
    """Parametric sensor plane — flat by default, optionally curved.

    The sensor collects rays that hit within ``r_max`` of the optical axis.
    ``pixel_pitch`` is informational (used by the bake path to decide LUT
    resolution).
    """
    z_pos:       float = -0.0444   # just behind the mount face
    r_max:       float = 0.0215   # APS-C half-diagonal ~21.5 mm
    pixel_pitch: float = 3.76e-6  # metres (e.g. 26 MP APS-C)
    curvature_R: float = 0.0      # 0 = flat; non-zero = curved sensor

    glsl_type = "sensor"

    def intersect(self, ro, rd):
        if abs(self.curvature_R) < 1e-9:
            return FlatSurface(z_pos=self.z_pos, r_max=self.r_max).intersect(ro, rd)
        s = SphericalSurface(R=self.curvature_R, r_max=self.r_max)
        return s.intersect(ro, rd)

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        if abs(self.curvature_R) < 1e-9:
            return FlatSurface(z_pos=self.z_pos, r_max=self.r_max).glsl_intercept_fn(fn_name)
        return SphericalSurface(R=self.curvature_R, r_max=self.r_max).glsl_intercept_fn(fn_name)

    def to_dict(self):
        return {"type": "sensor", "z_pos": self.z_pos,
                "r_max": self.r_max, "pixel_pitch": self.pixel_pitch,
                "curvature_R": self.curvature_R}


# ─────────────────────────────────────────────────────────────────────────────
# Toric surface (anamorphic / cylindrical)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToricSurface(ParametricSurface):
    """Toric surface: different radii in X and Y meridians.

    Implicit: (√(x²+y²) - R_major)² + z² = R_minor²
    In the paraxial XZ plane: spherical with R=R_minor.
    In the YZ plane: different curvature.

    Used for cylindrical lenses and anamorphic optics.
    """
    R_tangential:  float = 0.050   # curvature in X meridian
    R_sagittal:    float = 0.060   # curvature in Y meridian
    r_max:         float = 0.020
    r_min:         float = 0.0

    glsl_type = "toric"

    def intersect(self, ro, rd):
        # Approximate: average radius for now; full toric solver is
        # iterative.  This is sufficient for the bake path which
        # samples densely enough to capture the aberration.
        R_avg = 0.5 * (self.R_tangential + self.R_sagittal)
        s = SphericalSurface(R=R_avg, r_max=self.r_max, r_min=self.r_min)
        t, hit, n = s.intersect(ro, rd)
        if not math.isfinite(t):
            return _INF, np.zeros(3), np.zeros(3)
        # Correct normal for toric shape
        rx = self.R_tangential; ry = self.R_sagittal
        gx = 2*hit[0]/rx**2 if abs(rx) > 1e-9 else 0.
        gy = 2*hit[1]/ry**2 if abs(ry) > 1e-9 else 0.
        gz = 2*hit[2]
        n2 = np.array([gx, gy, gz], np.float64)
        n2 /= max(np.linalg.norm(n2), 1e-30)
        if np.dot(n2, rd) > 0:
            n2 = -n2
        return t, hit, n2

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        # Use the average-sphere approximation in the shader too
        R_avg = 0.5 * (self.R_tangential + self.R_sagittal)
        s = SphericalSurface(R=R_avg, r_max=self.r_max, r_min=self.r_min)
        return s.glsl_intercept_fn(fn_name)

    def to_dict(self):
        return {"type": "toric",
                "R_tangential": self.R_tangential,
                "R_sagittal":   self.R_sagittal,
                "r_max": self.r_max, "r_min": self.r_min}


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {
    "flat":          FlatSurface,
    "spherical":     SphericalSurface,
    "conic":         ConicSurface,
    "aperture_stop": ApertureStop,
    "polynomial":    PolynomialSurface,
    "mount_ring":    LensMountRing,
    "sensor":        SensorSurface,
    "toric":         ToricSurface,
    "mirror":        None,   # filled after MirrorSurface defined below
    "shutter":       None,   # filled after ShutterPlane defined below
    "cylinder":      None,   # filled after CylinderSurface defined below
}


def surface_from_dict(d: dict) -> ParametricSurface:
    """Reconstruct any ParametricSurface from its ``to_dict()`` representation."""
    kind = d.get("type", "flat")
    cls  = _REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"Unknown surface type: {kind!r}")
    kwargs = {k: v for k, v in d.items() if k != "type"}
    return cls(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Mirror surface  (reflects; does not refract)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MirrorSurface(ParametricSurface):
    """Flat reflective plane.  The surface normal can be tilted about X and Y
    to produce fold-mirror and beamsplitter geometry.

    Parameters
    ----------
    z_pos      : Z position of the mirror plane vertex (metres)
    r_max      : semi-diameter of the reflective surface
    r_min      : inner hole (0 = solid)
    tilt_x_deg : rotation about X axis (degrees); 45 → fold mirror
    tilt_y_deg : rotation about Y axis (degrees)
    """
    z_pos:      float = 0.0
    r_max:      float = 0.025
    r_min:      float = 0.0
    tilt_x_deg: float = 0.0   # 45 ° = fold mirror
    tilt_y_deg: float = 0.0

    glsl_type = "mirror"

    def _normal_vec(self) -> np.ndarray:
        """Outward surface normal in local frame (before ray-sign flip)."""
        tx = math.radians(self.tilt_x_deg)
        ty = math.radians(self.tilt_y_deg)
        # Rotate the +Z normal by ty then tx
        nx =  math.sin(ty)
        ny = -math.sin(tx) * math.cos(ty)
        nz =  math.cos(tx) * math.cos(ty)
        return np.array([nx, ny, nz], np.float64)

    def intersect(self, ro, rd):
        ro = np.asarray(ro, np.float64)
        rd = np.asarray(rd, np.float64)
        n = self._normal_vec()
        # Plane through (0, 0, z_pos) with normal n
        # t = (z_pos - ro·n_hat * e3_n) / (rd·n)
        #   = ((z_pos*n.z - ro·n*(0,0,z_pos)) ... use signed plane formula:
        # plane: n · (p - p0) = 0  where p0 = (0,0,z_pos)
        p0 = np.array([0., 0., self.z_pos], np.float64)
        denom = np.dot(rd, n)
        if abs(denom) < 1e-12:
            return _INF, np.zeros(3), np.zeros(3)
        t = np.dot(p0 - ro, n) / denom
        if t <= 1e-9:
            return _INF, np.zeros(3), np.zeros(3)
        hit = ro + t * rd
        r2 = hit[0]**2 + hit[1]**2
        if r2 > self.r_max**2 or r2 < self.r_min**2:
            return _INF, np.zeros(3), np.zeros(3)
        nout = n if np.dot(n, rd) < 0 else -n
        return t, hit, nout

    def reflect(self, rd: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Reflect direction *rd* off surface normal *n*."""
        rd = np.asarray(rd, np.float64)
        n  = np.asarray(n,  np.float64)
        return rd - 2.0 * np.dot(rd, n) * n

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        nx, ny, nz = self._normal_vec()
        return f"""
bool {fn_name}(vec3 ro, vec3 rd, out float t, out vec3 normal) {{
    vec3 n  = vec3({nx}, {ny}, {nz});
    vec3 p0 = vec3(0.0, 0.0, {self.z_pos});
    float denom = dot(rd, n);
    if (abs(denom) < 1e-9) {{ t = 1e30; return false; }}
    float tt = dot(p0 - ro, n) / denom;
    if (tt <= 1e-6) {{ t = 1e30; return false; }}
    vec3 h = ro + tt * rd;
    float r2 = h.x*h.x + h.y*h.y;
    if (r2 > {self.r_max*self.r_max} || r2 < {self.r_min*self.r_min})
        {{ t = 1e30; return false; }}
    t = tt;
    normal = dot(n, rd) < 0.0 ? n : -n;
    return true;
}}"""

    def to_dict(self):
        return {"type": "mirror", "z_pos": self.z_pos,
                "r_max": self.r_max, "r_min": self.r_min,
                "tilt_x_deg": self.tilt_x_deg, "tilt_y_deg": self.tilt_y_deg}


# ─────────────────────────────────────────────────────────────────────────────
# Shutter plane  (toggleable opaque / transparent flat surface)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShutterPlane(ParametricSurface):
    """Flat plane that is either fully open (transparent) or fully closed
    (opaque/blocking).  When open the surface is invisible to rays.
    When closed all rays are absorbed (t=inf return).

    Useful for: mechanical shutters, beam blocks, aperture flags.
    """
    z_pos:   float = 0.0
    r_max:   float = 0.020
    r_min:   float = 0.0
    is_open: bool  = True

    glsl_type = "shutter"

    def intersect(self, ro, rd):
        if self.is_open:
            return _INF, np.zeros(3), np.zeros(3)
        return FlatSurface(z_pos=self.z_pos, r_max=self.r_max,
                           r_min=self.r_min).intersect(ro, rd)

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        if self.is_open:
            return f"""
bool {fn_name}(vec3 ro, vec3 rd, out float t, out vec3 normal) {{
    t = 1e30; return false;  // shutter open
}}"""
        return FlatSurface(z_pos=self.z_pos, r_max=self.r_max,
                           r_min=self.r_min).glsl_intercept_fn(fn_name)

    def to_dict(self):
        return {"type": "shutter", "z_pos": self.z_pos,
                "r_max": self.r_max, "r_min": self.r_min,
                "is_open": self.is_open}


# ─────────────────────────────────────────────────────────────────────────────
# Cylinder surface  (tube wall, Z-axis aligned)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CylinderSurface(ParametricSurface):
    """Infinite cylinder of radius *r* centred on the Z axis, clipped to
    [z_min, z_max].  Used as the wall of a ``TubeShell``.

    The normal points radially outward (outward from cylinder axis).
    """
    r:     float = 0.025    # cylinder radius
    z_min: float = -0.010  # start Z
    z_max: float =  0.010  # end Z

    glsl_type = "cylinder"

    def intersect(self, ro, rd):
        ro = np.asarray(ro, np.float64)
        rd = np.asarray(rd, np.float64)
        # Ray vs infinite cylinder x²+y² = r²
        A = rd[0]**2 + rd[1]**2
        if A < 1e-14:
            return _INF, np.zeros(3), np.zeros(3)
        B = 2.0*(ro[0]*rd[0] + ro[1]*rd[1])
        C = ro[0]**2 + ro[1]**2 - self.r**2
        disc = B*B - 4*A*C
        if disc < 0:
            return _INF, np.zeros(3), np.zeros(3)
        sq = math.sqrt(disc)
        t = None
        for tt in sorted([(-B-sq)/(2*A), (-B+sq)/(2*A)]):
            if tt <= 1e-9:
                continue
            hit = ro + tt * rd
            if self.z_min <= hit[2] <= self.z_max:
                t = tt; break
        if t is None:
            return _INF, np.zeros(3), np.zeros(3)
        hit = ro + t * rd
        n = np.array([hit[0]/self.r, hit[1]/self.r, 0.], np.float64)
        if np.dot(n, rd) > 0:
            n = -n
        return t, hit, n

    def glsl_intercept_fn(self, fn_name="surf_intercept") -> str:
        return f"""
bool {fn_name}(vec3 ro, vec3 rd, out float t, out vec3 normal) {{
    float A = rd.x*rd.x + rd.y*rd.y;
    if (A < 1e-14) {{ t = 1e30; return false; }}
    float B = 2.0*(ro.x*rd.x + ro.y*rd.y);
    float C = ro.x*ro.x + ro.y*ro.y - {self.r*self.r};
    float disc = B*B - 4.0*A*C;
    if (disc < 0.0) {{ t = 1e30; return false; }}
    float sq = sqrt(disc);
    float t1 = (-B-sq)/(2.0*A);
    float t2 = (-B+sq)/(2.0*A);
    t = 1e30;
    for (int i = 0; i < 2; i++) {{
        float tt = (i==0) ? t1 : t2;
        if (tt > 1e-6) {{
            vec3 h = ro + tt*rd;
            if (h.z >= {self.z_min} && h.z <= {self.z_max}) {{
                t = tt; break;
            }}
        }}
    }}
    if (t >= 1e29) return false;
    vec3 h = ro + t*rd;
    normal = normalize(vec3(h.x/{self.r}, h.y/{self.r}, 0.0));
    if (dot(normal, rd) > 0.0) normal = -normal;
    return true;
}}"""

    def to_dict(self):
        return {"type": "cylinder", "r": self.r,
                "z_min": self.z_min, "z_max": self.z_max}


# Register new types
_REGISTRY["mirror"]   = MirrorSurface
_REGISTRY["shutter"]  = ShutterPlane
_REGISTRY["cylinder"] = CylinderSurface


# ─────────────────────────────────────────────────────────────────────────────
# Piecewise Catmull-Rom n-D mesh surface
# ─────────────────────────────────────────────────────────────────────────────

class CatmullRomMesh(ParametricSurface):
    """Piecewise bicubic Catmull-Rom surface over an (nu × nv) control grid
    in **n** dimensions.

    Dimensions 0..2 are always xyz (used for ray intersection and mesh export).
    Higher dimensions are carried through interpolation without interpretation —
    useful for spatially-varying optical properties (refractive index field,
    Sellmeier coefficients, absorption, reflectance tint, etc.).

    Parameterisation
    ----------------
    (u, v) ∈ [0, 1]²  maps onto a grid of (nu-1) × (nv-1) bicubic patches.
    Each patch uses the four surrounding rows/columns of control points with
    Catmull-Rom basis (C1 at every shared edge, standard α=0.5 uniform knots).

    Parameters
    ----------
    control_pts : array-like, shape (nu, nv, ndim)
        Grid of n-D control points.  Minimum 2×2; interior points with 1-cell
        boundary are required for smooth tangents at the edges (ghost rows are
        clamped-repeat by default).
    r_max : float
        Maximum radial distance from the Z axis used as a fast reject guard
        in ``intersect``.  Set to a value slightly larger than the physical
        aperture.
    r_min : float
        Inner hole radius (0 = solid).
    clamp_edges : bool
        If True (default), edge tangents are computed by clamping the phantom
        control point to the nearest real point (clamp-repeat boundary).
        If False, phantom points are reflected (mirrored boundary).
    """

    glsl_type = "catmull_rom_mesh"

    def __init__(
        self,
        control_pts,                # (nu, nv, ndim) array-like
        r_max: float = 0.025,
        r_min: float = 0.0,
        clamp_edges: bool = True,
    ):
        pts = np.asarray(control_pts, dtype=np.float64)
        if pts.ndim != 3 or pts.shape[2] < 3:
            raise ValueError(
                "control_pts must be shape (nu, nv, ndim) with ndim >= 3"
            )
        self.control_pts = pts
        self.r_max       = r_max
        self.r_min       = r_min
        self.clamp_edges = clamp_edges

    # ── Shape helpers ─────────────────────────────────────────────────────────

    @property
    def nu(self) -> int:
        return self.control_pts.shape[0]

    @property
    def nv(self) -> int:
        return self.control_pts.shape[1]

    @property
    def ndim(self) -> int:
        return self.control_pts.shape[2]

    # ── Catmull-Rom basis (uniform α=0.5) ─────────────────────────────────────

    @staticmethod
    def _cr_weights(t: float) -> np.ndarray:
        """Return the 4 Catmull-Rom basis weights for local parameter t ∈ [0,1].

        The weights apply to points [p_{i-1}, p_i, p_{i+1}, p_{i+2}] where
        t=0 → p_i, t=1 → p_{i+1}.
        """
        t2 = t * t
        t3 = t2 * t
        return 0.5 * np.array([
            -t3 + 2.0*t2 - t,          # w0
             3.0*t3 - 5.0*t2 + 2.0,    # w1
            -3.0*t3 + 4.0*t2 + t,      # w2
             t3 - t2,                   # w3
        ], dtype=np.float64)

    @staticmethod
    def _cr_dweights(t: float) -> np.ndarray:
        """Derivative of Catmull-Rom weights w.r.t. local parameter t."""
        t2 = t * t
        return 0.5 * np.array([
            -3.0*t2 + 4.0*t - 1.0,    # dw0
             9.0*t2 - 10.0*t,          # dw1
            -9.0*t2 +  8.0*t + 1.0,   # dw2
             3.0*t2 -  2.0*t,          # dw3
        ], dtype=np.float64)

    # ── Index helpers with boundary condition ─────────────────────────────────

    def _idx(self, i: int, j: int) -> np.ndarray:
        """Return control_pts[i, j] with clamped or mirrored boundary."""
        if self.clamp_edges:
            i = max(0, min(self.nu - 1, i))
            j = max(0, min(self.nv - 1, j))
        else:
            # Mirror: reflect outside boundary
            i = i if 0 <= i < self.nu else (-1 - i if i < 0 else 2*(self.nu-1) - i)
            j = j if 0 <= j < self.nv else (-1 - j if j < 0 else 2*(self.nv-1) - j)
            i = max(0, min(self.nu - 1, i))
            j = max(0, min(self.nv - 1, j))
        return self.control_pts[i, j]

    def _patch(self, iu: int, iv: int) -> np.ndarray:
        """Return the 4×4×ndim control-point sub-grid for patch (iu, iv).

        ``iu`` ∈ [0, nu-2], ``iv`` ∈ [0, nv-2].
        The patch spans control rows [iu-1 .. iu+2] and cols [iv-1 .. iv+2]
        with boundary conditions applied by ``_idx``.
        """
        return np.array(
            [[self._idx(iu + di - 1, iv + dj - 1) for dj in range(4)]
             for di in range(4)],
            dtype=np.float64,
        )  # shape (4, 4, ndim)

    # ── Core evaluation ───────────────────────────────────────────────────────

    def _uv_to_patch(self, u: float, v: float):
        """Decompose global (u,v) into patch index (iu,iv) and local (lu,lv)."""
        nu_p = max(1, self.nu - 1)
        nv_p = max(1, self.nv - 1)
        ui   = u * nu_p
        vi   = v * nv_p
        iu   = int(math.floor(ui))
        iv   = int(math.floor(vi))
        iu   = max(0, min(nu_p - 1, iu))
        iv   = max(0, min(nv_p - 1, iv))
        lu   = ui - iu   # ∈ [0, 1)
        lv   = vi - iv
        return iu, iv, lu, lv, nu_p, nv_p

    def eval(self, u: float, v: float) -> np.ndarray:
        """Evaluate the n-D surface at (u, v) ∈ [0, 1]².

        Returns an ndim-D point computed by bicubic Catmull-Rom interpolation.
        """
        iu, iv, lu, lv, _, _ = self._uv_to_patch(u, v)
        patch = self._patch(iu, iv)                    # (4, 4, ndim)
        wu    = self._cr_weights(lu)                    # (4,)
        wv    = self._cr_weights(lv)                    # (4,)
        # Interpolate along u (axis 0) then v (axis 0 of result)
        u_interp = np.tensordot(wu, patch, axes=([0], [0]))  # (4, ndim)
        return np.tensordot(wv, u_interp, axes=([0], [0]))   # (ndim,)

    def eval_grad(self, u: float, v: float):
        """Evaluate surface point and partial derivatives.

        Returns
        -------
        point  : (ndim,) — surface value
        dp_du  : (ndim,) — ∂/∂u (scaled to global [0,1] u)
        dp_dv  : (ndim,) — ∂/∂v (scaled to global [0,1] v)
        """
        iu, iv, lu, lv, nu_p, nv_p = self._uv_to_patch(u, v)
        patch = self._patch(iu, iv)                     # (4, 4, ndim)
        wu    = self._cr_weights(lu)
        wv    = self._cr_weights(lv)
        dwu   = self._cr_dweights(lu) * nu_p            # chain rule ∂/∂u_global
        dwv   = self._cr_dweights(lv) * nv_p

        u_interp  = np.tensordot(wu,  patch, axes=([0], [0]))   # (4, ndim)
        du_interp = np.tensordot(dwu, patch, axes=([0], [0]))   # (4, ndim)

        point  = np.tensordot(wv,  u_interp,  axes=([0], [0]))  # (ndim,)
        dp_du  = np.tensordot(wv,  du_interp, axes=([0], [0]))  # (ndim,)
        dp_dv  = np.tensordot(dwv, u_interp,  axes=([0], [0]))  # (ndim,)
        return point, dp_du, dp_dv

    # ── Ray intersection ──────────────────────────────────────────────────────

    def intersect(self, ro, rd):
        """Ray–surface intersection via Newton iteration in (u, v, t) space.

        Solves: xyz(u,v) − (ro + t·rd) = 0  for (u, v, t).

        Only dims 0..2 of the control grid are used here.
        Returns (t, hit_pt, normal) in surface-local float64; t=inf on miss.
        """
        ro = np.asarray(ro, np.float64)
        rd = np.asarray(rd, np.float64)

        # ── Grid search for a starting (u, v, t) ──
        # Sample a coarse grid and pick the closest guess.
        N_COARSE = 8
        best_dist = math.inf
        best_uvt  = None
        for gi in range(N_COARSE + 1):
            for gj in range(N_COARSE + 1):
                ug = gi / N_COARSE
                vg = gj / N_COARSE
                xyz = self.eval(ug, vg)[:3]
                # Project point onto ray: t = (xyz - ro)·rd
                tt = float(np.dot(xyz - ro, rd))
                if tt < 1e-9:
                    continue
                proj = ro + tt * rd
                d    = float(np.linalg.norm(xyz - proj))
                if d < best_dist:
                    best_dist = d
                    best_uvt  = (ug, vg, tt)

        if best_uvt is None:
            return _INF, np.zeros(3), np.zeros(3)

        u, v, t = best_uvt

        # ── Newton refinement in (u, v, t) ──
        # Residual F(u,v,t) = xyz(u,v) - (ro + t*rd)
        for _ in range(32):
            pt, dpu, dpv = self.eval_grad(u, v)
            xyz  = pt[:3]
            xu   = dpu[:3]
            xv   = dpv[:3]
            F    = xyz - (ro + t * rd)         # (3,)
            # Jacobian cols: [xu, xv, -rd]  → J @ [Δu, Δv, Δt]^T = -F
            J    = np.column_stack([xu, xv, -rd])  # (3, 3)
            det  = np.linalg.det(J)
            if abs(det) < 1e-30:
                break
            delta = np.linalg.solve(J, -F)
            u += delta[0]
            v += delta[1]
            t += delta[2]
            # Clamp (u, v) to [0, 1]
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            if np.linalg.norm(delta) < 1e-11:
                break

        if t < 1e-9:
            return _INF, np.zeros(3), np.zeros(3)

        hit = ro + t * rd
        r2  = hit[0]**2 + hit[1]**2
        if r2 > self.r_max**2 or r2 < self.r_min**2:
            return _INF, np.zeros(3), np.zeros(3)

        # Surface normal from cross product of partial derivatives
        _, dpu, dpv = self.eval_grad(u, v)
        n = np.cross(dpu[:3], dpv[:3])
        nm = float(np.linalg.norm(n))
        if nm < 1e-30:
            return _INF, np.zeros(3), np.zeros(3)
        n /= nm
        if np.dot(n, rd) > 0:
            n = -n
        return t, hit, n

    # ── Mesh export ───────────────────────────────────────────────────────────

    def to_mesh(
        self,
        nu_out: int = 32,
        nv_out: int = 32,
    ):
        """Sample the surface into a triangulated mesh.

        Parameters
        ----------
        nu_out, nv_out : int
            Number of sample divisions in each parameter direction.

        Returns
        -------
        verts : (nv+1, nu+1, ndim) float64
            Dense sample grid (all n dimensions preserved).
        xyz   : ((nv+1)*(nu+1), 3) float64
            Flattened 3-D positions for the triangle vertices.
        faces : (2*nu_out*nv_out, 3) int32
            Triangle face indices into xyz.
        """
        us = np.linspace(0.0, 1.0, nu_out + 1)
        vs = np.linspace(0.0, 1.0, nv_out + 1)

        verts = np.empty((nv_out + 1, nu_out + 1, self.ndim), dtype=np.float64)
        for vi, vv in enumerate(vs):
            for ui, uu in enumerate(us):
                verts[vi, ui] = self.eval(uu, vv)

        # Flatten to vertex list (row-major: row=v, col=u)
        xyz = verts.reshape(-1, self.ndim)[:, :3].copy()

        # Build quads → two triangles each
        faces = []
        stride = nu_out + 1
        for vi in range(nv_out):
            for ui in range(nu_out):
                a = vi * stride + ui
                b = a + 1
                c = (vi + 1) * stride + ui
                d = c + 1
                faces.append((a, b, d))
                faces.append((a, d, c))

        return verts, xyz, np.array(faces, dtype=np.int32)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_surface(
        cls,
        surface: "ParametricSurface",
        nu: int = 8,
        nv: int = 8,
        r_sample: float = 0.020,
        extra_dims: Optional[np.ndarray] = None,
    ) -> "CatmullRomMesh":
        """Build a Catmull-Rom mesh by dense-sampling any ParametricSurface.

        The ray fan samples the surface on a polar (r, φ) grid projected onto
        the local XY plane and mapped to (u, v) ∈ [0,1]².

        Parameters
        ----------
        surface     : any ParametricSurface — must implement ``intersect``
        nu, nv      : control-grid resolution (≥ 2)
        r_sample    : radial extent of the sampling aperture (metres)
        extra_dims  : optional (nu, nv, k) float64 extra dimension values to
                      append after xyz.  If None, only xyz are stored.
        """
        control_pts_xyz = np.empty((nu, nv, 3), dtype=np.float64)
        ro_base = np.array([0., 0., 1.0], np.float64)  # ray origin above

        for i in range(nu):
            u = i / max(nu - 1, 1)       # radial [0,1]
            r = u * r_sample
            for j in range(nv):
                phi = j / max(nv - 1, 1) * 2.0 * math.pi
                x0  = r * math.cos(phi)
                y0  = r * math.sin(phi)
                ro  = np.array([x0, y0, ro_base[2]], np.float64)
                rd  = np.array([0., 0., -1.],        np.float64)
                t, hit, _ = surface.intersect(ro, rd)
                if math.isfinite(t):
                    control_pts_xyz[i, j] = hit
                else:
                    # Miss: use flat fallback
                    control_pts_xyz[i, j] = np.array([x0, y0, 0.], np.float64)

        if extra_dims is not None:
            extra = np.asarray(extra_dims, dtype=np.float64)
            if extra.shape[:2] != (nu, nv):
                raise ValueError(
                    f"extra_dims shape {extra.shape} must start with ({nu}, {nv})"
                )
            control_pts = np.concatenate([control_pts_xyz, extra], axis=2)
        else:
            control_pts = control_pts_xyz

        r_max = getattr(surface, "r_max", r_sample)
        r_min = getattr(surface, "r_min", 0.0)
        return cls(control_pts=control_pts, r_max=r_max, r_min=r_min)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def glsl_intercept_fn(self, fn_name: str = "surf_intercept") -> str:
        # A GPU Catmull-Rom mesh evaluator would require uploading the control
        # grid as a texture / SSBO and is left for the bake pipeline.  Return
        # a stub that always misses — the CPU intersect path is used instead.
        return f"""
bool {fn_name}(vec3 ro, vec3 rd, out float t, out vec3 normal) {{
    // CatmullRomMesh: GPU intercept not yet implemented; use CPU path.
    t = 1e30; normal = vec3(0.0); return false;
}}"""

    def to_dict(self) -> dict:
        return {
            "type":        "catmull_rom_mesh",
            "control_pts": self.control_pts.tolist(),
            "r_max":       self.r_max,
            "r_min":       self.r_min,
            "clamp_edges": self.clamp_edges,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CatmullRomMesh":
        return cls(
            control_pts = np.array(d["control_pts"], dtype=np.float64),
            r_max       = float(d.get("r_max", 0.025)),
            r_min       = float(d.get("r_min", 0.0)),
            clamp_edges = bool(d.get("clamp_edges", True)),
        )


_REGISTRY["catmull_rom_mesh"] = CatmullRomMesh
