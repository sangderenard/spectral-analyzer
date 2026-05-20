"""camera_designer/bake_worker.py
==================================

DEPRECATION NOTICE
------------------
This module is a transitional utility.  The CPU Snell ray tracer (trace_ray,
trace_ray_backward, trace_ray_batch, BakeWorker) duplicates T1→T2→T3 without
wave simulation and must NOT be used as the primary pipeline.

Intended final state:
  - trace_ray / trace_ray_backward / trace_ray_batch  →  replace with GPU dispatch
    (submit T1→T2→T3 and read back the SSBO output hit records)
  - BakeWorker.bake() / bake_assembly() / _trace_batch()  →  deprecated; use
    LensAssemblySpec.bake_lut() which wraps ManifoldEndpoint (same CPU path today,
    but will call the GPU pipeline once the read-back API is available)
  - bake_neural_training_data()  →  TODO: replace with GPU dispatch + wave sim
  - bake_glsl_source()  →  potentially useful for parametric equation generation;
    kept pending evaluation

64-bit parametric ray tracer that bakes a CameraPreset into a LensManifold
noodle LUT.

Architecture
------------
All ray tracing uses the ParametricSurface CPU intersect() path — float64,
no meshes, exact conic/polynomial geometry.  Snell's law is applied at each
surface interface using the GlassSpec.n_at(wavelength) Sellmeier formula.

The aperture plane (aperture_stop) is the manifold coordinate origin.  Rays
that pass through the aperture are parameterised by their normalised aperture
hit position (u,v) ∈ [-1,1]² and their scene-side incident direction
(in_dx, in_dy, in_dz).  The LUT output is the sensor-side exit direction
(out_dx, out_dy, out_dz) and the optical path length (OPL).

Standard noodle schema (camera_software/lens_manifold.py compatible, 11 cols):
  col 0,1  : u, v         aperture normalised hit position
  col 2,3  : fu, fv       scene field-angle factors (tan of angular deviation)
  col 4,5,6: in_dir       unit incident ray direction (scene → aperture)
  col 7,8,9: out_dir      unit exit ray direction (aperture → sensor)
  col 10   : opl          optical path length (metres)

Full-assembly noodle schema (14 cols, produced by bake_assembly()):
  col 0-10 : same as above
  col 11,12: sensor_x, sensor_y   pre-baked sensor hit (metres)
  col 13   : focus_z              sensor z offset from nominal (metres)

Usage
-----
    from camera_designer import CameraPreset, BakeWorker

    preset = CameraPreset.load("my_lens.camera.json")
    worker = BakeWorker(preset, n_rays=65536, n_wavelengths=3)
    manifold = worker.bake()          # returns LensManifold
    manifold.save("my_lens.npz")
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .camera_preset import CameraPreset
from .parametric_surfaces import ParametricSurface, ApertureStop

__all__ = ["BakeWorker", "trace_ray", "trace_ray_backward"]

_INF = np.inf


# ─────────────────────────────────────────────────────────────────────────────
# Snell's law in vector form (64-bit)
# ─────────────────────────────────────────────────────────────────────────────

def _snell(rd: np.ndarray, n: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
    """Refract *rd* (unit, pointing into surface) through normal *n* (unit, facing ray).

    Returns the refracted direction or None on total internal reflection.
    ``n`` must face the incoming ray (dot(rd, n) < 0).
    """
    n = np.asarray(n, np.float64)
    rd = np.asarray(rd, np.float64)
    cos_i = -np.dot(rd, n)
    ratio = n1 / n2
    sin2_t = ratio**2 * (1.0 - cos_i**2)
    if sin2_t > 1.0:
        return None  # TIR
    cos_t = math.sqrt(max(0., 1.0 - sin2_t))
    return ratio * rd + (ratio * cos_i - cos_t) * n


# ─────────────────────────────────────────────────────────────────────────────
# Single-ray tracer
# ─────────────────────────────────────────────────────────────────────────────

def trace_ray(
    preset: CameraPreset,
    ro: np.ndarray,
    rd: np.ndarray,
    wavelength_um: float = 0.587,
) -> Optional[dict]:
    """Trace one ray through the full lens group.

    TRANSITIONAL: replace with GPU dispatch (T1→T2→T3 SSBO read-back).

    Parameters
    ----------
    preset        : fully specified CameraPreset
    ro            : ray origin in camera space (float64, metres)
    rd            : unit ray direction (float64, scene → lens)
    wavelength_um : wavelength in micrometres for dispersion

    Returns
    -------
    dict with keys:
      'aperture_uv'   : (2,) normalised aperture hit  [-1,1]
      'aperture_hit'  : (3,) world-space aperture point
      'in_dir'        : (3,) incident direction at aperture (= rd)
      'out_dir'       : (3,) exit direction at sensor
      'opl'           : float  optical path length metres
      'sensor_hit'    : (3,) sensor hit point
    or None if the ray misses, hits the mount, or undergoes TIR.
    """
    ro = np.asarray(ro, np.float64)
    rd = np.asarray(rd, np.float64)
    rd = rd / max(np.linalg.norm(rd), 1e-30)

    opl         = 0.0
    n_current   = 1.0       # index of medium the ray is currently in
    ap_hit      = None      # aperture plane hit point (set on first ap hit)
    in_dir_ap   = None

    # Check aperture stop first — it gates the manifold key
    ap_surf = preset.aperture_stop

    # ── Propagate through lens elements (front-to-back) ────────────────────
    elements = sorted(preset.lens_group.elements,
                      key=lambda e: e.z_vertex, reverse=True)  # front = largest z

    for el in elements:
        # Transform ray into surface-local frame (vertex at z=0 local)
        ro_local = ro.copy()
        ro_local[2] -= el.z_vertex
        t, hit_local, normal = el.surface.intersect(ro_local, rd)
        if not math.isfinite(t):
            return None  # missed this element — ray lost
        hit_world = hit_local.copy()
        hit_world[2] += el.z_vertex

        # Accumulate OPL: n * distance
        opl += n_current * t

        # Refract at the surface
        n_out = el.glass_out.n_at(wavelength_um)
        rd_new = _snell(rd, normal, n_current, n_out)
        if rd_new is None:
            return None  # TIR
        rd       = rd_new / max(np.linalg.norm(rd_new), 1e-30)
        n_current = n_out
        ro        = hit_world

        # Check if we just passed through the aperture plane
        if ap_hit is None:
            z_ap = ap_surf.z_pos
            if (ro[2] - z_ap) * (hit_world[2] - z_ap) <= 0:
                # Interpolate aperture crossing
                dz = rd[2]
                if abs(dz) > 1e-9:
                    t_ap = (z_ap - ro[2]) / dz
                    pt_ap = ro + t_ap * rd
                    r_ap  = math.sqrt(pt_ap[0]**2 + pt_ap[1]**2)
                    if (ap_surf.r_inner <= r_ap <= ap_surf.r_outer):
                        ap_hit   = pt_ap
                        in_dir_ap = rd.copy()

    # ── Hit the sensor ────────────────────────────────────────────────────
    ro_local = ro.copy()
    ro_local[2] -= preset.sensor.z_pos
    t_s, hit_s_local, _ = preset.sensor.intersect(ro_local, rd)
    if not math.isfinite(t_s):
        return None  # missed sensor
    hit_sensor = hit_s_local.copy()
    hit_sensor[2] += preset.sensor.z_pos
    opl += n_current * t_s

    # ── Aperture UV ────────────────────────────────────────────────────────
    if ap_hit is None:
        # Fallback: intersect the aperture plane directly with incoming ray
        if abs(rd[2]) > 1e-9:
            t_ap = (ap_surf.z_pos - ro[2]) / rd[2]
            ap_hit = ro + t_ap * rd
            in_dir_ap = rd.copy()
        else:
            return None

    r_max  = ap_surf.r_outer
    u = ap_hit[0] / max(r_max, 1e-12)
    v = ap_hit[1] / max(r_max, 1e-12)
    if abs(u) > 1.0 or abs(v) > 1.0:
        return None  # outside aperture

    # fu, fv — field-angle factors (tangent of angle in scene cone)
    safe_z = rd[2] if abs(rd[2]) > 1e-9 else 1e-9
    fu = rd[0] / safe_z
    fv = rd[1] / safe_z

    return {
        "aperture_uv":  np.array([u, v],         np.float64),
        "field_angle":  np.array([fu, fv],        np.float64),
        "aperture_hit": ap_hit,
        "in_dir":       in_dir_ap / max(np.linalg.norm(in_dir_ap), 1e-30),
        "out_dir":      rd / max(np.linalg.norm(rd), 1e-30),
        "opl":          opl,
        "sensor_hit":   hit_sensor,
    }


def trace_ray_backward(
    preset: CameraPreset,
    ro: np.ndarray,
    rd: np.ndarray,
    wavelength_um: float = 0.587,
) -> Optional[dict]:
    """Trace a ray from sensor side backward through the lens to scene.

    TRANSITIONAL: replace with GPU dispatch (T1→T2→T3 SSBO read-back).

    Applies Snell's law with swapped n1/n2 through elements in reverse order
    (sensor-side first), giving the time-reversed optical path.

    Parameters
    ----------
    ro : (3,) array — ray origin near the sensor plane (camera local space)
    rd : (3,) unit direction pointing toward the lens / scene (+z nominally)

    Returns
    -------
    dict with same keys as trace_ray, or None on miss or TIR.
      'aperture_hit' : (3,) world-space aperture crossing point
      'in_dir'       : (3,) ray direction AT the aperture crossing
      'out_dir'      : (3,) final scene-side direction after exiting the lens
      'aperture_uv'  : (2,) normalised aperture UV [-1, 1]
      'field_angle'  : (2,) (fu, fv) tangent-field factors
      'opl'          : float  optical path length (metres)
    """
    ro = np.asarray(ro, np.float64)
    rd = np.asarray(rd, np.float64)
    rd = rd / max(np.linalg.norm(rd), 1e-30)

    opl      = 0.0
    ap_surf  = preset.aperture_stop
    z_ap     = float(ap_surf.z_pos)
    ap_hit   = None
    ap_dir   = None

    # Build forward n-sequence then reverse it for backward traversal.
    elements_fwd = sorted(preset.lens_group.elements,
                          key=lambda e: e.z_vertex, reverse=True)
    n_seq = [1.0] + [el.glass_out.n_at(wavelength_um) for el in elements_fwd]
    N = len(elements_fwd)
    elements_bwd = list(reversed(elements_fwd))

    # Start in sensor-side air (n_seq[N] should be 1.0).
    n_current = n_seq[N]
    prev_ro   = ro.copy()

    for bwd_k, el in enumerate(elements_bwd):
        fwd_k  = N - 1 - bwd_k
        n_exit = n_seq[fwd_k]

        ro_local        = ro.copy()
        ro_local[2]    -= el.z_vertex
        t, hit_local, normal = el.surface.intersect(ro_local, rd)
        if not math.isfinite(t):
            return None

        hit_world        = hit_local.copy()
        hit_world[2]    += el.z_vertex

        # Detect aperture crossing on this ray segment (prev_ro → hit_world).
        if ap_hit is None:
            dz_seg = hit_world[2] - prev_ro[2]
            if abs(dz_seg) > 1e-12:
                s = (z_ap - prev_ro[2]) / dz_seg
                if 0.0 < s < 1.0:
                    ap_pt = prev_ro + s * (hit_world - prev_ro)
                    r_ap  = math.sqrt(ap_pt[0] ** 2 + ap_pt[1] ** 2)
                    if ap_surf.r_inner <= r_ap <= ap_surf.r_outer:
                        ap_hit = ap_pt
                        ap_dir = rd.copy()

        opl += n_current * t

        # Snell's law — ensure normal faces the incoming ray.
        if np.dot(rd, normal) > 0:
            normal = -normal
        rd_new = _snell(rd, normal, n_current, n_exit)
        if rd_new is None:
            return None
        rd        = rd_new / max(np.linalg.norm(rd_new), 1e-30)
        n_current = n_exit
        prev_ro   = hit_world
        ro        = hit_world

    # Fallback: extrapolate scene-side ray back to aperture plane.
    if ap_hit is None:
        if abs(rd[2]) > 1e-9:
            t_ap  = (z_ap - ro[2]) / rd[2]   # negative (aperture is behind scene pos)
            ap_pt = ro + t_ap * rd
            r_ap  = math.sqrt(ap_pt[0] ** 2 + ap_pt[1] ** 2)
            if r_ap <= ap_surf.r_outer:
                ap_hit = ap_pt
                ap_dir = rd.copy()
        if ap_hit is None:
            return None

    r_max = ap_surf.r_outer
    u = ap_hit[0] / max(r_max, 1e-12)
    v = ap_hit[1] / max(r_max, 1e-12)
    if abs(u) > 1.0 or abs(v) > 1.0:
        return None

    safe_z = rd[2] if abs(rd[2]) > 1e-9 else 1e-9
    return {
        "aperture_uv":  np.array([u, v],       np.float64),
        "field_angle":  np.array([rd[0] / safe_z, rd[1] / safe_z], np.float64),
        "aperture_hit": ap_hit,
        "in_dir":       ap_dir / max(np.linalg.norm(ap_dir), 1e-30),
        "out_dir":      rd    / max(np.linalg.norm(rd),      1e-30),
        "opl":          opl,
        "sensor_hit":   prev_ro,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised batch ray tracer
# ─────────────────────────────────────────────────────────────────────────────

def _snell_batch(
    rd:     np.ndarray,   # (N,3) unit incident directions
    normal: np.ndarray,   # (N,3) unit surface normals (oriented against ray)
    n1:     np.ndarray,   # (N,) or scalar — medium before
    n2:     np.ndarray,   # (N,) or scalar — medium after
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Snell's law.  Returns (rd_out (N,3), tir_mask (N,) bool)."""
    n1 = np.asarray(n1, np.float64)
    n2 = np.asarray(n2, np.float64)
    ratio  = n1 / np.maximum(n2, 1e-30)
    cos_i  = -np.einsum("ij,ij->i", normal, rd)          # (N,)  >0 for front face
    sin2_t = ratio ** 2 * (1.0 - cos_i ** 2)
    tir    = sin2_t > 1.0
    cos_t  = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))
    if np.ndim(ratio) == 0:
        rd_out = ratio * rd + (ratio * cos_i - cos_t)[:, None] * normal
    else:
        rd_out = ratio[:, None] * rd + (ratio * cos_i - cos_t)[:, None] * normal
    norms  = np.linalg.norm(rd_out, axis=1, keepdims=True)
    rd_out = rd_out / np.maximum(norms, 1e-30)
    return rd_out, tir


def trace_ray_batch(
    preset:        "CameraPreset",
    ro_arr:        np.ndarray,   # (N,3) float64 — ray origins at entrance plane
    rd_arr:        np.ndarray,   # (N,3) float64 — unit ray directions
    wavelength_um: float,
) -> dict:
    """Batch-vectorised analogue of trace_ray().

    TRANSITIONAL: replace with GPU dispatch (T1→T2→T3 SSBO read-back).

    Propagates N rays simultaneously through every lens element using
    NumPy array operations instead of a per-ray Python loop.  Each element
    step is O(N) NumPy; the sequential element loop (M elements deep) is
    unavoidable.

    Returns a dict with arrays of shape (N,) / (N,3):
        alive      (N,) bool  — False for TIR / miss / aperture-blocked
        sensor_hit (N,3)      — world-space sensor intersection
        out_dir    (N,3)      — unit exit direction
        opl        (N,)       — optical path length entrance→sensor (m)
        ap_uv      (N,2)      — normalised aperture UV [-1,1]
    Dead-ray entries are undefined (use alive mask to filter).
    """
    N   = len(ro_arr)
    ro  = np.asarray(ro_arr, np.float64).copy()
    rd  = np.asarray(rd_arr, np.float64).copy()
    rd  = rd / np.maximum(np.linalg.norm(rd, axis=1, keepdims=True), 1e-30)

    alive   = np.ones(N, bool)
    opl     = np.zeros(N, np.float64)
    n_cur   = np.ones(N, np.float64)  # current medium refractive index

    ap      = preset.aperture_stop
    z_ap    = float(ap.z_pos)
    r_outer = float(ap.r_outer)
    r_inner = float(getattr(ap, "r_inner", 0.0))

    # Track first aperture-plane crossing
    ap_hit_xy = np.zeros((N, 2), np.float64)
    ap_found  = np.zeros(N, bool)

    elements = sorted(preset.lens_group.elements,
                      key=lambda e: e.z_vertex, reverse=True)

    for el in elements:
        if not np.any(alive):
            break
        ro_local      = ro.copy()
        ro_local[:, 2] -= el.z_vertex

        t_el, hit_local, nrm = el.surface.intersect_batch(ro_local, rd)

        # Rays that miss this element die
        alive &= np.isfinite(t_el)

        hit_world      = hit_local.copy()
        hit_world[:, 2] += el.z_vertex

        # OPL accumulation
        opl = np.where(alive, opl + n_cur * t_el, opl)

        # Refraction
        n_out_v = float(el.glass_out.n_at(wavelength_um))
        n_out   = np.full(N, n_out_v, np.float64)
        rd_new, tir = _snell_batch(rd, nrm, n_cur, n_out)
        alive   &= ~tir

        # Advance state for surviving rays
        ro    = np.where(alive[:, None], hit_world, ro)
        rd    = np.where(alive[:, None], rd_new, rd)
        n_cur = np.where(alive, n_out, n_cur)

        # Capture aperture crossing (first time ray passes z_ap)
        unset = alive & ~ap_found
        if np.any(unset):
            # Check whether this step crossed z_ap
            prev_z = ro[:, 2] - t_el * rd[:, 2]   # approx pre-step z
            crossed = unset & (
                ((prev_z - z_ap) * (hit_world[:, 2] - z_ap) <= 0) |
                (np.abs(hit_world[:, 2] - z_ap) < 1e-6)
            )
            if np.any(crossed):
                dz_safe = np.where(np.abs(rd[:, 2]) > 1e-9, rd[:, 2], 1.0)
                t_ap    = (z_ap - ro[:, 2]) / dz_safe
                ap_pt   = ro + t_ap[:, None] * rd
                ap_hit_xy = np.where(
                    crossed[:, None],
                    ap_pt[:, :2],
                    ap_hit_xy,
                )
                ap_found |= crossed

    # Rays that never found an aperture crossing: project from current pos
    no_ap = alive & ~ap_found
    if np.any(no_ap):
        dz_safe = np.where(np.abs(rd[:, 2]) > 1e-9, rd[:, 2], 1.0)
        t_ap    = (z_ap - ro[:, 2]) / dz_safe
        ap_pt   = ro + t_ap[:, None] * rd
        ap_hit_xy = np.where(no_ap[:, None], ap_pt[:, :2], ap_hit_xy)
        ap_found |= no_ap

    # Aperture radius check — kill rays outside aperture disk
    ap_r2 = ap_hit_xy[:, 0] ** 2 + ap_hit_xy[:, 1] ** 2
    alive &= (ap_r2 <= r_outer ** 2) & (ap_r2 >= r_inner ** 2)

    # Propagate to sensor
    sensor_z = float(preset.sensor.z_pos)
    ro_sensor = ro.copy()
    ro_sensor[:, 2] -= sensor_z
    t_s, hit_s_local, _ = preset.sensor.intersect_batch(ro_sensor, rd)
    alive &= np.isfinite(t_s)
    hit_sensor = hit_s_local.copy()
    hit_sensor[:, 2] += sensor_z
    opl = np.where(alive, opl + n_cur * t_s, opl)

    ap_uv = ap_hit_xy / max(r_outer, 1e-12)

    return {
        "alive":      alive,
        "sensor_hit": hit_sensor,
        "out_dir":    rd / np.maximum(np.linalg.norm(rd, axis=1, keepdims=True), 1e-30),
        "opl":        opl,
        "ap_uv":      ap_uv,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BakeWorker
# ─────────────────────────────────────────────────────────────────────────────

class BakeWorker:
    """Bake a CameraPreset into a LensManifold noodle LUT.

    Parameters
    ----------
    preset        : CameraPreset defining the full optical system
    n_rays        : base number of rays to trace (before adaptive refinement)
    n_wavelengths : number of wavelength samples (uses preset.wavelengths list)
    n_refine      : adaptive refinement passes
    threshold     : out-direction variance threshold to trigger refinement
    seed          : RNG seed for reproducibility
    verbose       : print progress to stdout
    """

    def __init__(
        self,
        preset:        CameraPreset,
        n_rays:        int   = 2_000_000,
        n_wavelengths: int   = 3,
        n_refine:      int   = 2,
        threshold:     float = 1e-4,
        seed:          int   = 42,
        verbose:       bool  = True,
    ) -> None:
        self.preset        = preset
        self.n_rays        = n_rays
        self.n_wavelengths = n_wavelengths
        self.n_refine      = n_refine
        self.threshold     = threshold
        self.rng           = np.random.default_rng(seed)
        self.verbose       = verbose

    # ── Internal helpers ───────────────────────────────────────────────────

    def _sample_rays(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample n (ro, rd) pairs: origins on the aperture disk, directions
        from a Fibonacci hemisphere covering all incoming angles (full θ ∈ [0, π/2]).

        Returns (ro_arr, rd_arr) each (n, 3) float64.
        """
        ap  = self.preset.aperture_stop
        PHI = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio

        # Aperture disk: uniform area sampling
        r2  = self.rng.uniform(0., ap.r_outer**2, n)
        ang = self.rng.uniform(0., 2*math.pi, n)
        r   = np.sqrt(r2)
        ax  = r * np.cos(ang)
        ay  = r * np.sin(ang)
        z_front = max(el.z_vertex
                      for el in self.preset.lens_group.elements) + 0.001
        az  = np.full(n, z_front)
        ro_arr = np.stack([ax, ay, az], axis=1)  # (n, 3)

        # Fibonacci hemisphere — uniform area coverage, θ ∈ [0, π/2]
        # rd_z < 0 = pointing into the lens (−Z camera direction)
        i_arr  = np.arange(n, dtype=np.float64)
        cos_th = 1.0 - (i_arr + 0.5) / n
        sin_th = np.sqrt(np.maximum(1.0 - cos_th * cos_th, 0.0))
        phi    = 2.0 * math.pi * i_arr / PHI + self.rng.uniform(0., 2*math.pi)
        rd_arr = np.stack([
            sin_th * np.cos(phi),
            sin_th * np.sin(phi),
            -cos_th,
        ], axis=1)
        norms  = np.linalg.norm(rd_arr, axis=1, keepdims=True)
        rd_arr = rd_arr / np.maximum(norms, 1e-30)
        return ro_arr, rd_arr

    def _trace_batch(
        self,
        ro_arr: np.ndarray,
        rd_arr: np.ndarray,
        focus_z: float = 0.0,
        preset_override=None,
    ) -> np.ndarray:
        """Trace a batch of rays; return noodle rows (M, 14) float64.

        DEPRECATED: uses CPU Snell tracer; replace with GPU dispatch.

        Columns 0-10 match the standard LensManifold schema.
        Columns 11-13 are the full-assembly extension:
          11, 12: sensor_x, sensor_y  — actual traced sensor landing position
          13:     focus_z             — sensor z offset from nominal

        Rows for missed/TIR rays are silently dropped.
        """
        preset = preset_override if preset_override is not None else self.preset
        wls = preset.wavelengths[:self.n_wavelengths]
        rows = []
        for i in range(len(ro_arr)):
            ro = ro_arr[i]
            rd = rd_arr[i]
            results = []
            for wl in wls:
                r = trace_ray(preset, ro, rd, wl)
                if r is not None:
                    results.append(r)
            if not results:
                continue
            u    = float(np.mean([r["aperture_uv"][0]  for r in results]))
            v    = float(np.mean([r["aperture_uv"][1]  for r in results]))
            fu   = float(np.mean([r["field_angle"][0]  for r in results]))
            fv   = float(np.mean([r["field_angle"][1]  for r in results]))
            in_d = np.mean([r["in_dir"]  for r in results], axis=0)
            out_d= np.mean([r["out_dir"] for r in results], axis=0)
            opl  = float(np.mean([r["opl"] for r in results]))
            in_d  /= max(np.linalg.norm(in_d), 1e-30)
            out_d /= max(np.linalg.norm(out_d), 1e-30)
            sh   = np.mean([r["sensor_hit"] for r in results], axis=0)
            rows.append([u, v, fu, fv,
                         in_d[0],  in_d[1],  in_d[2],
                         out_d[0], out_d[1], out_d[2],
                         opl,
                         float(sh[0]), float(sh[1]),
                         focus_z])
        if not rows:
            return np.zeros((0, 14), np.float64)
        return np.array(rows, np.float64)

    # ── Public API ─────────────────────────────────────────────────────────

    def bake(self):
        """Trace rays and build a LensManifold.

        DEPRECATED: uses CPU Snell tracer; use LensAssemblySpec.bake_lut() instead.

        Returns
        -------
        LensManifold
        """
        try:
            from camera_software.lens_manifold import LensManifold
        except ImportError:
            raise ImportError(
                "camera_software.lens_manifold is required for baking. "
                "Ensure camera_software/ is on the Python path."
            )

        if self.verbose:
            print(f"[BakeWorker] baking {self.n_rays} rays for "
                  f"'{self.preset.name}'...", flush=True)

        ro_arr, rd_arr = self._sample_rays(self.n_rays)
        data = self._trace_batch(ro_arr, rd_arr)

        if self.verbose:
            print(f"[BakeWorker]   {len(data)} noodles traced", flush=True)

        # Adaptive refinement: split (u,v) regions with high out-dir variance
        for pass_i in range(self.n_refine):
            if len(data) == 0:
                break
            # Partition by quadrant in aperture UV
            quads = [
                (data[:,0] >= 0) & (data[:,1] >= 0),
                (data[:,0] <  0) & (data[:,1] >= 0),
                (data[:,0] >= 0) & (data[:,1] <  0),
                (data[:,0] <  0) & (data[:,1] <  0),
            ]
            extras = []
            for mask in quads:
                chunk = data[mask]
                if len(chunk) < 4:
                    continue
                var = float(np.mean(np.var(chunk[:, 7:10], axis=0)))
                if var > self.threshold:
                    n_extra = max(64, len(chunk) // 2)
                    ro2, rd2 = self._sample_rays(n_extra)
                    extra = self._trace_batch(ro2, rd2)
                    if len(extra):
                        extras.append(extra)
            if extras:
                data = np.concatenate([data, *extras], axis=0)
                if self.verbose:
                    print(f"[BakeWorker]   pass {pass_i+1}: "
                          f"{len(data)} noodles after refinement", flush=True)

        if len(data) == 0:
            raise RuntimeError(
                "BakeWorker: no rays traced successfully. "
                "Check the CameraPreset optical geometry."
            )

        # Build the manifold from the noodle array using from_pairs()
        manifold = LensManifold.from_pairs(
            uv       = data[:, :2],
            fufv     = data[:, 2:4],
            in_dirs  = data[:, 4:7],
            out_dirs = data[:, 7:10],
            opls     = data[:, 10],
            meta     = {
                "preset_name": self.preset.name,
                "focal_mm":    self.preset.focal_mm,
                "f_number":    self.preset.f_number,
                "fov_deg":     self.preset.fov_deg,
                "n_noodles":   len(data),
            },
        )

        if self.verbose:
            print(f"[BakeWorker] done. {len(data)} noodles in manifold.", flush=True)

        return manifold

    def bake_assembly(
        self,
        focus_offsets: list[float] | None = None,
        n_focus_steps: int = 1,
        focus_range_m: float = 2e-3,
    ) -> np.ndarray:
        """Bake a full-assembly (N, 14) noodle array covering focus range.

        DEPRECATED: uses CPU Snell tracer; use LensAssemblySpec.bake_lut() instead.

        Each focus step shifts ``preset.sensor.z_pos`` by a different offset and
        traces ``self.n_rays`` rays, tagging each noodle with its ``focus_z``
        value.  The result is a single (N_total, 14) float64 array covering
        the full front-lens → sensor path across all focus positions.

        Parameters
        ----------
        focus_offsets : explicit list of sensor z offsets (metres).  If None,
                        ``n_focus_steps`` offsets are sampled uniformly over
                        ±``focus_range_m`` / 2 around the nominal position.
        n_focus_steps : number of focus steps when ``focus_offsets`` is None.
        focus_range_m : total focus sweep range in metres (default 2 mm).

        Returns
        -------
        (N, 14) float64 array.  Columns 0-10 match the standard noodle schema;
        cols 11-12 are baked sensor x/y; col 13 is focus_z offset (metres).
        """
        import copy

        if focus_offsets is None:
            if n_focus_steps == 1:
                focus_offsets = [0.0]
            else:
                half = focus_range_m * 0.5
                focus_offsets = list(
                    np.linspace(-half, half, n_focus_steps))

        slices = []
        for idx, fz in enumerate(focus_offsets):
            # Deep-copy the preset so each focus step is independent
            p = copy.deepcopy(self.preset)
            p.sensor.z_pos = float(p.sensor.z_pos) + fz

            ro_arr, rd_arr = self._sample_rays(self.n_rays)
            chunk = self._trace_batch(ro_arr, rd_arr,
                                      focus_z=fz, preset_override=p)

            # Adaptive refinement at each focus step
            for _ in range(self.n_refine):
                if len(chunk) == 0:
                    break
                quads = [
                    (chunk[:, 0] >= 0) & (chunk[:, 1] >= 0),
                    (chunk[:, 0] <  0) & (chunk[:, 1] >= 0),
                    (chunk[:, 0] >= 0) & (chunk[:, 1] <  0),
                    (chunk[:, 0] <  0) & (chunk[:, 1] <  0),
                ]
                extras = []
                for mask in quads:
                    seg = chunk[mask]
                    if len(seg) < 4:
                        continue
                    if float(np.mean(np.var(seg[:, 7:10], axis=0))) > self.threshold:
                        ro2, rd2 = self._sample_rays(max(64, len(seg) // 2))
                        ex = self._trace_batch(ro2, rd2,
                                               focus_z=fz, preset_override=p)
                        if len(ex):
                            extras.append(ex)
                if extras:
                    chunk = np.concatenate([chunk, *extras], axis=0)

            if self.verbose:
                print(f"[BakeWorker.bake_assembly] step {idx+1}/{len(focus_offsets)}"
                      f"  fz={fz*1e3:+.2f}mm  {len(chunk)} noodles", flush=True)
            if len(chunk):
                slices.append(chunk)

        if not slices:
            raise RuntimeError("bake_assembly: no rays traced successfully.")

        full = np.concatenate(slices, axis=0)
        if self.verbose:
            print(f"[BakeWorker.bake_assembly] done — {len(full):,} noodles "
                  f"across {len(focus_offsets)} focus steps", flush=True)
        return full

    def bake_training_table(
        self,
        path: str,
        target_gb: float,
        n_focus_steps: int = 1,
        focus_range_m: float = 2e-3,
        batch_size: int = 2_000_000,
        max_attempt_factor: float = 20.0,
    ) -> tuple[int, int]:
        """Stream a neural-lens training table to a float32 ``.npy`` memmap.

        Row schema, 16 float32 columns:
          0,1    aperture u,v
          2,3    field-angle fu,fv
          4..6   in_dir
          7..9   out_dir
          10     opl
          11,12  sensor_x,sensor_y
          13     focus_z
          14     wavelength_um
          15     reserved sample weight, currently 1

        Unlike ``_trace_batch()``, this does not average wavelengths.  Each
        successful ray/wavelength/focus combination becomes its own row.
        """
        import copy
        from numpy.lib.format import open_memmap

        row_cols = 16
        target_rows = max(1, int(float(target_gb) * (1024.0 ** 3) // (row_cols * 4)))
        table = open_memmap(path, mode="w+", dtype=np.float32,
                            shape=(target_rows, row_cols))

        if n_focus_steps <= 1:
            focus_offsets = [0.0]
        else:
            half = focus_range_m * 0.5
            focus_offsets = list(np.linspace(-half, half, n_focus_steps))
        wavelengths = list(self.preset.wavelengths[:self.n_wavelengths])

        written = 0
        attempted = 0
        max_attempts = max(batch_size, int(target_rows * max_attempt_factor))
        combo = 0
        while written < target_rows and attempted < max_attempts:
            fz = float(focus_offsets[combo % len(focus_offsets)])
            wl = float(wavelengths[(combo // len(focus_offsets)) % len(wavelengths)])
            combo += 1

            p = copy.deepcopy(self.preset)
            p.sensor.z_pos = float(p.sensor.z_pos) + fz
            n = min(batch_size, max_attempts - attempted)
            ro_arr, rd_arr = self._sample_rays(n)
            attempted += n

            rows = []
            for i in range(n):
                r = trace_ray(p, ro_arr[i], rd_arr[i], wl)
                if r is None:
                    continue
                in_d = r["in_dir"] / max(np.linalg.norm(r["in_dir"]), 1e-30)
                out_d = r["out_dir"] / max(np.linalg.norm(r["out_dir"]), 1e-30)
                sh = r["sensor_hit"]
                rows.append([
                    r["aperture_uv"][0], r["aperture_uv"][1],
                    r["field_angle"][0], r["field_angle"][1],
                    in_d[0], in_d[1], in_d[2],
                    out_d[0], out_d[1], out_d[2],
                    r["opl"], sh[0], sh[1], fz, wl, 1.0,
                ])

            if rows:
                arr = np.asarray(rows, dtype=np.float32)
                take = min(len(arr), target_rows - written)
                table[written:written + take, :] = arr[:take]
                written += take
                if self.verbose:
                    print(f"[BakeWorker.training] {written:,}/{target_rows:,} rows"
                          f"  attempts={attempted:,}  fz={fz*1e3:+.3f}mm"
                          f"  wl={wl:.4f}um", flush=True)

        table.flush()
        return written, target_rows

    def bake_neural_training_data(
        self,
        path: str,
        target_rows: int,
        batch_size: int = 2_000_000,
    ) -> tuple[int, int]:
        """Stream an (N, 11) noodle training table to a float32 ``.npy`` memmap.

        TODO: replace with GPU dispatch (T1→T2→T3 read-back + wave simulation).

        Row schema — 11 float32 columns (entry-surface canonical frame):

          Inputs (5):
            0  r_in          radial hit distance on entry surface (m)
            1  dir_r_in      radial direction component (into surface)
            2  dir_phi_in    azimuthal direction component (relative to theta_hit)
            3  dir_z_in      axial direction, positive = into front surface
            4  wavelength_um vacuum wavelength (µm)

          Outputs (6):
            5  r_out         radial distance on exit surface (m)
            6  delta_phi     exit azimuth offset = theta_out − theta_hit (radians)
            7  dir_r_out     radial direction at exit (theta_hit frame)
            8  dir_phi_out   azimuthal direction at exit (theta_hit frame)
            9  dir_z_out     axial direction at exit, positive = away from entry
           10  opl           optical path length entry→exit (m)

        Only transmitted rays are recorded; blocked rays are dropped entirely.
        All angles are in the entry-surface canonical frame (relative to theta_hit).
        """
        from numpy.lib.format import open_memmap

        N_COLS = 11
        table = open_memmap(path, mode="w+", dtype=np.float32,
                            shape=(target_rows, N_COLS))

        wavelengths = list(self.preset.wavelengths[:self.n_wavelengths])
        z_ent = self._z_entrance()
        z_ext = self._z_exit()

        written = 0
        page    = 0
        rows_per_page = math.ceil(target_rows / len(wavelengths))
        n_batch       = max(batch_size, rows_per_page * 3)

        bnd_r: list = []; bnd_dz: list = []; bnd_ok: list = []; bnd_count = 0

        while written < target_rows:
            wl = float(wavelengths[page % len(wavelengths)])
            page += 1

            ro_arr, rd_arr = self._sample_rays(n_batch)
            ro_arr[:, 2] = z_ent

            result = trace_ray_batch(self.preset, ro_arr, rd_arr, wl)
            alive = result["alive"]

            # Collect (r_in, dir_z_in, alive) for all rays before alive filter
            if bnd_count < 300_000:
                _r_all  = np.hypot(ro_arr[:, 0], ro_arr[:, 1])
                _dz_all = -rd_arr[:, 2]   # positive = into entrance surface
                bnd_r.append(_r_all); bnd_dz.append(_dz_all); bnd_ok.append(alive)
                bnd_count += len(_r_all)

            sh    = result["sensor_hit"]   # (N,3)
            od    = result["out_dir"]      # (N,3) unit, od_z < 0 toward sensor
            opl_v = result["opl"]          # (N,)

            if not alive.any():
                if self.verbose:
                    print(f"[BakeWorker.neural] wl={wl:.4f}µm: 0 hits, skipping",
                          flush=True)
                continue

            # ── Mask to live rays ─────────────────────────────────────────────
            ro_live = ro_arr[alive]
            rd_live = rd_arr[alive]
            sh_live = sh[alive]
            od_live = od[alive]
            op_live = opl_v[alive]

            # ── Entry-surface canonical frame ─────────────────────────────────
            x_hit     = ro_live[:, 0]
            y_hit     = ro_live[:, 1]
            theta_hit = np.arctan2(y_hit, x_hit)
            r_in      = np.hypot(x_hit, y_hit)
            cos_t     = np.cos(theta_hit)
            sin_t     = np.sin(theta_hit)

            dir_r_in  =  rd_live[:, 0] * cos_t + rd_live[:, 1] * sin_t
            dir_phi_in= -rd_live[:, 0] * sin_t + rd_live[:, 1] * cos_t
            dir_z_in  = -rd_live[:, 2]   # positive: rd_z < 0 into front surface

            # ── Backproject sensor hit to exit surface ────────────────────────
            dz_safe  = np.where(np.abs(od_live[:, 2]) > 1e-9, od_live[:, 2], -1e-9)
            t_exit   = (z_ext - sh_live[:, 2]) / dz_safe
            exit_pos = sh_live + t_exit[:, None] * od_live

            x_out     = exit_pos[:, 0]
            y_out     = exit_pos[:, 1]
            theta_out = np.arctan2(y_out, x_out)
            r_out     = np.hypot(x_out, y_out)

            delta_phi = theta_out - theta_hit
            delta_phi = delta_phi - (2.0 * np.pi) * np.round(delta_phi / (2.0 * np.pi))

            dir_r_out  =  od_live[:, 0] * cos_t + od_live[:, 1] * sin_t
            dir_phi_out= -od_live[:, 0] * sin_t + od_live[:, 1] * cos_t
            dir_z_out  = -od_live[:, 2]   # positive: od_z < 0 away from entry

            rows = np.column_stack([
                r_in, dir_r_in, dir_phi_in, dir_z_in,
                np.full(alive.sum(), wl, np.float32),
                r_out, delta_phi, dir_r_out, dir_phi_out, dir_z_out,
                op_live,
            ]).astype(np.float32)

            take = min(len(rows), target_rows - written)
            table[written:written + take] = rows[:take]
            written += take

            if self.verbose:
                print(f"[BakeWorker.neural] {written:,}/{target_rows:,}  "
                      f"wl={wl:.4f}µm  hits={alive.sum():,}", flush=True)

        # Fit parametric acceptance boundary from collected ray data
        c0, c1 = 0.0, 0.0
        if bnd_r:
            from .neural_assembly import fit_acceptance_boundary
            c0, c1 = fit_acceptance_boundary(
                np.concatenate(bnd_r),
                np.concatenate(bnd_dz),
                np.concatenate(bnd_ok),
                r_lens=float(self.preset.aperture_stop.r_outer),
            )
        if self.verbose:
            print(f"[BakeWorker.neural] acceptance boundary c0={c0:.4f} c1={c1:.4f}",
                  flush=True)
        table.flush()
        return written, target_rows, np.array([c0, c1], np.float32)

    def _z_exit(self) -> float:
        """Z coordinate of the last optical surface (back/exit plane)."""
        return float(min(el.z_vertex
                         for el in self.preset.lens_group.elements) - 0.001)

    def _z_entrance(self) -> float:
        """Z coordinate of the first optical surface (entrance plane).

        This is the front-most lens vertex plus a small safety margin — the
        same value used internally by bake_neural_training_data as z_ent.
        Pass to export_payload(z_entrance=...) so the C++ header matches the
        training geometry exactly.
        """
        return float(max(el.z_vertex
                         for el in self.preset.lens_group.elements) + 0.001)

    def bake_glsl_source(self) -> str:
        """Emit a GLSL compute shader that traces one ray per invocation.

        This is the GPU bake path for when the number of rays is very large
        (> 1M).  Returns a complete GLSL 460 compute shader source string.
        The shader reads ray (ro, rd) from binding 0 SSBO and writes noodle
        rows to binding 1 SSBO.

        Surface intercept functions are inlined for every element in the
        lens group, indexed by element order.
        """
        lines = [
            "#version 460 core",
            "layout(local_size_x = 64) in;",
            "",
            "// Input rays (ro.xyz + rd.xyz per ray, float64 emulated as two float32 pairs)",
            "layout(std430, binding=0) readonly buffer RayBuf { float rays[]; };",
            "// Output noodle rows: 11 float64 per noodle → 22 float32",
            "layout(std430, binding=1) writeonly buffer NoodleBuf { float noodles[]; };",
            "layout(std430, binding=2) writeonly buffer HitCountBuf { uint hit_count; };",
            "",
        ]

        # Inline all surface intercept functions
        for idx, el in enumerate(self.preset.lens_group.elements):
            fn_name = f"surf_{idx}"
            lines.append(f"// Element {idx}: {el.label}")
            lines.append(el.surface.glsl_intercept_fn(fn_name))
            lines.append("")

        # Inline aperture stop
        lines.append("// Aperture stop")
        lines.append(self.preset.aperture_stop.glsl_intercept_fn("surf_aperture"))
        lines.append("")

        # Inline sensor surface
        lines.append("// Sensor surface")
        lines.append(self.preset.sensor.glsl_intercept_fn("surf_sensor"))
        lines.append("")

        # Snell's law helper
        lines += [
            "vec3 snell(vec3 rd, vec3 n, float n1, float n2) {",
            "    float cos_i = -dot(rd, n);",
            "    float ratio = n1 / n2;",
            "    float sin2t = ratio*ratio*(1.0 - cos_i*cos_i);",
            "    if (sin2t > 1.0) return vec3(0.0); // TIR sentinel",
            "    float cos_t = sqrt(max(0.0, 1.0 - sin2t));",
            "    return normalize(ratio*rd + (ratio*cos_i - cos_t)*n);",
            "}",
            "",
        ]

        # Main trace function
        n_els = len(self.preset.lens_group.elements)
        wl_list = ", ".join(f"{w}" for w in self.preset.wavelengths[:self.n_wavelengths])

        lines += [
            "void main() {",
            "    uint gid = gl_GlobalInvocationID.x;",
            "    vec3 ro = vec3(rays[gid*6+0], rays[gid*6+1], rays[gid*6+2]);",
            "    vec3 rd = normalize(vec3(rays[gid*6+3], rays[gid*6+4], rays[gid*6+5]));",
            "",
            f"    float[{self.n_wavelengths}] wls = float[]({wl_list});",
            "    vec3 out_dir_sum = vec3(0.0);",
            "    float opl_sum = 0.0;",
            "    int   hit_count_wl = 0;",
            "    vec3  ap_hit = vec3(0.0);",
            "    vec3  in_dir = rd;",
            "",
            f"    for (int wi = 0; wi < {self.n_wavelengths}; wi++) {{",
            "        vec3  r  = ro; vec3 d = rd;",
            "        float n1 = 1.0; float opl = 0.0;",
            "        float t; vec3 normal;",
            "        bool ok = true;",
        ]

        # Unroll element loop
        for idx, el in enumerate(self.preset.lens_group.elements):
            n_out = el.glass_out.n_d  # simplified: no Sellmeier in GLSL path
            lines += [
                f"        if (ok) {{",
                f"            vec3 r_local = r; r_local.z -= {el.z_vertex};",
                f"            if (!surf_{idx}(r_local, d, t, normal)) {{ ok = false; }}",
                f"            else {{",
                f"                opl += n1 * t;",
                f"                vec3 nd = snell(d, normal, n1, {n_out});",
                f"                if (length(nd) < 0.5) {{ ok = false; }}",
                f"                else {{ d = normalize(nd); n1 = {n_out}; r = r_local + t*d; r.z += {el.z_vertex}; }}",
                f"            }}",
                f"        }}",
            ]

        lines += [
            "        if (ok) {",
            f"            vec3 r_s = r; r_s.z -= {self.preset.sensor.z_pos};",
            "            float t_s; vec3 ns;",
            "            if (surf_sensor(r_s, d, t_s, ns)) {",
            "                out_dir_sum += d; opl_sum += opl + n1*t_s;",
            "                hit_count_wl++;",
            "            }",
            "        }",
            "    }",
            "",
            "    if (hit_count_wl == 0) return;",
            "",
            "    // Aperture UV",
            f"    float r_max = {self.preset.aperture_stop.r_outer};",
            f"    float z_ap  = {self.preset.aperture_stop.z_pos};",
            "    float t_ap = (z_ap - ro.z) / max(abs(rd.z), 1e-9);",
            "    vec3  ap   = ro + t_ap * rd;",
            "    float u    = ap.x / max(r_max, 1e-9);",
            "    float v    = ap.y / max(r_max, 1e-9);",
            "    if (abs(u) > 1.0 || abs(v) > 1.0) return;",
            "",
            "    vec3  out_dir = normalize(out_dir_sum / float(hit_count_wl));",
            "    float opl_avg = opl_sum / float(hit_count_wl);",
            "    float fu  = rd.x / max(abs(rd.z), 1e-9);",
            "    float fv  = rd.y / max(abs(rd.z), 1e-9);",
            "",
            "    uint slot = atomicAdd(hit_count, 1u);",
            "    uint base = slot * 11u;",
            "    noodles[base+0]  = u;",
            "    noodles[base+1]  = v;",
            "    noodles[base+2]  = fu;",
            "    noodles[base+3]  = fv;",
            "    noodles[base+4]  = in_dir.x;",
            "    noodles[base+5]  = in_dir.y;",
            "    noodles[base+6]  = in_dir.z;",
            "    noodles[base+7]  = out_dir.x;",
            "    noodles[base+8]  = out_dir.y;",
            "    noodles[base+9]  = out_dir.z;",
            "    noodles[base+10] = opl_avg;",
            "}",
        ]

        return "\n".join(lines)
