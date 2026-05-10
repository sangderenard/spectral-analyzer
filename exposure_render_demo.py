"""Long-exposure dual-backend ray-traced demo & emissivity-gain calibration.

Goal
----
Drive **both** spectral ray tracers — the C++ ``_spectral_kernels.RayTracer``
and the GLSL compute ``_gpu_ray_field`` pipeline — through a single
*finished-exposure* loop in which every emitted output frame represents one
fully-accumulated exposure of N_total rays (N_total can be 10M+ and is
sub-batched / streamed automatically) rather than one tiny dispatch.

That inversion ("frame == exposure", not "frame == ray bunch") is the central
calibration semantic: we know

    photons/pixel = L · Ω · τ · A_pix · t / (h c / λ)

so the exposure budget yields a *known* total radiant exposure target H_J,
and the tracer accumulates rays until the running estimate of H_J matches
that target.  Per-exposure emissivity gain is the scalar that brings the
measured sensor signal to the predicted target — recorded and saved so
later renders can short-circuit the calibration loop.

Scene
-----
Borrowed verbatim from ``test_basic_gl_cpp_window.py`` (central textured
sphere, saddle stage floor, 12 orbiters on distinct 3-D orbital planes).
The basic-rasterizer harness uses the same scene to validate shading; this
file repurposes that scene for *photometric* ray-traced exposures using the
unified Phase-2 ``mat_buf`` material pipeline (32 spectral bands, central
``MaterialDatabase`` registry, hard cutover — no legacy refl_re/refl_im
buffers).

Output
------
- A live pygame window with the C++ pane on the left and the GLSL pane on
  the right.  Each finished exposure replaces the visible image.
- One ``./exposures/{frame:04d}_{backend}.png`` dump per accumulated frame.
- One ``./exposures/{frame:04d}_summary.json`` log per frame with the
  full scientific budget (N_rays, gain dB, virtual exposure time s,
  H_target J, H_measured J, photons/pixel, SNR).

Status
------
- C++ backend: fully wired against the Phase 2b SLICE 1 pybind ctor
  ``RayTracer(n_tri, verts, normals, mat_idx, mat_buf, mat_n_mats,
  freq_hz, speed_m_s, atmo_abs)`` and ``integrate_image_into`` (which
  accumulates additively into the caller-owned float32 (n_bands, H, W)
  buffer — perfect for sub-batch streaming).
- GLSL backend: scaffolded.  The full SensorAccumulator + BVH +
  ScaleContext + FilmStack pipeline lives in ``demo_pluck_gl``; we wire it
  best-effort.  When unavailable, the right pane shows the C++ image
  monochrome-toned and labelled "GLSL pending" so the harness still runs
  end-to-end and prints the same calibration log.

CLI
---
    python exposure_render_demo.py \\
        --total-rays 10_000_000 --rays-per-batch 250_000 \\
        --width 320 --height 200 --frames 6 --backend both
"""
from __future__ import annotations

import argparse
import gc
import faulthandler
import json
import math
import os
import sys
import tempfile
import threading
import time
import traceback
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import numpy as np

import _spectral_kernels as _sk  # type: ignore[import]

from camera_exposure_budget import (
    CameraOptics,
    FilmExposure,
    RayDispatchPlan,
    lambertian_emitter_radiance,
    plan_ray_budget,
    summarize_plan,
    H_PLANCK,
    C_LIGHT,
)
from camera_parametric_solver import solve_sane_pinhole_camera
from material_db import MaterialDatabase, MAX_SPECTRAL_BANDS
from sensor_film_db import SensorFilmDatabase, MAX_SENSOR_FILM_SLOTS

# Borrow scene authoring verbatim from the basic-rasterizer harness.
import test_basic_gl_cpp_window as scene_mod   # noqa: E402

try:
    from surface_spline_utils import parameterize_mesh as _ss_parameterize_mesh
    _HAS_SURFACE_SPLINE = True
except ImportError:
    _HAS_SURFACE_SPLINE = False

try:
    from sdf_plugins import get_sdf_driver as _get_sdf_driver
except Exception:
    _get_sdf_driver = None

try:
    from bdpt_integrator import CameraSensor, TriangleGroup
    from bdpt_integrator import TRI_GROUP_ROLE_SENSOR, TRI_GROUP_SAMPLE_PIXEL_CONE
    from bdpt_integrator import aggregate_to_image_pixel_cone
    _HAS_BDPT_INTEGRATION = True
except ImportError:
    _HAS_BDPT_INTEGRATION = False
    CameraSensor = None
    TriangleGroup = None
    TRI_GROUP_ROLE_SENSOR = None
    TRI_GROUP_SAMPLE_PIXEL_CONE = None
    aggregate_to_image_pixel_cone = None


def _weld_mesh(verts_flat_n9: np.ndarray,
              tol: float = 1.0e-5) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat (N_tri, 9) triangle buffer to indexed (V, 3) + (N_tri, 3) faces.

    Uses lexicographic sorting with a quantization tolerance.  Suitable for
    the scene meshes built by scene_mod (typical edge length >> 1e-5 m).
    Returns (unique_verts float64, faces int32).
    """
    pts = verts_flat_n9.reshape(-1, 3).astype(np.float64)
    quant = np.round(pts / tol).astype(np.int64)
    order = np.lexsort(quant.T[::-1])
    sq = quant[order]
    diff = np.empty(len(sq), dtype=bool)
    diff[0] = True
    diff[1:] = np.any(sq[1:] != sq[:-1], axis=1)
    uid_sorted = np.cumsum(diff, dtype=np.int32) - 1
    inv = np.empty(len(pts), dtype=np.int32)
    inv[order] = uid_sorted
    first_sorted = np.where(diff)[0]
    unique_verts = pts[order[first_sorted]]
    faces = inv.reshape(-1, 3)
    return unique_verts, faces


def _cam_vis_name(mode: int) -> str:
    return {0: "AS_IS", 1: "DIRECT_HIT", 2: "FULL_MARCH"}.get(mode, str(mode))


def _transp_name(mode: int) -> str:
    return {0: "BLOCK", 1: "XRAY"}.get(mode, str(mode))


def _build_flat_sensor_geometry(optics: CameraOptics, 
                                height_px: int,
                                width_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a flat rectangular sensor geometry as triangles.
    
    Returns (positions, uvs, triangle_indices) for a simple 2x1 grid (2 tris)
    at z=0 spanning the sensor dimensions based on optics.
    
    Parameters:
    - optics: CameraOptics with pixel_pitch_um, sensor dimensions
    - height_px, width_px: sensor pixel dimensions (for reference)
    
    Returns:
    - positions: (4, 3) float64 — corners of flat sensor plane
    - uvs: (4, 2) float64 — normalized [0,1]² coordinates
    - tris: (2, 3) int32 — triangle indices
    """
    # Sensor half-dimensions in metres
    sensor_w_m = optics.sensor_w_mm * 1.0e-3
    sensor_h_m = optics.sensor_h_mm * 1.0e-3
    
    # Four corners at z=0 (sensor plane)
    positions = np.array([
        [-sensor_w_m/2, -sensor_h_m/2, 0.0],
        [ sensor_w_m/2, -sensor_h_m/2, 0.0],
        [ sensor_w_m/2,  sensor_h_m/2, 0.0],
        [-sensor_w_m/2,  sensor_h_m/2, 0.0],
    ], dtype=np.float64)
    
    # UVs map [0,1]² uniformly
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float64)
    
    # Two triangles covering the quad
    tris = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    
    return positions, uvs, tris


# ─────────────────────────────────────────────────────────────────────────────
# Scientific defaults — grounded in real photographic conventions.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_OPTICS = CameraOptics(
    focal_mm           = 35.0,
    aperture_mm        = 25.0,         # f/1.4
    pixel_pitch_um     = 8.4,
    sensor_w_mm        = 36.0,
    sensor_h_mm        = 24.0,
    lens_transmission  = 0.95,
)

DEFAULT_FILM = FilmExposure(
    iso                 = 100.0,
    exposure_time_s     = 1.0 / 60.0,
    quantum_efficiency  = 0.5,
    target_mid_grey     = 0.18,
)

# Spectral grid for the C++ tracer.  We keep this short (8 bands across the
# visible) so RAM stays manageable; the unified mat_buf still allocates 32
# slots per material with the unused tail zeroed.
DEFAULT_FREQ_HZ = (C_LIGHT / np.linspace(700e-9, 400e-9, 8)).astype(np.float64)

ENABLE_EXPOSURE_PROFILING = False


class StageProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._data: dict[str, list[float]] = {}
        self._order: list[str] = []

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1e3
            self._data.setdefault(name, []).append(ms)
            if name not in self._order:
                self._order.append(name)

    def format_report(self, prefix: str = "[profile]") -> str:
        if not self.enabled or not self._order:
            return ""
        parts = [f"{k}={np.mean(self._data[k]):.1f}ms" for k in self._order if self._data.get(k)]
        current = peak = None
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
        if current is not None and peak is not None:
            parts.append(f"mem={current / (1024 ** 2):.1f}MB/{peak / (1024 ** 2):.1f}MB")
        return f"{prefix} " + "  ".join(parts)

    def report(self, prefix: str = "[profile]") -> None:
        msg = self.format_report(prefix=prefix)
        if msg:
            print(msg, flush=True)

DEFAULT_CALIBRATION_SCENE_SEQUENCE = (
    "calib-grid",
    "calib-bw-rgb",
    "calib-rgb-diagram",
    "calib-step-wedge",
    "calib-prism-backplate",
    "tungsten-cavity",
)


def _build_default_scene_schedule(n_frames_planned: int, demo_scene: str = "orbiters") -> tuple[str, ...]:
    n_frames = max(1, int(n_frames_planned))
    if n_frames == 1:
        return (demo_scene,)
    calibs = DEFAULT_CALIBRATION_SCENE_SEQUENCE
    prefix = tuple(calibs[i % len(calibs)] for i in range(n_frames - 1))
    return prefix + (demo_scene,)

# EndpointRecord in Python is float32 (N, 16): 64 bytes/record payload.
_BDPT_RECORD_FLOATS = 16
_BDPT_RECORD_BYTES = _BDPT_RECORD_FLOATS * np.dtype(np.float32).itemsize


def _bdpt_cap_from_bytes(max_bytes: int) -> int:
    """Analytical endpoint-record cap from byte budget."""
    return max(0, int(max_bytes) // _BDPT_RECORD_BYTES)


# ─────────────────────────────────────────────────────────────────────────────
# Scene → tracer geometry adapter
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TracerScene:
    """Everything both ray tracers need to be built once per camera frame."""
    verts:        np.ndarray   # (N_tri, 9)  float64 — flat triangle vertices
    normals:      np.ndarray   # (N_tri, 3)  float64
    mat_idx:      np.ndarray   # (N_tri,)    int32   — index into MaterialDatabase
    mat_buf:      np.ndarray   # (N_mat * MAX_SPECTRAL_BANDS, 12) float32
    mat_n_mats:   int
    src_pos:      np.ndarray   # (N_src, 3)  float64 — emissive triangle centroids
    src_dir:      np.ndarray   # (N_src, 3)  float64 — outward normals
    src_directivity: np.ndarray  # (N_src,)  float64 — Lambertian → 1.0
    src_area_m2:  np.ndarray   # (N_src,)    float64 — for power bookkeeping
    src_emit_W:   np.ndarray   # (N_src,)    float64 — luminance-weighted emission
    src_emit_rgb_W: np.ndarray # (N_src, 3)  float64 — per-channel emissive power proxy
    src_tri_idx:  np.ndarray   # (N_src,)    int32   — triangle indices in verts/normals
    bounds_min:   np.ndarray   # (3,)        float32
    bounds_max:   np.ndarray   # (3,)        float32

    @property
    def total_emissive_power_W(self) -> float:
        return float(self.src_emit_W.sum())

    @property
    def total_emissive_area_m2(self) -> float:
        return float(self.src_area_m2.sum())

    def scene_radiance_W_sr_m2(self) -> float:
        """Lambertian-disk radiance the camera will see if the emitters
        completely fill its FOV (upper-bound; capture_efficiency in the budget
        plan is what closes the gap empirically)."""
        return lambertian_emitter_radiance(
            self.total_emissive_power_W, self.total_emissive_area_m2)


_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _build_tracer_scene(t: float, scene_mode: str = "orbiters") -> TracerScene:
    """Materialise the borrowed scene into ray-tracer geometry.

    The MaterialDatabase is built ONCE per process (we trust scene_mod's
    ``register_materials``); ``build_mat_buf`` returns the unified Phase-2
    buffer that both backends index by ``mat_idx[tri] * MAX_SPECTRAL_BANDS
    + band``.
    """
    db, idx = scene_mod.register_materials()
    verts8, _mat_per_v, _gid_per_v, mat_per_tri, groups = \
        scene_mod.scene_for_phase(idx, t, scene_mode=scene_mode)

    # ── Triangulate ──────────────────────────────────────────────────────
    pts = verts8[:, 0:3].astype(np.float64).reshape(-1, 3, 3)
    n_tri = pts.shape[0]
    verts_flat = pts.reshape(n_tri, 9)

    # Per-triangle geometric normal (right-hand rule, normalised).
    e1 = pts[:, 1] - pts[:, 0]
    e2 = pts[:, 2] - pts[:, 0]
    cr = np.cross(e1, e2)
    nrm = np.linalg.norm(cr, axis=1, keepdims=True)
    normals = (cr / np.where(nrm > 1.0e-12, nrm, 1.0)).astype(np.float64)
    area = (0.5 * nrm.ravel()).astype(np.float64)
    centroid = pts.mean(axis=1).astype(np.float64)

    # ── Material side ────────────────────────────────────────────────────
    mat_idx_arr = np.ascontiguousarray(mat_per_tri, np.int32)
    mat_buf = db.build_mat_buf()                                # (N_mat*32, 12) f32
    mat_n_mats = int(mat_buf.shape[0] // MAX_SPECTRAL_BANDS)

    # ── Emissive sources: any triangle whose material has nonzero PBR
    #    emission row (build_tensors()['pbr'][mat,8:11]).  We use the same
    #    reference the emissive_ray_packer does, but emit one *source* per
    #    triangle (the C++ tracer expands each source into n_rays Monte-Carlo
    #    samples internally — sub-batching is just N_BATCH calls with
    #    different seeds).
    pbr = db.build_tensors().get("pbr", np.zeros((0, 16), np.float32))
    emis_rgb = pbr[mat_idx_arr.clip(0, max(0, pbr.shape[0]-1)), 8:11]
    luma = np.maximum(0.0, emis_rgb @ _LUMA)
    emissive = luma > 1.0e-8
    if not np.any(emissive):
        # Synthesise a dim ambient point so the budget plan still has a source.
        emissive = np.zeros(n_tri, bool); emissive[0] = True
        luma = np.zeros(n_tri, np.float32); luma[0] = 1.0e-3

    sel = np.where(emissive)[0]
    src_pos = centroid[sel]
    src_dir = normals[sel]
    # Lambertian (cos^1) directivity for area emitters.  The C++ tracer reads
    # this as the cosine exponent; 1.0 ≡ Lambertian.
    src_directivity = np.ones(sel.size, np.float64)
    src_area = area[sel]
    # Emissive power per triangle: luminance × area, scaled to the global
    # ``DEFAULT_FILM`` exposure later by the plan's energy_per_ray_J.
    src_emit_W = (luma[sel].astype(np.float64) * src_area)
    src_emit_rgb_W = emis_rgb[sel].astype(np.float64) * src_area[:, None]

    # ── Scene AABB for the GLSL pipeline ─────────────────────────────────
    pts_all = pts.reshape(-1, 3)
    bmin = pts_all.min(axis=0).astype(np.float32)
    bmax = pts_all.max(axis=0).astype(np.float32)

    return TracerScene(
        verts            = verts_flat,
        normals          = normals,
        mat_idx          = mat_idx_arr,
        mat_buf          = mat_buf,
        mat_n_mats       = mat_n_mats,
        src_pos          = src_pos,
        src_dir          = src_dir,
        src_directivity  = src_directivity,
        src_area_m2      = src_area,
        src_emit_W       = src_emit_W,
        src_emit_rgb_W   = src_emit_rgb_W,
        src_tri_idx      = sel.astype(np.int32, copy=False),
        bounds_min       = bmin,
        bounds_max       = bmax,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Camera geometry — pinhole at origin, looking at SCENE_CENTER
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PinholeCamera:
    pos:    np.ndarray    # (3,) double
    fwd:    np.ndarray    # (3,) double
    up:     np.ndarray    # (3,) double
    fov_y_rad: float
    width:  int
    height: int

    @classmethod
    def looking_at_scene(cls, width: int, height: int,
                         optics: CameraOptics) -> "PinholeCamera":
        eye = np.array([0.0, 0.0, 0.0], np.float64)
        tgt = np.asarray(scene_mod.SCENE_CENTER, np.float64)
        fwd = tgt - eye
        fwd /= max(np.linalg.norm(fwd), 1.0e-9)
        up = np.array([0.0, 1.0, 0.0], np.float64)
        # Real fov_y from sensor height and focal length.
        fov_y = 2.0 * math.atan2(optics.sensor_h_mm * 0.5, optics.focal_mm)
        return cls(pos=eye, fwd=fwd, up=up,
                   fov_y_rad=float(fov_y), width=width, height=height)


# ─────────────────────────────────────────────────────────────────────────────
# Backend interface
# ─────────────────────────────────────────────────────────────────────────────
class ExposureBackend:
    name = "abstract"

    def __init__(self, scene: TracerScene, cam: PinholeCamera,
                 freq_hz: np.ndarray, *, max_bounces: int = 4,
                 min_amplitude: float = 1.0e-3, atmo_abs_db_per_m: float = 0.0):
        self.scene = scene
        self.cam   = cam
        self.freq_hz = np.ascontiguousarray(freq_hz, np.float64)
        self.n_bands = int(self.freq_hz.size)
        self.max_bounces = int(max_bounces)
        self.min_amplitude = float(min_amplitude)
        self.atmo_abs = np.full(self.n_bands, float(atmo_abs_db_per_m), np.float64)
        # Persistent per-exposure accumulators (n_bands, H, W) in float32.
        self.accum       = np.zeros((self.n_bands, cam.height, cam.width), np.float32)
        # surf_accum: sensor-group EndpointRecord contributions (direct hits).
        # field_accum: non-sensor EndpointRecord contributions (ambient/field).
        self.surf_accum  = np.zeros((self.n_bands, cam.height, cam.width), np.float32)
        self.field_accum = np.zeros((self.n_bands, cam.height, cam.width), np.float32)
        self.n_rays_accumulated = 0

    # ── Sub-class hooks ──────────────────────────────────────────────────
    def reset_exposure(self) -> None:
        self.accum.fill(0.0)
        self.surf_accum.fill(0.0)
        self.field_accum.fill(0.0)
        self.n_rays_accumulated = 0

    def render_batch(self, n_rays: int, seed: int) -> None:
        raise NotImplementedError

    def finalize_image(self, gain: float) -> np.ndarray:
        """Return (H, W, 3) float32 RGB image after applying ``gain``.

        The accumulator holds per-band amplitude magnitudes |A[b]|;
        we map the band index into a smooth wavelength-driven RGB triplet
        (CIE-style) for visualisation.
        """
        a = self.accum * float(gain)
        rgb = _bands_to_rgb(a, self.freq_hz)
        # Tone-map (Reinhard) for display only; calibration uses raw ``a``.
        m = float(rgb.max()) if rgb.size else 0.0
        if m > 0.0:
            disp = rgb / (1.0 + rgb)
        else:
            disp = rgb
        return np.clip(disp, 0.0, 1.0).astype(np.float32)

    def measured_radiant_exposure_J(self, energy_per_ray_J: float) -> float:
        """Translate the running accumulator into a Joule estimate.

        The C++ ``integrate_image_into`` deposits ``|A| · cos / r²`` in
        normalised units per ray.  We close the loop by saying the total
        deposited "amplitude mass" times ``energy_per_ray_J`` divided by the
        actual ray count is the per-ray Joule contribution; summed across
        the image gives total H.  Calibration discovers the multiplicative
        constant that makes this match the predicted ``plan.target_H_J``.
        """
        if self.n_rays_accumulated == 0:
            return 0.0
        amplitude_mass = float(self.accum.sum())
        return amplitude_mass * energy_per_ray_J


# ─────────────────────────────────────────────────────────────────────────────
# Spectral → display RGB (HSL compositor with soft wavelength gating)
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid01(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def _hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Vectorized HSL → RGB conversion for display compositing."""
    h = np.mod(np.asarray(h, np.float64), 1.0)
    s = np.clip(np.asarray(s, np.float64), 0.0, 1.0)
    l = np.clip(np.asarray(l, np.float64), 0.0, 1.0)

    q = np.where(l < 0.5, l * (1.0 + s), l + s - l * s)
    p = 2.0 * l - q

    def _hue_to_rgb(t: np.ndarray) -> np.ndarray:
        t = np.mod(t, 1.0)
        return np.where(
            t < 1.0 / 6.0, p + (q - p) * 6.0 * t,
            np.where(
                t < 1.0 / 2.0, q,
                np.where(
                    t < 2.0 / 3.0, p + (q - p) * (2.0 / 3.0 - t) * 6.0,
                    p,
                ),
            ),
        )

    r = _hue_to_rgb(h + 1.0 / 3.0)
    g = _hue_to_rgb(h)
    b = _hue_to_rgb(h - 1.0 / 3.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _bands_to_rgb(image_b_h_w: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    """Project spectral bands to display RGB via HSL accumulation.

    H tracks the wavelength-to-visible remap, L tracks deposited power with a
    tuned sigmoid, and S comes from circular hue coherence after alpha-weighted
    mixing. UV and deep-red extremes are softened via alpha so they still
    contribute without overpowering the visible band.
    """
    n_b = image_b_h_w.shape[0]
    if n_b == 0:
        H, W = image_b_h_w.shape[1:]
        return np.zeros((H, W, 3), np.float32)

    power = np.maximum(0.0, np.asarray(image_b_h_w, np.float64))
    wl_nm = (C_LIGHT / np.asarray(freq_hz, np.float64)) * 1.0e9

    # Map all wavelengths into the visible hue range, saturating toward UV at
    # the short end and red at the long end.
    hue_sigmoid_nm = 42.0
    hue_mix = _sigmoid01((wl_nm - 555.0) / hue_sigmoid_nm)
    hue = (1.0 - hue_mix) * 0.74

    # Fade the extremes with alpha rather than hard-clamping them out.
    edge_dist = np.abs(wl_nm - 555.0)
    alpha = 0.18 + 0.82 * _sigmoid01((150.0 - edge_dist) / 24.0)
    sat_gate = _sigmoid01((132.0 - edge_dist) / 28.0)

    weighted = power * alpha[:, None, None]
    total_weight = weighted.sum(axis=0)
    total_power = power.sum(axis=0)

    cos_h = np.cos((2.0 * math.pi) * hue)[:, None, None]
    sin_h = np.sin((2.0 * math.pi) * hue)[:, None, None]

    hue_x = np.sum(weighted * cos_h, axis=0)
    hue_y = np.sum(weighted * sin_h, axis=0)
    coherence = np.sqrt(hue_x * hue_x + hue_y * hue_y) / np.maximum(total_weight, 1.0e-9)

    sat_weighted = power * (alpha * sat_gate)[:, None, None]
    sat_mean = np.sum(sat_weighted, axis=0) / np.maximum(total_power, 1.0e-9)
    saturation = np.clip((coherence ** 0.82) * np.sqrt(np.maximum(sat_mean, 0.0)), 0.0, 1.0)

    hue_map = (np.arctan2(hue_y, hue_x) / (2.0 * math.pi)) % 1.0

    white = float(np.percentile(total_power, 99.8)) if total_power.size else 0.0
    white = max(white, 1.0e-8)
    lightness = _sigmoid01((total_power / white - 0.5) * 5.5)

    return _hsl_to_rgb(hue_map, saturation, lightness)


# ─────────────────────────────────────────────────────────────────────────────
# C++ backend — _spectral_kernels.RayTracer  (Phase 2b SLICE 1 ctor)
# ─────────────────────────────────────────────────────────────────────────────
class CppExposureBackend(ExposureBackend):
    name = "cpp"

    def __init__(self, scene: TracerScene, cam: PinholeCamera,
                 freq_hz: np.ndarray, **kw: Any):
        self._adaptive_mode = str(kw.pop("adaptive_mode", "stochastic")).strip().lower()
        if self._adaptive_mode not in ("stochastic", "quota", "uniform"):
            self._adaptive_mode = "stochastic"
        super().__init__(scene, cam, freq_hz, **kw)
        self.tracer = _sk.RayTracer(
            n_tri      = int(scene.verts.shape[0]),
            verts      = scene.verts,
            normals    = scene.normals,
            mat_idx    = scene.mat_idx,
            mat_buf    = scene.mat_buf,
            mat_n_mats = scene.mat_n_mats,
            freq_hz    = self.freq_hz,
            speed_m_s  = float(C_LIGHT),
            atmo_abs   = self.atmo_abs,
        )
        self._src_tri_verts = np.ascontiguousarray(
            self.scene.verts[self.scene.src_tri_idx].reshape(-1, 3, 3),
            np.float64,
        )
        self._source_need_ema: Optional[np.ndarray] = None
        self._source_rays_emitted = np.zeros(int(self.scene.src_pos.shape[0]), dtype=np.float64)

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        cam_f = np.asarray(self.cam.fwd, np.float64)
        cam_f = cam_f / max(np.linalg.norm(cam_f), 1.0e-12)
        cam_up_hint = np.asarray(self.cam.up, np.float64)
        cam_r = np.cross(cam_f, cam_up_hint)
        if np.linalg.norm(cam_r) < 1.0e-12:
            cam_r = np.cross(cam_f, np.array([1.0, 0.0, 0.0], np.float64))
        cam_r = cam_r / max(np.linalg.norm(cam_r), 1.0e-12)
        cam_u = np.cross(cam_r, cam_f)
        tan_half_v = math.tan(float(self.cam.fov_y_rad) * 0.5)
        tan_half_h = tan_half_v * (float(self.cam.width) / max(1.0, float(self.cam.height)))
        return cam_f, cam_r, cam_u, tan_half_h, tan_half_v

    def _density_need_map(self) -> np.ndarray:
        # Prioritize under-resolved regions and high local contrast zones.
        energy = np.sum(self.accum, axis=0, dtype=np.float64)
        p95 = max(1.0e-12, float(np.percentile(energy, 95.0)))
        en = np.clip(energy / p95, 0.0, 1.0)
        gx = np.abs(np.roll(en, -1, axis=1) - en)
        gy = np.abs(np.roll(en, -1, axis=0) - en)
        g = np.sqrt(gx * gx + gy * gy)
        gp95 = max(1.0e-12, float(np.percentile(g, 95.0)))
        gn = np.clip(g / gp95, 0.0, 1.0)
        under = 1.0 - np.sqrt(np.clip(en, 0.0, 1.0))
        need = 0.25 + under + 0.75 * gn
        return need.astype(np.float64, copy=False)

    def _project_source_need(self, src_pos: np.ndarray, need_map: np.ndarray) -> np.ndarray:
        cam_pos = np.asarray(self.cam.pos, np.float64)
        cam_f, cam_r, cam_u, tan_half_h, tan_half_v = self._camera_basis()
        v = np.asarray(src_pos, np.float64) - cam_pos[None, :]
        depth = np.dot(v, cam_f)

        need = np.full((src_pos.shape[0],), float(np.mean(need_map, dtype=np.float64)), dtype=np.float64)
        valid = depth > 1.0e-9
        if np.any(valid):
            vv = v[valid]
            d = depth[valid]
            x_img = np.dot(vv, cam_r)
            y_img = np.dot(vv, cam_u)
            ndc_x = x_img / (d * tan_half_h)
            ndc_y = -y_img / (d * tan_half_v)
            px = ((ndc_x + 1.0) * 0.5 * float(self.cam.width)).astype(np.int64)
            py = ((ndc_y + 1.0) * 0.5 * float(self.cam.height)).astype(np.int64)
            in_view = ((px >= 0) & (px < int(self.cam.width)) &
                       (py >= 0) & (py < int(self.cam.height)))
            idx_valid = np.where(valid)[0]
            if np.any(in_view):
                idx = idx_valid[in_view]
                need[idx] = need_map[py[in_view], px[in_view]]
        return need

    def _allocate_source_rays(self, n_rays: int, src_pos: np.ndarray,
                              mode: Optional[str] = None,
                              seed: Optional[int] = None) -> np.ndarray:
        n_sources = int(src_pos.shape[0])
        total_rays = max(n_sources, int(n_rays) * n_sources)
        alloc_mode = str(mode or self._adaptive_mode).strip().lower()
        if alloc_mode not in ("stochastic", "quota", "uniform"):
            alloc_mode = "stochastic"
        need_map = self._density_need_map()
        per_source_need = self._project_source_need(src_pos, need_map)

        if self._source_need_ema is None or self._source_need_ema.shape[0] != n_sources:
            self._source_need_ema = np.asarray(per_source_need, np.float64).copy()
        else:
            self._source_need_ema *= 0.8
            self._source_need_ema += 0.2 * per_source_need

        hist = np.asarray(self._source_rays_emitted[:n_sources], np.float64)
        hmean = max(1.0, float(np.mean(hist)))
        novelty = 1.0 / np.sqrt(1.0 + (hist / hmean))

        weights = np.maximum(1.0e-9, self._source_need_ema * novelty)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0.0:
            weights = np.full((n_sources,), 1.0 / max(1, n_sources), dtype=np.float64)
        else:
            weights = weights / weights_sum

        if alloc_mode == "uniform":
            counts = np.full((n_sources,), total_rays // max(1, n_sources), dtype=np.int64)
            rem = int(total_rays - int(np.sum(counts)))
            if rem > 0:
                counts[:rem] += 1
        else:
            raw = weights * float(total_rays - n_sources)
            counts = np.ones((n_sources,), dtype=np.int64)
            if alloc_mode == "stochastic":
                rng = np.random.default_rng(int(seed if seed is not None else 0xC0FFEE))
                draws = rng.multinomial(int(total_rays - n_sources), weights)
                counts += draws.astype(np.int64, copy=False)
            else:
                counts += np.floor(raw).astype(np.int64)
                remain = int(total_rays - int(np.sum(counts)))
                if remain > 0:
                    frac = raw - np.floor(raw)
                    order = np.argsort(-frac)
                    counts[order[:remain]] += 1

        self._source_rays_emitted[:n_sources] += counts.astype(np.float64)
        return counts.astype(np.int32, copy=False)

    @staticmethod
    def _sample_cosine_hemisphere_axes(normals: np.ndarray,
                                       rng: np.random.Generator) -> np.ndarray:
        n = np.ascontiguousarray(normals, np.float64)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.where(n_norm > 1.0e-12, n_norm, 1.0)

        ref = np.tile(np.array([[1.0, 0.0, 0.0]], np.float64), (n.shape[0], 1))
        mask = np.abs(n[:, 0]) >= 0.9
        ref[mask] = np.array([0.0, 1.0, 0.0], np.float64)

        t = np.cross(n, ref)
        t_norm = np.linalg.norm(t, axis=1, keepdims=True)
        t = t / np.where(t_norm > 1.0e-12, t_norm, 1.0)
        b = np.cross(n, t)

        r1 = rng.random(n.shape[0], dtype=np.float64)
        r2 = rng.random(n.shape[0], dtype=np.float64)
        r = np.sqrt(r1)
        phi = (2.0 * math.pi) * r2
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.sqrt(np.maximum(0.0, 1.0 - r1))

        d = (t * x[:, None]) + (b * y[:, None]) + (n * z[:, None])
        d_norm = np.linalg.norm(d, axis=1, keepdims=True)
        return np.ascontiguousarray(d / np.where(d_norm > 1.0e-12, d_norm, 1.0),
                                    np.float64)

    def _prepare_emissive_batch_sources(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Prepare randomized launch sites/axes for every emissive each batch."""
        rng = np.random.default_rng(int(seed))
        tri = self._src_tri_verts

        u = rng.random(tri.shape[0], dtype=np.float64)
        v = rng.random(tri.shape[0], dtype=np.float64)
        flip = (u + v) > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]

        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        src_pos = tri[:, 0] + (u[:, None] * e1) + (v[:, None] * e2)
        src_dir = self._sample_cosine_hemisphere_axes(self.scene.src_dir, rng)
        return np.ascontiguousarray(src_pos, np.float64), src_dir

    def run_bdpt_batch(self, emit_rays: np.ndarray, seed: int) -> np.ndarray:
        """Fire one BDPT batch using the pre-registered tri-group layout.

        Returns raw float32 (N, 16) EndpointRecord array.  The caller must
        follow up with scatter_bdpt_records() to accumulate surf/field_accum.
        """
        if not (hasattr(self.tracer, "bidirectional_packed") and emit_rays.size > 0):
            return np.zeros((0, 16), np.float32)
        # Size the buffer exactly to this batch's worst-case output.
        # Every ray can produce at most (max_bounces + 1) records.
        # Records are scattered and discarded immediately after; no session
        # cap is needed — overflow is impossible by construction.
        batch_max = int(np.sum(emit_rays)) * (self.max_bounces + 1)
        return self.tracer.bidirectional_packed(
            n_rays_per_emitter = np.ascontiguousarray(emit_rays, np.int32),
            max_bounces        = self.max_bounces,
            min_amplitude      = self.min_amplitude,
            seed               = int(seed),
            max_records        = batch_max,
        )

    def scatter_bdpt_records(self, recs: np.ndarray, sensor_group_id: int) -> None:
        """Scatter EndpointRecord rows into surf_accum (sensor hits) and field_accum (scene).

        EndpointRecord float32 column layout (bdpt_record.h, 64 bytes = 16 floats):
          col[0] subpath_id (uint32), col[1] band_id (uint32),
          col[2] group_id  (int32),   col[3] vertex_index (int32),
          col[4:7] pos xyz,           col[7] pathlen_m,
          col[8:11] dir xyz,          col[11] pdf,
          col[12] amp_re, col[13] amp_im, col[14] cos_theta, col[15] _pad

        Records with group_id == sensor_group_id are camera-sensor hits
        (surf_accum); all others are ambient/field paths (field_accum).
        self.accum is kept in sync as surf_accum + field_accum so that
        measured_radiant_exposure_J() and _density_need_map() remain valid.
        """
        if recs.shape[0] == 0:
            return
        recs = np.ascontiguousarray(recs, np.float32)
        n = recs.shape[0]
        H, W = int(self.cam.height), int(self.cam.width)

        # Reinterpret float32 storage as int32 to read integer-typed columns.
        col_i32  = recs.view(np.int32).reshape(n, 16)
        raw_band_ids = col_i32[:, 1].astype(np.int64)   # uint32 in struct
        group_ids    = col_i32[:, 2]                     # int32, signed

        # Discard records whose band_id is out of range — never silently wrap.
        valid_band = (raw_band_ids >= 0) & (raw_band_ids < self.n_bands)
        if not np.all(valid_band):
            n_bad = int(np.sum(~valid_band))
            print(f"  [bdpt] scatter: discarding {n_bad} records with out-of-range band_id")
            recs      = recs[valid_band]
            col_i32   = recs.view(np.int32).reshape(-1, 16)
            raw_band_ids = raw_band_ids[valid_band]
            group_ids    = col_i32[:, 2]
            n = recs.shape[0]
            if n == 0:
                return
        band_ids    = raw_band_ids
        subpath_ids = col_i32[:, 0].astype(np.int64)   # uint32 → int64
        vertex_idx  = col_i32[:, 3].astype(np.int64)   # int32, -1 = forward path

        amp_re = recs[:, 12].astype(np.float64)
        amp_im = recs[:, 13].astype(np.float64)
        amp    = np.sqrt(amp_re * amp_re + amp_im * amp_im)

        # Split sensor-group records by semantic:
        #   pixel_cone     — vertex_index >= 0: PIXEL_CONE sensor hit.
        #                    subpath_id encodes the pixel as py*W + px.
        #                    Use subpath_id directly for pixel mapping.
        #   forward_sensor — vertex_index < 0, group == sensor_group_id:
        #                    Forward/emission path that happened to hit a
        #                    sensor-group triangle.  Project world-pos.
        #   field          — group != sensor_group_id: ambient scene path.
        #                    Project world-pos; only in-frustum kept.
        sensor_group  = (group_ids == int(sensor_group_id))
        pixel_cone    = sensor_group & (vertex_idx >= 0)
        forward_snsr  = sensor_group & (vertex_idx < 0)
        field         = ~sensor_group

        # --- PIXEL_CONE scatter: subpath_id = py*W + px already. ---
        def _scatter_pixel_cone() -> None:
            m = pixel_cone & (subpath_ids >= 0) & (subpath_ids < H * W)
            if not np.any(m):
                n_bad = int(np.sum(pixel_cone)) - int(np.sum(m))
                if n_bad:
                    print(f"  [bdpt] scatter: discarding {n_bad} "
                          f"pixel_cone records with out-of-range subpath_id")
                return
            flat = band_ids[m] * (H * W) + subpath_ids[m]
            counts = np.bincount(flat, weights=amp[m],
                                 minlength=self.n_bands * H * W)
            self.surf_accum.ravel()[:] += counts.astype(np.float32)

        # --- World-pos projection for forward_sensor + field records. ---
        # Only compute the projection when there are records that need it.
        proj_mask = forward_snsr | field
        if np.any(proj_mask):
            pos = recs[:, 4:7].astype(np.float64)   # (N, 3) world-space
            cam_pos = np.asarray(self.cam.pos, np.float64)
            cam_f, cam_r, cam_u, tan_half_h, tan_half_v = self._camera_basis()
            v      = pos - cam_pos[None, :]
            depth  = v @ cam_f
            fwd_valid = depth > 1.0e-6
            safe_d = np.where(fwd_valid, depth, 1.0)
            x_ndc  = (v @ cam_r) / (safe_d * tan_half_h)
            y_ndc  = -(v @ cam_u) / (safe_d * tan_half_v)
            px_proj = ((x_ndc + 1.0) * 0.5 * W).astype(np.int64)
            py_proj = ((y_ndc + 1.0) * 0.5 * H).astype(np.int64)
            in_frame = fwd_valid & (px_proj >= 0) & (px_proj < W) & \
                       (py_proj >= 0) & (py_proj < H)
        else:
            in_frame = np.zeros(n, dtype=bool)
            px_proj  = np.zeros(n, dtype=np.int64)
            py_proj  = np.zeros(n, dtype=np.int64)

        def _scatter_proj(mask: np.ndarray, target: np.ndarray) -> None:
            m = mask & in_frame
            if not np.any(m):
                return
            flat = band_ids[m] * (H * W) + py_proj[m] * W + px_proj[m]
            counts = np.bincount(flat, weights=amp[m],
                                 minlength=self.n_bands * H * W)
            target.ravel()[:] += counts.astype(np.float32)

        _scatter_pixel_cone()
        _scatter_proj(forward_snsr, self.surf_accum)
        _scatter_proj(field,        self.field_accum)
        # Keep combined accum in sync for measured_radiant_exposure_J / density map.
        np.add(self.surf_accum, self.field_accum, out=self.accum)
        self.n_rays_accumulated += n

    def render_batch(self, n_rays: int, seed: int) -> None:
        # Ray preparation randomness is externalized here: every emissive
        # receives a fresh launch site and axis each batch before tracing.
        src_pos, src_dir = self._prepare_emissive_batch_sources(seed)
        if hasattr(self.tracer, "integrate_image_into_packed"):
            src_n_rays = self._allocate_source_rays(int(n_rays), src_pos,
                                                    mode=self._adaptive_mode,
                                                    seed=int(seed))
            self.tracer.integrate_image_into_packed(
                src_pos         = src_pos,
                src_dir         = src_dir,
                src_directivity = self.scene.src_directivity,
                src_n_rays      = src_n_rays,
                cam_pos         = self.cam.pos,
                cam_fwd         = self.cam.fwd,
                cam_up          = self.cam.up,
                out_image       = self.accum,
                fov_rad         = float(self.cam.fov_y_rad),
                max_bounces     = self.max_bounces,
                min_amplitude   = self.min_amplitude,
                seed            = int(seed),
            )
            self.n_rays_accumulated += int(np.sum(src_n_rays, dtype=np.int64))
            return

        self.tracer.integrate_image_into(
            src_pos         = src_pos,
            src_dir         = src_dir,
            src_directivity = self.scene.src_directivity,
            cam_pos         = self.cam.pos,
            cam_fwd         = self.cam.fwd,
            cam_up          = self.cam.up,
            out_image       = self.accum,
            fov_rad         = float(self.cam.fov_y_rad),
            n_rays          = int(n_rays),
            max_bounces     = self.max_bounces,
            min_amplitude   = self.min_amplitude,
            seed            = int(seed),
        )
        self.n_rays_accumulated += int(n_rays) * int(src_pos.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# GLSL backend — scaffold around demo_pluck_gl._gpu_ray_field_prebuilt
# ─────────────────────────────────────────────────────────────────────────────
class GlslExposureBackend(ExposureBackend):
    """Best-effort GLSL backend.

    Building the full SSBO context (BVH + ScaleContext + FilmStack +
    SensorAccumulator) outside ``demo_pluck_gl``'s harness is a substantial
    port; for now the backend either:
      (a) constructs the minimum SSBO set if a live GL context exists and
          the scene_builder fast path succeeds, or
      (b) falls back to a band-modulated copy of the C++ accumulator marked
          "GLSL pending" so the side-by-side window still functions and the
          calibration log still prints two backends.

    Either way, the public ``render_batch`` / ``finalize_image`` interface
    matches the C++ backend exactly.
    """
    name = "glsl"

    def __init__(self, scene: TracerScene, cam: PinholeCamera,
                 freq_hz: np.ndarray, *, mirror_from: Optional[ExposureBackend] = None,
                 **kw: Any):
        super().__init__(scene, cam, freq_hz, **kw)
        self._mirror = mirror_from
        self._gl_ready = False
        # TODO(phase2b): instantiate _gpu_ray_field_prebuilt + SensorAccumulator
        # against scene.{verts, mat_idx, mat_buf, bounds_*}, build a BVH from
        # csrc/kernels (already wrapped as _sk.build_bvh in older code), and
        # accumulate sensor pixels each batch.

    def render_batch(self, n_rays: int, seed: int) -> None:
        if self._gl_ready:
            return  # full path goes here once wired
        if self._mirror is not None:
            # Track the C++ accumulators so the right pane has something
            # spectrally meaningful to show during calibration testing.
            self.accum[:]       = self._mirror.accum
            self.surf_accum[:]  = self._mirror.surf_accum
            self.field_accum[:] = self._mirror.field_accum
            self.n_rays_accumulated = self._mirror.n_rays_accumulated


# ─────────────────────────────────────────────────────────────────────────────
# Exposure session — orchestrates plan → batches → gain training → frame
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExposureFrameResult:
    frame_index:        int
    backend:            str
    plan:               dict
    n_rays_emitted:     int
    n_batches:          int
    measured_H_J:       float
    target_H_J:         float
    gain_linear:        float
    gain_db:            float
    virtual_t_s:        float
    photons_per_pixel:  float
    snr_estimate:       float
    image_path:         str
    image16_path:       str
    image_linear_path:  str
    field_object_path:  str
    surface_object_path:str
    sensor_object_path: str
    field_capture_grid_path: str
    field_capture_strikes_path: str
    summary_path:       str
    frame_config_summary: dict
    image_data:         Optional[np.ndarray] = None  # (H, W, 3) float32 [0,1]
    image16_data:       Optional[np.ndarray] = None  # (H, W, 3) uint16 for 16-bit
    field_integrated_data: Optional[np.ndarray] = None  # (B, H, W)
    surface_integrated_data: Optional[np.ndarray] = None  # (B, H, W)
    sensor_photons_data: Optional[np.ndarray] = None  # (H, W)
    sensor_snr_data: Optional[np.ndarray] = None  # (H, W)


@dataclass
class IntegralSplitConfig:
    """Controls how much energy is integrated vs retained as bookkeeping."""
    field_integrate_frac: float = 0.35
    field_bookkeep_frac:  float = 0.65
    surface_integrate_frac: float = 0.85
    surface_bookkeep_frac:  float = 0.15
    hdr_white_percentile: float = 99.8


@dataclass
class CameraVisibilityConfig:
    """Wrapper around tracer camera-visibility policies for this session."""
    camera_vis_mode: int = int(getattr(_sk, "RT_CAM_VIS_AS_IS", 0))
    transparent_mode: int = int(getattr(_sk, "RT_CAM_TRANSPARENCY_BLOCK", 0))
    depth_cull_enabled: bool = False
    depth_cull_m: float = 0.0


@dataclass
class ConvergenceConfig:
    """Early-stop controls for exposure batch convergence."""
    enabled: bool = True
    drive_batches: bool = True
    target_pct: float = 99.99
    max_rel_drift: float = 1.0e-4
    check_every_batches: int = 1
    min_batches: int = 4
    hold_checks: int = 3
    probe_count: int = 8192
    max_batches: int = 0


@dataclass
class FieldCaptureConfig:
    """Full-complex field capture controls for exposure tracing."""
    enabled: bool = True
    grid_kind: str = "regular"  # "regular" | "kdtree"
    nx: int = 128
    ny: int = 128
    nz: int = 128
    capture_strikes: bool = True
    max_strikes: int = 1_000_000


@dataclass
class SurfaceSplineConfig:
    """Quadratic POLY_BARY surface spline fitter settings."""
    enabled: bool = False
    ridge_lambda: float = 0.0       # Tikhonov regularisation (0 = off)
    n_threads: int = 0              # 0 = hardware_concurrency
    fit_all_tris: bool = False      # True = per-tri groups; False = mean group


@dataclass
class ParametricSdfConfig:
    """Parametric SDF-style presets mapped onto POLY_BARY payloads."""
    enabled: bool = False
    model: str = "off"              # off | saddle | sphere | mixed
    saddle_amplitude_m: float = 2.0e-3
    sphere_radius_m: float = 0.12
    neighborhood_margin_uv: float = 8.0e-2


# ── Camera-visibility mode constants (resolved lazily so module loads without ext) ──
_CAM_VIS_AS_IS      = int(getattr(_sk, "RT_CAM_VIS_AS_IS",      0))
_CAM_VIS_DIRECT_HIT = int(getattr(_sk, "RT_CAM_VIS_DIRECT_HIT", 1))
_CAM_VIS_FULL_MARCH = int(getattr(_sk, "RT_CAM_VIS_FULL_MARCH", 2))
_CAM_TRANSP_BLOCK   = int(getattr(_sk, "RT_CAM_TRANSPARENCY_BLOCK", 0))
_CAM_TRANSP_XRAY    = int(getattr(_sk, "RT_CAM_TRANSPARENCY_XRAY",  1))


@dataclass
class FrameConfig:
    """Full per-frame feature set built by _build_frame_config."""
    field_capture:     FieldCaptureConfig
    camera_visibility: CameraVisibilityConfig
    surface_spline:    SurfaceSplineConfig
    parametric_sdf:    ParametricSdfConfig
    integral_split:    IntegralSplitConfig
    description:       str = ""
    detail_level:      int = 0   # 0-5; drives HUD verbosity


def _poly_bary_coeffs_sdf(model: str,
                          tri_id: int,
                          saddle_amplitude_m: float,
                          sphere_radius_m: float) -> np.ndarray:
    """Build float64[6] POLY_BARY coeffs for parametric SDF-style presets."""
    m = str(model).strip().lower()
    if m == "mixed":
        m = "sphere" if (int(tri_id) % 2) == 0 else "saddle"

    if m == "sphere":
        # Near-center paraboloid approximation of sphere SDF displacement.
        # delta ~ k * ((u-1/3)^2 + (v-1/3)^2), k ~= 1/(2R)
        r = max(float(sphere_radius_m), 1.0e-6)
        k = 0.5 / r
        c0 = (2.0 / 9.0) * k
        cu = -(2.0 / 3.0) * k
        cv = -(2.0 / 3.0) * k
        cuu = k
        cuv = 0.0
        cvv = k
        return np.asarray([c0, cu, cv, cuu, cuv, cvv], dtype=np.float64)

    # Default to saddle if unsupported string is supplied.
    # delta = a * ((u-1/3)^2 - (v-1/3)^2)
    a = float(saddle_amplitude_m)
    c0 = 0.0
    cu = -(2.0 / 3.0) * a
    cv = (2.0 / 3.0) * a
    cuu = a
    cuv = 0.0
    cvv = -a
    return np.asarray([c0, cu, cv, cuu, cuv, cvv], dtype=np.float64)


def _resolve_native_parametric_payload(raw: Any) -> dict[str, Any]:
    """Normalize plugin output to register_tri_group parametric_surface payload.

    Accepted forms:
    - dict: {"kind": int|str, "coeffs": float64[N]}
    - array-like: float64[6] interpreted as POLY_BARY
    """
    if isinstance(raw, dict):
        kind_raw = raw.get("kind", "poly_bary")
        coeffs = np.asarray(raw.get("coeffs", []), dtype=np.float64).ravel()
    else:
        kind_raw = "poly_bary"
        coeffs = np.asarray(raw, dtype=np.float64).ravel()

    if coeffs.ndim != 1 or coeffs.size == 0:
        raise RuntimeError("parametric plugin returned empty coeff payload")

    if isinstance(kind_raw, str):
        k = kind_raw.strip().lower()
        if k in ("poly_bary", "poly", "bary"):
            kind = int(getattr(_sk, "TRI_PARAM_SURFACE_POLY_BARY", 1))
            if coeffs.size != 6:
                raise RuntimeError(f"POLY_BARY expects 6 coeffs, got {coeffs.size}")
        elif k in ("sdf_saddle", "saddle"):
            kind = int(getattr(_sk, "TRI_PARAM_SURFACE_SDF_SADDLE", -1))
            if kind < 0:
                raise RuntimeError("TRI_PARAM_SURFACE_SDF_SADDLE unavailable in native extension")
            if coeffs.size != 2:
                raise RuntimeError(f"SDF_SADDLE expects 2 coeffs [amp, margin], got {coeffs.size}")
        elif k in ("sdf_sphere", "sphere"):
            kind = int(getattr(_sk, "TRI_PARAM_SURFACE_SDF_SPHERE", -1))
            if kind < 0:
                raise RuntimeError("TRI_PARAM_SURFACE_SDF_SPHERE unavailable in native extension")
            if coeffs.size != 2:
                raise RuntimeError(f"SDF_SPHERE expects 2 coeffs [radius, margin], got {coeffs.size}")
        else:
            raise RuntimeError(f"unknown parametric kind '{kind_raw}'")
    else:
        kind = int(kind_raw)

    return {"kind": int(kind), "coeffs": coeffs.astype(np.float64, copy=False)}


def _build_frame_config(frame_idx: int,
                        n_frames_total: int,
                        base_split: IntegralSplitConfig) -> FrameConfig:
    """Build the per-frame feature config according to the progressive schedule.

    When n_frames_total == 1 only level 0 (baseline) is used.  Each additional
    frame slot unlocks the next feature level up to MAX_LEVELS - 1.

    Level 0  aggressive bootstrap — field capture + strike scatter + full march
    Level 1  field regular 64³ + strike capture
    Level 2  field regular 96³ + DIRECT_HIT + emissive-only spline
    Level 3  field kdtree 64³ + DIRECT_HIT + XRAY + full-mesh spline
    Level 4  field regular 128³ + FULL_MARCH + XRAY + depth cull + ridge spline
    Level 5  field kdtree 128³ + FULL_MARCH + XRAY + max ridge + field-heavy
    """
    MAX_LEVELS = 8
    n_levels = min(n_frames_total, MAX_LEVELS)
    level = frame_idx % max(1, n_levels)

    fi = float(base_split.field_integrate_frac)
    fb = float(base_split.field_bookkeep_frac)
    si = float(base_split.surface_integrate_frac)
    sb = float(base_split.surface_bookkeep_frac)
    wp = float(base_split.hdr_white_percentile)

    def _split(dfi: float = 0.0, dsi: float = 0.0, dwp: float = 0.0) -> IntegralSplitConfig:
        return IntegralSplitConfig(
            field_integrate_frac   = float(np.clip(fi + dfi, 0.0, 1.0)),
            field_bookkeep_frac    = float(np.clip(fb - dfi, 0.0, 1.0)),
            surface_integrate_frac = float(np.clip(si + dsi, 0.0, 1.0)),
            surface_bookkeep_frac  = float(np.clip(sb - dsi, 0.0, 1.0)),
            hdr_white_percentile   = float(max(wp + dwp, 90.0)),
        )

    schedules: list[FrameConfig] = [
        # level 0 — fastest baseline: AS_IS cam, coarse regular grid, no extras
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=48, ny=48, nz=48,
                capture_strikes=True, max_strikes=200_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_AS_IS,
                transparent_mode = _CAM_TRANSP_BLOCK,
            ),
            surface_spline    = SurfaceSplineConfig(enabled=False),
            parametric_sdf    = ParametricSdfConfig(enabled=False),
            integral_split    = _split(),
            description       = "baseline · AS_IS · regular-48³ · no extras",
            detail_level      = 0,
        ),
        # level 1 — add parametric saddle on emissive tris + DIRECT_HIT
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=64, ny=64, nz=64,
                capture_strikes=True, max_strikes=350_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_DIRECT_HIT,
                transparent_mode = _CAM_TRANSP_BLOCK,
            ),
            surface_spline    = SurfaceSplineConfig(enabled=False),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="saddle", saddle_amplitude_m=1.5e-3,
                neighborhood_margin_uv=8.0e-2,
            ),
            integral_split    = _split(dfi=0.05),
            description       = "DIRECT_HIT · regular-64³ · parametric saddle",
            detail_level      = 1,
        ),
        # level 2 — parametric sphere + emissive-only spline + DIRECT_HIT
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=80, ny=80, nz=80,
                capture_strikes=True, max_strikes=500_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_DIRECT_HIT,
                transparent_mode = _CAM_TRANSP_BLOCK,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=False, ridge_lambda=0.0,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="sphere", sphere_radius_m=0.12,
                neighborhood_margin_uv=8.0e-2,
            ),
            integral_split    = _split(dfi=0.10),
            description       = "DIRECT_HIT · regular-80³ · spline emissive · parametric sphere",
            detail_level      = 2,
        ),
        # level 3 — DIRECT_HIT + XRAY + kdtree 64³ + mixed parametric + all spline
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="kdtree",
                nx=64, ny=64, nz=64,
                capture_strikes=True, max_strikes=750_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_DIRECT_HIT,
                transparent_mode = _CAM_TRANSP_XRAY,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=True, ridge_lambda=0.0,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="mixed", saddle_amplitude_m=2.0e-3,
                sphere_radius_m=0.12, neighborhood_margin_uv=8.0e-2,
            ),
            integral_split    = _split(dfi=0.15, dsi=0.05),
            description       = "DIRECT_HIT+XRAY · kdtree-64³ · spline all · parametric mixed",
            detail_level      = 3,
        ),
        # level 4 — FULL_MARCH + regular 96³ + parametric saddle + depth cull
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=96, ny=96, nz=96,
                capture_strikes=True, max_strikes=1_000_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode    = _CAM_VIS_FULL_MARCH,
                transparent_mode   = _CAM_TRANSP_BLOCK,
                depth_cull_enabled = True,
                depth_cull_m       = 80.0,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=False, ridge_lambda=0.0,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="saddle", saddle_amplitude_m=2.0e-3,
                neighborhood_margin_uv=8.0e-2,
            ),
            integral_split    = _split(dfi=0.20, dsi=0.08, dwp=-0.2),
            description       = "FULL_MARCH · regular-96³ · emissive spline · parametric saddle · depth 80m",
            detail_level      = 4,
        ),
        # level 5 — FULL_MARCH + regular 128³ + ridge spline + parametric mixed + depth cull
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=128, ny=128, nz=128,
                capture_strikes=True, max_strikes=1_000_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode    = _CAM_VIS_FULL_MARCH,
                transparent_mode   = _CAM_TRANSP_BLOCK,
                depth_cull_enabled = True,
                depth_cull_m       = 50.0,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=True, ridge_lambda=1.0e-4,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="mixed", saddle_amplitude_m=2.0e-3,
                sphere_radius_m=0.12, neighborhood_margin_uv=8.0e-2,
            ),
            integral_split    = _split(dfi=0.25, dsi=0.10, dwp=-0.3),
            description       = "FULL_MARCH · regular-128³ · spline ridge 1e-4 · parametric mixed · depth 50m",
            detail_level      = 5,
        ),
        # level 6 — FULL_MARCH + XRAY + kdtree 128³ + ridge spline + parametric mixed
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="kdtree",
                nx=128, ny=128, nz=128,
                capture_strikes=True, max_strikes=1_000_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode    = _CAM_VIS_FULL_MARCH,
                transparent_mode   = _CAM_TRANSP_XRAY,
                depth_cull_enabled = True,
                depth_cull_m       = 100.0,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=True, ridge_lambda=1.0e-3,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="mixed", saddle_amplitude_m=2.5e-3,
                sphere_radius_m=0.12, neighborhood_margin_uv=6.0e-2,
            ),
            integral_split    = _split(dfi=0.30, dsi=0.12, dwp=-0.6),
            description       = "FULL_MARCH+XRAY · kdtree-128³ · spline ridge 1e-3 · parametric mixed",
            detail_level      = 6,
        ),
        # level 7 — most expensive: FULL_MARCH + XRAY + kdtree 192³ + max ridge + heavy field
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="kdtree",
                nx=192, ny=192, nz=192,
                capture_strikes=True, max_strikes=2_000_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode    = _CAM_VIS_FULL_MARCH,
                transparent_mode   = _CAM_TRANSP_XRAY,
                depth_cull_enabled = True,
                depth_cull_m       = 150.0,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=True, ridge_lambda=1.0e-3,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="mixed", saddle_amplitude_m=3.0e-3,
                sphere_radius_m=0.12, neighborhood_margin_uv=5.0e-2,
            ),
            integral_split    = _split(dfi=0.40, dsi=0.15, dwp=-0.8),
            description       = "FULL_MARCH+XRAY · kdtree-192³ · spline max ridge · parametric mixed · deep",
            detail_level      = 7,
        ),
    ]
    return schedules[level]


@dataclass
class FieldIntegralObject:
    backend: str
    frame_index: int
    freq_hz: np.ndarray
    integrated_bands: np.ndarray
    bookkeeping_bands: np.ndarray
    metrics: dict


@dataclass
class SurfaceIntegralObject:
    backend: str
    frame_index: int
    freq_hz: np.ndarray
    integrated_bands: np.ndarray
    bookkeeping_bands: np.ndarray
    metrics: dict


@dataclass
class SensorGeometryConfig:
    """Camera sensor plane geometry for BDPT backward integration."""
    enabled: bool = True
    use_pixel_cone_sampling: bool = True
    n_aperture_samples: int = 16
    aperture_stop_group_id: int = -1


@dataclass
class SensorIntegralObject:
    """Sensor-plane photometric integration: rays reaching the camera back."""
    backend: str
    frame_index: int
    optics: CameraOptics
    film: FilmExposure
    n_pixels: int
    sensor_w_m: float
    sensor_h_m: float
    pixel_pitch_m: float
    focal_m: float
    aperture_radius_m: float
    qe_peak: float
    photons_per_pixel: np.ndarray    # (H, W) float32 accumulated photon count
    electrons_per_pixel: np.ndarray  # (H, W) with QE applied
    noise_floor_e: float
    full_well_e: int
    snr_linear: np.ndarray           # (H, W) per-pixel SNR in linear units
    peak_snr: float
    mean_snr: float
    metrics: dict

    def to_dict(self) -> dict:
        """Serialize to JSON-safe format for frame summary."""
        return {
            "n_pixels": int(self.n_pixels),
            "sensor_w_m": float(self.sensor_w_m),
            "sensor_h_m": float(self.sensor_h_m),
            "pixel_pitch_m": float(self.pixel_pitch_m),
            "qe_peak": float(self.qe_peak),
            "peak_snr": float(self.peak_snr),
            "mean_snr": float(self.mean_snr),
            "metrics": self.metrics,
        }


@dataclass
class IntegrationSnapshot:
    """Live or frame-final integration snapshot for one backend."""
    backend: str
    frame_index: int
    batch_index: int
    measured_H_J: float
    target_H_J: float
    gain_linear: float
    image_data: np.ndarray
    rgb_linear: np.ndarray
    field_integrated_data: np.ndarray
    surface_integrated_data: np.ndarray
    sensor_photons_data: Optional[np.ndarray] = None
    sensor_snr_data: Optional[np.ndarray] = None
    sensor_rgb_data: Optional[np.ndarray] = None
    mode: str = "stream"


# ─────────────────────────────────────────────────────────────────────────────
# Sensor registration helpers for bidirectional integration
# ─────────────────────────────────────────────────────────────────────────────

def _build_camera_sensor_descriptor(optics: CameraOptics,
                                    width_px: int,
                                    height_px: int,
                                    n_aperture_samples: int = 16) -> Optional[dict]:
    """Build a CameraSensor descriptor for PIXEL_CONE BDPT sampling.
    
    Parameters:
    - optics: CameraOptics with camera geometry
    - width_px, height_px: sensor resolution in pixels
    - n_aperture_samples: number of aperture samples per ray
    
    Returns a dict compatible with bdpt_integrator.TriangleGroup.sensor_camera
    parameter, or None if sensor registration is unavailable.
    """
    if not _HAS_BDPT_INTEGRATION or CameraSensor is None:
        return None
    
    # Camera frame: position at origin, looking at +Z, up = +Y
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # toward scene
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    
    # Physical sensor dimensions
    sensor_w_m = optics.sensor_w_mm * 1.0e-3
    sensor_h_m = optics.sensor_h_mm * 1.0e-3
    
    # Focal length and aperture
    focal_m = optics.focal_mm * 1.0e-3
    aperture_radius_m = focal_m / (2.0 * optics.f_number)
    
    # Build the descriptor as a dict (pybind expects dict, not CameraSensor)
    descriptor = {
        "pos": pos,
        "fwd": fwd,
        "up": up,
        "sensor_w_m": float(sensor_w_m),
        "sensor_h_m": float(sensor_h_m),
        "focal_m": float(focal_m),
        "aperture_radius_m": float(aperture_radius_m),
        "n_px": int(width_px),
        "n_py": int(height_px),
        "n_aperture_samples": int(n_aperture_samples),
        "aperture_stop_group_id": -1,  # no blade masking for now
    }
    return descriptor


def _register_sensor_group(tracer: Any,
                          optics: CameraOptics,
                          width_px: int,
                          height_px: int) -> int:
    """Register the sensor plane as a SENSOR TriGroup for BDPT backward pass.
    
    Returns the assigned group_id, or -1 if registration failed.
    """
    if not _HAS_BDPT_INTEGRATION:
        return -1
    
    try:
        # Build flat sensor geometry
        positions, uvs, tris = _build_flat_sensor_geometry(optics, height_px, width_px)
        
        # Triangle indices for the sensor group (all tris in flat 2-tri quad)
        sensor_tri_indices = np.array([0, 1], dtype=np.int32)
        
        # Build sensor descriptor
        sensor_desc = _build_camera_sensor_descriptor(optics, width_px, height_px)
        if sensor_desc is None:
            return -1
        
        # Create TriangleGroup descriptor
        group = TriangleGroup(
            role_bits=int(TRI_GROUP_ROLE_SENSOR),
            tri_indices=sensor_tri_indices,
            sample_policy=int(TRI_GROUP_SAMPLE_PIXEL_CONE),
            sensor_camera=sensor_desc,
        )
        
        # Register with the tracer
        group_id = group.register_with(tracer)
        return int(group_id)
    
    except Exception as e:
        print(f"WARNING: sensor group registration failed: {e}")
        return -1


class ExposureSession:
    def __init__(self, *,
                 optics: CameraOptics,
                 film:   FilmExposure,
                 width:  int,
                 height: int,
                 total_rays: int,
                 rays_per_batch: int,
                 max_bounces: int = 4,
                 freq_hz: Optional[np.ndarray] = None,
                 backends: tuple[str, ...] = ("cpp",),
                 out_dir: str = "exposures",
                 integrator: str = "bdpt",
                 bdpt_records_cap: int = 0,
                 scene_mode: str = "orbiters",
                 scene_mode_schedule: Optional[tuple[str, ...]] = None,
                 profile_enabled: bool = False,
                 integral_split: Optional[IntegralSplitConfig] = None,
                 camera_visibility: Optional[CameraVisibilityConfig] = None,
                 field_capture: Optional[FieldCaptureConfig] = None,
                 convergence: Optional[ConvergenceConfig] = None,
                 adaptive_allocation_mode: str = "stochastic",
                 n_frames_planned: int = 1,
                 output_width: Optional[int] = None,
                 output_height: Optional[int] = None,
                 output_oversample_stencil: str = "box",
                 show_hud: bool = True,
                 rgb_source: str = "sensor",
                 sensor_film_slots: Optional[list[tuple[int, int]]] = None,
                 save_files: bool = False,
                 bdpt_intermediate_mode: str = "file",
                 bdpt_intermediate_max_bytes: int = 0,
                 retain_bdpt_intermediate: bool = False,
                 bdpt_intermediate_dir: Optional[str] = None):
        self.optics = optics
        self.film   = film
        self.width  = int(width)
        self.height = int(height)
        self.total_rays = int(total_rays)
        self.rays_per_batch = max(1, int(rays_per_batch))
        self.max_bounces = int(max_bounces)
        self.freq_hz = np.asarray(freq_hz if freq_hz is not None else DEFAULT_FREQ_HZ,
                                  np.float64)
        self.backends_requested = tuple(backends)
        self.out_dir = out_dir
        self.integrator = str(integrator)
        requested_bdpt_cap = int(bdpt_records_cap)
        self.scene_mode = str(scene_mode)
        self.scene_mode_schedule = tuple(str(s) for s in scene_mode_schedule) if scene_mode_schedule else None
        self.profile_enabled = bool(profile_enabled)
        self._profiler = StageProfiler(self.profile_enabled)
        self.integral_split = (integral_split
                       if integral_split is not None
                       else IntegralSplitConfig())
        self.camera_visibility = (camera_visibility
                      if camera_visibility is not None
                      else CameraVisibilityConfig())
        self.field_capture = (field_capture
                      if field_capture is not None
                      else FieldCaptureConfig())
        self.convergence = (convergence
                 if convergence is not None
                 else ConvergenceConfig())
        self.adaptive_allocation_mode = str(adaptive_allocation_mode).strip().lower()
        if self.adaptive_allocation_mode not in ("stochastic", "quota", "uniform"):
            self.adaptive_allocation_mode = "stochastic"
        self.n_frames_planned = max(1, int(n_frames_planned))
        self.output_width = int(output_width) if output_width is not None else int(self.width)
        self.output_height = int(output_height) if output_height is not None else int(self.height)
        self.output_width = max(1, self.output_width)
        self.output_height = max(1, self.output_height)
        self.output_oversample_stencil = str(output_oversample_stencil).strip().lower()
        if self.output_oversample_stencil not in ("box", "polar"):
            self.output_oversample_stencil = "box"
        self.show_hud = bool(show_hud)
        self.rgb_source = str(rgb_source).strip().lower()
        if self.rgb_source not in ("accum", "endpoint", "sensor"):
            self.rgb_source = "sensor"
        self.save_files = bool(save_files)
        self.bdpt_intermediate_mode = str(bdpt_intermediate_mode).strip().lower()
        if self.bdpt_intermediate_mode not in ("memory", "file"):
            self.bdpt_intermediate_mode = "file"
        self.bdpt_intermediate_max_bytes = max(0, int(bdpt_intermediate_max_bytes))
        self.retain_bdpt_intermediate = bool(retain_bdpt_intermediate)
        self.bdpt_intermediate_dir = str(bdpt_intermediate_dir or out_dir)
        if requested_bdpt_cap > 0:
            self.bdpt_records_cap = max(4096, requested_bdpt_cap)
        else:
            self.bdpt_records_cap = max(4096, _bdpt_cap_from_bytes(self.bdpt_intermediate_max_bytes))
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(self.bdpt_intermediate_dir, exist_ok=True)

        # Load sensor/film database (T2: ExposureSession integration)
        self._sensor_film_db = SensorFilmDatabase.instance()
        self._sensor_film_tensors = self._sensor_film_db.build_tensors()
        
        if sensor_film_slots is None:
            # Default: slot 0 with Canon EOS R6 + Kodak Portra 400
            sensor_film_slots = [(0, 0)] + [(-1, -1)] * (MAX_SENSOR_FILM_SLOTS - 1)
        
        self.sensor_film_slots = sensor_film_slots
        self._sensor_film_metadata = []  # For HUD and reporting
        
        # Build metadata for each slot
        for slot_id, (sensor_id, film_id) in enumerate(self.sensor_film_slots):
            if sensor_id >= 0 and film_id >= 0:
                sensor_name = self._sensor_film_db._sensor_order[sensor_id] if sensor_id < len(self._sensor_film_db._sensor_order) else "unknown"
                film_name = self._sensor_film_db._film_order[film_id] if film_id < len(self._sensor_film_db._film_order) else "unknown"
                sensor_record = self._sensor_film_tensors['sensor'][sensor_id]
                meta = {
                    "slot_id": slot_id,
                    "sensor_id": sensor_id,
                    "sensor_name": sensor_name,
                    "film_id": film_id,
                    "film_name": film_name,
                    "active": True,
                    "qe_peak": float(sensor_record[8]),  # qe_peak at offset 8
                    "read_noise_e": float(sensor_record[10]),  # read_noise_e at offset 10
                    "dark_current_e_s": float(sensor_record[11]),  # dark_current_e_s at offset 11
                }
                self._sensor_film_metadata.append(meta)
            else:
                self._sensor_film_metadata.append({
                    "slot_id": slot_id,
                    "sensor_id": -1,
                    "film_id": -1,
                    "active": False,
                })
        
        print(f"  [sensor_film] loaded {len(self._sensor_film_db._sensor_order)} sensors, "
              f"{len(self._sensor_film_db._film_order)} films, {sum(1 for m in self._sensor_film_metadata if m['active'])} active slots")
        self._sensor_film_lock = threading.Lock()

        if self.integrator == "bdpt":
            cap_gb = (self.bdpt_records_cap * _BDPT_RECORD_BYTES) / float(1024 ** 3)
            print(f"  [bdpt] records cap={self.bdpt_records_cap:_} (~{cap_gb:.2f} GiB payload)")
        print(
            f"  [conv] enabled={bool(self.convergence.enabled)} "
            f"drive_batches={bool(self.convergence.drive_batches)} "
            f"target={self.convergence.target_pct:.4f}% "
            f"H_error<= {float(self.convergence.max_rel_drift):.3e} "
            f"every={max(1, int(self.convergence.check_every_batches))} "
            f"min={max(1, int(self.convergence.min_batches))} "
            f"hold={max(1, int(self.convergence.hold_checks))} "
            f"max_batches={int(self.convergence.max_batches)}"
        )
        print(f"  [adaptive] allocation={self.adaptive_allocation_mode}")
        print(f"  [rgb] source={self.rgb_source}")
        print(f"  [res] render={self.width}x{self.height} output={self.output_width}x{self.output_height} stencil={self.output_oversample_stencil}")

        self._rng_seed = 1
        self._frame_index = 0
        self._sensor_group_id = -1  # BDPT sensor group ID for this frame
        self._last_bdpt_records: Optional[np.ndarray] = None
        self._last_target_photons_per_pixel: float = 0.0
        self._last_bdpt_records_path: Optional[str] = None
        self._last_bdpt_records_is_temp: bool = False
        self._last_bdpt_emit_counts: Optional[np.ndarray] = None
        self._bdpt_emit_rays_total: Optional[np.ndarray] = None

    def swap_sensor_film_slot(self, slot_idx: int, sensor_delta: int = 0, film_delta: int = 0) -> dict:
        """Cycle sensor and/or film on slot_idx by delta steps (wraps around).
        Returns the new metadata dict for the slot."""
        n_sensors = len(self._sensor_film_db._sensor_order)
        n_films   = len(self._sensor_film_db._film_order)
        if n_sensors == 0 or n_films == 0:
            return {}
        with self._sensor_film_lock:
            s_id, f_id = self.sensor_film_slots[slot_idx]
            if s_id < 0: s_id = 0
            if f_id < 0: f_id = 0
            s_id = (s_id + sensor_delta) % n_sensors
            f_id = (f_id + film_delta)   % n_films
            self.sensor_film_slots[slot_idx] = (s_id, f_id)
            sensor_name = self._sensor_film_db._sensor_order[s_id]
            film_name   = self._sensor_film_db._film_order[f_id]
            sensor_record = self._sensor_film_tensors['sensor'][s_id]
            meta = {
                "slot_id": slot_idx,
                "sensor_id": s_id,
                "sensor_name": sensor_name,
                "film_id": f_id,
                "film_name": film_name,
                "active": True,
                "qe_peak": float(sensor_record[8]),
                "read_noise_e": float(sensor_record[10]),
                "dark_current_e_s": float(sensor_record[11]),
            }
            self._sensor_film_metadata[slot_idx] = meta
        return meta

    def _release_last_bdpt_records(self) -> None:
        arr = self._last_bdpt_records
        self._last_bdpt_records = None
        if isinstance(arr, np.memmap):
            try:
                arr.flush()
            except Exception:
                pass
            try:
                arr._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        gc.collect()

    def _scene_mode_for_frame(self) -> str:
        if self.scene_mode_schedule:
            idx = min(max(self._frame_index, 0), len(self.scene_mode_schedule) - 1)
            return self.scene_mode_schedule[idx]
        return self.scene_mode

    def _cleanup_temp_bdpt_file(self) -> None:
        if not self._last_bdpt_records_is_temp or not self._last_bdpt_records_path:
            return
        self._release_last_bdpt_records()
        try:
            os.remove(self._last_bdpt_records_path)
        except OSError:
            pass
        self._last_bdpt_records_path = None
        self._last_bdpt_records_is_temp = False

    def _stage_bdpt_records(self, recs: np.ndarray) -> None:
        self._cleanup_temp_bdpt_file()
        recs32 = np.ascontiguousarray(recs, dtype=np.float32)
        self._last_bdpt_records_path = None
        self._last_bdpt_records_is_temp = False

        if self.bdpt_intermediate_mode != "file":
            self._last_bdpt_records = recs32
            return

        est_bytes = int(recs32.nbytes)
        if self.bdpt_intermediate_max_bytes > 0 and est_bytes > self.bdpt_intermediate_max_bytes:
            lim_gb = self.bdpt_intermediate_max_bytes / float(1024 ** 3)
            cur_gb = est_bytes / float(1024 ** 3)
            print(f"  [warn] BDPT intermediate {cur_gb:.2f}GB exceeds cap {lim_gb:.2f}GB; using memory")
            self._last_bdpt_records = recs32
            return

        frame_tag = f"{self._frame_index:04d}"
        if self.retain_bdpt_intermediate:
            path = os.path.join(self.bdpt_intermediate_dir, f"bdpt_records_{frame_tag}.npy")
            is_temp = False
        else:
            fd, path = tempfile.mkstemp(
                prefix=f"bdpt_records_{frame_tag}_",
                suffix=".npy",
                dir=self.bdpt_intermediate_dir,
            )
            os.close(fd)
            is_temp = True

        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=recs32.shape)
        mm[:] = recs32
        mm.flush()
        del mm

        self._last_bdpt_records = np.load(path, mmap_mode="r")
        self._last_bdpt_records_path = path
        self._last_bdpt_records_is_temp = is_temp

        size_gb = est_bytes / float(1024 ** 3)
        keep_msg = "retained" if self.retain_bdpt_intermediate else "ephemeral"
        print(f"  bdpt records staged ({keep_msg}) → {path}  [{size_gb:.2f}GB]")

    def _allocate_bdpt_emit_rays(self,
                                 cpp_back: CppExposureBackend,
                                 emitter_centers: np.ndarray,
                                 target_total_rays: int,
                                 seed: int) -> np.ndarray:
        """Adaptive per-emitter allocation for BDPT using endpoint uncertainty.

        Default mode (stochastic) uses EndpointRecord-derived uncertainty and
        novelty; quota/uniform remain explicit fallback options.
        """
        n_emit = int(emitter_centers.shape[0])
        if n_emit <= 0:
            return np.zeros((0,), dtype=np.int32)

        total_rays = max(n_emit, int(target_total_rays))
        mode = str(self.adaptive_allocation_mode).strip().lower()
        if mode not in ("stochastic", "quota", "uniform"):
            mode = "stochastic"

        if mode in ("quota", "uniform"):
            per_emit = max(1, total_rays // n_emit)
            return cpp_back._allocate_source_rays(
                n_rays=per_emit,
                src_pos=emitter_centers,
                mode=mode,
                seed=seed,
            )

        # Smart default: uncertainty + completeness pressure + novelty.
        if self._bdpt_emit_rays_total is None or self._bdpt_emit_rays_total.shape[0] != n_emit:
            self._bdpt_emit_rays_total = np.zeros((n_emit,), dtype=np.float64)

        scores = np.ones((n_emit,), dtype=np.float64)
        recs = self._last_bdpt_records
        prev_counts = self._last_bdpt_emit_counts
        if recs is not None and recs.size > 0 and prev_counts is not None and prev_counts.size == n_emit:
            rec_arr = np.asarray(recs, dtype=np.float32)
            subpath = rec_arr[:, 0].astype(np.int64, copy=False)
            amp_re = rec_arr[:, 12].astype(np.float64, copy=False)
            amp_im = rec_arr[:, 13].astype(np.float64, copy=False)
            e = amp_re * amp_re + amp_im * amp_im

            prefix = np.cumsum(prev_counts.astype(np.int64, copy=False))
            emit_idx = np.searchsorted(prefix, subpath, side="right")
            valid = (emit_idx >= 0) & (emit_idx < n_emit)

            if np.any(valid):
                idx = emit_idx[valid]
                ev = e[valid]
                cnt = np.bincount(idx, minlength=n_emit).astype(np.float64)
                s1 = np.bincount(idx, weights=ev, minlength=n_emit).astype(np.float64)
                s2 = np.bincount(idx, weights=ev * ev, minlength=n_emit).astype(np.float64)

                mean = np.divide(s1, np.maximum(1.0, cnt))
                var = np.maximum(0.0, np.divide(s2, np.maximum(1.0, cnt)) - mean * mean)
                uncertainty = np.sqrt(var) / np.sqrt(np.maximum(1.0, cnt))
                completeness = 1.0 / np.sqrt(1.0 + cnt)
                novelty = 1.0 / np.sqrt(1.0 + self._bdpt_emit_rays_total)

                def _norm(v: np.ndarray) -> np.ndarray:
                    vmax = float(np.max(v)) if v.size else 0.0
                    if vmax <= 1.0e-20:
                        return np.ones_like(v, dtype=np.float64)
                    return np.clip(v / vmax, 0.0, 1.0)

                scores = (0.50 * _norm(uncertainty) +
                          0.30 * _norm(completeness) +
                          0.20 * _norm(novelty))
                scores = np.maximum(scores, 1.0e-9)

        weights = scores / max(1.0e-20, float(np.sum(scores)))
        rng = np.random.default_rng(int(seed))
        draw = rng.multinomial(int(total_rays - n_emit), weights)
        out = np.ones((n_emit,), dtype=np.int32)
        out += draw.astype(np.int32, copy=False)
        self._bdpt_emit_rays_total += out.astype(np.float64)
        return out

    # ── Build per-frame plan + scene + backends ──────────────────────────
    def _build_plan(self, scene: TracerScene) -> RayDispatchPlan:
        radiance = scene.scene_radiance_W_sr_m2()
        if radiance <= 0.0:
            # Avoid divide-by-zero: bias to dim moonlight (~1e-4 W·sr⁻¹·m⁻²).
            radiance = 1.0e-4
        # Per-pixel sample budget = total_rays / n_pixels.
        n_pix = max(1, self.optics.n_pixels())
        rps = max(1.0, float(self.total_rays) / float(n_pix))
        n_batches = max(1,
                        (self.total_rays + self.rays_per_batch - 1) //
                        self.rays_per_batch)
        return plan_ray_budget(
            self.optics, self.film,
            scene_radiance_W_sr_m2     = radiance,
            rays_per_pixel_per_second  = rps,
            n_batches                  = int(n_batches),
            sensor_spp                 = 1.0,
            capture_efficiency         = 1.0,
            ref_wavelength_nm          = 555.0,
        )

    def _build_backends(self, scene: TracerScene,
                        cam: PinholeCamera,
                        frame_cfg: FrameConfig) -> dict[str, ExposureBackend]:
        backends: dict[str, ExposureBackend] = {}
        cpp_b: Optional[CppExposureBackend] = None
        if "cpp" in self.backends_requested:
            cpp_b = CppExposureBackend(
                scene, cam, self.freq_hz,
                max_bounces   = self.max_bounces,
                min_amplitude = 1.0e-3,
                atmo_abs_db_per_m = 0.0,
                adaptive_mode = self.adaptive_allocation_mode,
            )
            if hasattr(cpp_b.tracer, "set_camera_visibility"):
                cv = frame_cfg.camera_visibility
                cpp_b.tracer.set_camera_visibility(
                    camera_vis_mode    = int(cv.camera_vis_mode),
                    transparent_mode   = int(cv.transparent_mode),
                    depth_cull_enabled = bool(cv.depth_cull_enabled),
                    depth_cull_m       = float(cv.depth_cull_m),
                )
            if self.profile_enabled and hasattr(cpp_b.tracer, "set_profile_pulse"):
                try:
                    cpp_b.tracer.set_profile_pulse(True, 2.0)
                except Exception as exc:
                    print(f"  [warn] native profile pulse enable failed: {exc}")
            fc = frame_cfg.field_capture
            if fc.enabled and hasattr(cpp_b.tracer, "enable_field_capture_regular"):
                bmin = np.asarray(scene.bounds_min, np.float32)
                bmax = np.asarray(scene.bounds_max, np.float32)
                n_bands = int(getattr(cpp_b, "n_bands", int(np.asarray(self.freq_hz).size)))
                n_cells = int(max(1, int(fc.nx)) * max(1, int(fc.ny)) * max(1, int(fc.nz)))
                est_field_bytes = int(n_cells * max(1, n_bands) * 8)
                est_field_mib = float(est_field_bytes) / float(1024 ** 2)
                if self.profile_enabled:
                    print(
                        "  [field_capture] "
                        f"kind={fc.grid_kind} dims={int(fc.nx)}x{int(fc.ny)}x{int(fc.nz)} "
                        f"bands={n_bands} est_field={est_field_mib:.2f} MiB "
                        f"capture_strikes={bool(fc.capture_strikes)} max_strikes={int(fc.max_strikes)}"
                    )
                    print(
                        "                  "
                        f"bounds min={np.asarray(bmin, dtype=np.float32).tolist()} "
                        f"max={np.asarray(bmax, dtype=np.float32).tolist()}"
                    )
                try:
                    if fc.grid_kind == "kdtree" and hasattr(cpp_b.tracer, "enable_field_capture_kdtree"):
                        nodes = [{
                            "bmin": bmin,
                            "bmax": bmax,
                            "child_lo": -1,
                            "child_hi": -1,
                            "split_axis": -1,
                            "split_pos": 0.0,
                            "leaf_dims": np.asarray([fc.nx, fc.ny, fc.nz], np.int32),
                            "first_data": 0,
                        }]
                        cpp_b.tracer.enable_field_capture_kdtree(
                            nodes,
                            capture_strikes=bool(fc.capture_strikes),
                            max_strikes=int(fc.max_strikes),
                            clear_existing=True,
                        )
                    else:
                        cpp_b.tracer.enable_field_capture_regular(
                            int(fc.nx), int(fc.ny), int(fc.nz),
                            bmin, bmax,
                            capture_strikes=bool(fc.capture_strikes),
                            max_strikes=int(fc.max_strikes),
                            clear_existing=True,
                        )
                except Exception as exc:
                    detail = (
                        "field capture gate failed: "
                        f"gate=enable_field_capture_{fc.grid_kind} "
                        f"native_error={exc}"
                    )
                    raise RuntimeError(detail) from exc
            backends["cpp"] = cpp_b
        if "glsl" in self.backends_requested:
            backends["glsl"] = GlslExposureBackend(
                scene, cam, self.freq_hz,
                max_bounces  = self.max_bounces,
                mirror_from  = cpp_b,
                min_amplitude = 1.0e-3,
                atmo_abs_db_per_m = 0.0,
            )
        return backends

    # ── Sensor/Film SSBO upload (T3: binding helper) ────────────────────────
    def _bind_sensor_film_ssbo(self, tracer) -> None:
        """Upload sensor and film tensors to C++ tracer SSBO."""
        if tracer is None or not hasattr(tracer, 'set_sensor_film_ssbo'):
            print("  [warn] tracer has no set_sensor_film_ssbo method; skipping SSBO upload")
            return
        
        try:
            tracer.set_sensor_film_ssbo(
                sensor_chunk=self._sensor_film_tensors['sensor'].astype(np.float32, copy=False),
                film_chunk=self._sensor_film_tensors['film'].astype(np.float32, copy=False),
                active_slots=self.sensor_film_slots,
            )
            active_count = sum(1 for s, f in self.sensor_film_slots if s >= 0 and f >= 0)
            print(f"  [sensor_film_ssbo] uploaded {active_count} active slots to tracer")
        except Exception as e:
            print(f"  [warn] sensor_film_ssbo upload failed: {e}")

    def _configure_default_wave_contexts(self, tracer: Any, cam: PinholeCamera, solved: Any | None = None) -> None:
        """Install default wave contexts so kernel wave path is active in BDPT."""
        if tracer is None:
            raise RuntimeError("wave context configuration requires a live tracer")
        if not hasattr(tracer, "clear_scale_contexts") or not hasattr(tracer, "add_scale_context"):
            raise RuntimeError("tracer does not expose scale-context API required for wave path")

        tracer.clear_scale_contexts()

        # Center contexts on the aperture/focus region used by PIXEL_CONE sensor rays.
        cam_pos = np.asarray(cam.pos, np.float64)
        cam_fwd = np.asarray(cam.fwd, np.float64)
        cam_fwd = cam_fwd / max(1.0e-12, float(np.linalg.norm(cam_fwd)))
        focus_m = float(np.linalg.norm(np.asarray(scene_mod.SCENE_CENTER, np.float64) - cam_pos))
        focal_len_m = max(1.0e-4, float(self.optics.focal_mm) * 1.0e-3)
        phase_scale = 1.0
        lens_tube_len_m = 0.05
        if solved is not None:
            focus_m = float(max(1.0e-3, solved.sanity_input.focus_distance_m))
            lens_tube_len_m = float(abs(solved.sanity_input.sensor_plane_z_m - solved.sanity_input.lens_center_z_m))
            phase_scale += min(2.0, max(0.0, float(solved.sanity_report.error_degree.overall)))

        aperture_center = cam_pos + cam_fwd * max(1.0e-3, focus_m)
        aperture_radius_m = max(1.0e-6, focal_len_m / max(2.0 * float(self.optics.f_number), 1.0e-6))

        wave_kind = int(getattr(_sk, "SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ", 1))
        thin_kind = int(getattr(_sk, "SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM", 2))
        thick_kind = int(getattr(_sk, "SCALE_CONTEXT_KIND_THICK_LENS_WAVE", 3))
        rt_wave = int(getattr(_sk, "RT_SCALE_WAVE", 1))

        thin_payload = np.asarray([focal_len_m], dtype=np.float64)
        tracer.add_scale_context(
            pos=aperture_center.astype(np.float64),
            radius=float(max(aperture_radius_m * 1.2, 1.0e-4)),
            scale_type=rt_wave,
            dt_m=float(max(1.0e-4, aperture_radius_m * 0.08)),
            n_substeps=2,
            n_real=1.0,
            n_imag=0.0,
            context_kind=thin_kind,
            payload=thin_payload,
        )

        wave_payload = np.asarray([focal_len_m], dtype=np.float64)
        tracer.add_scale_context(
            pos=aperture_center.astype(np.float64),
            radius=float(max(aperture_radius_m * 1.5, 1.0e-4)),
            scale_type=rt_wave,
            dt_m=float(max(1.0e-4, aperture_radius_m * 0.1)),
            n_substeps=2,
            n_real=1.0,
            n_imag=0.0,
            context_kind=wave_kind,
            payload=wave_payload,
        )

        tube_center = aperture_center - cam_fwd * max(1.0e-3, 0.5 * lens_tube_len_m)
        tracer.add_scale_context(
            pos=tube_center.astype(np.float64),
            radius=float(max(aperture_radius_m * 1.1, 1.0e-4)),
            scale_type=rt_wave,
            dt_m=float(max(1.0e-4, lens_tube_len_m * 0.1)),
            n_substeps=2,
            n_real=1.0,
            n_imag=0.0,
            context_kind=wave_kind,
            payload=wave_payload,
        )

        thick_payload = np.asarray([focal_len_m, phase_scale], dtype=np.float64)
        tracer.add_scale_context(
            pos=aperture_center.astype(np.float64),
            radius=float(max(aperture_radius_m * 2.0, 1.0e-4)),
            scale_type=rt_wave,
            dt_m=float(max(1.0e-4, aperture_radius_m * 0.1)),
            n_substeps=2,
            n_real=1.0,
            n_imag=0.0,
            context_kind=thick_kind,
            payload=thick_payload,
        )

    # ── Calibration: gain to match measured H to target H ────────────────
    @staticmethod
    def _train_emissivity_gain(measured_H_J: float, target_H_J: float) -> float:
        if measured_H_J <= 0.0 or target_H_J <= 0.0:
            return 1.0
        return float(target_H_J / measured_H_J)

    def _make_integral_objects(self,
                               backend: str,
                               surf_bhw: np.ndarray,
                               field_bhw: np.ndarray,
                               gain: float,
                               integral_split: Optional[IntegralSplitConfig] = None
                               ) -> tuple[FieldIntegralObject,
                                          SurfaceIntegralObject,
                                          np.ndarray,
                                          np.ndarray]:
        """Build field/surface integral objects from separated BDPT accumulators.

        surf_bhw  — (n_bands, H, W) sensor-hit accumulation (surf_accum).
        field_bhw — (n_bands, H, W) ambient/field accumulation (field_accum).
        Both are produced by scatter_bdpt_records() per batch.
        """
        cfg = integral_split if integral_split is not None else self.integral_split
        s = np.asarray(surf_bhw,  np.float64) * float(gain)
        f = np.asarray(field_bhw, np.float64) * float(gain)

        fi = float(np.clip(cfg.field_integrate_frac, 0.0, 1.0))
        fb = float(np.clip(cfg.field_bookkeep_frac, 0.0, 1.0))
        si = float(np.clip(cfg.surface_integrate_frac, 0.0, 1.0))
        sb = float(np.clip(cfg.surface_bookkeep_frac, 0.0, 1.0))

        field_integrated = fi * f
        field_bookkeep   = fb * np.maximum(0.0, f - field_integrated)
        surf_integrated  = si * s
        surf_bookkeep    = sb * np.maximum(0.0, s - surf_integrated)

        field_obj = FieldIntegralObject(
            backend=backend,
            frame_index=int(self._frame_index),
            freq_hz=self.freq_hz.copy(),
            integrated_bands=field_integrated.astype(np.float32, copy=False),
            bookkeeping_bands=field_bookkeep.astype(np.float32, copy=False),
            metrics={
                "integrated_energy": float(field_integrated.sum()),
                "bookkeep_energy": float(field_bookkeep.sum()),
            },
        )
        surface_obj = SurfaceIntegralObject(
            backend=backend,
            frame_index=int(self._frame_index),
            freq_hz=self.freq_hz.copy(),
            integrated_bands=surf_integrated.astype(np.float32, copy=False),
            bookkeeping_bands=surf_bookkeep.astype(np.float32, copy=False),
            metrics={
                "integrated_energy": float(surf_integrated.sum()),
                "bookkeep_energy": float(surf_bookkeep.sum()),
            },
        )

        merged = (field_integrated + surf_integrated).astype(np.float32, copy=False)
        rgb_linear = _bands_to_rgb(merged, self.freq_hz).astype(np.float32, copy=False)
        return field_obj, surface_obj, merged, rgb_linear

    def _tone_map_delicate(self, rgb_linear: np.ndarray,
                           integral_split: Optional[IntegralSplitConfig] = None
                           ) -> np.ndarray:
        """Soft-knee tone mapper that preserves low-light separation."""
        cfg = integral_split if integral_split is not None else self.integral_split
        x = np.maximum(0.0, np.asarray(rgb_linear, np.float32))
        nz = x[x > 0.0]
        pct = float(np.clip(cfg.hdr_white_percentile, 75.0, 100.0))
        white = float(np.percentile(nz, pct)) if nz.size >= 16 else float(x.max())
        white = max(white, 1.0e-8)
        y = np.log1p(x / white * 6.0) / math.log1p(6.0)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    def _sensor_display_rgb(self, sensor_obj: SensorIntegralObject) -> np.ndarray:
        """Convert backward-pass sensor integral into display RGB."""
        photons = np.asarray(sensor_obj.photons_per_pixel, dtype=np.float32)
        electrons = np.asarray(sensor_obj.electrons_per_pixel, dtype=np.float32)

        x = np.maximum(0.0, photons)
        white = float(np.percentile(x, 99.5)) if x.size else 1.0
        white = max(white, 1.0e-8)
        y = np.log1p((x / white) * 6.0) / math.log1p(6.0)
        y = np.clip(y, 0.0, 1.0).astype(np.float32)

        # Subtle SNR tint preserves sensor intensity as the dominant signal.
        snr_proxy = np.sqrt(np.maximum(electrons, 0.0))
        snr_w = float(np.percentile(snr_proxy, 99.0)) if snr_proxy.size else 1.0
        snr_w = max(snr_w, 1.0e-8)
        t = np.clip(snr_proxy / snr_w, 0.0, 1.0).astype(np.float32)
        r = y
        g = np.clip(y * (0.92 + 0.08 * t), 0.0, 1.0)
        b = np.clip(y * (0.86 + 0.14 * t), 0.0, 1.0)
        return np.stack([r, g, b], axis=-1)

    @staticmethod
    def _downsample_hw(arr: np.ndarray, sy: int, sx: int, stencil: str = "box") -> np.ndarray:
        """Area-downsample 2D array while preserving dtype."""
        if sy <= 1 and sx <= 1:
            return arr
        h, w = int(arr.shape[0]), int(arr.shape[1])
        ny = max(1, h // max(1, sy))
        nx = max(1, w // max(1, sx))
        trimmed = np.asarray(arr[: ny * sy, : nx * sx])
        block = trimmed.reshape(ny, sy, nx, sx)
        if str(stencil).lower() == "polar" and sy > 1 and sx > 1:
            yy = (np.arange(sy, dtype=np.float64) + 0.5) / float(sy)
            xx = (np.arange(sx, dtype=np.float64) + 0.5) / float(sx)
            gy, gx = np.meshgrid(yy, xx, indexing="ij")
            ry = (gy - 0.5) / 0.5
            rx = (gx - 0.5) / 0.5
            r2 = (rx * rx) + (ry * ry)
            # Polar stencil: circular support with soft radial falloff.
            wmask = np.clip(1.0 - r2, 0.0, 1.0)
            wsum = float(np.sum(wmask))
            if wsum > 1.0e-20:
                reduced = np.tensordot(block, wmask, axes=([1, 3], [0, 1])) / wsum
            else:
                reduced = block.mean(axis=(1, 3), dtype=np.float64)
        else:
            reduced = block.mean(axis=(1, 3), dtype=np.float64)
        if np.issubdtype(arr.dtype, np.integer):
            reduced = np.rint(reduced)
        return reduced.astype(arr.dtype, copy=False)

    @classmethod
    def _downsample_bhw(cls, arr: np.ndarray, sy: int, sx: int, stencil: str = "box") -> np.ndarray:
        """Area-downsample (B,H,W) tensor while preserving dtype."""
        if sy <= 1 and sx <= 1:
            return arr
        bands = [cls._downsample_hw(arr[b], sy, sx, stencil=stencil) for b in range(int(arr.shape[0]))]
        return np.stack(bands, axis=0).astype(arr.dtype, copy=False)

    @classmethod
    def _downsample_hw3(cls, img: np.ndarray, sy: int, sx: int, stencil: str = "box") -> np.ndarray:
        """Area-downsample (H,W,3) RGB image while preserving dtype."""
        if sy <= 1 and sx <= 1:
            return img
        ch = [cls._downsample_hw(img[:, :, c], sy, sx, stencil=stencil) for c in range(int(img.shape[2]))]
        return np.stack(ch, axis=-1).astype(img.dtype, copy=False)

    def _output_downsample_factors(self) -> tuple[int, int]:
        sy = max(1, int(self.height) // max(1, int(self.output_height)))
        sx = max(1, int(self.width) // max(1, int(self.output_width)))
        return sy, sx

    def build_integration_snapshot(self,
                                   backend_name: str,
                                   back: ExposureBackend,
                                   plan: RayDispatchPlan,
                                   frame_cfg: FrameConfig,
                                   batch_index: int,
                                   include_sensor: bool = False,
                                   snapshot_mode: str = "stream") -> IntegrationSnapshot:
        """Public API: build one integration snapshot from current accum state."""
        measured = float(back.measured_radiant_exposure_J(plan.energy_per_ray_J))
        gain = float(self._train_emissivity_gain(measured, float(plan.target_H_J)))
        field_obj, surface_obj, _merged_bands, rgb_linear = self._make_integral_objects(
            backend_name,
            back.surf_accum,
            back.field_accum,
            gain,
            frame_cfg.integral_split,
        )
        img = self._tone_map_delicate(rgb_linear, frame_cfg.integral_split)

        sensor_obj: Optional[SensorIntegralObject] = None
        sensor_rgb: Optional[np.ndarray] = None
        if include_sensor:
            tracer_obj = getattr(back, "tracer", None)
            sensor_obj = self._make_sensor_integral(backend_name, gain, tracer_obj,
                                                    surf_accum=back.surf_accum)
            if sensor_obj is not None:
                sensor_rgb = self._sensor_display_rgb(sensor_obj)
                if self.rgb_source == "sensor":
                    img = sensor_rgb
                    rgb_linear = sensor_rgb.copy()

        return IntegrationSnapshot(
            backend=str(backend_name),
            frame_index=int(self._frame_index),
            batch_index=int(batch_index),
            measured_H_J=float(measured),
            target_H_J=float(plan.target_H_J),
            gain_linear=float(gain),
            image_data=np.asarray(img, dtype=np.float32),
            rgb_linear=np.asarray(rgb_linear, dtype=np.float32),
            field_integrated_data=np.asarray(field_obj.integrated_bands, dtype=np.float32),
            surface_integrated_data=np.asarray(surface_obj.integrated_bands, dtype=np.float32),
            sensor_photons_data=(None if sensor_obj is None
                                 else np.asarray(sensor_obj.photons_per_pixel, dtype=np.float32)),
            sensor_snr_data=(None if sensor_obj is None
                             else np.asarray(sensor_obj.snr_linear, dtype=np.float32)),
            sensor_rgb_data=(None if sensor_rgb is None
                             else np.asarray(sensor_rgb, dtype=np.float32)),
            mode=str(snapshot_mode),
        )

    def stream_integration_snapshots(self,
                                     backs: dict[str, ExposureBackend],
                                     plan: RayDispatchPlan,
                                     frame_cfg: FrameConfig,
                                     batch_index: int) -> dict[str, IntegrationSnapshot]:
        """Public API: build stream snapshots for all active backends."""
        out: dict[str, IntegrationSnapshot] = {}
        sensor_ready = bool(self._last_bdpt_records is not None and self._last_bdpt_records.size > 0)
        for name, back in backs.items():
            out[str(name)] = self.build_integration_snapshot(
                backend_name=str(name),
                back=back,
                plan=plan,
                frame_cfg=frame_cfg,
                batch_index=int(batch_index),
                include_sensor=sensor_ready,
                snapshot_mode="stream",
            )
        return out

    def _make_sensor_integral(self, backend: str, gain: float, tracer: Any = None,
                               surf_accum: Optional[np.ndarray] = None) -> Optional[SensorIntegralObject]:
        """Create a SensorIntegralObject from endpoint-derived accumulation (T4).

        Primary path is now C++ reduction when available; Python fallback keeps
        orchestration continuity.
        Requires: BDPT endpoint records with position and spectral amplitude.

        Returns SensorIntegralObject or None if sensor integration unavailable.
        """
        try:
            # Only accumulate if we have active slots and have just run BDPT
            active_slots = [i for i, (s, f) in enumerate(self.sensor_film_slots) if s >= 0 and f >= 0]
            if not active_slots:
                print("  [warn] no active sensor/film slots; sensor integral unavailable")
                return None

            if aggregate_to_image_pixel_cone is None:
                print("  [warn] bdpt_integrator.aggregate_to_image_pixel_cone unavailable; sensor integral unavailable")
                return None

            if self._last_bdpt_records is None or self._last_bdpt_records.size == 0:
                print("  [warn] no bdpt endpoint records available; sensor integral unavailable")
                return None

            if self._sensor_group_id < 0:
                print("  [warn] sensor group not registered; sensor integral unavailable")
                return None

            # In streaming BDPT mode surf_accum holds the full accumulated
            # scatter of every batch.  Use it directly instead of the
            # single-batch _last_bdpt_records, which is too sparse.
            if surf_accum is not None and surf_accum.size > 0 and np.any(surf_accum):
                # surf_accum is (n_bands, H, W) of per-band amplitude magnitude.
                # Sum across bands for total intensity proxy.
                endpoint_intensity = surf_accum.sum(axis=0, dtype=np.float32)
                endpoint_intensity = endpoint_intensity.astype(np.float32, copy=False)
                endpoint_intensity *= float(max(gain, 0.0) ** 2)
            else:
                # Fallback: derive from _last_bdpt_records when surf_accum
                # is unavailable (non-streaming paths only).
                if self._last_bdpt_records is None or self._last_bdpt_records.size == 0:
                    print("  [warn] no bdpt endpoint records available; sensor integral unavailable")
                    return None

                sensor_img = aggregate_to_image_pixel_cone(
                    self._last_bdpt_records,
                    n_bands=int(self.freq_hz.shape[0]),
                    n_px=int(self.width),
                    n_py=int(self.height),
                    sensor_group_id=int(self._sensor_group_id),
                )
                if sensor_img.size == 0:
                    print("  [warn] empty sensor image after endpoint aggregation")
                    return None
                endpoint_intensity = np.sum(np.abs(sensor_img) ** 2, axis=0, dtype=np.float32)
                endpoint_intensity = endpoint_intensity.astype(np.float32, copy=False)
                endpoint_intensity *= float(max(gain, 0.0) ** 2)

            mean_intensity = float(np.mean(endpoint_intensity))
            target_photons = float(max(self._last_target_photons_per_pixel, 0.0))
            if mean_intensity > 1.0e-20 and target_photons > 0.0:
                photons_per_pixel = (endpoint_intensity / mean_intensity) * target_photons
            else:
                photons_per_pixel = endpoint_intensity.copy()
            photons_per_pixel = photons_per_pixel.astype(np.float32, copy=False)

            if (tracer is not None
                    and hasattr(tracer, "reduce_endpoint_records_to_sensor_integral")
                    and self._last_bdpt_emit_counts is None):
                # Only use the C++ per-record reducer in non-streaming mode.
                # In streaming mode surf_accum already holds the full
                # accumulated scatter from all batches; no need to re-reduce
                # the last (sparse) batch.
                cpp_result = tracer.reduce_endpoint_records_to_sensor_integral(
                    self._last_bdpt_records,
                    int(self.width),
                    int(self.height),
                    int(self._sensor_group_id),
                    float(self._last_target_photons_per_pixel),
                    float(max(gain, 0.0)),
                )

                photons_per_pixel = np.asarray(cpp_result["photons_per_pixel"], dtype=np.float32)
                electrons_per_pixel = np.asarray(cpp_result["electrons_per_pixel"], dtype=np.float32)
                snr_linear = np.asarray(cpp_result["snr_linear"], dtype=np.float32)
                metrics = dict(cpp_result["metrics"])

                qe_peak = float(metrics.get("qe_peak", 0.0))
                read_noise_e = float(metrics.get("read_noise_e", 0.0))
                full_well_e = float(metrics.get("full_well_e", 0.0))
                peak_snr = float(metrics.get("snr_peak", 0.0))
                mean_snr = float(metrics.get("snr_mean", 0.0))

                sensor_w_m = float(self.optics.sensor_w_mm) * 1.0e-3
                sensor_h_m = float(self.optics.sensor_h_mm) * 1.0e-3
                pixel_pitch_m = float(self.optics.pixel_pitch_um) * 1.0e-6
                focal_m = float(self.optics.focal_mm) * 1.0e-3
                # Use aperture_mm directly (more precise for complex aperture sims than f_number)
                aperture_radius_m = float(self.optics.aperture_mm) * 0.5 * 1.0e-3
                n_px = self.optics.n_pixels()

                return SensorIntegralObject(
                    backend=backend,
                    frame_index=int(self._frame_index),
                    optics=self.optics,
                    film=self.film,
                    n_pixels=int(n_px),
                    sensor_w_m=float(sensor_w_m),
                    sensor_h_m=float(sensor_h_m),
                    pixel_pitch_m=float(pixel_pitch_m),
                    focal_m=float(focal_m),
                    aperture_radius_m=float(aperture_radius_m),
                    qe_peak=float(qe_peak),
                    photons_per_pixel=photons_per_pixel,
                    electrons_per_pixel=electrons_per_pixel,
                    noise_floor_e=float(read_noise_e),
                    full_well_e=int(round(full_well_e)),
                    snr_linear=snr_linear,
                    peak_snr=float(peak_snr),
                    mean_snr=float(mean_snr),
                    metrics=metrics,
                )

            # Multi-slot accumulation: every active slot contributes its own
            # sensor/film noise model without re-aggregating endpoint records.
            h, w = photons_per_pixel.shape
            electrons_accum = np.zeros((h, w), dtype=np.float32)
            snr_accum = np.zeros((h, w), dtype=np.float32)
            qe_accum = 0.0
            read_noise_accum = 0.0
            full_well_accum = 0.0
            slot_metrics: list[dict[str, Any]] = []

            for slot_id in active_slots:
                sensor_id, film_id = self.sensor_film_slots[int(slot_id)]
                if sensor_id < 0 or film_id < 0:
                    continue

                sensor_row = self._sensor_film_tensors['sensor'][sensor_id]
                film_row = self._sensor_film_tensors['film'][film_id]

                qe_peak = float(sensor_row[8])
                read_noise_e = float(sensor_row[10])
                dark_current_e_s = float(sensor_row[11])
                full_well_e = float(sensor_row[9])
                exposure_time_s = float(film_row[1])

                electrons_slot = photons_per_pixel * np.float32(qe_peak)
                dark_current_accumulated_e = dark_current_e_s * exposure_time_s
                noise_variance = ((read_noise_e ** 2) + dark_current_accumulated_e +
                                  electrons_slot)
                snr_slot = (np.sqrt(np.maximum(electrons_slot, 0.0)) /
                            np.sqrt(np.maximum(noise_variance, 1.0e-10)))
                snr_slot = snr_slot.astype(np.float32, copy=False)

                electrons_accum += electrons_slot.astype(np.float32, copy=False)
                snr_accum += snr_slot
                qe_accum += qe_peak
                read_noise_accum += read_noise_e
                full_well_accum += full_well_e

                slot_meta = self._sensor_film_metadata[int(slot_id)]
                slot_metrics.append({
                    "slot_id": int(slot_id),
                    "sensor_id": int(sensor_id),
                    "film_id": int(film_id),
                    "sensor_name": slot_meta.get("sensor_name", "unknown"),
                    "film_name": slot_meta.get("film_name", "unknown"),
                    "qe_peak": float(qe_peak),
                    "read_noise_e": float(read_noise_e),
                    "dark_current_e_s": float(dark_current_e_s),
                    "dark_current_accumulated_e": float(dark_current_accumulated_e),
                    "exposure_time_s": float(exposure_time_s),
                    "full_well_e": float(full_well_e),
                    "snr_peak": float(np.nanmax(snr_slot)),
                    "snr_mean": float(np.nanmean(snr_slot)),
                    "photons_flux_hz": float(np.mean(photons_per_pixel)),
                    "electrons_flux_hz": float(np.mean(electrons_slot)),
                })

            if not slot_metrics:
                print("  [warn] no valid active sensor/film slots after filtering")
                return None

            inv_slots = np.float32(1.0 / max(1, len(slot_metrics)))
            electrons_per_pixel = (electrons_accum * inv_slots).astype(np.float32, copy=False)
            snr_linear = (snr_accum * inv_slots).astype(np.float32, copy=False)
            qe_peak = float(qe_accum / max(1, len(slot_metrics)))
            read_noise_e = float(read_noise_accum / max(1, len(slot_metrics)))
            full_well_e = float(full_well_accum / max(1, len(slot_metrics)))

            peak_snr = float(np.nanmax(snr_linear))
            mean_snr = float(np.nanmean(snr_linear))
            
            sensor_w_m = float(self.optics.sensor_w_mm) * 1.0e-3
            sensor_h_m = float(self.optics.sensor_h_mm) * 1.0e-3
            pixel_pitch_m = float(self.optics.pixel_pitch_um) * 1.0e-6
            focal_m = float(self.optics.focal_mm) * 1.0e-3
            # Use aperture_mm directly (more precise for complex aperture sims than f_number)
            aperture_radius_m = float(self.optics.aperture_mm) * 0.5 * 1.0e-3
            n_px = self.optics.n_pixels()
            
            # Build metadata with explicit noise model (T5)
            metrics = {
                "active_slot_ids": [int(s) for s in active_slots],
                "n_active_slots": int(len(slot_metrics)),
                "qe_peak": float(qe_peak),
                "read_noise_e": float(read_noise_e),
                "full_well_e": float(full_well_e),
                "snr_peak": float(peak_snr),
                "snr_mean": float(mean_snr),
                "endpoint_records_count": int(self._last_bdpt_records.shape[0]),
                "sensor_group_id": int(self._sensor_group_id),
                "photons_flux_hz": float(np.mean(photons_per_pixel)),
                "electrons_flux_hz": float(np.mean(electrons_per_pixel)),
                "slot_metrics": slot_metrics,
            }
            
            return SensorIntegralObject(
                backend=backend,
                frame_index=int(self._frame_index),
                optics=self.optics,
                film=self.film,
                n_pixels=int(n_px),
                sensor_w_m=float(sensor_w_m),
                sensor_h_m=float(sensor_h_m),
                pixel_pitch_m=float(pixel_pitch_m),
                focal_m=float(focal_m),
                aperture_radius_m=float(aperture_radius_m),
                qe_peak=float(qe_peak),
                photons_per_pixel=photons_per_pixel,
                electrons_per_pixel=electrons_per_pixel.astype(np.float32),
                noise_floor_e=float(read_noise_e),
                full_well_e=int(round(full_well_e)),
                snr_linear=snr_linear.astype(np.float32),
                peak_snr=float(peak_snr),
                mean_snr=float(mean_snr),
                metrics=metrics,
            )
        except Exception as e:
            print(f"  [warn] sensor integral creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_integral_objects(self,
                               field_obj: FieldIntegralObject,
                               surface_obj: SurfaceIntegralObject,
                               sensor_obj: Optional[SensorIntegralObject] = None,
                               field_history_bands: Optional[np.ndarray] = None,
                               surface_history_bands: Optional[np.ndarray] = None
                               ) -> tuple[str, str, Optional[str]]:
        """Write field/surface/sensor objects and uncompressed full histories."""
        field_path = os.path.join(
            self.out_dir,
            f"{self._frame_index:04d}_{field_obj.backend}_field_integral.npz")
        surf_path = os.path.join(
            self.out_dir,
            f"{self._frame_index:04d}_{surface_obj.backend}_surface_integral.npz")
        sensor_path = None
        if sensor_obj is not None:
            sensor_path = os.path.join(
                self.out_dir,
                f"{self._frame_index:04d}_{sensor_obj.backend}_sensor_integral.npz")

        field_history = (np.asarray(field_history_bands, dtype=np.float32)
                         if field_history_bands is not None
                         else (field_obj.integrated_bands + field_obj.bookkeeping_bands).astype(np.float32, copy=False))
        surface_history = (np.asarray(surface_history_bands, dtype=np.float32)
                           if surface_history_bands is not None
                           else (surface_obj.integrated_bands + surface_obj.bookkeeping_bands).astype(np.float32, copy=False))

        np.savez(
            field_path,
            freq_hz=field_obj.freq_hz,
            integrated_bands=field_obj.integrated_bands,
            bookkeeping_bands=field_obj.bookkeeping_bands,
            full_history_bands=field_history,
            metrics=np.asarray(json.dumps(field_obj.metrics)),
        )
        np.savez(
            surf_path,
            freq_hz=surface_obj.freq_hz,
            integrated_bands=surface_obj.integrated_bands,
            bookkeeping_bands=surface_obj.bookkeeping_bands,
            full_history_bands=surface_history,
            metrics=np.asarray(json.dumps(surface_obj.metrics)),
        )
        if sensor_path is not None and sensor_obj is not None:
            np.savez(
                sensor_path,
                photons_per_pixel=sensor_obj.photons_per_pixel,
                electrons_per_pixel=sensor_obj.electrons_per_pixel,
                snr_linear=sensor_obj.snr_linear,
                metrics=np.asarray(json.dumps(sensor_obj.metrics)),
            )
        
        return field_path, surf_path, sensor_path

    # ── Render one full exposure (drives all sub-batches) ────────────────
    def render_one_exposure(
        self,
        t: float = 0.0,
        batch_preview_cb: Optional[Callable[[dict[str, ExposureBackend], dict[str, IntegrationSnapshot], int, int, float], None]] = None,
    ) -> list[ExposureFrameResult]:
        # ── Build per-frame progressive feature configuration ────────────
        with self._profiler.section("frame_config"):
            frame_cfg = _build_frame_config(
                self._frame_index, self.n_frames_planned, self.integral_split)

        scene_mode = self._scene_mode_for_frame()
        with self._profiler.section("build_scene"):
            scene = _build_tracer_scene(t, scene_mode=scene_mode)
        with self._profiler.section("build_camera"):
            solved = solve_sane_pinhole_camera(
                camera_cls=PinholeCamera,
                width=self.width,
                height=self.height,
                optics=self.optics,
                film=self.film,
                scene_center=np.asarray(scene_mod.SCENE_CENTER, np.float64),
                eye=np.array([0.0, 0.0, 0.0], np.float64),
            )
            cam = solved.camera
        with self._profiler.section("build_plan"):
            plan  = self._build_plan(scene)
        with self._profiler.section("build_backends"):
            backs = self._build_backends(scene, cam, frame_cfg)

        plan_dict = summarize_plan(plan, self.optics, self.film)
        n_pix = max(1, self.optics.n_pixels())
        # Visible-band reference: λ ≈ 555 nm (peak of photopic response).
        photon_E = H_PLANCK * C_LIGHT / (plan.ref_wavelength_nm * 1.0e-9)
        photons_per_pix = plan.target_H_J / max(1, n_pix) / max(photon_E, 1.0e-30)
        self._last_target_photons_per_pixel = float(photons_per_pix)
        self._cleanup_temp_bdpt_file()
        self._release_last_bdpt_records()

        fc_active = frame_cfg.field_capture
        cv_active = frame_cfg.camera_visibility
        ss_active = frame_cfg.surface_spline

        print(f"\n══════════════════════ EXPOSURE {self._frame_index:04d} "
              f"[level {frame_cfg.detail_level}] ══════════════════════")
        print(f"  config   : {frame_cfg.description}")
        print(f"  cam vis  : {_cam_vis_name(cv_active.camera_vis_mode)} · "
              f"transp={_transp_name(cv_active.transparent_mode)}"
              + (f" · depth_cull={cv_active.depth_cull_m:.0f}m"
                 if cv_active.depth_cull_enabled else ""))
        print(f"  field    : {'ENABLED ' + fc_active.grid_kind + ' ' + str(fc_active.nx) + '³' if fc_active.enabled else 'disabled'}")
        print(f"  spline   : {'ENABLED fit_all=' + str(ss_active.fit_all_tris) + ' λ=' + str(ss_active.ridge_lambda) if ss_active.enabled else 'disabled'}")
        print(f"  scene:  {scene.verts.shape[0]} tris, {scene.src_pos.shape[0]} emissive sources")
        print(f"  emissive total power = {scene.total_emissive_power_W:.4g} W "
              f"over {scene.total_emissive_area_m2:.4g} m²")
        if scene.src_emit_rgb_W.size > 0:
            rgb_emit = np.sum(scene.src_emit_rgb_W, axis=0)
            rgb_total = float(np.sum(rgb_emit))
            if rgb_total > 1.0e-12:
                rgb_frac = rgb_emit / rgb_total
                print("  emissive rgb share = "
                      f"R {rgb_frac[0]*100.0:5.1f}%  "
                      f"G {rgb_frac[1]*100.0:5.1f}%  "
                      f"B {rgb_frac[2]*100.0:5.1f}%")
        print(f"  budget (total)       : N_rays={plan.total_rays:_}  "
              f"n_batches={plan.n_batches}  rays/batch={plan.rays_per_batch:_}")
        print(f"  per-pixel target     : H={plan.target_H_J/n_pix:.3e} J  "
              f"photons={photons_per_pix:.3e} @ λ={plan.ref_wavelength_nm:.0f} nm")
        print(f"  shutter t            : {self.film.exposure_time_s:.4g} s @ "
              f"f/{self.optics.f_number:.2f}, ISO {self.film.iso:.0f}")
        print(f"  cam sanity           : {solved.sanity_report.status.upper()}"
              f"  coc={solved.sanity_report.circle_of_confusion_um:.2f} um"
              f"  sensor_adjust={solved.sanity_report.sensor_adjustment_needed_mm:+.2f} mm"
              f"  err={solved.sanity_report.error_degree.overall:.3f}"
              f"  iter={solved.iterations}")
        if solved.sanity_report.warnings:
            print("  cam warnings         : " + " | ".join(solved.sanity_report.warnings))
        if solved.sanity_report.failures:
            print("  cam failures         : " + " | ".join(solved.sanity_report.failures))

        # ── Sub-batch streaming loop ─────────────────────────────────────
        # rays_per_batch in plan is per-pixel·n_pixels = per-frame ray budget /
        # n_batches.  We translate to "per-source" by dividing by source count.
        rays_per_batch_total = max(1, plan.rays_per_batch)
        n_sources = max(1, int(scene.src_pos.shape[0]))
        rays_per_source_per_batch = max(1, rays_per_batch_total // n_sources)

        for back in backs.values():
            back.reset_exposure()
            tracer_obj = getattr(back, "tracer", None)
            if tracer_obj is not None and hasattr(tracer_obj, "clear_field_capture"):
                tracer_obj.clear_field_capture(clear_grid=False, clear_strikes=True)

        conv_cfg = self.convergence
        conv_target_pct = float(np.clip(conv_cfg.target_pct, 0.0, 100.0))
        conv_error_target = max(0.0, float(conv_cfg.max_rel_drift))
        if conv_error_target <= 0.0:
            conv_error_target = max(0.0, 1.0 - conv_target_pct / 100.0)
        conv_every = max(1, int(conv_cfg.check_every_batches))
        conv_min_batches = max(1, int(conv_cfg.min_batches))
        conv_hold = max(1, int(conv_cfg.hold_checks))
        conv_enabled = bool(conv_cfg.enabled)
        conv_drive_batches = bool(conv_cfg.drive_batches) and conv_enabled
        conv_max_batches = max(0, int(conv_cfg.max_batches))
        conv_consecutive_hits = 0
        conv_last_measured_pct = 0.0
        conv_last_pct = 0.0

        def _convergence_backend() -> Optional[ExposureBackend]:
            if "cpp" in backs:
                return backs["cpp"]
            return next(iter(backs.values()), None)

        if conv_drive_batches:
            print("  [conv] acquisition mode: open-ended batches until exposure target is met")

        # ── BDPT pre-batch registration ──────────────────────────────────
        # Register emissive/sensor tri-groups once before the batch loop so
        # that each per-batch bidirectional_packed call sees a stable topology.
        _bdpt_stream_active = False
        _bdpt_emitter_centers: np.ndarray = np.zeros((0, 3), np.float64)
        _bdpt_target_total: int = 0
        if self.integrator == "bdpt" and "cpp" in backs:
            cpp_back = backs["cpp"]  # type: ignore[assignment]
            tracer = getattr(cpp_back, "tracer", None)
            if tracer is not None and hasattr(tracer, "register_tri_group"):
                tracer.clear_tri_groups()
                emissive_tris = np.ascontiguousarray(scene.src_tri_idx, dtype=np.int32)

                # ── Surface spline fitting ────────────────────────────────
                ss_cfg = frame_cfg.surface_spline
                ps_cfg = frame_cfg.parametric_sdf
                spline_coeffs: Optional[np.ndarray] = None
                if ss_cfg.enabled and _HAS_SURFACE_SPLINE and emissive_tris.size > 0:
                    try:
                        uv, faces = _weld_mesh(scene.verts)
                        subset_arg = (None if ss_cfg.fit_all_tris else emissive_tris)
                        spline_coeffs = _ss_parameterize_mesh(
                            uv, faces,
                            tri_subset   = subset_arg,
                            ridge_lambda = float(ss_cfg.ridge_lambda),
                            n_threads    = int(ss_cfg.n_threads),
                        )  # (n_tris, 6)
                        print(f"  surface spline: {uv.shape[0]} unique verts, "
                              f"{'all' if ss_cfg.fit_all_tris else 'emissive-subset'} tris, "
                              f"λ={ss_cfg.ridge_lambda:.1e}")
                    except Exception as exc:
                        print(f"  [warn] surface spline fit failed: {exc}")
                        spline_coeffs = None
                elif ss_cfg.enabled and not _HAS_SURFACE_SPLINE:
                    print("  [warn] surface spline requested but surface_spline_utils "
                          "unavailable (rebuild _spectral_kernels)")

                _TRI_POLY_BARY = int(getattr(_sk, "TRI_PARAM_SURFACE_POLY_BARY", 1))
                emissive_group_centers: list[np.ndarray] = []

                parametric_driver = None
                if ps_cfg.enabled:
                    if _get_sdf_driver is None:
                        raise RuntimeError("parametric SDF requested but sdf_plugins is unavailable")
                    if _TRI_POLY_BARY <= 0:
                        raise RuntimeError("parametric SDF requested but TRI_PARAM_SURFACE_POLY_BARY is unavailable")
                    parametric_driver = _get_sdf_driver(ps_cfg.model, strict=True)

                if emissive_tris.size > 0:
                    if (parametric_driver is not None) or (spline_coeffs is not None and ss_cfg.fit_all_tris):
                        # Per-triangle groups — each emissive tri gets individual coefficients.
                        n_tri_total = scene.verts.shape[0]
                        for ti in emissive_tris:
                            ti_int = int(ti)
                            if ti_int < 0 or ti_int >= n_tri_total:
                                continue
                            if parametric_driver is not None:
                                payload = _resolve_native_parametric_payload(parametric_driver(
                                    tri_id=ti_int,
                                    tri_vertices=np.asarray(scene.verts[ti_int], dtype=np.float64),
                                    params={
                                        "saddle_amplitude_m": float(ps_cfg.saddle_amplitude_m),
                                        "sphere_radius_m": float(ps_cfg.sphere_radius_m),
                                        "neighborhood_margin_uv": float(ps_cfg.neighborhood_margin_uv),
                                    },
                                ))
                            else:
                                payload = {
                                    "kind": int(getattr(_sk, "TRI_PARAM_SURFACE_POLY_BARY", 1)),
                                    "coeffs": spline_coeffs[ti_int].ravel().astype(np.float64),
                                }
                            tracer.register_tri_group(
                                role_bits          = 1,
                                sample_policy      = 1,
                                tri_indices        = np.asarray([ti], np.int32),
                                parametric_surface = payload,
                            )
                            tri_verts = np.asarray(scene.verts[ti_int], np.float64)
                            emissive_group_centers.append(np.mean(tri_verts, axis=0))
                        if parametric_driver is not None:
                            print(f"  bdpt: {emissive_tris.size} per-tri emissive groups with parametric SDF '{ps_cfg.model}'")
                        else:
                            print(f"  bdpt: {emissive_tris.size} per-tri emissive groups with POLY_BARY spline")
                    elif (parametric_driver is not None) or (spline_coeffs is not None):
                        # Single group with mean coefficients across emissive tris.
                        if parametric_driver is not None:
                            payloads: list[dict[str, Any]] = []
                            for ti in emissive_tris:
                                ti_int = int(ti)
                                payload = _resolve_native_parametric_payload(parametric_driver(
                                    tri_id=ti_int,
                                    tri_vertices=np.asarray(scene.verts[ti_int], dtype=np.float64),
                                    params={
                                        "saddle_amplitude_m": float(ps_cfg.saddle_amplitude_m),
                                        "sphere_radius_m": float(ps_cfg.sphere_radius_m),
                                        "neighborhood_margin_uv": float(ps_cfg.neighborhood_margin_uv),
                                    },
                                ))
                                payloads.append(payload)
                            if not payloads:
                                raise RuntimeError("parametric SDF requested but plugin returned no payloads")
                            kind0 = int(payloads[0]["kind"])
                            if any(int(p["kind"]) != kind0 for p in payloads):
                                raise RuntimeError(
                                    "parametric SDF plugin returned mixed kinds for merged group; use per-triangle mode")
                            coeff_stack = np.asarray([np.asarray(p["coeffs"], dtype=np.float64) for p in payloads], dtype=np.float64)
                            mean_payload = {
                                "kind": kind0,
                                "coeffs": np.mean(coeff_stack, axis=0).astype(np.float64, copy=False),
                            }
                        else:
                            valid_ids = emissive_tris[emissive_tris < spline_coeffs.shape[0]]
                            mean_payload = {
                                "kind": _TRI_POLY_BARY,
                                "coeffs": (spline_coeffs[valid_ids].mean(axis=0)
                                           if valid_ids.size > 0 else np.zeros(6, np.float64)),
                            }
                        tracer.register_tri_group(
                            role_bits          = 1,
                            sample_policy      = 1,
                            tri_indices        = emissive_tris,
                            parametric_surface = {
                                "kind": int(mean_payload["kind"]),
                                "coeffs": np.asarray(mean_payload["coeffs"], dtype=np.float64).ravel(),
                            },
                        )
                        tri_pos = np.asarray(scene.verts[emissive_tris], np.float64).reshape(-1, 3)
                        emissive_group_centers.append(np.mean(tri_pos, axis=0))
                        if parametric_driver is not None:
                            print(f"  bdpt: 1 emissive group with mean parametric SDF '{ps_cfg.model}'")
                        else:
                            print("  bdpt: 1 emissive group with mean POLY_BARY spline")
                    else:
                        if ps_cfg.enabled:
                            raise RuntimeError(
                                "parametric SDF requested but no emissive tris are available for registration")
                        tracer.register_tri_group(
                            role_bits     = 1,
                            sample_policy = 1,
                            tri_indices   = emissive_tris,
                        )
                        tri_pos = np.asarray(scene.verts[emissive_tris], np.float64).reshape(-1, 3)
                        emissive_group_centers.append(np.mean(tri_pos, axis=0))

                # ── Sensor group (PIXEL_CONE) ─────────────────────────────
                all_tris = np.ascontiguousarray(
                    np.arange(scene.verts.shape[0], dtype=np.int32))
                sensor_w_m = float(self.optics.sensor_w_mm) * 1.0e-3
                sensor_h_m = float(self.optics.sensor_h_mm) * 1.0e-3
                focal_m = max(1.0e-4, float(self.optics.focal_mm) * 1.0e-3)
                aperture_radius_m = max(0.0, float(self.optics.aperture_mm) * 0.5e-3)
                self._sensor_group_id = tracer.register_tri_group(
                    role_bits     = 2,
                    sample_policy = 3,
                    tri_indices   = all_tris,
                    sensor_camera = {
                        "pos": np.asarray(cam.pos, np.float64),
                        "fwd": np.asarray(cam.fwd, np.float64),
                        "up":  np.asarray(cam.up,  np.float64),
                        "sensor_w_m":         float(sensor_w_m),
                        "sensor_h_m":         float(sensor_h_m),
                        "focal_m":            float(max(focal_m, 1.0e-3)),
                        "aperture_radius_m":  float(aperture_radius_m),
                        "n_px":               int(self.width),
                        "n_py":               int(self.height),
                        "n_aperture_samples": 1,
                        "aperture_stop_group_id": -1,
                    },
                )
                print(f"  bdpt: sensor group registered with id {self._sensor_group_id}")

                # Bind sensor/film SSBO before batch dispatch.
                self._bind_sensor_film_ssbo(tracer)
                self._configure_default_wave_contexts(tracer, cam, solved)

                _bdpt_emitter_centers = (
                    np.asarray(emissive_group_centers, np.float64)
                    if emissive_group_centers else np.zeros((0, 3), np.float64)
                )
                n_emit_pre = len(emissive_group_centers)
                _rays_per_emit = max(64, self.total_rays // max(1, tracer.n_tri_groups()))
                _bdpt_target_total = max(n_emit_pre, int(_rays_per_emit) * max(1, n_emit_pre))
                _bdpt_stream_active = True
                print(f"  bdpt: streaming {n_emit_pre} emitter groups, "
                      f"~{_bdpt_target_total:_} rays/batch target")

        t0 = time.perf_counter()
        batches_executed = 0
        converged_early = False
        b_idx = 0
        while True:
            if not conv_drive_batches and b_idx >= plan.n_batches:
                break
            if conv_drive_batches and conv_max_batches > 0 and b_idx >= conv_max_batches:
                print(f"  [conv] reached max_batches={conv_max_batches}; stopping acquisition")
                break

            seed = (self._rng_seed * 1_000_003) + b_idx + 1
            if _bdpt_stream_active:
                _cpp = backs.get("cpp")
                if _cpp is not None:
                    _emit_rays = self._allocate_bdpt_emit_rays(
                        cpp_back=_cpp,
                        emitter_centers=_bdpt_emitter_centers,
                        target_total_rays=_bdpt_target_total,
                        seed=int(seed),
                    )
                    _recs = _cpp.run_bdpt_batch(_emit_rays, seed)
                    _cpp.scatter_bdpt_records(_recs, self._sensor_group_id)
                    # Store last batch's records for adaptive allocator only.
                    # Do NOT call _stage_bdpt_records here — it would overwrite
                    # the intermediary file every batch (the cause of narrow
                    # noise bands / blank first frame in streaming BDPT mode).
                    self._last_bdpt_records = np.ascontiguousarray(_recs, dtype=np.float32)
                    self._last_bdpt_emit_counts = np.asarray(_emit_rays, np.int32)
                    for _b in backs.values():
                        if isinstance(_b, GlslExposureBackend):
                            _b.surf_accum[:]      = _cpp.surf_accum
                            _b.field_accum[:]     = _cpp.field_accum
                            _b.accum[:]           = _cpp.accum
                            _b.n_rays_accumulated = _cpp.n_rays_accumulated
            else:
                for back in backs.values():
                    back.render_batch(rays_per_source_per_batch, seed)
                    # Forward-trace path writes into accum only; there is no
                    # field/surface split available.  Treat all accumulated
                    # energy as surface so _make_integral_objects has real data.
                    back.surf_accum[:] = back.accum
                    # field_accum stays zero — no ambient data in forward mode.
            batches_executed = b_idx + 1

            if batch_preview_cb is not None:
                try:
                    n_batches_hint = -1 if conv_drive_batches else int(plan.n_batches)
                    snaps = self.stream_integration_snapshots(
                        backs=backs,
                        plan=plan,
                        frame_cfg=frame_cfg,
                        batch_index=int(b_idx + 1),
                    )
                    batch_preview_cb(
                        backs,
                        snaps,
                        int(b_idx + 1),
                        int(n_batches_hint),
                        float(time.perf_counter() - t0),
                    )
                except Exception as exc:
                    print(f"  [warn] batch preview callback failed: {exc}")
                    batch_preview_cb = None

            progress_every = 10 if conv_drive_batches else max(1, plan.n_batches // 10)
            if (b_idx + 1) % progress_every == 0:
                elapsed = time.perf_counter() - t0
                if conv_drive_batches:
                    print(f"    batch {b_idx+1:>5}  elapsed={elapsed:6.2f}s")
                else:
                    pct = 100.0 * (b_idx + 1) / plan.n_batches
                    print(f"    batch {b_idx+1:>5}/{plan.n_batches}  "
                          f"({pct:5.1f}%)  elapsed={elapsed:6.2f}s")

            if conv_enabled and ((b_idx + 1) % conv_every) == 0:
                conv_back = _convergence_backend()
                if conv_back is not None:
                    measured_H_J = float(conv_back.measured_radiant_exposure_J(plan.energy_per_ray_J))
                    target_H_J = max(float(plan.target_H_J), 1.0e-30)
                    conv_last_measured_pct = 100.0 * measured_H_J / target_H_J
                    conv_last_pct = 100.0 * abs(measured_H_J - target_H_J) / target_H_J
                    if (b_idx + 1) >= conv_min_batches and conv_last_pct <= (conv_error_target * 100.0):
                        conv_consecutive_hits += 1
                    else:
                        conv_consecutive_hits = 0
                    if conv_consecutive_hits >= conv_hold:
                        converged_early = True
                        print(f"  [conv] reached H={measured_H_J:.3e} J "
                              f"({conv_last_measured_pct:.5f}% of target) at batch {b_idx+1}")
                        print(f"         target H={target_H_J:.3e} J, "
                              f"error<={conv_error_target:.3e}, hold={conv_hold}")
                        break
            b_idx += 1

        elapsed = time.perf_counter() - t0
        if converged_early:
            if conv_drive_batches:
                print(f"  [conv] stop after {batches_executed} open-ended batches")
            else:
                print(f"  [conv] early stop after {batches_executed}/{plan.n_batches} batches")
        print(f"  {batches_executed} batches in {elapsed:.2f}s "
              f"({(batches_executed/max(elapsed,1e-9)):.1f} batches/s)")

        # ── Build human-readable frame config summary (HUD + JSON) ───────
        fc_s = frame_cfg.field_capture
        cv_s = frame_cfg.camera_visibility
        ss_s = frame_cfg.surface_spline
        ps_s = frame_cfg.parametric_sdf
        is_s = frame_cfg.integral_split
        frame_config_summary = {
            "description":   frame_cfg.description,
            "detail_level":  int(frame_cfg.detail_level),
            "scene_mode":    scene_mode,
            "field":         (f"{fc_s.grid_kind} {fc_s.nx}³"
                              if fc_s.enabled else "off"),
            "field_strikes": fc_s.capture_strikes if fc_s.enabled else False,
            "cam_vis":       _cam_vis_name(cv_s.camera_vis_mode),
            "transparent":   _transp_name(cv_s.transparent_mode),
            "depth":         (f"{cv_s.depth_cull_m:.0f}m"
                              if cv_s.depth_cull_enabled else "off"),
            "spline":        (f"{'all' if ss_s.fit_all_tris else 'emissive'} "
                              f"λ={ss_s.ridge_lambda:.1e}"
                              if ss_s.enabled else "off"),
            "parametric":    (f"{ps_s.model} "
                               f"saddle={ps_s.saddle_amplitude_m:.2e}m "
                               f"sphereR={ps_s.sphere_radius_m:.2e}m "
                               f"marginUV={ps_s.neighborhood_margin_uv:.2e}"
                               if ps_s.enabled else "off"),
            "fi": float(is_s.field_integrate_frac),
            "si": float(is_s.surface_integrate_frac),
            "fb": float(is_s.field_bookkeep_frac),
            "sb": float(is_s.surface_bookkeep_frac),
            "hdr_wp": float(is_s.hdr_white_percentile),
        }

        # ── Per-backend calibration + dump ───────────────────────────────
        results: list[ExposureFrameResult] = []
        for name, back in backs.items():
            with self._profiler.section(f"backend_{name}_measure"):
                measured = back.measured_radiant_exposure_J(plan.energy_per_ray_J)
            with self._profiler.section(f"backend_{name}_gain"):
                gain     = self._train_emissivity_gain(measured, plan.target_H_J)
                gain_db  = 20.0 * math.log10(max(gain, 1.0e-12))
            virtual_t = self.film.exposure_time_s
            qe = float(self.film.quantum_efficiency)
            snr = math.sqrt(max(photons_per_pix * qe, 0.0))
            tracer_obj = getattr(back, "tracer", None)

            with self._profiler.section(f"backend_{name}_integrals"):
                field_obj, surface_obj, _merged_bands, rgb_linear = \
                    self._make_integral_objects(name, back.surf_accum, back.field_accum,
                                                gain, frame_cfg.integral_split)
                sensor_obj = self._make_sensor_integral(name, gain, tracer_obj,
                                                    surf_accum=back.surf_accum)

            img = self._tone_map_delicate(rgb_linear, frame_cfg.integral_split)

            if self.rgb_source == "sensor":
                if sensor_obj is not None:
                    img = self._sensor_display_rgb(sensor_obj)
                    rgb_linear = img.copy()
                else:
                    print("  [warn] rgb_source=sensor but sensor integral unavailable; using accum RGB")

            # Keep EndpointRecord history intact and optionally derive endpoint RGB.
            # In streaming BDPT mode _last_bdpt_records holds only the final
            # batch — not the full session.  Endpoint-RGB reduction on a single
            # sparse batch produces noise.  Skip it; rgb_linear already carries
            # the fully-accumulated surf_accum/field_accum result.
            _streaming_bdpt = self._last_bdpt_emit_counts is not None
            if (not _streaming_bdpt
                and tracer_obj is not None
                and hasattr(tracer_obj, "reduce_endpoint_records_to_rgb_image")
                and self._last_bdpt_records is not None
                and self._last_bdpt_records.size > 0
                and self._sensor_group_id >= 0):
                try:
                    cpp_rgb = tracer_obj.reduce_endpoint_records_to_rgb_image(
                        self._last_bdpt_records,
                        int(self.width),
                        int(self.height),
                        int(self._sensor_group_id),
                        float(max(gain, 0.0)),
                        float(frame_cfg.integral_split.hdr_white_percentile),
                    )
                    endpoint_rgb_linear = np.asarray(cpp_rgb["rgb_linear"], dtype=np.float32)
                    endpoint_img = np.asarray(cpp_rgb["rgb_tonemapped"], dtype=np.float32)
                    rgb_telemetry = dict(cpp_rgb.get("telemetry", {}))
                    kept = int(rgb_telemetry.get("kept_records", 0))
                    inp = int(rgb_telemetry.get("input_records", 0))
                    drop = max(0, inp - kept)
                    if inp > 0:
                        drop_frac = float(drop) / float(inp)
                        if drop_frac > 0.05:
                            print(f"  [warn] endpoint reduction dropped {drop:_}/{inp:_} records ({100.0*drop_frac:.1f}%)")
                    ord_reg = int(rgb_telemetry.get("order_regressions", 0))
                    if ord_reg > 0:
                        print(f"  [warn] endpoint order regressions detected: {ord_reg:_}")
                    if sensor_obj is not None and isinstance(sensor_obj.metrics, dict):
                        sensor_obj.metrics["endpoint_rgb_telemetry"] = rgb_telemetry
                    if self.rgb_source == "endpoint":
                        rgb_linear = endpoint_rgb_linear
                        img = endpoint_img
                except Exception as exc:
                    print(f"  [warn] C++ endpoint->RGB reduction failed; using accum RGB: {exc}")

            # Optional output conversion path for oversampled rendering.
            sy, sx = self._output_downsample_factors()
            if (self.output_height != self.height) or (self.output_width != self.width):
                st = self.output_oversample_stencil
                rgb_linear = self._downsample_hw3(np.asarray(rgb_linear), sy, sx, stencil=st)
                img = self._downsample_hw3(np.asarray(img), sy, sx, stencil=st)
                field_obj.integrated_bands = self._downsample_bhw(np.asarray(field_obj.integrated_bands), sy, sx, stencil=st)
                field_obj.bookkeeping_bands = self._downsample_bhw(np.asarray(field_obj.bookkeeping_bands), sy, sx, stencil=st)
                surface_obj.integrated_bands = self._downsample_bhw(np.asarray(surface_obj.integrated_bands), sy, sx, stencil=st)
                surface_obj.bookkeeping_bands = self._downsample_bhw(np.asarray(surface_obj.bookkeeping_bands), sy, sx, stencil=st)
                if sensor_obj is not None:
                    sensor_obj.photons_per_pixel = self._downsample_hw(np.asarray(sensor_obj.photons_per_pixel), sy, sx, stencil=st)
                    sensor_obj.electrons_per_pixel = self._downsample_hw(np.asarray(sensor_obj.electrons_per_pixel), sy, sx, stencil=st)
                    sensor_obj.snr_linear = self._downsample_hw(np.asarray(sensor_obj.snr_linear), sy, sx, stencil=st)

            field_history = (np.asarray(back.accum, np.float32) * np.float32(max(gain, 0.0)))
            surface_history = (np.asarray(back.accum, np.float32) * np.float32(max(gain, 0.0)))
            if (self.output_height != self.height) or (self.output_width != self.width):
                st = self.output_oversample_stencil
                field_history = self._downsample_bhw(field_history, sy, sx, stencil=st)
                surface_history = self._downsample_bhw(surface_history, sy, sx, stencil=st)
            
            field_obj_path, surface_obj_path, sensor_obj_path = \
                self._save_integral_objects(
                    field_obj,
                    surface_obj,
                    sensor_obj,
                    field_history_bands=field_history,
                    surface_history_bands=surface_history,
                )

            png_path   = os.path.join(self.out_dir,
                                      f"{self._frame_index:04d}_{name}.png")
            png16_path = os.path.join(self.out_dir,
                                      f"{self._frame_index:04d}_{name}_16bit.png")
            linear_path = os.path.join(self.out_dir,
                                       f"{self._frame_index:04d}_{name}_linear.npy")
            json_path  = os.path.join(self.out_dir,
                                      f"{self._frame_index:04d}_{name}_summary.json")

            field_capture_grid_path = ""
            field_capture_strikes_path = ""
            tracer_obj = getattr(back, "tracer", None)
            if tracer_obj is not None and hasattr(tracer_obj, "get_field_capture_meta"):
                try:
                    fc_meta = tracer_obj.get_field_capture_meta()
                    if int(fc_meta.get("grid_kind", -1)) >= 0:
                        grid_reim = tracer_obj.get_field_capture_grid_reim()
                        strikes = tracer_obj.get_field_capture_strikes()
                        field_capture_grid_path = os.path.join(
                            self.out_dir,
                            f"{self._frame_index:04d}_{name}_field_capture_grid_reim.npy")
                        field_capture_strikes_path = os.path.join(
                            self.out_dir,
                            f"{self._frame_index:04d}_{name}_field_capture_strikes.npy")
                        np.save(field_capture_grid_path, grid_reim)
                        np.save(field_capture_strikes_path, strikes)
                except Exception:
                    field_capture_grid_path = ""
                    field_capture_strikes_path = ""

            r = ExposureFrameResult(
                frame_index        = self._frame_index,
                backend            = name,
                plan               = plan_dict,
                n_rays_emitted     = int(back.n_rays_accumulated),
                n_batches          = int(batches_executed),
                measured_H_J       = float(measured),
                target_H_J         = float(plan.target_H_J),
                gain_linear        = float(gain),
                gain_db            = float(gain_db),
                virtual_t_s        = float(virtual_t),
                photons_per_pixel  = float(photons_per_pix),
                snr_estimate       = float(snr),
                image_path         = png_path,
                image16_path       = png16_path,
                image_linear_path  = linear_path,
                field_object_path  = field_obj_path,
                surface_object_path = surface_obj_path,
                sensor_object_path = sensor_obj_path if sensor_obj_path else "",
                field_capture_grid_path    = field_capture_grid_path,
                field_capture_strikes_path = field_capture_strikes_path,
                summary_path       = json_path,
                frame_config_summary = frame_config_summary,
            )
            r.frame_config_summary["convergence_target_pct"] = float(conv_target_pct)
            r.frame_config_summary["convergence_error_target"] = float(conv_error_target)
            r.frame_config_summary["convergence_last_measured_pct"] = float(conv_last_measured_pct)
            r.frame_config_summary["convergence_last_error_pct"] = float(conv_last_pct)
            r.frame_config_summary["converged_early"] = bool(converged_early)
            r.frame_config_summary["convergence_drive_batches"] = bool(conv_drive_batches)

            # Burn frame details into the standard preview PNG (8-bit only).
            # Keep the 16-bit output pristine for numeric post-processing.
            detail_lv = int(r.frame_config_summary.get("detail_level", 0))
            preview_lines = _make_hud_lines(r, detail_lv) if self.show_hud else []
            img_preview = _burn_overlay_into_preview(img, preview_lines)

            # Store image data in memory for display
            r.image_data = np.clip(img_preview, 0.0, 1.0).astype(np.float32)
            r.image16_data = (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)
            r.field_integrated_data = field_obj.integrated_bands
            r.surface_integrated_data = surface_obj.integrated_bands
            if sensor_obj is not None:
                r.sensor_photons_data = sensor_obj.photons_per_pixel
                r.sensor_snr_data = sensor_obj.snr_linear

            # Only save files if explicitly enabled via --save-files
            if self.save_files:
                with self._profiler.section(f"backend_{name}_write_files"):
                    _write_png(png_path, img_preview)
                    _write_png16(png16_path, img)
                    np.save(linear_path, rgb_linear)

            if self.save_files:
                # Convert to dict but exclude non-JSON-serializable fields (numpy arrays)
                r_dict = asdict(r)
                r_dict.pop('image_data', None)
                r_dict.pop('image16_data', None)
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(r_dict, fh, indent=2)
            results.append(r)
            print(f"  [{name:>4}] N_rays={r.n_rays_emitted:_}  "
                  f"H_meas={measured:.3e} J  H_targ={plan.target_H_J:.3e} J  "
                  f"gain={gain:.3e}× ({gain_db:+.2f} dB)  "
                  f"photons/pix={photons_per_pix:.2e}  SNR≈{snr:.2f}")
            if self.save_files:
                print(f"        → {png_path}")
            else:
                print(f"        (in-memory; use --save-files to save to disk)")

        self._profiler.report(prefix=f"[frame {self._frame_index:04d}]")
        self._frame_index += 1
        self._rng_seed += 1
        self._cleanup_temp_bdpt_file()
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Tiny PNG writer — pure stdlib, no Pillow dependency
# ─────────────────────────────────────────────────────────────────────────────
def _write_png(path: str, img_hw3: np.ndarray) -> None:
    """Encode (H, W, 3) float32 in [0,1] as 8-bit RGB PNG via stdlib zlib."""
    import struct
    import zlib
    img = (np.clip(img_hw3, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)   # 8-bit RGB
    idat = zlib.compress(raw, 9)
    iend = b""
    with open(path, "wb") as fh:
        fh.write(sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat)
                 + _chunk(b"IEND", iend))


def _write_png16(path: str, img_hw3: np.ndarray) -> None:
    """Encode (H, W, 3) float32 in [0,1] as 16-bit RGB PNG via stdlib zlib."""
    import struct
    import zlib
    img = (np.clip(img_hw3, 0.0, 1.0) * 65535.0 + 0.5).astype(">u2", copy=False)
    h, w = img.shape[:2]
    # PNG 16-bit channels are network-byte-order, already satisfied by >u2.
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)  # 16-bit RGB
    idat = zlib.compress(raw, 9)
    with open(path, "wb") as fh:
        fh.write(sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat)
                 + _chunk(b"IEND", b""))


def _burn_overlay_into_preview(img_hw3: np.ndarray,
                               lines: list[str]) -> np.ndarray:
    """Burn HUD-like text into an RGB image using pygame (if available).

    Returns the original image unchanged when pygame/font rendering is not
    available, so headless runs remain functional.
    """
    if not lines:
        return img_hw3
    try:
        import pygame
    except Exception:
        return img_hw3

    try:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        img = np.asarray(img_hw3, np.float32)
        h, w = img.shape[:2]
        u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        surf = pygame.Surface((w, h))
        pygame.surfarray.blit_array(surf, np.transpose(u8, (1, 0, 2)))

        font = pygame.font.SysFont("consolas", 14)
        rendered = [font.render(str(line), True, (180, 220, 255))
                    for line in lines]
        line_h = 16
        box_h = len(rendered) * line_h + 8
        max_w = max((s.get_width() for s in rendered), default=180)
        box_w = int(max(180, min(w - 8, max_w + 10)))
        box_x = 4
        box_y = max(4, h - box_h - 4)

        panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        panel.fill((8, 8, 16, 185))
        surf.blit(panel, (box_x, box_y))

        y = box_y + 4
        for text_surf in rendered:
            surf.blit(text_surf, (box_x + 4, y))
            y += line_h

        arr = pygame.surfarray.array3d(surf)  # (W, H, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.float32) / 255.0
    except Exception:
        return img_hw3


# ─────────────────────────────────────────────────────────────────────────────
# Pygame side-by-side viewer
# ─────────────────────────────────────────────────────────────────────────────

def _make_hud_lines(r: ExposureFrameResult, detail_level: int) -> list[str]:
    """Build HUD text lines shown in the lower-left pane corner."""
    cfg = r.frame_config_summary
    lines: list[str] = [
        f"frame {r.frame_index} | {r.backend}",
        f"N_rays  {r.n_rays_emitted:_}",
    ]
    if detail_level >= 1:
        lines.append(f"gain    {r.gain_linear:.3e} ({r.gain_db:+.1f} dB)")
        lines.append(f"H_meas  {r.measured_H_J:.2e} J")
    if detail_level >= 2:
        lines.append(f"H_targ  {r.target_H_J:.2e} J")
        lines.append(f"field   {cfg.get('field', 'off')}")
    if detail_level >= 3:
        lines.append(f"spline  {cfg.get('spline', 'off')}")
        lines.append(f"cam     {cfg.get('cam_vis', '?')} / {cfg.get('transparent', '?')}")
    if detail_level >= 4:
        lines.append(f"photons {r.photons_per_pixel:.2e}/px")
        lines.append(f"splits  fi={cfg.get('fi', 0):.2f} si={cfg.get('si', 0):.2f}")
    if detail_level >= 5:
        lines.append(f"SNR\u2248    {r.snr_estimate:.2f}")
        lines.append(f"depth   {cfg.get('depth', 'off')}")
        lines.append(f"{cfg.get('description', '')}")
    return lines


def _run_viewer(session: ExposureSession, n_frames: int,
                pane_w: int, pane_h: int,
                show_hud: bool = True,
                preview_cycle_s: float = 1.5,
                preview_modes: tuple[str, ...] = ("rgb", "spectral")) -> None:
    try:
        import pygame
    except ImportError:
        print("[viewer] pygame not available; running headless and dumping PNGs only.")
        for k in range(n_frames):
            session.render_one_exposure(t=float(k) * 0.5)
        return

    pygame.init()
    win = pygame.display.set_mode((pane_w * 2 + 30, pane_h + 80))
    pygame.display.set_caption("Exposure Render Demo — C++ (left) vs GLSL (right)")
    font  = pygame.font.SysFont("consolas", 14)
    bigf  = pygame.font.SysFont("consolas", 18, bold=True)
    display_order = ("cpp", "glsl")
    allowed_modes = {
        "rgb", "spectral",
        "field-rgb", "field-spectral",
        "surface-rgb", "surface-spectral",
        "sensor-color", "sensor-spectral",
        "endpoint-rgb", "endpoint-spectral",
    }
    mode_cycle = tuple(m for m in preview_modes if m in allowed_modes) or ("rgb",)
    cycle_s = max(0.25, float(preview_cycle_s))

    def _spectral_falsecolor(accum_bhw: np.ndarray) -> np.ndarray:
        """False-color view showing spectral centroid and intensity."""
        power = np.maximum(np.asarray(accum_bhw, np.float64), 0.0)
        if power.ndim != 3 or power.shape[0] <= 0:
            return np.zeros((session.height, session.width, 3), np.float32)
        n_b = int(power.shape[0])
        axis = np.arange(n_b, dtype=np.float64)[:, None, None]
        total = power.sum(axis=0)
        centroid = (power * axis).sum(axis=0) / np.maximum(total, 1.0e-20)
        hue = centroid / max(1.0, float(n_b - 1))
        p99 = float(np.percentile(total, 99.0)) if total.size else 1.0
        p99 = max(1.0e-8, p99)
        light = _sigmoid01((total / p99 - 0.45) * 5.0)
        sat = np.full_like(light, 0.95, dtype=np.float64)
        return _hsl_to_rgb(hue, sat, light)

    def _scalar_falsecolor(img_hw: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(img_hw, np.float64), 0.0)
        p99 = float(np.percentile(x, 99.0)) if x.size else 1.0
        p99 = max(1.0e-8, p99)
        xn = np.clip(x / p99, 0.0, 1.0)
        hue = (1.0 - xn) * 0.72
        sat = np.full_like(xn, 0.9, dtype=np.float64)
        light = _sigmoid01((xn - 0.35) * 5.0)
        return _hsl_to_rgb(hue, sat, light)

    def _scalar_clear(img_hw: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(img_hw, np.float64), 0.0)
        p99 = float(np.percentile(x, 99.0)) if x.size else 1.0
        p99 = max(1.0e-8, p99)
        xn = np.clip(x / p99, 0.0, 1.0)
        y = _sigmoid01((xn - 0.35) * 5.0).astype(np.float32)
        return np.stack([y, y, y], axis=-1)

    def _missing_mode_image(mode: str) -> np.ndarray:
        img = np.zeros((session.height, session.width, 3), dtype=np.float32)
        img[:, :, 0] = 0.20
        return img

    def _mode_image_for_snapshot(mode: str,
                                 snap: IntegrationSnapshot) -> tuple[np.ndarray, str]:
        if mode == "rgb":
            return np.asarray(snap.image_data, dtype=np.float32), "rgb"
        if mode == "spectral":
            src = snap.field_integrated_data
            if src is not None:
                return _spectral_falsecolor(src), "spectral-index"
            return _missing_mode_image("spectral"), "missing:spectral-index"
        if mode == "field-rgb":
            rgb = _bands_to_rgb(snap.field_integrated_data, session.freq_hz)
            return np.asarray(rgb, dtype=np.float32), "field-rgb-visible"
        if mode == "field-spectral":
            return _spectral_falsecolor(snap.field_integrated_data), "field-spectral-index"
        if mode == "surface-rgb":
            rgb = _bands_to_rgb(snap.surface_integrated_data, session.freq_hz)
            return np.asarray(rgb, dtype=np.float32), "surface-rgb-visible"
        if mode == "surface-spectral":
            return _spectral_falsecolor(snap.surface_integrated_data), "surface-spectral-index"
        if mode == "sensor-color":
            if snap.sensor_photons_data is not None:
                return _scalar_falsecolor(snap.sensor_photons_data), "sensor-color"
            return np.asarray(snap.image_data, dtype=np.float32), "sensor-color:live-rgb"
        if mode == "sensor-spectral":
            if snap.sensor_snr_data is not None:
                return _scalar_clear(snap.sensor_snr_data), "sensor-spectral"
            return np.asarray(snap.image_data, dtype=np.float32), "sensor-spectral:live-rgb"
        if mode == "endpoint-rgb":
            return np.asarray(snap.image_data, dtype=np.float32), "endpoint-rgb:live"
        if mode == "endpoint-spectral":
            if snap.sensor_snr_data is not None:
                return _scalar_clear(snap.sensor_snr_data), "endpoint-spectral"
            return np.asarray(snap.image_data, dtype=np.float32), "endpoint-spectral:live-rgb"
        return _missing_mode_image(mode), f"missing:{mode}"

    def _available_modes_for_result(result: ExposureFrameResult) -> tuple[str, ...]:
        modes: list[str] = []
        if result.image_data is not None:
            modes.append("rgb")
        if result.field_integrated_data is not None:
            modes.extend(["field-rgb", "field-spectral"])
        if result.surface_integrated_data is not None:
            modes.extend(["surface-rgb", "surface-spectral"])
        if result.sensor_photons_data is not None:
            modes.append("sensor-color")
        if result.sensor_snr_data is not None:
            modes.append("sensor-spectral")
        if session.rgb_source == "endpoint" and result.image_data is not None:
            modes.append("endpoint-rgb")
        if session.rgb_source == "endpoint" and result.sensor_snr_data is not None:
            modes.append("endpoint-spectral")
        return tuple(modes)

    def _current_mode(elapsed_s: float) -> str:
        return mode_cycle[int(elapsed_s / cycle_s) % len(mode_cycle)]

    def _mode_image_for_result(mode: str,
                               result: ExposureFrameResult) -> tuple[np.ndarray, str]:
        if mode == "rgb":
            return result.image_data if result.image_data is not None else np.zeros((session.height, session.width, 3), np.float32), "rgb"
        if mode == "spectral":
            src = (result.field_integrated_data
                   if result.field_integrated_data is not None
                   else result.surface_integrated_data)
            if src is not None:
                return _spectral_falsecolor(src), "spectral-index"
            return _missing_mode_image("spectral"), "missing:spectral-index"
        if mode == "field-rgb":
            if result.field_integrated_data is not None:
                rgb = _bands_to_rgb(result.field_integrated_data, session.freq_hz)
                return np.asarray(rgb, dtype=np.float32), "field-rgb-visible"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "field-spectral":
            if result.field_integrated_data is not None:
                return _spectral_falsecolor(result.field_integrated_data), "field-spectral-index"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "surface-rgb":
            if result.surface_integrated_data is not None:
                rgb = _bands_to_rgb(result.surface_integrated_data, session.freq_hz)
                return np.asarray(rgb, dtype=np.float32), "surface-rgb-visible"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "surface-spectral":
            if result.surface_integrated_data is not None:
                return _spectral_falsecolor(result.surface_integrated_data), "surface-spectral-index"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "sensor-color":
            if result.sensor_photons_data is not None:
                return _scalar_falsecolor(result.sensor_photons_data), "sensor-color"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "sensor-spectral":
            if result.sensor_snr_data is not None:
                return _scalar_clear(result.sensor_snr_data), "sensor-spectral"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "endpoint-rgb":
            if session.rgb_source == "endpoint" and result.image_data is not None:
                return result.image_data, "endpoint-rgb"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "endpoint-spectral":
            if result.sensor_snr_data is not None:
                return _scalar_clear(result.sensor_snr_data), "endpoint-spectral"
            return _missing_mode_image(mode), f"missing:{mode}"
        return _missing_mode_image(mode), f"missing:{mode}"

    # Flip buffers for live preview: reuse both pixel arrays and source surfaces.
    flip_buffers: dict[str, list[np.ndarray]] = {}
    flip_surfaces: dict[str, list[Any]] = {}
    flip_index: dict[str, int] = {}
    for backend_name in display_order:
        flip_buffers[backend_name] = [
            np.empty((session.height, session.width, 3), dtype=np.uint8),
            np.empty((session.height, session.width, 3), dtype=np.uint8),
        ]
        flip_surfaces[backend_name] = [
            pygame.Surface((session.width, session.height)),
            pygame.Surface((session.width, session.height)),
        ]
        flip_index[backend_name] = 0

    def _blit_image(img: np.ndarray, dest_rect: tuple[int, int, int, int],
                    label: str, info: str,
                    hud_lines: Optional[list[str]] = None) -> None:
        """Blit a float32 [0,1] RGB image into dest_rect with optional HUD."""
        h, w = img.shape[:2]
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        surf = pygame.image.frombuffer(u8.tobytes(), (w, h), "RGB")
        surf = pygame.transform.smoothscale(surf, (dest_rect[2], dest_rect[3]))
        win.blit(surf, (dest_rect[0], dest_rect[1]))
        win.blit(bigf.render(label, True, (255, 255, 255)),
                 (dest_rect[0] + 6, dest_rect[1] + 6))
        for i, line in enumerate(info.split("\n")):
            win.blit(font.render(line, True, (210, 230, 255)),
                     (dest_rect[0] + 6,
                      dest_rect[1] + dest_rect[3] - 18 * (3 - i)))
        # ── HUD overlay — lower-left semi-transparent panel ───────────────
        if hud_lines:
            line_h = 16
            box_h  = len(hud_lines) * line_h + 8
            box_w  = 288
            box_x  = dest_rect[0] + 4
            box_y  = dest_rect[1] + dest_rect[3] - box_h - 4
            panel  = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            panel.fill((8, 8, 16, 185))
            win.blit(panel, (box_x, box_y))
            for idx, line in enumerate(hud_lines):
                txt = font.render(line, True, (180, 220, 255))
                win.blit(txt, (box_x + 4, box_y + 4 + idx * line_h))

    def _blit_progress_accum(backend_name: str,
                             accum_bhw: np.ndarray,
                             dest_rect: tuple[int, int, int, int],
                             label: str,
                             info: str,
                             mode: str) -> None:
        """Render an incremental preview from spectral accum via persistent flip buffers."""
        rgb_linear = _bands_to_rgb(accum_bhw, session.freq_hz)
        img = (_spectral_falsecolor(accum_bhw)
               if mode == "spectral"
               else session._tone_map_delicate(rgb_linear, session.integral_split))

        idx = flip_index[backend_name]
        u8buf = flip_buffers[backend_name][idx]
        np.clip(img, 0.0, 1.0, out=rgb_linear)
        np.multiply(rgb_linear, 255.0, out=rgb_linear)
        u8buf[:] = rgb_linear.astype(np.uint8)

        surf = flip_surfaces[backend_name][idx]
        pygame.surfarray.blit_array(surf, np.transpose(u8buf, (1, 0, 2)))
        surf_scaled = pygame.transform.smoothscale(surf, (dest_rect[2], dest_rect[3]))
        win.blit(surf_scaled, (dest_rect[0], dest_rect[1]))

        win.blit(bigf.render(label, True, (255, 255, 255)),
                 (dest_rect[0] + 6, dest_rect[1] + 6))
        for i, line in enumerate(info.split("\n")):
            win.blit(font.render(line, True, (210, 230, 255)),
                     (dest_rect[0] + 6,
                      dest_rect[1] + dest_rect[3] - 18 * (3 - i)))
        flip_index[backend_name] = 1 - idx

    state_lock = threading.Lock()
    stop_event = threading.Event()
    render_done = threading.Event()

    state_snapshots: dict[str, IntegrationSnapshot] = {}
    state_results: dict[str, ExposureFrameResult] = {}
    state_meta: dict[str, float | int] = {
        "frame": 0,
        "batch": 0,
        "n_batches": 0,
        "elapsed_s": 0.0,
        "telemetry_frame": 0,
        "telemetry_batch": 0,
    }
    state_telemetry: dict[str, str] = {
        "profile": "",
        "alloc": "",
    }
    # Per-backend pre-rendered uint8 image cache: {backend_name: {mode: uint8 HxWx3}}
    # Worker deposits here; UI reads here — no computation on the main thread.
    state_images: dict[str, dict[str, np.ndarray]] = {}
    # Viewer-side UI state (mutated by event loop, read by display)
    ui_mode_idx: list[int] = [0]        # index into mode_cycle (manual override)
    ui_cycle_paused: list[bool] = [False]
    ui_sensor_film_label: list[str] = [""]  # display label for current slot 0 sensor+film
    # Seed display label from current session state
    if session._sensor_film_metadata and session._sensor_film_metadata[0].get("active"):
        m = session._sensor_film_metadata[0]
        ui_sensor_film_label[0] = f"{m['sensor_name']} / {m['film_name']}"
    worker_error_summary: Optional[str] = None
    worker_error_report: Optional[str] = None
    pulse_stop = threading.Event()

    def _render_worker() -> None:
        nonlocal worker_error_summary, worker_error_report
        try:
            for frame_idx in range(n_frames):
                if stop_event.is_set():
                    break

                next_telemetry_collect_t = 0.0

                def _on_batch_preview(_backs: dict[str, ExposureBackend],
                                      snapshots: dict[str, IntegrationSnapshot],
                                      batch_idx: int,
                                      n_batches: int,
                                      elapsed_s: float) -> None:
                    nonlocal next_telemetry_collect_t
                    if n_batches > 0:
                        cadence = max(1, n_batches // 24)
                    else:
                        cadence = 6
                    if n_batches > 0 and batch_idx < n_batches and (batch_idx % cadence) != 0:
                        return

                    now = time.perf_counter()
                    collect_telemetry = bool(session.profile_enabled) and (now >= next_telemetry_collect_t)
                    if collect_telemetry:
                        next_telemetry_collect_t = now + 1.0

                    profile_msg = ""
                    alloc_msg = ""
                    if collect_telemetry:
                        profile_msg = session._profiler.format_report(
                            prefix=(f"[profile pulse frame={frame_idx+1} "
                                    f"batch={batch_idx} elapsed={elapsed_s:0.1f}s]")
                        )
                        cpp_back = _backs.get("cpp")
                        tracer_obj = getattr(cpp_back, "tracer", None) if cpp_back is not None else None
                        if tracer_obj is not None and hasattr(tracer_obj, "allocation_table"):
                            try:
                                alloc_msg = str(tracer_obj.allocation_table())
                            except Exception as exc:
                                alloc_msg = f"[allocation_table error] {exc}"

                    # Pre-render every display mode as uint16 — no computation on the UI thread.
                    fresh_images: dict[str, dict[str, np.ndarray]] = {}
                    for _bname, _snap in snapshots.items():
                        _per: dict[str, np.ndarray] = {}
                        for _m in mode_cycle:
                            try:
                                _img, _ = _mode_image_for_snapshot(_m, _snap)
                                _per[_m] = (np.clip(_img, 0.0, 1.0) * 65535.0).astype(np.uint16)
                            except Exception:
                                pass
                        fresh_images[_bname] = _per

                    with state_lock:
                        state_snapshots.clear()
                        state_snapshots.update(snapshots)
                        state_images.update(fresh_images)
                        state_meta["frame"] = int(frame_idx + 1)
                        state_meta["batch"] = int(batch_idx)
                        state_meta["n_batches"] = int(n_batches)
                        state_meta["elapsed_s"] = float(elapsed_s)
                        if collect_telemetry:
                            state_meta["telemetry_frame"] = int(frame_idx + 1)
                            state_meta["telemetry_batch"] = int(batch_idx)
                            state_telemetry["profile"] = profile_msg
                            state_telemetry["alloc"] = alloc_msg

                results = session.render_one_exposure(
                    t=float(frame_idx) * 0.5,
                    batch_preview_cb=_on_batch_preview,
                )
                with state_lock:
                    for r in results:
                        state_results[r.backend] = r
                    state_meta["frame"] = int(frame_idx + 1)
                    state_meta["batch"] = int(state_meta.get("n_batches", 0))
        except Exception as exc:
            worker_error_summary = str(exc)
            worker_error_report = traceback.format_exc()
            print(f"  [error] render worker failed: {worker_error_summary}", file=sys.stderr, flush=True)
            if worker_error_report:
                print(worker_error_report, file=sys.stderr, flush=True)
        finally:
            render_done.set()

    def _telemetry_pulse_worker() -> None:
        last_signature = ""
        while not pulse_stop.is_set():
            with state_lock:
                t_frame = int(state_meta.get("telemetry_frame", 0))
                t_batch = int(state_meta.get("telemetry_batch", 0))
                profile_msg = str(state_telemetry.get("profile", ""))
                alloc_msg = str(state_telemetry.get("alloc", ""))
            if profile_msg or alloc_msg:
                sig = f"{t_frame}:{t_batch}:{len(profile_msg)}:{len(alloc_msg)}"
                if sig != last_signature:
                    print(f"[pulse] frame={t_frame} batch={t_batch}", file=sys.stderr, flush=True)
                    if profile_msg:
                        print(profile_msg, file=sys.stderr, flush=True)
                    if alloc_msg:
                        print(alloc_msg, file=sys.stderr, flush=True)
                    last_signature = sig
            pulse_stop.wait(2.0)

    worker = threading.Thread(target=_render_worker, name="exposure-render-worker", daemon=True)
    worker.start()
    pulse_worker: Optional[threading.Thread] = None
    if session.profile_enabled:
        pulse_worker = threading.Thread(
            target=_telemetry_pulse_worker,
            name="exposure-telemetry-pulse",
            daemon=True,
        )
        pulse_worker.start()

    running = True
    t_cycle0 = time.perf_counter()
    done_hold_until: Optional[float] = None

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
                break
            if ev.type == pygame.KEYDOWN:
                # Mode cycling: left/right step through mode_cycle manually
                if ev.key == pygame.K_RIGHT:
                    ui_mode_idx[0] = (ui_mode_idx[0] + 1) % len(mode_cycle)
                    ui_cycle_paused[0] = True
                    t_cycle0 = time.perf_counter()
                elif ev.key == pygame.K_LEFT:
                    ui_mode_idx[0] = (ui_mode_idx[0] - 1) % len(mode_cycle)
                    ui_cycle_paused[0] = True
                    t_cycle0 = time.perf_counter()
                elif ev.key == pygame.K_SPACE:
                    ui_cycle_paused[0] = not ui_cycle_paused[0]
                    if not ui_cycle_paused[0]:
                        t_cycle0 = time.perf_counter()
                # Sensor cycling: [ / ] step sensor on slot 0
                elif ev.key == pygame.K_LEFTBRACKET:
                    m = session.swap_sensor_film_slot(0, sensor_delta=-1)
                    if m:
                        ui_sensor_film_label[0] = f"{m['sensor_name']} / {m['film_name']}"
                elif ev.key == pygame.K_RIGHTBRACKET:
                    m = session.swap_sensor_film_slot(0, sensor_delta=+1)
                    if m:
                        ui_sensor_film_label[0] = f"{m['sensor_name']} / {m['film_name']}"
                # Film cycling: - / = step film on slot 0
                elif ev.key == pygame.K_MINUS:
                    m = session.swap_sensor_film_slot(0, film_delta=-1)
                    if m:
                        ui_sensor_film_label[0] = f"{m['sensor_name']} / {m['film_name']}"
                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    m = session.swap_sensor_film_slot(0, film_delta=+1)
                    if m:
                        ui_sensor_film_label[0] = f"{m['sensor_name']} / {m['film_name']}"
        if not running:
            break

        with state_lock:
            snapshots = dict(state_snapshots)
            results = dict(state_results)
            cached_images = {k: dict(v) for k, v in state_images.items()}
            frame_no = int(state_meta.get("frame", 0))
            batch_idx = int(state_meta.get("batch", 0))
            n_batches = int(state_meta.get("n_batches", 0))
            elapsed_s = float(state_meta.get("elapsed_s", 0.0))

        # Mode selection: auto-cycle unless paused/manually stepped
        if ui_cycle_paused[0]:
            mode = mode_cycle[ui_mode_idx[0] % len(mode_cycle)]
        else:
            mode = mode_cycle[int((time.perf_counter() - t_cycle0) / cycle_s) % len(mode_cycle)]
            ui_mode_idx[0] = mode_cycle.index(mode)

        win.fill((12, 12, 18))

        if worker_error_summary:
            title = f"Render worker error: {worker_error_summary}"
        elif n_batches > 0:
            pct = 100.0 * float(batch_idx) / max(1, n_batches)
            title = f"Exposure {max(1, frame_no)}/{n_frames} - {pct:5.1f}% - [{mode}]{'⏸' if ui_cycle_paused[0] else ''}"
        elif frame_no > 0 and not render_done.is_set():
            title = f"Exposure {frame_no}/{n_frames} - batch {batch_idx} (open-ended) - [{mode}]{'⏸' if ui_cycle_paused[0] else ''}"
        elif render_done.is_set():
            title = f"Render complete - [{mode}]{'⏸' if ui_cycle_paused[0] else ''}"
        else:
            title = f"Exposure 1/{n_frames} - initializing - [{mode}]"

        sf_label = ui_sensor_film_label[0]
        if sf_label:
            title += f"  ·  {sf_label}"

        win.blit(bigf.render(title, True, (255, 255, 200)), (16, 8))

        for i, name in enumerate(display_order):
            x = 10 + i * (pane_w + 10)
            y = 40

            if name in snapshots:
                batch_text = (f"{batch_idx}/{n_batches}" if n_batches > 0 else f"{batch_idx}/∞")
                cycle_hint = "⏸" if ui_cycle_paused[0] else f"↻{cycle_s:.1f}s"
                info = (f"frame={frame_no}/{n_frames}  batch={batch_text}\n"
                        f"elapsed={elapsed_s:.1f}s\n"
                        f"view={mode} {cycle_hint} · stream")
                u16 = cached_images.get(name, {}).get(mode)
                if u16 is not None:
                    # Worker-rendered uint16 cache — convert to display, no computation.
                    _blit_image(
                        u16.astype(np.float32) * (1.0 / 65535.0),
                        (x, y, pane_w, pane_h),
                        label=f"{name.upper()} LIVE [{mode}]",
                        info=info,
                        hud_lines=None,
                    )
                else:
                    pygame.draw.rect(win, (40, 40, 50), (x, y, pane_w, pane_h))
                    win.blit(font.render(f"{name.upper()}: rendering…", True, (200, 200, 200)),
                             (x + 12, y + 12))
                continue

            if name in results:
                r = results[name]
                img, shown_mode = _mode_image_for_result(mode, r)
                info = (f"N_rays={r.n_rays_emitted:_}\n"
                        f"gain={r.gain_linear:.2e}× ({r.gain_db:+.2f} dB)\n"
                        f"H_meas/targ={r.measured_H_J:.2e}/{r.target_H_J:.2e} J")
                detail_lv = r.frame_config_summary.get("detail_level", 0)
                hud = _make_hud_lines(r, detail_lv) if show_hud else None
                _blit_image(
                    img,
                    (x, y, pane_w, pane_h),
                    label=f"{name.upper()} [{shown_mode}]",
                    info=info,
                    hud_lines=hud,
                )
                continue

            pygame.draw.rect(win, (40, 40, 50), (x, y, pane_w, pane_h))
            win.blit(font.render(f"{name.upper()}: waiting for first snapshot", True, (200, 200, 200)),
                     (x + 12, y + 12))

        pygame.display.flip()

        if render_done.is_set() and done_hold_until is None:
            done_hold_until = time.perf_counter() + max(1.0, cycle_s * max(1, len(mode_cycle)))
        if done_hold_until is not None and time.perf_counter() >= done_hold_until:
            running = False

        pygame.time.wait(16)

    stop_event.set()
    pulse_stop.set()
    worker.join(timeout=2.0)
    if pulse_worker is not None:
        pulse_worker.join(timeout=1.0)

    pygame.quit()

    if worker_error_summary:
        raise RuntimeError(worker_error_report or worker_error_summary)


def _load_png_rgb(path: str) -> Optional[np.ndarray]:
    try:
        import pygame
        s = pygame.image.load(path)
        s = s.convert()
        w, h = s.get_size()
        buf = pygame.image.tostring(s, "RGB")
        arr = np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
        return arr
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width",          type=int,   default=1280)
    p.add_argument("--height",         type=int,   default=720)
    p.add_argument("--input-size-scale", type=float, default=1.0,
                   help="Scale input width/height before rendering (e.g., 0.5, 1.0, 1.5).")
    p.add_argument("--output-oversample", type=int, default=1,
                   help="Internal render oversample factor; output is downsampled back to scaled size.")
    p.add_argument("--oversample-stencil", choices=("box", "polar"), default="box",
                   help="Downsample stencil for oversampled output conversion.")
    p.add_argument("--total-rays",     type=int,   default=1_000_000,
                   help="Total rays per FINISHED exposure (default 1M; "
                        "raise to 10_000_000+ for production exposures).")
    p.add_argument("--rays-per-batch", type=int,   default=50_000,
                   help="Sub-batch size; many small batches stream cleanly.")
    p.add_argument("--frames",         type=int,   default=4)
    p.add_argument("--max-bounces",    type=int,   default=8,
                   help="Per-ray bounce cap. Raised from 4 because most\n"
                        "BVH walks terminate before hitting the cap.")
    p.add_argument("--integrator",     choices=("splat", "bdpt"),
                    default="bdpt",
                   help="splat = legacy pinhole projection (image accum). "
                        "bdpt  = bidirectional path tracing — emits N rays "
                        "per registered EMISSIVE TriGroup, dumps complex "
                        "EndpointRecord array (N,16) per frame to .npy.")
    p.add_argument("--bdpt-records-cap", type=int, default=0,
                   help="Max EndpointRecords per bdpt pass. Use 0 to auto-derive from "
                        "--bdpt-intermediate-max-gb.")
    p.add_argument("--bdpt-intermediate-mode", choices=("memory", "file"), default="file",
                   help="How to stage BDPT intermediates before reduction."
                        " 'file' uses mmap-friendly .npy backing.")
    p.add_argument("--bdpt-intermediate-max-gb", type=float, default=20.0,
                   help="Hard cap for BDPT file-backed intermediate size (GB)."
                        " Larger buffers stay in memory.")
    p.add_argument("--retain-bdpt-intermediate", action="store_true",
                   help="Keep file-backed BDPT intermediate .npy files."
                        " Default behavior is ephemeral cleanup.")
    p.add_argument("--bdpt-intermediate-dir", default=None,
                   help="Directory for BDPT file-backed intermediates"
                        " (default: out-dir).")
    p.add_argument("--calibration-scene", action="store_true",
                   help="Use the tungsten-cavity blackbody calibration "
                        "scene instead of the default orbiters scene. "
                        "Equivalent to --scene-mode tungsten-cavity.")
    p.add_argument("--scene-mode",     default=None,
                   help="Scene mode passed to scene_mod.scene_for_phase "
                        "(default: orbiters; tungsten-cavity for blackbody "
                        "calibration; also supports calib-rgb-diagram / calib-bw-rgb "
                        "for black-white-primary chain validation, and calib-grid / "
                        "calib-step-wedge / calib-prism-backplate for grid, wedge, "
                        "and prism comparator scenes). Overrides --calibration-scene.")
    p.add_argument("--backend",        choices=("cpp", "glsl", "both"),
                   default="cpp")
    p.add_argument("--exposure-time-s", type=float, default=DEFAULT_FILM.exposure_time_s)
    p.add_argument("--iso",            type=float, default=DEFAULT_FILM.iso)
    p.add_argument("--focal-mm",       type=float, default=DEFAULT_OPTICS.focal_mm)
    p.add_argument("--aperture-mm",    type=float, default=DEFAULT_OPTICS.aperture_mm)
    p.add_argument("--no-window",      action="store_true",
                   help="Skip pygame window; just dump PNG/JSON to disk.")
    p.add_argument("--no-hud",         action="store_true",
                   help="Suppress the HUD overlay in the viewer window.")
    p.add_argument("--save-files",     action="store_true",
                   help="Save PNG, JSON, and NPZ artifacts to disk (opt-in). "
                        "Without this, only in-memory rendering and display is performed.")
    p.add_argument("--profile",        action="store_true",
                   help="Enable Python stage profiling and native failure breadcrumbs.")
    p.add_argument("--out-dir",        default="exposures")
    p.add_argument("--pane-w",         type=int, default=640)
    p.add_argument("--pane-h",         type=int, default=360)
    p.add_argument("--preview-cycle-s", type=float, default=1.5,
                   help="Viewer interval (seconds) for RGB/spectral auto-cycling.")
    p.add_argument(
        "--preview-modes",
        default=("rgb,spectral,field-rgb,field-spectral,"
                 "surface-rgb,surface-spectral,sensor-color,sensor-spectral,"
                 "endpoint-rgb,endpoint-spectral"),
        help=("Comma-separated viewer modes: rgb,spectral,field-rgb,"
              "field-spectral,surface-rgb,surface-spectral,"
              "sensor-color,sensor-spectral,endpoint-rgb,endpoint-spectral"),
    )
    p.add_argument(
        "--rgb-source",
        choices=("sensor", "accum", "endpoint"),
        default="sensor",
        help=("Visible RGB source: sensor uses backward-pass sensor integration; "
              "accum uses camera-integrated spectral accum; endpoint uses C++ endpoint reducer output."),
    )
    p.add_argument("--convergence-target-pct", type=float, default=99.99,
                   help="Convergence target percentage for early-stop detector.")
    p.add_argument("--convergence-max-rel-drift", type=float, default=1.0e-4,
                   help="Maximum relative drift for convergence stop criterion.")
    p.add_argument("--no-convergence", action="store_true",
                   help="Disable convergence detector and run planned finite batches.")
    p.add_argument("--no-convergence-drive-batches", action="store_true",
                   help="Keep finite batch planning even when convergence detector is enabled.")
    p.add_argument("--convergence-check-every", type=int, default=1,
                   help="Run convergence detector every N batches.")
    p.add_argument("--convergence-min-batches", type=int, default=4,
                   help="Minimum batches before convergence early-stop can trigger.")
    p.add_argument("--convergence-hold-checks", type=int, default=3,
                   help="Required consecutive convergence hits before early-stop.")
    p.add_argument("--convergence-probe-count", type=int, default=8192,
                   help="Probe sample count used by convergence detector.")
    p.add_argument("--convergence-max-batches", type=int, default=0,
                   help="Optional hard cap for convergence-driven open-ended mode (0 = unlimited).")
    p.add_argument("--adaptive-allocation", choices=("stochastic", "quota", "uniform"),
                   default="stochastic",
                   help="Adaptive allocation policy for forward and backward ray dispatch.")
    p.add_argument("--field-integrate-frac", type=float, default=0.35)
    p.add_argument("--field-bookkeep-frac", type=float, default=0.65)
    p.add_argument("--surface-integrate-frac", type=float, default=0.85)
    p.add_argument("--surface-bookkeep-frac", type=float, default=0.15)
    p.add_argument("--hdr-white-percentile", type=float, default=99.8)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    profile_enabled = bool(ENABLE_EXPOSURE_PROFILING or args.profile)
    if profile_enabled:
        faulthandler.enable(all_threads=True)
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    backends = {
        "cpp":  ("cpp",),
        "glsl": ("glsl",),
        "both": ("cpp", "glsl"),
    }[args.backend]

    optics = CameraOptics(
        focal_mm           = args.focal_mm,
        aperture_mm        = args.aperture_mm,
        pixel_pitch_um     = DEFAULT_OPTICS.pixel_pitch_um,
        sensor_w_mm        = DEFAULT_OPTICS.sensor_w_mm,
        sensor_h_mm        = DEFAULT_OPTICS.sensor_h_mm,
        lens_transmission  = DEFAULT_OPTICS.lens_transmission,
    )
    film = FilmExposure(
        iso                 = args.iso,
        exposure_time_s     = args.exposure_time_s,
        quantum_efficiency  = DEFAULT_FILM.quantum_efficiency,
        target_mid_grey     = DEFAULT_FILM.target_mid_grey,
    )

    input_scale = max(1.0e-6, float(args.input_size_scale))
    oversample = max(1, int(args.output_oversample))
    oversample_stencil = str(args.oversample_stencil)
    if bool(args.save_files) and int(args.output_oversample) == 1 and str(args.oversample_stencil) == "box":
        oversample = 4
        oversample_stencil = "polar"
        print("  [defaults] --save-files detected: using output oversample 4x with polar stencil")
    output_w = max(1, int(round(float(args.width) * input_scale)))
    output_h = max(1, int(round(float(args.height) * input_scale)))
    render_w = max(1, int(output_w * oversample))
    render_h = max(1, int(output_h * oversample))

    scene_mode_schedule = None if args.scene_mode is not None or args.calibration_scene else _build_default_scene_schedule(args.frames, "orbiters")

    session = ExposureSession(
        optics         = optics,
        film           = film,
        width          = render_w,
        height         = render_h,
        total_rays     = args.total_rays,
        rays_per_batch = args.rays_per_batch,
        max_bounces    = args.max_bounces,
        backends       = backends,
        out_dir        = args.out_dir,
        integrator     = args.integrator,
        bdpt_records_cap = args.bdpt_records_cap,
        bdpt_intermediate_mode = args.bdpt_intermediate_mode,
        bdpt_intermediate_max_bytes = int(max(0.0, args.bdpt_intermediate_max_gb) * (1024 ** 3)),
        retain_bdpt_intermediate = bool(args.retain_bdpt_intermediate),
        bdpt_intermediate_dir = args.bdpt_intermediate_dir,
        scene_mode     = (args.scene_mode if args.scene_mode is not None
                          else ("tungsten-cavity" if args.calibration_scene
                                else "orbiters")),
          scene_mode_schedule = scene_mode_schedule,
          profile_enabled = profile_enabled,
        integral_split = IntegralSplitConfig(
            field_integrate_frac   = float(args.field_integrate_frac),
            field_bookkeep_frac    = float(args.field_bookkeep_frac),
            surface_integrate_frac = float(args.surface_integrate_frac),
            surface_bookkeep_frac  = float(args.surface_bookkeep_frac),
            hdr_white_percentile   = float(args.hdr_white_percentile),
        ),
        convergence     = ConvergenceConfig(
            enabled             = not bool(args.no_convergence),
            drive_batches       = not bool(args.no_convergence_drive_batches),
            target_pct          = float(args.convergence_target_pct),
            max_rel_drift       = float(args.convergence_max_rel_drift),
            check_every_batches = int(args.convergence_check_every),
            min_batches         = int(args.convergence_min_batches),
            hold_checks         = int(args.convergence_hold_checks),
            probe_count         = int(args.convergence_probe_count),
            max_batches         = int(args.convergence_max_batches),
        ),
        adaptive_allocation_mode = str(args.adaptive_allocation),
        n_frames_planned = args.frames,
        output_width     = output_w,
        output_height    = output_h,
        output_oversample_stencil = oversample_stencil,
        show_hud         = not args.no_hud,
        rgb_source       = args.rgb_source,
        save_files       = args.save_files,
    )

    if args.no_window:
        for k in range(args.frames):
            session.render_one_exposure(t=float(k) * 0.5)
    else:
        preview_modes = tuple(
            token.strip().lower() for token in str(args.preview_modes).split(",") if token.strip()
        )
        _run_viewer(session, args.frames, args.pane_w, args.pane_h,
                    show_hud=not args.no_hud,
                    preview_cycle_s=float(args.preview_cycle_s),
                    preview_modes=preview_modes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
