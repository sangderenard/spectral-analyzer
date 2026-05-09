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
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

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
    from bdpt_integrator import CameraSensor, TriangleGroup
    from bdpt_integrator import TRI_GROUP_ROLE_SENSOR, TRI_GROUP_SAMPLE_PIXEL_CONE
    _HAS_BDPT_INTEGRATION = True
except ImportError:
    _HAS_BDPT_INTEGRATION = False
    CameraSensor = None
    TriangleGroup = None
    TRI_GROUP_ROLE_SENSOR = None
    TRI_GROUP_SAMPLE_PIXEL_CONE = None


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
        # Persistent per-exposure accumulator (n_bands, H, W) in float32.
        self.accum = np.zeros((self.n_bands, cam.height, cam.width), np.float32)
        self.n_rays_accumulated = 0

    # ── Sub-class hooks ──────────────────────────────────────────────────
    def reset_exposure(self) -> None:
        self.accum.fill(0.0)
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

    def render_batch(self, n_rays: int, seed: int) -> None:
        # Ray preparation randomness is externalized here: every emissive
        # receives a fresh launch site and axis each batch before tracing.
        src_pos, src_dir = self._prepare_emissive_batch_sources(seed)
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
            # Track the C++ accumulator so the right pane has something
            # spectrally meaningful to show during calibration testing.
            self.accum[:] = self._mirror.accum
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
    integral_split:    IntegralSplitConfig
    description:       str = ""
    detail_level:      int = 0   # 0-5; drives HUD verbosity


def _build_frame_config(frame_idx: int,
                        n_frames_total: int,
                        base_split: IntegralSplitConfig) -> FrameConfig:
    """Build the per-frame feature config according to the progressive schedule.

    When n_frames_total == 1 only level 0 (baseline) is used.  Each additional
    frame slot unlocks the next feature level up to MAX_LEVELS - 1.

    Level 0  baseline — no field cap, AS_IS camera vis, no spline
    Level 1  field regular 64³ + strike capture
    Level 2  field regular 96³ + DIRECT_HIT + emissive-only spline
    Level 3  field kdtree 64³ + DIRECT_HIT + XRAY + full-mesh spline
    Level 4  field regular 128³ + FULL_MARCH + XRAY + depth cull + ridge spline
    Level 5  field kdtree 128³ + FULL_MARCH + XRAY + max ridge + field-heavy
    """
    MAX_LEVELS = 6
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
        # level 0 — pure baseline
        FrameConfig(
            field_capture     = FieldCaptureConfig(enabled=False),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_AS_IS,
                transparent_mode = _CAM_TRANSP_BLOCK,
            ),
            surface_spline    = SurfaceSplineConfig(enabled=False),
            integral_split    = _split(),
            description       = "baseline · no field · AS_IS cam",
            detail_level      = 0,
        ),
        # level 1 — field capture, regular 64³
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=64, ny=64, nz=64,
                capture_strikes=True, max_strikes=500_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_AS_IS,
                transparent_mode = _CAM_TRANSP_BLOCK,
            ),
            surface_spline    = SurfaceSplineConfig(enabled=False),
            integral_split    = _split(),
            description       = "field regular-64³ · AS_IS cam",
            detail_level      = 1,
        ),
        # level 2 — field 96³ + emissive spline + DIRECT_HIT
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=96, ny=96, nz=96,
                capture_strikes=True, max_strikes=750_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_DIRECT_HIT,
                transparent_mode = _CAM_TRANSP_BLOCK,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=False, ridge_lambda=0.0,
            ),
            integral_split    = _split(dfi=0.10, dsi=0.0),
            description       = "field 96³ · spline emissive · DIRECT_HIT",
            detail_level      = 2,
        ),
        # level 3 — field kdtree 64³ + full-mesh spline + DIRECT_HIT + XRAY
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="kdtree",
                nx=64, ny=64, nz=64,
                capture_strikes=True, max_strikes=1_000_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_DIRECT_HIT,
                transparent_mode = _CAM_TRANSP_XRAY,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=True, ridge_lambda=0.0,
            ),
            integral_split    = _split(dfi=0.20, dsi=0.05),
            description       = "field kdtree-64³ · spline all · DIRECT_HIT+XRAY",
            detail_level      = 3,
        ),
        # level 4 — FULL_MARCH + depth cull + ridge spline + regular 128³
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
            integral_split    = _split(dfi=0.25, dsi=0.10, dwp=-0.3),
            description       = "FULL_MARCH 128³ · spline ridge 1e-4 · depth 50m",
            detail_level      = 4,
        ),
        # level 5 — FULL_MARCH + XRAY + kdtree 128³ + max ridge + field-heavy
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
            integral_split    = _split(dfi=0.35, dsi=0.12, dwp=-0.8),
            description       = "FULL_MARCH kdtree · XRAY · ridge 1e-3 · depth 100m",
            detail_level      = 5,
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
                 bdpt_records_cap: int = 1_048_576,
                 scene_mode: str = "orbiters",
                 integral_split: Optional[IntegralSplitConfig] = None,
                 camera_visibility: Optional[CameraVisibilityConfig] = None,
                 field_capture: Optional[FieldCaptureConfig] = None,
                 n_frames_planned: int = 1,
                 show_hud: bool = True,
                 sensor_film_slots: Optional[list[tuple[int, int]]] = None,
                 save_files: bool = False):
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
        self.bdpt_records_cap = int(bdpt_records_cap)
        self.scene_mode = str(scene_mode)
        self.integral_split = (integral_split
                       if integral_split is not None
                       else IntegralSplitConfig())
        self.camera_visibility = (camera_visibility
                      if camera_visibility is not None
                      else CameraVisibilityConfig())
        self.field_capture = (field_capture
                      if field_capture is not None
                      else FieldCaptureConfig())
        self.n_frames_planned = max(1, int(n_frames_planned))
        self.show_hud = bool(show_hud)
        self.save_files = bool(save_files)
        os.makedirs(out_dir, exist_ok=True)

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

        self._rng_seed = 1
        self._frame_index = 0
        self._sensor_group_id = -1  # BDPT sensor group ID for this frame

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
            )
            if hasattr(cpp_b.tracer, "set_camera_visibility"):
                cv = frame_cfg.camera_visibility
                cpp_b.tracer.set_camera_visibility(
                    camera_vis_mode    = int(cv.camera_vis_mode),
                    transparent_mode   = int(cv.transparent_mode),
                    depth_cull_enabled = bool(cv.depth_cull_enabled),
                    depth_cull_m       = float(cv.depth_cull_m),
                )
            fc = frame_cfg.field_capture
            if fc.enabled and hasattr(cpp_b.tracer, "enable_field_capture_regular"):
                bmin = np.asarray(scene.bounds_min, np.float32)
                bmax = np.asarray(scene.bounds_max, np.float32)
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

    # ── Calibration: gain to match measured H to target H ────────────────
    @staticmethod
    def _train_emissivity_gain(measured_H_J: float, target_H_J: float) -> float:
        if measured_H_J <= 0.0 or target_H_J <= 0.0:
            return 1.0
        return float(target_H_J / measured_H_J)

    def _make_integral_objects(self,
                               backend: str,
                               accum_bhw: np.ndarray,
                               gain: float,
                               integral_split: Optional[IntegralSplitConfig] = None
                               ) -> tuple[FieldIntegralObject,
                                          SurfaceIntegralObject,
                                          np.ndarray,
                                          np.ndarray]:
        """Build field/surface integral objects and return merged band tensor."""
        cfg = integral_split if integral_split is not None else self.integral_split
        a = np.asarray(accum_bhw, np.float64) * float(gain)

        fi = float(np.clip(cfg.field_integrate_frac, 0.0, 1.0))
        fb = float(np.clip(cfg.field_bookkeep_frac, 0.0, 1.0))
        si = float(np.clip(cfg.surface_integrate_frac, 0.0, 1.0))
        sb = float(np.clip(cfg.surface_bookkeep_frac, 0.0, 1.0))

        # A light-touch branch: field tracks diffuse/global energy envelope,
        # surface tracks direct projected sensor-space accumulation.
        band_mean = a.mean(axis=(1, 2), keepdims=True)
        field_integrated = fi * np.broadcast_to(band_mean, a.shape)
        field_bookkeep = fb * np.maximum(0.0, a - field_integrated)
        surf_integrated = si * np.maximum(0.0, a - field_integrated)
        surf_bookkeep = sb * np.maximum(0.0, a - surf_integrated)

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
        white = float(np.percentile(x, float(np.clip(cfg.hdr_white_percentile, 75.0, 100.0))))
        white = max(white, 1.0e-8)
        y = np.log1p(x / white * 6.0) / math.log1p(6.0)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    def _make_sensor_integral(self, backend: str, gain: float) -> Optional[SensorIntegralObject]:
        """Create a SensorIntegralObject from endpoint-derived accumulation (T4).
        
        For Phase 3C, this implements slot-0-only endpoint-driven photon/electron/SNR accumulation.
        Requires: BDPT endpoint records with position and spectral amplitude.
        
        Returns SensorIntegralObject or None if sensor integration unavailable.
        """
        try:
            # Only accumulate if we have active slots and have just run BDPT
            active_slots = [i for i, (s, f) in enumerate(self.sensor_film_slots) if s >= 0 and f >= 0]
            if not active_slots:
                print("  [warn] no active sensor/film slots; sensor integral unavailable")
                return None
            
            # For now, Phase 3C (immediate): implement slot 0 only
            slot_id = 0
            sensor_id, film_id = self.sensor_film_slots[slot_id]
            
            if sensor_id < 0 or film_id < 0:
                return None
            
            # Get sensor and film parameters from tensors
            sensor_row = self._sensor_film_tensors['sensor'][sensor_id]
            film_row = self._sensor_film_tensors['film'][film_id]
            
            qe_peak = float(sensor_row[8])  # offset 8: qe_peak
            read_noise_e = float(sensor_row[10])  # offset 10: read_noise_e
            dark_current_e_s = float(sensor_row[11])  # offset 11: dark_current_e_s
            full_well_e = float(sensor_row[9])  # offset 9: full_well_e
            
            exposure_time_s = float(film_row[1])  # offset 1: exposure_time_s
            
            # Initialize accumulator arrays
            photons_per_pixel = np.zeros((self.height, self.width), dtype=np.float32)
            
            # Placeholder for Phase 3C (immediate): uniform distribution pending endpoint integration
            # TODO Phase 3E: Replace with actual endpoint record filtering and accumulation
            # For now, use a synthetic pattern to validate the pipeline
            photons_per_pixel = np.ones((self.height, self.width), dtype=np.float32) * 1000.0
            
            # Apply QE to get electrons
            electrons_per_pixel = photons_per_pixel * qe_peak
            
            # Compute SNR from explicit noise model (T5)
            # SNR = signal / noise = sqrt(electrons) / sqrt(read_noise^2 + dark_current*t + electrons)
            dark_current_accumulated_e = dark_current_e_s * exposure_time_s
            noise_variance = (read_noise_e ** 2) + dark_current_accumulated_e + electrons_per_pixel
            snr_linear = np.sqrt(np.maximum(electrons_per_pixel, 0.0)) / np.sqrt(np.maximum(noise_variance, 1.0e-10))
            
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
                "slot_id": slot_id,
                "sensor_id": sensor_id,
                "film_id": film_id,
                "sensor_name": self._sensor_film_metadata[slot_id].get("sensor_name", "unknown"),
                "film_name": self._sensor_film_metadata[slot_id].get("film_name", "unknown"),
                "qe_peak": float(qe_peak),
                "read_noise_e": float(read_noise_e),
                "dark_current_e_s": float(dark_current_e_s),
                "dark_current_accumulated_e": float(dark_current_accumulated_e),
                "exposure_time_s": float(exposure_time_s),
                "full_well_e": float(full_well_e),
                "snr_peak": float(peak_snr),
                "snr_mean": float(mean_snr),
                "photons_flux_hz": float(np.mean(photons_per_pixel)),
                "electrons_flux_hz": float(np.mean(electrons_per_pixel)),
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
                full_well_e=int(full_well_e),
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
                               sensor_obj: Optional[SensorIntegralObject] = None
                               ) -> tuple[str, str, Optional[str]]:
        """Write field/surface/sensor integral objects to npz artifacts."""
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

        np.savez_compressed(
            field_path,
            freq_hz=field_obj.freq_hz,
            integrated_bands=field_obj.integrated_bands,
            bookkeeping_bands=field_obj.bookkeeping_bands,
            metrics=np.asarray(json.dumps(field_obj.metrics)),
        )
        np.savez_compressed(
            surf_path,
            freq_hz=surface_obj.freq_hz,
            integrated_bands=surface_obj.integrated_bands,
            bookkeeping_bands=surface_obj.bookkeeping_bands,
            metrics=np.asarray(json.dumps(surface_obj.metrics)),
        )
        if sensor_path is not None and sensor_obj is not None:
            np.savez_compressed(
                sensor_path,
                photons_per_pixel=sensor_obj.photons_per_pixel,
                electrons_per_pixel=sensor_obj.electrons_per_pixel,
                snr_linear=sensor_obj.snr_linear,
                metrics=np.asarray(json.dumps(sensor_obj.metrics)),
            )
        
        return field_path, surf_path, sensor_path

    # ── Render one full exposure (drives all sub-batches) ────────────────
    def render_one_exposure(self, t: float = 0.0) -> list[ExposureFrameResult]:
        # ── Build per-frame progressive feature configuration ────────────
        frame_cfg = _build_frame_config(
            self._frame_index, self.n_frames_planned, self.integral_split)

        scene = _build_tracer_scene(t, scene_mode=self.scene_mode)
        cam   = PinholeCamera.looking_at_scene(self.width, self.height, self.optics)
        plan  = self._build_plan(scene)
        backs = self._build_backends(scene, cam, frame_cfg)

        plan_dict = summarize_plan(plan, self.optics, self.film)
        n_pix = max(1, self.optics.n_pixels())
        # Visible-band reference: λ ≈ 555 nm (peak of photopic response).
        photon_E = H_PLANCK * C_LIGHT / (plan.ref_wavelength_nm * 1.0e-9)
        photons_per_pix = plan.target_H_J / max(1, n_pix) / max(photon_E, 1.0e-30)

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
        print(f"  budget (total)       : N_rays={plan.total_rays:_}  "
              f"n_batches={plan.n_batches}  rays/batch={plan.rays_per_batch:_}")
        print(f"  per-pixel target     : H={plan.target_H_J/n_pix:.3e} J  "
              f"photons={photons_per_pix:.3e} @ λ={plan.ref_wavelength_nm:.0f} nm")
        print(f"  shutter t            : {self.film.exposure_time_s:.4g} s @ "
              f"f/{self.optics.f_number:.2f}, ISO {self.film.iso:.0f}")

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

        t0 = time.perf_counter()
        for b_idx in range(plan.n_batches):
            seed = (self._rng_seed * 1_000_003) + b_idx + 1
            for back in backs.values():
                back.render_batch(rays_per_source_per_batch, seed)
            if (b_idx + 1) % max(1, plan.n_batches // 10) == 0:
                elapsed = time.perf_counter() - t0
                pct = 100.0 * (b_idx + 1) / plan.n_batches
                print(f"    batch {b_idx+1:>5}/{plan.n_batches}  "
                      f"({pct:5.1f}%)  elapsed={elapsed:6.2f}s")
        elapsed = time.perf_counter() - t0
        print(f"  {plan.n_batches} batches in {elapsed:.2f}s "
              f"({(plan.n_batches/max(elapsed,1e-9)):.1f} batches/s)")

        # ── Optional bidirectional pass ──────────────────────────────────
        # Per the integrator-rewrite directive: complex EndpointRecords are
        # written verbatim, no abs(), no quantize, no band collapse.
        # We register both EMISSIVE and SENSOR groups here so the camera
        # PIXEL_CONE path is active by default for exposure calibration.
        if self.integrator == "bdpt" and "cpp" in backs:
            cpp_back = backs["cpp"]  # type: ignore[assignment]
            tracer = getattr(cpp_back, "tracer", None)
            if tracer is not None and hasattr(tracer, "register_tri_group"):
                tracer.clear_tri_groups()
                emissive_tris = np.ascontiguousarray(scene.src_tri_idx, dtype=np.int32)

                # ── Surface spline fitting ────────────────────────────────
                ss_cfg = frame_cfg.surface_spline
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

                if emissive_tris.size > 0:
                    if spline_coeffs is not None and ss_cfg.fit_all_tris:
                        # Per-triangle groups — each emissive tri gets individual coefficients.
                        n_tri_total = scene.verts.shape[0]
                        for ti in emissive_tris:
                            ti_int = int(ti)
                            if ti_int < 0 or ti_int >= n_tri_total:
                                continue
                            c6 = spline_coeffs[ti_int].ravel().astype(np.float64)
                            tracer.register_tri_group(
                                role_bits          = 1,
                                sample_policy      = 1,
                                tri_indices        = np.asarray([ti], np.int32),
                                parametric_surface = {"kind": _TRI_POLY_BARY, "coeffs": c6},
                            )
                        print(f"  bdpt: {emissive_tris.size} per-tri emissive groups "
                              f"with POLY_BARY spline")
                    elif spline_coeffs is not None:
                        # Single group with mean coefficients across emissive tris.
                        valid_ids = emissive_tris[emissive_tris < spline_coeffs.shape[0]]
                        mean_c6 = (spline_coeffs[valid_ids].mean(axis=0)
                                   if valid_ids.size > 0 else np.zeros(6, np.float64))
                        tracer.register_tri_group(
                            role_bits          = 1,
                            sample_policy      = 1,
                            tri_indices        = emissive_tris,
                            parametric_surface = {
                                "kind": _TRI_POLY_BARY,
                                "coeffs": mean_c6.ravel().astype(np.float64),
                            },
                        )
                        print("  bdpt: 1 emissive group with mean POLY_BARY spline")
                    else:
                        tracer.register_tri_group(
                            role_bits     = 1,
                            sample_policy = 1,
                            tri_indices   = emissive_tris,
                        )

                # ── Sensor group (PIXEL_CONE) ─────────────────────────────
                all_tris = np.ascontiguousarray(
                    np.arange(scene.verts.shape[0], dtype=np.int32))
                sensor_w_m = float(self.optics.sensor_w_mm) * 1.0e-3
                sensor_h_m = float(self.optics.sensor_h_mm) * 1.0e-3
                focal_m = float(np.linalg.norm(
                    np.asarray(scene_mod.SCENE_CENTER, np.float64) - cam.pos))
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
                
                # Bind sensor/film SSBO before dispatch (T3)
                self._bind_sensor_film_ssbo(tracer)
                
                t_bd = time.perf_counter()
                rays_per_emitter = max(64, self.total_rays //
                                       max(1, tracer.n_tri_groups()) // 16)
                recs = tracer.bidirectional(
                    n_rays_per_emitter = rays_per_emitter,
                    max_bounces        = self.max_bounces,
                    min_amplitude      = 1.0e-3,
                    seed               = self._rng_seed,
                    max_records        = self.bdpt_records_cap,
                )
                print(f"  bdpt: {recs.shape[0]:_} EndpointRecords "
                      f"in {time.perf_counter()-t_bd:.2f}s "
                      f"(tri_groups={tracer.n_tri_groups()}, "
                      f"rays/emitter={rays_per_emitter:_})")
                bdpt_path = os.path.join(
                    self.out_dir, f"bdpt_records_{self._frame_index:04d}.npy")
                np.save(bdpt_path, recs)
                print(f"  bdpt records saved → {bdpt_path}")

        # ── Build human-readable frame config summary (HUD + JSON) ───────
        fc_s = frame_cfg.field_capture
        cv_s = frame_cfg.camera_visibility
        ss_s = frame_cfg.surface_spline
        is_s = frame_cfg.integral_split
        frame_config_summary = {
            "description":   frame_cfg.description,
            "detail_level":  int(frame_cfg.detail_level),
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
            "fi": float(is_s.field_integrate_frac),
            "si": float(is_s.surface_integrate_frac),
            "fb": float(is_s.field_bookkeep_frac),
            "sb": float(is_s.surface_bookkeep_frac),
            "hdr_wp": float(is_s.hdr_white_percentile),
        }

        # ── Per-backend calibration + dump ───────────────────────────────
        results: list[ExposureFrameResult] = []
        for name, back in backs.items():
            measured = back.measured_radiant_exposure_J(plan.energy_per_ray_J)
            gain     = self._train_emissivity_gain(measured, plan.target_H_J)
            gain_db  = 20.0 * math.log10(max(gain, 1.0e-12))
            virtual_t = self.film.exposure_time_s
            qe = float(self.film.quantum_efficiency)
            snr = math.sqrt(max(photons_per_pix * qe, 0.0))

            field_obj, surface_obj, _merged_bands, rgb_linear = \
                self._make_integral_objects(name, back.accum, gain,
                                            frame_cfg.integral_split)
            sensor_obj = self._make_sensor_integral(name, gain)
            
            field_obj_path, surface_obj_path, sensor_obj_path = \
                self._save_integral_objects(field_obj, surface_obj, sensor_obj)

            img = self._tone_map_delicate(rgb_linear, frame_cfg.integral_split)
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
                n_batches          = int(plan.n_batches),
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

            # Burn frame details into the standard preview PNG (8-bit only).
            # Keep the 16-bit output pristine for numeric post-processing.
            detail_lv = int(r.frame_config_summary.get("detail_level", 0))
            preview_lines = _make_hud_lines(r, detail_lv) if self.show_hud else []
            img_preview = _burn_overlay_into_preview(img, preview_lines)

            # Store image data in memory for display
            r.image_data = np.clip(img_preview, 0.0, 1.0).astype(np.float32)
            r.image16_data = (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)

            # Only save files if explicitly enabled via --save-files
            if self.save_files:
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

        self._frame_index += 1
        self._rng_seed += 1
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
                show_hud: bool = True) -> None:
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

    last: dict[str, ExposureFrameResult] = {}
    running = True
    frame = 0
    while running and frame < n_frames:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
                break
        if not running:
            break

        win.fill((12, 12, 18))
        win.blit(bigf.render(f"Exposure {frame+1}/{n_frames} — accumulating…",
                             True, (255, 255, 200)), (16, 8))
        pygame.display.flip()

        results = session.render_one_exposure(t=float(frame) * 0.5)
        for r in results:
            last[r.backend] = r

        win.fill((12, 12, 18))
        win.blit(bigf.render(f"Exposure {frame+1}/{n_frames}",
                             True, (255, 255, 255)), (16, 8))

        # Display images from memory (not from disk)
        for i, (name, _) in enumerate([("cpp", None), ("glsl", None)]):
            x = 10 + i * (pane_w + 10)
            y = 40
            if name not in last or last[name].image_data is None:
                pygame.draw.rect(win, (40, 40, 50), (x, y, pane_w, pane_h))
                win.blit(font.render(f"{name.upper()}: not available",
                                     True, (200, 200, 200)),
                         (x + 12, y + 12))
                continue

            r = last[name]
            img = r.image_data
            info = (f"N_rays={r.n_rays_emitted:_}\n"
                    f"gain={r.gain_linear:.2e}× ({r.gain_db:+.2f} dB)\n"
                    f"H_meas/targ={r.measured_H_J:.2e}/{r.target_H_J:.2e} J")
            detail_lv = r.frame_config_summary.get("detail_level", 0)
            hud = _make_hud_lines(r, detail_lv) if show_hud else None
            _blit_image(img, (x, y, pane_w, pane_h),
                        label=name.upper(), info=info, hud_lines=hud)

        pygame.display.flip()
        frame += 1
        time.sleep(0.1)

    pygame.quit()


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
    p.add_argument("--bdpt-records-cap", type=int, default=1_048_576,
                   help="Max EndpointRecords per bdpt pass.")
    p.add_argument("--calibration-scene", action="store_true",
                   help="Use the tungsten-cavity blackbody calibration "
                        "scene instead of the default orbiters scene. "
                        "Equivalent to --scene-mode tungsten-cavity.")
    p.add_argument("--scene-mode",     default=None,
                   help="Scene mode passed to scene_mod.scene_for_phase "
                        "(default: orbiters; tungsten-cavity for blackbody "
                        "calibration). Overrides --calibration-scene.")
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
    p.add_argument("--out-dir",        default="exposures")
    p.add_argument("--pane-w",         type=int, default=640)
    p.add_argument("--pane-h",         type=int, default=360)
    p.add_argument("--field-integrate-frac", type=float, default=0.35)
    p.add_argument("--field-bookkeep-frac", type=float, default=0.65)
    p.add_argument("--surface-integrate-frac", type=float, default=0.85)
    p.add_argument("--surface-bookkeep-frac", type=float, default=0.15)
    p.add_argument("--hdr-white-percentile", type=float, default=99.8)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
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

    session = ExposureSession(
        optics         = optics,
        film           = film,
        width          = args.width,
        height         = args.height,
        total_rays     = args.total_rays,
        rays_per_batch = args.rays_per_batch,
        max_bounces    = args.max_bounces,
        backends       = backends,
        out_dir        = args.out_dir,
        integrator     = args.integrator,
        bdpt_records_cap = args.bdpt_records_cap,
        scene_mode     = (args.scene_mode if args.scene_mode is not None
                          else ("tungsten-cavity" if args.calibration_scene
                                else "orbiters")),
        integral_split = IntegralSplitConfig(
            field_integrate_frac   = float(args.field_integrate_frac),
            field_bookkeep_frac    = float(args.field_bookkeep_frac),
            surface_integrate_frac = float(args.surface_integrate_frac),
            surface_bookkeep_frac  = float(args.surface_bookkeep_frac),
            hdr_white_percentile   = float(args.hdr_white_percentile),
        ),
        n_frames_planned = args.frames,
        show_hud         = not args.no_hud,
        save_files       = args.save_files,
    )

    if args.no_window:
        for k in range(args.frames):
            session.render_one_exposure(t=float(k) * 0.5)
    else:
        _run_viewer(session, args.frames, args.pane_w, args.pane_h,
                    show_hud=not args.no_hud)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
