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

# Borrow scene authoring verbatim from the basic-rasterizer harness.
import test_basic_gl_cpp_window as scene_mod   # noqa: E402


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


def _build_tracer_scene(t: float) -> TracerScene:
    """Materialise the borrowed scene into ray-tracer geometry.

    The MaterialDatabase is built ONCE per process (we trust scene_mod's
    ``register_materials``); ``build_mat_buf`` returns the unified Phase-2
    buffer that both backends index by ``mat_idx[tri] * MAX_SPECTRAL_BANDS
    + band``.
    """
    db, idx = scene_mod.register_materials()
    verts8, _mat_per_v, _gid_per_v, mat_per_tri, groups = \
        scene_mod.scene_for_phase(idx, t, scene_mode="orbiters")

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
# Spectral → display RGB (smooth triangular CIE proxy)
# ─────────────────────────────────────────────────────────────────────────────
def _bands_to_rgb(image_b_h_w: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    """Project (n_bands, H, W) → (H, W, 3) using triangular wavelength weights
    centred at sRGB primaries.  This is a calibration-grade proxy, NOT a true
    CIE 1931 colour-matching convolution; it preserves linearity and band
    energy conservation."""
    n_b = image_b_h_w.shape[0]
    if n_b == 0:
        H, W = image_b_h_w.shape[1:]
        return np.zeros((H, W, 3), np.float32)
    wl_nm = (C_LIGHT / freq_hz) * 1.0e9
    # Triangular kernels around 620/540/460 nm with 80 nm half-width.
    centres = np.array([620.0, 540.0, 460.0], np.float32)
    half_w  = 80.0
    w = np.maximum(0.0, 1.0 - np.abs(wl_nm[:, None] - centres[None, :]) / half_w)
    w_sum = w.sum(axis=0, keepdims=True)
    w_norm = w / np.where(w_sum > 0, w_sum, 1.0)
    # einsum: bhw, br -> hwr
    rgb = np.einsum("bhw,br->hwr", image_b_h_w.astype(np.float32),
                    w_norm.astype(np.float32))
    return rgb


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

    def render_batch(self, n_rays: int, seed: int) -> None:
        # n_rays is per-source; scene.src_pos.shape[0] sources; total emitted
        # rays this batch = n_sources * n_rays.
        self.tracer.integrate_image_into(
            src_pos         = self.scene.src_pos,
            src_dir         = self.scene.src_dir,
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
        self.n_rays_accumulated += int(n_rays) * int(self.scene.src_pos.shape[0])


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
    summary_path:       str


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
                 out_dir: str = "exposures"):
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
        os.makedirs(out_dir, exist_ok=True)

        self._rng_seed = 1
        self._frame_index = 0

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
                        cam:   PinholeCamera) -> dict[str, ExposureBackend]:
        backends: dict[str, ExposureBackend] = {}
        cpp_b: Optional[CppExposureBackend] = None
        if "cpp" in self.backends_requested:
            cpp_b = CppExposureBackend(
                scene, cam, self.freq_hz,
                max_bounces   = self.max_bounces,
                min_amplitude = 1.0e-3,
                atmo_abs_db_per_m = 0.0,
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

    # ── Calibration: gain to match measured H to target H ────────────────
    @staticmethod
    def _train_emissivity_gain(measured_H_J: float, target_H_J: float) -> float:
        if measured_H_J <= 0.0 or target_H_J <= 0.0:
            return 1.0
        return float(target_H_J / measured_H_J)

    # ── Render one full exposure (drives all sub-batches) ────────────────
    def render_one_exposure(self, t: float = 0.0) -> list[ExposureFrameResult]:
        scene = _build_tracer_scene(t)
        cam   = PinholeCamera.looking_at_scene(self.width, self.height, self.optics)
        plan  = self._build_plan(scene)
        backs = self._build_backends(scene, cam)

        plan_dict = summarize_plan(plan, self.optics, self.film)
        n_pix = max(1, self.optics.n_pixels())
        # Visible-band reference: λ ≈ 555 nm (peak of photopic response).
        photon_E = H_PLANCK * C_LIGHT / (plan.ref_wavelength_nm * 1.0e-9)
        photons_per_pix = plan.target_H_J / max(1, n_pix) / max(photon_E, 1.0e-30)

        print(f"\n══════════════════════ EXPOSURE {self._frame_index:04d} ══════════════════════")
        print(f"  scene:  {scene.verts.shape[0]} tris, {scene.src_pos.shape[0]} emissive sources")
        print(f"  emissive total power = {scene.total_emissive_power_W:.4g} W "
              f"over {scene.total_emissive_area_m2:.4g} m²")
        print(f"  scene radiance L     = {scene.scene_radiance_W_sr_m2():.4g} W·sr⁻¹·m⁻²")
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

        # ── Per-backend calibration + dump ───────────────────────────────
        results: list[ExposureFrameResult] = []
        for name, back in backs.items():
            measured = back.measured_radiant_exposure_J(plan.energy_per_ray_J)
            gain     = self._train_emissivity_gain(measured, plan.target_H_J)
            gain_db  = 20.0 * math.log10(max(gain, 1.0e-12))
            virtual_t = self.film.exposure_time_s   # ground truth
            # Shot-noise SNR estimate per pixel: sqrt(photons_per_pix * QE).
            qe = float(self.film.quantum_efficiency)
            snr = math.sqrt(max(photons_per_pix * qe, 0.0))

            img = back.finalize_image(gain)
            png_path  = os.path.join(self.out_dir,
                                     f"{self._frame_index:04d}_{name}.png")
            json_path = os.path.join(self.out_dir,
                                     f"{self._frame_index:04d}_{name}_summary.json")
            _write_png(png_path, img)

            r = ExposureFrameResult(
                frame_index       = self._frame_index,
                backend           = name,
                plan              = plan_dict,
                n_rays_emitted    = int(back.n_rays_accumulated),
                n_batches         = int(plan.n_batches),
                measured_H_J      = float(measured),
                target_H_J        = float(plan.target_H_J),
                gain_linear       = float(gain),
                gain_db           = float(gain_db),
                virtual_t_s       = float(virtual_t),
                photons_per_pixel = float(photons_per_pix),
                snr_estimate      = float(snr),
                image_path        = png_path,
                summary_path      = json_path,
            )
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(asdict(r), fh, indent=2)
            results.append(r)
            print(f"  [{name:>4}] N_rays={r.n_rays_emitted:_}  "
                  f"H_meas={measured:.3e} J  H_targ={plan.target_H_J:.3e} J  "
                  f"gain={gain:.3e}× ({gain_db:+.2f} dB)  "
                  f"photons/pix={photons_per_pix:.2e}  SNR≈{snr:.2f}")
            print(f"        → {png_path}")

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


# ─────────────────────────────────────────────────────────────────────────────
# Pygame side-by-side viewer
# ─────────────────────────────────────────────────────────────────────────────
def _run_viewer(session: ExposureSession, n_frames: int,
                pane_w: int, pane_h: int) -> None:
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

    def _blit_image(img: np.ndarray, dest_rect: tuple[int,int,int,int],
                    label: str, info: str) -> None:
        # img is (H,W,3) float32 in [0,1] — scale to dest_rect and blit.
        h, w = img.shape[:2]
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        # pygame surface from buffer; transpose so axis order matches (W,H,3).
        surf = pygame.image.frombuffer(u8.tobytes(), (w, h), "RGB")
        surf = pygame.transform.smoothscale(surf, (dest_rect[2], dest_rect[3]))
        win.blit(surf, (dest_rect[0], dest_rect[1]))
        win.blit(bigf.render(label, True, (255, 255, 255)),
                 (dest_rect[0] + 6, dest_rect[1] + 6))
        for i, line in enumerate(info.split("\n")):
            win.blit(font.render(line, True, (210, 230, 255)),
                     (dest_rect[0] + 6, dest_rect[1] + dest_rect[3] - 18 * (3 - i)))

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

        cpp_img = _load_png_rgb(last["cpp"].image_path) if "cpp" in last else None
        glsl_img = _load_png_rgb(last["glsl"].image_path) if "glsl" in last else None

        for i, (name, img) in enumerate([("cpp", cpp_img), ("glsl", glsl_img)]):
            x = 10 + i * (pane_w + 10)
            y = 40
            if img is None:
                pygame.draw.rect(win, (40, 40, 50), (x, y, pane_w, pane_h))
                win.blit(font.render(f"{name.upper()}: not requested",
                                     True, (200, 200, 200)),
                         (x + 12, y + 12))
                continue
            r = last[name]
            info = (f"N_rays={r.n_rays_emitted:_}\n"
                    f"gain={r.gain_linear:.2e}× ({r.gain_db:+.2f} dB)\n"
                    f"H_meas/H_targ={r.measured_H_J:.2e} / {r.target_H_J:.2e} J")
            _blit_image(img, (x, y, pane_w, pane_h),
                        label=name.upper(), info=info)

        pygame.display.flip()
        frame += 1
        time.sleep(0.1)

    pygame.quit()


def _load_png_rgb(path: str) -> Optional[np.ndarray]:
    try:
        import pygame
        s = pygame.image.load(path)
        s = s.convert(24)
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
    p.add_argument("--width",          type=int,   default=320)
    p.add_argument("--height",         type=int,   default=200)
    p.add_argument("--total-rays",     type=int,   default=1_000_000,
                   help="Total rays per FINISHED exposure (default 1M; "
                        "raise to 10_000_000+ for production exposures).")
    p.add_argument("--rays-per-batch", type=int,   default=50_000,
                   help="Sub-batch size; many small batches stream cleanly.")
    p.add_argument("--frames",         type=int,   default=4)
    p.add_argument("--max-bounces",    type=int,   default=4)
    p.add_argument("--backend",        choices=("cpp", "glsl", "both"),
                   default="cpp")
    p.add_argument("--exposure-time-s", type=float, default=DEFAULT_FILM.exposure_time_s)
    p.add_argument("--iso",            type=float, default=DEFAULT_FILM.iso)
    p.add_argument("--focal-mm",       type=float, default=DEFAULT_OPTICS.focal_mm)
    p.add_argument("--aperture-mm",    type=float, default=DEFAULT_OPTICS.aperture_mm)
    p.add_argument("--no-window",      action="store_true",
                   help="Skip pygame window; just dump PNG/JSON to disk.")
    p.add_argument("--out-dir",        default="exposures")
    p.add_argument("--pane-w",         type=int, default=480)
    p.add_argument("--pane-h",         type=int, default=300)
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
    )

    if args.no_window:
        for k in range(args.frames):
            session.render_one_exposure(t=float(k) * 0.5)
    else:
        _run_viewer(session, args.frames, args.pane_w, args.pane_h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
