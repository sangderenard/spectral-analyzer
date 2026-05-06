"""Side-by-side pygame smoke test for the basic C and OpenGL material paths.

Left viewport:  _spectral_kernels.BaseRasterizer  (CPU tile rasterizer)
Right viewport: csrc/shaders/base_material.* through BaseGLRenderer (GPU)

Scene: central UV sphere as a texture/translucence test target, a saddle stage,
and 12 orbiters on distinct 3-D orbital planes (different inclination + RAAN,
elliptical, evenly-phased so there is always one near closest approach and one
near apoapsis).  Each orbiter demonstrates a different shader feature.

Physics profiles wired in both shaders:
  0 = standard dielectric / conductor  (default)
  1 = emissive lobe  (NdotV^direct_lobe_power forward cone)
  2 = translucent SSS  (thickness_mm drives wrap lighting, 4x scale)
  3 = frosted scatter  (depth_uv.A softens specular, adds diffuse halo)

Missing / not yet wired (printed at startup):
  - remit_uv sampling: upload plumbed, neither shader reads it
  - smooth saddle normals: flat per-triangle; uv_sphere has smooth normals
  - depth buffer readback: br_readback_depth() not implemented
  - spatial anti-aliasing: no MSAA/SSAA in either path
  - laser / colinear light axis: direct_lobe_power shapes NdotV, not a
    per-light source_axis field (SceneParams carries no axis yet)
  - remit temporal FIR: remit_attack/decay exist in stack but the
    frame-delayed convolution engine is not called from the render loop
"""
from __future__ import annotations

import argparse
import ctypes
import math
import time

import numpy as np
import pygame
from OpenGL.GL import *

import _spectral_kernels as sk
from base_gl_renderer import BaseGLRenderer
from material_db import MaterialDatabase
from spherical_mesh import uv_sphere, equirect_texture

# ── Window / texture constants ────────────────────────────────────────────────
WIN_W, WIN_H = 1280, 720
PANE_W       = WIN_W // 2
TEX_W, TEX_H = 256, 128

# ── Orbital mechanics constants ───────────────────────────────────────────────
GOLDEN_ANGLE = 2.3999632297286533   # ≈ 137.5° — distributes orbital planes
SCENE_CENTER = np.array([0.0, 0.0, -3.05], np.float32)
R_MID = 1.90    # orbit semi-major-axis midpoint
R_AMP = 0.80    # eccentricity amplitude  → r ∈ [1.10, 2.70]
ORBIT_SPEED = 0.45  # radians / second (all orbiters same period)

# ── Sphere meshes ─────────────────────────────────────────────────────────────
SPHERE = uv_sphere(56, 28)
SMALL  = uv_sphere(24, 12)

# ── Feature / missing report ──────────────────────────────────────────────────
MISSING = [
    "remit_uv sampling: upload is wired end-to-end but neither shader samples it; "
    "frame-delayed re-emission is a future step",

    "smooth saddle normals: flat per-triangle; needs Laplacian averaging pass; "
    "uv_sphere already carries smooth normals",

    "depth buffer readback: br_readback_depth() not implemented; "
    "the C zbuffer is internal to BaseRasterizerState",

    "spatial anti-aliasing: no MSAA/SSAA in either path",

    "laser / colinear falloff: direct_lobe_power shapes NdotV (view-angle cone), "
    "not a per-light source_axis; SceneParams carries no axis field yet",

    "remit temporal FIR: remit_attack/remit_decay scalars live in the texture "
    "stack but the frame-delayed convolution engine is never called from the "
    "render loop",
]


# ── Lightweight per-frame profiler ────────────────────────────────────────────

class FrameProfiler:
    """Accumulates named stage durations; reports mean every N frames."""

    def __init__(self, report_every: int = 60):
        self._n     = report_every
        self._frame = 0
        self._data: dict[str, list[float]] = {}
        self._order: list[str] = []
        self._t0:   dict[str, float] = {}

    def begin(self, name: str) -> None:
        self._t0[name] = time.perf_counter()
        if name not in self._data:
            self._data[name] = []
            self._order.append(name)

    def end(self, name: str) -> None:
        ms = (time.perf_counter() - self._t0[name]) * 1e3
        self._data[name].append(ms)

    def tick(self) -> None:
        self._frame += 1
        if self._frame % self._n == 0:
            parts = [
                f"{k}={np.mean(self._data[k]):.1f}ms"
                for k in self._order if self._data[k]
            ]
            print(f"[f{self._frame}] " + "  ".join(parts), flush=True)
            for k in self._data:
                self._data[k].clear()


# ── Texture generation ────────────────────────────────────────────────────────

def make_depth_uv_layers() -> np.ndarray:
    """Return (3, TEX_H, TEX_W, 4) uint8.

    Layer 0 — center_texture test pattern (4 longitudinal bands):
        R = depth offset, G = thickness, B = translucence_mask (0.6 base),
        A = scatter_mask (0 = not frosted)

    Layer 1 — jade SSS (smooth latitudinal gradient):
        R = gentle depth undulation, G = thickness (high near equator),
        B = translucence_mask (~0.7), A = 0

    Layer 2 — frosted glass:
        R = 0.5 (neutral depth), G = 0.78 (fairly thick),
        B = 0.6 (translucence_mask), A = smooth noise (frost pattern)
    """
    rng = np.random.default_rng(7)
    u = (np.arange(TEX_W, dtype=np.float32) + 0.5) / TEX_W
    v = (np.arange(TEX_H, dtype=np.float32) + 0.5) / TEX_H
    UU, VV = np.meshgrid(u, v, indexing="xy")   # (H, W)

    # ── Layer 0: test pattern ────────────────────────────────────────────
    noise = rng.random((TEX_H, TEX_W), dtype=np.float32)
    band  = np.floor(np.clip(UU, 0.0, 0.9999) * 4.0).astype(np.int32)

    L0 = np.zeros((TEX_H, TEX_W, 4), np.float32)
    L0[..., 2] = 0.60   # B = translucence_mask base
    L0[..., 3] = 0.0    # A = no scatter

    grain     = 0.35 + 0.85 * noise
    curvature = 0.18 + 0.82 * np.sin(np.pi * np.clip(VV, 0.0, 1.0))
    wiggle    = 0.5 + 0.5 * np.cos(8.0 * np.pi * UU) * np.sin(np.pi * VV)

    m1 = band == 1
    L0[m1, 1] = grain[m1]
    L0[m1, 3] = 0.55 + 0.45 * grain[m1]

    m2 = band == 2
    L0[m2, 1] = curvature[m2]
    L0[m2, 0] = 0.2 + 0.6 * curvature[m2]

    m3 = band == 3
    L0[m3, 1] = 0.25 + 0.75 * wiggle[m3]
    L0[m3, 2] = 0.4 + 0.4 * wiggle[m3]

    # ── Layer 1: jade SSS ─────────────────────────────────────────────────
    L1 = np.zeros((TEX_H, TEX_W, 4), np.float32)
    L1[..., 0] = 0.3 + 0.4 * np.cos(4.0 * np.pi * UU)
    L1[..., 1] = np.sin(np.pi * VV)
    L1[..., 2] = 0.65 + 0.25 * np.cos(2.0 * np.pi * UU) * np.sin(np.pi * VV)
    L1[..., 3] = 0.0

    # ── Layer 2: frosted glass ────────────────────────────────────────────
    rng2 = np.random.default_rng(42)
    raw_noise = rng2.random((TEX_H, TEX_W), dtype=np.float32)
    # 5×5 box-filter to smooth the frost pattern
    from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
    pad = np.pad(raw_noise, 2, mode="wrap")
    smooth = sliding_window_view(pad, (5, 5)).mean(axis=(-2, -1)).astype(np.float32)

    L2 = np.zeros((TEX_H, TEX_W, 4), np.float32)
    L2[..., 0] = 0.5
    L2[..., 1] = 0.78
    L2[..., 2] = 0.6
    L2[..., 3] = np.clip(smooth * 1.2, 0.0, 1.0)

    # ── Layer 3: emissive bulb enclosure (profile 1 use case) ────────────
    # R = depth from surface to interior (polar regions deepest).
    # A = bulb_radius (raw [0,1] value, independent of other parameters).
    # G = B = 0 (unused for profile 1).
    depth_from_surface = 0.18 + 0.82 * (1.0 - np.sin(np.pi * np.clip(VV, 0.0, 1.0)))
    L3 = np.zeros((TEX_H, TEX_W, 4), np.float32)
    L3[..., 0] = np.clip(depth_from_surface, 0.0, 1.0)   # R = depth
    L3[..., 3] = 0.35                                      # A = bulb_radius

    stack = np.stack([L0, L1, L2, L3], axis=0)   # (4, H, W, 4) float32 [0,1]
    return np.clip(stack * 255.0 + 0.5, 0, 255).astype(np.uint8)


def make_emit_uv_layers() -> np.ndarray:
    """Return (1, TEX_H, TEX_W, 4) uint8.

    Layer 0 — radial ring pattern for emit_pattern material:
        R = direct_lobe_gain (hot ring at radius 0.25 + bright center)
        G = diffuse_visible_gain (broad central glow)
        B = saturation_adjust (0.5 = identity)
        A = dim/mask (1.0 = full brightness)
    """
    u = (np.arange(TEX_W, dtype=np.float32) + 0.5) / TEX_W
    v = (np.arange(TEX_H, dtype=np.float32) + 0.5) / TEX_H
    UU, VV = np.meshgrid(u, v, indexing="xy")
    dx, dy = UU - 0.5, VV - 0.5
    r2 = dx * dx + dy * dy

    ring_r = 0.22
    ring   = np.exp(-((np.sqrt(r2) - ring_r) ** 2) / 0.008)
    glow   = np.exp(-r2 / 0.04)
    hot    = np.exp(-r2 / 0.006)

    L = np.zeros((TEX_H, TEX_W, 4), np.float32)
    L[..., 0] = np.clip(ring * 1.5 + hot * 2.0, 0, 1)
    L[..., 1] = np.clip(glow + ring * 0.35, 0, 1)
    L[..., 2] = 0.5
    L[..., 3] = 1.0

    return np.clip(L[None] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def make_color_uv_layers() -> np.ndarray:
    """Return (1, TEX_H, TEX_W, 4) uint8.

    Layer 0 — marble-like color pattern for color_mosaic material:
        RGB = warm veined marble   A = 1.0 (full blend weight)
    """
    u = (np.arange(TEX_W, dtype=np.float32) + 0.5) / TEX_W
    v = (np.arange(TEX_H, dtype=np.float32) + 0.5) / TEX_H
    UU, VV = np.meshgrid(u, v, indexing="xy")

    marble = 0.5 + 0.5 * np.sin(10.0 * UU + 3.5 * np.sin(8.0 * VV))
    R = np.clip(0.65 + 0.30 * marble, 0, 1)
    G = np.clip(0.52 + 0.22 * marble, 0, 1)
    B = np.clip(0.30 + 0.15 * (1.0 - marble), 0, 1)

    L = np.stack([R, G, B, np.ones_like(R)], axis=-1).astype(np.float32)
    return np.clip(L[None] * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ── Material registry ─────────────────────────────────────────────────────────

ORBIT_MATS = [
    "ruby_emit",     # emissive + enamel (thin-film iridescence)
    "emerald_emit",  # emissive green
    "sapphire_emit", # emissive blue
    "chrome",        # pure metallic
    "copper",        # warm metallic
    "gold",          # metallic + enamel
    "acrylic",       # semi-transparent dielectric
    "color_mosaic",  # color UV texture (profile 0)
    "emit_pattern",  # emit UV texture (profile 0)
    "amber_lobe",    # profile 1: emissive forward cone
    "jade_sss",      # profile 2: translucent SSS
    "frosted_glass", # profile 3: frosted scatter
]
N_ORBIT = len(ORBIT_MATS)
C_RASTER_MAX_LIGHTS = 100


def register_materials() -> tuple[MaterialDatabase, dict[str, int]]:
    db = MaterialDatabase()
    mats: dict = {
        # ── Receivers / stage ─────────────────────────────────────────────
        "center_texture": {
            "albedo_rgb": [0.95, 0.82, 0.54], "roughness": 0.35, "metallic": 0.0,
            "ior": 1.46, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
            "texture_stack": {
                "depth_uv_layer": 0.0,
                "depth_scale_mm": 6.0, "thickness_scale_mm": 6.0,
                "model_flags": 2.0,        # profile 2: translucent SSS
                "translucence_gain": 0.9,
            },
        },
        "stage_slate": {
            "albedo_rgb": [0.16, 0.19, 0.21], "roughness": 0.82, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },

        # ── Emitters (profile 0 standard) ─────────────────────────────────
        "ruby_emit": {
            "albedo_rgb": [0.62, 0.04, 0.08], "roughness": 0.18, "metallic": 0.0,
            "ior": 1.76, "opacity": 0.92, "emission_rgb": [2.3, 0.15, 0.08],
            "enamel": {"thickness_nm": 380.0, "ior_real": 1.52,
                       "color_rgb": [1.0, 0.65, 0.65]},
        },
        "emerald_emit": {
            "albedo_rgb": [0.02, 0.48, 0.20], "roughness": 0.22, "metallic": 0.0,
            "ior": 1.58, "opacity": 0.95, "emission_rgb": [0.05, 1.9, 0.62],
        },
        "sapphire_emit": {
            "albedo_rgb": [0.04, 0.16, 0.62], "roughness": 0.20, "metallic": 0.0,
            "ior": 1.76, "opacity": 0.95, "emission_rgb": [0.08, 0.36, 2.2],
        },

        # ── Metallic ──────────────────────────────────────────────────────
        "chrome": {
            "albedo_rgb": [0.85, 0.87, 0.90], "roughness": 0.08, "metallic": 1.0,
            "ior": 2.0, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "copper": {
            "albedo_rgb": [0.95, 0.52, 0.22], "roughness": 0.20, "metallic": 1.0,
            "ior": 1.9, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "gold": {
            "albedo_rgb": [1.0, 0.76, 0.20], "roughness": 0.12, "metallic": 1.0,
            "ior": 1.9, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
            "enamel": {"thickness_nm": 220.0, "ior_real": 1.45,
                       "color_rgb": [1.0, 0.95, 0.6]},
        },

        # ── Transparent ───────────────────────────────────────────────────
        "acrylic": {
            "albedo_rgb": [0.55, 0.85, 1.0], "roughness": 0.12, "metallic": 0.0,
            "transmission": 0.28, "ior": 1.49, "opacity": 0.70,
            "emission_rgb": [0.0, 0.0, 0.0],
        },

        # ── Color UV texture (profile 0) ──────────────────────────────────
        "color_mosaic": {
            "albedo_rgb": [0.8, 0.75, 0.65], "roughness": 0.35, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
            "texture_stack": {
                "color_uv_layer": 0.0,
                "color_blend": 0.88,
            },
        },

        # ── Emit UV texture (profile 0) ───────────────────────────────────
        "emit_pattern": {
            "albedo_rgb": [0.15, 0.15, 0.15], "roughness": 0.5, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [1.0, 0.85, 0.6],
            "texture_stack": {
                "emit_uv_layer": 0.0,
                "emit_gain": 2.5,
            },
        },

        # ── Profile 1: emissive forward cone with bulb darkening ─────────
        # depth_uv layer 3: R=depth (scaled by depth_scale_mm), A=bulb_radius (raw)
        # Lorentzian applied to emission: full at pole center, dark at extremes.
        "amber_lobe": {
            "albedo_rgb": [0.35, 0.20, 0.05], "roughness": 0.25, "metallic": 0.0,
            "ior": 1.55, "opacity": 1.0, "emission_rgb": [2.0, 1.1, 0.25],
            "texture_stack": {
                "model_flags": 1.0,
                "direct_lobe_power": 7.0,
                "emit_gain": 1.0,
                "depth_uv_layer": 3.0,
                "depth_scale_mm": 0.5,
            },
        },

        # ── Profile 2: translucent SSS ────────────────────────────────────
        "jade_sss": {
            "albedo_rgb": [0.22, 0.52, 0.30], "roughness": 0.40, "metallic": 0.0,
            "ior": 1.60, "opacity": 0.88, "emission_rgb": [0.0, 0.0, 0.0],
            "texture_stack": {
                "depth_uv_layer": 1.0,
                "depth_scale_mm": 3.0, "thickness_scale_mm": 4.0,
                "model_flags": 2.0,
                "translucence_gain": 1.4,
            },
        },

        # ── Profile 3: frosted scatter ────────────────────────────────────
        "frosted_glass": {
            "albedo_rgb": [0.78, 0.82, 0.88], "roughness": 0.08, "metallic": 0.0,
            "transmission": 0.30, "ior": 1.52, "opacity": 0.75,
            "emission_rgb": [0.0, 0.0, 0.0],
            "texture_stack": {
                "depth_uv_layer": 2.0,
                "depth_scale_mm": 1.0, "thickness_scale_mm": 2.0,
                "model_flags": 3.0,
                "translucence_gain": 0.5,
            },
        },
    }
    idx = {name: db.register(name, mat) for name, mat in mats.items()}
    return db, idx


# ── Orbital mechanics ─────────────────────────────────────────────────────────

def _orbit_frame(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (e1, e2) orthonormal basis for the k-th orbital plane.

    RAAN (longitude of ascending node) is evenly spread over [0, 2π].
    Inclination is distributed via the golden angle to avoid clustering,
    capped to ±65° so orbiters don't dive through the stage floor.
    """
    raan = 2.0 * math.pi * k / N_ORBIT
    incl = math.sin(GOLDEN_ANGLE * k) * (65.0 * math.pi / 180.0)

    # e1 = ascending-node direction in the equatorial (XZ) plane
    e1 = np.array([math.cos(raan), 0.0, math.sin(raan)], np.float32)

    # Orbit normal: standard Euler rotation result
    n_orb = np.array([
         math.sin(incl) * math.sin(raan),
         math.cos(incl),
        -math.sin(incl) * math.cos(raan),
    ], np.float32)

    # e2 = n_orb × e1 (90° ahead of ascending node in the orbital plane)
    e2 = np.cross(n_orb, e1).astype(np.float32)
    nrm = float(np.linalg.norm(e2))
    if nrm > 1e-8:
        e2 /= nrm
    else:
        e2 = np.array([-math.sin(raan), 0.0, math.cos(raan)], np.float32)
    return e1, e2


def orbit_position(k: int, t: float) -> tuple[float, float, float]:
    """3-D position of orbiter k at time t, centred on SCENE_CENTER.

    True anomaly θ = ORBIT_SPEED*t + 2π*k/N_ORBIT → even phase spread.
    r(θ) = R_MID + R_AMP*cos(θ) → one orbiter always near closest approach.
    """
    e1, e2 = _orbit_frame(k)
    theta = ORBIT_SPEED * t + 2.0 * math.pi * k / N_ORBIT
    r = R_MID + R_AMP * math.cos(theta)
    local = r * (math.cos(theta) * e1 + math.sin(theta) * e2)
    pos = SCENE_CENTER + local
    return float(pos[0]), float(pos[1]), float(pos[2])


# ── Mesh helpers ──────────────────────────────────────────────────────────────

def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    p = np.zeros((4, 4), np.float32)
    p[0, 0] = f / aspect
    p[1, 1] = f
    p[2, 2] = (far + near) / (near - far)
    p[2, 3] = (2.0 * far * near) / (near - far)
    p[3, 2] = -1.0
    return p


def saddle_mesh(mat_id: int, n: int = 28) -> tuple[np.ndarray, np.ndarray]:
    lin = np.linspace(-3.2, 3.2, n, dtype=np.float32)
    xs, zs = np.meshgrid(lin, lin, indexing="xy")
    ys = -0.75 - (xs * xs - zs * zs) / 5.2
    grid = np.stack([xs, ys, zs - 5.9], axis=-1).astype(np.float32)
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            v00, v10 = grid[i, j], grid[i + 1, j]
            v01, v11 = grid[i, j + 1], grid[i + 1, j + 1]
            tris.append([v00, v01, v10])
            tris.append([v10, v01, v11])
    tri = np.asarray(tris, np.float32)
    return tri, np.full((len(tri),), mat_id, np.int32)


def flat_from_tris(tris: np.ndarray, mat_ids: np.ndarray,
                   *, uv: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    pts = tris.reshape(-1, 3)
    e1  = tris[:, 1] - tris[:, 0]
    e2  = tris[:, 2] - tris[:, 0]
    n   = np.cross(e1, e2).astype(np.float32)
    nlen = np.linalg.norm(n, axis=1, keepdims=True)
    n  /= np.where(nlen > 1e-8, nlen, 1.0)
    nrms = np.repeat(n, 3, axis=0)
    if uv is None:
        uv = np.zeros((pts.shape[0], 2), np.float32)
    verts = np.concatenate([pts, nrms, uv.astype(np.float32)], axis=1)
    mids  = np.repeat(mat_ids, 3).astype(np.int32)
    return np.ascontiguousarray(verts, np.float32), np.ascontiguousarray(mids, np.int32)


def scene_for_phase(idx: dict[str, int], t: float):
    chunks_v     = []
    tri_mids     = []
    draw_mids    = []
    group_ids    = []
    group_mids   = []
    group_offsets = []
    group_counts  = []

    def add_object(verts8: np.ndarray, mat_id: int, group_id: int) -> None:
        tri_offset = sum(group_counts)
        tri_count  = verts8.shape[0] // 3
        chunks_v.append(verts8)
        draw_mids.append(np.full((verts8.shape[0],), mat_id, np.int32))
        tri_mids.append(np.full((tri_count,), mat_id, np.int32))
        group_ids.append(group_id)
        group_mids.append(mat_id)
        group_offsets.append(tri_offset)
        group_counts.append(tri_count)

    # Stage
    st, sm = saddle_mesh(idx["stage_slate"])
    sv, _  = flat_from_tris(st, sm)
    add_object(sv, idx["stage_slate"], 1)

    # Central sphere (texture + SSS target)
    cv = SPHERE.flat_vertices(center=tuple(SCENE_CENTER.tolist()), radius=0.72, include_uv=True)
    add_object(cv, idx["center_texture"], 2)

    # Orbiters — each on its own 3-D orbital plane, evenly phased
    orbiter_radii = [0.17, 0.19, 0.16, 0.20, 0.18, 0.21,
                     0.17, 0.19, 0.18, 0.20, 0.17, 0.19]
    for k, name in enumerate(ORBIT_MATS):
        center = orbit_position(k, t)
        radius = orbiter_radii[k]
        ov = SMALL.flat_vertices(center=center, radius=radius, include_uv=True)
        add_object(ov, idx[name], 10 + k)

    verts8      = np.ascontiguousarray(np.concatenate(chunks_v,   axis=0), np.float32)
    mat_per_v   = np.ascontiguousarray(np.concatenate(draw_mids,  axis=0), np.int32)
    mat_per_tri = np.ascontiguousarray(np.concatenate(tri_mids,   axis=0), np.int32)

    n_groups = len(group_ids)
    groups = (
        np.asarray(group_ids,    np.int32),
        np.asarray(group_mids,   np.int32),
        np.asarray(group_offsets, np.int32),
        np.asarray(group_counts,  np.int32),
        np.tile(np.eye(4, dtype=np.float32).T.reshape(1, 16), (n_groups, 1)),
        np.full((n_groups,), 1, np.int32),
    )
    # Central sphere is a receiver, not a light source; override its group mat_id
    if len(groups[1]) > 1:
        groups[1][1] = idx["stage_slate"]
    return verts8, mat_per_v, mat_per_tri, groups


# ── GL helpers ────────────────────────────────────────────────────────────────

def make_vao(verts8: np.ndarray, mat_per_vertex: np.ndarray) -> tuple[int, int, int]:
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    mbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts8.nbytes, verts8, GL_DYNAMIC_DRAW)
    stride = 8 * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(3)
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
    glBindBuffer(GL_ARRAY_BUFFER, mbo)
    glBufferData(GL_ARRAY_BUFFER, mat_per_vertex.nbytes, mat_per_vertex, GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(2)
    glVertexAttribIPointer(2, 1, GL_INT, 4, ctypes.c_void_p(0))
    glBindVertexArray(0)
    return int(vao), int(vbo), int(mbo)


def update_vao(vbo: int, mbo: int, verts8: np.ndarray, mat_per_v: np.ndarray) -> None:
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts8.nbytes, verts8, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, mbo)
    glBufferData(GL_ARRAY_BUFFER, mat_per_v.nbytes, mat_per_v, GL_DYNAMIC_DRAW)


def make_blit_program() -> int:
    vs = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vs, """
    #version 330 core
    const vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));
    out vec2 uv;
    void main(){ vec2 q=p[gl_VertexID]; uv=q*0.5+0.5; gl_Position=vec4(q,0,1); }
    """)
    glCompileShader(vs)
    fs = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fs, """
    #version 330 core
    uniform sampler2D uTex;
    in vec2 uv; out vec4 frag;
    void main(){
        vec3 color = texture(uTex, vec2(uv.x, 1.0-uv.y)).rgb;
        color = max(color, vec3(0.0));
        vec3 s1 = color * 12.92;
        vec3 s2 = 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055;
        vec3 condition = step(vec3(0.0031308), color);
        vec3 srgb = mix(s1, s2, condition);
        frag = vec4(srgb, 1.0);
    }
    """)
    glCompileShader(fs)
    prog = glCreateProgram()
    glAttachShader(prog, vs); glAttachShader(prog, fs); glLinkProgram(prog)
    glDeleteShader(vs); glDeleteShader(fs)
    return int(prog)


def upload_c_texture(tex: int, rgba: np.ndarray) -> None:
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, rgba.shape[1], rgba.shape[0],
                 0, GL_RGBA, GL_FLOAT, np.ascontiguousarray(rgba, dtype=np.float32))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)


def derive_group_emitters(verts8: np.ndarray, groups, pbr: np.ndarray,
                          max_lights: int = C_RASTER_MAX_LIGHTS):
    gids, mids, offs, cnts, _mvs, _dirty = groups
    cands = []
    for gid, mid, off, cnt in zip(gids, mids, offs, cnts):
        if int(gid) < 10:      # stage (1) and central sphere (2) are receivers
            continue
        mat_id = int(mid)
        if mat_id < 0 or mat_id >= len(pbr):
            continue
        rgb = np.asarray(pbr[mat_id, 8:11], np.float32)
        if float(np.linalg.norm(rgb)) < 1e-6:
            continue
        s   = int(off) * 3
        e   = s + int(cnt) * 3
        pts = verts8[s:e, 0:3].reshape(-1, 3, 3)
        if pts.size == 0:
            continue
        e1_  = pts[:, 1] - pts[:, 0]
        e2_  = pts[:, 2] - pts[:, 0]
        area = 0.5 * np.linalg.norm(np.cross(e1_, e2_), axis=1)
        total = float(area.sum())
        if total <= 1e-8:
            continue
        cent = pts.mean(axis=1)
        pos  = (cent * area[:, None]).sum(axis=0) / total
        cands.append((total, pos.astype(np.float32), rgb))
    cands.sort(key=lambda x: x[0], reverse=True)
    cands = cands[:max_lights]
    if not cands:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0,),   np.float32),
        )
    return (
        np.ascontiguousarray([c[1] for c in cands], np.float32),
        np.ascontiguousarray([c[2] for c in cands], np.float32),
        np.ascontiguousarray([c[0] for c in cands], np.float32),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0,
                    help="Render N frames then exit (0 = interactive)")
    args = ap.parse_args()

    print("=" * 70)
    print("Basic shader test - C rasterizer (left) vs OpenGL (right)")
    print(f"  {N_ORBIT} orbiters on distinct 3-D orbital planes, even phase spread")
    print(f"  r in [{R_MID - R_AMP:.2f}, {R_MID + R_AMP:.2f}] - always one near closest approach")
    print()
    print("NOT YET WIRED (see file header for details):")
    for line in MISSING:
        print(f"  * {line[:78]}")
    print("=" * 70)

    pygame.init()
    pygame.display.set_mode((WIN_W, WIN_H), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Basic material: C rasterizer vs OpenGL (profiles 0–3)")

    prof = FrameProfiler(report_every=60)

    # ── Material + texture build ──────────────────────────────────────────
    prof.begin("mat_build")
    db, idx = register_materials()
    tensors  = db.build_tensors()
    prof.end("mat_build")

    prof.begin("tex_build")
    depth_tex = make_depth_uv_layers()   # (3, H, W, 4) uint8
    emit_tex  = make_emit_uv_layers()    # (1, H, W, 4)
    color_tex = make_color_uv_layers()   # (1, H, W, 4)
    prof.end("tex_build")

    print(f"[textures] depth={depth_tex.shape}  emit={emit_tex.shape}  "
          f"color={color_tex.shape}", flush=True)

    # ── Emitter angle metrics on emit UV ─────────────────────────────────
    scrim = np.asarray(
        sk.analyze_emitter_rgba8_layers(
            np.ascontiguousarray(emit_tex, np.uint8), 0.001
        ),
        np.float32,
    )
    print(f"[emitter-angle] emit layer0  flux={scrim[0,0]:.4f}  "
          f"active={scrim[0,1]:.4f}  cone_cos={scrim[0,9]:.4f}", flush=True)

    # ── GL renderer ───────────────────────────────────────────────────────
    gl_r = BaseGLRenderer(db, auto_drain=False)
    gl_r.init_gl()
    gl_r.set_depth_uv_texture_array(depth_tex)
    gl_r.set_emit_uv_texture_array(emit_tex)
    gl_r.set_color_uv_texture_array(color_tex)
    gl_r.set_specular_enabled(True)
    gl_r.set_emission_direct_enabled(True)

    # ── C rasterizer ──────────────────────────────────────────────────────
    c_r = sk.BaseRasterizer(PANE_W, WIN_H, 16)
    c_r.set_pbr_chunk(np.ascontiguousarray(tensors["pbr"],          np.float32))
    c_r.set_phong_chunk(np.ascontiguousarray(tensors["phong_compat"], np.float32))
    c_r.set_enamel_chunk(np.ascontiguousarray(tensors["enamel"],    np.float32))
    c_r.set_texture_stack_chunk(np.ascontiguousarray(tensors["texture_stack"], np.float32))
    c_r.set_depth_uv_texture_array(depth_tex)
    c_r.set_emit_uv_texture_array(emit_tex)
    c_r.set_color_uv_texture_array(color_tex)
    c_r.set_max_lights(C_RASTER_MAX_LIGHTS)
    c_r.set_specular_enabled(True)
    c_r.set_emission_direct_enabled(True)

    proj      = perspective(43.0, PANE_W / WIN_H, 0.1, 60.0)
    proj_flat = np.ascontiguousarray(proj.T.reshape(-1), np.float32)
    mv        = np.eye(4, dtype=np.float32).T.reshape(-1)

    prof.begin("scene_build")
    verts8, mat_v, mat_tri, groups = scene_for_phase(idx, 0.0)
    prof.end("scene_build")

    vao, vbo, mbo = make_vao(verts8, mat_v)
    blit_prog     = make_blit_program()
    c_tex         = int(glGenTextures(1))

    clock = pygame.time.Clock()
    t0    = time.perf_counter()
    frame_n = 0
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        t = time.perf_counter() - t0

        # Scene rebuild
        prof.begin("scene_build")
        verts8, mat_v, mat_tri, groups = scene_for_phase(idx, t)
        prof.end("scene_build")

        lpos, lcol, lint = derive_group_emitters(verts8, groups, tensors["pbr"])

        # C rasterizer
        prof.begin("c_render")
        c_r.clear(0.0, 0.0, 0.0, 1.0)
        c_r.set_groups(*groups)
        c_r.render_textured(verts8, mat_tri, proj_flat)
        rgba_c = np.asarray(c_r.readback_f32_view(), np.float32)
        prof.end("c_render")

        # Upload C output to GL texture
        prof.begin("c_upload")
        upload_c_texture(c_tex, rgba_c)
        prof.end("c_upload")

        glClearColor(0.02, 0.025, 0.03, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Blit C render to left pane
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, PANE_W, WIN_H)
        glUseProgram(blit_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, c_tex)
        glUniform1i(glGetUniformLocation(blit_prog, "uTex"), 0)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glUseProgram(0)

        # GL render to right pane
        prof.begin("gl_draw")
        update_vao(vbo, mbo, verts8, mat_v)
        gl_r.set_point_lights(lpos, lcol, lint)
        glViewport(PANE_W, 0, PANE_W, WIN_H)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)
        gl_r.draw_mesh(vao, verts8.shape[0], proj_flat, mv)
        prof.end("gl_draw")

        pygame.display.flip()
        prof.tick()
        frame_n += 1
        if args.frames > 0 and frame_n >= args.frames:
            running = False
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
