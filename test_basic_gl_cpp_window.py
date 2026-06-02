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
import os
import time

import numpy as np
import pygame
from OpenGL.GL import *

import _spectral_kernels as sk
from base_gl_renderer import BaseGLRenderer
from camera_exposure_budget import (
    CameraOptics,
    FilmExposure,
    lambertian_emitter_radiance,
    plan_ray_budget,
    summarize_plan,
)
from emissive_ray_packer import pack_emissive_area_rays, summarize_packed_rays
from material_db import MaterialDatabase
from shader_calibration_profiles import load_shader_calibration_profile, save_shader_calibration_gains
from platonic_solids import triangular_prism
from spherical_mesh import uv_sphere, equirect_texture

try:
    import yaml as _yaml
except ImportError as _yaml_exc:  # pragma: no cover - runtime environment check
    _yaml = None
    _yaml_import_error = _yaml_exc
else:
    _yaml_import_error = None

# ── Window / texture constants ────────────────────────────────────────────────
WIN_W, WIN_H = 1280, 720
PANE_W       = WIN_W // 2
TEX_W, TEX_H = 256, 128

# ── Orbital mechanics constants ───────────────────────────────────────────────
GOLDEN_ANGLE = 2.3999632297286533   # ≈ 137.5° — distributes orbital planes
SCENE_CENTER = np.array([0.0, 0.0, -3.05], np.float32)
TUNGSTEN_CAMERA_POS = SCENE_CENTER.copy()
TUNGSTEN_BULB_OFFSET = np.array([0.0, 0.22, 0.0], np.float32)
R_MID = 1.90    # orbit semi-major-axis midpoint
R_AMP = 0.80    # eccentricity amplitude  → r ∈ [1.10, 2.70]
ORBIT_SPEED = 0.45  # radians / second (all orbiters same period)

# ── Sphere meshes ─────────────────────────────────────────────────────────────
SPHERE = uv_sphere(56, 28)
SMALL  = uv_sphere(24, 12)
CALIB_STEP_MATERIALS = tuple(f"calib_step_{i}" for i in range(8))

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
    "ruby_emit",     # glowing red glass
    "emerald_emit",  # glowing green glass
    "sapphire_emit", # glowing blue glass
    "amber_lobe",    # warm directional glow
    "emit_pattern",  # textured glow
    "acrylic",       # clear dielectric
    "frosted_glass", # clear/frosted scatter
    "color_mosaic",  # color UV texture
    "chrome",        # pure metallic
    "copper",        # warm metallic
    "gold",          # metallic + enamel
    "jade_sss",      # translucent SSS
]
N_ORBIT = len(ORBIT_MATS)
C_RASTER_MAX_LIGHTS = 100
_CONFIGS_MATERIALS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "materials"
)


def _load_material_yaml_dict(name: str) -> dict:
    if _yaml is None:
        raise RuntimeError(f"PyYAML is required to load {name}.yaml: {_yaml_import_error}")
    path = os.path.join(_CONFIGS_MATERIALS, f"{name}.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        d = _yaml.safe_load(fh) or {}
    for k in ("name", "description", "notes", "gameplay"):
        d.pop(k, None)
    return d


def _yaml_to_material_dict(d: dict) -> dict:
    """Convert canonical material YAML payload to MaterialDatabase dict fields.

    Keeps nested `pbr` authoring values intact so emissive and albedo terms
    survive tensor bake for both GL and C calibration paths.
    """
    out: dict = {}
    pbr = d.get("pbr", {}) if isinstance(d.get("pbr", {}), dict) else {}

    albedo = pbr.get("albedo", pbr.get("albedo_rgb", d.get("albedo_rgb", [0.5, 0.5, 0.5])))
    out["albedo_rgb"] = [float(albedo[0]), float(albedo[1]), float(albedo[2])]

    if "roughness" in pbr:
        out["roughness"] = float(pbr["roughness"])
    elif "roughness" in d:
        out["roughness"] = float(d["roughness"])
    elif "smoothness" in d:
        out["roughness"] = float(max(0.0, min(1.0, 1.0 - float(d["smoothness"])) ))

    if "metallic" in pbr:
        out["metallic"] = float(pbr["metallic"])
    elif "metallic" in d:
        out["metallic"] = float(d["metallic"])

    if "transmission" in pbr:
        out["transmission"] = float(pbr["transmission"])
    elif "transmission" in d:
        out["transmission"] = float(d["transmission"])

    out["ior"] = float(pbr.get("ior", d.get("ior", 1.5)))
    out["opacity"] = float(pbr.get("opacity", d.get("opacity", 1.0)))

    emis = pbr.get("emission_rgb", d.get("emission_rgb", [0.0, 0.0, 0.0]))
    out["emission_rgb"] = [float(emis[0]), float(emis[1]), float(emis[2])]

    if isinstance(d.get("texture_stack"), dict):
        out["texture_stack"] = dict(d["texture_stack"])
    if isinstance(d.get("enamel"), dict):
        out["enamel"] = dict(d["enamel"])
    if isinstance(d.get("spectral_bands"), list):
        out["spectral_bands"] = list(d["spectral_bands"])

    for key in (
        "reflectivity", "diffusion", "absorption",
        "emit_profile_name", "remit_profile_name", "color_profile_name",
        "reactive_shift_hz",
    ):
        if key in d:
            out[key] = d[key]

    return out


def _register_yaml_material(db: MaterialDatabase, name: str) -> int:
    return db.register(name, _yaml_to_material_dict(_load_material_yaml_dict(name)))


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

        # ── Emitters — loaded from configs/materials/ ──────────────────────
        # (ruby_emit, emerald_emit, sapphire_emit, amber_lobe are registered
        #  below via _register_yaml_material after the inline mats loop)
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
            # Warm broadband white: single lobe centred near yellow-green (~560 nm)
            "bands": [
                {"center_hz": 5.353e14, "bandwidth_hz": {"type": "q_factor", "q": 5.0},
                 "reflectance": 0.12, "diffuse_frac": 0.50, "ior_real": 1.5, "emission": 0.82},
            ],
        },

        # ── Calibration materials (explicit BW + RGB chain checks) ─────
        "calib_black": {
            "albedo_rgb": [0.0, 0.0, 0.0], "roughness": 1.0, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "calib_white": {
            "albedo_rgb": [1.0, 1.0, 1.0], "roughness": 0.35, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "calib_red_emit": {
            "albedo_rgb": [0.2, 0.0, 0.0], "roughness": 0.2, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [2.5, 0.0, 0.0],
            "bands": [
                {"center_hz": 4.759e14, "bandwidth_hz": {"type": "q_factor", "q": 15.0},
                 "reflectance": 0.04, "diffuse_frac": 0.20, "ior_real": 1.5, "emission": 2.5},
            ],
        },
        "calib_green_emit": {
            "albedo_rgb": [0.0, 0.2, 0.0], "roughness": 0.2, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 2.5, 0.0],
            "bands": [
                {"center_hz": 5.657e14, "bandwidth_hz": {"type": "q_factor", "q": 15.0},
                 "reflectance": 0.04, "diffuse_frac": 0.20, "ior_real": 1.5, "emission": 2.5},
            ],
        },
        "calib_blue_emit": {
            "albedo_rgb": [0.0, 0.0, 0.2], "roughness": 0.2, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 2.5],
            "bands": [
                {"center_hz": 6.662e14, "bandwidth_hz": {"type": "q_factor", "q": 15.0},
                 "reflectance": 0.04, "diffuse_frac": 0.20, "ior_real": 1.5, "emission": 2.5},
            ],
        },

        # ── Calibration step wedge ───────────────────────────────────────
        "calib_step_0": {
            "albedo_rgb": [0.06, 0.06, 0.06], "roughness": 1.0, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "calib_step_1": {
            "albedo_rgb": [0.16, 0.16, 0.16], "roughness": 1.0, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "calib_step_2": {
            "albedo_rgb": [0.28, 0.28, 0.28], "roughness": 1.0, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "calib_step_3": {
            "albedo_rgb": [0.40, 0.40, 0.40], "roughness": 1.0, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "calib_step_4": {
            "albedo_rgb": [0.52, 0.52, 0.52], "roughness": 0.95, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "calib_step_5": {
            "albedo_rgb": [0.66, 0.66, 0.66], "roughness": 0.75, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "calib_step_6": {
            "albedo_rgb": [0.80, 0.80, 0.80], "roughness": 0.50, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },
        "calib_step_7": {
            "albedo_rgb": [0.94, 0.94, 0.94], "roughness": 0.30, "metallic": 0.0,
            "ior": 1.5, "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        },

        # amber_lobe: loaded from configs/materials/amber_lobe.yaml below

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
    # Physical emitters — spectral emission bands live in their YAML files.
    for _yaml_name in ("ruby_emit", "emerald_emit", "sapphire_emit", "amber_lobe"):
        idx[_yaml_name] = _register_yaml_material(db, _yaml_name)
    idx["cavity_receiver"] = _register_yaml_material(db, "painted_plaster_wall")
    idx["tungsten_bulb_emit"] = _register_yaml_material(db, "tungsten_filament")
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
    # Paraboloid: center closest to camera, rim curves away → concave face toward camera.
    # Using (xs²+zs²) so all edges recede equally (bowl shape, not saddle).
    ys = -0.75 + (xs * xs + zs * zs) / 5.2
    # Remove the -0.75 Y-offset so the dish is centered on the middle orb.
    grid = np.stack([xs, zs, -ys - 6.65], axis=-1).astype(np.float32)
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


def quad_patch(x0: float, y0: float, x1: float, y1: float, z: float) -> np.ndarray:
    return np.array([
        [[x0, y0, z], [x1, y0, z], [x1, y1, z]],
        [[x0, y0, z], [x1, y1, z], [x0, y1, z]],
    ], dtype=np.float32)


def mesh_tris(mesh, *, center: tuple[float, float, float], scale: float = 1.0,
              rot_z_deg: float = 0.0, rot_y_deg: float = 0.0) -> np.ndarray:
    verts = np.asarray(mesh.verts, np.float32) * float(scale)
    if rot_z_deg:
        a = math.radians(rot_z_deg)
        cz, sz = math.cos(a), math.sin(a)
        rot = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], np.float32)
        verts = verts @ rot.T
    if rot_y_deg:
        a = math.radians(rot_y_deg)
        cy, sy = math.cos(a), math.sin(a)
        rot = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], np.float32)
        verts = verts @ rot.T
    verts += np.asarray(center, np.float32)[None, :]
    return np.ascontiguousarray(verts[np.asarray(mesh.tris, np.int32)], np.float32)


def scene_for_phase(idx: dict[str, int], t: float, scene_mode: str = "orbiters"):
    chunks_v     = []
    tri_mids     = []
    draw_mids    = []
    draw_gids    = []
    group_ids    = []
    group_mids   = []
    group_offsets = []
    group_counts  = []

    def add_object(verts8: np.ndarray, mat_id: int, group_id: int) -> None:
        tri_offset = sum(group_counts)
        tri_count  = verts8.shape[0] // 3
        chunks_v.append(verts8)
        draw_mids.append(np.full((verts8.shape[0],), mat_id, np.int32))
        draw_gids.append(np.full((verts8.shape[0],), group_id, np.int32))
        tri_mids.append(np.full((tri_count,), mat_id, np.int32))
        group_ids.append(group_id)
        group_mids.append(mat_id)
        group_offsets.append(tri_offset)
        group_counts.append(tri_count)

    def add_quad(x0: float, y0: float, x1: float, y1: float, z: float,
                 mat_id: int, group_id: int) -> None:
        tris = quad_patch(x0, y0, x1, y1, z)
        verts8, _ = flat_from_tris(tris, np.full((tris.shape[0],), mat_id, np.int32))
        add_object(verts8, mat_id, group_id)

    if scene_mode == "tungsten-cavity":
        cavity_center = TUNGSTEN_CAMERA_POS
        bulb_center = TUNGSTEN_CAMERA_POS + TUNGSTEN_BULB_OFFSET

        # Existing MaterialDB material: painted_plaster_wall receiving shell,
        # with normals flipped inward so the interior is directly lit.
        cavity = SPHERE.flat_vertices(
            center=tuple(cavity_center.tolist()), radius=5.0, include_uv=True
        )
        cavity[:, 3:6] *= -1.0
        # Keep winding consistent with inward normals (true inside-facing shell).
        cavity = cavity.reshape(-1, 3, 8)
        cavity[:, [1, 2], :] = cavity[:, [2, 1], :]
        cavity = cavity.reshape(-1, 8)
        add_object(cavity, idx["cavity_receiver"], 2)

        # Existing MaterialDB material: tungsten_filament emitter at center.
        bulb = SMALL.flat_vertices(
            center=tuple(bulb_center.tolist()), radius=0.11, include_uv=True
        )
        add_object(bulb, idx["tungsten_bulb_emit"], 10)
    elif scene_mode in ("calib-rgb-diagram", "calib-bw-rgb"):
        # Procedural calibration scene with unambiguous black/white and
        # primary emissive anchors for end-to-end sensor->RGB validation.
        st, _ = saddle_mesh(idx["calib_black"])
        sv, _ = flat_from_tris(st, np.full((st.shape[0],), idx["calib_black"], np.int32))
        add_object(sv, idx["calib_black"], 1)

        white_ref = SPHERE.flat_vertices(
            center=(0.0, -0.10, -3.05), radius=0.60, include_uv=True
        )
        add_object(white_ref, idx["calib_white"], 2)

        red_emit = SMALL.flat_vertices(
            center=(-1.20, 0.35, -2.75), radius=0.22, include_uv=True
        )
        green_emit = SMALL.flat_vertices(
            center=(0.0, 0.45, -2.55), radius=0.22, include_uv=True
        )
        blue_emit = SMALL.flat_vertices(
            center=(1.20, 0.35, -2.75), radius=0.22, include_uv=True
        )
        add_object(red_emit, idx["calib_red_emit"], 20)
        add_object(green_emit, idx["calib_green_emit"], 21)
        add_object(blue_emit, idx["calib_blue_emit"], 22)
    elif scene_mode == "calib-grid":
        # Checkerboard / marker target for geometry, sampling, and reprojection.
        nx = 12
        ny = 8
        x0, x1 = -2.2, 2.2
        y0, y1 = -1.6, 1.6
        z = -3.00
        dx = (x1 - x0) / nx
        dy = (y1 - y0) / ny
        for iy in range(ny):
            for ix in range(nx):
                mx0 = x0 + ix * dx
                my0 = y0 + iy * dy
                mat_name = "calib_black" if (ix + iy) % 2 == 0 else "calib_white"
                add_quad(mx0, my0, mx0 + dx, my0 + dy, z, idx[mat_name], 30 + iy * nx + ix)
        for j, (cx, cy) in enumerate(((-1.9, -1.2), (1.9, -1.2), (1.9, 1.2), (-1.9, 1.2))):
            marker = SMALL.flat_vertices(center=(cx, cy, -2.92), radius=0.10, include_uv=True)
            add_object(marker, idx["calib_white" if j % 2 == 0 else "calib_black"], 200 + j)
        add_quad(-1.85, -1.85, -1.15, -1.72, -2.88, idx["calib_red_emit"], 210)
        add_quad(-0.35, -1.85, 0.35, -1.72, -2.88, idx["calib_green_emit"], 211)
        add_quad(1.15, -1.85, 1.85, -1.72, -2.88, idx["calib_blue_emit"], 212)
    elif scene_mode == "calib-step-wedge":
        # Radiometric step wedge: increasing reflectance across the frame.
        x0, x1 = -2.35, 2.35
        y0, y1 = -0.9, 0.95
        dx = (x1 - x0) / float(len(CALIB_STEP_MATERIALS))
        for i, mat_name in enumerate(CALIB_STEP_MATERIALS):
            sx0 = x0 + i * dx
            sx1 = sx0 + dx
            add_quad(sx0, y0, sx1, y1, -3.02, idx[mat_name], 50 + i)
        add_quad(-2.55, -1.25, 2.55, -1.05, -2.96, idx["calib_black"], 70)
        add_quad(-2.55, 1.05, 2.55, 1.25, -2.96, idx["calib_white"], 71)
        add_quad(-2.55, -0.05, 2.55, 0.05, -2.94, idx["calib_black"], 72)
    elif scene_mode == "calib-prism-backplate":
        # Prism comparator: a chromatic backplate with a triangular prism in front.
        back_z = -4.05
        add_quad(-2.8, -1.3, 2.8, 1.3, back_z, idx["calib_white"], 80)
        for i, (mat_name, x0) in enumerate((("calib_red_emit", -2.5), ("calib_green_emit", -0.8), ("calib_blue_emit", 0.9))):
            add_quad(x0, -0.28, x0 + 1.25, 0.28, back_z + 0.01, idx[mat_name], 90 + i)
        prism = triangular_prism()
        prism_tris = mesh_tris(prism, center=(0.0, -0.12, -3.08), scale=0.60, rot_y_deg=28.0)
        prism_verts8, _ = flat_from_tris(prism_tris, np.full((prism_tris.shape[0],), idx["acrylic"], np.int32))
        add_object(prism_verts8, idx["acrylic"], 95)
        add_quad(-0.22, -1.05, 0.22, 1.05, -3.96, idx["calib_black"], 96)
    else:
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
    if scene_mode == "tungsten-cavity":
        # Render in explicit view-space so the camera sits at cavity center.
        verts8[:, 0:3] -= TUNGSTEN_CAMERA_POS[None, :]
    mat_per_v   = np.ascontiguousarray(np.concatenate(draw_mids,  axis=0), np.int32)
    gid_per_v   = np.ascontiguousarray(np.concatenate(draw_gids,  axis=0), np.int32)
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
    # Group 2 is always a receiver shell and should not be emitted as a light.
    if len(groups[1]) > 0 and 2 in set(groups[0].tolist()):
        recv_i = int(np.where(groups[0] == 2)[0][0])
        groups[1][recv_i] = idx["stage_slate"]
    return verts8, mat_per_v, gid_per_v, mat_per_tri, groups


# ── GL helpers ────────────────────────────────────────────────────────────────

def make_vao(verts8: np.ndarray, mat_per_vertex: np.ndarray,
             group_per_vertex: np.ndarray) -> tuple[int, int, int, int]:
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    mbo = glGenBuffers(1)
    gbo = glGenBuffers(1)
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
    glBindBuffer(GL_ARRAY_BUFFER, gbo)
    glBufferData(GL_ARRAY_BUFFER, group_per_vertex.nbytes, group_per_vertex, GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(4)
    glVertexAttribIPointer(4, 1, GL_INT, 4, ctypes.c_void_p(0))
    glBindVertexArray(0)
    return int(vao), int(vbo), int(mbo), int(gbo)


def update_vao(vbo: int, mbo: int, gbo: int,
               verts8: np.ndarray, mat_per_v: np.ndarray, gid_per_v: np.ndarray) -> None:
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts8.nbytes, verts8, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, mbo)
    glBufferData(GL_ARRAY_BUFFER, mat_per_v.nbytes, mat_per_v, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, gbo)
    glBufferData(GL_ARRAY_BUFFER, gid_per_v.nbytes, gid_per_v, GL_DYNAMIC_DRAW)


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
        cands.append((total, pos.astype(np.float32), rgb, int(gid)))
    cands.sort(key=lambda x: x[0], reverse=True)
    cands = cands[:max_lights]
    if not cands:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0,),   np.float32),
            np.zeros((0,),   np.int32),
        )
    return (
        np.ascontiguousarray([c[1] for c in cands], np.float32),
        np.ascontiguousarray([c[2] for c in cands], np.float32),
        np.ascontiguousarray([c[0] for c in cands], np.float32),
        np.ascontiguousarray([c[3] for c in cands], np.int32),
    )


def _default_pearl_roi_mask(width: int, height: int) -> np.ndarray:
    """Circular shell ROI for the tungsten cavity receiver on one pane.

    The mask is an annulus centered in the pane. Inner radius removes the bulb,
    outer radius removes the background around the shell silhouette.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = r / float(min(width, height))
    return np.logical_and(r_norm >= 0.085, r_norm <= 0.205)


def _rgb_luma_stats(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Return mean/std/p95 luma over masked pixels in display space."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return 0.0, 0.0, 0.0
    if mask.shape != rgb.shape[:2]:
        return 0.0, 0.0, 0.0
    sel = np.asarray(mask, dtype=bool)
    if not np.any(sel):
        return 0.0, 0.0, 0.0
    vals = (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )[sel]
    vals = np.asarray(vals, np.float32)
    return float(vals.mean()), float(vals.std()), float(np.percentile(vals, 95.0))


def _rgb_mean(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return masked mean RGB (float32, shape (3,)) in display space."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return np.zeros((3,), np.float32)
    if mask.shape != rgb.shape[:2]:
        return np.zeros((3,), np.float32)
    sel = np.asarray(mask, dtype=bool)
    if not np.any(sel):
        return np.zeros((3,), np.float32)
    vals = np.asarray(rgb[sel, :3], np.float32)
    return np.asarray(vals.mean(axis=0), np.float32)


def _read_gl_pane_rgb() -> np.ndarray:
    """Read the right GL pane as float32 RGB in display orientation."""
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    raw = glReadPixels(PANE_W, 0, PANE_W, WIN_H, GL_RGBA, GL_FLOAT)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(WIN_H, PANE_W, 4)
    return np.flipud(arr)[..., :3].copy()


def _draw_overlay_text_rgba(text: str, x: int, y: int, color: tuple[int, int, int, int] = (255, 48, 48, 255)) -> None:
    """Blit a small RGBA text sprite into the OpenGL backbuffer at top-left coords."""
    font = pygame.font.SysFont("Consolas", 20, bold=True)
    txt = font.render(text, True, color[:3])
    w, h = txt.get_size()
    if w <= 0 or h <= 0:
        return

    # Build a fully transparent RGBA surface and place glyphs onto it so
    # glDrawPixels receives valid alpha instead of an opaque rectangle.
    surf = pygame.Surface((w, h), flags=pygame.SRCALPHA, depth=32)
    surf.fill((0, 0, 0, 0))
    surf.blit(txt, (0, 0))
    rgba = pygame.image.tostring(surf, "RGBA", True)

    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glWindowPos2i(int(x), int(max(0, WIN_H - y - h)))
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba)


def _pack_light_stats(pos: np.ndarray, col: np.ndarray, inten: np.ndarray, gids: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    out: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    n = min(pos.shape[0], col.shape[0], inten.shape[0], gids.shape[0])
    for i in range(n):
        gid = int(gids[i])
        out[gid] = (
            np.asarray(pos[i], np.float32),
            np.asarray(col[i], np.float32),
            float(inten[i]),
        )
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0,
                    help="Render N frames then exit (0 = interactive)")
    ap.add_argument("--hold-final-frame", action="store_true",
                    help="After a --frames-limited run, keep the final frame on screen until Esc/close.")
    ap.add_argument("--gl-calibration", type=float, default=1.0,
                    help="Global GL light calibration factor.")
    ap.add_argument("--c-calibration", type=float, default=1.0,
                    help="Global C rasterizer light calibration factor.")
    ap.add_argument("--scene", type=str, default="orbiters",
                    choices=[
                        "orbiters", "tungsten-cavity", "calib-rgb-diagram",
                        "calib-bw-rgb", "calib-grid", "calib-step-wedge",
                        "calib-prism-backplate",
                    ],
                    help=("Scene preset: default orbiters, tungsten bulb inside inward receiver sphere, "
                          "or one of the calibration plates/backplate scenes."))
    ap.add_argument("--calibration-file", type=str,
                    default=os.path.join("configs", "shader_calibration_profiles.json"),
                    help="JSON file containing shared GL/C shader calibration profiles.")
    ap.add_argument("--calibration-profile", type=str,
                    default="tungsten_white_pearl",
                    help="Calibration profile name from --calibration-file.")
    ap.add_argument("--temperature-k", type=float, default=None,
                    help="Optional blackbody temperature override for white balance gain generation.")
    ap.add_argument("--print-light-stats", action="store_true",
                    help="Print one-frame C vs GL derived light stats and deltas.")
    ap.add_argument("--print-pearl-metrics", action="store_true",
                    help="Print one-frame pearl receiver brightness metrics for both panes.")
    ap.add_argument("--calib-target-luma", type=float, default=0.84,
                    help="Target display-space luma for pearl during adaptive calibration.")
    ap.add_argument("--calib-luma-tol", type=float, default=0.02,
                    help="Absolute luma tolerance for convergence.")
    ap.add_argument("--calib-white-tol", type=float, default=0.035,
                    help="Max channel spread tolerance (max(rgb)-min(rgb)) for white convergence.")
    ap.add_argument("--calib-integral-rate", type=float, default=0.075,
                    help="Integral gain rate in 1/s for adaptive calibration search.")
    ap.add_argument("--calib-max-search", type=float, default=20.0,
                    help="Clamp for adaptive scalar search gain.")
    ap.add_argument("--calib-hold-frames", type=int, default=24,
                    help="Consecutive converged frames required before switching to orbiters.")
    ap.add_argument("--calib-stall-eps", type=float, default=5.0e-4,
                    help="Absolute per-frame luma-error delta below which a path is considered stalled.")
    ap.add_argument("--calib-stall-frames", type=int, default=45,
                    help="Consecutive stalled frames before freezing that path's gain search.")
    ap.add_argument("--print-emissive-rays", type=int, default=0,
                    help="On the first frame, distribute N random ray sources across emissive\n"
                         "triangle areas (using the same verts8+groups+MaterialDatabase the\n"
                         "shaders consume) and print calibration stats. 0 = disabled.")
    ap.add_argument("--emissive-rays-seed", type=int, default=0,
                    help="RNG seed for --print-emissive-rays.")
    # ── Camera + film exposure budget (drives the calibration ray count) ──
    ap.add_argument("--cam-focal-mm",      type=float, default=35.0)
    ap.add_argument("--cam-aperture-mm",   type=float, default=25.0,
                    help="Entrance pupil diameter in mm. 0 = use --cam-fstop.")
    ap.add_argument("--cam-fstop",         type=float, default=1.4,
                    help="Fallback f-number when --cam-aperture-mm == 0.")
    ap.add_argument("--cam-pixel-pitch-um", type=float, default=8.4)
    ap.add_argument("--cam-sensor-w-mm",   type=float, default=36.0)
    ap.add_argument("--cam-sensor-h-mm",   type=float, default=24.0)
    ap.add_argument("--cam-lens-tx",       type=float, default=0.95)
    ap.add_argument("--film-iso",          type=float, default=100.0)
    ap.add_argument("--film-exposure-s",   type=float, default=1.0 / 60.0)
    ap.add_argument("--film-qe",           type=float, default=0.5)
    ap.add_argument("--budget-rays-per-pixel", type=float, default=1.0,
                    help="Author-side density: rays per pixel per exposure.")
    ap.add_argument("--budget-spp",        type=float, default=1.0,
                    help="Sample multiplier (matches sensor_spp).")
    ap.add_argument("--budget-batches",    type=int,   default=0,
                    help="Force batch count. 0 = derive from fps * exposure.")
    ap.add_argument("--budget-fps",        type=float, default=60.0,
                    help="Frame rate used when --budget-batches == 0.")
    ap.add_argument("--budget-emitter-area-m2", type=float, default=0.152,
                    help="Effective emitter area for radiance estimate "
                         "(default = 4*pi*0.11**2 ≈ small bulb).")
    ap.add_argument("--budget-capture-eta", type=float, default=1.0,
                    help="Initial estimate of forward-ray capture efficiency.")
    args = ap.parse_args()

    calib = load_shader_calibration_profile(
        args.calibration_profile,
        args.calibration_file,
        temperature_override=args.temperature_k,
    )
    cat_ccm = np.ascontiguousarray(calib.cat_ccm_matrix, np.float32)
    gl_gain = float(args.gl_calibration) * float(calib.gl_intensity_gain)
    c_gain = float(args.c_calibration) * float(calib.c_intensity_gain)
    base_gl_gain = gl_gain
    base_c_gain = c_gain

    print("=" * 70)
    print("Basic shader test - C rasterizer (left) vs OpenGL (right)")
    print(f"  {N_ORBIT} orbiters on distinct 3-D orbital planes, even phase spread")
    print(f"  r in [{R_MID - R_AMP:.2f}, {R_MID + R_AMP:.2f}] - always one near closest approach")
    print()
    print("NOT YET WIRED (see file header for details):")
    for line in MISSING:
        print(f"  * {line[:78]}")
    print("=" * 70)
    print(f"[scene] {args.scene}")
    print(f"[calibration] profile={calib.name} T={calib.temperature_k:.1f}K")
    print(f"[calibration] emitter={calib.emitter_material} receiver={calib.receiver_material}")
    print(f"[calibration] gl_gain={gl_gain:.6g} c_gain={c_gain:.6g} cat_ccm={cat_ccm.tolist()}")

    pygame.init()
    pygame.font.init()
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
    # Start from zero and let the integral search ramp up in tungsten calibration phase.
    gl_r.set_light_calibration(0.0)
    gl_r.set_cat_ccm_matrix(cat_ccm)

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
    c_r.set_light_calibration(0.0)
    c_r.set_cat_ccm_matrix(cat_ccm)

    proj      = perspective(43.0, PANE_W / WIN_H, 0.1, 60.0)
    proj_flat = np.ascontiguousarray(proj.T.reshape(-1), np.float32)
    mv        = np.eye(4, dtype=np.float32).T.reshape(-1)

    prof.begin("scene_build")
    verts8, mat_v, gid_v, mat_tri, groups = scene_for_phase(idx, 0.0, args.scene)
    prof.end("scene_build")

    vao, vbo, mbo, gbo = make_vao(verts8, mat_v, gid_v)
    blit_prog     = make_blit_program()
    c_tex         = int(glGenTextures(1))

    clock = pygame.time.Clock()
    t0    = time.perf_counter()
    frame_n = 0
    post_frame_n = 0
    running = True
    diag_printed = False
    pearl_mask = _default_pearl_roi_mask(PANE_W, WIN_H)
    calib_scene_active = args.scene == "tungsten-cavity"
    scene_mode = "tungsten-cavity" if calib_scene_active else args.scene
    search_gain_gl = 0.0
    search_gain_c = 0.0
    converged_frames = 0
    solved_gain_gl = 1.0
    solved_gain_c = 1.0
    freeze_gl = False
    freeze_c = False
    stall_frames_gl = 0
    stall_frames_c = 0
    prev_luma_err_gl = None
    prev_luma_err_c = None
    calibration_saved = False
    calibration_reason = ""
    t_prev = 0.0

    if not calib_scene_active:
        gl_r.set_light_calibration(solved_gain_gl * base_gl_gain)
        c_r.set_light_calibration(solved_gain_c * base_c_gain)

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        t = time.perf_counter() - t0
        dt = max(1.0 / 240.0, min(0.25, t - t_prev))
        t_prev = t

        # Scene rebuild
        prof.begin("scene_build")
        verts8, mat_v, gid_v, mat_tri, groups = scene_for_phase(idx, t, scene_mode)
        prof.end("scene_build")

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
        update_vao(vbo, mbo, gbo, verts8, mat_v, gid_v)
        gl_r.derive_emissive_area_lights(
            verts8,
            groups=groups,
            min_emitter_group_id=10,
            max_lights=C_RASTER_MAX_LIGHTS,
        )

        # One-shot calibration: distribute N ray sources across emissive
        # triangle areas using the SAME (verts8, groups, db) inputs the
        # shaders read above — no simplified-shader translation step.
        # Couples the ray count to the camera+film exposure equation so
        # Σ ray energy across all batches matches the sensor's expected
        # radiant exposure (no calibration ambiguity in the gain solve).
        if int(args.print_emissive_rays) > 0 and frame_n == 0:
            _optics = CameraOptics(
                focal_mm           = float(args.cam_focal_mm),
                aperture_mm        = float(args.cam_aperture_mm),
                max_aperture_fstop = float(args.cam_fstop),
                pixel_pitch_um     = float(args.cam_pixel_pitch_um),
                sensor_w_mm        = float(args.cam_sensor_w_mm),
                sensor_h_mm        = float(args.cam_sensor_h_mm),
                lens_transmission  = float(args.cam_lens_tx),
            )
            _film = FilmExposure(
                iso                = float(args.film_iso),
                exposure_time_s    = float(args.film_exposure_s),
                quantum_efficiency = float(args.film_qe),
            )
            # Use the tungsten profile total power (W) as the scene-side
            # ground truth.  EmissionProfile carries it; if missing, fall
            # back to a 60 W bulb so the equation system is still defined.
            _emit_power_W = 60.0
            try:
                from material_db import EmissionProfileDatabase as _EPD
                _eptpl = _EPD.instance()._registry.get(
                    "tungsten_filament_2400K")
                if _eptpl is not None:
                    _emit_power_W = float(getattr(_eptpl, "total_power_W", 60.0))
            except Exception:
                pass
            _L = lambertian_emitter_radiance(
                _emit_power_W,
                max(float(args.budget_emitter_area_m2), 1.0e-6),
            )
            _plan = plan_ray_budget(
                _optics, _film,
                scene_radiance_W_sr_m2    = _L,
                rays_per_pixel_per_second = float(args.budget_rays_per_pixel),
                n_batches                 = (int(args.budget_batches)
                                              if args.budget_batches > 0 else None),
                sensor_fps                = float(args.budget_fps),
                sensor_spp                = float(args.budget_spp),
                capture_efficiency        = float(args.budget_capture_eta),
            )
            _budget_summary = summarize_plan(_plan, _optics, _film)
            print("[exposure-budget] " + "  ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in _budget_summary.items()
            ), flush=True)

            # Reconcile the user's --print-emissive-rays count with the
            # exposure-budget plan: the CLI-requested ray count wins for
            # this one-shot smoke print, but Σ energy is rescaled to the
            # plan's target_H_J so the gain search has a known anchor.
            _rays = pack_emissive_area_rays(
                verts8,
                groups,
                db,
                n_rays         = int(args.print_emissive_rays),
                min_emitter_group_id = 10,
                seed           = int(args.emissive_rays_seed),
                total_energy_J = float(_plan.target_H_J),
            )
            _stats = summarize_packed_rays(_rays)
            print("[emissive-rays] " + "  ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in _stats.items()
            ), flush=True)
            # Sanity: directions should sit on the unit sphere within fp32
            # tolerance, and energy should be strictly non-negative.
            assert _stats["n_rays"] == 0 or abs(_stats["dir_norm_mean"] - 1.0) < 1.0e-3, (
                f"emissive ray directions not unit-length: {_stats['dir_norm_mean']}"
            )
            assert _stats["n_rays"] == 0 or _stats["energy_min"] >= 0.0, (
                f"emissive ray energy went negative: {_stats['energy_min']}"
            )
            # Σ energy must equal the plan target (within fp32 round-off)
            # — this is the calibration anchor the gain solve will hit.
            if _stats["n_rays"] > 0:
                _sum_J = float(_stats["energy_total"])
                _ratio = _sum_J / max(float(_plan.target_H_J), 1.0e-30)
                assert abs(_ratio - 1.0) < 1.0e-3, (
                    f"Σ ray energy {_sum_J:.6g} J != target_H_J "
                    f"{_plan.target_H_J:.6g} J (ratio={_ratio:.6f})"
                )
            # NOTE: the actual bidirectional solve is the existing BDPT in
            # demo_pluck_gl.py — forward _GPU_RAY_FIELD_CS over bdpt_sources
            # plus backward _GPU_SENSOR_CS, driven by
            # SensorAccumulator.pump_forward() / .tick() / tick_sensor().
            # The plan above tells that solver how many rays to emit per
            # batch and what Σ energy must equal at shutter close; the
            # gain search then matches the measured sensor integral to
            # _plan.target_H_J.  Do NOT add a parallel solver here.
        glViewport(PANE_W, 0, PANE_W, WIN_H)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)
        gl_r.draw_mesh(vao, verts8.shape[0], proj_flat, mv)
        prof.end("gl_draw")

        # Adaptive calibration phase: integrate toward white pearl in tungsten scene.
        if calib_scene_active:
            gl_rgb = _read_gl_pane_rgb()
            c_rgb = np.ascontiguousarray(rgba_c[..., :3], np.float32)

            pearl_rgb_gl = _rgb_mean(gl_rgb, pearl_mask)
            pearl_rgb_c = _rgb_mean(c_rgb, pearl_mask)

            pearl_luma_gl = float(
                pearl_rgb_gl[0] * 0.2126 + pearl_rgb_gl[1] * 0.7152 + pearl_rgb_gl[2] * 0.0722
            )
            pearl_luma_c = float(
                pearl_rgb_c[0] * 0.2126 + pearl_rgb_c[1] * 0.7152 + pearl_rgb_c[2] * 0.0722
            )
            luma_err_gl = float(args.calib_target_luma) - pearl_luma_gl
            luma_err_c = float(args.calib_target_luma) - pearl_luma_c
            white_spread_gl = float(np.max(pearl_rgb_gl) - np.min(pearl_rgb_gl))
            white_spread_c = float(np.max(pearl_rgb_c) - np.min(pearl_rgb_c))

            # Freeze a path when its luma error stops changing for long enough.
            if not freeze_gl and prev_luma_err_gl is not None:
                if abs(luma_err_gl - prev_luma_err_gl) <= float(args.calib_stall_eps):
                    stall_frames_gl += 1
                else:
                    stall_frames_gl = 0
                if stall_frames_gl >= int(args.calib_stall_frames):
                    freeze_gl = True
                    print(
                        f"[calibration-freeze] path=gl err={luma_err_gl:.6g} "
                        f"stall_frames={stall_frames_gl}",
                        flush=True,
                    )
            if not freeze_c and prev_luma_err_c is not None:
                if abs(luma_err_c - prev_luma_err_c) <= float(args.calib_stall_eps):
                    stall_frames_c += 1
                else:
                    stall_frames_c = 0
                if stall_frames_c >= int(args.calib_stall_frames):
                    freeze_c = True
                    print(
                        f"[calibration-freeze] path=c err={luma_err_c:.6g} "
                        f"stall_frames={stall_frames_c}",
                        flush=True,
                    )
            prev_luma_err_gl = luma_err_gl
            prev_luma_err_c = luma_err_c

            # Slow integral-only adaptation with independent GL/C gains.
            if not freeze_gl:
                search_gain_gl += float(args.calib_integral_rate) * luma_err_gl * dt
            if not freeze_c:
                search_gain_c += float(args.calib_integral_rate) * luma_err_c * dt
            search_gain_gl = float(np.clip(search_gain_gl, 0.0, float(args.calib_max_search)))
            search_gain_c = float(np.clip(search_gain_c, 0.0, float(args.calib_max_search)))

            cur_gl_gain = search_gain_gl * base_gl_gain
            cur_c_gain = search_gain_c * base_c_gain
            gl_r.set_light_calibration(cur_gl_gain)
            c_r.set_light_calibration(cur_c_gain)

            luma_ok = (
                (abs(luma_err_gl) <= float(args.calib_luma_tol) or freeze_gl)
                and (abs(luma_err_c) <= float(args.calib_luma_tol) or freeze_c)
            )
            white_ok = (
                white_spread_gl <= float(args.calib_white_tol)
                and white_spread_c <= float(args.calib_white_tol)
            )
            # Completion policy:
            #   1) usual convergence (luma + white), OR
            #   2) both paths stalled/frozen (no longer changing) with acceptable luma.
            done_by_goal = luma_ok and white_ok
            done_by_stall = luma_ok and freeze_gl and freeze_c
            if done_by_goal or done_by_stall:
                converged_frames += 1
            else:
                converged_frames = 0

            if converged_frames >= int(args.calib_hold_frames):
                solved_gain_gl = search_gain_gl
                solved_gain_c = search_gain_c
                final_gl_gain = solved_gain_gl * base_gl_gain
                final_c_gain = solved_gain_c * base_c_gain
                calibration_reason = "goal" if done_by_goal else "stall"
                if not calibration_saved:
                    save_shader_calibration_gains(
                        args.calibration_profile,
                        args.calibration_file,
                        gl_intensity_gain=final_gl_gain,
                        c_intensity_gain=final_c_gain,
                    )
                    calibration_saved = True
                calib_scene_active = False
                scene_mode = "orbiters"
                post_frame_n = 0
                t0 = time.perf_counter()
                t_prev = 0.0
                diag_printed = False
                print(
                    f"[calibration-converged] reason={calibration_reason} search_gain_gl={solved_gain_gl:.6g} "
                    f"search_gain_c={solved_gain_c:.6g} "
                    f"gl_gain={final_gl_gain:.6g} "
                    f"c_gain={final_c_gain:.6g} "
                    f"gl_luma={pearl_luma_gl:.6g} c_luma={pearl_luma_c:.6g} "
                    f"gl_spread={white_spread_gl:.6g} c_spread={white_spread_c:.6g} "
                    f"saved_profile={args.calibration_profile} file={args.calibration_file}",
                    flush=True,
                )

        else:
            cur_gl_gain = solved_gain_gl * base_gl_gain
            cur_c_gain = solved_gain_c * base_c_gain
            gl_r.set_light_calibration(cur_gl_gain)
            c_r.set_light_calibration(cur_c_gain)

        if (args.print_light_stats or args.print_pearl_metrics) and not diag_printed:
            c_pos = c_col = c_int = c_gid = None
            if hasattr(c_r, "readback_lights"):
                try:
                    c_l = c_r.readback_lights()
                    c_pos = np.ascontiguousarray(c_l["positions"], np.float32)
                    c_col = np.ascontiguousarray(c_l["colors"], np.float32)
                    c_int = np.ascontiguousarray(c_l["intensities"], np.float32)
                    c_gid = np.ascontiguousarray(c_l["group_ids"], np.int32)
                except Exception:
                    c_pos = c_col = c_int = c_gid = None
            if c_pos is None:
                c_pos, c_col, c_int, c_gid = derive_group_emitters(verts8, groups, tensors["pbr"])

            gl_pos = np.ascontiguousarray(gl_r._light_pos, np.float32)
            gl_col = np.ascontiguousarray(gl_r._light_color, np.float32)
            gl_int = np.ascontiguousarray(gl_r._light_intensity, np.float32)
            gl_gid = np.ascontiguousarray(gl_r._light_group_id, np.int32)

            if args.print_light_stats:
                c_map = _pack_light_stats(c_pos, c_col, c_int, c_gid)
                gl_map = _pack_light_stats(gl_pos, gl_col, gl_int, gl_gid)
                gids = sorted(set(c_map.keys()) | set(gl_map.keys()))
                print(f"[light-stats] c_n={len(c_map)} gl_n={len(gl_map)} gids={gids}", flush=True)
                for gid in gids:
                    c_rec = c_map.get(gid)
                    g_rec = gl_map.get(gid)
                    if c_rec is None or g_rec is None:
                        print(f"[light-stats][gid={gid}] present_only_in={'c' if g_rec is None else 'gl'}", flush=True)
                        continue
                    dpos = np.max(np.abs(c_rec[0] - g_rec[0]))
                    dcol = np.max(np.abs(c_rec[1] - g_rec[1]))
                    dint = abs(c_rec[2] - g_rec[2])
                    print(
                        f"[light-stats][gid={gid}] "
                        f"c_pos={c_rec[0].tolist()} gl_pos={g_rec[0].tolist()} "
                        f"c_col={c_rec[1].tolist()} gl_col={g_rec[1].tolist()} "
                        f"c_int={c_rec[2]:.6g} gl_int={g_rec[2]:.6g} "
                        f"dpos_max={dpos:.6g} dcol_max={dcol:.6g} dint={dint:.6g}",
                        flush=True,
                    )

            if args.print_pearl_metrics:
                c_rgb = np.ascontiguousarray(rgba_c[..., :3], np.float32)
                gl_rgb = _read_gl_pane_rgb()
                c_mean, c_std, c_p95 = _rgb_luma_stats(c_rgb, pearl_mask)
                g_mean, g_std, g_p95 = _rgb_luma_stats(gl_rgb, pearl_mask)
                print(
                    f"[pearl-metrics] roi=annulus(r_norm in [0.085, 0.205]) pixels={int(pearl_mask.sum())} "
                    f"c_luma_mean={c_mean:.6g} c_luma_std={c_std:.6g} c_luma_p95={c_p95:.6g} "
                    f"gl_luma_mean={g_mean:.6g} gl_luma_std={g_std:.6g} gl_luma_p95={g_p95:.6g} "
                    f"delta_mean={abs(c_mean - g_mean):.6g}",
                    flush=True,
                )
            diag_printed = True

        if calib_scene_active:
            overlay = (
                f"CAL SEARCH  g_gl={search_gain_gl:.6f} g_c={search_gain_c:.6f} "
                f"gl={search_gain_gl * base_gl_gain:.6f} c={search_gain_c * base_c_gain:.6f} "
                f"f_gl={int(freeze_gl)} f_c={int(freeze_c)} "
                f"hold={converged_frames}/{int(args.calib_hold_frames)}"
            )
        else:
            overlay = (
                f"CAL LOCKED  g_gl={solved_gain_gl:.6f} g_c={solved_gain_c:.6f} "
                f"gl={solved_gain_gl * base_gl_gain:.6f} c={solved_gain_c * base_c_gain:.6f} "
                f"scene=orbiters saved={int(calibration_saved)} reason={calibration_reason or 'n/a'}"
            )
        _draw_overlay_text_rgba(overlay, 14, 12, (255, 56, 56, 255))

        pygame.display.flip()
        prof.tick()
        frame_n += 1
        if calib_scene_active:
            # During search, ignore --frames and stop only by convergence or user exit.
            pass
        else:
            post_frame_n += 1
            if args.frames > 0 and post_frame_n >= args.frames:
                running = False
        clock.tick(30)

    if args.hold_final_frame and args.frames > 0 and frame_n >= args.frames:
        hold = True
        while hold:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    hold = False
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    hold = False
            clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
