"""demo_pluck_gl.py — 3-D acoustic instrument FDTD visualiser.

Perspective 3-D render of the guitar body with live physics overlays:

  • Guitar body     — extruded side walls (Phong-lit, semi-transparent wood)
                      top plate displaced by live Kirchhoff field
                      back plate (dark wood, opaque)
                      soundhole cut-out with dark disc
  • Pressure volume — volumetric ray march through the 3-D FDTD pressure texture
  • String polylines — live velocity displacing each string path
  • Ray-tracer segs  — geometric reflections coloured by bounce count

Controls
--------
  1–6    Cycle layer: OPAQUE → ALPHA → HIDDEN
          1 body   2 plate   3 pressure   4 strings   5 markers   6 ray-segs
  SPACE  Pause / resume
  R      Restart excitation
  Q      Quit
  Mouse  Left-drag: orbit   |   Wheel: zoom
"""
from __future__ import annotations

import ctypes
import faulthandler
import hashlib
import math
import os
import sys
import collections
import argparse
import atexit
import traceback
import multiprocessing as mp
import threading
import queue
import wave
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ── optional physics ──────────────────────────────────────────────────────────
try:
    from _spectral_kernels import AcousticCoEvolver as _CCoEvolver
    _HAS_PHYSICS = True
except ImportError:
    _CCoEvolver = None
    _HAS_PHYSICS = False

try:
    from sm_plugins.orchestral_resonance import _build_body_scene as _build_body_scene_fn
    _HAS_SCENE = True
except ImportError:
    _HAS_SCENE = False

try:
    from acoustic_fdtd_bridge import (
        voxelise_guitar_body,
        build_acoustic_coevolver_from_scene,
        _extract_guitar_geometry,
        GUITAR_SCALE_LENGTH_M,
    )
    _HAS_BRIDGE = True
except ImportError:
    _HAS_BRIDGE = False

try:
    from ray_tracer_bridge import (
        extract_scene_geometry as _extract_scene_geometry_fn,
        extract_scene_geometry_with_materials as _extract_scene_geometry_materials_fn,
        trace_cavity_scene as _trace_fn,
    )
    _HAS_RAY = True
except ImportError:
    _trace_fn = None
    _extract_scene_geometry_fn = None
    _extract_scene_geometry_materials_fn = None
    _HAS_RAY = False

# ── pygame + OpenGL ───────────────────────────────────────────────────────────
try:
    import pygame
    from pygame.locals import (
        DOUBLEBUF, OPENGL, QUIT, KEYDOWN, MOUSEBUTTONDOWN,
        MOUSEMOTION, MOUSEWHEEL,
        K_SPACE, K_r, K_q, K_p, K_w,
        K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8,
    )
except ImportError:
    print("pygame not available"); sys.exit(1)

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_CULL_FACE,
        GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_DYNAMIC_DRAW, GL_ELEMENT_ARRAY_BUFFER,
        GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER, GL_TRUE,
        GL_LINE_LOOP, GL_LINE_STRIP, GL_LINES, GL_LINEAR,
        GL_NEAREST, GL_ONE,
        GL_ONE_MINUS_SRC_ALPHA, GL_R32F, GL_RED, GL_SRC_ALPHA,
        GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D,
        GL_TEXTURE_3D,
        GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_R, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
        GL_CLAMP_TO_EDGE, GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, GL_TRIANGLES,
        GL_UNSIGNED_INT, GL_UNSIGNED_BYTE,
        GL_RGBA, GL_VERTEX_SHADER,
        glActiveTexture, glAttachShader,
        glBindBuffer, glBindTexture, glBindVertexArray,
        glBlendFunc, glBufferData, glBufferSubData, glClear,
        glClearColor, glCompileShader, glCreateProgram, glCreateShader,
        glDeleteBuffers, glDeleteProgram, glDeleteShader,
        glDeleteTextures, glDeleteVertexArrays, glDepthMask,
        glDisable, glDrawArrays, glDrawElements,
        glEnable, glEnableVertexAttribArray, glGenBuffers, glGenTextures,
        glGenVertexArrays, glGetShaderInfoLog, glGetUniformLocation,
        glHint, glLinkProgram, glGetProgramInfoLog, glShaderSource,
        glTexImage2D, glTexImage3D, glTexParameteri, glTexSubImage3D,
        glUniform1f, glUniform1i, glUniform2f, glUniform3f, glUniform3i, glUniform4f,
        glUniformMatrix4fv, glUseProgram, glVertexAttribPointer,
        glViewport, glLineWidth, glGetTexImage,
        GL_LINE_SMOOTH, GL_LINE_SMOOTH_HINT, GL_NICEST,
    )
    from OpenGL.GL import (
        GL_COMPUTE_SHADER, GL_R32UI, GL_RED_INTEGER, GL_SHADER_STORAGE_BUFFER,
        GL_SHADER_STORAGE_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT,
        GL_BUFFER_UPDATE_BARRIER_BIT, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT,
        GL_COMPILE_STATUS, GL_LINK_STATUS, GL_NO_ERROR,
        glGetShaderiv, glGetProgramiv, glGetError,
        GL_WRITE_ONLY, GL_READ_WRITE, glBindBufferBase, glBindImageTexture, glClearTexImage,
        glDispatchCompute, glMemoryBarrier, glGetBufferSubData,
        GL_TEXTURE_BUFFER, GL_RGBA32F, glTexBuffer,
    )
except ImportError:
    print("PyOpenGL not available"); sys.exit(1)


_RAY_DIAG_LAST: dict = {}


def _ray_diag_update(stage: str, **items) -> None:
    """Keep a compact breadcrumb for hard-to-reproduce ray-path exits."""
    _RAY_DIAG_LAST.clear()
    _RAY_DIAG_LAST.update({"stage": stage})
    _RAY_DIAG_LAST.update(items)


def _dump_ray_diag(prefix: str = "[ray crash]") -> None:
    print(prefix, flush=True)
    if _RAY_DIAG_LAST:
        for key in sorted(_RAY_DIAG_LAST):
            print(f"  {key}: {_RAY_DIAG_LAST[key]}", flush=True)
    try:
        gl_err = glGetError()
        print(f"  gl_error: 0x{int(gl_err):04x}", flush=True)
    except Exception as exc:
        print(f"  gl_error: unavailable ({exc})", flush=True)


def _report_exception(context: str, exc: BaseException) -> None:
    print(f"\n[fatal] {context}: {type(exc).__name__}: {exc}", flush=True)
    _dump_ray_diag("[fatal] last ray/OpenGL context")
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def _install_crash_reporting() -> None:
    try:
        faulthandler.enable(all_threads=True)
    except Exception as exc:
        print(f"[fatal] faulthandler unavailable: {exc}", flush=True)
    if not getattr(_install_crash_reporting, "_registered_exit_dump", False):
        atexit.register(_dump_ray_diag, "[exit] last ray/OpenGL context")
        _install_crash_reporting._registered_exit_dump = True


def _quit_pygame_with_diag(context: str) -> None:
    previous_stage = _RAY_DIAG_LAST.get("stage", "")
    _dump_ray_diag(f"[{context}] before pygame.quit")
    _ray_diag_update(
        "pygame:quit:start",
        context=context,
        previous_stage=previous_stage,
    )
    pygame.quit()
    print(f"[{context}] pygame.quit completed", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WIN_W, WIN_H  = 1400, 900
SAMPLE_RATE   = 44100
BLOCK_SAMPLES = 512
MAX_FRAMES    = 240

LAYER_OPAQUE, LAYER_ALPHA, LAYER_HIDDEN = 0, 1, 2
N_LAYERS    = 8
LAYER_KEYS  = [K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8]
LAYER_NAMES = ['body', 'plate', 'pressure', 'strings', 'markers', 'ray-segs', 'stage', 'illum']

DX             = 0.006   # 6 mm cells — tractable with the generous domain below
N_PML          = 28      # 168 mm PML on each face — effective down to ~80 Hz
PAD_CELLS      = 56      # 168 mm free air + 168 mm PML = 336 mm total buffer each side
STAND_HEIGHT_M = 0.40   # height of guitar body bottom above stage floor (world Z)
STAGE_W_M      = 4.0    # stage width  (world X)
STAGE_D_M      = 3.5    # stage depth  (world Y, back-to-front)
STAGE_H_M      = 3.0    # stage height (world Z)

N_RENDER_SEGS = 240    # string segments for rendering; passed to get_string_displacement_n
PLATE_THETA_SEGS = 1024
PLATE_RADIAL_SEGS = 96
USE_GPU_RAY_FIELD = True
GPU_RAY_FIELD_DIMS = (512, 640, 160)
GPU_RAY_FIELD_SCALE = 4.0
GPU_RAY_FIELD_REFERENCE_RAYS = 100_000
GPU_PRESSURE_SCALE  = 4.0   # FDTD signed Pa → normalised; same scale worked pre-rename
GPU_RAY_LOG_SCALE = True
GPU_RAY_FIELD_GAMMA = 0.62
GPU_RAY_SEGMENT_CAP = 200000
STAGE_LIGHT_RAYS = 200_000
STAGE_LIGHT_DIMS = (384, 384, 256)
# Maximum rays dispatched per glDispatchCompute call.  Keeping batches small
# prevents the GPU TDR watchdog (Windows: ~2 s) from killing the process when
# running high ray counts (e.g. 10 M).  The field texture accumulates correctly
# across batches because splat() uses imageAtomicAdd.
GPU_DISPATCH_BATCH = 65_536
RAY_TRACE_RAYS = 512
RAY_MAX_BOUNCES = 8

# ── Spectral frequency bands ──────────────────────────────────────────────────
# 4 bands spanning the audible guitar range, anchored at 440 Hz.
# Rays carry energy in each band; bands mix to white when balanced.
# Deviation from balance produces a warm (sub-440) or cool (super-440) tint.
SPEC_BAND_EDGES      = (55.0, 220.0, 440.0, 880.0, 5000.0)  # Hz boundaries
STRING_FUNDAMENTALS_HZ = (82.4, 110.0, 146.8, 196.0, 246.9, 329.6)  # E2 A2 D3 G3 B3 E4
STRING_GAUGES_IN = (0.046, 0.036, 0.026, 0.017, 0.013, 0.010)
STRING_TENSIONS_N = (61.0, 76.8, 90.5, 112.9, 106.5, 73.0)
SCALE_LENGTH_M = GUITAR_SCALE_LENGTH_M if _HAS_BRIDGE else 0.648

# Canonical guitar geometry (from orchestral_resonance)
BODY_H        = 0.060
SOUNDHOLE_CX  = 0.0
SOUNDHOLE_CY  = 0.050
SOUNDHOLE_R   = 0.028
BRIDGE_POS    = [(-0.030, -0.070), (0.000, -0.070), (0.030, -0.070)]

STRUM_OFFSETS = [0, 2205, 4410, 6615, 8820, 11025]
A_STRING_INDEX = 1
PLUCK_POS     = 0.20
PLUCK_AMP     = 0.003
BRIDGE_FORCE_SCALE = 1.0

PLATE_SCALE  = 0.012
ENV_DECAY         = 0.985  # retained for old cached renderer state; no peak-hold draw
STRING_CLEARANCE = 0.010

STRING_COLORS = [
    # Steel strings: plain treble (bright silver) → wound bass (darker)
    (0.92, 0.92, 0.88, 1.0),   # e1 — plain high-carbon steel
    (0.88, 0.88, 0.84, 1.0),   # B  — plain steel
    (0.84, 0.84, 0.80, 1.0),   # G  — plain steel
    (0.76, 0.74, 0.68, 1.0),   # D  — wound (nickel wrap, slightly warm)
    (0.70, 0.68, 0.62, 1.0),   # A  — wound (darker)
    (0.62, 0.60, 0.54, 1.0),   # E6 — wound (darkest, most oxidised)
]

RAY_BOUNCE_RGBA = np.array([
    [1.00, 0.90, 0.20, 0.80],
    [1.00, 0.50, 0.10, 0.65],
    [0.90, 0.10, 0.10, 0.50],
    [0.70, 0.20, 0.90, 0.40],
], dtype=np.float32)


@dataclass(frozen=True)
class SurfaceMaterialSpec:
    color: tuple[float, float, float]
    inner_color: tuple[float, float, float]
    ambient: float
    spec_strength: float
    shininess: float
    grain: float
    opaque_alpha: float = 1.0
    alpha_alpha: float = 0.42


@dataclass(frozen=True)
class PlateMaterialSpec:
    opaque_alpha: float
    alpha_alpha: float
    color_mix: float


_MAT_BACK = SurfaceMaterialSpec(
    color=(0.16, 0.035, 0.018),
    inner_color=(0.64, 0.38, 0.18),
    ambient=0.24,
    spec_strength=0.42,
    shininess=96.0,
    grain=0.25,
    opaque_alpha=0.98,
    alpha_alpha=0.48,
)
_MAT_SIDES = SurfaceMaterialSpec(
    color=(0.18, 0.035, 0.018),
    inner_color=(0.72, 0.44, 0.21),
    ambient=0.22,
    spec_strength=0.55,
    shininess=128.0,
    grain=0.65,
    opaque_alpha=1.0,
    alpha_alpha=0.36,
)
_MAT_STAGE = SurfaceMaterialSpec(
    color=(0.46, 0.45, 0.42),
    inner_color=(0.46, 0.45, 0.42),
    ambient=0.22,
    spec_strength=0.02,
    shininess=12.0,
    grain=0.08,
    opaque_alpha=0.82,
    alpha_alpha=0.44,
)
_MAT_NECK = SurfaceMaterialSpec(
    color=(0.20, 0.085, 0.035),
    inner_color=(0.50, 0.28, 0.12),
    ambient=0.26,
    spec_strength=0.36,
    shininess=88.0,
    grain=0.55,
    opaque_alpha=0.95,
    alpha_alpha=0.52,
)
_MAT_PLATE = PlateMaterialSpec(
    opaque_alpha=1.0,
    alpha_alpha=0.42,
    color_mix=1.0,
)


def _layer_material_alpha(layer_mode: int, *, opaque_alpha: float, alpha_alpha: float) -> float:
    if layer_mode == LAYER_HIDDEN:
        return 0.0
    if layer_mode == LAYER_OPAQUE:
        return float(opaque_alpha)
    return float(alpha_alpha)


def _format_hud_time(samples: int, sample_rate: int = SAMPLE_RATE) -> str:
    samples = max(0, int(samples))
    seconds = samples / float(sample_rate)
    minutes = int(seconds // 60.0)
    rem = seconds - minutes * 60.0
    return f"{minutes}:{rem:05.2f}"


# ── Disk cache ────────────────────────────────────────────────────────────────
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "demo_pluck_gl")


def _band_maps_cache_path(dx: float, n_segs: int, margin_cells: int, pml_cells: int) -> str:
    key = hashlib.md5(
        f"band_maps_v2_dx{dx:.6f}_segs{n_segs}_margin{margin_cells}_pml{pml_cells}".encode()
    ).hexdigest()
    return os.path.join(_CACHE_DIR, f"band_maps_{key}.npz")


def _pressure_pad_cells_from_margin(margin_cells: int, pml_cells: int) -> int:
    return int(margin_cells) + int(pml_cells)


def _try_load_band_maps(path: str):
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        arr = d['band_maps']
        print(f"  [cache] band_maps loaded from {path}", flush=True)
        return arr
    except Exception as e:
        print(f"  [cache] failed to load band_maps: {e}", flush=True)
        return None


def _save_band_maps(band_maps: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, band_maps=band_maps)
    print(f"  [cache] band_maps saved → {path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Math
# ─────────────────────────────────────────────────────────────────────────────

def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def _lookat(eye, center, up):
    f = _norm(center - eye)
    r = _norm(np.cross(f, up))
    u = np.cross(r, f)
    return np.array([
        [ r[0],  r[1],  r[2], -np.dot(r, eye)],
        [ u[0],  u[1],  u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [    0,     0,     0,               1 ],
    ], dtype=np.float32)

def _persp(fov_y, aspect, near, far):
    f = 1.0 / math.tan(fov_y * 0.5)
    return np.array([
        [f/aspect,  0,  0,                        0            ],
        [0,         f,  0,                        0            ],
        [0,         0,  (far+near)/(near-far),    2*far*near/(near-far)],
        [0,         0, -1,                        0            ],
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Guitar geometry
# ─────────────────────────────────────────────────────────────────────────────

def _guitar_outline(n_pts: int = 128) -> np.ndarray:
    """Canonical guitar outline — mirrors orchestral_resonance._guitar_outline exactly."""
    lower_r = 0.175; upper_r = 0.135; waist_x = 0.105
    lower_cy = -0.090; upper_cy = 0.100
    y_bot = lower_cy - lower_r;  y_top = upper_cy + upper_r
    y_span = y_top - y_bot
    n_half = n_pts // 2
    ys = np.linspace(y_bot + 1e-6, y_top - 1e-6, n_half)
    xs = np.empty(n_half, dtype=np.float64)
    for k, y in enumerate(ys):
        d2_lo = max(0.0, lower_r**2 - (y - lower_cy)**2)
        d2_hi = max(0.0, upper_r**2 - (y - upper_cy)**2)
        x_env = max(math.sqrt(d2_lo), math.sqrt(d2_hi))
        t    = (y - y_bot) / y_span
        wenv = math.exp(-((t - (0.0 - y_bot) / y_span) / 0.14)**2)
        xs[k] = x_env * (1.0 - wenv) + waist_x * wenv
    right = np.column_stack([ xs,        ys      ])
    left  = np.column_stack([-xs[::-1],  ys[::-1]])
    return np.concatenate([right, left]).astype(np.float32)[:n_pts]


def _effective_scale_length(fret: int = 0) -> float:
    fret = max(0, int(fret))
    return float(SCALE_LENGTH_M / (2.0 ** (fret / 12.0)))


def _string_paths(outline: np.ndarray, body_h: float,
                  n_strings: int = 6, n_segs: int = N_RENDER_SEGS,
                  fret: int = 0) -> List[np.ndarray]:
    """Matches build_acoustic_coevolver_from_scene string layout exactly."""
    y_saddle = -0.070   # bridge saddle y (matches BRIDGE_POS)
    y_nut    = y_saddle + _effective_scale_length(fret)
    x_span   = 0.0088 * max(0, n_strings - 1)   # ~8.8 mm per gap → 44 mm for 6
    string_z = body_h + STRING_CLEARANCE
    paths = []
    for si in range(n_strings):
        x = -x_span / 2 + si * (x_span / max(1, n_strings - 1))
        x_nut = x * 0.72
        ys   = np.linspace(y_nut, y_saddle, n_segs + 1, dtype=np.float32)
        xs   = np.linspace(x_nut, x, n_segs + 1, dtype=np.float32)
        path = np.column_stack([
            xs,
            ys,
            np.full(n_segs + 1, string_z, dtype=np.float32),
        ])
        paths.append(path)
    return paths


def _neck_geometry(outline: np.ndarray, body_h: float, n_strings: int = 6,
                   active_fret: int = 0, fretless: bool = False):
    y_body = float(outline[:, 1].max()) * 0.85
    y_nut = BRIDGE_POS[1][1] + SCALE_LENGTH_M
    y_head = y_nut + 0.13
    z = body_h + 0.003
    neck_w0, neck_w1 = 0.056, 0.044
    head_w = 0.092
    neck = np.array([
        [-neck_w0 * 0.5, y_body, z], [ neck_w0 * 0.5, y_body, z],
        [ neck_w1 * 0.5, y_nut,  z], [-neck_w1 * 0.5, y_nut,  z],
    ], np.float32)
    head = np.array([
        [-head_w * 0.42, y_nut, z], [ head_w * 0.42, y_nut, z],
        [ head_w * 0.58, y_head, z + 0.002], [-head_w * 0.58, y_head, z + 0.002],
    ], np.float32)
    wood = np.vstack([neck[[0,1,2, 0,2,3]], head[[0,1,2, 0,2,3]]]).astype(np.float32)

    fret_lines = []
    scale_len = SCALE_LENGTH_M
    for fret in range(1, 21):
        y = y_nut - scale_len / (2.0 ** (fret / 12.0))
        if y <= y_body:
            continue
        t = (y - y_body) / max(y_nut - y_body, 1e-6)
        half_w = 0.5 * ((1.0 - t) * neck_w0 + t * neck_w1)
        fret_lines.extend([[-half_w, y, z + 0.0025], [half_w, y, z + 0.0025]])

    x_span = 0.0088 * max(0, n_strings - 1)
    ext_strings = []
    pin_lines = []
    for si in range(n_strings):
        x_saddle = -x_span / 2 + si * (x_span / max(1, n_strings - 1))
        x_nut = x_saddle * 0.72
        x_tune = (-0.038 if si < n_strings // 2 else 0.038)
        y_tune = y_nut + 0.025 + (si % 3) * 0.035
        ext_strings.append(np.array([
            [x_nut, y_nut, body_h + STRING_CLEARANCE],
            [x_tune, y_tune, body_h + STRING_CLEARANCE + 0.003],
        ], np.float32))
        pin_lines.extend([[x_tune - 0.010, y_tune, z + 0.008],
                          [x_tune + 0.010, y_tune, z + 0.008]])

    anchor_lines = []
    if active_fret > 0:
        y = y_nut - scale_len / (2.0 ** (int(active_fret) / 12.0))
        if y > y_body:
            t = (y - y_body) / max(y_nut - y_body, 1e-6)
            half_w = 0.5 * ((1.0 - t) * neck_w0 + t * neck_w1)
            z_anchor = z + (0.006 if fretless else 0.004)
            anchor_lines.extend([[-half_w, y, z_anchor], [half_w, y, z_anchor]])

    return (wood, np.asarray(fret_lines, np.float32),
            np.asarray(pin_lines, np.float32), ext_strings,
            np.asarray(anchor_lines or [[0, 0, 0], [0, 0, 0]], np.float32))


def _electroacoustic_fixtures(body_h: float, y_saddle: float = -0.070) -> tuple[np.ndarray, np.ndarray]:
    """Small visual pickup bar and soundhole mic marker in guitar coordinates."""
    # Guitar Z becomes world stage-depth after _guitar_model_matrix().  Keep the
    # pickup nearly flush with the soundboard; the mic intentionally stands off.
    z = body_h + 0.002
    pickup_y = y_saddle - 0.020
    pickup = np.array([
        [-0.037, pickup_y - 0.004, z], [ 0.037, pickup_y - 0.004, z],
        [ 0.037, pickup_y + 0.004, z], [-0.037, pickup_y + 0.004, z],
    ], np.float32)
    mic_z = body_h + 0.075
    mic = np.array([
        [-0.010, 0.000, mic_z], [0.010, 0.000, mic_z],
        [0.000, -0.010, mic_z], [0.000, 0.010, mic_z],
        [0.000, 0.000, mic_z], [0.000, 0.000, body_h + 0.006],
    ], np.float32)
    return pickup, mic


def _pip_grid_demo(X: np.ndarray, Y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    inside = np.zeros(X.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        inside ^= ((yi > Y) != (yj > Y)) & (
            X < (xj - xi) * (Y - yi) / (yj - yi + 1e-15) + xi
        )
        j = i
    return inside


def _soundhole_cutout(plate_active: np.ndarray, info: dict) -> np.ndarray:
    dx = float(info.get('dx', DX))
    xs = info['gx_min'] + np.arange(info['Nx']) * dx
    ys = info['gy_min'] + np.arange(info['Ny']) * dx
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    mask = (X - SOUNDHOLE_CX)**2 + (Y - SOUNDHOLE_CY)**2 < SOUNDHOLE_R**2
    out = plate_active.copy()
    out[mask] = 0
    return out


def _disc_verts(cx, cy, z, r=SOUNDHOLE_R, n=48) -> np.ndarray:
    a = np.linspace(0, 2*math.pi, n, endpoint=False)
    ring = np.column_stack([cx + r*np.cos(a), cy + r*np.sin(a),
                            np.full(n, z, np.float32)]).astype(np.float32)
    cen = np.array([[cx, cy, z]], np.float32)
    return np.vstack([cen, ring, ring[[0]]])


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

def _build_physics(n_strings=6, *, dx=DX, pad_cells=PAD_CELLS, n_pml=N_PML,
                   n_segs=N_RENDER_SEGS,
                   force_scale=BRIDGE_FORCE_SCALE,
                   fret: int = 0,
                   fretless: bool = False):
    """Returns (scene, ce, info, body_h) or (None, None, fallback_info, BODY_H)."""
    if not (_HAS_PHYSICS and _HAS_BRIDGE and _HAS_SCENE):
        return None, None, None, None

    scene = _build_body_scene_fn("string_plate")
    if scene is None:
        return None, None, None, None

    ce, info = build_acoustic_coevolver_from_scene(
        scene, n_strings=n_strings, sample_rate=float(SAMPLE_RATE),
        dx=dx, pad_cells=pad_cells, n_pml=n_pml, n_segs=n_segs,
        force_scale=force_scale,
        scale_length_m=_effective_scale_length(fret),
        fretless=bool(fretless),
        fret_number=int(fret))
    if ce is None:
        return scene, None, info, None

    # Extract body_h from scene geometry
    _, body_h, _ = _extract_guitar_geometry(scene)
    body_h = body_h or BODY_H

    # Get plate_active for rendering mesh.
    # Build it on the same grid as info (Nx×Ny at info['dx']) so that
    # _soundhole_cutout's boolean indexing is always shape-safe.  This matters
    # especially in AMR mode where 'dx' is _viz_dx, not the physics dx.
    outline_for_vox, _, _ = _extract_guitar_geometry(scene)
    if outline_for_vox is None or len(outline_for_vox) < 8:
        outline_for_vox = _guitar_outline()

    _pdx  = float(info.get('dx', DX))
    _pNx  = int(info['Nx'])
    _pNy  = int(info['Ny'])
    _pxs  = info['gx_min'] + (np.arange(_pNx) + 0.5) * _pdx
    _pys  = info['gy_min'] + (np.arange(_pNy) + 0.5) * _pdx
    _pX, _pY = np.meshgrid(_pxs, _pys, indexing='ij')
    plate_active = _pip_grid_demo(_pX, _pY,
                                  np.asarray(outline_for_vox, dtype=np.float64)
                                  ).astype(np.uint8)
    info['plate_active_2d'] = _soundhole_cutout(plate_active, info)

    return scene, ce, info, body_h


def _schedule_strum(ce, n_strings=6):
    ce.clear_pluck_schedule()
    for si in range(n_strings):
        onset = STRUM_OFFSETS[si] if si < len(STRUM_OFFSETS) else si * 2205
        ce.schedule_pluck(onset, si, PLUCK_POS, PLUCK_AMP)


def _schedule_a_pluck(ce, n_strings=6):
    ce.clear_pluck_schedule()
    si = min(A_STRING_INDEX, max(0, n_strings - 1))
    ce.schedule_pluck(0, si, PLUCK_POS, PLUCK_AMP)


def _schedule_excitation(ce, kind: str, n_strings=6):
    if hasattr(ce, "clear_pluck_schedule"):
        ce.clear_pluck_schedule()
    if kind == "rest":
        return
    if kind == "a-pluck":
        _schedule_a_pluck(ce, n_strings=n_strings)
    else:
        _schedule_strum(ce, n_strings=n_strings)


def _excitation_total_samples(kind: str, diagnostic_frames: int = 600) -> int:
    if kind == "rest":
        return max(1, int(diagnostic_frames)) * BLOCK_SAMPLES
    if kind == "a-pluck":
        return SAMPLE_RATE * 5
    return max(STRUM_OFFSETS) + SAMPLE_RATE * 5


# ─────────────────────────────────────────────────────────────────────────────
# Frame
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Frame:
    pressure: np.ndarray       # (Nx, Ny, Nz)
    plate:    np.ndarray       # (Nx, Ny)
    strings:  List[np.ndarray] # [n_strings] each (N_RENDER_SEGS+1, 3) absolute positions
    mic:      Optional[np.ndarray] = None
    pickup:   Optional[np.ndarray] = None
    frame_index: int = -1


def physics_frame(ce, n_strings: int, n_render_segs: int = N_RENDER_SEGS) -> Frame:
    ce.step(BLOCK_SAMPLES)
    return Frame(
        pressure = ce.get_pressure_field(),
        plate    = ce.get_plate_displacement(),
        strings  = [ce.get_string_position_n(si, n_render_segs + 1)
                    for si in range(n_strings)],
        mic      = ce.get_mic_output(0, BLOCK_SAMPLES) if hasattr(ce, "get_mic_output") else None,
        pickup   = ce.get_pickup_output(0, BLOCK_SAMPLES) if hasattr(ce, "get_pickup_output") else None,
    )


# ── Divergence diagnostics ─────────────────────────────────────────────────────
# Rolling window of per-frame field statistics; consulted when step() throws.

_DIAG_HISTORY: collections.deque = collections.deque(maxlen=20)


def _record_diag(frame: Frame, fi: int) -> None:
    p_max  = float(np.abs(frame.pressure).max())
    pl_max = float(np.abs(frame.plate).max())
    s_max = 0.0
    for s in frame.strings:
        if len(s) < 2:
            continue
        t = np.linspace(0.0, 1.0, len(s), dtype=np.float32)[:, None]
        chord = (1.0 - t) * s[0:1] + t * s[-1:]
        s_max = max(s_max, float(np.linalg.norm(s - chord, axis=1).max()))
    _DIAG_HISTORY.append((fi, p_max, pl_max, s_max))


def _dump_diag() -> None:
    print("\n[divergence] Field statistics before explosion:", flush=True)
    if not _DIAG_HISTORY:
        print("  (no history — diverged on first step)", flush=True)
        return
    print(f"  {'frame':>6}  {'pressure_max':>14}  {'plate_max':>12}  {'string_max':>12}",
          flush=True)
    prev_p = None
    for (fi, p, pl, s) in _DIAG_HISTORY:
        growth = f"  ×{p/prev_p:.2f}" if (prev_p and prev_p > 0) else ""
        print(f"  {fi:>6}  {p:>14.6g}  {pl:>12.6g}  {s:>12.6g}{growth}", flush=True)
        prev_p = p
    print(flush=True)


def _print_diag_summary(label: str, start_index: int = 0) -> None:
    rows = list(_DIAG_HISTORY)[start_index:]
    if not rows:
        return
    p_max = max(row[1] for row in rows)
    pl_max = max(row[2] for row in rows)
    s_max = max(row[3] for row in rows)
    print(f"{label}: pressure_max={p_max:.6g} "
          f"plate_max={pl_max:.6g} string_max={s_max:.6g}", flush=True)


def _frame_equilibrium_stats(frame: Frame) -> tuple[float, float, float, float, float]:
    p_rms = float(np.sqrt(np.mean(np.asarray(frame.pressure, np.float32) ** 2)))
    p_max = float(np.max(np.abs(frame.pressure)))
    pl_rms = float(np.sqrt(np.mean(np.asarray(frame.plate, np.float32) ** 2)))
    pl_max = float(np.max(np.abs(frame.plate)))
    s_max = 0.0
    for s in frame.strings:
        if len(s) < 2:
            continue
        t = np.linspace(0.0, 1.0, len(s), dtype=np.float32)[:, None]
        chord = (1.0 - t) * s[0:1] + t * s[-1:]
        s_max = max(s_max, float(np.linalg.norm(s - chord, axis=1).max()))
    return p_rms, p_max, pl_rms, pl_max, s_max


def _pack_ready(scene, ce, info, body_h, config: dict) -> dict:
    outline, scene_body_h = _scene_outline_or_default(scene)
    if body_h is None:
        body_h = scene_body_h
    n_str = int(getattr(ce, "n_strings", config.get("n_strings", 6)))
    return {
        "type": "ready",
        "info": info,
        "body_h": float(body_h or BODY_H),
        "n_strings": n_str,
        "outline": np.asarray(outline, dtype=np.float32),
        "config": dict(config),
    }


def _physics_worker(cmd_q, out_q, initial_config: dict) -> None:
    """Own the coevolver in a separate process and stream renderable frames."""
    config = dict(initial_config)
    ce = scene = info = None
    body_h = BODY_H
    n_str = int(config.get("n_strings", 6))
    fi = 0

    def rebuild() -> bool:
        nonlocal ce, scene, info, body_h, n_str, fi
        scene, ce, info, bh = _build_physics(
            n_strings=int(config.get("n_strings", 6)),
            dx=float(config["dx"]),
            pad_cells=_pressure_pad_cells_from_margin(
                int(config["pressure_margin_cells"]),
                int(config["pressure_pml_cells"])),
            n_pml=int(config["pressure_pml_cells"]),
            n_segs=int(config["render_segs"]),
            force_scale=float(config["bridge_force_scale"]),
            fret=int(config.get("fret", 0)),
            fretless=bool(config.get("fretless", False)))
        if ce is None:
            out_q.put({"type": "error", "message": "Physics build failed"})
            return False
        body_h = bh if bh is not None else BODY_H
        n_str = int(ce.n_strings)
        if hasattr(ce, "clear_pluck_schedule"):
            ce.clear_pluck_schedule()
        ce.reset()
        _schedule_excitation(ce, str(config["excitation"]), n_str)
        fi = 0
        out_q.put(_pack_ready(scene, ce, info, body_h, config))
        return True

    try:
        if not rebuild():
            return
        while True:
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd:
                ctype = cmd.get("type")
                if ctype == "quit":
                    return
                if ctype == "restart":
                    ce.reset()
                    _schedule_excitation(ce, str(config["excitation"]), n_str)
                    fi = 0
                    out_q.put({"type": "restarted"})
                elif ctype == "reconfigure":
                    config.update(cmd.get("config", {}))
                    out_q.put({"type": "rebuilding", "config": dict(config)})
                    if not rebuild():
                        return
                continue

            total = _excitation_total_samples(
                str(config["excitation"]),
                int(config.get("diagnostic_frames", 600)))
            if fi * BLOCK_SAMPLES >= total:
                out_q.put({"type": "done", "frames": fi})
                try:
                    cmd = cmd_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if cmd.get("type") == "quit":
                    return
                if cmd.get("type") == "restart":
                    ce.reset()
                    _schedule_excitation(ce, str(config["excitation"]), n_str)
                    fi = 0
                    out_q.put({"type": "restarted"})
                elif cmd.get("type") == "reconfigure":
                    config.update(cmd.get("config", {}))
                    out_q.put({"type": "rebuilding", "config": dict(config)})
                    if not rebuild():
                        return
                continue

            frame = physics_frame(ce, n_str, int(config["render_segs"]))
            out_q.put({"type": "frame", "index": fi, "frame": frame})
            fi += 1
    except BaseException as exc:
        out_q.put({
            "type": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


class _PhysicsProcess:
    def __init__(self, config: dict):
        ctx = mp.get_context("spawn")
        self._thread = None
        try:
            self.cmd_q = ctx.Queue()
            self.out_q = ctx.Queue(maxsize=3)
            self.proc = ctx.Process(target=_physics_worker, args=(self.cmd_q, self.out_q, dict(config)))
            self.proc.daemon = True
            self.proc.start()
            self.mode = "process"
        except PermissionError as exc:
            print(f"  multiprocessing unavailable ({exc}); using worker thread fallback",
                  flush=True)
            self.cmd_q = queue.Queue()
            self.out_q = queue.Queue(maxsize=3)
            self.proc = None
            self._thread = threading.Thread(
                target=_physics_worker,
                args=(self.cmd_q, self.out_q, dict(config)),
                daemon=True)
            self._thread.start()
            self.mode = "thread"

    def send(self, message: dict) -> None:
        self.cmd_q.put(message)

    def poll(self) -> list:
        out = []
        while True:
            try:
                out.append(self.out_q.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self) -> None:
        try:
            self.cmd_q.put({"type": "quit"})
        except Exception:
            pass
        if self.proc is not None:
            self.proc.join(timeout=1.0)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=1.0)
        elif self._thread is not None:
            self._thread.join(timeout=1.0)


class _WavCapture:
    def __init__(self, path: str, channels: int, sample_rate: int = SAMPLE_RATE):
        self.path = path
        self.channels = max(1, int(channels))
        self._wf = wave.open(path, "wb")
        self._wf.setnchannels(self.channels)
        self._wf.setsampwidth(2)
        self._wf.setframerate(int(sample_rate))
        self._samples = 0

    def write(self, blocks: list[np.ndarray]) -> None:
        if not blocks:
            return
        arr = np.column_stack([np.asarray(b, np.float32) for b in blocks])
        peak = max(1.0, float(np.max(np.abs(arr))) * 1.05)
        pcm = np.clip(arr / peak, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype("<i2")
        self._wf.writeframes(pcm_i16.tobytes())
        self._samples += len(arr)

    def close(self) -> None:
        self._wf.close()


class _CaptureSet:
    def __init__(self, path: str, source: str):
        self.source = source
        self.captures = []
        root, ext = os.path.splitext(path or "mic_capture.wav")
        ext = ext or ".wav"
        if source == "both":
            self.mic = _WavCapture(f"{root}_mic{ext}", 1)
            self.pickup = _WavCapture(f"{root}_pickup{ext}", 1)
            self.captures = [self.mic, self.pickup]
        else:
            self.single = _WavCapture(path, 1)
            self.captures = [self.single]

    @property
    def paths(self) -> list[str]:
        return [c.path for c in self.captures]

    def write(self, mic: np.ndarray, pickup: np.ndarray) -> None:
        if self.source == "mic":
            self.single.write([mic])
        elif self.source == "pickup":
            self.single.write([pickup])
        elif self.source == "mix":
            self.single.write([mic + pickup])
        else:
            self.mic.write([mic])
            self.pickup.write([pickup])

    def close(self) -> None:
        for cap in self.captures:
            cap.close()


# ─────────────────────────────────────────────────────────────────────────────
# Ray-segment buffer
# ─────────────────────────────────────────────────────────────────────────────

def _band_colors(n_bands: int) -> np.ndarray:
    """Spectral palette: band 0 = deep red (lowest freq) → band N-1 = violet (highest)."""
    t = np.linspace(0.0, 1.0, n_bands)
    # Piecewise RGB spectral sweep: red → orange → yellow → green → cyan → blue → violet
    r = np.clip(np.where(t < 0.25, 1.0,
                np.where(t < 0.50, 1.0 - (t - 0.25) * 4,
                np.where(t < 0.75, 0.0, (t - 0.75) * 4))), 0, 1)
    g = np.clip(np.where(t < 0.25, t * 4,
                np.where(t < 0.50, 1.0,
                np.where(t < 0.75, 1.0 - (t - 0.50) * 4, 0.0))), 0, 1)
    b = np.clip(np.where(t < 0.50, 0.0,
                np.where(t < 0.75, (t - 0.50) * 4, 1.0)), 0, 1)
    a = np.full(n_bands, 0.55, dtype=np.float32)
    return np.column_stack([r, g, b, a]).astype(np.float32)


def _ray_vbo(segs: Optional[np.ndarray]) -> np.ndarray:
    """(N,12) segments → interleaved (2N, 7) [xyz rgba] for GL_LINES.
    RGB = band spectral hue blended with bounce tint.
    Alpha = bounce opacity * normalised amplitude."""
    if segs is None or len(segs) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    n_bands = int(segs[:, 8].max()) + 1
    palette  = _band_colors(n_bands)                             # (n_bands, 4)
    bi  = np.clip(segs[:, 8].astype(np.int32), 0, n_bands - 1)
    bci = np.clip(segs[:, 7].astype(np.int32), 0, len(RAY_BOUNCE_RGBA) - 1)
    band_col   = palette[bi]                                     # (N, 4)
    bounce_col = RAY_BOUNCE_RGBA[bci]                            # (N, 4)
    # RGB: 60 % spectral band hue + 40 % bounce tint
    rgb   = band_col[:, :3] * 0.6 + bounce_col[:, :3] * 0.4
    # Alpha: bounce opacity table scaled by normalised amplitude
    amp = segs[:, 9]; amp_max = amp.max()
    if amp_max > 1e-9: amp = amp / amp_max
    alpha = (bounce_col[:, 3] * np.clip(amp, 0.0, 1.0)).reshape(-1, 1)
    rgba  = np.concatenate([rgb, alpha], axis=1).astype(np.float32)
    n = len(segs)
    v0 = np.concatenate([segs[:, 0:3], rgba], axis=1)
    v1 = np.concatenate([segs[:, 3:6], rgba], axis=1)
    out = np.empty((2 * n, 7), np.float32)
    out[0::2] = v0; out[1::2] = v1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mesh builders
# ─────────────────────────────────────────────────────────────────────────────

def _side_walls(outline: np.ndarray, body_h: float):
    """Indexed triangle mesh for the extruded guitar side walls.

    Returns (verts (N*4, 3), normals (N*4, 3), indices (N*6,)) float32/int32.
    Each edge of the outline generates one quad, normals pointing outward.
    """
    N = len(outline)
    cen = outline.mean(axis=0)
    verts   = np.empty((N * 4, 3), np.float32)
    normals = np.empty((N * 4, 3), np.float32)
    for i in range(N):
        a = outline[i]
        b = outline[(i + 1) % N]
        mid = 0.5 * (a + b)
        nx, ny = mid - cen
        nlen = math.hypot(nx, ny)
        if nlen > 1e-9:
            nx /= nlen; ny /= nlen
        base = i * 4
        for j, (pt, z) in enumerate([(a, 0.0), (b, 0.0), (b, body_h), (a, body_h)]):
            verts  [base + j] = [pt[0], pt[1], z]
            normals[base + j] = [nx, ny, 0.0]

    idx = []
    for i in range(N):
        b = i * 4
        idx += [b, b+1, b+2,  b, b+2, b+3]
    return verts, normals, np.array(idx, np.int32)


def _ray_outline_radius(outline: np.ndarray, cx: float, cy: float, theta: float) -> float:
    """Distance from (cx, cy) to the furthest outline intersection on a ray."""
    dx, dy = math.cos(theta), math.sin(theta)
    best = None
    n = len(outline)
    for i in range(n):
        x0, y0 = float(outline[i, 0]) - cx, float(outline[i, 1]) - cy
        x1, y1 = float(outline[(i + 1) % n, 0]) - cx, float(outline[(i + 1) % n, 1]) - cy
        ex, ey = x1 - x0, y1 - y0
        det = dx * (-ey) - dy * (-ex)
        if abs(det) < 1e-10:
            continue
        t = (x0 * (-ey) - y0 * (-ex)) / det
        u = (dx * y0 - dy * x0) / det
        if t > 0.0 and -1e-6 <= u <= 1.0 + 1e-6:
            best = t if best is None else max(best, t)
    return float(best if best is not None else SOUNDHOLE_R)


def _plate_mesh(outline: np.ndarray, *, n_theta: int = 192, n_radial: int = 18):
    """Smooth fan mesh for the guitar top, with a real soundhole boundary."""
    verts = []
    for ri in range(n_radial + 1):
        frac = ri / float(n_radial)
        for ai in range(n_theta):
            th = 2.0 * math.pi * ai / float(n_theta)
            outer = _ray_outline_radius(outline, SOUNDHOLE_CX, SOUNDHOLE_CY, th)
            r = SOUNDHOLE_R + frac * max(0.0, outer - SOUNDHOLE_R)
            verts.append((SOUNDHOLE_CX + r * math.cos(th),
                          SOUNDHOLE_CY + r * math.sin(th)))
    idx = []
    for ri in range(n_radial):
        row = ri * n_theta
        nxt = (ri + 1) * n_theta
        for ai in range(n_theta):
            a = row + ai
            b = row + ((ai + 1) % n_theta)
            c = nxt + ((ai + 1) % n_theta)
            d = nxt + ai
            idx += [a, d, c, a, c, b]
    return np.asarray(verts, np.float32), np.asarray(idx, np.int32)


def _outline_mask(outline: np.ndarray, info: dict, *, cut_soundhole: bool = True) -> np.ndarray:
    """Return float32 (Nx, Ny) mask for grid cell centres inside outline."""
    Nx, Ny = int(info['Nx']), int(info['Ny'])
    dx = float(info.get('dx', DX))
    xs = float(info['gx_min']) + (np.arange(Nx, dtype=np.float32) + 0.5) * dx
    ys = float(info['gy_min']) + (np.arange(Ny, dtype=np.float32) + 0.5) * dx
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    inside = np.zeros((Nx, Ny), dtype=bool)
    poly = np.asarray(outline, dtype=np.float32)
    n = len(poly)
    for k in range(n):
        x0, y0 = poly[k]
        x1, y1 = poly[(k + 1) % n]
        crosses = ((y0 > Y) != (y1 > Y))
        x_at_y = (x1 - x0) * (Y - y0) / ((y1 - y0) + 1e-20) + x0
        inside ^= crosses & (X < x_at_y)
    if cut_soundhole:
        inside &= (X - SOUNDHOLE_CX) ** 2 + (Y - SOUNDHOLE_CY) ** 2 >= SOUNDHOLE_R ** 2
    return inside.astype(np.float32)


def _guitar_model_matrix(outline: np.ndarray, stand_height: float = STAND_HEIGHT_M):
    """4×4 model matrix: guitar-frame → world-frame (column-vector convention).

    Guitar frame: X=lateral, Y=longitudinal (neck→+Y), Z=depth (back=0, soundboard=body_h)
    World frame:  X=lateral, Y=stage-depth (+Y toward audience), Z=up (+Z = up)

    The swap Y↔Z makes the neck point skyward and the soundboard face the audience.
    The Z-translation lifts the bottom of the body to `stand_height` above the floor.
    """
    y_bot = float(outline[:, 1].min())
    tz = stand_height - y_bot   # lifts outline y_bot → world Z = stand_height

    M = np.array([
        [1., 0., 0., 0. ],   # X_world = X_guitar
        [0., 0., 1., 0. ],   # Y_world = Z_guitar  (soundboard depth → stage front/back)
        [0., 1., 0., tz ],   # Z_world = Y_guitar + tz  (neck axis → up, lifted)
        [0., 0., 0., 1. ],
    ], dtype=np.float32)

    # Inverse: swap Y↔Z then undo the Z translation.
    # Derivation: R=[[1,0,0],[0,0,1],[0,1,0]] is its own inverse; -R*t = (0,0,-tz)→(0,-tz,0)
    Minv = np.array([
        [1., 0., 0.,  0. ],   # X_guitar = X_world
        [0., 0., 1., -tz ],   # Y_guitar = Z_world - tz
        [0., 1., 0.,  0. ],   # Z_guitar = Y_world
        [0., 0., 0.,  1. ],
    ], dtype=np.float32)

    return M, Minv


def _stage_mesh(outline: np.ndarray, body_h: float):
    """Floor, three walls, and a guitar stand for the world-space stage.

    World frame: X=lateral, Y=stage-depth (guitar soundboard faces +Y),
                 Z=up.  Guitar body occupies roughly X∈[-0.20,+0.20],
                 Y∈[0, body_h], Z∈[STAND_HEIGHT_M, STAND_HEIGHT_M+0.50].

    Returns (verts, norms, mats) shaped (-1,9), (-1,3), (-1,6) — same
    layout as the old _stage_mesh so the VAO upload code is unchanged.
    """
    hw = STAGE_W_M * 0.5          # half-width  (X)
    hd = STAGE_D_M * 0.5          # half-depth  (Y)
    sh = STAGE_H_M                # ceiling height (Z)
    y_back = -hd                  # back of stage
    y_front = hd                  # front (open to audience)

    verts, norms, mats = [], [], []

    def quad(a, b, c, d, n, mat):
        verts.extend([[a, b, c], [a, c, d]])
        norms.extend([n, n])
        mats.extend([mat, mat])

    # ── Floor (faces up, Z-normal = +Z) ──────────────────────────────────────
    diff_floor = [0.30, 0.22, 0.16,  0.30, 0.22, 0.16]   # dark hardwood
    quad([-hw, y_back, 0.], [ hw, y_back, 0.],
         [ hw, y_front, 0.], [-hw, y_front, 0.],
         [0., 0., 1.], diff_floor)

    # ── Back wall (faces audience, +Y normal) ─────────────────────────────────
    diff_wall = [0.22, 0.20, 0.18,  0.22, 0.20, 0.18]   # grey concrete
    quad([-hw, y_back, 0.], [ hw, y_back, 0.],
         [ hw, y_back, sh], [-hw, y_back, sh],
         [0., 1., 0.], diff_wall)

    # ── Left wall (faces right, +X normal) ────────────────────────────────────
    quad([-hw, y_back, 0.], [-hw, y_front, 0.],
         [-hw, y_front, sh], [-hw, y_back, sh],
         [1., 0., 0.], diff_wall)

    # ── Right wall (faces left, -X normal) ────────────────────────────────────
    quad([ hw, y_front, 0.], [ hw, y_back, 0.],
         [ hw, y_back, sh], [ hw, y_front, sh],
         [-1., 0., 0.], diff_wall)

    # ── Guitar stand — simple X-frame in world space ───────────────────────────
    # Stand legs go from guitar bottom (world X=0, Y=body_h/2, Z=STAND_HEIGHT_M)
    # to four floor contacts (world Z=0).
    gx, gy, gz = 0., body_h * 0.5, STAND_HEIGHT_M
    leg_spread_x, leg_spread_y, leg_h = 0.16, 0.06, 0.02
    diff_stand = [0.12, 0.10, 0.08,  0.12, 0.10, 0.08]
    contacts = [
        (-leg_spread_x, gy - leg_spread_y, 0.),
        ( leg_spread_x, gy - leg_spread_y, 0.),
        (-leg_spread_x, gy + leg_spread_y, 0.),
        ( leg_spread_x, gy + leg_spread_y, 0.),
    ]
    top = [gx, gy, gz]
    for cx_, cy_, cz_ in contacts:
        # Make each leg a thin flat quad (two tris) with an outward normal
        dx_ = cx_ - gx; dy_ = cy_ - gy; dz_ = cz_ - gz
        ln  = np.linalg.norm([dx_, dy_, dz_])
        lx, ly, lz = dx_/ln, dy_/ln, dz_/ln
        # perpendicular in the horizontal plane for leg width
        px, py = -ly, lx
        w = 0.009
        a = [gx + px*w, gy + py*w, gz + leg_h]
        b = [gx - px*w, gy - py*w, gz + leg_h]
        c = [cx_ - px*w, cy_ - py*w, cz_ + leg_h]
        d = [cx_ + px*w, cy_ + py*w, cz_ + leg_h]
        n = [0., 0., 1.]   # approximate upward normal for flat strips
        verts.extend([[a, b, c], [a, c, d]])
        norms.extend([n, n])
        mats.extend([diff_stand, diff_stand])

    return (np.asarray(verts, np.float32).reshape(-1, 9),
            np.asarray(norms, np.float32),
            np.asarray(mats, np.float32))


def _back_fan(outline: np.ndarray) -> np.ndarray:
    cen = np.array([[outline[:,0].mean(), outline[:,1].mean(), 0.0]], np.float32)
    pts = np.column_stack([outline, np.zeros(len(outline), np.float32)])
    return np.vstack([cen, pts, pts[[0]]])


# ─────────────────────────────────────────────────────────────────────────────
# GLSL shaders — version 330 core throughout
# ─────────────────────────────────────────────────────────────────────────────

# layout(location) eliminates the need for glBindAttribLocation.
#
#   loc 0 = position (vec3)
#   loc 1 = normal   (vec3) or color (vec4) depending on program
#   loc 2 = data     (float, plate displacement)

# ── Phong body shader (walls, back plate, used with uniform colour) ───────────

# ── Ray-lit surface shader — samples the GPU band textures for irradiance ─────
#
# When the ILLUM layer is active the band 3D textures already contain ray energy.
# _RAY_SURFACE_FS samples these at the fragment world position to get irradiance
# and uses it as the diffuse term.  A small ambient Phong contribution is kept
# so the surface is never fully black in shadowed areas.
#
# The band textures use the same uBand0-uBand3, uBoxMin/Max, uWorldToGrid
# uniforms as _MARCH_FS so the same infrastructure is reused.
_RAY_SURFACE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;

uniform mat4 uMVP;
uniform mat4 uMV;
uniform mat4 uM;   // model matrix — world position

out vec3 vNormV;
out vec3 vPosV;
out vec3 vPosW;   // world-space position for texture sampling

void main() {
    vPosW       = (uM * vec4(aPos, 1.0)).xyz;
    vec4 posV   = uMV * vec4(aPos, 1.0);
    vPosV       = posV.xyz;
    vNormV      = mat3(uMV) * aNorm;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_RAY_SURFACE_FS = """
#version 330 core
in vec3 vNormV;
in vec3 vPosV;
in vec3 vPosW;
out vec4 FragColor;

uniform vec4  uColor;
uniform vec3  uInnerColor;
uniform vec3  uLightV;
uniform float uAmbient;
uniform float uSpecStrength;
uniform float uShininess;
uniform float uGrain;

// Band textures (GL_R32UI 3D)
uniform usampler3D uBand0;
uniform usampler3D uBand1;
uniform usampler3D uBand2;
uniform usampler3D uBand3;
uniform usampler3D uBaseBand0;
uniform usampler3D uBaseBand1;
uniform usampler3D uBaseBand2;
uniform usampler3D uBaseBand3;
uniform vec3  uBoxMin;
uniform vec3  uBoxMax;
uniform mat4  uWorldToGrid;   // maps world pos → normalised [0,1] voxel UVW
uniform vec3  uBaseBoxMin;
uniform vec3  uBaseBoxMax;
uniform mat4  uBaseWorldToGrid;
uniform int   uUseRayField;
uniform int   uUseBaseField;

// Tone-mapping
uniform float uRayExposure;   // default 1.0
uniform float uBaseExposure;  // default 1.0
uniform float uRayGamma;      // default 0.45

float sampleBand(usampler3D tex, vec3 uvw) {
    return float(texture(tex, uvw).r) / 65535.0;
}

void main() {
    vec3 N    = normalize(gl_FrontFacing ? vNormV : -vNormV);
    vec3 L    = normalize(uLightV);
    vec3 V    = normalize(-vPosV);
    vec3 H    = normalize(L + V);

    // Map world position into voxel UVW for band texture lookup
    vec4 field4 = uWorldToGrid * vec4(vPosW, 1.0);
    vec3 field_pos = field4.xyz / field4.w;
    vec3 uvw  = (field_pos - uBoxMin) / max(uBoxMax - uBoxMin, vec3(1e-6));
    bool in_volume = all(greaterThanEqual(uvw, vec3(0.0))) &&
                     all(lessThanEqual(uvw, vec3(1.0)));

    float dyn_irradiance = 0.0;
    if (uUseRayField != 0 && in_volume) {
        float b0 = sampleBand(uBand0, uvw);
        float b1 = sampleBand(uBand1, uvw);
        float b2 = sampleBand(uBand2, uvw);
        float b3 = sampleBand(uBand3, uvw);
        dyn_irradiance = (b0 + b1 + b2 + b3) * 0.25;
        // Beer-Lambert exposure + gamma
        dyn_irradiance = pow(clamp(dyn_irradiance * uRayExposure, 0.0, 1.0), uRayGamma);
    }

    vec4 bfield4 = uBaseWorldToGrid * vec4(vPosW, 1.0);
    vec3 bfield_pos = bfield4.xyz / bfield4.w;
    vec3 buvw = (bfield_pos - uBaseBoxMin) / max(uBaseBoxMax - uBaseBoxMin, vec3(1e-6));
    bool in_base = all(greaterThanEqual(buvw, vec3(0.0))) &&
                   all(lessThanEqual(buvw, vec3(1.0)));
    float base_irradiance = 0.0;
    if (uUseBaseField != 0 && in_base) {
        float b0 = sampleBand(uBaseBand0, buvw);
        float b1 = sampleBand(uBaseBand1, buvw);
        float b2 = sampleBand(uBaseBand2, buvw);
        float b3 = sampleBand(uBaseBand3, buvw);
        base_irradiance = (b0 + b1 + b2 + b3) * 0.25;
        base_irradiance = pow(clamp(base_irradiance * uBaseExposure, 0.0, 1.0), uRayGamma);
    }

    // Phong ambient fallback (still in view space)
    float phong_diff = max(dot(N, L), 0.0);
    float spec       = pow(max(dot(N, H), 0.0), max(uShininess, 1.0));

    vec3  base = gl_FrontFacing ? uColor.rgb : uInnerColor;
    float grain = 0.5 + 0.5 * sin(vPosV.x * 80.0 + vPosV.y * 31.0 + vPosV.z * 17.0);
    base *= mix(1.0, 0.82 + 0.28 * grain, uGrain);

    // Cached room light is the baseline; guitar emissions add on top.
    float cached = clamp(base_irradiance + dyn_irradiance, 0.0, 1.2);
    bool have_cached = (uUseBaseField != 0 && in_base) || (uUseRayField != 0 && in_volume);
    float diff = mix(uAmbient + 0.78 * phong_diff,
                     uAmbient + cached,
                     have_cached ? 1.0 : 0.0);
    vec3  col  = base * diff + vec3(1.0, 0.88, 0.62) * (uSpecStrength * spec);
    float rim  = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    col += base * rim * 0.16;
    FragColor  = vec4(col, uColor.a);
}
"""

@dataclass
class RayLightingState:
    """Cached ray-traced surface illumination data from trace_cavity_scene."""
    surface_flux:     np.ndarray   # (n_tri, n_bands) float32
    surface_direct:   np.ndarray   # (n_tri, n_bands) float32
    surface_indirect: np.ndarray   # (n_tri, n_bands) float32
    surface_rgb:      np.ndarray   # (n_tri, 3)       float32
    surface_scalar:   np.ndarray   # (n_tri,)          float32


_BODY_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;

uniform mat4 uMVP;
uniform mat4 uMV;   // view * model (for normal transform)

out vec3 vNormV;
out vec3 vPosV;

void main() {
    vec4 posV   = uMV * vec4(aPos, 1.0);
    vPosV       = posV.xyz;
    vNormV      = mat3(uMV) * aNorm;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_BODY_FS = """
#version 330 core
in vec3 vNormV;
in vec3 vPosV;
out vec4 FragColor;

uniform vec4  uColor;   // base RGBA
uniform vec3  uInnerColor;
uniform vec3  uLightV;  // light direction in view space
uniform float uAmbient;
uniform float uSpecStrength;
uniform float uShininess;
uniform float uGrain;

void main() {
    vec3 N    = normalize(gl_FrontFacing ? vNormV : -vNormV);
    vec3 L    = normalize(uLightV);
    vec3 V    = normalize(-vPosV);
    vec3 H    = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), max(uShininess, 1.0));
    vec3  base = gl_FrontFacing ? uColor.rgb : uInnerColor;
    float grain = 0.5 + 0.5 * sin(vPosV.x * 80.0 + vPosV.y * 31.0 + vPosV.z * 17.0);
    base *= mix(1.0, 0.82 + 0.28 * grain, uGrain);
    vec3  col  = base * (uAmbient + 0.78 * diff) + vec3(1.0, 0.88, 0.62) * (uSpecStrength * spec);
    float rim  = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    col += base * rim * 0.16;
    FragColor  = vec4(col, uColor.a);
}
"""

# ── Plate shader — same as body but data channel drives a heatmap ─────────────
_PLATE_VS = """
#version 330 core
layout(location=0) in vec3  aPos;
layout(location=2) in float aData;

uniform mat4 uMVP;
uniform mat4 uMV;

out vec3  vNormV;
out vec3  vPosV;
out float vData;

void main() {
    vec4 posV   = uMV * vec4(aPos, 1.0);
    vPosV       = posV.xyz;
    vNormV      = mat3(uMV) * vec3(0.0, 0.0, 1.0);
    vData       = aData;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_PLATE_FS = """
#version 330 core
in vec3  vNormV;
in vec3  vPosV;
in float vData;
out vec4 FragColor;

uniform float uAlpha;
uniform vec3  uLightV;
uniform float uColorMix;

vec3 heatmap(float v) {
    float t = clamp(v * 0.5 + 0.5, 0.0, 1.0);
    // dark blue → mid grey → hot orange/white
    vec3 a = vec3(0.04, 0.06, 0.35);
    vec3 b = vec3(0.05, 0.05, 0.07);
    vec3 c = vec3(0.55, 0.18, 0.03);
    vec3 d = vec3(1.00, 0.72, 0.12);
    if (t < 0.33) return mix(a, b, t / 0.33);
    if (t < 0.66) return mix(b, c, (t - 0.33) / 0.33);
    return mix(c, d, (t - 0.66) / 0.34);
}

void main() {
    vec3 N    = normalize(vNormV);
    vec3 L    = normalize(uLightV);
    vec3 V    = normalize(-vPosV);
    vec3 H    = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 96.0);
    float grain = 0.5 + 0.5 * sin((vPosV.x * 75.0 + vPosV.y * 18.0) + sin(vPosV.y * 45.0) * 0.35);
    vec3 wood0 = vec3(0.56, 0.30, 0.12);
    vec3 wood1 = vec3(0.82, 0.54, 0.24);
    vec3 base  = mix(wood0, wood1, grain * 0.35 + 0.30);
    vec3 wave  = heatmap(vData);
    base = mix(base, wave, uColorMix * 0.55 * smoothstep(0.04, 0.85, abs(vData)));
    vec3 col   = base * (0.20 + 0.74 * diff) + vec3(1.0, 0.82, 0.45) * (0.38 * spec);
    float rim  = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    col += base * rim * 0.12;
    FragColor  = vec4(col, uAlpha);
}
"""

# ── Flat coloured line shader ──────────────────────────────────────────────────
_LINE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
"""
_LINE_FS = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main() {
    // Strings rendered with additive blending — boost brightness for glow
    vec3 glow = uColor.rgb * 1.80;
    FragColor = vec4(glow, uColor.a);
}
"""

# ── Per-vertex-colour shader (ray segments) ───────────────────────────────────
_VCOL_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
uniform mat4 uMVP;
out vec4 vColor;
void main() { vColor = aColor; gl_Position = uMVP * vec4(aPos, 1.0); }
"""
_VCOL_FS = """
#version 330 core
in  vec4 vColor;
out vec4 FragColor;
void main() { FragColor = vColor; }
"""
# ── Volumetric ray-march ───────────────────────────────────────────────────────
_MARCH_VS = """
#version 330 core
layout(location=0) in vec2 aUV;
out vec2 vNDC;
void main() { vNDC = aUV; gl_Position = vec4(aUV, 0.0, 1.0); }
"""

_MARCH_FS = """
#version 330 core
in  vec2 vNDC;
out vec4 FragColor;

uniform mat4       uInvMVP;
uniform sampler3D  uPressure;
uniform sampler2D  uBodyMask;
uniform sampler3D  uExteriorMask;   // 1.0 = outside guitar cavity, 0.0 = inside
// ── RAY FIELD PATH (uFieldMode == 1): 4 spectral band textures ───────────────
// Band 0: 55–220 Hz  (warm amber)   Band 1: 220–440 Hz (warm white)
// Band 2: 440–880 Hz (cool white)   Band 3: 880+ Hz    (cool violet)
// All-equal bands → white.  Imbalance → warm or cool tint.
uniform usampler3D uBand0;
uniform usampler3D uBand1;
uniform usampler3D uBand2;
uniform usampler3D uBand3;
uniform vec3       uBoxMin;
uniform vec3       uBoxMax;
uniform mat4       uWorldToGrid;   // inverse guitar model matrix: world → guitar/grid frame
uniform vec2       uMaskMin;
uniform vec2       uMaskMax;
// ── PRESSURE PATH (uFieldMode == 0) ──────────────────────────────────────────
uniform float      uPressureScale;
uniform float      uPressureGamma;
// ── SHARED ────────────────────────────────────────────────────────────────────
uniform float      uRayFieldScale;
uniform float      uRayFieldGamma;
uniform float      uAlpha;
uniform int        uLogScale;
uniform int        uFieldMode;
uniform int        uUseBodyMask;
// Z-range clip: samples outside [uZClipMin, uZClipMax] are skipped.
// Use to restrict pressure to inside the guitar cavity, ray field to its box.
uniform float      uZClipMin;
uniform float      uZClipMax;
// ── AMR TBO path (uAMRMode != 0) ─────────────────────────────────────────────
// uAMRData packs (cx, cy, cz, pressure) per AMR cell as RGBA32F.
// The shader computes inverse-distance-squared weighted pressure at each
// ray-march sample point directly from the raw AMR data, giving full mesh
// resolution where cells are fine and smooth interpolation where they are coarse.
uniform samplerBuffer uAMRData;    // RGBA32F: (cx, cy, cz, pressure) per cell
uniform int           uAMRNCells;  // number of AMR cells in the TBO
uniform int           uAMRMode;    // 0 = use uPressure 3D texture; 1 = TBO IDW
uniform float         uAMREps2;    // IDW denominator floor = (half_min_spacing)^2
// uAMRHalfPow = p/2 where p is the IDW distance power.
//   p=2 (default) → uAMRHalfPow=1.0  → w = 1/r²   (no sqrt, cheapest)
//   p=3 (3-D natural) → uAMRHalfPow=1.5  → w = 1/r³
//   p=4            → uAMRHalfPow=2.0  → w = 1/r⁴  (also sqrt-free)
// Any float value works; the shader uses pow(r², -uAMRHalfPow).
uniform float         uAMRHalfPow; // default 1.0

vec3 diverge(float v) {
    float t = clamp(v * 0.5 + 0.5, 0.0, 1.0);
    vec3 a = vec3(0.00, 0.00, 0.60);
    vec3 b = vec3(0.15, 0.60, 0.90);
    vec3 c = vec3(1.00, 1.00, 1.00);
    vec3 d = vec3(0.95, 0.65, 0.00);
    vec3 e = vec3(0.85, 0.00, 0.00);
    if (t < 0.25) return mix(a, b, t / 0.25);
    if (t < 0.50) return mix(b, c, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c, d, (t - 0.50) / 0.25);
    return mix(d, e, (t - 0.75) / 0.25);
}

// ACES filmic tone mapping
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// Inverse-distance-squared IDW from raw AMR cell TBO.
// Naturally tracks AMR resolution: fine-cell regions give high interpolation
// precision; coarse-cell regions give smooth transitions.
float sampleAMRPressure(vec3 pos) {
    float sum_w = 0.0;
    float sum_wp = 0.0;
    for (int i = 0; i < uAMRNCells; i++) {
        vec4 cell = texelFetch(uAMRData, i);  // (cx, cy, cz, pressure)
        vec3 d = pos - cell.xyz;
        float r2 = dot(d, d);
        // w = r^(-p) = (r²)^(-p/2).  Using pow() keeps the formula exact for
        // any p, including odd powers like p=3 (uAMRHalfPow=1.5), while even
        // powers (p=2,4,6 → uAMRHalfPow=1,2,3) compile to multiply chains when
        // the driver constant-folds the exponent.
        float w = pow(max(r2, uAMREps2), -uAMRHalfPow);
        sum_w  += w;
        sum_wp += w * cell.w;
    }
    return (sum_w > 0.0) ? sum_wp / sum_w : 0.0;
}

void main() {
    // Reconstruct world-space ray from NDC
    vec4 nh = uInvMVP * vec4(vNDC, -1.0, 1.0);
    vec4 fh = uInvMVP * vec4(vNDC,  1.0, 1.0);
    vec3 ro_w = nh.xyz / nh.w;
    vec3 rd_w = normalize(fh.xyz / fh.w - ro_w);

    // Transform ray into guitar/grid frame for texture lookup
    vec3 ro = (uWorldToGrid * vec4(ro_w, 1.0)).xyz;
    vec3 rd = mat3(uWorldToGrid) * rd_w;

    // Slab / AABB intersection (in grid frame)
    vec3 inv_rd = 1.0 / rd;
    vec3 t0 = (uBoxMin - ro) * inv_rd;
    vec3 t1 = (uBoxMax - ro) * inv_rd;
    vec3 tmi = min(t0, t1), tma = max(t0, t1);
    float t_in  = max(max(tmi.x, tmi.y), tmi.z);
    float t_out = min(min(tma.x, tma.y), tma.z);
    if (t_out < t_in || t_out < 0.0) discard;
    t_in = max(t_in, 0.0);

    // Per-pixel step jitter — eliminates banding at zero cost
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(127.1, 311.7))) * 43758.5453);

    const int STEPS = 128;
    float dt     = (t_out - t_in) / float(STEPS);
    float step_a = 0.018 * uAlpha;

    // Vivid spectral band colours — white-balanced so equal-energy → white
    // WB scale: reciprocal of equal-mix avg so sum → (1,1,1)
    // Band 0: amber (55–220 Hz)   Band 1: warm white (220–440 Hz)
    // Band 2: cool white (440–880 Hz)  Band 3: violet (880+ Hz)
    const vec3 C0 = vec3(1.000, 0.620, 0.080);  // amber
    const vec3 C1 = vec3(1.000, 0.920, 0.740);  // warm white
    const vec3 C2 = vec3(0.760, 0.900, 1.000);  // cool white
    const vec3 C3 = vec3(0.560, 0.380, 1.000);  // violet
    // White-balance: scale so C0+C1+C2+C3 averages to (1,1,1)
    const vec3 WB = vec3(1.176, 1.370, 1.303);

    vec4 acc = vec4(0.0);
    for (int i = 0; i < STEPS; i++) {
        float fi  = float(i) + jitter;
        vec3 pos  = ro + (t_in + fi * dt) * rd;
        vec3 uvc  = (pos - uBoxMin) / (uBoxMax - uBoxMin);
        vec2 mask_uv = (pos.xy - uMaskMin) / (uMaskMax - uMaskMin);
        if (pos.z < uZClipMin || pos.z > uZClipMax) continue;
        if (uUseBodyMask != 0 && texture(uBodyMask, clamp(mask_uv, 0.0, 1.0)).r < 0.5)
            continue;
        float p;
        vec3  spectral_color;

        if (uFieldMode == 1) {
            ivec3 dims = textureSize(uBand0, 0);
            ivec3 q = clamp(ivec3(floor(uvc * vec3(dims))), ivec3(0), dims - ivec3(1));
            float sc = uRayFieldScale / 65535.0;
            float b0 = float(texelFetch(uBand0, q, 0).r) * sc;
            float b1 = float(texelFetch(uBand1, q, 0).r) * sc;
            float b2 = float(texelFetch(uBand2, q, 0).r) * sc;
            float b3 = float(texelFetch(uBand3, q, 0).r) * sc;
            if (uLogScale != 0) {
                float ls = log(1.0 + uRayFieldScale);
                b0 = log(1.0 + b0) / ls;
                b1 = log(1.0 + b1) / ls;
                b2 = log(1.0 + b2) / ls;
                b3 = log(1.0 + b3) / ls;
            }
            float total = b0 + b1 + b2 + b3 + 1e-6;
            p = pow(clamp(total, 0.0, 1.0), max(uRayFieldGamma, 0.05));

            // Raw spectral mix from 4 vivid band colours
            vec3 raw = (b0 * C0 + b1 * C1 + b2 * C2 + b3 * C3) / total;
            // Energy-adaptive saturation: sparse voxels → white, dense → vivid
            float sat = clamp(p * 3.5, 0.0, 1.0);
            spectral_color = mix(vec3(1.0), clamp(raw * WB, 0.0, 2.0), sat * 0.90);
        } else {
            float raw = (uAMRMode != 0)
                        ? sampleAMRPressure(pos)
                        : texture(uPressure, uvc).r;
            float v   = raw * uPressureScale;
            p = sign(v) * pow(clamp(abs(v), 0.0, 1.0), max(uPressureGamma, 0.05));
            // Interior cavity (ext≈0) → full diverge colormap.
            // Exterior air (ext≈1) → cyan tint at 40% opacity so the near-field
            // radiation is visible but clearly distinct from the cavity resonance.
            float ext = texture(uExteriorMask, uvc).r;
            vec3 ext_color = vec3(0.20, 0.80, 0.80);  // cyan for exterior radiation
            spectral_color = mix(diverge(p), ext_color * sign(v+0.001), ext * 0.6);
            p *= mix(1.0, 0.4, ext);  // exterior contributes less opacity
        }

        float mag = abs(p);
        if (mag > 0.0002) {
            float a  = step_a * smoothstep(0.0002, 0.10, mag);
            acc.rgb += (1.0 - acc.a) * a * spectral_color;
            acc.a   += (1.0 - acc.a) * a;
            if (acc.a > 0.97) break;
        }
    }

    if (acc.a < 0.004) discard;
    // ACES filmic tone mapping + slight exposure boost
    vec3 mapped = aces(acc.rgb * 1.40);
    FragColor = vec4(mapped, acc.a);
}
"""

_GPU_RAY_FIELD_CS = """
#version 430 core
layout(local_size_x = 128) in;

struct Tri {
    vec4 v0;
    vec4 e1;
    vec4 e2;
    vec4 normal;
    vec4 mat_in;
    vec4 mat_out;
};

struct Node {
    vec4 lo_left;
    vec4 hi_right;
    vec4 start_count;
};

layout(std430, binding = 0) readonly buffer TriBuf { Tri tris[]; };
layout(std430, binding = 1) readonly buffer NodeBuf { Node nodes[]; };
layout(std430, binding = 2) readonly buffer TriIdBuf { int tri_ids[]; };
layout(std430, binding = 3) buffer SegmentBuf { vec4 segs[]; };
layout(std430, binding = 4) buffer CounterBuf {
    uint seg_count;
    uint ray_count;
    uint hit_count;
    uint record_count;
};
layout(r32ui, binding = 0) uniform uimage3D uBand0;
layout(r32ui, binding = 1) uniform uimage3D uBand1;
layout(r32ui, binding = 2) uniform uimage3D uBand2;
layout(r32ui, binding = 3) uniform uimage3D uBand3;

uniform int   uTriCount;
uniform int   uNodeCount;
uniform int   uSegmentCap;
uniform int   uSegmentStride;
// Batched dispatch: uBatchSize rays in this call, uBatchOffset = first global ray index.
// uTotalRaysPerSource controls the append_segment thinning ratio.
uniform int   uBatchSize;
uniform int   uBatchOffset;
uniform int   uTotalRaysPerSource;
uniform int   uMaxBounces;
uniform int   uSeed;
uniform vec3  uSrcPos;
uniform vec3  uSrcDir;
uniform vec3  uBoxMin;
uniform vec3  uBoxMax;
uniform ivec3 uDims;
// Per-source spectral distribution: normalised fractions summing to ~1.0
// x=band0(55-220Hz) y=band1(220-440Hz) z=band2(440-880Hz) w=band3(880Hz+)
uniform vec4  uSrcSpectrum;
// Beer-Lambert participating medium uniforms
uniform float uVolumeStepMeters;  // sample step size in metres (default 0.003)
uniform float uMediumScattering;  // scattering coefficient (default 1.0)
uniform float uMediumExtinction;  // extinction coefficient (default 0.5)
uniform vec4  uBandExtinction;    // per-band extinction scaling (default all 1.0)

const float EPS = 1e-7;
const float PI = 3.14159265358979323846;

uint hash_u(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rand01(inout uint s) {
    s = hash_u(s);
    return float(s & 0x00ffffffu) / 16777215.0;
}

vec3 basis_dir(float u, float v, vec3 axis) {
    float z = 1.0 - 2.0 * u;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float a = 2.0 * PI * v;
    vec3 d = vec3(r * cos(a), r * sin(a), z);
    vec3 w = normalize(axis);
    vec3 up = abs(w.z) < 0.9 ? vec3(0,0,1) : vec3(0,1,0);
    vec3 x = normalize(cross(up, w));
    vec3 y = cross(w, x);
    return normalize(x * d.x + y * d.y + w * d.z);
}

vec3 cosine_dir(float u, float v, vec3 normal) {
    float r = sqrt(max(u, 0.0));
    float a = 2.0 * PI * v;
    vec3 local = vec3(r * cos(a), r * sin(a), sqrt(max(0.0, 1.0 - u)));
    vec3 w = normalize(normal);
    vec3 up = abs(w.z) < 0.9 ? vec3(0,0,1) : vec3(0,1,0);
    vec3 x = normalize(cross(up, w));
    vec3 y = cross(w, x);
    return normalize(x * local.x + y * local.y + w * local.z);
}

bool hit_tri(vec3 ro, vec3 rd, Tri t, out float hit_t) {
    vec3 h = cross(rd, t.e2.xyz);
    float a = dot(t.e1.xyz, h);
    if (abs(a) < 1e-8) return false;
    float f = 1.0 / a;
    vec3 s = ro - t.v0.xyz;
    float u = f * dot(s, h);
    if (u < -1e-5 || u > 1.00001) return false;
    vec3 q = cross(s, t.e1.xyz);
    float v = f * dot(rd, q);
    if (v < -1e-5 || u + v > 1.00001) return false;
    float tt = f * dot(t.e2.xyz, q);
    if (tt <= 1e-6) return false;
    hit_t = tt;
    return true;
}

bool slab_axis(float ro, float rd, float lo, float hi, inout float near_t, inout float far_t) {
    if (abs(rd) < 1e-9) {
        return ro >= lo && ro <= hi;
    }
    float inv_rd = 1.0 / rd;
    float a = (lo - ro) * inv_rd;
    float b = (hi - ro) * inv_rd;
    near_t = max(near_t, min(a, b));
    far_t = min(far_t, max(a, b));
    return near_t <= far_t;
}

bool hit_aabb(vec3 ro, vec3 rd, vec3 lo, vec3 hi, float best_t) {
    float near_t = EPS;
    float far_t = best_t;
    if (!slab_axis(ro.x, rd.x, lo.x, hi.x, near_t, far_t)) return false;
    if (!slab_axis(ro.y, rd.y, lo.y, hi.y, near_t, far_t)) return false;
    if (!slab_axis(ro.z, rd.z, lo.z, hi.z, near_t, far_t)) return false;
    return near_t <= far_t;
}

int nearest_hit(vec3 ro, vec3 rd, out float best) {
    best = 1e30;
    int best_tri = -1;
    int stack[96];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int ni = stack[--sp];
        if (ni < 0 || ni >= uNodeCount) continue;   // -1 is the leaf sentinel
        Node node = nodes[ni];
        if (!hit_aabb(ro, rd, node.lo_left.xyz, node.hi_right.xyz, best)) {
            continue;
        }

        int left = int(node.lo_left.w);
        int right = int(node.hi_right.w);
        int start = int(node.start_count.x);
        int count = int(node.start_count.y);
        if (left < 0) {
            for (int k = 0; k < count; ++k) {
                int ti = tri_ids[start + k];
                float ht;
                if (hit_tri(ro, rd, tris[ti], ht) && ht < best) {
                    best = ht;
                    best_tri = ti;
                }
            }
        } else {
            if (sp < 94) {
                stack[sp++] = left;
                stack[sp++] = right;
            }
        }
    }
    return best_tri;
}

void splat_spectral(vec3 p, vec4 spectrum) {
    vec3 uvw = (p - uBoxMin) / max(uBoxMax - uBoxMin, vec3(1e-6));
    ivec3 q = ivec3(floor(uvw * vec3(uDims)));
    if (any(lessThan(q, ivec3(0))) || any(greaterThanEqual(q, uDims))) return;
    uint u0 = uint(clamp(spectrum.x * 65535.0, 0.0, 1e6));
    uint u1 = uint(clamp(spectrum.y * 65535.0, 0.0, 1e6));
    uint u2 = uint(clamp(spectrum.z * 65535.0, 0.0, 1e6));
    uint u3 = uint(clamp(spectrum.w * 65535.0, 0.0, 1e6));
    if (u0 > 0u) imageAtomicAdd(uBand0, q, u0);
    if (u1 > 0u) imageAtomicAdd(uBand1, q, u1);
    if (u2 > 0u) imageAtomicAdd(uBand2, q, u2);
    if (u3 > 0u) imageAtomicAdd(uBand3, q, u3);
}

void splat_segment_spectral(vec3 a, vec3 b, vec4 spectrum) {
    float len = length(b - a);
    int steps = max(1, int(len / uVolumeStepMeters));
    vec3 dir = (b - a) / max(len, 1e-6);
    for (int i = 0; i <= steps; ++i) {
        float s    = float(i) / float(steps);
        float dist = s * len;
        vec3  p    = a + dir * dist;
        // Beer-Lambert transmittance: per-band extinction along path
        vec4 transmit = exp(-uMediumExtinction * dist * uBandExtinction);
        vec4 scatter  = spectrum * transmit * uMediumScattering * uVolumeStepMeters;
        splat_spectral(p, scatter);
    }
}

void append_segment(vec3 p0, vec3 p1, float energy) {
    if (uSegmentCap <= 0) return;
    uint idx = atomicAdd(seg_count, 1u);
    if (idx >= uint(uSegmentCap)) return;
    int base = int(idx) * uSegmentStride;
    float a = clamp(energy, 0.12, 0.9);
    // Interleaved vertex layout: vertex 0: position, color; vertex 1: position, color
    segs[base + 0] = vec4(p0, 1.0);
    segs[base + 1] = vec4(1.0, 1.0, 1.0, a);
    segs[base + 2] = vec4(p1, 1.0);
    segs[base + 3] = vec4(1.0, 1.0, 1.0, a);
}

void main() {
    uint groups_x = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * groups_x;
    if (gid >= uint(uBatchSize)) return;
    // Global ray index stays unique across batches for RNG independence.
    uint ray_id = gid + uint(uBatchOffset);
    atomicAdd(ray_count, 1u);
    uint rng = hash_u(ray_id ^ uint(uSeed));
    vec3 ro = uSrcPos;
    vec3 rd = basis_dir(rand01(rng), rand01(rng), uSrcDir);
    // Each ray carries spectral energy per band (initialised from source spectrum).
    vec4 spectrum = uSrcSpectrum;

    for (int bounce = 0; bounce < uMaxBounces; ++bounce) {
        float best = 1e30;
        int hit = nearest_hit(ro, rd, best);
        if (hit < 0) break;
        atomicAdd(hit_count, 1u);
        vec3 hp = ro + rd * best;
        splat_segment_spectral(ro, hp, spectrum);
        // Thin the segment overlay proportionally to total ray count.
        uint thin = max(1u, uint(uTotalRaysPerSource) / 16384u);
        if ((ray_id % thin) == 0u) {
            atomicAdd(record_count, 1u);
            append_segment(ro, hp, dot(spectrum, vec4(0.25)));
        }
        Tri tri = tris[hit];
        vec3 geom_n = normalize(tri.normal.xyz);
        bool interior_face = dot(rd, geom_n) > 0.0;
        vec4 mat = interior_face ? tri.mat_in : tri.mat_out;
        float reflectivity = clamp(mat.x, 0.02, 0.98);
        float diffusion    = clamp(mat.y, 0.0, 1.0);
        float absorption   = clamp(mat.z, 0.0, 2.0);
        vec3 n = interior_face ? -geom_n : geom_n;
        if (rand01(rng) < diffusion) {
            rd = cosine_dir(rand01(rng), rand01(rng), n);
        } else {
            rd = normalize(reflect(rd, n));
            if (dot(rd, n) < 0.0) {
                rd = cosine_dir(rand01(rng), rand01(rng), n);
            }
        }
        ro = hp + rd * 1e-5;
        // Frequency-dependent absorption: wood attenuates high bands more than low.
        // band_decay[b] = exp(-absorption * distance * freq_factor[b])
        // freq_factors: 0.70 (sub-220 Hz) → 1.30 (880 Hz+)
        vec4 band_decay = exp(-absorption * best * vec4(0.70, 0.85, 1.00, 1.30));
        float dist_falloff = 1.0 / (1.0 + 0.08 * best);
        spectrum *= reflectivity * band_decay * dist_falloff;
        if (dot(spectrum, vec4(1.0)) < 0.002) break;
    }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# HUD 2-D shaders  (pixel-space overlay, no depth)
# ─────────────────────────────────────────────────────────────────────────────

_HUD2D_VS = """
#version 330 core
layout(location = 0) in vec2 aPos;
uniform vec2 uRes;
void main() {
    vec2 ndc = aPos / uRes * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

_HUD2D_FS = """
#version 330 core
uniform vec4 uColor;
out vec4 FragColor;
void main() { FragColor = uColor; }
"""

_HUD2D_TEX_VS = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
uniform vec2 uRes;
out vec2 vUV;
void main() {
    vec2 ndc = aPos / uRes * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUV = aUV;
}
"""

_HUD2D_TEX_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uTex;
uniform vec4 uColor;
out vec4 FragColor;
void main() {
    vec4 t = texture(uTex, vUV);
    float bright = (t.r + t.g + t.b) * 0.333;
    FragColor = vec4(uColor.rgb, bright * uColor.a);
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# GL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compile(src: str, kind: int) -> int:
    sh = glCreateShader(kind)
    glShaderSource(sh, src)
    glCompileShader(sh)
    ok = glGetShaderiv(sh, GL_COMPILE_STATUS)
    log = glGetShaderInfoLog(sh)
    if log:
        tag = "VS" if kind == GL_VERTEX_SHADER else ("CS" if kind == GL_COMPUTE_SHADER else "FS")
        print(f"[{tag} compile{'  OK' if ok else ' FAIL'}] {log.decode()}")
    elif not ok:
        tag = "VS" if kind == GL_VERTEX_SHADER else ("CS" if kind == GL_COMPUTE_SHADER else "FS")
        print(f"[{tag} compile FAIL] (no log)")
    return sh

def _prog(*shader_pairs) -> int:
    """Build program from (source, kind) pairs."""
    p = glCreateProgram()
    handles = []
    for src, kind in shader_pairs:
        s = _compile(src, kind)
        glAttachShader(p, s)
        handles.append(s)
    glLinkProgram(p)
    ok = glGetProgramiv(p, GL_LINK_STATUS)
    log = glGetProgramInfoLog(p)
    if log:
        print(f"[link{'  OK' if ok else ' FAIL'}] {log.decode()}")
    elif not ok:
        print(f"[link FAIL] (no log)")
    for s in handles:
        glDeleteShader(s)
    return p

def _build_gpu_bvh(tris: np.ndarray, leaf_size: int = 4):
    tri_min = tris.min(axis=1)
    tri_max = tris.max(axis=1)
    cent = (tri_min + tri_max) * 0.5
    ids = np.arange(len(tris), dtype=np.int32)
    nodes = []

    def rec(begin: int, end: int) -> int:
        idx = len(nodes)
        nodes.append(None)
        sub = ids[begin:end]
        lo = tri_min[sub].min(axis=0)
        hi = tri_max[sub].max(axis=0)
        count = end - begin
        if count <= leaf_size:
            nodes[idx] = (lo, hi, -1, -1, begin, count)
            return idx
        c = cent[sub]
        axis = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
        order = np.argsort(c[:, axis], kind="mergesort")
        ids[begin:end] = sub[order]
        mid = begin + count // 2
        left = rec(begin, mid)
        right = rec(mid, end)
        nodes[idx] = (lo, hi, left, right, 0, 0)
        return idx

    if len(tris):
        rec(0, len(tris))

    packed = np.zeros((len(nodes), 12), np.float32)
    for i, (lo, hi, left, right, start, count) in enumerate(nodes):
        packed[i, 0:3] = lo
        packed[i, 3] = float(left)
        packed[i, 4:7] = hi
        packed[i, 7] = float(right)
        packed[i, 8] = float(start)
        packed[i, 9] = float(count)
    return np.ascontiguousarray(packed), np.ascontiguousarray(ids, dtype=np.int32)


def _compute_spectral_emission_map(
    ce,
    n_steps: int = 8192,
    band_edges=SPEC_BAND_EDGES,
) -> np.ndarray:
    """
    Run physics at single-sample resolution for n_steps, FFT the plate
    displacement time series, and return a (N_BANDS, Nh, Nw) float32 array
    of per-band acoustic power.  Physics state is left advanced; caller must
    reset and reschedule the excitation afterward.
    """
    test  = ce.get_plate_displacement()
    shape = test.shape                    # (Nh, Nw) — axis order as returned by C
    frames = np.empty((n_steps, *shape), np.float32)
    for t in range(n_steps):
        ce.step(1)
        frames[t] = ce.get_plate_displacement()

    spectrum = np.fft.rfft(frames, axis=0)                           # (bins, Nh, Nw)
    freqs    = np.fft.rfftfreq(n_steps, d=1.0 / SAMPLE_RATE)
    power    = (np.abs(spectrum) ** 2).astype(np.float32)

    N_BANDS  = len(band_edges) - 1
    band_maps = np.zeros((N_BANDS, *shape), np.float32)
    for b in range(N_BANDS):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        if mask.any():
            band_maps[b] = power[mask].sum(axis=0)

    idx_440 = int(np.argmin(np.abs(freqs - 440.0)))
    print(f"  [spectral_map] 440 Hz → bin {idx_440} ({freqs[idx_440]:.1f} Hz)  "
          f"band edges: {band_edges}  n_steps={n_steps}", flush=True)
    return band_maps


def _string_spectral_weights(
    fundamental_hz: float,
    band_edges=SPEC_BAND_EDGES,
    n_harmonics: int = 12,
) -> np.ndarray:
    """
    Compute normalised per-band energy fractions for a string with the given
    fundamental, assuming 1/h amplitude decay for harmonic h.
    """
    N_BANDS = len(band_edges) - 1
    w = np.zeros(N_BANDS, np.float32)
    for h in range(1, n_harmonics + 1):
        f = fundamental_hz * h
        for b in range(N_BANDS):
            if band_edges[b] <= f < band_edges[b + 1]:
                w[b] += 1.0 / h
                break
    total = w.sum()
    return (w / total).astype(np.float32) if total > 0 else w


def _sample_soundboard_sources(
    band_maps: np.ndarray,
    plate_active: np.ndarray,
    outline: np.ndarray,
    body_h: float,
    total_rays: int,
    stride: int = 4,
) -> list:
    """
    Sample the active soundboard area at grid stride, weighting each point by
    its total spectral power.  Returns list of (pos_f32, dir_f32, n_rays, spec_f32)
    where spec_f32 is a (4,) array of normalised per-band fractions.
    Sorted strongest-first.
    """
    N_BANDS, Nx, Ny = band_maps.shape
    min_xy = outline.min(axis=0).astype(np.float32)
    max_xy = outline.max(axis=0).astype(np.float32)
    xs = np.linspace(float(min_xy[0]), float(max_xy[0]), Nx)
    ys = np.linspace(float(min_xy[1]), float(max_xy[1]), Ny)
    _z  = float(body_h) - 0.001
    _dn = np.array([0.0, 0.0, -1.0], np.float32)

    pts, weights, spectra = [], [], []
    for ix in range(0, Nx, stride):
        for iy in range(0, Ny, stride):
            if not plate_active[ix, iy]:
                continue
            spec = band_maps[:, ix, iy]
            total_e = float(spec.sum())
            if total_e <= 0.0:
                continue
            pts.append(np.array([xs[ix], ys[iy], _z], np.float32))
            weights.append(total_e)
            spectra.append((spec / total_e).astype(np.float32))

    if not pts:
        raise RuntimeError(
            "_sample_soundboard_sources: no active soundboard points with nonzero energy"
        )

    w_arr = np.array(weights, np.float32)
    w_arr /= w_arr.sum()
    n_rays_each = np.maximum(1, (w_arr * total_rays).astype(int))

    order = np.argsort(n_rays_each)[::-1]
    return [(pts[k], _dn.copy(), int(n_rays_each[k]), spectra[k]) for k in order]


def _string_emission_sources(
    str_paths: list,
    body_h: float,
    total_rays_per_string: int = 4096,
    band_edges=SPEC_BAND_EDGES,
    fundamentals=STRING_FUNDAMENTALS_HZ,
    active_strings: Optional[set[int]] = None,
) -> list:
    """
    Emit from each string's physical 3-D positions with a spectrum derived from
    the string's harmonic series.  Direction: straight down into the cavity.
    Returns list of (pos_f32, dir_f32, n_rays, spec_f32).
    """
    sources = []
    _dn = np.array([0.0, 0.0, -1.0], np.float32)
    for si, path in enumerate(str_paths):
        if active_strings is not None and si not in active_strings:
            continue
        if si >= len(fundamentals):
            break
        spec = _string_spectral_weights(fundamentals[si], band_edges)
        path_arr = np.asarray(path, np.float32)
        n_pts = max(1, len(path_arr))
        rays_per_pt = max(1, total_rays_per_string // n_pts)
        for pt in path_arr:
            # Emit from the string's world position downward into the cavity
            pos = pt.copy()
            sources.append((pos, _dn.copy(), rays_per_pt, spec.copy()))
    return sources


def _instant_soundboard_sources(
    disp: np.ndarray,
    plate_active: np.ndarray,
    outline: np.ndarray,
    body_h: float,
    total_rays: int,
    stride: int = 4,
) -> list:
    """Current-frame plate sources for dynamic ray-field refresh."""
    Nx, Ny = disp.shape
    min_xy = outline.min(axis=0).astype(np.float32)
    max_xy = outline.max(axis=0).astype(np.float32)
    xs = np.linspace(float(min_xy[0]), float(max_xy[0]), Nx)
    ys = np.linspace(float(min_xy[1]), float(max_xy[1]), Ny)
    energy = np.abs(disp).astype(np.float32)
    peak = float(energy.max())
    if peak <= 1e-12:
        return []
    spec = np.array([0.22, 0.34, 0.30, 0.14], np.float32)
    pts, weights = [], []
    for ix in range(0, Nx, stride):
        for iy in range(0, Ny, stride):
            if not plate_active[ix, iy]:
                continue
            w = float(energy[ix, iy])
            if w <= peak * 0.01:
                continue
            pts.append(np.array([xs[ix], ys[iy], float(body_h) - 0.001], np.float32))
            weights.append(w)
    if not pts:
        return []
    w_arr = np.asarray(weights, np.float32)
    w_arr /= max(float(w_arr.sum()), 1e-12)
    n_each = np.maximum(1, (w_arr * int(total_rays)).astype(np.int32))
    dn = np.array([0.0, 0.0, -1.0], np.float32)
    order = np.argsort(n_each)[::-1]
    return [(pts[k], dn.copy(), int(n_each[k]), spec.copy()) for k in order]


def _stage_light_sources(total_rays: int, n_emitters: int = 9) -> list:
    """Diffuse area source above/front of the stage in world coordinates."""
    total_rays = max(1, int(total_rays))
    n_emitters = max(1, int(n_emitters))
    cols = int(math.ceil(math.sqrt(n_emitters)))
    rows = int(math.ceil(n_emitters / cols))
    lx0, lx1 = -0.65, 0.65
    ly = STAGE_D_M * 0.28
    lz = STAGE_H_M * 0.82
    xs = np.linspace(lx0, lx1, cols, dtype=np.float32)
    zs = np.linspace(lz - 0.10, lz + 0.10, rows, dtype=np.float32)
    src = []
    warm_white = np.array([0.30, 0.32, 0.23, 0.15], np.float32)
    axis = np.array([0.0, -0.42, -0.91], np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    rays_each = max(1, total_rays // n_emitters)
    count = 0
    for z in zs:
        for x in xs:
            if count >= n_emitters:
                break
            src.append((np.array([float(x), float(ly), float(z)], np.float32),
                        axis.copy(), rays_each, warm_white.copy()))
            count += 1
    remainder = total_rays - rays_each * len(src)
    if remainder > 0 and src:
        p, d, n, s = src[0]
        src[0] = (p, d, n + remainder, s)
    return src


def _stage_light_bounds() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([-STAGE_W_M * 0.55, -STAGE_D_M * 0.56, 0.0], np.float32),
        np.array([ STAGE_W_M * 0.55,  STAGE_D_M * 0.52, STAGE_H_M], np.float32),
    )


def _normalise_materials(materials: np.ndarray, n_tris: int) -> np.ndarray:
    mat = np.asarray(materials, dtype=np.float32)
    if mat.ndim != 2 or len(mat) != n_tris:
        mat = np.zeros((n_tris, 0), dtype=np.float32)
    if mat.shape[1] < 3:
        mat_in = np.tile(np.array([0.58, 0.62, 0.28], dtype=np.float32), (n_tris, 1))
    else:
        mat_in = mat[:, 0:3]
    if mat.shape[1] < 6:
        mat_out = np.tile(np.array([0.88, 0.10, 0.055], dtype=np.float32), (n_tris, 1))
    else:
        mat_out = mat[:, 3:6]
    return np.ascontiguousarray(np.column_stack([mat_in, mat_out]), dtype=np.float32)


def _stage_light_cache_key(scene, outline, body_h, total_rays: int, emitters: int,
                           max_bounces: int, dims=STAGE_LIGHT_DIMS) -> str:
    h = hashlib.sha256()
    h.update(b"stage-light-v2")
    h.update(np.asarray(dims, np.int32).tobytes())
    h.update(np.asarray([total_rays, emitters, max_bounces], np.int64).tobytes())
    h.update(np.asarray(_stage_light_bounds()[0], np.float32).tobytes())
    h.update(np.asarray(_stage_light_bounds()[1], np.float32).tobytes())
    src = _stage_light_sources(total_rays, emitters)
    for pos, direction, n_rays, spec in src:
        h.update(np.asarray(pos, np.float32).tobytes())
        h.update(np.asarray(direction, np.float32).tobytes())
        h.update(np.asarray([n_rays], np.int64).tobytes())
        h.update(np.asarray(spec, np.float32).tobytes())
    if scene is not None:
        if _extract_scene_geometry_materials_fn is not None:
            verts, normals, mats = _extract_scene_geometry_materials_fn(scene)
        else:
            verts, normals = _extract_scene_geometry_fn(scene)
            mats = np.tile(np.array([0.58, 0.62, 0.28, 0.88, 0.10, 0.055],
                                    dtype=np.float32), (len(verts), 1))
        gm, _ = _guitar_model_matrix(outline)
        M = np.asarray(gm, np.float32)
        vf = np.asarray(verts, np.float32).reshape(-1, 3)
        vf_h = np.column_stack([vf, np.ones(len(vf), dtype=np.float32)])
        verts_w = (vf_h @ M.T)[:, :3].astype(np.float32)
        nrm = np.asarray(normals, np.float32)
        mats6 = _normalise_materials(mats, len(verts_w) // 3)
        h.update(np.ascontiguousarray(verts_w).tobytes())
        h.update(np.ascontiguousarray(nrm).tobytes())
        h.update(mats6.tobytes())
    st_v, st_n, st_m = _stage_mesh(outline, body_h)
    h.update(np.ascontiguousarray(st_v).tobytes())
    h.update(np.ascontiguousarray(st_n).tobytes())
    h.update(_normalise_materials(st_m, len(st_n)).tobytes())
    return h.hexdigest()[:24]


def _stage_light_cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"stage_light_{key}.npz")

def _gpu_ray_field(scene, outline, body_h, sources, max_bounces,
                   dims=GPU_RAY_FIELD_DIMS, segment_cap=GPU_RAY_SEGMENT_CAP,
                   dispatch_batch=GPU_DISPATCH_BATCH, include_stage=True,
                   model_matrix: Optional[np.ndarray] = None,
                   bounds: Optional[tuple[np.ndarray, np.ndarray]] = None):
    _ray_diag_update(
        "gpu_ray_field:start",
        n_sources=len(sources) if sources is not None else 0,
        max_bounces=int(max_bounces),
        dims=tuple(int(v) for v in dims),
        segment_cap=int(segment_cap),
        dispatch_batch=int(dispatch_batch),
    )
    if scene is None or _extract_scene_geometry_fn is None:
        return None, None, None, None, 0, None, 0
    if _extract_scene_geometry_materials_fn is not None:
        verts_flat, normals, materials = _extract_scene_geometry_materials_fn(scene)
    else:
        verts_flat, normals = _extract_scene_geometry_fn(scene)
        materials = np.tile(np.array([0.58, 0.62, 0.28, 0.88, 0.10, 0.055],
                                     dtype=np.float32), (len(verts_flat), 1))
    verts_flat = np.asarray(verts_flat, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    if model_matrix is not None and len(verts_flat):
        M = np.asarray(model_matrix, dtype=np.float32)
        Rm = M[:3, :3]
        vf_h = np.column_stack([verts_flat, np.ones(len(verts_flat), dtype=np.float32)])
        verts_flat = (vf_h @ M.T)[:, :3].astype(np.float32)
        normals = (normals @ Rm.T)
        nl = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = (normals / np.maximum(nl, 1e-9)).astype(np.float32)
    if include_stage:
        st_v, st_n, st_m = _stage_mesh(outline, body_h)
        verts_flat = np.vstack([np.asarray(verts_flat, np.float32),
                                np.asarray(st_v, np.float32).reshape(-1, 3)])
        normals = np.vstack([np.asarray(normals, np.float32), st_n])
        materials = np.vstack([np.asarray(materials, np.float32), st_m])
    if len(verts_flat) == 0:
        return None, None, None, None, 0, None, 0

    tris = verts_flat.reshape(-1, 3, 3).astype(np.float32)
    nrm = normals.astype(np.float32)
    mat6 = _normalise_materials(materials, len(tris))
    mat_in = mat6[:, 0:3]
    mat_out = mat6[:, 3:6]
    packed = np.zeros((len(tris), 24), np.float32)
    packed[:, 0:3] = tris[:, 0, :]
    packed[:, 4:7] = tris[:, 1, :] - tris[:, 0, :]
    packed[:, 8:11] = tris[:, 2, :] - tris[:, 0, :]
    packed[:, 12:15] = nrm
    packed[:, 16:19] = mat_in
    packed[:, 20:23] = mat_out
    packed = np.ascontiguousarray(packed)
    bvh_nodes, bvh_ids = _build_gpu_bvh(tris)
    _ray_diag_update(
        "gpu_ray_field:geometry",
        n_sources=len(sources),
        total_source_rays=sum(int(n) for _, _, n, _ in sources),
        n_tris=len(tris),
        n_bvh_nodes=len(bvh_nodes),
        packed_bytes=int(packed.nbytes),
        bvh_bytes=int(bvh_nodes.nbytes + bvh_ids.nbytes),
        dims=tuple(int(v) for v in dims),
        segment_cap=int(segment_cap),
    )

    if bounds is not None:
        bmin = np.asarray(bounds[0], dtype=np.float32)
        bmax = np.asarray(bounds[1], dtype=np.float32)
    else:
        min_xy = outline.min(axis=0).astype(np.float32)
        max_xy = outline.max(axis=0).astype(np.float32)
        bmin = np.array([min_xy[0] - 0.012, min_xy[1] - 0.012, -0.002], np.float32)
        bmax = np.array([max_xy[0] + 0.012, max_xy[1] + 0.012, body_h + 0.004], np.float32)

    # ── Create 4 spectral band accumulation textures (one per frequency band) ─
    zero = np.array([0], dtype=np.uint32)
    tex_bands = list(glGenTextures(4))
    for tex in tex_bands:
        glBindTexture(GL_TEXTURE_3D, tex)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, dims[0], dims[1], dims[2], 0,
                     GL_RED_INTEGER, GL_UNSIGNED_INT, None)
        for param, val in [(GL_TEXTURE_MIN_FILTER, GL_NEAREST),
                           (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                           (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                           (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                           (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)]:
            glTexParameteri(GL_TEXTURE_3D, param, val)
        glClearTexImage(tex, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, zero)
    glBindTexture(GL_TEXTURE_3D, 0)

    seg_cap = max(0, int(segment_cap))
    seg_stride = 4
    seg_bytes = max(1, seg_cap * seg_stride * 4 * 4)
    counter = np.zeros(4, dtype=np.uint32)

    ssbo = glGenBuffers(5)
    _ray_diag_update(
        "gpu_ray_field:alloc_buffers",
        n_sources=len(sources),
        total_source_rays=sum(int(n) for _, _, n, _ in sources),
        n_tris=len(tris),
        n_bvh_nodes=len(bvh_nodes),
        tex_bands=tuple(int(t) for t in tex_bands),
        ssbo=tuple(int(b) for b in ssbo),
        seg_bytes=int(seg_bytes),
    )
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[0])
    glBufferData(GL_SHADER_STORAGE_BUFFER, packed.nbytes, packed, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0])
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[1])
    glBufferData(GL_SHADER_STORAGE_BUFFER, bvh_nodes.nbytes, bvh_nodes, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1])
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2])
    glBufferData(GL_SHADER_STORAGE_BUFFER, bvh_ids.nbytes, bvh_ids, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo[2])
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[3])
    glBufferData(GL_SHADER_STORAGE_BUFFER, seg_bytes, None, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo[3])
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[4])
    glBufferData(GL_SHADER_STORAGE_BUFFER, counter.nbytes, counter, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssbo[4])

    prog = _prog((_GPU_RAY_FIELD_CS, GL_COMPUTE_SHADER))
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        print("  [gpu_ray_field] compute shader did not link; ray volume disabled", flush=True)
        glDeleteProgram(prog)
        for tex in tex_bands:
            glDeleteTextures([tex])
        for buf in ssbo:
            glDeleteBuffers(1, [buf])
        return None, None, None, None, 0, None, 0
    glUseProgram(prog)
    # imageAtomicAdd on all 4 band textures — must be GL_READ_WRITE
    for band_idx, tex in enumerate(tex_bands):
        glBindImageTexture(band_idx, tex, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI)
    glUniform1i(glGetUniformLocation(prog, b'uTriCount'), len(tris))
    glUniform1i(glGetUniformLocation(prog, b'uNodeCount'), len(bvh_nodes))
    glUniform1i(glGetUniformLocation(prog, b'uSegmentCap'), seg_cap)
    glUniform1i(glGetUniformLocation(prog, b'uSegmentStride'), seg_stride)
    glUniform1i(glGetUniformLocation(prog, b'uMaxBounces'), int(max_bounces))
    glUniform3f(glGetUniformLocation(prog, b'uBoxMin'), *bmin)
    glUniform3f(glGetUniformLocation(prog, b'uBoxMax'), *bmax)
    glUniform3i(glGetUniformLocation(prog, b'uDims'), int(dims[0]), int(dims[1]), int(dims[2]))
    # Beer-Lambert participating medium — tuned defaults for a guitar-scale scene
    glUniform1f(glGetUniformLocation(prog, b'uVolumeStepMeters'), 0.003)
    glUniform1f(glGetUniformLocation(prog, b'uMediumScattering'),  1.0)
    glUniform1f(glGetUniformLocation(prog, b'uMediumExtinction'),  0.5)
    glUniform4f(glGetUniformLocation(prog, b'uBandExtinction'),    1.0, 1.15, 1.30, 1.50)

    loc_seed        = glGetUniformLocation(prog, b'uSeed')
    loc_src_pos     = glGetUniformLocation(prog, b'uSrcPos')
    loc_src_dir     = glGetUniformLocation(prog, b'uSrcDir')
    loc_batch_size  = glGetUniformLocation(prog, b'uBatchSize')
    loc_batch_off   = glGetUniformLocation(prog, b'uBatchOffset')
    loc_total_rays  = glGetUniformLocation(prog, b'uTotalRaysPerSource')
    loc_src_spec    = glGetUniformLocation(prog, b'uSrcSpectrum')

    dispatch_batch = max(128, int(dispatch_batch))
    total_rays = sum(n for _, _, n, _ in sources)
    print(f"  [gpu_ray_field] {len(sources)} sources, "
          f"{total_rays} total rays, "
          f"dispatch_batch={dispatch_batch}, "
          f"box Z=[{float(bmin[2]):.4f}, {float(bmax[2]):.4f}]",
          flush=True)

    rays_done = 0
    next_report = max(dispatch_batch, total_rays // 20) if total_rays > 0 else dispatch_batch
    for si, (sp, sd, n_rays, spectrum) in enumerate(sources):
        _ray_diag_update(
            "gpu_ray_field:dispatch_source",
            source_index=int(si),
            source_rays=int(n_rays),
            rays_done=int(rays_done),
            total_rays=int(total_rays),
            src_pos=tuple(float(v) for v in np.asarray(sp).ravel()[:3]),
            src_dir=tuple(float(v) for v in np.asarray(sd).ravel()[:3]),
        )
        glUniform1i(loc_seed,       1337 + si * 104729)
        glUniform1i(loc_total_rays, max(1, n_rays))
        glUniform3f(loc_src_pos, *sp)
        glUniform3f(loc_src_dir, *sd)
        glUniform4f(loc_src_spec, float(spectrum[0]), float(spectrum[1]),
                                  float(spectrum[2]), float(spectrum[3]))

        offset = 0
        while offset < n_rays:
            batch = min(dispatch_batch, n_rays - offset)
            glUniform1i(loc_batch_size, batch)
            glUniform1i(loc_batch_off, offset)
            n_groups = max(1, int(math.ceil(batch / 128.0)))
            gx = min(n_groups, 65535)
            gy = max(1, int(math.ceil(n_groups / gx)))
            glDispatchCompute(gx, gy, 1)
            # Barrier between batches: forces the GPU to drain each batch
            # before the next begins, preventing TDR timeout on 10 M+ rays.
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                            GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            offset += batch
            rays_done += batch
            if total_rays >= 1_000_000 and rays_done >= next_report:
                print(f"  [gpu_ray_field] streamed {rays_done}/{total_rays} rays",
                      flush=True)
                next_report += max(dispatch_batch, total_rays // 20)

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT)

    gl_err = glGetError()
    if gl_err != GL_NO_ERROR:
        print(f"  [gpu_ray_field] GL error after dispatch: 0x{gl_err:04x}", flush=True)

    # Read back shader diagnostics: [segments, rays, hits, selected-for-recording].
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[4])
    counter_raw = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4 * 4)
    counter_readback = np.frombuffer(counter_raw, dtype=np.uint32, count=4).copy()
    actual_seg_count = int(min(int(counter_readback[0]), seg_cap))
    rays_seen = int(counter_readback[1])
    hit_count = int(counter_readback[2])
    record_count = int(counter_readback[3])
    _ray_diag_update(
        "gpu_ray_field:readback",
        total_rays=int(total_rays),
        rays_seen=int(rays_seen),
        hit_count=int(hit_count),
        record_count=int(record_count),
        segment_counter=int(counter_readback[0]),
        actual_seg_count=int(actual_seg_count),
        segment_cap=int(seg_cap),
        ssbo=tuple(int(b) for b in ssbo),
        tex_bands=tuple(int(t) for t in tex_bands),
    )
    if total_rays > 0 and rays_seen > int(total_rays):
        raise RuntimeError(
            f"GPU ray dispatch counter overflow: reported {rays_seen} "
            f"rays with max {int(total_rays)}")
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    glUseProgram(0)
    if actual_seg_count == 0 and seg_cap > 0:
        loc_nc = glGetUniformLocation(prog, b'uNodeCount')
        print(f"  [gpu_ray_field] 0 segs — locs: node_count={loc_nc} "
              f"batch_size={loc_batch_size} batch_off={loc_batch_off} "
              f"total_rays_loc={loc_total_rays} "
              f"sources={len(sources)}", flush=True)
        print(f"  [gpu_ray_field] counters: rays={rays_seen} "
              f"hits={hit_count} selected={record_count} "
              f"seg_counter={int(counter_readback[0])} cap={seg_cap}", flush=True)
        print(f"  [gpu_ray_field] geo: {len(tris)} tris {len(bvh_nodes)} BVH nodes  "
              f"box X[{float(bmin[0]):.3f},{float(bmax[0]):.3f}] "
              f"Y[{float(bmin[1]):.3f},{float(bmax[1]):.3f}] "
              f"Z[{float(bmin[2]):.3f},{float(bmax[2]):.3f}]", flush=True)
    print(f"  GPU ray field {dims[0]}x{dims[1]}x{dims[2]}, "
          f"{len(sources)} sources / {total_rays} total rays, {len(tris)} tris, "
          f"{len(bvh_nodes)} BVH nodes, {actual_seg_count} segments emitted", flush=True)
    for buf in ssbo[:3]:
        glDeleteBuffers(1, [buf])
    glDeleteProgram(prog)
    return tex_bands, bmin, bmax, ssbo[3], actual_seg_count, ssbo[4], int(total_rays)


def _read_uint_ray_textures(tex_bands: list[int], dims: tuple[int, int, int]) -> np.ndarray:
    arrays = []
    for tex in tex_bands:
        glBindTexture(GL_TEXTURE_3D, tex)
        raw = glGetTexImage(GL_TEXTURE_3D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT)
        arr = np.frombuffer(raw, dtype=np.uint32, count=int(np.prod(dims))).copy()
        arrays.append(arr.reshape((dims[2], dims[1], dims[0])))
    glBindTexture(GL_TEXTURE_3D, 0)
    return np.stack(arrays, axis=0)


def _upload_uint_ray_textures(data: np.ndarray) -> list[int]:
    tex_bands = list(glGenTextures(4))
    for tex, arr in zip(tex_bands, data):
        # npz stores as (z, y, x); OpenGL expects width, height, depth bytes.
        arr = np.ascontiguousarray(arr, dtype=np.uint32)
        depth, height, width = arr.shape
        glBindTexture(GL_TEXTURE_3D, tex)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, width, height, depth, 0,
                     GL_RED_INTEGER, GL_UNSIGNED_INT, arr)
        for param, val in [(GL_TEXTURE_MIN_FILTER, GL_NEAREST),
                           (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                           (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                           (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                           (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)]:
            glTexParameteri(GL_TEXTURE_3D, param, val)
    glBindTexture(GL_TEXTURE_3D, 0)
    return tex_bands


def _load_stage_light_cache(path: str):
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            data = z["bands"]
            bounds = (z["bounds_min"].astype(np.float32),
                      z["bounds_max"].astype(np.float32))
            total_rays = int(z["total_rays"])
        if data.shape[0] != 4:
            return None
        return _upload_uint_ray_textures(data), bounds, total_rays
    except Exception as exc:
        print(f"  [cache] stage light cache unreadable: {exc}", flush=True)
        return None


def _save_stage_light_cache(path: str, tex_bands: list[int], dims, bounds, total_rays: int) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = _read_uint_ray_textures(tex_bands, dims)
        np.savez_compressed(
            path,
            bands=data,
            bounds_min=np.asarray(bounds[0], np.float32),
            bounds_max=np.asarray(bounds[1], np.float32),
            total_rays=np.asarray(total_rays, np.int64),
        )
        print(f"  [cache] saved stage light field: {path}", flush=True)
    except Exception as exc:
        print(f"  [cache] could not save stage light field: {exc}", flush=True)

def _mvp(prog, mvp, mv=None, m=None):
    glUniformMatrix4fv(glGetUniformLocation(prog, b'uMVP'), 1, GL_TRUE, mvp)
    if mv is not None:
        loc = glGetUniformLocation(prog, b'uMV')
        if loc >= 0:
            glUniformMatrix4fv(loc, 1, GL_TRUE, mv)
    if m is not None:
        loc = glGetUniformLocation(prog, b'uM')
        if loc >= 0:
            glUniformMatrix4fv(loc, 1, GL_TRUE, m)

def _vao(data: np.ndarray, attribs: list, usage=GL_DYNAMIC_DRAW):
    """Create VAO+VBO. attribs: [(loc, n, stride, offset), ...]"""
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, usage)
    for loc, nc, stride, offset in attribs:
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, nc, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(offset))
    glBindVertexArray(0)
    return vao, vbo, len(data)


# ─────────────────────────────────────────────────────────────────────────────
# Orbit camera
# ─────────────────────────────────────────────────────────────────────────────

class Camera:
    def __init__(self, target=(0.0, 0.4, 0.65), dist=1.6, elev=14.0, az=22.0):
        self.target  = np.array(target, np.float64)
        self.dist    = dist
        self.elev    = elev
        self.az      = az
        self._auto   = 0.18

    def orbit(self, daz, delev):
        self.az   = (self.az + daz) % 360.0
        self.elev = float(np.clip(self.elev + delev, -75.0, 80.0))

    def zoom(self, d):
        self.dist = float(np.clip(self.dist + d, 0.12, 3.0))

    def tick(self):
        self.az = (self.az + self._auto) % 360.0

    @property
    def eye(self):
        az = math.radians(self.az);  el = math.radians(self.elev)
        return self.target + self.dist * np.array([
            math.cos(el)*math.cos(az),
            math.cos(el)*math.sin(az),
            math.sin(el),
        ])

    def _view(self):
        eye = self.eye
        up  = np.array([0., 0., 1.])
        if abs(np.dot(_norm(self.target - eye), up)) > 0.97:
            up = np.array([0., 1., 0.])
        return _lookat(eye, self.target, up)

    def mvp(self, aspect):
        P = _persp(math.radians(52.0), aspect, 0.005, 10.0)
        return (P @ self._view()).astype(np.float32)

    def mv(self):
        return self._view().astype(np.float32)

    def inv_mvp(self, aspect):
        return np.linalg.inv(self.mvp(aspect)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, win_w, win_h, outline, info, body_h,
                 bridge_pos, str_paths, ray_segs=None,
                 ray_field_bands=None, ray_field_bounds=None,
                 ray_field_total_rays=0,
                 baseline_light_bands=None, baseline_light_bounds=None,
                 baseline_light_total_rays=0,
                 gpu_seg_vbo=None, gpu_seg_cap=0,
                 max_cached_frames=MAX_FRAMES,
                 plate_theta=PLATE_THETA_SEGS,
                 plate_radial=PLATE_RADIAL_SEGS,
                 active_fret: int = 0,
                 fretless: bool = False,
                 show_pickup: bool = True,
                 show_mic: bool = True):
        self.win_w   = win_w
        self.win_h   = win_h
        self.outline = outline
        self.info    = info
        self.body_h  = body_h
        self.bridge  = bridge_pos
        self.paths   = str_paths
        self.n_str   = len(str_paths)
        self._plate_mode = 2  # 0=color, 1=height, 2=both
        self._plate_theta = int(plate_theta)
        self._plate_radial = int(plate_radial)
        self.active_fret = int(active_fret)
        self.fretless = bool(fretless)
        self.show_pickup = bool(show_pickup)
        self.show_mic = bool(show_mic)

        self._frames: collections.deque = collections.deque(
            maxlen=max(1, int(max_cached_frames)))
        self._cursor  = 0
        self._paused  = False
        self._layers  = [LAYER_OPAQUE, LAYER_ALPHA, LAYER_HIDDEN,
                         LAYER_HIDDEN, LAYER_HIDDEN, LAYER_ALPHA, LAYER_ALPHA,
                         LAYER_ALPHA]   # 7=illum (ray field volume + spotlight cone)
        self.cam = Camera()

        # Guitar model matrix: guitar-frame → world-frame (upright on stand)
        self._guitar_M, self._guitar_Minv = _guitar_model_matrix(outline)

        dx     = float(info.get('dx', DX))
        gx_min = float(info['gx_min'])
        gy_min = float(info['gy_min'])
        gz_min = float(info.get('gz_min', -PAD_CELLS * dx))
        Nx, Ny, Nz = info['Nx'], info['Ny'], info['Nz']
        self._bmin = np.array([gx_min, gy_min, gz_min], np.float32)
        self._bmax = np.array([gx_min + Nx*dx, gy_min + Ny*dx, gz_min + Nz*dx],
                               np.float32)

        self._plate_dyn  = None
        self._ray_segs   = ray_segs
        self._ray_field_bands = ray_field_bands
        self._ray_field_bounds = ray_field_bounds
        self._ray_field_total_rays = int(ray_field_total_rays or 0)
        self._baseline_light_bands = baseline_light_bands
        self._baseline_light_bounds = baseline_light_bounds
        self._baseline_light_total_rays = int(baseline_light_total_rays or 0)
        self._gpu_seg_vbo = gpu_seg_vbo
        self._gpu_seg_cap = int(gpu_seg_cap or 0)
        self._ray_exposure = 1.0
        self._ray_gamma = float(GPU_RAY_FIELD_GAMMA)
        self._ray_lighting: Optional[RayLightingState] = None   # set after init via set_ray_lighting()
        self._init_gl()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def set_ray_lighting(self, meta: dict) -> None:
        """Store per-triangle ray-traced illumination from trace_cavity_scene meta dict.

        Called after the C ray tracer completes (before or after init_gl).
        The data is used by _p_ray_surface when the ILLUM layer is active.
        """
        if 'surface_flux' not in meta:
            return
        self._ray_lighting = RayLightingState(
            surface_flux     = meta['surface_flux'],
            surface_direct   = meta['surface_direct'],
            surface_indirect = meta['surface_indirect'],
            surface_rgb      = meta['surface_rgb'],
            surface_scalar   = meta['surface_scalar'],
        )

    def _init_gl(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        self._p_body  = _prog((_BODY_VS, GL_VERTEX_SHADER),
                               (_BODY_FS, GL_FRAGMENT_SHADER))
        self._p_ray_surface = _prog((_RAY_SURFACE_VS, GL_VERTEX_SHADER),
                                     (_RAY_SURFACE_FS, GL_FRAGMENT_SHADER))
        self._p_plate = _prog((_PLATE_VS, GL_VERTEX_SHADER),
                               (_PLATE_FS, GL_FRAGMENT_SHADER))
        self._p_line  = _prog((_LINE_VS, GL_VERTEX_SHADER),
                               (_LINE_FS, GL_FRAGMENT_SHADER))
        self._p_vcol  = _prog((_VCOL_VS, GL_VERTEX_SHADER),
                               (_VCOL_FS, GL_FRAGMENT_SHADER))
        self._p_march = _prog((_MARCH_VS, GL_VERTEX_SHADER),
                               (_MARCH_FS, GL_FRAGMENT_SHADER))

        self._mk_body()
        self._mk_plate()
        self._mk_strings()
        self._mk_neck()
        self._mk_electroacoustic()
        self._mk_stage()
        self._mk_markers()
        self._mk_pressure_tex()
        self._mk_exterior_mask_tex()
        self._mk_body_mask_tex()
        self._mk_march_quad()
        self._mk_ray_segs()

    def _mk_body(self):
        # Side walls — indexed mesh, per-quad normals
        wv, wn, wi = _side_walls(self.outline, self.body_h)
        comb = np.ascontiguousarray(np.column_stack([wv, wn]), np.float32)
        self._wall_vao, self._wall_vbo, _ = _vao(
            comb, [(0,3,24,0),(1,3,24,12)], GL_STATIC_DRAW)
        self._wall_ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._wall_ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, wi.nbytes, wi, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        self._wall_n_idx = len(wi)
        # Bind IBO to VAO properly
        glBindVertexArray(self._wall_vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._wall_ibo)
        glBindVertexArray(0)

        # Back plate
        back = _back_fan(self.outline)
        back_n = np.tile(np.array([0.,0.,-1.], np.float32), (len(back),1))
        back_c = np.ascontiguousarray(np.column_stack([back, back_n]), np.float32)
        self._back_vao, _, self._back_n = _vao(
            back_c, [(0,3,24,0),(1,3,24,12)], GL_STATIC_DRAW)

        # Outline rings (thin wire guides)
        top = np.column_stack([self.outline,
            np.full(len(self.outline), self.body_h, np.float32)]).astype(np.float32)
        bot = np.column_stack([self.outline,
            np.zeros(len(self.outline), np.float32)]).astype(np.float32)
        self._top_vao, _, _ = _vao(top, [(0,3,12,0)], GL_STATIC_DRAW)
        self._bot_vao, _, _ = _vao(bot, [(0,3,12,0)], GL_STATIC_DRAW)
        self._ring_n = len(top)

    def _mk_plate(self):
        vxy, idx = _plate_mesh(
            self.outline,
            n_theta=self._plate_theta,
            n_radial=self._plate_radial)
        self._plate_vxy = vxy
        n = len(vxy)
        self._plate_dyn = np.zeros((n, 4), np.float32)
        self._plate_dyn[:,:2] = vxy
        self._plate_dyn[:,2]  = self.body_h
        self._plate_idx  = idx
        self._plate_n    = len(idx)

        self._plate_vao = glGenVertexArrays(1)
        self._plate_vbo = glGenBuffers(1)
        self._plate_ibo = glGenBuffers(1)
        glBindVertexArray(self._plate_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._plate_vbo)
        glBufferData(GL_ARRAY_BUFFER, self._plate_dyn.nbytes,
                     self._plate_dyn, GL_DYNAMIC_DRAW)
        # loc 0: xyz (stride 16, offset 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16, None)
        # loc 2: data (stride 16, offset 12)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(12))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._plate_ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
        glBindVertexArray(0)

        # Soundhole dark disc
        dv = _disc_verts(SOUNDHOLE_CX, SOUNDHOLE_CY, self.body_h + 5e-4)
        self._hole_vao, _, self._hole_n = _vao(dv, [(0,3,12,0)], GL_STATIC_DRAW)

    def _mk_strings(self):
        self._str_vaos, self._str_vbos, self._str_n = [], [], []
        self._str_env_hi_vaos, self._str_env_hi_vbos = [], []
        self._str_env_lo_vaos, self._str_env_lo_vbos = [], []
        # per-string per-segment peak-hold envelope (physical mode)
        self._str_env = [np.zeros(len(p) - 1, np.float32) for p in self.paths]
        self._str_physical_mode = True
        for path in self.paths:
            va, vb, _ = _vao(path.copy(), [(0,3,12,0)])
            self._str_vaos.append(va)
            self._str_vbos.append(vb)
            self._str_n.append(len(path))
            va_hi, vb_hi, _ = _vao(path.copy(), [(0,3,12,0)])
            self._str_env_hi_vaos.append(va_hi)
            self._str_env_hi_vbos.append(vb_hi)
            va_lo, vb_lo, _ = _vao(path.copy(), [(0,3,12,0)])
            self._str_env_lo_vaos.append(va_lo)
            self._str_env_lo_vbos.append(vb_lo)

    def _mk_neck(self):
        wood, frets, pins, ext_strings, anchor = _neck_geometry(
            self.outline, self.body_h, self.n_str,
            active_fret=self.active_fret,
            fretless=self.fretless)
        nrm = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (len(wood), 1))
        comb = np.ascontiguousarray(np.column_stack([wood, nrm]), np.float32)
        self._neck_vao, _, self._neck_n = _vao(comb, [(0,3,24,0),(1,3,24,12)], GL_STATIC_DRAW)
        self._fret_vao, _, self._fret_n = _vao(frets, [(0,3,12,0)], GL_STATIC_DRAW)
        self._anchor_vao, _, self._anchor_n = _vao(anchor, [(0,3,12,0)], GL_STATIC_DRAW)
        self._pin_vao, _, self._pin_n = _vao(pins, [(0,3,12,0)], GL_STATIC_DRAW)
        self._ext_str_vaos, self._ext_str_n = [], []
        for path in ext_strings:
            va, _, _ = _vao(path, [(0,3,12,0)], GL_STATIC_DRAW)
            self._ext_str_vaos.append(va)
            self._ext_str_n.append(len(path))

    def _mk_electroacoustic(self):
        pickup, mic = _electroacoustic_fixtures(self.body_h)
        self._pickup_vao, _, self._pickup_n = _vao(pickup, [(0,3,12,0)], GL_STATIC_DRAW)
        self._mic_vao, _, self._mic_n = _vao(mic, [(0,3,12,0)], GL_STATIC_DRAW)

    def _mk_stage(self):
        sv, sn, _ = _stage_mesh(self.outline, self.body_h)
        verts = sv.reshape(-1, 3)
        comb = np.ascontiguousarray(np.column_stack([verts, np.repeat(sn, 3, axis=0)]), np.float32)
        self._stage_vao, _, self._stage_n = _vao(comb, [(0,3,24,0),(1,3,24,12)], GL_STATIC_DRAW)
        # Spotlight: above and in front of the guitar in world space.
        # Guitar body centre ≈ (0, body_h/2, STAND_HEIGHT_M+0.25); lamp hangs above-front.
        lx, ly, lz = 0.25, STAGE_D_M * 0.35, STAGE_H_M * 0.88
        r = 0.06
        lamp = np.array([
            [lx-r, ly, lz], [lx+r, ly, lz],
            [lx+r, ly, lz-r*0.5], [lx-r, ly, lz-r*0.5],
        ], np.float32)
        self._lamp_vao, _, self._lamp_n = _vao(lamp, [(0,3,12,0)], GL_STATIC_DRAW)

    def _mk_markers(self):
        R = 0.007
        lines = []
        for bx, by in self.bridge:
            bz = self.body_h + 0.001
            lines += [[bx-R,by,bz],[bx+R,by,bz],[bx,by-R,bz],[bx,by+R,bz]]
        if not lines:
            lines = [[0,0,0],[0,0,0]]
        mk = np.array(lines, np.float32)
        self._mk_vao, _, self._mk_n = _vao(mk, [(0,3,12,0)], GL_STATIC_DRAW)

    def _mk_pressure_tex(self):
        Nx, Ny, Nz = self.info['Nx'], self.info['Ny'], self.info['Nz']
        self._tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self._tex)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, Nx, Ny, Nz, 0,
                     GL_RED, GL_FLOAT, None)
        for p, v in [(GL_TEXTURE_MIN_FILTER, GL_LINEAR),
                     (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                     (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                     (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                     (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)]:
            glTexParameteri(GL_TEXTURE_3D, p, v)
        glBindTexture(GL_TEXTURE_3D, 0)
        # AMR mode: also create a Texture Buffer Object for cell-centre IDW.
        self._amr_tbo_buf = None
        self._amr_tbo_tex = None
        self._amr_n_cells = 0
        self._amr_eps2    = 1e-6
        self._amr_cell_data = None  # (n, 4) RGBA32F: (cx,cy,cz,pressure)
        self._mk_amr_tbo()

    def _mk_amr_tbo(self):
        """Create a GL_TEXTURE_BUFFER holding (cx,cy,cz,pressure) per AMR cell.

        Cell centres are uploaded once here (they never move).  Only the
        pressure component (w) is updated each frame by _up_amr_pressure().
        When info has no 'amr_cell_centers' key this is a no-op.
        """
        centers = self.info.get('amr_cell_centers', None)
        n = int(self.info.get('n_cells_amr', 0))
        if centers is None or n == 0:
            return
        centers_f32 = np.asarray(centers, np.float32).reshape(n, 3)
        data = np.zeros((n, 4), dtype=np.float32)
        data[:, :3] = centers_f32
        self._amr_cell_data = data           # kept for per-frame pressure writes
        self._amr_n_cells   = n
        min_dx = float(self.info.get('amr_min_dx', self.info.get('min_dx', 0.005)))
        self._amr_eps2      = float((min_dx * 0.5) ** 2)
        # IDW distance power p.  p/2 is what the shader receives so it can use
        # pow(r², -p/2) without a sqrt.  Default p=2 (classic IDW, cheapest).
        # Set p=3 for the 3-D Shepard optimum; p=4 for sharper local detail.
        p = float(self.info.get('amr_idw_power', 2.0))
        self._amr_half_pow  = p * 0.5

        self._amr_tbo_buf = glGenBuffers(1)
        glBindBuffer(GL_TEXTURE_BUFFER, self._amr_tbo_buf)
        glBufferData(GL_TEXTURE_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_TEXTURE_BUFFER, 0)

        self._amr_tbo_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_BUFFER, self._amr_tbo_tex)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, self._amr_tbo_buf)
        glBindTexture(GL_TEXTURE_BUFFER, 0)

    def _up_amr_pressure(self, P_flat):
        """Write new per-cell pressures into the TBO (only the w component)."""
        if self._amr_tbo_buf is None or self._amr_cell_data is None:
            return
        n = self._amr_n_cells
        self._amr_cell_data[:n, 3] = np.asarray(P_flat[:n], np.float32)
        glBindBuffer(GL_TEXTURE_BUFFER, self._amr_tbo_buf)
        glBufferSubData(GL_TEXTURE_BUFFER, 0,
                        self._amr_cell_data.nbytes, self._amr_cell_data)
        glBindBuffer(GL_TEXTURE_BUFFER, 0)

    def _mk_exterior_mask_tex(self):
        """Upload the exterior_mask bool volume as a GL_R8 texture.

        exterior_mask[i,j,k] == True (1.0) means the cell is outside the
        guitar body cavity (free air, PML, or rib wall region).  The march
        shader samples this to tint exterior radiation cyan and dim its
        opacity so the interior cavity resonance reads clearly distinct.
        """
        ext = self.info.get('exterior_mask', None)
        Nx, Ny, Nz = self.info['Nx'], self.info['Ny'], self.info['Nz']
        self._ext_mask_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self._ext_mask_tex)
        if ext is not None:
            data = np.ascontiguousarray(ext.astype(np.float32))
        else:
            data = np.ones((Nx, Ny, Nz), dtype=np.float32)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, Nx, Ny, Nz, 0,
                     GL_RED, GL_FLOAT, data)
        for p, v in [(GL_TEXTURE_MIN_FILTER, GL_NEAREST),
                     (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                     (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                     (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                     (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)]:
            glTexParameteri(GL_TEXTURE_3D, p, v)
        glBindTexture(GL_TEXTURE_3D, 0)

    def _mk_body_mask_tex(self):
        Nx, Ny = self.info['Nx'], self.info['Ny']
        mask = _outline_mask(self.outline, self.info).T.astype(np.float32)
        self._mask_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._mask_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, Nx, Ny, 0,
                     GL_RED, GL_FLOAT, np.ascontiguousarray(mask))
        for p, v in [(GL_TEXTURE_MIN_FILTER, GL_LINEAR),
                     (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                     (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                     (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)]:
            glTexParameteri(GL_TEXTURE_2D, p, v)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _mk_march_quad(self):
        quad = np.array([[-1,-1],[1,-1],[1,1],[-1,1]], np.float32)
        self._quad_vao, _, _ = _vao(quad, [(0,2,8,0)], GL_STATIC_DRAW)

    def _mk_ray_segs(self):
        vdata = _ray_vbo(self._ray_segs)
        self._ray_n = len(vdata)
        if self._gpu_seg_vbo is not None and self._gpu_seg_cap > 0:
            self._ray_vao = glGenVertexArrays(1)
            glBindVertexArray(self._ray_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self._gpu_seg_vbo)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(16))
            glBindVertexArray(0)
            self._ray_n = self._gpu_seg_cap * 2
            return
        if self._ray_n == 0:
            self._ray_vao = None
            return
        # stride 28: xyz(12) + rgba(16)
        self._ray_vao, _, _ = _vao(
            vdata, [(0,3,28,0),(1,4,28,12)], GL_STATIC_DRAW)

    # ── Frame management ──────────────────────────────────────────────────────

    def push(self, f: Frame): self._frames.append(f)

    def _cur(self) -> Optional[Frame]:
        if not self._frames: return None
        return self._frames[self._cursor % len(self._frames)]

    def tick(self):
        if not self._paused and self._frames:
            self._cursor = (self._cursor + 1) % max(1, len(self._frames))
        self.cam.tick()

    def toggle(self, i):
        if 0 <= i < N_LAYERS:
            self._layers[i] = (self._layers[i] + 1) % 3

    def cycle_plate_mode(self):
        self._plate_mode = (self._plate_mode + 1) % 3
        return ("color", "height", "both")[self._plate_mode]

    def _ray_scale_for_display(self) -> float:
        rays = max(1, int(self._ray_field_total_rays or GPU_RAY_FIELD_REFERENCE_RAYS))
        return (float(GPU_RAY_FIELD_SCALE)
                * self._ray_exposure
                * (float(GPU_RAY_FIELD_REFERENCE_RAYS) / float(rays)))

    def _baseline_scale_for_display(self) -> float:
        rays = max(1, int(self._baseline_light_total_rays or STAGE_LIGHT_RAYS))
        return 1.8 * self._ray_exposure * (float(STAGE_LIGHT_RAYS) / float(rays))

    def set_ray_tonemap(self, *, exposure: float | None = None, gamma: float | None = None) -> None:
        if exposure is not None:
            self._ray_exposure = float(np.clip(exposure, 0.1, 8.0))
        if gamma is not None:
            self._ray_gamma = float(np.clip(gamma, 0.1, 2.5))

    def replace_ray_field(self, ray_field_bands, ray_field_bounds, gpu_seg_vbo, gpu_seg_cap, gpu_counter=None, total_rays=0):
        _ray_diag_update(
            "renderer:replace_ray_field:start",
            old_textures=tuple(int(t) for t in (self._ray_field_bands or [])),
            new_textures=tuple(int(t) for t in (ray_field_bands or [])),
            old_gpu_seg_vbo=int(self._gpu_seg_vbo or 0),
            new_gpu_seg_vbo=int(gpu_seg_vbo or 0),
            gpu_counter=int(gpu_counter or 0),
            gpu_seg_cap=int(gpu_seg_cap or 0),
            total_rays=int(total_rays or 0),
        )
        if self._ray_field_bands:
            for tex in self._ray_field_bands:
                glDeleteTextures([tex])
        if self._gpu_seg_vbo is not None:
            glDeleteBuffers(1, [self._gpu_seg_vbo])
        if gpu_counter is not None:
            glDeleteBuffers(1, [gpu_counter])
        self._ray_field_bands = ray_field_bands
        self._ray_field_bounds = ray_field_bounds
        self._ray_field_total_rays = int(total_rays or 0)
        self._gpu_seg_vbo = gpu_seg_vbo
        self._gpu_seg_cap = int(gpu_seg_cap or 0)
        self._mk_ray_segs()
        _ray_diag_update(
            "renderer:replace_ray_field:done",
            active_textures=tuple(int(t) for t in (self._ray_field_bands or [])),
            active_gpu_seg_vbo=int(self._gpu_seg_vbo or 0),
            gpu_seg_cap=int(self._gpu_seg_cap or 0),
            ray_draw_vertices=int(getattr(self, "_ray_n", 0)),
            total_rays=int(self._ray_field_total_rays or 0),
        )

    def rebuild_plate(self, theta: int, radial: int):
        """Rebuild soundboard mesh with new resolution (live, no physics restart needed)."""
        self._plate_theta  = int(theta)
        self._plate_radial = int(radial)
        glDeleteVertexArrays(1, [self._plate_vao])
        glDeleteBuffers(1, [self._plate_vbo])
        glDeleteBuffers(1, [self._plate_ibo])
        glDeleteVertexArrays(1, [self._hole_vao])
        self._mk_plate()

    def rebuild_physics(self, ce, info: dict, paths: list, body_h: float,
                        active_fret: int = 0, fretless: bool = False):
        """Hot-swap physics after a parameter change (segs or dx)."""
        self.info   = info
        self.paths  = paths
        self.n_str  = len(paths)
        self.body_h = body_h
        self.active_fret = int(active_fret)
        self.fretless = bool(fretless)
        dx_val  = float(info.get('dx', DX))
        gx_min  = float(info['gx_min'])
        gy_min  = float(info['gy_min'])
        gz_min  = float(info.get('gz_min', -PAD_CELLS * dx_val))
        Nx, Ny, Nz = info['Nx'], info['Ny'], info['Nz']
        self._bmin = np.array([gx_min, gy_min, gz_min], np.float32)
        self._bmax = np.array([gx_min + Nx*dx_val, gy_min + Ny*dx_val,
                               gz_min + Nz*dx_val], np.float32)
        glDeleteTextures([self._tex])
        if hasattr(self, '_ext_mask_tex'):
            glDeleteTextures([self._ext_mask_tex])
        self._mk_pressure_tex()
        self._mk_exterior_mask_tex()
        self._mk_strings()
        self._mk_neck()
        self._mk_electroacoustic()
        self._frames.clear()
        self._cursor = 0

    def _a(self, i):
        s = self._layers[i]
        return 0.0 if s == LAYER_HIDDEN else (1.0 if s == LAYER_OPAQUE else 0.40)

    def current_frame(self) -> Optional[Frame]:
        return self._cur()

    # ── Dynamic updates ───────────────────────────────────────────────────────

    def _up_pressure(self, P):
        if self._amr_tbo_buf is not None and np.ndim(P) == 1:
            self._up_amr_pressure(P)
            return
        Nx, Ny, Nz = self.info['Nx'], self.info['Ny'], self.info['Nz']
        data = np.ascontiguousarray(P[:Nx,:Ny,:Nz], np.float32)
        glBindTexture(GL_TEXTURE_3D, self._tex)
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0,0,0, Nx,Ny,Nz, GL_RED, GL_FLOAT, data)
        glBindTexture(GL_TEXTURE_3D, 0)

    def _up_plate(self, disp):
        Nx, Ny = self.info['Nx'], self.info['Ny']
        d = disp[:Nx, :Ny]
        mx = float(np.abs(d).max()) + 1e-9
        dx = float(self.info.get('dx', DX))
        gx_min = float(self.info['gx_min'])
        gy_min = float(self.info['gy_min'])
        gx = np.clip((self._plate_vxy[:, 0] - gx_min) / dx - 0.5, 0.0, Nx - 1.001)
        gy = np.clip((self._plate_vxy[:, 1] - gy_min) / dx - 0.5, 0.0, Ny - 1.001)
        i0 = gx.astype(np.int32); j0 = gy.astype(np.int32)
        i1 = np.minimum(i0 + 1, Nx - 1); j1 = np.minimum(j0 + 1, Ny - 1)
        fx = gx - i0; fy = gy - j0
        vals = (
            (1.0 - fx) * (1.0 - fy) * d[i0, j0]
          + fx * (1.0 - fy) * d[i1, j0]
          + (1.0 - fx) * fy * d[i0, j1]
          + fx * fy * d[i1, j1]
        ).astype(np.float32)
        height_on = self._plate_mode in (1, 2)
        color_on = self._plate_mode in (0, 2)
        self._plate_dyn[:, 2] = self.body_h + (vals * PLATE_SCALE if height_on else 0.0)
        self._plate_dyn[:, 3] = (vals / mx if color_on else 0.0)
        glBindBuffer(GL_ARRAY_BUFFER, self._plate_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self._plate_dyn.nbytes, self._plate_dyn)

    def _up_string(self, si, pts):
        self._str_n[si] = len(pts)
        pts = np.ascontiguousarray(pts, np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._str_vbos[si])
        glBufferSubData(GL_ARRAY_BUFFER, 0, pts.nbytes, pts)
        glBindBuffer(GL_ARRAY_BUFFER, self._str_env_hi_vbos[si])
        glBufferSubData(GL_ARRAY_BUFFER, 0, pts.nbytes, pts)
        glBindBuffer(GL_ARRAY_BUFFER, self._str_env_lo_vbos[si])
        glBufferSubData(GL_ARRAY_BUFFER, 0, pts.nbytes, pts)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        glClearColor(0.015, 0.010, 0.040, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, self.win_w, self.win_h)

        frame = self._cur()
        if frame is None: return

        aspect  = self.win_w / self.win_h
        MVP     = self.cam.mvp(aspect)
        MV      = self.cam.mv()
        iMVP    = self.cam.inv_mvp(aspect)

        # Guitar geometry lives in guitar-frame; transform to world via model matrix.
        MVP_guitar = (MVP @ self._guitar_M).astype(np.float32)
        MV_guitar  = (MV  @ self._guitar_M).astype(np.float32)

        # Stage spotlight: from above-front in world space (mostly +Y and +Z).
        light_w = np.array([0.15, 0.85, 0.55], np.float32)
        light_w /= np.linalg.norm(light_w)
        light_v = (MV[:3,:3].T @ light_w).astype(np.float32)

        self._up_pressure(frame.pressure)
        self._up_plate(frame.plate)
        for si, disp in enumerate(frame.strings):
            if disp is not None and len(disp) >= 2:
                self._up_string(si, disp)

        # Whether to use ray-lit surface shader for body geometry.
        # Active when ILLUM layer (7) is not hidden and band textures exist.
        _use_ray_surf = (
            self._a(7) > 0.0
            and (
                (self._ray_field_bands is not None and len(self._ray_field_bands) >= 4)
                or (self._baseline_light_bands is not None and len(self._baseline_light_bands) >= 4)
            )
        )

        def _bind_body(prog, material: SurfaceMaterialSpec, alpha,
                       mvp_mat=MVP_guitar, mv_mat=MV_guitar, model_mat=None):
            """Bind a body-shader program with standard material uniforms.
            For _p_ray_surface also binds band textures."""
            if model_mat is None:
                model_mat = self._guitar_M.astype(np.float32)
            glUseProgram(prog)
            _mvp(prog, mvp_mat, mv_mat, model_mat)
            glUniform3f(glGetUniformLocation(prog, b'uLightV'), *light_v)
            glUniform4f(glGetUniformLocation(prog, b'uColor'),
                        material.color[0], material.color[1], material.color[2], alpha)
            glUniform3f(glGetUniformLocation(prog, b'uInnerColor'), *material.inner_color)
            glUniform1f(glGetUniformLocation(prog, b'uAmbient'), material.ambient)
            glUniform1f(glGetUniformLocation(prog, b'uSpecStrength'), material.spec_strength)
            glUniform1f(glGetUniformLocation(prog, b'uShininess'), material.shininess)
            glUniform1f(glGetUniformLocation(prog, b'uGrain'), material.grain)
            if prog == self._p_ray_surface:
                use_dyn = self._ray_field_bands is not None and len(self._ray_field_bands) >= 4
                use_base = self._baseline_light_bands is not None and len(self._baseline_light_bands) >= 4
                glUniform1i(glGetUniformLocation(prog, b'uUseRayField'), 1 if use_dyn else 0)
                glUniform1i(glGetUniformLocation(prog, b'uUseBaseField'), 1 if use_base else 0)
                rf_min = self._ray_field_bounds[0] if self._ray_field_bounds is not None else self._bmin
                rf_max = self._ray_field_bounds[1] if self._ray_field_bounds is not None else self._bmax
                glUniform3f(glGetUniformLocation(prog, b'uBoxMin'), *rf_min)
                glUniform3f(glGetUniformLocation(prog, b'uBoxMax'), *rf_max)
                glUniformMatrix4fv(
                    glGetUniformLocation(prog, b'uWorldToGrid'),
                    1, GL_TRUE, self._guitar_Minv)
                glUniform1f(glGetUniformLocation(prog, b'uRayExposure'),
                            self._ray_scale_for_display())
                glUniform1f(glGetUniformLocation(prog, b'uBaseExposure'),
                            self._baseline_scale_for_display())
                glUniform1f(glGetUniformLocation(prog, b'uRayGamma'),
                            self._ray_gamma)
                band_names = [b'uBand0', b'uBand1', b'uBand2', b'uBand3']
                for bi, (bname, btex) in enumerate(zip(band_names, self._ray_field_bands or [])):
                    glUniform1i(glGetUniformLocation(prog, bname), 2 + bi)
                    glActiveTexture(GL_TEXTURE2 + bi)
                    glBindTexture(GL_TEXTURE_3D, btex)
                base_min = (self._baseline_light_bounds[0]
                            if self._baseline_light_bounds is not None
                            else _stage_light_bounds()[0])
                base_max = (self._baseline_light_bounds[1]
                            if self._baseline_light_bounds is not None
                            else _stage_light_bounds()[1])
                glUniform3f(glGetUniformLocation(prog, b'uBaseBoxMin'), *base_min)
                glUniform3f(glGetUniformLocation(prog, b'uBaseBoxMax'), *base_max)
                glUniformMatrix4fv(
                    glGetUniformLocation(prog, b'uBaseWorldToGrid'),
                    1, GL_TRUE, np.eye(4, dtype=np.float32))
                base_names = [b'uBaseBand0', b'uBaseBand1', b'uBaseBand2', b'uBaseBand3']
                for bi, (bname, btex) in enumerate(zip(base_names, self._baseline_light_bands or [])):
                    glUniform1i(glGetUniformLocation(prog, bname), 6 + bi)
                    glActiveTexture(GL_TEXTURE0 + 6 + bi)
                    glBindTexture(GL_TEXTURE_3D, btex)
                glActiveTexture(GL_TEXTURE0)

        _body_prog = self._p_ray_surface if _use_ray_surf else self._p_body
        body_mode = self._layers[0]
        body_back_alpha = _layer_material_alpha(
            body_mode,
            opaque_alpha=_MAT_BACK.opaque_alpha,
            alpha_alpha=_MAT_BACK.alpha_alpha)
        body_side_alpha = _layer_material_alpha(
            body_mode,
            opaque_alpha=_MAT_SIDES.opaque_alpha,
            alpha_alpha=_MAT_SIDES.alpha_alpha)

        # ── 1. Back plate (opaque dark wood) ─────────────────────────────────
        a_body = self._a(0)
        if body_back_alpha > 0:
            _bind_body(_body_prog, _MAT_BACK, body_back_alpha)
            glBindVertexArray(self._back_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, self._back_n)

        # ── 2. Side walls (semi-transparent mahogany, both faces) ─────────────
        if body_side_alpha > 0:
            glDisable(GL_CULL_FACE)
            glDepthMask(GL_FALSE if body_mode == LAYER_ALPHA else GL_TRUE)
            _bind_body(_body_prog, _MAT_SIDES, body_side_alpha)
            glBindVertexArray(self._wall_vao)
            glDrawElements(GL_TRIANGLES, self._wall_n_idx, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            glDepthMask(GL_TRUE)

        # ── 3. Outline rings (crisp wire guide) ───────────────────────────────
        if a_body > 0:
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP_guitar)
            glLineWidth(1.6)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        0.90, 0.68, 0.28, a_body * 0.95)
            glBindVertexArray(self._top_vao)
            glDrawArrays(GL_LINE_LOOP, 0, self._ring_n)
            glBindVertexArray(self._bot_vao)
            glDrawArrays(GL_LINE_LOOP, 0, self._ring_n)

        # ── 4. Pressure volume (FDTD, key-3 toggle only) ──────────────────────
        a_pres = self._a(2)
        if a_pres > 0:
            glDepthMask(GL_FALSE)
            glDisable(GL_DEPTH_TEST)
            glUseProgram(self._p_march)
            glUniformMatrix4fv(
                glGetUniformLocation(self._p_march, b'uInvMVP'),
                1, GL_TRUE, iMVP)
            glUniformMatrix4fv(
                glGetUniformLocation(self._p_march, b'uWorldToGrid'),
                1, GL_TRUE, self._guitar_Minv)
            glUniform3f(glGetUniformLocation(self._p_march, b'uBoxMin'), *self._bmin)
            glUniform3f(glGetUniformLocation(self._p_march, b'uBoxMax'), *self._bmax)
            glUniform2f(glGetUniformLocation(self._p_march, b'uMaskMin'),
                        float(self._bmin[0]), float(self._bmin[1]))
            glUniform2f(glGetUniformLocation(self._p_march, b'uMaskMax'),
                        float(self._bmax[0]), float(self._bmax[1]))
            glUniform1f(glGetUniformLocation(self._p_march, b'uPressureScale'),
                        GPU_PRESSURE_SCALE)
            glUniform1f(glGetUniformLocation(self._p_march, b'uPressureGamma'), 1.0)
            glUniform1f(glGetUniformLocation(self._p_march, b'uRayFieldScale'),
                        GPU_RAY_FIELD_SCALE)
            glUniform1f(glGetUniformLocation(self._p_march, b'uRayFieldGamma'),
                        self._ray_gamma)
            glUniform1f(glGetUniformLocation(self._p_march, b'uAlpha'), a_pres)
            glUniform1i(glGetUniformLocation(self._p_march, b'uLogScale'), 0)
            glUniform1i(glGetUniformLocation(self._p_march, b'uFieldMode'), 0)
            glUniform1i(glGetUniformLocation(self._p_march, b'uPressure'), 0)
            # No body-mask for pressure: the FDTD simulation already has zero
            # pressure in solid regions (guitar walls).  Masking by the 2D guitar
            # outline would create a featureless air-cylinder and hole out the
            # soundhole.  Let the pressure data define its own boundaries.
            glUniform1i(glGetUniformLocation(self._p_march, b'uBodyMask'), 1)
            glUniform1i(glGetUniformLocation(self._p_march, b'uUseBodyMask'), 0)
            # Show full domain Z so the near-field can be seen radiating into the
            # free-air buffer.  The exterior_mask shader tint distinguishes cavity
            # from exterior air — no need to clip away the surrounding atmosphere.
            dx_val = float(self.info.get('dx', DX))
            gz_min = float(self.info.get('gz_min', -PAD_CELLS * dx_val))
            Nz     = int(self.info['Nz'])
            glUniform1f(glGetUniformLocation(self._p_march, b'uZClipMin'), gz_min)
            glUniform1f(glGetUniformLocation(self._p_march, b'uZClipMax'), gz_min + Nz * dx_val)
            # AMR TBO IDW path: bind cell data to unit 3 and set mode uniforms.
            # Falls back to the legacy 3D uPressure texture when not in AMR mode.
            _amr_mode = 1 if self._amr_tbo_tex is not None else 0
            glUniform1i(glGetUniformLocation(self._p_march, b'uAMRMode'), _amr_mode)
            if _amr_mode:
                glUniform1i(glGetUniformLocation(self._p_march, b'uAMRNCells'), self._amr_n_cells)
                glUniform1f(glGetUniformLocation(self._p_march, b'uAMREps2'),   self._amr_eps2)
                glUniform1f(glGetUniformLocation(self._p_march, b'uAMRHalfPow'), self._amr_half_pow)
                glUniform1i(glGetUniformLocation(self._p_march, b'uAMRData'),   3)
            else:
                glUniform1i(glGetUniformLocation(self._p_march, b'uAMRMode'), 0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_3D, self._tex)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self._mask_tex)
            glUniform1i(glGetUniformLocation(self._p_march, b'uExteriorMask'), 2)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_3D, self._ext_mask_tex)
            if _amr_mode:
                glActiveTexture(GL_TEXTURE3)
                glBindTexture(GL_TEXTURE_BUFFER, self._amr_tbo_tex)
            glActiveTexture(GL_TEXTURE0)
            glBindVertexArray(self._quad_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
            glBindTexture(GL_TEXTURE_3D, 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_3D, 0)
            if _amr_mode:
                glActiveTexture(GL_TEXTURE3)
                glBindTexture(GL_TEXTURE_BUFFER, 0)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)

        # ── 4a. Neutral diffusive room / stage ───────────────────────────────
        a_stage = self._a(6)
        if a_stage > 0:
            glEnable(GL_CULL_FACE)
            _bind_body(_body_prog,
                       _MAT_STAGE,
                       a_stage * _MAT_STAGE.alpha_alpha,
                       mvp_mat=MVP, mv_mat=MV,
                       model_mat=np.eye(4, dtype=np.float32))
            glBindVertexArray(self._stage_vao)
            glDrawArrays(GL_TRIANGLES, 0, self._stage_n)
            glDisable(GL_CULL_FACE)

        # ── 4b. Cached room light volume (ILLUM layer — key 8) ─────────────────
        if self._baseline_light_bands is not None and self._a(7) > 0.0:
            lb_min = self._baseline_light_bounds[0] if self._baseline_light_bounds is not None else _stage_light_bounds()[0]
            lb_max = self._baseline_light_bounds[1] if self._baseline_light_bounds is not None else _stage_light_bounds()[1]
            glDepthMask(GL_FALSE)
            glDisable(GL_DEPTH_TEST)
            glUseProgram(self._p_march)
            glUniformMatrix4fv(glGetUniformLocation(self._p_march, b'uInvMVP'), 1, GL_TRUE, iMVP)
            glUniformMatrix4fv(glGetUniformLocation(self._p_march, b'uWorldToGrid'),
                               1, GL_TRUE, np.eye(4, dtype=np.float32))
            glUniform3f(glGetUniformLocation(self._p_march, b'uBoxMin'), *lb_min)
            glUniform3f(glGetUniformLocation(self._p_march, b'uBoxMax'), *lb_max)
            glUniform2f(glGetUniformLocation(self._p_march, b'uMaskMin'), float(lb_min[0]), float(lb_min[1]))
            glUniform2f(glGetUniformLocation(self._p_march, b'uMaskMax'), float(lb_max[0]), float(lb_max[1]))
            glUniform1f(glGetUniformLocation(self._p_march, b'uPressureScale'), GPU_PRESSURE_SCALE)
            glUniform1f(glGetUniformLocation(self._p_march, b'uPressureGamma'), 1.0)
            glUniform1f(glGetUniformLocation(self._p_march, b'uRayFieldScale'),
                        self._baseline_scale_for_display())
            glUniform1f(glGetUniformLocation(self._p_march, b'uRayFieldGamma'), self._ray_gamma)
            glUniform1f(glGetUniformLocation(self._p_march, b'uAlpha'), 0.22)
            glUniform1i(glGetUniformLocation(self._p_march, b'uLogScale'), 1 if GPU_RAY_LOG_SCALE else 0)
            glUniform1i(glGetUniformLocation(self._p_march, b'uFieldMode'), 1)
            glUniform1i(glGetUniformLocation(self._p_march, b'uUseBodyMask'), 0)
            glUniform1f(glGetUniformLocation(self._p_march, b'uZClipMin'), float(lb_min[2]))
            glUniform1f(glGetUniformLocation(self._p_march, b'uZClipMax'), float(lb_max[2]))
            for bi, (bname, btex) in enumerate(zip([b'uBand0', b'uBand1', b'uBand2', b'uBand3'],
                                                   self._baseline_light_bands)):
                glUniform1i(glGetUniformLocation(self._p_march, bname), 2 + bi)
                glActiveTexture(GL_TEXTURE2 + bi)
                glBindTexture(GL_TEXTURE_3D, btex)
            glActiveTexture(GL_TEXTURE0)
            glBindVertexArray(self._quad_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
            for bi in range(4):
                glActiveTexture(GL_TEXTURE2 + bi)
                glBindTexture(GL_TEXTURE_3D, 0)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)

        # ── 4c. Guitar-added ray field volume (ILLUM layer — key 8) ───────────
        if self._ray_field_bands is not None and self._a(7) > 0.0:
            rf_min = self._ray_field_bounds[0] if self._ray_field_bounds is not None else self._bmin
            rf_max = self._ray_field_bounds[1] if self._ray_field_bounds is not None else self._bmax
            glDepthMask(GL_FALSE)
            glDisable(GL_DEPTH_TEST)
            glUseProgram(self._p_march)
            glUniformMatrix4fv(
                glGetUniformLocation(self._p_march, b'uInvMVP'),
                1, GL_TRUE, iMVP)
            glUniformMatrix4fv(
                glGetUniformLocation(self._p_march, b'uWorldToGrid'),
                1, GL_TRUE, self._guitar_Minv)
            glUniform3f(glGetUniformLocation(self._p_march, b'uBoxMin'), *rf_min)
            glUniform3f(glGetUniformLocation(self._p_march, b'uBoxMax'), *rf_max)
            glUniform2f(glGetUniformLocation(self._p_march, b'uMaskMin'),
                        float(self._bmin[0]), float(self._bmin[1]))
            glUniform2f(glGetUniformLocation(self._p_march, b'uMaskMax'),
                        float(self._bmax[0]), float(self._bmax[1]))
            glUniform1f(glGetUniformLocation(self._p_march, b'uPressureScale'),
                        GPU_PRESSURE_SCALE)
            glUniform1f(glGetUniformLocation(self._p_march, b'uPressureGamma'), 1.0)
            glUniform1f(glGetUniformLocation(self._p_march, b'uRayFieldScale'),
                        self._ray_scale_for_display())
            glUniform1f(glGetUniformLocation(self._p_march, b'uRayFieldGamma'),
                        self._ray_gamma)
            glUniform1f(glGetUniformLocation(self._p_march, b'uAlpha'), 0.55)
            glUniform1i(glGetUniformLocation(self._p_march, b'uLogScale'),
                        1 if GPU_RAY_LOG_SCALE else 0)
            glUniform1i(glGetUniformLocation(self._p_march, b'uFieldMode'), 1)
            glUniform1i(glGetUniformLocation(self._p_march, b'uPressure'), 0)
            # Ray field = full acoustic radiation volume — do not clip to guitar XY
            # footprint.  The ray-field box already defines its own spatial extent.
            glUniform1i(glGetUniformLocation(self._p_march, b'uBodyMask'), 1)
            glUniform1i(glGetUniformLocation(self._p_march, b'uUseBodyMask'), 0)
            # Z clip: use the ray field's own Z extent (avoids showing PML fringe)
            glUniform1f(glGetUniformLocation(self._p_march, b'uZClipMin'),
                        float(rf_min[2]))
            glUniform1f(glGetUniformLocation(self._p_march, b'uZClipMax'),
                        float(rf_max[2]))
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self._mask_tex)
            # Bind 4 spectral band textures to units 2-5
            band_names = [b'uBand0', b'uBand1', b'uBand2', b'uBand3']
            for bi, (bname, btex) in enumerate(zip(band_names, self._ray_field_bands)):
                glUniform1i(glGetUniformLocation(self._p_march, bname), 2 + bi)
                glActiveTexture(GL_TEXTURE2 + bi)
                glBindTexture(GL_TEXTURE_3D, btex)
            glActiveTexture(GL_TEXTURE0)
            glBindVertexArray(self._quad_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
            # Unbind all 4 band textures
            for bi in range(4):
                glActiveTexture(GL_TEXTURE2 + bi)
                glBindTexture(GL_TEXTURE_3D, 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)

        # ── 5. Top plate surface (displaced, heatmap) ─────────────────────────
        a_plate = self._a(1)
        if a_plate > 0 and self._plate_n > 0:
            glUseProgram(self._p_plate)
            _mvp(self._p_plate, MVP_guitar, MV_guitar)
            glUniform3f(glGetUniformLocation(self._p_plate, b'uLightV'),
                        *light_v)
            glUniform1f(
                glGetUniformLocation(self._p_plate, b'uAlpha'),
                _layer_material_alpha(
                    self._layers[1],
                    opaque_alpha=_MAT_PLATE.opaque_alpha,
                    alpha_alpha=_MAT_PLATE.alpha_alpha))
            glUniform1f(glGetUniformLocation(self._p_plate, b'uColorMix'),
                        _MAT_PLATE.color_mix if self._plate_mode in (0, 2) else 0.0)
            glBindVertexArray(self._plate_vao)
            glDrawElements(GL_TRIANGLES, self._plate_n, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

        # ── 6. Soundhole dark disc ─────────────────────────────────────────────
        if a_plate > 0:
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP_guitar)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        0.03, 0.01, 0.01, 1.0)
            glBindVertexArray(self._hole_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, self._hole_n)

        # ── 7. Strings ────────────────────────────────────────────────────────
        a_str = self._a(3)
        if a_str > 0:
            _bind_body(self._p_body, _MAT_NECK, a_str * _MAT_NECK.opaque_alpha)
            glBindVertexArray(self._neck_vao)
            glDrawArrays(GL_TRIANGLES, 0, self._neck_n)

            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE)          # additive — strings glow into scene
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP_guitar)
            glLineWidth(1.2)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        0.95, 0.82, 0.52, a_str * 0.50)
            glBindVertexArray(self._fret_vao)
            glDrawArrays(GL_LINES, 0, self._fret_n)
            if self.active_fret > 0:
                glLineWidth(3.0 if self.fretless else 2.4)
                glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                            0.40, 0.85, 1.0, a_str * (0.65 if self.fretless else 0.85))
                glBindVertexArray(self._anchor_vao)
                glDrawArrays(GL_LINES, 0, self._anchor_n)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        0.88, 0.74, 0.46, a_str * 0.70)
            glBindVertexArray(self._pin_vao)
            glDrawArrays(GL_LINES, 0, self._pin_n)
            glLineWidth(2.2)
            for si in range(self.n_str):
                c = STRING_COLORS[si % len(STRING_COLORS)]
                gauge = STRING_GAUGES_IN[si] if si < len(STRING_GAUGES_IN) else 0.012
                tension = STRING_TENSIONS_N[si] if si < len(STRING_TENSIONS_N) else 68.0
                glLineWidth(float(np.clip(1.1 + gauge * 55.0 + (tension - 65.0) * 0.01, 1.4, 4.0)))
                glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                            c[0], c[1], c[2], a_str * c[3])
                if si < len(self._ext_str_vaos):
                    glBindVertexArray(self._ext_str_vaos[si])
                    glDrawArrays(GL_LINE_STRIP, 0, self._ext_str_n[si])
                glBindVertexArray(self._str_vaos[si])
                glDrawArrays(GL_LINE_STRIP, 0, self._str_n[si])
            glBindVertexArray(0)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # restore

        # ── 8. Bridge / pickup / mic markers ─────────────────────────────────
        a_mk = self._a(4)
        if a_mk > 0 and self._mk_n > 0:
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP_guitar)
            glLineWidth(1.8)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        1.0, 1.0, 0.5, a_mk * 0.85)
            glBindVertexArray(self._mk_vao)
            glDrawArrays(GL_LINES, 0, self._mk_n)
            if self.show_pickup:
                glLineWidth(2.6)
                glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                            0.2, 0.9, 1.0, a_mk * 0.80)
                glBindVertexArray(self._pickup_vao)
                glDrawArrays(GL_LINE_LOOP, 0, self._pickup_n)
            if self.show_mic:
                glLineWidth(1.8)
                glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                            1.0, 0.45, 0.85, a_mk * 0.85)
                glBindVertexArray(self._mic_vao)
                glDrawArrays(GL_LINES, 0, self._mic_n)
            glBindVertexArray(0)

        # ── 9. Ray-tracer segments ─────────────────────────────────────────────
        a_ray = self._a(5)
        if a_ray > 0 and self._ray_vao is not None:
            glDepthMask(GL_FALSE)
            glUseProgram(self._p_vcol)
            _mvp(self._p_vcol, MVP_guitar)
            glLineWidth(1.0)
            glBindVertexArray(self._ray_vao)
            glDrawArrays(GL_LINES, 0, self._ray_n)
            glBindVertexArray(0)
            glDepthMask(GL_TRUE)


# ─────────────────────────────────────────────────────────────────────────────
# Quality slider HUD
# ─────────────────────────────────────────────────────────────────────────────

class _SliderPanel:
    """Top-left quality-level sliders rendered as a 2-D OpenGL 330 overlay.

    Four axes, each starting at low quality:
      - Img rays   (imaging resolution)   — live
      - Str segs   (string resolution)    — rebuild on R
      - Board res  (soundboard mesh)      — live
      - Press dx   (pressure solver)      — rebuild on R

    Colors: green = live param, blue = rebuild needed but unchanged,
            amber = rebuild needed and value has been moved.
    """

    # Panel geometry (pixels, origin = top-left of window)
    PX, PY = 10, 10
    PW     = 264
    ROW    = 26
    TX     = 86     # track left edge within panel
    TW     = 162    # track width
    TH     = 6      # track height

    # (key, label, lo, hi, default, log_scale, live)
    # lo = left-end value (low quality), hi = right-end value (high quality)
    _DEFS = [
        ('rays',     'Img rays',    256,    50_000, 1_000, True,  True ),
        ('segs',     'Str segs',     30,       240,    60, False, False),
        ('plate_th', 'Board res',    64,     1_024,   128, False, True ),
        ('dx',       'Press dx',  0.016,     0.004, 0.010, False, False),
        ('ray_exposure', 'Exposure', 0.25,    6.0,    1.0, False, True ),
        ('ray_gamma', 'Gamma',      0.20,    1.60, float(GPU_RAY_FIELD_GAMMA), False, True ),
        ('mic_gain', 'Mic',        0.0,       1.0,   1.0, False, True ),
        ('pickup_gain', 'Pickup',  0.0,       1.0,   1.0, False, True ),
    ]

    _C_BG    = (0.04, 0.04, 0.07, 0.82)
    _C_TRACK = (0.22, 0.22, 0.27, 1.00)
    _C_LIVE  = (0.18, 0.78, 0.28, 1.00)   # green  — live
    _C_BUILD = (0.22, 0.46, 0.96, 1.00)   # blue   — needs rebuild (no pending change)
    _C_PEND  = (1.00, 0.62, 0.10, 1.00)   # amber  — pending rebuild change
    _C_KNOB  = (0.90, 0.90, 0.90, 1.00)
    _C_TEXT  = (0.82, 0.82, 0.82, 1.00)

    def __init__(self):
        self.keys   = [d[0] for d in self._DEFS]
        self._lo    = [d[2] for d in self._DEFS]
        self._hi    = [d[3] for d in self._DEFS]
        self.values = {d[0]: d[4] for d in self._DEFS}
        self._log   = [d[5] for d in self._DEFS]
        self._live  = [d[6] for d in self._DEFS]
        self._pend  = {d[0]: False for d in self._DEFS}
        self._prev  = dict(self.values)
        self._drag  = -1

        self._p_col = _prog((_HUD2D_VS, GL_VERTEX_SHADER),
                             (_HUD2D_FS, GL_FRAGMENT_SHADER))
        self._p_tex = _prog((_HUD2D_TEX_VS, GL_VERTEX_SHADER),
                             (_HUD2D_TEX_FS, GL_FRAGMENT_SHADER))

        # Quad VAO: 4 × (x, y)
        self._qvao = glGenVertexArrays(1)
        self._qvbo = glGenBuffers(1)
        glBindVertexArray(self._qvao)
        glBindBuffer(GL_ARRAY_BUFFER, self._qvbo)
        glBufferData(GL_ARRAY_BUFFER, 32, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glBindVertexArray(0)

        # Text-quad VAO: 4 × (x, y, u, v)
        self._tvao = glGenVertexArrays(1)
        self._tvbo = glGenBuffers(1)
        glBindVertexArray(self._tvao)
        glBindBuffer(GL_ARRAY_BUFFER, self._tvbo)
        glBufferData(GL_ARRAY_BUFFER, 64, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)

        # Bake label textures: white text on black → brightness-as-alpha in shader
        pygame.font.init()
        font = pygame.font.SysFont("monospace", 13)
        self._font = font
        self._text_cache: dict[str, tuple[int, tuple[int, int]]] = {}
        self._ltex = []
        self._ldim = []
        for label_str in [d[1] for d in self._DEFS]:
            surf = font.render(label_str, True, (255, 255, 255))
            w, h = surf.get_size()
            raw  = pygame.image.tobytes(surf, "RGBA")
            tex  = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, raw)
            glBindTexture(GL_TEXTURE_2D, 0)
            self._ltex.append(tex)
            self._ldim.append((w, h))

    # ── value ↔ slider-position helpers ──────────────────────────────────────

    def _t(self, idx: int) -> float:
        """Value → position [0,1]."""
        v, lo, hi = self.values[self.keys[idx]], self._lo[idx], self._hi[idx]
        if self._log[idx]:
            denom = math.log(max(hi, 1e-12)) - math.log(max(lo, 1e-12))
            t = (math.log(max(v, 1e-12)) - math.log(max(lo, 1e-12))) / denom if denom else 0.0
        else:
            t = (v - lo) / (hi - lo) if hi != lo else 0.0
        return float(np.clip(t, 0.0, 1.0))

    def _v(self, idx: int, t: float):
        """Position [0,1] → value."""
        lo, hi = self._lo[idx], self._hi[idx]
        t = float(np.clip(t, 0.0, 1.0))
        if self._log[idx]:
            return math.exp(math.log(max(lo, 1e-12)) +
                            t * (math.log(max(hi, 1e-12)) - math.log(max(lo, 1e-12))))
        return lo + t * (hi - lo)

    # ── layout helpers ────────────────────────────────────────────────────────

    def _row_y(self, row: int) -> int:
        return self.PY + 5 + row * self.ROW

    def _track_rect(self, row: int) -> tuple:
        return (self.PX + self.TX,
                self._row_y(row) + (self.ROW - self.TH) // 2,
                self.TW, self.TH)

    def _panel_h(self) -> int:
        return 5 + len(self.keys) * self.ROW + 4

    def _in_panel(self, mx: int, my: int) -> bool:
        return (self.PX <= mx <= self.PX + self.PW and
                self.PY <= my <= self.PY + self._panel_h())

    # ── event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev) -> bool:
        """Return True if the event was consumed by the panel."""
        if ev.type == MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            if not self._in_panel(mx, my):
                return False
            for i in range(len(self.keys)):
                tx, ty, tw, th = self._track_rect(i)
                if tx <= mx <= tx + tw and ty - 8 <= my <= ty + th + 8:
                    self._drag = i
                    self._apply_mouse(i, mx)
                    return True
            return True   # click in panel but not on a track — absorb it
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self._drag >= 0:
                self._drag = -1
                return True
        elif ev.type == MOUSEMOTION:
            if self._drag >= 0:
                self._apply_mouse(self._drag, ev.pos[0])
                return True
        return False

    def _apply_mouse(self, idx: int, mx: int):
        tx, _, tw, _ = self._track_rect(idx)
        t   = (mx - tx) / max(tw, 1)
        raw = self._v(idx, t)
        key = self.keys[idx]
        if   key == 'rays':     raw = int(np.clip(round(raw), 256, 50_000))
        elif key == 'segs':     raw = int(np.clip(round(raw), 30, 240))
        elif key == 'plate_th': raw = int(np.clip(round(raw), 64, 1024))
        elif key == 'dx':       raw = float(np.clip(raw, 0.004, 0.016))
        elif key == 'ray_exposure':
            raw = float(np.clip(raw, 0.25, 6.0))
        elif key == 'ray_gamma':
            raw = float(np.clip(raw, 0.20, 1.60))
        elif key in ('mic_gain', 'pickup_gain'):
            raw = float(np.clip(raw, 0.0, 1.0))
        old = self.values[key]
        self.values[key] = raw
        if raw != old and not self._live[idx]:
            self._pend[key] = True

    def _get_text_texture(self, text: str) -> tuple[int, tuple[int, int]]:
        cached = self._text_cache.get(text)
        if cached is not None:
            return cached
        surf = self._font.render(text, True, (255, 255, 255))
        w, h = surf.get_size()
        raw = pygame.image.tobytes(surf, "RGBA")
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glBindTexture(GL_TEXTURE_2D, 0)
        cached = (tex, (w, h))
        self._text_cache[text] = cached
        return cached

    # ── query helpers called by main() ────────────────────────────────────────

    def changed(self) -> dict:
        """Return {key: value} for sliders moved since the last call."""
        out = {k: self.values[k] for k in self.keys if self.values[k] != self._prev[k]}
        self._prev = dict(self.values)
        return out

    def mark_applied(self, key: str):
        self._pend[key] = False

    # ── GL draw ───────────────────────────────────────────────────────────────

    def _draw_quad(self, x, y, w, h, color, rw, rh):
        v = np.array([x, y, x+w, y, x, y+h, x+w, y+h], np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._qvbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, v.nbytes, v)
        glUseProgram(self._p_col)
        glUniform2f(glGetUniformLocation(self._p_col, b'uRes'), rw, rh)
        glUniform4f(glGetUniformLocation(self._p_col, b'uColor'), *color)
        glBindVertexArray(self._qvao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def _draw_label(self, idx: int, x, y, rw, rh):
        tex = self._ltex[idx]
        tw, th = self._ldim[idx]
        self._draw_textured(tex, tw, th, x, y, rw, rh)

    def _draw_textured(self, tex: int, tw: int, th: int, x, y, rw, rh):
        v = np.array([
            x,    y,    0., 0.,
            x+tw, y,    1., 0.,
            x,    y+th, 0., 1.,
            x+tw, y+th, 1., 1.,
        ], np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._tvbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, v.nbytes, v)
        glUseProgram(self._p_tex)
        glUniform2f(glGetUniformLocation(self._p_tex, b'uRes'), rw, rh)
        glUniform4f(glGetUniformLocation(self._p_tex, b'uColor'), *self._C_TEXT)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(glGetUniformLocation(self._p_tex, b'uTex'), 0)
        glBindVertexArray(self._tvao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def draw_progress(self, win_w: int, win_h: int, *, recorded_samples: int,
                      total_samples: int, replaying: bool, paused: bool,
                      pending_rebuild: bool, cached_frames: int):
        total_samples = max(1, int(total_samples))
        recorded_samples = int(np.clip(recorded_samples, 0, total_samples))
        frac = recorded_samples / float(total_samples)
        remaining = max(0, total_samples - recorded_samples)
        if pending_rebuild:
            state = "BUILD"
        elif replaying:
            state = "REPLAY"
        elif paused:
            state = "PAUSE"
        else:
            state = "REC"
        text = (
            f"{state} {_format_hud_time(recorded_samples)} / {_format_hud_time(total_samples)}"
            f"  left {_format_hud_time(remaining)}  cache {cached_frames:03d}"
        )
        tex, (tw, th) = self._get_text_texture(text)
        pad = 10
        panel_w = max(240, tw + pad * 2)
        panel_h = th + 18
        x = win_w - panel_w - 14
        y = win_h - panel_h - 14
        bar_x = x + pad
        bar_y = y + 5
        bar_w = panel_w - pad * 2
        fill_w = int(round(bar_w * frac))
        accent = (0.28, 0.82, 0.62, 0.95) if not pending_rebuild else self._C_PEND
        self._draw_quad(x, y, panel_w, panel_h, (0.03, 0.03, 0.06, 0.74), win_w, win_h)
        self._draw_quad(bar_x, bar_y, bar_w, 4, self._C_TRACK, win_w, win_h)
        if fill_w > 0:
            self._draw_quad(bar_x, bar_y, fill_w, 4, accent, win_w, win_h)
        self._draw_textured(tex, tw, th, x + pad, y + 10, win_w, win_h)

    def draw(self, win_w: int, win_h: int):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._draw_quad(self.PX, self.PY, self.PW, self._panel_h(),
                        self._C_BG, win_w, win_h)

        for i, key in enumerate(self.keys):
            ry = self._row_y(i)
            tx, ty, tw, th = self._track_rect(i)

            # Track background
            self._draw_quad(tx, ty, tw, th, self._C_TRACK, win_w, win_h)

            # Filled portion + knob color
            t    = self._t(i)
            fill = (self._C_LIVE if self._live[i]
                    else (self._C_PEND if self._pend[key] else self._C_BUILD))
            fw = max(2, int(t * tw))
            self._draw_quad(tx, ty, fw, th, fill, win_w, win_h)

            # Knob
            kx = tx + int(t * tw) - 4
            ky = ty + th // 2 - 6
            self._draw_quad(kx, ky, 8, 12, self._C_KNOB, win_w, win_h)

            # Label (vertically centred in row)
            _, lh = self._ldim[i]
            self._draw_label(i, self.PX + 2, ry + (self.ROW - lh) // 2,
                             win_w, win_h)

        glEnable(GL_DEPTH_TEST)
        glBindVertexArray(0)
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="3-D acoustic pluck visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trace-rays", type=int, default=RAY_TRACE_RAYS,
                   help="Rays/source for the line-segment overlay")
    p.add_argument("--max-bounces", type=int, default=RAY_MAX_BOUNCES,
                   help="Maximum ray bounces")
    p.add_argument("--gpu-rays", action=argparse.BooleanOptionalAction,
                   default=USE_GPU_RAY_FIELD,
                   help="Use OpenGL compute shader to build a rotatable 3-D ray field")
    p.add_argument("--dx", type=float, default=0.008,
                   help="FDTD grid spacing in metres; smaller means higher pressure/plate physics resolution")
    p.add_argument("--pressure-margin-cells", "--margin-cells", dest="pressure_margin_cells",
                   type=int, default=max(0, PAD_CELLS - N_PML),
                   help="Free-air pressure FDTD margin cells around the instrument on each side, before the PML starts")
    p.add_argument("--pressure-pml-cells", "--pml-cells", dest="pressure_pml_cells",
                   type=int, default=N_PML,
                   help="Pressure FDTD PML thickness in cells at the outer grid boundary")
    p.add_argument("--render-segs", type=int, default=60,
                   help="String FDTD/render segments per string")
    p.add_argument("--bridge-force-scale", type=float, default=BRIDGE_FORCE_SCALE,
                   help="Deprecated compatibility argument; structural coupling uses SI impedances")
    p.add_argument("--plate-theta", type=int, default=128,
                   help="Angular subdivisions for the rendered soundboard mesh")
    p.add_argument("--plate-radial", type=int, default=PLATE_RADIAL_SEGS,
                   help="Radial subdivisions for the rendered soundboard mesh")
    p.add_argument("--gpu-field-rays", type=int, default=1000,
                   help="GPU rays/source for the 3-D ray field")
    p.add_argument("--gpu-segment-cap", type=int, default=GPU_RAY_SEGMENT_CAP,
                   help="Maximum GPU ray line segments kept for the overlay")
    p.add_argument("--gpu-dispatch-batch", type=int, default=GPU_DISPATCH_BATCH,
                   help="Rays per GPU compute dispatch; lower values avoid driver-side transient allocation/TDR")
    p.add_argument("--gpu-refresh-every", type=int, default=1,
                   help="Recompute GPU ray field every N rendered physics frames from current plate state; 0 keeps the initial static field")
    p.add_argument("--gpu-refresh-rays", type=int, default=0,
                   help="Rays used for each dynamic refresh; 0 reuses --gpu-field-rays")
    p.add_argument("--stage-light-field", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Build a cached diffuse room/stage light field before rendering")
    p.add_argument("--stage-light-rays", type=int, default=STAGE_LIGHT_RAYS,
                   help="Rays for the cached diffuse stage light field")
    p.add_argument("--stage-light-emitters", type=int, default=9,
                   help="Emitter samples across the diffuse stage light area")
    p.add_argument("--stage-light-cache-dir", default=".cache",
                   help="Directory for cached room/stage light fields")
    p.add_argument("--rebuild-stage-light", action="store_true",
                   help="Ignore any cached room/stage light field and recompute it")
    p.add_argument("--initial-ray-field", action="store_true",
                   help="Build the old static spectral ray field before the render loop")
    p.add_argument("--cpu-ray-overlay", action="store_true",
                   help="Use CPU ray tracing for the line overlay instead of GPU segments")
    p.add_argument("--ray-only-view", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Start with only ray field and ray lines visible")
    p.add_argument("--gpu-smoke-exit", action="store_true",
                   help="Build the GPU ray field once and exit before the render loop")
    p.add_argument("--excitation", choices=("strum", "a-pluck", "rest"), default="strum",
                   help="Excitation rendered into the repeatable animation")
    p.add_argument("--diagnostic-rest", action="store_true",
                   help="Run from rest with no plucks or injected performance events")
    p.add_argument("--diagnostic-frames", type=int, default=600,
                   help="Frame budget for --diagnostic-rest / --excitation rest")
    p.add_argument("--equilibrium-report-every", type=int, default=30,
                   help="Print rest/equilibrium statistics every N streamed frames; 0 disables")
    p.add_argument("--capture-wav", nargs="?", const="mic_capture.wav",
                   default="mic_capture.wav",
                   help="Write simulated mic/pickup capture to this WAV file; empty disables")
    p.add_argument("--capture-source", choices=("mic", "pickup", "both", "mix"),
                   default="both",
                   help="Which transducer output to write")
    p.add_argument("--pickup", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable pickup output capture and pickup mesh marker")
    p.add_argument("--mic", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable acoustic microphone capture and mic mesh marker")
    p.add_argument("--fret", type=int, default=0,
                   help="Stopped fret number. 0 leaves the open scale length")
    p.add_argument("--fretless", action=argparse.BooleanOptionalAction, default=False,
                   help="Use a damped fretless/finger stop for --fret instead of a hard fret")
    args = p.parse_args()
    if args.pressure_margin_cells < 0:
        p.error("--pressure-margin-cells must be non-negative")
    if args.pressure_pml_cells < 0:
        p.error("--pressure-pml-cells must be non-negative")
    if args.diagnostic_rest:
        args.excitation = "rest"
    if args.diagnostic_frames < 1:
        p.error("--diagnostic-frames must be positive")
    return args


def _scene_outline_or_default(scene) -> tuple[np.ndarray, float]:
    if scene is not None and _HAS_BRIDGE:
        try:
            outline, body_h, _ = _extract_guitar_geometry(scene)
            if outline is not None and len(outline) >= 8:
                return np.asarray(outline, dtype=np.float32), float(body_h or BODY_H)
        except Exception:
            pass
    return _guitar_outline(), BODY_H


def main():
    args = _parse_args()
    pygame.init()
    pygame.display.set_mode((WIN_W, WIN_H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Guitar — FDTD 3-D Visualiser")

    config = {
        "n_strings": 6,
        "dx": float(args.dx),
        "pressure_margin_cells": int(args.pressure_margin_cells),
        "pressure_pml_cells": int(args.pressure_pml_cells),
        "render_segs": int(args.render_segs),
        "bridge_force_scale": float(args.bridge_force_scale),
        "excitation": str(args.excitation),
        "diagnostic_frames": int(args.diagnostic_frames),
        "fret": max(0, int(args.fret)),
        "fretless": bool(args.fretless),
    }

    # ── Build physics in a worker process ────────────────────────────────────
    print("Starting physics worker ...", flush=True)
    physics = _PhysicsProcess(config)
    ready = None
    while ready is None:
        for msg in physics.poll():
            if msg.get("type") == "ready":
                ready = msg
                break
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("message", "physics worker failed"))
        pygame.event.pump()
        clock_wait = pygame.time.Clock()
        clock_wait.tick(30)

    info = ready["info"]
    body_h = float(ready["body_h"])
    n_str = int(ready["n_strings"])
    outline = np.asarray(ready["outline"], dtype=np.float32)
    print(f"  Grid {info['Nx']}×{info['Ny']}×{info['Nz']}  "
          f"body_h={body_h:.3f}m", flush=True)
    scene = _build_body_scene_fn("string_plate") if _HAS_SCENE else None

    paths = _string_paths(outline, body_h, n_strings=n_str,
                          n_segs=args.render_segs,
                          fret=config["fret"])

    ray_field_bands = None
    ray_field_bounds = None
    ray_field_total_rays = 0
    baseline_light_bands = None
    baseline_light_bounds = None
    baseline_light_total_rays = 0
    gpu_seg_vbo = None
    gpu_seg_cap = 0
    if args.excitation == "rest":
        active_strings = set()
    elif args.excitation == "a-pluck":
        active_strings = {min(A_STRING_INDEX, max(0, n_str - 1))}
    else:
        active_strings = None
    total       = _excitation_total_samples(args.excitation, args.diagnostic_frames)
    total_frames = int(math.ceil(total / float(BLOCK_SAMPLES)))
    if scene is not None and args.gpu_rays and args.stage_light_field:
        try:
            light_key = _stage_light_cache_key(
                scene, outline, body_h,
                int(args.stage_light_rays),
                int(args.stage_light_emitters),
                max(1, int(args.max_bounces)),
                STAGE_LIGHT_DIMS)
            light_cache_path = _stage_light_cache_path(args.stage_light_cache_dir, light_key)
            cached = None if args.rebuild_stage_light else _load_stage_light_cache(light_cache_path)
            if cached is not None:
                baseline_light_bands, baseline_light_bounds, baseline_light_total_rays = cached
                print(f"  [cache] stage light field hit: {light_cache_path}", flush=True)
            else:
                print("Computing cached diffuse stage light field ...", flush=True)
            gm, _ = _guitar_model_matrix(outline)
            lb0, lb1 = _stage_light_bounds()
            if cached is None:
                (baseline_light_bands, b_lb0, b_lb1, base_vbo, base_cap,
                 base_counter, baseline_light_total_rays) = _gpu_ray_field(
                    scene, outline, body_h,
                    sources=_stage_light_sources(args.stage_light_rays, args.stage_light_emitters),
                    max_bounces=max(1, args.max_bounces),
                    dims=STAGE_LIGHT_DIMS,
                    segment_cap=0,
                    dispatch_batch=args.gpu_dispatch_batch,
                    include_stage=True,
                    model_matrix=gm,
                    bounds=(lb0, lb1))
                baseline_light_bounds = (b_lb0, b_lb1)
                _save_stage_light_cache(
                    light_cache_path,
                    baseline_light_bands,
                    STAGE_LIGHT_DIMS,
                    baseline_light_bounds,
                    baseline_light_total_rays)
                if base_vbo is not None:
                    glDeleteBuffers(1, [base_vbo])
                if base_counter is not None:
                    glDeleteBuffers(1, [base_counter])
        except BaseException as exc:
            _report_exception("cached stage light field", exc)
            raise
    elif args.stage_light_field:
        print("Cached stage light field skipped: GPU ray field unavailable.", flush=True)
    if scene is not None and args.gpu_rays and args.initial_ray_field:
        print("Initial spectral prepass is skipped in decoupled mode; "
              "dynamic ray refresh will use streamed worker frames.", flush=True)
    if False:
        # ── Optional static spectral prepass: FFT plate displacement → ray field ─
        print("Computing static spectral plate emission map (8192 steps) ...", flush=True)
        _bm_path  = _band_maps_cache_path(
            args.dx, args.render_segs, args.pressure_margin_cells, args.pressure_pml_cells
        )
        band_maps = _try_load_band_maps(_bm_path)
        if band_maps is None:
            ce.reset()
            _schedule_excitation(ce, args.excitation, n_strings=n_str)
            band_maps = _compute_spectral_emission_map(ce)
            _save_band_maps(band_maps, _bm_path)
            ce.reset()
            if hasattr(ce, "clear_pluck_schedule"):
                ce.clear_pluck_schedule()
        else:
            print("  [cache] spectral map hit — skipping 8192-step prepass", flush=True)
        soundboard_sources = _sample_soundboard_sources(
            band_maps,
            info['plate_active_2d'],
            outline, body_h,
            total_rays=args.gpu_field_rays,
            stride=4)
        string_sources = _string_emission_sources(
            paths, body_h,
            total_rays_per_string=max(256, args.gpu_field_rays // max(1, n_str) // 8),
            active_strings=active_strings)
        all_sources = soundboard_sources + string_sources
        print(f"  {len(soundboard_sources)} soundboard + "
              f"{len(string_sources)} string emission sources", flush=True)

        ce.reset()
        _schedule_excitation(ce, args.excitation, n_strings=n_str)

        print("GPU ray field integration ...", flush=True)
        try:
            ray_field_bands, rb0, rb1, gpu_seg_vbo, gpu_seg_cap, _gpu_counter, ray_field_total_rays = _gpu_ray_field(
                scene, outline, body_h,
                sources=all_sources,
                max_bounces=args.max_bounces,
                segment_cap=args.gpu_segment_cap,
                dispatch_batch=args.gpu_dispatch_batch)
        except BaseException as exc:
            _report_exception("initial GPU ray field", exc)
            raise
        ray_field_bounds = (rb0, rb1)
        if args.gpu_smoke_exit:
            _quit_pygame_with_diag("initial-gpu-smoke-exit")
            return
    elif args.initial_ray_field:
        print("GPU ray field prepass skipped: scene/GPU ray field unavailable.", flush=True)

    if args.gpu_smoke_exit:
        if scene is not None and args.gpu_rays:
            print("GPU dynamic ray field smoke frame ...", flush=True)
            frame = None
            while frame is None:
                for msg in physics.poll():
                    if msg.get("type") == "frame":
                        frame = msg["frame"]
                        break
                    if msg.get("type") == "error":
                        raise RuntimeError(msg.get("message", "physics worker failed"))
                pygame.event.pump()
            if args.excitation == "rest":
                stats = _frame_equilibrium_stats(frame)
                print(
                    f"[equilibrium smoke] "
                    f"p_rms={stats[0]:.3e} p_max={stats[1]:.3e} "
                    f"plate_rms={stats[2]:.3e} plate_max={stats[3]:.3e} "
                    f"string_max={stats[4]:.3e}",
                    flush=True)
            dyn_sources = _instant_soundboard_sources(
                frame.plate, info['plate_active_2d'], outline, body_h,
                total_rays=int(args.gpu_refresh_rays or args.gpu_field_rays),
                stride=4)
            dyn_sources += _string_emission_sources(
                paths, body_h,
                total_rays_per_string=max(
                    64,
                    int(args.gpu_refresh_rays or args.gpu_field_rays) // max(1, n_str) // 16),
                active_strings=active_strings)
            if dyn_sources:
                try:
                    rbands, rb0, rb1, rvbo, rcap, rcounter, _rtotal = _gpu_ray_field(
                        scene, outline, body_h,
                        sources=dyn_sources,
                        max_bounces=args.max_bounces,
                        segment_cap=args.gpu_segment_cap,
                        dispatch_batch=args.gpu_dispatch_batch)
                except BaseException as exc:
                    _report_exception("GPU smoke ray field", exc)
                    raise
                for tex in rbands or []:
                    glDeleteTextures([tex])
                if rvbo is not None:
                    glDeleteBuffers(1, [rvbo])
                if rcounter is not None:
                    glDeleteBuffers(1, [rcounter])
            else:
                print("GPU dynamic ray field smoke frame produced no sources.", flush=True)
        physics.close()
        _quit_pygame_with_diag("dynamic-gpu-smoke-exit")
        return

    # ── Ray trace (once, explicit opt-in only) ────────────────────────────────
    ray_segs = None
    if scene is not None and _HAS_RAY and args.cpu_ray_overlay:
        print("Tracing geometry ...", flush=True)
        try:
            _ray_diag_update(
                "cpu_ray_overlay:trace:start",
                n_rays=int(args.trace_rays),
                max_bounces=int(args.max_bounces),
            )
            ray_segs, meta = _trace_fn(
                scene, n_rays=args.trace_rays, max_bounces=args.max_bounces)
            _ray_diag_update(
                "cpu_ray_overlay:trace:done",
                n_segments=int(len(ray_segs)),
                meta={k: str(v) for k, v in list(meta.items())[:8]},
            )
        except BaseException as exc:
            _report_exception("CPU ray overlay trace", exc)
            raise
        print(f"  {len(ray_segs)} segments", flush=True)

    # ── Renderer ──────────────────────────────────────────────────────────────
    R = Renderer(WIN_W, WIN_H, outline=outline, info=info, body_h=body_h,
                 bridge_pos=BRIDGE_POS, str_paths=paths,
                 ray_segs=ray_segs,
                 ray_field_bands=ray_field_bands,
                 ray_field_bounds=ray_field_bounds,
                 ray_field_total_rays=ray_field_total_rays,
                 baseline_light_bands=baseline_light_bands,
                 baseline_light_bounds=baseline_light_bounds,
                 baseline_light_total_rays=baseline_light_total_rays,
                 gpu_seg_vbo=gpu_seg_vbo,
                 gpu_seg_cap=gpu_seg_cap,
                 max_cached_frames=total_frames + 8,
                 plate_theta=args.plate_theta,
                 plate_radial=args.plate_radial,
                 active_fret=config["fret"],
                 fretless=config["fretless"],
                 show_pickup=args.pickup,
                 show_mic=args.mic)
    if not args.ray_only_view:
        R._layers = [LAYER_ALPHA, LAYER_OPAQUE, LAYER_OPAQUE,
                     LAYER_OPAQUE, LAYER_OPAQUE, LAYER_ALPHA, LAYER_ALPHA,
                     LAYER_ALPHA]   # 8th = illum

    # Transfer per-triangle ray-lit illumination if a CPU ray trace was run.
    try:
        R.set_ray_lighting(meta)
    except NameError:
        pass  # meta not defined (no CPU ray overlay was run)

    panel = _SliderPanel()
    # Sync panel default values with whatever args resolved to
    panel.values['rays']     = int(args.gpu_field_rays or 1000)
    panel.values['segs']     = int(args.render_segs)
    panel.values['plate_th'] = int(args.plate_theta)
    panel.values['dx']       = float(args.dx)
    panel.values['ray_exposure'] = 1.0
    panel.values['ray_gamma'] = float(GPU_RAY_FIELD_GAMMA)
    panel.values['mic_gain'] = 1.0 if args.mic else 0.0
    panel.values['pickup_gain'] = 1.0 if args.pickup else 0.0
    panel._prev = dict(panel.values)
    R.set_ray_tonemap(
        exposure=float(panel.values['ray_exposure']),
        gamma=float(panel.values['ray_gamma']))

    print(f"Physics worker is streaming frames ({physics.mode}); "
          "UI will keep cached frames while rebuilds run.", flush=True)
    if args.excitation == "rest":
        print("Diagnostic rest mode: no plucks are scheduled; reporting equilibrium drift.",
              flush=True)

    clock       = pygame.time.Clock()
    running     = True
    fi          = 0
    recorded_samples = 0
    replaying   = False
    dragging    = False
    last_mouse  = (0, 0)
    refresh_every = max(0, int(args.gpu_refresh_every))
    refresh_rays = int(args.gpu_refresh_rays or args.gpu_field_rays)
    force_ray_refresh = False
    pending_rebuild = False
    active_ray_frame_index = None
    last_displayed_frame_index = None
    equilibrium_report_every = max(0, int(args.equilibrium_report_every))
    rest_stats0 = None
    capture = None
    if args.capture_wav:
        capture = _CaptureSet(args.capture_wav, args.capture_source)

    print("1-7 toggle layers | P plate mode | SPACE pause | R restart | "
          f"Q quit | drag=orbit wheel=zoom | excitation={args.excitation}", flush=True)

    def _refresh_ray_field_from_frame(frame: Frame | None, frame_index: int, reason: str) -> bool:
        nonlocal active_ray_frame_index, force_ray_refresh
        if frame is None or scene is None or not args.gpu_rays or refresh_every <= 0:
            return False
        if frame_index < 0:
            return False
        if not force_ray_refresh and (frame_index % refresh_every) != 0:
            return False
        dyn_sources = _instant_soundboard_sources(
            frame.plate, info['plate_active_2d'], outline, body_h,
            total_rays=refresh_rays, stride=4)
        dyn_sources += _string_emission_sources(
            paths, body_h,
            total_rays_per_string=max(64, refresh_rays // max(1, n_str) // 16),
            active_strings=active_strings)
        if not dyn_sources:
            force_ray_refresh = False
            return False
        print(f"Refreshing GPU ray field from cached frame {frame_index} [{reason}] "
              f"({sum(s[2] for s in dyn_sources)} rays)", flush=True)
        try:
            rbands, rb0, rb1, rvbo, rcap, rcounter, dyn_total_rays = _gpu_ray_field(
                scene, outline, body_h,
                sources=dyn_sources,
                max_bounces=args.max_bounces,
                segment_cap=args.gpu_segment_cap,
                dispatch_batch=args.gpu_dispatch_batch)
        except BaseException as exc:
            _report_exception(f"dynamic GPU ray field frame {frame_index}", exc)
            raise
        try:
            R.replace_ray_field(rbands, (rb0, rb1), rvbo, rcap, rcounter,
                                total_rays=dyn_total_rays)
        except BaseException as exc:
            _report_exception(f"dynamic GPU ray field replace frame {frame_index}", exc)
            raise
        active_ray_frame_index = frame_index
        force_ray_refresh = False
        return True

    try:
      while running:
        for ev in pygame.event.get():
            if panel.handle_event(ev):
                continue
            if ev.type == QUIT:
                running = False
            elif ev.type == KEYDOWN:
                if ev.key == K_q:
                    running = False
                elif ev.key == K_SPACE:
                    R._paused = not R._paused
                elif ev.key == K_r:
                    new_segs = int(panel.values['segs'])
                    new_dx   = float(panel.values['dx'])
                    if new_segs != args.render_segs or abs(new_dx - args.dx) > 1e-6:
                        print(f"Queueing physics rebuild (segs={new_segs}, "
                              f"dx={new_dx:.4f}, margin={args.pressure_margin_cells} cells, "
                              f"pml={args.pressure_pml_cells} cells) ...", flush=True)
                        config.update({"render_segs": new_segs, "dx": new_dx})
                        physics.send({"type": "reconfigure", "config": dict(config)})
                        pending_rebuild = True
                    else:
                        physics.send({"type": "restart"})
                    R._frames.clear(); R._cursor = 0; fi = 0
                    recorded_samples = 0
                    for env in R._str_env: env[:] = 0.0
                    replaying = False
                    force_ray_refresh = refresh_every > 0
                    active_ray_frame_index = None
                    last_displayed_frame_index = None
                    print("Restart requested")
                elif ev.key == K_p:
                    print(f"Plate mode: {R.cycle_plate_mode()}")
                else:
                    for ki, kv in enumerate(LAYER_KEYS):
                        if ev.key == kv:
                            R.toggle(ki)
                            s = ['OPAQUE','ALPHA','HIDDEN'][R._layers[ki]]
                            print(f"Layer {ki+1} ({LAYER_NAMES[ki]}): {s}")
            elif ev.type == MOUSEBUTTONDOWN and ev.button == 1:
                dragging = True; last_mouse = ev.pos
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == MOUSEMOTION and dragging:
                dx, dy = ev.pos[0]-last_mouse[0], ev.pos[1]-last_mouse[1]
                R.cam.orbit(dx*0.35, dy*0.25); R.cam._auto = 0.0
                last_mouse = ev.pos
            elif ev.type == MOUSEWHEEL:
                R.cam.zoom(-ev.y * 0.04)

        for msg in physics.poll():
            mtype = msg.get("type")
            if mtype == "error":
                raise RuntimeError(msg.get("message", "physics worker failed"))
            if mtype == "rebuilding":
                pending_rebuild = True
                continue
            if mtype == "ready":
                info = msg["info"]
                body_h = float(msg["body_h"])
                n_str = int(msg["n_strings"])
                outline = np.asarray(msg["outline"], dtype=np.float32)
                args.render_segs = int(msg["config"]["render_segs"])
                args.dx = float(msg["config"]["dx"])
                paths = _string_paths(outline, body_h, n_strings=n_str,
                                      n_segs=args.render_segs,
                                      fret=int(msg["config"].get("fret", 0)))
                total = _excitation_total_samples(args.excitation, args.diagnostic_frames)
                total_frames = int(math.ceil(total / float(BLOCK_SAMPLES)))
                R._frames = collections.deque(maxlen=max(1, total_frames + 8))
                R.rebuild_physics(None, info, paths, body_h,
                                  active_fret=int(msg["config"].get("fret", 0)),
                                  fretless=bool(msg["config"].get("fretless", False)))
                panel.mark_applied('segs')
                panel.mark_applied('dx')
                pending_rebuild = False
                replaying = False
                rest_stats0 = None
                fi = 0
                recorded_samples = 0
                active_ray_frame_index = None
                last_displayed_frame_index = None
                print(f"  Worker ready: Grid {info['Nx']}×{info['Ny']}×{info['Nz']}",
                      flush=True)
                continue
            if mtype == "done":
                if not replaying:
                    R._cursor = 0
                    replaying = True
                    recorded_samples = total
                    active_ray_frame_index = None
                    last_displayed_frame_index = None
                    print(f"Cached {len(R._frames)} frames for repeatable replay", flush=True)
                continue
            if mtype != "frame":
                continue
            frame = msg["frame"]
            fi = int(msg.get("index", fi))
            frame.frame_index = fi
            _record_diag(frame, fi)
            R.push(frame)
            recorded_samples = min(total, max(recorded_samples, (fi + 1) * BLOCK_SAMPLES))
            if args.excitation == "rest":
                stats = _frame_equilibrium_stats(frame)
                if rest_stats0 is None:
                    rest_stats0 = stats
                if equilibrium_report_every and (fi == 0 or fi % equilibrium_report_every == 0):
                    dp = stats[0] - rest_stats0[0]
                    dpl = stats[2] - rest_stats0[2]
                    print(
                        f"[equilibrium {fi:05d}] "
                        f"p_rms={stats[0]:.3e} p_max={stats[1]:.3e} "
                        f"plate_rms={stats[2]:.3e} plate_max={stats[3]:.3e} "
                        f"string_max={stats[4]:.3e} "
                        f"drift(p_rms={dp:+.2e}, plate_rms={dpl:+.2e})",
                        flush=True)
            if capture is not None:
                mic = np.asarray(frame.mic if frame.mic is not None else np.zeros(BLOCK_SAMPLES), np.float32)
                pickup = np.asarray(frame.pickup if frame.pickup is not None else np.zeros(BLOCK_SAMPLES), np.float32)
                mic *= float(panel.values.get("mic_gain", 1.0))
                pickup *= float(panel.values.get("pickup_gain", 1.0))
                capture.write(mic, pickup)
            if scene is not None and args.gpu_rays and refresh_every > 0:
                do_refresh = force_ray_refresh or (fi % refresh_every == 0)
                if do_refresh:
                    _refresh_ray_field_from_frame(frame, fi, "stream")
            fi += 1

        # ── Apply live slider changes ──────────────────────────────────────────
        changed = panel.changed()
        if 'rays' in changed:
            refresh_rays = int(panel.values['rays'])
            if refresh_every > 0:
                force_ray_refresh = True
        if 'plate_th' in changed:
            R.rebuild_plate(int(panel.values['plate_th']), R._plate_radial)
            panel.mark_applied('plate_th')
        if 'ray_exposure' in changed or 'ray_gamma' in changed:
            R.set_ray_tonemap(
                exposure=float(panel.values['ray_exposure']),
                gamma=float(panel.values['ray_gamma']))
        if 'mic_gain' in changed:
            R.show_mic = panel.values['mic_gain'] > 0.0
        if 'pickup_gain' in changed:
            R.show_pickup = panel.values['pickup_gain'] > 0.0

        try:
            _ray_diag_update(
                "render:frame",
                frame_index=int(fi),
                ray_textures=tuple(int(t) for t in (R._ray_field_bands or [])),
                gpu_seg_vbo=int(R._gpu_seg_vbo or 0),
                gpu_seg_cap=int(R._gpu_seg_cap or 0),
                ray_draw_vertices=int(getattr(R, "_ray_n", 0)),
                cached_frames=int(len(R._frames)),
            )
            R.tick()
            cur_frame = R.current_frame()
            cur_frame_index = int(getattr(cur_frame, 'frame_index', -1)) if cur_frame is not None else -1
            if (replaying or R._paused) and cur_frame_index != last_displayed_frame_index:
                target_ray_frame_index = cur_frame_index
                if target_ray_frame_index >= 0 and refresh_every > 0:
                    target_ray_frame_index = (target_ray_frame_index // refresh_every) * refresh_every
                if target_ray_frame_index != active_ray_frame_index:
                    refresh_frame = cur_frame
                    if target_ray_frame_index != cur_frame_index:
                        refresh_frame = next(
                            (cached for cached in reversed(R._frames)
                             if int(getattr(cached, 'frame_index', -1)) == target_ray_frame_index),
                            None)
                    _refresh_ray_field_from_frame(refresh_frame, target_ray_frame_index, "replay")
                last_displayed_frame_index = cur_frame_index
            R.render()
            panel.draw(WIN_W, WIN_H)
            panel.draw_progress(
                WIN_W, WIN_H,
                recorded_samples=max(recorded_samples, min(total, len(R._frames) * BLOCK_SAMPLES)),
                total_samples=total,
                replaying=replaying,
                paused=R._paused,
                pending_rebuild=pending_rebuild,
                cached_frames=len(R._frames))
            pygame.display.flip()
        except BaseException as exc:
            _report_exception(f"render frame {fi}", exc)
            raise
        clock.tick(60)
    finally:
        if capture is not None:
            capture.close()
            print(f"Wrote capture: {', '.join(capture.paths)}", flush=True)
        physics.close()

    _quit_pygame_with_diag("main-exit")


if __name__ == '__main__':
    _install_crash_reporting()
    try:
        main()
    except SystemExit as exc:
        raise
    except BaseException as exc:
        _report_exception("top-level", exc)
        raise
