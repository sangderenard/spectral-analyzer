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
import math
import sys
import collections
import argparse
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
        K_1, K_2, K_3, K_4, K_5, K_6, K_7,
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
        GL_UNSIGNED_INT, GL_VERTEX_SHADER,
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
        glViewport, glLineWidth,
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
    )
except ImportError:
    print("PyOpenGL not available"); sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WIN_W, WIN_H  = 1400, 900
SAMPLE_RATE   = 44100
BLOCK_SAMPLES = 512
MAX_FRAMES    = 240

LAYER_OPAQUE, LAYER_ALPHA, LAYER_HIDDEN = 0, 1, 2
N_LAYERS    = 7
LAYER_KEYS  = [K_1, K_2, K_3, K_4, K_5, K_6, K_7]
LAYER_NAMES = ['body', 'plate', 'pressure', 'strings', 'markers', 'ray-segs', 'stage']

DX        = 0.004
N_PML     = 10
PAD_CELLS = 10

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

PLATE_SCALE  = 0.012
STRING_DISP_SCALE = 15.0   # amplify string displacement for physical visibility
ENV_DECAY         = 0.985  # per-frame peak-hold decay for envelope mode (~60 fps)
STRING_CLEARANCE = 0.010

STRING_COLORS = [
    (1.00, 0.75, 0.15, 1.0),
    (0.90, 0.85, 0.40, 1.0),
    (0.30, 0.90, 0.40, 1.0),
    (0.20, 0.85, 1.00, 1.0),
    (0.45, 0.50, 1.00, 1.0),
    (1.00, 0.30, 0.70, 1.0),
]

RAY_BOUNCE_RGBA = np.array([
    [1.00, 0.90, 0.20, 0.80],
    [1.00, 0.50, 0.10, 0.65],
    [0.90, 0.10, 0.10, 0.50],
    [0.70, 0.20, 0.90, 0.40],
], dtype=np.float32)


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


def _string_paths(outline: np.ndarray, body_h: float,
                  n_strings: int = 6, n_segs: int = N_RENDER_SEGS) -> List[np.ndarray]:
    """Matches build_acoustic_coevolver_from_scene string layout exactly."""
    y_saddle = -0.070   # bridge saddle y (matches BRIDGE_POS)
    y_nut    = y_saddle + SCALE_LENGTH_M
    x_span   = 0.0088 * max(0, n_strings - 1)   # ~8.8 mm per gap → 44 mm for 6
    string_z = body_h + STRING_CLEARANCE
    paths = []
    for si in range(n_strings):
        x = -x_span / 2 + si * (x_span / max(1, n_strings - 1))
        ys   = np.linspace(y_nut, y_saddle, n_segs + 1, dtype=np.float32)
        path = np.column_stack([
            np.full(n_segs + 1, x,      dtype=np.float32),
            ys,
            np.full(n_segs + 1, string_z, dtype=np.float32),
        ])
        paths.append(path)
    return paths


def _neck_geometry(outline: np.ndarray, body_h: float, n_strings: int = 6):
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
    for fret in range(1, 13):
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
            [x_saddle, BRIDGE_POS[1][1], body_h + STRING_CLEARANCE],
            [x_nut, y_nut, body_h + STRING_CLEARANCE],
            [x_tune, y_tune, body_h + STRING_CLEARANCE + 0.003],
        ], np.float32))
        pin_lines.extend([[x_tune - 0.010, y_tune, z + 0.008],
                          [x_tune + 0.010, y_tune, z + 0.008]])

    return wood, np.asarray(fret_lines, np.float32), np.asarray(pin_lines, np.float32), ext_strings


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
                   n_segs=N_RENDER_SEGS):
    """Returns (scene, ce, info, body_h) or (None, None, fallback_info, BODY_H)."""
    if not (_HAS_PHYSICS and _HAS_BRIDGE and _HAS_SCENE):
        return None, None, None, None

    scene = _build_body_scene_fn("string_plate")
    if scene is None:
        return None, None, None, None

    ce, info = build_acoustic_coevolver_from_scene(
        scene, n_strings=n_strings, sample_rate=float(SAMPLE_RATE),
        dx=dx, pad_cells=pad_cells, n_pml=n_pml, n_segs=n_segs)
    if ce is None:
        return scene, None, info, None

    # Extract body_h from scene geometry
    _, body_h, _ = _extract_guitar_geometry(scene)
    body_h = body_h or BODY_H

    # Get plate_active for rendering mesh (re-voxelise with same params)
    outline_for_vox, _, _ = _extract_guitar_geometry(scene)
    if outline_for_vox is None or len(outline_for_vox) < 8:
        outline_for_vox = _guitar_outline()

    _, plate_active, _ = voxelise_guitar_body(
        outline_for_vox, body_h, dx=dx, pad_cells=pad_cells, n_pml=n_pml)
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
    if kind == "a-pluck":
        _schedule_a_pluck(ce, n_strings=n_strings)
    else:
        _schedule_strum(ce, n_strings=n_strings)


def _excitation_total_samples(kind: str) -> int:
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
    strings:  List[np.ndarray] # [n_strings] each (N_RENDER_SEGS, 3)


def physics_frame(ce, n_strings: int, n_render_segs: int = N_RENDER_SEGS) -> Frame:
    ce.step(BLOCK_SAMPLES)
    return Frame(
        pressure = ce.get_pressure_field(),
        plate    = ce.get_plate_displacement(),
        strings  = [ce.get_string_displacement_n(si, n_render_segs)
                    for si in range(n_strings)],
    )


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


def _stage_mesh(outline: np.ndarray, body_h: float):
    min_xy = outline.min(axis=0).astype(np.float32)
    max_xy = outline.max(axis=0).astype(np.float32)
    cx = float((min_xy[0] + max_xy[0]) * 0.5)
    cy = float((min_xy[1] + max_xy[1]) * 0.5)
    z_floor = -0.030
    z_ceil = body_h + 0.55
    x0, x1 = cx - 0.62, cx + 0.62
    y0, y1 = cy - 0.72, cy + 0.72
    verts, norms, mats = [], [], []

    def quad(a, b, c, d, n, mat):
        verts.extend([[a, b, c], [a, c, d]])
        norms.extend([n, n])
        mats.extend([mat, mat])

    diff_wall = [0.48, 0.82, 0.20, 0.48, 0.82, 0.20]
    diff_floor = [0.42, 0.78, 0.24, 0.42, 0.78, 0.24]
    quad([x0,y0,z_floor], [x1,y0,z_floor], [x1,y1,z_floor], [x0,y1,z_floor],
         [0,0,1], diff_floor)
    quad([x0,y1,z_floor], [x1,y1,z_floor], [x1,y1,z_ceil], [x0,y1,z_ceil],
         [0,-1,0], diff_wall)
    quad([x0,y0,z_floor], [x0,y1,z_floor], [x0,y1,z_ceil], [x0,y0,z_ceil],
         [1,0,0], diff_wall)
    quad([x1,y1,z_floor], [x1,y0,z_floor], [x1,y0,z_ceil], [x1,y1,z_ceil],
         [-1,0,0], diff_wall)
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

void main() {
    // Reconstruct world-space ray from NDC
    vec4 nh = uInvMVP * vec4(vNDC, -1.0, 1.0);
    vec4 fh = uInvMVP * vec4(vNDC,  1.0, 1.0);
    vec3 ro = nh.xyz / nh.w;
    vec3 rd = normalize(fh.xyz / fh.w - ro);

    // Slab / AABB intersection
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
            float raw = texture(uPressure, uvc).r;
            float v   = raw * uPressureScale;
            p = sign(v) * pow(clamp(abs(v), 0.0, 1.0), max(uPressureGamma, 0.05));
            spectral_color = diverge(p);
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
    int steps = max(1, int(len / 0.003));  // ~3 mm samples
    vec4 s_per = spectrum / float(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        float t = float(i) / float(steps);
        splat_spectral(mix(a, b, t), s_per);
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
    N_BANDS, Nh, Nw = band_maps.shape
    min_xy = outline.min(axis=0).astype(np.float32)
    max_xy = outline.max(axis=0).astype(np.float32)
    xs = np.linspace(float(min_xy[0]), float(max_xy[0]), Nw)
    ys = np.linspace(float(min_xy[1]), float(max_xy[1]), Nh)
    _z  = float(body_h) - 0.001
    _dn = np.array([0.0, 0.0, -1.0], np.float32)

    pts, weights, spectra = [], [], []
    for j in range(0, Nh, stride):
        for i in range(0, Nw, stride):
            if not plate_active[j, i]:
                continue
            spec = band_maps[:, j, i]
            total_e = float(spec.sum())
            if total_e <= 0.0:
                continue
            pts.append(np.array([xs[i], ys[j], _z], np.float32))
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
    Nh, Nw = disp.shape
    min_xy = outline.min(axis=0).astype(np.float32)
    max_xy = outline.max(axis=0).astype(np.float32)
    xs = np.linspace(float(min_xy[0]), float(max_xy[0]), Nw)
    ys = np.linspace(float(min_xy[1]), float(max_xy[1]), Nh)
    energy = np.abs(disp).astype(np.float32)
    peak = float(energy.max())
    if peak <= 1e-12:
        return []
    spec = np.array([0.22, 0.34, 0.30, 0.14], np.float32)
    pts, weights = [], []
    for j in range(0, Nh, stride):
        for i in range(0, Nw, stride):
            if not plate_active[j, i]:
                continue
            w = float(energy[j, i])
            if w <= peak * 0.01:
                continue
            pts.append(np.array([xs[i], ys[j], float(body_h) - 0.001], np.float32))
            weights.append(w)
    if not pts:
        return []
    w_arr = np.asarray(weights, np.float32)
    w_arr /= max(float(w_arr.sum()), 1e-12)
    n_each = np.maximum(1, (w_arr * int(total_rays)).astype(np.int32))
    dn = np.array([0.0, 0.0, -1.0], np.float32)
    order = np.argsort(n_each)[::-1]
    return [(pts[k], dn.copy(), int(n_each[k]), spec.copy()) for k in order]


def _gpu_ray_field(scene, outline, body_h, sources, max_bounces,
                   dims=GPU_RAY_FIELD_DIMS, segment_cap=GPU_RAY_SEGMENT_CAP,
                   dispatch_batch=GPU_DISPATCH_BATCH, include_stage=True):
    if scene is None or _extract_scene_geometry_fn is None:
        return None, None, None, None, 0, None, 0
    if _extract_scene_geometry_materials_fn is not None:
        verts_flat, normals, materials = _extract_scene_geometry_materials_fn(scene)
    else:
        verts_flat, normals = _extract_scene_geometry_fn(scene)
        materials = np.tile(np.array([0.58, 0.62, 0.28, 0.88, 0.10, 0.055],
                                     dtype=np.float32), (len(verts_flat), 1))
    if include_stage:
        st_v, st_n, st_m = _stage_mesh(outline, body_h)
        verts_flat = np.vstack([np.asarray(verts_flat, np.float32), st_v])
        normals = np.vstack([np.asarray(normals, np.float32), st_n])
        materials = np.vstack([np.asarray(materials, np.float32), st_m])
    if len(verts_flat) == 0:
        return None, None, None, None, 0, None, 0

    tris = verts_flat.reshape(-1, 3, 3).astype(np.float32)
    nrm = normals.astype(np.float32)
    mat = np.asarray(materials, dtype=np.float32)
    if mat.ndim != 2 or len(mat) != len(tris):
        mat = np.zeros((len(tris), 0), dtype=np.float32)
    if mat.shape[1] < 3:
        mat_in = np.tile(np.array([0.58, 0.62, 0.28], dtype=np.float32), (len(tris), 1))
    else:
        mat_in = mat[:, 0:3]
    if mat.shape[1] < 6:
        mat_out = np.tile(np.array([0.88, 0.10, 0.055], dtype=np.float32), (len(tris), 1))
    else:
        mat_out = mat[:, 3:6]
    packed = np.zeros((len(tris), 24), np.float32)
    packed[:, 0:3] = tris[:, 0, :]
    packed[:, 4:7] = tris[:, 1, :] - tris[:, 0, :]
    packed[:, 8:11] = tris[:, 2, :] - tris[:, 0, :]
    packed[:, 12:15] = nrm
    packed[:, 16:19] = mat_in
    packed[:, 20:23] = mat_out
    packed = np.ascontiguousarray(packed)
    bvh_nodes, bvh_ids = _build_gpu_bvh(tris)

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
    counter_readback = np.zeros(4, dtype=np.uint32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[4])
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, counter_readback.nbytes, counter_readback)
    actual_seg_count = int(min(int(counter_readback[0]), seg_cap))
    rays_seen = int(counter_readback[1])
    hit_count = int(counter_readback[2])
    record_count = int(counter_readback[3])
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    glUseProgram(0)
    if actual_seg_count == 0:
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

def _mvp(prog, mvp, mv=None):
    glUniformMatrix4fv(glGetUniformLocation(prog, b'uMVP'), 1, GL_TRUE, mvp)
    if mv is not None:
        loc = glGetUniformLocation(prog, b'uMV')
        if loc >= 0:
            glUniformMatrix4fv(loc, 1, GL_TRUE, mv)

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
    def __init__(self, target=(0.0, 0.0, 0.03), dist=0.85, elev=32.0, az=28.0):
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
                 gpu_seg_vbo=None, gpu_seg_cap=0,
                 max_cached_frames=MAX_FRAMES,
                 plate_theta=PLATE_THETA_SEGS,
                 plate_radial=PLATE_RADIAL_SEGS):
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

        self._frames: collections.deque = collections.deque(
            maxlen=max(1, int(max_cached_frames)))
        self._cursor  = 0
        self._paused  = False
        self._layers  = [LAYER_OPAQUE, LAYER_ALPHA, LAYER_HIDDEN,
                         LAYER_HIDDEN, LAYER_HIDDEN, LAYER_ALPHA, LAYER_ALPHA]
        self.cam = Camera()

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
        self._gpu_seg_vbo = gpu_seg_vbo
        self._gpu_seg_cap = int(gpu_seg_cap or 0)
        self._init_gl()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _init_gl(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        self._p_body  = _prog((_BODY_VS, GL_VERTEX_SHADER),
                               (_BODY_FS, GL_FRAGMENT_SHADER))
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
        self._mk_stage()
        self._mk_markers()
        self._mk_pressure_tex()
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
        wood, frets, pins, ext_strings = _neck_geometry(self.outline, self.body_h, self.n_str)
        nrm = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (len(wood), 1))
        comb = np.ascontiguousarray(np.column_stack([wood, nrm]), np.float32)
        self._neck_vao, _, self._neck_n = _vao(comb, [(0,3,24,0),(1,3,24,12)], GL_STATIC_DRAW)
        self._fret_vao, _, self._fret_n = _vao(frets, [(0,3,12,0)], GL_STATIC_DRAW)
        self._pin_vao, _, self._pin_n = _vao(pins, [(0,3,12,0)], GL_STATIC_DRAW)
        self._ext_str_vaos, self._ext_str_n = [], []
        for path in ext_strings:
            va, _, _ = _vao(path, [(0,3,12,0)], GL_STATIC_DRAW)
            self._ext_str_vaos.append(va)
            self._ext_str_n.append(len(path))

    def _mk_stage(self):
        sv, sn, _ = _stage_mesh(self.outline, self.body_h)
        verts = sv.reshape(-1, 3)
        comb = np.ascontiguousarray(np.column_stack([verts, np.repeat(sn, 3, axis=0)]), np.float32)
        self._stage_vao, _, self._stage_n = _vao(comb, [(0,3,24,0),(1,3,24,12)], GL_STATIC_DRAW)
        min_xy = self.outline.min(axis=0); max_xy = self.outline.max(axis=0)
        cx = float((min_xy[0] + max_xy[0]) * 0.5)
        cy = float((min_xy[1] + max_xy[1]) * 0.5)
        z = self.body_h + 0.42
        lamp = np.array([[cx-0.055,cy-0.055,z],[cx+0.055,cy-0.055,z],
                         [cx+0.055,cy+0.055,z],[cx-0.055,cy+0.055,z]], np.float32)
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
        return float(GPU_RAY_FIELD_SCALE) * (float(GPU_RAY_FIELD_REFERENCE_RAYS) / float(rays))

    def replace_ray_field(self, ray_field_bands, ray_field_bounds, gpu_seg_vbo, gpu_seg_cap, gpu_counter=None, total_rays=0):
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

    def _a(self, i):
        s = self._layers[i]
        return 0.0 if s == LAYER_HIDDEN else (1.0 if s == LAYER_OPAQUE else 0.40)

    # ── Dynamic updates ───────────────────────────────────────────────────────

    def _up_pressure(self, P):
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

    def _up_string(self, si, disp):
        Nseg = len(disp)
        self._str_n[si] = Nseg + 1

        if self._str_physical_mode:
            # Physical envelope mode: peak-hold of |Z displacement| decayed each frame.
            # Renders as symmetric upper/lower lines — how the string actually appears
            # to the eye when vibrating (blurred spindle shape).
            z_mag = np.abs(disp[:, 2]) * STRING_DISP_SCALE
            env = self._str_env[si]
            np.maximum(z_mag, env * ENV_DECAY, out=env)

            dn = np.zeros((Nseg + 1, 3), np.float32)
            dn[0, 2]    = env[0]
            dn[1:-1, 2] = 0.5 * (env[:-1] + env[1:])
            dn[-1, 2]   = env[-1]

            pts_hi = self.paths[si].copy(); pts_hi += dn
            pts_lo = self.paths[si].copy(); pts_lo -= dn

            glBindBuffer(GL_ARRAY_BUFFER, self._str_env_hi_vbos[si])
            glBufferSubData(GL_ARRAY_BUFFER, 0, pts_hi.nbytes,
                            np.ascontiguousarray(pts_hi, np.float32))
            glBindBuffer(GL_ARRAY_BUFFER, self._str_env_lo_vbos[si])
            glBufferSubData(GL_ARRAY_BUFFER, 0, pts_lo.nbytes,
                            np.ascontiguousarray(pts_lo, np.float32))
        else:
            # Waveform mode: instantaneous spatial displacement (standing-wave shape).
            pts = self.paths[si].copy()
            dn = np.zeros((Nseg + 1, 3), np.float32)
            dn[0] = disp[0]; dn[1:-1] = 0.5 * (disp[:-1] + disp[1:]); dn[-1] = disp[-1]
            pts += dn
            glBindBuffer(GL_ARRAY_BUFFER, self._str_vbos[si])
            glBufferSubData(GL_ARRAY_BUFFER, 0, pts.nbytes,
                            np.ascontiguousarray(pts, np.float32))

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

        # light direction in view space (fixed world light → view transform)
        light_w = np.array([0.4, 0.5, 1.0], np.float32)
        light_w /= np.linalg.norm(light_w)
        light_v = (MV[:3,:3].T @ light_w).astype(np.float32)

        self._up_pressure(frame.pressure)
        self._up_plate(frame.plate)
        for si, disp in enumerate(frame.strings):
            if disp is not None and len(disp) >= 2:
                self._up_string(si, disp)

        # ── 1. Back plate (opaque dark wood) ─────────────────────────────────
        a_body = self._a(0)
        if a_body > 0:
            glUseProgram(self._p_body)
            _mvp(self._p_body, MVP, MV)
            glUniform3f(glGetUniformLocation(self._p_body, b'uLightV'),
                        *light_v)
            glUniform4f(glGetUniformLocation(self._p_body, b'uColor'),
                        0.16, 0.035, 0.018, a_body * 0.96)
            glUniform3f(glGetUniformLocation(self._p_body, b'uInnerColor'),
                        0.64, 0.38, 0.18)
            glUniform1f(glGetUniformLocation(self._p_body, b'uAmbient'), 0.24)
            glUniform1f(glGetUniformLocation(self._p_body, b'uSpecStrength'), 0.42)
            glUniform1f(glGetUniformLocation(self._p_body, b'uShininess'), 96.0)
            glUniform1f(glGetUniformLocation(self._p_body, b'uGrain'), 0.25)
            glBindVertexArray(self._back_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, self._back_n)

        # ── 2. Side walls (semi-transparent mahogany, both faces) ─────────────
        if a_body > 0:
            glDisable(GL_CULL_FACE)
            glDepthMask(GL_FALSE)
            glUseProgram(self._p_body)
            _mvp(self._p_body, MVP, MV)
            glUniform3f(glGetUniformLocation(self._p_body, b'uLightV'),
                        *light_v)
            glUniform4f(glGetUniformLocation(self._p_body, b'uColor'),
                        0.18, 0.035, 0.018, a_body * 0.72)
            glUniform3f(glGetUniformLocation(self._p_body, b'uInnerColor'),
                        0.72, 0.44, 0.21)
            glUniform1f(glGetUniformLocation(self._p_body, b'uAmbient'), 0.22)
            glUniform1f(glGetUniformLocation(self._p_body, b'uSpecStrength'), 0.55)
            glUniform1f(glGetUniformLocation(self._p_body, b'uShininess'), 128.0)
            glUniform1f(glGetUniformLocation(self._p_body, b'uGrain'), 0.65)
            glBindVertexArray(self._wall_vao)
            glDrawElements(GL_TRIANGLES, self._wall_n_idx, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            glDepthMask(GL_TRUE)

        # ── 3. Outline rings (crisp wire guide) ───────────────────────────────
        if a_body > 0:
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP)
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
                        GPU_RAY_FIELD_GAMMA)
            glUniform1f(glGetUniformLocation(self._p_march, b'uAlpha'), a_pres)
            glUniform1i(glGetUniformLocation(self._p_march, b'uLogScale'), 0)
            glUniform1i(glGetUniformLocation(self._p_march, b'uFieldMode'), 0)
            glUniform1i(glGetUniformLocation(self._p_march, b'uPressure'), 0)
            glUniform1i(glGetUniformLocation(self._p_march, b'uBodyMask'), 1)
            glUniform1i(glGetUniformLocation(self._p_march, b'uUseBodyMask'), 1)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_3D, self._tex)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self._mask_tex)
            glActiveTexture(GL_TEXTURE0)
            glBindVertexArray(self._quad_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
            glBindTexture(GL_TEXTURE_3D, 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)

        # ── 4a. Neutral diffusive room / stage ───────────────────────────────
        a_stage = self._a(6)
        if a_stage > 0:
            glUseProgram(self._p_body)
            _mvp(self._p_body, MVP, MV)
            glUniform3f(glGetUniformLocation(self._p_body, b'uLightV'), *light_v)
            glUniform4f(glGetUniformLocation(self._p_body, b'uColor'),
                        0.46, 0.45, 0.42, a_stage * 0.72)
            glUniform3f(glGetUniformLocation(self._p_body, b'uInnerColor'),
                        0.46, 0.45, 0.42)
            glUniform1f(glGetUniformLocation(self._p_body, b'uAmbient'), 0.34)
            glUniform1f(glGetUniformLocation(self._p_body, b'uSpecStrength'), 0.04)
            glUniform1f(glGetUniformLocation(self._p_body, b'uShininess'), 12.0)
            glUniform1f(glGetUniformLocation(self._p_body, b'uGrain'), 0.08)
            glBindVertexArray(self._stage_vao)
            glDrawArrays(GL_TRIANGLES, 0, self._stage_n)
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        1.0, 0.90, 0.62, a_stage * 0.55)
            glBindVertexArray(self._lamp_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, self._lamp_n)

        # ── 4b. Ray field volume (always shown — independent of pressure toggle) ──
        if self._ray_field_bands is not None:
            rf_min = self._ray_field_bounds[0] if self._ray_field_bounds is not None else self._bmin
            rf_max = self._ray_field_bounds[1] if self._ray_field_bounds is not None else self._bmax
            glDepthMask(GL_FALSE)
            glDisable(GL_DEPTH_TEST)
            glUseProgram(self._p_march)
            glUniformMatrix4fv(
                glGetUniformLocation(self._p_march, b'uInvMVP'),
                1, GL_TRUE, iMVP)
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
                        GPU_RAY_FIELD_GAMMA)
            glUniform1f(glGetUniformLocation(self._p_march, b'uAlpha'), 0.55)
            glUniform1i(glGetUniformLocation(self._p_march, b'uLogScale'),
                        1 if GPU_RAY_LOG_SCALE else 0)
            glUniform1i(glGetUniformLocation(self._p_march, b'uFieldMode'), 1)
            glUniform1i(glGetUniformLocation(self._p_march, b'uPressure'), 0)
            glUniform1i(glGetUniformLocation(self._p_march, b'uBodyMask'), 1)
            glUniform1i(glGetUniformLocation(self._p_march, b'uUseBodyMask'), 1)
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
            _mvp(self._p_plate, MVP, MV)
            glUniform3f(glGetUniformLocation(self._p_plate, b'uLightV'),
                        *light_v)
            glUniform1f(glGetUniformLocation(self._p_plate, b'uAlpha'), a_plate)
            glUniform1f(glGetUniformLocation(self._p_plate, b'uColorMix'),
                        1.0 if self._plate_mode in (0, 2) else 0.0)
            glBindVertexArray(self._plate_vao)
            glDrawElements(GL_TRIANGLES, self._plate_n, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

        # ── 6. Soundhole dark disc ─────────────────────────────────────────────
        if a_plate > 0:
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        0.03, 0.01, 0.01, 1.0)
            glBindVertexArray(self._hole_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, self._hole_n)

        # ── 7. Strings ────────────────────────────────────────────────────────
        a_str = self._a(3)
        if a_str > 0:
            glUseProgram(self._p_body)
            _mvp(self._p_body, MVP, MV)
            glUniform3f(glGetUniformLocation(self._p_body, b'uLightV'), *light_v)
            glUniform4f(glGetUniformLocation(self._p_body, b'uColor'),
                        0.20, 0.085, 0.035, a_str * 0.95)
            glUniform3f(glGetUniformLocation(self._p_body, b'uInnerColor'),
                        0.50, 0.28, 0.12)
            glUniform1f(glGetUniformLocation(self._p_body, b'uAmbient'), 0.26)
            glUniform1f(glGetUniformLocation(self._p_body, b'uSpecStrength'), 0.36)
            glUniform1f(glGetUniformLocation(self._p_body, b'uShininess'), 88.0)
            glUniform1f(glGetUniformLocation(self._p_body, b'uGrain'), 0.55)
            glBindVertexArray(self._neck_vao)
            glDrawArrays(GL_TRIANGLES, 0, self._neck_n)

            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE)          # additive — strings glow into scene
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP)
            glLineWidth(1.2)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        0.95, 0.82, 0.52, a_str * 0.50)
            glBindVertexArray(self._fret_vao)
            glDrawArrays(GL_LINES, 0, self._fret_n)
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
                if self._str_physical_mode:
                    # Physical mode: draw upper and lower envelope lines
                    glBindVertexArray(self._str_env_hi_vaos[si])
                    glDrawArrays(GL_LINE_STRIP, 0, self._str_n[si])
                    glBindVertexArray(self._str_env_lo_vaos[si])
                    glDrawArrays(GL_LINE_STRIP, 0, self._str_n[si])
                else:
                    # Waveform mode: draw instantaneous displaced shape
                    glBindVertexArray(self._str_vaos[si])
                    glDrawArrays(GL_LINE_STRIP, 0, self._str_n[si])
            glBindVertexArray(0)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # restore

        # ── 8. Bridge markers ─────────────────────────────────────────────────
        a_mk = self._a(4)
        if a_mk > 0 and self._mk_n > 0:
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP)
            glLineWidth(1.8)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        1.0, 1.0, 0.5, a_mk * 0.85)
            glBindVertexArray(self._mk_vao)
            glDrawArrays(GL_LINES, 0, self._mk_n)
            glBindVertexArray(0)

        # ── 9. Ray-tracer segments ─────────────────────────────────────────────
        a_ray = self._a(5)
        if a_ray > 0 and self._ray_vao is not None:
            glDepthMask(GL_FALSE)
            glUseProgram(self._p_vcol)
            _mvp(self._p_vcol, MVP)
            glLineWidth(1.0)
            glBindVertexArray(self._ray_vao)
            glDrawArrays(GL_LINES, 0, self._ray_n)
            glBindVertexArray(0)
            glDepthMask(GL_TRUE)


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
    p.add_argument("--dx", type=float, default=DX,
                   help="FDTD grid spacing in metres; smaller means higher pressure/plate physics resolution")
    p.add_argument("--render-segs", type=int, default=N_RENDER_SEGS,
                   help="String FDTD/render segments per string")
    p.add_argument("--plate-theta", type=int, default=PLATE_THETA_SEGS,
                   help="Angular subdivisions for the rendered soundboard mesh")
    p.add_argument("--plate-radial", type=int, default=PLATE_RADIAL_SEGS,
                   help="Radial subdivisions for the rendered soundboard mesh")
    p.add_argument("--gpu-field-rays", type=int, default=10_000,
                   help="GPU rays/source for the 3-D ray field")
    p.add_argument("--gpu-segment-cap", type=int, default=GPU_RAY_SEGMENT_CAP,
                   help="Maximum GPU ray line segments kept for the overlay")
    p.add_argument("--gpu-dispatch-batch", type=int, default=GPU_DISPATCH_BATCH,
                   help="Rays per GPU compute dispatch; lower values avoid driver-side transient allocation/TDR")
    p.add_argument("--gpu-refresh-every", type=int, default=1,
                   help="Recompute GPU ray field every N rendered physics frames from current plate state; 0 keeps the initial static field")
    p.add_argument("--gpu-refresh-rays", type=int, default=0,
                   help="Rays used for each dynamic refresh; 0 reuses --gpu-field-rays")
    p.add_argument("--initial-ray-field", action="store_true",
                   help="Build the old static spectral ray field before the render loop")
    p.add_argument("--cpu-ray-overlay", action="store_true",
                   help="Use CPU ray tracing for the line overlay instead of GPU segments")
    p.add_argument("--ray-only-view", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Start with only ray field and ray lines visible")
    p.add_argument("--gpu-smoke-exit", action="store_true",
                   help="Build the GPU ray field once and exit before the render loop")
    p.add_argument("--excitation", choices=("strum", "a-pluck"), default="strum",
                   help="Excitation rendered into the repeatable animation")
    return p.parse_args()


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

    # ── Build physics ─────────────────────────────────────────────────────────
    print("Building physics ...", flush=True)
    scene, ce, info, bh = _build_physics(
        n_strings=6,
        dx=args.dx,
        pad_cells=PAD_CELLS,
        n_pml=N_PML,
        n_segs=args.render_segs)
    body_h = bh if bh is not None else BODY_H
    _schedule_excitation(ce, args.excitation, n_strings=6)
    print(f"  Grid {info['Nx']}×{info['Ny']}×{info['Nz']}  "
          f"body_h={body_h:.3f}m", flush=True)

    n_str  = ce.n_strings
    outline, scene_body_h = _scene_outline_or_default(scene)
    if scene is not None:
        body_h = scene_body_h
    paths   = _string_paths(outline, body_h, n_strings=n_str,
                            n_segs=args.render_segs)

    ray_field_bands = None
    ray_field_bounds = None
    ray_field_total_rays = 0
    gpu_seg_vbo = None
    gpu_seg_cap = 0
    active_strings = {min(A_STRING_INDEX, max(0, n_str - 1))} if args.excitation == "a-pluck" else None
    total       = _excitation_total_samples(args.excitation)
    total_frames = int(math.ceil(total / float(BLOCK_SAMPLES)))
    if scene is not None and args.gpu_rays and args.initial_ray_field:
        # ── Optional static spectral prepass: FFT plate displacement → ray field ─
        print("Computing static spectral plate emission map (8192 steps) ...", flush=True)
        band_maps = _compute_spectral_emission_map(ce)
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
        ray_field_bands, rb0, rb1, gpu_seg_vbo, gpu_seg_cap, _gpu_counter, ray_field_total_rays = _gpu_ray_field(
            scene, outline, body_h,
            sources=all_sources,
            max_bounces=args.max_bounces,
            segment_cap=args.gpu_segment_cap,
            dispatch_batch=args.gpu_dispatch_batch)
        ray_field_bounds = (rb0, rb1)
        if args.gpu_smoke_exit:
            pygame.quit()
            return
    elif args.initial_ray_field:
        print("GPU ray field prepass skipped: scene/GPU ray field unavailable.", flush=True)

    if args.gpu_smoke_exit:
        if scene is not None and args.gpu_rays:
            print("GPU dynamic ray field smoke frame ...", flush=True)
            frame = physics_frame(ce, n_str, args.render_segs)
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
                rbands, rb0, rb1, rvbo, rcap, rcounter, _rtotal = _gpu_ray_field(
                    scene, outline, body_h,
                    sources=dyn_sources,
                    max_bounces=args.max_bounces,
                    segment_cap=args.gpu_segment_cap,
                    dispatch_batch=args.gpu_dispatch_batch)
                for tex in rbands or []:
                    glDeleteTextures([tex])
                if rvbo is not None:
                    glDeleteBuffers(1, [rvbo])
                if rcounter is not None:
                    glDeleteBuffers(1, [rcounter])
            else:
                print("GPU dynamic ray field smoke frame produced no sources.", flush=True)
        pygame.quit()
        return

    # ── Ray trace (once, explicit opt-in only) ────────────────────────────────
    ray_segs = None
    if scene is not None and _HAS_RAY and args.cpu_ray_overlay:
        print("Tracing geometry ...", flush=True)
        ray_segs, meta = _trace_fn(
            scene, n_rays=args.trace_rays, max_bounces=args.max_bounces)
        print(f"  {len(ray_segs)} segments", flush=True)

    # ── Renderer ──────────────────────────────────────────────────────────────
    R = Renderer(WIN_W, WIN_H, outline=outline, info=info, body_h=body_h,
                 bridge_pos=BRIDGE_POS, str_paths=paths,
                 ray_segs=ray_segs,
                 ray_field_bands=ray_field_bands,
                 ray_field_bounds=ray_field_bounds,
                 ray_field_total_rays=ray_field_total_rays,
                 gpu_seg_vbo=gpu_seg_vbo,
                 gpu_seg_cap=gpu_seg_cap,
                 max_cached_frames=total_frames + 8,
                 plate_theta=args.plate_theta,
                 plate_radial=args.plate_radial)
    if not args.ray_only_view:
        R._layers = [LAYER_ALPHA, LAYER_OPAQUE, LAYER_OPAQUE,
                     LAYER_OPAQUE, LAYER_OPAQUE, LAYER_ALPHA, LAYER_ALPHA]

    print("Pre-warming ...", flush=True)
    for _ in range(8):
        R.push(physics_frame(ce, n_str, args.render_segs))
    ce.reset()
    _schedule_excitation(ce, args.excitation, n_str)
    R._frames.clear()

    clock       = pygame.time.Clock()
    running     = True
    fi          = 0
    replaying   = False
    dragging    = False
    last_mouse  = (0, 0)
    refresh_every = max(0, int(args.gpu_refresh_every))
    refresh_rays = int(args.gpu_refresh_rays or args.gpu_field_rays)
    force_ray_refresh = False

    print("1-7 toggle layers | P plate mode | W string mode | SPACE pause | R restart | "
          f"Q quit | drag=orbit wheel=zoom | excitation={args.excitation}", flush=True)

    while running:
        for ev in pygame.event.get():
            if ev.type == QUIT:
                running = False
            elif ev.type == KEYDOWN:
                if ev.key == K_q:
                    running = False
                elif ev.key == K_SPACE:
                    R._paused = not R._paused
                elif ev.key == K_r:
                    ce.reset(); _schedule_excitation(ce, args.excitation, n_str)
                    R._frames.clear(); R._cursor = 0; fi = 0
                    for env in R._str_env: env[:] = 0.0
                    replaying = False
                    force_ray_refresh = refresh_every > 0
                    print("Restarted")
                elif ev.key == K_p:
                    print(f"Plate mode: {R.cycle_plate_mode()}")
                elif ev.key == K_w:
                    R._str_physical_mode = not R._str_physical_mode
                    if not R._str_physical_mode:
                        # reset envelope so waveform starts clean
                        for env in R._str_env:
                            env[:] = 0.0
                    mode_name = "physical (envelope)" if R._str_physical_mode else "waveform (instantaneous)"
                    print(f"String mode: {mode_name}")
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

        if not replaying and fi * BLOCK_SAMPLES < total:
            frame = physics_frame(ce, n_str, args.render_segs)
            R.push(frame)
            if scene is not None and args.gpu_rays and refresh_every > 0:
                do_refresh = force_ray_refresh or (fi % refresh_every == 0)
                if do_refresh:
                    dyn_sources = _instant_soundboard_sources(
                        frame.plate, info['plate_active_2d'], outline, body_h,
                        total_rays=refresh_rays, stride=4)
                    dyn_sources += _string_emission_sources(
                        paths, body_h,
                        total_rays_per_string=max(64, refresh_rays // max(1, n_str) // 16),
                        active_strings=active_strings)
                    if dyn_sources:
                        print(f"Refreshing GPU ray field from frame {fi} "
                              f"({sum(s[2] for s in dyn_sources)} rays)", flush=True)
                        rbands, rb0, rb1, rvbo, rcap, rcounter, dyn_total_rays = _gpu_ray_field(
                            scene, outline, body_h,
                            sources=dyn_sources,
                            max_bounces=args.max_bounces,
                            segment_cap=args.gpu_segment_cap,
                            dispatch_batch=args.gpu_dispatch_batch)
                        R.replace_ray_field(rbands, (rb0, rb1), rvbo, rcap, rcounter,
                                            total_rays=dyn_total_rays)
                    force_ray_refresh = False
            fi += 1
        elif not replaying:
            R._cursor = 0
            replaying = True
            print(f"Cached {len(R._frames)} frames for repeatable replay", flush=True)

        R.tick()
        R.render()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
