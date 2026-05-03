"""guitar_geometry.py
=====================
Pure NumPy geometry algorithms for building a guitar as a structured object.

All functions are adapted from demo_pluck_gl.py and are deliberately kept
free of GL, pygame, physics, and GuitarItem references so they can be used
at import time with no side effects.

Public API
----------
guitar_outline(n_pts)           → (N, 2) float32 outline polygon
string_paths(outline, cfg)      → list[(n_segs+1, 3) float32]
neck_geometry(outline, cfg)     → NeckGeometry(namedtuple)
electroacoustic_fixtures(cfg)   → (pickup (4,3), mic (6,3)) float32
plate_mesh(outline, cfg)        → (verts (M, 2), indices (K,)) float32/int32
side_walls(outline, cfg)        → (verts (N,3), norms (N,3), indices (M,)) float32/int32
back_fan(outline)               → (N+2, 3) float32  (GL_TRIANGLE_FAN)
disc_verts(cx, cy, z, r, n)     → (n+2, 3) float32  (GL_TRIANGLE_FAN)
guitar_model_matrix(outline, stand_height) → (M 4x4 float32, Minv 4x4 float32)
build_guitar_parts(cfg)         → list[GuitarPart]
"""
from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, TYPE_CHECKING

import numpy as np

from guitar_part import (
    GuitarPart, PartVisibility,
    RenderMaterial, RayMaterial,
    MAT_BODY_BACK, MAT_BODY_SIDES, MAT_SOUNDBOARD, MAT_NECK,
    MAT_FRETS, MAT_TUNERS, MAT_PICKUP,
    RAYMAT_SPRUCE, RAYMAT_ROSEWOOD, RAYMAT_MAHOGANY,
    RAYMAT_STEEL_PLAIN, RAYMAT_STEEL_WOUND, RAYMAT_NICKEL, RAYMAT_CERAMIC,
)

if TYPE_CHECKING:
    from guitar_item import GuitarConfig


# ─────────────────────────────────────────────────────────────────────────────
# Defaults (mirrors demo_pluck_gl.py constants; overridden by GuitarConfig)
# ─────────────────────────────────────────────────────────────────────────────

_SOUNDHOLE_CX = 0.0
_SOUNDHOLE_CY = 0.050
_SOUNDHOLE_R  = 0.028
_BODY_H       = 0.060
_SCALE_LENGTH = 0.648
_STRING_CLEARANCE = 0.010
_STAND_HEIGHT = 0.40
_N_RENDER_SEGS = 120   # lighter default for standalone geometry use


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _effective_scale(scale_length: float, fret: int = 0) -> float:
    return float(scale_length / (2.0 ** (max(0, fret) / 12.0)))


def _ray_outline_radius(outline: np.ndarray, cx: float, cy: float, theta: float) -> float:
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
    return float(best if best is not None else _SOUNDHOLE_R)


# ─────────────────────────────────────────────────────────────────────────────
# Outline
# ─────────────────────────────────────────────────────────────────────────────

def guitar_outline(n_pts: int = 128) -> np.ndarray:
    """Canonical guitar body outline — (n_pts, 2) float32."""
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
        t     = (y - y_bot) / y_span
        wenv  = math.exp(-((t - (0.0 - y_bot) / y_span) / 0.14)**2)
        xs[k] = x_env * (1.0 - wenv) + waist_x * wenv
    right = np.column_stack([ xs,        ys      ])
    left  = np.column_stack([-xs[::-1],  ys[::-1]])
    return np.concatenate([right, left]).astype(np.float32)[:n_pts]


# ─────────────────────────────────────────────────────────────────────────────
# String paths
# ─────────────────────────────────────────────────────────────────────────────

def string_paths(outline: np.ndarray,
                 n_strings: int = 6,
                 n_segs: int = _N_RENDER_SEGS,
                 body_h: float = _BODY_H,
                 scale_length: float = _SCALE_LENGTH,
                 string_clearance: float = _STRING_CLEARANCE,
                 fret: int = 0) -> List[np.ndarray]:
    """Returns list of n_strings arrays, each shaped (n_segs+1, 3) float32."""
    y_saddle = -0.070
    y_nut    = y_saddle + _effective_scale(scale_length, fret)
    x_span   = 0.0088 * max(0, n_strings - 1)
    string_z = body_h + string_clearance
    paths = []
    for si in range(n_strings):
        x = -x_span / 2 + si * (x_span / max(1, n_strings - 1))
        x_nut = x * 0.72
        ys = np.linspace(y_nut, y_saddle, n_segs + 1, dtype=np.float32)
        xs = np.linspace(x_nut, x, n_segs + 1, dtype=np.float32)
        path = np.column_stack([
            xs, ys, np.full(n_segs + 1, string_z, dtype=np.float32),
        ])
        paths.append(path)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Neck
# ─────────────────────────────────────────────────────────────────────────────

class NeckGeometry(NamedTuple):
    wood_tris:     np.ndarray   # (N, 3) float32 — triangle-list XYZ
    fret_lines:    np.ndarray   # (M, 3) float32 — GL_LINES pairs
    pin_lines:     np.ndarray   # (K, 3) float32 — tuning pin stubs
    ext_strings:   List[np.ndarray]  # per-string (2, 3) — nut→tuner segment
    anchor_lines:  np.ndarray   # (2, 3) float32 — active-fret bar


def neck_geometry(outline: np.ndarray,
                  body_h: float = _BODY_H,
                  n_strings: int = 6,
                  scale_length: float = _SCALE_LENGTH,
                  active_fret: int = 0,
                  fretless: bool = False) -> NeckGeometry:
    y_body = float(outline[:, 1].max()) * 0.85
    y_nut  = -0.070 + scale_length
    y_head = y_nut + 0.13
    z = body_h + 0.003
    neck_w0, neck_w1 = 0.056, 0.044
    head_w = 0.092

    neck = np.array([
        [-neck_w0 * 0.5, y_body, z], [ neck_w0 * 0.5, y_body, z],
        [ neck_w1 * 0.5, y_nut,  z], [-neck_w1 * 0.5, y_nut,  z],
    ], np.float32)
    head = np.array([
        [-head_w * 0.42, y_nut,  z], [ head_w * 0.42, y_nut,  z],
        [ head_w * 0.58, y_head, z + 0.002], [-head_w * 0.58, y_head, z + 0.002],
    ], np.float32)
    wood = np.vstack([neck[[0,1,2, 0,2,3]], head[[0,1,2, 0,2,3]]]).astype(np.float32)

    fret_lines = []
    for fi in range(1, 21):
        y = y_nut - scale_length / (2.0 ** (fi / 12.0))
        if y <= y_body:
            continue
        t = (y - y_body) / max(y_nut - y_body, 1e-6)
        half_w = 0.5 * ((1.0 - t) * neck_w0 + t * neck_w1)
        fret_lines.extend([[-half_w, y, z + 0.0025], [half_w, y, z + 0.0025]])

    x_span = 0.0088 * max(0, n_strings - 1)
    ext_strings = []
    pin_lines   = []
    for si in range(n_strings):
        x_s = -x_span / 2 + si * (x_span / max(1, n_strings - 1))
        x_nut = x_s * 0.72
        x_tune = (-0.038 if si < n_strings // 2 else 0.038)
        y_tune = y_nut + 0.025 + (si % 3) * 0.035
        ext_strings.append(np.array([
            [x_nut,     y_nut,  body_h + 0.010],
            [x_tune,    y_tune, body_h + 0.013],
        ], np.float32))
        pin_lines.extend([[x_tune - 0.010, y_tune, z + 0.008],
                          [x_tune + 0.010, y_tune, z + 0.008]])

    anchor_lines = []
    if active_fret > 0:
        y = y_nut - scale_length / (2.0 ** (active_fret / 12.0))
        if y > y_body:
            t = (y - y_body) / max(y_nut - y_body, 1e-6)
            half_w = 0.5 * ((1.0 - t) * neck_w0 + t * neck_w1)
            z_anchor = z + (0.006 if fretless else 0.004)
            anchor_lines.extend([[-half_w, y, z_anchor], [half_w, y, z_anchor]])

    return NeckGeometry(
        wood_tris    = wood,
        fret_lines   = np.asarray(fret_lines or [[0,0,0],[0,0,0]], np.float32),
        pin_lines    = np.asarray(pin_lines, np.float32),
        ext_strings  = ext_strings,
        anchor_lines = np.asarray(anchor_lines or [[0,0,0],[0,0,0]], np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Electroacoustic fixtures
# ─────────────────────────────────────────────────────────────────────────────

def electroacoustic_fixtures(body_h: float = _BODY_H,
                              y_saddle: float = -0.070
                              ) -> tuple[np.ndarray, np.ndarray]:
    """(pickup_quad (4,3), mic_cross (6,3)) float32 in guitar frame."""
    z = body_h + 0.002
    py = y_saddle - 0.020
    pickup = np.array([
        [-0.037, py - 0.004, z], [ 0.037, py - 0.004, z],
        [ 0.037, py + 0.004, z], [-0.037, py + 0.004, z],
    ], np.float32)
    mic_z = body_h + 0.075
    mic = np.array([
        [-0.010, 0.000, mic_z], [0.010, 0.000, mic_z],
        [0.000, -0.010, mic_z], [0.000, 0.010, mic_z],
        [0.000,  0.000, mic_z], [0.000, 0.000, body_h + 0.006],
    ], np.float32)
    return pickup, mic


# ─────────────────────────────────────────────────────────────────────────────
# Soundboard / plate
# ─────────────────────────────────────────────────────────────────────────────

def plate_mesh(outline: np.ndarray,
               soundhole_cx: float = _SOUNDHOLE_CX,
               soundhole_cy: float = _SOUNDHOLE_CY,
               soundhole_r:  float = _SOUNDHOLE_R,
               n_theta: int = 192,
               n_radial: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """Smooth fan mesh for the guitar top with a real soundhole boundary.

    Returns (verts (M, 2), indices (K,)) float32 / int32.
    Z = 0; the caller adds body_h to get world height.
    """
    verts = []
    for ri in range(n_radial + 1):
        frac = ri / float(n_radial)
        for ai in range(n_theta):
            th = 2.0 * math.pi * ai / float(n_theta)
            outer = _ray_outline_radius(outline, soundhole_cx, soundhole_cy, th)
            r = soundhole_r + frac * max(0.0, outer - soundhole_r)
            verts.append((soundhole_cx + r * math.cos(th),
                          soundhole_cy + r * math.sin(th)))
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


def disc_verts(cx: float, cy: float, z: float,
               r: float = _SOUNDHOLE_R, n: int = 48) -> np.ndarray:
    """Circle fan (n+2, 3) float32: centre, ring×n, ring[0] — for GL_TRIANGLE_FAN."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a),
                            np.full(n, z, np.float32)]).astype(np.float32)
    cen = np.array([[cx, cy, z]], np.float32)
    return np.vstack([cen, ring, ring[[0]]])


# ─────────────────────────────────────────────────────────────────────────────
# Body side walls
# ─────────────────────────────────────────────────────────────────────────────

def side_walls(outline: np.ndarray,
               body_h: float = _BODY_H) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indexed triangle mesh for the extruded guitar side walls.

    Returns (verts (N,3), norms (N,3), indices (M,)) float32 / int32.
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


def back_fan(outline: np.ndarray) -> np.ndarray:
    """Triangle fan (N+2, 3) float32 for the back plate (Z=0 plane)."""
    cen = np.array([[outline[:, 0].mean(), outline[:, 1].mean(), 0.0]], np.float32)
    pts = np.column_stack([outline, np.zeros(len(outline), np.float32)])
    return np.vstack([cen, pts, pts[[0]]])


# ─────────────────────────────────────────────────────────────────────────────
# Model matrix
# ─────────────────────────────────────────────────────────────────────────────

def guitar_model_matrix(outline: np.ndarray,
                        stand_height: float = _STAND_HEIGHT
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Guitar-frame → world-frame 4×4 matrices (M, Minv) float32.

    Guitar frame: X=lateral, Y=longitudinal, Z=depth.
    World frame:  X=lateral, Y=stage-depth, Z=up.
    """
    y_bot = float(outline[:, 1].min())
    tz = stand_height - y_bot
    M = np.array([
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., tz],
        [0., 0., 0., 1.],
    ], dtype=np.float32)
    Minv = np.array([
        [1., 0., 0.,  0.],
        [0., 0., 1., -tz],
        [0., 1., 0.,  0.],
        [0., 0., 0.,  1.],
    ], dtype=np.float32)
    return M, Minv


# ─────────────────────────────────────────────────────────────────────────────
# PiP helper (for plate mask)
# ─────────────────────────────────────────────────────────────────────────────

def pip_grid(X: np.ndarray, Y: np.ndarray, poly: np.ndarray) -> np.ndarray:
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


# ─────────────────────────────────────────────────────────────────────────────
# Part factory
# ─────────────────────────────────────────────────────────────────────────────

# String materials per string index (0=e1, 1=B, 2=G wound starts around 3)
_STRING_COLORS_RGBA = [
    (0.92, 0.92, 0.88), (0.88, 0.88, 0.84), (0.84, 0.84, 0.80),
    (0.76, 0.74, 0.68), (0.70, 0.68, 0.62), (0.62, 0.60, 0.54),
]
_STRING_WOUND_THRESHOLD = 3   # strings >= this index are wound (more diffuse)


def build_guitar_parts(outline: np.ndarray,
                       body_h: float = _BODY_H,
                       n_strings: int = 6,
                       scale_length: float = _SCALE_LENGTH,
                       string_clearance: float = _STRING_CLEARANCE,
                       soundhole_cx: float = _SOUNDHOLE_CX,
                       soundhole_cy: float = _SOUNDHOLE_CY,
                       soundhole_r:  float = _SOUNDHOLE_R,
                       active_fret: int = 0,
                       fretless: bool = False,
                       n_render_segs: int = _N_RENDER_SEGS) -> List[GuitarPart]:
    """Build the canonical set of GuitarPart instances for a standard guitar.

    Returns parts in draw order (opaque back → sides → neck → soundboard →
    strings → frets → fixtures).
    """
    parts: List[GuitarPart] = []

    # ── Back plate ────────────────────────────────────────────────────────────
    bf = back_fan(outline)
    back_norms = np.tile([0., 0., -1.], (len(bf), 1)).astype(np.float32)
    parts.append(GuitarPart(
        name="body_back",
        render_mat=MAT_BODY_BACK, ray_mat=RAYMAT_ROSEWOOD,
        visibility=PartVisibility.OPAQUE,
        verts=bf, norms=back_norms,
        sim_tag="",
    ))

    # ── Side walls ────────────────────────────────────────────────────────────
    sw_v, sw_n, sw_i = side_walls(outline, body_h)
    parts.append(GuitarPart(
        name="body_sides",
        render_mat=MAT_BODY_SIDES, ray_mat=RAYMAT_ROSEWOOD,
        visibility=PartVisibility.OPAQUE,
        verts=sw_v, norms=sw_n, indices=sw_i,
        sim_tag="",
    ))

    # ── Soundboard (plate) ────────────────────────────────────────────────────
    pv, pi = plate_mesh(outline,
                        soundhole_cx=soundhole_cx,
                        soundhole_cy=soundhole_cy,
                        soundhole_r=soundhole_r)
    parts.append(GuitarPart(
        name="soundboard",
        render_mat=MAT_SOUNDBOARD, ray_mat=RAYMAT_SPRUCE,
        visibility=PartVisibility.OPAQUE,
        plate_verts=pv, plate_indices=pi,
        sim_tag="plate",
    ))

    # ── Soundhole ring (decorative) ───────────────────────────────────────────
    ring = disc_verts(soundhole_cx, soundhole_cy, body_h + 0.0005,
                      r=soundhole_r, n=64)
    ring_norms = np.tile([0., 0., 1.], (len(ring), 1)).astype(np.float32)
    parts.append(GuitarPart(
        name="soundhole",
        render_mat=MAT_BODY_BACK, ray_mat=RAYMAT_ROSEWOOD,
        visibility=PartVisibility.HIDDEN,   # hidden: just a depth stopper
        verts=ring, norms=ring_norms,
        sim_tag="",
    ))

    # ── Neck + headstock ──────────────────────────────────────────────────────
    ng = neck_geometry(outline, body_h=body_h, n_strings=n_strings,
                       scale_length=scale_length,
                       active_fret=active_fret, fretless=fretless)
    # Wood face
    neck_norms = np.tile([0., 0., 1.], (len(ng.wood_tris), 1)).astype(np.float32)
    parts.append(GuitarPart(
        name="neck",
        render_mat=MAT_NECK, ray_mat=RAYMAT_MAHOGANY,
        visibility=PartVisibility.OPAQUE,
        verts=ng.wood_tris, norms=neck_norms,
        sim_tag="",
    ))
    # Fret lines
    if len(ng.fret_lines) >= 2:
        parts.append(GuitarPart(
            name="frets",
            render_mat=MAT_FRETS, ray_mat=RAYMAT_NICKEL,
            visibility=PartVisibility.OPAQUE,
            line_verts=ng.fret_lines,
            sim_tag="",
        ))
    # Tuner pins
    if len(ng.pin_lines) >= 2:
        parts.append(GuitarPart(
            name="tuners",
            render_mat=MAT_TUNERS, ray_mat=RAYMAT_NICKEL,
            visibility=PartVisibility.OPAQUE,
            line_verts=ng.pin_lines,
            sim_tag="",
        ))
    # Headstock extension strings
    for si, seg in enumerate(ng.ext_strings):
        col = _STRING_COLORS_RGBA[si % len(_STRING_COLORS_RGBA)]
        rm = RenderMaterial(
            color=col, inner_color=col,
            ambient=0.30, spec_strength=0.75, shininess=160.0, grain=0.02,
        )
        ray = RAYMAT_STEEL_WOUND if si >= _STRING_WOUND_THRESHOLD else RAYMAT_STEEL_PLAIN
        parts.append(GuitarPart(
            name=f"headstock_string_{si}",
            render_mat=rm, ray_mat=ray,
            visibility=PartVisibility.OPAQUE,
            line_verts=seg,
            sim_tag="",
        ))
    # Active fret bar
    if len(ng.anchor_lines) >= 2 and active_fret > 0:
        parts.append(GuitarPart(
            name="fret_bar",
            render_mat=MAT_FRETS, ray_mat=RAYMAT_NICKEL,
            visibility=PartVisibility.OPAQUE,
            line_verts=ng.anchor_lines,
            sim_tag="",
        ))

    # ── Strings (sim-driven, updated per frame) ───────────────────────────────
    spaths = string_paths(outline,
                          n_strings=n_strings,
                          n_segs=n_render_segs,
                          body_h=body_h,
                          scale_length=scale_length,
                          string_clearance=string_clearance,
                          fret=active_fret)
    for si, path in enumerate(spaths):
        col = _STRING_COLORS_RGBA[si % len(_STRING_COLORS_RGBA)]
        rm = RenderMaterial(
            color=col, inner_color=col,
            ambient=0.30, spec_strength=0.75, shininess=160.0, grain=0.02,
        )
        ray = RAYMAT_STEEL_WOUND if si >= _STRING_WOUND_THRESHOLD else RAYMAT_STEEL_PLAIN
        parts.append(GuitarPart(
            name=f"string_{si}",
            render_mat=rm, ray_mat=ray,
            visibility=PartVisibility.OPAQUE,
            line_verts=path,
            sim_tag=f"string_{si}",
        ))

    # ── Electroacoustic fixtures ──────────────────────────────────────────────
    pickup_quad, mic_cross = electroacoustic_fixtures(body_h=body_h)
    # Pickup bar (quad → 2 triangles)
    p_v = np.array([
        pickup_quad[0], pickup_quad[1], pickup_quad[2],
        pickup_quad[0], pickup_quad[2], pickup_quad[3],
    ], np.float32)
    p_n = np.tile([0., 0., 1.], (6, 1)).astype(np.float32)
    parts.append(GuitarPart(
        name="pickup",
        render_mat=MAT_PICKUP, ray_mat=RAYMAT_CERAMIC,
        visibility=PartVisibility.OPAQUE,
        verts=p_v, norms=p_n,
        sim_tag="",
    ))
    # Mic cross / stand
    parts.append(GuitarPart(
        name="mic_stand",
        render_mat=MAT_PICKUP, ray_mat=RAYMAT_CERAMIC,
        visibility=PartVisibility.OPAQUE,
        line_verts=mic_cross,
        sim_tag="",
    ))

    return parts
