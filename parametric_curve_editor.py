"""parametric_curve_editor.py
─────────────────────────────────────────────────────────────────────────────
Two-panel pygame editor for an amplitude / chirp envelope pair.
Fully surface-based — no OpenGL dependency.  Compatible as a subunit panel
alongside PlotWidget / graph_widget in bass_viewer and analytic_driver.

Architecture
─────────────
  Panel 0 (bottom)  — default role: amplitude
  Panel 1 (top)     — default role: chirp

  Both panels SHARE markers, region modes, and the rule tree structure.
  Each panel has individual:
    • control points (the curve shape)
    • activation function / drive
    • Y-axis extents (v_lo, v_hi) and display scale (linear / log / tanh)
    • editable title  →  used as the curve filename for save/load
    • role dropdown   →  amplitude | chirp | fm_depth | am_depth
    • scale dropdown  →  linear | log | tanh
    • [SAVE] [LOAD] buttons

Shared structure sync
──────────────────────
  Adding / moving / deleting a marker, or changing a region mode on either
  panel, is immediately mirrored to the other panel.  The control points
  are never touched during sync.

Display (passive)
──────────────────
  Channel B is populated externally by analytic_driver with the rendered voice
  signal.  PCE never synthesises audio or drives playback.  Channel B's
  display_signals are read in _refresh_render_points("B") to build the
  overlay pts shown behind the editable curves and in the output panel.

Y-axis scales
──────────────
  linear   v_norm → v_lo + v_norm*(v_hi - v_lo)
  log      v_norm → v_lo * (v_hi/v_lo)^v_norm   [positive ranges only]
  tanh     emphasises midpoint; neutral y=0.5 always maps to the midpoint
           of the physical range (useful for bipolar chirp / FM curves)

  The chirp panel defaults to v_lo=-200, v_hi=200 so that y=0.5 = 0 Hz.

Keyboard
  D          toggle break_after on hovered point
  M          cycle region mode under mouse (synced to both panels)
  T          rename hovered time marker (synced)
  A          cycle activation function (focused panel only)
  = / -      activation drive ±0.5
  Tab        toggle focus between panels
  Escape     quit
"""
from __future__ import annotations

import copy
import math
import os
import time as _time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
from scipy.io import wavfile

from parametric_curve import (
    ParametricCurve, ControlPoint, TimeMarker, RegionEffect,
    RuleNode, EnvelopeRuleTree,
    default_envelope, default_chirp, default_blank,
    normalize_channel_complex_signals,
    _REGION_MODES, _REGION_COLORS, _ACTIVATION_MODES,
    _split_into_chains,
    _y_from_physical, _y_to_physical,
)

try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

try:
    import pygame
    from pygame.locals import (
        KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP,
        MOUSEMOTION, QUIT, RESIZABLE,
        K_ESCAPE, K_TAB, K_d, K_m, K_s, K_l, K_t, K_RETURN,
        K_BACKSPACE, KMOD_CTRL, K_a, K_EQUALS, K_MINUS,
    )
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_ML, _MR, _MT, _MB = 56, 28, 28, 36   # margins (left, right, top, bottom)
_HEADER_H   = 22                        # per-panel header/toolbar height
_PANEL_GAP  = 10                        # gap between panels
_PREVIEW_SR = 44100
_RENDER_TIMEOUT_S = 0.5   # seconds after last note-off before post-session render fires

# Roles and their default (v_lo, v_hi, y_scale)
_PARAM_ROLES = ("amplitude", "chirp", "fm_depth", "am_depth", "analytic")
_ROLE_DEFAULTS: dict = {
    "amplitude": (0.0,    1.0,    "linear"),
    "chirp":     (-200.0, 200.0,  "linear"),
    "fm_depth":  (0.0,    200.0,  "linear"),
    "am_depth":  (0.0,    1.0,    "linear"),
    "analytic":  (-1.0,   1.0,    "linear"),
}
_SCALE_MODES = ("linear", "log", "tanh")
_CHANNEL_KEYS = ("A", "B", "C", "D")

# Colours
_COL_BG            = (0.10, 0.10, 0.12, 1.0)
_COL_GRID          = (0.22, 0.22, 0.25, 0.5)
_COL_GRID_ZERO     = (0.40, 0.40, 0.50, 0.7)   # zero-crossing line (bipolar panels)
_COL_AXIS          = (0.50, 0.50, 0.55, 1.0)
_COL_SPLINE_0      = (0.25, 0.80, 0.45, 1.0)   # green  — panel 0
_COL_SPLINE_1      = (0.35, 0.65, 1.00, 1.0)   # blue   — panel 1
_COL_PT_NORMAL     = (0.90, 0.90, 0.95, 1.0)
_COL_PT_BREAK      = (0.95, 0.55, 0.20, 1.0)
_COL_PT_HOVER      = (1.00, 0.95, 0.20, 1.0)
_COL_PT_DRAG       = (0.40, 0.90, 1.00, 1.0)
_COL_MARKER        = (0.60, 0.60, 0.90, 0.9)
_COL_MARKER_HOVER  = (0.90, 0.80, 1.00, 1.0)
_COL_MARKER_PIN    = (0.60, 0.40, 0.40, 0.9)
_COL_FOCUS_BORDER  = (0.95, 0.85, 0.30, 0.8)
_COL_UNFOCUS_BDR   = (0.28, 0.28, 0.33, 0.5)
_COL_HEADER_FOCUS  = (0.18, 0.17, 0.12, 1.0)
_COL_HEADER_IDLE   = (0.12, 0.12, 0.15, 1.0)
_COL_REND_0_FILL   = (0.20, 0.70, 0.40, 0.22)
_COL_REND_0_LINE   = (0.30, 0.90, 0.55, 0.60)
_COL_REND_1_FILL   = (0.25, 0.55, 0.95, 0.22)
_COL_REND_1_LINE   = (0.45, 0.75, 1.00, 0.60)
_COL_AUDIO_FILL    = (0.25, 0.55, 0.80, 0.15)
_COL_AUDIO_LINE    = (0.85, 0.95, 0.45, 0.65)
_COL_PLAYHEAD      = (1.00, 0.70, 0.20, 0.90)
_COL_KNOB_BG       = (0.20, 0.20, 0.24, 1.0)
_COL_KNOB_RIM      = (0.50, 0.50, 0.58, 1.0)
_COL_KNOB_ARC      = (0.30, 0.80, 0.55, 1.0)
_COL_KNOB_HOVER    = (0.55, 0.90, 0.70, 1.0)

# Background signal overlay colours
_COL_RAW_OSC_FILL  = (0.18, 0.32, 0.58, 0.13)   # raw oscillator fill (chirp panel bg)
_COL_RAW_OSC_LINE  = (0.40, 0.62, 0.95, 0.50)   # raw oscillator line (chirp panel bg)
_COL_CRTOSC_FILL   = (0.10, 0.42, 0.25, 0.13)   # courtesy oscillator fill (amp panel bg)
_COL_CRTOSC_LINE   = (0.28, 0.80, 0.50, 0.38)   # courtesy oscillator line (amp panel bg)
_COL_OUT_AMP_FILL  = (0.20, 0.65, 0.38, 0.18)   # amplitude envelope fill (output panel)
_COL_OUT_AMP_LINE  = (0.30, 0.90, 0.55, 0.60)   # amplitude envelope line (output panel)
_COL_LAYER_R_FILL  = (0.95, 0.22, 0.22, 0.18)
_COL_LAYER_R_LINE  = (1.00, 0.40, 0.40, 0.88)
_COL_LAYER_G_FILL  = (0.20, 0.82, 0.34, 0.18)
_COL_LAYER_G_LINE  = (0.34, 0.98, 0.52, 0.88)
_COL_LAYER_B_FILL  = (0.20, 0.48, 1.00, 0.18)
_COL_LAYER_B_LINE  = (0.46, 0.72, 1.00, 0.88)

# Panel height modes
_PANEL_HEIGHT_PX: dict[str, int] = {
    "collapsed": 0,
    "small":     110,
    "medium":    200,
    "large":     380,
}
_PANEL_HEIGHT_MODES = ("collapsed", "small", "medium", "large")
_PANELS_SCROLL_STEP = 35   # pixels per mouse-wheel tick

# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _panel_plot_rect(panel_idx: int, w: int, h: int, panel_count: int = 2) -> tuple:
    """Return (x0, y0, pw, ph) of the spline-drawing area for panel_idx.
    Panel 0 = bottom. Higher indices stack upward.
    """
    pw = float(w - _ML - _MR)
    usable_h = float(h - _MT - _MB - panel_count * _HEADER_H - max(0, panel_count - 1) * _PANEL_GAP)
    ph = usable_h / float(max(panel_count, 1))
    x0 = float(_ML)
    y0 = float(_MB) + panel_idx * (ph + _HEADER_H + _PANEL_GAP)
    return x0, y0, pw, ph


def _panel_header_rect(panel_idx: int, w: int, h: int, panel_count: int = 2) -> tuple:
    """Return (x0, y0, pw, ph) of the header toolbar just above the plot."""
    x0, y0, pw, ph = _panel_plot_rect(panel_idx, w, h, panel_count)
    return x0, y0 + ph, pw, float(_HEADER_H)


# Backward-compat alias used internally
def _panel_rect(panel_idx: int, w: int, h: int, panel_count: int = 2) -> tuple:
    return _panel_plot_rect(panel_idx, w, h, panel_count)


def _panels_view_gl_rect(w: int, h: int) -> tuple:
    """Return (vx, vy_bot, vw, vh) of the scrollable panels viewport in GL coords.
    Leaves 20 px at the bottom for the status bar and _MT at the top.
    """
    return 0, 20, w, max(0, h - 20 - _MT)


def _panels_content_h(height_modes: list) -> int:
    """Total pixel height of all stacked panels (plots + headers + gaps)."""
    total = 0
    for i, hm in enumerate(height_modes):
        total += _HEADER_H + _PANEL_HEIGHT_PX.get(hm, 200)
        if i < len(height_modes) - 1:
            total += _PANEL_GAP
    return total


def _panel_at(py: float, w: int, h: int, panel_count: int = 2) -> int:
    """Return which panel a GL y-coordinate belongs to."""
    for panel_idx in reversed(range(max(panel_count, 1))):
        _, y0, _, ph = _panel_plot_rect(panel_idx, w, h, panel_count)
        if y0 <= py <= y0 + ph + _HEADER_H:
            return panel_idx
    return 0


def _header_button_rects(panel_idx: int, w: int, h: int, panel_count: int = 2) -> dict:
    """Return a dict of named rects (GL coords) for header toolbar elements.

    Keys: "title", "role", "channel", "stretch", "scale", "save", "load"
    """
    hx, hy, hw, hh = _panel_header_rect(panel_idx, w, h, panel_count)
    pad = 3.0
    btn_h = hh - 2 * pad
    x = hx + pad

    title_w  = min(180.0, hw * 0.22)
    role_w   = 90.0
    channel_w = 54.0
    stretch_w = 54.0
    scale_w  = 62.0
    collapse_w = 36.0
    btn_w    = 44.0
    gap      = 6.0

    title_r  = (x, hy + pad, title_w, btn_h)
    x += title_w + gap
    role_r   = (x, hy + pad, role_w, btn_h)
    x += role_w + gap
    channel_r = (x, hy + pad, channel_w, btn_h)
    x += channel_w + gap
    stretch_r = (x, hy + pad, stretch_w, btn_h)
    x += stretch_w + gap
    scale_r  = (x, hy + pad, scale_w, btn_h)
    x += scale_w + gap
    collapse_r = (x, hy + pad, collapse_w, btn_h)

    save_x = hx + hw - 2 * btn_w - gap - pad
    load_x = hx + hw - btn_w - pad
    save_r  = (save_x, hy + pad, btn_w, btn_h)
    load_r  = (load_x, hy + pad, btn_w, btn_h)

    hmode_w = 28.0
    hmode_x = save_x - hmode_w - gap
    hmode_r = (hmode_x, hy + pad, hmode_w, btn_h)

    return {"title": title_r, "role": role_r, "channel": channel_r,
            "stretch": stretch_r, "scale": scale_r, "collapse": collapse_r,
            "hmode": hmode_r, "save": save_r,   "load": load_r}


def _hit_rect(px: float, py: float, rect: tuple) -> bool:
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def _header_button_rects_from_rect(hx: float, hy: float, hw: float, hh: float) -> dict:
    """Same as _header_button_rects but takes the pre-computed header rect directly.
    Used by the instance-level _hbtns() so it can pass height-mode-aware coords.
    """
    pad = 3.0
    btn_h = hh - 2 * pad
    x = hx + pad

    title_w   = min(180.0, hw * 0.22)
    role_w    = 90.0
    channel_w = 54.0
    stretch_w = 54.0
    scale_w   = 62.0
    collapse_w = 36.0
    btn_w     = 44.0
    gap       = 6.0

    title_r   = (x, hy + pad, title_w, btn_h); x += title_w + gap
    role_r    = (x, hy + pad, role_w,  btn_h); x += role_w + gap
    channel_r = (x, hy + pad, channel_w, btn_h); x += channel_w + gap
    stretch_r = (x, hy + pad, stretch_w, btn_h); x += stretch_w + gap
    scale_r   = (x, hy + pad, scale_w,  btn_h); x += scale_w + gap
    collapse_r = (x, hy + pad, collapse_w, btn_h)

    hmode_w = 28.0
    save_x  = hx + hw - 2 * btn_w - gap - hmode_w - gap - pad
    load_x  = hx + hw - btn_w - pad
    hmode_x = hx + hw - 2 * btn_w - gap - hmode_w - pad
    save_r  = (save_x,  hy + pad, btn_w,   btn_h)
    load_r  = (load_x,  hy + pad, btn_w,   btn_h)
    hmode_r = (hmode_x, hy + pad, hmode_w, btn_h)

    return {"title": title_r, "role": role_r, "channel": channel_r,
            "stretch": stretch_r, "scale": scale_r, "collapse": collapse_r,
            "hmode": hmode_r, "save": save_r, "load": load_r}


# ─────────────────────────────────────────────────────────────────────────────
# Drawing primitives  (pure pygame — no OpenGL)
# ─────────────────────────────────────────────────────────────────────────────

class _DrawCtx:
    """Mutable drawing context set at the start of each frame."""
    def __init__(self):
        self.surf = None      # pygame.Surface to draw on
        self.h    = 0         # surface height (for GL→pygame Y-flip)
        self.color = (255, 255, 255, 255)
        self.line_width = 1

_ctx = _DrawCtx()


def _py(gl_y: float, rect_h: float = 0.0) -> int:
    """Convert a GL y-coordinate (y from bottom) to a pygame y-coordinate (y from top)."""
    return _ctx.h - int(gl_y) - int(rect_h)


def _gl_color(c):
    a = int(c[3] * 255) if len(c) > 3 else 255
    _ctx.color = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), a)


def glLineWidth(w: float) -> None:  # noqa: N802 — intentional GL-style name
    _ctx.line_width = max(1, int(round(w)))


def _draw_line(x0, y0, x1, y1):
    if _ctx.surf is None:
        return
    pygame.draw.line(_ctx.surf, _ctx.color[:3],
                     (int(x0), _py(y0)), (int(x1), _py(y1)),
                     _ctx.line_width)


def _draw_diamond(cx, cy, r=5.0):
    if _ctx.surf is None:
        return
    pts = [
        (int(cx),       _py(cy + r)),
        (int(cx + r),   _py(cy)),
        (int(cx),       _py(cy - r)),
        (int(cx - r),   _py(cy)),
    ]
    pygame.draw.polygon(_ctx.surf, _ctx.color[:3], pts)


def _draw_rect_outline(x, y, w, h, col):
    if _ctx.surf is None:
        return
    _gl_color(col)
    pygame.draw.rect(_ctx.surf, _ctx.color[:3],
                     pygame.Rect(int(x), _py(y, h), int(w), int(h)),
                     _ctx.line_width)


def _draw_rect_fill(x, y, w, h):
    if _ctx.surf is None:
        return
    pygame.draw.rect(_ctx.surf, _ctx.color,
                     pygame.Rect(int(x), _py(y, h), int(w), int(h)))


def _draw_rect_gl(x, y, w, h, col):
    _gl_color(col)
    _draw_rect_fill(x, y, w, h)


def _alpha_blit_rgb(
    dst: "pygame.Surface",
    src: "pygame.Surface",
    dest: tuple[int, int],
) -> None:
    dx, dy = int(dest[0]), int(dest[1])
    sw, sh = src.get_size()
    dw, dh = dst.get_size()
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return
    x0 = max(0, dx)
    y0 = max(0, dy)
    x1 = min(dw, dx + sw)
    y1 = min(dh, dy + sh)
    if x1 <= x0 or y1 <= y0:
        return

    sx0 = x0 - dx
    sy0 = y0 - dy
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    dst_rgb = pygame.surfarray.pixels3d(dst)
    src_rgb = pygame.surfarray.pixels3d(src)
    src_a = pygame.surfarray.pixels_alpha(src)
    try:
        dst_view = dst_rgb[x0:x1, y0:y1].astype(_np.float32, copy=False)
        src_view = src_rgb[sx0:sx1, sy0:sy1].astype(_np.float32, copy=False)
        alpha = (src_a[sx0:sx1, sy0:sy1].astype(_np.float32, copy=False) / 255.0)[..., None]
        blended = src_view * alpha + dst_view * (1.0 - alpha)
        dst_rgb[x0:x1, y0:y1] = blended.clip(0.0, 255.0).astype(_np.uint8)
    finally:
        del dst_rgb
        del src_rgb
        del src_a

# ─────────────────────────────────────────────────────────────────────────────
# Text overlay  (pygame surface drawn as a GL pixel buffer)
# ─────────────────────────────────────────────────────────────────────────────

class _TextOverlay:
    def __init__(self, w, h):
        self.surface  = pygame.Surface((w, h), pygame.SRCALPHA)
        self.font_sm  = pygame.font.SysFont("monospace", 11)
        self.font_md  = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_ttl = pygame.font.SysFont("monospace", 12, bold=True)
        self.w, self.h = w, h

    def resize(self, w, h):
        self.surface   = pygame.Surface((w, h), pygame.SRCALPHA)
        self.w, self.h = w, h

    def begin(self):
        self.surface.fill((0, 0, 0, 0))

    # All drawing below uses pygame coords (y from top).
    # gl_y helpers convert GL-space coords (y from bottom) to pygame coords.

    def _gl_to_py(self, gl_y: float, rect_h: float = 0.0) -> int:
        return int(self.h - gl_y - rect_h)

    def text(self, txt: str, x: float, y: float,
             col=(220, 220, 220), small: bool = True):
        """Draw text in pygame coords (y from top)."""
        f  = self.font_sm if small else self.font_md
        ts = f.render(txt, True, col)
        self.surface.blit(ts, (int(x), int(y)))

    def gl_text(self, txt: str, gl_x: float, gl_y: float,
                col=(220, 220, 220), small: bool = True):
        """Draw text positioned in GL space (y from bottom)."""
        f    = self.font_sm if small else self.font_md
        ts   = f.render(txt, True, col)
        py_y = self.h - int(gl_y) - ts.get_height() - 1
        self.surface.blit(ts, (int(gl_x), py_y))

    def gl_button(self, txt: str,
                  gl_x: float, gl_y: float, gl_w: float, gl_h: float,
                  col_bg=(50, 60, 70), col_text=(200, 220, 200),
                  hover: bool = False, active: bool = False,
                  bold: bool = False):
        """Draw a filled button in GL coordinates."""
        py_y = self._gl_to_py(gl_y, gl_h)
        bg   = (90, 110, 80) if active else (75, 88, 100) if hover else col_bg
        pygame.draw.rect(self.surface, bg,
                         (int(gl_x), py_y, int(gl_w), int(gl_h)))
        pygame.draw.rect(self.surface, (110, 130, 120),
                         (int(gl_x), py_y, int(gl_w), int(gl_h)), 1)
        f  = self.font_ttl if bold else self.font_sm
        ts = f.render(txt, True, col_text)
        cx = int(gl_x) + max(0, (int(gl_w) - ts.get_width())  // 2)
        cy = py_y       + max(0, (int(gl_h) - ts.get_height()) // 2)
        self.surface.blit(ts, (cx, cy))

    def gl_dropdown_list(self, items: list, current: str,
                         hover_idx: int,
                         gl_x: float, gl_y_top: float,
                         gl_w: float, item_h: float = 16.0):
        """Draw an open dropdown list in GL space, opening downward from gl_y_top."""
        for i, item in enumerate(items):
            iy_gl = gl_y_top - (i + 1) * item_h
            is_cur   = (item == current)
            is_hover = (i == hover_idx)
            bg = (70, 100, 70) if is_cur else (60, 80, 90) if is_hover else (35, 42, 48)
            py_y = self._gl_to_py(iy_gl, item_h)
            pygame.draw.rect(self.surface, bg,
                             (int(gl_x), py_y, int(gl_w), int(item_h)))
            pygame.draw.rect(self.surface, (80, 100, 90),
                             (int(gl_x), py_y, int(gl_w), int(item_h)), 1)
            col = (180, 240, 160) if is_cur else (200, 215, 210)
            ts  = self.font_sm.render(item, True, col)
            self.surface.blit(ts, (int(gl_x) + 4,
                                   py_y + max(0, (int(item_h) - ts.get_height()) // 2)))

    def blit_to(self, target_surf: "pygame.Surface") -> None:
        """Blit the overlay surface onto target_surf at (0, 0)."""
        target_surf.blit(self.surface, (0, 0))

# ─────────────────────────────────────────────────────────────────────────────
# Knob widget
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Knob:
    label:  str
    v_min:  float
    v_max:  float
    value:  float
    cx:     float = 0.0
    cy:     float = 0.0
    radius: float = 18.0
    log:    bool  = False
    fmt:    str   = "{:.1f}"

    def norm(self):
        if self.log:
            lo = math.log(self.v_min); hi = math.log(self.v_max)
            return (math.log(max(self.value, self.v_min)) - lo) / max(hi - lo, 1e-9)
        return (self.value - self.v_min) / max(self.v_max - self.v_min, 1e-9)

    def set_norm(self, n):
        n = max(0.0, min(1.0, n))
        if self.log:
            lo = math.log(self.v_min); hi = math.log(self.v_max)
            self.value = math.exp(lo + n * (hi - lo))
        else:
            self.value = self.v_min + n * (self.v_max - self.v_min)

    def hit(self, px, py):
        return (px - self.cx) ** 2 + (py - self.cy) ** 2 <= (self.radius + 4) ** 2


@dataclass
class _ChannelState:
    key: str
    rule_tree: Optional[EnvelopeRuleTree] = None
    time_stretch: bool = False
    raw_signals: dict[str, torch.Tensor] = field(default_factory=dict)
    display_signals: dict[str, torch.Tensor] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Spline polyline builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_display_polylines(curve: ParametricCurve, steps: int = 512) -> list:
    t_flat = torch.linspace(0.0, 1.0, steps, dtype=torch.float64)
    v_flat = curve.evaluate_normalized(t_flat).abs().to(torch.float64)
    sorted_pts = sorted(curve.points, key=lambda p: p.t)
    raw_chains = _split_into_chains(sorted_pts)
    polys = []
    for chain in raw_chains:
        if not chain:
            continue
        t_lo = chain[0].t;  t_hi = chain[-1].t
        mask = (t_flat >= t_lo) & (t_flat <= t_hi)
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            continue
        polys.append([(float(t_flat[i]), float(v_flat[i])) for i in idxs])
    return polys

# ─────────────────────────────────────────────────────────────────────────────
# Playback downsampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _downsample_to_pts(arr, w_px: int) -> list:
    arr_np = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else arr
    arr_np = _np.asarray(arr_np)
    n = len(arr_np)
    if n == 0:
        return []
    bucket = max(1, n // max(1, w_px))
    pts = []
    for i in range(min(w_px, n)):
        lo = i * bucket;  hi = min(n, lo + bucket)
        seg = arr_np[lo:hi]
        if len(seg) == 0:
            continue
        idx = int(_np.abs(seg).argmax())
        pts.append(((lo + hi) * 0.5 / n, float(_np.abs(seg[idx]))))
    return pts


def _downsample_env_to_pts(arr, w_px: int) -> list:
    arr_np = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else arr
    arr_np = _np.asarray(arr_np)
    if _np.iscomplexobj(arr_np):
        arr_np = arr_np.real
    n = len(arr_np)
    if n == 0:
        return []
    bucket = max(1, n // max(1, w_px))
    pts = []
    for i in range(min(w_px, n)):
        lo = i * bucket;  hi = min(n, lo + bucket)
        seg = arr_np[lo:hi]
        if len(seg) == 0:
            continue
        pts.append(((lo + hi) * 0.5 / n, float(seg.mean())))
    return pts


def _downsample_signal_lane_pts(sig: torch.Tensor | Any, w_px: int, lane: int) -> list:
    sig_t = torch.as_tensor(sig, dtype=torch.complex128).reshape(-1)
    if sig_t.numel() <= 0:
        return []
    arr_np = torch.view_as_real(sig_t)[:, lane].to(torch.float64).detach().cpu().numpy()
    n = len(arr_np)
    if n == 0:
        return []
    bucket = max(1, n // max(1, w_px))
    pts = []
    for i in range(min(w_px, n)):
        lo = i * bucket; hi = min(n, lo + bucket)
        seg = arr_np[lo:hi]
        if len(seg) == 0:
            continue
        idx = int(_np.abs(seg).argmax())
        pts.append(((lo + hi) * 0.5 / n, float(seg[idx])))  # signed peak preserves oscillation
    return pts


def _downsample_analytic_pts(sig, w_px: int) -> list:
    """Downsample complex signal to (t_frac, re_norm, im_norm) triples, peak-normalised.

    Used for the 3-D oblique EM-wave projection in the output panel.
    """
    sig_t = torch.as_tensor(sig, dtype=torch.complex128).reshape(-1)
    if sig_t.numel() <= 0:
        return []
    peak = float(torch.max(torch.abs(sig_t)).item())
    if peak < 1e-12:
        peak = 1.0
    real_np = torch.view_as_real(sig_t)[:, 0].to(torch.float64).detach().cpu().numpy() / peak
    imag_np = torch.view_as_real(sig_t)[:, 1].to(torch.float64).detach().cpu().numpy() / peak
    n = len(real_np)
    bucket = max(1, n // max(1, w_px))
    pts = []
    for i in range(min(w_px, n)):
        lo = i * bucket
        hi = min(n, lo + bucket)
        if hi <= lo:
            continue
        seg_re = real_np[lo:hi]
        seg_im = imag_np[lo:hi]
        idx = int(_np.abs(seg_re + 1j * seg_im).argmax())
        pts.append(((lo + hi) * 0.5 / n, float(seg_re[idx]), float(seg_im[idx])))
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Y-axis tick label helper
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_physical(v: float, v_lo: float, v_hi: float) -> str:
    span = abs(v_hi - v_lo)
    if span == 0.0:
        return f"{v:.2f}"
    if abs(v) < 1e-9 * span:
        return "0"
    if span >= 100:
        return f"{v:.0f}"
    if span >= 1:
        return f"{v:.1f}"
    return f"{v:.3f}"

# ═════════════════════════════════════════════════════════════════════════════
# Main editor class
# ═════════════════════════════════════════════════════════════════════════════

class ParametricCurveEditor:
    _PT_SNAP = 8
    _MK_SNAP = 6

    def __init__(self,
                 amp_curve:    ParametricCurve,
                 chirp_curve:  ParametricCurve,
                 signal_curve: Optional[ParametricCurve] = None,
                 w:             int = 960,
                 h:             int = 1040,
                 library_folder: str = "envelopes"):
        self._curves = [
            amp_curve,
            chirp_curve,
            signal_curve if signal_curve is not None else default_blank("analytic"),
        ]
        self.focused_panel = 0
        self.w, self.h     = w, h
        self.library_folder = library_folder
        self._overlay: Optional[_TextOverlay] = None
        self._panel_count = len(self._curves)

        # ── per-panel config ──────────────────────────────────────────────────
        self._panel_roles  = ["amplitude", "chirp", "output"]
        self._panel_channels = ["A", "A", "B"]
        # Height mode per panel: "collapsed" | "small" | "medium" | "large"
        self._panel_height_modes: list[str] = ["medium", "medium", "small"]
        # Vertical scroll offset for the panels viewport (pixels, clamped ≥ 0)
        self._panels_scroll_y: int = 0
        self._channels: dict[str, _ChannelState] = {
            "A": _ChannelState(key="A", rule_tree=EnvelopeRuleTree.default()),
            "B": _ChannelState(key="B", rule_tree=None),
        }
        # v_lo / v_hi live directly on the ParametricCurve; we just read them
        # y_scale also lives on the ParametricCurve (self._curves[i].y_scale)

        # ── per-panel edit state ──────────────────────────────────────────────
        self._drag_pt   = [None] * self._panel_count
        self._drag_mk   = [None] * self._panel_count
        self._hover_pt  = [None] * self._panel_count
        self._hover_mk  = [None] * self._panel_count
        self._hover_ri  = [None] * self._panel_count
        self._dirty     = [True] * self._panel_count
        self._display_polys: list = [[] for _ in range(self._panel_count)]

        # title in-line editing
        self._title_editing = False
        self._title_panel   = 0
        self._title_buf     = ""

        # marker label in-line editing
        self._editing_label = False
        self._edit_panel    = 0
        self._edit_mk_idx   = -1
        self._edit_buf      = ""

        # open dropdown: None or ("role"|"scale", panel_idx)
        self._open_dropdown: Optional[tuple] = None
        self._dropdown_hover_idx = -1

        # ── mouse state ───────────────────────────────────────────────────────
        self._mouse_t     = 0.0
        self._mouse_v     = 0.0
        self._mouse_panel = 0
        self._hover_header_btn: Optional[tuple] = None  # (panel_idx, key)

        # ── stored render pts (populated externally via _refresh_render_points)
        self._render_buf:    Optional[Any] = None
        self._wave_pts:  list = []
        self._amp_bias_pts: list = []
        self._amp_pts:   list = []
        self._amp_total_pts: list = []
        self._chirp_bias_pts: list = []
        self._chirp_pts: list = []
        self._chirp_total_pts: list = []
        self._sig_re_pts: list = []
        self._sig_im_pts: list = []
        self._analytic_pts: list = []

        # ── phase-rotation animation for the output/analytic panel ────────────
        # Angle advances at _phase_rotation_hz full turns per second.
        self._phase_rotation_angle: float = 0.0
        self._phase_rotation_hz:    float = 0.2   # ~1 full turn every 5 s
        self._phase_rotation_last_t: Optional[float] = None
        self._sync_all_channel_structures()

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def curve(self) -> ParametricCurve:
        return self._curves[self.focused_panel]

    @property
    def active_channel_key(self) -> str:
        return self._panel_channels[self.focused_panel]

    def _panel_channel_state(self, panel_idx: int) -> _ChannelState:
        key = self._panel_channels[panel_idx]
        return self._channel_state(key)

    def _channel_state(self, channel_key: str) -> _ChannelState:
        state = self._channels.get(channel_key)
        if state is None:
            state = _ChannelState(key=channel_key)
            self._channels[channel_key] = state
        return state

    def _active_channel_state(self) -> _ChannelState:
        return self._panel_channel_state(self.focused_panel)

    def _panel_is_curve(self, panel_idx: int) -> bool:
        return self._panel_roles[panel_idx] not in ("analytic", "output")

    def _panel_supports_load(self, panel_idx: int) -> bool:
        return self._panel_is_curve(panel_idx)

    def _panel_save_label(self, panel_idx: int) -> str:
        return "WAV" if self._panel_roles[panel_idx] in ("analytic", "output") else "SAVE"

    def _channel_curve_indices(self, channel_key: str) -> list[int]:
        return [
            i for i in range(self._panel_count)
            if self._panel_channels[i] == channel_key and self._panel_is_curve(i)
        ]

    def _channel_primary_curve_index(self, channel_key: str) -> Optional[int]:
        curve_idxs = self._channel_curve_indices(channel_key)
        return curve_idxs[0] if curve_idxs else None

    def _channel_role_index(self, channel_key: str, role: str) -> Optional[int]:
        for i in range(self._panel_count):
            if self._panel_channels[i] == channel_key and self._panel_roles[i] == role:
                return i
        return None

    def _channel_amp_curve(self, channel_key: str) -> Optional[ParametricCurve]:
        idx = self._channel_role_index(channel_key, "amplitude")
        return self._curves[idx] if idx is not None else None

    def _channel_chirp_curve(self, channel_key: str) -> Optional[ParametricCurve]:
        idx = self._channel_role_index(channel_key, "chirp")
        return self._curves[idx] if idx is not None else None

    def _channel_render_curves(self, channel_key: str) -> tuple[Optional[ParametricCurve], Optional[ParametricCurve]]:
        return self._channel_amp_curve(channel_key), self._channel_chirp_curve(channel_key)

    def _channel_rule_tree(self, channel_key: str) -> Optional[EnvelopeRuleTree]:
        return self._channel_state(channel_key).rule_tree

    def _ensure_channel_for_panel(self, panel_idx: int) -> _ChannelState:
        key = self._panel_channels[panel_idx]
        state = self._channels.get(key)
        if state is None:
            state = _ChannelState(key=key)
            self._channels[key] = state
        if self._panel_is_curve(panel_idx) and state.rule_tree is None:
            state.rule_tree = EnvelopeRuleTree.default()
        return state

    def _sync_all_channel_structures(self) -> None:
        seen: set[str] = set()
        for panel_idx in range(self._panel_count):
            key = self._panel_channels[panel_idx]
            if key in seen:
                continue
            seen.add(key)
            src_idx = self._channel_primary_curve_index(key)
            if src_idx is None:
                continue
            self._sync_structure(src_idx)

    # ── layout helpers (height-mode & scroll aware) ───────────────────────────

    def _prect(self, panel_idx: int) -> tuple:
        """Return (x0, y0, pw, ph) for the plot area of panel_idx.
        Uses per-panel height modes and the current panels scroll offset.
        Panel 0 is at the bottom of the stack; higher indices stack upward.
        """
        _, vy, _, _ = _panels_view_gl_rect(self.w, self.h)
        cum = 0
        for i in range(panel_idx):
            cum += _HEADER_H + _PANEL_HEIGHT_PX.get(self._panel_height_modes[i], 200) + _PANEL_GAP
        y0 = vy + cum - self._panels_scroll_y
        ph = float(_PANEL_HEIGHT_PX.get(self._panel_height_modes[panel_idx], 200))
        return float(_ML), float(y0), float(self.w - _ML - _MR), ph

    def _hrect(self, panel_idx: int) -> tuple:
        """Return (hx, hy, hw, hh) for the header bar of panel_idx."""
        x0, y0, pw, ph = self._prect(panel_idx)
        return x0, y0 + ph, pw, float(_HEADER_H)

    def _panel_at_pos(self, py: float) -> int:
        """Return which panel a GL y-coordinate belongs to (scroll-aware)."""
        for pi in reversed(range(self._panel_count)):
            x0, y0, pw, ph = self._prect(pi)
            if y0 <= py <= y0 + ph + _HEADER_H:
                return pi
        return 0

    def _hbtns(self, panel_idx: int) -> dict:
        """Return named header button rects (GL coords) for panel_idx."""
        hx, hy, hw, hh = self._hrect(panel_idx)
        return _header_button_rects_from_rect(hx, hy, hw, hh)

    def _panels_max_scroll(self) -> int:
        """Return the maximum allowed scroll offset (pixels)."""
        _, _, _, vh = _panels_view_gl_rect(self.w, self.h)
        return max(0, _panels_content_h(self._panel_height_modes) - vh)

    def _clamp_panels_scroll(self) -> None:
        self._panels_scroll_y = max(0, min(self._panels_scroll_y, self._panels_max_scroll()))

    # ── dirty / invalidate ────────────────────────────────────────────────────

    def _mark_dirty(self, panel_idx: Optional[int] = None, clear_render: bool = True):
        idx = self.focused_panel if panel_idx is None else panel_idx
        self._dirty[idx] = True
        self._curves[idx]._invalidate()
        if clear_render:
            # Invalidate the stale render result; do NOT pre-bake anything.
            # Audio is produced only by _do_post_render() after a session ends.
            self._render_buf          = None
            self._wave_pts            = []
            self._amp_pts             = []
            self._chirp_pts           = []
            self._sig_re_pts          = []
            self._sig_im_pts          = []

    # ── shared structure sync ─────────────────────────────────────────────────

    def _sync_structure(self, from_idx: int) -> None:
        """Mirror markers and regions from one panel to the other.

        Control points and activation are NOT touched.
        """
        src = self._curves[from_idx]
        channel_key = self._panel_channels[from_idx]
        for tgt_idx in self._channel_curve_indices(channel_key):
            if tgt_idx == from_idx:
                continue
            tgt = self._curves[tgt_idx]
            tgt.markers = [TimeMarker(m.t, m.label, m.pinned) for m in src.markers]
            tgt.regions = {
                k: RegionEffect(v.mode, None, v.lfo_ref, v.lfo_depth,
                                v.loop_count, v.gate_threshold)
                for k, v in src.regions.items()
            }
            self._dirty[tgt_idx] = True
            tgt._invalidate()
        if self._panel_is_curve(from_idx):
            state = self._channel_state(channel_key)
            if state.rule_tree is None:
                state.rule_tree = EnvelopeRuleTree.default()

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _from_screen(self, px, py, panel_idx):
        x0, y0, pw, ph = self._prect(panel_idx)
        t = (px - x0) / max(pw, 1e-9)
        v = (py - y0) / max(ph, 1e-9)
        return t, v

    def _clamp(self, t, v):
        return max(0.0, min(1.0, t)), max(0.0, min(1.0, v))

    # ── hit testing ───────────────────────────────────────────────────────────

    def _nearest_point(self, px, py, panel_idx):
        x0, y0, pw, ph = self._prect(panel_idx)
        curve = self._curves[panel_idx]
        best_d, best_i = float(self._PT_SNAP ** 2), None
        for i, pt in enumerate(curve.points):
            d = (px - (x0 + pt.t * pw)) ** 2 + (py - (y0 + pt.v * ph)) ** 2
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _nearest_marker(self, px, panel_idx):
        x0, _, pw, _ = self._prect(panel_idx)
        best_d, best_i = float(self._MK_SNAP), None
        for i, mk in enumerate(self._curves[panel_idx].markers):
            d = abs(px - (x0 + mk.t * pw))
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _header_btn_at(self, px, py) -> Optional[tuple]:
        """Return (panel_idx, btn_key) if px,py is inside a header button."""
        for pi in range(self._panel_count):
            rects = self._hbtns(pi)
            for key, rect in rects.items():
                if _hit_rect(px, py, rect):
                    return (pi, key)
        return None

    def _dropdown_item_at(self, px, py) -> int:
        """Return item index under cursor in the open dropdown, or -1."""
        if self._open_dropdown is None:
            return -1
        kind, pi = self._open_dropdown
        rects = self._hbtns(pi)
        btn_rect = rects["role"] if kind == "role" else rects["scale"] if kind == "scale" else rects["channel"]
        bx, by, bw, _ = btn_rect
        items = list(_PARAM_ROLES) if kind == "role" else list(_SCALE_MODES) if kind == "scale" else list(_CHANNEL_KEYS)
        item_h   = 16.0
        for i in range(len(items)):
            iy = by - (i + 1) * item_h
            if _hit_rect(px, py, (bx, iy, bw, item_h)):
                return i
        return -1

    # ── role / scale change ───────────────────────────────────────────────────

    def _set_panel_role(self, pi: int, role: str) -> None:
        self._panel_roles[pi] = role
        v_lo, v_hi, scale = _ROLE_DEFAULTS.get(role, (0.0, 1.0, "linear"))
        c = self._curves[pi]
        if c.v_lo == c.v_hi or (c.v_lo, c.v_hi) != (_ROLE_DEFAULTS.get(self._panel_roles[pi], (c.v_lo, c.v_hi, ""))[0],
                                                       _ROLE_DEFAULTS.get(self._panel_roles[pi], (c.v_lo, c.v_hi, ""))[1]):
            c.v_lo    = v_lo
            c.v_hi    = v_hi
            c.y_scale = scale
        self._ensure_channel_for_panel(pi)
        if self._panel_is_curve(pi):
            self._sync_structure(pi)
        self._mark_dirty(pi)

    def _set_panel_scale(self, pi: int, scale: str) -> None:
        self._curves[pi].y_scale = scale
        self._mark_dirty(pi, clear_render=False)

    def _set_panel_channel(self, pi: int, channel_key: str) -> None:
        old_key = self._panel_channels[pi]
        if channel_key == old_key:
            return
        src_state = self._channel_state(old_key)
        self._panel_channels[pi] = channel_key
        dst_state = self._ensure_channel_for_panel(pi)
        if self._panel_is_curve(pi):
            peer_idx = self._channel_primary_curve_index(channel_key)
            if peer_idx is not None and peer_idx != pi:
                src_curve = self._curves[peer_idx]
                dst_curve = self._curves[pi]
                dst_curve.markers = [TimeMarker(m.t, m.label, m.pinned) for m in src_curve.markers]
                dst_curve.regions = {
                    k: RegionEffect(v.mode, None, v.lfo_ref, v.lfo_depth,
                                    v.loop_count, v.gate_threshold)
                    for k, v in src_curve.regions.items()
                }
                if dst_state.rule_tree is None and self._channel_state(channel_key).rule_tree is not None:
                    dst_state.rule_tree = copy.deepcopy(self._channel_state(channel_key).rule_tree)
            elif dst_state.rule_tree is None and src_state.rule_tree is not None:
                dst_state.rule_tree = copy.deepcopy(src_state.rule_tree)
        self._sync_all_channel_structures()
        self._mark_dirty(pi)

    def _toggle_panel_channel_stretch(self, pi: int) -> None:
        state = self._ensure_channel_for_panel(pi)
        state.time_stretch = not state.time_stretch
        if state.raw_signals:
            state.display_signals = normalize_channel_complex_signals(
                state.raw_signals,
                time_stretch=state.time_stretch,
            )
        self._mark_dirty(pi, clear_render=False)

    # ── note / render ─────────────────────────────────────────────────────────

    # ── event handlers ────────────────────────────────────────────────────────

    def on_mouse_down(self, button, px, py, mods):
        # Mouse-wheel scroll for panels area
        if button in (4, 5):
            _, vy, _, vh = _panels_view_gl_rect(self.w, self.h)
            if vy <= py <= vy + vh:
                delta = _PANELS_SCROLL_STEP if button == 4 else -_PANELS_SCROLL_STEP
                self._panels_scroll_y = max(0, min(
                    self._panels_scroll_y - delta,
                    self._panels_max_scroll()))
            return

        # Close any open dropdown on a click elsewhere
        if self._open_dropdown is not None:
            item = self._dropdown_item_at(px, py)
            if item >= 0:
                kind, pi = self._open_dropdown
                items = (
                    list(_PARAM_ROLES) if kind == "role"
                    else list(_SCALE_MODES) if kind == "scale"
                    else list(_CHANNEL_KEYS)
                )
                if kind == "role":
                    self._set_panel_role(pi, items[item])
                elif kind == "scale":
                    self._set_panel_scale(pi, items[item])
                else:
                    self._set_panel_channel(pi, items[item])
            self._open_dropdown = None
            return

        # Commit any in-progress title/label edit on outside click
        if self._title_editing:
            self._commit_title()
            return
        if self._editing_label:
            self._commit_label()
            return

        # Header buttons
        hb = self._header_btn_at(px, py)
        if hb is not None and button == 1:
            pi, key = hb
            self.focused_panel = pi
            if key == "save":
                if self._panel_roles[pi] == "analytic":
                    self._save_channel_wav(self._panel_channels[pi])
                else:
                    self._curves[pi].save(self.library_folder)
            elif key == "load":
                if self._panel_supports_load(pi):
                    self._do_load(pi)
            elif key == "title":
                self._title_editing = True
                self._title_panel   = pi
                self._title_buf     = self._curves[pi].name
            elif key == "role":
                self._open_dropdown = ("role", pi)
            elif key == "channel":
                self._open_dropdown = ("channel", pi)
            elif key == "stretch":
                self._toggle_panel_channel_stretch(pi)
            elif key == "collapse":
                cur = self._curves[pi].complex_collapse_mode
                self._curves[pi].complex_collapse_mode = "abs" if cur == "real" else "real"
            elif key == "hmode":
                modes = list(_PANEL_HEIGHT_MODES)
                cur = self._panel_height_modes[pi]
                self._panel_height_modes[pi] = modes[(modes.index(cur) + 1) % len(modes)]
                self._clamp_panels_scroll()
            return

        # Plot area
        panel = self._panel_at_pos(py)
        if panel != self.focused_panel:
            self.focused_panel = panel

        t, v    = self._from_screen(px, py, panel)
        in_plot = 0.0 <= t <= 1.0 and 0.0 <= v <= 1.0
        if not self._panel_is_curve(panel):
            return

        if button == 1:
            if bool(mods & KMOD_CTRL) and in_plot:
                self.curve.add_marker(t, "")
                self._sync_structure(panel)
                self._mark_dirty()
                return
            mi = self._nearest_marker(px, panel)
            if mi is not None:
                self._drag_mk[panel] = mi
                return
            pi2 = self._nearest_point(px, py, panel)
            if pi2 is not None:
                self._drag_pt[panel] = pi2
                return
            if in_plot:
                self.curve.add_point(t, v)
                self._mark_dirty()
        elif button == 3:
            mi = self._nearest_marker(px, panel)
            if mi is not None:
                self.curve.remove_marker(mi)
                self._sync_structure(panel)
                self._mark_dirty()
                return
            pi2 = self._nearest_point(px, py, panel)
            if pi2 is not None:
                self.curve.remove_point(pi2)
                self._mark_dirty()

    def on_mouse_up(self, button):
        if button == 1:
            for i in range(self._panel_count):
                self._drag_pt[i] = None
                self._drag_mk[i] = None

    def on_mouse_move(self, px, py):
        panel = self._panel_at_pos(py)
        t, v  = self._from_screen(px, py, panel)
        self._mouse_t     = t
        self._mouse_v     = v
        self._mouse_panel = panel

        for pi in range(self._panel_count):
            self._hover_pt[pi] = self._nearest_point(px, py, pi)
            self._hover_mk[pi] = self._nearest_marker(px, pi)
            self._hover_ri[pi] = self._curves[pi].region_index_at(
                max(0.0, min(1.0, t)))

        self._hover_header_btn = self._header_btn_at(px, py)

        # Update dropdown hover
        if self._open_dropdown is not None:
            self._dropdown_hover_idx = self._dropdown_item_at(px, py)

        fp = self.focused_panel
        if not self._panel_is_curve(fp):
            return
        if self._drag_pt[fp] is not None:
            tc, vc = self._clamp(t, v)
            self._curves[fp].points[self._drag_pt[fp]].t = tc
            self._curves[fp].points[self._drag_pt[fp]].v = vc
            self._curves[fp].points.sort(key=lambda p: p.t)
            self._mark_dirty()
        elif self._drag_mk[fp] is not None:
            self._curves[fp].slide_marker(self._drag_mk[fp], t)
            self._sync_structure(fp)
            self._mark_dirty()

    def on_key_down(self, key, mods):
        # Title edit mode
        if self._title_editing:
            if key == K_RETURN:
                self._commit_title()
            elif key == K_BACKSPACE:
                self._title_buf = self._title_buf[:-1]
            return

        # Marker label edit mode
        if self._editing_label:
            if key == K_RETURN:
                self._commit_label()
            elif key == K_BACKSPACE:
                self._edit_buf = self._edit_buf[:-1]
            return

        fp = self.focused_panel
        c  = self.curve

        if key == K_ESCAPE:
            pygame.event.post(pygame.event.Event(QUIT))
        elif key == K_TAB:
            self.focused_panel = (self.focused_panel + 1) % self._panel_count
        elif key == K_d and self._panel_is_curve(fp) and self._hover_pt[fp] is not None:
            c.toggle_break(self._hover_pt[fp])
            self._mark_dirty()
        elif key == K_m and self._panel_is_curve(fp) and self._hover_ri[fp] is not None:
            eff = c.regions.get(self._hover_ri[fp], RegionEffect())
            eff.mode = _REGION_MODES[
                (_REGION_MODES.index(eff.mode) + 1) % len(_REGION_MODES)]
            c.regions[self._hover_ri[fp]] = eff
            self._sync_structure(fp)
            self._mark_dirty()
        elif key == K_t and self._panel_is_curve(fp) and self._hover_mk[fp] is not None:
            self._editing_label = True
            self._edit_panel    = fp
            self._edit_mk_idx   = self._hover_mk[fp]
            self._edit_buf      = c.markers[self._hover_mk[fp]].label
        elif key == K_s:
            if self._panel_roles[fp] in ("analytic", "output"):
                self._save_channel_wav(self._panel_channels[fp])
            else:
                c.save(self.library_folder)
        elif key == K_l:
            if self._panel_supports_load(fp):
                self._do_load(fp)
        elif key == K_a and self._panel_is_curve(fp):
            modes = _ACTIVATION_MODES
            c.activation = modes[(modes.index(c.activation) + 1) % len(modes)]
        elif key == K_EQUALS and self._panel_is_curve(fp):
            c.activation_drive = round(min(20.0, c.activation_drive + 0.5), 2)
        elif key == K_MINUS and self._panel_is_curve(fp):
            c.activation_drive = round(max(0.1, c.activation_drive - 0.5), 2)

    def on_key_up(self, key):
        pass

    def on_text_input(self, char):
        if self._title_editing:
            self._title_buf += char
        elif self._editing_label:
            self._edit_buf += char

    def _commit_title(self):
        name = self._title_buf.strip() or self._curves[self._title_panel].name
        self._curves[self._title_panel].name = name
        self._title_editing = False
        self._title_buf     = ""

    def _commit_label(self):
        fp = self._edit_panel
        if 0 <= self._edit_mk_idx < len(self._curves[fp].markers):
            self._curves[fp].markers[self._edit_mk_idx].label = self._edit_buf
            self._sync_structure(fp)
        self._editing_label = False
        self._edit_buf      = ""
        self._edit_mk_idx   = -1

    def _do_load(self, pi: int) -> None:
        lib = ParametricCurve.load_library(self.library_folder)
        c   = self._curves[pi]
        if c.name in lib:
            loaded = lib[c.name]
            self._curves[pi] = loaded
            self._sync_structure(pi)
            self._mark_dirty(pi)

    def _refresh_render_points(self, channel_key: str) -> None:
        """Rebuild display point lists from channel state's display_signals.

        Channel A: expects "amplitude" and/or "chirp" tensors.
                   Builds _amp_pts, _chirp_pts from those.
                   _wave_pts is built from channel B's "analytic" (background).
        Channel B: expects "analytic" tensor.
                   Builds _sig_re_pts, _sig_im_pts, _render_buf.
        """
        state = self._channel_state(channel_key)
        x0, _, pw, _ = self._prect(0)
        w_px = max(64, int(pw))
        _ZERO = torch.zeros(0, dtype=torch.complex128)

        if channel_key == "B":
            analytic_sig = state.display_signals.get("analytic")
            if analytic_sig is None:
                self._sig_re_pts = []
                self._sig_im_pts = []
                self._analytic_pts  = []
                self._render_buf = None
                return
            self._render_buf = analytic_sig
            # ── advance phase-rotation animation ─────────────────────────────
            now = _time.monotonic()
            if self._phase_rotation_last_t is not None:
                dt = now - self._phase_rotation_last_t
                self._phase_rotation_angle += 2.0 * math.pi * self._phase_rotation_hz * dt
            self._phase_rotation_last_t = now
            # Rotate the complex signal for display only; raw data is untouched.
            rotated = analytic_sig * torch.exp(
                torch.tensor(1j * self._phase_rotation_angle, dtype=torch.complex128)
            )
            self._sig_re_pts = _downsample_signal_lane_pts(rotated, w_px, 0)
            self._sig_im_pts = _downsample_signal_lane_pts(rotated, w_px, 1)
            # Build analytic pts from RAW signal — view projection applied at draw time
            self._analytic_pts  = _downsample_analytic_pts(analytic_sig, w_px)
            # Also build background wave from analytic envelope magnitude
            self._wave_pts = _downsample_to_pts(torch.abs(analytic_sig), w_px)
        else:
            amp_bias_sig   = state.display_signals.get("amplitude_bias")
            amp_sig        = state.display_signals.get("amplitude")
            amp_total_sig  = state.display_signals.get("amplitude_total")
            chirp_bias_sig = state.display_signals.get("chirp_bias")
            chirp_sig      = state.display_signals.get("chirp")
            chirp_total_sig = state.display_signals.get("chirp_total")
            self._amp_bias_pts   = _downsample_env_to_pts(amp_bias_sig   if amp_bias_sig   is not None else _ZERO, w_px)
            self._amp_pts        = _downsample_env_to_pts(amp_sig        if amp_sig        is not None else _ZERO, w_px)
            self._amp_total_pts  = _downsample_env_to_pts(amp_total_sig  if amp_total_sig  is not None else _ZERO, w_px)
            self._chirp_bias_pts = _downsample_env_to_pts(chirp_bias_sig if chirp_bias_sig is not None else _ZERO, w_px)
            self._chirp_pts      = _downsample_env_to_pts(chirp_sig      if chirp_sig      is not None else _ZERO, w_px)
            self._chirp_total_pts = _downsample_env_to_pts(chirp_total_sig if chirp_total_sig is not None else _ZERO, w_px)

    def _save_channel_wav(self, channel_key: str) -> Optional[str]:
        if not _HAS_NP:
            return None
        state = self._channel_state(channel_key)
        sig = state.display_signals.get("analytic")
        if sig is None:
            sig = state.raw_signals.get("analytic")
        if sig is None or int(torch.as_tensor(sig).numel()) <= 0:
            print(f"[save] no analytic signal available on channel {channel_key}")
            return None
        os.makedirs(self.library_folder, exist_ok=True)
        base = self._curves[self.focused_panel].name or f"channel_{channel_key.lower()}"
        stamp = _time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.library_folder, f"{base}_{channel_key}_{stamp}.wav")
        frames = torch.view_as_real(torch.as_tensor(sig, dtype=torch.complex128).reshape(-1))
        wavfile.write(path, _PREVIEW_SR, frames.detach().cpu().numpy().astype(_np.float64, copy=False))
        print(f"[save] analytic wav -> {path}")
        return path

    # ── resize ────────────────────────────────────────────────────────────────

    def resize(self, w, h):
        self.w, self.h = w, h
        if self._overlay:
            self._overlay.resize(w, h)
        self._dirty = [True] * self._panel_count
        self._clamp_panels_scroll()
        if self._render_buf is not None:
            self._refresh_render_points("B")

    # ═══════════════════════════════════════════════════════════════════════════
    # Drawing
    # ═══════════════════════════════════════════════════════════════════════════

    def _draw_panel_bg_regions(self, x0, y0, pw, ph, pi):
        if not self._panel_is_curve(pi):
            return
        c        = self._curves[pi]
        sorted_m = sorted(c.markers, key=lambda m: m.t)
        bounds   = [0.0] + [m.t for m in sorted_m] + [1.0]
        for ri in range(len(bounds) - 1):
            eff = c.regions.get(ri, RegionEffect())
            col = _REGION_COLORS.get(eff.mode, _REGION_COLORS["normal"])
            _gl_color(col)
            _draw_rect_fill(x0 + bounds[ri] * pw, y0,
                            (bounds[ri + 1] - bounds[ri]) * pw, ph)

    def _draw_grid(self, x0, y0, pw, ph, curve: ParametricCurve):
        glLineWidth(1.0)
        _gl_color(_COL_GRID)
        for i in range(1, 10):
            gx = x0 + pw * i / 10.0
            gy = y0 + ph * i / 10.0
            _draw_line(gx, y0, gx, y0 + ph)
            _draw_line(x0, gy, x0 + pw, gy)
        _gl_color(_COL_AXIS)
        glLineWidth(1.5)
        for ax0, ay0, ax1, ay1 in [
            (x0, y0, x0 + pw, y0),     (x0, y0 + ph, x0 + pw, y0 + ph),
            (x0, y0, x0, y0 + ph),     (x0 + pw, y0, x0 + pw, y0 + ph),
        ]:
            _draw_line(ax0, ay0, ax1, ay1)
        # Zero line for bipolar panels (v_lo < 0 < v_hi)
        if curve.v_lo < 0.0 < curve.v_hi:
            v_zero = _y_from_physical(0.0, curve.v_lo, curve.v_hi, curve.y_scale)
            yz = y0 + v_zero * ph
            _gl_color(_COL_GRID_ZERO)
            glLineWidth(1.5)
            _draw_line(x0, yz, x0 + pw, yz)

    def _draw_y_axis_labels(self, x0, y0, pw, ph, curve: ParametricCurve, ov):
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        for vn in ticks:
            v_phys = _y_to_physical(vn, curve.v_lo, curve.v_hi, curve.y_scale)
            gy     = y0 + vn * ph
            label  = _fmt_physical(v_phys, curve.v_lo, curve.v_hi)
            ov.gl_text(label, x0 - _ML + 2, gy - 1, col=(110, 125, 130))

    def _draw_rendered_overlay(self, x0, y0, pw, ph, pi):
        if _ctx.surf is None:
            return
        role = self._panel_roles[pi]
        panel_w = max(1, int(math.ceil(pw)) + 2)
        panel_h = max(1, int(math.ceil(ph)) + 2)
        x_base = int(x0)
        y_base = int(y0)
        dest_xy = (x_base, self.h - y_base - panel_h)

        def _panel_py(y_abs: float) -> int:
            return int(round((self.h - 1 - y_abs) - (self.h - 1 - y_base - (panel_h - 1))))

        def _new_layer() -> "pygame.Surface":
            layer = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            layer.fill((0, 0, 0, 0))
            return layer

        # ── helper: draw an envelope fill+line from normalized (t, v) pts ──
        def _draw_env(dst_layer, pts, fill_col, line_col, *, draw_line: bool = True):
            if len(pts) < 2:
                return
            fill_rgba = tuple(int(c * 255) for c in fill_col)
            line_rgba = tuple(int(c * 255) for c in line_col)
            for i in range(len(pts) - 1):
                t0_, v0 = pts[i]; t1_, v1 = pts[i + 1]
                v0 = max(0.0, min(1.0, float(v0)))
                v1 = max(0.0, min(1.0, float(v1)))
                xl = int(round(t0_ * pw))
                xr = int(round(t1_ * pw))
                poly = [
                    (xl, _panel_py(y0)),
                    (xr, _panel_py(y0)),
                    (xr, _panel_py(y0 + v1 * ph)),
                    (xl, _panel_py(y0 + v0 * ph)),
                ]
                pygame.draw.polygon(dst_layer, fill_rgba, poly)
            if draw_line:
                line_pts = [
                    (int(round(t_ * pw)), _panel_py(y0 + max(0.0, min(1.0, float(v))) * ph))
                    for t_, v in pts
                ]
                if len(line_pts) >= 2:
                    pygame.draw.lines(dst_layer, line_rgba, False, line_pts, 3)

        def _composite_env(pts, fill_col, line_col, *, draw_line: bool = True):
            layer = _new_layer()
            _draw_env(layer, pts, fill_col, line_col, draw_line=draw_line)
            _alpha_blit_rgb(_ctx.surf, layer, dest_xy)

        def _draw_analytic_waves():
            if len(self._analytic_pts) < 2:
                return
            layer = _new_layer()
            cy = y0 + ph * 0.5
            scale = ph * 0.42
            center_py = _panel_py(cy)

            pygame.draw.line(layer, (55, 55, 55, 255),
                             (0, center_py), (int(round(pw)), center_py), 1)

            c_re = math.cos(self._phase_rotation_angle)
            c_im = math.sin(self._phase_rotation_angle)

            re_screen = []
            im_screen = []
            for t_frac, re, im in self._analytic_pts:
                sx = int(round(t_frac * pw))
                re_screen.append((sx, _panel_py(cy - re * c_re * scale)))
                im_screen.append((sx, _panel_py(cy - im * c_im * scale)))

            step = max(1, len(self._analytic_pts) // 48)
            for i in range(0, len(self._analytic_pts), step):
                pygame.draw.line(layer, (40, 100, 55, 255),
                                 (re_screen[i][0], center_py), re_screen[i], 1)
                pygame.draw.line(layer, (40, 75, 120, 255),
                                 (im_screen[i][0], center_py), im_screen[i], 1)

            if len(re_screen) >= 2:
                pygame.draw.lines(layer, (75, 210, 110, 255), False, re_screen, 2)
            if len(im_screen) >= 2:
                pygame.draw.lines(layer, (75, 148, 230, 255), False, im_screen, 2)
            _alpha_blit_rgb(_ctx.surf, layer, dest_xy)

        if role == "analytic":
            _draw_analytic_waves()

        elif role == "amplitude":
            _composite_env(self._amp_bias_pts, _COL_LAYER_R_FILL, _COL_LAYER_R_LINE)
            _composite_env(self._amp_pts, _COL_LAYER_G_FILL, _COL_LAYER_G_LINE)
            _composite_env(self._amp_total_pts or self._wave_pts, _COL_LAYER_B_FILL, _COL_LAYER_B_LINE)

        elif role == "chirp":
            _composite_env(self._chirp_bias_pts, _COL_LAYER_R_FILL, _COL_LAYER_R_LINE)
            _composite_env(self._chirp_pts, _COL_LAYER_G_FILL, _COL_LAYER_G_LINE)
            _composite_env(self._chirp_total_pts, _COL_LAYER_B_FILL, _COL_LAYER_B_LINE)

        elif role == "output":
            _composite_env(self._amp_total_pts or self._wave_pts, _COL_LAYER_B_FILL, _COL_LAYER_B_LINE, draw_line=False)
            _composite_env(self._chirp_total_pts, _COL_LAYER_G_FILL, _COL_LAYER_G_LINE, draw_line=False)
            _draw_analytic_waves()

    def _draw_spline(self, x0, y0, pw, ph, pi):
        if not self._panel_is_curve(pi):
            return
        c = self._curves[pi]
        if self._dirty[pi]:
            self._display_polys[pi] = _build_display_polylines(c, 512)
            self._dirty[pi] = False
        col = _COL_SPLINE_1 if pi == 1 else _COL_SPLINE_0
        rgb = tuple(int(c * 255) for c in col[:3])
        for poly in self._display_polys[pi]:
            pts_py = [(int(x0 + tn * pw), _py(y0 + vn * ph)) for tn, vn in poly]
            if len(pts_py) >= 2:
                pygame.draw.lines(_ctx.surf, rgb, False, pts_py, 3)

    def _draw_points(self, x0, y0, pw, ph, pi):
        if not self._panel_is_curve(pi):
            return
        c  = self._curves[pi]
        fp = self.focused_panel
        for i, pt in enumerate(c.points):
            sx = x0 + pt.t * pw;  sy = y0 + pt.v * ph
            if i == self._drag_pt[pi]:
                col = _COL_PT_DRAG
            elif i == self._hover_pt[pi] and pi == fp:
                col = _COL_PT_HOVER
            elif pt.break_after:
                col = _COL_PT_BREAK
            else:
                col = _COL_PT_NORMAL
            _gl_color((0.0, 0.0, 0.0, 0.8));  _draw_diamond(sx, sy, 7.0)
            _gl_color(col);                     _draw_diamond(sx, sy, 5.0)

    def _draw_playhead(self, x0, y0, pw, ph):
        pass  # no playback state in passive display mode

    def _draw_markers(self, x0, y0, pw, ph, pi, ov):
        if not self._panel_is_curve(pi):
            return
        fp = self.focused_panel
        for mi, mk in enumerate(self._curves[pi].markers):
            sx  = x0 + mk.t * pw
            col = (_COL_MARKER_PIN if mk.pinned
                   else _COL_MARKER_HOVER if mi == self._hover_mk[pi] and pi == fp
                   else _COL_MARKER)
            _gl_color(col)
            glLineWidth(1.5)
            _draw_line(sx, y0, sx, y0 + ph)
            tc = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
            if mk.label and ov:
                ov.gl_text(mk.label, sx + 2, y0 + ph + 2, col=tc)
            if (self._editing_label and pi == self._edit_panel
                    and mi == self._edit_mk_idx and ov):
                ov.gl_text(self._edit_buf + "|", sx + 2, y0 + 16,
                           col=(255, 230, 80), small=False)

    def _draw_panel_border(self, x0, y0, pw, ph, pi):
        col = _COL_FOCUS_BORDER if pi == self.focused_panel else _COL_UNFOCUS_BDR
        _draw_rect_outline(x0, y0, pw, ph, col)

    def _draw_header(self, pi, ov):
        if not ov:
            return
        hx, hy, hw, hh = self._hrect(pi)
        fp = self.focused_panel
        bg  = _COL_HEADER_FOCUS if pi == fp else _COL_HEADER_IDLE
        _gl_color(bg)
        _draw_rect_fill(hx, hy, hw, hh)

        rects = self._hbtns(pi)
        hb    = self._hover_header_btn

        # Title
        rx, ry, rw, rh = rects["title"]
        is_editing = self._title_editing and self._title_panel == pi
        txt   = (self._title_buf + "|") if is_editing else self._curves[pi].name
        t_col = (230, 210, 100) if pi == fp else (140, 150, 160)
        ov.gl_button(txt, rx, ry, rw, rh,
                     col_bg=(35, 32, 18) if pi == fp else (25, 25, 32),
                     col_text=t_col,
                     hover=(hb == (pi, "title")),
                     bold=True)

        # Role dropdown button
        rx, ry, rw, rh = rects["role"]
        role_txt = self._panel_roles[pi]
        ov.gl_button(role_txt + " \u25be", rx, ry, rw, rh,
                     hover=(hb == (pi, "role")),
                     active=(self._open_dropdown == ("role", pi)))

        # Channel dropdown button
        rx, ry, rw, rh = rects["channel"]
        ch_txt = self._panel_channels[pi]
        ov.gl_button(ch_txt + " \u25be", rx, ry, rw, rh,
                     hover=(hb == (pi, "channel")),
                     active=(self._open_dropdown == ("channel", pi)))

        # Channel time normalization toggle
        rx, ry, rw, rh = rects["stretch"]
        st = self._channel_state(self._panel_channels[pi]).time_stretch
        ov.gl_button("TS" if st else "PAD", rx, ry, rw, rh,
                     col_bg=(40, 54, 40) if st else (34, 34, 42),
                     col_text=(170, 220, 170) if st else (175, 180, 190),
                     hover=(hb == (pi, "stretch")))

        # Scale dropdown button
        rx, ry, rw, rh = rects["scale"]
        scale_txt = self._curves[pi].y_scale
        ov.gl_button(scale_txt + " \u25be", rx, ry, rw, rh,
                     hover=(hb == (pi, "scale")),
                     active=(self._open_dropdown == ("scale", pi)))

        # Complex-collapse mode toggle  (Re = .real  |  |z| = abs)
        rx, ry, rw, rh = rects["collapse"]
        ccm = self._curves[pi].complex_collapse_mode
        ccm_lbl = "Re" if ccm == "real" else "|z|"
        ov.gl_button(ccm_lbl, rx, ry, rw, rh,
                     col_bg=(40, 40, 55) if ccm == "real" else (55, 40, 40),
                     col_text=(160, 180, 230) if ccm == "real" else (230, 170, 140),
                     hover=(hb == (pi, "collapse")))

        # SAVE / LOAD buttons
        rx, ry, rw, rh = rects["save"]
        ov.gl_button(self._panel_save_label(pi), rx, ry, rw, rh,
                     col_bg=(30, 50, 35), col_text=(160, 210, 160),
                     hover=(hb == (pi, "save")))
        rx, ry, rw, rh = rects["load"]
        load_col_bg = (30, 40, 55) if self._panel_supports_load(pi) else (28, 28, 34)
        load_col_txt = (150, 180, 220) if self._panel_supports_load(pi) else (95, 100, 110)
        ov.gl_button("LOAD", rx, ry, rw, rh,
                     col_bg=load_col_bg, col_text=load_col_txt,
                     hover=(hb == (pi, "load")))

        # Height-mode cycle button
        rx, ry, rw, rh = rects["hmode"]
        hm = self._panel_height_modes[pi]
        hm_lbl = {"collapsed": "C", "small": "S", "medium": "M", "large": "L"}.get(hm, "M")
        ov.gl_button(hm_lbl, rx, ry, rw, rh,
                     col_bg=(42, 42, 52), col_text=(180, 200, 180),
                     hover=(hb == (pi, "hmode")))

        # v_lo / v_hi extents shown on right margin
        c = self._curves[pi]
        ov.gl_text(f"[{c.v_lo:.0f}\u2026{c.v_hi:.0f}]",
                   hx + hw + 2, hy + 4, col=(80, 100, 110))

    def _draw_open_dropdown(self, ov):
        if self._open_dropdown is None or ov is None:
            return
        kind, pi = self._open_dropdown
        rects    = self._hbtns(pi)
        btn_rect = rects["role"] if kind == "role" else rects["scale"] if kind == "scale" else rects["channel"]
        bx, by, bw, _ = btn_rect
        items = list(_PARAM_ROLES) if kind == "role" else list(_SCALE_MODES) if kind == "scale" else list(_CHANNEL_KEYS)
        current = self._panel_roles[pi] if kind == "role" else self._curves[pi].y_scale if kind == "scale" else self._panel_channels[pi]
        ov.gl_dropdown_list(items, current, self._dropdown_hover_idx,
                            bx, by, bw, item_h=16.0)

    def _draw_knobs(self, ov):
        pass  # knobs removed

    def _draw_status(self, ov):
        if not ov:
            return
        fp = self.focused_panel
        c  = self.curve
        v_phys = _y_to_physical(max(0.0, min(1.0, self._mouse_v)),
                                 c.v_lo, c.v_hi, c.y_scale)
        act_tag = (f"  act={c.activation}({c.activation_drive:.2f})"
                   if self._panel_is_curve(fp) and c.activation != "none" else "")
        ov.text(
            f"  [{self._panel_roles[fp]}:{self._panel_channels[fp]}]  t={self._mouse_t:.3f}"
            f"  v={v_phys:.3g}  pts={len(c.points)}{act_tag}",
            4, 4, col=(160, 165, 170))
        for i, ln in enumerate([
            "LMB=add/drag  RMB=del  Ctrl+LMB=add marker  Tab=focus/cycle",
            "Role/Ch in header  PAD|TS per channel  Analytic save=complex WAV",
        ]):
            ov.text(ln, 4, self.h - 14 * (2 - i) - 2, col=(80, 88, 96))

    # ── main draw ─────────────────────────────────────────────────────────────

    def _draw_frame(self, *, clear_bg: bool, flip: bool,
                     win_x: int = 0, win_y_bottom: int = 0) -> "pygame.Surface":
        # Ensure frame surface exists at the current logical size
        if (not hasattr(self, '_frame_surf')
                or self._frame_surf.get_size() != (self.w, self.h)):
            self._frame_surf = pygame.Surface((self.w, self.h))
        surf = self._frame_surf

        if clear_bg:
            surf.fill(tuple(int(c * 255) for c in _COL_BG[:3]))

        # Point the module-level drawing context at this surface
        _ctx.surf = surf
        _ctx.h    = self.h

        if self._overlay:
            self._overlay.begin()

        # Clip drawing to the scrollable panels viewport
        vx, vy, vw, vh = _panels_view_gl_rect(self.w, self.h)
        clip_rect = pygame.Rect(int(vx), self.h - int(vy) - int(vh), int(vw), int(vh))
        surf.set_clip(clip_rect)

        for pi in range(self._panel_count):
            x0, y0, pw, ph = self._prect(pi)
            curve = self._curves[pi]

            self._draw_panel_bg_regions(x0, y0, pw, ph, pi)
            self._draw_grid(x0, y0, pw, ph, curve)
            self._draw_rendered_overlay(x0, y0, pw, ph, pi)
            self._draw_spline(x0, y0, pw, ph, pi)
            self._draw_points(x0, y0, pw, ph, pi)
            self._draw_playhead(x0, y0, pw, ph)
            self._draw_panel_border(x0, y0, pw, ph, pi)

            if self._overlay:
                self._draw_markers(x0, y0, pw, ph, pi, self._overlay)
                self._draw_y_axis_labels(x0, y0, pw, ph, curve, self._overlay)
                self._draw_header(pi, self._overlay)

        # Un-clip for knobs and dropdowns (drawn outside the panels viewport)
        surf.set_clip(None)

        self._draw_knobs(self._overlay)

        if self._overlay:
            self._draw_open_dropdown(self._overlay)   # drawn last = on top
            self._draw_status(self._overlay)
            self._overlay.blit_to(surf)

        if flip:
            win = pygame.display.get_surface()
            if win is not None:
                win.blit(surf, (0, 0))
            pygame.display.flip()

        return surf

    def draw_embedded(self, target_surf: "pygame.Surface" = None,
                      dest: tuple = (0, 0)) -> "pygame.Surface":
        """Draw the editor; return the frame surface and optionally blit into target_surf.

        target_surf  — if provided, the frame is blitted at dest (top-left pygame coords)
        dest         — (x, y) blit destination on target_surf
        """
        surf = self._draw_frame(clear_bg=False, flip=False)
        if target_surf is not None:
            target_surf.blit(surf, dest)
        return surf

    def render_to_surface(self, w: int, h: int) -> "pygame.Surface":
        """Render the editor into a pygame.Surface and return it.

        Compatible with the surface-based centre-display pipeline used by all
        other EditorCanvas tabs (e.g. bass_viewer, analytic_driver).
        """
        self.resize(w, h)
        return self._draw_frame(clear_bg=True, flip=False)

    def draw(self):
        self._draw_frame(clear_bg=True, flip=True)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

def run(amp_curve:    Optional[ParametricCurve] = None,
        chirp_curve:  Optional[ParametricCurve] = None,
        signal_curve: Optional[ParametricCurve] = None,
        w:             int = 960,
        h:             int = 820,
        title:         str = "Parametric Curve Editor",
        library_folder: str = "envelopes"):
    if not _HAS_PYGAME:
        raise RuntimeError("pygame is required for the standalone editor UI.")

    if amp_curve   is None: amp_curve   = default_envelope("amplitude")
    if chirp_curve is None: chirp_curve = default_chirp("chirp")

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
    pygame.display.set_caption(title)

    editor = ParametricCurveEditor(amp_curve, chirp_curve, signal_curve,
                                   w=w, h=h, library_folder=library_folder)
    editor._overlay = _TextOverlay(w, h)

    clock   = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                w, h = ev.w, ev.h
                screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                editor.resize(w, h)
            elif ev.type == MOUSEBUTTONDOWN:
                mx, my = ev.pos
                editor.on_mouse_down(ev.button, float(mx), float(h - my),
                                     pygame.key.get_mods())
            elif ev.type == MOUSEBUTTONUP:
                editor.on_mouse_up(ev.button)
            elif ev.type == MOUSEMOTION:
                mx, my = ev.pos
                editor.on_mouse_move(float(mx), float(h - my))
            elif ev.type == KEYDOWN:
                editor.on_key_down(ev.key, ev.mod)
            elif ev.type == pygame.KEYUP:
                editor.on_key_up(ev.key)
            elif ev.type == pygame.TEXTINPUT:
                editor.on_text_input(ev.text)
        editor.draw()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    run()
