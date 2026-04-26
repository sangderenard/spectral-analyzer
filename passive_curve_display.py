"""passive_curve_display.py — Read-only ParametricCurve display + voice pre-mix viewer.

Three stacked panels, driven by a float playhead_t (0..1):

  Top     — GLAnalyticWaveWidget (or 2-D fallback) fed from all voices in the
             sidecar.  Uses premix_gl (stored at ~4 kHz) so phasors rotate
             visibly.  Shows a 1-second window centred on the playhead.

  Middle  — Chirp curve of the atom currently playing at playhead_t.

  Bottom  — Envelope curve of the atom currently playing at playhead_t.

The curves are drawn as anti-aliased polylines — read-only, no editing.
A vertical cursor line at playhead_t is drawn across the curve panels.
A dot on each curve tracks the exact evaluated value at the cursor.

Usage
-----
    from passive_curve_display import PassiveCurveDisplay

    disp = PassiveCurveDisplay(atoms=atoms, sidecar=sidecar, total_s=SONG_DUR_S)
    disp.playhead_t = 0.35
    surf = disp.render(width=900, height=600)
    screen.blit(surf, (0, 0))
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

try:
    from opengl_widget import (
        GLAnalyticWaveWidget as _GLWidget,
        PERSPECTIVE_MODES  as _GL_MODES,
        PERSPECTIVE_LABELS as _GL_LABELS,
    )
    _HAS_GL = True
except Exception:
    _HAS_GL = False
    _GL_MODES  = ["perspective", "ortho", "top", "re_plane", "lissajous"]
    _GL_LABELS = {"perspective": "3D", "ortho": "Orth", "top": "Top",
                  "re_plane": "Re", "lissajous": "Liss"}

# ──────────────────────────────────────────────────────────────────────────────
# Palette  (r, g, b)
# ──────────────────────────────────────────────────────────────────────────────
_C_BG       = (16,  16,  20)
_C_PANEL_BG = (22,  22,  28)
_C_DIVIDER  = (50,  50,  65)
_C_ENVELOPE = (64,  210, 110)
_C_CHIRP    = (90,  165, 255)
_C_PLAYHEAD = (255, 178,  50)
_C_GRID     = (40,  40,  50)
_C_LABEL    = (160, 160, 175)
_C_HEADER   = (30,  30,  38)
_C_DOT_RING = (255, 255, 255)

_HEADER_H   = 20
_CTRL_H     = 26   # GL perspective strip
_MARGIN_L   = 44   # left margin (room for axis labels)
_MARGIN_R   = 12
_MARGIN_TB  = 6

_CURVE_STEPS = 320    # polyline evaluation points

# GL window: how many seconds of premix_gl to show centred on playhead
_GL_WINDOW_S = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Curve helpers
# ──────────────────────────────────────────────────────────────────────────────

def _eval_curve(curve: Any, n: int = _CURVE_STEPS) -> List[tuple]:
    """Evaluate a ParametricCurve at n evenly-spaced t values → [(t, v), ...].

    evaluate_normalized requires a torch.Tensor; returns complex128 Tensor.
    """
    if not _HAS_TORCH:
        return [(i / max(n - 1, 1), 0.0) for i in range(n)]
    try:
        ts = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        zs = curve.evaluate_normalized(ts)          # (n,) complex128
        vs = zs.real.tolist()
        return [(float(ts[i]), float(vs[i])) for i in range(n)]
    except Exception:
        return [(i / max(n - 1, 1), 0.0) for i in range(n)]


def _eval_at(curve: Any, t: float) -> float:
    """Evaluate curve at a single t, return real component."""
    if not _HAS_TORCH:
        return 0.0
    try:
        z = curve.evaluate_normalized(torch.tensor(t, dtype=torch.float64))
        return float(z.real)
    except Exception:
        return 0.0


def _curve_to_screen(pts: List[tuple], x0: int, y0: int, w: int, h: int,
                     v_lo: float, v_hi: float) -> List[tuple]:
    v_range = max(v_hi - v_lo, 1e-9)
    out = []
    for t, v in pts:
        sx = int(x0 + t * w)
        vy = (v - v_lo) / v_range
        sy = int(y0 + h - vy * h)
        out.append((sx, sy))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Atom selection
# ──────────────────────────────────────────────────────────────────────────────

def _has_interesting_envelope(atom: Any) -> bool:
    """True if the atom's envelope curve has more than 2 control points."""
    c = getattr(atom, "envelope_curve", None)
    if c is None:
        return False
    pts = getattr(c, "points", None)
    return pts is not None and len(pts) > 2


def _find_current_atom(atoms: List[Any], playhead_t: float, total_s: float,
                       prefer_voice_keys: Optional[List[str]] = None) -> Optional[Any]:
    """Return the best atom to display at playhead_s.

    Prefers atoms with non-trivial envelope curves and, if given,
    those whose voice_key is in prefer_voice_keys.
    Falls back progressively: interesting+preferred → any active → nearest onset.
    """
    if not atoms:
        return None
    playhead_s = playhead_t * total_s

    def _onset(a):  return float(getattr(a, "onset_time_s", 0.0))
    def _end(a):    return _onset(a) + float(getattr(a, "duration_s", 0.0)) + float(getattr(a, "release_tail_s", 0.0))
    def _preferred(a): return prefer_voice_keys is None or getattr(a, "voice_key", "") in prefer_voice_keys

    active = [a for a in atoms if _onset(a) <= playhead_s <= _end(a)]

    # 1. active + preferred + interesting
    for a in active:
        if _preferred(a) and _has_interesting_envelope(a):
            return a
    # 2. active + interesting
    for a in active:
        if _has_interesting_envelope(a):
            return a
    # 3. any active
    if active:
        return max(active, key=_onset)

    # No atom straddles the cursor: return the one whose onset is the nearest
    # before playhead (so envelope doesn't suddenly jump to a future atom).
    past = [a for a in atoms if _onset(a) <= playhead_s]
    if past:
        best = max(past, key=lambda a: (_preferred(a), _has_interesting_envelope(a), _onset(a)))
        return best
    # Last resort: nearest atom overall
    return min(atoms, key=lambda a: abs(_onset(a) - playhead_s))


# ──────────────────────────────────────────────────────────────────────────────
# Main widget
# ──────────────────────────────────────────────────────────────────────────────

class PassiveCurveDisplay:
    """Passive three-panel display driven by playhead_t.

    Parameters
    ----------
    atoms:
        List of PerformanceAtom objects.
    sidecar:
        Optional dict from VoicePremixCapture.load().  Must contain
        ``premix_gl`` (stored at ~4 kHz) for the GL phasor panel.
    total_s:
        Total arrangement duration in seconds.
    prefer_voice_keys:
        Voice keys to prefer when selecting which atom's envelope/chirp to show.
        Pass e.g. ``["v_mel", "v_bass"]`` so the display shows the melodic
        envelope rather than a default instrument envelope.  None = no preference.
    """

    def __init__(
        self,
        atoms: List[Any],
        *,
        sidecar: Optional[dict] = None,
        total_s: float = 16.0,
        prefer_voice_keys: Optional[List[str]] = None,
    ) -> None:
        self.atoms             = list(atoms)
        self.sidecar           = sidecar
        self.total_s           = max(float(total_s), 1e-3)
        self.prefer_voice_keys = prefer_voice_keys

        self.playhead_t: float = 0.0

        self._gl_widget: Optional[Any] = None
        self._gl_mode:   str = _GL_MODES[0] if _GL_MODES else "perspective"
        self._gl_rect:   Optional[tuple] = None

        self._surf:   Optional[Any] = None
        self._surf_w: int = 0
        self._surf_h: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def set_atoms(self, atoms: List[Any]) -> None:
        self.atoms = list(atoms)

    def set_sidecar(self, sidecar: dict) -> None:
        self.sidecar = sidecar

    def on_mouse_down(self, button: int, x: float, y: float) -> bool:
        if button == 1 and self._gl_rect is not None:
            gx, gy, gw, gh = self._gl_rect
            if gx <= x <= gx + gw and gy + gh - _CTRL_H <= y <= gy + gh:
                self._cycle_gl_mode()
                return True
        return False

    def render(self, width: int, height: int) -> "pygame.Surface":
        if not _HAS_PYGAME:
            raise RuntimeError("pygame not available")
        if self._surf is None or self._surf_w != width or self._surf_h != height:
            self._surf   = pygame.Surface((width, height))
            self._surf_w = width
            self._surf_h = height

        surf = self._surf
        surf.fill(_C_BG)

        panel_h = height // 3
        rects = [
            (0, 0,           width, panel_h),
            (0, panel_h,     width, panel_h),
            (0, panel_h * 2, width, height - panel_h * 2),
        ]

        atom = _find_current_atom(
            self.atoms, self.playhead_t, self.total_s,
            prefer_voice_keys=self.prefer_voice_keys,
        )

        self._draw_gl_panel   (surf, *rects[0], atom)
        self._draw_curve_panel(surf, *rects[1], atom, "chirp",    _C_CHIRP,    "Chirp")
        self._draw_curve_panel(surf, *rects[2], atom, "envelope", _C_ENVELOPE, "Envelope")

        # Playhead across the two curve panels
        cursor_x = int(_MARGIN_L + self.playhead_t * (width - _MARGIN_L - _MARGIN_R))
        for _, py, _, ph in rects[1:]:
            pygame.draw.line(surf, _C_PLAYHEAD,
                             (cursor_x, py + _HEADER_H),
                             (cursor_x, py + ph - 1), 2)
        return surf

    # ── Panel renderers ───────────────────────────────────────────────────────

    def _draw_gl_panel(self, surf, px, py, pw, ph, atom):
        self._draw_panel_bg(surf, px, py, pw, ph, "Pre-mix  (all voices)")

        cx  = px + _MARGIN_L
        cy  = py + _HEADER_H + _MARGIN_TB
        cw  = pw - _MARGIN_L - _MARGIN_R
        ch  = max(4, ph - _HEADER_H - _CTRL_H - _MARGIN_TB * 2)

        if _HAS_GL and _HAS_NP and self.sidecar is not None:
            batch = self._gl_batch_at_playhead()
            if batch is not None and batch.size > 0:
                if self._gl_widget is None:
                    self._gl_widget = _GLWidget(phase_steps=72)
                self._gl_widget.set_perspective_mode(self._gl_mode)
                existing = getattr(self._gl_widget, "_csig_batch", None)
                if batch is not existing:
                    self._gl_widget.update_data(batch)
                self._gl_rect = (cx, cy, cw, ch)
                # GL renders directly into the OpenGL framebuffer at this rect.
                self._gl_widget._gl_analytic_rect = (cx, cy, cw, ch)
                self._draw_gl_ctrl(surf, px, py + ph - _CTRL_H, pw, _CTRL_H)
                return

        self._gl_rect = None
        self._draw_rms_fallback(surf, cx, cy, cw, ch)
        self._draw_gl_ctrl(surf, px, py + ph - _CTRL_H, pw, _CTRL_H)

    def _draw_rms_fallback(self, surf, x0, y0, w, h):
        if not _HAS_NP or self.sidecar is None:
            return
        premix = self.sidecar.get("premix")
        if premix is None or premix.ndim < 2 or premix.shape[0] == 0:
            return

        gl_fps = float(self.sidecar.get("gl_fps", self.sidecar.get("display_fps", 120.0)))
        n_frames = premix.shape[1]
        t_c  = self.playhead_t * self.total_s
        f_lo = max(0, int((t_c - _GL_WINDOW_S * 0.5) * gl_fps))
        f_hi = min(n_frames, int((t_c + _GL_WINDOW_S * 0.5) * gl_fps) + 1)
        if f_hi <= f_lo:
            return

        colors = [_C_ENVELOPE, _C_CHIRP, (200, 100, 220), (220, 180, 80),
                  (255, 80, 120), (80, 220, 200), (180, 255, 100)]
        n_voices = min(premix.shape[0], 7)
        win  = premix[:n_voices, f_lo:f_hi]
        peak = float(win.max()) if win.size else 1.0
        if peak < 1e-9:
            peak = 1.0
        cy = y0 + h // 2
        n_win = f_hi - f_lo
        for vi in range(n_voices):
            col = colors[vi % len(colors)]
            pts = [
                (int(x0 + fi * w / max(n_win - 1, 1)),
                 int(cy - float(premix[vi, f_lo + fi]) / peak * h * 0.42))
                for fi in range(n_win)
            ]
            if len(pts) >= 2:
                pygame.draw.lines(surf, col, False, pts, 1)

    def _draw_gl_ctrl(self, surf, px, py, pw, ph):
        pygame.draw.rect(surf, _C_HEADER, (px, py, pw, ph))
        f = self._small_font()
        if f:
            lbl = _GL_LABELS.get(self._gl_mode, self._gl_mode)
            txt = f.render(f"[GL: {lbl}]  click to cycle perspective", True, _C_LABEL)
            surf.blit(txt, (px + 6, py + (ph - txt.get_height()) // 2))

    def _draw_curve_panel(self, surf, px, py, pw, ph, atom, curve_attr, color, label):
        self._draw_panel_bg(surf, px, py, pw, ph, label)

        x0 = px + _MARGIN_L
        y0 = py + _HEADER_H + _MARGIN_TB
        w  = pw - _MARGIN_L - _MARGIN_R
        h  = ph - _HEADER_H - _MARGIN_TB * 2

        self._draw_grid(surf, x0, y0, w, h)

        # Axis labels
        f = self._small_font()
        if atom is None:
            if f:
                msg = f.render("no atom at playhead", True, _C_LABEL)
                surf.blit(msg, (x0 + 4, y0 + h // 2 - msg.get_height() // 2))
            return

        curve = getattr(atom, f"{curve_attr}_curve", None)
        if curve is None:
            return

        v_lo, v_hi = _curve_range(curve_attr, curve)

        # Draw the curve polyline + filled area
        pts = _eval_curve(curve)
        if pts:
            spx = _curve_to_screen(pts, x0, y0, w, h, v_lo, v_hi)
            if len(spx) >= 2:
                pygame.draw.lines(surf, color, False, spx, 2)
                self._fill_curve(surf, spx, x0, y0, w, h, color)

        # Draw axis value labels
        if f:
            hi_t = f.render(f"{v_hi:.3g}", True, _C_LABEL)
            lo_t = f.render(f"{v_lo:.3g}", True, _C_LABEL)
            surf.blit(hi_t, (px + 2, y0))
            surf.blit(lo_t, (px + 2, y0 + h - lo_t.get_height()))

        # Playhead dot on the curve
        t_in = self._playhead_t_in_atom(atom)
        v    = _eval_at(curve, max(0.0, min(1.0, t_in)))
        v_range = max(v_hi - v_lo, 1e-9)
        dot_x = int(x0 + t_in * w)
        dot_y = int(y0 + h - (v - v_lo) / v_range * h)
        dot_x = max(x0, min(x0 + w, dot_x))
        dot_y = max(y0, min(y0 + h, dot_y))
        pygame.draw.circle(surf, _C_PLAYHEAD, (dot_x, dot_y), 6)
        pygame.draw.circle(surf, _C_DOT_RING,  (dot_x, dot_y), 4)

        # Voice label
        vk = getattr(atom, "voice_key", "")
        if f and vk:
            vt = f.render(vk, True, _C_LABEL)
            surf.blit(vt, (x0 + w - vt.get_width() - 2, y0 + 2))

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_panel_bg(self, surf, px, py, pw, ph, label):
        pygame.draw.rect(surf, _C_PANEL_BG, (px, py, pw, ph))
        pygame.draw.line(surf, _C_DIVIDER, (px, py), (px + pw, py), 1)
        f = self._small_font()
        if f:
            t = f.render(label, True, _C_LABEL)
            surf.blit(t, (px + 4, py + (_HEADER_H - t.get_height()) // 2))

    def _draw_grid(self, surf, x0, y0, w, h):
        for i in range(1, 4):
            gx = int(x0 + i * w / 4)
            pygame.draw.line(surf, _C_GRID, (gx, y0), (gx, y0 + h), 1)
        for j in range(1, 4):
            gy = int(y0 + j * h / 4)
            pygame.draw.line(surf, _C_GRID, (x0, gy), (x0 + w, gy), 1)

    def _fill_curve(self, surf, screen_pts, x0, y0, w, h, color):
        if len(screen_pts) < 2:
            return
        floor_y = y0 + h
        poly = list(screen_pts) + [(screen_pts[-1][0], floor_y), (screen_pts[0][0], floor_y)]
        layer = pygame.Surface((w + 2, h + 2), pygame.SRCALPHA)
        fill_col = (color[0], color[1], color[2], 45)
        shifted = [(sx - x0, sy - y0) for sx, sy in poly]
        if len(shifted) >= 3:
            pygame.draw.polygon(layer, fill_col, shifted)
        surf.blit(layer, (x0, y0))

    # ── GL data ───────────────────────────────────────────────────────────────

    def _gl_batch_at_playhead(self) -> Optional[Any]:
        """Return (n_voices, n_gl_frames) complex64 window for the GL widget."""
        if not _HAS_NP or self.sidecar is None:
            return None

        gl = self.sidecar.get("premix_gl")
        if gl is None:
            # Graceful fallback to display-rate complex if gl array absent (old sidecar)
            gl = self.sidecar.get("premix_complex")
        if gl is None or gl.ndim < 2 or gl.shape[0] == 0:
            return None

        gl_fps  = float(self.sidecar.get("gl_fps", 4000.0))
        n_total = gl.shape[1]
        t_c     = self.playhead_t * self.total_s
        t_lo    = max(0.0, t_c - _GL_WINDOW_S * 0.5)
        t_hi    = min(self.total_s, t_c + _GL_WINDOW_S * 0.5)
        f_lo    = int(t_lo * gl_fps)
        f_hi    = min(n_total, int(t_hi * gl_fps) + 1)
        if f_hi <= f_lo:
            return None

        return gl[:, f_lo:f_hi].astype(np.complex128)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _playhead_t_in_atom(self, atom: Any) -> float:
        onset = float(getattr(atom, "onset_time_s", 0.0))
        dur   = float(getattr(atom, "duration_s", 0.0)) + float(getattr(atom, "release_tail_s", 0.0))
        elapsed = self.playhead_t * self.total_s - onset
        return max(0.0, min(1.0, elapsed / max(dur, 1e-9)))

    def _cycle_gl_mode(self):
        try:
            idx = _GL_MODES.index(self._gl_mode)
        except ValueError:
            idx = 0
        self._gl_mode = _GL_MODES[(idx + 1) % len(_GL_MODES)]

    def _small_font(self):
        try:
            return pygame.font.SysFont("monospace", 11)
        except Exception:
            return None


def _curve_range(curve_attr: str, curve: Any) -> tuple:
    v_lo = float(getattr(curve, "v_lo", 0.0))
    v_hi = float(getattr(curve, "v_hi", 1.0))
    if curve_attr == "chirp":
        v_lo = min(v_lo, -200.0)
        v_hi = max(v_hi,  200.0)
    if abs(v_hi - v_lo) < 1e-9:
        v_hi = v_lo + 1.0
    return v_lo, v_hi
