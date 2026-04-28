"""passive_curve_display.py — Configurable multi-panel display for the instrument demo.

Panel system
------------
Each of the three stacked panels has an independent *mode* and *type*:

  mode  — "active" | "passive"
      Passive: read-only visualisation, no mouse editing.
      Active:  user can edit (envelope/chirp editors respond to drag events).

  type  — "opengl" | "chirp" | "envelope" | "waveform" | "accumulator"
      opengl      — ComplexPhaseCloud GL phasor display (or RMS fallback)
      chirp       — Read-only chirp ParametricCurve polyline
      envelope    — Read-only envelope ParametricCurve polyline
      waveform    — Per-string real waveform (rainbow, one trace per driver)
      accumulator — Volumetric density heat-map (body resonator baked state)

Routing
-------
Routes are simple (source_key, dest_panel) tuples that map sidecar data keys
to panel slots.  The display walks ``routes`` to discover which sidecar array
feeds each of the three panels ("top", "mid", "bot").

Default panel config (backwards-compatible with old usage):

    top  — opengl,      routed from "premix_gl"
    mid  — chirp,       routed from atoms
    bot  — envelope,    routed from atoms

Instrument-demo config (used by demo_instrument_song.py):

    top  — opengl,      routed from "pre_body_gl"
    mid  — waveform,    routed from "string_gl"
    bot  — accumulator, routed from "resonator_volume"

Usage
-----
    from passive_curve_display import PassiveCurveDisplay, INSTRUMENT_PANELS, INSTRUMENT_ROUTES

    disp = PassiveCurveDisplay(
        atoms=atoms,
        sidecar=sidecar,
        total_s=SONG_DUR_S,
        panel_configs=INSTRUMENT_PANELS,
        routes=INSTRUMENT_ROUTES,
    )
    disp.playhead_t = 0.35
    surf = disp.render(width=900, height=600)
    screen.blit(surf, (0, 0))
"""
from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Optional, Tuple

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
        GLAnalyticWaveWidget        as _GLWidget,
        VolumetricAccumulatorWidget,
        PERSPECTIVE_MODES           as _GL_MODES,
        PERSPECTIVE_LABELS          as _GL_LABELS,
    )
    _HAS_GL = True
except Exception:
    _HAS_GL = False
    VolumetricAccumulatorWidget = None  # type: ignore[assignment,misc]
    _GL_MODES  = ["perspective", "ortho", "top", "re_plane", "lissajous"]
    _GL_LABELS = {"perspective": "3D", "ortho": "Orth", "top": "Top",
                  "re_plane": "Re", "lissajous": "Liss"}

# ──────────────────────────────────────────────────────────────────────────────
# Panel config & routing types
# ──────────────────────────────────────────────────────────────────────────────

# A panel config is a plain dict with keys:
#   "mode"  : "active" | "passive"
#   "type"  : "opengl" | "chirp" | "envelope" | "waveform" | "accumulator"
# A route is a (source_key, dest_panel) tuple.

PanelConfig = Dict[str, str]
Route       = Tuple[str, str]   # (sidecar_key, panel_id)

# ---- pre-defined panel configurations ----------------------------------------

DEFAULT_PANELS: Dict[str, PanelConfig] = {
    "top": {"mode": "passive", "type": "opengl"},
    "mid": {"mode": "passive", "type": "chirp"},
    "bot": {"mode": "passive", "type": "envelope"},
}

DEFAULT_ROUTES: List[Route] = [
    ("premix_gl", "top"),
]

# Used by demo_instrument_song.py
INSTRUMENT_PANELS: Dict[str, PanelConfig] = {
    "top": {"mode": "passive", "type": "opengl"},
    "mid": {"mode": "passive", "type": "waveform"},
    "bot": {"mode": "passive", "type": "accumulator"},
}

INSTRUMENT_ROUTES: List[Route] = [
    ("pre_body_gl",      "top"),
    ("string_gl",        "mid"),
    ("resonator_volume", "bot"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Palette
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

# Per-driver rainbow for the waveform panel (3 drivers)
_DRIVER_COLORS: List[Tuple[int, int, int]] = [
    (255,  80,  80),   # driver 0 — red
    ( 60, 220, 160),   # driver 1 — cyan-green
    (140,  90, 255),   # driver 2 — violet
]
_SYM_COLOR: Tuple[int, int, int] = (255, 220, 80)   # sympathetic injections

_HEADER_H   = 20
_CTRL_H     = 26   # GL perspective strip
_MARGIN_L   = 44
_MARGIN_R   = 12
_MARGIN_TB  = 6

_CURVE_STEPS = 320
_GL_WINDOW_S = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Cavity pressure-field volume (exported for demo_instrument_song.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_cavity_pressure_volume(
    body_scene: Any,
    driver_signal: "np.ndarray",
    sr: float,
    n_time: int = 64,
    n_spatial: int = 32,
) -> "np.ndarray":
    """Build a (n_time, n_spatial, n_spatial) float32 acoustic pressure field.

    Uses the actual CavityScene geometry (panel positions/normals) and the
    image-source method to compute the instantaneous pressure at every point
    on a 2-D XY cross-section grid inside the body cavity.

    Parameters
    ----------
    body_scene:
        CavityScene from ``instrument._body_scene``.  If None, returns zeros.
    driver_signal:
        1-D real float32 array at ``sr`` Hz — the summed signal entering the body.
    sr:
        Sample rate in Hz.
    n_time:
        Number of equally-spaced time frames across the full signal.
    n_spatial:
        Grid resolution: grid is n_spatial × n_spatial.

    Returns
    -------
    volume : np.ndarray, shape (n_time, n_spatial, n_spatial), float32
        Normalised instantaneous pressure magnitude (0..1).
        The X/Y axes map to the cavity's horizontal cross-section; panels that
        define the cylindrical body wall show as a dark ring (zero outside).

    Volume layout matches GL_TEXTURE_3D sampling as
    ``texture(uVolume, vec3(spatial_x, spatial_y, time_norm))``.
    """
    if not _HAS_NP or body_scene is None:
        if _HAS_NP:
            return np.zeros((n_time, n_spatial, n_spatial), dtype=np.float32)
        return None  # type: ignore[return-value]

    c = float(getattr(body_scene, "speed_of_sound_m_s", 343.0))

    sources = getattr(body_scene, "sources", [])
    geometry = getattr(body_scene, "geometry", None)

    # geometry is a room object (PolygonalRoom / CircularRoom / MeshRoom),
    # not a list — convert it to CavityPanel list via build_room_panels.
    panels: list = []
    if geometry is not None:
        try:
            from cavity_engine import build_room_panels as _brp
            panels = _brp(geometry)
        except Exception:
            pass

    if not sources:
        return np.zeros((n_time, n_spatial, n_spatial), dtype=np.float32)

    src_pos = np.array(sources[0].position, dtype=np.float64)  # (3,)

    # ── Cavity bounds from panel XY positions ────────────────────────────────
    if panels:
        pts = np.array([p.point for p in panels], dtype=np.float64)   # (P, 3)
        nrm = np.array([p.normal for p in panels], dtype=np.float64)  # (P, 3)
        ref = np.array(
            [float(getattr(p, "reflectivity", 0.6)) for p in panels],
            dtype=np.float64,
        )  # (P,)
    else:
        pts = np.zeros((1, 3), dtype=np.float64)
        nrm = np.array([[0.0, 0.0, 1.0]])
        ref = np.array([0.6])

    xy_max = float(np.max(np.abs(pts[:, :2]))) * 1.05
    if xy_max < 1e-6:
        xy_max = 0.20  # 20 cm default

    # Z cross-section: midpoint of panel Z extent (cavity interior)
    z_mid = float(pts[:, 2].mean())

    # ── 2-D spatial grid ─────────────────────────────────────────────────────
    lin = np.linspace(-xy_max, xy_max, n_spatial, dtype=np.float64)
    gx, gy = np.meshgrid(lin, lin, indexing="xy")   # (n_spatial, n_spatial)
    gz = np.full_like(gx, z_mid)
    M = n_spatial * n_spatial
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (M, 3)

    # Circular cavity mask (inward-facing normals → inside = below surface)
    r_body  = xy_max / 1.05
    inside  = (grid_pts[:, 0] ** 2 + grid_pts[:, 1] ** 2) <= r_body ** 2  # (M,)

    # ── Image-source positions for each panel ────────────────────────────────
    # image = src - 2 * ((src - panel_point) · normal) * normal
    sp          = src_pos[np.newaxis, :] - pts          # (P, 3)
    plane_dist  = (sp * nrm).sum(axis=1, keepdims=True) # (P, 1)
    image_pos   = src_pos[np.newaxis, :] - 2.0 * plane_dist * nrm  # (P, 3)
    P = len(panels)

    # ── Propagation delays (in samples) ──────────────────────────────────────
    # Direct path: (M,)
    d_direct     = np.linalg.norm(grid_pts - src_pos[np.newaxis, :], axis=1)
    delay_direct = np.round(d_direct / c * sr).astype(np.int64)
    amp_direct   = np.where(d_direct > 1e-6, 1.0 / np.maximum(d_direct, 1e-6), 0.0)

    # Per-panel: (M, P)
    diff_panel   = grid_pts[:, np.newaxis, :] - image_pos[np.newaxis, :, :]  # (M,P,3)
    d_panel      = np.linalg.norm(diff_panel, axis=2)                         # (M,P)
    delay_panel  = np.round(d_panel / c * sr).astype(np.int64)                # (M,P)
    amp_panel    = ref[np.newaxis, :] / np.maximum(d_panel, 1e-6)             # (M,P)

    # ── Driver signal ─────────────────────────────────────────────────────────
    driver = np.asarray(driver_signal, dtype=np.float32).ravel()
    if np.iscomplexobj(driver):
        driver = driver.real
    N = len(driver)
    if N == 0:
        return np.zeros((n_time, n_spatial, n_spatial), dtype=np.float32)

    # ── Time frames ───────────────────────────────────────────────────────────
    frame_idx = np.linspace(0, N - 1, n_time).astype(np.int64)  # (n_time,)

    volume = np.zeros((n_time, M), dtype=np.float32)

    for ti, t in enumerate(frame_idx):
        # Direct contribution
        si  = np.clip(t - delay_direct, 0, N - 1)   # (M,)
        p   = amp_direct * driver[si]               # (M,)

        # Per-panel image-source contributions
        for pi in range(P):
            si_p  = np.clip(t - delay_panel[:, pi], 0, N - 1)  # (M,)
            p    += amp_panel[:, pi] * driver[si_p]

        p[~inside] = 0.0
        volume[ti]  = p.astype(np.float32)

    volume = np.abs(volume)
    mx = volume.max()
    if mx > 1e-12:
        volume /= mx

    return volume.reshape(n_time, n_spatial, n_spatial)


# ──────────────────────────────────────────────────────────────────────────────
# Curve helpers
# ──────────────────────────────────────────────────────────────────────────────

def _eval_curve(curve: Any, n: int = _CURVE_STEPS) -> List[tuple]:
    if not _HAS_TORCH:
        return [(i / max(n - 1, 1), 0.0) for i in range(n)]
    try:
        ts = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        zs = curve.evaluate_normalized(ts)
        vs = zs.real.tolist()
        return [(float(ts[i]), float(vs[i])) for i in range(n)]
    except Exception:
        return [(i / max(n - 1, 1), 0.0) for i in range(n)]


def _eval_at(curve: Any, t: float) -> float:
    if not _HAS_TORCH:
        return 0.0
    try:
        z = curve.evaluate_normalized(torch.tensor(t, dtype=torch.float64))
        return float(z.real)
    except Exception:
        return 0.0


def _curve_to_screen(pts, x0, y0, w, h, v_lo, v_hi):
    v_range = max(v_hi - v_lo, 1e-9)
    out = []
    for t, v in pts:
        sx = int(x0 + t * w)
        vy = (v - v_lo) / v_range
        sy = int(y0 + h - vy * h)
        out.append((sx, sy))
    return out


def _has_interesting_envelope(atom: Any) -> bool:
    c = getattr(atom, "envelope_curve", None)
    if c is None:
        return False
    pts = getattr(c, "points", None)
    return pts is not None and len(pts) > 2


def _find_current_atom(atoms, playhead_t, total_s, prefer_voice_keys=None):
    if not atoms:
        return None
    playhead_s = playhead_t * total_s

    def _onset(a):     return float(getattr(a, "onset_time_s", 0.0))
    def _end(a):       return (_onset(a) + float(getattr(a, "duration_s", 0.0))
                               + float(getattr(a, "release_tail_s", 0.0)))
    def _preferred(a): return (prefer_voice_keys is None
                               or getattr(a, "voice_key", "") in prefer_voice_keys)

    active = [a for a in atoms if _onset(a) <= playhead_s <= _end(a)]
    for a in active:
        if _preferred(a) and _has_interesting_envelope(a):
            return a
    for a in active:
        if _has_interesting_envelope(a):
            return a
    if active:
        return max(active, key=_onset)
    past = [a for a in atoms if _onset(a) <= playhead_s]
    if past:
        return max(past, key=lambda a: (_preferred(a),
                                        _has_interesting_envelope(a), _onset(a)))
    return min(atoms, key=lambda a: abs(_onset(a) - playhead_s))


def _curve_range(curve_attr, curve):
    v_lo = float(getattr(curve, "v_lo", 0.0))
    v_hi = float(getattr(curve, "v_hi", 1.0))
    if curve_attr == "chirp":
        v_lo = min(v_lo, -200.0)
        v_hi = max(v_hi,  200.0)
    if abs(v_hi - v_lo) < 1e-9:
        v_hi = v_lo + 1.0
    return v_lo, v_hi


# ──────────────────────────────────────────────────────────────────────────────
# Main widget
# ──────────────────────────────────────────────────────────────────────────────

class PassiveCurveDisplay:
    """Configurable three-panel display driven by a float playhead_t (0..1).

    Parameters
    ----------
    atoms:
        List of PerformanceAtom objects (used by chirp/envelope panels).
    sidecar:
        Optional dict from VoicePremixCapture.load() or the extended sidecar
        produced by demo_instrument_song._capture_premix().  May contain:
          premix_gl        — (n_voices, n_gl_frames) complex, GL phasor cloud
          pre_body_gl      — (1, n_gl_frames) complex, pre-body-resonance mix
          string_gl        — (n_drivers, n_gl_frames) complex, per-string signals
          resonator_volume — (n_time, n_sources, n_freq) float32, 3-D spectral volume
    total_s:
        Total arrangement duration in seconds.
    prefer_voice_keys:
        Voice keys preferred for atom selection (envelope/chirp panels).
    panel_configs:
        Dict mapping "top"/"mid"/"bot" to PanelConfig dicts.
        Use INSTRUMENT_PANELS for the instrument demo, or DEFAULT_PANELS for
        the original chirp/envelope layout.
    routes:
        List of (sidecar_key, panel_id) tuples routing data to panels.
        Use INSTRUMENT_ROUTES for the instrument demo.
    """

    def __init__(
        self,
        atoms: List[Any],
        *,
        sidecar: Optional[dict] = None,
        total_s: float = 16.0,
        prefer_voice_keys: Optional[List[str]] = None,
        panel_configs: Optional[Dict[str, PanelConfig]] = None,
        routes: Optional[List[Route]] = None,
    ) -> None:
        self.atoms             = list(atoms)
        self.sidecar           = sidecar
        self.total_s           = max(float(total_s), 1e-3)
        self.prefer_voice_keys = prefer_voice_keys

        self._panel_configs: Dict[str, PanelConfig] = (
            panel_configs if panel_configs is not None else dict(DEFAULT_PANELS)
        )
        self._routes: List[Route] = list(routes) if routes is not None else list(DEFAULT_ROUTES)

        self.playhead_t: float = 0.0

        self._gl_widget:    Optional[Any] = None
        self._gl_mode:      str           = _GL_MODES[0] if _GL_MODES else "perspective"
        self._gl_rect:      Optional[tuple] = None
        self._accum_widget: Optional[Any] = None   # VolumetricAccumulatorWidget

        self._surf:   Optional[Any] = None
        self._surf_w: int = 0
        self._surf_h: int = 0

        # Per-panel progress state — written from render thread, read on main thread
        self._prog_lock: threading.Lock = threading.Lock()
        self._panel_progress: Dict[str, Tuple[str, float]] = {}
        # { panel_id: (label, fraction 0..1) }

    # ── Public API ────────────────────────────────────────────────────────────

    def set_atoms(self, atoms: List[Any]) -> None:
        self.atoms = list(atoms)

    def set_sidecar(self, sidecar: dict) -> None:
        self.sidecar = sidecar

    def set_panel_configs(self, configs: Dict[str, PanelConfig]) -> None:
        self._panel_configs = dict(configs)

    def set_routes(self, routes: List[Route]) -> None:
        self._routes = list(routes)

    def set_panel_progress(self, panel_id: str, label: str,
                           fraction: float) -> None:
        """Set a loading-progress state for a panel.

        Thread-safe — may be called from a background render thread.

        Parameters
        ----------
        panel_id:
            One of ``"top"``, ``"mid"``, ``"bot"``.
        label:
            Short description of the current phase (e.g. ``"building waveforms…"``).
        fraction:
            0.0 → 1.0.  Pass a value < 0 to clear/hide the progress bar.
        """
        with self._prog_lock:
            if fraction < 0.0:
                self._panel_progress.pop(panel_id, None)
            else:
                self._panel_progress[panel_id] = (label, max(0.0, min(1.0, fraction)))

    def clear_panel_progress(self, panel_id: str) -> None:
        """Remove the progress bar for *panel_id*.  Thread-safe."""
        with self._prog_lock:
            self._panel_progress.pop(panel_id, None)

    def draw_gl(self, offset_x: int = 0, offset_y: int = 0,
                win_w: int = 0, win_h: int = 0) -> None:
        """Draw pending GL widgets into the current OpenGL framebuffer.

        Must be called *after* :meth:`render` (which configures the widget
        state and records the rects) and *after* the 2-D surface has been
        uploaded to GL (e.g. via ``SurfaceBlitter.blit``).

        Parameters
        ----------
        offset_x, offset_y:
            Position of the surface returned by :meth:`render` within the
            pygame window (i.e. ``CTR_X, 0`` for the centre panel).
        win_w, win_h:
            Full pygame window dimensions, needed by ``GLViewportWidget.draw``.
        """
        if not _HAS_GL:
            return
        if self._gl_widget is not None and self._gl_rect is not None:
            cx, cy, cw, ch = self._gl_rect
            self._gl_widget.draw(
                offset_x + cx, offset_y + cy, cw, ch,
                win_w, win_h,
                current_slot=getattr(self._gl_widget, "_current_slot", 0),
            )
        if self._accum_widget is not None:
            rect = getattr(self._accum_widget, "_gl_accumulator_rect", None)
            if rect is not None:
                cx, cy, cw, ch = rect
                self._accum_widget.draw(
                    offset_x + cx, offset_y + cy, cw, ch,
                    win_w, win_h,
                )

    def on_mouse_down(self, button: int, x: float, y: float) -> bool:
        """Cycle GL perspective when clicking the control strip in the top panel."""
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

        for panel_id, rect in zip(("top", "mid", "bot"), rects):
            cfg      = self._panel_configs.get(panel_id, {})
            ptype    = cfg.get("type", "opengl" if panel_id == "top" else
                               ("chirp" if panel_id == "mid" else "envelope"))
            data_key = self._resolve_data_key(panel_id)

            if ptype == "opengl":
                self._draw_gl_panel(surf, *rect, atom, data_key=data_key)
            elif ptype == "chirp":
                self._draw_curve_panel(surf, *rect, atom, "chirp", _C_CHIRP, "Chirp")
            elif ptype == "envelope":
                self._draw_curve_panel(surf, *rect, atom,
                                       "envelope", _C_ENVELOPE, "Envelope")
            elif ptype == "waveform":
                self._draw_string_waveform_panel(surf, *rect, data_key=data_key)
            elif ptype == "accumulator":
                self._draw_accumulator_panel(surf, *rect, data_key=data_key)
            else:
                self._draw_curve_panel(surf, *rect, atom,
                                       "envelope", _C_ENVELOPE, ptype)

            # Progress overlay — drawn on top of panel content if active
            with self._prog_lock:
                prog = self._panel_progress.get(panel_id)
            if prog is not None:
                px, py, pw, ph = rect
                self._draw_progress_overlay(surf, px, py, pw, ph, *prog)

        # Playhead cursor for chirp/envelope panels only
        cursor_x = int(_MARGIN_L + self.playhead_t * (width - _MARGIN_L - _MARGIN_R))
        for panel_id, (_, py, _, ph) in zip(("top", "mid", "bot"), rects):
            ptype = self._panel_configs.get(panel_id, {}).get("type", "")
            if ptype in ("chirp", "envelope"):
                pygame.draw.line(surf, _C_PLAYHEAD,
                                 (cursor_x, py + _HEADER_H),
                                 (cursor_x, py + ph - 1), 2)
        return surf

    # ── Routing helper ────────────────────────────────────────────────────────

    def _resolve_data_key(self, panel_id: str) -> Optional[str]:
        """Return the first sidecar key routed to this panel."""
        for src, dst in self._routes:
            if dst == panel_id:
                return src
        return None

    # ── Panel renderers ───────────────────────────────────────────────────────

    def _draw_gl_panel(self, surf, px, py, pw, ph, atom, data_key=None):
        self._draw_panel_bg(surf, px, py, pw, ph, "Pre-mix  (all voices)")

        cx = px + _MARGIN_L
        cy = py + _HEADER_H + _MARGIN_TB
        cw = pw - _MARGIN_L - _MARGIN_R
        ch = max(4, ph - _HEADER_H - _CTRL_H - _MARGIN_TB * 2)

        if _HAS_GL and _HAS_NP and self.sidecar is not None:
            batch = self._gl_batch_at_playhead(data_key=data_key)
            if batch is not None and batch.size > 0:
                if self._gl_widget is None:
                    self._gl_widget = _GLWidget(phase_steps=72)
                self._gl_widget.set_perspective_mode(self._gl_mode)
                existing = getattr(self._gl_widget, "_csig_batch", None)
                if batch is not existing:
                    self._gl_widget.update_data(batch)
                self._gl_rect = (cx, cy, cw, ch)
                self._gl_widget._gl_analytic_rect = (cx, cy, cw, ch)
                self._draw_gl_ctrl(surf, px, py + ph - _CTRL_H, pw, _CTRL_H)
                return

        self._gl_rect = None
        self._draw_rms_fallback(surf, cx, cy, cw, ch, data_key=data_key)
        self._draw_gl_ctrl(surf, px, py + ph - _CTRL_H, pw, _CTRL_H)

    def _draw_string_waveform_panel(self, surf, px, py, pw, ph, data_key=None):
        """Middle panel: per-string waveforms at physical body positions.

        Each string occupies a horizontal lane whose vertical centre is mapped
        from the string's physical ``y`` coordinate (from ``string_positions``
        in the sidecar).  Within each lane, the string's own waveform is drawn
        in its driver colour, and every coupling contribution from another
        string is drawn in *that source string's* colour with alpha proportional
        to the coupling magnitude.  All layers alpha-blend on the panel.
        """
        self._draw_panel_bg(surf, px, py, pw, ph,
                            "Strings  (physical layout · coupling by source colour · alpha blend)")

        cx = px + _MARGIN_L
        cy = py + _HEADER_H + _MARGIN_TB
        cw = pw - _MARGIN_L - _MARGIN_R
        ch = max(4, ph - _HEADER_H - _MARGIN_TB * 2)

        if not _HAS_NP or self.sidecar is None:
            return

        key = data_key or "string_gl"
        sg  = self.sidecar.get(key)
        if sg is None or sg.ndim < 2 or sg.shape[0] == 0:
            f = self._small_font()
            if f:
                msg = f.render("no string data — render audio first", True, _C_LABEL)
                surf.blit(msg, (cx + 4, cy + ch // 2 - msg.get_height() // 2))
            return

        gl_fps   = float(self.sidecar.get("gl_fps", 4000.0))
        n_total  = sg.shape[1]
        n_str    = sg.shape[0]
        t_c      = self.playhead_t * self.total_s
        f_lo     = max(0, int((t_c - _GL_WINDOW_S * 0.5) * gl_fps))
        f_hi     = min(n_total, int((t_c + _GL_WINDOW_S * 0.5) * gl_fps) + 1)
        if f_hi <= f_lo:
            return

        window   = sg[:, f_lo:f_hi]
        real_win = window.real if np.iscomplexobj(window) else window.astype(np.float32)
        n_win    = real_win.shape[1]

        # Shared amplitude normalisation across all strings
        peak = float(np.abs(real_win).max())
        if peak < 1e-9:
            peak = 1.0

        # ── Lane centres from physical y positions ────────────────────────────
        positions  = self.sidecar.get("string_positions")   # (n_str, 2) or None
        coupling   = self.sidecar.get("coupling_matrix")    # (n_str, n_str) or None

        if positions is not None and len(positions) == n_str:
            phys_y = np.asarray(positions, dtype=np.float32)[:, 1]  # (n_str,)
            y_min, y_max = float(phys_y.min()), float(phys_y.max())
            span = y_max - y_min
            if span < 1e-6:
                # All strings at same y — lay out evenly
                y_norm = np.linspace(0.12, 0.88, n_str, dtype=np.float32)
            else:
                # Map physical y → [0.12, 0.88], top of panel = highest y
                y_norm = 0.88 - (phys_y - y_min) / span * 0.76
        else:
            y_norm = np.linspace(0.12, 0.88, n_str, dtype=np.float32)

        # Pixel y of each lane centre
        centers = [int(cy + float(yn) * ch) for yn in y_norm]

        # Waveform excursion: half of the smallest gap between adjacent lanes,
        # capped at 35 % of total panel height.
        if n_str > 1:
            sorted_c = sorted(centers)
            min_gap  = min(sorted_c[i+1] - sorted_c[i]
                           for i in range(len(sorted_c) - 1))
            excursion = max(4, min(int(min_gap * 0.42), int(ch * 0.35)))
        else:
            excursion = int(ch * 0.35)

        # ── Draw grid lines at each lane centre ───────────────────────────────
        for c_y in centers:
            pygame.draw.line(surf, _C_GRID, (cx, c_y), (cx + cw, c_y), 1)

        # ── Draw waveforms ────────────────────────────────────────────────────
        # Pass 1: coupling contributions (drawn beneath the own-signal line)
        if coupling is not None and coupling.shape == (n_str, n_str):
            for di in range(n_str):
                c_y = centers[di]
                for src in range(n_str):
                    if src == di:
                        continue
                    w = float(coupling[di, src])
                    if w < 0.02:
                        continue
                    # Scale: coupling weight × peak-normalised source signal
                    contrib = real_win[src] * (w / peak)
                    alpha   = max(30, min(180, int(w * 220)))
                    r, g, b = _DRIVER_COLORS[src % len(_DRIVER_COLORS)]
                    layer   = pygame.Surface((cw, ch), pygame.SRCALPHA)
                    pts = [
                        (int(fi * cw / max(n_win - 1, 1)),
                         int(c_y - cy + float(contrib[fi]) * excursion))
                        for fi in range(n_win)
                    ]
                    if len(pts) >= 2:
                        pygame.draw.lines(layer, (r, g, b, alpha), False, pts, 1)
                    surf.blit(layer, (cx, cy))

        # Pass 2: own-signal lines (drawn on top, full alpha)
        for di in range(n_str):
            c_y   = centers[di]
            r, g, b = _DRIVER_COLORS[di % len(_DRIVER_COLORS)]
            layer = pygame.Surface((cw, ch), pygame.SRCALPHA)
            pts = [
                (int(fi * cw / max(n_win - 1, 1)),
                 int(c_y - cy - float(real_win[di, fi]) / peak * excursion))
                for fi in range(n_win)
            ]
            if len(pts) >= 2:
                pygame.draw.lines(layer, (r, g, b, 210), False, pts, 2)
            surf.blit(layer, (cx, cy))

        # ── Left-margin labels at lane centres ────────────────────────────────
        f = self._small_font()
        if f:
            for di in range(n_str):
                col  = _DRIVER_COLORS[di % len(_DRIVER_COLORS)]
                lbl  = f.render(f"s{di+1}", True, col)
                surf.blit(lbl, (px + 2, centers[di] - lbl.get_height() // 2))

        # ── Playhead cursor ───────────────────────────────────────────────────
        phx = int(cx + self.playhead_t * cw)
        pygame.draw.line(surf, _C_PLAYHEAD, (phx, cy), (phx, cy + ch), 2)

    def _draw_accumulator_panel(self, surf, px, py, pw, ph, data_key=None):
        """Bottom panel: volumetric body-resonator spectral display.

        Feeds the 3-D volume (n_time, n_sources, n_freq) to
        ``VolumetricAccumulatorWidget`` when a GL context is available.
        The software fallback max-projects the time window around the playhead
        and renders per-source coloured frequency bands via pygame.surfarray.
        """
        self._draw_panel_bg(surf, px, py, pw, ph,
                            "Body resonator  (spectral volume · per-source colour · ray-march)")

        cx = px + _MARGIN_L
        cy = py + _HEADER_H
        cw = pw - _MARGIN_L - _MARGIN_R
        ch = max(4, ph - _HEADER_H)

        key    = data_key or "resonator_volume"
        volume = self.sidecar.get(key) if self.sidecar is not None else None

        if volume is None or not _HAS_NP:
            f = self._small_font()
            if f:
                msg = f.render("no volume data — render audio first", True, _C_LABEL)
                surf.blit(msg, (cx + 4, cy + ch // 2 - msg.get_height() // 2))
            return

        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3:
            return

        # Store GL widget rect for external draw_gl() calls
        if _HAS_GL and VolumetricAccumulatorWidget is not None:
            if self._accum_widget is None:
                self._accum_widget = VolumetricAccumulatorWidget(spatial_mode=True)
            self._accum_widget.update_volume(volume)
            self._accum_widget.set_playhead(self.playhead_t)
            self._accum_widget._gl_accumulator_rect = (cx, cy, cw, ch)


    def _draw_rms_fallback(self, surf, x0, y0, w, h, data_key=None):
        if not _HAS_NP or self.sidecar is None:
            return
        key    = data_key or "premix_gl"
        premix = self.sidecar.get(key)
        if premix is None:
            premix = self.sidecar.get("premix")
        if premix is None or premix.ndim < 2 or premix.shape[0] == 0:
            return
        if np.iscomplexobj(premix):
            premix = premix.real

        gl_fps   = float(self.sidecar.get("gl_fps",
                         self.sidecar.get("display_fps", 120.0)))
        n_frames = premix.shape[1]
        t_c  = self.playhead_t * self.total_s
        f_lo = max(0, int((t_c - _GL_WINDOW_S * 0.5) * gl_fps))
        f_hi = min(n_frames, int((t_c + _GL_WINDOW_S * 0.5) * gl_fps) + 1)
        if f_hi <= f_lo:
            return

        colors   = [_C_ENVELOPE, _C_CHIRP, (200, 100, 220), (220, 180, 80),
                    (255, 80, 120), (80, 220, 200), (180, 255, 100)]
        n_voices = min(premix.shape[0], 7)
        win      = premix[:n_voices, f_lo:f_hi]
        peak     = float(win.max()) if win.size else 1.0
        if peak < 1e-9:
            peak = 1.0
        cy    = y0 + h // 2
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
        pts = _eval_curve(curve)
        if pts:
            spx = _curve_to_screen(pts, x0, y0, w, h, v_lo, v_hi)
            if len(spx) >= 2:
                pygame.draw.lines(surf, color, False, spx, 2)
                self._fill_curve(surf, spx, x0, y0, w, h, color)

        if f:
            hi_t = f.render(f"{v_hi:.3g}", True, _C_LABEL)
            lo_t = f.render(f"{v_lo:.3g}", True, _C_LABEL)
            surf.blit(hi_t, (px + 2, y0))
            surf.blit(lo_t, (px + 2, y0 + h - lo_t.get_height()))

        t_in    = self._playhead_t_in_atom(atom)
        v       = _eval_at(curve, max(0.0, min(1.0, t_in)))
        v_range = max(v_hi - v_lo, 1e-9)
        dot_x   = int(x0 + t_in * w)
        dot_y   = int(y0 + h - (v - v_lo) / v_range * h)
        dot_x   = max(x0, min(x0 + w, dot_x))
        dot_y   = max(y0, min(y0 + h, dot_y))
        pygame.draw.circle(surf, _C_PLAYHEAD, (dot_x, dot_y), 6)
        pygame.draw.circle(surf, _C_DOT_RING,  (dot_x, dot_y), 4)

        vk = getattr(atom, "voice_key", "")
        if f and vk:
            vt = f.render(vk, True, _C_LABEL)
            surf.blit(vt, (x0 + w - vt.get_width() - 2, y0 + 2))

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_progress_overlay(self, surf, px, py, pw, ph,
                                label: str, fraction: float) -> None:
        """Draw a semi-transparent progress bar at the bottom of a panel rect.

        Shows even when the panel already has content, so partial-load state
        is always visible.  Disappears once ``clear_panel_progress`` is called.
        """
        bar_h  = 20
        pad    = 4
        bar_y  = py + ph - bar_h - 2
        track_y = bar_y + bar_h - 7

        # Background strip
        bg = pygame.Surface((pw, bar_h), pygame.SRCALPHA)
        bg.fill((14, 12, 22, 210))
        surf.blit(bg, (px, bar_y))

        # Track
        pygame.draw.rect(surf, (40, 34, 60),
                         (px + pad, track_y, pw - pad * 2, 5),
                         border_radius=2)
        # Fill
        fill_w = max(0, int((pw - pad * 2) * fraction))
        if fill_w > 0:
            col = (70, 185, 95) if fraction >= 1.0 else (110, 70, 200)
            pygame.draw.rect(surf, col,
                             (px + pad, track_y, fill_w, 5),
                             border_radius=2)

        f = self._small_font()
        if f:
            lbl_s = f.render(label, True, (180, 165, 215))
            surf.blit(lbl_s, (px + pad + 2, bar_y + 2))
            pct_s = f.render(f"{int(fraction * 100)}%", True, (130, 120, 160))
            surf.blit(pct_s, (px + pw - pct_s.get_width() - pad - 2, bar_y + 2))

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
        poly    = list(screen_pts) + [(screen_pts[-1][0], floor_y),
                                      (screen_pts[0][0],  floor_y)]
        layer   = pygame.Surface((w + 2, h + 2), pygame.SRCALPHA)
        fill_col = (color[0], color[1], color[2], 45)
        shifted  = [(sx - x0, sy - y0) for sx, sy in poly]
        if len(shifted) >= 3:
            pygame.draw.polygon(layer, fill_col, shifted)
        surf.blit(layer, (x0, y0))

    # ── GL data extraction ────────────────────────────────────────────────────

    def _gl_batch_at_playhead(self, data_key: Optional[str] = None) -> Optional[Any]:
        """Return (n_voices, n_gl_frames) complex128 window for the phase-cloud widget."""
        if not _HAS_NP or self.sidecar is None:
            return None
        key = data_key or "premix_gl"
        gl  = self.sidecar.get(key)
        if gl is None:
            gl = self.sidecar.get("premix_gl")
        if gl is None:
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
        onset   = float(getattr(atom, "onset_time_s", 0.0))
        dur     = (float(getattr(atom, "duration_s", 0.0))
                   + float(getattr(atom, "release_tail_s", 0.0)))
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
