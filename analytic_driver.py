#!/usr/bin/env python3
"""analytic_driver.py

Analytic synthesizer voice editor — companion to the spectral-analyzer toolkit.

Layout
------
  left panel  (280 px) — voice list (voices + LFOs)
  center                — waveform / envelope / chirp editor
                          PlotWidget surface → GL texture + GL overlay
                          (cursors, control-point diamonds, loop handles)
  right panel (280 px)  — parameters for the selected voice

Run
---
    python analytic_driver.py [patch.json]

Dependencies
------------
    pip install numpy pygame PyOpenGL Pillow scipy
    (imports GlyphAtlas, Panel, PanelDock from bass_viewer)
"""
from __future__ import annotations

import argparse
import copy
import contextlib
from fractions import Fraction
import hashlib
import io
import json
import math
import os
import random as _random
import threading
import traceback
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, Optional
import torch
import numpy as np
import pygame
from scipy.signal import resample_poly as _scipy_resample_poly
from pygame.locals import (
    DOUBLEBUF, KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP,
    MOUSEMOTION, MOUSEWHEEL, OPENGL, QUIT, RESIZABLE, VIDEORESIZE,
    K_SPACE, K_ESCAPE, K_TAB, K_DELETE, K_s, K_o, K_n,
    K_LCTRL, K_RCTRL, K_z,
)
from pygame import KMOD_CTRL, KMOD_SHIFT
try:
    import OpenGL.GL as _ad_gl_stub
    for _ad_gl_name in (
        "GL_BLEND", "GL_CLAMP_TO_EDGE", "GL_COLOR_BUFFER_BIT", "GL_LINEAR",
        "GL_LINE_LOOP", "GL_LINE_STRIP", "GL_LINES", "GL_NEAREST",
        "GL_ONE_MINUS_SRC_ALPHA", "GL_QUADS", "GL_RGBA", "GL_SRC_ALPHA",
        "GL_TEXTURE_2D", "GL_TEXTURE_MAG_FILTER", "GL_TEXTURE_MIN_FILTER",
        "GL_TEXTURE_WRAP_S", "GL_TEXTURE_WRAP_T", "GL_TRIANGLES",
        "GL_UNSIGNED_BYTE", "GL_MODELVIEW", "GL_PROJECTION", "GL_SCISSOR_TEST",
    ):
        if not hasattr(_ad_gl_stub, _ad_gl_name):
            setattr(_ad_gl_stub, _ad_gl_name, 0)
    for _ad_gl_name in (
        "glBegin", "glBindTexture", "glBlendFunc", "glClear", "glClearColor",
        "glColor4f", "glDeleteTextures", "glDisable", "glEnable", "glEnd",
        "glGenTextures", "glLineWidth", "glTexCoord2f", "glTexImage2D",
        "glTexSubImage2D", "glTexParameteri", "glVertex2f", "glViewport",
        "glLoadIdentity", "glMatrixMode", "glOrtho", "glScissor",
    ):
        if not hasattr(_ad_gl_stub, _ad_gl_name):
            setattr(_ad_gl_stub, _ad_gl_name, lambda *args, **kwargs: 0)
except Exception:
    pass
from OpenGL.GL import (
    GL_BLEND, GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT, GL_LINEAR,
    GL_LINE_LOOP, GL_LINE_STRIP, GL_LINES, GL_NEAREST, GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS, GL_RGBA, GL_SRC_ALPHA, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_TRIANGLES, GL_UNSIGNED_BYTE, GL_MODELVIEW, GL_PROJECTION, GL_SCISSOR_TEST,
    glBegin, glBindTexture, glBlendFunc, glClear, glClearColor,
    glColor4f, glDeleteTextures, glDisable, glEnable, glEnd,
    glGenTextures, glLineWidth, glTexCoord2f, glTexImage2D, glTexSubImage2D,
    glLoadIdentity, glMatrixMode, glOrtho, glScissor, glTexParameteri, glVertex2f, glViewport,
)

from plot_widget import PlotWidget, PlotSeries, PlotMarker
from bass_viewer import (GlyphAtlas, Panel, PanelDock,
                         ScrollableSubpanelList, ModularSubpanelSpec, SubpanelAddOption,
                         FilterBankDecomposition)
import complex_phase_vocoder as cpv
from analysis_itinerary import AnalysisInventory
from parametric_curve import (
    ParametricCurve,
    EnvelopeRuleTree,
    ParametricCurveEngine,
    default_blank as _pc_default_blank,
    default_chirp as _pc_default_chirp,
    default_envelope as _pc_default_envelope,
    render_piecewise_audio,
    normalize_channel_complex_signals as _norm_ch_sigs,
    GateEvent,
)
from parametric_curve_editor import ParametricCurveEditor, _TextOverlay
from graph_solver import _T as _PROF

try:
    from signal_generator_v2 import KnobSpec
    _HAS_SGV2 = True
except Exception:
    _HAS_SGV2 = False
    # Minimal stub so knobs() classmethods can always be defined
    from dataclasses import dataclass as _kdc, field as _kfield
    @_kdc
    class KnobSpec:  # type: ignore[no-redef]
        name: str = ""; label: str = ""
        dtype: str = "float"; default: object = None
        low: float = 0.0; high: float = 1.0; step: float = 0.0; unit: str = ""
        choices: list = _kfield(default_factory=list)
        is_log: bool = False; group: str = ""; fmt: str = ".3g"
        source_class: str = ""; rebuild_layout: bool = False
        visible_when: object = None

try:
    from sequence_engine import (
        MODAL_SCALES, CHORD_PROGRESSIONS,
        ArpeggioRule, NoteEvent, NoteSchedule,
        scale_degrees_hz, semitones_to_hz,
        adsr_envelope_factory,
        piecewise_envelope_factory,
        SequenceProbabilities,
        NoteStream,
    )
    _HAS_SEQ_ENG = True
except Exception:
    _HAS_SEQ_ENG = False
    MODAL_SCALES: dict = {}
    CHORD_PROGRESSIONS: dict = {}
    class SequenceProbabilities:  # type: ignore[no-redef]
        def __init__(self): self.double_back = self.subversion = self.chromatic = self.modal = 0.0
        def to_dict(self): return {}
        @classmethod
        def from_dict(cls, d): return cls()
    class NoteStream:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): pass
        def reset(self): pass
        def exhausted(self): return True
        def next_hz(self): return None

ANALYTIC_GRAPH_SHADOW = 1
ANALYTIC_GRAPH_SHADOW_PROFILE = 1

try:
    from dynamics_engine import (
        DynamicsProgram, DynamicsCurve, AccentPattern, apply_dynamics,
        CURVE_SHAPES, _SCOPE_VALUES, _SCOPE_LABELS,
    )
    _HAS_DYN_ENG = True
except Exception:
    _HAS_DYN_ENG = False
    class DynamicsProgram:  # type: ignore[no-redef]
        def __init__(self): self.enabled = False; self.curve = type('C', (), {'shape': 'flat', 'scope_bars': 1.0, 'intensity': 0.5, 'to_dict': lambda s: {}})(); self.accent = type('A', (), {'levels': [], 'ensure_size': lambda s, n: None, 'level_at': lambda s, i: 1.0, 'cycle_level': lambda s, i: None, 'to_dict': lambda s: {}})()  # noqa
        def to_dict(self): return {}
        @classmethod
        def from_dict(cls, d): return cls()
    def apply_dynamics(*a, **kw): pass
    CURVE_SHAPES = ["flat"]
    _SCOPE_VALUES = [1.0]
    _SCOPE_LABELS = ["1"]

try:
    from improv_engine import (
        ImprovProgram, GraceParams, ChirpParams, EchoParams,
        apply_improv,
        GRACE_MODES, GRACE_POSNS, CHIRP_SHAPES, CHIRP_MODES,
    )
    _HAS_IMPROV_ENG = True
except Exception:
    _HAS_IMPROV_ENG = False
    class GraceParams:  # type: ignore[no-redef]
        def __init__(self): self.mode="chromatic"; self.position="pre"; self.duration_frac=0.10; self.trim_main=True; self.vel_scale=0.6; self.direction=0  # noqa
        def to_dict(self): return {}
        @classmethod
        def from_dict(cls, d): return cls()
    class ChirpParams:  # type: ignore[no-redef]
        def __init__(self): self.steps=4; self.shape="up"; self.mode="chromatic"; self.duration_frac=0.25; self.vel_scale=0.75  # noqa
        def to_dict(self): return {}
        @classmethod
        def from_dict(cls, d): return cls()
    class EchoParams:  # type: ignore[no-redef]
        def __init__(self): self.lookback_bars=1; self.duration_frac=0.5; self.vel_falloff=0.7; self.max_notes=4  # noqa
        def to_dict(self): return {}
        @classmethod
        def from_dict(cls, d): return cls()
    class ImprovProgram:  # type: ignore[no-redef]
        def __init__(self): self.enabled=False; self.prob_grace=0.0; self.prob_chirp=0.0; self.prob_echo=0.0; self.grace=GraceParams(); self.chirp=ChirpParams(); self.echo=EchoParams(); self.improv_steps=[[]]  # noqa
        def ensure_pattern_size(self, pi, n): pass
        def eligible(self, pi, si): return False
        def toggle_step(self, pi, si, n): pass
        def to_dict(self): return {}
        @classmethod
        def from_dict(cls, d): return cls()
    def apply_improv(*a, **kw): return []
    GRACE_MODES  = ["chromatic", "modal", "either"]
    GRACE_POSNS  = ["pre", "post", "both"]
    CHIRP_SHAPES = ["up", "down", "bounce", "random"]
    CHIRP_MODES  = ["chromatic", "modal"]

from rhythm_tree import (
    BeatNode, BeatTree, WarpCurve,
    build_warp_curve, iter_leaf_events, iter_grouped_events,
    GROUP_COLORS,
)

# ---------------------------------------------------------------------------
# Sequence / demo UI presets
# ---------------------------------------------------------------------------
_SEQ_SCALE_NAMES: list[str] = list(MODAL_SCALES.keys()) or ["pentatonic_minor"]
_SEQ_CHORD_NAMES: list[str] = list(CHORD_PROGRESSIONS.keys()) or ["I_IV_V_I"]

_RHYTHM_DIVISIONS: list[int] = [4, 8, 12, 16, 24, 32]
_SEQ_PATTERN_PRESETS: list[tuple[str, list[int]]] = [
    ("triad",    [0, 2, 4]),
    ("4-note",   [0, 2, 4, 7]),
    ("triad\u2195", [0, 2, 4, 2]),
    ("up-4",     [0, 1, 2, 3]),
    ("up-8",     [0, 1, 2, 3, 4, 5, 6, 7]),
    ("down-4",   [3, 2, 1, 0]),
    ("zigzag",   [0, 3, 1, 4, 2, 5]),
    ("walk",     [0, 1, 2, 3, 4, 3, 2, 1]),
    ("cascade",  [0, 2, 4, 7, 4, 2]),
    ("skip",     [0, 4, 2, 6, 1, 5]),
]
_SEQ_PATTERN_NAMES: list[str] = [n for n, _ in _SEQ_PATTERN_PRESETS]
_SEQ_RUBATO_SHAPES: list[str] = ["off", "sine", "troughs", "slow_go", "go_slow"]
_SEQ_RUBATO_SCOPES: list[str] = ["bar", "phrase"]
_METER_IRRATIONAL_SNAPS: list[float] = sorted([
    math.sqrt(2.0),
    math.sqrt(3.0),
    math.sqrt(5.0),
    (1.0 + math.sqrt(5.0)) * 0.5,
    math.e,
    math.pi,
    math.tau,
])

# Roman numeral → 0-based scale degree (try longest match first)
_ROMAN_DEGREE: dict[str, int] = {
    "VII": 6, "VI": 5, "IV": 3, "III": 2,
    "V": 4, "II": 1, "I": 0,
    "vii": 6, "vi": 5, "iv": 3, "iii": 2,
    "v": 4, "ii": 1, "i": 0,
}


def _chord_to_degree(chord_str: str) -> int:
    """Return 0-based scale degree from a Roman-numeral chord symbol."""
    s = chord_str.lstrip("-")
    for rn in ("VII", "VI", "IV", "III", "V", "II", "I",
               "vii", "vi",  "iv", "iii", "v",  "ii", "i"):
        if s.startswith(rn):
            return _ROMAN_DEGREE[rn]
    return 0


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
WIN_W_DEFAULT = 1600
WIN_H_DEFAULT = 900
PANEL_W = 280
TOPBAR_H = 30       # status / title bar
MODEBAR_H = 32      # mode-tab row height
BOTTOM_H = 28       # status-bar height

CP_HIT_PX   = 9    # control-point hit radius
CP_DRAW_PX  = 6    # diamond half-size for drawing
LOOP_HIT_PX = 8    # loop-handle hit column width

# ---------------------------------------------------------------------------
# Modulate-able voice/LFO/module attributes available for ParamNode targets.
# Each entry is the dot-path used in param_overrides / _synthesize_voice.
# ---------------------------------------------------------------------------
_VOICE_PARAM_ATTRS: list[str] = [
    "freq_hz",
    "amplitude",
    "semitone_offset",
    "chirp.f_delta_start",
    "chirp.f_delta_end",
    "adsr.attack",
    "adsr.decay",
    "adsr.sustain",
    "adsr.release",
    "fm.depth_hz",
    "fm.depth_amp",
    "am.depth_hz",
    "am.depth_amp",
    "harmonic_brightness",
    "harmonic_warp_strength",
    "harmonic_count",
    "granular.grain_density_hz",
    "granular.grain_duration_s",
    "granular.grain_scatter",
    "granular.grain_pitch_scatter",
    "granular.grain_manifold_mix",
    "granular.grain_amplitude_jitter",
]


# PlotWidget inner margins (must mirror PlotWidget constants)
_PW_ML = 36
_PW_MR = 6
_PW_MT = 14
_PW_MB = 14

# Colour palette (dark theme)
_C_BG       = (14,  14,  18,  255)
_C_PANEL    = (22,  22,  28,  255)
_C_WAVE     = (0.30, 0.62, 1.00, 1.0)
_C_ENV      = (1.00, 0.40, 0.20, 1.0)
_C_CHIRP    = (0.70, 0.30, 1.00, 1.0)
_C_LOOP     = (0.24, 0.78, 0.40, 1.0)
_C_LOOP_FILL= (0.10, 0.40, 0.18, 0.25)
_C_CURSOR   = (1.00, 0.72, 0.20, 1.0)
_C_CP_FILL  = (1.00, 0.80, 0.30, 0.9)
_C_CP_BORD  = (1.00, 1.00, 1.00, 1.0)
_C_GRID     = (0.16, 0.16, 0.20, 0.5)
_C_TAB_ACT  = (0.27, 0.51, 0.78, 1.0)
_C_TAB_IDLE = (0.14, 0.14, 0.18, 1.0)
_C_TXT      = (200, 200, 205)

# Pygame RGBA colour versions of some of the above
_PY_BG   = (14, 14, 18)
_PY_TXT  = (200, 200, 205)
_PY_DIM  = (90, 90, 100)
_PY_ACT  = (70, 130, 200)
_PY_WARN = (220, 100, 50)

# ---------------------------------------------------------------------------
# Split-module imports
# ---------------------------------------------------------------------------
import analytic_model as _analytic_model
import analytic_score as _analytic_score
import analytic_routing as _analytic_routing
import analytic_synth_legacy as _analytic_synth_legacy
import analytic_gl as _analytic_gl
from analytic_runtime import (
    clear_compiled_graph_cache,
    graph_shadow_summary,
    render_patch_graph,
    render_patch_legacy,
)


def _import_all_from(module):
    globals().update({
        k: v for k, v in vars(module).items()
        if k != "_import_all_from" and not (k.startswith('__') and k.endswith('__'))
    })


_import_all_from(_analytic_model)
_import_all_from(_analytic_score)
_import_all_from(_analytic_routing)
_import_all_from(_analytic_synth_legacy)
_import_all_from(_analytic_gl)

_SYNTH_LIST_AUDIO_DEVICES = _analytic_synth_legacy._list_audio_devices
_SYNTH_PROBE_DEFAULT_SOUNDDEVICE = _analytic_synth_legacy._probe_default_sounddevice
_SYNTH_PROBE_AUDIO_DEVICE = _analytic_synth_legacy._probe_audio_device
_SYNTH_REFRESH_SYSTEM_AUDIO_REPORT = _analytic_synth_legacy._refresh_system_audio_report
_SYNTH_PREPARE_OUTPUT_BUS_FOR_DEVICE = _analytic_synth_legacy._prepare_output_bus_for_device

# Keep audio-device helpers monkeypatch-compatible through the analytic_driver facade.
def _list_audio_devices(iscapture: bool) -> list[str]:
    return _SYNTH_LIST_AUDIO_DEVICES(iscapture)


def _probe_default_sounddevice(iscapture: bool, requested_channels: int, requested_rate: int | None = None):
    return _SYNTH_PROBE_DEFAULT_SOUNDDEVICE(iscapture, requested_channels, requested_rate)


def _probe_audio_device(name: str, iscapture: bool, requested_channels: int, requested_rate: int | None = None):
    _analytic_synth_legacy._list_audio_devices = globals()["_list_audio_devices"]
    _analytic_synth_legacy._probe_default_sounddevice = globals()["_probe_default_sounddevice"]
    return _SYNTH_PROBE_AUDIO_DEVICE(name, iscapture, requested_channels, requested_rate)


def _refresh_system_audio_report(sysdev: "SystemAudioDevice") -> None:
    _analytic_synth_legacy._list_audio_devices = globals()["_list_audio_devices"]
    _analytic_synth_legacy._probe_audio_device = globals()["_probe_audio_device"]
    return _SYNTH_REFRESH_SYSTEM_AUDIO_REPORT(sysdev)


def _prepare_output_bus_for_device(out_bus: np.ndarray, src_sr: int, dst_sr: int, dst_channels: int) -> np.ndarray:
    return _SYNTH_PREPARE_OUTPUT_BUS_FOR_DEVICE(out_bus, src_sr, dst_sr, dst_channels)

class EditorMode(Enum):
    WAVEFORM      = auto()
    ENVELOPE      = auto()
    PIECEWISE_EDITOR = auto()
    CHIRP         = auto()
    COMPLEX_RI    = auto()   # real and imaginary components
    COMPLEX_MP    = auto()   # magnitude and phase
    HARMONICS     = auto()   # harmonic amplitude spectrum bars
    LFO_VIEW      = auto()   # all LFO waveforms over time
    FM_VIEW       = auto()   # instantaneous frequency including FM deviation
    MIX           = auto()   # all voices summed into final mix
    ROUTING       = auto()   # N×N signal routing matrix (knob grid)
    CONTROL_ROUTING = auto() # rack/backplane view for analytic control routing
    PARAM_ROUTING = auto()   # parametric routing view for ParamNode targets
    SM_LOG        = auto()   # state-machine plugin output log
    SCORE         = auto()   # phrase table: all registered parts × full phrase bars
    PIANO_ROLL    = auto()   # resolved sequence roll with pitch/time editing
    PLACEMENT     = auto()   # top-down orchestral placement vector diagram


@dataclass
class DragState:
    active:      bool  = False
    kind:        str   = ""   # "knot"|"loop_start"|"loop_end"|"cursor"
    index:       int   = 0
    voice_key:   str   = ""
    start_px:    tuple = (0, 0)
    start_val:   tuple = (0.0, 0.0)  # (t_norm, v_norm) at drag start


# ---------------------------------------------------------------------------
# Phase-boundary snapping for loop handles
# ---------------------------------------------------------------------------

def _snap_to_phase_boundary(
    t_norm:       float,
    voice:        "AnalyticVoice",
    patch:        "AnalyticPatch",
    phase_cycles: "np.ndarray | None",
) -> float:
    """Snap normalized time to the nearest complete-revolution boundary.

    Loop segments must begin and end on integer-cycle boundaries so that
    the analytic phase is phase-continuous at the splice point.  The
    ``phase_cycles`` array holds the cumulative cycle count at every
    sample (computed from the chirp-resolved instantaneous frequency,
    excluding FM modulation which is a small overlay).
    """
    if phase_cycles is None or len(phase_cycles) == 0:
        return max(0.0, min(1.0, t_norm))
    n = len(phase_cycles)
    i_raw = int(round(t_norm * (n - 1)))
    i_raw = max(0, min(n - 1, i_raw))

    # Revolution crossings: samples where floor(phase_cycles) increments.
    # Search within ±10 % of total length around the raw position.
    half = max(int(n * 0.10), 4)
    lo = max(0, i_raw - half)
    hi = min(n - 1, i_raw + half)

    window = phase_cycles[lo : hi + 1]
    floor_diff = np.floor(window[1:]) - np.floor(window[:-1])
    crossing_rel = np.where(floor_diff != 0)[0]   # offsets within [lo, hi-1]

    if len(crossing_rel) == 0:
        # No crossing in window — interpolate to nearest integer cycle.
        target = round(float(phase_cycles[i_raw]))
        target = max(1.0, float(target))
        if target >= phase_cycles[-1]:
            return 1.0
        idx = int(np.searchsorted(phase_cycles, target))
        return float(np.clip(idx / n, 0.0, 1.0))

    crossing_abs = crossing_rel + lo
    nearest_idx = int(crossing_abs[np.argmin(np.abs(crossing_abs - i_raw))])
    return float(np.clip(nearest_idx / n, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------

def _get_nested_attr(obj: Any, path: str) -> Any:
    """Traverse a dot-separated attribute path, returning None on any missing step."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def _set_nested_attr(obj: Any, path: str, float_val: float, knob: KnobSpec) -> None:
    """Apply *float_val* to *obj* at the dot-path described by *knob*."""
    parts = path.split(".")
    target = obj
    for p in parts[:-1]:
        sub = getattr(target, p, None)
        if sub is None:
            if p in ("fm", "am"):
                setattr(target, p, ModRouting())
            elif p == "granular":
                _ensure_granular(target)
            sub = getattr(target, p)
        target = sub
    attr = parts[-1]
    if knob.dtype == "choice":
        idx = max(0, min(len(knob.choices) - 1, round(float_val)))
        setattr(target, attr, knob.choices[idx])
    elif knob.dtype == "int":
        setattr(target, attr, int(round(float_val)))
    elif knob.dtype == "bool":
        setattr(target, attr, bool(round(float_val)))
    else:
        setattr(target, attr, float(float_val))
def _hz_to_note_name(hz: float, tuning: "GlobalTuning | None" = None) -> str:
    """Return a human-readable pitch name for *hz* appropriate to *tuning*.

    - 12-TET / pythagorean / just (12 pitch classes): Western note name
      (C4, A#3, …) — MIDI-based, invariant to tuning.root_hz.
    - 22-śruti custom (Indian ragas): Sargam name (Sa, Re, koGa, Pa, …)
      expressed as śruti degree from tuning.root_hz (Sa).
    - Any other custom N-TET: shows "+N st" step offset from root.
    """
    if hz <= 0.0:
        return "---"

    # ── 12-pitch-class path (12tet, just, pythagorean, or 12-entry custom) ──
    is_12_class = (
        tuning is None
        or tuning.temperament in ("12tet", "just", "pythagorean")
        or (tuning.temperament == "custom" and len(tuning.custom_cents) == 12)
    )
    if is_12_class:
        semitones_from_a4 = 12.0 * math.log2(hz / 440.0)
        midi   = round(semitones_from_a4) + 69   # A4 = MIDI 69
        octave = (midi // 12) - 1
        pc     = midi % 12
        return f"{_NOTE_NAMES_12[pc]}{octave}"

    # ── 22-śruti path ────────────────────────────────────────────────────────
    if tuning.temperament == "custom" and len(tuning.custom_cents) == 22:
        st     = tuning.hz_to_semitones(hz)        # steps in 22-śruti space from root_hz
        idx    = round(st)
        octave = idx // 22
        sruti  = idx % 22
        name   = _SRUTI_NAMES_22[sruti]
        if octave == 0:
            return name
        return f"{name}{'+' if octave > 0 else ''}{octave}oct"

    # ── Generic N-TET fallback ───────────────────────────────────────────────
    dpo = tuning.divisions_per_octave
    st  = tuning.hz_to_semitones(hz)
    idx = round(st)
    octave = idx // dpo
    deg    = idx % dpo
    if octave == 0:
        return f"st{deg}"
    return f"st{deg}{'+' if octave > 0 else ''}{octave}oct"


def _draw_routing_knob(surf: pygame.Surface, cx: int, cy: int, r: int,
                       value: float, active: bool, is_diag: bool) -> None:
    """Draw one bipolar rotary knob.

    Geometry (standard-math angles, CCW from +x):
        7 o'clock (min = −2) → 240°
        12 o'clock (zero)    →  90°
        5 o'clock (max = +2) → 300°  (going CW through 12 o'clock)
    """
    r = max(r, 6)
    v = max(-2.0, min(2.0, value))

    # Background circle
    bg = (38, 28, 28) if is_diag else (24, 24, 34)
    pygame.draw.circle(surf, bg, (cx, cy), r)
    pygame.draw.circle(surf, (55, 55, 72), (cx, cy), r, 1)

    # Track arc: from 300° CCW to 240° (= long way through 12 o'clock)
    tr = max(r - 3, 3)
    t_rect = pygame.Rect(cx - tr, cy - tr, tr * 2, tr * 2)
    pygame.draw.arc(surf, (50, 50, 65), t_rect,
                    math.radians(300.0), math.radians(240.0), max(1, r // 7))

    # Needle angle: t=0 → 240°, t=1 → 300° going CW (= decreasing standard angle)
    t_norm      = (v + 2.0) / 4.0          # [0, 1]
    needle_deg  = 240.0 - 300.0 * t_norm   # 240° → -60° (=300°)
    needle_rad  = math.radians(needle_deg)
    center_rad  = math.pi / 2.0             # 12 o'clock

    # Fill arc (bipolar, emanating from center)
    fr = max(r - 5, 2)
    f_rect = pygame.Rect(cx - fr, cy - fr, fr * 2, fr * 2)
    arc_w = max(2, r // 5)
    if abs(v) > 0.02:
        if v > 0:
            # CCW from needle_rad up to center_rad
            pygame.draw.arc(surf, (40, 100, 220), f_rect,
                            needle_rad, center_rad, arc_w)
        else:
            # CCW from center_rad to needle_rad
            pygame.draw.arc(surf, (210, 70, 50), f_rect,
                            center_rad, needle_rad, arc_w)

    # Snap-grid tick marks (at each 1/12 step, -2 to +2)
    if r >= 14:
        for tick_v in np.arange(-2.0, 2.0 + 1e-9, _SNAP_12TH):
            if abs(abs(tick_v) % 1.0) < 1e-9:
                continue   # skip integer ticks (too prominent, drawn separately)
            td  = 240.0 - 300.0 * (tick_v + 2.0) / 4.0
            tra = math.radians(td)
            tx0 = cx + int((r - 2) * math.cos(tra))
            ty0 = cy - int((r - 2) * math.sin(tra))
            tx1 = cx + int((r - 4) * math.cos(tra))
            ty1 = cy - int((r - 4) * math.sin(tra))
            pygame.draw.line(surf, (45, 45, 58), (tx0, ty0), (tx1, ty1), 1)

    # Integer value tick marks
    if r >= 10:
        for tick_v in [-2, -1, 0, 1, 2]:
            td  = 240.0 - 300.0 * (tick_v + 2.0) / 4.0
            tra = math.radians(td)
            tlen = 5 if tick_v == 0 else 4
            tx0 = cx + int((r - 1) * math.cos(tra))
            ty0 = cy - int((r - 1) * math.sin(tra))
            tx1 = cx + int((r - 1 - tlen) * math.cos(tra))
            ty1 = cy - int((r - 1 - tlen) * math.sin(tra))
            col = (80, 80, 100) if tick_v != 0 else (130, 130, 150)
            pygame.draw.line(surf, col, (tx0, ty0), (tx1, ty1), 1)

    # Needle
    nr = max(3, r - 5)
    nx = cx + int(nr * math.cos(needle_rad))
    ny = cy - int(nr * math.sin(needle_rad))
    n_col = (240, 240, 255) if abs(v) > 0.02 else (60, 60, 80)
    pygame.draw.line(surf, n_col, (cx, cy), (nx, ny), 2 if r > 12 else 1)

    # Center dot
    pygame.draw.circle(surf, (80, 80, 110), (cx, cy), max(2, r // 6))

    # Active highlight ring
    if active:
        pygame.draw.circle(surf, (80, 155, 255), (cx, cy), r, 2)


class RoutingGridView:
    """Center-view panel: N×N knob grid for analytic signal routing.

    Rows = destination nodes, Columns = source nodes.
    Knob (row i, col j) = value (amplitude / angle / delay) from node-j to node-i.

    Controls
    --------
    Drag up/down      change value, snaps unless Shift held
    Shift + drag      fine movement, no snap
    Right-click       reset cell to zero
    Tab bar at top    switch between Mix / Angle / Delay grids
    """

    KNOB_MIN  = -2.0
    KNOB_MAX  =  2.0
    DRAG_PX_PER_UNIT       = 80.0
    DRAG_PX_PER_UNIT_SHIFT = 800.0

    # Routing sub-tabs
    _ROUTING_TABS  = [("mix",    "Mix"),
                      ("angle",  "Angle"),
                      ("delay",  "Delay"),
                      ("export", "Export")]
    _TAB_H = 22   # pixel height of the tab bar
    _DCLICK_CYCLE = [0.0, 1.0, -1.0]
    _DCLICK_MS    = 300  # double-click timeout in milliseconds

    def __init__(self) -> None:
        self._surf:  pygame.Surface | None = None
        self._dirty: bool = True
        self._font:  pygame.font.Font | None = None
        self.active_key: str = ""
        # Geometry (filled during render)
        self._cell_size: int = 48
        self._grid_x0:   int = 0
        self._grid_y0:   int = 0
        self._N:         int = 0
        # Interaction state
        self._active_cell: tuple[int, int] | None = None
        self._drag_cell:   tuple[int, int] | None = None
        self._drag_y0:     int   = 0
        self._drag_val0:   float = 0.0
        self._fb_rect:     pygame.Rect | None = None
        # Double-click detection — first click is deferred (pending);
        # drag only starts after _DCLICK_MS expires without a second click.
        self._pending_click: tuple[int, int] | None = None  # (ri, ci)
        self._pending_click_tick: int  = 0
        self._pending_click_y0:   int  = 0   # raw screen y for drag origin
        self._pending_click_val0: float = 0.0
        self._last_click_cell: tuple[int, int] | None = None
        self._last_click_tick: int = 0
        # Sub-tab state
        self._routing_tab: str = "mix"
        self._tab_rects:   list = []
        self._export_rows: list = []  # [{mixer, chk_r, sr_l, sr_r, bd_l, bd_r}]
        # Scroll state
        self._scroll_col:  int = 0
        self._scroll_row:  int = 0
        self._vis_cols:    int = 0   # filled during render
        self._vis_rows:    int = 0
        self._vp_x:        int = 0   # pixel x of data area left edge
        self._vp_y:        int = 0   # pixel y of data area top edge

    def mark_dirty(self) -> None:
        self._dirty = True

    def _font_(self) -> pygame.font.Font:
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("consolas", 11)
        return self._font

    # ---- Rendering ---------------------------------------------------------

    def render_surface(self, patch: AnalyticPatch, w: int, h: int,
                       font: pygame.font.Font | None = None) -> pygame.Surface:
        # ensure_defaults is intentionally NOT called here — it would silently
        # recreate edges the user just deleted (e.g. cycled to zero), undoing
        # explicit user edits on every frame.
        if (self._surf is None
                or self._surf.get_width() != w
                or self._surf.get_height() != h):
            self._surf = pygame.Surface((w, h))
            self._dirty = True
        if not self._dirty:
            return self._surf
        self._surf.fill(_PY_BG)
        self._render_grid(self._surf, patch, font)
        self._dirty = False
        return self._surf

    def _render_grid(self, surf: pygame.Surface, patch: AnalyticPatch,
                     ext_font: pygame.font.Font | None) -> None:
        keys   = _routing_grid_node_keys(patch, getattr(self, "active_key", ""))
        labels = [_routing_node_label(k, patch) for k in keys]
        colors = [_routing_node_color(k, patch) for k in keys]
        N      = len(keys)
        self._N = N
        g, active_router = _active_routing_graph_and_instance(
            patch, getattr(self, "active_key", "")
        )
        font   = ext_font or self._font_()
        fh     = font.get_height()
        w, h   = surf.get_size()

        # ---- Tab bar -------------------------------------------------------
        tab_keys   = [t[0] for t in self._ROUTING_TABS]
        tab_labels = [t[1] for t in self._ROUTING_TABS]
        TAB_H = self._TAB_H
        tab_w  = 64
        tab_pad = 4
        self._tab_rects = []
        for ti, (tk, tl) in enumerate(zip(tab_keys, tab_labels)):
            tr = pygame.Rect(tab_pad + ti * (tab_w + 2), tab_pad,
                             tab_w, TAB_H - 2 * tab_pad)
            self._tab_rects.append(tr)
            active = (self._routing_tab == tk)
            bg = (40, 90, 160) if active else (28, 28, 38)
            pygame.draw.rect(surf, bg, tr, border_radius=3)
            pygame.draw.rect(surf, (60, 60, 85), tr, 1, border_radius=3)
            tc = (220, 235, 255) if active else (100, 100, 120)
            ts = font.render(tl, True, tc)
            surf.blit(ts, (tr.x + (tr.w - ts.get_width()) // 2,
                           tr.y + (tr.h - fh) // 2))

        # Tab hint line
        hint = {"mix":   "amplitude weights  |  drag \u2195  right-click=0  shift=fine",
                "angle": "phase angle per edge (\u00b1180\u00b0)  |  drag \u2195  right-click=0  shift=fine",
                "delay": "propagation delay per edge (ms)  |  drag \u2195  right-click=0  shift=fine",
                "export": "select mixers to export and configure sample rate / bit depth",
                }.get(self._routing_tab, "")
        hs = font.render(hint, True, (60, 60, 80))
        surf.blit(hs, (tab_pad + len(tab_keys) * (tab_w + 2) + 8, tab_pad + (TAB_H - 2 * tab_pad - fh) // 2))
        if active_router is not None:
            r_lbl = font.render(
                f"router: {active_router.label} [{active_router.router_type}]",
                True,
                (120, 165, 210),
            )
            surf.blit(r_lbl, (8, TAB_H + 2))

        # Export tab — show per-mixer export settings; skip NxN routing grid
        if self._routing_tab == "export":
            self._draw_export_panel(surf, patch, font, w, h, TAB_H)
            return

        # ---- Scrollable grid geometry -----------------------------------------
        # Two header strips on each axis (top+bottom, left+right).
        status_h = (fh + 4) * (3 if active_router is not None else 2) + 10
        avail_w   = w - 8
        avail_h   = h - TAB_H - status_h - 8
        cs        = max(24, min(48, min(avail_w // 3, max(1, avail_h // 3))))
        self._cell_size = cs

        gx = 8
        gy = TAB_H + 4
        vp_x = gx + cs                  # left edge of data area
        vp_y = gy + cs                  # top edge of data area
        vis_cols = max(1, (avail_w - 2 * cs) // cs)
        vis_rows = max(1, (avail_h - 2 * cs) // cs)
        vis_cols = min(vis_cols, N)
        vis_rows = min(vis_rows, N)
        self._scroll_col = max(0, min(self._scroll_col, N - vis_cols))
        self._scroll_row = max(0, min(self._scroll_row, N - vis_rows))
        sc = self._scroll_col
        sr = self._scroll_row
        self._grid_x0 = gx
        self._grid_y0 = gy
        self._vis_cols = vis_cols
        self._vis_rows = vis_rows
        self._vp_x     = vp_x
        self._vp_y     = vp_y

        right_x  = vp_x + vis_cols * cs
        bottom_y = vp_y + vis_rows * cs
        total_w  = (vis_cols + 2) * cs
        total_h  = (vis_rows + 2) * cs

        # Grid background
        pygame.draw.rect(surf, (18, 18, 24),
                         pygame.Rect(gx, gy, total_w, total_h), border_radius=6)

        # Helper: draw one header cell
        def _draw_header(hx, hy, lbl, col):
            dark = (max(0, col[0] - 80), max(0, col[1] - 80), max(0, col[2] - 80))
            pygame.draw.rect(surf, dark,
                             pygame.Rect(hx + 2, hy + 2, cs - 4, cs - 4),
                             border_radius=4)
            ts = font.render(lbl[:5], True, (195, 195, 210))
            surf.blit(ts, (hx + (cs - ts.get_width()) // 2, hy + (cs - fh) // 2))

        # Column headers — top and bottom
        for jv in range(vis_cols):
            j = sc + jv
            _draw_header(vp_x + jv * cs, gy,       labels[j], colors[j])
            _draw_header(vp_x + jv * cs, bottom_y, labels[j], colors[j])

        # Row headers — left and right
        for iv in range(vis_rows):
            i = sr + iv
            _draw_header(gx,      vp_y + iv * cs, labels[i], colors[i])
            _draw_header(right_x, vp_y + iv * cs, labels[i], colors[i])

        # Corner labels
        def _corner(cx0, cy0, txt_s):
            ts = font.render(txt_s, True, (60, 60, 80))
            surf.blit(ts, (cx0 + (cs - ts.get_width()) // 2, cy0 + (cs - fh) // 2))

        _corner(gx,      gy,        "src\u2192")
        _corner(gx,      bottom_y,  "src\u2192")
        _corner(right_x, gy,        "\u2193dst")
        _corner(right_x, bottom_y,  "\u2193dst")

        # Scroll indicators
        if N > vis_cols and vis_cols > 0:
            bar_w = max(4, cs * vis_cols // N)
            bar_x = vp_x + sc * (vis_cols * cs - bar_w) // max(1, N - vis_cols)
            for _by in (gy + cs - 4, bottom_y + cs - 4):
                pygame.draw.rect(surf, (60, 90, 150),
                                 pygame.Rect(bar_x, _by, bar_w, 3))
        if N > vis_rows and vis_rows > 0:
            bar_h = max(4, cs * vis_rows // N)
            bar_y = vp_y + sr * (vis_rows * cs - bar_h) // max(1, N - vis_rows)
            for _bx in (gx + cs - 4, right_x + cs - 4):
                pygame.draw.rect(surf, (60, 90, 150),
                                 pygame.Rect(_bx, bar_y, 3, bar_h))

        # Knob cells (clipped to data viewport)
        knob_r    = max(6, cs // 2 - 5)
        data_clip = pygame.Rect(vp_x, vp_y, vis_cols * cs, vis_rows * cs)
        surf.set_clip(data_clip)
        for iv in range(vis_rows):
            i = sr + iv
            for jv in range(vis_cols):
                j = sc + jv
                cx = vp_x + jv * cs + cs // 2
                cy = vp_y + iv * cs + cs // 2
                src_k, dst_k = keys[j], keys[i]
                active  = (self._active_cell == (i, j))
                is_diag = (i == j)

                if self._routing_tab == "mix":
                    raw    = g.get_weight(src_k, dst_k)
                    knob_v = raw
                    show   = abs(raw) > 0.005 or active
                    lbl_s  = f"{raw:+.2f}"
                elif self._routing_tab == "angle":
                    raw_r  = g.get_angle_rad(src_k, dst_k)
                    deg    = raw_r * 180.0 / math.pi
                    knob_v = deg / 90.0
                    show   = abs(raw_r) > 0.005 or active
                    lbl_s  = f"{deg:+.0f}\u00b0"
                else:  # delay
                    raw_ms = g.get_delay_s(src_k, dst_k) * 1000.0
                    knob_v = raw_ms / 1000.0
                    show   = abs(raw_ms) > 0.5 or active
                    lbl_s  = f"{raw_ms:+.1f}ms"

                _draw_routing_knob(surf, cx, cy, knob_r, knob_v, active, is_diag)
                if show and cs >= 40:
                    v_txt = font.render(lbl_s, True, _PY_DIM)
                    surf.blit(v_txt, (cx - v_txt.get_width() // 2,
                                      cy + knob_r + 2))
        surf.set_clip(None)

        # Grid lines
        gl_col = (38, 38, 50)
        for iv in range(vis_rows + 3):
            y = gy + iv * cs
            pygame.draw.line(surf, gl_col, (gx, y), (gx + total_w, y))
        for jv in range(vis_cols + 3):
            x = gx + jv * cs
            pygame.draw.line(surf, gl_col, (x, gy), (x, gy + total_h))

        # Active cell readout
        status_y = gy + total_h + 6
        if self._active_cell is not None:
            ri, ci = self._active_cell
            if 0 <= ri < N and 0 <= ci < N:
                src_k2, dst_k2 = keys[ci], keys[ri]
                if self._routing_tab == "mix":
                    val = g.get_weight(src_k2, dst_k2)
                    snap_v = round(val / _SNAP_12TH) * _SNAP_12TH
                    info = (f"{labels[ci]} \u2192 {labels[ri]}:  {val:+.4f}"
                            f"   (snap \u00b11/12: {snap_v:+.4f})")
                elif self._routing_tab == "angle":
                    deg = g.get_angle_rad(src_k2, dst_k2) * 180.0 / math.pi
                    snap_deg = round(deg / 15.0) * 15.0
                    info = (f"{labels[ci]} \u2192 {labels[ri]}:  {deg:+.1f}\u00b0"
                            f"   (snap 15\u00b0: {snap_deg:+.0f}\u00b0)")
                else:
                    ms = g.get_delay_s(src_k2, dst_k2) * 1000.0
                    info = f"{labels[ci]} \u2192 {labels[ri]}:  {ms:.2f}ms"
                info_s = font.render(info, True, (180, 210, 255))
                surf.blit(info_s, (gx, status_y))
                status_y += fh + 2

        fb = g.feedback
        fb_col  = (80, 220, 80) if fb.enabled else _PY_DIM
        fb_txt  = (f"[Feedback: {'ON' if fb.enabled else 'off'}]"
                   f"  delay={fb.delay_s * 1000:.1f}ms"
                   f"  decay={fb.decay:.3f}   (click to toggle)")
        fb_s = font.render(fb_txt, True, fb_col)
        surf.blit(fb_s, (gx, status_y))
        self._fb_rect = pygame.Rect(gx, status_y, fb_s.get_width(), fb_s.get_height())

    # ---- Export panel (shown when routing_tab == "export") -----------------

    _SR_OPTIONS = [22050, 44100, 48000, 88200, 96000, 192000]
    _BD_OPTIONS = [16, 24, 32]

    def _draw_export_panel(self, surf: pygame.Surface, patch: "AnalyticPatch",
                           font: pygame.font.Font, w: int, h: int, top_y: int) -> None:
        """Draw the mixer export-settings panel; populate self._export_rows."""
        fh    = font.get_height()
        row_h = max(fh + 10, 26)
        px    = 14
        py    = top_y + 10
        self._export_rows = []

        hdr = font.render("Mixer export settings", True, (160, 180, 220))
        surf.blit(hdr, (px, py))
        py += fh + 8

        for m in patch.mixers:
            bg_col = (30, 50, 70) if m.export_to_file else (25, 25, 35)
            pygame.draw.rect(surf, bg_col,
                             pygame.Rect(px - 4, py - 2, w - 2 * px + 8, row_h + 4),
                             border_radius=4)
            # Checkbox
            chk_r = pygame.Rect(px, py + (row_h - 14) // 2, 14, 14)
            chk_col = (80, 200, 80) if m.export_to_file else (55, 55, 75)
            pygame.draw.rect(surf, chk_col, chk_r, border_radius=2)
            if m.export_to_file:
                surf.blit(font.render("\u2713", True, (20, 20, 20)), (chk_r.x + 1, chk_r.y - 1))
            # Mixer label
            lbl_col = (220, 220, 120) if m.export_to_file else (130, 130, 140)
            surf.blit(font.render(m.label[:14], True, lbl_col),
                      (chk_r.right + 6, py + (row_h - fh) // 2))
            # Sample-rate stepper
            sr_lbl = f"SR:{m.export_sample_rate}"
            sr_lbl_w = font.size(sr_lbl)[0]
            sr_x = w - (sr_lbl_w + 40) - (font.size(f"{m.export_bit_depth}b")[0] + 40) - 8
            sr_l  = pygame.Rect(sr_x,               py + 2, 16, row_h - 4)
            sr_r  = pygame.Rect(sr_x + 18 + sr_lbl_w, py + 2, 16, row_h - 4)
            pygame.draw.rect(surf, (50, 55, 70), sr_l, border_radius=2)
            pygame.draw.rect(surf, (50, 55, 70), sr_r, border_radius=2)
            surf.blit(font.render("\u25c4", True, (160, 200, 220)), (sr_l.x + 2, sr_l.y + 1))
            surf.blit(font.render("\u25ba", True, (160, 200, 220)), (sr_r.x + 2, sr_r.y + 1))
            surf.blit(font.render(sr_lbl, True, (160, 180, 200)),
                      (sr_l.right + 2, py + (row_h - fh) // 2))
            # Bit-depth stepper
            bd_lbl   = f"{m.export_bit_depth}b"
            bd_lbl_w = font.size(bd_lbl)[0]
            bd_x = sr_r.right + 10
            bd_l  = pygame.Rect(bd_x,               py + 2, 16, row_h - 4)
            bd_r  = pygame.Rect(bd_x + 18 + bd_lbl_w, py + 2, 16, row_h - 4)
            pygame.draw.rect(surf, (50, 55, 70), bd_l, border_radius=2)
            pygame.draw.rect(surf, (50, 55, 70), bd_r, border_radius=2)
            surf.blit(font.render("\u25c4", True, (160, 200, 220)), (bd_l.x + 2, bd_l.y + 1))
            surf.blit(font.render("\u25ba", True, (160, 200, 220)), (bd_r.x + 2, bd_r.y + 1))
            surf.blit(font.render(bd_lbl, True, (160, 180, 200)),
                      (bd_l.right + 2, py + (row_h - fh) // 2))

            self._export_rows.append({
                "mixer": m, "chk_r": chk_r,
                "sr_l":  sr_l, "sr_r":  sr_r,
                "bd_l":  bd_l, "bd_r":  bd_r,
            })
            py += row_h + 8

        if not patch.mixers:
            surf.blit(font.render("(no mixers defined)", True, (80, 80, 100)), (px, py))

    # ---- Hit testing -------------------------------------------------------

    def _cell_at(self, lx: int, ly: int) -> tuple[int, int] | None:
        cs = self._cell_size
        vp_x, vp_y = self._vp_x, self._vp_y
        vis_w = self._vis_cols * cs
        vis_h = self._vis_rows * cs
        if not (vp_x <= lx < vp_x + vis_w and vp_y <= ly < vp_y + vis_h):
            return None
        col_i = (lx - vp_x) // cs + self._scroll_col
        row_i = (ly - vp_y) // cs + self._scroll_row
        if 0 <= row_i < self._N and 0 <= col_i < self._N:
            return (row_i, col_i)
        return None

    # ---- Event handling ----------------------------------------------------

    def handle_event(self, event: pygame.event.Event,
                     patch: AnalyticPatch,
                     canvas_x: int, canvas_y: int) -> bool:
        """Returns True if event was consumed (caller should mark rebuild)."""
        from pygame.locals import K_LSHIFT, K_RSHIFT
        keys_list = _routing_grid_node_keys(patch, self.active_key)
        g, _active_router = _active_routing_graph_and_instance(patch, self.active_key)
        tab = self._routing_tab
        tab_keys = [t[0] for t in self._ROUTING_TABS]

        def _lx_ly(pos):
            return pos[0] - canvas_x, pos[1] - canvas_y

        def _read_cell(ri, ci):
            sk, dk = keys_list[ci], keys_list[ri]
            if tab == "mix":   return g.get_weight(sk, dk)
            if tab == "angle": return g.get_angle_rad(sk, dk)
            return g.get_delay_s(sk, dk)

        def _write_cell(ri, ci, v):
            sk, dk = keys_list[ci], keys_list[ri]
            if tab == "mix":
                g.set_weight(sk, dk, round(v, 10))
                # M1 fix: prune zero-weight edges immediately so the grid doesn't
                # display ghost knobs after a weight is dragged to zero.
                if abs(v) <= 1e-12:
                    g.prune()
            elif tab == "angle": g.set_angle_rad(sk, dk, round(v, 10))
            else:              g.set_delay_s(sk, dk, round(v, 8))

        def _snap(v, shift):
            if shift: return v
            if tab == "mix":   return round(v / _SNAP_12TH) * _SNAP_12TH
            if tab == "angle": return round(v / (math.pi / 12.0)) * (math.pi / 12.0)
            return v  # delay: no snap

        def _clamp(v):
            if tab == "mix":   return max(-2.0, min(2.0, v))
            if tab == "angle": return max(-math.pi, min(math.pi, v))
            return max(-2.0, min(2.0, v))  # delay: negative = pre-advance (ALC)

        def _delta_from_dy(dy, shift):
            # Returns raw-unit delta from pixel drag distance (upward = positive)
            if tab == "mix":
                ppu = self.DRAG_PX_PER_UNIT_SHIFT if shift else self.DRAG_PX_PER_UNIT
                return dy * 4.0 / ppu
            if tab == "angle":
                # 80px normal = π rad (180°); shift = 800px = π rad
                scale = (math.pi / 800.0) if shift else (math.pi / 80.0)
                return dy * scale
            # delay: 1px normal = 1ms = 0.001s; shift = 0.1ms
            return dy * (0.0001 if shift else 0.001)

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            lx, ly = _lx_ly(event.pos)
            # Sub-tab clicks
            for ti, tr in enumerate(self._tab_rects):
                if tr.collidepoint(lx, ly):
                    self._routing_tab = tab_keys[ti]
                    self._active_cell = None
                    self._drag_cell   = None
                    self._dirty = True
                    return True
            # Feedback toggle
            if self._fb_rect is not None and self._fb_rect.collidepoint(lx, ly):
                g.feedback.enabled = not g.feedback.enabled
                self._dirty = True
                return True
            # Export tab: checkbox and stepper clicks (grid cells are not shown here)
            if self._routing_tab == "export":
                for row in self._export_rows:
                    if row["chk_r"].collidepoint(lx, ly):
                        row["mixer"].export_to_file = not row["mixer"].export_to_file
                        self._dirty = True
                        return True
                    if row["sr_l"].collidepoint(lx, ly):
                        opts = self._SR_OPTIONS
                        idx  = opts.index(row["mixer"].export_sample_rate) if row["mixer"].export_sample_rate in opts else 2
                        row["mixer"].export_sample_rate = opts[(idx - 1) % len(opts)]
                        self._dirty = True
                        return True
                    if row["sr_r"].collidepoint(lx, ly):
                        opts = self._SR_OPTIONS
                        idx  = opts.index(row["mixer"].export_sample_rate) if row["mixer"].export_sample_rate in opts else 2
                        row["mixer"].export_sample_rate = opts[(idx + 1) % len(opts)]
                        self._dirty = True
                        return True
                    if row["bd_l"].collidepoint(lx, ly):
                        opts = self._BD_OPTIONS
                        idx  = opts.index(row["mixer"].export_bit_depth) if row["mixer"].export_bit_depth in opts else 1
                        row["mixer"].export_bit_depth = opts[(idx - 1) % len(opts)]
                        self._dirty = True
                        return True
                    if row["bd_r"].collidepoint(lx, ly):
                        opts = self._BD_OPTIONS
                        idx  = opts.index(row["mixer"].export_bit_depth) if row["mixer"].export_bit_depth in opts else 1
                        row["mixer"].export_bit_depth = opts[(idx + 1) % len(opts)]
                        self._dirty = True
                        return True
                return False
            cell = self._cell_at(lx, ly)
            if cell is None:
                self._last_click_cell = None
                self._pending_click   = None
                return False
            ri, ci = cell
            now_tick = pygame.time.get_ticks()
            # Second click on same cell within timeout → double-click cycle
            if (cell == self._last_click_cell
                    and (now_tick - self._last_click_tick) < self._DCLICK_MS):
                cur = _read_cell(ri, ci)
                cyc = self._DCLICK_CYCLE
                best_idx = min(range(len(cyc)),
                               key=lambda k: abs(cyc[k] - cur))
                nxt = cyc[(best_idx + 1) % len(cyc)]
                _write_cell(ri, ci, nxt)
                self._active_cell     = cell
                self._drag_cell       = None
                self._pending_click   = None
                self._last_click_cell = None
                self._dirty = True
                return True
            # First click — defer drag, record pending.
            # Return False so the caller does NOT trigger a heavy audio rebuild
            # between the first and second click (which would stall the event
            # loop and make double-click unreliable).
            self._last_click_cell     = cell
            self._last_click_tick     = now_tick
            self._pending_click       = cell
            self._pending_click_tick  = now_tick
            self._pending_click_y0    = event.pos[1]
            self._pending_click_val0  = _read_cell(ri, ci)
            self._active_cell = cell
            self._drag_cell   = None  # drag NOT started yet
            self._dirty = True
            return False

        elif event.type == MOUSEBUTTONUP and event.button == 1:
            # If pending click never became a drag, just snap current value
            if self._pending_click is not None:
                self._pending_click = None
            if self._drag_cell is not None:
                if self._active_cell is not None:
                    ri, ci = self._active_cell
                    held  = pygame.key.get_pressed()
                    shift = held[K_LSHIFT] or held[K_RSHIFT]
                    cur   = _read_cell(ri, ci)
                    _write_cell(ri, ci, _clamp(_snap(cur, shift)))
                self._drag_cell = None
                self._dirty = True
            return False

        elif event.type == MOUSEBUTTONDOWN and event.button == 3:
            lx, ly = _lx_ly(event.pos)
            cell = self._cell_at(lx, ly)
            if cell is not None:
                ri, ci = cell
                _write_cell(ri, ci, 0.0)
                self._active_cell = cell
                self._dirty = True
                return True

        elif event.type == MOUSEMOTION and self._drag_cell is not None:
            held  = pygame.key.get_pressed()
            shift = held[K_LSHIFT] or held[K_RSHIFT]
            dy    = self._drag_y0 - event.pos[1]
            raw   = self._drag_val0 + _delta_from_dy(dy, shift)
            new_v = _clamp(_snap(raw, shift))
            ri, ci = self._drag_cell
            _write_cell(ri, ci, new_v)
            self._dirty = True
            return True

        elif event.type == MOUSEMOTION and self._pending_click is not None:
            # Promote pending click to drag only after double-click window expires
            now_tick = pygame.time.get_ticks()
            if (now_tick - self._pending_click_tick) >= self._DCLICK_MS:
                ri, ci = self._pending_click
                self._drag_cell  = self._pending_click
                self._drag_y0    = self._pending_click_y0
                self._drag_val0  = self._pending_click_val0
                self._pending_click = None
                # Apply the motion that just arrived
                held  = pygame.key.get_pressed()
                shift = held[K_LSHIFT] or held[K_RSHIFT]
                dy    = self._drag_y0 - event.pos[1]
                raw   = self._drag_val0 + _delta_from_dy(dy, shift)
                new_v = _clamp(_snap(raw, shift))
                _write_cell(ri, ci, new_v)
                self._dirty = True
                return True

        elif event.type == pygame.MOUSEWHEEL:
            if self._routing_tab == "export":
                return False
            held  = pygame.key.get_pressed()
            shift = held[K_LSHIFT] or held[K_RSHIFT]
            # scroll up (event.y > 0) → show earlier rows/cols → decrease offset
            if shift:
                self._scroll_col = max(0, min(
                    self._scroll_col - event.y,
                    max(0, self._N - self._vis_cols)))
            else:
                self._scroll_row = max(0, min(
                    self._scroll_row - event.y,
                    max(0, self._N - self._vis_rows)))
            self._dirty = True
            return True

        return False


class EditorCanvas:
    """Manages the center waveform / envelope / chirp editor."""

    MODES = [EditorMode.WAVEFORM, EditorMode.ENVELOPE, EditorMode.PIECEWISE_EDITOR, EditorMode.CHIRP,
             EditorMode.COMPLEX_RI, EditorMode.COMPLEX_MP,
             EditorMode.HARMONICS, EditorMode.LFO_VIEW, EditorMode.FM_VIEW,
             EditorMode.MIX, EditorMode.ROUTING]
    MODE_LABELS = ["Wave", "Envelope", "Piecewise", "Chirp", "Re/Im", "Mag/Phase",
                   "Harmonics", "LFOs", "FM", "Mix", "Routing"]

    # Modes available on the __mix__ node (routing module)
    _MIX_NODE_MODES  = [EditorMode.ROUTING, EditorMode.CONTROL_ROUTING, EditorMode.MIX,
                        EditorMode.COMPLEX_RI, EditorMode.COMPLEX_MP]
    _MIX_NODE_LABELS = ["Routing", "Control", "Mix", "Re/Im", "Mag/Phase"]

    # Modes available on LFO nodes
    _LFO_NODE_MODES   = [EditorMode.WAVEFORM, EditorMode.LFO_VIEW, EditorMode.COMPLEX_RI]
    _PARAM_NODE_MODES  = [EditorMode.WAVEFORM, EditorMode.COMPLEX_RI,
                          EditorMode.COMPLEX_MP, EditorMode.PARAM_ROUTING,
                          EditorMode.CONTROL_ROUTING]
    _PARAM_NODE_LABELS = ["Wave", "Re/Im", "Mag/Phase", "Param Routing", "Control"]
    _LFO_NODE_LABELS = ["Wave", "LFOs", "Re/Im"]
    _STATE_MACHINE_NODE_MODES = [EditorMode.ROUTING, EditorMode.CONTROL_ROUTING, EditorMode.WAVEFORM,
                                 EditorMode.COMPLEX_RI, EditorMode.COMPLEX_MP,
                                 EditorMode.SM_LOG]
    _STATE_MACHINE_NODE_LABELS = ["Routing", "Control", "Wave", "Re/Im", "Mag/Phase", "Log"]

    # Modes available on regular voice nodes — Piecewise editor is permanent first tab
    _VOICE_MODES  = [EditorMode.PIECEWISE_EDITOR,
                     EditorMode.COMPLEX_RI, EditorMode.COMPLEX_MP,
                     EditorMode.HARMONICS, EditorMode.LFO_VIEW, EditorMode.FM_VIEW,
                     EditorMode.MIX]
    _VOICE_LABELS = ["Piecewise", "Re/Im", "Mag/Phase",
                     "Harmonics", "LFOs", "FM", "Mix"]

    # Modes for the __patch__ pseudo-node (patch-global views)
    _PATCH_NODE_MODES  = [EditorMode.WAVEFORM, EditorMode.MIX, EditorMode.SCORE, EditorMode.PIANO_ROLL, EditorMode.PLACEMENT]
    _PATCH_NODE_LABELS = ["Wave", "Mix", "Score", "Roll", "Placement"]
    _SYSTEM_NODE_MODES  = [EditorMode.WAVEFORM, EditorMode.MIX]
    _SYSTEM_NODE_LABELS = ["Wave", "Mix"]

    def _visible_modes(self, patch: "AnalyticPatch") -> tuple[list, list]:
        """Return (modes, labels) appropriate for the currently active node."""
        if self.active_key == "__patch__":
            modes, labels = list(self._PATCH_NODE_MODES), list(self._PATCH_NODE_LABELS)
            # Hide Placement tab when placement resonator is disabled
            res_cfg = getattr(patch, "placement_resonator", None)
            if res_cfg is None or not res_cfg.enabled:
                pairs = [(m, l) for m, l in zip(modes, labels) if m != EditorMode.PLACEMENT]
                modes = [p[0] for p in pairs]
                labels = [p[1] for p in pairs]
            return modes, labels
        if self.active_key == "__system__":
            return self._SYSTEM_NODE_MODES, self._SYSTEM_NODE_LABELS
        if any(m.key == self.active_key for m in patch.mixers):
            return self._MIX_NODE_MODES, self._MIX_NODE_LABELS
        if _router_instance_for_active_key(patch, self.active_key) is not None:
            return self._MIX_NODE_MODES, self._MIX_NODE_LABELS
        if any(l.key == self.active_key for l in patch.lfos):
            return self._LFO_NODE_MODES, self._LFO_NODE_LABELS
        if any(pn.key == self.active_key for pn in patch.param_nodes):
            return self._PARAM_NODE_MODES, self._PARAM_NODE_LABELS
        mod = next((m for m in patch.modules if m.key == self.active_key), None)
        if mod is not None and mod.module_type == "state_machine":
            return self._STATE_MACHINE_NODE_MODES, self._STATE_MACHINE_NODE_LABELS
        return self._VOICE_MODES, self._VOICE_LABELS

    def __init__(self) -> None:
        self.mode: EditorMode = EditorMode.WAVEFORM
        self.active_key: str = ""
        self.plot   = PlotWidget()
        self.plot.grid_lines = 5
        self.routing_view = RoutingGridView()
        self.routing_view.active_key = self.active_key
        self._canvas_font: pygame.font.Font | None = None
        # Mix ctrl drag state for rotation knob
        self._rot_drag_x0:   int   = 0
        self._rot_drag_val0: float = 0.0
        self._rot_dragging:  bool  = False

        # Cached signal & envelope
        self._signal:       np.ndarray | None = None
        self._env_curve:    np.ndarray | None = None
        self._chirp_t:      np.ndarray | None = None
        self._chirp_f:      np.ndarray | None = None
        # Complex analytic signal (full, before projection)
        self._complex_sig:  np.ndarray | None = None
        # Cumulative phase in cycles at each sample (from chirp f_inst, no FM)
        self._phase_cycles: np.ndarray | None = None
        # Full routing solver output for tail-view (mixer node only)
        self._tail_X:         np.ndarray | None = None
        self._tail_node_keys: list | None       = None
        self._tail_note_n:    int               = 0
        self._sm_log_text:    str               = ""
        self._sm_log_scroll:  int               = 0
        # Tail view toggle — when True show the full ringdown tail in plots
        self.show_tail: bool = False

        # Time/value view range
        self.t_lo: float = 0.0
        self.t_hi: float = 1.0   # normalized [0, 1]
        self.v_lo: float = -1.0
        self.v_hi: float =  1.0

        # Cursor position (normalized time [0, 1])
        self.cursor_t: float = 0.0

        # Drag state
        self.drag = DragState()

        # GL texture for the center surface
        self._surf: pygame.Surface | None = None
        self._tex:  int = 0
        self._surf_dirty: bool = True
        self._piecewise_atom_tex: dict[str, int] = {}
        self._piecewise_atoms: list[Any] = []
        self._piecewise_stale_tex_ids: list[int] = []
        self._piecewise_preview_dirty: bool = True
        self._piecewise_sync_revision: int = -1
        self._piecewise_snapshot_revision: int = -1
        self._piecewise_snapshot: dict[str, Any] | None = None
        self._piecewise_preview_lock = threading.Lock()
        self._piecewise_preview_request_key: str = ""
        self._piecewise_preview_ready_key: str = ""
        self._piecewise_preview_data: dict[str, Any] | None = None
        self._piecewise_preview_cancel = threading.Event()
        self._piecewise_preview_thread: threading.Thread | None = None
        self._piecewise_preview_debounce: threading.Timer | None = None

        # Cache of active voice key (drives overlay rendering)
        # (initialized at top of __init__ before routing_view)

        # Seed offset for granular voices — advanced by viewer when seed_animate is on
        self.granular_seed_offset: int = 0

        # ── Placement viewer state ───────────────────────────────────────────
        self._placement_perf_rects: list = []   # [{rect, pf_key, part_key}, ...]
        self._placement_hover_key: str = ""
        self._placement_zoom: float = 1.0
        # Control routing rack view state
        self._control_device_rects: list = []
        self._control_port_rects: dict[str, dict] = {}
        self._control_edge_hits: list = []
        self._control_armed_port: str = ""
        self._control_hover_port: str = ""

        # Mix-tab per-series visibility toggles and signal cache
        self._mix_series_hidden: set[str] = set()
        self._mix_toggle_rects:  list[dict] = []
        self._mix_series_info:   list[dict] = []
        self._mix_voice_cache:   dict[str, np.ndarray] = {}
        self._mix_left_cache:    "np.ndarray | None" = None
        self._mix_right_cache:   "np.ndarray | None" = None
        self._mix_t_ax:          "np.ndarray | None" = None
        self._score_scroll_y:    int = 0
        self._score_max_scroll:  int = 0
        self._piano_note_rects:  list[dict] = []
        self._piano_ctrl_rects:  dict[str, pygame.Rect] = {}
        self._piano_selected_note_id: str = ""
        self._piano_dragging: bool = False
        self._piano_drag_kind: str = ""
        self._piano_drag_note_id: str = ""
        self._piano_drag_origin: dict = {}
        self._piano_pitch_snap: bool = True
        self._piano_time_snap_idx: int = 2
        self._piano_len_snap_idx: int = 2
        self._piano_rest_mode: bool = False

        # Hover state
        self._hover_cp: int = -1   # index of hovered knot (-1 = none)

        # Background rebuild infrastructure
        self._rebuild_thread: threading.Thread | None = None
        self._rebuild_cancel: threading.Event = threading.Event()
        self._rebuild_debounce: threading.Timer | None = None
        self._rebuild_result: dict | None = None
        self._rebuild_lock = threading.Lock()
        self._rebuild_hash: str = ""  # fingerprint of last completed rebuild
        self._piecewise_editor: ParametricCurveEditor | None = None
        self._piecewise_voice_key: str = ""

    # ---- Coordinate mapping ------------------------------------------------

    def _font_(self) -> pygame.font.Font:
        if self._canvas_font is None:
            pygame.font.init()
            self._canvas_font = pygame.font.SysFont("consolas", 11)
        return self._canvas_font

    def _canvas_rect(self, win_w: int, win_h: int) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) of the whole center canvas in window pixels."""
        x = PANEL_W
        y = TOPBAR_H
        w = win_w - 2 * PANEL_W
        h = win_h - TOPBAR_H - BOTTOM_H
        return x, y, w, h

    def _inner_rect(self, win_w: int, win_h: int) -> tuple[int, int, int, int]:
        """Inner rect of the PlotWidget (data plot area)."""
        cx, cy, cw, ch = self._canvas_rect(win_w, win_h)
        # subtract the mode-tab bar from the top
        cy2 = cy + MODEBAR_H
        ch2 = ch - MODEBAR_H
        ix = cx + _PW_ML
        iy = cy2 + _PW_MT
        iw = cw - _PW_ML - _PW_MR
        ih = ch2 - _PW_MT - _PW_MB
        return ix, iy, iw, ih

    def data_to_px(self, t: float, v: float,
                   win_w: int, win_h: int) -> tuple[float, float]:
        """Map normalized (t∈[t_lo,t_hi], v∈[v_lo,v_hi]) → pixel (px, py)."""
        ix, iy, iw, ih = self._inner_rect(win_w, win_h)
        px = ix + (t - self.t_lo) / max(self.t_hi - self.t_lo, 1e-9) * iw
        py = iy + ih - (v - self.v_lo) / max(self.v_hi - self.v_lo, 1e-9) * ih
        return px, py

    def px_to_data(self, px: float, py: float,
                   win_w: int, win_h: int) -> tuple[float, float]:
        """Map pixel → normalized (t, v)."""
        ix, iy, iw, ih = self._inner_rect(win_w, win_h)
        t = self.t_lo + (px - ix) / max(iw, 1) * (self.t_hi - self.t_lo)
        v = self.v_lo + (1.0 - (py - iy) / max(ih, 1)) * (self.v_hi - self.v_lo)
        return t, v

    def _mode_tab_rects(self, win_w: int, win_h: int, patch: "AnalyticPatch | None" = None) -> list[pygame.Rect]:
        cx, cy, cw, _ = self._canvas_rect(win_w, win_h)
        modes, _ = self._visible_modes(patch) if patch is not None else (self.MODES, self.MODE_LABELS)
        _TAIL_BTN_W = 44  # reserved for tail-toggle button at right of mode bar
        tab_area_w = cw - _TAIL_BTN_W
        tab_w = tab_area_w // max(len(modes), 1)
        return [pygame.Rect(cx + i * tab_w, cy, tab_w, MODEBAR_H)
                for i in range(len(modes))]

    def _set_plot_y_range_from_arrays(self, *arrays: np.ndarray,
                                      symmetric: bool = True,
                                      min_half_span: float = 1e-3) -> None:
        """Autoscale the plot to contain the given series with a small margin."""
        vals: list[np.ndarray] = []
        for arr in arrays:
            if arr is None:
                continue
            a = np.asarray(arr, dtype=np.float64)
            if a.size == 0:
                continue
            vals.append(a)
        if not vals:
            self.plot.y_min, self.plot.y_max = -1.05, 1.05
            self.v_lo, self.v_hi = -1.05, 1.05
            return
        merged = np.concatenate(vals)
        lo = float(np.min(merged))
        hi = float(np.max(merged))
        if symmetric:
            peak = max(abs(lo), abs(hi), min_half_span)
            pad = max(peak * 0.08, min_half_span * 0.5)
            self.v_lo, self.v_hi = -(peak + pad), (peak + pad)
        else:
            span = max(hi - lo, min_half_span)
            pad = max(span * 0.08, min_half_span * 0.5)
            self.v_lo, self.v_hi = lo - pad, hi + pad
        self.plot.y_min, self.plot.y_max = self.v_lo, self.v_hi

    # ---- Data refresh ------------------------------------------------------

    @staticmethod
    def _patch_fingerprint(patch: "AnalyticPatch", active_key: str,
                           mode, show_tail: bool, seed: int) -> str:
        import hashlib
        blob = json.dumps(patch.to_dict(), sort_keys=True, default=str)
        blob += f"|{active_key}|{mode}|{show_tail}|{seed}"
        return hashlib.sha256(blob.encode()).hexdigest()

    def _ensure_piecewise_editor(self, voice: "AnalyticVoice", win_w: int, win_h: int) -> ParametricCurveEditor:
        piecewise = voice.piecewise_env or PiecewiseVoiceEnvelope()
        voice.piecewise_env = piecewise
        cx, cy, cw, ch = self._canvas_rect(win_w, win_h)
        plot_y = cy + MODEBAR_H
        plot_h = ch - MODEBAR_H
        if self._piecewise_editor is None or self._piecewise_voice_key != voice.key:
            if self._piecewise_atom_tex:
                self._piecewise_stale_tex_ids.extend(self._piecewise_atom_tex.values())
                self._piecewise_atom_tex.clear()
                self._piecewise_atoms = []
            editor = ParametricCurveEditor(
                piecewise.curve,
                piecewise.chirp_curve,
                piecewise.signal_curve,
                w=max(64, cw),
                h=max(64, plot_h),
                library_folder=os.path.join(os.getcwd(), "envelopes"),
            )
            editor._overlay = _TextOverlay(editor.w, editor.h)
            editor.set_async_invalidate_callback(
                lambda: (
                    setattr(self, "_piecewise_preview_dirty", True),
                    setattr(self, "_surf_dirty", True),
                )
            )
            editor._channels["A"] = editor._channel_state("A")
            editor._channels["A"].rule_tree = piecewise.rule_tree
            self._piecewise_editor = editor
            self._piecewise_voice_key = voice.key
            self._piecewise_sync_revision = -1
            self._piecewise_preview_dirty = True
            self._surf_dirty = True
        else:
            if self._piecewise_editor.w != max(64, cw) or self._piecewise_editor.h != max(64, plot_h):
                self._piecewise_editor.resize(max(64, cw), max(64, plot_h))
                self._surf_dirty = True
        return self._piecewise_editor

    def _pull_piecewise_editor_state_into_voice(self, voice: "AnalyticVoice") -> None:
        if self._piecewise_editor is None:
            return
        if voice.piecewise_env is None:
            voice.piecewise_env = PiecewiseVoiceEnvelope()
        snap = self._piecewise_editor.snapshot_state()
        voice.piecewise_env.curve = snap["curves"][0]
        voice.piecewise_env.chirp_curve = snap["curves"][1]
        voice.piecewise_env.signal_curve = snap["curves"][2]
        voice.piecewise_env.rule_tree = snap["rule_tree"] or EnvelopeRuleTree.default()
        self._piecewise_sync_revision = int(snap["revision"])

    def _piecewise_preview_key(self, patch: "AnalyticPatch", voice: "AnalyticVoice",
                               editor: ParametricCurveEditor) -> tuple[str, dict[str, Any]]:
        editor_rev = editor.get_revision()
        if self._piecewise_snapshot is None or editor_rev != self._piecewise_snapshot_revision:
            self._piecewise_snapshot = editor.snapshot_state()
            self._piecewise_snapshot_revision = editor_rev
        snap = self._piecewise_snapshot
        payload = {
            "voice_key": voice.key,
            "voice": voice.to_dict(),
            "duration": float(patch.duration),
            "preview_sr": int(patch.preview_sr),
            "granular_seed_offset": int(self.granular_seed_offset),
            "lfos": [l.to_dict() for l in patch.lfos],
            "voices": [v.to_dict() for v in patch.voices],
            "piecewise_curves": [c.to_dict() for c in snap["curves"]],
            "rule_tree": snap["rule_tree"].to_dict(),
            "revision": int(snap["revision"]),
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return key, snap

    def _request_piecewise_preview(self, patch: "AnalyticPatch", voice: "AnalyticVoice",
                                   editor: ParametricCurveEditor) -> None:
        req_key, snap = self._piecewise_preview_key(patch, voice, editor)
        with self._piecewise_preview_lock:
            if req_key == self._piecewise_preview_request_key:
                return
            self._piecewise_preview_request_key = req_key
            self._piecewise_preview_cancel.set()
            self._piecewise_preview_cancel = threading.Event()
            cancel = self._piecewise_preview_cancel
        patch_snap = copy.deepcopy(patch)
        voice_snap = next((v for v in patch_snap.voices if v.key == voice.key), None)
        if voice_snap is None:
            return
        voice_snap.piecewise_env = PiecewiseVoiceEnvelope(
            curve=snap["curves"][0],
            chirp_curve=snap["curves"][1],
            signal_curve=snap["curves"][2],
            rule_tree=snap["rule_tree"],
        )

        def _worker() -> None:
            try:
                if cancel.is_set():
                    return
                lfo_map = {l.key: l for l in patch_snap.lfos}
                p_map = {p.key: p for p in patch_snap.voices}
                if voice_snap and not _voice_effectively_muted(voice_snap, patch_snap):
                    csig = _synthesize_voice(
                        voice_snap,
                        patch_snap,
                        lfo_map,
                        p_map,
                        granular_seed_offset=self.granular_seed_offset,
                    )
                    if hasattr(csig, "detach"):
                        csig = csig.detach().cpu().numpy()
                    csig = np.asarray(csig, dtype=np.complex128)
                    env_curve = _compute_envelope(voice_snap, len(csig), patch_snap.duration).astype(np.float32, copy=False)
                    chirp_f = _compute_chirp_frequency_series(voice_snap, len(csig), patch_snap.duration).astype(np.float32, copy=False)
                else:
                    csig = np.zeros(0, dtype=np.complex128)
                    env_curve = np.zeros(0, dtype=np.float32)
                    chirp_f = np.zeros(0, dtype=np.float32)

                # Batch synthesis for GL display
                _N_PHASE = 72
                if (len(csig) > 0
                        and voice_snap.piecewise_env is None
                        and voice_snap.emission_mode not in ("granular",)):
                    if cancel.is_set():
                        return
                    _phase_origins = torch.linspace(
                        0.0, 2.0 * math.pi, _N_PHASE, endpoint=False, dtype=torch.float64
                    )
                    csig_batch = _synthesize_voice_batched(
                        voice_snap, patch_snap, lfo_map, p_map, _phase_origins,
                        granular_seed_offset=self.granular_seed_offset,
                    ).detach().cpu().numpy()  # (72, n) complex128
                elif len(csig) > 0:
                    # Piecewise/granular: phase_origin doesn't affect output; rotation is exact
                    _angles    = np.linspace(0.0, 2.0 * np.pi, _N_PHASE, endpoint=False)
                    csig_batch = csig[np.newaxis, :] * np.exp(1j * _angles)[:, np.newaxis]
                else:
                    csig_batch = np.zeros((_N_PHASE, 0), dtype=np.complex128)

                raw: dict[str, torch.Tensor] = {}
                if len(csig) > 0:
                    n_sig = len(csig)
                    env_arr = env_curve.astype(float) if env_curve is not None else np.zeros(n_sig)
                    amp_total = np.abs(np.asarray(csig, dtype=np.complex128))
                    amp_peak = float(np.max(amp_total)) if amp_total.size > 0 else 0.0
                    amp_total = amp_total / amp_peak if amp_peak > 1e-9 else np.zeros_like(amp_total, dtype=np.float64)
                    base_chirp = _compute_knob_chirp_deviation_series(voice_snap, n_sig, patch_snap.duration)
                    chirp_curve = snap["curves"][1]
                    t_norm = torch.linspace(0.0, 1.0, n_sig, dtype=torch.float64)
                    chirp_env_curve = chirp_curve.evaluate_normalized(t_norm).real.clamp(0.0, 1.0)
                    chirp_env_hz = chirp_curve.to_physical(chirp_env_curve).detach().cpu().numpy().astype(np.float64, copy=False)
                    chirp_total_hz = base_chirp + chirp_env_hz
                    v_lo = float(chirp_curve.v_lo)
                    v_hi = float(chirp_curve.v_hi)
                    span = max(v_hi - v_lo, 1e-9)
                    chirp_bias_norm = np.clip((base_chirp - v_lo) / span, 0.0, 1.0)
                    chirp_env_norm = np.clip((chirp_env_hz - v_lo) / span, 0.0, 1.0)
                    chirp_total_norm = np.clip((chirp_total_hz - v_lo) / span, 0.0, 1.0)
                    raw = {
                        "analytic": torch.as_tensor(csig, dtype=torch.complex128).reshape(-1),
                        "amplitude_bias": torch.zeros(n_sig, dtype=torch.complex128),
                        "amplitude": torch.as_tensor(env_arr, dtype=torch.complex128).reshape(-1),
                        "amplitude_total": torch.as_tensor(amp_total, dtype=torch.complex128).reshape(-1),
                        "chirp_bias": torch.as_tensor(chirp_bias_norm, dtype=torch.complex128).reshape(-1),
                        "chirp": torch.as_tensor(chirp_env_norm, dtype=torch.complex128).reshape(-1),
                        "chirp_total": torch.as_tensor(chirp_total_norm, dtype=torch.complex128).reshape(-1),
                    }
                data = {
                    "csig":       csig,
                    "csig_batch": csig_batch,
                    "env_curve":  env_curve,
                    "chirp_f":    chirp_f,
                    "raw":        raw,
                }
                if cancel.is_set():
                    return
                with self._piecewise_preview_lock:
                    if req_key != self._piecewise_preview_request_key:
                        return
                    self._piecewise_preview_ready_key = req_key
                    self._piecewise_preview_data = data
                self._surf_dirty = True
            except Exception:
                pass

        with self._piecewise_preview_lock:
            if self._piecewise_preview_debounce is not None:
                self._piecewise_preview_debounce.cancel()
            def _launch():
                self._piecewise_preview_thread = threading.Thread(target=_worker, daemon=True)
                self._piecewise_preview_thread.start()
            t = threading.Timer(0.25, _launch)
            self._piecewise_preview_debounce = t
        t.start()

    def _apply_piecewise_preview_if_ready(self, editor: ParametricCurveEditor) -> None:
        with self._piecewise_preview_lock:
            if (self._piecewise_preview_ready_key != self._piecewise_preview_request_key
                    or self._piecewise_preview_data is None):
                return
            data = self._piecewise_preview_data
        self._complex_sig = data["csig"]
        self._signal = self._complex_sig.real.astype(np.float32, copy=False) if len(self._complex_sig) else np.zeros(0, dtype=np.float32)
        self._env_curve = data["env_curve"]
        self._chirp_f = data["chirp_f"]
        raw = data["raw"]
        if raw:
            state_a = editor._channel_state("A")
            state_a.raw_signals = {
                "amplitude_bias": raw["amplitude_bias"],
                "amplitude": raw["amplitude"],
                "amplitude_total": raw["amplitude_total"],
                "chirp_bias": raw["chirp_bias"],
                "chirp": raw["chirp"],
                "chirp_total": raw["chirp_total"],
            }
            state_a.display_signals = _norm_ch_sigs(state_a.raw_signals, time_stretch=state_a.time_stretch)
            state_b = editor._channel_state("B")
            state_b.raw_signals = {"analytic": raw["analytic"]}
            state_b.display_signals = _norm_ch_sigs(state_b.raw_signals, time_stretch=state_b.time_stretch)
            editor._render_buf = state_b.display_signals.get("analytic")
            editor._refresh_render_points("A")
            # Push batch before refreshing B so _refresh_render_points can use it
            csig_batch = data.get("csig_batch")
            if csig_batch is not None and csig_batch.shape[1] > 0:
                editor._csig_batch    = torch.as_tensor(csig_batch, dtype=torch.complex128)
                editor._csig_batch_np = csig_batch  # stable numpy ref for lazy widget creation
                if editor._gl_wave_widget is not None:
                    editor._gl_wave_widget.update_data(csig_batch)
            else:
                editor._csig_batch    = None
                editor._csig_batch_np = None
            editor._refresh_render_points("B")
            # Consume the data — prevent re-applying (and re-running torch ops) every frame
            with self._piecewise_preview_lock:
                self._piecewise_preview_data = None

    def rebuild(self, patch: AnalyticPatch, active_key: str) -> None:
        """Kick off a background rebuild.  Cancels any in-flight rebuild first.
        Skips entirely if the patch state hasn't changed since the last completed
        rebuild (fingerprint match)."""
        import copy as _rcopy

        snap_mode = self.mode
        snap_show_tail = self.show_tail
        snap_seed = self.granular_seed_offset

        # Fingerprint — skip if nothing changed
        fp = self._patch_fingerprint(patch, active_key, snap_mode,
                                     snap_show_tail, snap_seed)
        if fp == self._rebuild_hash:
            return

        # Cancel any running rebuild and any pending debounce
        self._rebuild_cancel.set()
        if self._rebuild_debounce is not None:
            self._rebuild_debounce.cancel()

        def _launch():
            # Re-check fingerprint — something may have changed during the delay
            cancel = threading.Event()
            self._rebuild_cancel = cancel
            snap = _rcopy.deepcopy(patch)

            def _worker():
                try:
                    self._rebuild_work(snap, active_key, snap_mode, snap_show_tail,
                                       snap_seed, cancel, fp)
                except Exception:
                    pass

            t = threading.Thread(target=_worker, daemon=True)
            self._rebuild_thread = t
            t.start()

        db = threading.Timer(0.25, _launch)
        self._rebuild_debounce = db
        db.start()

    def rebuild_ready(self) -> bool:
        """Return True when the background rebuild thread has finished."""
        t = self._rebuild_thread
        return t is not None and not t.is_alive()

    def _rebuild_work(self, patch: AnalyticPatch, active_key: str,
                      mode, show_tail: bool, granular_seed_offset: int,
                      cancel: threading.Event, fingerprint: str) -> None:
        """Heavy compute — runs on a background thread.
        Checks *cancel* periodically; if set, bails out without writing results."""
        if cancel.is_set():
            return
        self.active_key = active_key
        self.mode = mode
        self.show_tail = show_tail
        self.granular_seed_offset = granular_seed_offset
        dur = patch.duration
        sr  = patch.preview_sr
        n   = int(sr * dur)
        t_ax = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)

        # --- find active voice ---
        voice = next((p for p in patch.voices if p.key == active_key), None)
        is_lfo_node = any(l.key == active_key for l in patch.lfos)
        is_param_node = any(pn.key == active_key for pn in patch.param_nodes)
        active_module = next((m for m in patch.modules if m.key == active_key), None)
        is_module_node = active_module is not None
        is_control_node = any(
            sl.key == active_key
            for cs in patch.controls
            for sl in cs.sliders
        )
        self._sm_log_text = ""

        # --- special case: mixer node selected ---
        is_mixer_node = any(m.key == active_key for m in patch.mixers)
        if is_mixer_node:
            # Synthesize routing system and expose the raw complex mix signal
            lfo_map = {l.key: l for l in patch.lfos}
            p_map   = {v.key: v for v in patch.voices}
            _g = _working_routing_graph_for_synthesis(patch)
            _node_keys = _patch_node_keys(patch)
            _ki = {k: i for i, k in enumerate(_node_keys)}
            _N = len(_node_keys)
            _Src = np.zeros((_N, n), dtype=np.complex128)
            for v in patch.voices:
                if cancel.is_set(): return
                if not _voice_effectively_muted(v, patch) and v.key in _ki:
                    _Src[_ki[v.key]] = _synthesize_voice(v, patch, lfo_map, p_map,
                                                         granular_seed_offset=self.granular_seed_offset)
            for l in patch.lfos:
                if l.key in _ki:
                    _Src[_ki[l.key]] = _synthesize_lfo_csig(l, n, sr)
            if cancel.is_set(): return
            _fb = _g.feedback
            _gd = max(0.0, 1.0 - float(_fb.decay)) if _fb.enabled else 1.0
            _X, _note_n = solve_routing_with_ringdown(_Src, _g.edges, _node_keys, sr, _gd, _fb)
            if cancel.is_set(): return
            # Store full tail buffer for optional tail-view; _note_n is the note boundary
            self._tail_X         = _X
            self._tail_node_keys = _node_keys
            self._tail_note_n    = _note_n
            _n_disp = _X.shape[1] if self.show_tail else n
            _mix_csig = _X[_ki[active_key]][:_n_disp]
            self._complex_sig = _mix_csig
            sig = _mix_csig.real.astype(np.float32)
            self._signal = sig
            self._env_curve = np.zeros(_n_disp, dtype=np.float32)
            self._chirp_f   = np.zeros(_n_disp, dtype=np.float32)
            t_ax = np.linspace(0.0, 1.0, _n_disp, endpoint=False, dtype=np.float32)
            self._chirp_t   = t_ax
            self._phase_cycles = np.zeros(_n_disp, dtype=np.float64)
            # Colour from mixer definition
            _mobj = next((m for m in patch.mixers if m.key == active_key), None)
            col = tuple((_mobj.color[:3]) if _mobj else (200, 200, 100))
            # For ROUTING mode, nothing to plot — routing_view handles it
            if self.mode not in (EditorMode.COMPLEX_RI, EditorMode.COMPLEX_MP,
                                  EditorMode.MIX, EditorMode.WAVEFORM):
                self.mode = EditorMode.ROUTING
            # Let mode-specific plotting below handle COMPLEX_RI / COMPLEX_MP / MIX
            # but skip voice-specific blocks — fall through with voice=None, sig set
        elif is_lfo_node or is_param_node or is_module_node or is_control_node:
            left_out, right_out, _sc = _synthesize_patch(
                patch,
                granular_seed_offset=self.granular_seed_offset,
                _return_sidecar=True,
            )
            if cancel.is_set(): return
            self._last_sidecar = _sc
            routed = _sc.get(active_key, "routed", n)
            if routed is None:
                routed = _sc.get(active_key, "value", n)
                if routed is not None:
                    routed = np.asarray(routed, dtype=np.float64).astype(np.complex128)
            if routed is None:
                routed = np.zeros(n, dtype=np.complex128)
            else:
                routed = np.asarray(routed, dtype=np.complex128)
            self._complex_sig = routed
            sig = routed.real.astype(np.float32)
            self._signal = sig
            self._env_curve = np.zeros(len(routed), dtype=np.float32)
            self._chirp_f = np.zeros(len(routed), dtype=np.float32)
            t_ax = np.linspace(0.0, 1.0, len(routed), endpoint=False, dtype=np.float32)
            self._chirp_t = t_ax
            self._phase_cycles = np.zeros(len(routed), dtype=np.float64)
            if active_module is not None and active_module.module_type == "state_machine":
                self._sm_log_text = str(getattr(active_module, "_sm_log_text", "") or "")
                self._sm_log_scroll = max(0, min(self._sm_log_scroll, max(0, len(self._sm_log_text.splitlines()) - 1)))
        else:
            # --- waveform (complex analytic) ---
            lfo_map = {l.key: l for l in patch.lfos}
            p_map   = {p.key: p for p in patch.voices}
            if cancel.is_set(): return
            if voice and not _voice_effectively_muted(voice, patch):
                csig = _synthesize_voice(voice, patch, lfo_map, p_map,
                                         granular_seed_offset=self.granular_seed_offset)
                if cancel.is_set(): return
                if hasattr(csig, 'detach'):
                    csig = csig.detach().cpu().numpy()
                csig = np.asarray(csig, dtype=np.complex128)
                sig  = csig.real.astype(np.float32)
                self._complex_sig = csig
            else:
                sig = np.zeros(n, dtype=np.float32)
                self._complex_sig = np.zeros(n, dtype=np.complex128)
            self._signal = sig

            # --- envelope ---
            if voice:
                env = _compute_envelope(voice, n, dur).astype(np.float32)
            else:
                env = np.zeros(n, dtype=np.float32)
            self._env_curve = env

        # lfo_map / p_map for mode-specific plotting below (may not be set if __mix__ path taken)
        lfo_map = {l.key: l for l in patch.lfos}
        p_map   = {v.key: v for v in patch.voices}

        # --- chirp frequency evolution ---
        if voice:
            self._chirp_f = _compute_chirp_frequency_series(voice, n, dur).astype(np.float32, copy=False)
        else:
            self._chirp_f = np.zeros(n, dtype=np.float32)
        self._chirp_t = t_ax

        # --- phase cycles (cumulative cycles from instantaneous chirp freq) ---
        sr_f = float(patch.preview_sr)
        self._phase_cycles = np.cumsum(
            self._chirp_f.astype(np.float64) / sr_f
        )

        # --- configure PlotWidget ---
        if active_key == "__mix__":
            col = (200, 200, 100)
            series_label = "Mix"
            env = self._env_curve  # zeros — ENVELOPE mode not valid for __mix__
        else:
            col = tuple(voice.color[:3]) if voice else _routing_node_color(active_key, patch)
            series_label = voice.label if voice else _routing_node_label(active_key, patch)
            env = self._env_curve

        self.plot.series.clear()
        self.plot.markers.clear()

        if self.mode == EditorMode.WAVEFORM:
            self.plot.add_series(PlotSeries(
                key="wave", label=series_label,
                color=col, line=True, dots=False,
                data_x=t_ax, data_y=sig,
            ))
            self._set_plot_y_range_from_arrays(sig)

        elif self.mode == EditorMode.ENVELOPE:
            self.plot.add_series(PlotSeries(
                key="env", label="Envelope",
                color=(220, 100, 50), line=True, dots=False,
                data_x=t_ax, data_y=env,
            ))
            self.plot.y_min, self.plot.y_max = -0.05, 1.1
            self.v_lo, self.v_hi = -0.05, 1.1

        elif self.mode == EditorMode.CHIRP:
            if len(self._chirp_f) > 0:
                cf_min = float(self._chirp_f.min())
                cf_max = float(self._chirp_f.max())
                margin = max((cf_max - cf_min) * 0.15, 1.0)
                self.v_lo = cf_min - margin
                self.v_hi = cf_max + margin
                self.plot.y_min = self.v_lo
                self.plot.y_max = self.v_hi
            self.plot.add_series(PlotSeries(
                key="chirp", label="Freq (Hz)",
                color=(178, 80, 255), line=True, dots=False,
                data_x=t_ax, data_y=self._chirp_f,
            ))

        elif self.mode == EditorMode.COMPLEX_RI:
            # Orthogonal real and imaginary projections of the analytic signal
            csig = self._complex_sig
            re_data = csig.real.astype(np.float32) if csig is not None else sig
            im_data = csig.imag.astype(np.float32) if csig is not None else np.zeros_like(sig)
            self._set_plot_y_range_from_arrays(re_data, im_data)
            self.plot.add_series(PlotSeries(
                key="re", label="Re",
                color=col, line=True, dots=False,
                data_x=t_ax, data_y=re_data,
            ))
            self.plot.add_series(PlotSeries(
                key="im", label="Im",
                color=(80, 200, 160), line=True, dots=False,
                data_x=t_ax, data_y=im_data,
            ))

        else:  # COMPLEX_MP
            # Magnitude envelope and unwrapped phase
            csig = self._complex_sig
            _n_cs = len(t_ax)  # match t_ax length (may include ringdown tail)
            if csig is not None and np.any(csig != 0):
                mag   = np.abs(csig).astype(np.float32)
                # Normalised instantaneous phase in [-π, π], downsampled for clarity
                ph_raw = np.angle(csig).astype(np.float32)
            else:
                mag   = np.zeros(_n_cs, dtype=np.float32)
                ph_raw = np.zeros(_n_cs, dtype=np.float32)
            # Ensure both arrays match t_ax length
            if len(mag) > _n_cs:
                mag, ph_raw = mag[:_n_cs], ph_raw[:_n_cs]
            elif len(mag) < _n_cs:
                mag    = np.pad(mag,    (0, _n_cs - len(mag)))
                ph_raw = np.pad(ph_raw, (0, _n_cs - len(ph_raw)))
            mag_max = float(mag.max()) if mag.max() > 1e-12 else 1.0
            mag_n = mag / mag_max  # normalise to [0, 1]
            # Scale phase to [-1, 1] for shared y-axis
            ph_n  = ph_raw / math.pi
            self.plot.y_min, self.plot.y_max = -1.05, 1.05
            self.v_lo, self.v_hi = -1.05, 1.05
            self.plot.add_series(PlotSeries(
                key="mag", label="|z| (norm)",
                color=(220, 180, 60), line=True, dots=False,
                data_x=t_ax, data_y=mag_n,
            ))
            self.plot.add_series(PlotSeries(
                key="phase", label="∠z / π",
                color=(140, 100, 240), line=True, dots=False,
                data_x=t_ax, data_y=ph_n,
            ))

        if self.mode == EditorMode.HARMONICS:
            # Bar chart: harmonic k → amplitude weight 1/k^brightness
            self.plot.series.clear()
            if voice and voice.manifold_type in ("harmonic", "harmonic_warp"):
                hc  = max(1, voice.harmonic_count)
                bri = voice.harmonic_brightness
                warp = voice.harmonic_warp_strength
            else:
                hc, bri, warp = 1, 1.0, 0.0
            ks    = np.arange(1, hc + 1, dtype=np.float32)
            amps  = np.array([1.0 / (k ** bri) if bri > 0 else 1.0 for k in ks],
                             dtype=np.float32)
            freqs = np.array([(voice.freq_hz if voice else 440.0)
                              * (k + warp * (k - 1)) for k in ks], dtype=np.float32)
            # Interleave x-values to draw vertical bars: [f, f, f+ε, f+ε] per bar
            bar_x, bar_y = [], []
            f_step = (freqs[1] - freqs[0]) * 0.4 if len(freqs) > 1 else freqs[0] * 0.1
            for fk, ak in zip(freqs, amps):
                bar_x += [fk, fk, fk + f_step, fk + f_step]
                bar_y += [0.0, float(ak), float(ak), 0.0]
            bar_x_arr = np.array(bar_x, dtype=np.float32)
            bar_y_arr = np.array(bar_y, dtype=np.float32)
            self.plot.add_series(PlotSeries(
                key="harmonics", label="Harmonic amp",
                color=col, line=True, dots=False,
                data_x=bar_x_arr / max(float(freqs.max()), 1.0),  # normalise x to [0,1]
                data_y=bar_y_arr,
            ))
            self.plot.y_min, self.plot.y_max = -0.05, 1.1
            self.v_lo, self.v_hi = -0.05, 1.1

        elif self.mode == EditorMode.LFO_VIEW:
            self.plot.series.clear()
            _COLORS = [(220, 160, 60), (80, 200, 160), (180, 80, 255),
                       (255, 130, 60), (120, 220, 120)]
            for li, lfo in enumerate(patch.lfos):
                lc  = tuple(lfo.color[:3]) if len(lfo.color) >= 3 else _COLORS[li % len(_COLORS)]
                t_s = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
                ldata = _lfo_signal(lfo, t_s).astype(np.float32)
                self.plot.add_series(PlotSeries(
                    key=f"lfo_{lfo.key}", label=lfo.label,
                    color=lc, line=True, dots=False,
                    data_x=t_ax, data_y=ldata,
                ))
            if not patch.lfos:
                dummy = np.zeros(n, dtype=np.float32)
                self.plot.add_series(PlotSeries(
                    key="lfo_none", label="(no LFOs)",
                    color=(60, 60, 80), line=True, dots=False,
                    data_x=t_ax, data_y=dummy,
                ))
            self.plot.y_min, self.plot.y_max = -1.1, 1.1
            self.v_lo, self.v_hi = -1.1, 1.1

        elif self.mode == EditorMode.FM_VIEW:
            # Show full instantaneous frequency (chirp + FM deviation) of active voice
            self.plot.series.clear()
            if voice and not voice.muted:
                t_s   = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
                f_full = np.full(n, voice.freq_hz, dtype=np.float64)
                ct2 = voice.chirp.chirp_type
                if ct2 == "linear":
                    f_full += np.linspace(voice.chirp.f_delta_start,
                                          voice.chirp.f_delta_end, n)
                elif ct2 == "exponential" and voice.chirp.tau > 0:
                    dec = np.exp(-t_s / voice.chirp.tau)
                    f_full += (voice.chirp.f_delta_start * dec
                               + voice.chirp.f_delta_end * (1 - dec))
                elif ct2 == "power" and dur > 0:
                    tau_n = (t_s / dur) ** max(voice.chirp.chirp_power, 1e-3)
                    f_full += (voice.chirp.f_delta_start * (1.0 - tau_n)
                               + voice.chirp.f_delta_end * tau_n)
                if voice.fm and voice.fm.source_key:
                    sk = voice.fm.source_key
                    if sk in lfo_map:
                        fmod = _lfo_signal(lfo_map[sk], t_s)
                    elif sk in p_map and sk != voice.key:
                        fmod = np.cos(2.0 * np.pi * p_map[sk].freq_hz * t_s)
                    else:
                        fmod = np.zeros(n)
                    f_full += voice.fm.depth_hz * fmod
                f_arr = f_full.astype(np.float32)
            else:
                f_arr = np.zeros(n, dtype=np.float32)
            fm_min, fm_max = float(f_arr.min()), float(f_arr.max())
            margin = max((fm_max - fm_min) * 0.15, 1.0)
            self.v_lo = fm_min - margin
            self.v_hi = fm_max + margin
            self.plot.y_min, self.plot.y_max = self.v_lo, self.v_hi
            self.plot.add_series(PlotSeries(
                key="fm_inst", label="f_inst (Hz)",
                color=col, line=True, dots=False,
                data_x=t_ax, data_y=f_arr,
            ))

        elif self.mode == EditorMode.MIX:
            # Show per-voice signals plus the actual routed mix.
            _n_plot = len(t_ax)
            def _fit(sig):
                sig = np.asarray(sig, dtype=np.float32)
                if len(sig) >= _n_plot:
                    return sig[:_n_plot]
                return np.pad(sig, (0, _n_plot - len(sig)))
            # Synthesise each voice individually and cache results
            self._mix_voice_cache = {}
            self._mix_series_info  = []
            for v in patch.voices:
                if cancel.is_set(): return
                if _voice_effectively_muted(v, patch):
                    continue
                vsig = _synthesize_voice(v, patch, lfo_map, p_map,
                                         granular_seed_offset=self.granular_seed_offset).real
                vc   = tuple(v.color[:3])
                self._mix_voice_cache[f"mix_v_{v.key}"] = _fit(vsig)
                self._mix_series_info.append({"key": f"mix_v_{v.key}", "label": v.label, "color": vc})
            if cancel.is_set(): return
            # Full routed mix output — consistent with all other render paths
            left_out, right_out, _sc = _synthesize_patch(
                patch,
                granular_seed_offset=self.granular_seed_offset,
                _return_sidecar=True,
            )
            if cancel.is_set(): return
            self._last_sidecar: SidecarBus = _sc
            is_stereo = patch.projection_mode != "mono"
            l_color   = (255, 60, 60) if is_stereo else (255, 255, 255)
            self._mix_left_cache  = _fit(left_out)
            self._mix_right_cache = _fit(right_out) if is_stereo else None
            self._mix_t_ax        = t_ax
            self._mix_series_info.append({"key": "mix_sum_L", "label": "Mix L", "color": l_color})
            if is_stereo:
                self._mix_series_info.append({"key": "mix_sum_R", "label": "Mix R",
                                               "color": (60, 255, 60)})
            # Populate plot series (respects per-series visibility toggles)
            self._refresh_mix_series(patch)

        if cancel.is_set(): return
        self.t_lo, self.t_hi = 0.0, 1.0
        self._surf_dirty = True
        self._rebuild_hash = fingerprint

    # ---- Mix-tab series cache / toggles -----------------------------------

    def _refresh_mix_series(self, patch: "AnalyticPatch") -> None:
        """Re-populate plot series from cached mix signals without re-synthesising."""
        t_ax = self._mix_t_ax
        if t_ax is None:
            return
        self.plot.series.clear()
        plotted: list[np.ndarray] = []
        for info in self._mix_series_info:
            key = info["key"]
            if key in self._mix_series_hidden:
                continue
            if key.startswith("mix_v_"):
                data_y = self._mix_voice_cache.get(key)
                if data_y is None:
                    continue
            elif key == "mix_sum_L":
                data_y = self._mix_left_cache
                if data_y is None:
                    continue
            elif key == "mix_sum_R":
                data_y = self._mix_right_cache
                if data_y is None:
                    continue
            else:
                continue
            self.plot.add_series(PlotSeries(
                key=key, label=info["label"],
                color=info["color"], line=True, dots=False,
                data_x=t_ax, data_y=data_y,
            ))
            plotted.append(np.asarray(data_y, dtype=np.float64))
        self._set_plot_y_range_from_arrays(*plotted)

    # ---- Mix-tab projection controls ---------------------------------------

    _MIX_CTRL_H   = 30   # pixel height of the projection control bar
    _MIX_TOGGLE_H = 26   # pixel height of per-series toggle strip above ctrl bar
    _PROJ_MODES = ["mono", "stereo_quadrature", "stereo_ms", "lissajous"]
    _PROJ_LABELS = ["Mono", "Quad", "M+S", "Liss"]

    def _mix_ctrl_rects(self, surf_w: int, surf_h: int) -> dict:
        """Return rects (local to mix surface) for each projection control widget.

        Keys: 'proj_0'..'proj_3'  (mode buttons)
              'normalize'          (toggle button)
              'rot_dec', 'rot_inc' (rotation nudge ±10 Hz)
              'rot_val'            (rotation value label area)
        """
        h  = self._MIX_CTRL_H
        y0 = surf_h - h
        pad = 4
        btn_w = 56
        rects: dict[str, pygame.Rect] = {}
        x = pad
        for i, lbl in enumerate(self._PROJ_LABELS):
            rects[f"proj_{i}"] = pygame.Rect(x, y0 + pad, btn_w, h - 2 * pad)
            x += btn_w + pad
        x += pad
        rects["normalize"] = pygame.Rect(x, y0 + pad, 72, h - 2 * pad)
        x += 72 + 2 * pad
        rects["rot_dec"] = pygame.Rect(x, y0 + pad, 24, h - 2 * pad)
        x += 26
        rects["rot_val"] = pygame.Rect(x, y0 + pad, 74, h - 2 * pad)
        x += 76
        rects["rot_inc"] = pygame.Rect(x, y0 + pad, 24, h - 2 * pad)
        x += 28 + pad
        # Sample rate
        rects["sr_dec"] = pygame.Rect(x, y0 + pad, 18, h - 2 * pad)
        x += 20
        rects["sr_val"] = pygame.Rect(x, y0 + pad, 68, h - 2 * pad)
        x += 70
        rects["sr_inc"] = pygame.Rect(x, y0 + pad, 18, h - 2 * pad)
        return rects

    def _piano_snap_defs(self, bar_s: float) -> tuple[list[tuple[str, float | None]], list[tuple[str, float | None]]]:
        time_opts = [("Free", None), ("1/1", bar_s), ("1/2", bar_s * 0.5),
                     ("1/4", bar_s * 0.25), ("1/8", bar_s * 0.125),
                     ("1/16", bar_s * 0.0625)]
        len_opts = [("Free", None), ("1/1", bar_s), ("1/2", bar_s * 0.5),
                    ("1/4", bar_s * 0.25), ("1/8", bar_s * 0.125),
                    ("1/16", bar_s * 0.0625)]
        self._piano_time_snap_idx = max(0, min(self._piano_time_snap_idx, len(time_opts) - 1))
        self._piano_len_snap_idx = max(0, min(self._piano_len_snap_idx, len(len_opts) - 1))
        return time_opts, len_opts

    def _piano_note_by_id(self, patch: "AnalyticPatch", note_id: str) -> "ResolvedNote | None":
        return next((n for n in patch.resolved_notes if n.note_id == note_id), None)

    def _render_piano_roll_view(self, surf: pygame.Surface,
                                patch: "AnalyticPatch",
                                font: "pygame.font.Font | None") -> None:
        fb = font or self._font_()
        fh = fb.get_height()
        w, h = surf.get_size()
        _sync_resolved_notes(patch, preserve_locked=True)
        notes: list[ResolvedNote] = sorted(
            patch.resolved_notes,
            key=lambda n: (n.start_time, n.fundamental_hz, n.voice_key, n.layer_key),
        )

        title = fb.render("Piano Roll  —  resolved union score", True, (190, 205, 230))
        surf.blit(title, (8, 6))
        beat_s = 60.0 / max(getattr(patch, "seq_bpm", 120.0), 1.0)
        beats_per_bar = patch.beats_per_bar()
        bar_s = _bar_duration_s_for_patch(patch, beat_s)
        time_opts, len_opts = self._piano_snap_defs(bar_s)

        ctrl_y = fh + 12
        ctrl_h = fh + 6
        ctrl_x = 8
        self._piano_ctrl_rects = {}
        ctrl_specs = [
            ("sync", "Sync"),
            ("rest", f"Rest: {'On' if self._piano_rest_mode else 'Off'}"),
            ("pitch", f"Pitch: {'Semi' if self._piano_pitch_snap else 'Free'}"),
            ("time", f"Time: {time_opts[self._piano_time_snap_idx][0]}"),
            ("length", f"Length: {len_opts[self._piano_len_snap_idx][0]}"),
        ]
        for key, label in ctrl_specs:
            tw = fb.size(label)[0] + 12
            rect = pygame.Rect(ctrl_x, ctrl_y, tw, ctrl_h)
            pygame.draw.rect(surf, (46, 52, 72), rect, border_radius=3)
            pygame.draw.rect(surf, (88, 102, 136), rect, 1, border_radius=3)
            surf.blit(fb.render(label, True, (215, 225, 245)), (rect.x + 6, rect.y + 3))
            self._piano_ctrl_rects[key] = rect
            ctrl_x += tw + 6

        info = f"Locked {sum(1 for n in notes if n.locked)}/{len(notes)}"
        info_sf = fb.render(info, True, (120, 135, 160))
        surf.blit(info_sf, (w - info_sf.get_width() - 8, ctrl_y + 3))

        roll_top = ctrl_y + ctrl_h + 8
        roll_bottom = h - 8
        guide_w = 72
        roll_x0 = guide_w + 4
        roll_w = max(32, w - roll_x0 - 6)
        roll_h = max(32, roll_bottom - roll_top)
        roll_rect = pygame.Rect(roll_x0, roll_top, roll_w, roll_h)
        guide_rect = pygame.Rect(0, roll_top, guide_w, roll_h)
        self._piano_note_rects = []

        pygame.draw.rect(surf, (12, 16, 22), guide_rect)
        pygame.draw.rect(surf, (10, 12, 18), roll_rect)

        if not notes:
            surf.blit(fb.render("No resolved notes. Click Sync after enabling sequence data.",
                                True, (120, 130, 150)), (8, roll_top + 8))
            return

        min_midi = int(math.floor(min(_hz_to_midi(n.fundamental_hz) for n in notes))) - 2
        max_midi = int(math.ceil(max(_hz_to_midi(n.fundamental_hz) for n in notes))) + 2
        lane_count = max(12, max_midi - min_midi + 1)
        lane_h = max(10, min(22, roll_h // lane_count if lane_count > 0 else 14))
        content_h = lane_count * lane_h
        if content_h < roll_h:
            roll_top += (roll_h - content_h) // 2
            guide_rect = pygame.Rect(0, roll_top, guide_w, content_h)
            roll_rect = pygame.Rect(roll_x0, roll_top, roll_w, content_h)
            roll_h = content_h

        total_end = max(n.start_time + n.duration_s for n in notes)
        total_end = max(total_end, _bar_duration_s_for_patch(patch, beat_s))
        px_per_s = roll_w / max(total_end, 1e-6)

        for lane in range(lane_count):
            midi_note = max_midi - lane
            y = roll_top + lane * lane_h
            is_black = (midi_note % 12) in {1, 3, 6, 8, 10}
            octave = (midi_note // 12) - 1
            shade = (22 + (octave % 3) * 8, 24 + (octave % 2) * 8, 30 + (octave % 4) * 6)
            bg = (shade[0] + 6, shade[1] + 6, shade[2] + 6) if is_black else shade
            pygame.draw.rect(surf, bg, pygame.Rect(guide_rect.x, y, guide_rect.w, lane_h))
            pygame.draw.rect(surf, (16, 20, 28) if is_black else (24, 28, 36),
                             pygame.Rect(roll_rect.x, y, roll_rect.w, lane_h))
            pygame.draw.line(surf, (36, 42, 54), (roll_rect.x, y), (roll_rect.right, y))
            if midi_note % 12 == 0 or lane_h >= 16:
                lbl = _midi_note_name(midi_note)
                surf.blit(fb.render(lbl, True, (205, 212, 225) if midi_note % 12 == 0 else (120, 126, 138)),
                          (6, y + 1))

        n_bars = max(1, int(math.ceil(total_end / max(bar_s, 1e-6))))
        for bi in range(n_bars + 1):
            x = roll_x0 + int(bi * bar_s * px_per_s)
            pygame.draw.line(surf, (95, 105, 135), (x, roll_top), (x, roll_top + roll_h), 1)
            if bi < n_bars:
                surf.blit(fb.render(f"B{bi + 1}", True, (120, 130, 150)), (x + 3, roll_top + 2))
            beat_count = max(1, int(round(beats_per_bar * 4.0)) if abs(beats_per_bar - round(beats_per_bar)) > 0.001 else int(round(beats_per_bar)))
            sub_beat_s = bar_s / max(1, beat_count)
            for sub in range(1, beat_count):
                sx = x + int(sub * sub_beat_s * px_per_s)
                if sx < roll_x0 + roll_w:
                    pygame.draw.line(surf, (44, 50, 64), (sx, roll_top), (sx, roll_top + roll_h), 1)

        voice_colors = {v.key: tuple(v.color[:3]) for v in patch.voices}
        for note in notes:
            midi_f = _hz_to_midi(note.fundamental_hz)
            lane_pos = max_midi - midi_f
            nx = roll_x0 + int(note.start_time * px_per_s)
            nw = max(4, int(note.duration_s * px_per_s))
            ny = roll_top + int(lane_pos * lane_h)
            rect = pygame.Rect(nx, ny + 1, nw, max(6, lane_h - 2))
            base = voice_colors.get(note.voice_key, (130, 170, 220))
            layer_tint = 22 if note.layer_key == "all" else 0
            fill = tuple(min(255, c + layer_tint) for c in base)
            if note.is_rest:
                fill = (70, 55, 55) if note.locked else (52, 42, 42)
            if note.locked:
                fill = tuple(min(255, c + 18) for c in fill)
            pygame.draw.rect(surf, fill, rect, border_radius=3)
            border = (245, 230, 130) if note.locked else (24, 24, 28)
            if note.note_id == self._piano_selected_note_id:
                border = (255, 255, 255)
            pygame.draw.rect(surf, border, rect, 2 if note.note_id == self._piano_selected_note_id else 1,
                             border_radius=3)
            handle = pygame.Rect(rect.right - 6, rect.y, 6, rect.h)
            self._piano_note_rects.append({"note_id": note.note_id, "rect": rect, "handle": handle})
            if rect.w > 30:
                note_lbl = ("REST" if note.is_rest else f"{note.voice_label}:{note.layer_key}")
                surf.blit(fb.render(note_lbl, True, (16, 18, 22)), (rect.x + 4, rect.y + 1))

    def _piano_snap_time(self, value: float, bar_s: float, snap_idx: int) -> float:
        opts, _ = self._piano_snap_defs(bar_s)
        snap = opts[snap_idx][1]
        if snap is None or snap <= 0:
            return max(0.0, value)
        return max(0.0, round(value / snap) * snap)

    def _piano_snap_length(self, value: float, bar_s: float, snap_idx: int) -> float:
        _, opts = self._piano_snap_defs(bar_s)
        snap = opts[snap_idx][1]
        if snap is None or snap <= 0:
            return max(1.0 / 48000.0, value)
        return max(snap, round(value / snap) * snap)

    def _handle_piano_roll_event(self, event: pygame.event.Event,
                                 patch: "AnalyticPatch",
                                 win_w: int, win_h: int) -> bool:
        cx, cy, cw, ch = self._canvas_rect(win_w, win_h)
        plot_y = cy + MODEBAR_H
        if event.type == MOUSEBUTTONUP and event.button == 1:
            self._piano_dragging = False
            self._piano_drag_kind = ""
            self._piano_drag_note_id = ""
            return False

        if event.type == MOUSEMOTION and self._piano_dragging:
            note = self._piano_note_by_id(patch, self._piano_drag_note_id)
            if note is None:
                return False
            beat_s = 60.0 / max(getattr(patch, "seq_bpm", 120.0), 1.0)
            bar_s = _bar_duration_s_for_patch(patch, beat_s)
            lx = event.pos[0] - cx
            ly = event.pos[1] - plot_y
            dx = lx - self._piano_drag_origin.get("mouse_x", lx)
            dy = ly - self._piano_drag_origin.get("mouse_y", ly)
            px_per_s = self._piano_drag_origin.get("px_per_s", 100.0)
            lane_h = max(1.0, self._piano_drag_origin.get("lane_h", 14.0))
            note.locked = True
            if self._piano_drag_kind == "move":
                start = self._piano_drag_origin["start_time"] + dx / px_per_s
                midi = self._piano_drag_origin["midi"] - dy / lane_h
                note.start_time = self._piano_snap_time(start, bar_s, self._piano_time_snap_idx)
                if not note.is_rest:
                    if self._piano_pitch_snap:
                        midi = round(midi)
                    note.fundamental_hz = _midi_to_hz(midi)
            elif self._piano_drag_kind == "resize":
                dur = self._piano_drag_origin["duration_s"] + dx / px_per_s
                note.duration_s = self._piano_snap_length(dur, bar_s, self._piano_len_snap_idx)
            self._surf_dirty = True
            return True

        if not hasattr(event, "pos"):
            return False
        lx = event.pos[0] - cx
        ly = event.pos[1] - plot_y

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            for key, rect in self._piano_ctrl_rects.items():
                if rect.collidepoint(lx, ly):
                    if key == "sync":
                        _sync_resolved_notes(patch, preserve_locked=True)
                    elif key == "rest":
                        self._piano_rest_mode = not self._piano_rest_mode
                    elif key == "pitch":
                        self._piano_pitch_snap = not self._piano_pitch_snap
                    elif key == "time":
                        time_opts, _ = self._piano_snap_defs(
                            _bar_duration_s_for_patch(patch))
                        self._piano_time_snap_idx = (self._piano_time_snap_idx + 1) % len(time_opts)
                    elif key == "length":
                        _, len_opts = self._piano_snap_defs(
                            _bar_duration_s_for_patch(patch))
                        self._piano_len_snap_idx = (self._piano_len_snap_idx + 1) % len(len_opts)
                    self._surf_dirty = True
                    return True
            for info in reversed(self._piano_note_rects):
                if info["handle"].collidepoint(lx, ly) or info["rect"].collidepoint(lx, ly):
                    note = self._piano_note_by_id(patch, info["note_id"])
                    if note is None:
                        continue
                    self._piano_selected_note_id = note.note_id
                    self._piano_dragging = True
                    self._piano_drag_kind = "resize" if info["handle"].collidepoint(lx, ly) else "move"
                    self._piano_drag_note_id = note.note_id
                    rect = info["rect"]
                    lane_h = max(1.0, rect.h)
                    px_per_s = rect.w / max(note.duration_s, 1e-6)
                    self._piano_drag_origin = {
                        "mouse_x": lx,
                        "mouse_y": ly,
                        "start_time": note.start_time,
                        "duration_s": note.duration_s,
                        "midi": _hz_to_midi(note.fundamental_hz),
                        "px_per_s": px_per_s,
                        "lane_h": lane_h,
                    }
                    self._surf_dirty = True
                    return True
            if self._piano_rest_mode:
                beat_s = 60.0 / max(getattr(patch, "seq_bpm", 120.0), 1.0)
                bar_s = _bar_duration_s_for_patch(patch, beat_s)
                guide_w = 72
                roll_x0 = guide_w + 4
                roll_w = max(32, cw - roll_x0 - 6)
                total_end = max((n.start_time + n.duration_s for n in patch.resolved_notes), default=bar_s)
                total_end = max(total_end, bar_s)
                note_start = self._piano_snap_time(
                    max(0.0, (lx - roll_x0) / max(1.0, roll_w) * total_end),
                    bar_s, self._piano_time_snap_idx)
                note_len = self._piano_snap_length(bar_s * 0.25, bar_s, self._piano_len_snap_idx)
                voice = next((v for v in patch.voices if not getattr(v, "muted", False)), None)
                if voice is not None:
                    rest = ResolvedNote(
                        note_id=uuid.uuid4().hex[:12],
                        voice_key=voice.key,
                        voice_label=getattr(voice, "label", voice.key[:6]),
                        layer_key="mask",
                        start_time=note_start,
                        duration_s=note_len,
                        fundamental_hz=float(getattr(voice, "freq_hz", 440.0)),
                        velocity=0.0,
                        locked=True,
                        is_rest=True,
                    )
                    patch.resolved_notes.append(rest)
                    self._piano_selected_note_id = rest.note_id
                    self._surf_dirty = True
                    return True
            self._piano_selected_note_id = ""
            self._surf_dirty = True
            return False

        if event.type == MOUSEBUTTONDOWN and event.button == 3:
            for info in reversed(self._piano_note_rects):
                if info["rect"].collidepoint(lx, ly):
                    note = self._piano_note_by_id(patch, info["note_id"])
                    if note is not None:
                        note.locked = not note.locked
                        self._piano_selected_note_id = note.note_id
                        self._surf_dirty = True
                        return True
        return False

    # ══════════════════════════════════════════════════════════════════════
    #  Placement vector diagram
    # ══════════════════════════════════════════════════════════════════════

    def _render_placement_view(self, surf: pygame.Surface,
                               patch: "AnalyticPatch",
                               font: "pygame.font.Font | None") -> None:
        """Top-down orchestral stage diagram with performer dots, face arrows, room walls and baffles."""
        fb = font or self._font_()
        fh = fb.get_height()
        w, h = surf.get_size()

        # Title
        title = fb.render("Placement  —  click to clone · right-click to remove", True, (180, 200, 230))
        surf.blit(title, (8, 6))

        res_cfg = getattr(patch, "placement_resonator", None)
        if res_cfg is None or not res_cfg.enabled:
            msg = fb.render("(placement resonator disabled — enable it in the patch panel)", True, (120, 120, 140))
            surf.blit(msg, (w // 2 - msg.get_width() // 2, h // 2))
            self._placement_perf_rects = []
            return

        parts = getattr(patch, "parts", [])
        if not parts:
            msg = fb.render("(solve to populate performers)", True, (100, 100, 120))
            surf.blit(msg, (w // 2 - msg.get_width() // 2, h // 2))
            self._placement_perf_rects = []
            return

        # ── Coordinate mapping: world meters → pixel ──
        room_r = max(1.0, res_cfg.room_radius)
        margin_top = fh + 20
        margin = 30
        view_w = w - 2 * margin
        view_h = h - margin_top - margin
        cx_px = margin + view_w // 2
        cy_px = margin_top + view_h // 2
        extent = room_r * 1.15  # a bit wider than room radius
        scale = min(view_w, view_h) / (2.0 * extent) * self._placement_zoom

        def w2px(wx: float, wy: float) -> tuple[int, int]:
            return int(cx_px + wx * scale), int(cy_px - wy * scale)

        # ── Room boundary ──
        room_shape = getattr(res_cfg, "room_shape", "polygon")
        if room_shape == "circular":
            rpx = int(room_r * scale)
            pygame.draw.circle(surf, (40, 45, 58), (cx_px, cy_px), rpx, 1)
        else:
            # Polygon vertices from the resonance plugin
            verts_xy = [
                (-room_r * 0.95, -room_r * 0.70),
                (room_r, -room_r * 0.62),
                (room_r * 0.88, room_r * 0.72),
                (-room_r * 0.84, room_r * 0.64),
            ]
            pts = [w2px(vx, vy) for vx, vy in verts_xy]
            pygame.draw.polygon(surf, (40, 45, 58), pts, 1)

        # ── Center baffle line ──
        baffle_y_world = res_cfg.room_height * 0.42
        baffle_len = room_r * 0.5
        bx0, by0 = w2px(-baffle_len, 0)
        bx1, by1 = w2px(baffle_len, 0)
        pygame.draw.line(surf, (65, 55, 85), (bx0, by0), (bx1, by1), 1)

        # ── Conductor marker at origin ──
        cond = w2px(0.0, 0.0)
        pygame.draw.circle(surf, (200, 180, 100), cond, 5, 0)
        cs = fb.render("C", True, (40, 35, 20))
        surf.blit(cs, (cond[0] - cs.get_width() // 2, cond[1] - cs.get_height() // 2))

        # ── Receiver position ──
        rx, ry = float(getattr(res_cfg, "receiver_pos_x", 0.0)), float(getattr(res_cfg, "receiver_pos_y", 0.0))
        rpx_pos = w2px(rx, ry)
        pygame.draw.circle(surf, (100, 180, 220), rpx_pos, 4, 0)
        rs = fb.render("R", True, (20, 40, 60))
        surf.blit(rs, (rpx_pos[0] - rs.get_width() // 2, rpx_pos[1] - rs.get_height() // 2))

        # ── Section arcs (visual guide) ──
        _sec_col = (30, 35, 48)
        for sec_name, (sec_angle, sec_radius, _sec_z, sec_arc) in _STAGE_SECTIONS.items():
            sa_rad = math.radians(sec_angle)
            arc_half = math.radians(sec_arc * 0.5)
            a0 = sa_rad - arc_half
            a1 = sa_rad + arc_half
            n_seg = 12
            pts_arc = []
            for si in range(n_seg + 1):
                a = a0 + (a1 - a0) * si / n_seg
                sx = sec_radius * math.sin(a)
                sy = -sec_radius * math.cos(a)
                pts_arc.append(w2px(sx, sy))
            if len(pts_arc) >= 2:
                pygame.draw.lines(surf, _sec_col, False, pts_arc, 1)

        # ── Register color map ──
        _REG_PERF_COL: dict[str, tuple[int, int, int]] = {
            "bass":  (90, 60, 160),
            "mid":   (60, 120, 160),
            "high":  (80, 180, 120),
        }

        # ── Draw performers ──
        perf_rects: list[dict] = []
        for pt in parts:
            reg = getattr(pt, "register", "all")
            dot_col = _REG_PERF_COL.get(reg, (120, 120, 140))
            hover_col = tuple(min(255, c + 50) for c in dot_col)
            for ch in getattr(pt, "chairs", []):
                for pf in getattr(ch, "performers", []):
                    px, py = float(pf.x), float(pf.y)
                    sx, sy = w2px(px, py)
                    pf_key = str(pf.key)
                    is_hover = (pf_key == self._placement_hover_key)
                    r = 6 if is_hover else 4
                    col = hover_col if is_hover else dot_col
                    pygame.draw.circle(surf, col, (sx, sy), r, 0)

                    # Face direction arrow (aperture normal)
                    fx, fy = float(pf.face_x), float(pf.face_y)
                    arrow_len = 0.35  # meters
                    ax, ay = w2px(px + fx * arrow_len, py + fy * arrow_len)
                    pygame.draw.line(surf, col, (sx, sy), (ax, ay), 1)

                    # Store clickable rect
                    hit_r = pygame.Rect(sx - 8, sy - 8, 16, 16)
                    perf_rects.append({"rect": hit_r, "pf_key": pf_key,
                                       "part_key": pt.key, "screen": (sx, sy)})

        self._placement_perf_rects = perf_rects

        # ── Legend ──
        lx_start = 8
        ly = h - fh - 8
        for reg, col in _REG_PERF_COL.items():
            pygame.draw.rect(surf, col, pygame.Rect(lx_start, ly + 2, 10, fh - 4))
            lbl = fb.render(reg, True, (160, 170, 190))
            surf.blit(lbl, (lx_start + 14, ly))
            lx_start += 14 + lbl.get_width() + 12
        # Total count
        total = sum(pt.player_count for pt in parts)
        ts = fb.render(f"  {total} performers", True, (130, 140, 160))
        surf.blit(ts, (lx_start, ly))

    def _handle_placement_event(self, event: pygame.event.Event,
                                patch: "AnalyticPatch",
                                win_w: int, win_h: int) -> bool:
        """Handle mouse events in PLACEMENT mode: clone (left) / remove (right) performers."""
        cx, cy, cw, ch = self._canvas_rect(win_w, win_h)
        plot_rect = pygame.Rect(cx, cy + MODEBAR_H, cw, ch - MODEBAR_H)
        mx, my = pygame.mouse.get_pos()

        # ── Scroll-wheel zoom ──
        if event.type == MOUSEWHEEL and plot_rect.collidepoint(mx, my):
            factor = 1.1 if event.y > 0 else 0.9
            self._placement_zoom = max(0.3, min(5.0, self._placement_zoom * factor))
            self._surf_dirty = True
            return True

        # ── Hover tracking ──
        if event.type == MOUSEMOTION and plot_rect.collidepoint(mx, my):
            # Convert to surface-local coords
            lx = mx - cx
            ly = my - cy - MODEBAR_H
            old_hover = self._placement_hover_key
            self._placement_hover_key = ""
            for info in reversed(self._placement_perf_rects):
                if info["rect"].collidepoint(lx, ly):
                    self._placement_hover_key = info["pf_key"]
                    break
            if self._placement_hover_key != old_hover:
                self._surf_dirty = True
            return False  # don't consume motion events

        if not plot_rect.collidepoint(mx, my):
            return False

        lx = mx - cx
        ly = my - cy - MODEBAR_H

        # ── Left click: clone performer ──
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            for info in reversed(self._placement_perf_rects):
                if info["rect"].collidepoint(lx, ly):
                    pm = PlacementModule(patch)
                    pm.increment_player_count(info["part_key"], +1)
                    self._surf_dirty = True
                    return True

        # ── Right click: remove performer (down to required minimum) ──
        if event.type == MOUSEBUTTONDOWN and event.button == 3:
            for info in reversed(self._placement_perf_rects):
                if info["rect"].collidepoint(lx, ly):
                    pm = PlacementModule(patch)
                    pm.increment_player_count(info["part_key"], -1)
                    self._surf_dirty = True
                    return True

        return False

    def _render_control_routing_view(self, surf: pygame.Surface,
                                     patch: "AnalyticPatch",
                                     font: "pygame.font.Font | None") -> None:
        fb = font or self._font_()
        fh = fb.get_height()
        w, h = surf.get_size()
        surf.fill(_PY_BG)

        title = fb.render("Control Routing  —  click out→in · SM↔SM links author metaedges · right-click deletes", True, (190, 210, 230))
        surf.blit(title, (8, 6))

        rack = _rack_device_views_for_patch(patch)
        port_lookup = _published_port_lookup(patch)
        conns = _control_connections_for_patch(patch)
        self._control_device_rects = []
        self._control_port_rects = {}
        self._control_edge_hits = []

        unit_h = 26
        slot_w = 108
        x0 = 16
        y0 = fh + 20

        for dev in rack:
            rx = x0 + dev.grid_x * slot_w
            ry = y0 + dev.grid_y * unit_h
            rw = max(72, dev.rack_w * slot_w - 8)
            rh = max(22, dev.rack_u * unit_h - 6)
            rr = pygame.Rect(rx, ry, rw, rh)
            pygame.draw.rect(surf, tuple(max(0, c - 40) for c in dev.color), rr, border_radius=4)
            pygame.draw.rect(surf, dev.color, rr, 1, border_radius=4)
            self._control_device_rects.append({"rect": rr, "device": dev})
            hdr = fb.render(dev.label[:18], True, (180, 190, 210))
            surf.blit(hdr, (rx + 6, ry + 4))
            for pv in dev.ports:
                px = rx + pv.local_x
                py = ry + pv.local_y
                col = pv.port.color
                if pv.port.key == self._control_armed_port:
                    col = tuple(min(255, c + 60) for c in col)
                pygame.draw.circle(surf, col, (px, py), pv.radius, 0)
                self._control_port_rects[pv.port.key] = {
                    "rect": pygame.Rect(px - pv.radius - 2, py - pv.radius - 2, pv.radius * 2 + 4, pv.radius * 2 + 4),
                    "port": pv.port,
                    "center": (px, py),
                }

        # Connections last
        for conn in conns:
            s = self._control_port_rects.get(conn.src_port_key)
            d = self._control_port_rects.get(conn.dst_port_key)
            if not s or not d:
                continue
            p0 = s["center"]
            p1 = d["center"]
            if conn.remove_kind == "meta":
                col = (220, 120, 200)
            elif conn.edge_kind == "control":
                col = (120, 180, 240)
            else:
                col = (210, 150, 90)
            pygame.draw.line(surf, col, p0, p1, 1)
            mx = (p0[0] + p1[0]) // 2
            my = (p0[1] + p1[1]) // 2
            self._control_edge_hits.append({
                "src_port_key": conn.src_port_key,
                "dst_port_key": conn.dst_port_key,
                "kind": conn.edge_kind,
                "remove_kind": conn.remove_kind,
                "rect": pygame.Rect(mx - 4, my - 4, 8, 8),
            })

        hover_key = self._control_hover_port or self._control_armed_port
        if hover_key:
            port = port_lookup.get(hover_key)
            if port is not None:
                msg = (
                    f"{port.label}  [{port.domain}/{port.direction}]  "
                    f"rank={port.tensor.tensor_rank} lanes={port.tensor.lane_count} "
                    f"group={port.tensor.parallel_group or '-'} "
                    f"role={(port.semantic_role or port.tensor.semantic_role or '-')}"
                )
                if port.projection_policy:
                    msg += f" proj={port.projection_policy}"
                hs = fb.render(msg, True, (170, 185, 210))
                surf.blit(hs, (8, h - fh - 8))

    def _handle_control_routing_event(self, event: pygame.event.Event,
                                      patch: "AnalyticPatch",
                                      win_w: int, win_h: int) -> bool:
        cx, cy, cw, ch = self._canvas_rect(win_w, win_h)
        plot_rect = pygame.Rect(cx, cy + MODEBAR_H, cw, ch - MODEBAR_H)
        mx, my = pygame.mouse.get_pos()
        if event.type == MOUSEMOTION and plot_rect.collidepoint(mx, my):
            lx = mx - cx
            ly = my - cy - MODEBAR_H
            self._control_hover_port = ""
            for key, info in self._control_port_rects.items():
                if info["rect"].collidepoint(lx, ly):
                    self._control_hover_port = key
                    break
            self._surf_dirty = True
            return False
        if not plot_rect.collidepoint(mx, my):
            return False
        lx = mx - cx
        ly = my - cy - MODEBAR_H
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            for key, info in self._control_port_rects.items():
                if not info["rect"].collidepoint(lx, ly):
                    continue
                port = info["port"]
                if self._control_armed_port:
                    src = self._control_port_rects.get(self._control_armed_port, {}).get("port")
                    if src is not None and _control_add_connection(patch, src, port):
                        self._control_armed_port = ""
                        self._surf_dirty = True
                        return True
                self._control_armed_port = key if port.direction == "out" else ""
                self._surf_dirty = True
                return True
            self._control_armed_port = ""
            self._surf_dirty = True
            return False
        if event.type == MOUSEBUTTONDOWN and event.button == 3:
            for key, info in self._control_port_rects.items():
                if info["rect"].collidepoint(lx, ly):
                    _control_remove_connections_for_port(patch, key)
                    self._control_armed_port = ""
                    self._surf_dirty = True
                    return True
            for eh in self._control_edge_hits:
                if eh["rect"].collidepoint(lx, ly):
                    src_key = eh["src_port_key"]
                    dst_key = eh["dst_port_key"]
                    if eh.get("remove_kind") == "meta":
                        patch.routing.meta_edges = [
                            me for me in getattr(patch.routing, "meta_edges", [])
                            if not ((me.a_port or me.a_key) == src_key and (me.b_port or me.b_key) == dst_key)
                        ]
                    else:
                        patch.routing.edges = [
                            e for e in patch.routing.edges
                            if not ((e.src_port or e.src_key) == src_key and (e.dst_port or e.dst_key) == dst_key)
                        ]
                        patch.routing.param_edges = [
                            pe for pe in patch.routing.param_edges
                            if not ((pe.src_port or pe.src_key) == src_key
                                    and (pe.dst_port or _control_connection_target_port_key(pe.dst_key, pe.param_path)) == dst_key)
                        ]
                    self._control_armed_port = ""
                    self._surf_dirty = True
                    return True
        return False

    def _render_score_view(self, surf: pygame.Surface,
                           patch: "AnalyticPatch",
                           font: "pygame.font.Font | None") -> None:
        """Draw a phrase table: voices as rows, bars as columns, step cells inside."""
        fb    = font or self._font_()
        fh    = fb.get_height()
        w, h  = surf.get_size()

        # Header
        title = fb.render("Score  —  phrase × voices", True, (180, 200, 230))
        surf.blit(title, (8, 6))
        y = fh + 14

        beat_s  = 60.0 / max(getattr(patch, "seq_bpm", 120), 1)
        n_bars  = max(1, len(patch.rhythm_phrase))
        voices  = [v for v in patch.voices if not getattr(v, "muted", False)]

        bar_w   = max(40, (w - 120) // max(1, n_bars))
        row_h   = fh + 8
        col_x0  = 120

        # Column headers (bar numbers)
        for bi in range(n_bars):
            bx = col_x0 + bi * bar_w
            surf.blit(fb.render(f"B{bi + 1}", True, (120, 120, 150)), (bx + 4, y))
        y += fh + 4

        rows_y0 = y
        legend_y = h - fh - 6
        viewport_h = max(0, legend_y - rows_y0 - 4)
        content_h = max(0, len(voices) * (row_h + 2) - 2)
        self._score_max_scroll = max(0, content_h - viewport_h)
        self._score_scroll_y = max(0, min(self._score_scroll_y, self._score_max_scroll))

        old_clip = surf.get_clip()
        surf.set_clip(pygame.Rect(0, rows_y0, w, max(0, legend_y - rows_y0)))

        # One row per voice
        for vi, voice in enumerate(voices):
            ry = rows_y0 + vi * (row_h + 2) - self._score_scroll_y
            if ry + row_h < rows_y0 or ry > legend_y:
                continue
            # Row label
            reg   = getattr(voice, "register", "all")
            lbl   = f"{getattr(voice, 'label', voice.key[:6])} [{reg}]"
            surf.blit(fb.render(lbl[:14], True, (180, 160, 220)), (4, ry + 2))

            stack = patch.score_stack_for_voice(voice)
            pg    = stack[-1]  # most specific page for this voice
            phrase = pg.rhythm_phrase
            pats   = pg.rhythm_patterns

            for bi in range(n_bars):
                bx   = col_x0 + bi * bar_w
                br   = pygame.Rect(bx, ry, bar_w - 2, row_h)
                pat_i = phrase[bi % len(phrase)] if phrase else 0
                pat   = pats[min(pat_i, len(pats) - 1)] if pats else None
                has_custom = len(stack) > 1

                bg = (28, 20, 44) if not has_custom else (20, 28, 44)
                pygame.draw.rect(surf, bg, br, border_radius=2)
                pygame.draw.rect(surf, (52, 40, 78) if not has_custom else (40, 60, 100),
                                 br, 1, border_radius=2)

                if pat is not None and pg.rhythm_enabled:
                    div   = max(1, pg.rhythm_division)
                    cw_s  = max(2, (bar_w - 4) // div)
                    for si in range(div):
                        on = pat.steps[si] if si < len(pat.steps) else False
                        if on:
                            sx = bx + 2 + si * cw_s
                            sc = pygame.Rect(sx, ry + 2, max(1, cw_s - 1), row_h - 4)
                            pygame.draw.rect(surf, (110, 68, 180), sc, border_radius=1)
                else:
                    # Rhythm disabled — show a muted bar indicator
                    surf.blit(fb.render("—", True, (60, 55, 80)), (bx + bar_w // 2 - 4, ry + 2))

        surf.set_clip(old_clip)

        # Legend
        ly = h - fh - 6
        surf.blit(fb.render("■ custom page   □ default page   ■ on-step",
                             True, (80, 80, 100)), (8, ly))
        if self._score_max_scroll > 0:
            scroll_lbl = f"Scroll {self._score_scroll_y}/{self._score_max_scroll}"
            scroll_sf = fb.render(scroll_lbl, True, (95, 105, 125))
            surf.blit(scroll_sf, (w - scroll_sf.get_width() - 8, ly))

    def _render_mix_series_toggles(self, surf: pygame.Surface,
                                    font: pygame.font.Font,
                                    surf_w: int, surf_h: int) -> None:
        """Draw per-series show/hide toggle buttons in the strip just above the ctrl bar."""
        toggle_h = self._MIX_TOGGLE_H
        ctrl_h   = self._MIX_CTRL_H
        y0       = surf_h - ctrl_h - toggle_h
        pygame.draw.rect(surf, (10, 12, 18), pygame.Rect(0, y0, surf_w, toggle_h))
        pygame.draw.line(surf, (45, 48, 65), (0, y0), (surf_w, y0))
        fh  = font.get_height()
        pad = 4
        x   = pad
        self._mix_toggle_rects = []
        for info in self._mix_series_info:
            key    = info["key"]
            label  = info["label"]
            col    = info["color"]
            hidden = key in self._mix_series_hidden
            tw     = font.size(label)[0] + 10
            r      = pygame.Rect(x, y0 + (toggle_h - fh - 4) // 2, tw, fh + 4)
            if hidden:
                bg  = (max(0, col[0] - 110), max(0, col[1] - 110), max(0, col[2] - 110))
                brd = (55, 55, 72)
                tc  = (95, 95, 108)
            else:
                bg  = col
                brd = (min(255, col[0] + 50), min(255, col[1] + 50), min(255, col[2] + 50))
                tc  = (20, 20, 22) if max(col) > 150 else (225, 225, 235)
            pygame.draw.rect(surf, bg, r, border_radius=3)
            pygame.draw.rect(surf, brd, r, 1, border_radius=3)
            surf.blit(font.render(label, True, tc), (r.x + 5, r.y + 2))
            self._mix_toggle_rects.append({"key": key, "rect": r})
            x += tw + pad

    def _render_mix_controls(self, surf: pygame.Surface,
                             patch: AnalyticPatch,
                             font: pygame.font.Font) -> None:
        """Draw the projection control bar onto the bottom of the Mix surface."""
        w, h = surf.get_size()
        ch   = self._MIX_CTRL_H
        bar_y = h - ch
        pygame.draw.rect(surf, (20, 20, 28), pygame.Rect(0, bar_y, w, ch))
        pygame.draw.line(surf, (55, 55, 72), (0, bar_y), (w, bar_y), 1)
        rects = self._mix_ctrl_rects(w, h)
        fh = font.get_height()

        # Projection mode buttons
        for i, (key, lbl) in enumerate(zip(self._PROJ_MODES, self._PROJ_LABELS)):
            r = rects[f"proj_{i}"]
            active = (patch.projection_mode == key)
            bg = (40, 90, 180) if active else (35, 35, 48)
            pygame.draw.rect(surf, bg, r, border_radius=3)
            pygame.draw.rect(surf, (70, 70, 95), r, 1, border_radius=3)
            tc = (230, 240, 255) if active else (120, 120, 140)
            t = font.render(lbl, True, tc)
            surf.blit(t, (r.x + (r.w - t.get_width()) // 2,
                          r.y + (r.h - fh) // 2))

        # Normalize toggle
        r = rects["normalize"]
        active = patch.normalize_output
        bg = (30, 140, 80) if active else (90, 35, 35)
        pygame.draw.rect(surf, bg, r, border_radius=3)
        pygame.draw.rect(surf, (70, 70, 95), r, 1, border_radius=3)
        lbl = "Norm ON" if active else "Norm OFF"
        tc  = (200, 255, 210) if active else (255, 180, 170)
        t = font.render(lbl, True, tc)
        surf.blit(t, (r.x + (r.w - t.get_width()) // 2,
                      r.y + (r.h - fh) // 2))

        # Rotation dec/inc/val
        rot_hz = patch.projection_rotation_hz
        for key, lbl in [("rot_dec", "−"), ("rot_inc", "+")]:
            r = rects[key]
            pygame.draw.rect(surf, (35, 35, 50), r, border_radius=3)
            pygame.draw.rect(surf, (70, 70, 95), r, 1, border_radius=3)
            t = font.render(lbl, True, (180, 180, 200))
            surf.blit(t, (r.x + (r.w - t.get_width()) // 2,
                          r.y + (r.h - fh) // 2))
        r = rects["rot_val"]
        pygame.draw.rect(surf, (22, 22, 32), r, border_radius=2)
        pygame.draw.rect(surf, (55, 55, 72), r, 1, border_radius=2)
        t = font.render(f"Rot {rot_hz:+.1f}Hz", True, (160, 200, 255))
        surf.blit(t, (r.x + (r.w - t.get_width()) // 2,
                      r.y + (r.h - fh) // 2))

        # Sample-rate buttons
        for key, lbl in [("sr_dec", "◀"), ("sr_inc", "▶")]:
            r = rects[key]
            pygame.draw.rect(surf, (35, 35, 50), r, border_radius=3)
            pygame.draw.rect(surf, (70, 70, 95), r, 1, border_radius=3)
            t = font.render(lbl, True, (180, 180, 200))
            surf.blit(t, (r.x + (r.w - t.get_width()) // 2,
                          r.y + (r.h - fh) // 2))
        r = rects["sr_val"]
        pygame.draw.rect(surf, (22, 22, 32), r, border_radius=2)
        pygame.draw.rect(surf, (55, 55, 72), r, 1, border_radius=2)
        sr_k = patch.preview_sr // 1000
        sr_rem = patch.preview_sr - sr_k * 1000
        sr_lbl = f"{sr_k}k" if sr_rem == 0 else f"{patch.preview_sr}"
        t = font.render(f"SR {sr_lbl}Hz", True, (255, 210, 120))
        surf.blit(t, (r.x + (r.w - t.get_width()) // 2,
                      r.y + (r.h - fh) // 2))

    # ---- Rendering ---------------------------------------------------------

    def render(self, win_w: int, win_h: int,
               patch: AnalyticPatch,
               atlas: GlyphAtlas,
               font: pygame.font.Font | None = None) -> None:
        """Full render: PlotWidget surface → GL tex + GL overlay."""
        cx, cy, cw, ch = self._canvas_rect(win_w, win_h)
        voice = next((p for p in patch.voices if p.key == self.active_key), None)

        # ── Mode-tab background ──
        _gl_rect(cx, cy, cw, MODEBAR_H, win_w, win_h, (0.08, 0.08, 0.11, 1.0))
        vis_modes, vis_labels = self._visible_modes(patch)
        if self.mode not in vis_modes and vis_modes:
            self.mode = vis_modes[0]
            self._surf_dirty = True
        # Reserve the rightmost 44 px of the mode bar for the tail-toggle button
        _TAIL_BTN_W = 44
        tab_rects = self._mode_tab_rects(win_w, win_h, patch)
        for i, (mode, label, rect) in enumerate(
                zip(vis_modes, vis_labels, tab_rects)):
            col = _C_TAB_ACT if mode == self.mode else _C_TAB_IDLE
            _gl_rect(rect.x + 1, rect.y + 2, rect.w - 2, rect.h - 2,
                     win_w, win_h, col)
            if atlas.tex_id:
                tc = (0.90, 0.90, 0.95, 1.0) if mode == self.mode else (0.5, 0.5, 0.55, 1.0)
                glEnable(GL_TEXTURE_2D)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                atlas.draw_string(
                    label,
                    float(rect.x + rect.w // 2),
                    float(rect.y + rect.h // 2),
                    win_w, win_h, anchor_x=0.5, anchor_y=0.5, color=tc)
                glDisable(GL_TEXTURE_2D)
        # Tail toggle button — always visible at right of mode bar
        _tail_col = (0.35, 0.55, 0.35, 1.0) if self.show_tail else (0.18, 0.18, 0.22, 1.0)
        _tbx = cx + cw - _TAIL_BTN_W + 1
        _gl_rect(_tbx, cy + 2, _TAIL_BTN_W - 2, MODEBAR_H - 4, win_w, win_h, _tail_col)
        if atlas.tex_id:
            _tc = (0.7, 1.0, 0.7, 1.0) if self.show_tail else (0.4, 0.45, 0.4, 1.0)
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            atlas.draw_string(
                "tail",
                float(_tbx + _TAIL_BTN_W // 2),
                float(cy + MODEBAR_H // 2),
                win_w, win_h, anchor_x=0.5, anchor_y=0.5, color=_tc)
            glDisable(GL_TEXTURE_2D)

        # ── PlotWidget surface (or RoutingGridView) ──
        plot_y = cy + MODEBAR_H
        plot_h = ch - MODEBAR_H
        if self._surf is None or self._surf.get_width() != cw or self._surf.get_height() != plot_h:
            self._surf = pygame.Surface((cw, plot_h))
            self._surf_dirty = True

        if self.mode == EditorMode.ROUTING:
            # Routing view renders its own surface — always delegate
            r_surf = self.routing_view.render_surface(patch, cw, plot_h, font)
            self._tex = _surface_to_gl_tex(r_surf, self._tex)
        
        elif self.mode == EditorMode.CONTROL_ROUTING:
            if self._surf_dirty:
                self._render_control_routing_view(self._surf, patch, font)
                self._tex = _surface_to_gl_tex(self._surf, self._tex)
                self._surf_dirty = False
        elif self.mode == EditorMode.PARAM_ROUTING:
            # Param routing view: informational display for the active ParamNode
            pn = next((p for p in patch.param_nodes if p.key == self.active_key), None)
            if self._surf_dirty:
                self._surf.fill(_PY_BG)
                fb = font or self._font_()
                fh = fb.get_height()
                y = 12
                title = fb.render("Parametric Routing", True, (180, 140, 220))
                self._surf.blit(title, (12, y)); y += fh + 6
                if pn:
                    lines: list[tuple[str, tuple]] = [
                        (f"Node:    {pn.label}  [{pn.key[:8]}]",    (200, 200, 210)),
                        (f"Extract: {pn.extractor}",                 (140, 190, 230)),
                        (f"Default: {pn.default_value:.4g}",         (200, 200, 210)),
                        (f"Clamp:   [{pn.low:.4g}, {pn.high:.4g}]", (200, 200, 210)),
                        ("", (0, 0, 0)),
                        (f"Targets ({len(pn.targets)}):",            (180, 200, 160)),
                    ]
                    for ti, tgt in enumerate(pn.targets):
                        vk = tgt.get("voice_key", "")
                        at = tgt.get("attr", "")
                        # resolve voice label from patch if possible
                        all_nodes = list(patch.voices) + list(patch.lfos) + list(patch.modules)
                        vname = next((n.label for n in all_nodes if n.key == vk), vk[:8] if vk else "—")
                        lines.append((f"  [{ti}] {vname} → {at or '—'}", (160, 210, 160)))
                    if not pn.targets:
                        lines.append(("  (none — add targets in the panel)", (100, 120, 140)))
                    lines += [
                        ("", (0, 0, 0)),
                        ("Route signal edges into this node via the Routing", (100, 120, 140)),
                        ("tab.  Its output drives all listed targets.",         (100, 120, 140)),
                    ]
                    for line, col in lines:
                        self._surf.blit(fb.render(line, True, col), (12, y))
                        y += fh + 3
                self._tex = _surface_to_gl_tex(self._surf, self._tex)
                self._surf_dirty = False
        elif self.mode == EditorMode.SM_LOG:
            mod = next((m for m in patch.modules if m.key == self.active_key), None)
            if self._surf_dirty:
                self._surf.fill(_PY_BG)
                fb = font or self._font_()
                fh = fb.get_height()
                y = 12
                title = fb.render("State Machine Log", True, (180, 210, 230))
                self._surf.blit(title, (12, y))
                y += fh + 8
                log_text = self._sm_log_text
                if mod is not None:
                    hdr = [
                        (f"Module: {mod.label}  [{mod.key[:8]}]", (200, 200, 210)),
                        (f"Plugin: {mod.sm_plugin or '(none)'}", (140, 190, 230)),
                        ("", (0, 0, 0)),
                    ]
                    for line, col in hdr:
                        self._surf.blit(fb.render(line, True, col), (12, y))
                        y += fh + 3
                lines = log_text.splitlines() if log_text else ["(no plugin output captured)"]
                visible_n = max(1, (plot_h - y - 8) // max(1, fh + 2))
                max_scroll = max(0, len(lines) - visible_n)
                self._sm_log_scroll = max(0, min(self._sm_log_scroll, max_scroll))
                for line in lines[self._sm_log_scroll:self._sm_log_scroll + visible_n]:
                    if y > plot_h - fh - 4:
                        break
                    self._surf.blit(fb.render(line or " ", True, (180, 185, 195)), (12, y))
                    y += fh + 2
                self._tex = _surface_to_gl_tex(self._surf, self._tex)
                self._surf_dirty = False
        elif self.mode == EditorMode.SCORE:
            if self._surf_dirty:
                self._surf.fill(_PY_BG)
                self._render_score_view(self._surf, patch, font)
                self._tex = _surface_to_gl_tex(self._surf, self._tex)
                self._surf_dirty = False
        elif self.mode == EditorMode.PIANO_ROLL:
            if self._surf_dirty:
                self._surf.fill(_PY_BG)
                self._render_piano_roll_view(self._surf, patch, font)
                self._tex = _surface_to_gl_tex(self._surf, self._tex)
                self._surf_dirty = False
        elif self.mode == EditorMode.PLACEMENT:
            if self._surf_dirty:
                self._surf.fill(_PY_BG)
                self._render_placement_view(self._surf, patch, font)
                self._tex = _surface_to_gl_tex(self._surf, self._tex)
                self._surf_dirty = False
        elif self.mode == EditorMode.PIECEWISE_EDITOR and voice is not None:
            editor = self._ensure_piecewise_editor(voice, win_w, win_h)
            if editor.get_revision() != self._piecewise_sync_revision:
                self._pull_piecewise_editor_state_into_voice(voice)
                self._surf_dirty = True
            if self._surf_dirty:
                if self._piecewise_preview_dirty:
                    with _PROF.span("gl.canvas.render.piecewise_preview_req"):
                        self._request_piecewise_preview(patch, voice, editor)
                    self._piecewise_preview_dirty = False
                with _PROF.span("gl.canvas.render.piecewise_preview_apply"):
                    self._apply_piecewise_preview_if_ready(editor)
                with _PROF.span("gl.canvas.render.piecewise_render_to_surf"):
                    bg_surf, atom_list = editor.snapshot_gl_atoms()
                    dirty_rects = editor.consume_gl_dirty_rects()
                with _PROF.span("gl.canvas.render.piecewise_tex_upload"):
                    if self._piecewise_stale_tex_ids:
                        glDeleteTextures([tid for tid in self._piecewise_stale_tex_ids if tid])
                        for tid in self._piecewise_stale_tex_ids:
                            _tex_size_cache.pop(tid, None)
                        self._piecewise_stale_tex_ids.clear()
                    self._tex = _surface_rects_to_gl_tex(bg_surf, dirty_rects, self._tex)
                    live_atom_keys: set[str] = set()
                    for atom in atom_list:
                        if atom.surface is None:
                            continue
                        live_atom_keys.add(atom.key)
                        old_tex = self._piecewise_atom_tex.get(atom.key, 0)
                        atom_changed = (
                            not old_tex
                            or not dirty_rects
                            or any(
                                rect.colliderect(getattr(atom, "motion_rect", getattr(atom, "visible_rect", rect)))
                                or rect.colliderect(getattr(atom, "visible_rect", rect))
                                for rect in dirty_rects
                            )
                        )
                        if atom_changed:
                            self._piecewise_atom_tex[atom.key] = _surface_to_gl_tex(atom.surface, old_tex)
                    stale_keys = [key for key in self._piecewise_atom_tex if key not in live_atom_keys]
                    if stale_keys:
                        glDeleteTextures([self._piecewise_atom_tex[key] for key in stale_keys if self._piecewise_atom_tex[key]])
                        for key in stale_keys:
                            tid = self._piecewise_atom_tex.pop(key, 0)
                            if tid:
                                _tex_size_cache.pop(tid, None)
                    self._piecewise_atoms = atom_list
                self._surf_dirty = False
        elif self._surf_dirty:
            self._surf.fill(_PY_BG)
            if self.mode == EditorMode.MIX:
                # Reserve bottom strip for projection controls; plot fills rest
                ctrl_h   = self._MIX_CTRL_H
                toggle_h = self._MIX_TOGGLE_H
                self.plot.render(self._surf, 0, 0, cw, plot_h - ctrl_h - toggle_h, font)
                fb = font or self._font_()
                self._render_mix_series_toggles(self._surf, fb, cw, plot_h)
                self._render_mix_controls(self._surf, patch, fb)
            else:
                self.plot.render(self._surf, 0, 0, cw, plot_h, font)
            self._tex = _surface_to_gl_tex(self._surf, self._tex)
            self._surf_dirty = False

        _draw_tex_quad(self._tex, cx, plot_y, cw, plot_h, win_w, win_h)

        # ── Direct GL overlays (wave widget bypasses texture path) ──
        if self.mode == EditorMode.PIECEWISE_EDITOR and voice is not None:
            editor.draw_gl_overlays(cx, plot_y, win_w, win_h)
            for atom in self._piecewise_atoms:
                tex = self._piecewise_atom_tex.get(atom.key, 0)
                rect = getattr(atom, "visible_rect", getattr(atom, "rect", None))
                if not tex or rect is None or rect.w <= 0 or rect.h <= 0:
                    continue
                _draw_tex_quad(tex, cx + rect.x, plot_y + rect.y, rect.w, rect.h, win_w, win_h)

        # ── GL overlay ──
        ix, iy, iw, ih = self._inner_rect(win_w, win_h)
        plot_top = cy + MODEBAR_H
        plot_bot = cy + ch

        # Loop region fill + handles
        if voice and voice.loop_enabled and self.mode == EditorMode.WAVEFORM:
            lsx, _ = self.data_to_px(voice.loop_start, 0, win_w, win_h)
            lex, _ = self.data_to_px(voice.loop_end,   0, win_w, win_h)
            lsx = max(ix, min(lsx, ix + iw))
            lex = max(ix, min(lex, ix + iw))
            if lex > lsx:
                _gl_rect(lsx, float(iy), lex - lsx, float(ih),
                         win_w, win_h, _C_LOOP_FILL)
            _gl_loop_handle(lsx, float(iy), float(iy + ih), win_w, win_h, _C_LOOP, True)
            _gl_loop_handle(lex, float(iy), float(iy + ih), win_w, win_h, _C_LOOP, False)

        # Cursor flag
        cpx, _ = self.data_to_px(self.cursor_t, 0, win_w, win_h)
        cpx = max(ix, min(cpx, ix + iw))
        _gl_cursor_flag(cpx, float(iy), float(iy + ih), win_w, win_h, _C_CURSOR)

        # Envelope control points
        if self.mode == EditorMode.ENVELOPE and voice:
            for ki, knot in enumerate(voice.active_knots()):
                kpx, kpy = self.data_to_px(knot[0], knot[1], win_w, win_h)
                hover = (ki == self._hover_cp)
                brd = (1.0, 1.0, 1.0, 1.0) if hover else _C_CP_BORD
                _gl_diamond(kpx, kpy, float(CP_DRAW_PX + 2 if hover else CP_DRAW_PX),
                            win_w, win_h, _C_CP_FILL, brd)

    # ---- Event handling ----------------------------------------------------

    def handle_event(self, event: pygame.event.Event,
                     patch: AnalyticPatch, win_w: int, win_h: int) -> bool:
        """Returns True if event was consumed."""
        voice = next((p for p in patch.voices if p.key == self.active_key), None)
        ix, iy, iw, ih = self._inner_rect(win_w, win_h)

        # ---- Tail toggle button (always first) ------------------------------
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            cx, cy, cw, _ = self._canvas_rect(win_w, win_h)
            _TAIL_BTN_W = 44
            _tbr = pygame.Rect(cx + cw - _TAIL_BTN_W + 1, cy + 2, _TAIL_BTN_W - 2, MODEBAR_H - 4)
            if _tbr.collidepoint(event.pos):
                self.show_tail = not self.show_tail
                self._surf_dirty = True
                return True

        # ---- Mix-tab projection controls -----------------------------------
        if self.mode == EditorMode.MIX:
            cx, cy, cw, ch = self._canvas_rect(win_w, win_h)
            plot_y = cy + MODEBAR_H
            plot_h = ch - MODEBAR_H
            # ctrl rects are in surface-local coords; convert mouse to local
            lx = (event.pos[0] if hasattr(event, "pos") else 0) - cx
            ly = (event.pos[1] if hasattr(event, "pos") else 0) - plot_y
            rects = self._mix_ctrl_rects(cw, plot_h)

            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Tab click first
                _vm, _ = self._visible_modes(patch)
                for mode, rect in zip(_vm, self._mode_tab_rects(win_w, win_h, patch)):
                    if rect.collidepoint(mx, my):
                        self.mode = mode
                        self._surf_dirty = True
                        return True
                # Series visibility toggle buttons
                for t in self._mix_toggle_rects:
                    if t["rect"].collidepoint(lx, ly):
                        k = t["key"]
                        if k in self._mix_series_hidden:
                            self._mix_series_hidden.discard(k)
                        else:
                            self._mix_series_hidden.add(k)
                        self._refresh_mix_series(patch)
                        self._surf_dirty = True
                        return True
                # Projection mode buttons
                for i, key in enumerate(self._PROJ_MODES):
                    if rects[f"proj_{i}"].collidepoint(lx, ly):
                        patch.projection_mode = key
                        self._surf_dirty = True
                        return True
                # Normalize toggle
                if rects["normalize"].collidepoint(lx, ly):
                    patch.normalize_output = not patch.normalize_output
                    self._surf_dirty = True
                    return True
                # Rotation buttons
                if rects["rot_dec"].collidepoint(lx, ly):
                    patch.projection_rotation_hz -= 10.0
                    self._surf_dirty = True
                    return True
                if rects["rot_inc"].collidepoint(lx, ly):
                    patch.projection_rotation_hz += 10.0
                    self._surf_dirty = True
                    return True
                # Drag on rot_val label → start rotation drag
                if rects["rot_val"].collidepoint(lx, ly):
                    self._rot_dragging  = True
                    self._rot_drag_x0   = mx
                    self._rot_drag_val0 = patch.projection_rotation_hz
                    return True
                # Sample rate step through standard rates
                _SR_STEPS = [8000, 11025, 16000, 22050, 32000, 44100,
                             48000, 88200, 96000, 176400, 192000]
                if rects["sr_dec"].collidepoint(lx, ly):
                    sr_opts = [s for s in _SR_STEPS if s < patch.preview_sr]
                    if sr_opts:
                        patch.preview_sr = sr_opts[-1]
                    self._surf_dirty = True
                    return True
                if rects["sr_inc"].collidepoint(lx, ly):
                    sr_opts = [s for s in _SR_STEPS if s > patch.preview_sr]
                    if sr_opts:
                        patch.preview_sr = sr_opts[0]
                    self._surf_dirty = True
                    return True

            elif event.type == MOUSEBUTTONUP and event.button == 1:
                self._rot_dragging = False

            elif event.type == MOUSEMOTION and self._rot_dragging:
                delta = (event.pos[0] - self._rot_drag_x0) * 0.5  # 0.5 Hz/px
                patch.projection_rotation_hz = self._rot_drag_val0 + delta
                self._surf_dirty = True
                return True

        if self.mode == EditorMode.PIECEWISE_EDITOR and voice is not None:
            cx, cy, cw, ch = self._canvas_rect(win_w, win_h)
            plot_y = cy + MODEBAR_H
            plot_h = ch - MODEBAR_H
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                _vm, _ = self._visible_modes(patch)
                for mode, rect in zip(_vm, self._mode_tab_rects(win_w, win_h, patch)):
                    if rect.collidepoint(mx, my):
                        self.mode = mode
                        self._surf_dirty = True
                        return True
            editor = self._ensure_piecewise_editor(voice, win_w, win_h)
            local_x = None
            local_y = None
            if hasattr(event, "pos"):
                mx, my = event.pos
                if not pygame.Rect(cx, plot_y, cw, plot_h).collidepoint(mx, my):
                    return False
                local_x = float(mx - cx)
                local_y = float(plot_h - (my - plot_y))
            if event.type == MOUSEBUTTONDOWN:
                editor.on_mouse_down(event.button, local_x or 0.0, local_y or 0.0, pygame.key.get_mods())
                if editor.is_frame_dirty():
                    self._surf_dirty = True
                return True
            if event.type == MOUSEBUTTONUP:
                editor.on_mouse_up(event.button)
                if editor.is_frame_dirty():
                    self._surf_dirty = True
                return True
            if event.type == MOUSEMOTION:
                editor.on_mouse_move(local_x or 0.0, local_y or 0.0)
                if editor.is_frame_dirty():
                    self._surf_dirty = True
                return True
            if event.type == KEYDOWN:
                editor.on_key_down(event.key, event.mod)
                if editor.is_frame_dirty():
                    self._surf_dirty = True
                return True
            if event.type == pygame.KEYUP:
                editor.on_key_up(event.key)
                if editor.is_frame_dirty():
                    self._surf_dirty = True
                return True
            if event.type == pygame.TEXTINPUT:
                editor.on_text_input(event.text)
                if editor.is_frame_dirty():
                    self._surf_dirty = True
                return True

        # Delegate to routing view when in ROUTING mode
        if self.mode == EditorMode.ROUTING:
            cx, cy, cw, ch = self._canvas_rect(win_w, win_h)
            plot_y = cy + MODEBAR_H

            # Still handle tab clicks first
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                _vm, _ = self._visible_modes(patch)
                for mode, rect in zip(_vm, self._mode_tab_rects(win_w, win_h, patch)):
                    if rect.collidepoint(mx, my):
                        self.mode = mode
                        self._surf_dirty = True
                        self.routing_view.mark_dirty()
                        return True

            consumed = self.routing_view.handle_event(event, patch, cx, plot_y)
            if consumed:
                # Routing graph changed — rebuild audio on next frame
                self._surf_dirty = True
                self.routing_view.mark_dirty()
            return consumed

        if self.mode == EditorMode.CONTROL_ROUTING:
            consumed = self._handle_control_routing_event(event, patch, win_w, win_h)
            if consumed:
                self._surf_dirty = True
            return consumed

        if self.mode == EditorMode.SM_LOG:
            if event.type == MOUSEWHEEL:
                lines = self._sm_log_text.splitlines() if self._sm_log_text else ["(no plugin output captured)"]
                step = -int(event.y)
                max_scroll = max(0, len(lines) - 1)
                new_scroll = max(0, min(self._sm_log_scroll + step, max_scroll))
                if new_scroll != self._sm_log_scroll:
                    self._sm_log_scroll = new_scroll
                    self._surf_dirty = True
                    return True
        if self.mode == EditorMode.SCORE and event.type == MOUSEWHEEL:
            cx, cy, cw, ch = self._canvas_rect(win_w, win_h)
            plot_rect = pygame.Rect(cx, cy + MODEBAR_H, cw, ch - MODEBAR_H)
            mx, my = pygame.mouse.get_pos()
            if plot_rect.collidepoint(mx, my):
                step = max(24, self._font_().get_height() + 10)
                new_scroll = max(0, min(self._score_scroll_y - int(event.y) * step,
                                        self._score_max_scroll))
                if new_scroll != self._score_scroll_y:
                    self._score_scroll_y = new_scroll
                    self._surf_dirty = True
                    return True
        if self.mode == EditorMode.PIANO_ROLL:
            consumed = self._handle_piano_roll_event(event, patch, win_w, win_h)
            if consumed:
                return True
        if self.mode == EditorMode.PLACEMENT:
            consumed = self._handle_placement_event(event, patch, win_w, win_h)
            if consumed:
                return True

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos

            # Mode tab click
            _vm, _ = self._visible_modes(patch)
            for i, (mode, rect) in enumerate(
                    zip(_vm, self._mode_tab_rects(win_w, win_h, patch))):
                if rect.collidepoint(mx, my):
                    self.mode = mode
                    if self.mode == EditorMode.WAVEFORM:
                        self.v_lo, self.v_hi = -1.05, 1.05
                    elif self.mode == EditorMode.ENVELOPE:
                        self.v_lo, self.v_hi = -0.05, 1.1
                    self._surf_dirty = True
                    return True

            if not (ix <= mx <= ix + iw and iy <= my <= iy + ih):
                return False

            t_click, v_click = self.px_to_data(mx, my, win_w, win_h)
            t_click = max(0.0, min(1.0, t_click))

            # Loop handles (check before cursor)
            if voice and voice.loop_enabled and self.mode == EditorMode.WAVEFORM:
                lsx, _ = self.data_to_px(voice.loop_start, 0, win_w, win_h)
                lex, _ = self.data_to_px(voice.loop_end,   0, win_w, win_h)
                if abs(mx - lsx) < LOOP_HIT_PX:
                    self.drag = DragState(active=True, kind="loop_start",
                                          voice_key=self.active_key,
                                          start_px=(mx, my),
                                          start_val=(voice.loop_start, 0))
                    return True
                if abs(mx - lex) < LOOP_HIT_PX:
                    self.drag = DragState(active=True, kind="loop_end",
                                          voice_key=self.active_key,
                                          start_px=(mx, my),
                                          start_val=(voice.loop_end, 0))
                    return True

            # Envelope knots
            if self.mode == EditorMode.ENVELOPE and voice:
                knots = voice.active_knots()
                best_i, best_d = -1, CP_HIT_PX + 1
                for ki, knot in enumerate(knots):
                    kpx, kpy = self.data_to_px(knot[0], knot[1], win_w, win_h)
                    d = math.hypot(mx - kpx, my - kpy)
                    if d < best_d:
                        best_d, best_i = d, ki
                if best_i >= 0:
                    knot = voice.active_knots()[best_i]
                    self.drag = DragState(active=True, kind="knot",
                                          index=best_i, voice_key=self.active_key,
                                          start_px=(mx, my),
                                          start_val=(knot[0], knot[1]))
                    return True
                # Click in envelope area but no knot hit → add knot (spline/monotone/linear only)
                if voice.env_type != "adsr":
                    t_n = max(0.0, min(1.0, t_click))
                    v_n = max(0.0, min(1.0, v_click))
                    voice.env_knots.append([t_n, v_n])
                    voice.env_knots.sort(key=lambda k: k[0])
                    self._surf_dirty = True
                    return True

            # Cursor drag
            cpx, _ = self.data_to_px(self.cursor_t, 0, win_w, win_h)
            if abs(mx - cpx) < LOOP_HIT_PX + 4 or ix <= mx <= ix + iw:
                self.drag = DragState(active=True, kind="cursor",
                                      start_px=(mx, my),
                                      start_val=(t_click, 0))
                self.cursor_t = t_click
                return True

        elif event.type == MOUSEBUTTONDOWN and event.button == 3:
            # Right-click envelope knot → delete (spline/monotone/linear only)
            mx, my = event.pos
            if self.mode == EditorMode.ENVELOPE and voice and voice.env_type != "adsr":
                knots = voice.active_knots()
                for ki, knot in enumerate(knots):
                    kpx, kpy = self.data_to_px(knot[0], knot[1], win_w, win_h)
                    if math.hypot(mx - kpx, my - kpy) < CP_HIT_PX:
                        # keep at least 2 knots
                        if len(voice.env_knots) > 2:
                            voice.env_knots.pop(ki)
                            self._surf_dirty = True
                        return True

        elif event.type == MOUSEBUTTONUP and event.button == 1:
            self.drag.active = False

        elif event.type == MOUSEMOTION:
            mx, my = event.pos

            if self.drag.active:
                voice_d = next(
                    (p for p in patch.voices if p.key == self.drag.voice_key), None)
                t_now, v_now = self.px_to_data(mx, my, win_w, win_h)
                t_now = max(0.0, min(1.0, t_now))
                v_now = max(self.v_lo, min(self.v_hi, v_now))

                if self.drag.kind == "cursor":
                    self.cursor_t = t_now
                    return True

                if voice_d and self.drag.kind == "loop_start":
                    voice_d.loop_start = _snap_to_phase_boundary(t_now, voice_d, patch, self._phase_cycles)
                    self._surf_dirty = True
                    return True

                if voice_d and self.drag.kind == "loop_end":
                    voice_d.loop_end = _snap_to_phase_boundary(t_now, voice_d, patch, self._phase_cycles)
                    self._surf_dirty = True
                    return True

                if voice_d and self.drag.kind == "knot":
                    ki = self.drag.index
                    if voice_d.env_type == "adsr":
                        # Map knot index to ADSR parameter
                        knots = voice_d.adsr.to_knots(1.0)
                        if 0 < ki < len(knots):
                            dur = 1.0
                            if ki == 1:
                                voice_d.adsr.attack = max(0.001, t_now * dur)
                                voice_d.adsr.peak   = max(0.0, min(1.0, v_now))
                            elif ki == 2:
                                a = voice_d.adsr.attack
                                voice_d.adsr.decay   = max(0.001, t_now * dur - a)
                                voice_d.adsr.sustain = max(0.0, min(1.0, v_now))
                            elif ki == 3:
                                voice_d.adsr.release = max(0.001, (1.0 - t_now) * dur)
                    else:
                        if 0 <= ki < len(voice_d.env_knots):
                            voice_d.env_knots[ki] = [
                                max(0.0, min(1.0, t_now)),
                                max(0.0, min(1.0, v_now)),
                            ]
                            # Keep sorted by time, anchoring endpoints
                            if 0 < ki < len(voice_d.env_knots) - 1:
                                voice_d.env_knots.sort(key=lambda k: k[0])
                    self._surf_dirty = True
                    return True

            # Hover detection for envelope knots
            if self.mode == EditorMode.ENVELOPE and voice:
                knots = voice.active_knots()
                self._hover_cp = -1
                for ki, knot in enumerate(knots):
                    kpx, kpy = self.data_to_px(knot[0], knot[1], win_w, win_h)
                    if math.hypot(mx - kpx, my - kpy) < CP_HIT_PX:
                        self._hover_cp = ki
                        break

        elif event.type == MOUSEWHEEL:
            # Horizontal zoom on the time axis
            mx, _ = pygame.mouse.get_pos()
            ix, iy, iw, ih = self._inner_rect(win_w, win_h)
            if ix <= mx <= ix + iw:
                t_pivot = self.t_lo + (mx - ix) / max(iw, 1) * (self.t_hi - self.t_lo)
                factor = 0.85 if event.y > 0 else 1.0 / 0.85
                span = (self.t_hi - self.t_lo) * factor
                span = max(0.05, min(1.0, span))
                self.t_lo = max(0.0, t_pivot - span * 0.5)
                self.t_hi = min(1.0, self.t_lo + span)
                self.t_lo = max(0.0, self.t_hi - span)
                self._surf_dirty = True
                return True

        return False


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PlacementModule — derives seating / performer allocation from the patch's
# solved Part list (populated by resolve_parts_from_patch after each solve).
# ---------------------------------------------------------------------------

class PlacementModule:
    """Manages performer allocation across resolved :class:`Part` objects."""

    def __init__(self, patch: "AnalyticPatch") -> None:
        self._patch = patch

    # ── Part access ──────────────────────────────────────────────────────────

    @property
    def parts(self) -> "list[Part]":
        return self._patch.parts

    def get_part(self, key: str) -> "Part | None":
        for pt in self._patch.parts:
            if pt.key == key:
                return pt
        return None

    # ── Mutation helpers ─────────────────────────────────────────────────────

    def set_player_count(self, part_key: str, count: int) -> None:
        """Clamp count to [1, 64] and apply to the matching Part."""
        count = max(1, min(64, int(count)))
        pt = self.get_part(part_key)
        if pt is not None:
            required = max(1, int(pt.solver_hints.get("required_chairs", 1)))
            pt.player_count = max(required, count)
            _refresh_part_placement_layout(self._patch)

    def increment_player_count(self, part_key: str, delta: int = 1) -> None:
        pt = self.get_part(part_key)
        if pt is not None:
            self.set_player_count(part_key, pt.player_count + delta)

    # ── Summary queries ──────────────────────────────────────────────────────

    def total_performers(self) -> int:
        return sum(pt.player_count for pt in self._patch.parts)

    def placement_summary(self) -> "list[dict]":
        return [
            {
                "key": pt.key,
                "label": pt.label,
                "register": pt.register,
                "seq_role": pt.seq_role,
                "voice_role": pt.voice_role,
                "voice_keys": list(pt.voice_keys),
                "player_count": pt.player_count,
                "solver_hints": dict(pt.solver_hints),
                "chairs": [
                    {
                        "key": ch.key,
                        "label": ch.label,
                        "chair_index": ch.chair_index,
                        "specificity_rank": ch.specificity_rank,
                        "source_voice_keys": list(ch.source_voice_keys),
                        "source_layer_keys": list(ch.source_layer_keys),
                        "performer_count": ch.performer_count,
                        "solver_hints": dict(ch.solver_hints),
                        "performers": [
                            {
                                "key": pf.key,
                                "label": pf.label,
                                "performer_index": pf.performer_index,
                                "source_voice_keys": list(pf.source_voice_keys),
                                "source_layer_keys": list(pf.source_layer_keys),
                                "assigned_note_keys": list(pf.assigned_note_keys),
                                "body_type": pf.body_type,
                                "x": pf.x,
                                "y": pf.y,
                                "z": pf.z,
                                "face_x": pf.face_x,
                                "face_y": pf.face_y,
                                "face_z": pf.face_z,
                                "radius": pf.radius,
                                "angle_deg": pf.angle_deg,
                                "geometric_delay_ms": pf.geometric_delay_ms,
                                "humanization_ms": pf.humanization_ms,
                                "phase_offset_rad": pf.phase_offset_rad,
                                "gain_db": pf.gain_db,
                                "pan": pf.pan,
                            }
                            for pf in ch.performers
                        ],
                    }
                    for ch in pt.chairs
                ],
            }
            for pt in self._patch.parts
        ]

    def resonator_summary(self) -> dict:
        cfg = getattr(self._patch, "placement_resonator", PlacementResonatorConfig())
        return {
            "enabled": bool(cfg.enabled),
            "deployed_module_key": str(cfg.deployed_module_key),
            "item_count": int(_placement_resonator_item_count(self._patch)),
            "body_types": list(_placement_body_types_for_patch(self._patch)),
            "params": dict(_placement_resonator_module_params(self._patch)),
        }

    def deployed_resonator_module(self) -> "AnalyticModule | None":
        cfg = getattr(self._patch, "placement_resonator", PlacementResonatorConfig())
        preferred = str(cfg.deployed_module_key or "")
        if preferred:
            for mod in self._patch.modules:
                if mod.key == preferred:
                    return mod
        for mod in self._patch.modules:
            if mod.module_type == "state_machine" and str(getattr(mod, "sm_plugin", "")) == "orchestral_resonance":
                return mod
        return None

    def ensure_resonator_module(self) -> "AnalyticModule":
        mod = self.deployed_resonator_module()
        if mod is None:
            mod = AnalyticModule(
                key=f"placement_res_{uuid.uuid4().hex[:6]}",
                label="Orchestral Resonance",
                module_type="state_machine",
            )
            self._patch.modules.append(mod)
        self.sync_resonator_module(mod.key)
        return mod

    def sync_resonator_module(self, module_key: str | None = None) -> "AnalyticModule | None":
        cfg = getattr(self._patch, "placement_resonator", PlacementResonatorConfig())
        mod = None
        if module_key:
            mod = next((m for m in self._patch.modules if m.key == module_key), None)
        if mod is None:
            mod = self.deployed_resonator_module()
        if mod is None:
            if not cfg.enabled:
                return None
            mod = self.ensure_resonator_module()
            return mod
        _sync_placement_resonator_module(self._patch, mod)
        cfg.deployed_module_key = mod.key
        return mod


# ---------------------------------------------------------------------------
# PatchPanel — left panel: voice list
# ---------------------------------------------------------------------------

class PatchPanel(Panel):
    """Left panel: scrollable list of voices + LFOs."""

    @property
    def panel_rect(self) -> pygame.Rect:
        sw = pygame.display.get_surface().get_width()
        sh = pygame.display.get_surface().get_height()
        top = self._top_offset
        h = sh - top - BOTTOM_H
        return pygame.Rect(0, top, self.PANEL_W, h)

    def __init__(self, side: str = "left") -> None:
        super().__init__(side=side)
        self.title = "Voices"
        self._font: pygame.font.Font | None = None
        self.on_select:        Any = None
        self.on_add_voice:     Any = None
        self.on_add_lfo:       Any = None
        self.on_add_param:     Any = None
        self.on_add_module:    Any = None
        self.on_add_control:   Any = None
        self.on_add_router:    Any = None
        self.on_toggle_mute:   Any = None
        self.on_remove_voice:  Any = None   # callback(key: str)
        self.on_deploy_chord:  Any = None   # callback()
        self.on_demo_play:     Any = None   # callback()
        self.on_render:        Any = None   # callback() — render sequence to file
        self.on_render_fund:   Any = None   # callback() — render fundamental (like spacebar)
        self._patch:  AnalyticPatch | None = None
        self._active: str = ""
        # Collapsible section state
        self._voices_collapsed: bool = False
        self._seq_collapsed:    bool = True
        # Voice list geometry (surface-local coords, filled by render)
        self._items:             list       = []
        self._row_h:             int        = 22
        self._btn_y:             int        = 9999
        self._btn_h:             int        = 20
        self._btn2_y:            int        = 9999
        self._btn2_h:            int        = 20
        self._voices_hdr_rect:   Any        = None
        self._voices_row_start_y: int       = 22
        # Sequence section geometry
        self._seq_hdr_rect:      Any        = None
        self._seq_sliders:       list[dict] = []
        self._seq_arrows:        list[dict] = []
        self._seq_btn_rects:     dict       = {}
        self._dragging_seq_slider: int      = -1
        # Rhythm section geometry
        self._rhythm_collapsed:       bool       = True
        self._rhythm_hdr_rect:        Any        = None
        self._rhythm_enable_rect:     Any        = None
        self._rhythm_div_rects:       list       = []
        self._rhythm_pat_tabs:        list       = []
        self._rhythm_pat_add_rect:    Any        = None
        self._rhythm_pat_del_rect:    Any        = None
        self._rhythm_page_rects:      list       = []
        self._rhythm_page_add_r:      Any        = None
        self._rhythm_page_del_r:      Any        = None
        self._rhythm_meter_inherit_r: Any        = None
        self._rhythm_stress_left_rect: Any       = None
        self._rhythm_stress_right_rect: Any      = None
        self._rhythm_auto_grid_rect:  Any        = None
        self._rhythm_stress_vel_rect: Any        = None
        self._frac_beat_warp_rect:    Any        = None
        self._frac_beat_grid_rect:    Any        = None
        self._placement_collapsed: bool          = True
        self._placement_hdr_rect: Any            = None
        self._placement_pm_rects: list           = []    # per-Part [(+r, -r, part_key), ...]
        self._rhythm_step_rects:      list       = []
        self._rhythm_ctx_menu:        dict | None = None  # context menu state
        self._rhythm_phrase_rects:    list       = []
        self._rhythm_phrase_add_rect: Any        = None
        self._rhythm_phrase_del_rect: Any        = None
        self._rhythm_prog_dec_rect:   Any        = None
        self._rhythm_prog_inc_rect:   Any        = None
        self._rhythm_fit_drop_rect:   Any        = None
        self._rhythm_fit_ext_rect:    Any        = None
        self._rhythm_sliders:         list[dict] = []
        self._dragging_rhythm_slider: int        = -1
        # Probabilities section geometry
        self._prob_collapsed:       bool       = True
        self._prob_hdr_rect:        Any        = None
        self._prob_sliders:         list[dict] = []
        self._dragging_prob_slider: int        = -1
        # Dynamics section geometry
        self._dyn_collapsed:       bool       = True
        self._dyn_hdr_rect:        Any        = None
        self._dyn_enable_rect:     Any        = None
        self._dyn_page_key:        str        = "all"
        self._dyn_page_rects:      list       = []
        self._dyn_curve_left_rect: Any        = None
        self._dyn_curve_right_rect:Any        = None
        self._dyn_scope_dec_rect:  Any        = None
        self._dyn_scope_inc_rect:  Any        = None
        self._dyn_auto_accent_rect:Any        = None
        self._dyn_accent_rects:    list       = []
        self._dyn_sliders:         list[dict] = []
        self._dragging_dyn_slider: int        = -1
        self._accent_layer:        Any        = None
        self._accent_tree:         Any        = None
        self._accent_snap_rect:    Any        = None
        self._accent_snap_pat:     Any        = None
        # Improv section geometry
        self._improv_collapsed:         bool       = True
        self._improv_grace_collapsed:   bool       = True
        self._improv_chirp_collapsed:   bool       = True
        self._improv_echo_collapsed:    bool       = True
        self._improv_hdr_rect:          Any        = None
        self._improv_enable_rect:       Any        = None
        self._improv_page_key:          str        = "all"
        self._improv_page_rects:        list       = []
        self._improv_prob_sliders:      list[dict] = []
        self._dragging_improv_prob_sl:  int        = -1
        # Grace sub-section
        self._improv_grace_hdr_rect:    Any        = None
        self._improv_grace_arrows:      list       = []
        self._improv_grace_sliders:     list[dict] = []
        self._dragging_grace_sl:        int        = -1
        self._improv_grace_trim_rect:   Any        = None
        # Chirp sub-section
        self._improv_chirp_hdr_rect:    Any        = None
        self._improv_chirp_arrows:      list       = []
        self._improv_chirp_sliders:     list[dict] = []
        self._dragging_chirp_sl:        int        = -1
        # Echo sub-section
        self._improv_echo_hdr_rect:     Any        = None
        self._improv_echo_sliders:      list[dict] = []
        self._dragging_echo_sl:         int        = -1
        # Improv step eligibility grid
        self._improv_step_rects:        list       = []
        self._improv_layer:             Any        = None
        self._improv_tree:              Any        = None
        self._improv_snap_rect:         Any        = None
        self._improv_snap_pat:          Any        = None
        # Rhythm tree reference for snap operations
        self._rhythm_tree:              Any        = None
        # Dropdown state for Module / Control add buttons
        self._open_dropdown:     str        = ""   # "module" | "control" | ""
        self._dropdown_items:    list       = []   # list[dict(label, rect, data)]
        self._btn_module_rect:   Any        = None
        self._btn_control_rect:  Any        = None

    def set_patch(self, patch: AnalyticPatch, active_key: str) -> None:
        self._patch  = patch
        self._active = active_key

    def _font_(self) -> pygame.font.Font:
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("consolas", 12)
        return self._font

    def _add_seq_slider(self, sliders: list, y: int, label: str, key: str,
                        val: float, lo: float, hi: float, fmt: str = ".3g") -> int:
        sliders.append(dict(
            label=label, key=key, val=val, lo=lo, hi=hi,
            fmt=fmt, is_log=False,
            rect=pygame.Rect(8, y, self.PANEL_W - 16, 14),
        ))
        return y + 14 + 4 + 14  # slider + pad + label

    def _draw_page_selector_row(self, surf: pygame.Surface,
                                font: pygame.font.Font,
                                y: int,
                                active_key: str,
                                rhythm_pages: dict,
                                active_col: tuple[int, int, int],
                                idle_col: tuple[int, int, int],
                                active_brd: tuple[int, int, int],
                                custom_brd: tuple[int, int, int],
                                idle_brd: tuple[int, int, int],
                                active_txt: tuple[int, int, int],
                                custom_txt: tuple[int, int, int],
                                idle_txt: tuple[int, int, int],
                                hdr_h: int) -> tuple[int, list]:
        surf.blit(font.render("Part:", True, _PY_DIM), (8, y + 3))
        rects: list = []
        pgx = 44
        for pg_key, pg_label in self._page_selector_entries(active_key, rhythm_pages):
            is_active = (active_key == pg_key)
            has_custom = (pg_key != "all" and pg_key in rhythm_pages)
            pg_col = active_col if is_active else idle_col
            pg_brd = active_brd if is_active else (custom_brd if has_custom else idle_brd)
            pg_txt = active_txt if is_active else (custom_txt if has_custom else idle_txt)
            pg_w = max(32, min(72, font.size(pg_label)[0] + 8))
            pg_r = pygame.Rect(pgx, y + 1, pg_w, hdr_h - 2)
            pygame.draw.rect(surf, pg_col, pg_r, border_radius=3)
            pygame.draw.rect(surf, pg_brd, pg_r, 1, border_radius=3)
            surf.blit(font.render(pg_label, True, pg_txt), (pgx + 3, y + 3))
            rects.append({"rect": pg_r, "key": pg_key})
            pgx += pg_w + 2
        return y + hdr_h + 2, rects

    def _page_selector_entries(self, active_key: str, page_dict: dict) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = [("all", "All"), ("bass", "Bass"), ("mid", "Mid"), ("high", "High")]
        seen = {key for key, _ in entries}
        if self._patch is not None and any(v.key == self._active for v in self._patch.voices):
            vk = self._active
            if vk not in seen:
                entries.append((vk, vk[:6]))
                seen.add(vk)
        if active_key not in seen:
            entries.append((active_key, active_key[:6] if active_key else "Pg"))
            seen.add(active_key)
        for key in sorted(page_dict.keys()):
            if key not in seen and key not in {"all", ""}:
                entries.append((key, key[:6]))
                seen.add(key)
        return entries

    def _meter_beat_units(self, meter_num: float, frac_beat_mode: str = "warp") -> list[float]:
        return _meter_beat_units(meter_num, frac_beat_mode)

    def _metric_grid_rows(self, meter_num: float, usable_w: int, beat_min_w: int = 28, frac_beat_mode: str = "warp") -> int:
        beat_count = len(self._meter_beat_units(meter_num, frac_beat_mode))
        usable_w = max(1, int(usable_w))
        beats_per_row = max(1, min(beat_count, usable_w // max(1, beat_min_w)))
        return max(1, (beat_count + beats_per_row - 1) // beats_per_row)

    def _metric_step_layout(
            self,
            y: int,
            div: int,
            meter_num: float,
            panel_w: int,
            cell_h: int = 16,
            x0: int = 6,
            gap: int = 2,
            beat_min_w: int = 28,
    ) -> tuple[list[dict], int]:
        """Lay out step cells grouped by metrical beat opportunities."""
        div = max(1, int(div))
        beat_units = self._meter_beat_units(meter_num, "warp")
        beat_count = len(beat_units)
        usable_w = max(1, panel_w - (x0 * 2))
        beats_per_row = max(1, min(beat_count, usable_w // max(1, beat_min_w)))
        row_count = max(1, (beat_count + beats_per_row - 1) // beats_per_row)
        row_step = cell_h + gap
        cells: list[dict] = []

        total_units = max(1e-9, sum(beat_units))
        unit_edges = [0.0]
        acc = 0.0
        for unit in beat_units:
            acc += unit
            unit_edges.append(acc)
        step_bounds = [int(round((edge / total_units) * div)) for edge in unit_edges]
        step_bounds[0] = 0
        step_bounds[-1] = div

        for beat_i, beat_unit in enumerate(beat_units):
            row = beat_i // beats_per_row
            col = beat_i % beats_per_row
            row_start = row * beats_per_row
            row_units = beat_units[row_start:row_start + beats_per_row]
            beats_in_row = len(row_units)
            if beats_in_row <= 0:
                continue
            row_unit_total = max(1e-9, sum(row_units))
            row_prefix = sum(row_units[:col])
            beat_x0 = x0 + int(round((row_prefix / row_unit_total) * usable_w))
            beat_x1 = x0 + int(round(((row_prefix + beat_unit) / row_unit_total) * usable_w))
            if col > 0:
                beat_x0 += gap // 2
            if col < beats_in_row - 1:
                beat_x1 -= gap - (gap // 2)
            beat_y = y + row * row_step

            step_start = max(0, min(div, step_bounds[beat_i]))
            step_end = max(step_start, min(div, step_bounds[beat_i + 1]))
            step_count = max(0, step_end - step_start)
            if step_count <= 0:
                continue

            beat_span = max(1, beat_x1 - beat_x0)
            for local_i, step_i in enumerate(range(step_start, step_end)):
                sx0 = beat_x0 + int(round(local_i * beat_span / float(step_count)))
                sx1 = beat_x0 + int(round((local_i + 1) * beat_span / float(step_count)))
                if local_i > 0:
                    sx0 += 1
                sx1 = max(sx0 + 6, sx1 - 1)
                cells.append({
                    "rect": pygame.Rect(sx0, beat_y, max(6, sx1 - sx0), cell_h),
                    "step_i": step_i,
                    "beat_i": beat_i,
                    "row": row,
                    "beat_start": (local_i == 0),
                    "step_count_in_beat": step_count,
                    "beat_unit": beat_unit,
                })

        total_h = row_count * row_step
        return cells, total_h

    def _tree_step_layout(
            self,
            y: int,
            tree: "BeatTree",
            meter_num: float,
            panel_w: int,
            cell_h: int = 16,
            x0: int = 6,
            gap: int = 2,
            beat_min_w: int = 28,
            frac_beat_mode: str = "warp",
    ) -> tuple[list[dict], int]:
        """Lay out tree leaves grouped by metrical beat, with sub-rows for depth."""
        from fractions import Fraction as _F
        beat_units = self._meter_beat_units(meter_num, frac_beat_mode)
        beat_count = len(beat_units)
        usable_w = max(1, panel_w - (x0 * 2))
        beats_per_row = max(1, min(beat_count, usable_w // max(1, beat_min_w)))
        row_count = max(1, (beat_count + beats_per_row - 1) // beats_per_row)
        row_step = cell_h + gap

        # Build cumulative beat boundaries as Fractions over one bar
        total_units = sum(beat_units)
        if total_units < 1e-9:
            total_units = 1.0
        cum = 0.0
        beat_edges_f: list[float] = [0.0]
        for u in beat_units:
            cum += u
            beat_edges_f.append(cum / total_units)

        # Collect all leaves with their depth
        leaves = tree.flat_leaves()
        max_depth = max((lf.depth() for n in tree.nodes for lf in n.leaves()), default=0)
        # Actually depth from root for each leaf:
        def _leaf_depth(node: "BeatNode", d: int = 0) -> list[tuple["BeatNode", int]]:
            if node.is_leaf():
                return [(node, d)]
            out = []
            for c in sorted(node.children, key=lambda c: c.position):
                out.extend(_leaf_depth(c, d + 1))
            return out

        all_leaf_depths: list[tuple["BeatNode", int]] = []
        max_d = 0
        for top_node in tree.nodes:
            pairs = _leaf_depth(top_node)
            all_leaf_depths.extend(pairs)
            for _, d in pairs:
                if d > max_d:
                    max_d = d

        # Subdivided cells get a depth indicator bar; for now all leaves on same
        # row, but we show depth via shading and a small depth marker.
        cells: list[dict] = []
        for beat_i, beat_unit in enumerate(beat_units):
            row = beat_i // beats_per_row
            col = beat_i % beats_per_row
            row_start = row * beats_per_row
            row_units = beat_units[row_start:row_start + beats_per_row]
            beats_in_row = len(row_units)
            if beats_in_row <= 0:
                continue
            row_unit_total = max(1e-9, sum(row_units))
            row_prefix = sum(row_units[:col])
            beat_x0 = x0 + int(round((row_prefix / row_unit_total) * usable_w))
            beat_x1 = x0 + int(round(((row_prefix + beat_unit) / row_unit_total) * usable_w))
            if col > 0:
                beat_x0 += gap // 2
            if col < beats_in_row - 1:
                beat_x1 -= gap - (gap // 2)
            beat_y = y + row * row_step
            beat_span = max(1, beat_x1 - beat_x0)

            # Leaves whose position falls within this beat
            b_lo = beat_edges_f[beat_i]
            b_hi = beat_edges_f[beat_i + 1]
            beat_leaves = [(lf, d) for lf, d in all_leaf_depths
                           if b_lo - 1e-12 <= float(lf.position) < b_hi - 1e-12]
            if not beat_leaves:
                continue

            # Sort by position
            beat_leaves.sort(key=lambda x: x[0].position)
            # Proportional widths by duration
            total_dur = sum(float(lf.duration) for lf, _ in beat_leaves)
            if total_dur < 1e-15:
                total_dur = 1.0
            px = beat_x0
            for li, (leaf, depth) in enumerate(beat_leaves):
                frac_w = float(leaf.duration) / total_dur
                sx1 = beat_x0 + int(round((px - beat_x0 + frac_w * beat_span)))
                if li > 0:
                    px += 1
                sx1 = max(px + 6, sx1 - 1)
                cells.append({
                    "rect": pygame.Rect(px, beat_y, max(6, sx1 - px), cell_h),
                    "node_id": leaf.node_id,
                    "leaf": leaf,
                    "depth": depth,
                    "beat_i": beat_i,
                    "row": row,
                    "beat_start": (li == 0),
                    "beat_unit": beat_unit,
                })
                px = sx1 + 1

        total_h = row_count * row_step
        return cells, total_h

    def _build_step_context_menu(
            self, cell: dict, lx: int, ly: int, tree: "BeatTree",
            div: int,
    ) -> dict:
        """Build context menu dict for a step cell at local (lx, ly)."""
        font = self._font_()
        item_h = font.get_height() + 6
        sep_h  = 6
        menu_w = 120
        items: list[dict] = []

        leaf = cell["leaf"]
        is_on = leaf.on

        # Toggle on/off
        items.append({"label": "Off" if is_on else "On",
                       "action": "toggle", "color": (180, 255, 180)})
        # Articulation submenu (only when on)
        if is_on:
            cur_art = int(leaf.art)
            art_names = {0: "Normal", 1: "Staccato", 2: "Legato", 3: "Drone"}
            for av, aname in art_names.items():
                marker = " \u2713" if av == cur_art else ""
                items.append({"label": f"  {aname}{marker}",
                               "action": "art", "art_val": av,
                               "color": (240, 220, 120) if av == cur_art else (200, 190, 220)})

        # Separator
        items.append({"separator": True})

        # Trigger group assignment
        cur_grp = int(leaf.group)
        items.append({"label": "Group: None" + (" \u2713" if cur_grp == 0 else ""),
                       "action": "group", "group_val": 0,
                       "color": (180, 180, 180) if cur_grp == 0 else (120, 120, 120)})
        for gi in range(1, len(GROUP_COLORS)):
            gc = GROUP_COLORS[gi]
            marker = " \u2713" if gi == cur_grp else ""
            items.append({"label": f"  Grp {gi}{marker}",
                           "action": "group", "group_val": gi,
                           "color": gc if gi == cur_grp else (gc[0] // 2, gc[1] // 2, gc[2] // 2)})
        items.append({"separator": True})

        # Subdivide options
        for n in (2, 3, 4, 5, 6, 7, 8, 9):
            items.append({"label": f"\u00f7{n}",
                           "action": "subdivide", "n": n,
                           "color": (160, 220, 255)})

        # Collapse (only if depth > 0)
        if cell.get("depth", 0) > 0:
            items.append({"separator": True})
            items.append({"label": "Collapse",
                           "action": "collapse",
                           "color": (255, 180, 160)})

        # Build rects
        menu_x = min(lx, max(4, (self.panel_rect.w if self.panel_rect else 200) - menu_w - 4))
        menu_y = ly + 2
        cy = menu_y + 4
        for item in items:
            if item.get("separator"):
                item["rect"] = pygame.Rect(menu_x, cy, menu_w, sep_h)
                cy += sep_h
            else:
                item["rect"] = pygame.Rect(menu_x + 2, cy, menu_w - 4, item_h)
                cy += item_h + 1

        menu_rect = pygame.Rect(menu_x, menu_y, menu_w, cy - menu_y + 4)
        return {"items": items, "menu_rect": menu_rect, "cell": cell,
                "tree": tree, "div": div}

    # ------------------------------------------------------------------
    # Shared grid view rendering & click handling
    # ------------------------------------------------------------------

    def _render_grid_layer(
        self,
        surf,
        font,
        y: int,
        layer: "GridViewLayer",
        tree: "BeatTree",
        div: int,
        meter_num: float,
        pw: int,
        cell_h: int = 16,
        extra=None,
        frac_beat_mode: str = "warp",
    ) -> tuple[list[dict], int]:
        """
        Render one grid layer at vertical offset *y* using *tree*.

        Returns (cell_rects, total_h) where each rect dict contains:
            rect, node_id, leaf, depth, tree_mode=True, value, layer_name
        Always uses the tree layout — the caller provides the BeatTree.
        """
        cells, total_h = self._tree_step_layout(y, tree, meter_num, pw, cell_h=cell_h,
                                                 frac_beat_mode=frac_beat_mode)

        rects: list[dict] = []
        for cell in cells:
            cr = cell["rect"]
            leaf    = cell["leaf"]
            depth   = cell["depth"]
            group   = leaf.group
            value   = layer.read_fn(leaf, None, extra)

            is_beat = bool(cell.get("beat_start", False))
            bg  = layer.bg_fn(value, is_beat, depth, group)
            brd = layer.brd_fn(value, is_beat, depth, group)

            pygame.draw.rect(surf, bg, cr, border_radius=2)
            pygame.draw.rect(surf, brd, cr, 1, border_radius=2)

            # Group dot
            if layer.show_groups and group:
                gc = GROUP_COLORS[min(group, len(GROUP_COLORS) - 1)]
                pygame.draw.circle(surf, gc, (cr.x + 5, cr.y + 5), 3)

            # Depth tick marks
            if layer.show_depth and depth > 0:
                for di in range(min(depth, 4)):
                    tx = cr.x + 2 + di * 3
                    pygame.draw.line(surf, (140, 110, 200),
                                     (tx, cr.bottom - 3), (tx, cr.bottom - 1))

            # Cell label (left-aligned)
            if layer.label_fn is not None:
                lbl_info = layer.label_fn(value)
                if lbl_info is not None:
                    lbl_text, lbl_col = lbl_info
                    ls = font.render(lbl_text, True, lbl_col)
                    surf.blit(ls, (cr.x + 2, cr.y + 2))

            # Cell label (right-aligned)
            if layer.label_right_fn is not None:
                rl_info = layer.label_right_fn(value)
                if rl_info is not None:
                    rl_text, rl_col = rl_info
                    rs = font.render(rl_text, True, rl_col)
                    surf.blit(rs, (cr.right - rs.get_width() - 2, cr.y + 1))

            rd = {"rect": cr, "tree_mode": True, "value": value,
                  "layer_name": layer.name,
                  "node_id": cell["node_id"],
                  "leaf": leaf, "depth": depth}
            rects.append(rd)

        # Group merge bars (only when layer shows groups)
        if layer.show_groups and rects:
            _grp_runs: list[tuple[int, int, int]] = []
            for cell_dict in rects:
                lf  = cell_dict.get("leaf")
                grp = lf.group if lf else 0
                if lf and lf.on and grp:
                    cr = cell_dict["rect"]
                    if _grp_runs and _grp_runs[-1][0] == grp:
                        _grp_runs[-1] = (_grp_runs[-1][0], _grp_runs[-1][1], cr.right)
                    else:
                        _grp_runs.append((grp, cr.x, cr.right))
            first_w = rects[0]["rect"].width if rects else 0
            for grp, x0, x1 in _grp_runs:
                if x1 - x0 > first_w + 2:
                    gc3 = GROUP_COLORS[min(grp, len(GROUP_COLORS) - 1)]
                    bar_y = rects[0]["rect"].y + rects[0]["rect"].height // 2
                    pygame.draw.line(surf, gc3, (x0 + 2, bar_y), (x1 - 2, bar_y), 2)

        return rects, total_h

    def _handle_grid_layer_click(
        self,
        rects: list[dict],
        lx: int,
        ly: int,
        layer: "GridViewLayer",
        extra=None,
    ) -> bool:
        """Test a left-click against stored rects for *layer*.  Returns True if handled."""
        for rd in rects:
            if rd["rect"].collidepoint(lx, ly):
                if layer.click_fn is not None:
                    leaf = rd.get("leaf")
                    layer.click_fn(leaf, None, extra)
                return True
        return False

    def _handle_grid_layer_right_click(
        self,
        rects: list[dict],
        lx: int,
        ly: int,
        layer: "GridViewLayer",
        tree: "BeatTree",
        div: int,
    ) -> bool:
        """Test right-click; opens context menu via layer callback. Returns True if handled."""
        if layer.right_click_fn is None:
            return False
        for rd in rects:
            if rd["rect"].collidepoint(lx, ly):
                result = layer.right_click_fn(rd, lx, ly, tree, div)
                if result is not None:
                    self._rhythm_ctx_menu = result
                return True
        return False

    def _apply_seq_arrow(self, key: str, direction: int) -> None:
        """Cycle a picker field on the patch by +1 or -1 step."""
        p = self._patch
        if p is None:
            return
        if key == "seq_scale":
            names = _SEQ_SCALE_NAMES
            cur   = names.index(p.seq_scale) if p.seq_scale in names else 0
            p.seq_scale = names[(cur + direction) % len(names)]
        elif key == "seq_pattern_idx":
            p.seq_pattern_idx = max(0, min(len(_SEQ_PATTERN_NAMES) - 1,
                                           p.seq_pattern_idx + direction))
        elif key == "seq_chord_prog":
            names = _SEQ_CHORD_NAMES
            cur   = names.index(p.seq_chord_prog) if p.seq_chord_prog in names else 0
            p.seq_chord_prog = names[(cur + direction) % len(names)]
        elif key == "seq_rubato_shape":
            names = _SEQ_RUBATO_SHAPES
            cur = names.index(p.seq_rubato_shape) if p.seq_rubato_shape in names else 0
            p.seq_rubato_shape = names[(cur + direction) % len(names)]
        elif key == "seq_rubato_scope":
            names = _SEQ_RUBATO_SCOPES
            cur = names.index(p.seq_rubato_scope) if p.seq_rubato_scope in names else 0
            p.seq_rubato_scope = names[(cur + direction) % len(names)]
        elif key == "seq_octave_span":
            p.seq_octave_span = max(1, min(5, p.seq_octave_span + direction))
        elif key == "seq_bass_octave":
            p.seq_bass_octave = max(-4, min(0, p.seq_bass_octave + direction))
        elif key == "seq_root_octave":
            p.seq_root_octave = max(-4, min(0, p.seq_root_octave + direction))
        elif key == "seq_stab_octave":
            p.seq_stab_octave = max(-2, min(4, p.seq_stab_octave + direction))
        elif key == "seq_tonic_hz":
            cur_st = p.tuning.hz_to_semitones(p.seq_tonic_hz)
            p.seq_tonic_hz = p.tuning.semitone_to_hz(round(cur_st) + direction)
        elif key == "seq_repeats":
            p.seq_repeats = max(1, min(16, p.seq_repeats + direction))

    def _apply_seq_slider(self, sl: dict, lx: int) -> None:
        """Map a mouse x position to the slider value and write it to the patch."""
        p = self._patch
        if p is None:
            return
        r = sl["rect"]
        frac = max(0.0, min(1.0, (lx - r.x) / max(r.w, 1)))
        val  = sl["lo"] + frac * (sl["hi"] - sl["lo"])
        _mods = pygame.key.get_mods()
        _shift_free = bool(_mods & KMOD_SHIFT)
        _ctrl_irr = bool(_mods & KMOD_CTRL)
        sl["val"] = val
        key = sl["key"]
        if key == "seq_bpm":
            p.seq_bpm = val
        elif key == "seq_legato":
            p.seq_legato = val
        elif key == "seq_portamento_s":
            p.seq_portamento_s = val
        elif key == "seq_rubato_amount":
            p.seq_rubato_amount = max(0.0, min(0.95, val))
        elif key == "meter_numerator":
            if _ctrl_irr:
                _opts = [v for v in _METER_IRRATIONAL_SNAPS if sl["lo"] <= v <= sl["hi"] and v < 10.0]
                _meter_val = min(_opts, key=lambda v: abs(v - val)) if _opts else val
            else:
                _meter_val = val if _shift_free else round(val)
            p.meter_numerator = max(sl["lo"], min(sl["hi"], _meter_val))
            sl["val"] = p.meter_numerator
        elif key == "meter_denominator":
            if _ctrl_irr:
                _opts = [v for v in _METER_IRRATIONAL_SNAPS if sl["lo"] <= v <= sl["hi"] and v < 10.0]
                _meter_val = min(_opts, key=lambda v: abs(v - val)) if _opts else val
            else:
                _meter_val = val if _shift_free else round(val)
            p.meter_denominator = max(sl["lo"], min(sl["hi"], _meter_val))
            sl["val"] = p.meter_denominator

    def _apply_rhythm_slider(self, sl: dict, lx: int) -> None:
        """Write rhythm slider value to the active page target."""
        p = self._patch
        if p is None:
            return
        r    = sl["rect"]
        frac = max(0.0, min(1.0, (lx - r.x) / max(r.w, 1)))
        val  = sl["lo"] + frac * (sl["hi"] - sl["lo"])
        sl["val"] = val
        _rw_key = p.rhythm_active_page
        _rw = (p.rhythm_pages[_rw_key]
               if (_rw_key != "all" and _rw_key in p.rhythm_pages)
               else p)
        key = sl["key"]
        if key == "rhythm_swing":
            _rw.rhythm_swing  = val
        elif key == "rhythm_pocket":
            _rw.rhythm_pocket = val
        elif key == "rhythm_gate":
            _rw.rhythm_gate   = val
        elif key == "meter_numerator":
            # Skip write if the page is set to inherit global meter (0.0 = inherit)
            _rw_key_m = p.rhythm_active_page
            if (_rw_key_m != "all" and _rw_key_m in p.rhythm_pages
                    and float(p.rhythm_pages[_rw_key_m].meter_numerator) == 0.0):
                return
            _mods = pygame.key.get_mods()
            _shift_free = bool(_mods & KMOD_SHIFT)
            _ctrl_irr = bool(_mods & KMOD_CTRL)
            if _ctrl_irr:
                _opts = [v for v in _METER_IRRATIONAL_SNAPS if sl["lo"] <= v <= sl["hi"] and v < 10.0]
                _meter_val = min(_opts, key=lambda v: abs(v - val)) if _opts else val
            else:
                _meter_val = val if _shift_free else round(val)
            _rw.meter_numerator = max(sl["lo"], min(sl["hi"], _meter_val))
            sl["val"] = _rw.meter_numerator
        elif key == "meter_denominator":
            _mods = pygame.key.get_mods()
            _shift_free = bool(_mods & KMOD_SHIFT)
            _ctrl_irr = bool(_mods & KMOD_CTRL)
            if _ctrl_irr:
                _opts = [v for v in _METER_IRRATIONAL_SNAPS if sl["lo"] <= v <= sl["hi"] and v < 10.0]
                _meter_val = min(_opts, key=lambda v: abs(v - val)) if _opts else val
            else:
                _meter_val = val if _shift_free else round(val)
            _rw.meter_denominator = max(sl["lo"], min(sl["hi"], _meter_val))
            sl["val"] = _rw.meter_denominator

    def _apply_prob_slider(self, sl: dict, lx: int) -> None:
        """Write a probability slider value (0.0–1.0) to seq_probabilities."""
        p = self._patch
        if p is None:
            return
        r    = sl["rect"]
        frac = max(0.0, min(1.0, (lx - r.x) / max(r.w, 1)))
        val  = sl["lo"] + frac * (sl["hi"] - sl["lo"])
        sl["val"] = val
        key = sl["key"]
        sp  = getattr(p, "seq_probabilities", None)
        if sp is not None:
            setattr(sp, key, val)

    def _apply_dyn_slider(self, sl: dict, lx: int) -> None:
        """Write a dynamics intensity slider value to the DynamicsProgram curve."""
        p = self._patch
        if p is None:
            return
        r    = sl["rect"]
        frac = max(0.0, min(1.0, (lx - r.x) / max(r.w, 1)))
        val  = sl["lo"] + frac * (sl["hi"] - sl["lo"])
        sl["val"] = val
        dp = p.ensure_dynamics_page(self._dyn_page_key)
        if dp is not None:
            dp.curve.intensity = val

    def _apply_improv_slider(self, sl: dict, lx: int) -> None:
        """Write an improv slider value to the ImprovProgram / sub-params."""
        p = self._patch
        if p is None:
            return
        r    = sl["rect"]
        frac = max(0.0, min(1.0, (lx - r.x) / max(r.w, 1)))
        val  = sl["lo"] + frac * (sl["hi"] - sl["lo"])
        sl["val"] = val
        ip = p.ensure_improv_page(self._improv_page_key)
        if ip is None:
            return
        key = sl["key"]
        if key == "prob_grace":
            ip.prob_grace = val
        elif key == "prob_chirp":
            ip.prob_chirp = val
        elif key == "prob_echo":
            ip.prob_echo = val
        elif key == "grace_dur_frac":
            ip.grace.duration_frac = val
        elif key == "grace_vel_scale":
            ip.grace.vel_scale = val
        elif key == "chirp_dur_frac":
            ip.chirp.duration_frac = val
        elif key == "chirp_vel_scale":
            ip.chirp.vel_scale = val
        elif key == "echo_dur_frac":
            ip.echo.duration_frac = val
        elif key == "echo_vel_fall":
            ip.echo.vel_falloff = val

    def render(self) -> pygame.Surface | None:
        if self._patch is None:
            return None
        pw   = self.PANEL_W
        font = self._font_()
        fh   = font.get_height()
        p    = self._patch

        items: list[tuple[str, str, list[int], bool]] = []
        # Pinned Patch entry always first
        items.append(("__patch__", "\u25c6 Patch", [80, 130, 200], False))
        items.append(("__system__", "\u2699 System Device", [200, 150, 90], False))
        for v in p.voices:
            em_tag = " [G]" if getattr(v, "emission_mode", "single") == "granular" else ""
            items.append((v.key, f"\u25b6 {v.label}{em_tag}  {v.freq_hz:.1f}Hz", v.color, v.muted))
        for l in p.lfos:
            items.append((l.key, f"~ {l.label}  {l.rate_hz:.2f}Hz", l.color, False))
        # Module nodes — ⬡ icon, mutable
        for mod in p.modules:
            mt_tag = f"[{mod.module_type}]"
            if mod.module_type == "lfo":
                if mod.lfo_channels:
                    mt_tag = f"~ {len(mod.lfo_channels)}ch"
                else:
                    mt_tag = f"~ {mod.rate_hz:.2f}Hz"
            mute_tag = " \u25a0" if mod.muted else ""
            items.append((mod.key, f"\u2B21 {mod.label}  {mt_tag}{mute_tag}", mod.color, mod.muted))
        # Control surfaces — ⊞ icon, N sliders shown in count
        for cs in p.controls:
            n_sl = len(cs.sliders)
            sl_tag = f"[{n_sl} sl]" if n_sl != 1 else "[1 sl]"
            items.append((cs.key, f"\u229e {cs.label}  {sl_tag}", cs.color, False))
        # Router instances — selectable editor items for the torch routing model
        for rt in getattr(p, "routers", []):
            rt_icon = {
                "voice_router": "\u21c9",
                "instrument": "\u266b",
                "master": "\u25c9",
            }.get(rt.router_type, "\u25c8")
            rt_col = {
                "voice_router": [90, 160, 230],
                "instrument": [110, 200, 150],
                "master": [230, 180, 90],
            }.get(rt.router_type, [160, 160, 180])
            items.append((
                _router_ui_key(rt.key),
                f"{rt_icon} {rt.label}  [{rt.router_type}]",
                rt_col,
                False,
            ))
        # Mixer nodes — ⊕ = projection active (output track), ○ = meta-mixer only
        for m in p.mixers:
            icon = "\u229e" if m.projection_active else "\u25cb"
            items.append((m.key, f"{icon} {m.label}", m.color, False))
        # Param nodes — ★ icon, purple-ish; show count of targets
        for pn in p.param_nodes:
            n_tgt = len(pn.targets)
            first_attr = pn.targets[0].get("attr", "") if pn.targets else ""
            tgt = f"\u2192{first_attr}" if first_attr else ""
            if n_tgt > 1:
                tgt += f"+{n_tgt - 1}"
            items.append((pn.key, f"\u2605 {pn.label}{tgt}", pn.color, False))

        row_h  = fh + 10
        btn_h  = fh + 8
        hdr_h  = fh + 6
        seq_row_h = fh + 8

        # Heights
        v_rows_h = 0 if self._voices_collapsed else len(items) * row_h
        v_btns_h = 0 if self._voices_collapsed else btn_h + 8
        voices_sec_h = hdr_h + v_rows_h + v_btns_h + 4

        seq_body_h = 0
        if not self._seq_collapsed and _HAS_SEQ_ENG:
            seq_body_h = seq_row_h * 11 + 14 + 32 * 6 + btn_h + 8  # pickers + label-clearance + sliders + btns
        seq_sec_h = hdr_h + seq_body_h + 4

        prob_body_h = 0
        if not self._prob_collapsed:
            prob_body_h = 14 + 32 * 4 + 8  # label-clearance + 4 probability sliders
        prob_sec_h = hdr_h + prob_body_h + 4

        dyn_body_h = 0
        if not self._dyn_collapsed and _HAS_DYN_ENG:
            _dyn_pg     = p.page_for(self._dyn_page_key)
            _rdiv_d    = getattr(_dyn_pg, "rhythm_division", 16)
            _cell_h_d    = 16
            _grid_rows_d = self._metric_grid_rows(
                p.page_meter(_dyn_pg)[0],
                max(1, pw - 12),
                frac_beat_mode=getattr(_dyn_pg, "frac_beat_mode", "warp"),
            )
            dyn_body_h = (
                (hdr_h + 2)                              # part selector
                + (hdr_h + 4)                            # curve shape picker
                + (hdr_h + 4)                            # scope stepper
                + (hdr_h + 4)                            # auto accent button
                + (32 + 8)                               # intensity slider
                + (_grid_rows_d * (_cell_h_d + 2) + 4)  # accent grid
            )
        dyn_sec_h = hdr_h + dyn_body_h + 4

        improv_body_h = 0
        if not self._improv_collapsed and _HAS_IMPROV_ENG:
            _improv_pg    = p.page_for(self._improv_page_key)
            _rdiv_i      = getattr(_improv_pg, "rhythm_division", 16)
            _cell_h_i    = 16
            _irows       = self._metric_grid_rows(
                p.page_meter(_improv_pg)[0],
                max(1, pw - 12),
                frac_beat_mode=getattr(_improv_pg, "frac_beat_mode", "warp"),
            )
            # 3 prob sliders + 3 sub-section headers (collapsed by default)
            # + improv step grid
            _sub_grace_h = 0 if self._improv_grace_collapsed else (
                (hdr_h + 4)    # mode picker
                + (hdr_h + 4)  # position picker
                + (32 + 8)     # duration_frac slider
                + (32 + 8)     # vel_scale slider
                + hdr_h        # trim_main toggle row
            )
            _sub_chirp_h = 0 if self._improv_chirp_collapsed else (
                (hdr_h + 4)    # shape picker
                + (hdr_h + 4)  # mode picker
                + (hdr_h + 4)  # steps stepper
                + (32 + 8)     # duration_frac slider
                + (32 + 8)     # vel_scale slider
            )
            _sub_echo_h = 0 if self._improv_echo_collapsed else (
                (hdr_h + 4)    # lookback stepper
                + (32 + 8)     # duration_frac slider
                + (32 + 8)     # vel_falloff slider
                + (hdr_h + 4)  # max_notes stepper
            )
            improv_body_h = (
                (hdr_h + 2)                              # part selector
                + (32 * 3 + 8)                           # 3 prob sliders
                + hdr_h + _sub_grace_h + 4               # grace sub-section
                + hdr_h + _sub_chirp_h + 4               # chirp sub-section
                + hdr_h + _sub_echo_h + 4                # echo sub-section
                + (_irows * (_cell_h_i + 2) + 4)         # step grid
            )
        improv_sec_h = hdr_h + improv_body_h + 4

        rhythm_body_h = 0
        if not self._rhythm_collapsed:
            _active_pg = p.page_for(p.rhythm_active_page)
            _rdiv      = getattr(_active_pg, "rhythm_division", getattr(p, "rhythm_division", 16))
            _cell_h    = 16
            _grid_rows = self._metric_grid_rows(
                p.page_meter(_active_pg)[0],
                max(1, pw - 12),
                frac_beat_mode=getattr(_active_pg, "frac_beat_mode", "warp"),
            )
            rhythm_body_h = (
                (hdr_h + 2)                 # division picker row
                + (hdr_h + 2)               # page selector row
                + (hdr_h + 4)               # stress row
                + (hdr_h + 4)               # pattern tabs row
                + (_grid_rows * (_cell_h + 2) + 4)   # step grid
                + (hdr_h + 8)               # phrase row
                + (hdr_h + 6)               # prog-bars + fit-mode row
                + 14                         # label-clearance for first slider
                + (32 * 5 + 8)              # 5 sliders (meter/swing/pocket/gate)
            )
        rhythm_sec_h = hdr_h + rhythm_body_h + 4

        # Placement section: always rendered (collapsed = header only; expanded = header + rows)
        _plc_n_rows = 0 if self._placement_collapsed else max(1, len(p.parts))
        plc_sec_h = (hdr_h + 2) + _plc_n_rows * (hdr_h + 2)

        total_h = (4 + voices_sec_h + 8 + seq_sec_h + 8 + prob_sec_h + 8
                   + dyn_sec_h + 8 + improv_sec_h + 8 + rhythm_sec_h + 8
                   + plc_sec_h + 8)
        surf = pygame.Surface((pw, max(200, total_h)))
        surf.fill(_PY_BG)

        y = 4

        # ── Voices header ────────────────────────────────────────────────────
        arrow_v = "\u25bc" if not self._voices_collapsed else "\u25b6"
        pygame.draw.rect(surf, (28, 44, 60), (0, y, pw, hdr_h))
        pygame.draw.line(surf, (60, 100, 140), (0, y), (pw, y))
        surf.blit(font.render(f"{arrow_v} Voices  [{max(0, len(items) - 2)}]",
                              True, (140, 190, 230)), (8, y + 3))
        self._voices_hdr_rect = pygame.Rect(0, y, pw, hdr_h)
        y += hdr_h
        self._voices_row_start_y = y
        self._items  = items
        self._row_h  = row_h

        if not self._voices_collapsed:
            for key, label, col, muted in items:
                is_active = (key == self._active)
                is_mix    = self._patch is not None and any(m.key == key for m in self._patch.mixers)
                is_router = self._patch is not None and _router_instance_for_active_key(self._patch, key) is not None
                bg = (40, 60, 80) if is_active else (24, 24, 30)
                if key in {"__patch__", "__system__"}:
                    bg = (50, 80, 120) if is_active else (28, 40, 60)
                elif is_mix:
                    bg = (44, 44, 22) if is_active else (30, 28, 18)
                elif is_router:
                    bg = (26, 46, 64) if is_active else (18, 28, 38)
                pygame.draw.rect(surf, bg, (2, y, pw - 4, row_h - 2), border_radius=3)
                c = tuple(col[:3]) if col else (120, 180, 255)
                pygame.draw.rect(surf, c, (2, y, 4, row_h - 2), border_radius=2)
                txt_col = (200, 200, 210) if not muted else (80, 80, 90)
                lbl = font.render(label, True, txt_col)
                surf.blit(lbl, (10, y + (row_h - 2 - fh) // 2))
                if not is_mix and not is_router and key not in {"__patch__", "__system__"}:
                    is_param_  = any(pn.key == key for pn in self._patch.param_nodes) if self._patch else False
                    is_lfo_    = any(l.key == key for l in self._patch.lfos) if self._patch else False
                    is_module_ = any(m.key == key for m in self._patch.modules) if self._patch else False
                    is_ctrl_   = any(cs.key == key for cs in self._patch.controls) if self._patch else False
                    # Solo button (voices and modules only)
                    if not is_lfo_ and not is_param_ and not is_ctrl_:
                        is_solo = (getattr(self._patch, "solo_key", None) == key)
                        sc = (210, 170, 20) if is_solo else (42, 42, 52)
                        pygame.draw.rect(surf, sc, (pw - 60, y + 3, 14, row_h - 8), border_radius=2)
                        stc = (245, 245, 60) if is_solo else (110, 110, 122)
                        surf.blit(font.render("S", True, stc), (pw - 59, y + 3))
                    # Mute button (voices, LFOs, modules — not param/ctrl/mixer)
                    if not is_param_ and not is_ctrl_:
                        mc = (200, 80, 60) if muted else (50, 50, 60)
                        pygame.draw.rect(surf, mc, (pw - 42, y + 3, 16, row_h - 8), border_radius=2)
                        surf.blit(font.render("M", True, (200, 200, 200)), (pw - 40, y + 3))
                    # Delete button
                    pygame.draw.rect(surf, (90, 40, 40), (pw - 24, y + 3, 16, row_h - 8), border_radius=2)
                    surf.blit(font.render("\xd7", True, (220, 120, 100)), (pw - 21, y + 3))
                y += row_h
            # Add buttons — row 1: Voice / Param  |  row 2: Module▾ / Control▾
            y += 4
            half = pw // 2
            pygame.draw.rect(surf, (35, 70, 100),  (4,        y, half - 8, btn_h), border_radius=3)
            surf.blit(font.render("+ Voice",  True, (160, 200, 240)), (8,        y + 2))
            pygame.draw.rect(surf, (60, 35, 90),   (half + 4, y, half - 8, btn_h), border_radius=3)
            surf.blit(font.render("+ Param",  True, (210, 170, 255)), (half + 8, y + 2))
            self._btn_y = y
            self._btn_h = btn_h
            y += btn_h + 4
            mod_lbl  = "+ Module \u25be" + (" [open]" if self._open_dropdown == "module" else "")
            ctrl_lbl = "+ Control \u25be" + (" [open]" if self._open_dropdown == "control" else "")
            mod_bg  = (100, 60, 130) if self._open_dropdown == "module"  else (80, 50, 100)
            ctrl_bg = (30, 120, 95)  if self._open_dropdown == "control" else (30, 100, 80)
            pygame.draw.rect(surf, mod_bg,   (4,        y, half - 8, btn_h), border_radius=3)
            surf.blit(font.render(mod_lbl,  True, (220, 160, 255)), (8,        y + 2))
            pygame.draw.rect(surf, ctrl_bg,  (half + 4, y, half - 8, btn_h), border_radius=3)
            surf.blit(font.render(ctrl_lbl, True, (140, 230, 180)), (half + 8, y + 2))
            self._btn_module_rect  = pygame.Rect(4,        y, half - 8, btn_h)
            self._btn_control_rect = pygame.Rect(half + 4, y, half - 8, btn_h)
            self._btn2_y = y
            self._btn2_h = btn_h
            y += btn_h + 4
            # Register dropdown item rects for hit-testing; actual draw happens
            # at the end of render() as an overlay so it covers sections below.
            if self._open_dropdown in ("module", "control"):
                drop_x = 4 if self._open_dropdown == "module" else half + 4
                drop_w = half - 8
                items_data = (
                    [("LFO",           "lfo"),
                     ("Passthrough",   "passthrough"),
                     ("Pitch Quant.",  "pitch_quantizer"),
                     ("Interaural",    "interaural"),
                     ("Voice Router",  "router:voice_router"),
                     ("Instrument Router", "router:instrument"),
                     ("Master Router", "router:master")]
                    if self._open_dropdown == "module"
                    else [("Control Surface", "control_surface"),
                          ("Voice Router", "router:voice_router"),
                          ("Instrument Router", "router:instrument"),
                          ("Master Router", "router:master")]
                )
                self._dropdown_items = []
                dy = y
                for dlabel, ddata in items_data:
                    dr = pygame.Rect(drop_x, dy, drop_w, btn_h)
                    self._dropdown_items.append(dict(label=dlabel, rect=dr, data=ddata))
                    dy += btn_h + 2

        y += 8

        # ── Sequence header ──────────────────────────────────────────────────
        arrow_s = "\u25bc" if not self._seq_collapsed else "\u25b6"
        pygame.draw.rect(surf, (28, 48, 34), (0, y, pw, hdr_h))
        pygame.draw.line(surf, (60, 140, 80), (0, y), (pw, y))
        surf.blit(font.render(f"{arrow_s} Sequence",
                              True, (140, 230, 160)), (8, y + 3))
        self._seq_hdr_rect = pygame.Rect(0, y, pw, hdr_h)
        y += hdr_h

        seq_sliders: list[dict] = []
        seq_arrows:  list[dict] = []

        if not self._seq_collapsed and _HAS_SEQ_ENG:

            def _picker_row(surf_: pygame.Surface, y_: int,
                            label_: str, value_str_: str,
                            key_: str) -> int:
                pygame.draw.rect(surf_, (26, 30, 38), (2, y_, pw - 4, seq_row_h - 2))
                surf_.blit(font.render(label_, True, _PY_DIM), (8, y_ + 2))
                vw = font.size(value_str_)[0]
                surf_.blit(font.render(value_str_, True, _PY_TXT),
                           (pw // 2 - vw // 2, y_ + 2))
                lb = pygame.Rect(pw - 46, y_ + 1, 19, seq_row_h - 4)
                rb = pygame.Rect(pw - 25, y_ + 1, 19, seq_row_h - 4)
                pygame.draw.rect(surf_, (50, 55, 70), lb, border_radius=2)
                pygame.draw.rect(surf_, (50, 55, 70), rb, border_radius=2)
                surf_.blit(font.render("\u25c4", True, (160, 200, 220)), (lb.x + 3, lb.y + 1))
                surf_.blit(font.render("\u25ba", True, (160, 200, 220)), (rb.x + 3, rb.y + 1))
                seq_arrows.append(dict(key=key_, left_r=lb, right_r=rb))
                return y_ + seq_row_h

            # Tonic picker (musical key — scale root, separate from tuning reference)
            y = _picker_row(surf, y, "Tonic", _hz_to_note_name(p.seq_tonic_hz, p.tuning), "seq_tonic_hz")

            # Scale picker
            sn = p.seq_scale if p.seq_scale in _SEQ_SCALE_NAMES else _SEQ_SCALE_NAMES[0]
            y = _picker_row(surf, y, "Scale", sn, "seq_scale")

            # Pattern picker
            pi = max(0, min(len(_SEQ_PATTERN_NAMES) - 1, p.seq_pattern_idx))
            y = _picker_row(surf, y, "Pattern", _SEQ_PATTERN_NAMES[pi], "seq_pattern_idx")

            # Chord progression picker
            cn = (p.seq_chord_prog if p.seq_chord_prog in _SEQ_CHORD_NAMES
                  else _SEQ_CHORD_NAMES[0])
            y = _picker_row(surf, y, "Chord", cn, "seq_chord_prog")

            # Octave span picker
            y = _picker_row(surf, y, "Octaves", str(p.seq_octave_span), "seq_octave_span")
            # Arrangement octave pickers (bass / root / stab)
            y = _picker_row(surf, y, "Bass oct",  str(p.seq_bass_octave), "seq_bass_octave")
            y = _picker_row(surf, y, "Root oct",  str(p.seq_root_octave), "seq_root_octave")
            y = _picker_row(surf, y, "Stab oct",  str(p.seq_stab_octave), "seq_stab_octave")

            # Repeats picker
            y = _picker_row(surf, y, "Repeats", str(p.seq_repeats), "seq_repeats")

            y = _picker_row(surf, y, "Rubato", p.seq_rubato_shape, "seq_rubato_shape")
            y = _picker_row(surf, y, "Rubato Scope", p.seq_rubato_scope, "seq_rubato_scope")

            def _fmt_meter(v: float) -> str:
                if abs(v - round(v)) < 1e-6:
                    return str(int(round(v)))
                return f"{v:.2f}".rstrip("0").rstrip(".")

            meter_hint = "Shift=free Ctrl=irr"
            hint_s = font.render(meter_hint, True, (98, 145, 112))
            y += 14  # label clearance: ensure first slider label clears the last picker row
            surf.blit(hint_s, (pw - hint_s.get_width() - 8, y - 13))
            y = self._add_seq_slider(seq_sliders, y, "Meter Num",
                                     "meter_numerator", p.meter_numerator, 1.0, 16.0, ".2f")
            y = self._add_seq_slider(seq_sliders, y, "Meter Den",
                                     "meter_denominator", p.meter_denominator, 1.0, 16.0, ".2f")
            # BPM slider
            y = self._add_seq_slider(seq_sliders, y, "BPM",
                                     "seq_bpm", p.seq_bpm, 30.0, 240.0, ".1f")
            # Legato slider
            y = self._add_seq_slider(seq_sliders, y, "Legato",
                                     "seq_legato", p.seq_legato, 0.05, 1.0, ".2f")
            # Portamento slider
            y = self._add_seq_slider(seq_sliders, y, "Portamento (s)",
                                     "seq_portamento_s", p.seq_portamento_s, 0.0, 2.0, ".3f")
            y = self._add_seq_slider(seq_sliders, y, "Rubato Amt",
                                     "seq_rubato_amount", p.seq_rubato_amount, 0.0, 0.95, ".2f")

            # Draw seq sliders
            for sl in seq_sliders:
                r = sl["rect"]
                lbl_y = r.y - 14
                surf.blit(font.render(sl["label"], True, _PY_DIM), (8, lbl_y))
                if sl["key"] in {"meter_numerator", "meter_denominator"}:
                    val_str = _fmt_meter(sl["val"])
                else:
                    val_str = format(sl["val"], sl["fmt"])
                vs = font.render(val_str, True, _PY_TXT)
                surf.blit(vs, (pw - vs.get_width() - 8, lbl_y))
                pygame.draw.rect(surf, (40, 40, 50), r, border_radius=3)
                frac = max(0.0, min(1.0,
                           (sl["val"] - sl["lo"]) / max(sl["hi"] - sl["lo"], 1e-9)))
                thumb_x = r.x + int(frac * r.w)
                pygame.draw.rect(surf, _PY_ACT,
                                 pygame.Rect(r.x, r.y, thumb_x - r.x + 4, r.h),
                                 border_radius=3)

            y += 4
            # Action buttons — Deploy / Demo / Rend.Fund / Rend.Seq in equal quarters
            quarter = max(1, (pw - 18) // 4)
            dep_r  = pygame.Rect(4,                      y, quarter, btn_h)
            dem_r  = pygame.Rect(4 + (quarter + 2),      y, quarter, btn_h)
            rfnd_r = pygame.Rect(4 + 2 * (quarter + 2),  y, quarter, btn_h)
            rseq_r = pygame.Rect(4 + 3 * (quarter + 2),  y, pw - 6 - 3 * (quarter + 2), btn_h)
            pygame.draw.rect(surf, (40, 70, 40),  dep_r,  border_radius=3)
            pygame.draw.rect(surf, (55, 35, 75),  dem_r,  border_radius=3)
            pygame.draw.rect(surf, (90, 25, 25),  rfnd_r, border_radius=3)
            pygame.draw.rect(surf, (90, 25, 25),  rseq_r, border_radius=3)
            surf.blit(font.render("\u266b Chord",   True, (150, 230, 150)), (dep_r.x + 3,  y + 2))
            surf.blit(font.render("\u25b6 Demo",    True, (190, 150, 240)), (dem_r.x + 3,  y + 2))
            surf.blit(font.render("\u23fa Fund.",   True, (255, 120, 120)), (rfnd_r.x + 3, y + 2))
            surf.blit(font.render("\u23fa Seq.",    True, (255, 120, 120)), (rseq_r.x + 3, y + 2))
            self._seq_btn_rects = {"deploy": dep_r, "demo": dem_r,
                                   "render_fund": rfnd_r, "render": rseq_r}
            y += btn_h + 4

        self._seq_sliders = seq_sliders
        self._seq_arrows  = seq_arrows

        y += 8

        # ── Probabilities header ─────────────────────────────────────────────
        arrow_pb = "\u25bc" if not self._prob_collapsed else "\u25b6"
        pygame.draw.rect(surf, (28, 44, 40), (0, y, pw, hdr_h))
        pygame.draw.line(surf, (60, 130, 110), (0, y), (pw, y))
        surf.blit(font.render(f"{arrow_pb} Probabilities", True, (130, 220, 190)),
                  (8, y + 3))
        self._prob_hdr_rect = pygame.Rect(0, y, pw, hdr_h)
        y += hdr_h

        prob_sliders: list[dict] = []
        if not self._prob_collapsed:
            sp = getattr(p, "seq_probabilities", None)
            if sp is None:
                sp = SequenceProbabilities()
            _PROB_LABELS = [
                ("Double Back",  "double_back",  sp.double_back),
                ("Subversion",   "subversion",   sp.subversion),
                ("Chromatic",    "chromatic",    sp.chromatic),
                ("Modal Color",  "modal",        sp.modal),
            ]
            _PR_ACT = (52, 138, 108)
            y += 14  # label clearance: ensure first slider label clears the section header
            for lbl, key, val in _PROB_LABELS:
                y = self._add_seq_slider(prob_sliders, y, lbl, key, val, 0.0, 1.0, ".2f")
            for sl in prob_sliders:
                r = sl["rect"]
                lbl_y = r.y - 14
                surf.blit(font.render(sl["label"], True, _PY_DIM), (8, lbl_y))
                vs = font.render(format(sl["val"], sl["fmt"]), True, _PY_TXT)
                surf.blit(vs, (pw - vs.get_width() - 8, lbl_y))
                pygame.draw.rect(surf, (32, 48, 44), r, border_radius=3)
                frac = max(0.0, min(1.0,
                           (sl["val"] - sl["lo"]) / max(sl["hi"] - sl["lo"], 1e-9)))
                thumb_x = r.x + int(frac * r.w)
                pygame.draw.rect(surf, _PR_ACT,
                                 pygame.Rect(r.x, r.y, thumb_x - r.x + 4, r.h),
                                 border_radius=3)
            y += 4

        self._prob_sliders = prob_sliders

        y += 8

        # ── Dynamics header ──────────────────────────────────────────────────
        dp          = p.dynamics_for(self._dyn_page_key)
        arrow_dy    = "\u25bc" if not self._dyn_collapsed else "\u25b6"
        dy_en_str   = "[ON]" if (dp and dp.enabled) else "[off]"
        pygame.draw.rect(surf, (26, 42, 50), (0, y, pw, hdr_h))
        pygame.draw.line(surf, (55, 135, 165), (0, y), (pw, y))
        dy_en_col = (38, 95, 125) if (dp and dp.enabled) else (28, 48, 62)
        dy_en_r   = pygame.Rect(pw - 50, y + 2, 46, hdr_h - 4)
        pygame.draw.rect(surf, dy_en_col, dy_en_r, border_radius=3)
        surf.blit(font.render(dy_en_str, True, (140, 210, 235)), (dy_en_r.x + 4, y + 3))
        surf.blit(font.render(f"{arrow_dy} Dynamics", True, (140, 210, 235)), (8, y + 3))
        self._dyn_hdr_rect    = pygame.Rect(0, y, pw - 52, hdr_h)
        self._dyn_enable_rect = dy_en_r
        y += hdr_h

        dyn_sliders: list[dict] = []
        if not self._dyn_collapsed and _HAS_DYN_ENG and dp is not None:
            _DY_ACT = (40, 130, 170)
            y, self._dyn_page_rects = self._draw_page_selector_row(
                surf, font, y, self._dyn_page_key, p.dynamics_pages,
                active_col=(36, 72, 92),
                idle_col=(24, 38, 46),
                active_brd=(110, 185, 220),
                custom_brd=(78, 132, 160),
                idle_brd=(52, 74, 86),
                active_txt=(185, 230, 245),
                custom_txt=(150, 195, 220),
                idle_txt=(105, 132, 145),
                hdr_h=hdr_h,
            )
            _dyn_pg = p.page_for(self._dyn_page_key)
            _dyn_meter_s = font.render(_page_meter_label(p, _dyn_pg), True, (118, 175, 205))
            surf.blit(_dyn_meter_s, (pw - _dyn_meter_s.get_width() - 8, y + 3))

            # ── Curve shape picker ────────────────────────────────────────────
            pygame.draw.rect(surf, (22, 36, 44), (2, y, pw - 4, hdr_h - 2))
            surf.blit(font.render("Shape:", True, _PY_DIM), (8, y + 2))
            sh_name = dp.curve.shape if dp.curve.shape in CURVE_SHAPES else "flat"
            sw      = font.size(sh_name)[0]
            surf.blit(font.render(sh_name, True, _PY_TXT), (pw // 2 - sw // 2, y + 2))
            dy_sh_l = pygame.Rect(pw - 46, y + 1, 19, hdr_h - 4)
            dy_sh_r = pygame.Rect(pw - 25, y + 1, 19, hdr_h - 4)
            pygame.draw.rect(surf, (45, 70, 88), dy_sh_l, border_radius=2)
            pygame.draw.rect(surf, (45, 70, 88), dy_sh_r, border_radius=2)
            surf.blit(font.render("\u25c4", True, (140, 200, 220)), (dy_sh_l.x + 3, dy_sh_l.y + 1))
            surf.blit(font.render("\u25ba", True, (140, 200, 220)), (dy_sh_r.x + 3, dy_sh_r.y + 1))
            self._dyn_curve_left_rect  = dy_sh_l
            self._dyn_curve_right_rect = dy_sh_r
            y += hdr_h + 4

            # ── Scope stepper ─────────────────────────────────────────────────
            pygame.draw.rect(surf, (22, 36, 44), (2, y, pw - 4, hdr_h - 2))
            surf.blit(font.render("Scope:", True, _PY_DIM), (8, y + 2))
            _sv_idx   = min(range(len(_SCOPE_VALUES)),
                            key=lambda j: abs(_SCOPE_VALUES[j] - dp.curve.scope_bars))
            scope_lbl = _SCOPE_LABELS[_sv_idx] + " bar" + ("s" if _SCOPE_VALUES[_sv_idx] > 1 else "")
            sw2 = font.size(scope_lbl)[0]
            surf.blit(font.render(scope_lbl, True, _PY_TXT), (pw // 2 - sw2 // 2, y + 2))
            dy_sc_l = pygame.Rect(pw - 46, y + 1, 19, hdr_h - 4)
            dy_sc_r = pygame.Rect(pw - 25, y + 1, 19, hdr_h - 4)
            pygame.draw.rect(surf, (45, 70, 88), dy_sc_l, border_radius=2)
            pygame.draw.rect(surf, (45, 70, 88), dy_sc_r, border_radius=2)
            surf.blit(font.render("\u25c4", True, (140, 200, 220)), (dy_sc_l.x + 3, dy_sc_l.y + 1))
            surf.blit(font.render("\u25ba", True, (140, 200, 220)), (dy_sc_r.x + 3, dy_sc_r.y + 1))
            self._dyn_scope_dec_rect = dy_sc_l
            self._dyn_scope_inc_rect = dy_sc_r
            y += hdr_h + 4

            # ── Auto accent from stress pattern ──────────────────────────────
            da_r = pygame.Rect(6, y + 1, pw - 12, hdr_h - 2)
            pygame.draw.rect(surf, (36, 78, 104), da_r, border_radius=3)
            surf.blit(font.render("Auto Accent from Stress", True, (185, 230, 245)), (12, y + 3))
            self._dyn_auto_accent_rect = da_r
            y += hdr_h + 4

            # ── Intensity slider ──────────────────────────────────────────────
            y = self._add_seq_slider(dyn_sliders, y, "Intensity",
                                     "dyn_intensity", dp.curve.intensity, 0.0, 1.0, ".2f")
            for sl in dyn_sliders:
                r = sl["rect"]
                lbl_y = r.y - 14
                surf.blit(font.render(sl["label"], True, _PY_DIM), (8, lbl_y))
                vs = font.render(format(sl["val"], sl["fmt"]), True, _PY_TXT)
                surf.blit(vs, (pw - vs.get_width() - 8, lbl_y))
                pygame.draw.rect(surf, (28, 44, 55), r, border_radius=3)
                frac = max(0.0, min(1.0,
                           (sl["val"] - sl["lo"]) / max(sl["hi"] - sl["lo"], 1e-9)))
                thumb_x = r.x + int(frac * r.w)
                pygame.draw.rect(surf, _DY_ACT,
                                 pygame.Rect(r.x, r.y, thumb_x - r.x + 4, r.h),
                                 border_radius=3)
            y += 4

            # ── Accent grid ──────────────────────────────────────────────────
            div_d   = _dyn_pg.rhythm_division
            ch_d    = 16
            _ACCENT_COLS = [
                (28, 22, 44),    # 0.0  — muted
                (52, 42, 80),    # 0.5  — soft
                (88, 62, 145),   # 1.0  — normal
                (138, 88, 210),  # 1.5  — accent
                (195, 150, 255), # 2.0  — strong
            ]
            _ACC_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]

            # Each pattern owns its own accent tree
            _dyn_rpats = _dyn_pg.rhythm_patterns
            _dyn_act_i = min(_dyn_pg.rhythm_active_pat, max(0, len(_dyn_rpats) - 1))
            _dyn_act_pat = _dyn_rpats[_dyn_act_i] if _dyn_rpats else None

            if _dyn_act_pat is not None:
                _acc_tree = _dyn_act_pat.get_accent_tree(div_d)

                def _acc_read(leaf, _si, _ex):
                    return leaf.vel

                def _acc_bg(val, is_beat, depth, group):
                    _ci = min(range(5), key=lambda j: abs((j * 0.5) - val))
                    return _ACCENT_COLS[_ci]

                def _acc_brd(val, is_beat, depth, group):
                    bg = _acc_bg(val, is_beat, depth, group)
                    brd = tuple(min(255, c + 40) for c in bg)
                    if is_beat:
                        brd = tuple(min(255, c + 24) for c in brd)
                    return brd

                def _acc_label(val):
                    if abs(val - 1.0) > 0.01:
                        lbl = "x" + (f"{val:.1f}".rstrip("0").rstrip(".") if val != 0.0 else "0")
                        return (lbl, (210, 190, 255))
                    return None

                def _acc_click(leaf, _si, _ex):
                    cur = leaf.vel
                    # Cycle through accent levels
                    for i, lv in enumerate(_ACC_LEVELS):
                        if abs(cur - lv) < 0.01:
                            leaf.vel = _ACC_LEVELS[(i + 1) % len(_ACC_LEVELS)]
                            return
                    leaf.vel = 1.0

                _accent_layer = GridViewLayer(
                    "accent",
                    read_fn        = _acc_read,
                    bg_fn          = _acc_bg,
                    brd_fn         = _acc_brd,
                    label_fn       = _acc_label,
                    click_fn       = _acc_click,
                    right_click_fn = lambda cell, lx, ly, tr, dv: self._build_step_context_menu(cell, lx, ly, tr, dv),
                    show_depth     = True,
                )

                accent_rects, accent_h = self._render_grid_layer(
                    surf, font, y, _accent_layer, _acc_tree, div_d,
                    p.page_meter(_dyn_pg)[0], pw, cell_h=ch_d,
                    frac_beat_mode=getattr(_dyn_pg, "frac_beat_mode", "warp"))
                self._accent_tree = _acc_tree
            else:
                accent_rects = []
                accent_h = ch_d + 2
                _accent_layer = None
            self._dyn_accent_rects = accent_rects
            self._accent_layer     = _accent_layer

            # Snap-to-rhythm button for accent grid
            snap_acc_r = pygame.Rect(pw - 62, y, 58, 14)
            pygame.draw.rect(surf, (44, 36, 64), snap_acc_r, border_radius=2)
            pygame.draw.rect(surf, (90, 70, 130), snap_acc_r, 1, border_radius=2)
            surf.blit(font.render("\u21bb Snap", True, (180, 160, 220)),
                      (snap_acc_r.x + 4, snap_acc_r.y + 1))
            self._accent_snap_rect = snap_acc_r
            self._accent_snap_pat  = _dyn_act_pat
            y += accent_h + 18

        self._dyn_sliders = dyn_sliders

        y += 8

        # ── Improv header ────────────────────────────────────────────────────
        ip          = p.improv_for(self._improv_page_key)
        arrow_im    = "\u25bc" if not self._improv_collapsed else "\u25b6"
        im_en_str   = "[ON]" if (ip and ip.enabled) else "[off]"
        pygame.draw.rect(surf, (48, 38, 22), (0, y, pw, hdr_h))
        pygame.draw.line(surf, (165, 125, 45), (0, y), (pw, y))
        im_en_col = (115, 86, 28) if (ip and ip.enabled) else (58, 46, 22)
        im_en_r   = pygame.Rect(pw - 50, y + 2, 46, hdr_h - 4)
        pygame.draw.rect(surf, im_en_col, im_en_r, border_radius=3)
        surf.blit(font.render(im_en_str, True, (250, 210, 120)), (im_en_r.x + 4, y + 3))
        surf.blit(font.render(f"{arrow_im} Improv", True, (250, 210, 120)), (8, y + 3))
        self._improv_hdr_rect    = pygame.Rect(0, y, pw - 52, hdr_h)
        self._improv_enable_rect = im_en_r
        y += hdr_h

        improv_prob_sliders: list[dict] = []
        improv_grace_sliders: list[dict] = []
        improv_chirp_sliders: list[dict] = []
        improv_echo_sliders:  list[dict] = []
        improv_grace_arrows:  list       = []
        improv_chirp_arrows:  list       = []

        if not self._improv_collapsed and _HAS_IMPROV_ENG and ip is not None:
            _IM_ACT = (165, 125, 45)
            _IM_BG  = (38, 30, 14)
            y, self._improv_page_rects = self._draw_page_selector_row(
                surf, font, y, self._improv_page_key, p.improv_pages,
                active_col=(98, 70, 24),
                idle_col=(40, 30, 14),
                active_brd=(232, 185, 92),
                custom_brd=(168, 126, 46),
                idle_brd=(82, 60, 24),
                active_txt=(250, 225, 150),
                custom_txt=(225, 180, 95),
                idle_txt=(140, 112, 58),
                hdr_h=hdr_h,
            )
            _improv_pg = p.page_for(self._improv_page_key)

            # ── Three top-level probability sliders ───────────────────────────
            y = self._add_seq_slider(improv_prob_sliders, y, "Grace Prob",
                                     "prob_grace",  ip.prob_grace,  0.0, 1.0, ".2f")
            y = self._add_seq_slider(improv_prob_sliders, y, "Chirp Prob",
                                     "prob_chirp",  ip.prob_chirp,  0.0, 1.0, ".2f")
            y = self._add_seq_slider(improv_prob_sliders, y, "Echo Prob",
                                     "prob_echo",   ip.prob_echo,   0.0, 1.0, ".2f")
            for sl in improv_prob_sliders:
                r = sl["rect"]
                lbl_y = r.y - 14
                surf.blit(font.render(sl["label"], True, _PY_DIM), (8, lbl_y))
                vs = font.render(format(sl["val"], sl["fmt"]), True, _PY_TXT)
                surf.blit(vs, (pw - vs.get_width() - 8, lbl_y))
                pygame.draw.rect(surf, _IM_BG, r, border_radius=3)
                frac = max(0.0, min(1.0,
                           (sl["val"] - sl["lo"]) / max(sl["hi"] - sl["lo"], 1e-9)))
                thumb_x = r.x + int(frac * r.w)
                pygame.draw.rect(surf, _IM_ACT,
                                 pygame.Rect(r.x, r.y, thumb_x - r.x + 4, r.h),
                                 border_radius=3)
            y += 4

            # ── Grace sub-section ─────────────────────────────────────────────
            ar_gr = "\u25bc" if not self._improv_grace_collapsed else "\u25b6"
            pygame.draw.rect(surf, (42, 34, 18), (4, y, pw - 8, hdr_h - 2))
            pygame.draw.line(surf, (140, 100, 35), (4, y), (pw - 4, y))
            surf.blit(font.render(f" {ar_gr} Grace", True, (220, 180, 90)), (10, y + 2))
            self._improv_grace_hdr_rect = pygame.Rect(4, y, pw - 8, hdr_h - 2)
            y += hdr_h

            if not self._improv_grace_collapsed:
                # Mode picker
                pygame.draw.rect(surf, (34, 28, 12), (6, y, pw - 12, hdr_h - 2))
                surf.blit(font.render("Mode:", True, _PY_DIM), (12, y + 2))
                gm_l = pygame.Rect(pw - 46, y + 1, 19, hdr_h - 4)
                gm_r = pygame.Rect(pw - 25, y + 1, 19, hdr_h - 4)
                pygame.draw.rect(surf, (55, 44, 20), gm_l, border_radius=2)
                pygame.draw.rect(surf, (55, 44, 20), gm_r, border_radius=2)
                surf.blit(font.render("\u25c4", True, (200, 165, 80)), (gm_l.x + 3, gm_l.y + 1))
                surf.blit(font.render("\u25ba", True, (200, 165, 80)), (gm_r.x + 3, gm_r.y + 1))
                gm_lbl = ip.grace.mode if ip.grace.mode in GRACE_MODES else "chromatic"
                mw = font.size(gm_lbl)[0]
                surf.blit(font.render(gm_lbl, True, _PY_TXT), (pw // 2 - mw // 2, y + 2))
                improv_grace_arrows.append({"key": "grace_mode",  "left_r": gm_l, "right_r": gm_r})
                y += hdr_h + 4

                # Position picker
                pygame.draw.rect(surf, (34, 28, 12), (6, y, pw - 12, hdr_h - 2))
                surf.blit(font.render("Posn:", True, _PY_DIM), (12, y + 2))
                gp_l = pygame.Rect(pw - 46, y + 1, 19, hdr_h - 4)
                gp_r = pygame.Rect(pw - 25, y + 1, 19, hdr_h - 4)
                pygame.draw.rect(surf, (55, 44, 20), gp_l, border_radius=2)
                pygame.draw.rect(surf, (55, 44, 20), gp_r, border_radius=2)
                surf.blit(font.render("\u25c4", True, (200, 165, 80)), (gp_l.x + 3, gp_l.y + 1))
                surf.blit(font.render("\u25ba", True, (200, 165, 80)), (gp_r.x + 3, gp_r.y + 1))
                gp_lbl = ip.grace.position if ip.grace.position in GRACE_POSNS else "pre"
                pw2 = font.size(gp_lbl)[0]
                surf.blit(font.render(gp_lbl, True, _PY_TXT), (pw // 2 - pw2 // 2, y + 2))
                improv_grace_arrows.append({"key": "grace_posn", "left_r": gp_l, "right_r": gp_r})
                y += hdr_h + 4

                # Duration frac slider
                y = self._add_seq_slider(improv_grace_sliders, y, "Dur Frac",
                                         "grace_dur_frac", ip.grace.duration_frac, 0.01, 0.5, ".2f")
                # Velocity scale slider
                y = self._add_seq_slider(improv_grace_sliders, y, "Vel Scale",
                                         "grace_vel_scale", ip.grace.vel_scale, 0.0, 1.5, ".2f")
                for sl in improv_grace_sliders:
                    r = sl["rect"]
                    lbl_y = r.y - 14
                    surf.blit(font.render(sl["label"], True, _PY_DIM), (12, lbl_y))
                    vs = font.render(format(sl["val"], sl["fmt"]), True, _PY_TXT)
                    surf.blit(vs, (pw - vs.get_width() - 8, lbl_y))
                    pygame.draw.rect(surf, _IM_BG, r, border_radius=3)
                    frac = max(0.0, min(1.0,
                               (sl["val"] - sl["lo"]) / max(sl["hi"] - sl["lo"], 1e-9)))
                    thumb_x = r.x + int(frac * r.w)
                    pygame.draw.rect(surf, _IM_ACT,
                                     pygame.Rect(r.x, r.y, thumb_x - r.x + 4, r.h),
                                     border_radius=3)

                # Trim main toggle
                tm_col = (115, 86, 28) if ip.grace.trim_main else (42, 34, 18)
                tm_r   = pygame.Rect(6, y + 1, pw - 12, hdr_h - 2)
                pygame.draw.rect(surf, tm_col, tm_r, border_radius=2)
                tm_lbl = "Trim Main: ON" if ip.grace.trim_main else "Trim Main: off"
                surf.blit(font.render(tm_lbl, True, (240, 200, 100)), (12, y + 2))
                self._improv_grace_trim_rect = tm_r
                y += hdr_h
            y += 4

            # ── Chirp sub-section ─────────────────────────────────────────────
            ar_ch = "\u25bc" if not self._improv_chirp_collapsed else "\u25b6"
            pygame.draw.rect(surf, (42, 34, 18), (4, y, pw - 8, hdr_h - 2))
            pygame.draw.line(surf, (140, 100, 35), (4, y), (pw - 4, y))
            surf.blit(font.render(f" {ar_ch} Chirp", True, (220, 180, 90)), (10, y + 2))
            self._improv_chirp_hdr_rect = pygame.Rect(4, y, pw - 8, hdr_h - 2)
            y += hdr_h

            if not self._improv_chirp_collapsed:
                # Shape picker
                pygame.draw.rect(surf, (34, 28, 12), (6, y, pw - 12, hdr_h - 2))
                surf.blit(font.render("Shape:", True, _PY_DIM), (12, y + 2))
                cs_l = pygame.Rect(pw - 46, y + 1, 19, hdr_h - 4)
                cs_r = pygame.Rect(pw - 25, y + 1, 19, hdr_h - 4)
                pygame.draw.rect(surf, (55, 44, 20), cs_l, border_radius=2)
                pygame.draw.rect(surf, (55, 44, 20), cs_r, border_radius=2)
                surf.blit(font.render("\u25c4", True, (200, 165, 80)), (cs_l.x + 3, cs_l.y + 1))
                surf.blit(font.render("\u25ba", True, (200, 165, 80)), (cs_r.x + 3, cs_r.y + 1))
                cs_lbl = ip.chirp.shape if ip.chirp.shape in CHIRP_SHAPES else "up"
                csw = font.size(cs_lbl)[0]
                surf.blit(font.render(cs_lbl, True, _PY_TXT), (pw // 2 - csw // 2, y + 2))
                improv_chirp_arrows.append({"key": "chirp_shape", "left_r": cs_l, "right_r": cs_r})
                y += hdr_h + 4

                # Mode picker
                pygame.draw.rect(surf, (34, 28, 12), (6, y, pw - 12, hdr_h - 2))
                surf.blit(font.render("Mode:", True, _PY_DIM), (12, y + 2))
                cm_l = pygame.Rect(pw - 46, y + 1, 19, hdr_h - 4)
                cm_r = pygame.Rect(pw - 25, y + 1, 19, hdr_h - 4)
                pygame.draw.rect(surf, (55, 44, 20), cm_l, border_radius=2)
                pygame.draw.rect(surf, (55, 44, 20), cm_r, border_radius=2)
                surf.blit(font.render("\u25c4", True, (200, 165, 80)), (cm_l.x + 3, cm_l.y + 1))
                surf.blit(font.render("\u25ba", True, (200, 165, 80)), (cm_r.x + 3, cm_r.y + 1))
                cm_lbl = ip.chirp.mode if ip.chirp.mode in CHIRP_MODES else "chromatic"
                cmw = font.size(cm_lbl)[0]
                surf.blit(font.render(cm_lbl, True, _PY_TXT), (pw // 2 - cmw // 2, y + 2))
                improv_chirp_arrows.append({"key": "chirp_mode", "left_r": cm_l, "right_r": cm_r})
                y += hdr_h + 4

                # Steps stepper
                pygame.draw.rect(surf, (34, 28, 12), (6, y, pw - 12, hdr_h - 2))
                surf.blit(font.render("Steps:", True, _PY_DIM), (12, y + 2))
                cst_l = pygame.Rect(pw - 46, y + 1, 19, hdr_h - 4)
                cst_r = pygame.Rect(pw - 25, y + 1, 19, hdr_h - 4)
                pygame.draw.rect(surf, (55, 44, 20), cst_l, border_radius=2)
                pygame.draw.rect(surf, (55, 44, 20), cst_r, border_radius=2)
                surf.blit(font.render("\u25c4", True, (200, 165, 80)), (cst_l.x + 3, cst_l.y + 1))
                surf.blit(font.render("\u25ba", True, (200, 165, 80)), (cst_r.x + 3, cst_r.y + 1))
                cst_lbl = str(ip.chirp.steps)
                cstw = font.size(cst_lbl)[0]
                surf.blit(font.render(cst_lbl, True, _PY_TXT), (pw // 2 - cstw // 2, y + 2))
                improv_chirp_arrows.append({"key": "chirp_steps", "left_r": cst_l, "right_r": cst_r})
                y += hdr_h + 4

                # Duration frac + vel scale sliders
                y = self._add_seq_slider(improv_chirp_sliders, y, "Dur Frac",
                                         "chirp_dur_frac", ip.chirp.duration_frac, 0.01, 0.9, ".2f")
                y = self._add_seq_slider(improv_chirp_sliders, y, "Vel Scale",
                                         "chirp_vel_scale", ip.chirp.vel_scale, 0.0, 1.5, ".2f")
                for sl in improv_chirp_sliders:
                    r = sl["rect"]
                    lbl_y = r.y - 14
                    surf.blit(font.render(sl["label"], True, _PY_DIM), (12, lbl_y))
                    vs = font.render(format(sl["val"], sl["fmt"]), True, _PY_TXT)
                    surf.blit(vs, (pw - vs.get_width() - 8, lbl_y))
                    pygame.draw.rect(surf, _IM_BG, r, border_radius=3)
                    frac = max(0.0, min(1.0,
                               (sl["val"] - sl["lo"]) / max(sl["hi"] - sl["lo"], 1e-9)))
                    thumb_x = r.x + int(frac * r.w)
                    pygame.draw.rect(surf, _IM_ACT,
                                     pygame.Rect(r.x, r.y, thumb_x - r.x + 4, r.h),
                                     border_radius=3)
            y += 4

            # ── Echo sub-section ──────────────────────────────────────────────
            ar_ec = "\u25bc" if not self._improv_echo_collapsed else "\u25b6"
            pygame.draw.rect(surf, (42, 34, 18), (4, y, pw - 8, hdr_h - 2))
            pygame.draw.line(surf, (140, 100, 35), (4, y), (pw - 4, y))
            surf.blit(font.render(f" {ar_ec} Echo", True, (220, 180, 90)), (10, y + 2))
            self._improv_echo_hdr_rect = pygame.Rect(4, y, pw - 8, hdr_h - 2)
            y += hdr_h

            if not self._improv_echo_collapsed:
                # Lookback stepper
                pygame.draw.rect(surf, (34, 28, 12), (6, y, pw - 12, hdr_h - 2))
                surf.blit(font.render("Lookbk:", True, _PY_DIM), (12, y + 2))
                el_l = pygame.Rect(pw - 46, y + 1, 19, hdr_h - 4)
                el_r = pygame.Rect(pw - 25, y + 1, 19, hdr_h - 4)
                pygame.draw.rect(surf, (55, 44, 20), el_l, border_radius=2)
                pygame.draw.rect(surf, (55, 44, 20), el_r, border_radius=2)
                surf.blit(font.render("\u25c4", True, (200, 165, 80)), (el_l.x + 3, el_l.y + 1))
                surf.blit(font.render("\u25ba", True, (200, 165, 80)), (el_r.x + 3, el_r.y + 1))
                el_lbl = f"{ip.echo.lookback_bars}b"
                elw = font.size(el_lbl)[0]
                surf.blit(font.render(el_lbl, True, _PY_TXT), (pw // 2 - elw // 2, y + 2))
                # Reuse chirp_arrows list for echo lookback with sentinel key
                improv_chirp_arrows.append({"key": "echo_lookback", "left_r": el_l, "right_r": el_r})
                y += hdr_h + 4

                # Duration frac, vel falloff, max notes
                y = self._add_seq_slider(improv_echo_sliders, y, "Dur Frac",
                                         "echo_dur_frac",  ip.echo.duration_frac, 0.01, 1.0, ".2f")
                y = self._add_seq_slider(improv_echo_sliders, y, "Vel Fall",
                                         "echo_vel_fall",  ip.echo.vel_falloff,   0.1,  0.99, ".2f")
                for sl in improv_echo_sliders:
                    r = sl["rect"]
                    lbl_y = r.y - 14
                    surf.blit(font.render(sl["label"], True, _PY_DIM), (12, lbl_y))
                    vs = font.render(format(sl["val"], sl["fmt"]), True, _PY_TXT)
                    surf.blit(vs, (pw - vs.get_width() - 8, lbl_y))
                    pygame.draw.rect(surf, _IM_BG, r, border_radius=3)
                    frac = max(0.0, min(1.0,
                               (sl["val"] - sl["lo"]) / max(sl["hi"] - sl["lo"], 1e-9)))
                    thumb_x = r.x + int(frac * r.w)
                    pygame.draw.rect(surf, _IM_ACT,
                                     pygame.Rect(r.x, r.y, thumb_x - r.x + 4, r.h),
                                     border_radius=3)

                # Max notes stepper
                pygame.draw.rect(surf, (34, 28, 12), (6, y, pw - 12, hdr_h - 2))
                surf.blit(font.render("Max Notes:", True, _PY_DIM), (12, y + 2))
                en_l2 = pygame.Rect(pw - 46, y + 1, 19, hdr_h - 4)
                en_r2 = pygame.Rect(pw - 25, y + 1, 19, hdr_h - 4)
                pygame.draw.rect(surf, (55, 44, 20), en_l2, border_radius=2)
                pygame.draw.rect(surf, (55, 44, 20), en_r2, border_radius=2)
                surf.blit(font.render("\u25c4", True, (200, 165, 80)), (en_l2.x + 3, en_l2.y + 1))
                surf.blit(font.render("\u25ba", True, (200, 165, 80)), (en_r2.x + 3, en_r2.y + 1))
                mn_lbl = str(ip.echo.max_notes)
                mnw = font.size(mn_lbl)[0]
                surf.blit(font.render(mn_lbl, True, _PY_TXT), (pw // 2 - mnw // 2, y + 2))
                improv_chirp_arrows.append({"key": "echo_max_notes", "left_r": en_l2, "right_r": en_r2})
                y += hdr_h + 4
            y += 4

            # ── Improv step eligibility grid ──────────────────────────────────
            div_i   = _improv_pg.rhythm_division
            act_i   = min(_improv_pg.rhythm_active_pat, max(0, len(_improv_pg.rhythm_patterns) - 1))
            ch_i    = 16
            _improv_rpats = _improv_pg.rhythm_patterns
            _improv_act_pat = _improv_rpats[act_i] if _improv_rpats else None

            if _improv_act_pat is not None:
                _imp_tree = _improv_act_pat.get_improv_tree(div_i)

                def _imp_read(leaf, _si, _ex):
                    return leaf.on

                def _imp_bg(val, is_beat, depth, group):
                    d_shift = min(depth * 8, 30)
                    if val:
                        return (105 - d_shift, 78 - d_shift, 28)
                    return (32 + d_shift // 3, 26 + d_shift // 3, 12)

                def _imp_brd(val, is_beat, depth, group):
                    if val:
                        brd = (190, 150, 60)
                    else:
                        brd = (70, 55, 22)
                    if is_beat:
                        brd = tuple(min(255, c + 18) for c in brd)
                    return brd

                def _imp_label(val):
                    if val:
                        return ("G", (240, 200, 100))
                    return None

                def _imp_click(leaf, _si, _ex):
                    leaf.on = not leaf.on

                _improv_layer = GridViewLayer(
                    "improv",
                    read_fn        = _imp_read,
                    bg_fn          = _imp_bg,
                    brd_fn         = _imp_brd,
                    label_fn       = _imp_label,
                    click_fn       = _imp_click,
                    right_click_fn = lambda cell, lx, ly, tr, dv: self._build_step_context_menu(cell, lx, ly, tr, dv),
                    show_depth     = True,
                )

                improv_step_rects, improv_h = self._render_grid_layer(
                    surf, font, y, _improv_layer, _imp_tree, div_i,
                    p.page_meter(_improv_pg)[0], pw, cell_h=ch_i,
                    frac_beat_mode=getattr(_improv_pg, "frac_beat_mode", "warp"))
                self._improv_tree  = _imp_tree
                self._improv_layer = _improv_layer
            else:
                improv_step_rects = []
                improv_h = ch_i + 2
                _improv_layer = None
            self._improv_step_rects = improv_step_rects

            # Snap-to-rhythm button for improv grid
            snap_imp_r = pygame.Rect(pw - 62, y, 58, 14)
            pygame.draw.rect(surf, (44, 36, 20), snap_imp_r, border_radius=2)
            pygame.draw.rect(surf, (90, 70, 30), snap_imp_r, 1, border_radius=2)
            surf.blit(font.render("\u21bb Snap", True, (200, 170, 80)),
                      (snap_imp_r.x + 4, snap_imp_r.y + 1))
            self._improv_snap_rect = snap_imp_r
            self._improv_snap_pat  = _improv_act_pat
            y += improv_h + 18

        self._improv_prob_sliders  = improv_prob_sliders
        self._improv_grace_sliders = improv_grace_sliders
        self._improv_chirp_sliders = improv_chirp_sliders
        self._improv_echo_sliders  = improv_echo_sliders
        self._improv_grace_arrows  = improv_grace_arrows
        self._improv_chirp_arrows  = improv_chirp_arrows

        y += 8

        # ── Rhythm header ────────────────────────────────────────────────────
        arrow_rh = "\u25bc" if not self._rhythm_collapsed else "\u25b6"
        en_str   = "[ON]" if p.rhythm_enabled else "[off]"
        pygame.draw.rect(surf, (34, 26, 48), (0, y, pw, hdr_h))
        pygame.draw.line(surf, (110, 70, 160), (0, y), (pw, y))
        # Enable toggle button (right side of header)
        en_col = (88, 48, 130) if p.rhythm_enabled else (48, 38, 68)
        en_r   = pygame.Rect(pw - 50, y + 2, 46, hdr_h - 4)
        pygame.draw.rect(surf, en_col, en_r, border_radius=3)
        surf.blit(font.render(en_str, True, (200, 165, 245)), (en_r.x + 4, y + 3))
        surf.blit(font.render(f"{arrow_rh} Rhythm", True, (185, 145, 240)), (8, y + 3))
        self._rhythm_hdr_rect    = pygame.Rect(0, y, pw - 52, hdr_h)
        self._rhythm_enable_rect = en_r
        y += hdr_h

        if not self._rhythm_collapsed:
            # Resolve page data aliases up-front so every section below uses the
            # correct source (flat patch fields for "all", RhythmPage object otherwise).
            _active_pg  = p.page_for(p.rhythm_active_page)
            _rp_pats    = _active_pg.rhythm_patterns
            _rp_phrase  = _active_pg.rhythm_phrase
            _rp_act_pat = _active_pg.rhythm_active_pat
            _rp_div     = _active_pg.rhythm_division
            _rp_swing   = _active_pg.rhythm_swing
            _rp_pocket  = _active_pg.rhythm_pocket
            _rp_gate    = _active_pg.rhythm_gate
            _rp_pbars   = _active_pg.rhythm_prog_bars
            _rp_fitmode = _active_pg.rhythm_fit_mode

            # ── Division picker ──────────────────────────────────────────────
            surf.blit(font.render("Div:", True, _PY_DIM), (8, y + 3))
            div_rects: list = []
            dx = 40
            for dv in _RHYTHM_DIVISIONS:
                is_sel = (_rp_div == dv)
                dc = (85, 52, 136) if is_sel else (42, 36, 58)
                dr = pygame.Rect(dx, y + 1, 30, hdr_h - 2)
                pygame.draw.rect(surf, dc, dr, border_radius=3)
                tc = (215, 185, 255) if is_sel else (110, 92, 148)
                surf.blit(font.render(str(dv), True, tc), (dx + 4, y + 3))
                div_rects.append({"rect": dr, "val": dv})
                dx += 32
            meter_lbl = _page_meter_label(p, _active_pg)
            meter_s = font.render(meter_lbl, True, (150, 132, 196))
            surf.blit(meter_s, (pw - meter_s.get_width() - 8, y + 3))
            self._rhythm_div_rects = div_rects
            y += hdr_h + 2

            # ── Page selector (register parts) ───────────────────────────────
            surf.blit(font.render("Part:", True, _PY_DIM), (8, y + 3))
            page_rects: list = []
            pgx = 44
            for pg_key, pg_label in self._page_selector_entries(p.rhythm_active_page, p.rhythm_pages):
                is_active = (p.rhythm_active_page == pg_key)
                has_custom = (pg_key != "all" and pg_key in p.rhythm_pages)
                pg_col = (88, 52, 148) if is_active else (42, 36, 62)
                pg_brd = (175, 130, 255) if is_active else ((110, 80, 160) if has_custom else (58, 48, 80))
                pg_w   = max(32, min(72, font.size(pg_label)[0] + 8))
                pg_r   = pygame.Rect(pgx, y + 1, pg_w, hdr_h - 2)
                pygame.draw.rect(surf, pg_col, pg_r, border_radius=3)
                pygame.draw.rect(surf, pg_brd, pg_r, 1, border_radius=3)
                tc = (220, 195, 255) if is_active else ((165, 130, 210) if has_custom else (110, 92, 148))
                surf.blit(font.render(pg_label, True, tc), (pgx + 3, y + 3))
                page_rects.append({"rect": pg_r, "key": pg_key})
                pgx += pg_w + 2
            # [+] button to create a new page for the active register slot
            pg_add_r = pygame.Rect(pgx, y + 1, 16, hdr_h - 2)
            pygame.draw.rect(surf, (42, 60, 42), pg_add_r, border_radius=3)
            surf.blit(font.render("+", True, (110, 190, 110)), (pgx + 3, y + 3))
            pgx += 18
            # [-] button to delete the currently selected page (not shown for "All")
            _show_del = (p.rhythm_active_page != "all" and p.rhythm_active_page in p.rhythm_pages)
            pg_del_r = pygame.Rect(pgx, y + 1, 16, hdr_h - 2) if _show_del else None
            if _show_del:
                pygame.draw.rect(surf, (76, 38, 38), pg_del_r, border_radius=3)
                surf.blit(font.render("\u2212", True, (220, 110, 110)), (pgx + 4, y + 3))
            self._rhythm_page_rects  = page_rects
            self._rhythm_page_add_r  = pg_add_r
            self._rhythm_page_del_r  = pg_del_r
            y += hdr_h + 2

            # ── Stress pattern row + auto deploy ────────────────────────────
            stress_opts = _stress_pattern_options_for_meter(
                p.page_meter(_active_pg)[0],
                getattr(_active_pg, "frac_beat_mode", "warp"))
            cur_pattern = _page_stress_pattern(p, _active_pg)
            cur_idx = stress_opts.index(cur_pattern) if cur_pattern in stress_opts else 0
            stress_lbl = "+".join(str(v) for v in stress_opts[cur_idx])
            pygame.draw.rect(surf, (24, 20, 38), (2, y, pw - 4, hdr_h - 2))
            surf.blit(font.render("Stress:", True, _PY_DIM), (8, y + 2))
            sw = font.size(stress_lbl)[0]
            surf.blit(font.render(stress_lbl, True, _PY_TXT), (pw // 2 - sw // 2, y + 2))
            st_l  = pygame.Rect(pw - 136, y + 1, 19, hdr_h - 4)
            st_r  = pygame.Rect(pw - 115, y + 1, 19, hdr_h - 4)
            ag_r  = pygame.Rect(pw - 92,  y + 1, 42, hdr_h - 4)
            vel_r = pygame.Rect(pw - 48,  y + 1, 42, hdr_h - 4)
            pygame.draw.rect(surf, (55, 45, 78), st_l, border_radius=2)
            pygame.draw.rect(surf, (55, 45, 78), st_r, border_radius=2)
            pygame.draw.rect(surf, (58, 82, 52), ag_r, border_radius=2)
            pygame.draw.rect(surf, (52, 68, 100), vel_r, border_radius=2)
            surf.blit(font.render("<", True, (175, 150, 220)), (st_l.x + 4, y + 2))
            surf.blit(font.render(">", True, (175, 150, 220)), (st_r.x + 4, y + 2))
            surf.blit(font.render("Deploy", True, (170, 225, 170)), (ag_r.x + 4, y + 2))
            surf.blit(font.render("Vel\u2192", True, (150, 195, 240)), (vel_r.x + 4, y + 2))
            self._rhythm_stress_left_rect  = st_l
            self._rhythm_stress_right_rect = st_r
            self._rhythm_auto_grid_rect    = ag_r
            self._rhythm_stress_vel_rect   = vel_r
            y += hdr_h + 4

            # ── Pattern tabs ─────────────────────────────────────────────────
            surf.blit(font.render("Pats:", True, _PY_DIM), (8, y + 3))
            pat_tabs: list = []
            px = 46
            for pi, _ in enumerate(_rp_pats):
                is_sel = (pi == _rp_act_pat)
                pc = (85, 52, 136) if is_sel else (42, 36, 58)
                pr_r = pygame.Rect(px, y + 1, 24, hdr_h - 2)
                pygame.draw.rect(surf, pc, pr_r, border_radius=3)
                tc = (215, 185, 255) if is_sel else (110, 92, 148)
                surf.blit(font.render(str(pi + 1), True, tc), (px + 6, y + 3))
                pat_tabs.append({"rect": pr_r, "pat_i": pi})
                px += 26
            add_pr = pygame.Rect(px, y + 1, 18, hdr_h - 2)
            pygame.draw.rect(surf, (48, 76, 48), add_pr, border_radius=3)
            surf.blit(font.render("+", True, (125, 210, 125)), (px + 4, y + 3))
            px += 20
            del_pr = pygame.Rect(px, y + 1, 18, hdr_h - 2)
            pygame.draw.rect(surf, (76, 48, 48), del_pr, border_radius=3)
            surf.blit(font.render("-", True, (210, 125, 125)), (px + 5, y + 3))
            self._rhythm_pat_tabs     = pat_tabs
            self._rhythm_pat_add_rect = add_pr
            self._rhythm_pat_del_rect = del_pr
            y += hdr_h + 4

            # ── Step grid ────────────────────────────────────────────────────
            div    = _rp_div
            cell_h = 16
            step_rects: list = []
            if _rp_pats:
                act_pat = _rp_pats[min(_rp_act_pat, len(_rp_pats) - 1)]
                _meter_for_grid = p.page_meter(_active_pg)[0]
                _rhy_tree = act_pat.get_tree(div)

                # Rhythm on/off layer definition
                def _rhy_read(leaf, _si, _ex):
                    return (leaf.on, int(leaf.art))

                def _rhy_bg(val, is_beat, depth, group):
                    on, _art = val
                    d_shift = min(depth * 12, 40)
                    if group and on:
                        gc = GROUP_COLORS[min(group, len(GROUP_COLORS) - 1)]
                        return (max(0, gc[0] - d_shift), max(0, gc[1] - d_shift), max(0, gc[2] - d_shift))
                    if on:
                        return (125 - d_shift, 72 + d_shift, 195) if is_beat else (88 - d_shift, 52 + d_shift, 152)
                    return (28, 22 + d_shift // 3, 44) if is_beat else (22, 18 + d_shift // 3, 34)

                def _rhy_brd(val, is_beat, depth, group):
                    on, _art = val
                    d_shift = min(depth * 12, 40)
                    if group and on:
                        gc = GROUP_COLORS[min(group, len(GROUP_COLORS) - 1)]
                        return (min(255, gc[0] + 50), min(255, gc[1] + 50), min(255, gc[2] + 50))
                    if on:
                        return (175, 125, 255)
                    return (52, 42 + d_shift // 2, 78)

                def _rhy_label_right(val):
                    on, art = val
                    if on:
                        lbl = _ART_LABELS.get(art, "")
                        if lbl:
                            return (lbl, (240, 220, 120))
                    return None

                def _rhy_click(leaf, _si, _ex):
                    leaf.on = not leaf.on
                    self._rhythm_ctx_menu = None

                _rhythm_layer = GridViewLayer(
                    "rhythm",
                    read_fn        = _rhy_read,
                    bg_fn          = _rhy_bg,
                    brd_fn         = _rhy_brd,
                    label_right_fn = _rhy_label_right,
                    click_fn       = _rhy_click,
                    right_click_fn = lambda cell, lx, ly, tr, dv: self._build_step_context_menu(cell, lx, ly, tr, dv),
                    show_groups    = True,
                    show_depth     = True,
                )

                step_rects, metric_h = self._render_grid_layer(
                    surf, font, y, _rhythm_layer, _rhy_tree, div,
                    _meter_for_grid, pw, cell_h=cell_h,
                    frac_beat_mode=getattr(_active_pg, "frac_beat_mode", "warp"))
                self._rhythm_layer = _rhythm_layer
                self._rhythm_tree  = _rhy_tree
            else:
                metric_h = cell_h + 2
            self._rhythm_step_rects = step_rects
            y += metric_h + 4

            # ── Phrase builder ────────────────────────────────────────────────
            pygame.draw.rect(surf, (20, 16, 30), (2, y, pw - 4, hdr_h + 4),
                             border_radius=3)
            surf.blit(font.render("Phrase:", True, _PY_DIM), (8, y + 3))
            phrase_rects: list = []
            phx = 58
            for si, pat_i in enumerate(_rp_phrase):
                pr_slot = pygame.Rect(phx, y + 2, 20, hdr_h)
                pygame.draw.rect(surf, (72, 48, 110), pr_slot, border_radius=3)
                surf.blit(font.render(str(pat_i + 1), True, (195, 165, 240)),
                          (phx + 5, y + 3))
                phrase_rects.append({"rect": pr_slot, "slot_i": si})
                phx += 22
                if phx + 22 > pw - 40:   # wrap guard
                    break
            ph_add_r = pygame.Rect(phx, y + 2, 16, hdr_h)
            pygame.draw.rect(surf, (48, 76, 48), ph_add_r, border_radius=3)
            surf.blit(font.render("+", True, (125, 210, 125)), (phx + 4, y + 3))
            phx += 18
            ph_del_r = pygame.Rect(phx, y + 2, 16, hdr_h)
            pygame.draw.rect(surf, (76, 48, 48), ph_del_r, border_radius=3)
            surf.blit(font.render("-", True, (210, 125, 125)), (phx + 5, y + 3))
            self._rhythm_phrase_rects    = phrase_rects
            self._rhythm_phrase_add_rect = ph_add_r
            self._rhythm_phrase_del_rect = ph_del_r
            y += hdr_h + 10

            # ── Prog bars + Fit mode row ──────────────────────────────────────
            # Left cluster: [ < ] N bars [ > ]
            surf.blit(font.render("Bars:", True, _PY_DIM), (8, y + 3))
            pb_dec_r = pygame.Rect(46, y + 1, 16, hdr_h - 2)
            pb_inc_r = pygame.Rect(86, y + 1, 16, hdr_h - 2)
            pygame.draw.rect(surf, (55, 45, 78), pb_dec_r, border_radius=3)
            pygame.draw.rect(surf, (55, 45, 78), pb_inc_r, border_radius=3)
            surf.blit(font.render("<", True, (175, 150, 220)), (pb_dec_r.x + 4, y + 3))
            surf.blit(font.render(">", True, (175, 150, 220)), (pb_inc_r.x + 4, y + 3))
            pb_val = str(_rp_pbars)
            surf.blit(font.render(pb_val, True, _PY_TXT),
                      (pb_dec_r.right + (pb_inc_r.x - pb_dec_r.right - font.size(pb_val)[0]) // 2,
                       y + 3))
            self._rhythm_prog_dec_rect = pb_dec_r
            self._rhythm_prog_inc_rect = pb_inc_r
            # Right cluster: [ Drop ] [ Ext ]
            fm_drop_r = pygame.Rect(pw - 92, y + 1, 40, hdr_h - 2)
            fm_ext_r  = pygame.Rect(pw - 48, y + 1, 44, hdr_h - 2)
            _is_drop  = (_rp_fitmode == "drop")
            pygame.draw.rect(surf,
                             (78, 48, 118) if _is_drop else (42, 36, 58), fm_drop_r,
                             border_radius=3)
            pygame.draw.rect(surf,
                             (78, 48, 118) if not _is_drop else (42, 36, 58), fm_ext_r,
                             border_radius=3)
            surf.blit(font.render("Drop", True,
                                  (210, 185, 255) if _is_drop else (110, 92, 148)),
                      (fm_drop_r.x + 4, y + 3))
            surf.blit(font.render("Ext", True,
                                  (210, 185, 255) if not _is_drop else (110, 92, 148)),
                      (fm_ext_r.x + 6, y + 3))
            self._rhythm_fit_drop_rect = fm_drop_r
            self._rhythm_fit_ext_rect  = fm_ext_r
            y += hdr_h + 6

            # ── Swing / Pocket / Gate sliders ─────────────────────────────────
            rhythm_sliders: list[dict] = []
            _page_num, _page_den = p.page_meter(_active_pg)
            # Meter inherit indicator (only for named pages, not "all")
            _is_named_page = (p.rhythm_active_page != "all")
            _meter_inherited = _is_named_page and (
                float(getattr(_active_pg, "meter_numerator", 0.0)) == 0.0)
            _mih_r = pygame.Rect(pw - 84, y - (hdr_h), 80, hdr_h - 4)
            if _is_named_page:
                _mih_col = (38, 64, 38) if not _meter_inherited else (62, 92, 62)
                _mih_brd = (80, 148, 80) if _meter_inherited else (55, 82, 55)
                pygame.draw.rect(surf, _mih_col, _mih_r, border_radius=3)
                pygame.draw.rect(surf, _mih_brd, _mih_r, 1, border_radius=3)
                _mih_lbl = "\u2713 Global m." if _meter_inherited else "  Global m."
                surf.blit(font.render(_mih_lbl, True,
                                      (175, 240, 175) if _meter_inherited else (110, 160, 110)),
                          (_mih_r.x + 3, _mih_r.y + 2))
            self._rhythm_meter_inherit_r = _mih_r if _is_named_page else None
            y += 14  # label clearance: ensure first slider label clears the prog-bars row
            y = self._add_seq_slider(rhythm_sliders, y, "Page Num",
                                     "meter_numerator", _page_num, 1.0, 16.0, ".2f")
            y = self._add_seq_slider(rhythm_sliders, y, "Page Den",
                                     "meter_denominator", _page_den, 1.0, 16.0, ".2f")
            y = self._add_seq_slider(rhythm_sliders, y, "Swing",
                                     "rhythm_swing",  _rp_swing,  0.0,  0.67, ".2f")
            y = self._add_seq_slider(rhythm_sliders, y, "Pocket",
                                     "rhythm_pocket", _rp_pocket, -0.5, 0.5,  ".2f")
            y = self._add_seq_slider(rhythm_sliders, y, "Gate",
                                     "rhythm_gate",   _rp_gate,   0.05, 2.0,  ".2f")
            _RH_ACT = (98, 68, 158)
            for sl in rhythm_sliders:
                r = sl["rect"]
                lbl_y = r.y - 14
                _is_meter_sl = sl["key"] in {"meter_numerator", "meter_denominator"}
                _greyed = _is_meter_sl and _meter_inherited
                surf.blit(font.render(sl["label"], True,
                                      (75, 68, 85) if _greyed else _PY_DIM), (8, lbl_y))
                if _is_meter_sl:
                    val_str = f"{sl['val']:.2f}".rstrip("0").rstrip(".")
                    if _greyed:
                        val_str = f"\u21d0 {val_str}"
                else:
                    val_str = format(sl["val"], sl["fmt"])
                vs = font.render(val_str, True, (95, 85, 105) if _greyed else _PY_TXT)
                surf.blit(vs, (pw - vs.get_width() - 8, lbl_y))
                pygame.draw.rect(surf, (28, 24, 36) if _greyed else (38, 32, 52), r, border_radius=3)
                if not _greyed:
                    frac = max(0.0, min(1.0,
                               (sl["val"] - sl["lo"]) / max(sl["hi"] - sl["lo"], 1e-9)))
                    thumb_x = r.x + int(frac * r.w)
                    pygame.draw.rect(surf, _RH_ACT,
                                     pygame.Rect(r.x, r.y, thumb_x - r.x + 4, r.h),
                                     border_radius=3)
            hint = font.render("Shift=free Ctrl=irr", True, (110, 96, 152))
            surf.blit(hint, (pw - hint.get_width() - 8, rhythm_sliders[0]["rect"].y - 28))
            self._rhythm_sliders = rhythm_sliders
            y += 4

            # ── Fractional-beat mode toggle ──────────────────────────────────
            _fbm = getattr(_active_pg, "frac_beat_mode", "warp")
            _fbm_is_warp = (_fbm != "grid")
            fbm_warp_r = pygame.Rect(8, y, 52, hdr_h - 2)
            fbm_grid_r = pygame.Rect(62, y, 52, hdr_h - 2)
            _fbm_w_bg = (58, 42, 98) if _fbm_is_warp else (32, 28, 44)
            _fbm_g_bg = (42, 68, 92) if not _fbm_is_warp else (28, 38, 44)
            pygame.draw.rect(surf, _fbm_w_bg, fbm_warp_r, border_radius=3)
            pygame.draw.rect(surf, _fbm_g_bg, fbm_grid_r, border_radius=3)
            pygame.draw.rect(surf, (95, 68, 158) if _fbm_is_warp else (55, 48, 72),
                             fbm_warp_r, 1, border_radius=3)
            pygame.draw.rect(surf, (68, 120, 158) if not _fbm_is_warp else (48, 62, 72),
                             fbm_grid_r, 1, border_radius=3)
            surf.blit(font.render("\u223c Warp", True,
                                  (200, 170, 255) if _fbm_is_warp else (100, 88, 130)),
                      (fbm_warp_r.x + 3, y + 2))
            surf.blit(font.render("\u2587 Grid", True,
                                  (170, 215, 255) if not _fbm_is_warp else (88, 110, 130)),
                      (fbm_grid_r.x + 3, y + 2))
            _fbm_hint = font.render(
                "frac \u2192 warp pockets" if _fbm_is_warp else "frac \u2192 beat cell",
                True, (90, 82, 115))
            surf.blit(_fbm_hint, (pw - _fbm_hint.get_width() - 8, y + 2))
            self._frac_beat_warp_rect = fbm_warp_r
            self._frac_beat_grid_rect = fbm_grid_r
            y += hdr_h + 2

        # ── Placement section ────────────────────────────────────────────────
        _plc_hdr = pygame.Rect(2, y, pw - 4, hdr_h)
        pygame.draw.rect(surf, (30, 38, 55), _plc_hdr, border_radius=3)
        _plc_arrow = "\u25b6" if self._placement_collapsed else "\u25bc"
        surf.blit(font.render(f"{_plc_arrow} Placement", True, (140, 175, 220)),
                  (8, y + 3))
        _plc_total = sum(pt.player_count for pt in p.parts)
        if _plc_total > 0:
            _tot_s = font.render(f"{_plc_total} performers", True, (110, 155, 190))
            surf.blit(_tot_s, (pw - _tot_s.get_width() - 8, y + 3))
        self._placement_hdr_rect = _plc_hdr
        y += hdr_h + 2

        if not self._placement_collapsed:
            _pm_rects: list = []
            if not p.parts:
                surf.blit(font.render("(solve to populate)", True, (70, 80, 100)), (12, y + 2))
                y += hdr_h + 2
            else:
                for pt in p.parts:
                    _reg_col: tuple = {
                        "bass":  (60, 45, 88),
                        "mid":   (45, 68, 88),
                        "high":  (48, 88, 68),
                    }.get(pt.register, (52, 52, 72))
                    _row_r = pygame.Rect(4, y, pw - 8, hdr_h)
                    pygame.draw.rect(surf, _reg_col, _row_r, border_radius=2)
                    # Label (truncated)
                    lbl_max = pw - 78
                    _lbl = pt.label
                    while font.size(_lbl)[0] > lbl_max - 8 and len(_lbl) > 4:
                        _lbl = _lbl[:-1]
                    surf.blit(font.render(_lbl, True, (190, 210, 235)), (8, y + 2))
                    # Player count ± buttons
                    _dec_r = pygame.Rect(pw - 70, y + 1, 20, hdr_h - 2)
                    _cnt_r = pygame.Rect(pw - 48, y + 1, 22, hdr_h - 2)
                    _inc_r = pygame.Rect(pw - 24, y + 1, 20, hdr_h - 2)
                    pygame.draw.rect(surf, (55, 45, 78), _dec_r, border_radius=2)
                    pygame.draw.rect(surf, (32, 30, 48), _cnt_r, border_radius=2)
                    pygame.draw.rect(surf, (45, 68, 55), _inc_r, border_radius=2)
                    surf.blit(font.render("-", True, (175, 150, 220)),
                              (_dec_r.x + 6, y + 2))
                    surf.blit(font.render(str(pt.player_count), True, (200, 200, 230)),
                              (_cnt_r.x + (22 - font.size(str(pt.player_count))[0]) // 2, y + 2))
                    surf.blit(font.render("+", True, (140, 210, 160)),
                              (_inc_r.x + 5, y + 2))
                    _pm_rects.append({"dec": _dec_r, "inc": _inc_r, "key": pt.key})
                    y += hdr_h + 2
            self._placement_pm_rects = _pm_rects

        # Dropdown overlay — drawn last so it paints over every section below it.
        if self._open_dropdown in ("module", "control") and self._dropdown_items:
            btn_h = font.get_height() + 6
            for di in self._dropdown_items:
                r = di["rect"]
                pygame.draw.rect(surf, (50, 50, 66), r, border_radius=2)
                pygame.draw.rect(surf, (90, 90, 120), r, 1, border_radius=2)
                surf.blit(font.render(di["label"], True, (200, 220, 255)),
                          (r.x + 4, r.y + 2))

        # ── Context menu overlay (right-click on step cell) ──────────────
        ctx = self._rhythm_ctx_menu
        if ctx is not None:
            _ctx_items = ctx.get("items", [])
            if _ctx_items:
                # Background panel
                menu_r = ctx.get("menu_rect")
                if menu_r:
                    pygame.draw.rect(surf, (38, 32, 54), menu_r, border_radius=3)
                    pygame.draw.rect(surf, (100, 80, 150), menu_r, 1, border_radius=3)
                for ci in _ctx_items:
                    r = ci["rect"]
                    if ci.get("separator"):
                        pygame.draw.line(surf, (70, 60, 100),
                                         (r.x + 4, r.y + r.h // 2),
                                         (r.x + r.w - 4, r.y + r.h // 2))
                        continue
                    pygame.draw.rect(surf, (52, 44, 72), r, border_radius=2)
                    pygame.draw.rect(surf, (85, 70, 120), r, 1, border_radius=2)
                    lbl_c = ci.get("color", (210, 200, 240))
                    surf.blit(font.render(ci["label"], True, lbl_c),
                              (r.x + 4, r.y + 2))

        return self._apply_panel_scroll(surf)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self._patch is None:
            return False
        if self._handle_panel_wheel(event):
            return True

        rect = self.panel_rect

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if not rect.collidepoint(event.pos):
                return False
            lx = event.pos[0] - rect.x
            ly = event.pos[1] - rect.y + self._panel_scroll_y
            pw = self.PANEL_W
            p  = self._patch

            # ── Context menu item click (takes priority) ─────────────────────
            ctx = self._rhythm_ctx_menu
            if ctx is not None:
                _ctx_hit = False
                for ci in ctx.get("items", []):
                    if ci.get("separator"):
                        continue
                    if ci["rect"].collidepoint(lx, ly):
                        _ctx_hit = True
                        _ctx_cell = ctx["cell"]
                        _ctx_tree = ctx["tree"]
                        _ctx_div  = ctx["div"]
                        action    = ci["action"]

                        if action == "toggle":
                            _ctx_cell["leaf"].on = not _ctx_cell["leaf"].on

                        elif action == "art":
                            _ctx_cell["leaf"].art = ci["art_val"]

                        elif action == "group":
                            _ctx_cell["leaf"].group = ci["group_val"]

                        elif action == "subdivide":
                            leaf = _ctx_cell["leaf"]
                            if leaf.is_leaf():
                                leaf.subdivide(ci["n"])

                        elif action == "collapse":
                            leaf = _ctx_cell["leaf"]
                            _ctx_tree.collapse_at(leaf.position)

                        self._rhythm_ctx_menu = None
                        return True
                # Click outside menu items — close it
                if not _ctx_hit:
                    self._rhythm_ctx_menu = None
                    # Don't consume the click — let it fall through

            # ── Voices header toggle ─────────────────────────────────────────
            if self._voices_hdr_rect and self._voices_hdr_rect.collidepoint(lx, ly):
                self._voices_collapsed = not self._voices_collapsed
                return True

            # ── Sequence header toggle ───────────────────────────────────────
            if self._seq_hdr_rect and self._seq_hdr_rect.collidepoint(lx, ly):
                self._seq_collapsed = not self._seq_collapsed
                return True

            # ── Sequence arrow buttons ───────────────────────────────────────
            for ar in self._seq_arrows:
                if ar["left_r"].collidepoint(lx, ly):
                    self._apply_seq_arrow(ar["key"], -1)
                    return True
                if ar["right_r"].collidepoint(lx, ly):
                    self._apply_seq_arrow(ar["key"], +1)
                    return True

            # ── Sequence sliders ─────────────────────────────────────────────
            for i, sl in enumerate(self._seq_sliders):
                if sl["rect"].collidepoint(lx, ly):
                    self._dragging_seq_slider = i
                    self._apply_seq_slider(sl, lx)
                    return True

            # ── Sequence action buttons ──────────────────────────────────────
            for name, br in self._seq_btn_rects.items():
                if br.collidepoint(lx, ly):
                    if name == "deploy" and self.on_deploy_chord:
                        self.on_deploy_chord()
                    elif name == "demo" and self.on_demo_play:
                        self.on_demo_play()
                    elif name == "render_fund" and self.on_render_fund:
                        self.on_render_fund()
                    elif name == "render" and self.on_render:
                        self.on_render()
                    return True

            # ── Probabilities header toggle ──────────────────────────────────
            if self._prob_hdr_rect and self._prob_hdr_rect.collidepoint(lx, ly):
                self._prob_collapsed = not self._prob_collapsed
                return True

            # ── Probability sliders ──────────────────────────────────────────
            for i, sl in enumerate(self._prob_sliders):
                if sl["rect"].collidepoint(lx, ly):
                    self._dragging_prob_slider = i
                    self._apply_prob_slider(sl, lx)
                    return True

            # ── Dynamics header toggle / enable ──────────────────────────────
            if self._dyn_hdr_rect and self._dyn_hdr_rect.collidepoint(lx, ly):
                self._dyn_collapsed = not self._dyn_collapsed
                return True
            if self._dyn_enable_rect and self._dyn_enable_rect.collidepoint(lx, ly):
                dp = p.ensure_dynamics_page(self._dyn_page_key)
                if dp is not None:
                    dp.enabled = not dp.enabled
                return True

            if not self._dyn_collapsed and _HAS_DYN_ENG:
                dp = p.ensure_dynamics_page(self._dyn_page_key)
                for pg in self._dyn_page_rects:
                    if pg["rect"].collidepoint(lx, ly):
                        self._dyn_page_key = pg["key"]
                        return True
                # Curve shape arrows
                if self._dyn_curve_left_rect and self._dyn_curve_left_rect.collidepoint(lx, ly):
                    if dp is not None:
                        idx = CURVE_SHAPES.index(dp.curve.shape) if dp.curve.shape in CURVE_SHAPES else 0
                        dp.curve.shape = CURVE_SHAPES[(idx - 1) % len(CURVE_SHAPES)]
                    return True
                if self._dyn_curve_right_rect and self._dyn_curve_right_rect.collidepoint(lx, ly):
                    if dp is not None:
                        idx = CURVE_SHAPES.index(dp.curve.shape) if dp.curve.shape in CURVE_SHAPES else 0
                        dp.curve.shape = CURVE_SHAPES[(idx + 1) % len(CURVE_SHAPES)]
                    return True
                # Scope stepper
                if self._dyn_scope_dec_rect and self._dyn_scope_dec_rect.collidepoint(lx, ly):
                    if dp is not None:
                        sv_idx = min(range(len(_SCOPE_VALUES)),
                                     key=lambda j: abs(_SCOPE_VALUES[j] - dp.curve.scope_bars))
                        dp.curve.scope_bars = _SCOPE_VALUES[max(0, sv_idx - 1)]
                    return True
                if self._dyn_scope_inc_rect and self._dyn_scope_inc_rect.collidepoint(lx, ly):
                    if dp is not None:
                        sv_idx = min(range(len(_SCOPE_VALUES)),
                                     key=lambda j: abs(_SCOPE_VALUES[j] - dp.curve.scope_bars))
                        dp.curve.scope_bars = _SCOPE_VALUES[min(len(_SCOPE_VALUES) - 1, sv_idx + 1)]
                    return True
                if self._dyn_auto_accent_rect and self._dyn_auto_accent_rect.collidepoint(lx, ly):
                    _dyn_page = p if self._dyn_page_key == "all" else p.page_for(self._dyn_page_key)
                    if _dyn_page is not None:
                        _auto_apply_dynamics_accent(p, _dyn_page, dp)
                    return True
                # Intensity slider
                for i, sl in enumerate(self._dyn_sliders):
                    if sl["rect"].collidepoint(lx, ly):
                        self._dragging_dyn_slider = i
                        self._apply_dyn_slider(sl, lx)
                        return True
                # Accent grid click
                if self._accent_layer is not None:
                    if self._handle_grid_layer_click(self._dyn_accent_rects, lx, ly, self._accent_layer):
                        return True
                # Accent snap-to-rhythm
                if getattr(self, '_accent_snap_rect', None) and self._accent_snap_rect.collidepoint(lx, ly):
                    _snap_pat = getattr(self, '_accent_snap_pat', None)
                    if _snap_pat is not None:
                        _snap_pat.snap_accent_to_rhythm()
                    return True

            # ── Improv header toggle / enable ─────────────────────────────────
            if self._improv_hdr_rect and self._improv_hdr_rect.collidepoint(lx, ly):
                self._improv_collapsed = not self._improv_collapsed
                return True
            if self._improv_enable_rect and self._improv_enable_rect.collidepoint(lx, ly):
                ip2 = p.ensure_improv_page(self._improv_page_key)
                if ip2 is not None:
                    ip2.enabled = not ip2.enabled
                return True

            if not self._improv_collapsed and _HAS_IMPROV_ENG:
                ip2 = p.ensure_improv_page(self._improv_page_key)
                for pg in self._improv_page_rects:
                    if pg["rect"].collidepoint(lx, ly):
                        self._improv_page_key = pg["key"]
                        return True
                # ── Probability sliders ───────────────────────────────────────
                for i, sl in enumerate(self._improv_prob_sliders):
                    if sl["rect"].collidepoint(lx, ly):
                        self._dragging_improv_prob_sl = i
                        self._apply_improv_slider(sl, lx)
                        return True
                # ── Grace sub-section ─────────────────────────────────────────
                if self._improv_grace_hdr_rect and self._improv_grace_hdr_rect.collidepoint(lx, ly):
                    self._improv_grace_collapsed = not self._improv_grace_collapsed
                    return True
                if not self._improv_grace_collapsed and ip2 is not None:
                    # Grace sliders
                    for i, sl in enumerate(self._improv_grace_sliders):
                        if sl["rect"].collidepoint(lx, ly):
                            self._dragging_grace_sl = i
                            self._apply_improv_slider(sl, lx)
                            return True
                    # Trim main toggle
                    if self._improv_grace_trim_rect and self._improv_grace_trim_rect.collidepoint(lx, ly):
                        ip2.grace.trim_main = not ip2.grace.trim_main
                        return True
                    # Grace arrows
                    for ar in self._improv_grace_arrows:
                        for side, delta in (("left_r", -1), ("right_r", +1)):
                            if ar[side].collidepoint(lx, ly):
                                key = ar["key"]
                                if key == "grace_mode":
                                    lst = GRACE_MODES
                                    cur = ip2.grace.mode if ip2.grace.mode in lst else lst[0]
                                    ip2.grace.mode = lst[(lst.index(cur) + delta) % len(lst)]
                                elif key == "grace_posn":
                                    lst = GRACE_POSNS
                                    cur = ip2.grace.position if ip2.grace.position in lst else lst[0]
                                    ip2.grace.position = lst[(lst.index(cur) + delta) % len(lst)]
                                return True
                # ── Chirp sub-section ─────────────────────────────────────────
                if self._improv_chirp_hdr_rect and self._improv_chirp_hdr_rect.collidepoint(lx, ly):
                    self._improv_chirp_collapsed = not self._improv_chirp_collapsed
                    return True
                if not self._improv_chirp_collapsed and ip2 is not None:
                    for i, sl in enumerate(self._improv_chirp_sliders):
                        if sl["rect"].collidepoint(lx, ly):
                            self._dragging_chirp_sl = i
                            self._apply_improv_slider(sl, lx)
                            return True
                    for ar in self._improv_chirp_arrows:
                        for side, delta in (("left_r", -1), ("right_r", +1)):
                            if ar[side].collidepoint(lx, ly):
                                key = ar["key"]
                                if key == "chirp_shape":
                                    lst = CHIRP_SHAPES
                                    cur = ip2.chirp.shape if ip2.chirp.shape in lst else lst[0]
                                    ip2.chirp.shape = lst[(lst.index(cur) + delta) % len(lst)]
                                elif key == "chirp_mode":
                                    lst = CHIRP_MODES
                                    cur = ip2.chirp.mode if ip2.chirp.mode in lst else lst[0]
                                    ip2.chirp.mode = lst[(lst.index(cur) + delta) % len(lst)]
                                elif key == "chirp_steps":
                                    ip2.chirp.steps = max(2, min(16, ip2.chirp.steps + delta))
                                elif key == "echo_lookback":
                                    ip2.echo.lookback_bars = max(1, min(8, ip2.echo.lookback_bars + delta))
                                elif key == "echo_max_notes":
                                    ip2.echo.max_notes = max(1, min(16, ip2.echo.max_notes + delta))
                                return True
                # ── Echo sub-section ──────────────────────────────────────────
                if self._improv_echo_hdr_rect and self._improv_echo_hdr_rect.collidepoint(lx, ly):
                    self._improv_echo_collapsed = not self._improv_echo_collapsed
                    return True
                if not self._improv_echo_collapsed and ip2 is not None:
                    for i, sl in enumerate(self._improv_echo_sliders):
                        if sl["rect"].collidepoint(lx, ly):
                            self._dragging_echo_sl = i
                            self._apply_improv_slider(sl, lx)
                            return True
                # ── Step grid ─────────────────────────────────────────────────
                if getattr(self, '_improv_layer', None) is not None:
                    if self._handle_grid_layer_click(self._improv_step_rects, lx, ly, self._improv_layer):
                        return True
                # Improv snap-to-rhythm
                if getattr(self, '_improv_snap_rect', None) and self._improv_snap_rect.collidepoint(lx, ly):
                    _snap_pat = getattr(self, '_improv_snap_pat', None)
                    if _snap_pat is not None:
                        _snap_pat.snap_improv_to_rhythm()
                    return True

            # ── Voice rows ───────────────────────────────────────────────────
            if not self._voices_collapsed:
                y0    = self._voices_row_start_y
                row_h = self._row_h
                for key, _, _, _ in self._items:
                    i_idx = self._items.index((key, _, _, _)) if False else None
                for i_idx, (key, _, _, _) in enumerate(self._items):
                    row_top = y0 + i_idx * row_h
                    if row_top <= ly < row_top + row_h:
                        # Pinned Patch row — select only, no delete/mute/solo
                        if key in {"__patch__", "__system__"}:
                            self._active = key
                            if self.on_select:
                                self.on_select(key)
                            return True
                        is_mix     = any(m.key == key for m in self._patch.mixers)  if self._patch else False
                        is_router  = _router_instance_for_active_key(self._patch, key) is not None if self._patch else False
                        is_param   = any(pn.key == key for pn in self._patch.param_nodes) if self._patch else False
                        is_lfo     = any(l.key == key for l in self._patch.lfos) if self._patch else False
                        is_module  = any(m.key == key for m in self._patch.modules) if self._patch else False
                        is_control = any(cs.key == key for cs in self._patch.controls) if self._patch else False
                        if not is_mix and not is_router and lx >= pw - 24:
                            # Delete
                            if self.on_remove_voice:
                                self.on_remove_voice(key)
                        elif not is_mix and not is_router and not is_param and not is_control and not is_lfo and lx >= pw - 60:
                            if lx >= pw - 42:
                                # Mute
                                if is_module:
                                    for mod in self._patch.modules:
                                        if mod.key == key:
                                            mod.muted = not mod.muted
                                else:
                                    for v in self._patch.voices:
                                        if v.key == key:
                                            v.muted = not v.muted
                                if self.on_toggle_mute:
                                    self.on_toggle_mute(key)
                            else:
                                # Solo (voices + modules)
                                if getattr(self._patch, "solo_key", None) == key:
                                    self._patch.solo_key = None
                                else:
                                    self._patch.solo_key = key
                                if self.on_toggle_mute:
                                    self.on_toggle_mute(key)
                        else:
                            self._active = key
                            if self.on_select:
                                self.on_select(key)
                        return True

                # Open dropdown clicks — handle before button rows
                if self._open_dropdown:
                    for di in self._dropdown_items:
                        if di["rect"].collidepoint(lx, ly):
                            if self._open_dropdown == "module":
                                _data = str(di["data"])
                                if _data.startswith("router:"):
                                    if self.on_add_router:
                                        self.on_add_router(_data.split(":", 1)[1])
                                elif self.on_add_module:
                                    self.on_add_module(_data)
                            else:
                                _data = str(di["data"])
                                if _data.startswith("router:"):
                                    if self.on_add_router:
                                        self.on_add_router(_data.split(":", 1)[1])
                                elif self.on_add_control:
                                    self.on_add_control()
                            self._open_dropdown = ""
                            return True
                    # Click outside dropdown — close it
                    self._open_dropdown = ""
                    return True

                # Add buttons row 1 — Voice / Param
                btn_y = self._btn_y
                btn_h = self._btn_h
                if btn_y <= ly < btn_y + btn_h:
                    half = pw // 2
                    if lx < half:
                        if self.on_add_voice:
                            self.on_add_voice()
                    else:
                        if self.on_add_param:
                            self.on_add_param()
                    return True
                # Add buttons row 2 — Module▾ / Control▾
                btn2_y = self._btn2_y
                btn2_h = self._btn2_h
                if btn2_y <= ly < btn2_y + btn2_h:
                    half = pw // 2
                    if lx < half:
                        self._open_dropdown = "" if self._open_dropdown == "module" else "module"
                    else:
                        self._open_dropdown = "" if self._open_dropdown == "control" else "control"
                    return True

            # ── Rhythm section ───────────────────────────────────────────────────
            p = self._patch
            if p is not None:
                # Header click (collapse toggle)
                if self._rhythm_hdr_rect and self._rhythm_hdr_rect.collidepoint(lx, ly):
                    self._rhythm_collapsed = not self._rhythm_collapsed
                    return True
                # Enable button
                if self._rhythm_enable_rect and self._rhythm_enable_rect.collidepoint(lx, ly):
                    p.rhythm_enabled = not p.rhythm_enabled
                    return True
                if not self._rhythm_collapsed:
                    # Page selector tabs
                    for pg in getattr(self, "_rhythm_page_rects", []):
                        if pg["rect"].collidepoint(lx, ly):
                            p.rhythm_active_page = pg["key"]
                            return True
                    # Page [+] — create a new per-register page for the active slot
                    if (getattr(self, "_rhythm_page_add_r", None) and
                            self._rhythm_page_add_r.collidepoint(lx, ly)):
                        key = p.rhythm_active_page
                        if key != "all" and key not in p.rhythm_pages:
                            p.rhythm_pages[key] = p._default_page()
                        return True
                    # Page [−] — delete the active per-register page
                    if (getattr(self, "_rhythm_page_del_r", None) and
                            self._rhythm_page_del_r is not None and
                            self._rhythm_page_del_r.collidepoint(lx, ly)):
                        _del_key = p.rhythm_active_page
                        if _del_key != "all":
                            p.remove_page(_del_key)
                            p.rhythm_active_page = "all"
                        return True
                    # Meter inherit toggle (Global meter checkbox)
                    if (getattr(self, "_rhythm_meter_inherit_r", None) and
                            self._rhythm_meter_inherit_r is not None and
                            self._rhythm_meter_inherit_r.collidepoint(lx, ly)):
                        _rw_key2 = p.rhythm_active_page
                        if _rw_key2 != "all" and _rw_key2 in p.rhythm_pages:
                            _pg2 = p.rhythm_pages[_rw_key2]
                            if float(_pg2.meter_numerator) == 0.0:
                                # Enable page-local meter: copy global as starting value
                                _pg2.meter_numerator   = float(p.meter_numerator)
                                _pg2.meter_denominator = float(p.meter_denominator)
                            else:
                                # Revert to inherit (0.0 = inherit from patch)
                                _pg2.meter_numerator   = 0.0
                                _pg2.meter_denominator = 0.0
                        return True
                    # Resolve write target — "all" page writes to flat patch fields,
                    # named pages write to the RhythmPage object in rhythm_pages.
                    _rw_key = p.rhythm_active_page
                    _rw = (p.rhythm_pages[_rw_key]
                           if (_rw_key != "all" and _rw_key in p.rhythm_pages)
                           else p)
                    _rw_page = p.page_for(_rw_key)
                    _stress_opts = _stress_pattern_options_for_meter(
                        p.page_meter(_rw_page)[0],
                        getattr(_rw_page, "frac_beat_mode", "warp"))
                    _cur_stress = _page_stress_pattern(p, _rw_page)
                    _stress_idx = _stress_opts.index(_cur_stress) if _cur_stress in _stress_opts else 0
                    if self._rhythm_stress_left_rect and self._rhythm_stress_left_rect.collidepoint(lx, ly):
                        _next = _stress_opts[(_stress_idx - 1) % len(_stress_opts)]
                        _rw.stress_pattern = list(_next)
                        return True
                    if self._rhythm_stress_right_rect and self._rhythm_stress_right_rect.collidepoint(lx, ly):
                        _next = _stress_opts[(_stress_idx + 1) % len(_stress_opts)]
                        _rw.stress_pattern = list(_next)
                        return True
                    if self._rhythm_auto_grid_rect and self._rhythm_auto_grid_rect.collidepoint(lx, ly):
                        _auto_apply_rhythm_grid(p, _rw)
                        return True
                    if (getattr(self, "_rhythm_stress_vel_rect", None) and
                            self._rhythm_stress_vel_rect.collidepoint(lx, ly)):
                        _vel_dyn_key = p.rhythm_active_page
                        _vel_dp = p.ensure_dynamics_page(_vel_dyn_key)
                        _vel_page = p if _vel_dyn_key == "all" else p.page_for(_vel_dyn_key)
                        _auto_apply_stress_velocity(p, _vel_page, _vel_dp)
                        return True
                    # Fractional-beat mode toggle
                    if getattr(self, "_frac_beat_warp_rect", None) and self._frac_beat_warp_rect.collidepoint(lx, ly):
                        _rw_page.frac_beat_mode = "warp"
                        if _rw_key == "all":
                            p.frac_beat_mode = "warp"
                        return True
                    if getattr(self, "_frac_beat_grid_rect", None) and self._frac_beat_grid_rect.collidepoint(lx, ly):
                        _rw_page.frac_beat_mode = "grid"
                        if _rw_key == "all":
                            p.frac_beat_mode = "grid"
                        return True
                    # Division picker
                    for dr in self._rhythm_div_rects:
                        if dr["rect"].collidepoint(lx, ly):
                            _rw.rhythm_division = dr["val"]
                            return True
                    # Pattern tabs
                    for pt in self._rhythm_pat_tabs:
                        if pt["rect"].collidepoint(lx, ly):
                            _rw.rhythm_active_pat = pt["pat_i"]
                            return True
                    # Add / remove pattern
                    if self._rhythm_pat_add_rect and self._rhythm_pat_add_rect.collidepoint(lx, ly):
                        if len(_rw.rhythm_patterns) < 8:
                            n = len(_rw.rhythm_patterns)
                            _rw.rhythm_patterns.append(RhythmPattern(name=f"Pat {n + 1}"))
                            _rw.rhythm_active_pat = n
                        return True
                    if self._rhythm_pat_del_rect and self._rhythm_pat_del_rect.collidepoint(lx, ly):
                        if len(_rw.rhythm_patterns) > 1:
                            _rw.rhythm_patterns.pop()
                            _rw.rhythm_active_pat = min(_rw.rhythm_active_pat,
                                                        len(_rw.rhythm_patterns) - 1)
                        return True
                    # Step grid toggle (left-click)
                    for sr in self._rhythm_step_rects:
                        if sr["rect"].collidepoint(lx, ly):
                            act_i = min(_rw.rhythm_active_pat, len(_rw.rhythm_patterns) - 1)
                            pat   = _rw.rhythm_patterns[act_i]
                            if sr.get("tree_mode"):
                                leaf = sr["leaf"]
                                leaf.on = not leaf.on
                            else:
                                si = sr["step_i"]
                                pat.ensure_size(_rw.rhythm_division)
                                pat.steps[si] = not pat.steps[si]
                            self._rhythm_ctx_menu = None
                            return True
                    # Phrase: click a slot to cycle to next pattern index
                    for ph in self._rhythm_phrase_rects:
                        if ph["rect"].collidepoint(lx, ly):
                            si = ph["slot_i"]
                            if 0 <= si < len(_rw.rhythm_phrase):
                                _rw.rhythm_phrase[si] = (
                                    (_rw.rhythm_phrase[si] + 1) % max(1, len(_rw.rhythm_patterns)))
                            return True
                    # Phrase add / remove bar
                    if self._rhythm_phrase_add_rect and self._rhythm_phrase_add_rect.collidepoint(lx, ly):
                        _rw.rhythm_phrase.append(_rw.rhythm_active_pat)
                        return True
                    if self._rhythm_phrase_del_rect and self._rhythm_phrase_del_rect.collidepoint(lx, ly):
                        if len(_rw.rhythm_phrase) > 1:
                            _rw.rhythm_phrase.pop()
                        return True
                    # Prog bars stepper
                    if self._rhythm_prog_dec_rect and self._rhythm_prog_dec_rect.collidepoint(lx, ly):
                        _rw.rhythm_prog_bars = max(1, _rw.rhythm_prog_bars - 1)
                        return True
                    if self._rhythm_prog_inc_rect and self._rhythm_prog_inc_rect.collidepoint(lx, ly):
                        _rw.rhythm_prog_bars = min(32, _rw.rhythm_prog_bars + 1)
                        return True
                    # Fit mode toggle
                    if self._rhythm_fit_drop_rect and self._rhythm_fit_drop_rect.collidepoint(lx, ly):
                        _rw.rhythm_fit_mode = "drop"
                        return True
                    if self._rhythm_fit_ext_rect and self._rhythm_fit_ext_rect.collidepoint(lx, ly):
                        _rw.rhythm_fit_mode = "extend"
                        return True
                    # Rhythm sliders
                    for i, sl in enumerate(self._rhythm_sliders):
                        if sl["rect"].collidepoint(lx, ly):
                            self._dragging_rhythm_slider = i
                            self._apply_rhythm_slider(sl, lx)
                            return True

            # ── Placement section header ─────────────────────────────────────
            if (getattr(self, "_placement_hdr_rect", None) and
                    self._placement_hdr_rect.collidepoint(lx, ly)):
                self._placement_collapsed = not self._placement_collapsed
                return True

            # ── Placement player count +/- buttons ───────────────────────────
            _pm = PlacementModule(p)
            for row in getattr(self, "_placement_pm_rects", []):
                if row["dec"].collidepoint(lx, ly):
                    _pm.increment_player_count(row["key"], -1)
                    return True
                if row["inc"].collidepoint(lx, ly):
                    _pm.increment_player_count(row["key"], +1)
                    return True

            return False

        elif event.type == MOUSEBUTTONDOWN and event.button == 3:
            # Right-click on step cell: open context menu
            p = self._patch
            rect = self.panel_rect
            if p is not None and rect.collidepoint(event.pos):
                lx = event.pos[0] - rect.x
                ly = event.pos[1] - rect.y + self._panel_scroll_y
                for sr in self._rhythm_step_rects:
                    if sr["rect"].collidepoint(lx, ly):
                        _rw_key = p.rhythm_active_page
                        _rw = (p.rhythm_pages[_rw_key]
                               if (_rw_key != "all" and _rw_key in p.rhythm_pages)
                               else p)
                        act_i = min(_rw.rhythm_active_pat, len(_rw.rhythm_patterns) - 1)
                        pat   = _rw.rhythm_patterns[act_i]
                        self._rhythm_ctx_menu = self._build_step_context_menu(
                            sr, lx, ly, pat, _rw.rhythm_division)
                        return True
                # Right-click elsewhere: close context menu
                self._rhythm_ctx_menu = None

        elif event.type == MOUSEBUTTONUP and event.button == 1:
            self._dragging_seq_slider    = -1
            self._dragging_rhythm_slider = -1
            self._dragging_prob_slider   = -1
            self._dragging_dyn_slider    = -1
            self._dragging_improv_prob_sl = -1
            self._dragging_grace_sl      = -1
            self._dragging_chirp_sl      = -1
            self._dragging_echo_sl       = -1

        elif event.type == MOUSEMOTION:
            if self._dragging_seq_slider >= 0 and rect.collidepoint(event.pos):
                lx = event.pos[0] - rect.x
                sl = self._seq_sliders[self._dragging_seq_slider]
                self._apply_seq_slider(sl, lx)
                return True
            if self._dragging_rhythm_slider >= 0 and rect.collidepoint(event.pos):
                lx = event.pos[0] - rect.x
                sl = self._rhythm_sliders[self._dragging_rhythm_slider]
                self._apply_rhythm_slider(sl, lx)
                return True
            if self._dragging_prob_slider >= 0 and rect.collidepoint(event.pos):
                lx = event.pos[0] - rect.x
                sl = self._prob_sliders[self._dragging_prob_slider]
                self._apply_prob_slider(sl, lx)
                return True
            if self._dragging_dyn_slider >= 0 and rect.collidepoint(event.pos):
                lx = event.pos[0] - rect.x
                sl = self._dyn_sliders[self._dragging_dyn_slider]
                self._apply_dyn_slider(sl, lx)
                return True
            if self._dragging_improv_prob_sl >= 0 and rect.collidepoint(event.pos):
                lx = event.pos[0] - rect.x
                sl = self._improv_prob_sliders[self._dragging_improv_prob_sl]
                self._apply_improv_slider(sl, lx)
                return True
            if self._dragging_grace_sl >= 0 and rect.collidepoint(event.pos):
                lx = event.pos[0] - rect.x
                sl = self._improv_grace_sliders[self._dragging_grace_sl]
                self._apply_improv_slider(sl, lx)
                return True
            if self._dragging_chirp_sl >= 0 and rect.collidepoint(event.pos):
                lx = event.pos[0] - rect.x
                sl = self._improv_chirp_sliders[self._dragging_chirp_sl]
                self._apply_improv_slider(sl, lx)
                return True
            if self._dragging_echo_sl >= 0 and rect.collidepoint(event.pos):
                lx = event.pos[0] - rect.x
                sl = self._improv_echo_sliders[self._dragging_echo_sl]
                self._apply_improv_slider(sl, lx)
                return True

        return False


# ---------------------------------------------------------------------------
# VoicePanel — right panel: parameters for the selected voice
# ---------------------------------------------------------------------------

class PartialPanel(Panel):
    """Right panel: editable parameters for the active voice/LFO."""

    _SLIDER_H = 16
    _SLIDER_PAD = 4

    @property
    def panel_rect(self) -> pygame.Rect:
        sw = pygame.display.get_surface().get_width()
        sh = pygame.display.get_surface().get_height()
        top = self._top_offset
        h = sh - top - BOTTOM_H
        return pygame.Rect(sw - self.PANEL_W, top, self.PANEL_W, h)

    def __init__(self, side: str = "right") -> None:
        super().__init__(side=side)
        self.title = "Parameters"
        self._font: pygame.font.Font | None = None
        self._patch: AnalyticPatch | None = None
        self._active_key: str = ""
        self._sliders: list[dict] = []    # runtime slider descriptors
        self._dragging_slider: int = -1
        self._dirty: bool = True          # need to rebuild slider layout
        # Inline text-editing state for str-dtype knobs
        self._text_edit_idx: int = -1     # slider index being edited (-1 = none)
        self._text_edit_buf: str = ""     # current edit buffer
        self._role_defaults_rect: pygame.Rect | None = None
        self.on_open_piecewise_editor = None

    def set_patch(self, patch: AnalyticPatch, active_key: str) -> None:
        if (patch is not self._patch) or (active_key != self._active_key):
            self._dirty = True
        self._patch       = patch
        self._active_key  = active_key

    def _font_(self) -> pygame.font.Font:
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("consolas", 12)
        return self._font

    def _add_slider(self, sliders: list, y: int, label: str, key: str,
                    val: float, lo: float, hi: float, fmt: str = ".3g",
                    is_log: bool = False, target: Any = None,
                    dtype: str = "float", choices: list = (),
                    raw_str: str | None = None,
                    on_change=None) -> int:
        sliders.append(dict(
            label=label, key=key, val=val, lo=lo, hi=hi,
            fmt=fmt, is_log=is_log,
            rect=pygame.Rect(8, y, self.PANEL_W - 16, self._SLIDER_H),
            target=target,   # explicit write target; None → resolved in _set_slider_val
            dtype=dtype, choices=list(choices), raw_str=raw_str,
            on_change=on_change,
        ))
        return y + self._SLIDER_H + self._SLIDER_PAD + 12  # label + slider

    def render(self) -> pygame.Surface | None:
        if self._patch is None:
            return None
        font = self._font_()
        fh = font.get_height()
        pw = self.PANEL_W

        voice      = next((p  for p  in self._patch.voices      if p.key  == self._active_key), None)
        lfo        = next((l  for l  in self._patch.lfos        if l.key  == self._active_key), None)
        mixer      = next((m  for m  in self._patch.mixers      if m.key  == self._active_key), None)
        router     = _router_instance_for_active_key(self._patch, self._active_key)
        param_node = next((pn for pn in self._patch.param_nodes if pn.key == self._active_key), None)
        module     = next((m  for m  in self._patch.modules     if m.key  == self._active_key), None)
        control    = next((cs for cs in self._patch.controls    if cs.key == self._active_key), None)
        system_dev = self._patch.system_audio if self._active_key == "__system__" else None
        obj = voice or lfo or mixer or param_node or module or control or system_dev

        # Target: voice/lfo/mixer/module/control if one is active, otherwise patch global knobs
        if router is not None:
            target = router
            knob_list = []
            header_lbl = f"{router.label} [{router.router_type}]"
        elif obj is not None:
            # Ensure granular sub-spec exists before building knob list so that
            # visible_when-gated granular knobs read real values, not None.
            if voice is not None and getattr(voice, "emission_mode", "single") == "granular":
                _ensure_granular(voice)
            target     = obj
            knob_list: list[KnobSpec] = type(obj).knobs() if hasattr(type(obj), "knobs") else []
            header_lbl = getattr(obj, "label", "System Device" if system_dev is not None else "—")
        else:
            target     = self._patch
            knob_list  = AnalyticPatch.knobs()
            header_lbl = "— Patch —"

        # Build slider list from knob descriptors
        sliders: list[dict] = []
        y = fh + 10
        last_group: str = ""

        # ControlSurface: render each sub-slider via ControlSlider.knobs(), one group per slider
        if control is not None:
            sliders.append(dict(
                label="— Surface —", key="__section__",
                val=0, lo=0, hi=1, fmt="", is_log=False,
                rect=pygame.Rect(0, y, pw, fh + 4), target=None,
            ))
            y += 2 * fh + 8  # fh+4 header + fh+2 label clearance + 2 gap
            # Surface-level label knob
            for k in ControlSurface.knobs():
                raw = _get_nested_attr(control, k.name)
                float_val, lo, hi = 0.0, 0.0, 1.0
                y = self._add_slider(sliders, y, k.label, k.name, float_val, lo, hi, k.fmt,
                                     False, target=control, dtype=k.dtype,
                                     choices=k.choices, raw_str=str(raw) if k.dtype == "str" else None)
                if k.dtype == "str":
                    sliders[-1]["str_val"] = str(raw) if raw is not None else str(k.default or "")
            for i, cs_sl in enumerate(control.sliders):
                sliders.append(dict(
                    label=f"— {cs_sl.label} —", key="__section__",
                    val=0, lo=0, hi=1, fmt="", is_log=False,
                    rect=pygame.Rect(0, y, pw, fh + 4), target=None,
                ))
                y += 2 * fh + 8  # fh+4 header + fh+2 label clearance + 2 gap
                for k in ControlSlider.knobs():
                    raw = _get_nested_attr(cs_sl, k.name)
                    if k.dtype == "choice":
                        raw_s = str(raw) if raw is not None else ""
                        ci = k.choices.index(raw_s) if raw_s in k.choices else 0
                        float_val, lo, hi = float(ci), 0.0, float(max(0, len(k.choices) - 1))
                    elif k.dtype == "str":
                        float_val, lo, hi = 0.0, 0.0, 1.0
                    elif k.dtype in ("bool",):
                        float_val, lo, hi = float(bool(raw)), 0.0, 1.0
                    elif k.dtype == "int":
                        float_val = float(raw) if raw is not None else float(k.default or 0)
                        lo, hi = k.low, k.high
                    else:
                        try:
                            float_val = float(raw) if raw is not None else float(k.default or 0.0)
                        except (TypeError, ValueError):
                            float_val = 0.0
                        lo, hi = k.low, k.high
                    y = self._add_slider(sliders, y, k.label, k.name, float_val, lo, hi, k.fmt,
                                         k.is_log, target=cs_sl,
                                         dtype=k.dtype, choices=k.choices,
                                         raw_str=str(raw) if k.dtype == "str" else None)
                    if k.dtype == "str":
                        sliders[-1]["str_val"] = str(raw) if raw is not None else str(k.default or "")

        # For ControlSurface, the custom section above already rendered everything;
        # skip the generic knob_list loop to avoid duplicating the label knob.
        if control is None:
            for k in knob_list:
                # Visibility gate
                if k.visible_when is not None:
                    cond = str(_get_nested_attr(target, k.visible_when[0]))
                    if cond != k.visible_when[1]:
                        continue
                _slider_target = target
                # Section header when group changes
                if k.group and k.group != last_group:
                    last_group = k.group
                    sliders.append(dict(
                        label=f"— {k.group} —", key="__section__",
                        val=0, lo=0, hi=1, fmt="", is_log=False,
                        rect=pygame.Rect(0, y, pw, fh + 4), target=None,
                    ))
                    y += 2 * fh + 8  # fh+4 header + fh+2 label clearance + 2 gap
                # Resolve current value → float for slider position
                raw = _get_nested_attr(target, k.name)
                if k.dtype == "choice":
                    raw_s = str(raw) if raw is not None else ""
                    ci = k.choices.index(raw_s) if raw_s in k.choices else 0
                    float_val, lo, hi = float(ci), 0.0, float(max(0, len(k.choices) - 1))
                elif k.dtype == "str":
                    # String knobs are rendered as text labels, not sliders — skip float conversion
                    float_val, lo, hi = 0.0, 0.0, 1.0
                elif k.dtype == "int":
                    float_val = float(raw) if raw is not None else float(k.default or 0)
                    lo, hi = k.low, k.high
                elif k.dtype == "bool":
                    float_val, lo, hi = (1.0 if raw else 0.0), 0.0, 1.0
                else:
                    float_val = float(raw) if raw is not None else float(k.default or 0.0)
                    lo, hi = k.low, k.high
                y = self._add_slider(sliders, y, k.label, k.name, float_val, lo, hi, k.fmt, k.is_log, target=_slider_target)
                if k.dtype == "str":
                    sliders[-1]["str_val"] = str(raw) if raw is not None else str(k.default or "")
                if k.choices:
                    sliders[-1]["choices"] = list(k.choices)

        if system_dev is not None:
            _refresh_system_audio_report(system_dev)
            sliders.append(dict(
                label="— Devices —", key="__section__",
                val=0, lo=0, hi=1, fmt="", is_log=False,
                rect=pygame.Rect(0, y, pw, fh + 4), target=None,
            ))
            y += 2 * fh + 8  # fh+4 header + fh+2 label clearance + 2 gap

            _out_opts = ["(default)"] + list(system_dev._reported_output_devices)
            _in_opts = ["(none)", "(default)"] + list(system_dev._reported_input_devices)
            _out_cur = system_dev.output_device_name if system_dev.output_device_name in _out_opts else "(default)"
            _in_cur = system_dev.input_device_name if system_dev.input_device_name in _in_opts else (
                "(default)" if system_dev.input_channels > 0 else "(none)"
            )

            def _make_out_dev_cb(_sys=system_dev, _patch=self._patch, _opts=_out_opts):
                def _cb(idx):
                    chosen = _opts[int(round(idx))] if 0 <= int(round(idx)) < len(_opts) else "(default)"
                    _sys.output_device_name = "" if chosen == "(default)" else chosen
                    _refresh_system_audio_report(_sys)
                return _cb

            def _make_in_dev_cb(_sys=system_dev, _patch=self._patch, _opts=_in_opts):
                def _cb(idx):
                    chosen = _opts[int(round(idx))] if 0 <= int(round(idx)) < len(_opts) else "(none)"
                    _sys.input_device_name = "" if chosen in {"(none)", "(default)"} else chosen
                    _refresh_system_audio_report(_sys)
                return _cb

            y = self._add_slider(
                sliders, y, "Output Dev", "__system_output_device__",
                float(_out_opts.index(_out_cur)), 0.0, float(max(0, len(_out_opts) - 1)), ".0f", False,
                target=None, dtype="choice", choices=_out_opts, on_change=_make_out_dev_cb())
            y = self._add_slider(
                sliders, y, "Input Dev", "__system_input_device__",
                float(_in_opts.index(_in_cur)), 0.0, float(max(0, len(_in_opts) - 1)), ".0f", False,
                target=None, dtype="choice", choices=_in_opts, on_change=_make_in_dev_cb())

            for _label, _text in [
                ("Preview Backend", f"{system_dev._preview_backend} ({system_dev._preview_backend_channels}ch negotiated)"),
                ("Output Probe", f"cfg {system_dev.output_channels}ch -> {system_dev._reported_output_name or '(missing)'} / {system_dev._reported_output_hw_channels}ch @ {system_dev._reported_output_hw_rate}Hz"),
                ("Input Probe", f"cfg {system_dev.input_channels}ch -> {system_dev._reported_input_name or '(missing)'} / {system_dev._reported_input_hw_channels}ch @ {system_dev._reported_input_hw_rate}Hz"),
            ]:
                sliders.append(dict(
                    label=_label, key=f"__sys_info_{len(sliders)}__",
                    val=0.0, lo=0.0, hi=1.0, fmt="", is_log=False,
                    rect=pygame.Rect(8, y, self.PANEL_W - 16, self._SLIDER_H),
                    target=None, dtype="str", choices=[], raw_str=_text, str_val=_text,
                ))
                y += self._SLIDER_H + self._SLIDER_PAD + 12

        # Multi-channel LFO channel list — rendered after the standard knobs.
        if module is not None and module.module_type == "lfo":
            sliders.append(dict(
                label="— LFO Channels —", key="__section__",
                val=0, lo=0, hi=1, fmt="", is_log=False,
                rect=pygame.Rect(0, y, pw, fh + 4), target=None,
            ))
            y += 2 * fh + 8  # fh+4 header + fh+2 label clearance + 2 gap
            _lfo_shapes = AnalyticModule._LFO_SHAPES
            for _ci, _lch in enumerate(module.lfo_channels):
                sliders.append(dict(
                    label=f"— ch{_ci} —", key=f"__lfo_ch_hdr_{_ci}__",
                    val=0, lo=0, hi=1, fmt="", is_log=False,
                    rect=pygame.Rect(0, y, pw, fh + 4), target=None,
                ))
                y += 2 * fh + 8  # fh+4 header + fh+2 label clearance + 2 gap
                for _attr, _lbl, _lo, _hi, _fmt in (
                    ("scale",        "Scale",   0.0,  4.0,  ".3f"),
                    ("amplitude",    "Amp",     0.0,  4.0,  ".3f"),
                    ("rate_hz",      "Rate Hz", 0.01, 50.0, ".3f"),
                    ("tension",      "Tension", 0.01, 4.0,  ".3f"),
                    ("phase_offset", "Phase",   -math.pi, math.pi, ".3f"),
                ):
                    _val = float(_lch.get(_attr, 1.0))
                    def _make_ch_cb(_mod=module, _i=_ci, _a=_attr):
                        def _cb(v):
                            if _i < len(_mod.lfo_channels):
                                _mod.lfo_channels[_i][_a] = v
                        return _cb
                    y = self._add_slider(
                        sliders, y, _lbl, f"__lfo_ch_{_ci}_{_attr}__",
                        _val, _lo, _hi, _fmt, False,
                        target=None, on_change=_make_ch_cb())
                # Shape choice
                _shape_val = _lch.get("shape", "Sine")
                _shape_idx = float(_lfo_shapes.index(_shape_val) if _shape_val in _lfo_shapes else 0)
                def _make_shape_cb(_mod=module, _i=_ci):
                    def _cb(idx):
                        if _i < len(_mod.lfo_channels):
                            _mod.lfo_channels[_i]["shape"] = (
                                _lfo_shapes[int(idx)] if int(idx) < len(_lfo_shapes) else "Sine")
                    return _cb
                y = self._add_slider(
                    sliders, y, "Shape", f"__lfo_ch_{_ci}_shape__",
                    _shape_idx, 0.0, float(len(_lfo_shapes) - 1), ".0f", False,
                    target=None, dtype="choice", choices=_lfo_shapes,
                    on_change=_make_shape_cb())
                # Resample choice (ZOH decimation factors)
                _rs_opts = ["1", "2", "4", "8", "16", "32", "64", "128", "256"]
                _rs_cur  = str(int(_lch.get("resample", 1)))
                _rs_idx  = float(_rs_opts.index(_rs_cur) if _rs_cur in _rs_opts else 0)
                def _make_rs_cb(_mod=module, _i=_ci, _opts=_rs_opts):
                    def _cb(idx):
                        if _i < len(_mod.lfo_channels):
                            _mod.lfo_channels[_i]["resample"] = int(_opts[int(idx)])
                    return _cb
                y = self._add_slider(
                    sliders, y, "Resample", f"__lfo_ch_{_ci}_resample__",
                    _rs_idx, 0.0, float(len(_rs_opts) - 1), ".0f", False,
                    target=None, dtype="choice", choices=_rs_opts,
                    on_change=_make_rs_cb())
                # Slew order choice
                _slew_order_opts = ["1st", "2nd"]
                _slew_order_cur  = max(1, min(2, int(_lch.get("slew_order", 1)))) - 1
                def _make_slew_ord_cb(_mod=module, _i=_ci):
                    def _cb(idx):
                        if _i < len(_mod.lfo_channels):
                            _mod.lfo_channels[_i]["slew_order"] = int(idx) + 1
                    return _cb
                y = self._add_slider(
                    sliders, y, "Slew Ord", f"__lfo_ch_{_ci}_slew_order__",
                    float(_slew_order_cur), 0.0, 1.0, ".0f", False,
                    target=None, dtype="choice", choices=_slew_order_opts,
                    on_change=_make_slew_ord_cb())
                # Slew amount
                _slew_val = float(_lch.get("slew", 0.0))
                def _make_slew_cb(_mod=module, _i=_ci):
                    def _cb(v):
                        if _i < len(_mod.lfo_channels):
                            _mod.lfo_channels[_i]["slew"] = v
                    return _cb
                y = self._add_slider(
                    sliders, y, "Slew", f"__lfo_ch_{_ci}_slew__",
                    _slew_val, 0.0, 1.0, ".3f", False,
                    target=None, on_change=_make_slew_cb())
                # Remove channel button
                def _make_rm_lfo_ch(_mod=module, _i=_ci):
                    def _action():
                        if _i < len(_mod.lfo_channels):
                            _mod.lfo_channels.pop(_i)
                    return _action
                sliders.append(dict(
                    label=f"✕ Remove ch{_ci}", key=f"__rm_lfo_ch_{_ci}__",
                    val=0, lo=0, hi=1, fmt="", is_log=False,
                    rect=pygame.Rect(8, y, pw - 16, self._SLIDER_H + 2),
                    target=None, action=_make_rm_lfo_ch(),
                ))
                y += self._SLIDER_H + self._SLIDER_PAD + 6
            # Add channel button
            def _add_lfo_ch(_mod=module):
                _mod.lfo_channels.append(AnalyticModule.default_lfo_channel())
            sliders.append(dict(
                label="+ Add Channel", key="__add_lfo_ch__",
                val=0, lo=0, hi=1, fmt="", is_log=False,
                rect=pygame.Rect(8, y, pw - 16, self._SLIDER_H + 4),
                target=None, action=_add_lfo_ch,
            ))
            y += self._SLIDER_H + self._SLIDER_PAD + 8

        # State machine module UI
        if module is not None and module.module_type == "state_machine":
            sliders.append(dict(
                label="— State Machine —", key="__section__",
                val=0, lo=0, hi=1, fmt="", is_log=False,
                rect=pygame.Rect(0, y, pw, fh + 4), target=None,
            ))
            y += 2 * fh + 8  # fh+4 header + fh+2 label clearance + 2 gap

            # Plugin selector
            _sm_plugins = _sm_plugin_list()
            _sm_none_list = ["(none)"] + _sm_plugins
            _sm_cur = module.sm_plugin if module.sm_plugin in _sm_plugins else ""
            _sm_idx = float(_sm_none_list.index(module.sm_plugin) if module.sm_plugin in _sm_none_list else 0)

            def _make_sm_plugin_cb(_mod=module, _opts=_sm_none_list):
                def _cb(idx):
                    chosen = _opts[int(idx)] if int(idx) < len(_opts) else "(none)"
                    _mod.sm_plugin = "" if chosen == "(none)" else chosen
                    # Reload plugin metadata into mod fields
                    if _mod.sm_plugin:
                        _plug = _load_sm_plugin(_mod.sm_plugin)
                        if _plug is not None:
                            _mod.sm_vars  = _sm_plugin_output_vars(_plug)
                            _mod.sm_state_vars = _sm_plugin_state_vars(_plug)
                            _mod.sm_items = _sm_plugin_item_names(_plug, _mod.sm_n_items)
                            _defaults = _sm_plugin_default_params(_plug)
                            _mod.sm_params = {**_defaults, **dict(_mod.sm_params)}
                            for _k in list(_mod.sm_params):
                                if _k not in _defaults:
                                    _mod.sm_params.pop(_k)
                    else:
                        _mod.sm_vars  = []
                        _mod.sm_state_vars = []
                        _mod.sm_items = []
                        _mod.sm_params = {}
                    _mod._sm_state     = {}
                    _mod._sm_out_cache = {}
                    _mod._sm_aux_state = {}
                return _cb
            y = self._add_slider(
                sliders, y, "Plugin", "__sm_plugin__",
                _sm_idx, 0.0, float(max(0, len(_sm_none_list) - 1)), ".0f", False,
                target=None, dtype="choice", choices=_sm_none_list,
                on_change=_make_sm_plugin_cb())

            # N items slider
            def _make_sm_nitems_cb(_mod=module):
                def _cb(v):
                    n = max(1, int(round(v)))
                    _mod.sm_n_items = n
                    _plug = _load_sm_plugin(_mod.sm_plugin) if _mod.sm_plugin else None
                    _mod.sm_items   = _sm_plugin_item_names(_plug, n)
                    _mod._sm_state     = {}
                    _mod._sm_out_cache = {}
                    _mod._sm_aux_state = {}
                return _cb
            y = self._add_slider(
                sliders, y, "N Items", "__sm_n_items__",
                float(module.sm_n_items), 1.0, 32.0, ".0f", False,
                target=None, on_change=_make_sm_nitems_cb())

            # Torch toggle
            _sm_torch_opts = ["numpy", "torch"]
            _sm_torch_idx  = float(1 if module.sm_use_torch else 0)

            def _make_sm_torch_cb(_mod=module):
                def _cb(idx):
                    _mod.sm_use_torch = (int(idx) == 1)
                return _cb
            y = self._add_slider(
                sliders, y, "Backend", "__sm_torch__",
                _sm_torch_idx, 0.0, 1.0, ".0f", False,
                target=None, dtype="choice", choices=_sm_torch_opts,
                on_change=_make_sm_torch_cb())

            # Read-only: items and vars from loaded plugin
            if module.sm_plugin:
                _items_str = ", ".join(module.sm_items) if module.sm_items else "—"
                _vars_str  = ", ".join(module.sm_vars)  if module.sm_vars  else "—"
                _state_vars_str = ", ".join(module.sm_state_vars) if module.sm_state_vars else "—"
                for _lbl, _val_str in (("Items", _items_str),
                                       ("Outputs", _vars_str),
                                       ("State", _state_vars_str)):
                    sliders.append(dict(
                        label=f"{_lbl}: {_val_str}", key=f"__sm_ro_{_lbl}__",
                        val=0, lo=0, hi=1, fmt="", is_log=False,
                        rect=pygame.Rect(0, y, pw, fh + 4), target=None,
                    ))
                    y += 2 * fh + 8  # fh+4 row + fh+2 label clearance + 2 gap
                _plug = _load_sm_plugin(module.sm_plugin)
                _param_specs = _sm_plugin_param_specs(_plug)
                _last_sm_group = ""
                for _spec in _param_specs:
                    _group = _spec.get("group", "Plugin")
                    if _group != _last_sm_group:
                        sliders.append(dict(
                            label=f"— {_group} —", key="__section__",
                            val=0, lo=0, hi=1, fmt="", is_log=False,
                            rect=pygame.Rect(0, y, pw, fh + 4), target=None,
                        ))
                        y += 2 * fh + 8  # fh+4 header + fh+2 label clearance + 2 gap
                        _last_sm_group = _group
                    _name = _spec["name"]
                    _dtype = _spec["dtype"]
                    _cur = module.sm_params.get(_name, _spec["default"])
                    if _dtype == "choice":
                        _choices = list(_spec["choices"])
                        _cur_s = str(_cur)
                        _ci = _choices.index(_cur_s) if _cur_s in _choices else 0
                        def _make_sm_param_choice_cb(_mod=module, _nm=_name, _chs=_choices):
                            def _cb(idx):
                                if 0 <= int(idx) < len(_chs):
                                    _mod.sm_params[_nm] = _chs[int(idx)]
                            return _cb
                        y = self._add_slider(
                            sliders, y, _spec["label"], f"__sm_param_{_name}__",
                            float(_ci), 0.0, float(max(0, len(_choices) - 1)),
                            ".0f", False, target=None, dtype="choice", choices=_choices,
                            on_change=_make_sm_param_choice_cb())
                    else:
                        _float_val = float(_cur)
                        def _make_sm_param_float_cb(_mod=module, _nm=_name):
                            def _cb(v):
                                _mod.sm_params[_nm] = float(v)
                            return _cb
                        y = self._add_slider(
                            sliders, y, _spec["label"], f"__sm_param_{_name}__",
                            _float_val, float(_spec["low"]), float(_spec["high"]),
                            _spec["fmt"], bool(_spec["is_log"]), target=None,
                            on_change=_make_sm_param_float_cb())

            # Reset state button
            def _make_sm_reset(_mod=module):
                def _action():
                    _mod._sm_state     = {}
                    _mod._sm_out_cache = {}
                    _mod._sm_aux_state = {}
                return _action
            sliders.append(dict(
                label="Reset State", key="__sm_reset__",
                val=0, lo=0, hi=1, fmt="", is_log=False,
                rect=pygame.Rect(8, y, pw - 16, self._SLIDER_H + 2),
                target=None, action=_make_sm_reset(),
            ))
            y += self._SLIDER_H + self._SLIDER_PAD + 8

        # When a param_node is active, append the dynamic Targets section below
        # the static knobs (label / extractor / default / low / high).
        if param_node is not None:
            pn = param_node
            node_specs = _param_target_node_specs(self._patch)
            node_keys = [""] + [k for k, _ in node_specs]
            node_labels = ["—"] + [lbl for _, lbl in node_specs]

            sliders.append(dict(
                label="— Targets —", key="__section__",
                val=0, lo=0, hi=1, fmt="", is_log=False,
                rect=pygame.Rect(0, y, pw, fh + 4), target=None,
            ))
            y += 2 * fh + 8  # fh+4 header + fh+2 label clearance + 2 gap

            for ti in range(len(pn.targets)):
                tgt = pn.targets[ti]

                # Node dropdown
                cur_vk  = tgt.get("voice_key", "")
                cur_vi  = node_keys.index(cur_vk) if cur_vk in node_keys else 0
                _ti_v = ti  # capture for closure

                def _make_voice_cb(_pn=pn, _ti=_ti_v, _vkeys=node_keys):
                    def _cb(idx):
                        _pn.targets[_ti]["voice_key"] = _vkeys[idx] if idx < len(_vkeys) else ""
                    return _cb

                sliders.append(dict(
                    label=f"T{ti} node", key=f"__tgt_voice_{ti}__",
                    val=float(cur_vi), lo=0.0, hi=float(max(0, len(node_labels) - 1)),
                    fmt=".0f", is_log=False,
                    rect=pygame.Rect(8, y, pw - 16, self._SLIDER_H),
                    target=None,
                    dtype="choice", choices=node_labels, raw_str=None,
                    on_change=_make_voice_cb(),
                ))
                y += self._SLIDER_H + self._SLIDER_PAD + 12

                # Attr dropdown — choices depend on the selected target node.
                _attr_specs = _param_target_attr_specs(self._patch, cur_vk)
                _tgt_attrs = [""] + [raw for raw, _ in _attr_specs]
                _tgt_attr_labels = ["—"] + [lbl for _, lbl in _attr_specs]
                cur_at  = tgt.get("attr", "")
                cur_ai  = _tgt_attrs.index(cur_at) if cur_at in _tgt_attrs else 0
                _ti_a = ti  # capture for closure

                def _make_attr_cb(_pn=pn, _ti=_ti_a, _attrs=_tgt_attrs):
                    def _cb(idx):
                        _pn.targets[_ti]["attr"] = _attrs[idx] if idx < len(_attrs) else ""
                    return _cb

                sliders.append(dict(
                    label=f"T{ti} attr", key=f"__tgt_attr_{ti}__",
                    val=float(cur_ai), lo=0.0, hi=float(max(0, len(_tgt_attr_labels) - 1)),
                    fmt=".0f", is_log=False,
                    rect=pygame.Rect(8, y, pw - 16, self._SLIDER_H),
                    target=None,
                    dtype="choice", choices=_tgt_attr_labels, raw_str=None,
                    on_change=_make_attr_cb(),
                ))
                y += self._SLIDER_H + self._SLIDER_PAD + 12

                # Remove button
                _ti_r = ti  # capture for closure

                def _make_rm_cb(_pn=pn, _ti=_ti_r):
                    def _action():
                        if _ti < len(_pn.targets):
                            _pn.targets.pop(_ti)
                    return _action

                sliders.append(dict(
                    label=f"✕ Remove target {ti}", key=f"__rm_target_{ti}__",
                    val=0, lo=0, hi=1, fmt="", is_log=False,
                    rect=pygame.Rect(8, y, pw - 16, self._SLIDER_H + 2),
                    target=None, action=_make_rm_cb(),
                ))
                y += self._SLIDER_H + self._SLIDER_PAD + 6

            # Add Target button
            def _add_target(_pn=pn):
                _pn.targets.append({"voice_key": "", "attr": ""})

            sliders.append(dict(
                label="+ Add Target", key="__add_target__",
                val=0, lo=0, hi=1, fmt="", is_log=False,
                rect=pygame.Rect(8, y, pw - 16, self._SLIDER_H + 4),
                target=None, action=_add_target,
            ))
            y += self._SLIDER_H + self._SLIDER_PAD + 8

        # When a mixer node is active, append RoutingGraph controls below
        if mixer is not None or router is not None:
            routing = router.graph if router is not None else self._patch.routing
            routing_knobs = routing.knobs() if hasattr(routing, 'knobs') else []
            last_group = ""
            for k in routing_knobs:
                if k.group and k.group != last_group:
                    last_group = k.group
                    sliders.append(dict(
                        label=f"— {k.group} —", key="__section__",
                        val=0, lo=0, hi=1, fmt="", is_log=False,
                        rect=pygame.Rect(0, y, pw, fh + 4), target=None,
                    ))
                    y += 2 * fh + 8  # fh+4 header + fh+2 label clearance + 2 gap
                raw = _get_nested_attr(routing, k.name)
                if k.dtype == "choice":
                    raw_s = str(raw) if raw is not None else ""
                    ci = k.choices.index(raw_s) if raw_s in k.choices else 0
                    float_val, lo, hi = float(ci), 0.0, float(max(0, len(k.choices) - 1))
                elif k.dtype == "bool":
                    float_val, lo, hi = (1.0 if raw else 0.0), 0.0, 1.0
                elif k.dtype == "int":
                    float_val = float(raw) if raw is not None else float(k.default or 0)
                    lo, hi = k.low, k.high
                else:
                    float_val = float(raw) if raw is not None else float(k.default or 0.0)
                    lo, hi = k.low, k.high
                y = self._add_slider(sliders, y, k.label, k.name, float_val, lo, hi, k.fmt, k.is_log, target=routing)
                if k.choices:
                    sliders[-1]["choices"] = list(k.choices)

        if voice is not None:
            if voice.piecewise_env is None:
                voice.piecewise_env = PiecewiseVoiceEnvelope()
            voice.env_type = "piecewise"

        self._sliders = sliders

        surf_h = max(200, y + 20)
        surf = pygame.Surface((pw, surf_h))
        surf.fill(_PY_BG)

        # Header
        pygame.draw.rect(surf, (30, 30, 40), (0, 0, pw, fh + 6))
        surf.blit(font.render(f"  {header_lbl}", True, _PY_TXT), (8, 3))
        self._role_defaults_rect = None
        if voice is not None and getattr(voice, "voice_role", "signal") != "signal":
            btn_lbl = "[ ↺ Role defaults ]"
            btn_sf = font.render(btn_lbl, True, (215, 225, 245))
            btn_pad_x = 8
            btn_w = btn_sf.get_width() + btn_pad_x * 2
            btn_h = fh + 2
            btn_r = pygame.Rect(pw - btn_w - 8, 2, btn_w, btn_h)
            pygame.draw.rect(surf, (55, 70, 108), btn_r, border_radius=3)
            pygame.draw.rect(surf, (105, 135, 200), btn_r, 1, border_radius=3)
            surf.blit(btn_sf, (btn_r.x + btn_pad_x, btn_r.y + 1))
            self._role_defaults_rect = btn_r

        # Draw sliders
        for sl in sliders:
            r = sl["rect"]
            if sl["key"] == "__section__":
                pygame.draw.line(surf, (50, 50, 64),
                                 (4, r.y + r.h - 2), (pw - 4, r.y + r.h - 2))
                surf.blit(font.render(sl["label"], True, (130, 160, 200)), (6, r.y))
                continue
            # Action buttons (Add/Remove target): render as a clickable button, no label row
            if "action" in sl:
                pygame.draw.rect(surf, (50, 60, 90), r, border_radius=3)
                pygame.draw.rect(surf, (80, 100, 160), r, width=1, border_radius=3)
                btn_s = font.render(sl["label"], True, (160, 200, 255))
                surf.blit(btn_s, (r.x + r.w // 2 - btn_s.get_width() // 2,
                                  r.y + r.h // 2 - btn_s.get_height() // 2))
                continue
            # Label row
            lbl_y = r.y - fh - 2
            surf.blit(font.render(sl["label"], True, _PY_DIM), (8, lbl_y))
            if sl.get("choices"):
                ci = max(0, min(len(sl["choices"]) - 1, round(sl["val"])))
                val_str = sl["choices"][ci]
            elif "str_val" in sl:
                idx_here = sliders.index(sl)
                if idx_here == self._text_edit_idx:
                    val_str = self._text_edit_buf + "|"  # blinking cursor
                else:
                    val_str = sl["str_val"] or "—"
            else:
                val_str = format(sl["val"], sl["fmt"])
            val_surf = font.render(val_str, True, _PY_TXT)
            surf.blit(val_surf, (pw - val_surf.get_width() - 8, lbl_y))
            # Track — skip for str knobs (display-only, no slider bar)
            if "str_val" in sl:
                continue
            pygame.draw.rect(surf, (40, 40, 50), r, border_radius=3)
            # Snap tick marks for tuning root
            if sl.get("key") == "tuning.root_hz":
                _lo_t, _hi_t = sl["lo"], sl["hi"]
                for _snap in _ROOT_HZ_SNAPS:
                    if _lo_t < _snap < _hi_t:
                        _frac_t = math.log(_snap / _lo_t) / math.log(_hi_t / _lo_t)
                        _tx = r.x + int(_frac_t * r.w)
                        pygame.draw.line(surf, (100, 160, 255), (_tx, r.y + 1), (_tx, r.y + r.h - 1))
            # Snap tick marks for sample-rate sliders
            if sl.get("key") in ("preview_sr", "export_sample_rate"):
                _lo_sr, _hi_sr = sl["lo"], sl["hi"]
                for _snap_sr in _COMMON_SAMPLE_RATES:
                    if _lo_sr <= _snap_sr <= _hi_sr:
                        _frac_sr = (_snap_sr - _lo_sr) / max(_hi_sr - _lo_sr, 1)
                        _tx_sr = r.x + int(_frac_sr * r.w)
                        pygame.draw.line(surf, (100, 200, 160), (_tx_sr, r.y + 1), (_tx_sr, r.y + r.h - 1))
            # Thumb
            lo, hi, val = sl["lo"], sl["hi"], sl["val"]
            if sl["is_log"] and lo > 0 and hi > 0:
                frac = math.log(max(val, lo) / lo) / math.log(hi / lo)
            else:
                frac = (val - lo) / max(hi - lo, 1e-9)
            frac = max(0.0, min(1.0, frac))
            thumb_x = r.x + int(frac * r.w)
            pygame.draw.rect(surf, _PY_ACT,
                             pygame.Rect(r.x, r.y, thumb_x - r.x + 4, r.h),
                             border_radius=3)

        return self._apply_panel_scroll(surf)

    def _set_slider_val(self, idx: int, val: float) -> None:
        """Apply a slider value change to the patch object via knob descriptors."""
        if self._patch is None or idx < 0 or idx >= len(self._sliders):
            return
        sl = self._sliders[idx]
        key = sl["key"]
        if key == "__section__":
            return
        # Dynamic choice sliders (ParamNode voice/attr selection): delegate to on_change
        on_change = sl.get("on_change")
        if on_change is not None:
            val = max(sl["lo"], min(sl["hi"], val))
            sl["val"] = val
            if sl.get("dtype", "float") == "float":
                on_change(val)
            else:
                on_change(int(round(val)))
            self._dirty = True
            return

        val = max(sl["lo"], min(sl["hi"], val))
        # Snap tuning root to popular reference pitches
        if key == "tuning.root_hz":
            for _snap in _ROOT_HZ_SNAPS:
                if abs(val - _snap) / _snap < _ROOT_HZ_SNAP_TOL:
                    val = _snap
                    break
        # Snap sample-rate sliders to standard rates
        if key in ("preview_sr", "export_sample_rate"):
            nearest = min(_COMMON_SAMPLE_RATES, key=lambda s: abs(s - val))
            val = float(nearest)
        sl["val"] = val

        # Use the explicit target stored at render time when available
        explicit_target = sl.get("target")
        if explicit_target is not None:
            obj = explicit_target
            # Build a knob list from the target type
            knob_list: list[KnobSpec] = type(obj).knobs() if hasattr(type(obj), "knobs") else []
            knob = next((k for k in knob_list if k.name == key), None)
        else:
            voice = next((p  for p  in self._patch.voices   if p.key == self._active_key), None)
            lfo   = next((l  for l  in self._patch.lfos     if l.key == self._active_key), None)
            mod   = next((m  for m  in self._patch.modules  if m.key == self._active_key), None)
            sysd  = self._patch.system_audio if self._active_key == "__system__" else None
            obj = voice or lfo or mod or sysd
            if obj is not None:
                knob_list = type(obj).knobs() if hasattr(type(obj), "knobs") else []
                knob = next((k for k in knob_list if k.name == key), None)
            else:
                knob_list = AnalyticPatch.knobs()
                knob = next((k for k in knob_list if k.name == key), None)
                obj = self._patch

        if knob is None:
            return
        _set_nested_attr(obj, key, val, knob)
        # Preset knob: reload the full GlobalTuning when preset_name changes.
        if key == "tuning.preset_name" and hasattr(obj, "tuning"):
            try:
                new_tuning = GlobalTuning.from_preset(obj.tuning.preset_name)
                obj.tuning = new_tuning
            except (ValueError, KeyError):
                pass
            self._dirty = True
        # Coherence meta-knob: propagate with_coherence so sibling sliders update
        if key == "granular.grain_coherence":
            voice_obj = next((v for v in self._patch.voices if v.key == self._active_key), None)
            if voice_obj is not None and getattr(voice_obj, "granular", None) is not None:
                voice_obj.granular = voice_obj.granular.with_coherence(float(val))
                self._dirty = True
        if isinstance(obj, SystemAudioDevice):
            if key in {"output_channels", "input_channels"}:
                obj.output_channels = max(1, int(obj.output_channels))
                obj.input_channels = max(0, int(obj.input_channels))
                _refresh_system_audio_report(obj)
                self._patch.routing.prune_keys(_patch_node_keys(self._patch))
            if key == "export_bit_depth":
                obj.export_bit_depth = 16 if obj.export_bit_depth < 20 else (24 if obj.export_bit_depth < 28 else 32)
        if knob.rebuild_layout:
            self._dirty = True

    def _px_to_slider_val(self, sl: dict, lx: int) -> float:
        r = sl["rect"]
        frac = max(0.0, min(1.0, (lx - r.x) / max(r.w, 1)))
        lo, hi = sl["lo"], sl["hi"]
        if sl["is_log"] and lo > 0 and hi > 0:
            return lo * (hi / lo) ** frac
        return lo + frac * (hi - lo)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self._patch is None:
            return False
        if self._handle_panel_wheel(event):
            return True

        rect = self.panel_rect
        sliders = self._sliders

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if not rect.collidepoint(event.pos):
                # Click outside panel commits any open text edit
                if self._text_edit_idx >= 0:
                    self._commit_text_edit()
                return False
            lx = event.pos[0] - rect.x
            ly = event.pos[1] - rect.y + self._panel_scroll_y
            if self._role_defaults_rect is not None and self._role_defaults_rect.collidepoint(lx, ly):
                if self._text_edit_idx >= 0:
                    self._commit_text_edit()
                voice = next((v for v in self._patch.voices if v.key == self._active_key), None)
                if voice is not None and getattr(voice, "voice_role", "signal") != "signal":
                    _apply_voice_role_preset(voice, voice.voice_role)
                    self._dirty = True
                    return True
            for i, sl in enumerate(sliders):
                if sl["key"] == "__section__":
                    continue
                if sl["rect"].collidepoint(lx, ly):
                    if "action" in sl:
                        # Button slider: invoke the action callable directly
                        if self._text_edit_idx >= 0:
                            self._commit_text_edit()
                        sl["action"]()
                        self._dirty = True
                        return True
                    if "str_val" in sl:
                        # Click on str-knob row: enter text-edit mode
                        if self._text_edit_idx != i:
                            if self._text_edit_idx >= 0:
                                self._commit_text_edit()
                            self._text_edit_idx = i
                            self._text_edit_buf = sl.get("str_val", "")
                            self._dirty = True
                        return True
                    # Commit any open text edit on slider interaction
                    if self._text_edit_idx >= 0:
                        self._commit_text_edit()
                    val = self._px_to_slider_val(sl, lx)
                    self._dragging_slider = i
                    self._set_slider_val(i, val)
                    return True
            return False

        elif event.type == MOUSEBUTTONUP and event.button == 1:
            self._dragging_slider = -1

        elif event.type == MOUSEMOTION:
            if self._dragging_slider >= 0 and rect.collidepoint(event.pos):
                lx = event.pos[0] - rect.x
                sl = sliders[self._dragging_slider]
                val = self._px_to_slider_val(sl, lx)
                self._set_slider_val(self._dragging_slider, val)
                return True

        elif event.type == pygame.KEYDOWN and self._text_edit_idx >= 0:
            k = event.key
            if k == pygame.K_RETURN or k == pygame.K_KP_ENTER:
                self._commit_text_edit()
            elif k == pygame.K_ESCAPE:
                self._text_edit_idx = -1
                self._text_edit_buf = ""
                self._dirty = True
            elif k == pygame.K_BACKSPACE:
                self._text_edit_buf = self._text_edit_buf[:-1]
                self._dirty = True
            else:
                ch = event.unicode
                if ch and ch.isprintable():
                    self._text_edit_buf += ch
                    self._dirty = True
            return True

        return False

    def _commit_text_edit(self) -> None:
        """Write the text edit buffer back to the target str-knob attribute."""
        idx = self._text_edit_idx
        if idx < 0 or idx >= len(self._sliders):
            self._text_edit_idx = -1
            return
        sl = self._sliders[idx]
        key = sl["key"]
        new_val = self._text_edit_buf.strip()
        sl["str_val"] = new_val

        # Find the target object and apply the string directly
        explicit_target = sl.get("target")
        if explicit_target is not None:
            obj = explicit_target
        else:
            voice = next((v for v in self._patch.voices if v.key == self._active_key), None)
            lfo   = next((l for l in self._patch.lfos   if l.key == self._active_key), None)
            pn    = next((p for p in self._patch.param_nodes if p.key == self._active_key), None)
            sysd  = self._patch.system_audio if self._active_key == "__system__" else None
            obj = voice or lfo or pn or sysd or self._patch

        # Resolve nested path for str attributes
        parts = key.split(".")
        target = obj
        for p in parts[:-1]:
            target = getattr(target, p, target)
        try:
            setattr(target, parts[-1], new_val)
        except (AttributeError, TypeError):
            pass

        self._text_edit_idx = -1
        self._text_edit_buf = ""
        self._dirty = True


# ---------------------------------------------------------------------------
# AnalyticDriverViewer — main application
# ---------------------------------------------------------------------------


def _cache_sidecar_path(json_path: str) -> str:
    """Derive the binary cavity-cache sidecar path from a .json patch path."""
    base, _ = os.path.splitext(json_path)
    return base + ".cache"


def _save_cavity_cache(json_path: str, cache: dict[str, dict]) -> None:
    """Pickle the per-module cavity cache dict to a binary sidecar file."""
    import pickle
    side = _cache_sidecar_path(json_path)
    if not cache:
        # Nothing to persist — remove stale sidecar if present.
        if os.path.isfile(side):
            try:
                os.remove(side)
            except OSError:
                pass
        return
    try:
        with open(side, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  cavity cache → {side}  ({len(cache)} modules)")
    except Exception as exc:
        print(f"  cavity cache write failed: {exc}")


def _load_cavity_cache(json_path: str) -> dict[str, dict]:
    """Load the cavity-cache sidecar if present.  Returns empty dict on miss."""
    import pickle
    side = _cache_sidecar_path(json_path)
    if not os.path.isfile(side):
        return {}
    try:
        with open(side, "rb") as f:
            data = pickle.load(f)  # noqa: S301  — trusted local file
        if isinstance(data, dict):
            print(f"  cavity cache ← {side}  ({len(data)} modules)")
            return data
    except Exception as exc:
        print(f"  cavity cache load skipped: {exc}")
    return {}


class AnalyticDriverViewer:
    """Main GL/Pygame application for the analytic synthesizer voice editor."""

    def __init__(self, patch_path: str | None = None) -> None:
        self.win_w = WIN_W_DEFAULT
        self.win_h = WIN_H_DEFAULT
        self.patch_path = patch_path

        if patch_path and os.path.isfile(patch_path):
            with open(patch_path, "r", encoding="utf-8") as f:
                self.patch = AnalyticPatch.from_dict(json.load(f))
        else:
            self.patch = AnalyticPatch.default_patch()

        # Persistent cavity scene/stream caches keyed by module key.
        # Loaded from the binary sidecar alongside the JSON patch.
        self._cavity_cache: dict[str, dict] = (
            _load_cavity_cache(patch_path) if patch_path else {}
        )

        self.active_key: str = (self.patch.voices[0].key
                                if self.patch.voices else "")

        self.canvas = EditorCanvas()
        self.patch_panel   = PatchPanel(side="left")
        self.partial_panel = PartialPanel(side="right")
        self.dock = PanelDock()
        self.dock.register("voices", self.patch_panel)
        self.dock.register("params", self.partial_panel)
        self.dock.left_key  = "voices"
        self.dock.right_key = "params"

        self._atlas = GlyphAtlas()
        self._font:  pygame.font.Font | None = None

        # Panel textures
        self._left_tex:  int = 0
        self._right_tex: int = 0

        # Preview playback
        self._preview_sound:   Any = None
        self._preview_channel: Any = None
        self._preview_dirty = True
        self._output_device: Any = None
        self._output_backend: str = ""
        self._output_buffer = bytearray()
        self._output_lock = threading.Lock()
        self._output_signature: tuple | None = None
        self._output_playing: bool = False
        self._input_capture_device: Any = None

        # Rebuild state
        self._needs_rebuild = True

        # Granular seed animation state
        self._seed_anim_tick:      int   = 0
        self._seed_anim_last_time: float = 0.0

        # Wire callbacks
        self.patch_panel.on_select       = self._on_select
        self.patch_panel.on_add_voice    = self._on_add_voice
        self.patch_panel.on_add_lfo      = self._on_add_lfo
        self.patch_panel.on_add_param    = self._on_add_param
        self.patch_panel.on_add_module   = self._on_add_module
        self.patch_panel.on_add_control  = self._on_add_control
        self.patch_panel.on_add_router   = self._on_add_router
        self.patch_panel.on_toggle_mute  = lambda _: self._mark_dirty()
        self.patch_panel.on_remove_voice = self._on_remove_voice
        self.patch_panel.on_deploy_chord = self._on_deploy_chord
        self.patch_panel.on_demo_play    = self._play_demo_sequence
        self.patch_panel.on_render_fund  = self._on_render_to_files
        self.patch_panel.on_render       = self._on_render_sequence
        self.partial_panel.on_open_piecewise_editor = self._open_piecewise_editor

    # ---- Callbacks ---------------------------------------------------------

    def _on_select(self, key: str) -> None:
        self.active_key = key
        self.canvas.active_key = key
        if hasattr(self.canvas, "routing_view"):
            self.canvas.routing_view.active_key = key
        self.canvas._score_scroll_y = 0
        # Auto-select an appropriate mode for the new node type
        is_mixer   = any(m.key == key for m in self.patch.mixers)
        is_router  = _router_instance_for_active_key(self.patch, key) is not None
        is_param   = any(pn.key == key for pn in self.patch.param_nodes)
        is_module  = any(m.key == key for m in self.patch.modules)
        is_control = any(cs.key == key for cs in self.patch.controls)
        if is_mixer or is_router:
            self.canvas.mode = EditorMode.ROUTING
        elif is_param:
            self.canvas.mode = EditorMode.PARAM_ROUTING
        elif is_module:
            self.canvas.mode = EditorMode.ROUTING
            self.canvas._sm_log_scroll = 0
        elif is_control:
            # Controls don't have routing-graph edges on the surface itself;
            # stay in WAVEFORM to show the parameter panel cleanly.
            if self.canvas.mode in (EditorMode.ROUTING, EditorMode.PARAM_ROUTING):
                self.canvas.mode = EditorMode.WAVEFORM
        else:
            if self.canvas.mode in (EditorMode.ROUTING, EditorMode.PARAM_ROUTING):
                self.canvas.mode = EditorMode.WAVEFORM
        modes, _ = self.canvas._visible_modes(self.patch)
        if self.canvas.mode not in modes and modes:
            self.canvas.mode = modes[0]
        self._needs_rebuild = True

    def _invalidate_graph_compile(self) -> None:
        clear_compiled_graph_cache(self.patch)

    def _on_add_voice(self) -> None:
        p = AnalyticVoice()
        p.label   = f"V{len(self.patch.voices) + 1}"
        p.freq_hz = 220.0 * (2 ** len(self.patch.voices))
        colors = [(100,180,255),(255,130,60),(120,220,120),(220,80,160),(200,200,80)]
        p.color = list(colors[len(self.patch.voices) % len(colors)])
        self.patch.voices.append(p)
        self.active_key = p.key
        self._invalidate_graph_compile()
        self._needs_rebuild = True

    def _on_add_lfo(self) -> None:
        l = LFODefinition()
        l.label   = f"LFO{len(self.patch.lfos) + 1}"
        l.rate_hz = 1.0
        self.patch.lfos.append(l)
        self.active_key = l.key
        self._invalidate_graph_compile()
        self._needs_rebuild = True

    def _mark_dirty(self) -> None:
        self._invalidate_graph_compile()
        self._needs_rebuild = True

    def _open_piecewise_editor(self, key: str) -> None:
        self._on_select(key)
        self.canvas.mode = EditorMode.PIECEWISE_EDITOR
        self._needs_rebuild = True

    def _refresh_input_capture(self) -> None:
        sysdev = self.patch.system_audio
        if self._input_capture_device is not None:
            try:
                if hasattr(self._input_capture_device, "abort"):
                    self._input_capture_device.abort()
                self._input_capture_device.close()
            except Exception:
                pass
            self._input_capture_device = None
        sysdev._input_buffers = [np.zeros(0, dtype=np.float32) for _ in range(max(0, sysdev.input_channels))]
        if sysdev.input_channels <= 0:
            return
        _refresh_system_audio_report(sysdev)

        def _write_capture(_patch, arr: np.ndarray, nch: int) -> None:
            try:
                if arr.size < nch:
                    return
                arr = arr[: (arr.size // nch) * nch].reshape(-1, nch)
                syscfg = _patch.system_audio
                bufs = list(getattr(syscfg, "_input_buffers", []))
                cfg_n = max(0, int(syscfg.input_channels))
                if len(bufs) < cfg_n:
                    bufs.extend(np.zeros(0, dtype=np.float32) for _ in range(cfg_n - len(bufs)))
                keep_n = max(2048, int(_patch.preview_sr * max(1.0, _patch.duration)))
                for ci in range(min(cfg_n, nch)):
                    merged = np.concatenate([bufs[ci], arr[:, ci]])
                    if len(merged) > keep_n:
                        merged = merged[-keep_n:]
                    bufs[ci] = merged.astype(np.float32, copy=False)
                if len(bufs) > cfg_n:
                    bufs = bufs[:cfg_n]
                syscfg._input_buffers = bufs
            except Exception:
                pass

        if not str(sysdev.input_device_name or "").strip():
            try:
                import sounddevice as sd
                def _sd_capture_cb(indata, frames, time_info, status, _patch=self.patch):
                    _write_capture(_patch, np.asarray(indata, dtype=np.float32).reshape(-1), indata.shape[1])
                self._input_capture_device = sd.InputStream(
                    samplerate=max(8000, int(self.patch.preview_sr)),
                    channels=max(1, int(sysdev.input_channels)),
                    dtype="float32",
                    blocksize=512,
                    callback=_sd_capture_cb,
                )
                self._input_capture_device.start()
                sysdev._reported_input_hw_channels = int(getattr(self._input_capture_device, "channels", max(1, sysdev.input_channels)))
                _in_name, _in_ch, _in_native_rate = _probe_default_sounddevice(True, sysdev.input_channels, self.patch.preview_sr)
                # Use the device's native (default) sample rate, not the stream's requested rate
                sysdev._reported_input_hw_rate = _in_native_rate if _in_native_rate > 0 else int(getattr(self._input_capture_device, "samplerate", max(8000, int(self.patch.preview_sr))))
                sysdev._reported_input_name = _in_name
                sysdev._input_buffers = [np.zeros(0, dtype=np.float32) for _ in range(max(0, sysdev.input_channels))]
                return
            except Exception as exc:
                print(f"System input capture error: {exc}")
                self._input_capture_device = None
                return

        try:
            from pygame._sdl2.audio import (
                AudioDevice, AUDIO_F32, AUDIO_ALLOW_ANY_CHANGE,
            )
        except Exception:
            return
        devname = sysdev.input_device_name or sysdev._reported_input_name
        if not devname:
            return

        def _capture_cb(dev, mv, _patch=self.patch):
            arr = np.frombuffer(mv, dtype=np.float32).copy()
            nch = max(1, int(getattr(dev, "numchannels", 1)))
            _write_capture(_patch, arr, nch)

        try:
            self._input_capture_device = AudioDevice(
                devicename=devname,
                iscapture=True,
                frequency=max(8000, int(self.patch.preview_sr)),
                audioformat=AUDIO_F32,
                numchannels=max(1, int(sysdev.input_channels)),
                chunksize=512,
                allowed_changes=AUDIO_ALLOW_ANY_CHANGE,
                callback=_capture_cb,
            )
            actual_ch = max(1, int(getattr(self._input_capture_device, "numchannels", max(1, sysdev.input_channels))))
            sysdev._reported_input_hw_channels = actual_ch
            sysdev._reported_input_hw_rate = int(getattr(self._input_capture_device, "frequency", max(8000, int(self.patch.preview_sr))))
            sysdev._reported_input_name = str(getattr(self._input_capture_device, "devicename", devname))
            sysdev._input_buffers = [np.zeros(0, dtype=np.float32) for _ in range(max(0, sysdev.input_channels))]
            self._input_capture_device.pause(0)
        except Exception as exc:
            print(f"System input capture error: {exc}")
            self._input_capture_device = None

    def _stop_output_playback(self) -> None:
        with self._output_lock:
            self._output_buffer.clear()
            self._output_playing = False

    def _ensure_output_device(self) -> bool:
        sysdev = self.patch.system_audio
        _refresh_system_audio_report(sysdev)
        devname = sysdev.output_device_name or sysdev._reported_output_name
        desired = (
            devname,
            max(1, int(sysdev.output_channels)),
            max(8000, int(self.patch.preview_sr)),
        )
        if self._output_device is not None and self._output_signature == desired:
            return True
        if self._output_device is not None:
            try:
                if hasattr(self._output_device, "abort"):
                    self._output_device.abort()
                self._output_device.close()
            except Exception:
                pass
            self._output_device = None
        self._output_backend = ""
        self._output_signature = None
        self._stop_output_playback()

        def _fill_output_bytes(n: int, _viewer=self) -> bytes:
            with _viewer._output_lock:
                if len(_viewer._output_buffer) >= n:
                    data = bytes(_viewer._output_buffer[:n])
                    del _viewer._output_buffer[:n]
                    _viewer._output_playing = True
                else:
                    copied = len(_viewer._output_buffer)
                    data = bytes(_viewer._output_buffer[:copied]) + (b"\x00" * (n - copied))
                    del _viewer._output_buffer[:copied]
                    _viewer._output_playing = False
                return data

        if not str(sysdev.output_device_name or "").strip():
            try:
                import sounddevice as sd
                def _sd_out_cb(outdata, frames, time_info, status, _viewer=self):
                    nbytes = outdata.size * outdata.dtype.itemsize
                    data = _fill_output_bytes(nbytes, _viewer)
                    arr = np.frombuffer(data, dtype=np.float32)
                    if arr.size != outdata.size:
                        arr = np.resize(arr, outdata.size)
                    outdata[:] = arr.reshape(outdata.shape)
                self._output_device = sd.OutputStream(
                    samplerate=desired[2],
                    channels=desired[1],
                    dtype="float32",
                    blocksize=1024,
                    callback=_sd_out_cb,
                )
                self._output_device.start()
                self._output_signature = desired
                self._output_backend = "sounddevice"
                _sd_name, _sd_ch, _sd_rate = _probe_default_sounddevice(False, desired[1], desired[2])
                if not _sd_name:
                    try:
                        devsel = getattr(self._output_device, "device", None)
                        if isinstance(devsel, (list, tuple)):
                            dev_idx = int(devsel[1])
                        else:
                            dev_idx = int(devsel)
                        _sd_name = str(sd.query_devices(dev_idx).get("name", ""))
                    except Exception:
                        _sd_name = "(default)"
                sysdev._reported_output_name = _sd_name
                sysdev._reported_output_hw_channels = int(getattr(self._output_device, "channels", desired[1]))
                # Use the device's native rate from query_devices, not the stream's
                # requested rate (sounddevice resamples internally so .samplerate
                # always echoes back the requested value).
                sysdev._reported_output_hw_rate = _sd_rate if _sd_rate > 0 else int(getattr(self._output_device, "samplerate", desired[2]))
                sysdev._preview_backend = "sounddevice default"
                sysdev._preview_backend_channels = sysdev._reported_output_hw_channels
                return True
            except Exception as exc:
                print(f"System output device error: {exc}")
                self._output_device = None
                return False

        try:
            from pygame._sdl2.audio import (
                AudioDevice, AUDIO_F32, AUDIO_ALLOW_ANY_CHANGE,
            )
        except Exception as exc:
            print(f"System output backend unavailable: {exc}")
            return False

        def _output_cb(_dev, mv, _viewer=self):
            mv[:] = _fill_output_bytes(len(mv), _viewer)

        try:
            self._output_device = AudioDevice(
                devicename=devname,
                iscapture=False,
                frequency=desired[2],
                audioformat=AUDIO_F32,
                numchannels=desired[1],
                chunksize=1024,
                allowed_changes=AUDIO_ALLOW_ANY_CHANGE,
                callback=_output_cb,
            )
            self._output_device.pause(0)
            self._output_signature = desired
            self._output_backend = "sdl2"
            sysdev._reported_output_name = str(getattr(self._output_device, "devicename", devname))
            sysdev._reported_output_hw_channels = int(getattr(self._output_device, "numchannels", desired[1]))
            sysdev._reported_output_hw_rate = int(getattr(self._output_device, "frequency", desired[2]))
            sysdev._preview_backend = "SDL2 AudioDevice"
            sysdev._preview_backend_channels = sysdev._reported_output_hw_channels
            return True
        except Exception as exc:
            print(f"System output device error: {exc}")
            self._output_device = None
            return False

    def _play_output_bus(self, out_bus: np.ndarray, src_sr: int) -> bool:
        if not self._ensure_output_device() or self._output_device is None:
            return False
        dst_sr = int(getattr(self._output_device, "frequency", src_sr))
        dst_ch = int(getattr(self._output_device, "numchannels", max(1, self.patch.system_audio.output_channels)))
        bus = _prepare_output_bus_for_device(out_bus, src_sr, dst_sr, dst_ch)
        payload = np.ascontiguousarray(bus, dtype=np.float32).tobytes()
        with self._output_lock:
            self._output_buffer = bytearray(payload)
            self._output_playing = len(payload) > 0
        return True

    def _on_remove_voice(self, key: str) -> None:
        router_key = _router_key_from_ui_key(key)
        if router_key:
            self.patch.routers = [r for r in self.patch.routers if r.key != router_key]
            if self.active_key == key:
                self.active_key = self.patch.voices[0].key if self.patch.voices else "__patch__"
            self._invalidate_graph_compile()
            self._needs_rebuild = True
            return
        self.patch.voices      = [v  for v  in self.patch.voices      if v.key  != key]
        self.patch.lfos        = [l  for l  in self.patch.lfos        if l.key  != key]
        self.patch.modules     = [m  for m  in self.patch.modules     if m.key  != key]
        self.patch.controls    = [cs for cs in self.patch.controls    if cs.key != key]
        self.patch.param_nodes = [pn for pn in self.patch.param_nodes if pn.key != key]
        # L4 fix: prune routing edges that reference the deleted node so they
        # don't show as ghost edges in the routing grid.
        remaining_keys = _patch_node_keys(self.patch)
        self.patch.routing.prune_keys(remaining_keys)
        for router in self.patch.routers:
            router.graph.prune_keys(remaining_keys)
        all_keys = remaining_keys  # reuse the already-computed list
        if self.active_key not in all_keys:
            self.active_key = all_keys[0] if all_keys else ""
        self._invalidate_graph_compile()
        self._needs_rebuild = True

    def _on_add_param(self) -> None:
        pn = ParamNode()
        pn.label = f"P{len(self.patch.param_nodes) + 1}"
        self.patch.param_nodes.append(pn)
        self.active_key = pn.key
        self.canvas.active_key = pn.key
        self.canvas.mode = EditorMode.PARAM_ROUTING
        self._invalidate_graph_compile()
        self._needs_rebuild = True

    def _on_add_module(self, module_type: str = "lfo") -> None:
        mod = AnalyticModule()
        mod.label = f"Mod{len(self.patch.modules) + 1}"
        mod.module_type = module_type if module_type in AnalyticModule._MODULE_TYPES else "lfo"
        colors = [(200, 140, 220), (160, 100, 200), (220, 160, 240)]
        mod.color = list(colors[len(self.patch.modules) % len(colors)])
        self.patch.modules.append(mod)
        self.active_key = mod.key
        self.canvas.active_key = mod.key
        self.canvas.mode = EditorMode.ROUTING
        self._invalidate_graph_compile()
        self._needs_rebuild = True

    def _on_add_control(self) -> None:
        cs = ControlSurface()
        cs.label = f"Ctrl{len(self.patch.controls) + 1}"
        # Add one default slider to make the surface immediately useful
        sl = ControlSlider()
        sl.label = "Slider 1"
        cs.sliders.append(sl)
        colors = [(100, 180, 140), (80, 160, 120), (120, 200, 160)]
        cs.color = list(colors[len(self.patch.controls) % len(colors)])
        self.patch.controls.append(cs)
        self.active_key = cs.key
        self._invalidate_graph_compile()
        self._needs_rebuild = True

    def _on_add_router(self, router_type: str = "voice_router") -> None:
        rt = str(router_type) if router_type in MIXER_LAYERS else "voice_router"
        idx = 1 + sum(1 for r in self.patch.routers if getattr(r, "router_type", "") == rt)
        label = {
            "voice_router": f"Voice Router {idx}",
            "instrument": f"Instrument Router {idx}",
            "master": f"Master Router {idx}",
        }.get(rt, f"Router {idx}")
        router = RouterInstance(label=label, router_type=rt)
        self.patch.routers.append(router)
        self.active_key = _router_ui_key(router.key)
        self.canvas.active_key = self.active_key
        self.canvas.routing_view.active_key = self.active_key
        self.canvas.mode = EditorMode.ROUTING
        self._invalidate_graph_compile()
        self._needs_rebuild = True

    def _on_deploy_chord(self) -> None:
        """Set voice frequencies to a triad derived from the first voice root."""
        p = self.patch
        if not _HAS_SEQ_ENG:
            return
        root_hz = p.seq_tonic_hz
        scale   = p.seq_scale if p.seq_scale in MODAL_SCALES else "pentatonic_minor"
        # Parse custom semitones if provided
        if p.seq_custom_semitones.strip():
            try:
                semis = [float(s) for s in p.seq_custom_semitones.split(",") if s.strip()]
                from sequence_engine import semitones_to_hz as _s2hz
                degrees: list[float] = []
                for octave in range(2):
                    for st in semis:
                        degrees.append(_s2hz(root_hz, st + 12 * octave))
            except Exception:
                degrees = scale_degrees_hz(root_hz, scale, octave_span=2)
        else:
            try:
                degrees = scale_degrees_hz(root_hz, scale, octave_span=2)
            except Exception:
                return
        # Find first chord degree from progression
        prog = CHORD_PROGRESSIONS.get(p.seq_chord_prog, {})
        first_chord = (prog.get("chords") or ["I"])[0]
        base_deg = _chord_to_degree(first_chord)
        chord_degs = [base_deg, base_deg + 2, base_deg + 4]
        chord_hz   = [degrees[d % len(degrees)] for d in chord_degs]
        _COLORS = [(100, 180, 255), (255, 130, 60), (120, 220, 120),
                   (220, 80, 160), (200, 200, 80)]
        # Ensure enough voices
        while len(p.voices) < len(chord_hz):
            v = AnalyticVoice()
            v.label = f"V{len(p.voices) + 1}"
            v.color = list(_COLORS[len(p.voices) % len(_COLORS)])
            p.voices.append(v)
        for i, hz in enumerate(chord_hz):
            p.voices[i].freq_hz = hz
        self._invalidate_graph_compile()
        self._needs_rebuild = True

    def _play_demo_sequence(self) -> None:
        """Render the demo through the torch graph solver and play a preview mix."""
        p = self.patch
        if not any(not v.muted for v in p.voices):
            return
        import time as _time
        sr = int(p.preview_sr)
        n = max(1, int(round(float(getattr(p, "duration", 1.0)) * sr)))
        demo_batch = max(1, int(getattr(p, "demo_batch_size", 0) or os.environ.get("ANALYTIC_DEMO_BATCH_SIZE", "4")))
        print(f"Demo graph: batch={demo_batch} samples={n} sr={sr}")

        _t0 = _time.monotonic()
        render_result = render_patch_graph(
            p,
            sample_rate=sr,
            n_samples=n,
            demo_batch_size=demo_batch,
            use_cache=False,
            profile=bool(ANALYTIC_GRAPH_SHADOW_PROFILE),
        )
        elapsed = _time.monotonic() - _t0
        timings = render_result.metadata.get("timings", {})
        print(
            f"Demo graph: complete in {elapsed:.2f}s "
            f"schedule_s={timings.get('schedule_s', 0.0):.3f} "
            f"nodes={timings.get('node_count', 0)} edges={timings.get('edge_count', 0)}"
        )
        out_bus = np.asarray(render_result.output_bus(), dtype=np.float32)
        if out_bus.ndim == 1:
            out_bus = out_bus[:, None]
        if out_bus.shape[1] > 2:
            if out_bus.shape[1] % 2 == 0:
                out_bus = out_bus.reshape(out_bus.shape[0], -1, 2).mean(axis=1)
            else:
                out_bus = out_bus.mean(axis=1, keepdims=True)
        peak = float(np.max(np.abs(out_bus))) if out_bus.size else 0.0
        if peak > 1e-9:
            out_bus = out_bus / peak
        try:
            if not self._play_output_bus(out_bus, sr):
                print("Demo playback error: no output device")
        except Exception as exc:
            print(f"Demo playback error: {exc}")

    # ---- Save / load -------------------------------------------------------

    def _save_patch(self) -> None:
        path = self.patch_path or "analytic_patch.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.patch.to_dict(), f, indent=2)
        print(f"Saved → {path}")
        _save_cavity_cache(path, self._cavity_cache)

    # ---- File export (Render button) ---------------------------------------

    def _on_render_to_files(self) -> None:
        """Synthesize the patch, write the main system bus, and optional mixer stems."""
        import datetime
        export_mixers = [m for m in self.patch.mixers if m.export_to_file]
        sysdev = self.patch.system_audio
        if not sysdev.export_to_file and not export_mixers:
            print("Render: system main render is off and no mixers are marked for export.")
            return
        print(f"Render: synthesising patch '{self.patch.name}' …")
        try:
            result = _synthesize_patch(self.patch, file_render=True,
                                       _return_mixer_sigs=True,
                                       _return_output_channels=True)
            _l, _r, main_bus, mixer_outs = result
        except Exception as exc:
            print(f"Render error during synthesis: {exc}")
            return
        stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        src_sr  = self.patch.preview_sr
        out_dir = os.path.dirname(self.patch_path or ".")
        safe_name = self.patch.name.replace(" ", "_").replace("/", "_")
        if sysdev.export_to_file:
            main_bus = np.asarray(main_bus, dtype=np.float32)
            if main_bus.ndim == 1:
                main_bus = main_bus[:, None]
            if sysdev.export_sample_rate != src_sr:
                cols = []
                for ci in range(main_bus.shape[1]):
                    cols.append(_resample_audio(main_bus[:, ci], src_sr, sysdev.export_sample_rate))
                main_bus = np.column_stack(cols).astype(np.float32, copy=False)
            main_name = os.path.join(out_dir or ".", f"{safe_name}_main_{stamp}.wav")
            try:
                import soundfile as _sf_export
                subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(
                    sysdev.export_bit_depth, "PCM_24")
                _sf_export.write(main_name, main_bus, samplerate=sysdev.export_sample_rate,
                                 subtype=subtype)
                print(f"  → {main_name}  ({main_bus.shape[1]} ch, {sysdev.export_sample_rate} Hz / {sysdev.export_bit_depth}-bit)")
            except Exception as exc:
                print(f"  Render write error for '{main_name}': {exc}")
        for m in export_mixers:
            if m.key not in mixer_outs:
                print(f"  Render: mixer '{m.label}' has no output signal — skipped.")
                continue
            ml, mr = mixer_outs[m.key]
            if m.export_sample_rate != src_sr:
                ml = _resample_audio(ml, src_sr, m.export_sample_rate)
                mr = _resample_audio(mr, src_sr, m.export_sample_rate)
            safe_label = m.label.replace(" ", "_").replace("/", "_")
            fname = os.path.join(out_dir or ".",
                                 f"{safe_name}_{safe_label}_{stamp}.wav")
            try:
                import soundfile as _sf_export
                data    = np.column_stack([ml.astype(np.float32), mr.astype(np.float32)])
                subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(
                    m.export_bit_depth, "PCM_24")
                _sf_export.write(fname, data, samplerate=m.export_sample_rate,
                                 subtype=subtype)
                print(f"  → {fname}  ({m.export_sample_rate} Hz / {m.export_bit_depth}-bit)")
            except Exception as exc:
                print(f"  Render write error for '{fname}': {exc}")
        print("Render complete.")

    def _on_render_sequence(self) -> None:
        """Synthesize the full demo sequence (rhythm + all modules) and save to WAV."""
        import datetime, copy as _copy, time as _time
        p = self.patch
        if not _HAS_SEQ_ENG:
            print("Render Sequence: sequence_engine not available")
            return
        if not p.system_audio.export_to_file:
            print("Render Sequence: system main render is off.")
            return
        template = next((v for v in p.voices if not v.muted), None)
        if template is None:
            print("Render Sequence: no unmuted voices")
            return

        # ── Build pitch list (mirrors _play_demo_sequence) ────────────────────
        scale  = p.seq_scale if p.seq_scale in MODAL_SCALES else "pentatonic_minor"
        beat_s = 60.0 / max(p.seq_bpm, 1.0)
        pattern = _SEQ_PATTERN_PRESETS[
            max(0, min(p.seq_pattern_idx, len(_SEQ_PATTERN_PRESETS) - 1))][1]
        try:
            if p.seq_custom_semitones.strip():
                custom_semi = [float(s) for s in p.seq_custom_semitones.split(",")
                               if s.strip()]
                degrees: list[float] = []
                for octave in range(p.seq_octave_span):
                    for st in custom_semi:
                        degrees.append(semitones_to_hz(template.freq_hz,
                                                       st + 12 * octave))
            else:
                degrees = scale_degrees_hz(template.freq_hz, scale,
                                           octave_span=p.seq_octave_span)
        except Exception as exc:
            print(f"Render Sequence: scale build error: {exc}")
            return

        try:
            _rdr_play_groups = _prepare_sequence_play_groups(p, beat_s, degrees, pattern)
        except Exception as exc:
            print(f"Render Sequence: schedule error: {exc}")
            return
        if not _rdr_play_groups:
            print("Render Sequence: no active play groups")
            return

        # ── Synthesize every note with full patch routing ─────────────────────
        sr = p.preview_sr
        _fb_cfg     = p.routing.feedback
        _gdecay_est = max(0.0, 1.0 - float(_fb_cfg.decay)) if _fb_cfg.enabled else 1.0
        _ringdown_n = estimate_ringdown_samples(p.routing.edges, sr, _gdecay_est, _fb_cfg)
        _rdr_dur    = max((s.total_duration for s, _ in _rdr_play_groups), default=0.0)
        total_n     = int((_rdr_dur + 0.5) * sr) + _ringdown_n
        out_ch  = max(2, int(getattr(p.system_audio, "output_channels", 2)))

        _total_events = sum(len(s.events) for s, _ in _rdr_play_groups)
        _has_sm = any(m.module_type == "state_machine" and not m.muted
                      and m.sm_plugin for m in p.modules)
        print(f"Render Sequence: {_total_events} events across "
              f"{len(_rdr_play_groups)} groups | SR {sr} "
              f"| SM modules {'ON' if _has_sm else 'off'}")

        # Per-module persistent aux state (cavity scenes survive across notes)
        _persistent_aux: dict[str, dict] = dict(self._cavity_cache)
        _t0 = _time.monotonic()

        _rdr_L, _rdr_R, mix_bus, _persistent_aux = _synthesize_sequence_full_batch(
            p, _rdr_play_groups, total_n,
            file_render=True,
            out_channels=out_ch,
            persistent_aux=_persistent_aux,
        )

        # Persist cavity caches back to the viewer for future runs / saving.
        self._cavity_cache.update(_persistent_aux)

        elapsed = _time.monotonic() - _t0
        print(f"Render Sequence: synthesis complete in {elapsed:.1f}s")
        peak = float(np.max(np.abs(mix_bus))) if mix_bus.size else 0.0
        if peak > 1e-9:
            mix_bus /= peak

        # ── Write WAV ─────────────────────────────────────────────────────────
        stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir  = os.path.dirname(self.patch_path or ".") or "."
        safe_name = p.name.replace(" ", "_").replace("/", "_")
        fname    = os.path.join(out_dir, f"{safe_name}_seq_{stamp}.wav")
        try:
            import soundfile as _sf_export
            data = mix_bus.astype(np.float32)
            if p.system_audio.export_sample_rate != sr:
                cols = []
                for ci in range(data.shape[1]):
                    cols.append(_resample_audio(data[:, ci], sr, p.system_audio.export_sample_rate))
                data = np.column_stack(cols).astype(np.float32, copy=False)
            subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(
                p.system_audio.export_bit_depth, "PCM_24")
            _sf_export.write(fname, data, samplerate=p.system_audio.export_sample_rate, subtype=subtype)
            print(f"Render Sequence → {fname}  ({data.shape[1]} ch)")
        except Exception as exc:
            print(f"Render Sequence write error: {exc}")

    # ---- Preview playback --------------------------------------------------

    def _play_preview(self) -> None:
        if self._output_playing:
            self._stop_output_playback()
            return
        active_voice = next((v for v in self.patch.voices if v.key == self.active_key), None)
        if active_voice is not None and self.canvas.mode == EditorMode.PIECEWISE_EDITOR:
            self.canvas._pull_piecewise_editor_state_into_voice(active_voice)
        # Seed module aux state from cavity cache before synthesis.
        for m in self.patch.modules:
            if m.key and m.key in self._cavity_cache and not m._sm_aux_state:
                m._sm_aux_state = dict(self._cavity_cache[m.key])
        try:
            render_result = render_patch_graph(
                self.patch,
                sample_rate=self.patch.preview_sr,
                n_samples=int(self.patch.preview_sr * self.patch.duration),
                demo_batch_size=1,
                use_cache=False,
                profile=bool(ANALYTIC_GRAPH_SHADOW_PROFILE),
            )
            timings = render_result.metadata.get("timings", {})
            print(
                "Preview graph: "
                f"samples={render_result.n_samples} "
                f"compile_s={timings.get('compile_s', 0.0):.3f} "
                f"schedule_s={timings.get('schedule_s', 0.0):.3f} "
                f"nodes={timings.get('node_count', 0)} "
                f"edges={timings.get('edge_count', 0)} "
                f"fifo_slots={timings.get('fifo_slots', 0)}"
            )
            src_sr = self.patch.preview_sr
            out_bus = np.asarray(render_result.output_bus(), dtype=np.float32)
            if out_bus.ndim == 1:
                out_bus = out_bus[:, None]
            if out_bus.shape[1] > 2:
                out_bus = out_bus[:, :2]
            if not self._play_output_bus(out_bus, src_sr):
                print("Preview error: no output device")
        except Exception as exc:
            print(f"Preview error: {exc}")

    # ---- GL init -----------------------------------------------------------

    def _init_gl(self) -> None:
        glViewport(0, 0, self.win_w, self.win_h)
        glClearColor(
            _C_BG[0] / 255, _C_BG[1] / 255, _C_BG[2] / 255, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # ---- Render panel tex --------------------------------------------------

    def _upload_panel(self, panel: Panel, old_tex: int) -> int:
        panel_surf = panel.render()
        if panel_surf is None:
            return old_tex
        # Pad to exact panel rect so GL quad doesn't stretch short surfaces
        pr = panel.panel_rect
        sw, sh = panel_surf.get_size()
        if sw != pr.w or sh != pr.h:
            padded = pygame.Surface((pr.w, pr.h))
            padded.fill(_PY_BG)
            padded.blit(panel_surf, (0, 0))
            panel_surf = padded
        return _surface_to_gl_tex(panel_surf, old_tex)

    def _panel_worker(self, stop_evt: "threading.Event",
                      left_slot: list, right_slot: list,
                      slot_lock: "threading.Lock") -> None:
        """Background thread: renders pygame panel surfaces and posts raw
        pixel bytes into left_slot / right_slot for the GL thread to upload.
        Runs as fast as panels produce dirty output; idles when nothing changed."""
        import threading as _th
        _IDLE_S = 1.0 / 30.0  # max 30 panel redraws/sec when idle
        while not stop_evt.is_set():
            with _PROF.span("panel_worker.render"):
                left_surf  = self.patch_panel.render()
                right_surf = self.partial_panel.render()
            now_left = now_right = None
            if left_surf is not None:
                pr = self.patch_panel.panel_rect
                sw, sh = left_surf.get_size()
                if sw != pr.w or sh != pr.h:
                    padded = pygame.Surface((pr.w, pr.h))
                    padded.fill(_PY_BG)
                    padded.blit(left_surf, (0, 0))
                    left_surf = padded
                now_left = (left_surf, left_surf.get_width(), left_surf.get_height())
            if right_surf is not None:
                pr = self.partial_panel.panel_rect
                sw, sh = right_surf.get_size()
                if sw != pr.w or sh != pr.h:
                    padded = pygame.Surface((pr.w, pr.h))
                    padded.fill(_PY_BG)
                    padded.blit(right_surf, (0, 0))
                    right_surf = padded
                now_right = (right_surf, right_surf.get_width(), right_surf.get_height())
            with slot_lock:
                if now_left  is not None:
                    left_slot[:]  = [now_left]
                if now_right is not None:
                    right_slot[:] = [now_right]
            stop_evt.wait(timeout=_IDLE_S)

    # ---- Render bottom status bar ------------------------------------------

    def _render_status(self) -> None:
        by = self.win_h - BOTTOM_H
        _gl_rect(0, by, self.win_w, BOTTOM_H,
                 self.win_w, self.win_h, (0.08, 0.08, 0.10, 1.0))
        _gl_hline(by, 0, self.win_w, self.win_w, self.win_h, (0.2, 0.2, 0.25, 1.0))
        if self._atlas.tex_id:
            p = self.patch
            msg = (f"  {p.name}   tonic={p.seq_tonic_hz:.1f} Hz  tuning={p.tuning.root_hz:.1f} Hz  "
                   f"dur={p.duration:.2f}s  sr={p.preview_sr}  "
                   f"voices={len(p.voices)}  lfos={len(p.lfos)}"
                   f"   [Space] preview   [Ctrl+S] save   [Tab] mode")
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self._atlas.draw_string(
                msg,
                4.0, float(by + BOTTOM_H // 2),
                self.win_w, self.win_h,
                anchor_x=0.0, anchor_y=0.5,
                color=(0.60, 0.60, 0.65, 1.0))
            glDisable(GL_TEXTURE_2D)

    # ---- Main loop ---------------------------------------------------------

    def run(self) -> None:
        pygame.init()
        _refresh_system_audio_report(self.patch.system_audio)
        self._ensure_output_device()
        self._refresh_input_capture()
        pygame.display.set_mode(
            (self.win_w, self.win_h), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Analytic Driver — Voice Editor")
        pygame.font.init()
        self._font = pygame.font.SysFont("consolas", 12)

        self._init_gl()
        self._atlas.build()

        # Wire Panel._top_offset so panels start below the title bar
        Panel._top_offset = TOPBAR_H

        # --- Panel worker thread -------------------------------------------
        # Renders pygame surfaces at ~30 Hz on a background thread and posts
        # raw pixel bytes into thread-safe slots.  The GL loop uploads them
        # whenever new data arrives, then reuses the last texture — so the
        # OpenGL overlay (wave / phase animation) runs fully uncapped.
        import threading as _th
        _stop_panels  = _th.Event()
        _slot_lock    = _th.Lock()
        _left_slot:  list = []   # each entry: (bytes, w, h)
        _right_slot: list = []
        _panel_thread = _th.Thread(
            target=self._panel_worker,
            args=(_stop_panels, _left_slot, _right_slot, _slot_lock),
            daemon=True,
        )
        _panel_thread.start()

        running = True
        _gl_clock = pygame.time.Clock()

        _PROF.start_reporter(interval_s=5.0, title="AnalyticDriver main loop", drain=False)
        while running:
            # ---- Events (must stay on main thread) ----
            with _PROF.span("events"):
             for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    continue

                if event.type == VIDEORESIZE:
                    self.win_w, self.win_h = event.size
                    pygame.display.set_mode(
                        (self.win_w, self.win_h), DOUBLEBUF | OPENGL | RESIZABLE)
                    self._init_gl()
                    self._atlas.build()
                    self._needs_rebuild = True
                    continue

                if event.type == KEYDOWN:
                    keys = pygame.key.get_pressed()
                    ctrl = keys[K_LCTRL] or keys[K_RCTRL]
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_SPACE:
                        self._play_preview()
                    elif ctrl and event.key == K_s:
                        self._save_patch()
                    elif event.key == K_TAB:
                        modes, _ = self.canvas._visible_modes(self.patch)
                        idx   = modes.index(self.canvas.mode) if self.canvas.mode in modes else -1
                        self.canvas.mode = modes[(idx + 1) % len(modes)]
                        self._needs_rebuild = True
                    continue

                # Panel dock handles left / right panels
                if self.dock.handle_event(event):
                    self._needs_rebuild = True
                    continue

                # Center canvas
                if self.canvas.handle_event(event, self.patch, self.win_w, self.win_h):
                    self._needs_rebuild = True
                    continue

            # ---- Sync panel state (cheap; no rendering here) ----
            with _PROF.span("panel.set_patch"):
                self.patch_panel.set_patch(self.patch, self.active_key)
                self.partial_panel.set_patch(self.patch, self.active_key)

            # ---- Granular seed animation ----
            with _PROF.span("granular.seed_anim"):
                _anim_voices = [
                    v for v in self.patch.voices
                    if v.emission_mode == "granular"
                    and v.granular is not None
                    and getattr(v.granular, "seed_animate", False)
                ]
                if _anim_voices:
                    _now = time.monotonic()
                    _period = min(getattr(v.granular, "seed_animate_period_s", 4.0)
                                  for v in _anim_voices)
                    if _now - self._seed_anim_last_time >= _period:
                        self._seed_anim_tick      += 1
                        self._seed_anim_last_time  = _now
                        self.canvas.granular_seed_offset = self._seed_anim_tick
                        self._needs_rebuild = True

            # ---- Rebuild waveform data if needed ----
            with _PROF.span("rebuild"):
                if self._needs_rebuild:
                    self.canvas.rebuild(self.patch, self.active_key)
                    self._needs_rebuild = False
                if self.canvas.rebuild_ready():
                    self.canvas._rebuild_thread = None
                    self.canvas._surf_dirty = True

            # ---- GL render (uncapped — no clock.tick) ----
            with _PROF.span("gl.clear"):
                glClear(GL_COLOR_BUFFER_BIT)

            # Center canvas — GL wave runs at full GL speed
            with _PROF.span("gl.canvas.render"):
                self.canvas.render(
                    self.win_w, self.win_h, self.patch, self._atlas, self._font)

            # Fold in latest panel textures from worker thread
            with _PROF.span("gl.panel_tex_upload"):
                with _slot_lock:
                    _pending_panels = []
                    if _left_slot:
                        _pending_panels.append(("left", _left_slot.pop()))
                    if _right_slot:
                        _pending_panels.append(("right", _right_slot.pop()))
                for _side, (surf, w, h) in _pending_panels:
                    raw = pygame.image.tobytes(surf, "RGBA", False)
                    if _side == "left":
                        if self._left_tex:
                            glDeleteTextures([self._left_tex])
                        tid = int(glGenTextures(1))
                        self._left_tex = tid
                    else:
                        if self._right_tex:
                            glDeleteTextures([self._right_tex])
                        tid = int(glGenTextures(1))
                        self._right_tex = tid
                    glBindTexture(GL_TEXTURE_2D, tid)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                                 GL_RGBA, GL_UNSIGNED_BYTE, raw)

            # Draw last known panel textures (stale is fine — worker updates async)
            with _PROF.span("gl.panel_tex_draw"):
                if self._left_tex:
                    lpr = self.patch_panel.panel_rect
                    _draw_tex_quad(
                        self._left_tex, lpr.x, lpr.y, lpr.w, lpr.h,
                        self.win_w, self.win_h)
                if self._right_tex:
                    rpr = self.partial_panel.panel_rect
                    _draw_tex_quad(
                        self._right_tex, rpr.x, rpr.y, rpr.w, rpr.h,
                        self.win_w, self.win_h)

            # Title bar
            with _PROF.span("gl.titlebar"):
                _gl_rect(0, 0, self.win_w, TOPBAR_H,
                         self.win_w, self.win_h, (0.07, 0.07, 0.10, 1.0))
                _gl_hline(TOPBAR_H - 1, 0, self.win_w,
                          self.win_w, self.win_h, (0.22, 0.22, 0.30, 1.0))
                if self._atlas.tex_id:
                    glEnable(GL_TEXTURE_2D)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    self._atlas.draw_string(
                        f"  Analytic Driver — {self.patch.name}",
                        4.0, float(TOPBAR_H // 2),
                        self.win_w, self.win_h,
                        anchor_x=0.0, anchor_y=0.5,
                        color=(0.75, 0.80, 0.90, 1.0))
                    glDisable(GL_TEXTURE_2D)

            # Bottom status
            with _PROF.span("gl.status"):
                self._render_status()

            with _PROF.span("gl.flip"):
                pygame.display.flip()
            _gl_clock.tick(60)   # cap at 60 FPS — prevents CPU/memory spiral from uncapped loop

        _stop_panels.set()
        _panel_thread.join(timeout=1.0)

        if self._input_capture_device is not None:
            try:
                self._input_capture_device.close()
            except Exception:
                pass
        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analytic synthesizer voice editor")
    parser.add_argument("patch", nargs="?", default=None,
                        help="Path to a patch JSON file to load")
    args = parser.parse_args()
    viewer = AnalyticDriverViewer(patch_path=args.patch)
    viewer.run()


if __name__ == "__main__":
    main()
