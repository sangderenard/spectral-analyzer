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
from OpenGL.GL import (
    GL_BLEND, GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT, GL_LINEAR,
    GL_LINE_LOOP, GL_LINE_STRIP, GL_LINES, GL_NEAREST, GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS, GL_RGBA, GL_SRC_ALPHA, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_TRIANGLES, GL_UNSIGNED_BYTE, GL_MODELVIEW, GL_PROJECTION, GL_SCISSOR_TEST,
    glBegin, glBindTexture, glBlendFunc, glClear, glClearColor,
    glColor4f, glDeleteTextures, glDisable, glEnable, glEnd,
    glGenTextures, glLineWidth, glTexCoord2f, glTexImage2D,
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
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ADSRParams:
    attack:  float = 0.005
    decay:   float = 0.04
    sustain: float = 0.75
    release: float = 0.08
    peak:    float = 1.0

    def to_knots(self, duration: float = 1.0) -> list[list[float]]:
        """Return ADSR as normalized knots list[[t, v], …]."""
        d = max(duration, 1e-9)
        a  = min(self.attack,  d)
        dk = min(self.decay,   d - a)
        r  = min(self.release, d - a - dk)
        ts = a + dk
        te = max(d - r, ts + 0.001 * d)
        return [
            [0.0,         0.0],
            [a  / d,      self.peak],
            [ts / d,      self.sustain],
            [te / d,      self.sustain],
            [1.0,         0.0],
        ]


@dataclass
class ChirpSpec:
    f_delta_start: float = 0.0    # Hz deviation at t=0 (added to base)
    f_delta_end:   float = 0.0    # Hz deviation at t=duration
    chirp_type:    str   = "none" # "none" | "linear" | "exponential" | "power"
    tau:           float = 0.5    # exponential decay constant (s)
    chirp_power:   float = 1.0    # exponent for "power" chirp type

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _CT = ["none", "linear", "exponential", "power"]
        return [
            KnobSpec("chirp_type",    "Chirp type", "choice", "none", 0, 3, 1, "",   _CT, False, "Chirp", ".0f", "",                     True),
            KnobSpec("f_delta_start", "\u0394f start",  "float",  0.0, -5000.0, 5000.0, 0, "Hz", [], False, "Chirp", ".1f", "LinearChirpPhasePath"),
            KnobSpec("f_delta_end",   "\u0394f end",    "float",  0.0, -5000.0, 5000.0, 0, "Hz", [], False, "Chirp", ".1f"),
            KnobSpec("tau",           "Tau",        "float",  0.5,  0.01,   10.0,   0, "s",  [], True,  "Chirp", ".3f", "ExponentialDecayPhasePath"),
            KnobSpec("chirp_power",   "Power",      "float",  1.0,  0.1,    8.0,    0, "",   [], False, "Chirp", ".2f", "PowerLawDecayPhasePath"),
        ]


@dataclass
class ModRouting:
    source_key: str   = ""
    depth_hz:   float = 2.0   # FM depth (Hz deviation)
    depth_amp:  float = 0.20  # AM depth (fraction of amplitude)


@dataclass
class PiecewiseVoiceEnvelope:
    curve: ParametricCurve = field(default_factory=lambda: _pc_default_envelope("voice_piecewise_amp"))
    chirp_curve: ParametricCurve = field(default_factory=lambda: _pc_default_chirp("voice_piecewise_chirp"))
    signal_curve: ParametricCurve = field(default_factory=lambda: _pc_default_blank("voice_piecewise_signal"))
    rule_tree: EnvelopeRuleTree = field(default_factory=EnvelopeRuleTree.default)
    source_path: str = ""
    detected_envelope_path: str = ""

    def to_dict(self) -> dict:
        return {
            "curve": self.curve.to_dict(),
            "chirp_curve": self.chirp_curve.to_dict(),
            "signal_curve": self.signal_curve.to_dict(),
            "rule_tree": self.rule_tree.to_dict(),
            "source_path": self.source_path,
            "detected_envelope_path": self.detected_envelope_path,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "PiecewiseVoiceEnvelope | None":
        if not d:
            return None
        curve = ParametricCurve.from_dict(dict(d.get("curve", {}))) if d.get("curve") else _pc_default_envelope("voice_piecewise_amp")
        chirp_curve = ParametricCurve.from_dict(dict(d.get("chirp_curve", {}))) if d.get("chirp_curve") else _pc_default_chirp("voice_piecewise_chirp")
        signal_curve = ParametricCurve.from_dict(dict(d.get("signal_curve", {}))) if d.get("signal_curve") else _pc_default_blank("voice_piecewise_signal")
        rule_tree = EnvelopeRuleTree.from_dict(dict(d.get("rule_tree", {}))) if d.get("rule_tree") else EnvelopeRuleTree.default()
        return cls(
            curve=curve,
            chirp_curve=chirp_curve,
            signal_curve=signal_curve,
            rule_tree=rule_tree,
            source_path=str(d.get("source_path", "") or ""),
            detected_envelope_path=str(d.get("detected_envelope_path", "") or ""),
        )


@dataclass
class AnalyticVoice:
    key:          str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:        str   = "Voice"
    freq_hz:      float = 440.0
    semitone_offset: float = 0.0   # semitones added on top of the context pitch
    note_tracking:   str   = "note"  # "note" | "root" | "free"
    amplitude:    float = 1.0
    phase_origin: float = 0.0      # radians
    chirp:  ChirpSpec  = field(default_factory=ChirpSpec)
    fm:     Optional[ModRouting] = None
    am:     Optional[ModRouting] = None
    env_type: str = "piecewise"
    adsr:     ADSRParams = field(default_factory=ADSRParams)
    env_knots: list[list[float]] = field(default_factory=lambda: [
        [0.0, 0.0], [0.01, 1.0], [0.1, 0.75], [0.85, 0.75], [1.0, 0.0]
    ])
    piecewise_env: Optional[PiecewiseVoiceEnvelope] = None
    loop_start:   float = 0.1
    loop_end:     float = 0.9
    loop_enabled: bool  = False
    muted:        bool  = False
    color: list[int] = field(default_factory=lambda: [100, 160, 255])
    pre_delay:           float = 0.0   # seconds of silence before voice onset
    manifold_type:       str   = "pure"  # "pure" | "harmonic" | "harmonic_warp"
    harmonic_count:      int   = 8       # number of harmonics to sum
    harmonic_brightness: float = 1.0    # amplitude rolloff exponent: amp_k = 1/k^brightness
    harmonic_warp_strength: float = 0.0  # stretches harmonic ratios; 0 = exact integer multiples
    voice_role:          str   = "signal"  # "signal" | "air" | "transient" | "body"
    seq_role:            str   = "melody"  # "melody" | "bass" | "root" | "stab"
    register:            str   = "all"     # "all" | "bass" | "mid" | "high" — rhythm page routing
    polyphony_count:     int   = 1         # how many simultaneous lines fit before another chair is needed
    polyphony_mode:      str   = "sympathetic"  # "sympathetic" | "unsympathetic"
    body_type:           str   = "direct"  # standard instrument body / resonator type tag
    emission_mode:       str   = "single"  # "single" | "granular"
    granular:            Any   = None      # GrainPopulationSpec when emission_mode=="granular"

    def active_knots(self) -> list[list[float]]:
        if self.env_type == "adsr":
            return self.adsr.to_knots(1.0)
        if self.piecewise_env is not None:
            return [[float(p.t), float(p.v)] for p in self.piecewise_env.curve.points]
        return self.env_knots

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label,
            "freq_hz": self.freq_hz,
            "semitone_offset": self.semitone_offset,
            "note_tracking":   self.note_tracking,
            "amplitude": self.amplitude,
            "phase_origin": self.phase_origin,
            "chirp": {
                "f_delta_start": self.chirp.f_delta_start,
                "f_delta_end":   self.chirp.f_delta_end,
                "chirp_type":    self.chirp.chirp_type,
                "tau":           self.chirp.tau,
                "chirp_power":   self.chirp.chirp_power,
            },
            "fm": ({"source_key": self.fm.source_key,
                    "depth_hz":   self.fm.depth_hz,
                    "depth_amp":  self.fm.depth_amp} if self.fm else None),
            "am": ({"source_key": self.am.source_key,
                    "depth_hz":   self.am.depth_hz,
                    "depth_amp":  self.am.depth_amp} if self.am else None),
            "env_type": "piecewise",
            "adsr": {"attack": self.adsr.attack, "decay": self.adsr.decay,
                     "sustain": self.adsr.sustain, "release": self.adsr.release,
                     "peak": self.adsr.peak},
            "env_knots": self.env_knots,
            "piecewise_env": self.piecewise_env.to_dict() if self.piecewise_env is not None else None,
            "loop_start": self.loop_start, "loop_end": self.loop_end,
            "loop_enabled": self.loop_enabled, "muted": self.muted,
            "color": self.color,
            "pre_delay": self.pre_delay,
            "manifold_type": self.manifold_type,
            "harmonic_count": self.harmonic_count,
            "harmonic_brightness": self.harmonic_brightness,
            "harmonic_warp_strength": self.harmonic_warp_strength,
            "voice_role": self.voice_role,
            "seq_role":   self.seq_role,
            "register":   self.register,
            "polyphony_count": self.polyphony_count,
            "polyphony_mode":  self.polyphony_mode,
            "body_type": self.body_type,
            "emission_mode": self.emission_mode,
            "granular": (self.granular.to_dict()
                         if self.granular is not None and hasattr(self.granular, "to_dict")
                         else self.granular),
        }

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _ROLES    = ["signal", "air", "transient", "body"]
        _MANIFOLD = ["pure", "harmonic", "harmonic_warp"]
        _POLY_MODES = ["sympathetic", "unsympathetic"]
        _BODY_TYPES = ["direct", "string_plate", "reed_box", "brass_bell", "drum_shell", "pipe_column", "voice_body"]
        return [
            # Oscillator
            KnobSpec("freq_hz",       "Freq",       "float",  440.0, 1.0,    20000.0, 0, "Hz",  [], True,  "Oscillator", ".1f", "ConstantPhasePath"),
            KnobSpec("semitone_offset","Offset",   "float",  0.0, -48.0,   48.0,    0, "st",  [], False, "Oscillator", ".2f"),
            KnobSpec("note_tracking",  "Tracking", "choice", "note", 0, 2, 1, "",
                     ["note", "root", "free"], False, "Oscillator", "", "", True),
            KnobSpec("amplitude",    "Amplitude",  "float",  1.0,   0.0,     4.0,     0, "",    [], False, "Oscillator", ".3f"),
            KnobSpec("phase_origin", "Phase",      "float",  0.0,  -math.pi, math.pi, 0, "rad", [], False, "Oscillator", ".3f", "ConstantPhasePath"),
            KnobSpec("pre_delay",    "Pre-delay",  "float",  0.0,   0.0,     2.0,     0, "s",   [], False, "Oscillator", ".3f"),
            KnobSpec("voice_role",   "Role",       "choice", "signal", 0, 3, 1, "",   _ROLES,    False, "Oscillator"),
            KnobSpec("seq_role",     "Arr. role",  "choice", "melody", 0, 3, 1, "",
                     ["melody", "bass", "root", "stab"], False, "Oscillator"),
            KnobSpec("register",     "Register",   "choice", "all",    0, 3, 1, "",
                     ["all", "bass", "mid", "high"],     False, "Oscillator"),
            KnobSpec("polyphony_count", "Polyphony", "int", 1, 1, 16, 1, "", [], False, "Oscillator", ".0f"),
            KnobSpec("polyphony_mode",  "Poly mode", "choice", "sympathetic", 0, 1, 1, "",
                     _POLY_MODES, False, "Oscillator"),
            KnobSpec("body_type",       "Body",       "choice", "direct", 0, max(0, len(_BODY_TYPES) - 1), 1, "",
                     _BODY_TYPES, False, "Oscillator"),
            # Chirp — delegate to ChirpSpec's own knob list with path prefix
            KnobSpec("chirp.chirp_type",    "Chirp type", "choice", "none", 0, 3, 1, "", ["none","linear","exponential","power"], False, "Chirp", ".0f", "", True),
            KnobSpec("chirp.f_delta_start", "\u0394f start",   "float",  0.0, -5000.0, 5000.0, 0, "Hz", [], False, "Chirp", ".1f", "LinearChirpPhasePath"),
            KnobSpec("chirp.f_delta_end",   "\u0394f end",     "float",  0.0, -5000.0, 5000.0, 0, "Hz", [], False, "Chirp", ".1f"),
            KnobSpec("chirp.tau",           "Tau",         "float",  0.5,  0.01,   10.0,   0, "s",  [], True,  "Chirp", ".3f", "ExponentialDecayPhasePath"),
            KnobSpec("chirp.chirp_power",   "Power",       "float",  1.0,  0.1,    8.0,    0, "",   [], False, "Chirp", ".2f", "PowerLawDecayPhasePath"),
            # FM
            KnobSpec("fm.depth_hz",  "FM depth",   "float", 0.0,  0.0, 2000.0, 0, "Hz", [], True,  "FM", ".1f", "SmoothSinusoidalDriftModel"),
            # AM
            KnobSpec("am.depth_amp", "AM depth",   "float", 0.0,  0.0, 1.0,    0, "",   [], False, "AM", ".3f"),
            # Loop
            KnobSpec("loop_start",   "Loop start", "float", 0.1,  0.0, 0.99,   0, "",   [], False, "Loop", ".3f"),
            KnobSpec("loop_end",     "Loop end",   "float", 0.9,  0.01, 1.0,   0, "",   [], False, "Loop", ".3f"),
            # Harmonic
            KnobSpec("manifold_type",           "Manifold",   "choice", "pure", 0, 2, 1, "", _MANIFOLD, False, "Harmonic", ".0f", "", True),
            KnobSpec("harmonic_count",          "Count",      "int",   8,   1,   32,  1, "",   [], False, "Harmonic", ".0f", "HarmonicMeasureManifold"),
            KnobSpec("harmonic_brightness",     "Brightness", "float", 1.0, 0.0, 4.0, 0, "",   [], False, "Harmonic", ".2f"),
            KnobSpec("harmonic_warp_strength",  "Warp str.",  "float", 0.0, 0.0, 2.0, 0, "",   [], False, "Harmonic", ".3f", "PhaseWarpedManifold"),
            # Emission mode
            KnobSpec("emission_mode", "Emission", "choice", "single", 0, 1, 1, "",
                     ["single", "granular"], False, "Emission", ".0f", "", True),
            # Granular population controls (visible only when emission_mode == "granular")
            KnobSpec("granular.center_frequency_hz",         "Center freq",   "float", 440.0,  20.0,   20000.0, 0, "Hz", [], True,  "Granular: Freq",     ".1f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_pitch_spread_semitones","Pitch spread",  "float", 3.0,    0.0,    24.0,    0, "st", [], False, "Granular: Freq",     ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_harmonic_lock",         "Harm. lock",    "float", 0.0,    0.0,    1.0,     0, "",   [], False, "Granular: Freq",     ".3f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_highband_bias",         "Highband bias", "float", 0.0,    0.0,    3.0,     0, "oct",[], False, "Granular: Freq",     ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_density_hz",            "Density",       "float", 20.0,   0.5,    500.0,   0, "/s", [], True,  "Granular: Birth",    ".1f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.birth_jitter",                "Jitter",        "float", 0.5,    0.0,    1.0,     0, "",   [], False, "Granular: Birth",    ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.burst_probability",           "Burst prob.",   "float", 0.0,    0.0,    1.0,     0, "",   [], False, "Granular: Birth",    ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.burst_size",                  "Burst size",    "int",   4,      2,      32,      1, "",   [], False, "Granular: Birth",    ".0f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.burst_spread_s",              "Burst spread",  "float", 0.015,  0.001,  0.2,     0, "s",  [], True,  "Granular: Birth",    ".3f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_duration_s",            "Duration",      "float", 0.05,   0.002,  2.0,     0, "s",  [], True,  "Granular: Duration", ".3f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_duration_jitter",       "Dur. jitter",   "float", 0.3,    0.0,    2.0,     0, "",   [], False, "Granular: Duration", ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_chirp_depth",           "Chirp depth",   "float", 0.0,    0.0,    2.0,     0, "",   [], False, "Granular: Chirp",    ".3f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_chirp_jitter",          "Chirp jitter",  "float", 0.5,    0.0,    1.0,     0, "",   [], False, "Granular: Chirp",    ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_phase_randomness",      "Phase rand.",   "float", 1.0,    0.0,    1.0,     0, "",   [], False, "Granular: Phase",    ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_manifold_mix",          "Manifold mix",  "float", 0.0,    0.0,    1.0,     0, "",   [], False, "Granular: Manifold", ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.gain",                        "Grain gain",    "float", 1.0,    0.0,    4.0,     0, "",   [], False, "Granular: Amp",      ".3f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_amp_jitter",            "Amp jitter",    "float", 0.2,    0.0,    2.0,     0, "",   [], False, "Granular: Amp",      ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.attack_frac",                 "Atk frac.",     "float", 0.15,   0.01,   0.5,     0, "",   [], False, "Granular: Envelope", ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.release_frac",                "Rel frac.",     "float", 0.35,   0.01,   0.8,     0, "",   [], False, "Granular: Envelope", ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_coherence",   "Coherence",    "float",  0.5, 0.0,  1.0,  0, "",   [], False, "Granular: Meta", ".2f", "", False, ("emission_mode", "granular")),
            KnobSpec("granular.coherence_mode",    "Coh. mode",    "choice", "uniform", 0, 0, 0, "",
                     ["uniform", "sine", "random_walk", "burst", "gradient", "perlin_walk"],
                     False, "Granular: Meta", "", "", True, ("emission_mode", "granular")),
            KnobSpec("granular.coherence_rate_hz", "Coh. rate",    "float",  0.5, 0.01, 10.0, 0, "Hz", [], False, "Granular: Meta", ".2f", "", False, ("emission_mode", "granular")),
            KnobSpec("granular.coherence_depth",   "Coh. depth",   "float",  0.3, 0.0,  1.0,  0, "",   [], False, "Granular: Meta", ".2f", "", False, ("emission_mode", "granular")),
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyticVoice":
        p = cls.__new__(cls)
        p.key          = d.get("key", uuid.uuid4().hex[:8])
        p.label        = d.get("label", "Voice")
        p.freq_hz      = float(d.get("freq_hz", 440.0))
        p.semitone_offset = float(d.get("semitone_offset", 0.0))
        p.note_tracking   = d.get("note_tracking", "note")
        if p.note_tracking not in ("note", "root", "free"):
            p.note_tracking = "note"
        p.amplitude    = float(d.get("amplitude", 1.0))
        p.phase_origin = float(d.get("phase_origin", 0.0))
        cd = d.get("chirp", {})
        p.chirp = ChirpSpec(
            f_delta_start=float(cd.get("f_delta_start", 0.0)),
            f_delta_end=float(cd.get("f_delta_end", 0.0)),
            chirp_type=cd.get("chirp_type", "none"),
            tau=float(cd.get("tau", 0.5)),
            chirp_power=float(cd.get("chirp_power", 1.0)),
        )
        fd = d.get("fm")
        p.fm = ModRouting(**fd) if fd else None
        ad = d.get("am")
        p.am = ModRouting(**ad) if ad else None
        p.env_type    = "piecewise"
        ad2 = d.get("adsr", {})
        p.adsr = ADSRParams(
            attack=float(ad2.get("attack",  0.005)),
            decay=float(ad2.get("decay",   0.04)),
            sustain=float(ad2.get("sustain", 0.75)),
            release=float(ad2.get("release", 0.08)),
            peak=float(ad2.get("peak", 1.0)),
        )
        p.env_knots    = d.get("env_knots", [[0,0],[0.01,1],[0.1,.75],[.85,.75],[1,0]])
        p.piecewise_env = PiecewiseVoiceEnvelope.from_dict(d.get("piecewise_env"))
        p.loop_start   = float(d.get("loop_start", 0.1))
        p.loop_end     = float(d.get("loop_end",   0.9))
        p.loop_enabled = bool(d.get("loop_enabled", False))
        p.muted        = bool(d.get("muted", False))
        p.color        = d.get("color", [100, 160, 255])
        p.pre_delay           = float(d.get("pre_delay", 0.0))
        p.manifold_type       = d.get("manifold_type", "pure")
        p.harmonic_count      = int(d.get("harmonic_count", 8))
        p.harmonic_brightness = float(d.get("harmonic_brightness", 1.0))
        p.harmonic_warp_strength = float(d.get("harmonic_warp_strength", 0.0))
        p.voice_role          = d.get("voice_role", "signal")
        p.seq_role            = d.get("seq_role", "melody")
        if p.seq_role not in ("melody", "bass", "root", "stab"):
            p.seq_role = "melody"
        p.register            = d.get("register", "all")
        if p.register not in ("all", "bass", "mid", "high"):
            p.register = "all"
        p.polyphony_count     = max(1, min(16, int(d.get("polyphony_count", 1))))
        p.polyphony_mode      = str(d.get("polyphony_mode", "sympathetic"))
        if p.polyphony_mode not in ("sympathetic", "unsympathetic"):
            p.polyphony_mode = "sympathetic"
        p.body_type           = str(d.get("body_type", "direct"))
        if p.body_type not in ("direct", "string_plate", "reed_box", "brass_bell", "drum_shell", "pipe_column", "voice_body"):
            p.body_type = "direct"
        p.emission_mode       = d.get("emission_mode", "single")
        # L2 fix: validate enum-like string fields — silently fall back to the
        # default rather than loading a typo that would synthesize silence.
        _VALID_EMISSION_MODES  = {"single", "granular"}
        _VALID_MANIFOLD_TYPES  = {"pure", "harmonic", "harmonic_warp"}
        if p.emission_mode not in _VALID_EMISSION_MODES:
            import warnings
            warnings.warn(
                f"AnalyticVoice.from_dict: unknown emission_mode {p.emission_mode!r}; "
                f"defaulting to 'single'.", stacklevel=2)
            p.emission_mode = "single"
        if p.manifold_type not in _VALID_MANIFOLD_TYPES:
            import warnings
            warnings.warn(
                f"AnalyticVoice.from_dict: unknown manifold_type {p.manifold_type!r}; "
                f"defaulting to 'pure'.", stacklevel=2)
            p.manifold_type = "pure"
        if p.piecewise_env is None:
            p.piecewise_env = PiecewiseVoiceEnvelope()
        raw_gran              = d.get("granular")
        if raw_gran and _HAS_GRANULAR:
            try:
                p.granular = _GrainPopulationSpec.from_dict(raw_gran)
            except Exception:
                p.granular = None
        else:
            p.granular = None
        return p


@dataclass
class PerformerPlacement:
    """One humanized performer instance derived from a chair assignment."""
    key: str
    label: str
    chair_key: str
    chair_index: int = 1
    performer_index: int = 1
    source_voice_keys: list = field(default_factory=list)
    source_layer_keys: list = field(default_factory=list)
    assigned_note_keys: list = field(default_factory=list)
    body_type: str = "direct"
    x: float = 0.0
    y: float = 0.0
    z: float = 1.1   # height above stage floor (meters); 1.1 = seated player reference
    face_x: float = 0.0    # aperture normal X — instrument faces conductor at origin
    face_y: float = -1.0   # aperture normal Y
    face_z: float = 0.0    # aperture normal Z
    radius: float = 0.0
    angle_deg: float = 0.0
    geometric_delay_ms: float = 0.0
    humanization_ms: float = 0.0
    phase_offset_rad: float = 0.0
    gain_db: float = 0.0
    pan: float = 0.0


@dataclass
class Chair:
    """A chair section such as 1st chair / 2nd chair within one Part."""
    key: str
    label: str
    part_key: str
    chair_index: int = 1
    specificity_rank: int = 0
    source_voice_keys: list = field(default_factory=list)
    source_layer_keys: list = field(default_factory=list)
    performer_count: int = 1
    performers: list = field(default_factory=list)  # list[PerformerPlacement]
    solver_hints: dict = field(default_factory=dict)


@dataclass
class NoteTarget:
    """The most holistic entity to which a NoteEvent is dispatched.

    The dispatch chain (most → least coordinated):
      "performer" — PerformerPlacement: knows position, delay, phase, gain, pan.
                    Multiple performers create a spatially-spread ensemble sound.
      "chair"     — Chair section: instrument-level grouping without per-seat
                    placement (uses shared voice signal, no geometric transforms).
      "voice"     — AnalyticVoice direct: no spatial context; raw synthesis only.

    ``voices`` is always the resolved list of AnalyticVoice objects that will
    actually synthesize audio regardless of the target_type chosen.
    """
    target_type: str              # "performer" | "chair" | "voice"
    voices: list                  # list[AnalyticVoice]
    performers: list              # list[PerformerPlacement]  — non-empty iff target_type=="performer"
    chairs: list                  # list[Chair]               — non-empty iff target_type in ("performer","chair")
    part: object                  # Part | None


@dataclass
class Part:
    """A resolved orchestral part, derived from the arrangement solver."""
    key: str                          # unique id, e.g. "bass-signal" or "high-melody"
    label: str                        # display name
    register: str                     # "bass" | "mid" | "high" | "all"
    seq_role: str                     # "melody" | "bass" | "root" | "stab" | ""
    voice_role: str                   # "signal" | "air" | "transient" | "body" | ""
    voice_keys: list = field(default_factory=list)   # AnalyticVoice.key values
    player_count: int = 1             # default one performer per part
    # Solver-derived performance envelope hint (optional, can be empty dict)
    solver_hints: dict = field(default_factory=dict)
    chairs: list = field(default_factory=list)       # list[Chair]


@dataclass
class PlacementResonatorConfig:
    """Patch-level placement-owned room/resonator defaults for deployed physics."""

    enabled: bool = False
    room_shape: str = "polygon"
    scene_path: str = ""
    room_radius: float = 3.4
    room_height: float = 3.6
    feedback_iterations: int = 1
    feedback_gain: float = 0.16
    passive_loss: float = 0.48
    band_split_mode: str = "fir"
    fir_taps: int = 65
    high_cone_deg: float = 70.0
    diffuse_strength: float = 0.42
    air_db_per_m: float = 0.01
    air_highband_db_per_m: float = 0.02
    temperature_c: float = 20.0
    humidity_rel: float = 0.5
    deployed_module_key: str = ""
    owner_module_type: str = "placement"
    # Mic array preset key from mic_arrays registry.
    # Empty string = legacy stereo pair (binaural_standard is the default when non-empty).
    receiver_array_key: str = "binaural_standard"
    # World-space receiver array center (meters, same coord space as room).
    receiver_pos_x: float = 0.0
    receiver_pos_y: float = 0.0
    receiver_pos_z: float = 1.5
    # Forward direction the array faces (normalized at use time).
    receiver_fwd_x: float = 0.0
    receiver_fwd_y: float = 1.0
    receiver_fwd_z: float = 0.0
    # Layout mode for performer packing.
    # "auto" = register-based semicircle (existing behaviour).
    # "stage" = dome-backed concert stage with proper orchestral section rows:
    #   1st/2nd Violins front-left arc, Violas center, Cellos right, Basses rear-right,
    #   Woodwinds center-rear, Brass right-rear, Percussion left-rear.
    layout_mode: str = "auto"

    def to_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "room_shape": str(self.room_shape),
            "scene_path": str(self.scene_path),
            "room_radius": float(self.room_radius),
            "room_height": float(self.room_height),
            "feedback_iterations": int(self.feedback_iterations),
            "feedback_gain": float(self.feedback_gain),
            "passive_loss": float(self.passive_loss),
            "band_split_mode": str(self.band_split_mode),
            "fir_taps": int(self.fir_taps),
            "high_cone_deg": float(self.high_cone_deg),
            "diffuse_strength": float(self.diffuse_strength),
            "air_db_per_m": float(self.air_db_per_m),
            "air_highband_db_per_m": float(self.air_highband_db_per_m),
            "temperature_c": float(self.temperature_c),
            "humidity_rel": float(self.humidity_rel),
            "deployed_module_key": str(self.deployed_module_key),
            "owner_module_type": str(self.owner_module_type),
            "receiver_array_key": str(self.receiver_array_key),
            "receiver_pos_x": float(self.receiver_pos_x),
            "receiver_pos_y": float(self.receiver_pos_y),
            "receiver_pos_z": float(self.receiver_pos_z),
            "receiver_fwd_x": float(self.receiver_fwd_x),
            "receiver_fwd_y": float(self.receiver_fwd_y),
            "receiver_fwd_z": float(self.receiver_fwd_z),
            "layout_mode": str(self.layout_mode),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlacementResonatorConfig":
        cfg = cls()
        cfg.enabled = bool(d.get("enabled", True))
        cfg.room_shape = str(d.get("room_shape", "polygon"))
        if cfg.room_shape not in ("polygon", "circular", "obj_mesh"):
            cfg.room_shape = "polygon"
        cfg.scene_path = str(d.get("scene_path", ""))
        cfg.room_radius = max(1.0, float(d.get("room_radius", 3.4)))
        cfg.room_height = max(1.5, float(d.get("room_height", 3.6)))
        cfg.feedback_iterations = max(0, int(d.get("feedback_iterations", 1)))
        cfg.feedback_gain = max(0.0, float(d.get("feedback_gain", 0.16)))
        cfg.passive_loss = max(0.0, min(0.98, float(d.get("passive_loss", 0.48))))
        cfg.band_split_mode = str(d.get("band_split_mode", "fir"))
        if cfg.band_split_mode not in ("fir", "fft"):
            cfg.band_split_mode = "fir"
        cfg.fir_taps = max(5, int(d.get("fir_taps", 65)) | 1)
        cfg.high_cone_deg = max(5.0, min(180.0, float(d.get("high_cone_deg", 70.0))))
        cfg.diffuse_strength = max(0.0, float(d.get("diffuse_strength", 0.42)))
        cfg.air_db_per_m = max(0.0, float(d.get("air_db_per_m", 0.01)))
        cfg.air_highband_db_per_m = max(0.0, float(d.get("air_highband_db_per_m", 0.02)))
        cfg.temperature_c = float(d.get("temperature_c", 20.0))
        cfg.humidity_rel = max(0.0, min(1.0, float(d.get("humidity_rel", 0.5))))
        cfg.deployed_module_key = str(d.get("deployed_module_key", ""))
        cfg.owner_module_type = str(d.get("owner_module_type", "placement") or "placement")
        cfg.receiver_array_key = str(d.get("receiver_array_key", "binaural_standard"))
        cfg.receiver_pos_x = float(d.get("receiver_pos_x", 0.0))
        cfg.receiver_pos_y = float(d.get("receiver_pos_y", 0.0))
        cfg.receiver_pos_z = float(d.get("receiver_pos_z", 1.5))
        cfg.receiver_fwd_x = float(d.get("receiver_fwd_x", 0.0))
        cfg.receiver_fwd_y = float(d.get("receiver_fwd_y", 1.0))
        cfg.receiver_fwd_z = float(d.get("receiver_fwd_z", 0.0))
        cfg.layout_mode = str(d.get("layout_mode", "auto"))
        if cfg.layout_mode not in ("auto", "stage"):
            cfg.layout_mode = "auto"
        return cfg


@dataclass
class LFODefinition:
    key:            str       = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:          str       = "LFO"
    # Slot 0 params — stored at top level for UI / knob-system compat.
    rate_hz:        float     = 1.0
    shape:          str       = "Sine"   # Sine | Triangle | Sawtooth | Square
    phase_offset:   float     = 0.0
    depth:          float     = 1.0
    # Packing capacity: number of parallel LFO slots this node carries.
    # Slot 0 uses the top-level scalar fields above; slots 1..capacity-1 are
    # stored in extra_channels as dicts {rate_hz, shape, phase_offset, depth}.
    capacity:       int       = 1
    extra_channels: list      = field(default_factory=list)
    color: list[int] = field(default_factory=lambda: [200, 160, 60])

    _SLOT_DEFAULTS: ClassVar[dict] = {
        "rate_hz": 1.0, "shape": "Sine", "phase_offset": 0.0, "depth": 1.0,
    }

    def _all_channels(self) -> list:
        """Return a list of per-slot param dicts of length ``capacity``.

        Slot 0 reflects the top-level scalar fields.  Slots 1..capacity-1
        come from ``extra_channels``, padded with defaults if needed.
        """
        slot0 = {"rate_hz": self.rate_hz, "shape": self.shape,
                 "phase_offset": self.phase_offset, "depth": self.depth}
        extras = list(self.extra_channels)
        while len(extras) < max(0, self.capacity - 1):
            extras.append(dict(self._SLOT_DEFAULTS))
        return [slot0] + extras[:max(0, self.capacity - 1)]

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "rate_hz": self.rate_hz, "shape": self.shape,
                "phase_offset": self.phase_offset, "depth": self.depth,
                "capacity": self.capacity,
                "extra_channels": list(self.extra_channels),
                "color": self.color}

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _SHAPES = ["Sine", "Triangle", "Sawtooth", "Square"]
        return [
            KnobSpec("capacity",     "Capacity",     "int",    1,   1,   64,       1, "",    [], False, "LFO", ".0f",
                     True),
            KnobSpec("rate_hz",      "Rate (slot 0)","float",  1.0, 0.01, 50.0,   0, "Hz",  [], True,  "LFO", ".3f"),
            KnobSpec("phase_offset", "Phase (slot 0)","float", 0.0,-math.pi, math.pi, 0, "rad", [], False, "LFO", ".3f"),
            KnobSpec("depth",        "Depth (slot 0)","float", 1.0, 0.0, 2.0,     0, "",    [], False, "LFO", ".3f"),
            KnobSpec("shape",        "Shape (slot 0)","choice","Sine", 0, 3, 1, "", _SHAPES, False, "LFO"),
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "LFODefinition":
        o = cls.__new__(cls)
        o.key            = d.get("key", uuid.uuid4().hex[:8])
        o.label          = d.get("label", "LFO")
        o.rate_hz        = float(d.get("rate_hz", 1.0))
        o.shape          = d.get("shape", "Sine")
        o.phase_offset   = float(d.get("phase_offset", 0.0))
        o.depth          = float(d.get("depth", 1.0))
        o.capacity       = int(d.get("capacity", 1))
        o.extra_channels = list(d.get("extra_channels", []))
        o.color          = d.get("color", [200, 160, 60])
        return o


# ---------------------------------------------------------------------------
# Temperament interval tables (module-level; shared by GlobalTuning).
# ---------------------------------------------------------------------------
_JUST_INTERVAL_RATIOS: list = [
    1.0, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 16/9, 15/8,
]
_PYTHAGOREAN_RATIOS: list = [
    1.0, 256/243, 9/8, 32/27, 81/64, 4/3, 729/512, 3/2, 128/81, 27/16, 16/9, 243/128,
]

# 22 śrutis of the Indian classical system (Bharata / Natya Shastra), in cents.
# Ordered chromatically; index = sruti number (0-based).
_22_SRUTI_CENTS: list = [
    0.000,      #  0 Sa            (tonic)
    90.225,     #  1 komal Re-1    (ek sruti)
    111.731,    #  2 komal Re-2    (do sruti)
    182.404,    #  3 Re-1          (tri sruti)
    203.910,    #  4 shuddha Re    (chatur sruti Rishab)
    294.135,    #  5 komal Ga-1    (sadharana Gandhar low)
    315.641,    #  6 komal Ga-2    (sadharana Gandhar / komal Ga)
    386.314,    #  7 antara Ga     (shuddha Ga in common usage)
    407.820,    #  8 shuddha Ga-2  (chatur sruti Gandhar)
    498.045,    #  9 shuddha Ma    (perfect fourth)
    519.551,    # 10 Ma-2
    590.224,    # 11 tivra Ma-1
    611.730,    # 12 tivra Ma-2    (tritone)
    701.955,    # 13 Pa            (perfect fifth)
    792.180,    # 14 komal Dha-1   (ek sruti)
    813.686,    # 15 komal Dha-2   (do sruti)
    884.359,    # 16 Dha-1         (tri sruti)
    905.865,    # 17 shuddha Dha   (chatur sruti Dhaivat)
    996.090,    # 18 komal Ni-1    (ek sruti)
    1017.596,   # 19 komal Ni-2    (kaisiki Nishad)
    1088.269,   # 20 Ni-1          (kakali Nishad low)
    1109.775,   # 21 shuddha Ni    (kakali Nishad / Ni-2)
]

# ¼-comma meantone cents for 12 pitch classes from the tonic.
_MEANTONE_QC_CENTS: list = [
    0.000, 76.049, 193.157, 269.205, 386.314, 503.422,
    579.471, 696.578, 772.627, 889.735, 965.784, 1082.892,
]

# Kirnberger III well-temperament cents from C.
_KIRNBERGER_III_CENTS: list = [
    0.000, 90.225, 193.157, 294.135, 386.314, 498.045,
    590.224, 696.578, 792.180, 889.735, 996.090, 1082.892,
]

# ---------------------------------------------------------------------------
# Global tuning presets.
# Each entry is a plain dict with all GlobalTuning field values plus a human-
# readable "label".  Keys are stable identifiers used as preset names.
# ---------------------------------------------------------------------------
GLOBAL_TUNING_PRESETS: dict = {
    # ── Standard Western ────────────────────────────────────────────────────
    "a440_12tet": {
        "label":         "A=440  12-TET (standard)",
        "root_hz":       440.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "a432_12tet": {
        "label":         "A=432  12-TET (Verdi / alternative concert pitch)",
        "root_hz":       432.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "a415_12tet": {
        "label":         "A=415  12-TET (Baroque low pitch)",
        "root_hz":       415.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    # ── Historical temperaments ──────────────────────────────────────────────
    "just_a440": {
        "label":         "A=440  Just Intonation (5-limit)",
        "root_hz":       440.0,
        "temperament":   "just",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "pythagorean_a440": {
        "label":         "A=440  Pythagorean (3-limit)",
        "root_hz":       440.0,
        "temperament":   "pythagorean",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "meantone_a440": {
        "label":         "A=440  ¼-comma Meantone",
        "root_hz":       440.0,
        "temperament":   "custom",
        "custom_cents":  list(_MEANTONE_QC_CENTS),
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "well_tempered_c": {
        "label":         "C=261.63  Kirnberger III Well-Temperament",
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_KIRNBERGER_III_CENTS),
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    # ── Indian classical (22-śruti) ──────────────────────────────────────────
    "raga_bhairav": {
        "label":         "Sa=261.63  Raga Bhairav (22 śrutis)",
        # Bhairav: Sa komal-Re Ga Ma Pa komal-Dha Ni
        # Evokes dawn; gravity, devotion, restraint.
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 1, 7, 9, 13, 14, 20],
        "scale_name":    "raga_bhairav",
    },
    "raga_yaman": {
        "label":         "Sa=261.63  Raga Yaman / Kalyan (22 śrutis)",
        # Yaman: Sa Re Ga tivra-Ma Pa Dha Ni  (Lydian-like)
        # Evening raga; floating, expansive, contemplative.
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 4, 7, 11, 13, 17, 20],
        "scale_name":    "raga_yaman",
    },
    "raga_bhairavi": {
        "label":         "Sa=261.63  Raga Bhairavi (22 śrutis)",
        # Bhairavi: Sa komal-Re komal-Ga Ma Pa komal-Dha komal-Ni
        # (All-flat Phrygian-like; morning farewell, melancholic beauty.)
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 1, 6, 9, 13, 14, 19],
        "scale_name":    "raga_bhairavi",
    },
}

# 22 śrutis of the Indian classical system (Bharata / Natya Shastra), in cents.
# Ordered chromatically; index = sruti number (0-based).
_22_SRUTI_CENTS: list = [
    0.000,      #  0 Sa            (tonic)
    90.225,     #  1 komal Re-1    (ek sruti)
    111.731,    #  2 komal Re-2    (do sruti)
    182.404,    #  3 Re-1          (tri sruti)
    203.910,    #  4 shuddha Re    (chatur sruti Rishab)
    294.135,    #  5 komal Ga-1    (sadharana Gandhar low)
    315.641,    #  6 komal Ga-2    (sadharana Gandhar / komal Ga)
    386.314,    #  7 antara Ga     (shuddha Ga in common usage)
    407.820,    #  8 shuddha Ga-2  (chatur sruti Gandhar)
    498.045,    #  9 shuddha Ma    (perfect fourth)
    519.551,    # 10 Ma-2
    590.224,    # 11 tivra Ma-1
    611.730,    # 12 tivra Ma-2    (tritone)
    701.955,    # 13 Pa            (perfect fifth)
    792.180,    # 14 komal Dha-1   (ek sruti)
    813.686,    # 15 komal Dha-2   (do sruti)
    884.359,    # 16 Dha-1         (tri sruti)
    905.865,    # 17 shuddha Dha   (chatur sruti Dhaivat)
    996.090,    # 18 komal Ni-1    (ek sruti)
    1017.596,   # 19 komal Ni-2    (kaisiki Nishad)
    1088.269,   # 20 Ni-1          (kakali Nishad low)
    1109.775,   # 21 shuddha Ni    (kakali Nishad / Ni-2)
]

# ¼-comma meantone cents for 12 pitch classes from the tonic.
_MEANTONE_QC_CENTS: list = [
    0.000, 76.049, 193.157, 269.205, 386.314, 503.422,
    579.471, 696.578, 772.627, 889.735, 965.784, 1082.892,
]

# Kirnberger III well-temperament cents from C.
_KIRNBERGER_III_CENTS: list = [
    0.000, 90.225, 193.157, 294.135, 386.314, 498.045,
    590.224, 696.578, 792.180, 889.735, 996.090, 1082.892,
]

# ---------------------------------------------------------------------------
# Global tuning presets.
# Each entry is a plain dict with all GlobalTuning field values plus a human-
# readable "label".  Keys are stable identifiers used as preset names.
# ---------------------------------------------------------------------------
GLOBAL_TUNING_PRESETS: dict = {
    # ── Standard Western ────────────────────────────────────────────────────
    "a440_12tet": {
        "label":         "A=440  12-TET (standard)",
        "root_hz":       440.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "a432_12tet": {
        "label":         "A=432  12-TET (Verdi / alternative concert pitch)",
        "root_hz":       432.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "a415_12tet": {
        "label":         "A=415  12-TET (Baroque low pitch)",
        "root_hz":       415.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    # ── Historical temperaments ──────────────────────────────────────────────
    "just_a440": {
        "label":         "A=440  Just Intonation (5-limit)",
        "root_hz":       440.0,
        "temperament":   "just",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "pythagorean_a440": {
        "label":         "A=440  Pythagorean (3-limit)",
        "root_hz":       440.0,
        "temperament":   "pythagorean",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "meantone_a440": {
        "label":         "A=440  ¼-comma Meantone",
        "root_hz":       440.0,
        "temperament":   "custom",
        "custom_cents":  list(_MEANTONE_QC_CENTS),
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "well_tempered_c": {
        "label":         "C=261.63  Kirnberger III Well-Temperament",
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_KIRNBERGER_III_CENTS),
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    # ── Indian classical (22-śruti) ──────────────────────────────────────────
    "raga_bhairav": {
        "label":         "Sa=261.63  Raga Bhairav (22 śrutis)",
        # Bhairav: Sa komal-Re Ga Ma Pa komal-Dha Ni
        # Evokes dawn; gravity, devotion, restraint.
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 1, 7, 9, 13, 14, 20],
        "scale_name":    "raga_bhairav",
    },
    "raga_yaman": {
        "label":         "Sa=261.63  Raga Yaman / Kalyan (22 śrutis)",
        # Yaman: Sa Re Ga tivra-Ma Pa Dha Ni  (Lydian-like)
        # Evening raga; floating, expansive, contemplative.
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 4, 7, 11, 13, 17, 20],
        "scale_name":    "raga_yaman",
    },
    "raga_bhairavi": {
        "label":         "Sa=261.63  Raga Bhairavi (22 śrutis)",
        # Bhairavi: Sa komal-Re komal-Ga Ma Pa komal-Dha komal-Ni
        # (All-flat Phrygian-like; morning farewell, melancholic beauty.)
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 1, 6, 9, 13, 14, 19],
        "scale_name":    "raga_bhairavi",
    },
}


# ---------------------------------------------------------------------------
# GlobalTuning — pitch reference frame for an AnalyticPatch.
#
# Defines what "semitone 0" means (root_hz), how semitone-to-Hz conversion is
# performed (temperament), and which pitch classes are active (scale_degrees /
# scale_name).  Voices and the PitchQuantizer both consult this object so that
# one global change (e.g. transposing root_hz) cascades everywhere.
# ---------------------------------------------------------------------------
@dataclass
class GlobalTuning:
    """Pitch reference frame for an AnalyticPatch.

    ``divisions_per_octave`` is a *derived* property — it equals
    ``len(custom_cents)`` when ``temperament == "custom"`` and 12 otherwise.
    This means changing ``custom_cents`` to a 22-entry śruti table automatically
    makes every offset/quantize operation work in 22-per-octave space.

    When using a preset via :meth:`from_preset`, ``scale_name`` and all related
    fields are set together so nothing falls out of sync.
    """
    root_hz:       float = 440.0
    temperament:   str   = "12tet"   # "12tet" | "just" | "pythagorean" | "custom"
    custom_cents:  list  = field(default_factory=lambda: [
        0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100
    ])
    scale_degrees: list  = field(default_factory=lambda: list(range(12)))
    scale_name:    str   = "chromatic"
    preset_name:   str   = "a440_12tet"  # last-applied preset (display hint)

    _TEMPERAMENT_CHOICES = ["12tet", "just", "pythagorean", "custom"]

    # ── Derived property ────────────────────────────────────────────────────
    @property
    def divisions_per_octave(self) -> int:
        """Number of equal-or-custom steps that span one octave.

        For the three named temperaments this is always 12.
        For ``"custom"`` it equals ``len(custom_cents)``, allowing non-12
        systems (e.g. 22 Indian śrutis) simply by supplying a different
        ``custom_cents`` list.
        """
        if self.temperament == "custom":
            return max(1, len(self.custom_cents))
        return 12

    # ── Conversion methods ──────────────────────────────────────────────────
    def semitone_to_hz(self, semitones: float) -> float:
        """Convert *semitones* (steps relative to root_hz) to Hz.

        One "semitone" is ``1 / divisions_per_octave`` of an octave in the
        active temperament.  For 22-śruti custom tuning, ``semitones=22``
        is exactly one octave up.
        """
        dpo      = self.divisions_per_octave
        octaves  = math.floor(semitones / dpo)
        frac     = semitones - dpo * octaves
        degree   = int(round(frac)) % dpo
        leftover = frac - degree
        t = self.temperament
        if t == "just":
            ratio = _JUST_INTERVAL_RATIOS[degree % 12]
        elif t == "pythagorean":
            ratio = _PYTHAGOREAN_RATIOS[degree % 12]
        elif t == "custom" and self.custom_cents:
            idx   = degree % len(self.custom_cents)
            ratio = 2.0 ** (self.custom_cents[idx] / 1200.0)
        else:  # 12tet
            ratio = 2.0 ** (degree / 12.0)
        if abs(leftover) > 1e-9:
            ratio *= 2.0 ** (leftover / dpo)
        return self.root_hz * (2.0 ** octaves) * ratio

    def hz_to_semitones(self, hz: float) -> float:
        """Convert *hz* to steps (semitones) relative to root_hz.

        The result is in the same unit system as :meth:`semitone_to_hz`:
        one step = one division of the octave in the active tuning.
        """
        if hz <= 0.0:
            return 0.0
        return self.divisions_per_octave * math.log2(hz / self.root_hz)

    def quantize(self, semitones: float) -> float:
        """Snap *semitones* to nearest active scale degree (octave-preserving)."""
        dpo     = self.divisions_per_octave
        degrees = sorted(self.scale_degrees)
        if not degrees:
            return semitones
        octave  = math.floor(semitones / dpo)
        pc      = semitones - dpo * octave
        nearest = min(degrees, key=lambda d: abs(d - pc))
        if abs(degrees[0] + dpo - pc) < abs(nearest - pc):
            nearest = degrees[0]
            octave += 1
        return dpo * octave + float(nearest)

    # ── Preset system ───────────────────────────────────────────────────────
    @classmethod
    def from_preset(cls, name: str) -> "GlobalTuning":
        """Return a :class:`GlobalTuning` configured from a named preset.

        Available preset names are the keys of :data:`GLOBAL_TUNING_PRESETS`.
        Raises ``ValueError`` for unknown names.
        """
        data = GLOBAL_TUNING_PRESETS.get(name)
        if data is None:
            raise ValueError(
                f"Unknown GlobalTuning preset {name!r}.  "
                f"Available: {sorted(GLOBAL_TUNING_PRESETS)}"
            )
        o = cls.__new__(cls)
        o.root_hz       = float(data.get("root_hz",  440.0))
        o.temperament   = data.get("temperament",    "12tet")
        o.custom_cents  = list(data.get("custom_cents",
                                        [0,100,200,300,400,500,600,700,800,900,1000,1100]))
        o.scale_degrees = list(data.get("scale_degrees", list(range(12))))
        o.scale_name    = data.get("scale_name",     "chromatic")
        o.preset_name   = name
        return o

    # ── Serialisation ───────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "root_hz":       self.root_hz,
            "temperament":   self.temperament,
            "custom_cents":  list(self.custom_cents),
            "scale_degrees": list(self.scale_degrees),
            "scale_name":    self.scale_name,
            "preset_name":   self.preset_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalTuning":
        o = cls.__new__(cls)
        o.root_hz       = float(d.get("root_hz", 440.0))
        o.temperament   = d.get("temperament", "12tet")
        if o.temperament not in cls._TEMPERAMENT_CHOICES:
            o.temperament = "12tet"
        o.custom_cents  = list(d.get("custom_cents",
                                     [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]))
        o.scale_degrees = list(d.get("scale_degrees", list(range(12))))
        o.scale_name    = d.get("scale_name", "chromatic")
        o.preset_name   = d.get("preset_name", "")
        return o

    @classmethod
    def knobs(cls) -> list:
        _preset_choices = list(GLOBAL_TUNING_PRESETS.keys())
        return [
            KnobSpec("preset_name",  "Preset",      "choice", "a440_12tet", 0,
                     max(0, len(_preset_choices) - 1), 1, "",
                     _preset_choices, False, "Tuning", "", "", True),
            KnobSpec("root_hz",      "Root Hz",     "float",  440.0, 20.0, 8000.0, 0, "Hz", [],
                     True,  "Tuning", ".2f"),
            KnobSpec("temperament",  "Temperament", "choice", "12tet", 0, 3, 1, "",
                     cls._TEMPERAMENT_CHOICES, False, "Tuning", "", "", True),
            KnobSpec("scale_name",   "Scale",       "str",    "chromatic", 0, 0, 0, "",
                     [], False, "Tuning"),
        ]


# ---------------------------------------------------------------------------
# QuantizerHandle — callable pitch quantizer with pluggable interpolation.
#
# Returned by make_quantizer_handle().  The caller injects:
#   value          : float — raw input to quantize
#   original_value : float — pre-quantization source value (same domain);
#                            used by interpolators to track continuity across
#                            unquantized motion
#   domain         : str   — "semitone" (relative to tuning root) | "hz"
#   dt             : float — elapsed seconds since last call (time-based modes)
#
# All configuration is captured at construction time; _state is mutable.
# Call handle.reset() before replaying a note or resetting the patch.
# ---------------------------------------------------------------------------
class QuantizerHandle:
    """
    Callable pitch quantizer encapsulating scale, interpolation mode, and
    mutable integrator state.

    Call signature::

        result = handle(value, original_value, domain="semitone", dt=0.0)

    Parameters
    ----------
    value          : input to quantize (semitones or Hz per *domain*)
    original_value : pre-quantization source in the same *domain*
    domain         : ``"semitone"`` | ``"hz"``
    dt             : elapsed seconds since previous call (required for
                     ``portamento``, ``slew``, ``slew2``, ``spline``,
                     ``legato``; use ``0.0`` for stateless / one-shot use)

    Returns the quantized output in the **same domain** as *value*.
    """

    _INTERP_MODES = ["discrete", "portamento", "slew", "slew2", "spline", "legato"]

    def __init__(
        self,
        tuning:          "GlobalTuning",
        scale_degrees:   list,
        mode:            str   = "discrete",
        portamento_time: float = 0.05,
        slew_rate:       float = 100.0,
        slew2_accel:     float = 200.0,
        spline_tension:  float = 0.5,
    ) -> None:
        self._tuning          = tuning
        self._scale_degrees   = sorted(set(scale_degrees)) if scale_degrees else list(range(12))
        self._mode            = mode if mode in self._INTERP_MODES else "discrete"
        self._portamento_time = float(portamento_time)
        self._slew_rate       = float(slew_rate)
        self._slew2_accel     = float(slew2_accel)
        self._spline_tension  = float(spline_tension)
        self._state: dict = {
            "position":    None,   # current smoothed semitone output; None = uninitialised
            "velocity":    0.0,    # semitones/s  (portamento / slew / slew2)
            "target":      None,   # last quantized target semitone
            "target_prev": None,   # for change detection (spline / legato)
            "spline_pos0": 0.0,    # start position for current spline segment
            "spline_t":    0.0,    # normalised time 0→1 along current spline segment
        }

    @property
    def state(self) -> dict:
        return self._state

    # ── internal helpers ──────────────────────────────────────────────────
    def _quantize_st(self, semitones: float) -> float:
        dpo     = self._tuning.divisions_per_octave
        degrees = self._scale_degrees
        octave  = math.floor(semitones / dpo)
        pc      = semitones - dpo * octave
        nearest = min(degrees, key=lambda d: abs(d - pc))
        if abs(degrees[0] + dpo - pc) < abs(nearest - pc):
            nearest = degrees[0]
            octave += 1
        return dpo * octave + float(nearest)

    def _quantize_st_array(self, semitones: np.ndarray) -> np.ndarray:
        arr = np.asarray(semitones, dtype=np.float64)
        dpo = float(self._tuning.divisions_per_octave)
        degrees = np.asarray(self._scale_degrees, dtype=np.float64)
        if arr.size == 0:
            return arr.copy()
        octave = np.floor(arr / dpo)
        pc = arr - dpo * octave
        deltas = np.abs(pc[:, None] - degrees[None, :])
        nearest_idx = np.argmin(deltas, axis=1)
        nearest = degrees[nearest_idx]
        wrap_delta = np.abs((degrees[0] + dpo) - pc)
        wrap_mask = wrap_delta < np.abs(nearest - pc)
        nearest = nearest.copy()
        nearest[wrap_mask] = degrees[0]
        octave = octave.copy()
        octave[wrap_mask] += 1.0
        return dpo * octave + nearest

    def _to_st(self, value: float, domain: str) -> float:
        if domain == "hz":
            if value <= 0.0:
                return 0.0
            return 12.0 * math.log2(value / self._tuning.root_hz)
        return float(value)

    def _to_st_array(self, values: np.ndarray, domain: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if domain != "hz":
            return arr.copy()
        out = np.zeros_like(arr)
        pos = arr > 0.0
        if np.any(pos):
            out[pos] = 12.0 * np.log2(arr[pos] / self._tuning.root_hz)
        return out

    def _from_st(self, st: float, domain: str) -> float:
        if domain == "hz":
            return self._tuning.semitone_to_hz(st)
        return st

    def _from_st_array(self, st: np.ndarray, domain: str) -> np.ndarray:
        arr = np.asarray(st, dtype=np.float64)
        if domain != "hz":
            return arr.copy()
        return self._tuning.root_hz * np.power(2.0, arr / 12.0)

    # ── main entry point ──────────────────────────────────────────────────
    def __call__(
        self,
        value:          float,
        original_value: float,
        domain:         str   = "semitone",
        dt:             float = 0.0,
    ) -> float:
        """
        Quantize *value* against the active scale and apply interpolation.

        Parameters
        ----------
        value          : raw input in *domain*
        original_value : unquantized source in *domain* (used for interpolation
                         continuity; may equal *value* when the caller has no
                         separate pre-quantization signal)
        domain         : ``"semitone"`` | ``"hz"``
        dt             : seconds since last call; ``0.0`` → stateless snap
        """
        value_st    = self._to_st(value, domain)
        original_st = self._to_st(original_value, domain)
        target_st   = self._quantize_st(value_st)
        self._state["target"] = target_st

        pos = self._state["position"]
        if pos is None:
            pos = self._quantize_st(original_st)

        mode = self._mode

        if mode == "discrete" or dt <= 0.0:
            pos = target_st

        elif mode == "portamento":
            if self._portamento_time > 0.0:
                alpha = 1.0 - math.exp(-dt / self._portamento_time)
                pos   = pos + alpha * (target_st - pos)
            else:
                pos = target_st

        elif mode == "slew":
            max_delta = self._slew_rate * dt
            dist      = target_st - pos
            pos       = pos + math.copysign(min(abs(dist), max_delta), dist)

        elif mode == "slew2":
            vel  = self._state["velocity"]
            dist = target_st - pos
            sign = math.copysign(1.0, dist) if dist != 0.0 else 0.0
            accel = sign * self._slew2_accel - vel * 2.0
            vel  += accel * dt
            if abs(vel) * dt > abs(dist) and dist * vel > 0:
                vel = dist / max(dt, 1e-9)
            pos += vel * dt
            self._state["velocity"] = vel

        elif mode == "spline":
            if target_st != self._state["target_prev"]:
                self._state["spline_pos0"] = pos
                self._state["spline_t"]    = 0.0
                self._state["target_prev"] = target_st
            t = self._state["spline_t"]
            if self._portamento_time > 0.0 and dt > 0.0:
                t = min(t + dt / self._portamento_time, 1.0)
            else:
                t = 1.0
            tension = self._spline_tension
            p0 = self._state["spline_pos0"]
            m0 = (target_st - p0) * tension
            m1 = m0
            h00 =  2*t**3 - 3*t**2 + 1
            h10 =    t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 =    t**3 -   t**2
            pos = h00 * p0 + h10 * m0 + h01 * target_st + h11 * m1
            self._state["spline_t"] = t

        elif mode == "legato":
            if target_st != self._state["target_prev"]:
                self._state["target_prev"] = target_st
                if self._portamento_time > 0.0 and dt > 0.0:
                    alpha = 1.0 - math.exp(-dt / self._portamento_time)
                    pos   = pos + alpha * (target_st - pos)
                else:
                    pos = target_st
            # else: hold current position

        self._state["position"] = pos
        return self._from_st(pos, domain)

    def process_series(
        self,
        values: np.ndarray,
        original_values: "np.ndarray | None" = None,
        domain: str = "semitone",
        dt: float = 0.0,
    ) -> np.ndarray:
        """Quantize a full series, using vectorized math when possible."""
        vals = np.asarray(values, dtype=np.float64)
        orig = vals if original_values is None else np.asarray(original_values, dtype=np.float64)
        if vals.shape != orig.shape:
            raise ValueError("values and original_values must have the same shape")
        if vals.ndim != 1:
            raise ValueError("process_series expects a 1-D array")
        if vals.size == 0:
            return vals.copy()
        if self._mode == "discrete" or dt <= 0.0:
            value_st = self._to_st_array(vals, domain)
            target_st = self._quantize_st_array(value_st)
            self._state["target"] = float(target_st[-1])
            self._state["position"] = float(target_st[-1])
            return self._from_st_array(target_st, domain)
        out = np.empty_like(vals, dtype=np.float64)
        for i in range(vals.size):
            out[i] = self(vals[i], orig[i], domain=domain, dt=dt)
        return out

    def reset(self) -> None:
        """Clear all integrator state (call before a new note or patch reset)."""
        self._state["position"]    = None
        self._state["velocity"]    = 0.0
        self._state["target"]      = None
        self._state["target_prev"] = None
        self._state["spline_pos0"] = 0.0
        self._state["spline_t"]    = 0.0


def make_quantizer_handle(
    module:  "AnalyticModule",
    tuning:  "GlobalTuning",
) -> QuantizerHandle:
    """
    Build a :class:`QuantizerHandle` from a ``pitch_quantizer``
    :class:`AnalyticModule` and a :class:`GlobalTuning`.

    ``module.quantizer_scale_degrees`` provides the active pitch classes;
    when empty the tuning's own ``scale_degrees`` are used.
    """
    degrees = (sorted(set(module.quantizer_scale_degrees))
               if module.quantizer_scale_degrees
               else list(tuning.scale_degrees))
    return QuantizerHandle(
        tuning          = tuning,
        scale_degrees   = degrees,
        mode            = module.interpolation_mode,
        portamento_time = module.portamento_time,
        slew_rate       = module.slew_rate,
        slew2_accel     = module.slew2_accel,
        spline_tension  = module.spline_tension,
    )


# ---------------------------------------------------------------------------
# AnalyticMixer — a named mixer node in the routing graph.
#
# projection_active = True  → its output is summed into the PCM bus (speaker /
#                             file).  The patch-level projection_mode / rotation
#                             settings are applied before writing to the bus.
# projection_active = False → analytic meta-mixer only.  Its output is a
#                             complex analytic signal that can be routed into
#                             other mixer nodes but does NOT contribute to the
#                             PCM bus.  Useful for sub-mixes, sidechains, etc.
# ---------------------------------------------------------------------------
@dataclass
class AnalyticMixer:
    key:               str  = field(default_factory=lambda: "__mix__")
    label:             str  = "Mix"
    projection_active: bool = True     # True → output track; False → meta-mixer
    color: list = field(default_factory=lambda: [200, 200, 100, 255])
    # File export settings (used by the Render button in the sequencer panel)
    export_to_file:     bool = False
    export_sample_rate: int  = 48000
    export_bit_depth:   int  = 24
    # Signal layer this mixer operates at.  The only valid mixer layer is
    # "master" — the single top-level mix bus that receives room SM mic
    # streams and any instrument signals routed directly here.
    mixer_layer: str = "master"

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "projection_active": self.projection_active,
                "color": self.color,
                "export_to_file":     self.export_to_file,
                "export_sample_rate": self.export_sample_rate,
                "export_bit_depth":   self.export_bit_depth,
                "mixer_layer":        self.mixer_layer}

    @classmethod
    def knobs(cls) -> list["KnobSpec"]:
        return [
            KnobSpec("label",             "Label",      "str",  "Mix", 0, 0, 0, "", [], False, "Mixer"),
            KnobSpec("projection_active", "Output/PCM", "bool", True,  0, 1, 0, "", [], False, "Mixer"),
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyticMixer":
        o = cls.__new__(cls)
        o.key               = d.get("key", "__mix__")
        o.label             = d.get("label", "Mix")
        o.projection_active = bool(d.get("projection_active", True))
        o.color             = d.get("color", [200, 200, 100, 255])
        o.export_to_file     = bool(d.get("export_to_file",     False))
        o.export_sample_rate = int(d.get("export_sample_rate",  48000))
        o.export_bit_depth   = int(d.get("export_bit_depth",    24))
        o.mixer_layer        = str(d.get("mixer_layer", "master"))
        return o


@dataclass
class PortTensorSpec:
    """Negotiation metadata for one published port.

    tensor_rank
        Conceptual rank of the payload.  0 = scalar control, 1 = vector lane,
        2+ = structured tensor batch.  This is descriptive for now and lets the
        graph/UI evolve toward massive parallel cables without changing the
        publishing API again.
    lane_count
        Number of parallel lanes exposed by this port when known.  0 means
        "dynamic / negotiated at compile time".
    parallel_group
        Optional symbolic grouping key used to validate wide-bus compatibility
        across related ports (for example room↔performer state exchange).
    """
    tensor_rank: int = 0
    lane_count: int = 1
    batch_axes: int = 1
    parallel_group: str = ""
    group_validity: str = "strict"   # strict | broadcast | reduce | remap
    semantic_role: str = ""
    channel_dims: list = field(default_factory=list)
    batchable: bool = True
    dtype: str = "complex128"
    analytic_only: bool = True

    def to_contract(self) -> "TensorPortContract":
        return TensorPortContract(
            dtype=str(self.dtype or "complex128"),
            analytic_only=bool(self.analytic_only),
            tensor_rank=max(0, int(self.tensor_rank)),
            lane_count=max(0, int(self.lane_count)),
            batch_axes=max(0, int(self.batch_axes)),
            parallel_group=str(self.parallel_group or ""),
            group_validity=str(self.group_validity or "strict"),
            semantic_role=str(self.semantic_role or ""),
            channel_dims=[int(x) for x in self.channel_dims],
        )


@dataclass
class PublishedPort:
    key: str = ""
    label: str = ""
    direction: str = "out"      # "in" | "out"
    domain: str = "control"     # "signal" | "control" | "param_target"
    owner_key: str = ""
    group: str = ""
    param_path: str = ""
    color: tuple[int, int, int] = (150, 150, 170)
    tensor: PortTensorSpec = field(default_factory=PortTensorSpec)
    semantic_role: str = ""
    projection_policy: str = ""   # only meaningful for scalar / non-complex destinations
    negotiates_group_validity: bool = False


@dataclass
class RackPortView:
    port: PublishedPort
    local_x: int = 0
    local_y: int = 0
    radius: int = 3


@dataclass
class RackDeviceView:
    device_key: str = ""
    label: str = ""
    device_kind: str = ""
    color: tuple[int, int, int] = (90, 110, 140)
    rack_u: int = 1
    rack_w: int = 1
    grid_x: int = 0
    grid_y: int = 0
    ports: list = field(default_factory=list)   # list[RackPortView]


@dataclass
class RackConnectionView:
    src_port_key: str = ""
    dst_port_key: str = ""
    edge_kind: str = "control"
    remove_kind: str = "control"   # control | param | meta


def _control_connection_target_port_key(dst_key: str, param_path: str) -> str:
    return f"{dst_key}:param:{param_path}" if param_path else dst_key


def _control_connections_for_patch(patch: "AnalyticPatch") -> list[RackConnectionView]:
    g = patch.routing
    conns: list[RackConnectionView] = []
    for e in g.edges:
        if getattr(e, "edge_kind", "signal") == "signal":
            continue
        conns.append(RackConnectionView(
            src_port_key=e.src_port or e.src_key,
            dst_port_key=e.dst_port or e.dst_key,
            edge_kind=getattr(e, "edge_kind", "control"),
            remove_kind="control",
        ))
    for pe in g.param_edges:
        conns.append(RackConnectionView(
            src_port_key=pe.src_port or pe.src_key,
            dst_port_key=pe.dst_port or _control_connection_target_port_key(pe.dst_key, pe.param_path),
            edge_kind=getattr(pe, "edge_kind", "control"),
            remove_kind="param",
        ))
    for me in getattr(g, "meta_edges", []):
        conns.append(RackConnectionView(
            src_port_key=me.a_port or me.a_key,
            dst_port_key=me.b_port or me.b_key,
            edge_kind=getattr(me, "edge_kind", "meta"),
            remove_kind="meta",
        ))
    return conns


def _published_port_lookup(patch: "AnalyticPatch") -> dict[str, PublishedPort]:
    lookup: dict[str, PublishedPort] = {}
    for ports in _published_ports_for_patch(patch).values():
        for port in ports:
            lookup[port.key] = port
    return lookup


def _is_state_machine_owner(patch: "AnalyticPatch", owner_key: str) -> bool:
    mod = next((m for m in getattr(patch, "modules", []) if m.key == owner_key), None)
    return mod is not None and getattr(mod, "module_type", "") == "state_machine"


def _negotiated_lane_policy(src: PublishedPort, dst: PublishedPort) -> str:
    src_policy = str(getattr(src.tensor, "group_validity", "strict") or "strict")
    dst_policy = str(getattr(dst.tensor, "group_validity", "strict") or "strict")
    if src_policy == "remap" or dst_policy == "remap":
        return "remap"
    if src_policy == "reduce" or dst_policy == "reduce":
        return "reduce"
    if src_policy == "broadcast" or dst_policy == "broadcast":
        return "broadcast"
    return "strict"


def _negotiate_edge_transfer(
    src_port: PublishedPort,
    dst_port: PublishedPort,
) -> EdgeTransferSpec | None:
    if src_port.direction != "out" or dst_port.direction != "in":
        return None
    if src_port.tensor.analytic_only is not True or dst_port.tensor.analytic_only is not True:
        return None
    if str(src_port.tensor.dtype or "") != "complex128":
        return None
    if str(dst_port.tensor.dtype or "") != "complex128":
        return None

    if dst_port.domain != "param_target":
        if src_port.domain != dst_port.domain:
            if src_port.domain not in {"signal", "control"} or dst_port.domain not in {"signal", "control"}:
                return None

    src_group = str(src_port.tensor.parallel_group or "")
    dst_group = str(dst_port.tensor.parallel_group or "")
    lane_policy = _negotiated_lane_policy(src_port, dst_port)
    if src_group and dst_group and src_group != dst_group:
        if not (src_port.negotiates_group_validity or dst_port.negotiates_group_validity):
            return None
        if lane_policy == "strict":
            lane_policy = "remap"

    src_lanes = max(1, int(src_port.tensor.lane_count or 1))
    dst_lanes = max(1, int(dst_port.tensor.lane_count or 1))
    src_dynamic = int(src_port.tensor.lane_count or 0) == 0
    dst_dynamic = int(dst_port.tensor.lane_count or 0) == 0

    if src_port.tensor.batch_axes != dst_port.tensor.batch_axes:
        if not (src_port.tensor.batchable and dst_port.tensor.batchable):
            return None
    if src_port.tensor.batch_axes == dst_port.tensor.batch_axes:
        batch_policy = "strict"
    elif src_port.tensor.batch_axes < dst_port.tensor.batch_axes:
        batch_policy = "broadcast"
    else:
        batch_policy = "reduce"

    reduction = ""
    if dst_port.domain == "param_target":
        transfer_policy = "remap" if lane_policy == "remap" else "identity"
    elif src_dynamic or dst_dynamic:
        transfer_policy = "broadcast"
        if lane_policy == "strict":
            lane_policy = "broadcast"
    elif src_lanes == dst_lanes:
        transfer_policy = "identity"
    elif src_lanes == 1 and dst_lanes > 1:
        transfer_policy = "broadcast"
        if lane_policy == "strict":
            lane_policy = "broadcast"
    elif src_lanes > 1 and dst_lanes == 1:
        transfer_policy = "reduce"
        if lane_policy == "strict":
            lane_policy = "reduce"
        reduction = "mean"
    else:
        transfer_policy = "remap"
        lane_policy = "remap"

    return EdgeTransferSpec(
        transfer_policy=transfer_policy,
        cable_count=max(src_lanes, dst_lanes, 1),
        batch_policy=batch_policy,
        lane_policy=lane_policy,
        reduction=reduction,
        analytic_only=True,
    )


def _negotiate_metaedge_transfer(
    src_port: PublishedPort,
    dst_port: PublishedPort,
) -> tuple[EdgeTransferSpec, EdgeTransferSpec] | None:
    fwd = _negotiate_edge_transfer(src_port, dst_port)
    if fwd is None:
        return None

    rev_policy = "identity"
    rev_lane = fwd.lane_policy
    rev_batch = fwd.batch_policy
    if fwd.transfer_policy == "broadcast":
        rev_policy = "reduce"
        if rev_lane == "strict":
            rev_lane = "reduce"
    elif fwd.transfer_policy == "reduce":
        rev_policy = "broadcast"
        if rev_lane == "strict":
            rev_lane = "broadcast"
    elif fwd.transfer_policy == "remap":
        rev_policy = "remap"

    rev = EdgeTransferSpec(
        transfer_policy=rev_policy,
        cable_count=fwd.cable_count,
        batch_policy=rev_batch,
        lane_policy=rev_lane,
        reduction="mean" if rev_policy == "reduce" else "",
        analytic_only=True,
    )
    return fwd, rev


def _ports_compatible(src: PublishedPort, dst: PublishedPort) -> bool:
    return _negotiate_edge_transfer(src, dst) is not None


def _control_remove_connections_for_port(patch: "AnalyticPatch", port_key: str) -> None:
    patch.routing.edges = [
        e for e in patch.routing.edges
        if (e.src_port or e.src_key) != port_key
        and (e.dst_port or e.dst_key) != port_key
    ]
    patch.routing.param_edges = [
        pe for pe in patch.routing.param_edges
        if (pe.src_port or pe.src_key) != port_key
        and (pe.dst_port or _control_connection_target_port_key(pe.dst_key, pe.param_path)) != port_key
    ]
    patch.routing.meta_edges = [
        me for me in getattr(patch.routing, "meta_edges", [])
        if (me.a_port or me.a_key) != port_key
        and (me.b_port or me.b_key) != port_key
    ]


def _control_add_connection(
    patch: "AnalyticPatch",
    src_port: PublishedPort,
    dst_port: PublishedPort,
) -> bool:
    transfer_spec = _negotiate_edge_transfer(src_port, dst_port)
    if transfer_spec is None:
        return False

    if (
        dst_port.domain != "param_target"
        and _is_state_machine_owner(patch, src_port.owner_key)
        and _is_state_machine_owner(patch, dst_port.owner_key)
    ):
        meta_specs = _negotiate_metaedge_transfer(src_port, dst_port)
        if meta_specs is None:
            return False
        a_to_b, b_to_a = meta_specs
        patch.routing.meta_edges.append(MetaEdge(
            a_key=src_port.owner_key,
            b_key=dst_port.owner_key,
            a_port=src_port.key,
            b_port=dst_port.key,
            edge_kind="meta",
            semantic_role=src_port.semantic_role or dst_port.semantic_role or src_port.tensor.semantic_role or dst_port.tensor.semantic_role,
            channel_count=max(1, int(src_port.tensor.lane_count or dst_port.tensor.lane_count or 1)),
            tensor_contract=src_port.tensor.to_contract(),
            a_to_b_transfer=a_to_b,
            b_to_a_transfer=b_to_a,
        ))
        return True

    if dst_port.domain == "param_target":
        patch.routing.param_edges.append(ParamEdge(
            src_key=src_port.key,
            dst_key=dst_port.owner_key,
            weight=1.0,
            extractor="magnitude",
            delay_samples=0,
            param_path=dst_port.param_path,
            src_port=src_port.key,
            dst_port=dst_port.key,
            edge_kind="control",
            projection_policy=str(getattr(dst_port, "projection_policy", "") or "magnitude_mean"),
            tensor_contract=src_port.tensor.to_contract(),
            transfer_spec=transfer_spec,
        ))
        return True

    patch.routing.add_node(src_port.key)
    patch.routing.add_node(dst_port.key)
    patch.routing.edges.append(RoutingEdge(
        src_key=src_port.key,
        dst_key=dst_port.key,
        weight=1.0,
        angle_rad=0.0,
        delay_s=0.0,
        edge_kind="control",
        src_port=src_port.key,
        dst_port=dst_port.key,
        tensor_contract=src_port.tensor.to_contract(),
        transfer_spec=transfer_spec,
    ))
    return True


def _make_param_target_port(
    owner_key: str,
    owner_label: str,
    param_path: str,
    *,
    group: str = "Params",
    color: tuple[int, int, int] = (180, 140, 220),
    tensor_rank: int = 0,
    lane_count: int = 1,
    parallel_group: str = "",
    semantic_role: str = "param_target",
    projection_policy: str = "magnitude_mean",
) -> PublishedPort:
    return PublishedPort(
        key=f"{owner_key}:param:{param_path}",
        label=f"{owner_label}.{param_path}",
        direction="in",
        domain="param_target",
        owner_key=owner_key,
        group=group,
        param_path=param_path,
        color=color,
        tensor=PortTensorSpec(
            tensor_rank=tensor_rank,
            lane_count=lane_count,
            group_validity="strict",
            parallel_group=parallel_group,
            semantic_role=semantic_role,
            batchable=True,
        ),
        semantic_role=semantic_role,
        projection_policy=projection_policy,
        negotiates_group_validity=True,
    )


def _published_ports_for_voice(v: "AnalyticVoice") -> list[PublishedPort]:
    ports: list[PublishedPort] = [
        PublishedPort(
            key=v.key,
            label=v.label,
            direction="out",
            domain="signal",
            owner_key=v.key,
            group="Signal",
            color=tuple(v.color[:3]),
            tensor=PortTensorSpec(tensor_rank=1, lane_count=0, parallel_group="voice_signal"),
            negotiates_group_validity=True,
        ),
    ]
    for path in (
        "amplitude",
        "phase_origin",
        "chirp.f_delta_start",
        "chirp.f_delta_end",
        "chirp.tau",
        "chirp.chirp_power",
        "harmonic_brightness",
        "harmonic_warp_strength",
    ):
        ports.append(_make_param_target_port(
            v.key, v.label, path,
            color=tuple(v.color[:3]),
            parallel_group="voice_param_batch",
        ))
    return ports


def _published_ports_for_mixer(m: "AnalyticMixer") -> list[PublishedPort]:
    ports: list[PublishedPort] = [
        PublishedPort(
            key=m.key,
            label=m.label,
            direction="out",
            domain="signal",
            owner_key=m.key,
            group="Signal",
            color=tuple(m.color[:3]),
            tensor=PortTensorSpec(tensor_rank=1, lane_count=0, parallel_group="mixer_signal"),
            negotiates_group_validity=True,
        ),
    ]
    for path in ("projection_active",):
        ports.append(_make_param_target_port(
            m.key, m.label, path,
            group="Mixer",
            color=tuple(m.color[:3]),
            parallel_group="mixer_param_batch",
        ))
    return ports


def _published_ports_for_param_node(pn: "ParamNode") -> list[PublishedPort]:
    ports: list[PublishedPort] = [
        PublishedPort(
            key=pn.key,
            label=pn.label,
            direction="in",
            domain="control",
            owner_key=pn.key,
            group="Param Node",
            color=tuple(pn.color[:3]),
            tensor=PortTensorSpec(tensor_rank=1, lane_count=0, parallel_group="param_control_in"),
            negotiates_group_validity=True,
        ),
        PublishedPort(
            key=f"{pn.key}:out",
            label=f"{pn.label}.out",
            direction="out",
            domain="control",
            owner_key=pn.key,
            group="Param Node",
            color=tuple(pn.color[:3]),
            tensor=PortTensorSpec(tensor_rank=1, lane_count=0, parallel_group="param_control_out"),
            negotiates_group_validity=True,
        ),
    ]
    for tgt in getattr(pn, "targets", []):
        path = str(tgt.get("attr", "") or "")
        if path:
            ports.append(_make_param_target_port(
                pn.key,
                pn.label,
                path,
                group="Targets",
                color=tuple(pn.color[:3]),
                parallel_group="param_target_batch",
            ))
    return ports


def _published_ports_for_module(mod: "AnalyticModule") -> list[PublishedPort]:
    ports: list[PublishedPort] = []
    base_color = tuple(mod.color[:3])
    ports.append(PublishedPort(
        key=mod.key,
        label=mod.label,
        direction="out",
        domain="signal",
        owner_key=mod.key,
        group="Signal",
        color=base_color,
        tensor=PortTensorSpec(
            tensor_rank=1,
            lane_count=0,
            parallel_group=f"module_signal:{mod.module_type}",
            semantic_role="signal",
        ),
        semantic_role="signal",
        negotiates_group_validity=True,
    ))
    if mod.module_type == "state_machine":
        for bundle in getattr(mod, "sm_bundle_ports", []):
            bundle_name = str(bundle.get("name", "") or "").strip()
            if not bundle_name:
                continue
            direction = str(bundle.get("direction", "out") or "out")
            domain = str(bundle.get("domain", "control") or "control")
            semantic_role = str(bundle.get("semantic_role", bundle_name) or bundle_name)
            channel_dims = [int(x) for x in bundle.get("channel_dims", [])]
            ports.append(PublishedPort(
                key=f"{mod.key}:{bundle_name}",
                label=f"{mod.label}.{bundle_name}",
                direction=direction,
                domain=domain,
                owner_key=mod.key,
                group=str(bundle.get("group", "SM Bundle") or "SM Bundle"),
                color=base_color,
                tensor=PortTensorSpec(
                    tensor_rank=max(0, int(bundle.get("tensor_rank", max(1, len(channel_dims))))),
                    lane_count=max(0, int(bundle.get("lane_count", 0))),
                    batch_axes=max(0, int(bundle.get("batch_axes", 1))),
                    parallel_group=str(bundle.get("parallel_group", f"sm_bundle:{mod.key}:{bundle_name}") or f"sm_bundle:{mod.key}:{bundle_name}"),
                    group_validity=str(bundle.get("group_validity", "remap") or "remap"),
                    semantic_role=semantic_role,
                    channel_dims=channel_dims,
                ),
                semantic_role=semantic_role,
                negotiates_group_validity=True,
            ))
        # Primary SM signal outputs
        for out_key in mod.sm_out_keys():
            ports.append(PublishedPort(
                key=out_key,
                label=out_key,
                direction="out",
                domain="signal",
                owner_key=mod.key,
                group="SM Signal",
                color=base_color,
                tensor=PortTensorSpec(
                    tensor_rank=1,
                    lane_count=max(0, int(getattr(mod, "sm_n_items", 1))),
                    parallel_group=f"sm_signal:{mod.key}",
                    semantic_role="sm_signal",
                ),
                semantic_role="sm_signal",
                negotiates_group_validity=True,
            ))
        # Declared control feedback ports — one per item/var pair to keep them
        # unique and compatible with the unified graph.
        for item in getattr(mod, "sm_items", []):
            for var in getattr(mod, "sm_vars", []):
                ctrl_key = f"{mod.key}.ctrl.{item}.{var}"
                ports.append(PublishedPort(
                    key=ctrl_key,
                    label=ctrl_key,
                    direction="out",
                    domain="control",
                    owner_key=mod.key,
                    group="SM Control",
                    color=base_color,
                    tensor=PortTensorSpec(
                        tensor_rank=1,
                        lane_count=max(0, int(getattr(mod, "sm_n_items", 1))),
                        parallel_group=f"sm_ctrl:{mod.key}",
                        semantic_role="sm_control",
                    ),
                    semantic_role="sm_control",
                    negotiates_group_validity=True,
                ))
    for path in ("rate_hz", "depth", "phase_offset", "sm_n_items"):
        if hasattr(mod, path.split(".")[0]):
            ports.append(_make_param_target_port(
                mod.key,
                mod.label,
                path,
                group="Module",
                color=base_color,
                parallel_group=f"module_param:{mod.module_type}",
            ))
    return ports


def _published_ports_for_router(router: "RouterInstance") -> list[PublishedPort]:
    label = f"{router.label} [{router.router_type}]"
    base_color = {
        "voice_router": (90, 160, 230),
        "instrument": (110, 200, 150),
        "master": (230, 180, 90),
    }.get(router.router_type, (160, 160, 180))
    return [
        _make_param_target_port(
            router.key, label, "feedback.enabled",
            group="Router", color=base_color, parallel_group="router_param_batch"),
        _make_param_target_port(
            router.key, label, "feedback.decay",
            group="Router", color=base_color, parallel_group="router_param_batch"),
        _make_param_target_port(
            router.key, label, "feedback.max_iterations",
            group="Router", color=base_color, parallel_group="router_param_batch"),
    ]


def _published_ports_for_patch(patch: "AnalyticPatch") -> dict[str, list[PublishedPort]]:
    out: dict[str, list[PublishedPort]] = {}
    for v in patch.voices:
        out[v.key] = _published_ports_for_voice(v)
    for m in patch.mixers:
        out[m.key] = _published_ports_for_mixer(m)
    for mod in patch.modules:
        out[mod.key] = _published_ports_for_module(mod)
    for cs in patch.controls:
        ports: list[PublishedPort] = []
        for sl in cs.sliders:
            ports.append(PublishedPort(
                key=sl.key,
                label=f"{cs.label}/{sl.label}",
                direction="out",
                domain="control",
                owner_key=cs.key,
                group="Controls",
                color=tuple(cs.color[:3]),
                tensor=PortTensorSpec(tensor_rank=0, lane_count=1, parallel_group="control_scalar"),
            ))
        out[cs.key] = ports
    for pn in patch.param_nodes:
        out[pn.key] = _published_ports_for_param_node(pn)
    for router in getattr(patch, "routers", []):
        out[_router_ui_key(router.key)] = _published_ports_for_router(router)
    return out


def _rack_device_views_for_patch(patch: "AnalyticPatch") -> list[RackDeviceView]:
    published = _published_ports_for_patch(patch)
    device_order: list[tuple[str, str, tuple[int, int, int], str]] = []
    for v in patch.voices:
        device_order.append((v.key, v.label, tuple(v.color[:3]), "voice"))
    for mod in patch.modules:
        device_order.append((mod.key, mod.label, tuple(mod.color[:3]), f"module:{mod.module_type}"))
    for cs in patch.controls:
        device_order.append((cs.key, cs.label, tuple(cs.color[:3]), "control"))
    for pn in patch.param_nodes:
        device_order.append((pn.key, pn.label, tuple(pn.color[:3]), "param"))
    for mix in patch.mixers:
        device_order.append((mix.key, mix.label, tuple(mix.color[:3]), "mixer"))
    for router in getattr(patch, "routers", []):
        device_order.append((_router_ui_key(router.key), router.label,
                             {
                                 "voice_router": (90, 160, 230),
                                 "instrument": (110, 200, 150),
                                 "master": (230, 180, 90),
                             }.get(router.router_type, (160, 160, 180)),
                             f"router:{router.router_type}"))

    rack: list[RackDeviceView] = []
    x_slot = 0
    y_u = 0
    max_cols = 6
    for device_key, label, color, kind in device_order:
        ports = list(published.get(device_key, []))
        width = max(1, min(4, (len(ports) + 7) // 8))
        height = max(1, min(4, (len(ports) + width * 7) // max(width * 8, 1)))
        if x_slot + width > max_cols:
            x_slot = 0
            y_u += 4
        port_views: list[RackPortView] = []
        cols = max(1, width * 4)
        for pi, port in enumerate(ports):
            port_views.append(RackPortView(
                port=port,
                local_x=8 + (pi % cols) * 8,
                local_y=12 + (pi // cols) * 10,
                radius=3,
            ))
        rack.append(RackDeviceView(
            device_key=device_key,
            label=label,
            device_kind=kind,
            color=color,
            rack_u=height,
            rack_w=width,
            grid_x=x_slot,
            grid_y=y_u,
            ports=port_views,
        ))
        x_slot += width
    return rack


@dataclass
class SystemAudioDevice:
    output_device_name: str = ""
    output_channels:    int = 2
    input_device_name:  str = ""
    input_channels:     int = 0
    export_to_file:     bool = True
    export_sample_rate: int  = 48000
    export_bit_depth:   int  = 24
    # Transient runtime state
    _reported_output_devices: list = field(default_factory=list)
    _reported_input_devices:  list = field(default_factory=list)
    _reported_output_name:    str  = ""
    _reported_input_name:     str  = ""
    _reported_output_hw_channels: int = 0
    _reported_input_hw_channels:  int = 0
    _reported_output_hw_rate: int = 0
    _reported_input_hw_rate:  int = 0
    _preview_backend:         str  = "pygame.mixer"
    _preview_backend_channels:int  = 2
    _input_buffers:           list = field(default_factory=list)  # list[np.ndarray]

    def to_dict(self) -> dict:
        return {
            "output_device_name": self.output_device_name,
            "output_channels":    self.output_channels,
            "input_device_name":  self.input_device_name,
            "input_channels":     self.input_channels,
            "export_to_file":     self.export_to_file,
            "export_sample_rate": self.export_sample_rate,
            "export_bit_depth":   self.export_bit_depth,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SystemAudioDevice":
        o = cls()
        o.output_device_name = str(d.get("output_device_name", ""))
        o.output_channels    = max(1, int(d.get("output_channels", 2)))
        o.input_device_name  = str(d.get("input_device_name", ""))
        o.input_channels     = max(0, int(d.get("input_channels", 0)))
        o.export_to_file     = bool(d.get("export_to_file", True))
        o.export_sample_rate = int(d.get("export_sample_rate", 48000))
        o.export_bit_depth   = int(d.get("export_bit_depth", 24))
        return o

    @classmethod
    def knobs(cls) -> list["KnobSpec"]:
        return [
            KnobSpec("output_channels",    "Output Ch.", "int",  2, 1, 16, 1, "", [], False, "Routing Device", ".0f"),
            KnobSpec("input_channels",     "Input Ch.",  "int",  0, 0, 16, 1, "", [], False, "Routing Device", ".0f"),
            KnobSpec("export_to_file",     "Render Main","bool", True, 0, 1, 0, "", [], False, "File Render"),
            KnobSpec("export_sample_rate", "Render SR",  "int", 48000, 8000, 384000, 1, "Hz", [], False, "File Render", ".0f"),
            KnobSpec("export_bit_depth",   "Render Bits","int", 24, 16, 32, 1, "", [], False, "File Render", ".0f"),
        ]

    def output_key(self, idx: int) -> str:
        return f"__sys_out_{idx + 1}__"

    def input_key(self, idx: int) -> str:
        return f"__sys_in_{idx + 1}__"

    def output_keys(self) -> list[str]:
        return [self.output_key(i) for i in range(max(1, int(self.output_channels)))]

    def input_keys(self) -> list[str]:
        return [self.input_key(i) for i in range(max(0, int(self.input_channels)))]


try:
    from granular_engine import GrainPopulationSpec as _GrainPopulationSpec
    from granular_engine import GranularClusterDriver as _GranularClusterDriver
    _HAS_GRANULAR = True
except ImportError:
    _HAS_GRANULAR = False
    _GrainPopulationSpec = None  # type: ignore[assignment,misc]
    _GranularClusterDriver = None  # type: ignore[assignment]

from routing_engine import (RoutingEdge, FeedbackConfig, RoutingGraph,
                             ParamEdge, solve_routing_complex,
                             solve_routing_with_ringdown, estimate_ringdown_samples,
                             compute_latency_compensation, solve_param_routing,
                             _safe_inverse, RouterInstance, MIXER_LAYERS,
                             EdgeTransferSpec, TensorPortContract, MetaEdge)
from patch_to_driver import (build_driver_config,
                              _resolve_f0, _build_harmonics, _build_env_knots,
                              _CHIRP_CODE, CHIRP_NONE)
from performer_engine import (init_driver_state, multi_level_driver_step,
                               DriverConfig, DriverState, driver_synthesis_step)
from routing_solve_torch import CompiledRouter


def _list_audio_devices(iscapture: bool) -> list[str]:
    try:
        if not pygame.get_init():
            return []
        from pygame._sdl2.audio import get_audio_device_names
        return [str(x) for x in get_audio_device_names(bool(iscapture))]
    except Exception:
        return []


def _probe_default_sounddevice(iscapture: bool, requested_channels: int,
                               requested_rate: int = 48000) -> tuple[str, int, int]:
    try:
        import sounddevice as sd
        devsel = sd.default.device
        if isinstance(devsel, (list, tuple)):
            dev_idx = int(devsel[0] if iscapture else devsel[1])
        else:
            dev_idx = int(devsel)
        info = sd.query_devices(dev_idx)
        name = str(info.get("name", ""))
        max_ch_key = "max_input_channels" if iscapture else "max_output_channels"
        channels = int(info.get(max_ch_key, 0))
        rate = int(float(info.get("default_samplerate", requested_rate) or requested_rate))
        return name, max(0, channels), max(0, rate)
    except Exception:
        return "", 0, 0


def _probe_audio_device(name: str, iscapture: bool, requested_channels: int,
                        requested_rate: int = 48000) -> tuple[str, int, int]:
    if not str(name or "").strip():
        return _probe_default_sounddevice(iscapture, requested_channels, requested_rate)
    try:
        if not pygame.get_init():
            return "", 0, 0
        from pygame._sdl2.audio import (
            AudioDevice, AUDIO_F32, AUDIO_ALLOW_ANY_CHANGE,
        )
        names = _list_audio_devices(iscapture)
        devname = str(name or "").strip()
        if not devname:
            if not names:
                return "", 0, 0
            devname = names[0]

        def _probe_cb(_dev, mv):
            if not iscapture:
                try:
                    mv[:] = b"\x00" * len(mv)
                except Exception:
                    pass

        dev = AudioDevice(
            devicename=devname,
            iscapture=bool(iscapture),
            frequency=max(8000, int(requested_rate)),
            audioformat=AUDIO_F32,
            numchannels=max(1, int(requested_channels)),
            chunksize=512,
            allowed_changes=AUDIO_ALLOW_ANY_CHANGE,
            callback=_probe_cb,
        )
        actual = max(0, int(getattr(dev, "numchannels", 0)))
        actual_rate = max(0, int(getattr(dev, "frequency", 0)))
        actual_name = str(getattr(dev, "devicename", devname))
        dev.close()
        return actual_name, actual, actual_rate
    except Exception:
        return str(name or "").strip(), 0, 0


def _refresh_system_audio_report(sysdev: "SystemAudioDevice") -> None:
    sysdev._reported_output_devices = _list_audio_devices(False)
    sysdev._reported_input_devices = _list_audio_devices(True)
    out_name, out_ch, out_rate = _probe_audio_device(
        sysdev.output_device_name, False,
        max(1, sysdev.output_channels),
        requested_rate=max(8000, int(sysdev.export_sample_rate)),
    )
    in_req = max(1, sysdev.input_channels) if sysdev.input_channels > 0 else 2
    in_name, in_ch, in_rate = _probe_audio_device(
        sysdev.input_device_name, True, in_req,
        requested_rate=max(8000, int(sysdev.export_sample_rate)),
    )
    sysdev._reported_output_name = out_name
    sysdev._reported_input_name = in_name
    sysdev._reported_output_hw_channels = out_ch
    sysdev._reported_input_hw_channels = in_ch
    sysdev._reported_output_hw_rate = out_rate
    sysdev._reported_input_hw_rate = in_rate


def _prepare_output_bus_for_device(out_bus: np.ndarray, src_sr: int,
                                   dst_sr: int, dst_channels: int) -> np.ndarray:
    """Resample and channel-map a float bus for an SDL audio device."""
    bus = np.asarray(out_bus, dtype=np.float32)
    if bus.ndim == 1:
        bus = bus[:, None]
    if bus.shape[1] < 1:
        bus = np.zeros((len(bus), 1), dtype=np.float32)
    if dst_sr != src_sr:
        cols = []
        for ci in range(bus.shape[1]):
            cols.append(_resample_audio(bus[:, ci], src_sr, dst_sr))
        bus = np.column_stack(cols).astype(np.float32, copy=False)
    dst_ch = max(1, int(dst_channels))
    if bus.shape[1] < dst_ch:
        if bus.shape[1] == 1:
            bus = np.repeat(bus, dst_ch, axis=1)
        else:
            reps = (dst_ch + bus.shape[1] - 1) // bus.shape[1]
            bus = np.tile(bus, (1, reps))[:, :dst_ch]
    elif bus.shape[1] > dst_ch:
        bus = bus[:, :dst_ch]
    return np.clip(bus, -1.0, 1.0).astype(np.float32, copy=False)



@dataclass
class ParamNode:
    """A routing-graph node whose complex output is extracted to a scalar time series
    and written to one or more target voice attributes, enabling smooth parametric modulation.

    Signal flow
    -----------
    1. ParamNode appears in the routing graph as a regular node (zero independent source).
    2. Other nodes feed into it via standard RoutingEdges (weight / angle / delay).
    3. After the routing solve, its complex output is converted to float64 via *extractor*.
    4. The resulting time series is injected as a *param_override* when re-synthesizing
       each target voice, keeping the derivative smooth even under heavy modulation.
    5. Multiple targets can share the same param signal (multi-source → multi-target).

    Extractor choices (mirror routing_engine.ParamEdge extractors)
    --------------------------------------------------------------
    magnitude  |z|   · always positive · good for density / amplitude driving
    real       Re(z) · signed · follows analytic real part
    imag       Im(z) · signed quadrature
    phase      arg(z) · [-π, π] · good for pitch tracking
    energy     |z|²  · heavier weighting of loud moments
    rms        smoothed magnitude (128-sample window)
    """
    key:           str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:         str   = "Param"
    targets:       list  = field(default_factory=list)   # list[{"voice_key": str, "attr": str}]
    extractor:     str   = "magnitude"
    default_value: float = 0.0     # output when no routing edges feed this node
    low:           float = 0.0     # output is clamped to [low, high]
    high:          float = 1.0
    color: list = field(default_factory=lambda: [180, 140, 220])

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "targets": list(self.targets),
                "extractor": self.extractor,
                "default_value": self.default_value,
                "low": self.low, "high": self.high, "color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "ParamNode":
        o = cls.__new__(cls)
        o.key          = d.get("key", uuid.uuid4().hex[:8])
        o.label        = d.get("label", "Param")
        # Migrate legacy single-target fields to the targets list
        if "targets" in d:
            o.targets = [dict(t) for t in d["targets"]]
        else:
            vk = d.get("target_voice_key", "")
            at = d.get("target_attr", "")
            o.targets = [{"voice_key": vk, "attr": at}] if (vk or at) else []
        o.extractor    = d.get("extractor", "magnitude")
        o.default_value = float(d.get("default_value", 0.0))
        o.low          = float(d.get("low", 0.0))
        o.high         = float(d.get("high", 1.0))
        o.color        = d.get("color", [180, 140, 220])
        return o

    _EXTRACTORS = ["magnitude", "real", "imag", "phase", "energy", "rms"]

    @classmethod
    def knobs(cls) -> list:
        """Static knobs — label, extractor, default/clamp.
        Target voice/attr are rendered dynamically by the panel (dynamic dropdown)."""
        return [
            KnobSpec("label",         "Label",     "str",    "Param", 0, 0, 0, "", [], False, "Param Node"),
            KnobSpec("extractor",     "Extractor", "choice", "magnitude", 0, 5, 1, "",
                     cls._EXTRACTORS, False, "Param Node"),
            KnobSpec("default_value", "Default",   "float", 0.0, -1e4, 1e4, 0, "", [], False, "Param Node"),
            KnobSpec("low",           "Min clamp", "float", 0.0, -1e4, 1e4, 0, "", [], False, "Param Node"),
            KnobSpec("high",          "Max clamp", "float", 1.0, -1e4, 1e4, 0, "", [], False, "Param Node"),
        ]


# ---------------------------------------------------------------------------
# ControlSlider / ControlSurface — user-configurable interactive controls
#
# Each ControlSlider is an independent routing-graph DC source node.  Its
# output is a constant complex signal at its current scaled value, so any
# ParamNode that receives it via a RoutingEdge will be driven by the slider.
# Multiple sliders are grouped under one ControlSurface for the UI — the
# surface itself is a voice-list item; its sliders each appear in the routing
# graph individually.
#
# "n-channel of any edge": each slider is its own key in the routing graph,
# so a single surface can fan out N independent DC lanes to N distinct
# ParamNodes (or share them) just by drawing routing edges.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SidecarBus — source-identified per-channel metadata carried alongside
# analytic signals through the synthesis and routing pipeline.
#
# Every synthesis source (voice, LFO, module, control slider) may register
# arbitrary named numpy arrays keyed by (source_key, channel_name).  The bus
# is built during _synthesize_patch and returned to callers that request it
# via the _return_sidecar parameter.
#
# Conventions (not enforced — any channel name is legal):
#   "envelope"   float64   amplitude envelope of a voice, shape (n,), range [0,1]
#   "amplitude"  float64   |z(t)| instantaneous magnitude
#   "phase"      float64   arg(z(t)) instantaneous phase in radians
#   "frequency"  float64   instantaneous frequency in Hz (derived from analytic phase)
#   "value"      float64   DC scalar for a ControlSlider, broadcast to (n,)
#   "routed"     complex128 full routed output X[node] after the routing solve
#
# Usage pattern:
#   bus.put(key, "envelope", env_array)          # populate during synthesis
#   env = bus.get(key, "envelope")               # retrieve later (or None)
#   up  = bus.upstream_of(dst_key, edges)        # view from nodes feeding dst
# ---------------------------------------------------------------------------
@dataclass
class SidecarBus:
    """Carries source-identified, named time-series metadata alongside analytic signals.

    data: dict[source_key: str, dict[channel_name: str, np.ndarray]]
    """
    data: dict = field(default_factory=dict)

    def put(self, source_key: str, channel: str, arr: np.ndarray) -> None:
        """Register a numpy array under (source_key, channel).  Arrays are stored
        by reference — callers should pass already-computed arrays; copying is the
        caller's responsibility when mutation is a concern."""
        if source_key not in self.data:
            self.data[source_key] = {}
        self.data[source_key][channel] = arr

    def get(self, source_key: str, channel: str, n: int = 0) -> "np.ndarray | None":
        """Return the channel array for source_key, or None if absent.
        When *n* > 0 the array is sliced to at most *n* samples."""
        d = self.data.get(source_key)
        if d is None:
            return None
        arr = d.get(channel)
        if arr is None:
            return None
        return arr[:n] if (n > 0 and n < len(arr)) else arr

    def sources(self) -> list:
        """All registered source keys."""
        return list(self.data.keys())

    def channels(self, source_key: str) -> list:
        """Channel names available for *source_key*."""
        return list(self.data.get(source_key, {}).keys())

    def upstream_of(self, dst_key: str, edges: list) -> "SidecarBus":
        """Return a new SidecarBus containing only sidecar from nodes that
        directly feed *dst_key* via a RoutingEdge (single-hop upstream)."""
        src_keys = {e.src_key for e in edges if e.dst_key == dst_key}
        result = SidecarBus()
        for sk in src_keys:
            if sk in self.data:
                result.data[sk] = self.data[sk]
        return result

    def all_upstream_of(self, dst_key: str, edges: list) -> "SidecarBus":
        """Return sidecar from *all* transitive ancestors of *dst_key*
        (BFS through the routing graph).  Useful for inspecting the full
        signal lineage that contributed to a node."""
        visited: set = set()
        queue: list = [dst_key]
        result = SidecarBus()
        edge_map: dict = {}
        for e in edges:
            edge_map.setdefault(e.dst_key, []).append(e.src_key)
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for src in edge_map.get(node, []):
                if src in self.data:
                    result.data[src] = self.data[src]
                queue.append(src)
        return result

    def trim(self, n: int) -> None:
        """Trim every channel array to at most *n* samples in-place."""
        for sk in self.data:
            for ch in list(self.data[sk]):
                arr = self.data[sk][ch]
                if len(arr) > n:
                    self.data[sk][ch] = arr[:n]

    def merge(self, other: "SidecarBus") -> None:
        """Absorb all channels from *other* (last-write-wins on key collision)."""
        for sk, chs in other.data.items():
            if sk not in self.data:
                self.data[sk] = {}
            self.data[sk].update(chs)


@dataclass
class ControlSlider:
    key:    str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:  str   = "Ctrl"
    value:  float = 0.5    # normalised position [0, 1]
    low:    float = 0.0    # maps value=0 → low
    high:   float = 1.0    # maps value=1 → high
    is_log: bool  = False
    color:  list  = field(default_factory=lambda: [140, 200, 160])

    def scaled_value(self) -> float:
        """Return the actual value in [low, high] from the normalised position."""
        if self.is_log and self.low > 0 and self.high > 0:
            return self.low * (self.high / self.low) ** max(0.0, min(1.0, self.value))
        return self.low + max(0.0, min(1.0, self.value)) * (self.high - self.low)

    @classmethod
    def knobs(cls) -> list:
        return [
            KnobSpec("label",  "Label", "str",   "Ctrl", 0,    0,   0, "",  [], False, "Slider"),
            KnobSpec("low",    "Low",   "float",  0.0, -1e6, 1e6,   0, "",  [], False, "Slider", ".4g"),
            KnobSpec("high",   "High",  "float",  1.0, -1e6, 1e6,   0, "",  [], False, "Slider", ".4g"),
            KnobSpec("is_log", "Log",   "bool",  False, 0,    1,    0, "",  [], False, "Slider"),
            KnobSpec("value",  "Value", "float",  0.5,  0.0,  1.0,  0, "",  [], False, "Slider", ".4f"),
        ]

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "value": self.value,
                "low": self.low, "high": self.high,
                "is_log": self.is_log, "color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "ControlSlider":
        o = cls.__new__(cls)
        o.key    = d.get("key",    uuid.uuid4().hex[:8])
        o.label  = d.get("label",  "Ctrl")
        o.value  = float(d.get("value",  0.5))
        o.low    = float(d.get("low",    0.0))
        o.high   = float(d.get("high",   1.0))
        o.is_log = bool(d.get("is_log",  False))
        o.color  = d.get("color", [140, 200, 160])
        return o


@dataclass
class ControlSurface:
    """Named group of ControlSliders; each slider is a routing-graph DC node.

    Selecting a ControlSurface in the voice list shows all its sliders in the
    PartialPanel, making them interactively adjustable in real time.  Adding a
    RoutingEdge from a slider's key to a ParamNode's key lets the slider drive
    any voice attribute continuously.
    """
    key:     str  = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:   str  = "Control"
    color:   list = field(default_factory=lambda: [100, 180, 140])
    sliders: list = field(default_factory=list)   # list[ControlSlider]

    @classmethod
    def knobs(cls) -> list:
        # Only the surface-level label is a knob; individual sliders are
        # rendered as a bespoke multi-slider UI in PartialPanel.
        return [
            KnobSpec("label", "Label", "str", "Control", 0, 0, 0, "", [], False, "Surface"),
        ]

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "color": self.color,
                "sliders": [s.to_dict() for s in self.sliders]}

    @classmethod
    def from_dict(cls, d: dict) -> "ControlSurface":
        o = cls.__new__(cls)
        o.key     = d.get("key",   uuid.uuid4().hex[:8])
        o.label   = d.get("label", "Control")
        o.color   = d.get("color", [100, 180, 140])
        o.sliders = [ControlSlider.from_dict(s) for s in d.get("sliders", [])]
        return o


# ---------------------------------------------------------------------------
# AnalyticModule — a bespoke signal-processing node in the routing graph.
#
# Unlike AnalyticVoice (full synthesis driver) or AnalyticMixer (routing
# aggregator), a Module has a self-contained signal generation algorithm
# selected by *module_type*.  It participates in the routing graph fully —
# receiving and sending analytic signals via RoutingEdges — but its
# independent source signal is produced entirely by its own algorithm.
#
# LFO is the first module type.  LFODefinition remains for backward
# compatibility with saved patches, but new patches should use AnalyticModule
# with module_type="lfo".  A Module LFO gains one capability LFODefinition
# lacks: because it is a proper routing node, signals from other nodes can
# be summed into its output analytically (ring-modulation, sub-mixing).
# To FM its rate, route a voice → ParamNode targeting the module's rate_hz.
#
# Current module types
# --------------------
#   lfo          Analytic LFO oscillator: rate_hz / shape / depth /
#                phase_offset.  Routing inputs add to the output (AM/ring-mod).
#   passthrough  Zero independent source — output is the sum of routing inputs.
#                Useful as a named sub-bus / side-chain point.
# ---------------------------------------------------------------------------
@dataclass
class AnalyticModule:
    key:         str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:       str   = "Module"
    module_type: str   = "lfo"
    muted:       bool  = False
    color:       list  = field(default_factory=lambda: [200, 140, 220])
    # LFO parameters (meaningful when module_type == "lfo")
    rate_hz:      float = 1.0
    shape:        str   = "Sine"
    phase_offset: float = 0.0
    depth:        float = 1.0
    # PitchQuantizer parameters (meaningful when module_type == "pitch_quantizer")
    quantizer_scale_degrees: list  = field(default_factory=list)  # [] = use tuning.scale_degrees
    interpolation_mode:      str   = "discrete"   # see QuantizerHandle._INTERP_MODES
    portamento_time:         float = 0.05   # s  — glide time (portamento/spline/legato)
    slew_rate:               float = 100.0  # semitones/s
    slew2_accel:             float = 200.0  # semitones/s²
    spline_tension:          float = 0.5    # Hermite tangent scale
    # Multi-channel LFO (meaningful when module_type == "lfo" and non-empty)
    # Each entry: {scale, amplitude, rate_hz, tension, phase_offset, shape}
    # Empty → legacy single-channel behaviour from rate_hz/shape/phase_offset/depth.
    lfo_channels:            list  = field(default_factory=list)
    # Interaural parameters (meaningful when module_type == "interaural")
    # ch1 params always active; ch2 params used only when ch2 input edges exist
    iau_azimuth:             float = 0.0    # [-1, 1]  left=−1, right=+1
    iau_elevation:           float = 0.0    # [-1, 1]  down=−1, up=+1
    iau_distance:            float = 0.0    # [0, 1]   near=0, far=1
    iau_width:               float = 0.0    # [0, 1]   point=0, spread=1
    iau_azimuth_ch2:         float = 0.0
    iau_elevation_ch2:       float = 0.0
    iau_distance_ch2:        float = 0.0
    iau_width_ch2:           float = 0.0
    # State machine parameters (meaningful when module_type == "state_machine")
    sm_plugin:      str   = ""     # plugin filename stem (no path, no .py)
    sm_n_items:     int   = 1      # number of physics items
    sm_items:       list  = field(default_factory=list)  # item names from plugin
    sm_vars:        list  = field(default_factory=list)  # output var names from plugin
    sm_bundle_ports: list = field(default_factory=list)  # declared wide/bundle ports for graph authoring
    sm_state_vars:  list  = field(default_factory=list)  # persisted scalar state vars from plugin
    sm_params:      dict  = field(default_factory=dict)  # plugin parameter values
    sm_use_torch:   bool  = False  # prefer torch tensors when available
    # Signal layer this SM module operates at.  Determines where in the
    # causal chain it sits:
    #   "performer"  — receives driver outputs (keyed by item_slot), owns
    #                  instrument states, emits per-instrument signals
    #   "room"       — receives instrument outputs, emits mic stream(s)
    # Empty string means unspecified (legacy / backward-compat).
    signal_layer:   str   = ""
    # Transient — not serialized
    _sm_state:     dict  = field(default_factory=dict)  # {item: {var: scalar}}
    _sm_out_cache: dict  = field(default_factory=dict)  # {node_key: complex128 array}
    _sm_log_text:  str   = ""                           # captured plugin log/output
    _sm_aux_state: dict  = field(default_factory=dict)  # plugin-owned transient caches/state

    _MODULE_TYPES = ["lfo", "passthrough", "pitch_quantizer", "interaural", "state_machine"]
    _LFO_SHAPES   = ["Sine", "Triangle", "Sawtooth", "Square"]
    _INTERP_MODES = ["discrete", "portamento", "slew", "slew2", "spline", "legato"]

    @classmethod
    def knobs(cls) -> list:
        _PQ = ("module_type", "pitch_quantizer")
        return [
            KnobSpec("module_type",  "Type",   "choice", "lfo",  0, 3, 1, "",
                     cls._MODULE_TYPES, False, "Module", "", "", True),
            # LFO
            KnobSpec("rate_hz",      "Rate",   "float",  1.0,  0.01, 50.0,      0, "Hz",  [],
                     True,  "LFO", ".3f", "", False, ("module_type", "lfo")),
            KnobSpec("shape",        "Shape",  "choice", "Sine", 0,  3,   1, "",
                     cls._LFO_SHAPES, False, "LFO", "", "", False, ("module_type", "lfo")),
            KnobSpec("phase_offset", "Phase",  "float",  0.0, -math.pi, math.pi, 0, "rad", [],
                     False, "LFO", ".3f", "", False, ("module_type", "lfo")),
            KnobSpec("depth",        "Depth",  "float",  1.0,  0.0,  4.0,  0, "",  [],
                     False, "LFO", ".3f", "", False, ("module_type", "lfo")),
            # PitchQuantizer
            KnobSpec("interpolation_mode", "Interp",   "choice", "discrete", 0, 5, 1, "",
                     cls._INTERP_MODES, False, "PitchQuantizer", "", "", True, _PQ),
            KnobSpec("portamento_time",    "Glide",    "float",  0.05, 0.0,   4.0,  0, "s",  [],
                     True,  "PitchQuantizer", ".3f", "", False, _PQ),
            KnobSpec("slew_rate",          "Slew rate","float",  100.0, 1.0, 1000.0, 0, "st/s", [],
                     False, "PitchQuantizer", ".1f", "", False, _PQ),
            KnobSpec("slew2_accel",        "Accel",    "float",  200.0, 1.0, 5000.0, 0, "st/s²", [],
                     False, "PitchQuantizer", ".1f", "", False, _PQ),
            KnobSpec("spline_tension",     "Tension",  "float",  0.5,  0.0,  2.0,  0, "",  [],
                     False, "PitchQuantizer", ".2f", "", False, _PQ),
            # Interaural — ch1 (always active)
            KnobSpec("iau_azimuth",    "Az ch1",   "float",  0.0, -1.0, 1.0, 0, "", [], False,
                     "Interaural ch1", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_elevation",  "El ch1",   "float",  0.0, -1.0, 1.0, 0, "", [], False,
                     "Interaural ch1", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_distance",   "Dist ch1", "float",  0.0,  0.0, 1.0, 0, "", [], False,
                     "Interaural ch1", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_width",      "Width ch1","float",  0.0,  0.0, 1.0, 0, "", [], False,
                     "Interaural ch1", ".3f", "", False, ("module_type", "interaural")),
            # Interaural — ch2 (active when ch2 input edges exist)
            KnobSpec("iau_azimuth_ch2",  "Az ch2",   "float",  0.0, -1.0, 1.0, 0, "", [], False,
                     "Interaural ch2", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_elevation_ch2","El ch2",   "float",  0.0, -1.0, 1.0, 0, "", [], False,
                     "Interaural ch2", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_distance_ch2", "Dist ch2", "float",  0.0,  0.0, 1.0, 0, "", [], False,
                     "Interaural ch2", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_width_ch2",    "Width ch2","float",  0.0,  0.0, 1.0, 0, "", [], False,
                     "Interaural ch2", ".3f", "", False, ("module_type", "interaural")),
        ]

    def ch1_key(self) -> str:
        return f"{self.key}_ch1"

    def ch2_key(self) -> str:
        return f"{self.key}_ch2"

    def lfo_ch_key(self, i: int) -> str:
        return f"{self.key}_lfoch{i}"

    def sm_out_key(self, item: str, var: str) -> str:
        return f"{self.key}_sm_{item}_{var}"

    def sm_out_keys(self) -> list:
        return [self.sm_out_key(item, var)
                for item in self.sm_items for var in self.sm_vars]

    @staticmethod
    def default_lfo_channel() -> dict:
        return {"scale": 0.0, "amplitude": 1.0, "rate_hz": 1.0,
                "tension": 1.0, "phase_offset": 0.0, "shape": "Sine",
                "resample": 1, "slew_order": 1, "slew": 0.0}

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "module_type": self.module_type, "muted": self.muted,
                "color": self.color, "rate_hz": self.rate_hz,
                "shape": self.shape, "phase_offset": self.phase_offset,
                "depth": self.depth,
                "quantizer_scale_degrees": list(self.quantizer_scale_degrees),
                "interpolation_mode":      self.interpolation_mode,
                "portamento_time":         self.portamento_time,
                "slew_rate":               self.slew_rate,
                "slew2_accel":             self.slew2_accel,
                "spline_tension":          self.spline_tension,
                "lfo_channels":            list(self.lfo_channels),
                "iau_azimuth":             self.iau_azimuth,
                "iau_elevation":           self.iau_elevation,
                "iau_distance":            self.iau_distance,
                "iau_width":               self.iau_width,
                "iau_azimuth_ch2":         self.iau_azimuth_ch2,
                "iau_elevation_ch2":       self.iau_elevation_ch2,
                "iau_distance_ch2":        self.iau_distance_ch2,
                "iau_width_ch2":           self.iau_width_ch2,
                "sm_plugin":               self.sm_plugin,
                "sm_n_items":              self.sm_n_items,
                "sm_items":                list(self.sm_items),
                "sm_vars":                 list(self.sm_vars),
                "sm_bundle_ports":         [dict(p) for p in self.sm_bundle_ports],
                "sm_state_vars":           list(self.sm_state_vars),
                "sm_params":               dict(self.sm_params),
                "sm_use_torch":            self.sm_use_torch,
                "signal_layer":            self.signal_layer}

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyticModule":
        o = cls.__new__(cls)
        o.key          = d.get("key",         uuid.uuid4().hex[:8])
        o.label        = d.get("label",        "Module")
        o.module_type  = d.get("module_type",  "lfo")
        if o.module_type not in cls._MODULE_TYPES:
            o.module_type = "lfo"
        o.muted        = bool(d.get("muted",   False))
        o.color        = d.get("color",        [200, 140, 220])
        o.rate_hz      = float(d.get("rate_hz",      1.0))
        o.shape        = d.get("shape",        "Sine")
        o.phase_offset = float(d.get("phase_offset", 0.0))
        o.depth        = float(d.get("depth",        1.0))
        o.quantizer_scale_degrees = list(d.get("quantizer_scale_degrees", []))
        o.interpolation_mode      = d.get("interpolation_mode", "discrete")
        if o.interpolation_mode not in cls._INTERP_MODES:
            o.interpolation_mode = "discrete"
        o.portamento_time    = float(d.get("portamento_time", 0.05))
        o.slew_rate          = float(d.get("slew_rate",       100.0))
        o.slew2_accel        = float(d.get("slew2_accel",     200.0))
        o.spline_tension     = float(d.get("spline_tension",  0.5))
        o.lfo_channels       = [dict(c) for c in d.get("lfo_channels", [])]
        o.iau_azimuth        = float(d.get("iau_azimuth",        0.0))
        o.iau_elevation      = float(d.get("iau_elevation",      0.0))
        o.iau_distance       = float(d.get("iau_distance",       0.0))
        o.iau_width          = float(d.get("iau_width",          0.0))
        o.iau_azimuth_ch2    = float(d.get("iau_azimuth_ch2",    0.0))
        o.iau_elevation_ch2  = float(d.get("iau_elevation_ch2",  0.0))
        o.iau_distance_ch2   = float(d.get("iau_distance_ch2",   0.0))
        o.iau_width_ch2      = float(d.get("iau_width_ch2",      0.0))
        o.sm_plugin          = str(d.get("sm_plugin",    ""))
        o.sm_n_items         = int(d.get("sm_n_items",   1))
        o.sm_items           = list(d.get("sm_items",    []))
        o.sm_vars            = list(d.get("sm_vars",     []))
        o.sm_bundle_ports    = [dict(p) for p in d.get("sm_bundle_ports", [])]
        o.sm_state_vars      = list(d.get("sm_state_vars", d.get("sm_vars", [])))
        o.sm_params          = dict(d.get("sm_params",   {}))
        o.sm_use_torch       = bool(d.get("sm_use_torch", False))
        if o.module_type == "state_machine" and o.sm_plugin:
            _plug = _load_sm_plugin(o.sm_plugin)
            if _plug is not None:
                if not o.sm_vars:
                    o.sm_vars = _sm_plugin_output_vars(_plug)
                if not o.sm_state_vars:
                    o.sm_state_vars = _sm_plugin_state_vars(_plug)
                if not o.sm_items:
                    o.sm_items = _sm_plugin_item_names(_plug, o.sm_n_items)
                _defs = _sm_plugin_default_params(_plug)
                o.sm_params = {**_defs, **o.sm_params}
        o._sm_state          = {}
        o._sm_out_cache      = {}
        o._sm_log_text       = ""
        o._sm_aux_state      = {}
        o.signal_layer       = str(d.get("signal_layer", ""))
        return o


def _module_param_attrs(module_type: str) -> list:
    """Modulatable attr names for a module_type, derived from AnalyticModule.knobs()."""
    return [
        k.name for k in AnalyticModule.knobs()
        if k.visible_when == ("module_type", module_type)
    ]


def _knob_label_map(knobs: list) -> dict[str, str]:
    return {k.name: k.label for k in knobs}


def _routing_edge_attr_specs(patch: "AnalyticPatch") -> list[tuple[str, str]]:
    """Return per-edge pseudo-attrs for routing-grid knobs with readable labels."""
    specs: list[tuple[str, str]] = []
    node_keys = _patch_node_keys(patch)
    for src_key in node_keys:
        src_lbl = _routing_node_label(src_key, patch)
        for dst_key in node_keys:
            dst_lbl = _routing_node_label(dst_key, patch)
            specs.append((f"routing.mix::{src_key}::{dst_key}",
                          f"Mix / {src_lbl} -> {dst_lbl}"))
            specs.append((f"routing.angle::{src_key}::{dst_key}",
                          f"Angle / {src_lbl} -> {dst_lbl}"))
            specs.append((f"routing.delay::{src_key}::{dst_key}",
                          f"Delay / {src_lbl} -> {dst_lbl}"))
    return specs


def _parse_routing_edge_attr(attr: str) -> "tuple[str, str, str] | None":
    prefix, sep, rest = attr.partition("::")
    if not sep or prefix not in {"routing.mix", "routing.angle", "routing.delay"}:
        return None
    src_key, sep2, dst_key = rest.partition("::")
    if not sep2 or not src_key or not dst_key:
        return None
    kind = prefix.split(".", 1)[1]
    return kind, src_key, dst_key


def _param_target_node_specs(patch: "AnalyticPatch") -> list[tuple[str, str]]:
    """Return selectable ParamNode target nodes with readable labels."""
    specs: list[tuple[str, str]] = []
    specs.extend((v.key, v.label) for v in patch.voices)
    specs.extend((l.key, f"~ {l.label}") for l in patch.lfos)
    for mod in patch.modules:
        specs.append((mod.key, f"\u2B21 {mod.label}"))
        if mod.module_type == "interaural":
            specs.append((mod.ch1_key(), f"\u2B21 {mod.label} ch1"))
            specs.append((mod.ch2_key(), f"\u2B21 {mod.label} ch2"))
    for mix in patch.mixers:
        specs.append((mix.key, f"\u2261 {mix.label}"))
    return specs


def _param_target_attr_specs(patch: "AnalyticPatch", node_key: str) -> list[tuple[str, str]]:
    """Return (raw_attr, display_label) choices for a ParamNode target node."""
    if not node_key:
        return []
    voice = next((v for v in patch.voices if v.key == node_key), None)
    if voice is not None:
        voice_labels = {
            "freq_hz": "Frequency",
            "amplitude": "Amplitude",
            "semitone_offset": "Semitone Offset",
            "chirp.f_delta_start": "Chirp Start",
            "chirp.f_delta_end": "Chirp End",
            "adsr.attack": "Attack",
            "adsr.decay": "Decay",
            "adsr.sustain": "Sustain",
            "adsr.release": "Release",
            "fm.depth_hz": "FM Depth Hz",
            "fm.depth_amp": "FM Depth Amp",
            "am.depth_hz": "AM Depth Hz",
            "am.depth_amp": "AM Depth Amp",
            "harmonic_brightness": "Harmonic Brightness",
            "harmonic_warp_strength": "Harmonic Warp",
            "harmonic_count": "Harmonic Count",
            "granular.grain_density_hz": "Grain Density",
            "granular.grain_duration_s": "Grain Duration",
            "granular.grain_scatter": "Grain Scatter",
            "granular.grain_pitch_scatter": "Grain Pitch Scatter",
            "granular.grain_manifold_mix": "Grain Manifold Mix",
            "granular.grain_amplitude_jitter": "Grain Amp Jitter",
        }
        return [(attr, voice_labels.get(attr, attr)) for attr in _VOICE_PARAM_ATTRS]
    lfo = next((l for l in patch.lfos if l.key == node_key), None)
    if lfo is not None:
        labels = _knob_label_map(LFODefinition.knobs())
        return [(k.name, labels.get(k.name, k.name)) for k in LFODefinition.knobs()
                if k.dtype in ("float", "int", "bool", "choice")]
    mod = next((m for m in patch.modules
                if m.key == node_key or
                (m.module_type == "interaural" and node_key in (m.ch1_key(), m.ch2_key()))), None)
    if mod is not None:
        labels = _knob_label_map(AnalyticModule.knobs())
        attrs = _module_param_attrs(mod.module_type)
        return [(attr, labels.get(attr, attr)) for attr in attrs]
    mix = next((m for m in patch.mixers if m.key == node_key), None)
    if mix is not None:
        return _routing_edge_attr_specs(patch)
    return []


_VOICE_ROLE_PRESETS: dict = {
    # "signal": default — no overrides needed; leave as-is
    "air": {
        # Noise-like: no defined pitch tracking, high inharmonicity via FM,
        # slow attack, looped sustain body
        "harmonic_mode": "inharmonic",
        "env_attack_s":   0.08,
        "env_decay_s":    0.2,
        "env_sustain":    0.85,
        "env_release_s":  0.5,
        "loop_enabled":   True,
        "loop_start":     0.15,
        "loop_end":       0.75,
        "fm_index":       0.8,
    },
    "transient": {
        # Percussive: instant attack, very fast decay, chirp sweep at onset,
        # high initial FM index for clang character
        "harmonic_mode": "harmonic",
        "env_attack_s":   0.001,
        "env_decay_s":    0.06,
        "env_sustain":    0.0,
        "env_release_s":  0.08,
        "loop_enabled":   False,
        "chirp.chirp_type":    "exponential",
        "chirp.f_delta_start": 220.0,   # Hz above nominal, swept to 0 at attack end
        "chirp.f_delta_end":   0.0,
        "chirp.tau":           0.015,   # sweep time constant in seconds
        "fm_index":       3.0,
    },
    "body": {
        # Full resonant core: medium attack, long sustain, looped
        "harmonic_mode": "harmonic",
        "env_attack_s":   0.04,
        "env_decay_s":    0.12,
        "env_sustain":    0.9,
        "env_release_s":  0.8,
        "loop_enabled":   True,
        "loop_start":     0.10,
        "loop_end":       0.85,
        "fm_index":       0.25,
    },
}


def _apply_voice_role_preset(voice: "AnalyticVoice", role: str) -> None:
    """Apply parameter overrides for the given voice_role to *voice* in-place."""
    overrides = _VOICE_ROLE_PRESETS.get(role)
    if overrides is None:
        return
    for attr, val in overrides.items():
        if "." in attr:
            parts = attr.split(".", 1)
            sub = getattr(voice, parts[0], None)
            if sub is not None:
                setattr(sub, parts[1], val)
        else:
            if hasattr(voice, attr):
                setattr(voice, attr, val)


def _ensure_granular(voice: "AnalyticVoice") -> "object":
    """Return voice.granular, creating a default GrainPopulationSpec if needed."""
    if voice.granular is None and _HAS_GRANULAR:
        voice.granular = _GrainPopulationSpec(center_frequency_hz=voice.freq_hz)
    return voice.granular


def _resample_audio(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample a 1-D float array from src_sr to dst_sr using polyphase FIR.
    Returns the same dtype as the input.  No-ops when src_sr == dst_sr."""
    if src_sr == dst_sr or len(arr) == 0:
        return arr
    g = math.gcd(int(dst_sr), int(src_sr))
    up   = int(dst_sr) // g
    down = int(src_sr) // g
    return _scipy_resample_poly(arr, up, down).astype(arr.dtype)


# Well-known virtual routing node keys for the patch's note-domain signals.
# These are constant-Hz DC sources that carry pitch information (real part = Hz)
# rather than audio — do NOT auto-route to the mix bus.
_PATCH_VIRTUAL_KEYS: tuple = ("__patch_tonic__", "__patch_seq__")
_ROUTER_UI_PREFIX: str = "__router__:"


def _system_output_keys(patch: "AnalyticPatch") -> list[str]:
    return patch.system_audio.output_keys() if getattr(patch, "system_audio", None) else []


def _system_input_keys(patch: "AnalyticPatch") -> list[str]:
    return patch.system_audio.input_keys() if getattr(patch, "system_audio", None) else []


def _router_ui_key(router_key: str) -> str:
    return f"{_ROUTER_UI_PREFIX}{router_key}"


def _router_key_from_ui_key(ui_key: str) -> str:
    return str(ui_key)[len(_ROUTER_UI_PREFIX):] if str(ui_key).startswith(_ROUTER_UI_PREFIX) else ""


def _router_instance_for_active_key(
    patch: "AnalyticPatch",
    active_key: str,
) -> "RouterInstance | None":
    router_key = _router_key_from_ui_key(active_key)
    if not router_key:
        return None
    return next((r for r in getattr(patch, "routers", []) if r.key == router_key), None)


def _active_routing_graph_and_instance(
    patch: "AnalyticPatch",
    active_key: str = "",
) -> "tuple[RoutingGraph, RouterInstance | None]":
    router = _router_instance_for_active_key(patch, active_key)
    if router is not None:
        return router.graph, router
    return patch.routing, None


def _filtered_router_edges(
    graph: RoutingGraph,
    *,
    router_key: str = "",
    router_type: str = "",
) -> list[RoutingEdge]:
    edges = list(graph.edges)
    if router_key:
        edges = [e for e in edges if e.router_key in {"", router_key}]
    if router_type:
        edges = [
            e for e in edges
            if graph.is_source_in_router(e.src_key, router_type)
            and graph.is_sink_in_router(e.dst_key, router_type)
        ]
    return edges


def _routing_grid_node_keys(
    patch: "AnalyticPatch",
    active_key: str = "",
) -> list[str]:
    graph, router = _active_routing_graph_and_instance(patch, active_key)
    if router is None:
        return _patch_node_keys(patch)

    edges = _filtered_router_edges(
        graph,
        router_key=router.key,
        router_type=router.router_type,
    )
    referenced: set[str] = set()
    for e in edges:
        referenced.add(e.src_key)
        referenced.add(e.dst_key)

    canonical = _patch_node_keys(patch)
    router_type = str(getattr(router, "router_type", "") or "")
    graph_nodes = set(graph.node_keys())
    result: list[str] = []
    for key in canonical:
        if key not in graph_nodes and key not in referenced:
            continue
        if not router_type:
            if key in referenced or key in graph_nodes:
                result.append(key)
            continue
        if (
            key in referenced
            or graph.is_source_in_router(key, router_type)
            or graph.is_sink_in_router(key, router_type)
        ):
            result.append(key)
    extras = [
        key for key in graph.node_keys()
        if key not in result and (
            key in referenced
            or not router_type
            or graph.is_source_in_router(key, router_type)
            or graph.is_sink_in_router(key, router_type)
        )
    ]
    result.extend(extras)
    return result


def _runtime_performer_keys(patch: "AnalyticPatch") -> list[str]:
    keys: list[str] = []
    for pt in getattr(patch, "parts", []):
        for ch in getattr(pt, "chairs", []):
            for pf in getattr(ch, "performers", []):
                if getattr(pf, "key", ""):
                    keys.append(pf.key)
    return keys


def _sanitize_system_io_edges(g: RoutingGraph, patch: "AnalyticPatch") -> None:
    sys_in = set(_system_input_keys(patch))
    sys_out = set(_system_output_keys(patch))
    if not sys_in and not sys_out:
        return
    g.edges = [
        e for e in g.edges
        if e.src_key not in sys_out and e.dst_key not in sys_in
    ]

def _patch_node_keys(
    patch: "AnalyticPatch",
    include_params:   bool = True,
    include_controls: bool = True,
    include_performers: bool = False,
) -> list:
    """Canonical ordered node key list:
    virtual-patch-nodes -> voices -> LFOs -> module signal nodes -> controls -> mixers -> param_nodes.

    The two virtual patch nodes (__patch_tonic__, __patch_seq__) carry Hz
    pitch-domain signals (not audio) and are excluded from auto-mix routing.
    """
    keys = list(_PATCH_VIRTUAL_KEYS)
    keys += _system_input_keys(patch)
    if include_performers:
        keys += _runtime_performer_keys(patch)
    keys += [v.key for v in patch.voices]
    keys += [l.key for l in patch.lfos]
    for m in patch.modules:
        if m.module_type == "interaural":
            keys.append(m.ch1_key())
            keys.append(m.ch2_key())
        elif m.module_type == "lfo" and m.lfo_channels:
            keys.append(m.key)
            for i in range(len(m.lfo_channels)):
                keys.append(m.lfo_ch_key(i))
        elif m.module_type == "state_machine":
            keys.append(m.key)
            for k in m.sm_out_keys():
                keys.append(k)
        else:
            keys.append(m.key)
    if include_controls:
        for cs in patch.controls:
            keys += [sl.key for sl in cs.sliders]
    keys += [m.key for m in patch.mixers]
    keys += _system_output_keys(patch)
    if include_params:
        keys += [pn.key for pn in patch.param_nodes]
    return keys


def _routing_node_label(key: str, patch: "AnalyticPatch") -> str:
    if key == "__patch_tonic__":
        return f"Tonic ({_hz_to_note_name(patch.seq_tonic_hz, patch.tuning)})"
    if key == "__patch_seq__":
        return "Seq.Pitch"
    for i, sys_key in enumerate(_system_input_keys(patch)):
        if sys_key == key:
            return f"System In {i + 1}"
    for i, sys_key in enumerate(_system_output_keys(patch)):
        if sys_key == key:
            return f"System Out {i + 1}"
    if key == "__mix__":
        return "Mix"
    for m in patch.mixers:
        if m.key == key:
            return m.label
    for v in patch.voices:
        if v.key == key:
            return v.label
    for l in patch.lfos:
        if l.key == key:
            return l.label
    for mod in patch.modules:
        if mod.key == key:
            return mod.label
        if mod.module_type == "interaural":
            if mod.ch1_key() == key:
                return f"{mod.label} ch1"
            if mod.ch2_key() == key:
                return f"{mod.label} ch2"
        elif mod.module_type == "lfo" and mod.lfo_channels:
            for i in range(len(mod.lfo_channels)):
                if mod.lfo_ch_key(i) == key:
                    return f"{mod.label} ch{i}"
        elif mod.module_type == "state_machine":
            for item in mod.sm_items:
                for var in mod.sm_vars:
                    if mod.sm_out_key(item, var) == key:
                        return f"{mod.label}.{item}.{var}"
    for cs in patch.controls:
        for sl in cs.sliders:
            if sl.key == key:
                return f"{cs.label}/{sl.label}"
    for pn in patch.param_nodes:
        if pn.key == key:
            return pn.label
    return key[:4]


def _routing_node_color(key: str, patch: "AnalyticPatch") -> tuple:
    if key == "__patch_tonic__":
        return (80, 180, 255)    # blue  -- tonal centre
    if key == "__patch_seq__":
        return (80, 220, 160)    # teal  -- note-plan pitch stream
    if key in _system_input_keys(patch):
        return (70, 170, 170)
    if key in _system_output_keys(patch):
        return (220, 170, 90)
    if key == "__mix__":
        return (200, 200, 100)
    for m in patch.mixers:
        if m.key == key:
            return tuple(m.color[:3])
    for v in patch.voices:
        if v.key == key:
            return tuple(v.color[:3])
    for l in patch.lfos:
        if l.key == key:
            return tuple(l.color[:3])
    for mod in patch.modules:
        if mod.key == key:
            return tuple(mod.color[:3])
        if mod.module_type == "interaural" and key in (mod.ch1_key(), mod.ch2_key()):
            return tuple(mod.color[:3])
        if mod.module_type == "lfo" and mod.lfo_channels:
            for i in range(len(mod.lfo_channels)):
                if mod.lfo_ch_key(i) == key:
                    return tuple(mod.color[:3])
        if mod.module_type == "state_machine" and key in mod.sm_out_keys():
            return tuple(mod.color[:3])
    for cs in patch.controls:
        for sl in cs.sliders:
            if sl.key == key:
                return tuple(cs.color[:3])
    for pn in patch.param_nodes:
        if pn.key == key:
            return tuple(pn.color[:3])
    return (120, 120, 120)

_ART_LABELS: dict = {0: "", 1: "S", 2: "L", 3: "D"}   # step cell overlay text
_ART_GATE:   dict = {0: 1.0, 1: 0.28, 2: 1.15, 3: None}  # None = drone (hold to next onset)
_ART_CHOICES: list = [0, 1, 2, 3]   # normal / staccato / legato / drone


# ---------------------------------------------------------------------------
# GridViewLayer — unified grid-cell view abstraction
# ---------------------------------------------------------------------------
# Every sub-module that overlays the step grid (rhythm on/off, dynamics
# accent, improv eligibility, …) instantiates one of these with its own
# callbacks for colour, label, and click behaviour.  The shared methods
# _render_grid_layer() / _handle_grid_layer_click() on NavPanel do the
# actual rendering and hit-testing, so all grids share identical layout,
# tree-awareness, and interaction code.

class GridViewLayer:
    """Configuration for one visual/interactive layer on the rhythm grid."""

    __slots__ = (
        "name",
        "read_fn",           # (leaf, step_i, extra) -> value
        "bg_fn",             # (value, is_beat, depth, group) -> (r,g,b)
        "brd_fn",            # (value, is_beat, depth, group) -> (r,g,b)
        "label_fn",          # (value) -> (text, (r,g,b)) | None   (left-aligned)
        "label_right_fn",    # (value) -> (text, (r,g,b)) | None   (right-aligned)
        "click_fn",          # (leaf, step_i, extra) -> None   (left click)
        "right_click_fn",    # (cell, lx, ly, pat, div) -> dict | None  (context menu)
        "show_groups",       # bool — render group dots + merge bars
        "show_depth",        # bool — render depth tick marks
    )

    def __init__(
        self,
        name: str,
        *,
        read_fn,
        bg_fn,
        brd_fn,
        label_fn=None,
        label_right_fn=None,
        click_fn=None,
        right_click_fn=None,
        show_groups: bool = False,
        show_depth: bool = False,
    ):
        self.name            = name
        self.read_fn         = read_fn
        self.bg_fn           = bg_fn
        self.brd_fn          = brd_fn
        self.label_fn        = label_fn
        self.label_right_fn  = label_right_fn
        self.click_fn        = click_fn
        self.right_click_fn  = right_click_fn
        self.show_groups     = show_groups
        self.show_depth      = show_depth


@dataclass
class RhythmPattern:
    """Per-bar on/off step grid with per-step velocity and articulation.

    Two representations coexist:
    - Flat (legacy): steps / vel / art lists indexed by integer step.
    - Tree:          beat_nodes (BeatTree) stores the same data with optional
                     per-cell subdivision.  When beat_nodes is not None it is
                     the authoritative representation; the flat lists serve as a
                     projection cache for systems that haven't been updated yet.

    Each pattern also owns independent layer trees for accent and improv.
    These share the same [0,1) bar spine and warp but subdivide independently.

    Use get_tree(div) to obtain a live BeatTree regardless of which mode is
    active.  Use ensure_size(n) as before — it is safe to call on either mode.
    """
    name:  str  = "Pat"
    steps: list = field(default_factory=lambda: [False] * 16)
    vel:   list = field(default_factory=lambda: [1.0] * 16)
    art:   list = field(default_factory=lambda: [0] * 16)   # 0=normal 1=staccato 2=legato 3=drone
    # Optional tree representation — None means flat mode
    beat_nodes: "BeatTree | None" = field(default=None, compare=False, repr=False)
    # Independent layer trees (created on demand, same spine, own subdivisions)
    accent_tree: "BeatTree | None" = field(default=None, compare=False, repr=False)
    improv_tree: "BeatTree | None" = field(default=None, compare=False, repr=False)

    # ------------------------------------------------------------------
    # Tree access
    # ------------------------------------------------------------------

    def get_tree(self, div: int = 16) -> "BeatTree":
        """Return the live BeatTree, building from flat if not yet promoted."""
        if self.beat_nodes is not None:
            return self.beat_nodes
        self.ensure_size(div)
        self.beat_nodes = BeatTree.from_flat(self.steps, self.vel, self.art, home_div=div)
        return self.beat_nodes

    def get_accent_tree(self, div: int = 16) -> "BeatTree":
        """Return the accent layer tree, creating a uniform one on first call.

        Accent uses ``vel`` on each leaf as the accent level (0.0 – 2.0).
        """
        if self.accent_tree is not None:
            return self.accent_tree
        self.accent_tree = BeatTree.new_uniform(div, default_on=True, default_vel=1.0)
        return self.accent_tree

    def get_improv_tree(self, div: int = 16) -> "BeatTree":
        """Return the improv-eligibility layer tree, creating uniform on first call.

        Improv uses ``on`` on each leaf as eligibility (True = eligible).
        """
        if self.improv_tree is not None:
            return self.improv_tree
        self.improv_tree = BeatTree.new_uniform(div, default_on=False, default_vel=1.0)
        return self.improv_tree

    def snap_accent_to_rhythm(self) -> None:
        """Restructure accent tree to match the rhythm tree's subdivisions."""
        if self.beat_nodes is None or self.accent_tree is None:
            return
        self.accent_tree.snap_structure_from(self.beat_nodes)

    def snap_improv_to_rhythm(self) -> None:
        """Restructure improv tree to match the rhythm tree's subdivisions."""
        if self.beat_nodes is None or self.improv_tree is None:
            return
        self.improv_tree.snap_structure_from(self.beat_nodes)

    def sync_flat_from_tree(self, div: int | None = None) -> None:
        """Project the tree back onto the flat arrays (for legacy consumers)."""
        if self.beat_nodes is None:
            return
        n = div if div is not None else self.beat_nodes.home_div
        self.ensure_size(n)
        self.steps = self.beat_nodes.steps_array(n)
        self.vel   = self.beat_nodes.vel_array(n)
        self.art   = self.beat_nodes.art_array(n)

    def is_tree_mode(self) -> bool:
        return self.beat_nodes is not None

    # ------------------------------------------------------------------
    # Legacy flat API (unchanged behavior)
    # ------------------------------------------------------------------

    def ensure_size(self, n: int) -> None:
        """Grow steps/vel/art lists to at least n slots."""
        while len(self.steps) < n:
            self.steps.append(False)
        while len(self.vel) < n:
            self.vel.append(1.0)
        while len(self.art) < n:
            self.art.append(0)

    def to_dict(self) -> dict:
        d = {"name": self.name, "steps": list(self.steps),
             "vel": list(self.vel), "art": list(self.art)}
        if self.beat_nodes is not None:
            d["beat_nodes"] = self.beat_nodes.to_dict()
        if self.accent_tree is not None:
            d["accent_tree"] = self.accent_tree.to_dict()
        if self.improv_tree is not None:
            d["improv_tree"] = self.improv_tree.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RhythmPattern":
        rp       = cls()
        rp.name  = d.get("name", "Pat")
        rp.steps = [bool(x) for x in d.get("steps", [])]
        rp.vel   = [float(x) for x in d.get("vel", [])]
        rp.art   = [int(x) for x in d.get("art", [])]
        if "beat_nodes" in d:
            rp.beat_nodes = BeatTree.from_dict(d["beat_nodes"])
        if "accent_tree" in d:
            rp.accent_tree = BeatTree.from_dict(d["accent_tree"])
        if "improv_tree" in d:
            rp.improv_tree = BeatTree.from_dict(d["improv_tree"])
        return rp


@dataclass
class ResolvedNote:
    """Persistent piano-roll note derived from the union score solve."""
    note_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    voice_key: str = ""
    voice_label: str = ""
    layer_key: str = "all"
    start_time: float = 0.0
    duration_s: float = 0.25
    fundamental_hz: float = 440.0
    velocity: float = 1.0
    locked: bool = False
    is_rest: bool = False

    def to_dict(self) -> dict:
        return {
            "note_id": self.note_id,
            "voice_key": self.voice_key,
            "voice_label": self.voice_label,
            "layer_key": self.layer_key,
            "start_time": self.start_time,
            "duration_s": self.duration_s,
            "fundamental_hz": self.fundamental_hz,
            "velocity": self.velocity,
            "locked": self.locked,
            "is_rest": self.is_rest,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResolvedNote":
        n = cls()
        n.note_id = str(d.get("note_id", n.note_id))
        n.voice_key = str(d.get("voice_key", ""))
        n.voice_label = str(d.get("voice_label", ""))
        n.layer_key = str(d.get("layer_key", "all"))
        n.start_time = float(d.get("start_time", 0.0))
        n.duration_s = float(d.get("duration_s", 0.25))
        n.fundamental_hz = float(d.get("fundamental_hz", 440.0))
        n.velocity = float(d.get("velocity", 1.0))
        n.locked = bool(d.get("locked", False))
        n.is_rest = bool(d.get("is_rest", False))
        return n


@dataclass
class RhythmPage:
    """Complete rhythm configuration for one register-part (e.g. 'bass', 'mid').

    The 'all' page is the global default; per-register pages override it for
    voices that self-declare a matching register.  Any field left at its
    default inherits from the 'all' page at schedule-build time.
    """
    # Core grid
    rhythm_enabled:   bool  = False
    rhythm_division:  int   = 16
    rhythm_patterns:  list  = field(default_factory=lambda: [RhythmPattern(name="Pat 1")])
    rhythm_phrase:    list  = field(default_factory=lambda: [0])
    rhythm_active_pat: int  = 0
    rhythm_swing:     float = 0.0
    rhythm_pocket:    float = 0.0
    rhythm_gate:      float = 0.5
    rhythm_prog_bars: int   = 1
    rhythm_fit_mode:  str   = "drop"
    meter_numerator:  float = 0.0   # 0 = inherit patch meter
    meter_denominator: float = 0.0  # 0 = inherit patch meter
    stress_pattern:   list  = field(default_factory=list)  # e.g. [3, 2]
    # Warp interpolation mode for the beat-tree engine ("linear" | "cosine")
    warp_interpolator: str  = "linear"
    # How the fractional part of meter_numerator is handled:
    #   "warp" — absorbed into the warp curve weighted by stretch (invisible)
    #   "grid" — shown as a visible fractional beat cell in the grid
    frac_beat_mode: str = "warp"
    # Per-module pattern binding: role → pattern index
    # "dynamics" / "improv" / "cadence" → int index into rhythm_patterns
    module_pats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rhythm_enabled":    self.rhythm_enabled,
            "rhythm_division":   self.rhythm_division,
            "rhythm_patterns":   [rp.to_dict() for rp in self.rhythm_patterns],
            "rhythm_phrase":     list(self.rhythm_phrase),
            "rhythm_active_pat": self.rhythm_active_pat,
            "rhythm_swing":      self.rhythm_swing,
            "rhythm_pocket":     self.rhythm_pocket,
            "rhythm_gate":       self.rhythm_gate,
            "rhythm_prog_bars":  self.rhythm_prog_bars,
            "rhythm_fit_mode":   self.rhythm_fit_mode,
            "meter_numerator":   self.meter_numerator,
            "meter_denominator": self.meter_denominator,
            "stress_pattern":    list(self.stress_pattern),
            "warp_interpolator": self.warp_interpolator,
            "frac_beat_mode":   self.frac_beat_mode,
            "module_pats":       dict(self.module_pats),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RhythmPage":
        pg = cls()
        pg.rhythm_enabled    = bool(d.get("rhythm_enabled",   False))
        pg.rhythm_division   = int(d.get("rhythm_division",   16))
        pg.rhythm_patterns   = [RhythmPattern.from_dict(x)
                                 for x in d.get("rhythm_patterns", [])] \
                               or [RhythmPattern(name="Pat 1")]
        pg.rhythm_phrase     = list(d.get("rhythm_phrase",    [0]))
        pg.rhythm_active_pat = int(d.get("rhythm_active_pat", 0))
        pg.rhythm_swing      = float(d.get("rhythm_swing",    0.0))
        pg.rhythm_pocket     = float(d.get("rhythm_pocket",   0.0))
        pg.rhythm_gate       = float(d.get("rhythm_gate",     0.5))
        pg.rhythm_prog_bars  = int(d.get("rhythm_prog_bars",  1))
        pg.rhythm_fit_mode   = str(d.get("rhythm_fit_mode",   "drop"))
        pg.meter_numerator   = max(0.0, float(d.get("meter_numerator", 0.0)))
        pg.meter_denominator = max(0.0, float(d.get("meter_denominator", 0.0)))
        pg.stress_pattern    = [int(max(1, int(x))) for x in d.get("stress_pattern", []) if int(x) > 0]
        pg.warp_interpolator = str(d.get("warp_interpolator", "linear"))
        pg.frac_beat_mode    = str(d.get("frac_beat_mode", "warp"))
        if pg.frac_beat_mode not in ("warp", "grid"):
            pg.frac_beat_mode = "warp"
        pg.module_pats       = dict(d.get("module_pats", {}))
        return pg

    @classmethod
    def from_patch_fields(cls, d: dict) -> "RhythmPage":
        """Build a RhythmPage from a legacy flat-field patch dict."""
        return cls.from_dict(d)


@dataclass
class AnalyticPatch:
    name:       str   = "untitled"
    system_audio: SystemAudioDevice = field(default_factory=SystemAudioDevice)
    voices:     list  = field(default_factory=list)
    lfos:       list  = field(default_factory=list)
    modules:    list  = field(default_factory=list)   # list[AnalyticModule]
    controls:   list  = field(default_factory=list)   # list[ControlSurface]
    mixers:     list  = field(default_factory=lambda: [AnalyticMixer()])  # list[AnalyticMixer]
    param_nodes: list = field(default_factory=list)                        # list[ParamNode]
    duration:   float = 2.0
    tuning:     "GlobalTuning" = field(default_factory=GlobalTuning)
    preview_sr: int   = 48000
    # Sequence / demo
    seq_scale:           str   = "pentatonic_minor"
    seq_bpm:             float = 120.0
    seq_pattern_idx:     int   = 1     # index into _SEQ_PATTERN_PRESETS
    seq_legato:          float = 0.85
    seq_portamento_s:    float = 0.0   # glide time in seconds
    seq_rubato_shape:    str   = "off"
    seq_rubato_scope:    str   = "bar"
    seq_rubato_amount:   float = 0.0
    seq_octave_span:     int   = 2
    seq_tonic_hz:        float = 440.0 # tonal center / scale root (key being performed)
    meter_numerator:     float = 4.0
    meter_denominator:   float = 4.0
    _seq_note_hz:        float = 0.0  # transient: current note Hz for __patch_seq__ (0 = use seq_tonic_hz)
    seq_bass_octave:     int   = -1   # octave offset for bass-role voices
    seq_root_octave:     int   = -2   # octave offset for root-role voices (pedal)
    seq_stab_octave:     int   =  1   # octave offset for stab-role voices
    seq_chord_prog:      str   = "I_IV_V_I"
    seq_repeats:         int   = 2
    seq_custom_semitones: str  = ""    # e.g. "0,2,4,5,7,9,11" – overrides named scale
    # Projection / output
    projection_mode:        str   = "mono"  # "mono"|"stereo_quadrature"|"stereo_ms"|"lissajous"
    projection_rotation_hz: float = 0.0    # rotates analytic projection plane at this rate
    normalize_output:       bool  = True   # peak-normalize the mix before projection
    performer_phase_mode:   str   = "coherent"  # "coherent" | "individual"
    # Signal routing graph (analytic, pre-projection) — legacy default voice router.
    # New code should use patch.routers[i].graph for named router instances.
    routing: RoutingGraph = field(default_factory=RoutingGraph)
    # Multi-router deployment.  Each RouterInstance owns its own RoutingGraph
    # and is typed by MIXER_LAYERS ("voice_router", "instrument", "master").
    # Empty list means only the legacy `routing` graph is active.
    routers: list = field(default_factory=list)   # list[RouterInstance]
    # UI-only state — not serialized.  When set, only this voice key produces audio.
    solo_key: str | None = None
    resolved_notes: list = field(default_factory=list)  # list[ResolvedNote]
    # Param-series cache — not serialized.  Holds last frame's extracted param
    # node series so _synthesize_patch can apply param overrides in a single
    # solve rather than two.  One-buffer latency; reset on patch load/clear.
    _param_series_cache: dict = field(default_factory=dict)
    # Runtime-only arrangement solve metrics for future placement/allocation UI.
    _arrangement_metrics: dict = field(default_factory=dict)
    # Placement solver output: list of Part objects derived from arrangement metrics.
    parts: list = field(default_factory=list)   # list[Part]
    # Patch-level placement-owned resonator deployment config.
    placement_resonator: PlacementResonatorConfig = field(default_factory=PlacementResonatorConfig)
    # Rhythm programmer
    rhythm_enabled:    bool  = False
    rhythm_division:   int   = 16   # steps per bar: 4 8 12 16 24 32
    rhythm_patterns:   list  = field(default_factory=lambda: [RhythmPattern(name="Pat 1")])
    rhythm_phrase:     list  = field(default_factory=lambda: [0])  # pattern index per bar
    rhythm_active_pat: int   = 0
    rhythm_swing:      float = 0.0  # 0=straight … 0.67=full-triplet push on upbeats
    rhythm_pocket:     float = 0.0  # onset offset in beats (−0.5 … +0.5)
    rhythm_gate:       float = 0.5  # note duration as fraction of step
    rhythm_prog_bars:  int   = 1    # bars one progression cycle spans
    rhythm_fit_mode:   str   = "drop"  # "drop"=truncate to pulses; "extend"=add bars to fit
    stress_pattern:    list  = field(default_factory=list)  # default/all-page stress pattern
    frac_beat_mode:    str   = "warp"  # "warp" | "grid" — how fractional meter is handled
    # Progression probability transforms (0.0 = never, 1.0 = always)
    seq_probabilities: "SequenceProbabilities" = field(
        default_factory=lambda: SequenceProbabilities())
    # Velocity dynamics program (curve + accent grid)
    dynamics_program:  "DynamicsProgram" = field(
        default_factory=lambda: DynamicsProgram())
    dynamics_pages: dict = field(default_factory=dict)  # Dict[str, DynamicsProgram]
    # Stochastic ornament program (grace / chirp / echo)
    improv_program:    "ImprovProgram" = field(
        default_factory=lambda: ImprovProgram())
    improv_pages: dict = field(default_factory=dict)  # Dict[str, ImprovProgram]
    # How score pages combine for a voice. "union" = additive layers,
    # "specific" = only the most-specific layer renders.
    rhythm_layer_mode: str = "union"
    # Per-register rhythm pages.  "all" = global default (mirrors the flat fields
    # above for backward compat).  Additional keys: "bass", "mid", "high".
    rhythm_pages: dict = field(default_factory=dict)  # Dict[str, RhythmPage]
    # UI-only: which page is displayed in the rhythm section (not serialized).
    rhythm_active_page: str = "all"

    def _default_page(self) -> "RhythmPage":
        """Build a RhythmPage that mirrors the current flat rhythm fields."""
        pg = RhythmPage()
        pg.rhythm_enabled    = self.rhythm_enabled
        pg.rhythm_division   = self.rhythm_division
        pg.rhythm_patterns   = self.rhythm_patterns
        pg.rhythm_phrase     = self.rhythm_phrase
        pg.rhythm_active_pat = self.rhythm_active_pat
        pg.rhythm_swing      = self.rhythm_swing
        pg.rhythm_pocket     = self.rhythm_pocket
        pg.rhythm_gate       = self.rhythm_gate
        pg.rhythm_prog_bars  = self.rhythm_prog_bars
        pg.rhythm_fit_mode   = self.rhythm_fit_mode
        pg.meter_numerator   = self.meter_numerator
        pg.meter_denominator = self.meter_denominator
        pg.stress_pattern    = list(self.stress_pattern)
        pg.frac_beat_mode    = self.frac_beat_mode
        return pg

    def page_for(self, register: str) -> "RhythmPage":
        """Return a single RhythmPage for *register* (used by UI page selector).

        The 'all' page is always the live flat-field default.  Named pages are
        stored in rhythm_pages with arbitrary tag-set keys (e.g. 'bass',
        'bass+transient', 'stab+mid').  For UI display of a single named page,
        pass the key directly.
        """
        if register != "all" and register in self.rhythm_pages:
            return self.rhythm_pages[register]
        return self._default_page()

    def dynamics_for(self, key: str) -> "DynamicsProgram":
        if key != "all" and key in self.dynamics_pages:
            return self.dynamics_pages[key]
        return self.dynamics_program

    def improv_for(self, key: str) -> "ImprovProgram":
        if key != "all" and key in self.improv_pages:
            return self.improv_pages[key]
        return self.improv_program

    def beats_per_bar(self) -> float:
        den = max(0.125, float(self.meter_denominator))
        num = max(0.125, float(self.meter_numerator))
        return num * (4.0 / den)

    def page_meter(self, page: "RhythmPage | None" = None) -> tuple[float, float]:
        if page is None:
            return float(self.meter_numerator), float(self.meter_denominator)
        num = float(getattr(page, "meter_numerator", 0.0)) or float(self.meter_numerator)
        den = float(getattr(page, "meter_denominator", 0.0)) or float(self.meter_denominator)
        return max(0.125, num), max(0.125, den)

    def page_beats_per_bar(self, page: "RhythmPage | None" = None) -> float:
        num, den = self.page_meter(page)
        return num * (4.0 / den)

    def bar_duration_s(self) -> float:
        beat_s = 60.0 / max(float(self.seq_bpm), 1.0)
        return beat_s * self.beats_per_bar()

    def ensure_dynamics_page(self, key: str) -> "DynamicsProgram":
        if key == "all":
            return self.dynamics_program
        if key not in self.dynamics_pages:
            self.dynamics_pages[key] = DynamicsProgram.from_dict(self.dynamics_program.to_dict())
        return self.dynamics_pages[key]

    def ensure_improv_page(self, key: str) -> "ImprovProgram":
        if key == "all":
            return self.improv_program
        if key not in self.improv_pages:
            self.improv_pages[key] = ImprovProgram.from_dict(self.improv_program.to_dict())
        return self.improv_pages[key]

    def remove_page(self, register: str) -> None:
        """Delete a named register page from all three page dicts. No-op for 'all'."""
        if register == "all":
            return
        self.rhythm_pages.pop(register, None)
        self.dynamics_pages.pop(register, None)
        self.improv_pages.pop(register, None)
        # Mark patch dirty for UI refresh
        self._arrangement_metrics = {}

    def score_page_items_for_voice(self, voice: "AnalyticVoice") -> "list[tuple[str, RhythmPage]]":
        """Return all score layers that apply to *voice*, ordered least→most specific."""
        voice_tags = {
            getattr(voice, "register",   "all"),
            getattr(voice, "seq_role",   "melody"),
            getattr(voice, "voice_role", "signal"),
            getattr(voice, "key", ""),
        } - {"all", "free", ""}

        stack: list[tuple[int, str, RhythmPage]] = [(0, "all", self._default_page())]
        for key, pg in self.rhythm_pages.items():
            key_tags = {t.strip() for t in key.split("+")} - {"all", ""}
            if not key_tags:
                continue
            if key_tags.issubset(voice_tags):
                stack.append((len(key_tags), key, pg))

        stack.sort(key=lambda x: (x[0], x[1]))
        return [(key, pg) for _, key, pg in stack]

    def score_stack_for_voice(self, voice: "AnalyticVoice") -> "list[RhythmPage]":
        """Return all pages that apply to *voice*, ordered least→most specific.

        A page applies when all of its tag tokens (split on '+') are present in
        the voice's declared tag set (register ∪ seq_role ∪ voice_role).  The
        global default page ('all'/flat fields) is always the base of the stack.
        More-specific pages (more tokens in their key) override it per-step.

        Example: voice has register='bass', seq_role='stab', voice_role='transient'.
          - page key 'bass'              → applies (1 token ⊆ voice tags)
          - page key 'bass+transient'    → applies (2 tokens ⊆ voice tags)
          - page key 'stab+mid'          → does NOT apply ('mid' not in voice tags)
        """
        return [pg for _, pg in self.score_page_items_for_voice(voice)]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "system_audio": self.system_audio.to_dict(),
            "voices":      [v.to_dict() for v in self.voices],
            "lfos":        [l.to_dict() for l in self.lfos],
            "modules":     [m.to_dict() for m in self.modules],
            "controls":    [cs.to_dict() for cs in self.controls],
            "mixers":      [m.to_dict() for m in self.mixers],
            "param_nodes": [pn.to_dict() for pn in self.param_nodes],
            "duration":   self.duration,
            "tuning":     self.tuning.to_dict(),
            "preview_sr": self.preview_sr,
            "seq_scale":           self.seq_scale,
            "seq_bpm":             self.seq_bpm,
            "seq_pattern_idx":     self.seq_pattern_idx,
            "seq_legato":          self.seq_legato,
            "seq_portamento_s":    self.seq_portamento_s,
            "seq_rubato_shape":    self.seq_rubato_shape,
            "seq_rubato_scope":    self.seq_rubato_scope,
            "seq_rubato_amount":   self.seq_rubato_amount,
            "seq_octave_span":     self.seq_octave_span,
            "seq_tonic_hz":        self.seq_tonic_hz,
            "meter_numerator":     self.meter_numerator,
            "meter_denominator":   self.meter_denominator,
            "seq_bass_octave":     self.seq_bass_octave,
            "seq_root_octave":     self.seq_root_octave,
            "seq_stab_octave":     self.seq_stab_octave,
            "seq_chord_prog":      self.seq_chord_prog,
            "seq_repeats":         self.seq_repeats,
            "seq_custom_semitones": self.seq_custom_semitones,
            "projection_mode":        self.projection_mode,
            "projection_rotation_hz": self.projection_rotation_hz,
            "normalize_output":       self.normalize_output,
            "performer_phase_mode":   self.performer_phase_mode,
            "routing":                self.routing.to_dict(),
            "routers":                [r.to_dict() for r in self.routers],
            # Per-register rhythm pages (excludes "all" — derived from flat fields on load)
            "rhythm_pages":      {k: v.to_dict() for k, v in self.rhythm_pages.items()
                                  if k != "all"},
            # Legacy flat fields — written as mirrors of the "all" page so old
            # readers can still load the patch without the page system.
            "rhythm_enabled":    self.rhythm_enabled,
            "rhythm_division":   self.rhythm_division,
            "rhythm_patterns":   [rp.to_dict() for rp in self.rhythm_patterns],
            "rhythm_phrase":     list(self.rhythm_phrase),
            "rhythm_active_pat": self.rhythm_active_pat,
            "rhythm_swing":      self.rhythm_swing,
            "rhythm_pocket":     self.rhythm_pocket,
            "rhythm_gate":       self.rhythm_gate,
            "rhythm_prog_bars":  self.rhythm_prog_bars,
            "rhythm_fit_mode":   self.rhythm_fit_mode,
            "stress_pattern":    list(self.stress_pattern),
            "frac_beat_mode":   self.frac_beat_mode,
            "seq_probabilities": self.seq_probabilities.to_dict(),
            "dynamics_pages":    {k: v.to_dict() for k, v in self.dynamics_pages.items()
                                  if k != "all"},
            "dynamics_program":  self.dynamics_program.to_dict(),
            "improv_pages":      {k: v.to_dict() for k, v in self.improv_pages.items()
                                  if k != "all"},
            "improv_program":    self.improv_program.to_dict(),
            "rhythm_layer_mode": self.rhythm_layer_mode,
            "resolved_notes":    [n.to_dict() for n in self.resolved_notes],
            "placement_resonator": self.placement_resonator.to_dict(),
            # Placement solver output — persists player counts and hints
            "placement": [
                {
                    "key":          pt.key,
                    "label":        pt.label,
                    "register":     pt.register,
                    "seq_role":     pt.seq_role,
                    "voice_role":   pt.voice_role,
                    "voice_keys":   list(pt.voice_keys),
                    "player_count": pt.player_count,
                    "solver_hints": dict(pt.solver_hints),
                    "chairs": [
                        {
                            "key": ch.key,
                            "label": ch.label,
                            "part_key": ch.part_key,
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
                                    "chair_key": pf.chair_key,
                                    "chair_index": pf.chair_index,
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
                for pt in self.parts
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyticPatch":
        p = cls()
        p.name       = d.get("name", "untitled")
        p.system_audio = SystemAudioDevice.from_dict(d.get("system_audio", {}))
        p.voices      = [AnalyticVoice.from_dict(x) for x in d.get("voices", [])]
        p.lfos        = [LFODefinition.from_dict(x) for x in d.get("lfos", [])]
        p.modules     = [AnalyticModule.from_dict(x) for x in d.get("modules", [])]
        p.controls    = [ControlSurface.from_dict(x) for x in d.get("controls", [])]
        raw_mixers    = d.get("mixers", [])
        p.mixers      = [AnalyticMixer.from_dict(x) for x in raw_mixers] \
                        if raw_mixers else [AnalyticMixer()]
        p.param_nodes = [ParamNode.from_dict(x) for x in d.get("param_nodes", [])]
        p.duration   = float(d.get("duration", 2.0))
        _raw_tuning  = d.get("tuning")
        # Legacy patches that predate the tuning block seed from root_hz if present.
        _legacy_root = float(d.get("root_hz", 440.0))
        p.tuning     = GlobalTuning.from_dict(_raw_tuning) if _raw_tuning else GlobalTuning(root_hz=_legacy_root)
        p.preview_sr = int(d.get("preview_sr", 48000))
        p.seq_scale           = d.get("seq_scale", "pentatonic_minor")
        p.seq_bpm             = float(d.get("seq_bpm", 120.0))
        p.seq_pattern_idx     = int(d.get("seq_pattern_idx", 1))
        p.seq_legato          = float(d.get("seq_legato", 0.85))
        p.seq_portamento_s    = float(d.get("seq_portamento_s", 0.0))
        p.seq_rubato_shape    = str(d.get("seq_rubato_shape", "off"))
        if p.seq_rubato_shape not in _SEQ_RUBATO_SHAPES:
            p.seq_rubato_shape = "off"
        p.seq_rubato_scope    = str(d.get("seq_rubato_scope", "bar"))
        if p.seq_rubato_scope not in _SEQ_RUBATO_SCOPES:
            p.seq_rubato_scope = "bar"
        p.seq_rubato_amount   = max(0.0, min(0.95, float(d.get("seq_rubato_amount", 0.0))))
        p.seq_octave_span     = int(d.get("seq_octave_span", 2))
        p.seq_tonic_hz        = float(d.get("seq_tonic_hz", p.tuning.root_hz))
        p.meter_numerator     = max(0.125, float(d.get("meter_numerator", 4.0)))
        p.meter_denominator   = max(0.125, float(d.get("meter_denominator", 4.0)))
        p.seq_bass_octave     = int(d.get("seq_bass_octave", -1))
        p.seq_root_octave     = int(d.get("seq_root_octave", -2))
        p.seq_stab_octave     = int(d.get("seq_stab_octave",  1))
        p.seq_chord_prog      = d.get("seq_chord_prog", "I_IV_V_I")
        p.seq_repeats         = int(d.get("seq_repeats", 2))
        p.seq_custom_semitones = d.get("seq_custom_semitones", "")
        p.projection_mode        = d.get("projection_mode", "mono")
        p.projection_rotation_hz = float(d.get("projection_rotation_hz", 0.0))
        p.normalize_output       = bool(d.get("normalize_output", True))
        p.performer_phase_mode   = str(d.get("performer_phase_mode", "coherent"))
        if p.performer_phase_mode not in ("coherent", "individual"):
            p.performer_phase_mode = "coherent"
        _raw_rp             = d.get("rhythm_patterns", [])
        p.rhythm_enabled    = bool(d.get("rhythm_enabled", False))
        p.rhythm_division   = int(d.get("rhythm_division", 16))
        p.rhythm_patterns   = ([RhythmPattern.from_dict(x) for x in _raw_rp]
                               if _raw_rp else [RhythmPattern(name="Pat 1")])
        p.rhythm_phrase     = list(d.get("rhythm_phrase", [0]))
        p.rhythm_active_pat = int(d.get("rhythm_active_pat", 0))
        p.rhythm_swing      = float(d.get("rhythm_swing", 0.0))
        p.rhythm_pocket     = float(d.get("rhythm_pocket", 0.0))
        p.rhythm_gate       = float(d.get("rhythm_gate", 0.5))
        p.rhythm_prog_bars  = int(d.get("rhythm_prog_bars", 1))
        p.rhythm_fit_mode   = str(d.get("rhythm_fit_mode", "drop"))
        p.stress_pattern    = [int(max(1, int(x))) for x in d.get("stress_pattern", []) if int(x) > 0]
        p.frac_beat_mode    = str(d.get("frac_beat_mode", "warp"))
        if p.frac_beat_mode not in ("warp", "grid"):
            p.frac_beat_mode = "warp"
        # Probabilities — support old flat keys for backward compat with saved patches
        _raw_prob = d.get("seq_probabilities")
        if _raw_prob and isinstance(_raw_prob, dict):
            p.seq_probabilities = SequenceProbabilities.from_dict(_raw_prob)
        else:
            sp = SequenceProbabilities()
            sp.double_back = float(d.get("prob_double_back", 0.0))
            sp.subversion  = float(d.get("prob_subversion",  0.0))
            sp.chromatic   = float(d.get("prob_chromatic",   0.0))
            sp.modal       = float(d.get("prob_modal",       0.0))
            p.seq_probabilities = sp
        _raw_dyn = d.get("dynamics_program")
        if _raw_dyn and isinstance(_raw_dyn, dict):
            p.dynamics_program = DynamicsProgram.from_dict(_raw_dyn)
        raw_dyn_pages = d.get("dynamics_pages", {})
        p.dynamics_pages = {}
        for pg_key, pg_dict in raw_dyn_pages.items():
            if pg_key != "all" and isinstance(pg_dict, dict):
                p.dynamics_pages[pg_key] = DynamicsProgram.from_dict(pg_dict)
        _raw_imp = d.get("improv_program")
        if _raw_imp and isinstance(_raw_imp, dict):
            p.improv_program = ImprovProgram.from_dict(_raw_imp)
        raw_imp_pages = d.get("improv_pages", {})
        p.improv_pages = {}
        for pg_key, pg_dict in raw_imp_pages.items():
            if pg_key != "all" and isinstance(pg_dict, dict):
                p.improv_pages[pg_key] = ImprovProgram.from_dict(pg_dict)
        p.rhythm_layer_mode = str(d.get("rhythm_layer_mode", "union"))
        if p.rhythm_layer_mode not in {"union", "specific"}:
            p.rhythm_layer_mode = "union"
        p.placement_resonator = PlacementResonatorConfig.from_dict(d.get("placement_resonator", {}))
        p.resolved_notes = [ResolvedNote.from_dict(x)
                            for x in d.get("resolved_notes", [])
                            if isinstance(x, dict)]
        # Placement: restore Part list; drop stale keys not matching saved entry
        _placement_raw = d.get("placement", [])
        if _placement_raw:
            p.parts = [
                Part(
                    key=          pr.get("key", ""),
                    label=        pr.get("label", ""),
                    register=     pr.get("register", "all"),
                    seq_role=     pr.get("seq_role", ""),
                    voice_role=   pr.get("voice_role", ""),
                    voice_keys=   list(pr.get("voice_keys", [])),
                    player_count= int(pr.get("player_count", 1)),
                    solver_hints= dict(pr.get("solver_hints", {})),
                    chairs=[
                        Chair(
                            key=ch.get("key", ""),
                            label=ch.get("label", ""),
                            part_key=ch.get("part_key", pr.get("key", "")),
                            chair_index=int(ch.get("chair_index", 1)),
                            specificity_rank=int(ch.get("specificity_rank", 0)),
                            source_voice_keys=list(ch.get("source_voice_keys", [])),
                            source_layer_keys=list(ch.get("source_layer_keys", [])),
                            performer_count=max(1, int(ch.get("performer_count", 1))),
                            solver_hints=dict(ch.get("solver_hints", {})),
                            performers=[
                                PerformerPlacement(
                                    key=pf.get("key", ""),
                                    label=pf.get("label", ""),
                                    chair_key=pf.get("chair_key", ch.get("key", "")),
                                    chair_index=int(pf.get("chair_index", ch.get("chair_index", 1))),
                                    performer_index=int(pf.get("performer_index", 1)),
                                    source_voice_keys=list(pf.get("source_voice_keys", [])),
                                    source_layer_keys=list(pf.get("source_layer_keys", [])),
                                    assigned_note_keys=list(pf.get("assigned_note_keys", [])),
                                    body_type=str(pf.get("body_type", "direct") or "direct"),
                                    x=float(pf.get("x", 0.0)),
                                    y=float(pf.get("y", 0.0)),
                                    z=float(pf.get("z", 1.1)),
                                    face_x=float(pf.get("face_x", 0.0)),
                                    face_y=float(pf.get("face_y", -1.0)),
                                    face_z=float(pf.get("face_z", 0.0)),
                                    radius=float(pf.get("radius", 0.0)),
                                    angle_deg=float(pf.get("angle_deg", 0.0)),
                                    geometric_delay_ms=float(pf.get("geometric_delay_ms", 0.0)),
                                    humanization_ms=float(pf.get("humanization_ms", 0.0)),
                                    phase_offset_rad=float(pf.get("phase_offset_rad", 0.0)),
                                    gain_db=float(pf.get("gain_db", 0.0)),
                                    pan=float(pf.get("pan", 0.0)),
                                )
                                for pf in ch.get("performers", [])
                                if isinstance(pf, dict)
                            ],
                        )
                        for ch in pr.get("chairs", [])
                        if isinstance(ch, dict)
                    ],
                )
                for pr in _placement_raw
                if isinstance(pr, dict) and pr.get("key")
            ]
            if p.parts:
                _refresh_part_placement_layout(p)
        # Rhythm pages — load per-register pages only; "all" is always live from flat fields.
        raw_pages = d.get("rhythm_pages", {})
        p.rhythm_pages = {}
        for pg_key, pg_dict in raw_pages.items():
            if pg_key != "all" and isinstance(pg_dict, dict):
                p.rhythm_pages[pg_key] = RhythmPage.from_dict(pg_dict)
        p.rhythm_active_page = "all"   # always start on the default page
        if "routing" in d:
            p.routing = RoutingGraph.from_dict(d["routing"])
            # H1 fix: prune stale edges whose node keys no longer exist in the patch
            valid_keys = _patch_node_keys(p)
            p.routing.prune_keys(valid_keys)
        else:
            p.routing = RoutingGraph()
        # Multi-router instances (new; absent in legacy patches)
        p.routers = [RouterInstance.from_dict(r) for r in d.get("routers", [])]
        p._param_series_cache = {}   # always start fresh on load
        return p

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _PROJ    = ["mono", "stereo_quadrature", "stereo_ms", "lissajous"]
        _SR      = [8000.0, 22050.0, 44100.0, 48000.0, 96000.0, 192000.0]
        _PRESETS = list(GLOBAL_TUNING_PRESETS.keys())
        _TEMPS   = GlobalTuning._TEMPERAMENT_CHOICES
        _PHASE   = ["coherent", "individual"]
        return [
            # Global
            KnobSpec("preview_sr",           "Sample rate", "int",   48000, 8000, 192000, 0, "Hz", [], True,  "Global", ".0f", "", True),
            KnobSpec("duration",             "Duration",    "float", 2.0,   0.1,  60.0,   0, "s",  [], False, "Global", ".2f", "", True),
            # Projection
            KnobSpec("projection_mode",        "Proj mode",  "choice", "mono", 0, 3, 1, "", _PROJ, False, "Projection", ".0f", "", True),
            KnobSpec("projection_rotation_hz", "Rot Hz",     "float",  0.0, -200.0, 200.0, 0, "Hz", [], False, "Projection", ".2f"),
            KnobSpec("normalize_output",       "Normalize",  "bool",   True, 0, 1, 1, "",  [], False, "Projection", ""),
            KnobSpec("performer_phase_mode",   "Perf phase", "choice", "coherent", 0, 1, 1, "", _PHASE, False, "Projection"),
            # Tuning
            KnobSpec("tuning.preset_name",   "Preset",      "choice", "a440_12tet", 0,
                     max(0, len(_PRESETS) - 1), 1, "", _PRESETS, False, "Tuning", "", "", True),
            KnobSpec("tuning.root_hz",       "Tuning root", "float",  440.0, 20.0, 8000.0, 0, "Hz", [], True, "Tuning", ".2f"),
            KnobSpec("tuning.temperament",   "Temperament", "choice", "12tet", 0,
                     max(0, len(_TEMPS) - 1), 1, "", _TEMPS, False, "Tuning", "", "", True),
            KnobSpec("tuning.scale_name",    "Scale",       "str",    "chromatic", 0, 0, 0, "", [], False, "Tuning"),
        ]

    @staticmethod
    def default_patch() -> "AnalyticPatch":
        p = AnalyticPatch()
        p.name = "default"
        for i, (hz, col) in enumerate([
            (440.0, [100, 160, 255]),
            (880.0, [255, 130, 60]),
            (220.0, [120, 220, 120]),
        ]):
            voice = AnalyticVoice()
            voice.label   = f"V{i+1}"
            voice.freq_hz = hz
            voice.color   = col
            p.voices.append(voice)
        lfo = LFODefinition()
        lfo.label = "LFO1"
        lfo.rate_hz = 2.5
        p.lfos.append(lfo)
        return p


# ---------------------------------------------------------------------------
# Editor state
# ---------------------------------------------------------------------------

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


def _lfo_signal(lfo: LFODefinition, t: np.ndarray) -> np.ndarray:
    ph = 2.0 * np.pi * lfo.rate_hz * t + lfo.phase_offset
    if lfo.shape == "Sine":
        return lfo.depth * np.sin(ph)
    elif lfo.shape == "Triangle":
        return lfo.depth * (2.0 * np.abs(2.0 * (ph / (2 * np.pi) % 1.0) - 1.0) - 1.0)
    elif lfo.shape == "Sawtooth":
        return lfo.depth * (2.0 * (ph / (2 * np.pi) % 1.0) - 1.0)
    else:  # Square
        return lfo.depth * np.sign(np.sin(ph))


def _compute_envelope(voice: AnalyticVoice, n: int, duration: float) -> np.ndarray:
    if voice.piecewise_env is not None:
        t_ax = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        vals = voice.piecewise_env.curve.evaluate_normalized(t_ax).abs().to(torch.float64)
        return vals.detach().cpu().numpy().astype(np.float64, copy=False)
    knots = voice.active_knots()
    ts = np.array([k[0] * duration for k in knots], dtype=np.float64)
    vs = np.array([k[1]            for k in knots], dtype=np.float64)
    t_ax = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
    return np.interp(t_ax, ts, vs)


def _compute_chirp_deviation_series(
    voice: AnalyticVoice,
    n: int,
    duration: float,
    *,
    t_axis_s: "np.ndarray | None" = None,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if t_axis_s is None:
        t_axis_s = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
    else:
        t_axis_s = np.asarray(t_axis_s, dtype=np.float64)
    piecewise = getattr(voice, "piecewise_env", None)
    piecewise_chirp = getattr(piecewise, "chirp_curve", None) if piecewise is not None else None
    piecewise_delta = np.zeros(len(t_axis_s), dtype=np.float64)
    if piecewise_chirp is not None:
        if duration > 0.0:
            t_norm = np.clip(np.maximum(t_axis_s, 0.0) / duration, 0.0, 1.0)
        else:
            t_norm = np.zeros(len(t_axis_s), dtype=np.float64)
        t_tensor = torch.as_tensor(t_norm, dtype=torch.float64)
        piecewise_raw = piecewise_chirp.evaluate_normalized(t_tensor).real.clamp(0.0, 1.0)
        piecewise_delta = (
            piecewise_chirp.to_physical(piecewise_raw)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
    return piecewise_delta + _compute_knob_chirp_deviation_series(
        voice,
        n,
        duration,
        t_axis_s=t_axis_s,
    )


def _compute_knob_chirp_deviation_series(
    voice: AnalyticVoice,
    n: int,
    duration: float,
    *,
    t_axis_s: "np.ndarray | None" = None,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if t_axis_s is None:
        t_axis_s = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
    else:
        t_axis_s = np.asarray(t_axis_s, dtype=np.float64)
    chirp = getattr(voice, "chirp", None)
    if chirp is None:
        return np.zeros(len(t_axis_s), dtype=np.float64)
    ct = str(getattr(chirp, "chirp_type", "none") or "none")
    if ct == "linear":
        if len(t_axis_s) == 1:
            return np.array([float(getattr(chirp, "f_delta_start", 0.0))], dtype=np.float64)
        return np.linspace(
            float(getattr(chirp, "f_delta_start", 0.0)),
            float(getattr(chirp, "f_delta_end", 0.0)),
            len(t_axis_s),
            dtype=np.float64,
        )
    if ct == "exponential":
        tau = max(float(getattr(chirp, "tau", 0.5)), 1e-9)
        dec = np.exp(-np.maximum(t_axis_s, 0.0) / tau)
        return (
            float(getattr(chirp, "f_delta_start", 0.0)) * dec
            + float(getattr(chirp, "f_delta_end", 0.0)) * (1.0 - dec)
        ).astype(np.float64, copy=False)
    if ct == "power" and duration > 0.0:
        tau_n = (np.maximum(t_axis_s, 0.0) / duration) ** max(float(getattr(chirp, "chirp_power", 1.0)), 1e-3)
        return (
            float(getattr(chirp, "f_delta_start", 0.0)) * (1.0 - tau_n)
            + float(getattr(chirp, "f_delta_end", 0.0)) * tau_n
        ).astype(np.float64, copy=False)
    return np.zeros(len(t_axis_s), dtype=np.float64)


def _compute_chirp_frequency_series(voice: AnalyticVoice, n: int, duration: float) -> np.ndarray:
    return float(voice.freq_hz) + _compute_chirp_deviation_series(voice, n, duration)


def _sample_curve_from_series(
    values: np.ndarray,
    *,
    name: str,
    v_lo: float = 0.0,
    v_hi: float = 1.0,
    n_points: int = 24,
) -> ParametricCurve:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return _pc_default_blank(name)
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - lo) / (hi - lo)
    curve = ParametricCurve(name=name, v_lo=v_lo, v_hi=v_hi)
    idxs = np.linspace(0, arr.size - 1, max(2, n_points), dtype=int)
    seen: set[int] = set()
    for idx in idxs.tolist():
        if idx in seen:
            continue
        seen.add(idx)
        t = 0.0 if arr.size <= 1 else float(idx) / float(arr.size - 1)
        curve.add_point(t, float(np.clip(norm[idx], 0.0, 1.0)))
    return curve


def _load_detected_piecewise_envelope(path: str) -> PiecewiseVoiceEnvelope | None:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return None
    lower = path.lower()
    try:
        if lower.endswith(".json"):
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict) and raw.get("curve"):
                pw = PiecewiseVoiceEnvelope.from_dict(raw)
                if pw is not None:
                    pw.source_path = path
                    pw.detected_envelope_path = path
                    return pw
            if isinstance(raw, dict) and "curve" in raw and "rule_tree" in raw:
                curve = ParametricCurve.from_dict(dict(raw.get("curve", {})))
                chirp_curve = ParametricCurve.from_dict(dict(raw.get("chirp", {}))) if raw.get("chirp") else _pc_default_chirp(f"{curve.name}_chirp")
                rule_tree = EnvelopeRuleTree.from_dict(dict(raw.get("rule_tree", {}))) if raw.get("rule_tree") else EnvelopeRuleTree.default()
                return PiecewiseVoiceEnvelope(
                    curve=curve,
                    chirp_curve=chirp_curve,
                    signal_curve=_pc_default_blank(f"{curve.name}_signal"),
                    rule_tree=rule_tree,
                    source_path=path,
                    detected_envelope_path=path,
                )
            curve = ParametricCurve.from_dict(raw)
            return PiecewiseVoiceEnvelope(
                curve=curve,
                chirp_curve=_pc_default_chirp(f"{curve.name}_chirp"),
                signal_curve=_pc_default_blank(f"{curve.name}_signal"),
                rule_tree=EnvelopeRuleTree.default(),
                source_path=path,
                detected_envelope_path=path,
            )
        if lower.endswith(".npz"):
            analysis_dir = os.path.dirname(os.path.dirname(path)) if os.path.basename(path).startswith("fb_envelopes") else os.path.dirname(path)
            loaded = FilterBankDecomposition.load_envelopes(analysis_dir)
            if not loaded:
                return None
            mags, phases, _hops = loaded
            mag_stack = [np.asarray(m, dtype=np.float64).reshape(-1) for m in mags if m is not None]
            if not mag_stack:
                return None
            width = max(len(m) for m in mag_stack)
            t_dst = np.linspace(0.0, 1.0, width, endpoint=True, dtype=np.float64)
            resampled_mag = []
            for mag in mag_stack:
                t_src = np.linspace(0.0, 1.0, len(mag), endpoint=True, dtype=np.float64)
                resampled_mag.append(np.interp(t_dst, t_src, mag))
            avg_mag = np.mean(np.stack(resampled_mag, axis=0), axis=0)
            amp_curve = _sample_curve_from_series(avg_mag, name=os.path.splitext(os.path.basename(path))[0], n_points=32)
            chirp_curve = _pc_default_chirp(f"{amp_curve.name}_chirp")
            phase_stack = [np.asarray(p, dtype=np.float64).reshape(-1) for p in phases if p is not None]
            if phase_stack:
                resampled_phase = []
                for phase in phase_stack:
                    t_src = np.linspace(0.0, 1.0, len(phase), endpoint=True, dtype=np.float64)
                    resampled_phase.append(np.interp(t_dst, t_src, phase))
                avg_phase = np.mean(np.stack(resampled_phase, axis=0), axis=0)
                phase_delta = np.diff(np.unwrap(avg_phase), prepend=avg_phase[:1])
                chirp_curve = _sample_curve_from_series(phase_delta, name=f"{amp_curve.name}_chirp", v_lo=-200.0, v_hi=200.0, n_points=24)
            return PiecewiseVoiceEnvelope(
                curve=amp_curve,
                chirp_curve=chirp_curve,
                signal_curve=_pc_default_blank(f"{amp_curve.name}_signal"),
                rule_tree=EnvelopeRuleTree.default(),
                source_path=path,
                detected_envelope_path=path,
            )
    except Exception:
        return None
    return None


def _detected_envelope_artifact_paths(root_dir: str) -> list[str]:
    root_dir = os.path.abspath(root_dir)
    found: list[str] = []
    seen: set[str] = set()
    for cur_root, _dirs, files in os.walk(root_dir):
        if len(found) >= 64:
            break
        if "analysis_inventory.json" in files:
            inv_path = os.path.join(cur_root, "analysis_inventory.json")
            try:
                with open(inv_path, "r", encoding="utf-8") as fh:
                    inv = AnalysisInventory.from_dict(json.load(fh))
                for ds in inv.datasets:
                    for art in ds.artifacts:
                        if art.kind != "envelopes":
                            continue
                        art_path = art.path
                        if not os.path.isabs(art_path):
                            art_path = os.path.join(cur_root, art_path)
                        art_path = os.path.abspath(art_path)
                        if art_path not in seen and os.path.isfile(art_path):
                            seen.add(art_path)
                            found.append(art_path)
            except Exception:
                pass
        for fn in files:
            if not (fn.endswith(".json") or fn.endswith(".npz")):
                continue
            if fn.startswith("fb_envelopes") or fn.endswith("_envelope.json") or "piecewise" in fn.lower():
                fp = os.path.abspath(os.path.join(cur_root, fn))
                if fp not in seen:
                    seen.add(fp)
                    found.append(fp)
    return sorted(found)


def _inst_freq_from_csig(csig: np.ndarray, sr: float) -> np.ndarray:
    """Extract instantaneous frequency (Hz) from a complex analytic signal.

    Uses the discrete phase-derivative estimator:
        f_inst[n] = angle(z[n] * conj(z[n-1])) * sr / (2π)
    The first sample is assumed equal to the second to avoid an index-zero edge.
    """
    if len(csig) < 2:
        return np.zeros(len(csig), dtype=np.float64)
    phase_diff = np.angle(csig[1:] * np.conj(csig[:-1]))
    f = phase_diff * (sr / (2.0 * math.pi))
    return np.concatenate(([f[0]], f))


def _build_rhythm_schedule(
        p: "AnalyticPatch",
        beat_s: float,
        degrees: list,
        deg_pattern: list,
        min_dur_s: float = 1.0 / 48000,
        page: "RhythmPage | None" = None,
        dynamics_program: "DynamicsProgram | None" = None,
        improv_program: "ImprovProgram | None" = None,
) -> "NoteSchedule":
    """Convert rhythm program + scale degrees into a NoteSchedule.

    *page* selects which RhythmPage to drive the schedule.  When None,
    the patch's 'all' page (or flat legacy fields) is used.

    One *progression cycle* = one full walk through ``deg_pattern`` via a
    ``NoteStream`` that applies ``p.seq_probabilities`` transforms per onset.
    One *phrase cycle*      = ``rhythm_prog_bars`` bars of the rhythm phrase.

    Fit modes
    ---------
    "drop"   — Run exactly ``rhythm_prog_bars`` bars.  Onsets beyond the
               progression length become rests.
    "extend" — Add whole phrase-length chunks until onset count ≥ n_pat.
               Extra onsets at the tail are rests.

    Swing:   odd-indexed steps pushed back by ``rhythm_swing × step_s``.
    Pocket:  every onset shifted by ``rhythm_pocket × beat_s`` (pos = lay-back).
    Gate:    note duration = ``rhythm_gate × step_s`` × articulation multiplier.
    Drone:   gate extends to reach the next onset in the bar (two-pass).
    """
    # Resolve the page: prefer explicit arg, then voice register page, then legacy flat.
    pg: "RhythmPage" = page if page is not None else p.page_for("all")

    sched      = NoteSchedule()
    div        = max(1, pg.rhythm_division)
    bar_s      = _bar_duration_s_for_page(p, pg, beat_s)
    step_s     = bar_s / div
    base_gate  = step_s * pg.rhythm_gate
    phrase     = pg.rhythm_phrase if pg.rhythm_phrase else [0]
    pats       = pg.rhythm_patterns if pg.rhythm_patterns else [RhythmPattern(name="Pat 1")]
    n_pat      = len(deg_pattern)
    prog_bars  = max(1, pg.rhythm_prog_bars)
    fit_mode   = pg.rhythm_fit_mode
    probs      = getattr(p, "seq_probabilities", None) or SequenceProbabilities()
    _rng       = _random.Random()   # local instance — doesn't touch global state
    stream     = NoteStream(degrees, deg_pattern, probs, _rng)

    # ── Warp curve — built once from home-grid params, shared across all bars ─
    _meter_num, _ = p.page_meter(pg)
    _warp = build_warp_curve(
        home_div      = div,
        swing         = pg.rhythm_swing,
        pocket        = pg.rhythm_pocket,
        rubato_shape  = getattr(p, "seq_rubato_shape",  "off"),
        rubato_amount = getattr(p, "seq_rubato_amount", 0.0),
        meter_num     = _meter_num,
        beats_per_bar = p.beats_per_bar(),
        interpolator  = getattr(pg, "warp_interpolator", "linear"),
        frac_beat_mode = getattr(pg, "frac_beat_mode", "warp"),
    )

    def _onsets_in_bars(start_bar: int, end_bar: int) -> int:
        total = 0
        for b in range(start_bar, end_bar):
            slot  = b % len(phrase)
            pat_i = phrase[slot]
            pat   = pats[min(pat_i, len(pats) - 1)]
            if pat.is_tree_mode():
                total += sum(1 for lf in pat.get_tree(div).flat_leaves() if lf.on)
            else:
                pat.ensure_size(div)
                total += sum(1 for s in pat.steps[:div] if s)
        return total

    # Determine cycle length in bars for this render
    if fit_mode == "extend":
        cycle_bars = prog_bars
        phrase_len = max(1, len(phrase))
        while _onsets_in_bars(0, cycle_bars) < n_pat and cycle_bars < prog_bars + phrase_len * 64:
            cycle_bars += phrase_len
    else:  # "drop"
        cycle_bars = prog_bars

    # ── First pass: collect all onset times per bar for drone gate lookahead ──
    # onset_map[abs_bar][leaf_key] = t_final for active leaves.
    # leaf_key is step_i (flat) or node_id (tree) — used only for drone lookups.
    onset_map: dict = {}
    for rep in range(max(1, p.seq_repeats)):
        for bar_i in range(cycle_bars):
            slot  = bar_i % len(phrase)
            pat_i = phrase[slot]
            pat   = pats[min(pat_i, len(pats) - 1)]
            abs_bar = rep * cycle_bars + bar_i
            if pat.is_tree_mode():
                for leaf, t_onset, _ in iter_leaf_events(pat.get_tree(div), _warp, bar_s, abs_bar):
                    onset_map.setdefault(abs_bar, {})[leaf.node_id] = t_onset
            else:
                pat.ensure_size(div)
                for step_i in range(div):
                    if not pat.steps[step_i]:
                        continue
                    t_onset = abs_bar * bar_s + _warp.warp_to_seconds(
                        step_i / div, bar_s)
                    onset_map.setdefault(abs_bar, {})[step_i] = t_onset

    def _next_onset_t(abs_bar: int, leaf_key: object, t_cur: float) -> float:
        """Return t of next onset after *leaf_key* in *abs_bar*, for drone gate."""
        max_bar = abs_bar + cycle_bars
        for b in range(abs_bar, max_bar + 1):
            bar_onsets = onset_map.get(b, {})
            for k in sorted(bar_onsets, key=lambda k: bar_onsets[k]):
                if b == abs_bar and bar_onsets[k] <= t_cur:
                    continue
                return bar_onsets[k]
        return t_cur + bar_s   # fallback: one bar ahead

    # ── Second pass: build schedule with articulation-aware gate ─────────────
    for rep in range(max(1, p.seq_repeats)):
        stream.reset()   # each repeat restarts the progression identically
        for bar_i in range(cycle_bars):
            slot  = bar_i % len(phrase)
            pat_i = phrase[slot]
            pat   = pats[min(pat_i, len(pats) - 1)]
            abs_bar = rep * cycle_bars + bar_i

            if pat.is_tree_mode():
                # ── Tree path (grouped events merge adjacent same-group leaves) ─
                for leaf, t_onset, dur_s in iter_grouped_events(
                        pat.get_tree(div), _warp, bar_s, abs_bar):
                    hz = stream.next_hz()
                    if hz is None:
                        continue
                    art_mul = _ART_GATE.get(int(leaf.art), 1.0)
                    if leaf.group != 0:
                        # Grouped: duration already merged, use it directly
                        gate_s_i = max(min_dur_s, dur_s)
                    elif art_mul is None:
                        next_t   = _next_onset_t(abs_bar, leaf.node_id, t_onset)
                        gate_s_i = max(min_dur_s, next_t - t_onset)
                    else:
                        gate_s_i = max(min_dur_s, dur_s * pg.rhythm_gate * art_mul)
                    sched.add(NoteEvent(hz, t_onset, gate_s_i, velocity=float(leaf.vel)))
            else:
                # ── Flat (legacy) path ───────────────────────────────────────
                pat.ensure_size(div)
                for step_i in range(div):
                    if not pat.steps[step_i]:
                        continue
                    hz = stream.next_hz()
                    if hz is None:
                        continue
                    t_onset = abs_bar * bar_s + _warp.warp_to_seconds(step_i / div, bar_s)
                    vel     = float(pat.vel[step_i]) if step_i < len(pat.vel) else 1.0
                    art_val = int(pat.art[step_i]) if step_i < len(pat.art) else 0
                    art_mul = _ART_GATE.get(art_val, 1.0)
                    if art_mul is None:
                        next_t   = _next_onset_t(abs_bar, step_i, t_onset)
                        gate_s_i = max(min_dur_s, next_t - t_onset)
                    else:
                        gate_s_i = max(min_dur_s, step_s * pg.rhythm_gate * art_mul)
                    sched.add(NoteEvent(hz, t_onset, gate_s_i, velocity=vel))

    # ── Post-process: apply velocity dynamics (curve + accent grid) ──────────
    dyn_prog = dynamics_program if dynamics_program is not None else getattr(p, "dynamics_program", None)
    if dyn_prog is not None and _HAS_DYN_ENG:
        # Get accent tree from the active rhythm pattern (if available)
        _acc_tree = None
        _act_i = min(pg.rhythm_active_pat, max(0, len(pats) - 1))
        _act_pat = pats[_act_i] if pats else None
        if _act_pat is not None:
            try:
                _acc_tree = _act_pat.get_accent_tree(div)
            except Exception:
                pass
        apply_dynamics(sched.events, dyn_prog, beat_s, div, _rng,
                       beats_per_bar=p.beats_per_bar(),
                       accent_tree=_acc_tree)

    # ── Post-process: apply stochastic ornaments (grace / chirp / echo) ──────
    imp_prog = improv_program if improv_program is not None else getattr(p, "improv_program", None)
    if imp_prog is not None and _HAS_IMPROV_ENG and imp_prog.enabled:
        extra = apply_improv(
            sched.events,
            imp_prog,
            beat_s,
            div,
            pats,
            phrase,
            cycle_bars,
            _rng,
            beats_per_bar=p.beats_per_bar(),
        )
        if extra:
            sched.events.extend(extra)
            sched.events.sort(key=lambda e: e.start_time)

    return sched


def _build_score_schedule_for_voice(
        p: "AnalyticPatch",
        voice: "AnalyticVoice",
        beat_s: float,
        degrees: list,
        deg_pattern: list,
        min_dur_s: float = 1.0 / 48000,
) -> "NoteSchedule":
    """Build the rendered schedule for one voice under the patch's layer mode."""
    layers = p.score_page_items_for_voice(voice)
    if not layers:
        return NoteSchedule()

    if getattr(p, "rhythm_layer_mode", "union") == "specific":
        layer_key, layer_pg = layers[-1]
        sched = _build_rhythm_schedule(
            p, beat_s, degrees, deg_pattern,
            min_dur_s=min_dur_s,
            page=layer_pg,
            dynamics_program=p.dynamics_for(layer_key),
            improv_program=p.improv_for(layer_key),
        )
        for ev in sched.events:
            ev._layer_key = layer_key
        return sched

    merged = NoteSchedule()
    for layer_key, layer_pg in layers:
        layer_sched = _build_rhythm_schedule(
            p, beat_s, degrees, deg_pattern,
            min_dur_s=min_dur_s,
            page=layer_pg,
            dynamics_program=p.dynamics_for(layer_key),
            improv_program=p.improv_for(layer_key),
        )
        if layer_sched.events:
            for ev in layer_sched.events:
                ev._layer_key = layer_key
            merged.events.extend(layer_sched.events)
    merged.events.sort(key=lambda e: e.start_time)
    return merged


def _build_sequence_pitch_context(
        p: "AnalyticPatch",
) -> "tuple[float, list[float], list[int]] | None":
    """Return (beat_s, degrees, pattern) for the patch sequence settings."""
    if not _HAS_SEQ_ENG:
        return None
    source_voices = [v for v in p.voices if not v.muted]
    template = source_voices[0] if source_voices else (p.voices[0] if p.voices else None)
    if template is None:
        return None
    scale = p.seq_scale if p.seq_scale in MODAL_SCALES else "pentatonic_minor"
    beat_s = 60.0 / max(p.seq_bpm, 1.0)
    pattern = _SEQ_PATTERN_PRESETS[
        max(0, min(p.seq_pattern_idx, len(_SEQ_PATTERN_PRESETS) - 1))
    ][1]
    if p.seq_custom_semitones.strip():
        custom_semi = [float(s) for s in p.seq_custom_semitones.split(",") if s.strip()]
        degrees: list[float] = []
        for octave in range(p.seq_octave_span):
            for st in custom_semi:
                degrees.append(semitones_to_hz(p.seq_tonic_hz, st + 12 * octave))
    else:
        base_hz = template.freq_hz if template is not None else p.seq_tonic_hz
        degrees = scale_degrees_hz(base_hz, scale, octave_span=p.seq_octave_span)
    return beat_s, degrees, pattern


def _build_legacy_sequence_schedule(
        p: "AnalyticPatch",
        beat_s: float,
        degrees: list[float],
        pattern: list[int],
) -> "NoteSchedule":
    """Return the non-rhythm schedule path used by preview/export."""
    if p.seq_custom_semitones.strip():
        schedule = NoteSchedule()
        t = 0.0
        for _rep in range(p.seq_repeats):
            for deg_i in pattern:
                hz = degrees[deg_i % len(degrees)]
                note_dur = beat_s * 0.5 * p.seq_legato
                schedule.add(NoteEvent(hz, t, max(note_dur, 1.0 / p.preview_sr)))
                t += beat_s * 0.5
        return schedule

    source_voices = [v for v in p.voices if not v.muted]
    template = source_voices[0] if source_voices else (p.voices[0] if p.voices else None)
    root_hz = template.freq_hz if template is not None else p.seq_tonic_hz
    rule = ArpeggioRule(
        root_hz=root_hz,
        scale=p.seq_scale if p.seq_scale in MODAL_SCALES else "pentatonic_minor",
        pattern=pattern,
        rhythm_beats=[0.5],
        bpm=p.seq_bpm,
        legato_fraction=p.seq_legato,
        octave_span=p.seq_octave_span,
        repeats=p.seq_repeats,
    )
    return rule.generate()


def _sync_resolved_notes(
        p: "AnalyticPatch",
        preserve_locked: bool = True,
) -> list[ResolvedNote]:
    """Refresh patch.resolved_notes from the current union solve."""
    ctx = _build_sequence_pitch_context(p)
    if ctx is None:
        return p.resolved_notes
    beat_s, degrees, pattern = ctx
    locked_map: dict[str, ResolvedNote] = {}
    locked_notes: list[ResolvedNote] = []
    if preserve_locked:
        locked_notes = [n for n in p.resolved_notes if getattr(n, "locked", False)]
        locked_map = {n.note_id: n for n in locked_notes}

    def _masked_by_locked(candidate: ResolvedNote) -> bool:
        for locked in locked_notes:
            if locked.voice_key != candidate.voice_key:
                continue
            a0 = candidate.start_time
            a1 = candidate.start_time + candidate.duration_s
            b0 = locked.start_time
            b1 = locked.start_time + locked.duration_s
            if max(a0, b0) < min(a1, b1):
                return True
        return False

    notes: list[ResolvedNote] = []
    source_voices = [v for v in p.voices if not v.muted]
    for voice in source_voices:
        if p.rhythm_enabled:
            sched = _build_score_schedule_for_voice(
                p, voice, beat_s, degrees, pattern, min_dur_s=1.0 / p.preview_sr)
        else:
            sched = _build_legacy_sequence_schedule(p, beat_s, degrees, pattern)
        for i, ev in enumerate(sched.events):
            layer_key = str(getattr(ev, "_layer_key", "legacy"))
            note_id = (
                f"{voice.key}:{layer_key}:"
                f"{round(float(ev.start_time), 6)}:"
                f"{round(float(ev.duration_s), 6)}:"
                f"{round(float(_resolved_event_hz(p, voice, ev.fundamental_hz)), 4)}"
            )
            generated = ResolvedNote(
                note_id=note_id,
                voice_key=voice.key,
                voice_label=getattr(voice, "label", voice.key[:6]),
                layer_key=layer_key,
                start_time=float(ev.start_time),
                duration_s=float(ev.duration_s),
                fundamental_hz=float(_resolved_event_hz(p, voice, ev.fundamental_hz)),
                velocity=float(ev.velocity),
                locked=False,
            )
            if note_id in locked_map:
                keep = locked_map.pop(note_id)
                keep.voice_key = generated.voice_key
                keep.voice_label = generated.voice_label
                keep.layer_key = generated.layer_key
                keep.velocity = generated.velocity
                notes.append(keep)
            elif preserve_locked and _masked_by_locked(generated):
                continue
            else:
                notes.append(generated)
    if preserve_locked and locked_notes:
        locked_ids = {n.note_id for n in notes}
        for locked in locked_notes:
            if locked.note_id not in locked_ids:
                notes.append(locked)
    notes.sort(key=lambda n: (n.start_time, n.fundamental_hz, n.voice_key, n.layer_key))
    p.resolved_notes = notes
    return notes


_REGISTER_BAND_ORDER = {"bass": 0, "mid": 1, "high": 2, "all": 3}
_REGISTER_ROW_RADIUS = {"bass": 4.3, "mid": 6.0, "high": 7.8, "all": 5.2}

# ── Stage layout: dome-backed concert stage with realistic orchestral seating ──
#
# Real orchestral layout principles encoded here:
#   • Strings (signal / body voices) occupy the front arcs in register order:
#       - High melody = 1st violins (front stage-left)
#       - High non-melody = 2nd violins (front stage-right)
#       - Mid signal/body = violas (center-left, 2nd row) or cellos (center-right)
#       - Bass signal/body = cellos (front-right) or double basses (far right, risers)
#   • Woodwinds (air voices, or mid/high non-body non-transient alternates):
#       center rear, paired in two rows behind strings
#   • Brass (body or signal voices in bass/mid with stab or root roles):
#       right rear on risers, behind woodwinds
#   • Percussion (transient voices, or stab role in any register):
#       rear center-to-left, highest risers
#   • "all" register parts default to center stage (viola territory)
#   • Body voices sit behind their timbral kin (same section, deeper row)
#   • Air voices go to woodwind section regardless of register
#   • First chairs (slot_index 0) get inside seats closest to conductor
#
# Section format: (center_angle_deg, base_radius_m, platform_z_m, arc_span_per_chair_deg)
# Angle: 0° = front-center facing audience, +deg = stage-right (audience-left),
#         −deg = stage-left (audience-right)
# Radius: distance from conductor position; deeper rows = larger radius
# Platform z: height above stage floor (risers for rear sections)
# Arc span: degrees consumed per chair for natural spacing

_STAGE_SECTIONS: dict[str, tuple[float, float, float, float]] = {
    # ── Front row: strings ────────────────────────────────────────────────
    "violin_1":       (-25.0,  3.8, 1.05, 6.0),   # front stage-left arc
    "violin_2":       (+18.0,  3.8, 1.05, 6.0),   # front stage-right arc
    # ── Second row: mid strings ──────────────────────────────────────────
    "viola":          (-12.0,  5.2, 1.08, 7.0),   # center-left, slightly deeper
    "cello":          (+26.0,  5.0, 1.08, 7.0),   # center-right
    # ── Third row: low strings, high riser ──────────────────────────────
    "bass_str":       (+52.0,  6.8, 1.22, 9.0),   # far right, standing riser
    # ── Fourth row: woodwinds center ─────────────────────────────────────
    "woodwind_1":     ( -5.0,  6.6, 1.18, 8.0),   # center-left rear (flutes/oboes)
    "woodwind_2":     ( +8.0,  7.4, 1.24, 8.0),   # center-right, deeper (clarinets/bassoons)
    # ── Fifth row: brass ─────────────────────────────────────────────────
    "brass_1":        (+30.0,  8.2, 1.35, 10.0),  # right rear (horns)
    "brass_2":        (+48.0,  8.8, 1.42, 10.0),  # far right rear (trumpets/trombones)
    # ── Sixth row: percussion ────────────────────────────────────────────
    "percussion_1":   (-38.0,  9.2, 1.52, 12.0),  # left rear, high riser (timpani, mallet)
    "percussion_2":   (  0.0,  9.6, 1.58, 12.0),  # center rear, highest (cymbals, misc)
    # ── Specialty: soloist / harp / keyboard / body-resonance ────────────
    "soloist":        (  0.0,  2.6, 1.05, 8.0),   # center-front, beside conductor
    "harp":           (-55.0,  5.8, 1.10, 10.0),  # far stage-left, behind 1st violins
    "body_front":     ( -8.0,  4.6, 1.06, 7.0),   # body voices behind front strings
    "body_rear":      (+15.0,  7.0, 1.20, 8.0),   # body voices behind mid sections
}

def _stage_section_for_part(register: str, seq_role: str, slot_index: int,
                            voice_role: str = "signal") -> str:
    """Map a Part's classification to an orchestral stage section.

    Uses register, seq_role, voice_role, and slot_index (for alternation within
    the same classification) to produce realistic orchestral seating.
    """
    # ── Transient: soloists (melody/root) go center-front near conductor,
    #    stab/bass transients go to percussion rear ────────────────────────
    if voice_role == "transient":
        if seq_role in ("melody", "root"):
            return "soloist"            # featured performer, center-front
        return "percussion_1" if slot_index % 2 == 0 else "percussion_2"

    # ── Air voices → woodwinds regardless of register ────────────────────
    if voice_role == "air":
        return "woodwind_1" if slot_index % 2 == 0 else "woodwind_2"

    # ── Body voices → behind their timbral kin ───────────────────────────
    if voice_role == "body":
        if register == "bass":
            return "body_rear"      # behind cellos/basses
        return "body_front"         # behind front-row strings

    # ── Stab (plosive/accent) placement ──────────────────────────────────
    if seq_role == "stab":
        if register == "bass":
            return "percussion_1"   # bass stabs center-back (timpani territory)
        if register == "high":
            return "brass_2"        # high stabs out wide (trumpet snaps)
        return "brass_1"            # mid stabs with horns

    # ── High register: strings front, woodwinds behind ───────────────────
    if register == "high":
        if seq_role in ("melody", ""):
            return "violin_1" if slot_index % 2 == 0 else "violin_2"
        if seq_role == "root":
            return "violin_2" if slot_index % 2 == 0 else "woodwind_1"
        return "woodwind_1" if slot_index % 2 == 0 else "woodwind_2"

    # ── Mid register: violas / cellos / woodwinds ────────────────────────
    if register == "mid":
        if seq_role == "melody":
            return "viola" if slot_index % 2 == 0 else "cello"
        if seq_role == "root":
            return "cello" if slot_index % 2 == 0 else "woodwind_2"
        if seq_role == "bass":
            return "cello"
        return "viola" if slot_index % 2 == 0 else "woodwind_1"

    # ── Bass register: cellos / basses / brass ───────────────────────────
    if register == "bass":
        if seq_role == "melody":
            return "cello" if slot_index % 2 == 0 else "bass_str"
        if seq_role == "bass":
            return "bass_str" if slot_index % 2 == 0 else "cello"
        if seq_role == "root":
            return "brass_1" if slot_index % 2 == 0 else "bass_str"
        return "cello" if slot_index % 2 == 0 else "brass_1"

    # ── "all" register → center stage (viola/cello territory) ────────────
    return "viola" if slot_index % 2 == 0 else "cello"


def _ordinal_label(index: int) -> str:
    if 10 <= (index % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(index % 10, "th")
    return f"{index}{suffix}"


def _page_specificity_score(layer_keys: list[str]) -> int:
    max_tokens = 0
    total_tokens = 0
    for key in layer_keys:
        toks = [t.strip() for t in str(key).split("+") if t.strip() and t.strip() != "all"]
        total_tokens += len(toks)
        max_tokens = max(max_tokens, len(toks))
    return max_tokens * 100 + total_tokens


def _stable_rng(seed: str) -> _random.Random:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]
    return _random.Random(int(digest, 16))


def _rotate_list(values: list[str], offset: int) -> list[str]:
    if not values:
        return []
    offset %= len(values)
    return list(values[offset:]) + list(values[:offset])


def _voice_polyphony_capacity(voice: "AnalyticVoice") -> int:
    return max(1, int(getattr(voice, "polyphony_count", 1)))


def _voice_polyphony_mode(voice: "AnalyticVoice") -> str:
    mode = str(getattr(voice, "polyphony_mode", "sympathetic"))
    return mode if mode in ("sympathetic", "unsympathetic") else "sympathetic"


def _performer_phase_mode(patch: "AnalyticPatch") -> str:
    mode = str(getattr(patch, "performer_phase_mode", "coherent"))
    return mode if mode in ("coherent", "individual") else "coherent"


def _group_event_note_key(group_key: str, voice_key: str, event: object, layer_key: str) -> str:
    start = float(getattr(event, "start_time", 0.0))
    dur = float(getattr(event, "duration_s", 0.0))
    hz = float(getattr(event, "fundamental_hz", 0.0))
    return (
        f"{group_key}:{voice_key}:{layer_key}:"
        f"{round(start, 6)}:{round(dur, 6)}:{round(hz, 4)}"
    )


def _note_demands_for_group(group_key: str, events: list, voices: list["AnalyticVoice"]) -> list[dict]:
    demands: list[dict] = []
    for voice in voices:
        for event in events:
            layer_key = str(getattr(event, "_layer_key", "all"))
            demands.append({
                "note_key": _group_event_note_key(group_key, getattr(voice, "key", ""), event, layer_key),
                "voice_key": getattr(voice, "key", ""),
                "layer_key": layer_key,
                "start_time": float(getattr(event, "start_time", 0.0)),
                "end_time": float(getattr(event, "start_time", 0.0)) + float(getattr(event, "duration_s", 0.0)),
                "duration_s": float(getattr(event, "duration_s", 0.0)),
                "fundamental_hz": float(getattr(event, "fundamental_hz", 0.0)),
            })
    demands.sort(key=lambda d: (d["start_time"], -(d["end_time"] - d["start_time"]), d["voice_key"], d["note_key"]))
    return demands


def _pack_note_demands_by_voice(demands: list[dict], voices: list["AnalyticVoice"]) -> tuple[list[dict], dict]:
    voice_map = {getattr(v, "key", ""): v for v in voices}
    bins: list[dict] = []
    summary = {
        "required_performers": 0,
        "sympathetic_bins": 0,
        "unsympathetic_bins": 0,
    }
    for voice in voices:
        voice_key = getattr(voice, "key", "")
        voice_demands = [d for d in demands if d["voice_key"] == voice_key]
        if not voice_demands:
            continue
        mode = _voice_polyphony_mode(voice)
        capacity = _voice_polyphony_capacity(voice) if mode == "sympathetic" else 1
        voice_bins: list[dict] = []
        for demand in voice_demands:
            placed = False
            for bin_info in voice_bins:
                active_ends = [et for et in bin_info["active_ends"] if et > demand["start_time"]]
                bin_info["active_ends"] = active_ends
                if len(active_ends) < capacity:
                    bin_info["active_ends"].append(demand["end_time"])
                    bin_info["note_demands"].append(demand)
                    placed = True
                    break
            if not placed:
                voice_bins.append({
                    "voice_key": voice_key,
                    "polyphony_mode": mode,
                    "capacity": capacity,
                    "active_ends": [demand["end_time"]],
                    "note_demands": [demand],
                })
        summary["required_performers"] += len(voice_bins)
        if mode == "sympathetic":
            summary["sympathetic_bins"] += len(voice_bins)
        else:
            summary["unsympathetic_bins"] += len(voice_bins)
        bins.extend(voice_bins)
    return bins, summary


def _required_chairs_for_overlap(active_events: int, voices: list["AnalyticVoice"]) -> tuple[int, dict]:
    active_events = max(0, int(active_events))
    if active_events <= 0 or not voices:
        return 0, {
            "sympathetic_sources": 0,
            "unsympathetic_sources": 0,
            "sympathetic_capacity": 0,
            "unsympathetic_capacity": 0,
            "interference_units": 0.0,
        }

    sympathetic = [v for v in voices if _voice_polyphony_mode(v) == "sympathetic"]
    unsympathetic = [v for v in voices if _voice_polyphony_mode(v) == "unsympathetic"]

    sympathetic_sources = active_events * len(sympathetic)
    sympathetic_capacity = sum(_voice_polyphony_capacity(v) for v in sympathetic)
    sympathetic_chairs = (
        math.ceil(sympathetic_sources / max(1, sympathetic_capacity))
        if sympathetic_sources > 0 else 0
    )

    unsympathetic_sources = active_events * len(unsympathetic)
    unsympathetic_capacity = sum(_voice_polyphony_capacity(v) for v in unsympathetic)
    unsympathetic_chairs = sum(
        math.ceil(active_events / _voice_polyphony_capacity(v))
        for v in unsympathetic
    )

    interference_units = float(unsympathetic_sources) + (
        float(sympathetic_sources) / max(1.0, float(sympathetic_capacity))
    )
    return sympathetic_chairs + unsympathetic_chairs, {
        "sympathetic_sources": sympathetic_sources,
        "unsympathetic_sources": unsympathetic_sources,
        "sympathetic_capacity": sympathetic_capacity,
        "unsympathetic_capacity": unsympathetic_capacity,
        "interference_units": interference_units,
    }


def _schedule_overlap_metrics(events: list, voices: list["AnalyticVoice"]) -> dict:
    if not events:
        return {
            "peak_active_events": 0,
            "peak_active_sources": 0,
            "required_chairs": max(1, len(voices)) if voices else 1,
            "interference_peak": 0.0,
        }

    times = sorted({
        float(getattr(ev, "start_time", 0.0))
        for ev in events
    } | {
        float(getattr(ev, "start_time", 0.0)) + float(getattr(ev, "duration_s", 0.0))
        for ev in events
    })
    if len(times) < 2:
        required, detail = _required_chairs_for_overlap(len(events), voices)
        return {
            "peak_active_events": len(events),
            "peak_active_sources": len(events) * len(voices),
            "required_chairs": max(1, required),
            "interference_peak": float(detail["interference_units"]),
        }

    peak_active_events = 0
    peak_active_sources = 0
    peak_required_chairs = 1
    interference_peak = 0.0
    for t0, t1 in zip(times[:-1], times[1:]):
        if t1 <= t0:
            continue
        mid_t = 0.5 * (t0 + t1)
        active_events = [
            ev for ev in events
            if float(getattr(ev, "start_time", 0.0)) <= mid_t <
            float(getattr(ev, "start_time", 0.0)) + float(getattr(ev, "duration_s", 0.0))
        ]
        if not active_events:
            continue
        active_count = len(active_events)
        required, detail = _required_chairs_for_overlap(active_count, voices)
        peak_active_events = max(peak_active_events, active_count)
        peak_active_sources = max(peak_active_sources, active_count * len(voices))
        peak_required_chairs = max(peak_required_chairs, required)
        interference_peak = max(interference_peak, float(detail["interference_units"]))

    return {
        "peak_active_events": peak_active_events,
        "peak_active_sources": peak_active_sources,
        "required_chairs": max(1, peak_required_chairs),
        "interference_peak": interference_peak,
    }


def _refresh_part_placement_layout(patch: "AnalyticPatch") -> None:
    metrics = getattr(patch, "_arrangement_metrics", {}) or {}
    groups = {
        str(gm.get("group_key", "")): gm
        for gm in metrics.get("groups", [])
        if isinstance(gm, dict) and gm.get("group_key")
    }
    voice_map = {v.key: v for v in getattr(patch, "voices", [])}

    patch.parts.sort(key=lambda pt: (
        _REGISTER_BAND_ORDER.get(pt.register, 99),
        -int(pt.solver_hints.get("page_specificity", 0)),
        -int(pt.solver_hints.get("stack_depth", 0)),
        pt.label,
    ))

    _res_cfg = getattr(patch, "placement_resonator", None)
    _layout_mode = str(getattr(_res_cfg, "layout_mode", "auto") or "auto")

    by_register: dict[str, list[Part]] = {}
    for pt in patch.parts:
        by_register.setdefault(pt.register, []).append(pt)

    for register, reg_parts in by_register.items():
        slot_count = len(reg_parts)
        for slot_index, pt in enumerate(reg_parts):
            gm = groups.get(pt.key, {})
            voice_keys = list(pt.voice_keys)
            layer_keys = list(gm.get("layer_keys", pt.solver_hints.get("layer_keys", [])))
            packed_performers = list(gm.get("packed_performers", []))
            required_chairs = max(1, len(packed_performers) or int(pt.solver_hints.get("required_chairs", 1)))
            total_players = max(required_chairs, int(pt.player_count))
            pt.solver_hints["required_chairs"] = required_chairs
            pt.solver_hints["section_slot"] = slot_index
            pt.solver_hints["section_slot_count"] = slot_count
            pt.solver_hints["layer_keys"] = list(layer_keys)

            # ── Section geometry: stage rows vs. register semicircle ──────────
            if _layout_mode == "stage":
                section_key = _stage_section_for_part(register, pt.seq_role, slot_index,
                                                     voice_role=pt.voice_role)
                sec_angle, sec_radius, sec_z, sec_arc = _STAGE_SECTIONS.get(
                    section_key, (0.0, 5.0, 1.1, 8.0))
                part_span_deg = max(sec_arc, min(sec_arc * required_chairs, 60.0))
                section_center_angle = sec_angle
                radius = sec_radius
                pt.solver_hints["section_shape"] = "semicircle"
                pt.solver_hints["stage_section"] = section_key
                pt.solver_hints["stage_z"] = sec_z
            else:
                sec_z = 1.1
                pt.solver_hints["section_shape"] = "line" if register == "all" else "semicircle"
                section_center = slot_index - 0.5 * (slot_count - 1)
                radius = _REGISTER_ROW_RADIUS.get(register, _REGISTER_ROW_RADIUS["all"])
                section_center_angle = section_center * 24.0
                part_span_deg = max(12.0, min(70.0, 10.0 * required_chairs))
                if slot_count == 1:
                    part_span_deg = min(80.0, part_span_deg + 8.0)

            chair_base = total_players // required_chairs
            chair_extra = total_players % required_chairs
            chairs: list[Chair] = []
            for chair_idx in range(required_chairs):
                chair_num = chair_idx + 1
                performer_count = chair_base + (1 if chair_idx < chair_extra else 0)
                packed = (
                    packed_performers[chair_idx]
                    if chair_idx < len(packed_performers)
                    else {}
                )
                source_voice_keys = list(packed.get("source_voice_keys", [])) or _rotate_list(voice_keys, chair_idx)
                source_layer_keys = list(packed.get("source_layer_keys", [])) or _rotate_list(layer_keys, chair_idx)
                body_types = [
                    str(getattr(voice_map.get(vk), "body_type", "direct") or "direct")
                    for vk in source_voice_keys
                    if vk
                ]
                chair_body_type = body_types[0] if body_types and len(set(body_types)) == 1 else "direct"
                if pt.solver_hints["section_shape"] == "line":
                    section_center_l = slot_index - 0.5 * (slot_count - 1)
                    if required_chairs == 1:
                        chair_x = section_center_l * 1.6
                    else:
                        chair_x = section_center_l * 1.6 + (
                            (chair_idx / (required_chairs - 1)) - 0.5
                        ) * 2.2
                    chair_y = radius
                    chair_angle_deg = 0.0
                else:
                    chair_angle_deg = (
                        section_center_angle if required_chairs == 1 else
                        section_center_angle + ((chair_idx / (required_chairs - 1)) - 0.5) * part_span_deg
                    )
                    ang = math.radians(chair_angle_deg)
                    chair_x = radius * math.sin(ang)
                    chair_y = radius * math.cos(ang)

                chair = Chair(
                    key=f"{pt.key}:chair:{chair_num}",
                    label=f"{_ordinal_label(chair_num)} chair",
                    part_key=pt.key,
                    chair_index=chair_num,
                    specificity_rank=int(pt.solver_hints.get("page_specificity", 0)),
                    source_voice_keys=source_voice_keys,
                    source_layer_keys=source_layer_keys,
                    performer_count=max(1, performer_count),
                    solver_hints={
                        "x": chair_x,
                        "y": chair_y,
                        "radius": radius,
                        "angle_deg": chair_angle_deg,
                        "section_center_angle": section_center_angle,
                    },
                )

                performers: list[PerformerPlacement] = []
                perf_center = 0.5 * (chair.performer_count - 1)
                for performer_idx in range(chair.performer_count):
                    perf_num = performer_idx + 1
                    rng = _stable_rng(f"{pt.key}:{chair.key}:{perf_num}")
                    primary_voice_key = (
                        (
                            list(packed.get("source_voice_keys", []))
                            or source_voice_keys
                        )[performer_idx % max(1, len(list(packed.get("source_voice_keys", [])) or source_voice_keys))]
                        if source_voice_keys else ""
                    )
                    primary_layer_key = (
                        (
                            list(packed.get("source_layer_keys", []))
                            or source_layer_keys
                        )[performer_idx % max(1, len(list(packed.get("source_layer_keys", [])) or source_layer_keys))]
                        if source_layer_keys else ""
                    )
                    lateral = (performer_idx - perf_center) * 0.22
                    depth = rng.uniform(-0.10, 0.10) + 0.06 * ((chair_idx % 2) - 0.5)
                    if pt.solver_hints["section_shape"] == "line":
                        perf_x = chair_x + lateral
                        perf_y = chair_y + depth
                        angle_deg = 0.0
                    else:
                        ang = math.radians(chair_angle_deg)
                        tangent_x = math.cos(ang)
                        tangent_y = -math.sin(ang)
                        radial_x = math.sin(ang)
                        radial_y = math.cos(ang)
                        perf_x = chair_x + tangent_x * lateral + radial_x * depth
                        perf_y = chair_y + tangent_y * lateral + radial_y * depth
                        angle_deg = chair_angle_deg + lateral * 6.0
                    distance = math.sqrt(perf_x * perf_x + perf_y * perf_y)
                    geometric_delay_ms = (distance * 0.85 / 343.0) * 1000.0
                    humanization_ms = rng.uniform(-4.0, 4.0) * (
                        1.0 + min(2.0, 0.01 * float(chair.specificity_rank))
                    )
                    gain_db = rng.uniform(-1.2, 1.2)
                    pan = max(-1.0, min(1.0, perf_x / 9.0))
                    phase_offset_rad = (
                        0.0 if _performer_phase_mode(patch) == "coherent"
                        else rng.uniform(0.0, 2.0 * math.pi)
                    )
                    # Aperture normal: performer faces the conductor at origin
                    _fdx, _fdy = -perf_x, -perf_y
                    _fdn = math.sqrt(_fdx * _fdx + _fdy * _fdy) or 1.0
                    _face_x, _face_y = _fdx / _fdn, _fdy / _fdn

                    performers.append(PerformerPlacement(
                        key=f"{chair.key}:performer:{perf_num}",
                        label=f"{chair.label} performer {perf_num}",
                        chair_key=chair.key,
                        chair_index=chair_num,
                        performer_index=perf_num,
                        source_voice_keys=[primary_voice_key] if primary_voice_key else [],
                        source_layer_keys=[primary_layer_key] if primary_layer_key else [],
                        assigned_note_keys=list(packed.get("assigned_note_keys", [])),
                        body_type=str(getattr(voice_map.get(primary_voice_key), "body_type", chair_body_type) or chair_body_type),
                        x=perf_x,
                        y=perf_y,
                        z=sec_z,
                        face_x=_face_x,
                        face_y=_face_y,
                        face_z=0.0,
                        radius=distance,
                        angle_deg=angle_deg,
                        geometric_delay_ms=geometric_delay_ms,
                        humanization_ms=humanization_ms,
                        phase_offset_rad=phase_offset_rad,
                        gain_db=gain_db,
                        pan=pan,
                    ))
                chair.performers = performers
                chairs.append(chair)
            pt.chairs = chairs

    # ── Auto-deploy resonator module when placement_resonator is enabled ──────
    _res_cfg2 = getattr(patch, "placement_resonator", None)
    if _res_cfg2 is not None and getattr(_res_cfg2, "enabled", False) and getattr(patch, "parts", []):
        _dep_key = str(getattr(_res_cfg2, "deployed_module_key", "") or "")
        _dep_mod = None
        if _dep_key:
            _dep_mod = next((m for m in getattr(patch, "modules", []) if m.key == _dep_key), None)
        if _dep_mod is None:
            _dep_mod = next(
                (m for m in getattr(patch, "modules", [])
                 if m.module_type == "state_machine"
                 and str(getattr(m, "sm_plugin", "")) == "orchestral_resonance"),
                None,
            )
        if _dep_mod is None:
            _dep_mod = AnalyticModule(
                key=f"placement_res_{uuid.uuid4().hex[:6]}",
                label="Orchestral Resonance",
                module_type="state_machine",
            )
            patch.modules.append(_dep_mod)
        _sync_placement_resonator_module(patch, _dep_mod)


def _performer_map_for_patch(patch: "AnalyticPatch") -> dict[str, list[PerformerPlacement]]:
    mapping: dict[str, list[PerformerPlacement]] = {}
    for pt in getattr(patch, "parts", []):
        for ch in getattr(pt, "chairs", []):
            for pf in getattr(ch, "performers", []):
                for vk in getattr(pf, "source_voice_keys", []):
                    if vk:
                        mapping.setdefault(vk, []).append(pf)
    return mapping


def _resolve_note_target(voices: list, patch: "AnalyticPatch") -> "NoteTarget":
    """Return the most holistic NoteTarget available for *voices* in *patch*.

    Dispatch hierarchy (most → least coordinated):
      1. Performers  — PerformerPlacement entries exist in chairs → apply
                       per-seat geometric delay, phase, gain, and pan.
      2. Chairs      — Chair sections exist but no performers yet → instrument-
                       level grouping without individual spatial transforms.
      3. Voice       — No Parts/Chairs resolved → synthesis-only direct path.

    The returned target always carries the resolved voice list so downstream
    synthesis is uniform regardless of which level was matched.
    """
    voice_keys = frozenset(getattr(v, "key", "") for v in voices) - {""}

    best_performers: list = []
    best_chairs: list = []
    best_part: object = None

    for pt in getattr(patch, "parts", []):
        pt_voice_keys = frozenset(vk for vk in getattr(pt, "voice_keys", []) if vk)
        if not pt_voice_keys.intersection(voice_keys):
            continue
        chairs = getattr(pt, "chairs", [])
        if not chairs:
            continue
        # Collect performers across all chairs that reference at least one of our voices
        local_performers: list = []
        local_chairs: list = []
        for ch in chairs:
            ch_voices = frozenset(vk for vk in getattr(ch, "source_voice_keys", []) if vk)
            if not ch_voices and not getattr(ch, "performers", []):
                # Chair has no voice filter — treat as matching all part voices
                ch_voices = pt_voice_keys
            if ch_voices.intersection(voice_keys) or not ch_voices:
                local_chairs.append(ch)
                local_performers.extend(getattr(ch, "performers", []))
        if local_performers:
            best_performers = local_performers
            best_chairs = local_chairs
            best_part = pt
            break  # first matching part with performers wins
        if local_chairs and best_part is None:
            best_chairs = local_chairs
            best_part = pt

    if best_performers:
        return NoteTarget(
            target_type="performer",
            voices=list(voices),
            performers=best_performers,
            chairs=best_chairs,
            part=best_part,
        )
    if best_chairs:
        return NoteTarget(
            target_type="chair",
            voices=list(voices),
            performers=[],
            chairs=best_chairs,
            part=best_part,
        )
    return NoteTarget(
        target_type="voice",
        voices=list(voices),
        performers=[],
        chairs=[],
        part=None,
    )


def _apply_performer_transforms_to_src(
    performer_parent_map: "dict[str, list[PerformerPlacement]]",
    voice_sigs: "dict[str, np.ndarray]",
    Src: "np.ndarray",
    ki: "dict[str, int]",
    n_ext: int,
    sr: int,
) -> None:
    """Inject performer-transformed voice signals into *Src* in-place.

    For every voice that has PerformerPlacement entries (those excluded from the
    normal ``Src`` population), synthesize the ensemble contribution:

      1. Take the raw synthesised voice signal.
      2. For each PerformerPlacement:
           a. Apply geometric + humanization delay (integer-sample circular shift,
              zeroing the pre-roll region so causality is preserved).
           b. Apply phase offset (complex rotation of the analytic signal).
           c. Apply gain_db (amplitude scale).
      3. Sum performer copies and average by performer count (preserves loudness
         regardless of section size).
      4. Write the result into ``Src[ki[voice_key]]``.

    This implements the "dispatch to performers" step: the solved score (NoteSchedule)
    was handed to the most holistic available target (PerformerPlacement).  When no
    performers exist, this function is a no-op and voices reach Src via the normal
    un-transformed path.
    """
    for vk, placements in performer_parent_map.items():
        if vk not in ki or vk not in voice_sigs:
            continue
        raw = np.asarray(voice_sigs[vk], dtype=np.complex128)
        if len(raw) < n_ext:
            raw = np.pad(raw, (0, n_ext - len(raw)))
        else:
            raw = raw[:n_ext]

        acc = np.zeros(n_ext, dtype=np.complex128)
        for pf in placements:
            delay_s = (float(getattr(pf, "geometric_delay_ms", 0.0))
                       + float(getattr(pf, "humanization_ms", 0.0))) * 1e-3
            delay_n = int(round(delay_s * sr))
            sig = raw.copy()
            if delay_n > 0:
                sig = np.roll(sig, delay_n)
                sig[:delay_n] = 0.0
            phase = float(getattr(pf, "phase_offset_rad", 0.0))
            if phase:
                sig = sig * complex(math.cos(phase), math.sin(phase))
            gain_db = float(getattr(pf, "gain_db", 0.0))
            if gain_db:
                sig = sig * (10.0 ** (gain_db / 20.0))
            acc += sig

        n_pl = len(placements)
        if n_pl > 1:
            acc /= n_pl
        Src[ki[vk]] = acc


def _placement_resonator_item_count(patch: "AnalyticPatch") -> int:
    performer_total = sum(
        len(getattr(ch, "performers", []))
        for pt in getattr(patch, "parts", [])
        for ch in getattr(pt, "chairs", [])
    )
    if performer_total > 0:
        return performer_total
    voice_total = len(getattr(patch, "voices", []))
    return max(1, voice_total)


def _placement_body_types_for_patch(patch: "AnalyticPatch") -> list[str]:
    types: list[str] = []
    for voice in getattr(patch, "voices", []):
        body_type = str(getattr(voice, "body_type", "direct") or "direct")
        if body_type not in types:
            types.append(body_type)
    return types or ["direct"]


def _placement_performer_geometry_json(patch: "AnalyticPatch") -> str:
    """Serialize all performer positions/directions to JSON for the resonance plugin."""
    import json as _json
    entries = []
    for pt in getattr(patch, "parts", []):
        for ch in getattr(pt, "chairs", []):
            for pf in getattr(ch, "performers", []):
                px, py, pz = float(getattr(pf, "x", 0.0)), float(getattr(pf, "y", 0.0)), float(getattr(pf, "z", 1.1))
                # Performer faces toward front-center (0, 0, pz): direction = normalize(-px, -py, 0)
                # In room coords performers face downstage (toward receiver/audience).
                dx, dy = -px, -py
                dn = math.sqrt(dx*dx + dy*dy) or 1.0
                entries.append({
                    "key": str(getattr(pf, "key", "")),
                    "x": px,
                    "y": py,
                    "z": pz,
                    "dir_x": round(float(getattr(pf, "face_x", dx / dn)), 4),
                    "dir_y": round(float(getattr(pf, "face_y", dy / dn)), 4),
                    "dir_z": round(float(getattr(pf, "face_z", 0.0)), 4),
                    "body_type": str(getattr(pf, "body_type", "direct")),
                    "part_key": str(getattr(pt, "key", "")),
                    "register": str(getattr(pt, "register", "")),
                })
    if not entries:
        return ""
    return _json.dumps(entries, separators=(",", ":"))


def _placement_resonator_module_params(patch: "AnalyticPatch") -> dict[str, object]:
    cfg = getattr(patch, "placement_resonator", PlacementResonatorConfig())
    params = {
        "room_shape": cfg.room_shape,
        "scene_path": str(cfg.scene_path),
        "room_radius": float(cfg.room_radius),
        "room_height": float(cfg.room_height),
        "feedback_iterations": int(cfg.feedback_iterations),
        "feedback_gain": float(cfg.feedback_gain),
        "passive_loss": float(cfg.passive_loss),
        "band_split_mode": cfg.band_split_mode,
        "fir_taps": int(cfg.fir_taps),
        "high_cone_deg": float(cfg.high_cone_deg),
        "diffuse_strength": float(cfg.diffuse_strength),
        "air_db_per_m": float(cfg.air_db_per_m),
        "air_highband_db_per_m": float(cfg.air_highband_db_per_m),
        "temperature_c": float(cfg.temperature_c),
        "humidity_rel": float(cfg.humidity_rel),
        "placement_owner": str(cfg.owner_module_type),
        "placement_body_types": ",".join(_placement_body_types_for_patch(patch)),
        "placement_item_count": int(_placement_resonator_item_count(patch)),
        # Receiver array preset
        "receiver_array_key": str(cfg.receiver_array_key),
        "receiver_pos_x": float(cfg.receiver_pos_x),
        "receiver_pos_y": float(cfg.receiver_pos_y),
        "receiver_pos_z": float(cfg.receiver_pos_z),
        "receiver_fwd_x": float(cfg.receiver_fwd_x),
        "receiver_fwd_y": float(cfg.receiver_fwd_y),
        "receiver_fwd_z": float(cfg.receiver_fwd_z),
        # Placement geometry for source positions
        "performer_geometry_json": _placement_performer_geometry_json(patch),
    }
    return params


def _sync_placement_resonator_module(patch: "AnalyticPatch", module: "AnalyticModule") -> None:
    cfg = getattr(patch, "placement_resonator", PlacementResonatorConfig())
    module.module_type = "state_machine"
    module.label = module.label or "Orchestral Resonance"
    module.sm_plugin = "orchestral_resonance"
    module.sm_n_items = _placement_resonator_item_count(patch)
    plug = _load_sm_plugin(module.sm_plugin)
    if plug is not None:
        module.sm_vars = _sm_plugin_output_vars(plug)
        module.sm_state_vars = _sm_plugin_state_vars(plug)
        module.sm_items = _sm_plugin_item_names(plug, module.sm_n_items)
        defaults = _sm_plugin_default_params(plug)
    else:
        defaults = {}
    module.sm_params = {
        **defaults,
        **dict(getattr(module, "sm_params", {}) or {}),
        **_placement_resonator_module_params(patch),
    }
    module.sm_use_torch = True
    module._sm_state = {}
    module._sm_out_cache = {}
    module._sm_aux_state = {}
    if not cfg.deployed_module_key:
        cfg.deployed_module_key = module.key


def _make_note_temp_patch(
    parent: "AnalyticPatch",
    note_voices: "list[AnalyticVoice]",
    duration_s: float,
    event_hz: float,
    note_keys: "list[str]",
    group_voice_keys: "list[str]",
    *,
    shared_modules: "list[AnalyticModule] | None" = None,
) -> "AnalyticPatch":
    """Build a lightweight per-note patch that shares read-only structures by reference.

    Only the module *state* needs isolation: ``_sm_state``, ``_sm_out_cache``,
    ``_sm_aux_state``, and ``_sm_log_text`` are the only fields that
    ``_synthesize_patch`` mutates on a module.  Everything else (routing, LFOs,
    controls, mixers, param_nodes, system_audio) is read-only during synthesis
    and can be shared safely.

    When *shared_modules* is provided those module objects are used directly
    (their mutable state slots are snapshotted/restored by the caller).
    Otherwise, fall back to a shallow copy with fresh state dicts.
    """
    tp = AnalyticPatch()
    tp.duration               = duration_s
    tp.preview_sr             = parent.preview_sr
    tp.voices                 = note_voices
    # Read-only — share by reference
    tp.lfos                   = parent.lfos
    tp.controls               = parent.controls
    tp.routing                = parent.routing
    tp.mixers                 = parent.mixers
    tp.param_nodes            = parent.param_nodes
    tp.system_audio           = parent.system_audio
    tp.tuning                 = parent.tuning
    tp.projection_mode        = parent.projection_mode
    tp.projection_rotation_hz = parent.projection_rotation_hz
    tp.normalize_output       = False
    tp.performer_phase_mode   = parent.performer_phase_mode
    tp.seq_tonic_hz           = parent.seq_tonic_hz
    tp._seq_note_hz           = float(event_hz)
    # Modules: shallow-copy list, reset mutable state slots so notes don't
    # cross-contaminate.  Scene caches live inside ``_sm_aux_state`` and are
    # persisted separately by the caller if desired.
    if shared_modules is not None:
        tp.modules = shared_modules
    else:
        _fresh: list[AnalyticModule] = []
        for m in parent.modules:
            mc = copy.copy(m)           # shallow — shares sm_params, sm_items etc.
            mc._sm_state     = {}
            mc._sm_out_cache = {}
            mc._sm_aux_state = {}
            mc._sm_log_text  = ""
            _fresh.append(mc)
        tp.modules = _fresh
    tp.parts = _copy_matching_parts_for_voice_keys(
        parent, group_voice_keys, note_keys)
    return tp


def _copy_matching_parts_for_voice_keys(
    source_patch: "AnalyticPatch",
    voice_keys: list[str],
    note_keys: list[str] | None = None,
) -> list[Part]:
    voice_set = frozenset(vk for vk in voice_keys if vk)
    if not voice_set:
        return []

    def _shallow_part(pt: Part) -> Part:
        """Shallow-copy a Part and its Chairs so we can reassign list fields
        without mutating the source patch.  PerformerPlacement objects are
        shared by reference (never mutated during synthesis)."""
        p2 = copy.copy(pt)
        p2.chairs = [copy.copy(ch) for ch in getattr(pt, "chairs", [])]
        return p2

    matches = [
        _shallow_part(pt)
        for pt in getattr(source_patch, "parts", [])
        if frozenset(vk for vk in getattr(pt, "voice_keys", []) if vk) == voice_set
    ]
    parts = matches if matches else [
        _shallow_part(pt)
        for pt in getattr(source_patch, "parts", [])
        if voice_set.issubset(frozenset(vk for vk in getattr(pt, "voice_keys", []) if vk))
    ]
    note_key_set = {nk for nk in (note_keys or []) if nk}
    if not note_key_set:
        return parts
    filtered_parts: list[Part] = []
    for pt in parts:
        kept_chairs: list[Chair] = []
        for ch in getattr(pt, "chairs", []):
            kept_performers = [
                pf for pf in getattr(ch, "performers", [])
                if not getattr(pf, "assigned_note_keys", [])
                or note_key_set.intersection(set(pf.assigned_note_keys))
            ]
            if not kept_performers:
                continue
            ch.performers = kept_performers
            ch.performer_count = len(kept_performers)
            kept_chairs.append(ch)
        if kept_chairs:
            pt.chairs = kept_chairs
            pt.player_count = sum(ch.performer_count for ch in kept_chairs)
            filtered_parts.append(pt)
    return filtered_parts


def _compute_arrangement_metrics(play_groups: list[tuple]) -> dict:
    """Summarize solved score groups for future placement/chair allocation."""
    all_events = []
    group_metrics = []
    voice_event_counts: dict[str, int] = {}
    for gi, (sched, voices) in enumerate(play_groups):
        events = list(getattr(sched, "events", []) or [])
        layer_keys = list(getattr(sched, "_layer_keys", []))
        group_key = str(getattr(sched, "_group_key", f"group:{gi}"))
        voice_keys = [getattr(v, "key", "") for v in voices]
        note_demands = _note_demands_for_group(group_key, events, voices)
        packed_bins, packed_summary = _pack_note_demands_by_voice(note_demands, voices)
        for vk in voice_keys:
            voice_event_counts[vk] = voice_event_counts.get(vk, 0) + len(events)
        overlap = _schedule_overlap_metrics(events, voices)
        group_metrics.append({
            "group_key": group_key,
            "layer_keys": layer_keys,
            "voice_keys": voice_keys,
            "event_count": len(events),
            "start_time": min((ev.start_time for ev in events), default=0.0),
            "end_time": max((ev.start_time + ev.duration_s for ev in events), default=0.0),
            "page_specificity": _page_specificity_score(layer_keys),
            "stack_depth": len(layer_keys),
            "note_demands": note_demands,
            "packed_performers": [
                {
                    "voice_key": pb.get("voice_key", ""),
                    "polyphony_mode": pb.get("polyphony_mode", "sympathetic"),
                    "capacity": int(pb.get("capacity", 1)),
                    "assigned_note_keys": [d["note_key"] for d in pb.get("note_demands", [])],
                    "source_voice_keys": list({
                        d["voice_key"] for d in pb.get("note_demands", []) if d.get("voice_key")
                    }),
                    "source_layer_keys": list({
                        d["layer_key"] for d in pb.get("note_demands", []) if d.get("layer_key")
                    }),
                }
                for pb in packed_bins
            ],
            "voice_profiles": [
                {
                    "voice_key": getattr(v, "key", ""),
                    "polyphony_count": _voice_polyphony_capacity(v),
                    "polyphony_mode": _voice_polyphony_mode(v),
                }
                for v in voices
            ],
            **packed_summary,
            **overlap,
        })
        all_events.extend(events)

    timeline = []
    for ev in all_events:
        t0 = float(getattr(ev, "start_time", 0.0))
        t1 = t0 + float(getattr(ev, "duration_s", 0.0))
        timeline.append((t0, 1))
        timeline.append((t1, -1))
    timeline.sort(key=lambda item: (item[0], item[1]))

    active = 0
    peak = 0
    peak_times: list[float] = []
    for t, delta in timeline:
        active += delta
        if active > peak:
            peak = active
            peak_times = [t]
        elif active == peak and peak > 0:
            peak_times.append(t)

    return {
        "group_count": len(play_groups),
        "event_count": len(all_events),
        "peak_simultaneity": peak,
        "peak_times": peak_times[:32],
        "groups": group_metrics,
        "voice_event_counts": voice_event_counts,
        "peak_required_chairs": max(
            (max(int(gm.get("required_chairs", 1)), int(gm.get("required_performers", 1))) for gm in group_metrics),
            default=1,
        ),
    }


def resolve_parts_from_patch(patch: "AnalyticPatch") -> "list[Part]":
    """Derive :class:`Part` objects from the patch's arrangement metrics.

    Each group in `_arrangement_metrics` becomes one Part. Parts with
    identical voice combinations are deduplicated. Player counts and solver
    hints are preserved from any existing parts already on the patch so that
    manually-set player counts survive a re-solve.

    Returns a fresh list ready to be stored as ``patch.parts``.
    """
    metrics = patch._arrangement_metrics
    if not metrics:
        return []
    groups = metrics.get("groups", [])
    existing_by_key = {pt.key: pt for pt in patch.parts}
    parts: list[Part] = []
    seen_voice_sets: set[frozenset] = set()
    for gm in groups:
        group_key: str = gm.get("group_key", "")
        voice_keys: list = list(gm.get("voice_keys", []))
        vset = frozenset(voice_keys)
        if vset in seen_voice_sets:
            continue
        seen_voice_sets.add(vset)
        layer_keys: list = list(gm.get("layer_keys", []))
        # Infer register and role from group / layer key naming
        combined = group_key + " ".join(layer_keys)
        combined_l = combined.lower()
        if "bass" in combined_l or "sub" in combined_l:
            register = "bass"
        elif "high" in combined_l or "treble" in combined_l or "soprano" in combined_l:
            register = "high"
        elif "mid" in combined_l or "tenor" in combined_l or "alto" in combined_l:
            register = "mid"
        else:
            register = "all"
        if "melody" in combined_l:
            seq_role = "melody"
        elif "root" in combined_l or "bass" in combined_l:
            seq_role = "bass"
        elif "stab" in combined_l or "comp" in combined_l:
            seq_role = "stab"
        else:
            seq_role = ""
        if "signal" in combined_l:
            voice_role = "signal"
        elif "air" in combined_l:
            voice_role = "air"
        elif "transient" in combined_l:
            voice_role = "transient"
        elif "body" in combined_l:
            voice_role = "body"
        else:
            voice_role = ""
        part_key = group_key or f"part-{len(parts)}"
        label_parts = [register, seq_role, voice_role]
        label = " / ".join(p for p in label_parts if p) or part_key
        existing = existing_by_key.get(part_key)
        required_chairs = max(
            1,
            int(gm.get("required_performers", gm.get("required_chairs", 1))),
        )
        player_count = max(required_chairs, existing.player_count if existing else 1)
        solver_hints: dict = {
            "event_count": gm.get("event_count", 0),
            "start_time": gm.get("start_time", 0.0),
            "end_time": gm.get("end_time", 0.0),
            "required_chairs": required_chairs,
            "required_performers": int(gm.get("required_performers", required_chairs)),
            "peak_active_events": int(gm.get("peak_active_events", 0)),
            "peak_active_sources": int(gm.get("peak_active_sources", 0)),
            "interference_peak": float(gm.get("interference_peak", 0.0)),
            "page_specificity": int(gm.get("page_specificity", 0)),
            "stack_depth": int(gm.get("stack_depth", 0)),
            "layer_keys": list(gm.get("layer_keys", [])),
            "voice_profiles": list(gm.get("voice_profiles", [])),
            "packed_performers": list(gm.get("packed_performers", [])),
        }
        if existing and existing.solver_hints:
            solver_hints.update({k: v for k, v in existing.solver_hints.items()
                                  if k not in solver_hints})
        parts.append(Part(
            key=part_key,
            label=label,
            register=register,
            seq_role=seq_role,
            voice_role=voice_role,
            voice_keys=voice_keys,
            player_count=player_count,
            solver_hints=solver_hints,
        ))
    patch.parts = parts
    _refresh_part_placement_layout(patch)
    return parts


def _build_play_groups_from_resolved_notes(
        p: "AnalyticPatch",
) -> "list[tuple[NoteSchedule, list[AnalyticVoice]]]":
    """Build per-note playback groups from locked/edited resolved notes."""
    voice_map = {v.key: v for v in p.voices if not getattr(v, "muted", False)}
    groups: list[tuple[NoteSchedule, list[AnalyticVoice]]] = []
    for note in sorted(p.resolved_notes, key=lambda n: (n.start_time, n.fundamental_hz, n.voice_key)):
        if getattr(note, "is_rest", False):
            continue
        voice = voice_map.get(note.voice_key)
        if voice is None:
            continue
        sched = NoteSchedule()
        ev = NoteEvent(
            fundamental_hz=float(note.fundamental_hz),
            start_time=float(note.start_time),
            duration_s=float(note.duration_s),
            velocity=float(note.velocity),
        )
        ev._layer_key = getattr(note, "layer_key", "roll")
        ev._exact_pitch = True
        sched.add(ev)
        sched._group_key = f"roll:{voice.key}:{note.note_id}"
        sched._layer_keys = [getattr(note, "layer_key", "roll")]
        sched._note_target = _resolve_note_target([voice], p)
        groups.append((sched, [voice]))
    return groups


def _prepare_sequence_play_groups(
        p: "AnalyticPatch",
        beat_s: float,
        degrees: list[float],
        pattern: list[int],
) -> "list[tuple[NoteSchedule, list[AnalyticVoice]]]":
    """Return the playback/export groups for the current patch state."""
    has_locked_roll = any(getattr(n, "locked", False) for n in p.resolved_notes)
    if has_locked_roll:
        _sync_resolved_notes(p, preserve_locked=True)
        groups = _build_play_groups_from_resolved_notes(p)
        p._arrangement_metrics = _compute_arrangement_metrics(groups)
        p.parts = resolve_parts_from_patch(p)
        return groups

    if p.rhythm_enabled:
        source_voices = [v for v in p.voices if not v.muted]
        groups = _group_voices_by_page(
            p, source_voices, beat_s, degrees, pattern,
            min_dur_s=1.0 / p.preview_sr)
    else:
        schedule = _build_legacy_sequence_schedule(p, beat_s, degrees, pattern)
        source_voices = [v for v in p.voices if not v.muted]
        if source_voices:
            schedule._note_target = _resolve_note_target(source_voices, p)
        groups = [(schedule, source_voices)] if source_voices else []
    if getattr(p, "seq_rubato_shape", "off") != "off" and getattr(p, "seq_rubato_amount", 0.0) > 1e-9:
        warped_groups = []
        phrase_supercycle_s = (
            _rubato_phrase_lcm_cycle_s(p, groups, beat_s)
            if getattr(p, "seq_rubato_scope", "bar") == "phrase"
            else 0.0
        )
        for sched, voices in groups:
            if getattr(p, "seq_rubato_scope", "bar") == "phrase":
                local_phrase_s = max(1e-6, _rubato_phrase_cycle_s_for_group(p, voices, beat_s))
                cycle_s = max(local_phrase_s, phrase_supercycle_s)
                repeats = max(1.0, cycle_s / local_phrase_s)
                amt_scale = 1.0 / repeats
            else:
                cycle_s = _rubato_cycle_s_for_group(p, voices, beat_s)
                amt_scale = 1.0
            warped = _apply_rubato_to_schedule(p, sched, cycle_s, amount_scale=amt_scale)
            # Preserve note-target annotation through rubato warp
            warped._note_target = getattr(sched, "_note_target",
                                          _resolve_note_target(voices, p))
            warped_groups.append((warped, voices))
        groups = warped_groups
    p._arrangement_metrics = _compute_arrangement_metrics(groups)
    p.parts = resolve_parts_from_patch(p)
    return groups


def _group_voices_by_page(
        p: "AnalyticPatch",
        source_voices: list,
        beat_s: float,
        degrees: list,
        deg_pattern: list,
        min_dur_s: float = 1.0 / 48000,
) -> "list[tuple]":
    """Return [(NoteSchedule, [voice, ...]), ...] grouped by resolved score stack."""
    pid_to_layers: dict[str, list[tuple[str, RhythmPage]]] = {}
    pid_to_voices: dict[str, list] = {}
    for v in source_voices:
        layers = p.score_page_items_for_voice(v)
        if getattr(p, "rhythm_layer_mode", "union") == "specific":
            stack_key = (layers[-1][0],) if layers else ("all",)
        else:
            stack_key = tuple(key for key, _ in layers) if layers else ("all",)
        pid = "|".join(stack_key)
        if pid not in pid_to_layers:
            pid_to_layers[pid] = layers
            pid_to_voices[pid] = []
        pid_to_voices[pid].append(v)

    result = []
    for pid, voices in pid_to_voices.items():
        sched = _build_score_schedule_for_voice(
            p, voices[0], beat_s, degrees, deg_pattern, min_dur_s=min_dur_s)
        sched._group_key = pid
        sched._layer_keys = [key for key, _ in p.score_page_items_for_voice(voices[0])]
        sched._note_target = _resolve_note_target(voices, p)
        result.append((sched, pid_to_voices[pid]))
    return result


# Sentinel: use spec.editor_seed for this synthesis (stable editor default).
# Pass None instead to get an unseeded (random) RNG — for file renders / performance.
_GRANULAR_SEED_EDITOR = object()



def _voice_effectively_muted(voice: "AnalyticVoice", patch: "AnalyticPatch") -> bool:
    """True when the voice should produce silence — muted, or not soloed when a solo is active."""
    if voice.muted:
        return True
    sk = getattr(patch, "solo_key", None)
    return sk is not None and voice.key != sk


def _resolve_voice_hz(
    voice:     "AnalyticVoice",
    tuning:    "GlobalTuning",
    played_hz: "float | None" = None,
) -> float:
    """
    Resolve the actual oscillator frequency for *voice* given a *tuning* context
    and an optional *played_hz* (the note pitch from the sequencer).

    note_tracking behaviour
    -----------------------
    ``"free"``
        Ignore tuning and played pitch entirely — use ``voice.freq_hz`` as-is.
        Preserves exact backward-compatibility with pre-tuning patches.
    ``"root"``
        Voice is anchored to the tuning root.  ``semitone_offset`` shifts it up/
        down in semitones from ``tuning.root_hz``.  Sequencer notes are ignored.
    ``"note"``  (default)
        Voice tracks the played note.  ``semitone_offset`` is an additive offset
        on top of the played pitch in semitones.  When no note is playing,
        falls back to root behaviour.
    """
    tracking = getattr(voice, "note_tracking", "free")
    offset   = getattr(voice, "semitone_offset", 0.0)
    if tracking == "free":
        return voice.freq_hz
    elif tracking == "root":
        return tuning.semitone_to_hz(offset)
    else:  # "note"
        if played_hz is None or played_hz <= 0.0:
            return tuning.semitone_to_hz(offset)
        played_st = tuning.hz_to_semitones(played_hz)
        return tuning.semitone_to_hz(played_st + offset)


def _resolved_event_hz(
    patch: "AnalyticPatch",
    voice: "AnalyticVoice",
    event_hz: float,
) -> float:
    """Return the actual rendered pitch for *voice* on a scheduled event."""
    resolved_hz = _resolve_voice_hz(voice, patch.tuning, event_hz)
    seq_role = getattr(voice, "seq_role", "melody")
    if seq_role == "bass":
        resolved_hz *= (2.0 ** patch.seq_bass_octave)
    elif seq_role == "root":
        resolved_hz = patch.seq_tonic_hz * (2.0 ** patch.seq_root_octave)
    elif seq_role == "stab":
        resolved_hz *= (2.0 ** patch.seq_stab_octave)
    return resolved_hz


def _hz_to_midi(hz: float) -> float:
    if hz <= 0.0:
        return 0.0
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def _midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def _midi_note_name(midi_note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = int(midi_note)
    return f"{names[note % 12]}{(note // 12) - 1}"


def _bar_duration_s_for_patch(patch: "AnalyticPatch", beat_s: float | None = None) -> float:
    beat = (60.0 / max(float(getattr(patch, "seq_bpm", 120.0)), 1.0)
            if beat_s is None else float(beat_s))
    return beat * patch.beats_per_bar()


def _bar_duration_s_for_page(
        patch: "AnalyticPatch",
        page: "RhythmPage | None",
        beat_s: float | None = None) -> float:
    beat = (60.0 / max(float(getattr(patch, "seq_bpm", 120.0)), 1.0)
            if beat_s is None else float(beat_s))
    return beat * patch.page_beats_per_bar(page)


def _page_meter_label(patch: "AnalyticPatch", page: "RhythmPage | None") -> str:
    num, den = patch.page_meter(page)

    def _fmt(v: float) -> str:
        if abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return f"{v:.2f}".rstrip("0").rstrip(".")

    return f"{_fmt(num)}/{_fmt(den)}"


def _rubato_phase_map(shape: str, amount: float, u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    amt = max(0.0, min(0.95, float(amount)))
    if amt <= 1e-9 or shape == "off":
        return u
    if shape == "sine":
        return u + amt * math.sin(2.0 * math.pi * u) / (2.0 * math.pi)
    if shape == "troughs":
        return u + amt * math.sin(4.0 * math.pi * u) / (4.0 * math.pi)
    if shape == "slow_go":
        eased = u * u
        return (1.0 - amt) * u + amt * eased
    if shape == "go_slow":
        eased = 1.0 - (1.0 - u) * (1.0 - u)
        return (1.0 - amt) * u + amt * eased
    return u


def _rubato_cycle_s_for_group(
        patch: "AnalyticPatch",
        voices: list["AnalyticVoice"],
        beat_s: float) -> float:
    if patch.rhythm_enabled and voices:
        layers = patch.score_page_items_for_voice(voices[0])
        page = layers[-1][1] if layers else None
        bar_s = _bar_duration_s_for_page(patch, page, beat_s)
        if patch.seq_rubato_scope == "phrase":
            bars = max(1, int(getattr(page, "rhythm_prog_bars", patch.rhythm_prog_bars) if page is not None
                              else patch.rhythm_prog_bars))
            return bar_s * bars
        return bar_s
    return _bar_duration_s_for_patch(patch, beat_s)


def _rubato_phrase_cycle_s_for_group(
        patch: "AnalyticPatch",
        voices: list["AnalyticVoice"],
        beat_s: float) -> float:
    if patch.rhythm_enabled and voices:
        layers = patch.score_page_items_for_voice(voices[0])
        page = layers[-1][1] if layers else None
        bar_s = _bar_duration_s_for_page(patch, page, beat_s)
        bars = max(1, int(getattr(page, "rhythm_prog_bars", patch.rhythm_prog_bars) if page is not None
                          else patch.rhythm_prog_bars))
        return bar_s * bars
    return _bar_duration_s_for_patch(patch, beat_s)


def _rubato_phrase_lcm_cycle_s(
        patch: "AnalyticPatch",
        groups: "list[tuple[NoteSchedule, list[AnalyticVoice]]]",
        beat_s: float) -> float:
    durations: list[Fraction] = []
    for _, voices in groups:
        phrase_s = max(1e-6, _rubato_phrase_cycle_s_for_group(patch, voices, beat_s))
        qbeats = Fraction(phrase_s / max(1e-9, beat_s)).limit_denominator(768)
        durations.append(qbeats)
    if not durations:
        return _bar_duration_s_for_patch(patch, beat_s)
    lcm_num = durations[0].numerator
    gcd_den = durations[0].denominator
    for frac in durations[1:]:
        lcm_num = math.lcm(lcm_num, frac.numerator)
        gcd_den = math.gcd(gcd_den, frac.denominator)
    supercycle_qbeats = Fraction(lcm_num, gcd_den)
    return float(supercycle_qbeats) * beat_s


def _apply_rubato_to_schedule(
        patch: "AnalyticPatch",
        schedule: "NoteSchedule",
        cycle_s: float,
        amount_scale: float = 1.0) -> "NoteSchedule":
    shape = getattr(patch, "seq_rubato_shape", "off")
    amount = float(getattr(patch, "seq_rubato_amount", 0.0)) * max(0.0, float(amount_scale))
    if shape == "off" or amount <= 1e-9 or cycle_s <= 1e-9 or not getattr(schedule, "events", None):
        return schedule
    warped = NoteSchedule()
    for ev in schedule.events:
        start = float(ev.start_time)
        end = max(start + 1e-6, float(ev.start_time + ev.duration_s))
        c0 = math.floor(start / cycle_s)
        c1 = math.floor(end / cycle_s)
        if c0 != c1:
            c1 = c0
            end = min(end, (c0 + 1.0) * cycle_s)
        u0 = (start - c0 * cycle_s) / cycle_s
        u1 = (end - c0 * cycle_s) / cycle_s
        t0 = c0 * cycle_s + cycle_s * _rubato_phase_map(shape, amount, u0)
        t1 = c0 * cycle_s + cycle_s * _rubato_phase_map(shape, amount, u1)
        new_ev = NoteEvent(
            fundamental_hz=float(ev.fundamental_hz),
            start_time=float(t0),
            duration_s=max(1e-6, float(t1 - t0)),
            velocity=float(getattr(ev, "velocity", 1.0)),
        )
        for attr in ("_layer_key", "_exact_pitch"):
            if hasattr(ev, attr):
                setattr(new_ev, attr, getattr(ev, attr))
        warped.add(new_ev)
    for attr in ("_group_key", "_layer_keys"):
        if hasattr(schedule, attr):
            setattr(warped, attr, getattr(schedule, attr))
    return warped


def _meter_beat_units(meter_num: float, frac_beat_mode: str = "warp") -> list[float]:
    """Return the beat-unit list for *meter_num*.

    frac_beat_mode
        ``"grid"``  — fractional remainder is a visible beat cell.
        ``"warp"``  — remainder is absorbed into the warp curve; only full
                      integer beats are returned.
    """
    meter_num = max(0.125, float(meter_num))
    full_beats = int(math.floor(meter_num + 1e-9))
    units = [1.0] * max(0, full_beats)
    frac = meter_num - float(full_beats)
    if frac > 1e-6 and frac_beat_mode == "grid":
        units.append(frac)
    if not units:
        units = [meter_num]
    return units


def _stress_pattern_options_for_meter(meter_num: float, frac_beat_mode: str = "warp") -> list[list[int]]:
    beat_units = max(1, len(_meter_beat_units(meter_num, frac_beat_mode)))
    curated: dict[int, list[list[int]]] = {
        1: [[1]],
        2: [[2], [1, 1]],
        3: [[3], [2, 1], [1, 2]],
        4: [[2, 2], [3, 1], [1, 3]],
        5: [[3, 2], [2, 3], [2, 2, 1], [1, 2, 2]],
        6: [[3, 3], [2, 2, 2], [3, 2, 1], [1, 2, 3]],
        7: [[2, 2, 3], [3, 2, 2], [2, 3, 2]],
        8: [[3, 3, 2], [2, 3, 3], [3, 2, 3], [4, 4], [2, 2, 2, 2]],
        9: [[3, 3, 3], [2, 2, 2, 3], [3, 2, 2, 2]],
        10: [[3, 3, 2, 2], [2, 3, 3, 2], [3, 2, 3, 2], [2, 2, 3, 3]],
        11: [[3, 3, 3, 2], [2, 3, 3, 3], [3, 2, 3, 3], [3, 3, 2, 3]],
    }
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def _push(pattern: list[int]) -> None:
        if sum(pattern) != beat_units:
            return
        key = tuple(pattern)
        if key not in seen:
            seen.add(key)
            out.append(pattern)

    for pattern in curated.get(beat_units, []):
        _push(pattern)

    allowed_primary = (2, 3)
    allowed_fallback = (1, 2, 3)

    def _build(rem: int, parts: tuple[int, ...], acc: list[int]) -> None:
        if rem == 0:
            _push(list(acc))
            return
        for part in parts:
            if part <= rem:
                acc.append(part)
                _build(rem - part, parts, acc)
                acc.pop()

    _build(beat_units, allowed_primary, [])
    _build(beat_units, allowed_fallback, [])
    if not out:
        out.append([beat_units])
    return out


def _page_stress_pattern(patch: "AnalyticPatch", page: Any | None) -> list[int]:
    _fbm = getattr(page, "frac_beat_mode", "warp") if page is not None else "warp"
    options = _stress_pattern_options_for_meter(
        patch.page_meter(page if isinstance(page, RhythmPage) else None)[0]
        if page is not None else patch.meter_numerator,
        frac_beat_mode=_fbm)
    current = [int(max(1, int(x))) for x in getattr(page, "stress_pattern", [])] if page is not None else []
    if sum(current) == sum(options[0]):
        return current
    return options[0]


def _stress_step_boundaries(
        patch: "AnalyticPatch",
        page: Any,
        div: int) -> tuple[list[int], list[int], list[int]]:
    num, _ = patch.page_meter(page if isinstance(page, RhythmPage) else None)
    _fbm = getattr(page, "frac_beat_mode", "warp") if page is not None else "warp"
    beat_units = _meter_beat_units(num, _fbm)
    total_units = max(1e-9, sum(beat_units))
    beat_edges = [0]
    acc = 0.0
    for unit in beat_units:
        acc += unit
        beat_edges.append(int(round((acc / total_units) * div)))
    beat_edges[0] = 0
    beat_edges[-1] = div

    pattern = _page_stress_pattern(patch, page)
    group_edges = [0]
    group_acc = 0
    for part in pattern:
        group_acc += int(part)
        idx = min(len(beat_edges) - 1, group_acc)
        group_edges.append(beat_edges[idx])
    if group_edges[-1] != div:
        group_edges[-1] = div
    beat_starts = sorted({max(0, min(div - 1, beat_edges[i])) for i in range(len(beat_edges) - 1)})
    group_starts = sorted({max(0, min(div - 1, group_edges[i])) for i in range(len(group_edges) - 1)})
    return beat_edges, beat_starts, group_starts


def _auto_apply_rhythm_grid(patch: "AnalyticPatch", page: Any) -> None:
    div = max(1, int(page.rhythm_division))
    if not page.rhythm_patterns:
        page.rhythm_patterns = [RhythmPattern(name="Pat 1")]
    act_i = min(page.rhythm_active_pat, len(page.rhythm_patterns) - 1)
    pat = page.rhythm_patterns[act_i]
    pat.ensure_size(div)
    _, beat_starts, group_starts = _stress_step_boundaries(patch, page, div)
    pat.steps = [False] * div
    pat.art = [0] * div
    for step_i in beat_starts:
        pat.steps[step_i] = True
    for step_i in group_starts:
        pat.steps[step_i] = True
        pat.art[step_i] = 1


def _auto_apply_dynamics_accent(
        patch: "AnalyticPatch",
        page: Any,
        dyn_program: "DynamicsProgram") -> None:
    """Distribute Western conventional accent hierarchy onto the accent tree.

    Uses the meter and stress grouping to assign accent levels:
        - Group-start beats  → 2.0  (strong / downbeat)
        - Other beat starts  → 1.5  (accent)
        - Off-beat positions  → 0.5  (weak / ghosted)

    For odd meters the grouping (e.g. 7/8 = 2+2+3) is respected so that the
    '1' of each group pocket gets the strong accent.

    Operates on the accent *tree* of the active rhythm pattern so that it
    works regardless of how the tree is subdivided.
    """
    if page is None:
        return
    div = max(1, int(page.rhythm_division))
    num, _ = patch.page_meter(page if isinstance(page, RhythmPage) else None)

    # Compute beat and group boundaries in [0, div] integer space
    _fbm = getattr(page, "frac_beat_mode", "warp") if page is not None else "warp"
    beat_units  = _meter_beat_units(num, _fbm)
    total_units = max(1e-9, sum(beat_units))
    beat_frac_edges: list[float] = [0.0]
    acc = 0.0
    for unit in beat_units:
        acc += unit
        beat_frac_edges.append(acc / total_units)
    beat_frac_edges[-1] = 1.0

    pattern     = _page_stress_pattern(patch, page)
    group_frac_edges: list[float] = [0.0]
    group_acc = 0
    for part in pattern:
        group_acc += int(part)
        idx = min(len(beat_frac_edges) - 1, group_acc)
        group_frac_edges.append(beat_frac_edges[idx])
    if group_frac_edges[-1] != 1.0:
        group_frac_edges[-1] = 1.0

    beat_set  = set(beat_frac_edges[:-1])   # fractional positions that start a beat
    group_set = set(group_frac_edges[:-1])  # fractional positions that start a group

    # Get the accent tree for the active rhythm pattern
    rpats = page.rhythm_patterns
    if not rpats:
        return
    act_i   = min(page.rhythm_active_pat, max(0, len(rpats) - 1))
    act_pat = rpats[act_i]
    acc_tree = act_pat.get_accent_tree(div)

    # Walk every leaf and assign accent level by positional proximity
    eps = 0.5 / max(1, div)  # tolerance: half a grid step
    for leaf in acc_tree.flat_leaves():
        pos = float(leaf.position)
        # Check group-start first (strongest)
        if any(abs(pos - g) < eps for g in group_set):
            leaf.vel = 2.0
        elif any(abs(pos - b) < eps for b in beat_set):
            leaf.vel = 1.5
        else:
            leaf.vel = 0.5


def _auto_apply_stress_velocity(
        patch: "AnalyticPatch",
        page: Any,
        dyn_program: "DynamicsProgram") -> None:
    """Map stress rank → per-leaf velocity on the accent tree.

    Strong beats (group_starts) receive velocity 2.0 (strong),
    normal beat starts receive 1.0 (normal), all others 0.5 (weak).
    Delegates to the same tree-native logic as Auto Accent.
    """
    _auto_apply_dynamics_accent(patch, page, dyn_program)


def _apply_loop_tiling(sig: np.ndarray, voice: "AnalyticVoice", n: int) -> np.ndarray:
    """Tile [loop_start, loop_end] to fill [loop_end, n) using the complex signal.

    Loop endpoints are snapped to integer phase-cycle boundaries by
    _snap_to_phase_boundary, so exp(i*phase) is continuous at every wrap.
    A short cosine-squared crossfade at each wrap boundary uses the analytic
    complex values directly to erase any sub-sample amplitude residual.
    """
    ls_n = max(0, min(n - 2, int(round(voice.loop_start * n))))
    le_n = max(ls_n + 2, min(n, int(round(voice.loop_end * n))))
    loop_len = le_n - ls_n
    tail_len = n - le_n
    if loop_len < 2 or tail_len <= 0:
        return sig
    out = sig.copy()
    body = sig[ls_n:le_n]
    reps = math.ceil(tail_len / loop_len)
    out[le_n:] = np.tile(body, reps)[:tail_len]
    # Complex crossfade at each wrap: blend tail-of-outgoing with head-of-incoming.
    # Both sides are the same body looped, so this smooths any floating-point seam.
    xfade_n = min(loop_len // 8, 32)
    if xfade_n > 1:
        t_fade   = np.linspace(0.0, math.pi / 2.0, xfade_n, dtype=np.float64)
        fade_out = np.cos(t_fade) ** 2
        fade_in  = np.sin(t_fade) ** 2
        for rep in range(reps):
            wrap = le_n + rep * loop_len
            head = wrap
            tail = wrap - xfade_n
            if tail < le_n or head + xfade_n > n:
                continue
            out[tail:wrap] = out[tail:wrap] * fade_out + out[head:head + xfade_n] * fade_in
    return out


def _synthesize_voice(
    voice: AnalyticVoice,
    patch:   AnalyticPatch,
    lfo_map: dict,
    p_map:   dict,
    t_offset: float = 0.0,    # seconds before nominal t=0 to start synthesis (pre-roll)
    n_samples: int  = 0,      # if >0, override the default sr*duration sample count
    param_overrides: "dict | None" = None,  # {attr: float64 time series} from param routing
    voice_signal_map: "dict | None" = None,  # {voice_key: complex128 array} for voice-to-voice FM/AM
    param_series: "dict | None" = None,  # {param_node_key: float64 series} for ParamNode FM/AM (H4)
    granular_rng_seed: object = _GRANULAR_SEED_EDITOR,  # sentinel→spec.editor_seed; None→random; int→fixed
    granular_seed_offset: int = 0,  # added to editor_seed when using sentinel (for animation)
) -> np.ndarray:
    sr  = patch.preview_sr
    dur = patch.duration
    n   = n_samples if n_samples > 0 else int(sr * dur)

    if voice.piecewise_env is not None:
        po = param_overrides or {}
        freq_hz = float(np.mean(po["freq_hz"][:n])) if "freq_hz" in po and len(po["freq_hz"]) >= n else float(voice.freq_hz)
        gain = float(np.mean(po["amplitude"][:n])) if "amplitude" in po and len(po["amplitude"]) >= n else float(voice.amplitude)
        gate_history = [GateEvent(t_on=0.0, t_off=float(dur), velocity=1.0)]
        env_engine = ParametricCurveEngine(
            voice.piecewise_env.curve,
            voice.piecewise_env.rule_tree,
            chirp_curve=voice.piecewise_env.chirp_curve,
            max_cache=8,
        )
        chirp_engine = ParametricCurveEngine(
            voice.piecewise_env.chirp_curve,
            voice.piecewise_env.rule_tree,
            max_cache=8,
        )
        env_fn = env_engine.interpret(gate_history, force_rebuild=True)
        chirp_fn = chirp_engine.interpret(gate_history, force_rebuild=True)
        def _base_piecewise_chirp(t_abs: "torch.Tensor | np.ndarray | Any") -> np.ndarray:
            if isinstance(t_abs, torch.Tensor):
                t_np = t_abs.detach().cpu().numpy().astype(np.float64, copy=False)
            else:
                t_np = np.asarray(t_abs, dtype=np.float64)
            return _compute_chirp_deviation_series(voice, len(t_np), float(dur), t_axis_s=t_np)
        audio, _amp_env, _chirp_env, _osc = render_piecewise_audio(
            env_fn=env_fn,
            chirp_fn=chirp_fn,
            gate_history=gate_history,
            amp_curve=voice.piecewise_env.curve,
            chirp_curve=voice.piecewise_env.chirp_curve,
            freq_hz=freq_hz,
            gain=gain,
            dur=float(dur),
            sr=int(sr),
            oversample=4,
            base_chirp_hz=_base_piecewise_chirp,
        )
        if audio is None:
            raise RuntimeError(
                f"render_piecewise_audio returned None for voice {voice.key!r}; "
                "check piecewise_env curve and chirp_curve definitions."
            )
        result = np.asarray(audio, dtype=np.complex128).reshape(-1)
        if voice.pre_delay > 0.0:
            silence_n = min(len(result), int(round(voice.pre_delay * float(sr))))
            result[:silence_n] = 0.0
        if len(result) >= n:
            return result[:n]
        out = np.zeros(n, dtype=np.complex128)
        out[:len(result)] = result
        return out

    # --- Granular emission branch ---
    if voice.emission_mode == "granular" and _HAS_GRANULAR:
        gspec = _ensure_granular(voice)
        if gspec is not None:
            import copy as _copy
            gspec_use = _copy.copy(gspec)
            # H2 fix: propagate parent voice manifold settings into the grain spec
            # so that warp, harmonic count, and brightness are not lost in granular mode.
            if voice.manifold_type in ("harmonic", "harmonic_warp"):
                # Map harmonic content to grain_manifold_mix (0=sine,1=harmonic)
                # Override only if the user hasn't explicitly deviated from 0.
                if gspec_use.grain_manifold_mix < 1e-6:
                    gspec_use = _copy.copy(gspec_use)
                    gspec_use.grain_manifold_mix = 1.0
            # Apply scalar param overrides to granular spec fields
            if param_overrides:
                import dataclasses as _dc
                gran_fields = {f.name for f in _dc.fields(gspec_use)}
                for attr, arr in param_overrides.items():
                    # "granular.foo" or bare "foo" both map to spec field "foo"
                    gran_attr = attr[len("granular."):] if attr.startswith("granular.") else attr
                    if gran_attr in gran_fields:
                        setattr(gspec_use, gran_attr, float(np.mean(arr)))
            # Gap 5: build parent phase callable so grains can phase-lock
            _p0  = float(voice.phase_origin)
            _f0  = float(gspec_use.center_frequency_hz)
            _ct  = getattr(voice.chirp, "chirp_type", "none")
            _cfd_start = float(getattr(voice.chirp, "f_delta_start", 0.0))
            _cfd_end   = float(getattr(voice.chirp, "f_delta_end",   0.0))
            _cdur      = max(float(dur), 1e-9)
            if _ct == "linear":
                _chirp_rate = (_cfd_end - _cfd_start) / _cdur
                def _phase_at(t: float, _p=_p0, _f=_f0, _cs=_cfd_start, _cr=_chirp_rate) -> float:
                    fi = _f + _cs + _cr * t
                    return _p + 2.0 * math.pi * (fi * t)
            else:
                def _phase_at(t: float, _p=_p0, _f=_f0) -> float:
                    return _p + 2.0 * math.pi * _f * t

            driver = _GranularClusterDriver(gspec_use, sr=float(sr), parent_phase_at=_phase_at,
                                            rng_seed=(int(gspec_use.editor_seed) + granular_seed_offset
                                                      if granular_rng_seed is _GRANULAR_SEED_EDITOR
                                                      else granular_rng_seed))
            raw = driver.synthesize(dur)
            env = _compute_envelope(voice, len(raw), dur)
            result = (raw * env).astype(np.complex128)
            # Apply pre_delay: silence the leading samples up to pre_delay seconds
            if voice.pre_delay > 0.0:
                silence_n = min(len(result), int(round(voice.pre_delay * float(sr))))
                result[:silence_n] = 0.0
            if len(result) >= n:
                return result[:n]
            out = np.zeros(n, dtype=np.complex128)
            out[:len(result)] = result
            return out
    # t axis: starts at t_offset (negative for pre-roll), advances at 1/sr per sample
    t = (torch.arange(n, dtype=torch.float64) / sr) + t_offset

    # Apply param_overrides to base frequency/amplitude before chirp/FM
    po = param_overrides or {}
    _po_freq = po.get("freq_hz")
    if _po_freq is not None and len(_po_freq) >= n:
        f_inst = torch.as_tensor(_po_freq[:n], dtype=torch.float64)
    else:
        f_inst = torch.full((n,), voice.freq_hz, dtype=torch.float64)
    ct = voice.chirp.chirp_type
    if ct == "linear":
        f_inst = f_inst + torch.linspace(voice.chirp.f_delta_start, voice.chirp.f_delta_end, n, dtype=torch.float64)
    elif ct == "exponential" and voice.chirp.tau > 0:
        decay   = torch.exp(-t / voice.chirp.tau)
        f_inst = f_inst + voice.chirp.f_delta_start * decay + voice.chirp.f_delta_end * (1 - decay)
    elif ct == "power" and dur > 0:
        tau_n = (t / dur) ** max(voice.chirp.chirp_power, 1e-3)
        f_inst = f_inst + voice.chirp.f_delta_start * (1.0 - tau_n) + voice.chirp.f_delta_end * tau_n

    if voice.fm and voice.fm.source_key:
        sk = voice.fm.source_key
        if sk in lfo_map:
            lfo = lfo_map[sk]
            ph = 2.0 * math.pi * lfo.rate_hz * t + lfo.phase_offset
            if lfo.shape == "Sine":
                mod = lfo.depth * torch.sin(ph)
            elif lfo.shape == "Triangle":
                mod = lfo.depth * (2.0 * torch.abs(2.0 * (ph / (2 * math.pi) % 1.0) - 1.0) - 1.0)
            elif lfo.shape == "Sawtooth":
                mod = lfo.depth * (2.0 * (ph / (2 * math.pi) % 1.0) - 1.0)
            else:  # Square
                mod = lfo.depth * torch.sign(torch.sin(ph))
        elif voice_signal_map is not None and sk in voice_signal_map:
            # Use the source voice's synthesized complex signal: extract
            # instantaneous frequency (normalised to [-0.5, 0.5] * Nyquist)
            src_csig = voice_signal_map[sk]
            src_t = torch.as_tensor(src_csig[:n], dtype=torch.complex128)
            phase_diff = torch.angle(src_t[1:] * src_t[:-1].conj())
            f_mod = phase_diff * (float(patch.preview_sr) / (2.0 * math.pi))
            f_mod = torch.cat([f_mod[:1], f_mod])
            f_mid = float(p_map[sk].freq_hz) if sk in p_map else float(torch.mean(torch.abs(f_mod)).item())
            mod = f_mod / max(f_mid, 1.0)  # normalise so depth_hz is in sensible units
        elif param_series is not None and sk in param_series:
            # H4 fix: ParamNode output as FM modulator (already a float64 series)
            ps = param_series[sk]
            if len(ps) >= n:
                mod = torch.as_tensor(ps[:n], dtype=torch.float64)
            else:
                mod = torch.nn.functional.pad(torch.as_tensor(ps[:n], dtype=torch.float64), (0, n - len(ps)))
        elif sk in p_map and sk != voice.key:
            mod = torch.cos(2.0 * math.pi * p_map[sk].freq_hz * t)
        else:
            mod = torch.zeros(n, dtype=torch.float64)
        f_inst = f_inst + voice.fm.depth_hz * mod

    phase = torch.cumsum(2.0 * math.pi * f_inst / sr, dim=0) + voice.phase_origin

    _po_amp = po.get("amplitude")
    if _po_amp is not None and len(_po_amp) >= n:
        amp = torch.as_tensor(_po_amp[:n], dtype=torch.float64)
    else:
        amp = torch.full((n,), voice.amplitude, dtype=torch.float64)
    if voice.am and voice.am.source_key:
        sk = voice.am.source_key
        if sk in lfo_map:
            lfo = lfo_map[sk]
            ph = 2.0 * math.pi * lfo.rate_hz * t + lfo.phase_offset
            if lfo.shape == "Sine":
                mod = lfo.depth * torch.sin(ph)
            elif lfo.shape == "Triangle":
                mod = lfo.depth * (2.0 * torch.abs(2.0 * (ph / (2 * math.pi) % 1.0) - 1.0) - 1.0)
            elif lfo.shape == "Sawtooth":
                mod = lfo.depth * (2.0 * (ph / (2 * math.pi) % 1.0) - 1.0)
            else:  # Square
                mod = lfo.depth * torch.sign(torch.sin(ph))
        elif voice_signal_map is not None and sk in voice_signal_map:
            # Use magnitude envelope of the source voice's synthesized signal
            src_csig = voice_signal_map[sk]
            src_t = torch.as_tensor(src_csig[:n], dtype=torch.complex128)
            mod = torch.abs(src_t).to(torch.float64)
            peak = float(torch.max(mod).item())
            if peak > 1e-12:
                mod = mod / peak
        elif param_series is not None and sk in param_series:
            # H4 fix: ParamNode output as AM modulator (already a float64 series)
            ps = param_series[sk]
            if len(ps) >= n:
                mod = torch.as_tensor(ps[:n], dtype=torch.float64)
            else:
                mod = torch.nn.functional.pad(torch.as_tensor(ps[:n], dtype=torch.float64), (0, n - len(ps)))
        elif sk in p_map and sk != voice.key:
            mod = torch.cos(2.0 * math.pi * p_map[sk].freq_hz * t)
        else:
            mod = torch.zeros(n, dtype=torch.float64)
        amp = amp * (1.0 + voice.am.depth_amp * mod)

    env = amp

    # --- manifold synthesis ---
    mt = voice.manifold_type
    if mt in ("harmonic", "harmonic_warp") and voice.harmonic_count > 1:
        sig = torch.zeros(n, dtype=torch.complex128)
        hc  = max(1, voice.harmonic_count)
        bri = voice.harmonic_brightness
        warp = voice.harmonic_warp_strength
        for k in range(1, hc + 1):
            h_ratio = k + warp * (k - 1)  # warp=0 → exact integer multiples
            h_amp   = 1.0 / (k ** bri) if bri > 0 else 1.0
            # C2 fix: k-th partial starts at k * phase_origin so all harmonics
            # are constructive at t=0 even when ratios are non-integer (warp > 0).
            h_phase = torch.cumsum(2.0 * math.pi * (f_inst * h_ratio) / sr, dim=0) + (k * voice.phase_origin) % (2.0 * math.pi)
            sig = sig + h_amp * torch.exp(1j * h_phase)
        # Normalise so amplitude 1 still means peak ~1 for a single harmonic baseline
        norm = sum(1.0 / (k ** bri) if bri > 0 else 1.0 for k in range(1, hc + 1))
        sig = sig / norm
        if voice.loop_enabled:
            sig = _apply_loop_tiling(sig, voice, n)
        out = env * sig
    else:
        sig = torch.exp(1j * phase)
        if voice.loop_enabled:
            sig = _apply_loop_tiling(sig, voice, n)
        out = env * sig

    # --- pre-delay: zero-pad the onset ---
    # The pre_delay is always relative to t=0; with a pre-roll (t_offset < 0),
    # the voice should be silent up to t = max(0, pre_delay), i.e. the first
    # abs(t_offset) samples are pre-roll so silence only applies in [0, pre_delay).
    if voice.pre_delay > 0.0:
        # Number of samples that sit before t=0 (the pre-roll prefix)
        preroll_n = max(0, int(round(abs(min(0.0, t_offset)) * sr)))
        # Silence from t=0 up to pre_delay (offset by the pre-roll prefix)
        silence_end = preroll_n + min(n, int(round(voice.pre_delay * sr)))
        out[preroll_n:silence_end] = 0.0

    return out


def _synthesize_lfo_csig(
    lfo: LFODefinition,
    n: int,
    sr: float,
    t_offset: float = 0.0,  # seconds before t=0 to start (pre-roll)
) -> np.ndarray:
    """Return an LFO as a complex128 analytic (Hilbert) signal.

    The imaginary part is the Hilbert transform of the real waveform so that:
    - ``extractor="magnitude"`` returns the true envelope, not |real|.
    - ``extractor="phase"`` returns the smooth analytic phase, not a degenerate
      square wave (M2 fix: previous version had imag=0 which caused
      angle(real + 0j) = 0 or π — a square wave rather than a smooth ramp).

    Sine LFOs have a natural analytic form: re + i·im = A·e^{iωt}.
    For other waveforms the analytic signal is approximated via the Hilbert
    transform of the real waveform using scipy.signal.hilbert when available,
    falling back to real-only (original behaviour) if scipy is absent.
    """
    t = (np.arange(n, dtype=np.float64) / max(sr, 1.0)) + t_offset
    real = _lfo_signal(lfo, t)  # float64 real waveform
    shape = getattr(lfo, "shape", "Sine")
    if shape == "Sine":
        # For a pure sine the analytic signal is exact: e^{i*(ωt + φ)}
        omega = 2.0 * math.pi * float(lfo.rate_hz)
        phi   = float(getattr(lfo, "phase_offset", 0.0))
        depth = float(getattr(lfo, "depth", 1.0))
        csig  = depth * np.exp(1j * (omega * t + phi))
        return csig.astype(np.complex128)
    # Non-sine shapes: attempt Hilbert lift
    try:
        from scipy.signal import hilbert as _hilbert
        analytic = _hilbert(real)
        return analytic.astype(np.complex128)
    except Exception:
        # Fallback: real-only (pre-M2 behaviour); phase extractor will be degenerate
        return real.astype(np.complex128)


def _synthesize_lfo_channel_csig(ch: dict, n: int, sr: float,
                                  t_offset: float = 0.0) -> np.ndarray:
    """Synthesize one LFO channel dict to a complex analytic signal (length n).

    ch keys: rate_hz, amplitude, phase_offset, shape, tension,
             resample (ZOH decimation factor), slew_order (1|2), slew (0-1).
    """
    rate_hz      = float(ch.get("rate_hz",      1.0))
    amplitude    = float(ch.get("amplitude",    1.0))
    phase_offset = float(ch.get("phase_offset", 0.0))
    shape        = ch.get("shape", "Sine")
    tension      = max(1e-3, float(ch.get("tension",    1.0)))
    resample     = max(1,    int(  ch.get("resample",   1)))
    slew_order   = max(1, min(2, int(ch.get("slew_order", 1))))
    slew_val     = float(np.clip(ch.get("slew", 0.0), 0.0, 0.9999))

    t  = (np.arange(n, dtype=np.float64) / max(sr, 1.0)) + t_offset
    ph = 2.0 * math.pi * rate_hz * t + phase_offset

    # Raw waveform (always real)
    if shape == "Sine":
        raw = np.sin(ph)
    elif shape == "Triangle":
        raw = 2.0 * np.abs(2.0 * (ph / (2 * math.pi) % 1.0) - 1.0) - 1.0
    elif shape == "Sawtooth":
        raw = 2.0 * (ph / (2 * math.pi) % 1.0) - 1.0
    else:  # Square
        raw = np.sign(np.sin(ph))

    # Tension shaping: sign(x)|x|^t
    shaped = np.sign(raw) * np.abs(raw) ** tension

    # Resample — zero-order hold (sample-and-hold decimation)
    if resample > 1:
        decimated = shaped[::resample]
        shaped = np.repeat(decimated, resample)[:n]
        if len(shaped) < n:
            pad = np.full(n - len(shaped), shaped[-1] if len(shaped) else 0.0)
            shaped = np.concatenate([shaped, pad])

    # Slew — exponential IIR low-pass, 1st or 2nd order
    # alpha=1 → passthrough; alpha→0 → DC (no response)
    # slew_val^2 gives a gentle nonlinear mapping so small values have effect
    if slew_val > 1e-6:
        alpha = (1.0 - slew_val) ** 2
        alpha = max(1e-6, alpha)
        try:
            from scipy.signal import lfilter as _lf
            b = [alpha]
            a = [1.0, -(1.0 - alpha)]
            shaped = _lf(b, a, shaped)
            if slew_order == 2:
                shaped = _lf(b, a, shaped)
        except Exception:
            y = float(shaped[0])
            out = np.empty(n)
            for k in range(n):
                y += alpha * (shaped[k] - y)
                out[k] = y
            if slew_order == 2:
                y2 = out[0]
                out2 = np.empty(n)
                for k in range(n):
                    y2 += alpha * (out[k] - y2)
                    out2[k] = y2
                out = out2
            shaped = out

    try:
        from scipy.signal import hilbert as _hilbert
        return (amplitude * _hilbert(shaped)).astype(np.complex128)
    except Exception:
        return (amplitude * shaped).astype(np.complex128)


def _sm_plugin_dir() -> str:
    """Return absolute path to the sm_plugins folder (sibling of this file)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "sm_plugins")


def _load_sm_plugin(plugin_name: str):
    """Import and return the plugin module for *plugin_name* (stem, no .py).

    Returns None if the plugin cannot be found or imported.
    """
    if not plugin_name:
        return None
    plugin_path = os.path.join(_sm_plugin_dir(), f"{plugin_name}.py")
    if not os.path.isfile(plugin_path):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"sm_plugin_{plugin_name}", plugin_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _sm_plugin_list() -> list:
    """Return sorted list of plugin name stems available in sm_plugins/."""
    d = _sm_plugin_dir()
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(d)
        if f.endswith(".py") and not f.startswith("_")
    )


def _sm_plugin_state_vars(plugin) -> list[str]:
    """Return normalized state/output variable names declared by a plugin."""
    vars_raw = getattr(plugin, "STATE_VARS", []) if plugin is not None else []
    vars_out: list[str] = []
    for v in vars_raw:
        s = str(v).strip()
        if s and s not in vars_out:
            vars_out.append(s)
    return vars_out


def _sm_plugin_output_vars(plugin) -> list[str]:
    """Return normalized output variable names declared by a plugin."""
    vars_raw = getattr(plugin, "OUTPUT_VARS", None) if plugin is not None else None
    if vars_raw is None:
        return _sm_plugin_state_vars(plugin)
    vars_out: list[str] = []
    for v in vars_raw:
        s = str(v).strip()
        if s and s not in vars_out:
            vars_out.append(s)
    return vars_out


def _sm_plugin_item_names(plugin, n_items: int) -> list[str]:
    """Return item names for a plugin, honoring optional naming helpers."""
    n = max(1, int(n_items))
    if plugin is None:
        return [f"m{i}" for i in range(n)]
    item_names_fn = getattr(plugin, "item_names", None)
    if callable(item_names_fn):
        try:
            names = [str(x).strip() for x in item_names_fn(n)]
            names = [x for x in names if x]
            if len(names) == n and len(set(names)) == n:
                return names
        except Exception:
            pass
    prefix = str(getattr(plugin, "ITEM_PREFIX", "m")).strip() or "m"
    return [f"{prefix}{i}" for i in range(n)]


def _sm_plugin_param_specs(plugin) -> list[dict]:
    """Return normalized plugin parameter specs for state-machine modules."""
    specs = getattr(plugin, "PARAM_SPECS", []) if plugin is not None else []
    out: list[dict] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "")).strip()
        if not name:
            continue
        dtype = str(spec.get("dtype", "float")).strip() or "float"
        out.append({
            "name": name,
            "label": str(spec.get("label", name)),
            "dtype": dtype,
            "default": spec.get("default", 0.0 if dtype != "choice" else ""),
            "low": float(spec.get("low", 0.0)),
            "high": float(spec.get("high", 1.0)),
            "fmt": str(spec.get("fmt", ".3g")),
            "is_log": bool(spec.get("is_log", False)),
            "choices": [str(c) for c in spec.get("choices", [])],
            "group": str(spec.get("group", "Plugin")),
        })
    return out


def _sm_plugin_log_text(log_payload: object) -> str:
    """Normalize optional plugin-provided log payloads into display text."""
    if log_payload is None:
        return ""
    if isinstance(log_payload, str):
        return log_payload
    if isinstance(log_payload, (list, tuple)):
        lines = [str(x) for x in log_payload if x is not None]
        return "\n".join(lines)
    return str(log_payload)


def _sm_plugin_default_params(plugin) -> dict[str, object]:
    """Return default parameter values for a state-machine plugin."""
    return {spec["name"]: spec["default"] for spec in _sm_plugin_param_specs(plugin)}


def _sm_to_complex(arr) -> np.ndarray:
    """Convert a real float64 array (or torch Tensor) to complex128 for routing."""
    if hasattr(arr, "detach"):          # torch tensor
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64).astype(np.complex128)


def _sm_wrap(arr, use_torch: bool):
    """Return arr as a torch Tensor if use_torch and torch is available."""
    if use_torch:
        try:
            import torch
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr)
        except ImportError:
            pass
    return arr


def _sm_unwrap(val) -> np.ndarray:
    """Convert torch Tensor or ndarray to ndarray preserving complex dtype."""
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    arr = np.asarray(val)
    if np.iscomplexobj(arr):
        return np.asarray(arr, dtype=np.complex128)
    return np.asarray(arr, dtype=np.float64)


def _place_signal(z: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray,
                  distance: np.ndarray, width: np.ndarray,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Spatialize one complex analytic signal into a (ch1_out, ch2_out) pair.

    Everything stays complex throughout.  The per-ear rotation mixes Re and Im
    of the input before producing each output — no component is discarded early.

    azimuth  [-1, 1]  left=−1, center=0, right=+1
    elevation[-1, 1]  down=−1, up=+1  (secondary tilt)
    distance [0, 1]   amplitude fall-off
    width    [0, 1]   angular spread between ears
    """
    dist_gain = 1.0 / (1.0 + np.clip(distance, 0.0, 1.0) * 4.0)
    z_s = z * dist_gain

    # Map azimuth to a center rotation angle; width spreads the two ears apart.
    # theta=0 → phase unchanged (Re dominant), theta=π/2 → Im dominant.
    # Equal-power panning arises naturally from the projection's Re read.
    az   = np.clip(azimuth,  -1.0, 1.0)
    el   = np.clip(elevation, -1.0, 1.0)
    half_w = np.clip(width, 0.0, 1.0) * (math.pi / 4.0)

    theta_c = (az + 1.0) * (math.pi / 4.0)        # 0 at full-left, π/2 at full-right
    el_tilt  = el * (math.pi / 8.0)                # ±π/8 elevation modifier

    theta_ch1 = theta_c - half_w + el_tilt
    theta_ch2 = theta_c + half_w - el_tilt

    ch1_out = z_s * np.exp(1j * theta_ch1)
    ch2_out = z_s * np.exp(1j * theta_ch2)
    return ch1_out, ch2_out


def _apply_projection(csig: np.ndarray, mode: str, rotation_hz: float,
                      sr: float) -> tuple[np.ndarray, np.ndarray]:
    """Project a complex analytic signal to a stereo (L, R) float32 pair.

    *rotation_hz* continuously rotates the analytic projection plane:
        z_rot(t) = csig(t) * exp(i * 2π * rotation_hz * t)
    This is a pure phase-rotation (frequency shift of rotation_hz Hz) applied
    before the spatial mode decode — all modes share this single knob.

    Modes
    -----
    mono              L = R = Re(z_rot)
    stereo_quadrature L = Re(z_rot), R = Im(z_rot)   (classic analytic stereo)
    stereo_ms         L = Re+Im, R = Re-Im            (mid–side)
    lissajous         amplitude-pan from instantaneous phase angle
    """
    n = len(csig)
    if n == 0:
        z = np.zeros(1, dtype=np.float32)
        return z, z
    t = np.arange(n, dtype=np.float64) / max(sr, 1.0)
    if rotation_hz != 0.0:
        csig = csig * np.exp(1j * (2.0 * math.pi * rotation_hz * t))
    re = np.asarray(csig.real, dtype=np.float32)
    im = np.asarray(csig.imag, dtype=np.float32)
    if mode == "stereo_quadrature":
        return re, im
    elif mode == "stereo_ms":
        return (re + im).astype(np.float32), (re - im).astype(np.float32)
    elif mode == "lissajous":
        phase = np.angle(csig).astype(np.float32)           # [-π, π]
        pan   = phase / math.pi                              # [-1, 1]
        env   = np.abs(csig).astype(np.float32)
        l_gain = np.clip(1.0 - pan, 0.0, 2.0).astype(np.float32) * 0.5
        r_gain = np.clip(1.0 + pan, 0.0, 2.0).astype(np.float32) * 0.5
        return env * l_gain, env * r_gain
    else:  # "mono" and fallback
        return re, re.copy()


def _auto_mix_signal_keys(patch: "AnalyticPatch") -> list[str]:
    """Return the signal-producing node keys eligible for legacy auto-mix."""
    return (
        [v.key for v in patch.voices] +
        [l.key for l in patch.lfos] +
        [m.key for m in patch.modules
         if m.module_type not in ("interaural",)
         and not (m.module_type == "lfo" and m.lfo_channels)] +
        [m.lfo_ch_key(i)
         for m in patch.modules if m.module_type == "lfo" and m.lfo_channels
         for i in range(len(m.lfo_channels))]
    )


def _working_routing_graph_for_synthesis(patch: "AnalyticPatch") -> RoutingGraph:
    """Return a non-mutating routing graph for preview / render solves.

    Legacy patches with an entirely empty signal-routing graph still need a
    default source->mix path so they remain audible. Once the user has created
    any explicit signal edges, missing edges stay missing; deleted routes are
    not silently reintroduced during preview or rendering.
    """
    if getattr(patch, "routers", None):
        g = RoutingGraph()
        base = RoutingGraph.from_dict(patch.routing.to_dict())
        for nk in base.node_keys():
            g.add_node(
                nk,
                node_type=base.get_node_type(nk),
                source_router_types=(
                    base.get_node_source_router_types(nk)
                    if nk in base.node_source_router_types else None
                ),
                sink_router_types=(
                    base.get_node_sink_router_types(nk)
                    if nk in base.node_sink_router_types else None
                ),
            )
        g.edges.extend(copy.copy(e) for e in base.edges)
        g.param_edges.extend(copy.copy(pe) for pe in base.param_edges)
        g.meta_edges.extend(copy.copy(me) for me in getattr(base, "meta_edges", []))
        g.feedback = copy.deepcopy(base.feedback)
        g.latency_compensation = bool(getattr(base, "latency_compensation", False))
        # New-model patches: merge deployed router graphs into one synthesis
        # graph so the render path can consume the same router instances the
        # editor exposes.  Feedback policy remains graph-global for now.
        for ri, router in enumerate(patch.routers):
            rg = RoutingGraph.from_dict(router.graph.to_dict())
            if ri == 0 and not base.edges and not base.param_edges:
                g.feedback = copy.deepcopy(rg.feedback)
                g.latency_compensation = bool(getattr(rg, "latency_compensation", False))
            for nk in rg.node_keys():
                g.add_node(
                    nk,
                    node_type=rg.get_node_type(nk),
                    source_router_types=(
                        rg.get_node_source_router_types(nk)
                        if nk in rg.node_source_router_types else None
                    ),
                    sink_router_types=(
                        rg.get_node_sink_router_types(nk)
                        if nk in rg.node_sink_router_types else None
                    ),
                )
            g.edges.extend(copy.copy(e) for e in rg.edges)
            g.param_edges.extend(copy.copy(pe) for pe in rg.param_edges)
            g.meta_edges.extend(copy.copy(me) for me in getattr(rg, "meta_edges", []))
    else:
        g = RoutingGraph.from_dict(patch.routing.to_dict())
    mixer_keys = [m.key for m in patch.mixers]
    default_mix_key = mixer_keys[0] if mixer_keys else "__mix__"
    auto_signal_keys = _auto_mix_signal_keys(patch)
    if not g.edges:
        g.ensure_defaults(auto_signal_keys, mix_key=default_mix_key)
    _sanitize_system_io_edges(g, patch)
    g.prune()
    return g


def _synthesize_voice_sources(
    patch: "AnalyticPatch",
    lfo_map: dict,
    p_map: dict,
    *,
    n_samples: int,
    t_offset_map: dict[str, float] | None = None,
    voice_param_overrides: dict | None = None,
    param_series: dict | None = None,
    file_render: bool = False,
    granular_seed_offset: int = 0,
) -> dict[str, np.ndarray]:
    import torch
    device = torch.device("cpu")
    sr = float(patch.preview_sr)

    # Determine muted voice keys (solo exclusion)
    solo_key = getattr(patch, "solo_key", None)
    muted_keys: set[str] = set()
    for v in patch.voices:
        if v.muted or (solo_key is not None and v.key != solo_key):
            muted_keys.add(v.key)

    # Flat performer list from placement solver (may be empty)
    all_performers = [
        pf
        for pt in getattr(patch, "parts", [])
        for ch in getattr(pt, "chairs", [])
        for pf in getattr(ch, "performers", [])
    ]

    note_hz = float(patch._seq_note_hz) if getattr(patch, "_seq_note_hz", 0.0) > 0 else float(patch.seq_tonic_hz)

    cfg, performers, driver_list = build_driver_config(
        patch, all_performers, device, sr,
        note_hz=note_hz,
        muted_voice_keys=muted_keys,
    )

    voices = list(patch.voices)
    voice_sigs: dict[str, np.ndarray] = {}

    # Zero-fill all voices (muted or absent from driver_list)
    for v in voices:
        voice_sigs[v.key] = np.zeros(n_samples, dtype=np.complex128)

    if cfg.D == 0:
        return voice_sigs

    # Build initial state; apply t_offset (pre-roll) to t_pos per driver
    state = init_driver_state(cfg)
    if t_offset_map:
        for d, (p_idx, v_idx) in enumerate(driver_list):
            vk = voices[v_idx].key
            offset_s = float((t_offset_map or {}).get(vk, 0.0))
            if offset_s != 0.0:
                state.t_pos[d] = state.t_pos[d] + offset_s

    driver_out, voice_out, _ = multi_level_driver_step(cfg, state, n_samples, sr)

    # voice_out: (V, T) — accumulated per voice across all its drivers
    for vi, v in enumerate(voices):
        if v.key not in muted_keys:
            voice_sigs[v.key] = voice_out[vi].numpy()

    return voice_sigs


def _build_sequence_driver_config(
    patch: "AnalyticPatch",
    play_groups: "list[tuple]",
    sr: float,
    device,
) -> "tuple[DriverConfig, list, DriverState]":
    """Build a DriverConfig with one driver slot per (note_event × voice).

    Each driver encodes note start time by initialising ``t_pos = -start_time``
    so that the envelope phase-zero aligns exactly with the note's onset sample.
    ``pre_delay_samples`` is left at 0 — the negative ``t_pos`` already gates
    output via the ``sample_global >= pre_delay_samples`` mask in
    ``driver_synthesis_step``.

    Returns ``(cfg, patch.voices, initial_state)``.
    """
    import copy as _copy
    from performer_engine import CHIRP_NONE as _CHIRP_NONE_PE

    voices = list(patch.voices)
    V = len(voices)
    voice_key_to_idx = {v.key: i for i, v in enumerate(voices)}
    root_hz = float(patch.seq_tonic_hz)

    # Determine padded H and K across all voices
    H_max = 1
    K_max = 5
    for v in voices:
        h_r, _ = _build_harmonics(v)
        H_max = max(H_max, len(h_r))
        K_max = max(K_max, len(v.active_knots()))

    f0_list, amplitude_list, phase_origin_list = [], [], []
    pre_delay_samp_list, note_dur_list, active_list = [], [], []
    chirp_type_list, chirp_fs_list, chirp_fe_list = [], [], []
    chirp_tau_list, chirp_pow_list = [], []
    h_ratios_list, h_amps_list, n_harmonics_list = [], [], []
    env_t_list, env_v_list, env_n_list = [], [], []
    voice_idx_list, instrument_idx_list = [], []
    fm_src_list, fm_depth_list = [], []
    am_src_list, am_depth_list = [], []
    t_pos_init_list: list[float] = []   # initial t_pos per driver

    perf_idx = 0  # monotone instrument_idx counter across all note×voice slots

    for grp_sched, grp_voices in play_groups:
        prev_hz: "float | None" = None
        for event in grp_sched.events:
            start_time = float(event.start_time)
            for src_v in grp_voices:
                vi = voice_key_to_idx.get(src_v.key)
                if vi is None:
                    continue

                v = _copy.copy(src_v)
                v.amplitude = float(src_v.amplitude) * float(event.velocity)

                # Resolve frequency for this note event
                if getattr(event, "_exact_pitch", False):
                    resolved_hz = float(event.fundamental_hz)
                else:
                    resolved_hz = _resolve_voice_hz(src_v, patch.tuning, event.fundamental_hz)
                    _role = getattr(src_v, "seq_role", "melody")
                    if _role == "bass":
                        resolved_hz *= (2.0 ** patch.seq_bass_octave)
                    elif _role == "root":
                        resolved_hz = patch.seq_tonic_hz * (2.0 ** patch.seq_root_octave)
                    elif _role == "stab":
                        resolved_hz *= (2.0 ** patch.seq_stab_octave)

                v.freq_hz = resolved_hz
                v.pre_delay = 0.0

                # Portamento chirp between consecutive notes in the same group
                if (patch.seq_portamento_s > 0 and prev_hz is not None
                        and abs(prev_hz - resolved_hz) > 0.5):
                    v.chirp = ChirpSpec(
                        chirp_type    = "exponential",
                        f_delta_start = prev_hz - resolved_hz,
                        f_delta_end   = 0.0,
                        tau           = max(patch.seq_portamento_s, 0.001),
                    )
                else:
                    v.chirp = ChirpSpec()

                chirp = getattr(v, "chirp", None)
                f0 = _resolve_f0(v, resolved_hz, root_hz)
                h_r, h_a = _build_harmonics(v)
                n_h = len(h_r)
                env_t_secs, env_v = _build_env_knots(v, event.duration_s)
                n_k = len(env_t_secs)
                last_v = env_v[-1] if env_v else 0.0

                f0_list.append(f0)
                amplitude_list.append(float(v.amplitude))
                phase_origin_list.append(float(v.phase_origin))
                pre_delay_samp_list.append(0)          # gated by t_pos < 0
                note_dur_list.append(float(event.duration_s))
                active_list.append(True)

                chirp_code = _CHIRP_CODE.get(
                    getattr(chirp, "chirp_type", "none"), CHIRP_NONE)
                chirp_type_list.append(chirp_code)
                chirp_fs_list.append(float(getattr(chirp, "f_delta_start", 0.0)) if chirp else 0.0)
                chirp_fe_list.append(float(getattr(chirp, "f_delta_end",   0.0)) if chirp else 0.0)
                chirp_tau_list.append(float(getattr(chirp, "tau",           0.5)) if chirp else 0.5)
                chirp_pow_list.append(float(getattr(chirp, "chirp_power",   1.0)) if chirp else 1.0)

                h_ratios_list.append(h_r + [0.0] * (H_max - n_h))
                h_amps_list.append(h_a   + [0.0] * (H_max - n_h))
                n_harmonics_list.append(n_h)

                env_t_list.append(env_t_secs + [1e30]   * (K_max - n_k))
                env_v_list.append(env_v      + [last_v] * (K_max - n_k))
                env_n_list.append(n_k)

                voice_idx_list.append(vi)
                instrument_idx_list.append(perf_idx)

                fm = getattr(v, "fm", None)
                fm_vi = (voice_key_to_idx.get(fm.source_key, -1)
                         if fm and getattr(fm, "source_key", "") else -1)
                fm_src_list.append(fm_vi)
                fm_depth_list.append(float(fm.depth_hz) if fm else 0.0)

                am = getattr(v, "am", None)
                am_vi = (voice_key_to_idx.get(am.source_key, -1)
                         if am and getattr(am, "source_key", "") else -1)
                am_src_list.append(am_vi)
                am_depth_list.append(float(am.depth_amp) if am else 0.0)

                # Negative t_pos so envelope onset aligns with note start sample
                t_pos_init_list.append(-start_time)
                perf_idx += 1

            # Track the last resolved hz for portamento (first voice governs)
            if grp_voices:
                if getattr(event, "_exact_pitch", False):
                    prev_hz = float(event.fundamental_hz)
                else:
                    prev_hz = _resolve_voice_hz(
                        grp_voices[0], patch.tuning, event.fundamental_hz)

    D = len(f0_list)
    if D == 0:
        from patch_to_driver import _empty_config as _ec
        empty = _ec(V, device)
        return empty, voices, init_driver_state(empty)

    def _ft(lst):  return torch.tensor(lst, dtype=torch.float64, device=device)
    def _it(lst):  return torch.tensor(lst, dtype=torch.int64,   device=device)
    def _bt(lst):  return torch.tensor(lst, dtype=torch.bool,    device=device)
    def _ft2(lst): return torch.tensor(lst, dtype=torch.float64, device=device)

    cfg = DriverConfig(
        D=D, H=H_max, K=K_max, V=V, device=device,
        f0                = _ft(f0_list),
        amplitude         = _ft(amplitude_list),
        phase_origin      = _ft(phase_origin_list),
        pre_delay_samples = _it(pre_delay_samp_list),
        note_duration     = _ft(note_dur_list),
        active            = _bt(active_list),
        chirp_type        = _it(chirp_type_list),
        chirp_f_start     = _ft(chirp_fs_list),
        chirp_f_end       = _ft(chirp_fe_list),
        chirp_tau         = _ft(chirp_tau_list),
        chirp_power       = _ft(chirp_pow_list),
        h_ratios          = _ft2(h_ratios_list),
        h_amps            = _ft2(h_amps_list),
        n_harmonics       = _it(n_harmonics_list),
        env_t             = _ft2(env_t_list),
        env_v             = _ft2(env_v_list),
        env_n             = _it(env_n_list),
        voice_idx         = _it(voice_idx_list),
        instrument_idx    = _it(instrument_idx_list),
        fm_source_voice   = _it(fm_src_list),
        fm_depth_hz       = _ft(fm_depth_list),
        am_source_voice   = _it(am_src_list),
        am_depth          = _ft(am_depth_list),
    )

    # Seed state: t_pos = -start_time per driver so envelope zero-phase aligns
    # with note onset. Phase accumulator seeded from phase_origin as usual.
    state = init_driver_state(cfg)
    state.t_pos = torch.tensor(t_pos_init_list, dtype=torch.float64, device=device)

    return cfg, voices, state


def _synthesize_sequence_full_batch(
    patch: "AnalyticPatch",
    play_groups: "list[tuple]",
    total_n: int,
    *,
    file_render: bool = False,
    out_channels: int = 2,
    persistent_aux: "dict[str, dict] | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, dict]":
    """Sample-wise causal step solver for full-sequence synthesis.

    Signal chain executed once per sample (all D drivers batched):

      1. Advance all D drivers by 1 sample via ``driver_synthesis_step``
         (FM cross-voice resolved by ``multi_level_driver_step`` topo sort).
         Atmospheric FM offset from the previous sample is injected here.
      2. Routing solve: x = M @ src  (M = (I − W)⁻¹ pre-computed once).
      3. Step all SM modules (room/body physics) on the 1-sample signals.
         SM state (CavityScene, stream state, body states) persists across
         samples — IR tails and resonance accumulate correctly.
      4. Extract ``feedback_pressure`` from each SM item → FM offset for
         the next sample's driver advance.  This is the sympathetic chirp
         coupling: atmosphere → driver → voice.

    Stereo projection is applied once, after the loop completes.

    Returns
    -------
    (left, right, output_channels, updated_persistent_aux)
    """
    import copy as _copy, time as _time, io, contextlib

    sr = float(patch.preview_sr)
    if persistent_aux is None:
        persistent_aux = {}

    device = torch.device("cpu")

    # ── 1. Build full-sequence DriverConfig ────────────────────────────────────
    cfg, voices, drv_state = _build_sequence_driver_config(
        patch, play_groups, sr, device)

    if cfg.D == 0:
        z = np.zeros(total_n)
        return z, z, np.zeros((total_n, out_channels)), dict(persistent_aux)

    f0_base = cfg.f0.clone()   # save nominal per-driver frequencies

    # ── 2. Compile torch routing solve once ───────────────────────────────────
    g = _working_routing_graph_for_synthesis(patch)
    node_keys = _patch_node_keys(patch)
    for _k in g.node_keys():
        if _k not in node_keys:
            node_keys.append(_k)
    N = len(node_keys)
    ki = {k: i for i, k in enumerate(node_keys)}
    global_decay = float(getattr(patch.routing, "global_decay", 1.0))
    compiled_router = CompiledRouter(
        node_keys,
        list(g.edges),
        sr,
        device,
        global_decay=global_decay,
        batch_size=1,
        max_iterations=max(1, int(getattr(g.feedback, "max_iterations", 64))),
        convergence_eps=1e-10,
        infinity_threshold=1e6,
    )

    # ── 3. Pre-compute LFO signals (deterministic, no feedback) ───────────────
    lfo_sigs: dict[str, np.ndarray] = {}
    for lfo in patch.lfos:
        lfo_sigs[lfo.key] = _synthesize_lfo_csig(lfo, total_n, sr)

    # ── 4. Initialise SM modules, seed room/body caches ───────────────────────
    sm_mods: list = []
    sm_plugs: list = []
    for m in patch.modules:
        if m.module_type != "state_machine" or m.muted:
            sm_mods.append(None)
            sm_plugs.append(None)
            continue
        mc = _copy.copy(m)
        mc._sm_state     = {}
        mc._sm_out_cache = {}   # {out_key: 1-element complex128 array}
        mc._sm_log_text  = ""
        mc._sm_aux_state = dict(persistent_aux.get(m.key, {}))
        sm_mods.append(mc)
        plug = _load_sm_plugin(mc.sm_plugin)
        sm_plugs.append(plug if (plug and callable(getattr(plug, "step", None))) else None)

    # Per-SM-module: list of (src_key, slot) tuples.
    # slot = edge.item_slot if set, else src_key.  The SM receives each
    # signal under `slot` so driver/instrument identity is preserved.
    sm_input_src: list[list[tuple]] = []
    for mc in sm_mods:
        if mc is None:
            sm_input_src.append([])
            continue
        sm_input_src.append([
            (e.src_key, e.item_slot if e.item_slot else e.src_key)
            for e in g.edges
            if e.dst_key == mc.key and e.src_key in ki
        ])

    # Voice index in DriverConfig's V dimension → node key mapping
    voice_list = voices   # list of VoiceDefinition objects, indexed by v_idx

    # ── 5. Sample-wise step loop ───────────────────────────────────────────────
    X_full = np.zeros((N, total_n), dtype=np.complex128)
    # Atmospheric FM offset per driver (Hz); fed back from SM feedback_pressure
    atm_fm_hz = torch.zeros(cfg.D, dtype=torch.float64, device=device)

    t0_wall = _time.monotonic()
    log_interval = max(1, total_n // 20)

    for t in range(total_n):
        # 5a. Inject atmospheric FM and advance all D drivers by 1 sample.
        #     multi_level_driver_step resolves FM/AM cross-voice topo ordering.
        cfg.f0 = f0_base + atm_fm_hz
        _d_out, v_out_t, drv_state = multi_level_driver_step(cfg, drv_state, 1, sr)
        # v_out_t: (V, 1) complex128 — per-voice signal for this sample

        # 5b. Populate Src vector: voices + LFOs + SM cached outputs
        Src_vec = np.zeros(N, dtype=np.complex128)
        for vi, v in enumerate(voice_list):
            if v.key in ki:
                Src_vec[ki[v.key]] = v_out_t[vi, 0].item()
        for lkey, lsig in lfo_sigs.items():
            if lkey in ki:
                Src_vec[ki[lkey]] = lsig[t]
        for mc in sm_mods:
            if mc is None:
                continue
            for ok, cached in mc._sm_out_cache.items():
                if ok in ki and len(cached) > 0:
                    Src_vec[ki[ok]] = cached[0]

        # 5c. Routing solve — single matrix-vector multiply (M pre-computed)
        X_t = compiled_router.step(
            torch.as_tensor(Src_vec, dtype=torch.complex128, device=device)
        )
        X_vec = np.asarray(X_t.detach().cpu().numpy(), dtype=np.complex128)
        X_full[:, t] = X_vec

        # 5d. Step each SM module on this 1-sample X; update atmospheric FM
        atm_fm_hz.zero_()
        for idx, (mc, plug) in enumerate(zip(sm_mods, sm_plugs)):
            if mc is None or plug is None:
                continue
            sm_inputs = {
                slot: _sm_wrap(
                    np.array([X_vec[ki[src_key]]], dtype=np.complex128),
                    mc.sm_use_torch)
                for src_key, slot in sm_input_src[idx]
            }
            _sm_stdout = io.StringIO()
            _sm_stderr = io.StringIO()
            try:
                with contextlib.redirect_stdout(_sm_stdout), \
                     contextlib.redirect_stderr(_sm_stderr):
                    _sm_traj = plug.step(
                        sm_inputs, dict(mc._sm_state), 1.0 / sr,
                        n_items=mc.sm_n_items,
                        use_torch=mc.sm_use_torch,
                        params=dict(mc.sm_params),
                        plugin_state=dict(mc._sm_aux_state or {}),
                    )
            except Exception:
                continue

            if not (isinstance(_sm_traj, dict) and "outputs" in _sm_traj):
                continue

            sm_out     = _sm_traj.get("outputs", {})
            mc._sm_state     = _sm_traj.get("state", {})
            mc._sm_aux_state = dict(_sm_traj.get("plugin_state", {}) or {})

            # Cache 1-sample outputs for the next iteration's Src population
            new_cache: dict = {}
            for item in mc.sm_items:
                for var in mc.sm_vars:
                    ok = mc.sm_out_key(item, var)
                    traj_val = sm_out.get(item, {}).get(var)
                    if traj_val is not None:
                        arr = np.atleast_1d(
                            np.asarray(_sm_unwrap(traj_val), dtype=np.complex128))
                        new_cache[ok] = arr[:1]
            mc._sm_out_cache = new_cache

            # Extract feedback_pressure per item → FM offset for each driver.
            # Instantaneous frequency from complex: ω = angle(z) * sr / 2π.
            # instrument_idx maps each driver slot to the SM item that owns it.
            for i, item in enumerate(mc.sm_items):
                fp = sm_out.get(item, {}).get("feedback_pressure")
                if fp is None:
                    continue
                fp_arr = np.atleast_1d(
                    np.asarray(_sm_unwrap(fp), dtype=np.complex128))
                if len(fp_arr) == 0:
                    continue
                fb_hz = float(np.angle(fp_arr[0])) * (sr / (2.0 * math.pi))
                mask = cfg.instrument_idx == i
                if mask.any():
                    atm_fm_hz[mask] += fb_hz

        if (t + 1) % log_interval == 0 or t == total_n - 1:
            elapsed = _time.monotonic() - t0_wall
            rate = (t + 1) / max(elapsed, 1e-6)
            eta  = (total_n - t - 1) / max(rate, 1e-6)
            print(f"  Step solver: {t+1}/{total_n} samples "
                  f"({100*(t+1)/total_n:.0f}%)  "
                  f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s")

    # ── 6. Stereo projection — once, after the loop ────────────────────────────
    active_mixer_keys = [m.key for m in patch.mixers if m.projection_active]
    sys_out_keys = [k for k in _system_output_keys(patch) if k in ki]
    has_system_routing = bool(sys_out_keys) and any(
        e.dst_key in set(sys_out_keys) for e in g.edges
    )

    if has_system_routing:
        out_bus = np.column_stack([
            X_full[ki[k], :total_n].real for k in sys_out_keys
        ])
        if patch.normalize_output and out_bus.size:
            peak = float(np.max(np.abs(out_bus)))
            if peak > 1e-9:
                out_bus /= peak
        left  = out_bus[:, 0].astype(np.float64)
        right = (out_bus[:, 1] if out_bus.shape[1] > 1 else out_bus[:, 0]).astype(np.float64)
    elif active_mixer_keys:
        mix = sum(X_full[ki[mk]] for mk in active_mixer_keys if mk in ki)
        if patch.normalize_output:
            peak = float(np.max(np.abs(mix)))
            if peak > 1e-9:
                mix /= peak
        left, right = _apply_projection(mix, patch.projection_mode,
                                        patch.projection_rotation_hz, sr)
        out_bus = np.column_stack([left, right])
    else:
        out_bus = np.zeros((total_n, max(2, out_channels)))
        left  = out_bus[:, 0]
        right = out_bus[:, 1]

    if out_bus.ndim == 2 and out_bus.shape[1] < out_channels:
        out_bus = np.pad(out_bus, ((0, 0), (0, out_channels - out_bus.shape[1])))
    elif out_bus.ndim == 1:
        out_bus = np.column_stack([left, right])

    # Harvest updated SM aux state (cavity/room scene caches)
    updated_aux: dict = dict(persistent_aux)
    for mc in sm_mods:
        if mc is not None and mc.key and getattr(mc, "_sm_aux_state", None):
            updated_aux[mc.key] = dict(mc._sm_aux_state)

    elapsed_total = _time.monotonic() - t0_wall
    print(f"  Step-solver complete: {total_n} samples, "
          f"{cfg.D} drivers in {elapsed_total:.1f}s")

    return (
        np.asarray(left),
        np.asarray(right),
        out_bus.astype(np.float64),
        updated_aux,
    )


def _synthesize_patch(
    patch: AnalyticPatch,
    *,
    granular_seed_offset: int = 0,
    file_render: bool = False,
    _return_mixer_sigs: bool = False,
    _return_sidecar: bool = False,
    _return_output_channels: bool = False,
    _prebuilt_voice_sigs: "dict[str, np.ndarray] | None" = None,
) -> tuple:
    """Return (left, right) float32 stereo after routing + projection.

    Routing model (N nodes = voices + LFOs + __mix__):
        x = src + W @ x  →  x = (I − W)⁻¹ · src          (instantaneous)
        x(t) = src(t) + decay · W @ x(t − d)               (delayed)

    When no routing edges exist, falls back to direct voice sum (backward-compat).
    The '__mix__' node output is the stereo output before projection.

    *_prebuilt_voice_sigs*: when provided, skip ``_synthesize_voice_sources`` and
    use these pre-assembled full-timeline buffers directly.  This is the entry
    point for the batched-sequence pipeline where all notes have been pre-rendered
    and accumulated into per-voice complex128 arrays before the routing solve.

    When *_return_sidecar=True* the return value gains a trailing SidecarBus:
        (left, right)                           default
        (left, right, output_channels)          _return_output_channels=True
        (left, right, sidecar)                  _return_sidecar=True
        (left, right, mixer_outs)               _return_mixer_sigs=True
        (left, right, mixer_outs, sidecar)      mixer + sidecar
    """
    lfo_map = {l.key: l for l in patch.lfos}
    p_map   = {p.key: p for p in patch.voices}
    sr      = float(patch.preview_sr)
    n       = int(patch.preview_sr * patch.duration)

    # 1. Synthesize independent sources for each node.
    # Build param overrides from the PREVIOUS frame's cached param series so that
    # voices can be synthesised with modulation in a single solve pass.
    # One-buffer latency is perceptually invisible at audio buffer sizes.
    _cached_ps = patch._param_series_cache   # {} on first frame
    _iau_mod_keys_set: set = {m.key for m in patch.modules if m.module_type == "interaural"}
    _iau_ch_to_mod_map: dict = {}
    for _m in patch.modules:
        if _m.module_type == "interaural":
            _iau_ch_to_mod_map[_m.ch1_key()] = _m
            _iau_ch_to_mod_map[_m.ch2_key()] = _m
            _iau_ch_to_mod_map[_m.key]       = _m
    _voice_param_ov: dict = {}
    _module_param_ov: dict = {}
    _routing_param_ov: dict = {}
    _mixer_keys = {m.key for m in patch.mixers}
    if patch.param_nodes and _cached_ps:
        for _pn in patch.param_nodes:
            if _pn.key not in _cached_ps:
                continue
            for _tgt in _pn.targets:
                _vk = _tgt.get("voice_key", "")
                _at = _tgt.get("attr", "")
                if not (_vk and _at):
                    continue
                if _vk in _iau_ch_to_mod_map:
                    _mm = _iau_ch_to_mod_map[_vk]
                    _module_param_ov.setdefault(_mm.key, {})[_at] = _cached_ps[_pn.key]
                elif _vk in _mixer_keys:
                    _routing_param_ov[_at] = _cached_ps[_pn.key]
                elif _vk not in _iau_mod_keys_set:
                    _voice_param_ov.setdefault(_vk, {})[_at] = _cached_ps[_pn.key]

    voice_sigs = (
        _prebuilt_voice_sigs
        if _prebuilt_voice_sigs is not None
        else _synthesize_voice_sources(
            patch,
            lfo_map,
            p_map,
            n_samples=n,
            voice_param_overrides=_voice_param_ov,
            param_series=_cached_ps or None,
            file_render=file_render,
            granular_seed_offset=granular_seed_offset,
        )
    )

    lfo_sigs: dict[str, np.ndarray] = {}
    for lfo in patch.lfos:
        lfo_sigs[lfo.key] = _synthesize_lfo_csig(lfo, n, sr)

    sys_in_sigs: dict[str, np.ndarray] = {}
    _sys_inputs = _system_input_keys(patch)
    _sys_bufs = list(getattr(patch.system_audio, "_input_buffers", []))
    for _i, _k in enumerate(_sys_inputs):
        if _i < len(_sys_bufs):
            _buf = np.asarray(_sys_bufs[_i], dtype=np.float32).reshape(-1)
            if len(_buf) >= n:
                _arr = _buf[-n:]
            elif len(_buf) > 0:
                _arr = np.pad(_buf, (0, n - len(_buf)), mode="constant")
            else:
                _arr = np.zeros(n, dtype=np.float32)
        else:
            _arr = np.zeros(n, dtype=np.float32)
        sys_in_sigs[_k] = _arr.astype(np.complex128)

    # Module signals — LFO type reuses the LFO synthesiser; passthrough starts
    # at zero (its signal comes entirely from routing edges).
    # Multi-channel LFO: main key gets zero; each channel key gets its own signal.
    module_sigs: dict[str, np.ndarray] = {}
    for mod in patch.modules:
        if mod.muted:
            module_sigs[mod.key] = np.zeros(n, dtype=np.complex128)
            if mod.module_type == "lfo" and mod.lfo_channels:
                for i in range(len(mod.lfo_channels)):
                    module_sigs[mod.lfo_ch_key(i)] = np.zeros(n, dtype=np.complex128)
        elif mod.module_type == "lfo":
            if mod.lfo_channels:
                module_sigs[mod.key] = np.zeros(n, dtype=np.complex128)
                for i, ch in enumerate(mod.lfo_channels):
                    module_sigs[mod.lfo_ch_key(i)] = _synthesize_lfo_channel_csig(ch, n, sr)
            else:
                _tmp = LFODefinition(key=mod.key, label=mod.label,
                                     rate_hz=mod.rate_hz, shape=mod.shape,
                                     phase_offset=mod.phase_offset, depth=mod.depth)
                module_sigs[mod.key] = _synthesize_lfo_csig(_tmp, n, sr)
        elif mod.module_type == "state_machine":
            module_sigs[mod.key] = np.zeros(n, dtype=np.complex128)
            # State machine output nodes: seed Src from last-frame cache so that
            # downstream routing sees these values in the single solve pass.
            for ok in mod.sm_out_keys():
                cached_sig = mod._sm_out_cache.get(ok)
                if cached_sig is not None:
                    Tpad = n
                    c = cached_sig[:Tpad] if len(cached_sig) >= Tpad else np.pad(
                        cached_sig, (0, Tpad - len(cached_sig)), mode="edge")
                    module_sigs[ok] = c.astype(np.complex128)
                else:
                    module_sigs[ok] = np.zeros(n, dtype=np.complex128)
        else:  # passthrough / pitch_quantizer — zero independent source
            module_sigs[mod.key] = np.zeros(n, dtype=np.complex128)

    # Control slider signals — constant DC at the slider's current scaled value.
    # These are routing nodes so any ParamNode wired to them is driven by the
    # slider position.  They are NOT auto-routed to the mix bus.
    ctrl_sigs: dict[str, np.ndarray] = {}
    for cs in patch.controls:
        for sl in cs.sliders:
            ctrl_sigs[sl.key] = np.full(n, sl.scaled_value(), dtype=np.complex128)

    # ---------------------------------------------------------------------------
    # SidecarBus: populate initial per-source channels from raw synthesis outputs.
    # Channels are registered from native-dtype arrays — no forced dtype change.
    # The bus is cheap to build (no extra synthesis); it references already-computed
    # arrays. The "routed" channel is added after the routing solve below.
    # ---------------------------------------------------------------------------
    sidecar = SidecarBus()
    # Voices: envelope (float64), amplitude |z|, phase arg(z)
    _voice_map = {v.key: v for v in patch.voices}
    for vkey, vsig in voice_sigs.items():
        voice = _voice_map.get(vkey)
        if voice is not None:
            env_curve = _compute_envelope(voice, len(vsig), patch.duration)
            sidecar.put(vkey, "envelope", env_curve)
        sidecar.put(vkey, "amplitude", np.abs(vsig))
        sidecar.put(vkey, "phase",     np.angle(vsig))
    # LFOs: amplitude and phase from the analytic signal
    for lkey, lsig in lfo_sigs.items():
        sidecar.put(lkey, "amplitude", np.abs(lsig))
        sidecar.put(lkey, "phase",     np.angle(lsig))
    for skey, ssig in sys_in_sigs.items():
        sidecar.put(skey, "amplitude", np.abs(ssig))
        sidecar.put(skey, "phase",     np.angle(ssig))
    # Modules: amplitude and phase
    for mkey, msig in module_sigs.items():
        sidecar.put(mkey, "amplitude", np.abs(msig))
        sidecar.put(mkey, "phase",     np.angle(msig))
    # Control sliders: scalar value broadcast to an array
    for cs in patch.controls:
        for sl in cs.sliders:
            sidecar.put(sl.key, "value",
                        np.full(n, sl.scaled_value(), dtype=np.float64))

    g = _working_routing_graph_for_synthesis(patch)
    if _routing_param_ov:
        for _at, _arr in _routing_param_ov.items():
            _parsed = _parse_routing_edge_attr(_at)
            if _parsed is None:
                continue
            _kind, _src, _dst = _parsed
            _scalar = float(np.mean(np.asarray(_arr, dtype=np.float64)))
            if _kind == "mix":
                g.set_weight(_src, _dst, _scalar)
            elif _kind == "angle":
                g.set_angle_rad(_src, _dst, _scalar)
            elif _kind == "delay":
                g.set_delay_s(_src, _dst, _scalar)
        g.prune()

    # 2. Build N-node routing system (voices + lfos + modules + ctrl-sliders + mixers + param_nodes)
    node_keys = _patch_node_keys(patch)
    N  = len(node_keys)
    ki = {k: i for i, k in enumerate(node_keys)}

    # --- Optional latency compensation (pre-roll synthesis) ---
    # When enabled, each source is pre-synthesised starting before t=0 by the
    # amount needed for its signal to arrive at every destination on time.
    # The tone generator evaluates at negative t exactly — no approximation.
    lead_map: dict[str, int] = {}
    max_lead: int = 0
    if g.latency_compensation:
        lead_map = compute_latency_compensation(g.edges, node_keys, sr)
        max_lead = max(lead_map.values(), default=0)

    n_ext = n + max_lead   # extended buffer length including pre-roll

    # Virtual patch nodes:
    # __patch_tonic__ = tonal center (scale root) — always seq_tonic_hz
    # __patch_seq__   = current note Hz — _seq_note_hz if set, else seq_tonic_hz
    # Both carry real=Hz, imag=0.  They are note-domain sources, not audio.
    _patch_tonic_val = complex(patch.seq_tonic_hz, 0.0)
    _patch_seq_hz    = patch._seq_note_hz if patch._seq_note_hz > 0 else patch.seq_tonic_hz
    _patch_seq_val   = complex(_patch_seq_hz, 0.0)
    patch_virtual_sigs: dict = {
        "__patch_tonic__": np.full(n_ext, _patch_tonic_val, dtype=np.complex128),
        "__patch_seq__":   np.full(n_ext, _patch_seq_val, dtype=np.complex128),
    }
    for _vk, _vsig in patch_virtual_sigs.items():
        sidecar.put(_vk, "value", _vsig.real.astype(np.float64))

    # Re-synthesise with pre-roll if needed
    if max_lead > 0:
        voice_sigs = _synthesize_voice_sources(
            patch,
            lfo_map,
            p_map,
            n_samples=n_ext,
            t_offset_map={k: -(lead_map.get(k, 0) / sr) for k in p_map.keys()},
            voice_param_overrides=_voice_param_ov,
            param_series=_cached_ps or None,
            file_render=file_render,
            granular_seed_offset=granular_seed_offset,
        )
        lfo_sigs = {}
        for lfo in patch.lfos:
            lead_n = lead_map.get(lfo.key, 0)
            lead_s = lead_n / sr
            lfo_sigs[lfo.key] = _synthesize_lfo_csig(lfo, n_ext, sr, t_offset=-lead_s)

        # Re-synthesise module signals with pre-roll
        module_sigs = {}
        for mod in patch.modules:
            if mod.muted:
                module_sigs[mod.key] = np.zeros(n_ext, dtype=np.complex128)
                if mod.module_type == "lfo" and mod.lfo_channels:
                    for i in range(len(mod.lfo_channels)):
                        module_sigs[mod.lfo_ch_key(i)] = np.zeros(n_ext, dtype=np.complex128)
            elif mod.module_type == "lfo":
                if mod.lfo_channels:
                    module_sigs[mod.key] = np.zeros(n_ext, dtype=np.complex128)
                    for i, ch in enumerate(mod.lfo_channels):
                        lead_s = lead_map.get(mod.lfo_ch_key(i), 0) / sr
                        module_sigs[mod.lfo_ch_key(i)] = _synthesize_lfo_channel_csig(
                            ch, n_ext, sr, t_offset=-lead_s)
                else:
                    lead_n = lead_map.get(mod.key, 0)
                    lead_s = lead_n / sr
                    _tmp = LFODefinition(key=mod.key, label=mod.label,
                                         rate_hz=mod.rate_hz, shape=mod.shape,
                                         phase_offset=mod.phase_offset, depth=mod.depth)
                    module_sigs[mod.key] = _synthesize_lfo_csig(_tmp, n_ext, sr, t_offset=-lead_s)
            else:
                module_sigs[mod.key] = np.zeros(n_ext, dtype=np.complex128)

        # DC control sliders don't change with pre-roll
        ctrl_sigs = {}
        for cs in patch.controls:
            for sl in cs.sliders:
                ctrl_sigs[sl.key] = np.full(n_ext, sl.scaled_value(), dtype=np.complex128)
        sys_in_sigs = {}
        for _i, _k in enumerate(_sys_inputs):
            if _i < len(_sys_bufs):
                _buf = np.asarray(_sys_bufs[_i], dtype=np.float32).reshape(-1)
                if len(_buf) >= n_ext:
                    _arr = _buf[-n_ext:]
                elif len(_buf) > 0:
                    _arr = np.pad(_buf, (0, n_ext - len(_buf)), mode="constant")
                else:
                    _arr = np.zeros(n_ext, dtype=np.float32)
            else:
                _arr = np.zeros(n_ext, dtype=np.float32)
            sys_in_sigs[_k] = _arr.astype(np.complex128)

    # Source matrix: Src[i, :] = node i's independent signal.
    # Mixer nodes have NO independent source -- driven purely by routing edges.
    # pitch_quantizer modules start at zero and are post-processed after solve.
    Src = np.zeros((N, n_ext), dtype=np.complex128)
    for key, sig in voice_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]
    for key, sig in lfo_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]
    for key, sig in module_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]
    for key, sig in ctrl_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]
    for key, sig in sys_in_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]
    for key, sig in patch_virtual_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]

    # Build node_transforms for nonlinear modules (e.g. pitch_quantizer).
    # Each transform is a callable(complex128 array) -> complex128 array that
    # is applied *inside* the routing solve (two-pass) so all downstream nodes
    # see the correctly transformed signal during the same solve.
    _node_transforms: dict = {}
    _dt_q = 1.0 / float(patch.preview_sr) if patch.preview_sr > 0 else 0.0
    for _qmod in patch.modules:
        if _qmod.module_type != "pitch_quantizer" or _qmod.muted:
            continue
        if _qmod.key not in ki:
            continue
        _qhandle = make_quantizer_handle(_qmod, patch.tuning)
        def _make_qtransform(_h=_qhandle, _dt=_dt_q):
            def _qtransform(row: "np.ndarray") -> "np.ndarray":
                # row is complex128: real = Hz input, imag = quadrature (preserved).
                hz_in = np.asarray(row.real, dtype=np.float64)
                hz_out = _h.process_series(hz_in, hz_in, domain="hz", dt=_dt)
                # Only the real (Hz) component is quantized; imaginary stays intact.
                return hz_out.astype(np.complex128) + 1j * row.imag
            return _qtransform
        _node_transforms[_qmod.key] = _make_qtransform()

    # Build coupled_transforms for interaural modules.
    # Each module couples its ch1+ch2 rows: both accumulate routed inputs
    # inside the solve, then _place_signal maps them to spatialized outputs.
    # In mono (nothing wired to ch2), X[ch2_idx] is naturally zero from the
    # solve — no copy/doubling needed.
    def _build_iau_coupled(param_overrides: dict) -> dict:
        """Return coupled_transforms dict for all active interaural modules.

        param_overrides: {mod.key: {attr: float64 series}} for time-varying params.
        """
        _ct: dict = {}
        for _im in patch.modules:
            if _im.module_type != "interaural" or _im.muted:
                continue
            _c1, _c2 = _im.ch1_key(), _im.ch2_key()
            if _c1 not in ki or _c2 not in ki:
                continue
            _ov = param_overrides.get(_im.key, {})
            def _make_iau_fn(_m=_im, _ov=_ov, _c1=_c1, _c2=_c2):
                def _bc(v_scalar, series, T):
                    if series is not None:
                        a = np.asarray(series, dtype=np.float64)
                        return np.pad(a, (0, max(0, T - len(a))), mode="edge")[:T]
                    return np.full(T, v_scalar, dtype=np.float64)
                def _iau_fn(rows):
                    inp1 = rows[_c1]
                    inp2 = rows[_c2]
                    T = inp1.shape[0]
                    az1  = _bc(_m.iau_azimuth,    _ov.get("iau_azimuth"),    T)
                    el1  = _bc(_m.iau_elevation,  _ov.get("iau_elevation"),  T)
                    dst1 = _bc(_m.iau_distance,   _ov.get("iau_distance"),   T)
                    wid1 = _bc(_m.iau_width,      _ov.get("iau_width"),      T)
                    az2  = _bc(_m.iau_azimuth_ch2,   _ov.get("iau_azimuth_ch2"),   T)
                    el2  = _bc(_m.iau_elevation_ch2, _ov.get("iau_elevation_ch2"), T)
                    dst2 = _bc(_m.iau_distance_ch2,  _ov.get("iau_distance_ch2"),  T)
                    wid2 = _bc(_m.iau_width_ch2,     _ov.get("iau_width_ch2"),     T)
                    out1_a, out2_a = _place_signal(inp1, az1, el1, dst1, wid1)
                    out1_b, out2_b = _place_signal(inp2, az2, el2, dst2, wid2)
                    return {_c1: out1_a + out1_b, _c2: out2_a + out2_b}
                return _iau_fn
            _ct[(_c1, _c2)] = _make_iau_fn()
        return _ct

    _coupled_transforms = _build_iau_coupled(_module_param_ov)

    # 3. Solve the routing system — interaural spatial transforms run inside
    # the convergence loop, so feedback loops through spatial modules are correct.
    fb = g.feedback
    global_decay = max(0.0, 1.0 - float(fb.decay)) if fb.enabled else 1.0
    X, _ = solve_routing_with_ringdown(
        Src, g.edges, node_keys, sr, global_decay, fb,
        node_transforms=_node_transforms or None,
        coupled_transforms=_coupled_transforms or None,
    )


    # 3a-ii. LFO channel scale post-solve.
    # Each multi-channel LFO channel's Src already holds amplitude*lfo_waveform.
    # After the solve: X[ch_key] = lfo_src + sum(routed_inputs).
    # We apply per-channel scale to the routed portion only:
    #   new = lfo_src + scale * (X[ch_key] - lfo_src)
    # Then delta-update downstream nodes.
    for _lmod in patch.modules:
        if _lmod.module_type != "lfo" or not _lmod.lfo_channels or _lmod.muted:
            continue
        for _li, _lch in enumerate(_lmod.lfo_channels):
            _lck = _lmod.lfo_ch_key(_li)
            if _lck not in ki:
                continue
            _scale  = float(_lch.get("scale", 0.0))
            _lsig   = module_sigs.get(_lck)
            if _lsig is None:
                continue
            _T       = X.shape[1]
            _lsig_T  = _lsig[:_T]
            _old_val = X[ki[_lck]].copy()
            _new_val = _lsig_T + _scale * (_old_val - _lsig_T)
            _delta   = _new_val - _old_val
            for _e in g.edges:
                if _e.src_key != _lck or _e.delay_s != 0.0:
                    continue
                _di = ki.get(_e.dst_key)
                if _di is None:
                    continue
                _cw = complex(
                    _e.weight * global_decay * math.cos(_e.angle_rad),
                    _e.weight * global_decay * math.sin(_e.angle_rad),
                )
                X[_di] += _cw * _delta
            X[ki[_lck]] = _new_val

    # Populate sidecar "routed" channel for every node: the full complex routed
    # output (post-solve, pre-projection).  This is the definitive signal state
    # at each node and enables retroactive inspection of envelope, mix balance,
    # modulation depth, etc. for any node in the graph.
    _X_n = min(n, X.shape[1])
    for _k in node_keys:
        if _k in ki:
            sidecar.put(_k, "routed", X[ki[_k], :_X_n].copy())

    # 3b. Update param-series cache for the NEXT frame.
    # Extract each param node's output from the just-solved X and store it on
    # the patch.  Next frame's voice synthesis will consume these values before
    # the solve, so there is exactly one solve per frame (one-buffer latency).
    def extract_param_series(csig: np.ndarray, extractor: str) -> np.ndarray:
        if extractor == "real":
            return csig.real.astype(np.float64)
        if extractor == "imag":
            return csig.imag.astype(np.float64)
        if extractor == "phase":
            return np.angle(csig)
        if extractor == "energy":
            return (csig.real ** 2 + csig.imag ** 2).astype(np.float64)
        if extractor == "rms":
            mag = np.abs(csig)
            kernel = np.ones(128) / 128.0
            return np.convolve(mag, kernel, mode="same").astype(np.float64)
        # default: magnitude
        return np.abs(csig).astype(np.float64)

    if patch.param_nodes:
        _new_cache: dict = {}
        X_cols = X.shape[1]
        for pn in patch.param_nodes:
            if pn.key not in ki:
                continue
            raw = extract_param_series(X[ki[pn.key], :min(n, X_cols)], pn.extractor)
            lo, hi = float(pn.low), float(pn.high)
            span = hi - lo
            is_driven = any(pe.dst_key == pn.key for pe in patch.routing.param_edges)
            if not is_driven:
                _new_cache[pn.key] = np.full(len(raw), np.clip(float(pn.default_value), lo, hi))
                continue
            if span > 0:
                rmin, rmax = float(raw.min()), float(raw.max())
                rspan = rmax - rmin
                if rspan > 1e-12:
                    raw = (raw - rmin) / rspan * span + lo
                else:
                    raw = np.full_like(raw, lo + span * 0.5)
            _new_cache[pn.key] = np.clip(raw, lo, hi)
        patch._param_series_cache = _new_cache

    # 3c. State machine post-solve step.
    # Each state_machine module reads its accumulated routing inputs from X,
    # calls the plugin's step() with those inputs + current state, writes the
    # output trajectories back to X for its output nodes, and caches them for
    # the next frame's Src pre-population.  Delta-propagation covers zero-delay
    # downstream edges.
    _T_sm = X.shape[1]
    for _smmod in patch.modules:
        if _smmod.module_type != "state_machine" or _smmod.muted:
            continue
        _smmod._sm_log_text = ""
        if not _smmod.sm_items or not _smmod.sm_vars or not _smmod.sm_plugin:
            continue
        _sm_plug = _load_sm_plugin(_smmod.sm_plugin)
        if _sm_plug is None or not callable(getattr(_sm_plug, "step", None)):
            continue
        # Gather inputs: all signals that have edges targeting the main node.
        # When an edge has item_slot set, the SM receives the signal under that
        # name so driver/instrument identity is preserved through the boundary.
        _sm_inputs: dict = {}
        for _e in g.edges:
            if _e.dst_key == _smmod.key and _e.src_key in ki:
                _slot = _e.item_slot if _e.item_slot else _e.src_key
                _sm_inputs[_slot] = _sm_wrap(
                    X[ki[_e.src_key], :_T_sm].copy(), _smmod.sm_use_torch)
        _sm_dt = 1.0 / max(sr, 1.0)
        _sm_state_in = dict(_smmod._sm_state)
        _sm_stdout = io.StringIO()
        _sm_stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(_sm_stdout), contextlib.redirect_stderr(_sm_stderr):
                _sm_traj = _sm_plug.step(
                    _sm_inputs, _sm_state_in, _sm_dt,
                    n_items=_smmod.sm_n_items,
                    use_torch=_smmod.sm_use_torch,
                    params=dict(_smmod.sm_params),
                    plugin_state=dict(getattr(_smmod, "_sm_aux_state", {}) or {}),
                )
        except Exception:
            _captured = []
            _stdout_txt = _sm_stdout.getvalue()
            _stderr_txt = _sm_stderr.getvalue()
            if _stdout_txt:
                _captured.append(_stdout_txt.rstrip())
            if _stderr_txt:
                _captured.append(_stderr_txt.rstrip())
            _captured.append(traceback.format_exc().rstrip())
            _smmod._sm_log_text = "\n".join(x for x in _captured if x)
            continue
        if isinstance(_sm_traj, dict) and "outputs" in _sm_traj:
            _sm_outputs = _sm_traj.get("outputs", {}) or {}
            _sm_state_out = _sm_traj.get("state", {}) or {}
            _sm_log_extra = _sm_plugin_log_text(_sm_traj.get("log"))
            _sm_aux_state_out = _sm_traj.get("plugin_state", {}) or {}
        else:
            _sm_outputs = _sm_traj or {}
            _sm_state_out = {}
            _sm_log_extra = _sm_plugin_log_text(_sm_traj.get("log")) if isinstance(_sm_traj, dict) else ""
            _sm_aux_state_out = {}
        _sm_log_parts = []
        _stdout_txt = _sm_stdout.getvalue()
        _stderr_txt = _sm_stderr.getvalue()
        if _stdout_txt:
            _sm_log_parts.append(_stdout_txt.rstrip())
        if _stderr_txt:
            _sm_log_parts.append(_stderr_txt.rstrip())
        if _sm_log_extra:
            _sm_log_parts.append(_sm_log_extra.rstrip())
        _smmod._sm_log_text = "\n".join(x for x in _sm_log_parts if x)
        # Write trajectories back to X output nodes + cache + update state
        _new_sm_state: dict = {}
        for _item in _smmod.sm_items:
            _new_sm_state[_item] = {}
            for _var in _smmod.sm_vars:
                _ok = _smmod.sm_out_key(_item, _var)
                if _ok not in ki:
                    continue
                _traj_val = _sm_outputs.get(_item, {}).get(_var)
                if _traj_val is None:
                    continue
                _traj_arr = _sm_unwrap(_traj_val)
                # Pad/trim to match X columns
                if len(_traj_arr) < _T_sm:
                    _traj_arr = np.pad(_traj_arr, (0, _T_sm - len(_traj_arr)), mode="edge")
                _csig = _traj_arr[:_T_sm].astype(np.complex128)
                _old  = X[ki[_ok]].copy()
                _delta = _csig - _old
                # Delta-propagate to zero-delay downstream edges
                for _e2 in g.edges:
                    if _e2.src_key != _ok or _e2.delay_s != 0.0:
                        continue
                    _di = ki.get(_e2.dst_key)
                    if _di is None:
                        continue
                    _cw2 = complex(
                        _e2.weight * global_decay * math.cos(_e2.angle_rad),
                        _e2.weight * global_decay * math.sin(_e2.angle_rad),
                    )
                    X[_di] += _cw2 * _delta
                X[ki[_ok]] = _csig
                _smmod._sm_out_cache[_ok] = _csig[:n].copy()
            for _state_var in _smmod.sm_state_vars:
                _state_val = _sm_state_out.get(_item, {}).get(_state_var, None)
                if _state_val is not None:
                    _new_sm_state[_item][_state_var] = float(_state_val)
                    continue
                _traj_val = _sm_outputs.get(_item, {}).get(_state_var)
                if _traj_val is None:
                    _traj_val = _sm_traj.get(_item, {}).get(_state_var) if isinstance(_sm_traj, dict) else None
                if _traj_val is None:
                    _new_sm_state[_item][_state_var] = float(
                        _sm_state_in.get(_item, {}).get(_state_var, 0.0)
                    )
                    continue
                _traj_arr = _sm_unwrap(_traj_val)
                if len(_traj_arr) > 0:
                    _new_sm_state[_item][_state_var] = float(_traj_arr[-1])
        _smmod._sm_state = _new_sm_state
        _smmod._sm_aux_state = dict(_sm_aux_state_out)

    # 4. Resolve the main PCM output bus.
    # If the user has routed signal into the built-in system output channels,
    # those channels define the main output directly. Otherwise, preserve the
    # legacy active-mixer projection path.
    active_mixer_keys = [m.key for m in patch.mixers if m.projection_active]
    sys_out_keys = [k for k in _system_output_keys(patch) if k in ki]
    has_system_routing = bool(sys_out_keys) and any(
        e.dst_key in set(sys_out_keys) for e in g.edges
    )
    if has_system_routing:
        out_channels = np.column_stack([
            X[ki[_k], :n].real.astype(np.float32) for _k in sys_out_keys
        ]).astype(np.float32, copy=False)
        if patch.normalize_output and out_channels.size:
            peak = float(np.max(np.abs(out_channels)))
            if peak > 1e-9:
                out_channels /= peak
        left = out_channels[:, 0]
        right = out_channels[:, 1] if out_channels.shape[1] > 1 else out_channels[:, 0]
    elif active_mixer_keys:
        mix = sum(X[ki[mk]] for mk in active_mixer_keys if mk in ki)
        if patch.normalize_output:
            peak = float(np.max(np.abs(mix)))
            if peak > 1e-9:
                mix /= peak
        left, right = _apply_projection(mix, patch.projection_mode,
                                        patch.projection_rotation_hz, sr)
        out_channels = np.column_stack([left, right]).astype(np.float32, copy=False)
    else:
        out_channels = np.zeros((n, max(2, len(sys_out_keys) or 2)), dtype=np.float32)
        left = out_channels[:, 0]
        right = out_channels[:, 1]
    out = (left, right)
    if _return_mixer_sigs:
        _mixer_outs: dict = {}
        for _m in patch.mixers:
            if _m.export_to_file and _m.key in ki:
                _m_sig = X[ki[_m.key]][:n].copy()
                if patch.normalize_output:
                    _pk = float(np.max(np.abs(_m_sig)))
                    if _pk > 1e-9:
                        _m_sig = _m_sig / _pk
                _ml, _mr = _apply_projection(_m_sig, patch.projection_mode,
                                             patch.projection_rotation_hz, sr)
                _mixer_outs[_m.key] = (_ml, _mr)
        ret: list = [*out]
        if _return_output_channels:
            ret.append(out_channels)
        ret.append(_mixer_outs)
        if _return_sidecar:
            ret.append(sidecar)
        return tuple(ret)
    if _return_output_channels and _return_sidecar:
        return (*out, out_channels, sidecar)
    if _return_output_channels:
        return (*out, out_channels)
    if _return_sidecar:
        return (*out, sidecar)
    return out


# ---------------------------------------------------------------------------
# GL helpers
# ---------------------------------------------------------------------------

def _surface_to_gl_tex(surf: pygame.Surface, old_id: int = 0) -> int:
    raw = pygame.image.tostring(surf, "RGBA", False)
    w, h = surf.get_size()
    if old_id:
        glDeleteTextures([old_id])
    tid = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tid)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
    return tid


def _draw_tex_quad(tid: int, x: int, y: int, w: int, h: int, ww: int, wh: int) -> None:
    x0 = 2.0 * x / ww - 1.0
    y1 = 1.0 - 2.0 * y / wh
    x1 = 2.0 * (x + w) / ww - 1.0
    y0 = 1.0 - 2.0 * (y + h) / wh
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D, tid)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x0, y1)
    glTexCoord2f(1, 0); glVertex2f(x1, y1)
    glTexCoord2f(1, 1); glVertex2f(x1, y0)
    glTexCoord2f(0, 1); glVertex2f(x0, y0)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def _ndc(px: float, py: float, ww: int, wh: int) -> tuple[float, float]:
    return 2.0 * px / ww - 1.0, 1.0 - 2.0 * py / wh


def _gl_vline(px, y0, y1, ww, wh, col=(1,1,1,1), lw=1.0):
    nx, na = _ndc(px, y0, ww, wh)
    _,  nb = _ndc(px, y1, ww, wh)
    glDisable(GL_TEXTURE_2D)
    glLineWidth(lw)
    glColor4f(*col)
    glBegin(GL_LINES); glVertex2f(nx, na); glVertex2f(nx, nb); glEnd()
    glLineWidth(1.0)


def _gl_hline(py, x0, x1, ww, wh, col=(1,1,1,1), lw=1.0):
    na, ny = _ndc(x0, py, ww, wh)
    nb, _  = _ndc(x1, py, ww, wh)
    glDisable(GL_TEXTURE_2D)
    glLineWidth(lw)
    glColor4f(*col)
    glBegin(GL_LINES); glVertex2f(na, ny); glVertex2f(nb, ny); glEnd()
    glLineWidth(1.0)


def _gl_rect(px, py, pw, ph, ww, wh, col):
    x0, y0 = _ndc(px,      py,      ww, wh)
    x1, y1 = _ndc(px + pw, py + ph, ww, wh)
    glDisable(GL_TEXTURE_2D)
    glColor4f(*col)
    glBegin(GL_QUADS)
    glVertex2f(x0, y0); glVertex2f(x1, y0)
    glVertex2f(x1, y1); glVertex2f(x0, y1)
    glEnd()


def _gl_diamond(cx, cy, r, ww, wh, fill_col, border_col=None):
    top   = _ndc(cx,     cy - r, ww, wh)
    right = _ndc(cx + r, cy,     ww, wh)
    bot   = _ndc(cx,     cy + r, ww, wh)
    left  = _ndc(cx - r, cy,     ww, wh)
    glDisable(GL_TEXTURE_2D)
    glColor4f(*fill_col)
    glBegin(GL_QUADS)
    glVertex2f(*top); glVertex2f(*right); glVertex2f(*bot); glVertex2f(*left)
    glEnd()
    if border_col:
        glColor4f(*border_col)
        glBegin(GL_LINE_LOOP)
        glVertex2f(*top); glVertex2f(*right); glVertex2f(*bot); glVertex2f(*left)
        glEnd()


def _gl_cursor_flag(px, y_top, y_bot, ww, wh, col):
    """Vertical cursor line + triangle flag at top."""
    _gl_vline(px, y_top + 10, y_bot, ww, wh, col, 1.5)
    a, b = _ndc(px,      y_top,      ww, wh)
    c, d = _ndc(px + 10, y_top + 7,  ww, wh)
    e, f = _ndc(px,      y_top + 10, ww, wh)
    glDisable(GL_TEXTURE_2D)
    glColor4f(*col)
    glBegin(GL_TRIANGLES)
    glVertex2f(a, b); glVertex2f(c, d); glVertex2f(e, f)
    glEnd()


def _gl_loop_handle(px, y_top, y_bot, ww, wh, col, is_left: bool):
    _gl_vline(px, y_top, y_bot, ww, wh, col, 2.5)
    bw = 10.0
    x0 = px if is_left else px - bw
    x1 = px + bw if is_left else px
    _gl_hline(y_top,     x0, x1, ww, wh, col, 2.5)
    _gl_hline(y_bot - 1, x0, x1, ww, wh, col, 2.5)


# ---------------------------------------------------------------------------
# EditorCanvas — center view with PlotWidget + GL interactive overlay
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RoutingGridView — center-view knob matrix for signal routing
# ---------------------------------------------------------------------------

_SNAP_12TH = 1.0 / 12.0   # semitone-based snap increment for routing knobs

# Popular tuning roots used as snap points on the tuning.root_hz slider.
# Covers: A2/A3/A4 series (110 / 220 / 432 / 440 / 466.16 Hz),
#         baroque A415, French baroque A392,
#         C4 in standard tuning (261.626) and philosophical C (256).
_ROOT_HZ_SNAPS: tuple[float, ...] = (
    110.0,       # A2  — standard
    130.813,     # C3  — standard (A=440)
    220.0,       # A3  — standard
    256.0,       # C4  — philosophical / Verdi
    261.626,     # C4  — standard (A=440)
    392.0,       # A392 — French baroque
    415.0,       # A415 — baroque
    432.0,       # A432 — alternative
    440.0,       # A440 — ISO 16 standard concert pitch
    466.16,      # A466 — Chorton (German high baroque)
)
_ROOT_HZ_SNAP_TOL = 0.015   # ±1.5 % relative tolerance

# Standard audio sample rates — used for slider snapping / tick marks
_COMMON_SAMPLE_RATES: tuple[int, ...] = (
    8000, 11025, 16000, 22050, 32000,
    44100, 48000, 88200, 96000, 176400, 192000,
)

# Western chromatic pitch-class names (12-TET / MIDI convention)
_NOTE_NAMES_12 = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# 22-śruti Sargam names (Bharata / Natya Shastra ordering).
# Index = śruti number 0-21 (chromatic, from Sa).
_SRUTI_NAMES_22 = (
    "Sa",       #  0 tonic
    "koRe",     #  1 komal Re (ek śruti)
    "koRe₂",    #  2 komal Re (do śruti)
    "Re₁",      #  3 Ri-1 (tri śruti)
    "Re",       #  4 shuddha Re / chatur-śruti Ri
    "koGa₁",    #  5 sadharana Ga low
    "koGa",     #  6 sadharana Ga / komal Ga
    "Ga",       #  7 antara Ga
    "Ga₂",      #  8 shuddha Ga-2
    "Ma",       #  9 shuddha Ma (perfect 4th)
    "Ma₂",      # 10
    "tiMa₁",    # 11 tivra Ma-1
    "tiMa",     # 12 tivra Ma-2 (tritone)
    "Pa",       # 13 perfect 5th
    "koDha",    # 14 komal Dha (ek śruti)
    "koDha₂",   # 15 komal Dha (do śruti)
    "Dha₁",     # 16 Dha-1
    "Dha",      # 17 shuddha Dha / chatur-śruti
    "koNi₁",    # 18 komal Ni (ek śruti)
    "koNi",     # 19 kaisiki Ni
    "Ni₁",      # 20 kakali Ni low
    "Ni",       # 21 shuddha Ni
)


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
            editor = ParametricCurveEditor(
                piecewise.curve,
                piecewise.chirp_curve,
                piecewise.signal_curve,
                w=max(64, cw),
                h=max(64, plot_h),
                library_folder=os.path.join(os.getcwd(), "envelopes"),
            )
            editor._overlay = _TextOverlay(editor.w, editor.h)
            editor._channels["A"] = editor._channel_state("A")
            editor._channels["A"].rule_tree = piecewise.rule_tree
            self._piecewise_editor = editor
            self._piecewise_voice_key = voice.key
            self._surf_dirty = True
        else:
            if self._piecewise_editor.w != max(64, cw) or self._piecewise_editor.h != max(64, plot_h):
                self._piecewise_editor.resize(max(64, cw), max(64, plot_h))
                self._surf_dirty = True
        self._pull_piecewise_editor_state_into_voice(voice)
        return self._piecewise_editor

    def _pull_piecewise_editor_state_into_voice(self, voice: "AnalyticVoice") -> None:
        if self._piecewise_editor is None:
            return
        if voice.piecewise_env is None:
            voice.piecewise_env = PiecewiseVoiceEnvelope()
        voice.piecewise_env.curve = self._piecewise_editor._curves[0]
        voice.piecewise_env.chirp_curve = self._piecewise_editor._curves[1]
        voice.piecewise_env.signal_curve = self._piecewise_editor._curves[2]
        voice.piecewise_env.rule_tree = self._piecewise_editor._channel_state("A").rule_tree or EnvelopeRuleTree.default()

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

        # Cancel any running rebuild
        self._rebuild_cancel.set()
        # Don't join — daemon thread will bail out on its own

        # Deep-copy the patch so the bg thread doesn't race with UI mutations
        snap = _rcopy.deepcopy(patch)
        cancel = threading.Event()
        self._rebuild_cancel = cancel

        def _worker():
            try:
                self._rebuild_work(snap, active_key, snap_mode, snap_show_tail,
                                   snap_seed, cancel, fp)
            except Exception:
                pass  # silently drop — stale visuals are better than a crash

        t = threading.Thread(target=_worker, daemon=True)
        self._rebuild_thread = t
        t.start()

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
            # Resize to final target dimensions first so _prect() is correct
            # when _refresh_render_points computes downsampled point lists.
            # render_to_surface will see the same size and skip the resize.
            editor.resize(max(64, cw), max(64, plot_h))
            self._pull_piecewise_editor_state_into_voice(voice)
            # Feed the current synthesised signals into the editor's output panel
            import torch as _torch
            lfo_map = {l.key: l for l in patch.lfos}
            p_map   = {p.key: p for p in patch.voices}
            if voice and not _voice_effectively_muted(voice, patch):
                _csig = _synthesize_voice(
                    voice,
                    patch,
                    lfo_map,
                    p_map,
                    granular_seed_offset=self.granular_seed_offset,
                )
                if hasattr(_csig, "detach"):
                    _csig = _csig.detach().cpu().numpy()
                _csig = np.asarray(_csig, dtype=np.complex128)
                self._complex_sig = _csig
                self._signal = _csig.real.astype(np.float32, copy=False)
                self._env_curve = _compute_envelope(voice, len(_csig), patch.duration).astype(np.float32, copy=False)
                self._chirp_f = _compute_chirp_frequency_series(voice, len(_csig), patch.duration).astype(np.float32, copy=False)
            else:
                _csig = self._complex_sig
            _env  = self._env_curve
            _cf   = self._chirp_f
            if _csig is not None and len(_csig) > 0:
                _n_sig = len(_csig)
                _env_arr = _env.astype(float) if _env is not None else np.zeros(_n_sig)
                _amp_total = np.abs(np.asarray(_csig, dtype=np.complex128))
                _amp_peak = float(np.max(_amp_total)) if _amp_total.size > 0 else 0.0
                if _amp_peak > 1e-9:
                    _amp_total = _amp_total / _amp_peak
                else:
                    _amp_total = np.zeros_like(_amp_total, dtype=np.float64)

                _base_chirp = (
                    _compute_knob_chirp_deviation_series(voice, _n_sig, patch.duration)
                    if voice is not None else
                    np.zeros(_n_sig, dtype=np.float64)
                )
                _chirp_env_norm = np.full(_n_sig, 0.5, dtype=np.float64)
                _chirp_total_norm = _chirp_env_norm.copy()
                if voice is not None and voice.piecewise_env is not None:
                    _t_norm = torch.linspace(0.0, 1.0, _n_sig, dtype=torch.float64)
                    _chirp_curve = editor._curves[1]
                    _chirp_env_curve = _chirp_curve.evaluate_normalized(_t_norm).real.clamp(0.0, 1.0)
                    _chirp_env_hz = _chirp_curve.to_physical(
                        _chirp_env_curve
                    ).detach().cpu().numpy().astype(np.float64, copy=False)
                    _chirp_total_hz = _base_chirp + _chirp_env_hz
                    _v_lo = float(_chirp_curve.v_lo)
                    _v_hi = float(_chirp_curve.v_hi)
                    _span = max(_v_hi - _v_lo, 1e-9)
                    _chirp_total_norm = np.clip((_chirp_total_hz - _v_lo) / _span, 0.0, 1.0)
                    _chirp_env_norm = np.clip((_chirp_env_hz - _v_lo) / _span, 0.0, 1.0)
                    _chirp_bias_norm = np.clip((_base_chirp - _v_lo) / _span, 0.0, 1.0)
                else:
                    _curve = editor._curves[1]
                    _v_lo = float(_curve.v_lo)
                    _v_hi = float(_curve.v_hi)
                    _span = max(_v_hi - _v_lo, 1e-9)
                    _chirp_bias_norm = np.clip((_base_chirp - _v_lo) / _span, 0.0, 1.0)
                    _chirp_total_norm = _chirp_bias_norm.copy()
                _raw = {
                    "analytic":  _torch.as_tensor(_csig, dtype=_torch.complex128).reshape(-1),
                    "amplitude_bias": _torch.zeros(_n_sig, dtype=_torch.complex128),
                    "amplitude": _torch.as_tensor(_env_arr, dtype=_torch.complex128).reshape(-1),
                    "amplitude_total": _torch.as_tensor(_amp_total, dtype=_torch.complex128).reshape(-1),
                    "chirp_bias": _torch.as_tensor(_chirp_bias_norm, dtype=_torch.complex128).reshape(-1),
                    "chirp":     _torch.as_tensor(_chirp_env_norm, dtype=_torch.complex128).reshape(-1),
                    "chirp_total": _torch.as_tensor(_chirp_total_norm, dtype=_torch.complex128).reshape(-1),
                }
                # amplitude+chirp panels use channel "A"; output panel uses channel "B"
                _state_a = editor._channel_state("A")
                _state_a.raw_signals = {
                    "amplitude_bias": _raw["amplitude_bias"],
                    "amplitude": _raw["amplitude"],
                    "amplitude_total": _raw["amplitude_total"],
                    "chirp_bias": _raw["chirp_bias"],
                    "chirp": _raw["chirp"],
                    "chirp_total": _raw["chirp_total"],
                }
                _state_a.display_signals = _norm_ch_sigs(_state_a.raw_signals,
                                                         time_stretch=_state_a.time_stretch)
                _state_b = editor._channel_state("B")
                _state_b.raw_signals     = {"analytic": _raw["analytic"]}
                _state_b.display_signals = _norm_ch_sigs(_state_b.raw_signals,
                                                         time_stretch=_state_b.time_stretch)
                # _render_buf drives resize() re-refresh; point it at the analytic signal
                editor._render_buf = _state_b.display_signals.get("analytic")
                editor._refresh_render_points("A")
                editor._refresh_render_points("B")
                self._surf_dirty = True
            if self._surf_dirty:
                editor_surf = editor.render_to_surface(cw, plot_h)
                self._surf.blit(editor_surf, (0, 0))
                self._pull_piecewise_editor_state_into_voice(voice)
                self._tex = _surface_to_gl_tex(self._surf, self._tex)
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
                self._pull_piecewise_editor_state_into_voice(voice)
                self._surf_dirty = True
                return True
            if event.type == MOUSEBUTTONUP:
                editor.on_mouse_up(event.button)
                self._pull_piecewise_editor_state_into_voice(voice)
                self._surf_dirty = True
                return True
            if event.type == MOUSEMOTION:
                editor.on_mouse_move(local_x or 0.0, local_y or 0.0)
                self._pull_piecewise_editor_state_into_voice(voice)
                self._surf_dirty = True
                return True
            if event.type == KEYDOWN:
                editor.on_key_down(event.key, event.mod)
                self._pull_piecewise_editor_state_into_voice(voice)
                self._surf_dirty = True
                return True
            if event.type == pygame.KEYUP:
                editor.on_key_up(event.key)
                self._pull_piecewise_editor_state_into_voice(voice)
                self._surf_dirty = True
                return True
            if event.type == pygame.TEXTINPUT:
                editor.on_text_input(event.text)
                self._pull_piecewise_editor_state_into_voice(voice)
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

    def _on_add_voice(self) -> None:
        p = AnalyticVoice()
        p.label   = f"V{len(self.patch.voices) + 1}"
        p.freq_hz = 220.0 * (2 ** len(self.patch.voices))
        colors = [(100,180,255),(255,130,60),(120,220,120),(220,80,160),(200,200,80)]
        p.color = list(colors[len(self.patch.voices) % len(colors)])
        self.patch.voices.append(p)
        self.active_key = p.key
        self._needs_rebuild = True

    def _on_add_lfo(self) -> None:
        l = LFODefinition()
        l.label   = f"LFO{len(self.patch.lfos) + 1}"
        l.rate_hz = 1.0
        self.patch.lfos.append(l)
        self.active_key = l.key
        self._needs_rebuild = True

    def _mark_dirty(self) -> None:
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
        self._needs_rebuild = True

    def _on_add_param(self) -> None:
        pn = ParamNode()
        pn.label = f"P{len(self.patch.param_nodes) + 1}"
        self.patch.param_nodes.append(pn)
        self.active_key = pn.key
        self.canvas.active_key = pn.key
        self.canvas.mode = EditorMode.PARAM_ROUTING
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
        self._needs_rebuild = True

    def _play_demo_sequence(self) -> None:
        """Synthesize and play an arpeggiated demo using the patch sequence settings."""
        if not _HAS_SEQ_ENG:
            print("sequence_engine not available for demo playback")
            return
        p = self.patch
        template = next((v for v in p.voices if not v.muted), None)
        if template is None:
            return
        scale = p.seq_scale if p.seq_scale in MODAL_SCALES else "pentatonic_minor"
        beat_s  = 60.0 / max(p.seq_bpm, 1.0)
        pattern = _SEQ_PATTERN_PRESETS[
            max(0, min(p.seq_pattern_idx, len(_SEQ_PATTERN_PRESETS) - 1))][1]

        # ── Build the pitch list (shared by rhythm and arpeggio paths) ────────
        try:
            if p.seq_custom_semitones.strip():
                custom_semi  = [float(s) for s in p.seq_custom_semitones.split(",")
                                if s.strip()]
                degrees: list[float] = []
                for octave in range(p.seq_octave_span):
                    for st in custom_semi:
                        degrees.append(semitones_to_hz(p.seq_tonic_hz,
                                                       st + 12 * octave))
            else:
                degrees = scale_degrees_hz(p.seq_tonic_hz, scale,
                                           octave_span=p.seq_octave_span)
        except Exception as exc:
            print(f"Scale build error: {exc}")
            return

        try:
            _play_groups = _prepare_sequence_play_groups(p, beat_s, degrees, pattern)
        except Exception as exc:
            print(f"Demo schedule error: {exc}")
            return
        if not _play_groups:
            return

        import time as _time
        sr = p.preview_sr

        _fb_cfg     = p.routing.feedback
        _gdecay_est = max(0.0, 1.0 - float(_fb_cfg.decay)) if _fb_cfg.enabled else 1.0
        _ringdown_n = estimate_ringdown_samples(p.routing.edges, sr, _gdecay_est, _fb_cfg)
        _total_dur  = max((s.total_duration for s, _ in _play_groups), default=0.0)
        total_n = int((_total_dur + 0.5) * sr) + _ringdown_n
        _total_events = sum(len(s.events) for s, _ in _play_groups)

        _has_sm = any(m.module_type == "state_machine" and not m.muted
                      and m.sm_plugin for m in p.modules)
        print(f"Demo: {_total_events} events across {len(_play_groups)} groups "
              f"| SR {sr} | SM modules {'ON' if _has_sm else 'off'}")

        # Per-module persistent aux state (cavity scenes, stream states) that
        # survives across notes so room/body geometry is built only once.
        _persistent_aux: dict[str, dict] = dict(self._cavity_cache)

        _t0 = _time.monotonic()
        mix_L, mix_R, _out_bus, _persistent_aux = _synthesize_sequence_full_batch(
            p, _play_groups, total_n,
            file_render=False,
            out_channels=2,
            persistent_aux=_persistent_aux,
        )

        # Persist cavity caches back to the viewer for future runs / saving.
        self._cavity_cache.update(_persistent_aux)

        elapsed = _time.monotonic() - _t0
        print(f"Demo: complete in {elapsed:.1f}s")
        peak = float(np.max(np.abs(np.maximum(np.abs(mix_L), np.abs(mix_R)))))
        if peak > 1e-9:
            mix_L = mix_L / peak
            mix_R = mix_R / peak
        try:
            if not self._play_output_bus(np.column_stack([mix_L, mix_R]), sr):
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
            _left, _right, out_bus = _synthesize_patch(
                self.patch,
                granular_seed_offset=self._seed_anim_tick,
                _return_output_channels=True,
            )
            # Harvest cavity caches produced during synthesis.
            for m in self.patch.modules:
                if m.key and getattr(m, "_sm_aux_state", None):
                    self._cavity_cache[m.key] = dict(m._sm_aux_state)
            src_sr = self.patch.preview_sr
            out_bus = np.asarray(out_bus, dtype=np.float32)
            if out_bus.ndim == 1:
                out_bus = out_bus[:, None]
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

        clock = pygame.time.Clock()
        running = True

        while running:
            # ---- Events ----
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

            # ---- Sync panel state ----
            self.patch_panel.set_patch(self.patch, self.active_key)
            self.partial_panel.set_patch(self.patch, self.active_key)

            # ---- Granular seed animation (opt-in per-voice, off by default) ----
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
            if self._needs_rebuild:
                self.canvas.rebuild(self.patch, self.active_key)
                self._needs_rebuild = False
            # Absorb finished background rebuild
            if self.canvas.rebuild_ready():
                self.canvas._rebuild_thread = None
                self.canvas._surf_dirty = True

            # ---- Render ----
            glClear(GL_COLOR_BUFFER_BIT)

            # Center canvas (PlotWidget surface + GL overlay)
            self.canvas.render(
                self.win_w, self.win_h, self.patch, self._atlas, self._font)

            # Left panel
            self._left_tex = self._upload_panel(self.patch_panel, self._left_tex)
            if self._left_tex:
                lpr = self.patch_panel.panel_rect
                _draw_tex_quad(
                    self._left_tex, lpr.x, lpr.y, lpr.w, lpr.h,
                    self.win_w, self.win_h)

            # Right panel
            self._right_tex = self._upload_panel(self.partial_panel, self._right_tex)
            if self._right_tex:
                rpr = self.partial_panel.panel_rect
                _draw_tex_quad(
                    self._right_tex, rpr.x, rpr.y, rpr.w, rpr.h,
                    self.win_w, self.win_h)

            # Title bar
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
            self._render_status()

            pygame.display.flip()
            clock.tick(60)

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
