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
import json
import math
import os
import random as _random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
import pygame
from scipy.signal import resample_poly as _scipy_resample_poly
from pygame.locals import (
    DOUBLEBUF, KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP,
    MOUSEMOTION, MOUSEWHEEL, OPENGL, QUIT, RESIZABLE, VIDEORESIZE,
    K_SPACE, K_ESCAPE, K_TAB, K_DELETE, K_s, K_o, K_n,
    K_LCTRL, K_RCTRL, K_z,
)
from OpenGL.GL import (
    GL_BLEND, GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT, GL_LINEAR,
    GL_LINE_LOOP, GL_LINE_STRIP, GL_LINES, GL_NEAREST, GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS, GL_RGBA, GL_SRC_ALPHA, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_TRIANGLES, GL_UNSIGNED_BYTE,
    glBegin, glBindTexture, glBlendFunc, glClear, glClearColor,
    glColor4f, glDeleteTextures, glDisable, glEnable, glEnd,
    glGenTextures, glLineWidth, glTexCoord2f, glTexImage2D,
    glTexParameteri, glVertex2f, glViewport,
)

from plot_widget import PlotWidget, PlotSeries, PlotMarker
from bass_viewer import (GlyphAtlas, Panel, PanelDock,
                         ScrollableSubpanelList, ModularSubpanelSpec, SubpanelAddOption)

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
        spline_envelope_factory,
        monotone_envelope_factory,
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
    env_type: str = "adsr"         # "adsr" | "spline" | "monotone" | "linear"
    adsr:     ADSRParams = field(default_factory=ADSRParams)
    env_knots: list[list[float]] = field(default_factory=lambda: [
        [0.0, 0.0], [0.01, 1.0], [0.1, 0.75], [0.85, 0.75], [1.0, 0.0]
    ])
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
    emission_mode:       str   = "single"  # "single" | "granular"
    granular:            Any   = None      # GrainPopulationSpec when emission_mode=="granular"

    def active_knots(self) -> list[list[float]]:
        if self.env_type == "adsr":
            return self.adsr.to_knots(1.0)
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
            "env_type": self.env_type,
            "adsr": {"attack": self.adsr.attack, "decay": self.adsr.decay,
                     "sustain": self.adsr.sustain, "release": self.adsr.release,
                     "peak": self.adsr.peak},
            "env_knots": self.env_knots,
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
            "emission_mode": self.emission_mode,
            "granular": (self.granular.to_dict()
                         if self.granular is not None and hasattr(self.granular, "to_dict")
                         else self.granular),
        }

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _ROLES    = ["signal", "air", "transient", "body"]
        _ENV      = ["adsr", "spline", "monotone", "linear"]
        _MANIFOLD = ["pure", "harmonic", "harmonic_warp"]
        return [
            # Oscillator
            KnobSpec("freq_hz",       "Freq",       "float",  440.0, 1.0,    20000.0, 0, "Hz",  [], True,  "Oscillator", ".1f", "ConstantPhasePath"),
            KnobSpec("semitone_offset","Offset",   "float",  0.0, -48.0,   48.0,    0, "st",  [], False, "Oscillator", ".2f"),
            KnobSpec("note_tracking",  "Tracking", "choice", "note", 0, 2, 1, "",
                     ["note", "root", "free"], False, "Oscillator", "", "", True),
            KnobSpec("amplitude",    "Amplitude",  "float",  1.0,   0.0,     4.0,     0, "",    [], False, "Oscillator", ".3f"),
            KnobSpec("phase_origin", "Phase",      "float",  0.0,  -math.pi, math.pi, 0, "rad", [], False, "Oscillator", ".3f", "ConstantPhasePath"),
            KnobSpec("pre_delay",    "Pre-delay",  "float",  0.0,   0.0,     2.0,     0, "s",   [], False, "Oscillator", ".3f"),
            KnobSpec("voice_role",   "Role",       "choice", "signal", 0, 3, 1, "",   _ROLES, False, "Oscillator"),
            KnobSpec("seq_role",     "Arr. role",  "choice", "melody", 0, 3, 1, "",
                     ["melody", "bass", "root", "stab"], False, "Oscillator"),
            # Envelope
            KnobSpec("env_type",     "Env type",   "choice", "adsr", 0, 3,   1, "",   _ENV,   False, "Envelope",   ".0f", "", True),
            KnobSpec("adsr.attack",  "Attack",     "float",  0.005, 0.001, 2.0,  0, "s", [], True,  "Envelope", ".4f", "ADSREnvelope", False, ("env_type", "adsr")),
            KnobSpec("adsr.decay",   "Decay",      "float",  0.04,  0.001, 2.0,  0, "s", [], True,  "Envelope", ".4f", "ADSREnvelope", False, ("env_type", "adsr")),
            KnobSpec("adsr.sustain", "Sustain",    "float",  0.75,  0.0,   1.0,  0, "",  [], False, "Envelope", ".3f", "ADSREnvelope", False, ("env_type", "adsr")),
            KnobSpec("adsr.release", "Release",    "float",  0.08,  0.001, 4.0,  0, "s", [], True,  "Envelope", ".4f", "ADSREnvelope", False, ("env_type", "adsr")),
            KnobSpec("adsr.peak",    "Peak",       "float",  1.0,   0.0,   2.0,  0, "",  [], False, "Envelope", ".3f", "ADSREnvelope", False, ("env_type", "adsr")),
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
        p.env_type    = d.get("env_type", "adsr")
        ad2 = d.get("adsr", {})
        p.adsr = ADSRParams(
            attack=float(ad2.get("attack",  0.005)),
            decay=float(ad2.get("decay",   0.04)),
            sustain=float(ad2.get("sustain", 0.75)),
            release=float(ad2.get("release", 0.08)),
            peak=float(ad2.get("peak", 1.0)),
        )
        p.env_knots    = d.get("env_knots", [[0,0],[0.01,1],[0.1,.75],[.85,.75],[1,0]])
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
        p.emission_mode       = d.get("emission_mode", "single")
        # L2 fix: validate enum-like string fields — silently fall back to the
        # default rather than loading a typo that would synthesize silence.
        _VALID_EMISSION_MODES  = {"single", "granular"}
        _VALID_MANIFOLD_TYPES  = {"pure", "harmonic", "harmonic_warp"}
        _VALID_ENV_TYPES       = {"adsr", "custom", "none", "spline", "monotone", "linear"}
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
        if p.env_type not in _VALID_ENV_TYPES:
            import warnings
            warnings.warn(
                f"AnalyticVoice.from_dict: unknown env_type {p.env_type!r}; "
                f"defaulting to 'adsr'.", stacklevel=2)
            p.env_type = "adsr"
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
class LFODefinition:
    key:          str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:        str   = "LFO"
    rate_hz:      float = 1.0
    shape:        str   = "Sine"   # Sine | Triangle | Sawtooth | Square
    phase_offset: float = 0.0
    depth:        float = 1.0
    color: list[int] = field(default_factory=lambda: [200, 160, 60])

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "rate_hz": self.rate_hz,
                "shape": self.shape, "phase_offset": self.phase_offset,
                "depth": self.depth, "color": self.color}

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _SHAPES = ["Sine", "Triangle", "Sawtooth", "Square"]
        return [
            KnobSpec("rate_hz",      "Rate",         "float",  1.0, 0.01, 50.0,     0, "Hz",  [], True,  "LFO", ".3f"),
            KnobSpec("phase_offset", "Phase offset", "float",  0.0,-math.pi, math.pi,0, "rad", [], False, "LFO", ".3f"),
            KnobSpec("depth",        "Depth",        "float",  1.0, 0.0,  2.0,      0, "",    [], False, "LFO", ".3f"),
            KnobSpec("shape",        "Shape",        "choice", "Sine", 0, 3, 1, "", _SHAPES, False, "LFO"),
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "LFODefinition":
        o = cls.__new__(cls)
        o.key          = d.get("key", uuid.uuid4().hex[:8])
        o.label        = d.get("label", "LFO")
        o.rate_hz      = float(d.get("rate_hz", 1.0))
        o.shape        = d.get("shape", "Sine")
        o.phase_offset = float(d.get("phase_offset", 0.0))
        o.depth        = float(d.get("depth", 1.0))
        o.color        = d.get("color", [200, 160, 60])
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

    def _to_st(self, value: float, domain: str) -> float:
        if domain == "hz":
            if value <= 0.0:
                return 0.0
            return 12.0 * math.log2(value / self._tuning.root_hz)
        return float(value)

    def _from_st(self, st: float, domain: str) -> float:
        if domain == "hz":
            return self._tuning.semitone_to_hz(st)
        return st

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

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "projection_active": self.projection_active,
                "color": self.color,
                "export_to_file":     self.export_to_file,
                "export_sample_rate": self.export_sample_rate,
                "export_bit_depth":   self.export_bit_depth}

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
        return o



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
                             compute_latency_compensation, solve_param_routing)


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

    _MODULE_TYPES = ["lfo", "passthrough", "pitch_quantizer", "interaural"]
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
                "iau_width_ch2":           self.iau_width_ch2}

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
        return o


def _module_param_attrs(module_type: str) -> list:
    """Modulatable attr names for a module_type, derived from AnalyticModule.knobs()."""
    return [
        k.name for k in AnalyticModule.knobs()
        if k.visible_when == ("module_type", module_type)
    ]


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

def _patch_node_keys(
    patch: "AnalyticPatch",
    include_params:   bool = True,
    include_controls: bool = True,
) -> list:
    """Canonical ordered node key list:
    virtual-patch-nodes -> voices -> LFOs -> modules -> controls -> mixers -> param_nodes.

    The two virtual patch nodes (__patch_tonic__, __patch_seq__) carry Hz
    pitch-domain signals (not audio) and are excluded from auto-mix routing.
    """
    keys = list(_PATCH_VIRTUAL_KEYS)
    keys += [v.key for v in patch.voices]
    keys += [l.key for l in patch.lfos]
    for m in patch.modules:
        keys.append(m.key)
        if m.module_type == "interaural":
            keys.append(m.ch1_key())
            keys.append(m.ch2_key())
        elif m.module_type == "lfo" and m.lfo_channels:
            for i in range(len(m.lfo_channels)):
                keys.append(m.lfo_ch_key(i))
    if include_controls:
        for cs in patch.controls:
            keys += [sl.key for sl in cs.sliders]
    keys += [m.key for m in patch.mixers]
    if include_params:
        keys += [pn.key for pn in patch.param_nodes]
    return keys


def _routing_node_label(key: str, patch: "AnalyticPatch") -> str:
    if key == "__patch_tonic__":
        return f"Tonic ({_hz_to_note_name(patch.seq_tonic_hz, patch.tuning)})"
    if key == "__patch_seq__":
        return "Seq.Pitch"
    if key == "__mix__":
        return "Mix"
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
    if key == "__mix__":
        return (200, 200, 100)
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
    for cs in patch.controls:
        for sl in cs.sliders:
            if sl.key == key:
                return tuple(cs.color[:3])
    for pn in patch.param_nodes:
        if pn.key == key:
            return tuple(pn.color[:3])
    return (120, 120, 120)

@dataclass
class RhythmPattern:
    """Per-bar on/off step grid with per-step velocity."""
    name:  str  = "Pat"
    steps: list = field(default_factory=lambda: [False] * 16)
    vel:   list = field(default_factory=lambda: [1.0] * 16)

    def ensure_size(self, n: int) -> None:
        """Grow steps/vel lists to at least n slots."""
        while len(self.steps) < n:
            self.steps.append(False)
        while len(self.vel) < n:
            self.vel.append(1.0)

    def to_dict(self) -> dict:
        return {"name": self.name, "steps": list(self.steps), "vel": list(self.vel)}

    @classmethod
    def from_dict(cls, d: dict) -> "RhythmPattern":
        rp       = cls()
        rp.name  = d.get("name", "Pat")
        rp.steps = [bool(x) for x in d.get("steps", [])]
        rp.vel   = [float(x) for x in d.get("vel", [])]
        return rp


@dataclass
class AnalyticPatch:
    name:       str   = "untitled"
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
    seq_octave_span:     int   = 2
    seq_tonic_hz:        float = 440.0 # tonal center / scale root (key being performed)
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
    # Signal routing graph (analytic, pre-projection)
    routing: RoutingGraph = field(default_factory=RoutingGraph)
    # UI-only state — not serialized.  When set, only this voice key produces audio.
    solo_key: str | None = None
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
    # Progression probability transforms (0.0 = never, 1.0 = always)
    seq_probabilities: "SequenceProbabilities" = field(
        default_factory=lambda: SequenceProbabilities())
    # Velocity dynamics program (curve + accent grid)
    dynamics_program:  "DynamicsProgram" = field(
        default_factory=lambda: DynamicsProgram())
    # Stochastic ornament program (grace / chirp / echo)
    improv_program:    "ImprovProgram" = field(
        default_factory=lambda: ImprovProgram())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
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
            "seq_octave_span":     self.seq_octave_span,
            "seq_tonic_hz":        self.seq_tonic_hz,
            "seq_bass_octave":     self.seq_bass_octave,
            "seq_root_octave":     self.seq_root_octave,
            "seq_stab_octave":     self.seq_stab_octave,
            "seq_chord_prog":      self.seq_chord_prog,
            "seq_repeats":         self.seq_repeats,
            "seq_custom_semitones": self.seq_custom_semitones,
            "projection_mode":        self.projection_mode,
            "projection_rotation_hz": self.projection_rotation_hz,
            "normalize_output":       self.normalize_output,
            "routing":                self.routing.to_dict(),
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
            "seq_probabilities": self.seq_probabilities.to_dict(),
            "dynamics_program":  self.dynamics_program.to_dict(),
            "improv_program":    self.improv_program.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyticPatch":
        p = cls()
        p.name       = d.get("name", "untitled")
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
        p.seq_octave_span     = int(d.get("seq_octave_span", 2))
        p.seq_tonic_hz        = float(d.get("seq_tonic_hz", p.tuning.root_hz))
        p.seq_bass_octave     = int(d.get("seq_bass_octave", -1))
        p.seq_root_octave     = int(d.get("seq_root_octave", -2))
        p.seq_stab_octave     = int(d.get("seq_stab_octave",  1))
        p.seq_chord_prog      = d.get("seq_chord_prog", "I_IV_V_I")
        p.seq_repeats         = int(d.get("seq_repeats", 2))
        p.seq_custom_semitones = d.get("seq_custom_semitones", "")
        p.projection_mode        = d.get("projection_mode", "mono")
        p.projection_rotation_hz = float(d.get("projection_rotation_hz", 0.0))
        p.normalize_output       = bool(d.get("normalize_output", True))
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
        _raw_imp = d.get("improv_program")
        if _raw_imp and isinstance(_raw_imp, dict):
            p.improv_program = ImprovProgram.from_dict(_raw_imp)
        if "routing" in d:
            p.routing = RoutingGraph.from_dict(d["routing"])
            # H1 fix: prune stale edges whose node keys no longer exist in the patch
            valid_keys = _patch_node_keys(p)
            p.routing.prune_keys(valid_keys)
        else:
            p.routing = RoutingGraph()
        return p

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _PROJ    = ["mono", "stereo_quadrature", "stereo_ms", "lissajous"]
        _SR      = [8000.0, 22050.0, 44100.0, 48000.0, 96000.0, 192000.0]
        _PRESETS = list(GLOBAL_TUNING_PRESETS.keys())
        _TEMPS   = GlobalTuning._TEMPERAMENT_CHOICES
        return [
            # Global
            KnobSpec("preview_sr",           "Sample rate", "int",   48000, 8000, 192000, 0, "Hz", [], True,  "Global", ".0f", "", True),
            KnobSpec("duration",             "Duration",    "float", 2.0,   0.1,  60.0,   0, "s",  [], False, "Global", ".2f", "", True),
            # Projection
            KnobSpec("projection_mode",        "Proj mode",  "choice", "mono", 0, 3, 1, "", _PROJ, False, "Projection", ".0f", "", True),
            KnobSpec("projection_rotation_hz", "Rot Hz",     "float",  0.0, -200.0, 200.0, 0, "Hz", [], False, "Projection", ".2f"),
            KnobSpec("normalize_output",       "Normalize",  "bool",   True, 0, 1, 1, "",  [], False, "Projection", ""),
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
    CHIRP         = auto()
    COMPLEX_RI    = auto()   # real and imaginary components
    COMPLEX_MP    = auto()   # magnitude and phase
    HARMONICS     = auto()   # harmonic amplitude spectrum bars
    LFO_VIEW      = auto()   # all LFO waveforms over time
    FM_VIEW       = auto()   # instantaneous frequency including FM deviation
    MIX           = auto()   # all voices summed into final mix
    ROUTING       = auto()   # N×N signal routing matrix (knob grid)
    PARAM_ROUTING = auto()   # parametric routing view for ParamNode targets


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
    knots = voice.active_knots()
    ts = np.array([k[0] * duration for k in knots], dtype=np.float64)
    vs = np.array([k[1]            for k in knots], dtype=np.float64)
    t_ax = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
    etype = getattr(voice, "env_type", "adsr")
    if etype == "spline" and len(knots) >= 4:
        try:
            from scipy.interpolate import CubicSpline
            return np.clip(CubicSpline(ts, vs, bc_type="clamped")(t_ax), 0.0, None)
        except Exception:
            pass
    elif etype == "monotone" and len(knots) >= 2:
        try:
            from scipy.interpolate import PchipInterpolator
            return np.clip(PchipInterpolator(ts, vs)(t_ax), 0.0, None)
        except Exception:
            pass
    # "adsr" and "linear" (and fallback)
    return np.interp(t_ax, ts, vs)


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
) -> "NoteSchedule":
    """Convert rhythm program + scale degrees into a NoteSchedule.

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
    Gate:    note duration = ``rhythm_gate × step_s``.

    After the schedule is built, ``apply_dynamics`` is called with
    ``p.dynamics_program`` to apply the velocity curve and accent grid.
    """
    sched      = NoteSchedule()
    div        = max(1, p.rhythm_division)
    step_s     = beat_s * 4.0 / div
    gate_s     = max(min_dur_s, step_s * p.rhythm_gate)
    pkt_s      = p.rhythm_pocket * beat_s
    phrase     = p.rhythm_phrase if p.rhythm_phrase else [0]
    pats       = p.rhythm_patterns if p.rhythm_patterns else [RhythmPattern(name="Pat 1")]
    n_pat      = len(deg_pattern)
    prog_bars  = max(1, getattr(p, "rhythm_prog_bars", 1))
    fit_mode   = getattr(p, "rhythm_fit_mode", "drop")
    probs      = getattr(p, "seq_probabilities", None) or SequenceProbabilities()
    _rng       = _random.Random()   # local instance — doesn't touch global state
    stream     = NoteStream(degrees, deg_pattern, probs, _rng)

    def _onsets_in_bars(start_bar: int, end_bar: int) -> int:
        total = 0
        for b in range(start_bar, end_bar):
            slot  = b % len(phrase)
            pat_i = phrase[slot]
            pat   = pats[min(pat_i, len(pats) - 1)]
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

    for rep in range(max(1, p.seq_repeats)):
        stream.reset()   # each repeat restarts the progression identically
        for bar_i in range(cycle_bars):
            slot  = bar_i % len(phrase)
            pat_i = phrase[slot]
            pat   = pats[min(pat_i, len(pats) - 1)]
            pat.ensure_size(div)
            abs_bar = rep * cycle_bars + bar_i
            for step_i in range(div):
                if not pat.steps[step_i]:
                    continue
                # Pull next Hz from the stream; None = progression exhausted → rest
                hz = stream.next_hz()
                if hz is None:
                    continue
                t_grid  = abs_bar * div * step_s + step_i * step_s
                if step_i % 2 == 1:
                    t_grid += p.rhythm_swing * step_s
                t_final = max(0.0, t_grid + pkt_s)
                vel     = float(pat.vel[step_i]) if step_i < len(pat.vel) else 1.0
                sched.add(NoteEvent(hz, t_final, gate_s, velocity=vel))

    # ── Post-process: apply velocity dynamics (curve + accent grid) ──────────
    dyn_prog = getattr(p, "dynamics_program", None)
    if dyn_prog is not None and _HAS_DYN_ENG:
        apply_dynamics(sched.events, dyn_prog, beat_s, div, _rng)

    # ── Post-process: apply stochastic ornaments (grace / chirp / echo) ──────
    imp_prog = getattr(p, "improv_program", None)
    if imp_prog is not None and _HAS_IMPROV_ENG and imp_prog.enabled:
        phrase_ref = p.rhythm_phrase if p.rhythm_phrase else [0]
        pats_ref   = p.rhythm_patterns if p.rhythm_patterns else []
        extra = apply_improv(
            sched.events,
            imp_prog,
            beat_s,
            div,
            pats_ref,
            phrase_ref,
            cycle_bars,
            _rng,
        )
        if extra:
            sched.events.extend(extra)
            sched.events.sort(key=lambda e: e.start_time)

    return sched


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
    t = (np.arange(n, dtype=np.float64) / sr) + t_offset

    # Apply param_overrides to base frequency/amplitude before chirp/FM
    po = param_overrides or {}
    _po_freq = po.get("freq_hz")
    if _po_freq is not None and len(_po_freq) >= n:
        f_inst = np.asarray(_po_freq[:n], dtype=np.float64)
    else:
        f_inst = np.full(n, voice.freq_hz, dtype=np.float64)
    ct = voice.chirp.chirp_type
    if ct == "linear":
        f_inst += np.linspace(voice.chirp.f_delta_start, voice.chirp.f_delta_end, n)
    elif ct == "exponential" and voice.chirp.tau > 0:
        decay   = np.exp(-t / voice.chirp.tau)
        f_inst += voice.chirp.f_delta_start * decay + voice.chirp.f_delta_end * (1 - decay)
    elif ct == "power" and dur > 0:
        tau_n = (t / dur) ** max(voice.chirp.chirp_power, 1e-3)
        f_inst += voice.chirp.f_delta_start * (1.0 - tau_n) + voice.chirp.f_delta_end * tau_n

    if voice.fm and voice.fm.source_key:
        sk = voice.fm.source_key
        if sk in lfo_map:
            mod = _lfo_signal(lfo_map[sk], t)
        elif voice_signal_map is not None and sk in voice_signal_map:
            # Use the source voice's synthesized complex signal: extract
            # instantaneous frequency (normalised to [-0.5, 0.5] * Nyquist)
            src_csig = voice_signal_map[sk]
            f_mod = _inst_freq_from_csig(src_csig[:n], float(patch.preview_sr))
            f_mid = float(p_map[sk].freq_hz) if sk in p_map else float(np.mean(np.abs(f_mod)))
            mod = f_mod / max(f_mid, 1.0)  # normalise so depth_hz is in sensible units
        elif param_series is not None and sk in param_series:
            # H4 fix: ParamNode output as FM modulator (already a float64 series)
            ps = param_series[sk]
            mod = np.asarray(ps[:n], dtype=np.float64) if len(ps) >= n else np.pad(ps[:n], (0, n - len(ps)))
        elif sk in p_map and sk != voice.key:
            mod = np.cos(2.0 * np.pi * p_map[sk].freq_hz * t)
        else:
            mod = np.zeros(n)
        f_inst += voice.fm.depth_hz * mod

    phase = np.cumsum(2.0 * np.pi * f_inst / sr) + voice.phase_origin

    _po_amp = po.get("amplitude")
    if _po_amp is not None and len(_po_amp) >= n:
        amp = np.asarray(_po_amp[:n], dtype=np.float64)
    else:
        amp = np.full(n, voice.amplitude, dtype=np.float64)
    if voice.am and voice.am.source_key:
        sk = voice.am.source_key
        if sk in lfo_map:
            mod = _lfo_signal(lfo_map[sk], t)
        elif voice_signal_map is not None and sk in voice_signal_map:
            # Use magnitude envelope of the source voice's synthesized signal
            src_csig = voice_signal_map[sk]
            mod = np.abs(src_csig[:n]).astype(np.float64)
            peak = float(np.max(mod))
            if peak > 1e-12:
                mod /= peak
        elif param_series is not None and sk in param_series:
            # H4 fix: ParamNode output as AM modulator (already a float64 series)
            ps = param_series[sk]
            mod = np.asarray(ps[:n], dtype=np.float64) if len(ps) >= n else np.pad(ps[:n], (0, n - len(ps)))
        elif sk in p_map and sk != voice.key:
            mod = np.cos(2.0 * np.pi * p_map[sk].freq_hz * t)
        else:
            mod = np.zeros(n)
        amp *= (1.0 + voice.am.depth_amp * mod)

    env = amp * _compute_envelope(voice, n, dur)

    # --- manifold synthesis ---
    mt = voice.manifold_type
    if mt in ("harmonic", "harmonic_warp") and voice.harmonic_count > 1:
        sig = np.zeros(n, dtype=np.complex128)
        hc  = max(1, voice.harmonic_count)
        bri = voice.harmonic_brightness
        warp = voice.harmonic_warp_strength
        for k in range(1, hc + 1):
            h_ratio = k + warp * (k - 1)  # warp=0 → exact integer multiples
            h_amp   = 1.0 / (k ** bri) if bri > 0 else 1.0
            # C2 fix: k-th partial starts at k * phase_origin so all harmonics
            # are constructive at t=0 even when ratios are non-integer (warp > 0).
            h_phase = np.cumsum(2.0 * np.pi * (f_inst * h_ratio) / sr) + (k * voice.phase_origin) % (2.0 * np.pi)
            sig    += h_amp * np.exp(1j * h_phase)
        # Normalise so amplitude 1 still means peak ~1 for a single harmonic baseline
        norm = sum(1.0 / (k ** bri) if bri > 0 else 1.0 for k in range(1, hc + 1))
        sig /= norm
        out = env * sig
    else:
        out = env * np.exp(1j * phase)

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


def _synthesize_patch(
    patch: AnalyticPatch,
    *,
    granular_seed_offset: int = 0,
    file_render: bool = False,
    _return_mixer_sigs: bool = False,
    _return_sidecar: bool = False,
) -> tuple:
    """Return (left, right) float32 stereo after routing + projection.

    Routing model (N nodes = voices + LFOs + __mix__):
        x = src + W @ x  →  x = (I − W)⁻¹ · src          (instantaneous)
        x(t) = src(t) + decay · W @ x(t − d)               (delayed)

    When no routing edges exist, falls back to direct voice sum (backward-compat).
    The '__mix__' node output is the stereo output before projection.

    When *_return_sidecar=True* the return value gains a trailing SidecarBus:
        (left, right)                           default
        (left, right, sidecar)                  _return_sidecar=True
        (left, right, mixer_outs)               _return_mixer_sigs=True
        (left, right, mixer_outs, sidecar)      both True
    """
    lfo_map = {l.key: l for l in patch.lfos}
    p_map   = {p.key: p for p in patch.voices}
    sr      = float(patch.preview_sr)
    n       = int(patch.preview_sr * patch.duration)

    # 1. Synthesize independent sources for each node
    # voice_sigs is built incrementally so that each voice can use already-synthesized
    # voices as FM/AM sources via voice_signal_map (C1 fix).
    voice_sigs: dict[str, np.ndarray] = {}
    for voice in patch.voices:
        if _voice_effectively_muted(voice, patch):
            voice_sigs[voice.key] = np.zeros(n, dtype=np.complex128)
        else:
            voice_sigs[voice.key] = _synthesize_voice(
                voice, patch, lfo_map, p_map, voice_signal_map=voice_sigs,
                granular_rng_seed=(None if file_render else _GRANULAR_SEED_EDITOR),
                granular_seed_offset=granular_seed_offset)

    lfo_sigs: dict[str, np.ndarray] = {}
    for lfo in patch.lfos:
        lfo_sigs[lfo.key] = _synthesize_lfo_csig(lfo, n, sr)

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
        else:  # passthrough / pitch_quantizer / interaural — zero independent source
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
    # Modules: amplitude and phase
    for mkey, msig in module_sigs.items():
        sidecar.put(mkey, "amplitude", np.abs(msig))
        sidecar.put(mkey, "phase",     np.angle(msig))
    # Control sliders: scalar value broadcast to an array
    for cs in patch.controls:
        for sl in cs.sliders:
            sidecar.put(sl.key, "value",
                        np.full(n, sl.scaled_value(), dtype=np.float64))

    g = patch.routing
    # Auto-route voices, LFOs, and modules to the mix bus by default.
    # Control slider nodes and virtual patch nodes are excluded -- they target
    # ParamNodes / pitch_quantizer modules only, never the audio mix bus.
    mixer_keys = [m.key for m in patch.mixers]
    default_mix_key = mixer_keys[0] if mixer_keys else "__mix__"
    auto_signal_keys = (
        [v.key for v in patch.voices] +
        [l.key for l in patch.lfos] +
        [m.key for m in patch.modules
         if m.module_type not in ("interaural",)
         and not (m.module_type == "lfo" and m.lfo_channels)] +
        [m.lfo_ch_key(i)
         for m in patch.modules if m.module_type == "lfo" and m.lfo_channels
         for i in range(len(m.lfo_channels))]
    )
    g.ensure_defaults(auto_signal_keys, mix_key=default_mix_key)

    # 2. Build N-node routing system (voices + lfos + modules + ctrl-sliders + mixers + param_nodes)
    node_keys = _patch_node_keys(patch)   # all nodes including param nodes and controls
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
        voice_sigs = {}
        for voice in patch.voices:
            if _voice_effectively_muted(voice, patch):
                voice_sigs[voice.key] = np.zeros(n_ext, dtype=np.complex128)
            else:
                lead_n = lead_map.get(voice.key, 0)
                lead_s = lead_n / sr
                # Synthesise over [-lead_s, duration) so delayed signal arrives at t=0
                voice_sigs[voice.key] = _synthesize_voice(
                    voice, patch, lfo_map, p_map,
                    t_offset=-lead_s, n_samples=n_ext,
                    voice_signal_map=voice_sigs,
                    granular_rng_seed=(None if file_render else _GRANULAR_SEED_EDITOR),
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
                hz_in  = row.real
                hz_out = np.empty(hz_in.shape[0], dtype=np.float64)
                for _qi in range(hz_in.shape[0]):
                    hz_out[_qi] = _h(
                        float(hz_in[_qi]), float(hz_in[_qi]),
                        domain="hz", dt=_dt,
                    )
                # Only the real (Hz) component is quantized; imaginary stays intact.
                return hz_out.astype(np.complex128) + 1j * row.imag
            return _qtransform
        _node_transforms[_qmod.key] = _make_qtransform()

    # 3. Solve the routing system via complex per-edge solver
    # Extends the buffer by a ringdown tail so feedback/echo decays to silence
    # rather than being hard-truncated at the note boundary.
    fb = g.feedback
    global_decay = max(0.0, 1.0 - float(fb.decay)) if fb.enabled else 1.0
    X, _ = solve_routing_with_ringdown(
        Src, g.edges, node_keys, sr, global_decay, fb,
        node_transforms=_node_transforms or None,
    )

    # 3a. Interaural post-solve: apply spatial placement per module.
    # ch1 and ch2 node rows in X hold the accumulated routed inputs.
    # We spatialize them and write the outputs back, then delta-update any
    # downstream node (typically a mixer) that has edges from ch1/ch2.
    _iau_edge_dst: dict = {}   # dst_key -> list of (src_key, weight, angle_rad)
    for _e in g.edges:
        _iau_edge_dst.setdefault(_e.dst_key, []).append(_e)
    for _imod in patch.modules:
        if _imod.module_type != "interaural" or _imod.muted:
            continue
        _ck1 = _imod.ch1_key()
        _ck2 = _imod.ch2_key()
        if _ck1 not in ki or _ck2 not in ki:
            continue
        _T    = X.shape[1]
        _ones = np.ones(_T, dtype=np.float64)
        # Detect stereo: any edge targets ch2?
        _ch2_has_input = any(e.dst_key == _ck2 for e in g.edges)
        # Read accumulated inputs
        _inp1 = X[ki[_ck1]].copy()
        _inp2 = X[ki[_ck2]].copy() if _ch2_has_input else _inp1.copy()
        # Build param arrays (static knob values; param-node overrides added later)
        _az1  = _ones * _imod.iau_azimuth
        _el1  = _ones * _imod.iau_elevation
        _dst1 = _ones * _imod.iau_distance
        _wid1 = _ones * _imod.iau_width
        if _ch2_has_input:
            _az2  = _ones * _imod.iau_azimuth_ch2
            _el2  = _ones * _imod.iau_elevation_ch2
            _dst2 = _ones * _imod.iau_distance_ch2
            _wid2 = _ones * _imod.iau_width_ch2
        else:
            _az2, _el2, _dst2, _wid2 = _az1, _el1, _dst1, _wid1
        # Spatialize: each input produces (ch1_contribution, ch2_contribution)
        _out1_a, _out2_a = _place_signal(_inp1, _az1, _el1, _dst1, _wid1)
        _out1_b, _out2_b = _place_signal(_inp2, _az2, _el2, _dst2, _wid2)
        _new1 = _out1_a + _out1_b
        _new2 = _out2_a + _out2_b
        # Delta-update downstream nodes that receive from ch1 or ch2
        for _src_key, _new_sig, _old_sig in (
            (_ck1, _new1, _inp1),
            (_ck2, _new2, _inp2),
        ):
            _delta = _new_sig - _old_sig
            for _e in g.edges:
                if _e.src_key != _src_key or _e.delay_s != 0.0:
                    continue
                _di = ki.get(_e.dst_key)
                if _di is None:
                    continue
                _cw = complex(
                    _e.weight * global_decay * math.cos(_e.angle_rad),
                    _e.weight * global_decay * math.sin(_e.angle_rad),
                )
                X[_di] += _cw * _delta
        # Write spatialized signals into the ch1/ch2 rows
        X[ki[_ck1]] = _new1
        X[ki[_ck2]] = _new2

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

    # 3b. Param-routing evaluation pass.
    # For each ParamNode: extract a float64 time series from the routing output,
    # then re-synthesise any voice whose attributes are targeted by a param node.
    # A second routing solve produces the final X used for output.
    if patch.param_nodes:
        # Extract each param node's float64 time series directly from its own
        # routing output (X[ki[pn.key]]), which already accumulated all signal
        # routed into it via RoutingEdges — no separate ParamEdge list needed.
        X_cols = X.shape[1]
        param_series: dict = {}
        for pn in patch.param_nodes:
            if pn.key not in ki:
                continue
            raw = extract_param_series(X[ki[pn.key], :min(n, X_cols)], pn.extractor)
            lo, hi = float(pn.low), float(pn.high)
            span = hi - lo
            # L3 fix: when no routing edge targets this ParamNode (graph fact, not
            # signal fact), use default_value.  A live edge with a zero signal means
            # the control is genuinely at zero and must map to lo — don't override it.
            is_driven = any(pe.dst_key == pn.key for pe in patch.routing.param_edges)
            if not is_driven:
                param_series[pn.key] = np.full(len(raw), np.clip(float(pn.default_value), lo, hi))
                continue
            # Normalise to [0,1] from the observed range, then scale to [low, high]
            if span > 0:
                rmin, rmax = float(raw.min()), float(raw.max())
                rspan = rmax - rmin
                if rspan > 1e-12:
                    raw = (raw - rmin) / rspan * span + lo
                else:
                    raw = np.full_like(raw, lo + span * 0.5)
            param_series[pn.key] = np.clip(raw, lo, hi)
        # Build per-voice and per-module param override dicts
        _iau_mod_keys: set = {m.key for m in patch.modules if m.module_type == "interaural"}
        _iau_ch_to_mod: dict = {}
        for _m in patch.modules:
            if _m.module_type == "interaural":
                _iau_ch_to_mod[_m.ch1_key()] = _m
                _iau_ch_to_mod[_m.ch2_key()] = _m
                _iau_ch_to_mod[_m.key]       = _m
        voice_param_overrides: dict = {}
        module_param_overrides: dict = {}   # mod.key -> {attr: series}
        for pn in patch.param_nodes:
            if pn.key not in param_series:
                continue
            for tgt in pn.targets:
                vk = tgt.get("voice_key", "")
                at = tgt.get("attr", "")
                if not (vk and at):
                    continue
                if vk in _iau_ch_to_mod:
                    _m = _iau_ch_to_mod[vk]
                    module_param_overrides.setdefault(_m.key, {})[at] = param_series[pn.key]
                elif vk not in _iau_mod_keys:
                    voice_param_overrides.setdefault(vk, {})[at] = param_series[pn.key]
        # Re-synthesise modulated voices and rebuild Src for final routing solve
        if voice_param_overrides:
            Src2 = np.zeros_like(Src)
            for key, sig in lfo_sigs.items():
                if key in ki:
                    Src2[ki[key]] = sig[:Src2.shape[1]]
            voice_sigs2: dict[str, np.ndarray] = {}
            for voice in patch.voices:
                if _voice_effectively_muted(voice, patch):
                    Src2[ki[voice.key]] = 0.0
                    voice_sigs2[voice.key] = np.zeros(n_ext, dtype=np.complex128)
                elif voice.key in voice_param_overrides:
                    lead_s = lead_map.get(voice.key, 0) / sr if max_lead > 0 else 0.0
                    n_syn  = n_ext
                    t_off  = -lead_s if max_lead > 0 else 0.0
                    sig = _synthesize_voice(
                        voice, patch, lfo_map, p_map, t_off, n_syn,
                        param_overrides=voice_param_overrides[voice.key],
                        voice_signal_map=voice_sigs2,
                        param_series=param_series,
                        granular_rng_seed=(None if file_render else _GRANULAR_SEED_EDITOR),
                        granular_seed_offset=granular_seed_offset,
                    )[:n_ext]
                    voice_sigs2[voice.key] = sig
                    Src2[ki[voice.key]] = sig
                elif voice.key in ki:
                    Src2[ki[voice.key]] = voice_sigs[voice.key][:n_ext]
            X, _ = solve_routing_with_ringdown(
                Src2, g.edges, node_keys, sr, global_decay, fb,
                node_transforms=_node_transforms or None)
            if max_lead > 0:
                X = X[:, max_lead: max_lead + n + estimate_ringdown_samples(
                    g.edges, sr, global_decay, fb)]
            # Re-run interaural post-solve on the updated X
            for _imod in patch.modules:
                if _imod.module_type != "interaural" or _imod.muted:
                    continue
                _ck1 = _imod.ch1_key()
                _ck2 = _imod.ch2_key()
                if _ck1 not in ki or _ck2 not in ki:
                    continue
                _T    = X.shape[1]
                _ones = np.ones(_T, dtype=np.float64)
                _ch2_has_input = any(e.dst_key == _ck2 for e in g.edges)
                _inp1 = X[ki[_ck1]].copy()
                _inp2 = X[ki[_ck2]].copy() if _ch2_has_input else _inp1.copy()
                _ov   = module_param_overrides.get(_imod.key, {})
                def _get(attr, default, T=_T):
                    v = _ov.get(attr)
                    if v is None:
                        return _ones * default
                    a = np.asarray(v, dtype=np.float64)
                    return np.pad(a, (0, max(0, T - len(a))), mode="edge")[:T]
                _az1  = _get("iau_azimuth",    _imod.iau_azimuth)
                _el1  = _get("iau_elevation",  _imod.iau_elevation)
                _dst1 = _get("iau_distance",   _imod.iau_distance)
                _wid1 = _get("iau_width",      _imod.iau_width)
                if _ch2_has_input:
                    _az2  = _get("iau_azimuth_ch2",   _imod.iau_azimuth_ch2)
                    _el2  = _get("iau_elevation_ch2", _imod.iau_elevation_ch2)
                    _dst2 = _get("iau_distance_ch2",  _imod.iau_distance_ch2)
                    _wid2 = _get("iau_width_ch2",     _imod.iau_width_ch2)
                else:
                    _az2, _el2, _dst2, _wid2 = _az1, _el1, _dst1, _wid1
                _out1_a, _out2_a = _place_signal(_inp1, _az1, _el1, _dst1, _wid1)
                _out1_b, _out2_b = _place_signal(_inp2, _az2, _el2, _dst2, _wid2)
                _new1 = _out1_a + _out1_b
                _new2 = _out2_a + _out2_b
                for _src_key, _new_sig, _old_sig in (
                    (_ck1, _new1, _inp1), (_ck2, _new2, _inp2),
                ):
                    _delta = _new_sig - _old_sig
                    for _e in g.edges:
                        if _e.src_key != _src_key or _e.delay_s != 0.0:
                            continue
                        _di = ki.get(_e.dst_key)
                        if _di is None:
                            continue
                        _cw = complex(
                            _e.weight * global_decay * math.cos(_e.angle_rad),
                            _e.weight * global_decay * math.sin(_e.angle_rad),
                        )
                        X[_di] += _cw * _delta
                X[ki[_ck1]] = _new1
                X[ki[_ck2]] = _new2

    # 4. Sum all projection-active mixers, optionally normalize, then project.
    # Meta-mixers (projection_active=False) contribute analytically to other
    # mixer nodes via routing edges but never reach the PCM bus.
    active_mixer_keys = [m.key for m in patch.mixers if m.projection_active]
    if not active_mixer_keys:
        # No active output mixer — silence
        _silence = (np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32))
        if _return_mixer_sigs and _return_sidecar:
            return (*_silence, {}, sidecar)
        if _return_mixer_sigs:
            return (*_silence, {})
        if _return_sidecar:
            return (*_silence, sidecar)
        return _silence
    mix = sum(X[ki[mk]] for mk in active_mixer_keys if mk in ki)
    if patch.normalize_output:
        peak = float(np.max(np.abs(mix)))
        if peak > 1e-9:
            mix /= peak
    out = _apply_projection(mix, patch.projection_mode,
                            patch.projection_rotation_hz, sr)
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
        if _return_sidecar:
            return (*out, _mixer_outs, sidecar)
        return (*out, _mixer_outs)
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
        keys   = _patch_node_keys(patch)
        labels = [_routing_node_label(k, patch) for k in keys]
        colors = [_routing_node_color(k, patch) for k in keys]
        N      = len(keys)
        self._N = N
        g      = patch.routing
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

        # Export tab — show per-mixer export settings; skip NxN routing grid
        if self._routing_tab == "export":
            self._draw_export_panel(surf, patch, font, w, h, TAB_H)
            return

        # ---- Scrollable grid geometry -----------------------------------------
        # Two header strips on each axis (top+bottom, left+right).
        status_h = (fh + 4) * 2 + 10
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
        keys_list = _patch_node_keys(patch)
        g = patch.routing
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

    MODES = [EditorMode.WAVEFORM, EditorMode.ENVELOPE, EditorMode.CHIRP,
             EditorMode.COMPLEX_RI, EditorMode.COMPLEX_MP,
             EditorMode.HARMONICS, EditorMode.LFO_VIEW, EditorMode.FM_VIEW,
             EditorMode.MIX, EditorMode.ROUTING]
    MODE_LABELS = ["Wave", "Envelope", "Chirp", "Re/Im", "Mag/Phase",
                   "Harmonics", "LFOs", "FM", "Mix", "Routing"]

    # Modes available on the __mix__ node (routing module)
    _MIX_NODE_MODES  = [EditorMode.ROUTING, EditorMode.MIX,
                        EditorMode.COMPLEX_RI, EditorMode.COMPLEX_MP]
    _MIX_NODE_LABELS = ["Routing", "Mix", "Re/Im", "Mag/Phase"]

    # Modes available on LFO nodes
    _LFO_NODE_MODES   = [EditorMode.WAVEFORM, EditorMode.LFO_VIEW, EditorMode.COMPLEX_RI]
    _PARAM_NODE_MODES  = [EditorMode.PARAM_ROUTING]
    _PARAM_NODE_LABELS = ["Param Routing"]
    _LFO_NODE_LABELS = ["Wave", "LFOs", "Re/Im"]

    # Modes available on regular voice nodes (everything except ROUTING)
    _VOICE_MODES  = [EditorMode.WAVEFORM, EditorMode.ENVELOPE, EditorMode.CHIRP,
                     EditorMode.COMPLEX_RI, EditorMode.COMPLEX_MP,
                     EditorMode.HARMONICS, EditorMode.LFO_VIEW, EditorMode.FM_VIEW,
                     EditorMode.MIX]
    _VOICE_LABELS = ["Wave", "Envelope", "Chirp", "Re/Im", "Mag/Phase",
                     "Harmonics", "LFOs", "FM", "Mix"]

    def _visible_modes(self, patch: "AnalyticPatch") -> tuple[list, list]:
        """Return (modes, labels) appropriate for the currently active node."""
        if any(m.key == self.active_key for m in patch.mixers):
            return self._MIX_NODE_MODES, self._MIX_NODE_LABELS
        if any(l.key == self.active_key for l in patch.lfos):
            return self._LFO_NODE_MODES, self._LFO_NODE_LABELS
        if any(pn.key == self.active_key for pn in patch.param_nodes):
            return self._PARAM_NODE_MODES, self._PARAM_NODE_LABELS
        return self._VOICE_MODES, self._VOICE_LABELS

    def __init__(self) -> None:
        self.mode: EditorMode = EditorMode.WAVEFORM
        self.plot   = PlotWidget()
        self.plot.grid_lines = 5
        self.routing_view = RoutingGridView()
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
        self.active_key: str = ""

        # Seed offset for granular voices — advanced by viewer when seed_animate is on
        self.granular_seed_offset: int = 0

        # Mix-tab per-series visibility toggles and signal cache
        self._mix_series_hidden: set[str] = set()
        self._mix_toggle_rects:  list[dict] = []
        self._mix_series_info:   list[dict] = []
        self._mix_voice_cache:   dict[str, np.ndarray] = {}
        self._mix_left_cache:    "np.ndarray | None" = None
        self._mix_right_cache:   "np.ndarray | None" = None
        self._mix_t_ax:          "np.ndarray | None" = None

        # Hover state
        self._hover_cp: int = -1   # index of hovered knot (-1 = none)

        # Background rebuild infrastructure
        self._rebuild_thread: threading.Thread | None = None
        self._rebuild_cancel: threading.Event = threading.Event()
        self._rebuild_result: dict | None = None
        self._rebuild_lock = threading.Lock()
        self._rebuild_hash: str = ""  # fingerprint of last completed rebuild

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

    # ---- Data refresh ------------------------------------------------------

    @staticmethod
    def _patch_fingerprint(patch: "AnalyticPatch", active_key: str,
                           mode, show_tail: bool, seed: int) -> str:
        import hashlib
        blob = json.dumps(patch.to_dict(), sort_keys=True, default=str)
        blob += f"|{active_key}|{mode}|{show_tail}|{seed}"
        return hashlib.sha256(blob.encode()).hexdigest()

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

        # --- special case: mixer node selected ---
        is_mixer_node = any(m.key == active_key for m in patch.mixers)
        if is_mixer_node:
            # Synthesize routing system and expose the raw complex mix signal
            lfo_map = {l.key: l for l in patch.lfos}
            p_map   = {v.key: v for v in patch.voices}
            _g = patch.routing
            _mixer_keys_all = [m.key for m in patch.mixers]
            _default_mix_key = _mixer_keys_all[0] if _mixer_keys_all else active_key
            _g.ensure_defaults(_patch_node_keys(patch), mix_key=_default_mix_key)
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
        else:
            # --- waveform (complex analytic) ---
            lfo_map = {l.key: l for l in patch.lfos}
            p_map   = {p.key: p for p in patch.voices}
            if cancel.is_set(): return
            if voice and not _voice_effectively_muted(voice, patch):
                csig = _synthesize_voice(voice, patch, lfo_map, p_map,
                                         granular_seed_offset=self.granular_seed_offset)
                if cancel.is_set(): return
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
            f_base = voice.freq_hz
            if voice.chirp.chirp_type == "linear":
                self._chirp_f = (f_base + np.linspace(
                    voice.chirp.f_delta_start,
                    voice.chirp.f_delta_end, n)).astype(np.float32)
            elif voice.chirp.chirp_type == "exponential":
                t_sec = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
                tau = max(voice.chirp.tau, 1e-9)
                dec = np.exp(-t_sec / tau)
                self._chirp_f = (
                    f_base
                    + voice.chirp.f_delta_start * dec
                    + voice.chirp.f_delta_end * (1 - dec)
                ).astype(np.float32)
            elif voice.chirp.chirp_type == "power" and dur > 0:
                t_sec = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
                tau_n = (t_sec / dur) ** max(voice.chirp.chirp_power, 1e-3)
                self._chirp_f = (
                    f_base
                    + voice.chirp.f_delta_start * (1.0 - tau_n)
                    + voice.chirp.f_delta_end * tau_n
                ).astype(np.float32)
            else:
                self._chirp_f = np.full(n, f_base, dtype=np.float32)
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
            env = self._env_curve  # zeros — ENVELOPE mode not valid for __mix__
        else:
            col = tuple(voice.color[:3]) if voice else (100, 160, 255)
            env = self._env_curve

        self.plot.series.clear()
        self.plot.markers.clear()

        if self.mode == EditorMode.WAVEFORM:
            self.plot.add_series(PlotSeries(
                key="wave", label=voice.label if voice else "wave",
                color=col, line=True, dots=False,
                data_x=t_ax, data_y=sig,
            ))
            self.plot.y_min, self.plot.y_max = -1.05, 1.05
            self.v_lo, self.v_hi = -1.05, 1.05

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
            self.plot.y_min, self.plot.y_max = -1.05, 1.05
            self.v_lo, self.v_hi = -1.05, 1.05
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
            self.plot.y_min, self.plot.y_max = -1.05, 1.05
            self.v_lo, self.v_hi = -1.05, 1.05

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

        # ── Mode-tab background ──
        _gl_rect(cx, cy, cw, MODEBAR_H, win_w, win_h, (0.08, 0.08, 0.11, 1.0))
        vis_modes, vis_labels = self._visible_modes(patch)
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
        voice = next((p for p in patch.voices if p.key == self.active_key), None)

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
        self._rhythm_step_rects:      list       = []
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
        self._dyn_curve_left_rect: Any        = None
        self._dyn_curve_right_rect:Any        = None
        self._dyn_scope_dec_rect:  Any        = None
        self._dyn_scope_inc_rect:  Any        = None
        self._dyn_accent_rects:    list       = []
        self._dyn_sliders:         list[dict] = []
        self._dragging_dyn_slider: int        = -1
        # Improv section geometry
        self._improv_collapsed:         bool       = True
        self._improv_grace_collapsed:   bool       = True
        self._improv_chirp_collapsed:   bool       = True
        self._improv_echo_collapsed:    bool       = True
        self._improv_hdr_rect:          Any        = None
        self._improv_enable_rect:       Any        = None
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
        sl["val"] = val
        key = sl["key"]
        if key == "seq_bpm":
            p.seq_bpm = val
        elif key == "seq_legato":
            p.seq_legato = val
        elif key == "seq_portamento_s":
            p.seq_portamento_s = val

    def _apply_rhythm_slider(self, sl: dict, lx: int) -> None:
        """Write rhythm slider value to the patch."""
        p = self._patch
        if p is None:
            return
        r    = sl["rect"]
        frac = max(0.0, min(1.0, (lx - r.x) / max(r.w, 1)))
        val  = sl["lo"] + frac * (sl["hi"] - sl["lo"])
        sl["val"] = val
        key = sl["key"]
        if key == "rhythm_swing":
            p.rhythm_swing  = val
        elif key == "rhythm_pocket":
            p.rhythm_pocket = val
        elif key == "rhythm_gate":
            p.rhythm_gate   = val

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
        dp = getattr(p, "dynamics_program", None)
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
        ip = getattr(p, "improv_program", None)
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
            seq_body_h = seq_row_h * 5 + 32 * 3 + btn_h + 8  # pickers + sliders + btns
        seq_sec_h = hdr_h + seq_body_h + 4

        prob_body_h = 0
        if not self._prob_collapsed:
            prob_body_h = 32 * 4 + 8  # 4 probability sliders
        prob_sec_h = hdr_h + prob_body_h + 4

        dyn_body_h = 0
        if not self._dyn_collapsed and _HAS_DYN_ENG:
            _rdiv_d    = getattr(p, "rhythm_division", 16)
            _grid_rows_d = max(1, (_rdiv_d + 7) // 8)
            _cell_h_d    = 16
            dyn_body_h = (
                (hdr_h + 4)                              # curve shape picker
                + (hdr_h + 4)                            # scope stepper
                + (32 + 8)                               # intensity slider
                + (_grid_rows_d * (_cell_h_d + 2) + 4)  # accent grid
            )
        dyn_sec_h = hdr_h + dyn_body_h + 4

        improv_body_h = 0
        if not self._improv_collapsed and _HAS_IMPROV_ENG:
            _rdiv_i      = getattr(p, "rhythm_division", 16)
            _irows       = max(1, (_rdiv_i + 7) // 8)
            _cell_h_i    = 16
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
                (32 * 3 + 8)                             # 3 prob sliders
                + hdr_h + _sub_grace_h + 4               # grace sub-section
                + hdr_h + _sub_chirp_h + 4               # chirp sub-section
                + hdr_h + _sub_echo_h + 4                # echo sub-section
                + (_irows * (_cell_h_i + 2) + 4)         # step grid
            )
        improv_sec_h = hdr_h + improv_body_h + 4

        rhythm_body_h = 0
        if not self._rhythm_collapsed:
            _rdiv      = getattr(p, "rhythm_division", 16)
            _grid_rows = max(1, (_rdiv + 7) // 8)
            _cell_h    = 16
            rhythm_body_h = (
                (hdr_h + 2)                 # division picker row
                + (hdr_h + 4)               # pattern tabs row
                + (_grid_rows * (_cell_h + 2) + 4)   # step grid
                + (hdr_h + 8)               # phrase row
                + (hdr_h + 6)               # prog-bars + fit-mode row
                + (32 * 3 + 8)              # 3 sliders (swing/pocket/gate)
            )
        rhythm_sec_h = hdr_h + rhythm_body_h + 4

        total_h = 4 + voices_sec_h + 8 + seq_sec_h + 8 + prob_sec_h + 8 + dyn_sec_h + 8 + improv_sec_h + 8 + rhythm_sec_h + 8
        surf = pygame.Surface((pw, max(200, total_h)))
        surf.fill(_PY_BG)

        y = 4

        # ── Voices header ────────────────────────────────────────────────────
        arrow_v = "\u25bc" if not self._voices_collapsed else "\u25b6"
        pygame.draw.rect(surf, (28, 44, 60), (0, y, pw, hdr_h))
        pygame.draw.line(surf, (60, 100, 140), (0, y), (pw, y))
        surf.blit(font.render(f"{arrow_v} Voices  [{len(items) - 1}]",
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
                bg = (40, 60, 80) if is_active else (24, 24, 30)
                if key == "__patch__":
                    bg = (50, 80, 120) if is_active else (28, 40, 60)
                elif is_mix:
                    bg = (44, 44, 22) if is_active else (30, 28, 18)
                pygame.draw.rect(surf, bg, (2, y, pw - 4, row_h - 2), border_radius=3)
                c = tuple(col[:3]) if col else (120, 180, 255)
                pygame.draw.rect(surf, c, (2, y, 4, row_h - 2), border_radius=2)
                txt_col = (200, 200, 210) if not muted else (80, 80, 90)
                lbl = font.render(label, True, txt_col)
                surf.blit(lbl, (10, y + (row_h - 2 - fh) // 2))
                if not is_mix and key != "__patch__":
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
            # Inline dropdown overlays (drawn immediately after buttons so they
            # cover rows below without requiring a separate pass)
            if self._open_dropdown in ("module", "control"):
                drop_x = 4 if self._open_dropdown == "module" else half + 4
                drop_w = half - 8
                items_data = (
                    [("LFO",           "lfo"),
                     ("Passthrough",   "passthrough"),
                     ("Pitch Quant.",  "pitch_quantizer")]
                    if self._open_dropdown == "module"
                    else [("Control Surface", "control_surface")]
                )
                self._dropdown_items = []
                dy = y
                for dlabel, ddata in items_data:
                    dr = pygame.Rect(drop_x, dy, drop_w, btn_h)
                    pygame.draw.rect(surf, (50, 50, 66), dr, border_radius=2)
                    pygame.draw.rect(surf, (90, 90, 120), dr, 1, border_radius=2)
                    surf.blit(font.render(dlabel, True, (200, 220, 255)), (drop_x + 4, dy + 2))
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

            # BPM slider
            y = self._add_seq_slider(seq_sliders, y, "BPM",
                                     "seq_bpm", p.seq_bpm, 30.0, 240.0, ".1f")
            # Legato slider
            y = self._add_seq_slider(seq_sliders, y, "Legato",
                                     "seq_legato", p.seq_legato, 0.05, 1.0, ".2f")
            # Portamento slider
            y = self._add_seq_slider(seq_sliders, y, "Portamento (s)",
                                     "seq_portamento_s", p.seq_portamento_s, 0.0, 2.0, ".3f")

            # Draw seq sliders
            for sl in seq_sliders:
                r = sl["rect"]
                lbl_y = r.y - 14
                surf.blit(font.render(sl["label"], True, _PY_DIM), (8, lbl_y))
                vs = font.render(format(sl["val"], sl["fmt"]), True, _PY_TXT)
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
        dp          = getattr(p, "dynamics_program", None)
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
            div_d   = p.rhythm_division
            cols_d  = 8
            rows_d  = max(1, (div_d + cols_d - 1) // cols_d)
            cw_d    = max(10, (pw - 12) // cols_d)
            ch_d    = 16
            dp.accent.ensure_size(div_d)
            accent_rects: list = []
            _ACCENT_COLS = [
                (28, 22, 44),    # 0.0  — muted
                (52, 42, 80),    # 0.5  — soft
                (88, 62, 145),   # 1.0  — normal
                (138, 88, 210),  # 1.5  — accent
                (195, 150, 255), # 2.0  — strong
            ]
            for ri in range(rows_d):
                for ci in range(cols_d):
                    step_i = ri * cols_d + ci
                    if step_i >= div_d:
                        break
                    lv  = dp.accent.level_at(step_i)
                    # map level to color index
                    _ci = min(range(5), key=lambda j: abs((j * 0.5) - lv))
                    bg  = _ACCENT_COLS[_ci]
                    brd = tuple(min(255, c + 40) for c in bg)
                    sx  = 6 + ci * cw_d
                    sy  = y + ri * (ch_d + 2)
                    cr  = pygame.Rect(sx, sy, cw_d - 2, ch_d)
                    pygame.draw.rect(surf, bg, cr, border_radius=2)
                    pygame.draw.rect(surf, brd, cr, 1, border_radius=2)
                    # show value if != 1.0
                    if abs(lv - 1.0) > 0.01:
                        lbl = "×" + (f"{lv:.1f}".rstrip("0").rstrip(".") if lv != 0.0 else "0")
                        surf.blit(font.render(lbl, True, (210, 190, 255)), (sx + 2, sy + 2))
                    accent_rects.append({"rect": cr, "step_i": step_i})
            self._dyn_accent_rects = accent_rects
            y += rows_d * (ch_d + 2) + 4

        self._dyn_sliders = dyn_sliders

        y += 8

        # ── Improv header ────────────────────────────────────────────────────
        ip          = getattr(p, "improv_program", None)
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
            # Inherits rhythm_division and rhythm_active_pat from the patch.
            div_i   = p.rhythm_division
            act_i   = min(p.rhythm_active_pat, max(0, len(p.rhythm_patterns) - 1))
            cols_i  = 8
            rows_i  = max(1, (div_i + cols_i - 1) // cols_i)
            cw_i    = max(10, (pw - 12) // cols_i)
            ch_i    = 16
            ip.ensure_pattern_size(act_i, div_i)
            improv_step_rects: list = []
            for ri in range(rows_i):
                for ci in range(cols_i):
                    si = ri * cols_i + ci
                    if si >= div_i:
                        break
                    is_on = ip.eligible(act_i, si)
                    bg  = (105, 78, 28) if is_on else (32, 26, 12)
                    brd = (190, 150, 60) if is_on else (70, 55, 22)
                    sx = 6 + ci * cw_i
                    sy = y + ri * (ch_i + 2)
                    cr = pygame.Rect(sx, sy, cw_i - 2, ch_i)
                    pygame.draw.rect(surf, bg, cr, border_radius=2)
                    pygame.draw.rect(surf, brd, cr, 1, border_radius=2)
                    if is_on:
                        surf.blit(font.render("G", True, (240, 200, 100)), (sx + 2, sy + 2))
                    improv_step_rects.append({"rect": cr, "step_i": si, "pat_i": act_i})
            self._improv_step_rects = improv_step_rects
            y += rows_i * (ch_i + 2) + 4

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
            # ── Division picker ──────────────────────────────────────────────
            surf.blit(font.render("Div:", True, _PY_DIM), (8, y + 3))
            div_rects: list = []
            dx = 40
            for dv in _RHYTHM_DIVISIONS:
                is_sel = (p.rhythm_division == dv)
                dc = (85, 52, 136) if is_sel else (42, 36, 58)
                dr = pygame.Rect(dx, y + 1, 30, hdr_h - 2)
                pygame.draw.rect(surf, dc, dr, border_radius=3)
                tc = (215, 185, 255) if is_sel else (110, 92, 148)
                surf.blit(font.render(str(dv), True, tc), (dx + 4, y + 3))
                div_rects.append({"rect": dr, "val": dv})
                dx += 32
            self._rhythm_div_rects = div_rects
            y += hdr_h + 2

            # ── Pattern tabs ─────────────────────────────────────────────────
            surf.blit(font.render("Pats:", True, _PY_DIM), (8, y + 3))
            pat_tabs: list = []
            px = 46
            for pi, _ in enumerate(p.rhythm_patterns):
                is_sel = (pi == p.rhythm_active_pat)
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
            div    = p.rhythm_division
            cols   = 8
            rows   = max(1, (div + cols - 1) // cols)
            cell_w = max(10, (pw - 12) // cols)
            cell_h = 16
            step_rects: list = []
            if p.rhythm_patterns:
                act_pat = p.rhythm_patterns[
                    min(p.rhythm_active_pat, len(p.rhythm_patterns) - 1)]
                act_pat.ensure_size(div)
                for ri in range(rows):
                    for ci in range(cols):
                        step_i = ri * cols + ci
                        if step_i >= div:
                            break
                        on       = act_pat.steps[step_i]
                        sx       = 6 + ci * cell_w
                        sy       = y + ri * (cell_h + 2)
                        is_beat  = (step_i % max(1, div // 4) == 0)
                        if on:
                            bg  = (125, 72, 195) if is_beat else (88, 52, 152)
                            brd = (175, 125, 255)
                        else:
                            bg  = (28, 22, 44) if is_beat else (22, 18, 34)
                            brd = (52, 42, 78)
                        cr = pygame.Rect(sx, sy, cell_w - 2, cell_h)
                        pygame.draw.rect(surf, bg, cr, border_radius=2)
                        pygame.draw.rect(surf, brd, cr, 1, border_radius=2)
                        step_rects.append({"rect": cr, "step_i": step_i})
            self._rhythm_step_rects = step_rects
            y += rows * (cell_h + 2) + 4

            # ── Phrase builder ────────────────────────────────────────────────
            pygame.draw.rect(surf, (20, 16, 30), (2, y, pw - 4, hdr_h + 4),
                             border_radius=3)
            surf.blit(font.render("Phrase:", True, _PY_DIM), (8, y + 3))
            phrase_rects: list = []
            phx = 58
            for si, pat_i in enumerate(p.rhythm_phrase):
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
            pb_val = str(p.rhythm_prog_bars)
            surf.blit(font.render(pb_val, True, _PY_TXT),
                      (pb_dec_r.right + (pb_inc_r.x - pb_dec_r.right - font.size(pb_val)[0]) // 2,
                       y + 3))
            self._rhythm_prog_dec_rect = pb_dec_r
            self._rhythm_prog_inc_rect = pb_inc_r
            # Right cluster: [ Drop ] [ Ext ]
            fm_drop_r = pygame.Rect(pw - 92, y + 1, 40, hdr_h - 2)
            fm_ext_r  = pygame.Rect(pw - 48, y + 1, 44, hdr_h - 2)
            _is_drop  = (p.rhythm_fit_mode == "drop")
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
            y = self._add_seq_slider(rhythm_sliders, y, "Swing",
                                     "rhythm_swing",  p.rhythm_swing,  0.0,  0.67, ".2f")
            y = self._add_seq_slider(rhythm_sliders, y, "Pocket",
                                     "rhythm_pocket", p.rhythm_pocket, -0.5, 0.5,  ".2f")
            y = self._add_seq_slider(rhythm_sliders, y, "Gate",
                                     "rhythm_gate",   p.rhythm_gate,   0.05, 2.0,  ".2f")
            _RH_ACT = (98, 68, 158)
            for sl in rhythm_sliders:
                r = sl["rect"]
                lbl_y = r.y - 14
                surf.blit(font.render(sl["label"], True, _PY_DIM), (8, lbl_y))
                vs = font.render(format(sl["val"], sl["fmt"]), True, _PY_TXT)
                surf.blit(vs, (pw - vs.get_width() - 8, lbl_y))
                pygame.draw.rect(surf, (38, 32, 52), r, border_radius=3)
                frac = max(0.0, min(1.0,
                           (sl["val"] - sl["lo"]) / max(sl["hi"] - sl["lo"], 1e-9)))
                thumb_x = r.x + int(frac * r.w)
                pygame.draw.rect(surf, _RH_ACT,
                                 pygame.Rect(r.x, r.y, thumb_x - r.x + 4, r.h),
                                 border_radius=3)
            self._rhythm_sliders = rhythm_sliders
            y += 4

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
                dp = getattr(p, "dynamics_program", None)
                if dp is not None:
                    dp.enabled = not dp.enabled
                return True

            if not self._dyn_collapsed and _HAS_DYN_ENG:
                dp = getattr(p, "dynamics_program", None)
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
                # Intensity slider
                for i, sl in enumerate(self._dyn_sliders):
                    if sl["rect"].collidepoint(lx, ly):
                        self._dragging_dyn_slider = i
                        self._apply_dyn_slider(sl, lx)
                        return True
                # Accent grid
                for ar in self._dyn_accent_rects:
                    if ar["rect"].collidepoint(lx, ly):
                        if dp is not None:
                            dp.accent.cycle_level(ar["step_i"])
                        return True

            # ── Improv header toggle / enable ─────────────────────────────────
            if self._improv_hdr_rect and self._improv_hdr_rect.collidepoint(lx, ly):
                self._improv_collapsed = not self._improv_collapsed
                return True
            if self._improv_enable_rect and self._improv_enable_rect.collidepoint(lx, ly):
                ip2 = getattr(p, "improv_program", None)
                if ip2 is not None:
                    ip2.enabled = not ip2.enabled
                return True

            if not self._improv_collapsed and _HAS_IMPROV_ENG:
                ip2 = getattr(p, "improv_program", None)
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
                for cell in self._improv_step_rects:
                    if cell["rect"].collidepoint(lx, ly) and ip2 is not None:
                        ip2.toggle_step(cell["pat_i"], cell["step_i"], p.rhythm_division)
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
                        if key == "__patch__":
                            self._active = "__patch__"
                            if self.on_select:
                                self.on_select("__patch__")
                            return True
                        is_mix     = any(m.key == key for m in self._patch.mixers)  if self._patch else False
                        is_param   = any(pn.key == key for pn in self._patch.param_nodes) if self._patch else False
                        is_lfo     = any(l.key == key for l in self._patch.lfos) if self._patch else False
                        is_module  = any(m.key == key for m in self._patch.modules) if self._patch else False
                        is_control = any(cs.key == key for cs in self._patch.controls) if self._patch else False
                        if not is_mix and lx >= pw - 24:
                            # Delete
                            if self.on_remove_voice:
                                self.on_remove_voice(key)
                        elif not is_mix and not is_param and not is_control and not is_lfo and lx >= pw - 60:
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
                                if self.on_add_module:
                                    self.on_add_module(di["data"])
                            else:
                                if self.on_add_control:
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
                    # Division picker
                    for dr in self._rhythm_div_rects:
                        if dr["rect"].collidepoint(lx, ly):
                            p.rhythm_division = dr["val"]
                            return True
                    # Pattern tabs
                    for pt in self._rhythm_pat_tabs:
                        if pt["rect"].collidepoint(lx, ly):
                            p.rhythm_active_pat = pt["pat_i"]
                            return True
                    # Add / remove pattern
                    if self._rhythm_pat_add_rect and self._rhythm_pat_add_rect.collidepoint(lx, ly):
                        if len(p.rhythm_patterns) < 8:
                            n = len(p.rhythm_patterns)
                            p.rhythm_patterns.append(RhythmPattern(name=f"Pat {n + 1}"))
                            p.rhythm_active_pat = n
                        return True
                    if self._rhythm_pat_del_rect and self._rhythm_pat_del_rect.collidepoint(lx, ly):
                        if len(p.rhythm_patterns) > 1:
                            p.rhythm_patterns.pop()
                            p.rhythm_active_pat = min(p.rhythm_active_pat,
                                                      len(p.rhythm_patterns) - 1)
                        return True
                    # Step grid toggle
                    for sr in self._rhythm_step_rects:
                        if sr["rect"].collidepoint(lx, ly):
                            act_i = min(p.rhythm_active_pat, len(p.rhythm_patterns) - 1)
                            pat   = p.rhythm_patterns[act_i]
                            si    = sr["step_i"]
                            pat.ensure_size(p.rhythm_division)
                            pat.steps[si] = not pat.steps[si]
                            return True
                    # Phrase: click a slot to cycle to next pattern index
                    for ph in self._rhythm_phrase_rects:
                        if ph["rect"].collidepoint(lx, ly):
                            si = ph["slot_i"]
                            if 0 <= si < len(p.rhythm_phrase):
                                p.rhythm_phrase[si] = (
                                    (p.rhythm_phrase[si] + 1) % max(1, len(p.rhythm_patterns)))
                            return True
                    # Phrase add / remove bar
                    if self._rhythm_phrase_add_rect and self._rhythm_phrase_add_rect.collidepoint(lx, ly):
                        p.rhythm_phrase.append(p.rhythm_active_pat)
                        return True
                    if self._rhythm_phrase_del_rect and self._rhythm_phrase_del_rect.collidepoint(lx, ly):
                        if len(p.rhythm_phrase) > 1:
                            p.rhythm_phrase.pop()
                        return True
                    # Prog bars stepper
                    if self._rhythm_prog_dec_rect and self._rhythm_prog_dec_rect.collidepoint(lx, ly):
                        p.rhythm_prog_bars = max(1, p.rhythm_prog_bars - 1)
                        return True
                    if self._rhythm_prog_inc_rect and self._rhythm_prog_inc_rect.collidepoint(lx, ly):
                        p.rhythm_prog_bars = min(32, p.rhythm_prog_bars + 1)
                        return True
                    # Fit mode toggle
                    if self._rhythm_fit_drop_rect and self._rhythm_fit_drop_rect.collidepoint(lx, ly):
                        p.rhythm_fit_mode = "drop"
                        return True
                    if self._rhythm_fit_ext_rect and self._rhythm_fit_ext_rect.collidepoint(lx, ly):
                        p.rhythm_fit_mode = "extend"
                        return True
                    # Rhythm sliders
                    for i, sl in enumerate(self._rhythm_sliders):
                        if sl["rect"].collidepoint(lx, ly):
                            self._dragging_rhythm_slider = i
                            self._apply_rhythm_slider(sl, lx)
                            return True

            return False

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
                    raw_str: str | None = None) -> int:
        sliders.append(dict(
            label=label, key=key, val=val, lo=lo, hi=hi,
            fmt=fmt, is_log=is_log,
            rect=pygame.Rect(8, y, self.PANEL_W - 16, self._SLIDER_H),
            target=target,   # explicit write target; None → resolved in _set_slider_val
            dtype=dtype, choices=list(choices), raw_str=raw_str,
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
        param_node = next((pn for pn in self._patch.param_nodes if pn.key == self._active_key), None)
        module     = next((m  for m  in self._patch.modules     if m.key  == self._active_key), None)
        control    = next((cs for cs in self._patch.controls    if cs.key == self._active_key), None)
        obj = voice or lfo or mixer or param_node or module or control

        # Target: voice/lfo/mixer/module/control if one is active, otherwise patch global knobs
        if obj is not None:
            # Ensure granular sub-spec exists before building knob list so that
            # visible_when-gated granular knobs read real values, not None.
            if voice is not None and getattr(voice, "emission_mode", "single") == "granular":
                _ensure_granular(voice)
            target     = obj
            knob_list: list[KnobSpec] = type(obj).knobs() if hasattr(type(obj), "knobs") else []
            header_lbl = getattr(obj, "label", "—")
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
            y += fh + 8
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
                y += fh + 8
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
                    y += fh + 8
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

        # Multi-channel LFO channel list — rendered after the standard knobs.
        if module is not None and module.module_type == "lfo":
            sliders.append(dict(
                label="— LFO Channels —", key="__section__",
                val=0, lo=0, hi=1, fmt="", is_log=False,
                rect=pygame.Rect(0, y, pw, fh + 4), target=None,
            ))
            y += fh + 8
            _lfo_shapes = AnalyticModule._LFO_SHAPES
            for _ci, _lch in enumerate(module.lfo_channels):
                sliders.append(dict(
                    label=f"— ch{_ci} —", key=f"__lfo_ch_hdr_{_ci}__",
                    val=0, lo=0, hi=1, fmt="", is_log=False,
                    rect=pygame.Rect(0, y, pw, fh + 4), target=None,
                ))
                y += fh + 8
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

        # When a param_node is active, append the dynamic Targets section below
        # the static knobs (label / extractor / default / low / high).
        if param_node is not None:
            pn = param_node
            # Build the live node catalogue: voices + lfos + modules
            # Interaural modules expose ch1/ch2 as separately addressable param targets.
            _iau_key_to_mod: dict = {}
            all_nodes = (
                [(v.key, f"{v.label}") for v in self._patch.voices] +
                [(l.key, f"~ {l.label}") for l in self._patch.lfos]
            )
            for _m in self._patch.modules:
                all_nodes.append((_m.key, f"\u2B21 {_m.label}"))
                if _m.module_type == "interaural":
                    all_nodes.append((_m.ch1_key(), f"\u2B21 {_m.label} ch1"))
                    all_nodes.append((_m.ch2_key(), f"\u2B21 {_m.label} ch2"))
                    _iau_key_to_mod[_m.ch1_key()] = _m
                    _iau_key_to_mod[_m.ch2_key()] = _m
                    _iau_key_to_mod[_m.key]       = _m
            voice_keys   = [""] + [k for k, _ in all_nodes]
            voice_labels = ["—"] + [lbl for _, lbl in all_nodes]
            attr_choices = ["—"] + _VOICE_PARAM_ATTRS  # default; overridden per target below

            sliders.append(dict(
                label="— Targets —", key="__section__",
                val=0, lo=0, hi=1, fmt="", is_log=False,
                rect=pygame.Rect(0, y, pw, fh + 4), target=None,
            ))
            y += fh + 8

            for ti in range(len(pn.targets)):
                tgt = pn.targets[ti]

                # Voice dropdown
                cur_vk  = tgt.get("voice_key", "")
                cur_vi  = voice_keys.index(cur_vk) if cur_vk in voice_keys else 0
                _ti_v = ti  # capture for closure

                def _make_voice_cb(_pn=pn, _ti=_ti_v, _vkeys=voice_keys):
                    def _cb(idx):
                        _pn.targets[_ti]["voice_key"] = _vkeys[idx] if idx < len(_vkeys) else ""
                    return _cb

                sliders.append(dict(
                    label=f"T{ti} voice", key=f"__tgt_voice_{ti}__",
                    val=float(cur_vi), lo=0.0, hi=float(max(0, len(voice_labels) - 1)),
                    fmt=".0f", is_log=False,
                    rect=pygame.Rect(8, y, pw - 16, self._SLIDER_H),
                    target=None,
                    dtype="choice", choices=voice_labels, raw_str=None,
                    on_change=_make_voice_cb(),
                ))
                y += self._SLIDER_H + self._SLIDER_PAD + 12

                # Attr dropdown — choices depend on whether target is a module
                if cur_vk in _iau_key_to_mod:
                    _tgt_attrs = ["—"] + _module_param_attrs(_iau_key_to_mod[cur_vk].module_type)
                else:
                    _tgt_attrs = ["—"] + _VOICE_PARAM_ATTRS
                cur_at  = tgt.get("attr", "")
                cur_ai  = _tgt_attrs.index(cur_at) if cur_at in _tgt_attrs else 0
                _ti_a = ti  # capture for closure

                def _make_attr_cb(_pn=pn, _ti=_ti_a, _attrs=_tgt_attrs):
                    def _cb(idx):
                        _pn.targets[_ti]["attr"] = _attrs[idx] if idx < len(_attrs) else ""
                    return _cb

                sliders.append(dict(
                    label=f"T{ti} attr", key=f"__tgt_attr_{ti}__",
                    val=float(cur_ai), lo=0.0, hi=float(max(0, len(_tgt_attrs) - 1)),
                    fmt=".0f", is_log=False,
                    rect=pygame.Rect(8, y, pw - 16, self._SLIDER_H),
                    target=None,
                    dtype="choice", choices=_tgt_attrs, raw_str=None,
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
        if mixer is not None:
            routing = self._patch.routing
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
                    y += fh + 8
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

        self._sliders = sliders

        surf_h = max(200, y + 20)
        surf = pygame.Surface((pw, surf_h))
        surf.fill(_PY_BG)

        # Header
        pygame.draw.rect(surf, (30, 30, 40), (0, 0, pw, fh + 6))
        surf.blit(font.render(f"  {header_lbl}", True, _PY_TXT), (8, 3))

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
            obj = voice or lfo or mod
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
            obj = voice or lfo or pn or self._patch

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
        self.patch_panel.on_toggle_mute  = lambda _: self._mark_dirty()
        self.patch_panel.on_remove_voice = self._on_remove_voice
        self.patch_panel.on_deploy_chord = self._on_deploy_chord
        self.patch_panel.on_demo_play    = self._play_demo_sequence
        self.patch_panel.on_render_fund  = self._on_render_to_files
        self.patch_panel.on_render       = self._on_render_sequence

    # ---- Callbacks ---------------------------------------------------------

    def _on_select(self, key: str) -> None:
        self.active_key = key
        self.canvas.active_key = key
        # Auto-select an appropriate mode for the new node type
        is_mixer   = any(m.key == key for m in self.patch.mixers)
        is_param   = any(pn.key == key for pn in self.patch.param_nodes)
        is_module  = any(m.key == key for m in self.patch.modules)
        is_control = any(cs.key == key for cs in self.patch.controls)
        if is_mixer:
            self.canvas.mode = EditorMode.ROUTING
        elif is_param:
            self.canvas.mode = EditorMode.PARAM_ROUTING
        elif is_module:
            self.canvas.mode = EditorMode.ROUTING
        elif is_control:
            # Controls don't have routing-graph edges on the surface itself;
            # stay in WAVEFORM to show the parameter panel cleanly.
            if self.canvas.mode in (EditorMode.ROUTING, EditorMode.PARAM_ROUTING):
                self.canvas.mode = EditorMode.WAVEFORM
        else:
            if self.canvas.mode in (EditorMode.ROUTING, EditorMode.PARAM_ROUTING):
                self.canvas.mode = EditorMode.WAVEFORM
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

    def _on_remove_voice(self, key: str) -> None:
        self.patch.voices      = [v  for v  in self.patch.voices      if v.key  != key]
        self.patch.lfos        = [l  for l  in self.patch.lfos        if l.key  != key]
        self.patch.modules     = [m  for m  in self.patch.modules     if m.key  != key]
        self.patch.controls    = [cs for cs in self.patch.controls    if cs.key != key]
        self.patch.param_nodes = [pn for pn in self.patch.param_nodes if pn.key != key]
        # L4 fix: prune routing edges that reference the deleted node so they
        # don't show as ghost edges in the routing grid.
        remaining_keys = _patch_node_keys(self.patch)
        self.patch.routing.prune_keys(remaining_keys)
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

        # ── Schedule ──────────────────────────────────────────────────────────
        if p.rhythm_enabled:
            # Rhythm programmer path — step grid drives onset/duration
            try:
                schedule = _build_rhythm_schedule(
                    p, beat_s, degrees, pattern,
                    min_dur_s=1.0 / p.preview_sr)
            except Exception as exc:
                print(f"Rhythm schedule error: {exc}")
                return
        elif p.seq_custom_semitones.strip():
            # Custom semitone scale, legacy arpeggio timing
            try:
                from sequence_engine import NoteSchedule, NoteEvent
                schedule = NoteSchedule()
                t = 0.0
                for rep in range(p.seq_repeats):
                    for deg_i in pattern:
                        hz       = degrees[deg_i % len(degrees)]
                        note_dur = beat_s * 0.5 * p.seq_legato
                        schedule.add(NoteEvent(hz, t, max(note_dur, 1.0 / p.preview_sr)))
                        t += beat_s * 0.5
            except Exception as exc:
                print(f"Custom scale demo error: {exc}")
                return
        else:
            # Arpeggio engine path (legacy)
            try:
                rule = ArpeggioRule(
                    root_hz         = template.freq_hz,
                    scale           = scale,
                    pattern         = pattern,
                    rhythm_beats    = [0.5],
                    bpm             = p.seq_bpm,
                    legato_fraction = p.seq_legato,
                    octave_span     = p.seq_octave_span,
                    repeats         = p.seq_repeats,
                )
                schedule = rule.generate()
            except Exception as exc:
                print(f"Demo schedule error: {exc}")
                return

        import copy as _copy
        sr      = p.preview_sr
        # All non-muted voices are included in every note — ensures granular layers,
        # secondary oscillators, etc. are synthesised alongside the lead voice.
        source_voices = [v for v in p.voices if not v.muted]
        if not source_voices:
            return
        template = source_voices[0]  # used for root pitch / portamento reference only
        # Compute the ringdown budget from the patch's routing feedback config so
        # the accumulation buffer is large enough to hold every note's tail.
        _fb_cfg       = p.routing.feedback
        _gdecay_est   = max(0.0, 1.0 - float(_fb_cfg.decay)) if _fb_cfg.enabled else 1.0
        _ringdown_n   = estimate_ringdown_samples(p.routing.edges, sr, _gdecay_est, _fb_cfg)
        total_n = int((schedule.total_duration + 0.5) * sr) + _ringdown_n
        mix_L   = np.zeros(total_n, dtype=np.float64)
        mix_R   = np.zeros(total_n, dtype=np.float64)
        prev_hz: float | None = None

        for event in schedule.events:
            # Clone and pitch every non-muted voice so the full patch texture plays.
            note_voices = []
            for src_v in source_voices:
                v = _copy.deepcopy(src_v)
                v.amplitude = src_v.amplitude * event.velocity
                # Resolve oscillator pitch from note_tracking + semitone_offset
                resolved_hz = _resolve_voice_hz(src_v, p.tuning, event.fundamental_hz)
                # Apply arrangement role octave offset
                _seq_role = getattr(src_v, "seq_role", "melody")
                if _seq_role == "bass":
                    resolved_hz = _resolve_voice_hz(src_v, p.tuning, event.fundamental_hz)
                    resolved_hz *= (2.0 ** p.seq_bass_octave)
                elif _seq_role == "root":
                    resolved_hz = p.seq_tonic_hz * (2.0 ** p.seq_root_octave)
                elif _seq_role == "stab":
                    resolved_hz = _resolve_voice_hz(src_v, p.tuning, event.fundamental_hz)
                    resolved_hz *= (2.0 ** p.seq_stab_octave)
                # else: "melody" — resolved_hz already set above
                # Portamento applies to all voices that track pitch
                if (p.seq_portamento_s > 0 and prev_hz is not None
                        and abs(prev_hz - resolved_hz) > 0.5):
                    v.chirp = ChirpSpec(
                        chirp_type    = "exponential",
                        f_delta_start = prev_hz - resolved_hz,
                        f_delta_end   = 0.0,
                        tau           = max(p.seq_portamento_s, 0.001),
                    )
                else:
                    v.chirp = ChirpSpec()
                v.freq_hz   = resolved_hz
                v.pre_delay = 0.0
                # Sync granular center frequency to the resolved pitch
                if getattr(v, "emission_mode", "single") == "granular":
                    _ensure_granular(v)
                    if v.granular is not None:
                        import copy as _gcopy
                        v.granular = _gcopy.copy(v.granular)
                        v.granular.center_frequency_hz = float(resolved_hz)
                note_voices.append(v)
            # Build a full patch so routing, feedback, delays, LFOs and
            # projection all apply to each note — not just bare voice synthesis.
            temp_patch = AnalyticPatch()
            temp_patch.duration               = event.duration_s
            temp_patch.preview_sr             = sr
            temp_patch.voices                 = note_voices
            temp_patch.lfos                   = _copy.deepcopy(p.lfos)
            temp_patch.modules                = _copy.deepcopy(p.modules)
            temp_patch.controls               = _copy.deepcopy(p.controls)
            temp_patch.routing                = _copy.deepcopy(p.routing)
            temp_patch.mixers                 = _copy.deepcopy(p.mixers)
            temp_patch.param_nodes            = _copy.deepcopy(p.param_nodes)
            temp_patch.tuning                 = p.tuning
            temp_patch.projection_mode        = p.projection_mode
            temp_patch.projection_rotation_hz = p.projection_rotation_hz
            temp_patch.normalize_output       = False  # normalize full mix later
            # __patch_seq__ carries the current note's Hz for this event;
            # __patch_tonic__ stays at the musical key root (p.seq_tonic_hz).
            temp_patch.seq_tonic_hz           = p.seq_tonic_hz
            temp_patch._seq_note_hz           = float(event.fundamental_hz)
            try:
                note_L, note_R = _synthesize_patch(temp_patch)
            except Exception:
                n_note = max(1, int(event.duration_s * sr))
                note_L = np.zeros(n_note, dtype=np.float32)
                note_R = np.zeros(n_note, dtype=np.float32)
            start_n = int(event.start_time * sr)
            end_n   = min(start_n + len(note_L), total_n)
            if start_n < total_n:
                mix_L[start_n:end_n] += note_L[:end_n - start_n]
                mix_R[start_n:end_n] += note_R[:end_n - start_n]
            # Track portamento using the template voice's resolved pitch
            prev_hz = _resolve_voice_hz(source_voices[0], p.tuning, event.fundamental_hz)

        peak = float(np.max(np.abs(np.maximum(np.abs(mix_L), np.abs(mix_R)))))
        if peak > 1e-9:
            mix_L /= peak
            mix_R /= peak
        msr  = getattr(self, '_mixer_sr', sr)
        fL   = _resample_audio(mix_L.astype(np.float32), sr, msr)
        fR   = _resample_audio(mix_R.astype(np.float32), sr, msr)
        iL   = (fL * 32767.0).clip(-32768, 32767).astype(np.int16)
        iR   = (fR * 32767.0).clip(-32768, 32767).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([iL, iR]))
        try:
            sound = pygame.sndarray.make_sound(stereo)
            if self._preview_channel and self._preview_channel.get_busy():
                self._preview_channel.stop()
            self._preview_sound   = sound
            self._preview_channel = sound.play()
        except Exception as exc:
            print(f"Demo playback error: {exc}")

    # ---- Save / load -------------------------------------------------------

    def _save_patch(self) -> None:
        path = self.patch_path or "analytic_patch.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.patch.to_dict(), f, indent=2)
        print(f"Saved → {path}")

    # ---- File export (Render button) ---------------------------------------

    def _on_render_to_files(self) -> None:
        """Synthesize the patch and write a WAV file for each export-marked mixer."""
        import datetime
        export_mixers = [m for m in self.patch.mixers if m.export_to_file]
        if not export_mixers:
            print("Render: no mixers marked for export. "
                  "Enable export in the routing grid's Export tab.")
            return
        print(f"Render: synthesising patch '{self.patch.name}' …")
        try:
            result = _synthesize_patch(self.patch, file_render=True,
                                       _return_mixer_sigs=True)
            _l, _r, mixer_outs = result
        except Exception as exc:
            print(f"Render error during synthesis: {exc}")
            return
        stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        src_sr  = self.patch.preview_sr
        out_dir = os.path.dirname(self.patch_path or ".")
        for m in export_mixers:
            if m.key not in mixer_outs:
                print(f"  Render: mixer '{m.label}' has no output signal — skipped.")
                continue
            ml, mr = mixer_outs[m.key]
            if m.export_sample_rate != src_sr:
                ml = _resample_audio(ml, src_sr, m.export_sample_rate)
                mr = _resample_audio(mr, src_sr, m.export_sample_rate)
            safe_name  = self.patch.name.replace(" ", "_").replace("/", "_")
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
        import datetime, copy as _copy
        p = self.patch
        if not _HAS_SEQ_ENG:
            print("Render Sequence: sequence_engine not available")
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

        # ── Build schedule ────────────────────────────────────────────────────
        try:
            if p.rhythm_enabled:
                schedule = _build_rhythm_schedule(
                    p, beat_s, degrees, pattern,
                    min_dur_s=1.0 / p.preview_sr)
            elif p.seq_custom_semitones.strip():
                from sequence_engine import NoteSchedule, NoteEvent
                schedule = NoteSchedule()
                t = 0.0
                for _rep in range(p.seq_repeats):
                    for deg_i in pattern:
                        hz       = degrees[deg_i % len(degrees)]
                        note_dur = beat_s * 0.5 * p.seq_legato
                        schedule.add(NoteEvent(hz, t, max(note_dur, 1.0 / p.preview_sr)))
                        t += beat_s * 0.5
            else:
                rule = ArpeggioRule(
                    root_hz         = template.freq_hz,
                    scale           = scale,
                    pattern         = pattern,
                    rhythm_beats    = [0.5],
                    bpm             = p.seq_bpm,
                    legato_fraction = p.seq_legato,
                    octave_span     = p.seq_octave_span,
                    repeats         = p.seq_repeats,
                )
                schedule = rule.generate()
        except Exception as exc:
            print(f"Render Sequence: schedule error: {exc}")
            return

        # ── Synthesize every note with full patch routing ─────────────────────
        sr            = p.preview_sr
        source_voices = [v for v in p.voices if not v.muted]
        _fb_cfg       = p.routing.feedback
        _gdecay_est   = max(0.0, 1.0 - float(_fb_cfg.decay)) if _fb_cfg.enabled else 1.0
        _ringdown_n   = estimate_ringdown_samples(p.routing.edges, sr, _gdecay_est, _fb_cfg)
        total_n       = int((schedule.total_duration + 0.5) * sr) + _ringdown_n
        mix_L = np.zeros(total_n, dtype=np.float64)
        mix_R = np.zeros(total_n, dtype=np.float64)
        prev_hz: float | None = None
        for event in schedule.events:
            note_voices = []
            for src_v in source_voices:
                v = _copy.deepcopy(src_v)
                v.amplitude = src_v.amplitude * event.velocity
                if (p.seq_portamento_s > 0 and prev_hz is not None
                        and abs(prev_hz - event.fundamental_hz) > 0.5):
                    v.chirp = ChirpSpec(
                        chirp_type    = "exponential",
                        f_delta_start = prev_hz - event.fundamental_hz,
                        f_delta_end   = 0.0,
                        tau           = max(p.seq_portamento_s, 0.001),
                    )
                else:
                    v.chirp = ChirpSpec()
                v.freq_hz   = event.fundamental_hz
                v.pre_delay = 0.0
                if getattr(v, "emission_mode", "single") == "granular":
                    _ensure_granular(v)
                    if v.granular is not None:
                        import copy as _gcopy
                        v.granular = _gcopy.copy(v.granular)
                        v.granular.center_frequency_hz = float(event.fundamental_hz)
                note_voices.append(v)
            temp_patch = AnalyticPatch()
            temp_patch.duration               = event.duration_s
            temp_patch.preview_sr             = sr
            temp_patch.voices                 = note_voices
            temp_patch.lfos                   = _copy.deepcopy(p.lfos)
            temp_patch.routing                = _copy.deepcopy(p.routing)
            temp_patch.mixers                 = _copy.deepcopy(p.mixers)
            temp_patch.projection_mode        = p.projection_mode
            temp_patch.projection_rotation_hz = p.projection_rotation_hz
            temp_patch.normalize_output       = False
            try:
                note_L, note_R = _synthesize_patch(temp_patch)
            except Exception:
                n_note = max(1, int(event.duration_s * sr))
                note_L = np.zeros(n_note, dtype=np.float32)
                note_R = np.zeros(n_note, dtype=np.float32)
            start_n = int(event.start_time * sr)
            end_n   = min(start_n + len(note_L), total_n)
            if start_n < total_n:
                mix_L[start_n:end_n] += note_L[:end_n - start_n]
                mix_R[start_n:end_n] += note_R[:end_n - start_n]
            prev_hz = event.fundamental_hz

        peak = float(np.max(np.abs(np.maximum(np.abs(mix_L), np.abs(mix_R)))))
        if peak > 1e-9:
            mix_L /= peak
            mix_R /= peak

        # ── Write WAV ─────────────────────────────────────────────────────────
        stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir  = os.path.dirname(self.patch_path or ".") or "."
        safe_name = p.name.replace(" ", "_").replace("/", "_")
        fname    = os.path.join(out_dir, f"{safe_name}_seq_{stamp}.wav")
        try:
            import soundfile as _sf_export
            data = np.column_stack([mix_L.astype(np.float32),
                                    mix_R.astype(np.float32)])
            _sf_export.write(fname, data, samplerate=sr, subtype="PCM_24")
            print(f"Render Sequence → {fname}")
        except Exception as exc:
            print(f"Render Sequence write error: {exc}")

    # ---- Preview playback --------------------------------------------------

    def _play_preview(self) -> None:
        if self._preview_channel and self._preview_channel.get_busy():
            self._preview_channel.stop()
            return
        try:
            left, right = _synthesize_patch(self.patch,
                                             granular_seed_offset=self._seed_anim_tick)
            src_sr = self.patch.preview_sr
            msr    = getattr(self, '_mixer_sr', src_sr)
            left   = _resample_audio(left,  src_sr, msr)
            right  = _resample_audio(right, src_sr, msr)
            l16 = np.clip(left  * 32767.0, -32768, 32767).astype(np.int16)
            r16 = np.clip(right * 32767.0, -32768, 32767).astype(np.int16)
            stereo = np.ascontiguousarray(np.column_stack([l16, r16]))
            sound  = pygame.sndarray.make_sound(stereo)
            self._preview_sound   = sound
            self._preview_channel = sound.play()
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
        pygame.mixer.pre_init(
            frequency=self.patch.preview_sr, size=-16, channels=2, buffer=2048)
        pygame.mixer.init()
        self._mixer_sr = pygame.mixer.get_init()[0]  # actual driver rate (may differ from request)
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
                        modes = EditorCanvas.MODES
                        idx   = modes.index(self.canvas.mode)
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
