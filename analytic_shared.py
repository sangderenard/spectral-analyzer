#!/usr/bin/env python3
"""Shared imports and constants for analytic driver split modules."""
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
                              _build_parametric_driver_batches,
                              _CHIRP_CODE, CHIRP_NONE)
from performer_engine import (init_driver_state, multi_level_driver_step,
                               DriverConfig, DriverState, driver_synthesis_step)
from routing_solve_torch import CompiledRouter

