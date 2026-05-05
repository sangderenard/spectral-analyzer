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
import io
import math
import os
import glob
import sys
import collections
import argparse
import atexit
import traceback
import multiprocessing as mp
import threading
import queue
import time
import wave
import enum
from dataclasses import dataclass
from typing import List, Optional


class RenderMode(enum.Enum):
    """Player-experience render modes.

    C       — C/software pipeline (default startup path; driven by shader walk).
    GL      — pure OpenGL rasterisation.
    HYBRID  — OpenGL rasterisation augmented with a baked radiant /
              irradiant / volumetric light field uploaded as textures.
    RAYTRACE — full ray-trace pass replaces rasterised 3-D rendering.
    """
    C        = "c"
    GL       = "gl"
    HYBRID   = "hybrid"
    RAYTRACE = "raytrace"

import numpy as np

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _HAS_YAML = False

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

# ── player controller + duty station ─────────────────────────────────────────
try:
    from player_controller import PlayerController as _PlayerController
    _HAS_PLAYER_CTRL = True
except ImportError:
    _PlayerController = None  # type: ignore[assignment,misc]
    _HAS_PLAYER_CTRL = False

try:
    from duty_station import DutyStation as _DutyStation, camera_pure_matrices as _cam_pure_matrices
    _HAS_DUTY_STATION = True
except ImportError:
    _DutyStation = None  # type: ignore[assignment,misc]
    _cam_pure_matrices = None  # type: ignore[assignment]
    _HAS_DUTY_STATION = False

try:
    from simulator_station import SimulatorStation as _SimulatorStation
    _HAS_SIMULATOR_STATION = True
except ImportError:
    _SimulatorStation = None  # type: ignore[assignment,misc]
    _HAS_SIMULATOR_STATION = False

try:
    from room_workspace import RoomWorkspace as _RoomWorkspace
    from room_station  import RoomStation   as _RoomStation
    _HAS_ROOM_STATION = True
except ImportError:
    _RoomWorkspace = None  # type: ignore[assignment,misc]
    _RoomStation   = None  # type: ignore[assignment,misc]
    _HAS_ROOM_STATION = False


def _build_default_scene(config_dir: str = "configs/room_station"):
    """Build the canonical default entry-point scene programmatically.

    Returns a blank RoomWorkspace containing exactly two stations:
      - a room control duty station (origin, yaw 0)
      - a fabricator duty station (2 m to the right in X)

    Does NOT read or write scene.yaml.  scene.yaml is a dev reference only.
    """
    if _RoomWorkspace is None:
        return None
    from placed_object import PlacedDutyStation, _make_id  # local import: not a top-level dep
    ws = _RoomWorkspace.blank(config_dir)
    room_ctrl = PlacedDutyStation(
        obj_id=_make_id("room_station"),
        label="Room Control",
        pos=np.array([0.0, 0.0, 0.0], np.float64),
        yaw_deg=0.0,
        station_type="room_control",
        config_dir="configs/duty_stations/room_control",
        build_state={
            "unfinished": True,
            "job_order_id": "job::room_control_bootstrap",
            "required_materials": {"grey_block": 6, "screen_block": 1},
            "delivered_materials": {"grey_block": 0, "screen_block": 0},
        },
    )
    ws.add_object(room_ctrl)
    fabricator = PlacedDutyStation(
        obj_id=_make_id("fabricator"),
        label="Fabricator",
        pos=np.array([2.0, 0.0, 0.0], np.float64),
        yaw_deg=0.0,
        station_type="fabricator",
        config_dir="configs/duty_stations/fabricator",
        build_state={
            "unfinished": True,
            "job_order_id": "job::fabricator_bootstrap",
            "required_materials": {"grey_block": 5, "screen_block": 1},
            "delivered_materials": {"grey_block": 0, "screen_block": 0},
        },
    )
    ws.add_object(fabricator)
    return ws

try:
    from camera_designer_station import CameraDesignerStation as _CameraDesignerStation
    _HAS_CAMERA_DESIGNER_STATION = True
except ImportError:
    _CameraDesignerStation = None  # type: ignore[assignment,misc]
    _HAS_CAMERA_DESIGNER_STATION = False

try:
    from camera_item import CameraItem as _CameraItem, build_camera_items as _build_camera_items
    _HAS_CAMERA_ITEM = True
except ImportError:
    _CameraItem = None                  # type: ignore[assignment,misc]
    _build_camera_items = None          # type: ignore[assignment]
    _HAS_CAMERA_ITEM = False

try:
    from camera_panel import CameraHudPanel as _CameraHudPanel
    _HAS_CAMERA_PANEL = True
except ImportError:
    _CameraHudPanel = None              # type: ignore[assignment,misc]
    _HAS_CAMERA_PANEL = False

try:
    from material_db import MaterialDatabase as _MaterialDatabase
    _HAS_MAT_DB = True
except ImportError:
    _MaterialDatabase = None  # type: ignore[assignment,misc]
    _HAS_MAT_DB = False

# Process-global material registry — populated by _load_material_yaml and
# by any DutyStation / scene-object __init__ that registers its materials.
_MAT_DB: object = _MaterialDatabase.instance() if _HAS_MAT_DB else None

# ── pygame + OpenGL ───────────────────────────────────────────────────────────
try:
    import pygame
    from pygame.locals import (
        DOUBLEBUF, OPENGL, QUIT, KEYDOWN, MOUSEBUTTONDOWN,
        MOUSEMOTION, MOUSEWHEEL,
        K_SPACE, K_r, K_q, K_p, K_w, K_e,
        K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9,
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
        glDeleteTextures, glDeleteVertexArrays, glDepthFunc, glDepthMask,
        GL_LEQUAL, GL_LESS,
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
N_LAYERS    = 9
LAYER_KEYS  = [K_1, K_2, K_3, K_4, K_5, K_6, K_7]          # 8/9 (illum/sensor) live in camera panel
LAYER_NAMES = ['body', 'plate', 'pressure', 'strings', 'markers', 'ray-segs', 'stage', 'illum', 'sensor']

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


@dataclass
class FilmLayer:
    """One reactive emulsion layer in a FilmStack.

    Each layer has a single Gaussian spectral response centred at
    response_center_hz with log2-octave sigma response_width_oct.  The GPU
    accumulator allocates one 3-D texture per layer and routes ray energy
    into it based on the ray's instantaneous frequency vs. this Gaussian.

    blend_mode: how this layer combines with layers below it in the stack.
        'add'      — opacity-weighted additive (overlaid sensor, EM + acoustic)
        'multiply' — not used in per-layer accumulator; kept for compat
    """
    name:           str   = 'default'
    domain:         str   = 'acoustic'  # 'acoustic' | 'em' | 'custom'
    iso:            float = 1.4
    gamma:          float = 2.2
    negative:       bool  = False
    rotate180:      bool  = True
    response_center_hz: float = 440.0     # Gaussian centre for this layer's spectral response
    response_width_oct: float = 0.55      # log2-octave sigma
    opacity:        float = 1.0
    blend_mode:     str   = 'add'       # 'add' | 'multiply'
    channel_labels: tuple = ('B0 55-220Hz', 'B1 220-440Hz', 'B2 440-880Hz', 'B3 880Hz+')
    positive_tone:  tuple | None = None
    # HSL colorisation: when hue is not None the blit shader forces that hue
    # and scales saturation, overriding the default spectral colour mix.
    hue:            float | None = None  # degrees [0, 360); None = keep spectral colours
    saturation:     float        = 1.0   # HSL saturation scale [0, 1]
    shadow_hue:          float | None = None   # degrees; None → hue + 180° complement
    shadow_saturation:   float        = 0.6    # shadow endpoint HSL saturation
    shadow_lightness:    float        = 0.15   # shadow endpoint HSL lightness [0,1]
    highlight_lightness: float        = 0.65   # highlight endpoint HSL lightness [0,1]
    shadow_point:        float        = 0.12   # gc at which shadow colour is fully saturated; below → ramps to black
    highlight_point:     float        = 0.88   # gc at which highlight colour begins washing to white
    # Temporal decay: how quickly this layer fades back to black when the
    # source is silent.  0.0 = stable (no decay, default).  Any positive
    # value is the half-life in seconds: display is at 50% brightness after
    # that many seconds with no active source emitting into the accumulator.
    half_life:      float        = 0.0   # seconds; 0.0 = stable

    def positive_tone_rgb(self) -> tuple:
        """Display-positive colour for this layer, derived from hue or white."""
        if self.positive_tone is not None:
            return tuple(float(v) for v in self.positive_tone[:3])
        if self.hue is not None:
            return _hsl_to_rgb(float(self.hue) % 360.0,
                               min(float(self.saturation), 1.0),
                               float(self.highlight_lightness))
        return (1.0, 1.0, 1.0)

    def duotone_pair(self) -> tuple:
        """Return (shadow_rgb, highlight_rgb) duotone endpoints for this layer.

        Neither endpoint is forced to black or white.  As band intensity goes
        0→1 the shader lerps from shadow_rgb to highlight_rgb, giving a full
        hue-to-hue duotone gradient.
        """
        if self.hue is None:
            hi = self.positive_tone_rgb()
            sh = tuple(max(float(v) * 0.07, 0.0) for v in hi)
            return sh, hi
        h_hi = float(self.hue) % 360.0
        h_sh = (float(self.shadow_hue) % 360.0) if self.shadow_hue is not None \
               else (h_hi + 180.0) % 360.0
        highlight = _hsl_to_rgb(h_hi, min(float(self.saturation),        1.0), float(self.highlight_lightness))
        shadow    = _hsl_to_rgb(h_sh, min(float(self.shadow_saturation),  1.0), float(self.shadow_lightness))
        return shadow, highlight


def _hsl_to_rgb(h_deg: float, s: float, l: float) -> tuple:
    """Convert HSL (h in degrees, s and l in [0,1]) to an sRGB triple."""
    h = (float(h_deg) % 360.0) / 360.0
    s, l = float(s), float(l)
    if s < 1e-7:
        return (l, l, l)
    q = l * (1.0 + s) if l < 0.5 else l + s - l * s
    p = 2.0 * l - q
    def _c(t: float) -> float:
        if t < 0.0: t += 1.0
        if t > 1.0: t -= 1.0
        if t < 1.0/6.0: return p + (q - p) * 6.0 * t
        if t < 0.5:     return q
        if t < 2.0/3.0: return p + (q - p) * (2.0/3.0 - t) * 6.0
        return p
    return (_c(h + 1.0/3.0), _c(h), _c(h - 1.0/3.0))


# Backward-compat alias.
FilmParams = FilmLayer


class FilmStack:
    """Ordered dict of FilmLayers with an active selection.

    Active layer   → drives display parameters (iso, gamma, negative, rotate180)
                     sent to the blit shader each frame.
    layer_specs()  → per-layer (center_hz, width_oct, gain, dark_rgb, light_rgb)
                     used to upload per-layer GPU uniforms and allocate one 3-D
                     accumulation texture per layer.
    """

    def __init__(self,
                 layers: 'dict[str, FilmLayer] | None' = None,
                 active: 'str | None' = None) -> None:
        self.layers: dict[str, FilmLayer] = dict(layers or {})
        if active is not None and active in self.layers:
            self._active = active
        else:
            self._active = next(iter(self.layers), '')

    # ------------------------------------------------------------------
    # Active layer selection
    # ------------------------------------------------------------------

    @property
    def active(self) -> str:
        return self._active

    @active.setter
    def active(self, name: str) -> None:
        if name not in self.layers:
            raise KeyError(f'No film layer {name!r}')
        self._active = name

    @property
    def active_layer(self) -> 'FilmLayer':
        if self._active and self._active in self.layers:
            return self.layers[self._active]
        return FilmLayer()

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add(self, layer: 'FilmLayer', *, set_active: bool = False) -> 'FilmStack':
        self.layers[layer.name] = layer
        if set_active or not self._active:
            self._active = layer.name
        return self

    def remove(self, name: str) -> None:
        del self.layers[name]
        if self._active == name:
            self._active = next(iter(self.layers), '')

    def layer_specs(self) -> list:
        """Return per-layer spectral + duotone spec, in insertion order.

        Each dict has: center_hz, width_oct, gain, dark_rgb, light_rgb.
        Used to upload uLayerCount, uLayerCentersHz[i], etc. to GPU shaders
        and to build the per-layer 3-D accumulation textures.
        """
        specs = []
        for layer in self.layers.values():
            dark, light = layer.duotone_pair()
            specs.append({
                'center_hz':      float(layer.response_center_hz),
                'width_oct':      float(layer.response_width_oct),
                'gain':           max(0.0, float(layer.opacity)),
                'dark_rgb':       tuple(float(v) for v in dark),
                'light_rgb':      tuple(float(v) for v in light),
                'shadow_point':   float(getattr(layer, 'shadow_point',   0.12)),
                'highlight_point': float(getattr(layer, 'highlight_point', 0.88)),
            })
        return specs

    # ------------------------------------------------------------------
    # Compositor
    # ------------------------------------------------------------------

    def composite_hsl(self) -> tuple:
        """Opacity-weighted circular mean of all layer hues.

        Returns (hue_deg: float, saturation: float, has_hue: bool).
        has_hue is False if no layer defines a hue, meaning the blit shader
        should fall back to the spectral positive_tone colour mix.
        """
        import math as _m
        sin_sum = cos_sum = sat_sum = weight_sum = 0.0
        for layer in self.layers.values():
            if layer.hue is None:
                continue
            op = max(0.0, float(layer.opacity))
            w = op
            if w <= 0.0:
                continue
            h_rad = _m.radians(float(layer.hue))
            sin_sum += _m.sin(h_rad) * w
            cos_sum += _m.cos(h_rad) * w
            sat_sum += float(layer.saturation) * w
            weight_sum += w
        if weight_sum <= 0.0:
            return (0.0, 1.0, False)
        hue_deg = _m.degrees(_m.atan2(sin_sum, cos_sum)) % 360.0
        sat     = sat_sum / weight_sum
        return (hue_deg, sat, True)

    def __repr__(self) -> str:
        return (f'FilmStack(active={self._active!r}, '
                f'layers={list(self.layers.keys())})')


# ---------------------------------------------------------------------------
# Film layer presets — individual emulsions (mix and match in a FilmStack).
# ---------------------------------------------------------------------------

# Acoustic domain — B0-B3 map to 55-220 / 220-440 / 440-880 / 880+ Hz.
_ACL = ('B0 55-220Hz', 'B1 220-440Hz', 'B2 440-880Hz', 'B3 880Hz+')
# EM domain     — same 4 slots re-labelled as R / G / B / UV-NIR.

# EM response centre frequencies in Hz and log2-octave sigma widths.
# Band layout: B0=Red 620-750nm, B1=Green 495-620nm, B2=Blue 380-495nm, B3=NIR>750nm.
#   Red   centre 685 nm  → c/685e-9 ≈ 4.38e14 Hz
#   Green centre 557 nm  → c/557e-9 ≈ 5.38e14 Hz
#   Blue  centre 437 nm  → c/437e-9 ≈ 6.86e14 Hz
#   NIR   centre 850 nm  → c/850e-9 ≈ 3.53e14 Hz
_EM_CENTERS = (4.38e14, 5.38e14, 6.86e14, 3.53e14)
_EM_WIDTHS  = (0.28,    0.30,    0.36,    0.50)
_EML = ('Red 620-750nm', 'Green 495-620nm', 'Blue 380-495nm', 'UV/NIR 750nm+')

FILM_LAYER_PRESETS: dict[str, FilmLayer] = {
    # --- Acoustic ---
    'acoustic_pan':    FilmLayer('acoustic_pan',    'acoustic',
                                 response_center_hz=440.0,  response_width_oct=3.0,
                                 channel_labels=_ACL),
    'bass_heavy':      FilmLayer('bass_heavy',      'acoustic',
                                 response_center_hz=110.0,  response_width_oct=1.5,
                                 channel_labels=_ACL,
                                 hue=28.0,  saturation=1.3),
    'treble_heavy':    FilmLayer('treble_heavy',    'acoustic',
                                 response_center_hz=1760.0, response_width_oct=1.5,
                                 channel_labels=_ACL,
                                 hue=262.0, saturation=1.2),
    'midrange':        FilmLayer('midrange',        'acoustic',
                                 response_center_hz=440.0,  response_width_oct=0.9,
                                 channel_labels=_ACL,
                                 hue=112.0, saturation=1.1),
    'ortho':           FilmLayer('ortho',           'acoustic',
                                 response_center_hz=330.0,  response_width_oct=1.2,
                                 blend_mode='multiply', channel_labels=_ACL,
                                 hue=185.0, saturation=0.9),
    'negative_pan':    FilmLayer('negative_pan',    'acoustic',
                                 response_center_hz=440.0,  response_width_oct=3.0,
                                 negative=True, channel_labels=_ACL),
    # --- EM (optical) ---
    'em_red':          FilmLayer('em_red',          'em', iso=1.0,
                                 response_center_hz=4.38e14, response_width_oct=0.28,
                                 channel_labels=_EML,
                                 hue=0.0,   saturation=1.0,
                                 shadow_hue=0.0,   shadow_saturation=0.0, shadow_lightness=0.0, highlight_lightness=0.5,
                                 shadow_point=0.0, highlight_point=1.0),
    'em_green':        FilmLayer('em_green',        'em', iso=1.0,
                                 response_center_hz=5.38e14, response_width_oct=0.30,
                                 channel_labels=_EML,
                                 hue=120.0, saturation=1.0,
                                 shadow_hue=120.0, shadow_saturation=0.0, shadow_lightness=0.0, highlight_lightness=0.5,
                                 shadow_point=0.0, highlight_point=1.0),
    'em_blue':         FilmLayer('em_blue',         'em', iso=1.0,
                                 response_center_hz=6.86e14, response_width_oct=0.36,
                                 channel_labels=_EML,
                                 hue=240.0, saturation=1.0,
                                 shadow_hue=240.0, shadow_saturation=0.0, shadow_lightness=0.0, highlight_lightness=0.5,
                                 shadow_point=0.0, highlight_point=1.0),
    'em_nir':          FilmLayer('em_nir',          'em', iso=1.8,
                                 response_center_hz=3.53e14, response_width_oct=0.50,
                                 channel_labels=_EML,
                                 hue=320.0, saturation=1.2),
    'em_panchromatic': FilmLayer('em_panchromatic', 'em', iso=1.4,
                                 response_center_hz=5.38e14, response_width_oct=2.0,
                                 channel_labels=_EML),
    'em_negative':     FilmLayer('em_negative',     'em', iso=1.4,
                                 response_center_hz=5.38e14, response_width_oct=2.0,
                                 negative=True, channel_labels=_EML),
}

# Backward-compat alias (old code: FILM_PRESETS['panchromatic']).
FILM_PRESETS: dict[str, FilmLayer] = {
    'panchromatic': FILM_LAYER_PRESETS['acoustic_pan'],
    'bass_heavy':   FILM_LAYER_PRESETS['bass_heavy'],
    'treble_heavy': FILM_LAYER_PRESETS['treble_heavy'],
    'midrange':     FILM_LAYER_PRESETS['midrange'],
    'ortho':        FILM_LAYER_PRESETS['ortho'],
    'negative_pan': FILM_LAYER_PRESETS['negative_pan'],
}

# ---------------------------------------------------------------------------
# FilmStack presets — named multi-layer configurations.
# ---------------------------------------------------------------------------

def _default_stack() -> FilmStack:
    # Three EM emulsion layers — red, green, blue — with correct optical
    # response centres in Hz.  The stage light emits a continuous distribution
    # of wavelengths that integrates to white over time; these three layers
    # split that into R / G / B channels and reconstruct full colour.
    return FilmStack({
        'R': FILM_LAYER_PRESETS['em_red'],
        'G': FILM_LAYER_PRESETS['em_green'],
        'B': FILM_LAYER_PRESETS['em_blue'],
    }, active='R')


FILM_STACKS: dict[str, FilmStack] = {
    # Single-layer acoustic
    'default':           _default_stack(),
    'acoustic_bass':     FilmStack({'bass':   FilmLayer('bass',   'acoustic', response_center_hz=110.0,  response_width_oct=1.5)},  active='bass'),
    'acoustic_treble':   FilmStack({'treble': FilmLayer('treble', 'acoustic', response_center_hz=1760.0, response_width_oct=1.5)},  active='treble'),
    'acoustic_negative': FilmStack({'neg': FilmLayer('neg', 'acoustic', negative=True)}, active='neg'),
    # Layered acoustic: additive pan then separate ortho layer
    'acoustic_ortho': FilmStack({
        'pan':  FilmLayer('pan',  'acoustic', response_center_hz=440.0,  response_width_oct=3.0),
        'ortho': FilmLayer('ortho', 'acoustic', response_center_hz=330.0, response_width_oct=1.2),
    }, active='pan'),
    # Three-band acoustic split (useful for audio-reactive visualisation)
    'tri_spectrum': FilmStack({
        'bass':   FilmLayer('bass',   'acoustic', response_center_hz=110.0,  response_width_oct=1.5, opacity=0.6),
        'mid':    FilmLayer('mid',    'acoustic', response_center_hz=440.0,  response_width_oct=0.9, opacity=0.6),
        'treble': FilmLayer('treble', 'acoustic', response_center_hz=1760.0, response_width_oct=1.5, opacity=0.6),
    }, active='bass'),
    # Single-layer EM
    'em_rgb': FilmStack({
        'R': FILM_LAYER_PRESETS['em_red'],
        'G': FILM_LAYER_PRESETS['em_green'],
        'B': FILM_LAYER_PRESETS['em_blue'],
    }, active='R'),
    'em_nir': FilmStack({'nir': FILM_LAYER_PRESETS['em_nir']}, active='nir'),
    # Dual-domain: acoustic + EM NIR
    'acoustic_em_dual': FilmStack({
        'acoustic': FilmLayer('acoustic', 'acoustic',
                              response_center_hz=440.0,  response_width_oct=3.0, opacity=0.7,
                              channel_labels=_ACL),
        'em_nir':   FilmLayer('em_nir',   'em',
                              response_center_hz=3.53e14, response_width_oct=0.50, opacity=0.5,
                              channel_labels=_EML),
    }, active='acoustic'),
    # Full-spectrum: acoustic tri-band + EM NIR, all additive
    'full_spectrum': FilmStack({
        'bass':   FilmLayer('bass',   'acoustic', response_center_hz=110.0,  response_width_oct=1.5, opacity=0.55, channel_labels=_ACL),
        'mid':    FilmLayer('mid',    'acoustic', response_center_hz=440.0,  response_width_oct=0.9, opacity=0.55, channel_labels=_ACL),
        'treble': FilmLayer('treble', 'acoustic', response_center_hz=1760.0, response_width_oct=1.5, opacity=0.55, channel_labels=_ACL),
        'em_nir': FilmLayer('em_nir', 'em',       response_center_hz=3.53e14, response_width_oct=0.50, opacity=0.4, channel_labels=_EML),
    }, active='bass'),
}
USE_GPU_RAY_FIELD = True
GPU_RAY_FIELD_DIMS = (512, 640, 160)

# ─────────────────────────────────────────────────────────────────────────────
# Sensor / Lens / Light spec dataclasses + YAML loaders
# ─────────────────────────────────────────────────────────────────────────────

_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def _load_yaml_file(path: str) -> dict:
    """Load a YAML file.  Requires PyYAML; returns empty dict on failure."""
    if not _HAS_YAML:
        print(f"[config] PyYAML not available — cannot load {path}", flush=True)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return _yaml.safe_load(fh) or {}
    except FileNotFoundError:
        print(f"[config] not found: {path}", flush=True)
        return {}
    except Exception as exc:
        print(f"[config] failed to load {path}: {exc}", flush=True)
        return {}


def _config_path(*parts: str) -> str:
    return os.path.join(_CONFIGS_DIR, *parts)


def _load_material_yaml(name: str) -> np.ndarray:
    """Load a material YAML from configs/materials/<name>.yaml and return a
    16-float32 array in the BVH mat16 layout:
        [ 0] refl_in,   diff_in,  abso_in,   # mat_in  .xyz
        [ 3] refl_out,  diff_out, abso_out,  # mat_out .xyz
        [ 6] albedo_r,  albedo_g, albedo_b,  # albedo  .xyz
        [ 9] ior,                             # mat_in  .w
        [10] opacity,                         # mat_out .w
        [11] mat_flags,  (uint bits as float, 0 = no flags)
        [12] emit_profile_idx,  (float(int), -1 = none)
        [13] remit_profile_idx, (float(int), -1 = none)
        [14] _pad
        [15] reactive_shift_hz               # Stokes shift; 0 = non-reactive
    Falls back to a neutral diffuse grey (all profile/flag fields zeroed).
    """
    _FALLBACK = np.zeros(16, np.float32)
    # Neutral matte "brutalist gray" fallback so missing materials never read as glass.
    _FALLBACK[:11] = [0.40, 0.90, 0.05,  0.40, 0.90, 0.05,  0.30, 0.29, 0.27,  1.55, 1.0]
    d = _load_yaml_file(_config_path("materials", f"{name}.yaml"))
    if not d:
        if _MAT_DB is not None:
            _MAT_DB.register_mat11(name, _FALLBACK[:11])
        return _FALLBACK
    refl  = float(d.get("reflectivity", 0.5))
    diff  = float(d.get("diffusion",    0.0))
    abso  = float(d.get("absorption",   0.0))
    alb   = d.get("albedo_rgb", [0.5, 0.5, 0.5])
    ior   = float(d.get("ior",     1.5))
    opac  = float(d.get("opacity", 1.0))
    # Optional emission profile indices
    emit_idx  = int(d.get("emit_profile_idx",  -1))
    remit_idx = int(d.get("remit_profile_idx", -1))
    react = float(d.get("reactive_shift_hz", 0.0))
    # Derive mat_flags
    _flags = np.uint32(0)
    if emit_idx >= 0:
        _flags |= np.uint32(1)   # MAT_FLAG_EMISSIVE = 1u
    if react != 0.0:
        _flags |= np.uint32(2)   # MAT_FLAG_REACTIVE = 2u
    arr = np.zeros(16, np.float32)
    arr[:11] = [refl, diff, abso,  refl, diff, abso,
                float(alb[0]), float(alb[1]), float(alb[2]), ior, opac]
    arr[11]  = np.frombuffer(np.array([_flags], np.uint32).tobytes(), np.float32)[0]
    arr[12]  = float(emit_idx)
    arr[13]  = float(remit_idx)
    arr[14]  = 0.0  # _pad
    arr[15]  = react
    if _MAT_DB is not None:
        _MAT_DB.register_mat11(name, arr)   # full mat16 — DB extracts emissive/flag cols
    return arr


@dataclass
class SensorSpec:
    """Physical imaging sensor dimensions, used for accurate FOV calculation."""
    name:                str   = "full_frame_35mm"
    width_mm:            float = 36.0
    height_mm:           float = 24.0
    pixel_pitch_um:      float = 8.4
    max_iso:             int   = 12800
    dynamic_range_stops: float = 14.0

    @classmethod
    def from_dict(cls, d: dict) -> "SensorSpec":
        return cls(
            name=str(d.get("name", "sensor")),
            width_mm=float(d.get("width_mm", 36.0)),
            height_mm=float(d.get("height_mm", 24.0)),
            pixel_pitch_um=float(d.get("pixel_pitch_um", 8.4)),
            max_iso=int(d.get("max_iso", 12800)),
            dynamic_range_stops=float(d.get("dynamic_range_stops", 14.0)),
        )

    @classmethod
    def load(cls, name: str) -> "SensorSpec":
        d = _load_yaml_file(_config_path("sensors", f"{name}.yaml"))
        return cls.from_dict(d) if d else cls()


@dataclass
class LensSpec:
    """Optical lens parameters for FOV, focus range, and aberration."""
    name:                 str   = "standard_35mm"
    focal_mm:             float = 35.0
    min_focal_mm:         float = 35.0
    max_focal_mm:         float = 35.0
    is_zoom:              bool  = False
    min_focus_m:          float = 0.45
    max_aperture_fstop:   float = 1.4
    distortion_k1:        float = 0.001
    distortion_k2:        float = 0.0
    vignetting:           float = 0.15
    # Optical transmission spectrum (4 bands) — mostly flat for modern glass
    transmission:         tuple = (0.98, 0.97, 0.95, 0.85)

    @classmethod
    def from_dict(cls, d: dict) -> "LensSpec":
        tx = tuple(float(v) for v in d.get("transmission", [0.98, 0.97, 0.95, 0.85]))
        return cls(
            name=str(d.get("name", "lens")),
            focal_mm=float(d.get("focal_mm", 35.0)),
            min_focal_mm=float(d.get("min_focal_mm", d.get("focal_mm", 35.0))),
            max_focal_mm=float(d.get("max_focal_mm", d.get("focal_mm", 35.0))),
            is_zoom=bool(d.get("is_zoom", False)),
            min_focus_m=float(d.get("min_focus_m", 0.45)),
            max_aperture_fstop=float(d.get("max_aperture_fstop", 1.4)),
            distortion_k1=float(d.get("distortion_k1", 0.001)),
            distortion_k2=float(d.get("distortion_k2", 0.0)),
            vignetting=float(d.get("vignetting", 0.15)),
            transmission=tx,
        )

    @classmethod
    def load(cls, name: str) -> "LensSpec":
        d = _load_yaml_file(_config_path("lenses", f"{name}.yaml"))
        return cls.from_dict(d) if d else cls()


def _sample_planck_hz(T: float, rng) -> float:
    """Sample a photon frequency from B(ν,T) restricted to 380–750 nm visible range.

    Uses a precomputed 512-point CDF over the visible band so every call is O(log N).
    """
    _C = 2.998e8
    _H = 6.626e-34
    _K = 1.381e-23
    N   = 512
    lo  = _C / 750e-9
    hi  = _C / 380e-9
    freqs = np.linspace(lo, hi, N, dtype=np.float64)
    x     = np.clip(_H * freqs / (_K * max(float(T), 100.0)), 1e-10, 700.0)
    bnu   = freqs ** 3 / (np.exp(x) - 1.0)
    cdf   = np.cumsum(bnu)
    cdf  /= cdf[-1]
    u     = float(rng.uniform(0.0, 1.0))
    idx   = min(int(np.searchsorted(cdf, u)), N - 1)
    return float(freqs[idx])


@dataclass
class LightSpec:
    """Spectral description of a physical light source.

    ``spectrum_def`` is parsed from the YAML ``spectrum:`` block and drives
    ``sample_freq_hz()``.  Supported types:

      planck          — Planckian blackbody, parameterised by color_temperature_k
      lines           — discrete emission lines [{nm, weight}, ...]
      gaussian_peaks  — LED-style gaussian peaks [{nm, fwhm_nm, weight}, ...]
      uniform         — flat log-uniform across 380–750 nm (true white)
    """
    name:                str   = "stage_warm_white"
    description:         str   = "Warm white stage lamp"
    color_temperature_k: float = 3200.0
    beam_angle_deg:      float = 60.0
    n_emitters:          int   = 9
    spectrum_def:        dict  = None   # type: ignore[assignment]

    def __post_init__(self):
        if self.spectrum_def is None:
            self.spectrum_def = {'type': 'planck',
                                 'color_temperature_k': self.color_temperature_k}

    def sample_freq_hz(self, rng) -> float:
        """Draw one photon frequency (Hz) from this source's continuous spectrum."""
        _C  = 2.998e8
        VIS_LO = _C / 750e-9   # ~3.99e14 Hz
        VIS_HI = _C / 380e-9   # ~7.89e14 Hz
        sdef  = self.spectrum_def or {}
        stype = str(sdef.get('type', 'planck'))

        if stype == 'planck':
            T = float(sdef.get('color_temperature_k', self.color_temperature_k))
            return _sample_planck_hz(T, rng)

        elif stype == 'lines':
            lines   = sdef.get('lines', [])
            nms     = [float(l['nm']) for l in lines]
            weights = np.array([float(l.get('weight', 1.0)) for l in lines], np.float64)
            weights /= weights.sum()
            nm = float(rng.choice(nms, p=weights))
            nm += float(rng.normal(0.0, 1.5))          # ~1.5 nm thermal broadening
            nm = float(np.clip(nm, 380.0, 750.0))
            return _C / (nm * 1e-9)

        elif stype == 'gaussian_peaks':
            peaks   = sdef.get('peaks', [])
            weights = np.array([float(p.get('weight', 1.0)) for p in peaks], np.float64)
            weights /= weights.sum()
            idx     = int(rng.choice(len(peaks), p=weights))
            p       = peaks[idx]
            nm_c    = float(p['nm'])
            sigma   = float(p.get('fwhm_nm', 20.0)) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            nm      = float(np.clip(rng.normal(nm_c, sigma), 380.0, 750.0))
            return _C / (nm * 1e-9)

        else:   # 'uniform' or unknown — log-uniform white
            return math.exp(float(rng.uniform(math.log(VIS_LO), math.log(VIS_HI))))

    @classmethod
    def from_dict(cls, d: dict) -> "LightSpec":
        sdef = d.get('spectrum')
        cct  = float(d.get('color_temperature_k', 3200.0))
        if sdef is None:
            sdef = {'type': 'planck', 'color_temperature_k': cct}
        elif isinstance(sdef, dict) and 'color_temperature_k' not in sdef:
            sdef = dict(sdef)
            sdef.setdefault('color_temperature_k', cct)
        return cls(
            name=str(d.get('name', 'light')),
            description=str(d.get('description', '')),
            color_temperature_k=cct,
            beam_angle_deg=float(d.get('beam_angle_deg', 60.0)),
            n_emitters=int(d.get('n_emitters', 9)),
            spectrum_def=sdef,
        )

    @classmethod
    def load(cls, name: str) -> "LightSpec":
        d = _load_yaml_file(_config_path("lights", f"{name}.yaml"))
        return cls.from_dict(d) if d else cls()


def load_film_layer_yaml(path: str) -> FilmLayer:
    """Load a FilmLayer from a YAML file.

    The YAML may include the full parametric spec (spectral_bands with
    center_hz / q / gain per band, activation, grain, tone endpoints).
    The band_filters tuple is computed from the spectral_bands gain values.
    """
    d = _load_yaml_file(path)
    if not d:
        return FilmLayer()

    # Read per-layer spectral response (new format)
    response_center_hz = float(d.get("response_center_hz", 440.0))
    response_width_oct  = float(d.get("response_width_oct",  0.55))

    # Support legacy spectral_bands list: use first band's center_hz/q
    if "spectral_bands" in d:
        sb = d["spectral_bands"]
        if isinstance(sb, list) and len(sb) > 0:
            first = sb[0]
            if isinstance(first, dict):
                response_center_hz = float(first.get("center_hz", response_center_hz))
                q = float(first.get("q", 2.0))
                response_width_oct = min(2.0, 1.0 / max(q * 0.5, 0.1))
        elif isinstance(sb, dict):
            vals = list(sb.values())
            if vals and isinstance(vals[0], dict):
                response_center_hz = float(vals[0].get("center_hz", response_center_hz))
                q = float(vals[0].get("q", 2.0))
                response_width_oct = min(2.0, 1.0 / max(q * 0.5, 0.1))

    # Tone endpoints → positive_tone (highlight colour)
    positive_tone = None
    if "tone_highlights" in d:
        positive_tone = tuple(float(v) for v in d["tone_highlights"][:3])

    # Grain → stored as extra attributes on the FilmLayer instance
    grain_sigma       = float(d.get("grain_sigma", 0.0))
    grain_size        = float(d.get("grain_size", 1.0))
    grain_colorimetry = str(d.get("grain_colorimetry", "luminance"))
    sensitivity       = float(d.get("sensitivity", 1.0))
    activation        = str(d.get("activation", "linear"))
    activation_params = dict(d.get("activation_params") or {})

    channel_labels_raw = d.get("channel_labels")
    if isinstance(channel_labels_raw, list):
        channel_labels = tuple(str(v) for v in channel_labels_raw)
        # Pad to length 4
        while len(channel_labels) < 4:
            channel_labels += (f"B{len(channel_labels)}",)
    else:
        is_em = bool(d.get("is_em", False))
        channel_labels = _EML if is_em else _ACL

    layer = FilmLayer(
        name=str(d.get("name", os.path.splitext(os.path.basename(path))[0])),
        domain=str(d.get("domain", "acoustic")),
        iso=float(d.get("iso", 1.4)) * sensitivity,
        gamma=float(d.get("gamma", 2.2)),
        negative=bool(d.get("negative", False)),
        rotate180=bool(d.get("rotate180", True)),
        response_center_hz=response_center_hz,
        response_width_oct=response_width_oct,
        opacity=float(d.get("opacity", 1.0)),
        blend_mode=str(d.get("blend_mode", "add")),
        channel_labels=channel_labels,
        positive_tone=positive_tone,
        hue=d.get("hue"),   # None if absent
        saturation=float(d.get("saturation", 1.0)),
        half_life=float(d.get("half_life", 0.0)),
        shadow_point=float(d.get("shadow_point", 0.12)),
        highlight_point=float(d.get("highlight_point", 0.88)),
    )
    # Attach extra attrs not on the dataclass (for downstream use)
    layer.__dict__.update(
        grain_sigma=grain_sigma,
        grain_size=grain_size,
        grain_colorimetry=grain_colorimetry,
        activation=activation,
        activation_params=activation_params,
    )
    return layer


def load_film_stack_yaml(path: str) -> FilmStack:
    """Load a FilmStack from a YAML file that defines multiple layers."""
    d = _load_yaml_file(path)
    if not d:
        return _default_stack()
    layers_raw = d.get("layers", [])
    active = d.get("active")
    stack = FilmStack()
    for ld in layers_raw:
        # Each entry may be an inline dict or a reference to a film layer file
        if "file" in ld:
            layer_path = _config_path("films", ld["file"])
            layer = load_film_layer_yaml(layer_path)
            # Allow overrides inline
            if "opacity" in ld:
                layer.opacity = float(ld["opacity"])
        else:
            # Inline — write a temp YAML dict and reuse the loader logic
            import tempfile, json as _json
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False,
                                              encoding="utf-8")
            if _HAS_YAML:
                _yaml.dump(ld, tmp)
            tmp.close()
            layer = load_film_layer_yaml(tmp.name)
            os.unlink(tmp.name)
        stack.add(layer, set_active=(layer.name == active))
    if active and active in stack.layers:
        stack.active = active
    return stack

GPU_RAY_FIELD_SCALE = 4.0
GPU_RAY_FIELD_REFERENCE_RAYS = 100_000
GPU_PRESSURE_SCALE  = 4.0   # FDTD signed Pa → normalised; same scale worked pre-rename
GPU_RAY_LOG_SCALE = True
GPU_RAY_FIELD_GAMMA = 0.62
GPU_RAY_SEGMENT_CAP = 2_000_000
STAGE_LIGHT_RAYS = 200_000
STAGE_LIGHT_DIMS = (384, 384, 256)
# Maximum rays dispatched per glDispatchCompute call.  Keeping batches small
# prevents the GPU TDR watchdog (Windows: ~2 s) from killing the process when
# running high ray counts (e.g. 10 M).  The field texture accumulates correctly
# across batches because splat() uses imageAtomicAdd.
GPU_DISPATCH_BATCH = 8_192
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
                   fretless: bool = False,
                   amr_backend: str = "cpu",
                   amr_cache_grid: bool = True,
                   progress_cb=None):
    """Returns (scene, ce, info, body_h) or (None, None, fallback_info, BODY_H)."""
    if not (_HAS_PHYSICS and _HAS_BRIDGE and _HAS_SCENE):
        return None, None, None, None

    scene = _build_body_scene_fn("string_plate")
    if scene is None:
        return None, None, None, None

    from graph_solver import _T as _PROF
    with _PROF.span("demo.physics.build_acoustic_coevolver_from_scene"):
        ce, info = build_acoustic_coevolver_from_scene(
            scene, n_strings=n_strings, sample_rate=float(SAMPLE_RATE),
            dx=dx, pad_cells=pad_cells, n_pml=n_pml, n_segs=n_segs,
            force_scale=force_scale,
            scale_length_m=_effective_scale_length(fret),
            fretless=bool(fretless),
            fret_number=int(fret),
            amr_backend=amr_backend,
            amr_cache_grid=bool(amr_cache_grid),
            amr_gradient_order=2,
            progress_cb=progress_cb)
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

    with _PROF.span("demo.physics.plate_active_2d"):
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


def _ensure_worker_gl_context() -> None:
    """Create the OpenGL context required by AMR topology compute in the worker."""
    if getattr(_ensure_worker_gl_context, "_ready", False):
        return
    pygame.init()
    pygame.display.init()
    try:
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                        pygame.GL_CONTEXT_PROFILE_CORE)
    except Exception:
        pass
    flags = OPENGL | DOUBLEBUF | getattr(pygame, "HIDDEN", 0)
    pygame.display.set_mode((64, 64), flags)
    pygame.display.set_caption("spectral-analyzer physics GL context")
    if not bool(glCreateShader):
        raise RuntimeError("physics worker OpenGL context does not expose glCreateShader")
    if not bool(glDispatchCompute):
        raise RuntimeError("physics worker OpenGL context does not expose glDispatchCompute")
    _ensure_worker_gl_context._ready = True


def _progress_put(progress_q, msg: dict) -> None:
    """Best-effort progress path. Progress is disposable; control messages are not."""
    try:
        progress_q.put_nowait(msg)
    except Exception:
        return


def _worker_profile_text(title: str = "GraphSolver — worker profile") -> str:
    from graph_solver import _T as _PROF
    buf = io.StringIO()
    _PROF.report(title=title, file=buf)
    return buf.getvalue()


def _physics_worker(cmd_q, out_q, progress_q, initial_config: dict) -> None:
    """Own the coevolver in a separate process and stream renderable frames."""
    config = dict(initial_config)
    ce = scene = info = None
    body_h = BODY_H
    n_str = int(config.get("n_strings", 6))
    fi = 0
    progress_state = {
        "frac": 0.0,
        "label": "worker starting",
        "updated": time.monotonic(),
        "stop": False,
    }

    def heartbeat_loop() -> None:
        while not progress_state["stop"]:
            time.sleep(2.0)
            age = time.monotonic() - float(progress_state["updated"])
            if age < 2.0:
                continue
            _progress_put(progress_q, {
                "type": "progress",
                "frac": float(progress_state["frac"]),
                "label": f"still running: {progress_state['label']} ({age:.0f}s)",
            })

    def worker_progress(frac: float, label: str) -> None:
        progress_state["frac"] = float(frac)
        progress_state["label"] = str(label)
        progress_state["updated"] = time.monotonic()
        _progress_put(progress_q, {
            "type": "progress",
            "frac": float(frac),
            "label": str(label),
        })

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
            fretless=bool(config.get("fretless", False)),
            amr_backend=str(config.get("amr_backend", "cpu")),
            amr_cache_grid=bool(config.get("amr_cache_grid", True)),
            progress_cb=worker_progress)

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
        hb = threading.Thread(target=heartbeat_loop, daemon=True)
        hb.start()
        _ensure_worker_gl_context()
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
                    out_q.put({
                        "type": "profile",
                        "text": _worker_profile_text(),
                    })
                    return
                if ctype == "profile":
                    out_q.put({
                        "type": "profile",
                        "text": _worker_profile_text(),
                    })
                    continue
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
                    out_q.put({
                        "type": "profile",
                        "text": _worker_profile_text(),
                    })
                    return
                if cmd.get("type") == "profile":
                    out_q.put({
                        "type": "profile",
                        "text": _worker_profile_text(),
                    })
                    continue
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
    finally:
        progress_state["stop"] = True
        try:
            text = _worker_profile_text()
            sys.stderr.write(text)
            sys.stderr.flush()
            out_q.put({"type": "profile", "text": text})
        except Exception:
            pass


def _worker_error_message(msg: dict) -> str:
    base = msg.get("message", "physics worker failed")
    tb = msg.get("traceback")
    return f"{base}\n\nWorker traceback:\n{tb}" if tb else base


def _make_tqdm(*args, **kwargs):
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(*args, **kwargs)


class _PhysicsProcess:
    def __init__(self, config: dict):
        ctx = mp.get_context("spawn")
        self._thread = None
        try:
            self.cmd_q = ctx.Queue()
            self.out_q = ctx.Queue(maxsize=8)
            self.progress_q = ctx.Queue(maxsize=64)
            self.proc = ctx.Process(
                target=_physics_worker,
                args=(self.cmd_q, self.out_q, self.progress_q, dict(config)),
            )
            self.proc.start()
            self.mode = "process"
        except PermissionError as exc:
            print(f"  multiprocessing unavailable ({exc}); using worker thread fallback",
                  flush=True)
            self.cmd_q = queue.Queue()
            self.out_q = queue.Queue(maxsize=8)
            self.progress_q = queue.Queue(maxsize=64)
            self.proc = None
            self._thread = threading.Thread(
                target=_physics_worker,
                args=(self.cmd_q, self.out_q, self.progress_q, dict(config)),
                daemon=True)
            self._thread.start()
            self.mode = "thread"

    def send(self, message: dict) -> None:
        self.cmd_q.put(message)

    def poll(self) -> list:
        out = []
        while True:
            try:
                out.append(self.progress_q.get_nowait())
            except queue.Empty:
                break
            except Exception:
                break
        while True:
            try:
                out.append(self.out_q.get_nowait())
            except queue.Empty:
                break
            except Exception:
                break
        return out

    def close(self) -> None:
        try:
            self.cmd_q.put({"type": "profile"})
            self.cmd_q.put({"type": "quit"})
        except Exception:
            pass
        deadline = pygame.time.get_ticks() + 5000 if pygame.get_init() else None
        while True:
            for msg in self.poll():
                if msg.get("type") == "profile":
                    text = msg.get("text", "")
                    if text:
                        sys.stderr.write(text)
                        sys.stderr.flush()
                elif msg.get("type") == "progress":
                    print(f"[physics worker] {float(msg.get('frac', 0.0)) * 100.0:6.2f}%  "
                          f"{msg.get('label', '')}", flush=True)
            if deadline is None or pygame.time.get_ticks() >= deadline:
                break
            alive = self.proc.is_alive() if self.proc is not None else (
                self._thread.is_alive() if self._thread is not None else False
            )
            if not alive:
                break
            pygame.time.wait(50)
        if self.proc is not None:
            self.proc.join(timeout=2.0)
            if self.proc.is_alive():
                print("[physics worker] still busy during shutdown; terminating after profile request",
                      flush=True)
                self.proc.terminate()
                self.proc.join(timeout=1.0)
        elif self._thread is not None:
            self._thread.join(timeout=2.0)


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


def _line_segments_pos(lines: Optional[np.ndarray]) -> np.ndarray:
    """(N,2,3) segments -> (2N,3) float32 positions for GL_LINES."""
    if lines is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(lines)
    if arr.ndim != 3 or arr.shape[1:] != (2, 3) or len(arr) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return arr.reshape(-1, 3).astype(np.float32, copy=False)


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

    DISPLAY-ONLY TRANSFORM — all acoustic physics systems (FDTD, ray tracer,
    GPU sensor/BVH) operate exclusively in guitar-frame.  This matrix is applied
    only by the OpenGL renderer to display the guitar upright on its stand.

    Guitar frame (canonical physics frame):
        X = lateral
        Y = longitudinal (nut → +Y, body bottom near Y=0)
        Z = body depth   (back plate Z=0, soundboard Z=body_h ≈ 0.06 m)

    World frame (OpenGL display frame, Z-up):
        X = lateral
        Y = stage depth  (+Y toward audience; guitar soundboard faces +Y)
        Z = up           (neck points +Z, lifted by stand_height)

    The swap Y↔Z makes the neck point skyward and the soundboard face the audience.
    The Z-translation lifts the body bottom to `stand_height` above the stage floor.
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
    # mat_in/out: (reflectivity, diffusion, absorption); albedo: RGB surface color
    diff_floor = [0.52, 0.80, 0.10,  0.52, 0.80, 0.10,  0.30, 0.22, 0.16]   # dark hardwood
    quad([-hw, y_back, 0.], [ hw, y_back, 0.],
         [ hw, y_front, 0.], [-hw, y_front, 0.],
         [0., 0., 1.], diff_floor)

    # ── Back wall (faces audience, +Y normal) ─────────────────────────────────
    diff_wall = [0.40, 0.90, 0.05,  0.40, 0.90, 0.05,  0.30, 0.28, 0.25]   # grey concrete
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
    diff_stand = [0.55, 0.60, 0.15,  0.55, 0.60, 0.15,  0.12, 0.10, 0.08]
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
# Glass bell-jar and opaque skirt — AMR simulation boundary visualisation
# ─────────────────────────────────────────────────────────────────────────────

_GLASS_THICKNESS_M = 0.02   # 2 cm glass wall thickness


def _build_sim_belljar_world(wb_min: np.ndarray, wb_max: np.ndarray,
                             thickness: float = _GLASS_THICKNESS_M,
                             bevel_segs: int = 6) -> np.ndarray:
    """Square glass bell-jar (4 side walls + top cap, open at bottom) with
    rounded vertical corners in world space.

    Each outer vertical edge is replaced by a quarter-cylinder arc of
    *bevel_segs* segments, radius = *thickness*.  The flat walls are trimmed
    to meet the arcs at their tangent points.  The top cap is a fan-tessellated
    rounded rectangle.

    Returns an interleaved float32 array shaped (-1, 6) with columns
    [x, y, z, nx, ny, nz], suitable for a VAO with
    ``[(0, 3, 24, 0), (1, 3, 24, 12)]`` attributes (stride 24 bytes).
    """
    t  = float(thickness)
    r  = t                               # bevel radius = wall thickness
    xi, xa = float(wb_min[0]), float(wb_max[0])
    yi, ya = float(wb_min[1]), float(wb_max[1])
    zi, za = float(wb_min[2]), float(wb_max[2])
    xo0, xo1 = xi - t, xa + t
    yo0, yo1 = yi - t, ya + t
    zt = za + t

    rows: list = []

    def qf(a, b, c, d, n):
        n3 = list(n)
        for tri in ((a, b, c), (a, c, d)):
            for v in tri:
                rows.append(list(v) + n3)

    def tf(a, b, c, n):
        n3 = list(n)
        for v in (a, b, c):
            rows.append(list(v) + n3)

    # ── Corner arc centres and angle ranges ──────────────────────────────────
    #  SW: (xo0+r, yo0+r)  π → 3π/2
    #  SE: (xo1-r, yo0+r)  3π/2 → 2π
    #  NE: (xo1-r, yo1-r)  0 → π/2
    #  NW: (xo0+r, yo1-r)  π/2 → π
    corners = [
        (xo0 + r, yo0 + r, math.pi,      3 * math.pi / 2),
        (xo1 - r, yo0 + r, 3 * math.pi / 2, 2 * math.pi),
        (xo1 - r, yo1 - r, 0.0,          math.pi / 2),
        (xo0 + r, yo1 - r, math.pi / 2,  math.pi),
    ]

    def arc_pts_2d(cx, cy, a0, a1, n):
        """Return (n+1) XY points along the arc, including endpoints."""
        return [(cx + r * math.cos(a0 + (a1 - a0) * i / n),
                 cy + r * math.sin(a0 + (a1 - a0) * i / n))
                for i in range(n + 1)]

    # ── Outer side walls — flat sections between arc tangent points ───────────
    qf([xo0, yo0 + r, zi], [xo0, yo1 - r, zi], [xo0, yo1 - r, zt], [xo0, yo0 + r, zt], [-1, 0, 0])
    qf([xo1, yo1 - r, zi], [xo1, yo0 + r, zi], [xo1, yo0 + r, zt], [xo1, yo1 - r, zt], [+1, 0, 0])
    qf([xo1 - r, yo0, zi], [xo0 + r, yo0, zi], [xo0 + r, yo0, zt], [xo1 - r, yo0, zt], [0, -1, 0])
    qf([xo0 + r, yo1, zi], [xo1 - r, yo1, zi], [xo1 - r, yo1, zt], [xo0 + r, yo1, zt], [0, +1, 0])

    # ── Outer corner arc columns ──────────────────────────────────────────────
    for cx, cy, a0, a1 in corners:
        pts = arc_pts_2d(cx, cy, a0, a1, bevel_segs)
        for i in range(bevel_segs):
            px0, py0 = pts[i]
            px1, py1 = pts[i + 1]
            amid = a0 + (a1 - a0) * (i + 0.5) / bevel_segs
            nx, ny = math.cos(amid), math.sin(amid)
            qf([px0, py0, zi], [px1, py1, zi], [px1, py1, zt], [px0, py0, zt], [nx, ny, 0])

    # ── Top cap — fan-tessellated rounded rectangle ────────────────────────────
    # Build the full outer boundary polygon (CCW from above) then fan from centre.
    # Arc order gives proper winding for +Z normal when read CCW.
    boundary_xy: list = []
    for cx, cy, a0, a1 in corners:
        pts = arc_pts_2d(cx, cy, a0, a1, bevel_segs)
        boundary_xy.extend(pts[:-1])   # skip last to avoid duplicating shared vertex
    n_b = len(boundary_xy)
    cx_cap = (xo0 + xo1) * 0.5
    cy_cap = (yo0 + yo1) * 0.5
    cap_n = [0, 0, 1]
    for i in range(n_b):
        px0, py0 = boundary_xy[i]
        px1, py1 = boundary_xy[(i + 1) % n_b]
        tf([cx_cap, cy_cap, zt], [px0, py0, zt], [px1, py1, zt], cap_n)

    # ── Inner side faces (visible from inside the jar) ────────────────────────
    qf([xi, ya, zi], [xi, yi, zi], [xi, yi, za], [xi, ya, za], [+1., 0., 0.])
    qf([xa, yi, zi], [xa, ya, zi], [xa, ya, za], [xa, yi, za], [-1., 0., 0.])
    qf([xa, yi, zi], [xi, yi, zi], [xi, yi, za], [xa, yi, za], [0., +1., 0.])
    qf([xi, ya, zi], [xa, ya, zi], [xa, ya, za], [xi, ya, za], [0., -1., 0.])
    qf([xi, ya, za], [xa, ya, za], [xa, yi, za], [xi, yi, za], [0., 0., -1.])

    return np.ascontiguousarray(np.asarray(rows, np.float32).reshape(-1, 6))


def _build_sim_skirt_world(wb_min: np.ndarray, wb_max: np.ndarray,
                           thickness: float = _GLASS_THICKNESS_M,
                           bevel_segs: int = 6) -> np.ndarray:
    """Opaque skirt: four outer walls from the floor (Z = 0) up to the open
    bottom of the bell jar (Z = *wb_min*[2]), with rounded vertical corners.

    No centre floor panel is added, so the guitar base inside is never clipped.
    Returns the same interleaved (pos3|norm3) format as
    :func:`_build_sim_belljar_world`, or an empty array when the sim region
    already starts at or below the stage floor.
    """
    floor_z = 0.0
    z_top   = float(wb_min[2])
    if z_top <= floor_z + 1e-4:
        return np.zeros((0, 6), np.float32)

    t  = float(thickness)
    r  = t
    xi, xa = float(wb_min[0]), float(wb_max[0])
    yi, ya = float(wb_min[1]), float(wb_max[1])
    xo0, xo1 = xi - t, xa + t
    yo0, yo1 = yi - t, ya + t

    rows: list = []

    def qf(a, b, c, d, n):
        n3 = list(n)
        for tri in ((a, b, c), (a, c, d)):
            for v in tri:
                rows.append(list(v) + n3)

    corners = [
        (xo0 + r, yo0 + r, math.pi,      3 * math.pi / 2),
        (xo1 - r, yo0 + r, 3 * math.pi / 2, 2 * math.pi),
        (xo1 - r, yo1 - r, 0.0,          math.pi / 2),
        (xo0 + r, yo1 - r, math.pi / 2,  math.pi),
    ]

    def arc_pts_2d(cx, cy, a0, a1, n):
        return [(cx + r * math.cos(a0 + (a1 - a0) * i / n),
                 cy + r * math.sin(a0 + (a1 - a0) * i / n))
                for i in range(n + 1)]

    # Flat wall sections
    qf([xo0, yo0 + r, floor_z], [xo0, yo1 - r, floor_z], [xo0, yo1 - r, z_top], [xo0, yo0 + r, z_top], [-1, 0, 0])
    qf([xo1, yo1 - r, floor_z], [xo1, yo0 + r, floor_z], [xo1, yo0 + r, z_top], [xo1, yo1 - r, z_top], [+1, 0, 0])
    qf([xo1 - r, yo0, floor_z], [xo0 + r, yo0, floor_z], [xo0 + r, yo0, z_top], [xo1 - r, yo0, z_top], [0, -1, 0])
    qf([xo0 + r, yo1, floor_z], [xo1 - r, yo1, floor_z], [xo1 - r, yo1, z_top], [xo0 + r, yo1, z_top], [0, +1, 0])

    # Floor cap — solid bottom panel at Z=0 facing downward
    qf([xo0 + r, yo0, floor_z], [xo1 - r, yo0, floor_z], [xo1 - r, yo1, floor_z], [xo0 + r, yo1, floor_z], [0, 0, -1])

    # Corner arc columns
    for cx, cy, a0, a1 in corners:
        pts = arc_pts_2d(cx, cy, a0, a1, bevel_segs)
        for i in range(bevel_segs):
            px0, py0 = pts[i]
            px1, py1 = pts[i + 1]
            amid = a0 + (a1 - a0) * (i + 0.5) / bevel_segs
            nx, ny = math.cos(amid), math.sin(amid)
            qf([px0, py0, floor_z], [px1, py1, floor_z], [px1, py1, z_top], [px0, py0, z_top], [nx, ny, 0])

    return np.ascontiguousarray(np.asarray(rows, np.float32).reshape(-1, 6))


def _sim_box_wireframe_world(wb_min: np.ndarray, wb_max: np.ndarray) -> np.ndarray:
    """24 position-only vertices (12 GL_LINES edges) tracing the inner sim box.

    Draws the exact inner boundary of the bell jar as a crisp wire overlay.
    Returns a flat float32 array shaped (-1, 3).
    """
    x0, y0, z0 = float(wb_min[0]), float(wb_min[1]), float(wb_min[2])
    x1, y1, z1 = float(wb_max[0]), float(wb_max[1]), float(wb_max[2])
    edges = [
        [x0, y0, z0], [x1, y0, z0],   [x1, y0, z0], [x1, y1, z0],
        [x1, y1, z0], [x0, y1, z0],   [x0, y1, z0], [x0, y0, z0],
        [x0, y0, z1], [x1, y0, z1],   [x1, y0, z1], [x1, y1, z1],
        [x1, y1, z1], [x0, y1, z1],   [x0, y1, z1], [x0, y0, z1],
        [x0, y0, z0], [x0, y0, z1],   [x1, y0, z0], [x1, y0, z1],
        [x1, y1, z0], [x1, y1, z1],   [x0, y1, z0], [x0, y1, z1],
    ]
    return np.asarray(edges, np.float32).reshape(-1, 3)


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
# The layer textures use the same uLayer0-uLayer7, uBoxMin/Max, uWorldToGrid
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

// Per-layer spectral band textures (GL_R32UI 3D)
uniform usampler3D uLayer0;
uniform usampler3D uLayer1;
uniform usampler3D uLayer2;
uniform usampler3D uLayer3;
uniform usampler3D uLayer4;
uniform usampler3D uLayer5;
uniform usampler3D uLayer6;
uniform usampler3D uLayer7;
uniform usampler3D uBaseLayer0;
uniform usampler3D uBaseLayer1;
uniform usampler3D uBaseLayer2;
uniform usampler3D uBaseLayer3;
uniform usampler3D uBaseLayer4;
uniform usampler3D uBaseLayer5;
uniform usampler3D uBaseLayer6;
uniform usampler3D uBaseLayer7;
uniform int   uLayerCount;
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

// Scene field seed — same values as in _BODY_FS, uploaded once per frame.
uniform vec3  uSceneRgb;
uniform float uSceneIndirectRatio;

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

    // Sum all active layers; use layer index to pseudo-assign R/G/B channels
    vec3 dyn_rgb = vec3(0.0);
    if (uUseRayField != 0 && in_volume) {
        int n = clamp(uLayerCount, 1, 8);
        float total = 0.0;
        for (int i = 0; i < n; i++) {
            float bv = 0.0;
            if      (i == 0) bv = sampleBand(uLayer0, uvw);
            else if (i == 1) bv = sampleBand(uLayer1, uvw);
            else if (i == 2) bv = sampleBand(uLayer2, uvw);
            else if (i == 3) bv = sampleBand(uLayer3, uvw);
            else if (i == 4) bv = sampleBand(uLayer4, uvw);
            else if (i == 5) bv = sampleBand(uLayer5, uvw);
            else if (i == 6) bv = sampleBand(uLayer6, uvw);
            else if (i == 7) bv = sampleBand(uLayer7, uvw);
            total += bv;
        }
        dyn_rgb = vec3(total) * uRayExposure;
        dyn_rgb = pow(clamp(dyn_rgb, vec3(0.0), vec3(1.0)), vec3(uRayGamma));
    }

    vec4 bfield4 = uBaseWorldToGrid * vec4(vPosW, 1.0);
    vec3 bfield_pos = bfield4.xyz / bfield4.w;
    vec3 buvw = (bfield_pos - uBaseBoxMin) / max(uBaseBoxMax - uBaseBoxMin, vec3(1e-6));
    bool in_base = all(greaterThanEqual(buvw, vec3(0.0))) &&
                   all(lessThanEqual(buvw, vec3(1.0)));
    vec3 base_rgb = vec3(0.0);
    if (uUseBaseField != 0 && in_base) {
        int n = clamp(uLayerCount, 1, 8);
        float total = 0.0;
        for (int i = 0; i < n; i++) {
            float bv = 0.0;
            if      (i == 0) bv = sampleBand(uBaseLayer0, buvw);
            else if (i == 1) bv = sampleBand(uBaseLayer1, buvw);
            else if (i == 2) bv = sampleBand(uBaseLayer2, buvw);
            else if (i == 3) bv = sampleBand(uBaseLayer3, buvw);
            else if (i == 4) bv = sampleBand(uBaseLayer4, buvw);
            else if (i == 5) bv = sampleBand(uBaseLayer5, buvw);
            else if (i == 6) bv = sampleBand(uBaseLayer6, buvw);
            else if (i == 7) bv = sampleBand(uBaseLayer7, buvw);
            total += bv;
        }
        base_rgb = vec3(total) * uBaseExposure;
        base_rgb = pow(clamp(base_rgb, vec3(0.0), vec3(1.0)), vec3(uRayGamma));
    }

    vec3  base = gl_FrontFacing ? uColor.rgb : uInnerColor;
    float grain = 0.5 + 0.5 * sin(vPosV.x * 80.0 + vPosV.y * 31.0 + vPosV.z * 17.0);
    base *= mix(1.0, 0.82 + 0.28 * grain, uGrain);

    // Spectral illumination × material reflectance + ambient lift.
    // Each band independently colours the surface; the material base colour
    // acts as a per-channel reflectance filter.
    // Scene field tints the ambient colour and brightens spectral highlights
    // in proportion to the environment's indirect (bounce) fraction.
    vec3 spectral = clamp(base_rgb + dyn_rgb, vec3(0.0), vec3(1.2));
    // Tint the spectral overlay toward the scene field colour for coherence.
    spectral = mix(spectral, spectral * uSceneRgb * 1.4, 0.25 * uSceneIndirectRatio);
    // Output only the spectral irradiance delta (material reflectance × band irradiance).
    // This is additively blended onto the Phong surface drawn earlier so the two
    // lighting models combine rather than one overwriting the other.
    FragColor = vec4(base * spectral, uColor.a);
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
    scene_field:      object       = None  # SceneFieldIntegration | None


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

// Scene field seed — from SceneFieldIntegration; uploaded once per frame.
// uSceneRgb: spectral colour of the ambient environment (unit range).
// uSceneIndirectRatio: fraction of scene power that is reflected/bounced;
//   0 = all direct (hard shadows), 1 = fully diffuse (soft fill light).
uniform vec3  uSceneRgb;
uniform float uSceneIndirectRatio;

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
    // Ambient tinted by scene spectral colour; indirect ratio adds fill light
    // in shadowed regions (simulates the environment's bounced-light term).
    vec3  ambLight   = uAmbient * mix(vec3(1.0), uSceneRgb, 0.55);
    float shadowFill = uSceneIndirectRatio * 0.28 * (1.0 - diff);
    vec3  col  = base * (ambLight + (0.78 + shadowFill) * diff)
               + mix(vec3(1.0, 0.88, 0.62), uSceneRgb, 0.30) * (uSpecStrength * spec);
    float rim  = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    col += base * rim * mix(0.16, 0.22, uSceneIndirectRatio);
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
// ── RAY FIELD PATH (uFieldMode == 1): N per-layer spectral textures ──────────
uniform usampler3D uLayer0;
uniform usampler3D uLayer1;
uniform usampler3D uLayer2;
uniform usampler3D uLayer3;
uniform usampler3D uLayer4;
uniform usampler3D uLayer5;
uniform usampler3D uLayer6;
uniform usampler3D uLayer7;
uniform int        uLayerCount;
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
// ── Duotone band colours: shadow→highlight lerp per band slot ───────────────────────
uniform vec3  uLayerDark0;  uniform vec3  uLayerLight0;
uniform vec3  uLayerDark1;  uniform vec3  uLayerLight1;
uniform vec3  uLayerDark2;  uniform vec3  uLayerLight2;
uniform vec3  uLayerDark3;  uniform vec3  uLayerLight3;
uniform vec3  uLayerDark4;  uniform vec3  uLayerLight4;
uniform vec3  uLayerDark5;  uniform vec3  uLayerLight5;
uniform vec3  uLayerDark6;  uniform vec3  uLayerLight6;
uniform vec3  uLayerDark7;  uniform vec3  uLayerLight7;
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

float sample_layer_val(int i, ivec3 q) {
    if      (i == 0) return float(texelFetch(uLayer0, q, 0).r);
    else if (i == 1) return float(texelFetch(uLayer1, q, 0).r);
    else if (i == 2) return float(texelFetch(uLayer2, q, 0).r);
    else if (i == 3) return float(texelFetch(uLayer3, q, 0).r);
    else if (i == 4) return float(texelFetch(uLayer4, q, 0).r);
    else if (i == 5) return float(texelFetch(uLayer5, q, 0).r);
    else if (i == 6) return float(texelFetch(uLayer6, q, 0).r);
    else if (i == 7) return float(texelFetch(uLayer7, q, 0).r);
    return 0.0;
}

vec3 layer_dark(int i) {
    if      (i == 0) return uLayerDark0;
    else if (i == 1) return uLayerDark1;
    else if (i == 2) return uLayerDark2;
    else if (i == 3) return uLayerDark3;
    else if (i == 4) return uLayerDark4;
    else if (i == 5) return uLayerDark5;
    else if (i == 6) return uLayerDark6;
    else if (i == 7) return uLayerDark7;
    return vec3(0.0);
}

vec3 layer_light(int i) {
    if      (i == 0) return uLayerLight0;
    else if (i == 1) return uLayerLight1;
    else if (i == 2) return uLayerLight2;
    else if (i == 3) return uLayerLight3;
    else if (i == 4) return uLayerLight4;
    else if (i == 5) return uLayerLight5;
    else if (i == 6) return uLayerLight6;
    else if (i == 7) return uLayerLight7;
    return vec3(1.0);
}

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
            ivec3 dims = textureSize(uLayer0, 0);
            ivec3 q = clamp(ivec3(floor(uvc * vec3(dims))), ivec3(0), dims - ivec3(1));
            float sc = uRayFieldScale / 65535.0;
            float ls = (uLogScale != 0) ? log(1.0 + uRayFieldScale) : 1.0;
            int n = clamp(uLayerCount, 1, 8);
            float total = 0.0;
            vec3  col_sum = vec3(0.0);
            for (int i = 0; i < n; i++) {
                float bv = sample_layer_val(i, q) * sc;
                if (uLogScale != 0) bv = log(1.0 + bv) / ls;
                float t = smoothstep(0.0, 0.12, bv);
                col_sum += mix(layer_dark(i), layer_light(i), t) * bv;
                total += bv;
            }
            total += 1e-6;
            p = pow(clamp(total, 0.0, 1.0), max(uRayFieldGamma, 0.05));
            spectral_color = col_sum / total;
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

// ─── Triangle geometry + material in a single 8×vec4 (128-byte) record ──────
// normal.w  : mat_flags, stored as uint bits via floatBitsToUint / uintBitsToFloat
// albedo.w  : 1.0 if an emission profile is assigned, 0.0 otherwise
// emissive  : .x = emit_profile_idx (float->int, index into EmissionProfileBuf)
//             .y = remit_profile_idx (float->int, index into EmissionProfileBuf)
//             .z = _pad
//             .w = reactive Stokes shift (Hz)
struct Tri {
    vec4 v0;        // .xyz = vertex 0,          .w unused
    vec4 e1;        // .xyz = edge 1,             .w unused
    vec4 e2;        // .xyz = edge 2,             .w unused
    vec4 normal;    // .xyz = face normal,        .w = mat_flags (uint bits)
    vec4 mat_in;    // .xyz = refl/diff/abso,     .w = IOR
    vec4 mat_out;   // .xyz = refl/diff/abso,     .w = opacity
    vec4 albedo;    // .xyz = surface albedo,     .w = 1.0 if emission profile assigned
    vec4 emissive;  // .x  = emit_profile_idx,   .y = remit_profile_idx,  .z = _pad,  .w = reactive shift Hz
};

// ─── Material flag bits (packed into Tri.normal.w as uint) ──────────────────
// Test with: (floatBitsToUint(tri.normal.w) & MAT_FLAG_*) != 0u
#define MAT_FLAG_EMISSIVE    1u  // surface adds energy to field on every hit
#define MAT_FLAG_REACTIVE    2u  // post-impact re-emitter: enqueues PendingRay
#define MAT_FLAG_ABSORBER    4u  // terminates ray without bounce
#define MAT_FLAG_NO_SHADOW   8u  // shadow/visibility rays pass through
#define MAT_FLAG_MANIFOLD      16u  // lens manifold surface: transform ray direction
                                    // emissive.y holds the manifold slot index
#define MAT_FLAG_TRANSMISSIVE  64u  // explicit glass surface (opacity→0 set by scene builder)

// ─── PendingRay: mid-bounce secondary emission queued for the reactive pass ─
// origin_flags.w holds pending-ray-specific pass flags (uint bits).
// dir_bounces.w  holds remaining bounce budget (uint bits).
#define PRAY_NO_REACTIVE    1u  // this ray won't spawn further reactive emissions
#define PRAY_DIRECT_ONLY    2u  // only volume-splat; skip geometry bounce
struct PendingRay {
    vec4 origin_flags;   // .xyz = world origin, .w = pray_flags (uint bits)
    vec4 dir_bounces;    // .xyz = direction,    .w = bounces remaining (uint bits)
    vec4 packet;         // .x=freq_hz, .y=phase, .z=energy, .w=coherence
    vec4 normal_triid;   // .xyz = hit normal at spawn, .w = source tri index
};

// ─── Pass-mode constants ─────────────────────────────────────────────────────
#define PASS_FORWARD  0   // forward emission from bdpt_sources + emissive tris
#define PASS_SENSOR   1   // backward camera/sensor visibility pass
#define PASS_REACTIVE 2   // reactive re-emission pass (processes PendingRayBuf)

struct Node {
    vec4 lo_left;
    vec4 hi_right;
    vec4 start_count;
};

layout(std430, binding = 0) readonly buffer TriBuf      { Tri        tris[];          };
layout(std430, binding = 1) readonly buffer NodeBuf     { Node       nodes[];          };
layout(std430, binding = 2) readonly buffer TriIdBuf    { int        tri_ids[];        };
layout(std430, binding = 3) buffer         SegmentBuf   { vec4       segs[];           };
layout(std430, binding = 4) buffer         CounterBuf   {
    uint seg_count;
    uint ray_count;
    uint hit_count;
    uint record_count;
    uint pending_count;  // reactive queue depth — written by forward pass
};
struct SourceRec {
    vec4 pos_weight; // xyz=position, w=ray-budget weight
    vec4 dir_kind;   // xyz=emission axis, w=reserved
    vec4 packet;     // x=freq/coord, y=phase, z=energy, w=coherence/sigma
};
layout(std430, binding = 5) readonly buffer SourceBuf     { SourceRec  bdpt_sources[]; };
layout(std430, binding = 6) buffer         PendingRayBuf  { PendingRay pending_rays[]; };

// Emission / remission / color profile data — flat float buffer at binding 7.
// Each profile header occupies exactly 20 consecutive floats (5 vec4s stride):
//   [0]  profile_type  0=EmissionProfile 1=RemissionProfile(spread)
//                      2=ColorProfile    3=PROFILE_MANIFOLD (remit+noodles)
//   [1]  noodle_start  — float index from buffer start where noodle rows begin
//                        (0 if not manifold)
//   [2]  noodle_count  — number of noodle rows in this manifold (0 if not manifold)
//   [3]  _pad
//   [4-7]  angular spread  (type, amp, center, q)
//   [8-11] phase spread
//   [12-15] frequency spread
//   [16-19] amplitude spread
// After all N headers, noodle rows follow (stride 12 floats each):
//   [0,1] u,v  [2,3] fu,fv  [4,5,6] in_dir  [7,8,9] out_dir  [10] opl  [11] _pad
#define PROFILE_MANIFOLD 3
layout(std430, binding = 7) readonly buffer EmissionProfileBuf { float ep_data[]; };
// ─── Wave-physics scale context spheres (binding 8) ──────────────────────────
// Each sphere names a medium region where wave-accurate propagation applies:
// in-medium phase shift k_n×Δr, evanescent decay exp(−k_im×Δr), and
// near-field 1/(1+r²) spreading vs geometric 1/(1+r) outside contexts.
// Binding 8 is in the SSBO namespace — safe alongside sampler bindings 8-15.
struct ScaleContext {
    vec4 center_radius;      // .xyz = sphere center  .w = sphere radius
    vec4 dtm_nre_nim_type;   // .x = dt_m  .y = n_real  .z = n_imag  .w = scale_type (1=wave)
};
layout(std430, binding = 8) readonly buffer ScaleContextBuf { ScaleContext scale_ctxs[]; };
layout(r32ui, binding = 0) uniform uimage3D uLayer0;
layout(r32ui, binding = 1) uniform uimage3D uLayer1;
layout(r32ui, binding = 2) uniform uimage3D uLayer2;
layout(r32ui, binding = 3) uniform uimage3D uLayer3;
layout(r32ui, binding = 4) uniform uimage3D uLayer4;
layout(r32ui, binding = 5) uniform uimage3D uLayer5;
layout(r32ui, binding = 6) uniform uimage3D uLayer6;
layout(r32ui, binding = 7) uniform uimage3D uLayer7;

uniform int   uTriCount;
uniform int   uNodeCount;
uniform int   uSegmentCap;
uniform int   uSegmentStride;
uniform int   uSourceCount;
// Batched dispatch: uBatchSize rays in this call, uBatchOffset = first global ray index.
// uTotalRaysPerSource controls the append_segment thinning ratio.
uniform int   uBatchSize;
uniform int   uBatchOffset;
uniform int   uTotalRaysPerSource;
uniform int   uMaxBounces;
uniform int   uSeed;
// Diagnostics: set 0 in production to eliminate ray/hit atomic overhead.
// SegmentCapture: set 0 to skip all segment writes (production accumulation only).
uniform int   uDiagnosticsEnabled;
uniform int   uSegmentCapture;
uniform vec3  uSrcPos;
uniform vec3  uSrcDir;
// Area-source parameters.  uSrcRadius > 0 jitters the origin uniformly over a
// disk of that radius lying in the plane perpendicular to uSrcDir.  uSrcConeCos
// is the cosine of the half-angle of the emission cone; -1.0 = full hemisphere,
// 0.0 = 90° half-angle, cos(30°)≈0.866 = narrow spotlight.
uniform float uSrcRadius;
uniform float uSrcConeCos;
// EmitterProfile directional model for per-source exact importance sampling.
// 0=LAMBERTIAN 1=DIPOLE_INPLANE_MIXED 2=GAUSSIAN_BEAM 3=ETENDUE_LIMITED
// 4=PROJECTIVE 5=HENYEY_GREENSTEIN 6=ISOTROPIC
uniform int   uSrcDirModel;
uniform float uSrcDirParam;  // theta_d_rad / cos_theta_max / hg_g / unused
uniform vec3  uBoxMin;
uniform vec3  uBoxMax;
uniform ivec3 uDims;
// Per-packet spectral sample.  The CPU expands source spectra stochastically:
// x=frequency/spectral coordinate, y=phase radians, z=packet energy, w=coherence.
uniform vec4  uSrcSpectrum;
uniform int   uLayerCount;
uniform float uLayerCentersHz[8];
uniform float uLayerWidthsOct[8];
uniform float uLayerGains[8];
// Beer-Lambert participating medium uniforms
uniform float uVolumeStepMeters;  // sample step size in metres (default 0.003)
uniform float uMediumScattering;  // scattering coefficient (default 1.0)
uniform float uMediumExtinction;  // extinction coefficient (default 0.5)
uniform float uAirDiffuseScatter;
uniform float uAirSpecularScatter;
uniform float uAirAnisotropy;
// Wave-physics context uniforms
uniform int   uScaleContextCount;   // entries in ScaleContextBuf (0 = none, disables context loops)
uniform float uSpeedOfMedium;       // propagation speed m/s (343 acoustic, 3e8 optical)
// Bidirectional sensor pass uniforms
// uMode PASS_FORWARD(0) = forward source emission (default).
// uMode PASS_SENSOR(1)  = backward sensor visibility; uSrcPos/uSrcDir are camera eye/dir.
// uMode PASS_REACTIVE(2)= reactive re-emission; processes pending_rays[0..pending_count-1].
// uFwdLayer[0-7] are the completed forward accumulation textures read as integer
// samplers — texelFetch returns raw uint counts.  uSensorNorm normalises them
// back to per-ray energy.  uSensorGain amplifies surface contributions so they
// emerge as bright shells in the volume march.
uniform int   uMode;
uniform int   uPendingCap;   // capacity of PendingRayBuf; 0 disables reactive queuing
uniform float uSensorNorm;
uniform float uSensorGain;
uniform float uSensorConeCos;
uniform float uSensorMisWeight;
layout(binding =  8) uniform usampler3D uFwdLayer0;
layout(binding =  9) uniform usampler3D uFwdLayer1;
layout(binding = 10) uniform usampler3D uFwdLayer2;
layout(binding = 11) uniform usampler3D uFwdLayer3;
layout(binding = 12) uniform usampler3D uFwdLayer4;
layout(binding = 13) uniform usampler3D uFwdLayer5;
layout(binding = 14) uniform usampler3D uFwdLayer6;
layout(binding = 15) uniform usampler3D uFwdLayer7;

const float EPS = 1e-7;
const float PI = 3.14159265358979323846;

uint hash_u(uint x) {
    x = x * 747796405u + 2891336453u;
    uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (word >> 22u) ^ word;
    x ^= x * 0x9e3779b9u;
    x ^= x >> 16;
    return x;
}

float rand01(inout uint s) {
    s = hash_u(s);
    return float(s & 0x00ffffffu) / 16777215.0;
}

float randn(inout uint s) {
    float u0 = max(rand01(s), 1e-6);
    float u1 = rand01(s);
    return sqrt(-2.0 * log(u0)) * cos(2.0 * PI * u1);
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

vec3 cone_dir(float u, float v, vec3 axis, float cos_theta_max) {
    float z = mix(1.0, clamp(cos_theta_max, -1.0, 1.0), u);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float a = 2.0 * PI * v;
    vec3 local = vec3(r * cos(a), r * sin(a), z);
    vec3 w = normalize(axis);
    vec3 up = abs(w.z) < 0.9 ? vec3(0,0,1) : vec3(0,1,0);
    vec3 x = normalize(cross(up, w));
    vec3 y = cross(w, x);
    return normalize(x * local.x + y * local.y + w * local.z);
}

// ─── EmitterProfile analytic importance samplers ────────────────────────────────────
#define DIRMODEL_LAMBERTIAN     0
#define DIRMODEL_DIPOLE_INPLANE 1
#define DIRMODEL_GAUSSIAN_BEAM  2
#define DIRMODEL_ETENDUE        3
// DIRMODEL_PROJECTIVE: ideal flat-top cone — BACKTRACE / SENSOR PASS ONLY.
// This is a programmatic ray-bundle model (e.g. structured-light sensor frustum).
// It is NOT a physical forward-emission model and MUST NOT be assigned to any
// real emitter (stage lights, LEDs, lasers).  Real etendue-limited sources use
// DIRMODEL_ETENDUE (model_int=3).  Using PROJECTIVE on a forward source produces
// physically meaningless energy distributions.
#define DIRMODEL_PROJECTIVE     4
#define DIRMODEL_HG             5
#define DIRMODEL_ISOTROPIC      6

// Build a world-space direction from a local frame aligned to `axis`.
vec3 _to_world_axis(vec3 local, vec3 axis) {
    vec3 w  = normalize(axis);
    vec3 up = abs(w.z) < 0.9 ? vec3(0,0,1) : vec3(0,1,0);
    vec3 x  = normalize(cross(up, w));
    vec3 y  = cross(w, x);
    return normalize(x * local.x + y * local.y + w * local.z);
}

// DIPOLE_INPLANE_MIXED: I(θ) = (1 + cos²θ) / 2  on the full sphere.
// Exact CDF inversion via Cardano's formula:
//   F(θ) = (c³ + 3c + 4) / 8  where c = cosθ
//   ⇒  c³ + 3c + (4 - 8ξ) = 0
float _sample_dipole_inplane(float xi) {
    float k = 4.0 - 8.0 * xi;
    float M = sqrt(k * k * 0.25 + 1.0);
    return pow(M - k * 0.5, 1.0 / 3.0) - pow(M + k * 0.5, 1.0 / 3.0);
}

// GAUSSIAN_BEAM: paraxial Rayleigh importance sampler.
// Exact for small divergence beams: θ ~ Rayleigh(θ_d)  ⇒  θ = θ_d * sqrt(-ln u)
vec3 _sample_gaussian_beam(float u, float v, vec3 axis, float theta_d) {
    float theta = theta_d * sqrt(max(0.0, -log(max(u, 1e-7))));
    theta = min(theta, PI);
    float sinT = sin(theta);
    float cosT = cos(theta);
    return _to_world_axis(vec3(sinT * cos(2.0*PI*v), sinT * sin(2.0*PI*v), cosT), axis);
}

// HENYEY_GREENSTEIN: exact analytic CDF inversion.
float _sample_hg_cos(float xi, float g) {
    if (abs(g) < 1e-3) return 2.0 * xi - 1.0;
    float s = (1.0 - g*g) / (1.0 - g + 2.0*g*xi);
    return clamp((1.0 + g*g - s*s) / (2.0*g), -1.0, 1.0);
}

// Master emission direction sampler -- dispatches on uSrcDirModel.
// u1, u2 are independent uniform [0,1) samples.
vec3 _emit_dir(float u1, float u2, vec3 axis) {
    int   model = uSrcDirModel;
    float param = uSrcDirParam;

    if (model == DIRMODEL_DIPOLE_INPLANE) {
        float cosT = _sample_dipole_inplane(u1);
        float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
        float phi  = 2.0 * PI * u2;
        return _to_world_axis(vec3(sinT*cos(phi), sinT*sin(phi), cosT), axis);
    }
    if (model == DIRMODEL_GAUSSIAN_BEAM) {
        return _sample_gaussian_beam(u1, u2, axis, param);
    }
    if (model == DIRMODEL_ETENDUE) {
        // param = cos(theta_max).  ETENDUE_LIMITED is the correct model for any real
        // optical system with a defined numerical aperture (spotlight, LED+lens, fibre).
        // It is forward-emission safe.  param = cos(asin(NA)).
        return cone_dir(u1, u2, axis, param);
    }
    if (model == DIRMODEL_PROJECTIVE) {
        // BACKTRACE / SENSOR PASS ONLY.  param = cos(theta_max).
        // Do not assign this model to any physical forward source.  It is used
        // exclusively for the sensor visibility backward pass where rays are
        // confined to a measurement frustum, NOT sampled from a real emitter.
        return cone_dir(u1, u2, axis, param);
    }
    if (model == DIRMODEL_HG) {
        float cosT = _sample_hg_cos(u1, param);
        float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
        float phi  = 2.0 * PI * u2;
        return _to_world_axis(vec3(sinT*cos(phi), sinT*sin(phi), cosT), axis);
    }
    if (model == DIRMODEL_ISOTROPIC) {
        // Full-sphere uniform
        float cosT = 2.0 * u1 - 1.0;
        float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
        float phi  = 2.0 * PI * u2;
        return vec3(sinT*cos(phi), sinT*sin(phi), cosT);
    }
    // Default: LAMBERTIAN cosine-weighted hemisphere (exact)
    return cosine_dir(u1, u2, axis);
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

// Recompute Möller-Trumbore barycentric (u,v) for a known-hit triangle.
// Returns the same u,v produced during intersection — direct surface params.
vec2 bary_uv(vec3 ro, vec3 rd, Tri t) {
    vec3 h = cross(rd, t.e2.xyz);
    float a = dot(t.e1.xyz, h);
    if (abs(a) < 1e-10) return vec2(0.0);
    float f = 1.0 / a;
    vec3 s = ro - t.v0.xyz;
    float u = f * dot(s, h);
    vec3 q = cross(s, t.e1.xyz);
    float v = f * dot(rd, q);
    return vec2(u, v);
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

vec4 film_activation(float freq_hz, float phase, float energy) {
    float coherence = 0.75 + 0.25 * cos(phase);
    float total = 0.0;
    int n = clamp(uLayerCount, 0, 8);
    for (int i = 0; i < n; i++) {
        float ctr = max(uLayerCentersHz[i], 1e-3);
        float wid = max(uLayerWidthsOct[i], 1e-3);
        float oct = log(max(freq_hz, 1e-3) / ctr) / log(2.0);
        float w = exp(-0.5 * oct * oct / (wid * wid));
        total += uLayerGains[i] * w;
    }
    total *= energy * coherence;
    float per = total / max(float(max(uLayerCount, 1)), 1.0);
    return vec4(per);
}

void layer_splat(int i, ivec3 q, uint val) {
    if      (i == 0) imageAtomicAdd(uLayer0, q, val);
    else if (i == 1) imageAtomicAdd(uLayer1, q, val);
    else if (i == 2) imageAtomicAdd(uLayer2, q, val);
    else if (i == 3) imageAtomicAdd(uLayer3, q, val);
    else if (i == 4) imageAtomicAdd(uLayer4, q, val);
    else if (i == 5) imageAtomicAdd(uLayer5, q, val);
    else if (i == 6) imageAtomicAdd(uLayer6, q, val);
    else if (i == 7) imageAtomicAdd(uLayer7, q, val);
}

void splat_spectral(vec3 p, vec4 packet) {
    vec3 uvw = (p - uBoxMin) / max(uBoxMax - uBoxMin, vec3(1e-6));
    ivec3 q = ivec3(floor(uvw * vec3(uDims)));
    if (any(lessThan(q, ivec3(0))) || any(greaterThanEqual(q, uDims))) return;
    float freq   = packet.x;
    float energy = packet.z;
    float coherence = 0.75 + 0.25 * cos(packet.y);
    int n = clamp(uLayerCount, 0, 8);
    for (int i = 0; i < n; i++) {
        float ctr = max(uLayerCentersHz[i], 1e-3);
        float wid = max(uLayerWidthsOct[i], 1e-3);
        float oct = log(max(freq, 1e-3) / ctr) / log(2.0);
        float w = exp(-0.5 * oct * oct / (wid * wid));
        float activation = uLayerGains[i] * energy * coherence * w;
        uint val = uint(clamp(activation * 65535.0, 0.0, 1e6));
        if (val > 0u) layer_splat(i, q, val);
    }
}

void splat_activation(vec3 p, vec4 response) {
    vec3 uvw = (p - uBoxMin) / max(uBoxMax - uBoxMin, vec3(1e-6));
    ivec3 q = ivec3(floor(uvw * vec3(uDims)));
    if (any(lessThan(q, ivec3(0))) || any(greaterThanEqual(q, uDims))) return;
    float total = max(response.x + response.y + response.z + response.w, 0.0);
    int n = clamp(uLayerCount, 1, 8);
    float per_layer = total / float(n);
    uint val = uint(clamp(per_layer * 65535.0, 0.0, 1e6));
    if (val == 0u) return;
    for (int i = 0; i < n; i++) {
        layer_splat(i, q, val);
    }
}

float fwd_sample_layer(int i, ivec3 q) {
    if      (i == 0) return float(texelFetch(uFwdLayer0, q, 0).r);
    else if (i == 1) return float(texelFetch(uFwdLayer1, q, 0).r);
    else if (i == 2) return float(texelFetch(uFwdLayer2, q, 0).r);
    else if (i == 3) return float(texelFetch(uFwdLayer3, q, 0).r);
    else if (i == 4) return float(texelFetch(uFwdLayer4, q, 0).r);
    else if (i == 5) return float(texelFetch(uFwdLayer5, q, 0).r);
    else if (i == 6) return float(texelFetch(uFwdLayer6, q, 0).r);
    else if (i == 7) return float(texelFetch(uFwdLayer7, q, 0).r);
    return 0.0;
}

vec4 sample_activation_field(vec3 p) {
    vec3 uvw = (p - uBoxMin) / max(uBoxMax - uBoxMin, vec3(1e-6));
    ivec3 q  = ivec3(floor(uvw * vec3(uDims)));
    if (any(lessThan(q, ivec3(0))) || any(greaterThanEqual(q, uDims))) {
        return vec4(0.0);
    }
    float norm = max(1.0, uSensorNorm) * 65535.0;
    float total = 0.0;
    int n = clamp(uLayerCount, 1, 8);
    for (int i = 0; i < n; i++) {
        total += fwd_sample_layer(i, q);
    }
    float avg = total / (norm * float(n));
    return vec4(avg);
}

float balance_weight(vec4 fwd, float camera_pdf) {
    float source_pdf = max(dot(fwd, vec4(0.25)), 1e-6);
    float cp = max(camera_pdf * max(uSensorMisWeight, 1e-6), 1e-6);
    return cp / (cp + source_pdf);
}

SourceRec sample_bdpt_source(inout uint rng, out float source_pdf) {
    int count = max(1, uSourceCount);
    int si = int(floor(rand01(rng) * float(count)));
    si = clamp(si, 0, count - 1);
    source_pdf = 1.0 / float(count);
    SourceRec src = bdpt_sources[si];
    src.packet.x *= exp2(randn(rng) * max(src.packet.w, 1e-4));
    src.packet.y += randn(rng) * PI;
    return src;
}

bool visible_segment(vec3 a, vec3 b) {
    vec3 d = b - a;
    float max_t = length(d);
    if (max_t <= 1e-5) return false;
    d /= max_t;
    float hit_t;
    int hit = nearest_hit(a + d * 1e-4, d, hit_t);
    return hit < 0 || hit_t >= max_t - 3e-4;
}

vec4 connect_emitter(vec3 p, vec3 normal, vec3 view_dir, float path_pdf,
                     float bsdf_weight, inout uint rng) {
    if (uSourceCount <= 0) return vec4(0.0);
    float src_pdf;
    SourceRec src = sample_bdpt_source(rng, src_pdf);
    vec3 to_src = src.pos_weight.xyz - p;
    float dist2 = max(dot(to_src, to_src), 1e-6);
    float dist = sqrt(dist2);
    vec3 wi = to_src / dist;
    if (!visible_segment(p, src.pos_weight.xyz)) return vec4(0.0);

    vec3 n = normalize(normal);
    float cos_surf = max(0.0, dot(n, wi));
    float cam_cos = max(0.0, dot(n, -view_dir));
    vec3 src_axis = normalize(src.dir_kind.xyz);
    float src_lobe = max(0.08, dot(-wi, src_axis));
    float geom = cos_surf * max(0.15, cam_cos) * src_lobe / (1.0 + dist2);
    vec4 emitted = film_activation(src.packet.x, src.packet.y,
                                   src.packet.z * max(src.pos_weight.w, 1e-9));

    // Power heuristic between sampled emitter connection and camera path scatter.
    float connect_pdf = max(src_pdf * dist2 / max(cos_surf, 1e-4), 1e-6);
    float cp = max(path_pdf, 1e-6);
    float mis = (connect_pdf * connect_pdf) /
                (connect_pdf * connect_pdf + cp * cp);
    return emitted * geom * bsdf_weight * mis * float(max(1, uSourceCount));
}

void splat_segment_spectral(vec3 a, vec3 b, vec4 packet) {
    float len = length(b - a);
    int steps = max(1, int(len / uVolumeStepMeters));
    vec3 dir = (b - a) / max(len, 1e-6);
    for (int i = 0; i <= steps; ++i) {
        float s    = float(i) / float(steps);
        float dist = s * len;
        vec3  p    = a + dir * dist;
        float freq_factor = clamp(packet.x / 440.0, 0.25, 8.0);
        float transmit = exp(-uMediumExtinction * dist * freq_factor);
        vec4 scatter  = vec4(packet.x, packet.y,
                             packet.z * transmit * uMediumScattering * uVolumeStepMeters,
                             packet.w);
        splat_spectral(p, scatter);
    }
}

// ─── Ray-sphere interval ─────────────────────────────────────────────────────
// Returns (t_enter, t_exit) along the ray ro+t*rd.
// Returns (1e30, -1.0) if the ray misses or exits behind the origin.
vec2 ray_sphere_t(vec3 ro, vec3 rd, vec3 center, float radius) {
    vec3  oc   = ro - center;
    float b    = dot(oc, rd);
    float c    = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) return vec2(1e30, -1.0);
    float sq = sqrt(disc);
    return vec2(-b - sq, -b + sq);
}

// ─── Multiscale segment splat ─────────────────────────────────────────────────
// Walks [a, b] through registered ScaleContext spheres.  Inside each wave-type
// sphere the sub-samples use: phase k_n×step (via display proportionality),
// evanescent exp(−k_im×step), and 1/(1+r²) near-field spreading.
// Outside contexts the existing Beer-Lambert geometric splat is used.
// path_start: accumulated ray path length at point `a` (for near-field spreading).
// Does NOT modify the caller's packet — only writes to the 3D accumulation images.
void splat_segment_multiscale(vec3 a, vec3 b, vec4 packet, float path_start) {
    float seg_len = length(b - a);
    if (seg_len < EPS) return;
    // Fast path: no contexts registered
    if (uScaleContextCount == 0) {
        splat_segment_spectral(a, b, packet);
        return;
    }
    vec3 seg_dir = (b - a) / seg_len;
    // Collect context-sphere intervals intersecting [0, seg_len], sorted by t_enter
    float iv_te[8]; float iv_tx[8]; int iv_ci[8]; int n_iv = 0;
    int n_ctx = min(uScaleContextCount, 16);
    for (int ci = 0; ci < n_ctx; ci++) {
        vec2 tt = ray_sphere_t(a, seg_dir,
                               scale_ctxs[ci].center_radius.xyz,
                               scale_ctxs[ci].center_radius.w);
        if (tt.y <= EPS) continue;
        float te = max(tt.x, 0.0);
        float tx = min(tt.y, seg_len);
        if (te >= tx - EPS) continue;
        int ins = n_iv;
        for (int j = 0; j < n_iv; j++) { if (te < iv_te[j]) { ins = j; break; } }
        for (int j = min(n_iv, 7); j > ins; j--) {
            iv_te[j] = iv_te[j-1]; iv_tx[j] = iv_tx[j-1]; iv_ci[j] = iv_ci[j-1];
        }
        if (n_iv < 8) { iv_te[ins] = te; iv_tx[ins] = tx; iv_ci[ins] = ci; n_iv++; }
    }
    if (n_iv == 0) {
        splat_segment_spectral(a, b, packet);
        return;
    }
    // Walk sub-regions
    float t_walk    = 0.0;
    float cur_phase = packet.y;
    for (int ii = 0; ii <= n_iv; ii++) {
        float t_end = (ii < n_iv) ? iv_te[ii] : seg_len;
        // Geometric sub-span [t_walk, t_end]
        if (t_end - t_walk > EPS) {
            float span = t_end - t_walk;
            vec3  p0s  = a + t_walk * seg_dir;
            vec3  p1s  = a + t_end  * seg_dir;
            int steps  = max(1, int(span / uVolumeStepMeters));
            for (int k = 0; k <= steps; k++) {
                float s    = float(k) / float(steps);
                float dist = s * span;
                vec3  p    = p0s + (p1s - p0s) * s;
                float freq_f   = clamp(packet.x / 440.0, 0.25, 8.0);
                float transmit = exp(-uMediumExtinction * dist * freq_f);
                vec4 geo_pkt = vec4(packet.x,
                                    cur_phase + dist * packet.x * 0.000021,
                                    packet.z * transmit * uMediumScattering * uVolumeStepMeters,
                                    packet.w);
                splat_spectral(p, geo_pkt);
            }
            cur_phase += span * packet.x * 0.000021;
        }
        t_walk = t_end;
        if (ii >= n_iv) break;
        // Wave-context sub-span [iv_te[ii], iv_tx[ii]]
        float ctx_t0   = iv_te[ii];
        float ctx_t1   = iv_tx[ii];
        float ctx_span = ctx_t1 - ctx_t0;
        int   ci       = iv_ci[ii];
        float n_re     = scale_ctxs[ci].dtm_nre_nim_type.y;
        float n_im     = scale_ctxs[ci].dtm_nre_nim_type.z;
        float dt_m     = max(scale_ctxs[ci].dtm_nre_nim_type.x, ctx_span * 0.03125);
        // Cap sub-steps at 32 for GPU budget; wave phase is exact per-step
        int n_sub = clamp(int(ctx_span / dt_m), 1, 32);
        for (int ss = 0; ss < n_sub; ss++) {
            float t0_ss  = ctx_t0 + float(ss)     / float(n_sub) * ctx_span;
            float step   = ctx_span / float(n_sub);
            vec3  p_mid  = a + (t0_ss + step * 0.5) * seg_dir;
            // Phase: k_n × step = 2π f n_re / c × step → via display proportionality n_re × legacy
            float phase_step = step * packet.x * 0.000021 * n_re;
            // Evanescent decay: exp(−k_im × step) via same proportionality
            float amp_decay  = exp(-step * packet.x * 0.000021 * n_im);
            // Near-field spreading: 1/(1+r²) where r = total path to midpoint
            float r_mid    = path_start + t0_ss + step * 0.5;
            float spread_w = 1.0 / (1.0 + r_mid * r_mid);
            float freq_f   = clamp(packet.x / 440.0, 0.25, 8.0);
            float transmit = exp(-uMediumExtinction * float(ss) / float(n_sub) * ctx_span * freq_f);
            vec4 wave_pkt = vec4(packet.x,
                                 cur_phase + phase_step * 0.5,
                                 packet.z * amp_decay * spread_w *
                                     uMediumScattering * step * transmit,
                                 packet.w);
            splat_spectral(p_mid, wave_pkt);
            cur_phase += phase_step;
        }
        t_walk = ctx_t1;
    }
}

// ─── Wave-context physics: correct master packet for in-medium phase & decay ──
// Call once per segment AFTER splatting.  Returns packet with:
//   phase  += k_display × (n_re - 1) × ctx_span  (extra phase in dense medium)
//   energy ×= exp(−k_display × n_im × ctx_span)   (evanescent decay)
// Only wave-type contexts (scale_type == 1) are applied.
vec4 apply_wave_context_physics(vec3 ro, vec3 rd, float best, vec4 packet) {
    int n_ctx = min(uScaleContextCount, 16);
    for (int ci = 0; ci < n_ctx; ci++) {
        if (int(round(scale_ctxs[ci].dtm_nre_nim_type.w)) != 1) continue;
        vec2 tt  = ray_sphere_t(ro, rd,
                                scale_ctxs[ci].center_radius.xyz,
                                scale_ctxs[ci].center_radius.w);
        if (tt.y <= EPS) continue;
        float te       = max(tt.x, 0.0);
        float tx       = min(tt.y, best);
        float ctx_span = tx - te;
        if (ctx_span < EPS) continue;
        float n_re = scale_ctxs[ci].dtm_nre_nim_type.y;
        float n_im = scale_ctxs[ci].dtm_nre_nim_type.z;
        // Extra phase beyond ambient (medium denser than vacuum/air)
        packet.y += ctx_span * packet.x * 0.000021 * (n_re - 1.0);
        // Evanescent amplitude decay through absorptive medium
        packet.z *= exp(-ctx_span * packet.x * 0.000021 * n_im);
    }
    return packet;
}

// ─── Forward declaration (definition follows spawn_edge_diffraction) ────────
void append_pending_ray(vec3 origin, vec3 dir, vec4 packet,
                        int bounces_rem, uint pray_flags,
                        vec3 hit_normal, int tri_id);

// ─── Keller GTD edge diffraction ─────────────────────────────────────────────
// Spawns one Huygens secondary into PendingRayBuf when hp is within ~2λ of
// edge [vA, vB].  Amplitude = packet.z × sqrt(λ/dist_edge) (cylindrical).
// Direction = random on the Keller diffraction cone (preserves angle with edge).
// Phase = packet.y + π/2 (Huygens secondary wavelet).
void spawn_edge_diffraction(
        vec3 hp, vec3 rd, vec4 packet, int bounce,
        vec3 vA, vec3 vB, vec3 face_normal, int hit_tri,
        inout uint rng)
{
    vec3  edge_v   = vB - vA;
    float edge_len = length(edge_v);
    if (edge_len < EPS) return;
    vec3  te     = edge_v / edge_len;
    float t_near = clamp(dot(hp - vA, te), 0.0, edge_len);
    vec3  edge_pt = vA + t_near * te;
    float dist_to_edge = length(hp - edge_pt);
    // λ in display-scale units: 1 / (freq × 0.000021)
    float k_scale  = max(packet.x * 0.000021, 1e-9);
    float lambda_d = 1.0 / k_scale;
    if (dist_to_edge > lambda_d * 2.0) return;
    int bounces_rem = uMaxBounces - bounce - 1;
    if (bounces_rem <= 0) return;
    // Keller cylindrical-wave amplitude: 0.3 × sqrt(λ / r_edge)
    float r_edge = max(dist_to_edge, lambda_d * 0.1);
    float keller = min(packet.z * sqrt(lambda_d / r_edge) * 0.3, packet.z * 0.5);
    if (keller < 0.002) return;
    // Keller diffraction-cone direction (random azimuth, same polar angle as rd)
    float cos_e = dot(-normalize(rd), te);
    float sin_e = sqrt(max(0.0, 1.0 - cos_e * cos_e));
    vec3  w     = te;
    vec3  up    = abs(w.z) < 0.9 ? vec3(0, 0, 1) : vec3(0, 1, 0);
    vec3  ex    = normalize(cross(w, up));
    vec3  ey    = cross(w, ex);
    float phi   = 2.0 * PI * rand01(rng);
    vec3  d_raw = w * cos_e + (ex * cos(phi) + ey * sin(phi)) * sin_e;
    if (dot(d_raw, face_normal) < 0.0) d_raw = reflect(d_raw, face_normal);
    vec3  diffr_dir = normalize(d_raw);
    vec4 diffr_pkt = vec4(packet.x, packet.y + PI * 0.5, keller, packet.w);
    append_pending_ray(edge_pt + diffr_dir * 1e-5, diffr_dir, diffr_pkt,
                       bounces_rem, PRAY_NO_REACTIVE, face_normal, hit_tri);
}

void append_segment(vec3 p0, vec3 p1, float energy) {
    // Skip all segment buffer writes unless capture is explicitly enabled.
    if (uSegmentCapture == 0 || uSegmentCap <= 0) return;
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

// ─── Reactive ray queue ──────────────────────────────────────────────────────
// Called during PASS_FORWARD when a MAT_FLAG_REACTIVE surface is hit.
// The secondary re-emission ray is enqueued for processing in PASS_REACTIVE.
// dir must already be normalised; bounces_rem is the budget after this hit.
void append_pending_ray(vec3 origin, vec3 dir, vec4 packet,
                        int bounces_rem, uint pray_flags, vec3 hit_normal, int tri_id) {
    if (uPendingCap <= 0) return;
    uint idx = atomicAdd(pending_count, 1u);
    if (idx >= uint(uPendingCap)) return;
    pending_rays[idx].origin_flags = vec4(origin, uintBitsToFloat(pray_flags));
    pending_rays[idx].dir_bounces  = vec4(dir,    uintBitsToFloat(uint(max(bounces_rem, 0))));
    pending_rays[idx].packet       = packet;
    pending_rays[idx].normal_triid = vec4(hit_normal, uintBitsToFloat(uint(tri_id)));
}

// ─── Sensor (camera-view) pass ──────────────────────────────────────────────
// Shoots rays from the camera eye into the scene.  At each surface hit it
// reads the already-accumulated forward light field and deposits a Lambert-
// weighted surface contribution back into the same band images.  This makes
// every illuminated surface visible as a bright voxel shell in the volume
// march, giving correct camera-POV surface radiance at essentially zero extra
// geometry overhead.
void sensor_main(uint ray_id, inout uint rng) {
    // Camera/sensor cone path.  This is the reverse half of the bidirectional
    // estimator: it samples source-built film activation in air and on
    // surfaces, then writes a MIS-weighted camera-correlated contribution.
    vec3 ro = uSrcPos;
    vec3 rd = cone_dir(rand01(rng), rand01(rng), uSrcDir, uSensorConeCos);
    float cone_pdf = 1.0 / max(2.0 * PI * (1.0 - clamp(uSensorConeCos, -1.0, 1.0)), 1e-6);
    float path_weight = 1.0;

    for (int bounce = 0; bounce < uMaxBounces; ++bounce) {
        float best = 1e30;
        int hit = nearest_hit(ro, rd, best);

        float march_len = (hit >= 0) ? best : length(uBoxMax - uBoxMin);
        int air_steps = max(1, int(march_len / max(uVolumeStepMeters * 6.0, 1e-4)));
        float jitter = rand01(rng);
        for (int i = 0; i < air_steps; ++i) {
            float t = (float(i) + jitter) / float(air_steps);
            vec3 ap = ro + rd * (t * march_len);
            vec4 air_fwd = sample_activation_field(ap);
            if (dot(air_fwd, vec4(1.0)) > 0.0) {
                float forward_lobe = pow(max(0.0, dot(rd, normalize(uSrcDir))), max(1.0, uAirAnisotropy));
                float air_phase = uAirDiffuseScatter + uAirSpecularScatter * forward_lobe;
                float mis = balance_weight(air_fwd, cone_pdf);
                float trans = exp(-uMediumExtinction * t * march_len);
                splat_activation(ap, air_fwd * air_phase * trans * mis * uSensorGain * uVolumeStepMeters);
            }
            vec4 air_conn = connect_emitter(ap, -rd, rd, cone_pdf,
                                            path_weight * uVolumeStepMeters *
                                            (uAirDiffuseScatter + uAirSpecularScatter),
                                            rng);
            if (dot(air_conn, vec4(1.0)) > 0.0) {
                splat_activation(ap, air_conn * uSensorGain);
            }
        }

        if (hit < 0) break;
        vec3 hp = ro + rd * best;
        vec4 fwd = sample_activation_field(hp);

        Tri  tri      = tris[hit];
        vec3 geom_n   = normalize(tri.normal.xyz);
        bool interior = dot(rd, geom_n) > 0.0;
        vec3 n        = interior ? -geom_n : geom_n;
        vec4 mat      = interior ? tri.mat_in : tri.mat_out;
        float reflectivity = clamp(mat.x, 0.02, 0.98);
        float diffusion    = clamp(mat.y, 0.0,  1.0);
        float ior          = max(tri.mat_in.w, 1.0);
        float opacity      = clamp(tri.mat_out.w, 0.0, 1.0);
        float cos_theta    = max(0.0, dot(-rd, n));

        if (opacity < 0.999) {
            // ── Refractive surface: forward rays bend through glass ───────────
            // Energy splats at the surface entry; ray continues refracted.
            float eta      = interior ? ior : (1.0 / ior);
            float cos_t_sq = 1.0 - eta * eta * (1.0 - cos_theta * cos_theta);
            float fresnel_r;
            vec3  refr_dir;
            if (cos_t_sq <= 0.0) {
                fresnel_r = 1.0;
                refr_dir  = rd;
            } else {
                float cos_t = sqrt(cos_t_sq);
                float rs = (cos_theta - ior * cos_t) / max(cos_theta + ior * cos_t, 1e-6);
                float rp = (ior * cos_theta - cos_t) / max(ior * cos_theta + cos_t, 1e-6);
                fresnel_r = clamp(0.5 * (rs * rs + rp * rp), 0.0, 1.0);
                refr_dir  = normalize(refract(rd, n, eta));
            }
            float p_reflect = mix(fresnel_r, 1.0, opacity);
            if (rand01(rng) < p_reflect) {
                rd = normalize(reflect(rd, n));
                if (dot(rd, n) < 0.0) rd = cosine_dir(rand01(rng), rand01(rng), n);
            } else {
                rd = refr_dir;
                cone_pdf *= (1.0 - fresnel_r);
            }
            ro = hp + rd * 1e-5;
        } else {
            // ── Opaque surface ─────────────────────────────────────────────────
            float mis = balance_weight(fwd, cone_pdf);
            float dist_falloff = 1.0 / (1.0 + best * best);
            vec4 contribution = fwd * reflectivity * cos_theta * dist_falloff * mis * uSensorGain;
            splat_activation(hp, contribution);

            vec4 direct = connect_emitter(hp, n, rd, cone_pdf,
                                          path_weight * reflectivity * max(0.05, diffusion),
                                          rng);
            if (dot(direct, vec4(1.0)) > 0.0) {
                splat_activation(hp, direct * uSensorGain);
            }

            if (dot(contribution + direct, vec4(1.0)) < 0.001) break;
            float survival = clamp(reflectivity, 0.05, 0.98);
            if (rand01(rng) > survival) break;
            path_weight *= reflectivity / survival;
            if (rand01(rng) < diffusion) {
                rd = cosine_dir(rand01(rng), rand01(rng), n);
                cone_pdf *= max(0.05, 1.0 / PI);
            } else {
                rd = normalize(reflect(rd, n));
                if (dot(rd, n) < 0.0) rd = cosine_dir(rand01(rng), rand01(rng), n);
                cone_pdf *= max(0.05, 1.0 - diffusion);
            }
            ro = hp + rd * 1e-5;
        }
    }
}

// Forward declaration — definition follows void main().
void reactive_main(uint gid, inout uint rng);

void main() {
    uint groups_x = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * groups_x;
    if (gid >= uint(uBatchSize)) return;
    // Global ray index stays unique across batches for RNG independence.
    uint ray_id = gid + uint(uBatchOffset);
    if (uDiagnosticsEnabled != 0) atomicAdd(ray_count, 1u);
    uint rng = hash_u(ray_id ^ uint(uSeed));

    // ── Pass dispatch ────────────────────────────────────────────────────────
    if (uMode == PASS_SENSOR) {
        sensor_main(ray_id, rng);
        return;
    }
    if (uMode == PASS_REACTIVE) {
        // gid indexes directly into PendingRayBuf; no source-based ray gen.
        reactive_main(gid, rng);
        return;
    }

    // ── Pre-baked ray: read pos/dir/phase/freq directly from SSBO ───────────
    // bdpt_sources[ray_idx] was fully sampled CPU-side from RayOrder.bake_rays()
    // using the source's EmitterProfile (directional model + PhaseState).
    // The GPU does NOT generate directions or phases — it only traces what it
    // is handed.
    int _ray_idx = int(ray_id % uint(max(uSourceCount, 1)));
    SourceRec _prebaked = bdpt_sources[_ray_idx];
    vec3 ro = _prebaked.pos_weight.xyz;
    vec3 rd = normalize(_prebaked.dir_kind.xyz);
    vec4 packet = _prebaked.packet;   // .x=freq_hz, .y=phase(pre-baked), .z=energy, .w=0

    float path_len = 0.0;   // accumulated path length for near-field and context spreading
    for (int bounce = 0; bounce < uMaxBounces; ++bounce) {
        float best = 1e30;
        int hit = nearest_hit(ro, rd, best);
        if (hit < 0) break;
        if (uDiagnosticsEnabled != 0) atomicAdd(hit_count, 1u);
        vec3 hp = ro + rd * best;
        splat_segment_multiscale(ro, hp, packet, path_len);
        packet = apply_wave_context_physics(ro, rd, best, packet);
        // Segment capture: thin proportionally to total ray count.
        // Only fires when uSegmentCapture != 0 (production disables this path).
        if (uSegmentCapture != 0) {
            uint thin = max(1u, uint(uTotalRaysPerSource) / 16384u);
            if ((ray_id % thin) == 0u) {
                if (uDiagnosticsEnabled != 0) atomicAdd(record_count, 1u);
                append_segment(ro, hp, packet.z);
            }
        }
        Tri tri = tris[hit];
        uint  mat_flags_bits = floatBitsToUint(tri.normal.w);
        vec3  geom_n         = normalize(tri.normal.xyz);
        bool  interior_face  = dot(rd, geom_n) > 0.0;
        vec4  mat            = interior_face ? tri.mat_in : tri.mat_out;
        float reflectivity   = clamp(mat.x, 0.02, 0.98);
        float diffusion      = clamp(mat.y, 0.0, 1.0);
        float absorption     = clamp(mat.z, 0.0, 2.0);
        float ior            = max(tri.mat_in.w, 1.0);
        float opacity        = clamp(tri.mat_out.w, 0.0, 1.0);
        vec3  n              = interior_face ? -geom_n : geom_n;
        float cos_i          = max(0.0, dot(-rd, n));

        // ── Emissive surface: deposit radiance into the volume field ──────────
        // Independent of bounce direction — any ray that grazes an emissive
        // surface sees its energy contribution.
        if ((mat_flags_bits & MAT_FLAG_EMISSIVE) != 0u && tri.albedo.w > 0.001) {
            float emit_scale = tri.albedo.w * cos_i;
            vec4  emit_pkt   = vec4(packet.x, packet.y,
                                    emit_scale * max(tri.emissive.x,
                                                 max(tri.emissive.y, tri.emissive.z)),
                                    packet.w);
            splat_spectral(hp, emit_pkt);
        }

        // ── Edge diffraction at every triangle hit ─────────────────────────
        // Fires unconditionally — before any break — so absorbers (iris blades,
        // walls, stops) produce real Huygens secondaries from their physical
        // edges.  Keller amplitude threshold culls negligible contributions.
        {
            vec3 v0_e = tri.v0.xyz;
            vec3 v1_e = v0_e + tri.e1.xyz;
            vec3 v2_e = v0_e + tri.e2.xyz;
            spawn_edge_diffraction(hp, rd, packet, bounce, v0_e, v1_e, n, hit, rng);
            spawn_edge_diffraction(hp, rd, packet, bounce, v1_e, v2_e, n, hit, rng);
            spawn_edge_diffraction(hp, rd, packet, bounce, v2_e, v0_e, n, hit, rng);
        }

        // ── Pure absorber (including iris blades): terminate after edge work ──
        if ((mat_flags_bits & MAT_FLAG_ABSORBER) != 0u) break;

        // ── Reactive (fluorescent/re-emitting) surface ────────────────────────
        // Enqueue a Stokes-shifted secondary ray into the pending queue.
        // The current ray still bounces normally so the primary interaction is
        // captured; the secondary channel is handled by PASS_REACTIVE.
        if ((mat_flags_bits & MAT_FLAG_REACTIVE) != 0u) {
            float stokes_shift = tri.emissive.w;   // Hz, positive = red-shift
            float react_freq   = max(packet.x - stokes_shift, 20.0);
            vec3  react_dir    = cosine_dir(rand01(rng), rand01(rng), n);
            vec4  react_pkt    = vec4(react_freq,
                                      packet.y + PI * 0.5,
                                      packet.z * clamp(reflectivity, 0.0, 1.0),
                                      packet.w);
            int bounces_rem = uMaxBounces - bounce - 1;
            if (bounces_rem > 0 && react_pkt.z > 0.002) {
                append_pending_ray(hp + react_dir * 1e-5, react_dir, react_pkt,
                                   bounces_rem, PRAY_NO_REACTIVE, n, hit);
            }
        }

        // ── Transform LUT (manifold remission profile) — forward pass ─────────
        // Mirror of reactive_main: if the struck triangle carries a
        // PROFILE_MANIFOLD remission entry, redirect via IDW lookup and continue.
        {
            int remit_idx_f = int(round(tri.emissive.y));
            if (remit_idx_f >= 0) {
                int pbase_f = remit_idx_f * 20;
                if (int(ep_data[pbase_f]) == PROFILE_MANIFOLD) {
                    int noodle_start_f = int(ep_data[pbase_f + 1]);
                    int noodle_count_f = int(ep_data[pbase_f + 2]);
                    if (noodle_count_f > 0) {
                        vec2 buv_f = bary_uv(ro, rd, tri);
                        float uf = buv_f.x;
                        float vf = buv_f.y;
                        const int KF = 8;
                        int   best_idx_f[KF];
                        float best_d2_f[KF];
                        for (int ki = 0; ki < KF; ki++) {
                            best_idx_f[ki] = -1;
                            best_d2_f[ki]  = 1e30;
                        }
                        for (int ni = 0; ni < noodle_count_f; ni++) {
                            int nb_f  = noodle_start_f + ni * 12;
                            float du = ep_data[nb_f+0] - uf;
                            float dv = ep_data[nb_f+1] - vf;
                            float dx = ep_data[nb_f+4] - rd.x;
                            float dy = ep_data[nb_f+5] - rd.y;
                            float dz = ep_data[nb_f+6] - rd.z;
                            float d2 = du*du + dv*dv + dx*dx + dy*dy + dz*dz;
                            if (d2 < best_d2_f[KF-1]) {
                                best_d2_f[KF-1]  = d2;
                                best_idx_f[KF-1] = ni;
                                for (int j = KF-2; j >= 0; j--) {
                                    if (best_d2_f[j+1] < best_d2_f[j]) {
                                        float td = best_d2_f[j];   best_d2_f[j]  = best_d2_f[j+1]; best_d2_f[j+1] = td;
                                        int   ti = best_idx_f[j];  best_idx_f[j] = best_idx_f[j+1]; best_idx_f[j+1] = ti;
                                    }
                                }
                            }
                        }
                        vec3  blended_f = vec3(0.0);
                        float w_sum_f   = 0.0;
                        for (int ki = 0; ki < KF; ki++) {
                            if (best_idx_f[ki] < 0) continue;
                            float w_f  = 1.0 / max(best_d2_f[ki], 1e-12);
                            int   nb_f = noodle_start_f + best_idx_f[ki] * 12;
                            blended_f += w_f * vec3(ep_data[nb_f+7], ep_data[nb_f+8], ep_data[nb_f+9]);
                            w_sum_f   += w_f;
                        }
                        rd = normalize(blended_f / max(w_sum_f, 1e-12));
                        ro = hp + rd * 1e-5;
                        path_len += best;
                        packet.y += best * packet.x * 0.000021;
                        if (packet.z < 0.002) break;
                        continue;
                    }
                }
            }
        }

        if (opacity < 0.999) {
            // Refractive: bend the forward ray through the medium
            float eta      = interior_face ? ior : (1.0 / ior);
            float cos_t_sq = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
            if (cos_t_sq <= 0.0) {
                rd = normalize(reflect(rd, n));
                if (dot(rd, n) < 0.0) rd = cosine_dir(rand01(rng), rand01(rng), n);
            } else {
                float cos_t = sqrt(cos_t_sq);
                float rs = (cos_i - ior * cos_t) / max(cos_i + ior * cos_t, 1e-6);
                float rp = (ior * cos_i - cos_t) / max(ior * cos_i + cos_t, 1e-6);
                float fr  = clamp(0.5 * (rs*rs + rp*rp), 0.0, 1.0);
                float p_r = mix(fr, 1.0, opacity);
                if (rand01(rng) < p_r) {
                    rd = normalize(reflect(rd, n));
                    if (dot(rd, n) < 0.0) rd = cosine_dir(rand01(rng), rand01(rng), n);
                } else {
                    rd = normalize(refract(rd, n, eta));
                    packet.z *= (1.0 - fr);
                }
            }
        } else {
            // Opaque: diffuse or specular bounce
            if (rand01(rng) < diffusion) {
                rd = cosine_dir(rand01(rng), rand01(rng), n);
            } else {
                rd = normalize(reflect(rd, n));
                if (dot(rd, n) < 0.0) rd = cosine_dir(rand01(rng), rand01(rng), n);
            }
            float band_decay = exp(-absorption * best * clamp(packet.x / 440.0, 0.25, 8.0));
            float dist_falloff = 1.0 / (1.0 + 0.08 * best);
            packet.z *= reflectivity * band_decay * dist_falloff;
        }
        ro = hp + rd * 1e-5;
        path_len += best;
        packet.y += best * packet.x * 0.000021;
        if (packet.z < 0.002) break;
    }
}

// ─── Reactive re-emission pass (PASS_REACTIVE) ───────────────────────────────
// Each invocation processes one PendingRay enqueued by PASS_FORWARD.
// PRAY_NO_REACTIVE is always set, so no further pending rays are created —
// this is a single-generation secondary pass preventing cascade explosions.
void reactive_main(uint gid, inout uint rng) {
    uint count = pending_count;   // written by forward pass; read here
    if (gid >= count) return;

    PendingRay pr      = pending_rays[gid];
    uint  pray_flags   = floatBitsToUint(pr.origin_flags.w);
    vec3  ro           = pr.origin_flags.xyz;
    vec3  rd           = normalize(pr.dir_bounces.xyz);
    int   bounces_rem  = int(floatBitsToUint(pr.dir_bounces.w));
    vec4  packet       = pr.packet;

    for (int bounce = 0; bounce < min(bounces_rem, uMaxBounces); ++bounce) {
        float best;
        int hit = nearest_hit(ro, rd, best);
        if (hit < 0) break;

        vec3 hp = ro + rd * best;
        splat_segment_spectral(ro, hp, packet);

        Tri   tri            = tris[hit];
        uint  flags          = floatBitsToUint(tri.normal.w);
        vec3  geom_n         = normalize(tri.normal.xyz);
        bool  interior_face  = dot(rd, geom_n) > 0.0;
        vec4  mat            = interior_face ? tri.mat_in : tri.mat_out;
        float reflectivity   = clamp(mat.x, 0.02, 0.98);
        float diffusion      = clamp(mat.y, 0.0, 1.0);
        float absorption     = clamp(mat.z, 0.0, 2.0);
        float ior            = max(tri.mat_in.w, 1.0);
        float opacity        = clamp(tri.mat_out.w, 0.0, 1.0);
        vec3  n              = interior_face ? -geom_n : geom_n;
        float cos_i          = max(0.0, dot(-rd, n));

        // Emissive secondary hit: deposit; no further queuing
        if ((flags & MAT_FLAG_EMISSIVE) != 0u && tri.albedo.w > 0.001) {
            vec4 emit_pkt = vec4(packet.x, packet.y,
                                 tri.albedo.w * cos_i *
                                 max(tri.emissive.x, max(tri.emissive.y, tri.emissive.z)),
                                 packet.w);
            splat_spectral(hp, emit_pkt);
        }

        if ((flags & MAT_FLAG_ABSORBER) != 0u) break;

        // ── Transform LUT (manifold remission profile) ────────────────────
        // Any material can reference a PROFILE_MANIFOLD entry via remit_profile_idx.
        // The noodle is a trained LUT: 5D key (barycentric u,v at hit + incident rd)
        // maps to exit rd via IDW. Bidirectional: dot(rd, normal) selects side.
        int remit_idx = int(round(tri.emissive.y));
        if (remit_idx >= 0) {
            int pbase = remit_idx * 20;
            if (int(ep_data[pbase]) == PROFILE_MANIFOLD) {
                int noodle_start = int(ep_data[pbase + 1]);
                int noodle_count = int(ep_data[pbase + 2]);
                if (noodle_count > 0) {
                    // Barycentric (u,v) at the struck triangle — same coords the
                    // noodle was trained on. Recomputed cheaply from ro and rd.
                    vec2 buv = bary_uv(ro, rd, tri);
                    float u = buv.x;
                    float v = buv.y;
                    // 5D key: (u, v, rd.x, rd.y, rd.z) — brute-force k=8 nearest
                    const int K = 8;
                    int   best_idx[K];
                    float best_d2[K];
                    for (int ki = 0; ki < K; ki++) {
                        best_idx[ki] = -1;
                        best_d2[ki]  = 1e30;
                    }
                    for (int ni = 0; ni < noodle_count; ni++) {
                        int nb  = noodle_start + ni * 12;
                        float du = ep_data[nb+0] - u;
                        float dv = ep_data[nb+1] - v;
                        float dx = ep_data[nb+4] - rd.x;
                        float dy = ep_data[nb+5] - rd.y;
                        float dz = ep_data[nb+6] - rd.z;
                        float d2 = du*du + dv*dv + dx*dx + dy*dy + dz*dz;
                        if (d2 < best_d2[K-1]) {
                            best_d2[K-1]  = d2;
                            best_idx[K-1] = ni;
                            for (int j = K-2; j >= 0; j--) {
                                if (best_d2[j+1] < best_d2[j]) {
                                    float td = best_d2[j];   best_d2[j]  = best_d2[j+1]; best_d2[j+1]  = td;
                                    int   ti = best_idx[j];  best_idx[j] = best_idx[j+1]; best_idx[j+1] = ti;
                                }
                            }
                        }
                    }
                    // IDW blend out_dir (cols 7-9)
                    vec3  blended = vec3(0.0);
                    float w_sum   = 0.0;
                    for (int ki = 0; ki < K; ki++) {
                        if (best_idx[ki] < 0) continue;
                        float w  = 1.0 / max(best_d2[ki], 1e-12);
                        int   nb = noodle_start + best_idx[ki] * 12;
                        blended += w * vec3(ep_data[nb+7], ep_data[nb+8], ep_data[nb+9]);
                        w_sum   += w;
                    }
                    rd = normalize(blended / max(w_sum, 1e-12));
                    ro = hp + rd * 1e-5;
                    packet.y += best * packet.x * 0.000021;
                    if (packet.z < 0.002) break;
                    continue;
                }
            }
        }

        // Bounce (no reactive queuing from reactive pass — PRAY_NO_REACTIVE)
        if (opacity < 0.999) {
            float eta      = interior_face ? ior : (1.0 / ior);
            float cos_t_sq = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
            if (cos_t_sq <= 0.0) {
                rd = normalize(reflect(rd, n));
                if (dot(rd, n) < 0.0) rd = cosine_dir(rand01(rng), rand01(rng), n);
            } else {
                float cos_t = sqrt(cos_t_sq);
                float rs = (cos_i - ior * cos_t) / max(cos_i + ior * cos_t, 1e-6);
                float rp = (ior * cos_i - cos_t) / max(ior * cos_i + cos_t, 1e-6);
                float fr  = clamp(0.5 * (rs*rs + rp*rp), 0.0, 1.0);
                if (rand01(rng) < mix(fr, 1.0, opacity)) {
                    rd = normalize(reflect(rd, n));
                    if (dot(rd, n) < 0.0) rd = cosine_dir(rand01(rng), rand01(rng), n);
                } else {
                    rd = normalize(refract(rd, n, eta));
                    packet.z *= (1.0 - fr);
                }
            }
        } else {
            if (rand01(rng) < diffusion) {
                rd = cosine_dir(rand01(rng), rand01(rng), n);
            } else {
                rd = normalize(reflect(rd, n));
                if (dot(rd, n) < 0.0) rd = cosine_dir(rand01(rng), rand01(rng), n);
            }
            float band_decay   = exp(-absorption * best * clamp(packet.x / 440.0, 0.25, 8.0));
            float dist_falloff = 1.0 / (1.0 + 0.08 * best);
            packet.z *= reflectivity * band_decay * dist_falloff;
        }
        ro = hp + rd * 1e-5;
        packet.y += best * packet.x * 0.000021;
        if (packet.z < 0.002) break;
    }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 2-D camera sensor compute shader
# Fires one thread per pixel; each thread shoots uSamplesPerPixel primary rays
# through the pixel (stratified AA), traces the BVH, reads the already-complete
# forward irradiance bands via integer samplers, applies Lambert BRDF and
# accumulates an RGBA32F pixel radiance into uSensorOut.
# ─────────────────────────────────────────────────────────────────────────────
_GPU_SENSOR_CS = """
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

// ─── BVH geometry (identical 8×vec4 layout to the forward pass) ─────────────
struct Tri  { vec4 v0; vec4 e1; vec4 e2; vec4 normal; vec4 mat_in; vec4 mat_out; vec4 albedo; vec4 emissive; };
struct Node { vec4 lo_left; vec4 hi_right; vec4 start_count; };
layout(std430, binding = 0) readonly buffer TriBuf    { Tri   tris[];    };
layout(std430, binding = 1) readonly buffer NodeBuf   { Node  nodes[];   };
layout(std430, binding = 2) readonly buffer TriIdBuf  { int   tri_ids[]; };
// Bindings 3-4: reserved (previously used for per-pixel manifold ray dirs/OPLs).
layout(std430, binding = 3) readonly buffer RayDirBuf { float ray_dirs[]; };
layout(std430, binding = 4) readonly buffer RayOplBuf { float ray_opls[]; };

// ─── Forward film-layer activation fields (uint, same storage as _MARCH_FS) ──
layout(binding = 0) uniform usampler3D uFwdLayer0;
layout(binding = 1) uniform usampler3D uFwdLayer1;
layout(binding = 2) uniform usampler3D uFwdLayer2;
layout(binding = 3) uniform usampler3D uFwdLayer3;
layout(binding = 4) uniform usampler3D uFwdLayer4;
layout(binding = 5) uniform usampler3D uFwdLayer5;
layout(binding = 6) uniform usampler3D uFwdLayer6;
layout(binding = 7) uniform usampler3D uFwdLayer7;

// ─── Per-layer output textures (one per film layer, up to 8, retained in memory)
// Image bindings are a separate namespace from the sampler bindings above.
layout(rgba32f, binding = 0) uniform image2D uSensorOut0;
layout(rgba32f, binding = 1) uniform image2D uSensorOut1;
layout(rgba32f, binding = 2) uniform image2D uSensorOut2;
layout(rgba32f, binding = 3) uniform image2D uSensorOut3;
layout(rgba32f, binding = 4) uniform image2D uSensorOut4;
layout(rgba32f, binding = 5) uniform image2D uSensorOut5;
layout(rgba32f, binding = 6) uniform image2D uSensorOut6;
layout(rgba32f, binding = 7) uniform image2D uSensorOut7;

// ─── Uniforms ─────────────────────────────────────────────────────────────────
uniform ivec2 uSensorSize;
uniform int   uRowOffset;
uniform int   uRandomPixels;
uniform int   uDispatchPixelCount;
uniform int   uSamplesPerPixel;
uniform int   uMaxBounces;
uniform int   uTriCount;
uniform int   uNodeCount;
uniform int   uSeed;
uniform float uRayFieldScale;
uniform float uRayFieldGamma;
uniform float uVolAlpha;
uniform int   uVolSteps;
uniform int   uLayerCount;
uniform vec3  uLayerDark0;  uniform vec3  uLayerDark1;
uniform vec3  uLayerDark2;  uniform vec3  uLayerDark3;
uniform vec3  uLayerDark4;  uniform vec3  uLayerDark5;
uniform vec3  uLayerDark6;  uniform vec3  uLayerDark7;
uniform vec3  uLayerLight0; uniform vec3  uLayerLight1;
uniform vec3  uLayerLight2; uniform vec3  uLayerLight3;
uniform vec3  uLayerLight4; uniform vec3  uLayerLight5;
uniform vec3  uLayerLight6; uniform vec3  uLayerLight7;
uniform vec3  uCamEye;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform vec3  uCamFwd;
uniform float uCamFovTan;
uniform float uCamAspect;
uniform float uApertureRadius;
uniform float uFocusDist;
uniform float uCAFactor;
uniform vec2  uTiltShift;
uniform int   uNBlades;       // 0 = circle; >=3 = regular N-gon
uniform float uApertureRot;   // first blade edge angle, radians
uniform vec2  uLensTilt;      // Scheimpflug (x=nod, y=pan) radians — per-ray focus dist modulation
uniform int   uUseManifold;   // 0 = analytical thin-lens; 1 = read precomputed ray dirs from SSBO
uniform vec3  uBoxMin;
uniform vec3  uBoxMax;
uniform ivec3 uDims;
uniform float uAirDiffuseScatter;
uniform float uAirSpecularScatter;
uniform float uAirAnisotropy;
uniform float uMediumExtinction;

vec3 get_layer_dark(int i) {
    if      (i == 0) return uLayerDark0;  else if (i == 1) return uLayerDark1;
    else if (i == 2) return uLayerDark2;  else if (i == 3) return uLayerDark3;
    else if (i == 4) return uLayerDark4;  else if (i == 5) return uLayerDark5;
    else if (i == 6) return uLayerDark6;  else if (i == 7) return uLayerDark7;
    return vec3(0.0);
}
vec3 get_layer_light(int i) {
    if      (i == 0) return uLayerLight0; else if (i == 1) return uLayerLight1;
    else if (i == 2) return uLayerLight2; else if (i == 3) return uLayerLight3;
    else if (i == 4) return uLayerLight4; else if (i == 5) return uLayerLight5;
    else if (i == 6) return uLayerLight6; else if (i == 7) return uLayerLight7;
    return vec3(1.0);
}

const float EPS = 1e-7;
const float PI  = 3.14159265358979323846;

// ─── RNG ──────────────────────────────────────────────────────────────────────
uint hash_u(uint x) {
    x = x * 747796405u + 2891336453u;
    uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (word >> 22u) ^ word;
    x ^= x * 0x9e3779b9u;
    x ^= x >> 16;
    return x;
}
float rand01(inout uint s) {
    s = hash_u(s); return float(s & 0x00ffffffu) / 16777215.0;
}
float randn(inout uint s) {
    float u0 = max(rand01(s), 1e-7);
    float u1 = rand01(s);
    return sqrt(-2.0 * log(u0)) * cos(2.0 * PI * u1);
}

// ─── Cosine-weighted hemisphere sample ───────────────────────────────────────
vec3 cosine_dir(float u, float v, vec3 normal) {
    float r = sqrt(max(u, 0.0)); float a = 2.0 * PI * v;
    vec3 local = vec3(r * cos(a), r * sin(a), sqrt(max(0.0, 1.0 - u)));
    vec3 w = normalize(normal);
    vec3 wup = abs(w.z) < 0.9 ? vec3(0,0,1) : vec3(0,1,0);
    vec3 bx = normalize(cross(wup, w)); vec3 by = cross(w, bx);
    return normalize(bx*local.x + by*local.y + w*local.z);
}

// ─── Möller-Trumbore triangle intersection ────────────────────────────────────
bool hit_tri(vec3 ro, vec3 rd, Tri t, out float hit_t) {
    vec3 h = cross(rd, t.e2.xyz); float a = dot(t.e1.xyz, h);
    if (abs(a) < 1e-8) return false;
    float f = 1.0 / a; vec3 s = ro - t.v0.xyz;
    float u = f * dot(s, h);
    if (u < -1e-5 || u > 1.00001) return false;
    vec3 q = cross(s, t.e1.xyz); float v = f * dot(rd, q);
    if (v < -1e-5 || u + v > 1.00001) return false;
    float tt = f * dot(t.e2.xyz, q);
    if (tt <= 1e-6) return false;
    hit_t = tt; return true;
}

// ─── Slab / AABB test ─────────────────────────────────────────────────────────
bool slab_axis(float ro, float rd, float lo, float hi,
               inout float nt, inout float ft) {
    if (abs(rd) < 1e-9) return ro >= lo && ro <= hi;
    float inv = 1.0/rd; float a = (lo-ro)*inv; float b = (hi-ro)*inv;
    nt = max(nt, min(a,b)); ft = min(ft, max(a,b)); return nt <= ft;
}
bool hit_aabb(vec3 ro, vec3 rd, vec3 lo, vec3 hi, float best) {
    float nt = EPS, ft = best;
    if (!slab_axis(ro.x,rd.x,lo.x,hi.x,nt,ft)) return false;
    if (!slab_axis(ro.y,rd.y,lo.y,hi.y,nt,ft)) return false;
    if (!slab_axis(ro.z,rd.z,lo.z,hi.z,nt,ft)) return false;
    return nt <= ft;
}

// ─── BVH nearest-hit traversal ───────────────────────────────────────────────
int nearest_hit(vec3 ro, vec3 rd, out float best) {
    best = 1e30; int best_tri = -1;
    int stack[96]; int sp = 0; stack[sp++] = 0;
    while (sp > 0) {
        int ni = stack[--sp];
        if (ni < 0 || ni >= uNodeCount) continue;
        Node node = nodes[ni];
        if (!hit_aabb(ro, rd, node.lo_left.xyz, node.hi_right.xyz, best)) continue;
        int left  = int(node.lo_left.w);
        int right = int(node.hi_right.w);
        int start = int(node.start_count.x);
        int count = int(node.start_count.y);
        if (left < 0) {
            for (int k = 0; k < count; ++k) {
                int ti = tri_ids[start+k]; float ht;
                if (hit_tri(ro, rd, tris[ti], ht) && ht < best) {
                    best = ht; best_tri = ti;
                }
            }
        } else {
            if (sp < 94) { stack[sp++] = left; stack[sp++] = right; }
        }
    }
    return best_tri;
}

// ─── Read one film layer's accumulated irradiance from the forward volume ──────
float fwd_layer(int i, ivec3 q) {
    if      (i == 0) return float(texelFetch(uFwdLayer0, q, 0).r);
    else if (i == 1) return float(texelFetch(uFwdLayer1, q, 0).r);
    else if (i == 2) return float(texelFetch(uFwdLayer2, q, 0).r);
    else if (i == 3) return float(texelFetch(uFwdLayer3, q, 0).r);
    else if (i == 4) return float(texelFetch(uFwdLayer4, q, 0).r);
    else if (i == 5) return float(texelFetch(uFwdLayer5, q, 0).r);
    else if (i == 6) return float(texelFetch(uFwdLayer6, q, 0).r);
    else if (i == 7) return float(texelFetch(uFwdLayer7, q, 0).r);
    return 0.0;
}

// ─── Scalar irradiance for one layer at world pos p ───────────────────────────
float sample_layer_energy(vec3 p, int i) {
    vec3  uvw = (p - uBoxMin) / max(uBoxMax - uBoxMin, vec3(1e-6));
    ivec3 q   = ivec3(floor(uvw * vec3(uDims)));
    if (any(lessThan(q, ivec3(0))) || any(greaterThanEqual(q, uDims))) return 0.0;
    return fwd_layer(i, q) * (uRayFieldScale / 65535.0);
}

// ─── Total irradiance across all layers at p (for alpha compositing) ──────────
float sample_total_energy(vec3 p) {
    vec3  uvw = (p - uBoxMin) / max(uBoxMax - uBoxMin, vec3(1e-6));
    ivec3 q   = ivec3(floor(uvw * vec3(uDims)));
    if (any(lessThan(q, ivec3(0))) || any(greaterThanEqual(q, uDims))) return 0.0;
    float sc = uRayFieldScale / 65535.0;
    int n = clamp(uLayerCount, 1, 8);
    float tot = 0.0;
    for (int i = 0; i < n; i++) tot += fwd_layer(i, q) * sc;
    return tot;
}

// ─── Volume march: one alpha pass shared across layers; per-layer scalar out ───
void march_volume_layers(vec3 ro, vec3 rd, float t_near, float t_far,
                         int steps, float jitter, int n,
                         out float out_lum[8], out float out_alpha) {
    for (int i = 0; i < 8; i++) out_lum[i] = 0.0;
    out_alpha = 0.0;
    if (t_near >= t_far) return;
    float dt = (t_far - t_near) / float(steps);
    float sc = uRayFieldScale / 65535.0;
    // Air phase function: Henyey-Greenstein-like forward-scatter lobe blended
    // with isotropic.  uAirAnisotropy controls forward-lobe sharpness;
    // uAirDiffuseScatter / uAirSpecularScatter weight isotropic vs. forward.
    // uMediumExtinction is the Beer-Lambert attenuation over distance.
    // uVolAlpha is the master opacity dial — 0 = invisible, 1 = full density.
    vec3 box_size = max(uBoxMax - uBoxMin, vec3(1e-6));
    float diag = length(box_size);
    for (int step = 0; step < steps; ++step) {
        float t_step = t_near + (float(step) + jitter) * dt;
        vec3  pos = ro + t_step * rd;
        vec3  uvw = (pos - uBoxMin) / box_size;
        ivec3 q   = ivec3(floor(uvw * vec3(uDims)));
        if (any(lessThan(q, ivec3(0))) || any(greaterThanEqual(q, uDims))) continue;
        float tot = 0.0;
        float bv[8];
        for (int i = 0; i < n; i++) { bv[i] = fwd_layer(i, q) * sc; tot += bv[i]; }
        float mag = pow(clamp(tot, 0.0, 1.0), max(uRayFieldGamma, 0.05));
        if (mag > 0.0001) {
            // Beer-Lambert transmittance from ray origin to this step
            float beer = exp(-uMediumExtinction * t_step);
            // Phase: how much of the stored energy density is visible from
            // this view direction.  Forward-scattered energy concentrates
            // along the propagation axis; diffuse spreads isotropically.
            // We have no per-voxel flow direction, so use a simplified
            // anisotropy that weights magnitude: forward lobes get phase>1,
            // sideways views get phase≈uAirDiffuseScatter.
            float phase = uAirDiffuseScatter
                        + uAirSpecularScatter
                          * pow(mag, 1.0 / max(uAirAnisotropy, 0.5));
            // Differential opacity: energy density × phase × master dial × dt
            float sa = uVolAlpha * phase * dt / max(diag, 1e-4);
            float a  = sa * smoothstep(0.0001, 0.08, mag) * beer;
            float transmit = 1.0 - out_alpha;
            for (int i = 0; i < n; i++) out_lum[i] += transmit * a * bv[i];
            out_alpha += transmit * a;
            if (out_alpha > 0.97) break;
        }
    }
}

// ─── Write scalar luminance into layer i's output image ───────────────────────
void sensor_store(int i, ivec2 px, float lum, float cnt) {
    // alpha is a presence mask: any ray strike sets it to 1.0 so decayed
    // weight is immediately replaced and re-exposure is instant.
    if      (i==0){vec4 p=imageLoad(uSensorOut0,px); imageStore(uSensorOut0,px,vec4(p.rgb+lum, 1.0));}
    else if (i==1){vec4 p=imageLoad(uSensorOut1,px); imageStore(uSensorOut1,px,vec4(p.rgb+lum, 1.0));}
    else if (i==2){vec4 p=imageLoad(uSensorOut2,px); imageStore(uSensorOut2,px,vec4(p.rgb+lum, 1.0));}
    else if (i==3){vec4 p=imageLoad(uSensorOut3,px); imageStore(uSensorOut3,px,vec4(p.rgb+lum, 1.0));}
    else if (i==4){vec4 p=imageLoad(uSensorOut4,px); imageStore(uSensorOut4,px,vec4(p.rgb+lum, 1.0));}
    else if (i==5){vec4 p=imageLoad(uSensorOut5,px); imageStore(uSensorOut5,px,vec4(p.rgb+lum, 1.0));}
    else if (i==6){vec4 p=imageLoad(uSensorOut6,px); imageStore(uSensorOut6,px,vec4(p.rgb+lum, 1.0));}
    else if (i==7){vec4 p=imageLoad(uSensorOut7,px); imageStore(uSensorOut7,px,vec4(p.rgb+lum, 1.0));}
}

// ─── AABB entry/exit along ray (returns false if no intersection) ─────────────
bool box_intersect(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax,
                   out float t_in, out float t_out) {
    vec3 inv = 1.0 / rd;
    vec3 t0  = (bmin - ro) * inv;
    vec3 t1  = (bmax - ro) * inv;
    vec3 tmi = min(t0, t1), tma = max(t0, t1);
    t_in  = max(max(tmi.x, tmi.y), tmi.z);
    t_out = min(min(tma.x, tma.y), tma.z);
    t_in  = max(t_in, 0.0);
    return t_out > t_in;
}

void main() {
    ivec2 tile_xy = ivec2(gl_GlobalInvocationID.xy);
    uint dispatch_w = uint(gl_NumWorkGroups.x) * uint(gl_WorkGroupSize.x);
    uint linear_id  = uint(gl_GlobalInvocationID.x) + uint(gl_GlobalInvocationID.y) * dispatch_w;
    ivec2 px;
    if (uRandomPixels != 0) {
        if (linear_id >= uint(max(0, uDispatchPixelCount))) return;
        uint s0 = hash_u(linear_id ^ uint(uSeed));
        uint s1 = hash_u(s0 + 0x9e3779b9u);
        uint s2 = hash_u(s1 + 0x85ebca6bu);
        uint s3 = hash_u(s2 + 0xc2b2ae35u);
        float u0 = max(float(s0 & 0x00ffffffu) / 16777215.0, 1e-6);
        float u1 = float(s1 & 0x00ffffffu) / 16777215.0;
        float u2 = max(float(s2 & 0x00ffffffu) / 16777215.0, 1e-6);
        float u3 = float(s3 & 0x00ffffffu) / 16777215.0;
        float r0 = sqrt(-2.0 * log(u0));
        float r1 = sqrt(-2.0 * log(u2));
        float gx = 0.5 + 0.22 * r0 * cos(2.0 * PI * u1);
        float gy = 0.5 + 0.22 * r1 * sin(2.0 * PI * u3);
        if (rand01(s3) < 0.18) {
            gx = rand01(s3);
            gy = rand01(s3);
        }
        px = ivec2(int(clamp(gx, 0.0, 0.999999) * float(uSensorSize.x)),
                   int(clamp(gy, 0.0, 0.999999) * float(uSensorSize.y)));
    } else {
        px = ivec2(tile_xy.x, tile_xy.y + uRowOffset);
    }
    if (any(greaterThanEqual(px, uSensorSize))) return;

    float W  = float(uSensorSize.x);
    float H  = float(uSensorSize.y);
    int   nl = clamp(uLayerCount, 1, 8);
    float layer_acc[8];
    for (int i = 0; i < 8; i++) layer_acc[i] = 0.0;

    for (int s = 0; s < uSamplesPerPixel; ++s) {
        uint rng = hash_u(
            (uint(px.x) + uint(px.y) * uint(uSensorSize.x)) * 104729u
            + uint(s) * 1013u + uint(uSeed) + linear_id * 9176u);

        float pu  = (float(px.x) + rand01(rng)) / W;
        float pv  = (float(px.y) + rand01(rng)) / H;
        vec3 ro   = uCamEye;
        vec3 rd;
        if (uUseManifold != 0) {
            // Manifold path: precomputed world ray direction for this pixel.
            // The CPU batch-interpolated all pixels; we just index into the result.
            int px_idx = px.x + px.y * uSensorSize.x;
            rd = normalize(vec3(ray_dirs[px_idx * 3],
                                ray_dirs[px_idx * 3 + 1],
                                ray_dirs[px_idx * 3 + 2]));
        } else {
            // Analytical thin-lens path.
            float fu  = ((pu - 0.5) * 2.0 + uTiltShift.x) * uCamFovTan * uCamAspect;
            float fv  = ((0.5 - pv) * 2.0 + uTiltShift.y) * uCamFovTan;
            vec3 rd0  = normalize(uCamFwd + uCamRight * fu + uCamUp * fv);
            // Scheimpflug: per-ray effective focus distance.
            float _lt_denom = 1.0
                            - fu * tan(uLensTilt.y)
                            - fv * tan(uLensTilt.x);
            float fd_eff  = uFocusDist / max(0.005, _lt_denom);
            vec3 focus_pt = ro + rd0 * max(fd_eff, 0.01);
            float ap = max(uApertureRadius, 0.0);
            if (ap > 1e-7) {
                vec2 apt_off;
                int nb = uNBlades;
                if (nb < 3) {
                    float lr = ap * sqrt(rand01(rng));
                    float la = 2.0 * PI * rand01(rng);
                    apt_off = vec2(cos(la), sin(la)) * lr;
                } else {
                    float sector = floor(rand01(rng) * float(nb));
                    float inv_n  = 2.0 * PI / float(nb);
                    float a0 = sector * inv_n + uApertureRot;
                    float a1 = a0 + inv_n;
                    float u = rand01(rng);
                    float v = rand01(rng);
                    if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
                    vec2 p1 = ap * vec2(cos(a0), sin(a0));
                    vec2 p2 = ap * vec2(cos(a1), sin(a1));
                    apt_off = u * p1 + v * p2;
                }
                ro += uCamRight * apt_off.x + uCamUp * apt_off.y;
            }
            rd = normalize(focus_pt - ro);
        }
        float lum[8];
        for (int i = 0; i < 8; i++) lum[i] = 0.0;
        float throughput = 1.0;

        for (int bounce = 0; bounce <= uMaxBounces; ++bounce) {
            float surf_t;
            int   hit_id = nearest_hit(ro, rd, surf_t);

            float vol_in, vol_out;
            bool in_box = box_intersect(ro, rd, uBoxMin, uBoxMax, vol_in, vol_out);
            if (in_box) {
                float march_end = (hit_id >= 0) ? min(surf_t, vol_out) : vol_out;
                float step_jitter = fract(
                    sin(float(px.x)*127.1 + float(px.y)*311.7 + float(s)*7.3) * 43758.5453);
                float vol_lum[8]; float vol_alpha;
                march_volume_layers(ro, rd, vol_in, march_end, uVolSteps,
                                    step_jitter, nl, vol_lum, vol_alpha);
                for (int i = 0; i < nl; i++) lum[i] += throughput * vol_lum[i];
                throughput *= (1.0 - vol_alpha);
            }

            if (hit_id < 0 || throughput < 0.003) break;

            vec3  hp       = ro + rd * surf_t;
            Tri   tri      = tris[hit_id];
            vec3  geom_n   = normalize(tri.normal.xyz);
            bool  interior = dot(rd, geom_n) > 0.0;
            vec3  n        = interior ? -geom_n : geom_n;
            vec4  mat      = interior ? tri.mat_in : tri.mat_out;
            float refl     = clamp(mat.x, 0.02, 0.98);
            float diff     = clamp(mat.y, 0.0, 1.0);
            float abso     = clamp(mat.z, 0.0, 2.0);
            float ior      = max(tri.mat_in.w, 1.0);
            float opacity  = clamp(tri.mat_out.w, 0.0, 1.0);
            float cos_in   = max(0.0, dot(-rd, n));
            vec3  surf_albedo = tri.albedo.rgb;

            // Accumulate field energy at surface — only opaque/semi-opaque surfaces scatter
            // Transparent surfaces (opacity≈0) transmit, not scatter; skip accumulation.
            if (opacity > 0.001) {
                for (int i = 0; i < nl; i++) {
                    vec3  lc = get_layer_light(i);
                    float aw = dot(surf_albedo, lc) / max(dot(lc, lc), 1e-6);
                    lum[i] += throughput * sample_layer_energy(hp, i) * refl * cos_in
                              * clamp(aw, 0.0, 1.0) * opacity;
                }
            }

            if (opacity < 0.999) {
                // ── Transparent / refractive surface ─────────────────────────
                float eta = interior ? ior : (1.0 / ior);
                float cos_i = cos_in;
                float cos_t_sq = 1.0 - eta * eta * (1.0 - cos_i * cos_i);

                float fresnel_r;
                vec3  refr_dir;
                if (cos_t_sq <= 0.0) {
                    fresnel_r = 1.0;  // total internal reflection
                    refr_dir  = rd;   // unused
                } else {
                    float cos_t = sqrt(cos_t_sq);
                    float rs = (cos_i - ior * cos_t) / max(cos_i + ior * cos_t, 1e-6);
                    float rp = (ior * cos_i - cos_t) / max(ior * cos_i + cos_t, 1e-6);
                    fresnel_r = clamp(0.5 * (rs * rs + rp * rp), 0.0, 1.0);
                    refr_dir  = normalize(refract(rd, n, eta));
                }

                float p_reflect = mix(fresnel_r, 1.0, opacity);
                if (rand01(rng) < p_reflect) {
                    rd = normalize(reflect(rd, n));
                    if (dot(rd, n) < 0.0) rd = cosine_dir(rand01(rng), rand01(rng), n);
                } else {
                    rd = refr_dir;
                }
                ro = hp + rd * 1e-5;
            } else {
                // ── Opaque surface ────────────────────────────────────────────
                float survival = max(refl * (1.0 - abso * 0.5), 0.05);
                if (rand01(rng) > survival) break;
                throughput *= refl / survival;
                if (throughput < 0.005) break;

                if (rand01(rng) < diff)
                    rd = cosine_dir(rand01(rng), rand01(rng), n);
                else {
                    rd = normalize(reflect(rd, n));
                    if (dot(rd, n) < 0.0) rd = cosine_dir(rand01(rng), rand01(rng), n);
                }
                ro = hp + rd * 1e-5;
            }
        }
        for (int i = 0; i < nl; i++) layer_acc[i] += max(lum[i], 0.0);
    }

    for (int i = 0; i < nl; i++)
        sensor_store(i, px, layer_acc[i], float(uSamplesPerPixel));
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Sensor image fullscreen blit shaders
# Reads the RGBA32F sensor texture, divides by sample count (.a), tonemaps.
# ─────────────────────────────────────────────────────────────────────────────
_SENSOR_BLIT_VS = """
#version 330 core
out vec2 vUV;
void main() {
    // Full-screen triangle trick: gl_VertexID 0,1,2 covers [-1,3] x [-1,3]
    vUV         = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
"""
# ─────────────────────────────────────────────────────────────────────────────
# Sensor accumulator decay compute shader
# Multiplies every texel (rgb AND alpha) by uDecayFactor each frame.
# Dispatched once per display frame when half-life decay is enabled.
# Running both channels through the same factor keeps the stored mean
# (rgb/alpha) valid while shrinking total accumulated weight — when
# combined with the _decay_total multiplier in the blit shader, the
# display fades at the user-specified half-life rate during silence,
# and new source samples quickly displace the decayed state on restart.
# ─────────────────────────────────────────────────────────────────────────────
_GPU_DECAY_CS = """
#version 430 core
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(rgba32f, binding = 0) uniform image2D uAccum;
uniform float uDecayFactor;   // per-frame multiplier: exp(-ln2 * dt / half_life)

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sz    = imageSize(uAccum);
    if (coord.x >= sz.x || coord.y >= sz.y) return;
    vec4 v = imageLoad(uAccum, coord);
    v *= uDecayFactor;   // decay rgb and alpha together — ratio preserved, weights shrink
    imageStore(uAccum, coord, v);
}
"""


_SENSOR_BLIT_FS = """
#version 330 core
in  vec2 vUV;
out vec4 FragColor;

uniform sampler2D uSensorLayer0;
uniform sampler2D uSensorLayer1;
uniform sampler2D uSensorLayer2;
uniform sampler2D uSensorLayer3;
uniform sampler2D uSensorLayer4;
uniform sampler2D uSensorLayer5;
uniform sampler2D uSensorLayer6;
uniform sampler2D uSensorLayer7;

uniform int   uLayerCount;
uniform vec3  uLayerDark0;  uniform vec3  uLayerDark1;
uniform vec3  uLayerDark2;  uniform vec3  uLayerDark3;
uniform vec3  uLayerDark4;  uniform vec3  uLayerDark5;
uniform vec3  uLayerDark6;  uniform vec3  uLayerDark7;
uniform vec3  uLayerLight0; uniform vec3  uLayerLight1;
uniform vec3  uLayerLight2; uniform vec3  uLayerLight3;
uniform vec3  uLayerLight4; uniform vec3  uLayerLight5;
uniform vec3  uLayerLight6; uniform vec3  uLayerLight7;
uniform vec2  uLayerTone0;  uniform vec2  uLayerTone1;
uniform vec2  uLayerTone2;  uniform vec2  uLayerTone3;
uniform vec2  uLayerTone4;  uniform vec2  uLayerTone5;
uniform vec2  uLayerTone6;  uniform vec2  uLayerTone7;

uniform float uExposure;
uniform float uGamma;
uniform float uAlpha;
uniform float uDecayTotal;
uniform int   uRotate180;
uniform int   uNegative;
uniform int   uDigitalPositive;  // 1 = digital positive sensor mode
uniform vec3  uDigitalRGB;       // CFA white-balance weights (neutral = 1,1,1)

vec4 sample_layer(int i, vec2 uv) {
    if (i == 0) return texture(uSensorLayer0, uv);
    if (i == 1) return texture(uSensorLayer1, uv);
    if (i == 2) return texture(uSensorLayer2, uv);
    if (i == 3) return texture(uSensorLayer3, uv);
    if (i == 4) return texture(uSensorLayer4, uv);
    if (i == 5) return texture(uSensorLayer5, uv);
    if (i == 6) return texture(uSensorLayer6, uv);
    return texture(uSensorLayer7, uv);
}
vec3 layer_dark(int i) {
    if (i == 0) return uLayerDark0;
    if (i == 1) return uLayerDark1;
    if (i == 2) return uLayerDark2;
    if (i == 3) return uLayerDark3;
    if (i == 4) return uLayerDark4;
    if (i == 5) return uLayerDark5;
    if (i == 6) return uLayerDark6;
    return uLayerDark7;
}
vec3 layer_light(int i) {
    if (i == 0) return uLayerLight0;
    if (i == 1) return uLayerLight1;
    if (i == 2) return uLayerLight2;
    if (i == 3) return uLayerLight3;
    if (i == 4) return uLayerLight4;
    if (i == 5) return uLayerLight5;
    if (i == 6) return uLayerLight6;
    return uLayerLight7;
}
vec2 layer_tone(int i) {
    if (i == 0) return uLayerTone0;
    if (i == 1) return uLayerTone1;
    if (i == 2) return uLayerTone2;
    if (i == 3) return uLayerTone3;
    if (i == 4) return uLayerTone4;
    if (i == 5) return uLayerTone5;
    if (i == 6) return uLayerTone6;
    return uLayerTone7;
}

void main() {
    // UV orientation:
    //   digital positive — full 180° rotation undoes aperture optical inversion
    //   rotate180 — full 180° rotation (both axes)
    //   default — raw, no flip
    vec2 uv;
    if (uDigitalPositive != 0) {
        uv = vec2(1.0 - vUV.x, 1.0 - vUV.y);
    } else if (uRotate180 != 0) {
        uv = vec2(1.0 - vUV.x, 1.0 - vUV.y);
    } else {
        uv = vUV;
    }
    int  nl = clamp(uLayerCount, 1, 8);

    vec3 col = vec3(0.0);
    for (int i = 0; i < nl; i++) {
        vec4  raw    = sample_layer(i, uv);
        float n_samp = max(1.0, raw.a);
        float lum    = raw.r / n_samp * uExposure * uDecayTotal;
        float mapped = lum / (1.0 + lum);
        float gc     = pow(max(mapped, 0.0), 1.0 / uGamma);
        if (gc < 0.001) continue;
        vec2  tone = layer_tone(i);
        float sp   = tone.x;
        float hp   = tone.y;
        vec3  dark = layer_dark(i);
        vec3  lit  = layer_light(i);
        vec3  duotone;
        if (gc < sp) {
            // Ramp from true black up to shadow colour
            duotone = dark * (gc / max(sp, 0.001));
        } else if (gc < hp) {
            // Core duotone range — shadow colour to highlight colour
            duotone = mix(dark, lit, (gc - sp) / max(hp - sp, 0.001));
        } else {
            // Wash highlight colour out to true white
            duotone = mix(lit, vec3(1.0), (gc - hp) / max(1.0 - hp, 0.001));
        }
        col += duotone;
    }

    col = clamp(col, 0.0, 1.0);
    if (uNegative != 0) col = vec3(1.0) - col;
    // Digital positive: apply CFA spectral white-balance tint.
    // uDigitalRGB is (R_w, G_w, B_w) normalised so peak == 1.0.
    // For neutral output pass (1,1,1); for D65-balanced Bayer RGGB
    // the renderer supplies the computed weights from DigitalPositiveSensor.
    if (uDigitalPositive != 0) col = clamp(col * uDigitalRGB, 0.0, 1.0);
    FragColor = vec4(col, uAlpha);
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
out vec4 FragColor;
void main() {
    FragColor = vec4(texture(uTex, vUV).rgb, 1.0);
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


def _geometry_soundboard_sources(
    outline: np.ndarray,
    body_h: float,
    total_rays: int,
    stride: int = 6,
) -> list:
    """
    Soundboard emission sources derived purely from body geometry — no physics
    frame required.  Samples the interior of the guitar outline on a uniform
    grid and emits downward with source spectra weighted by the BRDF emission
    characteristic of a Sitka spruce top.

    BRDF emission spectrum per band (4 spectral bands):
      The soundboard is modelled as a diffuse emitter whose output is the
      product of assumed internal energy distribution and band-dependent
      transmission through the wood.  Spruce has ~20-30% higher stiffness
      along the grain than across it, giving stronger radiation in the low
      and lower-mid bands.  The per-band emission weights here encode that
      directional acoustic transmission:
        Band 0 (<220 Hz):   0.44  — strong fundamental, low absorption
        Band 1 (220-880):   0.32  — mid-range, moderate absorption
        Band 2 (880-3500):  0.17  — upper-mid, noticeable wood absorption
        Band 3 (>3500 Hz):  0.07  — high freq, heavily absorbed by wood grain
    """
    try:
        from matplotlib.path import Path as _MplPath  # type: ignore
        _have_mpl = True
    except ImportError:
        _have_mpl = False

    # Spruce top T1 resonance ~220 Hz, broad 2.5-oct spread covers 28–1700 Hz.
    # Packet layout: (freq_hz, phase, energy, coherence_oct)
    _spruce_packet = np.array([220.0, 0.0, 1.0, 2.5], np.float32)

    min_xy = outline.min(axis=0).astype(np.float32)
    max_xy = outline.max(axis=0).astype(np.float32)
    span_x = float(max_xy[0] - min_xy[0])
    span_y = float(max_xy[1] - min_xy[1])
    _z  = float(body_h) - 0.001
    _dn = np.array([0.0, 0.0, -1.0], np.float32)

    # Random (non-grid) sampling inside the outline so emission centres are
    # spatially fluid rather than a visible rectangular lattice.
    rng = np.random.default_rng(42)
    n_target  = max(32, int(span_x * span_y * 10000.0 / max(stride, 1) ** 2))
    n_attempt = n_target * 6
    candidates = rng.uniform(
        [float(min_xy[0]), float(min_xy[1])],
        [float(max_xy[0]), float(max_xy[1])],
        size=(n_attempt, 2),
    ).astype(np.float64)

    if _have_mpl:
        try:
            poly = _MplPath(outline.astype(np.float64))
            inside_mask = poly.contains_points(candidates)
            pts_inside  = candidates[inside_mask]
        except Exception:
            pts_inside = candidates
    else:
        pts_inside = candidates

    if len(pts_inside) == 0:
        return []
    if len(pts_inside) > n_target:
        idx = rng.choice(len(pts_inside), n_target, replace=False)
        pts_inside = pts_inside[idx]

    pts = [np.array([float(p[0]), float(p[1]), _z], np.float32) for p in pts_inside]
    if not pts:
        return []

    rays_each = max(1, total_rays // len(pts))
    remainder = total_rays - rays_each * len(pts)
    sources = [(p, _dn.copy(), rays_each, _spruce_packet.copy()) for p in pts]
    if remainder > 0:
        p, d, n, s = sources[0]
        sources[0] = (p, d, n + remainder, s)
    return sources


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
    fundamentals=STRING_FUNDAMENTALS_HZ,
    active_strings: Optional[set[int]] = None,
) -> list:
    """
    Emit from each string's physical 3-D positions at the string's own
    fundamental frequency.  Packet: (freq_hz, phase=0, energy=1, coherence=2.0 oct).
    Direction: straight down into the cavity.
    """
    sources = []
    _dn = np.array([0.0, 0.0, -1.0], np.float32)
    for si, path in enumerate(str_paths):
        if active_strings is not None and si not in active_strings:
            continue
        if si >= len(fundamentals):
            break
        spec = np.array([float(fundamentals[si]), 0.0, 1.0, 2.0], np.float32)
        path_arr = np.asarray(path, np.float32)
        n_pts = max(1, len(path_arr))
        rays_per_pt = max(1, total_rays_per_string // n_pts)
        for pt in path_arr:
            # Emit from the string's world position downward into the cavity
            pos = pt.copy()
            sources.append((pos, _dn.copy(), rays_per_pt, spec.copy()))
    return sources


def _instant_string_sources(
    str_paths_current: list,
    str_paths_equilibrium: list,
    total_rays_per_string: int = 4096,
    fundamentals=STRING_FUNDAMENTALS_HZ,
    active_strings=None,
    tube_radius: float = 0.003,
) -> list:
    """Emit from each string's *actual displaced* positions this frame.

    Unlike the static ``_string_emission_sources`` (which emits from every path
    point equally), this function:

    1. Computes per-point transverse displacement (deviation from the straight
       chord connecting the string endpoints).
    2. Importance-samples emission positions proportional to |displacement|, so
       emission concentrates at vibration anti-nodes and is absent at nodes.
    3. Derives per-source spectral content from the spatial mode decomposition
       (spatial FFT of the displacement profile), mapping each mode number to
       its harmonic frequency to assign the correct frequency band.
    4. Offsets each emitter a small random distance perpendicular to the string
       axis so emission is distributed across the string tube, not on the
       geometric centreline.

    Strings with sub-threshold peak displacement (|disp| < 1e-12 m) are
    silently skipped — they are not vibrating and should not emit.
    """
    sources = []
    seed = int(abs(time.monotonic_ns()) & 0xFFFF_FFFF)
    rng = np.random.default_rng(seed)

    for si, (path_now, path_eq) in enumerate(
            zip(str_paths_current, str_paths_equilibrium)):
        if active_strings is not None and si not in active_strings:
            continue
        if si >= len(fundamentals):
            break

        path_now = np.asarray(path_now, np.float32)
        path_eq  = np.asarray(path_eq,  np.float32)
        N = min(len(path_now), len(path_eq))
        if N < 4:
            continue
        path_now = path_now[:N]
        path_eq  = path_eq[:N]

        # Transverse displacement = deviation from chord (endpoint–endpoint line)
        t    = np.linspace(0.0, 1.0, N, dtype=np.float32)[:, None]
        chord = path_now[0:1] * (1.0 - t) + path_now[-1:] * t
        delta    = path_now - chord          # (N, 3) — purely transverse
        disp_mag = np.linalg.norm(delta, axis=1).astype(np.float64)  # (N,)

        peak = float(disp_mag.max())
        if peak < 1e-12:
            continue   # string is silent — no emission this frame

        disp_mag[disp_mag < peak * 0.01] = 0.0  # threshold at 1% of peak
        total_disp = float(disp_mag.sum())
        if total_disp < 1e-15:
            continue

        # ── Spectral content via spatial mode decomposition ───────────────────
        # The spatial FFT of the displacement profile gives modal power.
        # Mode k has spatial frequency k/(2*L) → acoustic frequency = k * f0.
        # Compute energy-weighted mean frequency and octave spread directly.
        fft_amp    = np.fft.rfft(disp_mag)
        mode_power = (np.abs(fft_amp) ** 2).astype(np.float64)
        f0         = float(fundamentals[si]) if si < len(fundamentals) else 110.0
        mode_freqs = np.array([(k + 1) * f0 for k in range(len(mode_power))], np.float64)
        total_mp   = float(mode_power.sum())
        if total_mp > 1e-12:
            mean_freq  = float(np.sum(mode_freqs * mode_power) / total_mp)
            log_modes  = np.log2(np.maximum(mode_freqs / max(f0, 1.0), 1.001))
            mean_log   = float(np.sum(log_modes * mode_power) / total_mp)
            var_log    = float(np.sum((log_modes - mean_log) ** 2 * mode_power) / total_mp)
            spread_oct = float(np.sqrt(max(var_log, 0.04)))
        else:
            mean_freq  = f0
            spread_oct = 2.0

        # ── Importance sampling: positions proportional to |displacement| ─────
        probs    = disp_mag / total_disp
        n_pts    = min(max(8, total_rays_per_string // 64), N)
        flat_idx = rng.choice(N, size=n_pts, p=probs)
        w_arr    = probs[flat_idx].astype(np.float32)
        w_sum    = float(w_arr.sum())
        if w_sum < 1e-12:
            continue
        w_arr  /= w_sum
        n_each  = np.maximum(1, (w_arr * total_rays_per_string).astype(np.int32))

        # String local tangent for cross-section spread
        tangent  = np.gradient(path_eq, axis=0).astype(np.float32)
        t_norms  = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent /= np.maximum(t_norms, 1e-9)

        spec_f32 = np.array([mean_freq, 0.0, 1.0, spread_oct], np.float32)
        dn       = np.array([0.0, 0.0, -1.0], np.float32)

        for k, i in enumerate(flat_idx):
            pos  = path_now[i].copy()
            tang = tangent[min(i, len(tangent) - 1)]
            # Perpendicular offset within tube cross-section
            perp = rng.standard_normal(3).astype(np.float32)
            perp -= np.dot(perp, tang) * tang
            pn = float(np.linalg.norm(perp))
            if pn > 1e-9:
                r   = float(rng.uniform(0.0, tube_radius))
                pos = pos + (perp / pn) * r
            sources.append((pos, dn.copy(), int(n_each[k]), spec_f32.copy()))

    return sources


def _instant_soundboard_sources(
    disp: np.ndarray,
    plate_active: np.ndarray,
    outline: np.ndarray,
    body_h: float,
    total_rays: int,
    stride: int = 4,
) -> list:
    """Current-frame plate sources — importance-sampled by displacement magnitude.

    Positions are drawn proportional to |displacement| so emission naturally
    clusters at vibration antinodes and is sparse at nodes.  Per-sample jitter
    within each grid cell gives a fluid, continuous spatial distribution instead
    of a regular rectangular grid of emission centres.
    """
    Nx, Ny = disp.shape
    min_xy = outline.min(axis=0).astype(np.float32)
    max_xy = outline.max(axis=0).astype(np.float32)
    xs = np.linspace(float(min_xy[0]), float(max_xy[0]), Nx, dtype=np.float32)
    ys = np.linspace(float(min_xy[1]), float(max_xy[1]), Ny, dtype=np.float32)
    dx_step = (float(max_xy[0]) - float(min_xy[0])) / max(Nx - 1, 1)
    dy_step = (float(max_xy[1]) - float(min_xy[1])) / max(Ny - 1, 1)

    energy = np.abs(disp).astype(np.float64)
    energy[~plate_active] = 0.0
    peak = float(energy.max())
    if peak <= 1e-12:
        return []
    energy[energy < peak * 0.01] = 0.0
    total_e = float(energy.sum())
    if total_e <= 0.0:
        return []

    # Spruce top T1 resonance ~220 Hz, 2.5-oct spread: (freq_hz, phase, energy, coherence_oct)
    spec = np.array([220.0, 0.0, 1.0, 2.5], np.float32)
    dn   = np.array([0.0, 0.0, -1.0], np.float32)
    _z   = float(body_h) - 0.001

    # Importance-sample positions proportional to |displacement| so the
    # emission density tracks the actual vibration mode shape.
    probs    = energy.ravel() / total_e
    n_sources = min(max(16, total_rays // 128), 8192)
    seed = int(peak * 1e9) & 0xFFFF_FFFF
    rng  = np.random.default_rng(seed)
    flat_idx = rng.choice(len(probs), size=n_sources, p=probs)
    ix_s = (flat_idx // Ny).astype(np.intp)
    iy_s = (flat_idx %  Ny).astype(np.intp)

    # Jitter each sample within its grid cell for a smooth, fluid appearance.
    jx = rng.uniform(-dx_step * 0.5, dx_step * 0.5, size=n_sources).astype(np.float32)
    jy = rng.uniform(-dy_step * 0.5, dy_step * 0.5, size=n_sources).astype(np.float32)
    pos_x = np.clip(xs[ix_s] + jx, float(min_xy[0]), float(max_xy[0]))
    pos_y = np.clip(ys[iy_s] + jy, float(min_xy[1]), float(max_xy[1]))

    # Rays proportional to sampled displacement weight.
    w_arr = probs[flat_idx].astype(np.float32)
    w_arr /= max(float(w_arr.sum()), 1e-12)
    n_each = np.maximum(1, (w_arr * int(total_rays)).astype(np.int32))
    order  = np.argsort(n_each)[::-1]
    return [
        (np.array([float(pos_x[k]), float(pos_y[k]), _z], np.float32),
         dn.copy(), int(n_each[k]), spec.copy())
        for k in order
    ]


def _stage_light_sources(total_rays: int, n_emitters: int = 9,
                         light_spec: "LightSpec | None" = None) -> list:
    """Diffuse area source above/front of the stage in world coordinates.

    Each emitter fires rays whose frequency is selected independently per ray
    by the forward CS: `freq = packet.x * exp2(randn() * packet.w)`.
    For white light set packet.x to the geometric mean of the visible spectrum
    (~548 nm → 5.48e14 Hz) and packet.w to 1.0 oct, which gives a log-uniform
    distribution spanning ~380–760 nm.  Integrated over many rays this is
    flat across the visible spectrum → white illumination.
    The R/G/B EM film layers then filter that distribution via their gaussian
    `film_activation()` response centred at optical band frequencies.

    If *light_spec* is supplied its spectral_array is used directly as the
    packet (freq_hz, phase, energy, coherence_oct) and n_emitters overrides.
    """
    # Geometric mean of visible range as placeholder center — actual per-ray
    # frequencies are drawn by _stochastic_spectral_packets via LightSpec.sample_freq_hz.
    _C_LIGHT = 2.998e8
    _WHITE_CENTER_HZ = float((_C_LIGHT / 380e-9 * _C_LIGHT / 750e-9) ** 0.5)  # ~5.48e14

    total_rays = max(1, int(total_rays))
    n_emitters = max(1, int(n_emitters))

    if light_spec is not None:
        n_emitters = max(1, light_spec.n_emitters)
        # Placeholder packet; actual freq drawn per-ray from light_spec.sample_freq_hz.
        packet = np.array([_WHITE_CENTER_HZ, 0.0, 1.0, 0.02], np.float32)
        lspec  = light_spec
    else:
        # Default: uniform white — sample_freq_hz will draw from log-uniform visible.
        packet = np.array([_WHITE_CENTER_HZ, 0.0, 1.0, 0.02], np.float32)
        lspec  = LightSpec()  # planck 3200 K by default

    cols = int(math.ceil(math.sqrt(n_emitters)))
    rows = int(math.ceil(n_emitters / cols))
    lx0, lx1 = -0.65, 0.65
    ly = STAGE_D_M * 0.28
    lz = STAGE_H_M * 0.82
    xs = np.linspace(lx0, lx1, cols, dtype=np.float32)
    zs = np.linspace(lz - 0.10, lz + 0.10, rows, dtype=np.float32)
    src = []
    axis = np.array([0.0, -0.42, -0.91], np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    _EMITTER_RADIUS = np.float32(0.12)

    # Build an explicit EmitterSpec asking for a bare Lambertian thermal body.
    # profile_name="thermal_3200K" is a tungsten-halogen Planckian blackbody
    # with DirectionalModel.LAMBERTIAN — no lens, no reflector, no aperture stop.
    from camera_designer.camera_preset   import EmitterSpec   as _EmitterSpec
    from camera_designer.emitter_profile import DirectionalModel as _DM

    # Integer codes must match the GLSL #define table in this file:
    #   DIRMODEL_LAMBERTIAN 0, DIRMODEL_DIPOLE_INPLANE 1, DIRMODEL_GAUSSIAN_BEAM 2,
    #   DIRMODEL_ETENDUE 3, DIRMODEL_PROJECTIVE 4, DIRMODEL_HG 5, DIRMODEL_ISOTROPIC 6
    _DIRMODEL_INT = {
        _DM.LAMBERTIAN:           0,
        _DM.DIPOLE_INPLANE_MIXED: 1,
        _DM.GAUSSIAN_BEAM:        2,
        _DM.ETENDUE_LIMITED:      3,
        _DM.PROJECTIVE:           4,
        _DM.HENYEY_GREENSTEIN:    5,
    }

    _stage_spec = _EmitterSpec(
        pos=(0.0, 0.0, 0.0),                 # overridden per-emitter below
        normal=(0.0, -0.42, -0.91),          # pointing direction (normalised below)
        radius=float(_EMITTER_RADIUS),
        profile_name="thermal_3200K",        # bare Lambertian blackbody, 3200 K Planckian
        label="stage_lamp",
    )
    _stage_profile   = _stage_spec.resolve_profile()
    _stage_dir_model = _stage_profile.directional.model  # must be LAMBERTIAN
    if _stage_dir_model != _DM.LAMBERTIAN:
        raise RuntimeError(
            f"_stage_light_sources: profile resolved to {_stage_dir_model!r}, "
            f"expected LAMBERTIAN.  Only bare thermal emission is permitted here."
        )
    _STAGE_MODEL_INT   = _DIRMODEL_INT[_stage_dir_model]   # 0 — LAMBERTIAN
    _STAGE_MODEL_PARAM = 0.0                                # unused for Lambertian
    rays_each = max(1, total_rays // n_emitters)
    count = 0
    for z in zs:
        for x in xs:
            if count >= n_emitters:
                break
            # Tuple layout: (pos, dir, n_rays, spectrum, radius, model_int, model_param, light_spec)
            src.append((np.array([float(x), float(ly), float(z)], np.float32),
                        axis.copy(), rays_each, packet.copy(),
                        _EMITTER_RADIUS, _STAGE_MODEL_INT, _STAGE_MODEL_PARAM, lspec))
            count += 1
    remainder = total_rays - rays_each * len(src)
    if remainder > 0 and src:
        p, d, n, s = src[0][0], src[0][1], src[0][2], src[0][3]
        src[0] = (p, d, n + remainder, s) + src[0][4:]
    return src

def _transform_sources(sources: list, matrix: np.ndarray) -> list:
    if not sources:
        return []
    M = np.asarray(matrix, np.float32)
    out = []
    for entry in sources:
        pos, direction, n_rays, spectrum = entry[0], entry[1], entry[2], entry[3]
        radius      = float(entry[4]) if len(entry) > 4 else 0.0
        # New tuple layout: [5]=model_int (int), [6]=model_param (float), [7]=light_spec
        # model_int and model_param are scale-invariant (angular model doesn't change
        # under rigid / uniform-scale transforms; PROJECTIVE half-angle is geometry-agnostic).
        model_int   = int(entry[5])   if len(entry) > 5 else 0
        model_param = float(entry[6]) if len(entry) > 6 else 0.0
        light_spec  = entry[7]        if len(entry) > 7 else None
        p = _transform_points(np.asarray(pos, np.float32).reshape(1, 3), M)[0]
        d = _transform_normals(np.asarray(direction, np.float32).reshape(1, 3), M)[0]
        scale = float(np.linalg.norm(M[:3, :3], ord='fro') / np.sqrt(3.0))
        out.append((p, d, int(n_rays), np.asarray(spectrum, np.float32).copy(),
                    radius * scale, model_int, model_param, light_spec))
    return out


def _stage_light_sources_guitar_frame(outline: np.ndarray,
                                      total_rays: int,
                                      n_emitters: int = 9) -> list:
    _, Minv = _guitar_model_matrix(outline)
    return _transform_sources(_stage_light_sources(total_rays, n_emitters), Minv)


def _sensor_scene_sources(outline: np.ndarray,
                          acoustic_sources: list,
                          *,
                          light_rays: int,
                          light_emitters: int) -> list:
    """One source list for the sensor scene: acoustic + EM/stage light."""
    sources = list(acoustic_sources or [])
    if int(light_rays) > 0:
        sources.extend(_stage_light_sources_guitar_frame(
            outline, int(light_rays), int(light_emitters)))
    return sources


def _stochastic_spectral_packets(sources: list,
                                 *,
                                 packets_per_source: int = 32,
                                 seed: int = 1337) -> list:
    """Expand sources into individually mono-spectral ray packets.

    Source tuple layout (indices 4-7 are optional):
      [0] pos          (3,) float32
      [1] dir          (3,) float32
      [2] n_rays       int
      [3] spectrum     (4,) float32  [freq_hz, phase, energy, coherence_oct]
      [4] radius       float         emitter disc radius in metres
      [5] model_int    int           DirectionalModel integer code:
                                       0=LAMBERTIAN  1=DIPOLE_INPLANE_MIXED
                                       2=GAUSSIAN_BEAM  3=ETENDUE_LIMITED
                                       4=PROJECTIVE  5=HENYEY_GREENSTEIN  6=ISOTROPIC
      [6] model_param  float         angular parameter for the model
                                       PROJECTIVE  → half_angle_rad
                                       GAUSSIAN_BEAM → divergence_half_angle_rad
                                       ETENDUE_LIMITED → cos(asin(NA))
                                       HENYEY_GREENSTEIN → hg_g
                                       all others → 0.0
      [7] light_spec   LightSpec|None  per-ray spectral sampling object

    Each source tuple may carry a ``LightSpec`` at index [7].  When present,
    every sub-ray calls ``light_spec.sample_freq_hz(rng)`` to draw its own
    independent frequency from the source's continuous spectrum — Planckian,
    emission lines, gaussian peaks, or log-uniform white, as defined in the
    YAML.  This is the only correct way to represent broadband light: every
    single ray is monochromatic at its own wavelength, and the ensemble
    integrates to the source spectrum.

    Sources without a LightSpec (acoustic strings, soundboard) use log-normal
    jitter around their physical freq_hz centre with sigma = coherence_oct.
    """
    if not sources:
        return []
    rng = np.random.default_rng(int(seed))
    out = []
    for _src in sources:
        pos, direction, n_rays, spectrum = _src[0], _src[1], _src[2], _src[3]
        n_rays     = max(1, int(n_rays))
        # light_spec now lives at index [7]; [4]=radius [5]=model_int [6]=model_param

        spec_arr    = np.asarray(spectrum, np.float64).ravel()
        energy      = float(spec_arr[2]) if spec_arr.size > 2 else 1.0
        coherence   = max(float(spec_arr[3]) if spec_arr.size > 3 else 0.5, 1e-3)

        light_spec = _src[7] if len(_src) > 7 else None
        n_packets = max(1, min(int(packets_per_source), n_rays))
        counts = rng.multinomial(n_rays, np.full(n_packets, 1.0 / n_packets))
        for count in counts:
            if count <= 0:
                continue
            if light_spec is not None:
                # Each ray independently samples the source's continuous spectrum.
                freq          = light_spec.sample_freq_hz(rng)
                pkt_coherence = 0.02   # monochromatic ray — narrow
            else:
                freq_center   = float(spec_arr[0]) if spec_arr.size > 0 else 440.0
                freq          = freq_center * (2.0 ** float(rng.normal(0.0, coherence)))
                pkt_coherence = coherence
            phase  = float(rng.uniform(-math.pi, math.pi))
            packet = np.array([freq, phase, energy, pkt_coherence], np.float32)
            # Pass radius, model_int, model_param (indices 4-6) through unchanged.
            # light_spec (index 7) is consumed above and not forwarded — the
            # expanded mono-spectral packets carry their own drawn frequency.
            out.append((np.asarray(pos, np.float32), np.asarray(direction, np.float32),
                        int(count), packet) + tuple(_src[4:7]))
    return out


def _stage_light_bounds() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([-STAGE_W_M * 0.55, -STAGE_D_M * 0.56, 0.0], np.float32),
        np.array([ STAGE_W_M * 0.55,  STAGE_D_M * 0.52, STAGE_H_M], np.float32),
    )


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    M = np.asarray(matrix, np.float32)
    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float32)])
    return (pts_h @ M.T)[:, :3].astype(np.float32)


def _transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    nrm = np.asarray(normals, np.float32).reshape(-1, 3)
    Rm = np.asarray(matrix, np.float32)[:3, :3]
    out = (nrm @ Rm.T).astype(np.float32)
    nl = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(nl, 1e-9)).astype(np.float32)


def _transform_bounds(bounds: tuple[np.ndarray, np.ndarray],
                      matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b0, b1 = (np.asarray(bounds[0], np.float32), np.asarray(bounds[1], np.float32))
    corners = np.array([
        [b0[0], b0[1], b0[2]], [b1[0], b0[1], b0[2]],
        [b0[0], b1[1], b0[2]], [b1[0], b1[1], b0[2]],
        [b0[0], b0[1], b1[2]], [b1[0], b0[1], b1[2]],
        [b0[0], b1[1], b1[2]], [b1[0], b1[1], b1[2]],
    ], np.float32)
    tc = _transform_points(corners, matrix)
    return tc.min(axis=0).astype(np.float32), tc.max(axis=0).astype(np.float32)


def _stage_bounds_guitar_frame(outline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, Minv = _guitar_model_matrix(outline)
    return _transform_bounds(_stage_light_bounds(), Minv)


def _gpu_sensor_render(ssbo_tris, ssbo_nodes, ssbo_ids,
                       n_tris, n_bvh_nodes,
                       tex_bands, bmin, bmax, dims,
                       camera_eye, camera_target,
                       sensor_w, sensor_h,
                       spp, max_bounces,
                       total_fwd_rays, dispatch_batch,
                       fov_deg=52.0,
                       ray_field_scale=6.0, ray_field_gamma=0.55,
                       vol_alpha=1.0, vol_steps=128,
                       air_diffuse_scatter=0.35,
                       air_specular_scatter=0.65,
                       air_anisotropy=12.0,
                       medium_extinction=0.5):
    """Render a 2-D sensor image from camera_eye toward camera_target.

    Each pixel integrates BOTH:
      - Volumetric emission: march through the forward irradiance 3D band
        textures (same spectral colormap as _MARCH_FS) from the ray entry to
        the nearest surface hit.
      - Surface radiance: BVH-hit surface shaded from the forward irradiance
        sampled at the hit point, BRDF-weighted, composited behind the volume
        with the remaining transmittance.

    Returns a GL_RGBA32F texture handle (sensor_w × sensor_h).
    .a accumulates the sample count for running-average normalisation.
    """
    from OpenGL.GL import (
        GL_COMPUTE_SHADER, GL_TEXTURE_2D, GL_RGBA32F,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
        GL_READ_WRITE, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
        GL_TEXTURE_FETCH_BARRIER_BIT,
    )
    from OpenGL.GL import glFinish

    eye    = np.asarray(camera_eye,    np.float32).ravel()[:3]
    target = np.asarray(camera_target, np.float32).ravel()[:3]
    fwd    = target - eye;  fwd_l = float(np.linalg.norm(fwd))
    if fwd_l < 1e-6:
        fwd = np.array([0.0, 0.0, 1.0], np.float32)
    else:
        fwd = (fwd / fwd_l).astype(np.float32)
    # Guitar-frame Y (neck direction) is the natural "up" for sensor views.
    # Fallback to Z if fwd is nearly parallel to Y (camera looking along the neck).
    world_up = np.array([0.0, 1.0, 0.0], np.float32)
    if abs(float(np.dot(fwd, world_up))) > 0.97:
        world_up = np.array([0.0, 0.0, 1.0], np.float32)
    right = np.cross(fwd, world_up); right /= max(np.linalg.norm(right), 1e-9)
    up    = np.cross(right, fwd);    up    /= max(np.linalg.norm(up),    1e-9)
    right = right.astype(np.float32)
    up    = up.astype(np.float32)
    fov_tan = float(math.tan(math.radians(fov_deg) * 0.5))
    aspect  = float(sensor_w) / float(max(1, sensor_h))

    # One RGBA32F output texture per film layer
    from OpenGL.GL import glClearTexImage
    _n_layers = max(1, len(tex_bands))
    sensor_textures: list[int] = [int(t) for t in glGenTextures(_n_layers)]
    for _st in sensor_textures:
        glBindTexture(GL_TEXTURE_2D, _st)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, sensor_w, sensor_h, 0,
                     GL_RGBA, GL_FLOAT, None)
        for param, val in [(GL_TEXTURE_MIN_FILTER, GL_NEAREST),
                           (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                           (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                           (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)]:
            glTexParameteri(GL_TEXTURE_2D, param, val)
        glClearTexImage(_st, 0, GL_RGBA, GL_FLOAT, np.zeros(4, np.float32))
    glBindTexture(GL_TEXTURE_2D, 0)
    sensor_tex = sensor_textures[0]

    prog = _prog((_GPU_SENSOR_CS, GL_COMPUTE_SHADER))
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        print("  [sensor] compute shader failed to link", flush=True)
        glDeleteProgram(prog)
        return sensor_textures

    glUseProgram(prog)

    # Bind BVH SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_tris)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_nodes)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo_ids)

    # Bind forward film-layer volumes as samplers at texture units 0..N-1
    for _i, _tex in enumerate(tex_bands):
        glActiveTexture(GL_TEXTURE0 + _i)
        glBindTexture(GL_TEXTURE_3D, _tex)
    glActiveTexture(GL_TEXTURE0)
    for _i in range(len(tex_bands)):
        glUniform1i(glGetUniformLocation(prog, f'uFwdLayer{_i}'.encode()), _i)
    for _i, _st in enumerate(sensor_textures):
        glBindImageTexture(_i, _st, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)

    def _u1i(n, v): glUniform1i(glGetUniformLocation(prog, n), int(v))
    def _u1f(n, v): glUniform1f(glGetUniformLocation(prog, n), float(v))
    def _u3f(n, x, y, z): glUniform3f(glGetUniformLocation(prog, n), float(x), float(y), float(z))
    def _u2i(n, x, y): glUniform2i(glGetUniformLocation(prog, n), int(x), int(y))
    def _u3i(n, x, y, z): glUniform3i(glGetUniformLocation(prog, n), int(x), int(y), int(z))
    glUniform1i(glGetUniformLocation(prog, b'uLayerCount'), _n_layers)
    for _li in range(_n_layers):
        glUniform3f(glGetUniformLocation(prog, f'uLayerDark{_li}'.encode()),  0.0, 0.0, 0.0)
        glUniform3f(glGetUniformLocation(prog, f'uLayerLight{_li}'.encode()), 1.0, 1.0, 1.0)

    _u1i(b'uTriCount',        n_tris)
    _u1i(b'uNodeCount',       n_bvh_nodes)
    _u1i(b'uSamplesPerPixel', max(1, spp))
    _u1i(b'uMaxBounces',      max(1, max_bounces))
    _u2i(b'uSensorSize',      sensor_w, sensor_h)
    _u1f(b'uRayFieldScale',   float(ray_field_scale))
    _u1f(b'uRayFieldGamma',   float(ray_field_gamma))
    _u1f(b'uVolAlpha',        float(vol_alpha))
    _u1i(b'uVolSteps',        int(vol_steps))
    _u1f(b'uAirDiffuseScatter',  float(air_diffuse_scatter))
    _u1f(b'uAirSpecularScatter', float(air_specular_scatter))
    _u1f(b'uAirAnisotropy',      float(air_anisotropy))
    _u1f(b'uMediumExtinction',   float(medium_extinction))
    _u3f(b'uCamEye',          *eye)
    _u3f(b'uCamRight',        *right)
    _u3f(b'uCamUp',           *up)
    _u3f(b'uCamFwd',          *fwd)
    _u1f(b'uCamFovTan',       fov_tan)
    _u1f(b'uCamAspect',       aspect)
    _u3f(b'uBoxMin',          *bmin.astype(np.float32))
    _u3f(b'uBoxMax',          *bmax.astype(np.float32))
    _u3i(b'uDims',            int(dims[0]), int(dims[1]), int(dims[2]))

    gx = max(1, int(math.ceil(sensor_w / 16.0)))
    # Dispatch in horizontal strips to stay well under TDR
    # Budget: ~500k pixels per strip × vol_steps work per pixel
    strip_rows = max(16, int(math.ceil(
        500_000 / max(1, sensor_w * max(1, vol_steps) * max(1, spp) // 256))))
    _finish_step = max(1, 500_000 // max(1, dispatch_batch))
    _tile = 0
    print(f"  [sensor] {sensor_w}×{sensor_h} × {spp} spp × {vol_steps} vol-steps  "
          f"(strips of {strip_rows} rows)", flush=True)

    from OpenGL.GL import glUniform1i as _u1i_raw
    row_off = 0
    while row_off < sensor_h:
        rows_this = min(strip_rows, sensor_h - row_off)
        cur_gy    = max(1, int(math.ceil(rows_this / 16.0)))
        # uSeed encodes the strip so each strip has unique per-pixel RNG
        glUniform1i(glGetUniformLocation(prog, b'uSeed'),      314159 + row_off)
        glUniform1i(glGetUniformLocation(prog, b'uRowOffset'), row_off)
        glDispatchCompute(gx, cur_gy, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        _tile += 1
        if _tile % _finish_step == 0:
            glFinish()
        row_off += rows_this

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                    GL_TEXTURE_FETCH_BARRIER_BIT)
    glUseProgram(0)
    for _i in range(_n_layers):
        glActiveTexture(GL_TEXTURE0 + _i)
        glBindTexture(GL_TEXTURE_3D, 0)
    glActiveTexture(GL_TEXTURE0)
    glDeleteProgram(prog)
    return sensor_textures


def _normalise_materials(materials: np.ndarray, n_tris: int) -> np.ndarray:
    """Return (n_tris, 11): mat_in[3] | mat_out[3] | albedo_rgb[3] | ior | opacity."""
    mat = np.asarray(materials, dtype=np.float32)
    if mat.ndim != 2 or len(mat) != n_tris:
        mat = np.zeros((n_tris, 0), dtype=np.float32)
    if mat.shape[1] < 3:
        mat_in = np.tile(np.array([0.40, 0.90, 0.05], dtype=np.float32), (n_tris, 1))
    else:
        mat_in = mat[:, 0:3]
    if mat.shape[1] < 6:
        mat_out = np.tile(np.array([0.40, 0.90, 0.05], dtype=np.float32), (n_tris, 1))
    else:
        mat_out = mat[:, 3:6]
    if mat.shape[1] < 9:
        albedo = np.tile(np.array([0.30, 0.29, 0.27], dtype=np.float32), (n_tris, 1))
    else:
        albedo = mat[:, 6:9]
    ior     = mat[:, 9:10]  if mat.shape[1] >= 10 else np.ones((n_tris, 1), np.float32)
    opacity = mat[:, 10:11] if mat.shape[1] >= 11 else np.ones((n_tris, 1), np.float32)
    return np.ascontiguousarray(
        np.column_stack([mat_in, mat_out, albedo, ior, opacity]), dtype=np.float32)


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
            mats = np.tile(np.array([0.40, 0.90, 0.05, 0.40, 0.90, 0.05],
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


# ─────────────────────────────────────────────────────────────────────────────
# _SourceWorker — background thread for CPU-side source computation
#
# Computes emission source lists from physics frames (pure NumPy, no GL).
# Results are posted to a deque(maxlen=1) so the GL thread always gets the
# freshest data without stalling.
# ─────────────────────────────────────────────────────────────────────────────

class _SourceWorker(threading.Thread):
    """Daemon thread: receives physics frame notifications and computes the
    next source list asynchronously.  All work is pure NumPy — no GL calls.

    The GL thread calls notify() to post a new job, then drains result_q
    (a deque(maxlen=1)) each display frame to check if new source data is
    ready.  If multiple frames arrive before the worker finishes, the stale
    jobs are discarded and only the latest is processed.
    """

    def __init__(self):
        super().__init__(daemon=True, name="SourceWorker")
        self._event   = threading.Event()
        self._lock    = threading.Lock()
        self._stop    = False
        self._pending: dict | None = None          # latest unprocessed job
        self.result_q: collections.deque = collections.deque(maxlen=1)

    # ── Public API (called from GL thread) ────────────────────────────────

    def notify(self, *,
               frame,
               info: dict,
               outline: np.ndarray,
               body_h: float,
               paths: list,
               active_strings,
               refresh_rays: int,
               sensor_light_rays: int,
               stage_light_emitters: int,
               n_sources_target: int = 0):
        """Post a new job.  Drops any previously queued but unstarted job."""
        job = dict(
            frame=frame,
            info=dict(info),
            outline=np.asarray(outline, dtype=np.float32),
            body_h=float(body_h),
            paths=list(paths),
            active_strings=active_strings,
            refresh_rays=int(refresh_rays),
            sensor_light_rays=int(sensor_light_rays),
            stage_light_emitters=int(stage_light_emitters),
            n_sources_target=int(n_sources_target),
        )
        with self._lock:
            self._pending = job
        self._event.set()

    def stop(self):
        self._stop = True
        self._event.set()

    # ── Internal ──────────────────────────────────────────────────────────

    def run(self):
        while not self._stop:
            self._event.wait()
            self._event.clear()
            if self._stop:
                break
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None:
                continue
            try:
                result = self._compute(job)
                if result is not None:
                    self.result_q.append(result)
            except Exception as exc:
                import traceback
                print(f"[SourceWorker] ERROR: {exc}\n{traceback.format_exc()}",
                      flush=True)

    @staticmethod
    def _compute(job: dict):
        frame               = job["frame"]
        info                = job["info"]
        outline             = job["outline"]
        body_h              = job["body_h"]
        paths               = job["paths"]
        active_strings      = job["active_strings"]
        refresh_rays        = job["refresh_rays"]
        sensor_light_rays   = job["sensor_light_rays"]
        stage_light_emitters= job["stage_light_emitters"]

        n_str = max(1, len(paths))
        if frame is not None and "plate_active_2d" in info:
            board_sources = _instant_soundboard_sources(
                frame.plate, info["plate_active_2d"], outline, body_h,
                total_rays=refresh_rays, stride=4)
        else:
            board_sources = _geometry_soundboard_sources(
                outline, body_h,
                total_rays=refresh_rays * 3 // 4,
                stride=6)

        # String emission: when physics is running use actual displaced positions
        # so emission follows the real vibration pattern (anti-nodes glow, nodes
        # are silent).  Fall back to the static geometry path when no frame.
        str_rays = max(64, refresh_rays // n_str // 16)
        if frame is not None and hasattr(frame, "strings") and frame.strings:
            str_sources = _instant_string_sources(
                str_paths_current=frame.strings,
                str_paths_equilibrium=paths,
                total_rays_per_string=str_rays,
                active_strings=active_strings)
            # If the string has zero displacement this frame fall back gracefully
            if not str_sources:
                str_sources = _string_emission_sources(
                    paths, body_h,
                    total_rays_per_string=str_rays,
                    active_strings=active_strings)
        else:
            str_sources = _string_emission_sources(
                paths, body_h,
                total_rays_per_string=str_rays,
                active_strings=active_strings)
        sources = _sensor_scene_sources(
            outline, board_sources + str_sources,
            light_rays=sensor_light_rays,
            light_emitters=stage_light_emitters)
        if not sources:
            return None

        # Expand into mono-spectral stochastic packets
        sources = _stochastic_spectral_packets(
            sources, packets_per_source=32,
            seed=1776 + len(sources))

        # Pack source records SSBO layout (N × 12 float32)
        n = max(1, len(sources))
        total_rays = max(1, sum(int(s[2]) for s in sources))
        rec = np.zeros((n, 12), np.float32)
        for i, _s in enumerate(sources):
            sp, sd, nr, spec = _s[0], _s[1], _s[2], _s[3]
            _d = np.asarray(sd, np.float32).ravel()[:3]
            _dl = float(np.linalg.norm(_d))
            _d = (_d / _dl if _dl > 1e-9 else np.array([0., 0., 1.], np.float32))
            # model_int and model_param encoded directly at source-construction time.
            _model_int   = int(_s[5])   if len(_s) > 5 else 0
            _model_param = float(_s[6]) if len(_s) > 6 else 0.0
            rec[i, 0:3]  = np.asarray(sp, np.float32).ravel()[:3]
            rec[i, 3]    = float(max(1, int(nr))) / float(total_rays)
            rec[i, 4:7]  = _d
            rec[i, 7]    = float(_model_int)
            rec[i, 8:12] = np.asarray(spec, np.float32).ravel()[:4]
            rec[i, 11]   = _model_param
        rec = np.ascontiguousarray(rec, np.float32)

        return {"sources_list": sources, "source_records_np": rec}


def _gpu_ray_field(scene, outline, body_h, sources, max_bounces,
                   dims=GPU_RAY_FIELD_DIMS, segment_cap=GPU_RAY_SEGMENT_CAP,
                   dispatch_batch=GPU_DISPATCH_BATCH, include_stage=False,
                   model_matrix: Optional[np.ndarray] = None,
                   bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
                   sim_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
                   diagnostics: bool = True,
                   segment_capture: bool = True,
                   time_queries: bool = False,
                   sensor_pos: Optional[np.ndarray] = None,
                   sensor_pos_space: str = "guitar",
                   sensor_rays: int = 0,
                   sensor_gain: float = 8.0,
                   sensor_w: int = 0,
                   sensor_h: int = 0,
                   film: Optional[FilmStack] = None,
                   air_diffuse_scatter: float = 0.35,
                   air_specular_scatter: float = 0.65,
                   air_anisotropy: float = 12.0):
    _ray_diag_update(
        "gpu_ray_field:start",
        n_sources=len(sources) if sources is not None else 0,
        max_bounces=int(max_bounces),
        dims=tuple(int(v) for v in dims),
        segment_cap=int(segment_cap),
        dispatch_batch=int(dispatch_batch),
    )
    if scene is None or _extract_scene_geometry_fn is None:
        return None, None, None, None, 0, None, 0, 0
    sources = _stochastic_spectral_packets(
        list(sources or []),
        packets_per_source=32,
        seed=1776 + len(sources or []))
    film_stack = film if film is not None else _default_stack()
    if _extract_scene_geometry_materials_fn is not None:
        verts_flat, normals, materials = _extract_scene_geometry_materials_fn(scene)
    else:
        verts_flat, normals = _extract_scene_geometry_fn(scene)
        materials = np.tile(np.array([0.40, 0.90, 0.05, 0.40, 0.90, 0.05],
                                     dtype=np.float32), (len(verts_flat), 1))
    verts_flat = np.asarray(verts_flat, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    if model_matrix is not None and len(verts_flat):
        M = np.asarray(model_matrix, dtype=np.float32)
        verts_flat = _transform_points(verts_flat, M)
        normals = _transform_normals(normals, M)
    def _pad_mat16(m: np.ndarray,
                   default_in=(0.40, 0.90, 0.05),
                   default_out=(0.40, 0.90, 0.05),
                   default_albedo=(0.30, 0.29, 0.27),
                   default_ior=1.55, default_opacity=1.0) -> np.ndarray:
        """Ensure material array is (N, 16): first 11 cols are mat11 with physical
        defaults; cols 11-15 are flags/emissive/reactive defaulting to zero."""
        m = np.asarray(m, np.float32)
        if m.ndim == 1:
            m = m.reshape(1, -1)
        defaults11 = [
            float(default_in[0]), float(default_in[1]), float(default_in[2]),
            float(default_out[0]), float(default_out[1]), float(default_out[2]),
            float(default_albedo[0]), float(default_albedo[1]), float(default_albedo[2]),
            float(default_ior), float(default_opacity),
        ]
        while m.shape[1] < 11:
            col = np.full((len(m), 1), defaults11[m.shape[1]], np.float32)
            m = np.hstack([m, col])
        while m.shape[1] < 16:
            m = np.hstack([m, np.zeros((len(m), 1), np.float32)])
        return m

    materials = _pad_mat16(materials)
    if include_stage:
        st_v, st_n, st_m = _stage_mesh(outline, body_h)
        if model_matrix is None:
            # Stage/room shell is authored in world coordinates.  The acoustic
            # band textures and guitar geometry are in guitar-frame, so bring
            # the room into that same frame for the sensor BVH.
            _, Minv = _guitar_model_matrix(outline)
            st_v = _transform_points(st_v.reshape(-1, 3), Minv).reshape(-1, 9)
            st_n = _transform_normals(st_n, Minv)
        verts_flat = np.vstack([np.asarray(verts_flat, np.float32),
                                np.asarray(st_v, np.float32).reshape(-1, 3)])
        normals = np.vstack([np.asarray(normals, np.float32), st_n])
        materials = np.vstack([materials, _pad_mat16(st_m)])

    # ── Neck wood and steel strings as BVH geometry ─────────────────────────────
    _y_nut_bvh  = BRIDGE_POS[1][1] + SCALE_LENGTH_M
    _y_head_bvh = _y_nut_bvh + 0.13
    _y_body_bvh = float(outline[:, 1].max()) * 0.85
    _nk_z       = body_h + 0.003
    _nk_mat6    = np.array([0.72, 0.38, 0.08, 0.72, 0.38, 0.08, 0.42, 0.22, 0.09, 1.52, 1.0], np.float32)
    _bvh_ev, _bvh_en, _bvh_em = [], [], []

    def _bvh_quad(corners, mat6):
        a, b, c, d = corners
        for tri in [(a, b, c), (a, c, d)]:
            e1 = tri[1] - tri[0]; e2 = tri[2] - tri[0]
            nm = np.cross(e1, e2); nl = np.linalg.norm(nm)
            if nl < 1e-12:
                continue
            nm /= nl
            _bvh_ev.extend(tri);       _bvh_en.append(nm);  _bvh_em.append(mat6)
            _bvh_ev.extend(tri[::-1]); _bvh_en.append(-nm); _bvh_em.append(mat6)

    _nw0, _nw1, _hw = 0.056, 0.044, 0.092
    _bvh_quad([
        np.array([-_nw0 * 0.5, _y_body_bvh, _nk_z], np.float32),
        np.array([ _nw0 * 0.5, _y_body_bvh, _nk_z], np.float32),
        np.array([ _nw1 * 0.5, _y_nut_bvh,  _nk_z], np.float32),
        np.array([-_nw1 * 0.5, _y_nut_bvh,  _nk_z], np.float32),
    ], _nk_mat6)
    _bvh_quad([
        np.array([-_hw * 0.42, _y_nut_bvh,  _nk_z],         np.float32),
        np.array([ _hw * 0.42, _y_nut_bvh,  _nk_z],         np.float32),
        np.array([ _hw * 0.58, _y_head_bvh, _nk_z + 0.002], np.float32),
        np.array([-_hw * 0.58, _y_head_bvh, _nk_z + 0.002], np.float32),
    ], _nk_mat6)

    _str_z_bvh = body_h + STRING_CLEARANCE
    # Plain steel strings (treble, unwound): bright silver, IOR 2.95
    _steel_plain = np.array([0.91, 0.04, 0.02, 0.91, 0.04, 0.02,
                              0.82, 0.82, 0.80, 2.95, 1.0], np.float32)
    # Wound strings (bass, phosphor-bronze wrap): warm gold-bronze, IOR 1.85
    _steel_wound = np.array([0.88, 0.06, 0.03, 0.88, 0.06, 0.03,
                              0.82, 0.62, 0.35, 1.85, 1.0], np.float32)
    _z_up      = np.array([0.0, 0.0, 1.0], np.float32)
    _x_span_bvh = 0.0088 * 5.0
    for _si in range(6):
        _xs_bvh = -_x_span_bvh * 0.5 + _si * (_x_span_bvh / 5.0)
        _xn_bvh = _xs_bvh * 0.72
        _gm = max(float(STRING_GAUGES_IN[_si] if _si < len(STRING_GAUGES_IN) else 0.012) * 0.0254, 0.0005)
        # Strings with gauge ≥ 0.024" are wound (bass); thinner are plain steel
        _str_mat = _steel_wound if (STRING_GAUGES_IN[_si] if _si < len(STRING_GAUGES_IN) else 0.010) >= 0.024 else _steel_plain
        _ys  = np.linspace(_y_nut_bvh, -0.070, 31, dtype=np.float32)
        _xs2 = np.linspace(_xn_bvh, _xs_bvh, 31, dtype=np.float32)
        for _k in range(30):
            _p0 = np.array([_xs2[_k],   _ys[_k],   _str_z_bvh], np.float32)
            _p1 = np.array([_xs2[_k+1], _ys[_k+1], _str_z_bvh], np.float32)
            _dv = _p1 - _p0; _dl = float(np.linalg.norm(_dv))
            if _dl < 1e-9:
                continue
            _perp = np.cross(_dv / _dl, _z_up)
            _pl = float(np.linalg.norm(_perp))
            if _pl < 1e-9:
                continue
            _perp = (_perp / _pl) * (_gm * 0.5)
            _a = _p0 + _perp; _b = _p0 - _perp
            _c = _p1 + _perp; _e = _p1 - _perp
            for _tri, _nm in [((_a, _b, _e), _z_up),  ((_a, _e, _c), _z_up),
                               ((_e, _b, _a), -_z_up), ((_c, _e, _a), -_z_up)]:
                _bvh_ev.extend(_tri); _bvh_en.append(_nm); _bvh_em.append(_str_mat)

    if _bvh_ev:
        verts_flat = np.vstack([verts_flat, np.array(_bvh_ev, np.float32)])
        normals    = np.vstack([normals,    np.array(_bvh_en, np.float32)])
        materials  = np.vstack([materials,  _pad_mat16(np.array(_bvh_em, np.float32))])

    # ── Glass bell jar for the BVH (refraction in sensor render) ────────────────
    # sim_bounds are in guitar-frame.  The builders expect world-frame (Z = up).
    # Build in world-frame then apply the inverse guitar matrix so the verts land
    # in the same guitar-frame space as the rest of the BVH geometry.
    if sim_bounds is not None:
        _gm_for_bj, _gm_inv_for_bj = _guitar_model_matrix(outline)
        _wb_min_bj, _wb_max_bj = _transform_bounds(sim_bounds, _gm_for_bj)
        _wb_min_bj = _wb_min_bj.astype(np.float32)
        _wb_max_bj = _wb_max_bj.astype(np.float32)
        _glass_mat = _load_material_yaml("borosilicate_glass")
        _gbell_w = _build_sim_belljar_world(_wb_min_bj, _wb_max_bj)
        _gv_w = _gbell_w[:, :3].reshape(-1, 3)
        _gn_w = _gbell_w[:, 3:].reshape(-1, 3)
        _gv = _transform_points(_gv_w, _gm_inv_for_bj)
        _gn = _transform_normals(_gn_w, _gm_inv_for_bj)
        _gm_rep = np.tile(_glass_mat, (len(_gv) // 3, 1))
        verts_flat = np.vstack([verts_flat, _gv])
        normals    = np.vstack([normals,    _gn[::3]])   # one normal per tri vertex 0
        materials  = np.vstack([materials,  _gm_rep])
        # Opaque skirt below the bell jar (floor → sim bottom)
        _gskirt_w = _build_sim_skirt_world(_wb_min_bj, _wb_max_bj)
        if len(_gskirt_w) > 0:
            _skirt_mat = _load_material_yaml("stage_floor")
            _sv_w = _gskirt_w[:, :3].reshape(-1, 3)
            _sn_w = _gskirt_w[:, 3:].reshape(-1, 3)
            _sv = _transform_points(_sv_w, _gm_inv_for_bj)
            _sn = _transform_normals(_sn_w, _gm_inv_for_bj)
            _sm_rep = np.tile(_skirt_mat, (len(_sv) // 3, 1))
            verts_flat = np.vstack([verts_flat, _sv])
            normals    = np.vstack([normals,    _sn[::3]])
            materials  = np.vstack([materials,  _sm_rep])

    if len(verts_flat) == 0:
        return None, None, None, None, 0, None, 0, 0

    tris = verts_flat.reshape(-1, 3, 3).astype(np.float32)
    nrm = normals.astype(np.float32)
    # _normalise_materials yields (N,11); pull emissive/flag cols from raw input
    mat11    = _normalise_materials(materials, len(tris))
    mat_in   = mat11[:, 0:3]
    mat_out  = mat11[:, 3:6]
    albedo   = mat11[:, 6:9]
    ior_col  = mat11[:, 9]
    opac_col = mat11[:, 10]
    _raw16  = np.asarray(materials, np.float32)
    # mat16 extension cols: [11]=flags, [12]=emit_profile_idx, [13]=remit_profile_idx, [15]=reactive_shift
    n_tris = len(tris)
    mat_flags_col     = _raw16[:, 11] if _raw16.shape[1] > 11 else np.zeros(n_tris, np.float32)
    emit_profile_idx  = _raw16[:, 12] if _raw16.shape[1] > 12 else np.full(n_tris, -1.0, np.float32)
    remit_profile_idx = _raw16[:, 13] if _raw16.shape[1] > 13 else np.full(n_tris, -1.0, np.float32)
    reactive_shift    = _raw16[:, 15] if _raw16.shape[1] > 15 else np.zeros(n_tris, np.float32)
    emit_has_profile  = (emit_profile_idx >= 0).astype(np.uint32)
    _react_mask = (np.abs(reactive_shift) > 0.01).astype(np.uint32)
    _flags_u32 = (np.frombuffer(mat_flags_col.tobytes(), np.uint32)
                  | (emit_has_profile * np.uint32(1))
                  | (_react_mask * np.uint32(2)))
    mat_flags_col = np.frombuffer(_flags_u32.tobytes(), np.float32)
    # 8×vec4 packed layout (32 floats per tri):
    #  [0-2]  v0.xyz          [3]    unused
    #  [4-6]  e1.xyz          [7]    unused
    #  [8-10] e2.xyz          [11]   unused
    #  [12-14] normal.xyz     [15]   mat_flags (uint bits as float)
    #  [16-18] mat_in.xyz     [19]   IOR
    #  [20-22] mat_out.xyz    [23]   opacity
    #  [24-26] albedo.xyz     [27]   emissive intensity (1.0 if profile assigned, else 0)
    #  [28]    emit_profile_idx       [29]  remit_profile_idx  [30] _pad  [31] reactive_shift_hz
    packed = np.zeros((n_tris, 32), np.float32)
    packed[:, 0:3]   = tris[:, 0, :]
    packed[:, 4:7]   = tris[:, 1, :] - tris[:, 0, :]
    packed[:, 8:11]  = tris[:, 2, :] - tris[:, 0, :]
    packed[:, 12:15] = nrm
    packed[:, 15]    = mat_flags_col    # normal.w  = mat_flags
    packed[:, 16:19] = mat_in
    packed[:, 19]    = ior_col          # mat_in.w  = IOR
    packed[:, 20:23] = mat_out
    packed[:, 23]    = opac_col         # mat_out.w = opacity
    packed[:, 24:27] = albedo
    packed[:, 27]    = emit_has_profile.astype(np.float32)  # albedo.w = 1 if emission profile assigned
    packed[:, 28]    = emit_profile_idx   # emissive.x = emit profile index
    packed[:, 29]    = remit_profile_idx  # emissive.y = remit profile index
    packed[:, 30]    = 0.0               # emissive.z = _pad
    packed[:, 31]    = reactive_shift    # emissive.w = reactive Stokes shift Hz
    packed = np.ascontiguousarray(packed)
    bvh_nodes, bvh_ids = _build_gpu_bvh(tris)
    _ray_diag_update(
        "gpu_ray_field:geometry",
        n_sources=len(sources),
        total_source_rays=sum(int(s[2]) for s in sources),
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
    elif include_stage and model_matrix is None:
        bmin, bmax = _stage_bounds_guitar_frame(outline)
    else:
        min_xy = outline.min(axis=0).astype(np.float32)
        max_xy = outline.max(axis=0).astype(np.float32)
        bmin = np.array([min_xy[0] - 0.012, min_xy[1] - 0.012, -0.002], np.float32)
        bmax = np.array([max_xy[0] + 0.012, max_xy[1] + 0.012, body_h + 0.004], np.float32)

    _bplate_cx = float(bmin[0] + bmax[0]) * 0.5
    _bplate_cy = float(bmin[1] + bmax[1]) * 0.5
    _bplate_z  = float(bmin[2])
    _sc = np.array([_bplate_cx, _bplate_cy, _bplate_z], np.float32)
    if sensor_pos is not None:
        _sp = np.asarray(sensor_pos, np.float32).ravel()[:3]
        if str(sensor_pos_space).lower() == "world":
            _, Minv = _guitar_model_matrix(outline)
            _sp = _transform_points(_sp.reshape(1, 3), Minv)[0]
    else:
        _standoff = max(0.30, float(bmax[2] - bmin[2]) * 5.0)
        _sp = np.array([_bplate_cx, _bplate_cy, _bplate_z - _standoff], np.float32)
    _sensor_dir = _sc - _sp
    _sensor_dir = (_sensor_dir / max(float(np.linalg.norm(_sensor_dir)), 1e-9)).astype(np.float32)

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
    counter = np.zeros(5, dtype=np.uint32)   # [seg, ray, hit, record, pending]
    # PendingRayBuf: capacity = fraction of forward ray budget (emissive hits rare)
    _pending_cap = max(0, min(int(dispatch_batch * 4), 1 << 18))  # max 256K pending rays
    _pending_stride = 16  # 4×vec4 per PendingRay = 16 floats = 64 bytes
    _pending_bytes = max(64, _pending_cap * _pending_stride * 4)
    total_source_rays = max(1, sum(int(s[2]) for s in sources))
    source_records = np.zeros((max(1, len(sources)), 12), np.float32)
    for _i, _src_rec in enumerate(sources):
        _sp_src, _sd_src, _nr_src, _spec_src = _src_rec[0], _src_rec[1], _src_rec[2], _src_rec[3]
        _sd_arr = np.asarray(_sd_src, np.float32).ravel()[:3]
        _sd_len = float(np.linalg.norm(_sd_arr))
        if _sd_len > 1e-9:
            _sd_arr = _sd_arr / _sd_len
        else:
            _sd_arr = np.array([0.0, 0.0, 1.0], np.float32)
        # model_int and model_param encoded directly at source-construction time.
        _model_int_src   = int(_src_rec[5])   if len(_src_rec) > 5 else 0
        _model_param_src = float(_src_rec[6]) if len(_src_rec) > 6 else 0.0
        source_records[_i, 0:3] = np.asarray(_sp_src, np.float32).ravel()[:3]
        source_records[_i, 3] = float(max(1, int(_nr_src))) / float(total_source_rays)
        source_records[_i, 4:7] = _sd_arr
        source_records[_i, 7] = float(_model_int_src)
        source_records[_i, 8:12] = np.asarray(_spec_src, np.float32).ravel()[:4]
        source_records[_i, 11]   = _model_param_src
    source_records = np.ascontiguousarray(source_records, np.float32)

    ssbo = list(glGenBuffers(9))   # 0=Tri 1=Node 2=TriId 3=Seg 4=Counter 5=Source 6=PendingRay 7=EmitProfiles 8=ScaleCtx
    _ray_diag_update(
        "gpu_ray_field:alloc_buffers",
        n_sources=len(sources),
        total_source_rays=sum(int(s[2]) for s in sources),
        n_tris=len(tris),
        n_bvh_nodes=len(bvh_nodes),
        tex_bands=tuple(int(t) for t in tex_bands),
        ssbo=tuple(int(b) for b in ssbo),
        seg_bytes=int(seg_bytes),
        source_record_bytes=int(source_records.nbytes),
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
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[5])
    glBufferData(GL_SHADER_STORAGE_BUFFER, source_records.nbytes, source_records, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssbo[5])
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[6])
    glBufferData(GL_SHADER_STORAGE_BUFFER, _pending_bytes, None, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ssbo[6])
    # Emission profile table — built from the module-level EmissionProfileDatabase singleton.
    try:
        from material_db import _EMISSION_DB as _ep_db
        _ep_tensor = _ep_db.build_gpu_tensor()
    except Exception:
        _ep_tensor = np.zeros((1, 20), np.float32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[7])
    glBufferData(GL_SHADER_STORAGE_BUFFER, _ep_tensor.nbytes, _ep_tensor, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssbo[7])
    # ScaleContext SSBO at binding 8 — empty placeholder; count=0 disables context loops.
    # Populated externally when GPU scene with wave contexts is used.
    _scale_ctx_placeholder = np.zeros((1, 8), np.float32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[8])
    glBufferData(GL_SHADER_STORAGE_BUFFER, _scale_ctx_placeholder.nbytes, _scale_ctx_placeholder, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, ssbo[8])

    prog = _prog((_GPU_RAY_FIELD_CS, GL_COMPUTE_SHADER))
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        print("  [gpu_ray_field] compute shader did not link; ray volume disabled", flush=True)
        glDeleteProgram(prog)
        from OpenGL.GL import glDeleteTextures as _glDelTex, glDeleteBuffers as _glDelBuf
        for tex in tex_bands:
            _glDelTex([tex])
        for buf in ssbo:
            _glDelBuf(1, [buf])
        return None, None, None, None, 0, None, 0, 0
    glUseProgram(prog)
    # imageAtomicAdd on all 4 band textures — must be GL_READ_WRITE
    for band_idx, tex in enumerate(tex_bands):
        glBindImageTexture(band_idx, tex, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI)
    glUniform1i(glGetUniformLocation(prog, b'uTriCount'), len(tris))
    glUniform1i(glGetUniformLocation(prog, b'uNodeCount'), len(bvh_nodes))
    glUniform1i(glGetUniformLocation(prog, b'uSegmentCap'), seg_cap)
    glUniform1i(glGetUniformLocation(prog, b'uSegmentStride'), seg_stride)
    glUniform1i(glGetUniformLocation(prog, b'uSourceCount'), len(sources))
    glUniform1i(glGetUniformLocation(prog, b'uMaxBounces'), int(max_bounces))
    glUniform3f(glGetUniformLocation(prog, b'uBoxMin'), *bmin)
    glUniform3f(glGetUniformLocation(prog, b'uBoxMax'), *bmax)
    glUniform3i(glGetUniformLocation(prog, b'uDims'), int(dims[0]), int(dims[1]), int(dims[2]))
    # Beer-Lambert participating medium — tuned defaults for a guitar-scale scene
    glUniform1f(glGetUniformLocation(prog, b'uVolumeStepMeters'), 0.003)
    glUniform1f(glGetUniformLocation(prog, b'uMediumScattering'),  1.0)
    glUniform1f(glGetUniformLocation(prog, b'uMediumExtinction'),  0.5)
    glUniform1f(glGetUniformLocation(prog, b'uAirDiffuseScatter'),  float(air_diffuse_scatter))
    glUniform1f(glGetUniformLocation(prog, b'uAirSpecularScatter'), float(air_specular_scatter))
    glUniform1f(glGetUniformLocation(prog, b'uAirAnisotropy'),      float(air_anisotropy))
    glUniform1i(glGetUniformLocation(prog, b'uScaleContextCount'), 0)   # default: no wave contexts
    glUniform1f(glGetUniformLocation(prog, b'uSpeedOfMedium'), 343.0)   # acoustic default; optical callers override
    _specs = film_stack.layer_specs()
    glUniform1i(glGetUniformLocation(prog, b'uLayerCount'), len(_specs))
    for _i, _sp_layer in enumerate(_specs):
        glUniform1f(glGetUniformLocation(prog, f'uLayerCentersHz[{_i}]'.encode()), _sp_layer['center_hz'])
        glUniform1f(glGetUniformLocation(prog, f'uLayerWidthsOct[{_i}]'.encode()), _sp_layer['width_oct'])
        glUniform1f(glGetUniformLocation(prog, f'uLayerGains[{_i}]'.encode()),     _sp_layer['gain'])

    loc_mode        = glGetUniformLocation(prog, b'uMode')
    loc_seed        = glGetUniformLocation(prog, b'uSeed')
    loc_src_pos     = glGetUniformLocation(prog, b'uSrcPos')
    loc_src_dir     = glGetUniformLocation(prog, b'uSrcDir')
    loc_batch_size  = glGetUniformLocation(prog, b'uBatchSize')
    loc_batch_off   = glGetUniformLocation(prog, b'uBatchOffset')
    loc_total_rays  = glGetUniformLocation(prog, b'uTotalRaysPerSource')
    loc_src_spec    = glGetUniformLocation(prog, b'uSrcSpectrum')
    glUniform1i(glGetUniformLocation(prog, b'uDiagnosticsEnabled'), 1 if diagnostics else 0)
    glUniform1i(glGetUniformLocation(prog, b'uSegmentCapture'),     1 if segment_capture else 0)
    glUniform1i(glGetUniformLocation(prog, b'uPendingCap'), _pending_cap)
    glUniform1i(loc_mode, 0)   # PASS_FORWARD

    dispatch_batch = max(128, int(dispatch_batch))
    total_rays = sum(s[2] for s in sources)
    print(f"  [gpu_ray_field] {len(sources)} sources, "
          f"{total_rays} total rays, "
          f"dispatch_batch={dispatch_batch}, "
          f"box Z=[{float(bmin[2]):.4f}, {float(bmax[2]):.4f}]",
          flush=True)

    rays_done = 0
    next_report = max(dispatch_batch, total_rays // 20) if total_rays > 0 else dispatch_batch
    for si, _src_entry in enumerate(sources):
        sp, sd, n_rays, spectrum = _src_entry[0], _src_entry[1], _src_entry[2], _src_entry[3]
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
        _q_elapsed_ns = 0
        if time_queries:
            from OpenGL.GL import (
                glGenQueries, glBeginQuery, glEndQuery,
                glGetQueryObjectuiv, glDeleteQueries,
                GL_TIME_ELAPSED, GL_QUERY_RESULT,
            )
            _tq = glGenQueries(1)[0]
            glBeginQuery(GL_TIME_ELAPSED, _tq)
        from OpenGL.GL import glFinish
        # How often to call glFinish() to reset the Windows TDR watchdog.
        # Each glFinish() lets the OS confirm the GPU is alive; without this,
        # a long sequence of dispatches with only glMemoryBarrier between them
        # can exceed the ~2s TDR timeout and silently kill the GL context.
        # At dispatch_batch=8192 a finish every 32 batches ≈ every 262k rays.
        _finish_every = max(1, 500_000 // max(1, dispatch_batch))
        _batch_count  = 0
        while offset < n_rays:
            batch = min(dispatch_batch, n_rays - offset)
            glUniform1i(loc_batch_size, batch)
            glUniform1i(loc_batch_off, offset)
            n_groups = max(1, int(math.ceil(batch / 128.0)))
            gx = min(n_groups, 65535)
            gy = max(1, int(math.ceil(n_groups / gx)))
            glDispatchCompute(gx, gy, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                            GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            _batch_count += 1
            if _batch_count % _finish_every == 0:
                # Full CPU/GPU sync: lets the TDR watchdog timer reset.
                glFinish()
            offset += batch
            rays_done += batch
            if total_rays >= 1_000_000 and rays_done >= next_report:
                print(f"  [gpu_ray_field] streamed {rays_done}/{total_rays} rays",
                      flush=True)
                next_report += max(dispatch_batch, total_rays // 20)
        if time_queries:
            glEndQuery(GL_TIME_ELAPSED)
            _q_elapsed_ns = int(glGetQueryObjectuiv(_tq, GL_QUERY_RESULT))
            glDeleteQueries([_tq])
            print(f"  [gpu_ray_field] source {si}: {n_rays} rays  "
                  f"{_q_elapsed_ns / 1e6:.2f} ms  "
                  f"{n_rays / max(_q_elapsed_ns / 1e9, 1e-9) / 1e6:.1f} Mrays/s",
                  flush=True)

    if int(sensor_rays) > 0:
        from OpenGL.GL import glCopyImageSubData, glDeleteTextures
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT)
        fwd_snapshot = list(glGenTextures(n_layers))
        for _dst, _src in zip(fwd_snapshot, tex_bands):
            glBindTexture(GL_TEXTURE_3D, _dst)
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, dims[0], dims[1], dims[2], 0,
                         GL_RED_INTEGER, GL_UNSIGNED_INT, None)
            for param, val in [(GL_TEXTURE_MIN_FILTER, GL_NEAREST),
                               (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                               (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                               (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                               (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)]:
                glTexParameteri(GL_TEXTURE_3D, param, val)
            glCopyImageSubData(_src, GL_TEXTURE_3D, 0, 0, 0, 0,
                               _dst, GL_TEXTURE_3D, 0, 0, 0, 0,
                               int(dims[0]), int(dims[1]), int(dims[2]))
        glBindTexture(GL_TEXTURE_3D, 0)
        glUniform1i(loc_mode, 1)
        glUniform1i(loc_total_rays, max(1, int(sensor_rays)))
        glUniform3f(loc_src_pos, *_sp)
        glUniform3f(loc_src_dir, *_sensor_dir)
        glUniform4f(loc_src_spec, 440.0, 0.0, 1.0, 1.0)
        glUniform1f(glGetUniformLocation(prog, b'uSensorNorm'), max(1.0, float(total_rays)))
        glUniform1f(glGetUniformLocation(prog, b'uSensorGain'), float(sensor_gain))
        glUniform1f(glGetUniformLocation(prog, b'uSensorConeCos'), float(math.cos(math.radians(32.0))))
        glUniform1f(glGetUniformLocation(prog, b'uSensorMisWeight'), 0.55)
        for _i, _tex in enumerate(fwd_snapshot):
            glActiveTexture(GL_TEXTURE0 + 8 + _i)
            glBindTexture(GL_TEXTURE_3D, _tex)
            glUniform1i(glGetUniformLocation(prog, f'uFwdLayer{_i}'.encode()), 8 + _i)
        glActiveTexture(GL_TEXTURE0)
        _srays_done = 0
        _sensor_finish_every = max(1, 500_000 // max(1, dispatch_batch))
        _sensor_batch_count = 0
        while _srays_done < int(sensor_rays):
            batch = min(dispatch_batch, int(sensor_rays) - _srays_done)
            glUniform1i(loc_seed, 8675309 + _srays_done)
            glUniform1i(loc_batch_size, batch)
            glUniform1i(loc_batch_off, _srays_done)
            n_groups = max(1, int(math.ceil(batch / 128.0)))
            gx = min(n_groups, 65535)
            gy = max(1, int(math.ceil(n_groups / gx)))
            glDispatchCompute(gx, gy, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                            GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                            GL_TEXTURE_FETCH_BARRIER_BIT)
            _sensor_batch_count += 1
            if _sensor_batch_count % _sensor_finish_every == 0:
                glFinish()
            _srays_done += batch
        for _i in range(4):
            glActiveTexture(GL_TEXTURE0 + 4 + _i)
            glBindTexture(GL_TEXTURE_3D, 0)
        glActiveTexture(GL_TEXTURE0)
        for _tex in fwd_snapshot:
            glDeleteTextures([_tex])
        glUniform1i(loc_mode, 0)   # reset to PASS_FORWARD

    # ── Reactive re-emission pass (PASS_REACTIVE) ─────────────────────────────
    # Read back pending_count (counter[4]) to decide if the pass is worth running.
    if _pending_cap > 0:
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[4])
        _ctr_raw = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 5 * 4)
        _pending_count_fwd = int(np.frombuffer(_ctr_raw, np.uint32)[4])
        if _pending_count_fwd > 0:
            _react_rays = min(_pending_count_fwd, _pending_cap)
            glUniform1i(loc_mode, 2)   # PASS_REACTIVE
            glUniform1i(loc_seed, 31415926)
            glUniform1i(loc_batch_size, _react_rays)
            glUniform1i(loc_batch_off, 0)
            n_groups = max(1, int(math.ceil(_react_rays / 128.0)))
            gx = min(n_groups, 65535)
            gy = max(1, int(math.ceil(n_groups / gx)))
            glDispatchCompute(gx, gy, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                            GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            glUniform1i(loc_mode, 0)   # reset to PASS_FORWARD
            print(f"  [gpu_ray_field] reactive pass: {_react_rays} pending rays", flush=True)

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT)

    gl_err = glGetError()
    if gl_err != GL_NO_ERROR:
        print(f"  [gpu_ray_field] GL error after dispatch: 0x{gl_err:04x}", flush=True)

    # Read back shader diagnostics: [segments, rays, hits, selected-for-recording, pending].
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[4])
    counter_raw = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 5 * 4)
    counter_readback = np.frombuffer(counter_raw, dtype=np.uint32, count=5).copy()
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
        sensor_rays=int(max(0, int(sensor_rays))),
        expected_diag_rays=int(int(total_rays) + max(0, int(sensor_rays))),
        segment_counter=int(counter_readback[0]),
        actual_seg_count=int(actual_seg_count),
        segment_cap=int(seg_cap),
        ssbo=tuple(int(b) for b in ssbo),
        tex_bands=tuple(int(t) for t in tex_bands),
    )
    expected_diag_rays = int(total_rays) + max(0, int(sensor_rays))
    if expected_diag_rays > 0 and rays_seen > expected_diag_rays:
        print(
            f"  [gpu_ray_field] diagnostic ray counter exceeded expected budget: "
            f"reported {rays_seen}, expected {expected_diag_rays} "
            f"({int(total_rays)} source + {max(0, int(sensor_rays))} sensor); "
            f"continuing with completed ray field",
            flush=True)
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
    for buf in [ssbo[0], ssbo[1], ssbo[2], ssbo[5], ssbo[7], ssbo[8]]:
        glDeleteBuffers(1, [buf])
    glDeleteProgram(prog)

    # ── 2-D camera sensor accumulator (progressive, volume + surface) ─────────
    # Operates in guitar-frame (canonical physics frame).
    # Camera looks at the back plate (Z=bmin face) from behind (Z < bmin).
    # BVH SSBOs (packed tris, nodes, ids) are re-uploaded here so the
    # accumulator owns persistent copies for its lifetime.
    _bplate_cx = float(bmin[0] + bmax[0]) * 0.5
    _bplate_cy = float(bmin[1] + bmax[1]) * 0.5
    _bplate_z  = float(bmin[2])  # back plate sits at Z=bmin (≈0 in guitar-frame)
    _standoff  = max(0.30, float(bmax[2] - bmin[2]) * 5.0)  # ≥5× body depth
    _sc = np.array([_bplate_cx, _bplate_cy, _bplate_z], np.float32)
    if sensor_pos is not None:
        _sp = np.asarray(sensor_pos, np.float32).ravel()[:3]
        if str(sensor_pos_space).lower() == "world":
            _, Minv = _guitar_model_matrix(outline)
            _sp = _transform_points(_sp.reshape(1, 3), Minv)[0]
    else:
        # Default: position sensor directly behind the back plate centre.
        _sp = np.array([_bplate_cx, _bplate_cy, _bplate_z - _standoff], np.float32)
    _sw  = max(1, sensor_w) if sensor_w and sensor_w > 0 else WIN_W
    _sh  = max(1, sensor_h) if sensor_h and sensor_h > 0 else WIN_H
    # Persistent SSBOs owned by the accumulator (not freed here)
    _s_ssbo = list(glGenBuffers(3))
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _s_ssbo[0])
    glBufferData(GL_SHADER_STORAGE_BUFFER, packed.nbytes, packed, GL_STATIC_DRAW)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _s_ssbo[1])
    glBufferData(GL_SHADER_STORAGE_BUFFER, bvh_nodes.nbytes, bvh_nodes, GL_STATIC_DRAW)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _s_ssbo[2])
    glBufferData(GL_SHADER_STORAGE_BUFFER, bvh_ids.nbytes, bvh_ids, GL_STATIC_DRAW)
    print(f"  [sensor] starting progressive accumulator  {_sw}x{_sh}  "
          f"{len(tris)} tris  {len(bvh_nodes)} BVH nodes", flush=True)
    sensor_acc = SensorAccumulator(
        ssbo_tris=_s_ssbo[0], ssbo_nodes=_s_ssbo[1], ssbo_ids=_s_ssbo[2],
        n_tris=len(tris), n_bvh_nodes=len(bvh_nodes),
        tex_bands=tex_bands, bmin=bmin, bmax=bmax, dims=dims,
        camera_eye=_sp, camera_target=_sc,
        sensor_w=_sw, sensor_h=_sh,
        rows_per_frame=16,
        max_bounces=max(1, int(max_bounces)),
        ray_field_scale=float(sensor_gain), ray_field_gamma=0.55,
        vol_alpha=1.0, vol_steps=64)
    # Hand the accumulator the source list so it can re-emit forward rays
    # every sensor-frame cadence tick (full per-frame BDPT).
    sensor_acc.update_sources(
        sources_list=sources,
        source_records_np=source_records,
        dispatch_batch=int(dispatch_batch),
        air_diffuse_scatter=float(air_diffuse_scatter),
        air_specular_scatter=float(air_specular_scatter),
        air_anisotropy=float(air_anisotropy))
    # Note: _s_ssbo ownership is transferred to sensor_acc; do NOT free here.

    return tex_bands, bmin, bmax, ssbo[3], actual_seg_count, ssbo[4], int(total_rays), sensor_acc


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
    tex_bands = list(glGenTextures(len(data)))
    for tex, arr in zip(tex_bands, data):
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


# ─── Pre-packed GPU pipeline (optical / camera-designer scenes) ───────────────
# These functions accept already-packed SSBO arrays (from build_gpu_scene())
# and keep all GL objects alive for per-frame pumping.  Unlike _gpu_ray_field(),
# they do NOT free prog/SSBOs on return — the caller owns them.

# Compact grid for optical scenes: 128 transverse × 128 transverse × 256 axial.
GPU_OPTICAL_FIELD_DIMS = (128, 128, 256)

def _gpu_ray_field_prebuilt(
    packed,          # (N, 32) float32  — pre-packed triangle SSBO
    bvh_nodes,       # float32 from _build_gpu_bvh
    bvh_ids,         # uint32  from _build_gpu_bvh
    context_buf,     # (M, 8)  float32  — ScaleContext SSBO rows
    source_buf,      # (N, 12) float32  — pre-baked rays from RayOrder.bake_rays()
                     #   [pos_xyz, amp_w, dir_xyz, 0, freq_hz, phase, energy, 0]
    bounds,          # (bmin, bmax) each float32 (3,)
    dims=GPU_OPTICAL_FIELD_DIMS,
    dispatch_batch=8192,
    max_bounces=8,
    initial_rays=32768,
    speed_of_medium=2.998e8,   # optical vacuum
    film_stack=None,           # FilmStack | None — drives layer uniforms & tex count
):
    """Compile the canonical _GPU_RAY_FIELD_CS with pre-packed scene SSBOs.

    Returns a state dict with all GL handles + cached uniform locations.
    The caller owns every GL object and must call _destroy_prebuilt(state)
    when the scene changes.  Returns None on shader link failure.
    """
    from OpenGL.GL import (
        GL_COMPUTE_SHADER, GL_LINK_STATUS, GL_READ_WRITE, GL_R32UI,
        GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R,
        GL_NEAREST, GL_CLAMP_TO_EDGE, GL_RED_INTEGER, GL_UNSIGNED_INT,
        GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW, GL_DYNAMIC_DRAW,
        GL_SHADER_STORAGE_BARRIER_BIT, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
        glGenTextures, glBindTexture, glTexImage3D, glTexParameteri,
        glClearTexImage, glBindImageTexture,
        glGenBuffers, glBindBuffer, glBufferData, glBindBufferBase,
        glGetProgramiv, glDeleteProgram, glDeleteTextures, glDeleteBuffers,
        glUseProgram, glGetUniformLocation,
        glUniform1i, glUniform1f, glUniform3f, glUniform4f, glUniform3i,
        glDispatchCompute, glMemoryBarrier, glFinish,
    )

    bmin = np.asarray(bounds[0], np.float32)
    bmax = np.asarray(bounds[1], np.float32)
    n_tris = len(packed)

    # ── Resolve film layers ───────────────────────────────────────────────
    if film_stack is not None and hasattr(film_stack, 'layer_specs'):
        _layer_specs = film_stack.layer_specs()
    else:
        # Broadband luminance fallback — one layer that passes every frequency
        _layer_specs = [{'center_hz': 1.0, 'width_oct': 100.0, 'gain': 1.0,
                         'dark_rgb': (0.0, 0.0, 0.0), 'light_rgb': (1.0, 1.0, 1.0)}]
    n_layers = min(len(_layer_specs), 8)

    # ── Source records: pre-baked rays from RayOrder.bake_rays() ─────────
    # source_buf is already (N, 12) float32 in SourceRec layout — no expansion.
    # Phase and direction are fully baked CPU-side; the GPU just traces them.
    n_src          = max(1, len(source_buf))
    source_records = np.ascontiguousarray(source_buf, np.float32)

    seg_cap    = 0   # camera station doesn't use CPU segment readback
    seg_stride = 4
    seg_bytes  = 64  # minimum valid allocation
    counter    = np.zeros(5, np.uint32)
    _pending_cap   = max(0, min(int(dispatch_batch * 4), 1 << 18))
    _pending_bytes = max(64, _pending_cap * 16 * 4)

    # ── n_layers spectral accumulation textures (one per film layer) ─────
    zero = np.array([0], np.uint32)
    tex_bands = list(glGenTextures(n_layers))
    for tex in tex_bands:
        glBindTexture(GL_TEXTURE_3D, tex)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI,
                     dims[0], dims[1], dims[2], 0,
                     GL_RED_INTEGER, GL_UNSIGNED_INT, None)
        for param, val in [(GL_TEXTURE_MIN_FILTER, GL_NEAREST),
                           (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                           (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                           (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                           (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)]:
            glTexParameteri(GL_TEXTURE_3D, param, val)
        glClearTexImage(tex, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, zero)
    glBindTexture(GL_TEXTURE_3D, 0)

    # ── 9 SSBOs ───────────────────────────────────────────────────────────
    ssbo = list(glGenBuffers(9))
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
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[5])
    glBufferData(GL_SHADER_STORAGE_BUFFER, source_records.nbytes, source_records, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssbo[5])
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[6])
    glBufferData(GL_SHADER_STORAGE_BUFFER, _pending_bytes, None, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ssbo[6])
    try:
        from material_db import _EMISSION_DB as _ep_db
        _ep_tensor = _ep_db.build_gpu_tensor()
    except Exception:
        _ep_tensor = np.zeros((1, 20), np.float32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[7])
    glBufferData(GL_SHADER_STORAGE_BUFFER, _ep_tensor.nbytes, _ep_tensor, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssbo[7])
    _ctx = np.ascontiguousarray(context_buf, np.float32) if len(context_buf) > 0 \
           else np.zeros((1, 8), np.float32)
    n_ctx = len(_ctx)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[8])
    glBufferData(GL_SHADER_STORAGE_BUFFER, _ctx.nbytes, _ctx, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, ssbo[8])

    # ── Compile shader ────────────────────────────────────────────────────
    prog = _prog((_GPU_RAY_FIELD_CS, GL_COMPUTE_SHADER))
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        print("  [gpu_ray_field_prebuilt] compute shader did not link", flush=True)
        glDeleteProgram(prog)
        for tex in tex_bands:
            glDeleteTextures([tex])
        for buf in ssbo:
            glDeleteBuffers(1, [buf])
        return None

    glUseProgram(prog)
    for band_idx, tex in enumerate(tex_bands):
        glBindImageTexture(band_idx, tex, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI)

    # Static uniforms (don't change between pumps)
    glUniform1i(glGetUniformLocation(prog, b'uTriCount'),        n_tris)
    glUniform1i(glGetUniformLocation(prog, b'uNodeCount'),       len(bvh_nodes))
    glUniform1i(glGetUniformLocation(prog, b'uSegmentCap'),      seg_cap)
    glUniform1i(glGetUniformLocation(prog, b'uSegmentStride'),   seg_stride)
    glUniform1i(glGetUniformLocation(prog, b'uSourceCount'),     n_src)
    glUniform1i(glGetUniformLocation(prog, b'uMaxBounces'),      int(max_bounces))
    glUniform3f(glGetUniformLocation(prog, b'uBoxMin'),          *bmin)
    glUniform3f(glGetUniformLocation(prog, b'uBoxMax'),          *bmax)
    glUniform3i(glGetUniformLocation(prog, b'uDims'),
                int(dims[0]), int(dims[1]), int(dims[2]))
    # Same participating-medium values as _gpu_ray_field for full volumetric accumulation
    glUniform1f(glGetUniformLocation(prog, b'uVolumeStepMeters'), 0.003)
    glUniform1f(glGetUniformLocation(prog, b'uMediumScattering'),  1.0)
    glUniform1f(glGetUniformLocation(prog, b'uMediumExtinction'),  0.5)
    glUniform1f(glGetUniformLocation(prog, b'uAirDiffuseScatter'),  0.35)
    glUniform1f(glGetUniformLocation(prog, b'uAirSpecularScatter'), 0.65)
    glUniform1f(glGetUniformLocation(prog, b'uAirAnisotropy'),      12.0)
    glUniform1i(glGetUniformLocation(prog, b'uScaleContextCount'),
                int(n_ctx) if n_ctx > 1 else 0)
    glUniform1f(glGetUniformLocation(prog, b'uSpeedOfMedium'),   float(speed_of_medium))
    # Film layer spectral uniforms — driven by FilmStack.layer_specs()
    glUniform1i(glGetUniformLocation(prog, b'uLayerCount'), n_layers)
    for _i, _sp in enumerate(_layer_specs[:n_layers]):
        glUniform1f(glGetUniformLocation(prog, f'uLayerCentersHz[{_i}]'.encode()), float(_sp['center_hz']))
        glUniform1f(glGetUniformLocation(prog, f'uLayerWidthsOct[{_i}]'.encode()), float(_sp['width_oct']))
        glUniform1f(glGetUniformLocation(prog, f'uLayerGains[{_i}]'.encode()),     float(_sp['gain']))
    glUniform1i(glGetUniformLocation(prog, b'uDiagnosticsEnabled'), 0)
    glUniform1i(glGetUniformLocation(prog, b'uSegmentCapture'),     0)
    glUniform1i(glGetUniformLocation(prog, b'uPendingCap'),      _pending_cap)
    glUniform1i(glGetUniformLocation(prog, b'uMode'),            0)  # PASS_FORWARD

    # Cache per-pump uniform locations (source params removed — rays are pre-baked)
    locs = {
        'mode':       glGetUniformLocation(prog, b'uMode'),
        'seed':       glGetUniformLocation(prog, b'uSeed'),
        'batch_size': glGetUniformLocation(prog, b'uBatchSize'),
        'batch_off':  glGetUniformLocation(prog, b'uBatchOffset'),
        'total_rays': glGetUniformLocation(prog, b'uTotalRaysPerSource'),
    }

    # ── Initial dispatch ──────────────────────────────────────────────────
    _n_init = max(128, int(initial_rays))
    glUniform1i(locs['seed'],       1337)
    glUniform1i(locs['total_rays'], _n_init)
    _off = 0
    while _off < _n_init:
        _b  = min(int(dispatch_batch), _n_init - _off)
        _ng = max(1, int(math.ceil(_b / 128.0)))
        glUniform1i(locs['batch_size'], _b)
        glUniform1i(locs['batch_off'],  _off)
        glDispatchCompute(min(_ng, 65535), max(1, int(math.ceil(_ng / 65535))), 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        _off += _b
    glFinish()
    glUseProgram(0)

    print(f"  [gpu_ray_field_prebuilt] {n_tris} tris  {n_src} src  "
          f"dims={dims}  initial={_n_init} rays  "
          f"n_ctx={n_ctx if n_ctx > 1 else 0}", flush=True)
    return {
        'tex_bands':      tex_bands,
        'ssbo':           ssbo,
        'prog':           prog,
        'bmin':           bmin,
        'bmax':           bmax,
        'dims':           tuple(int(d) for d in dims),
        'locs':           locs,
        'n_sources':      n_src,
        'source_buf':     source_records,   # pre-baked (N_rays, 12)
        'dispatch_batch': int(dispatch_batch),
        '_pending_cap':   _pending_cap,
        '_pump_seed':     0,
        'layer_specs':    list(_layer_specs[:n_layers]),
        'n_layers':       n_layers,
    }


def _gpu_pump_prebuilt(state: dict, n_rays: int = 8192) -> None:
    """Dispatch n_rays PASS_FORWARD rays into an existing prebuilt GPU state.

    Rays accumulate additively into state['tex_bands'].
    Call _clear_prebuilt(state) before starting a fresh solve cycle.
    """
    if state is None:
        return
    from OpenGL.GL import (
        GL_SHADER_STORAGE_BARRIER_BIT, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
        GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_READ_WRITE, GL_TEXTURE_3D,
        glUseProgram, glBindBufferBase, glBindImageTexture,
        glUniform1i, glDispatchCompute, glMemoryBarrier,
    )
    prog  = state['prog']
    locs  = state['locs']
    batch = state['dispatch_batch']

    glUseProgram(prog)
    for band_idx, tex in enumerate(state['tex_bands']):
        glBindImageTexture(band_idx, tex, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI)
    for bi, buf in enumerate(state['ssbo']):
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bi, buf)

    glUniform1i(locs['mode'], 0)  # PASS_FORWARD
    seed_base = state['_pump_seed']
    state['_pump_seed'] = (seed_base + 7) & 0x7FFFFFFF

    # Pre-baked rays: dispatch n_rays round-robin over the baked buffer.
    # No per-source loop — the CS reads bdpt_sources[ray_id % uSourceCount].
    glUniform1i(locs['seed'],       seed_base)
    glUniform1i(locs['total_rays'], n_rays)
    offset = 0
    while offset < n_rays:
        _b  = min(batch, n_rays - offset)
        _ng = max(1, int(math.ceil(_b / 128.0)))
        glUniform1i(locs['batch_size'], _b)
        glUniform1i(locs['batch_off'],  offset)
        glDispatchCompute(min(_ng, 65535), max(1, int(math.ceil(_ng / 65535))), 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        offset += _b
    glUseProgram(0)


def _clear_prebuilt(state: dict) -> None:
    """Zero all 4 accumulation textures in a prebuilt GPU state."""
    if state is None:
        return
    from OpenGL.GL import (
        GL_TEXTURE_3D, GL_RED_INTEGER, GL_UNSIGNED_INT,
        glBindTexture, glClearTexImage,
    )
    zero = np.array([0], np.uint32)
    for tex in state['tex_bands']:
        glBindTexture(GL_TEXTURE_3D, tex)
        glClearTexImage(tex, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, zero)
    glBindTexture(GL_TEXTURE_3D, 0)
    state['_pump_seed'] = 0


def _destroy_prebuilt(state: dict) -> None:
    """Release all GL objects owned by a prebuilt GPU state dict."""
    if state is None:
        return
    from OpenGL.GL import (
        glDeleteProgram, glDeleteTextures, glDeleteBuffers,
    )
    glDeleteProgram(state['prog'])
    for tex in state['tex_bands']:
        glDeleteTextures([tex])
    for buf in state['ssbo']:
        glDeleteBuffers(1, [buf])
    state.clear()


def _gpu_readback_prebuilt(tex_bands: list, dims: tuple) -> "np.ndarray":
    """Read all spectral accumulation textures back to CPU and return as float32.

    Returns an (n_bands, dz, dy, dx) float32 array with raw uint32 counts
    cast to float32.  Primarily a diagnostic hook — called at most once per
    N frames from the debug pump loop.
    """
    from OpenGL.GL import (
        GL_TEXTURE_3D, GL_RED_INTEGER, GL_UNSIGNED_INT,
        glBindTexture, glGetTexImage,
    )
    dx, dy, dz = int(dims[0]), int(dims[1]), int(dims[2])
    bands = []
    for tex in tex_bands:
        buf = np.empty((dz, dy, dx), dtype=np.uint32)
        glBindTexture(GL_TEXTURE_3D, tex)
        glGetTexImage(GL_TEXTURE_3D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, buf)
        bands.append(buf.astype(np.float32))
    glBindTexture(GL_TEXTURE_3D, 0)
    return np.stack(bands, axis=0)  # (n_bands, dz, dy, dx)


# ─────────────────────────────────────────────────────────────────────────────


def _load_stage_light_cache(path: str):
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            data = z["bands"]
            bounds = (z["bounds_min"].astype(np.float32),
                      z["bounds_max"].astype(np.float32))
            total_rays = int(z["total_rays"])
        if data.shape[0] < 1:
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
    def __init__(self, target=(0.0, 0.4, 0.65), dist=1.6, elev=14.0, az=22.0,
                 sensor: "SensorSpec | None" = None,
                 lens: "LensSpec | None" = None):
        self.target  = np.array(target, np.float64)
        self.dist    = dist
        self.elev    = elev
        self.az      = az
        self._auto   = 0.18
        self.focal_mm = 35.0
        self.focus_m = 1.6
        self.aperture = 0.0
        self.ca = 0.0
        self.tilt_shift = np.zeros(2, np.float64)
        # PlayerController sets this to override the orbit eye calculation
        self._forced_eye: "np.ndarray | None" = None
        # Sensor and lens specs (may be loaded from YAML)
        self.sensor: SensorSpec = sensor or SensorSpec()
        self.lens:   LensSpec   = lens   or LensSpec()
        # Sync focal_mm from lens spec
        self.focal_mm = self.lens.focal_mm
        # Aperture shape: 0 = circle, >=3 = regular N-gon (e.g. 6 for hexagonal bokeh)
        self.n_blades: int = 0
        self.aperture_rot: float = 0.0   # first-blade angle in radians
        # Full lens transform parameters (compiled by LensTransform.compile())
        self.lens_tilt    = np.zeros(2, np.float64)  # Scheimpflug (x=nod, y=pan) radians
        self.extension_mm: float = 0.0               # barrel extension in mm
        self.gimbal       = np.zeros(2, np.float64)  # (pan, tilt) lens-axis rotation radians
        # Generic optical back — parametric surface for ray→UV projection.
        # Assign a CameraBack instance (FlatBack, SphericalBack, ManifoldBack, …).
        # None = renderer uses the standard perspective projection path.
        self.camera_back: "Optional[object]" = None
        # Generic lens manifold — routes aperture × scene-dir → sensor-dir.
        # Assign a LensManifold instance (e.g. bake_eye_manifold(HUMAN_EYE)).
        # None = standard thin-lens model.
        self.lens_manifold: "Optional[object]" = None
        # Camera software modules (DigitalPositiveSensor etc.)
        # Each entry must implement tick(dt, ctx) with a CameraContext-like interface.
        # The Camera builds a minimal _CamProxy shim so software can write back to it.
        self.software: list = []

        # ── Eye manifold: load from cache or bake on first run ─────────────────
        # The human eye is the player camera — not a special mode, just this camera.
        # The manifold is baked once to disk; subsequent launches load it in ~50 ms.
        import pathlib as _pl
        _eye_cache = _pl.Path.home() / ".spectral_analyzer" / "eye_manifold_v1.npz"
        try:
            from camera_software import (HUMAN_EYE as _EYE,
                                         bake_eye_manifold as _bake,
                                         LensManifold as _LM)
            if _eye_cache.exists():
                self.lens_manifold = _LM.load(str(_eye_cache))
            else:
                print("[Camera] baking eye manifold — first run, one time (~10 s)...",
                      flush=True)
                _eye_cache.parent.mkdir(parents=True, exist_ok=True)
                self.lens_manifold = _bake(_EYE)
                self.lens_manifold.save(str(_eye_cache))
                print(f"[Camera] eye manifold cached → {_eye_cache}", flush=True)
            self.camera_back = _EYE.make_manifold_back(
                self.lens_manifold, res_w=512, res_h=512)
            # Register a PROFILE_MANIFOLD remission entry so the GPU shader
            # can do the transform LUT lookup at binding 7.
            from material_db import RemissionProfile as _RP, _EMISSION_DB as _epdb
            _eye_rp = _RP()
            _eye_rp.set_manifold(self.lens_manifold)
            self._eye_remit_idx = _epdb.register("eye_lens_manifold", _eye_rp)
        except Exception as _eye_err:
            self._eye_remit_idx = -1
            pass   # camera_software absent or bake failed; manifold stays None
        # Digital positive display state (read by Renderer for blit uniforms).
        # Set True by default; DigitalPositiveSensor.tick() refreshes _digital_rgb.
        self._digital_positive: bool = True
        self._digital_rgb: tuple = (1.0, 1.0, 1.0)
        # Active film back chosen by camera software (DigitalPositiveSensor).
        # None = renderer uses its own self.film.  When set, this is a list of
        # layer-spec dicts (same format as FilmStack.layer_specs()) of up to 8
        # entries batched from all registered backs.
        self.digital_film: Optional[list] = None

    def orbit(self, daz, delev):
        self.az   = (self.az + daz) % 360.0
        self.elev = float(np.clip(self.elev + delev, -75.0, 80.0))

    def zoom(self, d):
        self.dist = float(np.clip(self.dist + d, 0.12, 3.0))

    def fov_y_rad(self) -> float:
        # fov_y = 2 * atan(sensor_height_mm/2 / focal_mm)
        # Uses sensor spec so swapping to MF, APS-C etc. changes the field of view.
        h_mm = max(1.0, self.sensor.height_mm)
        return 2.0 * math.atan(h_mm * 0.5 / max(1.0, self.focal_mm))

    def pan(self, right_m: float = 0.0, forward_m: float = 0.0, up_m: float = 0.0):
        fwd = _norm(self.target - self.eye)
        world_up = np.array([0.0, 0.0, 1.0], np.float64)
        right = _norm(np.cross(fwd, world_up))
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0], np.float64)
        flat_fwd = fwd.copy()
        flat_fwd[2] = 0.0
        flat_fwd = _norm(flat_fwd)
        if np.linalg.norm(flat_fwd) < 1e-9:
            flat_fwd = np.array([0.0, 1.0, 0.0], np.float64)
        self.target += right * right_m + flat_fwd * forward_m + world_up * up_m

    def set_lens(self, *, focal_delta: float = 0.0, focus_delta: float = 0.0,
                 aperture_scale: float = 1.0, ca_delta: float = 0.0):
        self.focal_mm = float(np.clip(self.focal_mm + focal_delta, 12.0, 180.0))
        self.focus_m = float(np.clip(self.focus_m + focus_delta, 0.05, 20.0))
        self.aperture = float(np.clip(self.aperture * aperture_scale, 0.0, 0.08))
        self.ca = float(np.clip(self.ca + ca_delta, 0.0, 0.02))

    def tick(self, dt: float = 0.0):
        self.az = (self.az + self._auto) % 360.0
        if self.software:
            # Build a lightweight proxy so software can read/write Camera attrs.
            # We pass self as both item and camera; CameraContext properties
            # fall back to getattr(cam, ...) which resolves to Camera fields.
            try:
                from camera_software import CameraContext
                ctx = CameraContext(self, self)  # type: ignore[arg-type]
            except Exception:
                ctx = self  # type: ignore[assignment]
            for sw in self.software:
                try:
                    sw.tick(dt, ctx)
                except Exception:
                    pass
            # Pull digital display state from the context.
            ds = getattr(ctx, 'digital_sensor', None)
            if ds is not None and getattr(ds, 'enabled', False):
                self._digital_positive = bool(getattr(ds, 'positive', True))
                self._digital_rgb      = (1.0, 1.0, 1.0)
            # Build a single batched spec list of ≤ 8 layers from ALL software
            # modules that expose any film — digital backs, chemical film stacks,
            # acoustic layers, etc.  Every module is a peer; they all work with
            # spectral responses and can share the accumulator slots equally.
            # Resolution priority per module (first match wins):
            #   1. sw.effective_layer_specs()  — multi-back modules (e.g. DigitalPositiveSensor)
            #   2. sw.active_back.layer_specs() — single active-back modules
            #   3. sw.film.layer_specs()        — module with a .film attribute
            #   4. sw.layer_specs()             — module that is itself film-like
            _all_specs: list = []
            for sw in self.software:
                if len(_all_specs) >= 8:
                    break
                if hasattr(sw, 'effective_layer_specs'):
                    _sw_specs = sw.effective_layer_specs()
                elif hasattr(sw, 'active_back') and sw.active_back is not None:
                    _sw_specs = sw.active_back.layer_specs()
                elif hasattr(sw, 'film') and sw.film is not None:
                    _sw_specs = sw.film.layer_specs()
                elif hasattr(sw, 'layer_specs') and callable(sw.layer_specs):
                    _sw_specs = sw.layer_specs()
                else:
                    continue
                for _sp in _sw_specs:
                    if len(_all_specs) >= 8:
                        break
                    _all_specs.append(_sp)
            self.digital_film = _all_specs if _all_specs else None

    # ── Construction from presets ─────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        """Build a Camera from a camera-preset dict (configs/cameras/*.yaml).

        Loads the referenced sensor and lens by name via their own YAML loaders
        so all physical parameters are fully resolved before the Camera is used.
        """
        sensor_name = str(d.get("sensor", "full_frame_35mm"))
        lens_name   = str(d.get("lens",   "standard_35mm"))
        sensor = SensorSpec.load(sensor_name)
        lens   = LensSpec.load(lens_name)
        ts_raw = d.get("tilt_shift", [0.0, 0.0])
        cam = cls(
            target = list(d.get("target", [0.0, 0.40, 0.65])),
            dist   = float(d.get("dist",  1.6)),
            elev   = float(d.get("elev",  14.0)),
            az     = float(d.get("az",    22.0)),
            sensor = sensor,
            lens   = lens,
        )
        cam.focus_m    = float(d.get("focus_m",      1.6))
        cam.aperture   = float(d.get("aperture_mm",  0.0))
        cam.ca         = float(d.get("ca",           0.0))
        cam.tilt_shift = np.array([float(ts_raw[0]), float(ts_raw[1])], np.float64)
        lt_raw = d.get("lens_tilt", [0.0, 0.0])
        cam.lens_tilt    = np.array([float(lt_raw[0]), float(lt_raw[1])], np.float64)
        cam.extension_mm = float(d.get("extension_mm", 0.0))
        gim_raw = d.get("gimbal", [0.0, 0.0])
        cam.gimbal = np.array([float(gim_raw[0]), float(gim_raw[1])], np.float64)
        return cam

    @classmethod
    def load_preset(cls, name: str) -> "Camera":
        """Load a Camera from configs/cameras/<name>.yaml."""
        d = _load_yaml_file(_config_path("cameras", f"{name}.yaml"))
        if not d:
            raise FileNotFoundError(f"Camera preset not found: configs/cameras/{name}.yaml")
        return cls.from_dict(d)

    def copy_optics_from(self, other: "Camera") -> None:
        """Copy all physical optical properties from *other* into self.

        Used by PlayerController when entering IN_CAMERA mode: R.cam inherits
        the placed camera's full physical configuration so that fov_y_rad()
        and any ray-tracing parameters reflect the actual camera being operated.
        Eye position and target are NOT touched — those come from the armature.
        """
        self.sensor     = other.sensor
        self.lens       = other.lens
        self.focal_mm   = other.focal_mm
        self.focus_m    = other.focus_m
        self.aperture   = other.aperture
        self.ca         = other.ca
        self.tilt_shift = other.tilt_shift.copy()
        self.lens_tilt    = getattr(other, 'lens_tilt',    np.zeros(2, np.float64)).copy()
        self.extension_mm = float(getattr(other, 'extension_mm', 0.0))
        self.gimbal       = getattr(other, 'gimbal',       np.zeros(2, np.float64)).copy()

    @property
    def eye(self):
        if self._forced_eye is not None:
            return self._forced_eye
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
        P = _persp(self.fov_y_rad(), aspect, 0.005, 10.0)
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
                 show_mic: bool = True,
                 no_stage: bool = False):
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
        self._frame_step = 1
        self._layers  = [LAYER_OPAQUE, LAYER_ALPHA, LAYER_HIDDEN,
                         LAYER_HIDDEN, LAYER_HIDDEN, LAYER_ALPHA, LAYER_ALPHA,
                         LAYER_ALPHA,   # 7=illum (ray field volume + spotlight cone)
                         LAYER_HIDDEN]  # 8=sensor (path-traced overlay, key-9)
        self.film = _default_stack()  # layered film stack; active_layer drives display
        self.cam = Camera()
        # ── Camera simulator state ─────────────────────────────────────────────
        # _film_negative : the sensor accumulator is a chemical negative
        #                  (bright scene → dark record).  Toggle with N.
        # _enlarger_mode : a second projection through an enlarger lens
        #                  re-inverts the optical flip AND the tone.
        #                  Toggle with E.
        # Optical inversion (rotate180) is always present — it is the physics
        # of the aperture, not a display choice.  The enlarger lens cancels it.
        self._film_negative: bool = True
        self._enlarger_mode: bool = False
        # Digital positive display — true by default for the player camera.
        # When True the blit pipeline outputs a colour-correct, right-way-up
        # positive using CFA spectral white-balance weights (_digital_rgb).
        # _digital_rgb is refreshed each frame from the camera's
        # DigitalPositiveSensor software module via CameraItem.tick() and
        # is passed to the blit shader as the uDigitalRGB uniform.
        self._digital_positive: bool = True
        self._digital_rgb: tuple = (1.0, 1.0, 1.0)  # neutral default; updated by sensor SW
        # Batched layer-spec list built by Camera.tick() from DigitalPositiveSensor backs.
        # None means fall back to self.film.layer_specs() at blit time.
        self.digital_film: Optional[list] = None

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
        self._no_stage = bool(no_stage)
        # ── Per-channel global default selection ─────────────────────────
        # 2D and 3D each pick their own backend (C or GL).  The walker
        # fires registered shaders unconditionally; whatever is left
        # over goes to the global default for each channel selected
        # below.  Defaults: both channels run the C backend.
        self._mode_2d: 'RenderMode' = RenderMode.C
        self._mode_3d: 'RenderMode' = RenderMode.C
        self._global_dispatcher = None  # built by the demo at startup
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
            scene_field      = meta.get('scene_field'),
        )

    def set_sensor_image(self, tex: int, w: int, h: int) -> None:
        """Register a RGBA32F 2-D sensor render texture for overlay display.

        tex=0 disables the overlay.  Old texture is deleted if replaced.
        """
        if self._sensor_tex and self._sensor_tex != tex:
            glDeleteTextures([self._sensor_tex])
        self._sensor_tex = int(tex)
        self._sensor_w   = int(w)
        self._sensor_h   = int(h)

    def attach_sensor_accumulator(self, acc: 'SensorAccumulator') -> None:
        """Hand the Renderer a SensorAccumulator for progressive rendering.

        The accumulator's texture is used as the sensor overlay; the
        Renderer calls acc.tick() every frame via tick_sensor().
        """
        if self._sensor_acc is not None:
            self._sensor_acc.destroy()
        if acc is None or not hasattr(acc, 'tex'):
            return
        self._sensor_acc = acc
        self._sensor_tex = acc.tex
        self._sensor_w   = acc._w
        self._sensor_h   = acc._h
        acc.film = self.film   # keep film params in sync
        self._sensor_last_tick = 0.0
        self.sync_sensor_camera(reset=False)

    def tick_sensor(self) -> None:
        """Call once per frame to advance the sensor camera exposure.

        Forward ray pump (lightfield bands for ILLUM) fires whenever layer 8
        (ILLUM, key-8) OR layer 9 (SENSOR, key-9) is visible — the band
        textures are shared between both modes.

        The sensor-specific backward pass (cadence reset, tick, tex update,
        temporal decay) only runs while layer 9 is visible.
        """
        if self._sensor_acc is None:
            return
        illum_on  = self._layers[7] != LAYER_HIDDEN
        sensor_on = self._layers[8] != LAYER_HIDDEN

        # ── Forward ray pump: feeds _ray_field_bands used by ILLUM and SENSOR ──
        if (illum_on or sensor_on) and self._sensor_acc._auto_advance:
            self._sensor_acc.pump_forward(budget_ms=float(getattr(self, "_ray_budget_ms", 5.0)))

        # ── Sensor-specific: backward pass, exposure cadence, temporal decay ──
        if not sensor_on or not self._sensor_acc._auto_advance:
            return
        fps = float(getattr(self, "_sensor_fps", 30.0))
        on_cadence = False
        if fps > 0.0:
            now = time.monotonic()
            if self._sensor_last_tick <= 0.0 or now - self._sensor_last_tick >= 1.0 / fps:
                self._sensor_last_tick = now
                on_cadence = True
        # fps=0: continuous integration — never reset
        if on_cadence:
            # Cadence reset: new exposure — clear forward field and sensor image.
            self._sensor_acc.clear_field()
            self._sensor_acc.clear()
        # Sensor backward pass.
        self._sensor_acc.tick()
        self._sensor_tex = self._sensor_acc.tex
        self._sensor_acc.apply_decay()

    def sync_sensor_camera(self, reset: bool = False) -> None:
        if self._sensor_acc is None:
            return
        # The sensor accumulator operates in guitar-frame (physics frame:
        # guitar on its back, Z = soundboard normal).  The Camera orbits in
        # world-frame (Z-up, guitar upright on stand).  Apply the inverse
        # model matrix so the sensor receives guitar-frame coordinates and
        # the orbit controls map correctly to what the user sees on screen.
        eye_gf = _transform_points(
            np.asarray(self.cam.eye,    np.float32).reshape(1, 3),
            self._guitar_Minv)[0]
        target_gf = _transform_points(
            np.asarray(self.cam.target, np.float32).reshape(1, 3),
            self._guitar_Minv)[0]

        # Compute initial camera-space basis (guitar frame)
        _fwd = target_gf - eye_gf
        _fwd_l = float(np.linalg.norm(_fwd))
        _fwd = (_fwd / _fwd_l if _fwd_l > 1e-6
                else np.array([0, 0, 1], np.float32)).astype(np.float32)
        _wup = np.array([0, 1, 0], np.float32)
        if abs(float(np.dot(_fwd, _wup))) > 0.97:
            _wup = np.array([0, 0, 1], np.float32)
        _right = np.cross(_fwd, _wup);  _right /= max(float(np.linalg.norm(_right)), 1e-9)
        _up    = np.cross(_right, _fwd); _up    /= max(float(np.linalg.norm(_up)),    1e-9)

        # Build and compile the full lens transform
        try:
            from camera_software import LensTransform as _LT
            _lt = _LT()
            _lt.shift[:]     = self.cam.tilt_shift
            _lt.lens_tilt[:] = getattr(self.cam, 'lens_tilt',    np.zeros(2))
            _lt.extension_mm = float(getattr(self.cam, 'extension_mm', 0.0))
            _lt.gimbal[:]    = getattr(self.cam, 'gimbal',       np.zeros(2))
            payload = _lt.compile(
                focal_mm    = self.cam.focal_mm,
                focus_m     = self.cam.focus_m,
                sensor_h_mm = self.cam.sensor.height_mm,
                right       = _right,
                up          = _up,
                fwd         = _fwd,
            )
            _fov_deg   = math.degrees(2.0 * math.atan(payload['fov_tan']))
            _tilt_sh   = payload['tilt_shift']
            _lens_tilt = payload['lens_tilt']
            _b_right   = payload['right']
            _b_up      = payload['up']
            _b_fwd     = payload['fwd']
        except Exception:
            _fov_deg   = math.degrees(self.cam.fov_y_rad())
            _tilt_sh   = tuple(float(v) for v in self.cam.tilt_shift)
            _lens_tilt = (0.0, 0.0)
            _b_right = _b_up = _b_fwd = None

        self._sensor_acc.set_camera(
            eye_gf, target_gf,
            fov_deg          = _fov_deg,
            aperture_radius  = float(self.cam.aperture),
            focus_dist       = float(self.cam.focus_m),
            ca_factor        = float(self.cam.ca),
            tilt_shift       = _tilt_sh,
            lens_tilt        = _lens_tilt,
            n_blades         = int(getattr(self.cam, 'n_blades', 0)),
            aperture_rot     = float(getattr(self.cam, 'aperture_rot', 0.0)),
            basis_right      = _b_right,
            basis_up         = _b_up,
            basis_fwd        = _b_fwd,
        )
        # ── Eye manifold: ep_tensor carries all data; no set_manifold call needed ─
        _mf = getattr(self.cam, 'lens_manifold', None)
        if _mf is not None:
            try:
                _cb = getattr(self.cam, 'camera_back', None)
                _sphere_back = _cb.back if hasattr(_cb, 'back') else None
                if _sphere_back is not None:
                    _half_fov = float(getattr(_sphere_back, 'max_field_angle',
                                              math.pi / 2.0))
                    _eye_fov_deg = math.degrees(_half_fov) * 2.0
                    _pupil_r = float(getattr(_sphere_back, 'pupil_radius', 0.003))
                    self._sensor_acc.set_camera(
                        eye_gf, target_gf,
                        fov_deg         = _eye_fov_deg,
                        aperture_radius = _pupil_r,
                        focus_dist      = float(self.cam.focus_m),
                        ca_factor       = float(self.cam.ca),
                        tilt_shift      = _tilt_sh,
                        lens_tilt       = _lens_tilt,
                        n_blades        = int(getattr(self.cam, 'n_blades', 0)),
                        aperture_rot    = float(getattr(self.cam, 'aperture_rot', 0.0)),
                        basis_right     = _b_right,
                        basis_up        = _b_up,
                        basis_fwd       = _b_fwd,
                    )
            except Exception:
                pass
        if reset:
            self.reset_sensor()

    def camera_hud(self) -> str:
        acc = self._sensor_acc
        rays = 0
        if acc is not None and acc._active:
            rays = int(acc._w) * int(acc._rows_per_frame) * int(acc._samples_per_pixel)
        return (
            f"CAM {self.cam.focal_mm:05.1f}mm  focus {self.cam.focus_m:04.2f}m  "
            f"ap {self.cam.aperture*1000.0:04.1f}mm  CA {self.cam.ca:0.4f}  "
            f"shift {self.cam.tilt_shift[0]:+.2f},{self.cam.tilt_shift[1]:+.2f}  "
            f"{rays} rays/frame"
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
        self._p_sensor_blit = _prog((_SENSOR_BLIT_VS, GL_VERTEX_SHADER),
                                    (_SENSOR_BLIT_FS, GL_FRAGMENT_SHADER))
        self._sensor_tex = 0
        self._sensor_w   = 0
        self._sensor_h   = 0
        self._sensor_acc: 'SensorAccumulator | None' = None
        self._sensor_fps = 0.0
        self._sensor_last_tick = 0.0

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
        if self._no_stage:
            # Blank / room-mode startup: create empty stage geometry.
            # _stage_mesh() is preserved but not called here; call sites in
            # cache-key hashing and _gpu_ray_field are also guarded.
            empty6 = np.zeros((0, 6), np.float32)
            self._stage_vao, _, self._stage_n = _vao(empty6, [(0,3,24,0),(1,3,24,12)], GL_STATIC_DRAW)
        else:
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

        # ── Glass bell jar + opaque skirt marking the AMR sim boundary ────────
        # Transform guitar-frame sim bounds into world space for display.
        _wb_min, _wb_max = _transform_bounds((self._bmin, self._bmax), self._guitar_M)
        print(f"[belljar] guitar bmin={self._bmin} bmax={self._bmax}", flush=True)
        print(f"[belljar] world  wb_min={_wb_min} wb_max={_wb_max}", flush=True)
        _bell = _build_sim_belljar_world(_wb_min, _wb_max)
        print(f"[belljar] bell verts={len(_bell)}  skirt check z_top={_wb_min[2]:.4f}", flush=True)
        self._glass_vao, _, self._glass_n = _vao(_bell, [(0,3,24,0),(1,3,24,12)], GL_STATIC_DRAW)
        _skirt = _build_sim_skirt_world(_wb_min, _wb_max)
        print(f"[belljar] skirt verts={len(_skirt)}", flush=True)
        if len(_skirt) > 0:
            self._skirt_vao, _, self._skirt_n = _vao(_skirt, [(0,3,24,0),(1,3,24,12)], GL_STATIC_DRAW)
        else:
            self._skirt_vao, self._skirt_n = None, 0
        _wire = _sim_box_wireframe_world(_wb_min, _wb_max)
        self._simbox_edge_vao, _, self._simbox_edge_n = _vao(
            _wire.reshape(-1, 3), [(0,3,12,0)], GL_STATIC_DRAW)
        print(f"[belljar] glass_n={self._glass_n} skirt_n={self._skirt_n} edge_n={self._simbox_edge_n}", flush=True)

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
        """Prepare AMR rendering metadata.

        Formerly built a GPU TBO for brute-force per-fragment IDW, but that
        path scales as O(n_cells × fragments × march_steps) and is not viable.
        Pressure is now scattered to the viz 3-D texture via
        coevolver_get_pressure_field_uniform() (O(n_cells)) before each
        frame upload, so _amr_tbo_tex stays None and the shader uses the
        existing 3-D texture ray-march (uAMRMode=0).
        """
        n = int(self.info.get('n_cells_amr', 0))
        if n == 0:
            return
        self._amr_n_cells = n
        min_dx = float(self.info.get('amr_min_dx', self.info.get('min_dx', 0.005)))
        self._amr_eps2     = float((min_dx * 0.5) ** 2)
        p = float(self.info.get('amr_idw_power', 2.0))
        self._amr_half_pow = p * 0.5
        # _amr_tbo_tex left None → _up_pressure uses 3-D texture path

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
            step = max(0, int(self._frame_step))
            if step > 0:
                self._cursor = (self._cursor + step) % max(1, len(self._frames))

    def advance_frame(self, step: int = 1) -> None:
        if self._frames:
            self._cursor = (self._cursor + max(1, int(step))) % max(1, len(self._frames))

    def advance_sensor(self, strips: int = 1) -> None:
        if self._sensor_acc is not None and self._layers[8] != LAYER_HIDDEN:
            self._sensor_acc.tick(max(1, int(strips)))
            self._sensor_tex = self._sensor_acc.tex

    def reset_sensor(self) -> None:
        if self._sensor_acc is not None:
            self._sensor_acc.clear()
            self._sensor_tex = self._sensor_acc.tex
            self._sensor_last_tick = 0.0

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
        d = np.asarray(disp, dtype=np.float32)
        if d.ndim != 2:
            d = np.reshape(d, (int(d.shape[0]), -1))
        Nx, Ny = d.shape
        if Nx <= 0 or Ny <= 0:
            return
        mx = float(np.abs(d).max()) + 1e-9
        plate_origin = self.info.get('plate_origin')
        dx = float(self.info.get('plate_dx', self.info.get('dx', DX)))
        gx_min = float(plate_origin[0]) if plate_origin is not None else float(self.info['gx_min'])
        gy_min = float(plate_origin[1]) if plate_origin is not None else float(self.info['gy_min'])
        gx = np.clip((self._plate_vxy[:, 0] - gx_min) / dx - 0.5, 0.0, float(Nx - 1))
        gy = np.clip((self._plate_vxy[:, 1] - gy_min) / dx - 0.5, 0.0, float(Ny - 1))
        if Nx < 2 or Ny < 2:
            i0 = np.clip(np.rint(gx).astype(np.int32), 0, Nx - 1)
            j0 = np.clip(np.rint(gy).astype(np.int32), 0, Ny - 1)
            vals = d[i0, j0].astype(np.float32)
        else:
            i0 = np.clip(np.floor(gx).astype(np.int32), 0, Nx - 2)
            j0 = np.clip(np.floor(gy).astype(np.int32), 0, Ny - 2)
            i1 = i0 + 1
            j1 = j0 + 1
            fx = gx - i0
            fy = gy - j0
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

        # ── Layer 9 OPAQUE: full-screen sensor render only, no scene ──────────
        if self._layers[8] == LAYER_OPAQUE:
            if self._sensor_tex:
                glDisable(GL_DEPTH_TEST)
                glDepthMask(GL_FALSE)
                glDisable(GL_BLEND)
                glUseProgram(self._p_sensor_blit)
                _blit_texs = self._sensor_acc._textures if self._sensor_acc else [self._sensor_tex]
                for _i, _t in enumerate(_blit_texs):
                    glActiveTexture(GL_TEXTURE0 + _i)
                    glBindTexture(GL_TEXTURE_2D, _t)
                    glUniform1i(glGetUniformLocation(self._p_sensor_blit, f'uSensorLayer{_i}'.encode()), _i)
                # Use digital back spec list when in digital positive mode,
                # otherwise fall through to the default film stack.
                _dig_specs = self.digital_film if (self._digital_positive and self.digital_film) else None
                _specs = _dig_specs if _dig_specs is not None else self.film.layer_specs()
                for _i, _sp in enumerate(_specs[:len(_blit_texs)]):
                    glUniform3f(glGetUniformLocation(self._p_sensor_blit, f'uLayerDark{_i}'.encode()),  *_sp['dark_rgb'])
                    glUniform3f(glGetUniformLocation(self._p_sensor_blit, f'uLayerLight{_i}'.encode()), *_sp['light_rgb'])
                    glUniform2f(glGetUniformLocation(self._p_sensor_blit, f'uLayerTone{_i}'.encode()),
                                float(_sp['shadow_point']), float(_sp['highlight_point']))
                glUniform1i(glGetUniformLocation(self._p_sensor_blit, b'uLayerCount'), len(_blit_texs))
                if _dig_specs is not None:
                    # Digital back: use neutral ISO 1.0 / gamma 2.2; active_layer not meaningful here
                    glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uExposure'), 1.0)
                    glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uGamma'),    2.2)
                else:
                    _fl = self.film.active_layer
                    glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uExposure'),  float(_fl.iso))
                    glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uGamma'),     float(_fl.gamma))
                glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uAlpha'),     1.0)
                # Camera simulator — three orthogonal physical stages:
                #   Optical inversion: the aperture inverts the image; the raw sensor
                #     texture is naturally upside-down (no correction applied by default).
                #     The enlarger adds a corrective second lens that cancels the flip.
                #   Tone: negative film records bright-as-dark.
                #     Enlarger re-exposes onto positive paper, cancelling the inversion.
                #   Digital positive: bypasses both film stages; always outputs a
                #     correctly-oriented colour positive using CFA spectral weights.
                if self._digital_positive:
                    _disp_rotate180 = False  # handled by uDigitalPositive path in shader
                    _disp_negative  = False
                else:
                    _disp_rotate180 = self._enlarger_mode
                    _disp_negative  = self._film_negative ^ self._enlarger_mode
                glUniform1i(glGetUniformLocation(self._p_sensor_blit, b'uRotate180'), int(_disp_rotate180))
                glUniform1i(glGetUniformLocation(self._p_sensor_blit, b'uNegative'),  int(_disp_negative))
                glUniform1i(glGetUniformLocation(self._p_sensor_blit, b'uDigitalPositive'), int(self._digital_positive))
                glUniform3f(glGetUniformLocation(self._p_sensor_blit, b'uDigitalRGB'), *self._digital_rgb)
                _dt = float(self._sensor_acc._decay_total) if self._sensor_acc else 1.0
                glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uDecayTotal'), _dt)
                glDrawArrays(GL_TRIANGLES, 0, 3)
                for _i in range(len(_blit_texs)):
                    glActiveTexture(GL_TEXTURE0 + _i)
                    glBindTexture(GL_TEXTURE_2D, 0)
                glActiveTexture(GL_TEXTURE0)
                glEnable(GL_DEPTH_TEST)
                glDepthMask(GL_TRUE)
                glEnable(GL_BLEND)
            # Always suppress scene in OPAQUE mode regardless of sensor tex
            return

        frame = self._cur()

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

        # ── Scene field uniforms (sensor-independent, ray-traced, static) ─────
        # Derived once from SceneFieldIntegration; uploaded to every shading
        # program that accepts uSceneRgb / uSceneIndirectRatio.  Defaults to
        # neutral (pure white ambient, zero indirect) when not yet computed.
        _sf = (self._ray_lighting.scene_field
               if self._ray_lighting is not None else None)
        if _sf is not None:
            _sf_rgb = _sf.rgb.tolist()                              # [r,g,b] float32
            _total  = float(_sf.total_power.sum()) + 1e-9
            _sf_ir  = float(_sf.surface_indirect.sum()) / _total   # [0, 1]
        else:
            _sf_rgb = [1.0, 1.0, 1.0]
            _sf_ir  = 0.0
        for _sf_prog in (self._p_body, self._p_ray_surface):
            glUseProgram(_sf_prog)
            glUniform3f(glGetUniformLocation(_sf_prog, b'uSceneRgb'), *_sf_rgb)
            glUniform1f(glGetUniformLocation(_sf_prog, b'uSceneIndirectRatio'), _sf_ir)
        glUseProgram(0)

        if frame is not None:
            self._up_pressure(frame.pressure)
            self._up_plate(frame.plate)
            for si, disp in enumerate(frame.strings):
                if disp is not None and len(disp) >= 2:
                    self._up_string(si, disp)

        # ── 1. Back plate (opaque dark wood) ──────────────────────────────────
        a_body = self._a(0)
        if a_body > 0:
            glUseProgram(self._p_body)
            _mvp(self._p_body, MVP_guitar, MV_guitar)
            glUniform3f(glGetUniformLocation(self._p_body, b'uLightV'), *light_v)
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

        # ── 2. Side walls (semi-transparent mahogany, both faces) ──────────────
        if a_body > 0:
            glDepthMask(GL_FALSE)
            glUseProgram(self._p_body)
            _mvp(self._p_body, MVP_guitar, MV_guitar)
            glUniform3f(glGetUniformLocation(self._p_body, b'uLightV'), *light_v)
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

        # ── 3. Outline rings ───────────────────────────────────────────────────
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

        # ── 4a. Stage ─────────────────────────────────────────────────────────
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
            # Cull back-faces only for the outer stage walls so the camera
            # can move outside the room without seeing inside faces.
            # All other geometry (skirt, bell jar, guitar) is drawn two-sided.
            glEnable(GL_CULL_FACE)
            glBindVertexArray(self._stage_vao)
            glDrawArrays(GL_TRIANGLES, 0, self._stage_n)
            glDisable(GL_CULL_FACE)
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        1.0, 0.90, 0.62, a_stage * 0.55)
            glBindVertexArray(self._lamp_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, self._lamp_n)

        # ── 4a-skirt. Opaque base skirt below the sim boundary ────────────────
        if a_stage > 0 and self._skirt_vao is not None and self._skirt_n > 0:
            glUseProgram(self._p_body)
            _mvp(self._p_body, MVP, MV)
            glUniform3f(glGetUniformLocation(self._p_body, b'uLightV'), *light_v)
            glUniform4f(glGetUniformLocation(self._p_body, b'uColor'),
                        0.22, 0.20, 0.18, 0.88)
            glUniform3f(glGetUniformLocation(self._p_body, b'uInnerColor'),
                        0.22, 0.20, 0.18)
            glUniform1f(glGetUniformLocation(self._p_body, b'uAmbient'), 0.30)
            glUniform1f(glGetUniformLocation(self._p_body, b'uSpecStrength'), 0.02)
            glUniform1f(glGetUniformLocation(self._p_body, b'uShininess'), 8.0)
            glUniform1f(glGetUniformLocation(self._p_body, b'uGrain'), 0.12)
            glBindVertexArray(self._skirt_vao)
            glDrawArrays(GL_TRIANGLES, 0, self._skirt_n)

        # ── 4a-glass. Semi-transparent bell jar (exact sim boundary) ──────────
        if a_stage > 0:
            glDepthMask(GL_FALSE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUseProgram(self._p_body)
            _mvp(self._p_body, MVP, MV)
            glUniform3f(glGetUniformLocation(self._p_body, b'uLightV'), *light_v)
            glUniform4f(glGetUniformLocation(self._p_body, b'uColor'),
                        0.86, 0.92, 0.96, 0.11)
            glUniform3f(glGetUniformLocation(self._p_body, b'uInnerColor'),
                        0.86, 0.92, 0.96)
            glUniform1f(glGetUniformLocation(self._p_body, b'uAmbient'), 0.14)
            glUniform1f(glGetUniformLocation(self._p_body, b'uSpecStrength'), 0.92)
            glUniform1f(glGetUniformLocation(self._p_body, b'uShininess'), 200.0)
            glUniform1f(glGetUniformLocation(self._p_body, b'uGrain'), 0.0)
            glBindVertexArray(self._glass_vao)
            glDrawArrays(GL_TRIANGLES, 0, self._glass_n)
            # Crisp wireframe tracing the exact inner sim boundary
            glBlendFunc(GL_ONE, GL_ONE)
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP)
            glLineWidth(1.0)
            glUniform4f(glGetUniformLocation(self._p_line, b'uColor'),
                        0.68, 0.86, 1.00, 0.55)
            glBindVertexArray(self._simbox_edge_vao)
            glDrawArrays(GL_LINES, 0, self._simbox_edge_n)
            glDepthMask(GL_TRUE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # ── 4b. Cached room light volume ─────────────────────────────────────
        if self._baseline_light_bands is not None:
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
            _lspecs = self.film.layer_specs()
            glUniform1i(glGetUniformLocation(self._p_march, b'uLayerCount'), len(_lspecs))
            for _bi, _sp in enumerate(_lspecs):
                glUniform3f(glGetUniformLocation(self._p_march, f'uLayerDark{_bi}'.encode()),  *_sp['dark_rgb'])
                glUniform3f(glGetUniformLocation(self._p_march, f'uLayerLight{_bi}'.encode()), *_sp['light_rgb'])
            for bi, btex in enumerate(self._baseline_light_bands):
                glUniform1i(glGetUniformLocation(self._p_march, f'uLayer{bi}'.encode()), 2 + bi)
                glActiveTexture(GL_TEXTURE2 + bi)
                glBindTexture(GL_TEXTURE_3D, btex)
            glActiveTexture(GL_TEXTURE0)
            glBindVertexArray(self._quad_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
            for bi in range(len(self._baseline_light_bands)):
                glActiveTexture(GL_TEXTURE2 + bi)
                glBindTexture(GL_TEXTURE_3D, 0)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)

        # ── 4c. Guitar-added ray field volume (ILLUM layer — key 8) ───────────
        if self._a(7) > 0 and self._ray_field_bands is not None:
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
            _lspecs = self.film.layer_specs()
            glUniform1i(glGetUniformLocation(self._p_march, b'uLayerCount'), len(_lspecs))
            for _bi, _sp in enumerate(_lspecs):
                glUniform3f(glGetUniformLocation(self._p_march, f'uLayerDark{_bi}'.encode()),  *_sp['dark_rgb'])
                glUniform3f(glGetUniformLocation(self._p_march, f'uLayerLight{_bi}'.encode()), *_sp['light_rgb'])
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self._mask_tex)
            # Bind N spectral layer textures to units 2+
            for bi, btex in enumerate(self._ray_field_bands):
                glUniform1i(glGetUniformLocation(self._p_march, f'uLayer{bi}'.encode()), 2 + bi)
                glActiveTexture(GL_TEXTURE2 + bi)
                glBindTexture(GL_TEXTURE_3D, btex)
            glActiveTexture(GL_TEXTURE0)
            glBindVertexArray(self._quad_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
            # Unbind all layer textures
            for bi in range(len(self._ray_field_bands)):
                glActiveTexture(GL_TEXTURE2 + bi)
                glBindTexture(GL_TEXTURE_3D, 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)

        # ── 4d. Ray-lit surface pass (ILLUM layer — body geometry with band textures) ──
        # Draws back plate + side walls through _p_ray_surface so fragment positions
        # are looked up in the spectral band volumes to get per-fragment irradiance.
        # Requires at least one of dyn or base band textures to be available.
        _has_dyn  = self._a(7) > 0 and self._ray_field_bands is not None and len(self._ray_field_bands) > 0
        _has_base = self._baseline_light_bands is not None and len(self._baseline_light_bands) > 0
        if _has_dyn or _has_base:
            a_illum = self._a(7) if _has_dyn else 0.72  # use layer alpha when dyn; fixed when base only
            _dyn_bands  = self._ray_field_bands        if _has_dyn  else []
            _base_bands = self._baseline_light_bands   if _has_base else []
            _n_dyn  = len(_dyn_bands)
            _n_base = len(_base_bands)
            # Dynamic field bounds / world-to-grid
            if _has_dyn and self._ray_field_bounds is not None:
                _drf_min = self._ray_field_bounds[0]
                _drf_max = self._ray_field_bounds[1]
            else:
                _drf_min = self._bmin
                _drf_max = self._bmax
            # Baseline field bounds
            if _has_base and self._baseline_light_bounds is not None:
                _blb_min = self._baseline_light_bounds[0]
                _blb_max = self._baseline_light_bounds[1]
            else:
                _blb_min, _blb_max = _stage_light_bounds()
            # Texture unit layout:
            #   units 2 .. 8   : dynamic band layers  (uLayer0..6, up to 7)
            #   units 9 .. 15  : baseline band layers (uBaseLayer0..6, up to 7)
            # GL 3.3 guarantees >= 16 fragment texture units (0-15); units 0,1
            # may be used by other passes so we stay within [2-15].
            _MAX_BANDS_PER_FIELD = 7
            _BASE_UNIT_OFFSET = 9  # first unit for baseline bands

            # Blend the spectral delta additively on top of the Phong surface.
            # GL_LEQUAL lets us re-draw at identical depth values that _p_body already wrote.
            # GL_DEPTH_MASK(GL_FALSE) ensures we don't clobber the existing depth buffer.
            # GL_SRC_ALPHA / GL_ONE: destination keeps full brightness, source alpha
            # controls the irradiance contribution strength.
            glDepthFunc(GL_LEQUAL)
            glDepthMask(GL_FALSE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)

            glUseProgram(self._p_ray_surface)
            # Matrices — guitar body lives in guitar-frame
            _mvp(self._p_ray_surface, MVP_guitar, MV_guitar, self._guitar_M)
            glUniform3f(glGetUniformLocation(self._p_ray_surface, b'uLightV'), *light_v)

            # Band texture unit bindings
            for _bi, _btex in enumerate(_dyn_bands[:_MAX_BANDS_PER_FIELD]):
                glUniform1i(glGetUniformLocation(self._p_ray_surface, f'uLayer{_bi}'.encode()), 2 + _bi)
                glActiveTexture(GL_TEXTURE2 + _bi)
                glBindTexture(GL_TEXTURE_3D, _btex)
            for _bi, _btex in enumerate(_base_bands[:_MAX_BANDS_PER_FIELD]):
                glUniform1i(glGetUniformLocation(self._p_ray_surface, f'uBaseLayer{_bi}'.encode()), _BASE_UNIT_OFFSET + _bi)
                glActiveTexture(GL_TEXTURE0 + _BASE_UNIT_OFFSET + _bi)
                glBindTexture(GL_TEXTURE_3D, _btex)

            glUniform1i(glGetUniformLocation(self._p_ray_surface, b'uLayerCount'),
                        max(min(_n_dyn, _MAX_BANDS_PER_FIELD), min(_n_base, _MAX_BANDS_PER_FIELD), 1))
            # Dynamic volume bounds + identity world-to-grid (volumes are in world space)
            glUniformMatrix4fv(glGetUniformLocation(self._p_ray_surface, b'uWorldToGrid'),
                               1, GL_TRUE, np.eye(4, dtype=np.float32))
            glUniform3f(glGetUniformLocation(self._p_ray_surface, b'uBoxMin'), *_drf_min)
            glUniform3f(glGetUniformLocation(self._p_ray_surface, b'uBoxMax'), *_drf_max)
            # Baseline volume bounds
            glUniformMatrix4fv(glGetUniformLocation(self._p_ray_surface, b'uBaseWorldToGrid'),
                               1, GL_TRUE, np.eye(4, dtype=np.float32))
            glUniform3f(glGetUniformLocation(self._p_ray_surface, b'uBaseBoxMin'), *_blb_min)
            glUniform3f(glGetUniformLocation(self._p_ray_surface, b'uBaseBoxMax'), *_blb_max)

            glUniform1i(glGetUniformLocation(self._p_ray_surface, b'uUseRayField'),  int(_has_dyn))
            glUniform1i(glGetUniformLocation(self._p_ray_surface, b'uUseBaseField'), int(_has_base))
            glUniform1f(glGetUniformLocation(self._p_ray_surface, b'uRayExposure'),  self._ray_exposure)
            glUniform1f(glGetUniformLocation(self._p_ray_surface, b'uBaseExposure'),
                        self._baseline_scale_for_display() if _has_base else 1.0)
            glUniform1f(glGetUniformLocation(self._p_ray_surface, b'uRayGamma'), self._ray_gamma)
            glUniform1f(glGetUniformLocation(self._p_ray_surface, b'uAmbient'), 0.0)   # unused in delta mode
            glUniform1f(glGetUniformLocation(self._p_ray_surface, b'uSpecStrength'), 0.0)
            glUniform1f(glGetUniformLocation(self._p_ray_surface, b'uShininess'), 1.0)
            glUniform1f(glGetUniformLocation(self._p_ray_surface, b'uGrain'), 0.25)

            # Back plate — alpha governs irradiance contribution weight
            glUniform4f(glGetUniformLocation(self._p_ray_surface, b'uColor'),
                        0.16, 0.035, 0.018, a_illum * 0.82)
            glUniform3f(glGetUniformLocation(self._p_ray_surface, b'uInnerColor'),
                        0.64, 0.38, 0.18)
            glBindVertexArray(self._back_vao)
            glDrawArrays(GL_TRIANGLE_FAN, 0, self._back_n)

            # Side walls
            glUniform4f(glGetUniformLocation(self._p_ray_surface, b'uColor'),
                        0.18, 0.035, 0.018, a_illum * 0.65)
            glUniform3f(glGetUniformLocation(self._p_ray_surface, b'uInnerColor'),
                        0.72, 0.44, 0.21)
            glUniform1f(glGetUniformLocation(self._p_ray_surface, b'uGrain'), 0.65)
            glBindVertexArray(self._wall_vao)
            glDrawElements(GL_TRIANGLES, self._wall_n_idx, GL_UNSIGNED_INT, None)

            # Unbind all band textures
            for _bi in range(min(_n_dyn, _MAX_BANDS_PER_FIELD)):
                glActiveTexture(GL_TEXTURE2 + _bi)
                glBindTexture(GL_TEXTURE_3D, 0)
            for _bi in range(min(_n_base, _MAX_BANDS_PER_FIELD)):
                glActiveTexture(GL_TEXTURE0 + _BASE_UNIT_OFFSET + _bi)
                glBindTexture(GL_TEXTURE_3D, 0)
            glActiveTexture(GL_TEXTURE0)
            glUseProgram(0)

            # Restore default depth/blend state
            glDepthFunc(GL_LESS)
            glDepthMask(GL_TRUE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # ── 5. Top plate surface (displaced, heatmap) ─────────────────────────
        a_plate = self._a(1)
        if a_plate > 0 and self._plate_n > 0:
            glUseProgram(self._p_plate)
            _mvp(self._p_plate, MVP_guitar, MV_guitar)
            glUniform3f(glGetUniformLocation(self._p_plate, b'uLightV'), *light_v)
            glUniform1f(glGetUniformLocation(self._p_plate, b'uAlpha'), a_plate)
            glUniform1f(glGetUniformLocation(self._p_plate, b'uColorMix'),
                        1.0 if self._plate_mode in (0, 2) else 0.0)
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

        # ── 7. Strings ─────────────────────────────────────────────────────────
        a_str = self._a(3)
        if a_str > 0:
            glUseProgram(self._p_body)
            _mvp(self._p_body, MVP_guitar, MV_guitar)
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
            glBlendFunc(GL_ONE, GL_ONE)
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP_guitar)
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
                    glBindVertexArray(self._str_env_hi_vaos[si])
                    glDrawArrays(GL_LINE_STRIP, 0, self._str_n[si])
                    glBindVertexArray(self._str_env_lo_vaos[si])
                    glDrawArrays(GL_LINE_STRIP, 0, self._str_n[si])
                else:
                    glBindVertexArray(self._str_vaos[si])
                    glDrawArrays(GL_LINE_STRIP, 0, self._str_n[si])
            glBindVertexArray(0)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # ── 8. Bridge markers ──────────────────────────────────────────────────
        a_mk = self._a(4)
        if a_mk > 0 and self._mk_n > 0:
            glUseProgram(self._p_line)
            _mvp(self._p_line, MVP_guitar)
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

        # ── Layer 9 (index 8): sensor render overlay ──────────────────────────
        # Composites the path-traced RGBA32F image on top of the 3-D scene.
        # OPAQUE → full opacity (α=1); ALPHA → semi-transparent (α=0.72).
        if self._sensor_tex and self._layers[8] != LAYER_HIDDEN:
            sensor_alpha = 1.0 if self._layers[8] == LAYER_OPAQUE else 0.72
            glDisable(GL_DEPTH_TEST)
            glDepthMask(GL_FALSE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUseProgram(self._p_sensor_blit)
            _blit_texs = self._sensor_acc._textures if self._sensor_acc else [self._sensor_tex]
            for _i, _t in enumerate(_blit_texs):
                glActiveTexture(GL_TEXTURE0 + _i)
                glBindTexture(GL_TEXTURE_2D, _t)
                glUniform1i(glGetUniformLocation(self._p_sensor_blit, f'uSensorLayer{_i}'.encode()), _i)
            _dig_specs = self.digital_film if (self._digital_positive and self.digital_film) else None
            _specs = _dig_specs if _dig_specs is not None else self.film.layer_specs()
            for _i, _sp in enumerate(_specs[:len(_blit_texs)]):
                glUniform3f(glGetUniformLocation(self._p_sensor_blit, f'uLayerDark{_i}'.encode()),  *_sp['dark_rgb'])
                glUniform3f(glGetUniformLocation(self._p_sensor_blit, f'uLayerLight{_i}'.encode()), *_sp['light_rgb'])
                glUniform2f(glGetUniformLocation(self._p_sensor_blit, f'uLayerTone{_i}'.encode()),
                            float(_sp['shadow_point']), float(_sp['highlight_point']))
            glUniform1i(glGetUniformLocation(self._p_sensor_blit, b'uLayerCount'), len(_blit_texs))
            if _dig_specs is not None:
                glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uExposure'), 1.0)
                glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uGamma'),    2.2)
            else:
                _fl = self.film.active_layer
                glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uExposure'),  float(_fl.iso))
                glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uGamma'),     float(_fl.gamma))
            glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uAlpha'),     sensor_alpha)
            if self._digital_positive:
                _disp_rotate180 = False  # uDigitalPositive path handles full 180° in shader
                _disp_negative  = False
            else:
                _disp_rotate180 = self._enlarger_mode
                _disp_negative  = self._film_negative ^ self._enlarger_mode
            glUniform1i(glGetUniformLocation(self._p_sensor_blit, b'uRotate180'), int(_disp_rotate180))
            glUniform1i(glGetUniformLocation(self._p_sensor_blit, b'uNegative'),  int(_disp_negative))
            glUniform1i(glGetUniformLocation(self._p_sensor_blit, b'uDigitalPositive'), int(self._digital_positive))
            glUniform3f(glGetUniformLocation(self._p_sensor_blit, b'uDigitalRGB'), *self._digital_rgb)
            _dt = float(self._sensor_acc._decay_total) if self._sensor_acc else 1.0
            glUniform1f(glGetUniformLocation(self._p_sensor_blit, b'uDecayTotal'), _dt)
            glDrawArrays(GL_TRIANGLES, 0, 3)
            for _i in range(len(_blit_texs)):
                glActiveTexture(GL_TEXTURE0 + _i)
                glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)




# ─────────────────────────────────────────────────────────────────────────────
# Sensor accumulator — progressive per-frame path-tracing
# ─────────────────────────────────────────────────────────────────────────────

class SensorAccumulator:
    """Owns the persistent RGBA32F sensor texture and dispatches a few rows of
    the GPU sensor compute shader each frame, accumulating indefinitely.

    The blit shader divides .rgb / .a to get the running per-pixel mean, so
    quality improves continuously without re-clearing.

    Usage:
        acc = SensorAccumulator(ssbo_tris, ssbo_nodes, ssbo_ids, n_tris,
                                n_bvh_nodes, tex_bands, bmin, bmax, dims,
                                camera_eye, camera_target,
                                sensor_w, sensor_h,
                                rows_per_frame=16, fov_deg=52.0,
                                max_bounces=RAY_MAX_BOUNCES,
                                ray_field_scale=6.0, ray_field_gamma=0.55,
                                vol_alpha=1.0, vol_steps=64)
        acc.tick()          # call once per frame; returns immediately
        tex = acc.tex       # current RGBA32F texture
        acc.destroy()       # free GL resources when done
    """

    def __init__(self, ssbo_tris, ssbo_nodes, ssbo_ids, n_tris, n_bvh_nodes,
                 tex_bands, bmin, bmax, dims,
                 camera_eye, camera_target,
                 sensor_w: int, sensor_h: int,
                 rows_per_frame: int = 16,
                 fov_deg: float = 52.0,
                 max_bounces: int = RAY_MAX_BOUNCES,
                 aperture_radius: float = 0.0,
                 focus_dist: float = 1.6,
                 ca_factor: float = 0.0,
                 tilt_shift: tuple[float, float] = (0.0, 0.0),
                 n_blades: int = 0,
                 aperture_rot: float = 0.0,
                 ray_field_scale: float = 6.0,
                 ray_field_gamma: float = 0.55,
                 vol_alpha: float = 1.0,
                 vol_steps: int = 64):
        import math
        from OpenGL.GL import (glGenTextures, glBindTexture, glTexImage2D,
                               glTexParameteri, GL_TEXTURE_2D, GL_RGBA32F,
                               GL_RGBA, GL_FLOAT, GL_TEXTURE_MIN_FILTER,
                               GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S,
                               GL_TEXTURE_WRAP_T, GL_NEAREST, GL_CLAMP_TO_EDGE,
                               glClearTexImage, glDeleteTextures, glDeleteProgram)
        from OpenGL.GL import glGetProgramiv, GL_LINK_STATUS

        self._w = int(sensor_w)
        self._h = int(sensor_h)
        self._rows_per_frame = max(1, int(rows_per_frame))
        self._row_off = 0
        self._pass   = 0   # how many full image passes completed
        self._frame  = 0   # total tick() calls
        self._active = True
        self._auto_advance = True
        self._samples_per_pixel = 1
        self._sprinkle = True

        self._eye = np.zeros(3, np.float32)
        self._right = np.array([1,0,0], np.float32)
        self._up = np.array([0,1,0], np.float32)
        self._fwd = np.array([0,0,1], np.float32)
        self._fov_tan = 1.0
        self._aspect  = float(self._w) / float(max(1, self._h))
        self._aperture_radius = max(0.0, float(aperture_radius))
        self._focus_dist = max(0.01, float(focus_dist))
        self._ca_factor = max(0.0, float(ca_factor))
        self._tilt_shift = np.asarray(tilt_shift, np.float32).ravel()[:2]
        if self._tilt_shift.size < 2:
            self._tilt_shift = np.zeros(2, np.float32)
        self._lens_tilt  = np.zeros(2, np.float32)   # Scheimpflug tilt (x=nod, y=pan)
        self._n_blades = max(0, int(n_blades))
        self._aperture_rot = float(aperture_rot)
        self.set_camera(camera_eye, camera_target, fov_deg=fov_deg)
        self._bmin    = np.asarray(bmin, np.float32)
        self._bmax    = np.asarray(bmax, np.float32)
        self._dims    = dims
        self._tex_bands   = tex_bands
        self._ssbo_tris   = int(ssbo_tris)
        self._ssbo_nodes  = int(ssbo_nodes)
        self._ssbo_ids    = int(ssbo_ids)
        self._n_tris      = int(n_tris)
        self._n_bvh_nodes = int(n_bvh_nodes)
        self._rfs  = float(ray_field_scale)
        self._rfg  = float(ray_field_gamma)
        self._va   = float(vol_alpha)
        self._vs   = int(vol_steps)
        self._max_bounces = max(1, int(max_bounces))
        # Forward emission state — populated by update_sources() after construction
        self._fwd_prog = 0
        self._ssbo_sources_fwd = 0
        self._ssbo_seg_dummy = 0
        self._ssbo_counter_fwd = 0
        self._sources_list: list = []        # [(pos, dir, n_rays, spectrum), ...]
        self._source_records_np: np.ndarray | None = None
        self._n_sources = 0
        self._dispatch_batch_fwd = 512
        self._air_ds = 0.35
        self._air_ss = 0.65
        self._air_an = 12.0
        self._medium_extinction = 0.5
        self._fwd_frame = 0
        # ── Streaming source worker ──────────────────────────────────────────
        self._source_worker: _SourceWorker | None = None
        self._dispatch_ring: list = []    # list of (si, batch_offset, batch_size)
        self._ring_cursor: int = 0        # rotating position in dispatch_ring
        self.film: FilmStack = _default_stack()

        # ── Temporal decay state ──────────────────────────────────────────────
        # half_life: 0.0 = stable (no decay); >0 = seconds for display to reach
        # 50% brightness after source goes silent.
        self.half_life: float = 5.0
        # Accumulated product of per-frame decay factors applied during silence.
        # Reset to 1.0 when a source dispatch occurs.  Sent to the blit shader
        # as uDecayTotal so the display fades at the correct half-life rate
        # without any per-pixel time tracking on the GPU.
        self._decay_total: float = 1.0
        # Flag set by pump_forward() when at least one forward-ray batch is
        # dispatched; consumed (and cleared) by apply_decay().
        self._source_dispatched: bool = False
        self._decay_prog: int = 0
        _decay_compiled = _prog((_GPU_DECAY_CS, GL_COMPUTE_SHADER))
        from OpenGL.GL import glGetProgramInfoLog
        if glGetProgramiv(_decay_compiled, GL_LINK_STATUS):
            self._decay_prog = _decay_compiled
        else:
            log = glGetProgramInfoLog(_decay_compiled)
            print(f"  [SensorAccumulator] decay shader LINK FAILED:\n{log}", flush=True)
            glDeleteProgram(_decay_compiled)

        # One RGBA32F accumulation texture per film layer — held independently in memory
        _n_out = max(1, len(tex_bands))
        self._textures: list[int] = [int(t) for t in glGenTextures(_n_out)]
        for _t in self._textures:
            glBindTexture(GL_TEXTURE_2D, _t)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self._w, self._h, 0,
                         GL_RGBA, GL_FLOAT, None)
            for _p, _v in [(GL_TEXTURE_MIN_FILTER, GL_NEAREST),
                           (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                           (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                           (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)]:
                glTexParameteri(GL_TEXTURE_2D, _p, _v)
            glClearTexImage(_t, 0, GL_RGBA, GL_FLOAT, np.zeros(4, np.float32))
        glBindTexture(GL_TEXTURE_2D, 0)

        # Compile compute shader once
        from OpenGL.GL import glGetProgramInfoLog
        self._prog = _prog((_GPU_SENSOR_CS, GL_COMPUTE_SHADER))
        if not glGetProgramiv(self._prog, GL_LINK_STATUS):
            log = glGetProgramInfoLog(self._prog)
            print(f"  [SensorAccumulator] compute shader LINK FAILED:\n{log}", flush=True)
            glDeleteProgram(self._prog)
            self._prog = 0
            self._active = False
        else:
            print(f"  [SensorAccumulator] compute shader linked OK  {self._w}x{self._h}  rows_per_frame={self._rows_per_frame}", flush=True)
        self._gx = max(1, int(math.ceil(self._w / 16.0)))

    @property
    def tex(self) -> int:
        """Return the output texture for the currently active film layer."""
        if not self._textures:
            return 0
        names = list(self.film.layers.keys())
        try:
            idx = names.index(self.film.active)
        except ValueError:
            idx = 0
        return self._textures[min(idx, len(self._textures) - 1)]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def update_sources(self, sources_list: list, source_records_np: np.ndarray,
                       dispatch_batch: int = 512,
                       air_diffuse_scatter: float = 0.35,
                       air_specular_scatter: float = 0.65,
                       air_anisotropy: float = 12.0,
                       medium_extinction: float = 0.5) -> None:
        """Store source data for per-frame forward re-emission.

        sources_list  — list of (pos, dir, n_rays, spectrum) tuples
        source_records_np — shape (N,12) float32 SSBO data for BDPT connections
        """
        from OpenGL.GL import (glGenBuffers, glBindBuffer, glBufferData,
                               glBufferSubData, glDeleteBuffers,
                               GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW, GL_DYNAMIC_DRAW)
        self._sources_list = list(sources_list)
        self._source_records_np = np.ascontiguousarray(source_records_np, np.float32)
        self._n_sources = len(self._sources_list)
        self._dispatch_batch_fwd = max(128, int(dispatch_batch))
        self._air_ds = float(air_diffuse_scatter)
        self._air_ss = float(air_specular_scatter)
        self._air_an = float(air_anisotropy)
        self._medium_extinction = float(medium_extinction)
        # (Re-)upload source records SSBO
        if self._ssbo_sources_fwd:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_sources_fwd)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self._source_records_np.nbytes,
                         self._source_records_np, GL_STATIC_DRAW)
        else:
            self._ssbo_sources_fwd = int(glGenBuffers(1))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_sources_fwd)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self._source_records_np.nbytes,
                         self._source_records_np, GL_STATIC_DRAW)
        # Dummy segment SSBO (segment capture disabled during live re-emit)
        if not self._ssbo_seg_dummy:
            self._ssbo_seg_dummy = int(glGenBuffers(1))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_seg_dummy)
            glBufferData(GL_SHADER_STORAGE_BUFFER, 64, None, GL_DYNAMIC_DRAW)
        # Counter SSBO
        if not self._ssbo_counter_fwd:
            self._ssbo_counter_fwd = int(glGenBuffers(1))
            counter = np.zeros(4, np.uint32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_counter_fwd)
            glBufferData(GL_SHADER_STORAGE_BUFFER, counter.nbytes, counter, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        # Compile forward shader if not already done
        if not self._fwd_prog:
            from OpenGL.GL import glGetProgramiv, GL_LINK_STATUS
            self._fwd_prog = _prog((_GPU_RAY_FIELD_CS, GL_COMPUTE_SHADER))
            if not glGetProgramiv(self._fwd_prog, GL_LINK_STATUS):
                from OpenGL.GL import glDeleteProgram
                glDeleteProgram(self._fwd_prog)
                self._fwd_prog = 0
                print("  [SensorAccumulator] forward shader LINK FAILED", flush=True)
            else:
                print(f"  [SensorAccumulator] forward shader linked OK  "
                      f"{self._n_sources} sources", flush=True)

    # ------------------------------------------------------------------
    def tick_forward(self) -> None:
        """Clear band textures and re-emit forward rays from all sources
        with fresh random seeds.  Called every sensor-frame cadence tick
        so the field is never stale."""
        if not self._fwd_prog or not self._sources_list:
            return
        from OpenGL.GL import (
            glUseProgram, glBindBufferBase, glBindBuffer, glBufferSubData,
            GL_SHADER_STORAGE_BUFFER,
            glBindImageTexture, GL_READ_WRITE, GL_R32UI,
            glUniform1i, glUniform1f, glUniform3f, glUniform4f,
            glGetUniformLocation, glUniform3i,
            glDispatchCompute, glMemoryBarrier,
            GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
            GL_SHADER_STORAGE_BARRIER_BIT,
            GL_TEXTURE_FETCH_BARRIER_BIT,
            glClearTexImage, glBindTexture, GL_TEXTURE_3D,
            GL_RED_INTEGER, GL_UNSIGNED_INT,
        )
        import math
        prog = self._fwd_prog
        # Just emit — never clear here.  Band textures accumulate forever until
        # an explicit clear_field() call on cadence reset.
        glUseProgram(prog)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._ssbo_tris)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self._ssbo_nodes)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self._ssbo_ids)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self._ssbo_seg_dummy)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self._ssbo_counter_fwd)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self._ssbo_sources_fwd)
        # Bind band images for atomic-add streaming accumulation
        for band_idx, tex in enumerate(self._tex_bands):
            glBindImageTexture(band_idx, tex, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI)

        def _u1i(n, v): glUniform1i(glGetUniformLocation(prog, n), int(v))
        def _u1f(n, v): glUniform1f(glGetUniformLocation(prog, n), float(v))
        def _u3f(n, x, y, z): glUniform3f(glGetUniformLocation(prog, n), float(x), float(y), float(z))
        def _u4f(n, x, y, z, w): glUniform4f(glGetUniformLocation(prog, n), float(x), float(y), float(z), float(w))

        _u1i(b'uTriCount',          self._n_tris)
        _u1i(b'uNodeCount',         self._n_bvh_nodes)
        _u1i(b'uSegmentCap',        0)
        _u1i(b'uSegmentStride',     4)
        _u1i(b'uSourceCount',       self._n_sources)
        _u1i(b'uMaxBounces',        self._max_bounces)
        _u1i(b'uMode',              0)
        _u1i(b'uDiagnosticsEnabled',0)
        _u1i(b'uSegmentCapture',    0)
        _u3f(b'uBoxMin',            *self._bmin)
        _u3f(b'uBoxMax',            *self._bmax)
        glUniform3i(glGetUniformLocation(prog, b'uDims'),
                    int(self._dims[0]), int(self._dims[1]), int(self._dims[2]))
        _u1f(b'uVolumeStepMeters',  0.003)
        _u1f(b'uMediumScattering',  1.0)
        _u1f(b'uMediumExtinction',  0.5)
        _u1f(b'uAirDiffuseScatter', self._air_ds)
        _u1f(b'uAirSpecularScatter',self._air_ss)
        _u1f(b'uAirAnisotropy',     self._air_an)
        _specs = self.film.layer_specs()
        glUniform1i(glGetUniformLocation(prog, b'uLayerCount'), len(_specs))
        for _i, _sp_layer in enumerate(_specs):
            glUniform1f(glGetUniformLocation(prog, f'uLayerCentersHz[{_i}]'.encode()), _sp_layer['center_hz'])
            glUniform1f(glGetUniformLocation(prog, f'uLayerWidthsOct[{_i}]'.encode()), _sp_layer['width_oct'])
            glUniform1f(glGetUniformLocation(prog, f'uLayerGains[{_i}]'.encode()),     _sp_layer['gain'])
        # 4. Dispatch one batch per source with fresh per-frame seed
        base_seed = self._fwd_frame * 6271 + 1337
        batch = self._dispatch_batch_fwd
        for si, _src_entry in enumerate(self._sources_list):
            sp, sd, n_rays, spectrum = _src_entry[0], _src_entry[1], _src_entry[2], _src_entry[3]
            _src_radius  = float(_src_entry[4]) if len(_src_entry) > 4 else 0.0
            _sr_row      = self._source_records_np[min(si, len(self._source_records_np) - 1)]
            _dir_model   = int(round(float(_sr_row[7])))
            _dir_param   = float(_sr_row[11])
            seed = base_seed + si * 104729
            _u1i(b'uSeed',              seed & 0x7FFFFFFF)
            _u1i(b'uTotalRaysPerSource',max(1, int(n_rays)))
            _u3f(b'uSrcPos',            *np.asarray(sp, np.float32).ravel()[:3])
            _u3f(b'uSrcDir',            *np.asarray(sd, np.float32).ravel()[:3])
            spec = np.asarray(spectrum, np.float32).ravel()
            _u4f(b'uSrcSpectrum',       float(spec[0]), float(spec[1]),
                                        float(spec[2]), float(spec[3]))
            glUniform1f(glGetUniformLocation(prog, b'uSrcRadius'),   _src_radius)
            glUniform1i(glGetUniformLocation(prog, b'uSrcDirModel'), _dir_model)
            glUniform1f(glGetUniformLocation(prog, b'uSrcDirParam'), _dir_param)
            offset = 0
            while offset < n_rays:
                b_ = min(batch, n_rays - offset)
                _u1i(b'uBatchSize',   b_)
                _u1i(b'uBatchOffset', offset)
                n_groups = max(1, int(math.ceil(b_ / 128.0)))
                gx = min(n_groups, 65535)
                gy = max(1, int(math.ceil(n_groups / gx)))
                glDispatchCompute(gx, gy, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                                GL_TEXTURE_FETCH_BARRIER_BIT)
                offset += b_
        glUseProgram(0)
        self._fwd_frame += 1

    def set_manifold(self, manifold) -> None:
        """No-op: manifold data is packed into EmissionProfileBuf by build_gpu_tensor()."""
        self._has_manifold = True

    def set_camera(self, camera_eye, camera_target, *, fov_deg: float,
                   aperture_radius: float | None = None,
                   focus_dist: float | None = None,
                   ca_factor: float | None = None,
                   tilt_shift=None,
                   lens_tilt=None,
                   n_blades: int | None = None,
                   aperture_rot: float | None = None,
                   basis_right=None,
                   basis_up=None,
                   basis_fwd=None) -> None:
        eye    = np.asarray(camera_eye,    np.float32).ravel()[:3]
        target = np.asarray(camera_target, np.float32).ravel()[:3]
        if basis_right is not None and basis_up is not None and basis_fwd is not None:
            # Accept pre-compiled gimbal-rotated basis (from LensTransform.compile)
            right = np.asarray(basis_right, np.float32).ravel()[:3]
            up    = np.asarray(basis_up,    np.float32).ravel()[:3]
            fwd   = np.asarray(basis_fwd,   np.float32).ravel()[:3]
        else:
            fwd    = target - eye
            fwd_l  = float(np.linalg.norm(fwd))
            fwd    = (fwd / fwd_l if fwd_l > 1e-6 else np.array([0,0,1], np.float32)).astype(np.float32)
            world_up = np.array([0,1,0], np.float32)
            if abs(float(np.dot(fwd, world_up))) > 0.97:
                world_up = np.array([0,0,1], np.float32)
            right = np.cross(fwd, world_up); right /= max(float(np.linalg.norm(right)), 1e-9)
            up    = np.cross(right, fwd);    up    /= max(float(np.linalg.norm(up)),    1e-9)
        self._eye   = eye
        self._right = right.astype(np.float32)
        self._up    = up.astype(np.float32)
        self._fwd   = fwd
        self._fov_tan = float(math.tan(math.radians(float(fov_deg)) * 0.5))
        if aperture_radius is not None:
            self._aperture_radius = max(0.0, float(aperture_radius))
        if focus_dist is not None:
            self._focus_dist = max(0.01, float(focus_dist))
        if ca_factor is not None:
            self._ca_factor = max(0.0, float(ca_factor))
        if tilt_shift is not None:
            ts = np.asarray(tilt_shift, np.float32).ravel()[:2]
            if ts.size >= 2:
                self._tilt_shift = ts.astype(np.float32)
        if lens_tilt is not None:
            lt = np.asarray(lens_tilt, np.float32).ravel()[:2]
            if lt.size >= 2:
                self._lens_tilt = lt.astype(np.float32)
        if n_blades is not None:
            self._n_blades = max(0, int(n_blades))
        if aperture_rot is not None:
            self._aperture_rot = float(aperture_rot)

    def set_lens(self, *, aperture_radius: float | None = None,
                 focus_dist: float | None = None,
                 ca_factor: float | None = None,
                 tilt_shift=None,
                 lens_tilt=None,
                 n_blades: int | None = None,
                 aperture_rot: float | None = None) -> None:
        if aperture_radius is not None:
            self._aperture_radius = max(0.0, float(aperture_radius))
        if focus_dist is not None:
            self._focus_dist = max(0.01, float(focus_dist))
        if ca_factor is not None:
            self._ca_factor = max(0.0, float(ca_factor))
        if tilt_shift is not None:
            ts = np.asarray(tilt_shift, np.float32).ravel()[:2]
            if ts.size >= 2:
                self._tilt_shift = ts.astype(np.float32)
        if lens_tilt is not None:
            lt = np.asarray(lens_tilt, np.float32).ravel()[:2]
            if lt.size >= 2:
                self._lens_tilt = lt.astype(np.float32)
        if n_blades is not None:
            self._n_blades = max(0, int(n_blades))
        if aperture_rot is not None:
            self._aperture_rot = float(aperture_rot)

    def tick(self, strips: int = 1):
        """Dispatch one strip of rows into the accumulation texture.

        Safe to call every frame — each call does at most `rows_per_frame`
        rows of 16-pixel workgroups (a small fraction of the full image).
        """
        if not self._active or not self._prog:
            return
        from OpenGL.GL import (glUseProgram, glBindBufferBase,
                               GL_SHADER_STORAGE_BUFFER,
                               glActiveTexture, glBindTexture,
                               GL_TEXTURE_2D, GL_TEXTURE_3D,
                               GL_TEXTURE0, glUniform1i, glUniform1f, glUniform4f,
                               glUniform2i, glUniform3i, glUniform3f,
                               glBindImageTexture, GL_READ_WRITE, GL_RGBA32F,
                               glDispatchCompute, glMemoryBarrier,
                               GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
                               glGetUniformLocation)
        import math

        strips = max(1, int(strips))
        for _strip in range(strips):
            self._dispatch_once()

    def _dispatch_once(self):
        from OpenGL.GL import (glUseProgram, glBindBufferBase,
                               GL_SHADER_STORAGE_BUFFER,
                               glActiveTexture, glBindTexture,
                               GL_TEXTURE_2D, GL_TEXTURE_3D,
                               GL_TEXTURE0, glUniform1i, glUniform1f, glUniform2f, glUniform4f,
                               glUniform2i, glUniform3i, glUniform3f,
                               glBindImageTexture, GL_READ_WRITE, GL_RGBA32F,
                               glDispatchCompute, glMemoryBarrier,
                               GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
                               glGetUniformLocation)
        import math

        glUseProgram(self._prog)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._ssbo_tris)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self._ssbo_nodes)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self._ssbo_ids)
        for _i, _t in enumerate(self._tex_bands):
            glActiveTexture(GL_TEXTURE0 + _i)
            glBindTexture(GL_TEXTURE_3D, _t)
        glActiveTexture(GL_TEXTURE0)
        for _i in range(len(self._tex_bands)):
            glUniform1i(glGetUniformLocation(self._prog, f'uFwdLayer{_i}'.encode()), _i)
        for _i, _t in enumerate(self._textures):
            glBindImageTexture(_i, _t, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)

        def _u1i(n, v): glUniform1i(glGetUniformLocation(self._prog, n), int(v))
        def _u1f(n, v): glUniform1f(glGetUniformLocation(self._prog, n), float(v))
        def _u3f(n, x,y,z): glUniform3f(glGetUniformLocation(self._prog, n), float(x),float(y),float(z))
        def _u4f(n, x,y,z,w): glUniform4f(glGetUniformLocation(self._prog, n), float(x),float(y),float(z),float(w))

        _u1i(b'uTriCount',        self._n_tris)
        _u1i(b'uNodeCount',       self._n_bvh_nodes)
        _u1i(b'uSamplesPerPixel', max(1, int(self._samples_per_pixel)))
        _u1i(b'uMaxBounces',      self._max_bounces)
        _u1i(b'uVolSteps',        self._vs)
        _u1f(b'uRayFieldScale',   self._rfs)
        _u1f(b'uRayFieldGamma',   self._rfg)
        _u1f(b'uVolAlpha',        self._va)
        _u1f(b'uAirDiffuseScatter',  self._air_ds)
        _u1f(b'uAirSpecularScatter', self._air_ss)
        _u1f(b'uAirAnisotropy',      self._air_an)
        _u1f(b'uMediumExtinction',   getattr(self, '_medium_extinction', 0.5))
        _sensor_specs = self.film.layer_specs()
        glUniform1i(glGetUniformLocation(self._prog, b'uLayerCount'), len(_sensor_specs))
        for _i, _sp in enumerate(_sensor_specs):
            glUniform3f(glGetUniformLocation(self._prog, f'uLayerDark{_i}'.encode()),  *_sp['dark_rgb'])
            glUniform3f(glGetUniformLocation(self._prog, f'uLayerLight{_i}'.encode()), *_sp['light_rgb'])
        _u3f(b'uCamEye',          *self._eye)
        _u3f(b'uCamRight',        *self._right)
        _u3f(b'uCamUp',           *self._up)
        _u3f(b'uCamFwd',          *self._fwd)
        _u1f(b'uCamFovTan',       self._fov_tan)
        _u1f(b'uCamAspect',       self._aspect)
        _u1f(b'uApertureRadius',  self._aperture_radius)
        _u1f(b'uFocusDist',       self._focus_dist)
        _u1f(b'uCAFactor',        self._ca_factor)
        glUniform2f(glGetUniformLocation(self._prog, b'uTiltShift'),
                    float(self._tilt_shift[0]), float(self._tilt_shift[1]))
        _u1i(b'uNBlades',     getattr(self, '_n_blades', 0))
        _u1f(b'uApertureRot', getattr(self, '_aperture_rot', 0.0))
        _lt = getattr(self, '_lens_tilt', None)
        if _lt is not None and len(_lt) >= 2:
            glUniform2f(glGetUniformLocation(self._prog, b'uLensTilt'),
                        float(_lt[0]), float(_lt[1]))
        else:
            glUniform2f(glGetUniformLocation(self._prog, b'uLensTilt'), 0.0, 0.0)
        _u3f(b'uBoxMin',          *self._bmin)
        _u3f(b'uBoxMax',          *self._bmax)
        glUniform3i(glGetUniformLocation(self._prog, b'uDims'),
                    int(self._dims[0]), int(self._dims[1]), int(self._dims[2]))
        glUniform2i(glGetUniformLocation(self._prog, b'uSensorSize'),
                    self._w, self._h)

        # Unique seed per strip per pass to avoid correlation
        seed = 314159 + self._row_off * 1009 + self._pass * 65537 + self._frame * 1000003
        _u1i(b'uSeed',      seed & 0x7FFFFFFF)
        _u1i(b'uRowOffset', self._row_off)

        rows_this = min(self._rows_per_frame, self._h - self._row_off)
        random_mode = bool(getattr(self, "_sprinkle", True))
        dispatch_px = max(1, min(self._w * self._h,
                                 int(self._w * max(1, self._rows_per_frame))))
        _u1i(b'uRandomPixels', 1 if random_mode else 0)
        _u1i(b'uDispatchPixelCount', dispatch_px if random_mode else self._w * rows_this)
        if random_mode:
            groups = max(1, int(math.ceil(dispatch_px / 256.0)))
            gx = min(groups, 65535)
            gy = max(1, int(math.ceil(groups / gx)))
        else:
            gx = self._gx
            gy = max(1, int(math.ceil(rows_this / 16.0)))
        glDispatchCompute(gx, gy, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        if self._frame == 0:
            mode = "sprinkle" if random_mode else "rows"
            print(f"  [SensorAccumulator] first dispatch: mode={mode} gx={gx} gy={gy} pixels={dispatch_px} rows={rows_this}", flush=True)

        if random_mode:
            self._pass += dispatch_px // max(1, self._w * self._h)
        else:
            self._row_off += rows_this
        if not random_mode and self._row_off >= self._h:
            self._row_off = 0
            self._pass   += 1

        for _i in range(len(self._tex_bands)):
            glActiveTexture(GL_TEXTURE0 + _i)
            glBindTexture(GL_TEXTURE_3D, 0)
        glActiveTexture(GL_TEXTURE0)
        glUseProgram(0)
        self._frame += 1

    # ------------------------------------------------------------------
    def apply_decay(self) -> None:
        import math as _math, time as _time
        now = _time.monotonic()
        if not hasattr(self, '_decay_last_wall') or self._decay_last_wall <= 0.0:
            self._decay_last_wall = now
            return
        dt_s = now - self._decay_last_wall
        self._decay_last_wall = now

        self._source_dispatched = False

        if self.half_life <= 0.0:
            self._decay_total = 1.0
            return

        f = _math.exp(-_math.log(2.0) * dt_s / max(self.half_life, 1e-6))
        f = float(max(0.0, min(1.0, f)))

        # Decay the GPU texture every frame, wall-clock timed.
        if self._decay_prog and self._textures and self._active:
            from OpenGL.GL import (glUseProgram, glBindImageTexture, GL_READ_WRITE,
                                   GL_RGBA32F, glDispatchCompute, glMemoryBarrier,
                                   GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
                                   glGetUniformLocation, glUniform1f, GL_FALSE)
            gx = max(1, int(_math.ceil(self._w / 16.0)))
            gy = max(1, int(_math.ceil(self._h / 16.0)))
            glUseProgram(self._decay_prog)
            glUniform1f(glGetUniformLocation(self._decay_prog, b'uDecayFactor'), f)
            for _t in self._textures:
                glBindImageTexture(0, _t, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
                glDispatchCompute(gx, gy, 1)
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            glUseProgram(0)

        # _decay_total is kept at 1.0 — the texture itself carries the decayed
        # state, so the blit must not double-multiply by a compounded factor.
        self._decay_total = 1.0

    def destroy(self):
        from OpenGL.GL import glDeleteTextures, glDeleteProgram, glDeleteBuffers
        if self._source_worker is not None:
            self._source_worker.stop()
            self._source_worker.join(timeout=1.0)
            self._source_worker = None
        if self._textures:
            glDeleteTextures(self._textures)
            self._textures = []
        if self._prog:
            glDeleteProgram(self._prog)
            self._prog = 0
        if self._fwd_prog:
            glDeleteProgram(self._fwd_prog)
            self._fwd_prog = 0
        if self._decay_prog:
            glDeleteProgram(self._decay_prog)
            self._decay_prog = 0
        for _b in [self._ssbo_sources_fwd, self._ssbo_seg_dummy, self._ssbo_counter_fwd]:
            if _b:
                glDeleteBuffers(1, [_b])
        self._ssbo_sources_fwd = self._ssbo_seg_dummy = self._ssbo_counter_fwd = 0
        self._active = False

    def clear(self):
        """Clear all per-layer RGBA32F sensor accumulation images."""
        from OpenGL.GL import glBindTexture, glClearTexImage, GL_TEXTURE_2D, GL_RGBA, GL_FLOAT
        if not self._textures:
            return
        for _t in self._textures:
            glBindTexture(GL_TEXTURE_2D, _t)
            glClearTexImage(_t, 0, GL_RGBA, GL_FLOAT, np.zeros(4, np.float32))
        glBindTexture(GL_TEXTURE_2D, 0)
        self._row_off = 0
        self._pass = 0
        # Reset decay accumulator so the freshly cleared display starts bright.
        self._decay_total = 1.0
        # _frame is intentionally NOT reset — keeps the seed unique across every
        # display dispatch regardless of cadence resets

    def clear_field(self):
        """Clear the forward band textures — call only on cadence reset."""
        from OpenGL.GL import (glBindTexture, glClearTexImage, GL_TEXTURE_3D,
                               GL_RED_INTEGER, GL_UNSIGNED_INT)
        zero = np.array([0], dtype=np.uint32)
        for tex in self._tex_bands:
            glBindTexture(GL_TEXTURE_3D, tex)
            glClearTexImage(tex, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, zero)
        glBindTexture(GL_TEXTURE_3D, 0)
        # _fwd_frame is intentionally NOT reset — seeds must stay unique across resets

    # ------------------------------------------------------------------
    # Streaming source pipeline
    # ------------------------------------------------------------------

    def notify_frame(self, frame, info: dict, outline: np.ndarray,
                     body_h: float, paths: list, active_strings,
                     refresh_rays: int, sensor_light_rays: int,
                     stage_light_emitters: int) -> None:
        """Post a scene-change job to the background source worker.
        Returns immediately — no GL calls, no blocking."""
        if self._source_worker is None:
            self._source_worker = _SourceWorker()
            self._source_worker.start()
        self._source_worker.notify(
            frame=frame,
            info=info,
            outline=outline,
            body_h=body_h,
            paths=paths,
            active_strings=active_strings,
            refresh_rays=refresh_rays,
            sensor_light_rays=sensor_light_rays,
            stage_light_emitters=stage_light_emitters,
        )

    def _drain_source_upload(self) -> bool:
        """Check if the source worker has new data ready; if so, upload to GPU.
        Called at the start of pump_forward().  Returns True if sources changed."""
        if self._source_worker is None:
            return False
        try:
            result = self._source_worker.result_q.pop()
        except IndexError:
            return False
        self._update_sources_gl(result["sources_list"], result["source_records_np"])
        return True

    def _update_sources_gl(self, sources_list: list,
                            source_records_np: np.ndarray) -> None:
        """Upload new source data to the GPU SSBO and rebuild the dispatch ring.
        All GL calls — must run on the GL thread.  Fast: only glBufferData."""
        from OpenGL.GL import (glGenBuffers, glBindBuffer, glBufferData,
                               GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW, GL_DYNAMIC_DRAW)
        self._sources_list = list(sources_list)
        self._source_records_np = np.ascontiguousarray(source_records_np, np.float32)
        self._n_sources = len(self._sources_list)

        if self._ssbo_sources_fwd:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_sources_fwd)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self._source_records_np.nbytes,
                         self._source_records_np, GL_STATIC_DRAW)
        else:
            self._ssbo_sources_fwd = int(glGenBuffers(1))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_sources_fwd)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self._source_records_np.nbytes,
                         self._source_records_np, GL_STATIC_DRAW)
        if not self._ssbo_seg_dummy:
            self._ssbo_seg_dummy = int(glGenBuffers(1))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_seg_dummy)
            glBufferData(GL_SHADER_STORAGE_BUFFER, 64, None, GL_DYNAMIC_DRAW)
        if not self._ssbo_counter_fwd:
            self._ssbo_counter_fwd = int(glGenBuffers(1))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._ssbo_counter_fwd)
            glBufferData(GL_SHADER_STORAGE_BUFFER, 16, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        # Compile forward shader once; never again.
        if not self._fwd_prog:
            from OpenGL.GL import glGetProgramiv, GL_LINK_STATUS
            self._fwd_prog = _prog((_GPU_RAY_FIELD_CS, GL_COMPUTE_SHADER))
            if not glGetProgramiv(self._fwd_prog, GL_LINK_STATUS):
                from OpenGL.GL import glDeleteProgram
                glDeleteProgram(self._fwd_prog)
                self._fwd_prog = 0
                print("  [SensorAccumulator] forward shader LINK FAILED", flush=True)

        self._rebuild_dispatch_ring()

    def _rebuild_dispatch_ring(self) -> None:
        """Build the rotating dispatch token list from current sources.
        Token = (source_index, batch_offset, batch_size).
        The ring is traversed continuously across frames — no per-call rebuild."""
        batch = self._dispatch_batch_fwd
        ring = []
        for si, _s in enumerate(self._sources_list):
            n_rays = int(_s[2])
            off = 0
            while off < n_rays:
                b = min(batch, n_rays - off)
                ring.append((si, off, b))
                off += b
        self._dispatch_ring = ring
        self._ring_cursor = 0

    def pump_forward(self, budget_ms: float = 5.0) -> int:
        """Fire forward ray dispatches for up to budget_ms wall-clock time.

        Drains the source-worker upload queue first (fast SSBO swap), then
        fires dispatch tokens from the rotating ring.  A single glMemoryBarrier
        is issued after all dispatches, not between them — the GPU executes
        each CS kernel as soon as its predecessor vacates the hardware queue.

        Returns the number of ray batches dispatched.
        """
        # 1. Absorb any freshly computed source data (non-blocking deque pop).
        self._drain_source_upload()

        if not self._fwd_prog or not self._sources_list or not self._dispatch_ring:
            return 0

        from OpenGL.GL import (
            glUseProgram, glBindBufferBase, GL_SHADER_STORAGE_BUFFER,
            glBindImageTexture, GL_READ_WRITE, GL_R32UI,
            glUniform1i, glUniform1f, glUniform3f, glUniform4f,
            glGetUniformLocation, glUniform3i,
            glDispatchCompute, glMemoryBarrier,
            GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
            GL_SHADER_STORAGE_BARRIER_BIT,
            GL_TEXTURE_FETCH_BARRIER_BIT,
        )
        import math as _math

        prog = self._fwd_prog
        glUseProgram(prog)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._ssbo_tris)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self._ssbo_nodes)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self._ssbo_ids)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self._ssbo_seg_dummy)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self._ssbo_counter_fwd)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self._ssbo_sources_fwd)
        for band_idx, tex in enumerate(self._tex_bands):
            glBindImageTexture(band_idx, tex, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI)

        def _u1i(n, v): glUniform1i(glGetUniformLocation(prog, n), int(v))
        def _u1f(n, v): glUniform1f(glGetUniformLocation(prog, n), float(v))
        def _u3f(n, x, y, z): glUniform3f(glGetUniformLocation(prog, n), float(x), float(y), float(z))
        def _u4f(n, x, y, z, w): glUniform4f(glGetUniformLocation(prog, n), float(x), float(y), float(z), float(w))

        # Uniform state that doesn't change within a pump call
        _u1i(b'uTriCount',           self._n_tris)
        _u1i(b'uNodeCount',          self._n_bvh_nodes)
        _u1i(b'uSegmentCap',         0)
        _u1i(b'uSegmentStride',      4)
        _u1i(b'uSourceCount',        self._n_sources)
        _u1i(b'uMaxBounces',         self._max_bounces)
        _u1i(b'uMode',               0)
        _u1i(b'uDiagnosticsEnabled', 0)
        _u1i(b'uSegmentCapture',     0)
        _u3f(b'uBoxMin',             *self._bmin)
        _u3f(b'uBoxMax',             *self._bmax)
        glUniform3i(glGetUniformLocation(prog, b'uDims'),
                    int(self._dims[0]), int(self._dims[1]), int(self._dims[2]))
        _u1f(b'uVolumeStepMeters',   0.003)
        _u1f(b'uMediumScattering',   1.0)
        _u1f(b'uMediumExtinction',   0.5)
        _u1f(b'uAirDiffuseScatter',  self._air_ds)
        _u1f(b'uAirSpecularScatter', self._air_ss)
        _u1f(b'uAirAnisotropy',      self._air_an)
        _specs = self.film.layer_specs()
        glUniform1i(glGetUniformLocation(prog, b'uLayerCount'), len(_specs))
        for _i, _sp_layer in enumerate(_specs):
            glUniform1f(glGetUniformLocation(prog, f'uLayerCentersHz[{_i}]'.encode()), _sp_layer['center_hz'])
            glUniform1f(glGetUniformLocation(prog, f'uLayerWidthsOct[{_i}]'.encode()), _sp_layer['width_oct'])
            glUniform1f(glGetUniformLocation(prog, f'uLayerGains[{_i}]'.encode()),     _sp_layer['gain'])

        # 2. Fire tokens from the rotating ring until budget exceeded.
        ring      = self._dispatch_ring
        n_tokens  = len(ring)
        cursor    = self._ring_cursor
        seed_base = self._fwd_frame * 6271 + 1337
        dispatched = 0
        t0 = time.perf_counter()

        # Cache per-source uniforms so we only re-upload when the source changes.
        last_si = -1
        while True:
            si, batch_off, batch_size = ring[cursor]
            # Update per-source uniforms only when source index changes.
            if si != last_si:
                _src_entry = self._sources_list[si]
                sp, sd, n_rays, spectrum = _src_entry[0], _src_entry[1], _src_entry[2], _src_entry[3]
                seed = seed_base + si * 104729 + batch_off
                _u1i(b'uSeed',               seed & 0x7FFFFFFF)
                _u1i(b'uTotalRaysPerSource',  max(1, int(n_rays)))
                _u3f(b'uSrcPos',             *np.asarray(sp, np.float32).ravel()[:3])
                _u3f(b'uSrcDir',             *np.asarray(sd, np.float32).ravel()[:3])
                spec = np.asarray(spectrum, np.float32).ravel()
                _u4f(b'uSrcSpectrum', float(spec[0]), float(spec[1]),
                                      float(spec[2]), float(spec[3]))
                _src_radius  = float(_src_entry[4]) if len(_src_entry) > 4 else 0.0
                _sr_row      = self._source_records_np[min(si, len(self._source_records_np) - 1)]
                _dir_model   = int(round(float(_sr_row[7])))
                _dir_param   = float(_sr_row[11])
                glUniform1f(glGetUniformLocation(prog, b'uSrcRadius'),   _src_radius)
                glUniform1i(glGetUniformLocation(prog, b'uSrcDirModel'), _dir_model)
                glUniform1f(glGetUniformLocation(prog, b'uSrcDirParam'), _dir_param)
                last_si = si
            else:
                # Same source, different offset — just update seed + offset.
                seed = seed_base + si * 104729 + batch_off
                _u1i(b'uSeed',       seed & 0x7FFFFFFF)

            _u1i(b'uBatchSize',   batch_size)
            _u1i(b'uBatchOffset', batch_off)
            n_groups = max(1, int(_math.ceil(batch_size / 128.0)))
            gx = min(n_groups, 65535)
            gy = max(1, int(_math.ceil(n_groups / gx)))
            glDispatchCompute(gx, gy, 1)
            dispatched += 1

            cursor = (cursor + 1) % n_tokens
            if (time.perf_counter() - t0) * 1000.0 >= budget_ms:
                break
            # Wrap-around: stop after one full ring to avoid infinite spin
            # when budget_ms is very large.
            if cursor == self._ring_cursor:
                break

        # One barrier after all dispatches — GPU queues them all then stalls
        # only once per pump call.
        if dispatched > 0:
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                            GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                            GL_TEXTURE_FETCH_BARRIER_BIT)
            # Mark that a source was active this frame so apply_decay() can
            # reset the display brightness multiplier.
            self._source_dispatched = True
        glUseProgram(0)
        self._ring_cursor = cursor
        self._fwd_frame += 1
        return dispatched


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
    BTN_H  = 24     # height of the "New Frame" action button

    # (key, label, lo, hi, default, log_scale, live)
    # lo = left-end value (low quality), hi = right-end value (high quality)
    _DEFS = [
        ('ray_density', 'Ray dens', 0.1,       8.0,   1.0, False, True ),
        ('segs',     'Str segs',     30,       240,    60, False, False),
        ('plate_th', 'Board res',    64,     1_024,   128, False, True ),
        ('dx',       'Press dx',  0.016,     0.004, 0.010, False, False),
        ('ray_exposure', 'Exposure', 0.25,    6.0,    1.0, False, True ),
        ('ray_gamma', 'Gamma',      0.20,    1.60, float(GPU_RAY_FIELD_GAMMA), False, True ),
        ('mic_gain', 'Mic',        0.0,       1.0,   1.0, False, True ),
        ('pickup_gain', 'Pickup',  0.0,       1.0,   1.0, False, True ),
        ('sensor_iso',  'Sen ISO', 0.25,     12.0,   1.4, False, True ),
        ('sensor_rate', 'Sen rows',1.0,     512.0,  16.0, False, True ),
        ('sensor_spp',  'Sen spp', 1.0,       8.0,   1.0, False, True ),
        ('sensor_fps',  'Sen FPS', 0.0,      60.0,   0.0, False, True ),
        ('frame_step',  'Frm step',0.0,       8.0,   1.0, False, True ),
        ('lens_aperture', 'Aperture', 0.0,  0.080,  0.0, False, True ),
        ('lens_ca',     'CA',       0.0,    0.020,  0.0, False, True ),
        # Decay half-life: 0 = stable; >0 = seconds for display to reach 50%
        # brightness after the source goes silent.  Controlled globally and
        # applied to the accumulator via the GPU decay compute shader.
        ('film_decay',  'Decay \u00bdlife', 0.0, 30.0,  5.0, False, True ),
        # Volumetric atmosphere \u2014 soft diffuse glow and sharp specular sparkle
        ('air_diff',    'Air glow',  0.0,  1.0,  0.35, False, False),
        ('air_spec',    'Air spark', 0.0,  1.0,  0.65, False, False),
        ('air_aniso',   'Air lobe',  1.0, 32.0, 12.0,  False, True ),
    ]

    _C_BG    = (0.04, 0.04, 0.07, 0.97)   # near-opaque so scene doesn't bleed through
    _C_TRACK = (0.22, 0.22, 0.27, 1.00)
    _C_LIVE  = (0.18, 0.78, 0.28, 1.00)   # green  — live
    _C_BUILD = (0.22, 0.46, 0.96, 1.00)   # blue   — needs rebuild (no pending change)
    _C_PEND  = (1.00, 0.62, 0.10, 1.00)   # amber  — pending rebuild change
    _C_KNOB  = (0.90, 0.90, 0.90, 1.00)
    _C_TEXT  = (1.00, 1.00, 1.00, 1.00)   # pure white for maximum contrast

    @classmethod
    def knobspec(cls):
        """Return KnobSpec descriptors mapped from legacy slider tuples.

        This is a migration bridge that keeps existing HUD behavior while
        exposing the same controls through the shared KnobSpec system.
        """
        try:
            from controls import slider_defs_to_knobs
            return slider_defs_to_knobs(
                cls._DEFS,
                group="Demo Pluck Slider",
                source_class="_SliderPanel",
            )
        except Exception:
            return []

    def hierarchy_nodes(self):
        """Return control-graph nodes for legacy slider definitions."""
        try:
            from controls import knobs_to_object_nodes
            return knobs_to_object_nodes(
                self.knobspec(),
                object_id="demo_pluck.slider_panel",
                parent_key="slider_panel",
            )
        except Exception:
            return []

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
        self._btn_callback = None  # called (no args) on "New Frame [C]" click
        self._btn_hover    = False

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
            surf = font.render(label_str, True, (0, 0, 0), (220, 220, 220))
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
        return 5 + len(self.keys) * self.ROW + 4 + self.BTN_H + 6

    def _btn_rect(self) -> tuple:
        """Pixel rect (x, y, w, h) of the 'New Frame [C]' button."""
        y = self.PY + 5 + len(self.keys) * self.ROW + 4 + 3
        return (self.PX + 4, y, self.PW - 8, self.BTN_H)

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
            bx, by, bw, bh = self._btn_rect()
            if bx <= mx <= bx + bw and by <= my <= by + bh:
                if self._btn_callback is not None:
                    self._btn_callback()
            return True   # click in panel but not on a track — absorb it
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self._drag >= 0:
                self._drag = -1
                return True
        elif ev.type == MOUSEMOTION:
            mx2, my2 = ev.pos
            bx, by, bw, bh = self._btn_rect()
            self._btn_hover = bx <= mx2 <= bx + bw and by <= my2 <= by + bh
            if self._drag >= 0:
                self._apply_mouse(self._drag, mx2)
                return True
        return False

    def _apply_mouse(self, idx: int, mx: int):
        tx, _, tw, _ = self._track_rect(idx)
        t   = (mx - tx) / max(tw, 1)
        raw = self._v(idx, t)
        key = self.keys[idx]
        if   key == 'ray_density': raw = float(np.clip(raw, 0.1, 16.0))
        elif key == 'segs':     raw = int(np.clip(round(raw), 30, 240))
        elif key == 'plate_th': raw = int(np.clip(round(raw), 64, 1024))
        elif key == 'dx':       raw = float(np.clip(raw, 0.004, 0.016))
        elif key == 'ray_exposure':
            raw = float(np.clip(raw, 0.25, 6.0))
        elif key == 'ray_gamma':
            raw = float(np.clip(raw, 0.20, 1.60))
        elif key in ('mic_gain', 'pickup_gain'):
            raw = float(np.clip(raw, 0.0, 1.0))
        elif key == 'sensor_iso':
            raw = float(np.clip(raw, 0.25, 12.0))
        elif key == 'sensor_rate':
            raw = int(np.clip(round(raw), 1, 1024))
        elif key == 'sensor_spp':
            raw = int(np.clip(round(raw), 1, 16))
        elif key == 'sensor_fps':
            raw = int(np.clip(round(raw), 0, 120))
        elif key == 'frame_step':
            raw = int(np.clip(round(raw), 0, 16))
        old = self.values[key]
        self.values[key] = raw
        if raw != old and not self._live[idx]:
            self._pend[key] = True

    def _get_text_texture(self, text: str) -> tuple[int, tuple[int, int]]:
        cached = self._text_cache.get(text)
        if cached is not None:
            return cached
        surf = self._font.render(text, True, (0, 0, 0), (220, 220, 220))
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

    def _draw_hud_text(self, text: str, x: int, y: int, rw: int, rh: int, center: bool = False):
        """Draw an arbitrary text string as a HUD overlay at pixel position (x, y)."""
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        tex, (tw, th) = self._get_text_texture(text)
        px = x - tw // 2 if center else x
        # dark pill behind the text
        pad = 5
        self._draw_quad(px - pad, y - pad, tw + pad * 2, th + pad * 2,
                        (0.02, 0.02, 0.04, 0.88), rw, rh)
        self._draw_textured(tex, tw, th, px, y, rw, rh)

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
        # Draw solid background first with no blending so scene never bleeds through
        glDisable(GL_BLEND)
        self._draw_quad(self.PX, self.PY, self.PW, self._panel_h(),
                        (0.04, 0.04, 0.08, 1.0), win_w, win_h)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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

        # "New Frame [C]" button at the bottom of the panel
        bx, by, bw, bh = self._btn_rect()
        btn_bg = (0.28, 0.52, 0.96, 1.0) if self._btn_hover else (0.14, 0.26, 0.56, 1.0)
        self._draw_quad(bx, by, bw, bh, btn_bg, win_w, win_h)
        btn_tex, (btw, bth) = self._get_text_texture("New Frame  [C]")
        self._draw_textured(btn_tex, btw, bth,
                            bx + (bw - btw) // 2, by + (bh - bth) // 2, win_w, win_h)

        glEnable(GL_DEPTH_TEST)
        glBindVertexArray(0)
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Player camera settings panel
# Opened by E in walk mode when no interactable is nearby.
# Non-blocking overlay — the main loop continues running while it is visible.
# Writes go directly to Camera attributes; sensor_iso/decay sync to _SliderPanel.
# Auto-* checkboxes are wired for display only (no logic yet).
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Camera optics schematic — three-view line diagram shown left of camera panel
# ─────────────────────────────────────────────────────────────────────────────

class _CameraOpticsView:
    """Three-view cross-section diagram of the camera's optical geometry.

    Drawn with GL_LINES using the same colour-quad shader as the camera panel.
    Sub-views (stacked vertically inside the panel):
      1. Side / meridional  (Y = optical axis, Z = vertical)
         Shows: sensor plane, lens plane, aperture half-extent, focal plane,
                marginal rays (cone from aperture edge to focal point),
                tilt/shift offset arrow, N-gon blade tick marks.
      2. Front / sagittal   (X horizontal, Z vertical)
         Shows: aperture shape projected face-on (circle or N-gon outline).
      3. Tilt-shift / top   (X horizontal, Y depth)
         Shows: sensor rectangle + tilt/shift crosshair displacement.

    The view auto-scales so the sensor-to-focal-plane range always fits.
    No textures are needed; labels are rendered via the shared text helper.
    """

    PW     = 260          # panel width (pixels)
    PAD    = 10           # inner padding
    V_GAP  = 8            # gap between sub-view frames
    TITLE  = 24           # title bar height
    LABEL  = 16           # label strip below each sub-view
    BTN_H  = 28           # render button height

    # colour palette (r,g,b,a)
    _C_BG       = (0.03, 0.03, 0.07, 1.00)
    _C_TITLE    = (0.06, 0.16, 0.36, 1.00)
    _C_FRAME    = (0.18, 0.25, 0.38, 1.00)
    _C_AXIS     = (0.22, 0.32, 0.52, 0.90)
    _C_SENSOR   = (0.25, 0.75, 1.00, 1.00)
    _C_LENS     = (0.85, 0.85, 0.30, 0.95)
    _C_APERTURE = (1.00, 0.55, 0.20, 1.00)
    _C_FOCAL    = (0.30, 0.90, 0.55, 0.85)
    _C_MARGINAL = (0.55, 0.55, 0.75, 0.60)
    _C_TILT     = (1.00, 0.40, 0.70, 0.90)

    def __init__(self):
        self._cam          = None   # Camera object (set by attach)
        self._cam_panel    = None   # _PlayerCameraPanel (for open state)
        self._p_col        = None
        self._p_tex        = None
        self._lvao         = None
        self._lvbo         = None
        self._qvao         = None
        self._qvbo         = None
        self._tvao         = None
        self._tvbo         = None
        self._text_cache: dict = {}
        self._line_buf     = []     # list of (x0,y0,x1,y1,color)
        self._render_btn_rect: tuple[int, int, int, int] | None = None

    # ── GL init ───────────────────────────────────────────────────────────────

    def init_gl(self):
        import ctypes
        from OpenGL.GL import (
            GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE,
            GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
            glGenVertexArrays, glGenBuffers, glBindVertexArray,
            glBindBuffer, glBufferData, glEnableVertexAttribArray,
            glVertexAttribPointer,
        )
        self._p_col = _prog((_HUD2D_VS, GL_VERTEX_SHADER),
                            (_HUD2D_FS, GL_FRAGMENT_SHADER))
        self._p_tex = _prog((_HUD2D_TEX_VS, GL_VERTEX_SHADER),
                            (_HUD2D_TEX_FS, GL_FRAGMENT_SHADER))
        # Quad VAO — 4 vertices × 2 floats = 32 bytes
        self._qvao = glGenVertexArrays(1)
        self._qvbo = glGenBuffers(1)
        glBindVertexArray(self._qvao)
        glBindBuffer(GL_ARRAY_BUFFER, self._qvbo)
        glBufferData(GL_ARRAY_BUFFER, 32, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glBindVertexArray(0)
        # Text quad VAO — 4 vertices × 4 floats (pos+uv) = 64 bytes
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
        # Line VAO — resized dynamically; start with room for 512 segments
        self._lvao = glGenVertexArrays(1)
        self._lvbo = glGenBuffers(1)
        glBindVertexArray(self._lvao)
        glBindBuffer(GL_ARRAY_BUFFER, self._lvbo)
        glBufferData(GL_ARRAY_BUFFER, 512 * 2 * 8, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glBindVertexArray(0)
        self._lvbo_cap = 512   # segments allocated

    # ── Attach ────────────────────────────────────────────────────────────────

    def attach(self, cam, cam_panel) -> None:
        self._cam       = cam
        self._cam_panel = cam_panel

    def handle_event(self, ev, win_w: int, win_h: int) -> bool:
        if self._cam_panel is None or not getattr(self._cam_panel, '_open', False):
            return False
        if ev.type != pygame.MOUSEBUTTONDOWN or ev.button != 1:
            return False
        if self._render_btn_rect is None:
            return False
        mx, my = ev.pos
        bx, by, bw, bh = self._render_btn_rect
        if bx <= mx <= bx + bw and by <= my <= by + bh:
            if hasattr(self._cam_panel, 'request_raytrace_render'):
                self._cam_panel.request_raytrace_render()
            return True
        return False

    # ── GL primitives ─────────────────────────────────────────────────────────

    def _draw_quad(self, x, y, w, h, color, win_w, win_h):
        import ctypes, struct
        from OpenGL.GL import (
            glUseProgram, glGetUniformLocation, glUniform2f, glUniform4f,
            glBindVertexArray, glBindBuffer, glBufferSubData, glDrawArrays,
            GL_ARRAY_BUFFER, GL_TRIANGLE_STRIP,
        )
        glUseProgram(self._p_col)
        glUniform2f(glGetUniformLocation(self._p_col, b'uRes'),
                    float(win_w), float(win_h))
        glUniform4f(glGetUniformLocation(self._p_col, b'uColor'), *color)
        v = struct.pack('8f',
                        float(x),   float(y),
                        float(x+w), float(y),
                        float(x),   float(y+h),
                        float(x+w), float(y+h))
        glBindVertexArray(self._qvao)
        glBindBuffer(GL_ARRAY_BUFFER, self._qvbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, len(v), v)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

    def _flush_lines(self, win_w, win_h):
        """Batch-draw all accumulated line segments by colour group."""
        if not self._line_buf:
            return
        import struct, ctypes
        from OpenGL.GL import (
            glUseProgram, glGetUniformLocation, glUniform2f, glUniform4f,
            glBindVertexArray, glBindBuffer, glBufferData, glBufferSubData,
            glDrawArrays, GL_ARRAY_BUFFER, GL_LINES, GL_DYNAMIC_DRAW,
        )
        # Group by colour so we minimise uniform changes
        from collections import defaultdict
        by_col = defaultdict(list)
        for x0, y0, x1, y1, col in self._line_buf:
            by_col[col].append((x0, y0, x1, y1))
        glUseProgram(self._p_col)
        glUniform2f(glGetUniformLocation(self._p_col, b'uRes'),
                    float(win_w), float(win_h))
        glBindVertexArray(self._lvao)
        glBindBuffer(GL_ARRAY_BUFFER, self._lvbo)
        for col, segs in by_col.items():
            n = len(segs)
            # Grow buffer if needed
            if n > self._lvbo_cap:
                glBufferData(GL_ARRAY_BUFFER, n * 2 * 8, None, GL_DYNAMIC_DRAW)
                self._lvbo_cap = n
            data = struct.pack(f'{n*4}f',
                               *[v for (x0,y0,x1,y1) in segs for v in (x0,y0,x1,y1)])
            glBufferSubData(GL_ARRAY_BUFFER, 0, len(data), data)
            glUniform4f(glGetUniformLocation(self._p_col, b'uColor'), *col)
            glDrawArrays(GL_LINES, 0, n * 2)
        glBindVertexArray(0)
        self._line_buf.clear()

    def _line(self, x0, y0, x1, y1, color):
        self._line_buf.append((float(x0), float(y0), float(x1), float(y1), color))

    def _get_tex(self, text: str):
        if text in self._text_cache:
            return self._text_cache[text]
        from OpenGL.GL import (
            glGenTextures, glBindTexture, glTexParameteri, glTexImage2D,
            GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
            GL_LINEAR, GL_RGBA, GL_UNSIGNED_BYTE,
        )
        pygame.font.init()
        f = pygame.font.SysFont("consolas,monospace", 12)
        s = f.render(str(text), True, (180, 200, 230), (8, 8, 18))
        w, h = s.get_size()
        raw = pygame.image.tobytes(s, "RGBA")
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glBindTexture(GL_TEXTURE_2D, 0)
        r = (tex, (w, h))
        self._text_cache[text] = r
        return r

    def _draw_text(self, text: str, x: int, y: int, win_w: int, win_h: int):
        import struct
        from OpenGL.GL import (
            glUseProgram, glGetUniformLocation, glUniform2f, glUniform1i,
            glActiveTexture, glBindTexture, glBindVertexArray, glBindBuffer,
            glBufferSubData, glDrawArrays,
            GL_ARRAY_BUFFER, GL_TRIANGLE_STRIP, GL_TEXTURE0, GL_TEXTURE_2D,
        )
        tex, (tw, th) = self._get_tex(text)
        glUseProgram(self._p_tex)
        glUniform2f(glGetUniformLocation(self._p_tex, b'uRes'),
                    float(win_w), float(win_h))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(glGetUniformLocation(self._p_tex, b'uTex'), 0)
        v = struct.pack('16f',
                        float(x),    float(y),    0.0, 0.0,
                        float(x+tw), float(y),    1.0, 0.0,
                        float(x),    float(y+th), 0.0, 1.0,
                        float(x+tw), float(y+th), 1.0, 1.0)
        glBindVertexArray(self._tvao)
        glBindBuffer(GL_ARRAY_BUFFER, self._tvbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, len(v), v)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _ngon_pts(cx, cy, rx, ry, n, rot=0.0, steps_circle=64):
        """Return list of (x,y) vertices for a regular N-gon (or circle if n<3)."""
        if n < 3:
            pts = []
            for i in range(steps_circle):
                a = 2.0 * math.pi * i / steps_circle + rot
                pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
            return pts
        pts = []
        for k in range(n):
            a = 2.0 * math.pi * k / n + rot
            pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
        return pts

    def _draw_polygon_lines(self, pts, color, close=True):
        """Draw pts as a closed or open polyline."""
        for i in range(len(pts)):
            if not close and i == len(pts) - 1:
                break
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            self._line(x0, y0, x1, y1, color)

    # ── Sub-view renderers ────────────────────────────────────────────────────

    def _draw_side_view(self, fx, fy, fw, fh, cam):
        """Meridional (side) cross section.

        Shows effective lens position (nominal + extension), biconvex arcs,
        aperture ticks, N-gon blade marks, marginal rays, and focal plane
        (tilted when Scheimpflug is non-zero).
        """
        PAD = self.PAD
        focal_mm      = float(getattr(cam, 'focal_mm', 35.0))
        extension_mm  = float(getattr(cam, 'extension_mm', 0.0))
        eff_focal_mm  = max(1.0, focal_mm + extension_mm)
        focus_m       = float(getattr(cam, 'focus_m',  1.6))
        ap_r          = float(getattr(cam, 'aperture', 0.0))
        tilt_y        = float(getattr(cam, 'tilt_shift', [0, 0])[1])
        lens_tilt     = getattr(cam, 'lens_tilt', [0.0, 0.0])
        scheimpflug_x = float(lens_tilt[0])
        n_blades      = int(getattr(cam, 'n_blades', 0))
        ap_rot        = float(getattr(cam, 'aperture_rot', 0.0))
        focal_m       = eff_focal_mm * 1e-3

        scene_depth = focus_m
        total_depth = focal_m + scene_depth
        scale       = (fw - 2*PAD) / max(total_depth, 1e-3)

        cz = fy + fh // 2
        x_sensor        = fx + PAD
        x_lens_nominal  = x_sensor + int(focal_mm * 1e-3 * scale)
        x_lens          = x_sensor + int(focal_m * scale)
        x_focal         = min(x_lens + int(scene_depth * scale), fx + fw - PAD)

        if ap_r > 1e-7:
            ap_px = min(int(ap_r * scale * 200), fh // 2 - 4)
        else:
            ap_px = max(4, fh // 8)

        # Optical axis
        self._line(x_sensor, cz, x_focal + 8, cz, self._C_AXIS)

        # Sensor plane shifted by tilt_y
        sh = fh // 2 - 4
        ts_px = int(tilt_y * scale * 0.5 * fh)
        self._line(x_sensor, cz - sh + ts_px, x_sensor, cz + sh + ts_px, self._C_SENSOR)
        if abs(ts_px) > 2:
            ax0, ay0 = x_sensor - 6, cz + ts_px
            self._line(ax0, cz, ax0, ay0, self._C_TILT)
            arr = 3 if ts_px < 0 else -3
            self._line(ax0, ay0, ax0 - 3, ay0 + arr, self._C_TILT)
            self._line(ax0, ay0, ax0 + 3, ay0 + arr, self._C_TILT)

        # Extension bracket: nominal → effective lens position
        if abs(extension_mm) > 0.5 and x_lens != x_lens_nominal:
            bk_y = cz - fh // 2 + 6
            self._line(x_lens_nominal, bk_y,     x_lens, bk_y,     self._C_FOCAL)
            self._line(x_lens_nominal, bk_y - 3, x_lens_nominal, bk_y + 3, self._C_FOCAL)
            self._line(x_lens,         bk_y - 3, x_lens,         bk_y + 3, self._C_FOCAL)

        # Lens plane (biconvex arcs) at effective position
        lh = fh // 2 - 2
        self._line(x_lens, cz - lh, x_lens, cz + lh, self._C_LENS)
        arc_bow = max(3, min(10, int(ap_px * 0.3)))
        for sign in (-1, 1):
            prev = None
            for i in range(13):
                t  = i / 12
                za = (t - 0.5) * 2.0 * lh
                xa = x_lens + sign * arc_bow * (1.0 - (za / lh) ** 2)
                pt = (xa, cz + za)
                if prev:
                    self._line(prev[0], prev[1], pt[0], pt[1], self._C_LENS)
                prev = pt

        # Aperture extent ticks
        self._line(x_lens - 5, cz - ap_px, x_lens + 5, cz - ap_px, self._C_APERTURE)
        self._line(x_lens - 5, cz + ap_px, x_lens + 5, cz + ap_px, self._C_APERTURE)
        self._line(x_lens - 5, cz - ap_px, x_lens - 5, cz + ap_px, self._C_APERTURE)
        if n_blades >= 3:
            for k in range(n_blades):
                a   = 2.0 * math.pi * k / n_blades + ap_rot
                py_ = cz + int(ap_px * math.sin(a))
                self._line(x_lens - 8, py_, x_lens - 3, py_, self._C_APERTURE)

        # Marginal ray cone
        for sign in (-1, 1):
            ey = cz + sign * ap_px
            self._line(x_sensor, cz + sign * (sh // 2) + ts_px, x_lens, ey, self._C_MARGINAL)
            self._line(x_lens, ey, x_focal, cz, self._C_MARGINAL)

        # Focal plane — tilted if Scheimpflug active
        dash, gap = 5, 4
        if abs(scheimpflug_x) < 0.02:
            y_cur = fy + PAD
            while y_cur < fy + fh - PAD:
                self._line(x_focal, y_cur, x_focal,
                           min(y_cur + dash, fy + fh - PAD), self._C_FOCAL)
                y_cur += dash + gap
        else:
            # Tilted line across the frame at x_focal, slope = tan(scheimpflug_x)
            tilt_slope = math.tan(scheimpflug_x)
            half = fh // 2 - PAD
            sign_t = 1.0 if scheimpflug_x > 0 else -1.0
            y_top = cz - half;  x_top = x_focal + int(half * abs(tilt_slope) * sign_t)
            y_bot = cz + half;  x_bot = x_focal - int(half * abs(tilt_slope) * sign_t)
            length = max(1, math.hypot(x_bot - x_top, y_bot - y_top))
            n_steps = max(1, int(length / (dash + gap)))
            for i in range(n_steps):
                t0 = i / n_steps
                t1 = min(1.0, t0 + dash / length)
                self._line(x_top + (x_bot - x_top) * t0, y_top + (y_bot - y_top) * t0,
                           x_top + (x_bot - x_top) * t1, y_top + (y_bot - y_top) * t1,
                           self._C_FOCAL)

    def _draw_front_view(self, fx, fy, fw, fh, cam):
        """Sagittal (front-on) view of the aperture: X horizontal, Z vertical.

        For a circle: draws the full circular outline.
        For an N-gon: draws the exact polygon outline, rotated by aperture_rot.
        Also draws the sensor rectangle outline behind the aperture.
        """
        cx = fx + fw // 2
        cy = fy + fh // 2
        n_blades = int(getattr(cam, 'n_blades', 0))
        ap_rot   = float(getattr(cam, 'aperture_rot', 0.0))
        ap_r     = float(getattr(cam, 'aperture', 0.0))

        # Scale aperture to fit: if ap_r > 0 use it, else draw placeholder
        ap_px = fh // 2 - self.PAD
        if ap_r < 1e-8:
            ap_px = max(8, ap_px // 3)   # pinhole symbol

        # Sensor outline (thin grey rectangle behind aperture)
        sw = int(fw * 0.72)
        sh = int(fh * 0.55)
        self._draw_polygon_lines([
            (cx - sw//2, cy - sh//2), (cx + sw//2, cy - sh//2),
            (cx + sw//2, cy + sh//2), (cx - sw//2, cy + sh//2),
        ], self._C_SENSOR)

        # Aperture outline
        pts = self._ngon_pts(cx, cy, ap_px, ap_px, n_blades, ap_rot)
        self._draw_polygon_lines(pts, self._C_APERTURE)

        # Axis crosshair
        self._line(cx - 6, cy, cx + 6, cy, self._C_AXIS)
        self._line(cx, cy - 6, cx, cy + 6, self._C_AXIS)

    def _draw_tilt_view(self, fx, fy, fw, fh, cam):
        """Tilt/shift/gimbal view: sensor face with all in-plane transforms.

        Shows:
          • Sensor rectangle
          • Principal-point shift (tilt_shift arrow)
          • Gimbal lens-axis rotation (arc + pointer showing gimbal pan/tilt)
          • Extension bar (depth indicator along the optical axis edge)
        """
        cx = fx + fw // 2
        cy = fy + fh // 2
        tilt_x       = float(getattr(cam, 'tilt_shift', [0, 0])[0])
        tilt_y       = float(getattr(cam, 'tilt_shift', [0, 0])[1])
        gimbal       = getattr(cam, 'gimbal', [0.0, 0.0])
        gimbal_pan   = float(gimbal[0])
        gimbal_tilt  = float(gimbal[1])
        extension_mm = float(getattr(cam, 'extension_mm', 0.0))
        focal_mm     = float(getattr(cam, 'focal_mm', 35.0))

        sw = int(fw * 0.72)
        sh = int(fh * 0.55)

        # Sensor rectangle
        self._draw_polygon_lines([
            (cx - sw//2, cy - sh//2), (cx + sw//2, cy - sh//2),
            (cx + sw//2, cy + sh//2), (cx - sw//2, cy + sh//2),
        ], self._C_SENSOR)

        # Centre cross
        self._line(cx - 4, cy, cx + 4, cy, self._C_AXIS)
        self._line(cx, cy - 4, cx, cy + 4, self._C_AXIS)

        # Principal-point shift arrow
        max_shift = 0.5
        ox = int(tilt_x / max_shift * (sw // 2 - 6))
        oy = int(tilt_y / max_shift * (sh // 2 - 6))
        if abs(ox) > 1 or abs(oy) > 1:
            self._line(cx, cy, cx + ox, cy - oy, self._C_TILT)
            px_, py_ = cx + ox, cy - oy
            self._line(px_ - 4, py_, px_ + 4, py_, self._C_TILT)
            self._line(px_, py_ - 4, px_, py_ + 4, self._C_TILT)

        # Gimbal indicator: arc showing pan (horizontal) and tilt (vertical)
        # radius of the gimbal arc symbol
        gr = min(sw, sh) // 2 - 4
        # Gimbal pan: arc in horizontal plane — draw as small arc at bottom of sensor
        if abs(gimbal_pan) > 0.005 or abs(gimbal_tilt) > 0.005:
            # Draw a small compass-rose style pointer at centre showing lens direction
            max_g = math.pi / 4   # ±45° max display range
            gx = int(gr * 0.5 * math.sin(gimbal_pan)  / (math.pi / 4))
            gy = int(gr * 0.5 * math.sin(gimbal_tilt) / (math.pi / 4))
            # Arrow from centre to gimbal-displaced axis point
            self._line(cx, cy, cx + gx, cy + gy, self._C_LENS)
            self._line(cx + gx - 3, cy + gy - 3, cx + gx + 3, cy + gy + 3, self._C_LENS)
            self._line(cx + gx - 3, cy + gy + 3, cx + gx + 3, cy + gy - 3, self._C_LENS)
            # Outer gimbal circle arc (partial, ±gimbal extent)
            arc_r = gr // 2
            arc_pts = self._ngon_pts(cx, cy, arc_r, arc_r, 0, 0.0, steps_circle=32)
            self._draw_polygon_lines(arc_pts, self._C_AXIS, close=True)

        # Extension bar along the right edge of the sensor
        if abs(extension_mm) > 0.1:
            max_ext  = 50.0   # mm — full bar
            ext_frac = min(1.0, abs(extension_mm) / max_ext)
            bar_x    = cx + sw // 2 + 4
            bar_top  = cy - sh // 2
            bar_bot  = cy + sh // 2
            bar_fill = int((bar_bot - bar_top) * ext_frac)
            # Background track
            self._line(bar_x, bar_top, bar_x, bar_bot, self._C_FRAME)
            # Filled portion
            self._line(bar_x, bar_bot, bar_x, bar_bot - bar_fill, self._C_FOCAL)
            # Tick at current fill level
            self._line(bar_x - 2, bar_bot - bar_fill,
                       bar_x + 2, bar_bot - bar_fill, self._C_FOCAL)


    # ── Main draw ─────────────────────────────────────────────────────────────

    def draw(self, win_w: int, win_h: int):
        """Draw the three-view optics schematic left of the camera settings panel."""
        if self._cam_panel is None or not getattr(self._cam_panel, '_open', False):
            return
        if self._p_col is None:
            return
        cam = self._cam
        if cam is None:
            return

        from OpenGL.GL import (
            glEnable, glDisable, glBlendFunc, glLineWidth,
            GL_BLEND, GL_DEPTH_TEST, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            glBindVertexArray, glUseProgram, glBindTexture, GL_TEXTURE_2D,
        )

        # Position: directly left of the camera panel
        cam_px   = self._cam_panel._px(win_w)
        cam_py   = self._cam_panel._py(win_h)
        cam_ph   = self._cam_panel._panel_h()
        panel_w  = self.PW
        panel_h  = cam_ph
        px       = cam_px - panel_w - 6
        py       = cam_py
        if px < 0:
            return   # not enough room

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        self._draw_quad(px, py, panel_w, panel_h, self._C_BG, win_w, win_h)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Title bar
        self._draw_quad(px, py, panel_w, self.TITLE - 2, self._C_TITLE, win_w, win_h)
        self._draw_text("OPTICS  side · front · shift", px + 6, py + 5, win_w, win_h)

        # Partition the remaining height into three equal sub-view slots
        inner_h = panel_h - self.TITLE - self.BTN_H - self.V_GAP
        slot_h  = (inner_h - 2 * self.V_GAP) // 3
        slot_w  = panel_w - 2 * self.PAD

        views = [
            ("side",  self._draw_side_view),
            ("front", self._draw_front_view),
            ("shift", self._draw_tilt_view),
        ]
        for vi, (label, draw_fn) in enumerate(views):
            vy = py + self.TITLE + vi * (slot_h + self.V_GAP)
            vx = px + self.PAD
            # Sub-view frame
            self._draw_quad(vx - 1, vy - 1, slot_w + 2, slot_h + 2,
                            self._C_FRAME, win_w, win_h)
            self._draw_quad(vx, vy, slot_w, slot_h, self._C_BG, win_w, win_h)
            # Draw content into the frame
            draw_fn(vx, vy, slot_w, slot_h, cam)
            # Flush all queued line segments for this sub-view
            glLineWidth(1.2)
            self._flush_lines(win_w, win_h)
            glLineWidth(1.0)
            # Label
            self._draw_text(label, vx + 3, vy + slot_h - self.LABEL + 2, win_w, win_h)

        btn_x = px + self.PAD
        btn_w = panel_w - 2 * self.PAD
        btn_y = py + panel_h - self.BTN_H
        self._render_btn_rect = (btn_x, btn_y, btn_w, self.BTN_H - 2)
        _armed = bool(getattr(self._cam_panel, '_raytrace_inspect_active', False))
        self._draw_quad(
            btn_x,
            btn_y,
            btn_w,
            self.BTN_H - 2,
            (0.20, 0.58, 0.34, 0.95) if _armed else (0.12, 0.34, 0.58, 0.95),
            win_w,
            win_h,
        )
        self._draw_text(
            "RENDER (raytrace)" if not _armed else "RENDER LOCKED",
            btn_x + 8,
            btn_y + 7,
            win_w,
            win_h,
        )

        glEnable(GL_DEPTH_TEST)
        glBindVertexArray(0)
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)


# ─────────────────────────────────────────────────────────────────────────────
class _PlayerCameraPanel:
    """Full-parameter camera HUD for the player's head camera.

    Covers every optical parameter on the Camera object plus sensor ISO,
    sensor gain, and decay half-life.  Auto-* checkboxes are present for
    future logic but perform no action yet.
    """

    PW      = 320
    ROW     = 26
    TX      = 130   # track left edge within panel
    TW      = 160   # track width
    TH      = 6     # track height
    PAD_TOP = 34    # title bar height

    # (key, label, lo, hi, default, is_log, fmt_spec)
    _SLIDERS = [
        # ── Optics ──────────────────────────────────────────────────────────
        ('focal_mm',    'Focal mm',      12.0,   180.0, 35.0,  False, '.1f'),
        ('focus_m',     'Focus m',        0.05,   20.0,  1.6,  False, '.2f'),
        ('aperture',    'Aperture r',     0.0,     0.08, 0.0,  False, '.4f'),
        ('ca',          'Chrom. Ab.',     0.0,     0.02, 0.0,  False, '.4f'),
        ('tilt_x',      'Tilt/Shift X',  -0.5,     0.5,  0.0, False, '.3f'),
        ('tilt_y',      'Tilt/Shift Y',  -0.5,     0.5,  0.0, False, '.3f'),
        # ── Sensor / exposure ────────────────────────────────────────────────
        ('sensor_iso',  'ISO',            0.25,   12.0,  1.4,  False, '.2f'),
        ('sensor_gain', 'Sensor Gain',    0.5,    32.0,  8.0,  False, '.1f'),
        ('decay',       'Decay \u00bdlife',0.0,   30.0,  5.0,  False, '.1f'),
        # ── Ray-camera config ────────────────────────────────────────────────
        ('ray_density',  'Ray Density',   0.1,    8.0,   1.0,  False, '.2f'),
        ('ray_exposure', 'Ray Exposure',  0.25,   6.0,   1.0,  False, '.2f'),
        ('ray_gamma',    'Ray Gamma',     0.20,   1.60,  GPU_RAY_FIELD_GAMMA, False, '.3f'),
        ('focus_pick_dist', 'Focus Dist',  0.10,  20.0,   6.0,  False, '.2f'),
        ('focus_cone_deg',  'Focus Cone',  0.0,   10.0,   0.0,  False, '.2f'),
        ('focus_cone_rays', 'Focus Rays',  1.0,   32.0,   1.0,  False, '.0f'),
        ('sensor_rate',  'Sen Rows',      1.0,  512.0,  16.0,  False, '.0f'),
        ('sensor_spp',   'Sen SPP',       1.0,    8.0,   1.0,  False, '.0f'),
        ('sensor_fps',   'Sen FPS',       0.0,   60.0,   0.0,  False, '.0f'),
        ('frame_step',   'Frame Step',    0.0,    8.0,   1.0,  False, '.0f'),
        ('mic_gain',     'Mic Gain',      0.0,    1.0,   1.0,  False, '.3f'),
        ('pickup_gain',  'Pickup Gain',   0.0,    1.0,   1.0,  False, '.3f'),
        # ── Rebuild params ───────────────────────────────────────────────────
        ('segs',         'Str Segs',     30.0,  240.0,  60.0,  False, '.0f'),
        ('plate_th',     'Board Res',    64.0, 1024.0, 128.0,  False, '.0f'),
        ('dx',           'Press dx',    0.004,  0.016, 0.010,  False, '.4f'),
        # ── Atmosphere ───────────────────────────────────────────────────────
        ('air_diff',     'Air Glow',     0.0,    1.0,   0.35,  False, '.3f'),
        ('air_spec',     'Air Spark',    0.0,    1.0,   0.65,  False, '.3f'),
        ('air_aniso',    'Air Lobe',     1.0,   32.0,  12.0,   False, '.1f'),
    ]

    # Auto checkboxes — display-only; no logic yet
    _AUTOS = [
        ('auto_focus',    'Auto Focus'),
        ('auto_aperture', 'Auto Aperture'),
        ('auto_iso',      'Auto ISO'),
        ('auto_decay',    'Auto Decay'),
    ]
    _FOCUS_ENGINES = [
        ('bridge', 'Bridge'),
        ('analytic', 'Local'),
    ]
    HUD_FULL = "full"
    HUD_INFO = "info"
    HUD_OFF = "off"

    _C_BG      = (0.04, 0.04, 0.09, 1.00)
    _C_TITLE   = (0.08, 0.20, 0.42, 1.00)
    _C_TRACK   = (0.20, 0.20, 0.26, 1.00)
    _C_FILL    = (0.25, 0.65, 1.00, 1.00)
    _C_KNOB    = (0.90, 0.90, 0.90, 1.00)
    _C_CHECK   = (0.22, 0.80, 0.38, 1.00)
    _C_UNCHECK = (0.28, 0.28, 0.35, 1.00)
    _C_SEP     = (0.18, 0.28, 0.44, 1.00)
    _DUTY_DEFAULTS = None

    @classmethod
    def knobspec(cls):
        """Return KnobSpec descriptors mapped from camera panel sliders."""
        try:
            from controls import slider_defs_to_knobs
            return slider_defs_to_knobs(
                cls._SLIDERS,
                group="Demo Pluck Camera",
                source_class="_PlayerCameraPanel",
            )
        except Exception:
            return []

    @classmethod
    def duty_station_defaults(cls):
        """Return shared duty-station defaults for camera-panel migration."""
        if cls._DUTY_DEFAULTS is not None:
            return cls._DUTY_DEFAULTS
        try:
            from ray_tracer.station_specs import default_station_material_slots
            cls._DUTY_DEFAULTS = default_station_material_slots()
        except Exception:
            cls._DUTY_DEFAULTS = {}
        return cls._DUTY_DEFAULTS

    @classmethod
    def camera_hud_sections(cls):
        """Return (left_sections, center_tabs, right_sections) for the duty HUD.

        Converts the flat ``_SLIDERS`` list into the three-panel section format
        consumed by ``CameraDutyStationHUD.from_sections``.
        """
        _groups: dict[str, list] = {
            "computer": [],
            "sensors": [],
            "film": [],
            "rebuild": [],
            "lights": [],
            "right_misc": [],
        }

        _group_map = {
            "focal_mm": "computer",
            "focus_m": "computer",
            "aperture": "computer",
            "ca": "computer",
            "tilt_x": "computer",
            "tilt_y": "computer",
            "ray_density": "computer",
            "ray_exposure": "computer",
            "focus_pick_dist": "computer",
            "focus_cone_deg": "computer",
            "focus_cone_rays": "computer",

            "sensor_iso": "sensors",
            "sensor_gain": "sensors",
            "decay": "sensors",
            "sensor_rate": "sensors",
            "sensor_spp": "sensors",
            "sensor_fps": "sensors",
            "frame_step": "sensors",

            "air_diff": "film",
            "air_spec": "film",
            "air_aniso": "film",
            "ray_gamma": "film",

            "segs": "rebuild",
            "plate_th": "rebuild",
            "dx": "rebuild",

            "mic_gain": "lights",
            "pickup_gain": "lights",
        }

        for key, label, lo, hi, default, is_log, fmt_spec in cls._SLIDERS:
            step = max(1e-6, (hi - lo) / 200.0)
            # Round step to 1 significant figure
            import math as _math
            mag = 10 ** _math.floor(_math.log10(step)) if step > 0 else 1e-4
            step = round(step / mag) * mag

            knob = {
                "name": key,
                "label": label,
                "dtype": "float",
                "low": lo,
                "high": hi,
                "default": default,
                "step": step,
                "fmt": fmt_spec.lstrip('.') and fmt_spec or ".3f",
                "unit": "",
            }
            _groups[_group_map.get(key, "right_misc")].append(knob)

        left_sections: list = []
        right_sections = [
            {"id": "status", "label": "Status",
             "knobs": _groups["right_misc"]},
        ]

        center_tabs = [
            {"key": "views", "label": "VIEWS", "sections": [], "title": "VIEWS"},
            {"key": "sensors", "label": "SENSORS", "sections": [
                {"id": "sensors", "label": "Sensor", "knobs": _groups["sensors"]}
            ], "title": "SENSORS", "accent_rgb": (70, 120, 190)},
            {"key": "film", "label": "FILM", "sections": [
                {"id": "film", "label": "Film / Emulsion", "knobs": _groups["film"]}
            ], "title": "FILM", "accent_rgb": (90, 120, 90)},
            {"key": "computer", "label": "COMPUTER", "sections": [
                {"id": "computer", "label": "Camera Computer", "knobs": _groups["computer"]}
            ], "title": "COMPUTER", "accent_rgb": (120, 105, 165)},
            {"key": "lights", "label": "LIGHTS", "sections": [
                {"id": "lights", "label": "Lights", "knobs": _groups["lights"]}
            ], "title": "LIGHTS", "accent_rgb": (145, 110, 70)},
            {"key": "rebuild", "label": "REBUILD", "sections": [
                {"id": "rebuild", "label": "Rebuild", "knobs": _groups["rebuild"]}
            ], "title": "REBUILD", "accent_rgb": (60, 100, 160)},
        ]
        return left_sections, center_tabs, right_sections

    def hierarchy_nodes(self):
        """Return hierarchy nodes for camera panel controls.

        Current rendering path remains 2D HUD; this method provides the data
        bridge for staged migration into hierarchical station rendering.
        """
        try:
            from controls import knobs_to_object_nodes
            return knobs_to_object_nodes(
                self.knobspec(),
                object_id="demo_pluck.camera_panel",
                parent_key="camera_panel",
            )
        except Exception:
            return []

    def __init__(self):
        self._hud_mode = self.HUD_OFF
        self._open   = False
        self._drag   = -1
        self._values = {d[0]: d[4] for d in self._SLIDERS}
        self._lo     = {d[0]: d[2] for d in self._SLIDERS}
        self._hi     = {d[0]: d[3] for d in self._SLIDERS}
        self._log    = {d[0]: d[5] for d in self._SLIDERS}
        self._fmt    = {d[0]: d[6] for d in self._SLIDERS}
        self._auto   = {d[0]: False for d in self._AUTOS}
        self._cam_render_mode: 'RenderMode' = RenderMode.C
        # Layer / display toggles driven by the panel
        self._show_illum:   bool = False
        self._show_sensor:  bool = False
        self._enlarger:     bool = False
        self._focus_ray_engine: str = 'bridge'
        # Bound live objects — set by attach()
        self._cam:          object = None
        self._slider_panel: object = None
        self._renderer:     object = None
        self._player_ctrl:  object = None
        self._on_change:    dict   = {}   # key → callable(value)
        # GL resources
        self._p_col  = None
        self._p_tex  = None
        self._qvao = self._qvbo = None
        self._tvao = self._tvbo = None
        self._text_cache: dict = {}
        # HUD shell state: no center panel, only top/left/bottom chrome.
        self._library_tabs = ["backpack", "ecosystem"]
        self._library_active_tab = 0
        self._library_items_inventory = []
        self._library_items_ecosystem = [
            "room_control_station", "fabricator_station", "simulator_station", "camera_item",
            "duty_module_tile", "network_contact_module", "portal_frame", "light_emitter",
            "material_slot", "wall_tile", "floor_tile", "acoustic_panel",
        ]
        self._hotbar_slots = [
            "select", "inspect", "render", "pan", "focus", "aperture", "material", "attach", "place",
        ]
        self._hotbar_index = 0
        self._raytrace_inspect_active = False
        self._raytrace_anchor_eye = None
        self._raytrace_anchor_target = None
        self._menu_anchor_eye = None
        self._menu_anchor_target = None
        self._hover_owner_label = "none"
        self._hover_owner_obj = None
        self._hover_material_label = "unknown"
        self._hover_dist_m = float('inf')
        self._inspect_poll_hz = 4.0
        self._inspect_last_poll_s = 0.0
        self._inspect_last_eye = None
        self._inspect_last_target = None

    # ── GL init ───────────────────────────────────────────────────────────────

    def init_gl(self):
        import ctypes
        from OpenGL.GL import (
            GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE,
            GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
            glGenVertexArrays, glGenBuffers, glBindVertexArray,
            glBindBuffer, glBufferData, glEnableVertexAttribArray,
            glVertexAttribPointer,
        )
        self._p_col = _prog((_HUD2D_VS, GL_VERTEX_SHADER),
                            (_HUD2D_FS, GL_FRAGMENT_SHADER))
        self._p_tex = _prog((_HUD2D_TEX_VS, GL_VERTEX_SHADER),
                            (_HUD2D_TEX_FS, GL_FRAGMENT_SHADER))
        self._qvao = glGenVertexArrays(1)
        self._qvbo = glGenBuffers(1)
        glBindVertexArray(self._qvao)
        glBindBuffer(GL_ARRAY_BUFFER, self._qvbo)
        glBufferData(GL_ARRAY_BUFFER, 32, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glBindVertexArray(0)
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

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def open(self) -> bool:
        return self._open

    @property
    def hud_mode(self) -> str:
        return self._hud_mode

    def _set_hud_mode(self, mode: str) -> None:
        if mode not in (self.HUD_FULL, self.HUD_INFO, self.HUD_OFF):
            return
        self._hud_mode = str(mode)
        self._open = (self._hud_mode == self.HUD_FULL)
        self._drag = -1
        if self._open and self._cam is not None:
            self._menu_anchor_eye = np.array(self._cam.eye, np.float64)
            self._menu_anchor_target = np.array(self._cam.target, np.float64)
            if hasattr(self._cam, '_forced_eye'):
                self._cam._forced_eye = np.array(self._menu_anchor_eye, np.float64)
        else:
            self._menu_anchor_eye = None
            self._menu_anchor_target = None
            if self._cam is not None and hasattr(self._cam, '_forced_eye'):
                self._cam._forced_eye = None

    def cycle_hud_mode(self) -> str:
        if self._hud_mode == self.HUD_FULL:
            self._set_hud_mode(self.HUD_INFO)
        elif self._hud_mode == self.HUD_INFO:
            self._set_hud_mode(self.HUD_OFF)
        else:
            self._set_hud_mode(self.HUD_FULL)
        return self._hud_mode

    def toggle(self):
        if self._open:
            self._set_hud_mode(self.HUD_OFF)
        else:
            self._set_hud_mode(self.HUD_FULL)

    def close(self):
        self._set_hud_mode(self.HUD_OFF)

    def enforce_menu_camera_lock(self) -> None:
        if not self._open or self._cam is None:
            return
        if self._menu_anchor_eye is None or self._menu_anchor_target is None:
            self._menu_anchor_eye = np.array(self._cam.eye, np.float64)
            self._menu_anchor_target = np.array(self._cam.target, np.float64)
            return
        if hasattr(self._cam, '_forced_eye'):
            self._cam._forced_eye = np.array(self._menu_anchor_eye, np.float64)
        self._cam.target = np.array(self._menu_anchor_target, np.float64)

    # ── Binding ───────────────────────────────────────────────────────────────

    def attach(self, cam, slider_panel, renderer, player_ctrl=None) -> None:
        """Bind live objects and build per-knob on_change callbacks.

        Each callback closes over the live objects so handle_event() needs no
        external context — it simply fires self._on_change[key](value).
        Re-call whenever the bound camera or renderer changes.
        """
        self._cam          = cam
        self._slider_panel = slider_panel
        self._renderer     = renderer
        self._player_ctrl  = player_ctrl

        c = cam
        p = slider_panel

        r = renderer  # direct renderer reference for immediate application

        def _cam_set(attr):
            return lambda v: setattr(c, attr, float(v))

        def _panel_set(pkey):
            # Write to both panel.values (for panel.changed() tracking) AND
            # apply directly to R so there is no one-frame lag.
            def _set(v):
                p.values[pkey] = v
                _apply_direct(pkey, v)
            return _set

        def _apply_direct(key, v):
            """Apply a panel key directly to the renderer/accumulator."""
            if r is None:
                return
            acc = getattr(r, '_sensor_acc', None)
            if key == 'ray_density':
                pass  # refresh_rays recalculated in main loop via panel.changed
            elif key == 'ray_exposure' or key == 'ray_gamma':
                r.set_ray_tonemap(
                    exposure=float(p.values.get('ray_exposure', 1.0)),
                    gamma=float(p.values.get('ray_gamma', 1.0)))
            elif key == 'sensor_rate' and acc is not None:
                acc._rows_per_frame = max(1, int(round(float(v))))
            elif key == 'sensor_spp' and acc is not None:
                acc._samples_per_pixel = max(1, int(round(float(v))))
            elif key == 'sensor_fps':
                r._sensor_fps = max(0.0, float(v))
            elif key == 'mic_gain':
                r.show_mic = float(v) > 0.0
            elif key == 'pickup_gain':
                r.show_pickup = float(v) > 0.0
            elif key == 'air_diff' and acc is not None:
                acc._air_ds = float(v)
            elif key == 'air_spec' and acc is not None:
                acc._air_ss = float(v)
            elif key == 'air_aniso' and acc is not None:
                acc._air_an = float(v)

        def _aperture_set(v):
            v = float(np.clip(v, 0.0, 0.08))
            c.aperture = v
            p.values['lens_aperture'] = v

        def _ca_set(v):
            v = float(np.clip(v, 0.0, 0.02))
            c.ca = v
            p.values['lens_ca'] = v

        def _iso_set(v):
            p.values['sensor_iso'] = float(v)
            if r is not None:
                r.film.active_layer.iso = float(v)

        def _decay_set(v):
            v = max(0.0, float(v))
            p.values['film_decay'] = v
            acc = getattr(r, '_sensor_acc', None)
            if acc is not None:
                acc.half_life = v

        self._on_change = {
            'focal_mm':    _cam_set('focal_mm'),
            'focus_m':     _cam_set('focus_m'),
            'aperture':    _aperture_set,
            'ca':          _ca_set,
            'tilt_x':      lambda v: c.tilt_shift.__setitem__(0, float(v)),
            'tilt_y':      lambda v: c.tilt_shift.__setitem__(1, float(v)),
            'sensor_iso':  _iso_set,
            'sensor_gain': lambda v: None,
            'decay':       _decay_set,
        }
        if self._player_ctrl is not None:
            self._on_change['focus_pick_dist'] = self._player_ctrl.set_focus_pick_max_dist
            self._on_change['focus_cone_deg'] = self._player_ctrl.set_focus_cone_angle_deg
            self._on_change['focus_cone_rays'] = self._player_ctrl.set_focus_cone_rays
        for key in ('ray_density', 'ray_exposure', 'ray_gamma',
                    'sensor_rate', 'sensor_spp', 'sensor_fps', 'frame_step',
                    'mic_gain', 'pickup_gain',
                    'segs', 'plate_th', 'dx',
                    'air_diff', 'air_spec', 'air_aniso'):
            if key in p.values:
                self._on_change[key] = _panel_set(key)

        self._pull()
        self._sync_render_mode()
        if player_ctrl is not None and hasattr(player_ctrl, 'focus_ray_engine'):
            self._focus_ray_engine = str(player_ctrl.focus_ray_engine)
            if hasattr(player_ctrl, 'focus_pick_max_dist'):
                self._values['focus_pick_dist'] = float(player_ctrl.focus_pick_max_dist)
            if hasattr(player_ctrl, 'focus_cone_angle_deg'):
                self._values['focus_cone_deg'] = float(player_ctrl.focus_cone_angle_deg)
            if hasattr(player_ctrl, 'focus_cone_rays'):
                self._values['focus_cone_rays'] = float(player_ctrl.focus_cone_rays)

        # Ensure a CameraComputer is present in cam.software so auto modes
        # have somewhere to write.  Create one if absent.
        try:
            from camera_software import CameraComputer as _CC
            if not any(isinstance(sw, _CC) for sw in cam.software):
                cam.software.append(_CC())
        except Exception:
            pass
        # Keep the controls discoverable by the global action manager.
        try:
            _ = self.hierarchy_nodes()
        except Exception:
            pass

    @property
    def freeze_player_motion(self) -> bool:
        return bool(self._open)

    def request_raytrace_render(self) -> None:
        self._cam_render_mode = RenderMode.RAYTRACE
        self.apply_render_mode()
        self._raytrace_inspect_active = True
        if self._cam is not None:
            self._raytrace_anchor_eye = np.array(self._cam.eye, np.float64)
            self._raytrace_anchor_target = np.array(self._cam.target, np.float64)
        if self._renderer is not None and hasattr(self._renderer, 'reset_sensor'):
            self._renderer.reset_sensor()

    def _clear_render_if_moved(self, eps_m: float = 0.55) -> None:
        if not self._raytrace_inspect_active or self._cam is None:
            return
        if self._raytrace_anchor_eye is None or self._raytrace_anchor_target is None:
            return
        eye_now = np.array(self._cam.eye, np.float64)
        tgt_now = np.array(self._cam.target, np.float64)
        if (float(np.linalg.norm(eye_now - self._raytrace_anchor_eye)) > float(eps_m)
                or float(np.linalg.norm(tgt_now - self._raytrace_anchor_target)) > float(eps_m)):
            self._raytrace_inspect_active = False
            if self._renderer is not None and hasattr(self._renderer, 'reset_sensor'):
                self._renderer.reset_sensor()

    def update_hover_pick(self, mouse_pos: tuple[int, int], win_w: int, win_h: int,
                          duty_stations: list, cameras: list) -> None:
        if (not self._raytrace_inspect_active
                or self._cam is None
                or self._player_ctrl is None
                or not hasattr(self._player_ctrl, 'pick_interactable_with_ray')):
            return
        _now_s = float(time.perf_counter())
        _min_dt = 1.0 / max(0.5, float(self._inspect_poll_hz))
        if (_now_s - float(self._inspect_last_poll_s)) < _min_dt:
            return
        self._inspect_last_poll_s = _now_s

        mx, my = int(mouse_pos[0]), int(mouse_pos[1])
        eye = np.array(self._cam.eye, np.float64)
        target = np.array(self._cam.target, np.float64)
        if self._inspect_last_eye is not None and self._inspect_last_target is not None:
            _eye_d = float(np.linalg.norm(eye - self._inspect_last_eye))
            _tgt_d = float(np.linalg.norm(target - self._inspect_last_target))
            if _eye_d <= 1e-4 and _tgt_d <= 1e-4:
                return
        self._inspect_last_eye = eye.copy()
        self._inspect_last_target = target.copy()
        fwd = target - eye
        nf = float(np.linalg.norm(fwd))
        if nf <= 1e-9:
            return
        fwd /= nf
        up = np.array([0.0, 0.0, 1.0], np.float64)
        if abs(float(np.dot(fwd, up))) > 0.97:
            up = np.array([0.0, 1.0, 0.0], np.float64)
        right = np.cross(fwd, up)
        nr = float(np.linalg.norm(right))
        if nr <= 1e-9:
            return
        right /= nr
        up = np.cross(right, fwd)
        up /= max(float(np.linalg.norm(up)), 1e-9)
        nx = (2.0 * ((float(mx) + 0.5) / max(1.0, float(win_w)))) - 1.0
        ny = 1.0 - (2.0 * ((float(my) + 0.5) / max(1.0, float(win_h))))
        tan_half = math.tan(float(self._cam.fov_y_rad()) * 0.5)
        aspect = float(win_w) / max(1.0, float(win_h))
        rd = fwd + (nx * aspect * tan_half) * right + (ny * tan_half) * up
        nd = float(np.linalg.norm(rd))
        if nd <= 1e-9:
            return
        rd /= nd
        owner, dist = self._player_ctrl.pick_interactable_with_ray(
            eye,
            rd,
            duty_stations,
            cameras,
        )
        if owner is None:
            self._hover_owner_obj = None
            self._hover_owner_label = "none"
            self._hover_material_label = "unknown"
            self._hover_dist_m = float('inf')
            return
        self._hover_owner_obj = owner
        self._hover_owner_label = str(
            getattr(owner, 'label', getattr(owner, 'name', owner.__class__.__name__))
        )
        _mat = getattr(owner, 'material_slot', None)
        if _mat is None:
            _mat = getattr(owner, 'material', None)
        if _mat is None:
            _mat = getattr(owner, 'material_name', None)
        self._hover_material_label = str(_mat) if _mat is not None else "unknown"
        self._hover_dist_m = float(dist)

    def _draw_hud_shell(self, win_w: int, win_h: int, full_shell: bool = True) -> None:
        if full_shell:
            # Full HUD backdrop so there is explicit structure under the controls.
            self._draw_quad(0, 0, win_w, win_h, (0.01, 0.02, 0.05, 0.52), win_w, win_h)

        # Top multiline status bar.
        top_h = 70
        self._draw_quad(0, 0, win_w, top_h,
                (0.05, 0.10, 0.16, 0.94) if full_shell else (0.03, 0.06, 0.10, 0.72),
                win_w, win_h)
        active_tool = self._hotbar_slots[self._hotbar_index]
        self._draw_text(f"tool: {active_tool}", 14, 8, win_w, win_h)
        self._draw_text(f"object: {self._hover_owner_label}", 14, 28, win_w, win_h)
        _dist_txt = "n/a" if not np.isfinite(self._hover_dist_m) else f"{self._hover_dist_m:.2f}m"
        self._draw_text(f"material: {self._hover_material_label}   ray: {_dist_txt}", 14, 48, win_w, win_h)
        _owner = self._hover_owner_obj
        if _owner is not None and hasattr(_owner, "unfinished_tooltip_lines"):
            _tips = list(_owner.unfinished_tooltip_lines())
            for i, line in enumerate(_tips[:2]):
                self._draw_text(str(line), 340, 8 + i * 20, win_w, win_h)

        if full_shell:
            # Left library palette with tabs.
            lp_x, lp_y, lp_w = 12, top_h + 10, 360
            lp_h = max(220, win_h - top_h - 120)
            self._draw_quad(lp_x, lp_y, lp_w, lp_h, (0.06, 0.09, 0.12, 0.92), win_w, win_h)
            tab_w = (lp_w - 16) // 2
            for ti, name in enumerate(self._library_tabs):
                tx = lp_x + 8 + ti * (tab_w + 4)
                active = (ti == self._library_active_tab)
                self._draw_quad(tx, lp_y + 6, tab_w, 26,
                                (0.24, 0.42, 0.60, 0.95) if active else (0.15, 0.18, 0.23, 0.95),
                                win_w, win_h)
                self._draw_text(name.upper(), tx + 7, lp_y + 11, win_w, win_h)
            if self._library_active_tab == 0 and self._player_ctrl is not None and hasattr(self._player_ctrl, 'backpack'):
                _bp = dict(self._player_ctrl.backpack)
                items = [f"{k} x{int(v)}" for k, v in sorted(_bp.items()) if int(v) > 0]
            else:
                items = self._library_items_ecosystem
            iy = lp_y + 42
            max_rows = max(1, (lp_h - 52) // 20)
            if items:
                for idx, name in enumerate(items[:max_rows]):
                    self._draw_text(f"- {name}", lp_x + 10, iy + idx * 20, win_w, win_h)
            elif self._library_active_tab == 0:
                self._draw_text("- empty backpack", lp_x + 10, iy, win_w, win_h)

        # Bottom hotbar, centered.
        hb_w = 9 * 62 + 8 * 6
        hb_h = 52
        hb_x = (win_w - hb_w) // 2
        hb_y = win_h - hb_h - 10
        self._draw_quad(hb_x - 8, hb_y - 6, hb_w + 16, hb_h + 12, (0.05, 0.08, 0.12, 0.92), win_w, win_h)
        for i, name in enumerate(self._hotbar_slots):
            bx = hb_x + i * 68
            active = (i == self._hotbar_index)
            self._draw_quad(bx, hb_y, 62, hb_h,
                            (0.23, 0.52, 0.30, 0.98) if active else (0.15, 0.20, 0.27, 0.95),
                            win_w, win_h)
            self._draw_text(str(i + 1), bx + 4, hb_y + 3, win_w, win_h)
            self._draw_text(name[:7], bx + 8, hb_y + 22, win_w, win_h)

    def _get_computer(self):
        """Return the first CameraComputer in the bound camera's software list, or None."""
        if self._cam is None:
            return None
        try:
            from camera_software import CameraComputer as _CC
            for sw in self._cam.software:
                if isinstance(sw, _CC):
                    return sw
        except Exception:
            pass
        return None

    def _pull(self) -> None:
        """Read current values from bound cam + slider_panel + renderer into _values."""
        if self._cam is None:
            return
        c, p, r = self._cam, self._slider_panel, self._renderer
        self._values['focal_mm'] = float(c.focal_mm)
        self._values['focus_m']  = float(c.focus_m)
        self._values['aperture'] = float(c.aperture)
        self._values['ca']       = float(c.ca)
        self._values['tilt_x']   = float(c.tilt_shift[0])
        self._values['tilt_y']   = float(c.tilt_shift[1])
        if r is not None:
            acc = getattr(r, '_sensor_acc', None)
            self._values['sensor_fps']  = float(getattr(r, '_sensor_fps', self._values['sensor_fps']))
            self._values['sensor_iso']  = float(getattr(r.film.active_layer, 'iso', self._values['sensor_iso']))
            if acc is not None:
                self._values['decay']       = float(acc.half_life)
                self._values['sensor_rate'] = float(acc._rows_per_frame)
                self._values['sensor_spp']  = float(acc._samples_per_pixel)
                self._values['air_diff']    = float(acc._air_ds)
                self._values['air_spec']    = float(acc._air_ss)
                self._values['air_aniso']   = float(acc._air_an)
        if p is not None:
            for key in ('ray_density', 'ray_exposure', 'ray_gamma',
                        'frame_step', 'mic_gain', 'pickup_gain',
                        'segs', 'plate_th', 'dx'):
                if key in p.values:
                    self._values[key] = p.values[key]
        if self._player_ctrl is not None and hasattr(self._player_ctrl, 'focus_ray_engine'):
            self._focus_ray_engine = str(self._player_ctrl.focus_ray_engine)
            if hasattr(self._player_ctrl, 'focus_pick_max_dist'):
                self._values['focus_pick_dist'] = float(self._player_ctrl.focus_pick_max_dist)
            if hasattr(self._player_ctrl, 'focus_cone_angle_deg'):
                self._values['focus_cone_deg'] = float(self._player_ctrl.focus_cone_angle_deg)
            if hasattr(self._player_ctrl, 'focus_cone_rays'):
                self._values['focus_cone_rays'] = float(self._player_ctrl.focus_cone_rays)
        if r is not None:
            self._show_illum  = (r._layers[7] != LAYER_HIDDEN)
            self._show_sensor = (r._layers[8] != LAYER_HIDDEN)
            self._enlarger    = bool(getattr(self._renderer, '_enlarger_mode', False))

    def _apply_value(self, key: str, value) -> None:
        """Store value and fire its on_change callback."""
        self._values[key] = value
        cb = self._on_change.get(key)
        if cb is not None:
            cb(value)

    def apply_render_mode(self) -> None:
        """Push current render-mode selection onto the bound renderer."""
        if self._renderer is None:
            return
        m = self._cam_render_mode
        if m is RenderMode.C:
            self._renderer._layers[7] = LAYER_HIDDEN
            self._renderer._layers[8] = LAYER_HIDDEN
        elif m is RenderMode.GL:
            self._renderer._layers[7] = LAYER_HIDDEN
            self._renderer._layers[8] = LAYER_HIDDEN
        elif m is RenderMode.HYBRID:
            self._renderer._layers[7] = LAYER_OPAQUE
            self._renderer._layers[8] = LAYER_HIDDEN
        else:  # RAYTRACE
            self._renderer._layers[7] = LAYER_HIDDEN
            self._renderer._layers[8] = LAYER_OPAQUE
        # Sync toggle buttons to match what the preset just set
        self._show_illum  = (self._renderer._layers[7] != LAYER_HIDDEN)
        self._show_sensor = (self._renderer._layers[8] != LAYER_HIDDEN)
        self._renderer._render_mode = m.value

    def _sync_render_mode(self) -> None:
        """Read bound renderer layer flags and set the matching render mode."""
        if self._renderer is None:
            return
        _stored = getattr(self._renderer, '_render_mode', None)
        if _stored in tuple(m.value for m in RenderMode):
            self._cam_render_mode = RenderMode(_stored)
            return
        l8 = self._renderer._layers[7]
        l9 = self._renderer._layers[8]
        if l9 != LAYER_HIDDEN:
            self._cam_render_mode = RenderMode.RAYTRACE
        elif l8 != LAYER_HIDDEN:
            self._cam_render_mode = RenderMode.HYBRID
        else:
            self._cam_render_mode = RenderMode.C

    # ── Layout helpers ────────────────────────────────────────────────────────

    @property
    def render_mode(self) -> 'RenderMode':
        return self._cam_render_mode

    def _panel_h(self) -> int:
        return (self.PAD_TOP
                + len(self._SLIDERS) * self.ROW
                + 10
                + len(self._AUTOS) * self.ROW
                + 10           # sep before render mode
                + self.ROW     # label row
                + self.ROW     # render-mode button row
                + self.ROW     # layer-toggle row (enlarger / illum / sensor)
                + self.ROW     # focus-ray engine row
                + 8)

    def _px(self, win_w: int) -> int:
        return win_w - self.PW - 20

    def _py(self, win_h: int) -> int:
        return (win_h - self._panel_h()) // 2

    def _slider_y(self, i: int, py: int) -> int:
        return py + self.PAD_TOP + i * self.ROW

    def _auto_y(self, i: int, py: int) -> int:
        return py + self.PAD_TOP + len(self._SLIDERS) * self.ROW + 10 + i * self.ROW

    def _render_mode_y(self, py: int) -> int:
        """Top of the render-mode section (label row)."""
        return (py + self.PAD_TOP
                + len(self._SLIDERS) * self.ROW + 10
                + len(self._AUTOS) * self.ROW + 10)

    def _track_rect(self, i: int, px: int, py: int):
        ry = self._slider_y(i, py)
        return (px + self.TX, ry + (self.ROW - self.TH) // 2, self.TW, self.TH)

    def _t(self, key: str) -> float:
        v, lo, hi = self._values[key], self._lo[key], self._hi[key]
        if self._log[key]:
            denom = math.log(max(hi, 1e-12)) - math.log(max(lo, 1e-12))
            t = ((math.log(max(v, 1e-12)) - math.log(max(lo, 1e-12))) / denom
                 if denom else 0.0)
        else:
            t = (v - lo) / (hi - lo) if hi != lo else 0.0
        return float(np.clip(t, 0.0, 1.0))

    def _v_from_t(self, key: str, t: float) -> float:
        lo, hi = self._lo[key], self._hi[key]
        t = float(np.clip(t, 0.0, 1.0))
        if self._log[key]:
            return math.exp(math.log(max(lo, 1e-12))
                            + t * (math.log(max(hi, 1e-12))
                                   - math.log(max(lo, 1e-12))))
        return lo + t * (hi - lo)

    # ── GL drawing primitives ─────────────────────────────────────────────────

    def _draw_quad(self, x, y, w, h, color, win_w, win_h):
        import ctypes, struct
        from OpenGL.GL import (
            glUseProgram, glGetUniformLocation, glUniform2f, glUniform4f,
            glBindVertexArray, glBindBuffer, glBufferSubData, glDrawArrays,
            GL_ARRAY_BUFFER, GL_TRIANGLE_STRIP,
        )
        glUseProgram(self._p_col)
        glUniform2f(glGetUniformLocation(self._p_col, b'uRes'),
                    float(win_w), float(win_h))
        glUniform4f(glGetUniformLocation(self._p_col, b'uColor'), *color)
        v = struct.pack('8f',
                        float(x),   float(y),
                        float(x+w), float(y),
                        float(x),   float(y+h),
                        float(x+w), float(y+h))
        glBindVertexArray(self._qvao)
        glBindBuffer(GL_ARRAY_BUFFER, self._qvbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, len(v), v)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

    def _get_tex(self, text: str):
        if text in self._text_cache:
            return self._text_cache[text]
        from OpenGL.GL import (
            glGenTextures, glBindTexture, glTexParameteri, glTexImage2D,
            GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
            GL_LINEAR, GL_RGBA, GL_UNSIGNED_BYTE,
        )
        pygame.font.init()
        f = pygame.font.SysFont("consolas,monospace", 13)
        s = f.render(str(text), True, (215, 225, 240), (10, 10, 22))
        w, h = s.get_size()
        raw = pygame.image.tobytes(s, "RGBA")
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glBindTexture(GL_TEXTURE_2D, 0)
        r = (tex, (w, h))
        self._text_cache[text] = r
        return r

    def _draw_text(self, text: str, x: int, y: int, win_w: int, win_h: int):
        import struct
        from OpenGL.GL import (
            glUseProgram, glGetUniformLocation, glUniform2f, glUniform1i,
            glActiveTexture, glBindTexture, glBindVertexArray, glBindBuffer,
            glBufferSubData, glDrawArrays,
            GL_ARRAY_BUFFER, GL_TRIANGLE_STRIP, GL_TEXTURE0, GL_TEXTURE_2D,
        )
        tex, (tw, th) = self._get_tex(text)
        glUseProgram(self._p_tex)
        glUniform2f(glGetUniformLocation(self._p_tex, b'uRes'),
                    float(win_w), float(win_h))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(glGetUniformLocation(self._p_tex, b'uTex'), 0)
        v = struct.pack('16f',
                        float(x),    float(y),    0.0, 0.0,
                        float(x+tw), float(y),    1.0, 0.0,
                        float(x),    float(y+th), 0.0, 1.0,
                        float(x+tw), float(y+th), 1.0, 1.0)
        glBindVertexArray(self._tvao)
        glBindBuffer(GL_ARRAY_BUFFER, self._tvbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, len(v), v)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev) -> bool:
        """Return True if the event was consumed.

        Requires attach() to have been called first.  No external context
        (cam / slider_panel) is needed — each knob fires its own callback.
        """
        if not self._open or self._cam is None:
            return False
        import pygame
        win_w, win_h = pygame.display.get_surface().get_size()
        px = self._px(win_w)
        py = self._py(win_h)
        ph = self._panel_h()

        if ev.type == pygame.KEYDOWN:
            if pygame.K_1 <= ev.key <= pygame.K_9:
                self._hotbar_index = int(ev.key - pygame.K_1)
                return True

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            top_h = 70
            lp_x, lp_y, lp_w = 12, top_h + 10, 360
            tab_w = (lp_w - 16) // 2
            if lp_x <= mx <= lp_x + lp_w and lp_y <= my <= lp_y + 34:
                for ti in range(2):
                    tx = lp_x + 8 + ti * (tab_w + 4)
                    if tx <= mx <= tx + tab_w:
                        self._library_active_tab = ti
                        return True
            if not (px <= mx <= px + self.PW and py <= my <= py + ph):
                return False
            for i, (key, *_) in enumerate(self._SLIDERS):
                ry = self._slider_y(i, py)
                if ry <= my <= ry + self.ROW:
                    tx, ty, tw, th = self._track_rect(i, px, py)
                    t = float(np.clip((mx - tx) / max(tw, 1), 0.0, 1.0))
                    self._apply_value(key, self._v_from_t(key, t))
                    self._drag = i
                    return True
            for i, (key, _) in enumerate(self._AUTOS):
                ry = self._auto_y(i, py)
                if ry <= my <= ry + self.ROW:
                    self._auto[key] = not self._auto[key]
                    cmp = self._get_computer()
                    if cmp is not None:
                        cmp.sync_flags(self._auto)
                    return True
            # Render mode buttons
            btn_y = self._render_mode_y(py) + self.ROW
            if btn_y <= my <= btn_y + self.ROW:
                _modes = [RenderMode.C, RenderMode.GL, RenderMode.HYBRID, RenderMode.RAYTRACE]
                bw = (self.PW - 20) // 4
                for bi, m in enumerate(_modes):
                    bx = px + 8 + bi * (bw + 4)
                    if bx <= mx <= bx + bw:
                        self._cam_render_mode = m
                        self.apply_render_mode()   # dispatch immediately
                        return True
            # Layer / display toggle buttons (Enlarger / Illum / Sensor)
            ltgl_y = self._render_mode_y(py) + 2 * self.ROW
            if ltgl_y <= my <= ltgl_y + self.ROW:
                bw3 = (self.PW - 16) // 3
                bxE = px + 8
                bxI = px + 8 + (bw3 + 4)
                bxS = px + 8 + 2 * (bw3 + 4)
                if bxE <= mx <= bxE + bw3:
                    self._enlarger = not self._enlarger
                    if self._renderer is not None:
                        self._renderer._enlarger_mode = self._enlarger
                    return True
                if bxI <= mx <= bxI + bw3:
                    self._show_illum = not self._show_illum
                    if self._renderer is not None:
                        self._renderer._layers[7] = (LAYER_OPAQUE
                                                     if self._show_illum
                                                     else LAYER_HIDDEN)
                    return True
                if bxS <= mx <= bxS + bw3:
                    self._show_sensor = not self._show_sensor
                    if self._renderer is not None:
                        self._renderer._layers[8] = (LAYER_OPAQUE
                                                     if self._show_sensor
                                                     else LAYER_HIDDEN)
                    return True
            focus_y = self._render_mode_y(py) + 3 * self.ROW
            if focus_y <= my <= focus_y + self.ROW:
                bwf = (self.PW - 16) // max(1, len(self._FOCUS_ENGINES))
                for bi, (eng_key, _eng_label) in enumerate(self._FOCUS_ENGINES):
                    bx = px + 8 + bi * (bwf + 4)
                    if bx <= mx <= bx + bwf:
                        self._focus_ray_engine = eng_key
                        if self._player_ctrl is not None and hasattr(self._player_ctrl, 'set_focus_ray_engine'):
                            self._player_ctrl.set_focus_ray_engine(eng_key)
                        return True
            return True  # absorb any click inside panel

        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self._drag >= 0:
                self._drag = -1
                return True
            return False

        if ev.type == pygame.MOUSEMOTION and self._drag >= 0:
            mx, _ = ev.pos
            key = self._SLIDERS[self._drag][0]
            tx, _, tw, _ = self._track_rect(self._drag, px, py)
            t = float(np.clip((mx - tx) / max(tw, 1), 0.0, 1.0))
            self._apply_value(key, self._v_from_t(key, t))
            return True

        return False

    # ── Render ────────────────────────────────────────────────────────────────

    def draw(self, win_w: int, win_h: int):
        if self._hud_mode == self.HUD_OFF or self._p_col is None:
            return
        # Pull current live values from cam/renderer every frame so the sliders
        # always reflect the actual state, not just what they were at attach time.
        self._pull()
        from OpenGL.GL import (
            glEnable, glDisable, glBlendFunc,
            GL_BLEND, GL_DEPTH_TEST, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            glBindVertexArray, glUseProgram, glBindTexture, GL_TEXTURE_2D,
        )
        px = self._px(win_w)
        py = self._py(win_h)
        ph = self._panel_h()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        if self._hud_mode == self.HUD_INFO:
            self._draw_hud_shell(win_w, win_h, full_shell=False)
            self._clear_render_if_moved()
            glEnable(GL_DEPTH_TEST)
            glBindVertexArray(0)
            glUseProgram(0)
            glBindTexture(GL_TEXTURE_2D, 0)
            return

        self._draw_hud_shell(win_w, win_h, full_shell=True)
        self._draw_quad(px, py, self.PW, ph, self._C_BG, win_w, win_h)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Title bar
        self._draw_quad(px, py, self.PW, self.PAD_TOP - 2,
                        self._C_TITLE, win_w, win_h)
        self._draw_text("CAMERA SETTINGS  [ E ] cycle HUD",
                        px + 8, py + 9, win_w, win_h)

        # Slider rows
        for i, (key, label, *_) in enumerate(self._SLIDERS):
            ry  = self._slider_y(i, py)
            tx, tty, tw, th = self._track_rect(i, px, py)
            # Track rail
            self._draw_quad(tx, tty, tw, th, self._C_TRACK, win_w, win_h)
            # Fill
            t  = self._t(key)
            fw = max(2, int(t * tw))
            self._draw_quad(tx, tty, fw, th, self._C_FILL, win_w, win_h)
            # Knob
            kx = tx + int(t * tw) - 4
            ky = tty + th // 2 - 6
            self._draw_quad(kx, ky, 8, 12, self._C_KNOB, win_w, win_h)
            # Label
            self._draw_text(label, px + 4, ry + 6, win_w, win_h)
            # Value readout
            val_str = f"{self._values[key]:{self._fmt[key]}}"
            self._draw_text(val_str, px + self.TX + self.TW + 6, ry + 6,
                            win_w, win_h)

        # Separator
        sep_y = py + self.PAD_TOP + len(self._SLIDERS) * self.ROW + 3
        self._draw_quad(px + 4, sep_y, self.PW - 8, 2,
                        self._C_SEP, win_w, win_h)

        # Auto checkboxes
        for i, (key, label) in enumerate(self._AUTOS):
            ry = self._auto_y(i, py)
            checked = self._auto[key]
            self._draw_quad(px + 8, ry + 6, 14, 14,
                            self._C_CHECK if checked else self._C_UNCHECK,
                            win_w, win_h)
            mark = "\u2713 " if checked else "  "
            self._draw_text(f"{mark} {label}", px + 28, ry + 6, win_w, win_h)

        # Render mode selector ─────────────────────────────────────────────
        rmy = self._render_mode_y(py)
        # Thin separator
        self._draw_quad(px + 4, rmy - 4, self.PW - 8, 2, self._C_SEP, win_w, win_h)
        self._draw_text("Render Mode", px + 8, rmy + 6, win_w, win_h)
        _modes  = [RenderMode.C, RenderMode.GL, RenderMode.HYBRID, RenderMode.RAYTRACE]
        _labels = ['C', 'OpenGL', 'Hybrid', 'Ray']
        bw = (self.PW - 20) // 4
        btn_y = rmy + self.ROW
        for bi, (m, lbl) in enumerate(zip(_modes, _labels)):
            bx = px + 8 + bi * (bw + 4)
            active = (self._cam_render_mode is m)
            self._draw_quad(bx, btn_y + 2, bw, self.ROW - 4,
                            self._C_FILL if active else self._C_TRACK,
                            win_w, win_h)
            self._draw_text(lbl, bx + 6, btn_y + 6, win_w, win_h)

        # Layer / display toggles: Enlarger · Illum · Sensor
        ltgl_y = rmy + 2 * self.ROW
        bw3 = (self.PW - 16) // 3
        _ltgls = [('Enlarger', self._enlarger),
                  ('Illum',    self._show_illum),
                  ('Sensor',   self._show_sensor)]
        for bi, (lbl, active) in enumerate(_ltgls):
            bx = px + 8 + bi * (bw3 + 4)
            self._draw_quad(bx, ltgl_y + 2, bw3, self.ROW - 4,
                            self._C_FILL if active else self._C_TRACK,
                            win_w, win_h)
            self._draw_text(lbl, bx + 6, ltgl_y + 6, win_w, win_h)

        focus_y = rmy + 3 * self.ROW
        self._draw_text("Focus Ray", px + 8, focus_y + 6, win_w, win_h)
        bwf = (self.PW - 16) // max(1, len(self._FOCUS_ENGINES))
        for bi, (eng_key, eng_label) in enumerate(self._FOCUS_ENGINES):
            bx = px + 8 + bi * (bwf + 4)
            active = (self._focus_ray_engine == eng_key)
            self._draw_quad(bx, focus_y + 2, bwf, self.ROW - 4,
                            self._C_FILL if active else self._C_TRACK,
                            win_w, win_h)
            self._draw_text(eng_label, bx + 6, focus_y + 6, win_w, win_h)

        self._clear_render_if_moved()

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
    p.add_argument("--ray-density", type=float, default=1.0,
                   help="Scene-wide ray density. Scales acoustic, EM, sensor, refresh, and debug ray budgets together")
    p.add_argument("--sound-energy-rays", type=int, default=1000,
                   help="Acoustic energy-to-ray conversion at ray-density=1")
    p.add_argument("--em-energy-rays", type=int, default=20000,
                   help="EM/light energy-to-ray conversion at ray-density=1")
    p.add_argument("--trace-rays", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--max-bounces", type=int, default=RAY_MAX_BOUNCES,
                   help="Maximum ray bounces")
    p.add_argument("--sensor-rays", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--sensor-gain", type=float, default=8.0,
                   help="Amplification factor for sensor-pass surface contributions (default 8.0).")
    p.add_argument("--ray-program-only", "--ray-trace-only", dest="ray_program_only",
                   action="store_true",
                   help="Skip the FDTD/plate physics worker and run only the static scene ray-trace/camera program")
    p.add_argument("--no-auto-sim", dest="no_auto_sim",
                   action="store_true", default=True,
                   help="Do not automatically start the FDTD/audio physics worker on launch; "
                        "physics can be started later via UI or key binding")
    p.add_argument("--render-mode", dest="render_mode",
                    choices=[m.value for m in RenderMode], default=RenderMode.C.value,
                    help="Player-experience render mode: c (default startup), gl, hybrid (GL + baked light field), "
                        "or raytrace (implies --ray-program-only)")
    p.add_argument("--lens-focal-mm", type=float, default=35.0,
                   help="Camera focal length in millimetres")
    p.add_argument("--lens-focus-m", type=float, default=1.6,
                   help="Camera focus distance in metres")
    p.add_argument("--lens-aperture", type=float, default=0.0,
                   help="Thin-lens aperture radius in metres; 0 is pinhole")
    p.add_argument("--lens-ca", type=float, default=0.0,
                   help="Chromatic aberration factor for the sensor camera")
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
    p.add_argument("--amr-build-backend", dest="amr_build_backend",
                   choices=["cpu", "gl"], default="cpu",
                   help="AMR grid build backend: cpu (default, stable) or gl (GPU compute, experimental)")
    p.add_argument("--gradient-order", dest="gradient_order", type=int, default=2,
                   choices=[2, 8],
                   help="AMR GL velocity gradient order: 2 (fast, default) or 8 (high-accuracy Fornberg)")
    p.add_argument("--headless-batch", dest="headless_batch", action="store_true",
                   help="Suppress per-step progress spans and tqdm bars for maximum throughput")
    p.add_argument("--benchmark-steps", dest="benchmark_steps", type=int, default=0,
                   help="If > 0, run a standalone AMR GL throughput benchmark for this many steps and exit")
    p.add_argument("--amr-cache-grid", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Enable disk cache for deterministic AMR topology builds")
    p.add_argument("--render-segs", type=int, default=60,
                   help="String FDTD/render segments per string")
    p.add_argument("--bridge-force-scale", type=float, default=BRIDGE_FORCE_SCALE,
                   help="Deprecated compatibility argument; structural coupling uses SI impedances")
    p.add_argument("--plate-theta", type=int, default=128,
                   help="Angular subdivisions for the rendered soundboard mesh")
    p.add_argument("--plate-radial", type=int, default=PLATE_RADIAL_SEGS,
                   help="Radial subdivisions for the rendered soundboard mesh")
    p.add_argument("--gpu-field-rays", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--gpu-segment-cap", type=int, default=GPU_RAY_SEGMENT_CAP,
                   help=argparse.SUPPRESS)
    p.add_argument("--gpu-dispatch-batch", type=int, default=GPU_DISPATCH_BATCH,
                   help=argparse.SUPPRESS)
    p.add_argument("--gpu-refresh-every", type=int, default=1,
                   help=argparse.SUPPRESS)
    p.add_argument("--gpu-refresh-rays", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--stage-light-field", action=argparse.BooleanOptionalAction,
                   default=False, help=argparse.SUPPRESS)
    p.add_argument("--stage-light-rays", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--stage-light-emitters", type=int, default=9,
                   help="Emitter samples across the diffuse stage light area")
    p.add_argument("--sensor-light-rays", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--stage-light-cache-dir", default=".cache",
                   help=argparse.SUPPRESS)
    p.add_argument("--rebuild-stage-light", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--initial-ray-field", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cpu-ray-overlay", action="store_true", help=argparse.SUPPRESS)
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
    p.add_argument("--station", default=None,
                   choices=["camera_designer"],
                   help="Open a specific design station instead of the full world render")
    p.add_argument("--level", default=None, metavar="PATH",
                   help="Scene save file (.yaml) to load on startup. "
                        "If omitted, starts in a blank space with room control + fabricator stations.")
    args = p.parse_args()
    if args.pressure_margin_cells < 0:
        p.error("--pressure-margin-cells must be non-negative")
    if args.pressure_pml_cells < 0:
        p.error("--pressure-pml-cells must be non-negative")
    if args.diagnostic_rest:
        args.excitation = "rest"
    if args.diagnostic_frames < 1:
        p.error("--diagnostic-frames must be positive")
    if args.ray_density < 0.0:
        p.error("--ray-density must be non-negative")
    _density = float(args.ray_density)
    if args.trace_rays is None:
        args.trace_rays = max(64, int(round(RAY_TRACE_RAYS * max(_density, 0.05))))
    if args.gpu_field_rays is None:
        args.gpu_field_rays = max(1, int(round(float(args.sound_energy_rays) * _density)))
    if args.gpu_refresh_rays is None:
        args.gpu_refresh_rays = args.gpu_field_rays
    if args.stage_light_rays is None:
        args.stage_light_rays = max(1, int(round(float(args.em_energy_rays) * _density)))
    if args.sensor_light_rays is None:
        args.sensor_light_rays = args.stage_light_rays
    if args.sensor_rays is None:
        args.sensor_rays = max(1, int(round((float(args.sound_energy_rays) + float(args.em_energy_rays)) * 0.5 * _density)))
    return args


def _run_benchmark(args) -> None:
    """Standalone AMR GL throughput benchmark.

    Builds a minimal uniform acoustic grid (no instrument geometry), runs
    ``args.benchmark_steps`` GPU steps, and prints a concise metrics table.
    Requires an OpenGL 4.3 context — call after pygame/GL initialisation.
    """
    import time
    import os
    import numpy as np

    # Set headless env var before importing acoustic_amr so the bypass flag is set.
    if args.headless_batch:
        os.environ["SPECTRAL_HEADLESS_BATCH"] = "1"

    from acoustic_amr import AcousticAMRGrid, AMRGLComputeBackend

    steps   = max(1, int(args.benchmark_steps))
    order   = int(args.gradient_order)

    # ── Build a small uniform test grid ──────────────────────────────────────
    dx = 0.02   # 20 mm cells → ~17 kHz bandwidth at 343 m/s
    side = 0.5  # half-metre cube
    t0_grid = time.perf_counter()
    grid = AcousticAMRGrid.build_uniform_box(
        x_min=-side, x_max=side,
        y_min=-side, y_max=side,
        z_min=-side, z_max=side,
        dx=dx,
    )
    t_grid = time.perf_counter() - t0_grid

    n_cells = grid.n_cells
    n_faces = grid.n_faces
    print(f"[benchmark] Grid: {n_cells} cells, {n_faces} faces  ({t_grid*1e3:.1f} ms build)")

    # ── Create GL backend ─────────────────────────────────────────────────────
    t0_init = time.perf_counter()
    backend = AMRGLComputeBackend(grid, gradient_order=order)
    t_init  = time.perf_counter() - t0_init

    t_stencil = 0.0
    if order == 8:
        # Stencil build is embedded in __init__ for order 8;
        # we report it as part of init time.
        t_stencil = t_init  # approximate: stencil dominates init for large grids

    print(f"[benchmark] Backend init: {t_init*1e3:.1f} ms  (gradient_order={order})")

    # ── Warm-up: 4 steps to trigger shader JIT and driver caching ─────────────
    backend.step(4)

    # ── Timed run ─────────────────────────────────────────────────────────────
    from OpenGL.GL import glFinish
    glFinish()
    t0 = time.perf_counter()
    backend.step(steps)
    glFinish()
    t_steps = time.perf_counter() - t0

    t_per_step  = t_steps / steps
    steps_per_s = steps / t_steps
    cells_per_s = n_cells * steps_per_s

    print()
    print("=" * 62)
    print(f"  AMR GL Throughput Benchmark")
    print("=" * 62)
    print(f"  Grid cells           : {n_cells:>12,}")
    print(f"  Grid faces           : {n_faces:>12,}")
    print(f"  Gradient order       : {order:>12}")
    print(f"  Steps                : {steps:>12,}")
    print(f"  Grid build time      : {t_grid*1e3:>10.2f} ms")
    print(f"  Backend init time    : {t_init*1e3:>10.2f} ms")
    print(f"  Total step time      : {t_steps*1e3:>10.2f} ms")
    print(f"  Per-step time        : {t_per_step*1e6:>10.1f} µs")
    print(f"  Steps/sec            : {steps_per_s:>10.1f}")
    print(f"  Cell·steps/sec       : {cells_per_s/1e6:>10.2f} M")
    print("=" * 62)


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

    if args.benchmark_steps > 0:
        _run_benchmark(args)
        pygame.quit()
        return

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
        "amr_backend": str(args.amr_build_backend),
        "amr_cache_grid": bool(args.amr_cache_grid),
    }

    # ── Phase 1: Build scene + geometry (fast, no physics) ───────────────────
    # Do this FIRST so the guitar is visible while the AMR grid builds.
    _t0_scene = time.monotonic()
    scene = None
    print("Room-only startup: guitar scene disabled.", flush=True)

    if scene is not None and _HAS_BRIDGE:
        outline, body_h, _early_bridge = _extract_guitar_geometry(scene)
        outline = np.asarray(outline, dtype=np.float32)
    else:
        try:
            from acoustic_fdtd_bridge import _default_guitar_outline
            outline = np.asarray(_default_guitar_outline(), dtype=np.float32)
        except Exception:
            outline = np.zeros((64, 2), dtype=np.float32)
        body_h = BODY_H
    n_str = int(config["n_strings"])
    paths = _string_paths(outline, body_h, n_strings=n_str,
                          n_segs=args.render_segs,
                          fret=config["fret"])
    print(f"  Scene ready in {(time.monotonic()-_t0_scene)*1000:.0f}ms", flush=True)

    # ── Phase 2: CPU ray trace (fast, needs only scene) ──────────────────────
    ray_segs = None
    meta: dict = {}
    if scene is not None and _HAS_RAY and args.cpu_ray_overlay:
        print("Tracing geometry ...", flush=True)
        try:
            _ray_diag_update("cpu_ray_overlay:trace:start",
                             n_rays=int(args.trace_rays), max_bounces=int(args.max_bounces))
            ray_segs, meta = _trace_fn(scene, n_rays=args.trace_rays, max_bounces=args.max_bounces)
            _ray_diag_update("cpu_ray_overlay:trace:done",
                             n_segments=int(len(ray_segs)),
                             meta={k: str(v) for k, v in list(meta.items())[:8]})
        except BaseException as exc:
            _report_exception("CPU ray overlay trace", exc)
            raise
        print(f"  {len(ray_segs)} segments", flush=True)

    # ── Phase 3: Stage light field (GPU, uses outline from scene — no physics needed) ──
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
    total        = _excitation_total_samples(args.excitation, args.diagnostic_frames)
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
                 base_counter, baseline_light_total_rays, _bsl_stex) = _gpu_ray_field(
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
        all_sources = _sensor_scene_sources(
            outline,
            soundboard_sources + string_sources,
            light_rays=int(args.sensor_light_rays),
            light_emitters=int(args.stage_light_emitters))
        print(f"  {len(soundboard_sources)} soundboard + "
              f"{len(string_sources)} string emission sources", flush=True)

        ce.reset()
        _schedule_excitation(ce, args.excitation, n_strings=n_str)

        print("GPU ray field integration ...", flush=True)
        try:
            ray_field_bands, rb0, rb1, gpu_seg_vbo, gpu_seg_cap, _gpu_counter, ray_field_total_rays, _init_sensor_tex = _gpu_ray_field(
                scene, outline, body_h,
                sources=all_sources,
                max_bounces=args.max_bounces,
                segment_cap=args.gpu_segment_cap,
                dispatch_batch=args.gpu_dispatch_batch,
                sensor_pos=np.asarray(Camera().eye, np.float32),
                sensor_pos_space="world",
                include_stage=True,
                sensor_rays=int(args.sensor_rays),
                sensor_gain=float(args.sensor_gain),
                sensor_w=WIN_W, sensor_h=WIN_H,
                film=R.film)
        except BaseException as exc:
            _report_exception("initial GPU ray field", exc)
            raise
        ray_field_bounds = (rb0, rb1)
        if args.gpu_smoke_exit:
            _quit_pygame_with_diag("initial-gpu-smoke-exit")
            return
    elif args.initial_ray_field:
        print("GPU ray field prepass skipped: scene/GPU ray field unavailable.", flush=True)

    # ── Phase 4: Stub info — geometry only, no AMR physics ───────────────────
    # Build minimal info from the scene outline so the Renderer can show the
    # guitar body, stage, and strings immediately before physics is ready.
    _dx_early = float(config["dx"])
    _pad_early = int(args.pressure_margin_cells)
    _ox, _oy = outline[:, 0], outline[:, 1]
    _gx_min_e = float(_ox.min()) - _pad_early * _dx_early
    _gx_max_e = float(_ox.max()) + _pad_early * _dx_early
    _gy_min_e = float(_oy.min()) - _pad_early * _dx_early
    # Extend Y to cover the full neck + headstock (nut at ~0.578 m, headstock ~0.708 m)
    _y_headstock_tip = BRIDGE_POS[1][1] + SCALE_LENGTH_M + 0.14
    _gy_max_e = max(float(_oy.max()), _y_headstock_tip) + _pad_early * _dx_early
    _gz_min_e = -_pad_early * _dx_early
    _gz_max_e = body_h + _pad_early * _dx_early
    info: dict = {
        'Nx': max(4, math.ceil((_gx_max_e - _gx_min_e) / _dx_early)),
        'Ny': max(4, math.ceil((_gy_max_e - _gy_min_e) / _dx_early)),
        'Nz': max(4, math.ceil((_gz_max_e - _gz_min_e) / _dx_early)),
        'dx': _dx_early,
        'gx_min': _gx_min_e, 'gy_min': _gy_min_e, 'gz_min': _gz_min_e,
        'n_pml': int(args.pressure_pml_cells),
    }

    # ── Phase 5: Start physics worker in background unless ray-program-only ──
    physics = None
    build_bar = None
    build_bar_value = 0
    # Resolve render mode; --render-mode=raytrace implies --ray-program-only
    _render_mode = RenderMode(args.render_mode)
    if _render_mode is RenderMode.RAYTRACE:
        args.ray_program_only = True

    physics_ready = True
    print("Room-only startup: guitar physics worker skipped.", flush=True)
    if args.ray_program_only:
        print("Ray-program-only mode: skipping FDTD/plate physics worker.", flush=True)
        args.capture_wav = ""
    elif args.no_auto_sim:
        print("[no-auto-sim] Physics worker deferred — will not start automatically.", flush=True)

    capture = None
    if args.capture_wav:
        try:
            capture = _CaptureSet(str(args.capture_wav), str(args.capture_source))
        except Exception as _cap_err:
            print(f"[capture] disabled: {_cap_err}", flush=True)
            capture = None

    # ── Phase 6: Renderer — visible immediately while physics builds ──────────
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
                 show_mic=args.mic,
                 no_stage=True)
    R.cam.focal_mm = float(np.clip(args.lens_focal_mm, 12.0, 180.0))
    R.cam.focus_m = float(np.clip(args.lens_focus_m, 0.05, 20.0))
    R.cam.aperture = float(np.clip(args.lens_aperture, 0.0, 0.08))
    R.cam.ca = float(np.clip(args.lens_ca, 0.0, 0.02))
    R._render_mode = _render_mode.value
    if args.ray_program_only:
        R._layers = [LAYER_ALPHA, LAYER_HIDDEN, LAYER_HIDDEN,
                     LAYER_ALPHA, LAYER_OPAQUE, LAYER_ALPHA, LAYER_ALPHA,
                     LAYER_ALPHA, LAYER_OPAQUE]

    try:
        R.set_ray_lighting(meta)
    except (NameError, KeyError):
        pass

    # ── Phase 6b: Initial GPU ray field + sensor accumulator (no physics needed) ─
    # Build BVH geometry and sensor accumulator only — no blocking initial ray dispatch.
    # The streaming pump (notify_frame / pump_forward) fires rays progressively per frame.
    if scene is not None and args.gpu_rays:
        print("Building sensor accumulator from geometry (streaming mode) ...", flush=True)
        try:
            (ray_field_bands, _rb0, _rb1,
             gpu_seg_vbo, gpu_seg_cap, _gpu_counter,
             ray_field_total_rays, _init_sensor_acc) = _gpu_ray_field(
                scene, outline, body_h,
                sources=[],
                max_bounces=args.max_bounces,
                segment_cap=args.gpu_segment_cap,
                dispatch_batch=args.gpu_dispatch_batch,
                sensor_pos=np.asarray(R.cam.eye, np.float32),
                sensor_pos_space="world",
                include_stage=True,
                sensor_rays=0,
                sensor_gain=float(args.sensor_gain),
                sensor_w=WIN_W, sensor_h=WIN_H,
                sim_bounds=(R._bmin, R._bmax))
            ray_field_bounds = (_rb0, _rb1)
            R.replace_ray_field(ray_field_bands, ray_field_bounds,
                                gpu_seg_vbo, gpu_seg_cap, _gpu_counter,
                                total_rays=ray_field_total_rays)
            if _init_sensor_acc is not None and hasattr(_init_sensor_acc, 'notify_frame'):
                R.attach_sensor_accumulator(_init_sensor_acc)
                # Seed the streaming source worker with the initial geometry
                # so pump_forward() has a dispatch ring ready on the first frame.
                _init_sensor_acc.notify_frame(
                    frame=None,
                    info=info,
                    outline=outline,
                    body_h=body_h,
                    paths=paths,
                    active_strings=active_strings,
                    refresh_rays=int(args.gpu_field_rays),
                    sensor_light_rays=int(args.sensor_light_rays),
                    stage_light_emitters=int(args.stage_light_emitters),
                )
        except BaseException as exc:
            _report_exception("initial geometry GPU ray field", exc)
            raise

    # gpu_smoke_exit: needs a physics frame; poll until one arrives then exit.
    if args.gpu_smoke_exit:
        if physics is None:
            _quit_pygame_with_diag("ray-program-only-gpu-smoke-exit")
            return
        if scene is not None and args.gpu_rays:
            print("GPU dynamic ray field smoke frame ...", flush=True)
            frame = None
            while frame is None:
                for msg in physics.poll():
                    if msg.get("type") == "frame":
                        frame = msg["frame"]
                        break
                    if msg.get("type") == "ready":
                        info = msg["info"]
                        body_h = float(msg["body_h"])
                        outline = np.asarray(msg["outline"], dtype=np.float32)
                        paths = _string_paths(outline, body_h, n_strings=n_str,
                                              n_segs=args.render_segs,
                                              fret=int(msg["config"].get("fret", 0)))
                        R.rebuild_physics(None, info, paths, body_h)
                        physics_ready = True
                    if msg.get("type") == "error":
                        raise RuntimeError(_worker_error_message(msg))
                pygame.event.pump()
            if physics_ready and 'plate_active_2d' in info:
                dyn_sources = _instant_soundboard_sources(
                    frame.plate, info['plate_active_2d'], outline, body_h,
                    total_rays=int(args.gpu_refresh_rays or args.gpu_field_rays),
                    stride=4)
            else:
                dyn_sources = []
            dyn_sources += _string_emission_sources(
                paths, body_h,
                total_rays_per_string=max(
                    64,
                    int(args.gpu_refresh_rays or args.gpu_field_rays) // max(1, n_str) // 16),
                active_strings=active_strings)
            dyn_sources = _sensor_scene_sources(
                outline, dyn_sources,
                light_rays=int(args.sensor_light_rays),
                light_emitters=int(args.stage_light_emitters))
            if dyn_sources:
                try:
                    rbands, rb0, rb1, rvbo, rcap, rcounter, _rtotal, _smoke_stex = _gpu_ray_field(
                        scene, outline, body_h,
                        sources=dyn_sources,
                        max_bounces=args.max_bounces,
                        segment_cap=args.gpu_segment_cap,
                        dispatch_batch=args.gpu_dispatch_batch,
                        sensor_pos=np.asarray(Camera().eye, np.float32),
                        sensor_pos_space="world",
                        include_stage=True,
                        sensor_rays=int(args.sensor_rays),
                        sensor_gain=float(args.sensor_gain),
                        film=R.film)
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
        if physics is not None:
            physics.close()
        _quit_pygame_with_diag("dynamic-gpu-smoke-exit")
        return

    panel = _SliderPanel()
    panel._btn_callback = R.reset_sensor  # "New Frame [C]" button
    # Sync panel default values with whatever args resolved to
    panel.values['ray_density'] = float(args.ray_density)
    panel.values['segs']     = int(args.render_segs)
    panel.values['plate_th'] = int(args.plate_theta)
    panel.values['dx']       = float(args.dx)
    panel.values['ray_exposure'] = 1.0
    panel.values['ray_gamma'] = float(GPU_RAY_FIELD_GAMMA)
    panel.values['mic_gain'] = 1.0 if args.mic else 0.0
    panel.values['pickup_gain'] = 1.0 if args.pickup else 0.0
    panel.values['lens_aperture'] = float(R.cam.aperture)
    panel.values['lens_ca'] = float(R.cam.ca)
    panel._prev = dict(panel.values)
    R.set_ray_tonemap(
        exposure=float(panel.values['ray_exposure']),
        gamma=float(panel.values['ray_gamma']))

    # Declare room workspace early so later code (player spawn, camera items, etc.)
    # can reference it regardless of whether _HAS_ROOM_STATION is True.
    _room_ws      = None
    _room_station = None

    # ── Player controller ─────────────────────────────────────────────────────
    _player_cfg_path = _config_path("player", "default.yaml")
    _player_cfg = _load_yaml_file(_player_cfg_path)
    player_ctrl = (_PlayerController(R.cam, _player_cfg,
                                     lens_loader   = LensSpec.load,
                                     sensor_loader = SensorSpec.load)
                   if _HAS_PLAYER_CTRL and _player_cfg else None)

    # ── Camera HUD panel ──────────────────────────────────────────────────────
    _camera_panel = None
    if _HAS_CAMERA_PANEL:
        _camera_panel = _CameraHudPanel()
        _camera_panel.init_gl()

    # ── Player camera settings panel ─────────────────────────────────────────
    _player_cam_panel = _PlayerCameraPanel()
    _player_cam_panel.init_gl()
    _player_cam_panel.attach(R.cam, panel, R, player_ctrl=player_ctrl)
    _player_cam_panel._cam_render_mode = _render_mode
    _player_cam_panel.apply_render_mode()

    # ── Doc renderer — HUD second channel ────────────────────────────────────
    # Registered into the ShaderFrameWalker as "doc_composite" so every HUD
    # panel is just another channel in the same pipeline (no special shader).
    _doc_rdr = None
    _doc_init_err: Exception | None = None
    _doc_slider_ids  = {}   # stable node-id maps, persist across frames
    _doc_camera_ids  = {}
    try:
        from doc_renderer import DocRenderer as _DocRenderer
        from controls import Panel as _DocPanel
        _doc_rdr = _DocRenderer(WIN_W, WIN_H)
        _doc_rdr.init_gl()
        # Best-effort glyph atlas from pygame monospace font via PIL shim
        try:
            from PIL import ImageFont as _PILFont
            _pil_font = _PILFont.load_default()
            _doc_rdr.load_glyph_atlas_from_pil(_pil_font)
        except Exception:
            pass  # layout-only mode; no text glyphs
        # Build persistent Panel specs from HUD class descriptors
        _doc_slider_spec = _DocPanel(
            name="slider_panel",
            label="Render Controls",
            knobs=list(_SliderPanel.knobspec()),
        )
        _doc_camera_spec = _DocPanel(
            name="camera_panel",
            label="Camera",
            knobs=list(_PlayerCameraPanel.knobspec()),
        )
        # The doc renderer is a *global default* 2D composer; it is not a
        # shader-node and is not registered.  The unified resolve below
        # invokes it on whatever leftover document items the registered
        # shaders (currently none in the global case) did not finalise.
        print("[doc_renderer] init OK (global 2D default; runs on leftovers)", flush=True)
    except Exception as _exc:
        _doc_init_err = _exc
        print(f"[doc_renderer] init failed: {_exc}", flush=True)
        _doc_rdr = None

    # ── Build the four-way global default dispatcher ─────────────────────────
    # The dispatcher owns the four globals (2D-C, 2D-GL, 3D-C, 3D-GL) and
    # per-channel cadence/min-period gates.  It is selected each frame by
    # R._mode_2d / R._mode_3d (independent), and only fires the leftovers
    # each channel's registered shaders did not finalise.
    try:
        from globals_renderer import GlobalChannelDispatcher as _GlobalChannelDispatcher
        # 3D-C geometry packer: pull world-space triangle soups out of the
        # leftover scene.geometry/<owner> payloads, transform to view
        # space using the active camera, compute per-tri face normals,
        # and pack into the (verts_view, mat_ids, proj_colmajor, light_v)
        # tuple the BaseRasterizer pybind binding expects.
        def _pack_3d_c_geometry(_leftovers):
            try:
                if not _leftovers or _cam_pure_matrices is None:
                    return None
                tri_chunks = []
                for _payload in _leftovers.values():
                    if not isinstance(_payload, dict):
                        continue
                    if _payload.get('kind') != 'triangles':
                        continue
                    _tris = _payload.get('triangles')
                    if _tris is None:
                        continue
                    _tris = np.asarray(_tris, dtype=np.float32)
                    if _tris.ndim != 3 or _tris.shape[1:] != (3, 3) or _tris.shape[0] == 0:
                        continue
                    tri_chunks.append(_tris)
                if not tri_chunks:
                    return None
                tris_world = np.concatenate(tri_chunks, axis=0)        # (Nt, 3, 3)
                Nt = int(tris_world.shape[0])

                # Build P (projection only) and V from the active camera.
                _P64, _V64 = _cam_pure_matrices(R.cam)
                _V = np.asarray(_V64, dtype=np.float32)
                _P = np.asarray(_P64, dtype=np.float32)

                # World -> view transform on every triangle vertex.
                pts_w = tris_world.reshape(-1, 3)                       # (Nt*3, 3)
                pts_h = np.concatenate(
                    [pts_w, np.ones((pts_w.shape[0], 1), dtype=np.float32)],
                    axis=1,
                )
                pts_v = (_V @ pts_h.T).T[:, :3].astype(np.float32)      # (Nt*3, 3)

                # Per-triangle face normal in view space.
                tris_v = pts_v.reshape(Nt, 3, 3)
                e1 = tris_v[:, 1, :] - tris_v[:, 0, :]
                e2 = tris_v[:, 2, :] - tris_v[:, 0, :]
                fn = np.cross(e1, e2).astype(np.float32)
                fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
                fn_len = np.where(fn_len > 1e-8, fn_len, 1.0)
                fn = fn / fn_len
                # Broadcast face normal to all 3 vertices.
                nrm_v = np.repeat(fn, 3, axis=0).astype(np.float32)     # (Nt*3, 3)

                verts_view = np.concatenate([pts_v, nrm_v], axis=1)     # (Nt*3, 6)
                verts_view = np.ascontiguousarray(verts_view, dtype=np.float32)
                mat_ids = np.zeros((Nt,), dtype=np.int32)

                # br_render parses mvp as a column-major 4x4: proj(r,c) = mvp[c*4+r]
                proj = np.ascontiguousarray(_P.T.reshape(-1), dtype=np.float32)

                # Light direction in view space (matches the GL pass).
                light_v = np.array([0.5, 1.0, 0.6], dtype=np.float32)
                light_v /= max(float(np.linalg.norm(light_v)), 1e-8)

                return (verts_view, mat_ids, proj, light_v)
            except Exception:
                return None

        R._global_dispatcher = _GlobalChannelDispatcher(
            width=WIN_W,
            height=WIN_H,
            gl_doc_renderer=_doc_rdr,
            gl_render_callback=R.render,
            c_doc_backend=getattr(_doc_rdr, "_backend", None),
            geometry_packer=_pack_3d_c_geometry,
            cadence_2d=1,
            cadence_3d=1,
            min_period_2d_s=0.0,
            min_period_3d_s=0.0,
        )
        print("[globals] dispatcher ready (2D mode={}, 3D mode={})".format(
            R._mode_2d.value, R._mode_3d.value), flush=True)
    except Exception as _exc:
        print(f"[globals] dispatcher init failed: {_exc}", flush=True)
        R._global_dispatcher = None

    # ── Camera optics schematic (three-view line diagram) ────────────────────
    _cam_optics_view = _CameraOpticsView()
    _cam_optics_view.init_gl()
    _cam_optics_view.attach(R.cam, _player_cam_panel)

    # ── Attach default digital positive sensor to player camera ──────────────
    # 35mm full-frame digital sensor, Bayer RGGB CFA, auto white balance.
    # Backs registered at startup; cycle with _dps.next_back() / prev_back().
    try:
        from camera_software import DigitalPositiveSensor
        _primary_back = FILM_STACKS.get('em_rgb') or _default_stack()
        _dps = DigitalPositiveSensor(
            backs=[_primary_back],
            wb_mode='auto',           # tracks ctx.scene_color_temp_K when set
            color_temp_K=6504.0,      # starting estimate: CIE D65
            sensor_spec=R.cam.sensor, # 35mm full-frame physical spec
        )
        # Register additional preset backs for cycling (silently skip if absent)
        for _back_key in ('full_spectrum', 'acoustic_ortho', 'tri_spectrum'):
            _b = FILM_STACKS.get(_back_key)
            if _b is not None:
                _dps.add_back(_b)
        R.cam.software.append(_dps)
        # Pre-compute initial white-balance weights immediately
        _dps._run_encoding()
        R.cam._digital_rgb = _dps._ds.rgb_weights
    except Exception:
        pass  # package unavailable; renderer keeps its (1,1,1) default

    duty_stations: list = []

    # ── Room station ──────────────────────────────────────────────────────────
    if _HAS_ROOM_STATION:
        try:
            _room_ws = _build_default_scene()
            if _room_ws is None:
                _room_ws = _RoomWorkspace.from_yaml("configs/room_station")
            _room_station = _RoomStation(_room_ws, WIN_W, WIN_H)
            _room_station.init_gl()
            print("[room_station] initialised", flush=True)
            # Build all placed duty-station objects (fabricator, simulator, …)
            # from the workspace registry and add them to the duty_stations list
            # that player_ctrl.tick() and the render loop already consume.
            _scene_stations = _room_ws.build_scene_objects(WIN_W, WIN_H)
            duty_stations.extend(_scene_stations)
            print(f"[room_station] {len(_scene_stations)} scene station(s) built",
                  flush=True)
        except Exception as _e:
            print(f"[room_station] init failed: {_e}", flush=True)
            _room_station = None

    # Spawn player on the room control floor panel, facing the station.
    # station_pos() / station_yaw_deg() come from configs/room_station/station.yaml.
    if player_ctrl is not None and _room_ws is not None:
        try:
            player_ctrl._walk_pos = np.array(_room_ws.station_pos(), np.float64)
            player_ctrl._walk_yaw = float(_room_ws.station_yaw_deg())
        except Exception as _spawn_err:
            print(f"[player_ctrl] spawn from station_pos failed: {_spawn_err}", flush=True)

    # ── Camera items (physical cameras placed in the scene) ───────────────────
    cameras: list = []
    if _HAS_CAMERA_ITEM and _room_ws is not None:
        try:
            cameras = _build_camera_items(_room_ws)
            print(f"[camera_items] {len(cameras)} camera(s) initialised", flush=True)
        except Exception as _e:
            print(f"[camera_items] init failed: {_e}", flush=True)

    # ── Camera designer station ──────────────────────────────────────────────
    _cam_designer_station = None
    if _HAS_CAMERA_DESIGNER_STATION and getattr(args, "station", None) == "camera_designer":
        try:
            _cam_designer_station = _CameraDesignerStation(win_w=WIN_W, win_h=WIN_H)
            _cam_designer_station.init_gl()
            print("[camera_designer_station] ready", flush=True)
        except Exception as _e:
            print(f"[camera_designer_station] init failed: {_e}", flush=True)
            _cam_designer_station = None
    # ─────────────────────────────────────────────────────────────────────────

    if physics is not None:
        print(f"Physics worker is streaming frames ({physics.mode}); "
              "UI will keep cached frames while rebuilds run.", flush=True)
    else:
        print("Ray camera mode: static scene ray tracer only; physics streaming disabled.",
              flush=True)
    if args.excitation == "rest":
        print("Diagnostic rest mode: no plucks are scheduled; reporting equilibrium drift.",
              flush=True)

    clock       = pygame.time.Clock()
    running     = True
    _exit_confirm_open = False
    fi          = 0
    recorded_samples = 0
    replaying   = False
    dragging    = False
    last_mouse  = (0, 0)
    refresh_every = max(0, int(args.gpu_refresh_every))
    refresh_rays = int(args.gpu_refresh_rays or args.gpu_field_rays)
    sensor_light_rays = int(args.sensor_light_rays)
    force_ray_refresh = False
    pending_rebuild = False
    active_ray_frame_index = None
    last_displayed_frame_index = None
    equilibrium_report_every = max(0, int(args.equilibrium_report_every))

    def _open_exit_confirm(source: str = "") -> None:
        nonlocal _exit_confirm_open
        if _exit_confirm_open:
            return
        _exit_confirm_open = True
        _src = f" ({source})" if source else ""
        print(f"[exit] confirm quit{_src}: Y/Enter=yes, N/Esc=no", flush=True)

    def _exit_confirm_layout(win_w: int, win_h: int) -> dict:
        panel_w = 520
        panel_h = 220
        panel_x = (win_w - panel_w) // 2
        panel_y = (win_h - panel_h) // 2
        btn_w = 170
        btn_h = 46
        gap = 20
        row_y = panel_y + panel_h - btn_h - 26
        no_x = panel_x + (panel_w - (btn_w * 2 + gap)) // 2
        yes_x = no_x + btn_w + gap
        return {
            "panel": (panel_x, panel_y, panel_w, panel_h),
            "no": (no_x, row_y, btn_w, btn_h),
            "yes": (yes_x, row_y, btn_w, btn_h),
        }

    def _pt_in_rect(pt: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
        px, py = pt
        rx, ry, rw, rh = rect
        return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)

    def _refresh_ray_field_from_frame(frame, frame_index: int, _reason: str) -> bool:
        nonlocal active_ray_frame_index, force_ray_refresh
        if scene is None or not args.gpu_rays:
            return False
        acc = R._sensor_acc
        if acc is None or frame is None:
            return False
        if not force_ray_refresh and refresh_every > 0 and frame_index >= 0 and (frame_index % refresh_every) != 0:
            return False
        acc.notify_frame(
            frame=frame,
            info=info,
            outline=outline,
            body_h=body_h,
            paths=paths,
            active_strings=active_strings,
            refresh_rays=refresh_rays,
            sensor_light_rays=sensor_light_rays,
            stage_light_emitters=int(args.stage_light_emitters),
        )
        active_ray_frame_index = frame_index
        force_ray_refresh = False
        return True

    try:
      from controls import (
          start_action_dispatcher as _start_action_dispatcher,
          stop_action_dispatcher as _stop_action_dispatcher,
          get_shader_walker as _get_shader_walker,
          get_control_graph as _get_control_graph,
      )
      _start_action_dispatcher()
      _shader_graph = _get_control_graph()
      _shader_walker = _get_shader_walker()
      _shader_frame_result = None

      def _submit_scene_object_buffers() -> None:
          # Submit routine scene geometry through the walker's universal
          # owner flip-buffer API (no shader registration required).
          # Submission is dirty-aware via change_key, so unchanged objects
          # keep prior geometry in the end-state buffer without re-flipping.
          for _i, _ds in enumerate(duty_stations):
              if not hasattr(_ds, 'interaction_triangles_world'):
                  continue
              try:
                  _tris = _ds.interaction_triangles_world()
              except Exception:
                  continue
              _pref = getattr(_ds, '_placed_ref', None)
              _owner = str(getattr(_pref, 'obj_id', f'duty_station_{_i}'))
              _wp = np.asarray(getattr(_ds, 'world_position', np.zeros(3, np.float64)), np.float64).reshape(3)
              _yaw = float(getattr(_ds, '_yaw_deg', 0.0))
              _sig = (
                  float(_wp[0]), float(_wp[1]), float(_wp[2]),
                  _yaw,
                  int(_tris.shape[0]) if hasattr(_tris, 'shape') else 0,
              )
              _shader_walker.publish_owner_target(
                  owner_id=_owner,
                  target_id=f"scene.geometry/{_owner}",
                  payload={
                      "owner_id": _owner,
                      "kind": "triangles",
                      "triangles": _tris,
                  },
                  flip_slots=2,
                  change_key=_sig,
              )

          for _i, _ci in enumerate(cameras):
              if not hasattr(_ci, 'interaction_triangles_world'):
                  continue
              try:
                  _tris = _ci.interaction_triangles_world()
              except Exception:
                  continue
              _placed = getattr(_ci, 'placed', None)
              _owner = str(getattr(_placed, 'obj_id', f'camera_{_i}'))
              _pos = np.asarray(getattr(_placed, 'pos', np.zeros(3, np.float64)), np.float64).reshape(3)
              _sig = (
                  float(_pos[0]), float(_pos[1]), float(_pos[2]),
                  float(getattr(_placed, 'yaw_deg', 0.0)),
                  float(getattr(_placed, 'pan_deg', 0.0)),
                  float(getattr(_placed, 'tilt_deg', 0.0)),
                  float(getattr(_placed, 'focal_mm', 0.0)),
                  str(getattr(_placed, 'mesh_id', 'camera_35mm')),
                  int(_tris.shape[0]) if hasattr(_tris, 'shape') else 0,
              )
              _shader_walker.publish_owner_target(
                  owner_id=_owner,
                  target_id=f"scene.geometry/{_owner}",
                  payload={
                      "owner_id": _owner,
                      "kind": "triangles",
                      "triangles": _tris,
                  },
                  flip_slots=2,
                  change_key=_sig,
              )

      _warned_no_c_present_target = False

      # ---- Naive star network bootstrap --------------------------------
      # Auto-wire every existing owner into a default star, spin up the
      # singleton gateway, and pre-create one duty station the player can
      # find in the world.  The controller is ticked once per frame below.
      try:
          from naive_graph import (
              auto_wire_naive_network as _auto_wire_naive_network,
              get_naive_graph_controller as _get_naive_graph_controller,
              NaiveStarDutyStation as _NaiveStarDutyStation,
          )
          from gateway_object import get_gateway as _get_gateway
          _naive_ctrl = _get_naive_graph_controller()
          _auto_wire_naive_network("global", priority=0)
          _player_duty_station = _NaiveStarDutyStation(
              "global", owner_id="_duty_station.global",
              priority=0, breakout_count=8,
          )
          _gateway = _get_gateway(world_pos=(0.0, 0.0, -5.0))
          print(f"[naive] bootstrap OK: {len(_naive_ctrl.stars)} star(s), "
                f"DS={_player_duty_station.owner_id}, gateway={_gateway.owner_id}",
                flush=True)
      except Exception as _naive_exc:
          import traceback as _tb
          print(f"[naive] bootstrap failed: {_naive_exc}", flush=True)
          _tb.print_exc()
          _naive_ctrl = None
          _player_duty_station = None
          _gateway = None
      while running:
        # Tick the naive star network once per frame.
        if _naive_ctrl is not None:
            try:
                _naive_ctrl.tick()
                if _gateway is not None:
                    _gateway.pump()
            except Exception as _ng_exc:
                # Never let routing kill the render loop.
                print(f"[naive] tick error: {_ng_exc}", flush=True)
        _dt = clock.tick(60) / 1000.0
        _keys_held = pygame.key.get_pressed()
        # Camera designer station owns the mouse — always keep it free.
        if _cam_designer_station is not None:
            if not pygame.mouse.get_visible():
                pygame.mouse.set_visible(True)
            if pygame.event.get_grab():
                pygame.event.set_grab(False)
        if player_ctrl is not None and not _player_cam_panel.freeze_player_motion:
            player_ctrl.tick(_dt, _keys_held, duty_stations, cameras=cameras)
        # Drive each station's menu visibility from player state — no blocking calls
        if player_ctrl is not None:
            _active_st = getattr(player_ctrl, '_active_station', None)
            _in_interact = player_ctrl.state.value == "interact"
            for _ds in duty_stations:
                _m = getattr(_ds, 'menu', None)
                if _m is not None and hasattr(_m, 'show_hud'):
                    _m.show_hud(_in_interact and _active_st is _ds)

        # Close player cam panel if player has left walk/in_camera states
        if (_player_cam_panel.open
                and player_ctrl is not None
                and player_ctrl.state.value not in ("walk", "in_camera")):
            _player_cam_panel.close()

        for ev in pygame.event.get():
            if _exit_confirm_open:
                if ev.type == KEYDOWN:
                    if ev.key in (pygame.K_y, pygame.K_RETURN, pygame.K_KP_ENTER):
                        running = False
                        continue
                    if ev.key in (pygame.K_n, pygame.K_ESCAPE):
                        _exit_confirm_open = False
                        print("[exit] cancelled", flush=True)
                        continue
                elif ev.type == MOUSEBUTTONDOWN and ev.button == 1:
                    _layout = _exit_confirm_layout(WIN_W, WIN_H)
                    if _pt_in_rect(ev.pos, _layout["yes"]):
                        running = False
                        continue
                    if _pt_in_rect(ev.pos, _layout["no"]):
                        _exit_confirm_open = False
                        print("[exit] cancelled", flush=True)
                        continue
                elif ev.type == QUIT:
                    continue
                continue
            if ev.type == KEYDOWN and ev.key == pygame.K_ESCAPE:
                _open_exit_confirm("Esc")
                continue
            # Camera designer station owns the full screen — it gets events
            # before everything else so its mouse-interactive ortho views are
            # never shadowed by the orbit camera handler.
            if _cam_designer_station is not None:
                if _cam_designer_station.handle_event(ev):
                    continue
                # Designer is active: skip player_ctrl entirely (orbit/walk
                # mouse handling must not fire while the designer is open).
                if ev.type == QUIT:
                    _open_exit_confirm("window close")
                continue
            # Camera panel consumes mouse events when in IN_CAMERA mode
            if (_camera_panel is not None
                    and player_ctrl is not None
                    and player_ctrl.state.value == "in_camera"):
                if _camera_panel.handle_event(ev, R.cam):
                    continue
            if _player_cam_panel.open and _cam_optics_view.handle_event(ev, WIN_W, WIN_H):
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
                pygame.mouse.get_rel()
                continue
            # Player camera settings panel (walk mode, non-blocking)
            if _player_cam_panel.handle_event(ev):
                continue
            # Route through player controller first; it may absorb movement events
            if player_ctrl is not None:
                if not _player_cam_panel.open:
                    if player_ctrl.handle_event(ev, duty_stations, cameras=cameras):
                        continue
            # Slider panel only visible / interactive when not in a walk state
            # (or always when player_ctrl is None / orbit mode)
            _panel_active = (player_ctrl is None or player_ctrl.sidebar_visible
                             or player_ctrl.state.value == "orbit")
            _active_st = getattr(player_ctrl, '_active_station', None) if player_ctrl is not None else None
            if _active_st is not None and _active_st.handle_menu_event(ev):
                continue
            if _panel_active and panel.handle_event(ev):
                continue
            if ev.type == QUIT:
                _open_exit_confirm("window close")
            elif ev.type == KEYDOWN:
                camera_dirty = False
                if ev.key == pygame.K_ESCAPE:
                    _open_exit_confirm("Esc")
                elif ev.key == K_SPACE:
                    R._paused = not R._paused
                elif ev.key == pygame.K_f:
                    R.advance_frame(1)
                    R.advance_sensor(1)
                    replaying = True if R._frames else replaying
                elif ev.key == pygame.K_c:
                    R.reset_sensor()
                elif ev.key == pygame.K_w:
                    R.cam.pan(forward_m=0.04); R.cam._auto = 0.0; camera_dirty = True
                elif ev.key == pygame.K_s:
                    R.cam.pan(forward_m=-0.04); R.cam._auto = 0.0; camera_dirty = True
                elif ev.key == pygame.K_a:
                    R.cam.pan(right_m=-0.04); R.cam._auto = 0.0; camera_dirty = True
                elif ev.key == pygame.K_d:
                    R.cam.pan(right_m=0.04); R.cam._auto = 0.0; camera_dirty = True
                elif ev.key == pygame.K_q:
                    R.cam.pan(up_m=0.04); R.cam._auto = 0.0; camera_dirty = True
                elif ev.key == pygame.K_z:
                    R.cam.pan(up_m=-0.04); R.cam._auto = 0.0; camera_dirty = True
                elif ev.key == pygame.K_LEFT:
                    R.cam.tilt_shift[0] = float(np.clip(R.cam.tilt_shift[0] - 0.03, -1.0, 1.0)); camera_dirty = True
                elif ev.key == pygame.K_RIGHT:
                    R.cam.tilt_shift[0] = float(np.clip(R.cam.tilt_shift[0] + 0.03, -1.0, 1.0)); camera_dirty = True
                elif ev.key == pygame.K_UP:
                    R.cam.tilt_shift[1] = float(np.clip(R.cam.tilt_shift[1] + 0.03, -1.0, 1.0)); camera_dirty = True
                elif ev.key == pygame.K_DOWN:
                    R.cam.tilt_shift[1] = float(np.clip(R.cam.tilt_shift[1] - 0.03, -1.0, 1.0)); camera_dirty = True
                elif ev.key in (pygame.K_EQUALS, getattr(pygame, "K_PLUS", pygame.K_EQUALS), pygame.K_KP_PLUS):
                    R.cam.set_lens(focal_delta=5.0); camera_dirty = True
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    R.cam.set_lens(focal_delta=-5.0); camera_dirty = True
                elif ev.key == pygame.K_LEFTBRACKET:
                    R.cam.set_lens(focus_delta=-0.10); camera_dirty = True
                elif ev.key == pygame.K_RIGHTBRACKET:
                    R.cam.set_lens(focus_delta=0.10); camera_dirty = True
                elif ev.key == pygame.K_COMMA:
                    R.cam.aperture = 0.0 if R.cam.aperture <= 0.001 else float(np.clip(R.cam.aperture / 1.25, 0.0, 0.08))
                    panel.values['lens_aperture'] = R.cam.aperture
                    camera_dirty = True
                elif ev.key == pygame.K_PERIOD:
                    R.cam.aperture = float(np.clip(max(0.002, R.cam.aperture * 1.25), 0.0, 0.08))
                    panel.values['lens_aperture'] = R.cam.aperture
                    camera_dirty = True
                elif ev.key == K_r:
                    if physics is None:
                        R.reset_sensor()
                        force_ray_refresh = refresh_every > 0
                    else:
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
                elif ev.key == pygame.K_n:
                    R._film_negative = not R._film_negative
                    print(f"Film: {'negative' if R._film_negative else 'positive'}")
                elif ev.key == pygame.K_e:
                    # E cycles HUD modes: full -> status+hotbar -> off -> full.
                    _player_cam_panel.attach(R.cam, panel, R, player_ctrl=player_ctrl)
                    _cam_optics_view.attach(R.cam, _player_cam_panel)
                    _player_cam_panel.cycle_hud_mode()
                    if _player_cam_panel.open:
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                        pygame.mouse.get_rel()
                    else:
                        _wants_grab = (
                            (player_ctrl is not None
                             and player_ctrl.state.value == "walk")
                            or _player_cam_panel._cam_render_mode is RenderMode.RAYTRACE
                        )
                        if _wants_grab:
                            pygame.mouse.set_visible(False)
                            pygame.event.set_grab(True)
                            pygame.mouse.get_rel()
                else:
                    for ki, kv in enumerate(LAYER_KEYS):
                        if ev.key == kv:
                            R.toggle(ki)
                            s = ['OPAQUE','ALPHA','HIDDEN'][R._layers[ki]]
                            print(f"Layer {ki+1} ({LAYER_NAMES[ki]}): {s}")
                if camera_dirty:
                    R.sync_sensor_camera(reset=False)
            elif ev.type == MOUSEBUTTONDOWN and ev.button == 1:
                dragging = True; last_mouse = ev.pos
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == MOUSEMOTION:
                if _player_cam_panel.open and _player_cam_panel.freeze_player_motion:
                    _player_cam_panel.update_hover_pick(ev.pos, WIN_W, WIN_H, duty_stations, cameras)
                    continue
                _in_ray_fps = (
                    _player_cam_panel._cam_render_mode is RenderMode.RAYTRACE
                    and not _player_cam_panel.open
                    and pygame.event.get_grab()
                    and (player_ctrl is None or player_ctrl.state.value != "walk")
                )
                if _in_ray_fps:
                    dx, dy = ev.rel
                    R.cam.orbit(dx * 0.35, -dy * 0.25)
                    R.cam._auto = 0.0
                    R.sync_sensor_camera(reset=False)
                elif dragging:
                    dx, dy = ev.pos[0]-last_mouse[0], ev.pos[1]-last_mouse[1]
                    R.cam.orbit(dx*0.35, -dy*0.25); R.cam._auto = 0.0
                    R.sync_sensor_camera(reset=False)
                    last_mouse = ev.pos
            elif ev.type == MOUSEWHEEL:
                if _player_cam_panel.open:
                    continue
                R.cam.zoom(-ev.y * 0.04)
                R.sync_sensor_camera(reset=False)

        for msg in (physics.poll() if physics is not None else []):
            mtype = msg.get("type")
            if mtype == "error":
                if build_bar is not None and not physics_ready:
                    build_bar.close()
                raise RuntimeError(_worker_error_message(msg))
            if mtype == "progress" and not physics_ready:
                frac = max(0.0, min(1.0, float(msg.get("frac", 0.0))))
                label = str(msg.get("label", ""))
                if build_bar is not None:
                    target = int(round(frac * 1000.0))
                    if target > build_bar_value:
                        build_bar.update(target - build_bar_value)
                        build_bar_value = target
                    build_bar.set_postfix_str(label[-80:])
                else:
                    print(f"[physics] {frac*100:.1f}%  {label}", flush=True)
                continue
            if mtype == "profile":
                text = msg.get("text", "")
                if text:
                    sys.stderr.write(text)
                    sys.stderr.flush()
                continue
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
                if not physics_ready:
                    physics_ready = True
                    if build_bar is not None:
                        if build_bar_value < 1000:
                            build_bar.update(1000 - build_bar_value)
                        build_bar.close()
                        build_bar = None
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
            if scene is not None and args.gpu_rays and R._sensor_acc is not None:
                # Non-blocking: posts to source worker; pump_forward() drains it.
                _refresh_ray_field_from_frame(frame, fi, "stream")
            fi += 1

        # ── Apply live slider changes ──────────────────────────────────────────
        changed = panel.changed()
        if 'ray_density' in changed:
            _density = float(panel.values['ray_density'])
            refresh_rays = max(1, int(round(float(args.sound_energy_rays) * _density)))
            sensor_light_rays = max(1, int(round(float(args.em_energy_rays) * _density)))
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
        if 'sensor_iso' in changed:
            R.film.active_layer.iso = float(panel.values['sensor_iso'])
        if 'sensor_rate' in changed and R._sensor_acc is not None:
            R._sensor_acc._rows_per_frame = max(1, int(round(panel.values['sensor_rate'])))
        if 'sensor_spp' in changed and R._sensor_acc is not None:
            R._sensor_acc._samples_per_pixel = max(1, int(round(panel.values['sensor_spp'])))
        if 'sensor_fps' in changed:
            R._sensor_fps = max(0.0, float(panel.values['sensor_fps']))
        if 'frame_step' in changed:
            R._frame_step = max(0, int(round(panel.values['frame_step'])))
        if 'lens_aperture' in changed:
            R.cam.aperture = float(np.clip(panel.values['lens_aperture'], 0.0, 0.08))
            R.sync_sensor_camera(reset=False)
        if 'lens_ca' in changed:
            R.cam.ca = float(np.clip(panel.values['lens_ca'], 0.0, 0.02))
            R.sync_sensor_camera(reset=False)
        if 'film_decay' in changed and R._sensor_acc is not None:
            # Update the accumulator's half-life; takes effect on next apply_decay().
            R._sensor_acc.half_life = max(0.0, float(panel.values['film_decay']))
        if 'air_diff' in changed and R._sensor_acc is not None:
            R._sensor_acc._air_ds = float(panel.values['air_diff'])
        if 'air_spec' in changed and R._sensor_acc is not None:
            R._sensor_acc._air_ss = float(panel.values['air_spec'])
        if 'air_aniso' in changed and R._sensor_acc is not None:
            R._sensor_acc._air_an = float(panel.values['air_aniso'])

        try:
            # Per-frame framebuffer clear. R.render() (GL 3D path) used
            # to do this; with 3D=C it's never called, so do it here
            # unconditionally so leftover GL state can't bleed through.
            glClearColor(0.015, 0.010, 0.040, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glViewport(0, 0, WIN_W, WIN_H)
            _ray_diag_update(
                "render:frame",
                frame_index=int(fi),
                ray_textures=tuple(int(t) for t in (R._ray_field_bands or [])),
                gpu_seg_vbo=int(R._gpu_seg_vbo or 0),
                gpu_seg_cap=int(R._gpu_seg_cap or 0),
                ray_draw_vertices=int(getattr(R, "_ray_n", 0)),
                cached_frames=int(len(R._frames)),
            )
            _cam_dt = clock.get_time() * 1e-3  # ms → s
            if not _player_cam_panel.open:
                R.cam.tick(_cam_dt)
            _player_cam_panel.enforce_menu_camera_lock()

            # ── HUD second channel — submit all panels to doc renderer ────────
            # Each HUD object is a channel in the same pipeline; the walker's
            # "doc_composite" node composites them into the "doc.layer" target
            # which the global final-pass resolves alongside the world render.
            if _doc_rdr is None:
                raise RuntimeError(
                    f"HUD second channel unavailable: DocRenderer failed to "
                    f"initialise (init error: {_doc_init_err!r}). "
                    f"Check _spectral_kernels build — run: "
                    f"cmake --build csrc_build --config Release"
                )
            # Slider panel — always visible
            _doc_rdr.submit_panel(
                _doc_slider_spec,
                (10, 10, 264, 30 + 42 * len(_doc_slider_spec.knobs)),
                node_id_map=_doc_slider_ids,
                knob_values=panel.values,
            )
            # Camera panel — only when open
            if _player_cam_panel.open:
                _doc_rdr.submit_panel(
                    _doc_camera_spec,
                    (WIN_W - 340, 10, 330, 30 + 42 * len(_doc_camera_spec.knobs)),
                    node_id_map=_doc_camera_ids,
                    knob_values={k: getattr(R.cam, k, None) for k in
                                 [s[0] for s in _PlayerCameraPanel._SLIDERS]},
                )
            # Active station — must expose submit_doc_channel; no GL fallback.
            _active_st = getattr(player_ctrl, '_active_station', None) if player_ctrl is not None else None
            if _active_st is not None:
                if not hasattr(_active_st, 'submit_doc_channel'):
                    raise RuntimeError(
                        f"Station {type(_active_st).__name__!r} has no submit_doc_channel(). "
                        "Update the station class to expose a panel_spec on its menu "
                        "and remove any direct render_hud/render_menu GL calls."
                    )
                _active_st.submit_doc_channel(_doc_rdr, WIN_W, WIN_H)

            _submit_scene_object_buffers()

            # ── Bottom-up shader walk ────────────────────────────────────────
            # Visit every SHADERS-axis node in the control hierarchy in
            # post-order. Each due shader runs on the main thread (so it
            # may touch the GL context), publishes its payload into its
            # flip buffer, and stamps its claimed targets as finalized so
            # the final draw below can gate fragments accordingly.
            try:
                _shader_frame_result = _shader_walker.tick(
                    frame_index=int(fi),
                    dt=float(_dt),
                )
            except Exception:
                _shader_frame_result = None

            # Final arena resolve: only run global cleanup for unresolved
            # target ranges after specific/local shader writes are accounted for.
            _latest_targets = _shader_graph.snapshot_latest_targets()
            _finalized = (
                set(_shader_frame_result.finalized_targets)
                if _shader_frame_result is not None else set()
            )
            _leftover_targets = {
                _tid: _payload
                for _tid, _payload in _latest_targets.items()
                if _tid not in _finalized
            }
            # ── Unified two-channel resolve ───────────────────────────────────
            #
            # Channel 3D (geometry):
            #   Owner flip targets under "scene.geometry/<owner>" are the
            #   contiguous geometry stream produced by the dirty walk.
            #   Anything that wants to contribute meshes publishes there;
            #   the final 3D pass consumes the merged set with a mask of
            #   regions already covered by earlier shader writes.
            #
            # Channel 2D (textures):
            #   Owner flip targets under "doc.layer/<owner>" plus the
            #   document composite are the hierarchically sorted texture
            #   set. Pre-run textures (from registered shaders) have already
            #   been written; the 2D composer fills in any leftover
            #   document-hierarchy items as textures.
            #
            # Both channels are produced every frame regardless of which
            # backend (C or OpenGL) executes the final stages.
            _geom_channel: dict = {}
            _tex_channel: dict = {}
            for _tid, _payload in _latest_targets.items():
                _key = str(_tid)
                if _key.startswith("scene.geometry/"):
                    _geom_channel[_key] = _payload
                elif _key.startswith("doc.layer/") or _key.startswith("doc.composite"):
                    _tex_channel[_key] = _payload

            setattr(R, "_geometry_channel", _geom_channel)
            setattr(R, "_texture_channel", _tex_channel)
            setattr(R, "_final_targets", _latest_targets)
            setattr(R, "_final_leftovers", _leftover_targets)
            setattr(R, "_finalized_targets", _finalized)

            # ── Global default fillers (run on leftovers only) ────────────
            #
            # Globals are NOT registered shader-nodes.  After the walker
            # has fired every registered shader (currently none in the
            # global case, so everything is leftover), the four-way
            # dispatcher selects one global per channel based on
            # R._mode_2d / R._mode_3d (each independently C or GL) and
            # fires it on the leftover items not covered by the
            # finalised mask.  The 2D and 3D channels obey independent
            # cadence and min-period gates and are not lock-step.
            _tex_leftovers = {
                _tid: _payload for _tid, _payload in _tex_channel.items()
                if _tid not in _finalized
            }
            _geom_leftovers = {
                _tid: _payload for _tid, _payload in _geom_channel.items()
                if _tid not in _finalized
            }
            setattr(R, "_geometry_leftovers", _geom_leftovers)
            setattr(R, "_texture_leftovers", _tex_leftovers)

            _global_result = None
            if getattr(R, "_global_dispatcher", None) is not None:
                try:
                    from globals_renderer import ChannelBackend as _ChannelBackend
                    _global_result = R._global_dispatcher.dispatch(
                        leftovers_2d=_tex_leftovers,
                        leftovers_3d=_geom_leftovers,
                        mode_2d=_ChannelBackend.from_render_mode(R._mode_2d),
                        mode_3d=_ChannelBackend.from_render_mode(R._mode_3d),
                        frame_index=int(fi),
                        dt=float(_dt),
                    )
                    _diag_n = getattr(R, "_diag_render_count", 0)
                    if _diag_n < 5:
                        _o3 = getattr(_global_result, "out_3d_rgba", None)
                        _o2 = getattr(_global_result, "out_2d_rgba", None)
                        _u3 = getattr(_global_result, "used_3d", None)
                        _u2 = getattr(_global_result, "used_2d", None)
                        _s3 = getattr(_global_result, "skipped_3d_reason", "")
                        _s2 = getattr(_global_result, "skipped_2d_reason", "")
                        # Quantify 3D-C output: lit-pixel count & alpha range
                        _o3_stats = "-"
                        if _o3 is not None:
                            try:
                                _arr = np.asarray(_o3)
                                _lit = int((_arr[..., 3] > 0).sum())
                                _amx = int(_arr[..., 3].max())
                                _rmx = int(_arr[..., :3].max())
                                _o3_stats = f"lit={_lit} amax={_amx} rgbmax={_rmx} shape={_arr.shape}"
                            except Exception as _e:
                                _o3_stats = f"stats-err:{_e}"
                        # Show first geom payload for sanity
                        _gsamp = "-"
                        try:
                            _kk = next(iter(_geom_leftovers))
                            _pp = _geom_leftovers[_kk]
                            _tt = _pp.get("triangles")
                            _gsamp = f"{_kk}: tris.shape={getattr(_tt,'shape',None)}"
                        except Exception:
                            pass
                        print(
                            f"[diag #{_diag_n} fi={fi}] geom_left={len(_geom_leftovers)} "
                            f"tex_left={len(_tex_leftovers)} fin={len(_finalized)} "
                            f"u2d={_u2}({_s2!r}) u3d={_u3}({_s3!r}) "
                            f"o3=[{_o3_stats}] sample={_gsamp}",
                            flush=True,
                        )
                        setattr(R, "_diag_render_count", _diag_n + 1)
                except Exception as _exc:
                    _report_exception("global dispatcher", _exc)

            setattr(R, "_global_dispatch_result", _global_result)

            # ── Blit policy ────────────────────────────────────────────────
            # The C globals produce CPU RGBA buffers.  Deposit them into
            # the active GL framebuffer (which the OPENGL pygame surface
            # is presenting) through the doc renderer's fullscreen-quad
            # blit pipeline.  Order: 3D-C first (it forms the background),
            # then 2D-C on top.  When a channel ran on its GL backend
            # there is nothing to blit here — its output is already in
            # the framebuffer (3D-GL) or was deposited inline (2D-GL).
            if _global_result is not None and _doc_rdr is not None:
                try:
                    _out_3d = getattr(_global_result, "out_3d_rgba", None)
                    if _out_3d is not None:
                        _doc_rdr.blit_rgba(_out_3d, alpha=1.0)
                except Exception as _exc:
                    _report_exception("blit 3D-C", _exc)
                try:
                    _out_2d = getattr(_global_result, "out_2d_rgba", None)
                    if _out_2d is not None:
                        _doc_rdr.blit_rgba(_out_2d, alpha=1.0)
                except Exception as _exc:
                    _report_exception("blit 2D-C", _exc)

            # Camera-item draw is part of the 3D channel: cameras publish
            # their geometry through publish_owner_target; their direct
            # GL draw remains here only as a transitional convenience until
            # the base material renderer consumes the geometry channel.
            _rs_lv = np.array([0.5, 1.0, 0.6], np.float32)
            _rs_lv /= np.linalg.norm(_rs_lv)
            if cameras and _cam_pure_matrices is not None:
                _P, _V = _cam_pure_matrices(R.cam)
                _MVP = (_P @ _V).astype(np.float32)
                _MV  = _V.astype(np.float32)
                _lv  = _rs_lv
                for _ci in cameras:
                    _ci.draw(_MVP, _MV, _lv)

            pygame.display.flip()
        except BaseException as exc:
            _report_exception(f"render frame {fi}", exc)
            raise
    finally:
        try:
            _stop_action_dispatcher()
        except Exception:
            pass
        if capture is not None:
            capture.close()
            print(f"Wrote capture: {', '.join(capture.paths)}", flush=True)
        if physics is not None:
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
