"""camera_panel.py
==================
CameraHudPanel — full in-screen HUD for editing every camera parameter while
in IN_CAMERA mode.

All sensor, lens, and exposure fields are editable through draggable sliders.
No preset is enforced: every value can be set to any physically plausible
range without reloading a YAML.  Changes write directly to the GL Camera
object (``camera.sensor``, ``camera.lens``, ``camera.focal_mm`` etc.) so
``fov_y_rad()`` and all ray-tracing parameters update immediately.

Layout (820 × 540 px, left panel)
──────────────────────────────────
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  CAMERA  [ name ]                                                        │
  ├─────────────────────┬────────────────────────┬───────────────────────────┤
  │  SENSOR             │  LENS                  │  EXPOSURE                 │
  │  width_mm     36.0  │  focal_mm      35.0    │  focus_m        1.60      │
  │  height_mm    24.0  │  min_focal_mm  35.0    │  aperture       0.00      │
  │  pitch_um      8.4  │  max_focal_mm  35.0    │  ca             0.000     │
  │  max_iso    12800   │  min_focus_m    0.45   │  tilt_shift X   0.00      │
  │  dyn_range    14.0  │  max_fstop      1.40   │  tilt_shift Y   0.00      │
  │                     │  distort_k1     0.001  │                           │
  │                     │  distort_k2     0.000  │  ↑ LENS ZOOM RANGE ↑     │
  │                     │  vignetting     0.15   │  (focal_min–max above)    │
  │                     │  tx band 1      0.98   │                           │
  │                     │  tx band 2      0.97   │                           │
  │                     │  tx band 3      0.95   │                           │
  │                     │  tx band 4      0.85   │                           │
  └─────────────────────┴────────────────────────┴───────────────────────────┘

Usage
-----
    panel = CameraHudPanel()
    panel.init_gl()

    # Each frame when in IN_CAMERA mode:
    panel.render(win_w, win_h, camera)   # writes pygame surface → GL texture

    # Route events to panel before other handlers:
    if panel.handle_event(ev, camera):
        pass  # consumed
"""
from __future__ import annotations

import ctypes
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_BLEND, GL_DEPTH_TEST, GL_FALSE, GL_FLOAT,
        GL_FRAGMENT_SHADER, GL_LINEAR, GL_ONE_MINUS_SRC_ALPHA, GL_RGBA,
        GL_SRC_ALPHA, GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TRIANGLES,
        GL_UNSIGNED_BYTE, GL_VERTEX_SHADER,
        glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray,
        glBlendFunc, glBufferData, glDisable, glDrawArrays, glEnable,
        glEnableVertexAttribArray, glGenBuffers, glGenTextures,
        glGenVertexArrays, glGetUniformLocation, glTexImage2D,
        glTexParameteri, glUniform1i, glUniform2f, glUniformMatrix4fv,
        glUseProgram, glVertexAttribPointer,
    )
    from OpenGL.GL import shaders as _gl_shaders
    _HAS_GL = True
except ImportError:
    _HAS_GL = False


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (same feel as room_duty_station)
# ─────────────────────────────────────────────────────────────────────────────

_BG          = ( 12,  16,  24, 215)
_ACCENT      = ( 40, 120, 220, 255)
_TEXT_FG     = (200, 215, 235, 255)
_TEXT_DIM    = (110, 125, 150, 255)
_TEXT_HEAD   = (255, 210,  80, 255)
_RAIL        = ( 38,  50,  72, 255)
_FILL        = ( 55, 140, 230, 255)
_KNOB        = (190, 215, 255, 255)
_SEP         = ( 38,  50,  72, 200)


# ─────────────────────────────────────────────────────────────────────────────
# Slider spec
# ─────────────────────────────────────────────────────────────────────────────

class _SliderSpec:
    """Metadata for one editable parameter row."""
    __slots__ = ("label", "unit", "vmin", "vmax", "fmt", "is_log",
                 "get", "set")

    def __init__(self, label: str, unit: str,
                 vmin: float, vmax: float, fmt: str = ".2f",
                 is_log: bool = False,
                 get=None, set=None):
        self.label  = label
        self.unit   = unit
        self.vmin   = float(vmin)
        self.vmax   = float(vmax)
        self.fmt    = fmt
        self.is_log = is_log
        self.get    = get   # callable(camera) → float
        self.set    = set   # callable(camera, float) → None


def _tx_get(idx: int):
    def _g(cam): return float(cam.lens.transmission[idx])
    return _g


def _tx_set(idx: int):
    def _s(cam, v):
        t  = list(cam.lens.transmission)
        t[idx] = float(np.clip(v, 0.0, 1.0))
        cam.lens.transmission = tuple(t)
    return _s


# ─────────────────────────────────────────────────────────────────────────────
# Column definitions: list of _SliderSpec per section
# ─────────────────────────────────────────────────────────────────────────────

def _make_sensor_sliders() -> List[_SliderSpec]:
    return [
        _SliderSpec("width_mm",   "mm",  4.0,  70.0, ".1f",
                    get=lambda c: c.sensor.width_mm,
                    set=lambda c, v: setattr(c.sensor, "width_mm", v)),
        _SliderSpec("height_mm",  "mm",  2.5,  55.0, ".1f",
                    get=lambda c: c.sensor.height_mm,
                    set=lambda c, v: setattr(c.sensor, "height_mm", v)),
        _SliderSpec("pitch_um",   "µm",  0.5,  25.0, ".2f",
                    get=lambda c: c.sensor.pixel_pitch_um,
                    set=lambda c, v: setattr(c.sensor, "pixel_pitch_um", v)),
        _SliderSpec("max_iso",    "",  100.0, 409600.0, ".0f", is_log=True,
                    get=lambda c: float(c.sensor.max_iso),
                    set=lambda c, v: setattr(c.sensor, "max_iso", int(round(v)))),
        _SliderSpec("dyn_range",  "ev",  4.0,  24.0, ".1f",
                    get=lambda c: c.sensor.dynamic_range_stops,
                    set=lambda c, v: setattr(c.sensor, "dynamic_range_stops", v)),
    ]


def _make_lens_sliders() -> List[_SliderSpec]:
    return [
        _SliderSpec("focal_mm",    "mm",  4.0,  600.0, ".1f",
                    get=lambda c: c.focal_mm,
                    set=lambda c, v: (
                        setattr(c, "focal_mm", v),
                        setattr(c.lens, "focal_mm", v))),
        _SliderSpec("min_focal",   "mm",  4.0,  600.0, ".1f",
                    get=lambda c: c.lens.min_focal_mm,
                    set=lambda c, v: setattr(c.lens, "min_focal_mm", v)),
        _SliderSpec("max_focal",   "mm",  4.0,  600.0, ".1f",
                    get=lambda c: c.lens.max_focal_mm,
                    set=lambda c, v: setattr(c.lens, "max_focal_mm", v)),
        _SliderSpec("min_focus",   "m",   0.05, 20.0, ".2f",
                    get=lambda c: c.lens.min_focus_m,
                    set=lambda c, v: setattr(c.lens, "min_focus_m", v)),
        _SliderSpec("max_fstop",   "f/",  0.7,  32.0, ".1f",
                    get=lambda c: c.lens.max_aperture_fstop,
                    set=lambda c, v: setattr(c.lens, "max_aperture_fstop", v)),
        _SliderSpec("distort_k1",  "",  -0.5,   0.5, ".4f",
                    get=lambda c: c.lens.distortion_k1,
                    set=lambda c, v: setattr(c.lens, "distortion_k1", v)),
        _SliderSpec("distort_k2",  "",  -0.2,   0.2, ".4f",
                    get=lambda c: c.lens.distortion_k2,
                    set=lambda c, v: setattr(c.lens, "distortion_k2", v)),
        _SliderSpec("vignetting",  "",   0.0,   1.0, ".3f",
                    get=lambda c: c.lens.vignetting,
                    set=lambda c, v: setattr(c.lens, "vignetting", v)),
        _SliderSpec("tx band 1",   "",   0.0,   1.0, ".3f",
                    get=_tx_get(0), set=_tx_set(0)),
        _SliderSpec("tx band 2",   "",   0.0,   1.0, ".3f",
                    get=_tx_get(1), set=_tx_set(1)),
        _SliderSpec("tx band 3",   "",   0.0,   1.0, ".3f",
                    get=_tx_get(2), set=_tx_set(2)),
        _SliderSpec("tx band 4",   "",   0.0,   1.0, ".3f",
                    get=_tx_get(3), set=_tx_set(3)),
    ]


def _make_exposure_sliders() -> List[_SliderSpec]:
    return [
        _SliderSpec("focus_m",      "m",   0.05, 100.0, ".2f",
                    get=lambda c: c.focus_m,
                    set=lambda c, v: setattr(c, "focus_m", v)),
        _SliderSpec("aperture",     "",    0.0,    0.12, ".4f",
                    get=lambda c: c.aperture,
                    set=lambda c, v: setattr(c, "aperture", v)),
        _SliderSpec("chrom. ab.",   "",    0.0,    0.05, ".4f",
                    get=lambda c: c.ca,
                    set=lambda c, v: setattr(c, "ca", v)),
        _SliderSpec("tilt_shift X", "",   -0.5,    0.5, ".3f",
                    get=lambda c: float(c.tilt_shift[0]),
                    set=lambda c, v: c.tilt_shift.__setitem__(0, v)),
        _SliderSpec("tilt_shift Y", "",   -0.5,    0.5, ".3f",
                    get=lambda c: float(c.tilt_shift[1]),
                    set=lambda c, v: c.tilt_shift.__setitem__(1, v)),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# GLSL — same fullscreen-quad approach as room_duty_station
# ─────────────────────────────────────────────────────────────────────────────

_HUD_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
uniform vec2 uPos;
uniform vec2 uSize;
uniform vec2 uWin;
out vec2 vUV;
void main() {
    vec2 ndc = ((uPos + aPos * uSize) / uWin) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUV = aUV;
}
"""

_HUD_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uTex;
out vec4 fragColor;
void main() { fragColor = texture(uTex, vUV); }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _font(size: int = 14) -> pygame.font.Font:
    try:
        return pygame.font.SysFont("consolas,monospace", size)
    except Exception:
        return pygame.font.Font(None, size)


def _label(surf, text, x, y, color=_TEXT_FG, size=14):
    f = _font(size)
    s = f.render(str(text), True, color[:3])
    surf.blit(s, (x, y))


def _slider(surf: pygame.Surface,
            x: int, y: int, w: int, h: int,
            spec: _SliderSpec, value: float) -> pygame.Rect:
    """Draw a labelled slider row.  Returns the rail Rect for hit-testing."""
    frac = _to_frac(value, spec)
    rail = pygame.Rect(x, y + 18, w, h)
    pygame.draw.rect(surf, _RAIL[:3], rail, border_radius=3)
    fw = max(4, int(frac * w))
    pygame.draw.rect(surf, _FILL[:3], pygame.Rect(x, y + 18, fw, h),
                     border_radius=3)
    kx = x + fw
    pygame.draw.circle(surf, _KNOB[:3], (kx, y + 18 + h // 2), 6)
    _label(surf, spec.label, x, y, _TEXT_DIM, 12)
    val_str = format(value, spec.fmt)
    if spec.unit:
        val_str += " " + spec.unit
    _label(surf, val_str, x + w + 6, y + 14, _TEXT_FG, 12)
    return rail


def _to_frac(value: float, spec: _SliderSpec) -> float:
    if spec.is_log:
        lo = math.log(max(spec.vmin, 1e-9))
        hi = math.log(max(spec.vmax, 1e-9))
        v  = math.log(max(value,     1e-9))
        return float(np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
    return float(np.clip((value - spec.vmin) / max(spec.vmax - spec.vmin, 1e-9),
                         0.0, 1.0))


def _from_frac(frac: float, spec: _SliderSpec) -> float:
    frac = float(np.clip(frac, 0.0, 1.0))
    if spec.is_log:
        lo = math.log(max(spec.vmin, 1e-9))
        hi = math.log(max(spec.vmax, 1e-9))
        return math.exp(lo + frac * (hi - lo))
    return spec.vmin + frac * (spec.vmax - spec.vmin)


# ─────────────────────────────────────────────────────────────────────────────
# CameraHudPanel
# ─────────────────────────────────────────────────────────────────────────────

class CameraHudPanel:
    """Full camera-parameter HUD panel.

    Call ``init_gl()`` once after a GL context exists.
    Call ``render(win_w, win_h, camera)`` every frame in IN_CAMERA mode.
    Call ``handle_event(ev, camera)`` for each pygame event.
    """

    # Panel pixel size
    PANEL_W = 820
    PANEL_H = 540

    # Column x-offsets and widths (relative to panel interior, 8px left pad)
    _COL_X = (8,   290,  570)
    _COL_W = (265, 265,  230)
    _COL_HEADS = ("SENSOR", "LENS", "EXPOSURE")

    def __init__(self) -> None:
        # Section slider lists (built once)
        self._sections: List[List[_SliderSpec]] = [
            _make_sensor_sliders(),
            _make_lens_sliders(),
            _make_exposure_sliders(),
        ]

        # GL state
        self._tex:   Optional[int] = None
        self._vao:   Optional[int] = None
        self._vbo:   Optional[int] = None
        self._prog:  Optional[int] = None
        self._ready: bool = False

        # Drag state: (section_idx, slider_idx) | None
        self._drag:    Optional[Tuple[int, int]] = None
        self._drag_x0: int   = 0
        self._drag_v0: float = 0.0
        # Rail rects: {(sec_idx, sl_idx): pygame.Rect} in panel local space
        self._rects: Dict[Tuple[int, int], pygame.Rect] = {}

        # Panel screen position (top-left), set each render call
        self._px = 0
        self._py = 0

        # Camera title string
        self._title: str = "CAMERA"

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def init_gl(self) -> None:
        if not _HAS_GL:
            return
        vs   = _gl_shaders.compileShader(_HUD_VS, GL_VERTEX_SHADER)
        fs   = _gl_shaders.compileShader(_HUD_FS, GL_FRAGMENT_SHADER)
        self._prog = _gl_shaders.compileProgram(vs, fs)
        quad = np.array([
            0, 0, 0, 1,
            1, 0, 1, 1,
            1, 1, 1, 0,
            0, 0, 0, 1,
            1, 1, 1, 0,
            0, 1, 0, 0,
        ], np.float32).reshape(-1, 4)
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)
        self._tex   = glGenTextures(1)
        self._ready = True

    def destroy_gl(self) -> None:
        self._ready = False

    # ── Panel build / render ──────────────────────────────────────────────────

    def _build_surface(self, camera) -> pygame.Surface:
        """Render the full panel to a pygame Surface."""
        W, H = self.PANEL_W, self.PANEL_H
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        surf.fill(_BG[:3])
        surf.set_alpha(_BG[3])
        # semi-transparent background
        pygame.draw.rect(surf, _BG[:3], pygame.Rect(0, 0, W, H))

        # Title bar
        title = self._title
        if hasattr(camera, "lens") and camera.lens.name:
            title += f"  /  {camera.lens.name}"
        if hasattr(camera, "sensor") and camera.sensor.name:
            title += f"  |  {camera.sensor.name}"
        _label(surf, title, 10, 8, _TEXT_HEAD, 15)
        pygame.draw.line(surf, _SEP[:3], (0, 28), (W, 28))

        # fov_y info — helpful heads-up for the ray tracer output
        try:
            fov_deg = math.degrees(camera.fov_y_rad())
            sens_h  = camera.sensor.height_mm
            foc     = camera.focal_mm
            fov_str = (f"fov_y = {fov_deg:.2f}°  "
                       f"(sensor_h={sens_h:.1f}mm  focal={foc:.1f}mm)")
            _label(surf, fov_str, 10, 32, _TEXT_DIM, 12)
        except Exception:
            pass

        # Column headers
        ROW_START = 52
        for col_i, head in enumerate(self._COL_HEADS):
            cx = self._COL_X[col_i]
            _label(surf, head, cx, ROW_START, _TEXT_HEAD, 13)
        pygame.draw.line(surf, _SEP[:3], (0, ROW_START + 16), (W, ROW_START + 16))

        self._rects.clear()

        # Slider rows
        ROW_H = 35     # pixels per slider row
        ROW_0 = ROW_START + 20

        for col_i, sliders in enumerate(self._sections):
            cx = self._COL_X[col_i]
            cw = self._COL_W[col_i] - 12  # leave margin for value text
            for sl_i, spec in enumerate(sliders):
                y = ROW_0 + sl_i * ROW_H
                if y + ROW_H > H - 8:
                    break  # panel too small for this slider
                try:
                    val = float(spec.get(camera))
                except Exception:
                    val = float(spec.vmin)
                rail = _slider(surf, cx, y, cw, 8, spec, val)
                self._rects[(col_i, sl_i)] = rail

        # Footer hint
        _label(surf, "DRAG sliders to adjust  |  all values live  |  E / Esc to exit",
               10, H - 18, _TEXT_DIM, 12)

        return surf

    def render(self, win_w: int, win_h: int, camera) -> None:
        """Build the pygame surface and upload to GL, then draw the quad."""
        if not self._ready:
            return
        # Position: left edge, vertically centred
        W, H = self.PANEL_W, self.PANEL_H
        self._px = 10
        self._py = max(0, (win_h - H) // 2)

        surf = self._build_surface(camera)
        raw  = pygame.image.tostring(surf, "RGBA", True)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Draw the quad
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self._prog)
        glUniform2f(glGetUniformLocation(self._prog, "uPos"),
                    float(self._px), float(self._py))
        glUniform2f(glGetUniformLocation(self._prog, "uSize"),
                    float(W), float(H))
        glUniform2f(glGetUniformLocation(self._prog, "uWin"),
                    float(win_w), float(win_h))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glUniform1i(glGetUniformLocation(self._prog, "uTex"), 0)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev, camera) -> bool:
        """Process mouse events.  Returns True if the event was consumed."""
        if not self._ready:
            return False

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            lx = ev.pos[0] - self._px
            ly = ev.pos[1] - self._py
            for (ci, si), rect in self._rects.items():
                if rect.collidepoint(lx, ly):
                    spec = self._sections[ci][si]
                    try:
                        self._drag_v0 = float(spec.get(camera))
                    except Exception:
                        self._drag_v0 = float(spec.vmin)
                    self._drag    = (ci, si)
                    self._drag_x0 = ev.pos[0]
                    return True

        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self._drag is not None:
                self._drag = None
                return True

        if ev.type == pygame.MOUSEMOTION and self._drag is not None:
            ci, si = self._drag
            spec   = self._sections[ci][si]
            cw     = float(self._COL_W[ci] - 12)
            dx     = ev.pos[0] - self._drag_x0
            # Map pixel delta to value delta through the same frac space
            frac0  = _to_frac(self._drag_v0, spec)
            frac1  = float(np.clip(frac0 + dx / max(1.0, cw), 0.0, 1.0))
            new_v  = _from_frac(frac1, spec)
            try:
                spec.set(camera, new_v)
            except Exception:
                pass
            return True

        return False

    # ── Title ─────────────────────────────────────────────────────────────────

    def set_title(self, name: str) -> None:
        """Set the camera item name shown in the panel header."""
        self._title = str(name)


