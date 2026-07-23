"""camera_designer_station.py
==============================
GL duty station for designing camera optics.

Layout
------
  ┌─────────────────┬──────────────────────────────────┬───────────────────┐
  │  LEFT PANEL     │  3-UP CROSS-SECTION VIEWPORTS     │  RIGHT PANEL      │
  │  component tree │  XZ sagittal / YZ tangential /    │  element props +  │
  │  + bake control │  XY aperture plane                │  LUT metrics      │
  └─────────────────┴──────────────────────────────────┴───────────────────┘

The 3-D viewport area is subdivided into three OpenGL sub-viewports:
  Top-left   : XZ sagittal cross-section (light paths through lens)
  Top-right  : YZ tangential cross-section (same but rotated 90°)
  Bottom     : XY aperture plane view (footprint + ray bundle)

All surface outlines are drawn as GL_LINE_STRIP primitives sampled directly
from the parametric surface equation — no mesh triangles.

The black-box scene (used for the fine-grain volumetric integrator preview)
is a flat black box enclosure + one glass sphere + one directional light
pointing toward the aperture plane.

Panels follow the same pattern as SimulatorStation:
  pygame.Surface → RGBA → glTexImage2D → HUD quad shader.
"""
from __future__ import annotations

import ctypes
import glob
import math
import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pygame

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_FALSE, GL_TRUE, GL_FLOAT,
        GL_FRAGMENT_SHADER, GL_LINE_LOOP, GL_LINE_STRIP, GL_LINES,
        GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_RGBA, GL_SCISSOR_TEST,
        GL_SRC_ALPHA, GL_STATIC_DRAW, GL_TEXTURE_2D, GL_TEXTURE_3D,
        GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_LINEAR,
        GL_TRIANGLE_FAN, GL_TRIANGLES, GL_UNSIGNED_BYTE, GL_VERTEX_SHADER,
        GL_TEXTURE0, GL_STREAM_DRAW,
        glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray,
        glBlendFunc, glBufferData, glClear, glClearColor,
        glDeleteBuffers, glDeleteTextures, glDeleteVertexArrays,
        glDepthMask, glDisable, glDrawArrays, glEnable,
        glEnableVertexAttribArray, glGenBuffers, glGenTextures,
        glGenVertexArrays, glGetUniformLocation, glLineWidth,
        glScissor, glTexImage2D, glTexParameteri,
        glUniform1f, glUniform1i, glUniform2f, glUniform3f, glUniform4f,
        glUniformMatrix4fv, glUseProgram, glVertexAttribPointer,
        glViewport,
    )
    from OpenGL.GL import shaders as _gl_shaders
    _HAS_GL = True
except ImportError:
    _HAS_GL = False

# Optical transport is deliberately absent from this module.  The station is
# an editor/presenter; the optical-engine pillar owns every tracer, compute
# context, transport mode, allocation and published preview product.
_HAS_GPU_FIELD = False
_GPU_RAY_FIELD_SCALE = 4.0
_GPU_RAY_LOG_SCALE = True
_GPU_RAY_FIELD_GAMMA = 0.62

try:
    from camera_designer.camera_preset import (
        CameraPreset, simple_doublet_preset, PRESET_REGISTRY, EmitterSpec,
    )
    _HAS_CAMERA_DESIGNER = True
except ImportError:
    _HAS_CAMERA_DESIGNER = False
    CameraPreset   = None  # type: ignore[assignment,misc]

# Compatibility name for old callers; no station-local transport is imported.
BakeWorker = None

try:
    from camera_software.auto_computer import CameraComputer as _CameraComputer
    _HAS_AUTO_COMPUTER = True
except Exception:
    _HAS_AUTO_COMPUTER = False
    _CameraComputer = None  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# Equipment colour palette — shared by _rebuild_lines + left-panel swatches
# ─────────────────────────────────────────────────────────────────────────────

_EL_COLORS: list = [                          # per-lens-element line colour (r,g,b,a)
    (0.3, 0.7, 1.0, 0.9),                     # 0 — blue
    (0.4, 1.0, 0.6, 0.9),                     # 1 — green
    (1.0, 0.8, 0.3, 0.9),                     # 2 — amber
    (0.9, 0.4, 0.9, 0.9),                     # 3 — violet
]
_APERTURE_COLOR = (1.0, 1.0, 0.0,  0.8)      # yellow
_SENSOR_COLOR   = (0.85, 0.25, 0.25, 0.95)   # red
_PLATE_COLOR    = (0.2, 0.8, 0.85, 0.70)     # cyan
_LIGHT_COLOR    = (1.0, 0.9, 0.3, 1.0)       # gold


def _float_rgba(c: tuple) -> tuple:
    """Convert float (r,g,b[,a]) tuple → int (r,g,b) for pygame."""
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


# ─────────────────────────────────────────────────────────────────────────────
# GLSL shaders
# ─────────────────────────────────────────────────────────────────────────────

# 2-D ortho line / point shader  (for cross-section outlines)
_LINE2D_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
uniform vec2 uScale;   // (half-width, half-height) of viewport in metres
uniform vec2 uOffset;  // pan offset in metres
void main() {
    vec2 p = (aPos - uOffset) / uScale;
    gl_Position = vec4(p.x, p.y, 0.0, 1.0);
}
"""

_LINE2D_FS = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main() { FragColor = uColor; }
"""

# HUD panel texture quad (pixel-space)
_HUD_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
uniform vec2 uRes;
out vec2 vUV;
void main() {
    vec2 ndc = aPos / uRes * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUV = aUV;
}
"""

_HUD_FS = """
#version 330 core
in  vec2 vUV;
uniform sampler2D uTex;
out vec4 FragColor;
void main() { FragColor = texture(uTex, vUV); }
"""




# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_tex(surf: pygame.Surface) -> int:
    raw = pygame.image.tobytes(surf, "RGBA", True)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 surf.get_width(), surf.get_height(), 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, raw)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def _update_tex(tex: int, surf: pygame.Surface) -> None:
    raw = pygame.image.tobytes(surf, "RGBA", True)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 surf.get_width(), surf.get_height(), 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, raw)
    glBindTexture(GL_TEXTURE_2D, 0)


def _quad_verts(x: int, y: int, w: int, h: int) -> np.ndarray:
    x0, y0, x1, y1 = float(x), float(y), float(x+w), float(y+h)
    return np.array([
        x0, y0, 0., 1.,
        x1, y0, 1., 1.,
        x1, y1, 1., 0.,
        x0, y0, 0., 1.,
        x1, y1, 1., 0.,
        x0, y1, 0., 0.,
    ], np.float32)


def _upload_lines(vao: int, vbo: int, pts: np.ndarray) -> int:
    """Re-upload a (N,2) float32 array into an existing VAO/VBO.
    Returns vertex count."""
    flat = np.ascontiguousarray(pts, np.float32).flatten()
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STREAM_DRAW)
    glBindVertexArray(0)
    return len(pts)


def _make_vao2() -> tuple[int, int]:
    """Create an empty VAO/VBO for 2-D (x,y) float32 vertices."""
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, 4, None, GL_STREAM_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
    glBindVertexArray(0)
    return vao, vbo


class _NumpyBack:
    """Thin wrapper so a raw numpy (H,W,C) image can be passed to _draw_back_image."""
    def __init__(self, img: np.ndarray) -> None:
        self._img = img

    def to_rect(self) -> np.ndarray:
        return self._img.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# SensorPlaneConfig — mutable runtime state for sensor + plate planes
# ─────────────────────────────────────────────────────────────────────────────

class SensorPlaneConfig:
    """Runtime offsets and modes for the digital sensor and large-format plate.

    Offsets are metres relative to the nominal sensor_z from the CameraPreset.
    Modes control how each plane participates in rendering:

        forming_positive    — transparent mesh, live-tinted by accumulated light
        clear               — fully transparent (passes all light through)
        fixed_positive      — frozen accumulated exposure (opaque positive film)
        emissive_lambertian — lambertian white-panel emitter (diffuse back-light)

    Projector recipe
    ----------------
    sensor_mode='fixed_positive' + plate_mode='emissive_lambertian':
    the plate emits white light back through the fixed positive on the sensor,
    projecting the latent image forward through the lens.

    Physical lens notes
    -------------------
    Each LensElement in the CameraPreset carries a refractive Surface
    (sphere / asphere / flat) with glass n_d, V_d.  The GPU ray tracer in
    camera_designer/scene_builder.py refracts every ray via Snell's law at
    each interface — the optics are already fully physical, not paraxial.
    A thin-lens approximation is only used for the initial LensManifold bake
    (LensTransform.compile); the live GPU path is completely ray-traced.

    Tilt-shift control plane (architecture note — deferred)
    -------------------------------------------------------
    A tilt-shift plane inside the lens barrel would work as follows:
      1. Insert a virtual TiltShiftNode between two LensElements at a chosen z.
         Node parameters: (shift_x, shift_y, tilt_x_deg, tilt_y_deg).
      2. In _rebuild_lines() segment the barrel outline at that z: draw
         tube-wall from barrel start to node-z offset by shift, then from
         node-z to barrel front.  Fill the gap between the two tube segments
         with ruled quads (lerp top-of-tube to top-of-tube, bottom to bottom).
      3. In scene_builder.py thread rays through the shift node using a thin
         wedge/prism model before the next refracting element.
    This ruled-segment tube geometry is non-trivial; deferred for now.
    """

    MODES = ('forming_positive', 'clear', 'fixed_positive', 'emissive_lambertian')
    MODE_LABELS = {
        'forming_positive':    'form +',
        'clear':               'clear',
        'fixed_positive':      'fixed +',
        'emissive_lambertian': 'emit',
    }

    def __init__(self) -> None:
        # Canonical camera solve controls.  These are seeded from the active
        # CameraManifest by CameraDesignerStation and written back to it before
        # the physical four-group train is solved again.
        self.zoom: float = 0.5
        self.focus_distance_m: float = 1.0
        # Sensor focal-plane fine adjustment (metres)
        self.sensor_x: float = 0.0
        self.sensor_y: float = 0.0
        self.sensor_z: float = 0.0
        self.sensor_mode: str = 'forming_positive'
        # Large-format plate offsets (metres); plate_z is depth behind sensor
        self.plate_x: float = 0.0
        self.plate_y: float = 0.0
        self.plate_z: float = 0.010
        self.plate_mode: str = 'forming_positive'
        # Per-element angular nudge (degrees +/-5):
        # {element_index: {pan_a, pan_b, tilt_a, tilt_b}}
        self.element_angles: dict = {}
        self.element_geometry: dict = {}

    def sync_elements(self, preset) -> None:
        self.element_geometry = {}
        for index, element in enumerate(preset.lens_group.elements):
            surface = element.surface
            k = float(getattr(surface, "K", 0.0))
            conic_type = (
                "sphere" if abs(k) < 1.0e-9 else
                "paraboloid" if abs(k + 1.0) < 1.0e-9 else
                "hyperboloid" if k < -1.0 else
                "prolate_ellipsoid" if k < 0.0 else
                "oblate_ellipsoid"
            )
            shift = tuple(getattr(element, "shift_xy_m", (0.0, 0.0)))
            tilt = tuple(getattr(element, "tilt_xy_deg", (0.0, 0.0)))
            self.element_geometry[index] = {
                "axial_offset": 0.0,
                "shift_x": float(shift[0]),
                "shift_y": float(shift[1]),
                "tilt_x": float(tilt[0]),
                "tilt_y": float(tilt[1]),
                "radius": float(getattr(surface, "R", 0.050)),
                "clear_radius": float(getattr(surface, "r_max", 0.020)),
                "conic_k": k,
                "conic_type": conic_type,
            }

    def el_geometry_value(self, idx: int, key: str, default=0.0):
        return self.element_geometry.get(idx, {}).get(key, default)

    def set_el_geometry_value(self, idx: int, key: str, value) -> None:
        self.element_geometry.setdefault(idx, {})[key] = value

    def el_angle(self, idx: int, key: str) -> float:
        return self.element_angles.get(idx, {}).get(key, 0.0)

    def set_el_angle(self, idx: int, key: str, val: float) -> None:
        if idx not in self.element_angles:
            self.element_angles[idx] = {}
        import numpy as _np
        self.element_angles[idx][key] = float(_np.clip(val, -5.0, 5.0))


# ─────────────────────────────────────────────────────────────────────────────
# Left panel — scrollable component tree + sensor/plate knobs
# ─────────────────────────────────────────────────────────────────────────────

class _ComponentTreePanel:
    """Left panel: fully-scrollable optics tree with sensor/plate knobs.

    All content is rendered into a virtual tall surface; only the visible
    window (clipped by _scroll) is blitted to the output surface.  Every
    section is reachable by scrolling.

    Interactive controls
    --------------------
    Float slider  — drag horizontally to adjust; right-click resets to 0;
                    mousewheel nudges by step while the cursor hovers.
    Choice knob   — left-click advances; right-click reverses.
    Element row   — left-click selects the element for the right panel.

    Sections (top to bottom)
    -------------------------
    PRESET / SPEC / MOUNT   — read-only optics summary
    LENS ELEMENTS           — physical-surface tree; each surface exposes
                              alignment, aperture, curvature and conic controls
    APERTURE STOP           — read-only
    SENSOR PLANE            — X/Y/Z offset knobs + mode selector
    PLATE PLANE             — X/Y/Z offset knobs + mode selector
    BAKE LUT                — bake button + status
    ORTHO MODE / SCENE      — existing slice-mode and light controls
    """
    PAD    = 6
    ROW_H  = 20
    KNOB_H = 36
    MODE_H = 28
    SECT_H = 18

    _XY_RANGE    = 0.05      # +/- 5 cm lateral
    _Z_RANGE     = 0.05      # +/- 5 cm axial
    _ANGLE_RANGE = 5.0       # +/- 5 degrees
    _XY_STEP     = 1e-4      # 0.1 mm
    _Z_STEP      = 1e-4      # 0.1 mm
    _ANGLE_STEP  = 0.05      # 0.05 degrees
    _FOCUS_MIN_M = 0.25
    _FOCUS_MAX_M = 20.0

    def __init__(self, preset, sensor_cfg=None) -> None:
        pygame.font.init()
        self._font   = pygame.font.SysFont("monospace", 13)
        self._font_s = pygame.font.SysFont("monospace", 11)
        self.preset      = preset
        self.sensor_cfg  = sensor_cfg or SensorPlaneConfig()
        self._selected_idx: int  = -1
        self._scroll:       int  = 0
        self._virtual_h:    int  = 1200
        # Virtual-space hit rects: name -> (lo, hi, kind, Rect)
        # kind is 'float', 'choice', or 'row'
        self._knob_rects: dict = {}
        # Screen-space ctrl dict kept for bake-button backward-compat
        self._ctrl: dict = {}
        # Drag state for float knobs
        self._drag_knob  = None
        self._drag_start = None
        self._drag_val0: float = 0.0
        self._hover_knob = None
        # External state flags
        self.bake_state: str = "idle"
        self.bake_msg:   str = ""
        self.manifold_bdpt_state: str = "idle"
        self.scene_lights: list = []
        self.glow_rays_done:  int   = 0
        self.glow_rays_per_s: float = 0.0

    # -------------------------------------------------------------------------
    # Render

    def render(self, w: int, h: int) -> pygame.Surface:
        # Eight physical surfaces expose full geometry controls; keep one
        # scrollable backing surface large enough that no later element is
        # silently clipped merely because earlier controls are expanded.
        VIRT = max(h, 4800)
        vs   = pygame.Surface((w, VIRT), pygame.SRCALPHA)
        vs.fill((14, 17, 26, 248))
        self._knob_rects = {}
        y = 0

        # Header strip
        pygame.draw.rect(vs, (22, 28, 42), pygame.Rect(0, y, w, 20))
        vs.blit(self._font.render("  OPTICS", True, (120, 180, 240)),
                (self.PAD, y + 2))
        y += 22

        # PRESET / SPEC / MOUNT
        y = self._sect(vs, y, "PRESET", w)
        vs.blit(self._font.render(f"  {self.preset.name}", True, (180, 200, 220)),
                (self.PAD, y + 2))
        y += 18

        y = self._sect(vs, y + 2, "SPEC", w)
        y = self._lbl(vs, y, f"  focal  {self.preset.focal_mm:.1f} mm")
        y = self._lbl(vs, y, f"  f/     {self.preset.f_number:.1f}")
        y = self._lbl(vs, y, f"  fov    {self.preset.fov_deg:.1f} deg")

        y = self._sect(vs, y + 2, "CAMERA SOLVE", w)
        cfg = self.sensor_cfg
        y = self._float_knob(
            vs, y, w, "camera_zoom", "  zoom",
            cfg.zoom, 0.0, 1.0, "", ".3f",
        )
        y = self._float_knob(
            vs, y, w, "focus_distance_m", "  focus distance",
            cfg.focus_distance_m,
            self._FOCUS_MIN_M, self._FOCUS_MAX_M, "m", ".3f",
        )
        y = self._lbl(
            vs, y,
            "  release knob to re-solve + refresh preview",
            (105, 145, 115),
        )

        y = self._sect(vs, y + 2, "MOUNT", w)
        mr = self.preset.mount_ring
        y = self._lbl(vs, y, f"  flange {mr.z_flange*1e3:.1f} mm")
        y = self._lbl(vs, y, f"  ap_z   {mr.aperture_plane_z*1e3:.2f} mm")
        y = self._lbl(vs, y, f"  clr    {mr.back_clearance*1e3:.1f} mm")

        # LENS ELEMENTS — canonical physical surface controls
        y = self._sect(vs, y + 2, "LENS ELEMENTS", w)
        y = self._lbl(vs, y, "  edits rebuild exact QR geometry on release",
                      (105, 145, 115))
        for idx, el in enumerate(self.preset.lens_group.elements):
            sel = (idx == self._selected_idx)
            bg  = (30, 55, 100) if sel else (18, 22, 32)
            pygame.draw.rect(vs, bg, pygame.Rect(0, y, w, self.ROW_H))
            pygame.draw.rect(vs, _float_rgba(_EL_COLORS[idx % len(_EL_COLORS)]),
                             pygame.Rect(4, y + 5, 10, 10))
            lbl_s = f" [{idx}] {el.label or el.surface.glsl_type}"
            vs.blit(self._font_s.render(lbl_s, True, (160, 175, 200)), (17, y + 4))
            self._knob_rects[f"el_{idx}"] = (0, 0, 'row',
                                              pygame.Rect(0, y, w, self.ROW_H))
            y += self.ROW_H
            cfg = self.sensor_cfg
            y = self._float_knob(vs, y, w,
                f"el{idx}_axial_offset", "  axial offset",
                cfg.el_geometry_value(idx, 'axial_offset'),
                -0.050, 0.050, "mm", ".3f", scale=1e3)
            y = self._float_knob(vs, y, w,
                f"el{idx}_shift_x", "  shift X",
                cfg.el_geometry_value(idx, 'shift_x'),
                -0.025, 0.025, "mm", ".3f", scale=1e3)
            y = self._float_knob(vs, y, w,
                f"el{idx}_shift_y", "  shift Y",
                cfg.el_geometry_value(idx, 'shift_y'),
                -0.025, 0.025, "mm", ".3f", scale=1e3)
            y = self._float_knob(vs, y, w,
                f"el{idx}_tilt_x", "  tilt X",
                cfg.el_geometry_value(idx, 'tilt_x'),
                -15.0, 15.0, "deg", ".3f")
            y = self._float_knob(vs, y, w,
                f"el{idx}_tilt_y", "  tilt Y",
                cfg.el_geometry_value(idx, 'tilt_y'),
                -15.0, 15.0, "deg", ".3f")
            y = self._float_knob(vs, y, w,
                f"el{idx}_radius", "  curvature radius",
                cfg.el_geometry_value(idx, 'radius', 0.050),
                -0.500, 0.500, "mm", ".3f", scale=1e3)
            y = self._float_knob(vs, y, w,
                f"el{idx}_clear_radius", "  clear radius",
                cfg.el_geometry_value(idx, 'clear_radius', 0.020),
                0.001, 0.120, "mm", ".3f", scale=1e3)
            y = self._choice_knob(vs, y, w,
                f"el{idx}_conic_type", "  conic type",
                cfg.el_geometry_value(idx, 'conic_type', 'sphere'),
                ["sphere", "paraboloid", "hyperboloid",
                 "prolate_ellipsoid", "oblate_ellipsoid"],
                {
                    "sphere": "sphere", "paraboloid": "parabola",
                    "hyperboloid": "hyperbola", "prolate_ellipsoid": "prolate",
                    "oblate_ellipsoid": "oblate",
                })
            y = self._float_knob(vs, y, w,
                f"el{idx}_conic_k", "  conic constant K",
                cfg.el_geometry_value(idx, 'conic_k'),
                -5.0, 5.0, "", ".4f")
            y += 3

        # APERTURE STOP
        y = self._sect_swatch(vs, y + 2, "APERTURE STOP", w, _float_rgba(_APERTURE_COLOR))
        ap = self.preset.aperture_stop
        y = self._lbl(vs, y, f"  z    {ap.z_pos*1e3:.2f} mm")
        y = self._lbl(vs, y, f"  r    {ap.r_outer*1e3:.2f} mm")

        # SENSOR PLANE
        y = self._sect_swatch(vs, y + 2, "SENSOR PLANE", w, _float_rgba(_SENSOR_COLOR))
        sn = self.preset.sensor
        y = self._lbl(vs, y, f"  nominal z  {sn.z_pos*1e3:.2f} mm")
        y = self._lbl(vs, y, f"  r          {sn.r_max*1e3:.2f} mm")
        y = self._lbl(vs, y, f"  pitch      {sn.pixel_pitch*1e6:.2f} um")
        cfg = self.sensor_cfg
        y = self._float_knob(vs, y, w, "sensor_x",
                             "  offset X", cfg.sensor_x,
                             -self._XY_RANGE, self._XY_RANGE, "mm",
                             ".4f", scale=1e3)
        y = self._float_knob(vs, y, w, "sensor_y",
                             "  offset Y", cfg.sensor_y,
                             -self._XY_RANGE, self._XY_RANGE, "mm",
                             ".4f", scale=1e3)
        y = self._float_knob(vs, y, w, "sensor_z",
                             "  offset Z", cfg.sensor_z,
                             -self._Z_RANGE, self._Z_RANGE, "mm",
                             ".4f", scale=1e3)
        y = self._choice_knob(vs, y, w, "sensor_mode",
                              "  mode", cfg.sensor_mode,
                              list(SensorPlaneConfig.MODES),
                              SensorPlaneConfig.MODE_LABELS)

        # PLATE PLANE
        y = self._sect_swatch(vs, y + 2, "PLATE PLANE", w, _float_rgba(_PLATE_COLOR))
        y = self._lbl(vs, y, "  large-format plate behind sensor")
        y = self._float_knob(vs, y, w, "plate_x",
                             "  offset X", cfg.plate_x,
                             -self._XY_RANGE, self._XY_RANGE, "mm",
                             ".4f", scale=1e3)
        y = self._float_knob(vs, y, w, "plate_y",
                             "  offset Y", cfg.plate_y,
                             -self._XY_RANGE, self._XY_RANGE, "mm",
                             ".4f", scale=1e3)
        y = self._float_knob(vs, y, w, "plate_z",
                             "  depth Z (behind sensor)", cfg.plate_z,
                             0.0, 0.15, "mm", ".3f", scale=1e3)
        y = self._choice_knob(vs, y, w, "plate_mode",
                              "  mode", cfg.plate_mode,
                              list(SensorPlaneConfig.MODES),
                              SensorPlaneConfig.MODE_LABELS)

        # BAKE LUT
        y += 6
        bake_clr = {"idle":   (40, 80, 160), "baking": (140, 100, 20),
                    "done":   (20, 120, 50),  "error":  (140, 30, 30),
                    }.get(self.bake_state, (40, 80, 160))
        br = pygame.Rect(self.PAD, y, w - 2 * self.PAD, 26)
        pygame.draw.rect(vs, bake_clr, br)
        pygame.draw.rect(vs, (100, 120, 150), br, 1)
        blbl = {"idle":   "BAKE LUT", "baking": "BAKING...",
                "done":   "BAKED",    "error":  "ERROR",
                }.get(self.bake_state, "BAKE LUT")
        bt = self._font.render(blbl, True, (220, 230, 240))
        vs.blit(bt, (br.x + (br.w - bt.get_width()) // 2, br.y + 4))
        self._knob_rects["btn_bake"] = (0, 0, 'row', br)
        self._ctrl["btn_bake"] = br   # backward-compat
        y += 32

        # MANIFOLD BDPT RENDER
        y += 2
        mbdpt_clr = {
            "idle":      (60, 50, 120),
            "rendering": (120, 80, 20),
            "done":      (20, 100, 60),
            "error":     (130, 30, 30),
        }.get(self.manifold_bdpt_state, (60, 50, 120))
        mbr = pygame.Rect(self.PAD, y, w - 2 * self.PAD, 26)
        pygame.draw.rect(vs, mbdpt_clr, mbr)
        pygame.draw.rect(vs, (100, 100, 160), mbr, 1)
        mblbl = {
            "idle":      "MANIFOLD BDPT",
            "rendering": "RENDERING...",
            "done":      "BDPT DONE",
            "error":     "BDPT ERROR",
        }.get(self.manifold_bdpt_state, "MANIFOLD BDPT")
        mbt = self._font.render(mblbl, True, (210, 200, 240))
        vs.blit(mbt, (mbr.x + (mbr.w - mbt.get_width()) // 2, mbr.y + 4))
        self._knob_rects["btn_manifold_bdpt"] = (0, 0, 'row', mbr)
        y += 32

        # ORTHO MODE
        mode_state = getattr(self, '_slice_mode_label', None)
        if mode_state is not None:
            mc = (255, 190, 60) if mode_state[0] else (80, 130, 200)
            if mode_state[0]:
                lbl_m = f"  SLICE  {mode_state[1]*1e3:.1f} mm slab"
            else:
                lbl_m = "  PROJECTION"
            y = self._sect(vs, y, "ORTHO MODE", w)
            y = self._lbl(vs, y, lbl_m, mc)
            y = self._lbl(vs, y, "  S=toggle  [=thinner  ]=wider", (60, 70, 80))

        # SCENE
        y = self._sect_swatch(vs, y, "SCENE", w, _float_rgba(_LIGHT_COLOR))
        if self.scene_lights:
            for i, lt in enumerate(self.scene_lights):
                p = lt["pos"]
                sl = f"  L{i} ({p[0]*1e3:.1f},{p[1]*1e3:.1f},{p[2]*1e3:.1f}) mm"
                y = self._lbl(vs, y, sl, (255, 220, 100))
        else:
            y = self._lbl(vs, y, "  (no lights in scene)",
                          (70, 80, 90))
        y = self._lbl(vs, y, "  SPACE=build  C=clr", (60, 70, 80))
        y = self._lbl(vs, y, "  G=brightness", (60, 70, 80))
        if self.bake_msg:
            vs.blit(self._font_s.render(f"  {self.bake_msg}", True, (140, 180, 140)),
                    (4, y + 2))
            y += 16

        self._virtual_h = y + 10

        # Blit visible slice onto output surface
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((14, 17, 26, 248))
        self._scroll = int(np.clip(self._scroll, 0, max(0, self._virtual_h - h)))
        vis_h = min(h, max(0, self._virtual_h - self._scroll))
        if vis_h > 0:
            clip = pygame.Rect(0, self._scroll, w, vis_h)
            surf.blit(vs.subsurface(clip), (0, 0))

        # Scrollbar
        if self._virtual_h > h:
            sb_h = max(18, int(h * h / self._virtual_h))
            sb_y = int(self._scroll * (h - sb_h) / max(self._virtual_h - h, 1))
            pygame.draw.rect(surf, (38, 42, 56), pygame.Rect(w - 5, 0, 5, h))
            pygame.draw.rect(surf, (80, 100, 160),
                             pygame.Rect(w - 5, sb_y, 5, sb_h))
        return surf

    # -------------------------------------------------------------------------
    # Knob row renderers (write into virtual surface vs at virtual y)

    def _float_knob(self, vs, y: int, w: int, name: str, label: str,
                    val: float, lo: float, hi: float,
                    unit: str = "", fmt: str = ".3f",
                    scale: float = 1.0) -> int:
        hov = (name == self._hover_knob or name == self._drag_knob)
        bg  = (32, 40, 56) if hov else (18, 22, 32)
        pygame.draw.rect(vs, bg, pygame.Rect(0, y, w, self.KNOB_H))
        vs.blit(self._font_s.render(label, True, (95, 115, 135)),
                (self.PAD, y + 2))
        disp = f"{val * scale:{fmt}} {unit}".strip()
        vs.blit(self._font.render(disp, True, (200, 215, 230)),
                (self.PAD, y + 13))
        bar_w = w - 2 * self.PAD
        frac  = (val - lo) / max(hi - lo, 1e-12)
        bar_y = y + self.KNOB_H - 7
        pygame.draw.rect(vs, (36, 40, 54), pygame.Rect(self.PAD, bar_y, bar_w, 5))
        fw = max(2, int(bar_w * float(np.clip(frac, 0, 1))))
        pygame.draw.rect(vs, (55, 105, 195),
                         pygame.Rect(self.PAD, bar_y, fw, 5))
        self._knob_rects[name] = (lo, hi, 'float',
                                  pygame.Rect(0, y, w, self.KNOB_H))
        return y + self.KNOB_H + 2

    def _choice_knob(self, vs, y: int, w: int, name: str, label: str,
                     val: str, choices: list, labels=None) -> int:
        labels  = labels or {c: c for c in choices}
        hov     = (name == self._hover_knob)
        bg      = (26, 34, 48) if hov else (20, 25, 36)
        pygame.draw.rect(vs, bg, pygame.Rect(0, y, w, self.MODE_H))
        vs.blit(self._font_s.render(label, True, (90, 110, 130)),
                (self.PAD, y + 2))
        display = labels.get(val, val)
        tw      = self._font.size(display)[0] + 10
        pill_x  = w - tw - self.PAD
        pygame.draw.rect(vs, (48, 78, 148),
                         pygame.Rect(pill_x, y + 5, tw, 18))
        vs.blit(self._font.render(display, True, (200, 220, 240)),
                (pill_x + 5, y + 6))
        vs.blit(self._font_s.render("< >", True, (70, 90, 120)),
                (self.PAD + 60, y + 8))
        self._knob_rects[name] = (choices, labels, 'choice',
                                  pygame.Rect(0, y, w, self.MODE_H))
        return y + self.MODE_H + 2

    # -------------------------------------------------------------------------
    # Event handling

    def handle_event(self, ev, x_off: int = 0, y_off: int = 0,
                     panel_w: int = 230) -> tuple:
        """Returns (consumed: bool, action_key: str).

        action_key is non-empty only for the bake button and element row
        selections.  Knob changes are applied directly to sensor_cfg.
        """
        cfg = self.sensor_cfg

        # Mousewheel: scroll or value nudge
        if ev.type == pygame.MOUSEWHEEL:
            mx, _ = pygame.mouse.get_pos()
            if x_off <= mx < x_off + panel_w:
                if self._hover_knob and self._hover_knob in self._knob_rects:
                    entry = self._knob_rects[self._hover_knob]
                    if entry[2] == 'float':
                        lo, hi = entry[0], entry[1]
                        step = (hi - lo) / 200.0
                        cur  = self._read_float(self._hover_knob, cfg)
                        self._apply_float(self._hover_knob, cfg, cur + step * ev.y)
                        return True, self._hover_knob
                self._scroll = int(np.clip(
                    self._scroll - ev.y * self.ROW_H,
                    0, max(0, self._virtual_h - 400)))
                return True, ""

        # Hover tracking + drag continuation
        if ev.type == pygame.MOUSEMOTION:
            mx, my = ev.pos
            lx, ly = mx - x_off, my - y_off
            vy = ly + self._scroll
            self._hover_knob = None
            for name, entry in self._knob_rects.items():
                if entry[3].collidepoint(lx, vy):
                    self._hover_knob = name
                    break
            if self._drag_knob and self._drag_start:
                dx = mx - self._drag_start[0]
                lo, hi = self._knob_rects[self._drag_knob][:2]
                bar_px  = panel_w - 2 * self.PAD
                new_val = self._drag_val0 + dx / max(bar_px, 1) * (hi - lo)
                self._apply_float(self._drag_knob, cfg, new_val)
                return True, self._drag_knob

        # End drag
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self._drag_knob:
                finished = self._drag_knob
                self._drag_knob  = None
                self._drag_start = None
                return True, finished

        # Click: start drag or activate choice / row
        if ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            lx, ly = mx - x_off, my - y_off
            if not (0 <= lx < panel_w):
                return False, ""
            vy = ly + self._scroll
            for name, entry in self._knob_rects.items():
                if not entry[3].collidepoint(lx, vy):
                    continue
                lo, hi, kind, _ = entry
                if kind == 'row':
                    if name.startswith("el_"):
                        self._selected_idx = int(name[3:])
                    return True, name
                if kind == 'float':
                    if ev.button == 3:
                        self._apply_float(name, cfg, 0.0)
                    else:
                        self._drag_knob  = name
                        self._drag_start = (mx, my)
                        self._drag_val0  = self._read_float(name, cfg)
                    return True, name
                if kind == 'choice':
                    choices   = lo   # lo holds choices list for choice knobs
                    direction = -1 if ev.button == 3 else 1
                    cur       = self._read_choice(name, cfg)
                    idx       = choices.index(cur) if cur in choices else 0
                    self._apply_choice(name, cfg,
                                       choices[(idx + direction) % len(choices)])
                    return True, name
        return False, ""

    # -------------------------------------------------------------------------
    # Knob state read / write dispatch

    def _read_float(self, name: str, cfg) -> float:
        if name == "camera_zoom": return cfg.zoom
        if name == "focus_distance_m": return cfg.focus_distance_m
        if name == "sensor_x":   return cfg.sensor_x
        if name == "sensor_y":   return cfg.sensor_y
        if name == "sensor_z":   return cfg.sensor_z
        if name == "plate_x":    return cfg.plate_x
        if name == "plate_y":    return cfg.plate_y
        if name == "plate_z":    return cfg.plate_z
        if name.startswith("el") and "_" in name:
            parts = name.split("_", 1)
            idx = int(parts[0][2:])
            key = parts[1]
            return float(cfg.el_geometry_value(idx, key, 0.0))
        return 0.0

    def _apply_float(self, name: str, cfg, val: float) -> None:
        if name == "camera_zoom":
            cfg.zoom = float(np.clip(val, 0.0, 1.0))
        elif name == "focus_distance_m":
            cfg.focus_distance_m = float(np.clip(
                val, self._FOCUS_MIN_M, self._FOCUS_MAX_M
            ))
        elif name == "sensor_x":
            cfg.sensor_x = float(np.clip(val, -self._XY_RANGE, self._XY_RANGE))
        elif name == "sensor_y":
            cfg.sensor_y = float(np.clip(val, -self._XY_RANGE, self._XY_RANGE))
        elif name == "sensor_z":
            cfg.sensor_z = float(np.clip(val, -self._Z_RANGE, self._Z_RANGE))
        elif name == "plate_x":
            cfg.plate_x = float(np.clip(val, -self._XY_RANGE, self._XY_RANGE))
        elif name == "plate_y":
            cfg.plate_y = float(np.clip(val, -self._XY_RANGE, self._XY_RANGE))
        elif name == "plate_z":
            cfg.plate_z = float(np.clip(val, 0.0, 0.15))
        elif name.startswith("el") and "_" in name:
            parts = name.split("_", 1)
            idx = int(parts[0][2:])
            key = parts[1]
            if key == "radius" and abs(float(val)) < 1.0e-3:
                val = cfg.el_geometry_value(idx, "radius", 0.050)
            if key == "clear_radius":
                val = max(
                    1.0e-3,
                    float(val) if float(val) > 0.0
                    else float(cfg.el_geometry_value(idx, "clear_radius", 0.020)),
                )
            cfg.set_el_geometry_value(idx, key, float(val))
            if key == "conic_k":
                k = float(val)
                conic_type = (
                    "sphere" if abs(k) < 1.0e-9 else
                    "paraboloid" if abs(k + 1.0) < 1.0e-9 else
                    "hyperboloid" if k < -1.0 else
                    "prolate_ellipsoid" if k < 0.0 else
                    "oblate_ellipsoid"
                )
                cfg.set_el_geometry_value(idx, "conic_type", conic_type)

    def _read_choice(self, name: str, cfg) -> str:
        if name == "sensor_mode": return cfg.sensor_mode
        if name == "plate_mode":  return cfg.plate_mode
        if name.startswith("el") and name.endswith("_conic_type"):
            idx = int(name[2:].split("_", 1)[0])
            return str(cfg.el_geometry_value(idx, "conic_type", "sphere"))
        return ""

    def _apply_choice(self, name: str, cfg, val: str) -> None:
        if name == "sensor_mode": cfg.sensor_mode = val
        if name == "plate_mode":  cfg.plate_mode  = val
        if name.startswith("el") and name.endswith("_conic_type"):
            idx = int(name[2:].split("_", 1)[0])
            cfg.set_el_geometry_value(idx, "conic_type", str(val))
            from camera_software.camera_designer_bridge import CONIC_TYPE_DEFAULT_K
            cfg.set_el_geometry_value(
                idx, "conic_k", float(CONIC_TYPE_DEFAULT_K[str(val)])
            )

    # -------------------------------------------------------------------------
    # Helpers

    def _sect(self, vs, y: int, label: str, w: int) -> int:
        pygame.draw.rect(vs, (26, 31, 45), pygame.Rect(0, y, w, self.SECT_H))
        vs.blit(self._font_s.render(label, True, (90, 120, 170)),
                (self.PAD, y + 2))
        return y + self.SECT_H + 2

    def _sect_swatch(self, vs, y: int, label: str, w: int,
                     swatch: tuple) -> int:
        """Section header with a small colour square on the right edge."""
        pygame.draw.rect(vs, (26, 31, 45), pygame.Rect(0, y, w, self.SECT_H))
        vs.blit(self._font_s.render(label, True, (90, 120, 170)),
                (self.PAD, y + 2))
        pygame.draw.rect(vs, swatch, pygame.Rect(w - 16, y + 4, 10, 10))
        return y + self.SECT_H + 2

    def _lbl(self, vs, y: int, text: str, color=(120, 140, 160)) -> int:
        vs.blit(self._font_s.render(text, True, color), (self.PAD, y + 1))
        return y + 15

    # Backward-compat aliases
    def _section(self, surf, y, label, w) -> int:
        return self._sect(surf, y, label, w)

    def _label(self, surf, y, text, color=(120, 140, 160)) -> int:
        return self._lbl(surf, y, text, color)


# ─────────────────────────────────────────────────────────────────────────────
# Right panel — selected element properties + LUT quality
# ─────────────────────────────────────────────────────────────────────────────

class _ElementPropsPanel:
    PAD    = 8
    ROW_H  = 20
    MODE_H = 28

    _BR_VIEWS    = (
        'surface_scan', 'wave_arena', 'complex_transport', 'gpu_field',
        'sensor', 'plate', 'manifold_bdpt',
    )
    _BR_LABELS   = {'gpu_field': 'GPU field', 'sensor': 'sensor',
                    'plate': 'plate', 'manifold_bdpt': 'manifold BDPT',
                    'surface_scan': 'pixel-ray preview',
                    'wave_arena': 'wave arena',
                    'complex_transport': 'complex transport'}
    _PLACE_MODES = ('mesh',)
    _PLACE_LABELS = {'mesh': 'mesh'}

    def __init__(self):
        pygame.font.init()
        self._font   = pygame.font.SysFont("monospace", 13)
        self._font_s = pygame.font.SysFont("monospace", 11)
        self.element_info: dict = {}
        self.lut_info:     dict = {}
        # Glow stats (fed from station trickle thread)
        self.glow_rays_done: int   = 0
        self.glow_rays_per_s: float = 0.0
        self.glow_segs_total: int  = 0
        # Interactive state
        self.bottom_right_view: str = 'surface_scan'
        self.place_mode: str = 'mesh'
        self._knob_rects: dict = {}
        # Sim parameters (control the GPU ray pipeline)
        self.sim_rays_per_frame: int   = 16384
        self.sim_max_bounces:    int   = 8
        self.sim_norm_mode:      str   = 'none'    # none / reinhard / clamp
        self.sim_activation:     str   = 'log'     # log / linear / sqrt / gamma
        self.sim_ctx_enabled:    dict  = {}         # label -> bool
        # Camera drive parameters (fed to CameraComputer + auto-ISO)
        self.cam_auto_iso:    bool  = True
        self.cam_iso:         float = 4.0
        self.cam_target_ev:   float = math.log2(0.18)   # 18 % grey standard
        self.surface_scan_generation: int = 0
        self.diagnostic_state: str = "waiting"
        self.diagnostic_rays: int = 0
        self.diagnostic_segments: int = 0
        self.diagnostic_sensor_hits: int = 0
        self.diagnostic_wavelengths: int = 0
        # Scroll + drag state for right-panel virtual surface
        self._scroll:     int   = 0
        self._virtual_h:  int   = 1200
        self._drag_knob   = None
        self._drag_start  = None
        self._drag_val0:  float = 0.0
        self._hover_knob  = None

    def render(self, w: int, h: int) -> pygame.Surface:
        VIRT = max(h, 2600)
        self._knob_rects = {}
        vs = pygame.Surface((w, VIRT), pygame.SRCALPHA)
        vs.fill((14, 17, 26, 248))
        pygame.draw.rect(vs, (22, 28, 42), pygame.Rect(0, 0, w, 20))
        vs.blit(self._font.render("  ELEMENT", True, (120, 180, 240)), (self.PAD, 2))

        y = 22
        if self.element_info:
            y = self._section(vs, y, "SURFACE", w)
            for k, v in self.element_info.items():
                y = self._kv(vs, y, k, str(v))
        else:
            vs.blit(self._font_s.render("  (select element ->)", True,
                    (80, 90, 110)), (self.PAD, y + 8))
            y += 30

        y = self._section(vs, y + 4, "LUT QUALITY", w)
        if self.lut_info:
            for k, v in self.lut_info.items():
                y = self._kv(vs, y, k, str(v))
        else:
            vs.blit(self._font_s.render("  (bake to measure)", True,
                    (80, 90, 110)), (self.PAD, y + 4))
            y += 20

        y = self._section(vs, y + 4, "GLOW TRACE", w)
        y = self._kv(vs, y, "rays", f"{self.glow_rays_done:,}")
        y = self._kv(vs, y, "segs", f"{self.glow_segs_total:,}")
        y = self._kv(vs, y, "r/s",
                     f"{self.glow_rays_per_s:.0f}" if self.glow_rays_per_s else "idle")

        y = self._section(vs, y + 4, "NATIVE RAY DIAGNOSTIC", w)
        y = self._kv(vs, y, "state", self.diagnostic_state)
        y = self._kv(vs, y, "rays", f"{self.diagnostic_rays:,}")
        y = self._kv(vs, y, "segments", f"{self.diagnostic_segments:,}")
        y = self._kv(vs, y, "sensor hits", f"{self.diagnostic_sensor_hits:,}")
        y = self._kv(vs, y, "spectral lanes", str(self.diagnostic_wavelengths))
        y = self._kv(vs, y, "source", "native mesh transport")
        vs.blit(self._font_s.render(
            "  line colour = wavelength; follows every bounce",
            True, (95, 125, 155),
        ), (self.PAD, y + 1))
        y += 15

        # VIEW CONTROLS
        y = self._section(vs, y + 6, "BOTTOM-RIGHT VIEW", w)
        y = self._choice_knob(vs, y, w, "bottom_right_view",
                              "  view mode", self.bottom_right_view,
                              self._BR_VIEWS, self._BR_LABELS)
        if self.bottom_right_view == "surface_scan":
            hints = (
                "  LIVE: left panel > SENSOR PLANE",
                "  drag or wheel offset X/Y/Z",
                f"  published generation {self.surface_scan_generation}",
            )
            hint_color = (105, 150, 115)
        else:
            hints = ("  sensor/plate require accumulated exposure",)
            hint_color = (60, 75, 90)
        for hint in hints:
            vs.blit(self._font_s.render(hint, True, hint_color),
                    (self.PAD, y + 2))
            y += 14

        y = self._section(vs, y + 4, "RIGHT-CLICK PLACE", w)
        vs.blit(self._font_s.render("  meshes only (light spawn removed)", True,
            (60, 75, 90)), (self.PAD, y + 2))
        y += 18

        # ── SIM PARAMS ────────────────────────────────────────────────────
        _RAYS_OPTS   = ['1024', '2048', '4096', '8192',
                        '16384', '32768', '65536', '131072']
        _BOUNCE_OPTS = ['0', '1', '2', '4', '6', '8', '12', '16', '24', '32']
        y = self._section(vs, y + 4, "SIM PARAMS", w)
        y = self._choice_knob(vs, y, w, "sim_rays_per_frame",
                              "  rays/frame", str(self.sim_rays_per_frame),
                              _RAYS_OPTS, {v: v for v in _RAYS_OPTS})
        y = self._choice_knob(vs, y, w, "sim_max_bounces",
                              "  max bounces", str(self.sim_max_bounces),
                              _BOUNCE_OPTS, {v: v for v in _BOUNCE_OPTS})
        y = self._choice_knob(vs, y, w, "sim_norm_mode",
                              "  normalization", self.sim_norm_mode,
                              ['none', 'reinhard', 'clamp'],
                              {'none': 'none', 'reinhard': 'reinhard',
                               'clamp': 'clamp'})
        y = self._choice_knob(vs, y, w, "sim_activation",
                              "  activation", self.sim_activation,
                              ['log', 'linear', 'sqrt', 'gamma'],
                              {'log': 'log', 'linear': 'linear',
                               'sqrt': 'sqrt', 'gamma': 'gamma'})

        # ── CONTEXT LEVELS (ScaleContext spheres — one per optical element) ─
        if self.sim_ctx_enabled:
            y = self._section(vs, y + 4, "CONTEXT LEVELS", w)
            for lbl, enabled in self.sim_ctx_enabled.items():
                y = self._choice_knob(vs, y, w, f"ctx_{lbl}",
                                      f"  {lbl[:14]}",
                                      'on' if enabled else 'off',
                                      ['on', 'off'],
                                      {'on': 'on', 'off': 'off'})
        else:
            y = self._section(vs, y + 4, "CONTEXT LEVELS", w)
            vs.blit(self._font_s.render("  build scene first (SPACE)",
                    True, (70, 80, 90)), (self.PAD, y + 2))
            y += 16

        # ── CAMERA ────────────────────────────────────────────────────────
        y = self._section(vs, y + 4, "CAMERA", w)
        y = self._choice_knob(vs, y, w, "cam_auto_iso",
                              "  auto ISO",
                              'on' if self.cam_auto_iso else 'off',
                              ['on', 'off'], {'on': 'on', 'off': 'off'})
        y = self._float_knob(vs, y, w, "cam_iso",
                             "  ISO / field scale", self.cam_iso,
                             0.25, 512.0, "", ".2f")
        y = self._float_knob(vs, y, w, "cam_target_ev",
                             "  target EV", self.cam_target_ev,
                             -8.0, 4.0, "", ".2f")
        vs.blit(self._font_s.render("  EV=log2(scene luminance)", True,
                (60, 75, 90)), (self.PAD, y + 2))
        y += 16

        self._virtual_h = y + 10

        # Blit visible slice onto output surface
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((14, 17, 26, 248))
        self._scroll = int(np.clip(self._scroll, 0, max(0, self._virtual_h - h)))
        vis_h = min(h, max(0, self._virtual_h - self._scroll))
        if vis_h > 0:
            clip = pygame.Rect(0, self._scroll, w, vis_h)
            surf.blit(vs.subsurface(clip), (0, 0))

        # Scrollbar
        if self._virtual_h > h:
            sb_h = max(18, int(h * h / self._virtual_h))
            sb_y = int(self._scroll * (h - sb_h) / max(self._virtual_h - h, 1))
            pygame.draw.rect(surf, (38, 42, 56), pygame.Rect(w - 5, 0, 5, h))
            pygame.draw.rect(surf, (80, 100, 160),
                             pygame.Rect(w - 5, sb_y, 5, sb_h))
        return surf

    def _section(self, surf, y, label, w) -> int:
        pygame.draw.rect(surf, (26, 31, 45), pygame.Rect(0, y, w, 18))
        surf.blit(self._font_s.render(label, True, (90, 120, 170)), (self.PAD, y+2))
        return y + 20

    def _kv(self, surf, y, key, val) -> int:
        kw = self._font_s.size(key + ": ")[0]
        surf.blit(self._font_s.render(key + ": ", True, (100, 115, 130)), (self.PAD, y+1))
        surf.blit(self._font_s.render(val, True, (180, 200, 220)), (self.PAD + kw, y+1))
        return y + 16

    def _choice_knob(self, surf, y: int, w: int, name: str, label: str,
                     val: str, choices: list, labels: dict) -> int:
        bg = (26, 34, 48)
        pygame.draw.rect(surf, bg, pygame.Rect(0, y, w, self.MODE_H))
        surf.blit(self._font_s.render(label, True, (90, 110, 130)), (self.PAD, y + 2))
        display = labels.get(val, val)
        tw      = self._font.size(display)[0] + 10
        pill_x  = w - tw - self.PAD
        pygame.draw.rect(surf, (48, 78, 148), pygame.Rect(pill_x, y + 5, tw, 18))
        surf.blit(self._font.render(display, True, (200, 220, 240)), (pill_x + 5, y + 6))
        surf.blit(self._font_s.render("< >", True, (70, 90, 120)), (self.PAD + 60, y + 8))
        self._knob_rects[name] = (choices, labels, 'choice',
                                  pygame.Rect(0, y, w, self.MODE_H))
        return y + self.MODE_H + 2

    KNOB_H = 36

    def _float_knob(self, surf, y: int, w: int, name: str, label: str,
                    val: float, lo: float, hi: float,
                    unit: str = "", fmt: str = ".3f") -> int:
        hov = (name == self._hover_knob or name == self._drag_knob)
        bg  = (32, 40, 56) if hov else (18, 22, 32)
        pygame.draw.rect(surf, bg, pygame.Rect(0, y, w, self.KNOB_H))
        surf.blit(self._font_s.render(label, True, (95, 115, 135)),
                  (self.PAD, y + 2))
        surf.blit(self._font.render(f"{val:{fmt}} {unit}".strip(),
                                    True, (200, 215, 230)),
                  (self.PAD, y + 13))
        bar_w = w - 2 * self.PAD
        frac  = (val - lo) / max(hi - lo, 1e-12)
        bar_y = y + self.KNOB_H - 7
        pygame.draw.rect(surf, (36, 40, 54), pygame.Rect(self.PAD, bar_y, bar_w, 5))
        fw = max(2, int(bar_w * float(np.clip(frac, 0, 1))))
        pygame.draw.rect(surf, (55, 105, 195),
                         pygame.Rect(self.PAD, bar_y, fw, 5))
        self._knob_rects[name] = (lo, hi, 'float', pygame.Rect(0, y, w, self.KNOB_H))
        return y + self.KNOB_H + 2

    def set_ctx_labels(self, labels: list) -> None:
        """Called by the station after _build_gpu_dispatch to populate context toggles.

        Each ScaleContext sphere region in the ray tracer gets an on/off toggle.
        Existing enable state is preserved; new contexts default to enabled.
        """
        existing = dict(self.sim_ctx_enabled)
        self.sim_ctx_enabled = {lbl: existing.get(lbl, True) for lbl in labels}

    def _read_knob_val(self, name: str):
        if name == 'sim_rays_per_frame': return str(self.sim_rays_per_frame)
        if name == 'sim_max_bounces':    return str(self.sim_max_bounces)
        if name == 'sim_norm_mode':      return self.sim_norm_mode
        if name == 'sim_activation':     return self.sim_activation
        if name == 'cam_auto_iso':       return 'on' if self.cam_auto_iso else 'off'
        if name == 'cam_iso':            return self.cam_iso
        if name == 'cam_target_ev':      return self.cam_target_ev
        if name.startswith('ctx_'):
            lbl = name[4:]
            return 'on' if self.sim_ctx_enabled.get(lbl, True) else 'off'
        return getattr(self, name, '')

    def _apply_knob_val(self, name: str, val) -> None:
        if name == 'sim_rays_per_frame':
            self.sim_rays_per_frame = int(val)
        elif name == 'sim_max_bounces':
            self.sim_max_bounces = int(val)
        elif name == 'sim_norm_mode':
            self.sim_norm_mode = val
        elif name == 'sim_activation':
            self.sim_activation = val
        elif name == 'cam_auto_iso':
            self.cam_auto_iso = (val == 'on')
        elif name == 'cam_iso':
            self.cam_iso = float(np.clip(float(val), 0.25, 512.0))
        elif name == 'cam_target_ev':
            self.cam_target_ev = float(np.clip(float(val), -8.0, 4.0))
        elif name.startswith('ctx_'):
            lbl = name[4:]
            if lbl in self.sim_ctx_enabled:
                self.sim_ctx_enabled[lbl] = (val == 'on')
        else:
            try:
                setattr(self, name, val)
            except AttributeError:
                pass

    def handle_event(self, ev, x_off: int = 0, y_off: int = 0,
                     panel_w: int = 230) -> tuple:
        """Returns (consumed: bool, action_key: str).

        action_key is the knob name that changed.  Station watches for
        'ctx_*' and 'sim_max_bounces' to trigger GPU scene rebuilds.
        """
        # Mousewheel: scroll panel or nudge hovered float knob
        if ev.type == pygame.MOUSEWHEEL:
            mx, _ = pygame.mouse.get_pos()
            if x_off <= mx < x_off + panel_w:
                if self._hover_knob and self._hover_knob in self._knob_rects:
                    entry = self._knob_rects[self._hover_knob]
                    if entry[2] == 'float':
                        lo, hi = entry[0], entry[1]
                        step = (hi - lo) / 200.0
                        cur  = float(self._read_knob_val(self._hover_knob))
                        self._apply_knob_val(self._hover_knob, cur + step * ev.y)
                        return True, self._hover_knob
                self._scroll = int(np.clip(
                    self._scroll - ev.y * self.ROW_H,
                    0, max(0, self._virtual_h - 400)))
                return True, ""

        # Hover tracking + drag continuation
        if ev.type == pygame.MOUSEMOTION:
            mx, my = ev.pos
            lx  = mx - x_off
            vy  = my + self._scroll
            self._hover_knob = None
            for name, entry in self._knob_rects.items():
                if entry[3].collidepoint(lx, vy):
                    self._hover_knob = name
                    break
            if self._drag_knob and self._drag_start:
                dx = mx - self._drag_start[0]
                lo, hi = self._knob_rects[self._drag_knob][:2]
                bar_px  = panel_w - 2 * self.PAD
                new_val = self._drag_val0 + dx / max(bar_px, 1) * (hi - lo)
                self._apply_knob_val(self._drag_knob, new_val)
                return True, self._drag_knob

        # End drag
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self._drag_knob:
                finished = self._drag_knob
                self._drag_knob  = None
                self._drag_start = None
                return True, finished

        # Clicks
        if ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            lx = mx - x_off
            if not (0 <= lx < panel_w):
                return False, ""
            vy = my + self._scroll
            for name, entry in self._knob_rects.items():
                lo, hi, kind, rect = entry
                if not rect.collidepoint(lx, vy):
                    continue
                if kind == 'choice':
                    choices   = lo   # lo holds the choices list
                    direction = -1 if ev.button == 3 else 1
                    cur       = self._read_knob_val(name)
                    if cur not in choices:
                        cur = choices[0]
                    idx     = choices.index(cur)
                    new_val = choices[(idx + direction) % len(choices)]
                    self._apply_knob_val(name, new_val)
                    return True, name
                if kind == 'float':
                    if ev.button == 3:
                        self._apply_knob_val(name, (lo + hi) / 2.0)
                    else:
                        self._drag_knob  = name
                        self._drag_start = (mx, my)
                        self._drag_val0  = float(self._read_knob_val(name))
                    return True, name
        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Orbit camera for a 2-D orthographic viewport
# ─────────────────────────────────────────────────────────────────────────────

class _OrthoView:
    """Pan/zoom controller for a 2-D cross-section viewport."""

    def __init__(self, scale_m: float = 0.05, offset=(0., 0.)):
        self.scale  = float(scale_m)   # metres visible per half-width
        self.offset = list(offset)     # pan in metres

    def zoom(self, factor: float) -> None:
        self.scale = float(np.clip(self.scale * factor, 1e-4, 10.0))

    def pan(self, dx_m: float, dy_m: float) -> None:
        self.offset[0] += dx_m
        self.offset[1] += dy_m

    def scale_vec(self, aspect: float) -> tuple[float, float]:
        return (self.scale, self.scale / aspect)

    def offset_vec(self) -> tuple[float, float]:
        return (float(self.offset[0]), float(self.offset[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Orthographic InvMVP builder for _MARCH_FS
# ─────────────────────────────────────────────────────────────────────────────

def _build_inv_mvp(
    view: "_OrthoView",
    bmin: "np.ndarray",
    bmax: "np.ndarray",
    axis: str,
    aspect: float,
) -> "np.ndarray":
    """Build a float32 (4,4) InvMVP for _MARCH_FS.

    Maps NDC (ndc_x, ndc_y, ndc_z, 1) → world (x, y, z, 1).
      axis 'xz'     : horizontal=Z, vertical=X, depth=Y  (looking -Y)
      axis 'yz'     : horizontal=Z, vertical=Y, depth=X  (looking -X)
      axis 'xy'/'sensor' : horizontal=X, vertical=Y, depth=Z  (looking -Z)

    Pass to glUniformMatrix4fv(..., 1, GL_TRUE, m).
    """
    m = np.zeros((4, 4), np.float32)
    m[3, 3] = 1.0
    sx = float(view.scale)            # horizontal half-range
    sy = float(view.scale / aspect)   # vertical half-range
    if axis == 'xz':
        sx_Z = sx;  sx_X = sy
        sy_Y = (float(bmax[1]) - float(bmin[1])) / 2.0 + 0.01
        oy_Y = (float(bmax[1]) + float(bmin[1])) / 2.0
        oz_Z = float(view.offset[0]);  ox_X = float(view.offset[1])
        m[0, 1] =  sx_X;  m[0, 3] = ox_X
        m[1, 2] = -sy_Y;  m[1, 3] = oy_Y
        m[2, 0] =  sx_Z;  m[2, 3] = oz_Z
    elif axis == 'yz':
        sx_Z = sx;  sx_Y = sy
        sx_X = (float(bmax[0]) - float(bmin[0])) / 2.0 + 0.01
        ox_X = (float(bmax[0]) + float(bmin[0])) / 2.0
        oz_Z = float(view.offset[0]);  oy_Y = float(view.offset[1])
        m[0, 2] = -sx_X;  m[0, 3] = ox_X
        m[1, 1] =  sx_Y;  m[1, 3] = oy_Y
        m[2, 0] =  sx_Z;  m[2, 3] = oz_Z
    else:  # 'xy' or 'sensor'
        sx_X = sx;  sx_Y = sy
        sz_Z = (float(bmax[2]) - float(bmin[2])) / 2.0 + 0.01
        oz_Z = (float(bmax[2]) + float(bmin[2])) / 2.0
        ox_X = float(view.offset[0]);  oy_Y = float(view.offset[1])
        m[0, 0] =  sx_X;  m[0, 3] = ox_X
        m[1, 1] =  sx_Y;  m[1, 3] = oy_Y
        m[2, 2] = -sz_Z;  m[2, 3] = oz_Z
    return m


# ─────────────────────────────────────────────────────────────────────────────
# CameraDesignerStation
# ─────────────────────────────────────────────────────────────────────────────

class CameraDesignerStation:
    """GL duty station for parametric camera optical design.

    Parameters
    ----------
    preset    : CameraPreset to display; defaults to simple_doublet_preset()
    win_w     : initial window width (pixels)
    win_h     : initial window height (pixels)
    """

    LEFT_W  = 230
    RIGHT_W = 230
    _CENTER_TABS  = [("views", "VIEWS"), ("lights", "LIGHTS"), ("rebuild", "REBUILD")]
    _CENTER_TAB_H = 22

    def __init__(
        self,
        preset: Optional["CameraPreset"] = None,
        win_w: int = 1280,
        win_h: int = 720,
        preview_registry: Optional[Any] = None,
        engine_graph: Optional[Any] = None,
        transport_mode: str = "ray",
        transport_contexts: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        if not _HAS_CAMERA_DESIGNER:
            raise ImportError("camera_designer package not found.")

        self.camera_manifest = None
        if preset is None:
            # The station is a frontend for the optical engine.  Its default
            # camera therefore comes from the same canonical manifest as the
            # text-render frontend, rather than from its historical doublet.
            from camera_software.camera_manifest import resolve_camera_manifest
            from camera_software.camera_designer_bridge import (
                camera_manifest_to_designer_preset,
            )
            self.camera_manifest = resolve_camera_manifest()
            preset = camera_manifest_to_designer_preset(self.camera_manifest)
        self.preset: CameraPreset = preset
        self.win_w  = win_w
        self.win_h  = win_h
        self._transport_mode = str(transport_mode).strip().lower()
        if self._transport_mode not in {"ray", "wave", "mixed"}:
            raise ValueError("transport_mode must be ray, wave, or mixed")
        self._transport_contexts = tuple(
            dict(value) for value in (transport_contexts or ())
        )

        # ── Scene state (init first so panel can hold reference) ──────────────
        self._scene_lights:  List[dict]  = []
        self._scene_meshes:  List[dict]  = []   # platonic meshes placed by user
        # A machine-readable physical witness exposes focus, mirroring, axis
        # swaps, vignetting and missing tiles more frankly than a smooth sphere.
        from camera_software.qr_optical_validator import qr_target_spec
        self._scene_object: Optional[dict] = qr_target_spec(z=0.30, width=0.15)
        self._ctx_map: Dict[str, int] = {}

        # Sensor / plate accumulation backs (software readback display)
        # Populated on demand; to_rect() yields (H, W, C) float64.
        try:
            from camera_software.camera_back import (
                FlatBack as _FlatBack,
                LargeFormatPlateBack as _LargeFormatPlateBack,
            )
            self._sensor_back = _FlatBack(res_w=128, res_h=128)
            self._plate_back  = _LargeFormatPlateBack(res_w=128, res_h=128)
        except Exception:
            self._sensor_back = None
            self._plate_back  = None

        # GL texture for the bottom-right live sensor/plate image overlay
        self._br_tex: Optional[int] = None
        # Manifold BDPT render result (set by _start_manifold_bdpt)
        self._bdpt_manifold_img: Optional[np.ndarray] = None

        # UI panels
        pygame.font.init()
        self.sensor_cfg   = SensorPlaneConfig()
        if self.camera_manifest is not None:
            _camera_mapping = self.camera_manifest.mapping()
            self.sensor_cfg.zoom = float(
                dict(_camera_mapping.get("lens", {})).get("zoom", 0.5)
            )
            self.sensor_cfg.focus_distance_m = float(
                dict(_camera_mapping.get("focus", {})).get("distance_m", 1.0)
            )
        self.sensor_cfg.sync_elements(self.preset)
        self._sync_surface_controls_from_manifest()
        self._left_panel  = _ComponentTreePanel(self.preset, self.sensor_cfg)
        self._left_panel.scene_lights = self._scene_lights  # shared reference
        self._right_panel = _ElementPropsPanel()

        # Ortho views for each cross-section pane
        self._view_xz = _OrthoView(scale_m=0.08)   # sagittal
        self._view_yz = _OrthoView(scale_m=0.08)   # tangential
        self._view_xy = _OrthoView(scale_m=0.025)  # aperture plane

        # Mouse drag state per pane
        self._drag_pane: Optional[str] = None
        self._drag_last: Optional[tuple[int,int]] = None

        # Slice / projection toggle for ortho volume renders
        # In slice mode the GPU march is clipped to a thin slab through the
        # plane of interest (y=0 for XZ, x=0 for YZ, aperture-z for XY).
        # [S] toggles mode; [ / ] adjust thickness.
        self._slice_mode:      bool  = False
        self._slice_thickness: float = 0.002   # metres (2 mm default)

        # Bake state
        self._manifold = None
        self._bake_thread = None

        # GL resources (allocated in init_gl)
        self._gl_ready   = False
        self._prog_line  = None
        self._prog_hud   = None

        self._hud_vao:  Optional[int] = None
        self._hud_vbo:  Optional[int] = None
        self._left_tex: Optional[int] = None
        self._right_tex: Optional[int] = None
        self._panels_dirty = True

        # Line geometry VAOs (one pair per visible cross-section element)
        self._line_vaos: list[tuple[int,int,int,str]] = []  # (vao, vbo, n, layer)

        # Pre-computed polylines from the preset cross-section
        self._xz_lines:  list[tuple[list, tuple]] = []   # (pts, rgba)
        self._yz_lines:  list[tuple[list, tuple]] = []
        self._xy_lines:  list[tuple[list, tuple]] = []

        # ── Glow accumulation buffers removed — GPU _MARCH_FS draws direct ──
        # _glow_field_scale is the auto-ISO exposure knob passed as uRayFieldScale
        self._glow_field_scale: float = float(_GPU_RAY_FIELD_SCALE)

        # March GL resources (allocated in init_gl)
        self._prog_march: Optional[int] = None
        self._march_vao:  Optional[int] = None
        self._march_vbo:  Optional[int] = None

        # Transport state is intentionally not retained here.  The optical
        # backend publishes immutable texture products through this registry.
        self._preview_registry = preview_registry
        self._engine_graph = engine_graph
        self._preview_request_id = ""
        self._preview_request_revision = 0
        self._preview_compositor = None
        if preview_registry is not None:
            from camera_software.gpu_preview import GLPreviewCompositor
            self._preview_compositor = GLPreviewCompositor()
        self._surface_scan_error: str = ""
        self._diagnostic_requested = False
        self._diagnostic_in_flight = False
        self._diagnostic_xz_lines: list[tuple[list, tuple]] = []
        self._diagnostic_yz_lines: list[tuple[list, tuple]] = []
        self._diagnostic_xy_lines: list[tuple[list, tuple]] = []

        # ── Auto-ISO (CameraComputer) state ──────────────────────────────────
        # _auto_cam_item: a real CameraItem built from the preset.
        # _auto_cam_gl: a SimpleNamespace acting as the GL Camera slot (‘_cam’ in
        #   CameraContext).  CameraComputer._tick_iso reads/writes cam.iso from
        #   this object.  After each tick we propagate .iso → _glow_field_scale.
        self._auto_cam_item = None   # CameraItem | None (built in _setup_auto_computer)
        self._auto_cam_gl   = None   # SimpleNamespace with .iso / .sensor_iso
        self._auto_computer = None   # CameraComputer | None
        self._auto_last_t: float = time.perf_counter()

        # Center tab bar state (follows RoutingGridView pattern in analytic_driver.py)
        self._center_tab: str = "views"
        self._center_tab_rects: list = []
        self._center_tab_tex:   Optional[int] = None
        self._center_panel_tex: Optional[int] = None
        self._lights_remove_rects: list = []
        self._view_layers = {
            "texture": True,
            "pictographic": True,
        }

    def set_view_layer_config(self, *, texture: Optional[bool] = None,
                              pictographic: Optional[bool] = None) -> None:
        """Configure layered rendering in viewports.

        ``texture`` toggles the volumetric texture layer.
        ``pictographic`` toggles line/pictograph overlays.
        """
        if texture is not None:
            self._view_layers["texture"] = bool(texture)
        if pictographic is not None:
            self._view_layers["pictographic"] = bool(pictographic)

    # ── Parts catalog ─────────────────────────────────────────────────────────

    @staticmethod
    def parts_catalog_items() -> list:
        """Return filesystem-backed camera parts catalog entries for the left panel.

        Scans the YAML preset library folders and returns a flat list of dicts::

            {"id": "thin_lenses:planar_50mm", "label": "planar_50mm",
             "category": "thin_lenses", "path": "/abs/path/to/file.yaml"}

        This belongs here — not on the player camera panel — because this
        station is where parts are placed, moved, and deleted.
        """
        root = os.path.dirname(os.path.abspath(__file__))
        buckets = [
            ("thin_lenses", os.path.join(root, "configs", "lenses"),     "*.yaml"),
            ("bodies",      os.path.join(root, "configs", "cameras"),    "*.yaml"),
            ("tubes",       os.path.join(root, "configs", "tubes"),      "*.yaml"),
            ("port_holes",  os.path.join(root, "configs", "ports"),      "*.yaml"),
            ("lights",      os.path.join(root, "configs", "lights"),     "*.yaml"),
            ("materials",   os.path.join(root, "configs", "materials"),  "*.yaml"),
            ("sensors",     os.path.join(root, "configs", "sensors"),    "*.yaml"),
            ("film",        os.path.join(root, "configs", "films"),      "*.yaml"),
            ("film",        os.path.join(root, "configs", "film_types"), "*.yaml"),
            ("lights",      os.path.join(root, "presets", "emitters"),   "*.yaml"),
        ]
        items: list = []
        for cat, base, pat in buckets:
            if not os.path.isdir(base):
                continue
            for fp in sorted(glob.glob(os.path.join(base, pat))):
                stem = os.path.splitext(os.path.basename(fp))[0]
                items.append({
                    "id":       f"{cat}:{stem}",
                    "label":    stem,
                    "category": cat,
                    "path":     fp,
                })
        return items

    # ── Geometry builder ──────────────────────────────────────────────────────

    def _rebuild_lines(self) -> None:
        """Sample parametric surfaces and build 2-D polyline lists."""
        self._xz_lines.clear()
        self._yz_lines.clear()
        self._xy_lines.clear()

        N = 128  # samples per surface outline

        # ── Lens elements (XZ and YZ profiles) ────────────────────────────
        for idx, el in enumerate(self.preset.lens_group.elements):
            surf = el.surface
            col  = _EL_COLORS[idx % len(_EL_COLORS)]
            r_max = surf.r_max
            from camera_designer.scene_builder import _element_surface_frame
            element_rotation, element_translation = _element_surface_frame(el)

            # XZ profile: vary r in [-r_max, r_max], trace in XZ plane
            xz_pts = []
            for r in np.linspace(-r_max, r_max, N):
                ro = np.array([r, 0., -1.0], np.float64)
                rd = np.array([0., 0.,  1.0], np.float64)
                ro_local = ro.copy(); ro_local[2] -= el.z_vertex
                t, hit, _ = surf.intersect(ro_local, rd)
                if math.isfinite(t):
                    world = element_rotation @ hit + element_translation
                    xz_pts.append((float(world[2]), float(world[0])))
            if xz_pts:
                self._xz_lines.append((xz_pts, col))

            # YZ profile: same but in Y direction
            yz_pts = []
            for r in np.linspace(-r_max, r_max, N):
                ro = np.array([0., r, -1.0], np.float64)
                rd = np.array([0., 0.,  1.0], np.float64)
                ro_local = ro.copy(); ro_local[2] -= el.z_vertex
                t, hit, _ = surf.intersect(ro_local, rd)
                if math.isfinite(t):
                    world = element_rotation @ hit + element_translation
                    yz_pts.append((float(world[2]), float(world[1])))
            if yz_pts:
                self._yz_lines.append((yz_pts, col))

            # XY aperture cross (circle at z=aperture plane)
            if r_max > 0:
                angs = np.linspace(0., 2*math.pi, 64)
                xy_pts = []
                for angle in angs:
                    local = np.array([
                        r_max * math.cos(angle),
                        r_max * math.sin(angle),
                        0.0,
                    ], np.float64)
                    world = element_rotation @ local + element_translation
                    xy_pts.append((float(world[0]), float(world[1])))
                self._xy_lines.append((xy_pts, (col[0]*0.7, col[1]*0.7, col[2]*0.7, 0.6)))

        # ── Aperture stop ─────────────────────────────────────────────────
        ap  = self.preset.aperture_stop
        angs = np.linspace(0., 2*math.pi, 64)
        # XZ: horizontal bar at z=ap.z_pos between +/-r_outer
        self._xz_lines.append(([(ap.z_pos, -ap.r_outer), (ap.z_pos, ap.r_outer)],
                                (1., 1., 0., 0.8)))
        self._yz_lines.append(([(ap.z_pos, -ap.r_outer), (ap.z_pos, ap.r_outer)],
                                (1., 1., 0., 0.8)))
        # XY: aperture circle
        xy_ap = [(ap.r_outer*math.cos(a), ap.r_outer*math.sin(a)) for a in angs]
        self._xy_lines.append((xy_ap, (1., 1., 0., 0.9)))

        # ── Sensor plane — live position from sensor_cfg ─────────────────
        sn    = self.preset.sensor
        cfg   = getattr(self, 'sensor_cfg', None)
        s_dz  = float(cfg.sensor_z)  if cfg else 0.0
        s_dx  = float(cfg.sensor_x)  if cfg else 0.0
        s_dy  = float(cfg.sensor_y)  if cfg else 0.0
        p_dz  = float(cfg.plate_z)   if cfg else 0.010
        sn_z  = sn.z_pos + s_dz
        sn_r  = sn.r_max
        pl_z  = sn_z - p_dz   # plate sits behind (lower z) the sensor

        # Sensor: red bar in XZ/YZ, red circle in XY
        self._xz_lines.append(([(sn_z + s_dx, -sn_r), (sn_z + s_dx, sn_r)],
                                (0.85, 0.25, 0.25, 0.95)))
        self._yz_lines.append(([(sn_z + s_dy, -sn_r), (sn_z + s_dy, sn_r)],
                                (0.85, 0.25, 0.25, 0.95)))
        xy_sn = [(sn_r * math.cos(a) + s_dx, sn_r * math.sin(a) + s_dy)
                 for a in np.linspace(0., 2 * math.pi, 64)]
        self._xy_lines.append((xy_sn, (0.85, 0.25, 0.25, 0.75)))

        # Plate: cyan dashed bar (drawn as two segments to mimic dashes)
        pl_r = sn_r * 1.3
        for seg_lo, seg_hi in [(-pl_r, -pl_r * 0.1), (pl_r * 0.1, pl_r)]:
            self._xz_lines.append(([(pl_z, seg_lo), (pl_z, seg_hi)],
                                    (0.2, 0.8, 0.85, 0.7)))
            self._yz_lines.append(([(pl_z, seg_lo), (pl_z, seg_hi)],
                                    (0.2, 0.8, 0.85, 0.7)))
        xy_pl = [(pl_r * math.cos(a), pl_r * math.sin(a))
                 for a in np.linspace(0., 2 * math.pi, 64)]
        self._xy_lines.append((xy_pl, (0.2, 0.8, 0.85, 0.55)))

        # ── Optical axis ──────────────────────────────────────────────────
        z0 = self.preset.mount_ring.z_flange + 0.080
        z1 = pl_z - 0.005
        axis_col = (0.4, 0.4, 0.4, 0.5)
        self._xz_lines.append(([(z0, 0.), (z1, 0.)], axis_col))
        self._yz_lines.append(([(z0, 0.), (z1, 0.)], axis_col))

        # ── Scene lights (markers) ────────────────────────────────────────
        light_col = (1.0, 0.9, 0.3, 1.0)
        d = 0.008
        for lt in getattr(self, '_scene_lights', []):
            lp = lt.get('pos', [0., 0., 5.0])
            lx, ly, lz = float(lp[0]), float(lp[1]), float(lp[2])
            self._xz_lines.append(([(lz, lx - d), (lz, lx + d)], light_col))
            self._xz_lines.append(([(lz - d, lx), (lz + d, lx)], light_col))
            self._yz_lines.append(([(lz, ly - d), (lz, ly + d)], light_col))
            self._yz_lines.append(([(lz - d, ly), (lz + d, ly)], light_col))
            self._xy_lines.append(([(lx - d, ly), (lx + d, ly)], light_col))
            self._xy_lines.append(([(lx, ly - d), (lx, ly + d)], light_col))

        # ── Placed meshes (diamond cross-hair markers) ────────────────────
        mesh_col = (0.7, 0.5, 1.0, 0.85)
        for msh in getattr(self, '_scene_meshes', []):
            mp = msh.get('pos', [0., 0., 0.])
            mx, my, mz = float(mp[0]), float(mp[1]), float(mp[2])
            r = float(msh.get('radius', 0.02))
            for ang_pair in [(0, math.pi / 2), (math.pi / 4, 3 * math.pi / 4)]:
                a0, a1 = ang_pair
                self._xz_lines.append(([(mz + r * math.cos(a0), mx + r * math.sin(a0)),
                                         (mz + r * math.cos(a1), mx + r * math.sin(a1))],
                                        mesh_col))
                self._xy_lines.append(([(mx + r * math.cos(a0), my + r * math.sin(a0)),
                                         (mx + r * math.cos(a1), my + r * math.sin(a1))],
                                        mesh_col))

        # ── Mount ring ────────────────────────────────────────────────────
        mr = self.preset.mount_ring
        self._xz_lines.append((
            [(mr.z_flange, -mr.r_outer), (mr.z_flange, mr.r_outer)],
            (0.5, 0.5, 0.6, 0.6)
        ))

        # NOTE: Camera body box and lens barrel outlines have been removed.
        # The GPU volumetric march (_draw_march_volume) provides real cross-
        # section slices of the actual ray-traced scene; lens element curves
        # (above) are drawn on top as physical-element overlays.  The housing
        # was purely an illustrative analogy and is no longer needed.
        if hasattr(self, "_diagnostic_requested"):
            self._diagnostic_requested = True
            self._right_panel.diagnostic_state = "waiting for scan"

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def init_gl(self) -> None:
        if not _HAS_GL:
            return
        self._prog_line = _gl_shaders.compileProgram(
            _gl_shaders.compileShader(_LINE2D_VS, GL_VERTEX_SHADER),
            _gl_shaders.compileShader(_LINE2D_FS, GL_FRAGMENT_SHADER),
        )
        self._prog_hud = _gl_shaders.compileProgram(
            _gl_shaders.compileShader(_HUD_VS, GL_VERTEX_SHADER),
            _gl_shaders.compileShader(_HUD_FS, GL_FRAGMENT_SHADER),
        )

        # HUD quad VAO (shared for both panels)
        self._hud_vao = glGenVertexArrays(1)
        self._hud_vbo = glGenBuffers(1)
        glBindVertexArray(self._hud_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._hud_vbo)
        glBufferData(GL_ARRAY_BUFFER, 96, None, GL_STREAM_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)

        # Allocate reusable line VAO/VBOs
        # We'll use a single big VAO per pane and re-upload each frame
        self._xz_vao, self._xz_vbo = _make_vao2()
        self._yz_vao, self._yz_vbo = _make_vao2()
        self._xy_vao, self._xy_vbo = _make_vao2()

        # Volume march program + NDC quad (for _MARCH_FS ray-marching)
        if _HAS_GPU_FIELD and _MARCH_VS is not None and _MARCH_FS is not None:
            self._prog_march = _gl_shaders.compileProgram(
                _gl_shaders.compileShader(_MARCH_VS, GL_VERTEX_SHADER),
                _gl_shaders.compileShader(_MARCH_FS, GL_FRAGMENT_SHADER),
            )
        ndc_quad = np.array([-1, -1,  1, -1,  1,  1, -1,  1], np.float32)
        self._march_vao = glGenVertexArrays(1)
        self._march_vbo = glGenBuffers(1)
        glBindVertexArray(self._march_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._march_vbo)
        glBufferData(GL_ARRAY_BUFFER, ndc_quad.nbytes, ndc_quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glBindVertexArray(0)

        self._rebuild_lines()
        self._gl_ready = True
        self._request_optical_preview()

    def destroy_gl(self) -> None:
        if not _HAS_GL or not self._gl_ready:
            return
        if self._preview_compositor is not None:
            self._preview_compositor.destroy()
            self._preview_compositor = None
        vaos = [v for v in (self._hud_vao, self._xz_vao, self._yz_vao,
                             self._xy_vao, self._march_vao) if v is not None]
        vbos = [v for v in (self._hud_vbo, self._xz_vbo, self._yz_vbo,
                             self._xy_vbo, self._march_vbo) if v is not None]
        texs = [t for t in (self._left_tex, self._right_tex, self._br_tex)
                if t is not None]
        for vao in vaos: glDeleteVertexArrays(1, [vao])
        for vbo in vbos: glDeleteBuffers(1, [vbo])
        for tex in texs: glDeleteTextures(1, [tex])
        self._gl_ready = False

    # ── Tracer & trickle management ───────────────────────────────────────────

    def _compute_confinement_box(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Derive the simulation confinement box from the current camera preset.

        Z extent covers light sources → sensor with 5 cm padding.
        XY extent is 2.5× the widest aperture radius, minimum 5 cm each side.
        Returns (bmin, bmax) as float32 (3,) arrays, or None if no preset.
        """
        if self.preset is None:
            return None
        preset = self.preset

        z_vals: List[float] = [0.0]
        for el in preset.lens_group.elements:
            z_vals.append(float(getattr(el.surface, 'z_pos',
                                        getattr(el.surface, 'z_vertex', 0.0))))
        if preset.sensor is not None:
            z_vals.append(float(getattr(preset.sensor, 'z_pos',
                                        getattr(preset.sensor, 'z_vertex', 0.0))))
        if preset.aperture_stop is not None:
            z_vals.append(float(getattr(preset.aperture_stop, 'z_pos', 0.0)))
        for lgt in self._scene_lights:
            z_vals.append(float(np.asarray(lgt.get('pos', [0, 0, 0]))[2]))

        for msh in self._scene_meshes:
            mp = msh.get('pos', [0, 0, 0])
            z_vals.append(float(np.asarray(mp)[2]))

        # 30 cm staging front, 10 cm back clearance — gives room for lights
        z_front = max(z_vals) + 0.30
        z_back  = min(z_vals) - 0.10

        r_max = 0.025
        for el in preset.lens_group.elements:
            r_max = max(r_max, float(getattr(el.surface, 'r_max', 0.025)))
        if preset.aperture_stop is not None:
            r_max = max(r_max, float(getattr(preset.aperture_stop, 'r_outer', r_max)))
        xy_half = max(r_max * 4.0, 0.15)

        bmin = np.array([-xy_half, -xy_half, z_back],  np.float32)
        bmax = np.array([ xy_half,  xy_half, z_front], np.float32)
        return bmin, bmax

    def _build_gpu_dispatch(self) -> None:
        """Invalidate the optical backend; retained as a UI call-site shim."""
        self._request_optical_preview()

    def _surface_scan_pose(self):
        from camera_software.surface_scan_preview import SurfaceScanPose

        sensor = self.preset.sensor
        aperture = self.preset.aperture_stop
        cfg = self.sensor_cfg
        return SurfaceScanPose(
            sensor_center=(
                float(cfg.sensor_x), float(cfg.sensor_y),
                float(sensor.z_pos + cfg.sensor_z),
            ),
            aperture_center=(0.0, 0.0, float(aperture.z_pos)),
            sensor_half_width=float(sensor.r_max),
            sensor_half_height=float(sensor.r_max),
            resolution=256,
        )

    def _request_optical_preview(self) -> bool:
        """Submit authored state to the optical engine's fast transport mode.

        This method does not build, configure, poll, or retain a tracer.  The
        render graph and its optical backend own the complete job lifetime.
        """
        if (
            getattr(self, "_engine_graph", None) is None
            or getattr(self, "camera_manifest", None) is None
        ):
            self._surface_scan_error = "optical backend unavailable"
            self._right_panel.diagnostic_state = "backend unavailable"
            return False
        try:
            from pluck_render_graph import EngineRequest
            from camera_software.optical_bench import (
                OpticalTransportMode,
                camera_bell_jar_scene_manifest,
            )

            pose = self._surface_scan_pose().validated()
            self._preview_request_revision += 1
            requested_products = [
                "surface_scan", "camera_geometry", "light_field",
                "transport_diagnostics",
            ]
            if self._right_panel.bottom_right_view == "wave_arena":
                requested_products.append("wave_arena")
            payload = {
                "bench_id": "camera-designer",
                "scene_manifest": camera_bell_jar_scene_manifest(
                    self.camera_manifest
                ),
                "scene_object": dict(self._scene_object or {}),
                "lights": [dict(light) for light in self._scene_lights],
                "transport_mode": OpticalTransportMode(
                    self._transport_mode
                ).value,
                "contexts": [
                    dict(context) for context in self._transport_contexts
                ],
                "backend_mode": "fast_preview",
                "execution_policy": "interactive_preview",
                "sensor_pose": {
                    "sensor_center": list(pose.sensor_center),
                    "aperture_center": list(pose.aperture_center),
                    "sensor_half_width": float(pose.sensor_half_width),
                    "sensor_half_height": float(pose.sensor_half_height),
                    "resolution": int(pose.resolution),
                },
                "revision": self._preview_request_revision,
                "requested_products": requested_products,
            }
            request = EngineRequest.create(
                "optical",
                payload,
                scene_revision=self.camera_manifest.hash,
                requested_products=payload["requested_products"],
                priority=10,
            )
            accepted = bool(self._engine_graph.submit(request))
            if not accepted:
                raise RuntimeError("optical render graph rejected preview request")
            self._preview_request_id = request.request_id
            self._surface_scan_error = ""
            self._right_panel.diagnostic_state = "backend queued"
            return True
        except Exception as exc:
            self._surface_scan_error = str(exc)
            self._right_panel.diagnostic_state = "backend error"
            print(
                f"[camera_designer] optical preview request failed: {exc}",
                flush=True,
            )
            return False

    def _build_surface_scan(self) -> None:
        """Request the backend-owned fast transport mode."""
        self._request_optical_preview()

    def configure_transport(
        self,
        mode: str,
        contexts: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Adopt graph-authored transport without constructing station physics."""

        resolved = str(mode).strip().lower()
        if resolved not in {"ray", "wave", "mixed"}:
            raise ValueError("transport mode must be ray, wave, or mixed")
        values = tuple(dict(value) for value in contexts)
        if resolved == "ray" and any(
            str(value.get("transport", "ray")).strip().lower() == "wave"
            for value in values
        ):
            raise ValueError("ray transport cannot install wave contexts")
        self._transport_mode = resolved
        self._transport_contexts = values
        self._request_optical_preview()

    def _poll_surface_scan(self) -> dict:
        if self._preview_registry is None:
            return {}
        product = self._preview_registry.get("camera.surface-scan")
        if product is None:
            return {}
        return {
            "texture_id": int(product.texture_id),
            "width": int(product.width),
            "height": int(product.height),
            "generation": int(product.generation),
            "producer": str(product.producer),
            "metadata": dict(product.metadata),
        }

    def bind_preview_registry(self, registry: Any) -> None:
        """Bind the host-owned product registry used only for presentation."""

        self._preview_registry = registry
        if self._preview_compositor is None:
            from camera_software.gpu_preview import GLPreviewCompositor
            self._preview_compositor = GLPreviewCompositor()

    def _gpu_pump_frame(self) -> None:
        """Transport execution belongs to the optical backend."""
        return None

    def _clear_glow(self) -> None:
        """Invalidate backend products after an authoring change."""
        self._request_optical_preview()

    # ── Per-frame render ──────────────────────────────────────────────────────

    def _render_center_tab_bar(self, vp_w: int) -> "pygame.Surface":
        """Render the center tab bar strip — exact RoutingGridView pattern."""
        TAB_H  = self._CENTER_TAB_H
        surf   = pygame.Surface((vp_w, TAB_H))
        surf.fill((14, 14, 18))
        pygame.font.init()
        font      = pygame.font.SysFont("consolas", 11)
        fh        = font.get_height()
        tab_keys  = [t[0] for t in self._CENTER_TABS]
        tab_labels = [t[1] for t in self._CENTER_TABS]
        tab_w     = 72
        tab_pad   = 4
        self._center_tab_rects = []
        for ti, (tk, tl) in enumerate(zip(tab_keys, tab_labels)):
            tr = pygame.Rect(tab_pad + ti * (tab_w + 2), tab_pad,
                             tab_w, TAB_H - 2 * tab_pad)
            self._center_tab_rects.append(tr)
            active = (self._center_tab == tk)
            bg = (40, 90, 160) if active else (28, 28, 38)
            pygame.draw.rect(surf, bg, tr, border_radius=3)
            pygame.draw.rect(surf, (60, 60, 85), tr, 1, border_radius=3)
            tc = (220, 235, 255) if active else (100, 100, 120)
            ts = font.render(tl, True, tc)
            surf.blit(ts, (tr.x + (tr.w - ts.get_width()) // 2,
                           tr.y + (tr.h - fh) // 2))
        return surf

    def _render_lights_panel(self, vp_w: int, vp_h: int) -> "pygame.Surface":
        """Render the scene lights list panel for the LIGHTS center tab."""
        TAB_H  = self._CENTER_TAB_H
        surf   = pygame.Surface((vp_w, vp_h))
        surf.fill((14, 14, 18))
        pygame.font.init()
        font  = pygame.font.SysFont("consolas", 11)
        fh    = font.get_height()
        ROW_H = fh + 10
        PAD   = 8
        y     = TAB_H + PAD
        hdr   = font.render(
            "Scene Lights  (right-click in VIEWS to place)", True, (160, 180, 220))
        surf.blit(hdr, (PAD, y))
        y += fh + 6
        self._lights_remove_rects = []
        if not self._scene_lights:
            surf.blit(
                font.render("(no lights placed)", True, (60, 60, 80)), (PAD, y))
        else:
            for i, lt in enumerate(self._scene_lights):
                if y + ROW_H > vp_h:
                    break
                pos  = lt.get("pos",   [0., 0., 0.])
                pwr  = lt.get("power", 1.0)
                col3 = lt.get("color", [1., 1., 1.])
                row_r = pygame.Rect(PAD, y, vp_w - 2 * PAD, ROW_H)
                pygame.draw.rect(surf, (22, 22, 30), row_r, border_radius=3)
                col_dot = (int(col3[0] * 200 + 55),
                           int(col3[1] * 200 + 55),
                           int(col3[2] * 200 + 55))
                pygame.draw.circle(surf, col_dot,
                                   (PAD + ROW_H // 2, y + ROW_H // 2), 5)
                lbl = font.render(
                    f"#{i}  x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}"
                    f"  pwr={pwr:.2f}",
                    True, (190, 195, 210))
                surf.blit(lbl, (PAD + 16, y + (ROW_H - fh) // 2))
                btn_w = 40
                btn_r = pygame.Rect(
                    vp_w - PAD - btn_w, y + 2, btn_w, ROW_H - 4)
                pygame.draw.rect(surf, (80, 28, 28), btn_r, border_radius=3)
                surf.blit(
                    font.render("del", True, (220, 100, 100)),
                    (btn_r.x + (btn_w - font.size("del")[0]) // 2,
                     btn_r.y + (btn_r.h - fh) // 2))
                self._lights_remove_rects.append(btn_r)
                y += ROW_H + 4
        return surf

    def _render_rebuild_panel(self, vp_w: int, vp_h: int) -> "pygame.Surface":
        """Render the rebuild / GPU scene status panel for the REBUILD center tab."""
        TAB_H = self._CENTER_TAB_H
        surf  = pygame.Surface((vp_w, vp_h))
        surf.fill((14, 14, 18))
        pygame.font.init()
        font  = pygame.font.SysFont("consolas", 11)
        fh    = font.get_height()
        PAD   = 8
        y     = TAB_H + PAD
        hdr   = font.render("Rebuild / GPU Scene Controls", True, (160, 180, 220))
        surf.blit(hdr, (PAD, y))
        y += fh + 8
        rp    = self._right_panel
        rows  = [
            ("Rays / frame",  getattr(rp, "sim_rays_per_frame", 0)),
            ("Max bounces",   getattr(rp, "sim_max_bounces",    4)),
            ("Norm mode",     getattr(rp, "sim_norm_mode",      "none")),
            ("Activation",    getattr(rp, "sim_activation",     "log")),
            ("Slice mode",    "ON" if self._slice_mode else "off"),
            ("Slice thick",   f"{self._slice_thickness * 1e3:.2f} mm"),
            ("Scene lights",  len(self._scene_lights)),
            ("Scene meshes",  len(self._scene_meshes)),
            ("Backend request", self._preview_request_id[:12] or "unavailable"),
            ("Preview generation", self._right_panel.surface_scan_generation),
        ]
        for label, value in rows:
            if y + fh > vp_h:
                break
            surf.blit(
                font.render(f"{label:<18}  {value}", True, (190, 195, 210)),
                (PAD, y))
            y += fh + 4
        y += 8
        if y + fh <= vp_h:
            surf.blit(
                font.render(
                    "[SPACE] rebuild GPU dispatch   "
                    "[S] toggle slice   [ / ] slab thickness",
                    True, (60, 65, 85)),
                (PAD, y))
        return surf

    def draw(self, win_w: int, win_h: int) -> None:
        """Render the full camera designer UI into the current GL context."""
        if not _HAS_GL or not self._gl_ready:
            return
        self.win_w, self.win_h = win_w, win_h
        lw = self.LEFT_W
        rw = self.RIGHT_W
        vp_x = lw
        vp_w = win_w - lw - rw
        vp_h = win_h
        surface_scan_info = self._poll_surface_scan()
        self._right_panel.surface_scan_generation = int(
            surface_scan_info.get("generation", 0)
        )

        # ── Auto-ISO tick — CameraComputer owns all auto logic ───────────────────────
        if self._auto_computer is not None:
            _rp = self._right_panel
            # Push panel cam settings into the auto-computer before each tick
            try:
                self._auto_computer.auto_iso  = _rp.cam_auto_iso
                self._auto_computer.target_ev = _rp.cam_target_ev
            except AttributeError:
                pass
            if not _rp.cam_auto_iso and self._auto_cam_gl is not None:
                self._auto_cam_gl.iso = float(np.clip(_rp.cam_iso, 0.25, 512.0))
            now = time.perf_counter()
            dt_auto = float(now - self._auto_last_t)
            self._auto_last_t = now
            try:
                from camera_software import CameraContext as _CameraContext
                ctx = _CameraContext(self._auto_cam_item, self._auto_cam_gl)
            except Exception:
                import types as _types
                ctx = _types.SimpleNamespace(
                    camera=self._auto_cam_gl, last_ev=None,
                    focus_score=None, depth_map=None)
            self._auto_computer.tick(dt_auto, ctx)
            self._glow_field_scale = float(
                np.clip(self._auto_cam_gl.iso, 0.25, 512.0))
            # Reflect computed ISO back into the right panel
            _rp.cam_iso = self._glow_field_scale

        glClearColor(0.03, 0.04, 0.08, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Center tab: VIEWS → GL quadrant viewports; LIGHTS/REBUILD → pygame panel
        half_vp_w = vp_w // 2
        half_vp_h = vp_h // 2

        if self._center_tab == "views":
            _sm = self._slice_mode
            _sb = (1.0, 0.65, 0.1, 0.85) if _sm else None   # amber border = slice mode

            # Top-left: XZ sagittal
            self._draw_view_layers(
                vp_x,             half_vp_h,
                half_vp_w,        half_vp_h,
                self._xz_lines,   self._view_xz,
                win_w, win_h,
                axis="xz",
                title="XZ sagittal",
                border_rgba=_sb,
            )

            # Top-right: YZ tangential
            self._draw_view_layers(
                vp_x + half_vp_w, half_vp_h,
                half_vp_w,        half_vp_h,
                self._yz_lines,   self._view_yz,
                win_w, win_h,
                axis="yz",
                title="YZ tangential",
                border_rgba=_sb,
            )

            # Bottom-left: XY aperture plane
            self._draw_view_layers(
                vp_x,             0,
                half_vp_w,        half_vp_h,
                self._xy_lines,   self._view_xy,
                win_w, win_h,
                axis="xy",
                title="XY aperture plane",
                border_rgba=_sb,
            )

            # Bottom-right: controlled by right-panel bottom_right_view knob
            _br_view = self._right_panel.bottom_right_view
            _br_title = {'gpu_field': 'Sensor plane (GPU)',
                         'sensor':    'Sensor accumulation',
                         'plate':     'Plate accumulation',
                         'wave_arena': 'Wave arena phase / amplitude',
                         'complex_transport': 'Complex ray transport',
                         'surface_scan': 'Pixel-ray camera preview'}.get(_br_view, _br_view)
            if _br_view == 'surface_scan':
                self._draw_cross_section(
                    vp_x + half_vp_w, 0,
                    half_vp_w, half_vp_h,
                    self._xy_lines, self._view_xy,
                    win_w, win_h,
                    title=_br_title,
                    border_rgba=_sb,
                )
                scan_tex = int(surface_scan_info.get("texture_id", 0))
                if scan_tex > 0:
                    screen_y = win_h - half_vp_h
                    product = (
                        None if self._preview_registry is None
                        else self._preview_registry.get("camera.surface-scan")
                    )
                    if self._preview_compositor is not None and product is not None:
                        self._preview_compositor.draw(
                            product,
                            (vp_x + half_vp_w, screen_y, half_vp_w, half_vp_h),
                            win_h,
                        )
                    else:
                        self._blit_panel(
                            vp_x + half_vp_w, screen_y,
                            half_vp_w, half_vp_h,
                            scan_tex, win_w, win_h,
                        )
            elif _br_view in {'wave_arena', 'complex_transport'}:
                self._draw_cross_section(
                    vp_x + half_vp_w, 0,
                    half_vp_w, half_vp_h,
                    self._xy_lines, self._view_xy,
                    win_w, win_h,
                    title=_br_title,
                    border_rgba=_sb,
                )
                product = (
                    None if self._preview_registry is None
                    else self._preview_registry.get(
                        "wave.arena-state" if _br_view == "wave_arena"
                        else "transport.complex-accumulation"
                    )
                )
                if self._preview_compositor is not None and product is not None:
                    self._preview_compositor.draw(
                        product,
                        (
                            vp_x + half_vp_w,
                            win_h - half_vp_h,
                            half_vp_w,
                            half_vp_h,
                        ),
                        win_h,
                    )
            elif _br_view == 'gpu_field':
                self._draw_view_layers(
                    vp_x + half_vp_w, 0,
                    half_vp_w,        half_vp_h,
                    self._xy_lines,   self._view_xy,
                    win_w, win_h,
                    axis="sensor",
                    title=_br_title,
                    border_rgba=_sb,
                )
            elif _br_view == 'sensor':
                self._draw_cross_section(
                    vp_x + half_vp_w, 0,
                    half_vp_w,        half_vp_h,
                    self._xy_lines,   self._view_xy,
                    win_w, win_h,
                    title=_br_title,
                    border_rgba=_sb,
                )
                self._draw_back_image(vp_x + half_vp_w, 0,
                                      half_vp_w, half_vp_h, self._sensor_back)
            elif _br_view == 'plate':
                self._draw_cross_section(
                    vp_x + half_vp_w, 0,
                    half_vp_w,        half_vp_h,
                    self._xy_lines,   self._view_xy,
                    win_w, win_h,
                    title=_br_title,
                    border_rgba=_sb,
                )
                self._draw_back_image(vp_x + half_vp_w, 0,
                                      half_vp_w, half_vp_h, self._plate_back)
            elif _br_view == 'manifold_bdpt':
                img = getattr(self, '_bdpt_manifold_img', None)
                self._draw_cross_section(
                    vp_x + half_vp_w, 0,
                    half_vp_w,        half_vp_h,
                    self._xy_lines,   self._view_xy,
                    win_w, win_h,
                    title=_br_title,
                    border_rgba=_sb,
                )
                if img is not None:
                    self._draw_back_image(vp_x + half_vp_w, 0,
                                          half_vp_w, half_vp_h,
                                          _NumpyBack(img))
        else:
            # Non-VIEWS tab: fill the center area with a pygame content panel
            glDisable(GL_SCISSOR_TEST)
            glViewport(0, 0, win_w, win_h)
            if self._center_tab == "lights":
                csurf = self._render_lights_panel(vp_w, win_h)
            else:
                csurf = self._render_rebuild_panel(vp_w, win_h)
            if self._center_panel_tex is None:
                self._center_panel_tex = _make_tex(csurf)
            else:
                _update_tex(self._center_panel_tex, csurf)
            self._blit_panel(vp_x, 0, vp_w, win_h,
                             self._center_panel_tex, win_w, win_h)

        # HUD panels
        glDisable(GL_SCISSOR_TEST)
        glViewport(0, 0, win_w, win_h)

        if self._panels_dirty or self._left_tex is None:
            # Propagate slice-mode state to the left panel before upload
            self._left_panel._slice_mode_label = (self._slice_mode, self._slice_thickness)
            self._upload_panels(win_w, win_h, lw, rw)
            self._panels_dirty = False

        self._blit_panel(0,          0, lw,  win_h, self._left_tex,  win_w, win_h)
        self._blit_panel(win_w - rw, 0, rw,  win_h, self._right_tex, win_w, win_h)

        # Center tab bar overlay — always on top of the center area
        tab_surf = self._render_center_tab_bar(vp_w)
        if self._center_tab_tex is None:
            self._center_tab_tex = _make_tex(tab_surf)
        else:
            _update_tex(self._center_tab_tex, tab_surf)
        self._blit_panel(vp_x, 0, vp_w, self._CENTER_TAB_H,
                         self._center_tab_tex, win_w, win_h)

    # ── Internal draw helpers ─────────────────────────────────────────────────

    def render(self) -> None:
        """Pluck full-screen station contract.

        The designer's native drawing API accepts explicit dimensions because
        it is also usable as an embedded viewport.  Pluck owns the full window
        and invokes stations through a no-argument ``render()`` method, so keep
        that host adapter here rather than making either side know the other's
        implementation detail.
        """

        self.draw(self.win_w, self.win_h)

    def _draw_view_layers(
        self,
        vp_x: int, vp_y: int, vp_w: int, vp_h: int,
        lines: list,
        view: _OrthoView,
        win_w: int, win_h: int,
        axis: str,
        title: str,
        border_rgba: Optional[tuple] = None,
    ) -> None:
        """Compose viewport layers: texture background, then pictographic lines."""
        if self._view_layers.get("texture", True):
            self._draw_march_volume(vp_x, vp_y, vp_w, vp_h, axis)
        if self._view_layers.get("pictographic", True):
            diagnostic = {
                "xz": self._diagnostic_xz_lines,
                "yz": self._diagnostic_yz_lines,
                "xy": self._diagnostic_xy_lines,
            }.get(axis, ())
            self._draw_cross_section(
                vp_x, vp_y, vp_w, vp_h,
                [*lines, *diagnostic], view, win_w, win_h,
                title=title,
                border_rgba=border_rgba,
            )

    def _draw_march_volume(
        self,
        vp_x: int, vp_y: int, vp_w: int, vp_h: int,
        axis: str,
    ) -> None:
        """Draw volumetric ray-march overlay for one viewport using _MARCH_FS.

        axis: 'xz' | 'yz' | 'xy' | 'sensor'
        Additive blend (GL_ONE, GL_ONE) — same as demo_pluck_gl.
        """
        # Volumetric products are composed from the backend registry in their
        # own tabs; the station has no local field texture or dispatch state.
        if getattr(self, "_gpu_state", None) is None or self._prog_march is None:
            return
        state      = self._gpu_state
        bmin       = np.asarray(state['bmin'], np.float32)
        bmax       = np.asarray(state['bmax'], np.float32)

        # In slice mode restrict the march to a thin slab through the plane of
        # interest: y=0 for XZ, x=0 for YZ, aperture-z for XY & sensor.
        if self._slice_mode and axis != 'sensor':
            t = self._slice_thickness
            bmin = bmin.copy();  bmax = bmax.copy()
            if axis == 'xz':
                bmin[1] = max(float(bmin[1]), -t)
                bmax[1] = min(float(bmax[1]),  t)
            elif axis == 'yz':
                bmin[0] = max(float(bmin[0]), -t)
                bmax[0] = min(float(bmax[0]),  t)
            elif axis == 'xy':
                ap_z = float(self.preset.mount_ring.aperture_plane_z)
                bmin[2] = max(float(bmin[2]), ap_z - t)
                bmax[2] = min(float(bmax[2]), ap_z + t)
        tex_bands  = state['tex_bands']
        layer_specs = state.get('layer_specs', [])
        n_bands    = max(1, min(len(tex_bands), 8))

        view_map = {'xz': self._view_xz, 'yz': self._view_yz,
                    'xy': self._view_xy, 'sensor': self._view_xy}
        view   = view_map.get(axis, self._view_xz)
        aspect = vp_w / max(1, vp_h)

        inv_mvp = _build_inv_mvp(view, bmin, bmax, axis, aspect)

        z_clip_min = float(bmin[2])
        z_clip_max = float(bmax[2])
        if axis == 'sensor':
            snr = getattr(self.preset, 'sensor', None)
            sz  = float(getattr(snr, 'z_pos', getattr(snr, 'z_vertex', 0.0))
                        if snr else 0.0)
            voxel_h = (z_clip_max - z_clip_min) / max(1, state['dims'][2])
            z_clip_min = sz - voxel_h * 2
            z_clip_max = sz + voxel_h * 2

        scale = float(self._glow_field_scale)

        glEnable(GL_SCISSOR_TEST)
        glScissor(vp_x, vp_y, vp_w, vp_h)
        glViewport(vp_x, vp_y, vp_w, vp_h)
        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)

        p = self._prog_march
        glUseProgram(p)

        def _ul(n):
            return glGetUniformLocation(p, n if isinstance(n, bytes) else n.encode())

        glUniformMatrix4fv(_ul(b'uInvMVP'), 1, GL_TRUE, inv_mvp)
        glUniformMatrix4fv(_ul(b'uWorldToGrid'), 1, GL_TRUE,
                           np.eye(4, dtype=np.float32))
        glUniform3f(_ul(b'uBoxMin'), *bmin)
        glUniform3f(_ul(b'uBoxMax'), *bmax)
        glUniform2f(_ul(b'uMaskMin'), float(bmin[0]), float(bmin[1]))
        glUniform2f(_ul(b'uMaskMax'), float(bmax[0]), float(bmax[1]))
        _rp_act  = getattr(self._right_panel, 'sim_activation', 'log')
        _rp_norm = getattr(self._right_panel, 'sim_norm_mode',  'none')
        _use_log = (_rp_act == 'log')
        _gamma   = {'log':    float(_GPU_RAY_FIELD_GAMMA),
                    'linear': 1.0,
                    'sqrt':   0.5,
                    'gamma':  float(_GPU_RAY_FIELD_GAMMA)}.get(_rp_act, float(_GPU_RAY_FIELD_GAMMA))
        _vscale  = {'none':    scale,
                    'reinhard': scale * 0.5,
                    'clamp':    1.0}.get(_rp_norm, scale)
        glUniform1f(_ul(b'uRayFieldScale'), _vscale)
        glUniform1f(_ul(b'uRayFieldGamma'), _gamma)
        glUniform1f(_ul(b'uAlpha'), 1.0)
        glUniform1i(_ul(b'uLogScale'), 1 if _use_log else 0)
        glUniform1i(_ul(b'uFieldMode'), 1)
        glUniform1i(_ul(b'uUseBodyMask'), 0)
        glUniform1f(_ul(b'uPressureScale'), 1.0)
        glUniform1f(_ul(b'uPressureGamma'), 1.0)
        glUniform1i(_ul(b'uAMRMode'), 0)
        # Point unused sampler uniforms to high empty units to avoid
        # cross-type conflicts on unit 0 (which holds uLayer0, a usampler3D).
        # uBodyMask=sampler2D and uAMRData=samplerBuffer must not share a unit
        # with a different-typed sampler — that generates GL_INVALID_OPERATION.
        glUniform1i(_ul(b'uPressure'),      8)   # sampler3D  — unused (uFieldMode=1)
        glUniform1i(_ul(b'uExteriorMask'),  8)   # sampler3D  — unused (uFieldMode=1)
        glUniform1i(_ul(b'uBodyMask'),      9)   # sampler2D  — unused (uUseBodyMask=0)
        glUniform1i(_ul(b'uAMRData'),      10)   # samplerBuffer — unused (uAMRMode=0)
        glUniform1f(_ul(b'uZClipMin'), z_clip_min)
        glUniform1f(_ul(b'uZClipMax'), z_clip_max)
        glUniform1i(_ul(b'uLayerCount'), n_bands)

        for bi in range(n_bands):
            dark  = (0.0, 0.0, 0.0)
            light = (1.0, 1.0, 1.0)
            if bi < len(layer_specs):
                dark  = tuple(float(v) for v in layer_specs[bi].get('dark_rgb',  (0.0, 0.0, 0.0)))
                light = tuple(float(v) for v in layer_specs[bi].get('light_rgb', (1.0, 1.0, 1.0)))
            glUniform3f(_ul(f'uLayerDark{bi}'),  *dark)
            glUniform3f(_ul(f'uLayerLight{bi}'), *light)
            glUniform1i(_ul(f'uLayer{bi}'), bi)
            glActiveTexture(GL_TEXTURE0 + bi)
            glBindTexture(GL_TEXTURE_3D, tex_bands[bi])

        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(self._march_vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)

        for bi in range(n_bands):
            glActiveTexture(GL_TEXTURE0 + bi)
            glBindTexture(GL_TEXTURE_3D, 0)
        glActiveTexture(GL_TEXTURE0)

        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_SCISSOR_TEST)

    def _draw_cross_section(
        self,
        vp_x: int, vp_y: int, vp_w: int, vp_h: int,
        lines: list,
        view: _OrthoView,
        win_w: int, win_h: int,
        title: str = "",
        border_rgba: Optional[tuple] = None,
    ) -> None:
        glEnable(GL_SCISSOR_TEST)
        glScissor(vp_x, vp_y, vp_w, vp_h)
        glViewport(vp_x, vp_y, vp_w, vp_h)

        # Dark background
        glClearColor(0.02, 0.03, 0.06, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self._prog_line)
        aspect = vp_w / max(1, vp_h)
        sx, sy = view.scale_vec(aspect)
        ox, oy = view.offset_vec()

        u_scale  = glGetUniformLocation(self._prog_line, "uScale")
        u_offset = glGetUniformLocation(self._prog_line, "uOffset")
        u_color  = glGetUniformLocation(self._prog_line, "uColor")

        glUniform2f(u_scale,  sx, sy)
        glUniform2f(u_offset, ox, oy)
        glLineWidth(1.5)

        for pts, rgba in lines:
            if len(pts) < 2:
                continue
            arr = np.array(pts, dtype=np.float32)  # (N, 2): col0=Z or X, col1=Y or X
            flat = arr.flatten()
            glBindVertexArray(self._xz_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self._xz_vbo)
            glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STREAM_DRAW)
            glUniform4f(u_color, rgba[0], rgba[1], rgba[2], rgba[3] if len(rgba) > 3 else 1.)
            glDrawArrays(GL_LINE_STRIP, 0, len(arr))
            glBindVertexArray(0)

        # Viewport border — drawn in NDC using scale=1, offset=0 so corners=(±0.99, ±0.99)
        if border_rgba is not None:
            glLineWidth(2.5)
            glUniform2f(u_scale,  1.0, 1.0)
            glUniform2f(u_offset, 0.0, 0.0)
            glUniform4f(u_color, *border_rgba)
            border_pts = np.array(
                [(-0.993, -0.993), (0.993, -0.993),
                 (0.993,  0.993), (-0.993,  0.993),
                 (-0.993, -0.993)], dtype=np.float32)
            flat = border_pts.flatten()
            glBindVertexArray(self._xz_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self._xz_vbo)
            glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STREAM_DRAW)
            glDrawArrays(GL_LINE_STRIP, 0, 5)
            glBindVertexArray(0)
            glLineWidth(1.5)

        glDisable(GL_SCISSOR_TEST)

    def _draw_back_image(
        self,
        vp_x_gl: int, vp_y_gl: int, vp_w: int, vp_h: int,
        back,
    ) -> None:
        """Render the accumulated CameraBack image into a GL viewport region.

        vp_x_gl / vp_y_gl are OpenGL-style coordinates (y from bottom).
        Converts to HUD pixel coords (y from top) and blits via the HUD shader.
        Second-pass bidirectional solve hook: when a BidirectionalSolveNode is
        wired to this station, it should call back._splat() with computed
        irradiance maps each frame before this method runs; until then the
        image shows zero-exposure (black sensor).
        """
        if back is None or not _HAS_GL:
            return
        try:
            img = back.to_rect()   # (H, W, C) float64
        except Exception:
            return
        h_img, w_img = img.shape[:2]
        rgb = np.clip(img[..., :3] * 255.0, 0, 255).astype(np.uint8)
        rgba = np.ones((h_img, w_img, 4), np.uint8) * 255
        rgba[..., :3] = rgb
        # Flip vertically: numpy row 0 = image top; GL textures load bottom-up
        raw = rgba[::-1].tobytes()
        if self._br_tex is None:
            self._br_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._br_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w_img, h_img, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glBindTexture(GL_TEXTURE_2D, 0)
        # Convert GL coords to HUD pixel coords (y from top)
        screen_y = self.win_h - (vp_y_gl + vp_h)
        glDisable(GL_SCISSOR_TEST)
        glViewport(0, 0, self.win_w, self.win_h)
        self._blit_panel(vp_x_gl, screen_y, vp_w, vp_h,
                         self._br_tex, self.win_w, self.win_h)

    def _blit_panel(
        self,
        x: int, y: int, w: int, h: int,
        tex: Optional[int],
        win_w: int, win_h: int,
    ) -> None:
        if tex is None:
            return
        verts = _quad_verts(x, y, w, h)
        glBindVertexArray(self._hud_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._hud_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STREAM_DRAW)
        glUseProgram(self._prog_hud)
        glUniform2f(glGetUniformLocation(self._prog_hud, "uRes"),
                    float(win_w), float(win_h))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(glGetUniformLocation(self._prog_hud, "uTex"), 0)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _upload_panels(self, win_w: int, win_h: int, lw: int, rw: int) -> None:
        lsurf = self._left_panel.render(lw, win_h)
        rsurf = self._right_panel.render(rw, win_h)
        if self._left_tex is None:
            self._left_tex  = _make_tex(lsurf)
            self._right_tex = _make_tex(rsurf)
        else:
            _update_tex(self._left_tex,  lsurf)
            _update_tex(self._right_tex, rsurf)

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev) -> bool:
        lw = self.LEFT_W
        rw = self.RIGHT_W
        vp_x  = lw
        vp_w  = self.win_w - lw - rw
        vp_h  = self.win_h
        hw    = vp_w // 2
        hh    = vp_h // 2

        # Left panel
        consumed, action = self._left_panel.handle_event(
            ev, x_off=0, panel_w=lw)
        if action == "btn_bake":
            self._start_bake()
            self._panels_dirty = True
        elif action == "btn_manifold_bdpt":
            self._start_manifold_bdpt()
            self._panels_dirty = True
            return True
        if consumed:
            self._panels_dirty = True
            if action in {"sensor_x", "sensor_y", "sensor_z"}:
                self._rebuild_lines()
                self._request_optical_preview()
            elif action in {"camera_zoom", "focus_distance_m"} and (
                ev.type == pygame.MOUSEWHEEL
                or (
                    ev.type == pygame.MOUSEBUTTONUP
                    and getattr(ev, "button", 0) == 1
                )
                or (
                    ev.type == pygame.MOUSEBUTTONDOWN
                    and getattr(ev, "button", 0) == 3
                )
            ):
                self._apply_canonical_camera_controls()
            elif action.startswith("el") and "_" in action and (
                ev.type == pygame.MOUSEWHEEL
                or (
                    ev.type == pygame.MOUSEBUTTONUP
                    and getattr(ev, "button", 0) == 1
                )
                or (
                    ev.type == pygame.MOUSEBUTTONDOWN
                    and (
                        getattr(ev, "button", 0) == 3
                        or action.endswith("_conic_type")
                    )
                )
            ):
                self._apply_canonical_surface_control(action)
            return True

        # Right panel
        consumed, rp_action = self._right_panel.handle_event(
            ev, x_off=self.win_w - rw, panel_w=rw)
        if consumed:
            self._panels_dirty = True
            if rp_action and (rp_action.startswith('ctx_')
                              or rp_action == 'sim_max_bounces'):
                self._build_gpu_dispatch()
            elif rp_action == "bottom_right_view":
                self._request_optical_preview()
            return True

        # Right-click in viewport → add point light at clicked position
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
            mx2, my2 = ev.pos
            if vp_x <= mx2 < vp_x + vp_w:
                lx2 = mx2 - vp_x
                # GL viewport layout (y measured from screen bottom):
                #   XZ top-left:  screen y ∈ [0, hh)   lx2 ∈ [0, hw)
                #   YZ top-right: screen y ∈ [0, hh)   lx2 ∈ [hw, vp_w)
                #   XY bottom:    screen y ∈ [hh, vh)  full width
                # Coordinate mapping: NDC = (world - offset) / scale_vec
                #   px = (NDC_x+1)/2 * w  →  world_h = offset[0] + (2*px/w - 1)*scale
                #   py = (1-NDC_y)/2 * h  →  world_v = offset[1] + (1 - 2*py/h)*(scale/aspect)
                if my2 < hh:  # top row → XZ or YZ
                    pane2 = "xz" if lx2 < hw else "yz"
                    view2 = self._view_xz if pane2 == "xz" else self._view_yz
                    pix_x = lx2 % hw
                    pix_y = my2
                    lx_m = view2.offset[0] + (2.0 * pix_x / max(hw, 1) - 1.0) * view2.scale
                    ly_m = view2.offset[1] + (1.0 - 2.0 * pix_y / max(hh, 1)) * (view2.scale * hh / max(hw, 1))
                    # XZ: horiz=Z, vert=X  /  YZ: horiz=Z, vert=Y
                    pos3 = [ly_m, 0.0, lx_m] if pane2 == "xz" else [0.0, ly_m, lx_m]
                else:  # bottom → XY aperture plane
                    pix_x = lx2
                    pix_y = my2 - hh
                    lx_m = self._view_xy.offset[0] + (2.0 * pix_x / max(vp_w, 1) - 1.0) * self._view_xy.scale
                    ly_m = self._view_xy.offset[1] + (1.0 - 2.0 * pix_y / max(hh, 1)) * (self._view_xy.scale * hh / max(vp_w, 1))
                    pos3 = [lx_m, ly_m, 0.0]
                import random as _rnd
                _shapes = ('cube', 'sphere', 'tetrahedron', 'octahedron', 'icosahedron')
                _mats   = ('diffuse', 'reflective', 'emissive', 'translucent')
                self._scene_meshes.append({
                    'pos':    [float(v) for v in pos3],
                    'shape':  _rnd.choice(_shapes),
                    'mat':    _rnd.choice(_mats),
                    'color':  [_rnd.uniform(0.25, 1.0) for _ in range(3)],
                    'radius': _rnd.uniform(0.005, 0.025),
                })
                # NOTE: scene_meshes appear as markers in cross-sections;
                # GPU geometry integration requires extending build_gpu_scene.
                self._rebuild_lines()
                self._panels_dirty = True
                return True

        # Keyboard shortcuts
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:   # SPACE = rebuild GPU dispatch
                self._build_gpu_dispatch()
                return True
            if ev.key == pygame.K_g:       # G = toggle glow brightness cycle
                self._glow_brightness = {
                    4.0: 8.0, 8.0: 1.0, 1.0: 4.0
                }.get(self._glow_brightness, 4.0)
                return True
            if ev.key == pygame.K_c:       # C = clear glow
                self._clear_glow()
                return True
            if ev.key == pygame.K_l:       # L = clear all lights
                self._scene_lights.clear()
                self._build_gpu_dispatch()
                self._panels_dirty = True
                return True
            if ev.key == pygame.K_s:       # S = toggle slice / projection mode
                self._slice_mode = not self._slice_mode
                self._panels_dirty = True
                print(f"[camera_designer] ortho mode: {'SLICE' if self._slice_mode else 'PROJECTION'} "
                      f"(thickness {self._slice_thickness*1e3:.1f} mm)", flush=True)
                return True
            if ev.key == pygame.K_LEFTBRACKET:    # [ = thinner slab
                self._slice_thickness = max(2e-4, self._slice_thickness / 1.5)
                self._panels_dirty = True
                print(f"[camera_designer] slice thickness → {self._slice_thickness*1e3:.2f} mm",
                      flush=True)
                return True
            if ev.key == pygame.K_RIGHTBRACKET:   # ] = thicker slab
                self._slice_thickness = min(0.05, self._slice_thickness * 1.5)
                self._panels_dirty = True
                print(f"[camera_designer] slice thickness → {self._slice_thickness*1e3:.2f} mm",
                      flush=True)
                return True

        # Center tab bar click — checked before viewport pan/zoom routing
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx0, my0 = getattr(ev, "pos", (0, 0))
            if vp_x <= mx0 < vp_x + vp_w and 0 <= my0 < self._CENTER_TAB_H:
                lx_tab   = mx0 - vp_x
                tab_keys = [t[0] for t in self._CENTER_TABS]
                for ti, tr in enumerate(self._center_tab_rects):
                    if tr.collidepoint(lx_tab, my0):
                        self._center_tab = tab_keys[ti]
                        return True

        # LIGHTS tab: del-button clicks in the lights list panel
        if (self._center_tab == "lights"
                and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1):
            mx0, my0 = getattr(ev, "pos", (0, 0))
            if vp_x <= mx0 < vp_x + vp_w:
                lx_tab = mx0 - vp_x
                for i, btn_r in enumerate(self._lights_remove_rects):
                    if btn_r.collidepoint(lx_tab, my0):
                        del self._scene_lights[i]
                        self._lights_remove_rects = []
                        self._rebuild_lines()
                        self._build_gpu_dispatch()
                        return True

        # Cross-section pane mouse events (VIEWS tab only)
        mx, my = (getattr(ev, "pos", (0, 0)))
        if self._center_tab == "views" and ev.type in (pygame.MOUSEBUTTONDOWN,
                       pygame.MOUSEMOTION,
                       pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL):
            if vp_x <= mx < vp_x + vp_w:
                # Determine which pane
                lx = mx - vp_x
                if my >= hh:   # top row
                    pane = "xz" if lx < hw else "yz"
                    view = self._view_xz if pane == "xz" else self._view_yz
                else:          # bottom row
                    pane = "xy"; view = self._view_xy

                if ev.type == pygame.MOUSEWHEEL:
                    factor = 0.85 if ev.y > 0 else 1.15
                    view.zoom(factor)
                    return True
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self._drag_pane = pane
                    self._drag_last = (mx, my)
                    return True
                if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self._drag_pane = None
                    return True
                if ev.type == pygame.MOUSEMOTION and self._drag_pane == pane:
                    if self._drag_last:
                        dx = mx - self._drag_last[0]
                        dy = my - self._drag_last[1]
                        self._drag_last = (mx, my)
                        # Convert pixel delta to metres
                        pix_per_m = (vp_w // 2) / max(view.scale, 1e-12)
                        view.pan(-dx / pix_per_m, dy / pix_per_m)
                    return True

        # Update right panel when selection changes
        sel_idx = self._left_panel._selected_idx
        if sel_idx >= 0 and sel_idx < len(self.preset.lens_group.elements):
            el = self.preset.lens_group.elements[sel_idx]
            self._right_panel.element_info = {
                "type":   el.surface.glsl_type,
                "label":  el.label or "-",
                "z_vert": f"{el.z_vertex*1e3:.3f} mm",
                "r_max":  f"{el.surface.r_max*1e3:.3f} mm",
                "glass":  el.glass_out.name,
                "n_d":    f"{el.glass_out.n_d:.4f}",
            }

        return False

    # ── Backend transport requests ────────────────────────────────────────────

    def _start_bake(self) -> None:
        """Retired local bake button; invalidate the engine-owned preview."""
        self._left_panel.bake_state = "queued"
        self._left_panel.bake_msg = "submitted to optical backend"
        self._request_optical_preview()
        self._panels_dirty = True

    def _start_manifold_bdpt(self) -> None:
        """Do not instantiate a station-local BDPT solver."""
        self._left_panel.manifold_bdpt_state = "backend"
        self._request_optical_preview()
        self._panels_dirty = True

    # ── Preset switching ──────────────────────────────────────────────────────

    def load_preset(self, preset: "CameraPreset") -> None:
        """Hot-swap the active preset and rebuild geometry."""
        self.preset = preset
        self.sensor_cfg.sync_elements(preset)
        self._sync_surface_controls_from_manifest()
        self._left_panel.preset     = preset
        self._left_panel.sensor_cfg = self.sensor_cfg   # keep live config
        self._left_panel._selected_idx = -1
        self._left_panel.bake_state = "idle"
        self._left_panel.bake_msg   = ""
        self._right_panel.element_info = {}
        self._right_panel.lut_info     = {}
        self._manifold = None
        self._ctx_map  = {}
        # Propagate sensor log-scale flag from the preset (set in sensor YAML /
        # camera preset YAML via CameraPreset.sensor_log_scale)
        self._glow_log_scale = bool(getattr(preset, 'sensor_log_scale', False))
        self._setup_auto_computer()
        self._rebuild_lines()
        if self._gl_ready:
            self._request_optical_preview()

    def load_camera_manifest(self, camera) -> None:
        """Load the optical engine's canonical camera into this frontend.

        ``CameraPreset`` remains only the designer/GPU-view projection.  The
        manifest is retained so saving, cache identity and subsequent engine
        requests continue to refer to the production camera rather than to the
        derived drawing object.
        """
        from camera_software.camera_designer_bridge import (
            camera_manifest_to_designer_preset,
        )
        from camera_software.optical_bench import resolve_bench_camera_manifest

        resolved = resolve_bench_camera_manifest(camera)
        self.camera_manifest = resolved
        mapping = resolved.mapping()
        if hasattr(self, "sensor_cfg"):
            self.sensor_cfg.zoom = float(
                dict(mapping.get("lens", {})).get("zoom", 0.5)
            )
            self.sensor_cfg.focus_distance_m = float(
                dict(mapping.get("focus", {})).get("distance_m", 1.0)
            )
        self.load_preset(camera_manifest_to_designer_preset(resolved))
        self._panels_dirty = True

    def _sync_surface_controls_from_manifest(self) -> None:
        if self.camera_manifest is None or not hasattr(self, "sensor_cfg"):
            return
        lens = dict(self.camera_manifest.mapping().get("lens", {}))
        adjustments = dict(lens.get("surface_adjustments", {}))
        by_label = {
            str(element.label): index
            for index, element in enumerate(self.preset.lens_group.elements)
        }
        field_map = {
            "axial_offset_mm": ("axial_offset", 1.0e-3),
            "shift_x_mm": ("shift_x", 1.0e-3),
            "shift_y_mm": ("shift_y", 1.0e-3),
            "tilt_x_deg": ("tilt_x", 1.0),
            "tilt_y_deg": ("tilt_y", 1.0),
            "radius_mm": ("radius", 1.0e-3),
            "clear_radius_mm": ("clear_radius", 1.0e-3),
            "conic_k": ("conic_k", 1.0),
            "conic_type": ("conic_type", 1.0),
        }
        for label, edit in adjustments.items():
            index = by_label.get(str(label))
            if index is None:
                continue
            for source, (target, scale) in field_map.items():
                if source not in edit:
                    continue
                value = edit[source]
                self.sensor_cfg.set_el_geometry_value(
                    index, target,
                    str(value) if source == "conic_type" else float(value) * scale,
                )

    def _apply_canonical_camera_controls(self) -> None:
        """Resolve zoom/focus edits through the production camera manifest."""

        zoom = float(self.sensor_cfg.zoom)
        from camera_software.camera_designer_bridge import (
            camera_manifest_to_designer_preset,
            camera_manifest_with_controls,
        )
        resolved = camera_manifest_with_controls(
            self.camera_manifest,
            zoom=zoom,
            focus_distance_m=self.sensor_cfg.focus_distance_m,
        )
        # Avoid feeding values back through load_camera_manifest: the controls
        # are already authoritative for this edit and should not jump while the
        # user releases the knob.
        self.camera_manifest = resolved
        self.load_preset(camera_manifest_to_designer_preset(resolved))
        print(
            "[camera_designer] canonical camera re-solved "
            f"zoom={zoom:.3f} focal={self.preset.focal_mm:.2f}mm "
            f"focus={self.sensor_cfg.focus_distance_m:.3f}m",
            flush=True,
        )

    def _apply_canonical_surface_control(self, action: str) -> None:
        """Commit one visible surface control through CameraManifest."""

        head, field = str(action).split("_", 1)
        index = int(head[2:])
        if not 0 <= index < len(self.preset.lens_group.elements):
            return
        element = self.preset.lens_group.elements[index]
        value = self.sensor_cfg.el_geometry_value(index, field)
        manifest_field = {
            "axial_offset": "axial_offset_mm",
            "shift_x": "shift_x_mm",
            "shift_y": "shift_y_mm",
            "tilt_x": "tilt_x_deg",
            "tilt_y": "tilt_y_deg",
            "radius": "radius_mm",
            "clear_radius": "clear_radius_mm",
            "conic_type": "conic_type",
            "conic_k": "conic_k",
        }.get(field)
        if manifest_field is None:
            return
        if manifest_field.endswith("_mm"):
            value = float(value) * 1.0e3
        from camera_software.camera_designer_bridge import (
            camera_manifest_to_designer_preset,
            camera_manifest_with_surface_adjustment,
        )
        update = {manifest_field: value}
        if field == "conic_k":
            update["conic_type"] = str(
                self.sensor_cfg.el_geometry_value(index, "conic_type", "sphere")
            )
        resolved = camera_manifest_with_surface_adjustment(
            self.camera_manifest, element.label, **update,
        )
        self.camera_manifest = resolved
        self.load_preset(camera_manifest_to_designer_preset(resolved))
        edit = dict(resolved.mapping()["lens"]["surface_adjustments"])[element.label]
        print(
            f"[camera_designer] surface {element.label} updated {edit}",
            flush=True,
        )

    def _setup_auto_computer(self) -> None:
        """Create (or replace) the CameraComputer that drives auto-ISO.

        Uses a real CameraItem so CameraContext is valid.  The GL Camera
        slot (_cam) is a minimal shim that carries iso/sensor_iso; no GL
        state is created.

        target_ev = log2(0.18) ≈ -2.47: 18 % grey — photographic metering
        standard (reflected-light K=12.5 calibration constant).
        """
        if not _HAS_AUTO_COMPUTER:
            return
        import types as _types
        # ── 1. GL-camera shim (carries iso; no GL resources needed) ────────────
        init_iso = float(getattr(self, '_glow_field_scale', 4.0))
        cam_gl   = _types.SimpleNamespace(iso=init_iso, sensor_iso=init_iso)
        # ── 2. Real CameraItem built from the current preset ────────────────
        try:
            from placed_object import PlacedCamera
            from camera_item   import CameraItem
            preset_name = getattr(self.preset, 'name', 'designer_bench')
            placed = PlacedCamera(
                obj_id      = "_designer_bench",
                label       = preset_name,
                pos         = np.zeros(3, np.float64),
                focal_mm    = float(getattr(self.preset, 'focal_mm', 50.0)),
                focal_min_mm= float(getattr(self.preset, 'focal_mm', 50.0)),
                focal_max_mm= float(getattr(self.preset, 'focal_mm', 50.0)),
            )
            cam_item = CameraItem(placed)
            # No init_gl() — we never call cam_item.draw()
        except Exception:
            cam_item = None
        self._auto_cam_item = cam_item
        self._auto_cam_gl   = cam_gl
        # ── 3. CameraComputer ──────────────────────────────────────────
        comp = _CameraComputer(
            target_ev  = math.log2(0.18),  # 18 % grey — photographic metering standard
            iso_min    = 0.25,
            iso_max    = 512.0,
            iso_speed  = 0.8,
            ev_lpf     = 0.88,             # broad smoothing — no flicker
        )
        comp.auto_iso = True
        self._auto_computer = comp
        self._auto_last_t   = time.perf_counter()
