"""room_duty_station.py
======================
RoomDutyStation — the room-configuration duty station.

This console lets the player reconfigure the room geometry per-quadrant
in real time.  It is one of the placeable duty-station types (station_type
= "room") and is instantiated by RoomWorkspace.build_scene_objects() when
found in scene.yaml.

Layout when in INTERACT mode
─────────────────────────────────────────────────────────────────────
  ┌────────────────────────────────────────────────────────────────┐
  │  ROOM CONFIG                                                   │
  ├──────────────────────────┬─────────────────────────────────────┤
  │  QUADRANT SELECT         │  DIMENSIONS                         │
  │  ○ NW  ○ NE              │  width  [====|========]  12.0 m     │
  │  ○ SW  ○ SE              │  depth  [=======|=====]  10.0 m     │
  │  (○ all)                 │  height [===|=========]   4.0 m     │
  ├──────────────────────────┼─────────────────────────────────────┤
  │  COORDINATE SYSTEM       │  GEOMETRY                           │
  │  ○ rectangular           │  grid spacing  [===]  1.0 m         │
  │  ○ cylindrical           │  wall thickness [=]  0.25 m         │
  │  ○ spherical             │                                     │
  └──────────────────────────┴─────────────────────────────────────┘

Each quadrant may have its own coordinate system.  "All" applies a change
to all four simultaneously.  Changes propagate to RoomWorkspace and
trigger a room mesh rebuild on next render.

Integration
-----------
``RoomDutyStation`` exposes the same minimal interface as other stations:
  - ``init_gl()``
  - ``draw(MVP, MV, light_v)``
  - ``handle_event(ev) -> bool``
  - ``render(win_w, win_h)``    # 2D HUD overlay, called from main loop
  - ``world_position``          # np.ndarray  for proximity
  - ``interaction_radius``      # float
  - ``interact_camera``         # dict with 'eye' and 'target'

No GL is required for the HUD: it is rendered via pygame.Surface
blitted into the GL framebuffer (same technique as SimulatorStation).
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_BLEND, GL_DEPTH_TEST, GL_FALSE, GL_FLOAT,
        GL_FRAGMENT_SHADER, GL_ONE_MINUS_SRC_ALPHA, GL_RGBA,
        GL_SCISSOR_TEST, GL_SRC_ALPHA, GL_STATIC_DRAW, GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_LINEAR,
        GL_TRIANGLES, GL_TRUE, GL_UNSIGNED_BYTE, GL_VERTEX_SHADER,
        glBindBuffer, glBindTexture, glBindVertexArray, glBlendFunc,
        glBufferData, glDisable, glDrawArrays, glEnable,
        glEnableVertexAttribArray, glGenBuffers, glGenTextures,
        glGenVertexArrays, glGetUniformLocation,
        glTexImage2D, glTexParameteri, glUniform1i, glUniform2f,
        glUniformMatrix4fv, glUseProgram, glVertexAttribPointer, glViewport,
        glActiveTexture, GL_TEXTURE0,
    )
    from OpenGL.GL import shaders as _gl_shaders
    _HAS_GL = True
except ImportError:
    _HAS_GL = False


# ─────────────────────────────────────────────────────────────────────────────
# Domain types
# ─────────────────────────────────────────────────────────────────────────────

class CoordSystem(Enum):
    RECTANGULAR  = "rectangular"
    CYLINDRICAL  = "cylindrical"
    SPHERICAL    = "spherical"


@dataclass
class QuadrantConfig:
    """Per-quadrant room configuration."""
    coord_system: CoordSystem = CoordSystem.RECTANGULAR
    # Rectangular
    width_m:   float = 6.0
    depth_m:   float = 5.0
    height_m:  float = 4.0
    # Cylindrical — radius replaces width/depth
    radius_m:  float = 3.0
    # Spherical — radius replaces all
    # (height_m is used as total sphere diameter)

    def to_dict(self) -> dict:
        return {
            "coord_system": self.coord_system.value,
            "width_m":  self.width_m,
            "depth_m":  self.depth_m,
            "height_m": self.height_m,
            "radius_m": self.radius_m,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuadrantConfig":
        cs = CoordSystem(d.get("coord_system", "rectangular"))
        return cls(
            coord_system = cs,
            width_m  = float(d.get("width_m",  6.0)),
            depth_m  = float(d.get("depth_m",  5.0)),
            height_m = float(d.get("height_m", 4.0)),
            radius_m = float(d.get("radius_m", 3.0)),
        )


# Quadrant names — NW/NE/SW/SE in the XZ floor plane
QUADRANT_NAMES = ("NW", "NE", "SW", "SE")


@dataclass
class RoomConfigState:
    """Full mutable room configuration, editable per-quadrant."""
    quadrants: Dict[str, QuadrantConfig] = field(
        default_factory=lambda: {n: QuadrantConfig() for n in QUADRANT_NAMES}
    )
    grid_spacing_m:   float = 1.0
    wall_thickness_m: float = 0.25
    selected_quadrant: str  = "all"   # "all" | "NW" | "NE" | "SW" | "SE"

    def to_dict(self) -> dict:
        return {
            "quadrants":        {k: v.to_dict() for k, v in self.quadrants.items()},
            "grid_spacing_m":   self.grid_spacing_m,
            "wall_thickness_m": self.wall_thickness_m,
        }

    @classmethod
    def from_room_yaml(cls, room_cfg: dict) -> "RoomConfigState":
        dims = room_cfg.get("dimensions", {})
        W = float(dims.get("width_m",  12.0))
        D = float(dims.get("depth_m",  10.0))
        H = float(dims.get("height_m",  4.0))
        grid = float(room_cfg.get("floor_grid", {}).get("spacing_m", 1.0))
        wt   = float(dims.get("wall_thickness_m", 0.25))
        base = QuadrantConfig(width_m=W / 2, depth_m=D / 2, height_m=H)
        return cls(
            quadrants        = {n: QuadrantConfig(
                                    width_m=W / 2, depth_m=D / 2, height_m=H)
                                for n in QUADRANT_NAMES},
            grid_spacing_m   = grid,
            wall_thickness_m = wt,
        )

    def effective_room_dims(self) -> dict:
        """Merge quadrant configs into a single room.yaml-compatible dict."""
        all_w = [q.width_m  for q in self.quadrants.values()]
        all_d = [q.depth_m  for q in self.quadrants.values()]
        all_h = [q.height_m for q in self.quadrants.values()]
        return {
            "dimensions": {
                "width_m":          sum(all_w) / 2,
                "depth_m":          sum(all_d) / 2,
                "height_m":         max(all_h),
                "wall_thickness_m": self.wall_thickness_m,
            },
            "floor_grid": {
                "spacing_m": self.grid_spacing_m,
                "enabled":   True,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# HUD rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

_PANEL_BG    = (14,  18,  28, 220)
_ACCENT      = (30, 100, 200, 255)
_TEXT_FG     = (200, 210, 230, 255)
_TEXT_DIM    = (110, 120, 140, 255)
_SEL_BG      = (30,  50,  90, 200)
_SLIDER_RAIL = (40,  50,  70, 255)
_SLIDER_FILL = (60, 130, 220, 255)


def _get_font(size: int = 15) -> pygame.font.Font:
    try:
        return pygame.font.SysFont("consolas,monospace", size)
    except Exception:
        return pygame.font.Font(None, size)


def _draw_label(surf: pygame.Surface, text: str, x: int, y: int,
                color=_TEXT_FG, size: int = 14) -> None:
    f = _get_font(size)
    s = f.render(text, True, color[:3])
    surf.blit(s, (x, y))


def _draw_slider(surf: pygame.Surface, x: int, y: int, w: int,
                 value: float, vmin: float, vmax: float,
                 label: str, unit: str = "m") -> pygame.Rect:
    """Draw a horizontal slider.  Returns the rail Rect for hit-testing."""
    frac   = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 1e-9)))
    rail_h = 8
    rail_y = y + 10
    rail   = pygame.Rect(x, rail_y, w, rail_h)
    pygame.draw.rect(surf, _SLIDER_RAIL[:3], rail, border_radius=4)
    fill_w = max(4, int(frac * w))
    pygame.draw.rect(surf, _SLIDER_FILL[:3],
                     pygame.Rect(x, rail_y, fill_w, rail_h), border_radius=4)
    # knob
    kx = x + fill_w
    pygame.draw.circle(surf, (200, 220, 255), (kx, rail_y + rail_h // 2), 7)
    _draw_label(surf, label,                  x,         y - 2,  _TEXT_DIM, 13)
    _draw_label(surf, f"{value:.1f} {unit}",  x + w + 6, y + 2,  _TEXT_FG,  13)
    return rail


def _draw_radio(surf: pygame.Surface, x: int, y: int, label: str,
                selected: bool) -> pygame.Rect:
    r = 7
    cx, cy = x + r, y + r
    color  = _ACCENT[:3] if selected else _SLIDER_RAIL[:3]
    pygame.draw.circle(surf, color, (cx, cy), r)
    if selected:
        pygame.draw.circle(surf, (200, 220, 255), (cx, cy), r - 3)
    _draw_label(surf, label, x + r * 2 + 4, y, _TEXT_FG, 14)
    return pygame.Rect(x, y, 120, r * 2)


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay — 2D panel drawn on a pygame Surface, uploaded as GL texture
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


class _HudPanel:
    """Upload a pygame.Surface to a GL texture and blit it as a full-quad."""

    def __init__(self) -> None:
        self._tex   = None
        self._vao   = None
        self._vbo   = None
        self._prog  = None
        self._ready = False

    def init_gl(self) -> None:
        if not _HAS_GL:
            return
        vs = _gl_shaders.compileShader(_HUD_VS, GL_VERTEX_SHADER)
        fs = _gl_shaders.compileShader(_HUD_FS, GL_FRAGMENT_SHADER)
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

    def upload(self, surf: pygame.Surface) -> None:
        if not self._ready:
            return
        raw  = pygame.image.tostring(surf, "RGBA", True)
        w, h = surf.get_size()
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw(self, x: int, y: int, w: int, h: int,
             win_w: int, win_h: int) -> None:
        if not self._ready:
            return
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self._prog)
        glUniform2f(glGetUniformLocation(self._prog, "uPos"),  float(x), float(y))
        glUniform2f(glGetUniformLocation(self._prog, "uSize"), float(w), float(h))
        glUniform2f(glGetUniformLocation(self._prog, "uWin"),  float(win_w), float(win_h))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glUniform1i(glGetUniformLocation(self._prog, "uTex"), 0)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)


# ─────────────────────────────────────────────────────────────────────────────
# RoomDutyStation
# ─────────────────────────────────────────────────────────────────────────────

class RoomDutyStation:
    """Room-configuration duty station.

    Manages per-quadrant room geometry settings.  The console is mounted
    flush against a wall (yaw_deg in PlacedDutyStation orients it).

    Parameters
    ----------
    placed : PlacedDutyStation
        The scene entry that spawned this station.
    room_workspace : RoomWorkspace
        Used to read initial room dims and push updates back.
    win_w, win_h : int
        Window pixel dimensions at init time (updated each render call).
    hud_w, hud_h : int
        HUD panel pixel size.
    """

    HUD_W = 620
    HUD_H = 360

    def __init__(self, placed, room_workspace,
                 win_w: int = 1400, win_h: int = 900,
                 hud_w: Optional[int] = None,
                 hud_h: Optional[int] = None):
        self._placed    = placed
        self._workspace = room_workspace
        self.win_w      = win_w
        self.win_h      = win_h
        self.hud_w      = hud_w or self.HUD_W
        self.hud_h      = hud_h or self.HUD_H
        self._visible   = False
        self._gl_ready  = False

        # Build config state from current room.yaml
        self._state = RoomConfigState.from_room_yaml(
            room_workspace.room_cfg if hasattr(room_workspace, "room_cfg") else {}
        )

        # Slider drag state: (quadrant_key, field_name) | None
        self._dragging: Optional[Tuple[str, str]] = None
        self._drag_x0:  int   = 0
        self._drag_v0:  float = 0.0

        # Hit-test rects for sliders  { (quadrant, field): pygame.Rect }
        self._slider_rects: Dict[Tuple[str, str], pygame.Rect] = {}

        # Radio rects for quadrant select and coord system
        self._quadrant_rects: Dict[str, pygame.Rect] = {}
        self._coord_rects:    Dict[str, Dict[str, pygame.Rect]] = {}

        self._hud_panel = _HudPanel()

    # ── Proximity interface (PlayerController) ────────────────────────────────

    @property
    def world_position(self) -> np.ndarray:
        return self._placed.pos

    @property
    def interaction_radius(self) -> float:
        return float(getattr(self._placed, "interaction_radius", 2.0))

    @property
    def interact_camera(self) -> dict:
        """Console viewpoint — slightly above and in front of the station."""
        p = self._placed.pos
        yaw_r = math.radians(float(self._placed.yaw_deg))
        fwd   = np.array([math.cos(yaw_r), math.sin(yaw_r), 0.0])
        eye    = p - fwd * 1.2 + np.array([0, 0, 1.4])
        target = p + np.array([0, 0, 1.0])
        return {"eye": eye.tolist(), "target": target.tolist()}

    # ── Visibility ────────────────────────────────────────────────────────────

    def show(self, v: bool = True) -> None:
        self._visible = v

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def init_gl(self) -> None:
        if not _HAS_GL:
            return
        self._hud_panel.init_gl()
        self._gl_ready = True

    # ── Draw (3D pass — just a small console box for now) ─────────────────────

    def draw(self, MVP: np.ndarray, MV: np.ndarray,
             light_v: np.ndarray) -> None:
        """3D presence.  The actual interaction is via the HUD overlay."""
        # Minimal: no 3D geometry right now — the DutyStation mesh handles that.
        pass

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev) -> bool:
        """Route a pygame event.  Returns True if consumed."""
        if not self._visible:
            return False

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            # Offset into HUD space
            hx = (self.win_w - self.hud_w) // 2
            hy = (self.win_h - self.hud_h) // 2
            lx, ly = mx - hx, my - hy

            # Quadrant select
            for qname, rect in self._quadrant_rects.items():
                if rect.collidepoint(lx, ly):
                    self._state.selected_quadrant = qname
                    self._push_update()
                    return True

            # Coord system radios
            sel = self._state.selected_quadrant
            targets = ([sel] if sel != "all" else list(QUADRANT_NAMES))
            for qname, crects in self._coord_rects.items():
                if qname not in targets:
                    continue
                for cs_name, rect in crects.items():
                    if rect.collidepoint(lx, ly):
                        for t in targets:
                            self._state.quadrants[t].coord_system = (
                                CoordSystem(cs_name))
                        self._push_update()
                        return True

            # Slider drag start
            for (q, f), rect in self._slider_rects.items():
                if rect.collidepoint(lx, ly) and q in targets:
                    self._dragging = (q, f)
                    self._drag_x0  = mx
                    self._drag_v0  = self._get_field(q, f)
                    return True

        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self._dragging is not None:
                self._dragging = None
                return True

        if ev.type == pygame.MOUSEMOTION and self._dragging is not None:
            q, f = self._dragging
            dx   = ev.pos[0] - self._drag_x0
            vmin, vmax = self._field_range(f)
            span = vmax - vmin
            delta = dx / max(1, self.hud_w) * span
            new_v = float(np.clip(self._drag_v0 + delta, vmin, vmax))
            sel   = self._state.selected_quadrant
            for tq in ([sel] if sel != "all" else list(QUADRANT_NAMES)):
                self._set_field(tq, f, new_v)
            self._push_update()
            return True

        return False

    # ── Render (HUD overlay) ──────────────────────────────────────────────────

    def render(self, win_w: int, win_h: int) -> None:
        if not self._visible or not self._gl_ready:
            return
        self.win_w, self.win_h = win_w, win_h

        surf = pygame.Surface((self.hud_w, self.hud_h), pygame.SRCALPHA)
        surf.fill(_PANEL_BG)

        # Title
        _draw_label(surf, "ROOM CONFIGURATION", 12, 8, _ACCENT[:3], 16)
        pygame.draw.line(surf, _ACCENT[:3], (8, 28), (self.hud_w - 8, 28), 1)

        # ── Left column: quadrant select + coord system ───────────────────────
        lx, ly = 14, 38
        _draw_label(surf, "QUADRANT", lx, ly, _TEXT_DIM, 12)
        ly += 18
        self._quadrant_rects.clear()
        for qname in ("all",) + QUADRANT_NAMES:
            rect = _draw_radio(surf, lx, ly, qname,
                               self._state.selected_quadrant == qname)
            self._quadrant_rects[qname] = rect
            ly += 22

        ly += 8
        _draw_label(surf, "COORDINATES", lx, ly, _TEXT_DIM, 12)
        ly += 18
        sel = self._state.selected_quadrant
        ref_q = sel if sel != "all" else "NW"
        cur_cs = self._state.quadrants[ref_q].coord_system.value
        self._coord_rects = {ref_q: {}}
        for cs in (CoordSystem.RECTANGULAR, CoordSystem.CYLINDRICAL,
                   CoordSystem.SPHERICAL):
            rect = _draw_radio(surf, lx, ly, cs.value, cur_cs == cs.value)
            self._coord_rects[ref_q][cs.value] = rect
            ly += 22

        # ── Right column: dimension sliders ───────────────────────────────────
        rx   = self.hud_w // 2 + 10
        ry   = 38
        sw   = self.hud_w // 2 - 80   # slider track width

        sel      = self._state.selected_quadrant
        ref_q    = sel if sel != "all" else "NW"
        qcfg     = self._state.quadrants[ref_q]
        targets  = ([sel] if sel != "all" else list(QUADRANT_NAMES))

        _draw_label(surf, "DIMENSIONS", rx, ry, _TEXT_DIM, 12)
        ry += 18

        self._slider_rects.clear()
        sliders: List[Tuple[str, str, float, float, float, str]] = []

        if qcfg.coord_system == CoordSystem.RECTANGULAR:
            sliders = [
                ("width",  "width_m",  1.0, 30.0, qcfg.width_m,  "m"),
                ("depth",  "depth_m",  1.0, 30.0, qcfg.depth_m,  "m"),
                ("height", "height_m", 1.5, 12.0, qcfg.height_m, "m"),
            ]
        elif qcfg.coord_system == CoordSystem.CYLINDRICAL:
            sliders = [
                ("radius", "radius_m", 0.5, 15.0, qcfg.radius_m, "m"),
                ("height", "height_m", 1.5, 12.0, qcfg.height_m, "m"),
            ]
        else:  # spherical
            sliders = [
                ("radius", "radius_m", 0.5, 15.0, qcfg.radius_m, "m"),
            ]

        for (slabel, fname, vmin, vmax, val, unit) in sliders:
            rect = _draw_slider(surf, rx, ry, sw, val, vmin, vmax, slabel, unit)
            for tq in targets:
                self._slider_rects[(tq, fname)] = rect.move(0, 0)  # copy per target
            ry += 34

        ry += 8
        _draw_label(surf, "ROOM GRID", rx, ry, _TEXT_DIM, 12)
        ry += 18
        rect = _draw_slider(surf, rx, ry, sw,
                            self._state.grid_spacing_m, 0.25, 5.0,
                            "grid spacing", "m")
        self._slider_rects[("_room", "grid_spacing_m")] = rect
        ry += 34

        rect = _draw_slider(surf, rx, ry, sw,
                            self._state.wall_thickness_m, 0.10, 1.0,
                            "wall thickness", "m")
        self._slider_rects[("_room", "wall_thickness_m")] = rect

        # Upload and blit
        self._hud_panel.upload(surf)
        hx = (win_w - self.hud_w) // 2
        hy = (win_h - self.hud_h) // 2
        self._hud_panel.draw(hx, hy, self.hud_w, self.hud_h, win_w, win_h)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_field(self, q: str, f: str) -> float:
        if q == "_room":
            return getattr(self._state, f, 1.0)
        return getattr(self._state.quadrants.get(q, QuadrantConfig()), f, 1.0)

    def _set_field(self, q: str, f: str, v: float) -> None:
        if q == "_room":
            setattr(self._state, f, v)
        elif q in self._state.quadrants:
            setattr(self._state.quadrants[q], f, v)

    @staticmethod
    def _field_range(f: str) -> Tuple[float, float]:
        return {
            "width_m":          (1.0, 30.0),
            "depth_m":          (1.0, 30.0),
            "height_m":         (1.5, 12.0),
            "radius_m":         (0.5, 15.0),
            "grid_spacing_m":   (0.25,  5.0),
            "wall_thickness_m": (0.10,  1.0),
        }.get(f, (0.0, 10.0))

    def _push_update(self) -> None:
        """Push current state back to the workspace so room mesh rebuilds."""
        if self._workspace is None:
            return
        new_dims = self._state.effective_room_dims()
        if hasattr(self._workspace, "room_cfg"):
            self._workspace.room_cfg.update(new_dims)
        if hasattr(self._workspace, "_dirty"):
            self._workspace._dirty = True
