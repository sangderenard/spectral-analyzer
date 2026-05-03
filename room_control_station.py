"""room_control_station.py
==========================
HUD renderer for the "room_control" duty-station module.

Layout (interact mode, full window)
────────────────────────────────────────────────────────────────────────
  ┌──────────────┬─────────────────────────────────┬───────────────────┐
  │  LEFT PANEL  │        CENTER (stub)             │   RIGHT PANEL     │
  │  Lighting    │   ROOM MAP — coming soon         │   Environment     │
  │  knobs       │                                  │   Security knobs  │
  └──────────────┴─────────────────────────────────┴───────────────────┘

Panel data is loaded from the YAML configs under:
  configs/duty_stations/room_control/
    left_controls.yaml
    center_controls.yaml   (status: stub)
    right_controls.yaml

Typical usage
─────────────
    from room_control_station import RoomControlStation

    hud = RoomControlStation.from_yaml(
        "configs/duty_stations/room_control/station.yaml"
    )
    hud.build_gl()           # after GL context ready

    # attach to a DutyStation:
    station.menu = hud

    # in render loop:
    hud.render_hud(win_w, win_h)

    # in event loop:
    if hud.handle_event(ev):
        pass  # consumed

    # read current knob state:
    state = hud.state       # dict {knob_name: current_value}
"""
from __future__ import annotations

import ctypes
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _yaml = None
    _HAS_YAML = False

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_BLEND, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER,
        GL_LINEAR, GL_ONE_MINUS_SRC_ALPHA, GL_RGBA, GL_SRC_ALPHA,
        GL_STATIC_DRAW, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER, GL_TRIANGLES, GL_TRUE,
        GL_UNSIGNED_BYTE, GL_VERTEX_SHADER,
        glBindBuffer, glBindTexture, glBindVertexArray, glBlendFunc,
        glBufferData, glDeleteTextures, glDisable, glDrawArrays,
        glEnable, glEnableVertexAttribArray, glGenBuffers, glGenTextures,
        glGenVertexArrays, glGetUniformLocation, glTexImage2D, glTexParameteri,
        glUniform1i, glUniform2f, glUseProgram, glVertexAttribPointer,
        glViewport,
    )
    from OpenGL.GL import shaders as _gl_shaders
    _HAS_GL = True
except ImportError:
    _HAS_GL = False


# ─────────────────────────────────────────────────────────────────────────────
# GLSL: HUD panel texture overlay
# ─────────────────────────────────────────────────────────────────────────────

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
in vec2 vUV;
uniform sampler2D uTex;
out vec4 FragColor;
void main() { FragColor = texture(uTex, vUV); }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    if not _HAS_YAML:
        raise RuntimeError("PyYAML is required")
    with open(path, "r", encoding="utf-8") as fh:
        return _yaml.safe_load(fh) or {}


def _compile(vs: str, fs: str) -> int:
    return _gl_shaders.compileProgram(
        _gl_shaders.compileShader(vs, GL_VERTEX_SHADER),
        _gl_shaders.compileShader(fs, GL_FRAGMENT_SHADER),
    )


def _surface_to_tex(surf: pygame.Surface, existing_tex: Optional[int] = None) -> int:
    w, h = surf.get_size()
    raw  = pygame.image.tobytes(surf, "RGBA", True)
    if existing_tex is not None:
        tex = existing_tex
        glBindTexture(GL_TEXTURE_2D, tex)
    else:
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def _make_quad_vao(x: int, y: int, w: int, h: int):
    """Create a VAO/VBO for a 2-D screen-space quad [x, y, w, h].
    Vertex format: (px, py, u, v)  float32.
    Returns (vao, vbo).
    """
    verts = np.array([
        x,     y,     0.0, 1.0,
        x + w, y,     1.0, 1.0,
        x + w, y + h, 1.0, 0.0,
        x,     y,     0.0, 1.0,
        x + w, y + h, 1.0, 0.0,
        x,     y + h, 0.0, 0.0,
    ], np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts.tobytes(), GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    glBindVertexArray(0)
    return vao, vbo


# ─────────────────────────────────────────────────────────────────────────────
# Knob value state
# ─────────────────────────────────────────────────────────────────────────────

def _default_state(sections: list) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for sec in sections:
        for k in sec.get("knobs", []):
            state[k["name"]] = k.get("default")
    return state


def _clamp_float(val: float, k: dict) -> float:
    lo  = float(k.get("low",  0.0))
    hi  = float(k.get("high", 1.0))
    stp = float(k.get("step", 0.01))
    val = round(round(val / stp) * stp, 10)
    return float(np.clip(val, lo, hi))


# ─────────────────────────────────────────────────────────────────────────────
# Knob panel renderer (data-driven from sections YAML)
# ─────────────────────────────────────────────────────────────────────────────

_BG      = (18, 20, 28, 240)
_HDR_BG  = (28, 32, 44)
_ITEM_BG = (22, 25, 35)
_SEL_BG  = (38, 55, 90)
_TEXT    = (210, 220, 230)
_DIM     = (120, 130, 145)
_ACCENT  = (60, 110, 200)


class _KnobPanel:
    """Data-driven knob panel rendered into a pygame.Surface.

    Parameters
    ----------
    sections : list
        Parsed YAML ``sections`` list (each entry has ``id``, ``label``,
        ``knobs`` list).
    state : dict
        Shared mutable dict mapping knob name → current value.
    title : str
        Header text.
    accent_rgb : tuple
        (r, g, b) 0–255 accent colour for the panel header and highlights.
    """

    ROW_H  = 22
    PAD    = 6
    HDR_H  = 20
    BAR_H  = 6

    def __init__(self, sections: list, state: dict,
                 title: str = "PANEL",
                 accent_rgb: Tuple[int, int, int] = (60, 110, 200)):
        pygame.font.init()
        self._sections  = sections
        self.state      = state
        self._title     = title
        self._accent    = tuple(int(c * 255) if isinstance(c, float) else int(c)
                                for c in accent_rgb)
        self._font      = pygame.font.SysFont("monospace", 13)
        self._font_s    = pygame.font.SysFont("monospace", 11)
        self._scroll    = 0
        self._knob_rects: Dict[str, pygame.Rect] = {}   # name → hit rect
        self._hover_knob: Optional[str] = None
        # Cache knobs by name for fast lookup
        self._knobs: Dict[str, dict] = {}
        for sec in sections:
            for k in sec.get("knobs", []):
                self._knobs[k["name"]] = k

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, w: int, h: int) -> pygame.Surface:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(_BG)
        self._knob_rects = {}
        y = 0

        # Title bar
        pygame.draw.rect(surf, self._accent, pygame.Rect(0, 0, w, self.HDR_H + 4))
        t = self._font.render(f"  {self._title}", True, _TEXT)
        surf.blit(t, (self.PAD, 4))
        y = self.HDR_H + 4 + 2

        for sec in self._sections:
            # Section header
            pygame.draw.rect(surf, _HDR_BG, pygame.Rect(0, y - self._scroll, w, self.HDR_H))
            lbl = self._font_s.render(f"  {sec.get('label', sec['id'])}", True, (160, 180, 200))
            surf.blit(lbl, (self.PAD, y - self._scroll + 3))
            y += self.HDR_H + 2

            for k in sec.get("knobs", []):
                ky = y - self._scroll
                if ky + self.ROW_H * 2 < 0 or ky > h:
                    y += self.ROW_H * 2 + 2
                    continue

                name  = k["name"]
                dtype = k.get("dtype", "float")
                val   = self.state.get(name, k.get("default"))

                # Background row
                hov = (name == self._hover_knob)
                bg  = (30, 38, 52) if hov else _ITEM_BG
                pygame.draw.rect(surf, bg, pygame.Rect(0, ky, w, self.ROW_H * 2 + 2))

                # Label
                t_lbl = self._font_s.render(k.get("label", name), True, _DIM)
                surf.blit(t_lbl, (self.PAD, ky + 2))

                # Value display
                val_str = self._format_value(k, val, dtype)
                t_val   = self._font.render(val_str, True, _TEXT)
                surf.blit(t_val, (self.PAD, ky + 2 + self._font_s.get_height() + 1))

                # Float: progress bar
                if dtype == "float":
                    lo  = float(k.get("low",  0.0))
                    hi  = float(k.get("high", 1.0))
                    bar_w = w - 2 * self.PAD
                    frac  = (float(val) - lo) / max(hi - lo, 1e-12)
                    bar_x = self.PAD
                    bar_y = ky + self.ROW_H * 2 - self.BAR_H - 2
                    pygame.draw.rect(surf, (40, 44, 58),
                                     pygame.Rect(bar_x, bar_y, bar_w, self.BAR_H))
                    fill_w = max(2, int(bar_w * np.clip(frac, 0, 1)))
                    pygame.draw.rect(surf, self._accent,
                                     pygame.Rect(bar_x, bar_y, fill_w, self.BAR_H))

                # Bool: mini toggle indicator
                elif dtype == "bool":
                    tx = w - 28
                    ty = ky + 4
                    col = (80, 180, 100) if val else (80, 80, 90)
                    pygame.draw.rect(surf, col, pygame.Rect(tx, ty, 20, 14))
                    tl = self._font_s.render("ON" if val else "OFF", True, _TEXT)
                    surf.blit(tl, (tx + (20 - tl.get_width()) // 2,
                                   ty + (14 - tl.get_height()) // 2))

                # Choice: arrow indicator
                elif dtype == "choice":
                    tx = w - 14
                    ty = ky + 6
                    pygame.draw.polygon(surf, self._accent,
                                        [(tx, ty), (tx + 8, ty), (tx + 4, ty + 6)])

                # Hit rect for interactions
                self._knob_rects[name] = pygame.Rect(0, ky, w, self.ROW_H * 2 + 2)
                y += self.ROW_H * 2 + 2

            y += 4  # inter-section gap

        # Scrollbar
        content_h = y
        if content_h > h:
            bar_h = max(20, int(h * h / content_h))
            bar_y = int(self._scroll * (h - bar_h) / max(content_h - h, 1))
            pygame.draw.rect(surf, (50, 55, 70),
                             pygame.Rect(w - 6, 0, 6, h))
            pygame.draw.rect(surf, self._accent,
                             pygame.Rect(w - 6, bar_y, 6, bar_h))

        return surf

    def _format_value(self, k: dict, val: Any, dtype: str) -> str:
        if dtype == "float" and val is not None:
            fmt = k.get("fmt", ".2f")
            unit = k.get("unit", "")
            return f"{float(val):{fmt}} {unit}".strip()
        if dtype == "bool":
            return ""   # shown as mini-toggle
        if dtype == "choice":
            return str(val)
        return str(val) if val is not None else "—"

    # ── Events ────────────────────────────────────────────────────────────────

    def handle_event(self, ev,
                     x_off: int = 0, y_off: int = 0) -> bool:
        """Returns True if the event was consumed.

        Parameters
        ----------
        x_off, y_off : int
            Screen-space offset of this panel's top-left corner.
        """
        if ev.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            # Is the pointer over the right column for this panel?  Caller checks.
            self._scroll = max(0, self._scroll - ev.y * 24)
            # Adjust value if hovering a float knob
            if self._hover_knob and self._hover_knob in self._knobs:
                k = self._knobs[self._hover_knob]
                if k.get("dtype") == "float":
                    delta = float(k.get("step", 0.01)) * ev.y
                    cur   = float(self.state.get(k["name"], k.get("default", 0.0)))
                    self.state[k["name"]] = _clamp_float(cur + delta, k)
                    return True
            return False

        if ev.type == pygame.MOUSEMOTION:
            mx, my = ev.pos
            lx, ly = mx - x_off, my - y_off
            self._hover_knob = None
            for name, r in self._knob_rects.items():
                if r.collidepoint(lx, ly):
                    self._hover_knob = name
                    break
            return False

        if ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            lx, ly = mx - x_off, my - y_off
            for name, r in self._knob_rects.items():
                if r.collidepoint(lx, ly):
                    k = self._knobs.get(name)
                    if k is None:
                        return True
                    dtype = k.get("dtype", "float")
                    if dtype == "bool":
                        self.state[name] = not bool(self.state.get(name, False))
                        return True
                    elif dtype == "choice":
                        choices = list(k.get("choices", []))
                        cur = self.state.get(name, choices[0] if choices else None)
                        if choices:
                            idx = choices.index(cur) if cur in choices else 0
                            # left click → advance, right click → reverse
                            direction = -1 if ev.button == 3 else 1
                            self.state[name] = choices[(idx + direction) % len(choices)]
                        return True
                    # float: click resets to default
                    elif dtype == "float" and ev.button == 3:
                        self.state[name] = k.get("default")
                        return True
                    return True

        return False


# ─────────────────────────────────────────────────────────────────────────────
# Center panel (stub)
# ─────────────────────────────────────────────────────────────────────────────

class _CenterStubPanel:
    def render(self, w: int, h: int) -> pygame.Surface:
        pygame.font.init()
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((14, 16, 22, 240))
        font = pygame.font.SysFont("monospace", 16)
        font_s = pygame.font.SysFont("monospace", 12)

        # Title bar
        pygame.draw.rect(surf, (28, 32, 44), pygame.Rect(0, 0, w, 28))
        t = font.render("  ROOM MAP", True, (210, 220, 230))
        surf.blit(t, (8, 6))

        # Stub notice
        msg_lines = [
            "[ MAP VIEW ]",
            "",
            "Coming soon — interactive room",
            "floor-plan and object placement",
            "overlay will appear here.",
            "",
            "Left panel:  Lighting controls",
            "Right panel: Environment controls",
        ]
        y = h // 3
        for line in msg_lines:
            t = font_s.render(line, True, (80, 100, 130))
            surf.blit(t, ((w - t.get_width()) // 2, y))
            y += font_s.get_height() + 4

        # Decorative border
        pygame.draw.rect(surf, (40, 50, 70), pygame.Rect(0, 0, w, h), 1)
        return surf


# ─────────────────────────────────────────────────────────────────────────────
# RoomControlStation
# ─────────────────────────────────────────────────────────────────────────────

class RoomControlStation:
    """HUD for the room_control duty-station module.

    Exposes the ``show_hud`` / ``render_hud`` / ``handle_event`` interface
    expected by ``DutyStation.menu``.
    """

    LEFT_W  = 280
    RIGHT_W = 280

    def __init__(self, station_cfg: dict, left_cfg: dict,
                 center_cfg: dict, right_cfg: dict):
        self._cfg   = station_cfg
        self._left_cfg   = left_cfg
        self._center_cfg = center_cfg
        self._right_cfg  = right_cfg

        lay = station_cfg.get("panels", {}).get("layout", {})
        self.LEFT_W  = int(lay.get("left_width",  self.LEFT_W))
        self.RIGHT_W = int(lay.get("right_width", self.RIGHT_W))

        # Build initial knob state from defaults
        left_secs  = left_cfg.get("sections",  [])
        right_secs = right_cfg.get("sections", [])
        self.state: Dict[str, Any] = {}
        self.state.update(_default_state(left_secs))
        self.state.update(_default_state(right_secs))

        # Accent colour (from left panel config by default)
        def _accent(cfg):
            c = cfg.get("panel", {}).get("accent_rgb", [0.0, 0.28, 0.78])
            return tuple(int(x * 255) for x in c)

        self._left_panel   = _KnobPanel(
            left_secs, self.state,
            title=left_cfg.get("panel", {}).get("title", "LEFT"),
            accent_rgb=_accent(left_cfg))
        self._right_panel  = _KnobPanel(
            right_secs, self.state,
            title=right_cfg.get("panel", {}).get("title", "RIGHT"),
            accent_rgb=_accent(right_cfg))
        self._center_panel = _CenterStubPanel()

        # HUD visibility
        self._hud_visible = False

        # GL handles
        self._prog:        Optional[int] = None
        self._tex_left:    Optional[int] = None
        self._tex_center:  Optional[int] = None
        self._tex_right:   Optional[int] = None
        self._vao_left     = self._vbo_left     = None
        self._vao_center   = self._vbo_center   = None
        self._vao_right    = self._vbo_right    = None
        self._last_size    = (0, 0)
        self._gl_ready     = False

    # ── Class-method constructors ─────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, station_yaml_path: str) -> "RoomControlStation":
        """Load from station.yaml; sibling panel YAMLs resolved relative to it."""
        station_cfg = _load_yaml(station_yaml_path)
        base_dir    = os.path.dirname(os.path.abspath(station_yaml_path))
        panels_cfg  = station_cfg.get("panels", {})

        def _load_panel(key: str, fallback_name: str) -> dict:
            rel = panels_cfg.get(key, fallback_name)
            path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
            try:
                return _load_yaml(path)
            except Exception as exc:
                print(f"[RoomControlStation] could not load {path}: {exc}")
                return {}

        left_cfg   = _load_panel("left",   "left_controls.yaml")
        center_cfg = _load_panel("center", "center_controls.yaml")
        right_cfg  = _load_panel("right",  "right_controls.yaml")
        return cls(station_cfg, left_cfg, center_cfg, right_cfg)

    @classmethod
    def from_yaml_safe(cls, path: str) -> Optional["RoomControlStation"]:
        try:
            return cls.from_yaml(path)
        except Exception as exc:
            print(f"[RoomControlStation] load failed: {exc}")
            return None

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def build_gl(self):
        if not _HAS_GL:
            return
        self._prog = _compile(_HUD_VS, _HUD_FS)
        self._gl_ready = True

    # ── HUD visibility ────────────────────────────────────────────────────────

    def show_hud(self, visible: bool):
        self._hud_visible = visible

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_hud(self, win_w: int, win_h: int):
        """Draw the full three-panel HUD.  Call in the GL render loop."""
        if not self._hud_visible or not self._gl_ready:
            return

        # Rebuild panel textures if window size changed
        if (win_w, win_h) != self._last_size:
            self._rebuild_quads(win_w, win_h)
            self._last_size = (win_w, win_h)

        center_w = win_w - self.LEFT_W - self.RIGHT_W
        h        = win_h

        # Render pygame surfaces → textures
        surf_l = self._left_panel.render(self.LEFT_W, h)
        surf_c = self._center_panel.render(center_w, h)
        surf_r = self._right_panel.render(self.RIGHT_W, h)

        self._tex_left   = _surface_to_tex(surf_l, self._tex_left)
        self._tex_center = _surface_to_tex(surf_c, self._tex_center)
        self._tex_right  = _surface_to_tex(surf_r, self._tex_right)

        # Draw
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self._prog)
        loc_res = glGetUniformLocation(self._prog, b"uRes")
        glUniform2f(loc_res, float(win_w), float(win_h))
        loc_tex = glGetUniformLocation(self._prog, b"uTex")
        glUniform1i(loc_tex, 0)

        for vao, tex in [
            (self._vao_left,   self._tex_left),
            (self._vao_center, self._tex_center),
            (self._vao_right,  self._tex_right),
        ]:
            if vao is None or tex is None:
                continue
            glBindTexture(GL_TEXTURE_2D, tex)
            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D, 0)

        glUseProgram(0)
        glDisable(GL_BLEND)

    def _rebuild_quads(self, win_w: int, win_h: int):
        """Rebuild screen-space quads for the three panels."""
        from OpenGL.GL import glDeleteBuffers, glDeleteVertexArrays
        for attr_vao, attr_vbo in [("_vao_left",   "_vbo_left"),
                                    ("_vao_center", "_vbo_center"),
                                    ("_vao_right",  "_vbo_right")]:
            if getattr(self, attr_vao) is not None:
                glDeleteVertexArrays(1, [getattr(self, attr_vao)])
                glDeleteBuffers(1, [getattr(self, attr_vbo)])

        center_w = win_w - self.LEFT_W - self.RIGHT_W
        h        = win_h

        self._vao_left,   self._vbo_left   = _make_quad_vao(0,                      0, self.LEFT_W, h)
        self._vao_center, self._vbo_center = _make_quad_vao(self.LEFT_W,             0, center_w,   h)
        self._vao_right,  self._vbo_right  = _make_quad_vao(self.LEFT_W + center_w,  0, self.RIGHT_W, h)

        # Upload textures at new size if panels already exist
        if self._tex_left is not None:
            glDeleteTextures([self._tex_left, self._tex_center, self._tex_right])
        self._tex_left = self._tex_center = self._tex_right = None

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev) -> bool:
        """Route events to the appropriate panel.
        Returns True if the event was consumed.
        """
        if not self._hud_visible:
            return False

        win_w, win_h = pygame.display.get_surface().get_size()
        center_w     = win_w - self.LEFT_W - self.RIGHT_W

        x_left   = 0
        x_center = self.LEFT_W
        x_right  = self.LEFT_W + center_w

        def _in_panel(x_off: int, pw: int) -> bool:
            """Is the mouse event within panel bounds?"""
            if ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                           pygame.MOUSEMOTION):
                mx = ev.pos[0]
                return x_off <= mx < x_off + pw
            if ev.type == pygame.MOUSEWHEEL:
                mx, _ = pygame.mouse.get_pos()
                return x_off <= mx < x_off + pw
            return False

        if _in_panel(x_left, self.LEFT_W):
            return self._left_panel.handle_event(ev, x_off=x_left)
        if _in_panel(x_right, self.RIGHT_W):
            return self._right_panel.handle_event(ev, x_off=x_right)
        if _in_panel(x_center, center_w):
            # center panel is stub — absorb but don't process
            if ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                return True

        return False

    # ── State access ──────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Return a copy of the current knob state dict."""
        return dict(self.state)

    def set_value(self, name: str, value: Any):
        """Programmatically set a knob value."""
        if name in self.state:
            k = self._left_panel._knobs.get(name) \
                or self._right_panel._knobs.get(name)
            if k and k.get("dtype") == "float":
                value = _clamp_float(float(value), k)
            self.state[name] = value

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        vis = "visible" if self._hud_visible else "hidden"
        return f"RoomControlStation(hud={vis}, state_keys={list(self.state)})"
