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

from controls import KnobSpec, Panel
from room_tile_editor import (
    _CELL_SIZE_M,
    RoomTileLibraryPanel,
    RoomTileWorkspace,
    load_room_tile_presets,
)

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


def _knobspec_from_yaml_sections(sections: list) -> list[KnobSpec]:
    knobs: list[KnobSpec] = []
    for sec in sections or []:
        group = str(sec.get("label", sec.get("id", "")))
        for raw in sec.get("knobs", []) or []:
            name = str(raw.get("name", ""))
            if not name:
                continue
            dtype = str(raw.get("dtype", "float"))
            choices = [str(choice) for choice in raw.get("choices", []) or []]
            default = raw.get("default")
            if dtype == "choice":
                default = choices.index(default) if default in choices else 0
            knobs.append(KnobSpec(
                name,
                str(raw.get("label", name)),
                dtype,
                default,
                float(raw.get("low", 0.0) or 0.0),
                float(raw.get("high", max(0, len(choices) - 1)) or 0.0),
                float(raw.get("step", 1.0 if dtype == "choice" else 0.0) or 0.0),
                str(raw.get("unit", "")),
                choices,
                bool(raw.get("is_log", False)),
                str(raw.get("group", group)),
                str(raw.get("fmt", ".3g")),
            ))
    return knobs


def _title_from_yaml_cfg(cfg: dict, fallback: str) -> str:
    if not isinstance(cfg, dict):
        return fallback
    return str(cfg.get("panel", {}).get("title", fallback))


def _panel_from_yaml_cfg(name: str, cfg: dict, fallback_label: str) -> Panel:
    return Panel(
        name,
        _title_from_yaml_cfg(cfg, fallback_label),
        knobs=_knobspec_from_yaml_sections(cfg.get("sections", [])),
    )


def _doc_knob_values_for_specs(panel: Panel, state: dict) -> dict[str, Any]:
    values: dict[str, Any] = {}

    def visit(node: Panel) -> None:
        for knob in node.knobs:
            name = getattr(knob, "name", "")
            value = state.get(name, getattr(knob, "default", None))
            choices = list(getattr(knob, "choices", []) or [])
            if choices:
                try:
                    value = choices.index(str(value))
                except ValueError:
                    try:
                        value = int(value)
                    except Exception:
                        value = int(getattr(knob, "default", 0) or 0)
            values[name] = value
        for sub in node.panels:
            visit(sub)

    visit(panel)
    return values


_ROOM_PALETTE_TOOLS = ["create", "move", "delete"]
_ROOM_PALETTE_CATEGORIES = [
    "room_tiles",
    "lights",
    "doors",
    "windows",
    "cameras",
    "duty_stations",
    "duty_modules",
]
_ROOM_VIEWS = ["plan", "front", "side", "tile_editor"]
_ROOM_EDIT_SCOPES = ["archetype", "placed"]


def _choice_knob(
    name: str,
    label: str,
    choices: list[str],
    *,
    default: int = 0,
    group: str = "",
) -> KnobSpec:
    return KnobSpec(
        name,
        label,
        "choice",
        int(default),
        0.0,
        float(max(0, len(choices) - 1)),
        1.0,
        "",
        choices,
        False,
        group,
        ".0f",
    )


def _choice_index(value: Any, choices: list[str]) -> int:
    try:
        return choices.index(str(value))
    except ValueError:
        try:
            idx = int(value)
            return int(np.clip(idx, 0, max(0, len(choices) - 1)))
        except Exception:
            return 0


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


class _LibraryPalettePanel:
    """Specialized panel: filesystem-backed parts list + tool selector row.

    Uses shared HUD state keys:
      - ``palette_tool``     : one of ``create``, ``move``, ``delete``
      - ``palette_selected`` : currently selected item id
    """

    PAD = 6
    HDR_H = 24
    ROW_H = 20
    TOOL_H = 24

    def __init__(self, state: dict,
                 title: str = "PARTS",
                 accent_rgb: Tuple[int, int, int] = (60, 110, 200),
                 library_items: list | None = None):
        pygame.font.init()
        self.state = state
        self._title = title
        self._accent = tuple(int(c * 255) if isinstance(c, float) else int(c)
                             for c in accent_rgb)
        self._font = pygame.font.SysFont("monospace", 13)
        self._font_s = pygame.font.SysFont("monospace", 11)
        self._scroll = 0
        self._items = list(library_items or [])
        self._tool_rects: dict[str, pygame.Rect] = {}
        self._item_rects: dict[str, pygame.Rect] = {}
        self._content_h = 0

        if "palette_tool" not in self.state:
            self.state["palette_tool"] = "create"
        if "palette_selected" not in self.state and self._items:
            self.state["palette_selected"] = str(self._items[0].get("id", ""))

    def render(self, w: int, h: int) -> pygame.Surface:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(_BG)
        self._tool_rects = {}
        self._item_rects = {}

        pygame.draw.rect(surf, self._accent, pygame.Rect(0, 0, w, self.HDR_H))
        t = self._font.render(f"  {self._title}", True, _TEXT)
        surf.blit(t, (self.PAD, 4))

        y = self.HDR_H + 4
        tool_w = max(30, (w - self.PAD * 2 - 4) // 3)
        for i, tool in enumerate(("create", "move", "delete")):
            tr = pygame.Rect(self.PAD + i * (tool_w + 2), y, tool_w, self.TOOL_H)
            self._tool_rects[tool] = tr
            active = (self.state.get("palette_tool") == tool)
            bg = self._accent if active else (28, 32, 44)
            pygame.draw.rect(surf, bg, tr, border_radius=3)
            pygame.draw.rect(surf, (70, 80, 100), tr, 1, border_radius=3)
            lbl = self._font_s.render(tool.upper(), True, _TEXT)
            surf.blit(lbl, (tr.x + (tr.w - lbl.get_width()) // 2,
                            tr.y + (tr.h - lbl.get_height()) // 2))
        y += self.TOOL_H + 6

        list_top = y
        cur_y = list_top - self._scroll
        selected = str(self.state.get("palette_selected", ""))
        for it in self._items:
            iid = str(it.get("id", ""))
            cat = str(it.get("category", "misc"))
            lbl = str(it.get("label", iid))
            r = pygame.Rect(0, cur_y, w, self.ROW_H)
            if cur_y + self.ROW_H >= list_top and cur_y < h:
                bg = _SEL_BG if iid == selected else _ITEM_BG
                pygame.draw.rect(surf, bg, r)
                s = self._font_s.render(f"{cat}: {lbl}", True, _TEXT)
                surf.blit(s, (self.PAD, cur_y + 3))
                self._item_rects[iid] = r
            cur_y += self.ROW_H

        self._content_h = max(0, len(self._items) * self.ROW_H)
        list_h = max(1, h - list_top)
        max_scroll = max(0, self._content_h - list_h)
        self._scroll = int(np.clip(self._scroll, 0, max_scroll))

        if self._content_h > list_h:
            bar_h = max(20, int(list_h * list_h / self._content_h))
            bar_y = list_top + int(self._scroll * (list_h - bar_h) / max(max_scroll, 1))
            pygame.draw.rect(surf, (50, 55, 70), pygame.Rect(w - 6, list_top, 6, list_h))
            pygame.draw.rect(surf, self._accent, pygame.Rect(w - 6, bar_y, 6, bar_h))

        return surf

    def handle_event(self, ev, x_off: int = 0, y_off: int = 0) -> bool:
        if ev.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            lx, ly = mx - x_off, my - y_off
            if lx < 0 or ly < self.HDR_H + self.TOOL_H + 4:
                return False
            list_h = max(1, pygame.display.get_surface().get_height() - (self.HDR_H + self.TOOL_H + 10))
            max_scroll = max(0, self._content_h - list_h)
            self._scroll = int(np.clip(self._scroll - ev.y * self.ROW_H, 0, max_scroll))
            return True

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            lx, ly = mx - x_off, my - y_off
            for tool, r in self._tool_rects.items():
                if r.collidepoint(lx, ly):
                    self.state["palette_tool"] = tool
                    return True
            for iid, r in self._item_rects.items():
                if r.collidepoint(lx, ly):
                    self.state["palette_selected"] = iid
                    return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Center panel (stub)
# ─────────────────────────────────────────────────────────────────────────────

class _CenterTabPanel:
    """Center panel with a tab bar (RoutingGridView pattern) and a _KnobPanel
    per tab.  Each tab entry is a dict::

        {"key": str, "label": str, "sections": list,
         "title": str (opt), "accent_rgb": tuple (opt)}

    ``active_tab`` exposes the current tab key so ``DutyStationHUD.render_hud``
    can gate special behaviour (e.g. delegating GL rendering for a "views" tab).
    """

    _TAB_H   = 22
    _TAB_W   = 72
    _TAB_PAD = 4

    def __init__(self, tabs: list, state: dict):
        pygame.font.init()
        self._tabs   = tabs or [{"key": "main", "label": "MAIN", "sections": []}]
        self._keys   = [t["key"]   for t in self._tabs]
        self._active = self._keys[0] if self._keys else ""
        self._tab_rects: list = []
        self._font   = pygame.font.SysFont("monospace", 12)
        self._w = self._h = 0

        # One _KnobPanel per tab — shares the same state dict
        self._panels: Dict[str, _KnobPanel] = {}
        for t in self._tabs:
            secs   = t.get("sections", [])
            accent = t.get("accent_rgb", (60, 110, 200))
            title  = t.get("title", t["label"])
            self._panels[t["key"]] = _KnobPanel(
                secs, state, title=title, accent_rgb=accent)

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def active_tab(self) -> str:
        return self._active

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, w: int, h: int) -> pygame.Surface:
        self._w, self._h = w, h
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(_BG)

        # Tab bar — identical colours / geometry to RoutingGridView
        self._tab_rects = []
        fh = self._font.get_height()
        for ti, t in enumerate(self._tabs):
            tr = pygame.Rect(
                self._TAB_PAD + ti * (self._TAB_W + 2),
                self._TAB_PAD,
                self._TAB_W,
                self._TAB_H - 2 * self._TAB_PAD,
            )
            self._tab_rects.append(tr)
            active = (t["key"] == self._active)
            bg = (40, 90, 160) if active else (28, 28, 38)
            pygame.draw.rect(surf, bg, tr, border_radius=3)
            pygame.draw.rect(surf, (60, 60, 85), tr, 1, border_radius=3)
            tc = (220, 235, 255) if active else (100, 100, 120)
            ts = self._font.render(t["label"], True, tc)
            surf.blit(ts, (tr.x + (tr.w - ts.get_width()) // 2,
                           tr.y + (tr.h - fh) // 2))

        # Active tab body — delegate to its _KnobPanel
        panel = self._panels.get(self._active)
        body_h = h - self._TAB_H
        if panel is not None and body_h > 0:
            psuf = panel.render(w, body_h)
            surf.blit(psuf, (0, self._TAB_H))

        return surf

    def render_tab_bar_only(self, w: int, h: int) -> pygame.Surface:
        """Return a surface with only the tab strip (rest fully transparent).

        Used by ``DutyStationHUD.render_hud`` when a tab delegates GL rendering
        to an external widget (e.g. the camera designer viewports).
        """
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        fh = self._font.get_height()
        self._tab_rects = []
        for ti, t in enumerate(self._tabs):
            tr = pygame.Rect(
                self._TAB_PAD + ti * (self._TAB_W + 2),
                self._TAB_PAD,
                self._TAB_W,
                self._TAB_H - 2 * self._TAB_PAD,
            )
            self._tab_rects.append(tr)
            active = (t["key"] == self._active)
            bg = (40, 90, 160) if active else (28, 28, 38)
            pygame.draw.rect(surf, bg, tr, border_radius=3)
            pygame.draw.rect(surf, (60, 60, 85), tr, 1, border_radius=3)
            tc = (220, 235, 255) if active else (100, 100, 120)
            ts = self._font.render(t["label"], True, tc)
            surf.blit(ts, (tr.x + (tr.w - ts.get_width()) // 2,
                           tr.y + (tr.h - fh) // 2))
        return surf

    # ── Events ────────────────────────────────────────────────────────────────

    def handle_event(self, ev, x_off: int = 0, y_off: int = 0) -> bool:
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            lx, ly = mx - x_off, my - y_off
            if 0 <= ly < self._TAB_H:
                for ti, tr in enumerate(self._tab_rects):
                    if tr.collidepoint(lx, ly):
                        self._active = self._keys[ti]
                        return True

        # Route all other events to the active tab's _KnobPanel
        panel = self._panels.get(self._active)
        if panel is not None:
            return panel.handle_event(ev, x_off=x_off,
                                      y_off=y_off + self._TAB_H)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# DutyStationHUD  (generic base)
# ─────────────────────────────────────────────────────────────────────────────

class DutyStationHUD:
    """Generic 3-panel HUD for any duty-station module.

    Exposes the ``show_hud`` / ``render_hud`` / ``handle_event`` interface
    expected by ``DutyStation.menu``.  Build via ``from_sections`` for
    programmatic section dicts, or a subclass for YAML/config-driven loading.
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

        # Build initial knob state from defaults (all three panels contribute)
        left_secs  = left_cfg.get("sections",  [])
        right_secs = right_cfg.get("sections", [])
        self.state: Dict[str, Any] = {}
        self.state.update(_default_state(left_secs))
        self.state.update(_default_state(right_secs))
        # Center state: collect from all tabs or from legacy sections key
        _center_tabs = center_cfg.get("tabs") if isinstance(center_cfg, dict) else None
        if _center_tabs is not None:
            for _t in _center_tabs:
                self.state.update(_default_state(_t.get("sections", [])))
        else:
            self.state.update(_default_state(
                center_cfg.get("sections", []) if isinstance(center_cfg, dict) else []))

        # Accent colour (from left panel config by default)
        def _accent(cfg):
            c = cfg.get("panel", {}).get("accent_rgb", [0.0, 0.28, 0.78])
            return tuple(int(x * 255) for x in c)

        _left_panel_cfg = left_cfg.get("panel", {})
        if _left_panel_cfg.get("panel_type") == "library_palette":
            self._left_panel = _LibraryPalettePanel(
                self.state,
                title=_left_panel_cfg.get("title", "PARTS"),
                accent_rgb=_accent(left_cfg),
                library_items=_left_panel_cfg.get("library_items", []),
            )
        else:
            self._left_panel = _KnobPanel(
                left_secs, self.state,
                title=_left_panel_cfg.get("title", "LEFT"),
                accent_rgb=_accent(left_cfg))
        self._right_panel = _KnobPanel(
            right_secs, self.state,
            title=right_cfg.get("panel", {}).get("title", "RIGHT"),
            accent_rgb=_accent(right_cfg))

        # Center panel: tab-based if "tabs" key present, else wrap sections in one tab
        if _center_tabs is not None:
            self._center_panel = _CenterTabPanel(_center_tabs, self.state)
        else:
            _fallback_secs = (center_cfg.get("sections", [])
                              if isinstance(center_cfg, dict) else [])
            self._center_panel = _CenterTabPanel(
                [{"key": "main", "label": "MAIN", "sections": _fallback_secs,
                  "title": "CENTER"}],
                self.state)

        # Optional GL designer widget — set externally to delegate "views" tab rendering
        self._designer: Optional[object] = None

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
    def from_sections(
        cls,
        left_sections: list,
        right_sections: list,
        *,
        center_tabs: list | None = None,
        center_sections: list | None = None,
        left_panel: dict | None = None,
        left_title: str = "LEFT",
        right_title: str = "RIGHT",
        left_accent: tuple = (0.0, 0.28, 0.78),
        right_accent: tuple = (0.0, 0.50, 0.35),
        left_width: int = 280,
        right_width: int = 280,
        station_cfg: dict | None = None,
    ) -> "DutyStationHUD":
        """Construct a HUD directly from section dicts — no YAML required.

        ``center_tabs`` is a list of tab dicts::

            [{"key": str, "label": str, "sections": list,
              "title": str (opt), "accent_rgb": tuple (opt)}, ...]

        ``center_sections`` is a legacy shorthand: it is wrapped in a single
        "MAIN" tab when ``center_tabs`` is not provided.
        """
        def _make_panel_cfg(sections, title, accent_rgb, panel_extra=None):
            panel = {"title": title, "accent_rgb": list(accent_rgb)}
            if isinstance(panel_extra, dict):
                panel.update(panel_extra)
            return {
                "panel": panel,
                "sections": sections,
            }

        _station_cfg = station_cfg or {
            "panels": {"layout": {"left_width": left_width, "right_width": right_width}}
        }
        _left_cfg  = _make_panel_cfg(left_sections,  left_title,  left_accent,
                         panel_extra=left_panel)
        _right_cfg = _make_panel_cfg(right_sections, right_title, right_accent)

        if center_tabs is not None:
            _center_cfg = {"tabs": center_tabs}
        else:
            _center_cfg = {"sections": center_sections or []}

        return cls(_station_cfg, _left_cfg, _center_cfg, _right_cfg)

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml_safe(cls, path: str) -> Optional["DutyStationHUD"]:
        """Call ``cls.from_yaml(path)`` and return None on any error."""
        try:
            return cls.from_yaml(path)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[{cls.__name__}] load failed: {exc}")
            return None

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

        # Detect "views" tab with an attached GL designer: let the designer own
        # the center GL area and render only the tab bar strip as overlay.
        _views_active = (
            self._designer is not None
            and getattr(self._center_panel, "active_tab", None) == "views"
        )

        # Render pygame surfaces → textures
        surf_l = self._left_panel.render(self.LEFT_W, h)
        surf_r = self._right_panel.render(self.RIGHT_W, h)
        if _views_active:
            surf_c = self._center_panel.render_tab_bar_only(center_w, h)
        else:
            surf_c = self._center_panel.render(center_w, h)

        self._tex_left   = _surface_to_tex(surf_l, self._tex_left)
        self._tex_center = _surface_to_tex(surf_c, self._tex_center)
        self._tex_right  = _surface_to_tex(surf_r, self._tex_right)

        # When views tab is active the designer draws its GL scene first so
        # the HUD quads (drawn with blending) composite on top.
        if _views_active:
            self._designer.draw(win_w, win_h)  # type: ignore[union-attr]

        # Draw HUD quads
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
            if self._center_panel.handle_event(ev, x_off=x_center, y_off=0):
                return True
            _views_active = (
                self._designer is not None
                and getattr(self._center_panel, "active_tab", None) == "views"
                and hasattr(self._designer, "handle_event")
            )
            if _views_active and self._designer.handle_event(ev):  # type: ignore[union-attr]
                return True
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
            left_knobs = getattr(self._left_panel, "_knobs", {})
            right_knobs = getattr(self._right_panel, "_knobs", {})
            k = left_knobs.get(name) or right_knobs.get(name)
            if k and k.get("dtype") == "float":
                value = _clamp_float(float(value), k)
            self.state[name] = value

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        vis = "visible" if self._hud_visible else "hidden"
        return f"{self.__class__.__name__}(hud={vis}, state_keys={list(self.state)})"


# ─────────────────────────────────────────────────────────────────────────────
# RoomControlStation  (YAML-driven room duty-station HUD)
# ─────────────────────────────────────────────────────────────────────────────

class RoomControlStation(DutyStationHUD):
    """Room-control duty-station HUD.  Loads knob layout from YAML files."""

    def __init__(self, station_cfg: dict, left_cfg: dict,
                 center_cfg: dict, right_cfg: dict):
        super().__init__(station_cfg, left_cfg, center_cfg, right_cfg)
        self._host_station = None
        self._room_workspace = None
        self._room_station_cfg = station_cfg
        self._room_left_cfg = left_cfg

        room_cfg = station_cfg.get("room_editor", {})
        library_dir = str(room_cfg.get("preset_library_dir", ""))
        self._room_presets = load_room_tile_presets(library_dir)

        def _accent(cfg):
            c = cfg.get("panel", {}).get("accent_rgb", [0.0, 0.28, 0.78])
            return tuple(int(x * 255) for x in c)

        self._room_accent = _accent(left_cfg)
        self._left_panel = _KnobPanel(
            [],
            self.state,
            title="ROOM LIBRARY (LOCKED)",
            accent_rgb=self._room_accent,
        )
        self._center_panel = _CenterTabPanel(
            [{"key": "locked", "label": "LOCKED", "sections": [],
              "title": "BUILD ROOM STATION"}],
            self.state,
        )

    @property
    def panel_spec(self) -> Panel:
        """Controls hierarchy for the live room-control HUD widgets."""
        self._ensure_room_workspace()
        selected_items: list[str] = []
        if isinstance(self._left_panel, RoomTileLibraryPanel):
            selected_items = [preset.preset_id for preset in self._left_panel.filtered_items()]

        return Panel(
            "room_control_station",
            "Room Control",
            panels=[
                Panel(
                    "room_tile_library",
                    "ROOM LIBRARY",
                    knobs=[
                        _choice_knob("palette_tool", "Tool", _ROOM_PALETTE_TOOLS, group="Palette"),
                        _choice_knob("palette_category", "Category", _ROOM_PALETTE_CATEGORIES, group="Palette"),
                        _choice_knob("palette_selected", "Preset", selected_items, group="Palette"),
                        KnobSpec("tile_import_scale", "Import Scale", "float", 1.0, 0.1, 20.0, 0.1, "", [], False, "Tile Editor", ".2f"),
                        _choice_knob("tile_editor_scope", "Edit Scope", _ROOM_EDIT_SCOPES, group="Tile Editor"),
                        KnobSpec("tile_editor_selected_mesh", "Selected Mesh", "str", "", 0, 0, 0, "", [], False, "Tile Editor"),
                    ],
                    payload={
                        "source_panel": "RoomTileLibraryPanel",
                        "actions": [
                            {"key": "rotate_left", "label": "ROT L"},
                            {"key": "rotate_right", "label": "ROT R"},
                            {"key": "import_mesh", "label": "IMPORT"},
                        ],
                    },
                ),
                Panel(
                    "room_tile_workspace",
                    "ROOM MAP",
                    knobs=[
                        _choice_knob("room_view", "View", _ROOM_VIEWS, group="Map"),
                        KnobSpec("room_level", "Level", "int", 0, -64, 64, 1, "", [], False, "Map", ".0f"),
                        KnobSpec("room_width_cells", "Width", "int", 8, 2, 128, 1, "cells", [], False, "Room", ".0f"),
                        KnobSpec("room_depth_cells", "Depth", "int", 8, 2, 128, 1, "cells", [], False, "Room", ".0f"),
                        KnobSpec("room_station_x", "Station X", "int", 0, 0, 128, 1, "cells", [], False, "Station Anchor", ".0f"),
                        KnobSpec("room_station_y", "Station Y", "int", 0, 0, 128, 1, "cells", [], False, "Station Anchor", ".0f"),
                        KnobSpec("room_selected_instance", "Selected Instance", "str", "", 0, 0, 0, "", [], False, "Selection"),
                    ],
                    payload={
                        "source_panel": "RoomTileWorkspace",
                        "actions": [
                            {"key": "level:-", "label": "Level -"},
                            {"key": "level:+", "label": "Level +"},
                            {"key": "dimx:-", "label": "Room X -"},
                            {"key": "dimx:+", "label": "Room X +"},
                            {"key": "dimy:-", "label": "Room Y -"},
                            {"key": "dimy:+", "label": "Room Y +"},
                        ],
                    },
                ),
                _panel_from_yaml_cfg("room_environment", self._right_cfg, "ENVIRONMENT"),
            ],
        )

    @property
    def knob_values(self) -> dict[str, Any]:
        return _doc_knob_values_for_specs(self.panel_spec, self.state)

    def _ensure_room_workspace(self) -> bool:
        if self._room_workspace is not None:
            return True
        if self._host_station is None:
            return False
        if bool(getattr(self._host_station, "is_unfinished", False)):
            return False

        self._room_workspace = RoomTileWorkspace(
            self.state,
            self._room_presets,
            self._room_station_cfg,
            on_station_anchor_changed=self._on_station_anchor_changed,
        )
        self._left_panel = RoomTileLibraryPanel(
            self.state,
            self._room_presets,
            title="ROOM LIBRARY",
            accent_rgb=self._room_accent,
            on_rotate_left=self._room_workspace.rotate_selection_left,
            on_rotate_right=self._room_workspace.rotate_selection_right,
            on_import_mesh=self._room_workspace.import_mesh_dialog,
        )
        self._center_panel = self._room_workspace
        if self._gl_ready:
            self._room_workspace.build_gl()
            self._last_size = (0, 0)
        return True

    def bind_host_station(self, station):
        """Attach the live DutyStation instance driven by this HUD."""
        self._host_station = station
        self._ensure_room_workspace()

    def _on_station_anchor_changed(self, old_x: int, old_y: int,
                                   new_x: int, new_y: int):
        """Propagate tile-anchor moves into the live station world transform."""
        if self._host_station is None:
            return
        dx_cells = int(new_x - old_x)
        dy_cells = int(new_y - old_y)
        if dx_cells == 0 and dy_cells == 0:
            return
        delta = np.array([
            float(dx_cells) * float(_CELL_SIZE_M),
            float(dy_cells) * float(_CELL_SIZE_M),
            0.0,
        ], np.float64)
        cur = np.array(getattr(self._host_station, "world_position", [0.0, 0.0, 0.0]), np.float64)
        if hasattr(self._host_station, "set_world_position"):
            self._host_station.set_world_position(cur + delta, update_interact_camera=True)

    def build_gl(self):
        super().build_gl()
        if self._ensure_room_workspace() and self._room_workspace is not None:
            self._room_workspace.build_gl()

    def draw_world(self, mvp: np.ndarray, mv: np.ndarray,
                   light_v: np.ndarray, prog: Optional[int]):
        if self._ensure_room_workspace() and self._room_workspace is not None:
            self._room_workspace.draw_world(mvp, mv, light_v, prog)

    @classmethod
    def from_yaml(cls, station_yaml_path: str) -> "RoomControlStation":
        """Load from station.yaml; sibling panel YAMLs resolved relative to it."""
        station_cfg = _load_yaml(station_yaml_path)
        base_dir    = os.path.dirname(os.path.abspath(station_yaml_path))
        room_cfg = station_cfg.setdefault("room_editor", {})
        library_rel = room_cfg.get("preset_library", "presets")
        room_cfg["preset_library_dir"] = (
            library_rel if os.path.isabs(library_rel)
            else os.path.join(base_dir, library_rel)
        )
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


# ─────────────────────────────────────────────────────────────────────────────
# CameraDutyStationHUD  (camera duty-station HUD built from section dicts)
# ─────────────────────────────────────────────────────────────────────────────

class CameraDutyStationHUD(DutyStationHUD):
    """Camera duty-station HUD.  Sections are supplied programmatically
    (e.g. from ``_PlayerCameraPanel.camera_hud_sections()``).
    No YAML loading — call ``from_sections`` inherited from ``DutyStationHUD``.
    """
