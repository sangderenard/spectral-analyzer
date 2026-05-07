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

from controls import (
    KnobSpec,
    Panel,
    choice_knob,
    enqueue_action,
    get_action_registry,
    readonly_knob,
    stepper_knob,
)
from bass_viewer import ScrollableSubpanelList, ModularSubpanelSpec
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
            spec = KnobSpec(
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
            )
            # Set control_widget attribute based on dtype and explicit widget override
            widget = str(raw.get("widget", ""))
            if not widget:
                # Default widget based on dtype
                if dtype == "choice":
                    widget = "segmented"
                elif dtype in ("int", "float"):
                    widget = str(raw.get("widget", "stepper")) if raw.get("widget") else "stepper"
                elif dtype == "bool":
                    widget = "toggle"
            setattr(spec, "control_widget", widget)
            knobs.append(spec)
    return knobs


def _environment_panel_from_cfg(cfg: dict) -> Panel:
    return Panel(
        "room_environment",
        str(cfg.get("panel", {}).get("title", "ENVIRONMENT")) if isinstance(cfg, dict) else "ENVIRONMENT",
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


def _seed_missing_state_defaults_for_specs(panel: Panel, state: dict) -> None:
    """Populate missing state keys from knob defaults in a panel spec tree."""

    def visit(node: Panel) -> None:
        for knob in node.knobs:
            name = getattr(knob, "name", "")
            if not name or name in state:
                continue
            choices = list(getattr(knob, "choices", []) or [])
            default = getattr(knob, "default", None)
            if choices:
                # Choice knobs in this HUD may store either index or label depending on knob.
                try:
                    idx = int(default)
                except Exception:
                    try:
                        idx = choices.index(str(default))
                    except Exception:
                        idx = 0
                idx = int(np.clip(idx, 0, max(0, len(choices) - 1)))
                state[name] = choices[idx]
            else:
                state[name] = default
        for sub in node.panels:
            visit(sub)

    visit(panel)


def _normalize_choice_state_value(state: dict, name: str,
                                  choices: list[str], default: int = 0) -> None:
    if not choices:
        return
    fallback = choices[int(np.clip(default, 0, len(choices) - 1))]
    cur = state.get(name, fallback)
    if str(cur) in choices:
        state[name] = str(cur)
        return
    try:
        idx = int(cur)
    except Exception:
        idx = int(default)
    idx = int(np.clip(idx, 0, len(choices) - 1))
    state[name] = choices[idx]


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
_ROOM_SNAP_POLICIES = ["gentle", "no_snap"]

_FLOOR_TYPES = ["rect", "polar", "polar_rect", "arc"]
_HULL_WALL_TYPES = ["flat", "cylindrical"]
_HULL_CORNER_TYPES = ["flat", "spherical"]

# Palette entry indices
_CELL_EMPTY      = 0   # valid, unoccupied floor
_CELL_OCCUPIED   = 1   # instance placed here
_CELL_STATION    = 2   # station anchor tile
_CELL_SELECTED   = 3   # selected instance
_CELL_SNAP       = 4   # network snap candidate
_CELL_DUAL_SNAP  = 5
_CELL_QUAD_SNAP  = 6
_CELL_COLLISION  = 7   # hull-clipped or out-of-plan collision
_CELL_VOID       = 8   # outside the floor plan entirely


def _make_polar_tile_cells(n_rays: int, radial_segs: int,
                           tile_radius_frac: float | None = None) -> list:
    """Return a list of packed polar tile cell dicts.

        Tiles are equal-size squares. The requested tile radius determines a fixed
        side length of ``2 * tile_radius_frac`` in inner-circle fractions.

        Packing strategy:
            1. Reclaim the center with a pixelated inscribed circle.
            2. For each ray sector, build outward in local square rows.
            3. Each new row sits directly on the previous row's top edge.
            4. Each row contains however many equal-size tiles safely fit between
                 the bounding rays and inside the unit outer circle.

    All distances are stored as fractions of the inner-circle radius so they
    can be resolved at render time regardless of panel size.

    Cell fields:
      polar_tile       – True (signals polar-tile rendering path)
            x                – tile slot within the wedge row
            y                – packed wedge-row index (0..N-1)
      state            – _CELL_EMPTY
            quad_xy_frac     – explicit 4-corner quad in polar-local XY fractions
            sector           – ray-sector index (0 .. n_rays-1)
            row              – packed wedge-row index (0..N-1)
            slot             – tile slot within the row (0..N-1)
            rotation_angle   – sector bisector angle
            outward_normal_angle – sector bisector angle
    """
    n_rays = max(2, int(n_rays or 0))
    radial_segs = max(1, int(radial_segs or 8))
    default_tile_radius = 0.5 / float(radial_segs)
    tr_src = default_tile_radius if tile_radius_frac is None else tile_radius_frac
    try:
        tr = float(tr_src)
    except Exception as exc:
        raise ValueError(
            f"polar tile radius is not numeric: {tr_src!r}"
        ) from exc
    if not math.isfinite(tr):
        raise ValueError(
            f"polar tile radius must be finite, got {tr!r} "
            f"(source={tr_src!r}, n_rays={n_rays}, radial_segs={radial_segs})"
        )
    tile_radius_frac = float(np.clip(tr, 1e-4, 0.49))
    tile_width_frac = 2.0 * tile_radius_frac
    if not math.isfinite(tile_width_frac) or tile_width_frac <= 0.0:
        raise ValueError(
            f"polar tile width invalid: {tile_width_frac!r} "
            f"(tile_radius_frac={tile_radius_frac!r})"
        )
    sin_half = math.sin(math.pi / float(n_rays))
    if sin_half < 1e-9:
        return []
    # First possible inner chord for the square side length.
    r_inner0_frac = tile_width_frac / (2.0 * sin_half)
    if not math.isfinite(r_inner0_frac):
        raise ValueError(
            f"polar inner radius invalid: {r_inner0_frac!r} "
            f"(tile_width_frac={tile_width_frac!r}, sin_half={sin_half!r})"
        )
    if r_inner0_frac >= 1.0:
        return []
    cells = []

    # Reclaim the wasted center with a pixelated inscribed circle: scan a
    # centered square lattice across the full first-ring diameter and keep
    # only those tiles whose four corners stay inside the first ring.
    core_span_frac = 2.0 * r_inner0_frac
    n_core = int(math.floor(core_span_frac / tile_width_frac))
    if n_core > 0:
        x0 = -0.5 * n_core * tile_width_frac
        y0 = -0.5 * n_core * tile_width_frac
        for jy in range(n_core):
            for ix in range(n_core):
                x_min = x0 + ix * tile_width_frac
                y_min = y0 + jy * tile_width_frac
                x_max = x_min + tile_width_frac
                y_max = y_min + tile_width_frac
                # Keep only tiles whose farthest corner is within first chord radius.
                if max(
                    math.hypot(x_min, y_min),
                    math.hypot(x_max, y_min),
                    math.hypot(x_max, y_max),
                    math.hypot(x_min, y_max),
                ) > r_inner0_frac + 1e-9:
                    continue
                cells.append({
                    "polar_tile": True,
                    "x": ix,
                    "y": jy,
                    "state": _CELL_EMPTY,
                    "quad_xy_frac": [
                        (x_min, y_min),
                        (x_max, y_min),
                        (x_max, y_max),
                        (x_min, y_max),
                    ],
                })
    sector_half_angle = math.pi / float(n_rays)
    tan_half = math.tan(sector_half_angle)
    row_bottom0 = r_inner0_frac * math.cos(sector_half_angle)
    if (not math.isfinite(row_bottom0)) or (not math.isfinite(tan_half)):
        raise ValueError(
            f"polar row seed invalid: row_bottom0={row_bottom0!r}, "
            f"tan_half={tan_half!r}, sector_half_angle={sector_half_angle!r}"
        )
    max_rows_est = int(math.ceil(max(0.0, (1.0 - row_bottom0)) / tile_width_frac)) + 2
    max_rows = max(1, min(10000, max_rows_est))
    row_i = 0
    while True:
        if row_i >= max_rows:
            break
        row_bottom = row_bottom0 + row_i * tile_width_frac
        row_top = row_bottom + tile_width_frac
        if (not math.isfinite(row_bottom)) or (not math.isfinite(row_top)):
            raise ValueError(
                f"polar row bounds invalid at row {row_i}: "
                f"row_bottom={row_bottom!r}, row_top={row_top!r}, "
                f"row_bottom0={row_bottom0!r}, tile_width_frac={tile_width_frac!r}"
            )
        if row_bottom >= 1.0 or tan_half < 1e-9:
            break
        circle_half_width = math.sqrt(max(0.0, 1.0 - row_top * row_top))
        if not math.isfinite(circle_half_width):
            raise ValueError(
                f"polar circle half-width invalid at row {row_i}: "
                f"circle_half_width={circle_half_width!r}, row_top={row_top!r}"
            )
        row_width = min(2.0 * row_bottom * tan_half, 2.0 * circle_half_width)
        if not math.isfinite(row_width):
            raise ValueError(
                f"polar row width invalid at row {row_i}: "
                f"row_width={row_width!r}, row_bottom={row_bottom!r}, "
                f"tan_half={tan_half!r}, circle_half_width={circle_half_width!r}"
            )
        slots_in_row = int(math.floor((row_width / tile_width_frac) + 1e-9))
        if slots_in_row <= 0:
            break
        row_u0 = -0.5 * slots_in_row * tile_width_frac
        emitted_this_row = 0
        for sector_i in range(n_rays):
            a_mid = (2.0 * math.pi * (float(sector_i) + 0.5)) / float(n_rays)
            nx = math.cos(a_mid)
            ny = math.sin(a_mid)
            tx = -ny
            ty = nx
            for slot_i in range(slots_in_row):
                u0 = row_u0 + slot_i * tile_width_frac
                u1 = u0 + tile_width_frac
                local_pts = [
                    (u0, row_bottom),
                    (u1, row_bottom),
                    (u1, row_top),
                    (u0, row_top),
                ]
                quad_xy_frac = [
                    (tx * u + nx * v, ty * u + ny * v)
                    for (u, v) in local_pts
                ]
                cells.append({
                    "polar_tile": True,
                    "x": slot_i,
                    "y": row_i,
                    "sector": sector_i,
                    "row": row_i,
                    "slot": slot_i,
                    "state": _CELL_EMPTY,
                    "quad_xy_frac": quad_xy_frac,
                    "rotation_angle": a_mid,
                    "outward_normal_angle": a_mid,
                })
                emitted_this_row += 1
        if emitted_this_row <= 0:
            break
        row_i += 1
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Floor plan generators  (common occupancy mask API)
# ─────────────────────────────────────────────────────────────────────────────

def build_floor_mask(floor_type: str, width: int, depth: int,
                     state: dict) -> np.ndarray:
    """Return (depth, width) int8 array.

    Values:
        0  – outside the floor plan
        1  – usable floor cell
    """
    return build_floor_plan(floor_type, width, depth, state)["mask"]


def build_floor_plan(floor_type: str, width: int, depth: int,
                     state: dict) -> dict:
    """Return a common floor-plan descriptor for all grid modes.

    The image-map transport is always a 2-D array, but each floor type chooses
    its own array coordinates:
      rect              -> x/y room cells
      polar             -> angular segment columns by radial ring rows
      polar_rect_center -> centered rect rows followed by radial ring rows
    """
    w, d = max(2, int(width)), max(2, int(depth))
    # Accept either the string name or an integer index into _FLOOR_TYPES.
    try:
        _idx = int(floor_type)
        ft   = _FLOOR_TYPES[_idx] if 0 <= _idx < len(_FLOOR_TYPES) else "rect"
    except (ValueError, TypeError):
        ft = str(floor_type)

    def _oriented_tile_rect(cx: float, cy: float, normal: tuple[float, float],
                            tangent: tuple[float, float], tile_w: float,
                            tile_d: float) -> list[tuple[float, float]]:
        tx, ty = tangent
        nx, ny = normal
        hw = 0.5 * max(0.05, float(tile_w))
        hd = 0.5 * max(0.05, float(tile_d))
        return [
            (cx - tx * hw - nx * hd, cy - ty * hw - ny * hd),
            (cx + tx * hw - nx * hd, cy + ty * hw - ny * hd),
            (cx + tx * hw + nx * hd, cy + ty * hw + ny * hd),
            (cx - tx * hw + nx * hd, cy - ty * hw + ny * hd),
        ]

    if ft == "rect":
        mask = np.ones((d, w), dtype=np.int8)
        cells = [
            {
                "x": gx,
                "y": gy,
                "zone": "rect",
                "coord": (gx, gy),
                "center": (float(gx) + 0.5, float(gy) + 0.5),
                "normal": (0.0, 1.0),
                "tangent": (1.0, 0.0),
                "yaw_deg": 0.0,
                "corners": [
                    (float(gx), float(gy)),
                    (float(gx + 1), float(gy)),
                    (float(gx + 1), float(gy + 1)),
                    (float(gx), float(gy + 1)),
                ],
                "tile_corners": [
                    (float(gx), float(gy)),
                    (float(gx + 1), float(gy)),
                    (float(gx + 1), float(gy + 1)),
                    (float(gx), float(gy + 1)),
                ],
            }
            for gy in range(d) for gx in range(w)
        ]
        return {
            "floor_type": "rect",
            "width": w,
            "height": d,
            "mask": mask,
            "cells": cells,
            "radial_segments": 0,
            "angular_segments": 0,
        }

    if ft == "polar":
        # Full-circle ring packing, tile_size = 1m × 1m physical tiles.
        #
        # Segment count per ring is derived from the INNER radius so that the
        # inner chord of every tile is exactly 1m:
        #   n_segs = floor(2π · r0 / 1.0)
        # Ring 0 (r0 = 0) has no tiles; first populated ring is ring 1 (r0 = 1m, 6 tiles).
        #
        # Tile shape: start from the standard trapezoid, then bring the outer
        # corners inward until the outer edge equals the inner chord.  The outer
        # half-angle is arcsin(r0 · sin(π/n) / r1).  The back corners therefore
        # fan outward, leaving triangular waste between tiles at the outer radius.
        radius = max(1.0, float(state.get("floor_radius", 8.0)))
        tile_size = 1.0
        n_rings = max(1, int(math.ceil(radius / tile_size)))
        ring_segs_list: list[int] = []
        for ring in range(n_rings):
            r0 = ring * tile_size
            n_segs = int(math.floor(2.0 * math.pi * r0 / tile_size)) if r0 > 1e-9 else 0
            ring_segs_list.append(n_segs)
        max_segs = max(ring_segs_list) if any(s > 0 for s in ring_segs_list) else 1
        grid_w = max_segs
        grid_h = n_rings
        mask = np.zeros((grid_h, grid_w), dtype=np.int8)
        cells = []
        for ring, n_segs in enumerate(ring_segs_list):
            if n_segs < 1:
                continue
            r0 = ring * tile_size
            r1 = min(radius, (ring + 1) * tile_size)
            r_mid = 0.5 * (r0 + r1)
            delta_a = 2.0 * math.pi / n_segs
            sin_half_inner = math.sin(delta_a * 0.5)
            # outer half-angle keeps outer edge = inner chord
            sin_half_outer = min(1.0, r0 * sin_half_inner / r1) if r1 > 1e-9 else 0.0
            half_out = math.asin(sin_half_outer)
            my = ring
            x0_ring = (grid_w - n_segs) // 2
            for seg in range(n_segs):
                mx = x0_ring + seg
                a0 = delta_a * seg
                a1 = delta_a * (seg + 1)
                am = 0.5 * (a0 + a1)
                normal = (math.cos(am), math.sin(am))
                tangent = (-math.sin(am), math.cos(am))
                cx = r_mid * normal[0]
                cy = r_mid * normal[1]
                p_il = (r0 * math.cos(a0), r0 * math.sin(a0))
                p_ir = (r0 * math.cos(a1), r0 * math.sin(a1))
                p_or = (r1 * math.cos(am + half_out), r1 * math.sin(am + half_out))
                p_ol = (r1 * math.cos(am - half_out), r1 * math.sin(am - half_out))
                mask[my, mx] = 1
                cells.append({
                    "x": mx, "y": my,
                    "zone": "polar_ring",
                    "ring": ring, "segment": seg, "segment_count": n_segs,
                    "center": (cx, cy),
                    "normal": normal, "tangent": tangent,
                    "yaw_deg": math.degrees(am),
                    "tile_width": 2.0 * r0 * sin_half_inner,
                    "tile_depth": r1 - r0,
                    "coord": (ring, seg),
                    "corners": [
                        p_il,
                        p_ir,
                        (r1 * math.cos(a1), r1 * math.sin(a1)),
                        (r1 * math.cos(a0), r1 * math.sin(a0)),
                    ],
                    "tile_corners": [p_il, p_ir, p_or, p_ol],
                })
        return {
            "floor_type": "polar",
            "width": grid_w, "height": grid_h,
            "mask": mask, "cells": cells,
            "radial_segments": n_rings,
            "angular_segments": max_segs,
            "ring_segments": ring_segs_list,
            "floor_radius": radius,
        }

    if ft == "polar_rect":
        # Fixed angular-segment lattice (no center rect). User controls rings x arcs.
        radius = max(1.0, float(state.get("floor_radius", 8.0)))
        radial_segments = max(1, int(state.get("floor_radial_segments", math.ceil(radius)) or math.ceil(radius)))
        angular_segments = max(8, int(state.get("floor_angular_segments", 24) or 24))
        mask = np.ones((radial_segments, angular_segments), dtype=np.int8)
        cells = []
        for ring in range(radial_segments):
            r0 = radius * ring / radial_segments
            r1 = radius * (ring + 1) / radial_segments
            for segment in range(angular_segments):
                a0 = 2.0 * math.pi * segment / angular_segments
                a1 = 2.0 * math.pi * (segment + 1) / angular_segments
                am = 0.5 * (a0 + a1)
                normal = (math.cos(am), math.sin(am))
                tangent = (-math.sin(am), math.cos(am))
                rm = 0.5 * (r0 + r1)
                cx = rm * normal[0]
                cy = rm * normal[1]
                tile_d = max(0.05, r1 - r0)
                tile_w = max(0.05, 2.0 * rm * math.sin(0.5 * (a1 - a0)))
                cells.append({
                    "x": segment, "y": ring,
                    "zone": "polar",
                    "ring": ring, "segment": segment, "segment_count": angular_segments,
                    "radius_inner": r0, "radius_outer": r1,
                    "angle_start": a0, "angle_end": a1,
                    "center": (cx, cy),
                    "normal": normal, "tangent": tangent,
                    "yaw_deg": math.degrees(am),
                    "tile_width": tile_w, "tile_depth": tile_d,
                    "coord": (ring, segment),
                    "corners": [
                        (r0 * math.cos(a0), r0 * math.sin(a0)),
                        (r1 * math.cos(a0), r1 * math.sin(a0)),
                        (r1 * math.cos(a1), r1 * math.sin(a1)),
                        (r0 * math.cos(a1), r0 * math.sin(a1)),
                    ],
                    "tile_corners": _oriented_tile_rect(cx, cy, normal, tangent, tile_w, tile_d),
                })
        return {
            "floor_type": "polar_rect",
            "width": angular_segments, "height": radial_segments,
            "mask": mask, "cells": cells,
            "radial_segments": radial_segments,
            "angular_segments": angular_segments,
            "floor_radius": radius,
        }

    if ft == "arc":
        # Arc is polar with the full-circle delta_a intact.
        # For each ring, compute n_segs_full = floor(2π·r0) — exactly as polar.
        # delta_a is 2π/n_segs_full (same tile size as polar).
        # Only keep the first floor(arc_span · n_segs_full) tiles (those within
        # the arc).  Tile geometry is identical to polar; arc_span just limits
        # how many tiles of the full ring are included.
        radius = max(1.0, float(state.get("floor_radius", 8.0)))
        arc_span = float(np.clip(state.get("floor_arc_span", 1.0) or 1.0, 0.0, 1.0))
        tile_size = 1.0
        n_rings = max(1, int(math.ceil(radius / tile_size)))
        arc_segs_list: list[int] = []
        for ring in range(n_rings):
            r0 = ring * tile_size
            n_segs_full = int(math.floor(2.0 * math.pi * r0 / tile_size)) if r0 > 1e-9 else 0
            n_segs_arc = int(math.floor(arc_span * n_segs_full))
            arc_segs_list.append(n_segs_arc)
        max_segs = max(arc_segs_list) if any(s > 0 for s in arc_segs_list) else 1
        grid_w = max_segs
        grid_h = n_rings
        mask = np.zeros((grid_h, grid_w), dtype=np.int8)
        cells = []
        for ring, n_segs_arc in enumerate(arc_segs_list):
            if n_segs_arc < 1:
                continue
            r0 = ring * tile_size
            r1 = min(radius, (ring + 1) * tile_size)
            r_mid = 0.5 * (r0 + r1)
            n_segs_full = int(math.floor(2.0 * math.pi * r0 / tile_size))
            delta_a = 2.0 * math.pi / n_segs_full
            sin_half_inner = math.sin(delta_a * 0.5)
            sin_half_outer = min(1.0, r0 * sin_half_inner / r1) if r1 > 1e-9 else 0.0
            half_out = math.asin(sin_half_outer)
            my = ring
            for seg in range(n_segs_arc):
                a0 = delta_a * seg
                a1 = delta_a * (seg + 1)
                am = 0.5 * (a0 + a1)
                normal = (math.cos(am), math.sin(am))
                tangent = (-math.sin(am), math.cos(am))
                cx = r_mid * normal[0]
                cy = r_mid * normal[1]
                p_il = (r0 * math.cos(a0), r0 * math.sin(a0))
                p_ir = (r0 * math.cos(a1), r0 * math.sin(a1))
                p_or = (r1 * math.cos(am + half_out), r1 * math.sin(am + half_out))
                p_ol = (r1 * math.cos(am - half_out), r1 * math.sin(am - half_out))
                mask[my, seg] = 1
                cells.append({
                    "x": seg, "y": my,
                    "zone": "arc_ring",
                    "ring": ring, "segment": seg, "segment_count": n_segs_arc,
                    "center": (cx, cy),
                    "normal": normal, "tangent": tangent,
                    "yaw_deg": math.degrees(am),
                    "tile_width": 2.0 * r0 * sin_half_inner,
                    "tile_depth": r1 - r0,
                    "coord": (ring, seg),
                    "corners": [
                        p_il,
                        p_ir,
                        (r1 * math.cos(a1), r1 * math.sin(a1)),
                        (r1 * math.cos(a0), r1 * math.sin(a0)),
                    ],
                    "tile_corners": [p_il, p_ir, p_or, p_ol],
                })
        return {
            "floor_type": "arc",
            "width": grid_w, "height": grid_h,
            "mask": mask, "cells": cells,
            "radial_segments": n_rings,
            "angular_segments": max_segs,
            "ring_segments": arc_segs_list,
            "floor_radius": radius,
            "arc_span": arc_span,
        }

    return build_floor_plan("rect", w, d, state)


def apply_hull_deformation(mask: np.ndarray, state: dict) -> np.ndarray:
    """Apply edge/corner deformation to a floor mask.

    Returns (depth, width) int8 array where 7 marks hull-clipped cells.
    Input cells with value 0 (void) are left as 0.
    """
    D, W = mask.shape
    result = mask.copy()

    # ── edge deformations ──────────────────────────────────────────────────────
    edges = [
        ("hull_n", "y_max"),
        ("hull_s", "y_min"),
        ("hull_e", "x_max"),
        ("hull_w", "x_min"),
    ]
    for prefix, direction in edges:
        type_idx = _choice_index(state.get(f"{prefix}_type", 0), _HULL_WALL_TYPES)
        htype    = _HULL_WALL_TYPES[max(0, min(type_idx, len(_HULL_WALL_TYPES) - 1))]
        hamount  = float(state.get(f"{prefix}_amount", 0.0))
        if htype == "flat" or hamount <= 0.0:
            continue
        chord = W if "y" in direction else D
        if chord <= 0:
            continue
        # sagitta formula: r = c²/(8s) + s/2
        r = chord ** 2 / (8.0 * hamount) + hamount / 2.0
        for gy in range(D):
            for gx in range(W):
                if result[gy, gx] != 1:
                    continue
                if direction == "y_max":
                    dist, t = D - 1 - gy, gx + 0.5 - W / 2.0
                elif direction == "y_min":
                    dist, t = gy, gx + 0.5 - W / 2.0
                elif direction == "x_max":
                    dist, t = W - 1 - gx, gy + 0.5 - D / 2.0
                else:
                    dist, t = gx, gy + 0.5 - D / 2.0
                if dist >= hamount:
                    continue
                arc_cut = hamount - (r - math.sqrt(max(0.0, r ** 2 - t ** 2)))
                if dist < arc_cut:
                    result[gy, gx] = 7

    # ── corner deformations ────────────────────────────────────────────────────
    corners = [
        ("hull_ne", W - 1, D - 1),
        ("hull_nw", 0,     D - 1),
        ("hull_se", W - 1, 0),
        ("hull_sw", 0,     0),
    ]
    for prefix, cx, cy in corners:
        type_idx = _choice_index(state.get(f"{prefix}_type", 0), _HULL_CORNER_TYPES)
        htype    = _HULL_CORNER_TYPES[max(0, min(type_idx, len(_HULL_CORNER_TYPES) - 1))]
        hradius  = float(state.get(f"{prefix}_radius", 0.0))
        if htype == "flat" or hradius <= 0.0:
            continue
        for gy in range(D):
            for gx in range(W):
                if result[gy, gx] != 1:
                    continue
                if (gx - cx) ** 2 + (gy - cy) ** 2 <= hradius ** 2:
                    result[gy, gx] = 7

    return result


def build_envelope_meshes(floor_result: np.ndarray,
                           state: dict,
                           cell_size_m: float = 1.0,
                           floor_plan: Optional[dict] = None) -> dict:
    """Triangulate floor/wall/ceiling meshes from an occupancy result array.

    Parameters
    ----------
    floor_result : (depth, width) int8 – output of apply_hull_deformation
    state        : room state dict for wall_height / ceil_height
    cell_size_m  : metres per cell

    Returns
    -------
    dict with keys "floor", "walls", "ceiling" → (N, 3, 3) float64 arrays,
    each row being one triangle with 3 XYZ vertices.
    """
    D, W    = floor_result.shape
    cs      = float(cell_size_m)
    wall_h  = float(state.get("wall_height", 3.0))
    ceil_h  = float(state.get("ceil_height", 3.0))
    plan_type = str((floor_plan or {}).get("floor_type", "rect"))
    cell_meta = {
        (int(cell["x"]), int(cell["y"])): cell
        for cell in (floor_plan or {}).get("cells", [])
        if isinstance(cell, dict) and "x" in cell and "y" in cell
    }

    floor_tris: list = []
    wall_tris:  list = []
    ceil_tris:  list = []

    for gy in range(D):
        for gx in range(W):
            if floor_result[gy, gx] != 1:
                continue
            meta = cell_meta.get((gx, gy), {})
            corners2 = meta.get("corners")
            if isinstance(corners2, list) and len(corners2) >= 4:
                quad = []
                for raw_pt in corners2[:4]:
                    if isinstance(raw_pt, (list, tuple)) and len(raw_pt) >= 2:
                        quad.append((float(raw_pt[0]) * cs, float(raw_pt[1]) * cs))
                if len(quad) != 4:
                    quad = [
                        (gx * cs, gy * cs),
                        ((gx + 1) * cs, gy * cs),
                        ((gx + 1) * cs, (gy + 1) * cs),
                        (gx * cs, (gy + 1) * cs),
                    ]
            else:
                quad = [
                    (gx * cs, gy * cs),
                    ((gx + 1) * cs, gy * cs),
                    ((gx + 1) * cs, (gy + 1) * cs),
                    (gx * cs, (gy + 1) * cs),
                ]
            p0, p1, p2, p3 = quad

            floor_tris += [
                [[p0[0], p0[1], 0.0], [p1[0], p1[1], 0.0], [p2[0], p2[1], 0.0]],
                [[p0[0], p0[1], 0.0], [p2[0], p2[1], 0.0], [p3[0], p3[1], 0.0]],
            ]
            ceil_tris += [
                [[p0[0], p0[1], ceil_h], [p2[0], p2[1], ceil_h], [p1[0], p1[1], ceil_h]],
                [[p0[0], p0[1], ceil_h], [p3[0], p3[1], ceil_h], [p2[0], p2[1], ceil_h]],
            ]

            edges = [
                (-1, 0, p3, p0),
                (1, 0, p1, p2),
                (0, -1, p0, p1),
                (0, 1, p2, p3),
            ]
            for dnx, dny, a, b in edges:
                nx, ny = gx + dnx, gy + dny
                if plan_type == "polar" and dny == 0:
                    nx %= W
                if 0 <= nx < W and 0 <= ny < D and floor_result[ny, nx] == 1:
                    continue
                wx0, wy0 = a
                wx1, wy1 = b
                wall_tris += [
                    [[wx0, wy0, 0.0], [wx1, wy1, 0.0], [wx1, wy1, wall_h]],
                    [[wx0, wy0, 0.0], [wx1, wy1, wall_h], [wx0, wy0, wall_h]],
                ]

    def _arr(tris: list) -> np.ndarray:
        if not tris:
            return np.zeros((0, 3, 3), np.float64)
        return np.array(tris, dtype=np.float64)

    return {"floor": _arr(floor_tris), "walls": _arr(wall_tris), "ceiling": _arr(ceil_tris)}


def build_floor_material_triangulation(
    floor_result: np.ndarray,
    state: dict,
    cell_size_m: float = 1.0,
    floor_plan: Optional[dict] = None,
) -> dict:
    """Split active floor cells into tile, border/fill, and border-line geometry.

    ``corners`` define the containing floor region for each logical tile slot.
    ``tile_corners`` define the material tile inside that region. If a plan's
    tile exactly matches its container, a small inset is applied so rectangular
    floors still produce a separate border/fill region.
    """
    D, W = floor_result.shape
    cs = float(cell_size_m)
    inset_m = max(0.0, float(state.get("floor_tile_border_width", 0.04) or 0.0))
    cell_meta = {
        (int(cell["x"]), int(cell["y"])): cell
        for cell in (floor_plan or {}).get("cells", [])
        if isinstance(cell, dict) and "x" in cell and "y" in cell
    }

    def _scaled_quad(raw_quad: Any, gx: int, gy: int) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        if isinstance(raw_quad, list):
            for raw_pt in raw_quad[:4]:
                if isinstance(raw_pt, (list, tuple)) and len(raw_pt) >= 2:
                    pts.append((float(raw_pt[0]) * cs, float(raw_pt[1]) * cs))
        if len(pts) == 4:
            return pts
        return [
            (gx * cs, gy * cs),
            ((gx + 1) * cs, gy * cs),
            ((gx + 1) * cs, (gy + 1) * cs),
            (gx * cs, (gy + 1) * cs),
        ]

    def _quad_area(q: list[tuple[float, float]]) -> float:
        return 0.5 * abs(sum(
            q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1]
            for i in range(4)
        ))

    def _inset_quad(q: list[tuple[float, float]], inset: float) -> list[tuple[float, float]]:
        if inset <= 0.0:
            return list(q)
        cx = 0.25 * sum(p[0] for p in q)
        cy = 0.25 * sum(p[1] for p in q)
        out: list[tuple[float, float]] = []
        for x, y in q:
            vx, vy = x - cx, y - cy
            dist = math.hypot(vx, vy)
            if dist <= 1e-9:
                out.append((x, y))
                continue
            scale = max(0.0, (dist - inset) / dist)
            out.append((cx + vx * scale, cy + vy * scale))
        if _quad_area(out) <= 1e-9:
            return list(q)
        return out

    def _same_quad(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
        return all(math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]) <= 1e-7 for i in range(4))

    def _tri(q: list[tuple[float, float]]) -> list:
        p0, p1, p2, p3 = q
        return [
            [[p0[0], p0[1], 0.0], [p1[0], p1[1], 0.0], [p2[0], p2[1], 0.0]],
            [[p0[0], p0[1], 0.0], [p2[0], p2[1], 0.0], [p3[0], p3[1], 0.0]],
        ]

    tile_tris: list = []
    fill_tris: list = []
    border_lines: list = []

    for gy in range(D):
        for gx in range(W):
            if floor_result[gy, gx] != 1:
                continue
            meta = cell_meta.get((gx, gy), {})
            outer = _scaled_quad(meta.get("corners"), gx, gy)
            inner = _scaled_quad(meta.get("tile_corners"), gx, gy)
            if _same_quad(outer, inner):
                inner = _inset_quad(outer, min(inset_m, 0.45 * math.sqrt(max(_quad_area(outer), 0.0))))
            tile_tris.extend(_tri(inner))

            for i in range(4):
                j = (i + 1) % 4
                strip = [outer[i], outer[j], inner[j], inner[i]]
                if _quad_area(strip) > 1e-10:
                    fill_tris.extend(_tri(strip))
                border_lines.append([inner[i][0], inner[i][1], 0.003])
                border_lines.append([inner[j][0], inner[j][1], 0.003])

    def _arr(tris: list) -> np.ndarray:
        if not tris:
            return np.zeros((0, 3, 3), np.float64)
        return np.array(tris, dtype=np.float64)

    border_arr = np.zeros((0, 3), np.float64) if not border_lines else np.array(border_lines, dtype=np.float64)
    return {
        "floor_tiles": _arr(tile_tris),
        "floor_fill": _arr(fill_tris),
        "floor_borders": border_arr,
        "material_slots": {
            "floor_tiles": str(state.get("floor_tile_material", "pearl_white_tile")),
            "floor_fill": str(state.get("floor_fill_material", "painted_concrete")),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Knob panel helpers
# ─────────────────────────────────────────────────────────────────────────────

def _choice_knob(
    name: str,
    label: str,
    choices: list[str],
    *,
    default: int = 0,
    group: str = "",
) -> KnobSpec:
    return choice_knob(name, label, choices, default=default, group=group, widget="segmented")


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


def _make_envelope_panel() -> Panel:
    return Panel(
        "room_floor",
        "FLOOR",
        knobs=[
            stepper_knob("room_width_cells",  "Width",    "int",   8,   2, 128,  1,  unit="cells", group="Floor", fmt=".0f"),
            stepper_knob("room_depth_cells",  "Depth",    "int",   8,   2, 128,  1,  unit="cells", group="Floor", fmt=".0f"),
            stepper_knob("floor_radius",           "Radius",   "float", 8.0, 1.0, 64.0, 0.5, unit="m", group="Floor", fmt=".1f"),
            stepper_knob("floor_radial_segments",  "Rings",    "int",   8,   1, 128,  1,  unit="", group="Floor (POLAR+RECT)", fmt=".0f"),
            stepper_knob("floor_angular_segments", "Arcs",     "int",  16,   4, 256,  1,  unit="", group="Floor (POLAR+RECT)", fmt=".0f"),
            stepper_knob("floor_polar_rays",        "Rays",     "int",  24,   2, 128,  1,  unit="", group="Floor (POLAR+RECT)", fmt=".0f"),
            stepper_knob("floor_polar_tile_radius", "Tile R",  "float", 0.0,  0.0, 0.49, 0.01, unit="", group="Floor (POLAR+RECT)", fmt=".2f"),
            stepper_knob("floor_arc_span",         "Arc Span", "float", 1.0, 0.0, 1.0, 0.05, unit="", group="Floor (ARC)", fmt=".2f"),
            stepper_knob("floor_center_w",    "Center W", "int",   4,   2,  64,   1,  unit="cells", group="Floor (POLAR)", fmt=".0f"),
            stepper_knob("floor_center_d",    "Center D", "int",   4,   2,  64,   1,  unit="cells", group="Floor (POLAR)", fmt=".0f"),
            stepper_knob("wall_height",       "Wall H",   "float", 3.0, 1.0, 20.0, 0.5, unit="m", group="Volume", fmt=".1f"),
            stepper_knob("ceil_height",       "Ceil H",   "float", 3.0, 1.0, 20.0, 0.5, unit="m", group="Volume", fmt=".1f"),
        ],
        payload={
            "action_first": True,
            "actions": [
                {"key": "floor:rect",       "label": "RECT"},
                {"key": "floor:polar",      "label": "POLAR"},
                {"key": "floor:polar_rect", "label": "POLAR+RECT"},
                {"key": "floor:arc",        "label": "ARC"},
            ],
        },
    )


def _make_hull_panel() -> Panel:
    knobs: list = []
    for edge_prefix, edge_label in [
        ("hull_n", "N Wall"), ("hull_s", "S Wall"),
        ("hull_e", "E Wall"), ("hull_w", "W Wall"),
    ]:
        knobs.append(_choice_knob(f"{edge_prefix}_type",   edge_label,       _HULL_WALL_TYPES, group="Edges"))
        knobs.append(stepper_knob(f"{edge_prefix}_amount", f"{edge_label} D", "float", 0.0, 0.0, 8.0, 0.5, unit="cells", group="Edges", fmt=".1f"))
    for corner_prefix, corner_label in [
        ("hull_ne", "NE"), ("hull_nw", "NW"),
        ("hull_se", "SE"), ("hull_sw", "SW"),
    ]:
        knobs.append(_choice_knob(f"{corner_prefix}_type",   f"{corner_label} Corner", _HULL_CORNER_TYPES, group="Corners"))
        knobs.append(stepper_knob(f"{corner_prefix}_radius", f"{corner_label} R",       "float", 0.0, 0.0, 8.0, 0.5, unit="cells", group="Corners", fmt=".1f"))
    return Panel("room_hull", "HULL", knobs=knobs)


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

    ``active_tab`` exposes the current tab key for callers that render tabbed panels.
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

        Used when a tab delegates GL rendering
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
# Station menu base
# ─────────────────────────────────────────────────────────────────────────────

class _StationMenuBase:
    """Minimal menu lifecycle shared by explicit station menu implementations."""

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

        self.state: Dict[str, Any] = {}

        # HUD visibility
        self._hud_visible = False

        self._last_size    = (0, 0)
        self._gl_ready     = False

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml_safe(cls, path: str) -> Optional["_StationMenuBase"]:
        """Call ``cls.from_yaml(path)`` and return None on any error."""
        try:
            return cls.from_yaml(path)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[{cls.__name__}] load failed: {exc}")
            return None

    def build_gl(self):
        self._gl_ready = True

    # ── HUD visibility ────────────────────────────────────────────────────────

    def show_hud(self, visible: bool):
        self._hud_visible = visible

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_hud(self, win_w: int, win_h: int):
        return

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev) -> bool:
        return False

    # ── State access ──────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Return a copy of the current knob state dict."""
        return dict(self.state)

    def set_value(self, name: str, value: Any):
        """Programmatically set a knob value."""
        if name in self.state:
            self.state[name] = value

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        vis = "visible" if self._hud_visible else "hidden"
        return f"{self.__class__.__name__}(hud={vis}, state_keys={list(self.state)})"


# ─────────────────────────────────────────────────────────────────────────────
# RoomControlStation  (YAML-driven room duty-station HUD)
# ─────────────────────────────────────────────────────────────────────────────

class RoomControlStation(_StationMenuBase):
    """Room-control duty-station HUD.  Loads knob layout from YAML files."""

    def __init__(self, station_cfg: dict, left_cfg: dict,
                 center_cfg: dict, right_cfg: dict,
                 library_yaml_path: Optional[str] = None):
        # Room control does not use the generic YAML left/center panels. Those
        # legacy configs are only accepted for compatibility with old station
        # files; the live HUD is RoomTileLibraryPanel + RoomTileWorkspace.
        super().__init__(station_cfg, {}, {}, right_cfg)
        self._host_station = None
        self._scene_workspace = None
        self._room_workspace = None
        self._room_station_cfg = station_cfg
        self._library_yaml_path = library_yaml_path  # Path to library.yaml for RoomTileLibraryPanel

        # Per-frame hit-test maps, populated by submit_doc_channel.
        self._doc_action_rects: dict = {}   # "{panel}.{action}" → (x,y,w,h)
        self._doc_knob_rects: dict   = {}   # knob_name → {rect, widget, ...}
        self._doc_grid_cells: list[dict[str, Any]] = []
        self._doc_right_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        # Cached envelope meshes, rebuilt on state change.
        self._envelope_meshes: dict  = {}
        self._envelope_mesh_key: str = ""
        self._applied_floor_meshes: dict = {}
        self._applied_floor_mesh_key: str = ""
        self._right_scroll_widget = ScrollableSubpanelList(
            "Room Controls",
            max_height=220,
            key_prefix="room_control_right",
        )

        room_cfg = station_cfg.get("room_editor", {})
        library_dir = str(room_cfg.get("preset_library_dir", ""))
        self._room_presets = load_room_tile_presets(library_dir)

        def _accent(cfg):
            c = cfg.get("panel", {}).get("accent_rgb", [0.0, 0.28, 0.78])
            return tuple(int(x * 255) for x in c)

        self._room_accent = _accent(left_cfg or station_cfg)
        self._ensure_room_workspace()
        self._register_doc_action_handlers()

    def _register_doc_action_handlers(self) -> None:
        """Register doc-channel panel actions under the room-control owner."""
        registry = get_action_registry()
        for panel_name, action_keys in {
            "room_tile_library": ("rotate_left", "rotate_right", "import_mesh"),
            "room_tile_workspace": ("level:-", "level:+"),
            "room_tile_workspace_controls": (
                "level:-",
                "level:+",
                "dimx:-",
                "dimx:+",
                "dimy:-",
                "dimy:+",
                "apply_floor",
            ),
            "room_floor": (
                "floor:rect",
                "floor:polar",
                "floor:polar_rect",
                "floor:arc",
            ),
        }.items():
            for action_key in action_keys:
                map_key = f"{panel_name}.{action_key}"
                registry.register_callable(
                    self._doc_dispatch_key(map_key),
                    self._handle_registered_doc_action,
                    control_action="room_control_doc_action",
                    metadata={
                        "origin": "doc_channel",
                        "owner_id": "room_control_station",
                        "panel_name": panel_name,
                        "action_key": action_key,
                    },
                )

    @staticmethod
    def _doc_dispatch_key(map_key: str) -> str:
        return f"room_control/{map_key}"

    def _current_floor_geometry(self) -> tuple[str, int, int, dict, np.ndarray]:
        """Return the current room-control floor plan and hull-clipped result."""
        ft_raw = self.state.get("floor_type", "rect")
        try:
            ft_idx = int(ft_raw)
            floor_type = _FLOOR_TYPES[ft_idx] if 0 <= ft_idx < len(_FLOOR_TYPES) else "rect"
        except (ValueError, TypeError):
            floor_type = str(ft_raw)
        width = max(2, int(self.state.get("room_width_cells", 8)))
        depth = max(2, int(self.state.get("room_depth_cells", 8)))
        floor_plan = build_floor_plan(floor_type, width, depth, self.state)
        floor_result = apply_hull_deformation(floor_plan["mask"], self.state)
        depth, width = floor_result.shape
        return floor_type, width, depth, floor_plan, floor_result

    def _current_floor_mesh_key(self, floor_type: str, width: int, depth: int) -> str:
        parts: list[Any] = [
            floor_type,
            width,
            depth,
            self.state.get("floor_radius", 8.0),
            self.state.get("floor_radial_segments", ""),
            self.state.get("floor_angular_segments", ""),
            self.state.get("floor_arc_span", ""),
            self.state.get("floor_center_w", 4),
            self.state.get("floor_center_d", 4),
            self.state.get("floor_tile_border_width", 0.04),
        ]
        for prefix in ("hull_n", "hull_s", "hull_e", "hull_w"):
            parts.extend((self.state.get(f"{prefix}_type", 0), self.state.get(f"{prefix}_amount", 0.0)))
        for prefix in ("hull_ne", "hull_nw", "hull_se", "hull_sw"):
            parts.extend((self.state.get(f"{prefix}_type", 0), self.state.get(f"{prefix}_radius", 0.0)))
        return ":".join(str(part) for part in parts)

    def apply_floor_triangulation(self) -> dict:
        """Deploy the current grid floor as material-split triangulated meshes."""
        floor_type, width, depth, floor_plan, floor_result = self._current_floor_geometry()
        meshes = build_floor_material_triangulation(
            floor_result,
            self.state,
            cell_size_m=float(_CELL_SIZE_M),
            floor_plan=floor_plan,
        )
        key = self._current_floor_mesh_key(floor_type, width, depth)
        tile_count = int(len(meshes.get("floor_tiles", ())) or 0)
        fill_count = int(len(meshes.get("floor_fill", ())) or 0)
        border_count = int(len(meshes.get("floor_borders", ())) // 2 if "floor_borders" in meshes else 0)
        self._applied_floor_meshes = meshes
        self._applied_floor_mesh_key = key
        target_ws = self._scene_workspace
        if target_ws is None and self._host_station is not None:
            target_ws = getattr(self._host_station, "_room_workspace_ref", None)
        if target_ws is not None and hasattr(target_ws, "room_cfg"):
            room_cfg = getattr(target_ws, "room_cfg")
            if isinstance(room_cfg, dict):
                room_cfg["applied_floor_meshes"] = meshes
                room_cfg["applied_floor_mesh_key"] = key
                room_cfg["applied_floor_material_slots"] = dict(meshes.get("material_slots", {}))
                room_cfg["applied_floor_revision"] = int(room_cfg.get("applied_floor_revision", 0) or 0) + 1
        self.state["floor_triangulation_applied"] = True
        self.state["floor_triangulation_key"] = key
        self.state["floor_triangulation_tile_triangles"] = tile_count
        self.state["floor_triangulation_fill_triangles"] = fill_count
        self.state["floor_triangulation_border_segments"] = border_count
        return dict(meshes)

    @property
    def panel_spec(self) -> Panel:
        """Controls hierarchy for the live room-control HUD widgets."""
        self._ensure_room_workspace()
        _normalize_choice_state_value(self.state, "palette_tool", _ROOM_PALETTE_TOOLS)
        _normalize_choice_state_value(self.state, "palette_category", _ROOM_PALETTE_CATEGORIES)
        _normalize_choice_state_value(self.state, "room_view", _ROOM_VIEWS)
        _normalize_choice_state_value(self.state, "tile_editor_scope", _ROOM_EDIT_SCOPES)
        _normalize_choice_state_value(self.state, "room_snap_policy", _ROOM_SNAP_POLICIES)
        selected_items: list[str] = []
        if isinstance(self._left_panel, RoomTileLibraryPanel):
            selected_items = [preset.preset_id for preset in self._left_panel.filtered_items()]
            if selected_items:
                current = str(self.state.get("palette_selected", ""))
                if current not in selected_items:
                    self.state["palette_selected"] = selected_items[0]
        grid_panel = self._room_grid_panel()

        panel = Panel(
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
                        stepper_knob("tile_import_scale", "Import Scale", "float", 1.0, 0.1, 20.0, 0.1, group="Tile Editor", fmt=".2f"),
                        _choice_knob("tile_editor_scope", "Edit Scope", _ROOM_EDIT_SCOPES, group="Tile Editor"),
                        readonly_knob("tile_editor_selected_mesh", "Selected Mesh", group="Tile Editor"),
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
                        stepper_knob("room_level", "Level", "int", 0, -64, 64, 1, group="Map", fmt=".0f"),
                        stepper_knob("room_station_x", "Station X", "int", 0, 0, 128, 1, unit="cells", group="Station Anchor", fmt=".0f"),
                        stepper_knob("room_station_y", "Station Y", "int", 0, 0, 128, 1, unit="cells", group="Station Anchor", fmt=".0f"),
                        _choice_knob("room_snap_policy", "Snap", _ROOM_SNAP_POLICIES, group="Placement"),
                        readonly_knob("room_selected_instance", "Selected Instance", group="Selection"),
                    ],
                    panels=[grid_panel],
                    payload={
                        "source_panel": "RoomTileWorkspace",
                        "actions": [
                            {"key": "level:-", "label": "Level -"},
                            {"key": "level:+", "label": "Level +"},
                        ],
                    },
                ),
                _make_envelope_panel(),
                _make_hull_panel(),
                _environment_panel_from_cfg(self._right_cfg),
            ],
        )
        # Keep runtime state fully populated to prevent silent fallback/coercion paths.
        _seed_missing_state_defaults_for_specs(panel, self.state)
        return panel

    @property
    def knob_values(self) -> dict[str, Any]:
        return _doc_knob_values_for_specs(self.panel_spec, self.state)

    def submit_doc_channel(self, doc_rdr, win_w: int, win_h: int) -> None:
        """Submit room control as left palette, center map, and right controls."""
        spec = self.panel_spec
        values = self.knob_values
        if not hasattr(self, "_doc_id_map"):
            self._doc_id_map = {}

        # Reset per-frame hit-test maps.
        self._doc_action_rects.clear()
        self._doc_knob_rects.clear()
        self._doc_grid_cells.clear()
        self._doc_right_rect = (0, 0, 0, 0)

        left_w   = int(self.LEFT_W)
        right_w  = int(self.RIGHT_W)
        center_w = max(120, int(win_w) - left_w - right_w)
        panel_h  = max(80, int(win_h))

        panels = list(getattr(spec, "panels", []) or [])
        if len(panels) < 3:
            doc_rdr.submit_panel(
                spec,
                (10, 10, min(420, int(win_w) - 20), min(panel_h - 20, 720)),
                node_id_map=self._doc_id_map,
                knob_values=values,
                action_rects=self._doc_action_rects,
                knob_rects=self._doc_knob_rects,
            )
            return

        doc_rdr.submit_panel(
            panels[0],
            (0, 0, left_w, panel_h),
            node_id_map=self._doc_id_map,
            knob_values=values,
            sibling_order=0,
            action_rects=self._doc_action_rects,
            knob_rects=self._doc_knob_rects,
        )

        center_panel = panels[1]
        grid_panel = (list(getattr(center_panel, "panels", []) or []) or [self._room_grid_panel()])[0]
        controls_panel = Panel(
            "room_tile_workspace_controls",
            "ROOM MAP",
            knobs=[*list(getattr(center_panel, "knobs", []) or [])],
            payload={
                "source_panel": "RoomTileWorkspace",
                "action_first": True,
                "actions": [
                    {"key": "apply_floor", "label": "Apply Floor"},
                    {"key": "level:-", "label": "Level -"},
                    {"key": "level:+", "label": "Level +"},
                    {"key": "dimx:-", "label": "Width -"},
                    {"key": "dimx:+", "label": "Width +"},
                    {"key": "dimy:-", "label": "Depth -"},
                    {"key": "dimy:+", "label": "Depth +"},
                ],
            },
        )
        controls_h = min(260, max(120, panel_h // 3))
        doc_rdr.submit_panel(
            controls_panel,
            (left_w, 0, center_w, controls_h),
            node_id_map=self._doc_id_map,
            knob_values=values,
            sibling_order=1,
            action_rects=self._doc_action_rects,
            knob_rects=self._doc_knob_rects,
        )
        doc_rdr.submit_panel(
            grid_panel,
            (left_w, controls_h, center_w, max(80, panel_h - controls_h)),
            node_id_map=self._doc_id_map,
            knob_values=values,
            sibling_order=2,
            action_rects=self._doc_action_rects,
            knob_rects=self._doc_knob_rects,
        )
        self._register_doc_image_map_hits(
            grid_panel,
            (left_w, controls_h, center_w, max(80, panel_h - controls_h)),
        )

        # Right column: stack all panels from index 2 (environment, envelope, hull).
        right_panels = panels[2:]
        right_x = left_w + center_w
        self._doc_right_rect = (right_x, 0, right_w, panel_h)

        def _estimate_panel_height(p: Panel) -> int:
            payload = getattr(p, "payload", {}) or {}
            action_count = len([a for a in list(payload.get("actions", []) or []) if isinstance(a, dict)]) if isinstance(payload, dict) else 0
            knob_count = len(list(getattr(p, "knobs", []) or []))
            # Match doc panel geometry constants (header + paddings + rows).
            h = 22 + knob_count * 42 + action_count * 26 + 8
            return max(120, int(h))

        specs: list[ModularSubpanelSpec] = []
        for rp in right_panels:
            specs.append(
                ModularSubpanelSpec(
                    key=str(getattr(rp, "name", "panel")),
                    title=str(getattr(rp, "label", getattr(rp, "name", "PANEL"))),
                    expanded=True,
                    body_height=_estimate_panel_height(rp),
                    render_body=lambda *_args, **_kwargs: None,
                    payload=rp,
                )
            )
        self._right_scroll_widget.max_height = max(80, int(panel_h))
        self._right_scroll_widget.set_subpanels(specs)
        if pygame.get_init():
            pygame.font.init()
            _font = pygame.font.SysFont("monospace", 13)
            _scratch = pygame.Surface((max(32, int(right_w)), max(32, int(panel_h))), pygame.SRCALPHA)
            self._right_scroll_widget.render(_scratch, _font, 0, 0, int(right_w))

        y_cursor = -int(self._right_scroll_widget.scroll_y)
        for rp_i, rp in enumerate(right_panels):
            rp_h = _estimate_panel_height(rp)
            if y_cursor + rp_h < 0:
                y_cursor += rp_h + 4
                continue
            if y_cursor > panel_h:
                break
            doc_rdr.submit_panel(
                rp,
                (right_x, int(y_cursor), right_w, rp_h),
                node_id_map=self._doc_id_map,
                knob_values=values,
                sibling_order=3 + rp_i,
                action_rects=self._doc_action_rects,
                knob_rects=self._doc_knob_rects,
            )
            y_cursor += rp_h + 4

    def _room_grid_panel(self) -> Panel:
        palette = [
            (0.12, 0.14, 0.18, 0.92),   # 0 EMPTY valid floor
            (0.25, 0.45, 0.62, 0.96),   # 1 OCCUPIED by instance
            (0.92, 0.72, 0.20, 1.00),   # 2 STATION anchor
            (0.20, 0.66, 0.90, 1.00),   # 3 SELECTED instance
            (0.18, 0.78, 0.68, 1.00),   # 4 SNAP candidate
            (0.82, 0.62, 0.18, 1.00),   # 5 DUAL snap
            (0.92, 0.38, 0.18, 1.00),   # 6 QUAD snap
            (0.95, 0.12, 0.22, 1.00),   # 7 HULL-CLIPPED / collision
            (0.04, 0.04, 0.06, 0.96),   # 8 VOID (outside floor plan)
        ]
        # ── Derive grid dimensions from floor type ─────────────────────────────
        # floor_type may be stored as an int index (from segmented knob) or str.
        _ft_raw = self.state.get("floor_type", "rect")
        try:
            _ft_idx  = int(_ft_raw)
            floor_type = _FLOOR_TYPES[_ft_idx] if 0 <= _ft_idx < len(_FLOOR_TYPES) else "rect"
        except (ValueError, TypeError):
            floor_type = str(_ft_raw)
        width = max(2, int(self.state.get("room_width_cells", 8)))
        depth = max(2, int(self.state.get("room_depth_cells", 8)))

        # ── Build occupancy result via floor+hull generators ───────────────────
        floor_plan   = build_floor_plan(floor_type, width, depth, self.state)
        floor_mask   = floor_plan["mask"]
        floor_result = apply_hull_deformation(floor_mask, self.state)
        depth, width = floor_result.shape
        floor_meta = {
            (int(cell["x"]), int(cell["y"])): cell
            for cell in floor_plan.get("cells", [])
            if isinstance(cell, dict)
        }

        use_geo_layout = floor_type in ("polar", "polar_rect", "arc")
        geo_bounds = None
        if use_geo_layout:
            pts_xy = []
            for cell in floor_meta.values():
                if not isinstance(cell, dict):
                    continue
                corners = cell.get("tile_corners") or cell.get("corners")
                if isinstance(corners, list) and corners:
                    for raw_pt in corners:
                        if isinstance(raw_pt, (list, tuple)) and len(raw_pt) >= 2:
                            pts_xy.append((float(raw_pt[0]), float(raw_pt[1])))
                else:
                    center = cell.get("center")
                    if isinstance(center, (list, tuple)) and len(center) >= 2:
                        pts_xy.append((float(center[0]), float(center[1])))
            if pts_xy:
                xs = [pt[0] for pt in pts_xy]
                ys = [pt[1] for pt in pts_xy]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                cx_mid = 0.5 * (min_x + max_x)
                cy_mid = 0.5 * (min_y + max_y)
                half = max(0.5, 0.5 * max(max_x - min_x, max_y - min_y))
                geo_bounds = (cx_mid, cy_mid, half)

        # ── Rebuild envelope meshes when the envelope shape changes ───────────
        mesh_parts = [
            floor_type,
            width,
            depth,
            self.state.get("wall_height", 3),
            self.state.get("ceil_height", 3),
            self.state.get("floor_radius", 8.0),
            self.state.get("floor_radial_segments", ""),
            self.state.get("floor_angular_segments", ""),
            self.state.get("floor_center_w", 4),
            self.state.get("floor_center_d", 4),
        ]
        for prefix in ("hull_n", "hull_s", "hull_e", "hull_w"):
            mesh_parts.extend((
                self.state.get(f"{prefix}_type", 0),
                self.state.get(f"{prefix}_amount", 0.0),
            ))
        for prefix in ("hull_ne", "hull_nw", "hull_se", "hull_sw"):
            mesh_parts.extend((
                self.state.get(f"{prefix}_type", 0),
                self.state.get(f"{prefix}_radius", 0.0),
            ))
        mesh_key = ":".join(str(part) for part in mesh_parts)
        if mesh_key != self._envelope_mesh_key:
            self._envelope_meshes  = build_envelope_meshes(floor_result, self.state, floor_plan=floor_plan)
            self._envelope_mesh_key = mesh_key

        # polar_rect display uses packed rectangular tiles in ray sectors.
        if floor_type == "polar_rect":
            n_rays = max(2, int(self.state.get("floor_polar_rays", 24) or 24))
            radial_segs = max(1, int(floor_plan.get("radial_segments", 8) or 8))
            raw_tr = float(self.state.get("floor_polar_tile_radius", 0.0) or 0.0)
            tr_arg = raw_tr if raw_tr > 1e-4 else None
            polar_cells = _make_polar_tile_cells(n_rays, radial_segs, tr_arg)
            _polar_fp = {
                "type": floor_type,
                "radial_segments": radial_segs,
                "angular_segments": n_rays,
                "polar_rays": n_rays,
                "ring_segments": [],
                "floor_radius": float(floor_plan.get("floor_radius", 8.0) or 8.0),
            }
            if self._room_workspace is not None:
                ws = self._room_workspace
                view_name = str(self.state.get("room_view", "plan"))
                level = int(self.state.get("room_level", 0))
                selected_id = str(self.state.get("room_selected_instance", ""))
                snap_policy = str(self.state.get("room_snap_policy", "gentle"))
                floor_radius = float(_polar_fp["floor_radius"])

                station_cells = set(ws._station_cells())
                occupied: dict[tuple[int, int], int] = {}
                for instance in ws.instances:
                    preset = ws.presets.get(instance.preset_id)
                    if preset is None:
                        continue
                    if view_name == "plan" and not (instance.level <= level < instance.level + preset.level_span):
                        continue
                    state_idx = _CELL_SELECTED if instance.instance_id == selected_id else _CELL_OCCUPIED
                    for dx in range(preset.footprint_xy[0]):
                        for dy in range(preset.footprint_xy[1]):
                            occupied[(instance.grid_x + dx, instance.grid_y + dy)] = state_idx

                object_marks: dict[tuple[int, int], dict[str, Any]] = {}
                if view_name == "plan" and hasattr(ws, "floor_plan_object_cell_marks"):
                    object_marks = ws.floor_plan_object_cell_marks(level, snap_policy=snap_policy)
                mark_state = {
                    "fit": _CELL_OCCUPIED,
                    "snap": _CELL_SNAP,
                    "dual": _CELL_DUAL_SNAP,
                    "quad": _CELL_QUAD_SNAP,
                    "collision": _CELL_COLLISION,
                }

                snap_cells: set[tuple[int, int]] = set()
                sel_preset = ws.selected_preset()
                if sel_preset is not None and bool(getattr(sel_preset, "network_strict_snap", False)):
                    snap_cells = {(int(x), int(y)) for x, y, _ in ws._network_contact_candidates(level, sel_preset)}

                def _world_cells_for_packed_quad(cell: dict) -> set[tuple[int, int]]:
                    qf = cell.get("quad_xy_frac")
                    if not (isinstance(qf, list) and len(qf) >= 4):
                        return set()
                    quad: list[tuple[float, float]] = []
                    for raw_pt in qf[:4]:
                        if isinstance(raw_pt, (list, tuple)) and len(raw_pt) >= 2:
                            quad.append((float(raw_pt[0]) * floor_radius, float(raw_pt[1]) * floor_radius))
                    if len(quad) != 4:
                        return set()

                    xs = [p[0] for p in quad]
                    ys = [p[1] for p in quad]
                    gx0 = int(math.floor(min(xs)))
                    gx1 = int(math.floor(max(xs)))
                    gy0 = int(math.floor(min(ys)))
                    gy1 = int(math.floor(max(ys)))
                    qcx = 0.25 * sum(p[0] for p in quad)
                    qcy = 0.25 * sum(p[1] for p in quad)
                    ordered = sorted(quad, key=lambda p: math.atan2(p[1] - qcy, p[0] - qcx))

                    def _inside(px: float, py: float) -> bool:
                        sign = 0
                        for i in range(4):
                            ax, ay = ordered[i]
                            bx, by = ordered[(i + 1) % 4]
                            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                            if abs(cross) <= 1e-9:
                                continue
                            cur = 1 if cross > 0.0 else -1
                            if sign == 0:
                                sign = cur
                            elif sign != cur:
                                return False
                        return True

                    cells_hit: set[tuple[int, int]] = set()
                    for gy in range(gy0 - 1, gy1 + 2):
                        for gx in range(gx0 - 1, gx1 + 2):
                            if _inside(float(gx) + 0.5, float(gy) + 0.5):
                                cells_hit.add((gx, gy))
                    for px, py in quad:
                        cells_hit.add((int(math.floor(px)), int(math.floor(py))))
                    cells_hit.add((int(math.floor(qcx)), int(math.floor(qcy))))
                    return cells_hit

                coupled_cells: list[dict[str, Any]] = []
                for raw_cell in polar_cells:
                    if not isinstance(raw_cell, dict):
                        coupled_cells.append(raw_cell)
                        continue
                    cell = dict(raw_cell)
                    world_cells = _world_cells_for_packed_quad(cell)
                    station_hit = any((gx, gy, level) in station_cells for gx, gy in world_cells)
                    snap_hit = any((gx, gy) in snap_cells for gx, gy in world_cells)
                    mark = next((object_marks[(gx, gy)] for gx, gy in world_cells if (gx, gy) in object_marks), {})
                    occ = next((occupied[(gx, gy)] for gx, gy in world_cells if (gx, gy) in occupied), None)
                    if station_hit:
                        idx = _CELL_STATION
                    elif snap_hit:
                        idx = _CELL_SNAP
                    elif mark:
                        idx = mark_state.get(str(mark.get("status", "")), occ if occ is not None else _CELL_EMPTY)
                    elif occ is not None:
                        idx = occ
                    else:
                        idx = _CELL_EMPTY
                    cell["state"] = int(idx)
                    if mark:
                        cell["label"] = str(mark.get("label", cell.get("label", "")))
                    coupled_cells.append(cell)
                polar_cells = coupled_cells

            _polar_payload: dict[str, Any] = {
                "type": "image_map",
                "source_panel": "RoomTileWorkspace",
                "width": n_rays,
                "height": radial_segs,
                "cells": polar_cells,
                "palette": palette,
                "action_key": "room_grid.click",
                "coord_mode": "cell",
                "floor_plan": _polar_fp,
            }
            if self._room_workspace is not None:
                _polar_payload["level"] = int(self.state.get("room_level", 0))
                _polar_payload["view"] = str(self.state.get("room_view", "plan"))
                _polar_payload["snap_policy"] = str(self.state.get("room_snap_policy", "gentle"))
            return Panel("room_grid", "GRID", payload=_polar_payload)

        if self._room_workspace is None:
            cells = []
            for y in range(depth):
                for x in range(width):
                    fr = int(floor_result[y, x]) if (0 <= y < floor_result.shape[0] and 0 <= x < floor_result.shape[1]) else 0
                    if fr == 0 and (x, y) not in floor_meta:
                        continue
                    meta = dict(floor_meta.get((x, y), {}))
                    state_idx = _CELL_VOID if fr == 0 else (_CELL_COLLISION if fr == 7 else _CELL_EMPTY)
                    if geo_bounds is not None:
                        cx_mid, cy_mid, half = geo_bounds
                        corners = meta.get("tile_corners") or meta.get("corners")
                        pts = []
                        if isinstance(corners, list):
                            for raw_pt in corners:
                                if isinstance(raw_pt, (list, tuple)) and len(raw_pt) >= 2:
                                    pts.append((float(raw_pt[0]), float(raw_pt[1])))
                        if pts:
                            ux0 = float(np.clip((min(p[0] for p in pts) - (cx_mid - half)) / (2.0 * half), 0.0, 1.0))
                            ux1 = float(np.clip((max(p[0] for p in pts) - (cx_mid - half)) / (2.0 * half), 0.0, 1.0))
                            uy0w = float(np.clip((min(p[1] for p in pts) - (cy_mid - half)) / (2.0 * half), 0.0, 1.0))
                            uy1w = float(np.clip((max(p[1] for p in pts) - (cy_mid - half)) / (2.0 * half), 0.0, 1.0))
                            meta["ui_x0"] = ux0
                            meta["ui_x1"] = ux1
                            meta["ui_y0"] = 1.0 - uy1w
                            meta["ui_y1"] = 1.0 - uy0w
                        else:
                            center = meta.get("center")
                            if isinstance(center, (list, tuple)) and len(center) >= 2:
                                cx = float(center[0])
                                cy = float(center[1])
                                meta["ui_cx"] = float(np.clip((cx - (cx_mid - half)) / (2.0 * half), 0.0, 1.0))
                                meta["ui_cy"] = float(np.clip(1.0 - ((cy - (cy_mid - half)) / (2.0 * half)), 0.0, 1.0))
                                tw = float(max(0.05, meta.get("tile_width", 1.0)))
                                td = float(max(0.05, meta.get("tile_depth", 1.0)))
                                meta["ui_w"] = float(np.clip(tw / (2.0 * half), 0.02, 0.28))
                                meta["ui_h"] = float(np.clip(td / (2.0 * half), 0.02, 0.28))
                    meta.update({"x": x, "y": y, "state": state_idx})
                    cells.append(meta)
            return Panel(
                "room_grid",
                "GRID",
                payload={
                    "type": "image_map",
                    "source_panel": "RoomTileWorkspace",
                    "width": width,
                    "height": depth,
                    "cells": cells,
                    "palette": palette,
                    "action_key": "room_grid.click",
                    "coord_mode": "cell",
                    "floor_plan": {
                        "type": floor_type,
                        "radial_segments": floor_plan.get("radial_segments", 0),
                        "angular_segments": floor_plan.get("angular_segments", 0),
                        "ring_segments": floor_plan.get("ring_segments", []),
                        "floor_radius": float(floor_plan.get("floor_radius", 8.0) or 8.0),
                    },
                },
            )

        ws          = self._room_workspace
        view_name   = str(self.state.get("room_view", "plan"))
        level       = int(self.state.get("room_level", 0))
        selected_id = str(self.state.get("room_selected_instance", ""))
        snap_policy = str(self.state.get("room_snap_policy", "gentle"))

        station_cells = set(ws._station_cells())
        occupied: dict[tuple[int, int], int] = {}
        for instance in ws.instances:
            preset = ws.presets.get(instance.preset_id)
            if preset is None:
                continue
            if view_name == "plan" and not (instance.level <= level < instance.level + preset.level_span):
                continue
            state_idx = _CELL_SELECTED if instance.instance_id == selected_id else _CELL_OCCUPIED
            for dx in range(preset.footprint_xy[0]):
                for dy in range(preset.footprint_xy[1]):
                    occupied[(instance.grid_x + dx, instance.grid_y + dy)] = state_idx

        object_marks: dict[tuple[int, int], dict[str, Any]] = {}
        if view_name == "plan" and hasattr(ws, "floor_plan_object_cell_marks"):
            object_marks = ws.floor_plan_object_cell_marks(level, snap_policy=snap_policy)
        mark_state = {
            "fit": _CELL_OCCUPIED,
            "snap": _CELL_SNAP,
            "dual": _CELL_DUAL_SNAP,
            "quad": _CELL_QUAD_SNAP,
            "collision": _CELL_COLLISION,
        }

        snap_cells: set[tuple[int, int]] = set()
        sel_preset = ws.selected_preset()
        if sel_preset is not None and bool(getattr(sel_preset, "network_strict_snap", False)):
            snap_cells = {(int(x), int(y)) for x, y, _ in ws._network_contact_candidates(level, sel_preset)}

        # ── World→tile translation for geo-layout floor types ─────────────────
        # occupied / object_marks / station_cells / snap_cells are all keyed by
        # world-meter integer coords (gx, gy).  For polar / arc / polar_rect the
        # cell loop iterates mask coords (mx, my).  Bridge them by collecting,
        # for each mask tile, every world-cell integer that any corner or the
        # centre of that tile overlaps.
        if use_geo_layout:
            # Build: world-cell (int, int) → set of mask tile keys (mx, my)
            _w2t: dict[tuple[int, int], set[tuple[int, int]]] = {}
            for (mx, my), _cell in floor_meta.items():
                _pts: list[tuple[float, float]] = []
                _tc = _cell.get("tile_corners") or _cell.get("corners")
                if isinstance(_tc, list):
                    for _p in _tc:
                        if isinstance(_p, (list, tuple)) and len(_p) >= 2:
                            _pts.append((float(_p[0]), float(_p[1])))
                _ctr = _cell.get("center")
                if isinstance(_ctr, (list, tuple)) and len(_ctr) >= 2:
                    _pts.append((float(_ctr[0]), float(_ctr[1])))
                for _px, _py in _pts:
                    _gx = int(math.floor(_px))
                    _gy = int(math.floor(_py))
                    _w2t.setdefault((_gx, _gy), set()).add((mx, my))

            # Re-key occupied in mask coords
            _new_occ: dict[tuple[int, int], int] = {}
            for instance in ws.instances:
                _pr = ws.presets.get(instance.preset_id)
                if _pr is None:
                    continue
                if view_name == "plan" and not (
                    instance.level <= level < instance.level + _pr.level_span
                ):
                    continue
                _s = _CELL_SELECTED if instance.instance_id == selected_id else _CELL_OCCUPIED
                for _dx in range(_pr.footprint_xy[0]):
                    for _dy in range(_pr.footprint_xy[1]):
                        for _tk in _w2t.get(
                            (instance.grid_x + _dx, instance.grid_y + _dy), ()
                        ):
                            if _new_occ.get(_tk) != _CELL_SELECTED:
                                _new_occ[_tk] = _s
            occupied = _new_occ

            # Re-key object_marks in mask coords
            _new_marks: dict[tuple[int, int], dict[str, Any]] = {}
            for (_gx, _gy), _mk in object_marks.items():
                for _tk in _w2t.get((_gx, _gy), ()):
                    _new_marks.setdefault(_tk, _mk)
            object_marks = _new_marks

            # Re-key station_cells in mask coords  (station_cells is set of (sx, sy, lv))
            _new_st: set[tuple[int, int, int]] = set()
            for (_sx, _sy, _slv) in station_cells:
                for (_mx2, _my2) in _w2t.get((_sx, _sy), ()):
                    _new_st.add((_mx2, _my2, _slv))
            station_cells = _new_st

            # Re-key snap_cells in mask coords
            _new_snap: set[tuple[int, int]] = set()
            for (_wx, _wy) in snap_cells:
                for _tk in _w2t.get((_wx, _wy), ()):
                    _new_snap.add(_tk)
            snap_cells = _new_snap

        cells = []
        for y in range(depth):
            for x in range(width):
                fr = int(floor_result[y, x]) if (0 <= y < floor_result.shape[0] and 0 <= x < floor_result.shape[1]) else 0
                if fr == 0 and (x, y) not in floor_meta:
                    continue
                if fr == 0:
                    idx = _CELL_VOID
                elif fr == 7:
                    # Hull-clipped: show collision if an instance tries to live here
                    idx = _CELL_COLLISION
                elif (x, y, level) in station_cells:
                    idx = _CELL_STATION
                elif (x, y) in snap_cells:
                    idx = _CELL_SNAP
                else:
                    mark = object_marks.get((x, y), {})
                    occ  = occupied.get((x, y))
                    if mark:
                        idx = mark_state.get(str(mark.get("status", "")), occ if occ is not None else _CELL_EMPTY)
                    elif occ is not None:
                        idx = occ
                    else:
                        idx = _CELL_EMPTY
                mark  = object_marks.get((x, y), {})
                label = str(mark.get("label", f"{x},{y}"))
                meta = dict(floor_meta.get((x, y), {}))
                if geo_bounds is not None:
                    cx_mid, cy_mid, half = geo_bounds
                    corners = meta.get("tile_corners") or meta.get("corners")
                    pts = []
                    if isinstance(corners, list):
                        for raw_pt in corners:
                            if isinstance(raw_pt, (list, tuple)) and len(raw_pt) >= 2:
                                pts.append((float(raw_pt[0]), float(raw_pt[1])))
                    if pts:
                        ux0 = float(np.clip((min(p[0] for p in pts) - (cx_mid - half)) / (2.0 * half), 0.0, 1.0))
                        ux1 = float(np.clip((max(p[0] for p in pts) - (cx_mid - half)) / (2.0 * half), 0.0, 1.0))
                        uy0w = float(np.clip((min(p[1] for p in pts) - (cy_mid - half)) / (2.0 * half), 0.0, 1.0))
                        uy1w = float(np.clip((max(p[1] for p in pts) - (cy_mid - half)) / (2.0 * half), 0.0, 1.0))
                        meta["ui_x0"] = ux0
                        meta["ui_x1"] = ux1
                        meta["ui_y0"] = 1.0 - uy1w
                        meta["ui_y1"] = 1.0 - uy0w
                    else:
                        center = meta.get("center")
                        if isinstance(center, (list, tuple)) and len(center) >= 2:
                            cx = float(center[0])
                            cy = float(center[1])
                            meta["ui_cx"] = float(np.clip((cx - (cx_mid - half)) / (2.0 * half), 0.0, 1.0))
                            meta["ui_cy"] = float(np.clip(1.0 - ((cy - (cy_mid - half)) / (2.0 * half)), 0.0, 1.0))
                            tw = float(max(0.05, meta.get("tile_width", 1.0)))
                            td = float(max(0.05, meta.get("tile_depth", 1.0)))
                            meta["ui_w"] = float(np.clip(tw / (2.0 * half), 0.02, 0.28))
                            meta["ui_h"] = float(np.clip(td / (2.0 * half), 0.02, 0.28))
                meta.update({"x": x, "y": y, "state": idx, "label": label})
                cells.append(meta)

        return Panel(
            "room_grid",
            "GRID",
            payload={
                "type": "image_map",
                "source_panel": "RoomTileWorkspace",
                "width": width,
                "height": depth,
                "cells": cells,
                "palette": palette,
                "action_key": "room_grid.click",
                "coord_mode": "cell",
                "level": level,
                "view": view_name,
                "snap_policy": snap_policy,
                "floor_plan": {
                    "type": floor_type,
                    "radial_segments": floor_plan.get("radial_segments", 0),
                    "angular_segments": floor_plan.get("angular_segments", 0),
                    "ring_segments": floor_plan.get("ring_segments", []),
                    "floor_radius": float(floor_plan.get("floor_radius", 8.0) or 8.0),
                },
            },
        )

    # ── Doc-channel event handling ─────────────────────────────────────────────

    def handle_event(self, ev) -> bool:
        """Route pygame mouse events to action buttons and interactive knobs."""
        if ev.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            rx, ry, rw, rh = self._doc_right_rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                if self._right_scroll_widget.handle_scroll(-int(ev.y)):
                    return True
                return False
        if ev.type != pygame.MOUSEBUTTONDOWN or ev.button != 1:
            return False
        mx, my = ev.pos

        for entry in self._doc_grid_cells:
            rx, ry, rw, rh = entry["rect"]
            if not (rx <= mx < rx + rw and ry <= my < ry + rh):
                continue
            quad = entry.get("quad_corners")
            if quad:
                # Point-in-convex-quad via cross product (CW winding in screen space).
                inside = True
                n_q = len(quad)
                for j in range(n_q):
                    ax, ay = quad[j]
                    bx_, by_ = quad[(j + 1) % n_q]
                    if ((mx - ax) * (by_ - ay) - (my - ay) * (bx_ - ax)) < 0:
                        inside = False
                        break
                if not inside:
                    continue
            if True:
                if self._room_workspace is not None:
                    if "rotation_angle" in entry:
                        self.state["room_polar_rotation_override_angle"] = float(entry.get("rotation_angle", 0.0))
                        self.state["room_polar_rotation_override_x"] = int(entry.get("grid_x", 0))
                        self.state["room_polar_rotation_override_y"] = int(entry.get("grid_y", 0))
                    else:
                        self.state.pop("room_polar_rotation_override_angle", None)
                        self.state.pop("room_polar_rotation_override_x", None)
                        self.state.pop("room_polar_rotation_override_y", None)
                    self._room_workspace.click_grid(
                        int(entry.get("grid_x", 0)),
                        int(entry.get("grid_y", 0)),
                        int(entry.get("level", 0)),
                    )
                    return True
        # Topmost UI wins when rects overlap: iterate in reverse submit order.
        for map_key, rect in reversed(list(self._doc_action_rects.items())):
            ax, ay, aw, ah = rect
            if ax <= mx < ax + aw and ay <= my < ay + ah:
                self._dispatch_doc_action(map_key)
                return True
        for knob_name, info in reversed(list(self._doc_knob_rects.items())):
            rx, ry, rw, rh = info["rect"]
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                self._handle_doc_knob_click(knob_name, info, float(mx - rx), float(rw))
                return True
        return False

    def _dispatch_doc_action(self, map_key: str) -> None:
        """Route a '{panel_name}.{action_key}' string to the right handler."""
        enqueue_action(
            self._doc_dispatch_key(map_key),
            map_key=map_key,
            station=self,
        )

    def _invalidate_doc_ui_cache(self) -> None:
        """Force doc-channel node-id reallocation on next submit."""
        if hasattr(self, "_doc_id_map"):
            self._doc_id_map.clear()

    def _handle_registered_doc_action(self, **context: Any) -> None:
        map_key = str(context.get("map_key", ""))
        parts = map_key.split(".", 1)
        if len(parts) != 2:
            return
        _panel_name, action_key = parts

        if action_key == "rotate_left":
            if self._room_workspace is not None:
                self._room_workspace.rotate_selection_left()
        elif action_key == "rotate_right":
            if self._room_workspace is not None:
                self._room_workspace.rotate_selection_right()
        elif action_key == "import_mesh":
            if self._room_workspace is not None:
                self._room_workspace.import_mesh_dialog()
        elif action_key == "level:-":
            self.state["room_level"] = int(self.state.get("room_level", 0)) - 1
        elif action_key == "level:+":
            self.state["room_level"] = int(self.state.get("room_level", 0)) + 1
        elif action_key == "dimx:-":
            self.state["room_width_cells"] = max(2, int(self.state.get("room_width_cells", 8)) - 1)
        elif action_key == "dimx:+":
            self.state["room_width_cells"] = min(128, int(self.state.get("room_width_cells", 8)) + 1)
        elif action_key == "dimy:-":
            self.state["room_depth_cells"] = max(2, int(self.state.get("room_depth_cells", 8)) - 1)
        elif action_key == "dimy:+":
            self.state["room_depth_cells"] = min(128, int(self.state.get("room_depth_cells", 8)) + 1)
        elif action_key == "apply_floor":
            self.apply_floor_triangulation()
        elif action_key.startswith("floor:"):
            mode = action_key.split(":", 1)[1]
            if mode in _FLOOR_TYPES:
                self.state["floor_type"] = int(_FLOOR_TYPES.index(mode))
        self._invalidate_doc_ui_cache()

    def _handle_doc_knob_click(self, knob_name: str, info: dict,
                                local_x: float, rect_w: float) -> None:
        """Advance or toggle a knob value based on click position within its rect."""
        widget = str(info.get("widget", ""))
        dtype  = str(info.get("dtype", "float"))
        lo     = float(info.get("low",  0.0))
        hi     = float(info.get("high", 1.0))
        step   = float(info.get("step", 0.0))
        choices: list = list(info.get("choices", []) or [])

        if widget == "stepper":
            if step == 0.0:
                step = 1.0 if dtype == "int" else 0.1
            delta   = -step if local_x < rect_w / 2.0 else step
            cur_src = self.state.get(knob_name, info.get("default", 0))
            cur     = float(cur_src or 0)
            new_val = max(lo, min(hi, cur + delta))
            self.state[knob_name] = int(round(new_val)) if dtype == "int" else new_val

        elif widget == "segmented":
            if choices:
                if knob_name == "floor_type":
                    # The rendered segmented text includes a leading "Label:" area.
                    # Map clicks only over the value region to avoid selecting the
                    # wrong floor type when clicking option text.
                    value_x0 = rect_w * 0.34
                    value_w = max(1.0, rect_w - value_x0)
                    local_vx = max(0.0, min(value_w - 1.0, local_x - value_x0))
                    seg_w = value_w / max(1, len(choices))
                else:
                    seg_w = rect_w / max(1, len(choices))
                    local_vx = local_x
                idx = min(len(choices) - 1, max(0, int(local_vx / seg_w)))
                if knob_name == "floor_type" or (
                    knob_name.startswith("hull_") and knob_name.endswith("_type")
                ):
                    self.state[knob_name] = int(idx)
                else:
                    self.state[knob_name] = choices[idx]

        elif widget == "toggle":
            self.state[knob_name] = not bool(self.state.get(knob_name, False))

        elif dtype in ("choice",) and choices:
            cur = self.state.get(knob_name, 0)
            try:
                idx = choices.index(str(cur))
            except ValueError:
                try:
                    idx = int(cur)
                except Exception:
                    idx = 0
            except Exception:
                idx = 0
            next_idx = (idx + 1) % len(choices)
            if knob_name == "floor_type" or (
                knob_name.startswith("hull_") and knob_name.endswith("_type")
            ):
                self.state[knob_name] = int(next_idx)
            else:
                self.state[knob_name] = choices[next_idx]

        elif dtype == "bool":
            self.state[knob_name] = not bool(self.state.get(knob_name, False))

        self._invalidate_doc_ui_cache()

    def _register_doc_image_map_hits(self, panel: Panel, rect: tuple[int, int, int, int]) -> None:
        payload = getattr(panel, "payload", {}) or {}
        if not isinstance(payload, dict) or payload.get("type") != "image_map":
            return
        if str(payload.get("action_key", "")) != "room_grid.click":
            return

        x, y, w, h = rect
        hdr_h = 18
        map_x = int(x + 4)
        map_y = int(y + hdr_h + 6)
        map_w = max(1, int(w - 8))
        map_h = max(1, int(h - hdr_h - 10))
        grid_w = max(1, int(payload.get("width", 1)))
        grid_h = max(1, int(payload.get("height", 1)))
        cell_w = max(1, map_w // grid_w)
        cell_h = max(1, map_h // grid_h)
        level = int(payload.get("level", int(self.state.get("room_level", 0))))

        for index, raw_cell in enumerate(list(payload.get("cells", []) or [])):
            if isinstance(raw_cell, dict) and raw_cell.get("polar_tile"):
                side_f  = float(max(8, min(map_w, map_h))) * 0.92
                ri      = max(1.0, side_f * 0.5 - 3.0 - max(1.0, (side_f * 0.5 - 3.0) * 0.01))
                ccx     = float(map_x + map_w * 0.5)
                ccy     = float(map_y + map_h * 0.5)
                quad_xy_frac = raw_cell.get("quad_xy_frac")
                if isinstance(quad_xy_frac, list) and len(quad_xy_frac) >= 4:
                    quad_px = []
                    for raw_pt in quad_xy_frac[:4]:
                        if isinstance(raw_pt, (list, tuple)) and len(raw_pt) >= 2:
                            quad_px.append((ccx + float(raw_pt[0]) * ri, ccy + float(raw_pt[1]) * ri))
                    if len(quad_px) != 4:
                        continue
                    p_bl, p_br, p_tr, p_tl = quad_px
                else:
                    a_mid        = float(raw_cell.get("a_mid", 0.0))
                    r_inner_frac = float(raw_cell.get("r_inner_frac", 0.0))
                    r_outer_frac = float(raw_cell.get("r_outer_frac", 0.0))
                    half_angle   = float(raw_cell.get("half_angle", 0.0))
                    r_inner_px   = r_inner_frac * ri
                    r_outer_px   = r_outer_frac * ri
                    a0 = a_mid - half_angle
                    a1 = a_mid + half_angle
                    p_bl = (ccx + math.cos(a0) * r_inner_px, ccy + math.sin(a0) * r_inner_px)
                    p_br = (ccx + math.cos(a1) * r_inner_px, ccy + math.sin(a1) * r_inner_px)
                    tile_depth_px = max(0.0, (r_outer_px - r_inner_px))
                    mpx = 0.5 * (p_bl[0] + p_br[0])
                    mpy = 0.5 * (p_bl[1] + p_br[1])
                    nx = mpx - ccx
                    ny = mpy - ccy
                    nlen = math.hypot(nx, ny)
                    if nlen > 1e-9:
                        nx /= nlen
                        ny /= nlen
                    else:
                        nx = math.cos(a_mid)
                        ny = math.sin(a_mid)
                    p_tl = (p_bl[0] + nx * tile_depth_px, p_bl[1] + ny * tile_depth_px)
                    p_tr = (p_br[0] + nx * tile_depth_px, p_br[1] + ny * tile_depth_px)
                quad_px = [p_bl, p_br, p_tr, p_tl]
                xs_q = [p[0] for p in quad_px]; ys_q = [p[1] for p in quad_px]
                aabb = (
                    int(math.floor(min(xs_q))),
                    int(math.floor(min(ys_q))),
                    int(math.ceil(max(xs_q))) - int(math.floor(min(xs_q))) + 1,
                    int(math.ceil(max(ys_q))) - int(math.floor(min(ys_q))) + 1,
                )
                quad_cx = 0.25 * sum(p[0] for p in quad_px)
                quad_cy = 0.25 * sum(p[1] for p in quad_px)
                angle_default = float(raw_cell.get("rotation_angle", math.atan2(quad_cy - ccy, quad_cx - ccx)))
                self._doc_grid_cells.append({
                    "rect": aabb,
                    "quad_corners": quad_px,
                    "rotation_angle": angle_default,
                    "outward_normal_angle": float(raw_cell.get("outward_normal_angle", angle_default)),
                    "grid_x": int(raw_cell.get("x", index)),
                    "grid_y": int(raw_cell.get("y", 0)),
                    "level": level,
                })
                continue

            if isinstance(raw_cell, dict):
                cx = int(raw_cell.get("x", index % grid_w))
                cy = int(raw_cell.get("y", index // grid_w))
            else:
                cx = index % grid_w
                cy = index // grid_w
            if not (0 <= cx < grid_w and 0 <= cy < grid_h):
                continue
            if isinstance(raw_cell, dict) and all(k in raw_cell for k in ("ui_x0", "ui_y0", "ui_x1", "ui_y1")):
                ux0 = float(np.clip(raw_cell.get("ui_x0", 0.0), 0.0, 1.0))
                uy0 = float(np.clip(raw_cell.get("ui_y0", 0.0), 0.0, 1.0))
                ux1 = float(np.clip(raw_cell.get("ui_x1", 1.0), 0.0, 1.0))
                uy1 = float(np.clip(raw_cell.get("ui_y1", 1.0), 0.0, 1.0))
                rx0 = int(round(map_x + ux0 * max(0, map_w - 1)))
                ry0 = int(round(map_y + uy0 * max(0, map_h - 1)))
                rx1 = int(round(map_x + ux1 * max(0, map_w - 1)))
                ry1 = int(round(map_y + uy1 * max(0, map_h - 1)))
                rx = max(map_x, min(rx0, rx1))
                ry = max(map_y, min(ry0, ry1))
                rw = max(2, abs(rx1 - rx0))
                rh = max(2, abs(ry1 - ry0))
                rw = min(rw, map_x + map_w - rx)
                rh = min(rh, map_y + map_h - ry)
                rect_hit = (rx, ry, rw, rh)
            elif isinstance(raw_cell, dict) and "ui_cx" in raw_cell and "ui_cy" in raw_cell:
                uxc = float(np.clip(raw_cell.get("ui_cx", 0.5), 0.0, 1.0))
                uyc = float(np.clip(raw_cell.get("ui_cy", 0.5), 0.0, 1.0))
                uw = float(np.clip(raw_cell.get("ui_w", 0.08), 0.01, 1.0))
                uh = float(np.clip(raw_cell.get("ui_h", 0.08), 0.01, 1.0))
                rw = max(2, int(round(uw * map_w)))
                rh = max(2, int(round(uh * map_h)))
                rx = int(round(map_x + uxc * max(0, map_w - 1))) - rw // 2
                ry = int(round(map_y + uyc * max(0, map_h - 1))) - rh // 2
                rx = max(map_x, min(rx, map_x + map_w - rw))
                ry = max(map_y, min(ry, map_y + map_h - rh))
                rect_hit = (rx, ry, rw, rh)
            else:
                rect_hit = (map_x + cx * cell_w, map_y + cy * cell_h, cell_w, cell_h)
            self._doc_grid_cells.append({
                "rect": rect_hit,
                "grid_x": cx,
                "grid_y": cy,
                "level": level,
            })

    # ── Envelope mesh accessor ─────────────────────────────────────────────────

    @property
    def envelope_meshes(self) -> dict:
        """Current floor/walls/ceiling triangle soups from the envelope model.

        Each value is an (N, 3, 3) float64 array (N triangles × 3 vertices × xyz).
        Empty on first access; populated after the first call to _room_grid_panel.
        """
        meshes = dict(self._envelope_meshes)
        if self._applied_floor_meshes:
            meshes.update(self._applied_floor_meshes)
        return meshes

    def _ensure_room_workspace(self) -> bool:
        if self._room_workspace is not None:
            return True
        if self._host_station is not None and bool(getattr(self._host_station, "is_unfinished", False)):
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
            config_path=self._library_yaml_path,
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
        scene_ws = getattr(station, "_room_workspace_ref", None)
        if scene_ws is not None:
            self.bind_scene_workspace(scene_ws)

    def bind_scene_workspace(self, room_workspace):
        """Attach the live room object registry used to seed grid occupancy."""
        self._scene_workspace = room_workspace
        if self._ensure_room_workspace() and self._room_workspace is not None:
            if hasattr(self._room_workspace, "set_scene_objects_from_workspace"):
                self._room_workspace.set_scene_objects_from_workspace(room_workspace)

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

        left_cfg: dict = {}
        center_cfg: dict = {}
        right_cfg  = _load_panel("right",  "right_controls.yaml")
        
        # Load library.yaml for RoomTileLibraryPanel configuration
        library_yaml_path = os.path.join(base_dir, "library.yaml")
        
        return cls(station_cfg, left_cfg, center_cfg, right_cfg, library_yaml_path)


# Generic camera duty-station HUD support was removed.
