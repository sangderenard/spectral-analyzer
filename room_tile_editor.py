from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
        GL_ARRAY_BUFFER, GL_BLEND, GL_FALSE, GL_FLOAT, GL_ONE_MINUS_SRC_ALPHA,
        GL_SRC_ALPHA, GL_STATIC_DRAW, GL_TRIANGLES, GL_TRUE,
        glBindBuffer, glBindVertexArray, glBlendFunc, glBufferData,
        glDeleteBuffers, glDeleteVertexArrays, glDrawArrays, glEnable,
        glEnableVertexAttribArray, glGenBuffers, glGenVertexArrays,
        glGetUniformLocation, glUniform1f, glUniform3f, glUniform4f,
        glUniformMatrix4fv, glUseProgram, glVertexAttribPointer,
    )
    _HAS_GL = True
except ImportError:
    _HAS_GL = False


_BG = (18, 20, 28, 240)
_HDR_BG = (28, 32, 44)
_ITEM_BG = (22, 25, 35)
_SEL_BG = (38, 55, 90)
_TEXT = (210, 220, 230)
_DIM = (120, 130, 145)
_ACCENT = (60, 110, 200)
_GRID = (48, 56, 72)
_GRID_ALT = (34, 39, 52)
_GRID_LINE = (74, 84, 104)
_HILITE_FLOOR = (72, 122, 210)
_HILITE_CEIL = (120, 176, 220)
_HILITE_WALL = (220, 154, 84)
_HILITE_MARKER = (120, 228, 172)

_CELL_SIZE_M = 1.0
_LEVEL_HEIGHT_M = 1.0
_ROT_STEPS = 12


@dataclass
class TileMeshItem:
    mesh_id: str
    label: str
    bbox_size: np.ndarray
    pos_xy: np.ndarray
    z_min: float = 0.0
    yaw_step: int = 0
    render_style: str = "box"
    blueprint_id: str = ""
    programmatic_blueprint_id: str = ""
    requires_custom_confirm: bool = False


def _parse_obj_vertices(path: str) -> Optional[np.ndarray]:
    verts: List[Tuple[float, float, float]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line.startswith("v "):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    except Exception:
        return None
    if not verts:
        return None
    return np.asarray(verts, dtype=np.float64)


def _parse_ply_vertices(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except Exception:
        return None
    if not lines or not lines[0].strip().lower().startswith("ply"):
        return None
    v_count = 0
    header_end = -1
    for idx, ln in enumerate(lines):
        low = ln.strip().lower()
        if low.startswith("element vertex"):
            parts = low.split()
            if len(parts) >= 3:
                v_count = int(parts[2])
        if low == "end_header":
            header_end = idx
            break
    if header_end < 0 or v_count <= 0:
        return None
    verts: List[Tuple[float, float, float]] = []
    for ln in lines[header_end + 1: header_end + 1 + v_count]:
        parts = ln.strip().split()
        if len(parts) < 3:
            continue
        verts.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not verts:
        return None
    return np.asarray(verts, dtype=np.float64)


def _parse_mesh_vertices(path: str) -> Optional[np.ndarray]:
    ext = Path(path).suffix.lower()
    if ext == ".obj":
        return _parse_obj_vertices(path)
    if ext == ".ply":
        return _parse_ply_vertices(path)
    if ext == ".npy":
        try:
            arr = np.load(path)
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3]
        except Exception:
            return None
    return None


@dataclass(frozen=True)
class RoomTilePreset:
    preset_id: str
    label: str
    category: str
    kind: str
    mesh_type: str
    surfaces: Tuple[str, ...]
    footprint_xy: Tuple[int, int]
    level_span: int
    color_rgb: Tuple[float, float, float]
    summary: str = ""
    radius_cells: float = 1.0
    blueprint_id: str = ""
    programmatic_blueprint_id: str = ""
    requires_custom_confirm: bool = False
    prefab_available: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "RoomTilePreset":
        color = tuple(float(v) for v in data.get("color_rgb", [0.25, 0.35, 0.50]))
        surfaces = tuple(str(v) for v in data.get("surfaces", []))
        footprint = data.get("footprint_xy", [1, 1])
        return cls(
            preset_id=str(data.get("id", "")),
            label=str(data.get("label", data.get("id", "Preset"))),
            category=str(data.get("category", "room_tiles")),
            kind=str(data.get("kind", "room_tile")),
            mesh_type=str(data.get("mesh_type", "flat")),
            surfaces=surfaces,
            footprint_xy=(max(1, int(footprint[0])), max(1, int(footprint[1]))),
            level_span=max(1, int(data.get("level_span", 1))),
            color_rgb=(float(color[0]), float(color[1]), float(color[2])),
            summary=str(data.get("summary", "")),
            radius_cells=float(data.get("radius_cells", 1.0)),
            blueprint_id=str(data.get("blueprint_id", "")),
            programmatic_blueprint_id=str(data.get("programmatic_blueprint_id", "")),
            requires_custom_confirm=bool(data.get("requires_custom_confirm", False)),
            prefab_available=bool(data.get("prefab_available", False)),
        )


@dataclass
class RoomTileInstance:
    instance_id: str
    preset_id: str
    grid_x: int
    grid_y: int
    level: int
    rotation: int = 0


@dataclass
class ProjectionCell:
    grid_x: int
    grid_y: int
    level: int
    rect: pygame.Rect


class RoomTileLibraryPanel:
    PAD = 6
    HDR_H = 24
    TOOL_H = 24
    TAB_H = 22
    ROW_H = 34

    def __init__(self, state: dict, presets: Dict[str, RoomTilePreset],
                 title: str = "ROOM LIBRARY",
                 accent_rgb: Tuple[int, int, int] = _ACCENT,
                 on_rotate_left: Optional[Callable[[], None]] = None,
                 on_rotate_right: Optional[Callable[[], None]] = None,
                 on_import_mesh: Optional[Callable[[], None]] = None):
        pygame.font.init()
        self.state = state
        self._presets = presets
        self._title = title
        self._accent = tuple(int(v * 255) if isinstance(v, float) else int(v)
                             for v in accent_rgb)
        self._font = pygame.font.SysFont("consolas", 13)
        self._font_s = pygame.font.SysFont("consolas", 11)
        self._scroll = 0
        self._tool_rects: Dict[str, pygame.Rect] = {}
        self._tab_rects: Dict[str, pygame.Rect] = {}
        self._item_rects: Dict[str, pygame.Rect] = {}
        self._content_h = 0
        self._rot_l_rect: Optional[pygame.Rect] = None
        self._rot_r_rect: Optional[pygame.Rect] = None
        self._import_rect: Optional[pygame.Rect] = None
        self._scale_minus_rect: Optional[pygame.Rect] = None
        self._scale_plus_rect: Optional[pygame.Rect] = None
        self._scope_arch_rect: Optional[pygame.Rect] = None
        self._scope_inst_rect: Optional[pygame.Rect] = None
        self._list_top = self.HDR_H + self.TOOL_H + self.TAB_H * 2 + 18
        self._on_rotate_left = on_rotate_left
        self._on_rotate_right = on_rotate_right
        self._on_import_mesh = on_import_mesh
        self._categories = [
            "room_tiles", "lights", "doors", "windows", "cameras", "duty_stations"
        ]
        self.state.setdefault("palette_tool", "create")
        self.state.setdefault("palette_category", self._categories[0])
        self.state.setdefault("tile_import_scale", 1.0)
        self.state.setdefault("tile_editor_scope", "archetype")
        if "palette_selected" not in self.state:
            if "flat_floor" in self._presets:
                self.state["palette_selected"] = "flat_floor"
            else:
                initial = self.filtered_items()
                self.state["palette_selected"] = initial[0].preset_id if initial else ""

    def filtered_items(self) -> List[RoomTilePreset]:
        category = str(self.state.get("palette_category", self._categories[0]))
        items = [preset for preset in self._presets.values() if preset.category == category]
        items.sort(key=lambda preset: (preset.label.lower(), preset.preset_id))
        return items

    def _sync_tile_editor_from_selected_preset(self):
        if str(self.state.get("room_view", "plan")) != "tile_editor":
            return
        if str(self.state.get("tile_editor_scope", "archetype")) != "archetype":
            return
        preset_id = str(self.state.get("palette_selected", ""))
        preset = self._presets.get(preset_id)
        if preset is None:
            return
        self.state["tile_editor_footprint_x"] = max(1, int(preset.footprint_xy[0]))
        self.state["tile_editor_footprint_y"] = max(1, int(preset.footprint_xy[1]))
        self.state["tile_editor_levels"] = max(1, int(preset.level_span))
        self.state["tile_editor_seed_request"] = True
        self.state["tile_editor_selected_mesh"] = ""

    def render(self, width: int, height: int) -> pygame.Surface:
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill(_BG)
        self._tool_rects = {}
        self._tab_rects = {}
        self._item_rects = {}
        self._rot_l_rect = None
        self._rot_r_rect = None
        self._import_rect = None
        self._scale_minus_rect = None
        self._scale_plus_rect = None
        self._scope_arch_rect = None
        self._scope_inst_rect = None

        pygame.draw.rect(surface, self._accent, pygame.Rect(0, 0, width, self.HDR_H))
        header = self._font.render(f"  {self._title}", True, _TEXT)
        surface.blit(header, (self.PAD, 4))

        tool_names = ("create", "move", "delete")
        tool_width = max(40, (width - self.PAD * 2 - 4) // len(tool_names))
        top = self.HDR_H + 4
        for index, tool_name in enumerate(tool_names):
            rect = pygame.Rect(self.PAD + index * (tool_width + 2), top, tool_width, self.TOOL_H)
            self._tool_rects[tool_name] = rect
            active = self.state.get("palette_tool") == tool_name
            bg = self._accent if active else (28, 32, 44)
            pygame.draw.rect(surface, bg, rect, border_radius=3)
            pygame.draw.rect(surface, (70, 80, 100), rect, 1, border_radius=3)
            label = self._font_s.render(tool_name.upper(), True, _TEXT)
            surface.blit(label, (rect.x + (rect.w - label.get_width()) // 2,
                                 rect.y + (rect.h - label.get_height()) // 2))

        top += self.TOOL_H + 6
        tab_width = max(30, (width - self.PAD * 2 - 12) // 3)
        row_height = self.TAB_H
        for index, category in enumerate(self._categories):
            row = index // 3
            column = index % 3
            rect = pygame.Rect(
                self.PAD + column * (tab_width + 2),
                top + row * (row_height + 2),
                tab_width,
                row_height,
            )
            self._tab_rects[category] = rect
            active = self.state.get("palette_category") == category
            bg = (40, 90, 160) if active else _HDR_BG
            pygame.draw.rect(surface, bg, rect, border_radius=3)
            pygame.draw.rect(surface, (70, 80, 100), rect, 1, border_radius=3)
            short = category.replace("_", " ").title()
            label = self._font_s.render(short, True, _TEXT if active else _DIM)
            surface.blit(label, (rect.x + (rect.w - label.get_width()) // 2,
                                 rect.y + (rect.h - label.get_height()) // 2))

        top += row_height * 3 + 8
        if str(self.state.get("room_view", "plan")) == "tile_editor":
            scope = str(self.state.get("tile_editor_scope", "archetype"))
            scope_btn_w = max(62, (width - self.PAD * 2 - 2) // 2)
            self._scope_arch_rect = pygame.Rect(self.PAD, top, scope_btn_w, self.TOOL_H)
            self._scope_inst_rect = pygame.Rect(self.PAD + scope_btn_w + 2, top, scope_btn_w, self.TOOL_H)
            for key, text, rect in (
                ("archetype", "ARCHETYPE", self._scope_arch_rect),
                ("placed", "PLACED", self._scope_inst_rect),
            ):
                active = (scope == key)
                bg = self._accent if active else (28, 32, 44)
                pygame.draw.rect(surface, bg, rect, border_radius=3)
                pygame.draw.rect(surface, (70, 80, 100), rect, 1, border_radius=3)
                lbl = self._font_s.render(text, True, _TEXT)
                surface.blit(lbl, (rect.x + (rect.w - lbl.get_width()) // 2,
                                   rect.y + (rect.h - lbl.get_height()) // 2))
            top += self.TOOL_H + 4

            btn_w = max(62, (width - self.PAD * 2 - 4) // 3)
            self._rot_l_rect = pygame.Rect(self.PAD, top, btn_w, self.TOOL_H)
            self._rot_r_rect = pygame.Rect(self.PAD + btn_w + 2, top, btn_w, self.TOOL_H)
            self._import_rect = pygame.Rect(self.PAD + 2 * (btn_w + 2), top, btn_w, self.TOOL_H)
            for text, rect in (("ROT L", self._rot_l_rect), ("ROT R", self._rot_r_rect), ("IMPORT", self._import_rect)):
                pygame.draw.rect(surface, (28, 32, 44), rect, border_radius=3)
                pygame.draw.rect(surface, (70, 80, 100), rect, 1, border_radius=3)
                lbl = self._font_s.render(text, True, _TEXT)
                surface.blit(lbl, (rect.x + (rect.w - lbl.get_width()) // 2,
                                   rect.y + (rect.h - lbl.get_height()) // 2))
            top += self.TOOL_H + 4

            self._scale_minus_rect = pygame.Rect(self.PAD, top, 22, 22)
            self._scale_plus_rect = pygame.Rect(self.PAD + 26, top, 22, 22)
            for glyph, rect in (("-", self._scale_minus_rect), ("+", self._scale_plus_rect)):
                pygame.draw.rect(surface, (28, 32, 44), rect, border_radius=3)
                pygame.draw.rect(surface, (70, 80, 100), rect, 1, border_radius=3)
                lbl = self._font.render(glyph, True, _TEXT)
                surface.blit(lbl, (rect.x + (rect.w - lbl.get_width()) // 2,
                                   rect.y + (rect.h - lbl.get_height()) // 2 - 1))
            scale = float(self.state.get("tile_import_scale", 1.0))
            scale_lbl = self._font_s.render(f"Scale {scale:.2f}", True, _TEXT)
            surface.blit(scale_lbl, (self.PAD + 56, top + 4))
            top += 28
            sel_mesh = str(self.state.get("tile_editor_selected_mesh", "")) or "(none selected)"
            selected = str(self.state.get("palette_selected", "")) or "(none selected)"
            info = self._font_s.render(f"Archetype: {selected}", True, _TEXT)
            surface.blit(info, (self.PAD, top + 1))
            info_scope = self._font_s.render(f"Scope: {scope}", True, _DIM)
            surface.blit(info_scope, (self.PAD, top + 16))
            info2 = self._font_s.render(f"Mesh: {sel_mesh}", True, _DIM)
            surface.blit(info2, (self.PAD, top + 31))
            top += 48

        list_top = top
        self._list_top = list_top
        items = self.filtered_items()
        selected = str(self.state.get("palette_selected", ""))
        current_y = list_top - self._scroll
        for preset in items:
            rect = pygame.Rect(self.PAD, current_y, width - self.PAD * 2 - 8, self.ROW_H)
            if rect.bottom >= list_top and rect.top < height:
                bg = _SEL_BG if preset.preset_id == selected else _ITEM_BG
                pygame.draw.rect(surface, bg, rect, border_radius=4)
                title = self._font_s.render(preset.label, True, _TEXT)
                meta = self._font_s.render(
                    f"{preset.mesh_type}  {preset.footprint_xy[0]}x{preset.footprint_xy[1]}  z{preset.level_span}",
                    True,
                    _DIM,
                )
                surface.blit(title, (rect.x + 6, rect.y + 4))
                surface.blit(meta, (rect.x + 6, rect.y + 18))
                self._item_rects[preset.preset_id] = rect
            current_y += self.ROW_H + 4

        self._content_h = max(0, len(items) * (self.ROW_H + 4))
        list_height = max(1, height - list_top - 2)
        max_scroll = max(0, self._content_h - list_height)
        self._scroll = int(np.clip(self._scroll, 0, max_scroll))
        if self._content_h > list_height:
            bar_height = max(20, int(list_height * list_height / self._content_h))
            bar_y = list_top + int(self._scroll * (list_height - bar_height) / max(max_scroll, 1))
            pygame.draw.rect(surface, (50, 55, 70), pygame.Rect(width - 6, list_top, 6, list_height))
            pygame.draw.rect(surface, self._accent, pygame.Rect(width - 6, bar_y, 6, bar_height))
        return surface

    def handle_event(self, event, x_off: int = 0, y_off: int = 0) -> bool:
        if event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            lx = mx - x_off
            ly = my - y_off
            if lx < 0 or ly < self._list_top:
                return False
            list_height = max(1, pygame.display.get_surface().get_height() - (self._list_top + 2))
            max_scroll = max(0, self._content_h - list_height)
            self._scroll = int(np.clip(self._scroll - event.y * self.ROW_H, 0, max_scroll))
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            lx = event.pos[0] - x_off
            ly = event.pos[1] - y_off
            for tool_name, rect in self._tool_rects.items():
                if rect.collidepoint(lx, ly):
                    self.state["palette_tool"] = tool_name
                    return True
            for category, rect in self._tab_rects.items():
                if rect.collidepoint(lx, ly):
                    self.state["palette_category"] = category
                    items = self.filtered_items()
                    if category == "room_tiles" and "flat_floor" in self._presets:
                        self.state["palette_selected"] = "flat_floor"
                    elif items:
                        self.state["palette_selected"] = items[0].preset_id
                    self._sync_tile_editor_from_selected_preset()
                    return True
            if str(self.state.get("room_view", "plan")) == "tile_editor":
                if self._scope_arch_rect and self._scope_arch_rect.collidepoint(lx, ly):
                    self.state["tile_editor_scope"] = "archetype"
                    self._sync_tile_editor_from_selected_preset()
                    return True
                if self._scope_inst_rect and self._scope_inst_rect.collidepoint(lx, ly):
                    self.state["tile_editor_scope"] = "placed"
                    self.state["tile_editor_seed_request"] = True
                    self.state["tile_editor_selected_mesh"] = ""
                    return True
                if self._rot_l_rect and self._rot_l_rect.collidepoint(lx, ly):
                    if self._on_rotate_left is not None:
                        self._on_rotate_left()
                    return True
                if self._rot_r_rect and self._rot_r_rect.collidepoint(lx, ly):
                    if self._on_rotate_right is not None:
                        self._on_rotate_right()
                    return True
                if self._import_rect and self._import_rect.collidepoint(lx, ly):
                    if self._on_import_mesh is not None:
                        self._on_import_mesh()
                    return True
                if self._scale_minus_rect and self._scale_minus_rect.collidepoint(lx, ly):
                    sc = float(self.state.get("tile_import_scale", 1.0))
                    self.state["tile_import_scale"] = float(np.clip(sc - 0.1, 0.1, 20.0))
                    return True
                if self._scale_plus_rect and self._scale_plus_rect.collidepoint(lx, ly):
                    sc = float(self.state.get("tile_import_scale", 1.0))
                    self.state["tile_import_scale"] = float(np.clip(sc + 0.1, 0.1, 20.0))
                    return True
            for preset_id, rect in self._item_rects.items():
                if rect.collidepoint(lx, ly):
                    self.state["palette_selected"] = preset_id
                    self._sync_tile_editor_from_selected_preset()
                    return True
        return False


class RoomTileWorkspace:
    def __init__(
        self,
        state: dict,
        presets: Dict[str, RoomTilePreset],
        cfg: Optional[dict] = None,
        on_station_anchor_changed: Optional[Callable[[int, int, int, int], None]] = None,
    ):
        pygame.font.init()
        self.state = state
        self.presets = presets
        self.cfg = cfg or {}
        editor_cfg = self.cfg.get("room_editor", {})
        grid_size = editor_cfg.get("grid_size", [8, 8])
        anchor = editor_cfg.get("station_anchor", [max(0, int(grid_size[0] // 2)), 0])
        self.state.setdefault("room_view", "plan")
        self.state.setdefault("room_level", 0)
        self.state.setdefault("room_width_cells", max(2, int(grid_size[0])))
        self.state.setdefault("room_depth_cells", max(2, int(grid_size[1])))
        self.state.setdefault("room_station_x", int(anchor[0]))
        self.state.setdefault("room_station_y", int(anchor[1]))
        # The deployed map anchor is always the room-control station.
        if "room_control_station" in self.presets:
            self.state["room_station_preset_id"] = "room_control_station"
        else:
            self.state.setdefault("room_station_preset_id", "")
        self.state.setdefault("room_selected_instance", "")
        self.state.setdefault("tile_editor_footprint_x", 2)
        self.state.setdefault("tile_editor_footprint_y", 2)
        self.state.setdefault("tile_editor_levels", 2)
        self.state.setdefault("tile_editor_selected_mesh", "")
        self.state.setdefault("tile_editor_scope", "archetype")
        self._on_station_anchor_changed = on_station_anchor_changed
        self.instances: List[RoomTileInstance] = []
        self.tile_mesh_items_by_tile: Dict[str, List[TileMeshItem]] = {}
        self.tile_mesh_items_by_instance: Dict[str, List[TileMeshItem]] = {}
        self.fabrication_orders_by_instance: Dict[str, Dict[str, Any]] = {}
        self._next_instance = 1
        self._seed_default_instances()
        self._move_payload: Optional[Tuple[str, str]] = None
        self._projection_cells: List[ProjectionCell] = []
        self._button_rects: Dict[str, pygame.Rect] = {}
        self._minus_plus_rects: Dict[str, pygame.Rect] = {}
        self._grid_rect = pygame.Rect(0, 0, 0, 0)
        self._vao = None
        self._vbo = None
        self._vert_count = 0
        self._gl_dirty = True
        self._font = pygame.font.SysFont("consolas", 13)
        self._font_s = pygame.font.SysFont("consolas", 11)
        self._active_levels: Tuple[int, int] = (0, 0)

    def build_gl(self):
        if not _HAS_GL:
            return
        self._rebuild_gl()

    def invalidate_gl(self):
        self._gl_dirty = True

    def occupied_level_bounds(self) -> Tuple[int, int]:
        low = 0
        high = 0
        for instance in self.instances:
            preset = self.presets.get(instance.preset_id)
            if preset is None:
                continue
            low = min(low, instance.level)
            high = max(high, instance.level + preset.level_span - 1)
        return low, high

    def selected_preset(self) -> Optional[RoomTilePreset]:
        preset_id = str(self.state.get("palette_selected", ""))
        return self.presets.get(preset_id)

    def _seed_default_instances(self):
        """Place the default room layout: fabricator station adjacent to the room-control anchor."""
        if self.instances:
            return  # already populated (e.g. loaded from saved state)
        fab_preset = self.presets.get("fabricator_station")
        if fab_preset is None:
            return
        anchor_x = int(self.state.get("room_station_x", 0))
        anchor_y = int(self.state.get("room_station_y", 0))
        sp = self._station_preset()
        anchor_w = int(sp.footprint_xy[0]) if sp is not None else 2
        fab_x = anchor_x + anchor_w  # immediately to the right of the anchor
        fab_y = anchor_y
        fab_instance = RoomTileInstance(
            instance_id=f"tile_{self._next_instance:04d}",
            preset_id="fabricator_station",
            grid_x=fab_x,
            grid_y=fab_y,
            level=0,
        )
        self._next_instance += 1
        self.instances.append(fab_instance)

    def _clamp_station(self):
        width = int(self.state.get("room_width_cells", 8))
        depth = int(self.state.get("room_depth_cells", 8))
        sp = self._station_preset()
        fx, fy = (1, 1) if sp is None else sp.footprint_xy
        self.state["room_station_x"] = int(np.clip(self.state.get("room_station_x", 0), 0, max(0, width - fx)))
        self.state["room_station_y"] = int(np.clip(self.state.get("room_station_y", 0), 0, max(0, depth - fy)))

    def _station_preset(self) -> Optional[RoomTilePreset]:
        # Map anchor station is canonical room-control station, never another tile type.
        if "room_control_station" in self.presets:
            return self.presets.get("room_control_station")
        pid = str(self.state.get("room_station_preset_id", ""))
        preset = self.presets.get(pid)
        if preset is not None:
            return preset
        return None

    def _station_cells(self) -> List[Tuple[int, int, int]]:
        sp = self._station_preset()
        if sp is None:
            return [(int(self.state.get("room_station_x", 0)), int(self.state.get("room_station_y", 0)), 0)]
        sx = int(self.state.get("room_station_x", 0))
        sy = int(self.state.get("room_station_y", 0))
        out: List[Tuple[int, int, int]] = []
        for dx in range(sp.footprint_xy[0]):
            for dy in range(sp.footprint_xy[1]):
                for dz in range(sp.level_span):
                    out.append((sx + dx, sy + dy, dz))
        return out

    def _overlaps_station(self, grid_x: int, grid_y: int, level: int,
                         fx: int, fy: int, fz: int) -> bool:
        if fz <= 0:
            return False
        placed: set[Tuple[int, int, int]] = set()
        for dx in range(fx):
            for dy in range(fy):
                for dz in range(fz):
                    placed.add((grid_x + dx, grid_y + dy, level + dz))
        return any(cell in placed for cell in self._station_cells())

    def resize_room(self, axis: str, delta: int):
        key = "room_width_cells" if axis == "x" else "room_depth_cells"
        value = int(self.state.get(key, 8)) + delta
        self.state[key] = int(np.clip(value, 2, 24))
        self._clamp_station()
        self.invalidate_gl()

    def adjust_level(self, delta: int):
        value = int(self.state.get("room_level", 0)) + delta
        self.state["room_level"] = int(np.clip(value, -99, 99))

    def set_view(self, view_name: str):
        self.state["room_view"] = view_name
        if view_name == "tile_editor":
            self.state["tile_editor_seed_request"] = True
            self.state["tile_editor_selected_mesh"] = ""

    def _active_tile_key(self) -> str:
        scope = str(self.state.get("tile_editor_scope", "archetype"))
        if scope == "placed":
            sel_id = str(self.state.get("room_selected_instance", ""))
            if sel_id:
                return f"inst:{sel_id}"
            return "__placed_none__"
        return str(self.state.get("palette_selected", "") or "__default__")

    def _selected_instance(self) -> Optional[RoomTileInstance]:
        sel_id = str(self.state.get("room_selected_instance", ""))
        if not sel_id:
            return None
        for instance in self.instances:
            if instance.instance_id == sel_id:
                return instance
        return None

    def _active_tile_preset(self) -> Optional[RoomTilePreset]:
        scope = str(self.state.get("tile_editor_scope", "archetype"))
        if scope == "placed":
            inst = self._selected_instance()
            if inst is not None:
                return self.presets.get(inst.preset_id)
        return self.selected_preset()

    def _active_tile_items(self) -> List[TileMeshItem]:
        scope = str(self.state.get("tile_editor_scope", "archetype"))
        key = self._active_tile_key()
        if scope == "placed":
            # If no placed tile is selected yet, fall back to archetype editing
            # so the user can still edit duty-station tile meshes from gallery.
            if self._selected_instance() is None:
                akey = str(self.state.get("palette_selected", "") or "__default__")
                if akey not in self.tile_mesh_items_by_tile:
                    self.tile_mesh_items_by_tile[akey] = []
                return self.tile_mesh_items_by_tile[akey]
            if key not in self.tile_mesh_items_by_instance:
                self.tile_mesh_items_by_instance[key] = []
            return self.tile_mesh_items_by_instance[key]
        if key not in self.tile_mesh_items_by_tile:
            self.tile_mesh_items_by_tile[key] = []
        return self.tile_mesh_items_by_tile[key]

    def _set_active_tile_items(self, items: List[TileMeshItem]):
        scope = str(self.state.get("tile_editor_scope", "archetype"))
        key = self._active_tile_key()
        if scope == "placed":
            if self._selected_instance() is None:
                akey = str(self.state.get("palette_selected", "") or "__default__")
                self.tile_mesh_items_by_tile[akey] = items
                return
            self.tile_mesh_items_by_instance[key] = items
            return
        self.tile_mesh_items_by_tile[key] = items

    def rotate_selection_left(self):
        self._rotate_selection(-1)

    def rotate_selection_right(self):
        self._rotate_selection(1)

    def _rotate_selection(self, delta_step: int):
        if str(self.state.get("room_view", "plan")) == "tile_editor":
            sel = str(self.state.get("tile_editor_selected_mesh", ""))
            if not sel:
                return
            for item in self._active_tile_items():
                if item.mesh_id == sel:
                    item.yaw_step = int((item.yaw_step + delta_step) % _ROT_STEPS)
                    self.invalidate_gl()
                    return
            return
        sel_id = str(self.state.get("room_selected_instance", ""))
        if not sel_id:
            return
        for instance in self.instances:
            if instance.instance_id == sel_id:
                instance.rotation = int((instance.rotation + delta_step) % _ROT_STEPS)
                self.invalidate_gl()
                return

    def import_mesh_dialog(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Import Mesh",
                filetypes=[
                    ("Mesh files", "*.obj *.ply *.npy"),
                    ("OBJ", "*.obj"),
                    ("PLY", "*.ply"),
                    ("NumPy", "*.npy"),
                    ("All files", "*.*"),
                ],
            )
            root.destroy()
        except Exception:
            return
        if not path:
            return
        verts = _parse_mesh_vertices(path)
        if verts is None or len(verts) == 0:
            return
        scale = float(self.state.get("tile_import_scale", 1.0))
        verts = verts * scale
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        size = np.maximum(vmax - vmin, 1e-5)
        fx = max(1, int(math.ceil(float(size[0]) / _CELL_SIZE_M)))
        fy = max(1, int(math.ceil(float(size[1]) / _CELL_SIZE_M)))
        fz = max(1, int(math.ceil(float(size[2]) / _LEVEL_HEIGHT_M)))
        self.state["tile_editor_footprint_x"] = fx
        self.state["tile_editor_footprint_y"] = fy
        self.state["tile_editor_levels"] = fz
        items = self._active_tile_items()
        mesh_id = f"mesh_{len(items) + 1:03d}"
        label = os.path.basename(path)
        items.append(TileMeshItem(
            mesh_id=mesh_id,
            label=label,
            bbox_size=np.array([float(size[0]), float(size[1]), float(size[2])], np.float64),
            pos_xy=np.array([fx * 0.5 * _CELL_SIZE_M, fy * 0.5 * _CELL_SIZE_M], np.float64),
            z_min=0.0,
            yaw_step=0,
        ))
        self.state["tile_editor_selected_mesh"] = mesh_id
        self.invalidate_gl()

    def _instance_cells(self, instance: RoomTileInstance) -> Iterable[Tuple[int, int, int]]:
        preset = self.presets.get(instance.preset_id)
        if preset is None:
            return []
        width, depth = preset.footprint_xy
        for dx in range(width):
            for dy in range(depth):
                for dz in range(preset.level_span):
                    yield instance.grid_x + dx, instance.grid_y + dy, instance.level + dz

    def instance_at(self, grid_x: int, grid_y: int, level: int) -> Optional[RoomTileInstance]:
        for instance in reversed(self.instances):
            preset = self.presets.get(instance.preset_id)
            if preset is None:
                continue
            if (instance.grid_x <= grid_x < instance.grid_x + preset.footprint_xy[0]
                    and instance.grid_y <= grid_y < instance.grid_y + preset.footprint_xy[1]
                    and instance.level <= level < instance.level + preset.level_span):
                return instance
        return None

    def _station_hit(self, grid_x: int, grid_y: int, level: int) -> bool:
        return (grid_x, grid_y, level) in set(self._station_cells())

    def _place_instance(self, grid_x: int, grid_y: int, level: int):
        preset = self.selected_preset()
        if preset is None:
            return
        width = int(self.state.get("room_width_cells", 8))
        depth = int(self.state.get("room_depth_cells", 8))
        if grid_x < 0 or grid_y < 0:
            return
        if grid_x + preset.footprint_xy[0] > width or grid_y + preset.footprint_xy[1] > depth:
            return
        if preset.category != "duty_stations":
            if self._overlaps_station(
                grid_x,
                grid_y,
                level,
                int(preset.footprint_xy[0]),
                int(preset.footprint_xy[1]),
                int(preset.level_span),
            ):
                return
        instance = RoomTileInstance(
            instance_id=f"tile_{self._next_instance:04d}",
            preset_id=preset.preset_id,
            grid_x=grid_x,
            grid_y=grid_y,
            level=level,
        )
        self._next_instance += 1
        self.instances.append(instance)
        self.state["room_selected_instance"] = instance.instance_id
        self.invalidate_gl()

    def _move_selected_to(self, kind: str, payload_id: str, grid_x: int, grid_y: int, level: int):
        if kind == "station":
            old_x = int(self.state.get("room_station_x", 0))
            old_y = int(self.state.get("room_station_y", 0))
            self.state["room_station_x"] = grid_x
            self.state["room_station_y"] = grid_y
            self._clamp_station()
            new_x = int(self.state.get("room_station_x", 0))
            new_y = int(self.state.get("room_station_y", 0))
            if (new_x != old_x or new_y != old_y) and self._on_station_anchor_changed is not None:
                self._on_station_anchor_changed(old_x, old_y, new_x, new_y)
            self.invalidate_gl()
            return
        for instance in self.instances:
            if instance.instance_id == payload_id:
                preset = self.presets.get(instance.preset_id)
                if preset is None:
                    return
                width = int(self.state.get("room_width_cells", 8))
                depth = int(self.state.get("room_depth_cells", 8))
                nx = int(np.clip(grid_x, 0, width - preset.footprint_xy[0]))
                ny = int(np.clip(grid_y, 0, depth - preset.footprint_xy[1]))
                nz = int(np.clip(level, -99, 99))
                if preset.category != "duty_stations":
                    if self._overlaps_station(nx, ny, nz,
                                             int(preset.footprint_xy[0]),
                                             int(preset.footprint_xy[1]),
                                             int(preset.level_span)):
                        return
                instance.grid_x = nx
                instance.grid_y = ny
                instance.level = nz
                self.state["room_selected_instance"] = instance.instance_id
                self.invalidate_gl()
                return

    def delete_at(self, grid_x: int, grid_y: int, level: int):
        instance = self.instance_at(grid_x, grid_y, level)
        if instance is None:
            return
        self.instances = [item for item in self.instances if item.instance_id != instance.instance_id]
        if self.state.get("room_selected_instance") == instance.instance_id:
            self.state["room_selected_instance"] = ""
        self.invalidate_gl()

    def click_grid(self, grid_x: int, grid_y: int, level: int):
        if str(self.state.get("room_view", "plan")) == "tile_editor":
            self._click_tile_editor(grid_x, grid_y)
            return
        tool = str(self.state.get("palette_tool", "create"))
        if tool == "delete":
            self.delete_at(grid_x, grid_y, level)
            return
        if tool == "move":
            if self._move_payload is None:
                instance = self.instance_at(grid_x, grid_y, level)
                if instance is not None:
                    self._move_payload = ("instance", instance.instance_id)
                    self.state["room_selected_instance"] = instance.instance_id
                    return
                if self._station_hit(grid_x, grid_y, level):
                    self._move_payload = ("station", "station_anchor")
                    return
                return
            self._move_selected_to(self._move_payload[0], self._move_payload[1], grid_x, grid_y, level)
            self._move_payload = None
            return
        self._place_instance(grid_x, grid_y, level)

    def _click_tile_editor(self, grid_x: int, grid_y: int):
        if str(self.state.get("tile_editor_scope", "archetype")) == "placed" and self._selected_instance() is None:
            return
        tool = str(self.state.get("palette_tool", "create"))
        click_xy = np.array([(grid_x + 0.5) * _CELL_SIZE_M, (grid_y + 0.5) * _CELL_SIZE_M], np.float64)
        items = self._active_tile_items()
        if tool == "delete":
            sel = str(self.state.get("tile_editor_selected_mesh", ""))
            if sel:
                kept = [m for m in items if m.mesh_id != sel]
                self._set_active_tile_items(kept)
                self.state["tile_editor_selected_mesh"] = kept[0].mesh_id if kept else ""
                self.invalidate_gl()
            return
        if tool == "move":
            if self._move_payload is None:
                hit = self._tile_mesh_hit(click_xy)
                if hit is not None:
                    self.state["tile_editor_selected_mesh"] = hit.mesh_id
                    self._move_payload = ("tile_mesh", hit.mesh_id)
                return
            if self._move_payload[0] == "tile_mesh":
                self._move_tile_mesh_to(self._move_payload[1], click_xy)
                self._move_payload = None
            return
        hit = self._tile_mesh_hit(click_xy)
        if hit is not None:
            self.state["tile_editor_selected_mesh"] = hit.mesh_id

    def _tile_mesh_hit(self, click_xy: np.ndarray) -> Optional[TileMeshItem]:
        for item in reversed(self._active_tile_items()):
            half = 0.5 * item.bbox_size[:2]
            lo = item.pos_xy - half
            hi = item.pos_xy + half
            if lo[0] <= click_xy[0] <= hi[0] and lo[1] <= click_xy[1] <= hi[1]:
                return item
        return None

    def _move_tile_mesh_to(self, mesh_id: str, click_xy: np.ndarray):
        tile_w = max(1, int(self.state.get("tile_editor_footprint_x", 2))) * _CELL_SIZE_M
        tile_d = max(1, int(self.state.get("tile_editor_footprint_y", 2))) * _CELL_SIZE_M
        for item in self._active_tile_items():
            if item.mesh_id != mesh_id:
                continue
            delta = click_xy - item.pos_xy
            # Cardinal slide: lock to dominant axis for move action.
            if abs(delta[0]) >= abs(delta[1]):
                target = np.array([click_xy[0], item.pos_xy[1]], np.float64)
            else:
                target = np.array([item.pos_xy[0], click_xy[1]], np.float64)

            # Structural components should align exactly to tile extents.
            label_l = str(getattr(item, "label", "")).lower()
            if "floor" in label_l:
                item.bbox_size[0] = float(tile_w)
                item.bbox_size[1] = float(tile_d)
                target = np.array([tile_w * 0.5, tile_d * 0.5], np.float64)
            elif "wall" in label_l:
                if float(item.bbox_size[0]) >= float(item.bbox_size[1]):
                    item.bbox_size[0] = float(tile_w)
                    half_pre = 0.5 * item.bbox_size[:2]
                    target = np.array([
                        tile_w * 0.5,
                        tile_d - half_pre[1] if click_xy[1] >= tile_d * 0.5 else half_pre[1],
                    ], np.float64)
                else:
                    item.bbox_size[1] = float(tile_d)
                    half_pre = 0.5 * item.bbox_size[:2]
                    target = np.array([
                        tile_w - half_pre[0] if click_xy[0] >= tile_w * 0.5 else half_pre[0],
                        tile_d * 0.5,
                    ], np.float64)

            half = 0.5 * item.bbox_size[:2]
            target[0] = float(np.clip(target[0], half[0], tile_w - half[0]))
            target[1] = float(np.clip(target[1], half[1], tile_d - half[1]))

            # Wall snap
            snap_eps = 0.20
            if abs(target[0] - half[0]) <= snap_eps:
                target[0] = half[0]
            if abs(target[0] - (tile_w - half[0])) <= snap_eps:
                target[0] = tile_w - half[0]
            if abs(target[1] - half[1]) <= snap_eps:
                target[1] = half[1]
            if abs(target[1] - (tile_d - half[1])) <= snap_eps:
                target[1] = tile_d - half[1]

            # Corner snap when rotated
            if int(item.yaw_step) % _ROT_STEPS != 0:
                corners = np.array([
                    [half[0], half[1]],
                    [tile_w - half[0], half[1]],
                    [half[0], tile_d - half[1]],
                    [tile_w - half[0], tile_d - half[1]],
                ], np.float64)
                dists = np.linalg.norm(corners - target.reshape(1, 2), axis=1)
                ci = int(np.argmin(dists))
                if float(dists[ci]) <= 0.25:
                    target = corners[ci]

            item.pos_xy = target
            self.invalidate_gl()
            return

    def _draw_plan_cell(self, surface: pygame.Surface, rect: pygame.Rect,
                        preset: RoomTilePreset, selected: bool = False):
        inset = 6 if rect.w >= 18 else 2
        if "floor" in preset.surfaces:
            inner = rect.inflate(-inset * 2, -inset * 2)
            pygame.draw.rect(surface, self._rgb255(preset.color_rgb), inner, border_radius=2)
        if "ceiling" in preset.surfaces:
            pygame.draw.rect(surface, self._mix(self._rgb255(preset.color_rgb), _HILITE_CEIL, 0.6), rect.inflate(-4, -4), 2, border_radius=3)
        wall_color = self._mix(self._rgb255(preset.color_rgb), _HILITE_WALL, 0.6)
        if "north" in preset.surfaces:
            pygame.draw.line(surface, wall_color, rect.topleft, rect.topright, 3)
        if "south" in preset.surfaces:
            pygame.draw.line(surface, wall_color, rect.bottomleft, rect.bottomright, 3)
        if "west" in preset.surfaces:
            pygame.draw.line(surface, wall_color, rect.topleft, rect.bottomleft, 3)
        if "east" in preset.surfaces:
            pygame.draw.line(surface, wall_color, rect.topright, rect.bottomright, 3)
        if selected:
            pygame.draw.rect(surface, _HILITE_MARKER, rect.inflate(-2, -2), 2, border_radius=4)

    def _draw_projection_cell(self, surface: pygame.Surface, rect: pygame.Rect,
                              preset: RoomTilePreset, selected: bool = False):
        mid_y = rect.y + rect.h // 2
        fill = self._rgb255(preset.color_rgb)
        if "floor" in preset.surfaces:
            pygame.draw.rect(surface, fill, pygame.Rect(rect.x + 4, rect.bottom - 8, rect.w - 8, 4))
        if "ceiling" in preset.surfaces:
            pygame.draw.rect(surface, self._mix(fill, _HILITE_CEIL, 0.6), pygame.Rect(rect.x + 4, rect.y + 4, rect.w - 8, 4))
        if any(side in preset.surfaces for side in ("north", "south", "east", "west")):
            pygame.draw.line(surface, self._mix(fill, _HILITE_WALL, 0.6), (rect.x + 4, mid_y), (rect.right - 4, mid_y), 3)
        if selected:
            pygame.draw.rect(surface, _HILITE_MARKER, rect.inflate(-2, -2), 2, border_radius=4)

    def _projection_levels(self) -> List[int]:
        center = int(self.state.get("room_level", 0))
        return [center + offset for offset in range(4, -5, -1)]

    @staticmethod
    def _is_room_control_station_preset(preset: Optional[RoomTilePreset]) -> bool:
        if preset is None:
            return False
        if preset.category != "duty_stations":
            return False
        pid = str(getattr(preset, "preset_id", ""))
        if pid == "room_control_station":
            return True
        kind = str(getattr(preset, "kind", ""))
        if kind == "room_control_station":
            return True
        return ("room_control" in pid)

    @staticmethod
    def _is_camera_designer_station_preset(preset: Optional[RoomTilePreset]) -> bool:
        if preset is None:
            return False
        if preset.category != "duty_stations":
            return False
        pid = str(getattr(preset, "preset_id", ""))
        return pid == "camera_designer_station"

    def _seed_room_control_station_components(
        self,
        active_items: List[TileMeshItem],
        fx: int,
        fy: int,
        fz: int,
        scope: str,
    ):
        tile_w = float(fx) * float(_CELL_SIZE_M)
        tile_d = float(fy) * float(_CELL_SIZE_M)
        tile_h = float(fz) * float(_LEVEL_HEIGHT_M)

        cons = self.cfg.get("console", {}) if isinstance(self.cfg, dict) else {}
        scr = self.cfg.get("screen", {}) if isinstance(self.cfg, dict) else {}
        wings = self.cfg.get("side_wings", {}) if isinstance(self.cfg, dict) else {}

        cons_w = max(0.45, min(tile_w * 0.96, float(cons.get("width", 1.40))))
        cons_d = max(0.35, min(tile_d * 0.96, float(cons.get("depth", 0.62))))
        cons_h = max(0.45, min(tile_h * 0.96, float(cons.get("height", 0.88))))

        scr_w = max(0.28, min(cons_w * 0.96, float(scr.get("width", 1.10))))
        scr_h = max(0.24, min(tile_h * 0.90, float(scr.get("height", 0.72))))
        scr_t = max(0.04, min(0.14, 0.08 * _CELL_SIZE_M))

        wing_enabled = bool(wings.get("enabled", True))
        wing_w = max(0.06, min(tile_w * 0.25, float(wings.get("width", 0.14))))
        wing_h = max(0.22, min(tile_h * 0.90, float(wings.get("height", 0.52))))
        wing_d = max(0.08, min(cons_d * 0.95, cons_d * 0.90))

        floor_t = max(0.08, 0.10 * float(_LEVEL_HEIGHT_M))
        floor_w = max(0.80, min(tile_w * 0.98, cons_w + 2.0 * wing_w + 0.20))
        floor_d = max(0.80, min(tile_d * 0.98, cons_d + 0.18))

        wall_t = max(0.10, 0.14 * float(_CELL_SIZE_M))
        wall_w = max(0.80, min(tile_w * 0.98, cons_w + 2.0 * wing_w + 0.18))
        wall_h = max(0.70, tile_h * 0.95)

        y_console_center = min(tile_d - cons_d * 0.5 - wall_t - 0.08, tile_d * 0.50)
        y_console_center = max(cons_d * 0.5 + 0.02, y_console_center)
        y_wall = tile_d - wall_t * 0.5
        y_screen = min(tile_d - scr_t * 0.5 - wall_t, y_console_center + cons_d * 0.45)
        y_screen = max(scr_t * 0.5, y_screen)

        label_prefix = "Placed " if scope == "placed" else ""
        active_items.extend([
            TileMeshItem(
                mesh_id="mesh_001",
                label=f"{label_prefix}Console Body",
                bbox_size=np.array([cons_w, cons_d, cons_h], np.float64),
                pos_xy=np.array([tile_w * 0.5, y_console_center], np.float64),
                z_min=0.0,
                yaw_step=0,
            ),
            TileMeshItem(
                mesh_id="mesh_002",
                label=f"{label_prefix}Screen",
                bbox_size=np.array([scr_w, scr_t, scr_h], np.float64),
                pos_xy=np.array([tile_w * 0.5, y_screen], np.float64),
                z_min=max(0.0, cons_h - scr_h * 0.35),
                yaw_step=0,
            ),
            TileMeshItem(
                mesh_id="mesh_003",
                label=f"{label_prefix}Required Wall",
                bbox_size=np.array([wall_w, wall_t, wall_h], np.float64),
                pos_xy=np.array([tile_w * 0.5, y_wall], np.float64),
                z_min=0.0,
                yaw_step=0,
            ),
            TileMeshItem(
                mesh_id="mesh_004",
                label=f"{label_prefix}Required Floor",
                bbox_size=np.array([floor_w, floor_d, floor_t], np.float64),
                pos_xy=np.array([tile_w * 0.5, tile_d * 0.5], np.float64),
                z_min=0.0,
                yaw_step=0,
            ),
        ])

        if wing_enabled:
            wing_x_off = max(0.06, 0.5 * (cons_w + wing_w))
            active_items.extend([
                TileMeshItem(
                    mesh_id="mesh_005",
                    label=f"{label_prefix}Left Wing",
                    bbox_size=np.array([wing_w, wing_d, wing_h], np.float64),
                    pos_xy=np.array([tile_w * 0.5 - wing_x_off, y_console_center], np.float64),
                    z_min=0.0,
                    yaw_step=0,
                ),
                TileMeshItem(
                    mesh_id="mesh_006",
                    label=f"{label_prefix}Right Wing",
                    bbox_size=np.array([wing_w, wing_d, wing_h], np.float64),
                    pos_xy=np.array([tile_w * 0.5 + wing_x_off, y_console_center], np.float64),
                    z_min=0.0,
                    yaw_step=0,
                ),
            ])

    def _seed_camera_designer_station_components(
        self,
        active_items: List[TileMeshItem],
        fx: int,
        fy: int,
        fz: int,
        scope: str,
    ):
        self._seed_room_control_station_components(active_items, fx, fy, fz, scope)

        # Camera-designer station has rear rig instead of a wall backing.
        active_items[:] = [it for it in active_items if "Required Wall" not in str(it.label)]

        tile_w = float(fx) * float(_CELL_SIZE_M)
        tile_d = float(fy) * float(_CELL_SIZE_M)

        table_w = max(0.80, min(1.00, tile_w * 0.96))
        table_d = max(1.50, min(2.00, tile_d * 0.96))
        table_h = 0.90
        table_y = max(table_d * 0.5, tile_d - table_d * 0.5)

        jar_w = max(0.70, table_w * 0.96)
        jar_d = max(1.20, table_d * 0.96)
        jar_h = max(0.90, min(float(fz) * _LEVEL_HEIGHT_M * 0.8, 1.30))

        mini_w = max(0.55, jar_w * 0.80)
        mini_d = max(0.90, jar_d * 0.80)
        mini_h = max(0.35, jar_h * 0.45)

        label_prefix = "Placed " if scope == "placed" else ""
        active_items.extend([
            TileMeshItem(
                mesh_id="mesh_007",
                label=f"{label_prefix}Test Rig Table",
                bbox_size=np.array([table_w, table_d, table_h], np.float64),
                pos_xy=np.array([tile_w * 0.5, table_y], np.float64),
                z_min=0.0,
                yaw_step=0,
            ),
            TileMeshItem(
                mesh_id="mesh_008",
                label=f"{label_prefix}Bell Jar (Smokey Glass)",
                bbox_size=np.array([jar_w, jar_d, jar_h], np.float64),
                pos_xy=np.array([tile_w * 0.5, table_y], np.float64),
                z_min=table_h,
                yaw_step=0,
            ),
            TileMeshItem(
                mesh_id="mesh_009",
                label=f"{label_prefix}Mini Scene Volume",
                bbox_size=np.array([mini_w, mini_d, mini_h], np.float64),
                pos_xy=np.array([tile_w * 0.5, table_y], np.float64),
                z_min=table_h + 0.05,
                yaw_step=0,
            ),
        ])

    def _ensure_tile_editor_seeded_mesh(self):
        # Clear one-shot seed flag if present; seeding is now robust even without it.
        self.state.pop("tile_editor_seed_request", None)
        preset = self._active_tile_preset()
        if preset is not None:
            fx = max(1, int(preset.footprint_xy[0]))
            fy = max(1, int(preset.footprint_xy[1]))
            fz = max(1, int(preset.level_span))
            self.state["tile_editor_footprint_x"] = fx
            self.state["tile_editor_footprint_y"] = fy
            self.state["tile_editor_levels"] = fz

        active_items = self._active_tile_items()
        if not active_items:
            if preset is None:
                fx = max(1, int(self.state.get("tile_editor_footprint_x", 2)))
                fy = max(1, int(self.state.get("tile_editor_footprint_y", 2)))
                fz = max(1, int(self.state.get("tile_editor_levels", 2)))
            else:
                fx = max(1, int(preset.footprint_xy[0]))
                fy = max(1, int(preset.footprint_xy[1]))
                fz = max(1, int(preset.level_span))

            mesh_id = "mesh_001"
            scope = str(self.state.get("tile_editor_scope", "archetype"))
            if preset is not None and preset.category == "duty_stations":
                self._seed_duty_station_gestalt_item(active_items, fx, fy, fz, preset, scope)
            elif self._is_room_control_station_preset(preset):
                self._seed_room_control_station_components(active_items, fx, fy, fz, scope)
            elif self._is_camera_designer_station_preset(preset):
                self._seed_camera_designer_station_components(active_items, fx, fy, fz, scope)
            else:
                label = "Tile Mesh"
                if scope == "placed":
                    label = "Placed Tile Mesh"
                if preset is not None and preset.category == "duty_stations":
                    label = "Duty Station Mesh" if scope == "archetype" else "Placed Duty Station Mesh"

                active_items.append(TileMeshItem(
                    mesh_id=mesh_id,
                    label=label,
                    bbox_size=np.array([
                        max(0.5, fx * _CELL_SIZE_M * 0.8),
                        max(0.5, fy * _CELL_SIZE_M * 0.8),
                        max(0.5, fz * _LEVEL_HEIGHT_M * 0.8),
                    ], np.float64),
                    pos_xy=np.array([fx * 0.5 * _CELL_SIZE_M, fy * 0.5 * _CELL_SIZE_M], np.float64),
                    z_min=0.0,
                    yaw_step=0,
                ))

        selected_mesh = str(self.state.get("tile_editor_selected_mesh", ""))
        if not selected_mesh or not any(item.mesh_id == selected_mesh for item in active_items):
            self.state["tile_editor_selected_mesh"] = active_items[0].mesh_id if active_items else ""

    def render(self, width: int, height: int) -> pygame.Surface:
        active_items = self._active_tile_items()
        if str(self.state.get("room_view", "plan")) == "tile_editor":
            self._ensure_tile_editor_seeded_mesh()
            active_items = self._active_tile_items()

        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill(_BG)
        self._button_rects = {}
        self._minus_plus_rects = {}
        self._projection_cells = []

        toolbar_h = 76
        pygame.draw.rect(surface, _HDR_BG, pygame.Rect(0, 0, width, toolbar_h))

        view_names = ("plan", "front", "side", "tile_editor")
        button_w = 66
        for index, view_name in enumerate(view_names):
            rect = pygame.Rect(8 + index * (button_w + 4), 8, button_w, 22)
            self._button_rects[f"view:{view_name}"] = rect
            active = self.state.get("room_view") == view_name
            bg = (40, 90, 160) if active else (28, 32, 44)
            pygame.draw.rect(surface, bg, rect, border_radius=3)
            pygame.draw.rect(surface, (70, 80, 100), rect, 1, border_radius=3)
            label = self._font_s.render(view_name.upper(), True, _TEXT)
            surface.blit(label, (rect.x + (rect.w - label.get_width()) // 2,
                                 rect.y + (rect.h - label.get_height()) // 2))

        level_label = self._font.render(f"LEVEL {int(self.state.get('room_level', 0)):>3d}", True, _TEXT)
        surface.blit(level_label, (240, 10))
        for name, x_pos in (("level:-", 334), ("level:+", 360), ("dimx:-", 438), ("dimx:+", 464), ("dimy:-", 542), ("dimy:+", 568)):
            rect = pygame.Rect(x_pos, 8, 22, 22)
            self._minus_plus_rects[name] = rect
            pygame.draw.rect(surface, (28, 32, 44), rect, border_radius=3)
            pygame.draw.rect(surface, (70, 80, 100), rect, 1, border_radius=3)
            glyph = "+" if name.endswith("+") else "-"
            label = self._font.render(glyph, True, _TEXT)
            surface.blit(label, (rect.x + (rect.w - label.get_width()) // 2,
                                 rect.y + (rect.h - label.get_height()) // 2 - 1))

        dims = self._font.render(
            f"ROOM X {int(self.state.get('room_width_cells', 8)):>2d}   Y {int(self.state.get('room_depth_cells', 8)):>2d}",
            True,
            _TEXT,
        )
        surface.blit(dims, (394, 10))

        selected = self._active_tile_preset()
        sel_text = selected.label if selected is not None else "No preset selected"
        summary = self._font_s.render(sel_text, True, _DIM)
        surface.blit(summary, (8, 42))
        if str(self.state.get("room_view", "plan")) == "tile_editor":
            scope = str(self.state.get("tile_editor_scope", "archetype"))
            if scope == "placed":
                inst = self._selected_instance()
                inst_txt = inst.instance_id if inst is not None else "(select a placed tile first)"
                scope_lbl = self._font_s.render(f"edit scope: placed  target: {inst_txt}", True, _DIM)
                surface.blit(scope_lbl, (220, 42))
                if inst is not None:
                    preset = self.presets.get(inst.preset_id)
                    if preset is not None and preset.category == "duty_stations":
                        fab = self._fabrication_state(inst, preset)
                        pending = len(fab.get("fixed_pending", [])) + len(fab.get("custom_pending", []))
                        stat = "built" if fab.get("fully_built", False) else f"pending {pending}"
                        ord_txt = self._font_s.render(
                            f"order: {fab.get('blueprint_id', 'duty_station_gestalt')}  status: {stat}",
                            True,
                            _DIM,
                        )
                        surface.blit(ord_txt, (220, 56))
            else:
                scope_lbl = self._font_s.render("edit scope: archetype", True, _DIM)
                surface.blit(scope_lbl, (220, 42))
        if self._move_payload is not None:
            move_note = self._font_s.render("MOVE armed: click destination", True, _HILITE_MARKER)
            surface.blit(move_note, (260, 42))

        if str(self.state.get("room_view", "plan")) == "tile_editor":
            self._grid_rect = pygame.Rect(0, 0, 0, 0)
            tile_w_cells = max(1, int(self.state.get("tile_editor_footprint_x", 2)))
            tile_d_cells = max(1, int(self.state.get("tile_editor_footprint_y", 2)))
            grid_margin = 12
            grid_top = toolbar_h + 8
            grid_width = width - grid_margin * 2
            grid_height = height - grid_top - 12
            cell_size = max(16, min((grid_width // tile_w_cells), (grid_height // tile_d_cells)))
            total_width = cell_size * tile_w_cells
            total_height = cell_size * tile_d_cells
            left = grid_margin + max(0, (grid_width - total_width) // 2)
            top = grid_top + max(0, (grid_height - total_height) // 2)
            self._grid_rect = pygame.Rect(left, top, total_width, total_height)
            for gy in range(tile_d_cells):
                for gx in range(tile_w_cells):
                    rect = pygame.Rect(left + gx * cell_size, top + gy * cell_size, cell_size, cell_size)
                    fill = _GRID if (gx + gy) % 2 == 0 else _GRID_ALT
                    pygame.draw.rect(surface, fill, rect)
                    pygame.draw.rect(surface, _GRID_LINE, rect, 1)
                    self._projection_cells.append(ProjectionCell(gx, gy, 0, rect))

            selected_mesh = str(self.state.get("tile_editor_selected_mesh", ""))
            scale_x = float(total_width) / max(1e-6, tile_w_cells * _CELL_SIZE_M)
            scale_y = float(total_height) / max(1e-6, tile_d_cells * _CELL_SIZE_M)
            for item in active_items:
                half = 0.5 * item.bbox_size[:2]
                min_xy = item.pos_xy - half
                max_xy = item.pos_xy + half
                rx = int(left + min_xy[0] * scale_x)
                ry = int(top + min_xy[1] * scale_y)
                rw = max(2, int((max_xy[0] - min_xy[0]) * scale_x))
                rh = max(2, int((max_xy[1] - min_xy[1]) * scale_y))
                col = _HILITE_MARKER if item.mesh_id == selected_mesh else (160, 170, 190)
                pygame.draw.rect(surface, col, pygame.Rect(rx, ry, rw, rh), 2, border_radius=3)
                tag = self._font_s.render(f"{item.label}  rot {item.yaw_step * 30:03d}", True, col)
                surface.blit(tag, (rx + 3, max(top + 2, ry - 14)))

            footer = self._font_s.render(
                f"tile footprint: {tile_w_cells}x{tile_d_cells}  levels: {int(self.state.get('tile_editor_levels', 1))}",
                True, _DIM,
            )
            surface.blit(footer, (12, height - 18))
            hint = self._font_s.render("O: copy selected fabrication order", True, _DIM)
            surface.blit(hint, (320, height - 18))
            return surface

        grid_margin = 12
        grid_top = toolbar_h + 8
        grid_width = width - grid_margin * 2
        grid_height = height - grid_top - 12
        view_name = str(self.state.get("room_view", "plan"))
        room_width = int(self.state.get("room_width_cells", 8))
        room_depth = int(self.state.get("room_depth_cells", 8))
        selected_id = str(self.state.get("room_selected_instance", ""))

        if view_name == "plan":
            cell_size = max(16, min((grid_width // room_width), (grid_height // room_depth)))
            total_width = cell_size * room_width
            total_height = cell_size * room_depth
            left = grid_margin + max(0, (grid_width - total_width) // 2)
            top = grid_top + max(0, (grid_height - total_height) // 2)
            self._grid_rect = pygame.Rect(left, top, total_width, total_height)
            for grid_y in range(room_depth):
                for grid_x in range(room_width):
                    rect = pygame.Rect(left + grid_x * cell_size, top + grid_y * cell_size, cell_size, cell_size)
                    fill = _GRID if (grid_x + grid_y) % 2 == 0 else _GRID_ALT
                    pygame.draw.rect(surface, fill, rect)
                    pygame.draw.rect(surface, _GRID_LINE, rect, 1)
                    self._projection_cells.append(ProjectionCell(grid_x, grid_y, int(self.state.get("room_level", 0)), rect))
            for instance in self.instances:
                preset = self.presets.get(instance.preset_id)
                if preset is None:
                    continue
                level = int(self.state.get("room_level", 0))
                if not (instance.level <= level < instance.level + preset.level_span):
                    continue
                for dx in range(preset.footprint_xy[0]):
                    for dy in range(preset.footprint_xy[1]):
                        rect = pygame.Rect(
                            left + (instance.grid_x + dx) * cell_size,
                            top + (instance.grid_y + dy) * cell_size,
                            cell_size,
                            cell_size,
                        )
                        self._draw_plan_cell(surface, rect, preset, selected=instance.instance_id == selected_id)
            if int(self.state.get("room_level", 0)) == 0:
                sp = self._station_preset()
                if sp is not None:
                    sx = int(self.state.get("room_station_x", 0))
                    sy = int(self.state.get("room_station_y", 0))
                    lv = int(self.state.get("room_level", 0))
                    if 0 <= lv < int(sp.level_span):
                        for dx in range(sp.footprint_xy[0]):
                            for dy in range(sp.footprint_xy[1]):
                                rect = pygame.Rect(
                                    left + (sx + dx) * cell_size,
                                    top + (sy + dy) * cell_size,
                                    cell_size,
                                    cell_size,
                                )
                                self._draw_plan_cell(surface, rect, sp, selected=True)
                        brect = pygame.Rect(
                            left + sx * cell_size,
                            top + sy * cell_size,
                            cell_size * sp.footprint_xy[0],
                            cell_size * sp.footprint_xy[1],
                        )
                        pygame.draw.rect(surface, _HILITE_MARKER, brect, 2, border_radius=3)
                else:
                    sx = int(self.state.get("room_station_x", 0))
                    sy = int(self.state.get("room_station_y", 0))
                    rect = pygame.Rect(left + sx * cell_size, top + sy * cell_size, cell_size, cell_size)
                    pygame.draw.rect(surface, _HILITE_MARKER, rect.inflate(-cell_size // 3, -cell_size // 3), 2, border_radius=2)
                    pygame.draw.line(surface, _HILITE_MARKER, rect.bottomleft, rect.bottomright, 3)
        else:
            columns = room_width if view_name == "front" else room_depth
            levels = self._projection_levels()
            self._active_levels = (levels[-1], levels[0])
            cell_size = max(16, min((grid_width // columns), (grid_height // len(levels))))
            total_width = cell_size * columns
            total_height = cell_size * len(levels)
            left = grid_margin + max(0, (grid_width - total_width) // 2)
            top = grid_top + max(0, (grid_height - total_height) // 2)
            self._grid_rect = pygame.Rect(left, top, total_width, total_height)
            for row_index, level in enumerate(levels):
                for column in range(columns):
                    rect = pygame.Rect(left + column * cell_size, top + row_index * cell_size, cell_size, cell_size)
                    fill = _GRID if (column + row_index) % 2 == 0 else _GRID_ALT
                    pygame.draw.rect(surface, fill, rect)
                    pygame.draw.rect(surface, _GRID_LINE, rect, 1)
                    grid_x = column if view_name == "front" else int(self.state.get("room_station_x", 0))
                    grid_y = int(self.state.get("room_station_y", 0)) if view_name == "front" else column
                    self._projection_cells.append(ProjectionCell(grid_x, grid_y, level, rect))
                level_lbl = self._font_s.render(str(level), True, _DIM)
                surface.blit(level_lbl, (left - level_lbl.get_width() - 6, top + row_index * cell_size + 2))
            for instance in self.instances:
                preset = self.presets.get(instance.preset_id)
                if preset is None:
                    continue
                for row_index, level in enumerate(levels):
                    if not (instance.level <= level < instance.level + preset.level_span):
                        continue
                    if view_name == "front":
                        for dx in range(preset.footprint_xy[0]):
                            rect = pygame.Rect(left + (instance.grid_x + dx) * cell_size,
                                               top + row_index * cell_size,
                                               cell_size,
                                               cell_size)
                            self._draw_projection_cell(surface, rect, preset, selected=instance.instance_id == selected_id)
                    else:
                        for dy in range(preset.footprint_xy[1]):
                            rect = pygame.Rect(left + (instance.grid_y + dy) * cell_size,
                                               top + row_index * cell_size,
                                               cell_size,
                                               cell_size)
                            self._draw_projection_cell(surface, rect, preset, selected=instance.instance_id == selected_id)
        bounds = self.occupied_level_bounds()
        footer = self._font_s.render(f"occupied levels: {bounds[0]} .. {bounds[1]}", True, _DIM)
        surface.blit(footer, (12, height - 18))
        return surface

    def handle_event(self, event, x_off: int = 0, y_off: int = 0) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            lx = event.pos[0] - x_off
            ly = event.pos[1] - y_off
            for name, rect in self._button_rects.items():
                if rect.collidepoint(lx, ly):
                    self.set_view(name.split(":", 1)[1])
                    return True
            if self._minus_plus_rects.get("level:-") and self._minus_plus_rects["level:-"].collidepoint(lx, ly):
                self.adjust_level(-1)
                return True
            if self._minus_plus_rects.get("level:+") and self._minus_plus_rects["level:+"].collidepoint(lx, ly):
                self.adjust_level(1)
                return True
            if self._minus_plus_rects.get("dimx:-") and self._minus_plus_rects["dimx:-"].collidepoint(lx, ly):
                self.resize_room("x", -1)
                return True
            if self._minus_plus_rects.get("dimx:+") and self._minus_plus_rects["dimx:+"].collidepoint(lx, ly):
                self.resize_room("x", 1)
                return True
            if self._minus_plus_rects.get("dimy:-") and self._minus_plus_rects["dimy:-"].collidepoint(lx, ly):
                self.resize_room("y", -1)
                return True
            if self._minus_plus_rects.get("dimy:+") and self._minus_plus_rects["dimy:+"].collidepoint(lx, ly):
                self.resize_room("y", 1)
                return True
            for cell in self._projection_cells:
                if cell.rect.collidepoint(lx, ly):
                    self.click_grid(cell.grid_x, cell.grid_y, cell.level)
                    return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._move_payload = None
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_o:
            txt = self._selected_fabrication_order_text()
            if txt:
                copied = False
                try:
                    if hasattr(pygame, "scrap"):
                        if not pygame.scrap.get_init():
                            pygame.scrap.init()
                        pygame.scrap.put(pygame.SCRAP_TEXT, txt.encode("utf-8"))
                        copied = True
                except Exception:
                    copied = False
                self.state["room_last_fabrication_order"] = txt
                self.state["room_last_fabrication_order_copied"] = copied
                return True
        return False

    def draw_world(self, mvp: np.ndarray, mv: np.ndarray, light_v: np.ndarray, prog: Optional[int]):
        if not _HAS_GL or prog is None:
            return
        if self._gl_dirty:
            self._rebuild_gl()
        if self._vao is None or not self._vert_count:
            return
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(prog)
        loc = glGetUniformLocation(prog, b"uMVP")
        if loc >= 0:
            glUniformMatrix4fv(loc, 1, GL_TRUE, mvp.astype(np.float32))
        loc = glGetUniformLocation(prog, b"uMV")
        if loc >= 0:
            glUniformMatrix4fv(loc, 1, GL_TRUE, mv.astype(np.float32))
        glUniform3f(glGetUniformLocation(prog, b"uLightV"), *light_v)
        glUniform4f(glGetUniformLocation(prog, b"uColor"), 0.20, 0.26, 0.34, 0.92)
        glUniform3f(glGetUniformLocation(prog, b"uInnerColor"), 0.20, 0.26, 0.34)
        glUniform1f(glGetUniformLocation(prog, b"uAmbient"), 0.22)
        glUniform1f(glGetUniformLocation(prog, b"uSpecStrength"), 0.18)
        glUniform1f(glGetUniformLocation(prog, b"uShininess"), 48.0)
        glUniform1f(glGetUniformLocation(prog, b"uGrain"), 0.04)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, self._vert_count)
        glBindVertexArray(0)
        glUseProgram(0)

    def _rebuild_gl(self):
        self._gl_dirty = False
        if not _HAS_GL:
            return
        vertices = np.asarray(self._build_world_vertices(), dtype=np.float32).reshape(-1, 6)
        self._vert_count = len(vertices)
        if self._vao is not None:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
            self._vao = None
            self._vbo = None
        if not self._vert_count:
            return
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.tobytes(), GL_STATIC_DRAW)
        stride = 24
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glBindVertexArray(0)

    def _build_world_vertices(self) -> List[Tuple[float, float, float, float, float, float]]:
        vertices: List[Tuple[float, float, float, float, float, float]] = []
        anchor_x = int(self.state.get("room_station_x", 0))
        anchor_y = int(self.state.get("room_station_y", 0))
        station_nodes: List[Tuple[RoomTileInstance, RoomTilePreset, float, float, float, bool]] = []

        # Emit the station anchor's gestalt tile item at the workspace origin (0,0,0).
        sp = self._station_preset()
        if sp is not None:
            anchor_key = f"station_anchor:{sp.preset_id}"
            anchor_items = self.tile_mesh_items_by_tile.get(anchor_key)
            if not anchor_items:
                anchor_items = []
                self._seed_duty_station_gestalt_item(
                    anchor_items,
                    int(sp.footprint_xy[0]),
                    int(sp.footprint_xy[1]),
                    int(sp.level_span),
                    sp,
                    "archetype",
                )
                self.tile_mesh_items_by_tile[anchor_key] = anchor_items
            # The station anchor is always present at startup — it is built.
            anchor_is_built = True
            for item in anchor_items:
                sx = float(item.bbox_size[0])
                sy = float(item.bbox_size[1])
                sz = float(item.bbox_size[2])
                if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
                    continue
                cx = float(item.pos_xy[0])
                cy = float(item.pos_xy[1])
                if str(getattr(item, "render_style", "box")) == "duty_station_gestalt":
                    g_verts = self._build_gestalt_station_vertices(
                        cx, cy, 0.0, sx, sy, sz, is_fully_built=anchor_is_built,
                    )
                    vertices.extend(g_verts)
                else:
                    half_x = 0.5 * sx
                    half_y = 0.5 * sy
                    z_min = float(item.z_min)
                    z_max = z_min + sz
                    vertices.extend(self._box(
                        (cx - half_x, cy - half_y, z_min),
                        (cx + half_x, cy + half_y, z_max),
                    ))

        for instance in self.instances:
            preset = self.presets.get(instance.preset_id)
            if preset is None:
                continue
            base_x = (instance.grid_x - anchor_x) * _CELL_SIZE_M
            base_y = (instance.grid_y - anchor_y) * _CELL_SIZE_M
            z0 = instance.level * _LEVEL_HEIGHT_M

            # Prefer explicit tile mesh contents (placed/archetype) when available.
            # Fallback to procedural preset proxy geometry only if no tile items exist.
            item_vertices = self._build_item_vertices(instance, preset, base_x, base_y, z0)
            if item_vertices:
                vertices.extend(item_vertices)
                if preset.category == "duty_stations":
                    cx = base_x + 0.5 * preset.footprint_xy[0] * _CELL_SIZE_M
                    cy = base_y + 0.5 * preset.footprint_xy[1] * _CELL_SIZE_M
                    fabricated = bool(self._fabrication_state(instance, preset).get("fully_built", False))
                    station_nodes.append((instance, preset, cx, cy, z0, fabricated))
                continue

            instance_vertices: List[Tuple[float, float, float, float, float, float]] = []
            if preset.mesh_type == "marker_box":
                instance_vertices.extend(self._box(
                    (base_x + 0.10, base_y + 0.10, z0),
                    (base_x + preset.footprint_xy[0] * _CELL_SIZE_M - 0.10,
                     base_y + preset.footprint_xy[1] * _CELL_SIZE_M - 0.10,
                     z0 + max(0.35, 0.45 * preset.level_span)),
                ))
            elif preset.mesh_type == "quarter_cylinder":
                instance_vertices.extend(self._quarter_cylinder(base_x, base_y, z0, preset))
            elif preset.mesh_type == "quarter_hemisphere":
                instance_vertices.extend(self._quarter_hemisphere(base_x, base_y, z0, preset))
            else:
                width, depth = preset.footprint_xy
                for dx in range(width):
                    for dy in range(depth):
                        cell_x = base_x + dx * _CELL_SIZE_M
                        cell_y = base_y + dy * _CELL_SIZE_M
                        instance_vertices.extend(self._flat_surfaces(cell_x, cell_y, z0, preset))

            step = int(getattr(instance, "rotation", 0)) % _ROT_STEPS
            if step:
                instance_vertices = self._rotate_instance_vertices(
                    instance_vertices,
                    base_x + 0.5 * preset.footprint_xy[0] * _CELL_SIZE_M,
                    base_y + 0.5 * preset.footprint_xy[1] * _CELL_SIZE_M,
                    step,
                )
            vertices.extend(instance_vertices)

        vertices.extend(self._build_station_network_vertices(station_nodes))
        return vertices

    def _build_item_vertices(
        self,
        instance: RoomTileInstance,
        preset: RoomTilePreset,
        base_x: float,
        base_y: float,
        z0: float,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        items = self._items_for_instance(instance, preset)
        if not items:
            return []

        verts: List[Tuple[float, float, float, float, float, float]] = []
        fab = self._fabrication_state(instance, preset)
        is_fully_built = bool(fab.get("fully_built", False))
        for item in items:
            sx = float(item.bbox_size[0])
            sy = float(item.bbox_size[1])
            sz = float(item.bbox_size[2])
            if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
                continue

            cx = base_x + float(item.pos_xy[0])
            cy = base_y + float(item.pos_xy[1])
            half_x = 0.5 * sx
            half_y = 0.5 * sy
            z_min = z0 + float(item.z_min)
            z_max = z_min + sz

            if str(getattr(item, "render_style", "box")) == "duty_station_gestalt":
                g_verts = self._build_gestalt_station_vertices(
                    cx,
                    cy,
                    z0,
                    sx,
                    sy,
                    sz,
                    is_fully_built=is_fully_built,
                )
                step = int(getattr(item, "yaw_step", 0)) % _ROT_STEPS
                if step:
                    g_verts = self._rotate_instance_vertices(g_verts, cx, cy, step)
                verts.extend(g_verts)
                continue

            box_verts = self._box(
                (cx - half_x, cy - half_y, z_min),
                (cx + half_x, cy + half_y, z_max),
            )
            step = int(getattr(item, "yaw_step", 0)) % _ROT_STEPS
            if step:
                box_verts = self._rotate_instance_vertices(box_verts, cx, cy, step)
            verts.extend(box_verts)

        inst_step = int(getattr(instance, "rotation", 0)) % _ROT_STEPS
        if inst_step:
            verts = self._rotate_instance_vertices(
                verts,
                base_x + 0.5 * preset.footprint_xy[0] * _CELL_SIZE_M,
                base_y + 0.5 * preset.footprint_xy[1] * _CELL_SIZE_M,
                inst_step,
            )
        return verts

    def _build_station_network_vertices(
        self,
        station_nodes: List[Tuple[RoomTileInstance, RoomTilePreset, float, float, float, bool]],
    ) -> List[Tuple[float, float, float, float, float, float]]:
        if len(station_nodes) < 2:
            return []
        out: List[Tuple[float, float, float, float, float, float]] = []
        by_row: Dict[Tuple[int, int], List[Tuple[float, float, float, bool]]] = {}
        for inst, _preset, cx, cy, z0, built in station_nodes:
            key = (int(inst.grid_y), int(inst.level))
            by_row.setdefault(key, []).append((cx, cy, z0, built))

        cable_h = 0.04
        cable_w = 0.06
        for key in by_row:
            row = sorted(by_row[key], key=lambda t: t[0])
            for i in range(len(row) - 1):
                ax, ay, az, ab = row[i]
                bx, by, bz, bb = row[i + 1]
                if not (ab and bb):
                    continue
                z0 = min(az, bz) + 0.34
                x0 = min(ax, bx)
                x1 = max(ax, bx)
                y0 = 0.5 * (ay + by) - 0.5 * cable_w
                y1 = 0.5 * (ay + by) + 0.5 * cable_w
                out.extend(self._box((x0, y0, z0), (x1, y1, z0 + cable_h)))
        return out

    def _build_gestalt_station_vertices(
        self,
        cx: float,
        cy: float,
        z0: float,
        sx: float,
        sy: float,
        sz: float,
        *,
        is_fully_built: bool,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        # Not fabricated yet: show the reservation block only.
        if not is_fully_built:
            return self._box(
                (cx - 0.5 * sx, cy - 0.5 * sy, z0),
                (cx + 0.5 * sx, cy + 0.5 * sy, z0 + sz),
            )

        out: List[Tuple[float, float, float, float, float, float]] = []

        cons_w = max(0.45, min(sx * 0.88, 1.40))
        cons_d = max(0.35, min(sy * 0.56, 0.64))
        cons_h = max(0.45, min(sz * 0.56, 0.92))
        out.extend(self._wedge_console(
            cx,
            cy,
            z0,
            cons_w,
            cons_d,
            cons_h,
            tilt_deg=14.0,
        ))

        scr_w = max(0.30, min(cons_w * 0.90, 1.10))
        scr_t = 0.06
        scr_h = max(0.24, min(sz * 0.40, 0.74))
        scr_y = cy + 0.5 * cons_d - 0.08
        scr_z0 = z0 + max(0.22, cons_h - 0.25)
        out.extend(self._box(
            (cx - 0.5 * scr_w, scr_y - 0.5 * scr_t, scr_z0),
            (cx + 0.5 * scr_w, scr_y + 0.5 * scr_t, scr_z0 + scr_h),
        ))

        wing_w = max(0.08, min(sx * 0.10, 0.16))
        wing_d = max(0.10, min(cons_d * 0.95, 0.56))
        wing_h = max(0.22, min(sz * 0.36, 0.58))
        wing_x = 0.5 * (cons_w + wing_w)
        out.extend(self._box(
            (cx - wing_x - 0.5 * wing_w, cy - 0.5 * wing_d, z0),
            (cx - wing_x + 0.5 * wing_w, cy + 0.5 * wing_d, z0 + wing_h),
        ))
        out.extend(self._box(
            (cx + wing_x - 0.5 * wing_w, cy - 0.5 * wing_d, z0),
            (cx + wing_x + 0.5 * wing_w, cy + 0.5 * wing_d, z0 + wing_h),
        ))

        # Under-wing plugs: short connector posts beneath each wing.
        plug_w = 0.06
        plug_d = 0.06
        plug_h = 0.14
        plug_y = cy - 0.20
        out.extend(self._box(
            (cx - wing_x - 0.5 * plug_w, plug_y - 0.5 * plug_d, z0),
            (cx - wing_x + 0.5 * plug_w, plug_y + 0.5 * plug_d, z0 + plug_h),
        ))
        out.extend(self._box(
            (cx + wing_x - 0.5 * plug_w, plug_y - 0.5 * plug_d, z0),
            (cx + wing_x + 0.5 * plug_w, plug_y + 0.5 * plug_d, z0 + plug_h),
        ))

        # Left side: hollow-pipe placeholders (outer shell + inner void cutout).
        pipe_len = max(0.26, min(sy * 0.30, 0.42))
        pipe_od = 0.08
        pipe_id = 0.04
        for k in (-0.10, 0.10):
            px = cx - wing_x - 0.12
            py = cy + k
            pz = z0 + 0.16
            out.extend(self._box(
                (px - 0.5 * pipe_len, py - 0.5 * pipe_od, pz - 0.5 * pipe_od),
                (px + 0.5 * pipe_len, py + 0.5 * pipe_od, pz + 0.5 * pipe_od),
            ))
            out.extend(self._box(
                (px - 0.5 * pipe_len * 0.96, py - 0.5 * pipe_id, pz - 0.5 * pipe_id),
                (px + 0.5 * pipe_len * 0.96, py + 0.5 * pipe_id, pz + 0.5 * pipe_id),
            ))

        # Right side: solid rods.
        rod_len = max(0.28, min(sy * 0.34, 0.46))
        rod_w = 0.05
        for k in (-0.10, 0.10):
            rx = cx + wing_x + 0.12
            ry = cy + k
            rz = z0 + 0.16
            out.extend(self._box(
                (rx - 0.5 * rod_len, ry - 0.5 * rod_w, rz - 0.5 * rod_w),
                (rx + 0.5 * rod_len, ry + 0.5 * rod_w, rz + 0.5 * rod_w),
            ))

        return out

    def _items_for_instance(
        self,
        instance: RoomTileInstance,
        preset: RoomTilePreset,
    ) -> List[TileMeshItem]:
        placed_key = f"inst:{instance.instance_id}"
        placed_items = self.tile_mesh_items_by_instance.get(placed_key)
        if placed_items:
            return placed_items

        arche_items = self.tile_mesh_items_by_tile.get(str(preset.preset_id), [])
        if arche_items:
            return arche_items

        # Seed known duty-station presets when no explicit mesh items exist yet.
        seeded: List[TileMeshItem] = []
        fx = max(1, int(preset.footprint_xy[0]))
        fy = max(1, int(preset.footprint_xy[1]))
        fz = max(1, int(preset.level_span))
        if preset.category == "duty_stations":
            self._seed_duty_station_gestalt_item(seeded, fx, fy, fz, preset, scope="placed")
        elif self._is_room_control_station_preset(preset):
            self._seed_room_control_station_components(seeded, fx, fy, fz, "placed")
        elif self._is_camera_designer_station_preset(preset):
            self._seed_camera_designer_station_components(seeded, fx, fy, fz, "placed")

        if seeded:
            self.tile_mesh_items_by_instance[placed_key] = seeded
            return seeded
        return []

    def _seed_duty_station_gestalt_item(
        self,
        active_items: List[TileMeshItem],
        fx: int,
        fy: int,
        fz: int,
        preset: RoomTilePreset,
        scope: str,
    ):
        tile_w = float(fx) * float(_CELL_SIZE_M)
        tile_d = float(fy) * float(_CELL_SIZE_M)
        tile_h = float(fz) * float(_LEVEL_HEIGHT_M)
        label_prefix = "Placed " if scope == "placed" else ""
        active_items.append(TileMeshItem(
            mesh_id="mesh_001",
            label=f"{label_prefix}{preset.label} (Gestalt)",
            bbox_size=np.array([
                max(0.70, tile_w * 0.96),
                max(0.70, tile_d * 0.96),
                max(0.70, tile_h * 0.96),
            ], np.float64),
            pos_xy=np.array([tile_w * 0.5, tile_d * 0.5], np.float64),
            z_min=0.0,
            yaw_step=0,
            render_style="duty_station_gestalt",
            blueprint_id=str(getattr(preset, "blueprint_id", "") or "duty_station_gestalt"),
            programmatic_blueprint_id=str(getattr(preset, "programmatic_blueprint_id", "")),
            requires_custom_confirm=bool(getattr(preset, "requires_custom_confirm", False)),
        ))

    def _fabrication_state(self, instance: RoomTileInstance, preset: RoomTilePreset) -> Dict[str, Any]:
        prefab_available = bool(getattr(preset, "prefab_available", False))
        adjacent_fabricator = self._has_adjacent_fabricator(instance)
        has_prog = bool(str(getattr(preset, "programmatic_blueprint_id", "")))
        requires_custom = bool(getattr(preset, "requires_custom_confirm", False)) or has_prog
        custom_confirmed = bool(self.state.get(f"fabricator_confirm::{instance.instance_id}", False))

        fixed_parts = ["console_body", "screen_panel", "left_wing", "right_wing", "floor_tile"]
        if prefab_available:
            fixed_done = list(fixed_parts)
            fixed_pending = []
        elif adjacent_fabricator:
            fixed_done = list(fixed_parts)
            fixed_pending: List[str] = []
        else:
            fixed_done = []
            fixed_pending = list(fixed_parts)

        custom_pending: List[str] = []
        custom_done: List[str] = []
        if requires_custom:
            custom_name = str(getattr(preset, "programmatic_blueprint_id", "") or "duty_station_parametric_connector")
            if prefab_available or custom_confirmed:
                custom_done.append(custom_name)
            else:
                custom_pending.append(custom_name)

        fully_built = (len(fixed_pending) == 0) and (len(custom_pending) == 0)
        self.fabrication_orders_by_instance[instance.instance_id] = {
            "instance_id": instance.instance_id,
            "preset_id": preset.preset_id,
            "blueprint_id": str(getattr(preset, "blueprint_id", "") or "duty_station_gestalt"),
            "programmatic_blueprint_id": str(getattr(preset, "programmatic_blueprint_id", "")),
            "adjacent_fabricator": adjacent_fabricator,
            "prefab_available": prefab_available,
            "fixed_done": fixed_done,
            "fixed_pending": fixed_pending,
            "custom_done": custom_done,
            "custom_pending": custom_pending,
            "fully_built": fully_built,
        }
        return self.fabrication_orders_by_instance[instance.instance_id]

    def _has_adjacent_fabricator(self, instance: RoomTileInstance) -> bool:
        p = self.presets.get(instance.preset_id)
        if p is None:
            return False
        x0 = int(instance.grid_x)
        y0 = int(instance.grid_y)
        x1 = x0 + int(p.footprint_xy[0]) - 1
        y1 = y0 + int(p.footprint_xy[1]) - 1

        for other in self.instances:
            if other.instance_id == instance.instance_id:
                continue
            op = self.presets.get(other.preset_id)
            if op is None:
                continue
            pid = str(getattr(op, "preset_id", "")).lower()
            kind = str(getattr(op, "kind", "")).lower()
            if ("fabricator" not in pid) and ("fabricator" not in kind):
                continue

            ox0 = int(other.grid_x)
            oy0 = int(other.grid_y)
            ox1 = ox0 + int(op.footprint_xy[0]) - 1
            oy1 = oy0 + int(op.footprint_xy[1]) - 1

            touch_x = (x1 + 1 == ox0) or (ox1 + 1 == x0)
            overlap_y = not ((y1 < oy0) or (oy1 < y0))
            touch_y = (y1 + 1 == oy0) or (oy1 + 1 == y0)
            overlap_x = not ((x1 < ox0) or (ox1 < x0))
            if (touch_x and overlap_y) or (touch_y and overlap_x):
                return True
        return False

    def _selected_fabrication_order_text(self) -> str:
        inst = self._selected_instance()
        if inst is None:
            return ""
        preset = self.presets.get(inst.preset_id)
        if preset is None or preset.category != "duty_stations":
            return ""
        order = self._fabrication_state(inst, preset)
        lines = [
            f"Fabrication Order: {order.get('instance_id', inst.instance_id)}",
            f"Preset: {order.get('preset_id', preset.preset_id)}",
            f"Blueprint: {order.get('blueprint_id', 'duty_station_gestalt')}",
            f"Programmatic: {order.get('programmatic_blueprint_id', '') or '(none)'}",
            f"Prefabricated: {bool(order.get('prefab_available', False))}",
            f"Adjacent Fabricator: {bool(order.get('adjacent_fabricator', False))}",
        ]
        fixed_pending = list(order.get("fixed_pending", []))
        custom_pending = list(order.get("custom_pending", []))
        lines.append("Fixed Pending: " + (", ".join(fixed_pending) if fixed_pending else "(none)"))
        lines.append("Custom Pending: " + (", ".join(custom_pending) if custom_pending else "(none)"))
        lines.append("Status: " + ("built" if bool(order.get("fully_built", False)) else "pending"))
        return "\n".join(lines)

    @staticmethod
    def _rotate_instance_vertices(
        verts: List[Tuple[float, float, float, float, float, float]],
        cx: float,
        cy: float,
        step: int,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        if not verts:
            return verts
        ang = (2.0 * math.pi / _ROT_STEPS) * float(step)
        c = math.cos(ang)
        s = math.sin(ang)
        out: List[Tuple[float, float, float, float, float, float]] = []
        for x, y, z, nx, ny, nz in verts:
            rx = cx + (x - cx) * c - (y - cy) * s
            ry = cy + (x - cx) * s + (y - cy) * c
            rnx = nx * c - ny * s
            rny = nx * s + ny * c
            out.append((rx, ry, z, rnx, rny, nz))
        return out

    def _flat_surfaces(self, base_x: float, base_y: float, z0: float, preset: RoomTilePreset):
        z1 = z0 + preset.level_span * _LEVEL_HEIGHT_M
        vertices: List[Tuple[float, float, float, float, float, float]] = []
        if "floor" in preset.surfaces:
            vertices.extend(self._quad(
                (base_x, base_y, z0),
                (base_x + _CELL_SIZE_M, base_y, z0),
                (base_x + _CELL_SIZE_M, base_y + _CELL_SIZE_M, z0),
                (base_x, base_y + _CELL_SIZE_M, z0),
                (0.0, 0.0, 1.0),
            ))
        if "ceiling" in preset.surfaces:
            vertices.extend(self._quad(
                (base_x, base_y + _CELL_SIZE_M, z1),
                (base_x + _CELL_SIZE_M, base_y + _CELL_SIZE_M, z1),
                (base_x + _CELL_SIZE_M, base_y, z1),
                (base_x, base_y, z1),
                (0.0, 0.0, -1.0),
            ))
        if "south" in preset.surfaces:
            vertices.extend(self._quad(
                (base_x, base_y, z0),
                (base_x + _CELL_SIZE_M, base_y, z0),
                (base_x + _CELL_SIZE_M, base_y, z1),
                (base_x, base_y, z1),
                (0.0, 1.0, 0.0),
            ))
        if "north" in preset.surfaces:
            vertices.extend(self._quad(
                (base_x + _CELL_SIZE_M, base_y + _CELL_SIZE_M, z0),
                (base_x, base_y + _CELL_SIZE_M, z0),
                (base_x, base_y + _CELL_SIZE_M, z1),
                (base_x + _CELL_SIZE_M, base_y + _CELL_SIZE_M, z1),
                (0.0, -1.0, 0.0),
            ))
        if "west" in preset.surfaces:
            vertices.extend(self._quad(
                (base_x, base_y + _CELL_SIZE_M, z0),
                (base_x, base_y, z0),
                (base_x, base_y, z1),
                (base_x, base_y + _CELL_SIZE_M, z1),
                (1.0, 0.0, 0.0),
            ))
        if "east" in preset.surfaces:
            vertices.extend(self._quad(
                (base_x + _CELL_SIZE_M, base_y, z0),
                (base_x + _CELL_SIZE_M, base_y + _CELL_SIZE_M, z0),
                (base_x + _CELL_SIZE_M, base_y + _CELL_SIZE_M, z1),
                (base_x + _CELL_SIZE_M, base_y, z1),
                (-1.0, 0.0, 0.0),
            ))
        return vertices

    def _quarter_cylinder(self, base_x: float, base_y: float, z0: float, preset: RoomTilePreset):
        radius = max(0.25, preset.radius_cells * _CELL_SIZE_M)
        height = preset.level_span * _LEVEL_HEIGHT_M
        corner_x = base_x + preset.footprint_xy[0] * _CELL_SIZE_M
        corner_y = base_y + preset.footprint_xy[1] * _CELL_SIZE_M
        segments = 12
        vertices: List[Tuple[float, float, float, float, float, float]] = []
        for index in range(segments):
            theta0 = (index / segments) * (math.pi / 2.0)
            theta1 = ((index + 1) / segments) * (math.pi / 2.0)
            p00 = (corner_x - radius * math.cos(theta0), corner_y - radius * math.sin(theta0), z0)
            p01 = (corner_x - radius * math.cos(theta0), corner_y - radius * math.sin(theta0), z0 + height)
            p10 = (corner_x - radius * math.cos(theta1), corner_y - radius * math.sin(theta1), z0)
            p11 = (corner_x - radius * math.cos(theta1), corner_y - radius * math.sin(theta1), z0 + height)
            mid = 0.5 * (theta0 + theta1)
            normal = (-math.cos(mid), -math.sin(mid), 0.0)
            vertices.extend(self._quad(p10, p00, p01, p11, normal))
        return vertices

    def _quarter_hemisphere(self, base_x: float, base_y: float, z0: float, preset: RoomTilePreset):
        radius = max(0.25, preset.radius_cells * _CELL_SIZE_M)
        corner_x = base_x + preset.footprint_xy[0] * _CELL_SIZE_M
        corner_y = base_y + preset.footprint_xy[1] * _CELL_SIZE_M
        center_z = z0 + preset.level_span * _LEVEL_HEIGHT_M
        theta_segments = 10
        phi_segments = 6
        vertices: List[Tuple[float, float, float, float, float, float]] = []
        for i_theta in range(theta_segments):
            theta0 = (i_theta / theta_segments) * (math.pi / 2.0)
            theta1 = ((i_theta + 1) / theta_segments) * (math.pi / 2.0)
            for i_phi in range(phi_segments):
                phi0 = (i_phi / phi_segments) * (math.pi / 2.0)
                phi1 = ((i_phi + 1) / phi_segments) * (math.pi / 2.0)
                p00 = self._sphere_point(corner_x, corner_y, center_z, radius, theta0, phi0)
                p10 = self._sphere_point(corner_x, corner_y, center_z, radius, theta1, phi0)
                p11 = self._sphere_point(corner_x, corner_y, center_z, radius, theta1, phi1)
                p01 = self._sphere_point(corner_x, corner_y, center_z, radius, theta0, phi1)
                mid_theta = 0.5 * (theta0 + theta1)
                mid_phi = 0.5 * (phi0 + phi1)
                normal = (
                    -math.cos(mid_theta) * math.sin(mid_phi),
                    -math.sin(mid_theta) * math.sin(mid_phi),
                    -math.cos(mid_phi),
                )
                vertices.extend(self._quad(p10, p00, p01, p11, normal))
        return vertices

    @staticmethod
    def _sphere_point(corner_x: float, corner_y: float, center_z: float,
                      radius: float, theta: float, phi: float):
        return (
            corner_x - radius * math.cos(theta) * math.sin(phi),
            corner_y - radius * math.sin(theta) * math.sin(phi),
            center_z - radius * math.cos(phi),
        )

    @staticmethod
    def _wedge_console(
        cx: float,
        cy: float,
        z0: float,
        width: float,
        depth: float,
        height: float,
        *,
        tilt_deg: float,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        hw = 0.5 * float(width)
        hy = 0.5 * float(depth)
        tilt = math.radians(float(tilt_deg))
        z_front = float(z0 + height)
        z_back = float(z_front + depth * math.tan(tilt))

        b_fl = (cx - hw, cy - hy, z0)
        b_fr = (cx + hw, cy - hy, z0)
        b_bl = (cx - hw, cy + hy, z0)
        b_br = (cx + hw, cy + hy, z0)
        t_fl = (cx - hw, cy - hy, z_front)
        t_fr = (cx + hw, cy - hy, z_front)
        t_bl = (cx - hw, cy + hy, z_back)
        t_br = (cx + hw, cy + hy, z_back)

        top_n = (0.0, -math.sin(tilt), math.cos(tilt))
        out: List[Tuple[float, float, float, float, float, float]] = []
        out.extend(RoomTileWorkspace._quad(b_fr, b_fl, t_fl, t_fr, (0.0, -1.0, 0.0)))
        out.extend(RoomTileWorkspace._quad(b_bl, b_br, t_br, t_bl, (0.0, 1.0, 0.0)))
        out.extend(RoomTileWorkspace._quad(b_fl, b_bl, t_bl, t_fl, (-1.0, 0.0, 0.0)))
        out.extend(RoomTileWorkspace._quad(b_br, b_fr, t_fr, t_br, (1.0, 0.0, 0.0)))
        out.extend(RoomTileWorkspace._quad(b_bl, b_br, b_fr, b_fl, (0.0, 0.0, -1.0)))
        out.extend(RoomTileWorkspace._quad(t_fl, t_fr, t_br, t_bl, top_n))
        return out

    @staticmethod
    def _quad(v0, v1, v2, v3, normal):
        nx, ny, nz = normal
        return [
            (v0[0], v0[1], v0[2], nx, ny, nz),
            (v1[0], v1[1], v1[2], nx, ny, nz),
            (v2[0], v2[1], v2[2], nx, ny, nz),
            (v0[0], v0[1], v0[2], nx, ny, nz),
            (v2[0], v2[1], v2[2], nx, ny, nz),
            (v3[0], v3[1], v3[2], nx, ny, nz),
        ]

    @staticmethod
    def _box(min_corner, max_corner):
        x0, y0, z0 = min_corner
        x1, y1, z1 = max_corner
        faces = []
        # Bottom / top
        faces.extend(RoomTileWorkspace._quad((x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0), (0.0, 0.0, -1.0)))
        faces.extend(RoomTileWorkspace._quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0.0, 0.0, 1.0)))
        # Front / back (y-min / y-max)
        faces.extend(RoomTileWorkspace._quad((x1, y0, z0), (x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (0.0, -1.0, 0.0)))
        faces.extend(RoomTileWorkspace._quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1), (0.0, 1.0, 0.0)))
        # Left / right (x-min / x-max)
        faces.extend(RoomTileWorkspace._quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1), (-1.0, 0.0, 0.0)))
        faces.extend(RoomTileWorkspace._quad((x1, y1, z0), (x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (1.0, 0.0, 0.0)))
        return faces

    @staticmethod
    def _rgb255(rgb: Tuple[float, float, float]) -> Tuple[int, int, int]:
        return tuple(int(np.clip(channel * 255.0, 0, 255)) for channel in rgb)

    @staticmethod
    def _mix(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        return (
            int(a[0] * (1.0 - t) + b[0] * t),
            int(a[1] * (1.0 - t) + b[1] * t),
            int(a[2] * (1.0 - t) + b[2] * t),
        )


def load_room_tile_presets(library_dir: str) -> Dict[str, RoomTilePreset]:
    presets: Dict[str, RoomTilePreset] = {}
    if not _HAS_YAML or not os.path.isdir(library_dir):
        return presets
    for root, _, file_names in os.walk(library_dir):
        for file_name in sorted(file_names):
            if not file_name.lower().endswith((".yaml", ".yml")):
                continue
            path = os.path.join(root, file_name)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = _yaml.safe_load(handle) or {}
                preset = RoomTilePreset.from_dict(data)
                if preset.preset_id:
                    presets[preset.preset_id] = preset
            except Exception as exc:
                print(f"[room_tile_editor] preset load failed for {path}: {exc}")

    # Import duty-station placeables exported by fabricator blueprints.
    # This is the authoritative available-object set for gestalt station units.
    fab_roots = [
        os.path.join("configs", "duty_stations", "fabricator", "blueprints"),
        os.path.join("configs", "duty_stations", "fabricator", "saves"),
    ]
    for root in fab_roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            low = str(name).lower()
            if not (low.endswith(".blueprint.yaml") or low.endswith(".blueprint.yml") or low.endswith(".blueprint.json")):
                continue
            bp_path = os.path.join(root, name)
            try:
                if low.endswith(".json"):
                    with open(bp_path, "r", encoding="utf-8") as handle:
                        bp = json.load(handle) or {}
                else:
                    with open(bp_path, "r", encoding="utf-8") as handle:
                        bp = _yaml.safe_load(handle) or {}
                if not isinstance(bp, dict):
                    continue
                if str(bp.get("save_type", "")) != "fabrication_blueprint":
                    continue

                bp_id = str(bp.get("id", "")).strip()
                bp_label = str(bp.get("label", bp_id or "Fabricated Duty Station"))
                rel = bp.get("relationships", [])
                rel_text = str(rel).lower()
                is_duty_station_bp = ("duty_station" in bp_id.lower()) or ("duty station" in bp_label.lower()) or ("console_body" in rel_text)
                if not is_duty_station_bp:
                    continue

                placeable = bp.get("placeable_entry", {}) if isinstance(bp.get("placeable_entry", {}), dict) else {}
                footprint = placeable.get("footprint_xy", [2, 2])
                if not (isinstance(footprint, (list, tuple)) and len(footprint) >= 2):
                    footprint = [2, 2]
                levels = int(placeable.get("level_span", 2))
                if levels < 1:
                    levels = 1
                surfaces = placeable.get("surfaces", ["floor"])
                if not isinstance(surfaces, (list, tuple)):
                    surfaces = ["floor"]

                preset_data = {
                    "id": str(placeable.get("id", "") or f"fabricated::{bp_id}"),
                    "label": str(placeable.get("label", "") or bp_label),
                    "category": "duty_stations",
                    "kind": "fabricated_duty_station",
                    "mesh_type": "duty_station_blueprint",
                    "blueprint_id": bp_id,
                    "programmatic_blueprint_id": str(bp.get("programmatic_blueprint_id", "") or placeable.get("programmatic_blueprint_id", "")),
                    "requires_custom_confirm": bool(bp.get("requires_custom_confirm", placeable.get("requires_custom_confirm", False))),
                    "prefab_available": bool(bp.get("prefab_available", placeable.get("prefab_available", False))),
                    "surfaces": [str(s) for s in surfaces],
                    "footprint_xy": [max(1, int(footprint[0])), max(1, int(footprint[1]))],
                    "level_span": max(1, levels),
                    "color_rgb": [0.30, 0.60, 0.78],
                    "summary": str(placeable.get("summary", "") or "Fabricator-exported blueprint duty station"),
                }
                preset = RoomTilePreset.from_dict(preset_data)
                presets[preset.preset_id] = preset
            except Exception as exc:
                print(f"[room_tile_editor] fabricated blueprint preset load failed for {bp_path}: {exc}")

    # Ensure the actual deployed room-control station exists as a tile preset.
    # This is sourced from the existing station definition, not a duplicate asset.
    station_yaml = os.path.normpath(os.path.join(library_dir, "..", "station.yaml"))
    if os.path.isfile(station_yaml):
        try:
            with open(station_yaml, "r", encoding="utf-8") as handle:
                station_cfg = _yaml.safe_load(handle) or {}

            cons = station_cfg.get("console", {}) if isinstance(station_cfg, dict) else {}
            scr = station_cfg.get("screen", {}) if isinstance(station_cfg, dict) else {}
            wings = station_cfg.get("side_wings", {}) if isinstance(station_cfg, dict) else {}

            cons_w = float(cons.get("width", 1.40))
            cons_d = float(cons.get("depth", 0.62))
            cons_h = float(cons.get("height", 0.88))
            scr_h = float(scr.get("height", 0.72))
            wing_w = float(wings.get("width", 0.14)) if bool(wings.get("enabled", True)) else 0.0

            width_cells = max(1, int(math.ceil((cons_w + 2.0 * wing_w) / _CELL_SIZE_M)))
            depth_cells = max(2, int(math.ceil(cons_d / _CELL_SIZE_M)) + 1)
            levels = max(2, int(math.ceil(max(cons_h, scr_h) / _LEVEL_HEIGHT_M)))

            rc_data = {
                "id": "room_control_station",
                "label": "Room Control Duty Station",
                "category": "duty_stations",
                "kind": "room_control_station",
                "mesh_type": "duty_station_blueprint",
                "blueprint_id": "duty_station_gestalt",
                "programmatic_blueprint_id": "duty_station_parametric_connector",
                "requires_custom_confirm": False,
                "prefab_available": True,
                "surfaces": ["floor", "north"],
                "footprint_xy": [width_cells, depth_cells],
                "level_span": levels,
                "color_rgb": [0.30, 0.60, 0.78],
                "summary": "Deployed room-control duty station tile sourced from station.yaml",
            }
            rc_preset = RoomTilePreset.from_dict(rc_data)
            presets[rc_preset.preset_id] = rc_preset
        except Exception as exc:
            print(f"[room_tile_editor] room_control station preset synthesis failed for {station_yaml}: {exc}")
    return presets
