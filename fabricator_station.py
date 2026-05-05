"""fabricator_station.py
========================
GL renderer for the fabricator duty station workspace.

Layout (in interact mode, full window):
  ┌──────────────┬───────────────────────────────┬──────────────┐
  │  LEFT PANEL  │     3-D WORKSPACE VIEWPORT     │ RIGHT PANEL  │
  │  solid list  │   floating DEC mesh + orbit    │  symmetry +  │
  │  (scrollable)│   camera + face highlight      │  actions     │
  └──────────────┴───────────────────────────────┴──────────────┘

Rendering approach
------------------
- 3D workspace: standard GL 3.3 core (VAO/VBO), scissor-clipped sub-viewport
- 2D panels: pygame.Surface rendered by analytic_driver-style draw calls,
  uploaded as GL texture, drawn with _HUD2D_TEX shaders (RGBA-pass variant)

Interaction
-----------
- Left panel: click selects a solid → workspace.pick_solid(id)
- Workspace: mouse drag orbits; click on face → workspace.snap_to_face(fi)
- Right panel: symmetry buttons, Confirm, Undo, Store, Clear
"""
from __future__ import annotations

import ctypes
import math
import os
from typing import Any, Optional

import numpy as np
import pygame

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER, GL_LINES,
        GL_ONE_MINUS_SRC_ALPHA, GL_RGBA, GL_SCISSOR_TEST, GL_SRC_ALPHA,
        GL_STATIC_DRAW, GL_DYNAMIC_DRAW, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER, GL_LINEAR, GL_TRIANGLES, GL_TRUE,
        GL_UNSIGNED_BYTE, GL_VERTEX_SHADER,
        glBindBuffer, glBindTexture, glBindVertexArray, glBlendFunc,
        glBufferData, glClear, glClearColor, glDeleteBuffers,
        glDeleteTextures, glDeleteVertexArrays, glDisable, glDrawArrays,
        glEnable, glEnableVertexAttribArray, glGenBuffers, glGenTextures,
        glGenVertexArrays, glGetUniformLocation, glLineWidth,
        glScissor, glTexImage2D, glTexParameteri, glUniform1f, glUniform1i,
        glUniform2f, glUniform3f, glUniform4f, glUniformMatrix4fv,
        glUseProgram, glVertexAttribPointer, glViewport,
        glActiveTexture, GL_TEXTURE0,
    )
    from OpenGL.GL import shaders as _gl_shaders
    _HAS_GL = True
except ImportError:
    _HAS_GL = False

from fabricator_workspace import FabricatorWorkspace, WorkspaceMode
from dec_mesh import DECMesh
from fabricator_programmatic_blueprints import (
    ProgrammaticBuildResult,
    ProgrammaticBlueprint,
    ProgrammaticKnobSpec,
    build_mesh as _build_programmatic_mesh,
    default_knob_values as _default_programmatic_knob_values,
    load_programmatic_blueprints,
)
from controls import KnobSpec, Panel


# ─────────────────────────────────────────────────────────────────────────────
# GLSL shaders
# ─────────────────────────────────────────────────────────────────────────────

_MESH_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
uniform mat4 uMVP;
uniform mat4 uMV;
out vec3 vNormV;
out vec3 vPosV;
void main() {
    vec4 posV = uMV * vec4(aPos, 1.0);
    vPosV     = posV.xyz;
    vNormV    = mat3(uMV) * aNorm;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_MESH_FS = """
#version 330 core
in  vec3 vNormV;
in  vec3 vPosV;
out vec4 FragColor;
uniform vec4  uColor;
uniform vec3  uLightV;
uniform float uAmbient;
uniform float uSpecStr;
void main() {
    vec3  N = normalize(gl_FrontFacing ? vNormV : -vNormV);
    vec3  L = normalize(uLightV);
    vec3  V = normalize(-vPosV);
    float d = max(dot(N,L), 0.0);
    float s = pow(max(dot(normalize(L+V),N),0.0), 64.0);
    float rim = pow(1.0-max(dot(N,V),0.0), 3.0);
    vec3 col = uColor.rgb * (uAmbient + 0.75*d) + vec3(0.9,0.95,1.0)*uSpecStr*s
             + uColor.rgb * rim * 0.15;
    FragColor = vec4(col, uColor.a);
}
"""

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
void main() { FragColor = uColor; }
"""

# HUD panel texture: pixel-coords quad + RGBA texture
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
# Minimal orbit camera for the workspace viewport
# ─────────────────────────────────────────────────────────────────────────────

class _WsCam:
    def __init__(self, cfg: dict):
        self.az    = float(cfg.get("az",    35.0))
        self.elev  = float(cfg.get("elev",  22.0))
        self.dist  = float(cfg.get("dist",   2.8))
        self.fov   = math.radians(float(cfg.get("fov_deg", 50.0)))
        self._spin = float(cfg.get("auto_spin", 0.25))
        self.target = np.zeros(3, np.float64)

    def orbit(self, daz, delev):
        self.az   = (self.az + daz) % 360.0
        self.elev = float(np.clip(self.elev + delev, -80, 80))

    def zoom(self, d):
        self.dist = float(np.clip(self.dist + d, 0.3, 8.0))

    def tick(self):
        self.az = (self.az + self._spin) % 360.0

    @property
    def eye(self):
        az = math.radians(self.az); el = math.radians(self.elev)
        return self.target + self.dist * np.array([
            math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])

    def _view(self):
        eye = self.eye
        fwd = self.target - eye; fwd /= max(np.linalg.norm(fwd), 1e-12)
        up  = np.array([0., 0., 1.])
        if abs(np.dot(fwd, up)) > 0.97:
            up = np.array([0., 1., 0.])
        right = np.cross(fwd, up); right /= max(np.linalg.norm(right), 1e-12)
        up2   = np.cross(right, fwd)
        V = np.eye(4, dtype=np.float64)
        V[0, :3] =  right; V[0, 3] = -np.dot(right, eye)
        V[1, :3] =  up2;   V[1, 3] = -np.dot(up2,   eye)
        V[2, :3] = -fwd;   V[2, 3] =  np.dot(fwd,   eye)
        return V

    def mvp(self, aspect):
        f    = 1.0 / math.tan(self.fov / 2.0)
        near, far = 0.02, 20.0
        P    = np.zeros((4, 4), np.float64)
        P[0,0]=f/aspect; P[1,1]=f
        P[2,2]=-(far+near)/(far-near); P[2,3]=-2*far*near/(far-near); P[3,2]=-1.0
        return (P @ self._view()).astype(np.float32)

    def mv(self):
        return self._view().astype(np.float32)

    def unproject(self, px, py, vw, vh) -> tuple[np.ndarray, np.ndarray]:
        """Return (ray_origin, ray_dir) for screen pixel (px, py)."""
        ndc_x = (2.0 * px / vw) - 1.0
        ndc_y = 1.0 - (2.0 * py / vh)
        aspect = vw / max(1, vh)
        f  = math.tan(self.fov / 2.0)
        rd = np.array([ndc_x * f * aspect, ndc_y * f, -1.0], np.float64)
        V  = self._view()
        R3 = V[:3, :3].T   # inverse rotation
        return self.eye.copy(), R3 @ rd


# ─────────────────────────────────────────────────────────────────────────────
# Panel base (analytic_driver style: render → pygame.Surface, handle_event)
# ─────────────────────────────────────────────────────────────────────────────

class _FabPanel:
    PANEL_W = 210
    ROW_H   = 22
    PAD     = 6
    HDR_H   = 20

    def __init__(self):
        pygame.font.init()
        self._font  = pygame.font.SysFont("monospace", 13)
        self._font_s= pygame.font.SysFont("monospace", 11)
        self._ctrl  : dict[str, pygame.Rect] = {}
        self._scroll: int = 0

    def render(self, w: int, h: int) -> pygame.Surface:
        raise NotImplementedError

    def handle_event(self, ev, x_off: int = 0, y_off: int = 0) -> bool:
        return False

    # ── drawing helpers ───────────────────────────────────────────────────────

    def _section(self, surf, y, label, color=(50, 55, 70)) -> int:
        rect = pygame.Rect(0, y, surf.get_width(), self.HDR_H)
        pygame.draw.rect(surf, color, rect)
        txt = self._font.render(label, True, (180, 200, 220))
        surf.blit(txt, (self.PAD, y + 3))
        return y + self.HDR_H + 2

    def _label(self, surf, y, text, color=(140, 150, 160)):
        t = self._font_s.render(text, True, color)
        surf.blit(t, (self.PAD, y + 2))
        return y + self.ROW_H

    def _button(self, surf, y, key, label, w=None, active=False,
                x=None, bh=None) -> int:
        bw = w or (surf.get_width() - 2 * self.PAD)
        bx = x if x is not None else self.PAD
        bh = bh or (self.ROW_H + 2)
        r  = pygame.Rect(bx, y, bw, bh)
        bg = (70, 100, 150) if active else (45, 50, 65)
        pygame.draw.rect(surf, bg, r)
        pygame.draw.rect(surf, (100, 110, 130), r, 1)
        t  = self._font.render(label, True, (210, 220, 230))
        surf.blit(t, (bx + (bw - t.get_width()) // 2,
                      y  + (bh - t.get_height()) // 2))
        self._ctrl[key] = r
        return y + bh + 3

    def _toggle_row(self, surf, y, keys_labels, active_key) -> int:
        """Row of mutually-exclusive toggle buttons."""
        n  = len(keys_labels)
        w  = surf.get_width()
        bw = (w - 2 * self.PAD - (n - 1) * 2) // n
        x  = self.PAD
        for k, lbl in keys_labels:
            self._button(surf, y, k, lbl, w=bw, active=(k == active_key), x=x, bh=self.ROW_H)
            x += bw + 2
        return y + self.ROW_H + 3


# ─────────────────────────────────────────────────────────────────────────────
# Left panel — solid picker
# ─────────────────────────────────────────────────────────────────────────────

class _SolidPickerPanel(_FabPanel):
    ITEM_H   = 56
    ICON_SZ  = 38

    def __init__(self, palette: list, selected_id: Optional[str] = None,
                 blueprint_dir: str = "configs/duty_stations/fabricator/blueprints"):
        super().__init__()
        self.palette    = palette       # list of dict from palette.yaml
        self.selected   = selected_id
        self._hover     = -1
        self.PANEL_W    = 210
        self.blueprint_dir = str(blueprint_dir)
        self.tab = "shape_primitives"
        self._blueprints: list[dict] = []

    def _refresh_blueprints(self):
        self._blueprints = []
        if not os.path.isdir(self.blueprint_dir):
            return
        files = []
        for nm in os.listdir(self.blueprint_dir):
            low = str(nm).lower()
            if low.endswith(".blueprint.yaml") or low.endswith(".blueprint.yml") or low.endswith(".blueprint.json"):
                files.append(os.path.join(self.blueprint_dir, nm))
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for fp in files:
            base = os.path.basename(fp)
            label = base
            for suf in (".blueprint.yaml", ".blueprint.yml", ".blueprint.json"):
                if label.lower().endswith(suf):
                    label = label[: -len(suf)]
                    break
            self._blueprints.append({
                "id": f"blueprint::{fp}",
                "label": label,
                "subtitle": "Blueprint",
                "kind": "blueprint",
                "path": fp,
                "color_rgb": [0.22, 0.64, 0.78],
            })

    def _shape_entries(self) -> list[dict]:
        out = []
        for e in self.palette:
            if str(e.get("kind", "")) == "solid":
                out.append(e)
        return out

    def _active_entries(self) -> list[dict]:
        if self.tab == "blueprints":
            return list(self._blueprints)
        return self._shape_entries()

    def render(self, w: int, h: int) -> pygame.Surface:
        self._refresh_blueprints()
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((18, 20, 28, 240))

        # Header
        y = self._section(surf, 0, "  COMPONENTS", (28, 32, 44))
        y = self._toggle_row(
            surf,
            y,
            [("tab_shapes", "Shapes"), ("tab_blueprints", "Blueprints")],
            "tab_shapes" if self.tab == "shape_primitives" else "tab_blueprints",
        )

        # Items
        self._ctrl = {}
        items = self._active_entries()
        for idx, entry in enumerate(items):
            iy = y + idx * self.ITEM_H - self._scroll
            if iy + self.ITEM_H < 0 or iy > h:
                continue
            selected = (entry['id'] == self.selected)
            hovered  = (idx == self._hover)

            bg = (38, 55, 90) if selected else ((30, 35, 48) if hovered else (22, 25, 35))
            pygame.draw.rect(surf, bg, pygame.Rect(0, iy, w, self.ITEM_H))
            if selected:
                pygame.draw.rect(surf, (80, 130, 220), pygame.Rect(0, iy, 3, self.ITEM_H))

            # Icon: coloured square representing the solid
            c = entry.get('color_rgb', [0.5, 0.5, 0.5])
            ic = (int(c[0]*220), int(c[1]*220), int(c[2]*220))
            icon_r = pygame.Rect(8, iy + 9, self.ICON_SZ, self.ICON_SZ)
            pygame.draw.rect(surf, ic, icon_r)
            pygame.draw.rect(surf, (180, 185, 200), icon_r, 1)

            # Label + subtitle
            lbl = self._font.render(entry.get('label', entry['id']), True, (210, 220, 230))
            sub = self._font_s.render(entry.get('subtitle', ''), True, (120, 130, 145))
            surf.blit(lbl, (8 + self.ICON_SZ + 6, iy + 12))
            surf.blit(sub, (8 + self.ICON_SZ + 6, iy + 12 + lbl.get_height() + 2))

            # Hit rect (in panel-local coords)
            self._ctrl[f"item_{idx}"] = pygame.Rect(0, iy, w, self.ITEM_H)

        if self.tab == "blueprints" and not items:
            t = self._font_s.render("No blueprints found", True, (120, 130, 145))
            surf.blit(t, (self.PAD, y + 8))

        # Divider at bottom
        pygame.draw.line(surf, (40, 44, 58), (0, h-1), (w, h-1))
        return surf

    def handle_event(self, ev, x_off=0, y_off=0) -> tuple[bool, Optional[dict]]:
        """Returns (consumed, selection_event_or_None)."""
        if ev.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if x_off <= mx < x_off + self.PANEL_W:
                self._scroll = max(0, self._scroll - ev.y * 24)
                return True, None
        if ev.type == pygame.MOUSEMOTION:
            mx, my = ev.pos
            lx, ly = mx - x_off, my - y_off
            self._hover = -1
            for idx, _entry in enumerate(self._active_entries()):
                iy = 42 + idx * self.ITEM_H - self._scroll
                r  = pygame.Rect(0, iy, self.PANEL_W, self.ITEM_H)
                if r.collidepoint(lx, ly):
                    self._hover = idx
                    break
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            lx, ly = mx - x_off, my - y_off
            if 0 <= lx < self.PANEL_W:
                if (self._ctrl.get("tab_shapes") and self._ctrl["tab_shapes"].collidepoint(lx, ly)):
                    self.tab = "shape_primitives"
                    self._scroll = 0
                    return True, None
                if (self._ctrl.get("tab_blueprints") and self._ctrl["tab_blueprints"].collidepoint(lx, ly)):
                    self.tab = "blueprints"
                    self._scroll = 0
                    return True, None
                items = self._active_entries()
                for idx, entry in enumerate(items):
                    r = self._ctrl.get(f"item_{idx}")
                    if r and r.collidepoint(lx, ly):
                        self.selected = entry['id']
                        return True, {
                            "id": entry.get("id"),
                            "kind": entry.get("kind", "solid"),
                            "path": entry.get("path", ""),
                        }
                return True, None
        return False, None


# ─────────────────────────────────────────────────────────────────────────────
# Right panel — symmetry + actions
# ─────────────────────────────────────────────────────────────────────────────

class _SymmetryPanel(_FabPanel):
    def __init__(self, workspace: FabricatorWorkspace):
        super().__init__()
        self.ws = workspace
        self.PANEL_W = 220

    def render(self, w: int, h: int) -> pygame.Surface:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((18, 20, 28, 240))
        self._ctrl = {}
        ws   = self.ws
        mode = ws.mode
        sym  = ws.symmetry
        y    = 0

        # ── Process tabs ─────────────────────────────────────────────────────
        y = self._section(surf, y, "  FABRICATION")
        y = self._toggle_row(surf, y,
            [
                ("tab_milling", "Mill"),
                ("tab_drilling", "Drill"),
                ("tab_subtractive", "Cut"),
            ],
            f"tab_{ws.process_tab}" if ws.process_tab in ("milling", "drilling", "subtractive") else "")
        y = self._toggle_row(surf, y,
            [
                ("tab_additive", "Add"),
                ("tab_beveling", "Bevel"),
                ("tab_assembling", "Assemble"),
            ],
            f"tab_{ws.process_tab}" if ws.process_tab in ("additive", "beveling", "assembling") else "")

        y = self._label(surf, y, f"Gimbal pan/tilt: {ws.gimbal[0]:+.0f} / {ws.gimbal[1]:+.0f} deg")
        y = self._toggle_row(surf, y,
            [("g_pan_l", "Pan-"), ("g_pan_r", "Pan+"), ("g_tilt_l", "Tilt-"), ("g_tilt_r", "Tilt+")],
            "")
        y = self._button(surf, y, "g_reset", "Reset Gimbal")

        y = self._label(surf, y, f"Snap: {'ON' if ws.snap_enabled else 'OFF'}  step {ws.snap_angle_deg:.0f} deg")
        y = self._toggle_row(surf, y,
            [("snap_toggle", "Toggle"), ("snap_5", "5"), ("snap_15", "15"), ("snap_30", "30")],
            "")

        # ── Symmetry section ──────────────────────────────────────────────────
        y = self._section(surf, y, "  SYMMETRY")
        y = self._label(surf, y, "Mode")
        y = self._toggle_row(surf, y,
            [("sym_none","None"),("sym_bilateral","Mirror"),("sym_radial","Radial")],
            "sym_" + sym.mode)

        y = self._label(surf, y, "Axis")
        y = self._toggle_row(surf, y,
            [("ax_x","X"),("ax_y","Y"),("ax_z","Z")], "ax_" + sym.axis)

        if sym.mode == "radial":
            y = self._label(surf, y, f"Copies: {sym.count}")
            y = self._toggle_row(surf, y,
                [(f"cnt_{n}", str(n)) for n in [2, 3, 4, 5, 6, 8]],
                f"cnt_{sym.count}")

        y += 6

        # ── Actions ───────────────────────────────────────────────────────────
        y = self._section(surf, y, "  BUILD")

        if ws.process_tab in ("drilling", "subtractive"):
            y = self._button(surf, y, "op_drill", "Drill Cut (face)")
        if ws.process_tab in ("milling", "subtractive"):
            y = self._button(surf, y, "op_mill", "Mill Plane Cut")
        if ws.process_tab == "beveling":
            y = self._button(surf, y, "op_bevel", "Tag Bevel Step")
        if ws.process_tab == "assembling":
            y = self._button(surf, y, "op_assemble", "Assemble Preview")

        if mode == WorkspaceMode.PLACED:
            y = self._button(surf, y, "confirm", "[ OK ] Confirm")
            y = self._button(surf, y, "cancel_place", "[  X ] Cancel")
        elif mode in (WorkspaceMode.PICKED, WorkspaceMode.PLACING):
            y = self._label(surf, y, "Click a face to attach")

        if ws.built_mesh is not None:
            y += 4
            y = self._button(surf, y, "undo", "[<--] Undo")

        y += 10
        y = self._section(surf, y, "  STORE")
        y = self._button(surf, y, "import_blueprint", "Import Latest Blueprint")
        y = self._button(surf, y, "replay_blueprint", "Replay Blueprint")
        if mode == WorkspaceMode.IDLE and ws.built_mesh is not None:
            y = self._button(surf, y, "review", "Review Build")
        if mode == WorkspaceMode.REVIEW:
            y = self._button(surf, y, "store",  "[+] Store to Inventory")
            y = self._button(surf, y, "export_blueprint", "Export Blueprint")
            y = self._button(surf, y, "clear",  "[x] Clear Build")
            y = self._button(surf, y, "exit_review", "< Back")

        if ws.last_export_path:
            y = self._label(surf, y, "Exported:")
            y = self._label(surf, y, f" {ws.last_export_path}")
        if ws.last_import_path:
            y = self._label(surf, y, "Imported:")
            y = self._label(surf, y, f" {ws.last_import_path}")

        # ── Inventory ─────────────────────────────────────────────────────────
        if ws.inventory:
            y += 10
            y = self._section(surf, y, f"  INVENTORY ({len(ws.inventory)})")
            for i, (lbl, _) in enumerate(ws.inventory):
                t = self._font_s.render(f" {i+1}. {lbl}", True, (150, 180, 160))
                surf.blit(t, (self.PAD, y + 2))
                y += self.ROW_H - 2

        pygame.draw.line(surf, (40, 44, 58), (0, 0), (0, h))
        return surf

    def handle_event(self, ev, x_off=0, y_off=0) -> bool:
        if ev.type != pygame.MOUSEBUTTONDOWN or ev.button != 1:
            return False
        mx, my = ev.pos
        lx, ly = mx - x_off, my - y_off
        if not (0 <= lx < self.PANEL_W):
            return False

        ws = self.ws
        sym = ws.symmetry

        def _hit(key):
            r = self._ctrl.get(key)
            return r is not None and r.collidepoint(lx, ly)

        for tab in ("milling", "drilling", "subtractive", "additive", "beveling", "assembling"):
            if _hit(f"tab_{tab}"):
                ws.set_process_tab(tab)
                return True

        if _hit("g_pan_l"): ws.adjust_gimbal(-ws.snap_angle_deg, 0.0); return True
        if _hit("g_pan_r"): ws.adjust_gimbal(+ws.snap_angle_deg, 0.0); return True
        if _hit("g_tilt_l"): ws.adjust_gimbal(0.0, -ws.snap_angle_deg); return True
        if _hit("g_tilt_r"): ws.adjust_gimbal(0.0, +ws.snap_angle_deg); return True
        if _hit("g_reset"): ws.reset_gimbal(); return True

        if _hit("snap_toggle"): ws.set_snap(enabled=(not ws.snap_enabled)); return True
        if _hit("snap_5"): ws.set_snap(angle_deg=5.0); return True
        if _hit("snap_15"): ws.set_snap(angle_deg=15.0); return True
        if _hit("snap_30"): ws.set_snap(angle_deg=30.0); return True

        if _hit("op_drill"):
            if ws.hover_face >= 0:
                c = ws.active_mesh.face_center(ws.hover_face)
            else:
                c = ws.active_mesh.verts.mean(axis=0)
            ws.apply_subtractive_sphere_cut(c, max(0.08, ws.snap_distance_m * 1.5))
            return True

        if _hit("op_mill"):
            z_mid = float(np.median(ws.active_mesh.verts[:, 2]))
            ws.apply_subtractive_plane_cut(axis="z", offset=z_mid, keep_negative=True)
            return True

        if _hit("op_bevel"):
            ws.mark_bevel(amount_m=max(0.002, ws.snap_distance_m * 0.5))
            return True

        if _hit("op_assemble"):
            ws.enter_review()
            return True

        if _hit("sym_none"):      ws.set_symmetry("none");      return True
        if _hit("sym_bilateral"): ws.set_symmetry("bilateral"); return True
        if _hit("sym_radial"):    ws.set_symmetry("radial");    return True
        if _hit("ax_x"):  ws.set_symmetry(sym.mode, axis="x"); return True
        if _hit("ax_y"):  ws.set_symmetry(sym.mode, axis="y"); return True
        if _hit("ax_z"):  ws.set_symmetry(sym.mode, axis="z"); return True
        for n in [2,3,4,5,6,8]:
            if _hit(f"cnt_{n}"): ws.set_symmetry(sym.mode, count=n); return True

        if _hit("confirm"):      ws.confirm_placement(); return True
        if _hit("cancel_place"): ws.cancel_placement(); return True
        if _hit("undo"):         ws.undo(); return True
        if _hit("import_blueprint"):
            ws.import_latest_blueprint()
            return True
        if _hit("replay_blueprint"):
            ws.replay_blueprint()
            return True
        if _hit("review"):       ws.enter_review(); return True
        if _hit("store"):        ws.store_to_inventory(); return True
        if _hit("export_blueprint"):
            ws.export_fabrication_blueprint()
            return True
        if _hit("clear"):        ws.clear_build(); return True
        if _hit("exit_review"):  ws.exit_review(); return True

        return True   # absorb any click in panel area


# ─────────────────────────────────────────────────────────────────────────────
# GL upload helper
# ─────────────────────────────────────────────────────────────────────────────

def _surface_to_tex(surf: pygame.Surface) -> int:
    w, h  = surf.get_size()
    raw   = pygame.image.tobytes(surf, "RGBA", True)   # flip → GL bottom-up convention
    tex   = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def _compile(vs, fs):
    return _gl_shaders.compileProgram(
        _gl_shaders.compileShader(vs, GL_VERTEX_SHADER),
        _gl_shaders.compileShader(fs, GL_FRAGMENT_SHADER))


def _make_vao_pn(data: np.ndarray):
    """Upload (N,6) float32 pos+norm.  Returns (vao, vbo, n)."""
    vao = glGenVertexArrays(1); vbo = glGenBuffers(1)
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data.tobytes(), GL_STATIC_DRAW)
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,24,ctypes.c_void_p(0))
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,24,ctypes.c_void_p(12))
    glBindVertexArray(0); return vao, vbo, len(data)


def _make_vao_p(data: np.ndarray):
    """Upload (N,3) float32 positions.  Returns (vao, vbo, n)."""
    vao = glGenVertexArrays(1); vbo = glGenBuffers(1)
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data.tobytes(), GL_STATIC_DRAW)
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,12,ctypes.c_void_p(0))
    glBindVertexArray(0); return vao, vbo, len(data)


# ─────────────────────────────────────────────────────────────────────────────
# FabricatorStation — top-level: wires workspace + renderer + panels
# ─────────────────────────────────────────────────────────────────────────────

class FabricatorStation:
    """GL renderer for the fabricator.  Create after GL context ready."""

    def __init__(self, ws_cfg: dict, palette: list):
        self.workspace = FabricatorWorkspace(ws_cfg)
        base_palette = self.workspace.catalog_items()
        if palette:
            by_id = {str(it.get("id", "")): dict(it) for it in base_palette}
            for it in palette:
                iid = str(it.get("id", ""))
                if iid:
                    by_id[iid] = dict(it)
            self._palette = list(by_id.values())
        else:
            self._palette = base_palette
        self._layout   = ws_cfg.get("layout", {})
        self.LEFT_W    = int(self._layout.get("left_panel_width",  210))
        self.RIGHT_W   = int(self._layout.get("right_panel_width", 220))

        display_cfg    = ws_cfg.get("display", {})
        self._face_col = display_cfg.get("face_color",         [0.15, 0.20, 0.30])
        self._edge_col = display_cfg.get("edge_color",         [0.20, 0.85, 0.65])
        self._sel_col  = display_cfg.get("selected_face_color",[0.10, 0.70, 0.90])
        self._ghost_a  = float(display_cfg.get("ghost_alpha",  0.42))
        self._solid_a  = float(display_cfg.get("solid_alpha",  0.80))
        self._bg       = display_cfg.get("bg_color",           [0.02, 0.03, 0.06])
        self._wf_w     = float(display_cfg.get("wireframe_width", 1.4))
        self._blueprint_dir = ws_cfg.get(
            "blueprint_library_dir",
            "configs/duty_stations/fabricator/blueprints",
        )
        self._programmatic_dir = ws_cfg.get(
            "programmatic_blueprint_dir",
            "configs/duty_stations/fabricator/programmatic_blueprints",
        )

        self.ws_cam = _WsCam(ws_cfg.get("camera", {}))

        self._left_panel  = _SolidPickerPanel(self._palette, blueprint_dir=self._blueprint_dir)
        self._right_panel = _SymmetryPanel(self.workspace)

        # Center-region mode tabs and programmatic blueprint state.
        self._center_mode = "workspace"  # workspace | programmatic
        self._center_tab_rects: dict[str, pygame.Rect] = {}
        self._prog_blueprints: list[ProgrammaticBlueprint] = []
        self._prog_selected_id: str = ""
        self._prog_values_by_id: dict[str, dict[str, Any]] = {}
        self._prog_library_rects: dict[str, pygame.Rect] = {}
        self._prog_knob_rects: dict[str, dict[str, pygame.Rect]] = {}
        self._prog_group_rects: dict[str, pygame.Rect] = {}
        self._prog_group_open_by_bp: dict[str, dict[str, bool]] = {}
        self._refresh_programmatic_library()

        # GL state
        self._gl_ready = False
        self._prog_mesh = self._prog_line = self._prog_hud = None
        self._mesh_vao = self._mesh_vbo = self._mesh_n = None
        self._edge_vao = self._edge_vbo = self._edge_n = None
        self._hud_vao  = self._hud_vbo  = None
        self._ltex = self._rtex = 0   # panel GL textures
        self._ctab_tex = self._cprog_l_tex = self._cprog_r_tex = 0
        self._mesh_dirty = True
        self._dragging   = False
        self._last_mouse = (0, 0)
        self._hud_visible = False
        self._last_win_w = 1280
        self._last_win_h = 720

    @staticmethod
    def _choice_index(value: Any, choices: list[str]) -> int:
        try:
            return choices.index(str(value))
        except ValueError:
            return 0

    @staticmethod
    def _programmatic_knob_to_knobspec(prefix: str, spec: ProgrammaticKnobSpec) -> KnobSpec:
        dtype = "choice" if spec.dtype == "choice" else spec.dtype
        choices = list(spec.choices or []) if dtype == "choice" else []
        return KnobSpec(
            f"{prefix}{spec.name}",
            spec.label,
            dtype,
            spec.default,
            float(spec.low or 0.0),
            float(spec.high or 1.0),
            float(spec.step or 0.0),
            "",
            choices,
            False,
            spec.group or "Parameters",
            spec.fmt or ".3f",
        )

    @property
    def panel_spec(self) -> Panel:
        """Hierarchical document descriptor for the station controls."""
        ws = self.workspace
        process_tabs = ["milling", "drilling", "subtractive", "additive", "beveling", "assembling"]
        mode_choices = [m.value for m in WorkspaceMode]
        active_bp = self._selected_programmatic_blueprint()

        panels = [
            Panel(
                "fabricator_library",
                "Library",
                knobs=[
                    KnobSpec("library_tab", "Catalog tab", "choice", 0, 0, 1, 1, "", ["shapes", "blueprints"], False, "Library"),
                    KnobSpec("selected_item", "Selected item", "str", "", 0, 0, 0, "", [], False, "Library"),
                    KnobSpec("center_mode", "Center mode", "choice", 0, 0, 1, 1, "", ["workspace", "programmatic"], False, "Library"),
                    KnobSpec("programmatic_blueprint", "Programmatic BP", "str", "", 0, 0, 0, "", [], False, "Library"),
                ],
            ),
            Panel(
                "fabricator_workspace",
                "Workspace",
                knobs=[
                    KnobSpec("workspace_mode", "Mode", "choice", 0, 0, max(0, len(mode_choices) - 1), 1, "", mode_choices, False, "Workspace"),
                    KnobSpec("process_tab", "Process", "choice", 0, 0, len(process_tabs) - 1, 1, "", process_tabs, False, "Workspace"),
                    KnobSpec("picked_id", "Picked", "str", "", 0, 0, 0, "", [], False, "Workspace"),
                    KnobSpec("hover_face", "Hover face", "int", -1, -1, 100000, 1, "", [], False, "Workspace", ".0f"),
                    KnobSpec("operation_count", "Operations", "int", 0, 0, 100000, 1, "", [], False, "Workspace", ".0f"),
                ],
            ),
            Panel(
                "fabricator_snap_gimbal",
                "Snap / Gimbal",
                knobs=[
                    KnobSpec("gimbal_pan_deg", "Pan", "float", 0.0, -360.0, 360.0, 0, "deg", [], False, "Gimbal", ".1f"),
                    KnobSpec("gimbal_tilt_deg", "Tilt", "float", 0.0, -85.0, 85.0, 0, "deg", [], False, "Gimbal", ".1f"),
                    KnobSpec("snap_enabled", "Snap", "bool", True, 0, 1, 1, "", [], False, "Snap"),
                    KnobSpec("snap_angle_deg", "Snap angle", "float", 15.0, 1.0, 90.0, 0, "deg", [], False, "Snap", ".1f"),
                    KnobSpec("snap_distance_m", "Snap dist", "float", 0.08, 0.001, 1.0, 0, "m", [], False, "Snap", ".3f"),
                ],
            ),
            Panel(
                "fabricator_symmetry",
                "Symmetry",
                knobs=[
                    KnobSpec("symmetry_mode", "Mode", "choice", 0, 0, 2, 1, "", ["none", "bilateral", "radial"], False, "Symmetry"),
                    KnobSpec("symmetry_axis", "Axis", "choice", 2, 0, 2, 1, "", ["x", "y", "z"], False, "Symmetry"),
                    KnobSpec("symmetry_count", "Count", "int", 4, 2, 16, 1, "", [], False, "Symmetry", ".0f"),
                ],
            ),
            Panel(
                "fabricator_store",
                "Store",
                knobs=[
                    KnobSpec("inventory_count", "Inventory", "int", 0, 0, 100000, 1, "", [], False, "Store", ".0f"),
                    KnobSpec("last_export_path", "Last export", "str", "", 0, 0, 0, "", [], False, "Store"),
                    KnobSpec("last_import_path", "Last import", "str", "", 0, 0, 0, "", [], False, "Store"),
                ],
            ),
        ]

        if active_bp is not None:
            panels.append(Panel(
                "fabricator_programmatic_params",
                f"Programmatic: {active_bp.label}",
                knobs=[
                    self._programmatic_knob_to_knobspec("programmatic.", spec)
                    for spec in active_bp.knobspec
                ],
            ))

        return Panel("fabricator_station", "Fabricator", panels=panels)

    @property
    def knob_values(self) -> dict[str, Any]:
        ws = self.workspace
        process_tabs = ["milling", "drilling", "subtractive", "additive", "beveling", "assembling"]
        mode_choices = [m.value for m in WorkspaceMode]
        sym_modes = ["none", "bilateral", "radial"]
        axes = ["x", "y", "z"]
        active_bp = self._selected_programmatic_blueprint()
        vals: dict[str, Any] = {
            "library_tab": 1 if self._left_panel.tab == "blueprints" else 0,
            "selected_item": self._left_panel.selected or "",
            "center_mode": 1 if self._center_mode == "programmatic" else 0,
            "programmatic_blueprint": self._prog_selected_id,
            "workspace_mode": self._choice_index(getattr(ws.mode, "value", ws.mode), mode_choices),
            "process_tab": self._choice_index(ws.process_tab, process_tabs),
            "picked_id": ws.picked_id or "",
            "hover_face": int(ws.hover_face),
            "operation_count": len(ws.operations),
            "gimbal_pan_deg": float(ws.gimbal[0]),
            "gimbal_tilt_deg": float(ws.gimbal[1]),
            "snap_enabled": bool(ws.snap_enabled),
            "snap_angle_deg": float(ws.snap_angle_deg),
            "snap_distance_m": float(ws.snap_distance_m),
            "symmetry_mode": self._choice_index(ws.symmetry.mode, sym_modes),
            "symmetry_axis": self._choice_index(ws.symmetry.axis, axes),
            "symmetry_count": int(ws.symmetry.count),
            "inventory_count": len(ws.inventory),
            "last_export_path": ws.last_export_path,
            "last_import_path": ws.last_import_path,
        }
        if active_bp is not None:
            bp_vals = self._prog_values_by_id.setdefault(active_bp.id, _default_programmatic_knob_values(active_bp))
            for spec in active_bp.knobspec:
                key = f"programmatic.{spec.name}"
                value = bp_vals.get(spec.name, spec.default)
                if spec.dtype == "choice":
                    vals[key] = self._choice_index(value, list(spec.choices or []))
                else:
                    vals[key] = value
        return vals

    @classmethod
    def from_yaml(cls, ws_path: str, palette_path: str) -> "FabricatorStation":
        import yaml as _y
        with open(ws_path,      encoding='utf-8') as f: ws  = _y.safe_load(f) or {}
        with open(palette_path, encoding='utf-8') as f: pal = _y.safe_load(f) or {}
        return cls(ws, pal.get("solids", []))

    def _refresh_programmatic_library(self):
        self._prog_blueprints = load_programmatic_blueprints(self._programmatic_dir)
        if self._prog_blueprints and (self._prog_selected_id not in {bp.id for bp in self._prog_blueprints}):
            self._prog_selected_id = self._prog_blueprints[0].id
        for bp in self._prog_blueprints:
            if bp.id not in self._prog_values_by_id:
                self._prog_values_by_id[bp.id] = _default_programmatic_knob_values(bp)
            self._ensure_prog_group_state(bp)

    def _ensure_prog_group_state(self, bp: ProgrammaticBlueprint):
        state = self._prog_group_open_by_bp.setdefault(bp.id, {})
        for spec in bp.knobspec:
            grp = str(spec.group or "").strip()
            if not grp:
                continue
            if grp not in state:
                state[grp] = bool(spec.group_default_expanded)

    @staticmethod
    def _grouped_knobs(bp: ProgrammaticBlueprint) -> list[tuple[str, list[ProgrammaticKnobSpec]]]:
        groups: list[tuple[str, list[ProgrammaticKnobSpec]]] = []
        idx: dict[str, int] = {}
        for spec in bp.knobspec:
            grp = str(spec.group or "").strip() or "Main"
            gi = idx.get(grp)
            if gi is None:
                idx[grp] = len(groups)
                groups.append((grp, [spec]))
            else:
                groups[gi][1].append(spec)
        return groups

    def _selected_programmatic_blueprint(self) -> Optional[ProgrammaticBlueprint]:
        for bp in self._prog_blueprints:
            if bp.id == self._prog_selected_id:
                return bp
        return None

    def _center_layout(self, win_w: int, win_h: int) -> dict[str, int]:
        cw = win_w - self.LEFT_W - self.RIGHT_W
        tab_h = 24
        prog_lw = 0
        prog_rw = 0
        if self._center_mode == "programmatic":
            prog_lw = max(180, min(260, cw // 3))
            prog_rw = max(200, min(300, cw // 3))
            if prog_lw + prog_rw > max(0, cw - 140):
                over = prog_lw + prog_rw - (cw - 140)
                cut_l = min(prog_lw - 120, over // 2)
                cut_r = min(prog_rw - 140, over - cut_l)
                prog_lw -= max(0, cut_l)
                prog_rw -= max(0, cut_r)
        view_x = self.LEFT_W + prog_lw
        view_w = max(120, cw - prog_lw - prog_rw)
        return {
            "cw": cw,
            "tab_h": tab_h,
            "prog_lw": prog_lw,
            "prog_rw": prog_rw,
            "view_x": view_x,
            "view_w": view_w,
            "view_h": max(1, win_h),
        }

    def _render_center_tabs(self, cw: int, ch: int) -> pygame.Surface:
        surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
        self._center_tab_rects = {}
        tab_h = 24
        pygame.draw.rect(surf, (22, 25, 34, 230), pygame.Rect(0, 0, cw, tab_h))
        tabs = [("workspace", "WORKSPACE"), ("programmatic", "PROGRAMMATIC")]
        x = 8
        for key, label in tabs:
            tw = 136 if key == "programmatic" else 104
            r = pygame.Rect(x, 3, tw, tab_h - 6)
            self._center_tab_rects[key] = r
            active = (self._center_mode == key)
            bg = (40, 90, 160) if active else (30, 34, 48)
            pygame.draw.rect(surf, bg, r, border_radius=3)
            pygame.draw.rect(surf, (70, 80, 100), r, 1, border_radius=3)
            t = self._left_panel._font_s.render(label, True, (220, 230, 245) if active else (150, 160, 180))
            surf.blit(t, (r.x + (r.w - t.get_width()) // 2, r.y + (r.h - t.get_height()) // 2))
            x += tw + 6
        return surf

    def _render_programmatic_library_panel(self, w: int, h: int) -> pygame.Surface:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((18, 20, 28, 230))
        self._prog_library_rects = {}
        y = 0
        y = self._left_panel._section(surf, y, "  PROGRAMMATIC LIBRARY", (28, 32, 44))
        y = self._left_panel._button(surf, y, "prog_refresh", "Refresh", w=max(80, w - 12))
        for bp in self._prog_blueprints:
            iy = y
            r = pygame.Rect(0, iy, w, 44)
            sel = (bp.id == self._prog_selected_id)
            bg = (38, 55, 90) if sel else (22, 25, 35)
            pygame.draw.rect(surf, bg, r)
            if sel:
                pygame.draw.rect(surf, (80, 130, 220), pygame.Rect(0, iy, 3, 44))
            t1 = self._left_panel._font.render(bp.label, True, (210, 220, 230))
            t2 = self._left_panel._font_s.render(bp.id, True, (120, 130, 145))
            surf.blit(t1, (8, iy + 6))
            surf.blit(t2, (8, iy + 24))
            self._prog_library_rects[bp.id] = r
            y += 44
            if y > h - 12:
                break
        if not self._prog_blueprints:
            t = self._left_panel._font_s.render("No programmatic blueprints", True, (120, 130, 145))
            surf.blit(t, (8, y + 4))
        return surf

    def _knob_value_text(self, spec: ProgrammaticKnobSpec, value: Any) -> str:
        if spec.dtype == "bool":
            return "ON" if bool(value) else "OFF"
        if spec.dtype == "choice":
            return str(value)
        if spec.dtype == "int":
            return f"{int(round(float(value)))}"
        try:
            fv = float(value)
            if "%" in spec.fmt:
                return spec.fmt % fv
            return f"{fv:{spec.fmt}}"
        except Exception:
            try:
                return f"{float(value):.3f}"
            except Exception:
                return str(value)

    def _render_programmatic_knob_panel(self, w: int, h: int) -> pygame.Surface:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((18, 20, 28, 230))
        self._prog_knob_rects = {}
        self._prog_group_rects = {}
        y = 0
        y = self._left_panel._section(surf, y, "  PARAMETERS", (28, 32, 44))
        bp = self._selected_programmatic_blueprint()
        if bp is None:
            t = self._left_panel._font_s.render("Select a blueprint", True, (140, 150, 160))
            surf.blit(t, (8, y + 8))
            return surf

        vals = self._prog_values_by_id.setdefault(bp.id, _default_programmatic_knob_values(bp))
        self._ensure_prog_group_state(bp)
        group_state = self._prog_group_open_by_bp.setdefault(bp.id, {})
        for grp_name, grp_specs in self._grouped_knobs(bp):
            if y > h - 28:
                break
            collapsible = any(bool(s.group_collapsible) for s in grp_specs)
            if collapsible:
                row = pygame.Rect(0, y, w, 24)
                pygame.draw.rect(surf, (24, 30, 42), row)
                pygame.draw.rect(surf, (52, 64, 86), row, 1)
                open_state = bool(group_state.get(grp_name, True))
                chevron = "v" if open_state else ">"
                gt = self._left_panel._font_s.render(f"{chevron} {grp_name}", True, (180, 196, 220))
                surf.blit(gt, (8, y + 5))
                self._prog_group_rects[grp_name] = row
                y += 24
            if collapsible and not bool(group_state.get(grp_name, True)):
                continue

            for spec in grp_specs:
                if y > h - 32:
                    break
                row = pygame.Rect(0, y, w, 34)
                pygame.draw.rect(surf, (22, 25, 35), row)
                label = self._left_panel._font_s.render(spec.label, True, (205, 215, 225))
                value = self._left_panel._font_s.render(self._knob_value_text(spec, vals.get(spec.name, spec.default)), True, (140, 170, 200))
                surf.blit(label, (8, y + 5))
                surf.blit(value, (8, y + 18))

                ctrl: dict[str, pygame.Rect] = {}
                if spec.dtype in ("float", "int"):
                    r_minus = pygame.Rect(w - 64, y + 7, 26, 20)
                    r_plus = pygame.Rect(w - 34, y + 7, 26, 20)
                    pygame.draw.rect(surf, (45, 50, 65), r_minus); pygame.draw.rect(surf, (100, 110, 130), r_minus, 1)
                    pygame.draw.rect(surf, (45, 50, 65), r_plus); pygame.draw.rect(surf, (100, 110, 130), r_plus, 1)
                    tm = self._left_panel._font.render("-", True, (210, 220, 230))
                    tp = self._left_panel._font.render("+", True, (210, 220, 230))
                    surf.blit(tm, (r_minus.x + (r_minus.w - tm.get_width()) // 2, r_minus.y + (r_minus.h - tm.get_height()) // 2 - 1))
                    surf.blit(tp, (r_plus.x + (r_plus.w - tp.get_width()) // 2, r_plus.y + (r_plus.h - tp.get_height()) // 2 - 1))
                    ctrl["minus"] = r_minus
                    ctrl["plus"] = r_plus
                else:
                    r_cycle = pygame.Rect(w - 88, y + 7, 80, 20)
                    pygame.draw.rect(surf, (45, 50, 65), r_cycle); pygame.draw.rect(surf, (100, 110, 130), r_cycle, 1)
                    tc = self._left_panel._font_s.render("Cycle", True, (210, 220, 230))
                    surf.blit(tc, (r_cycle.x + (r_cycle.w - tc.get_width()) // 2, r_cycle.y + (r_cycle.h - tc.get_height()) // 2))
                    ctrl["cycle"] = r_cycle
                self._prog_knob_rects[spec.name] = ctrl
                y += 34

        y += 6
        r_print = pygame.Rect(8, y, max(90, w - 16), 24)
        pygame.draw.rect(surf, (70, 100, 150), r_print); pygame.draw.rect(surf, (100, 120, 160), r_print, 1)
        tprint = self._left_panel._font.render("Print Item", True, (220, 230, 240))
        surf.blit(tprint, (r_print.x + (r_print.w - tprint.get_width()) // 2, r_print.y + (r_print.h - tprint.get_height()) // 2))
        self._prog_knob_rects["__print__"] = {"click": r_print}
        return surf

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def build_gl(self):
        if not _HAS_GL: return
        self._prog_mesh = _compile(_MESH_VS, _MESH_FS)
        self._prog_line = _compile(_LINE_VS, _LINE_FS)
        self._prog_hud  = _compile(_HUD_VS,  _HUD_FS)
        # HUD quad VAO: 4 vertices × (x,y,u,v)
        self._hud_vao = glGenVertexArrays(1); self._hud_vbo = glGenBuffers(1)
        glBindVertexArray(self._hud_vao); glBindBuffer(GL_ARRAY_BUFFER, self._hud_vbo)
        glBufferData(GL_ARRAY_BUFFER, 4*4*4, None, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,16,ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,16,ctypes.c_void_p(8))
        glBindVertexArray(0)
        self._gl_ready = True
        self._upload_mesh()

    def _upload_mesh(self):
        mesh = self.workspace.active_mesh
        tris = mesh.gl_triangles()
        if self._mesh_vao is not None:
            glDeleteVertexArrays(1, [self._mesh_vao]); glDeleteBuffers(1, [self._mesh_vbo])
        self._mesh_vao, self._mesh_vbo, self._mesh_n = _make_vao_pn(tris)

        edges = mesh.gl_edges().reshape(-1, 3).astype(np.float32)
        if self._edge_vao is not None:
            glDeleteVertexArrays(1, [self._edge_vao]); glDeleteBuffers(1, [self._edge_vbo])
        self._edge_vao, self._edge_vbo, self._edge_n = _make_vao_p(edges)
        self._mesh_dirty = False

    # ── Per-frame ─────────────────────────────────────────────────────────────

    def show_hud(self, visible: bool) -> None:
        self._hud_visible = bool(visible)

    def render_hud(self, win_w: int, win_h: int) -> None:
        if not self._hud_visible:
            return
        self._last_win_w = int(win_w)
        self._last_win_h = int(win_h)
        self.tick(0.0)
        self.draw(win_w, win_h)

    def tick(self, dt: float):
        self.ws_cam.tick()
        if self._mesh_dirty and self._gl_ready:
            self._upload_mesh()

    # ── Main draw ─────────────────────────────────────────────────────────────

    def draw(self, win_w: int, win_h: int):
        if not self._gl_ready: return
        lay = self._center_layout(win_w, win_h)
        cw = lay["cw"]
        tab_h = lay["tab_h"]
        prog_lw = lay["prog_lw"]
        prog_rw = lay["prog_rw"]
        view_x = lay["view_x"]
        view_w = lay["view_w"]
        view_h = lay["view_h"]

        # ── 3D workspace viewport ─────────────────────────────────────────────
        glEnable(GL_SCISSOR_TEST)
        glScissor(view_x, 0, view_w, view_h)
        bg = self._bg
        glClearColor(bg[0], bg[1], bg[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_SCISSOR_TEST)
        glViewport(view_x, 0, view_w, view_h)

        aspect = view_w / max(1, view_h)
        MVP = self.ws_cam.mvp(aspect)
        MV  = self.ws_cam.mv()
        lv  = np.array([0.6, 0.8, 0.5], np.float32)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Solid faces
        glUseProgram(self._prog_mesh)
        self._set_mvp(self._prog_mesh, MVP, MV)
        r,g,b = self._face_col
        glUniform4f(glGetUniformLocation(self._prog_mesh, b'uColor'), r,g,b, self._solid_a)
        glUniform3f(glGetUniformLocation(self._prog_mesh, b'uLightV'), *lv)
        glUniform1f(glGetUniformLocation(self._prog_mesh, b'uAmbient'),  0.22)
        glUniform1f(glGetUniformLocation(self._prog_mesh, b'uSpecStr'),  0.55)
        glBindVertexArray(self._mesh_vao)
        glDrawArrays(GL_TRIANGLES, 0, self._mesh_n)

        # Selected face highlight
        ws = self.workspace
        hi = ws.hover_face
        if hi >= 0 and self._mesh_n > 0:
            try:
                hl = ws.active_mesh.gl_face_highlight(hi)
                if len(hl):
                    hl_vao, hl_vbo, hl_n = _make_vao_pn(hl)
                    r,g,b = self._sel_col
                    glUniform4f(glGetUniformLocation(self._prog_mesh, b'uColor'), r,g,b, 0.70)
                    glBindVertexArray(hl_vao)
                    glDrawArrays(GL_TRIANGLES, 0, hl_n)
                    glDeleteVertexArrays(1, [hl_vao]); glDeleteBuffers(1, [hl_vbo])
            except Exception:
                pass

        # Wireframe
        glUseProgram(self._prog_line)
        self._set_mvp_line(self._prog_line, MVP)
        r,g,b = self._edge_col
        glUniform4f(glGetUniformLocation(self._prog_line, b'uColor'), r,g,b, 0.85)
        glLineWidth(self._wf_w)
        glBindVertexArray(self._edge_vao)
        glDrawArrays(GL_LINES, 0, self._edge_n)
        glBindVertexArray(0)

        # Restore full viewport for 2D overlays
        glViewport(0, 0, win_w, win_h)
        glDisable(GL_DEPTH_TEST)

        # ── Left panel ────────────────────────────────────────────────────────
        lsurf = self._left_panel.render(self.LEFT_W, win_h)
        if self._ltex:
            glDeleteTextures([self._ltex])
        self._ltex = _surface_to_tex(lsurf)
        self._draw_panel_tex(self._ltex, 0, 0, self.LEFT_W, win_h, win_w, win_h)

        # ── Right panel ───────────────────────────────────────────────────────
        rsurf = self._right_panel.render(self.RIGHT_W, win_h)
        if self._rtex:
            glDeleteTextures([self._rtex])
        self._rtex = _surface_to_tex(rsurf)
        self._draw_panel_tex(self._rtex, win_w - self.RIGHT_W, 0,
                             self.RIGHT_W, win_h, win_w, win_h)

        # ── Center tab bar and programmatic side panels ─────────────────────
        ctab = self._render_center_tabs(cw, tab_h)
        if self._ctab_tex:
            glDeleteTextures([self._ctab_tex])
        self._ctab_tex = _surface_to_tex(ctab)
        self._draw_panel_tex(self._ctab_tex, self.LEFT_W, 0, cw, tab_h, win_w, win_h)

        if self._center_mode == "programmatic":
            lp = self._render_programmatic_library_panel(prog_lw, win_h - tab_h)
            rp = self._render_programmatic_knob_panel(prog_rw, win_h - tab_h)
            if self._cprog_l_tex:
                glDeleteTextures([self._cprog_l_tex])
            if self._cprog_r_tex:
                glDeleteTextures([self._cprog_r_tex])
            self._cprog_l_tex = _surface_to_tex(lp)
            self._cprog_r_tex = _surface_to_tex(rp)
            self._draw_panel_tex(self._cprog_l_tex, self.LEFT_W, tab_h, prog_lw, win_h - tab_h, win_w, win_h)
            self._draw_panel_tex(self._cprog_r_tex, self.LEFT_W + cw - prog_rw, tab_h, prog_rw, win_h - tab_h, win_w, win_h)

    def _set_mvp(self, prog, mvp, mv):
        loc = glGetUniformLocation(prog, b'uMVP')
        if loc >= 0: glUniformMatrix4fv(loc, 1, GL_TRUE, mvp)
        loc = glGetUniformLocation(prog, b'uMV')
        if loc >= 0: glUniformMatrix4fv(loc, 1, GL_TRUE, mv)

    def _set_mvp_line(self, prog, mvp):
        loc = glGetUniformLocation(prog, b'uMVP')
        if loc >= 0: glUniformMatrix4fv(loc, 1, GL_TRUE, mvp)

    def _draw_panel_tex(self, tex, x, y, w, h, win_w, win_h):
        """Draw a GL texture as a screen-space quad using the HUD shader."""
        quad = np.array([
            [x,   y,   0.0, 1.0],
            [x+w, y,   1.0, 1.0],
            [x,   y+h, 0.0, 0.0],
            [x,   y+h, 0.0, 0.0],
            [x+w, y,   1.0, 1.0],
            [x+w, y+h, 1.0, 0.0],
        ], np.float32)
        glUseProgram(self._prog_hud)
        glUniform2f(glGetUniformLocation(self._prog_hud, b'uRes'),
                    float(win_w), float(win_h))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(glGetUniformLocation(self._prog_hud, b'uTex'), 0)
        glBindVertexArray(self._hud_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._hud_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad.tobytes(), GL_DYNAMIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

    def _adjust_programmatic_knob(self, spec: ProgrammaticKnobSpec, direction: int):
        bp = self._selected_programmatic_blueprint()
        if bp is None:
            return
        vals = self._prog_values_by_id.setdefault(bp.id, _default_programmatic_knob_values(bp))
        cur = vals.get(spec.name, spec.default)
        step = spec.step
        if spec.dtype == "int":
            try:
                nv = int(round(float(cur))) + int(round(float(step))) * int(direction)
            except Exception:
                nv = int(spec.default)
            if spec.low is not None:
                nv = max(int(round(float(spec.low))), nv)
            if spec.high is not None:
                nv = min(int(round(float(spec.high))), nv)
            vals[spec.name] = nv
            return
        try:
            nv = float(cur) + float(step) * float(direction)
        except Exception:
            nv = float(spec.default)
        if spec.low is not None:
            nv = max(float(spec.low), nv)
        if spec.high is not None:
            nv = min(float(spec.high), nv)
        vals[spec.name] = nv

    def _cycle_programmatic_knob(self, spec: ProgrammaticKnobSpec):
        bp = self._selected_programmatic_blueprint()
        if bp is None:
            return
        vals = self._prog_values_by_id.setdefault(bp.id, _default_programmatic_knob_values(bp))
        cur = vals.get(spec.name, spec.default)
        if spec.dtype == "bool":
            vals[spec.name] = not bool(cur)
            return
        if spec.dtype == "choice" and spec.choices:
            choices = list(spec.choices)
            try:
                i = choices.index(cur)
                vals[spec.name] = choices[(i + 1) % len(choices)]
            except Exception:
                vals[spec.name] = choices[0]

    def _print_selected_programmatic_blueprint(self):
        bp = self._selected_programmatic_blueprint()
        if bp is None:
            return
        vals = dict(self._prog_values_by_id.get(bp.id, _default_programmatic_knob_values(bp)))
        build = _build_programmatic_mesh(bp, vals)
        if not isinstance(build, ProgrammaticBuildResult) or build.mesh is None:
            return
        self.workspace.pick_generated_mesh(
            bp.id,
            build.mesh,
            metadata={
                "face_normals": build.face_normals,
                "side_policy": build.side_policy,
            },
        )
        self._center_mode = "workspace"
        self._mesh_dirty = True

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev, win_w: int | None = None,
                     win_h: int | None = None) -> bool:
        if not self._hud_visible:
            return False
        if win_w is None:
            win_w = self._last_win_w
        if win_h is None:
            win_h = self._last_win_h
        self._last_win_w = int(win_w)
        self._last_win_h = int(win_h)
        ws  = self.workspace
        lay = self._center_layout(win_w, win_h)
        cw = lay["cw"]
        tab_h = lay["tab_h"]
        prog_lw = lay["prog_lw"]
        prog_rw = lay["prog_rw"]
        view_x = lay["view_x"]
        view_w = lay["view_w"]

        # Left panel
        consumed, picked = self._left_panel.handle_event(ev, x_off=0)
        if picked is not None:
            kind = str(picked.get("kind", "solid")) if isinstance(picked, dict) else "solid"
            if kind == "blueprint":
                path = str(picked.get("path", "")) if isinstance(picked, dict) else ""
                if path:
                    ws.replay_blueprint(path)
                    self._mesh_dirty = True
            else:
                item_id = str(picked.get("id", "")) if isinstance(picked, dict) else ""
                if item_id:
                    ws.pick_catalog_item(item_id)
                    self._mesh_dirty = True
        if consumed:
            return True

        # Right panel
        if self._right_panel.handle_event(ev, x_off=win_w - self.RIGHT_W):
            self._mesh_dirty = True
            return True

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and hasattr(ev, "pos"):
            mx, my = ev.pos
            # Center tab bar
            if self.LEFT_W <= mx < self.LEFT_W + cw and 0 <= my < tab_h:
                lx = mx - self.LEFT_W
                for key, rect in self._center_tab_rects.items():
                    if rect.collidepoint(lx, my):
                        self._center_mode = key
                        if key == "programmatic":
                            self._refresh_programmatic_library()
                        self._mesh_dirty = True
                        return True

            if self._center_mode == "programmatic":
                # Center-left programmatic library panel
                if self.LEFT_W <= mx < self.LEFT_W + prog_lw and tab_h <= my < win_h:
                    lx = mx - self.LEFT_W
                    ly = my - tab_h
                    if self._left_panel._buttons.get("prog_refresh", pygame.Rect(0,0,0,0)).collidepoint(lx, ly):
                        self._refresh_programmatic_library()
                        self._mesh_dirty = True
                        return True
                    for bp_id, rect in self._prog_library_rects.items():
                        if rect.collidepoint(lx, ly):
                            self._prog_selected_id = bp_id
                            bp = self._selected_programmatic_blueprint()
                            if bp is not None and bp.id not in self._prog_values_by_id:
                                self._prog_values_by_id[bp.id] = _default_programmatic_knob_values(bp)
                            self._mesh_dirty = True
                            return True

                # Center-right knobs panel
                right_x = self.LEFT_W + cw - prog_rw
                if right_x <= mx < self.LEFT_W + cw and tab_h <= my < win_h:
                    lx = mx - right_x
                    ly = my - tab_h
                    bp = self._selected_programmatic_blueprint()
                    if bp is not None:
                        group_state = self._prog_group_open_by_bp.setdefault(bp.id, {})
                        for grp_name, rect in self._prog_group_rects.items():
                            if rect.collidepoint(lx, ly):
                                group_state[grp_name] = not bool(group_state.get(grp_name, True))
                                return True
                        for spec in bp.knobspec:
                            ctrl = self._prog_knob_rects.get(spec.name, {})
                            if spec.dtype in ("float", "int"):
                                if "minus" in ctrl and ctrl["minus"].collidepoint(lx, ly):
                                    self._adjust_programmatic_knob(spec, -1)
                                    return True
                                if "plus" in ctrl and ctrl["plus"].collidepoint(lx, ly):
                                    self._adjust_programmatic_knob(spec, +1)
                                    return True
                            else:
                                if "cycle" in ctrl and ctrl["cycle"].collidepoint(lx, ly):
                                    self._cycle_programmatic_knob(spec)
                                    return True
                        if self._prog_knob_rects.get("__print__", {}).get("click", pygame.Rect(0,0,0,0)).collidepoint(lx, ly):
                            self._print_selected_programmatic_blueprint()
                            return True

        # Workspace events (middle region)
        mx = getattr(ev, 'pos', (0,0))[0] if hasattr(ev, 'pos') else 0
        if not (view_x <= mx < view_x + view_w):
            return False

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self._dragging   = True
            self._last_mouse = ev.pos
            return True

        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if not self._dragging:
                return False
            # Short click → face pick
            lx, ly = ev.pos
            dx = lx - self._last_mouse[0]; dy = ly - self._last_mouse[1]
            if abs(dx) < 4 and abs(dy) < 4:
                self._pick_face(lx - view_x, ly, view_w, win_h)
            self._dragging = False
            return True

        if ev.type == pygame.MOUSEMOTION:
            if self._dragging:
                dx = ev.pos[0] - self._last_mouse[0]
                dy = ev.pos[1] - self._last_mouse[1]
                mods = pygame.key.get_mods()
                if mods & pygame.KMOD_SHIFT:
                    ws.adjust_gimbal(dx * 0.2, -dy * 0.2)
                else:
                    self.ws_cam.orbit(dx * 0.35, -dy * 0.28)
                self._last_mouse = ev.pos
                return True
            # Hover face detection
            lx, ly = ev.pos
            if view_x <= lx < view_x + view_w:
                fi = self._hit_face(lx - view_x, ly, view_w, win_h)
                prev = ws.hover_face
                ws.hover_over_face(fi)
                if fi != prev:
                    self._mesh_dirty = True
            return False

        if ev.type == pygame.MOUSEWHEEL:
            self.ws_cam.zoom(-ev.y * 0.08)
            return True

        return False

    def _hit_face(self, px, py, vw, vh) -> int:
        ro, rd = self.ws_cam.unproject(px, py, vw, vh)
        rd     = rd / max(np.linalg.norm(rd), 1e-12)
        fi, _  = self.workspace.active_mesh.ray_intersect_face(ro, rd)
        return fi

    def _pick_face(self, px, py, vw, vh):
        fi = self._hit_face(px, py, vw, vh)
        if fi >= 0:
            if self.workspace.snap_to_face(fi):
                self._mesh_dirty = True
