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
from typing import Optional

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

    def __init__(self, palette: list, selected_id: Optional[str] = None):
        super().__init__()
        self.palette    = palette       # list of dict from palette.yaml
        self.selected   = selected_id
        self._hover     = -1
        self.PANEL_W    = 210

    def render(self, w: int, h: int) -> pygame.Surface:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((18, 20, 28, 240))

        # Header
        y = self._section(surf, 0, "  COMPONENTS", (28, 32, 44))

        # Items
        self._ctrl = {}
        for idx, entry in enumerate(self.palette):
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
            self._ctrl[f"solid_{entry['id']}"] = pygame.Rect(0, iy, w, self.ITEM_H)

        # Divider at bottom
        pygame.draw.line(surf, (40, 44, 58), (0, h-1), (w, h-1))
        return surf

    def handle_event(self, ev, x_off=0, y_off=0) -> tuple[bool, Optional[str]]:
        """Returns (consumed, selected_id_or_None)."""
        if ev.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if x_off <= mx < x_off + self.PANEL_W:
                self._scroll = max(0, self._scroll - ev.y * 24)
                return True, None
        if ev.type == pygame.MOUSEMOTION:
            mx, my = ev.pos
            lx, ly = mx - x_off, my - y_off
            self._hover = -1
            for idx, entry in enumerate(self.palette):
                iy = 20 + idx * self.ITEM_H - self._scroll
                r  = pygame.Rect(0, iy, self.PANEL_W, self.ITEM_H)
                if r.collidepoint(lx, ly):
                    self._hover = idx
                    break
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            lx, ly = mx - x_off, my - y_off
            if 0 <= lx < self.PANEL_W:
                for entry in self.palette:
                    r = self._ctrl.get(f"solid_{entry['id']}")
                    if r and r.collidepoint(lx, ly):
                        self.selected = entry['id']
                        return True, entry['id']
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
        if mode == WorkspaceMode.IDLE and ws.built_mesh is not None:
            y = self._button(surf, y, "review", "Review Build")
        if mode == WorkspaceMode.REVIEW:
            y = self._button(surf, y, "store",  "[+] Store to Inventory")
            y = self._button(surf, y, "clear",  "[x] Clear Build")
            y = self._button(surf, y, "exit_review", "< Back")

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
        if _hit("review"):       ws.enter_review(); return True
        if _hit("store"):        ws.store_to_inventory(); return True
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
        self._palette  = palette
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

        self.ws_cam = _WsCam(ws_cfg.get("camera", {}))

        self._left_panel  = _SolidPickerPanel(palette)
        self._right_panel = _SymmetryPanel(self.workspace)

        # GL state
        self._gl_ready = False
        self._prog_mesh = self._prog_line = self._prog_hud = None
        self._mesh_vao = self._mesh_vbo = self._mesh_n = None
        self._edge_vao = self._edge_vbo = self._edge_n = None
        self._hud_vao  = self._hud_vbo  = None
        self._ltex = self._rtex = 0   # panel GL textures
        self._mesh_dirty = True
        self._dragging   = False
        self._last_mouse = (0, 0)

    @classmethod
    def from_yaml(cls, ws_path: str, palette_path: str) -> "FabricatorStation":
        import yaml as _y
        with open(ws_path,      encoding='utf-8') as f: ws  = _y.safe_load(f) or {}
        with open(palette_path, encoding='utf-8') as f: pal = _y.safe_load(f) or {}
        return cls(ws, pal.get("solids", []))

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

    def tick(self, dt: float):
        self.ws_cam.tick()
        if self._mesh_dirty and self._gl_ready:
            self._upload_mesh()

    # ── Main draw ─────────────────────────────────────────────────────────────

    def draw(self, win_w: int, win_h: int):
        if not self._gl_ready: return
        cw = win_w - self.LEFT_W - self.RIGHT_W

        # ── 3D workspace viewport ─────────────────────────────────────────────
        glEnable(GL_SCISSOR_TEST)
        glScissor(self.LEFT_W, 0, cw, win_h)
        bg = self._bg
        glClearColor(bg[0], bg[1], bg[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_SCISSOR_TEST)
        glViewport(self.LEFT_W, 0, cw, win_h)

        aspect = cw / max(1, win_h)
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

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev, win_w: int, win_h: int) -> bool:
        ws  = self.workspace
        cw  = win_w - self.LEFT_W - self.RIGHT_W

        # Left panel
        consumed, picked_id = self._left_panel.handle_event(ev, x_off=0)
        if picked_id is not None:
            ws.pick_solid(picked_id)
            self._mesh_dirty = True
        if consumed:
            return True

        # Right panel
        if self._right_panel.handle_event(ev, x_off=win_w - self.RIGHT_W):
            self._mesh_dirty = True
            return True

        # Workspace events (middle region)
        mx = getattr(ev, 'pos', (0,0))[0] if hasattr(ev, 'pos') else 0
        if not (self.LEFT_W <= mx < self.LEFT_W + cw):
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
                self._pick_face(lx - self.LEFT_W, ly, cw, win_h)
            self._dragging = False
            return True

        if ev.type == pygame.MOUSEMOTION:
            if self._dragging:
                dx = ev.pos[0] - self._last_mouse[0]
                dy = ev.pos[1] - self._last_mouse[1]
                self.ws_cam.orbit(dx * 0.35, -dy * 0.28)
                self._last_mouse = ev.pos
                return True
            # Hover face detection
            lx, ly = ev.pos
            if self.LEFT_W <= lx < self.LEFT_W + cw:
                fi = self._hit_face(lx - self.LEFT_W, ly, cw, win_h)
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
