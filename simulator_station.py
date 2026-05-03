"""simulator_station.py
=======================
GL renderer for the simulator duty station.

Layout (in interact mode, full window):
  ┌──────────────┬──────────────────────────────────┬──────────────────┐
  │  LEFT PANEL  │   3-D GLASS ROOM VIEWPORT         │  RIGHT PANEL     │
  │ plugin picker│  bell-jar + sim box wireframe +   │ controls + evo   │
  │ (scrollable) │  orbiting camera                  │ stats            │
  └──────────────┴──────────────────────────────────┴──────────────────┘

Rendering approach
------------------
- 3D viewport:  GL 3.3 core, scissor-clipped sub-viewport.
                Glass/skirt geometry drawn with alpha-blending (transparent
                glass front, opaque skirt back).  Inner wireframe overlaid.
- 2D panels:    pygame.Surface → glTexImage2D → HUD quad shaders.

All geometry is rebuilt from GlassRoom when the sim world bounds change;
individual VBOs are cached and only re-uploaded on a ``dirty`` flag.
"""
from __future__ import annotations

import ctypes
import math
from typing import Optional

import numpy as np
import pygame

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_BACK, GL_BLEND, GL_COLOR_BUFFER_BIT,
        GL_CULL_FACE, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_FALSE,
        GL_FLOAT, GL_FRAGMENT_SHADER, GL_FRONT, GL_LINES,
        GL_ONE_MINUS_SRC_ALPHA, GL_RGBA, GL_SCISSOR_TEST,
        GL_SRC_ALPHA, GL_STATIC_DRAW, GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_LINEAR,
        GL_TRIANGLES, GL_TRUE, GL_UNSIGNED_BYTE, GL_VERTEX_SHADER,
        GL_TEXTURE0,
        glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray,
        glBlendFunc, glBufferData, glClear, glClearColor,
        glCullFace, glDeleteBuffers, glDeleteTextures,
        glDeleteVertexArrays, glDisable, glDrawArrays, glEnable,
        glEnableVertexAttribArray, glGenBuffers, glGenTextures,
        glGenVertexArrays, glGetUniformLocation, glLineWidth,
        glScissor, glTexImage2D, glTexParameteri, glUniform1f,
        glUniform1i, glUniform2f, glUniform3f, glUniform4f,
        glUniformMatrix4fv, glUseProgram, glVertexAttribPointer,
        glViewport,
    )
    from OpenGL.GL import shaders as _gl_shaders
    _HAS_GL = True
except ImportError:
    _HAS_GL = False

from glass_room import GlassRoom
from simulator_workspace import SimulatorWorkspace, SimulatorMode


# ─────────────────────────────────────────────────────────────────────────────
# GLSL shaders
# ─────────────────────────────────────────────────────────────────────────────

# Phong with per-fragment alpha (for translucent glass)
_GLASS_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
uniform mat4 uMVP;
uniform mat4 uMV;
out vec3 vNormV;
out vec3 vPosV;
void main() {
    vec4 posV = uMV * vec4(aPos, 1.0);
    vPosV      = posV.xyz;
    vNormV     = mat3(uMV) * aNorm;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_GLASS_FS = """
#version 330 core
in  vec3 vNormV;
in  vec3 vPosV;
out vec4 FragColor;
uniform vec4  uColor;       // RGBA — alpha drives glass transparency
uniform vec3  uLightV;
uniform float uAmbient;
uniform float uSpecStr;
uniform float uShininess;
void main() {
    vec3  N = normalize(gl_FrontFacing ? vNormV : -vNormV);
    vec3  L = normalize(uLightV);
    vec3  V = normalize(-vPosV);
    float d = max(dot(N, L), 0.0);
    float s = pow(max(dot(normalize(L + V), N), 0.0), uShininess);
    // Fresnel-ish rim for glass edges
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3  col = uColor.rgb * (uAmbient + 0.75 * d)
              + vec3(0.85, 0.92, 1.0) * uSpecStr * s
              + uColor.rgb * rim * 0.18;
    float a = uColor.a + rim * (1.0 - uColor.a) * 0.35;
    FragColor = vec4(col, clamp(a, 0.0, 1.0));
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

# HUD panel texture quad in pixel-space
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
# Orbit camera for the glass-room viewport
# ─────────────────────────────────────────────────────────────────────────────

class _SimCam:
    def __init__(self, az=35.0, elev=22.0, dist=2.8, fov_deg=50.0):
        self.az    = float(az)
        self.elev  = float(elev)
        self.dist  = float(dist)
        self.fov   = math.radians(fov_deg)
        self.target = np.zeros(3, np.float64)

    def orbit(self, daz, delev):
        self.az   = (self.az + daz) % 360.0
        self.elev = float(np.clip(self.elev + delev, -80, 80))

    def zoom(self, d):
        self.dist = float(np.clip(self.dist + d, 0.2, 12.0))

    @property
    def eye(self):
        az = math.radians(self.az); el = math.radians(self.elev)
        return self.target + self.dist * np.array([
            math.cos(el)*math.cos(az),
            math.cos(el)*math.sin(az),
            math.sin(el),
        ], np.float64)

    def _view(self) -> np.ndarray:
        eye = self.eye
        fwd = self.target - eye; fwd /= max(np.linalg.norm(fwd), 1e-12)
        up  = np.array([0., 0., 1.])
        if abs(np.dot(fwd, up)) > 0.97:
            up = np.array([0., 1., 0.])
        right = np.cross(fwd, up); right /= max(np.linalg.norm(right), 1e-12)
        up2   = np.cross(right, fwd)
        V = np.eye(4, dtype=np.float64)
        V[0,:3] =  right; V[0,3] = -np.dot(right, eye)
        V[1,:3] =  up2;   V[1,3] = -np.dot(up2,   eye)
        V[2,:3] = -fwd;   V[2,3] =  np.dot(fwd,   eye)
        return V

    def mvp(self, aspect) -> np.ndarray:
        f = 1.0 / math.tan(self.fov / 2.0)
        near, far = 0.01, 40.0
        P = np.zeros((4, 4), np.float64)
        P[0,0]=f/aspect; P[1,1]=f
        P[2,2]=-(far+near)/(far-near); P[2,3]=-2*far*near/(far-near); P[3,2]=-1.0
        return (P @ self._view()).astype(np.float32)

    def mv(self) -> np.ndarray:
        return self._view().astype(np.float32)

    def light_view(self) -> np.ndarray:
        """Key-light direction in view space (top-right-front)."""
        ld = np.array([0.6, 0.8, 1.0], np.float64)
        ld /= np.linalg.norm(ld)
        R3 = self._view()[:3, :3]
        return (R3 @ ld).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Left panel — plugin picker
# ─────────────────────────────────────────────────────────────────────────────

class _PluginPickerPanel:
    ITEM_H  = 60
    ICON_SZ = 44
    PAD     = 8

    def __init__(self, palette: list[dict], selected_id: Optional[str] = None):
        pygame.font.init()
        self._font  = pygame.font.SysFont("monospace", 13)
        self._font_s = pygame.font.SysFont("monospace", 11)
        self.palette    = palette
        self.selected   = selected_id
        self._hover     = -1
        self._scroll    = 0
        self._ctrl: dict = {}

    def render(self, w: int, h: int) -> pygame.Surface:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((16, 18, 26, 245))

        # Header bar
        pygame.draw.rect(surf, (24, 30, 44), pygame.Rect(0, 0, w, 20))
        hdr = self._font.render("  PLUGINS", True, (120, 180, 240))
        surf.blit(hdr, (self.PAD, 2))
        y = 22

        self._ctrl = {}
        for idx, entry in enumerate(self.palette):
            iy = y + idx * self.ITEM_H - self._scroll
            if iy + self.ITEM_H < 0 or iy > h:
                continue
            sel = (entry["id"] == self.selected)
            hov = (idx == self._hover)
            bg  = (35, 50, 85) if sel else ((26, 30, 44) if hov else (18, 21, 32))
            pygame.draw.rect(surf, bg, pygame.Rect(0, iy, w, self.ITEM_H))
            if sel:
                pygame.draw.rect(surf, (50, 120, 230), pygame.Rect(0, iy, 3, self.ITEM_H))

            # Colour icon
            c  = entry.get("color_rgb", [0.5, 0.5, 0.5])
            ic = (int(c[0]*220), int(c[1]*220), int(c[2]*220))
            ir = pygame.Rect(8, iy + 8, self.ICON_SZ, self.ICON_SZ)
            pygame.draw.rect(surf, ic, ir)
            pygame.draw.rect(surf, (160, 170, 190), ir, 1)

            # Text
            lbl = self._font.render(entry.get("label", entry["id"]), True, (210, 220, 235))
            sub = self._font_s.render(entry.get("subtitle", ""), True, (100, 115, 135))
            tx  = 8 + self.ICON_SZ + 6
            surf.blit(lbl, (tx, iy + 12))
            surf.blit(sub, (tx, iy + 12 + lbl.get_height() + 2))

            self._ctrl[f"plug_{entry['id']}"] = pygame.Rect(0, iy, w, self.ITEM_H)

        pygame.draw.line(surf, (35, 40, 55), (0, h-1), (w, h-1))
        return surf

    def handle_event(self, ev, x_off=0, y_off=0) -> tuple[bool, Optional[str]]:
        """Returns (consumed, plugin_id_or_None)."""
        if ev.type == pygame.MOUSEWHEEL:
            mx, _ = pygame.mouse.get_pos()
            if x_off <= mx < x_off + 210:
                self._scroll = max(0, self._scroll - ev.y * 24)
                return True, None
        if ev.type == pygame.MOUSEMOTION:
            mx, my = ev.pos; lx, ly = mx - x_off, my - y_off
            self._hover = -1
            for idx, entry in enumerate(self.palette):
                iy = 22 + idx * self.ITEM_H - self._scroll
                if pygame.Rect(0, iy, 210, self.ITEM_H).collidepoint(lx, ly):
                    self._hover = idx; break
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos; lx, ly = mx - x_off, my - y_off
            if 0 <= lx < 210:
                for entry in self.palette:
                    r = self._ctrl.get(f"plug_{entry['id']}")
                    if r and r.collidepoint(lx, ly):
                        self.selected = entry["id"]
                        return True, entry["id"]
        return False, None


# ─────────────────────────────────────────────────────────────────────────────
# Right panel — controls + evolution stats
# ─────────────────────────────────────────────────────────────────────────────

class _SimControlPanel:
    PAD   = 8
    ROW_H = 24

    def __init__(self):
        pygame.font.init()
        self._font   = pygame.font.SysFont("monospace", 13)
        self._font_s = pygame.font.SysFont("monospace", 11)
        self._ctrl: dict = {}
        # State references (set by SimulatorStation after workspace is built)
        self.mode_str    = "idle"
        self.gen         = 0
        self.n_gen       = 32
        self.mean_fit    = 0.0
        self.best_fit    = 0.0
        self.pop_size    = 8
        self.plugin_name = "—"
        self.param_specs : list[dict] = []
        self.current_params: dict     = {}

    def render(self, w: int, h: int) -> pygame.Surface:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((16, 18, 26, 245))

        pygame.draw.rect(surf, (24, 30, 44), pygame.Rect(0, 0, w, 20))
        hdr = self._font.render("  SIMULATOR", True, (120, 180, 240))
        surf.blit(hdr, (self.PAD, 2))

        self._ctrl = {}
        y = 24

        # ── Transport buttons ────────────────────────────────────────────────
        y = self._section(surf, y, "TRANSPORT", w)
        transport_btns = [("btn_start", "▶ RUN"), ("btn_pause", "⏸ PAUSE"), ("btn_stop", "■ STOP")]
        bw = (w - 2*self.PAD - 4) // 3
        x  = self.PAD
        for key, lbl in transport_btns:
            act = (key == "btn_start"  and self.mode_str == "running") or \
                  (key == "btn_pause"  and self.mode_str == "paused") or \
                  (key == "btn_stop"   and self.mode_str == "idle")
            bg  = (30, 80, 170) if act else (38, 44, 58)
            r   = pygame.Rect(x, y, bw, self.ROW_H)
            pygame.draw.rect(surf, bg, r)
            pygame.draw.rect(surf, (80, 90, 110), r, 1)
            t   = self._font.render(lbl, True, (210, 220, 235))
            surf.blit(t, (x + (bw - t.get_width())//2, y + (self.ROW_H - t.get_height())//2))
            self._ctrl[key] = r
            x += bw + 2
        y += self.ROW_H + 4

        # Evolve toggle
        act = (self.mode_str == "evolving")
        y = self._button(surf, y, "btn_evolve", "🔬 EVOLVE",
                         w - 2*self.PAD, active=act, x=self.PAD)

        # ── Active plugin ─────────────────────────────────────────────────────
        y = self._section(surf, y + 2, "ACTIVE PLUGIN", w)
        y = self._label(surf, y, f"  {self.plugin_name}")

        # ── Current params ────────────────────────────────────────────────────
        if self.param_specs:
            y = self._section(surf, y + 2, "PARAMS", w)
            for spec in self.param_specs[:6]:   # cap at 6 rows
                name = spec["name"]
                val  = self.current_params.get(name, 0.0)
                lo   = float(spec.get("min", 0.0))
                hi   = float(spec.get("max", 1.0))
                frac = (val - lo) / max(hi - lo, 1e-9)
                # Bar bg
                bar_w = w - 2*self.PAD - 2
                pygame.draw.rect(surf, (28, 32, 44), pygame.Rect(self.PAD, y, bar_w, 14))
                pygame.draw.rect(surf, (30, 80, 160), pygame.Rect(self.PAD, y,
                                 int(bar_w * frac), 14))
                lbl_t = self._font_s.render(
                    f"{name}  {val:.3f}", True, (160, 175, 200))
                surf.blit(lbl_t, (self.PAD + 3, y))
                y += 16

        # ── Evolution stats ───────────────────────────────────────────────────
        y = self._section(surf, y + 4, "EVOLUTION", w)
        evo_active = (self.mode_str == "evolving")
        bar_w = w - 2*self.PAD - 2
        # Generation progress bar
        prog = self.gen / max(self.n_gen, 1)
        pygame.draw.rect(surf, (28, 32, 44), pygame.Rect(self.PAD, y, bar_w, 10))
        pygame.draw.rect(surf, (0, 140, 80), pygame.Rect(self.PAD, y,
                         int(bar_w * prog), 10))
        txt = f"Gen {self.gen}/{self.n_gen}  pop {self.pop_size}"
        y = self._label(surf, y + 12, txt)
        y = self._label(surf, y, f"  mean fit: {self.mean_fit:.4f}")
        y = self._label(surf, y, f"  best fit: {self.best_fit:.4f}")

        pygame.draw.line(surf, (35, 40, 55), (0, h-1), (w, h-1))
        return surf

    def handle_event(self, ev, x_off=0, y_off=0) -> tuple[bool, str]:
        """Returns (consumed, action_key_or_empty_string)."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos; lx, ly = mx - x_off, my - y_off
            if 0 <= lx:
                for key, r in self._ctrl.items():
                    if r.collidepoint(lx, ly):
                        return True, key
        return False, ""

    def _section(self, surf, y, label, w) -> int:
        pygame.draw.rect(surf, (28, 33, 47), pygame.Rect(0, y, w, 18))
        t = self._font_s.render(label, True, (90, 120, 170))
        surf.blit(t, (self.PAD, y + 2))
        return y + 20

    def _label(self, surf, y, text, color=(130, 145, 160)) -> int:
        t = self._font_s.render(text, True, color)
        surf.blit(t, (self.PAD, y + 1))
        return y + 16

    def _button(self, surf, y, key, label, w, active=False, x=0) -> int:
        bh = self.ROW_H
        r  = pygame.Rect(x, y, w, bh)
        bg = (30, 80, 170) if active else (38, 44, 58)
        pygame.draw.rect(surf, bg, r)
        pygame.draw.rect(surf, (80, 90, 110), r, 1)
        t  = self._font.render(label, True, (210, 220, 235))
        surf.blit(t, (x + (w - t.get_width())//2, y + (bh - t.get_height())//2))
        self._ctrl[key] = r
        return y + bh + 3


# ─────────────────────────────────────────────────────────────────────────────
# GPU buffer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_vao_pos_norm(data: np.ndarray) -> tuple[int, int, int]:
    """Upload interleaved (pos3|norm3) float32 array.
    Returns (vao, vbo, n_verts)."""
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    flat = np.ascontiguousarray(data, np.float32).flatten()
    glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
    stride = 24  # 6 floats × 4 bytes
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glBindVertexArray(0)
    return vao, vbo, len(data)


def _make_vao_pos3(data: np.ndarray) -> tuple[int, int, int]:
    """Upload position-only float32 (-1,3) array.
    Returns (vao, vbo, n_verts)."""
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    flat = np.ascontiguousarray(data, np.float32).flatten()
    glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glBindVertexArray(0)
    return vao, vbo, len(data)


def _make_hud_tex(surf: pygame.Surface) -> tuple[int, int]:
    """Upload a pygame.Surface as an RGBA GL texture.
    Returns (tex_id, tex_id) — second is the same id (for symmetry)."""
    raw = pygame.image.tobytes(surf, "RGBA", True)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 surf.get_width(), surf.get_height(), 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, raw)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex, tex


def _quad_verts(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Return float32 array of 6 vertices (2 tris) for a screen-space quad."""
    x0, y0, x1, y1 = float(x), float(y), float(x + w), float(y + h)
    return np.array([
        x0, y0, 0., 1.,
        x1, y0, 1., 1.,
        x1, y1, 1., 0.,
        x0, y0, 0., 1.,
        x1, y1, 1., 0.,
        x0, y1, 0., 0.,
    ], np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SimulatorStation
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorStation:
    """GL renderer for the simulator duty station.

    Parameters
    ----------
    palette_cfg:
        Parsed simulator_palette.yaml content.
    coevo_cfg:
        Parsed coevolution.yaml content.
    glass_cfg:
        Parsed glass_room.yaml content.
    wb_min, wb_max:
        World-space bounding box for the simulation region (float32/float64,
        shape (3,)).  Must be provided before ``init_gl`` is called.
    win_w, win_h:
        Window resolution at interact time.
    left_panel_w, right_panel_w:
        Panel widths in pixels (from station.yaml layout).
    """

    def __init__(
        self,
        palette_cfg:     dict,
        coevo_cfg:       dict,
        glass_cfg:       dict,
        wb_min:          np.ndarray,
        wb_max:          np.ndarray,
        win_w:           int = 1280,
        win_h:           int = 720,
        left_panel_w:    int = 210,
        right_panel_w:   int = 240,
    ) -> None:
        self._palette_cfg   = palette_cfg
        self._coevo_cfg     = coevo_cfg
        self._glass_cfg     = glass_cfg
        self.wb_min         = np.asarray(wb_min, np.float64)
        self.wb_max         = np.asarray(wb_max, np.float64)
        self.win_w          = win_w
        self.win_h          = win_h
        self.left_panel_w   = left_panel_w
        self.right_panel_w  = right_panel_w

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.workspace = SimulatorWorkspace(
            palette_cfg=palette_cfg,
            coevo_cfg=coevo_cfg,
        )
        self.room = GlassRoom(glass_cfg)
        self._cam = _SimCam()

        # ── UI panels ─────────────────────────────────────────────────────────
        palette = palette_cfg.get("plugins", [])
        self._left_panel  = _PluginPickerPanel(palette)
        self._right_panel = _SimControlPanel()

        # ── GL resources (initialised in init_gl) ─────────────────────────────
        self._gl_ready   = False
        self._prog_glass = None
        self._prog_line  = None
        self._prog_hud   = None

        # Geometry VAOs
        self._jar_vao  : Optional[int] = None
        self._jar_vbo  : Optional[int] = None
        self._jar_n    : int = 0
        self._skirt_vao: Optional[int] = None
        self._skirt_vbo: Optional[int] = None
        self._skirt_n  : int = 0
        self._wire_vao : Optional[int] = None
        self._wire_vbo : Optional[int] = None
        self._wire_n   : int = 0

        # Panel HUD VAO + texture
        self._hud_vao : Optional[int] = None
        self._hud_vbo : Optional[int] = None
        self._left_tex : Optional[int] = None
        self._right_tex: Optional[int] = None
        self._panels_dirty = True

        # Mouse drag state
        self._drag: Optional[tuple[int, int]] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def init_gl(self) -> None:
        """Compile shaders and upload initial geometry.  Call once after
        the GL context is current."""
        if not _HAS_GL:
            return

        self._prog_glass = _gl_shaders.compileProgram(
            _gl_shaders.compileShader(_GLASS_VS, GL_VERTEX_SHADER),
            _gl_shaders.compileShader(_GLASS_FS, GL_FRAGMENT_SHADER),
        )
        self._prog_line = _gl_shaders.compileProgram(
            _gl_shaders.compileShader(_LINE_VS, GL_VERTEX_SHADER),
            _gl_shaders.compileShader(_LINE_FS, GL_FRAGMENT_SHADER),
        )
        self._prog_hud = _gl_shaders.compileProgram(
            _gl_shaders.compileShader(_HUD_VS, GL_VERTEX_SHADER),
            _gl_shaders.compileShader(_HUD_FS, GL_FRAGMENT_SHADER),
        )

        # HUD quad VAO
        self._hud_vao = glGenVertexArrays(1)
        self._hud_vbo = glGenBuffers(1)
        glBindVertexArray(self._hud_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._hud_vbo)
        glBufferData(GL_ARRAY_BUFFER, 96, None, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)

        self._rebuild_geometry()
        self._gl_ready = True

    def destroy_gl(self) -> None:
        """Release all GL resources."""
        if not _HAS_GL or not self._gl_ready:
            return
        for vao_attr in ("_jar_vao", "_skirt_vao", "_wire_vao", "_hud_vao"):
            v = getattr(self, vao_attr, None)
            if v is not None:
                glDeleteVertexArrays(1, [v])
        for vbo_attr in ("_jar_vbo", "_skirt_vbo", "_wire_vbo", "_hud_vbo"):
            v = getattr(self, vbo_attr, None)
            if v is not None:
                glDeleteBuffers(1, [v])
        for tex_attr in ("_left_tex", "_right_tex"):
            v = getattr(self, tex_attr, None)
            if v is not None:
                glDeleteTextures(1, [v])
        self._gl_ready = False

    # ── Per-frame render ──────────────────────────────────────────────────────

    def render(self, win_w: int, win_h: int) -> None:
        """Render the full interact-mode display into the current GL context.

        Parameters
        ----------
        win_w, win_h:
            Current window pixel dimensions.
        """
        if not _HAS_GL or not self._gl_ready:
            return

        self.win_w, self.win_h = win_w, win_h
        lw = self.left_panel_w
        rw = self.right_panel_w
        vp_x = lw
        vp_w = win_w - lw - rw
        vp_h = win_h

        # ── 3D viewport ───────────────────────────────────────────────────────
        glEnable(GL_SCISSOR_TEST)
        glScissor(vp_x, 0, vp_w, vp_h)
        glViewport(vp_x, 0, vp_w, vp_h)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.04, 0.06, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        aspect = vp_w / max(1, vp_h)
        MVP    = self._cam.mvp(aspect)
        MV     = self._cam.mv()
        Lv     = self._cam.light_view()

        self._draw_skirt(MVP, MV, Lv)
        self._draw_jar(MVP, MV, Lv)
        self._draw_wireframe(MVP)

        # ── 2D panel HUDs ─────────────────────────────────────────────────────
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_SCISSOR_TEST)
        glViewport(0, 0, win_w, win_h)

        self._sync_right_panel()
        if self._panels_dirty:
            self._upload_panels(win_w, win_h, lw, rw)
            self._panels_dirty = False

        self._draw_hud_panel(0,         0, lw,  win_h, self._left_tex,  win_w, win_h)
        self._draw_hud_panel(win_w - rw, 0, rw, win_h, self._right_tex, win_w, win_h)

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev) -> bool:
        """Route a pygame event.  Returns True if consumed."""
        lw = self.left_panel_w
        rw = self.right_panel_w
        vp_x = lw
        vp_w = self.win_w - lw - rw

        # Left panel
        consumed, plugin_id = self._left_panel.handle_event(ev, x_off=0)
        if plugin_id:
            ok = self.workspace.select_plugin(plugin_id)
            if ok:
                self._right_panel.plugin_name = plugin_id
                self._right_panel.param_specs = self.workspace.param_specs
                self._right_panel.current_params = self.workspace.current_params
                self._panels_dirty = True
        if consumed:
            return True

        # Right panel
        consumed, action = self._right_panel.handle_event(ev, x_off=self.win_w - rw)
        if consumed:
            self._dispatch_action(action)
            self._panels_dirty = True
            return True

        # 3D viewport — orbit drag
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, _ = ev.pos
            if vp_x <= mx < vp_x + vp_w:
                self._drag = ev.pos
                return True
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._drag = None
        if ev.type == pygame.MOUSEMOTION and self._drag is not None:
            dx = ev.pos[0] - self._drag[0]
            dy = ev.pos[1] - self._drag[1]
            self._cam.orbit(dx * 0.4, dy * 0.4)
            self._drag = ev.pos
            return True
        if ev.type == pygame.MOUSEWHEEL:
            mx, _ = pygame.mouse.get_pos()
            if vp_x <= mx < vp_x + vp_w:
                self._cam.zoom(-ev.y * 0.12)
                return True

        return False

    def mark_panels_dirty(self) -> None:
        self._panels_dirty = True

    # ── Private: GL draw calls ────────────────────────────────────────────────

    def _draw_skirt(self, MVP, MV, Lv) -> None:
        if self._skirt_n == 0 or self._prog_glass is None:
            return
        glUseProgram(self._prog_glass)
        self._set_glass_uniforms(MVP, MV, Lv, self.room.skirt_color)
        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE); glCullFace(GL_BACK)
        glBindVertexArray(self._skirt_vao)
        glDrawArrays(GL_TRIANGLES, 0, self._skirt_n)
        glBindVertexArray(0)
        glDisable(GL_CULL_FACE)

    def _draw_jar(self, MVP, MV, Lv) -> None:
        if self._jar_n == 0 or self._prog_glass is None:
            return
        glUseProgram(self._prog_glass)
        self._set_glass_uniforms(MVP, MV, Lv, self.room.glass_color)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)
        glBindVertexArray(self._jar_vao)
        glDrawArrays(GL_TRIANGLES, 0, self._jar_n)
        glBindVertexArray(0)
        glDisable(GL_BLEND)

    def _draw_wireframe(self, MVP) -> None:
        if (not self.room.show_wireframe) or self._wire_n == 0:
            return
        if self._prog_line is None:
            return
        glUseProgram(self._prog_line)
        loc = glGetUniformLocation(self._prog_line, "uMVP")
        glUniformMatrix4fv(loc, 1, GL_FALSE, MVP)
        loc_col = glGetUniformLocation(self._prog_line, "uColor")
        wc = self.room.wireframe_color
        glUniform4f(loc_col, *wc)
        glLineWidth(self.room.wireframe_width)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindVertexArray(self._wire_vao)
        glDrawArrays(GL_LINES, 0, self._wire_n)
        glBindVertexArray(0)
        glDisable(GL_BLEND)
        glLineWidth(1.0)

    def _set_glass_uniforms(self, MVP, MV, Lv, color_rgba) -> None:
        prog = self._prog_glass
        def u(name): return glGetUniformLocation(prog, name)
        glUniformMatrix4fv(u("uMVP"), 1, GL_FALSE, MVP)
        glUniformMatrix4fv(u("uMV"),  1, GL_FALSE, MV)
        glUniform3f(u("uLightV"), *Lv)
        glUniform4f(u("uColor"),  *color_rgba)
        glUniform1f(u("uAmbient"),   self.room.ambient)
        glUniform1f(u("uSpecStr"),   self.room.spec_strength)
        glUniform1f(u("uShininess"), self.room.shininess)

    # ── Private: HUD panels ───────────────────────────────────────────────────

    def _upload_panels(self, win_w, win_h, lw, rw) -> None:
        lsurf = self._left_panel.render(lw, win_h)
        rsurf = self._right_panel.render(rw, win_h)

        if self._left_tex is not None:
            glDeleteTextures(1, [self._left_tex])
        if self._right_tex is not None:
            glDeleteTextures(1, [self._right_tex])
        self._left_tex,  _ = _make_hud_tex(lsurf)
        self._right_tex, _ = _make_hud_tex(rsurf)

    def _draw_hud_panel(self, x, y, w, h, tex_id, win_w, win_h) -> None:
        if tex_id is None or self._prog_hud is None:
            return
        verts = _quad_verts(x, y, w, h)
        glBindBuffer(GL_ARRAY_BUFFER, self._hud_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

        glUseProgram(self._prog_hud)
        loc = glGetUniformLocation(self._prog_hud, "uRes")
        glUniform2f(loc, float(win_w), float(win_h))
        loc_tex = glGetUniformLocation(self._prog_hud, "uTex")
        glUniform1i(loc_tex, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindVertexArray(self._hud_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glDisable(GL_BLEND)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _sync_right_panel(self) -> None:
        """Copy current workspace stats into the right panel for rendering."""
        ws = self.workspace
        rp = self._right_panel
        rp.mode_str = ws.mode.value
        stats = ws.population_stats
        rp.gen      = int(stats.get("generation", 0))
        rp.n_gen    = self._coevo_cfg.get("n_generations", 32)
        rp.mean_fit = float(stats.get("mean_fitness", 0.0))
        rp.best_fit = float(stats.get("best_fitness", 0.0))
        rp.pop_size = int(stats.get("pop_size", 0))
        rp.current_params = dict(ws.current_params)

    def _dispatch_action(self, action: str) -> None:
        ws = self.workspace
        if action == "btn_start":
            ws.start()
        elif action == "btn_pause":
            if ws.mode == SimulatorMode.RUNNING:
                ws.pause()
            elif ws.mode == SimulatorMode.PAUSED:
                ws.resume()
        elif action == "btn_stop":
            ws.stop()
        elif action == "btn_evolve":
            if ws.mode == SimulatorMode.EVOLVING:
                ws.stop_evolving()
            else:
                ws.start_evolving()

    # ── Private: geometry ─────────────────────────────────────────────────────

    def _rebuild_geometry(self) -> None:
        """(Re-)upload glass room geometry into GPU buffers."""
        if not _HAS_GL:
            return

        # Clean up old buffers
        for vao_attr, vbo_attr in (
            ("_jar_vao",   "_jar_vbo"),
            ("_skirt_vao", "_skirt_vbo"),
            ("_wire_vao",  "_wire_vbo"),
        ):
            v = getattr(self, vao_attr, None)
            if v is not None: glDeleteVertexArrays(1, [v])
            v = getattr(self, vbo_attr, None)
            if v is not None: glDeleteBuffers(1, [v])

        jar, skirt, wire = self.room.build_all(self.wb_min, self.wb_max)

        if jar is not None and len(jar):
            self._jar_vao, self._jar_vbo, self._jar_n = _make_vao_pos_norm(jar)
        else:
            self._jar_vao = self._jar_vbo = None; self._jar_n = 0

        if skirt is not None and len(skirt):
            self._skirt_vao, self._skirt_vbo, self._skirt_n = _make_vao_pos_norm(skirt)
        else:
            self._skirt_vao = self._skirt_vbo = None; self._skirt_n = 0

        if wire is not None and len(wire):
            self._wire_vao, self._wire_vbo, self._wire_n = _make_vao_pos3(wire)
        else:
            self._wire_vao = self._wire_vbo = None; self._wire_n = 0

        # Centre the camera on the sim box
        centre = (self.wb_min + self.wb_max) * 0.5
        self._cam.target = centre
        span   = float(np.linalg.norm(self.wb_max - self.wb_min))
        self._cam.dist   = span * 1.4

    def update_bounds(self, wb_min: np.ndarray, wb_max: np.ndarray) -> None:
        """Call when the sim world bounding box changes to re-upload geometry."""
        self.wb_min = np.asarray(wb_min, np.float64)
        self.wb_max = np.asarray(wb_max, np.float64)
        if self._gl_ready:
            self._rebuild_geometry()
