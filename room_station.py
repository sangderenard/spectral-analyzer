"""room_station.py
==================
RoomStation — the master GL renderer that owns the room environment.

Layout (interact / HUD mode)
─────────────────────────────────────────────────────────────────────
  ┌───────────┬─────────────────────────────────┬──────────────────┐
  │  SCENE    │         MINIMAP                 │   PROPERTIES     │
  │  TREE     │   top-down 2-D floor plan       │   selected obj   │
  │  (scroll) │   rooms / enclosures / lights   │   fields + edit  │
  └───────────┴─────────────────────────────────┴──────────────────┘

Room geometry (walls / floor / ceiling / grid) is rendered by
``render_room(MVP, MV, light_v)`` in the main 3-D pass.

The HUD overlay (minimap + panels) is rendered by
``render_hud(win_w, win_h)`` and is only visible when the player
is in INTERACT mode at the room console.

Dependencies
────────────
* ``room_workspace.py``
* ``room_geometry.py``
* ``enclosure_geometry.py``
* ``placed_object.py``
* OpenGL 3.3 core profile
* pygame
"""
from __future__ import annotations

import ctypes
import math
from typing import Optional, List

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

from room_workspace  import RoomWorkspace
from room_geometry   import build_room_mesh, build_room_floor_grid
from enclosure_geometry import build_enclosure_mesh, build_enclosure_wireframe
from placed_object   import (
    PlacedLight, PlacedEnclosure, PlacedDutyStation, PlacedPortalFrame,
)


# ─────────────────────────────────────────────────────────────────────────────
# GLSL shaders
# ─────────────────────────────────────────────────────────────────────────────

# ── Room surface (Phong, opaque) ───────────────────────────────────────────────
_ROOM_VS = """
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

_ROOM_FS = """
#version 330 core
in  vec3 vNormV;
in  vec3 vPosV;
out vec4 fragColor;
uniform vec4  uColor;        // (r,g,b,1)
uniform vec3  uLightV;       // view-space light direction (normalised)
uniform float uAmbient;
uniform float uSpec;
uniform float uShin;
void main() {
    vec3 N = normalize(vNormV);
    vec3 L = normalize(uLightV);
    float d = max(dot(N, L), 0.0);
    vec3 V = normalize(-vPosV);
    vec3 H = normalize(L + V);
    float s = (d > 0.0) ? pow(max(dot(N, H), 0.0), uShin) * uSpec : 0.0;
    vec3 col = uColor.rgb * (uAmbient + (1.0 - uAmbient) * d) + vec3(s);
    fragColor = vec4(col, 1.0);
}
"""

# ── Glass enclosure (Phong + Fresnel rim, alpha) ───────────────────────────────
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
    vPosV   = posV.xyz;
    vNormV  = mat3(uMV) * aNorm;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_GLASS_FS = """
#version 330 core
in  vec3 vNormV;
in  vec3 vPosV;
out vec4 fragColor;
uniform vec4  uColor;
uniform vec3  uLightV;
uniform float uAmbient;
uniform float uSpec;
uniform float uShin;
void main() {
    vec3 N = normalize(vNormV);
    vec3 V = normalize(-vPosV);
    vec3 L = normalize(uLightV);
    float d = max(dot(N, L), 0.0);
    vec3 H  = normalize(L + V);
    float s = pow(max(dot(N, H), 0.0), uShin) * uSpec;
    float fr = pow(1.0 - abs(dot(N, V)), 3.0);
    float a  = uColor.a + (1.0 - uColor.a) * fr * 0.7;
    vec3 col = uColor.rgb * (uAmbient + (1.0 - uAmbient) * d) + vec3(s);
    fragColor = vec4(col, clamp(a, 0.0, 1.0));
}
"""

# ── Lines (wireframe / grid) ────────────────────────────────────────────────
_LINE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
"""

_LINE_FS = """
#version 330 core
out vec4 fragColor;
uniform vec4 uColor;
void main() { fragColor = uColor; }
"""

# ── HUD 2-D panel texture ────────────────────────────────────────────────────
_HUD_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_HUD_FS = """
#version 330 core
in  vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() { fragColor = texture(uTex, vUV); }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Internal GL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compile(vs_src: str, fs_src: str):
    if not _HAS_GL:
        return 0
    vs = _gl_shaders.compileShader(vs_src, GL_VERTEX_SHADER)
    fs = _gl_shaders.compileShader(fs_src, GL_FRAGMENT_SHADER)
    return _gl_shaders.compileProgram(vs, fs)


def _vao_pos_norm(data: np.ndarray):
    """Upload float32 (-1,6) [x,y,z,nx,ny,nz] into a VAO.
    Returns (vao, vbo, n_verts)."""
    if not _HAS_GL or len(data) == 0:
        return 0, 0, 0
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    flat = data.astype(np.float32, copy=False).flatten()
    glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
    stride = 24
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glBindVertexArray(0)
    return vao, vbo, len(data)


def _vao_pos3(data: np.ndarray):
    """Upload float32 (-1,3) [x,y,z] into a VAO.  Returns (vao, vbo, n_verts)."""
    if not _HAS_GL or len(data) == 0:
        return 0, 0, 0
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    flat = data.astype(np.float32, copy=False).flatten()
    glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glBindVertexArray(0)
    return vao, vbo, len(data)


def _room_floor_transform_points(points: np.ndarray, cfg: dict, transform: tuple[float, float, float] | None = None) -> np.ndarray:
    """Map applied grid-space floor points into the room floor coordinate space."""
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.size == 0:
        return np.zeros((0, 3), np.float32)
    if transform is None:
        min_xy = np.min(pts[:, :2], axis=0)
        max_xy = np.max(pts[:, :2], axis=0)
        transform = (
            0.5 * float(min_xy[0] + max_xy[0]),
            float(min_xy[1]),
            max(0.0, float(cfg.get("applied_floor_z_lift", 0.006) or 0.006)),
        )
    center_x, min_y, z_lift = transform
    pts[:, 0] -= center_x
    pts[:, 1] -= min_y
    pts[:, 2] += z_lift
    return pts


def _floor_tris_to_pos_norm(tris: np.ndarray, cfg: dict, transform: tuple[float, float, float] | None = None) -> np.ndarray:
    """Convert (N,3,3) floor triangles to interleaved position/normal rows."""
    arr = np.asarray(tris, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3) or arr.size == 0:
        return np.zeros((0, 6), np.float32)
    pts = arr.reshape(-1, 3)
    pts = _room_floor_transform_points(pts, cfg, transform=transform)
    norms = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (len(pts), 1))
    return np.concatenate([pts, norms], axis=1).astype(np.float32, copy=False)


def _make_tex(surf: pygame.Surface) -> int:
    if not _HAS_GL:
        return 0
    data = pygame.image.tostring(surf, "RGBA", True)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 surf.get_width(), surf.get_height(),
                 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def _update_tex(tex: int, surf: pygame.Surface):
    if not _HAS_GL:
        return
    data = pygame.image.tostring(surf, "RGBA", True)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 surf.get_width(), surf.get_height(),
                 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glBindTexture(GL_TEXTURE_2D, 0)


def _quad_verts(x: float, y: float, w: float, h: float,
                win_w: int, win_h: int) -> np.ndarray:
    """NDC quad for a screen-space rect (pixels → NDC)."""
    x0 = 2.0 * x / win_w - 1.0
    y0 = 2.0 * y / win_h - 1.0
    x1 = 2.0 * (x + w) / win_w - 1.0
    y1 = 2.0 * (y + h) / win_h - 1.0
    return np.array([
        x0, y0, 0.0, 0.0,
        x1, y0, 1.0, 0.0,
        x1, y1, 1.0, 1.0,
        x0, y0, 0.0, 0.0,
        x1, y1, 1.0, 1.0,
        x0, y1, 0.0, 1.0,
    ], np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Matrix helpers (self-contained, no demo_pluck_gl import)
# ─────────────────────────────────────────────────────────────────────────────

def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y * 0.5)
    M = np.zeros((4, 4), np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2.0 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def _look_at(eye: np.ndarray, target: np.ndarray,
             up: np.ndarray = None) -> np.ndarray:
    if up is None:
        up = np.array([0, 0, 1], np.float64)
    f = target - eye
    f = f / (np.linalg.norm(f) or 1.0)
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) or 1.0)
    u = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = r
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3]  = -float(np.dot(r, eye))
    M[1, 3]  = -float(np.dot(u, eye))
    M[2, 3]  =  float(np.dot(f, eye))
    return M


# ─────────────────────────────────────────────────────────────────────────────
# Pygame panel: Scene Tree
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_ICON = {
    "PlacedLight":       "💡",
    "PlacedCamera":      "📷",
    "PlacedEnclosure":   "◻",
    "PlacedDutyStation": "⬡",
    "PlacedPortalFrame": "⬟",
}
_TYPE_COLOR = {
    "PlacedLight":       (255, 230,  80),
    "PlacedCamera":      ( 80, 220, 100),
    "PlacedEnclosure":   ( 80, 160, 255),
    "PlacedDutyStation": ( 80, 220, 220),
    "PlacedPortalFrame": (210,  80, 255),
}

_BG      = (18,  20,  28, 245)
_SEL_BG  = (30,  55,  90, 255)
_FG      = (210, 215, 230, 255)
_DIM     = (110, 115, 130, 255)
_FONT_SZ = 14


class _SceneTreePanel:
    """Left panel — scrollable list of all placed objects."""

    def __init__(self, width: int, height: int, ws: RoomWorkspace):
        self._w  = width
        self._h  = height
        self._ws = ws
        self._surf: Optional[pygame.Surface] = None
        self._tex:  int = 0
        self._scroll = 0
        self._row_h  = 26
        self._dirty  = True
        if pygame.get_init():
            self._font = pygame.font.SysFont("monospace", _FONT_SZ)

    def mark_dirty(self):
        self._dirty = True

    def _draw(self):
        if not pygame.get_init():
            return
        surf = pygame.Surface((self._w, self._h), pygame.SRCALPHA)
        surf.fill(_BG)

        # Title
        pygame.draw.rect(surf, (30, 40, 60, 255), (0, 0, self._w, 24))
        t = self._font.render("  SCENE", True, (160, 200, 255))
        surf.blit(t, (6, 4))

        y = 28 - self._scroll
        objects = list(self._ws.objects.values())
        sel_id  = self._ws.selected_id
        for obj in objects:
            if y + self._row_h < 0:
                y += self._row_h
                continue
            if y > self._h:
                break
            cls_name = type(obj).__name__
            bg_color = _SEL_BG if obj.obj_id == sel_id else (0, 0, 0, 0)
            if bg_color[3]:
                pygame.draw.rect(surf, bg_color, (0, y, self._w, self._row_h))
            col  = _TYPE_COLOR.get(cls_name, _FG)
            icon = _TYPE_ICON.get(cls_name, "·")
            lbl  = obj.label[:22] if len(obj.label) > 22 else obj.label
            text = self._font.render(f"{icon} {lbl}", True, col)
            surf.blit(text, (6, y + 4))
            y += self._row_h

        self._surf = surf
        self._dirty = False

    def get_tex(self) -> int:
        if self._dirty or self._surf is None:
            self._draw()
        if self._surf is None:
            return 0
        if self._tex == 0:
            self._tex = _make_tex(self._surf)
        else:
            _update_tex(self._tex, self._surf)
        return self._tex

    def handle_event(self, ev, px: int, py: int) -> bool:
        """Handle a mouse event translated to panel-local coords."""
        if not pygame.get_init():
            return False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if 0 <= px < self._w and 0 <= py < self._h:
                row = (py + self._scroll - 28) // self._row_h
                objects = list(self._ws.objects.values())
                if 0 <= row < len(objects):
                    self._ws.select(objects[row].obj_id)
                    self.mark_dirty()
                    return True
        if ev.type == pygame.MOUSEWHEEL:
            self._scroll = max(0, self._scroll - ev.y * self._row_h)
            self.mark_dirty()
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Tile type constants
# ─────────────────────────────────────────────────────────────────────────────

TILE_FLAT    = 0   # standard flat geometry
TILE_ARC     = 1   # wall curves to cylindrical arc
TILE_QSPHERE = 2   # corner transitions to quarter-sphere

_TILE_FILL = {
    TILE_FLAT:    ( 22,  34,  52, 255),
    TILE_ARC:     ( 14,  54,  36, 255),
    TILE_QSPHERE: ( 44,  18,  60, 255),
}
_TILE_LINE = {
    TILE_FLAT:    ( 80, 120, 200, 200),
    TILE_ARC:     ( 60, 200, 110, 200),
    TILE_QSPHERE: (170,  60, 230, 200),
}
_TILE_NAMES = {TILE_FLAT: "Flat", TILE_ARC: "Arc", TILE_QSPHERE: "QSph"}
_TILE_CYCLE = {TILE_FLAT: TILE_ARC, TILE_ARC: TILE_QSPHERE, TILE_QSPHERE: TILE_FLAT}

_GRID_COLOR  = (220, 200,   0, 190)   # yellow tile-border lines
_GRID_WIDTH  = 2                       # px
_SEL_COLOR   = (255, 255,   0, 255)   # selected-tile thick border
_HOV_COLOR   = ( 90, 110, 160,  50)   # hover overlay (RGBA)
_OCC_COLOR   = ( 80,  90, 110, 100)   # occupied overlay (RGBA)


# ─────────────────────────────────────────────────────────────────────────────
# Pygame panel: Tile grid editor
# ─────────────────────────────────────────────────────────────────────────────

class _TileEditorPanel:
    """Centre panel — tile-based room grid editor.

    Floor plan is divided into 1 m × 1 m tiles.  Left-click cycles the tile
    type (Flat → Arc → QSph → Flat) unless an object occupies the tile.
    Scroll=zoom, middle-drag=pan.

    Coordinate convention (Z-up):
        X = left/right,  Y = depth (0 = entrance, D = back wall)
    The 2-D map shows X horizontal, Y vertical with entrance at the bottom.

    Tile grid keys: (gx, gy) where
        gx = floor(wx + W/2)  ∈ [0, W-1]
        gy = floor(wy)        ∈ [0, D-1]
    """

    def __init__(self, width: int, height: int, ws: RoomWorkspace,
                 tile_grid: dict, on_tile_changed=None):
        self._w    = width
        self._h    = height
        self._ws   = ws
        self._tile_grid    = tile_grid          # { (gx,gy): tile_type }
        self._on_tile_changed = on_tile_changed

        self._surf: Optional[pygame.Surface] = None
        self._tex:  int = 0
        self._dirty = True

        # Viewport — fit the whole room on first draw
        dims = ws.room_dims()
        W = float(dims.get("width_m",  12.0))
        D = float(dims.get("depth_m",  10.0))
        margin = 2.0
        self._zoom  = min(width  / (W + margin),
                          height / (D + margin))
        self._pan_x = 0.0          # world X at panel centre
        self._pan_y = D / 2.0      # world Y at panel centre

        self._hovered:  Optional[tuple] = None  # (gx, gy) under cursor
        self._sel_tile: Optional[tuple] = None  # (gx, gy) selected tile

        if pygame.get_init():
            self._font   = pygame.font.SysFont("monospace", 11)
            self._font_t = pygame.font.SysFont("monospace", 13, bold=True)

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def mark_dirty(self):
        self._dirty = True

    def _world_to_px(self, wx: float, wy: float):
        """World (X, Y_depth) → panel pixel.  Y increases upward on screen."""
        px = self._w / 2 + (wx - self._pan_x) * self._zoom
        py = self._h / 2 - (wy - self._pan_y) * self._zoom
        return int(px), int(py)

    def _px_to_world(self, px: int, py: int):
        wx = (px - self._w / 2) / self._zoom + self._pan_x
        wy = -(py - self._h / 2) / self._zoom + self._pan_y
        return wx, wy

    def _px_to_tile(self, px: int, py: int):
        """Panel pixel → tile (gx, gy), or None if outside the room."""
        dims = self._ws.room_dims()
        W = float(dims.get("width_m",  12.0))
        D = float(dims.get("depth_m",  10.0))
        wx, wy = self._px_to_world(px, py)
        gx = int(math.floor(wx + W / 2))
        gy = int(math.floor(wy))
        if 0 <= gx < int(W) and 0 <= gy < int(D):
            return gx, gy
        return None

    def _tile_px_rect(self, gx: int, gy: int):
        """Screen-space rect (x, y, w, h) for tile (gx, gy)."""
        dims = self._ws.room_dims()
        W = float(dims.get("width_m", 12.0))
        wx0 = gx - W / 2
        wx1 = wx0 + 1.0
        # wy1 > wy0 → deeper into room → higher on screen (smaller py)
        px0, py0 = self._world_to_px(wx0, gy + 1.0)   # top-left
        px1, py1 = self._world_to_px(wx1, float(gy))  # bottom-right
        return px0, py0, max(1, px1 - px0), max(1, py1 - py0)

    def _obj_at_tile(self, gx: int, gy: int):
        """Return the first object whose centre falls in tile (gx, gy)."""
        dims = self._ws.room_dims()
        W = float(dims.get("width_m", 12.0))
        tx0, ty0 = gx - W / 2,  float(gy)
        tx1, ty1 = tx0 + 1.0,   float(gy + 1)
        for obj in self._ws.objects.values():
            ox, oy = float(obj.pos[0]), float(obj.pos[1])
            if tx0 <= ox < tx1 and ty0 <= oy < ty1:
                return obj
        return None

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_tile_indicator(self, surf, tt: int,
                              rx: int, ry: int, rw: int, rh: int):
        """Draw a small type-specific decoration inside a tile."""
        col = _TILE_LINE[tt]
        cx, cy = rx + rw // 2, ry + rh // 2
        r = max(3, min(rw, rh) // 3)
        if tt == TILE_ARC:
            pygame.draw.arc(surf, col,
                            (cx - r, cy - r, 2 * r, 2 * r),
                            0.0, math.pi, 2)
        elif tt == TILE_QSPHERE:
            pygame.draw.arc(surf, col,
                            (rx + 1, ry + 1, 2 * r, 2 * r),
                            -math.pi / 2, 0.0, 2)

    def _draw_legend(self, surf):
        lx = self._w - 88
        ly = 28
        for i, tt in enumerate((TILE_FLAT, TILE_ARC, TILE_QSPHERE)):
            fill = _TILE_FILL[tt]
            pygame.draw.rect(surf, fill, (lx, ly + i * 18, 14, 14))
            pygame.draw.rect(surf, _GRID_COLOR[:3] + (160,),
                             (lx, ly + i * 18, 14, 14), 1)
            lbl = self._font.render(_TILE_NAMES[tt], True, _TILE_LINE[tt][:3])
            surf.blit(lbl, (lx + 17, ly + i * 18 + 1))

    def _draw_status(self, surf, W: int, D: int):
        sh = 22
        pygame.draw.rect(surf, (18, 26, 44, 220),
                         (0, self._h - sh, self._w, sh))
        if self._sel_tile:
            gx, gy = self._sel_tile
            tt  = self._tile_grid.get((gx, gy), TILE_FLAT)
            obj = self._obj_at_tile(gx, gy)
            occ = f"  [{obj.label[:12]}]" if obj else ""
            msg = (f"  Tile ({gx},{gy})  {_TILE_NAMES[tt]}{occ}"
                   f"  — click:cycle  RMB:deselect")
        elif self._hovered:
            gx, gy = self._hovered
            tt  = self._tile_grid.get((gx, gy), TILE_FLAT)
            msg = f"  Hover ({gx},{gy})  {_TILE_NAMES[tt]}  — click to cycle"
        else:
            msg = (f"  Room {W}×{D} m  "
                   f"  scroll:zoom  mid-drag:pan  click tile:cycle")
        t = self._font.render(msg, True, (155, 165, 185))
        surf.blit(t, (0, self._h - sh + 5))

    def _draw_compass(self, surf):
        bx, by = self._w - 22, self._h - 50
        # N arrow = deeper into room = top of panel
        pygame.draw.polygon(surf, (200, 80, 80),
                            [(bx, by - 10), (bx - 4, by + 2), (bx + 4, by + 2)])
        n = self._font.render("N", True, (200, 80, 80))
        surf.blit(n, (bx - 4, by - 22))

    # ── Main draw ─────────────────────────────────────────────────────────────

    def _draw(self):
        if not pygame.get_init():
            return

        surf = pygame.Surface((self._w, self._h), pygame.SRCALPHA)
        surf.fill((10, 14, 22, 255))

        dims = self._ws.room_dims()
        W = int(dims.get("width_m",  12.0))
        D = int(dims.get("depth_m",  10.0))

        # ── Tile fills ────────────────────────────────────────────────────────
        for gy in range(D):
            for gx in range(W):
                tt = self._tile_grid.get((gx, gy), TILE_FLAT)
                rx, ry, rw, rh = self._tile_px_rect(gx, gy)

                # Base colour
                pygame.draw.rect(surf, _TILE_FILL[tt], (rx, ry, rw, rh))

                # Hover overlay
                if self._hovered == (gx, gy):
                    hs = pygame.Surface((rw, rh), pygame.SRCALPHA)
                    hs.fill(_HOV_COLOR)
                    surf.blit(hs, (rx, ry))

                # Occupied overlay
                if self._obj_at_tile(gx, gy) is not None:
                    os_ = pygame.Surface((rw, rh), pygame.SRCALPHA)
                    os_.fill(_OCC_COLOR)
                    surf.blit(os_, (rx, ry))

                # Type indicator (arc / qsphere)
                if tt != TILE_FLAT and rw > 8 and rh > 8:
                    self._draw_tile_indicator(surf, tt, rx, ry, rw, rh)

        # ── Yellow grid lines ─────────────────────────────────────────────────
        grid_surf = pygame.Surface((self._w, self._h), pygame.SRCALPHA)
        Wf = float(W)
        for gx in range(W + 1):
            wx = gx - Wf / 2
            p0 = self._world_to_px(wx, 0.0)
            p1 = self._world_to_px(wx, float(D))
            pygame.draw.line(grid_surf, _GRID_COLOR, p0, p1, _GRID_WIDTH)
        for gy in range(D + 1):
            p0 = self._world_to_px(-Wf / 2, float(gy))
            p1 = self._world_to_px( Wf / 2, float(gy))
            pygame.draw.line(grid_surf, _GRID_COLOR, p0, p1, _GRID_WIDTH)
        surf.blit(grid_surf, (0, 0))

        # ── Selected tile border ──────────────────────────────────────────────
        if self._sel_tile is not None:
            gx, gy = self._sel_tile
            if 0 <= gx < W and 0 <= gy < D:
                rx, ry, rw, rh = self._tile_px_rect(gx, gy)
                pygame.draw.rect(surf, _SEL_COLOR, (rx, ry, rw, rh), 3)

        # ── Object markers ────────────────────────────────────────────────────
        sel_id = self._ws.selected_id
        for obj in self._ws.objects.values():
            ox = float(obj.pos[0])
            oy = float(obj.pos[1])   # Y = depth (Z-up convention)
            px, py = self._world_to_px(ox, oy)
            is_sel  = obj.obj_id == sel_id
            cls     = type(obj).__name__
            col3    = _TYPE_COLOR.get(cls, (210, 215, 230))
            col     = tuple(min(255, c + 60) for c in col3) if is_sel else col3

            if isinstance(obj, PlacedLight):
                pygame.draw.circle(surf, col, (px, py), 5)
                pygame.draw.circle(surf, (255, 255, 200), (px, py), 5, 1)
            elif isinstance(obj, PlacedEnclosure):
                pygame.draw.circle(surf, col, (px, py), 8, 2)
            elif isinstance(obj, PlacedDutyStation):
                r = 7
                pts = [(px, py - r), (px + r, py + r // 2), (px - r, py + r // 2)]
                pygame.draw.polygon(surf, col, pts)
            elif isinstance(obj, PlacedPortalFrame):
                pygame.draw.rect(surf, col, (px - 6, py - 10, 12, 20), 2)
            else:
                pygame.draw.circle(surf, col, (px, py), 4)

            lbl = self._font.render(obj.label[:14], True, (175, 180, 200))
            surf.blit(lbl, (px + 8, py - 5))

        # ── Title bar ─────────────────────────────────────────────────────────
        pygame.draw.rect(surf, (18, 28, 48, 245), (0, 0, self._w, 22))
        title = self._font_t.render("  TILE EDITOR", True, (160, 200, 255))
        surf.blit(title, (4, 3))

        # ── Legend, status, compass ───────────────────────────────────────────
        self._draw_legend(surf)
        self._draw_status(surf, W, D)
        self._draw_compass(surf)

        self._surf  = surf
        self._dirty = False

    # ── Public interface ──────────────────────────────────────────────────────

    def get_tex(self) -> int:
        if self._dirty or self._surf is None:
            self._draw()
        if self._surf is None:
            return 0
        if self._tex == 0:
            self._tex = _make_tex(self._surf)
        else:
            _update_tex(self._tex, self._surf)
        return self._tex

    def handle_event(self, ev, local_px: int, local_py: int) -> bool:
        if not pygame.get_init():
            return False

        if ev.type == pygame.MOUSEMOTION:
            new_hov = self._px_to_tile(local_px, local_py)
            if new_hov != self._hovered:
                self._hovered = new_hov
                self.mark_dirty()
            if ev.buttons[1]:   # middle-drag → pan
                self._pan_x -= ev.rel[0] / self._zoom
                self._pan_y += ev.rel[1] / self._zoom   # Y flip
                self.mark_dirty()
            return False

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            tile = self._px_to_tile(local_px, local_py)
            if tile is not None:
                gx, gy = tile
                obj = self._obj_at_tile(gx, gy)
                if obj is not None:
                    # Click on occupied tile → select the object
                    self._ws.select(obj.obj_id)
                    self._sel_tile = tile
                    self.mark_dirty()
                    return True
                # Cycle tile type
                self._sel_tile = tile
                old_tt = self._tile_grid.get(tile, TILE_FLAT)
                new_tt = _TILE_CYCLE[old_tt]
                if new_tt == TILE_FLAT:
                    self._tile_grid.pop(tile, None)
                else:
                    self._tile_grid[tile] = new_tt
                self.mark_dirty()
                if self._on_tile_changed is not None:
                    self._on_tile_changed(gx, gy, new_tt)
                return True

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
            self._sel_tile = None
            self.mark_dirty()
            return True

        if ev.type == pygame.MOUSEWHEEL:
            self._zoom = max(8.0, min(300.0, self._zoom * (1.0 + ev.y * 0.12)))
            self.mark_dirty()
            return True

        return False


# ─────────────────────────────────────────────────────────────────────────────
# Pygame panel: Minimap  (kept for reference — superseded by _TileEditorPanel)
# ─────────────────────────────────────────────────────────────────────────────

class _MinimapPanel:
    """Centre panel — top-down floor plan rendered to a pygame surface."""

    def __init__(self, width: int, height: int, ws: RoomWorkspace):
        self._w  = width
        self._h  = height
        self._ws = ws
        self._surf: Optional[pygame.Surface] = None
        self._tex:  int = 0
        self._dirty = True
        self._pan_x = 0.0   # metres
        self._pan_z = 0.0
        self._zoom  = 35.0  # pixels per metre
        if pygame.get_init():
            self._font = pygame.font.SysFont("monospace", 11)

    def mark_dirty(self):
        self._dirty = True

    def _world_to_px(self, wx: float, wz: float):
        """Convert world (x, z) to panel pixel coordinates."""
        px = self._w / 2 + (wx - self._pan_x) * self._zoom
        py = self._h / 2 + (wz - self._pan_z) * self._zoom
        return int(px), int(py)

    def _draw(self):
        if not pygame.get_init():
            return
        surf = pygame.Surface((self._w, self._h), pygame.SRCALPHA)
        surf.fill((12, 14, 20, 245))

        dims = self._ws.room_dims()
        W = float(dims.get("width_m", 12.0))
        D = float(dims.get("depth_m", 10.0))

        # Room outline
        x0_px, z0_px = self._world_to_px(-W / 2, 0.0)
        x1_px, z1_px = self._world_to_px( W / 2, D)
        pygame.draw.rect(surf, (40, 55, 80),
                         (x0_px, z0_px, x1_px - x0_px, z1_px - z0_px), 0)
        pygame.draw.rect(surf, (80, 110, 160),
                         (x0_px, z0_px, x1_px - x0_px, z1_px - z0_px), 1)

        # Grid lines
        spacing = 1.0
        grid_col = (40, 50, 70)
        x = -W / 2
        while x <= W / 2 + 1e-3:
            px0, py0 = self._world_to_px(x, 0.0)
            px1, py1 = self._world_to_px(x, D)
            pygame.draw.line(surf, grid_col, (px0, py0), (px1, py1))
            x += spacing
        z = 0.0
        while z <= D + 1e-3:
            px0, py0 = self._world_to_px(-W / 2, z)
            px1, py1 = self._world_to_px( W / 2, z)
            pygame.draw.line(surf, grid_col, (px0, py0), (px1, py1))
            z += spacing

        # Objects
        sel_id = self._ws.selected_id
        for obj in self._ws.objects.values():
            ox = float(obj.pos[0])
            oz = float(obj.pos[1])   # Y is depth in Z-up convention
            px, pz = self._world_to_px(ox, oz)
            is_sel = obj.obj_id == sel_id
            from placed_object import (PlacedLight, PlacedEnclosure,
                                       PlacedDutyStation, PlacedPortalFrame)
            if isinstance(obj, PlacedLight):
                col = (255, 230, 80) if not is_sel else (255, 255, 120)
                pygame.draw.circle(surf, col, (px, pz), 5)
                pygame.draw.circle(surf, (255, 255, 200), (px, pz), 5, 1)
            elif isinstance(obj, PlacedEnclosure):
                col = (80, 160, 255) if not is_sel else (140, 210, 255)
                pygame.draw.circle(surf, col, (px, pz), 8, 2)
            elif isinstance(obj, PlacedDutyStation):
                col = (80, 220, 220) if not is_sel else (160, 255, 255)
                r = 7
                points = [(px, pz - r), (px + r, pz + r//2),
                           (px - r, pz + r//2)]
                pygame.draw.polygon(surf, col, points)
            elif isinstance(obj, PlacedPortalFrame):
                col = (210, 80, 255) if not is_sel else (240, 140, 255)
                pygame.draw.rect(surf, col,
                                  (px - 6, pz - 10, 12, 20), 2)
            label_surf = self._font.render(obj.label[:12], True, (180, 185, 200))
            surf.blit(label_surf, (px + 8, pz - 6))

        # Compass
        pygame.draw.polygon(surf, (200, 80, 80),
                             [(self._w - 20, 20), (self._w - 24, 32),
                              (self._w - 16, 32)])
        n_lbl = self._font.render("N", True, (200, 80, 80))
        surf.blit(n_lbl, (self._w - 22, 6))

        self._surf = surf
        self._dirty = False

    def get_tex(self) -> int:
        if self._dirty or self._surf is None:
            self._draw()
        if self._surf is None:
            return 0
        if self._tex == 0:
            self._tex = _make_tex(self._surf)
        else:
            _update_tex(self._tex, self._surf)
        return self._tex

    def handle_event(self, ev, px: int, py: int) -> bool:
        if ev.type == pygame.MOUSEWHEEL:
            self._zoom = max(10.0, min(200.0, self._zoom * (1.0 + ev.y * 0.1)))
            self.mark_dirty()
            return True
        if ev.type == pygame.MOUSEMOTION and ev.buttons[1]:
            # Middle-drag to pan
            self._pan_x -= ev.rel[0] / self._zoom
            self._pan_z -= ev.rel[1] / self._zoom
            self.mark_dirty()
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Pygame panel: Properties
# ─────────────────────────────────────────────────────────────────────────────

class _PropertiesPanel:
    """Right panel — shows fields of the currently selected object."""

    def __init__(self, width: int, height: int, ws: RoomWorkspace):
        self._w   = width
        self._h   = height
        self._ws  = ws
        self._surf: Optional[pygame.Surface] = None
        self._tex:  int = 0
        self._dirty = True
        if pygame.get_init():
            self._font   = pygame.font.SysFont("monospace", _FONT_SZ)
            self._font_s = pygame.font.SysFont("monospace", 12)

    def mark_dirty(self):
        self._dirty = True

    def _draw(self):
        if not pygame.get_init():
            return
        surf = pygame.Surface((self._w, self._h), pygame.SRCALPHA)
        surf.fill(_BG)
        pygame.draw.rect(surf, (30, 40, 60, 255), (0, 0, self._w, 24))
        t = self._font.render("  PROPERTIES", True, (160, 200, 255))
        surf.blit(t, (6, 4))

        obj = self._ws.selected
        if obj is None:
            msg = self._font_s.render("  (nothing selected)", True, _DIM)
            surf.blit(msg, (8, 40))
            self._surf = surf
            self._dirty = False
            return

        y = 34
        lh = 20

        def row(label: str, value: str, col=_FG):
            nonlocal y
            lbl_s = self._font_s.render(f"  {label}:", True, _DIM)
            val_s = self._font_s.render(f"  {value}", True, col)
            surf.blit(lbl_s, (4, y))
            surf.blit(val_s, (4, y + lh - 2))
            y += lh * 2

        row("ID",    obj.obj_id)
        row("Type",  type(obj).__name__,  (180, 220, 255))
        row("Label", obj.label)
        px, py, pz = obj.pos
        row("Pos",   f"[{px:.2f}, {py:.2f}, {pz:.2f}]", (200, 255, 200))
        row("Yaw",   f"{obj.yaw_deg:.1f}°")

        from placed_object import PlacedLight, PlacedEnclosure
        if isinstance(obj, PlacedLight):
            row("Kind",      obj.kind,          (255, 230, 80))
            row("Intensity", f"{obj.intensity:.2f}", (255, 220, 100))
            row("Radius",    f"{obj.radius_m:.1f} m")
            r, g, b = obj.color[:3]
            row("Color",     f"({r:.2f}, {g:.2f}, {b:.2f})")
        elif isinstance(obj, PlacedEnclosure):
            row("Shape",  obj.shape,  (80, 160, 255))
            has_sim = obj.simulator is not None
            row("Sim",    "yes" if has_sim else "no",
                          (160, 255, 160) if has_sim else _DIM)
            if has_sim:
                ws_obj = getattr(obj, "_simulator_ws", None)
                if ws_obj is not None:
                    n = ws_obj._n_items
                    mode = ws_obj.mode.value
                    row("Grid",   f"{n} items")
                    row("State",  mode, (200, 200, 100))

        self._surf = surf
        self._dirty = False

    def get_tex(self) -> int:
        if self._dirty or self._surf is None:
            self._draw()
        if self._surf is None:
            return 0
        if self._tex == 0:
            self._tex = _make_tex(self._surf)
        else:
            _update_tex(self._tex, self._surf)
        return self._tex

    def handle_event(self, ev, px: int, py: int) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# RoomStation
# ─────────────────────────────────────────────────────────────────────────────

class RoomStation:
    """Master GL renderer for the room environment.

    Parameters
    ----------
    ws : RoomWorkspace
        The owning state machine.
    win_w, win_h : int
        Initial window dimensions.
    """

    def __init__(self, ws: RoomWorkspace, win_w: int = 1400, win_h: int = 900):
        self._ws    = ws
        self._win_w = win_w
        self._win_h = win_h

        lay = ws.layout()
        self._lw = int(lay.get("left_panel_width",  180))
        self._rw = int(lay.get("right_panel_width", 220))

        self._hud_visible = False   # show HUD only in INTERACT mode

        # GL object storage
        self._prog_room:  int = 0
        self._prog_glass: int = 0
        self._prog_line:  int = 0
        self._prog_hud:   int = 0
        self._hud_vao:    int = 0
        self._hud_vbo:    int = 0

        # Room surface VAOs
        self._floor_vao:   int = 0;  self._floor_n:   int = 0
        self._ceiling_vao: int = 0;  self._ceiling_n: int = 0
        self._walls_vao:   int = 0;  self._walls_n:   int = 0
        self._grid_vao:    int = 0;  self._grid_n:    int = 0
        self._floor_tile_vao: int = 0; self._floor_tile_n: int = 0
        self._floor_fill_vao: int = 0; self._floor_fill_n: int = 0
        self._floor_border_vao: int = 0; self._floor_border_n: int = 0
        self._room_floor_revision: int = -1

        # Per-enclosure VAOs  { obj_id: (mesh_vao, mesh_n, wire_vao, wire_n) }
        self._enc_vaos: dict = {}

        # Tile grid — sparse dict of non-flat tiles: { (gx,gy): tile_type }
        self._tile_grid: dict = {}

        # Panels
        self._panel_tree: Optional[_SceneTreePanel]    = None
        self._panel_map:  Optional[_TileEditorPanel]   = None
        self._panel_props:Optional[_PropertiesPanel]   = None

        # Material colours from room.yaml
        rcfg = ws.room_cfg
        def _rgba(key, default):
            v = rcfg.get(key, default)
            return [float(x) for x in (v + [1.0])[:4]]
        self._floor_col   = _rgba("floor_color",   [0.18, 0.18, 0.22, 1.0])
        self._floor_tile_col = _rgba("floor_tile_color", [0.96, 0.93, 0.86, 1.0])
        self._floor_fill_col = _rgba("floor_fill_color", [0.18, 0.18, 0.20, 1.0])
        self._floor_border_col = _rgba("floor_border_color", [0.98, 0.95, 0.82, 0.95])
        self._ceiling_col = _rgba("ceiling_color", [0.12, 0.12, 0.15, 1.0])
        self._wall_col    = _rgba("wall_color",     [0.15, 0.16, 0.20, 1.0])
        self._grid_col    = rcfg.get("floor_grid", {}).get(
                                "color", [0.30, 0.30, 0.40, 0.45])
        self._ambient     = float(rcfg.get("ambient",      0.18))
        self._spec        = float(rcfg.get("spec_strength", 0.15))
        self._shin        = float(rcfg.get("shininess",    32.0))

    # ── Public lifecycle ──────────────────────────────────────────────────────

    def init_gl(self):
        """Compile shaders and upload all static geometry."""
        if not _HAS_GL:
            return

        self._prog_room  = _compile(_ROOM_VS, _ROOM_FS)
        self._prog_glass = _compile(_GLASS_VS, _GLASS_FS)
        self._prog_line  = _compile(_LINE_VS, _LINE_FS)
        self._prog_hud   = _compile(_HUD_VS,  _HUD_FS)

        self._hud_vao = glGenVertexArrays(1)
        self._hud_vbo = glGenBuffers(1)

        self._rebuild_room_surfaces()

        # Enclosure geometry
        self._rebuild_enclosures()

        # Panels
        if pygame.get_init():
            win_h = self._win_h
            self._panel_tree  = _SceneTreePanel(
                self._lw, win_h, self._ws)
            map_w = self._win_w - self._lw - self._rw
            self._panel_map   = _TileEditorPanel(
                max(100, map_w), win_h, self._ws,
                self._tile_grid, self._on_tile_changed)
            self._panel_props = _PropertiesPanel(
                self._rw, win_h, self._ws)

    def destroy_gl(self):
        if not _HAS_GL:
            return
        for prog in [self._prog_room, self._prog_glass,
                     self._prog_line, self._prog_hud]:
            if prog:
                glDeleteBuffers(1, [prog])   # programs are deleted differently
        for vao_n in [(self._floor_vao, self._floor_n),
                      (self._ceiling_vao, self._ceiling_n),
                      (self._walls_vao, self._walls_n),
                      (self._grid_vao, self._grid_n),
                      (self._floor_tile_vao, self._floor_tile_n),
                      (self._floor_fill_vao, self._floor_fill_n),
                      (self._floor_border_vao, self._floor_border_n)]:
            if vao_n[0]:
                glDeleteVertexArrays(1, [vao_n[0]])
        for mv, mw, wv, ww in self._enc_vaos.values():
            if mv: glDeleteVertexArrays(1, [mv])
            if wv: glDeleteVertexArrays(1, [wv])

    def _rebuild_room_surfaces(self):
        """Build room VAOs, replacing the base floor when an applied floor exists."""
        if not _HAS_GL:
            return
        meshes = build_room_mesh(self._ws.room_cfg)
        self._floor_vao,   _, self._floor_n   = _vao_pos_norm(meshes["floor"])
        self._ceiling_vao, _, self._ceiling_n = _vao_pos_norm(meshes["ceiling"])
        self._walls_vao,   _, self._walls_n   = _vao_pos_norm(meshes["walls"])

        applied = self._ws.room_cfg.get("applied_floor_meshes", {})
        if isinstance(applied, dict):
            raw_tiles = np.asarray(applied.get("floor_tiles", []), dtype=np.float32)
            raw_fill = np.asarray(applied.get("floor_fill", []), dtype=np.float32)
            borders = np.asarray(applied.get("floor_borders", []), dtype=np.float32)
            floor_pts: list[np.ndarray] = []
            if raw_tiles.ndim == 3 and raw_tiles.shape[1:] == (3, 3) and raw_tiles.size:
                floor_pts.append(raw_tiles.reshape(-1, 3))
            if raw_fill.ndim == 3 and raw_fill.shape[1:] == (3, 3) and raw_fill.size:
                floor_pts.append(raw_fill.reshape(-1, 3))
            if borders.ndim == 2 and borders.shape[1] == 3 and borders.size:
                floor_pts.append(borders)
            transform = None
            if floor_pts:
                all_pts = np.concatenate(floor_pts, axis=0)
                min_xy = np.min(all_pts[:, :2], axis=0)
                max_xy = np.max(all_pts[:, :2], axis=0)
                transform = (
                    0.5 * float(min_xy[0] + max_xy[0]),
                    float(min_xy[1]),
                    max(0.0, float(self._ws.room_cfg.get("applied_floor_z_lift", 0.006) or 0.006)),
                )
            tiles = _floor_tris_to_pos_norm(raw_tiles, self._ws.room_cfg, transform=transform)
            fill = _floor_tris_to_pos_norm(raw_fill, self._ws.room_cfg, transform=transform)
            if tiles.size:
                self._floor_tile_vao, _, self._floor_tile_n = _vao_pos_norm(tiles)
            else:
                self._floor_tile_vao, self._floor_tile_n = 0, 0
            if fill.size:
                self._floor_fill_vao, _, self._floor_fill_n = _vao_pos_norm(fill)
            else:
                self._floor_fill_vao, self._floor_fill_n = 0, 0
            if borders.ndim == 2 and borders.shape[1] == 3 and borders.size:
                borders = _room_floor_transform_points(borders, self._ws.room_cfg, transform=transform)
                self._floor_border_vao, _, self._floor_border_n = _vao_pos3(borders)
            else:
                self._floor_border_vao, self._floor_border_n = 0, 0
        else:
            self._floor_tile_vao, self._floor_tile_n = 0, 0
            self._floor_fill_vao, self._floor_fill_n = 0, 0
            self._floor_border_vao, self._floor_border_n = 0, 0

        grid = build_room_floor_grid(self._ws.room_cfg)
        self._grid_vao, _, self._grid_n = _vao_pos3(grid)
        self._room_floor_revision = int(self._ws.room_cfg.get("applied_floor_revision", 0) or 0)

    def _rebuild_enclosures(self):
        """Build VAOs for all PlacedEnclosure objects."""
        if not _HAS_GL:
            return
        self._enc_vaos.clear()
        for obj in self._ws.enclosures():
            try:
                mesh  = build_enclosure_mesh(obj)
                wire  = build_enclosure_wireframe(obj)
                mv, _, mn = _vao_pos_norm(mesh)
                wv, _, wn = _vao_pos3(wire)
                self._enc_vaos[obj.obj_id] = (mv, mn, wv, wn)
            except Exception as exc:
                print(f"[RoomStation] enclosure VAO failed {obj.obj_id}: {exc}")

    # ── Render: 3-D room geometry ─────────────────────────────────────────────

    def render_room(self, win_w: int, win_h: int,
                    MVP: np.ndarray, MV: np.ndarray,
                    light_v: np.ndarray):
        """Draw room surfaces and enclosures into the current GL context.

        Call this once per frame in the main 3-D pass (before any transparent
        objects from other systems).

        Parameters
        ----------
        MVP, MV : float32 (4,4)
        light_v : float32 (3,)  view-space light direction (unit)
        """
        if not _HAS_GL:
            return
        revision = int(self._ws.room_cfg.get("applied_floor_revision", 0) or 0)
        if revision != self._room_floor_revision:
            self._rebuild_room_surfaces()

        lv = light_v.astype(np.float32)

        # ── Opaque: floor / ceiling / walls ───────────────────────────────────
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        prog = self._prog_room
        glUseProgram(prog)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uMVP"), 1, GL_TRUE, MVP)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uMV"),  1, GL_TRUE, MV)
        glUniform3f(glGetUniformLocation(prog, "uLightV"), *lv)
        glUniform1f(glGetUniformLocation(prog, "uAmbient"),   self._ambient)
        glUniform1f(glGetUniformLocation(prog, "uSpec"),      self._spec)
        glUniform1f(glGetUniformLocation(prog, "uShin"),      self._shin)

        floor_draws = []
        if self._floor_tile_vao or self._floor_fill_vao:
            floor_draws.extend([
                (self._floor_fill_vao, self._floor_fill_n, self._floor_fill_col, 0.05, 18.0),
                (self._floor_tile_vao, self._floor_tile_n, self._floor_tile_col, 0.48, 110.0),
            ])
        else:
            floor_draws.append((self._floor_vao, self._floor_n, self._floor_col, self._spec, self._shin))

        for (vao, n, col, spec, shin) in [
            *floor_draws,
            (self._ceiling_vao, self._ceiling_n, self._ceiling_col, self._spec, self._shin),
            (self._walls_vao,   self._walls_n,   self._wall_col, self._spec, self._shin),
        ]:
            if vao and n:
                glUniform1f(glGetUniformLocation(prog, "uSpec"), float(spec))
                glUniform1f(glGetUniformLocation(prog, "uShin"), float(shin))
                glUniform4f(glGetUniformLocation(prog, "uColor"), *col)
                glBindVertexArray(vao)
                glDrawArrays(GL_TRIANGLES, 0, n)
                glBindVertexArray(0)

        # ── Grid lines (floor overlay) ─────────────────────────────────────────
        if self._grid_vao and self._grid_n:
            lp = self._prog_line
            glUseProgram(lp)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUniformMatrix4fv(glGetUniformLocation(lp, "uMVP"), 1, GL_TRUE, MVP)
            gc = self._grid_col
            glUniform4f(glGetUniformLocation(lp, "uColor"), *gc)
            glLineWidth(1.0)
            glBindVertexArray(self._grid_vao)
            glDrawArrays(GL_LINES, 0, self._grid_n)
            glBindVertexArray(0)
            glDisable(GL_BLEND)

        if self._floor_border_vao and self._floor_border_n:
            lp = self._prog_line
            glUseProgram(lp)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUniformMatrix4fv(glGetUniformLocation(lp, "uMVP"), 1, GL_TRUE, MVP)
            glUniform4f(glGetUniformLocation(lp, "uColor"), *self._floor_border_col)
            glLineWidth(1.25)
            glBindVertexArray(self._floor_border_vao)
            glDrawArrays(GL_LINES, 0, self._floor_border_n)
            glBindVertexArray(0)
            glDisable(GL_BLEND)

        # ── Glass enclosures (alpha, depth-write off) ─────────────────────────
        if self._enc_vaos:
            gp = self._prog_glass
            glUseProgram(gp)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUniformMatrix4fv(glGetUniformLocation(gp, "uMVP"), 1, GL_TRUE, MVP)
            glUniformMatrix4fv(glGetUniformLocation(gp, "uMV"),  1, GL_TRUE, MV)
            glUniform3f(glGetUniformLocation(gp, "uLightV"),  *lv)
            glUniform1f(glGetUniformLocation(gp, "uAmbient"), 0.22)
            glUniform1f(glGetUniformLocation(gp, "uSpec"),    0.65)
            glUniform1f(glGetUniformLocation(gp, "uShin"),    128.0)

            enclosures = {o.obj_id: o for o in self._ws.enclosures()}
            for obj_id, (mv, mn, wv, wn) in self._enc_vaos.items():
                obj = enclosures.get(obj_id)
                if obj is None:
                    continue
                gc = obj.glass_color
                if mv and mn:
                    glUniform4f(glGetUniformLocation(gp, "uColor"), *gc)
                    glBindVertexArray(mv)
                    glDrawArrays(GL_TRIANGLES, 0, mn)
                    glBindVertexArray(0)
                if wv and wn:
                    lp = self._prog_line
                    glUseProgram(lp)
                    glUniformMatrix4fv(glGetUniformLocation(lp, "uMVP"), 1, GL_TRUE, MVP)
                    wc = obj.wireframe_color
                    glUniform4f(glGetUniformLocation(lp, "uColor"), *wc)
                    glLineWidth(1.2)
                    glBindVertexArray(wv)
                    glDrawArrays(GL_LINES, 0, wn)
                    glBindVertexArray(0)
                    glUseProgram(gp)
            glDisable(GL_BLEND)

        glUseProgram(0)

    # ── Render: HUD overlay ───────────────────────────────────────────────────

    def render_hud(self, win_w: int, win_h: int):
        """Draw the 2-D HUD overlay (panels) when in interact mode.

        Must be called after the main 3-D pass.
        """
        if not self._hud_visible:
            return
        if not _HAS_GL:
            return
        if (self._panel_tree is None or self._panel_map is None
                or self._panel_props is None):
            return

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self._prog_hud)

        map_w = win_w - self._lw - self._rw

        panels = [
            (self._panel_tree,  0,               0, self._lw,  win_h),
            (self._panel_map,   self._lw,        0, map_w,     win_h),
            (self._panel_props, self._lw + map_w, 0, self._rw, win_h),
        ]
        for panel, px, py, pw, ph in panels:
            tex = panel.get_tex()
            if tex == 0:
                continue
            verts = _quad_verts(px, py, pw, ph, win_w, win_h)
            glBindVertexArray(self._hud_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self._hud_vbo)
            glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, tex)
            glUniform1i(glGetUniformLocation(self._prog_hud, "uTex"), 0)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glUseProgram(0)

    # ── Event routing ─────────────────────────────────────────────────────────

    def show_hud(self, visible: bool):
        self._hud_visible = visible
        self._mark_panels_dirty()

    def _mark_panels_dirty(self):
        if self._panel_tree:  self._panel_tree.mark_dirty()
        if self._panel_map:   self._panel_map.mark_dirty()
        if self._panel_props: self._panel_props.mark_dirty()

    def handle_event(self, ev) -> bool:
        """Route a pygame event to the appropriate panel.

        Returns True if the event was consumed.
        """
        if not self._hud_visible:
            return False
        if not pygame.get_init():
            return False

        mx, my = pygame.mouse.get_pos()
        win_h  = self._win_h
        map_w  = self._win_w - self._lw - self._rw

        if mx < self._lw:
            if self._panel_tree:
                changed = self._panel_tree.handle_event(ev, mx, my)
                if changed:
                    self._panel_props.mark_dirty()
                return changed
        elif mx < self._lw + map_w:
            if self._panel_map:
                return self._panel_map.handle_event(ev, mx - self._lw, my)
        else:
            if self._panel_props:
                return self._panel_props.handle_event(
                    ev, mx - self._lw - map_w, my)
        return False

    # ── Interaction with player controller ───────────────────────────────────

    def player_near(self, player_eye: np.ndarray) -> bool:
        """Return True when the player is within interact_radius of the console."""
        sp   = self._ws.station_pos()
        dist = float(np.linalg.norm(player_eye[[0, 1]] - sp[[0, 1]]))
        return dist < self._ws.interact_radius()

    def _on_tile_changed(self, gx: int, gy: int, new_type: int) -> None:
        """Called by the tile editor when a tile type changes.

        Hook for future tile-aware geometry rebuild; for now just marks
        panels dirty so the HUD redraws.
        """
        self._mark_panels_dirty()

    def update_window_size(self, win_w: int, win_h: int):
        self._win_w = win_w
        self._win_h = win_h
        # Rebuild panels at new size
        if _HAS_GL and pygame.get_init():
            map_w = win_w - self._lw - self._rw
            self._panel_tree  = _SceneTreePanel(self._lw, win_h, self._ws)
            self._panel_map   = _TileEditorPanel(
                max(100, map_w), win_h, self._ws,
                self._tile_grid, self._on_tile_changed)
            self._panel_props = _PropertiesPanel(self._rw, win_h, self._ws)

    # ── Convenience: refresh after scene edit ─────────────────────────────────

    def refresh_scene(self):
        """Call after adding/removing enclosures to rebuild GL geometry."""
        self._rebuild_enclosures()
        self._mark_panels_dirty()
