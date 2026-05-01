"""duty_station.py
==================
Star Trek-style LCARS duty station — procedural mesh + OpenGL renderer.

The station consists of:
  • Console body  — dark metallic Phong-shaded box with tilted top surface
  • Viewscreen    — near-vertical emissive panel (blue glow when active)
  • Side wings    — thin vertical panels flanking the console

All geometry is generated from YAML parameters; no external mesh files.
Vertex format matches demo_pluck_gl _BODY_VS: (aPos xyz, aNorm xyz), float32.

Typical usage
-------------
    from duty_station import DutyStation

    station = DutyStation.from_yaml("configs/meshes/duty_station.yaml")
    station.build_gl()                    # call once after GL context ready

    # in render loop:
    station.draw(MVP, MV, light_v)        # Phong shaded body + screen
    station.draw_screen(MVP, MV)          # separate emissive draw for screen

    # in player controller tick:
    if station.player_near(player_eye):
        show_hint()
"""
from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _yaml = None
    _HAS_YAML = False

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER,
        GL_STATIC_DRAW, GL_TRIANGLES, GL_TRUE, GL_VERTEX_SHADER,
        glBindBuffer, glBindVertexArray, glBufferData,
        glDrawArrays, glEnableVertexAttribArray,
        glGenBuffers, glGenVertexArrays,
        glGetUniformLocation, glUniform1f, glUniform3f, glUniform4f,
        glUniformMatrix4fv, glUseProgram, glVertexAttribPointer,
    )
    from OpenGL.GL import shaders as _gl_shaders
    _HAS_GL = True
except ImportError:
    _HAS_GL = False


# ─────────────────────────────────────────────────────────────────────────────
# GLSL shaders
# ─────────────────────────────────────────────────────────────────────────────

_STATION_VS = """
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

_STATION_BODY_FS = """
#version 330 core
in  vec3 vNormV;
in  vec3 vPosV;
out vec4 FragColor;

uniform vec4  uColor;
uniform vec3  uLightV;
uniform float uAmbient;
uniform float uSpecStrength;
uniform float uShininess;
uniform float uGrain;

void main() {
    vec3  N    = normalize(gl_FrontFacing ? vNormV : -vNormV);
    vec3  L    = normalize(uLightV);
    vec3  V    = normalize(-vPosV);
    vec3  H    = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), max(uShininess, 1.0));
    vec3  base = uColor.rgb;
    float g    = 0.5 + 0.5 * sin(vPosV.x * 60.0 + vPosV.z * 40.0);
    base *= mix(1.0, 0.85 + 0.30 * g, uGrain);
    vec3  col  = base * (uAmbient + 0.78 * diff)
               + vec3(1.0, 0.90, 0.65) * uSpecStrength * spec;
    float rim  = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    col       += base * rim * 0.18;
    FragColor  = vec4(col, uColor.a);
}
"""

_STATION_SCREEN_FS = """
#version 330 core
in  vec3 vNormV;
in  vec3 vPosV;
out vec4 FragColor;

uniform vec4 uColor;
uniform vec3 uEmissive;

void main() {
    float facing = gl_FrontFacing ? 1.0 : 0.05;
    FragColor = vec4(uColor.rgb * facing + uEmissive * facing, uColor.a);
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Mesh generation
# ─────────────────────────────────────────────────────────────────────────────

def _quad_tris(v0, v1, v2, v3, normal) -> list:
    """Two CCW triangles for a planar quad. Returns list of (x,y,z,nx,ny,nz) tuples."""
    n = np.asarray(normal, np.float32)
    out = []
    for tri in [(v0, v1, v2), (v0, v2, v3)]:
        for v in tri:
            out.append((*np.asarray(v, np.float32).tolist(), *n.tolist()))
    return out


def _build_console_verts(cfg: dict) -> np.ndarray:
    """Console body mesh in local coords.  Returns (N, 6) float32."""
    w    = float(cfg.get('width',        1.40))
    d    = float(cfg.get('depth',        0.62))
    h    = float(cfg.get('height',       0.88))
    tilt = math.radians(float(cfg.get('top_tilt_deg', 14.0)))

    top_z_front = h
    top_z_back  = h + d * math.tan(tilt)
    top_n = np.array([0.0, -math.sin(tilt), math.cos(tilt)], np.float32)

    hw = w / 2.0
    b_fl = np.array([-hw, 0,  0])
    b_fr = np.array([ hw, 0,  0])
    b_bl = np.array([-hw, d,  0])
    b_br = np.array([ hw, d,  0])
    t_fl = np.array([-hw, 0,  top_z_front])
    t_fr = np.array([ hw, 0,  top_z_front])
    t_bl = np.array([-hw, d,  top_z_back])
    t_br = np.array([ hw, d,  top_z_back])

    verts = []
    # Front (facing -Y / player)
    verts += _quad_tris(b_fr, b_fl, t_fl, t_fr, [0, -1, 0])
    # Back
    verts += _quad_tris(b_bl, b_br, t_br, t_bl, [0,  1, 0])
    # Left
    verts += _quad_tris(b_fl, b_bl, t_bl, t_fl, [-1, 0, 0])
    # Right
    verts += _quad_tris(b_br, b_fr, t_fr, t_br, [ 1, 0, 0])
    # Bottom
    verts += _quad_tris(b_fl, b_fr, b_br, b_bl, [0, 0, -1])
    # Top (tilted)
    verts += _quad_tris(t_fl, t_fr, t_br, t_bl, top_n.tolist())

    return np.array(verts, np.float32).reshape(-1, 6)


def _build_screen_verts(cons_cfg: dict, scr_cfg: dict) -> np.ndarray:
    """Viewscreen panel mesh in local coords.  Returns (N, 6) float32."""
    d   = float(cons_cfg.get('depth',        0.62))
    h   = float(cons_cfg.get('height',       0.88))
    tlt = math.radians(float(cons_cfg.get('top_tilt_deg', 14.0)))

    sw  = float(scr_cfg.get('width',          1.10))
    sh  = float(scr_cfg.get('height',         0.72))
    ts  = math.radians(float(scr_cfg.get('tilt_back_deg', 7.0)))

    z_back = h + d * math.tan(tlt)   # Z of console back top edge
    hsw    = sw / 2.0

    # Screen corners in local space
    bl = np.array([-hsw, d, z_back])
    br = np.array([ hsw, d, z_back])
    tl = np.array([-hsw, d + sh * math.sin(ts), z_back + sh * math.cos(ts)])
    tr = np.array([ hsw, d + sh * math.sin(ts), z_back + sh * math.cos(ts)])

    # Normal: front face toward -Y (toward player)
    n_front = np.array([0.0, -math.cos(ts), math.sin(ts)])

    verts  = _quad_tris(bl, br, tr, tl, n_front.tolist())
    verts += _quad_tris(br, bl, tl, tr, (-n_front).tolist())  # back face

    return np.array(verts, np.float32).reshape(-1, 6)


def _build_wing_verts(cons_cfg: dict, wing_cfg: dict) -> np.ndarray:
    """Left + right side wing panels in local coords.  Returns (N, 6) float32."""
    w  = float(cons_cfg.get('width', 1.40))
    ww = float(wing_cfg.get('width', 0.14))
    wh = float(wing_cfg.get('height', 0.52))
    d  = float(cons_cfg.get('depth', 0.62))
    hw = w / 2.0

    verts = []
    for side, nx in [(-1, -1), (1, 1)]:
        x0 = side * hw
        x1 = side * (hw + ww)
        fl = np.array([x0, 0,  0])
        fr = np.array([x1, 0,  0])
        bl = np.array([x0, d,  0])
        br = np.array([x1, d,  0])
        ft = np.array([x0, 0,  wh])
        gt = np.array([x1, 0,  wh])
        bt = np.array([x0, d,  wh])
        ht = np.array([x1, d,  wh])

        # outer face
        if side == 1:
            verts += _quad_tris(fr, fl, ft, gt, [nx, 0, 0])
        else:
            verts += _quad_tris(fl, fr, gt, ft, [nx, 0, 0])
        # top cap
        verts += _quad_tris(ft, gt, ht, bt, [0, 0, 1])
        # front face
        verts += _quad_tris(fr, fl, ft, gt, [0, -1, 0])

    return np.array(verts, np.float32).reshape(-1, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Transform helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    M = np.eye(4, dtype=np.float64)
    M[0, 0] =  c;  M[0, 1] = -s
    M[1, 0] =  s;  M[1, 1] =  c
    return M


def _translation(xyz) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[0, 3] = xyz[0]
    M[1, 3] = xyz[1]
    M[2, 3] = xyz[2]
    return M


# ─────────────────────────────────────────────────────────────────────────────
# GL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compile_prog(vs_src: str, fs_src: str) -> int:
    vs = _gl_shaders.compileShader(vs_src, GL_VERTEX_SHADER)
    fs = _gl_shaders.compileShader(fs_src, GL_FRAGMENT_SHADER)
    return _gl_shaders.compileProgram(vs, fs)


def _make_vao(data: np.ndarray) -> tuple[int, int, int]:
    """Upload (N, 6) float32 pos+norm data.  Returns (vao, vbo, n_verts)."""
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data.tobytes(), GL_STATIC_DRAW)
    stride = 6 * 4  # 6 floats × 4 bytes
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glBindVertexArray(0)
    return vao, vbo, len(data)


def _set_mvp(prog: int, mvp: np.ndarray, mv: np.ndarray):
    loc = glGetUniformLocation(prog, b'uMVP')
    if loc >= 0:
        glUniformMatrix4fv(loc, 1, GL_TRUE, mvp.astype(np.float32))
    loc = glGetUniformLocation(prog, b'uMV')
    if loc >= 0:
        glUniformMatrix4fv(loc, 1, GL_TRUE, mv.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# DutyStation
# ─────────────────────────────────────────────────────────────────────────────

class DutyStation:
    """Procedural duty-station mesh with GL renderer and interaction state."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._gl_ready = False

        # World transform
        pos = cfg.get('position', [0.0, 2.4, 0.0])
        yaw = float(cfg.get('yaw_deg', 180.0))
        self.world_position = np.array(pos, np.float64)
        self._model_matrix  = _translation(pos) @ _rotation_z(yaw)

        self.interaction_radius = float(cfg.get('interaction_radius', 1.80))
        self.interact_camera    = cfg.get('interact_camera', {
            'eye':    [0.0, 0.70, 1.30],
            'target': [0.0, 2.40, 1.28],
        })
        self._screen_active = True   # toggleable

        # Generate mesh data (CPU side; GL upload deferred to build_gl)
        cons_cfg = cfg.get('console', {})
        scr_cfg  = cfg.get('screen',  {})
        wing_cfg = cfg.get('side_wings', {})

        self._body_data   = _build_console_verts(cons_cfg)
        self._screen_data = _build_screen_verts(cons_cfg, scr_cfg)
        self._wing_data   = (
            _build_wing_verts(cons_cfg, wing_cfg)
            if wing_cfg.get('enabled', True) else np.zeros((0, 6), np.float32)
        )

        # Material dicts
        mats = cfg.get('materials', {})
        self._mat_body   = mats.get('body',          {})
        self._mat_screen = mats.get('screen_active',  {})
        self._mat_scr_off= mats.get('screen_inactive',{})

        # GL handles (populated by build_gl)
        self._body_vao    = self._body_vbo    = self._body_n    = None
        self._screen_vao  = self._screen_vbo  = self._screen_n  = None
        self._wing_vao    = self._wing_vbo    = self._wing_n    = None
        self._prog_body   = self._prog_screen = None

    # ── Class-method constructors ─────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "DutyStation":
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is required to load duty_station.yaml")
        with open(path, 'r', encoding='utf-8') as fh:
            cfg = _yaml.safe_load(fh) or {}
        return cls(cfg)

    @classmethod
    def from_yaml_safe(cls, path: str) -> Optional["DutyStation"]:
        try:
            return cls.from_yaml(path)
        except Exception as exc:
            print(f"[duty_station] could not load {path}: {exc}", flush=True)
            return None

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def build_gl(self):
        """Upload mesh to GPU and compile shaders.  Call after GL context ready."""
        if not _HAS_GL:
            return
        self._prog_body   = _compile_prog(_STATION_VS, _STATION_BODY_FS)
        self._prog_screen = _compile_prog(_STATION_VS, _STATION_SCREEN_FS)

        self._body_vao,   self._body_vbo,   self._body_n   = _make_vao(self._body_data)
        self._screen_vao, self._screen_vbo, self._screen_n = _make_vao(self._screen_data)
        if len(self._wing_data):
            self._wing_vao, self._wing_vbo, self._wing_n = _make_vao(self._wing_data)
        else:
            self._wing_n = 0

        self._gl_ready = True

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw(self, cam_mvp: np.ndarray, cam_mv: np.ndarray,
             light_v: np.ndarray, alpha: float = 1.0):
        """Draw console body + wings (Phong shaded) and screen (emissive).
        cam_mvp / cam_mv should already incorporate the camera but NOT the
        station model matrix — this method applies the model transform."""
        if not self._gl_ready:
            return

        M   = self._model_matrix.astype(np.float64)
        MVP = (cam_mvp @ M).astype(np.float32)
        MV  = (cam_mv  @ M).astype(np.float32)

        self._draw_body(MVP, MV, light_v, alpha)
        self._draw_screen(MVP, MV, alpha)

    def _draw_body(self, MVP, MV, light_v, alpha):
        m = self._mat_body
        r, g, b = m.get('albedo_rgb', [0.07, 0.09, 0.13])

        glUseProgram(self._prog_body)
        _set_mvp(self._prog_body, MVP, MV)
        glUniform3f(glGetUniformLocation(self._prog_body, b'uLightV'),    *light_v)
        glUniform4f(glGetUniformLocation(self._prog_body, b'uColor'),     r, g, b, alpha)
        glUniform1f(glGetUniformLocation(self._prog_body, b'uAmbient'),   m.get('ambient',       0.18))
        glUniform1f(glGetUniformLocation(self._prog_body, b'uSpecStrength'), m.get('spec_strength', 0.55))
        glUniform1f(glGetUniformLocation(self._prog_body, b'uShininess'), m.get('shininess',     112.0))
        glUniform1f(glGetUniformLocation(self._prog_body, b'uGrain'),     m.get('grain',          0.05))

        glBindVertexArray(self._body_vao)
        glDrawArrays(GL_TRIANGLES, 0, self._body_n)

        if self._wing_n:
            glBindVertexArray(self._wing_vao)
            glDrawArrays(GL_TRIANGLES, 0, self._wing_n)

        glBindVertexArray(0)
        glUseProgram(0)

    def _draw_screen(self, MVP, MV, alpha):
        mat = self._mat_screen if self._screen_active else self._mat_scr_off
        r, g, b = mat.get('albedo_rgb', [0.02, 0.04, 0.08])
        er, eg, eb = mat.get('emissive', [0.0, 0.0, 0.0])

        glUseProgram(self._prog_screen)
        _set_mvp(self._prog_screen, MVP, MV)
        glUniform4f(glGetUniformLocation(self._prog_screen, b'uColor'),    r, g, b, alpha)
        glUniform3f(glGetUniformLocation(self._prog_screen, b'uEmissive'), er, eg, eb)

        glBindVertexArray(self._screen_vao)
        glDrawArrays(GL_TRIANGLES, 0, self._screen_n)
        glBindVertexArray(0)
        glUseProgram(0)

    # ── Interaction helpers ───────────────────────────────────────────────────

    def player_near(self, player_eye: np.ndarray) -> bool:
        return (float(np.linalg.norm(player_eye - self.world_position))
                < self.interaction_radius)

    def set_screen_active(self, active: bool):
        self._screen_active = active

    def toggle_screen(self):
        self._screen_active = not self._screen_active


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a camera pure-view matrix (no model) for use in draw()
# ─────────────────────────────────────────────────────────────────────────────

def camera_pure_matrices(camera) -> tuple[np.ndarray, np.ndarray]:
    """Return (P, V) from a Camera instance so you can compose MVP = P @ V @ M.

    The camera's mvp(aspect) already folds in the identity model, so we
    reconstruct P from the camera's fov and V from the view matrix.
    Returns float64 for composition precision.
    """
    import math
    w, h = __import__('pygame').display.get_surface().get_size()
    aspect = w / max(1, h)
    fov    = camera.fov_y_rad()
    near, far = 0.005, 10.0

    f  = 1.0 / math.tan(fov / 2.0)
    P  = np.zeros((4, 4), np.float64)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0

    V = camera._view().astype(np.float64)
    return P, V
