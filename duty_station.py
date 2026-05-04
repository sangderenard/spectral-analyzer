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
    from depth_mesh import DepthMesh as _DepthMesh
    _HAS_DEPTH_MESH = True
except ImportError:
    _DepthMesh = None
    _HAS_DEPTH_MESH = False

try:
    from spectral_material import (
        Material              as _SpectralMaterial,
        EnamelCoating         as _EnamelCoating,
        RadianceProfile       as _RadianceProfile,
        LightSource           as _LightSource,
        LightDistributionPolicy as _LightDistributionPolicy,
        set_spectral_uniforms as _set_spectral_uniforms,
        SPECTRAL_PBR_BODY_FS  as _SPECTRAL_PBR_BODY_FS,
        parse_wall_bands      as _parse_wall_bands,
        MATERIAL_PRESETS      as _MATERIAL_PRESETS,
        ENAMEL_PRESETS        as _ENAMEL_PRESETS,
    )
    _HAS_SPECTRAL = True
except ImportError:
    _SpectralMaterial        = None
    _EnamelCoating           = None
    _RadianceProfile         = None
    _LightSource             = None
    _LightDistributionPolicy = None
    _set_spectral_uniforms   = None
    _SPECTRAL_PBR_BODY_FS    = None
    _parse_wall_bands        = None
    _MATERIAL_PRESETS        = {}
    _ENAMEL_PRESETS          = {}
    _HAS_SPECTRAL            = False

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _yaml = None
    _HAS_YAML = False

try:
    from material_db import MaterialDatabase as _MaterialDatabase
    _HAS_MAT_DB = True
except ImportError:
    _MaterialDatabase = None  # type: ignore[assignment,misc]
    _HAS_MAT_DB = False

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


def _build_back_wall_verts(cons_cfg: dict, scr_cfg: dict,
                           wing_cfg: dict, wall_cfg: dict) -> np.ndarray:
    """Wall behind the screen, floor-to-ceiling, full console+wing width."""
    if not wall_cfg.get('enabled', False):
        return np.zeros((0, 6), np.float32)
    w   = float(cons_cfg.get('width',        1.40))
    d   = float(cons_cfg.get('depth',        0.62))
    tlt = math.radians(float(cons_cfg.get('top_tilt_deg', 14.0)))
    sh  = float(scr_cfg.get('height',         0.72))
    ts  = math.radians(float(scr_cfg.get('tilt_back_deg', 7.0)))
    ww  = float(wing_cfg.get('width',  0.14)) if wing_cfg.get('enabled', True) else 0.0
    wall_h = float(wall_cfg.get('height', 4.0))
    wall_t = float(wall_cfg.get('thickness', 0.05))

    hw   = w / 2.0 + ww
    y_back = d + sh * math.sin(ts) + wall_t   # just behind screen top edge

    bl = np.array([-hw, y_back, 0.0])
    br = np.array([ hw, y_back, 0.0])
    tl = np.array([-hw, y_back, wall_h])
    tr = np.array([ hw, y_back, wall_h])

    n = [0.0, -1.0, 0.0]   # faces toward player
    verts  = _quad_tris(bl, br, tr, tl, n)
    verts += _quad_tris(br, bl, tl, tr, [0.0, 1.0, 0.0])  # back face
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
# Module-specific geometry builders
# ─────────────────────────────────────────────────────────────────────────────

def _pbr_to_phong(mat: dict) -> dict:
    """Approximate conversion from a PBR material dict to Phong rasteriser params.

    Accepts both old-style (albedo_rgb, ambient, …) and new PBR-style
    (albedo, roughness, metallic, …) dicts and returns a unified dict with
    the keys expected by the Phong shader uniforms.
    """
    # Colour — accept 'albedo' (new) or 'albedo_rgb' (legacy)
    albedo = mat.get('albedo', mat.get('albedo_rgb', [0.5, 0.5, 0.5]))
    rough  = float(mat.get('roughness', 0.5))
    metal  = float(mat.get('metallic',  0.0))
    # If the mat already has explicit Phong keys, honour them
    ambient  = mat.get('ambient',       0.15 + (1.0 - rough) * 0.05)
    specstr  = mat.get('spec_strength', metal * 0.85 + (1.0 - metal) * 0.04 * (1.0 - rough))
    shiny    = mat.get('shininess',     max(4.0, 2.0 / max(rough ** 2, 0.01)))
    grain    = mat.get('grain',         rough * 0.06)
    return {
        'albedo_rgb':    albedo,
        'ambient':       float(ambient),
        'spec_strength': float(specstr),
        'shininess':     float(shiny),
        'grain':         float(grain),
    }


def _build_cornerstone_wall_mesh(cfg: dict) -> Optional["_DepthMesh"]:
    """Build a DepthMesh for the cornerstone wall behind the screen.

    Uses DepthMesh.wall() so the caller can select flat / arc / vertical_arc
    / spherical via the YAML shape_type key.  The DepthMesh origin is
    bottom_left so callers can position by translating.
    """
    if not _HAS_DEPTH_MESH:
        return None
    cons_cfg = cfg.get('console', {})
    wing_cfg = cfg.get('side_wings', {})
    cw_cfg   = cfg.get('cornerstone_wall', {})

    if not cw_cfg.get('enabled', False):
        return None

    w  = float(cons_cfg.get('width', 1.40))
    ww = float(wing_cfg.get('width', 0.14)) if wing_cfg.get('enabled', True) else 0.0

    wall_w      = float(cw_cfg.get('width',  w + 2 * ww))
    wall_h      = float(cw_cfg.get('height', 3.20))
    gu          = int(cw_cfg.get('grid_u', 16))
    gv          = int(cw_cfg.get('grid_v', 24))
    dscale      = float(cw_cfg.get('depth_scale', 0.08))
    origin      = cw_cfg.get('origin', 'bottom_left')
    shape_type  = cw_cfg.get('shape_type', 'flat')

    # Radius — select the key matching shape_type, fall back to generic arc_radius
    _radius_key = {
        'arc':          'arc_radius',
        'vertical_arc': 'vertical_arc_radius',
        'spherical':    'spherical_radius',
    }.get(shape_type, 'arc_radius')
    arc_radius = float(cw_cfg.get(_radius_key, cw_cfg.get('arc_radius', 8.0)))

    dm = None
    dm_path = cw_cfg.get('depth_map_path')
    if dm_path:
        try:
            from depth_mesh import _load_depth_map
            dm = _load_depth_map(dm_path)
        except Exception as exc:
            print(f'[duty_station] cornerstone_wall depth_map load failed: {exc}')

    return _DepthMesh.wall(
        shape_type=shape_type,
        width=wall_w,
        height=wall_h,
        grid_u=gu,
        grid_v=gv,
        depth_map=dm,
        depth_scale=dscale,
        origin=origin,
        arc_radius=arc_radius,
    )


def _build_floor_tile_mesh(cfg: dict) -> Optional["_DepthMesh"]:
    """Build a DepthMesh for the floor tile under the console.

    Horizontal tile: axis_u = +X, axis_v = +Y (depth into room), normal = +Z.
    Origin = bottom_left (front-left corner at console front edge).
    """
    if not _HAS_DEPTH_MESH:
        return None
    ft_cfg = cfg.get('floor_tile', {})
    if not ft_cfg.get('enabled', False):
        return None

    tile_w = float(ft_cfg.get('width', 2.00))
    tile_d = float(ft_cfg.get('depth', 2.20))
    gu     = int(ft_cfg.get('grid_u', 8))
    gv     = int(ft_cfg.get('grid_v', 8))
    dscale = float(ft_cfg.get('depth_scale', 0.02))
    origin = ft_cfg.get('origin', 'bottom_left')
    dm_path = ft_cfg.get('depth_map_path')

    dm = None
    if dm_path:
        try:
            from depth_mesh import _load_depth_map
            dm = _load_depth_map(dm_path)
        except Exception as exc:
            print(f'[duty_station] floor_tile depth_map load failed: {exc}')

    # Horizontal floor: axis_u = +X, axis_v = +Y, normal = +Z
    return _DepthMesh(
        shape='rect',
        size=(tile_w, tile_d),
        grid_u=gu, grid_v=gv,
        depth_map=dm,
        depth_scale=dscale,
        origin=origin,
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
    )


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
        self._yaw_deg = yaw
        self.world_position = np.array(pos, np.float64)
        self._model_matrix  = _translation(pos) @ _rotation_z(yaw)

        self.interaction_radius = float(cfg.get('interaction_radius', 1.80))
        self.interact_camera    = cfg.get('interact_camera', {
            'eye':    [0.0, 0.70, 1.30],
            'target': [0.0, 2.40, 1.28],
        })
        self._screen_active = True   # toggleable
        self.menu           = None   # attach any object with show_hud/render_hud/handle_event

        # Module type — optional, controls extra geometry and menu class
        self.module_type: Optional[str] = cfg.get('module_type')

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
        wall_cfg = cfg.get('back_wall', {})
        self._wall_data   = _build_back_wall_verts(cons_cfg, scr_cfg, wing_cfg, wall_cfg)

        # ── Module-specific geometry ──────────────────────────────────────────
        self._cornerstone_wall: Optional[_DepthMesh] = None
        self._floor_tile:       Optional[_DepthMesh] = None
        if self.module_type == 'room_control' and _HAS_DEPTH_MESH:
            self._cornerstone_wall = _build_cornerstone_wall_mesh(cfg)
            self._floor_tile       = _build_floor_tile_mesh(cfg)

        # Material dicts — support both old-style 'materials' and new 'gl_materials' key
        gl_mats  = cfg.get('gl_materials', cfg.get('materials', {}))
        self._mat_body    = _pbr_to_phong(gl_mats.get('body',           {}))
        self._mat_screen  = _pbr_to_phong(gl_mats.get('screen_active',  {}))
        self._mat_scr_off = _pbr_to_phong(gl_mats.get('screen_inactive',{}))
        self._mat_wall    = _pbr_to_phong(gl_mats.get('back_wall', gl_mats.get('wall', {
            'albedo_rgb': [0.12, 0.12, 0.16],
            'ambient': 0.2, 'spec_strength': 0.1, 'shininess': 8.0})))

        # Per-piece PBR materials (for ray tracing / future deferred pass)
        self._piece_materials: dict = cfg.get('piece_materials', {})

        # Parse piece_materials into SpectralMaterial objects for shader upload
        self._spectral_materials: dict = {}
        if _HAS_SPECTRAL:
            for piece, mdict in self._piece_materials.items():
                try:
                    self._spectral_materials[piece] = _SpectralMaterial.from_dict(
                        mdict, name=piece)
                except Exception:
                    pass

        # Apply default thin clear-coat enamel to all settable surfaces, and
        # ensure the monitor is always emissive blue.
        if _HAS_SPECTRAL:
            _default_enamel = _EnamelCoating(
                thickness_m=80e-9, ior_real=1.52, ior_imag=0.0,
                color_rgb=[1.0, 1.0, 1.0], roughness=0.06)
            for piece, mat in self._spectral_materials.items():
                if mat.enamel is None:
                    mat.enamel = _default_enamel
            # Monitor: force emissive blue if not already emissive
            mon = self._spectral_materials.get('monitor')
            if mon is not None:
                if max(mon.emission_rgb) < 0.05:
                    mon.emission_rgb = [0.05, 0.28, 0.82]
                # Blue enamel on screen surface
                if mon.enamel is None or mon.enamel.color_rgb == [1.0, 1.0, 1.0]:
                    mon.enamel = _EnamelCoating(
                        thickness_m=120e-9, ior_real=1.52, ior_imag=0.0,
                        color_rgb=[0.20, 0.55, 1.0], roughness=0.03)

        # Module geometry materials — derive Phong params from wall bands / floor PBR
        cw_cfg = cfg.get('cornerstone_wall', {})
        ft_cfg = cfg.get('floor_tile', {})

        # Wall bands (PBR per zone, stored raw for ray-tracing consumers)
        self._wall_bands: list = cw_cfg.get('bands', [])
        # Parse into WallBand objects for spectral system
        self._wall_bands_spectral: list = []
        if _HAS_SPECTRAL and self._wall_bands:
            try:
                self._wall_bands_spectral = _parse_wall_bands(self._wall_bands)
            except Exception:
                pass
        # GL fast-path: use the mid-wall band albedo; fall back to old 'material' key
        _cw_mid_mat = {}
        if self._wall_bands:
            mid_idx = len(self._wall_bands) // 2
            _cw_mid_mat = self._wall_bands[mid_idx].get('material', {})
        elif 'material' in cw_cfg:
            _cw_mid_mat = cw_cfg['material']
        self._mat_cornerstone = _pbr_to_phong(_cw_mid_mat or {
            'albedo': [0.10, 0.11, 0.15], 'roughness': 0.77, 'metallic': 0.03})

        # Floor tile material
        _ft_mat = ft_cfg.get('material', {})
        self._mat_floor_tile = _pbr_to_phong(_ft_mat or {
            'albedo': [0.10, 0.12, 0.16], 'roughness': 0.85, 'metallic': 0.05})
        self._spectral_floor: Optional[object] = None
        if _HAS_SPECTRAL and _ft_mat:
            try:
                self._spectral_floor = _SpectralMaterial.from_dict(_ft_mat, name="floor_tile")
            except Exception:
                pass
        # Cornerstone mid-wall spectral material (for GL uniform upload)
        self._spectral_cornerstone: Optional[object] = None
        if _HAS_SPECTRAL and self._wall_bands_spectral:
            mid = len(self._wall_bands_spectral) // 2
            self._spectral_cornerstone = self._wall_bands_spectral[mid].material

        # ── Register all materials with the global database ───────────────────
        if _HAS_MAT_DB:
            _db = _MaterialDatabase.instance()
            _prefix = f"duty_station.{id(self)}"
            # Per-piece spectral materials (richest path)
            for _pname, _mat in self._spectral_materials.items():
                _db.register(f"{_prefix}.{_pname}", _mat)
            # Phong-derived materials for pieces without spectral objects
            for _pname, _phong in [
                ('body',            self._mat_body),
                ('screen_active',   self._mat_screen),
                ('screen_inactive', self._mat_scr_off),
                ('back_wall',       self._mat_wall),
                ('cornerstone',     self._mat_cornerstone),
                ('floor_tile',      self._mat_floor_tile),
            ]:
                _full = f"{_prefix}.{_pname}"
                if _full not in _db:
                    _db.register(_full, _phong)
            # Floor and cornerstone spectral materials if available
            if self._spectral_floor is not None:
                _db.register(f"{_prefix}.floor_tile", self._spectral_floor)
            if self._spectral_cornerstone is not None:
                _db.register(f"{_prefix}.cornerstone", self._spectral_cornerstone)

        # GL handles (populated by build_gl)
        self._body_vao    = self._body_vbo    = self._body_n    = None
        self._screen_vao  = self._screen_vbo  = self._screen_n  = None
        self._wing_vao    = self._wing_vbo    = self._wing_n    = None
        self._wall_vao    = self._wall_vbo    = self._wall_n    = None
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
        # Prefer spectral-PBR shader when spectral_material is available;
        # fall back to legacy Phong shader otherwise.
        body_fs = _SPECTRAL_PBR_BODY_FS if _HAS_SPECTRAL else _STATION_BODY_FS
        self._prog_body   = _compile_prog(_STATION_VS, body_fs)
        self._prog_screen = _compile_prog(_STATION_VS, _STATION_SCREEN_FS)
        self._use_spectral_shader = _HAS_SPECTRAL

        self._body_vao,   self._body_vbo,   self._body_n   = _make_vao(self._body_data)
        self._screen_vao, self._screen_vbo, self._screen_n = _make_vao(self._screen_data)
        if len(self._wing_data):
            self._wing_vao, self._wing_vbo, self._wing_n = _make_vao(self._wing_data)
        else:
            self._wing_n = 0
        self._wall_vao, self._wall_vbo, self._wall_n = _make_vao(self._wall_data)

        # Module geometry
        if self._cornerstone_wall is not None:
            self._cornerstone_wall.build_gl()
        if self._floor_tile is not None:
            self._floor_tile.build_gl()

        # Auto-attach RoomControlStation menu if not already set
        if self.module_type == 'room_control' and self.menu is None:
            self._try_attach_room_control_menu()
        if self.menu is not None and hasattr(self.menu, 'build_gl'):
            self.menu.build_gl()

        self._gl_ready = True

    def _try_attach_room_control_menu(self):
        """Load and attach a RoomControlStation from the module YAML folder."""
        try:
            from room_control_station import RoomControlStation
            # Resolve the station YAML path relative to configs/duty_stations/room_control/
            candidates = [
                os.path.join(os.path.dirname(__file__),
                             'configs', 'duty_stations', 'room_control', 'station.yaml'),
                'configs/duty_stations/room_control/station.yaml',
            ]
            for path in candidates:
                if os.path.isfile(path):
                    self.menu = RoomControlStation.from_yaml_safe(path)
                    if self.menu is not None and hasattr(self.menu, 'bind_host_station'):
                        self.menu.bind_host_station(self)
                    return
            print('[duty_station] room_control: station.yaml not found; no menu attached')
        except Exception as exc:
            print(f'[duty_station] room_control menu attach failed: {exc}')

    def set_world_position(self, new_pos: np.ndarray, update_interact_camera: bool = True):
        """Update station world position and dependent transforms at runtime."""
        new_pos = np.asarray(new_pos, np.float64).reshape(3)
        delta = new_pos - self.world_position
        if float(np.linalg.norm(delta)) <= 1e-12:
            return
        self.world_position = new_pos
        self._model_matrix = _translation(self.world_position.tolist()) @ _rotation_z(self._yaw_deg)
        self._cfg['position'] = self.world_position.tolist()

        if update_interact_camera:
            for key in ('eye', 'target'):
                v = self.interact_camera.get(key)
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    self.interact_camera[key] = (np.asarray(v, np.float64) + delta).tolist()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw(self, cam_mvp: np.ndarray, cam_mv: np.ndarray,
             light_v: np.ndarray, alpha: float = 1.0,
             body_prog: Optional[int] = None):
        """Draw console body + wings (Phong shaded) and screen (emissive).
        cam_mvp / cam_mv should already incorporate the camera but NOT the
        station model matrix — this method applies the model transform.

        body_prog: when provided, the station uses this external OpenGL program
        handle (e.g. the main renderer's _p_body) instead of its own private
        shader.  This makes it participate in the same lighting model as all
        other scene objects.  The program must accept the same uniforms as
        _BODY_FS / _BODY_VS (uMVP, uMV, uColor, uInnerColor, uLightV,
        uAmbient, uSpecStrength, uShininess, uGrain, uSceneRgb,
        uSceneIndirectRatio).
        """
        if not self._gl_ready:
            return

        M   = self._model_matrix.astype(np.float64)
        MVP = (cam_mvp @ M).astype(np.float32)
        MV  = (cam_mv  @ M).astype(np.float32)

        prog = body_prog if body_prog is not None else self._prog_body
        self._draw_body(MVP, MV, light_v, alpha, prog)
        self._draw_wall(MVP, MV, light_v, alpha, prog)
        self._draw_screen(MVP, MV, alpha)
        if self.menu is not None and hasattr(self.menu, 'draw_world'):
            self.menu.draw_world(MVP, MV, light_v, prog)

    def _draw_body(self, MVP, MV, light_v, alpha, prog: Optional[int] = None):
        m = self._mat_body
        r, g, b = m.get('albedo_rgb', [0.07, 0.09, 0.13])
        if prog is None:
            prog = self._prog_body

        glUseProgram(prog)
        _set_mvp(prog, MVP, MV)
        glUniform3f(glGetUniformLocation(prog, b'uLightV'),    *light_v)
        glUniform4f(glGetUniformLocation(prog, b'uColor'),     r, g, b, alpha)
        glUniform3f(glGetUniformLocation(prog, b'uInnerColor'), r, g, b)
        glUniform1f(glGetUniformLocation(prog, b'uAmbient'),   m.get('ambient',       0.18))
        glUniform1f(glGetUniformLocation(prog, b'uSpecStrength'), m.get('spec_strength', 0.55))
        glUniform1f(glGetUniformLocation(prog, b'uShininess'), m.get('shininess',     112.0))
        glUniform1f(glGetUniformLocation(prog, b'uGrain'),     m.get('grain',          0.05))
        if getattr(self, '_use_spectral_shader', False) and prog == self._prog_body:
            spec_mat = self._spectral_materials.get('panels',
                       self._spectral_materials.get('main_surface', None))
            if spec_mat is not None:
                _set_spectral_uniforms(prog, spec_mat,
                                       probe_freq_norm=0.5, spec_tint_strength=0.12)

        glBindVertexArray(self._body_vao)
        glDrawArrays(GL_TRIANGLES, 0, self._body_n)

        if self._wing_n:
            wing_mat = (self._spectral_materials.get('left_wing', None)
                        if getattr(self, '_use_spectral_shader', False) and prog == self._prog_body
                        else None)
            if wing_mat is not None:
                _set_spectral_uniforms(prog, wing_mat,
                                       probe_freq_norm=0.5, spec_tint_strength=0.10)
            glBindVertexArray(self._wing_vao)
            glDrawArrays(GL_TRIANGLES, 0, self._wing_n)

        glBindVertexArray(0)
        glUseProgram(0)

    def _draw_wall(self, MVP, MV, light_v, alpha, prog: Optional[int] = None):
        if not self._wall_n:
            return
        if prog is None:
            prog = self._prog_body
        m = self._mat_wall
        r, g, b = m.get('albedo_rgb', [0.12, 0.12, 0.16])
        glUseProgram(prog)
        _set_mvp(prog, MVP, MV)
        glUniform3f(glGetUniformLocation(prog, b'uLightV'),       *light_v)
        glUniform4f(glGetUniformLocation(prog, b'uColor'),        r, g, b, alpha)
        glUniform3f(glGetUniformLocation(prog, b'uInnerColor'),   r, g, b)
        glUniform1f(glGetUniformLocation(prog, b'uAmbient'),      m.get('ambient',       0.20))
        glUniform1f(glGetUniformLocation(prog, b'uSpecStrength'), m.get('spec_strength', 0.10))
        glUniform1f(glGetUniformLocation(prog, b'uShininess'),    m.get('shininess',     8.0))
        glUniform1f(glGetUniformLocation(prog, b'uGrain'),        m.get('grain',         0.02))
        if getattr(self, '_use_spectral_shader', False) and prog == self._prog_body and self._spectral_cornerstone:
            _set_spectral_uniforms(prog, self._spectral_cornerstone,
                                   probe_freq_norm=0.3, spec_tint_strength=0.08)
        glBindVertexArray(self._wall_vao)
        glDrawArrays(GL_TRIANGLES, 0, self._wall_n)
        glBindVertexArray(0)
        glUseProgram(0)

    def _draw_module_geometry(self, MVP, MV, light_v, alpha, prog: Optional[int] = None):
        """Draw cornerstone wall + floor tile when module_type == 'room_control'."""
        _body_prog = prog if prog is not None else self._prog_body
        if self._cornerstone_wall is not None and self._cornerstone_wall._gl_ready:
            m = self._mat_cornerstone
            r, g, b = m.get('albedo_rgb', [0.10, 0.11, 0.15])
            _spec_cw = (self._spectral_cornerstone
                        if getattr(self, '_use_spectral_shader', False) and prog is None
                        else None)

            def _cw_uniforms(_p):
                from OpenGL.GL import glUniform3f, glUniform4f, glUniform1f, glGetUniformLocation
                glUniform3f(glGetUniformLocation(_p, b'uLightV'),       *light_v)
                glUniform4f(glGetUniformLocation(_p, b'uColor'),        r, g, b, alpha)
                glUniform3f(glGetUniformLocation(_p, b'uInnerColor'),   r, g, b)
                glUniform1f(glGetUniformLocation(_p, b'uAmbient'),      m.get('ambient',       0.22))
                glUniform1f(glGetUniformLocation(_p, b'uSpecStrength'), m.get('spec_strength', 0.12))
                glUniform1f(glGetUniformLocation(_p, b'uShininess'),    m.get('shininess',     12.0))
                glUniform1f(glGetUniformLocation(_p, b'uGrain'),        m.get('grain',          0.04))
                if _spec_cw is not None:
                    _set_spectral_uniforms(_p, _spec_cw,
                                           probe_freq_norm=0.4, spec_tint_strength=0.08)

            self._cornerstone_wall.draw(MVP, MV, _body_prog, _cw_uniforms)

        if self._floor_tile is not None and self._floor_tile._gl_ready:
            m = self._mat_floor_tile
            r, g, b = m.get('albedo_rgb', [0.08, 0.10, 0.14])
            _spec_ft = (self._spectral_floor
                        if getattr(self, '_use_spectral_shader', False) and prog is None
                        else None)

            def _ft_uniforms(_p):
                from OpenGL.GL import glUniform3f, glUniform4f, glUniform1f, glGetUniformLocation
                glUniform3f(glGetUniformLocation(_p, b'uLightV'),       *light_v)
                glUniform4f(glGetUniformLocation(_p, b'uColor'),        r, g, b, alpha)
                glUniform3f(glGetUniformLocation(_p, b'uInnerColor'),   r, g, b)
                glUniform1f(glGetUniformLocation(_p, b'uAmbient'),      m.get('ambient',       0.20))
                glUniform1f(glGetUniformLocation(_p, b'uSpecStrength'), m.get('spec_strength', 0.35))
                glUniform1f(glGetUniformLocation(_p, b'uShininess'),    m.get('shininess',     40.0))
                glUniform1f(glGetUniformLocation(_p, b'uGrain'),        m.get('grain',          0.08))
                if _spec_ft is not None:
                    _set_spectral_uniforms(_p, _spec_ft,
                                           probe_freq_norm=0.2, spec_tint_strength=0.06)

            self._floor_tile.draw(MVP, MV, _body_prog, _ft_uniforms)

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

    # ── Menu slot ─────────────────────────────────────────────────────────────
    # Any object with show_hud(bool), render_hud(w,h), handle_event(ev)->bool
    # can be attached here.  The room-station HUD is one example.

    def open_menu(self):
        if self.menu is not None:
            self.menu.show_hud(True)

    def close_menu(self):
        if self.menu is not None:
            self.menu.show_hud(False)

    def render_menu(self, win_w: int, win_h: int):
        if self.menu is not None:
            self.menu.render_hud(win_w, win_h)

    def handle_menu_event(self, ev) -> bool:
        if self.menu is not None and getattr(self.menu, '_hud_visible', False):
            return self.menu.handle_event(ev)
        return False


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
    near, far = 0.05, 500.0

    f  = 1.0 / math.tan(fov / 2.0)
    P  = np.zeros((4, 4), np.float64)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0

    V = camera._view().astype(np.float64)
    return P, V
