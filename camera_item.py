"""camera_item.py
=================
CameraItem — GL-renderable physical camera on an armature.

Each CameraItem wraps a PlacedCamera and adds:
  * A procedural body mesh (box + lens barrel)
  * A GL VAO/VBO for Phong rendering in the room pass
  * An armature transform (pan, tilt) that is visualised live
  * ``interaction_radius`` / ``world_position`` properties so
    PlayerController._nearest_camera() can find it by proximity

The mesh is generated from the PlacedCamera's ``mesh_id`` preset.
New presets can be registered via ``register_mesh_preset(id, builder_fn)``.

No game logic here — pan_deg / tilt_deg / zoom_fov_deg are plain
float attributes written by PlayerController._tick_camera() and read
by _update_camera_view().
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_BLEND, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER,
        GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_STATIC_DRAW,
        GL_TRIANGLES, GL_TRUE, GL_VERTEX_SHADER,
        glBindBuffer, glBindVertexArray, glBlendFunc, glBufferData,
        glDeleteBuffers, glDeleteVertexArrays, glDepthMask,
        glDisable, glDrawArrays, glEnable, glEnableVertexAttribArray,
        glGenBuffers, glGenVertexArrays,
        glGetUniformLocation, glUniform1f, glUniform3f, glUniformMatrix4fv,
        glUseProgram, glVertexAttribPointer,
    )
    from OpenGL.GL import shaders as _gl_shaders
    _HAS_GL = True
except ImportError:
    _HAS_GL = False


# ─────────────────────────────────────────────────────────────────────────────
# Procedural mesh helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quad(p00, p01, p10, p11) -> list:
    return [p00, p10, p11, p00, p11, p01]


def _vn(pos, n) -> list:
    return [float(pos[0]), float(pos[1]), float(pos[2]),
            float(n[0]),   float(n[1]),   float(n[2])]


def _box(cx: float, cy: float, cz: float,
         hw: float, hh: float, hd: float) -> np.ndarray:
    """Axis-aligned box centred at (cx,cy,cz), half-extents (hw,hh,hd)."""
    rows: list = []
    # ±X faces
    for sx, nx in ((-1, -1), (1, 1)):
        x = cx + sx * hw
        rows += _quad(
            _vn([x, cy - hh, cz - hd], [nx, 0, 0]),
            _vn([x, cy + hh, cz - hd], [nx, 0, 0]),
            _vn([x, cy - hh, cz + hd], [nx, 0, 0]),
            _vn([x, cy + hh, cz + hd], [nx, 0, 0]),
        )
    # ±Y faces
    for sy, ny in ((-1, -1), (1, 1)):
        y = cy + sy * hh
        rows += _quad(
            _vn([cx - hw, y, cz - hd], [0, ny, 0]),
            _vn([cx + hw, y, cz - hd], [0, ny, 0]),
            _vn([cx - hw, y, cz + hd], [0, ny, 0]),
            _vn([cx + hw, y, cz + hd], [0, ny, 0]),
        )
    # ±Z faces
    for sz, nz in ((-1, -1), (1, 1)):
        z = cz + sz * hd
        rows += _quad(
            _vn([cx - hw, cy - hh, z], [0, 0, nz]),
            _vn([cx + hw, cy - hh, z], [0, 0, nz]),
            _vn([cx - hw, cy + hh, z], [0, 0, nz]),
            _vn([cx + hw, cy + hh, z], [0, 0, nz]),
        )
    return np.array(rows, np.float32).reshape(-1, 6)


def _box_open_back(cx: float, cy: float, cz: float,
                   hw: float, hh: float, hd: float) -> np.ndarray:
    """Box without the -Z face — open at the back for plate/projector cameras."""
    rows: list = []
    # ±X faces
    for sx, nx in ((-1, -1), (1, 1)):
        x = cx + sx * hw
        rows += _quad(
            _vn([x, cy - hh, cz - hd], [nx, 0, 0]),
            _vn([x, cy + hh, cz - hd], [nx, 0, 0]),
            _vn([x, cy - hh, cz + hd], [nx, 0, 0]),
            _vn([x, cy + hh, cz + hd], [nx, 0, 0]),
        )
    # ±Y faces
    for sy, ny in ((-1, -1), (1, 1)):
        y = cy + sy * hh
        rows += _quad(
            _vn([cx - hw, y, cz - hd], [0, ny, 0]),
            _vn([cx + hw, y, cz - hd], [0, ny, 0]),
            _vn([cx - hw, y, cz + hd], [0, ny, 0]),
            _vn([cx + hw, y, cz + hd], [0, ny, 0]),
        )
    # +Z face only (front wall toward lens; -Z rear omitted)
    z = cz + hd
    rows += _quad(
        _vn([cx - hw, cy - hh, z], [0, 0, 1]),
        _vn([cx + hw, cy - hh, z], [0, 0, 1]),
        _vn([cx - hw, cy + hh, z], [0, 0, 1]),
        _vn([cx + hw, cy + hh, z], [0, 0, 1]),
    )
    return np.array(rows, np.float32).reshape(-1, 6)


def _cylinder(cx: float, cy: float, cz: float,
              radius: float, half_len: float,
              axis: str = 'z', segs: int = 16) -> np.ndarray:
    """Closed cylinder cap, axis in 'x'/'y'/'z'."""
    rows: list = []
    angles = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    for i in range(segs):
        a0, a1 = angles[i], angles[(i + 1) % segs]
        if axis == 'z':
            def _pt(a, s): return [cx + math.cos(a) * radius,
                                   cy + math.sin(a) * radius,
                                   cz + s * half_len]
            def _ptn(a, s): return _pt(a, s) + [math.cos(a), math.sin(a), 0]
        elif axis == 'x':
            def _pt(a, s): return [cx + s * half_len,
                                   cy + math.cos(a) * radius,
                                   cz + math.sin(a) * radius]
            def _ptn(a, s): return _pt(a, s) + [0, math.cos(a), math.sin(a)]
        else:
            def _pt(a, s): return [cx + math.cos(a) * radius,
                                   cy + s * half_len,
                                   cz + math.sin(a) * radius]
            def _ptn(a, s): return _pt(a, s) + [math.cos(a), 0, math.sin(a)]

        rows += [_ptn(a0, -1), _ptn(a1, -1), _ptn(a1, 1),
                 _ptn(a0, -1), _ptn(a1,  1), _ptn(a0, 1)]
    return np.array(rows, np.float32).reshape(-1, 6)


def _disk_cap(cx: float, cy: float, cz: float,
             radius: float, normal_z: float = 1.0,
             segs: int = 20) -> np.ndarray:
    """Filled circular disc at z=cz facing +z or -z."""
    rows: list = []
    angles = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    n = [0.0, 0.0, float(normal_z)]
    ctr = [cx, cy, cz]
    for i in range(segs):
        a0, a1 = angles[i], angles[(i + 1) % segs]
        p0 = [cx + math.cos(a0) * radius, cy + math.sin(a0) * radius, cz]
        p1 = [cx + math.cos(a1) * radius, cy + math.sin(a1) * radius, cz]
        if normal_z >= 0:
            rows += [_vn(ctr, n), _vn(p0, n), _vn(p1, n)]
        else:
            rows += [_vn(ctr, n), _vn(p1, n), _vn(p0, n)]
    return np.array(rows, np.float32).reshape(-1, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh presets  (body, tube, glass_caps) — each (-1, 6)
# ─────────────────────────────────────────────────────────────────────────────

def _mesh_camera_35mm() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """35mm SLR-style camera."""
    body  = _box(0, 0, 0, 0.075, 0.055, 0.035)
    lens  = _cylinder(0, 0, 0.035, 0.030, 0.040, axis='z', segs=20)
    # Front cap at z=0.075, rear cap at z=-0.005
    glass = np.concatenate([
        _disk_cap(0, 0,  0.075, 0.030, normal_z= 1.0, segs=20),
        _disk_cap(0, 0, -0.005, 0.030, normal_z=-1.0, segs=20),
    ], axis=0)
    return body, lens, glass


def _mesh_camera_video() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shoulder-mount video camera (longer, narrower)."""
    body  = _box(0, 0, 0, 0.100, 0.045, 0.055)
    lens  = _cylinder(0, 0, 0.055, 0.025, 0.060, axis='z', segs=18)
    glass = np.concatenate([
        _disk_cap(0, 0,  0.115, 0.025, normal_z= 1.0, segs=18),
        _disk_cap(0, 0, -0.005, 0.025, normal_z=-1.0, segs=18),
    ], axis=0)
    return body, lens, glass


def _mesh_camera_box() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Box camera open at the rear — supports digital sensor + large-format plate.

    The -Z back wall is omitted so light that passes through the transparent
    digital sensor can reach the large-format plate sensor placed behind it.
    The body is otherwise a closed five-sided box (four side walls + front
    wall at +Z where the lens barrel attaches).
    """
    body  = _box_open_back(0, 0, 0, 0.060, 0.060, 0.080)
    lens  = _cylinder(0, 0, 0.080, 0.018, 0.025, axis='z', segs=14)
    glass = np.concatenate([
        _disk_cap(0, 0,  0.105, 0.018, normal_z= 1.0, segs=14),
        _disk_cap(0, 0,  0.055, 0.018, normal_z=-1.0, segs=14),
    ], axis=0)
    return body, lens, glass


# Registry: mesh_id → builder returning (body_verts, tube_verts, glass_verts)
_MESH_PRESETS: Dict[str, Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
    "camera_35mm":  _mesh_camera_35mm,
    "camera_video": _mesh_camera_video,
    "camera_box":   _mesh_camera_box,
}


def register_mesh_preset(
    mesh_id: str,
    builder: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Register a new camera mesh preset callable."""
    _MESH_PRESETS[mesh_id] = builder


def _build_mesh_parts(
    mesh_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (body_verts, tube_verts, glass_verts) each shape (-1, 6)."""
    fn = _MESH_PRESETS.get(mesh_id, _mesh_camera_35mm)
    return fn()


def _build_mesh(mesh_id: str) -> np.ndarray:
    """Return concatenated (body + tube + glass) vertex array, shape (-1, 6)."""
    body, tube, glass = _build_mesh_parts(mesh_id)
    return np.concatenate([body, tube, glass], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# GLSL — simple Phong, same vertex format as duty_station.py
# ─────────────────────────────────────────────────────────────────────────────

_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
uniform mat4 uMVP;
uniform mat4 uMV;
out vec3 vNormV;
out vec3 vPosV;
void main() {
    vec4 pv = uMV * vec4(aPos, 1.0);
    vPosV   = pv.xyz;
    vNormV  = normalize(mat3(uMV) * aNorm);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_FS = """
#version 330 core
in vec3 vNormV;
in vec3 vPosV;
uniform vec3 uLightV;
uniform vec3 uBodyColor;
out vec4 fragColor;
void main() {
    vec3 n   = normalize(vNormV);
    vec3 lv  = normalize(uLightV);
    float diff = max(dot(n, lv), 0.0);
    vec3 col = uBodyColor * (0.25 + 0.75 * diff);
    // specular
    vec3 rv  = reflect(-lv, n);
    vec3 vv  = normalize(-vPosV);
    float sp = pow(max(dot(rv, vv), 0.0), 32.0) * 0.4;
    fragColor = vec4(col + sp, 1.0);
}
"""

# Glass caps — semi-transparent, strong Phong highlight, coated blue-grey tint
_FS_GLASS = """
#version 330 core
in vec3 vNormV;
in vec3 vPosV;
uniform vec3 uLightV;
uniform vec3 uBodyColor;
uniform float uAlpha;
out vec4 fragColor;
void main() {
    vec3 n   = normalize(vNormV);
    vec3 lv  = normalize(uLightV);
    float diff = max(dot(n, lv), 0.0);
    vec3 col = uBodyColor * (0.08 + 0.22 * diff);
    vec3 rv  = reflect(-lv, n);
    vec3 vv  = normalize(-vPosV);
    float sp = pow(max(dot(rv, vv), 0.0), 128.0) * 1.1;
    fragColor = vec4(col + vec3(sp), uAlpha);
}
"""

# Emitter surfaces — flat unlit, additive blend
_VS_EMIT = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_FS_EMIT = """
#version 330 core
uniform vec3 uEmitColor;
uniform float uEmitAlpha;
out vec4 fragColor;
void main() {
    fragColor = vec4(uEmitColor, uEmitAlpha);
}
"""


@dataclass
class EmitterFace:
    """One emissive disc face on the in-world camera GL mesh.

    ``pos``    : disc centre in camera-local space (same coords as EmitterSpec)
    ``normal`` : outward emission direction (unit vector)
    ``radius`` : disc radius in metres
    ``color``  : (r, g, b) emission tint, 0–1 each
    ``power``  : brightness multiplier for the alpha channel (clamped 0–1 for GL)
    ``enabled``: False → geometry drawn black (absorber stand-in)
    """
    pos:     Tuple[float, float, float] = (0.0, 0.0, -0.010)
    normal:  Tuple[float, float, float] = (0.0, 0.0, -1.0)
    radius:  float = 0.005
    color:   Tuple[float, float, float] = (1.0, 0.92, 0.80)
    power:   float = 1.0
    enabled: bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# CameraItem
# ─────────────────────────────────────────────────────────────────────────────

class CameraItem:
    """GL-renderable physical camera item.

    Wraps a ``PlacedCamera`` data object.  The ``PlacedCamera`` is the
    source of truth for serialisable state (pan_deg, tilt_deg, focal_mm,
    pos, yaw_deg, mesh_id).  This class only adds GL handles and the
    armature transform.

    The GL camera is subservient to this item's optical configuration:
    ``focal_mm`` is pushed into ``Camera.focal_mm`` every tick so the
    GL viewport is derived from the real lens spec via ``fov_y_rad()``.

    Typical lifecycle
    -----------------
    ::

        item = CameraItem(placed_camera)
        item.init_gl()          # after GL context

        # player tick writes:
        placed_camera.pan_deg  += delta
        placed_camera.tilt_deg += delta
        placed_camera.focal_mm  = new_focal

        # render loop:
        item.draw(MVP, MV, light_view)

        # cleanup:
        item.destroy_gl()
    """

    # Outer body: matte black for all presets
    _BODY_COLORS: Dict[str, Tuple[float, float, float]] = {
        "camera_35mm":  (0.05, 0.05, 0.06),
        "camera_video": (0.04, 0.04, 0.05),
        "camera_box":   (0.05, 0.05, 0.06),
    }
    # Lens tube: Canon L warm grey
    _TUBE_COLORS: Dict[str, Tuple[float, float, float]] = {
        "camera_35mm":  (0.76, 0.74, 0.70),
        "camera_video": (0.72, 0.70, 0.67),
        "camera_box":   (0.76, 0.74, 0.70),
    }

    def __init__(self, placed: "PlacedCamera"):  # type: ignore[name-defined]
        self.placed        = placed
        self._gl_ready     = False
        self._vao          = None
        self._vbo          = None
        self._n_body       = 0   # vertex count for opaque body
        self._n_tube       = 0   # vertex count for opaque tube/barrel
        self._n_glass      = 0   # vertex count for translucent lens caps
        self._prog         = None
        self._prog_glass   = None
        self._prog_emit    = None
        self._vao_emit     = None
        self._vbo_emit     = None
        self._n_emit       = 0    # total emitter vertices across all faces
        self._emitters: List[EmitterFace] = []  # live emitter face list
        # Projector back panel (optional)
        self._projector_back       = None   # ProjectorBackSpec-like object | None
        self._pb_sensor_r_max      = 0.020  # default APS-C half-diagonal
        self._pb_sensor_z          = -0.0444
        self._vao_proj             = None
        self._vbo_proj             = None
        self._n_proj               = 0
        self.software: list = []  # CameraSoftware instances; ticked each frame

    # ── Properties that PlayerController expects ──────────────────────────────

    @property
    def sensor_name(self) -> str:
        return str(getattr(self.placed, 'sensor_name', 'full_frame_35mm'))

    @property
    def pos(self) -> np.ndarray:
        return self.placed.pos

    @property
    def pan_deg(self) -> float:
        return self.placed.pan_deg

    @pan_deg.setter
    def pan_deg(self, v: float) -> None:
        self.placed.pan_deg = float(v) % 360.0

    @property
    def tilt_deg(self) -> float:
        return self.placed.tilt_deg

    @tilt_deg.setter
    def tilt_deg(self, v: float) -> None:
        self.placed.tilt_deg = float(np.clip(v,
                                             self.placed.tilt_min_deg,
                                             self.placed.tilt_max_deg))

    @property
    def focal_mm(self) -> float:
        return self.placed.focal_mm

    @focal_mm.setter
    def focal_mm(self, v: float) -> None:
        f_min = float(getattr(self.placed, 'focal_min_mm', v))
        f_max = float(getattr(self.placed, 'focal_max_mm', v))
        self.placed.focal_mm = float(np.clip(v, min(f_min, f_max), max(f_min, f_max)))

    @property
    def focal_min_mm(self) -> float:
        return float(getattr(self.placed, 'focal_min_mm', self.placed.focal_mm))

    @property
    def focal_max_mm(self) -> float:
        return float(getattr(self.placed, 'focal_max_mm', self.placed.focal_mm))

    @property
    def interaction_radius(self) -> float:
        return self.placed.interaction_radius

    @property
    def world_position(self) -> np.ndarray:
        return self.placed.pos

    # ── Emitter face management ───────────────────────────────────────────────

    def add_emitter(self, face: EmitterFace) -> None:
        """Register an emissive disc face; call before init_gl."""
        self._emitters.append(face)

    def set_projector_back(self, spec, sensor_r_max: float = 0.020,
                           sensor_z: float = -0.0444) -> None:
        """Attach a projector back panel spec for GL rendering.

        ``spec`` is any object with ``enabled``, ``power``, ``color``,
        ``z_offset``, and ``radius_scale`` attributes — a
        ``ProjectorBackSpec`` instance or a compatible namespace.

        Call before ``init_gl()`` so the panel VAO is built from the correct
        sensor geometry.  Can also be called after ``init_gl()`` if you then
        call ``_rebuild_projector_vao()`` manually.

        ``sensor_r_max`` and ``sensor_z`` are read from the CameraPreset sensor
        at station setup time and forwarded here.
        """
        self._projector_back  = spec
        self._pb_sensor_r_max = float(sensor_r_max)
        self._pb_sensor_z     = float(sensor_z)

    def _build_emitter_verts(self) -> np.ndarray:
        """Triangulate all registered EmitterFace discs into a (-1, 6) vertex array."""
        segs = 24
        parts: List[np.ndarray] = []
        for em in self._emitters:
            pos = np.asarray(em.pos, np.float64)
            nrm = np.asarray(em.normal, np.float64)
            nl = np.linalg.norm(nrm)
            nrm = nrm / nl if nl > 1e-12 else np.array([0., 0., -1.])
            up = np.array([0., 1., 0.])
            if abs(nrm @ up) > 0.9:
                up = np.array([1., 0., 0.])
            t1 = np.cross(nrm, up); t1 /= np.linalg.norm(t1)
            t2 = np.cross(nrm, t1)
            angles = np.linspace(0, 2 * math.pi, segs, endpoint=False)
            for i in range(segs):
                a0, a1 = angles[i], angles[(i + 1) % segs]
                p0 = pos + em.radius * (math.cos(a0) * t1 + math.sin(a0) * t2)
                p1 = pos + em.radius * (math.cos(a1) * t1 + math.sin(a1) * t2)
                tri = np.array([
                    [pos[0], pos[1], pos[2], nrm[0], nrm[1], nrm[2]],
                    [p0[0],  p0[1],  p0[2],  nrm[0], nrm[1], nrm[2]],
                    [p1[0],  p1[1],  p1[2],  nrm[0], nrm[1], nrm[2]],
                ], np.float32)
                parts.append(tri)
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 6), np.float32)

    def _build_projector_verts(self) -> np.ndarray:
        """Triangulate the projector back panel into a (-1, 6) vertex array.

        The panel is a filled disc in the XY plane at ``_pb_sensor_z - z_offset``,
        oriented toward +Z (into the lens), with radius
        ``_pb_sensor_r_max * radius_scale``.
        """
        if self._projector_back is None:
            return np.zeros((0, 6), np.float32)
        pb = self._projector_back
        r    = self._pb_sensor_r_max * float(getattr(pb, 'radius_scale', 1.0))
        z    = self._pb_sensor_z - float(getattr(pb, 'z_offset', 0.002))
        segs = 32
        nrm  = np.array([0., 0., 1.], np.float64)   # face toward lens
        t1   = np.array([1., 0., 0.], np.float64)
        t2   = np.array([0., 1., 0.], np.float64)
        ctr  = np.array([0., 0., z],  np.float64)
        angles = np.linspace(0, 2 * math.pi, segs, endpoint=False)
        parts: List[np.ndarray] = []
        for i in range(segs):
            a0, a1 = angles[i], angles[(i + 1) % segs]
            p0 = ctr + r * (math.cos(a0) * t1 + math.sin(a0) * t2)
            p1 = ctr + r * (math.cos(a1) * t1 + math.sin(a1) * t2)
            tri = np.array([
                [ctr[0], ctr[1], ctr[2], 0., 0., 1.],
                [p0[0],  p0[1],  p0[2],  0., 0., 1.],
                [p1[0],  p1[1],  p1[2],  0., 0., 1.],
            ], np.float32)
            parts.append(tri)
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 6), np.float32)

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def init_gl(self) -> None:
        if not _HAS_GL:
            return
        body_verts, tube_verts, glass_verts = _build_mesh_parts(self.placed.mesh_id)
        self._n_body  = len(body_verts)
        self._n_tube  = len(tube_verts)
        self._n_glass = len(glass_verts)
        mesh = np.concatenate([body_verts, tube_verts, glass_verts], axis=0)

        vs = _gl_shaders.compileShader(_VS, GL_VERTEX_SHADER)
        fs = _gl_shaders.compileShader(_FS, GL_FRAGMENT_SHADER)
        self._prog = _gl_shaders.compileProgram(vs, fs)

        vs_g = _gl_shaders.compileShader(_VS, GL_VERTEX_SHADER)
        fs_g = _gl_shaders.compileShader(_FS_GLASS, GL_FRAGMENT_SHADER)
        self._prog_glass = _gl_shaders.compileProgram(vs_g, fs_g)

        vs_e = _gl_shaders.compileShader(_VS_EMIT, GL_VERTEX_SHADER)
        fs_e = _gl_shaders.compileShader(_FS_EMIT, GL_FRAGMENT_SHADER)
        self._prog_emit = _gl_shaders.compileProgram(vs_e, fs_e)

        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh.nbytes, mesh, GL_STATIC_DRAW)
        stride = 6 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glBindVertexArray(0)

        # Emitter disc VAO (separate; may be empty)
        emit_verts = self._build_emitter_verts()
        self._n_emit = len(emit_verts)
        self._vao_emit = glGenVertexArrays(1)
        self._vbo_emit = glGenBuffers(1)
        glBindVertexArray(self._vao_emit)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_emit)
        glBufferData(GL_ARRAY_BUFFER,
                     emit_verts.nbytes if self._n_emit > 0 else 1,
                     emit_verts if self._n_emit > 0 else np.zeros((1,), np.float32),
                     GL_STATIC_DRAW)
        if self._n_emit > 0:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glBindVertexArray(0)

        # Projector back panel VAO (separate; may be empty)
        proj_verts = self._build_projector_verts()
        self._n_proj = len(proj_verts)
        self._vao_proj = glGenVertexArrays(1)
        self._vbo_proj = glGenBuffers(1)
        glBindVertexArray(self._vao_proj)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_proj)
        glBufferData(GL_ARRAY_BUFFER,
                     proj_verts.nbytes if self._n_proj > 0 else 1,
                     proj_verts if self._n_proj > 0 else np.zeros((1,), np.float32),
                     GL_STATIC_DRAW)
        if self._n_proj > 0:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glBindVertexArray(0)

        self._gl_ready = True

    def destroy_gl(self) -> None:
        if not self._gl_ready:
            return
        if self._vbo is not None:
            glDeleteBuffers(1, [self._vbo])
        if self._vao is not None:
            glDeleteVertexArrays(1, [self._vao])
        if self._vbo_emit is not None:
            glDeleteBuffers(1, [self._vbo_emit])
        if self._vao_emit is not None:
            glDeleteVertexArrays(1, [self._vao_emit])
        if self._vbo_proj is not None:
            glDeleteBuffers(1, [self._vbo_proj])
        if self._vao_proj is not None:
            glDeleteVertexArrays(1, [self._vao_proj])
        self._prog_glass = None
        self._prog_emit  = None
        self._gl_ready = False

    # ── Armature model matrix ─────────────────────────────────────────────────

    def _model_matrix(self) -> np.ndarray:
        """World transform: translate to pos, yaw the base, then pan+tilt head."""
        p  = self.placed.pos
        by = math.radians(self.placed.yaw_deg)  # base rotation (armature mount)
        pn = math.radians(self.placed.pan_deg)  # head pan
        ti = math.radians(self.placed.tilt_deg) # head tilt

        # Tilt (X-axis rotation)
        Rx = np.eye(4, dtype=np.float64)
        Rx[1, 1] =  math.cos(ti); Rx[1, 2] = -math.sin(ti)
        Rx[2, 1] =  math.sin(ti); Rx[2, 2] =  math.cos(ti)

        # Pan (Z-axis rotation — yaw in world space)
        Rz_pan = np.eye(4, dtype=np.float64)
        Rz_pan[0, 0] =  math.cos(pn); Rz_pan[0, 1] = -math.sin(pn)
        Rz_pan[1, 0] =  math.sin(pn); Rz_pan[1, 1] =  math.cos(pn)

        # Base yaw (armature on stand, fixed)
        Rz_base = np.eye(4, dtype=np.float64)
        Rz_base[0, 0] =  math.cos(by); Rz_base[0, 1] = -math.sin(by)
        Rz_base[1, 0] =  math.sin(by); Rz_base[1, 1] =  math.cos(by)

        T = np.eye(4, dtype=np.float64)
        T[0, 3] = p[0]; T[1, 3] = p[1]; T[2, 3] = p[2]

        return (T @ Rz_base @ Rz_pan @ Rx).astype(np.float32)

    # ── Software tick ─────────────────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        """Drive all attached CameraSoftware modules.

        Builds a CameraContext wrapping this item, calls each software
        module's tick(dt, ctx) in order.  Modules write back to the camera
        directly through the context; no return value is used.
        """
        if not self.software:
            return
        try:
            from camera_software import CameraContext
            ctx = CameraContext(self)
        except ImportError:
            ctx = self  # fallback: pass self if package not available
        for sw in self.software:
            sw.tick(dt, ctx)

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, MVP: np.ndarray, MV: np.ndarray,
             light_v: np.ndarray) -> None:
        """Render the camera body into the current GL context.

        ``MVP`` and ``MV`` are the scene view-projection matrices.  The
        camera applies its own model matrix on top so it appears at the
        correct world position and orientation.
        """
        if not _HAS_GL or not self._gl_ready:
            return

        M  = self._model_matrix()
        mv = (MV  @ M.astype(np.float64)).astype(np.float32)
        mvp = (MVP.astype(np.float64) @ M.astype(np.float64)).astype(np.float32)

        body_col = self._BODY_COLORS.get(self.placed.mesh_id, (0.05, 0.05, 0.06))
        tube_col = self._TUBE_COLORS.get(self.placed.mesh_id, (0.76, 0.74, 0.70))
        # Coated optical glass: blue-grey AR tint
        glass_col = (0.55, 0.72, 0.82)

        glUseProgram(self._prog)
        glUniformMatrix4fv(glGetUniformLocation(self._prog, "uMVP"),
                           1, GL_FALSE, mvp.T)
        glUniformMatrix4fv(glGetUniformLocation(self._prog, "uMV"),
                           1, GL_FALSE, mv.T)
        glUniform3f(glGetUniformLocation(self._prog, "uLightV"),
                    float(light_v[0]), float(light_v[1]), float(light_v[2]))

        loc_col = glGetUniformLocation(self._prog, "uBodyColor")
        glBindVertexArray(self._vao)
        # Body — matte black
        glUniform3f(loc_col, body_col[0], body_col[1], body_col[2])
        glDrawArrays(GL_TRIANGLES, 0, self._n_body)
        # Lens tube — Canon L warm grey
        glUniform3f(loc_col, tube_col[0], tube_col[1], tube_col[2])
        glDrawArrays(GL_TRIANGLES, self._n_body, self._n_tube)
        glBindVertexArray(0)
        glUseProgram(0)

        # Glass caps — semi-transparent, rendered last so blend works
        if self._n_glass > 0 and self._prog_glass is not None:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)
            glUseProgram(self._prog_glass)
            glUniformMatrix4fv(glGetUniformLocation(self._prog_glass, "uMVP"),
                               1, GL_FALSE, mvp.T)
            glUniformMatrix4fv(glGetUniformLocation(self._prog_glass, "uMV"),
                               1, GL_FALSE, mv.T)
            glUniform3f(glGetUniformLocation(self._prog_glass, "uLightV"),
                        float(light_v[0]), float(light_v[1]), float(light_v[2]))
            glUniform3f(glGetUniformLocation(self._prog_glass, "uBodyColor"),
                        glass_col[0], glass_col[1], glass_col[2])
            glUniform1f(glGetUniformLocation(self._prog_glass, "uAlpha"), 0.38)
            glBindVertexArray(self._vao)
            glDrawArrays(GL_TRIANGLES, self._n_body + self._n_tube, self._n_glass)
            glBindVertexArray(0)
            glUseProgram(0)
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)

        # Emitter discs — flat unlit additive blend
        if self._n_emit > 0 and self._prog_emit is not None and self._emitters:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)
            glUseProgram(self._prog_emit)
            glUniformMatrix4fv(glGetUniformLocation(self._prog_emit, "uMVP"),
                               1, GL_FALSE, mvp.T)
            loc_ec  = glGetUniformLocation(self._prog_emit, "uEmitColor")
            loc_ea  = glGetUniformLocation(self._prog_emit, "uEmitAlpha")
            glBindVertexArray(self._vao_emit)
            v_off = 0
            segs = 24
            for em in self._emitters:
                n_tris = segs
                if em.enabled:
                    glUniform3f(loc_ec, float(em.color[0]), float(em.color[1]), float(em.color[2]))
                    glUniform1f(loc_ea, float(min(1.0, max(0.0, em.power))))
                else:
                    glUniform3f(loc_ec, 0.0, 0.0, 0.0)
                    glUniform1f(loc_ea, 0.6)
                glDrawArrays(GL_TRIANGLES, v_off, n_tris * 3)
                v_off += n_tris * 3
            glBindVertexArray(0)
            glUseProgram(0)
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)

        # Projector back panel — glowing rect behind sensor, alpha-blend emissive
        if self._n_proj > 0 and self._prog_emit is not None and self._projector_back is not None:
            pb = self._projector_back
            pb_on = bool(getattr(pb, 'enabled', False))
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)
            glUseProgram(self._prog_emit)
            glUniformMatrix4fv(glGetUniformLocation(self._prog_emit, "uMVP"),
                               1, GL_FALSE, mvp.T)
            loc_ec = glGetUniformLocation(self._prog_emit, "uEmitColor")
            loc_ea = glGetUniformLocation(self._prog_emit, "uEmitAlpha")
            if pb_on:
                _c = getattr(pb, 'color', (0.95, 0.97, 1.0))
                _pw = float(getattr(pb, 'power', 1.0))
                glUniform3f(loc_ec, float(_c[0]), float(_c[1]), float(_c[2]))
                glUniform1f(loc_ea, float(min(1.0, max(0.05, _pw / max(_pw, 1.0)))))
            else:
                glUniform3f(loc_ec, 0.05, 0.05, 0.05)  # charcoal grey when off
                glUniform1f(loc_ea, 0.7)
            glBindVertexArray(self._vao_proj)
            glDrawArrays(GL_TRIANGLES, 0, self._n_proj)
            glBindVertexArray(0)
            glUseProgram(0)
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_camera_items(workspace) -> list:
    """Build CameraItem instances for every PlacedCamera in *workspace*.

    *workspace* is any object with a ``cameras()`` method returning a list
    of ``PlacedCamera`` (e.g. a ``RoomWorkspace``).

    GL must be initialised before calling this.
    """
    from placed_object import PlacedCamera as _PlacedCamera  # avoid circular at module level
    items = []
    for pc in workspace.cameras():
        if not isinstance(pc, _PlacedCamera):
            continue
        item = CameraItem(pc)
        item.init_gl()
        items.append(item)
    return items
