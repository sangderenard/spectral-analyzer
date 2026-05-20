"""
base_gl_renderer.py
===================

Hardware-accelerated base-material renderer that consumes the three
material-database SSBOs (PBR binding=10, Phong binding=11, Enamel binding=14)
and renders geometry through a GL 4.3 core Phong shader with Schlick Fresnel
and optional enamel thin-film iridescence.

Usage
-----
>>> from material_db import MaterialDatabase
>>> from controls   import get_shader_walker
>>> db  = MaterialDatabase(); db.load_all()
>>> rdr = BaseGLRenderer(db)
>>> # Call after pygame / GL context is live:
>>> rdr.init_gl()
>>> rdr.register()     # installs into the ShaderFrameWalker

Each frame the walker calls ``_run()``, which iterates ``_mesh_list`` and
draws every (VAO, n_verts, mvp, mv, light_dirs, light_colors, light_intens) tuple
that was enqueued via ``enqueue_mesh()``.  Callers must drain the list each
frame by calling ``clear_mesh_list()`` **after** ``_run`` fires, or just let
``_run`` drain it automatically (see ``auto_drain`` parameter in ``__init__``).
"""

from __future__ import annotations

import os
import sys
import ctypes
from typing import Optional, List, Tuple

import numpy as np

# ── OpenGL ────────────────────────────────────────────────────────────────────
try:
    from OpenGL import GL
    from OpenGL.GL import (
        glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
        glGetShaderInfoLog, glCreateProgram, glAttachShader, glLinkProgram,
        glGetProgramiv, glGetProgramInfoLog, glUseProgram, glDeleteShader,
        glGenBuffers, glBindBuffer, glBufferData, glBindBufferBase,
        glUniformMatrix4fv, glUniformMatrix3fv, glUniform3f, glUniform1f, glUniform1i,
        glUniform3fv, glUniform1fv,
        glGetUniformLocation, glEnable, glDisable, glBlendFunc,
        glDepthMask,
        glGenTextures, glBindTexture, glTexParameteri, glTexImage3D,
        glActiveTexture,
        GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS, GL_LINK_STATUS,
        GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, GL_STATIC_DRAW,
        GL_FLOAT, GL_TRUE, GL_FALSE, GL_BLEND, GL_CULL_FACE,
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        GL_TEXTURE_2D_ARRAY, GL_TEXTURE_3D, GL_TEXTURE0, GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R,
        GL_LINEAR, GL_NEAREST, GL_CLAMP_TO_EDGE, GL_REPEAT,
        GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE,
    )
    _GL_OK = True
except ImportError:
    _GL_OK = False

# ── Project imports ───────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from controls import register_shader_node  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# GLSL source paths
# ─────────────────────────────────────────────────────────────────────────────
_VERT_PATH = os.path.join(_HERE, "csrc", "shaders", "base_material.vert.glsl")
_FRAG_PATH = os.path.join(_HERE, "csrc", "shaders", "base_material.frag.glsl")

# SSBO binding points — must match material_db.py and the GLSL sources
_BINDING_PBR   = 10
_BINDING_PHONG = 11
_BINDING_ENAMEL = 14
_BINDING_TEXSTACK = 15

# ─────────────────────────────────────────────────────────────────────────────

def _read_glsl(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _compile_shader(src: str, shader_type: int) -> int:
    """Compile one GL shader stage; raise RuntimeError on failure."""
    s = glCreateShader(shader_type)
    glShaderSource(s, src)
    glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(s).decode(errors="replace")
        kind = "VERTEX" if shader_type == GL_VERTEX_SHADER else "FRAGMENT"
        raise RuntimeError(f"[BaseGLRenderer] {kind} shader compile error:\n{log}")
    return s


def _link_program(vert: int, frag: int) -> int:
    """Link vert+frag into a GL program; raise RuntimeError on failure."""
    prog = glCreateProgram()
    glAttachShader(prog, vert)
    glAttachShader(prog, frag)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        log = glGetProgramInfoLog(prog).decode(errors="replace")
        raise RuntimeError(f"[BaseGLRenderer] program link error:\n{log}")
    glDeleteShader(vert)
    glDeleteShader(frag)
    return prog


# ─────────────────────────────────────────────────────────────────────────────

class BaseGLRenderer:
    """
    Parameters
    ----------
    material_db:
        A ``MaterialDatabase`` instance from ``material_db.py``.
        Must already have materials loaded (``db.load_all()`` or equivalent).
    auto_drain:
        When True (default), ``_run`` clears the mesh list after drawing.
        Set to False if you manage the list manually.
    """

    def __init__(self, material_db, *, auto_drain: bool = True):
        self._db         = material_db
        self._auto_drain = auto_drain
        self._prog:   Optional[int] = None
        self._ssbo:   dict[int, int] = {}   # binding → GL buffer name

        # Cached uniform locations (populated in init_gl)
        self._u_mvp             = -1
        self._u_mv              = -1
        self._u_num_lights      = -1
        self._u_light_pos       = -1
        self._u_light_color     = -1
        self._u_light_intensity = -1
        self._u_light_group_id  = -1
        self._u_light_calibration = -1
        self._u_cat_ccm_matrix = -1
        self._u_enable_specular = -1
        self._u_enable_emission_direct = -1
        self._enable_specular = True
        self._enable_emission_direct = False
        self._light_calibration = 1.0
        self._cat_ccm_matrix = np.eye(3, dtype=np.float32)
        self._light_pos = np.zeros((0, 3), dtype=np.float32)
        self._light_color = np.zeros((0, 3), dtype=np.float32)
        self._light_intensity = np.zeros((0,), dtype=np.float32)
        self._light_group_id = np.zeros((0,), dtype=np.int32)

        # ── UV texture-pack stack (Stage 2 wired) ───────────────────
        # Default 1×1×1 identity texel `(R=0, G=255, B=128, A=255)` =
        # (R=0    → no direct/specular-coupled emission gain,
        #  G=1.0  → full diffuse-lobe emission gain (passes mat.emit through),
        #  B=0.5  → saturation identity (mix factor 1.0 = unchanged chroma),
        #  A=1.0  → no per-texel dim).
        # With this texel the Stage 2 fragment math collapses to the
        # pre-Stage-2 behaviour `col += emission` exactly, so any caller
        # that has not yet authored an emission UV texture sees no change.
        self._tex_emit_uv: Optional[int] = None
        self._tex_color_uv: Optional[int] = None
        self._tex_depth_uv: Optional[int] = None
        self._tex_remit_uv: Optional[int] = None
        self._tex_field_vol: Optional[int] = None
        self._uv_tex_unit_emit = 0
        self._uv_tex_unit_color = 1
        self._uv_tex_unit_depth = 2
        self._uv_tex_unit_remit = 3
        self._uv_tex_unit_field = 4
        self._u_emit_uv = -1
        self._u_color_uv = -1
        self._u_depth_uv = -1
        self._u_remit_uv = -1
        self._u_field_vol = -1
        self._u_field_gain = -1
        self._field_gain = 0.0

        # Mesh draw queue: list of tuples (vao, n_verts, mvp, mv,
        #                                  light_dirs, light_colors, light_intens)
        # Types: vao=int, n_verts=int, mvp=np.ndarray(16,f32),
        #        mv=np.ndarray(16,f32), light_dirs=np.ndarray(N,3,f32),
        #        light_colors=np.ndarray(N,3,f32), light_intens=np.ndarray(N,f32)
        self._mesh_list: List[Tuple] = []

        self._registered = False
        self._mat_tensors_ref = None

    # ── GL initialisation ─────────────────────────────────────────────────────

    def init_gl(self) -> None:
        """Compile shaders and upload material SSBOs.  Call once after GL context is current."""
        if not _GL_OK:
            raise RuntimeError("[BaseGLRenderer] PyOpenGL not available")
        self._build_program()
        self._build_ssbos()
        self._build_default_uv_textures()

    def _build_default_uv_textures(self) -> None:
        """Allocate the 1×1×1 default `emit_uv` texture array.

        Identity texel for the Stage 2 fragment math — see UV_EMISSION_STACK_PLAN.md.
        Any mesh that does not bind a real emission texture array samples this
        single texel and gets pre-Stage-2 behaviour (`col += emission`) exactly.
        """
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex)
        # Layer 0: identity texel repeated over a 16x16 tile.
        # Layer 1: missing-material checkerboard emission mask.
        emit_layers = np.zeros((2, 16, 16, 4), dtype=np.uint8)
        emit_layers[0, :, :, :] = np.array([0, 255, 128, 255], dtype=np.uint8)
        for y in range(16):
            for x in range(16):
                on = ((x // 4) + (y // 4)) % 2 == 0
                emit_layers[1, y, x, :] = (
                    np.array([32, 255, 220, 255], dtype=np.uint8) if on
                    else np.array([0, 42, 70, 92], dtype=np.uint8)
                )
        glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
            16, 16, 2,                    # width, height, layers
            0, GL_RGBA, GL_UNSIGNED_BYTE,
            emit_layers.ctypes.data_as(ctypes.c_void_p),
        )
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        self._tex_emit_uv = int(tex)

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex)
        default_color = np.array([255, 255, 255, 0], dtype=np.uint8)
        glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
            1, 1, 1,
            0, GL_RGBA, GL_UNSIGNED_BYTE,
            default_color.ctypes.data_as(ctypes.c_void_p),
        )
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        self._tex_color_uv = int(tex)

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex)
        default_depth = np.array([0, 0, 0, 255], dtype=np.uint8)
        glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
            1, 1, 1,
            0, GL_RGBA, GL_UNSIGNED_BYTE,
            default_depth.ctypes.data_as(ctypes.c_void_p),
        )
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        self._tex_depth_uv = int(tex)

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex)
        default_remit = np.array([0, 0, 0, 0], dtype=np.uint8)
        glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
            1, 1, 1,
            0, GL_RGBA, GL_UNSIGNED_BYTE,
            default_remit.ctypes.data_as(ctypes.c_void_p),
        )
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        self._tex_remit_uv = int(tex)

    def set_emit_uv_texture_id(self, tex_id: int) -> None:
        """Point uEmitUv at an externally-managed GL texture (no upload)."""
        self._tex_emit_uv = int(tex_id)

    def set_field_volume_texture(self, tex_id: int, gain: float = 0.0) -> None:
        """Point uFieldVolume at an externally-managed GL_TEXTURE_3D (no upload)."""
        self._tex_field_vol = int(tex_id)
        self._field_gain = float(gain)

    def set_emit_uv_texture_array(self, rgba_layers: "np.ndarray") -> None:
        """Upload an RGBA8 emission texture array as (layers, height, width, 4)."""
        arr = np.ascontiguousarray(rgba_layers, dtype=np.uint8)
        if arr.ndim != 4 or arr.shape[-1] != 4:
            raise ValueError("rgba_layers must be (layers, height, width, 4) uint8")
        layers, height, width, _ = arr.shape
        if self._tex_emit_uv is None:
            self._tex_emit_uv = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_emit_uv)
        glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
            int(width), int(height), int(layers),
            0, GL_RGBA, GL_UNSIGNED_BYTE,
            arr.ctypes.data_as(ctypes.c_void_p),
        )
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def set_color_uv_texture_array(self, rgba_layers: "np.ndarray") -> None:
        """Upload an RGBA8 color override texture array as (layers, height, width, 4)."""
        arr = np.ascontiguousarray(rgba_layers, dtype=np.uint8)
        if arr.ndim != 4 or arr.shape[-1] != 4:
            raise ValueError("rgba_layers must be (layers, height, width, 4) uint8")
        layers, height, width, _ = arr.shape
        if self._tex_color_uv is None:
            self._tex_color_uv = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_color_uv)
        glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
            int(width), int(height), int(layers),
            0, GL_RGBA, GL_UNSIGNED_BYTE,
            arr.ctypes.data_as(ctypes.c_void_p),
        )
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def set_depth_uv_texture_array(self, rgba_layers: "np.ndarray") -> None:
        """Upload an RGBA8 depth/thickness texture array as (layers, height, width, 4)."""
        arr = np.ascontiguousarray(rgba_layers, dtype=np.uint8)
        if arr.ndim != 4 or arr.shape[-1] != 4:
            raise ValueError("rgba_layers must be (layers, height, width, 4) uint8")
        layers, height, width, _ = arr.shape
        if self._tex_depth_uv is None:
            self._tex_depth_uv = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_depth_uv)
        glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
            int(width), int(height), int(layers),
            0, GL_RGBA, GL_UNSIGNED_BYTE,
            arr.ctypes.data_as(ctypes.c_void_p),
        )
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def set_remit_uv_texture_array(self, rgba_layers: "np.ndarray") -> None:
        """Upload an RGBA8 simple reemission texture array as (layers, height, width, 4)."""
        arr = np.ascontiguousarray(rgba_layers, dtype=np.uint8)
        if arr.ndim != 4 or arr.shape[-1] != 4:
            raise ValueError("rgba_layers must be (layers, height, width, 4) uint8")
        layers, height, width, _ = arr.shape
        if self._tex_remit_uv is None:
            self._tex_remit_uv = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_remit_uv)
        glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
            int(width), int(height), int(layers),
            0, GL_RGBA, GL_UNSIGNED_BYTE,
            arr.ctypes.data_as(ctypes.c_void_p),
        )
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def _build_program(self) -> None:
        vert_src = _read_glsl(_VERT_PATH)
        frag_src = _read_glsl(_FRAG_PATH)
        vert     = _compile_shader(vert_src, GL_VERTEX_SHADER)
        frag     = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
        self._prog = _link_program(vert, frag)

        self._u_mvp             = glGetUniformLocation(self._prog, "uMVP")
        self._u_mv              = glGetUniformLocation(self._prog, "uMV")
        self._u_num_lights      = glGetUniformLocation(self._prog, "uNumLights")
        self._u_light_pos       = glGetUniformLocation(self._prog, "uLightPos")
        self._u_light_color     = glGetUniformLocation(self._prog, "uLightColor")
        self._u_light_intensity = glGetUniformLocation(self._prog, "uLightIntensity")
        self._u_light_group_id  = glGetUniformLocation(self._prog, "uLightGroupId")
        self._u_light_calibration = glGetUniformLocation(self._prog, "uLightCalibration")
        self._u_cat_ccm_matrix = glGetUniformLocation(self._prog, "uCatCcmMatrix")
        self._u_emit_uv         = glGetUniformLocation(self._prog, "uEmitUv")
        self._u_color_uv        = glGetUniformLocation(self._prog, "uColorUv")
        self._u_depth_uv        = glGetUniformLocation(self._prog, "uDepthUv")
        self._u_remit_uv        = glGetUniformLocation(self._prog, "uRemitUv")
        self._u_enable_specular = glGetUniformLocation(self._prog, "uEnableSpecular")
        self._u_enable_emission_direct = glGetUniformLocation(self._prog, "uEnableEmissionDirect")
        self._u_field_vol   = glGetUniformLocation(self._prog, "uFieldVolume")
        self._u_field_gain  = glGetUniformLocation(self._prog, "uFieldGain")
        self._u_render_pass = glGetUniformLocation(self._prog, "uRenderPass")

        # Assign every active sampler to a distinct texture unit immediately.
        # Otherwise optional samplers that are not rebound later (notably the
        # 3D field texture when --no-field is active) keep GL's default unit 0.
        # A sampler2DArray and sampler3D sharing one unit makes glDrawArrays
        # fail with GL_INVALID_OPERATION even if the branch sampling the 3D
        # texture is disabled by uFieldGain.
        glUseProgram(self._prog)
        if self._u_emit_uv != -1:
            glUniform1i(self._u_emit_uv, self._uv_tex_unit_emit)
        if self._u_color_uv != -1:
            glUniform1i(self._u_color_uv, self._uv_tex_unit_color)
        if self._u_depth_uv != -1:
            glUniform1i(self._u_depth_uv, self._uv_tex_unit_depth)
        if self._u_remit_uv != -1:
            glUniform1i(self._u_remit_uv, self._uv_tex_unit_remit)
        if self._u_field_vol != -1:
            glUniform1i(self._u_field_vol, self._uv_tex_unit_field)
        if self._u_field_gain != -1:
            glUniform1f(self._u_field_gain, 0.0)
        glUseProgram(0)

    def _build_ssbos(self) -> None:
        """Create and populate the three material SSBOs from the current database state."""
        t = self._db.build_tensors()
        if t is self._mat_tensors_ref:
            return

        pbr_chunk    = t.get('pbr', np.zeros((0, 16), np.float32))
        phong_chunk  = t.get('phong_compat', np.zeros((0, 8), np.float32))
        enamel_chunk = t.get('enamel', np.zeros((0, 8), np.float32))
        texstack_chunk = t.get('texture_stack', np.zeros((0, 16), np.float32))

        for binding, data in (
            (_BINDING_PBR,    pbr_chunk),
            (_BINDING_PHONG,  phong_chunk),
            (_BINDING_ENAMEL, enamel_chunk),
            (_BINDING_TEXSTACK, texstack_chunk),
        ):
            buf_id = self._ssbo.get(binding, None)
            if buf_id is None:
                buf_id = glGenBuffers(1)
                self._ssbo[binding] = buf_id

            # Ensure contiguous float32 row-major layout
            flat = np.ascontiguousarray(data, dtype=np.float32).ravel()

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_id)
            glBufferData(
                GL_SHADER_STORAGE_BUFFER,
                flat.nbytes,
                flat.ctypes.data_as(ctypes.c_void_p),
                GL_STATIC_DRAW,
            )
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf_id)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self._mat_tensors_ref = t

    def update_material_ssbo(self) -> None:
        """Re-upload SSBO data after the material database has changed."""
        self._build_ssbos()

    def set_point_lights(self, positions: "np.ndarray", colors: "np.ndarray",
                         intensities: "np.ndarray",
                         group_ids: "np.ndarray | None" = None) -> None:
        """Set view-space point emitters derived by the caller from scene groups."""
        pos = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        col = np.ascontiguousarray(colors, dtype=np.float32).reshape(-1, 3)
        inten = np.ascontiguousarray(intensities, dtype=np.float32).reshape(-1)
        if group_ids is None:
            gids = np.full((pos.shape[0],), -1, dtype=np.int32)
        else:
            gids = np.ascontiguousarray(group_ids, dtype=np.int32).reshape(-1)
        n = min(100, pos.shape[0], col.shape[0], inten.shape[0])
        self._light_pos = pos[:n]
        self._light_color = col[:n]
        self._light_intensity = inten[:n]
        self._light_group_id = gids[:n] if gids.shape[0] >= n else np.pad(gids, (0, n - gids.shape[0]), constant_values=-1)

    def derive_emissive_area_lights(
        self,
        verts8: "np.ndarray",
        mat_per_vertex: "np.ndarray | None" = None,
        group_per_vertex: "np.ndarray | None" = None,
        *,
        groups: "tuple | None" = None,
        min_emitter_group_id: int = 10,
        max_lights: int = 100,
    ) -> None:
        """Derive emitter lights from mesh geometry/materials (area-weighted).

        This keeps light derivation in the renderer side from emissive surfaces,
        instead of requiring host-prebaked light arrays.
        """
        verts = np.ascontiguousarray(verts8, dtype=np.float32)
        agg: dict[int, dict[str, np.ndarray | float | int]] = {}

        if groups is not None:
            gids, mids, offs, cnts = groups[0], groups[1], groups[2], groups[3]
            gids = np.ascontiguousarray(gids, dtype=np.int32).reshape(-1)
            mids = np.ascontiguousarray(mids, dtype=np.int32).reshape(-1)
            offs = np.ascontiguousarray(offs, dtype=np.int32).reshape(-1)
            cnts = np.ascontiguousarray(cnts, dtype=np.int32).reshape(-1)
            n_groups = min(gids.shape[0], mids.shape[0], offs.shape[0], cnts.shape[0])
            for i in range(n_groups):
                gid = int(gids[i])
                if gid < min_emitter_group_id:
                    continue
                off = int(offs[i])
                cnt = int(cnts[i])
                if cnt <= 0:
                    continue
                s = off * 3
                e = s + cnt * 3
                if s < 0 or e > verts.shape[0]:
                    continue
                tri = verts[s:e, 0:3].reshape(-1, 3, 3)
                if tri.size == 0:
                    continue
                e1 = tri[:, 1] - tri[:, 0]
                e2 = tri[:, 2] - tri[:, 0]
                area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
                total = float(area.sum())
                if total <= 1e-8:
                    continue
                cent = tri.mean(axis=1)
                pos_acc = (cent * area[:, None]).sum(axis=0)
                agg[gid] = {
                    "area": total,
                    "pos_acc": np.asarray(pos_acc, np.float32),
                    "mat_id": int(mids[i]),
                }
        else:
            if mat_per_vertex is None or group_per_vertex is None:
                self.set_point_lights(
                    np.zeros((0, 3), np.float32),
                    np.zeros((0, 3), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32),
                )
                return
            mats_v = np.ascontiguousarray(mat_per_vertex, dtype=np.int32).reshape(-1)
            gids_v = np.ascontiguousarray(group_per_vertex, dtype=np.int32).reshape(-1)
            if verts.shape[0] < 3 or verts.shape[0] != mats_v.shape[0] or verts.shape[0] != gids_v.shape[0]:
                self.set_point_lights(
                    np.zeros((0, 3), np.float32),
                    np.zeros((0, 3), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32),
                )
                return

            n_tri = verts.shape[0] // 3
            if n_tri <= 0:
                self.set_point_lights(
                    np.zeros((0, 3), np.float32),
                    np.zeros((0, 3), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32),
                )
                return

            tri = verts[: n_tri * 3, 0:3].reshape(n_tri, 3, 3)
            tri_m = mats_v[: n_tri * 3].reshape(n_tri, 3)[:, 0]
            tri_g = gids_v[: n_tri * 3].reshape(n_tri, 3)[:, 0]
            e1 = tri[:, 1] - tri[:, 0]
            e2 = tri[:, 2] - tri[:, 0]
            area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
            cent = tri.mean(axis=1)

            for i in range(n_tri):
                gid = int(tri_g[i])
                if gid < min_emitter_group_id:
                    continue
                a = float(area[i])
                if a <= 1e-8:
                    continue
                rec = agg.get(gid)
                if rec is None:
                    agg[gid] = {
                        "area": a,
                        "pos_acc": cent[i].astype(np.float32) * a,
                        "mat_id": int(tri_m[i]),
                    }
                else:
                    rec["area"] = float(rec["area"]) + a
                    rec["pos_acc"] = np.asarray(rec["pos_acc"], np.float32) + cent[i].astype(np.float32) * a

        pbr = np.ascontiguousarray(self._db.build_tensors().get("pbr", np.zeros((0, 16), np.float32)), np.float32)
        cands: list[tuple[float, np.ndarray, np.ndarray, int]] = []
        for gid, rec in agg.items():
            mat_id = int(rec["mat_id"])
            if mat_id < 0 or mat_id >= pbr.shape[0]:
                continue
            rgb = np.asarray(pbr[mat_id, 8:11], np.float32)
            if float(np.linalg.norm(rgb)) < 1e-6:
                continue
            total = float(rec["area"])
            if total <= 1e-8:
                continue
            pos = np.asarray(rec["pos_acc"], np.float32) / total
            cands.append((total, pos.astype(np.float32), rgb, gid))

        cands.sort(key=lambda x: x[0], reverse=True)
        cands = cands[: int(max(1, min(100, max_lights)))]
        if not cands:
            self.set_point_lights(
                np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32),
            )
            return
        self.set_point_lights(
            np.ascontiguousarray([c[1] for c in cands], np.float32),
            np.ascontiguousarray([c[2] for c in cands], np.float32),
            np.ascontiguousarray([c[0] for c in cands], np.float32),
            np.ascontiguousarray([c[3] for c in cands], np.int32),
        )

    def set_specular_enabled(self, enabled: bool) -> None:
        self._enable_specular = bool(enabled)

    def set_emission_direct_enabled(self, enabled: bool) -> None:
        self._enable_emission_direct = bool(enabled)

    def set_light_calibration(self, factor: float) -> None:
        self._light_calibration = float(factor)

    def set_cat_ccm_matrix(self, matrix: "np.ndarray") -> None:
        m = np.ascontiguousarray(matrix, dtype=np.float32)
        if m.shape != (3, 3):
            m = m.reshape(-1)
            if m.shape[0] != 9:
                raise ValueError("set_cat_ccm_matrix: matrix must be shape (3,3)")
            m = m.reshape(3, 3)
        self._cat_ccm_matrix = np.ascontiguousarray(m, dtype=np.float32)

    def _upload_feature_toggles(self) -> None:
        if self._u_enable_specular != -1:
            glUniform1i(self._u_enable_specular, 1 if self._enable_specular else 0)
        if self._u_enable_emission_direct != -1:
            glUniform1i(self._u_enable_emission_direct, 1 if self._enable_emission_direct else 0)

    def _upload_point_lights(self, mv_mat: "np.ndarray | None" = None) -> None:
        n = int(min(100, self._light_pos.shape[0]))
        if self._u_num_lights != -1:
            glUniform1i(self._u_num_lights, n)
        if n <= 0:
            return
        if self._u_light_pos != -1:
            if mv_mat is not None and n > 0:
                # Transform object-space light positions into view space so the
                # fragment shader's vPosV (also view-space) produces correct Lvec.
                R = mv_mat[:3, :3]
                t = mv_mat[:3, 3]
                pos_vs = np.ascontiguousarray(
                    (self._light_pos[:n] @ R.T) + t, dtype=np.float32
                )
                glUniform3fv(self._u_light_pos, n, pos_vs.ctypes.data_as(ctypes.c_void_p))
            else:
                glUniform3fv(self._u_light_pos, n, self._light_pos.ctypes.data_as(ctypes.c_void_p))
        if self._u_light_color != -1:
            glUniform3fv(self._u_light_color, n, self._light_color.ctypes.data_as(ctypes.c_void_p))
        if self._u_light_intensity != -1:
            glUniform1fv(self._u_light_intensity, n, self._light_intensity.ctypes.data_as(ctypes.c_void_p))
        if self._u_light_group_id != -1:
            from OpenGL.GL import glUniform1iv
            glUniform1iv(self._u_light_group_id, n, self._light_group_id.ctypes.data_as(ctypes.c_void_p))
        if self._u_light_calibration != -1:
            glUniform1f(self._u_light_calibration, float(self._light_calibration))
        if self._u_cat_ccm_matrix != -1:
            mat_col_major = np.ascontiguousarray(self._cat_ccm_matrix.T, dtype=np.float32)
            glUniformMatrix3fv(self._u_cat_ccm_matrix, 1, GL_FALSE,
                               mat_col_major.ctypes.data_as(ctypes.c_void_p))

    # ── Mesh queue ────────────────────────────────────────────────────────────

    def enqueue_mesh(
        self,
        vao: int,
        n_verts: int,
        mvp: "np.ndarray",
        mv: "np.ndarray",
    ) -> None:
        """Schedule one mesh for drawing in the next _run() call.

        Lights are not a parameter.  The engine derives illumination from
        emissive materials inside the draw — there is no host-side light
        path.  (Stage 1: emission-only via SSBO `pbr.emission`; Stage 2
        will mirror per-tri positions + mat_ids in this renderer to enable
        the same per-material cluster-light derivation the C rasterizer
        already performs.)
        """
        self._mesh_list.append((
            vao, n_verts,
            np.ascontiguousarray(mvp, dtype=np.float32).ravel(),
            np.ascontiguousarray(mv,  dtype=np.float32).ravel(),
        ))

    def clear_mesh_list(self) -> None:
        self._mesh_list.clear()

    # ── Draw ─────────────────────────────────────────────────────────────────

    def draw_mesh(
        self,
        vao: int,
        n_verts: int,
        mvp: "np.ndarray",
        mv: "np.ndarray",
        *,
        enable_blend: bool = True,
        depth_write: bool = True,
    ) -> None:
        """Draw a single mesh immediately (requires active GL context + program)."""
        from OpenGL.GL import glBindVertexArray, glDrawArrays, GL_TRIANGLES

        glUseProgram(self._prog)

        # Bind SSBOs
        for binding, buf_id in self._ssbo.items():
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf_id)

        # Bind default UV texture array on its dedicated texture unit and
        # point the sampler uniform at that unit.  With the identity texel
        # this is a no-op for current callers; once a caller uploads a real
        # emission UV array the Stage 2 fragment math activates.
        if self._u_emit_uv != -1:
            glUniform1i(self._u_emit_uv, self._uv_tex_unit_emit)
        if self._u_color_uv != -1:
            glUniform1i(self._u_color_uv, self._uv_tex_unit_color)
        if self._u_depth_uv != -1:
            glUniform1i(self._u_depth_uv, self._uv_tex_unit_depth)
        if self._u_remit_uv != -1:
            glUniform1i(self._u_remit_uv, self._uv_tex_unit_remit)
        if self._u_field_vol != -1:
            glUniform1i(self._u_field_vol, self._uv_tex_unit_field)
        if self._tex_emit_uv is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_emit)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_emit_uv)
        if self._tex_color_uv is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_color)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_color_uv)
        if self._tex_depth_uv is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_depth)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_depth_uv)
        if self._tex_remit_uv is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_remit)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_remit_uv)
        if self._tex_field_vol is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_field)
            glBindTexture(GL_TEXTURE_3D, self._tex_field_vol)
        if self._u_field_gain != -1:
            glUniform1f(self._u_field_gain, self._field_gain if self._tex_field_vol is not None else 0.0)

        # Upload uniforms
        if self._u_mvp != -1:
            glUniformMatrix4fv(self._u_mvp, 1, GL_FALSE, mvp.ctypes.data_as(ctypes.c_void_p))
        if self._u_mv != -1:
            glUniformMatrix4fv(self._u_mv, 1, GL_FALSE, mv.ctypes.data_as(ctypes.c_void_p))

        self._upload_feature_toggles()
        # mv arrives column-major (V.T.ravel()); reshape+transpose recovers row-major V.
        _mv_mat = mv.reshape(4, 4).T
        self._upload_point_lights(mv_mat=_mv_mat)

        glDisable(GL_CULL_FACE)
        glDepthMask(GL_TRUE if depth_write else GL_FALSE)
        if enable_blend:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        else:
            glDisable(GL_BLEND)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, n_verts)
        glBindVertexArray(0)
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)

        glUseProgram(0)

    # ── ShaderFrameWalker callback ────────────────────────────────────────────

    def _run(self, spec, node, fifos, frame_index: int, dt: float, **extra) -> None:
        """Called by ShaderFrameWalker each frame.  Draws all enqueued meshes.

        Invariant state (program, SSBO bindings, sampler, blend) is bound
        ONCE per frame; the inner loop does only uniform updates + draw.
        """
        if not self._mesh_list:
            return

        from OpenGL.GL import (
            glBindVertexArray, glDrawArrays, GL_TRIANGLES,
            glEnable, glDisable, glBlendFunc, GL_BLEND,
            GL_CULL_FACE,
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        )

        glUseProgram(self._prog)

        # Bind invariant per-frame state.
        for binding, buf_id in self._ssbo.items():
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf_id)

        if self._u_emit_uv != -1:
            glUniform1i(self._u_emit_uv, self._uv_tex_unit_emit)
        if self._u_color_uv != -1:
            glUniform1i(self._u_color_uv, self._uv_tex_unit_color)
        if self._u_depth_uv != -1:
            glUniform1i(self._u_depth_uv, self._uv_tex_unit_depth)
        if self._u_remit_uv != -1:
            glUniform1i(self._u_remit_uv, self._uv_tex_unit_remit)
        if self._u_field_vol != -1:
            glUniform1i(self._u_field_vol, self._uv_tex_unit_field)
        if self._tex_emit_uv is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_emit)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_emit_uv)
        if self._tex_color_uv is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_color)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_color_uv)
        if self._tex_depth_uv is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_depth)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_depth_uv)
        if self._tex_remit_uv is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_remit)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_remit_uv)
        if self._tex_field_vol is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_field)
            glBindTexture(GL_TEXTURE_3D, self._tex_field_vol)
        if self._u_field_gain != -1:
            glUniform1f(self._u_field_gain, self._field_gain if self._tex_field_vol is not None else 0.0)

        self._upload_feature_toggles()
        self._upload_point_lights()

        glDisable(GL_CULL_FACE)

        u_mvp = self._u_mvp
        u_mv  = self._u_mv
        u_rp  = self._u_render_pass
        c_void_p = ctypes.c_void_p

        # Pass 1 — opaque fragments (alpha >= 0.85): depth-write on, blend off
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        if u_rp != -1:
            glUniform1i(u_rp, 1)
        for entry in self._mesh_list:
            vao, n_verts, mvp, mv = entry
            if u_mvp != -1:
                glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp.ctypes.data_as(c_void_p))
            if u_mv != -1:
                glUniformMatrix4fv(u_mv, 1, GL_FALSE, mv.ctypes.data_as(c_void_p))
            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, n_verts)

        # Pass 2 — transparent fragments (alpha < 0.85): depth-write off, blend on
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        if u_rp != -1:
            glUniform1i(u_rp, 2)
        for entry in self._mesh_list:
            vao, n_verts, mvp, mv = entry
            if u_mvp != -1:
                glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp.ctypes.data_as(c_void_p))
            if u_mv != -1:
                glUniformMatrix4fv(u_mv, 1, GL_FALSE, mv.ctypes.data_as(c_void_p))
            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, n_verts)

        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        if u_rp != -1:
            glUniform1i(u_rp, 0)  # reset so draw_mesh callers see uRenderPass=0 (draw all)
        glUseProgram(0)

        if self._auto_drain:
            self._mesh_list.clear()

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, owner_id: str = "global", shader_id: str = "base_material") -> None:
        """No-op. Base material is a *global default* 3D-channel renderer;
        it is invoked on whatever the registered shaders did not finalise
        (the leftover mask) by the unified resolve.
        """
        return
