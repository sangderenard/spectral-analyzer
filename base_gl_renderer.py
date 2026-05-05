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
        glUniformMatrix4fv, glUniform3f, glUniform1f, glUniform1i,
        glUniform3fv, glUniform1fv,
        glGetUniformLocation, glEnable, glDisable, glBlendFunc,
        glGenTextures, glBindTexture, glTexParameteri, glTexImage3D,
        glActiveTexture,
        GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS, GL_LINK_STATUS,
        GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, GL_STATIC_DRAW,
        GL_FLOAT, GL_TRUE, GL_FALSE, GL_BLEND,
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        GL_TEXTURE_2D_ARRAY, GL_TEXTURE0, GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
        GL_LINEAR, GL_CLAMP_TO_EDGE, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE,
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
        self._u_light_dir       = -1
        self._u_light_color     = -1
        self._u_light_intensity = -1

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
        self._uv_tex_unit_emit = 0
        self._u_emit_uv = -1

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
        # Identity texel: R=0 (no direct gain), G=255 (full diffuse pass-through),
        # B=128 (≈0.5 saturation identity), A=255 (no dim).
        default_texel = np.array([0, 255, 128, 255], dtype=np.uint8)
        glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
            1, 1, 1,                      # width, height, layers
            0, GL_RGBA, GL_UNSIGNED_BYTE,
            default_texel.ctypes.data_as(ctypes.c_void_p),
        )
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        self._tex_emit_uv = int(tex)

    def _build_program(self) -> None:
        vert_src = _read_glsl(_VERT_PATH)
        frag_src = _read_glsl(_FRAG_PATH)
        vert     = _compile_shader(vert_src, GL_VERTEX_SHADER)
        frag     = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
        self._prog = _link_program(vert, frag)

        self._u_mvp             = glGetUniformLocation(self._prog, "uMVP")
        self._u_mv              = glGetUniformLocation(self._prog, "uMV")
        self._u_num_lights      = glGetUniformLocation(self._prog, "uNumLights")
        self._u_light_dir       = glGetUniformLocation(self._prog, "uLightDir")
        self._u_light_color     = glGetUniformLocation(self._prog, "uLightColor")
        self._u_light_intensity = glGetUniformLocation(self._prog, "uLightIntensity")
        self._u_emit_uv         = glGetUniformLocation(self._prog, "uEmitUv")

    def _build_ssbos(self) -> None:
        """Create and populate the three material SSBOs from the current database state."""
        t = self._db.build_tensors()
        if t is self._mat_tensors_ref:
            return

        pbr_chunk    = t.get('pbr', np.zeros((0, 16), np.float32))
        phong_chunk  = t.get('phong_compat', np.zeros((0, 8), np.float32))
        enamel_chunk = t.get('enamel', np.zeros((0, 8), np.float32))

        for binding, data in (
            (_BINDING_PBR,    pbr_chunk),
            (_BINDING_PHONG,  phong_chunk),
            (_BINDING_ENAMEL, enamel_chunk),
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
        if self._tex_emit_uv is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_emit)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_emit_uv)
            if self._u_emit_uv != -1:
                glUniform1i(self._u_emit_uv, self._uv_tex_unit_emit)

        # Upload uniforms
        if self._u_mvp != -1:
            glUniformMatrix4fv(self._u_mvp, 1, GL_FALSE, mvp.ctypes.data_as(ctypes.c_void_p))
        if self._u_mv != -1:
            glUniformMatrix4fv(self._u_mv, 1, GL_FALSE, mv.ctypes.data_as(ctypes.c_void_p))

        # No host-side lights.  Until the CPU geometry mirror lands here,
        # the GL path renders self-emission only (uNumLights=0).  That is
        # an honest under-shading, not fake fill.
        if self._u_num_lights != -1:
            glUniform1i(self._u_num_lights, 0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, n_verts)
        glBindVertexArray(0)
        glDisable(GL_BLEND)

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
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        )

        glUseProgram(self._prog)

        # Bind invariant per-frame state.
        for binding, buf_id in self._ssbo.items():
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf_id)

        if self._tex_emit_uv is not None:
            glActiveTexture(GL_TEXTURE0 + self._uv_tex_unit_emit)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_emit_uv)
            if self._u_emit_uv != -1:
                glUniform1i(self._u_emit_uv, self._uv_tex_unit_emit)

        if self._u_num_lights != -1:
            glUniform1i(self._u_num_lights, 0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        u_mvp = self._u_mvp
        u_mv  = self._u_mv
        c_void_p = ctypes.c_void_p
        for entry in self._mesh_list:
            vao, n_verts, mvp, mv = entry
            if u_mvp != -1:
                glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp.ctypes.data_as(c_void_p))
            if u_mv != -1:
                glUniformMatrix4fv(u_mv, 1, GL_FALSE, mv.ctypes.data_as(c_void_p))
            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, n_verts)

        glBindVertexArray(0)
        glDisable(GL_BLEND)
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
