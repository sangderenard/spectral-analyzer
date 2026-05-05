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
draws every (VAO, n_verts, mvp, mv, light_v, scene_rgb, scene_indirect) tuple
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
        glUniformMatrix4fv, glUniform3f, glUniform1f,
        glGetUniformLocation, glEnable, glDisable, glBlendFunc,
        GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS, GL_LINK_STATUS,
        GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, GL_STATIC_DRAW,
        GL_FLOAT, GL_TRUE, GL_FALSE, GL_BLEND,
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
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
        self._u_light_v         = -1
        self._u_scene_rgb       = -1
        self._u_scene_indirect  = -1

        # Mesh draw queue: list of tuples (vao, n_verts, mvp, mv, light_v,
        #                                  scene_rgb, scene_indirect)
        # Types: vao=int, n_verts=int, mvp=np.ndarray(16,f32),
        #        mv=np.ndarray(16,f32), light_v=np.ndarray(3,f32),
        #        scene_rgb=np.ndarray(3,f32), scene_indirect=float
        self._mesh_list: List[Tuple] = []

        self._registered = False

    # ── GL initialisation ─────────────────────────────────────────────────────

    def init_gl(self) -> None:
        """Compile shaders and upload material SSBOs.  Call once after GL context is current."""
        if not _GL_OK:
            raise RuntimeError("[BaseGLRenderer] PyOpenGL not available")
        self._build_program()
        self._build_ssbos()

    def _build_program(self) -> None:
        vert_src = _read_glsl(_VERT_PATH)
        frag_src = _read_glsl(_FRAG_PATH)
        vert     = _compile_shader(vert_src, GL_VERTEX_SHADER)
        frag     = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
        self._prog = _link_program(vert, frag)

        self._u_mvp            = glGetUniformLocation(self._prog, "uMVP")
        self._u_mv             = glGetUniformLocation(self._prog, "uMV")
        self._u_light_v        = glGetUniformLocation(self._prog, "uLightV")
        self._u_scene_rgb      = glGetUniformLocation(self._prog, "uSceneRgb")
        self._u_scene_indirect = glGetUniformLocation(self._prog, "uSceneIndirectRatio")

    def _build_ssbos(self) -> None:
        """Create and populate the three material SSBOs from the current database state."""
        pbr_chunk    = self._db.pbr_chunk()    # (N,16) float32
        phong_chunk  = self._db.phong_chunk()  # (N, 8) float32
        enamel_chunk = self._db.enamel_chunk() # (N, 8) float32

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
        light_v: "np.ndarray",
        scene_rgb: "np.ndarray",
        scene_indirect: float,
    ) -> None:
        """Schedule one mesh for drawing in the next _run() call.

        Parameters
        ----------
        vao:
            OpenGL VAO name. The VAO must bind:
              attrib 0 → vec3 position  (offset 0, stride 24 bytes)
              attrib 1 → vec3 normal    (offset 12, stride 24 bytes)
              attrib 2 → int  mat_id    (separate integer attrib, or packed)
        n_verts:
            Vertex count to pass to glDrawArrays(GL_TRIANGLES, 0, n_verts).
        mvp:
            4×4 float32 ndarray (column-major, shape (4,4) or (16,)).
        mv:
            4×4 float32 ndarray (column-major, shape (4,4) or (16,)).
        light_v:
            View-space light direction, shape (3,), float32.
        scene_rgb:
            Environment spectral tint, shape (3,), float32.
        scene_indirect:
            Indirect fill fraction [0, 1], scalar float.
        """
        self._mesh_list.append((
            vao, n_verts,
            np.ascontiguousarray(mvp, dtype=np.float32).ravel(),
            np.ascontiguousarray(mv,  dtype=np.float32).ravel(),
            np.ascontiguousarray(light_v,   dtype=np.float32).ravel(),
            np.ascontiguousarray(scene_rgb, dtype=np.float32).ravel(),
            float(scene_indirect),
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
        light_v: "np.ndarray",
        scene_rgb: "np.ndarray",
        scene_indirect: float,
    ) -> None:
        """Draw a single mesh immediately (requires active GL context + program)."""
        from OpenGL.GL import glBindVertexArray, glDrawArrays, GL_TRIANGLES

        glUseProgram(self._prog)

        # Bind SSBOs
        for binding, buf_id in self._ssbo.items():
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf_id)

        # Upload uniforms
        if self._u_mvp != -1:
            glUniformMatrix4fv(self._u_mvp, 1, GL_FALSE, mvp.ctypes.data_as(ctypes.c_void_p))
        if self._u_mv != -1:
            glUniformMatrix4fv(self._u_mv, 1, GL_FALSE, mv.ctypes.data_as(ctypes.c_void_p))
        if self._u_light_v != -1:
            glUniform3f(self._u_light_v, float(light_v[0]), float(light_v[1]), float(light_v[2]))
        if self._u_scene_rgb != -1:
            glUniform3f(self._u_scene_rgb, float(scene_rgb[0]), float(scene_rgb[1]), float(scene_rgb[2]))
        if self._u_scene_indirect != -1:
            glUniform1f(self._u_scene_indirect, float(scene_indirect))

        # Draw
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, n_verts)
        glBindVertexArray(0)
        glDisable(GL_BLEND)

        glUseProgram(0)

    # ── ShaderFrameWalker callback ────────────────────────────────────────────

    def _run(self, spec, node, fifos, frame_index: int, dt: float, **extra) -> None:
        """Called by ShaderFrameWalker each frame.  Draws all enqueued meshes."""
        for entry in self._mesh_list:
            vao, n_verts, mvp, mv, lv, sr, si = entry
            self.draw_mesh(vao, n_verts, mvp, mv, lv, sr, si)
        if self._auto_drain:
            self._mesh_list.clear()

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, owner_id: str = "global", shader_id: str = "base_material") -> None:
        """No-op. Base material is a *global default* 3D-channel renderer;
        it is invoked on whatever the registered shaders did not finalise
        (the leftover mask) by the unified resolve.
        """
        return
