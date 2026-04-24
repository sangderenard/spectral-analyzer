"""opengl_widget.py
==================
Modern OpenGL 3.0 / GLSL 1.30 viewport widgets for pygame-based applications.

All rendering is offscreen (FBO) so it can be called mid-frame from inside
analytic_driver / bass_viewer without disturbing the host GL state.

Architecture
------------
  GLViewportWidget
    Base.  Manages one dedicated FBO (colour texture + depth renderbuffer),
    one VAO + VBO, perspective or orthographic camera, and an assignable GLSL
    shader programme.
    Subclasses override _build_geometry() to fill self._vertices (numpy array,
    shape (N, 5), columns [x, y, z, phase_angle, component_id]) and
    self._draw_calls (list of (GL_mode, first, count, line_width, alpha_boost)).

  ComplexPhaseCloud(GLViewportWidget)
    Phase-rotating analytic signal display.
    N_phases discrete rotation angles are applied to every point, giving
    N_phases full rotated copies of the signal stored as one VBO.

    Pass 1 - Cloud  (GL_POINTS, all vertices)
      Vertex shader activates points near the current phase angle via a
      Gaussian falloff; hue rotates with phase, giving the probability-cloud /
      neon-comet effect.

    Pass 2 - Active lines  (GL_LINE_STRIP x 3 per frame)
      The three component strips for the currently-active phase angle are
      drawn as connected lines on top of the cloud.

    Perspective modes: "perspective", "ortho", "top", "re_plane", "lissajous"
    Auto-fit: camera distance scales with data peak.

  GLPanel
    Panel-interface wrapper (bass_viewer PanelDock compatible).

Coordinate convention
---------------------
  X = time  (-1 to +1, left to right)
  Y = real amplitude (up = +1)
  Z = imaginary amplitude (negative Z = into screen in default view)

Requirements: PyOpenGL >= 3.1, numpy, pygame
"""
from __future__ import annotations

import ctypes
import math
from typing import Any, Optional

import numpy as np

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER,
        GL_BLEND,
        GL_COLOR_BUFFER_BIT,
        GL_CURRENT_PROGRAM,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_DYNAMIC_DRAW,
        GL_FLOAT,
        GL_FRAGMENT_SHADER,
        GL_LINE_SMOOTH,
        GL_LINE_SMOOTH_HINT,
        GL_LINE_STRIP,
        GL_LINES,
        GL_NICEST,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_POINTS,
        GL_PROGRAM_POINT_SIZE,
        GL_SCISSOR_TEST,
        GL_SRC_ALPHA,
        GL_VERTEX_SHADER,
        GL_VIEWPORT,
        glAttachShader,
        glBindBuffer,
        glBindVertexArray,
        glBlendFunc,
        glBufferData,
        glClear,
        glClearColor,
        glClearDepth,
        glCompileShader,
        glCreateProgram,
        glCreateShader,
        glDeleteBuffers,
        glDeleteProgram,
        glDeleteShader,
        glDeleteVertexArrays,
        glDisable,
        glDrawArrays,
        glEnable,
        glEnableVertexAttribArray,
        glGenBuffers,
        glGenVertexArrays,
        glGetAttribLocation,
        glGetIntegerv,
        glGetUniformLocation,
        glHint,
        glLineWidth,
        glLinkProgram,
        glScissor,
        glShaderSource,
        glUniform1f,
        glUniform1i,
        glUniformMatrix4fv,
        glUseProgram,
        glVertexAttribPointer,
        glViewport,
    )
    _HAS_GL = True
except ImportError:
    _HAS_GL = False


# ---------------------------------------------------------------------------
# Cloud display modes  (uDisplayMode uniform)
# ---------------------------------------------------------------------------
# 0  SINGLE      Only the active slot is drawn; inactive vertices are clipped
#                before the MVP multiply.  Cheapest — use for single-angle
#                playback.
#
# 1  HUE_CYCLE   All slots lit simultaneously.  Hue is offset by uCurrentSlot
#                so the colour wheel slips forward as playback advances.
#                uCloudAlpha controls inactive-slot opacity; active slot gets
#                uAlphaBoost applied on top.
#
# Add new modes here and branch in void main() below.

CLOUD_MODE_SINGLE    = 0
CLOUD_MODE_HUE_CYCLE = 1

# ---------------------------------------------------------------------------
# GLSL shader sources
# ---------------------------------------------------------------------------
# Vertex layout (4 floats per vertex, stride = 16 bytes):
#   offset  0: vec3  aPos         (x, y, z)
#   offset 12: float aComponent   (0=real, 1=imag, 2=phasor, 3=axis)
#
# Slot is derived in the shader via gl_VertexID integer division:
#   slot = (gl_VertexID - uAxCount) % (N * uNPts) / uNPts

_CLOUD_VERT_SRC = """
#version 130
uniform mat4  uMVP;
uniform int   uCurrentSlot;
uniform int   uNPts;
uniform int   uAxCount;
uniform int   uDisplayMode;
uniform float uCloudAlpha;
uniform float uPointSize;
uniform float uAlphaBoost;

in vec3  aPos;
in float aComponent;

out vec4 vColor;

vec3 hue_rgb(float h) {
    float r = abs(h * 6.0 - 3.0) - 1.0;
    float g = 2.0 - abs(h * 6.0 - 2.0);
    float b = 2.0 - abs(h * 6.0 - 4.0);
    return clamp(vec3(r, g, b), 0.0, 1.0);
}

void main() {
    const int N = 72;

    // Axis lines bypass all mode logic.
    if (aComponent > 2.5) {
        gl_Position  = uMVP * vec4(aPos, 1.0);
        gl_PointSize = uPointSize;
        vColor = vec4(0.35, 0.37, 0.42, 0.65);
        return;
    }

    // Cloud vertices start at uAxCount.
    // Layout: [real: N*uNPts | imag: N*uNPts | phasor: N*uNPts]
    // Within each block: slot 0..N-1, each uNPts points (row-major).
    int cid  = gl_VertexID - uAxCount;
    int slot = (cid % (N * uNPts)) / uNPts;
    bool active = (slot == uCurrentSlot);

    // ---- mode 0: SINGLE ------------------------------------------------
    if (uDisplayMode == 0) {
        if (!active) {
            // Clip before MVP multiply — no fragment invocations for inactive.
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            vColor = vec4(0.0);
            return;
        }
        gl_Position  = uMVP * vec4(aPos, 1.0);
        gl_PointSize = uPointSize;
        float hue = float(slot) / float(N);
        vColor = vec4(hue_rgb(hue), clamp(0.92 * uAlphaBoost, 0.0, 1.0));
        return;
    }

    // ---- mode 1: HUE_CYCLE ---------------------------------------------
    if (uDisplayMode == 1) {
        gl_Position  = uMVP * vec4(aPos, 1.0);
        gl_PointSize = uPointSize;
        // Hue offset by current slot so the wheel slips as playback advances.
        float hue   = fract(float(slot - uCurrentSlot) / float(N));
        float alpha = active ? clamp(uAlphaBoost * 0.92, 0.0, 1.0)
                             : uCloudAlpha;
        vColor = vec4(hue_rgb(hue), 1.0);
        return;
    }
}
"""

_CLOUD_FRAG_SRC = """
#version 130
in  vec4 vColor;
void main() {
    gl_FragColor = vColor;
}
"""


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------

def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = (center - eye).astype(np.float64)
    n = np.linalg.norm(f)
    if n < 1e-12:
        f = np.array([0.0, 0.0, -1.0])
    else:
        f /= n
    r = np.cross(f, up.astype(np.float64))
    nr = np.linalg.norm(r)
    if nr < 1e-12:
        r = np.array([1.0, 0.0, 0.0])
    else:
        r /= nr
    u = np.cross(r, f)
    mat = np.eye(4, dtype=np.float32)
    mat[0, :3] = r.astype(np.float32)
    mat[1, :3] = u.astype(np.float32)
    mat[2, :3] = (-f).astype(np.float32)
    mat[0, 3]  = float(-np.dot(r, eye))
    mat[1, 3]  = float(-np.dot(u, eye))
    mat[2, 3]  = float( np.dot(f, eye))
    return mat


def _perspective_mat(fov_y_deg: float, aspect: float,
                     near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / max(aspect, 1e-6)
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def _ortho_mat(left: float, right: float, bottom: float, top: float,
               near: float, far: float) -> np.ndarray:
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] =  2.0 / (right - left)
    mat[1, 1] =  2.0 / (top - bottom)
    mat[2, 2] = -2.0 / (far - near)
    mat[0, 3] = -(right + left) / (right - left)
    mat[1, 3] = -(top + bottom) / (top - bottom)
    mat[2, 3] = -(far + near)   / (far - near)
    mat[3, 3] =  1.0
    return mat


_CAMERA_PRESETS: dict = {
    "perspective": dict(eye=(0.4,  0.7,  2.0), center=(0.0, 0.0, 0.0),
                        up=(0.0,   1.0,  0.0), fov=45.0, use_ortho=False),
    "ortho":       dict(eye=(0.4,  0.7,  2.0), center=(0.0, 0.0, 0.0),
                        up=(0.0,   1.0,  0.0), fov=45.0, use_ortho=True),
    "top":         dict(eye=(0.0,  3.0,  0.01), center=(0.0, 0.0, 0.0),
                        up=(0.0,   0.0, -1.0), fov=50.0, use_ortho=False),
    "re_plane":    dict(eye=(0.0,  0.0,  3.0), center=(0.0, 0.0, 0.0),
                        up=(0.0,   1.0,  0.0), fov=50.0, use_ortho=False),
    "lissajous":   dict(eye=(3.0,  0.1,  0.1), center=(0.0, 0.0, 0.0),
                        up=(0.0,   1.0,  0.0), fov=50.0, use_ortho=False),
}

PERSPECTIVE_MODES  = list(_CAMERA_PRESETS.keys())
PERSPECTIVE_LABELS = {
    "perspective": "3D",
    "ortho":       "Orth",
    "top":         "Top",
    "re_plane":    "Re",
    "lissajous":   "Liss",
}


# ---------------------------------------------------------------------------
# GLViewportWidget - reusable FBO + shader base
# ---------------------------------------------------------------------------

class GLViewportWidget:
    """Offscreen OpenGL viewport.  Subclass and override _build_geometry().

    Default vertex format: 4 floats per vertex (x, y, z, component_id).
    Subclasses may override _STRIDE and _setup_vao for different layouts.
    """

    _STRIDE: int = 4 * 4   # bytes per vertex

    def __init__(self,
                 vert_src: str = _CLOUD_VERT_SRC,
                 frag_src: str = _CLOUD_FRAG_SRC,
                 bg_color: tuple = (0.06, 0.06, 0.09, 1.0)):
        self._vert_src = vert_src
        self._frag_src = frag_src
        self.bg_color  = bg_color

        self._prog: Optional[int] = None
        self._vao:  Optional[int] = None
        self._vbo:  Optional[int] = None

        # shape (N, 5) float32; draw_calls: (mode, first, count, lw, alpha_boost)
        self._vertices:   np.ndarray = np.zeros((0, 5), dtype=np.float32)
        self._draw_calls: list       = []
        self._geo_dirty:  bool       = True

        self._current_phase: float = 0.0
        self._cloud_alpha:   float = 0.04
        self._point_size:    float = 2.0

        # Caches: avoids per-frame glGetUniformLocation and MVP recomputation
        self._uloc_cache: dict[str, int] = {}
        self._mvp_key:    Optional[tuple] = None
        self._mvp_val:    Optional[np.ndarray] = None

    # -- public API -----------------------------------------------------------

    def set_shader(self, vert_src: str, frag_src: str) -> None:
        self._vert_src = vert_src
        self._frag_src = frag_src
        if self._prog is not None:
            glDeleteProgram(self._prog)
            self._prog = None
        if self._vao is not None:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
            self._vao = self._vbo = None
        self._uloc_cache.clear()
        self._mvp_key = None

    def _uloc(self, name: str) -> int:
        """Return cached uniform location — avoids a shader string-lookup every frame."""
        loc = self._uloc_cache.get(name, -2)
        if loc == -2:
            loc = glGetUniformLocation(self._prog, name)
            self._uloc_cache[name] = loc
        return loc

    def draw(self, x: int, y: int, w: int, h: int, win_w: int, win_h: int) -> None:
        """Render directly into the current framebuffer at the given screen rect.
        (x, y) is the top-left corner in pygame window coordinates (y from top).
        Saves and restores the GL viewport and current shader program.
        """
        if not _HAS_GL:
            return
        prev_viewport = list(glGetIntegerv(GL_VIEWPORT))
        prev_prog     = int(glGetIntegerv(GL_CURRENT_PROGRAM))
        try:
            self._ensure_resources(w, h)
            self._render_scene(x, y, w, h, win_w, win_h)
        finally:
            glViewport(int(prev_viewport[0]), int(prev_viewport[1]),
                       int(prev_viewport[2]), int(prev_viewport[3]))
            glUseProgram(prev_prog)

    def invalidate_geometry(self) -> None:
        self._geo_dirty = True

    def destroy(self) -> None:
        if not _HAS_GL:
            return
        pairs = [
            (self._vbo,  lambda h: glDeleteBuffers(1, [h])),
            (self._vao,  lambda h: glDeleteVertexArrays(1, [h])),
            (self._prog, glDeleteProgram),
        ]
        for h, fn in pairs:
            if h is not None:
                try:
                    fn(h)
                except Exception:
                    pass
        self._prog = self._vao = self._vbo = None
        self._uloc_cache.clear()
        self._mvp_key = None

    # -- GL resource management -----------------------------------------------

    def _ensure_resources(self, w: int, h: int) -> None:
        if self._prog is None:
            self._prog = self._compile_programme(self._vert_src, self._frag_src)
        if self._vao is None:
            self._vao = glGenVertexArrays(1)
            self._vbo = glGenBuffers(1)
            self._setup_vao()
        if self._geo_dirty:
            self._build_geometry()
            self._upload_geometry()
            self._geo_dirty = False

    @staticmethod
    def _compile_programme(vert_src: str, frag_src: str) -> int:
        def _sh(src, kind):
            sh = glCreateShader(kind)
            glShaderSource(sh, src)
            glCompileShader(sh)
            return sh
        vs   = _sh(vert_src, GL_VERTEX_SHADER)
        fs   = _sh(frag_src, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vs)
        glAttachShader(prog, fs)
        glLinkProgram(prog)
        glDeleteShader(vs)
        glDeleteShader(fs)
        return prog

    def _setup_vao(self) -> None:
        stride = self._STRIDE
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        for name, n_comp, byte_off in [
            ("aPos",       3, 0),
            ("aComponent", 1, 12),
        ]:
            loc = glGetAttribLocation(self._prog, name)
            if loc >= 0:
                glEnableVertexAttribArray(loc)
                glVertexAttribPointer(loc, n_comp, GL_FLOAT, False,
                                      stride, ctypes.c_void_p(byte_off))
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _build_geometry(self) -> None:
        """Override: populate self._vertices (N,5) and self._draw_calls."""
        self._vertices   = np.zeros((0, 5), dtype=np.float32)
        self._draw_calls = []

    def _upload_geometry(self) -> None:
        if self._vertices.size == 0:
            return
        data = np.ascontiguousarray(self._vertices, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _build_mvp(self, w: int, h: int) -> np.ndarray:
        eye    = np.array([0.4, 0.7, 2.0], dtype=np.float64)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        up     = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        view   = _look_at(eye, center, up)
        proj   = _perspective_mat(45.0, w / max(h, 1), 0.05, 50.0)
        return (proj @ view).astype(np.float32)

    def _set_uniforms(self, prog: int) -> None:
        """Override to set shader-specific uniforms."""

    def _after_draw_calls(self, w: int, h: int) -> None:
        """Hook called after _draw_calls loop, while VAO and program are still bound."""

    def _render_scene(self, x: int, y: int, w: int, h: int,
                      win_w: int, win_h: int) -> None:
        # Convert pygame top-left coords to GL bottom-left coords
        gl_x = int(x)
        gl_y = int(win_h - y - h)
        glViewport(gl_x, gl_y, w, h)
        glScissor(gl_x, gl_y, w, h)
        glEnable(GL_SCISSOR_TEST)
        r, g, b, a = self.bg_color
        glClearColor(r, g, b, a)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        try:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        except Exception:
            pass
        try:
            glEnable(GL_PROGRAM_POINT_SIZE)
        except Exception:
            pass

        if len(self._vertices) == 0 or not self._draw_calls:
            glDisable(GL_SCISSOR_TEST)
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)
            return

        glUseProgram(self._prog)

        mvp = self._build_mvp(w, h)
        loc_mvp = self._uloc("uMVP")
        if loc_mvp >= 0:
            glUniformMatrix4fv(loc_mvp, 1, True, mvp)

        self._set_uniforms(self._prog)
        loc_ab = self._uloc("uAlphaBoost")

        glBindVertexArray(self._vao)
        for entry in self._draw_calls:
            mode, first, count, lw, alpha_boost = entry
            if count <= 0:
                continue
            if lw > 0:
                glLineWidth(float(lw))
            if loc_ab >= 0:
                glUniform1f(loc_ab, float(alpha_boost))
            glDrawArrays(mode, first, count)
        self._after_draw_calls(w, h)
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)


# ---------------------------------------------------------------------------
# ComplexPhaseCloud - analytic signal / EM-wave display with phase cloud
# ---------------------------------------------------------------------------

class ComplexPhaseCloud(GLViewportWidget):
    """Phase-rotating analytic signal display with haze cloud.

    Three display components in true 3-D world space:
      Real      (t, re,  0 ) -- oscillates in XY plane
      Imaginary (t, 0,  im ) -- oscillates in XZ plane
      Phasor    (t, re, im ) -- tip of the complex vector (helix)

    N_phases discrete rotation angles are pre-computed and cached in the VBO.
    The vertex shader activates points near the current phase and dims the
    rest, producing the probability-cloud / neon-comet effect.
    """

    def __init__(self,
                 phase_steps:      int   = 72,
                 perspective_mode: str   = "perspective",
                 cloud_alpha:      float = 0.06,
                 point_size:       float = 2.0,
                 auto_fit:         bool  = True,
                 display_mode:     int   = CLOUD_MODE_HUE_CYCLE,
                 show_axes:        bool  = False,
                 show_real:        bool  = False,
                 show_imag:        bool  = False,
                 show_phasor:      bool  = True,
                 **kw):
        super().__init__(**kw)
        self.phase_steps      = phase_steps
        self.perspective_mode = perspective_mode
        self._cloud_alpha     = cloud_alpha
        self._point_size      = point_size
        self._auto_fit        = auto_fit
        self._display_mode    = display_mode
        self.show_axes        = show_axes
        self.show_real        = show_real
        self.show_imag        = show_imag
        self.show_phasor      = show_phasor

        self._csig_batch:   Optional[np.ndarray] = None   # (N_phases, n) complex128
        self._current_slot: int  = 0
        self._cam_dist:    float = 2.2

        # VBO layout metadata set in _build_geometry
        self._N_phases:   int = 0
        self._n_pts_each: int = 0
        self._ax_cnt: int = 0

    # -- public API -----------------------------------------------------------

    def update_data(self, csig_batch: np.ndarray) -> None:
        """Accept an (N_phases, n_samples) complex128 array of pre-synthesized phase slots."""
        self._csig_batch = csig_batch
        self._geo_dirty  = True
        if self._auto_fit and csig_batch is not None and csig_batch.size > 0:
            peak = float(np.max(np.abs(csig_batch)))
            self._cam_dist = max(peak, 0.1) * 2.6

    def set_current_slot(self, slot: int) -> None:
        self._current_slot = int(slot) % max(2, self.phase_steps)

    def set_perspective_mode(self, mode: str) -> None:
        if mode in _CAMERA_PRESETS:
            self.perspective_mode = mode

    def set_display_mode(self, mode: int) -> None:
        self._display_mode = int(mode)

    def draw(self, x: int, y: int, w: int, h: int,  # type: ignore[override]
             win_w: int, win_h: int, current_slot: int = 0) -> None:
        self.set_current_slot(current_slot)
        super().draw(x, y, w, h, win_w, win_h)

    # -- geometry builder ----------------------------------------------------

    def _build_geometry(self) -> None:
        csig = self._csig_batch
        if csig is None or csig.ndim != 2 or csig.shape[1] < 2:
            self._vertices   = np.zeros((0, 5), dtype=np.float32)
            self._draw_calls = []
            self._N_phases   = 0
            self._n_pts_each = 0
            self._ax_cnt     = 0
            return

        N, n = csig.shape
        self._N_phases   = N
        self._n_pts_each = n

        peak = float(np.max(np.abs(csig)))
        if peak > 1e-12:
            csig = csig / peak

        x_arr = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        x_2d  = np.broadcast_to(x_arr[np.newaxis, :], (N, n))
        zeros = np.zeros((N, n), dtype=np.float32)
        re_rot = csig.real.astype(np.float32)
        im_rot = csig.imag.astype(np.float32)

        def _pack(x, y, z, comp_val):
            comp = np.full((N, n), comp_val, dtype=np.float64)
            return np.stack([x, y, z, comp], axis=-1).reshape(N * n, 4).astype(np.float32)

        real_v   = _pack(x_2d, re_rot, zeros,  0.0)
        imag_v   = _pack(x_2d, zeros,  im_rot, 1.0)
        phasor_v = _pack(x_2d, re_rot, im_rot, 2.0)

        # Axis reference lines (4 floats: x,y,z,component)
        axis_raw = np.array([
            [-1.0, 0.0, 0.0,  3.0],
            [ 1.0, 0.0, 0.0,  3.0],
            [ 0.0,-1.1, 0.0,  3.0],
            [ 0.0, 1.1, 0.0,  3.0],
            [ 0.0, 0.0,-1.1,  3.0],
            [ 0.0, 0.0, 1.1,  3.0],
        ], dtype=np.float32)
        self._ax_cnt = len(axis_raw)

        self._vertices = np.concatenate(
            [axis_raw, real_v, imag_v, phasor_v], axis=0
        )

        ax = self._ax_cnt
        # draw_calls: (GL_mode, first, count, line_width, alpha_boost)
        calls = []
        if self.show_axes:
            calls.append((GL_LINES,  0,            ax,    1, 0.5))
        if self.show_real:
            calls.append((GL_POINTS, ax,            N * n, 0, 1.0))
        if self.show_imag:
            calls.append((GL_POINTS, ax + N * n,    N * n, 0, 1.0))
        if self.show_phasor:
            calls.append((GL_POINTS, ax + 2 * N * n, N * n, 0, 1.0))
        self._draw_calls = calls

    def set_visibility(self, *, axes: bool = None, real: bool = None,
                       imag: bool = None, phasor: bool = None) -> None:
        """Toggle channel visibility; takes effect on the next draw."""
        if axes   is not None: self.show_axes   = axes
        if real   is not None: self.show_real   = real
        if imag   is not None: self.show_imag   = imag
        if phasor is not None: self.show_phasor = phasor
        self._geo_dirty = True

    # -- MVP with perspective modes ------------------------------------------

    def _build_mvp(self, w: int, h: int) -> np.ndarray:
        key = (self.perspective_mode, self._cam_dist, w, h)
        if key == self._mvp_key and self._mvp_val is not None:
            return self._mvp_val
        preset  = _CAMERA_PRESETS.get(self.perspective_mode, _CAMERA_PRESETS["perspective"])
        dist    = self._cam_dist
        raw_eye = np.array(preset["eye"], dtype=np.float64)
        eye_dir = raw_eye / (np.linalg.norm(raw_eye) + 1e-12)
        eye     = eye_dir * dist
        ctr     = np.array(preset["center"], dtype=np.float64)
        up      = np.array(preset["up"],     dtype=np.float64)
        fov     = float(preset["fov"])
        aspect  = w / max(h, 1)
        view    = _look_at(eye, ctr, up)
        if preset["use_ortho"]:
            sc   = dist * 0.55
            proj = _ortho_mat(-sc * aspect, sc * aspect, -sc, sc, 0.01, 50.0)
        else:
            proj = _perspective_mat(fov, aspect, 0.05, 50.0)
        mvp = (proj @ view).astype(np.float32)
        self._mvp_key = key
        self._mvp_val = mvp
        return mvp

    # -- cloud-specific uniforms ---------------------------------------------

    def _set_uniforms(self, prog: int) -> None:
        for name, val in [
            ("uCurrentSlot", self._current_slot),
            ("uNPts",        self._n_pts_each),
            ("uAxCount",     self._ax_cnt),
            ("uDisplayMode", self._display_mode),
        ]:
            loc = self._uloc(name)
            if loc >= 0:
                glUniform1i(loc, int(val))
        for name, val in [
            ("uCloudAlpha", self._cloud_alpha),
            ("uPointSize",  self._point_size),
            ("uAlphaBoost", 1.0),
        ]:
            loc = self._uloc(name)
            if loc >= 0:
                glUniform1f(loc, float(val))


# Backwards-compat alias
GLAnalyticWaveWidget = ComplexPhaseCloud


# ---------------------------------------------------------------------------
# GLPanel - bass_viewer PanelDock-compatible wrapper
# ---------------------------------------------------------------------------

class GLPanel:
    """Wraps ComplexPhaseCloud as a bass_viewer Panel.

    Usage:
        panel = GLPanel(title="Analytic Wave", side="right")
        dock.register("gl_wave", panel)
        dock.set_right("gl_wave")
        # each frame:
        panel.update_data(analytic_pts, current_slot=slot)
    """

    PANEL_W      = 280
    _top_offset: int = 0

    def __init__(self, title: str = "Analytic Wave", side: str = "right",
                 phase_steps: int = 72, perspective_mode: str = "perspective"):
        self.title   = title
        self.side    = side
        self.visible = True
        self.font    = None
        self._panel_scroll_y: int = 0
        self._content_h:      int = 0
        self._wave = ComplexPhaseCloud(
            phase_steps=phase_steps,
            perspective_mode=perspective_mode,
        )

    @property
    def panel_rect(self) -> "pygame.Rect":
        sw = pygame.display.get_surface().get_width()
        sh = pygame.display.get_surface().get_height()
        top = self._top_offset
        if self.side == "left":
            return pygame.Rect(0, top, self.PANEL_W, sh - top)
        return pygame.Rect(sw - self.PANEL_W, top, self.PANEL_W, sh - top)

    def update_data(self, csig_batch: np.ndarray, current_slot: int = 0) -> None:
        self._wave.update_data(csig_batch)
        self._wave.set_current_slot(current_slot)

    def set_perspective_mode(self, mode: str) -> None:
        self._wave.set_perspective_mode(mode)

    def draw_gl(self, win_w: int, win_h: int) -> None:
        """Render the GL wave directly into the current framebuffer."""
        pr = self.panel_rect
        self._wave.draw(
            pr.x, pr.y, pr.w, pr.h,
            win_w, win_h, current_slot=self._wave._current_slot,
        )

    def render_dropdown_overlay(self):
        return None

    def handle_event(self, event) -> bool:
        return False

    def _apply_panel_scroll(self, surf: "pygame.Surface") -> "pygame.Surface":
        return surf

    def _clamp_panel_scroll(self) -> None:
        pass

    def destroy(self) -> None:
        self._wave.destroy()