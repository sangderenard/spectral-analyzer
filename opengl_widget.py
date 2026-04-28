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
        GL_CLAMP_TO_EDGE,
        GL_COLOR_BUFFER_BIT,
        GL_CURRENT_PROGRAM,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_DYNAMIC_DRAW,
        GL_FLOAT,
        GL_FRAGMENT_SHADER,
        GL_LINEAR,
        GL_LINE_SMOOTH,
        GL_LINE_SMOOTH_HINT,
        GL_LINE_STRIP,
        GL_LINES,
        GL_NICEST,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_ONE,
        GL_POINTS,
        GL_PROGRAM_POINT_SIZE,
        GL_R32F,
        GL_RED,
        GL_SCISSOR_TEST,
        GL_SRC_ALPHA,
        GL_TEXTURE0,
        GL_RGBA,
        GL_TEXTURE_2D,
        GL_TEXTURE_3D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_R,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_TRIANGLE_FAN,
        GL_TRIANGLES,
        GL_UNSIGNED_BYTE,
        GL_VERTEX_SHADER,
        GL_VIEWPORT,
        GL_FRONT_AND_BACK,
        GL_LINE,
        GL_FILL,
        glPolygonMode,
        glActiveTexture,
        glBindTexture,
        glDeleteTextures,
        glGenTextures,
        glTexImage2D,
        glTexImage3D,
        glTexParameteri,
        glTexSubImage2D,
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
        glUniform4f,
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
# Volumetric accumulator shader
# ---------------------------------------------------------------------------
# Renders a pre-baked 3-D resonator volume as a ray-marched glow field.
#
# The volume texture is GL_TEXTURE_3D with shape (depth=n_time, height=n_sources,
# width=n_freq) — i.e., each 2D layer is a (source × frequency) spectral frame
# and the Z axis is time.
#
# Screen UV coordinates:
#   UV.x  →  frequency axis  (0 = f_min, 1 = f_max, log-mapped at bake time)
#   UV.y  →  source axis     (0 = str_1 top, 1 = body bottom, equal-height bands)
#
# Ray marches along the Z (time) axis, integrating a Gaussian-weighted window
# around uPlayhead.  Each voxel's emission is coloured by its source layer's
# hue so strings glow their own colour and sympathetics emerge as interference.
#
# Vertex layout: 2 floats pos (NDC) + 2 floats UV — stride = 16 bytes.
# Uniforms:
#   uVolume      — GL_TEXTURE_3D, R32F, shape (n_time × n_sources × n_freq)
#   uPlayhead    — 0..1 current playback position (centre of integration window)
#   uWindowHalf  — half-width of integration window in normalised time (e.g. 0.06)
#   uBrightness  — linear emission multiplier
#   uGamma       — power applied after brightness (< 1 opens shadows)
#   uNSteps      — ray-march step count (32–64 recommended)

# ---------------------------------------------------------------------------
# Cavity pressure field shader
# ---------------------------------------------------------------------------
# Identical vertex shader; fragment uses a thermal colormap instead of
# discrete source-band hues.  Volume layout: (n_time, ny, nx) float32 where
# X/Y are the horizontal cross-section of the cavity and Z is time.

_CAVITY_FRAG_SRC = """
#version 130
uniform sampler3D uVolume;
uniform float uPlayhead;
uniform float uWindowHalf;
uniform float uBrightness;
uniform float uGamma;
uniform int   uNSteps;

in vec2 vUV;

// Acoustic pressure colormap: void → indigo → teal → amber → white-hot
vec3 pressure_rgb(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.00, 0.00, 0.00);
    vec3 c1 = vec3(0.05, 0.02, 0.28);
    vec3 c2 = vec3(0.00, 0.35, 0.75);
    vec3 c3 = vec3(0.05, 0.80, 0.65);
    vec3 c4 = vec3(1.00, 0.68, 0.08);
    vec3 c5 = vec3(1.00, 1.00, 1.00);
    float s = t * 5.0;
    int   i = int(s);
    float f = s - float(i);
    vec3 a, b;
    if      (i == 0) { a = c0; b = c1; }
    else if (i == 1) { a = c1; b = c2; }
    else if (i == 2) { a = c2; b = c3; }
    else if (i == 3) { a = c3; b = c4; }
    else             { a = c4; b = c5; }
    return mix(a, b, clamp(f, 0.0, 1.0));
}

void main() {
    vec3  color_acc = vec3(0.0);
    float alpha_acc = 0.0;

    float t_lo  = uPlayhead - uWindowHalf;
    float t_hi  = uPlayhead + uWindowHalf;
    float inv_n = 1.0 / float(max(uNSteps - 1, 1));

    for (int i = 0; i < uNSteps; i++) {
        if (alpha_acc >= 0.98) break;

        float z = mix(t_lo, t_hi, float(i) * inv_n);
        if (z < 0.0 || z > 1.0) continue;

        // Volume: x=spatial_x, y=spatial_y, z=time
        float density = texture(uVolume, vec3(vUV.x, vUV.y, z)).r;
        if (density < 5e-4) continue;

        float dt   = (z - uPlayhead) / max(uWindowHalf, 1e-4);
        float tw   = exp(-dt * dt * 3.0);
        float emit = pow(clamp(density * uBrightness * tw, 0.0, 1.0), uGamma);

        float step_alpha = clamp(emit * inv_n * 6.0, 0.0, 0.25);
        vec3  col        = pressure_rgb(emit);

        color_acc += col * step_alpha * (1.0 - alpha_acc);
        alpha_acc += step_alpha       * (1.0 - alpha_acc);
    }

    // Faint cavity wall ring: darken pixels with no pressure (outside cylinder)
    // The pressure field already has zeros outside — nothing extra needed.
    gl_FragColor = vec4(color_acc, 1.0);
}
"""

_ACCUM_VERT_SRC = """
#version 130
in vec2 aPos;
in vec2 aUV;
out vec2 vUV;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUV = aUV;
}
"""

_ACCUM_FRAG_SRC = """
#version 130
uniform sampler3D uVolume;
uniform float uPlayhead;
uniform float uWindowHalf;
uniform float uBrightness;
uniform float uGamma;
uniform int   uNSteps;

in vec2 vUV;

// Vivid hue-cycle colour: maps source layer Y to a saturated RGB
vec3 source_rgb(float vy) {
    // vy in [0,1]: map each equal-height band to its string colour
    //   band 0 (str_1)  → hue 0.00  red
    //   band 1 (str_2)  → hue 0.45  cyan-green
    //   band 2 (str_3)  → hue 0.73  blue-violet
    //   band 3+ (body)  → hue 0.13  amber
    float h;
    if      (vy < 0.25) h = 0.00;
    else if (vy < 0.50) h = 0.45;
    else if (vy < 0.75) h = 0.73;
    else                h = 0.13;
    float r = abs(h * 6.0 - 3.0) - 1.0;
    float g = 2.0 - abs(h * 6.0 - 2.0);
    float b = 2.0 - abs(h * 6.0 - 4.0);
    return clamp(vec3(r, g, b), 0.0, 1.0);
}

void main() {
    vec3  accum     = vec3(0.0);
    float alpha_acc = 0.0;

    float t_lo = uPlayhead - uWindowHalf;
    float t_hi = uPlayhead + uWindowHalf;
    float inv_n = 1.0 / float(max(uNSteps - 1, 1));

    for (int i = 0; i < uNSteps; i++) {
        if (alpha_acc >= 0.98) break;

        float z = mix(t_lo, t_hi, float(i) * inv_n);
        if (z < 0.0 || z > 1.0) continue;

        // Sample volume: (x=freq, y=source, z=time)
        float density = texture(uVolume, vec3(vUV.x, vUV.y, z)).r;
        if (density < 5e-4) continue;

        // Temporal Gaussian weight — hottest at playhead
        float dt = (z - uPlayhead) / max(uWindowHalf, 1e-4);
        float tw = exp(-dt * dt * 3.0);

        float emit = pow(clamp(density * uBrightness * tw, 0.0, 1.0), uGamma);

        vec3  col        = source_rgb(vUV.y);
        float step_alpha = clamp(emit * inv_n * 6.0, 0.0, 0.25);

        // Front-to-back alpha compositing
        accum     += col * step_alpha * (1.0 - alpha_acc);
        alpha_acc += step_alpha       * (1.0 - alpha_acc);
    }

    // Soft vertical playhead cursor overlay (maps to time at x=0.5 of each band)
    float t_dist = abs(0.5 - vUV.x);   // cursor drawn at centre-frequency column
    // Instead: a horizontal glowing band at the current playhead time slice
    // Compute which z-slice is closest to uPlayhead in screen-x:
    // (We embed a full-width cursor as a separate narrow horizontal scan.)
    // Simple approach: draw a 1-pixel-wide horizontal band for each source
    // at the frequency column corresponding to uPlayhead mapped to x.
    float cx   = uPlayhead;
    float ddx  = abs(vUV.x - cx);
    float glow = exp(-ddx * ddx * 12000.0) * 0.45;
    accum = mix(accum, vec3(1.0, 0.82, 0.30), glow * (1.0 - alpha_acc));

    gl_FragColor = vec4(accum, 1.0);
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
        self._skip_clear: bool = False

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
# VolumetricAccumulatorWidget — heat-map density display for body resonator
# ---------------------------------------------------------------------------

class VolumetricAccumulatorWidget(GLViewportWidget):
    """True 3-D spectral volume renderer for the body resonator.

    The volume has shape ``(n_time, n_sources, n_freq)`` float32.  Each
    depth-slice is a per-source spectral frame; the shader ray-marches along
    the time axis, compositing contributions weighted by temporal proximity to
    the current playhead.

    Usage::

        w = VolumetricAccumulatorWidget()
        w.update_volume(volume)    # (n_time, n_sources, n_freq) float32
        w.set_playhead(0.42)       # 0..1 normalised position in time
        w.draw(x, y, pw, ph, win_w, win_h)
    """

    _STRIDE: int = 4 * 4   # 2 floats pos + 2 floats UV = 16 bytes

    def __init__(self,
                 gamma: float = 0.45,
                 brightness: float = 3.0,
                 window_half: float = 0.15,
                 n_steps: int = 64,
                 spatial_mode: bool = False,
                 **kw):
        super().__init__(
            vert_src=_ACCUM_VERT_SRC,
            frag_src=_CAVITY_FRAG_SRC if spatial_mode else _ACCUM_FRAG_SRC,
            bg_color=(0.04, 0.02, 0.06, 1.0),
            **kw,
        )
        self._gamma:       float                 = gamma
        self._brightness:  float                 = brightness
        self._window_half: float                 = window_half
        self._n_steps:     int                   = n_steps
        self._volume:      Optional[np.ndarray]  = None   # (n_time, n_sources, n_freq)
        self._vol_tex:     Optional[int]         = None
        self._tex_dirty:   bool                  = False
        self._playhead:    float                  = 0.0

    # -- public API -----------------------------------------------------------

    def update_volume(self, volume: np.ndarray) -> None:
        """Set a (n_time, n_sources, n_freq) float32 spectral volume."""
        self._volume    = np.ascontiguousarray(volume, dtype=np.float32)
        self._tex_dirty = True
        self._geo_dirty = True   # forces quad rebuild on first frame

    def set_playhead(self, t: float) -> None:
        self._playhead = float(t)

    # -- geometry ------------------------------------------------------------

    def _build_geometry(self) -> None:
        verts = np.array([
            [-1.0, -1.0,  0.0, 1.0],   # bottom-left
            [ 1.0, -1.0,  1.0, 1.0],   # bottom-right
            [ 1.0,  1.0,  1.0, 0.0],   # top-right
            [-1.0,  1.0,  0.0, 0.0],   # top-left
        ], dtype=np.float32)
        self._vertices   = verts
        self._draw_calls = [(GL_TRIANGLE_FAN, 0, 4, 0, 1.0)]

    def _setup_vao(self) -> None:
        stride = 4 * 4
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        for name, n_comp, byte_off in [("aPos", 2, 0), ("aUV", 2, 8)]:
            loc = glGetAttribLocation(self._prog, name)
            if loc >= 0:
                glEnableVertexAttribArray(loc)
                glVertexAttribPointer(loc, n_comp, GL_FLOAT, False,
                                      stride, ctypes.c_void_p(byte_off))
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _build_mvp(self, w: int, h: int) -> np.ndarray:
        return np.eye(4, dtype=np.float32)   # identity — fullscreen quad

    # -- uniforms ------------------------------------------------------------

    def _set_uniforms(self, prog: int) -> None:
        for name, val in [
            ("uGamma",      self._gamma),
            ("uBrightness", self._brightness),
            ("uPlayhead",   self._playhead),
            ("uWindowHalf", self._window_half),
        ]:
            loc = self._uloc(name)
            if loc >= 0:
                glUniform1f(loc, float(val))
        loc_steps = self._uloc("uNSteps")
        if loc_steps >= 0:
            glUniform1i(loc_steps, int(self._n_steps))
        # Bind 3-D volume texture to unit 0
        if self._vol_tex is not None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_3D, self._vol_tex)
        loc_tex = self._uloc("uVolume")
        if loc_tex >= 0:
            glUniform1i(loc_tex, 0)

    def _after_draw_calls(self, w: int, h: int) -> None:
        glBindTexture(GL_TEXTURE_3D, 0)

    # -- resource management ------------------------------------------------

    def _ensure_resources(self, w: int, h: int) -> None:
        super()._ensure_resources(w, h)
        if self._tex_dirty and self._volume is not None:
            self._upload_volume_texture()
            self._tex_dirty = False

    def _upload_volume_texture(self) -> None:
        vol = self._volume
        if vol is None:
            return
        n_time, n_sources, n_freq = vol.shape
        if self._vol_tex is None:
            self._vol_tex = int(glGenTextures(1))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self._vol_tex)
        # depth=n_time, height=n_sources, width=n_freq
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
                     n_freq, n_sources, n_time,
                     0, GL_RED, GL_FLOAT, vol.tobytes())
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_3D, 0)

    def destroy(self) -> None:
        if self._vol_tex is not None and _HAS_GL:
            try:
                glDeleteTextures(1, [self._vol_tex])
            except Exception:
                pass
            self._vol_tex = None
        super().destroy()


# ---------------------------------------------------------------------------
# SurfaceBlitter — upload a pygame Surface to GL and draw as a fullscreen quad
# ---------------------------------------------------------------------------
# Enables mixing pygame 2D content with GL shader widgets when the pygame
# display is initialised with ``pygame.OPENGL | pygame.DOUBLEBUF``.
#
# Usage:
#   blitter = SurfaceBlitter()
#   # each frame, before drawing GL widgets:
#   blitter.blit(offscreen_surface, win_w, win_h)

_BLIT_VERT_SRC = """
#version 130
in vec2 aPos;
in vec2 aUV;
out vec2 vUV;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUV = aUV;
}
"""

_BLIT_FRAG_SRC = """
#version 130
uniform sampler2D uTex;
in vec2 vUV;
void main() {
    gl_FragColor = texture(uTex, vUV);
}
"""


class SurfaceBlitter:
    """Upload a pygame Surface as a 2-D GL texture and draw it fullscreen.

    Call :meth:`blit` once per frame before drawing any GL widgets.  The
    surface is uploaded with a vertical flip so pygame's top-left origin
    maps correctly to OpenGL's bottom-left origin.
    """

    _STRIDE: int = 4 * 4   # 2 floats pos + 2 floats UV = 16 bytes

    def __init__(self) -> None:
        self._prog: Optional[int] = None
        self._vao:  Optional[int] = None
        self._vbo:  Optional[int] = None
        self._tex:  Optional[int] = None
        self._tex_w: int = 0
        self._tex_h: int = 0

    def blit(self, surf: Any, win_w: int, win_h: int) -> None:
        """Clear the GL framebuffer, upload *surf*, and draw it fullscreen."""
        if not _HAS_GL or not _HAS_PYGAME:
            return
        self._ensure_resources()

        # Upload the pygame surface (flipped for GL)
        data = pygame.image.tostring(surf, "RGBA", True)
        w, h = surf.get_size()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        if w != self._tex_w or h != self._tex_h:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            self._tex_w, self._tex_h = w, h
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                            GL_RGBA, GL_UNSIGNED_BYTE, data)

        # Clear and draw fullscreen quad
        glViewport(0, 0, win_w, win_h)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self._prog)
        loc = glGetUniformLocation(self._prog, "uTex")
        if loc >= 0:
            glUniform1i(loc, 0)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _ensure_resources(self) -> None:
        if self._prog is None:
            self._prog = GLViewportWidget._compile_programme(
                _BLIT_VERT_SRC, _BLIT_FRAG_SRC
            )
        if self._tex is None:
            self._tex = int(glGenTextures(1))
        if self._vao is None:
            self._vao = glGenVertexArrays(1)
            self._vbo = glGenBuffers(1)
            # NDC fullscreen quad; UV.y=0 at bottom matches pygame's flipped data
            verts = np.array([
                [-1.0, -1.0,  0.0, 0.0],   # bottom-left
                [ 1.0, -1.0,  1.0, 0.0],   # bottom-right
                [ 1.0,  1.0,  1.0, 1.0],   # top-right
                [-1.0,  1.0,  0.0, 1.0],   # top-left
            ], dtype=np.float32)
            glBindVertexArray(self._vao)
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
            stride = self._STRIDE
            for name, n_comp, byte_off in [("aPos", 2, 0), ("aUV", 2, 8)]:
                loc = glGetAttribLocation(self._prog, name)
                if loc >= 0:
                    glEnableVertexAttribArray(loc)
                    glVertexAttribPointer(loc, n_comp, GL_FLOAT, False,
                                          stride, ctypes.c_void_p(byte_off))
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def destroy(self) -> None:
        if not _HAS_GL:
            return
        for h, fn in [
            (self._vbo,  lambda h: glDeleteBuffers(1, [h])),
            (self._vao,  lambda h: glDeleteVertexArrays(1, [h])),
            (self._prog, glDeleteProgram),
        ]:
            if h is not None:
                try:
                    fn(h)
                except Exception:
                    pass
        if self._tex is not None:
            try:
                glDeleteTextures(1, [self._tex])
            except Exception:
                pass
        self._prog = self._vao = self._vbo = self._tex = None
        self._tex_w = self._tex_h = 0


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


# ---------------------------------------------------------------------------
# RayAccumulatorWidget — complex spectral ray-accumulation display
# ---------------------------------------------------------------------------
# Visualises the segment buffer produced by the C ray tracer as a volumetric
# glowing fog that builds up over time as the acoustic wavefront propagates
# through the room.
#
# Rendering model
# ---------------
# The segment buffer (N_seg, 12) is pre-processed on the CPU into a point-
# sprite cloud: each segment is sampled at PTS_PER_SEG equally-spaced
# positions, giving N_seg * PTS_PER_SEG GL_POINTS.  The sprites are
# sorted globally by cumulative path length so that animating the wavefront
# is just a matter of incrementing the draw count.
#
# Fragment shader: Gaussian circular glow within each point sprite.  Additive
# blending (GL_SRC_ALPHA, GL_ONE) accumulates overlapping sprites so that
# regions of ray convergence bloom into bright hotspots while isolated rays
# leave faint coloured trails.
#
# Vertex layout (9 floats × 4 bytes = 36 bytes stride):
#   offset  0: vec3  aPos       — interpolated 3-D world position
#   offset 12: float aSrcId     — source index (drives primary hue)
#   offset 16: float aBounce    — bounce number (drives fade)
#   offset 20: float aBand      — freq band index (drives hue tint)
#   offset 24: float aAmplitude — |A|, post-propagation
#   offset 28: float aPhase     — arg(A) (drives shimmer)
#   offset 32: float aPathFrac  — path_length / max_path_length in [0,1]
#
# Uniforms:
#   uMVP, uNSources, uNBands, uBounceDecay, uPointSize, uAmpScale

_RAY_ACCUM_VERT_SRC = """
#version 130
uniform mat4  uMVP;
uniform float uNSources;
uniform float uNBands;
uniform float uBounceDecay;
uniform float uPointSize;
uniform float uAmpScale;

in vec3  aPos;
in float aSrcId;
in float aBounce;
in float aBand;
in float aAmplitude;
in float aPhase;
in float aPathFrac;

out vec4 vColor;

vec3 hue_rgb(float h) {
    float r = abs(h * 6.0 - 3.0) - 1.0;
    float g = 2.0 - abs(h * 6.0 - 2.0);
    float b = 2.0 - abs(h * 6.0 - 4.0);
    return clamp(vec3(r, g, b), 0.0, 1.0);
}

void main() {
    gl_Position  = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;

    /* Primary hue from source index — each source gets its own colour family. */
    float src_hue   = (aSrcId + 0.5) / max(uNSources, 1.0);

    /* Frequency band adds a subtle hue shift — distinguishes spectral families. */
    float band_hue  = (aBand / max(uNBands, 1.0)) * 0.18;

    /* Complex phase drives a gentle shimmer so the fog breathes with the wave. */
    float shimmer   = sin(aPhase) * 0.04;

    vec3 rgb = hue_rgb(fract(src_hue + band_hue + shimmer));

    /* Deeper bounces fade out exponentially — direct paths are brightest. */
    float bounce_fade = exp(-aBounce * uBounceDecay);

    /* Amplitude modulates overall brightness. */
    float alpha = clamp(aAmplitude * uAmpScale * bounce_fade, 0.0, 1.0);

    vColor = vec4(rgb, alpha);
}
"""

_RAY_ACCUM_FRAG_SRC = """
#version 130
in vec4 vColor;

void main() {
    /* Circular Gaussian glow within the point sprite.
       gl_PointCoord is (0,0)...(1,1); centre is (0.5, 0.5).
       r2 is 0 at centre and 1 at the corners. */
    vec2  p    = gl_PointCoord - vec2(0.5);
    float r2   = dot(p, p) * 4.0;          /* 0 at centre, 1 at edge        */
    float glow = exp(-r2 * 2.8);           /* tight gaussian for crisp core  */

    /* A wider halo ring adds volumetric depth at no extra draw cost. */
    float halo = exp(-r2 * 0.6) * 0.18;

    float alpha = vColor.a * (glow + halo);

    /* Premultiply for additive blending (src_alpha * src_rgb, dest * 1). */
    gl_FragColor = vec4(vColor.rgb * alpha, alpha);
}
"""

_RAY_ACCUM_STRIDE = 9 * 4   # 9 floats × 4 bytes
_RAY_PTS_PER_SEG  = 16      # sample points per segment (fog density)


class RayAccumulatorWidget(GLViewportWidget):
    """Complex spectral ray-accumulation display.

    Feed the float32 segment buffer from _spectral_kernels.RayTracer.trace()
    via update_segments(), then call set_playhead() each frame to animate the
    wavefront.

    Parameters
    ----------
    n_sources : int
        Number of acoustic sources (for colour-family normalisation).
    n_bands : int
        Number of frequency bands (for hue-tint normalisation).
    bounce_decay : float
        Exponential fade multiplier per bounce (1.2–3.0 works well).
    point_size : float
        GL_POINT size in pixels (8–20 gives good fog density).
    amp_scale : float
        Global amplitude → alpha multiplier.
    """

    _STRIDE: int = _RAY_ACCUM_STRIDE

    def __init__(self,
                 n_sources:    int   = 3,
                 n_bands:      int   = 12,
                 bounce_decay: float = 1.6,
                 point_size:   float = 12.0,
                 amp_scale:    float = 3.0,
                 norm_mode:    str   = "percentile",
                 norm_plo:     float = 2.0,
                 norm_phi:     float = 95.0,
                 **kw):
        super().__init__(
            vert_src=_RAY_ACCUM_VERT_SRC,
            frag_src=_RAY_ACCUM_FRAG_SRC,
            bg_color=(0.02, 0.02, 0.04, 1.0),
            **kw)

        self._n_sources    = n_sources
        self._n_bands      = n_bands
        self._bounce_decay = bounce_decay
        self._point_size   = point_size
        self._amp_scale    = amp_scale
        self._norm_mode    = norm_mode   # "raw" | "percentile" | "log" | "rms" | "max"
        self._norm_plo     = norm_plo    # lower percentile clip (percentile mode)
        self._norm_phi     = norm_phi    # upper percentile clip (percentile mode)

        self._playhead:      float = 0.0
        self._total_points:  int   = 0
        self._scene_center:  np.ndarray = np.zeros(3, dtype=np.float32)
        self._scene_radius:  float = 5.0

    # -- public API -----------------------------------------------------------

    def set_norm_mode(self, mode: str,
                      plo: float = 2.0, phi: float = 95.0) -> None:
        """Change amplitude normalization mode and reprocess the last segment buffer."""
        self._norm_mode = mode
        self._norm_plo  = plo
        self._norm_phi  = phi
        if hasattr(self, '_raw_segs') and self._raw_segs is not None:
            self.update_segments(self._raw_segs)

    def _normalize_amp(self, amp: np.ndarray) -> np.ndarray:
        """Map raw amplitude values to [0, 1] according to self._norm_mode."""
        amp = amp.copy().astype(np.float32)
        mode = self._norm_mode

        if mode == "raw":
            # No normalization — caller relies on uAmpScale.
            peak = float(amp.max()) if amp.size else 1.0
            return amp / max(peak, 1e-9)

        if mode == "max":
            peak = float(amp.max()) if amp.size else 1.0
            return amp / max(peak, 1e-9)

        if mode == "rms":
            rms = float(np.sqrt(np.mean(amp ** 2))) if amp.size else 1.0
            return np.clip(amp / max(rms * 3.0, 1e-9), 0.0, 1.0)

        if mode == "log":
            # log(1 + k·amp) normalised so the 95th percentile maps to 0.9.
            k = 1.0
            if amp.size:
                p95 = float(np.percentile(amp, 95))
                if p95 > 1e-9:
                    k = (np.exp(0.9 * np.log(10.0)) - 1.0) / p95
            return np.clip(np.log1p(k * amp) / np.log1p(k * amp.max() + 1e-9),
                           0.0, 1.0)

        # default: percentile — clip to [plo, phi] then rescale.
        lo = float(np.percentile(amp, self._norm_plo)) if amp.size else 0.0
        hi = float(np.percentile(amp, self._norm_phi)) if amp.size else 1.0
        if hi <= lo:
            hi = lo + 1e-9
        return np.clip((amp - lo) / (hi - lo), 0.0, 1.0)

    def update_segments(self, segs: np.ndarray) -> None:
        """Accept (N_seg, 12) float32 segment buffer from the C ray tracer.

        Columns: x0,y0,z0, x1,y1,z1, src_id, bounce, band, amp, phase, path_len
        Sorts by path_length so set_playhead() animates the wavefront.
        """
        if segs is None or len(segs) == 0:
            self._raw_segs      = None
            self._vertices      = np.zeros((0, 9), dtype=np.float32)
            self._total_points  = 0
            self._draw_calls    = []
            return

        segs = np.ascontiguousarray(segs, dtype=np.float32)
        self._raw_segs = segs  # keep for norm_mode changes

        # Sort by cumulative path length (wavefront order).
        order = np.argsort(segs[:, 11], kind='stable')
        segs  = segs[order]

        # Compute scene bounding box for camera auto-fit.
        all_pts = np.vstack([segs[:, :3], segs[:, 3:6]])
        bbox_min = all_pts.min(axis=0)
        bbox_max = all_pts.max(axis=0)
        self._scene_center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
        self._scene_radius = max(0.01, float(np.linalg.norm(bbox_max - bbox_min)) * 0.5)

        max_path = float(segs[:, 11].max()) if len(segs) > 0 else 1.0

        # Normalize amplitude column (index 9) before building vertex buffer.
        segs = segs.copy()
        segs[:, 9] = self._normalize_amp(segs[:, 9])

        # Generate PTS_PER_SEG sample points per segment.
        N_seg  = len(segs)
        t_vals = np.linspace(0.0, 1.0, _RAY_PTS_PER_SEG,
                             dtype=np.float32)                  # (PTS,)

        p0 = segs[:, :3]     # (N_seg, 3)
        p1 = segs[:, 3:6]    # (N_seg, 3)
        # Interpolate: (N_seg, PTS, 3)
        pts = (p0[:, None, :]
               + t_vals[None, :, None] * (p1 - p0)[:, None, :])

        # Metadata (same for every point along a segment): src_id, bounce,
        # band, amp, phase, path_frac  → 6 columns per segment.
        path_frac = (segs[:, 11:12] / max(max_path, 1e-6))    # (N_seg, 1)
        meta = np.concatenate(
            [segs[:, 6:11], path_frac], axis=1,
        ).astype(np.float32)                                    # (N_seg, 6)

        # Broadcast metadata across PTS dimension.
        meta_exp = np.broadcast_to(
            meta[:, None, :],
            (N_seg, _RAY_PTS_PER_SEG, 6),
        )                                                       # (N_seg, PTS, 6)

        # Combine to (N_seg, PTS, 9) then flatten.
        verts = np.concatenate(
            [pts, meta_exp], axis=2,
        ).reshape(-1, 9).astype(np.float32)                    # (N_seg*PTS, 9)

        self._vertices     = verts
        self._total_points = len(verts)
        self._geo_dirty    = True
        self._update_draw_count()

    def set_playhead(self, t: float) -> None:
        """Set animation position (0.0 = no rays, 1.0 = all rays visible)."""
        self._playhead = float(np.clip(t, 0.0, 1.0))
        self._update_draw_count()

    def set_n_sources(self, n: int) -> None:
        self._n_sources = max(1, int(n))

    def set_n_bands(self, n: int) -> None:
        self._n_bands = max(1, int(n))

    # -- internal helpers -----------------------------------------------------

    def _update_draw_count(self) -> None:
        n_vis = int(self._total_points * self._playhead)
        self._draw_calls = [(GL_POINTS, 0, n_vis, 0, 1.0)]

    # -- GLViewportWidget overrides -------------------------------------------

    def _setup_vao(self) -> None:
        stride = _RAY_ACCUM_STRIDE
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        attribs = [
            ("aPos",       3,  0),
            ("aSrcId",     1, 12),
            ("aBounce",    1, 16),
            ("aBand",      1, 20),
            ("aAmplitude", 1, 24),
            ("aPhase",     1, 28),
            ("aPathFrac",  1, 32),
        ]
        for name, n_comp, byte_off in attribs:
            loc = glGetAttribLocation(self._prog, name)
            if loc >= 0:
                glEnableVertexAttribArray(loc)
                glVertexAttribPointer(loc, n_comp, GL_FLOAT, False,
                                     stride, ctypes.c_void_p(byte_off))
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _build_geometry(self) -> None:
        # Geometry is built externally via update_segments(); nothing to do here.
        pass

    def _upload_geometry(self) -> None:
        if self._vertices is None or self._vertices.size == 0:
            return
        data = np.ascontiguousarray(self._vertices, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _set_uniforms(self, prog: int) -> None:
        glUniform1f(self._uloc("uNSources"),    float(self._n_sources))
        glUniform1f(self._uloc("uNBands"),      float(self._n_bands))
        glUniform1f(self._uloc("uBounceDecay"), self._bounce_decay)
        glUniform1f(self._uloc("uPointSize"),   self._point_size)
        glUniform1f(self._uloc("uAmpScale"),    self._amp_scale)

    def _build_mvp(self, w: int, h: int) -> np.ndarray:
        """Autoscale: compute camera distance so bounding sphere fills the viewport."""
        c      = self._scene_center.astype(np.float64)
        r      = float(self._scene_radius)
        off    = np.array([0.55, 0.65, 1.0], dtype=np.float64)
        off   /= np.linalg.norm(off)
        aspect  = w / max(h, 1)
        fov_y   = 50.0
        hfov_y  = math.radians(fov_y * 0.5)
        hfov_x  = math.atan(math.tan(hfov_y) * aspect)
        # Place camera so bounding sphere fills 88% of the smaller viewport dimension
        fill    = 0.88
        d       = r / (fill * math.tan(min(hfov_y, hfov_x)))
        eye     = c + off * d
        view    = _look_at(eye, c, np.array([0.0, 1.0, 0.0]))
        r_s     = max(r, 0.001)
        proj    = _perspective_mat(fov_y, aspect, r_s * 0.01, r_s * 30.0)
        return (proj @ view).astype(np.float32)

    def _render_scene(self, x: int, y: int, w: int, h: int,
                      win_w: int, win_h: int) -> None:
        """Override to use additive blending and disable depth writes for fog."""
        if not _HAS_GL:
            return

        gl_x = int(x)
        gl_y = int(win_h - y - h)
        glViewport(gl_x, gl_y, w, h)
        glScissor(gl_x, gl_y, w, h)
        glEnable(GL_SCISSOR_TEST)

        if not self._skip_clear:
            r, g, b, a = self.bg_color
            glClearColor(r, g, b, a)
            glClearDepth(1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Disable depth testing so every sprite layer accumulates.
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        # Additive: bright fog at ray-convergence hotspots.
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        try:
            glEnable(GL_PROGRAM_POINT_SIZE)
        except Exception:
            pass

        if self._total_points == 0 or not self._draw_calls:
            glDisable(GL_SCISSOR_TEST)
            glDisable(GL_BLEND)
            return

        self._ensure_resources(w, h)

        glUseProgram(self._prog)
        mvp = self._build_mvp(w, h)
        loc_mvp = self._uloc("uMVP")
        if loc_mvp >= 0:
            glUniformMatrix4fv(loc_mvp, 1, True, mvp)

        self._set_uniforms(self._prog)

        glBindVertexArray(self._vao)
        for entry in self._draw_calls:
            mode, first, count, lw, _ = entry
            if count <= 0:
                continue
            glDrawArrays(mode, first, count)
        glBindVertexArray(0)
        glUseProgram(0)

        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)   # restore for host GL context


# ---------------------------------------------------------------------------
# SurfaceIllumWidget — per-triangle acoustic illumination heatmap
# ---------------------------------------------------------------------------
# Renders a triangulated room geometry as a colored mesh where each triangle's
# colour encodes the cumulative acoustic energy deposited by ray hit endpoints.
# A thin grey wireframe pass follows the filled pass so panel edges are visible.
#
# Vertex layout (4 floats × 4 bytes = 16 bytes stride):
#   offset  0: vec3  aPos        — triangle vertex world position
#   offset 12: float aIntensity  — normalised energy [0, 1]

_SURF_ILLUM_VERT_SRC = """
#version 130
uniform mat4 uMVP;
in vec3  aPos;
in float aIntensity;
out float vIntensity;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vIntensity  = aIntensity;
}
"""

_SURF_ILLUM_FRAG_SRC = """
#version 130
uniform int  uWireframe;
uniform vec4 uWireColor;
in float vIntensity;

vec3 thermal(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.00, 0.00, 0.00);
    vec3 c1 = vec3(0.05, 0.02, 0.28);
    vec3 c2 = vec3(0.00, 0.35, 0.75);
    vec3 c3 = vec3(0.05, 0.80, 0.65);
    vec3 c4 = vec3(1.00, 0.68, 0.08);
    vec3 c5 = vec3(1.00, 1.00, 1.00);
    float s = t * 5.0;
    int   i = int(s);
    float f = s - float(i);
    vec3 a, b;
    if      (i == 0) { a = c0; b = c1; }
    else if (i == 1) { a = c1; b = c2; }
    else if (i == 2) { a = c2; b = c3; }
    else if (i == 3) { a = c3; b = c4; }
    else             { a = c4; b = c5; }
    return mix(a, b, clamp(f, 0.0, 1.0));
}

void main() {
    if (uWireframe != 0) {
        gl_FragColor = uWireColor;
    } else {
        vec3 col = thermal(vIntensity);
        gl_FragColor = vec4(col, 0.85);
    }
}
"""


class SurfaceIllumWidget(GLViewportWidget):
    """Per-triangle acoustic surface illumination map.

    Feed triangulated geometry + per-triangle amplitude with update_geometry().
    Renders a thermal-colourmap mesh: dark-blue = silent, white-hot = loudest.
    A thin grey wireframe overlay shows panel boundaries.

    Usage::

        w = SurfaceIllumWidget()
        w.update_geometry(verts_flat, normals, illum)   # (N_tri,9), (N_tri,3), (N_tri,)
        w.draw(x, y, pw, ph, win_w, win_h)
    """

    _STRIDE = 4 * 4   # vec3 pos + float intensity = 16 bytes

    def __init__(self, wireframe_only: bool = False,
                 wire_color: tuple = (0.55, 0.60, 0.65, 0.55), **kw):
        super().__init__(
            vert_src=_SURF_ILLUM_VERT_SRC,
            frag_src=_SURF_ILLUM_FRAG_SRC,
            bg_color=(0.02, 0.02, 0.04, 1.0),
            **kw,
        )
        self._wireframe_only = wireframe_only
        self._wire_color     = wire_color
        self._tri_verts:    Optional[np.ndarray] = None  # (N_tri, 9)  float32
        self._tri_illum:    Optional[np.ndarray] = None  # (N_tri,)    float32
        self._scene_center: np.ndarray = np.zeros(3, dtype=np.float32)
        self._scene_radius: float      = 5.0

    # -- public API -----------------------------------------------------------

    def update_geometry(self, verts_flat: np.ndarray,
                        normals: np.ndarray,
                        illum:   np.ndarray) -> None:
        """Set geometry and per-triangle illumination.

        Parameters
        ----------
        verts_flat : (N_tri, 9) float32 — triangle vertices (3 verts × 3 coords)
        normals    : (N_tri, 3) float32 — outward unit normals (not used for rendering)
        illum      : (N_tri,)   float32 — normalised energy [0, 1]
        """
        self._tri_verts = np.ascontiguousarray(verts_flat, dtype=np.float32)
        self._tri_illum = np.ascontiguousarray(illum,      dtype=np.float32)
        # Bounding sphere for camera auto-fit.
        pts = self._tri_verts.reshape(-1, 3)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        self._scene_center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
        self._scene_radius = max(0.01, float(np.linalg.norm(bbox_max - bbox_min)) * 0.5)
        self._geo_dirty = True

    # -- geometry builder ----------------------------------------------------

    def _build_geometry(self) -> None:
        verts = self._tri_verts
        illum = self._tri_illum
        if verts is None or illum is None or len(verts) == 0:
            self._vertices   = np.zeros((0, 4), dtype=np.float32)
            self._draw_calls = []
            return
        N_tri = len(verts)
        pts3    = verts.reshape(N_tri, 3, 3).reshape(-1, 3)            # (N*3, 3)
        illum3  = np.broadcast_to(illum[:, None], (N_tri, 3)).reshape(-1, 1)  # (N*3, 1)
        self._vertices   = np.concatenate([pts3, illum3], axis=1).astype(np.float32)
        self._draw_calls = [(GL_TRIANGLES, 0, N_tri * 3, 0, 1.0)]

    def _setup_vao(self) -> None:
        stride = self._STRIDE
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        for name, n_comp, byte_off in [("aPos", 3, 0), ("aIntensity", 1, 12)]:
            loc = glGetAttribLocation(self._prog, name)
            if loc >= 0:
                glEnableVertexAttribArray(loc)
                glVertexAttribPointer(loc, n_comp, GL_FLOAT, False,
                                      stride, ctypes.c_void_p(byte_off))
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _build_mvp(self, w: int, h: int) -> np.ndarray:
        c   = self._scene_center.astype(np.float64)
        r   = float(self._scene_radius)
        off = np.array([0.55, 0.65, 1.0], dtype=np.float64)
        off /= np.linalg.norm(off)
        aspect    = w / max(h, 1)
        fov_y_deg = 50.0
        hfov_y    = math.radians(fov_y_deg * 0.5)
        tan_hfov_y = math.tan(hfov_y)
        tan_hfov_x = tan_hfov_y * aspect

        # True autoscale: project actual vertices with a unit-distance view matrix,
        # then compute the exact camera distance needed to fill the viewport.
        d = r * 2.2   # fallback
        pts = self._tri_verts
        if pts is not None and len(pts) > 0:
            p = pts.reshape(-1, 3).astype(np.float64)
            # View matrix with camera at c + off (distance 1 from centre)
            view0 = _look_at(c + off, c, np.array([0.0, 1.0, 0.0]))
            ones  = np.ones((len(p), 1))
            vp    = (view0 @ np.concatenate([p, ones], axis=1).T).T
            z_fwd = -vp[:, 2]          # positive = in front of camera
            mask  = z_fwd > 1e-6
            if mask.any():
                mx = float(np.max(np.abs(vp[mask, 0]) / z_fwd[mask]))
                my = float(np.max(np.abs(vp[mask, 1]) / z_fwd[mask]))
                fill = 0.88
                # d = scale factor so geometry fills `fill` of viewport in each axis
                d = max(my / (tan_hfov_y * fill),
                        mx / (tan_hfov_x * fill),
                        r * 0.1)

        eye  = c + off * d
        view = _look_at(eye, c, np.array([0.0, 1.0, 0.0]))
        r_s  = max(r, 0.001)
        proj = _perspective_mat(fov_y_deg, aspect, r_s * 0.01, r_s * 30.0)
        return (proj @ view).astype(np.float32)

    # -- render override (filled + wireframe) --------------------------------

    def _render_scene(self, x: int, y: int, w: int, h: int,
                      win_w: int, win_h: int) -> None:
        if not _HAS_GL or self._tri_verts is None:
            return

        gl_x = int(x)
        gl_y = int(win_h - y - h)
        glViewport(gl_x, gl_y, w, h)
        glScissor(gl_x, gl_y, w, h)
        glEnable(GL_SCISSOR_TEST)

        if not self._skip_clear:
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

        self._ensure_resources(w, h)
        if self._vertices is None or len(self._vertices) == 0:
            glDisable(GL_SCISSOR_TEST)
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)
            return

        glUseProgram(self._prog)
        mvp = self._build_mvp(w, h)
        loc_mvp  = self._uloc("uMVP")
        loc_wire = self._uloc("uWireframe")
        if loc_mvp >= 0:
            glUniformMatrix4fv(loc_mvp, 1, True, mvp)
        loc_wc = self._uloc("uWireColor")
        if loc_wc >= 0:
            wr, wg, wb, wa = self._wire_color
            glUniform4f(loc_wc, wr, wg, wb, wa)

        n_v = len(self._vertices)
        glBindVertexArray(self._vao)

        # Pass 1: filled thermal triangles (skipped in wireframe_only mode).
        if not self._wireframe_only:
            if loc_wire >= 0:
                glUniform1i(loc_wire, 0)
            glDrawArrays(GL_TRIANGLES, 0, n_v)

        # Pass 2: wireframe overlay.
        if loc_wire >= 0:
            glUniform1i(loc_wire, 1)
        try:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(1.0)
            glDrawArrays(GL_TRIANGLES, 0, n_v)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        except Exception:
            pass

        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)


# ---------------------------------------------------------------------------
# CompositeBodyWidget — instrument body wireframe + acoustic ray fog overlay
# ---------------------------------------------------------------------------
# Renders both visualization layers in a single GL viewport sharing one camera:
#
#   Pass 1  SurfaceIllumWidget (wireframe_only=True)
#           Draws the triangulated instrument geometry as thin near-white lines
#           on a dark background, using a perspective camera fitted to the
#           geometry bounding sphere.
#
#   Pass 2  RayAccumulatorWidget (skip_clear=True, same camera)
#           Blends the acoustic ray-fog sprite cloud on top with additive
#           transparency so ray-convergence hotspots bloom into bright spots
#           while the geometry wireframe remains visible underneath.
#
# The camera is always derived from the geometry bounds.  When update_segments()
# is called, the ray widget's scene_center / scene_radius are overridden with
# the geometry values so both passes share an identical MVP.

class CompositeBodyWidget:
    """Instrument body wireframe + acoustic ray fog in a single shared perspective.

    Usage::

        w = CompositeBodyWidget()
        w.update_geometry(verts_flat, normals, illum)   # (N_tri,9), (N_tri,3), (N_tri,)
        w.update_segments(ray_segs)                     # (N_seg, 12) float32
        w.set_playhead(0.42)                            # 0..1 normalised
        w.draw(x, y, pw, ph, win_w, win_h)
    """

    def __init__(self,
                 n_sources:    int   = 3,
                 n_bands:      int   = 12,
                 bounce_decay: float = 1.6,
                 point_size:   float = 12.0,
                 amp_scale:    float = 3.0,
                 norm_mode:    str   = "percentile",
                 norm_plo:     float = 2.0,
                 norm_phi:     float = 95.0):
        # Wireframe layer: instrument geometry with surface illumination + wire overlay.
        self._geo = SurfaceIllumWidget(
            wireframe_only=False,
            wire_color=(0.82, 0.86, 0.92, 0.40),
        )
        # Ray-fog layer: acoustic ray accumulation (additive, same camera).
        self._ray = RayAccumulatorWidget(
            n_sources=n_sources, n_bands=n_bands,
            bounce_decay=bounce_decay, point_size=point_size,
            amp_scale=amp_scale, norm_mode=norm_mode,
            norm_plo=norm_plo, norm_phi=norm_phi,
        )
        self._has_geo = False

    # -- public API -----------------------------------------------------------

    def update_geometry(self, verts_flat: np.ndarray,
                        normals:    np.ndarray,
                        illum:      np.ndarray) -> None:
        """Feed geometry to the wireframe layer; also locks the shared camera."""
        self._geo.update_geometry(verts_flat, normals, illum)
        self._has_geo = True
        self._sync_camera()

    def update_segments(self, segs: np.ndarray) -> None:
        """Feed ray segments; camera is overridden from geometry bounds."""
        self._ray.update_segments(segs)
        # update_segments() overwrites _scene_center/_scene_radius from ray data;
        # restore the geometry-derived camera so both passes align.
        if self._has_geo:
            self._sync_camera()

    def set_playhead(self, t: float) -> None:
        self._ray.set_playhead(t)

    def set_norm_mode(self, mode: str,
                      plo: float = 2.0, phi: float = 95.0) -> None:
        self._ray.set_norm_mode(mode, plo, phi)

    @property
    def _norm_mode(self) -> str:
        return self._ray._norm_mode  # type: ignore[attr-defined]

    def draw(self, x: int, y: int, w: int, h: int,
             win_w: int, win_h: int) -> None:
        if not _HAS_GL:
            return
        # Pass 1: geometry wireframe with clear (sets up dark background).
        self._geo._skip_clear = False
        self._geo.draw(x, y, w, h, win_w, win_h)
        # Pass 2: ray fog — no clear, additive blend, same camera.
        if self._has_geo:
            self._sync_camera()
        self._ray._skip_clear = True
        self._ray.draw(x, y, w, h, win_w, win_h)
        self._ray._skip_clear = False

    def destroy(self) -> None:
        self._geo.destroy()
        self._ray.destroy()

    # -- internal helpers -----------------------------------------------------

    def _sync_camera(self) -> None:
        """Copy geometry bounding sphere to ray widget; invalidate MVP cache."""
        self._ray._scene_center = self._geo._scene_center.copy()
        self._ray._scene_radius = self._geo._scene_radius
        self._ray._mvp_key = None   # force MVP recompute on next draw