"""
doc_renderer.py
===============

Python wrapper around the ``_spectral_kernels.DocRenderer`` pybind11 class.

The renderer maintains a document hierarchy of *nodes* (KnobSpec instances,
Panel trees, static labels, primitive shapes) in a thread-safe FIFO.  A
background C worker renders each node into its own RGBA8 tile; the main
thread composites all live tiles into a flat RGBA texture and uploads it to
GL for a single fullscreen-quad blit at the end of the frame.

Typical usage
-------------
>>> from controls import get_shader_walker, KnobSpec, Panel
>>> rdr = DocRenderer(1920, 1080)
>>> rdr.init_gl()
>>> # build your panel
>>> panel = Panel(name="osc", label="Oscillator", knobs=[...])
>>> rdr.submit_panel(panel, rect=(10, 10, 260, 400))
>>> rdr.register()   # installs into ShaderFrameWalker

Each frame the walker calls ``_run()``, which:
  1. Calls ``dr_flush()`` to drain the FIFO.
  2. Calls ``dr_composite()`` to blend tiles into the output buffer.
  3. Uploads the RGBA buffer to a GL texture.
  4. Draws a fullscreen quad using the doc_composite shader pair.

Glyph atlas
-----------
Call ``load_glyph_atlas_from_pil(font, glyph_w, glyph_h)`` after init_gl()
to populate the atlas from a PIL/Pillow ImageFont object.  Falls back to no
glyphs (layout-only) if PIL is unavailable.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# -- pybind11 backend ---------------------------------------------------------
import _spectral_kernels as _sk

# -- Optional OpenGL ----------------------------------------------------------
try:
    from OpenGL import GL
    from OpenGL.GL import (
        glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
        glTexSubImage2D,
        glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
        glGetShaderInfoLog, glCreateProgram, glAttachShader, glLinkProgram,
        glGetProgramiv, glGetProgramInfoLog, glUseProgram, glDeleteShader,
        glGenVertexArrays, glBindVertexArray,
        glGetUniformLocation, glUniform1i, glUniform1f,
        glEnable, glDisable, glBlendFuncSeparate,
        glBlendEquation, glDrawArrays,
        GL_TEXTURE_2D, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
        GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
        GL_COMPILE_STATUS, GL_LINK_STATUS,
        GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
        GL_FUNC_ADD, GL_TRIANGLES,
    )
    _GL_OK = True
except ImportError:
    _GL_OK = False

# -- Shader paths -------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_VERT_PATH = os.path.join(_HERE, "csrc", "shaders", "doc_composite.vert.glsl")
_FRAG_PATH = os.path.join(_HERE, "csrc", "shaders", "doc_composite.frag.glsl")

# -- DocNodeType enum values (must match doc_renderer.h) ---------------------
DR_NODE_KNOB_FLOAT     = 0
DR_NODE_KNOB_INT       = 1
DR_NODE_KNOB_ENUM      = 2
DR_NODE_KNOB_BOOL      = 3
DR_NODE_PANEL_HEADER   = 4
DR_NODE_PANEL_BODY     = 5
DR_NODE_TEXT_LABEL     = 6
DR_NODE_PRIM_RECT      = 7
DR_NODE_PRIM_ROUNDRECT = 8
DR_NODE_PRIM_ICON      = 9

# -- Theme defaults -----------------------------------------------------------
_THEME = {
    "bg":        (0.10, 0.10, 0.12, 0.88),
    "fg":        (0.90, 0.90, 0.90, 1.00),
    "border":    (0.30, 0.30, 0.35, 1.00),
    "accent":    (0.25, 0.55, 0.90, 1.00),
    "header_bg": (0.15, 0.15, 0.20, 0.95),
}

_ZERO4 = (0.0, 0.0, 0.0, 0.0)


def _rgba_arr(rgba: tuple) -> np.ndarray:
    r = rgba or _ZERO4
    return np.array([r[0], r[1], r[2], r[3]], dtype=np.float32)


def _knob_value_norm(knob: Any, current_value: Any) -> float:
    try:
        lo, hi = float(knob.low), float(knob.high)
        if hi == lo:
            return 0.0
        v = float(current_value) if current_value is not None else float(knob.default or 0)
        if getattr(knob, "is_log", False) and lo > 0.0 and hi > 0.0:
            import math
            v  = math.log(max(v, lo))
            lo = math.log(lo)
            hi = math.log(hi)
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))
    except Exception:
        return 0.0


# -- Main class ---------------------------------------------------------------

class DocRenderer:
    """
    Document-hierarchy texture renderer backed by ``_spectral_kernels.DocRenderer``.

    Parameters
    ----------
    width, height : int
        Output texture size in pixels.
    owner_id : str
        ShaderFrameWalker owner name used when registering the walker node.
    """

    def __init__(self, width: int, height: int, owner_id: str = "doc_hierarchy"):
        self._width    = width
        self._height   = height
        self._owner_id = owner_id

        self._backend: _sk.DocRenderer = _sk.DocRenderer(width, height)

        # GL state
        self._tex_id : int | None = None
        self._prog   : int | None = None
        self._vao    : int | None = None
        self._u_atlas: int = -1
        self._u_alpha: int = -1

        self._next_id = 1

    # -- ID allocation --------------------------------------------------------

    def _alloc_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    # -- Atlas loading --------------------------------------------------------

    def load_glyph_atlas_rgba(self, rgba: np.ndarray,
                               glyph_w: int, glyph_h: int) -> None:
        arr = np.ascontiguousarray(rgba, dtype=np.uint8)
        self._backend.load_glyph_atlas(arr, glyph_w, glyph_h)

    def load_glyph_atlas_from_pil(self, font: Any,
                                   glyph_w: int = 8,
                                   glyph_h: int = 12) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return

        cols    = 16
        rows    = 6
        atlas_w = cols * glyph_w
        atlas_h = rows * glyph_h
        img     = Image.new("RGBA", (atlas_w, atlas_h), (0, 0, 0, 0))
        draw    = ImageDraw.Draw(img)
        for idx in range(96):
            col = idx % cols
            row = idx // cols
            draw.text((col * glyph_w, row * glyph_h), chr(idx + 32),
                      font=font, fill=(255, 255, 255, 255))
        self.load_glyph_atlas_rgba(np.array(img, dtype=np.uint8), glyph_w, glyph_h)

    def load_primitive_atlas_rgba(self, rgba: np.ndarray,
                                   prim_w: int, prim_h: int,
                                   prim_cols: int = 8) -> None:
        arr = np.ascontiguousarray(rgba, dtype=np.uint8)
        self._backend.load_primitive_atlas(arr, prim_w, prim_h, prim_cols)

    # -- Node submission ------------------------------------------------------

    def submit_raw(self, node_id: int, rect: tuple,
                   node_type: int,
                   label: str = "",
                   value_str: str = "",
                   bg: tuple = None,
                   fg: tuple = None,
                   border: tuple = None,
                   accent: tuple = None,
                   corner_radius: float = 2.0,
                   font_scale: float = 1.0,
                   value_norm: float = 0.0,
                   icon_id: int = -1,
                   border_px: int = 1) -> None:
        self._backend.submit_node(
            node_id,
            rect[0], rect[1], rect[2], rect[3],
            node_type,
            label,
            value_str,
            _rgba_arr(bg     or _THEME["bg"]),
            _rgba_arr(fg     or _THEME["fg"]),
            _rgba_arr(border or _THEME["border"]),
            _rgba_arr(accent or _THEME["accent"]),
            corner_radius,
            font_scale,
            value_norm,
            icon_id,
            border_px,
        )

    def submit_knobspec(self, knob: Any, rect: tuple,
                        current_value: Any = None,
                        node_id: int | None = None) -> int:
        if node_id is None:
            node_id = self._alloc_id()

        dtype   = getattr(knob, "dtype",   "float")
        choices = getattr(knob, "choices", None)
        if choices:
            ntype = DR_NODE_KNOB_ENUM
        elif dtype == "bool":
            ntype = DR_NODE_KNOB_BOOL
        elif dtype == "int":
            ntype = DR_NODE_KNOB_INT
        else:
            ntype = DR_NODE_KNOB_FLOAT

        label = getattr(knob, "label", None) or getattr(knob, "name", "?")
        unit  = getattr(knob, "unit",  "")
        fmt   = getattr(knob, "fmt",   ".3g")

        if current_value is None:
            current_value = getattr(knob, "default", None)

        if choices:
            try:
                idx     = int(current_value or 0)
                val_str = str(choices[idx]) if idx < len(choices) else ""
            except Exception:
                val_str = str(current_value or "")
        elif dtype == "bool":
            val_str = "ON" if (current_value or 0) else "OFF"
        else:
            try:
                val_str = format(float(current_value or 0), fmt)
                if unit:
                    val_str += " " + unit
            except Exception:
                val_str = str(current_value or "")

        self.submit_raw(node_id, rect, ntype,
                        label=label, value_str=val_str,
                        value_norm=_knob_value_norm(knob, current_value))
        return node_id

    def submit_panel(self, panel: Any, rect: tuple,
                     node_id_map: dict | None = None,
                     knob_values: dict | None = None) -> dict:
        if node_id_map is None:
            node_id_map = {}
        if knob_values is None:
            knob_values = {}

        x, y, w, h = rect

        body_key = f"__body__{panel.name}"
        if body_key not in node_id_map:
            node_id_map[body_key] = self._alloc_id()
        self.submit_raw(node_id_map[body_key], (x, y, w, h),
                        DR_NODE_PANEL_BODY, bg=_THEME["bg"], border_px=1)

        HDR_H   = 20
        hdr_key = f"__hdr__{panel.name}"
        if hdr_key not in node_id_map:
            node_id_map[hdr_key] = self._alloc_id()
        lbl = getattr(panel, "label", None) or panel.name
        self.submit_raw(node_id_map[hdr_key], (x, y, w, HDR_H),
                        DR_NODE_PANEL_HEADER, label=lbl,
                        bg=_THEME["header_bg"], border_px=0)

        cursor_y = y + HDR_H + 2
        KNOB_H   = 40
        PAD      = 2

        for knob in (getattr(panel, "knobs", []) or []):
            kname = getattr(knob, "name", str(id(knob)))
            if kname not in node_id_map:
                node_id_map[kname] = self._alloc_id()
            self.submit_knobspec(
                knob,
                (x + PAD, cursor_y, w - PAD * 2, KNOB_H),
                current_value=knob_values.get(kname),
                node_id=node_id_map[kname],
            )
            cursor_y += KNOB_H + PAD

        for sub in (getattr(panel, "panels", []) or []):
            sub_h    = max(60, h - (cursor_y - y) - PAD)
            sub_rect = (x + PAD, cursor_y, w - PAD * 2, sub_h)
            self.submit_panel(sub, sub_rect, node_id_map, knob_values)
            cursor_y += sub_h + PAD

        return node_id_map

    def remove_node(self, node_id: int) -> None:
        self._backend.remove_node(node_id)

    def mark_dirty(self, node_id: int) -> None:
        self._backend.mark_dirty(node_id)

    def clear(self) -> None:
        self._backend.clear()

    # -- GL initialisation ----------------------------------------------------

    def init_gl(self) -> None:
        if not _GL_OK:
            return

        def _compile(src: str, kind: int) -> int:
            sh = glCreateShader(kind)
            glShaderSource(sh, src)
            glCompileShader(sh)
            if not glGetShaderiv(sh, GL_COMPILE_STATUS):
                raise RuntimeError(
                    "doc_composite shader compile error:\n"
                    + glGetShaderInfoLog(sh).decode())
            return sh

        with open(_VERT_PATH) as f:
            vert_src = f.read()
        with open(_FRAG_PATH) as f:
            frag_src = f.read()

        vs   = _compile(vert_src, GL_VERTEX_SHADER)
        fs   = _compile(frag_src, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vs)
        glAttachShader(prog, fs)
        glLinkProgram(prog)
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            raise RuntimeError(
                "doc_composite link error:\n"
                + glGetProgramInfoLog(prog).decode())
        glDeleteShader(vs)
        glDeleteShader(fs)

        self._prog    = prog
        self._u_atlas = glGetUniformLocation(prog, "uDocAtlas")
        self._u_alpha = glGetUniformLocation(prog, "uAlpha")

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     self._width, self._height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)
        self._tex_id = tex

        self._vao = glGenVertexArrays(1)

    # -- Per-frame run --------------------------------------------------------

    def _run(self, targets=None, frame_index: int = 0, **_kw) -> None:
        if not _GL_OK or self._prog is None:
            return

        self._backend.flush()

        if self._backend.composite_dirty:
            rgba = self._backend.composite()   # (H, W, 4) uint8
            glBindTexture(GL_TEXTURE_2D, self._tex_id)
            glTexSubImage2D(
                GL_TEXTURE_2D, 0,
                0, 0, self._width, self._height,
                GL_RGBA, GL_UNSIGNED_BYTE,
                rgba,
            )
            glBindTexture(GL_TEXTURE_2D, 0)

        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE,       GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(self._prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glUniform1i(self._u_atlas, 0)
        glUniform1f(self._u_alpha, 1.0)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

        glUseProgram(0)
        glDisable(GL_BLEND)

    # -- External RGBA blit (used by the global dispatcher) -------------------

    def blit_rgba(self, rgba, *, alpha: float = 1.0) -> None:
        """Upload an arbitrary (H, W, 4) uint8 RGBA buffer through the
        same fullscreen-quad pipeline used for the doc composite.

        The 2D-C and 3D-C globals produce CPU RGBA frames; this method
        is the blit landing site that deposits them into the active GL
        framebuffer per the unified resolve policy.
        """
        if not _GL_OK or self._prog is None or rgba is None:
            return
        # Tolerate (H, W, 4) shape with arbitrary dimensions; if it does
        # not match our pre-allocated texture, do a full glTexImage2D.
        h = int(rgba.shape[0]); w = int(rgba.shape[1])
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        if w == self._width and h == self._height:
            glTexSubImage2D(
                GL_TEXTURE_2D, 0,
                0, 0, w, h,
                GL_RGBA, GL_UNSIGNED_BYTE,
                rgba,
            )
        else:
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA8,
                w, h, 0,
                GL_RGBA, GL_UNSIGNED_BYTE,
                rgba,
            )
            self._width  = w
            self._height = h
        glBindTexture(GL_TEXTURE_2D, 0)

        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE,       GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(self._prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glUniform1i(self._u_atlas, 0)
        glUniform1f(self._u_alpha, float(alpha))

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

        glUseProgram(0)
        glDisable(GL_BLEND)

    # -- ShaderFrameWalker registration ---------------------------------------

    def register(self) -> None:
        """No-op. The doc renderer is a *global default* 2D-channel
        composer; it is invoked on whatever the registered shaders did
        not finalise (the leftover mask) by the unified resolve.
        """
        return

    # -- Introspection --------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self._backend.node_count

    @property
    def queue_depth(self) -> int:
        return self._backend.queue_depth

    # -- Cleanup --------------------------------------------------------------

    def destroy(self) -> None:
        pass   # _sk.DocRenderer.__del__ calls dr_destroy automatically

    def __del__(self) -> None:
        self.destroy()
