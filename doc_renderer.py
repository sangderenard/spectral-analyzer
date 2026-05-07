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
import math
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
        glBlendEquation, glDrawArrays, glDepthMask, glIsEnabled, glGetBooleanv,
        GL_TEXTURE_2D, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
        GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
        GL_COMPILE_STATUS, GL_LINK_STATUS,
        GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
        GL_FUNC_ADD, GL_TRIANGLES, GL_DEPTH_TEST, GL_DEPTH_WRITEMASK,
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
DR_NODE_PRIM_QUAD_POLAR = 10

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
        self._polar_floor_sig: tuple | None = None

    @staticmethod
    def _begin_overlay_blit() -> tuple[bool, bool]:
        depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
        depth_write_was_enabled = bool(glGetBooleanv(GL_DEPTH_WRITEMASK))
        glDisable(GL_DEPTH_TEST)
        glDepthMask(False)
        return depth_was_enabled, depth_write_was_enabled

    @staticmethod
    def _end_overlay_blit(state: tuple[bool, bool]) -> None:
        depth_was_enabled, depth_write_was_enabled = state
        glDepthMask(bool(depth_write_was_enabled))
        if depth_was_enabled:
            glEnable(GL_DEPTH_TEST)
        else:
            glDisable(GL_DEPTH_TEST)

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
                   border_px: int = 1,
                   parent_id: int = 0,
                   sibling_order: int = -1,
                   rotation_angle: float = 0.0,
                   polar_cx: float = 0.0,
                   polar_cy: float = 0.0,
                   polar_r0: float = 0.0,
                   polar_a0: float = 0.0,
                   polar_r1: float = 0.0,
                   polar_a1: float = 0.0,
                   polar_r2: float = 0.0,
                   polar_a2: float = 0.0,
                   polar_r3: float = 0.0,
                   polar_a3: float = 0.0) -> None:
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
            int(parent_id),
            int(sibling_order),
            rotation_angle,
            polar_cx,
            polar_cy,
            polar_r0,
            polar_a0,
            polar_r1,
            polar_a1,
            polar_r2,
            polar_a2,
            polar_r3,
            polar_a3,
        )

    def submit_knobspec(self, knob: Any, rect: tuple,
                        current_value: Any = None,
                        node_id: int | None = None,
                        parent_id: int = 0,
                        sibling_order: int = -1) -> int:
        if node_id is None:
            node_id = self._alloc_id()

        dtype   = getattr(knob, "dtype",   "float")
        choices = getattr(knob, "choices", None)
        widget = str(getattr(knob, "control_widget", "") or "")
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

        if widget == "stepper":
            val_str = f"-   {val_str}   +"
            ntype = DR_NODE_TEXT_LABEL
        elif widget == "segmented":
            opts = list(choices or [])
            try:
                idx = opts.index(str(current_value))
            except Exception:
                try:
                    idx = int(current_value or 0)
                except Exception:
                    idx = 0
            if opts:
                idx = max(0, min(int(idx), len(opts) - 1))
            val_str = "  ".join(
                f"[{opt}]" if i == idx else str(opt)
                for i, opt in enumerate(opts)
            )
            ntype = DR_NODE_TEXT_LABEL
        elif widget == "readonly":
            ntype = DR_NODE_TEXT_LABEL
            val_str = f"{label}: {val_str}" if val_str else str(label)
        elif widget == "button":
            ntype = DR_NODE_TEXT_LABEL
            val_str = str(label)

        if widget == "toggle":
            ntype = DR_NODE_KNOB_BOOL

        if ntype == DR_NODE_TEXT_LABEL and widget in ("stepper", "segmented"):
            label = f"{label}: {val_str}"
        elif widget in ("readonly", "button"):
            label = val_str

        self.submit_raw(node_id, rect, ntype,
                        label=label, value_str=val_str,
                        value_norm=_knob_value_norm(knob, current_value),
                        parent_id=parent_id,
                        sibling_order=sibling_order)
        return node_id

    def submit_image_map_panel(self, panel: Any, rect: tuple,
                               node_id_map: dict,
                               parent_id: int = 0,
                               sibling_order: int = -1) -> dict:
        """Submit a non-parameter image-map Panel with one coordinate action."""
        x, y, w, h = rect
        body_key = f"__body__{panel.name}"
        if body_key not in node_id_map:
            node_id_map[body_key] = self._alloc_id()
        body_id = node_id_map[body_key]
        label = getattr(panel, "label", None) or getattr(panel, "name", "map")
        self.submit_raw(
            body_id,
            rect,
            DR_NODE_PANEL_BODY,
            bg=_THEME["bg"],
            border_px=1,
            parent_id=parent_id,
            sibling_order=sibling_order,
        )
        image_map = getattr(panel, "payload", {}) or {}
        mw = max(1, int(image_map.get("width", 1)))
        mh = max(1, int(image_map.get("height", 1)))
        palette = list(image_map.get("palette", []) or [])
        cells = list(image_map.get("cells", []) or [])

        hdr_h = 18
        hdr_key = f"__hdr__{panel.name}"
        if hdr_key not in node_id_map:
            node_id_map[hdr_key] = self._alloc_id()
        self.submit_raw(
            node_id_map[hdr_key],
            (x + 2, y + 2, max(1, w - 4), hdr_h),
            DR_NODE_PANEL_HEADER,
            label=label,
            bg=_THEME["header_bg"],
            border_px=0,
            parent_id=body_id,
            sibling_order=0,
        )

        map_x = x + 4
        map_y = y + hdr_h + 6
        map_w = max(1, w - 8)
        map_h = max(1, h - hdr_h - 10)
        cell_w = max(1, map_w // mw)
        cell_h = max(1, map_h // mh)
        fp = image_map.get("floor_plan", {}) if isinstance(image_map, dict) else {}
        fp_type = str(fp.get("type", ""))
        has_arc_geom = fp_type in ("polar", "arc")
        polar_floor_only = fp_type == "polar"

        def _clear_nodes_with_prefix(prefix: str) -> None:
            stale = [k for k in list(node_id_map.keys()) if isinstance(k, str) and k.startswith(prefix)]
            for k in stale:
                try:
                    self.remove_node(int(node_id_map[k]))
                except Exception:
                    pass
                try:
                    del node_id_map[k]
                except Exception:
                    pass

        def _clear_stale_prefixed_nodes(prefix: str, active_keys: set[str]) -> None:
            stale = [
                k for k in list(node_id_map.keys())
                if isinstance(k, str) and k.startswith(prefix) and k not in active_keys
            ]
            for k in stale:
                try:
                    self.remove_node(int(node_id_map[k]))
                except Exception:
                    pass
                try:
                    del node_id_map[k]
                except Exception:
                    pass

        if polar_floor_only:
            _clear_nodes_with_prefix(f"__circle__.{panel.name}.")
            # Remove non-polar rect cell nodes left from rectangular floor mode.
            # Rect cell keys use __cell__{name}.X.Y (no dot after __cell__).
            stale_rect_cells = [
                k for k in list(node_id_map.keys())
                if isinstance(k, str)
                and k.startswith(f"__cell__{panel.name}.")
                and not k.startswith(f"__cell__{panel.name}.polar.")
            ]
            for k in stale_rect_cells:
                try:
                    self.remove_node(int(node_id_map[k]))
                except Exception:
                    pass
                try:
                    del node_id_map[k]
                except Exception:
                    pass
            side = int(max(8, min(map_w, map_h) * 0.92))
            cx = int(map_x + map_w * 0.5)
            cy = int(map_y + map_h * 0.5)
            ox = int(cx - side * 0.5)
            oy = int(cy - side * 0.5)

            rays = 0
            try:
                rays = max(0, int(fp.get("polar_rays", 0) or 0))
            except Exception:
                rays = 0

            fill_rgba = (31, 38, 48, 242)
            edge_rgba = (82, 97, 117, 255)
            tile_bg = (0.82, 0.78, 0.55, 0.86)
            tile_border = (0.28, 0.32, 0.38, 1.0)
            palette = image_map.get("palette", []) if isinstance(image_map, dict) else []

            # Compute circle geometry for this frame (used for both rasterization and placement).
            _side_f = float(max(8, min(map_w, map_h) * 0.92))
            _ri = max(1.0, _side_f * 0.5 - 3.0 - max(1.0, (_side_f * 0.5 - 3.0) * 0.01))
            _ccx = float(map_x + map_w * 0.5)
            _ccy = float(map_y + map_h * 0.5)

            state_hash = 1469598103934665603
            for cell in cells:
                if isinstance(cell, dict) and cell.get("polar_tile"):
                    state_hash ^= int(cell.get("state", 0)) & 0xFF
                    state_hash = (state_hash * 1099511628211) & 0xFFFFFFFFFFFFFFFF

            sig = (
                side,
                int(fp.get("polar_rays", 0) or 0),
                int(fp.get("radial_segments", 0) or 0),
                len(cells),
                state_hash,
                tuple(int(v) for v in fill_rgba),
                tuple(int(v) for v in edge_rgba),
            )

            if self._polar_floor_sig != sig:
                img = np.zeros((side, side, 4), dtype=np.uint8)
                r = max(1.0, 0.5 * float(side - 3))
                c = 0.5 * float(side - 1)
                edge_px = max(1.0, r * 0.01)
                yy, xx = np.ogrid[:side, :side]
                dx = xx.astype(np.float64) - c
                dy = yy.astype(np.float64) - c
                dist2 = dx * dx + dy * dy
                r2 = r * r
                ri = max(0.0, r - edge_px)
                ri2 = ri * ri
                fill_mask = dist2 <= r2
                edge_mask = np.logical_and(fill_mask, dist2 >= ri2)
                img[fill_mask] = np.array(fill_rgba, dtype=np.uint8)
                img[edge_mask] = np.array(edge_rgba, dtype=np.uint8)

                def _draw_quad(pts: list[tuple[float, float]], accent_rgba: tuple[int, int, int, int], border_rgba: tuple[int, int, int, int], border_px: int = 1) -> None:
                    # Canonicalize winding/order so fill tests are stable.
                    cx_q = 0.25 * (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0])
                    cy_q = 0.25 * (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1])
                    pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy_q, p[0] - cx_q))

                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x0 = max(0, int(math.floor(min(xs))))
                    x1 = min(side - 1, int(math.ceil(max(xs))))
                    y0 = max(0, int(math.floor(min(ys))))
                    y1 = min(side - 1, int(math.ceil(max(ys))))
                    if x1 < x0 or y1 < y0:
                        return

                    def _cross(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> float:
                        abx = b[0] - a[0]
                        aby = b[1] - a[1]
                        apx = p[0] - a[0]
                        apy = p[1] - a[1]
                        return abx * apy - aby * apx

                    def _inside(p: tuple[float, float]) -> bool:
                        s = 0.0
                        for i in range(4):
                            cval = _cross(pts[i], pts[(i + 1) % 4], p)
                            if abs(cval) < 1e-6:
                                continue
                            if s == 0.0:
                                s = 1.0 if cval > 0.0 else -1.0
                            elif (cval > 0.0 and s < 0.0) or (cval < 0.0 and s > 0.0):
                                return False
                        return True

                    def _dist_seg(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> float:
                        abx = b[0] - a[0]
                        aby = b[1] - a[1]
                        apx = p[0] - a[0]
                        apy = p[1] - a[1]
                        den = abx * abx + aby * aby
                        t = (apx * abx + apy * aby) / den if den > 1e-9 else 0.0
                        t = max(0.0, min(1.0, t))
                        dx2 = p[0] - (a[0] + t * abx)
                        dy2 = p[1] - (a[1] + t * aby)
                        return math.hypot(dx2, dy2)

                    edge_half = max(0.5, float(border_px))
                    for py in range(y0, y1 + 1):
                        for px in range(x0, x1 + 1):
                            pp = (px + 0.5, py + 0.5)
                            if not _inside(pp):
                                continue
                            on_edge = False
                            if border_px > 0:
                                dmin = 1e30
                                for i in range(4):
                                    dmin = min(dmin, _dist_seg(pts[i], pts[(i + 1) % 4], pp))
                                on_edge = dmin <= edge_half
                            img[py, px] = np.array(border_rgba if on_edge else accent_rgba, dtype=np.uint8)

                def _draw_line(x0: float, y0: float, x1: float, y1: float, rgba: tuple[int, int, int, int]) -> None:
                    dxl = x1 - x0
                    dyl = y1 - y0
                    steps = max(1, int(max(abs(dxl), abs(dyl))))
                    for si in range(steps + 1):
                        t = float(si) / float(steps)
                        px = int(round(x0 + dxl * t))
                        py = int(round(y0 + dyl * t))
                        if 0 <= px < side and 0 <= py < side:
                            img[py, px] = np.array(rgba, dtype=np.uint8)

                for cell in cells:
                    if not isinstance(cell, dict) or not cell.get("polar_tile"):
                        continue
                    qf = cell.get("quad_xy_frac")
                    if not (isinstance(qf, list) and len(qf) >= 4):
                        continue
                    pts = []
                    for raw_pt in qf[:4]:
                        if isinstance(raw_pt, (list, tuple)) and len(raw_pt) >= 2:
                            pts.append((c + float(raw_pt[0]) * ri, c + float(raw_pt[1]) * ri))
                    if len(pts) != 4:
                        continue
                    state = int(cell.get("state", 0))
                    col_f = palette[state] if palette and 0 <= state < len(palette) else tile_bg
                    col = (
                        max(0, min(255, int(float(col_f[0]) * 255.0 + 0.5))),
                        max(0, min(255, int(float(col_f[1]) * 255.0 + 0.5))),
                        max(0, min(255, int(float(col_f[2]) * 255.0 + 0.5))),
                        max(0, min(255, int(float(col_f[3]) * 255.0 + 0.5))),
                    )
                    bcol = (
                        max(0, min(255, int(float(tile_border[0]) * 255.0 + 0.5))),
                        max(0, min(255, int(float(tile_border[1]) * 255.0 + 0.5))),
                        max(0, min(255, int(float(tile_border[2]) * 255.0 + 0.5))),
                        max(0, min(255, int(float(tile_border[3]) * 255.0 + 0.5))),
                    )
                    _draw_quad(pts, col, bcol, border_px=1)

                # Deterministic rays/rings overlay ("rays and chords" guide).
                try:
                    mg_rays = max(0, int(fp.get("polar_rays", 0) or 0))
                except Exception:
                    mg_rays = 0
                try:
                    mg_rings = max(0, int(fp.get("radial_segments", 0) or 0))
                except Exception:
                    mg_rings = 0
                gcol = (
                    max(0, min(255, int(0.55 * 255.0 + 0.5))),
                    max(0, min(255, int(0.62 * 255.0 + 0.5))),
                    max(0, min(255, int(0.72 * 255.0 + 0.5))),
                    max(0, min(255, int(0.30 * 255.0 + 0.5))),
                )
                if mg_rays > 1:
                    for ray_i in range(mg_rays):
                        a = (2.0 * math.pi * float(ray_i)) / float(mg_rays)
                        _draw_line(c, c, c + math.cos(a) * ri, c + math.sin(a) * ri, gcol)
                if mg_rings > 0:
                    for ring_i in range(1, mg_rings + 1):
                        rr = (ri * float(ring_i)) / float(mg_rings)
                        segs = max(24, int(2.0 * math.pi * rr))
                        px0 = c + rr
                        py0 = c
                        for si in range(1, segs + 1):
                            a = (2.0 * math.pi * float(si)) / float(segs)
                            px1 = c + math.cos(a) * rr
                            py1 = c + math.sin(a) * rr
                            _draw_line(px0, py0, px1, py1, gcol)
                            px0, py0 = px1, py1

                self.load_primitive_atlas_rgba(img, side, side, prim_cols=1)
                self._polar_floor_sig = sig

            outer_key = f"__polar_floor__.{panel.name}.outer"
            if outer_key not in node_id_map:
                node_id_map[outer_key] = self._alloc_id()
            self.submit_raw(
                node_id_map[outer_key],
                (ox, oy, int(_side_f), int(_side_f)),
                DR_NODE_PRIM_ICON,
                bg=(0.0, 0.0, 0.0, 0.0),
                border=(0.0, 0.0, 0.0, 0.0),
                border_px=0,
                icon_id=0,
                parent_id=body_id,
                sibling_order=10,
            )

            _clear_nodes_with_prefix(f"__metagrid__.{panel.name}.")
            _clear_nodes_with_prefix(f"__cell__.{panel.name}.polar.")

            return node_id_map

        # Non-polar modes: remove polar-floor helper nodes if present.
        _clear_nodes_with_prefix(f"__polar_floor__.{panel.name}.")
        _clear_nodes_with_prefix(f"__cell__.{panel.name}.polar.")
        _clear_nodes_with_prefix(f"__metagrid__.{panel.name}.")

        max_r = 0.0
        if has_arc_geom:
            for c in cells:
                if isinstance(c, dict):
                    try:
                        max_r = max(max_r, float(c.get("radius_outer", 0.0)))
                    except Exception:
                        pass
            max_r = max(1e-6, max_r)
        side = float(max(1, min(map_w, map_h)))
        cxp = float(map_x + map_w * 0.5)
        cyp = float(map_y + map_h * 0.5)
        scale = 0.48 * side / max_r if has_arc_geom else 0.0
        default_col = (0.12, 0.14, 0.18, 0.92)
        border_col = (0.28, 0.32, 0.38, 1.0)
        for index, raw_cell in enumerate(cells):
            if isinstance(raw_cell, dict):
                cx = int(raw_cell.get("x", index % mw))
                cy = int(raw_cell.get("y", index // mw))
                state = int(raw_cell.get("state", 0))
                text = str(raw_cell.get("label", ""))
            else:
                cx = index % mw
                cy = index // mw
                state = int(raw_cell)
                text = ""
            if not (0 <= cx < mw and 0 <= cy < mh):
                continue
            col = palette[state] if 0 <= state < len(palette) else default_col
            cell_key = f"__cell__{panel.name}.{cx}.{cy}"
            if cell_key not in node_id_map:
                node_id_map[cell_key] = self._alloc_id()
            if has_arc_geom and isinstance(raw_cell, dict) and all(
                k in raw_cell for k in ("radius_inner", "radius_outer", "angle_start", "angle_end")
            ):
                corners = raw_cell.get("corners")
                pts = []
                if isinstance(corners, list) and corners:
                    for raw_pt in corners[:4]:
                        if isinstance(raw_pt, (list, tuple)) and len(raw_pt) >= 2:
                            pts.append((float(raw_pt[0]), float(raw_pt[1])))
                if not pts:
                    r0 = float(raw_cell.get("radius_inner", 0.0))
                    r1 = float(raw_cell.get("radius_outer", r0))
                    a0 = float(raw_cell.get("angle_start", 0.0))
                    a1 = float(raw_cell.get("angle_end", a0))
                    pts = [
                        (r0 * math.cos(a0), r0 * math.sin(a0)),
                        (r1 * math.cos(a0), r1 * math.sin(a0)),
                        (r1 * math.cos(a1), r1 * math.sin(a1)),
                        (r0 * math.cos(a1), r0 * math.sin(a1)),
                    ]
                pxs = [cxp + p[0] * scale for p in pts]
                pys = [cyp - p[1] * scale for p in pts]
                rx0 = int(round(min(pxs)))
                ry0 = int(round(min(pys)))
                rx1 = int(round(max(pxs)))
                ry1 = int(round(max(pys)))
                rx = max(map_x, min(rx0, map_x + map_w - 2))
                ry = max(map_y, min(ry0, map_y + map_h - 2))
                rw = max(2, min(rx1 - rx0, map_x + map_w - rx))
                rh = max(2, min(ry1 - ry0, map_y + map_h - ry))
                cell_rect = (rx, ry, rw, rh)
            elif isinstance(raw_cell, dict) and all(k in raw_cell for k in ("ui_x0", "ui_y0", "ui_x1", "ui_y1")):
                ux0 = float(np.clip(raw_cell.get("ui_x0", 0.0), 0.0, 1.0))
                uy0 = float(np.clip(raw_cell.get("ui_y0", 0.0), 0.0, 1.0))
                ux1 = float(np.clip(raw_cell.get("ui_x1", 1.0), 0.0, 1.0))
                uy1 = float(np.clip(raw_cell.get("ui_y1", 1.0), 0.0, 1.0))
                rx0 = int(round(map_x + ux0 * max(0, map_w - 1)))
                ry0 = int(round(map_y + uy0 * max(0, map_h - 1)))
                rx1 = int(round(map_x + ux1 * max(0, map_w - 1)))
                ry1 = int(round(map_y + uy1 * max(0, map_h - 1)))
                rx = max(map_x, min(rx0, rx1))
                ry = max(map_y, min(ry0, ry1))
                rw = max(2, abs(rx1 - rx0))
                rh = max(2, abs(ry1 - ry0))
                rw = min(rw, map_x + map_w - rx)
                rh = min(rh, map_y + map_h - ry)
                cell_rect = (rx, ry, rw, rh)
            elif isinstance(raw_cell, dict) and "ui_cx" in raw_cell and "ui_cy" in raw_cell:
                uxc = float(np.clip(raw_cell.get("ui_cx", 0.5), 0.0, 1.0))
                uyc = float(np.clip(raw_cell.get("ui_cy", 0.5), 0.0, 1.0))
                uw = float(np.clip(raw_cell.get("ui_w", 0.08), 0.01, 1.0))
                uh = float(np.clip(raw_cell.get("ui_h", 0.08), 0.01, 1.0))
                rw = max(2, int(round(uw * map_w)))
                rh = max(2, int(round(uh * map_h)))
                rx = int(round(map_x + uxc * max(0, map_w - 1))) - rw // 2
                ry = int(round(map_y + uyc * max(0, map_h - 1))) - rh // 2
                rx = max(map_x, min(rx, map_x + map_w - rw))
                ry = max(map_y, min(ry, map_y + map_h - rh))
                cell_rect = (rx, ry, rw, rh)
            else:
                cell_rect = (map_x + cx * cell_w, map_y + cy * cell_h, cell_w, cell_h)
            self.submit_raw(
                node_id_map[cell_key],
                cell_rect,
                DR_NODE_PRIM_RECT,
                label=text,
                bg=(0.0, 0.0, 0.0, 0.0),
                accent=col,
                border=border_col,
                border_px=1,
                parent_id=body_id,
                sibling_order=10 + cy * mw + cx,
            )
        return node_id_map

    def submit_panel(self, panel: Any, rect: tuple,
                     node_id_map: dict | None = None,
                     knob_values: dict | None = None,
                     parent_id: int = 0,
                     sibling_order: int = -1,
                     action_rects: dict | None = None,
                     knob_rects: dict | None = None) -> dict:
        if node_id_map is None:
            node_id_map = {}
        if knob_values is None:
            knob_values = {}

        x, y, w, h = rect
        payload = getattr(panel, "payload", {}) or {}
        if isinstance(payload, dict) and payload.get("type") == "image_map":
            return self.submit_image_map_panel(
                panel,
                rect,
                node_id_map,
                parent_id=parent_id,
                sibling_order=sibling_order,
            )

        body_key = f"__body__{panel.name}"
        if body_key not in node_id_map:
            node_id_map[body_key] = self._alloc_id()
        body_id = node_id_map[body_key]
        self.submit_raw(node_id_map[body_key], (x, y, w, h),
                        DR_NODE_PANEL_BODY, bg=_THEME["bg"], border_px=1,
                        parent_id=parent_id,
                        sibling_order=sibling_order)

        HDR_H   = 20
        hdr_key = f"__hdr__{panel.name}"
        if hdr_key not in node_id_map:
            node_id_map[hdr_key] = self._alloc_id()
        lbl = getattr(panel, "label", None) or panel.name
        self.submit_raw(node_id_map[hdr_key], (x, y, w, HDR_H),
                        DR_NODE_PANEL_HEADER, label=lbl,
                        bg=_THEME["header_bg"], border_px=0,
                        parent_id=body_id,
                        sibling_order=0)

        cursor_y = y + HDR_H + 2
        KNOB_H   = 40
        PAD      = 2
        panel_bottom = y + h
        action_first = bool(payload.get("action_first", False)) if isinstance(payload, dict) else False
        actions = payload.get("actions", []) if isinstance(payload, dict) else []

        def _submit_action_rows(start_y: int) -> int:
            cy = int(start_y)
            for action_i, action in enumerate(actions or []):
                if not isinstance(action, dict):
                    continue
                if cy + 24 > panel_bottom - PAD:
                    break
                action_key = str(action.get("key", action_i))
                label = str(action.get("label", action_key))
                map_key = f"__action__{panel.name}.{action_key}"
                if map_key not in node_id_map:
                    node_id_map[map_key] = self._alloc_id()
                self.submit_raw(
                    node_id_map[map_key],
                    (x + PAD, cy, w - PAD * 2, 24),
                    DR_NODE_TEXT_LABEL,
                    label=label,
                    bg=_THEME["header_bg"],
                    border_px=1,
                    parent_id=body_id,
                    sibling_order=50 + action_i,
                )
                if action_rects is not None:
                    action_rects[f"{panel.name}.{action_key}"] = (x + PAD, cy, w - PAD * 2, 24)
                cy += 24 + PAD
            return cy

        if action_first:
            cursor_y = _submit_action_rows(cursor_y)

        for knob_i, knob in enumerate(getattr(panel, "knobs", []) or []):
            if cursor_y + KNOB_H > panel_bottom - PAD:
                break
            kname = getattr(knob, "name", str(id(knob)))
            if kname not in node_id_map:
                node_id_map[kname] = self._alloc_id()
            self.submit_knobspec(
                knob,
                (x + PAD, cursor_y, w - PAD * 2, KNOB_H),
                current_value=knob_values.get(kname),
                node_id=node_id_map[kname],
                parent_id=body_id,
                sibling_order=10 + knob_i,
            )
            if knob_rects is not None:
                knob_rects[kname] = {
                    "rect": (x + PAD, cursor_y, w - PAD * 2, KNOB_H),
                    "widget": str(getattr(knob, "control_widget", "") or ""),
                    "choices": list(getattr(knob, "choices", []) or []),
                    "default": getattr(knob, "default", None),
                    "low": float(getattr(knob, "low", 0.0)),
                    "high": float(getattr(knob, "high", 1.0)),
                    "step": float(getattr(knob, "step", 0.0)),
                    "dtype": str(getattr(knob, "dtype", "float")),
                }
            cursor_y += KNOB_H + PAD

        if not action_first:
            cursor_y = _submit_action_rows(cursor_y)

        for sub_i, sub in enumerate(getattr(panel, "panels", []) or []):
            remaining_h = panel_bottom - cursor_y - PAD
            if remaining_h < 60:
                break
            sub_h    = max(60, remaining_h)
            sub_rect = (x + PAD, cursor_y, w - PAD * 2, sub_h)
            self.submit_panel(sub, sub_rect, node_id_map, knob_values,
                              parent_id=body_id,
                              sibling_order=100 + sub_i,
                              action_rects=action_rects,
                              knob_rects=knob_rects)
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

        _depth_state = self._begin_overlay_blit()
        try:
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
        finally:
            self._end_overlay_blit(_depth_state)

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

        _depth_state = self._begin_overlay_blit()
        try:
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
        finally:
            self._end_overlay_blit(_depth_state)

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
