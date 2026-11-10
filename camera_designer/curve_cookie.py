"""Finite planar cookies from original Bezier curves, never a raster mask.

This implements the existing ParametricSurface CPU/GLSL contract. It is not
registered with the production mixed-scene reader: that reader must dispatch
this predicate before accepting a proxy plane hit. A proxy alone is not a cookie.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

from .parametric_surfaces import ParametricSurface

MAX_CURVES = 512
SCHEMA = "spectral.curve-cookie.v1"


def _coefficients(points):
    p = np.asarray(points, np.float64)
    if len(p) == 2:
        return np.stack((p[0], p[1]-p[0], p[0]*0, p[0]*0))
    if len(p) == 3:
        return np.stack((p[0], 2*(p[1]-p[0]), p[0]-2*p[1]+p[2], p[0]*0))
    return np.stack((p[0], 3*(p[1]-p[0]), 3*(p[0]-2*p[1]+p[2]),
                     -p[0]+3*p[1]-3*p[2]+p[3]))


def _value(c, t):
    return ((c[3]*t+c[2])*t+c[1])*t+c[0]


def _y_intervals(c):
    # Exact polynomial derivative roots partition the curve into monotone
    # intervals. The subsequent root solve samples the curve, not a polyline.
    a, b, d = 3*c[3, 1], 2*c[2, 1], c[1, 1]
    roots = []
    if a == 0:
        if b != 0:
            roots = [-d/b]
    else:
        disc = b*b-4*a*d
        if disc >= 0:
            q = -.5*(b+math.copysign(math.sqrt(disc), b))
            roots = [q/a, d/q] if q != 0 else [-b/(2*a)]
    knots = [0., *sorted(set(float(t) for t in roots if 0 < t < 1)), 1.]
    return tuple(zip(knots[:-1], knots[1:]))


def _validated_contours(contours):
    result = []
    count = 0
    for contour in contours:
        curves = []
        for curve in contour:
            p = np.asarray(curve, np.float64)
            if p.shape not in ((2, 2), (3, 2), (4, 2)) or not np.isfinite(p).all():
                raise ValueError("Cookie curves require finite line/quadratic/cubic control points")
            if curves and tuple(p[0]) != curves[-1][-1]:
                raise ValueError("Cookie contour endpoints must join exactly")
            curves.append(tuple(tuple(float(x) for x in row) for row in p))
            count += 1
            if count > MAX_CURVES:
                raise ValueError(f"Cookie exceeds {MAX_CURVES} source curves")
        if not curves or curves[-1][-1] != curves[0][0]:
            raise ValueError("Cookie contours must close exactly")
        result.append(tuple(curves))
    if not result:
        raise ValueError("Cookie needs at least one closed outline")
    return tuple(result)


@dataclass
class CurveCookie(ParametricSurface):
    """Zero-thickness opaque rectangular sheet with even-odd curve openings.

    All local coordinates are metres; local +Z is the sheet normal. ``invert``
    swaps open and blocked regions inside the sheet, not outside its bounds.
    Curve boundary ties use the declared half-open horizontal crossing rule.
    """
    contours: tuple
    half_size: tuple[float, float]
    z_pos: float = 0.
    invert: bool = False
    source: dict | None = None
    glsl_type = "curve_cookie"

    def __post_init__(self):
        self.contours = _validated_contours(self.contours)
        self.half_size = tuple(float(x) for x in self.half_size)
        self.z_pos = float(self.z_pos)
        if len(self.half_size) != 2 or not all(math.isfinite(x) and 0 < x <= 10 for x in self.half_size):
            raise ValueError("Cookie half-size must be positive and at most 10 metres")
        if not math.isfinite(self.z_pos) or type(self.invert) is not bool:
            raise ValueError("Invalid cookie plane or inversion")
        # Source metadata is detached from the caller and cannot alter geometry.
        import json
        self.source = json.loads(json.dumps(self.source or {}, allow_nan=False))
        self._pieces = tuple((c, a, b) for loop in self.contours for curve in loop
                             for c in [_coefficients(curve)] for a, b in _y_intervals(c))

    def open_at(self, x: float, y: float) -> bool:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Cookie coordinates must be finite")
        inside = False
        for c, a, b in self._pieces:
            ya, yb = _value(c, a)[1], _value(c, b)[1]
            if (ya > y) == (yb > y):
                continue
            lo, hi = a, b
            for _ in range(48):
                mid = .5*(lo+hi)
                if (_value(c, mid)[1] > y) == (ya > y):
                    lo = mid
                else:
                    hi = mid
            if x < _value(c, .5*(lo+hi))[0]:
                inside = not inside
        return inside != self.invert

    def intersect(self, ro, rd):
        ro, rd = np.asarray(ro, np.float64), np.asarray(rd, np.float64)
        miss = (math.inf, np.zeros(3), np.zeros(3))
        if ro.shape != (3,) or rd.shape != (3,) or not np.isfinite([ro, rd]).all():
            raise ValueError("Cookie ray must have finite XYZ origin and direction")
        if abs(rd[2]) < 1e-12:
            return miss
        t = (self.z_pos-ro[2])/rd[2]
        if t <= 1e-9:
            return miss
        hit = ro+t*rd
        if abs(hit[0]) > self.half_size[0] or abs(hit[1]) > self.half_size[1] or self.open_at(*hit[:2]):
            return miss
        return t, hit, np.array([0., 0., 1. if rd[2] < 0 else -1.])

    def glsl_intercept_fn(self, fn_name="surf_intercept"):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", fn_name):
            raise ValueError("Invalid GLSL function name")
        # Constants retain the polynomial and parameter intervals. Float32
        # evaluation and 24 bracket steps have explicit finite precision.
        f = lambda x: format(float(x), ".17e")
        x = ",\n".join("vec4("+",".join(f(v) for v in c[:, 0])+")" for c, _, _ in self._pieces)
        y = ",\n".join("vec4("+",".join(f(v) for v in c[:, 1])+")" for c, _, _ in self._pieces)
        limits = ",\n".join(f"vec2({f(a)},{f(b)})" for _, a, b in self._pieces)
        n = len(self._pieces)
        return f"""
const vec4 {fn_name}_cx[{n}]=vec4[{n}]({x});
const vec4 {fn_name}_cy[{n}]=vec4[{n}]({y});
const vec2 {fn_name}_interval[{n}]=vec2[{n}]({limits});
float {fn_name}_poly(vec4 c,float t){{return ((c.w*t+c.z)*t+c.y)*t+c.x;}}
bool {fn_name}(vec3 ro,vec3 rd,out float t,out vec3 normal){{
 t=1e30;normal=vec3(0);
 if(abs(rd.z)<1e-12)return false;
 float distance=({f(self.z_pos)}-ro.z)/rd.z;
 if(distance<=1e-9)return false;
 vec3 h=ro+distance*rd;
 if(abs(h.x)>{f(self.half_size[0])}||abs(h.y)>{f(self.half_size[1])})return false;
 bool inside=false;
 for(int i=0;i<{n};i++){{
  vec4 cy={fn_name}_cy[i];float lo={fn_name}_interval[i].x,hi={fn_name}_interval[i].y;
  float ya={fn_name}_poly(cy,lo),yb={fn_name}_poly(cy,hi);
  if((ya>h.y)==(yb>h.y))continue;
  for(int step=0;step<24;step++){{float mid=(lo+hi)*0.5;
   if(({fn_name}_poly(cy,mid)>h.y)==(ya>h.y))lo=mid;else hi=mid;}}
  if(h.x<{fn_name}_poly({fn_name}_cx[i],(lo+hi)*0.5))inside=!inside;
 }}
 if(inside!={'true' if self.invert else 'false'})return false;
 t=distance;normal=vec3(0,0,rd.z<0.0?1.0:-1.0);return true;
}}"""

    def to_dict(self):
        return {"schema": SCHEMA, "type": self.glsl_type, "fill_rule": "even-odd",
                "boundary_rule": "half-open-horizontal-crossing", "unit": "metres",
                "contours": [[[[*p] for p in curve] for curve in loop] for loop in self.contours],
                "half_size": list(self.half_size), "z_pos": self.z_pos, "invert": self.invert,
                "source": self.source}

    @classmethod
    def from_dict(cls, d):
        if d.get("schema") != SCHEMA or d.get("type") != cls.glsl_type or d.get("fill_rule") != "even-odd" or d.get("unit") != "metres" or d.get("boundary_rule") != "half-open-horizontal-crossing":
            raise ValueError("Unsupported curve cookie schema or geometry semantics")
        return cls(d["contours"], d["half_size"], d["z_pos"], d["invert"], d.get("source"))


def text_cookie(text: str, *, family="DejaVu Sans", weight="bold", style="normal", height_m=.02, border_m=.002):
    """Reuse SceneOrder's original font path, preserving all Bezier controls.

    This bounded plain-text builder refuses missing glyphs, TeX/math, multiline
    layout and silent font fallback. Existing mesh extrusions are a separate
    explicitly tessellated product; no mesh or bitmap is used here.
    """
    if not isinstance(text, str) or not text.strip() or not 1 <= len(text) <= 64 or "$" in text or any(ord(c)<32 for c in text):
        raise ValueError("Cookie text needs 1-64 plain single-line characters without math syntax")
    if not math.isfinite(height_m) or not 1e-5 <= height_m <= 1 or not math.isfinite(border_m) or not 0 <= border_m <= 1:
        raise ValueError("Invalid cookie height or border in metres")
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.ft2font import FT2Font
    from matplotlib.path import Path
    from scene_orders import _font_path
    from camera_software.render_assets import FontAssetSpec
    filename = findfont(FontProperties(family=family, weight=weight, style=style), fallback_to_default=False)
    ft = FT2Font(filename)
    if any(not c.isspace() and ft.get_char_index(ord(c)) == 0 for c in text):
        raise ValueError("Selected font lacks a requested glyph")
    font = FontAssetSpec(family=family, weight=weight, style=style, file=filename)
    path = _font_path(text, font.scene_order_mapping())
    bounds = path.get_extents()
    if not np.isfinite(bounds.extents).all() or bounds.height <= 0:
        raise ValueError("Text has no visible outline")
    scale = height_m/bounds.height
    center = np.array([(bounds.x0+bounds.x1)/2, (bounds.y0+bounds.y1)/2])
    contours, loop, current, start = [], [], None, None
    for vertices, code in path.iter_segments(curves=True, simplify=False, remove_nans=False):
        points = (np.asarray(vertices).reshape(-1, 2)-center)*scale
        if code == Path.MOVETO:
            if loop:
                raise ValueError("Font path contains an unclosed contour")
            current = start = points[0]
        elif code in (Path.LINETO, Path.CURVE3, Path.CURVE4):
            if current is None:
                raise ValueError("Font path segment without contour start")
            loop.append(np.vstack((current, points)).tolist()); current = points[-1]
        elif code == Path.CLOSEPOLY:
            if loop:
                if not np.array_equal(current, start):
                    loop.append(np.vstack((current, start)).tolist())
                contours.append(loop)
            loop, current, start = [], None, None
        else:
            raise ValueError("Unsupported font path command")
    if loop:
        raise ValueError("Font path ended with an open contour")
    return CurveCookie(contours, (bounds.width*scale/2+border_m, height_m/2+border_m),
                       source={"kind":"font-outline", "text":text, "font":font.identity_payload(),
                               "source_owner":"scene_orders._font_path", "curve_conversion":"none"})
