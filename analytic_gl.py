#!/usr/bin/env python3
"""OpenGL rendering primitives for the analytic driver UI."""
from __future__ import annotations

from analytic_shared import *  # noqa: F401,F403
import analytic_shared as _analytic_shared

globals().update({
    k: v for k, v in vars(_analytic_shared).items()
    if not (k.startswith('__') and k.endswith('__'))
})

_tex_size_cache: dict[int, tuple[int, int]] = {}

def _surface_to_gl_tex(surf: pygame.Surface, old_id: int = 0) -> int:
    raw = pygame.image.tostring(surf, "RGBA", False)
    w, h = surf.get_size()
    if old_id and _tex_size_cache.get(old_id) == (w, h):
        # Same size — replace pixel data in-place; no GPU texture re-allocation
        glBindTexture(GL_TEXTURE_2D, old_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, raw)
        return old_id
    if old_id:
        glDeleteTextures([old_id])
        _tex_size_cache.pop(old_id, None)
    tid = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tid)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
    _tex_size_cache[tid] = (w, h)
    return tid


def _surface_rects_to_gl_tex(
    surf: pygame.Surface,
    rects: list[pygame.Rect] | None,
    old_id: int = 0,
) -> int:
    w, h = surf.get_size()
    if not old_id or _tex_size_cache.get(old_id) != (w, h):
        return _surface_to_gl_tex(surf, old_id)
    if not rects:
        return old_id
    viewport = pygame.Rect(0, 0, w, h)
    clipped: list[pygame.Rect] = []
    seen: set[tuple[int, int, int, int]] = set()
    for rect in rects:
        cr = pygame.Rect(rect).clip(viewport)
        if cr.w <= 0 or cr.h <= 0:
            continue
        key = (cr.x, cr.y, cr.w, cr.h)
        if key in seen:
            continue
        seen.add(key)
        clipped.append(cr)
    if not clipped:
        return old_id
    glBindTexture(GL_TEXTURE_2D, old_id)
    for rect in clipped:
        raw = pygame.image.tostring(surf.subsurface(rect), "RGBA", False)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            rect.x,
            rect.y,
            rect.w,
            rect.h,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            raw,
        )
    return old_id


def _draw_tex_quad(tid: int, x: int, y: int, w: int, h: int, ww: int, wh: int) -> None:
    x0 = 2.0 * x / ww - 1.0
    y1 = 1.0 - 2.0 * y / wh
    x1 = 2.0 * (x + w) / ww - 1.0
    y0 = 1.0 - 2.0 * (y + h) / wh
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D, tid)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x0, y1)
    glTexCoord2f(1, 0); glVertex2f(x1, y1)
    glTexCoord2f(1, 1); glVertex2f(x1, y0)
    glTexCoord2f(0, 1); glVertex2f(x0, y0)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def _ndc(px: float, py: float, ww: int, wh: int) -> tuple[float, float]:
    return 2.0 * px / ww - 1.0, 1.0 - 2.0 * py / wh


def _gl_vline(px, y0, y1, ww, wh, col=(1,1,1,1), lw=1.0):
    nx, na = _ndc(px, y0, ww, wh)
    _,  nb = _ndc(px, y1, ww, wh)
    glDisable(GL_TEXTURE_2D)
    glLineWidth(lw)
    glColor4f(*col)
    glBegin(GL_LINES); glVertex2f(nx, na); glVertex2f(nx, nb); glEnd()
    glLineWidth(1.0)


def _gl_hline(py, x0, x1, ww, wh, col=(1,1,1,1), lw=1.0):
    na, ny = _ndc(x0, py, ww, wh)
    nb, _  = _ndc(x1, py, ww, wh)
    glDisable(GL_TEXTURE_2D)
    glLineWidth(lw)
    glColor4f(*col)
    glBegin(GL_LINES); glVertex2f(na, ny); glVertex2f(nb, ny); glEnd()
    glLineWidth(1.0)


def _gl_rect(px, py, pw, ph, ww, wh, col):
    x0, y0 = _ndc(px,      py,      ww, wh)
    x1, y1 = _ndc(px + pw, py + ph, ww, wh)
    glDisable(GL_TEXTURE_2D)
    glColor4f(*col)
    glBegin(GL_QUADS)
    glVertex2f(x0, y0); glVertex2f(x1, y0)
    glVertex2f(x1, y1); glVertex2f(x0, y1)
    glEnd()


def _gl_diamond(cx, cy, r, ww, wh, fill_col, border_col=None):
    top   = _ndc(cx,     cy - r, ww, wh)
    right = _ndc(cx + r, cy,     ww, wh)
    bot   = _ndc(cx,     cy + r, ww, wh)
    left  = _ndc(cx - r, cy,     ww, wh)
    glDisable(GL_TEXTURE_2D)
    glColor4f(*fill_col)
    glBegin(GL_QUADS)
    glVertex2f(*top); glVertex2f(*right); glVertex2f(*bot); glVertex2f(*left)
    glEnd()
    if border_col:
        glColor4f(*border_col)
        glBegin(GL_LINE_LOOP)
        glVertex2f(*top); glVertex2f(*right); glVertex2f(*bot); glVertex2f(*left)
        glEnd()


def _gl_cursor_flag(px, y_top, y_bot, ww, wh, col):
    """Vertical cursor line + triangle flag at top."""
    _gl_vline(px, y_top + 10, y_bot, ww, wh, col, 1.5)
    a, b = _ndc(px,      y_top,      ww, wh)
    c, d = _ndc(px + 10, y_top + 7,  ww, wh)
    e, f = _ndc(px,      y_top + 10, ww, wh)
    glDisable(GL_TEXTURE_2D)
    glColor4f(*col)
    glBegin(GL_TRIANGLES)
    glVertex2f(a, b); glVertex2f(c, d); glVertex2f(e, f)
    glEnd()


def _gl_loop_handle(px, y_top, y_bot, ww, wh, col, is_left: bool):
    _gl_vline(px, y_top, y_bot, ww, wh, col, 2.5)
    bw = 10.0
    x0 = px if is_left else px - bw
    x1 = px + bw if is_left else px
    _gl_hline(y_top,     x0, x1, ww, wh, col, 2.5)
    _gl_hline(y_bot - 1, x0, x1, ww, wh, col, 2.5)
