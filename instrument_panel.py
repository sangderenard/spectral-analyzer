"""instrument_panel.py — Standalone pygame panel for InstrumentNode settings.

Controls the identity-level properties of an InstrumentNode: which body type
the cavity simulation uses, per-instrument jitter, and the score-delivery
duration/release parameters.

API
---
    panel = InstrumentPanel(instrument_node)
    panel.render(surf, x=0, y=0, w=320, h=240, font=font)
    panel.on_mouse_down(mx, my)
    panel.on_mouse_move(mx, my)
    panel.on_mouse_up()
    panel.on_rebuild = callable   # fired when body_type or jitter_seed changes
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

_BODY_TYPES: List[str] = [
    "string_plate", "reed_box",  "brass_bell",
    "drum_shell",   "pipe_column", "voice_body", "direct",
]
_BODY_DESC: dict = {
    "string_plate": "Ribs + top plate + back plate  (violin / cello)",
    "reed_box":     "Cylindrical bore               (clarinet / sax)",
    "brass_bell":   "Tapered bore + flaring bell    (trumpet / trombone)",
    "drum_shell":   "Cylindrical shell + membrane   (drum)",
    "pipe_column":  "Open cylinder, both ends       (flute / organ pipe)",
    "voice_body":   "Tapered vocal tract + mouth    (singing voice)",
    "direct":       "Pass-through — no body solve",
}

_C_BG     = (22,  22,  28)
_C_HEADER = (30,  30,  38)
_C_SEP    = (50,  50,  65)
_C_TEXT   = (180, 180, 195)
_C_DIM    = (110, 110, 130)
_C_AMBER  = (240, 170,  50)
_C_BLUE   = (90,  160, 250)
_C_GREEN  = (60,  205, 105)
_C_TRACK  = (55,  55,  68)

_ROW_H  = 26
_GAP    = 4
_PAD    = 6


class _Slider:
    __slots__ = ("label", "lo", "hi", "value", "_drag", "_rect", "_lbl_w")

    def __init__(self, label: str, lo: float, hi: float, value: float,
                 lbl_w: int = 120) -> None:
        self.label  = label
        self.lo     = float(lo)
        self.hi     = float(hi)
        self.value  = float(value)
        self._drag  = False
        self._rect  = (0, 0, 1, _ROW_H)
        self._lbl_w = lbl_w

    def draw(self, surf: Any, x: int, y: int, w: int, h: int, font: Any) -> None:
        import pygame
        self._rect = (x, y, w, h)
        pygame.draw.rect(surf, _C_HEADER, (x, y, w, h))
        if font:
            t = font.render(self.label, True, _C_TEXT)
            surf.blit(t, (x + 4, y + (h - t.get_height()) // 2))
        tx = x + self._lbl_w + 4
        tw = w - self._lbl_w - 56
        ty = y + h // 2
        pygame.draw.line(surf, _C_TRACK, (tx, ty), (tx + tw, ty), 2)
        frac = (self.value - self.lo) / max(self.hi - self.lo, 1e-9)
        thumb_x = int(tx + frac * tw)
        pygame.draw.circle(surf, _C_GREEN, (thumb_x, ty), 5)
        if font:
            vt = font.render(f"{self.value:.4g}", True, _C_DIM)
            surf.blit(vt, (tx + tw + 4, y + (h - vt.get_height()) // 2))

    def on_mouse_down(self, mx: int, my: int) -> bool:
        x, y, w, h = self._rect
        if x <= mx <= x + w and y <= my <= y + h:
            self._drag = True
            self._set_from_x(mx)
            return True
        return False

    def on_mouse_move(self, mx: int, my: int) -> None:
        if self._drag:
            self._set_from_x(mx)

    def on_mouse_up(self) -> None:
        self._drag = False

    def _set_from_x(self, mx: int) -> None:
        x, _, w, _ = self._rect
        tx = x + self._lbl_w + 4
        tw = w - self._lbl_w - 56
        frac = (mx - tx) / max(tw, 1)
        self.value = float(max(self.lo, min(self.hi, self.lo + frac * (self.hi - self.lo))))


class InstrumentPanel:
    """Standalone widget for InstrumentNode identity-level settings."""

    def __init__(self, instrument: Any) -> None:
        self.instrument  = instrument
        self._body_idx   = _BODY_TYPES.index(
            getattr(instrument, "body_type", "string_plate")
        ) if getattr(instrument, "body_type", "string_plate") in _BODY_TYPES else 0
        seed = getattr(instrument, "_body_jitter_seed", None) or getattr(instrument, "body_jitter_seed", 1) or 1
        self._jitter_seed = int(seed)
        self._dur_slider  = _Slider("Duration s",     0.5, 60.0, float(getattr(instrument, "duration_s",     16.0)))
        self._rel_slider  = _Slider("Release tail s",  0.0,  2.0, float(getattr(instrument, "release_tail_s", 0.4)))

        self.on_rebuild: Optional[Callable] = None   # called when body_type / seed changes

        self._body_btn_rect = (0, 0, 1, _ROW_H)
        self._seed_rect     = (0, 0, 1, _ROW_H)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, surf: Any, x: int, y: int, w: int, h: int, font: Any) -> None:
        import pygame
        pygame.draw.rect(surf, _C_BG, (x, y, w, h))
        cy = y + _PAD

        # Title
        if font:
            t = font.render("INSTRUMENT", True, _C_AMBER)
            surf.blit(t, (x + (w - t.get_width()) // 2, cy))
            cy += t.get_height() + _GAP * 2

        pygame.draw.line(surf, _C_SEP, (x, cy), (x + w, cy), 1)
        cy += _GAP

        # Body type button
        bt = _BODY_TYPES[self._body_idx]
        self._body_btn_rect = (x + _PAD, cy, w - _PAD * 2, _ROW_H)
        pygame.draw.rect(surf, _C_HEADER, self._body_btn_rect)
        pygame.draw.rect(surf, _C_SEP, self._body_btn_rect, 1)
        if font:
            tl = font.render(f"Body: {bt}", True, _C_BLUE)
            surf.blit(tl, (x + _PAD + 4, cy + (_ROW_H - tl.get_height()) // 2))
        cy += _ROW_H + _GAP

        # Body description
        if font:
            desc = _BODY_DESC.get(bt, "")
            dt = font.render(desc, True, _C_DIM)
            surf.blit(dt, (x + _PAD, cy))
            cy += dt.get_height() + _GAP * 2

        # Jitter seed row
        self._seed_rect = (x + _PAD, cy, w - _PAD * 2, _ROW_H)
        pygame.draw.rect(surf, _C_HEADER, self._seed_rect)
        if font:
            st = font.render(f"Jitter seed: {self._jitter_seed}", True, _C_TEXT)
            surf.blit(st, (x + _PAD + 4, cy + (_ROW_H - st.get_height()) // 2))
            minus_t = font.render("−", True, _C_AMBER)
            plus_t  = font.render("+", True, _C_AMBER)
            btn_w = 22
            mr = x + w - _PAD - btn_w
            ml = mr - btn_w - 2
            pygame.draw.rect(surf, _C_TRACK, (ml, cy + 2, btn_w, _ROW_H - 4), border_radius=3)
            pygame.draw.rect(surf, _C_TRACK, (mr, cy + 2, btn_w, _ROW_H - 4), border_radius=3)
            surf.blit(minus_t, (ml + (btn_w - minus_t.get_width()) // 2, cy + (_ROW_H - minus_t.get_height()) // 2))
            surf.blit(plus_t,  (mr + (btn_w - plus_t.get_width())  // 2, cy + (_ROW_H - plus_t.get_height())  // 2))
            self._seed_minus = (ml, cy + 2, btn_w, _ROW_H - 4)
            self._seed_plus  = (mr, cy + 2, btn_w, _ROW_H - 4)
        cy += _ROW_H + _GAP * 2

        pygame.draw.line(surf, _C_SEP, (x, cy), (x + w, cy), 1)
        cy += _GAP

        # Duration / release sliders
        self._dur_slider.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
        cy += _ROW_H + _GAP
        self._rel_slider.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
        cy += _ROW_H + _GAP * 2

        # Driver keys (read-only display)
        pygame.draw.line(surf, _C_SEP, (x, cy), (x + w, cy), 1)
        cy += _GAP
        if font:
            dk = getattr(self.instrument, "driver_keys", [])
            t = font.render("Drivers: " + "  ".join(dk), True, _C_DIM)
            surf.blit(t, (x + _PAD, cy))

    # ── Events ────────────────────────────────────────────────────────────────

    def on_mouse_down(self, mx: int, my: int) -> bool:
        bx, by, bw, bh = self._body_btn_rect
        if bx <= mx <= bx + bw and by <= my <= by + bh:
            self._body_idx = (self._body_idx + 1) % len(_BODY_TYPES)
            self.instrument.body_type = _BODY_TYPES[self._body_idx]
            self._invalidate_body()
            return True
        if hasattr(self, "_seed_minus"):
            for delta, rect in [(-1, self._seed_minus), (+1, self._seed_plus)]:
                rx, ry, rw, rh = rect
                if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                    self._jitter_seed = max(0, self._jitter_seed + delta)
                    self._invalidate_body()
                    return True
        if self._dur_slider.on_mouse_down(mx, my):
            self.instrument.duration_s = self._dur_slider.value
            return True
        if self._rel_slider.on_mouse_down(mx, my):
            self.instrument.release_tail_s = self._rel_slider.value
            return True
        return False

    def on_mouse_move(self, mx: int, my: int) -> None:
        self._dur_slider.on_mouse_move(mx, my)
        self._rel_slider.on_mouse_move(mx, my)

    def on_mouse_up(self) -> None:
        self._dur_slider.on_mouse_up()
        self._rel_slider.on_mouse_up()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _invalidate_body(self) -> None:
        """Clear the lazy body scene so it rebuilds on next render."""
        if hasattr(self.instrument, "_body_scene"):
            self.instrument._body_scene  = None
        if hasattr(self.instrument, "_cavity_state"):
            self.instrument._cavity_state = None
        if callable(self.on_rebuild):
            self.on_rebuild()
