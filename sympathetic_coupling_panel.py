"""sympathetic_coupling_panel.py — Standalone pygame panel for sympathetic coupling settings.

Controls StringCouplingConfig (global), per-driver ResonatorString x/y/gain,
sympathy_threshold and sympathy_velocity_floor on an InstrumentNode.

A "Recompute coupling" button calls instrument.rebuild_coupling() to apply any
changes to the StringCouplingConfig or ResonatorString positions.

API
---
    panel = SympatheticCouplingPanel(instrument_node)
    panel.render(surf, x=0, y=0, w=320, h=560, font=font)
    panel.on_mouse_down(mx, my)
    panel.on_mouse_move(mx, my)
    panel.on_mouse_up()
    panel.on_scroll(dy, mx, my)   # optional — scroll driver list
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

# ── Palette ────────────────────────────────────────────────────────────────────
_C_BG     = (22,  22,  28)
_C_HEADER = (30,  30,  38)
_C_SEP    = (50,  50,  65)
_C_TEXT   = (180, 180, 195)
_C_DIM    = (110, 110, 130)
_C_AMBER  = (240, 170,  50)
_C_BLUE   = (90,  160, 250)
_C_GREEN  = (60,  205, 105)
_C_PURPLE = (175, 120, 250)
_C_TRACK  = (55,  55,  68)
_C_ROW_A  = (28,  28,  36)
_C_ROW_B  = (34,  34,  44)

_ROW_H = 26
_GAP   = 4
_PAD   = 6


# ──────────────────────────────────────────────────────────────────────────────
# Generic slider (same pattern as other panels)
# ──────────────────────────────────────────────────────────────────────────────

class _Slider:
    __slots__ = ("label", "lo", "hi", "value", "_drag", "_rect", "_lbl_w",
                 "_getter", "_setter")

    def __init__(self, label: str, lo: float, hi: float, value: float,
                 lbl_w: int = 130,
                 getter: Optional[Callable] = None,
                 setter: Optional[Callable] = None) -> None:
        self.label   = label
        self.lo      = float(lo)
        self.hi      = float(hi)
        self.value   = float(value)
        self._drag   = False
        self._rect   = (0, 0, 1, _ROW_H)
        self._lbl_w  = lbl_w
        self._getter = getter
        self._setter = setter

    def sync(self) -> None:
        if self._getter:
            try:
                v = float(self._getter())
                self.value = max(self.lo, min(self.hi, v))
            except Exception:
                pass

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
        if self._setter:
            try:
                self._setter(self.value)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Per-driver row (compact: x, y, gain — three mini sliders in one row height*3)
# ──────────────────────────────────────────────────────────────────────────────

class _DriverRow:
    """Three compact sliders for one ResonatorString: x, y, drive_gain."""

    def __init__(self, string: Any) -> None:
        self.string = string
        self._sliders: List[_Slider] = [
            _Slider("x pos", 0.0, 1.0,
                    float(getattr(string, "x", 0.0)),
                    lbl_w=50,
                    getter=lambda s=string: getattr(s, "x", 0.0),
                    setter=lambda v, s=string: setattr(s, "x", v)),
            _Slider("y pos", -1.0, 1.0,
                    float(getattr(string, "y", 0.0)),
                    lbl_w=50,
                    getter=lambda s=string: getattr(s, "y", 0.0),
                    setter=lambda v, s=string: setattr(s, "y", v)),
            _Slider("gain", 0.0, 4.0,
                    float(getattr(string, "drive_gain", 1.0)),
                    lbl_w=50,
                    getter=lambda s=string: getattr(s, "drive_gain", 1.0),
                    setter=lambda v, s=string: setattr(s, "drive_gain", v)),
        ]
        self._total_h = _ROW_H * 3 + _GAP * 2

    @property
    def total_h(self) -> int:
        return self._total_h

    def draw(self, surf: Any, x: int, y: int, w: int, font: Any,
             row_col: Any) -> None:
        import pygame
        key = getattr(self.string, "key", "?")
        # Key header row
        pygame.draw.rect(surf, row_col, (x, y, w, _ROW_H))
        if font:
            kt = font.render(key, True, _C_BLUE)
            surf.blit(kt, (x + 4, y + (_ROW_H - kt.get_height()) // 2))
        cy = y + _ROW_H + _GAP
        # sync sliders from live values before drawing
        for sl in self._sliders:
            sl.sync()
            sl.draw(surf, x + _PAD, cy, w - _PAD, _ROW_H, font)
            cy += _ROW_H + _GAP

    def on_mouse_down(self, mx: int, my: int) -> bool:
        return any(sl.on_mouse_down(mx, my) for sl in self._sliders)

    def on_mouse_move(self, mx: int, my: int) -> None:
        for sl in self._sliders:
            sl.on_mouse_move(mx, my)

    def on_mouse_up(self) -> None:
        for sl in self._sliders:
            sl.on_mouse_up()


# ──────────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────────

class SympatheticCouplingPanel:
    """Standalone widget for InstrumentNode sympathetic coupling settings."""

    def __init__(self, instrument: Any) -> None:
        self.instrument = instrument

        cfg = getattr(instrument, "_coupling_config", None)

        def _cfg_attr(name, default, setter_name=None, cfg_ref=cfg):
            sn = setter_name or name

            def _get(n=name, c=None):
                c2 = getattr(self.instrument, "_coupling_config", None)
                return float(getattr(c2, n, default)) if c2 else default

            def _set(v, n=sn, c=None):
                c2 = getattr(self.instrument, "_coupling_config", None)
                if c2:
                    setattr(c2, n, v)

            return _get, _set

        # Global coupling sliders
        def _mk(label, name, lo, hi, default, lbl_w=130):
            g, s = _cfg_attr(name, default)
            return _Slider(label, lo, hi, default, lbl_w=lbl_w, getter=g, setter=s)

        self._base_strength  = _mk("Base strength",  "base_strength",  0.0, 1.0, 0.18)
        self._max_coupling   = _mk("Max coupling",   "max_coupling",   0.0, 1.0, 0.45)
        self._harm_strength  = _mk("Harm. strength", "harmonic_strength", 0.0, 1.0, 0.7)
        self._dist_strength  = _mk("Dist. strength", "distance_strength", 0.0, 1.0, 0.3)
        self._harm_sigma     = _mk("Harm. sigma",    "harmonic_sigma", 0.001, 0.5, 0.08)
        self._dist_sigma     = _mk("Dist. sigma",    "distance_sigma", 0.01, 5.0, 1.25)

        self._sympathy_thr   = _Slider(
            "Sympathy threshold", 0.0, 1.0,
            float(getattr(instrument, "sympathy_threshold", 0.05)),
            getter=lambda: float(getattr(self.instrument, "sympathy_threshold", 0.05)),
            setter=lambda v: setattr(self.instrument, "sympathy_threshold", v),
        )
        self._velocity_floor = _Slider(
            "Velocity floor", 0.0, 0.1,
            float(getattr(instrument, "sympathy_velocity_floor", 0.005)),
            getter=lambda: float(getattr(self.instrument, "sympathy_velocity_floor", 0.005)),
            setter=lambda v: setattr(self.instrument, "sympathy_velocity_floor", v),
        )

        self._global_sliders: List[_Slider] = [
            self._base_strength, self._max_coupling,
            self._harm_strength, self._dist_strength,
            self._harm_sigma,    self._dist_sigma,
            self._sympathy_thr,  self._velocity_floor,
        ]

        # Per-driver rows
        strings = getattr(instrument, "resonator_strings", []) or []
        self._driver_rows: List[_DriverRow] = [_DriverRow(s) for s in strings]

        self._recompute_rect = (0, 0, 1, _ROW_H)
        self._scroll_y: int  = 0     # pixel offset for driver list
        self._driver_list_rect = (0, 0, 1, 1)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, surf: Any, x: int, y: int, w: int, h: int, font: Any) -> None:
        import pygame
        pygame.draw.rect(surf, _C_BG, (x, y, w, h))
        cy = y + _PAD

        # Title
        if font:
            t = font.render("SYMPATHETIC COUPLING", True, _C_AMBER)
            surf.blit(t, (x + (w - t.get_width()) // 2, cy))
            cy += t.get_height() + _GAP * 2

        pygame.draw.line(surf, _C_SEP, (x, cy), (x + w, cy), 1)
        cy += _GAP

        # Global coupling config
        if font:
            hdr = font.render("— Coupling matrix —", True, _C_DIM)
            surf.blit(hdr, (x + _PAD, cy))
            cy += hdr.get_height() + _GAP

        for sl in self._global_sliders[:6]:
            sl.sync()
            sl.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
            cy += _ROW_H + _GAP

        cy += _GAP
        pygame.draw.line(surf, _C_SEP, (x, cy), (x + w, cy), 1)
        cy += _GAP

        # Sympathy thresholds
        if font:
            hdr = font.render("— Injection threshold —", True, _C_DIM)
            surf.blit(hdr, (x + _PAD, cy))
            cy += hdr.get_height() + _GAP

        for sl in self._global_sliders[6:]:
            sl.sync()
            sl.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
            cy += _ROW_H + _GAP

        cy += _GAP

        # Recompute button
        self._recompute_rect = (x + _PAD, cy, w - _PAD * 2, _ROW_H)
        btn_col = _C_PURPLE
        pygame.draw.rect(surf, btn_col, self._recompute_rect, border_radius=4)
        if font:
            bt_txt = font.render("Recompute coupling matrix", True, (240, 240, 240))
            bx, by, bw, bh = self._recompute_rect
            surf.blit(bt_txt, (bx + (bw - bt_txt.get_width()) // 2,
                               by + (bh - bt_txt.get_height()) // 2))
        cy += _ROW_H + _GAP * 2

        pygame.draw.line(surf, _C_SEP, (x, cy), (x + w, cy), 1)
        cy += _GAP

        # Per-driver rows (scrollable)
        if font:
            hdr = font.render(f"— Drivers ({len(self._driver_rows)}) —", True, _C_DIM)
            surf.blit(hdr, (x + _PAD, cy))
            cy += hdr.get_height() + _GAP

        driver_area_y = cy
        driver_area_h = (y + h) - cy
        self._driver_list_rect = (x, driver_area_y, w, driver_area_h)

        # Clip drawing to driver area
        clip_orig = surf.get_clip()
        surf.set_clip((x, driver_area_y, w, driver_area_h))

        draw_y = driver_area_y - self._scroll_y
        for i, row in enumerate(self._driver_rows):
            if draw_y + row.total_h >= driver_area_y and draw_y < driver_area_y + driver_area_h:
                row_col = _C_ROW_A if i % 2 == 0 else _C_ROW_B
                row.draw(surf, x + _PAD, draw_y, w - _PAD * 2, font, row_col)
            draw_y += row.total_h + _GAP

        surf.set_clip(clip_orig)

        # Scroll indicator if content overflows
        total_content = sum(r.total_h + _GAP for r in self._driver_rows)
        if total_content > driver_area_h and font:
            scroll_pct = self._scroll_y / max(1, total_content - driver_area_h)
            bar_h = max(20, int(driver_area_h * driver_area_h / max(total_content, 1)))
            bar_y = driver_area_y + int((driver_area_h - bar_h) * scroll_pct)
            pygame.draw.rect(surf, _C_TRACK,
                             (x + w - 5, bar_y, 4, bar_h), border_radius=2)

    # ── Events ────────────────────────────────────────────────────────────────

    def on_mouse_down(self, mx: int, my: int) -> bool:
        # Recompute button
        bx, by, bw, bh = self._recompute_rect
        if bx <= mx <= bx + bw and by <= my <= by + bh:
            self._recompute()
            return True
        # Driver rows (offset by scroll)
        ax, ay, aw, ah = self._driver_list_rect
        if ax <= mx <= ax + aw and ay <= my <= ay + ah:
            adjusted_y = my + self._scroll_y - ay
            row_y = 0
            for row in self._driver_rows:
                if row_y <= adjusted_y < row_y + row.total_h + _GAP:
                    if row.on_mouse_down(mx, ay + row_y - self._scroll_y +
                                         (my - ay - (adjusted_y - row_y))):
                        return True
                row_y += row.total_h + _GAP
        # Global sliders
        for sl in self._global_sliders:
            if sl.on_mouse_down(mx, my):
                return True
        return False

    def on_mouse_move(self, mx: int, my: int) -> None:
        for sl in self._global_sliders:
            sl.on_mouse_move(mx, my)
        for row in self._driver_rows:
            row.on_mouse_move(mx, my)

    def on_mouse_up(self) -> None:
        for sl in self._global_sliders:
            sl.on_mouse_up()
        for row in self._driver_rows:
            row.on_mouse_up()

    def on_scroll(self, dy: int, mx: int, my: int) -> bool:
        """Call with pygame wheel dy (+1 up / -1 down).  Returns True if consumed."""
        ax, ay, aw, ah = self._driver_list_rect
        if ax <= mx <= ax + aw and ay <= my <= ay + ah:
            total = sum(r.total_h + _GAP for r in self._driver_rows)
            self._scroll_y = max(0, min(max(0, total - ah), self._scroll_y - dy * 24))
            return True
        return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _recompute(self) -> None:
        try:
            self.instrument.rebuild_coupling()
        except AttributeError:
            # Fallback: call build_string_coupling_matrix directly
            try:
                from instrument_node import build_string_coupling_matrix
                self.instrument._coupling_matrix = build_string_coupling_matrix(
                    self.instrument.resonator_strings,
                    self.instrument._coupling_config,
                )
            except Exception:
                pass
