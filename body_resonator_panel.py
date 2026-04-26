"""body_resonator_panel.py — Standalone pygame panel for acoustic body/cavity settings.

Exposes the live CavityScene parameters on an InstrumentNode — aperture feedback,
diffuse tail, and atmospheric conditions.  Values are read from
``instrument._body_scene`` when a scene exists and written back in real-time as
sliders change.  "Rebuild" clears the scene so it is recreated fresh on next use.

API
---
    panel = BodyResonatorPanel(instrument_node)
    panel.render(surf, x=0, y=0, w=320, h=400, font=font)
    panel.on_mouse_down(mx, my)
    panel.on_mouse_move(mx, my)
    panel.on_mouse_up()
    panel.on_rebuild = callable   # fired when Rebuild is clicked
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

# ── Palette ────────────────────────────────────────────────────────────────────
_C_BG     = (22,  22,  28)
_C_HEADER = (30,  30,  38)
_C_SEP    = (50,  50,  65)
_C_TEXT   = (180, 180, 195)
_C_DIM    = (110, 110, 130)
_C_AMBER  = (240, 170,  50)
_C_BLUE   = (90,  160, 250)
_C_GREEN  = (60,  205, 105)
_C_RED    = (220,  80,  80)
_C_TRACK  = (55,  55,  68)

_ROW_H = 26
_GAP   = 4
_PAD   = 6

# ── Body descriptions (mirrors instrument_panel._BODY_DESC) ───────────────────
_BODY_DESC: dict = {
    "string_plate": "Ribs + top plate + back plate  (violin / cello)",
    "reed_box":     "Cylindrical bore               (clarinet / sax)",
    "brass_bell":   "Tapered bore + flaring bell    (trumpet / trombone)",
    "drum_shell":   "Cylindrical shell + membrane   (drum)",
    "pipe_column":  "Open cylinder, both ends       (flute / organ pipe)",
    "voice_body":   "Tapered vocal tract + mouth    (singing voice)",
    "direct":       "Pass-through — no body solve",
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
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
        """Pull current value from the live scene attribute (if getter exists)."""
        if self._getter:
            try:
                self.value = float(max(self.lo, min(self.hi, self._getter())))
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


class _IntStepper:
    """Integer stepper with − / + buttons."""
    __slots__ = ("label", "lo", "hi", "value", "_rect",
                 "_minus_rect", "_plus_rect", "_getter", "_setter")

    def __init__(self, label: str, lo: int, hi: int, value: int,
                 getter: Optional[Callable] = None,
                 setter: Optional[Callable] = None) -> None:
        self.label   = label
        self.lo      = lo
        self.hi      = hi
        self.value   = value
        self._rect         = (0, 0, 1, _ROW_H)
        self._minus_rect   = (0, 0, 1, _ROW_H)
        self._plus_rect    = (0, 0, 1, _ROW_H)
        self._getter = getter
        self._setter = setter

    def sync(self) -> None:
        if self._getter:
            try:
                self.value = int(max(self.lo, min(self.hi, self._getter())))
            except Exception:
                pass

    def draw(self, surf: Any, x: int, y: int, w: int, h: int, font: Any) -> None:
        import pygame
        self._rect = (x, y, w, h)
        pygame.draw.rect(surf, _C_HEADER, (x, y, w, h))
        if font:
            st = font.render(f"{self.label}: {self.value}", True, _C_TEXT)
            surf.blit(st, (x + 4, y + (h - st.get_height()) // 2))
            btn_w = 22
            mr = x + w - _PAD - btn_w
            ml = mr - btn_w - 2
            pygame.draw.rect(surf, _C_TRACK, (ml, y + 2, btn_w, h - 4), border_radius=3)
            pygame.draw.rect(surf, _C_TRACK, (mr, y + 2, btn_w, h - 4), border_radius=3)
            mt = font.render("−", True, _C_AMBER)
            pt = font.render("+", True, _C_AMBER)
            surf.blit(mt, (ml + (btn_w - mt.get_width()) // 2, y + (h - mt.get_height()) // 2))
            surf.blit(pt, (mr + (btn_w - pt.get_width()) // 2, y + (h - pt.get_height()) // 2))
            self._minus_rect = (ml, y + 2, btn_w, h - 4)
            self._plus_rect  = (mr, y + 2, btn_w, h - 4)

    def on_mouse_down(self, mx: int, my: int) -> bool:
        for delta, rect in [(-1, self._minus_rect), (+1, self._plus_rect)]:
            rx, ry, rw, rh = rect
            if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                self.value = max(self.lo, min(self.hi, self.value + delta))
                if self._setter:
                    try:
                        self._setter(self.value)
                    except Exception:
                        pass
                return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Panel
# ──────────────────────────────────────────────────────────────────────────────

class BodyResonatorPanel:
    """Standalone widget for InstrumentNode acoustic cavity settings.

    Reads / writes parameters on ``instrument._body_scene`` (CavityScene) in
    real-time.  When the scene is None (not yet built or after a rebuild) sliders
    show sensible defaults and writes are deferred until the next scene exists.
    """

    def __init__(self, instrument: Any) -> None:
        self.instrument = instrument
        self.on_rebuild: Optional[Callable] = None

        # Build sliders with live getters/setters tied to the scene
        def _scene(self=self):
            return getattr(self.instrument, "_body_scene", None)

        def _aperture(self=self):
            s = _scene()
            aps = getattr(s, "apertures", None) if s else None
            return aps[0] if aps else None

        def _diffuse(self=self):
            s = _scene()
            return getattr(s, "diffuse_tail", None) if s else None

        def _atmo(self=self):
            s = _scene()
            return getattr(s, "atmosphere", None) if s else None

        self._fb_iter = _IntStepper(
            "Feedback iters", 0, 4, 1,
            getter=lambda: getattr(_scene(), "aperture_feedback_iterations", 1),
            setter=lambda v: setattr(_scene(), "aperture_feedback_iterations", v)
                             if _scene() else None,
        )

        self._fb_gain = _Slider(
            "Aperture gain", 0.0, 0.6, 0.16,
            getter=lambda: getattr(_aperture(), "feedback_gain", 0.16),
            setter=lambda v: setattr(_aperture(), "feedback_gain", v)
                             if _aperture() else None,
        )

        self._passive_loss = _Slider(
            "Passive loss", 0.0, 0.98, 0.48,
            getter=lambda: getattr(_aperture(), "passive_loss", 0.48),
            setter=lambda v: setattr(_aperture(), "passive_loss", v)
                             if _aperture() else None,
        )

        self._diff_strength = _Slider(
            "Diffuse strength", 0.0, 1.0, 0.20,
            getter=lambda: getattr(_diffuse(), "strength", 0.20),
            setter=lambda v: setattr(_diffuse(), "strength", v)
                             if _diffuse() else None,
        )

        self._diff_decay = _Slider(
            "Diffuse decay s", 0.01, 1.0, 0.15,
            getter=lambda: getattr(_diffuse(), "decay_s", 0.15),
            setter=lambda v: setattr(_diffuse(), "decay_s", v)
                             if _diffuse() else None,
        )

        self._temp = _Slider(
            "Temperature °C", -20.0, 50.0, 20.0,
            getter=lambda: getattr(_atmo(), "temperature_c", 20.0),
            setter=lambda v: setattr(_atmo(), "temperature_c", v)
                             if _atmo() else None,
        )

        self._humidity = _Slider(
            "Humidity", 0.0, 1.0, 0.5,
            getter=lambda: getattr(_atmo(), "humidity_rel", 0.5),
            setter=lambda v: setattr(_atmo(), "humidity_rel", v)
                             if _atmo() else None,
        )

        self._all_sliders: List[_Slider] = [
            self._fb_gain, self._passive_loss,
            self._diff_strength, self._diff_decay,
            self._temp, self._humidity,
        ]
        self._all_steppers: List[_IntStepper] = [self._fb_iter]

        self._rebuild_rect = (0, 0, 1, _ROW_H)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, surf: Any, x: int, y: int, w: int, h: int, font: Any) -> None:
        import pygame
        pygame.draw.rect(surf, _C_BG, (x, y, w, h))
        cy = y + _PAD

        # Title
        if font:
            t = font.render("BODY RESONATOR", True, _C_AMBER)
            surf.blit(t, (x + (w - t.get_width()) // 2, cy))
            cy += t.get_height() + _GAP * 2

        pygame.draw.line(surf, _C_SEP, (x, cy), (x + w, cy), 1)
        cy += _GAP

        # Body type + description (read-only)
        bt = getattr(self.instrument, "body_type", "string_plate")
        pygame.draw.rect(surf, _C_HEADER, (x + _PAD, cy, w - _PAD * 2, _ROW_H))
        if font:
            tl = font.render(f"Body: {bt}", True, _C_BLUE)
            surf.blit(tl, (x + _PAD + 4, cy + (_ROW_H - tl.get_height()) // 2))
        cy += _ROW_H + _GAP
        if font:
            desc = _BODY_DESC.get(bt, "")
            dt = font.render(desc, True, _C_DIM)
            surf.blit(dt, (x + _PAD, cy))
            cy += dt.get_height() + _GAP * 2

        pygame.draw.line(surf, _C_SEP, (x, cy), (x + w, cy), 1)
        cy += _GAP

        # Scene status indicator
        scene_alive = getattr(self.instrument, "_body_scene", None) is not None
        status_col  = _C_GREEN if scene_alive else _C_DIM
        status_txt  = "Scene: live" if scene_alive else "Scene: not built"
        if font:
            st = font.render(status_txt, True, status_col)
            surf.blit(st, (x + _PAD, cy + (_ROW_H - st.get_height()) // 2))
        cy += _ROW_H + _GAP

        # Sync sliders/steppers from live scene
        for sl in self._all_sliders:
            sl.sync()
        for sp in self._all_steppers:
            sp.sync()

        # Section: Aperture feedback
        if font:
            hdr = font.render("— Aperture —", True, _C_DIM)
            surf.blit(hdr, (x + _PAD, cy))
            cy += hdr.get_height() + _GAP

        self._fb_iter.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
        cy += _ROW_H + _GAP
        self._fb_gain.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
        cy += _ROW_H + _GAP
        self._passive_loss.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
        cy += _ROW_H + _GAP * 2

        # Section: Diffuse tail
        if font:
            hdr = font.render("— Diffuse tail —", True, _C_DIM)
            surf.blit(hdr, (x + _PAD, cy))
            cy += hdr.get_height() + _GAP

        self._diff_strength.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
        cy += _ROW_H + _GAP
        self._diff_decay.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
        cy += _ROW_H + _GAP * 2

        # Section: Atmosphere
        if font:
            hdr = font.render("— Atmosphere —", True, _C_DIM)
            surf.blit(hdr, (x + _PAD, cy))
            cy += hdr.get_height() + _GAP

        self._temp.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
        cy += _ROW_H + _GAP
        self._humidity.draw(surf, x + _PAD, cy, w - _PAD * 2, _ROW_H, font)
        cy += _ROW_H + _GAP * 2

        pygame.draw.line(surf, _C_SEP, (x, cy), (x + w, cy), 1)
        cy += _GAP

        # Rebuild button
        self._rebuild_rect = (x + _PAD, cy, w - _PAD * 2, _ROW_H)
        pygame.draw.rect(surf, _C_RED, self._rebuild_rect, border_radius=4)
        if font:
            bt_txt = font.render("Rebuild body scene", True, (240, 240, 240))
            bx, by, bw, bh = self._rebuild_rect
            surf.blit(bt_txt, (bx + (bw - bt_txt.get_width()) // 2,
                               by + (bh - bt_txt.get_height()) // 2))

    # ── Events ────────────────────────────────────────────────────────────────

    def on_mouse_down(self, mx: int, my: int) -> bool:
        # Rebuild button
        bx, by, bw, bh = self._rebuild_rect
        if bx <= mx <= bx + bw and by <= my <= by + bh:
            self._do_rebuild()
            return True
        # Int stepper
        for sp in self._all_steppers:
            if sp.on_mouse_down(mx, my):
                return True
        # Sliders
        for sl in self._all_sliders:
            if sl.on_mouse_down(mx, my):
                return True
        return False

    def on_mouse_move(self, mx: int, my: int) -> None:
        for sl in self._all_sliders:
            sl.on_mouse_move(mx, my)

    def on_mouse_up(self) -> None:
        for sl in self._all_sliders:
            sl.on_mouse_up()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _do_rebuild(self) -> None:
        if hasattr(self.instrument, "_body_scene"):
            self.instrument._body_scene = None
        if hasattr(self.instrument, "_cavity_state"):
            self.instrument._cavity_state = None
        if callable(self.on_rebuild):
            self.on_rebuild()
