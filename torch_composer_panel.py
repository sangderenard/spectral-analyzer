"""torch_composer_panel.py — Composition UI panel for TorchComposerNode.

Standalone surface-based widget.  Import and deploy alongside
ParametricCurveEditor, PlotWidget, and other sub-unit panels — no
analytic_driver Panel inheritance, no patch-level assumptions.

Event API (ParametricCurveEditor-compatible)
─────────────────────────────────────────────
    panel.on_mouse_down(button, x, y)   # screen-space coords
    panel.on_mouse_up(button)
    panel.on_mouse_move(x, y)
    panel.on_scroll(dx, dy, x, y)       # mousewheel
    panel.on_key_down(key, mods)        # pygame key constant + mod flags
    panel.render()                      # → pygame.Surface

Action callbacks — assign externally before use
────────────────────────────────────────────────
    panel.on_recompose = callable       # ↺ Recompose button or R key
    panel.on_play      = callable       # ▶ Play button
    panel.on_save      = callable       # 💾 Save button
"""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pygame
    from torch_composer_node import TorchComposerNode


# ─────────────────────────────────────────────────────────────────────────────
# TorchComposerPanel
# ─────────────────────────────────────────────────────────────────────────────

class TorchComposerPanel:
    """Composition panel wired to a TorchComposerNode.

    Uses CompositionPanel from analytic_driver as a private rendering
    engine.  Exposes a clean, purpose-named surface-widget API with no
    patch-panel ancestry visible to the host.
    """

    def __init__(self, node: "TorchComposerNode") -> None:
        from analytic_driver import CompositionPanel

        self._node: Any = node

        self._inner: Any = CompositionPanel()
        self._inner.set_patch(node.patch, node.key)

        # Relabel the four sequence action buttons to sequencer semantics.
        # Empty string hides a button.
        self._inner._seq_btn_labels = {
            "deploy":      "\u21ba Recompose",
            "demo":        "\u25b6 Play",
            "render_fund": "\U0001f4cb Score",
            "render":      "\U0001f4be Save",
        }
        self._inner.on_deploy_chord = lambda: self._fire(self.on_recompose)
        self._inner.on_demo_play    = lambda: self._fire(self.on_play)
        self._inner.on_render       = lambda: self._fire(self.on_save)
        self._inner.on_render_fund  = lambda: self._fire(self.on_save_score)

        # Start with the Sequence section expanded so buttons are immediately visible.
        self._inner._seq_collapsed = False

        # Public action callbacks — assign before the first event is dispatched
        self.on_recompose:  Any = None
        self.on_play:       Any = None
        self.on_save:       Any = None
        self.on_save_score: Any = None   # serialize arrangement to .score.json

    # ── Event API ────────────────────────────────────────────────────────────

    def on_mouse_down(self, button: int, x: float, y: float,
                      mods: int = 0) -> bool:
        import pygame
        ev = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            pos=(int(x), int(y)),
            button=button,
        )
        return bool(self._inner.handle_event(ev))

    def on_mouse_up(self, button: int) -> None:
        import pygame
        ev = pygame.event.Event(
            pygame.MOUSEBUTTONUP,
            pos=pygame.mouse.get_pos(),
            button=button,
        )
        self._inner.handle_event(ev)

    def on_mouse_move(self, x: float, y: float) -> None:
        import pygame
        ev = pygame.event.Event(
            pygame.MOUSEMOTION,
            pos=(int(x), int(y)),
            rel=(0, 0),
            buttons=(0, 0, 0),
        )
        self._inner.handle_event(ev)

    def on_scroll(self, dx: float, dy: float,
                  x: float = 0.0, y: float = 0.0) -> bool:
        """Mousewheel scroll.  dy > 0 = scroll up."""
        import pygame
        button = 4 if dy > 0 else 5
        ev = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            pos=(int(x), int(y)),
            button=button,
        )
        return bool(self._inner.handle_event(ev))

    def on_key_down(self, key: int, mods: int = 0) -> bool:
        """Handle a keydown.  R key fires on_recompose; all others forwarded."""
        import pygame
        if key == pygame.K_r:
            self._fire(self.on_recompose)
            return True
        ev = pygame.event.Event(
            pygame.KEYDOWN,
            key=key,
            mod=mods,
            unicode="",
        )
        return bool(self._inner.handle_event(ev))

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self) -> "pygame.Surface | None":
        """Return a rendered pygame.Surface for the composition panel."""
        self._inner.set_patch(self._node.patch, self._node.key)
        return self._inner.render()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _fire(self, cb: Any) -> None:
        if callable(cb):
            cb()


# ─────────────────────────────────────────────────────────────────────────────
# TorchComposerPianoRoll
# ─────────────────────────────────────────────────────────────────────────────

class TorchComposerPianoRoll:
    """Piano-roll view for a TorchComposerNode.

    Wraps EditorCanvas in piano-roll mode.  resolved_notes are populated by
    TorchComposerNode._hook after each composition pass.
    """

    def __init__(self, node: "TorchComposerNode") -> None:
        from analytic_driver import EditorCanvas
        self._node: Any = node
        self._canvas: Any = EditorCanvas()
        try:
            from analytic_driver import EditorMode
            self._canvas._mode = EditorMode.PIANO_ROLL
        except Exception:
            pass
        self._canvas.suppress_auto_sync = True

    # ── Event API ────────────────────────────────────────────────────────────

    def on_mouse_down(self, button: int, x: float, y: float,
                      mods: int = 0) -> bool:
        import pygame
        ev = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            pos=(int(x), int(y)),
            button=button,
        )
        handle_fn = getattr(self._canvas, "_handle_piano_roll_event", None)
        if callable(handle_fn):
            return bool(handle_fn(ev, self._node.patch, 0, 0))
        return False

    def on_mouse_up(self, button: int) -> None:
        import pygame
        ev = pygame.event.Event(
            pygame.MOUSEBUTTONUP,
            pos=pygame.mouse.get_pos(),
            button=button,
        )
        handle_fn = getattr(self._canvas, "_handle_piano_roll_event", None)
        if callable(handle_fn):
            handle_fn(ev, self._node.patch, 0, 0)

    def on_mouse_move(self, x: float, y: float) -> None:
        import pygame
        ev = pygame.event.Event(
            pygame.MOUSEMOTION,
            pos=(int(x), int(y)),
            rel=(0, 0),
            buttons=(0, 0, 0),
        )
        handle_fn = getattr(self._canvas, "_handle_piano_roll_event", None)
        if callable(handle_fn):
            handle_fn(ev, self._node.patch, 0, 0)

    def on_key_down(self, key: int, mods: int = 0) -> bool:
        import pygame
        ev = pygame.event.Event(pygame.KEYDOWN, key=key, mod=mods, unicode="")
        handle_fn = getattr(self._canvas, "_handle_piano_roll_event", None)
        if callable(handle_fn):
            return bool(handle_fn(ev, self._node.patch, 0, 0))
        return False

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self, surf: "pygame.Surface", font=None) -> None:
        render_fn = getattr(self._canvas, "_render_piano_roll_view", None)
        if callable(render_fn):
            render_fn(surf, self._node.patch, font)
