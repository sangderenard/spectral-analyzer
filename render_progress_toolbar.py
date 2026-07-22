"""Read-only canonical knob grid showing whole-render progress and ETA."""
from __future__ import annotations

from typing import Any

from camera_software.grid_knob_toolbar import GridKnobToolbar
from camera_software.render_eta import format_duration
from camera_software.toolbar_manifests import render_progress_panel


class RenderProgressToolbar:
    HEIGHT = GridKnobToolbar.HEIGHT

    def __init__(self) -> None:
        self.panel = render_progress_panel()
        self._grid = GridKnobToolbar(self.panel, palette="status")
        self._values = {
            "phase": "IDLE",
            "overall": "--",
            "elapsed": "--",
            "eta": "--",
            "pause_render": 0,
            "cancel_render": False,
        }
        self._fraction = 0.0
        self._paused = False

    def update(self, event: Any | None) -> None:
        if event is None:
            return
        message = str(getattr(event, "message", ""))
        phase = "ACTIVE"
        if message.startswith("bundle "):
            phase = message.split(" | ", 1)[0].upper()
        completed = int(getattr(event, "completed_work", 0))
        total = int(getattr(event, "total_work", 0))
        fraction = float(getattr(event, "progress_fraction", 0.0))
        if fraction <= 0.0 and total > 0:
            fraction = completed / total
        self._fraction = min(1.0, max(0.0, fraction))
        self._values = {
            "phase": phase,
            "overall": (
                f"{self._fraction * 100.0:.2f}%  {completed}/{total}"
                if total > 0 else "STARTING"
            ),
            "elapsed": format_duration(float(getattr(event, "elapsed_s", 0.0))),
            "eta": format_duration(getattr(event, "eta_s", None)),
            "pause_render": 1 if self._paused else 0,
            "cancel_render": False,
        }

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        self._values["pause_render"] = 1 if self._paused else 0
        self._values["phase"] = "PAUSED" if self._paused else "ACTIVE"

    def handle_event(
        self, event: Any, rect: tuple[int, int, int, int]
    ) -> str | None:
        routed = self._grid.route_event(event, rect)
        if routed is None:
            return None
        if routed[0] == "pause_render":
            return "toggle-render-pause"
        if routed[0] == "cancel_render":
            return "cancel-render"
        return None

    def render(self, destination: Any, rect: tuple[int, int, int, int]) -> None:
        import pygame

        self._grid.render(destination, rect, self._values)
        x, y, width, height = map(int, rect)
        if width <= 0 or height <= 0:
            return
        bar_y = y + height - 4
        pygame.draw.rect(destination, (33, 48, 53), (x, bar_y, width, 3))
        pygame.draw.rect(
            destination,
            (91, 194, 158),
            (x, bar_y, int(round(width * self._fraction)), 3),
        )


__all__ = ["RenderProgressToolbar"]
