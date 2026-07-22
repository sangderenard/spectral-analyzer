"""Panel/knob-backed view tabs over the work preview."""
from __future__ import annotations

from typing import Any

from camera_software.grid_knob_toolbar import GridKnobToolbar
from camera_software.toolbar_manifests import work_preview_tabs_panel


class WorkPreviewToolbar:
    HEIGHT = 26
    MODES = ("whole-work", "work-piece")

    def __init__(self, mode: str = "whole-work") -> None:
        selected = str(mode).strip().lower()
        if selected not in self.MODES:
            raise ValueError(f"unsupported work preview mode {selected!r}")
        self.mode = selected
        self.panel = work_preview_tabs_panel()
        self._grid = GridKnobToolbar(self.panel, palette="status")

    def render(self, destination: Any, rect: tuple[int, int, int, int]) -> None:
        self._grid.render(destination, rect, {
            name: name == self.mode for name in self.MODES
        })

    def handle_event(
        self, event: Any, rect: tuple[int, int, int, int]
    ) -> str | None:
        routed = self._grid.route_event(event, rect)
        if routed is None:
            return None
        name, _delta = routed
        if name not in self.MODES:
            return None
        self.mode = name
        return f"work-preview:{name}"


__all__ = ["WorkPreviewToolbar"]
