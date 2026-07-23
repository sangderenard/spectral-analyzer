"""Seven manifest-owned toolbar rows sharing camera and trace settings."""
from __future__ import annotations

from typing import Any

from camera_software.grid_knob_toolbar import GridKnobToolbar
from camera_software.toolbar_manifests import (
    arena_toolbar_panel,
    camera_toolbar_panel,
    exposure_toolbar_panel,
    film_toolbar_panel,
    integrator_toolbar_panel,
    lens_toolbar_panel,
    light_toolbar_panel,
)


class TopToolbarRows:
    ROW_KEYS = (
        "camera-toolbar",
        "lens-toolbar",
        "light-toolbar",
        "film-toolbar",
        "arena-toolbar",
        "integrator-toolbar",
        "exposure-toolbar",
    )

    def __init__(self, ray_trace_toolbar: Any, exposure_toolbar: Any) -> None:
        self.ray_trace_toolbar = ray_trace_toolbar
        self.exposure_toolbar = exposure_toolbar
        panels = (
            camera_toolbar_panel(),
            lens_toolbar_panel(),
            light_toolbar_panel(),
            film_toolbar_panel(),
            arena_toolbar_panel(),
            integrator_toolbar_panel(),
            exposure_toolbar_panel(),
        )
        self.panels = {panel.name: panel for panel in panels}
        self._grids = {
            panel.name: GridKnobToolbar(
                panel,
                palette=(
                    "blue"
                    if panel.name in {"arena-toolbar", "integrator-toolbar"}
                    else "amber"
                ),
            )
            for panel in panels
        }

    def _values(self) -> dict[str, int]:
        return {
            **self.ray_trace_toolbar.knob_values(),
            **self.exposure_toolbar.knob_values(),
        }

    def render(
        self, row_key: str, destination: Any,
        rect: tuple[int, int, int, int],
    ) -> None:
        self._grids[row_key].render(destination, rect, self._values())

    def handle_event(
        self, row_key: str, event: Any, rect: tuple[int, int, int, int],
    ) -> str | None:
        routed = self._grids[row_key].route_event(event, rect)
        if routed is None:
            return None
        name, delta = routed
        return self.handle_routed(name, delta)

    def handle_routed(self, name: str, delta: int) -> str:
        if name in self.ray_trace_toolbar.knob_values():
            return self.ray_trace_toolbar.handle_routed(name, delta)
        return self.exposure_toolbar.handle_routed(name, delta)

    def advanced_rows(self, row_key: str) -> tuple[tuple[str, str, str], ...]:
        """Resolve the compact row into values for the expanded form host."""

        panel = self.panels[row_key]
        values = self._values()
        rows: list[tuple[str, str, str]] = []
        for knob in panel.knobs:
            name = str(getattr(knob, "name", ""))
            value = values[name]
            choices = tuple(getattr(knob, "choices", ()) or ())
            if choices:
                try:
                    value_text = str(choices[int(value)])
                except (IndexError, TypeError, ValueError):
                    value_text = str(value)
            else:
                value_text = str(value)
            rows.append((name, str(getattr(knob, "label", name)), value_text))
        return tuple(rows)


__all__ = ["TopToolbarRows"]
