"""Canonical knob toolbar for ray-trace settings."""
from __future__ import annotations

from typing import Any

from camera_software.grid_knob_toolbar import GridKnobToolbar
from camera_software.ray_trace_settings import RayTraceSettings
from camera_software.toolbar_manifests import (
    BOUNCE_OPTIONS,
    BUNDLE_OPTIONS,
    EPOCH_OPTIONS,
    RAY_OPTIONS,
    SAMPLE_OPTIONS,
    T5_OPTIONS,
    GRID_MODE_OPTIONS,
    SUBDIVISION_AXIS_OPTIONS,
    LOCKED_GRID_OPTIONS,
    ray_trace_toolbar_panel,
)


class RayTraceToolbar:
    HEIGHT = GridKnobToolbar.HEIGHT

    def __init__(self, settings: RayTraceSettings | None = None) -> None:
        self.settings = (settings or RayTraceSettings()).validated()
        self._ray_options = RAY_OPTIONS
        self._epoch_options = EPOCH_OPTIONS
        self._bundle_options = BUNDLE_OPTIONS
        self._sample_options = SAMPLE_OPTIONS
        self._t5_options = T5_OPTIONS
        self._bounce_options = BOUNCE_OPTIONS
        self._grid_mode_options = tuple(value.lower() for value in GRID_MODE_OPTIONS)
        self._subdivision_options = SUBDIVISION_AXIS_OPTIONS
        self._locked_grid_options = LOCKED_GRID_OPTIONS
        self.panel = ray_trace_toolbar_panel()
        self._grid = GridKnobToolbar(self.panel, palette="blue")

    @staticmethod
    def _cycle(value: int, options: tuple[int, ...], delta: int) -> int:
        nearest = min(range(len(options)), key=lambda i: abs(options[i] - value))
        return options[(nearest + delta) % len(options)]

    def _change(self, field: str, options: tuple[int, ...], delta: int) -> str:
        values = self.settings.mapping()
        values[field] = self._cycle(int(values[field]), options, delta)
        self.settings = RayTraceSettings(**values).validated()
        return "ray-settings-changed"

    def _toggle_transport(self, delta: int = 1) -> str:
        values = self.settings.mapping()
        modes = ("continuous", "fixed", "depth")
        current = modes.index(self.settings.transport_mode)
        values["transport_mode"] = modes[(current + delta) % len(modes)]
        allowed = (1, 3, 8, 16, 32)
        if values["lane_count"] not in allowed:
            values["lane_count"] = 3
        self.settings = RayTraceSettings(**values).validated()
        return "ray-settings-changed"

    def _cycle_lanes(self, delta: int) -> str:
        values = self.settings.mapping()
        options = (1, 3, 8, 16, 32)
        values["lane_count"] = self._cycle(
            self.settings.lane_count, options, delta
        )
        self.settings = RayTraceSettings(**values).validated()
        return "ray-settings-changed"

    @staticmethod
    def _index(value: Any, options: tuple[Any, ...]) -> int:
        return options.index(value)

    def _knob_values(self) -> dict[str, int]:
        settings = self.settings
        return {
            "transport_mode": self._index(
                settings.transport_mode, ("continuous", "fixed", "depth")
            ),
            "lane_count": self._index(settings.lane_count, (1, 3, 8, 16, 32)),
            "total_rays": self._index(settings.total_rays, self._ray_options),
            "max_sensor_epochs": self._index(settings.max_sensor_epochs, self._epoch_options),
            "epoch_bundle_count": self._index(settings.epoch_bundle_count, self._bundle_options),
            "sensor_samples_per_node": self._index(settings.sensor_samples_per_node, self._sample_options),
            "max_bounces": self._index(settings.max_bounces, self._bounce_options),
            "sensor_t5_pair_budget": self._index(settings.sensor_t5_pair_budget, self._t5_options),
            "grid_mode": self._index(settings.grid_mode, self._grid_mode_options),
            "subdivision_axis": self._index(settings.subdivision_axis, self._subdivision_options),
            "locked_grid_columns": self._index(settings.locked_grid_columns, self._locked_grid_options),
            "locked_grid_rows": self._index(settings.locked_grid_rows, self._locked_grid_options),
        }

    def knob_values(self) -> dict[str, int]:
        return self._knob_values()

    def handle_routed(self, name: str, delta: int) -> str:
        if name == "transport_mode":
            return self._toggle_transport(delta)
        if name == "lane_count":
            return self._cycle_lanes(delta)
        if name == "grid_mode":
            values = self.settings.mapping()
            choices = self._grid_mode_options
            values[name] = choices[
                (choices.index(values[name]) + delta) % len(choices)
            ]
            self.settings = RayTraceSettings(**values).validated()
            return "ray-settings-changed"
        options = {
            "total_rays": self._ray_options,
            "max_sensor_epochs": self._epoch_options,
            "epoch_bundle_count": self._bundle_options,
            "sensor_samples_per_node": self._sample_options,
            "max_bounces": self._bounce_options,
            "sensor_t5_pair_budget": self._t5_options,
            "subdivision_axis": self._subdivision_options,
            "locked_grid_columns": self._locked_grid_options,
            "locked_grid_rows": self._locked_grid_options,
        }[name]
        return self._change(name, options, delta)

    def render(self, destination: Any, rect: tuple[int, int, int, int]) -> None:
        self._grid.render(destination, rect, self._knob_values())

    def handle_event(self, event: Any, rect: tuple[int, int, int, int]) -> str | None:
        routed = self._grid.route_event(event, rect)
        if routed is None:
            return None
        name, delta = routed
        return self.handle_routed(name, delta)


__all__ = ["RayTraceToolbar"]
