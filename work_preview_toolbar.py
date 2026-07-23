"""Panel/knob-backed view tabs over the work preview."""
from __future__ import annotations

from typing import Any

from camera_software.grid_knob_toolbar import GridKnobToolbar
from controls import Panel, button_knob
from camera_software.toolbar_manifests import work_preview_tabs_panel


class WorkPreviewToolbar:
    HEIGHT = 26
    MODES = ("whole-work", "work-piece")
    FAST_PREVIEW_PRODUCT_ID = "camera.surface-scan"

    def __init__(self, mode: str = "whole-work") -> None:
        selected = str(mode).strip().lower()
        if selected not in self.MODES:
            raise ValueError(f"unsupported work preview mode {selected!r}")
        self.mode = selected
        self.panel = work_preview_tabs_panel()
        self._grid = GridKnobToolbar(self.panel, palette="status")
        self._gpu_products: tuple[tuple[str, str], ...] = ()
        self._user_selected_mode = False
        self.sequence_capture_armed = False

    @property
    def selected_product_id(self) -> str:
        return self.mode[4:] if self.mode.startswith("gpu:") else ""

    def set_gpu_products(self, products: Any) -> None:
        """Expose published GPU products as tabs without coupling to producers."""
        current = tuple(
            (str(item.product_id), str(item.tab_label)) for item in products
        )
        if current == self._gpu_products:
            return
        self._gpu_products = current
        knobs = [
            button_knob("whole-work", "WHOLE WORK"),
            button_knob("work-piece", "WORK PIECE"),
            *(button_knob(f"gpu:{product_id}", label)
              for product_id, label in current),
            button_knob("capture-frame", "CAPTURE"),
            button_knob("capture-sequence", "SEQUENCE"),
        ]
        base = work_preview_tabs_panel()
        payload = dict(base.payload)
        grid = dict(payload.get("grid", {}))
        grid["columns"] = max(2, len(knobs))
        payload["grid"] = grid
        self.panel = Panel(base.name, base.label, knobs=knobs, payload=payload)
        self._grid = GridKnobToolbar(self.panel, palette="status")
        valid = set(self.MODES) | {f"gpu:{key}" for key, _label in current}
        if self.mode not in valid:
            self.mode = "whole-work"
        fast_mode = f"gpu:{self.FAST_PREVIEW_PRODUCT_ID}"
        if not self._user_selected_mode and fast_mode in valid:
            # The optical-engine frontend keeps its fast compositional product
            # in the centre by default. Explicit user tab choices remain sticky.
            self.mode = fast_mode

    def render(self, destination: Any, rect: tuple[int, int, int, int]) -> None:
        values = {name: name == self.mode for name in self.MODES}
        values.update({
            f"gpu:{key}": self.mode == f"gpu:{key}"
            for key, _label in self._gpu_products
        })
        values["capture-frame"] = False
        values["capture-sequence"] = self.sequence_capture_armed
        self._grid.render(destination, rect, values)

    def handle_event(
        self, event: Any, rect: tuple[int, int, int, int]
    ) -> str | None:
        routed = self._grid.route_event(event, rect)
        if routed is None:
            return None
        name, _delta = routed
        if name == "capture-frame":
            return "work-preview:capture-frame"
        if name == "capture-sequence":
            self.sequence_capture_armed = not self.sequence_capture_armed
            return "work-preview:capture-sequence"
        if name not in self.MODES and not name.startswith("gpu:"):
            return None
        self.mode = name
        self._user_selected_mode = True
        return f"work-preview:{name}"


__all__ = ["WorkPreviewToolbar"]
