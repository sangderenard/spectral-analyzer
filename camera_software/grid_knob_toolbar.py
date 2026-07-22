"""Physical-resolution renderer for canonical grid-laid-out knob panels."""
from __future__ import annotations

from typing import Any, Mapping

from controls import Panel

from .control_layout import ControlLayoutDesign, layout_control_panel


class GridKnobToolbar:
    """Render and route one ``Panel`` without inventing parallel hitboxes."""

    HEIGHT = 52

    def __init__(
        self,
        panel: Panel,
        *,
        palette: str = "blue",
    ) -> None:
        self.panel = panel
        self.palette = str(palette)
        self._surface = None
        self._font = None
        self._small_font = None
        self._design: ControlLayoutDesign | None = None
        self._origin = (0, 0)
        self._knobs = {
            str(getattr(knob, "name", "")): knob
            for knob in panel.knobs
        }

    def _colors(self):
        if self.palette == "amber":
            return {
                "background": (16, 20, 25, 250),
                "cell": (35, 31, 29),
                "border": (119, 91, 68),
                "label": (187, 157, 128),
                "value": (242, 233, 221),
                "accent": (197, 157, 119),
            }
        if self.palette == "status":
            return {
                "background": (12, 16, 20, 250),
                "cell": (22, 29, 35),
                "border": (65, 91, 98),
                "label": (127, 166, 174),
                "value": (220, 236, 232),
                "accent": (113, 194, 170),
            }
        return {
            "background": (13, 16, 22, 250),
            "cell": (28, 33, 44),
            "border": (72, 87, 112),
            "label": (141, 158, 186),
            "value": (232, 235, 242),
            "accent": (125, 154, 205),
        }

    @staticmethod
    def _value_text(knob: Any, value: Any) -> str:
        choices = list(getattr(knob, "choices", []) or [])
        if choices:
            try:
                index = int(value)
                return str(choices[index]) if 0 <= index < len(choices) else str(value)
            except (TypeError, ValueError):
                return str(value)
        dtype = str(getattr(knob, "dtype", ""))
        if dtype == "bool":
            return "ON" if bool(value) else "OFF"
        fmt = str(getattr(knob, "fmt", ".3g"))
        unit = str(getattr(knob, "unit", ""))
        try:
            text = format(float(value), fmt)
        except (TypeError, ValueError):
            text = str(value)
        return f"{text} {unit}".strip()

    def render(
        self,
        destination: Any,
        rect: tuple[int, int, int, int],
        values: Mapping[str, Any],
    ) -> None:
        import pygame

        x, y, width, height = map(int, rect)
        if width <= 0 or height <= 0:
            return
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("monospace", 10, bold=True)
            self._small_font = pygame.font.SysFont("monospace", 7)
        if self._surface is None or self._surface.get_size() != (width, height):
            self._surface = pygame.Surface((width, height), pygame.SRCALPHA)
        colors = self._colors()
        self._surface.fill(colors["background"])
        self._design = layout_control_panel(
            self.panel, width, height, knob_values=values
        )
        self._origin = (x, y)
        headers = []
        for element in self._design.elements:
            ex, ey, ew, eh = element.rect
            if element.kind == "header":
                headers.append(element)
            elif element.kind == "knob":
                host = pygame.Rect(ex, ey, ew, eh)
                pygame.draw.rect(self._surface, colors["cell"], host)
                pygame.draw.rect(self._surface, colors["border"], host, 1)
                knob = self._knobs[element.route_key]
                value_text = self._value_text(knob, element.value)
                widget = str(getattr(knob, "control_widget", ""))
                is_button = widget == "button"
                if is_button and bool(element.value):
                    pygame.draw.rect(self._surface, colors["accent"], host)
                    pygame.draw.rect(self._surface, colors["value"], host, 1)
                label_text = (
                    value_text
                    if is_button and list(getattr(knob, "choices", []) or [])
                    else element.label
                )
                label_font = self._font if is_button else self._small_font
                label = label_font.render(
                    label_text,
                    True,
                    colors["value"] if is_button else colors["label"],
                )
                self._surface.blit(
                    label,
                    (
                        host.centerx - label.get_width() // 2
                        if is_button else host.right - label.get_width() - 4,
                        host.y + max(0, (host.height - label.get_height()) // 2)
                        if is_button else host.y,
                    ),
                )
                value = self._font.render(
                    "" if is_button else value_text, True, colors["value"]
                )
                self._surface.blit(
                    value,
                    (
                        host.centerx - value.get_width() // 2,
                        host.centery - value.get_height() // 2,
                    ),
                )
                if widget not in {"readonly", "button"}:
                    left = self._small_font.render("<", True, colors["accent"])
                    right = self._small_font.render(">", True, colors["accent"])
                    bottom = host.bottom - left.get_height()
                    self._surface.blit(left, (host.x + 5, bottom))
                    self._surface.blit(
                        right,
                        (host.right - right.get_width() - 5, bottom),
                    )
        # Panel titles are annotations over the first cell, not layout columns.
        # Paint them last so centered values retain the full row height behind
        # the top-left annotation instead of being displaced by it.
        for element in headers:
            ex, ey, _ew, _eh = element.rect
            title = self._font.render(element.label, True, colors["value"])
            self._surface.blit(title, (ex + 3, ey))
        pygame.draw.line(
            self._surface, colors["border"],
            (0, height - 1), (width, height - 1), 2,
        )
        destination.blit(self._surface, (x, y))

    def route_event(
        self,
        event: Any,
        _rect: tuple[int, int, int, int],
    ) -> tuple[str, int] | None:
        import pygame

        if (
            event.type != pygame.MOUSEBUTTONDOWN
            or event.button != 1
            or self._design is None
        ):
            return None
        local_x = int(event.pos[0]) - self._origin[0]
        local_y = int(event.pos[1]) - self._origin[1]
        route = self._design.route_at(local_x, local_y)
        if route is None or route[0] != "knob":
            return None
        name = route[1]
        knob = self._knobs[name]
        if str(getattr(knob, "control_widget", "")) == "readonly":
            return None
        if str(getattr(knob, "control_widget", "")) == "button":
            return name, 1
        rx, _ry, rw, _rh = self._design.knob_routes[name]["rect"]
        return name, (-1 if local_x < rx + rw // 2 else 1)


__all__ = ["GridKnobToolbar"]
