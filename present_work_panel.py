"""Selection-driven content host for the program's large detail region."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PresentWorkDocument:
    kind: str
    title: str
    subtitle: str = ""
    lines: tuple[str, ...] = ()
    toolbar_row: str = ""


def _flatten_mapping(value: Any, prefix: str = "") -> tuple[str, ...]:
    lines: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, Mapping):
                lines.extend(_flatten_mapping(child, path))
            elif isinstance(child, (list, tuple)):
                summary = ", ".join(map(str, child[:6]))
                if len(child) > 6:
                    summary += f", … ({len(child)} items)"
                lines.append(f"{path}: {summary}")
            else:
                lines.append(f"{path}: {child}")
    return tuple(lines)


class PresentWorkPanel:
    """Own text input and interchangeable detail/advanced presentations."""

    UI_MATERIAL_KINDS = {
        "glyph", "token", "token_string", "interface", "material",
        "parametric_layout", "layout", "sprite", "text", "alphabet_tile",
        "whole_token", "whole_token_string", "interface_after_render",
    }

    def __init__(self, initial_text: str, *, maximum_text_length: int = 500) -> None:
        self.maximum_text_length = max(1, int(maximum_text_length))
        self.text = str(initial_text)[:self.maximum_text_length]
        self.cursor = len(self.text)
        self.document = PresentWorkDocument(
            "text-material", "UI MATERIAL DEMONSTRATION",
            "The selected UI material is demonstrated with editable text.",
        )
        self._surface = None
        self._font = None
        self._small_font = None
        self._scroll = 0
        self._advanced_controls: dict[str, tuple[Any, Any]] = {}

    @property
    def text_editor_active(self) -> bool:
        return self.document.kind == "text-material"

    def show_text_material(self, title: str = "UI MATERIAL DEMONSTRATION") -> None:
        self.document = PresentWorkDocument(
            "text-material", str(title),
            "Editable demonstration content for the selected UI material.",
        )
        self._scroll = 0

    def show_calibration(self, mode: Any, parameters: Mapping[str, Any]) -> None:
        manifest = mode.work_asset_manifest()
        lines = (
            str(getattr(mode, "description", "")),
            "",
            "TEST CONTRACT",
            *_flatten_mapping(manifest),
            "",
            "CURRENT PARAMETERS",
            *_flatten_mapping(dict(parameters)),
        )
        self.document = PresentWorkDocument(
            "calibration",
            f"{str(getattr(mode, 'label', getattr(mode, 'key', 'CALIBRATION'))).upper()} DETAILS",
            "Test-specific information and extended controls.",
            tuple(lines),
        )
        self._scroll = 0

    def show_toolbar(self, row_key: str, title: str) -> None:
        self.document = PresentWorkDocument(
            "toolbar", f"{str(title).upper()} / ADVANCED",
            "The compact row remains immediate; this host owns its expanded form.",
            toolbar_row=str(row_key),
        )
        self._scroll = 0

    def show_work(self, payload: Mapping[str, Any]) -> None:
        payload = dict(payload or {})
        kind = str(payload.get("kind", ""))
        subtype = payload.get("subtype")
        subtype_kind = str(getattr(subtype, "kind", "")).lower()
        if kind in {"active", "job"} or (
            kind == "asset" and subtype_kind in self.UI_MATERIAL_KINDS
        ):
            label = str(
                getattr(subtype, "display_name", "")
                or getattr(subtype, "text", "")
                or "UI MATERIAL DEMONSTRATION"
            )
            self.show_text_material(label)
            return
        lines = _flatten_mapping({
            key: value for key, value in payload.items()
            if key not in {"subtype", "bundle", "request", "result"}
        })
        self.document = PresentWorkDocument(
            "detail", (kind or "PRESENT WORK").replace("_", " ").upper(),
            "Selection-specific detail presentation.", lines,
        )
        self._scroll = 0

    def handle_text_event(self, event: Any) -> bool:
        """Apply one Pygame text/key event only while the text module owns focus."""

        if not self.text_editor_active:
            return False
        import pygame

        changed = False
        if event.type == pygame.TEXTINPUT:
            inserted = str(event.text)
            self.text = (
                self.text[:self.cursor] + inserted + self.text[self.cursor:]
            )[:self.maximum_text_length]
            self.cursor = min(len(self.text), self.cursor + len(inserted))
            changed = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE and self.cursor > 0:
                self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
                self.cursor -= 1
                changed = True
            elif event.key == pygame.K_DELETE and self.cursor < len(self.text):
                self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]
                changed = True
            elif event.key == pygame.K_LEFT:
                self.cursor = max(0, self.cursor - 1)
            elif event.key == pygame.K_RIGHT:
                self.cursor = min(len(self.text), self.cursor + 1)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.text = (
                    self.text[:self.cursor] + "\n" + self.text[self.cursor:]
                )[:self.maximum_text_length]
                self.cursor = min(len(self.text), self.cursor + 1)
                changed = True
        return changed

    @staticmethod
    def _wrap(font: Any, text: str, width: int) -> list[str]:
        words = str(text).split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def render(
        self, destination: Any, rect: tuple[int, int, int, int],
        toolbar_rows: Any,
    ) -> None:
        """Render non-text content; text material remains spectrally composed."""

        if self.text_editor_active:
            self._advanced_controls = {}
            return
        import pygame

        x, y, width, height = map(int, rect)
        if width <= 0 or height <= 0:
            return
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("monospace", 13, bold=True)
            self._small_font = pygame.font.SysFont("monospace", 11)
        if self._surface is None or self._surface.get_size() != (width, height):
            self._surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface = self._surface
        surface.fill((17, 20, 27, 255))
        pygame.draw.rect(surface, (82, 93, 116), surface.get_rect(), 1)
        title = self._font.render(self.document.title, True, (235, 238, 244))
        surface.blit(title, (10, 7))
        subtitle = self._small_font.render(
            self.document.subtitle, True, (148, 166, 194)
        )
        surface.blit(subtitle, (10, 27))
        y_cursor = 48 - self._scroll
        line_height = self._small_font.get_linesize() + 3
        self._advanced_controls = {}

        if self.document.kind == "toolbar":
            rows = toolbar_rows.advanced_rows(self.document.toolbar_row)
            for name, label, value in rows:
                if y_cursor + 26 >= 44 and y_cursor < height:
                    host = pygame.Rect(8, y_cursor, max(1, width - 16), 24)
                    pygame.draw.rect(surface, (28, 33, 44), host)
                    pygame.draw.rect(surface, (70, 85, 110), host, 1)
                    rendered = self._small_font.render(
                        f"{label}   {value}", True, (222, 228, 238)
                    )
                    surface.blit(rendered, (host.x + 7, host.y + 5))
                    left = pygame.Rect(host.right - 62, host.y + 2, 26, 20)
                    right = pygame.Rect(host.right - 31, host.y + 2, 26, 20)
                    for button, glyph in ((left, "<"), (right, ">")):
                        pygame.draw.rect(surface, (42, 50, 66), button)
                        pygame.draw.rect(surface, (108, 132, 176), button, 1)
                        mark = self._small_font.render(glyph, True, (232, 236, 244))
                        surface.blit(mark, (
                            button.centerx - mark.get_width() // 2,
                            button.centery - mark.get_height() // 2,
                        ))
                    self._advanced_controls[name] = (left.move(x, y), right.move(x, y))
                y_cursor += 28
        else:
            for source_line in self.document.lines:
                for line in self._wrap(self._small_font, source_line, width - 20):
                    if y_cursor + line_height >= 44 and y_cursor < height:
                        rendered = self._small_font.render(
                            line, True, (205, 214, 226)
                        )
                        surface.blit(rendered, (10, y_cursor))
                    y_cursor += line_height
        destination.blit(surface, (x, y))

    def handle_panel_event(
        self, event: Any, rect: tuple[int, int, int, int],
    ) -> tuple[str, int] | None:
        if self.text_editor_active:
            return None
        import pygame

        host = pygame.Rect(*map(int, rect))
        if event.type == pygame.MOUSEWHEEL:
            pointer = getattr(event, "pos", pygame.mouse.get_pos())
            if host.collidepoint(pointer):
                self._scroll = max(0, self._scroll - int(event.y) * 28)
            return None
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for name, (left, right) in self._advanced_controls.items():
                if left.collidepoint(event.pos):
                    return name, -1
                if right.collidepoint(event.pos):
                    return name, 1
        return None


__all__ = ["PresentWorkDocument", "PresentWorkPanel"]
