"""Live work/asset browser built on the repository scrolling-list widget."""
from __future__ import annotations

from typing import Any, Iterable


class RenderWorkBrowser:
    """Adapter from bakery jobs/objects to ``ScrollableSubpanelList``."""

    def __init__(self, title: str = "WORK / ASSETS") -> None:
        from bass_viewer import ScrollableSubpanelList

        self.widget = ScrollableSubpanelList(
            title, max_height=220, key_prefix="render_work"
        )
        self._signature: tuple[Any, ...] = ()
        self._payloads: dict[str, dict[str, Any]] = {}
        self._surface = None
        self._font = None

    @staticmethod
    def _artifact_preview(subtype: Any) -> str:
        for artifact in getattr(subtype, "artifacts", ()):
            path = str(
                getattr(artifact, "preview_path", "")
                or getattr(artifact, "linear_path", "")
            )
            if path:
                return path
        return ""

    def sync(
        self,
        pending: Iterable[Any],
        bundles: Iterable[Any],
        interfaces: Iterable[Any] = (),
        *,
        active: tuple[str, str, int, bool] = ("", "", 0, False),
    ) -> None:
        """Refresh rows while retaining selection, expansion, and scroll."""

        from bass_viewer import ModularSubpanelSpec

        active_key, active_token, active_pass, is_active = active
        rows: list[tuple[str, str, list[str], tuple[int, int, int], dict[str, Any]]] = []
        if active_token:
            rows.append((
                f"active:{active_key or active_token}",
                f"ACTIVE  {active_token}",
                [
                    f"state: {'integrating' if is_active else 'checkpointed'}",
                    f"pass: {int(active_pass)}",
                ],
                (220, 150, 55),
                {"kind": "active", "token": active_token},
            ))
        for request in pending:
            token_asset = getattr(request, "token_asset", None)
            token = str(getattr(token_asset, "token", "") or "interface")
            target_kind = str(getattr(getattr(request, "target_kind", None), "value", "job"))
            request_key = str(getattr(request, "request_key", token))
            rows.append((
                f"job:{request_key}",
                f"QUEUED  {token}",
                [
                    f"kind: {target_kind}",
                    f"resume pass: {int(getattr(request, 'refinement_pass', 0))}",
                ],
                (95, 135, 210),
                {"kind": "job", "request": request},
            ))
        for bundle in bundles:
            bundle_name = str(getattr(bundle, "display_name", "render object"))
            for subtype in getattr(bundle, "subtypes", ()):
                subtype_key = str(getattr(subtype, "subtype_key", ""))
                text = str(getattr(subtype, "text", ""))
                kind = str(getattr(subtype, "kind", "asset"))
                complete = bool(getattr(subtype, "complete", False))
                artifacts = tuple(getattr(subtype, "artifacts", ()))
                samples = max(
                    (int(getattr(item, "samples", 0)) for item in artifacts),
                    default=0,
                )
                preview = self._artifact_preview(subtype)
                rows.append((
                    f"asset:{subtype_key}",
                    f"{'DONE' if complete else 'PARTIAL'}  {text}",
                    [
                        f"{bundle_name} / {kind}",
                        f"samples: {samples}",
                    ],
                    (70, 165, 105) if complete else (180, 135, 55),
                    {
                        "kind": "asset",
                        "bundle": bundle,
                        "subtype": subtype,
                        "preview_path": preview,
                    },
                ))
        for assembly in interfaces:
            latest = getattr(assembly, "latest", None)
            if latest is None:
                continue
            key = str(getattr(assembly, "assembly_key", "interface"))
            rows.append((
                f"interface:{key}",
                f"INTERFACE  {getattr(assembly, 'display_name', key)}",
                [
                    f"revision: {int(getattr(latest, 'revision', 0))}",
                    "after-render complete",
                ],
                (135, 95, 190),
                {
                    "kind": "interface",
                    "assembly": assembly,
                    "preview_path": str(getattr(latest, "image_path", "")),
                },
            ))
        signature = tuple(
            (key, title, tuple(summary), tuple(accent), payload.get("preview_path", ""))
            for key, title, summary, accent, payload in rows
        )
        if signature == self._signature:
            return
        expanded = {
            spec.key: bool(spec.expanded) for spec in self.widget.subpanels
        }
        specs = [
            ModularSubpanelSpec(
                key=key,
                title=title,
                summary_lines=summary,
                expanded=expanded.get(key, False),
                accent_rgb=accent,
                payload=payload,
            )
            for key, title, summary, accent, payload in rows
        ]
        self._payloads = {spec.key: dict(spec.payload or {}) for spec in specs}
        self.widget.set_subpanels(specs)
        self._signature = signature

    @property
    def selected_payload(self) -> dict[str, Any]:
        return dict(self._payloads.get(self.widget.selected_key or "", {}))

    @property
    def selected_preview_path(self) -> str:
        return str(self.selected_payload.get("preview_path", ""))

    def render(self, destination: Any, rect: tuple[int, int, int, int]) -> None:
        import pygame

        x, y, width, height = map(int, rect)
        if width <= 0 or height <= 0:
            return
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("monospace", 12)
        if self._surface is None or self._surface.get_size() != (width, height):
            self._surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._surface.fill((18, 20, 26, 245))
        self.widget.max_height = max(
            60, height - self.widget.TITLE_H - 4
        )
        self.widget.render(self._surface, self._font, 0, 0, width)
        destination.blit(self._surface, (x, y))

    def handle_event(
        self, event: Any, rect: tuple[int, int, int, int]
    ) -> str | None:
        import pygame

        x, y, width, height = map(int, rect)
        host = pygame.Rect(x, y, width, height)
        if event.type == pygame.MOUSEWHEEL:
            if not host.collidepoint(pygame.mouse.get_pos()):
                return None
            return "scroll" if self.widget.handle_scroll(-int(event.y)) else ""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not host.collidepoint(event.pos):
                return None
            return self.widget.handle_click(
                int(event.pos[0]) - x, int(event.pos[1]) - y
            )
        return None


__all__ = ["RenderWorkBrowser"]