"""Pure layout bridge from canonical control manifests to photographed UI.

The existing OpenGL document renderer and the ray-traced photographer consume
the same Panel tree through this projection. Rectangles are also the click
routes, so the visual design and event ownership cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

from controls import Panel, choice_knob, readonly_knob, stepper_knob, toggle_knob

from .layout_panel import (
    LayoutPanelPrimitive,
    PanelPatchObjectRequest,
    layout_panel_primitive_from_mapping,
)


Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class ControlLayoutElement:
    key: str
    kind: str
    label: str
    rect: Rect
    parent_key: str = ""
    sibling_order: int = 0
    route_key: str = ""
    value: Any = None


@dataclass(frozen=True)
class ControlLayoutDesign:
    """A photographable control layout whose rectangles also route clicks."""

    width: int
    height: int
    elements: tuple[ControlLayoutElement, ...]
    action_routes: Mapping[str, Rect]
    knob_routes: Mapping[str, Mapping[str, Any]]

    def route_at(self, x: int, y: int) -> tuple[str, str] | None:
        """Return the deepest visible knob/action at one design coordinate."""

        point_x, point_y = int(x), int(y)
        for element in reversed(self.elements):
            if element.kind not in {"knob", "action"}:
                continue
            rx, ry, rw, rh = element.rect
            if rx <= point_x < rx + rw and ry <= point_y < ry + rh:
                return element.kind, element.route_key
        return None


def _knob_value(knob: Any, values: Mapping[str, Any]) -> Any:
    name = str(getattr(knob, "name", ""))
    return values[name] if name in values else getattr(knob, "default", None)


def layout_control_panel(
    panel: Panel,
    width: int,
    height: int,
    *,
    rect: Rect | None = None,
    knob_values: Mapping[str, Any] | None = None,
) -> ControlLayoutDesign:
    """Project a Panel tree with the established DocRenderer row geometry."""

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("control layout dimensions must be positive")
    root_rect = rect or (0, 0, width, height)
    if min(root_rect[2:]) <= 0:
        raise ValueError("control layout rectangle must have positive size")
    values = dict(knob_values or {})
    elements: list[ControlLayoutElement] = []
    action_routes: dict[str, Rect] = {}
    knob_routes: dict[str, dict[str, Any]] = {}

    def add_panel(current: Panel, current_rect: Rect, parent: str, order: int) -> None:
        x, y, w, h = current_rect
        body_key = f"panel:{current.name}"
        elements.append(ControlLayoutElement(
            body_key, "panel", current.label or current.name, current_rect,
            parent, order,
        ))
        payload = current.payload if isinstance(current.payload, dict) else {}
        raw_grid = payload.get("grid", {}) or {}
        grid = dict(raw_grid) if isinstance(raw_grid, Mapping) else {}
        grid_mode = str(grid.get("mode", "")).lower() == "grid"
        title_width = (
            max(0, min(w - 1, int(grid.get("title_width", 0))))
            if grid_mode else 0
        )
        title_overlay = grid_mode and bool(grid.get("title_overlay", False))
        header_rect = (
            (x, y, title_width, h)
            if grid_mode and title_width > 0
            else (x, y, w, min(20, h))
        )
        elements.append(ControlLayoutElement(
            f"header:{current.name}", "header", current.label or current.name,
            header_rect, body_key, 0,
        ))
        if grid_mode:
            knobs = list(current.knobs or [])
            columns = max(1, int(grid.get("columns", len(knobs) or 1)))
            rows = max(1, int(math.ceil(len(knobs) / columns)))
            pad = max(0, int(grid.get("padding", 3)))
            pad_x = max(0, int(grid.get("padding_x", pad)))
            pad_y = max(0, int(grid.get("padding_y", pad)))
            column_gap = max(0, int(grid.get("column_gap", 4)))
            row_gap = max(0, int(grid.get("row_gap", 3)))
            grid_x = x if title_overlay else x + title_width
            grid_w = max(1, w if title_overlay else w - title_width)
            inner_w = max(1, grid_w - 2 * pad_x - column_gap * (columns - 1))
            inner_h = max(1, h - 2 * pad_y - row_gap * (rows - 1))
            x_edges = [grid_x + pad_x + (inner_w * index) // columns
                       + column_gap * index for index in range(columns + 1)]
            y_edges = [y + pad_y + (inner_h * index) // rows
                       + row_gap * index for index in range(rows + 1)]
            for knob_i, knob in enumerate(knobs):
                column = knob_i % columns
                row = knob_i // columns
                knob_x = x_edges[column]
                knob_y = y_edges[row]
                knob_w = max(1, x_edges[column + 1] - column_gap - knob_x)
                knob_h = max(1, y_edges[row + 1] - row_gap - knob_y)
                if column == columns - 1:
                    knob_w = max(1, x + w - pad_x - knob_x)
                if row == rows - 1:
                    knob_h = max(1, y + h - pad_y - knob_y)
                name = str(getattr(knob, "name", knob_i))
                current_value = _knob_value(knob, values)
                knob_rect = (knob_x, knob_y, knob_w, knob_h)
                knob_routes[name] = {
                    "rect": knob_rect,
                    "widget": str(getattr(knob, "control_widget", "") or ""),
                    "choices": list(getattr(knob, "choices", []) or []),
                    "default": getattr(knob, "default", None),
                    "low": float(getattr(knob, "low", 0.0)),
                    "high": float(getattr(knob, "high", 1.0)),
                    "step": float(getattr(knob, "step", 0.0)),
                    "dtype": str(getattr(knob, "dtype", "float")),
                }
                elements.append(ControlLayoutElement(
                    f"knob:{name}", "knob",
                    str(getattr(knob, "label", name)), knob_rect,
                    body_key, 10 + knob_i, name, current_value,
                ))
            return
        cursor_y = y + 22
        panel_bottom = y + h
        pad = 2
        actions = payload.get("actions", []) or []

        def add_actions(start_y: int) -> int:
            cy = start_y
            for action_i, action in enumerate(actions):
                if not isinstance(action, dict) or cy + 24 > panel_bottom - pad:
                    continue
                action_key = str(action.get("key", action_i))
                route_key = f"{current.name}.{action_key}"
                action_rect = (x + pad, cy, max(1, w - 2 * pad), 24)
                action_routes[route_key] = action_rect
                elements.append(ControlLayoutElement(
                    f"action:{route_key}", "action",
                    str(action.get("label", action_key)), action_rect,
                    body_key, 50 + action_i, route_key,
                ))
                cy += 26
            return cy

        if bool(payload.get("action_first", False)):
            cursor_y = add_actions(cursor_y)
        for knob_i, knob in enumerate(current.knobs or []):
            if cursor_y + 40 > panel_bottom - pad:
                break
            name = str(getattr(knob, "name", knob_i))
            knob_rect = (x + pad, cursor_y, max(1, w - 2 * pad), 40)
            current_value = _knob_value(knob, values)
            knob_routes[name] = {
                "rect": knob_rect,
                "widget": str(getattr(knob, "control_widget", "") or ""),
                "choices": list(getattr(knob, "choices", []) or []),
                "default": getattr(knob, "default", None),
                "low": float(getattr(knob, "low", 0.0)),
                "high": float(getattr(knob, "high", 1.0)),
                "step": float(getattr(knob, "step", 0.0)),
                "dtype": str(getattr(knob, "dtype", "float")),
            }
            elements.append(ControlLayoutElement(
                f"knob:{name}", "knob", str(getattr(knob, "label", name)),
                knob_rect, body_key, 10 + knob_i, name, current_value,
            ))
            cursor_y += 42
        if not bool(payload.get("action_first", False)):
            cursor_y = add_actions(cursor_y)
        for sub_i, subpanel in enumerate(current.panels or []):
            remaining = panel_bottom - cursor_y - pad
            if remaining < 60:
                break
            sub_rect = (x + pad, cursor_y, max(1, w - 2 * pad), remaining)
            add_panel(subpanel, sub_rect, body_key, 100 + sub_i)
            cursor_y += remaining + pad

    add_panel(panel, root_rect, "", 0)
    return ControlLayoutDesign(
        width, height, tuple(elements), dict(action_routes), dict(knob_routes)
    )


def photography_control_manifest() -> Panel:
    """Initial controls for authoring and inspecting one-shot photographs."""

    return Panel(
        name="photography",
        label="PHOTOGRAPHY",
        knobs=[
            readonly_knob(
                "capture_mode", "Capture", default="single shot"
            ),
            choice_knob(
                "object_view", "Object view",
                ("scene", "light field", "default image"), default=2,
            ),
            stepper_knob(
                "additional_passes", "Additional passes", "int",
                0, 0, 4096, 1,
            ),
            toggle_knob(
                "shrink_whole_tokens", "Shrink whole tokens", default=True
            ),
        ],
        payload={
            "actions": [
                {"key": "integrate_pass", "label": "INTEGRATE ONE PASS"},
                {"key": "resume", "label": "RESUME"},
            ]
        },
    )


class PanelGeometryKind(str, Enum):
    PLANE = "plane"
    VOLUME = "volume"


@dataclass(frozen=True)
class PanelGeometrySpec:
    """Physical panel treatment chosen independently of layout/render owner."""

    kind: PanelGeometryKind = PanelGeometryKind.PLANE
    thickness_m: float = 0.002
    bevel_width_m: float = 0.0015
    bevel_depth_m: float = 0.0005
    geometry_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PanelGeometryKind(self.kind))
        if self.thickness_m <= 0.0:
            raise ValueError("panel geometry thickness must be positive")
        if self.bevel_width_m < 0.0 or self.bevel_depth_m < 0.0:
            raise ValueError("panel bevel dimensions must be non-negative")
        if (self.bevel_width_m == 0.0) != (self.bevel_depth_m == 0.0):
            raise ValueError("panel bevel width and depth must both be zero or positive")


def _panel_geometry(panel: Panel, *, default_thickness: float = 0.002) -> PanelGeometrySpec:
    payload = panel.payload if isinstance(panel.payload, dict) else {}
    raw = payload.get("geometry", {})
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"panel {panel.name!r} geometry must be a mapping")
    return PanelGeometrySpec(
        kind=PanelGeometryKind(str(raw.get("kind", "plane"))),
        thickness_m=float(raw.get("thickness_m", default_thickness)),
        bevel_width_m=float(raw.get("bevel_width_m", 0.0015)),
        bevel_depth_m=float(raw.get("bevel_depth_m", 0.0005)),
        geometry_ref=str(raw.get("geometry_ref", "")),
    )


class LayoutRenderTier(str, Enum):
    MONOFONT_GLYPHS = "monofont_glyphs"
    WHOLE_TOKENS = "whole_token_single_shots"
    INTERFACE_AFTER_RENDER = "interface_after_render"


@dataclass(frozen=True)
class LayoutRenderPipeline:
    """Renderer-neutral production contract carried by a UI manifest."""

    font_family: str
    font_weight: str
    font_style: str
    glyph_alphabet: str
    tiers: tuple[LayoutRenderTier, ...]
    static_tokens: tuple[str, ...] = ()
    whole_token_policy: str = "single_shot_shrink_to_fit"
    final_policy: str = "shared_camera_interface_after_render"

    def __post_init__(self) -> None:
        if not self.font_family.strip() or not self.glyph_alphabet:
            raise ValueError("layout render pipeline requires a font and alphabet")
        required = (
            LayoutRenderTier.MONOFONT_GLYPHS,
            LayoutRenderTier.WHOLE_TOKENS,
            LayoutRenderTier.INTERFACE_AFTER_RENDER,
        )
        if tuple(self.tiers) != required:
            raise ValueError(
                "layout render tiers must be monofont glyphs, whole tokens, "
                "then interface after-render"
            )

    def mapping(self) -> dict[str, Any]:
        return {
            "font": {
                "family": self.font_family,
                "weight": self.font_weight,
                "style": self.font_style,
                "mode": "monofont",
            },
            "glyph_alphabet": self.glyph_alphabet,
            "tiers": [tier.value for tier in self.tiers],
            "static_tokens": list(self.static_tokens),
            "whole_token_policy": self.whole_token_policy,
            "final_policy": self.final_policy,
        }


def layout_render_pipeline(manifest: Panel) -> LayoutRenderPipeline:
    """Read and validate the rendering contract authored by a layout."""

    payload = manifest.payload if isinstance(manifest.payload, dict) else {}
    raw = payload.get("render_pipeline", {})
    if not isinstance(raw, dict):
        raise ValueError("program manifest render_pipeline must be a mapping")
    font = raw.get("font", {})
    if not isinstance(font, dict) or str(font.get("mode", "monofont")) != "monofont":
        raise ValueError("program UI typography must declare monofont mode")
    raw_tiers = raw.get("tiers", (
        LayoutRenderTier.MONOFONT_GLYPHS.value,
        LayoutRenderTier.WHOLE_TOKENS.value,
        LayoutRenderTier.INTERFACE_AFTER_RENDER.value,
    ))
    return LayoutRenderPipeline(
        font_family=str(font.get("family", "DejaVu Sans Mono")),
        font_weight=str(font.get("weight", "bold")),
        font_style=str(font.get("style", "normal")),
        glyph_alphabet=str(raw.get(
            "glyph_alphabet",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )),
        tiers=tuple(LayoutRenderTier(str(tier)) for tier in raw_tiers),
        static_tokens=tuple(dict.fromkeys((
            *(str(token) for token in raw.get("static_tokens", ())),
            *(
                str(panel.label)
                for panel in manifest.panels
                if str((panel.payload or {}).get("role", ""))
                in {"viewport", "widget_host"}
                and str(panel.label).strip()
            ),
            *(
                str(action.get("label", ""))
                for action in (payload.get("actions", ()) or ())
                if str(action.get("label", "")).strip()
            ),
        ))),
        whole_token_policy=str(raw.get(
            "whole_token_policy", "single_shot_shrink_to_fit"
        )),
        final_policy=str(raw.get(
            "final_policy", "shared_camera_interface_after_render"
        )),
    )


@dataclass(frozen=True)
class OpenGLContextRequest:
    """Context ownership requested by one manifest viewport."""

    api: str = "opengl"
    minimum_version: tuple[int, int] = (3, 0)
    prefer_parent: bool = True
    create_if_missing: bool = True


@dataclass(frozen=True)
class ProgramActionSpec:
    """One manifest-authored action placed and photographed by the layout."""

    key: str
    label: str = ""
    icon: str = ""
    align: str = "start"
    width_units: float = 1.0

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("program action key must be non-empty")
        if not self.label.strip() and not self.icon.strip():
            raise ValueError("program action requires a label or icon")
        if self.align not in {"start", "end"}:
            raise ValueError("program action align must be start or end")
        if self.width_units <= 0.0:
            raise ValueError("program action width_units must be positive")

    def mapping(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "icon": self.icon,
            "align": self.align,
            "width_units": self.width_units,
        }


@dataclass(frozen=True)
class ProgramUILayout:
    """Resolved authoritative program manifest in photographed coordinates."""

    manifest: Panel
    width: int
    height: int
    regions: Mapping[str, Rect]
    labels: Mapping[str, str]
    roles: Mapping[str, str]
    context_requests: Mapping[str, OpenGLContextRequest]
    panel_geometry: Mapping[str, PanelGeometrySpec]
    panel_primitives: Mapping[str, LayoutPanelPrimitive]
    actions: Mapping[str, ProgramActionSpec]
    action_primitives: Mapping[str, LayoutPanelPrimitive]
    render_pipeline: LayoutRenderPipeline

    def region(self, key: str) -> Rect:
        return self.regions[str(key)]

    @property
    def panel_object_requests(
        self,
    ) -> Mapping[str, tuple[PanelPatchObjectRequest, ...]]:
        """Object/material requests awaiting a panel-photo job adapter."""

        return {
            owner_id: primitive.object_requests
            for owner_id, primitive in {
                **self.panel_primitives,
                **self.action_primitives,
            }.items()
            if primitive.object_requests
        }

    @property
    def authored_text(self) -> Mapping[str, str]:
        """Every static prototype string owned by the resolved layout."""

        return {
            **{
                key: value for key, value in self.labels.items()
                if key in self.regions and str(value).strip()
            },
            **{
                key: action.label
                for key, action in self.actions.items()
                if action.label.strip()
            },
        }

    def mapping(self) -> dict[str, Any]:
        return {
            "manifest_name": self.manifest.name,
            "size": [self.width, self.height],
            "regions": {
                key: list(rect) for key, rect in self.regions.items()
            },
            "roles": dict(self.roles),
            "actions": {
                key: action.mapping() for key, action in self.actions.items()
            },
            "authored_text": dict(self.authored_text),
            "panel_object_requests": {
                owner_id: [request.mapping() for request in requests]
                for owner_id, requests in self.panel_object_requests.items()
            },
            "action_primitives": {
                key: primitive.mapping()
                for key, primitive in self.action_primitives.items()
            },
            "panel_geometry": {
                key: {
                    "kind": spec.kind.value,
                    "thickness_m": spec.thickness_m,
                    "bevel_width_m": spec.bevel_width_m,
                    "bevel_depth_m": spec.bevel_depth_m,
                    "geometry_ref": spec.geometry_ref,
                }
                for key, spec in self.panel_geometry.items()
            },
            "panel_primitives": {
                key: primitive.mapping()
                for key, primitive in self.panel_primitives.items()
            },
            "render_pipeline": self.render_pipeline.mapping(),
        }


def program_ui_manifest() -> Panel:
    """Authoritative manifest for the ray-photography application frame."""

    from .equipment_manifest import default_equipment_manifest
    from .toolbar_manifests import (
        camera_toolbar_panel,
        exposure_toolbar_panel,
        film_toolbar_panel,
        integrator_toolbar_panel,
        lens_toolbar_panel,
        light_toolbar_panel,
    )

    return Panel(
        name="program-backdrop",
        label="RAY PHOTOGRAPHY BAKERY",
        panels=[
            camera_toolbar_panel(),
            lens_toolbar_panel(),
            light_toolbar_panel(),
            film_toolbar_panel(),
            integrator_toolbar_panel(),
            exposure_toolbar_panel(),
            Panel(
                "camera-panel", "CAMERA",
                payload={
                    "role": "viewport",
                    "column": "camera",
                    "context": "opengl",
                    "context_policy": "parent_then_owned",
                    "geometry": {"kind": "plane"},
                },
            ),
            Panel(
                "work-panel", "WORK",
                payload={
                    "role": "viewport",
                    "column": "work",
                    "context": "opengl",
                    "context_policy": "parent_then_owned",
                    "geometry": {"kind": "plane"},
                },
            ),
            Panel(
                "asset-browser", "ASSETS / JOBS",
                payload={
                    "role": "widget_host",
                    "column": "right",
                    "widget": "ScrollableSubpanelList",
                    "context": "surface",
                    "geometry": {"kind": "plane"},
                },
            ),
            Panel(
                "editor-text", "PRESENT WORK",
                payload={
                    "role": "present_work_host",
                    "widget": "PresentWorkPanel",
                    "content_modules": [
                        "text-material", "calibration", "toolbar", "detail",
                    ],
                    "geometry": {"kind": "plane"},
                },
            ),
            Panel(
                "status-text", "STATUS",
                payload={
                    "role": "status",
                    "geometry": {"kind": "plane"},
                },
            ),
        ],
        payload={
            "type": "program_ui",
            # Equipment is manifest-authored. Resolvers treat every nested
            # group as partial, falling back field-by-field to current defaults.
            "equipment": default_equipment_manifest(),
            # This is the authored viewport geometry, not a render-product
            # resolution.  Work tiles and calibration images may be any size
            # without reflowing the program UI.
            "layout_reference_size_px": [960, 600],
            # Stable authored viewport geometry. This is deliberately
            # independent of the live render tile dimensions.
            "layout_work_viewport_size_px": [256, 256],
            "geometry": {
                "kind": "plane",
                "thickness_m": 0.001,
                "bevel_width_m": 0.0,
                "bevel_depth_m": 0.0,
            },
            "panel_primitive": {
                "kind": "nine_slice",
                "object_key": "layout-panel-style:bakery-slate",
                "subtype_key": "representative-square",
                "border_px": [12, 12, 12, 12],
                "corner_rgba": [62, 69, 84, 255],
                "edge_rgba": [48, 53, 65, 255],
                "center_rgba": [28, 31, 38, 255],
            },
            "control_primitive": {
                "kind": "nine_slice",
                "object_key": "layout-control-style:bakery-slate",
                "subtype_key": "representative-square",
                "border_px": [3, 3, 3, 3],
                "corner_rgba": [86, 94, 112, 255],
                "edge_rgba": [66, 73, 89, 255],
                "center_rgba": [37, 42, 52, 255],
            },
            # The hierarchy is a first-class inspector, not a narrow sidebar.
            # Its host scales from the authored work width just like the camera
            # inspector and may grow beyond the old 640 px ceiling.
            "browser_width_ratio": 1.25,
            "browser_width_min": 240,
            "browser_width_max": 1200,
            # The camera manifest and validation/work hierarchy need more
            # horizontal room than the central image preview. Ratios are
            # relative to the authored work-width baseline.
            "camera_panel_width_ratio": 1.20,
            # Keep the visualization at its authored width.  Side-list growth
            # expands the containing window instead of consuming image space.
            "work_panel_width_ratio": 1.00,
            "render_pipeline": {
                "font": {
                    "mode": "monofont",
                    "family": "DejaVu Sans Mono",
                    "weight": "bold",
                    "style": "normal",
                },
                "glyph_alphabet": (
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz"
                ),
                "tiers": [
                    "monofont_glyphs",
                    "whole_token_single_shots",
                    "interface_after_render",
                ],
                # Viewport/widget labels are collected automatically. Only
                # dynamic authored strings need to be named explicitly.
                "static_tokens": ["SPECTRAL EXPOSURE ACTIVE"],
                "whole_token_policy": "single_shot_shrink_to_fit",
                "final_policy": "shared_camera_interface_after_render",
            },
            "actions": [
                {
                    "key": "work-visual-pass",
                    "label": "VISUAL PASS",
                    "align": "start",
                    "width_units": 7.5,
                },
                {
                    "key": "queue-pause-auto",
                    "label": "TOGGLE WORK",
                    "align": "start",
                    "width_units": 7.5,
                },
                {"key": "window-minimize", "icon": "minimize", "align": "end"},
                {"key": "window-maximize", "icon": "maximize", "align": "end"},
                {"key": "window-close", "icon": "close", "align": "end"},
            ],
        },
    )


def program_frame_metrics(
    work_width: int,
    work_height: int,
    manifest: Panel | None = None,
) -> dict[str, int]:
    """Resolve frame dimensions from the program manifest's sizing policy."""

    manifest = manifest or program_ui_manifest()
    payload = manifest.payload if isinstance(manifest.payload, dict) else {}
    preview_width = max(1, int(work_width))
    preview_height = max(1, int(work_height))
    ratio = max(0.1, float(payload.get("browser_width_ratio", 0.80)))
    browser_min = max(1, int(payload.get("browser_width_min", 160)))
    browser_max = max(browser_min, int(payload.get("browser_width_max", 640)))
    browser_width = max(
        browser_min,
        min(browser_max, int(round(preview_width * ratio))),
    )
    camera_width = max(1, int(round(
        preview_width * float(payload.get("camera_panel_width_ratio", 1.0))
    )))
    work_panel_width = max(1, int(round(
        preview_width * float(payload.get("work_panel_width_ratio", 1.0))
    )))
    frame_width = camera_width + work_panel_width + browser_width
    scale = min(1.0, max(0.4, float(frame_width) / 480.0))
    tool_rows = tuple(
        panel for panel in manifest.panels
        if str((panel.payload or {}).get("role", "")) == "control_grid"
    )
    tool_gap = 1
    tool_row_h = max(1, min(
        int(dict((panel.payload or {}).get("grid", {})).get("row_height", 13))
        for panel in tool_rows
    )) if tool_rows else 0
    tool_h = (
        len(tool_rows) * tool_row_h
        + max(0, len(tool_rows) - 1) * tool_gap
    )
    # The action row, six physical tool rows, and viewport labels each own
    # actual vertical space; none overlap the photographed work panels.
    control_h = max(13, int(round(26.0 * scale)))
    label_h = max(13, int(round(26.0 * scale)))
    label_clearance_h = max(3, int(round(5.0 * scale)))
    editor_h = max(24, int(round(preview_height * 0.35)))
    status_h = max(13, int(round(26.0 * scale)))
    frame_height = (
        control_h + tool_h + label_h + label_clearance_h
        + preview_height + editor_h + status_h
    )
    return {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "browser_width": browser_width,
        "camera_width": camera_width,
        "work_panel_width": work_panel_width,
        "control_h": control_h,
        "tool_h": tool_h,
        "tool_row_h": tool_row_h,
        "tool_gap": tool_gap,
        "label_h": label_h,
        "label_clearance_h": label_clearance_h,
        "editor_h": editor_h,
        "status_h": status_h,
    }


def layout_program_ui(
    manifest: Panel,
    width: int,
    height: int,
    *,
    origin: tuple[int, int] = (0, 0),
    work_width: int | None = None,
    work_height: int | None = None,
) -> ProgramUILayout:
    """Lay out all program hosts from their Panel roles and sizing policy."""

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("program layout dimensions must be positive")
    x0, y0 = map(int, origin)
    if work_width is None:
        payload = manifest.payload if isinstance(manifest.payload, dict) else {}
        ratio = max(0.1, float(payload.get("browser_width_ratio", 0.80)))
        browser_min = max(1, int(payload.get("browser_width_min", 160)))
        browser_max = max(browser_min, int(payload.get("browser_width_max", 640)))
        estimated_browser = max(
            browser_min, min(browser_max, int(round(width * ratio / 2.4)))
        )
        work_width = max(1, (width - estimated_browser) // 2)
    if work_height is None:
        work_height = max(1, height // 2)
    metrics = program_frame_metrics(work_width, work_height, manifest)
    payload = manifest.payload if isinstance(manifest.payload, dict) else {}
    baseline_width = min(int(work_width), max(1, width // 2))
    camera_width = max(1, int(round(
        baseline_width * float(payload.get("camera_panel_width_ratio", 1.0))
    )))
    work_panel_width = max(1, int(round(
        baseline_width * float(payload.get("work_panel_width_ratio", 1.0))
    )))
    if camera_width + work_panel_width >= width:
        available = max(2, width - 1)
        scale_width = available / float(camera_width + work_panel_width)
        camera_width = max(1, int(round(camera_width * scale_width)))
        work_panel_width = max(1, available - camera_width)
    browser_width = max(1, width - camera_width - work_panel_width)
    control_h = min(metrics["control_h"], max(1, height - 1))
    # Preserve at least one pixel each for labels, viewport, editor, and status
    # when a deliberately tiny test/window asks six real rows to fit.
    tool_h = min(metrics["tool_h"], max(0, height - control_h - 4))
    label_h = min(
        metrics["label_h"], max(1, height - control_h - tool_h - 3)
    )
    label_clearance_h = min(
        metrics["label_clearance_h"],
        max(0, height - control_h - tool_h - label_h - 3),
    )
    panel_h = min(
        int(work_height),
        max(1, height - control_h - tool_h - label_h
            - label_clearance_h - 2),
    )
    remaining_h = max(
        2,
        height - control_h - tool_h - label_h
        - label_clearance_h - panel_h,
    )
    status_h = min(metrics["status_h"], max(1, remaining_h - 1))
    editor_h = remaining_h - status_h
    label_y = y0 + control_h + tool_h
    panel_y = label_y + label_h + label_clearance_h
    editor_y = panel_y + panel_h
    status_y = editor_y + editor_h
    main_width = camera_width + work_panel_width

    panels = {str(panel.name): panel for panel in manifest.panels}
    required = {
        "camera-panel", "work-panel", "asset-browser",
        "editor-text", "status-text",
    }
    missing = sorted(required - set(panels))
    if missing:
        raise ValueError(f"program UI manifest is missing panels: {missing}")
    regions: dict[str, Rect] = {
        manifest.name: (x0, y0, width, height),
        "camera-panel": (x0, panel_y, camera_width, panel_h),
        "work-panel": (x0 + camera_width, panel_y, work_panel_width, panel_h),
        "asset-browser": (
            x0 + main_width, panel_y, browser_width,
            max(1, height - (panel_y - y0)),
        ),
        "camera-label": (x0, label_y, camera_width, label_h),
        "work-label": (x0 + camera_width, label_y, work_panel_width, label_h),
        "asset-browser-label": (
            x0 + main_width, label_y, browser_width, label_h,
        ),
        "editor-text": (x0, editor_y, main_width, editor_h),
        "status-text": (x0, status_y, main_width, status_h),
    }
    tool_rows = tuple(
        name for name, panel in panels.items()
        if str((panel.payload or {}).get("role", "")) == "control_grid"
    )
    if tool_rows:
        # The manifest-owned action/window row is first. Every toolbar is a
        # real row below it, before the separate viewport-label row.
        tool_top = y0 + control_h
        available = max(1, label_y - tool_top)
        tool_gap = metrics["tool_gap"]
        total_gap = tool_gap * max(0, len(tool_rows) - 1)
        requested_height = min(
            max(1, int(dict((panels[key].payload or {}).get("grid", {})).get(
                "row_height", 13
            )))
            for key in tool_rows
        )
        row_height = max(1, min(
            requested_height,
            (available - total_gap) // len(tool_rows),
        ))
        used = row_height * len(tool_rows) + total_gap
        tool_y = tool_top
        for row_index, row_key in enumerate(tool_rows):
            regions[row_key] = (
                x0,
                tool_y + row_index * (row_height + tool_gap),
                width,
                row_height,
            )
    margin = min(2, max(0, (width - 1) // 24))
    gap = max(0, min(3, width // 20))
    size = max(1, min(48, max(1, control_h - 2 * margin)))
    right = x0 + width - margin
    raw_actions = list(payload.get("actions", ()) or ())
    actions = {
        str(item.get("key", index)): ProgramActionSpec(
            key=str(item.get("key", index)),
            label=str(item.get("label", "")),
            icon=str(item.get("icon", "")),
            align=str(item.get("align", "start")),
            width_units=float(item.get("width_units", 1.0)),
        )
        for index, item in enumerate(raw_actions)
    }
    end_cursor = right
    for action in reversed(tuple(
        item for item in actions.values() if item.align == "end"
    )):
        action_width = max(size, int(round(size * action.width_units)))
        regions[action.key] = (
            end_cursor - action_width, y0 + margin, action_width, size,
        )
        end_cursor -= action_width + gap
    start_cursor = x0 + margin
    start_actions = [
        item for item in actions.values() if item.align == "start"
    ]
    for index, action in enumerate(start_actions):
        requested_width = max(size, int(round(size * action.width_units)))
        remaining = max(1, len(start_actions) - index)
        remaining_gap = gap * remaining
        available_width = max(
            1, (end_cursor - start_cursor - remaining_gap) // remaining
        )
        action_width = min(requested_width, available_width)
        regions[action.key] = (
            start_cursor, y0 + margin, action_width, size,
        )
        start_cursor += action_width + gap
    labels = {
        "camera-label": panels["camera-panel"].label or "CAMERA",
        "work-label": panels["work-panel"].label or "WORK",
        "asset-browser-label": panels["asset-browser"].label or "ASSETS / JOBS",
        "status-text": panels["status-text"].label or "STATUS",
    }
    roles = {
        name: str((panel.payload or {}).get("role", "panel"))
        for name, panel in panels.items()
    }
    roles.update({key: "control_grid" for key in tool_rows})
    contexts = {
        name: OpenGLContextRequest()
        for name, panel in panels.items()
        if str((panel.payload or {}).get("context", "")) == "opengl"
    }
    geometry = {
        manifest.name: _panel_geometry(manifest, default_thickness=0.001),
        **{
            name: _panel_geometry(panel)
            for name, panel in panels.items()
        },
    }
    root_payload = manifest.payload if isinstance(manifest.payload, dict) else {}
    default_panel_primitive = root_payload.get("panel_primitive", {})
    panel_primitives = {
        manifest.name: layout_panel_primitive_from_mapping(
            default_panel_primitive, primitive_id=manifest.name
        ),
        **{
            name: layout_panel_primitive_from_mapping(
                (
                    panel.payload.get("panel_primitive", default_panel_primitive)
                    if isinstance(panel.payload, dict)
                    else default_panel_primitive
                ),
                primitive_id=name,
            )
            for name, panel in panels.items()
        },
    }
    default_control_primitive = root_payload.get(
        "control_primitive", default_panel_primitive
    )
    raw_actions_by_key = {
        str(item.get("key", index)): item
        for index, item in enumerate(raw_actions)
    }
    action_primitives = {
        key: layout_panel_primitive_from_mapping(
            raw_actions_by_key[key].get(
                "panel_primitive", default_control_primitive
            ),
            primitive_id=key,
        )
        for key in actions
    }
    return ProgramUILayout(
        manifest,
        width,
        height,
        dict(regions),
        labels,
        roles,
        contexts,
        geometry,
        panel_primitives,
        actions,
        action_primitives,
        layout_render_pipeline(manifest),
    )

__all__ = [
    "Rect",
    "ControlLayoutElement",
    "ControlLayoutDesign",
    "layout_control_panel",
    "photography_control_manifest",
    "PanelGeometryKind",
    "PanelGeometrySpec",
    "ProgramActionSpec",
    "LayoutRenderTier",
    "LayoutRenderPipeline",
    "layout_render_pipeline",
    "OpenGLContextRequest",
    "ProgramUILayout",
    "program_ui_manifest",
    "program_frame_metrics",
    "layout_program_ui",
]
