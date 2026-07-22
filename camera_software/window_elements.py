"""Semantic window elements shared by 2D layout, scene files, and crop reuse."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .control_layout import ProgramUILayout
from .layout_panel import layout_panel_composition_trace


WINDOW_ELEMENT_SCHEMA_VERSION = 1


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rect_tuple(value: Any) -> tuple[int, int, int, int]:
    if hasattr(value, "x"):
        return (int(value.x), int(value.y), int(value.w), int(value.h))
    result = tuple(int(item) for item in value)
    if len(result) != 4:
        raise ValueError("window element rectangle must contain x/y/width/height")
    return result


@dataclass(frozen=True)
class WindowElement:
    element_key: str
    parent_key: str
    sibling_order: int
    z_index: int
    kind: str
    state_subtype: str
    layout_rect_px: tuple[int, int, int, int]
    clip_rect_px: tuple[int, int, int, int]
    visible_rect_px: tuple[int, int, int, int]
    coordinate_space: str = "sensor_px"
    style_object_key: str = ""
    style_subtype_key: str = ""
    authored_text: str = ""
    icon_name: str = ""
    compositing_mode: str = "source_over"
    opacity: float = 1.0
    color_encoding: str = "display_srgb"
    alpha_mode: str = "straight"
    dependency_keys: tuple[str, ...] = ()
    content_signature: str = ""
    transform_chain: tuple[Mapping[str, Any], ...] = ()
    fallback_raster_path: str = ""
    representation: str = "parametric"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def mapping(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("layout_rect_px", "clip_rect_px", "visible_rect_px"):
            value[key] = list(value[key])
        value["dependency_keys"] = list(self.dependency_keys)
        value["transform_chain"] = [dict(item) for item in self.transform_chain]
        value["metadata"] = dict(self.metadata)
        return value


@dataclass(frozen=True)
class WindowElementSceneManifest:
    source_layout: str
    root_rect_px: tuple[int, int, int, int]
    elements: tuple[WindowElement, ...]
    schema_version: int = WINDOW_ELEMENT_SCHEMA_VERSION

    @property
    def content_signature(self) -> str:
        return _canonical_hash([
            element.mapping() for element in self.elements
        ])

    def mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_kind": "window_element_scene",
            "source_layout": self.source_layout,
            "root_rect_px": list(self.root_rect_px),
            "content_signature": self.content_signature,
            "elements": [element.mapping() for element in self.elements],
        }

    def save(self, path: str) -> str:
        final = os.path.abspath(path)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        temporary = final + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.mapping(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, final)
        return final


def _intersect(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    return (left, top, max(0, right - left), max(0, bottom - top))


def build_window_element_scene_manifest(
    layout: ProgramUILayout,
    *,
    widgets: Mapping[str, Any] | None = None,
    widget_display_origin_sensor_px: tuple[int, int] = (0, 0),
) -> WindowElementSceneManifest:
    """Export retained program and scroll-list semantics before rasterization."""

    widgets = dict(widgets or {})
    root_key = str(layout.manifest.name)
    root_rect = tuple(layout.regions[root_key])
    elements: list[WindowElement] = []

    def append(
        key: str,
        parent: str,
        order: int,
        z: int,
        kind: str,
        rect: Any,
        *,
        clip: Any | None = None,
        state: str = "default",
        text: str = "",
        icon: str = "",
        style_object_key: str = "",
        style_subtype_key: str = "",
        dependencies: Iterable[str] = (),
        transform_chain: Iterable[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        final_rect = _rect_tuple(rect)
        final_clip = root_rect if clip is None else _rect_tuple(clip)
        content = {
            "key": key, "kind": kind, "state": state,
            "rect": final_rect, "text": text, "icon": icon,
            "style": (style_object_key, style_subtype_key),
            "metadata": dict(metadata or {}),
        }
        elements.append(WindowElement(
            element_key=str(key), parent_key=str(parent),
            sibling_order=int(order), z_index=int(z), kind=str(kind),
            state_subtype=str(state), layout_rect_px=final_rect,
            clip_rect_px=final_clip,
            visible_rect_px=_intersect(final_rect, final_clip),
            style_object_key=str(style_object_key),
            style_subtype_key=str(style_subtype_key), authored_text=str(text),
            icon_name=str(icon), dependency_keys=tuple(map(str, dependencies)),
            content_signature=_canonical_hash(content),
            transform_chain=tuple(dict(item) for item in transform_chain),
            metadata=dict(metadata or {}),
        ))

    append(root_key, "", 0, 0, "window", root_rect, clip=root_rect,
           text=layout.manifest.label or root_key)
    primitives = {**layout.panel_primitives, **layout.action_primitives}
    action_keys = set(layout.actions)
    for order, (owner_id, rect) in enumerate(layout.regions.items(), 1):
        if owner_id == root_key:
            continue
        role = str(layout.roles.get(owner_id, ""))
        action = layout.actions.get(owner_id)
        kind = (
            "window.button" if owner_id in action_keys else
            "window.text_run" if owner_id in layout.authored_text else
            "window.image_host" if role in {"viewport", "widget_host"} else
            "window.panel"
        )
        primitive = primitives.get(owner_id)
        requests = () if primitive is None else primitive.object_requests
        style_object = requests[0].object_key if requests else ""
        style_subtype = requests[0].subtype_key if requests else ""
        text = str(layout.authored_text.get(owner_id, ""))
        append(
            owner_id, root_key, order, 10 + order, kind, rect,
            text=text or ("" if action is None else action.label),
            icon="" if action is None else action.icon,
            style_object_key=style_object,
            style_subtype_key=style_subtype,
            metadata={"role": role, "layout_owner": owner_id},
        )
        if primitive is not None:
            trace = layout_panel_composition_trace(
                primitive, int(rect[2]), int(rect[3])
            )
            for patch_order, patch in enumerate(trace.patches):
                local = patch.target_rect_px
                patch_rect = (
                    int(rect[0]) + local[0], int(rect[1]) + local[1],
                    local[2], local[3],
                )
                append(
                    f"{owner_id}/patch/{patch.role.value}", owner_id,
                    patch_order, 11 + order, "window.panel_patch", patch_rect,
                    clip=rect, style_object_key=patch.object_key,
                    style_subtype_key=patch.subtype_key,
                    dependencies=(patch.object_key, patch.subtype_key),
                    metadata=patch.mapping(),
                )

    origin_x, origin_y = map(int, widget_display_origin_sensor_px)
    for host_id, widget_reference in widgets.items():
        if host_id not in layout.regions:
            continue
        widget = getattr(widget_reference, "widget", widget_reference)
        host_rect = tuple(layout.regions[host_id])
        viewport = getattr(widget, "_viewport_rect", None)
        if viewport is None or int(getattr(viewport, "w", 0)) <= 0:
            continue

        def sensor_rect(display_rect: Any) -> tuple[int, int, int, int]:
            value = _rect_tuple(display_rect)
            return (value[0] + origin_x, value[1] + origin_y, value[2], value[3])

        viewport_rect = sensor_rect(viewport)
        list_key = f"{host_id}/scroll-list"
        append(
            list_key, host_id, 0, 100, "window.list_body", viewport_rect,
            clip=host_rect, state="scrolled" if widget.scroll_y else "default",
            text=str(widget.title),
            transform_chain=({
                "from": "widget_window_px", "to": "sensor_px",
                "translate_px": [origin_x, origin_y],
            },),
            metadata={
                "scroll_y": int(widget.scroll_y),
                "content_height_px": int(getattr(widget, "_content_h", 0)),
            },
        )
        headers = dict(getattr(widget, "_header_rects", {}) or {})
        subpanels = tuple(widget.subpanels)
        for row_order, spec in enumerate(subpanels):
            header = headers.get(spec.key)
            if header is None:
                continue
            header_rect = sensor_rect(header)
            state = (
                "selected" if spec.key == widget.selected_key else
                "disabled" if not spec.enabled else
                "expanded" if spec.expanded else "collapsed"
            )
            row_key = f"{list_key}/row/{spec.key}"
            append(
                row_key, list_key, row_order, 101 + row_order,
                "window.list_row", header_rect, clip=viewport_rect,
                state=state, text=str(spec.title),
                metadata={
                    "summary_lines": list(spec.summary_lines),
                    "accent_rgb": list(spec.accent_rgb),
                },
            )
            if spec.expanded and spec.summary_lines:
                later_headers = [
                    sensor_rect(headers[later.key])
                    for later in subpanels[row_order + 1:]
                    if later.key in headers
                ]
                body_bottom = min(
                    [item[1] for item in later_headers]
                    or [viewport_rect[1] + viewport_rect[3]]
                )
                body_top = header_rect[1] + header_rect[3]
                if body_bottom > body_top:
                    append(
                        f"{row_key}/body-text", row_key, 0,
                        102 + row_order, "window.list_body_text",
                        (
                            header_rect[0] + 6, body_top,
                            max(1, header_rect[2] - 12),
                            body_bottom - body_top,
                        ),
                        clip=viewport_rect, state="expanded",
                        text="\n".join(map(str, spec.summary_lines)),
                    )
        for control_order, record in enumerate(
            getattr(widget.control_table, "sorted_records", lambda: [])()
        ):
            control_rect = sensor_rect(record.rect)
            path = tuple(map(str, record.path))
            append(
                f"{list_key}/control/{'/'.join(path)}", list_key,
                control_order, 1000 + control_order,
                f"window.{record.kind}", control_rect, clip=viewport_rect,
                state="interactive", dependencies=path,
                metadata={"control_path": list(path)},
            )
        progress = dict(
            getattr(widget_reference, "progress_metrics", {}) or {}
        )
        if progress and host_id == "camera-panel":
            pie_size = max(12, min(34, host_rect[2] // 5, host_rect[3] // 5))
            for pie_order, (name, value) in enumerate((
                ("convergence", float(progress.get("convergence", 0.0))),
                ("priority", float(progress.get("priority_share", 0.0))),
            )):
                append(
                    f"{host_id}/progress/{name}", host_id, pie_order,
                    2000 + pie_order, "window.progress_pie",
                    (
                        host_rect[0] + 6 + pie_order * (pie_size + 8),
                        host_rect[1] + 6, pie_size, pie_size,
                    ),
                    clip=host_rect,
                    state=(
                        "working" if progress.get("working") else
                        str(progress.get("status", "waiting"))
                    ),
                    text=f"{name.upper()} {100.0 * value:.1f}%",
                    metadata={
                        "value": max(0.0, min(1.0, value)),
                        "shape": "pie",
                        "convergence_velocity_per_pass": float(progress.get(
                            "convergence_velocity_per_pass", 0.0
                        )),
                    },
                )
    return WindowElementSceneManifest(root_key, root_rect, tuple(elements))


@dataclass(frozen=True)
class HarvestedWindowPlate:
    plate_key: str
    element_key: str
    element_content_signature: str
    condition_signature: str
    rect_px: tuple[int, int, int, int]
    display_path: str = ""
    linear_path: str = ""
    created_at_s: float = 0.0


class WindowElementHarvestCache:
    """Exact-condition cache for crops from a holistic camera exposure."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        self.records: dict[str, HarvestedWindowPlate] = {}
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for item in payload.get("records", ()):
                record = HarvestedWindowPlate(**item)
                self.records[record.plate_key] = record

    @staticmethod
    def condition_signature(
        *, camera_key: str, lighting_key: str, material_key: str,
        reflection_boundary_key: str, color_transform_key: str,
    ) -> str:
        return _canonical_hash({
            "camera": camera_key, "lighting": lighting_key,
            "material": material_key,
            "reflection_boundary": reflection_boundary_key,
            "color_transform": color_transform_key,
        })

    def find(
        self, element: WindowElement, condition_signature: str
    ) -> HarvestedWindowPlate | None:
        key = _canonical_hash((
            element.element_key, element.content_signature,
            str(condition_signature), element.visible_rect_px,
        ))
        record = self.records.get(key)
        if record is None:
            return None
        if not (record.display_path or record.linear_path):
            return None
        return record

    def harvest(
        self,
        manifest: WindowElementSceneManifest,
        *,
        condition_signature: str,
        display_image_path: str = "",
        linear_image_path: str = "",
        output_directory: str,
        kinds: Iterable[str] = (
            "window", "window.panel", "window.image_host", "window.list_body"
        ),
    ) -> tuple[HarvestedWindowPlate, ...]:
        from PIL import Image
        import numpy as np

        display = Image.open(display_image_path).convert("RGBA") if display_image_path else None
        linear = np.load(linear_image_path, allow_pickle=False) if linear_image_path else None
        root_x, root_y, _root_w, _root_h = manifest.root_rect_px
        if display is not None and display.size != (_root_w, _root_h):
            raise ValueError(
                "holistic display raster must exactly match manifest root size"
            )
        if linear is not None and tuple(linear.shape[:2]) != (_root_h, _root_w):
            raise ValueError(
                "holistic linear raster must exactly match manifest root size"
            )
        allowed = set(map(str, kinds))
        produced = []
        os.makedirs(output_directory, exist_ok=True)
        for element in manifest.elements:
            if element.kind not in allowed or element.visible_rect_px[2] <= 0 or element.visible_rect_px[3] <= 0:
                continue
            x, y, width, height = element.visible_rect_px
            local = (x - root_x, y - root_y, width, height)
            plate_key = _canonical_hash((
                element.element_key, element.content_signature,
                condition_signature, element.visible_rect_px,
            ))
            stem = plate_key[:24]
            display_path = ""
            linear_path = ""
            if display is not None:
                display_path = os.path.abspath(os.path.join(output_directory, stem + ".png"))
                display.crop((local[0], local[1], local[0] + width, local[1] + height)).save(display_path)
            if linear is not None:
                linear_path = os.path.abspath(os.path.join(output_directory, stem + "_linear.npy"))
                np.save(
                    linear_path,
                    linear[
                        local[1]:local[1] + height,
                        local[0]:local[0] + width,
                    ],
                    allow_pickle=False,
                )
            record = HarvestedWindowPlate(
                plate_key, element.element_key, element.content_signature,
                condition_signature, element.visible_rect_px,
                display_path, linear_path, time.time(),
            )
            self.records[plate_key] = record
            produced.append(record)
        self._save()
        return tuple(produced)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temporary = self.path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "records": [asdict(self.records[key]) for key in sorted(self.records)],
            }, handle, indent=2, sort_keys=True)
        os.replace(temporary, self.path)


__all__ = [
    "WINDOW_ELEMENT_SCHEMA_VERSION", "WindowElement",
    "WindowElementSceneManifest", "build_window_element_scene_manifest",
    "HarvestedWindowPlate", "WindowElementHarvestCache",
]
