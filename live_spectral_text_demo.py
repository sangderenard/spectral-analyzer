"""Editable paragraph UI backed by asynchronous spectral thick-lens renders."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from camera_software import (
    ColorScienceProfile,
    ExposureProgressBroker,
    ExposureProgressEvent,
    ExposureProgressKind,
    ProgressSink,
    SensorRegion,
    SensorPixelSlice,
    DisplayPrimitiveKind,
    DisplayProductKind,
    FixedSceneCamera,
    WorldPlacement,
    DisplayPrimitive,
    DisplayProductRequest,
    DisplayObjectSpec,
    DisplaySceneSpec,
    DisplaySceneInventory,
    BevelProfile,
    BevelRegion,
    LayoutRectangle,
    LayoutSeam,
    process_linear_sensor_image,
    save_linear_sensor_image,
)


DEFAULT_TEXT = "Actual light takes the long way home through glass."
JOB_ID = "live_paragraph"
DEFAULT_DISPLAY_WIDTH = 960
DEFAULT_DISPLAY_HEIGHT = 600
DEFAULT_SENSOR_SWEEPS = 4
DEFAULT_TARGETED_FRACTION = 0.75
SENSOR_CROP_SCALE = 5
DEFAULT_EXTRUSION_DEPTH_RATIO = 0.08
PROGRAM_SCENE_ID = "spectral-program-scene"
PROGRAM_STATIC_GEOMETRY_REVISION = 7
# The native thick-camera sensor is square. Its exact 200x153 product readback
# restores the UI aspect afterward, so the camera-facing world layout must use
# one square physical frame or it is vertically compressed twice.
PROGRAM_SCENE_EXTENT_M = 0.375


def _program_frame_metrics(
    work_width: int, work_height: int
) -> dict[str, int]:
    """Derive the full UI frame around fixed-size preview panes."""

    preview_width = max(1, int(work_width))
    preview_height = max(1, int(work_height))
    frame_width = 2 * preview_width
    scale = min(1.0, max(0.4, float(frame_width) / 480.0))
    control_h = max(12, int(round(24.0 * scale)))
    label_h = max(10, int(round(20.0 * scale)))
    label_clearance_h = max(1, label_h // 2)
    editor_h = max(24, int(round(preview_height * 0.35)))
    status_h = max(12, int(round(18.0 * scale)))
    frame_height = (
        control_h + label_h + label_clearance_h
        + preview_height + editor_h + status_h
    )
    return {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "control_h": control_h,
        "label_h": label_h,
        "label_clearance_h": label_clearance_h,
        "editor_h": editor_h,
        "status_h": status_h,
    }


def render_contract_summary(
    display_width: int,
    display_height: int,
    sensor_sweeps: int = DEFAULT_SENSOR_SWEEPS,
    *,
    scene_width: int | None = None,
    scene_height: int | None = None,
) -> str:
    """Describe full UI product separately from its maximum scan chunk."""
    width = int(display_width)
    height = int(display_height)
    full_width = int(scene_width if scene_width is not None else width)
    full_height = int(scene_height if scene_height is not None else height)
    native_resolution = max(full_width, full_height)
    return (
        f"ui_scene={full_width}x{full_height} "
        f"scan_region<={width}x{height} "
        f"native_sensor={native_resolution}x{native_resolution} "
        f"composition_frame={full_width * SENSOR_CROP_SCALE}x"
        f"{full_height * SENSOR_CROP_SCALE} "
        f"authored_sensor_sweeps={int(sensor_sweeps)} live_exposure=continuous; "
        "requested regions and pixel slices accumulate at their scene coordinates"
    )


def build_paragraph_order(
    text: str,
    *,
    display_width: int = DEFAULT_DISPLAY_WIDTH,
    display_height: int = DEFAULT_DISPLAY_HEIGHT,
    sensor_sweeps: int = DEFAULT_SENSOR_SWEEPS,
) -> dict[str, Any]:
    """Build a paragraph order sampled at the digital display's exact raster."""
    token = str(text).strip()
    if not token:
        token = " "
    output_width = int(display_width)
    output_height = int(display_height)
    if output_width <= 0 or output_height <= 0:
        raise ValueError("display dimensions must be positive")
    if int(sensor_sweeps) <= 0:
        raise ValueError("sensor_sweeps must be positive")
    full_width = output_width * SENSOR_CROP_SCALE
    full_height = output_height * SENSOR_CROP_SCALE
    plane_normal = [0.8017837257, -0.5345224838, 0.2672612419]
    order = {
        "schema_version": 1,
        "defaults": {
            "image": {
                "width": full_width,
                "height": full_height,
                "region": {
                    "x": 2 * output_width,
                    "y": 2 * output_height,
                    "width": output_width,
                    "height": output_height,
                },
            },
            "camera": {
                "focal_mm": 35.0,
                "aperture_mm": 25.0,
                "position_m": [3.0, -2.0, 1.0],
                "target_m": [0.0, 0.0, 0.0],
                # Filled from the formatter-resolved glyph depth below. This
                # changes focus only; plane and camera pose remain authored.
                "focus_target_m": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
            },
            "exposure": {
                "time_s": 0.016666666666666666,
                "iso": 100.0,
                "sensor_sweeps": int(sensor_sweeps),
                "t5_pair_budget": 20000000,
            },
            "flash": {"intensity_scale": 1.0},
            "font": {
                "family": "DejaVu Sans",
                "weight": "bold",
                "style": "normal",
            },
            "planes": [{
                "id": "background",
                "center_m": [0.0, 0.0, 0.0],
                "normal": plane_normal,
                "up": [-0.2223747950, 0.1482498633, 0.9636241117],
                "size_m": [1.4, 1.0],
                "thickness_m": 0.018,
                "material": "quiet_background",
            }],
            "materials": {
                "quiet_background": {
                    "albedo_rgb": [0.018, 0.021, 0.022],
                    "reflectivity": 0.065,
                    "diffusion": 0.98,
                    "absorption": 0.92,
                    "roughness": 0.98,
                    "metallic": 0.0,
                },
                "text_surface": {
                    "albedo_rgb": [0.82, 0.76, 0.62],
                    "reflectivity": 0.58,
                    "diffusion": 0.72,
                    "absorption": 0.25,
                    "roughness": 0.3,
                    "metallic": 0.0,
                    "ior": 1.5,
                },
            },
            "geometry": {
                "embed_plane": "background",
                "height_m": 0.055,
                "line_height_m": 0.055,
                "line_spacing": 1.22,
                "text_box_m": [0.31, 0.18],
                "horizontal_align": "center",
                "vertical_align": "center",
                "extrusion_depth_ratio": DEFAULT_EXTRUSION_DEPTH_RATIO,
                "embed_fraction": 0.25,
                "offset_m": [0.0, 0.0],
                "profile": "straight",
                "outline_subdivisions": 4,
                "cap_grid": 64,
                "material": "text_surface",
            },
        },
        "jobs": [{"id": JOB_ID, "token": token}],
    }
    # Thickness is a property of the formatted glyphs, so resolve wrapping
    # before setting the front-face focus distance. No scene placement changes.
    from scene_orders import resolved_glyph_depth
    formatting_job = {
        "token": token,
        "font": order["defaults"]["font"],
        "geometry": order["defaults"]["geometry"],
    }
    exposed_text_depth_m = (
        resolved_glyph_depth(formatting_job)
        * (1.0 - float(order["defaults"]["geometry"]["embed_fraction"]))
    )
    order["defaults"]["camera"]["focus_target_m"] = [
        component * exposed_text_depth_m for component in plane_normal
    ]
    return order


def build_program_display_scene(
    text: str,
    *,
    display_width: int = DEFAULT_DISPLAY_WIDTH,
    display_height: int = DEFAULT_DISPLAY_HEIGHT,
    revision: int = 1,
    extra_objects: tuple[DisplayObjectSpec, ...] = (),
) -> DisplaySceneSpec:
    """Author retained program objects in one fixed-camera perspective scene."""

    width = int(display_width)
    height = int(display_height)
    if width <= 0 or height <= 0:
        raise ValueError("display dimensions must be positive")
    plane_normal = (0.8017837257, -0.5345224838, 0.2672612419)
    plane_up = (-0.2223747950, 0.1482498633, 0.9636241117)
    main = DisplayObjectSpec(
        object_id="main-text",
        primitive=DisplayPrimitive(DisplayPrimitiveKind.TEXT, content=str(text)),
        placement=WorldPlacement(
            center_m=(0.0, 0.0, 0.0),
            normal=plane_normal,
            up=plane_up,
            size_m=(1.4, 1.0),
            thickness_m=0.018,
        ),
        products=(
            DisplayProductRequest(
                DisplayProductKind.IMAGE,
                program_display_region(width, height),
            ),
        ),
        revision=max(1, int(revision)),
    )
    camera = FixedSceneCamera(
        position_m=(3.0, -2.0, 1.0),
        target_m=(0.0, 0.0, 0.0),
        focus_target_m=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        focal_mm=35.0,
        aperture_mm=25.0,
    )
    return DisplaySceneSpec(
        scene_id=PROGRAM_SCENE_ID,
        camera=camera,
        sensor_width=width * SENSOR_CROP_SCALE,
        sensor_height=height * SENSOR_CROP_SCALE,
        objects=(main, *tuple(extra_objects)),
        revision=max(1, int(revision)),
    )


def management_display_object(
    object_id: str,
    kind: DisplayPrimitiveKind,
    *,
    center_m: tuple[float, float, float],
    size_m: tuple[float, float],
    icon_name: str = "",
    label: str = "",
    sensor_region: SensorRegion | None = None,
    revision: int = 1,
) -> DisplayObjectSpec:
    """Create box/icon geometry that shares the program scene's fixed camera."""

    if kind not in (DisplayPrimitiveKind.BOX, DisplayPrimitiveKind.ICON):
        raise ValueError("management object kind must be box or icon")
    return DisplayObjectSpec(
        object_id=object_id,
        primitive=DisplayPrimitive(kind, icon_name=icon_name, label=label),
        placement=WorldPlacement(
            center_m=center_m,
            normal=(0.8017837257, -0.5345224838, 0.2672612419),
            up=(-0.2223747950, 0.1482498633, 0.9636241117),
            size_m=size_m,
            thickness_m=0.012,
        ),
        products=(
            DisplayProductRequest(DisplayProductKind.IMAGE, sensor_region),
        ),
        revision=max(1, int(revision)),
    )


def program_display_region(
    display_width: int, display_height: int,
) -> SensorRegion:
    """The photographed UI rectangle inside the larger physical sensor."""

    width, height = int(display_width), int(display_height)
    return SensorRegion(2 * width, 2 * height, width, height)


def program_ui_sensor_regions(
    display_width: int,
    display_height: int,
    *,
    work_width: int | None = None,
    work_height: int | None = None,
) -> dict[str, SensorRegion]:
    """Stable sensor-space layout for UI products photographed together."""

    crop = program_display_region(display_width, display_height)

    if work_width is None:
        work_width = max(1, crop.width // 2)
    if work_height is None:
        work_height = max(1, crop.height // 2)
    metrics = _program_frame_metrics(work_width, work_height)
    left_width = min(crop.width, metrics["preview_width"])
    right_width = min(crop.width - left_width, metrics["preview_width"])
    control_h = min(metrics["control_h"], max(1, crop.height - 1))
    label_h = min(metrics["label_h"], max(1, crop.height - control_h - 1))
    panel_h = min(
        metrics["preview_height"],
        max(
            1,
            crop.height - control_h - label_h
            - metrics["label_clearance_h"] - 1,
        ),
    )
    remaining_h = max(
        1,
        crop.height - control_h - label_h
        - metrics["label_clearance_h"] - panel_h,
    )
    status_h = min(metrics["status_h"], max(1, remaining_h))
    editor_h = max(1, remaining_h - status_h)
    label_margin = min(4, max(0, (left_width - 1) // 4))
    control_margin = min(2, max(0, (crop.width - 1) // 24))
    available_control_w = max(1, crop.width - 2 * control_margin)
    control_gap = max(0, min(3, available_control_w // 20))
    control_size = max(
        1,
        min(
            14,
            max(1, control_h - 2 * control_margin),
            max(1, (available_control_w - 2 * control_gap) // 3),
        ),
    )
    control_right = crop.x + crop.width - control_margin
    close_x = control_right - control_size
    maximize_x = close_x - control_gap - control_size
    minimize_x = maximize_x - control_gap - control_size
    label_y = crop.y + control_h
    panel_y = label_y + label_h + metrics["label_clearance_h"]
    editor_y = panel_y + panel_h
    status_y = editor_y + editor_h
    return {
        "program-backdrop": SensorRegion(
            crop.x, crop.y, crop.width, crop.height
        ),
        "camera-panel": SensorRegion(crop.x, panel_y, left_width, panel_h),
        "work-panel": SensorRegion(
            crop.x + left_width, panel_y, right_width, panel_h
        ),
        "camera-label": SensorRegion(
            crop.x + label_margin,
            label_y,
            max(1, left_width - 2 * label_margin),
            label_h,
        ),
        "work-label": SensorRegion(
            crop.x + left_width + label_margin,
            label_y,
            max(1, right_width - 2 * label_margin),
            label_h,
        ),
        "editor-text": SensorRegion(
            crop.x, editor_y, crop.width, editor_h
        ),
        "status-text": SensorRegion(
            crop.x,
            status_y,
            crop.width,
            status_h,
        ),
        "window-minimize": SensorRegion(
            minimize_x, crop.y + control_margin, control_size, control_size
        ),
        "window-maximize": SensorRegion(
            maximize_x, crop.y + control_margin, control_size, control_size
        ),
        "window-close": SensorRegion(
            close_x, crop.y + control_margin, control_size, control_size
        ),
    }


def _placement_for_sensor_region(
    sensor_region: SensorRegion,
    photographed_region: SensorRegion,
    *,
    thickness_m: float,
    front_offset_m: float = 0.0,
) -> WorldPlacement:
    """Place one layout rectangle inside the single camera-framed UI plane."""

    normal = np.asarray((0.8017837257, -0.5345224838, 0.2672612419), np.float64)
    up = np.asarray((-0.2223747950, 0.1482498633, 0.9636241117), np.float64)
    right = np.cross(up, normal)
    right /= np.linalg.norm(right)
    center_u = (
        (sensor_region.x + 0.5 * sensor_region.width - photographed_region.x)
        / photographed_region.width
        - 0.5
    )
    center_v = (
        (sensor_region.y + 0.5 * sensor_region.height - photographed_region.y)
        / photographed_region.height
        - 0.5
    )
    base_width_m = PROGRAM_SCENE_EXTENT_M
    base_height_m = PROGRAM_SCENE_EXTENT_M
    center = (
        right * (center_u * base_width_m)
        - up * (center_v * base_height_m)
        + normal * float(front_offset_m)
    )
    return WorldPlacement(
        center_m=tuple(map(float, center)),
        normal=tuple(map(float, normal)),
        up=tuple(map(float, up)),
        size_m=(
            base_width_m * sensor_region.width / photographed_region.width,
            base_height_m * sensor_region.height / photographed_region.height,
        ),
        thickness_m=thickness_m,
    )


def _region_pixel_slice(
    region: SensorRegion, sensor_width: int, sensor_height: int
) -> SensorPixelSlice:
    indices = tuple(
        y * sensor_width + x
        for y in range(region.y, region.y + region.height)
        for x in range(region.x, region.x + region.width)
    )
    return SensorPixelSlice(sensor_width, sensor_height, indices)


def build_self_rendering_program_scene(
    text: str,
    *,
    display_width: int = DEFAULT_DISPLAY_WIDTH,
    display_height: int = DEFAULT_DISPLAY_HEIGHT,
    work_width: int | None = None,
    work_height: int | None = None,
    status_text: str = "SPECTRAL EXPOSURE ACTIVE",
    revision: int = 1,
) -> DisplaySceneSpec:
    """Build editor and window-control geometry for one physical photograph."""

    regions = program_ui_sensor_regions(
        display_width,
        display_height,
        work_width=work_width,
        work_height=work_height,
    )
    crop = program_display_region(display_width, display_height)
    base = build_program_display_scene(
        text,
        display_width=display_width,
        display_height=display_height,
        revision=revision,
    )
    editor_region = regions["editor-text"]
    backdrop_region = regions["program-backdrop"]
    backdrop = DisplayObjectSpec(
        object_id="program-backdrop",
        primitive=DisplayPrimitive(DisplayPrimitiveKind.BOX),
        placement=_placement_for_sensor_region(
            backdrop_region,
            crop,
            thickness_m=0.001,
            front_offset_m=-0.001,
        ),
        products=(
            DisplayProductRequest(DisplayProductKind.IMAGE, backdrop_region),
        ),
        layout_rectangle=LayoutRectangle(
            backdrop_region,
            BevelRegion(0.0, 0.0, BevelProfile.CHAMFER),
            (LayoutSeam.FLUSH,) * 4,
        ),
        revision=PROGRAM_STATIC_GEOMETRY_REVISION,
    )
    editor = replace(
        base.objects[0],
        object_id="editor-text",
        horizontal_align="left",
        vertical_align="top",
        placement=_placement_for_sensor_region(
            editor_region, crop, thickness_m=0.002
        ),
        products=(
            DisplayProductRequest(
                DisplayProductKind.IMAGE,
                editor_region,
                _region_pixel_slice(
                    editor_region, base.sensor_width, base.sensor_height
                ),
            ),
        ),
        layout_rectangle=LayoutRectangle(
            editor_region,
            BevelRegion(0.0015, 0.0005, BevelProfile.CHAMFER),
            (LayoutSeam.BEVEL,) * 4,
        ),
    )
    authored_text_objects = []
    for object_id, content in (
        ("camera-label", "CAMERA"),
        ("work-label", "WORK VALUE"),
        ("status-text", str(status_text)),
    ):
        if object_id not in regions:
            continue
        sensor_region = regions[object_id]
        dynamic = object_id == "status-text"
        authored_text_objects.append(DisplayObjectSpec(
            object_id=object_id,
            primitive=DisplayPrimitive(
                DisplayPrimitiveKind.TEXT, content=content
            ),
            placement=_placement_for_sensor_region(
                sensor_region,
                crop,
                thickness_m=0.0008,
                front_offset_m=0.0006,
            ),
            products=(
                DisplayProductRequest(
                    DisplayProductKind.IMAGE,
                    sensor_region,
                    (
                        _region_pixel_slice(
                            sensor_region,
                            base.sensor_width,
                            base.sensor_height,
                        )
                        if dynamic else None
                    ),
                ),
            ),
            material="text_surface",
            surface_material="quiet_background",
            horizontal_align=(
                "left" if object_id == "status-text" else "center"
            ),
            vertical_align=(
                "bottom" if object_id == "status-text" else "center"
            ),
            revision=(
                max(1, int(revision))
                if dynamic else PROGRAM_STATIC_GEOMETRY_REVISION
            ),
        ))
    controls = []
    panels = []
    for object_id in ("camera-panel", "work-panel"):
        if object_id not in regions:
            continue
        sensor_region = regions[object_id]
        panels.append(DisplayObjectSpec(
            object_id=object_id,
            primitive=DisplayPrimitive(DisplayPrimitiveKind.BOX),
            placement=_placement_for_sensor_region(
                sensor_region, crop, thickness_m=0.002
            ),
            products=(
                DisplayProductRequest(DisplayProductKind.IMAGE, sensor_region),
            ),
            layout_rectangle=LayoutRectangle(
                sensor_region,
                BevelRegion(0.0015, 0.0005, BevelProfile.CHAMFER),
                (LayoutSeam.BEVEL,) * 4,
            ),
            revision=PROGRAM_STATIC_GEOMETRY_REVISION,
        ))
    for object_id, icon_name in (
        ("window-minimize", "minimize"),
        ("window-maximize", "maximize"),
        ("window-close", "close"),
    ):
        sensor_region = regions[object_id]
        controls.append(DisplayObjectSpec(
            object_id=object_id,
            primitive=DisplayPrimitive(
                DisplayPrimitiveKind.ICON, icon_name=icon_name
            ),
            placement=_placement_for_sensor_region(
                sensor_region,
                crop,
                thickness_m=0.0008,
                front_offset_m=0.0006,
            ),
            products=(
                DisplayProductRequest(DisplayProductKind.IMAGE, sensor_region),
            ),
            layout_rectangle=LayoutRectangle(
                sensor_region,
                BevelRegion(0.001, 0.0004, BevelProfile.CHAMFER),
                (LayoutSeam.BEVEL,) * 4,
            ),
            # Controls retain their exposure when only editor content changes.
            revision=PROGRAM_STATIC_GEOMETRY_REVISION,
        ))
    return replace(
        base,
        objects=(backdrop, editor, *panels, *authored_text_objects, *controls),
    )


def _static_scene_is_reusable(
    previous: DisplaySceneSpec, current: DisplaySceneSpec
) -> bool:
    """Require an exact static-geometry match before restoring sensor evidence."""

    if (
        previous.camera != current.camera
        or previous.sensor_width != current.sensor_width
        or previous.sensor_height != current.sensor_height
    ):
        return False
    previous_static = {
        item.object_id: item
        for item in previous.objects
        if item.object_id not in {"editor-text", "status-text"}
    }
    current_static = {
        item.object_id: item
        for item in current.objects
        if item.object_id not in {"editor-text", "status-text"}
    }
    return previous_static == current_static


def build_ui_next_scan_control(
    scene: DisplaySceneSpec,
    photographed_region: SensorRegion,
    *,
    sequence: int,
    targeted_fraction: float,
    scan_width: int | None = None,
    scan_height: int | None = None,
    delta_restore: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert retained product regions to local UV requests for one GPU scan."""

    requests = []
    pixel_requests = []
    for spec in scene.objects:
        if not spec.enabled:
            continue
        for product in spec.products:
            region = product.sensor_region
            if region is None:
                continue
            x0 = max(region.x, photographed_region.x)
            y0 = max(region.y, photographed_region.y)
            x1 = min(
                region.x + region.width,
                photographed_region.x + photographed_region.width,
            )
            y1 = min(
                region.y + region.height,
                photographed_region.y + photographed_region.height,
            )
            if x1 <= x0 or y1 <= y0:
                continue
            chunk_width = max(1, int(scan_width or (x1 - x0)))
            chunk_height = max(1, int(scan_height or (y1 - y0)))
            for chunk_y in range(y0, y1, chunk_height):
                for chunk_x in range(x0, x1, chunk_width):
                    chunk_x1 = min(x1, chunk_x + chunk_width)
                    chunk_y1 = min(y1, chunk_y + chunk_height)
                    requests.append({
                        "uv_bounds": [
                            (chunk_x - photographed_region.x)
                            / photographed_region.width,
                            (chunk_y - photographed_region.y)
                            / photographed_region.height,
                            (chunk_x1 - photographed_region.x)
                            / photographed_region.width,
                            (chunk_y1 - photographed_region.y)
                            / photographed_region.height,
                        ],
                        "target_level": max(2, int(product.zoom_level)),
                        "work_value": (
                            1.25
                            if spec.primitive.kind is DisplayPrimitiveKind.ICON
                            else 1.0
                        ),
                        "object_id": spec.object_id,
                    })
            selected = product.sensor_pixel_slice
            if selected is not None:
                local_indices_by_chunk: dict[tuple[int, int], list[int]] = {}
                for index in selected.site_indices:
                    source_y, source_x = divmod(index, selected.width)
                    local_x = source_x - photographed_region.x
                    local_y = source_y - photographed_region.y
                    if (
                        0 <= local_x < photographed_region.width
                        and 0 <= local_y < photographed_region.height
                    ):
                        chunk_key = (
                            (source_x - x0) // chunk_width,
                            (source_y - y0) // chunk_height,
                        )
                        local_indices_by_chunk.setdefault(chunk_key, []).append(
                            local_y * photographed_region.width + local_x
                        )
                for local_indices in local_indices_by_chunk.values():
                    pixel_requests.append({
                        "width": photographed_region.width,
                        "height": photographed_region.height,
                        "site_indices": local_indices,
                        "target_level": max(2, int(product.zoom_level)),
                        "work_value": 1.0,
                        "object_id": spec.object_id,
                    })
    # NextSiteScan deliberately accepts only its transport-neutral fields.
    return {
        "sequence": max(0, int(sequence)),
        "targeted_fraction": float(targeted_fraction),
        "uv_requests": [
            {key: value for key, value in request.items() if key != "object_id"}
            for request in requests
        ],
        "pixel_slice_requests": [
            {key: value for key, value in request.items() if key != "object_id"}
            for request in pixel_requests
        ],
        "metadata": {
            "purpose": "self-rendered program UI products",
            "object_ids": [request["object_id"] for request in requests],
            "dirty_object_ids": [
                request["object_id"] for request in pixel_requests
            ],
            "delta_restore": dict(delta_restore or {}),
        },
    }


def _icon_stroke_planes(spec: DisplayObjectSpec) -> list[dict[str, Any]]:
    """Manufacture management icons as physical strips, never font glyphs."""

    placement = spec.placement
    normal = np.asarray(placement.normal, np.float64)
    normal /= np.linalg.norm(normal)
    up = np.asarray(placement.up, np.float64)
    up -= normal * np.dot(up, normal)
    up /= np.linalg.norm(up)
    right = np.cross(up, normal)
    right /= np.linalg.norm(right)
    center = (
        np.asarray(placement.center_m, np.float64)
        + normal * (0.5 * placement.thickness_m + 5.0e-4)
    )
    extent = min(placement.size_m)
    length = max(0.002, 0.48 * extent)
    stroke = max(0.001, 0.10 * extent)
    thickness = max(5.0e-4, 0.12 * placement.thickness_m)
    planes: list[dict[str, Any]] = []

    def add_stroke(
        suffix: str,
        stroke_center: np.ndarray,
        stroke_up: np.ndarray,
        width: float,
        height: float,
    ) -> None:
        planes.append({
            "id": f"display-icon-{spec.object_id}-{suffix}",
            "center_m": list(map(float, stroke_center)),
            "normal": list(map(float, normal)),
            "up": list(map(float, stroke_up)),
            "size_m": [float(width), float(height)],
            "thickness_m": float(thickness),
            "material": spec.material,
        })

    def add_line(
        suffix: str, start: np.ndarray, end: np.ndarray
    ) -> None:
        direction = end - start
        line_length = float(np.linalg.norm(direction))
        direction /= line_length
        stroke_up = np.cross(normal, direction)
        stroke_up /= np.linalg.norm(stroke_up)
        add_stroke(
            suffix, 0.5 * (start + end), stroke_up, line_length, stroke
        )

    icon_name = spec.primitive.icon_name
    if icon_name == "minimize":
        add_stroke("bar", center - up * (0.16 * length), up, length, stroke)
    elif icon_name == "maximize":
        offset = 0.5 * (length - stroke)
        add_stroke("top", center + up * offset, up, length, stroke)
        add_stroke("bottom", center - up * offset, up, length, stroke)
        add_stroke("left", center - right * offset, up, stroke, length)
        add_stroke("right", center + right * offset, up, stroke, length)
    elif icon_name == "close":
        diagonal = 0.5 * length
        add_line(
            "forward",
            center - (right + up) * diagonal,
            center + (right + up) * diagonal,
        )
        add_line(
            "backward",
            center - (right - up) * diagonal,
            center + (right - up) * diagonal,
        )
    elif icon_name == "menu":
        for suffix, offset in (("top", 0.28), ("middle", 0.0), ("bottom", -0.28)):
            add_stroke(
                suffix, center + up * (offset * length), up, length, stroke
            )
    elif icon_name == "pause":
        offset = 0.20 * length
        add_stroke("left", center - right * offset, up, stroke, length)
        add_stroke("right", center + right * offset, up, stroke, length)
    elif icon_name == "play":
        half = 0.5 * length
        nose = center + right * half
        top = center - right * (0.35 * length) + up * half
        bottom = center - right * (0.35 * length) - up * half
        add_line("upper", top, nose)
        add_line("lower", nose, bottom)
        add_line("back", bottom, top)
    elif icon_name == "settings":
        half = 0.5 * length
        add_line("horizontal", center - right * half, center + right * half)
        add_line("vertical", center - up * half, center + up * half)
        add_line(
            "forward",
            center - (right + up) * (0.35 * length),
            center + (right + up) * (0.35 * length),
        )
        add_line(
            "backward",
            center - (right - up) * (0.35 * length),
            center + (right - up) * (0.35 * length),
        )
    else:
        raise ValueError(f"unsupported physical management icon {icon_name!r}")
    return planes


def build_display_scene_order(
    scene: DisplaySceneSpec,
    *,
    sensor_sweeps: int = DEFAULT_SENSOR_SWEEPS,
) -> dict[str, Any]:
    """Compile a shared-camera display scene into the existing scene-order ABI."""

    if int(sensor_sweeps) <= 0:
        raise ValueError("sensor_sweeps must be positive")
    planes = []
    objects = []
    for spec in scene.objects:
        plane_id = f"display-surface-{spec.object_id}"
        placement = spec.placement
        planes.append({
            "id": plane_id,
            "center_m": list(placement.center_m),
            "normal": list(placement.normal),
            "up": list(placement.up),
            "size_m": list(placement.size_m),
            "thickness_m": float(placement.thickness_m),
            "material": spec.surface_material,
            **(
                {}
                if spec.layout_rectangle is None
                else {
                    "bevel_width_m": spec.layout_rectangle.bevel.width_m,
                    "bevel_depth_m": spec.layout_rectangle.bevel.depth_m,
                    "bevel_profile": spec.layout_rectangle.bevel.profile.value,
                    "bevel_seams": [
                        seam.value for seam in spec.layout_rectangle.seams
                    ],
                }
            ),
        })
        if spec.primitive.kind is DisplayPrimitiveKind.BOX:
            continue
        if spec.primitive.kind is DisplayPrimitiveKind.ICON:
            planes.extend(_icon_stroke_planes(spec))
            continue
        text_height = max(
            0.0015,
            min(
                placement.size_m[1] * 0.90,
                placement.size_m[0] * 0.30,
            ),
        )
        objects.append({
            "id": spec.object_id,
            "token": spec.primitive.authored_text(),
            "embed_plane": plane_id,
            "enabled": bool(spec.enabled),
            "font": {
                "family": spec.font_family,
                "weight": spec.font_weight,
                "style": spec.font_style,
            },
            "geometry": {
                "height_m": text_height,
                "line_height_m": text_height,
                "line_spacing": 1.05,
                "horizontal_align": spec.horizontal_align,
                "vertical_align": spec.vertical_align,
                "text_box_m": [
                    placement.size_m[0] * 0.94,
                    placement.size_m[1] * 0.90,
                ],
                "embed_fraction": 0.25,
                "extrusion_depth_ratio": DEFAULT_EXTRUSION_DEPTH_RATIO,
                "offset_m": [0.0, 0.0],
                "profile": "straight",
                "outline_subdivisions": 4,
                "cap_grid": 64,
                "material": spec.material,
            },
        })
    first_plane = planes[0]
    first_geometry = objects[0]["geometry"]
    return {
        "schema_version": 1,
        "defaults": {
            "image": {
                "width": scene.sensor_width,
                "height": scene.sensor_height,
                "region": {
                    "x": 2 * (scene.sensor_width // SENSOR_CROP_SCALE),
                    "y": 2 * (scene.sensor_height // SENSOR_CROP_SCALE),
                    "width": scene.sensor_width // SENSOR_CROP_SCALE,
                    "height": scene.sensor_height // SENSOR_CROP_SCALE,
                },
            },
            "camera": {
                "focal_mm": scene.camera.focal_mm,
                "aperture_mm": scene.camera.aperture_mm,
                "position_m": list(scene.camera.position_m),
                "target_m": list(scene.camera.target_m),
                "focus_target_m": list(scene.camera.focus_target_m),
                "up": list(scene.camera.up),
            },
            "exposure": {
                "time_s": 1.0 / 60.0,
                "iso": 100.0,
                "sensor_sweeps": int(sensor_sweeps),
                "t5_pair_budget": 20_000_000,
            },
            "flash": {"intensity_scale": 1.0},
            "font": objects[0]["font"],
            "planes": planes,
            "materials": {
                "quiet_background": {
                    "albedo_rgb": [0.004, 0.005, 0.006],
                    "reflectivity": 0.012,
                    "diffusion": 1.0,
                    "absorption": 0.985,
                    "roughness": 1.0,
                    "metallic": 0.0,
                },
                "text_surface": {
                    "albedo_rgb": [0.92, 0.89, 0.78],
                    "reflectivity": 0.92,
                    "diffusion": 0.98,
                    "absorption": 0.04,
                    "roughness": 0.98,
                    "metallic": 0.0,
                    "ior": 1.5,
                },
            },
            # Legacy geometry remains as the shared default inherited by each
            # object. The compiler consumes every object in this same scene.
            "geometry": {
                **first_geometry,
                "embed_plane": first_plane["id"],
            },
            "objects": objects,
        },
        "jobs": [{"id": JOB_ID}],
    }


def _prepare_native_delta_restore(
    revision_dir: str,
    next_scan_control: dict[str, Any],
) -> tuple[str, str, str] | None:
    """Expand retained display evidence back to the native square accumulator."""

    restore = next_scan_control.get("metadata", {}).get("delta_restore", {})
    os.makedirs(revision_dir, exist_ok=True)
    sum_path = str(restore.get("sensor_sum_path", ""))
    weight_path = str(restore.get("exposure_weight_path", ""))
    if not sum_path or not weight_path:
        return None
    if not os.path.isfile(sum_path) or not os.path.isfile(weight_path):
        return None
    display_sum = np.asarray(np.load(sum_path, allow_pickle=False), np.float32)
    display_weight = np.asarray(np.load(weight_path, allow_pickle=False), np.float32)
    if (
        display_sum.ndim != 3 or display_sum.shape[2] != 3
        or display_weight.shape != display_sum.shape[:2]
    ):
        raise ValueError("retained delta exposure artifacts have incompatible shapes")
    height, width = display_sum.shape[:2]
    resolution = max(width, height)
    source_y = np.minimum(
        ((np.arange(resolution) + 0.5) * height / resolution).astype(np.int64),
        height - 1,
    )
    source_x = np.minimum(
        ((np.arange(resolution) + 0.5) * width / resolution).astype(np.int64),
        width - 1,
    )
    display_square_sum = display_sum[source_y[:, None], source_x[None, :]]
    display_square_weight = display_weight[source_y[:, None], source_x[None, :]]
    dirty_display = np.zeros((height, width), dtype=bool)
    for request in next_scan_control.get("pixel_slice_requests", ()):
        if int(request["width"]) != width or int(request["height"]) != height:
            raise ValueError("dirty pixel slice must match retained display exposure")
        indices = np.asarray(request["site_indices"], dtype=np.int64)
        dirty_display.reshape(-1)[indices] = True
    dirty_square = dirty_display[source_y[:, None], source_x[None, :]]
    # Native storage is [camera-right, camera-up], the transpose of display.
    native_sum = np.ascontiguousarray(display_square_sum.transpose(1, 0, 2))
    native_weight = np.ascontiguousarray(display_square_weight.T)
    native_dirty = np.ascontiguousarray(
        np.flatnonzero(dirty_square.T.reshape(-1)), np.uint32
    )
    native_sum_path = os.path.join(revision_dir, "restore_sensor_sum_native.npy")
    native_weight_path = os.path.join(
        revision_dir, "restore_sensor_weight_native.npy"
    )
    native_dirty_path = os.path.join(revision_dir, "dirty_sensor_sites_native.npy")
    np.save(native_sum_path, native_sum, allow_pickle=False)
    np.save(native_weight_path, native_weight, allow_pickle=False)
    np.save(native_dirty_path, native_dirty, allow_pickle=False)
    return native_sum_path, native_weight_path, native_dirty_path


def _blend_delta_exposure(
    current_linear: np.ndarray,
    current_weight: np.ndarray,
    previous_sum: np.ndarray,
    previous_weight: np.ndarray,
    dirty_slice: SensorPixelSlice,
) -> np.ndarray:
    """Fade killed sites from retained evidence to statistically regrown evidence."""

    current = np.asarray(current_linear, np.float32)
    weight = np.asarray(current_weight, np.float32)
    old_sum = np.asarray(previous_sum, np.float32)
    old_weight = np.asarray(previous_weight, np.float32)
    if (
        current.ndim != 3 or current.shape[2] != 3
        or weight.shape != current.shape[:2]
        or old_sum.shape != current.shape
        or old_weight.shape != current.shape[:2]
        or (dirty_slice.height, dirty_slice.width) != current.shape[:2]
    ):
        raise ValueError("delta exposure layers and pixel slice must share a raster")
    old = old_sum / np.maximum(old_weight[..., None], 1.0e-12)
    dirty = dirty_slice.mask()
    old_reference = old_weight[dirty & (old_weight > 0.0)]
    target_weight = (
        float(np.median(old_reference)) if old_reference.size else 1.0
    )
    alpha = np.clip(weight / max(target_weight, 1.0e-12), 0.0, 1.0)
    result = current.copy()
    result[dirty] = (
        old[dirty] * (1.0 - alpha[dirty, None])
        + current[dirty] * alpha[dirty, None]
    )
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class RenderedTextRevision:
    sequence: int
    text: str
    image_path: str
    linear_path: str
    manifest_path: str
    elapsed_s: float
    diagnostic_image_path: str = ""
    orthographic_path: str = ""
    priority_overlay_path: str = ""


RenderFunction = Callable[
    [int, str, dict[str, Any], dict[str, Any] | None, ProgressSink],
    RenderedTextRevision,
]


class SpectralTextRenderWorker:
    """Single render owner with a coalescing one-item pending slot."""

    def __init__(
        self,
        render_function: RenderFunction,
        progress_observer: ProgressSink | None = None,
    ):
        self._render_function = render_function
        self._progress_observer = progress_observer
        self._condition = threading.Condition()
        self._pending: tuple[
            int, str, dict[str, Any], dict[str, Any] | None
        ] | None = None
        self._latest: RenderedTextRevision | None = None
        self._latest_sequence = 0
        self._progress = ExposureProgressBroker()
        self._submitted_sequence = 0
        self._busy = False
        self._error = ""
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run,
            name="spectral-text-render-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        text: str,
        order: dict[str, Any],
        next_scan_control: dict[str, Any] | None = None,
    ) -> int:
        cancel = getattr(self._render_function, "cancel_current", None)
        if callable(cancel):
            cancel()
        with self._condition:
            self._submitted_sequence += 1
            sequence = self._submitted_sequence
            self._pending = (
                sequence,
                str(text),
                copy.deepcopy(order),
                copy.deepcopy(next_scan_control),
            )
            self._progress.reset()
            self._condition.notify()
            return sequence

    def snapshot(self) -> tuple[RenderedTextRevision | None, bool, str, int]:
        with self._condition:
            return self._latest, self._busy, self._error, self._submitted_sequence

    def progress_snapshot(self):
        return self._progress.snapshot()

    def close(self) -> None:
        cancel = getattr(self._render_function, "cancel_current", None)
        if callable(cancel):
            cancel()
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                sequence, text, order, next_scan_control = self._pending
                self._pending = None
                self._busy = True
                self._error = ""
            try:
                def publish_progress(event: ExposureProgressEvent) -> None:
                    # Old revisions may finish expensive native work after a new
                    # edit has been submitted.  Keep their logs/artifacts, but do
                    # not let them replace the active exposure stream.
                    with self._condition:
                        if sequence != self._submitted_sequence:
                            return
                    self._progress.publish(event)
                    if self._progress_observer is not None:
                        self._progress_observer(event)

                result = self._render_function(
                    sequence, text, order, next_scan_control, publish_progress
                )
                with self._condition:
                    self._latest = result
                    self._latest_sequence = sequence
            except Exception as exc:
                with self._condition:
                    if sequence == self._submitted_sequence and not self._stopping:
                        self._error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()


def make_subprocess_renderer(
    output_root: str,
    camera_profile: ColorScienceProfile,
    targeted_fraction: float = DEFAULT_TARGETED_FRACTION,
    priority_model_path: str = "",
) -> RenderFunction:
    root = os.path.abspath(output_root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    process_lock = threading.Lock()
    active_process: list[subprocess.Popen[str] | None] = [None]

    def cancel_current() -> None:
        with process_lock:
            process = active_process[0]
        if process is not None and process.poll() is None:
            process.terminate()

    def render(
        sequence: int,
        text: str,
        order: dict[str, Any],
        next_scan_control: dict[str, Any] | None,
        progress_sink: ProgressSink,
    ) -> RenderedTextRevision:
        revision_dir = os.path.join(root, f"revision_{sequence:04d}")
        os.makedirs(revision_dir, exist_ok=True)
        job_dir = os.path.join(revision_dir, JOB_ID)
        os.makedirs(job_dir, exist_ok=True)
        order_path = os.path.join(revision_dir, "scene_order.json")
        with open(order_path, "w", encoding="utf-8") as handle:
            json.dump(order, handle, indent=2)
        next_scan_path = ""
        native_delta_restore = None
        if next_scan_control is not None:
            next_scan_path = os.path.join(revision_dir, "next_scan_control.json")
            with open(next_scan_path, "w", encoding="utf-8") as handle:
                json.dump(next_scan_control, handle, indent=2)
            native_delta_restore = _prepare_native_delta_restore(
                revision_dir, next_scan_control
            )
        region = order["defaults"]["image"]["region"]
        reusable_model = str(priority_model_path).strip()
        flat_reference_path = ""
        attention_model_path = ""
        attention_priority_path = ""
        if reusable_model:
            reusable_model = os.path.abspath(reusable_model)
            if not os.path.isfile(reusable_model):
                raise FileNotFoundError(reusable_model)
            attention_model_path = reusable_model
            print(
                f"[live] reusable ray-trained NN revision {sequence}: "
                f"model={attention_model_path}",
                flush=True,
            )
        else:
            print(
                f"[live] revision {sequence}: no learned model supplied; "
                "using measured heuristic/exploration scheduling",
                flush=True,
            )
        progress_sink(ExposureProgressEvent(
            exposure_id=f"revision-{sequence:04d}",
            sequence=0,
            kind=ExposureProgressKind.STARTED,
            region=SensorRegion(
                x=int(region.get("x", 0)),
                y=int(region.get("y", 0)),
                width=int(region["width"]),
                height=int(region["height"]),
            ),
            preview_path=attention_priority_path,
            priority_map_path=attention_priority_path,
            message="shared-camera UI exposure started",
        ))
        started = time.perf_counter()
        command = [
            sys.executable,
            os.path.join(script_dir, "exposure_render_demo.py"),
            "--scene-order", order_path,
            "--scene-job", JOB_ID,
            "--integrator", "bdpt",
            "--backend", "cpp",
            "--frames", "1",
            "--gpu-resident",
            "--no-window",
            "--save-files",
            "--out-dir", job_dir,
            "--bdpt-native-packages", "1",
            "--progress-dir",
            os.path.join(revision_dir, "progress", JOB_ID),
            "--progress-exposure-id",
            f"revision-{sequence:04d}",
        ]
        log_path = os.path.join(revision_dir, "render.log")
        print(f"[live] revision {sequence} process: {' '.join(command)}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            environment = os.environ.copy()
            # A live camera keeps exposing until the window closes or edited
            # content supersedes this revision. Zero means no authored epoch
            # cap; the statistical scorer continues to order every GPU layer.
            environment["SPECTRAL_SENSOR_MAX_EPOCHS"] = "0"
            environment["SPECTRAL_SENSOR_CONTINUOUS"] = "1"
            environment["SPECTRAL_SENSOR_TARGETED_FRACTION"] = str(
                float(targeted_fraction)
            )
            if next_scan_path:
                environment["SPECTRAL_NEXT_SCAN_CONTROL"] = next_scan_path
            else:
                environment.pop("SPECTRAL_NEXT_SCAN_CONTROL", None)
            if native_delta_restore is not None:
                environment["SPECTRAL_SENSOR_RESTORE_SUM"] = native_delta_restore[0]
                environment["SPECTRAL_SENSOR_RESTORE_WEIGHT"] = native_delta_restore[1]
                environment["SPECTRAL_SENSOR_DIRTY_SITES"] = native_delta_restore[2]
            else:
                environment.pop("SPECTRAL_SENSOR_RESTORE_SUM", None)
                environment.pop("SPECTRAL_SENSOR_RESTORE_WEIGHT", None)
                environment.pop("SPECTRAL_SENSOR_DIRTY_SITES", None)
            if attention_model_path:
                environment["SPECTRAL_SENSOR_PRIORITY_MODEL"] = attention_model_path
            else:
                environment.pop("SPECTRAL_SENSOR_PRIORITY_MODEL", None)
            environment["SPECTRAL_PROGRESS_RETAIN_LAYERS"] = "3"
            process = subprocess.Popen(
                command,
                cwd=script_dir,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with process_lock:
                active_process[0] = process
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = ExposureProgressEvent.from_line(line.rstrip("\r\n"))
                    if event is not None:
                        progress_sink(event)
                    print(f"[render {sequence:04d}] {line}", end="", flush=True)
                return_code = process.wait()
            finally:
                with process_lock:
                    if active_process[0] is process:
                        active_process[0] = None
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
        linear_path = os.path.join(job_dir, "0000_cpp_linear.npy")
        diagnostic_image_path = os.path.join(job_dir, "0000_cpp.png")
        camera_image_path = os.path.join(job_dir, "0000_cpp_camera.png")
        linear_sensor = np.load(linear_path, allow_pickle=False)
        save_linear_sensor_image(linear_sensor, camera_image_path, camera_profile)
        print(
            f"[live] camera output revision {sequence}: raw={linear_path} "
            f"display={camera_image_path} white_level={camera_profile.sensor_white_level:g} "
            f"exposure_ev={camera_profile.exposure_compensation_ev:+g} "
            f"tone={camera_profile.tone_curve_mode}",
            flush=True,
        )
        return RenderedTextRevision(
            sequence=sequence,
            text=text,
            image_path=camera_image_path,
            linear_path=linear_path,
            manifest_path=os.path.join(job_dir, "composition_manifest.json"),
            elapsed_s=time.perf_counter() - started,
            diagnostic_image_path=diagnostic_image_path,
            orthographic_path=flat_reference_path,
            priority_overlay_path=os.path.join(job_dir, "0000_cpp_priority.png"),
        )

    setattr(render, "cancel_current", cancel_current)
    return render


def _texture_display_rect(
    source_width: int,
    source_height: int,
    window_width: int,
    window_height: int,
) -> tuple[int, int, int, int]:
    """Fit a texture into the window without enlarging traced pixels."""
    scale = min(
        1.0,
        float(window_width) / float(source_width),
        float(window_height) / float(source_height),
    )
    width = max(1, int(round(float(source_width) * scale)))
    height = max(1, int(round(float(source_height) * scale)))
    return (window_width - width) // 2, (window_height - height) // 2, width, height


def _texture_panel_rect(
    source_width: int,
    source_height: int,
    panel_width: int,
    panel_height: int,
) -> tuple[int, int, int, int]:
    """Fit and zoom a diagnostic texture to use its available panel."""

    scale = min(
        float(panel_width) / max(1, source_width),
        float(panel_height) / max(1, source_height),
    )
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    return (
        (panel_width - width) // 2,
        (panel_height - height) // 2,
        width,
        height,
    )


def _hud_layout(width: int) -> dict[str, int]:
    """Size a separate editor strip for the available display width."""
    scale = min(1.0, max(0.4, float(width) / 480.0))
    editor_font_px = max(10, int(round(24.0 * scale)))
    status_font_px = max(9, int(round(20.0 * scale)))
    padding = max(4, int(round(14.0 * scale)))
    editor_lines = 2 if width < 320 else 4
    editor_line_height = max(12, int(round(editor_font_px * 1.2)))
    status_line_height = max(11, int(round(status_font_px * 1.2)))
    height = padding * 3 + editor_lines * editor_line_height + status_line_height
    return {
        "height": height,
        "padding": padding,
        "editor_font_px": editor_font_px,
        "status_font_px": status_font_px,
        "editor_lines": editor_lines,
    }


def _window_regions(width: int, height: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Partition the window into non-overlapping canvas and HUD regions."""
    hud_height = min(_hud_layout(width)["height"], max(0, height - 1))
    canvas_height = height - hud_height
    return (0, 0, width, canvas_height), (0, canvas_height, width, hud_height)


def _initial_window_size(display_width: int, display_height: int) -> tuple[int, int]:
    """Derive the photographed UI scene around fixed-size preview panes."""
    work_width, work_height = int(display_width), int(display_height)
    if work_width <= 0 or work_height <= 0:
        raise ValueError("scan chunk dimensions must be positive")
    metrics = _program_frame_metrics(work_width, work_height)
    return metrics["frame_width"], metrics["frame_height"]


def _sensor_product_texture_rect(
    product: SensorRegion,
    photographed: SensorRegion,
    texture_width: int,
    texture_height: int,
) -> tuple[int, int, int, int]:
    """Map an absolute sensor product to its progressive tile texture."""

    x0 = int(round(
        (product.x - photographed.x) * texture_width / photographed.width
    ))
    y0 = int(round(
        (product.y - photographed.y) * texture_height / photographed.height
    ))
    x1 = int(round(
        (product.x + product.width - photographed.x)
        * texture_width / photographed.width
    ))
    y1 = int(round(
        (product.y + product.height - photographed.y)
        * texture_height / photographed.height
    ))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(texture_width, x1), min(texture_height, y1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, 0, 0
    return x0, y0, x1 - x0, y1 - y0


def _sensor_product_window_rect(
    product: SensorRegion,
    photographed: SensorRegion,
    window_width: int,
    window_height: int,
) -> tuple[int, int, int, int]:
    """Map an authored camera product to its exact UI-view destination."""

    x0 = int(round(
        (product.x - photographed.x) * window_width / photographed.width
    ))
    y0 = int(round(
        (product.y - photographed.y) * window_height / photographed.height
    ))
    x1 = int(round(
        (product.x + product.width - photographed.x)
        * window_width / photographed.width
    ))
    y1 = int(round(
        (product.y + product.height - photographed.y)
        * window_height / photographed.height
    ))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _active_work_texture(
    texture: Any,
    uv_bounds: tuple[float, float, float, float] | None,
    viewport_width: int,
    viewport_height: int,
    sequence: int,
) -> Any:
    """Copy one exact-size 2-D region of the active sensor footprint."""

    if texture is None:
        return texture
    width, height = texture.get_size()
    if uv_bounds is None:
        uv_bounds = (0.0, 0.0, 1.0, 1.0)
    u0, v0, u1, v1 = uv_bounds
    active = SensorRegion(
        max(0, min(width - 1, int(math.floor(u0 * width)))),
        max(0, min(height - 1, int(math.floor(v0 * height)))),
        max(1, int(math.ceil((u1 - u0) * width))),
        max(1, int(math.ceil((v1 - v0) * height))),
    )
    active = SensorRegion(
        active.x,
        active.y,
        min(active.width, width - active.x),
        min(active.height, height - active.y),
    )
    regions = [
        SensorRegion(
            x,
            y,
            min(viewport_width, active.x + active.width - x),
            min(viewport_height, active.y + active.height - y),
        )
        for y in range(active.y, active.y + active.height, viewport_height)
        for x in range(active.x, active.x + active.width, viewport_width)
    ]
    # Morton order keeps presentation viewports spatially coherent while the
    # underlying work contract remains arbitrary 2-D regions and pixel slices.
    def morton(region: SensorRegion) -> int:
        local_x = (region.x - active.x) // viewport_width
        local_y = (region.y - active.y) // viewport_height
        result = 0
        for bit in range(max(local_x.bit_length(), local_y.bit_length(), 1)):
            result |= ((local_x >> bit) & 1) << (2 * bit)
            result |= ((local_y >> bit) & 1) << (2 * bit + 1)
        return result

    regions.sort(key=morton)
    selected = regions[max(0, int(sequence) - 1) % len(regions)]
    work = texture.__class__((viewport_width, viewport_height))
    work.fill((0, 0, 0))
    work.blit(
        texture,
        (0, 0),
        (selected.x, selected.y, selected.width, selected.height),
    )
    return work


def run_window(
    output_root: str,
    initial_text: str,
    display_width: int,
    display_height: int,
    sensor_sweeps: int,
    camera_profile: ColorScienceProfile,
    targeted_fraction: float = DEFAULT_TARGETED_FRACTION,
    priority_model_path: str = "",
) -> int:
    import pygame

    scene_width, scene_height = _initial_window_size(
        display_width, display_height
    )
    print(
        "[live] " + render_contract_summary(
            display_width,
            display_height,
            sensor_sweeps,
            scene_width=scene_width,
            scene_height=scene_height,
        ),
        flush=True,
    )
    print(
        "[live] recursive GPU exposure continues until the window closes or the "
        "text changes; measured Monte Carlo noise/ambiguity orders each new layer",
        flush=True,
    )
    print(
        "[live] camera output "
        f"white_level={camera_profile.sensor_white_level:g} "
        f"exposure_ev={camera_profile.exposure_compensation_ev:+g} "
        f"white_balance={np.asarray(camera_profile.white_balance).tolist()} "
        f"tone={camera_profile.tone_curve_mode} output={camera_profile.output_space}",
        flush=True,
    )
    pygame.init()
    pygame.key.start_text_input()
    window = pygame.display.set_mode((scene_width, scene_height))
    pygame.display.set_caption("Spectral text surface")
    clock = pygame.time.Clock()

    inventory = DisplaySceneInventory(os.path.join(
        os.path.abspath(output_root), "display_inventory.json"
    ))
    active_object_ids: list[tuple[str, ...]] = [()]
    active_exposure_id = [""]
    active_scene_lock = threading.Lock()

    def retain_progress(event: ExposureProgressEvent) -> None:
        with active_scene_lock:
            object_ids = active_object_ids[0]
            expected_exposure_id = active_exposure_id[0]
        if object_ids and event.exposure_id == expected_exposure_id:
            inventory.record_progress(PROGRAM_SCENE_ID, object_ids, event)

    worker = SpectralTextRenderWorker(
        make_subprocess_renderer(
            output_root, camera_profile, targeted_fraction, priority_model_path
        ),
        progress_observer=retain_progress,
    )
    text = str(initial_text)[:500]
    cursor = len(text)
    submitted_text = ""
    dirty_at = time.monotonic() - 2.0
    loaded_sequence = 0
    loaded_progress: tuple[str, int] | None = None
    progress_event: ExposureProgressEvent | None = None
    texture = None
    priority_texture = None
    photographed_region = program_display_region(scene_width, scene_height)
    ui_sensor_regions = program_ui_sensor_regions(
        scene_width,
        scene_height,
        work_width=display_width,
        work_height=display_height,
    )
    ray_control_hitboxes: dict[str, Any] = {}
    transition_restore: dict[str, str] = {}
    transition_dirty_slice: SensorPixelSlice | None = None
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if (
                        "window-close" in ray_control_hitboxes
                        and ray_control_hitboxes["window-close"].collidepoint(event.pos)
                    ):
                        running = False
                    elif (
                        "window-minimize" in ray_control_hitboxes
                        and ray_control_hitboxes["window-minimize"].collidepoint(event.pos)
                    ):
                        pygame.display.iconify()
                    elif (
                        "window-maximize" in ray_control_hitboxes
                        and ray_control_hitboxes["window-maximize"].collidepoint(event.pos)
                    ):
                        # The authored UI exposes the control geometrically, but
                        # work viewports are a fixed acquisition contract and
                        # must never be resized by presentation state.
                        pass
                elif event.type == pygame.TEXTINPUT:
                    text = (text[:cursor] + event.text + text[cursor:])[:500]
                    cursor = min(len(text), cursor + len(event.text))
                    dirty_at = time.monotonic()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_BACKSPACE and cursor > 0:
                        text = text[:cursor - 1] + text[cursor:]
                        cursor -= 1
                        dirty_at = time.monotonic()
                    elif event.key == pygame.K_DELETE and cursor < len(text):
                        text = text[:cursor] + text[cursor + 1:]
                        dirty_at = time.monotonic()
                    elif event.key == pygame.K_LEFT:
                        cursor = max(0, cursor - 1)
                    elif event.key == pygame.K_RIGHT:
                        cursor = min(len(text), cursor + 1)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        text = text[:cursor] + "\n" + text[cursor:]
                        cursor += 1
                        dirty_at = time.monotonic()

            if text.strip() and text != submitted_text and time.monotonic() - dirty_at >= 1.0:
                next_revision = worker.snapshot()[3] + 1
                scene = build_self_rendering_program_scene(
                    text,
                    display_width=scene_width,
                    display_height=scene_height,
                    work_width=display_width,
                    work_height=display_height,
                    status_text=(
                        f"EXPOSING REVISION {next_revision} - "
                        "CONTINUOUS SENSOR REFINEMENT"
                    ),
                    revision=next_revision,
                )
                delta_restore: dict[str, str] = {}
                for retained_scene in inventory.snapshot():
                    if retained_scene.scene.scene_id != PROGRAM_SCENE_ID:
                        continue
                    if not _static_scene_is_reusable(
                        retained_scene.scene, scene
                    ):
                        continue
                    for retained_object in retained_scene.objects:
                        if retained_object.object_id != "editor-text":
                            continue
                        for retained_product in retained_object.products:
                            if retained_product.kind is DisplayProductKind.IMAGE:
                                delta_restore = {
                                    "sensor_sum_path": retained_product.sensor_sum_path,
                                    "exposure_weight_path": (
                                        retained_product.exposure_weight_path
                                    ),
                                }
                inventory.put_scene(scene)
                dynamic_objects = {
                    spec.object_id: spec
                    for spec in scene.objects
                    if spec.object_id in {"editor-text", "status-text"}
                }
                inventory.mark_dirty(
                    PROGRAM_SCENE_ID,
                    tuple(dynamic_objects),
                    {
                        object_id: spec.products[0].sensor_pixel_slice
                        for object_id, spec in dynamic_objects.items()
                    },
                )
                with active_scene_lock:
                    active_object_ids[0] = tuple(
                        spec.object_id for spec in scene.objects if spec.enabled
                    )
                    # Ignore the tail of the previous process and the tiny
                    # submission handoff window until the new exposure id is known.
                    active_exposure_id[0] = ""
                order = build_display_scene_order(
                    scene, sensor_sweeps=sensor_sweeps,
                )
                next_scan = build_ui_next_scan_control(
                    scene,
                    photographed_region,
                    sequence=worker.snapshot()[3] + 1,
                    targeted_fraction=targeted_fraction,
                    scan_width=display_width,
                    scan_height=display_height,
                    delta_restore=delta_restore,
                )
                sequence = worker.submit(text, order, next_scan)
                transition_restore = delta_restore
                requests = next_scan.get("pixel_slice_requests", ())
                transition_dirty_slice = (
                    None if not requests else SensorPixelSlice(
                        int(requests[0]["width"]),
                        int(requests[0]["height"]),
                        tuple(
                            int(index)
                            for request in requests
                            for index in request["site_indices"]
                        ),
                    )
                )
                with active_scene_lock:
                    active_exposure_id[0] = f"revision-{sequence:04d}"
                print(
                    f"[live] submitted revision {sequence}: characters={len(text)} "
                    + render_contract_summary(
                        display_width,
                        display_height,
                        sensor_sweeps,
                        scene_width=scene_width,
                        scene_height=scene_height,
                    ),
                    flush=True,
                )
                submitted_text = text
                loaded_progress = None
                progress_event = None

            latest, _busy, _error, _submitted_sequence = worker.snapshot()
            newest_progress, _progress_layers = worker.progress_snapshot()
            if newest_progress is not None:
                progress_key = (newest_progress.exposure_id, newest_progress.sequence)
                if progress_key != loaded_progress:
                    if (
                        newest_progress.linear_accumulation_path
                        and os.path.isfile(newest_progress.linear_accumulation_path)
                    ):
                        linear_progress = np.load(
                            newest_progress.linear_accumulation_path,
                            allow_pickle=False,
                        )
                        if (
                            transition_dirty_slice is not None
                            and newest_progress.exposure_weight_path
                            and os.path.isfile(newest_progress.exposure_weight_path)
                            and os.path.isfile(
                                transition_restore.get("sensor_sum_path", "")
                            )
                            and os.path.isfile(
                                transition_restore.get(
                                    "exposure_weight_path", ""
                                )
                            )
                        ):
                            linear_progress = _blend_delta_exposure(
                                linear_progress,
                                np.load(
                                    newest_progress.exposure_weight_path,
                                    allow_pickle=False,
                                ),
                                np.load(
                                    transition_restore["sensor_sum_path"],
                                    allow_pickle=False,
                                ),
                                np.load(
                                    transition_restore["exposure_weight_path"],
                                    allow_pickle=False,
                                ),
                                transition_dirty_slice,
                            )
                        # Presentation-only exposure for an in-progress Monte
                        # Carlo layer. Raw evidence remains linear and untouched.
                        preview_rgb = np.maximum(linear_progress[..., :3], 0.0)
                        preview_luma = np.mean(preview_rgb, axis=2)
                        positive = preview_luma[preview_luma > 0.0]
                        preview_white = (
                            float(np.percentile(positive, 99.0))
                            if positive.size else float(camera_profile.sensor_white_level)
                        )
                        preview_profile = replace(
                            camera_profile,
                            sensor_white_level=max(preview_white, 1.0e-12),
                        )
                        display_progress = process_linear_sensor_image(
                            linear_progress, preview_profile,
                        )
                        pixels = np.ascontiguousarray(
                            np.clip(display_progress[..., :3], 0.0, 1.0) * 255.0,
                            dtype=np.uint8,
                        )
                        texture = pygame.surfarray.make_surface(pixels.swapaxes(0, 1))
                    priority_path = newest_progress.priority_map_path
                    if priority_path and os.path.isfile(priority_path):
                        if priority_path.lower().endswith(".npy"):
                            values = np.maximum(
                                np.load(priority_path, allow_pickle=False), 0.0
                            )
                            peak = max(float(np.max(values)), 1.0e-12)
                            value = np.clip(values / peak, 0.0, 1.0)
                            heat = np.stack([
                                value,
                                np.sqrt(value) * 0.35,
                                1.0 - value,
                            ], axis=-1)
                            priority_pixels = np.ascontiguousarray(
                                heat * 255.0, dtype=np.uint8
                            )
                            priority_texture = pygame.surfarray.make_surface(
                                priority_pixels.swapaxes(0, 1)
                            )
                        else:
                            priority_texture = pygame.image.load(priority_path).convert()
                    loaded_progress = progress_key
                    progress_event = newest_progress
            if latest is not None and latest.sequence != loaded_sequence:
                # Preserve the last auto-exposed progressive texture in the
                # window. The fixed-profile final image is still saved to disk.
                if progress_event is None:
                    texture = pygame.image.load(latest.image_path).convert()
                if latest.priority_overlay_path and os.path.isfile(latest.priority_overlay_path):
                    priority_texture = pygame.image.load(latest.priority_overlay_path).convert()
                loaded_sequence = latest.sequence
                print(
                    f"[live] completed revision {latest.sequence}: "
                    f"texture={texture.get_width()}x{texture.get_height()} "
                    f"elapsed={latest.elapsed_s:.3f}s image={latest.image_path}",
                    flush=True,
                )

            # CLI dimensions permanently define the complete preview product.
            # Every panel, label, control, editor, and status pixel stays inside.
            width, height = scene_width, scene_height
            window.fill((15, 18, 19))
            if texture is not None:
                window.blit(texture, (0, 0))
            panel_products = (
                ("camera-panel", texture),
                ("work-panel", priority_texture),
            )
            header_bottom = max(
                region.y + region.height
                for object_id, region in ui_sensor_regions.items()
                if object_id in {
                    "camera-label", "work-label",
                    "window-minimize", "window-maximize", "window-close",
                }
            )
            for panel_id, panel_texture in panel_products:
                panel_region = ui_sensor_regions[panel_id]
                content_y = min(
                    panel_region.y + panel_region.height,
                    max(panel_region.y, header_bottom + 1),
                )
                content_region = SensorRegion(
                    panel_region.x,
                    content_y,
                    panel_region.width,
                    max(1, panel_region.y + panel_region.height - content_y),
                )
                destination = _sensor_product_window_rect(
                    content_region,
                    photographed_region,
                    width,
                    height,
                )
                panel_texture = _active_work_texture(
                    panel_texture,
                    None if progress_event is None
                    else progress_event.global_uv_bounds,
                    destination[2],
                    destination[3],
                    1 if progress_event is None else progress_event.sequence,
                )
                if panel_texture is None:
                    continue
                window.blit(panel_texture, destination[:2])

            # Controls remain part of the camera-rendered scene. Only their
            # sensor-space rectangles are reused for pointer hit testing.
            ray_control_hitboxes.clear()
            for object_id in (
                "window-close", "window-maximize", "window-minimize",
            ):
                destination = _sensor_product_window_rect(
                    ui_sensor_regions[object_id],
                    photographed_region,
                    width,
                    height,
                )
                ray_control_hitboxes[object_id] = pygame.Rect(*destination)
            pygame.display.flip()
            clock.tick(60)
    finally:
        worker.close()
        pygame.key.stop_text_input()
        pygame.quit()
    return 0


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="exposures/live_spectral_text")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--display-width", type=int, default=DEFAULT_DISPLAY_WIDTH,
                        help="Exact width of the complete camera-rendered program preview.")
    parser.add_argument("--display-height", type=int, default=DEFAULT_DISPLAY_HEIGHT,
                        help="Exact height of the complete camera-rendered program preview.")
    parser.add_argument("--sensor-sweeps", type=int, default=DEFAULT_SENSOR_SWEEPS,
                        help="Scene-order compatibility value; the live camera itself exposes continuously.")
    parser.add_argument("--targeted-fraction", type=float,
                        default=DEFAULT_TARGETED_FRACTION,
                        help="Fraction of recursive GPU work selected by learned/measured value; the remainder guarantees broad coverage (default: 0.75).")
    parser.add_argument("--priority-model", default="",
                        help="Reusable ray-trained .npz model; skips per-text synthetic training.")
    parser.add_argument("--camera-white-level", type=float, default=1.0,
                        help="Fixed linear sensor value mapped as camera white (default: 1).")
    parser.add_argument("--camera-exposure-ev", type=float, default=0.0,
                        help="Camera output exposure compensation in stops (default: 0).")
    parser.add_argument("--camera-white-balance", type=float, nargs=3,
                        metavar=("R", "G", "B"), default=(1.0, 1.0, 1.0),
                        help="Fixed camera RGB white-balance gains (default: 1 1 1).")
    parser.add_argument("--camera-tone-curve", choices=("linear", "reinhard"),
                        default="reinhard",
                        help="Fixed camera output tone curve (default: reinhard).")
    parser.add_argument("--write-order", metavar="PATH",
                        help="Write the resolved demo order and exit without opening a window.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(sys.argv[1:] if argv is None else argv)
    if not 0.0 <= float(args.targeted_fraction) <= 1.0:
        raise SystemExit("--targeted-fraction must be in [0, 1]")
    camera_profile = ColorScienceProfile.spectral_sensor_srgb(
        sensor_white_level=args.camera_white_level,
        exposure_compensation_ev=args.camera_exposure_ev,
        white_balance=np.asarray(args.camera_white_balance, dtype=np.float64),
        tone_curve_mode=args.camera_tone_curve,
    )
    if args.write_order:
        scene_width, scene_height = _initial_window_size(
            args.display_width, args.display_height
        )
        scene = build_self_rendering_program_scene(
            args.text,
            display_width=scene_width,
            display_height=scene_height,
            work_width=args.display_width,
            work_height=args.display_height,
        )
        with open(args.write_order, "w", encoding="utf-8") as handle:
            json.dump(build_display_scene_order(
                scene, sensor_sweeps=args.sensor_sweeps,
            ), handle, indent=2)
        return 0
    return run_window(
        args.out_dir,
        args.text,
        args.display_width,
        args.display_height,
        args.sensor_sweeps,
        camera_profile,
        args.targeted_fraction,
        args.priority_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
