"""Editable paragraph UI backed by asynchronous spectral thick-lens renders."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
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
    FontAssetSpec,
    ExtrusionAssetSpec,
    ExtrudedTokenAsset,
    BakeRequest,
    BakeTargetKind,
    SENSOR_DISPLAY_ORIENTATION,
    RenderAssetCatalog,
    plan_ink_atlas_bake,
    ink_order_for_request,
    extract_raytraced_sprite,
    token_alpha_mask,
    measure_sprite_exposure_quality,
    CachedTokenStringComposer,
    RenderObjectLibrary,
    canonical_subtype_directory,
    render_style_key,
    DEFAULT_INK_CONDITION,
    DEFAULT_MONOFONT_HORIZONTAL_SPACING_PX,
    DEFAULT_MONOFONT_VERTICAL_SPACING_PX,
    ControlLayoutDesign,
    ProgramUILayout,
    LayoutRenderPipeline,
    PanelGeometryKind,
    compose_layout_panel_rgba,
    program_ui_manifest,
    layout_render_pipeline,
    program_frame_metrics as manifest_program_frame_metrics,
    layout_program_ui,
    OpenGLContextHost,
    process_linear_sensor_image,
    save_linear_sensor_image,
)


DEFAULT_TEXT = "Actual light takes the long way home through glass."
JOB_ID = "live_paragraph"
DEFAULT_DISPLAY_WIDTH = 960
DEFAULT_DISPLAY_HEIGHT = 600
DEFAULT_SENSOR_SWEEPS = 4
DEFAULT_TARGETED_FRACTION = 0.75
DEFAULT_LIVE_FOREGROUND_EPOCHS = 64
DEFAULT_ATLAS_EPOCHS_PER_EXPOSURE = 1
DEFAULT_ATLAS_SENSOR_TOP_K = 1024
DEFAULT_ATLAS_STEPS_PER_EPOCH = 64
DEFAULT_ATLAS_SAMPLES_PER_NODE = 1024
SENSOR_CROP_SCALE = 5
DEFAULT_EXTRUSION_DEPTH_RATIO = 0.08
PROGRAM_SCENE_ID = "spectral-program-scene"
PROGRAM_STATIC_GEOMETRY_REVISION = 7
_DEFAULT_PROGRAM_MANIFEST = program_ui_manifest()
_DEFAULT_LAYOUT_RENDER_PIPELINE = layout_render_pipeline(
    _DEFAULT_PROGRAM_MANIFEST
)
# Compatibility names are derived views of the manifest contract, not a
# second source of production truth.
FIXED_IMAGE_ALPHABET = tuple(
    _DEFAULT_LAYOUT_RENDER_PIPELINE.glyph_alphabet
)
FIXED_IMAGE_FONT = FontAssetSpec(
    family=_DEFAULT_LAYOUT_RENDER_PIPELINE.font_family,
    weight=_DEFAULT_LAYOUT_RENDER_PIPELINE.font_weight,
    style=_DEFAULT_LAYOUT_RENDER_PIPELINE.font_style,
)
FIXED_UI_TOKEN_STRINGS = _DEFAULT_LAYOUT_RENDER_PIPELINE.static_tokens


def _manifest_render_font(manifest: Any = None) -> FontAssetSpec:
    pipeline = layout_render_pipeline(
        manifest or _DEFAULT_PROGRAM_MANIFEST
    )
    return FontAssetSpec(
        family=pipeline.font_family,
        weight=pipeline.font_weight,
        style=pipeline.font_style,
    )
# The native thick-camera sensor is square. Its exact 200x153 product readback
# restores the UI aspect afterward, so the camera-facing world layout must use
# one square physical frame or it is vertically compressed twice.
PROGRAM_SCENE_EXTENT_M = 0.375


def _program_frame_metrics(
    work_width: int, work_height: int
) -> dict[str, int]:
    """Compatibility wrapper around the authoritative program manifest."""

    return manifest_program_frame_metrics(
        work_width, work_height, program_ui_manifest()
    )
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
        f"authored_sensor_sweeps={int(sensor_sweeps)} "
        f"foreground_epochs<={DEFAULT_LIVE_FOREGROUND_EPOCHS}; "
        "production=alphabet->tokens->token-string->total-scene"
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

    if kind not in (
        DisplayPrimitiveKind.PLANE,
        DisplayPrimitiveKind.BOX,
        DisplayPrimitiveKind.ICON,
    ):
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


def resolved_program_ui_layout(
    display_width: int,
    display_height: int,
    *,
    work_width: int | None = None,
    work_height: int | None = None,
    manifest: Any = None,
) -> ProgramUILayout:
    """Resolve the authoritative manifest inside the photographed crop."""

    crop = program_display_region(display_width, display_height)
    return layout_program_ui(
        manifest or program_ui_manifest(),
        crop.width,
        crop.height,
        origin=(crop.x, crop.y),
        work_width=work_width,
        work_height=work_height,
    )


def program_ui_sensor_regions(
    display_width: int,
    display_height: int,
    *,
    work_width: int | None = None,
    work_height: int | None = None,
    manifest: Any = None,
) -> dict[str, SensorRegion]:
    """Manifest-resolved sensor-space products photographed together."""

    layout = resolved_program_ui_layout(
        display_width,
        display_height,
        work_width=work_width,
        work_height=work_height,
        manifest=manifest,
    )
    return {
        key: SensorRegion(*rect)
        for key, rect in layout.regions.items()
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


def control_layout_scene_objects(
    design: ControlLayoutDesign,
    photographed_region: SensorRegion,
    *,
    sensor_width: int,
    sensor_height: int,
    revision: int = PROGRAM_STATIC_GEOMETRY_REVISION,
    font: FontAssetSpec | None = None,
) -> tuple[DisplayObjectSpec, ...]:
    """Turn one knob layout into physical objects in the same UI photograph."""

    font = font or FIXED_IMAGE_FONT
    objects: list[DisplayObjectSpec] = []
    for element in design.elements:
        x, y, width, height = element.rect
        region = SensorRegion(int(x), int(y), int(width), int(height))
        if (
            region.x < 0 or region.y < 0
            or region.x + region.width > sensor_width
            or region.y + region.height > sensor_height
        ):
            raise ValueError(f"control element {element.key!r} is outside the sensor")
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", element.key).strip("-")
        object_id = f"control-layout-{safe_key}"
        if element.kind == "panel":
            primitive = DisplayPrimitive(DisplayPrimitiveKind.PLANE)
            front_offset = 0.0001
            thickness = 0.0015
        else:
            content = element.label
            if element.kind == "knob":
                metadata = design.knob_routes.get(element.route_key, {})
                value = element.value
                choices = list(metadata.get("choices", []) or [])
                if choices and isinstance(value, (int, np.integer)):
                    index = int(value)
                    if 0 <= index < len(choices):
                        value = choices[index]
                content = f"{element.label}  {value}"
            primitive = DisplayPrimitive(DisplayPrimitiveKind.TEXT, content=content)
            front_offset = 0.0007
            thickness = 0.0008
        objects.append(DisplayObjectSpec(
            object_id=object_id,
            primitive=primitive,
            placement=_placement_for_sensor_region(
                region, photographed_region, thickness_m=thickness,
                front_offset_m=front_offset,
            ),
            products=(DisplayProductRequest(DisplayProductKind.IMAGE, region),),
            layout_rectangle=LayoutRectangle(
                region,
                BevelRegion(0.001, 0.0004, BevelProfile.CHAMFER),
                (LayoutSeam.BEVEL,) * 4,
            ),
            material="text_surface",
            surface_material="quiet_background",
            font_family=font.family,
            font_weight=font.weight,
            font_style=font.style,
            horizontal_align="left" if element.kind == "knob" else "center",
            vertical_align="center",
            revision=max(1, int(revision)),
        ))
    return tuple(objects)


def build_self_rendering_program_scene(
    text: str,
    *,
    display_width: int = DEFAULT_DISPLAY_WIDTH,
    display_height: int = DEFAULT_DISPLAY_HEIGHT,
    work_width: int | None = None,
    work_height: int | None = None,
    status_text: str = "SPECTRAL EXPOSURE ACTIVE",
    revision: int = 1,
    control_layout: ControlLayoutDesign | None = None,
    program_manifest: Any = None,
) -> DisplaySceneSpec:
    """Build editor and window-control geometry for one physical photograph."""

    manifest = program_manifest or program_ui_manifest()
    program_layout = resolved_program_ui_layout(
        display_width,
        display_height,
        work_width=work_width,
        work_height=work_height,
        manifest=manifest,
    )
    regions = {
        key: SensorRegion(*rect)
        for key, rect in program_layout.regions.items()
    }
    program_font = _manifest_render_font(manifest)

    def panel_primitive_kind(panel_id: str) -> DisplayPrimitiveKind:
        geometry = program_layout.panel_geometry[panel_id]
        return (
            DisplayPrimitiveKind.PLANE
            if geometry.kind is PanelGeometryKind.PLANE
            else DisplayPrimitiveKind.BOX
        )

    def panel_bevel(panel_id: str) -> BevelRegion:
        geometry = program_layout.panel_geometry[panel_id]
        return BevelRegion(
            geometry.bevel_width_m,
            geometry.bevel_depth_m,
            BevelProfile.CHAMFER,
        )

    crop = program_display_region(display_width, display_height)
    base = build_program_display_scene(
        text,
        display_width=display_width,
        display_height=display_height,
        revision=revision,
    )
    editor_region = regions["editor-text"]
    backdrop_region = regions[manifest.name]
    backdrop = DisplayObjectSpec(
        object_id=manifest.name,
        primitive=DisplayPrimitive(panel_primitive_kind(manifest.name)),
        placement=_placement_for_sensor_region(
            backdrop_region,
            crop,
            thickness_m=program_layout.panel_geometry[manifest.name].thickness_m,
            front_offset_m=-0.001,
        ),
        products=(
            DisplayProductRequest(DisplayProductKind.IMAGE, backdrop_region),
        ),
        layout_rectangle=LayoutRectangle(
            backdrop_region,
            panel_bevel(manifest.name),
            (LayoutSeam.FLUSH,) * 4,
        ),
        revision=PROGRAM_STATIC_GEOMETRY_REVISION,
    )
    editor = replace(
        base.objects[0],
        object_id="editor-text",
        font_family=program_font.family,
        font_weight=program_font.weight,
        font_style=program_font.style,
        horizontal_align="left",
        vertical_align="top",
        placement=_placement_for_sensor_region(
            editor_region, crop,
            thickness_m=program_layout.panel_geometry["editor-text"].thickness_m
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
            panel_bevel("editor-text"),
            (LayoutSeam.BEVEL,) * 4,
        ),
    )
    authored_text_objects = []
    for object_id, content in (
        ("camera-label", program_layout.labels["camera-label"]),
        ("work-label", program_layout.labels["work-label"]),
        (
            "asset-browser-label",
            program_layout.labels["asset-browser-label"],
        ),
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
            font_family=program_font.family,
            font_weight=program_font.weight,
            font_style=program_font.style,
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
    for object_id in ("camera-panel", "work-panel", "asset-browser"):
        if object_id not in regions:
            continue
        sensor_region = regions[object_id]
        panels.append(DisplayObjectSpec(
            object_id=object_id,
            primitive=DisplayPrimitive(panel_primitive_kind(object_id)),
            placement=_placement_for_sensor_region(
                sensor_region,
                crop,
                thickness_m=program_layout.panel_geometry[object_id].thickness_m,
            ),
            products=(
                DisplayProductRequest(DisplayProductKind.IMAGE, sensor_region),
            ),
            layout_rectangle=LayoutRectangle(
                sensor_region,
                panel_bevel(object_id),
                (LayoutSeam.BEVEL,) * 4,
            ),
            revision=PROGRAM_STATIC_GEOMETRY_REVISION,
        ))
    for object_id, action in program_layout.actions.items():
        if object_id not in regions:
            continue
        sensor_region = regions[object_id]
        primitive = (
            DisplayPrimitive(
                DisplayPrimitiveKind.TEXT, content=action.label
            )
            if action.label else DisplayPrimitive(
                DisplayPrimitiveKind.ICON, icon_name=action.icon
            )
        )
        controls.append(DisplayObjectSpec(
            object_id=object_id,
            primitive=primitive,
            placement=_placement_for_sensor_region(
                sensor_region,
                crop,
                thickness_m=0.0008,
                front_offset_m=0.0006,
            ),
            products=(
                DisplayProductRequest(DisplayProductKind.IMAGE, sensor_region),
            ),
            material="control_surface",
            surface_material="quiet_background",
            font_family=program_font.family,
            font_weight=program_font.weight,
            font_style=program_font.style,
            horizontal_align="center",
            vertical_align="center",
            layout_rectangle=LayoutRectangle(
                sensor_region,
                BevelRegion(0.001, 0.0004, BevelProfile.CHAMFER),
                (LayoutSeam.BEVEL,) * 4,
            ),
            # Controls retain their exposure when only editor content changes.
            revision=PROGRAM_STATIC_GEOMETRY_REVISION,
        ))
    layout_objects = (
        control_layout_scene_objects(
            control_layout,
            crop,
            sensor_width=base.sensor_width,
            sensor_height=base.sensor_height,
            revision=revision,
            font=program_font,
        )
        if control_layout is not None else ()
    )
    return replace(
        base,
        objects=(
            backdrop, editor, *panels, *authored_text_objects, *controls,
            *layout_objects,
        ),
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


def _sensor_orientation_marker_path(sensor_sum_path: str) -> str:
    return str(sensor_sum_path) + ".display-orientation"


def _mark_sensor_display_orientation(sensor_sum_path: str) -> None:
    """Mark evidence produced under the current display/readback convention."""

    path = str(sensor_sum_path)
    if not path:
        return
    marker = _sensor_orientation_marker_path(path)
    temporary = marker + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(SENSOR_DISPLAY_ORIENTATION)
    os.replace(temporary, marker)


def _sensor_sum_has_current_orientation(sensor_sum_path: str) -> bool:
    """Never mix unmarked legacy sums with current progressive frames."""

    marker = _sensor_orientation_marker_path(sensor_sum_path)
    if not sensor_sum_path or not os.path.isfile(marker):
        return False
    try:
        with open(marker, "r", encoding="utf-8") as handle:
            return handle.read().strip() == SENSOR_DISPLAY_ORIENTATION
    except OSError:
        return False


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
        if spec.primitive.kind in {
            DisplayPrimitiveKind.PLANE,
            DisplayPrimitiveKind.BOX,
        }:
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
        token_asset = ExtrudedTokenAsset(
            token=spec.primitive.authored_text(),
            font=FontAssetSpec(
                family=spec.font_family,
                weight=spec.font_weight,
                style=spec.font_style,
            ),
            extrusion=ExtrusionAssetSpec(
                height_m=text_height,
                line_height_m=text_height,
                line_spacing=1.05,
                text_box_m=(
                    placement.size_m[0] * 0.94,
                    placement.size_m[1] * 0.90,
                ),
                horizontal_align=spec.horizontal_align,
                vertical_align=spec.vertical_align,
                depth_ratio=DEFAULT_EXTRUSION_DEPTH_RATIO,
                embed_fraction=0.25,
                profile="straight",
                outline_subdivisions=4,
                cap_grid=64,
            ),
            material=spec.material,
        )
        objects.append(token_asset.scene_order_object(
            spec.object_id, plane_id, enabled=spec.enabled
        ))
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


def _display_raster_to_native_square(array: np.ndarray) -> np.ndarray:
    """Invert the C++ getter's y-flip plus the Python display transpose."""

    display = np.asarray(array)
    if display.ndim not in (2, 3):
        raise ValueError("display sensor raster must be HxW or HxWxC")
    height, width = display.shape[:2]
    resolution = max(width, height)
    source_y = np.minimum(
        ((np.arange(resolution) + 0.5) * height / resolution).astype(np.int64),
        height - 1,
    )
    source_x = np.minimum(
        ((np.arange(resolution) + 0.5) * width / resolution).astype(np.int64),
        width - 1,
    )
    square = display[source_y[:, None], source_x[None, :]]
    # C++ readback writes getter[out_y, z] = native[y, z], with
    # out_y = resolution - 1 - y. Display then transposes getter[z, out_y].
    # Therefore native[y, z] = display.T[resolution - 1 - y, z].
    return np.ascontiguousarray(
        np.flip(np.swapaxes(square, 0, 1), axis=0)
    )


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
    dirty_display = np.zeros((height, width), dtype=bool)
    for request in next_scan_control.get("pixel_slice_requests", ()):
        if int(request["width"]) != width or int(request["height"]) != height:
            raise ValueError("dirty pixel slice must match retained display exposure")
        indices = np.asarray(request["site_indices"], dtype=np.int64)
        dirty_display.reshape(-1)[indices] = True
    native_sum = _display_raster_to_native_square(display_sum)
    native_weight = _display_raster_to_native_square(display_weight)
    native_dirty = np.ascontiguousarray(
        np.flatnonzero(
            _display_raster_to_native_square(dirty_display).reshape(-1)
        ),
        np.uint32,
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
    interface_assembly_key: str = ""
    after_render_key: str = ""


RenderFunction = Callable[
    [int, str, dict[str, Any], dict[str, Any] | None, ProgressSink],
    RenderedTextRevision,
]
BackgroundRenderFunction = Callable[[BakeRequest], Any]
BackgroundRequestSource = Callable[[], BakeRequest | None]


ATLAS_CHECKPOINT_SCHEMA_VERSION = 1


def _write_atlas_checkpoint(path: str, payload: dict[str, Any]) -> None:
    """Atomically publish one resumable, completed atlas epoch."""

    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _load_atlas_checkpoint(
    path: str,
    request: BakeRequest,
    catalog_refinement_pass: int,
) -> dict[str, Any] | None:
    """Return a compatible checkpoint whose evidence files still exist."""

    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            checkpoint = dict(json.load(handle))
        completed = int(checkpoint.get("completed_epochs", 0))
        target = int(checkpoint.get("target_refinement_pass", -1))
        compatible = bool(
            int(checkpoint.get("schema_version", -1))
            == ATLAS_CHECKPOINT_SCHEMA_VERSION
            and checkpoint.get("request_key") == request.request_key
            and checkpoint.get("target_key") == request.target_key
            and int(checkpoint.get("catalog_refinement_pass", -1))
            == int(catalog_refinement_pass)
            and completed > 0
            and target >= int(catalog_refinement_pass) + completed
        )
        evidence_paths = tuple(
            str(checkpoint.get(name, ""))
            for name in (
                "linear_accumulation_path",
                "sensor_sum_path",
                "exposure_weight_path",
            )
        )
        if not compatible or not all(
            evidence_path and os.path.isfile(evidence_path)
            for evidence_path in evidence_paths
        ):
            return None
        return checkpoint
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _atlas_request_directory(output_root: str, request: BakeRequest) -> str:
    """Place every bake directly in its style/subtype/condition object."""

    return canonical_subtype_directory(
        output_root,
        request.token_asset,
        request.condition,
        target_kind=request.target_kind.value,
    )

class InkAtlasSubprocessRenderer:
    """Advance one persistent atlas asset by one substantial resumable burst."""

    def __init__(
        self,
        output_root: str,
        catalog: RenderAssetCatalog,
        *,
        epochs_per_exposure: int = DEFAULT_ATLAS_EPOCHS_PER_EXPOSURE,
        sensor_top_k: int = DEFAULT_ATLAS_SENSOR_TOP_K,
        steps_per_epoch: int = DEFAULT_ATLAS_STEPS_PER_EPOCH,
        samples_per_node: int = DEFAULT_ATLAS_SAMPLES_PER_NODE,
    ) -> None:
        self.output_root = os.path.abspath(output_root)
        self.catalog = catalog
        self.object_library = RenderObjectLibrary(os.path.join(
            self.output_root, "render_object_library.json"
        ))
        self.object_library.migrate_catalog(self.catalog)
        self.epochs_per_exposure = max(1, int(epochs_per_exposure))
        self.sensor_top_k = max(64, min(1024, int(sensor_top_k)))
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        # Native lineage storage is bounded to 2^20 records. With the maximum
        # 1024 selected nodes, 1024 rays/node fills that arena exactly.
        self.samples_per_node = max(1, min(1024, int(samples_per_node)))
        self._lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None
        self._work_revision = 0
        self._work_state: tuple[str, str, int, bool] = ("", "", 0, False)
        self._work_object_key = ""
        self._visual_accept_requested = False

    def work_snapshot(self) -> tuple[int, str, str, int, bool]:
        """Latest individual asset shown by the UI work panel."""

        with self._lock:
            path, token, refinement_pass, active = self._work_state
            return (
                self._work_revision,
                path,
                token,
                refinement_pass,
                active,
            )

    def work_object_snapshot(self) -> tuple[int, str, str, int, bool]:
        """Identity for scene/image/light-field views of the active object."""

        with self._lock:
            _path, token, refinement_pass, active = self._work_state
            return (
                self._work_revision,
                self._work_object_key,
                token,
                refinement_pass,
                active,
            )

    def cancel_current(self) -> None:
        with self._lock:
            process = self._active_process
        if process is not None and process.poll() is None:
            process.terminate()

    def request_active_visual_pass(self) -> bool:
        """Accept the latest published epoch of the active automatic job."""

        with self._lock:
            process = self._active_process
            current_path = self._work_state[0]
            if (
                process is None
                or process.poll() is not None
                or not current_path
                or not os.path.isfile(current_path)
            ):
                return False
            self._visual_accept_requested = True
        process.terminate()
        return True

    def __call__(self, request: BakeRequest):
        package = ink_order_for_request(request)
        job_id = str(package["jobs"][0]["id"])
        request_dir = _atlas_request_directory(self.output_root, request)
        os.makedirs(request_dir, exist_ok=True)
        preview_path = os.path.join(request_dir, "0000_cpp.png")
        sum_path = os.path.join(request_dir, "0000_cpp_sum_linear.npy")
        weight_path = os.path.join(
            request_dir, "0000_cpp_exposure_weight.npy"
        )
        checkpoint_path = os.path.join(
            request_dir, "active_epoch_checkpoint.json"
        )
        with self._lock:
            self._visual_accept_requested = False
            self._work_revision += 1
            self._work_object_key = render_style_key(request.token_asset)
            self._work_state = (
                preview_path if os.path.isfile(preview_path) else "",
                request.token_asset.token,
                request.refinement_pass,
                True,
            )
        prior_record = self.catalog.find(
            request.target_key, request.condition, request.product_kind
        )
        prior_metadata = (
            {} if prior_record is None else dict(prior_record.metadata)
        )
        resumable_prior = (
            prior_metadata.get("refinement_state") in {
                "developing", "converged",
            }
            and prior_metadata.get("sensor_display_orientation")
            == SENSOR_DISPLAY_ORIENTATION
        )
        prior_quality = dict(prior_metadata.get("atlas_quality", {}))
        prior_linear = None
        if (
            resumable_prior
            and
            prior_record is not None
            and prior_record.linear_path
            and os.path.isfile(prior_record.linear_path)
        ):
            prior_linear = np.asarray(
                np.load(prior_record.linear_path, allow_pickle=False),
                np.float32,
            )[..., :3].copy()
        catalog_refinement_pass = max(
            int(request.refinement_pass),
            (
                int(prior_metadata.get("refinement_pass", 0))
                if resumable_prior else 0
            ),
        )
        checkpoint = _load_atlas_checkpoint(
            checkpoint_path, request, catalog_refinement_pass
        )
        restored_checkpoint_epochs = (
            0 if checkpoint is None
            else int(checkpoint["completed_epochs"])
        )
        refinement_pass = (
            catalog_refinement_pass + restored_checkpoint_epochs
        )
        configured_epoch_count = max(
            1, self.epochs_per_exposure * self.steps_per_epoch
        )
        target_refinement_pass = (
            catalog_refinement_pass + configured_epoch_count
            if checkpoint is None
            else int(checkpoint["target_refinement_pass"])
        )
        remaining_epochs = target_refinement_pass - refinement_pass
        if remaining_epochs <= 0:
            # The parent may have been interrupted after the final checkpoint
            # but before catalog commit. One more epoch safely completes the
            # ordinary final-image/sprite path without discarding evidence.
            target_refinement_pass = refinement_pass + 1
            remaining_epochs = 1
        if checkpoint is not None:
            with self._lock:
                self._work_revision += 1
                self._work_state = (
                    str(checkpoint["linear_accumulation_path"]),
                    request.token_asset.token,
                    refinement_pass,
                    True,
                )

        order_path = os.path.join(request_dir, "scene_order.json")
        progress_dir = os.path.join(request_dir, "progress")
        with open(order_path, "w", encoding="utf-8") as handle:
            json.dump(package, handle, indent=2)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        command = [
            sys.executable,
            os.path.join(script_dir, "exposure_render_demo.py"),
            "--scene-order", order_path,
            "--scene-job", job_id,
            "--integrator", "bdpt",
            "--backend", "cpp",
            "--frames", "1",
            "--gpu-resident",
            "--no-window",
            "--save-files",
            "--out-dir", request_dir,
            "--progress-dir", progress_dir,
            "--progress-exposure-id", request.request_key,
            "--bdpt-native-packages", "1",
            "--no-convergence-drive-batches",
        ]
        environment = os.environ.copy()
        # Amortize expensive scene/lens/GPU preparation across a large ray
        # burst. This is a work quantum, not a lifetime target: the catalog
        # keeps offering the asset until external image deltas converge.
        # One native submission is one visible and resumable epoch. The total
        # burst remains epochs_per_exposure * steps_per_epoch submissions.
        environment["SPECTRAL_SENSOR_MAX_EPOCHS"] = str(remaining_epochs)
        environment["SPECTRAL_SENSOR_CONTINUOUS"] = "1"
        environment["SPECTRAL_SENSOR_TOP_K"] = str(self.sensor_top_k)
        environment["SPECTRAL_SENSOR_STEPS_PER_LAYER"] = "1"
        environment["SPECTRAL_SENSOR_SAMPLES_PER_NODE"] = str(
            self.samples_per_node
        )
        environment["SPECTRAL_SENSOR_TARGETED_FRACTION"] = "0.75"
        environment["SPECTRAL_PROGRESS_RETAIN_LAYERS"] = "2"
        environment["SPECTRAL_EXPOSURE_SEED_OFFSET"] = str(
            refinement_pass * 1_000_003
        )
        restore_sum_source = (
            str(checkpoint["sensor_sum_path"])
            if checkpoint is not None else sum_path
        )
        restore_weight_source = (
            str(checkpoint["exposure_weight_path"])
            if checkpoint is not None else weight_path
        )
        if (
            refinement_pass > 0
            and os.path.isfile(restore_sum_source)
            and os.path.isfile(restore_weight_source)
        ):
            restore_sum_path = os.path.join(
                request_dir, "restore_sensor_sum_native.npy"
            )
            restore_weight_path = os.path.join(
                request_dir, "restore_sensor_weight_native.npy"
            )
            dirty_path = os.path.join(request_dir, "sensor_dirty_sites.npy")
            np.save(
                restore_sum_path,
                _display_raster_to_native_square(
                    np.load(restore_sum_source, allow_pickle=False)
                ),
                allow_pickle=False,
            )
            np.save(
                restore_weight_path,
                _display_raster_to_native_square(
                    np.load(restore_weight_source, allow_pickle=False)
                ),
                allow_pickle=False,
            )
            np.save(dirty_path, np.empty(0, np.uint32))
            environment["SPECTRAL_SENSOR_RESTORE_SUM"] = restore_sum_path
            environment["SPECTRAL_SENSOR_RESTORE_WEIGHT"] = restore_weight_path
            environment["SPECTRAL_SENSOR_DIRTY_SITES"] = dirty_path
        renderer_context_path = os.path.join(
            request_dir, "render_context.json"
        )
        _write_atlas_checkpoint(renderer_context_path, {
            "schema_version": 1,
            "request": {
                "request_key": request.request_key,
                "target_key": request.target_key,
                "target_kind": request.target_kind.value,
                "condition": {
                    "condition_key": request.condition.condition_key,
                    "azimuth_deg": request.condition.azimuth_deg,
                    "elevation_deg": request.condition.elevation_deg,
                    "light_rig": request.condition.light_rig,
                    "material_variant": request.condition.material_variant,
                    "frame": request.condition.frame,
                },
                "product_kind": request.product_kind.value,
                "scene_id": request.scene_id,
                "object_ids": list(request.object_ids),
            },
            "command": command,
            "cwd": script_dir,
            "environment": {
                key: value for key, value in sorted(environment.items())
                if key.startswith("SPECTRAL_")
            },
            "created_at_s": time.time(),
        })
        log_path = os.path.join(request_dir, "render.log")
        print(
            f"[ink-atlas] rendering {request.target_kind.value} "
            f"{request.token_asset.token!r} -> {request_dir}",
            flush=True,
        )
        with open(log_path, "w", encoding="utf-8") as log:
            executed_epochs = 0
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
            with self._lock:
                self._active_process = process
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if "[sensor-refine] epoch " in line:
                        executed_epochs += 1
                    progress = ExposureProgressEvent.from_line(
                        line.rstrip("\r\n")
                    )
                    if (
                        progress is not None
                        and progress.linear_accumulation_path
                        and os.path.isfile(progress.linear_accumulation_path)
                    ):
                        completed_epochs = (
                            restored_checkpoint_epochs
                            + int(progress.completed_work)
                        )
                        if (
                            progress.sensor_sum_path
                            and progress.exposure_weight_path
                            and os.path.isfile(progress.sensor_sum_path)
                            and os.path.isfile(progress.exposure_weight_path)
                        ):
                            _write_atlas_checkpoint(checkpoint_path, {
                                "schema_version": ATLAS_CHECKPOINT_SCHEMA_VERSION,
                                "request_key": request.request_key,
                                "target_key": request.target_key,
                                "catalog_refinement_pass": catalog_refinement_pass,
                                "completed_epochs": completed_epochs,
                                "target_refinement_pass": target_refinement_pass,
                                "linear_accumulation_path": (
                                    progress.linear_accumulation_path
                                ),
                                "sensor_sum_path": progress.sensor_sum_path,
                                "exposure_weight_path": (
                                    progress.exposure_weight_path
                                ),
                            })
                        with self._lock:
                            self._work_revision += 1
                            self._work_state = (
                                progress.linear_accumulation_path,
                                request.token_asset.token,
                                catalog_refinement_pass + completed_epochs,
                                True,
                            )
                    print(f"[ink-atlas] {line}", end="", flush=True)
                return_code = process.wait()
            finally:
                with self._lock:
                    if self._active_process is process:
                        self._active_process = None
            if return_code != 0:
                with self._lock:
                    visual_accept = self._visual_accept_requested
                    self._visual_accept_requested = False
                    current_path, current_token, current_pass, _active = (
                        self._work_state
                    )
                    self._work_revision += 1
                    self._work_state = (
                        current_path, current_token, current_pass, False
                    )
                if visual_accept:
                    checkpoint = _load_atlas_checkpoint(
                        checkpoint_path, request, catalog_refinement_pass
                    )
                    if checkpoint is None:
                        raise RuntimeError(
                            "visual pass requested before the first epoch checkpoint"
                        )
                    accepted_linear = str(
                        checkpoint["linear_accumulation_path"]
                    )
                    accepted_image = np.asarray(
                        np.load(accepted_linear, allow_pickle=False),
                        np.float32,
                    )[..., :3]
                    capture = request.capture
                    if capture is None:
                        raise RuntimeError(
                            "visual pass requires an atlas capture contract"
                        )
                    sprite = extract_raytraced_sprite(
                        accepted_image, request.token_asset, capture
                    )
                    sprite_path = sprite.save(os.path.join(
                        request_dir, "raytraced_sprite.npz"
                    ))
                    accepted_manifest = os.path.join(
                        request_dir, "atlas_capture_manifest.json"
                    )
                    _write_atlas_checkpoint(accepted_manifest, {
                        "schema_version": 1,
                        "target_key": request.target_key,
                        "condition_key": request.condition.condition_key,
                        "token": request.token_asset.token,
                        "capture": {
                            "width": capture.width,
                            "height": capture.height,
                            "content_region": list(capture.content_region),
                        },
                        "linear_path": accepted_linear,
                        "completion_basis": "human_visual_pass",
                    })
                    record = self.catalog.complete(
                        request,
                        linear_path=accepted_linear,
                        manifest_path=accepted_manifest,
                        width=capture.width,
                        height=capture.height,
                        samples=int(current_pass),
                        metadata={
                            "bounded_background_render": True,
                            "sensor_display_orientation": SENSOR_DISPLAY_ORIENTATION,
                            "refinement_state": "developing",
                            "refinement_pass": int(current_pass),
                            "completion_basis": "human_visual_pass",
                            "human_visual_pass": True,
                            "human_visual_pass_at_s": time.time(),
                            "atlas_quality": {
                                "composable": True,
                                "converged": False,
                                "human_visual_pass": True,
                            },
                            "sprite_path": sprite_path,
                            "sum_linear_path": str(
                                checkpoint["sensor_sum_path"]
                            ),
                            "exposure_weight_path": str(
                                checkpoint["exposure_weight_path"]
                            ),
                        },
                    )
                    self.object_library.adopt_record(
                        record,
                        scene_path=order_path,
                        condition=request.condition,
                        token=request.token_asset.token,
                        tags=("human-visual-pass",),
                        renderer_context_path=renderer_context_path,
                    )
                    print(
                        f"[ink-atlas] human visual pass "
                        f"{request.token_asset.token!r} pass={current_pass}",
                        flush=True,
                    )
                    return record
                raise subprocess.CalledProcessError(return_code, command)
        linear_path = os.path.join(request_dir, "0000_cpp_linear.npy")
        manifest_path = os.path.join(request_dir, "atlas_capture_manifest.json")
        capture = request.capture
        if capture is None:
            raise RuntimeError("ink atlas render requires a capture contract")
        if not os.path.isfile(linear_path) or not os.path.isfile(weight_path):
            raise RuntimeError("ink atlas render did not produce linear/weight evidence")
        linear_image = np.asarray(
            np.load(linear_path, allow_pickle=False), np.float32
        )[..., :3]
        exposure_weight = np.asarray(
            np.load(weight_path, allow_pickle=False), np.float32
        )
        alpha = token_alpha_mask(request.token_asset, capture)
        quality = measure_sprite_exposure_quality(
            linear_image,
            exposure_weight,
            alpha,
            refinement_pass=refinement_pass + max(1, executed_epochs),
            previous_image=prior_linear,
            previous_stable_hold=int(prior_quality.get("stable_hold", 0)),
        )
        manifest = {
            "schema_version": 1,
            "target_key": request.target_key,
            "condition_key": request.condition.condition_key,
            "token": request.token_asset.token,
            "font": request.token_asset.font.identity_payload(),
            "capture": (
                {}
                if capture is None
                else {
                    "width": capture.width,
                    "height": capture.height,
                    "content_region": list(capture.content_region),
                    "preserves_neighboring_light": True,
                }
            ),
            "linear_path": linear_path,
            "sum_linear_path": sum_path,
            "exposure_weight_path": weight_path,
            "preview_path": preview_path,
            "quality": quality.as_metadata(),
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        sprite = extract_raytraced_sprite(
            linear_image, request.token_asset, capture
        )
        sprite_path = sprite.save(
            os.path.join(request_dir, "raytraced_sprite.npz")
        )
        record = self.catalog.complete(
            request,
            linear_path=linear_path,
            preview_path=preview_path,
            manifest_path=manifest_path,
            width=capture.width,
            height=capture.height,
            samples=quality.refinement_pass,
            metadata={
                "bounded_background_render": True,
                "refinement_state": (
                    "converged" if quality.converged else "developing"
                ),
                "refinement_pass": quality.refinement_pass,
                "epochs_per_exposure": self.epochs_per_exposure,
                "sensor_top_k": self.sensor_top_k,
                "steps_per_epoch": self.steps_per_epoch,
                "samples_per_node": self.samples_per_node,
                "checkpoint_epochs_per_exposure": configured_epoch_count,
                "refinement_steps_per_checkpoint": 1,
                "camera_primary_rays_per_epoch": (
                    self.sensor_top_k * self.samples_per_node
                ),
                "camera_primary_rays_per_burst": (
                    self.sensor_top_k
                    * configured_epoch_count
                    * self.samples_per_node
                ),
                "sensor_display_orientation": SENSOR_DISPLAY_ORIENTATION,
                "atlas_quality": quality.as_metadata(),
                "sum_linear_path": sum_path,
                "exposure_weight_path": weight_path,
                "sprite_path": sprite_path,
                "sprite_composition": (
                    "premultiplied + (1-alpha)*destination + signed_additive"
                ),
            },
        )
        bundle = self.object_library.adopt_record(
            record,
            scene_path=order_path,
            condition=request.condition,
            object_kind=request.target_kind.value,
            token=request.token_asset.token,
            tags=("ink-on-slate", "ray-traced-ui-bakery"),
            renderer_context_path=renderer_context_path,
        )
        print(
            f"[object-library] object={bundle.object_key} "
            f"modes={[mode.value for mode in bundle.available_view_modes]} "
            f"manifest={bundle.manifest_path}",
            flush=True,
        )
        try:
            os.unlink(checkpoint_path)
        except FileNotFoundError:
            pass
        print(
            f"[ink-atlas] refined {request.token_asset.token!r} "
            f"pass={quality.refinement_pass} "
            f"state={record.metadata['refinement_state']} "
            f"stable={quality.stable_hold} "
            f"record={record.record_key}",
            flush=True,
        )
        with self._lock:
            self._work_revision += 1
            self._work_state = (
                preview_path,
                request.token_asset.token,
                quality.refinement_pass,
                False,
            )
        return record


class SpectralTextRenderWorker:
    """Single render owner with a coalescing one-item pending slot."""

    def __init__(
        self,
        render_function: RenderFunction,
        progress_observer: ProgressSink | None = None,
        background_source: BackgroundRequestSource | None = None,
        background_render_function: BackgroundRenderFunction | None = None,
    ):
        self._render_function = render_function
        self._progress_observer = progress_observer
        if (background_source is None) != (background_render_function is None):
            raise ValueError(
                "background source and render function must be supplied together"
            )
        self._background_source = background_source
        self._background_render_function = background_render_function
        self._background_paused = False
        self._background_user_paused = False
        self._background_error = ""
        self._background_busy = False
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
        cancel_background = getattr(
            self._background_render_function, "cancel_current", None
        )
        if callable(cancel_background):
            cancel_background()
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

    def set_background_paused(self, paused: bool) -> None:
        """Pause/resume only automatic queue resolution; keep checkpoints."""

        paused = bool(paused)
        with self._condition:
            self._background_user_paused = paused
            self._background_paused = paused
            self._background_error = ""
            self._condition.notify_all()
        if paused:
            cancel_background = getattr(
                self._background_render_function, "cancel_current", None
            )
            if callable(cancel_background):
                cancel_background()

    def toggle_background_paused(self) -> bool:
        with self._condition:
            paused = not self._background_user_paused
        self.set_background_paused(paused)
        return paused

    def background_is_paused(self) -> bool:
        with self._condition:
            return self._background_user_paused

    def wake_background(self) -> None:
        """Retry/replan background work after new tokens or external recovery."""

        with self._condition:
            if not self._background_user_paused:
                self._background_paused = False
                self._background_error = ""
                self._condition.notify_all()

    def background_snapshot(self) -> tuple[bool, str]:
        with self._condition:
            return self._background_busy, self._background_error

    def snapshot(self) -> tuple[RenderedTextRevision | None, bool, str, int]:
        with self._condition:
            return self._latest, self._busy, self._error, self._submitted_sequence

    def progress_snapshot(self):
        return self._progress.snapshot()

    def close(self) -> None:
        cancel = getattr(self._render_function, "cancel_current", None)
        if callable(cancel):
            cancel()
        cancel_background = getattr(
            self._background_render_function, "cancel_current", None
        )
        if callable(cancel_background):
            cancel_background()
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            background_request: BakeRequest | None = None
            with self._condition:
                while self._pending is None and not self._stopping:
                    if (
                        self._background_source is not None
                        and not self._background_paused
                    ):
                        background_request = self._background_source()
                        if background_request is not None:
                            break
                    self._condition.wait()
                if self._stopping:
                    return
                foreground = self._pending is not None
                if foreground:
                    sequence, text, order, next_scan_control = self._pending
                    self._pending = None
                    background_request = None
                self._busy = True
                self._background_busy = not foreground
                self._error = ""
            try:
                if not foreground:
                    assert background_request is not None
                    assert self._background_render_function is not None
                    self._background_render_function(background_request)
                    continue

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
                    if not foreground:
                        # A foreground submission intentionally terminates an
                        # atlas subprocess. Retry it after that foreground work.
                        if self._pending is None and not self._stopping:
                            if not self._background_paused:
                                self._background_error = (
                                    f"{type(exc).__name__}: {exc}"
                                )
                            self._background_paused = True
                    elif sequence == self._submitted_sequence and not self._stopping:
                        self._error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._condition:
                    self._busy = False
                    self._background_busy = False
                    self._condition.notify_all()


def make_subprocess_renderer(
    output_root: str,
    camera_profile: ColorScienceProfile,
    targeted_fraction: float = DEFAULT_TARGETED_FRACTION,
    priority_model_path: str = "",
    object_library: RenderObjectLibrary | None = None,
    layout_pipeline: LayoutRenderPipeline | None = None,
    layout_contract: dict[str, Any] | None = None,
) -> RenderFunction:
    root = os.path.abspath(output_root)
    final_pipeline = (
        layout_pipeline or _DEFAULT_LAYOUT_RENDER_PIPELINE
    )
    final_library = object_library or RenderObjectLibrary(os.path.join(
        root, "render_object_library.json"
    ))
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
            # Foreground work is urgent but bounded. It may stop earlier when
            # the clarity discriminator accepts the frame; afterward the same
            # render owner can manufacture one missing atlas asset.
            environment["SPECTRAL_SENSOR_MAX_EPOCHS"] = str(
                DEFAULT_LIVE_FOREGROUND_EPOCHS
            )
            environment["SPECTRAL_SENSOR_CONTINUOUS"] = "0"
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
        elapsed_s = time.perf_counter() - started
        final = final_library.register_interface_after_render(
            PROGRAM_SCENE_ID,
            sequence,
            scene_path=order_path,
            image_path=camera_image_path,
            linear_path=linear_path,
            manifest_path=os.path.join(job_dir, "composition_manifest.json"),
            text=text,
            display_name="Live spectral text interface",
            diagnostic_image_path=diagnostic_image_path,
            orthographic_path=flat_reference_path,
            priority_overlay_path=os.path.join(job_dir, "0000_cpp_priority.png"),
            render_log_path=log_path,
            metadata={
                "tier": final_pipeline.tiers[-1].value,
                "layout_render_pipeline": final_pipeline.mapping(),
                "layout_contract": dict(layout_contract or {}),
                "whole_token_policy": final_pipeline.whole_token_policy,
                "final_policy": final_pipeline.final_policy,
                "all_ui_elements_in_place": True,
                "elapsed_s": elapsed_s,
                "camera_profile": {
                    "sensor_white_level": camera_profile.sensor_white_level,
                    "exposure_compensation_ev": camera_profile.exposure_compensation_ev,
                    "tone_curve_mode": camera_profile.tone_curve_mode,
                },
            },
        )
        return RenderedTextRevision(
            sequence=sequence,
            text=text,
            image_path=final.image_path,
            linear_path=final.linear_path,
            manifest_path=final.manifest_path,
            elapsed_s=elapsed_s,
            diagnostic_image_path=final.artifact_paths.get("diagnostic_image", ""),
            orthographic_path=final.artifact_paths.get("orthographic", ""),
            priority_overlay_path=final.artifact_paths.get("priority_overlay", ""),
            interface_assembly_key=final.assembly_key,
            after_render_key=final.render_key,
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


def _initial_window_size(
    display_width: int,
    display_height: int,
    *,
    manifest: Any = None,
) -> tuple[int, int]:
    """Derive the photographed UI scene from its manifest host columns."""

    work_width, work_height = int(display_width), int(display_height)
    if work_width <= 0 or work_height <= 0:
        raise ValueError("scan chunk dimensions must be positive")
    metrics = manifest_program_frame_metrics(
        work_width, work_height, manifest or program_ui_manifest()
    )
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


def _load_work_preview_surface(
    path: str,
    pygame: Any,
    camera_profile: ColorScienceProfile,
) -> Any:
    """Load a completed/job preview for the manifest work viewport."""

    source = str(path)
    if not source or not os.path.isfile(source):
        return None
    if not source.lower().endswith(".npy"):
        return pygame.image.load(source).convert()
    linear = np.asarray(np.load(source, allow_pickle=False), np.float32)[..., :3]
    rgb = np.maximum(linear, 0.0)
    positive = np.mean(rgb, axis=2)
    positive = positive[positive > 0.0]
    white = (
        float(np.percentile(positive, 99.0))
        if positive.size else float(camera_profile.sensor_white_level)
    )
    profile = replace(
        camera_profile, sensor_white_level=max(white, 1.0e-12)
    )
    display = process_linear_sensor_image(linear, profile)
    pixels = np.ascontiguousarray(
        np.clip(display[..., :3], 0.0, 1.0) * 255.0,
        dtype=np.uint8,
    )
    return pygame.surfarray.make_surface(pixels.swapaxes(0, 1))

def _ink_tokens_from_text(text: str) -> tuple[str, ...]:
    """Exact non-whitespace sequences whose characters should seed the atlas."""

    return tuple(dict.fromkeys(str(text).split()))


def _ink_production_plan(
    text: str,
    catalog: RenderAssetCatalog,
    manifest: Any = None,
):
    """Execute the manifest's monofont -> token -> interface ladder."""

    effective_manifest = manifest or _DEFAULT_PROGRAM_MANIFEST
    pipeline = layout_render_pipeline(effective_manifest)
    font = _manifest_render_font(effective_manifest)
    token_strings = tuple(dict.fromkeys(
        (*pipeline.static_tokens, str(text))
    ))
    tokens = tuple(dict.fromkeys((
        *tuple(pipeline.glyph_alphabet),
        *(
            token
            for token_string in token_strings
            for token in _ink_tokens_from_text(token_string)
        ),
    )))
    return plan_ink_atlas_bake(
        tokens,
        catalog,
        token_strings=token_strings,
        font=font,
    )


def _total_scene_is_ready(
    text: str,
    catalog: RenderAssetCatalog,
    manifest: Any = None,
) -> bool:
    return _ink_production_plan(
        text, catalog, manifest
    ).next_request is None


def _fixed_image_alphabet_is_ready(
    catalog: RenderAssetCatalog,
    manifest: Any = None,
) -> bool:
    """Whether the manifest's complete monofont tier has converged."""

    effective_manifest = manifest or _DEFAULT_PROGRAM_MANIFEST
    pipeline = layout_render_pipeline(effective_manifest)
    return plan_ink_atlas_bake(
        tuple(pipeline.glyph_alphabet),
        catalog,
        font=_manifest_render_font(effective_manifest),
    ).next_request is None
def run_window(
    output_root: str,
    initial_text: str,
    display_width: int,
    display_height: int,
    sensor_sweeps: int,
    camera_profile: ColorScienceProfile,
    targeted_fraction: float = DEFAULT_TARGETED_FRACTION,
    priority_model_path: str = "",
    atlas_epochs_per_exposure: int = DEFAULT_ATLAS_EPOCHS_PER_EXPOSURE,
    atlas_sensor_top_k: int = DEFAULT_ATLAS_SENSOR_TOP_K,
    atlas_steps_per_epoch: int = DEFAULT_ATLAS_STEPS_PER_EPOCH,
    atlas_samples_per_node: int = DEFAULT_ATLAS_SAMPLES_PER_NODE,
    atlas_character_horizontal_spacing_px: int = (
        DEFAULT_MONOFONT_HORIZONTAL_SPACING_PX
    ),
    atlas_character_vertical_spacing_px: int = (
        DEFAULT_MONOFONT_VERTICAL_SPACING_PX
    ),
    program_manifest_override: Any = None,
    parent_gl_context: Any = None,
    gl_context_factory: Callable[[Any], Any] | None = None,
) -> int:
    import pygame

    manifest = program_manifest_override or program_ui_manifest()
    scene_width, scene_height = _initial_window_size(
        display_width, display_height, manifest=manifest
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
        "[live] strict production order: fixed-width alphabet cells, tokens, "
        "complete token string, then total UI scene; "
        f"atlas camera rays/burst="
        f"{int(atlas_sensor_top_k) * int(atlas_steps_per_epoch) * int(atlas_samples_per_node):,} "
        f"({int(atlas_steps_per_epoch)} visible/resumable epochs x "
        f"{int(atlas_sensor_top_k)} nodes x "
        f"{int(atlas_samples_per_node)} rays/node)",
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
    from render_work_browser import RenderWorkBrowser

    runtime_layout = resolved_program_ui_layout(
        scene_width,
        scene_height,
        work_width=display_width,
        work_height=display_height,
        manifest=manifest,
    )
    context_host = OpenGLContextHost(
        parent_gl_context, owned_factory=gl_context_factory
    )
    viewport_contexts = {
        key: context_host.acquire(request)
        for key, request in runtime_layout.context_requests.items()
    }
    work_browser = RenderWorkBrowser("WORK / ASSETS")

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
            if event.sensor_sum_path:
                _mark_sensor_display_orientation(event.sensor_sum_path)
            inventory.record_progress(PROGRAM_SCENE_ID, object_ids, event)

    atlas_catalog = RenderAssetCatalog(os.path.join(
        os.path.abspath(output_root), "render_asset_catalog.json"
    ))
    atlas_composer = CachedTokenStringComposer(
        atlas_catalog,
        font=_manifest_render_font(manifest),
        horizontal_spacing_px=atlas_character_horizontal_spacing_px,
        vertical_spacing_px=atlas_character_vertical_spacing_px,
    )
    atlas_renderer = InkAtlasSubprocessRenderer(
        output_root,
        atlas_catalog,
        epochs_per_exposure=atlas_epochs_per_exposure,
        sensor_top_k=atlas_sensor_top_k,
        steps_per_epoch=atlas_steps_per_epoch,
        samples_per_node=atlas_samples_per_node,
    )
    atlas_lock = threading.Lock()
    atlas_text = [str(initial_text)[:500]]

    def next_atlas_request() -> BakeRequest | None:
        with atlas_lock:
            planned_text = atlas_text[0]
        return _ink_production_plan(
            planned_text, atlas_catalog, manifest
        ).next_request

    worker = SpectralTextRenderWorker(
        make_subprocess_renderer(
            output_root, camera_profile, targeted_fraction, priority_model_path,
            object_library=atlas_renderer.object_library,
            layout_pipeline=layout_render_pipeline(manifest),
            layout_contract=runtime_layout.mapping(),
        ),
        progress_observer=retain_progress,
        background_source=next_atlas_request,
        background_render_function=atlas_renderer,
    )

    def accept_selected_visual_pass() -> str:
        payload = work_browser.selected_payload
        kind = str(payload.get("kind", ""))
        if kind == "active" or not payload:
            if atlas_renderer.request_active_visual_pass():
                return "active epoch accepted at its latest checkpoint"
        record_key = ""
        if kind == "asset":
            subtype = payload.get("subtype")
            artifacts = tuple(getattr(subtype, "artifacts", ()))
            if artifacts:
                record_key = max(
                    artifacts,
                    key=lambda item: (
                        int(getattr(item, "samples", 0)),
                        float(getattr(item, "created_at_s", 0.0)),
                    ),
                ).record_key
        elif kind == "job":
            request = payload.get("request")
            if request is not None:
                record = atlas_catalog.find(
                    request.target_key,
                    request.condition,
                    request.product_kind,
                )
                record_key = "" if record is None else record.record_key
        selected_record = next(
            (
                record for record in atlas_catalog.snapshot()
                if record.record_key == record_key
            ),
            None,
        )
        if selected_record is None:
            if atlas_renderer.request_active_visual_pass():
                return "active epoch accepted at its latest checkpoint"
            return "select cached work, or wait for its first epoch checkpoint"
        accepted = atlas_catalog.accept_visual_pass(selected_record)
        try:
            atlas_renderer.object_library.accept_visual_pass(
                accepted.record_key
            )
        except KeyError:
            atlas_renderer.object_library.migrate_catalog(atlas_catalog)
        worker.wake_background()
        return f"accepted cached work {accepted.target_key}"

    text = str(initial_text)[:500]
    cursor = len(text)
    submitted_text = ""
    dirty_at = time.monotonic() - 2.0
    loaded_sequence = 0
    loaded_progress: tuple[str, int] | None = None
    progress_event: ExposureProgressEvent | None = None
    texture = None
    priority_texture = None
    atlas_work_texture = None
    atlas_work_revision = -1
    browser_preview_texture = None
    browser_preview_path = ""
    photographed_region = program_display_region(scene_width, scene_height)
    ui_sensor_regions = {
        key: SensorRegion(*rect)
        for key, rect in runtime_layout.regions.items()
    }
    browser_window_rect = _sensor_product_window_rect(
        ui_sensor_regions["asset-browser"],
        photographed_region,
        scene_width,
        scene_height,
    )
    pre_render_panel_surfaces: list[tuple[tuple[int, int, int, int], Any]] = []
    pre_render_primitives = {
        **runtime_layout.panel_primitives,
        **runtime_layout.action_primitives,
    }
    for panel_id, primitive in pre_render_primitives.items():
        if panel_id not in ui_sensor_regions:
            continue
        region = ui_sensor_regions[panel_id]
        destination = _sensor_product_window_rect(
            region, photographed_region, scene_width, scene_height
        )
        rgba = compose_layout_panel_rgba(
            primitive,
            destination[2],
            destination[3],
        )
        panel_surface = pygame.image.frombuffer(
            rgba.tobytes(), (destination[2], destination[3]), "RGBA"
        ).convert_alpha()
        pre_render_panel_surfaces.append((destination, panel_surface))
    ray_control_hitboxes: dict[str, Any] = {}
    transition_restore: dict[str, str] = {}
    transition_dirty_slice: SensorPixelSlice | None = None
    cached_overlay_key: tuple[Any, ...] | None = None
    cached_overlay_texture = None
    cached_overlay_used: tuple[str, ...] = ()
    cached_layout_text_key: tuple[Any, ...] | None = None
    cached_layout_text_surfaces: list[
        tuple[tuple[int, int, int, int], Any]
    ] = []
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type != pygame.QUIT:
                    browser_action = work_browser.handle_event(
                        event, browser_window_rect
                    )
                    if browser_action is not None:
                        continue
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    action_key = next(
                        (
                            key for key, hitbox in ray_control_hitboxes.items()
                            if hitbox.collidepoint(event.pos)
                        ),
                        "",
                    )
                    if action_key == "window-close":
                        running = False
                    elif action_key == "window-minimize":
                        pygame.display.iconify()
                    elif action_key == "window-maximize":
                        # Acquisition dimensions remain a camera contract.
                        pass
                    elif action_key == "work-visual-pass":
                        print(
                            f"[layout-action] {accept_selected_visual_pass()}",
                            flush=True,
                        )
                    elif action_key == "queue-pause-auto":
                        paused = worker.toggle_background_paused()
                        print(
                            "[layout-action] automatic queue resolution "
                            + ("paused" if paused else "resumed"),
                            flush=True,
                        )
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

            with atlas_lock:
                atlas_text_changed = atlas_text[0] != text
                if atlas_text_changed:
                    atlas_text[0] = text
            if atlas_text_changed:
                worker.wake_background()
            production_plan = _ink_production_plan(
                text, atlas_catalog, manifest
            )
            production_ready = production_plan.next_request is None
            alphabet_ready = _fixed_image_alphabet_is_ready(
                atlas_catalog, manifest
            )

            if (
                text.strip()
                and text != submitted_text
                and production_ready
                and time.monotonic() - dirty_at >= 1.0
            ):
                next_revision = worker.snapshot()[3] + 1
                scene = build_self_rendering_program_scene(
                    text,
                    display_width=scene_width,
                    display_height=scene_height,
                    work_width=display_width,
                    work_height=display_height,
                    status_text="SPECTRAL EXPOSURE ACTIVE",
                    revision=next_revision,
                    program_manifest=manifest,
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
                            if (
                                retained_product.kind is DisplayProductKind.IMAGE
                                and _sensor_sum_has_current_orientation(
                                    retained_product.sensor_sum_path
                                )
                            ):
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
            (
                current_work_revision,
                current_work_path,
                current_work_token,
                current_work_pass,
                current_work_active,
            ) = atlas_renderer.work_snapshot()
            if current_work_revision != atlas_work_revision:
                if current_work_path and os.path.isfile(current_work_path):
                    if current_work_path.lower().endswith(".npy"):
                        linear_work = np.asarray(
                            np.load(current_work_path, allow_pickle=False),
                            np.float32,
                        )[..., :3]
                        work_rgb = np.maximum(linear_work, 0.0)
                        work_luma = np.mean(work_rgb, axis=2)
                        work_positive = work_luma[work_luma > 0.0]
                        work_white = (
                            float(np.percentile(work_positive, 99.0))
                            if work_positive.size
                            else float(camera_profile.sensor_white_level)
                        )
                        work_profile = replace(
                            camera_profile,
                            sensor_white_level=max(work_white, 1.0e-12),
                        )
                        display_work = process_linear_sensor_image(
                            linear_work, work_profile
                        )
                        work_pixels = np.ascontiguousarray(
                            np.clip(display_work[..., :3], 0.0, 1.0) * 255.0,
                            dtype=np.uint8,
                        )
                        atlas_work_texture = pygame.surfarray.make_surface(
                            work_pixels.swapaxes(0, 1)
                        )
                    else:
                        atlas_work_texture = pygame.image.load(
                            current_work_path
                        ).convert()
                    print(
                        "[ink-work] "
                        f"asset={current_work_token!r} "
                        f"pass={current_work_pass} "
                        f"active={current_work_active}",
                        flush=True,
                    )
                elif current_work_active:
                    # Do not present the preceding asset while this one has
                    # started but has not published its first integration yet.
                    atlas_work_texture = None
                atlas_work_revision = current_work_revision
            (
                _object_revision,
                active_work_object_key,
                _object_token,
                _object_pass,
                _object_active,
            ) = atlas_renderer.work_object_snapshot()
            work_browser.sync(
                production_plan.requests,
                atlas_renderer.object_library.snapshot(),
                atlas_renderer.object_library.interface_snapshot(),
                active=(
                    active_work_object_key,
                    current_work_token,
                    current_work_pass,
                    current_work_active,
                ),
            )
            selected_preview_path = work_browser.selected_preview_path
            if selected_preview_path != browser_preview_path:
                browser_preview_texture = None
                if selected_preview_path:
                    try:
                        browser_preview_texture = _load_work_preview_surface(
                            selected_preview_path, pygame, camera_profile
                        )
                    except (OSError, ValueError):
                        browser_preview_texture = None
                browser_preview_path = selected_preview_path
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
            for panel_destination, panel_surface in pre_render_panel_surfaces:
                window.blit(panel_surface, panel_destination[:2])
            if texture is not None and production_ready:
                window.blit(texture, (0, 0))
            editor_destination = _sensor_product_window_rect(
                ui_sensor_regions["editor-text"],
                photographed_region,
                width,
                height,
            )
            catalog_state = atlas_catalog.snapshot()
            catalog_signature = tuple(
                (
                    record.record_key,
                    str(record.metadata.get("sprite_path", "")),
                    int(record.metadata.get(
                        "refinement_pass", record.samples
                    )),
                    str(record.metadata.get("refinement_state", "")),
                    str(record.metadata.get("completion_basis", "")),
                )
                for record in catalog_state
            )
            layout_text_key = (
                tuple(runtime_layout.authored_text.items()),
                tuple(
                    (key, tuple(runtime_layout.regions[key]))
                    for key in runtime_layout.authored_text
                ),
                alphabet_ready,
                catalog_signature,
            )
            if layout_text_key != cached_layout_text_key:
                cached_layout_text_surfaces = []
                for object_id, content in runtime_layout.authored_text.items():
                    destination = _sensor_product_window_rect(
                        ui_sensor_regions[object_id],
                        photographed_region,
                        width,
                        height,
                    )
                    composition = atlas_composer.compose(
                        content,
                        destination[2],
                        destination[3],
                        character_tiles_only=not alphabet_ready,
                    )
                    display_composition = process_linear_sensor_image(
                        composition.linear_rgb, camera_profile
                    )
                    prototype_pixels = np.ascontiguousarray(
                        np.clip(display_composition[..., :3], 0.0, 1.0)
                        * 255.0,
                        dtype=np.uint8,
                    )
                    prototype_surface = pygame.surfarray.make_surface(
                        prototype_pixels.swapaxes(0, 1)
                    ).convert()
                    prototype_surface.set_colorkey(
                        prototype_surface.get_at((0, 0))[:3]
                    )
                    cached_layout_text_surfaces.append((
                        destination, prototype_surface
                    ))
                cached_layout_text_key = layout_text_key
            if latest is None or latest.text != text:
                for destination, prototype_surface in cached_layout_text_surfaces:
                    window.blit(prototype_surface, destination[:2])
            overlay_key = (
                text,
                editor_destination[2],
                editor_destination[3],
                alphabet_ready,
                catalog_signature,
            )
            if overlay_key != cached_overlay_key:
                composition = atlas_composer.compose(
                    text,
                    editor_destination[2],
                    editor_destination[3],
                    character_tiles_only=not alphabet_ready,
                )
                cached_overlay_used = composition.used_tokens
                # A blank composition is still meaningful: it clears stale
                # ray-traced text when the editor becomes empty.
                display_composition = process_linear_sensor_image(
                    composition.linear_rgb, camera_profile
                )
                overlay_pixels = np.ascontiguousarray(
                    np.clip(display_composition[..., :3], 0.0, 1.0)
                    * 255.0,
                    dtype=np.uint8,
                )
                cached_overlay_texture = pygame.surfarray.make_surface(
                    overlay_pixels.swapaxes(0, 1)
                )
                print(
                    "[ink-compose] "
                    f"cached={list(composition.used_tokens)} "
                    f"missing_tokens={list(composition.missing_tokens)} "
                    f"missing_characters={list(composition.missing_characters)}",
                    flush=True,
                )
                cached_overlay_key = overlay_key
            # While the exact full-page revision is unavailable, replace only
            # the editor region with the best cached token/glyph composition.
            if (
                cached_overlay_texture is not None
                and (latest is None or latest.text != text)
            ):
                window.blit(cached_overlay_texture, editor_destination[:2])
            selected_work_texture = (
                browser_preview_texture
                if browser_preview_texture is not None
                else (
                    atlas_work_texture
                    if atlas_work_texture is not None
                    else priority_texture
                )
            )
            panel_products = (
                ("camera-panel", texture),
                ("work-panel", selected_work_texture),
            )
            first_panel_y = min(
                ui_sensor_regions[key].y
                for key in ("camera-panel", "work-panel", "asset-browser")
            )
            header_bottom = max(
                region.y + region.height
                for region in ui_sensor_regions.values()
                if (
                    region.y < first_panel_y
                    and region.y + region.height <= first_panel_y
                )
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
                    (
                        None
                        if panel_id == "work-panel" and (
                            browser_preview_texture is not None
                            or atlas_work_texture is not None
                        )
                        else (
                            None if progress_event is None
                            else progress_event.global_uv_bounds
                        )
                    ),
                    destination[2],
                    destination[3],
                    (
                        current_work_revision
                        if panel_id == "work-panel" and (
                            browser_preview_texture is not None
                            or atlas_work_texture is not None
                        )
                        else (
                            1 if progress_event is None
                            else progress_event.sequence
                        )
                    ),
                )
                if panel_texture is None:
                    continue
                window.blit(panel_texture, destination[:2])

            # The manifest reserves this host; the repository's specialized
            # ScrollableSubpanelList owns all list rendering and interaction.
            work_browser.render(window, browser_window_rect)

            # Controls remain part of the camera-rendered scene. Only their
            # sensor-space rectangles are reused for pointer hit testing.
            ray_control_hitboxes.clear()
            for object_id in runtime_layout.actions:
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
        context_host.close()
        pygame.key.stop_text_input()
        pygame.quit()
    return 0


def clear_camera_storage(output_root: str) -> tuple[str, ...]:
    """Clear this program's exposures, object bakery, and UI inventory."""

    root = os.path.abspath(str(output_root))
    drive, tail = os.path.splitdrive(root)
    if not tail.strip("\\/") or root == os.path.abspath(os.path.expanduser("~")):
        raise ValueError("refusing to clear a filesystem or home-directory root")
    if not os.path.isdir(root):
        return ()

    targets: list[str] = []
    for entry in os.scandir(root):
        if (
            entry.name in {
                "render_objects",
                "render_interfaces",
                "render_asset_catalog.json",
                "render_object_library.json",
            }
            or entry.name in {
                "display_inventory.json",
                "display_inventory.json.tmp",
            }
            or re.fullmatch(r"revision_[0-9]+", entry.name)
        ):
            target = os.path.abspath(entry.path)
            if os.path.commonpath((root, target)) != root or target == root:
                raise ValueError(f"refusing camera-clear target: {target}")
            targets.append(target)

    removed: list[str] = []
    for target in sorted(targets):
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target)
        else:
            os.unlink(target)
        removed.append(target)
    return tuple(removed)


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="exposures/live_spectral_text")
    parser.add_argument(
        "--clear-camera",
        action="store_true",
        help=(
            "Clear this demo's revision exposures, retained display inventory, "
            "and ink/token cache before starting."
        ),
    )
    parser.add_argument(
        "--clear-camera-only",
        action="store_true",
        help="Clear this demo's camera storage and exit without opening a window.",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--display-width", type=int, default=DEFAULT_DISPLAY_WIDTH,
                        help="Exact width of the complete camera-rendered program preview.")
    parser.add_argument("--display-height", type=int, default=DEFAULT_DISPLAY_HEIGHT,
                        help="Exact height of the complete camera-rendered program preview.")
    parser.add_argument("--sensor-sweeps", type=int, default=DEFAULT_SENSOR_SWEEPS,
                        help="Scene-order compatibility value; live foreground work is clarity/cap bounded.")
    parser.add_argument(
        "--atlas-epochs-per-exposure",
        type=int,
        default=DEFAULT_ATLAS_EPOCHS_PER_EXPOSURE,
        help=(
            "Groups of visible/resumable sensor epochs per prepared atlas "
            f"process (default: {DEFAULT_ATLAS_EPOCHS_PER_EXPOSURE}). Each "
            "group contains --atlas-steps-per-epoch one-submission epochs."
        ),
    )
    parser.add_argument(
        "--atlas-sensor-top-k",
        type=int,
        default=DEFAULT_ATLAS_SENSOR_TOP_K,
        help=(
            "Sensor mip nodes selected per atlas refinement submission "
            f"(default: {DEFAULT_ATLAS_SENSOR_TOP_K}, maximum: 1024)."
        ),
    )
    parser.add_argument(
        "--atlas-steps-per-epoch",
        type=int,
        default=DEFAULT_ATLAS_STEPS_PER_EPOCH,
        help=(
            "Visible, atomically checkpointed one-submission epochs in each "
            f"atlas work group (default: {DEFAULT_ATLAS_STEPS_PER_EPOCH})."
        ),
    )
    parser.add_argument(
        "--atlas-samples-per-node",
        type=int,
        default=DEFAULT_ATLAS_SAMPLES_PER_NODE,
        help=(
            "Actual camera rays generated per selected sensor node in each "
            f"native atlas submission (default: {DEFAULT_ATLAS_SAMPLES_PER_NODE})."
        ),
    )
    parser.add_argument(
        "--atlas-character-horizontal-spacing-px",
        "--monofont-horizontal-spacing-px",
        dest="atlas_character_horizontal_spacing_px",
        type=int,
        default=DEFAULT_MONOFONT_HORIZONTAL_SPACING_PX,
        help=(
            "Horizontal pixels added to each worst-case monospace character "
            f"cell (default: {DEFAULT_MONOFONT_HORIZONTAL_SPACING_PX}; "
            "negative values crop)."
        ),
    )
    parser.add_argument(
        "--atlas-character-vertical-spacing-px",
        "--monofont-vertical-spacing-px",
        dest="atlas_character_vertical_spacing_px",
        type=int,
        default=DEFAULT_MONOFONT_VERTICAL_SPACING_PX,
        help=(
            "Vertical pixels added to each worst-case monospace line cell "
            f"(default: {DEFAULT_MONOFONT_VERTICAL_SPACING_PX}; "
            "negative values crop)."
        ),
    )
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
    if int(args.atlas_epochs_per_exposure) <= 0:
        raise SystemExit("--atlas-epochs-per-exposure must be positive")
    if not 64 <= int(args.atlas_sensor_top_k) <= 1024:
        raise SystemExit("--atlas-sensor-top-k must be in [64, 1024]")
    if int(args.atlas_steps_per_epoch) <= 0:
        raise SystemExit("--atlas-steps-per-epoch must be positive")
    if not 1 <= int(args.atlas_samples_per_node) <= 1024:
        raise SystemExit("--atlas-samples-per-node must be in [1, 1024]")
    if args.clear_camera or args.clear_camera_only:
        removed = clear_camera_storage(args.out_dir)
        print(
            f"[camera-clear] root={os.path.abspath(args.out_dir)} "
            f"removed={len(removed)}",
            flush=True,
        )
        for path in removed:
            print(f"[camera-clear] removed {path}", flush=True)
    if args.clear_camera_only:
        return 0
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
        args.atlas_epochs_per_exposure,
        args.atlas_sensor_top_k,
        args.atlas_steps_per_epoch,
        args.atlas_samples_per_node,
        args.atlas_character_horizontal_spacing_px,
        args.atlas_character_vertical_spacing_px,
    )


if __name__ == "__main__":
    raise SystemExit(main())
