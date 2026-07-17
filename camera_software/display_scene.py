"""Persistent contracts for progressively rendered display geometry.

A display scene owns one fixed physical camera.  Every object is authored in
that scene's world coordinates, so perspective, focus, and lens distortion are
shared consequences of one exposure rather than per-object presentation
effects.  Schedulers may prioritize sensor regions, but never reframe objects
with private cameras.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Iterable

from .progressive_exposure import (
    ExposureProgressEvent,
    SensorPixelSlice,
    SensorRegion,
)


DISPLAY_SCENE_SCHEMA_VERSION = 1


def _finite_tuple(value: Iterable[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(component) for component in value)
    if len(result) != size or not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


class DisplayPrimitiveKind(str, Enum):
    TEXT = "text"
    BOX = "box"
    ICON = "icon"


class DisplayProductKind(str, Enum):
    IMAGE = "image"
    LIGHT_FIELD = "light_field"


class DisplayObjectStatus(str, Enum):
    AUTHORED = "authored"
    DIRTY = "dirty"
    EXPOSING = "exposing"
    PLACED = "placed"
    FAILED = "failed"


class BevelProfile(str, Enum):
    SQUARE = "square"
    CHAMFER = "chamfer"


class LayoutSeam(str, Enum):
    BEVEL = "bevel"
    FLUSH = "flush"


@dataclass(frozen=True)
class BevelRegion:
    width_m: float = 0.0
    depth_m: float = 0.0
    profile: BevelProfile = BevelProfile.SQUARE

    def __post_init__(self) -> None:
        if self.width_m < 0.0 or self.depth_m < 0.0:
            raise ValueError("bevel width and depth must be non-negative")
        if (self.width_m == 0.0) != (self.depth_m == 0.0):
            raise ValueError("bevel width and depth must both be zero or both positive")


@dataclass(frozen=True)
class LayoutRectangle:
    """Sensor allocation plus physical edge treatment for a UI surface.

    Seam order is left, right, bottom, top. A flush edge keeps the front face
    square to meet an adjacent rectangle; a bevel edge consumes the authored
    bevel band and slopes toward the surface wall.
    """

    sensor_region: SensorRegion
    bevel: BevelRegion = BevelRegion()
    seams: tuple[LayoutSeam, LayoutSeam, LayoutSeam, LayoutSeam] = (
        LayoutSeam.FLUSH,
        LayoutSeam.FLUSH,
        LayoutSeam.FLUSH,
        LayoutSeam.FLUSH,
    )

    def __post_init__(self) -> None:
        if len(self.seams) != 4:
            raise ValueError("layout rectangle requires left/right/bottom/top seams")
        object.__setattr__(self, "seams", tuple(LayoutSeam(item) for item in self.seams))


MANAGEMENT_ICON_GLYPHS = {
    "close": "×",
    "minimize": "−",
    "maximize": "□",
    "menu": "≡",
    "play": "▶",
    "pause": "Ⅱ",
    "settings": "⚙",
}


@dataclass(frozen=True)
class FixedSceneCamera:
    position_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    focus_target_m: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    focal_mm: float = 35.0
    aperture_mm: float = 25.0

    def __post_init__(self) -> None:
        position = _finite_tuple(self.position_m, 3, "position_m")
        target = _finite_tuple(self.target_m, 3, "target_m")
        focus = _finite_tuple(self.focus_target_m, 3, "focus_target_m")
        up = _finite_tuple(self.up, 3, "up")
        if position == target:
            raise ValueError("fixed scene camera position and target must differ")
        if self.focal_mm <= 0.0 or self.aperture_mm <= 0.0:
            raise ValueError("camera focal length and aperture must be positive")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "target_m", target)
        object.__setattr__(self, "focus_target_m", focus)
        object.__setattr__(self, "up", up)


@dataclass(frozen=True)
class WorldPlacement:
    center_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    up: tuple[float, float, float]
    size_m: tuple[float, float]
    thickness_m: float = 0.01
    layer: int = 0

    def __post_init__(self) -> None:
        center = _finite_tuple(self.center_m, 3, "center_m")
        normal = _finite_tuple(self.normal, 3, "normal")
        up = _finite_tuple(self.up, 3, "up")
        size = _finite_tuple(self.size_m, 2, "size_m")
        dot = sum(a * b for a, b in zip(normal, up))
        normal_len = math.sqrt(sum(value * value for value in normal))
        up_len = math.sqrt(sum(value * value for value in up))
        if normal_len <= 1.0e-12 or up_len <= 1.0e-12:
            raise ValueError("placement normal and up must be nonzero")
        if abs(dot / (normal_len * up_len)) >= 1.0 - 1.0e-8:
            raise ValueError("placement normal and up must not be parallel")
        if any(value <= 0.0 for value in size) or self.thickness_m <= 0.0:
            raise ValueError("placement size and thickness must be positive")
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "size_m", size)


@dataclass(frozen=True)
class DisplayPrimitive:
    kind: DisplayPrimitiveKind
    content: str = ""
    icon_name: str = ""
    label: str = ""

    def authored_text(self) -> str:
        if self.kind is DisplayPrimitiveKind.TEXT:
            text = self.content
        elif self.kind is DisplayPrimitiveKind.BOX:
            # A box is physical layout geometry. Its optional label is metadata,
            # not an instruction to manufacture a glyph on its face.
            return self.label.strip()
        else:
            try:
                glyph = MANAGEMENT_ICON_GLYPHS[self.icon_name]
            except KeyError as exc:
                raise ValueError(f"unsupported management icon {self.icon_name!r}") from exc
            text = f"{glyph} {self.label}".rstrip()
        if not text.strip():
            raise ValueError("display primitive must produce non-empty authored geometry")
        return text


@dataclass(frozen=True)
class DisplayProductRequest:
    kind: DisplayProductKind
    sensor_region: SensorRegion | None = None
    sensor_pixel_slice: SensorPixelSlice | None = None
    zoom_level: int = 0
    minimum_passes: int = 1

    def __post_init__(self) -> None:
        if self.zoom_level < 0 or self.minimum_passes <= 0:
            raise ValueError("product zoom must be non-negative and passes positive")


@dataclass(frozen=True)
class DisplayObjectSpec:
    object_id: str
    primitive: DisplayPrimitive
    placement: WorldPlacement
    products: tuple[DisplayProductRequest, ...] = (
        DisplayProductRequest(DisplayProductKind.IMAGE),
    )
    layout_rectangle: LayoutRectangle | None = None
    material: str = "text_surface"
    surface_material: str = "quiet_background"
    font_family: str = "DejaVu Sans"
    font_weight: str = "bold"
    font_style: str = "normal"
    horizontal_align: str = "center"
    vertical_align: str = "center"
    revision: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        object_id = str(self.object_id).strip()
        if not object_id:
            raise ValueError("display object id must be non-empty")
        if self.revision <= 0 or not self.products:
            raise ValueError("display object revision and product list must be positive")
        if self.primitive.kind is not DisplayPrimitiveKind.BOX:
            self.primitive.authored_text()
        horizontal_align = str(self.horizontal_align).strip().lower()
        vertical_align = str(self.vertical_align).strip().lower()
        if horizontal_align not in {"left", "center", "right"}:
            raise ValueError("horizontal alignment must be left, center, or right")
        if vertical_align not in {"bottom", "center", "top"}:
            raise ValueError("vertical alignment must be bottom, center, or top")
        if self.layout_rectangle is not None:
            regions = {
                request.sensor_region for request in self.products
                if request.sensor_region is not None
            }
            if self.layout_rectangle.sensor_region not in regions:
                raise ValueError(
                    "layout rectangle sensor region must be one of the object's products"
                )
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "products", tuple(self.products))
        object.__setattr__(self, "horizontal_align", horizontal_align)
        object.__setattr__(self, "vertical_align", vertical_align)


@dataclass(frozen=True)
class DisplaySceneSpec:
    scene_id: str
    camera: FixedSceneCamera
    sensor_width: int
    sensor_height: int
    objects: tuple[DisplayObjectSpec, ...]
    revision: int = 1

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        if not scene_id or self.sensor_width <= 0 or self.sensor_height <= 0:
            raise ValueError("scene id and sensor dimensions must be valid")
        if not self.objects:
            raise ValueError("a display scene requires at least one object")
        ids = [item.object_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("display object ids must be unique inside a scene")
        for spec in self.objects:
            for request in spec.products:
                region = request.sensor_region
                if region is not None and (
                    region.x + region.width > self.sensor_width
                    or region.y + region.height > self.sensor_height
                ):
                    raise ValueError(
                        f"display object {spec.object_id!r} product region "
                        "must fit within the scene sensor"
                    )
                pixel_slice = request.sensor_pixel_slice
                if pixel_slice is not None and (
                    pixel_slice.width != self.sensor_width
                    or pixel_slice.height != self.sensor_height
                ):
                    raise ValueError(
                        f"display object {spec.object_id!r} pixel slice dimensions "
                        "must match the scene sensor"
                    )
        if self.revision <= 0:
            raise ValueError("scene revision must be positive")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "objects", tuple(self.objects))


@dataclass(frozen=True)
class DisplayProductState:
    kind: DisplayProductKind
    pass_index: int = 0
    sequence: int = 0
    linear_path: str = ""
    sensor_sum_path: str = ""
    exposure_weight_path: str = ""
    preview_path: str = ""
    previous_linear_path: str = ""
    previous_sensor_sum_path: str = ""
    previous_exposure_weight_path: str = ""
    dirty_pixel_slice: SensorPixelSlice | None = None
    replacement_confidence: float = 1.0
    updated_at_s: float = 0.0


@dataclass(frozen=True)
class DisplayObjectState:
    object_id: str
    authored_revision: int
    status: DisplayObjectStatus = DisplayObjectStatus.AUTHORED
    products: tuple[DisplayProductState, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class DisplaySceneState:
    scene: DisplaySceneSpec
    objects: tuple[DisplayObjectState, ...]


@dataclass(frozen=True)
class SceneWorkLease:
    scene_id: str
    object_ids: tuple[str, ...]
    sensor_region: SensorRegion
    work_units: int
    issued_at_s: float
    sensor_pixel_slices: tuple[SensorPixelSlice, ...] = ()


class DisplaySceneInventory:
    """Thread-safe retained scene/object/product inventory with atomic JSON saves."""

    def __init__(self, path: str = "") -> None:
        self.path = os.path.abspath(path) if path else ""
        self._lock = threading.RLock()
        self._scenes: dict[str, DisplaySceneState] = {}
        if self.path and os.path.isfile(self.path):
            self._load()

    def put_scene(self, scene: DisplaySceneSpec) -> None:
        with self._lock:
            previous = self._scenes.get(scene.scene_id)
            old_states = {
                state.object_id: state for state in previous.objects
            } if previous is not None else {}
            states = []
            for spec in scene.objects:
                prior = old_states.get(spec.object_id)
                if prior is None or prior.authored_revision != spec.revision:
                    product_kinds = sorted(
                        {request.kind for request in spec.products},
                        key=lambda item: item.value,
                    )
                    previous_products = {
                        product.kind: product
                        for product in (() if prior is None else prior.products)
                    }
                    prior = DisplayObjectState(
                        spec.object_id,
                        spec.revision,
                        status=(
                            DisplayObjectStatus.AUTHORED
                            if not previous_products else DisplayObjectStatus.DIRTY
                        ),
                        products=tuple(
                            DisplayProductState(
                                kind,
                                previous_linear_path=previous_products.get(
                                    kind, DisplayProductState(kind)
                                ).linear_path,
                                previous_sensor_sum_path=previous_products.get(
                                    kind, DisplayProductState(kind)
                                ).sensor_sum_path,
                                previous_exposure_weight_path=previous_products.get(
                                    kind, DisplayProductState(kind)
                                ).exposure_weight_path,
                                replacement_confidence=0.0,
                            )
                            for kind in product_kinds
                        ),
                    )
                states.append(prior)
            self._scenes[scene.scene_id] = DisplaySceneState(scene, tuple(states))
            self._save()

    def remove_scene(self, scene_id: str) -> None:
        with self._lock:
            self._scenes.pop(str(scene_id), None)
            self._save()

    def mark_dirty(
        self,
        scene_id: str,
        object_ids: Iterable[str],
        pixel_slices: dict[str, SensorPixelSlice] | None = None,
    ) -> None:
        """Invalidate only selected object products; retain every other exposure."""

        selected = set(map(str, object_ids))
        with self._lock:
            current = self._scenes[str(scene_id)]
            states = tuple(
                replace(
                    state,
                    status=DisplayObjectStatus.DIRTY,
                    products=tuple(
                        replace(
                            product,
                            pass_index=0,
                            sequence=0,
                            previous_linear_path=(
                                product.linear_path or product.previous_linear_path
                            ),
                            previous_sensor_sum_path=(
                                product.sensor_sum_path
                                or product.previous_sensor_sum_path
                            ),
                            previous_exposure_weight_path=(
                                product.exposure_weight_path
                                or product.previous_exposure_weight_path
                            ),
                            linear_path="",
                            sensor_sum_path="",
                            exposure_weight_path="",
                            preview_path="",
                            dirty_pixel_slice=(
                                None if pixel_slices is None
                                else pixel_slices.get(state.object_id)
                            ),
                            replacement_confidence=0.0,
                        )
                        for product in state.products
                    ),
                    error="",
                )
                if state.object_id in selected else state
                for state in current.objects
            )
            self._scenes[str(scene_id)] = replace(current, objects=states)
            self._save()

    def snapshot(self) -> tuple[DisplaySceneState, ...]:
        with self._lock:
            return tuple(self._scenes[key] for key in sorted(self._scenes))

    def record_progress(
        self,
        scene_id: str,
        object_ids: Iterable[str],
        event: ExposureProgressEvent,
        kind: DisplayProductKind = DisplayProductKind.IMAGE,
    ) -> None:
        selected = set(map(str, object_ids))
        with self._lock:
            current = self._scenes[str(scene_id)]
            states = []
            for state in current.objects:
                if state.object_id not in selected:
                    states.append(state)
                    continue
                products = {product.kind: product for product in state.products}
                products[kind] = DisplayProductState(
                    kind=kind,
                    pass_index=event.pass_index,
                    sequence=event.sequence,
                    linear_path=event.linear_accumulation_path,
                    sensor_sum_path=event.sensor_sum_path,
                    exposure_weight_path=event.exposure_weight_path,
                    preview_path=event.preview_path,
                    previous_linear_path=products.get(
                        kind, DisplayProductState(kind)
                    ).previous_linear_path,
                    previous_sensor_sum_path=products.get(
                        kind, DisplayProductState(kind)
                    ).previous_sensor_sum_path,
                    previous_exposure_weight_path=products.get(
                        kind, DisplayProductState(kind)
                    ).previous_exposure_weight_path,
                    dirty_pixel_slice=products.get(
                        kind, DisplayProductState(kind)
                    ).dirty_pixel_slice,
                    replacement_confidence=min(
                        1.0,
                        float(event.completed_work)
                        / max(1.0, float(event.total_work or event.pass_index or 1)),
                    ),
                    updated_at_s=time.time(),
                )
                states.append(replace(
                    state,
                    status=(
                        DisplayObjectStatus.PLACED
                        if event.kind.value == "completed"
                        else DisplayObjectStatus.FAILED
                        if event.kind.value == "failed"
                        else DisplayObjectStatus.EXPOSING
                    ),
                    products=tuple(products[key] for key in sorted(products, key=lambda item: item.value)),
                    error=event.message if event.kind.value == "failed" else "",
                ))
            self._scenes[str(scene_id)] = replace(current, objects=tuple(states))
            self._save()

    def _save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {
            "schema_version": DISPLAY_SCENE_SCHEMA_VERSION,
            "scenes": [_scene_state_payload(state) for state in self.snapshot()],
        }
        temp = self.path + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temp, self.path)

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("schema_version", -1)) != DISPLAY_SCENE_SCHEMA_VERSION:
            raise ValueError("unsupported display scene inventory schema")
        self._scenes = {
            state.scene.scene_id: state
            for state in (_scene_state_from_payload(item) for item in payload.get("scenes", ()))
        }


class DisplaySceneWorkScheduler:
    """Fair scene-level scheduler; leases never change a scene's fixed camera."""

    def __init__(self, inventory: DisplaySceneInventory) -> None:
        self.inventory = inventory
        self._cursor = 0

    def next_lease(self, work_units: int = 1) -> SceneWorkLease | None:
        scenes = [state for state in self.inventory.snapshot() if state.scene.objects]
        if not scenes:
            return None
        scene_state = scenes[self._cursor % len(scenes)]
        self._cursor += 1
        candidates = [
            (spec, next(
                state for state in scene_state.objects if state.object_id == spec.object_id
            ))
            for spec in scene_state.scene.objects if spec.enabled
        ]
        if not candidates:
            return None
        refinement = {
            spec.object_id: _object_refinement_pass(spec, state)
            for spec, state in candidates
        }
        minimum_pass = min(refinement.values())
        selected = [
            spec for spec, state in candidates
            if refinement[spec.object_id] == minimum_pass
        ]
        regions = [
            request.sensor_region
            for spec in selected for request in spec.products
            if request.sensor_region is not None
        ]
        pixel_slices = tuple(
            request.sensor_pixel_slice
            for spec in selected for request in spec.products
            if request.sensor_pixel_slice is not None
        )
        region = _union_regions(
            regions,
            scene_state.scene.sensor_width,
            scene_state.scene.sensor_height,
        )
        return SceneWorkLease(
            scene_id=scene_state.scene.scene_id,
            object_ids=tuple(spec.object_id for spec in selected),
            sensor_region=region,
            sensor_pixel_slices=pixel_slices,
            work_units=max(1, int(work_units)),
            issued_at_s=time.time(),
        )


def _object_refinement_pass(
    spec: DisplayObjectSpec, state: DisplayObjectState
) -> int:
    passes = {product.kind: product.pass_index for product in state.products}
    return min(passes.get(request.kind, 0) for request in spec.products)


def _union_regions(
    regions: Iterable[SensorRegion], sensor_width: int, sensor_height: int
) -> SensorRegion:
    values = tuple(regions)
    if not values:
        return SensorRegion(0, 0, sensor_width, sensor_height)
    x0 = min(region.x for region in values)
    y0 = min(region.y for region in values)
    x1 = max(region.x + region.width for region in values)
    y1 = max(region.y + region.height for region in values)
    return SensorRegion(x0, y0, x1 - x0, y1 - y0)


def _scene_state_payload(state: DisplaySceneState) -> dict:
    payload = asdict(state)
    payload["scene"]["camera"] = asdict(state.scene.camera)
    for raw, spec in zip(payload["scene"]["objects"], state.scene.objects):
        raw["primitive"]["kind"] = spec.primitive.kind.value
        for product, request in zip(raw["products"], spec.products):
            product["kind"] = request.kind.value
    for raw, object_state in zip(payload["objects"], state.objects):
        raw["status"] = object_state.status.value
        for product, product_state in zip(raw["products"], object_state.products):
            product["kind"] = product_state.kind.value
    return payload


def _scene_state_from_payload(payload: dict) -> DisplaySceneState:
    raw_scene = dict(payload["scene"])
    raw_scene["camera"] = FixedSceneCamera(**raw_scene["camera"])
    specs = []
    for raw in raw_scene.get("objects", ()):
        item = dict(raw)
        primitive = dict(item["primitive"])
        primitive["kind"] = DisplayPrimitiveKind(primitive["kind"])
        item["primitive"] = DisplayPrimitive(**primitive)
        item["placement"] = WorldPlacement(**item["placement"])
        if item.get("layout_rectangle") is not None:
            rectangle = dict(item["layout_rectangle"])
            rectangle["sensor_region"] = SensorRegion(**rectangle["sensor_region"])
            raw_bevel = dict(rectangle.get("bevel", {}))
            raw_bevel["profile"] = BevelProfile(
                raw_bevel.get("profile", BevelProfile.SQUARE.value)
            )
            rectangle["bevel"] = BevelRegion(**raw_bevel)
            rectangle["seams"] = tuple(
                LayoutSeam(value) for value in rectangle.get("seams", ())
            )
            item["layout_rectangle"] = LayoutRectangle(**rectangle)
        item["products"] = tuple(
            DisplayProductRequest(
                kind=DisplayProductKind(product["kind"]),
                sensor_region=(
                    None if product.get("sensor_region") is None
                    else SensorRegion(**product["sensor_region"])
                ),
                sensor_pixel_slice=(
                    None if product.get("sensor_pixel_slice") is None
                    else SensorPixelSlice(**product["sensor_pixel_slice"])
                ),
                zoom_level=int(product.get("zoom_level", 0)),
                minimum_passes=int(product.get("minimum_passes", 1)),
            )
            for product in item.get("products", ())
        )
        specs.append(DisplayObjectSpec(**item))
    raw_scene["objects"] = tuple(specs)
    scene = DisplaySceneSpec(**raw_scene)
    states = []
    for raw in payload.get("objects", ()):
        item = dict(raw)
        item["status"] = DisplayObjectStatus(item["status"])
        item["products"] = tuple(
            DisplayProductState(
                **{
                    **product,
                    "kind": DisplayProductKind(product["kind"]),
                    "dirty_pixel_slice": (
                        None if product.get("dirty_pixel_slice") is None
                        else SensorPixelSlice(**product["dirty_pixel_slice"])
                    ),
                }
            )
            for product in item.get("products", ())
        )
        states.append(DisplayObjectState(**item))
    return DisplaySceneState(scene=scene, objects=tuple(states))
