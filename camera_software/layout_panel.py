"""Nine-slice layout panel primitive for arbitrary pre-render window sizes."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

import numpy as np


class PanelPatchRole(str, Enum):
    CENTER = "center"
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_LEFT = "bottom_left"


class PanelPatchFill(str, Enum):
    STRETCH = "stretch"
    TILE = "tile"


@dataclass(frozen=True)
class PanelCropVectors:
    """Normalized vector pair selecting material from a square photograph."""

    minimum_uv: tuple[float, float] = (0.0, 0.0)
    maximum_uv: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        minimum = tuple(float(value) for value in self.minimum_uv)
        maximum = tuple(float(value) for value in self.maximum_uv)
        if len(minimum) != 2 or len(maximum) != 2:
            raise ValueError("panel crop requires two 2D vectors")
        if any(value < 0.0 or value > 1.0 for value in (*minimum, *maximum)):
            raise ValueError("panel crop vectors must be normalized")
        if minimum[0] >= maximum[0] or minimum[1] >= maximum[1]:
            raise ValueError("panel crop maximum must exceed its minimum")
        object.__setattr__(self, "minimum_uv", minimum)
        object.__setattr__(self, "maximum_uv", maximum)

    def mapping(self) -> list[list[float]]:
        return [list(self.minimum_uv), list(self.maximum_uv)]


@dataclass(frozen=True)
class PanelPatchObjectRequest:
    """Renderer-neutral request for one panel subtype material patch."""

    role: PanelPatchRole
    object_key: str
    subtype_key: str
    crop_vectors: PanelCropVectors
    fill: PanelPatchFill
    condition_key: str = ""
    representative_square: bool = True

    def mapping(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "object_key": self.object_key,
            "subtype_key": self.subtype_key,
            "condition_key": self.condition_key,
            "source_capture": "representative_square",
            "crop_vectors": self.crop_vectors.mapping(),
            "fill": self.fill.value,
        }


@dataclass(frozen=True)
class PanelPatchAsset:
    """One independently renderable image/light-field panel component."""

    role: PanelPatchRole
    asset_key: str = ""
    image_path: str = ""
    object_key: str = ""
    subtype_key: str = ""
    condition_key: str = ""
    crop_vectors: PanelCropVectors = PanelCropVectors()
    representative_square: bool = True
    color_rgba: tuple[int, int, int, int] = (32, 35, 42, 255)
    fill: PanelPatchFill = PanelPatchFill.STRETCH

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", PanelPatchRole(self.role))
        object.__setattr__(self, "fill", PanelPatchFill(self.fill))
        crop = (
            self.crop_vectors
            if isinstance(self.crop_vectors, PanelCropVectors)
            else PanelCropVectors(*self.crop_vectors)
        )
        object.__setattr__(self, "crop_vectors", crop)
        color = tuple(int(value) for value in self.color_rgba)
        if len(color) != 4 or any(value < 0 or value > 255 for value in color):
            raise ValueError("panel patch color must be four uint8 values")
        object.__setattr__(self, "color_rgba", color)


@dataclass(frozen=True)
class LayoutPanelPrimitive:
    """Four corners, four sides, and center assembled at requested size."""

    primitive_id: str
    patches: tuple[PanelPatchAsset, ...]
    border_px: tuple[int, int, int, int] = (12, 12, 12, 12)
    revision: int = 1

    def __post_init__(self) -> None:
        if not str(self.primitive_id).strip():
            raise ValueError("layout panel primitive id must be non-empty")
        border = tuple(int(value) for value in self.border_px)
        if len(border) != 4 or any(value < 0 for value in border):
            raise ValueError("panel border is left/top/right/bottom non-negative pixels")
        roles = [PanelPatchRole(patch.role) for patch in self.patches]
        if len(roles) != len(set(roles)):
            raise ValueError("panel patch roles must be unique")
        if self.revision <= 0:
            raise ValueError("panel primitive revision must be positive")
        object.__setattr__(self, "border_px", border)
        object.__setattr__(self, "patches", tuple(self.patches))

    @property
    def patch_map(self) -> Mapping[PanelPatchRole, PanelPatchAsset]:
        return {PanelPatchRole(patch.role): patch for patch in self.patches}

    @property
    def object_requests(self) -> tuple[PanelPatchObjectRequest, ...]:
        return tuple(
            PanelPatchObjectRequest(
                role=patch.role,
                object_key=patch.object_key,
                subtype_key=patch.subtype_key,
                condition_key=patch.condition_key,
                crop_vectors=patch.crop_vectors,
                fill=patch.fill,
                representative_square=patch.representative_square,
            )
            for patch in self.patches
            if patch.object_key and patch.subtype_key
        )

    def mapping(self) -> dict[str, Any]:
        return {
            "kind": "nine_slice",
            "primitive_id": self.primitive_id,
            "border_px": list(self.border_px),
            "revision": self.revision,
            "object_requests": [
                request.mapping() for request in self.object_requests
            ],
            "patches": [
                {
                    "role": patch.role.value,
                    "asset_key": patch.asset_key,
                    "image_path": patch.image_path,
                    "object_key": patch.object_key,
                    "subtype_key": patch.subtype_key,
                    "condition_key": patch.condition_key,
                    "source_capture": (
                        "representative_square"
                        if patch.representative_square else "unspecified"
                    ),
                    "crop_vectors": patch.crop_vectors.mapping(),
                    "color_rgba": list(patch.color_rgba),
                    "fill": patch.fill.value,
                }
                for patch in self.patches
            ],
        }


@dataclass(frozen=True)
class PanelCompositionPatchTrace:
    """One semantic patch placement shared by raster and scene assembly."""

    role: PanelPatchRole
    target_rect_px: tuple[int, int, int, int]
    crop_vectors: PanelCropVectors
    fill: PanelPatchFill
    object_key: str = ""
    subtype_key: str = ""
    condition_key: str = ""

    def mapping(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "target_rect_px": list(self.target_rect_px),
            "crop_vectors": self.crop_vectors.mapping(),
            "fill": self.fill.value,
            "object_key": self.object_key,
            "subtype_key": self.subtype_key,
            "condition_key": self.condition_key,
            "uv_transform": "repeat_square_tiles_clip_partial_terminal_tile",
        }


@dataclass(frozen=True)
class PanelCompositionTrace:
    primitive_id: str
    output_size_px: tuple[int, int]
    effective_border_px: tuple[int, int, int, int]
    patches: tuple[PanelCompositionPatchTrace, ...]

    def mapping(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "output_size_px": list(self.output_size_px),
            "effective_border_px": list(self.effective_border_px),
            "patches": [patch.mapping() for patch in self.patches],
        }


def _representative_square_crop(role: PanelPatchRole) -> PanelCropVectors:
    cells = {
        PanelPatchRole.TOP_LEFT: (0, 0),
        PanelPatchRole.TOP: (1, 0),
        PanelPatchRole.TOP_RIGHT: (2, 0),
        PanelPatchRole.LEFT: (0, 1),
        PanelPatchRole.CENTER: (1, 1),
        PanelPatchRole.RIGHT: (2, 1),
        PanelPatchRole.BOTTOM_LEFT: (0, 2),
        PanelPatchRole.BOTTOM: (1, 2),
        PanelPatchRole.BOTTOM_RIGHT: (2, 2),
    }
    column, row = cells[PanelPatchRole(role)]
    return PanelCropVectors(
        (column / 3.0, row / 3.0),
        ((column + 1) / 3.0, (row + 1) / 3.0),
    )


def layout_panel_primitive_from_mapping(
    value: Mapping[str, Any] | None,
    *,
    primitive_id: str,
) -> LayoutPanelPrimitive:
    raw = dict(value or {})
    if str(raw.get("kind", "nine_slice")) != "nine_slice":
        raise ValueError("layout panel primitive kind must be nine_slice")
    raw_patches = list(raw.get("patches", ()) or ())
    if raw_patches:
        patches = tuple(
            PanelPatchAsset(
                role=PanelPatchRole(str(item["role"])),
                asset_key=str(item.get("asset_key", "")),
                image_path=str(item.get("image_path", "")),
                object_key=str(item.get("object_key", "")),
                subtype_key=str(item.get("subtype_key", "")),
                condition_key=str(item.get("condition_key", "")),
                crop_vectors=PanelCropVectors(*item.get(
                    "crop_vectors",
                    _representative_square_crop(
                        PanelPatchRole(str(item["role"]))
                    ).mapping(),
                )),
                representative_square=(
                    str(item.get(
                        "source_capture", "representative_square"
                    )) == "representative_square"
                ),
                color_rgba=tuple(item.get("color_rgba", (32, 35, 42, 255))),
                fill=PanelPatchFill(str(item.get("fill", "stretch"))),
            )
            for item in raw_patches
        )
    else:
        edge = tuple(raw.get("edge_rgba", (48, 53, 65, 255)))
        corner = tuple(raw.get("corner_rgba", (62, 69, 84, 255)))
        center = tuple(raw.get("center_rgba", (28, 31, 38, 255)))
        object_key = str(raw.get("object_key", ""))
        subtype_key = str(raw.get("subtype_key", "representative-square"))
        condition_key = str(raw.get("condition_key", ""))
        raw_crops = dict(raw.get("crop_vectors", {}) or {})
        patches = tuple(
            PanelPatchAsset(
                role=role,
                object_key=object_key,
                subtype_key=subtype_key if object_key else "",
                condition_key=condition_key,
                crop_vectors=PanelCropVectors(*raw_crops.get(
                    role.value,
                    _representative_square_crop(role).mapping(),
                )),
                representative_square=True,
                color_rgba=(
                    center if role is PanelPatchRole.CENTER
                    else corner if "_" in role.value
                    else edge
                ),
                fill=(
                    PanelPatchFill.TILE
                    if role in {
                        PanelPatchRole.TOP, PanelPatchRole.RIGHT,
                        PanelPatchRole.BOTTOM, PanelPatchRole.LEFT,
                    }
                    else PanelPatchFill.STRETCH
                ),
            )
            for role in PanelPatchRole
        )
    return LayoutPanelPrimitive(
        primitive_id=str(raw.get("primitive_id", primitive_id)),
        patches=patches,
        border_px=tuple(raw.get("border_px", (12, 12, 12, 12))),
        revision=int(raw.get("revision", 1)),
    )


def _effective_borders(
    width: int,
    height: int,
    border: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = border
    horizontal = left + right
    vertical = top + bottom
    if horizontal > width and horizontal > 0:
        left = int(round(width * left / horizontal))
        right = width - left
    if vertical > height and vertical > 0:
        top = int(round(height * top / vertical))
        bottom = height - top
    return left, top, right, bottom


def layout_panel_composition_trace(
    primitive: LayoutPanelPrimitive,
    width: int,
    height: int,
) -> PanelCompositionTrace:
    """Resolve exact nine-slice geometry without erasing its semantics."""

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("composed panel size must be positive")
    border = _effective_borders(width, height, primitive.border_px)
    left, top, right, bottom = border
    x = (0, left, width - right, width)
    y = (0, top, height - bottom, height)
    cells = {
        PanelPatchRole.TOP_LEFT: (0, 0),
        PanelPatchRole.TOP: (1, 0),
        PanelPatchRole.TOP_RIGHT: (2, 0),
        PanelPatchRole.LEFT: (0, 1),
        PanelPatchRole.CENTER: (1, 1),
        PanelPatchRole.RIGHT: (2, 1),
        PanelPatchRole.BOTTOM_LEFT: (0, 2),
        PanelPatchRole.BOTTOM: (1, 2),
        PanelPatchRole.BOTTOM_RIGHT: (2, 2),
    }
    patches = []
    for role, (column, row) in cells.items():
        patch = primitive.patch_map.get(role)
        if patch is None:
            continue
        patches.append(PanelCompositionPatchTrace(
            role=role,
            target_rect_px=(
                x[column], y[row],
                max(0, x[column + 1] - x[column]),
                max(0, y[row + 1] - y[row]),
            ),
            crop_vectors=patch.crop_vectors,
            fill=patch.fill,
            object_key=patch.object_key,
            subtype_key=patch.subtype_key,
            condition_key=patch.condition_key,
        ))
    return PanelCompositionTrace(
        primitive.primitive_id, (width, height), border, tuple(patches)
    )


def compose_layout_panel_rgba(
    primitive: LayoutPanelPrimitive,
    width: int,
    height: int,
    *,
    image_loader: Callable[[PanelPatchAsset], np.ndarray | None] | None = None,
) -> np.ndarray:
    """Compose a nine-slice panel at any positive requested pixel size."""

    from PIL import Image

    trace = layout_panel_composition_trace(primitive, width, height)
    width, height = trace.output_size_px
    targets = {
        item.role: (
            item.target_rect_px[0], item.target_rect_px[1],
            item.target_rect_px[0] + item.target_rect_px[2],
            item.target_rect_px[1] + item.target_rect_px[3],
        )
        for item in trace.patches
    }
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    def source_image(patch: PanelPatchAsset) -> Image.Image:
        loaded = image_loader(patch) if image_loader is not None else None
        if loaded is not None:
            rgba = np.ascontiguousarray(loaded, dtype=np.uint8)
            if rgba.ndim != 3 or rgba.shape[2] != 4:
                raise ValueError("panel patch loader must return HxWx4 uint8")
            return Image.fromarray(rgba, "RGBA")
        if patch.image_path:
            with Image.open(patch.image_path) as opened:
                return opened.convert("RGBA")
        return Image.new("RGBA", (1, 1), patch.color_rgba)

    for role in (
        PanelPatchRole.CENTER,
        PanelPatchRole.TOP, PanelPatchRole.RIGHT,
        PanelPatchRole.BOTTOM, PanelPatchRole.LEFT,
        PanelPatchRole.TOP_LEFT, PanelPatchRole.TOP_RIGHT,
        PanelPatchRole.BOTTOM_RIGHT, PanelPatchRole.BOTTOM_LEFT,
    ):
        target = targets.get(role)
        if target is None:
            continue
        target_width = target[2] - target[0]
        target_height = target[3] - target[1]
        if target_width <= 0 or target_height <= 0:
            continue
        patch = primitive.patch_map.get(role)
        if patch is None:
            continue
        source = source_image(patch)
        crop = patch.crop_vectors
        crop_left = min(
            source.width - 1,
            max(0, int(crop.minimum_uv[0] * source.width)),
        )
        crop_top = min(
            source.height - 1,
            max(0, int(crop.minimum_uv[1] * source.height)),
        )
        crop_right = min(
            source.width,
            max(crop_left + 1, int(np.ceil(
                crop.maximum_uv[0] * source.width
            ))),
        )
        crop_bottom = min(
            source.height,
            max(crop_top + 1, int(np.ceil(
                crop.maximum_uv[1] * source.height
            ))),
        )
        source = source.crop((
            crop_left, crop_top, crop_right, crop_bottom
        ))
        if patch.fill is PanelPatchFill.STRETCH:
            rendered = source.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )
        else:
            rendered = Image.new("RGBA", (target_width, target_height))
            for tile_y in range(0, target_height, max(1, source.height)):
                for tile_x in range(0, target_width, max(1, source.width)):
                    rendered.alpha_composite(source, (tile_x, tile_y))
        canvas.alpha_composite(rendered, (target[0], target[1]))
    return np.asarray(canvas, dtype=np.uint8).copy()


__all__ = [
    "PanelPatchRole", "PanelPatchFill", "PanelCropVectors",
    "PanelPatchObjectRequest", "PanelPatchAsset",
    "LayoutPanelPrimitive", "layout_panel_primitive_from_mapping",
    "PanelCompositionPatchTrace", "PanelCompositionTrace",
    "layout_panel_composition_trace", "compose_layout_panel_rgba",
]
