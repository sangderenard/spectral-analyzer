"""Reusable render assets for ray-traced display scenes.

This module is deliberately renderer-neutral.  It gives font/token geometry,
light-field conditions, cached artifacts, and page/part bake work stable
identities before CPU or GPU transport is selected.  Scene-order compilation
and exposure execution remain adapters on the other side of this boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import functools
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .display_scene import (
    DisplayObjectSpec,
    DisplayPrimitiveKind,
    DisplayProductKind,
    DisplaySceneSpec,
)


RENDER_ASSET_SCHEMA_VERSION = 1
INK_ATLAS_SCENE_ID = "ink-on-black-slate-atlas"
SENSOR_DISPLAY_ORIENTATION = "native-getter-yflip-transpose-v3"
MIN_INK_GLYPH_EXPOSURE_COVERAGE = 0.95
MIN_INK_GLYPH_RADIANCE_COVERAGE = 0.90
INK_ATLAS_CONVERGENCE_EXPOSURE_COVERAGE = 0.995
INK_ATLAS_CONVERGENCE_RELATIVE_RMSE = 0.005
INK_ATLAS_CONVERGENCE_P95_DELTA = 0.01
INK_ATLAS_CONVERGENCE_HOLD = 3


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("render asset identities require finite numbers")
        return value
    return value


def stable_asset_key(namespace: str, value: Any) -> str:
    """Return a versioned, portable content identity."""

    encoded = json.dumps(
        {
            "schema_version": RENDER_ASSET_SCHEMA_VERSION,
            "namespace": str(namespace),
            "value": _canonical(value),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"{namespace}:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class FontAssetSpec:
    """Font identity used by both sequence glyphs and atlas fallbacks."""

    family: str = "DejaVu Sans"
    weight: str = "bold"
    style: str = "normal"
    file: str = ""
    source_digest: str = ""

    def __post_init__(self) -> None:
        family = str(self.family).strip()
        if not family and not self.file:
            raise ValueError("font family or font file must be supplied")
        object.__setattr__(self, "family", family)
        object.__setattr__(
            self, "file", os.path.abspath(self.file) if self.file else ""
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FontAssetSpec":
        return cls(
            family=str(value.get("family", "DejaVu Sans")),
            weight=str(value.get("weight", "bold")),
            style=str(value.get("style", "normal")),
            file=str(value.get("file", "")),
            source_digest=str(value.get("source_digest", "")),
        )

    def scene_order_mapping(self) -> dict[str, str]:
        result = {
            "family": self.family,
            "weight": self.weight,
            "style": self.style,
        }
        if self.file:
            result["file"] = self.file
        return result

    def identity_payload(self) -> dict[str, str]:
        resolved_file, digest = _font_source_identity(
            self.family,
            self.weight,
            self.style,
            self.file,
            self.source_digest,
        )
        return {
            "family": self.family,
            "weight": self.weight,
            "style": self.style,
            # The digest makes equal font bytes portable across installations.
            # The resolved name remains useful when a source cannot be hashed.
            "source": os.path.basename(resolved_file) if resolved_file else "",
            "source_digest": digest,
        }


@functools.lru_cache(maxsize=128)
def _font_source_identity(
    family: str,
    weight: str,
    style: str,
    filename: str,
    supplied_digest: str,
) -> tuple[str, str]:
    if supplied_digest:
        return filename, supplied_digest
    resolved = filename
    if not resolved:
        try:
            from matplotlib.font_manager import FontProperties, findfont

            resolved = str(findfont(FontProperties(
                family=family, weight=weight, style=style
            )))
        except Exception:
            resolved = ""
    if resolved and os.path.isfile(resolved):
        digest = hashlib.sha256()
        with open(resolved, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return os.path.abspath(resolved), digest.hexdigest()
    return resolved, stable_asset_key(
        "logical-font", (family, weight, style, resolved)
    )


@dataclass(frozen=True)
class ExtrusionAssetSpec:
    """Placement-independent text layout and extrusion recipe."""

    height_m: float
    line_height_m: float
    line_spacing: float = 1.2
    text_box_m: tuple[float, float] | None = None
    horizontal_align: str = "center"
    vertical_align: str = "center"
    depth_ratio: float | None = 0.08
    depth_m: float | None = None
    embed_fraction: float = 0.25
    offset_m: tuple[float, float] = (0.0, 0.0)
    profile: str = "straight"
    profile_segments: int = 1
    profile_bulge: float = 0.0
    outline_subdivisions: int = 4
    cap_grid: int = 64

    def __post_init__(self) -> None:
        if self.height_m <= 0.0 or self.line_height_m <= 0.0:
            raise ValueError("glyph height and line height must be positive")
        if self.line_spacing <= 0.0:
            raise ValueError("line spacing must be positive")
        if (self.depth_ratio is None) == (self.depth_m is None):
            raise ValueError("supply exactly one of depth ratio or fixed depth")
        if self.depth_ratio is not None and self.depth_ratio <= 0.0:
            raise ValueError("extrusion depth ratio must be positive")
        if self.depth_m is not None and self.depth_m <= 0.0:
            raise ValueError("extrusion depth must be positive")
        if self.text_box_m is not None and (
            len(self.text_box_m) != 2 or any(value <= 0.0 for value in self.text_box_m)
        ):
            raise ValueError("text box must contain two positive dimensions")
        if self.horizontal_align not in {"left", "center", "right"}:
            raise ValueError("unsupported horizontal alignment")
        if self.vertical_align not in {"bottom", "center", "top"}:
            raise ValueError("unsupported vertical alignment")
        if not 0.0 <= self.embed_fraction <= 1.0:
            raise ValueError("embed fraction must be in [0, 1]")
        if self.profile not in {"straight", "circular"}:
            raise ValueError("extrusion profile must be straight or circular")
        if self.outline_subdivisions < 1 or self.cap_grid < 4:
            raise ValueError("outline and cap detail must be positive")

    def scene_order_mapping(self, embed_plane: str, material: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "embed_plane": str(embed_plane),
            "height_m": float(self.height_m),
            "line_height_m": float(self.line_height_m),
            "line_spacing": float(self.line_spacing),
            "horizontal_align": self.horizontal_align,
            "vertical_align": self.vertical_align,
            "embed_fraction": float(self.embed_fraction),
            "offset_m": list(self.offset_m),
            "profile": self.profile,
            "outline_subdivisions": int(self.outline_subdivisions),
            "cap_grid": int(self.cap_grid),
            "material": str(material),
        }
        if self.depth_m is not None:
            result["depth_m"] = float(self.depth_m)
        else:
            result["extrusion_depth_ratio"] = float(self.depth_ratio)
        if self.text_box_m is not None:
            result["text_box_m"] = list(self.text_box_m)
        if self.profile != "straight" or self.profile_segments != 1:
            result["profile_segments"] = int(self.profile_segments)
        if self.profile_bulge:
            result["profile_bulge"] = float(self.profile_bulge)
        return result


@dataclass(frozen=True)
class ExtrudedTokenAsset:
    """A token sequence that can be compiled, rendered, and cached."""

    token: str
    font: FontAssetSpec
    extrusion: ExtrusionAssetSpec
    material: str = "text_surface"
    material_revision: str = "1"

    def __post_init__(self) -> None:
        if not str(self.token).strip():
            raise ValueError("an extruded token asset requires visible content")
        if not str(self.material).strip():
            raise ValueError("token material must be named")

    @property
    def asset_key(self) -> str:
        return stable_asset_key("token", {
            "token": self.token,
            "font": self.font.identity_payload(),
            "extrusion": self.extrusion,
            "material": self.material,
            "material_revision": self.material_revision,
        })

    def with_token(self, token: str) -> "ExtrudedTokenAsset":
        return ExtrudedTokenAsset(
            token=str(token),
            font=self.font,
            extrusion=self.extrusion,
            material=self.material,
            material_revision=self.material_revision,
        )

    def scene_order_object(
        self, object_id: str, embed_plane: str, *, enabled: bool = True
    ) -> dict[str, Any]:
        """Adapt the asset to the existing scene-order ABI."""

        return {
            "id": str(object_id),
            "token": self.token,
            "embed_plane": str(embed_plane),
            "enabled": bool(enabled),
            "font": self.font.scene_order_mapping(),
            "geometry": self.extrusion.scene_order_mapping(
                embed_plane, self.material
            ),
        }


@dataclass(frozen=True)
class LightFieldCondition:
    """One cacheable view/light/material state.

    The production default is a single fixed camera/ring-light frame. Angles
    exist so a later recorded action sequence can use the same catalog ABI.
    """

    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    light_rig: str = "camera_ring"
    material_variant: str = "red_ink_on_black_slate"
    frame: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.azimuth_deg) or not math.isfinite(self.elevation_deg):
            raise ValueError("light-field angles must be finite")
        if not self.light_rig or not self.material_variant or self.frame < 0:
            raise ValueError("light-field condition fields must be valid")

    @property
    def condition_key(self) -> str:
        return stable_asset_key("condition", self)


@dataclass(frozen=True)
class RotatingStageSpec:
    """Optional condition expansion; production does not rotate by default."""

    azimuth_views: int = 1
    elevations_deg: tuple[float, ...] = (0.0,)
    light_rigs: tuple[str, ...] = ("camera_ring",)
    material_variants: tuple[str, ...] = ("red_ink_on_black_slate",)

    def __post_init__(self) -> None:
        if self.azimuth_views <= 0 or not self.elevations_deg:
            raise ValueError("a rotating stage requires views and elevations")
        if not self.light_rigs or not self.material_variants:
            raise ValueError("a rotating stage requires light and material conditions")

    def conditions(self) -> tuple[LightFieldCondition, ...]:
        return tuple(
            LightFieldCondition(
                azimuth_deg=360.0 * view / self.azimuth_views,
                elevation_deg=float(elevation),
                light_rig=str(light_rig),
                material_variant=str(material),
                frame=view,
            )
            for material in self.material_variants
            for light_rig in self.light_rigs
            for elevation in self.elevations_deg
            for view in range(self.azimuth_views)
        )

    @classmethod
    def knobs(cls) -> list[Any]:
        """Expose the stage through the repository's canonical KnobSpec ABI."""

        from controls import KnobSpec

        return [
            KnobSpec(
                "azimuth_views", "Recorded views", "int", 1,
                1.0, 256.0, 1.0, "views", group="Light-field stage",
                source_class=cls.__name__,
            ),
            KnobSpec(
                "elevation_deg", "Elevation", "float", 0.0,
                -90.0, 90.0, 1.0, "deg", group="Light-field stage",
                source_class=cls.__name__,
            ),
        ]


DEFAULT_INK_CONDITION = LightFieldCondition()


@dataclass(frozen=True)
class AtlasCaptureSpec:
    """Padded atlas frame used to preserve light spilled around a glyph."""

    width: int = 128
    height: int = 128
    content_width: int = 80
    content_height: int = 80

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.content_width, self.content_height) <= 0:
            raise ValueError("atlas capture dimensions must be positive")
        if self.content_width > self.width or self.content_height > self.height:
            raise ValueError("atlas content region must fit inside its capture")

    @property
    def content_region(self) -> tuple[int, int, int, int]:
        return (
            (self.width - self.content_width) // 2,
            (self.height - self.content_height) // 2,
            self.content_width,
            self.content_height,
        )


MATTE_BLACK_SLATE = {
    "albedo_rgb": [0.002, 0.002, 0.002],
    "reflectivity": 0.012,
    "diffusion": 0.94,
    "absorption": 0.97,
    "roughness": 0.9,
    "metallic": 0.0,
}

GLOSSY_RED_INK = {
    "albedo_rgb": [0.78, 0.006, 0.004],
    "reflectivity": 0.5,
    "diffusion": 0.18,
    "absorption": 0.31,
    "roughness": 0.065,
    "metallic": 0.08,
    "ior": 1.52,
}


def ink_token_asset(
    token: str,
    *,
    font: FontAssetSpec | None = None,
    character: bool | None = None,
) -> ExtrudedTokenAsset:
    """Create the accepted wet red-ink extrusion for a char or token."""

    text = str(token)
    is_character = len(text) == 1 if character is None else bool(character)
    height = 0.2 if is_character else 0.13
    depth = 0.032 if is_character else 0.028
    return ExtrudedTokenAsset(
        token=text,
        font=font or FontAssetSpec(),
        extrusion=ExtrusionAssetSpec(
            height_m=height,
            line_height_m=height,
            line_spacing=1.0,
            text_box_m=None,
            depth_ratio=None,
            depth_m=depth,
            embed_fraction=0.5,
            profile="circular",
            profile_segments=10,
            profile_bulge=0.055,
            outline_subdivisions=2,
            cap_grid=30 if is_character else 40,
        ),
        material="glossy_red",
        material_revision="accepted-red-ink-v1",
    )


def _ink_job_id(token: str, index: int) -> str:
    if len(token) == 1:
        return f"ink_glyph_U{ord(token):04X}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return f"ink_token_{index:04d}_{digest}"


def build_ink_on_slate_order(
    tokens: Sequence[str],
    *,
    capture: AtlasCaptureSpec | None = None,
    font: FontAssetSpec | None = None,
    sensor_sweeps: int = 1,
) -> dict[str, Any]:
    """Build ordinary scene-order jobs for the fixed production look.

    No camera pose, camera script, light script, or stage motion is authored.
    The existing lab camera and its retained ring-light rig execute the order.
    """

    capture = capture or AtlasCaptureSpec()
    font = font or FontAssetSpec()
    unique: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        token = str(raw)
        if not token.strip():
            continue
        if token not in seen:
            seen.add(token)
            unique.append(token)
    if not unique:
        raise ValueError("at least one visible character or token is required")
    jobs = []
    for index, token in enumerate(unique):
        asset = ink_token_asset(token, font=font)
        jobs.append({
            "id": _ink_job_id(token, index),
            "token": token,
            "geometry": asset.extrusion.scene_order_mapping(
                "backplate", "glossy_red"
            ),
        })
    character_default = ink_token_asset("A", font=font, character=True)
    return {
        "schema_version": 1,
        "defaults": {
            "image": {"width": capture.width, "height": capture.height},
            "camera": {"focal_mm": 35.0, "aperture_mm": 25.0},
            "exposure": {
                "time_s": 1.0 / 60.0,
                "iso": 100.0,
                "sensor_sweeps": int(sensor_sweeps),
                "t5_pair_budget": 20_000_000,
            },
            "flash": {"intensity_scale": 1.0},
            "font": font.scene_order_mapping(),
            "planes": [{
                "id": "backplate",
                "center_m": [0.0, 0.0, 0.0],
                "normal": [1.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "size_m": [0.52, 0.42],
                "thickness_m": 0.012,
                "material": "matte_black",
            }],
            "materials": {
                "matte_black": dict(MATTE_BLACK_SLATE),
                "glossy_red": dict(GLOSSY_RED_INK),
            },
            "geometry": character_default.extrusion.scene_order_mapping(
                "backplate", "glossy_red"
            ),
        },
        "jobs": jobs,
    }


@dataclass(frozen=True)
class RenderedAssetRecord:
    target_key: str
    condition_key: str
    product_kind: DisplayProductKind
    linear_path: str = ""
    preview_path: str = ""
    manifest_path: str = ""
    width: int = 0
    height: int = 0
    samples: int = 0
    created_at_s: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def record_key(self) -> str:
        return stable_asset_key(
            "artifact",
            (self.target_key, self.condition_key, self.product_kind.value),
        )


class RenderAssetCatalog:
    """Atomic manifest of reusable page, part, token, and atlas artifacts."""

    def __init__(self, path: str = "") -> None:
        self.path = os.path.abspath(path) if path else ""
        self._lock = threading.RLock()
        self._records: dict[str, RenderedAssetRecord] = {}
        if self.path and os.path.isfile(self.path):
            self._load()

    def find(
        self,
        target_key: str,
        condition: LightFieldCondition,
        product_kind: DisplayProductKind,
    ) -> RenderedAssetRecord | None:
        key = stable_asset_key(
            "artifact",
            (str(target_key), condition.condition_key, product_kind.value),
        )
        with self._lock:
            return self._records.get(key)

    def record(self, artifact: RenderedAssetRecord) -> None:
        with self._lock:
            self._records[artifact.record_key] = artifact
            self._save()

    def complete(
        self,
        request: "BakeRequest",
        *,
        linear_path: str = "",
        preview_path: str = "",
        manifest_path: str = "",
        width: int = 0,
        height: int = 0,
        samples: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> RenderedAssetRecord:
        """Commit an executed bake request to the reusable catalog."""

        record = RenderedAssetRecord(
            target_key=request.target_key,
            condition_key=request.condition.condition_key,
            product_kind=request.product_kind,
            linear_path=str(linear_path),
            preview_path=str(preview_path),
            manifest_path=str(manifest_path),
            width=int(width),
            height=int(height),
            samples=int(samples),
            metadata={
                "target_kind": request.target_kind.value,
                "scene_id": request.scene_id,
                "object_ids": list(request.object_ids),
                **(
                    {}
                    if request.capture is None
                    else {
                        "capture": {
                            "width": request.capture.width,
                            "height": request.capture.height,
                            "content_region": list(
                                request.capture.content_region
                            ),
                        }
                    }
                ),
                **dict(metadata or {}),
            },
        )
        self.record(record)
        return record

    def snapshot(self) -> tuple[RenderedAssetRecord, ...]:
        with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def _save(self) -> None:
        if not self.path:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "schema_version": RENDER_ASSET_SCHEMA_VERSION,
            "records": [_canonical(record) for record in self.snapshot()],
        }
        temp = self.path + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temp, self.path)

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("schema_version", -1)) != RENDER_ASSET_SCHEMA_VERSION:
            raise ValueError("unsupported render asset catalog schema")
        for raw in payload.get("records", ()):
            item = dict(raw)
            item["product_kind"] = DisplayProductKind(item["product_kind"])
            item["metadata"] = dict(item.get("metadata", {}))
            record = RenderedAssetRecord(**item)
            self._records[record.record_key] = record


def ink_record_is_converged(record: RenderedAssetRecord | None) -> bool:
    """Return whether image evidence, rather than a ray count, finished the asset."""

    if record is None:
        return False
    metadata = dict(record.metadata)
    if not bool(metadata.get("bounded_background_render", False)):
        return ink_record_is_usable(record)
    quality = dict(metadata.get("atlas_quality", {}))
    return bool(
        metadata.get("sensor_display_orientation")
        == SENSOR_DISPLAY_ORIENTATION
        and
        metadata.get("refinement_state") == "converged"
        and quality.get("converged", False)
    )


def ink_record_is_usable(record: RenderedAssetRecord | None) -> bool:
    """Allow covered developing sprites while rejecting nearly-empty warmups."""

    if record is None:
        return False
    metadata = dict(record.metadata)
    if not bool(metadata.get("bounded_background_render", False)):
        # Imported/test artifacts with an explicit sprite are trusted. The
        # evidence gate applies to production background renders.
        sprite_path = str(metadata.get("sprite_path", ""))
        return bool(
            (sprite_path and os.path.isfile(sprite_path))
            or (record.linear_path and os.path.isfile(record.linear_path))
        )
    quality = dict(metadata.get("atlas_quality", {}))
    return bool(
        metadata.get("sensor_display_orientation")
        == SENSOR_DISPLAY_ORIENTATION
        and quality.get("composable", False)
        and float(quality.get("glyph_exposure_coverage", 0.0))
        >= MIN_INK_GLYPH_EXPOSURE_COVERAGE
        and float(quality.get("glyph_radiance_coverage", 0.0))
        >= MIN_INK_GLYPH_RADIANCE_COVERAGE
    )


def _record_refinement_pass(record: RenderedAssetRecord | None) -> int:
    if record is None:
        return 0
    metadata = dict(record.metadata)
    if metadata.get("refinement_state") not in {"developing", "converged"}:
        # Legacy bounded captures did not preserve resumable convergence state
        # or the readable sensor orientation.
        return 0
    if (
        metadata.get("sensor_display_orientation")
        != SENSOR_DISPLAY_ORIENTATION
    ):
        return 0
    return max(0, int(metadata.get("refinement_pass", record.samples)))


@dataclass(frozen=True)
class AtlasResolution:
    exact: RenderedAssetRecord | None
    character_artifacts: tuple[RenderedAssetRecord, ...]
    missing_characters: tuple[ExtrudedTokenAsset, ...]

    @property
    def ready(self) -> bool:
        return self.exact is not None or not self.missing_characters

    @property
    def uses_character_fallback(self) -> bool:
        return self.exact is None


class CharacterAtlas:
    """Resolve a sequence exactly, then fall back to independently cached chars."""

    def __init__(
        self,
        catalog: RenderAssetCatalog,
        product_kind: DisplayProductKind = DisplayProductKind.LIGHT_FIELD,
    ) -> None:
        self.catalog = catalog
        self.product_kind = product_kind

    def resolve(
        self, asset: ExtrudedTokenAsset, condition: LightFieldCondition
    ) -> AtlasResolution:
        exact = self.catalog.find(asset.asset_key, condition, self.product_kind)
        if exact is not None:
            return AtlasResolution(exact, (), ())
        artifacts: list[RenderedAssetRecord] = []
        missing: list[ExtrudedTokenAsset] = []
        seen: set[str] = set()
        for character in asset.token:
            if character.isspace() or character in seen:
                continue
            seen.add(character)
            glyph = asset.with_token(character)
            record = self.catalog.find(
                glyph.asset_key, condition, self.product_kind
            )
            if record is None:
                missing.append(glyph)
            else:
                artifacts.append(record)
        return AtlasResolution(exact, tuple(artifacts), tuple(missing))


class BakeTargetKind(str, Enum):
    PAGE = "page"
    PART = "part"
    TOKEN_SEQUENCE = "token_sequence"
    TOKEN_STRING = "token_string"
    ATLAS_GLYPH = "atlas_glyph"


@dataclass(frozen=True)
class BakeRequest:
    target_key: str
    target_kind: BakeTargetKind
    condition: LightFieldCondition
    product_kind: DisplayProductKind
    scene_id: str
    object_ids: tuple[str, ...]
    token_asset: ExtrudedTokenAsset | None = None
    capture: AtlasCaptureSpec | None = None
    refinement_pass: int = 0

    @property
    def request_key(self) -> str:
        return stable_asset_key(
            "bake-request",
            (
                self.target_key,
                self.target_kind.value,
                self.condition.condition_key,
                self.product_kind.value,
            ),
        )


@dataclass(frozen=True)
class BakePlan:
    scene_id: str
    requests: tuple[BakeRequest, ...]
    cached: tuple[RenderedAssetRecord, ...]
    atlas_resolutions: Mapping[str, AtlasResolution]

    @property
    def next_request(self) -> BakeRequest | None:
        """One bounded unit the continually running renderer may take."""

        return self.requests[0] if self.requests else None


def plan_ink_atlas_bake(
    tokens: Sequence[str],
    catalog: RenderAssetCatalog,
    *,
    token_strings: Sequence[str] = (),
    capture: AtlasCaptureSpec | None = None,
    font: FontAssetSpec | None = None,
) -> BakePlan:
    """Queue unfinished fixed-frame token and canonical character refinements.

    This is the production path. It does not create page, part, rotation, or
    scripted-action work. Orders can be submitted a few tokens at a time; the
    Only image-converged records leave the queue. Developing records remain
    composable once covered and receive one more ray slice whenever idle.
    """

    capture = capture or AtlasCaptureSpec()
    font = font or FontAssetSpec()
    pending: dict[str, BakeRequest] = {}
    cached: dict[str, RenderedAssetRecord] = {}
    resolutions: dict[str, AtlasResolution] = {}
    seen_tokens: set[str] = set()
    for raw in tokens:
        token = str(raw)
        if not token.strip() or token in seen_tokens:
            continue
        seen_tokens.add(token)
        sequence_asset = ink_token_asset(token, font=font)
        exact = catalog.find(
            sequence_asset.asset_key,
            DEFAULT_INK_CONDITION,
            DisplayProductKind.IMAGE,
        )
        exact_composable = ink_record_is_usable(exact)
        exact_converged = ink_record_is_converged(exact)
        if not exact_composable:
            exact = None
        character_records: list[RenderedAssetRecord] = []
        missing_characters: list[ExtrudedTokenAsset] = []
        seen_characters: set[str] = set()
        for character in token:
            if character.isspace() or character in seen_characters:
                continue
            seen_characters.add(character)
            glyph = ink_token_asset(character, font=font, character=True)
            record = catalog.find(
                glyph.asset_key,
                DEFAULT_INK_CONDITION,
                DisplayProductKind.IMAGE,
            )
            raw_record = record
            if not ink_record_is_usable(record):
                record = None
            if record is None:
                missing_characters.append(glyph)
            else:
                character_records.append(record)
                cached[record.record_key] = record
            if not ink_record_is_converged(raw_record):
                refinement_pass = _record_refinement_pass(raw_record)
                request = BakeRequest(
                    glyph.asset_key,
                    BakeTargetKind.ATLAS_GLYPH,
                    DEFAULT_INK_CONDITION,
                    DisplayProductKind.IMAGE,
                    INK_ATLAS_SCENE_ID,
                    (_ink_job_id(character, 0),),
                    glyph,
                    capture,
                    refinement_pass,
                )
                pending[request.request_key] = request
        resolution = AtlasResolution(
            exact,
            tuple(character_records),
            tuple(missing_characters),
        )
        resolutions[stable_asset_key(
            "ink-atlas-resolution", sequence_asset.asset_key
        )] = resolution
        if len(token) > 1 and not exact_converged:
            raw_exact = catalog.find(
                sequence_asset.asset_key,
                DEFAULT_INK_CONDITION,
                DisplayProductKind.IMAGE,
            )
            refinement_pass = _record_refinement_pass(raw_exact)
            request = BakeRequest(
                sequence_asset.asset_key,
                BakeTargetKind.TOKEN_SEQUENCE,
                DEFAULT_INK_CONDITION,
                DisplayProductKind.IMAGE,
                INK_ATLAS_SCENE_ID,
                (_ink_job_id(token, 0),),
                sequence_asset,
                capture,
                refinement_pass,
            )
            pending[request.request_key] = request
        if exact is not None:
            cached[exact.record_key] = exact
    for raw in token_strings:
        token_string = str(raw)
        if not token_string.strip():
            continue
        string_asset = ink_token_asset(token_string, font=font)
        record = catalog.find(
            string_asset.asset_key,
            DEFAULT_INK_CONDITION,
            DisplayProductKind.IMAGE,
        )
        if ink_record_is_usable(record):
            cached[record.record_key] = record
        if not ink_record_is_converged(record):
            request = BakeRequest(
                string_asset.asset_key,
                BakeTargetKind.TOKEN_STRING,
                DEFAULT_INK_CONDITION,
                DisplayProductKind.IMAGE,
                INK_ATLAS_SCENE_ID,
                (_ink_job_id(token_string, 0),),
                string_asset,
                capture,
                _record_refinement_pass(record),
            )
            pending[request.request_key] = request
    # Larger token gestalts are not eligible while even one constituent glyph
    # remains unfinished. This is a hard dependency, not merely sort priority.
    if any(
        request.target_kind is BakeTargetKind.ATLAS_GLYPH
        for request in pending.values()
    ):
        pending = {
            key: request
            for key, request in pending.items()
            if request.target_kind is BakeTargetKind.ATLAS_GLYPH
        }
    elif any(
        request.target_kind is BakeTargetKind.TOKEN_SEQUENCE
        for request in pending.values()
    ):
        pending = {
            key: request
            for key, request in pending.items()
            if request.target_kind is BakeTargetKind.TOKEN_SEQUENCE
        }
    ordered = sorted(
        pending.values(),
        key=lambda request: (
            0 if request.target_kind is BakeTargetKind.ATLAS_GLYPH else 1,
            0 if request.target_kind is BakeTargetKind.TOKEN_SEQUENCE else 1,
            request.refinement_pass,
            "" if request.token_asset is None else request.token_asset.token,
            request.request_key,
        ),
    )
    return BakePlan(
        INK_ATLAS_SCENE_ID,
        tuple(ordered),
        tuple(cached[key] for key in sorted(cached)),
        resolutions,
    )


def ink_order_for_request(request: BakeRequest) -> dict[str, Any]:
    """Turn one production atlas request into the ordinary renderer order."""

    if request.token_asset is None:
        raise ValueError("ink atlas request has no token asset")
    if request.condition != DEFAULT_INK_CONDITION:
        raise ValueError("ink production orders use the fixed ring-light condition")
    return build_ink_on_slate_order(
        (request.token_asset.token,),
        capture=request.capture or AtlasCaptureSpec(),
        font=request.token_asset.font,
    )


def _object_token_asset(spec: DisplayObjectSpec) -> ExtrudedTokenAsset | None:
    if spec.primitive.kind is not DisplayPrimitiveKind.TEXT:
        return None
    text_height = max(
        0.0015,
        min(spec.placement.size_m[1] * 0.90, spec.placement.size_m[0] * 0.30),
    )
    return ExtrudedTokenAsset(
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
                spec.placement.size_m[0] * 0.94,
                spec.placement.size_m[1] * 0.90,
            ),
            horizontal_align=spec.horizontal_align,
            vertical_align=spec.vertical_align,
        ),
        material=spec.material,
    )


def plan_display_scene_bake(
    scene: DisplaySceneSpec,
    stage: RotatingStageSpec,
    catalog: RenderAssetCatalog,
) -> BakePlan:
    """Plan whole-page, constituent-part, sequence, and atlas fallback work."""

    atlas = CharacterAtlas(catalog)
    pending: dict[str, BakeRequest] = {}
    cached: dict[str, RenderedAssetRecord] = {}
    resolutions: dict[str, AtlasResolution] = {}
    page_key = stable_asset_key("page", scene)
    for condition in stage.conditions():
        page_record = catalog.find(
            page_key, condition, DisplayProductKind.LIGHT_FIELD
        )
        page_request = BakeRequest(
            page_key,
            BakeTargetKind.PAGE,
            condition,
            DisplayProductKind.LIGHT_FIELD,
            scene.scene_id,
            tuple(spec.object_id for spec in scene.objects if spec.enabled),
        )
        if page_record is None:
            pending[page_request.request_key] = page_request
        else:
            cached[page_record.record_key] = page_record

        for spec in scene.objects:
            if not spec.enabled:
                continue
            part_key = stable_asset_key("part", spec)
            part_record = catalog.find(
                part_key, condition, DisplayProductKind.LIGHT_FIELD
            )
            part_request = BakeRequest(
                part_key,
                BakeTargetKind.PART,
                condition,
                DisplayProductKind.LIGHT_FIELD,
                scene.scene_id,
                (spec.object_id,),
            )
            if part_record is None:
                pending[part_request.request_key] = part_request
            else:
                cached[part_record.record_key] = part_record

            token_asset = _object_token_asset(spec)
            if token_asset is None:
                continue
            resolution = atlas.resolve(token_asset, condition)
            resolution_key = stable_asset_key(
                "atlas-resolution",
                (token_asset.asset_key, condition.condition_key),
            )
            resolutions[resolution_key] = resolution
            if resolution.exact is None:
                sequence_request = BakeRequest(
                    token_asset.asset_key,
                    BakeTargetKind.TOKEN_SEQUENCE,
                    condition,
                    DisplayProductKind.LIGHT_FIELD,
                    scene.scene_id,
                    (spec.object_id,),
                    token_asset,
                )
                pending[sequence_request.request_key] = sequence_request
            else:
                cached[resolution.exact.record_key] = resolution.exact
            for glyph in resolution.missing_characters:
                glyph_request = BakeRequest(
                    glyph.asset_key,
                    BakeTargetKind.ATLAS_GLYPH,
                    condition,
                    DisplayProductKind.LIGHT_FIELD,
                    scene.scene_id,
                    (spec.object_id,),
                    glyph,
                )
                pending[glyph_request.request_key] = glyph_request
            for artifact in resolution.character_artifacts:
                cached[artifact.record_key] = artifact
    return BakePlan(
        scene_id=scene.scene_id,
        requests=tuple(pending[key] for key in sorted(pending)),
        cached=tuple(cached[key] for key in sorted(cached)),
        atlas_resolutions=resolutions,
    )


class LightFieldMaterialWidget:
    """Control-panel model for inspecting and baking a scene on a stage."""

    def __init__(
        self,
        scene: DisplaySceneSpec,
        catalog: RenderAssetCatalog,
        stage: RotatingStageSpec | None = None,
    ) -> None:
        self.scene = scene
        self.catalog = catalog
        self.stage = stage or RotatingStageSpec()

    @classmethod
    def knobs(cls) -> list[Any]:
        return RotatingStageSpec.knobs()

    def plan(self) -> BakePlan:
        return plan_display_scene_bake(self.scene, self.stage, self.catalog)


__all__ = [
    "RENDER_ASSET_SCHEMA_VERSION",
    "INK_ATLAS_SCENE_ID",
    "SENSOR_DISPLAY_ORIENTATION",
    "MIN_INK_GLYPH_EXPOSURE_COVERAGE",
    "MIN_INK_GLYPH_RADIANCE_COVERAGE",
    "INK_ATLAS_CONVERGENCE_EXPOSURE_COVERAGE",
    "INK_ATLAS_CONVERGENCE_RELATIVE_RMSE",
    "INK_ATLAS_CONVERGENCE_P95_DELTA",
    "INK_ATLAS_CONVERGENCE_HOLD",
    "stable_asset_key",
    "FontAssetSpec",
    "ExtrusionAssetSpec",
    "ExtrudedTokenAsset",
    "LightFieldCondition",
    "RotatingStageSpec",
    "DEFAULT_INK_CONDITION",
    "AtlasCaptureSpec",
    "MATTE_BLACK_SLATE",
    "GLOSSY_RED_INK",
    "ink_token_asset",
    "build_ink_on_slate_order",
    "RenderedAssetRecord",
    "RenderAssetCatalog",
    "ink_record_is_usable",
    "ink_record_is_converged",
    "AtlasResolution",
    "CharacterAtlas",
    "BakeTargetKind",
    "BakeRequest",
    "BakePlan",
    "plan_ink_atlas_bake",
    "ink_order_for_request",
    "plan_display_scene_bake",
    "LightFieldMaterialWidget",
]
