"""Renderable design-object work for manifest-authored layout primitives.

The layout owns placements and nine-slice crops.  This module supplies the
missing production layer: stable design objects, representative-square
subtypes, pose-animation work, and a restart-safe artifact/progress cache.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from .control_layout import ProgramUILayout
from .layout_panel import PanelPatchAsset, layout_panel_composition_trace


LAYOUT_OBJECT_WORK_SCHEMA_VERSION = 2
LAYOUT_WORK_CACHE_SCHEMA_VERSION = 1


def _stable_key(prefix: str, value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return f"{prefix}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _atomic_json(path: str, payload: Mapping[str, Any]) -> None:
    final_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    temporary = final_path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, final_path)


@dataclass(frozen=True)
class LayoutObjectConsumer:
    owner_id: str
    role: str
    crop_vectors: tuple[tuple[float, float], tuple[float, float]]
    fill: str
    target_rect_px: tuple[int, int, int, int]
    layout_rect_px: tuple[int, int, int, int]
    sampling_spec: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PoseFrameWork:
    frame_index: int
    phase: str
    phase_frame: int
    focus_location_percent: float
    progress_cache_key: str


@dataclass(frozen=True)
class PoseAnimationWork:
    """A renderable pose-animation subset belonging to one object subtype."""

    animation_key: str
    name: str
    rack_frames: int
    hold_frames: int
    focus_location_count: int
    frames: tuple[PoseFrameWork, ...]
    replacement_frame_index: int

    @property
    def replacement_cache_key(self) -> str:
        return self.frames[self.replacement_frame_index].progress_cache_key


@dataclass(frozen=True)
class LayoutAspectVariant:
    """One work-resolution derivative of the canonical square exposure."""

    variant_key: str
    aspect_ratio: tuple[int, int]
    work_resolution_px: tuple[int, int]
    square_pane_rect: tuple[float, float, float, float]
    owner_ids: tuple[str, ...]
    source_capture: str = "representative_square"
    sampling_policy: str = "repeat_square_tiles_clip_partial_terminal_tile"
    physical_pixel_aspect: float = 1.0


@dataclass(frozen=True)
class LayoutDesignSubtypeWork:
    subtype_key: str
    display_name: str
    source_capture: str
    condition_key: str
    still_cache_key: str
    consumers: tuple[LayoutObjectConsumer, ...]
    pose_animations: tuple[PoseAnimationWork, ...]
    parameter_spec: Mapping[str, Any] = field(default_factory=dict)
    aspect_variants: tuple[LayoutAspectVariant, ...] = ()


@dataclass(frozen=True)
class LayoutDesignObjectWork:
    object_key: str
    display_name: str
    object_kind: str
    subtypes: tuple[LayoutDesignSubtypeWork, ...]


@dataclass(frozen=True)
class LayoutObjectWorkManifest:
    manifest_key: str
    source_layout: str
    objects: tuple[LayoutDesignObjectWork, ...]
    progress_cache_path: str
    replacement_policy: str = (
        "first_valid_developing_hold_pose_then_still_then_color_fallback"
    )
    schema_version: int = LAYOUT_OBJECT_WORK_SCHEMA_VERSION

    def mapping(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> str:
        _atomic_json(path, self.mapping())
        return os.path.abspath(path)


def rack_focus_pose_animation(
    object_key: str,
    subtype_key: str,
    *,
    rack_frames: int = 12,
    hold_frames: int = 6,
    focus_location_count: int = 7,
) -> PoseAnimationWork:
    """Build rack-in, hold, rack-out poses quantized to M focus locations."""

    rack_frames = max(2, int(rack_frames))
    hold_frames = max(1, int(hold_frames))
    focus_location_count = max(2, int(focus_location_count))
    animation_key = _stable_key(
        "pose-animation",
        {
            "object_key": object_key,
            "subtype_key": subtype_key,
            "name": "rack-focus-demonstration",
            "rack_frames": rack_frames,
            "hold_frames": hold_frames,
            "focus_location_count": focus_location_count,
        },
    )
    locations = tuple(
        100.0 * index / (focus_location_count - 1)
        for index in range(focus_location_count)
    )

    def quantized(percent: float) -> float:
        return min(locations, key=lambda location: abs(location - percent))

    raw_frames: list[tuple[str, int, float]] = []
    for frame in range(rack_frames):
        t = frame / float(rack_frames - 1)
        eased = t * t * (3.0 - 2.0 * t)
        raw_frames.append(("rack_in", frame, quantized(100.0 * (1.0 - eased))))
    raw_frames.extend(("hold", frame, 0.0) for frame in range(hold_frames))
    for frame in range(rack_frames):
        t = frame / float(rack_frames - 1)
        eased = t * t * (3.0 - 2.0 * t)
        raw_frames.append(("rack_out", frame, quantized(100.0 * eased)))
    frames = tuple(
        PoseFrameWork(
            frame_index=index,
            phase=phase,
            phase_frame=phase_frame,
            focus_location_percent=focus_percent,
            progress_cache_key=_stable_key(
                "pose-progress",
                (animation_key, index, phase, focus_percent),
            ),
        )
        for index, (phase, phase_frame, focus_percent) in enumerate(raw_frames)
    )
    return PoseAnimationWork(
        animation_key=animation_key,
        name="rack focus in / hold / rack focus out",
        rack_frames=rack_frames,
        hold_frames=hold_frames,
        focus_location_count=focus_location_count,
        frames=frames,
        replacement_frame_index=rack_frames + hold_frames // 2,
    )


def build_layout_object_work_manifest(
    layout: ProgramUILayout,
    output_root: str,
    *,
    rack_frames: int = 12,
    hold_frames: int = 6,
    focus_location_count: int = 7,
) -> LayoutObjectWorkManifest:
    """Discover every bespoke panel request and organize its complete work."""

    grouped: dict[
        tuple[str, str, str], list[LayoutObjectConsumer]
    ] = {}
    primitives = {
        **layout.panel_primitives,
        **layout.action_primitives,
    }
    primitive_specs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for primitive in primitives.values():
        for patch in primitive.patches:
            if not patch.object_key or not patch.subtype_key:
                continue
            key = (
                str(patch.object_key),
                str(patch.subtype_key),
                str(patch.condition_key),
            )
            spec = primitive_specs.setdefault(key, {
                "scene_family": "parametric-nine-slice-panel-v1",
                "source_capture": "representative_square",
                "representative_grid": [3, 3],
                "capture_resolution_px": [128, 128],
                "border_px": list(primitive.border_px),
                "primitive_revision": int(primitive.revision),
                "patches": {},
                "instance_parameters": [
                    "target_width_px", "target_height_px", "patch_role",
                    "crop_minimum_uv", "crop_maximum_uv", "fill_mode",
                    "sensor_crop_px", "sample_resolution_px",
                    "sample_pitch_scale", "uv_transform",
                    "terminal_tile_policy",
                ],
                "pose_parameters": [
                    "focus_location_percent", "focus_distance_cm",
                    "frame_index", "phase",
                ],
            })
            spec["patches"][patch.role.value] = {
                "crop_vectors": patch.crop_vectors.mapping(),
                "fill": patch.fill.value,
                "fallback_color_rgba": list(patch.color_rgba),
            }
    layout_rect = tuple(layout.regions[layout.manifest.name])

    def patch_target(
        owner_id: str, role: str
    ) -> tuple[int, int, int, int]:
        owner = tuple(layout.regions[owner_id])
        primitive = primitives[owner_id]
        x, y, width, height = owner
        trace = layout_panel_composition_trace(primitive, width, height)
        local = next(
            patch.target_rect_px for patch in trace.patches
            if patch.role.value == role
        )
        return (x + local[0], y + local[1], local[2], local[3])

    for owner_id, requests in layout.panel_object_requests.items():
        for request in requests:
            key = (
                str(request.object_key),
                str(request.subtype_key),
                str(request.condition_key),
            )
            target_rect = patch_target(str(owner_id), request.role.value)
            grouped.setdefault(key, []).append(LayoutObjectConsumer(
                owner_id=str(owner_id),
                role=request.role.value,
                crop_vectors=(
                    tuple(request.crop_vectors.minimum_uv),
                    tuple(request.crop_vectors.maximum_uv),
                ),
                fill=request.fill.value,
                target_rect_px=target_rect,
                layout_rect_px=layout_rect,
                sampling_spec={
                    "sensor_crop_px": list(target_rect),
                    "sample_resolution_px": [
                        max(1, int(target_rect[2])),
                        max(1, int(target_rect[3])),
                    ],
                    "source_domain": "unit_square",
                    "source_uv": request.crop_vectors.mapping(),
                    "uv_transform": "repeat_square_tiles_clip_partial_terminal_tile",
                    "terminal_tile_policy": "clip_at_panel_boundary",
                    "physical_sensor_pixel_aspect": 1.0,
                    "sample_pitch_scale": [1.0, 1.0],
                },
            ))
    objects: dict[str, list[LayoutDesignSubtypeWork]] = {}
    for (object_key, subtype_key, condition_key), consumers in sorted(
        grouped.items()
    ):
        variant_owners: dict[tuple[int, int], list[str]] = {}
        for owner_id in sorted({item.owner_id for item in consumers}):
            _, _, width, height = tuple(layout.regions[owner_id])
            divisor = math.gcd(max(1, int(width)), max(1, int(height)))
            ratio = (max(1, int(width)) // divisor, max(1, int(height)) // divisor)
            variant_owners.setdefault((int(width), int(height)), []).append(owner_id)
        aspect_variants = []
        for (width, height), owner_ids in sorted(variant_owners.items()):
            divisor = math.gcd(max(1, width), max(1, height))
            ratio = (max(1, width) // divisor, max(1, height) // divisor)
            if width >= height:
                pane = (0.0, (1.0 - height / width) * 0.5, 1.0, height / width)
            else:
                pane = ((1.0 - width / height) * 0.5, 0.0, width / height, 1.0)
            aspect_variants.append(LayoutAspectVariant(
                variant_key=_stable_key(
                    "layout-aspect-variant",
                    (object_key, subtype_key, ratio, (width, height)),
                ),
                aspect_ratio=ratio,
                work_resolution_px=(width, height),
                square_pane_rect=tuple(float(value) for value in pane),
                owner_ids=tuple(owner_ids),
            ))
        animation = rack_focus_pose_animation(
            object_key,
            subtype_key,
            rack_frames=rack_frames,
            hold_frames=hold_frames,
            focus_location_count=focus_location_count,
        )
        objects.setdefault(object_key, []).append(LayoutDesignSubtypeWork(
            subtype_key=subtype_key,
            display_name="representative nine-slice square",
            source_capture="representative_square",
            condition_key=condition_key,
            still_cache_key=_stable_key(
                "layout-still", (object_key, subtype_key, condition_key)
            ),
            consumers=tuple(sorted(
                consumers, key=lambda item: (item.owner_id, item.role)
            )),
            pose_animations=(animation,),
            parameter_spec=primitive_specs.get(
                (object_key, subtype_key, condition_key), {}
            ),
            aspect_variants=tuple(aspect_variants),
        ))
    design_objects = tuple(
        LayoutDesignObjectWork(
            object_key=object_key,
            display_name=(
                "Bakery slate controls"
                if "control" in object_key else "Bakery slate panels"
            ),
            object_kind="layout_design_object",
            subtypes=tuple(subtypes),
        )
        for object_key, subtypes in sorted(objects.items())
    )
    root = os.path.abspath(output_root)
    manifest_key = _stable_key(
        "layout-object-work",
        {
            "layout": layout.manifest.name,
            "objects": [
                (item.object_key, [subtype.subtype_key for subtype in item.subtypes])
                for item in design_objects
            ],
        },
    )
    return LayoutObjectWorkManifest(
        manifest_key=manifest_key,
        source_layout=layout.manifest.name,
        objects=design_objects,
        progress_cache_path=os.path.join(
            root, "layout_object_work", "progress_cache.json"
        ),
    )


class LayoutWorkProgressCache:
    """Restart-safe status and artifact evidence for still and pose work."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        self._entries: dict[str, dict[str, Any]] = {}
        self._cache_revision: tuple[int, int] = (-1, -1)
        self.refresh(force=True)

    def _disk_revision(self) -> tuple[int, int]:
        try:
            stat = os.stat(self.path)
            return int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return (-1, -1)

    def refresh(self, *, force: bool = False) -> bool:
        """Adopt cache changes written by a renderer or another process."""

        revision = self._disk_revision()
        if not force and revision == self._cache_revision:
            return False
        entries: dict[str, dict[str, Any]] = {}
        if revision != (-1, -1):
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if int(payload.get("schema_version", -1)) == (
                    LAYOUT_WORK_CACHE_SCHEMA_VERSION
                ):
                    entries = {
                        str(key): dict(value)
                        for key, value in dict(payload.get("entries", {})).items()
                    }
            except (OSError, ValueError, TypeError):
                # Keep the last coherent snapshot while an atomic replacement
                # is in flight or an external writer has produced bad data.
                return False
        changed = entries != self._entries
        self._entries = entries
        self._cache_revision = revision
        return changed

    @staticmethod
    def _artifact_revision(path: str) -> tuple[str, int, int]:
        absolute = os.path.abspath(path) if path else ""
        try:
            stat = os.stat(absolute)
            return absolute, int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return absolute, -1, -1

    def revision_signature(self) -> tuple[Any, ...]:
        """Describe constituent availability and content, including same-path updates."""

        self.refresh()
        return tuple(
            (
                key,
                str(value.get("status", "")),
                float(value.get("convergence", 0.0)),
                int(value.get("samples", 0)),
                self._artifact_revision(str(value.get("artifact_path", ""))),
            )
            for key, value in sorted(self._entries.items())
        )

    def snapshot(self) -> Mapping[str, Mapping[str, Any]]:
        self.refresh()
        return {
            key: dict(value) for key, value in self._entries.items()
        }

    def update(
        self,
        cache_key: str,
        *,
        status: str,
        convergence: float = 0.0,
        samples: int = 0,
        artifact_path: str = "",
    ) -> None:
        path = os.path.abspath(artifact_path) if artifact_path else ""
        self._entries[str(cache_key)] = {
            "status": str(status),
            "convergence": max(0.0, min(1.0, float(convergence))),
            "samples": max(0, int(samples)),
            "artifact_path": path,
            "updated_at_s": time.time(),
        }
        _atomic_json(self.path, {
            "schema_version": LAYOUT_WORK_CACHE_SCHEMA_VERSION,
            "entries": self._entries,
        })
        self._cache_revision = self._disk_revision()

    def entry(self, cache_key: str) -> Mapping[str, Any]:
        self.refresh()
        return dict(self._entries.get(str(cache_key), {}))

    def replacement_path(
        self, subtype: LayoutDesignSubtypeWork
    ) -> str:
        self.refresh()
        keys = [
            animation.replacement_cache_key
            for animation in subtype.pose_animations
        ]
        keys.append(subtype.still_cache_key)
        for cache_key in keys:
            entry = self._entries.get(cache_key, {})
            path = str(entry.get("artifact_path", ""))
            status = str(entry.get("status", ""))
            has_evidence = (
                float(entry.get("convergence", 0.0)) > 0.0
                or int(entry.get("samples", 0)) > 0
            )
            if status not in {"missing", "failed", "rejected"} and (
                has_evidence and path and os.path.isfile(path)
            ):
                return path
        return ""

    def patch_loader(
        self, manifest: LayoutObjectWorkManifest
    ):
        subtype_map = {
            (item.object_key, subtype.subtype_key, subtype.condition_key): subtype
            for item in manifest.objects for subtype in item.subtypes
        }

        def load(patch: PanelPatchAsset) -> np.ndarray | None:
            subtype = subtype_map.get((
                str(patch.object_key),
                str(patch.subtype_key),
                str(patch.condition_key),
            ))
            if subtype is None:
                return None
            path = self.replacement_path(subtype)
            if not path:
                return None
            from PIL import Image

            try:
                with Image.open(path) as image:
                    return np.asarray(
                        image.convert("RGBA"), dtype=np.uint8
                    ).copy()
            except (OSError, ValueError):
                # A partially published or corrupt constituent never takes
                # down the UI; the panel primitive remains as its fallback.
                return None

        return load


__all__ = [
    "LAYOUT_OBJECT_WORK_SCHEMA_VERSION",
    "LAYOUT_WORK_CACHE_SCHEMA_VERSION",
    "LayoutObjectConsumer",
    "PoseFrameWork",
    "PoseAnimationWork",
    "LayoutDesignSubtypeWork",
    "LayoutDesignObjectWork",
    "LayoutObjectWorkManifest",
    "LayoutAspectVariant",
    "rack_focus_pose_animation",
    "build_layout_object_work_manifest",
    "LayoutWorkProgressCache",
]
