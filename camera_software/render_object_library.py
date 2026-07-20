"""Canonical style objects containing renderable text subtypes and evidence."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from .display_scene import DisplayProductKind
from .render_assets import (
    DEFAULT_INK_CONDITION,
    ExtrudedTokenAsset,
    FontAssetSpec,
    LightFieldCondition,
    RenderAssetCatalog,
    RenderedAssetRecord,
    RotatingStageSpec,
    ink_token_asset,
    stable_asset_key,
)

RENDER_OBJECT_LIBRARY_SCHEMA_VERSION = 3
RENDER_OBJECT_BUNDLE_SCHEMA_VERSION = 3


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    return value


def _atomic_json(path: str, payload: Mapping[str, Any]) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(_canonical(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing(path: str) -> str:
    absolute = os.path.abspath(str(path)) if path else ""
    return absolute if absolute and os.path.isfile(absolute) else ""


def render_style_spec(asset: ExtrudedTokenAsset) -> dict[str, Any]:
    """Identity shared by glyph and whole-token subtypes of one visual style."""
    return {
        "family": "ray-traced-text",
        "font": asset.font.scene_order_mapping(),
        "font_identity": asset.font.identity_payload(),
        "material": asset.material,
        "material_revision": asset.material_revision,
        "stage_family": "ink-on-slate-v1",
    }


def render_style_key(asset: ExtrudedTokenAsset) -> str:
    return stable_asset_key("render-style", render_style_spec(asset))


def render_subtype_kind(token: str, target_kind: str = "") -> str:
    return "glyph" if target_kind == "atlas_glyph" or len(str(token)) == 1 else "token"


def render_subtype_key(kind: str, token: str, source_asset_key: str) -> str:
    return stable_asset_key("render-subtype", (str(kind), str(token), str(source_asset_key)))


def canonical_subtype_directory(
    output_root: str,
    asset: ExtrudedTokenAsset,
    condition: LightFieldCondition,
    *,
    target_kind: str = "",
) -> str:
    style = render_style_key(asset).rsplit(":", 1)[-1]
    kind = render_subtype_kind(asset.token, target_kind)
    subtype = render_subtype_key(kind, asset.token, asset.asset_key).rsplit(":", 1)[-1]
    condition_id = condition.condition_key.rsplit(":", 1)[-1]
    return os.path.join(
        os.path.abspath(output_root), "render_objects", style,
        "subtypes", kind, subtype, "conditions", condition_id,
    )


class ObjectViewMode(str, Enum):
    SCENE = "scene"
    LIGHT_FIELD = "light_field"
    IMAGE = "image"


@dataclass(frozen=True)
class ArchivedObjectScene:
    subtype_key: str
    condition_key: str
    scene_path: str
    scene_digest: str
    condition_spec: Mapping[str, Any] = field(default_factory=dict)
    camera_spec: Mapping[str, Any] = field(default_factory=dict)
    stage_spec: Mapping[str, Any] = field(default_factory=dict)
    lighting_spec: Mapping[str, Any] = field(default_factory=dict)
    material_specs: Mapping[str, Any] = field(default_factory=dict)
    exposure_spec: Mapping[str, Any] = field(default_factory=dict)
    geometry_spec: Mapping[str, Any] = field(default_factory=dict)
    font_spec: Mapping[str, Any] = field(default_factory=dict)
    object_template: Mapping[str, Any] = field(default_factory=dict)
    resolved_summary_path: str = ""
    renderer_context_path: str = ""
    render_log_path: str = ""
    inferred_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectProductArtifact:
    subtype_key: str
    record_key: str
    source_asset_key: str
    condition_key: str
    product_kind: DisplayProductKind
    linear_path: str = ""
    preview_path: str = ""
    manifest_path: str = ""
    sprite_path: str = ""
    data_paths: Mapping[str, str] = field(default_factory=dict)
    width: int = 0
    height: int = 0
    samples: int = 0
    created_at_s: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderObjectSubtype:
    subtype_key: str
    kind: str
    text: str
    source_asset_key: str
    display_name: str
    scenes: tuple[ArchivedObjectScene, ...] = ()
    artifacts: tuple[ObjectProductArtifact, ...] = ()
    tags: tuple[str, ...] = ()
    complete: bool = False
    created_at_s: float = field(default_factory=time.time)
    updated_at_s: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RenderObjectBundle:
    object_key: str
    bundle_key: str
    display_name: str
    object_kind: str
    style_spec: Mapping[str, Any] = field(default_factory=dict)
    subtypes: tuple[RenderObjectSubtype, ...] = ()
    subtype_sets: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    canonical_directory: str = ""
    manifest_path: str = ""
    created_at_s: float = field(default_factory=time.time)
    updated_at_s: float = field(default_factory=time.time)

    @property
    def available_view_modes(self) -> tuple[ObjectViewMode, ...]:
        scenes = any(subtype.scenes for subtype in self.subtypes)
        artifacts = any(subtype.artifacts for subtype in self.subtypes)
        modes = [ObjectViewMode.SCENE] if scenes else []
        if artifacts:
            modes.extend((ObjectViewMode.IMAGE, ObjectViewMode.LIGHT_FIELD))
        return tuple(modes)


@dataclass(frozen=True)
class ObjectViewSelection:
    object_key: str
    bundle_key: str
    subtype_key: str
    subtype_kind: str
    text: str
    mode: ObjectViewMode
    condition_key: str
    scene_path: str = ""
    primary_path: str = ""
    texture_paths: tuple[str, ...] = ()
    artifact_record_keys: tuple[str, ...] = ()
    fallback_used: bool = False


@dataclass(frozen=True)
class SceneSubtypeComposition:
    object_key: str
    scene_objects: tuple[Mapping[str, Any], ...]
    used_subtypes: tuple[str, ...]
    missing_tokens: tuple[str, ...]
    missing_characters: tuple[str, ...]

@dataclass(frozen=True)
class InterfaceComponentReference:
    """One reusable subtype used by an object in the assembled interface."""

    scene_object_id: str
    object_key: str
    subtype_key: str
    subtype_kind: str
    text: str
    text_offset: int = 0


@dataclass(frozen=True)
class InterfaceAfterRender:
    """Terminal shared-camera render with every interface element in place."""

    render_key: str
    assembly_key: str
    interface_id: str
    revision: int
    text: str
    scene_path: str
    scene_digest: str
    image_path: str
    linear_path: str
    manifest_path: str
    component_references: tuple[InterfaceComponentReference, ...] = ()
    scene_object_ids: tuple[str, ...] = ()
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    completed_at_s: float = field(default_factory=time.time)


@dataclass(frozen=True)
class InterfaceAssembly:
    """A growing revision history of final in-place interface renders."""

    assembly_key: str
    interface_id: str
    display_name: str
    after_renders: tuple[InterfaceAfterRender, ...] = ()
    manifest_path: str = ""
    created_at_s: float = field(default_factory=time.time)
    updated_at_s: float = field(default_factory=time.time)

    @property
    def latest(self) -> InterfaceAfterRender | None:
        return max(
            self.after_renders,
            key=lambda item: (item.revision, item.completed_at_s),
            default=None,
        )

def _read_scene(path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = dict(json.load(handle))
    defaults = dict(payload.get("defaults", {}))
    jobs = tuple(payload.get("jobs", ()))
    return payload, defaults, dict(jobs[0]) if jobs else {}


def _asset_from_scene(scene_path: str, record: RenderedAssetRecord) -> tuple[ExtrudedTokenAsset, str, str]:
    _payload, defaults, job = _read_scene(scene_path)
    token = str(job.get("token", ""))
    font = FontAssetSpec.from_mapping(job.get("font", defaults.get("font", {})))
    target_kind = str(record.metadata.get("target_kind", ""))
    asset = ink_token_asset(token, font=font, character=render_subtype_kind(token, target_kind) == "glyph")
    return asset, token, target_kind


def _copy_file(source: str, destination_dir: str) -> str:
    source = _existing(source)
    if not source:
        return ""
    destination = os.path.join(destination_dir, os.path.basename(source))
    os.makedirs(destination_dir, exist_ok=True)
    if os.path.normcase(source) != os.path.normcase(os.path.abspath(destination)):
        shutil.copy2(source, destination)
    return os.path.abspath(destination)


def _record_paths(record: RenderedAssetRecord, scene_path: str) -> dict[str, str]:
    metadata = dict(record.metadata)
    directory = os.path.dirname(record.linear_path or record.preview_path or scene_path)
    candidates = {
        "scene": scene_path,
        "linear": record.linear_path,
        "preview": record.preview_path,
        "manifest": record.manifest_path,
        "sprite": str(metadata.get("sprite_path", "")),
        "sensor_sum": str(metadata.get("sum_linear_path", "")),
        "exposure_weight": str(metadata.get("exposure_weight_path", "")),
        "priority": os.path.join(directory, "0000_cpp_priority.npy"),
        "image_16bit": os.path.join(directory, "0000_cpp_16bit.png"),
        "field_integral": os.path.join(directory, "0000_cpp_field_integral.npz"),
        "surface_integral": os.path.join(directory, "0000_cpp_surface_integral.npz"),
        "field_capture_grid": os.path.join(directory, "0000_cpp_field_capture_grid_reim.npy"),
        "field_capture_strikes": os.path.join(directory, "0000_cpp_field_capture_strikes.npy"),
        "renderer_summary": os.path.join(directory, "0000_cpp_summary.json"),
        "render_log": os.path.join(directory, "render.log"),
        "renderer_context": os.path.join(directory, "render_context.json"),
        "checkpoint": os.path.join(directory, "active_epoch_checkpoint.json"),
    }
    return {name: path for name, path in candidates.items() if _existing(path)}


def _canonicalize_record(
    record: RenderedAssetRecord, scene_path: str, destination: str
) -> tuple[RenderedAssetRecord, str, Mapping[str, str]]:
    source_directory = os.path.dirname(os.path.abspath(scene_path))
    copied = {
        name: _copy_file(path, destination)
        for name, path in _record_paths(record, scene_path).items()
    }
    copied = {name: path for name, path in copied.items() if path}
    for filename in os.listdir(source_directory):
        source = os.path.join(source_directory, filename)
        if not os.path.isfile(source) or filename == "object_bundle.json":
            continue
        copied.setdefault(
            f"cache:{filename}", _copy_file(source, destination)
        )
    metadata = dict(record.metadata)
    aliases = {
        "sprite_path": "sprite",
        "sum_linear_path": "sensor_sum",
        "exposure_weight_path": "exposure_weight",
    }
    for metadata_key, data_key in aliases.items():
        if data_key in copied:
            metadata[metadata_key] = copied[data_key]
    canonical_record = replace(
        record,
        linear_path=copied.get("linear", ""),
        preview_path=copied.get("preview", ""),
        manifest_path=copied.get("manifest", ""),
        metadata=metadata,
    )
    return canonical_record, copied["scene"], copied


def _archive_scene(
    scene_path: str, subtype_key: str, condition: LightFieldCondition, paths: Mapping[str, str]
) -> ArchivedObjectScene:
    _payload, defaults, job = _read_scene(scene_path)
    stage = RotatingStageSpec(azimuth_views=1, elevations_deg=(condition.elevation_deg,), light_rigs=(condition.light_rig,), material_variants=(condition.material_variant,))
    return ArchivedObjectScene(
        subtype_key=subtype_key,
        condition_key=condition.condition_key,
        scene_path=scene_path,
        scene_digest=_file_digest(scene_path),
        condition_spec=asdict(condition),
        camera_spec=dict(defaults.get("camera", {})),
        stage_spec=asdict(stage),
        lighting_spec={"light_rig": condition.light_rig, "flash": dict(defaults.get("flash", {}))},
        material_specs=dict(defaults.get("materials", {})),
        exposure_spec=dict(defaults.get("exposure", {})),
        geometry_spec=dict(job.get("geometry", defaults.get("geometry", {}))),
        font_spec=dict(job.get("font", defaults.get("font", {}))),
        object_template=job,
        resolved_summary_path=paths.get("renderer_summary", ""),
        renderer_context_path=paths.get("renderer_context", ""),
        render_log_path=paths.get("render_log", ""),
        inferred_fields=("stage_spec", "camera_simulator_defaults_outside_scene_order"),
    )


class RenderObjectLibrary:
    """Style-object catalog with glyph and whole-token subtype collections."""

    def __init__(self, path: str = "") -> None:
        self.path = os.path.abspath(path) if path else ""
        base = os.path.dirname(self.path) if self.path else os.getcwd()
        self.object_root = os.path.join(base, "render_objects")
        self.interface_root = os.path.join(base, "render_interfaces")
        self._lock = threading.RLock()
        self._bundles: dict[str, RenderObjectBundle] = {}
        self._interfaces: dict[str, InterfaceAssembly] = {}
        if self.path and os.path.isfile(self.path):
            self._load()

    def snapshot(self) -> tuple[RenderObjectBundle, ...]:
        with self._lock:
            return tuple(self._bundles[key] for key in sorted(self._bundles))

    def find(self, object_key: str) -> RenderObjectBundle | None:
        with self._lock:
            return self._bundles.get(str(object_key))

    def accept_visual_pass(self, record_key: str) -> RenderObjectSubtype:
        """Mirror catalog visual acceptance into its rich object subtype."""

        target = str(record_key)
        with self._lock:
            for object_key, bundle in self._bundles.items():
                for subtype_index, subtype in enumerate(bundle.subtypes):
                    if not any(
                        artifact.record_key == target
                        for artifact in subtype.artifacts
                    ):
                        continue
                    accepted_artifacts = tuple(
                        replace(
                            artifact,
                            metadata={
                                **dict(artifact.metadata),
                                "completion_basis": "human_visual_pass",
                                "human_visual_pass": True,
                            },
                        )
                        if artifact.record_key == target else artifact
                        for artifact in subtype.artifacts
                    )
                    accepted_subtype = replace(
                        subtype,
                        artifacts=accepted_artifacts,
                        complete=True,
                        updated_at_s=time.time(),
                    )
                    subtypes = list(bundle.subtypes)
                    subtypes[subtype_index] = accepted_subtype
                    accepted_bundle = replace(
                        bundle,
                        subtypes=tuple(subtypes),
                        updated_at_s=time.time(),
                    )
                    self._bundles[object_key] = accepted_bundle
                    self._save_bundle(accepted_bundle)
                    self._save()
                    return accepted_subtype
        raise KeyError(target)

    def interface_snapshot(self) -> tuple[InterfaceAssembly, ...]:
        with self._lock:
            return tuple(self._interfaces[key] for key in sorted(self._interfaces))

    def find_interface(self, interface_id_or_key: str) -> InterfaceAssembly | None:
        value = str(interface_id_or_key)
        with self._lock:
            direct = self._interfaces.get(value)
            if direct is not None:
                return direct
            return next((item for item in self._interfaces.values() if item.interface_id == value), None)

    def find_subtype(self, object_key: str, text: str, *, kind: str = "") -> RenderObjectSubtype | None:
        bundle = self.find(object_key)
        if bundle is None:
            return None
        candidates = [s for s in bundle.subtypes if s.text == str(text) and (not kind or s.kind == kind)]
        return candidates[0] if candidates else None

    def adopt_record(
        self,
        record: RenderedAssetRecord,
        *,
        scene_path: str,
        condition: LightFieldCondition = DEFAULT_INK_CONDITION,
        display_name: str = "",
        object_kind: str = "render_style",
        token: str = "",
        tags: Sequence[str] = (),
        renderer_context_path: str = "",
        save: bool = True,
    ) -> RenderObjectBundle:
        if record.condition_key != condition.condition_key:
            raise ValueError("record condition does not match archived condition")
        asset, scene_token, target_kind = _asset_from_scene(scene_path, record)
        text = str(token or scene_token or asset.token)
        kind = render_subtype_kind(text, target_kind)
        object_key = render_style_key(asset)
        subtype_key = render_subtype_key(kind, text, record.target_key)
        destination = canonical_subtype_directory(
            os.path.dirname(self.object_root), asset, condition, target_kind=target_kind
        )
        canonical_record, canonical_scene, paths = _canonicalize_record(record, scene_path, destination)
        scene = _archive_scene(canonical_scene, subtype_key, condition, paths)
        artifact = ObjectProductArtifact(
            subtype_key=subtype_key,
            record_key=canonical_record.record_key,
            source_asset_key=canonical_record.target_key,
            condition_key=canonical_record.condition_key,
            product_kind=canonical_record.product_kind,
            linear_path=canonical_record.linear_path,
            preview_path=canonical_record.preview_path,
            manifest_path=canonical_record.manifest_path,
            sprite_path=str(canonical_record.metadata.get("sprite_path", "")),
            data_paths=paths,
            width=canonical_record.width,
            height=canonical_record.height,
            samples=canonical_record.samples,
            created_at_s=canonical_record.created_at_s,
            metadata=dict(canonical_record.metadata),
        )
        with self._lock:
            current = self._bundles.get(object_key)
            subtypes = {s.subtype_key: s for s in (() if current is None else current.subtypes)}
            prior = subtypes.get(subtype_key)
            scenes = {s.condition_key: s for s in (() if prior is None else prior.scenes)}
            artifacts = {a.record_key: a for a in (() if prior is None else prior.artifacts)}
            scenes[scene.condition_key] = scene
            artifacts[artifact.record_key] = artifact
            now = time.time()
            subtype = RenderObjectSubtype(
                subtype_key=subtype_key, kind=kind, text=text,
                source_asset_key=record.target_key, display_name=text,
                scenes=tuple(scenes[k] for k in sorted(scenes)),
                artifacts=tuple(artifacts[k] for k in sorted(artifacts)),
                tags=tuple(sorted(set((*(() if prior is None else prior.tags), *map(str, tags))))),
                complete=any(bool(a.metadata.get("atlas_quality", {}).get("converged", False)) or a.metadata.get("completion_basis") for a in artifacts.values()),
                created_at_s=prior.created_at_s if prior else now, updated_at_s=now,
            )
            subtypes[subtype_key] = subtype
            sets: dict[str, tuple[str, ...]] = {}
            for subtype_kind in sorted({s.kind for s in subtypes.values()} | {"glyph", "token"}):
                sets[subtype_kind] = tuple(sorted(s.subtype_key for s in subtypes.values() if s.kind == subtype_kind))
            canonical_directory = os.path.join(self.object_root, object_key.rsplit(":", 1)[-1])
            bundle = RenderObjectBundle(
                object_key=object_key,
                bundle_key=stable_asset_key("render-object-bundle", object_key),
                display_name=str(display_name) or (current.display_name if current else "") or f"{asset.font.family} {asset.font.weight} {asset.material}",
                object_kind="render_style",
                style_spec=render_style_spec(asset), subtypes=tuple(subtypes[k] for k in sorted(subtypes)),
                subtype_sets=sets,
                tags=tuple(sorted(set((*(() if current is None else current.tags), "text-style", *map(str, tags))))),
                canonical_directory=canonical_directory,
                manifest_path=os.path.join(canonical_directory, "object.json"),
                created_at_s=current.created_at_s if current else now, updated_at_s=now,
            )
            self._bundles[object_key] = bundle
            if save:
                self._save_bundle(bundle)
                self._save()
            return bundle

    def adopt_records(self, records: Sequence[RenderedAssetRecord], *, condition: LightFieldCondition = DEFAULT_INK_CONDITION) -> tuple[RenderObjectBundle, ...]:
        adopted: dict[str, RenderObjectBundle] = {}
        for record in records:
            if record.condition_key != condition.condition_key:
                continue
            directory = os.path.dirname(record.linear_path or record.preview_path or record.manifest_path)
            scene_path = os.path.join(directory, "scene_order.json")
            if not os.path.isfile(scene_path):
                continue
            bundle = self.adopt_record(record, scene_path=scene_path, condition=condition, tags=("migrated-cache",), save=False)
            adopted[bundle.object_key] = bundle
        with self._lock:
            for bundle in adopted.values():
                self._save_bundle(bundle)
            if adopted:
                self._save()
        return tuple(adopted[key] for key in sorted(adopted))

    def migrate_catalog(self, catalog: RenderAssetCatalog, *, condition: LightFieldCondition = DEFAULT_INK_CONDITION) -> tuple[RenderObjectBundle, ...]:
        bundles = self.adopt_records(catalog.snapshot(), condition=condition)
        for bundle in bundles:
            for subtype in bundle.subtypes:
                for artifact in subtype.artifacts:
                    catalog.record(RenderedAssetRecord(
                        artifact.source_asset_key, artifact.condition_key, artifact.product_kind,
                        linear_path=artifact.linear_path, preview_path=artifact.preview_path,
                        manifest_path=artifact.manifest_path, width=artifact.width, height=artifact.height,
                        samples=artifact.samples, created_at_s=artifact.created_at_s,
                        metadata=dict(artifact.metadata),
                    ))
        return bundles

    def select_view(self, object_key: str, mode: ObjectViewMode, *, subtype_key: str = "", text: str = "", kind: str = "", condition_key: str = "") -> ObjectViewSelection:
        bundle = self.find(object_key)
        if bundle is None:
            raise KeyError(object_key)
        subtype = next((s for s in bundle.subtypes if subtype_key and s.subtype_key == subtype_key), None)
        subtype = subtype or self.find_subtype(object_key, text, kind=kind) if text else subtype
        subtype = subtype or (bundle.subtypes[0] if bundle.subtypes else None)
        if subtype is None:
            raise LookupError("render style has no subtypes")
        selected_condition = condition_key or (subtype.scenes[0].condition_key if subtype.scenes else "")
        scene = next((s for s in subtype.scenes if s.condition_key == selected_condition), subtype.scenes[0] if subtype.scenes else None)
        mode = ObjectViewMode(mode)
        if mode is ObjectViewMode.SCENE:
            if scene is None:
                raise LookupError("subtype has no archived scene")
            return ObjectViewSelection(object_key, bundle.bundle_key, subtype.subtype_key, subtype.kind, subtype.text, mode, scene.condition_key, scene_path=scene.scene_path, primary_path=scene.scene_path)
        exact = tuple(a for a in subtype.artifacts if not selected_condition or a.condition_key == selected_condition)
        if mode is ObjectViewMode.IMAGE:
            candidates = tuple(a for a in exact if a.product_kind is DisplayProductKind.IMAGE) or exact
            if not candidates:
                raise LookupError("subtype has no image product")
            artifact = candidates[0]
            primary = artifact.preview_path or artifact.linear_path or artifact.sprite_path
            return ObjectViewSelection(object_key, bundle.bundle_key, subtype.subtype_key, subtype.kind, subtype.text, mode, artifact.condition_key, scene_path=scene.scene_path if scene else "", primary_path=primary, texture_paths=(primary,) if primary else (), artifact_record_keys=(artifact.record_key,), fallback_used=artifact.product_kind is not DisplayProductKind.IMAGE)
        light_field = tuple(a for a in subtype.artifacts if a.product_kind is DisplayProductKind.LIGHT_FIELD)
        candidates = light_field or subtype.artifacts
        if not candidates:
            raise LookupError("subtype has no textured product")
        textures = tuple(path for a in candidates if (path := a.preview_path or a.linear_path or a.sprite_path))
        return ObjectViewSelection(object_key, bundle.bundle_key, subtype.subtype_key, subtype.kind, subtype.text, mode, selected_condition, scene_path=scene.scene_path if scene else "", primary_path=textures[0] if textures else "", texture_paths=textures, artifact_record_keys=tuple(a.record_key for a in candidates), fallback_used=not bool(light_field))

    def sprite_catalog(self, object_key: str) -> RenderAssetCatalog:
        bundle = self.find(object_key)
        if bundle is None:
            raise KeyError(object_key)
        catalog = RenderAssetCatalog()
        for subtype in bundle.subtypes:
            for artifact in subtype.artifacts:
                catalog.record(RenderedAssetRecord(
                    artifact.source_asset_key, artifact.condition_key, artifact.product_kind,
                    linear_path=artifact.linear_path, preview_path=artifact.preview_path,
                    manifest_path=artifact.manifest_path, width=artifact.width, height=artifact.height,
                    samples=artifact.samples, created_at_s=artifact.created_at_s, metadata=dict(artifact.metadata),
                ))
        return catalog

    def compose_sprite(
        self,
        object_key: str,
        text: str,
        width: int,
        height: int,
        *,
        horizontal_spacing_px: int = -10,
        vertical_spacing_px: int = -10,
        **kwargs: Any,
    ):
        from .sprite_compositor import CachedTokenStringComposer
        bundle = self.find(object_key)
        if bundle is None:
            raise KeyError(object_key)
        font = FontAssetSpec.from_mapping(dict(bundle.style_spec.get("font", {})))
        composer = CachedTokenStringComposer(
            self.sprite_catalog(object_key),
            font=font,
            horizontal_spacing_px=horizontal_spacing_px,
            vertical_spacing_px=vertical_spacing_px,
        )
        return composer.compose(text, width, height, **kwargs)

    def compose_scene(self, object_key: str, text: str, *, object_id_prefix: str = "composed-text", embed_plane: str = "backplate", origin_m: tuple[float, float] = (0.0, 0.0), character_advance_m: float | None = None, prefer_whole_tokens: bool = True) -> SceneSubtypeComposition:
        bundle = self.find(object_key)
        if bundle is None:
            raise KeyError(object_key)
        by_text = {(s.kind, s.text): s for s in bundle.subtypes}
        pieces: list[RenderObjectSubtype | None] = []
        missing_tokens: set[str] = set()
        missing_characters: set[str] = set()
        import re
        for segment in re.findall(r"\s+|\S+", str(text)):
            if segment.isspace():
                pieces.extend([None] * len(segment))
                continue
            exact = by_text.get(("token", segment)) if prefer_whole_tokens else None
            if exact is not None:
                pieces.append(exact)
                continue
            if len(segment) > 1:
                missing_tokens.add(segment)
            for character in segment:
                glyph = by_text.get(("glyph", character))
                if glyph is None:
                    missing_characters.add(character)
                pieces.append(glyph)
        default_height = 0.2
        first_scene = next((s.scenes[0] for s in pieces if s is not None and s.scenes), None)
        if first_scene:
            default_height = float(first_scene.geometry_spec.get("height_m", default_height))
        advance = float(character_advance_m if character_advance_m is not None else default_height * 0.72)
        objects: list[Mapping[str, Any]] = []
        used: list[str] = []
        cursor = 0
        line = 0
        for subtype in pieces:
            if subtype is None:
                cursor += 1
                continue
            scene = subtype.scenes[0] if subtype.scenes else None
            if scene is None:
                cursor += max(1, len(subtype.text))
                continue
            template = dict(scene.object_template)
            template["id"] = f"{object_id_prefix}-{len(objects):04d}"
            template["token"] = subtype.text
            template["embed_plane"] = embed_plane
            geometry = dict(template.get("geometry", scene.geometry_spec))
            geometry["embed_plane"] = embed_plane
            geometry["offset_m"] = [float(origin_m[0]) + cursor * advance, float(origin_m[1]) - line * default_height]
            template["geometry"] = geometry
            objects.append(template)
            used.append(subtype.subtype_key)
            cursor += max(1, len(subtype.text))
        return SceneSubtypeComposition(object_key, tuple(objects), tuple(used), tuple(sorted(missing_tokens)), tuple(sorted(missing_characters)))

    def _resolve_interface_components(
        self, scene_payload: Mapping[str, Any]
    ) -> tuple[InterfaceComponentReference, ...]:
        objects = tuple(dict(scene_payload.get("defaults", {})).get("objects", ()))
        references: list[InterfaceComponentReference] = []
        for raw_object in objects:
            scene_object = dict(raw_object)
            object_id = str(scene_object.get("id", ""))
            token = str(scene_object.get("token", ""))
            exact = next(
                (
                    (bundle, subtype)
                    for bundle in self.snapshot()
                    for subtype in bundle.subtypes
                    if subtype.kind == "token" and subtype.text == token
                ),
                None,
            )
            if exact is not None:
                bundle, subtype = exact
                references.append(InterfaceComponentReference(
                    object_id, bundle.object_key, subtype.subtype_key,
                    subtype.kind, token, 0,
                ))
                continue
            for offset, character in enumerate(token):
                if character.isspace():
                    continue
                found = next(
                    (
                        (bundle, subtype)
                        for bundle in self.snapshot()
                        for subtype in bundle.subtypes
                        if subtype.kind == "glyph" and subtype.text == character
                    ),
                    None,
                )
                if found is None:
                    continue
                bundle, subtype = found
                references.append(InterfaceComponentReference(
                    object_id, bundle.object_key, subtype.subtype_key,
                    subtype.kind, character, offset,
                ))
        return tuple(references)

    def register_interface_after_render(
        self,
        interface_id: str,
        revision: int,
        *,
        scene_path: str,
        image_path: str,
        linear_path: str,
        manifest_path: str = "",
        text: str = "",
        display_name: str = "",
        diagnostic_image_path: str = "",
        orthographic_path: str = "",
        priority_overlay_path: str = "",
        render_log_path: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> InterfaceAfterRender:
        """Commit the terminal render of the fully assembled shared-camera UI."""

        scene_path = _existing(scene_path)
        image_path = _existing(image_path)
        linear_path = _existing(linear_path)
        if not scene_path or not image_path or not linear_path:
            raise ValueError("final interface render requires scene, image, and linear evidence")
        with open(scene_path, "r", encoding="utf-8") as handle:
            scene_payload = dict(json.load(handle))
        interface_id = str(interface_id).strip()
        if not interface_id:
            raise ValueError("final interface render requires an interface id")
        assembly_key = stable_asset_key("interface-assembly", interface_id)
        scene_digest = _file_digest(scene_path)
        render_key = stable_asset_key(
            "interface-after-render",
            (assembly_key, int(revision), scene_digest, str(text)),
        )
        destination = os.path.join(
            self.interface_root,
            assembly_key.rsplit(":", 1)[-1],
            "after_renders",
            render_key.rsplit(":", 1)[-1],
        )
        source_root = os.path.dirname(scene_path)
        if os.path.normcase(source_root) != os.path.normcase(destination):
            shutil.copytree(source_root, destination, dirs_exist_ok=True)

        def canonical_path(path: str) -> str:
            source = _existing(path)
            if not source:
                return ""
            try:
                relative = os.path.relpath(source, source_root)
                if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
                    candidate = os.path.abspath(os.path.join(destination, relative))
                    if os.path.isfile(candidate):
                        return candidate
            except ValueError:
                pass
            return _copy_file(source, destination)

        canonical_scene = canonical_path(scene_path)
        artifacts = {
            name: path
            for name, raw_path in {
                "diagnostic_image": diagnostic_image_path,
                "orthographic": orthographic_path,
                "priority_overlay": priority_overlay_path,
                "render_log": render_log_path,
            }.items()
            if (path := canonical_path(raw_path))
        }
        final = InterfaceAfterRender(
            render_key=render_key,
            assembly_key=assembly_key,
            interface_id=interface_id,
            revision=int(revision),
            text=str(text),
            scene_path=canonical_scene,
            scene_digest=scene_digest,
            image_path=canonical_path(image_path),
            linear_path=canonical_path(linear_path),
            manifest_path=canonical_path(manifest_path),
            component_references=self._resolve_interface_components(scene_payload),
            scene_object_ids=tuple(
                str(item.get("id", ""))
                for item in dict(scene_payload.get("defaults", {})).get("objects", ())
                if str(item.get("id", ""))
            ),
            artifact_paths=artifacts,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            current = self._interfaces.get(assembly_key)
            renders = {
                item.render_key: item
                for item in (() if current is None else current.after_renders)
            }
            renders[render_key] = final
            now = time.time()
            directory = os.path.join(
                self.interface_root, assembly_key.rsplit(":", 1)[-1]
            )
            assembly = InterfaceAssembly(
                assembly_key=assembly_key,
                interface_id=interface_id,
                display_name=str(display_name) or (
                    current.display_name if current else interface_id
                ),
                after_renders=tuple(renders[key] for key in sorted(renders)),
                manifest_path=os.path.join(directory, "interface.json"),
                created_at_s=current.created_at_s if current else now,
                updated_at_s=now,
            )
            self._interfaces[assembly_key] = assembly
            self._save_interface(assembly)
            self._save()
        return final

    def select_interface_after_render(
        self, interface_id_or_key: str, *, render_key: str = ""
    ) -> InterfaceAfterRender:
        assembly = self.find_interface(interface_id_or_key)
        if assembly is None:
            raise KeyError(interface_id_or_key)
        if render_key:
            selected = next(
                (item for item in assembly.after_renders if item.render_key == render_key),
                None,
            )
            if selected is None:
                raise KeyError(render_key)
            return selected
        latest = assembly.latest
        if latest is None:
            raise LookupError("interface assembly has no final after-render")
        return latest

    def _save_interface(self, assembly: InterfaceAssembly) -> None:
        _atomic_json(assembly.manifest_path, {
            "schema_version": RENDER_OBJECT_BUNDLE_SCHEMA_VERSION,
            "interface": assembly,
        })

    def _save_bundle(self, bundle: RenderObjectBundle) -> None:
        _atomic_json(bundle.manifest_path, {"schema_version": RENDER_OBJECT_BUNDLE_SCHEMA_VERSION, "bundle": bundle})
        for subtype in bundle.subtypes:
            if subtype.scenes:
                subtype_dir = os.path.dirname(os.path.dirname(os.path.dirname(subtype.scenes[0].scene_path)))
                _atomic_json(os.path.join(subtype_dir, "subtype.json"), {"schema_version": RENDER_OBJECT_BUNDLE_SCHEMA_VERSION, "object_key": bundle.object_key, "subtype": subtype})

    def _save(self) -> None:
        if self.path:
            _atomic_json(self.path, {"schema_version": RENDER_OBJECT_LIBRARY_SCHEMA_VERSION, "objects": self.snapshot(), "interfaces": self.interface_snapshot()})

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        version = int(payload.get("schema_version", -1))
        if version == 1:
            self._bundles = {}
            return
        if version not in {2, RENDER_OBJECT_LIBRARY_SCHEMA_VERSION}:
            raise ValueError("unsupported render object library schema")
        bundles: dict[str, RenderObjectBundle] = {}
        for raw_bundle in payload.get("objects", ()):
            item = dict(raw_bundle)
            subtypes = []
            for raw_subtype in item.get("subtypes", ()):
                subtype = dict(raw_subtype)
                subtype["scenes"] = tuple(ArchivedObjectScene(**scene) for scene in subtype.get("scenes", ()))
                artifacts = []
                for raw_artifact in subtype.get("artifacts", ()):
                    artifact = dict(raw_artifact)
                    artifact["product_kind"] = DisplayProductKind(artifact["product_kind"])
                    artifacts.append(ObjectProductArtifact(**artifact))
                subtype["artifacts"] = tuple(artifacts)
                subtype["tags"] = tuple(subtype.get("tags", ()))
                subtypes.append(RenderObjectSubtype(**subtype))
            item["subtypes"] = tuple(subtypes)
            item["subtype_sets"] = {k: tuple(v) for k, v in dict(item.get("subtype_sets", {})).items()}
            item["tags"] = tuple(item.get("tags", ()))
            bundle = RenderObjectBundle(**item)
            bundles[bundle.object_key] = bundle
        self._bundles = bundles
        interfaces: dict[str, InterfaceAssembly] = {}
        for raw_assembly in payload.get("interfaces", ()):
            item = dict(raw_assembly)
            renders = []
            for raw_render in item.get("after_renders", ()):
                rendered = dict(raw_render)
                rendered["component_references"] = tuple(
                    InterfaceComponentReference(**reference)
                    for reference in rendered.get("component_references", ())
                )
                rendered["scene_object_ids"] = tuple(
                    rendered.get("scene_object_ids", ())
                )
                renders.append(InterfaceAfterRender(**rendered))
            item["after_renders"] = tuple(renders)
            assembly = InterfaceAssembly(**item)
            interfaces[assembly.assembly_key] = assembly
        self._interfaces = interfaces


__all__ = [
    "RENDER_OBJECT_LIBRARY_SCHEMA_VERSION", "RENDER_OBJECT_BUNDLE_SCHEMA_VERSION",
    "ObjectViewMode", "ArchivedObjectScene", "ObjectProductArtifact",
    "RenderObjectSubtype", "RenderObjectBundle", "ObjectViewSelection",
    "SceneSubtypeComposition", "InterfaceComponentReference",
    "InterfaceAfterRender", "InterfaceAssembly", "RenderObjectLibrary",
    "render_style_spec",
    "render_style_key", "render_subtype_kind", "render_subtype_key",
    "canonical_subtype_directory",
]