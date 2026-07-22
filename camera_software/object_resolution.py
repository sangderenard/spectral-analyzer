"""One hierarchy for layout objects, monofont text, scenes, and evidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .control_layout import ProgramUILayout
from .layout_object_work import (
    LayoutObjectWorkManifest,
    LayoutWorkProgressCache,
)
from .render_object_library import (
    RenderObjectBundle,
    RenderObjectLibrary,
    RenderObjectSubtype,
)
from .convergence_metrics import (
    artifact_convergence_metric,
    artifact_convergence_velocity,
)


@dataclass(frozen=True)
class ResolvedTextPart:
    text: str
    text_offset: int
    object_key: str
    subtype_key: str
    subtype_kind: str
    scene_path: str
    artifact_path: str
    convergence: float
    convergence_velocity_per_pass: float
    subtype: RenderObjectSubtype | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class ResolvedTextImage:
    owner_id: str
    text: str
    resolution: str
    target_rect_px: tuple[int, int, int, int]
    layout_rect_px: tuple[int, int, int, int]
    parts: tuple[ResolvedTextPart, ...]
    missing_characters: tuple[str, ...] = ()

    @property
    def convergence(self) -> float:
        return (
            sum(part.convergence for part in self.parts) / len(self.parts)
            if self.parts else 0.0
        )

    @property
    def convergence_velocity_per_pass(self) -> float:
        return (
            sum(
                part.convergence_velocity_per_pass for part in self.parts
            ) / len(self.parts)
            if self.parts else 0.0
        )


@dataclass(frozen=True)
class ResolvedObjectNode:
    key: str
    kind: str
    label: str
    status: str
    children: tuple["ResolvedObjectNode", ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def mapping(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "metadata": dict(self.metadata),
            "children": [child.mapping() for child in self.children],
        }


@dataclass(frozen=True)
class AssemblyProviderReadiness:
    provider_key: str
    tier: str
    ready: bool
    readiness_basis: str
    supplied_object_keys: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    fallback_permitted: bool = False


@dataclass(frozen=True)
class SceneAssemblyReadiness:
    ready: bool
    providers: tuple[AssemblyProviderReadiness, ...]
    tier_order: tuple[str, ...] = (
        "alphabet", "token", "panel", "scene", "holistic_static_texture"
    )

    def mapping(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "tier_order": list(self.tier_order),
            "providers": [
                {
                    "provider_key": item.provider_key,
                    "tier": item.tier,
                    "ready": item.ready,
                    "readiness_basis": item.readiness_basis,
                    "supplied_object_keys": list(item.supplied_object_keys),
                    "missing": list(item.missing),
                    "fallback_permitted": item.fallback_permitted,
                }
                for item in self.providers
            ],
        }


def _artifact_evidence(
    subtype: RenderObjectSubtype,
) -> tuple[str, float, float]:
    artifact = max(
        tuple(subtype.artifacts),
        key=lambda item: (
            int(item.samples),
            float(item.created_at_s),
        ),
        default=None,
    )
    if artifact is None:
        return "", 0.0, 0.0
    path = (
        artifact.preview_path
        or artifact.sprite_path
        or artifact.linear_path
    )
    return (
        path,
        artifact_convergence_metric(artifact),
        artifact_convergence_velocity(artifact),
    )


class HierarchicalObjectResolver:
    """Resolve every UI object through the same subtype/scene/evidence ladder."""

    def __init__(
        self,
        layout: ProgramUILayout,
        render_library: RenderObjectLibrary,
        layout_manifest: LayoutObjectWorkManifest,
        layout_cache: LayoutWorkProgressCache,
    ) -> None:
        self.layout = layout
        self.render_library = render_library
        self.layout_manifest = layout_manifest
        self.layout_cache = layout_cache

    def _text_bundle(self) -> RenderObjectBundle | None:
        requested = self.layout.render_pipeline
        bundles = tuple(self.render_library.snapshot())
        text_capable = [
            bundle for bundle in bundles
            if bundle.subtype_sets.get("glyph")
            or bundle.subtype_sets.get("token")
        ]
        matching = [
            bundle for bundle in text_capable
            if str(dict(bundle.style_spec.get("font", {})).get("family", ""))
            == requested.font_family
        ]
        candidates = matching or text_capable
        return max(
            candidates,
            key=lambda item: (
                len(item.subtype_sets.get("token", ())),
                len(item.subtype_sets.get("glyph", ())),
                item.updated_at_s,
            ),
            default=None,
        )

    def resolve_text_image(
        self, owner_id: str, text: str
    ) -> ResolvedTextImage:
        owner_id, text = str(owner_id), str(text)
        target_rect = tuple(self.layout.regions[owner_id])
        layout_rect = tuple(
            self.layout.regions[self.layout.manifest.name]
        )
        bundle = self._text_bundle()
        if bundle is None:
            return ResolvedTextImage(
                owner_id, text, "pending", target_rect, layout_rect, (),
                tuple(dict.fromkeys(character for character in text if not character.isspace())),
            )
        exact = next(
            (
                subtype for subtype in bundle.subtypes
                if subtype.kind == "token" and subtype.text == text
            ),
            None,
        )
        selected: list[tuple[int, RenderObjectSubtype]] = []
        resolution = "exact_token"
        missing: list[str] = []
        if exact is not None:
            selected.append((0, exact))
        else:
            resolution = "glyph_fallback"
            glyphs = {
                subtype.text: subtype for subtype in bundle.subtypes
                if subtype.kind == "glyph"
            }
            for offset, character in enumerate(text):
                if character.isspace():
                    continue
                subtype = glyphs.get(character)
                if subtype is None:
                    missing.append(character)
                else:
                    selected.append((offset, subtype))
            if not selected:
                resolution = "pending"
        parts = []
        for offset, subtype in selected:
            (
                artifact_path,
                convergence,
                convergence_velocity,
            ) = _artifact_evidence(subtype)
            scene_path = subtype.scenes[0].scene_path if subtype.scenes else ""
            parts.append(ResolvedTextPart(
                text=subtype.text,
                text_offset=offset,
                object_key=bundle.object_key,
                subtype_key=subtype.subtype_key,
                subtype_kind=subtype.kind,
                scene_path=scene_path,
                artifact_path=artifact_path,
                convergence=convergence,
                convergence_velocity_per_pass=convergence_velocity,
                subtype=subtype,
            ))
        return ResolvedTextImage(
            owner_id=owner_id,
            text=text,
            resolution=resolution,
            target_rect_px=target_rect,
            layout_rect_px=layout_rect,
            parts=tuple(parts),
            missing_characters=tuple(dict.fromkeys(missing)),
        )

    def text_images(self) -> tuple[ResolvedTextImage, ...]:
        return tuple(
            self.resolve_text_image(owner_id, text)
            for owner_id, text in self.layout.authored_text.items()
            if owner_id in self.layout.regions
        )

    def assembly_readiness(self) -> SceneAssemblyReadiness:
        """Provider sign-offs consumed by the retained USD assembly layer."""

        text_images = self.text_images()
        missing_text = tuple(sorted({
            character
            for image in text_images
            for character in image.missing_characters
        }))
        text_parts = tuple(part for image in text_images for part in image.parts)
        font_ready = not missing_text and all(
            part.scene_path for part in text_parts
        ) and all(
            not image.text.strip() or image.parts for image in text_images
        )
        find_bundle = getattr(self.render_library, "find", None)
        panel_missing = []
        panel_keys = []
        for design_object in self.layout_manifest.objects:
            panel_keys.append(design_object.object_key)
            bundle = (
                find_bundle(design_object.object_key)
                if callable(find_bundle) else None
            )
            for subtype in design_object.subtypes:
                registered = next((
                    item for item in (() if bundle is None else bundle.subtypes)
                    if item.source_asset_key == subtype.subtype_key
                    and item.scenes
                ), None)
                if registered is None:
                    panel_missing.append(
                        f"{design_object.object_key}/{subtype.subtype_key}"
                    )
        panel_ready = not panel_missing
        providers = (
            AssemblyProviderReadiness(
                provider_key="monofont-render-library",
                tier="alphabet_to_token",
                ready=font_ready,
                readiness_basis="all_authored_text_has_retained_scene_supply",
                supplied_object_keys=tuple(sorted({
                    part.object_key for part in text_parts
                })),
                missing=missing_text,
            ),
            AssemblyProviderReadiness(
                provider_key="parametric-panel-library",
                tier="panel",
                ready=panel_ready,
                readiness_basis=(
                    "retained_parametric_scene_with_empty_panel_fallback"
                ),
                supplied_object_keys=tuple(panel_keys),
                missing=tuple(panel_missing),
                fallback_permitted=True,
            ),
        )
        return SceneAssemblyReadiness(
            ready=all(provider.ready for provider in providers),
            providers=providers,
        )

    def tree(self) -> ResolvedObjectNode:
        design_nodes = []
        for design_object in self.layout_manifest.objects:
            find_bundle = getattr(self.render_library, "find", None)
            library_bundle = (
                find_bundle(design_object.object_key)
                if callable(find_bundle) else None
            )
            subtype_nodes = []
            for subtype in design_object.subtypes:
                replacement = self.layout_cache.replacement_path(subtype)
                library_subtype = next(
                    (
                        item for item in (
                            () if library_bundle is None
                            else library_bundle.subtypes
                        )
                        if item.source_asset_key == subtype.subtype_key
                    ),
                    None,
                )
                source_scene = (
                    "" if library_subtype is None or not library_subtype.scenes
                    else library_subtype.scenes[0].scene_path
                )
                animations = tuple(
                    ResolvedObjectNode(
                        animation.animation_key,
                        "pose_animation",
                        animation.name,
                        "cached" if replacement else "manifested",
                        metadata={
                            "frames": len(animation.frames),
                            "focus_locations": animation.focus_location_count,
                            "replacement_frame": animation.replacement_frame_index,
                        },
                    )
                    for animation in subtype.pose_animations
                )
                variants = tuple(
                    ResolvedObjectNode(
                        variant.variant_key,
                        "aspect_variant",
                        (
                            f"{variant.aspect_ratio[0]}:{variant.aspect_ratio[1]} "
                            f"@ {variant.work_resolution_px[0]}x"
                            f"{variant.work_resolution_px[1]}"
                        ),
                        "registered" if source_scene else "manifested",
                        metadata={
                            "work_resolution_px": variant.work_resolution_px,
                            "square_pane_rect": variant.square_pane_rect,
                            "sampling_policy": variant.sampling_policy,
                            "physical_pixel_aspect": variant.physical_pixel_aspect,
                            "owner_ids": variant.owner_ids,
                        },
                    )
                    for variant in subtype.aspect_variants
                )
                subtype_nodes.append(ResolvedObjectNode(
                    subtype.subtype_key,
                    "subtype",
                    subtype.display_name,
                    (
                        "resolved" if replacement else
                        "registered" if source_scene else "unregistered"
                    ),
                    animations + variants,
                    {
                        "scene_consumers": len(subtype.consumers),
                        "artifact_path": replacement,
                        "library_subtype_key": (
                            "" if library_subtype is None
                            else library_subtype.subtype_key
                        ),
                        "source_scene": source_scene,
                        "parameters": dict(subtype.parameter_spec),
                    },
                ))
            design_nodes.append(ResolvedObjectNode(
                design_object.object_key,
                "design_object",
                design_object.display_name,
                "resolved" if all(
                    self.layout_cache.replacement_path(subtype)
                    for subtype in design_object.subtypes
                ) else "registered" if library_bundle is not None
                else "unregistered",
                tuple(subtype_nodes),
                {
                    "library_bundle_key": (
                        "" if library_bundle is None
                        else library_bundle.bundle_key
                    ),
                    "object_kind": design_object.object_kind,
                },
            ))
        text_nodes = []
        for image in self.text_images():
            part_nodes = tuple(
                ResolvedObjectNode(
                    part.subtype_key,
                    part.subtype_kind,
                    part.text,
                    "resolved" if part.artifact_path else "developing",
                    metadata={
                        "object_key": part.object_key,
                        "scene_path": part.scene_path,
                        "artifact_path": part.artifact_path,
                        "convergence": part.convergence,
                        "convergence_velocity_per_pass": (
                            part.convergence_velocity_per_pass
                        ),
                    },
                )
                for part in image.parts
            )
            text_nodes.append(ResolvedObjectNode(
                f"text-image:{image.owner_id}",
                "text_image",
                f"{image.owner_id}: {image.text}",
                image.resolution,
                part_nodes,
                {
                    "missing_characters": list(image.missing_characters),
                    "convergence": image.convergence,
                    "convergence_velocity_per_pass": (
                        image.convergence_velocity_per_pass
                    ),
                },
            ))
        return ResolvedObjectNode(
            f"scene:{self.layout.manifest.name}",
            "scene",
            self.layout.manifest.label or self.layout.manifest.name,
            "composing",
            (
                ResolvedObjectNode(
                    "layout-design-objects",
                    "object_collection",
                    "layout design objects",
                    "composing",
                    tuple(design_nodes),
                ),
                ResolvedObjectNode(
                    "monofont-text-images",
                    "object_collection",
                    "monofont text images",
                    "progressive",
                    tuple(text_nodes),
                ),
            ),
        )


__all__ = [
    "ResolvedTextPart",
    "ResolvedTextImage",
    "ResolvedObjectNode",
    "AssemblyProviderReadiness",
    "SceneAssemblyReadiness",
    "HierarchicalObjectResolver",
]
