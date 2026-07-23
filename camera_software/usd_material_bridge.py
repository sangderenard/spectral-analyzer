"""Project canonical materials into USD without transferring authority to USD.

Pluck's material definitions remain the scientific source of truth.  This
module publishes interoperable ``UsdShade`` approximations plus stable links
back to those definitions.  Renderer records, spectral tensors, structural
colour patches, and other engine-specific products are still compiled by the
project material system.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import yaml


def _identifier(value: str) -> str:
    result = "_".join(re.findall(r"[A-Za-z0-9]+", str(value))) or "Material"
    return f"_{result}" if result[0].isdigit() else result


@dataclass(frozen=True, slots=True)
class CanonicalMaterialProjection:
    """A portable USD view of one authoritative project material."""

    key: str
    definition_path: str
    revision: str
    preview_albedo: tuple[float, float, float]
    preview_roughness: float
    preview_metallic: float
    preview_opacity: float
    engine_requirements: tuple[str, ...] = ()


def read_canonical_material_projection(
    key: str, *, material_directory: str = "configs/materials",
) -> CanonicalMaterialProjection:
    """Read authoring metadata without compiling or replacing the material."""

    material_key = str(key).strip()
    if not material_key:
        raise ValueError("canonical material key must be non-empty")
    path = os.path.abspath(os.path.join(material_directory, f"{material_key}.yaml"))
    with open(path, "rb") as handle:
        raw = handle.read()
    data = yaml.safe_load(raw.decode("utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"material {material_key!r} is not a mapping")

    pbr = data.get("pbr", {})
    pbr = pbr if isinstance(pbr, Mapping) else {}
    albedo = pbr.get("albedo", data.get("albedo_rgb", [0.5, 0.5, 0.5]))
    if not isinstance(albedo, (list, tuple)) or len(albedo) < 3:
        albedo = [0.5, 0.5, 0.5]
    requirements: list[str] = []
    if data.get("maxwell_patch") is not None:
        requirements.append("maxwell_patch")
    for item in data.get("engine_requirements", ()) or ():
        name = str(item).strip()
        if name and name not in requirements:
            requirements.append(name)

    return CanonicalMaterialProjection(
        key=material_key,
        definition_path=path,
        revision=hashlib.sha256(raw).hexdigest(),
        preview_albedo=tuple(float(value) for value in albedo[:3]),
        preview_roughness=float(pbr.get("roughness", 0.5)),
        preview_metallic=float(pbr.get("metallic", 0.0)),
        preview_opacity=float(data.get("opacity", 1.0)),
        engine_requirements=tuple(requirements),
    )


def author_material_library(
    stage: Any,
    material_keys: Iterable[str],
    *,
    material_directory: str = "configs/materials",
    root_path: str = "/PluckMaterials",
    asset_anchor_path: str = "",
) -> dict[str, Any]:
    """Author derived USD materials and return ``key -> UsdShade.Material``.

    Custom attributes point back to canonical definitions.  The preview shader
    is deliberately labelled as a projection; it is not a scientific export.
    """

    from pxr import Sdf, UsdGeom, UsdShade

    UsdGeom.Scope.Define(stage, root_path)
    anchor = str(asset_anchor_path or stage.GetRootLayer().realPath)
    stage_directory = os.path.dirname(os.path.abspath(anchor or "scene.usda"))
    materials_path = f"{root_path}/Materials"
    UsdGeom.Scope.Define(stage, materials_path)
    result: dict[str, Any] = {}
    for key in dict.fromkeys(str(item) for item in material_keys if str(item).strip()):
        projection = read_canonical_material_projection(
            key, material_directory=material_directory,
        )
        path = f"{materials_path}/{_identifier(key)}"
        material = UsdShade.Material.Define(stage, path)
        prim = material.GetPrim()
        prim.CreateAttribute("pluck:materialKey", Sdf.ValueTypeNames.String).Set(key)
        prim.CreateAttribute("pluck:materialRevision", Sdf.ValueTypeNames.String).Set(
            projection.revision
        )
        prim.CreateAttribute("pluck:definitionAsset", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(
                os.path.relpath(projection.definition_path, stage_directory).replace("\\", "/")
            )
        )
        prim.CreateAttribute("pluck:authority", Sdf.ValueTypeNames.Token).Set(
            "canonical-project-material"
        )
        prim.CreateAttribute("pluck:projectionKind", Sdf.ValueTypeNames.Token).Set(
            "portable-preview"
        )
        prim.CreateAttribute(
            "pluck:engineRequirements", Sdf.ValueTypeNames.TokenArray
        ).Set(list(projection.engine_requirements))

        shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            projection.preview_albedo
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            projection.preview_roughness
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            projection.preview_metallic
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
            projection.preview_opacity
        )
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        result[key] = material
    return result


def material_binding_metadata(bindings: Mapping[str, str]) -> str:
    """Stable role mapping retained when one prim represents several pieces."""

    return json.dumps(
        {str(role): str(key) for role, key in sorted(bindings.items())},
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "CanonicalMaterialProjection",
    "author_material_library",
    "material_binding_metadata",
    "read_canonical_material_projection",
]
