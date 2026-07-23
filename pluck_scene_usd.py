"""OpenUSD package boundary for Pluck's live initial-map state.

USD owns portable scene composition.  ``RoomWorkspace`` remains the pure live
state container, and Pluck remains the coordinator which instantiates runtime
physics, stations, render engines, and interaction handlers from that state.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

from camera_software.usd_material_bridge import (
    author_material_library,
    material_binding_metadata,
)
from placed_object import PlacedObject, placed_object_from_dict


PLUCK_USD_SCHEMA_VERSION = 1
DEFAULT_INITIAL_STAGE = os.path.join("configs", "pluck", "initial_map.usda")


def _identifier(value: str) -> str:
    result = "_".join(re.findall(r"[A-Za-z0-9]+", str(value))) or "Object"
    return f"_{result}" if result[0].isdigit() else result


def _pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    except ImportError as exc:  # pragma: no cover - exercised by packaged fallback
        raise RuntimeError(
            "OpenUSD runtime unavailable; install the bundled usd-core dependency"
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdShade


def _material_keys(objects: Iterable[PlacedObject]) -> tuple[str, ...]:
    result: list[str] = []
    for obj in objects:
        for key in getattr(obj, "material_bindings", {}).values():
            name = str(key).strip()
            if name and name not in result:
                result.append(name)
    return tuple(result)


def write_workspace_usd_package(
    workspace,
    stage_path: str = DEFAULT_INITIAL_STAGE,
    *,
    material_directory: str = "configs/materials",
    require_material_bindings: bool = True,
) -> str:
    """Publish a referenced USD package for a ``RoomWorkspace``.

    Each placed object is a component layer.  The root stage carries only
    placement opinions and material bindings, so components can be reused or
    overridden without rebuilding the general map.
    """

    Gf, Sdf, Usd, UsdGeom, UsdShade = _pxr()
    if require_material_bindings:
        workspace.require_material_bindings()
    final_path = os.path.abspath(stage_path)
    root_dir = os.path.dirname(final_path)
    object_dir = os.path.join(root_dir, "objects")
    os.makedirs(object_dir, exist_ok=True)

    references: list[tuple[PlacedObject, str]] = []
    for obj in workspace.objects.values():
        object_path = os.path.join(object_dir, f"{_identifier(obj.obj_id)}.usda")
        layer = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(layer, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(layer, 1.0)
        root = UsdGeom.Xform.Define(layer, "/PlacedObject")
        layer.SetDefaultPrim(root.GetPrim())
        prim = root.GetPrim()
        payload = obj.to_dict()
        payload.pop("pos", None)
        payload.pop("yaw_deg", None)
        prim.CreateAttribute("pluck:schemaVersion", Sdf.ValueTypeNames.Int).Set(
            PLUCK_USD_SCHEMA_VERSION
        )
        prim.CreateAttribute("pluck:objectType", Sdf.ValueTypeNames.Token).Set(
            str(payload.get("type", "object"))
        )
        prim.CreateAttribute("pluck:objectId", Sdf.ValueTypeNames.String).Set(obj.obj_id)
        prim.CreateAttribute("pluck:label", Sdf.ValueTypeNames.String).Set(obj.label)
        prim.CreateAttribute("pluck:componentData", Sdf.ValueTypeNames.String).Set(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        bindings = getattr(obj, "material_bindings", {})
        prim.CreateAttribute("pluck:materialBindings", Sdf.ValueTypeNames.String).Set(
            material_binding_metadata(bindings)
        )
        runtime_handler = getattr(obj, "station_type", "")
        if runtime_handler:
            prim.CreateAttribute("pluck:runtimeHandler", Sdf.ValueTypeNames.Token).Set(
                str(runtime_handler)
            )
        configuration_root = getattr(obj, "config_dir", "")
        if configuration_root:
            prim.CreateAttribute("pluck:configurationRoot", Sdf.ValueTypeNames.String).Set(
                str(configuration_root).replace("\\", "/")
            )
        if not layer.GetRootLayer().Export(object_path):
            raise OSError(f"failed to publish USD component {object_path}")
        references.append((obj, object_path))

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/PluckWorld")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().CreateAttribute("pluck:schemaVersion", Sdf.ValueTypeNames.Int).Set(
        PLUCK_USD_SCHEMA_VERSION
    )
    world.GetPrim().CreateAttribute("pluck:sceneRole", Sdf.ValueTypeNames.Token).Set(
        "general-initial-map"
    )
    world.GetPrim().CreateAttribute("pluck:movementPolicy", Sdf.ValueTypeNames.Token).Set(
        str(getattr(workspace, "movement_policy", "room-authority-bounded"))
    )
    world.GetPrim().CreateAttribute("pluck:movementEnforced", Sdf.ValueTypeNames.Bool).Set(
        bool(getattr(workspace, "movement_enforced", False))
    )
    world.GetPrim().CreateAttribute("pluck:roomMaterialBindings", Sdf.ValueTypeNames.String).Set(
        material_binding_metadata(
            dict(getattr(workspace, "room_cfg", {}).get("material_bindings", {}))
        )
    )
    objects_scope = UsdGeom.Scope.Define(stage, "/PluckWorld/Objects")
    material_keys = list(_material_keys(workspace.objects.values()))
    for key in dict(workspace.room_cfg.get("material_bindings", {})).values():
        name = str(key).strip()
        if name and name not in material_keys:
            material_keys.append(name)
    material_map = author_material_library(
        stage,
        material_keys,
        material_directory=material_directory,
        asset_anchor_path=final_path,
    )

    room_authority_path = None
    published_references: list[tuple[str, str]] = []
    for obj, object_path in references:
        prim_path = f"{objects_scope.GetPath()}/{_identifier(obj.obj_id)}"
        xform = UsdGeom.Xform.Define(stage, prim_path)
        prim = xform.GetPrim()
        relative = os.path.relpath(object_path, root_dir).replace("\\", "/")
        # Anonymous authoring stages have no filesystem anchor, so compose
        # against the absolute component during construction. Immediately
        # after export we rewrite the published layer to its portable relative
        # reference.
        prim.GetReferences().AddReference(object_path, "/PlacedObject")
        published_references.append((prim_path, relative))
        xform.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in obj.pos)))
        xform.AddRotateZOp().Set(float(obj.yaw_deg))
        prim.CreateAttribute("pluck:worldYawDegrees", Sdf.ValueTypeNames.Double).Set(
            float(obj.yaw_deg)
        )
        bindings = getattr(obj, "material_bindings", {})
        prim.CreateAttribute("pluck:materialBindings", Sdf.ValueTypeNames.String).Set(
            material_binding_metadata(bindings)
        )
        primary_key = next(iter(bindings.values()), "")
        if primary_key in material_map:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material_map[primary_key])
        declared_authority = getattr(workspace, "room_authority_id", None)
        is_authority = (
            obj.obj_id == declared_authority
            if declared_authority else
            getattr(obj, "station_type", "") in {"room", "room_control"}
        )
        if is_authority:
            if room_authority_path is not None:
                raise ValueError("general map may declare only one Room Station authority")
            room_authority_path = prim.GetPath()

    if room_authority_path is None:
        raise ValueError("general map requires a self-describing Room Station authority")
    world.GetPrim().CreateRelationship("pluck:roomAuthority").SetTargets(
        [room_authority_path]
    )

    if not stage.GetRootLayer().Export(final_path):
        raise OSError(f"failed to publish Pluck USD stage {final_path}")
    published = Usd.Stage.Open(final_path)
    if published is None:
        raise OSError(f"failed to reopen published Pluck USD stage {final_path}")
    for prim_path, relative in published_references:
        prim = published.GetPrimAtPath(prim_path)
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(relative, "/PlacedObject")
    published.GetRootLayer().Save()
    return final_path


def workspace_from_usd(
    stage_path: str = DEFAULT_INITIAL_STAGE,
    *,
    config_dir: str = "configs/room_station",
):
    """Compose a USD stage and restore its objects into live workspace state."""

    _Gf, _Sdf, Usd, UsdGeom, _UsdShade = _pxr()
    from room_workspace import RoomWorkspace

    final_path = os.path.abspath(stage_path)
    stage = Usd.Stage.Open(final_path)
    if stage is None:
        raise ValueError(f"cannot open Pluck USD stage {final_path}")
    world = stage.GetPrimAtPath("/PluckWorld")
    if not world:
        raise ValueError("Pluck USD stage has no /PluckWorld prim")
    version = world.GetAttribute("pluck:schemaVersion").Get()
    if int(version or 0) != PLUCK_USD_SCHEMA_VERSION:
        raise ValueError(f"unsupported Pluck USD schema version {version!r}")
    authority_targets = world.GetRelationship("pluck:roomAuthority").GetTargets()
    if len(authority_targets) != 1:
        raise ValueError("Pluck USD stage must identify exactly one Room Station authority")

    scope = stage.GetPrimAtPath("/PluckWorld/Objects")
    if not scope:
        raise ValueError("Pluck USD stage has no /PluckWorld/Objects scope")
    cache = UsdGeom.XformCache()
    workspace = RoomWorkspace.blank(config_dir)
    for prim in scope.GetChildren():
        object_type = prim.GetAttribute("pluck:objectType").Get()
        data_text = prim.GetAttribute("pluck:componentData").Get()
        if not object_type or not data_text:
            continue
        data = json.loads(str(data_text))
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        data["id"] = str(prim.GetAttribute("pluck:objectId").Get() or data.get("id", ""))
        data["label"] = str(prim.GetAttribute("pluck:label").Get() or data.get("label", ""))
        data["type"] = str(object_type)
        data["pos"] = [float(translation[0]), float(translation[1]), float(translation[2])]
        data["yaw_deg"] = float(
            prim.GetAttribute("pluck:worldYawDegrees").Get() or 0.0
        )
        bindings_text = prim.GetAttribute("pluck:materialBindings").Get()
        if bindings_text:
            data["material_bindings"] = json.loads(str(bindings_text))
        obj = placed_object_from_dict(data)
        workspace.add_object(obj)
    workspace._dirty = False
    workspace.usd_stage_path = final_path
    authority_prim = stage.GetPrimAtPath(authority_targets[0])
    workspace.declare_room_authority(
        str(authority_prim.GetAttribute("pluck:objectId").Get()),
        movement_policy=str(
            world.GetAttribute("pluck:movementPolicy").Get() or ""
        ),
        movement_enforced=bool(
            world.GetAttribute("pluck:movementEnforced").Get() or False
        ),
    )
    room_bindings = world.GetAttribute("pluck:roomMaterialBindings").Get()
    if room_bindings:
        workspace.room_cfg["material_bindings"] = json.loads(str(room_bindings))
    return workspace


__all__ = [
    "DEFAULT_INITIAL_STAGE",
    "PLUCK_USD_SCHEMA_VERSION",
    "workspace_from_usd",
    "write_workspace_usd_package",
]
