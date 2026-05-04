"""Station panel specification primitives.

This module defines a repository-native station spec format that can be driven
from KnobSpec descriptors and rendered through a hierarchical control table.

Design notes
------------
- Z order is implicit in declaration order and hierarchy traversal.
- Material slots support advanced optical/spectral metadata while still
  exposing simple defaults for legacy panel rendering.
- Mesh output is intentionally paper-thin layers so callers can either:
  1. draw as 2D HUD quads, or
  2. pre-bake in an OpenGL lit scene, and later swap to ray-traced outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class StationMaterialSlot:
    name: str
    base_color_rgb: tuple[float, float, float] = (0.2, 0.24, 0.3)
    metallic: float = 0.0
    roughness: float = 0.85
    opacity: float = 1.0
    emission_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emission_strength: float = 0.0
    remission_profile: str | None = None
    emission_profile: str | None = None
    advanced: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StationControlNode:
    key: str
    label: str
    knob: Any | None = None
    children: list["StationControlNode"] = field(default_factory=list)
    depth_mode: str = "raise"
    thickness_m: float = 0.0008
    material_slot: str = "panel_default"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StationHierarchyRecord:
    path: tuple[str, ...]
    z_index: int
    node: StationControlNode


@dataclass(slots=True)
class StationHierarchyTable:
    records: list[StationHierarchyRecord] = field(default_factory=list)

    def sorted(self) -> list[StationHierarchyRecord]:
        return sorted(self.records, key=lambda rec: rec.z_index)


@dataclass(slots=True)
class LayerMesh:
    key: str
    z: float
    material_slot: str
    vertices_xyz: list[tuple[float, float, float]] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StationPrebakePlan:
    mode: str = "hud2d"
    mesh_layers: list[LayerMesh] = field(default_factory=list)
    material_slots: dict[str, StationMaterialSlot] = field(default_factory=dict)
    texture_slots: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


def default_station_material_slots() -> dict[str, StationMaterialSlot]:
    def _safe_yaml(path: Path) -> dict[str, Any]:
        try:
            import yaml
        except Exception:
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    root = Path(__file__).resolve().parent.parent
    rc_cfg = _safe_yaml(root / "configs" / "duty_stations" / "room_control" / "station.yaml")
    ds_cfg = _safe_yaml(root / "configs" / "meshes" / "duty_station.yaml")

    piece = rc_cfg.get("piece_materials") if isinstance(rc_cfg.get("piece_materials"), dict) else {}
    mesh_mats = ds_cfg.get("materials") if isinstance(ds_cfg.get("materials"), dict) else {}

    def _pick_rgb(d: dict[str, Any], *keys: str, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        for key in keys:
            raw = d.get(key)
            if isinstance(raw, list) and len(raw) >= 3:
                return (float(raw[0]), float(raw[1]), float(raw[2]))
        return fallback

    left_src = piece.get("left_wing") if isinstance(piece.get("left_wing"), dict) else {}
    right_src = piece.get("right_wing") if isinstance(piece.get("right_wing"), dict) else left_src
    center_src = piece.get("monitor") if isinstance(piece.get("monitor"), dict) else {}
    shell_src = piece.get("main_surface") if isinstance(piece.get("main_surface"), dict) else {}

    body_src = mesh_mats.get("body") if isinstance(mesh_mats.get("body"), dict) else {}
    trim_src = mesh_mats.get("trim") if isinstance(mesh_mats.get("trim"), dict) else {}
    screen_src = mesh_mats.get("screen_active") if isinstance(mesh_mats.get("screen_active"), dict) else {}

    left_rgb = _pick_rgb(left_src, "albedo", fallback=_pick_rgb(body_src, "albedo_rgb", fallback=(0.18, 0.21, 0.27)))
    right_rgb = _pick_rgb(right_src, "albedo", fallback=left_rgb)
    shell_rgb = _pick_rgb(shell_src, "albedo", fallback=_pick_rgb(body_src, "albedo_rgb", fallback=(0.18, 0.21, 0.27)))
    center_rgb = _pick_rgb(center_src, "albedo", fallback=_pick_rgb(screen_src, "albedo_rgb", fallback=(0.01, 0.02, 0.04)))
    center_em = _pick_rgb(center_src, "emission", fallback=_pick_rgb(screen_src, "emissive", fallback=(0.06, 0.32, 0.88)))
    trim_em = _pick_rgb(trim_src, "emissive", fallback=(0.00, 0.36, 0.92))

    center_tex = None
    if isinstance(center_src.get("albedo_texture"), str):
        center_tex = center_src.get("albedo_texture")

    return {
        "panel_default": StationMaterialSlot(
            name="panel_default",
            base_color_rgb=shell_rgb,
            roughness=0.88,
            advanced={
                "ui_role": "panel",
                "notes": "Default duty-station shell from room/mesh material config",
            },
        ),
        "left_physical": StationMaterialSlot(
            name="left_physical",
            base_color_rgb=left_rgb,
            roughness=float(left_src.get("roughness", 0.60)) if isinstance(left_src, dict) else 0.60,
            metallic=float(left_src.get("metallic", 0.35)) if isinstance(left_src, dict) else 0.35,
            advanced={"ui_role": "left_controls", "source": "room_control.piece_materials.left_wing"},
        ),
        "right_physical": StationMaterialSlot(
            name="right_physical",
            base_color_rgb=right_rgb,
            roughness=float(right_src.get("roughness", 0.60)) if isinstance(right_src, dict) else 0.60,
            metallic=float(right_src.get("metallic", 0.35)) if isinstance(right_src, dict) else 0.35,
            advanced={"ui_role": "right_controls", "source": "room_control.piece_materials.right_wing"},
        ),
        "center_screen": StationMaterialSlot(
            name="center_screen",
            base_color_rgb=center_rgb,
            roughness=float(center_src.get("roughness", 0.08)) if isinstance(center_src, dict) else 0.08,
            metallic=float(center_src.get("metallic", 0.0)) if isinstance(center_src, dict) else 0.0,
            emission_rgb=center_em,
            emission_strength=1.4,
            emission_profile="room_control.monitor",
            remission_profile="room_control.monitor_reflect",
            advanced={
                "ui_role": "center_screen",
                "screen_texture": center_tex or "generated:center_screen",
                "source": "room_control.piece_materials.monitor",
            },
        ),
        "control_raised": StationMaterialSlot(
            name="control_raised",
            base_color_rgb=_pick_rgb(trim_src, "albedo_rgb", fallback=(0.30, 0.36, 0.48)),
            roughness=0.60,
            emission_rgb=trim_em,
            emission_strength=0.08,
            advanced={
                "ui_role": "control",
                "surface_kind": "raised",
            },
        ),
        "control_engraved": StationMaterialSlot(
            name="control_engraved",
            base_color_rgb=(0.12, 0.14, 0.18),
            roughness=0.92,
            advanced={
                "ui_role": "control",
                "surface_kind": "engraved",
            },
        ),
        "indicator_emissive": StationMaterialSlot(
            name="indicator_emissive",
            base_color_rgb=(0.10, 0.12, 0.14),
            emission_rgb=(0.2, 0.8, 1.0),
            emission_strength=2.0,
            emission_profile="ui_indicator_default",
            remission_profile="ui_matte_neutral",
            advanced={
                "ui_role": "indicator",
                "supports_sensor": True,
            },
        ),
    }


def build_station_hierarchy(
    root_key: str,
    nodes: Iterable[StationControlNode],
) -> StationHierarchyTable:
    records: list[StationHierarchyRecord] = []
    z_index = 0

    def _walk(parent_path: tuple[str, ...], node: StationControlNode) -> None:
        nonlocal z_index
        path = parent_path + (node.key,)
        records.append(StationHierarchyRecord(path=path, z_index=z_index, node=node))
        z_index += 1
        for child in node.children:
            _walk(path, child)

    base = (root_key,)
    for node in nodes:
        _walk(base, node)

    return StationHierarchyTable(records=records)


def build_paper_layer_meshes(
    hierarchy: StationHierarchyTable,
    *,
    panel_origin_xy: tuple[float, float] = (0.0, 0.0),
    panel_size_xy: tuple[float, float] = (1.0, 1.0),
    layer_step_m: float = 0.0012,
) -> list[LayerMesh]:
    x0, y0 = panel_origin_xy
    w, h = panel_size_xy
    meshes: list[LayerMesh] = []

    for rec in hierarchy.sorted():
        node = rec.node
        local_step = float(node.payload.get("z_step_m", layer_step_m)) if isinstance(node.payload, dict) else layer_step_m
        z = rec.z_index * local_step
        if node.depth_mode == "engrave":
            z = -z

        v0 = (x0, y0, z)
        v1 = (x0 + w, y0, z)
        v2 = (x0 + w, y0 + h, z)
        v3 = (x0, y0 + h, z)
        mesh = LayerMesh(
            key="/".join(rec.path),
            z=z,
            material_slot=node.material_slot,
            vertices_xyz=[v0, v1, v2, v3],
            uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            triangles=[(0, 1, 2), (0, 2, 3)],
            payload={
                "depth_mode": node.depth_mode,
                "thickness_m": node.thickness_m,
                "label": node.label,
            },
        )
        meshes.append(mesh)

    return meshes


def build_label_ridge_mesh(
    text: str,
    *,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    z: float = 0.0,
    stroke_h: float = 0.03,
    char_w: float = 0.02,
    spacing: float = 0.008,
    ridge_radius: float = 0.001,
    cap_segments: int = 6,
    material_slot: str = "control_raised",
) -> LayerMesh:
    """Create a raised label mesh using capsule strokes.

    The geometry is intentionally simple: each character becomes one raised
    horizontal ridge with half-circle end caps, similar to label-maker tape.
    """
    ox, oy = origin_xy
    vertices: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []

    def _add_capsule(x0: float, y0: float, x1: float, y1: float) -> None:
        import math

        dx = x1 - x0
        dy = y1 - y0
        ln = (dx * dx + dy * dy) ** 0.5
        if ln <= 1e-12:
            return
        tx = dx / ln
        ty = dy / ln
        nx = -ty
        ny = tx

        pts: list[tuple[float, float]] = []
        a0 = math.atan2(ty, tx)
        for i in range(cap_segments + 1):
            a = a0 + math.pi * 0.5 - math.pi * (i / cap_segments)
            pts.append((x0 + math.cos(a) * ridge_radius, y0 + math.sin(a) * ridge_radius))
        for i in range(cap_segments + 1):
            a = a0 - math.pi * 0.5 + math.pi * (i / cap_segments)
            pts.append((x1 + math.cos(a) * ridge_radius, y1 + math.sin(a) * ridge_radius))

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cidx = len(vertices)
        vertices.append((cx, cy, z))
        base = len(vertices)
        for px, py in pts:
            vertices.append((px, py, z))
        n = len(pts)
        for i in range(n):
            i0 = base + i
            i1 = base + ((i + 1) % n)
            tris.append((cidx, i0, i1))

    cursor_x = ox
    for ch in text.upper():
        if ch == " ":
            cursor_x += char_w + spacing
            continue
        x0 = cursor_x
        x1 = cursor_x + char_w
        y = oy + stroke_h * 0.5
        _add_capsule(x0, y, x1, y)
        cursor_x += char_w + spacing

    return LayerMesh(
        key=f"label:{text}",
        z=z,
        material_slot=material_slot,
        vertices_xyz=vertices,
        uvs=[(0.0, 0.0)] * len(vertices),
        triangles=tris,
        payload={"label_text": text, "ridge": True},
    )


def make_prebake_plan(
    hierarchy: StationHierarchyTable,
    *,
    mode: str = "hud2d",
    material_slots: dict[str, StationMaterialSlot] | None = None,
) -> StationPrebakePlan:
    mats = dict(material_slots or default_station_material_slots())
    meshes = build_paper_layer_meshes(hierarchy)
    label_meshes: list[LayerMesh] = []
    for rec in hierarchy.sorted():
        if rec.node.label:
            label_meshes.append(
                build_label_ridge_mesh(
                    rec.node.label,
                    origin_xy=(0.02, 0.02 + rec.z_index * 0.004),
                    z=rec.z_index * 0.0012 + 0.0004,
                    material_slot=rec.node.material_slot,
                )
            )
    tex_center = mats.get("center_screen").advanced.get("screen_texture") if mats.get("center_screen") else "generated:center_screen"
    return StationPrebakePlan(
        mode=mode,
        mesh_layers=meshes + label_meshes,
        material_slots=mats,
        texture_slots={"center_screen": tex_center},
        payload={
            "z_order": "implicit_by_hierarchy",
            "lighting_enabled": mode in {"opengl_lit", "raytrace_preview"},
            "z_height_tuneable": True,
        },
    )
