"""fabricator_workspace.py
==========================
FabricatorWorkspace — state machine for the fabricator station build process.

States
------
IDLE        Rotating base solid, nothing picked.
PICKED      A solid from the palette is "in hand" — follows cursor.
PLACING     Cursor hovering over base solid; nearest face highlighted.
PLACED      Ghost piece snapped to a face; awaiting symmetry + confirm.
REVIEW      Final mesh shown; Fabricate / Store / Clear available.

The workspace holds all data (current mesh, history, inventory) and exposes
the state needed by FabricatorStation (renderer) and the event handler.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from dec_mesh import DECMesh
import platonic_solids
from fabricator_primitives import (
    duty_station_catalog_entries,
    duty_station_primitive_meshes,
)

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _yaml = None
    _HAS_YAML = False


# ─────────────────────────────────────────────────────────────────────────────

class WorkspaceMode(Enum):
    IDLE    = "idle"
    PICKED  = "picked"
    PLACING = "placing"
    PLACED  = "placed"
    REVIEW  = "review"


class FabricationTab(Enum):
    MILLING = "milling"
    DRILLING = "drilling"
    SUBTRACTIVE = "subtractive"
    ADDITIVE = "additive"
    BEVELING = "beveling"
    ASSEMBLING = "assembling"


@dataclass
class SymmetryConfig:
    mode:  str = "none"    # none | bilateral | radial
    axis:  str = "z"       # x | y | z
    count: int = 4         # radial copies


@dataclass
class PlacementState:
    """Active placement — piece snapped to self_face of the base mesh."""
    piece_id:    str
    piece:       DECMesh
    self_face:   int        # face on base mesh being targeted
    piece_face:  int        # face on piece that mates with self_face
    symmetry:    SymmetryConfig = field(default_factory=SymmetryConfig)

    # Preview mesh (base + placed piece + symmetry copies, lazily computed)
    _preview: Optional[DECMesh] = field(default=None, repr=False, compare=False)

    def preview(self, base: DECMesh) -> DECMesh:
        if self._preview is None:
            self._preview = self._build_preview(base)
        return self._preview

    def invalidate(self):
        self._preview = None

    def _build_preview(self, base: DECMesh) -> DECMesh:
        attached = base.attach(self.piece, self.self_face, self.piece_face)
        if self.symmetry.mode == "none":
            return attached
        # Apply symmetry to just the newly attached piece, then merge with base
        new_piece = base.attach(self.piece, self.self_face, self.piece_face)
        piece_copy = DECMesh(
            verts    = new_piece.verts[len(base.verts):],
            edges    = new_piece.edges[len(base.edges):] - len(base.verts),
            faces    = [[v - len(base.verts) for v in f]
                        for f in new_piece.faces[len(base.faces):]],
            tris     = new_piece.tris[len(base.tris):] - len(base.verts),
            tri_face = new_piece.tri_face[len(base.tris):]
                       - len(base.faces),
        )
        sym = piece_copy.with_symmetry(self.symmetry.mode,
                                        self.symmetry.axis,
                                        self.symmetry.count)
        from dec_mesh import _merge_many
        return _merge_many([base, sym])


@dataclass
class FabricationOperation:
    op_type: str
    tab: str
    params: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────

class FabricatorWorkspace:
    """Fabricator state machine.  Pure logic, no GL."""

    def __init__(self, cfg: dict):
        self._cfg        = cfg
        self.mode        = WorkspaceMode.IDLE
        self.base_solid  = platonic_solids.get(cfg.get("base_solid", "icosahedron"))
        self.built_mesh: Optional[DECMesh] = None   # committed build
        self.placement:  Optional[PlacementState] = None
        self.review_mesh:Optional[DECMesh] = None
        self.process_tab = str(cfg.get("process_tab", FabricationTab.ASSEMBLING.value))
        self.snap_enabled = bool(cfg.get("snap", {}).get("enabled", True))
        self.snap_angle_deg = float(cfg.get("snap", {}).get("angle_deg", 15.0))
        self.snap_distance_m = float(cfg.get("snap", {}).get("distance_m", 0.08))
        self.gimbal = np.array(cfg.get("gimbal_deg", [0.0, 0.0]), np.float64)
        self.operations: list[FabricationOperation] = []
        self._primitive_meshes: dict[str, DECMesh] = duty_station_primitive_meshes(
            cfg.get("duty_station_config", "configs/meshes/duty_station.yaml")
        )
        self._primitive_catalog: list[dict] = duty_station_catalog_entries(
            cfg.get("duty_station_config", "configs/meshes/duty_station.yaml")
        )

        # Palette selection
        self.picked_id:      Optional[str] = None
        self.picked_piece:   Optional[DECMesh] = None
        self.picked_special: bool = False   # True when picked item is a special catalog entry

        # Face under cursor (for PLACING highlight)
        self.hover_face: int = -1

        # Inventory: list of (label, mesh) committed items
        self.inventory: list[tuple[str, DECMesh]] = []
        self.fabricated_objects: list[dict] = []
        self.last_export_path: str = ""
        self.last_import_path: str = ""
        self.last_blueprint: Optional[dict] = None

        # History stack for undo
        self._history: list[DECMesh] = []

        # Symmetry config (shared, used when confirming placement)
        self.symmetry = SymmetryConfig(
            mode  = cfg.get("symmetry", {}).get("mode",  "none"),
            axis  = cfg.get("symmetry", {}).get("axis",  "z"),
            count = int(cfg.get("symmetry", {}).get("count", 4)),
        )

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def active_mesh(self) -> DECMesh:
        """The mesh to display in the workspace viewport."""
        if self.mode == WorkspaceMode.REVIEW and self.review_mesh is not None:
            return self.review_mesh
        if self.mode == WorkspaceMode.PLACED and self.placement is not None:
            return self.placement.preview(self._current_base())
        return self._current_base()

    def _current_base(self) -> DECMesh:
        return self.built_mesh if self.built_mesh is not None else self.base_solid

    # ── Palette actions ───────────────────────────────────────────────────────

    def pick_solid(self, solid_id: str) -> bool:
        """User selected a solid from the palette panel."""
        if solid_id not in platonic_solids.available():
            return False
        self.picked_id    = solid_id
        self.picked_piece = platonic_solids.get(solid_id)
        self.mode         = WorkspaceMode.PICKED
        self.placement    = None
        self.hover_face   = -1
        return True

    def _pick_mesh(self, mesh_id: str, mesh: DECMesh) -> bool:
        self.picked_id = str(mesh_id)
        self.picked_piece = mesh
        self.picked_special = False
        self.mode = WorkspaceMode.PICKED
        self.placement = None
        self.hover_face = -1
        return True

    def pick_generated_mesh(self, mesh_id: str, mesh: DECMesh) -> bool:
        """Pick a programmatically generated mesh as the current piece."""
        return self._pick_mesh(mesh_id, mesh)

    def set_process_tab(self, tab: str):
        self.process_tab = str(tab)

    def set_snap(self, enabled: Optional[bool] = None,
                 angle_deg: Optional[float] = None,
                 distance_m: Optional[float] = None):
        if enabled is not None:
            self.snap_enabled = bool(enabled)
        if angle_deg is not None:
            self.snap_angle_deg = float(np.clip(angle_deg, 1.0, 90.0))
        if distance_m is not None:
            self.snap_distance_m = float(np.clip(distance_m, 0.001, 1.0))

    def adjust_gimbal(self, d_pan_deg: float, d_tilt_deg: float):
        self.gimbal[0] = float(self.gimbal[0] + d_pan_deg)
        self.gimbal[1] = float(np.clip(self.gimbal[1] + d_tilt_deg, -85.0, 85.0))
        if self.snap_enabled:
            step = max(1.0, float(self.snap_angle_deg))
            self.gimbal[0] = round(self.gimbal[0] / step) * step
            self.gimbal[1] = round(self.gimbal[1] / step) * step

    def reset_gimbal(self):
        self.gimbal[:] = 0.0

    def _piece_with_gimbal(self, piece: DECMesh) -> DECMesh:
        pan = math.radians(float(self.gimbal[0]))
        tilt = math.radians(float(self.gimbal[1]))
        cp = math.cos(pan)
        sp = math.sin(pan)
        ct = math.cos(tilt)
        st = math.sin(tilt)
        rz = np.array([
            [cp, -sp, 0.0],
            [sp, cp, 0.0],
            [0.0, 0.0, 1.0],
        ], np.float64)
        rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, ct, -st],
            [0.0, st, ct],
        ], np.float64)
        R = rz @ rx
        c = piece.verts.mean(axis=0)
        return piece.centred().transform(R, c)

    def cancel_pick(self):
        self.picked_id      = None
        self.picked_piece   = None
        self.picked_special = False
        self.placement      = None
        self.hover_face     = -1
        self.mode           = WorkspaceMode.IDLE

    # ── Hover / face selection ─────────────────────────────────────────────────

    def hover_over_face(self, face_idx: int):
        """Called each frame with the face index under the cursor (-1 = none)."""
        if self.mode not in (WorkspaceMode.PICKED, WorkspaceMode.PLACING):
            return
        self.hover_face = face_idx
        if face_idx >= 0:
            self.mode = WorkspaceMode.PLACING
        else:
            if self.mode == WorkspaceMode.PLACING:
                self.mode = WorkspaceMode.PICKED

    def snap_to_face(self, face_idx: int) -> bool:
        """User clicked on face_idx — snap the picked piece there."""
        if self.mode not in (WorkspaceMode.PICKED, WorkspaceMode.PLACING):
            return False
        if self.picked_piece is None or face_idx < 0:
            return False

        piece_to_snap = self._piece_with_gimbal(self.picked_piece)

        # Pick the mating face on the piece: the one whose normal most
        # closely opposes the base-mesh face normal
        base_n = self._current_base().face_normal(face_idx)
        piece_normals = piece_to_snap.face_normals()
        dots = piece_normals @ (-base_n)
        best_piece_face = int(np.argmax(dots))

        self.placement = PlacementState(
            piece_id   = self.picked_id,
            piece      = piece_to_snap,
            self_face  = face_idx,
            piece_face = best_piece_face,
            symmetry   = SymmetryConfig(
                mode=self.symmetry.mode,
                axis=self.symmetry.axis,
                count=self.symmetry.count),
        )
        self.mode = WorkspaceMode.PLACED
        return True

    def _record_operation(self, op_type: str, params: Optional[dict] = None):
        self.operations.append(FabricationOperation(
            op_type=op_type,
            tab=str(self.process_tab),
            params=dict(params or {}),
        ))

    @staticmethod
    def _stable_vertex_key(v: np.ndarray, decimals: int = 9) -> tuple:
        return tuple(float(round(float(c), decimals)) for c in v)

    def _build_minimized_object_payload(self, mesh: DECMesh) -> dict:
        verts = np.asarray(mesh.verts, np.float64)
        edges = np.asarray(mesh.edges, np.int32)
        tris = np.asarray(mesh.tris, np.int32)
        tri_face = np.asarray(mesh.tri_face, np.int32)

        out_verts: list[list[float]] = []
        remap: dict[tuple, int] = {}
        old_to_new = np.zeros(len(verts), np.int32)
        for vi, v in enumerate(verts):
            key = self._stable_vertex_key(v)
            idx = remap.get(key)
            if idx is None:
                idx = len(out_verts)
                remap[key] = idx
                out_verts.append([float(v[0]), float(v[1]), float(v[2])])
            old_to_new[vi] = int(idx)

        out_edges: list[list[int]] = []
        seen_edges: set[tuple[int, int]] = set()
        for e in edges:
            a = int(old_to_new[int(e[0])])
            b = int(old_to_new[int(e[1])])
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            out_edges.append([a, b])

        faces_ccw: list[list[int]] = []
        faces_cw: list[list[int]] = []
        for f in mesh.faces:
            ff = [int(old_to_new[int(v)]) for v in f]
            if len(ff) < 3:
                continue
            faces_ccw.append(ff)
            faces_cw.append(list(reversed(ff)))

        tris_ccw = [[int(old_to_new[int(t[0])]), int(old_to_new[int(t[1])]), int(old_to_new[int(t[2])])] for t in tris]
        tris_cw = [[t[0], t[2], t[1]] for t in tris_ccw]

        tri_groups_by_polygon: list[dict[str, Any]] = []
        for face_idx, _ in enumerate(faces_ccw):
            tri_ids = [int(i) for i in np.where(tri_face == face_idx)[0].tolist()]
            tri_groups_by_polygon.append({
                "polygon_face_index": int(face_idx),
                "triangle_indices": tri_ids,
            })

        work_edge_groups: list[dict[str, Any]] = []
        for oi, op in enumerate(self.operations):
            if op.op_type not in ("subtractive_sphere_cut", "subtractive_plane_cut", "bevel_tag"):
                continue
            selector = {
                "op_type": op.op_type,
                "params": dict(op.params),
                "defer_to_depth_texture_bake": True,
            }
            work_edge_groups.append({
                "group_id": f"work_edge_{oi+1:03d}",
                "operation_index": int(oi),
                "edge_indices": [],
                "selector": selector,
            })

        if out_verts:
            arr_v = np.asarray(out_verts, np.float64)
            vmin = arr_v.min(axis=0).tolist()
            vmax = arr_v.max(axis=0).tolist()
        else:
            vmin = [0.0, 0.0, 0.0]
            vmax = [0.0, 0.0, 0.0]

        return {
            "vertex_count": int(len(out_verts)),
            "edge_count": int(len(out_edges)),
            "face_count": int(len(faces_ccw)),
            "triangle_count": int(len(tris_ccw)),
            "vertices": out_verts,
            "edges": out_edges,
            "faces_ccw": faces_ccw,
            "faces_cw": faces_cw,
            "triangles_ccw": tris_ccw,
            "triangles_cw": tris_cw,
            "accepted_face_winding": ["cw", "ccw"],
            "triangle_groups": {
                "by_polygon": tri_groups_by_polygon,
                "work_edges": work_edge_groups,
            },
            "bounds": {
                "min": [float(vmin[0]), float(vmin[1]), float(vmin[2])],
                "max": [float(vmax[0]), float(vmax[1]), float(vmax[2])],
            },
        }

    def build_fabrication_blueprint(self, label: Optional[str] = None) -> dict:
        mesh = self.review_mesh if self.review_mesh is not None else self._current_base()
        export_id = str(label or f"fabricated_{len(self.fabricated_objects)+1}")

        relationships = []
        process_itinerary = []
        for oi, op in enumerate(self.operations):
            item = {
                "index": int(oi),
                "op_type": str(op.op_type),
                "tab": str(op.tab),
                "params": dict(op.params),
            }
            process_itinerary.append(item)
            if op.op_type == "attach_primitive":
                relationships.append({
                    "relation_type": "face_attach",
                    "from_primitive": op.params.get("piece_id", ""),
                    "to_face": op.params.get("self_face", -1),
                    "from_face": op.params.get("piece_face", -1),
                    "symmetry": dict(op.params.get("symmetry", {})),
                    "gimbal_deg": list(op.params.get("gimbal_deg", [0.0, 0.0])),
                })

        minimized = self._build_minimized_object_payload(mesh)
        material_assignment = {
            "polygon_groups": [
                {
                    "group_id": f"poly_{i+1:04d}",
                    "polygon_face_index": int(g["polygon_face_index"]),
                    "triangle_indices": list(g["triangle_indices"]),
                    "material_slot": "default",
                }
                for i, g in enumerate(minimized["triangle_groups"]["by_polygon"])
            ],
            "work_edge_groups": [
                {
                    "group_id": str(g["group_id"]),
                    "operation_index": int(g["operation_index"]),
                    "edge_indices": list(g["edge_indices"]),
                    "material_slot": "work_edge_default",
                    "defer_to_depth_texture_bake": True,
                }
                for g in minimized["triangle_groups"]["work_edges"]
            ],
        }

        polygon_primitives = [
            {
                "primitive_id": f"poly_face_{i:04d}",
                "polygon_face_index": int(i),
                "vertex_indices_ccw": list(face),
                "vertex_indices_cw": list(reversed(face)),
                "triangle_indices": list(minimized["triangle_groups"]["by_polygon"][i]["triangle_indices"]),
            }
            for i, face in enumerate(minimized["faces_ccw"])
        ]

        return {
            "version": 2,
            "save_type": "fabrication_blueprint",
            "id": export_id,
            "label": str(label or f"Fabricated Object {len(self.fabricated_objects)+1}"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workspace": {
                "base_solid": self._cfg.get("base_solid", "icosahedron"),
                "process_tab": str(self.process_tab),
                "gimbal_deg": [float(self.gimbal[0]), float(self.gimbal[1])],
                "snap": {
                    "enabled": bool(self.snap_enabled),
                    "angle_deg": float(self.snap_angle_deg),
                    "distance_m": float(self.snap_distance_m),
                },
                "symmetry": {
                    "mode": str(self.symmetry.mode),
                    "axis": str(self.symmetry.axis),
                    "count": int(self.symmetry.count),
                },
            },
            "relationships": relationships,
            "process_itinerary": process_itinerary,
            "polygon_primitives": polygon_primitives,
            "optimized_object": minimized,
            "material_assignment": material_assignment,
            "placeable_entry": self.export_placeable_entry(label=label),
        }

    def export_fabrication_blueprint(self, file_path: Optional[str] = None,
                                     label: Optional[str] = None) -> str:
        blueprint = self.build_fabrication_blueprint(label=label)
        out_path = file_path
        if not out_path:
            slug = str(blueprint["id"]).strip().lower().replace(" ", "_")
            if not slug:
                slug = f"fabricated_{len(self.fabricated_objects)+1}"
            out_dir = self._blueprint_save_dir()
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{slug}.blueprint.yaml")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if _HAS_YAML:
            with open(out_path, "w", encoding="utf-8") as handle:
                _yaml.safe_dump(blueprint, handle, sort_keys=False, allow_unicode=False)
        else:
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(blueprint, handle, indent=2, sort_keys=False)
                handle.write("\n")
        self.last_export_path = str(out_path)
        return str(out_path)

    def _blueprint_save_dir(self) -> str:
        return os.path.join("configs", "duty_stations", "fabricator", "blueprints")

    def _legacy_blueprint_save_dir(self) -> str:
        return os.path.join("configs", "duty_stations", "fabricator", "saves")

    def _discover_blueprints(self) -> list[str]:
        roots = [self._blueprint_save_dir(), self._legacy_blueprint_save_dir()]
        out: list[str] = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                n = str(name).lower()
                if not (n.endswith(".blueprint.yaml") or n.endswith(".blueprint.yml") or n.endswith(".blueprint.json")):
                    continue
                out.append(os.path.join(root, name))
        out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return out

    def _load_blueprint_file(self, file_path: str) -> Optional[dict]:
        if not file_path or not os.path.isfile(file_path):
            return None
        low = str(file_path).lower()
        try:
            if low.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return data if isinstance(data, dict) else None
            if _HAS_YAML:
                with open(file_path, "r", encoding="utf-8") as handle:
                    data = _yaml.safe_load(handle) or {}
                return data if isinstance(data, dict) else None
            return None
        except Exception:
            return None

    def _restore_workspace_state_from_blueprint(self, blueprint: dict):
        ws = blueprint.get("workspace", {}) if isinstance(blueprint, dict) else {}
        self.process_tab = str(ws.get("process_tab", self.process_tab))
        g = ws.get("gimbal_deg", [0.0, 0.0])
        if isinstance(g, (list, tuple)) and len(g) >= 2:
            self.gimbal = np.array([float(g[0]), float(g[1])], np.float64)
        snap = ws.get("snap", {}) if isinstance(ws, dict) else {}
        self.snap_enabled = bool(snap.get("enabled", self.snap_enabled))
        self.snap_angle_deg = float(snap.get("angle_deg", self.snap_angle_deg))
        self.snap_distance_m = float(snap.get("distance_m", self.snap_distance_m))
        sym = ws.get("symmetry", {}) if isinstance(ws, dict) else {}
        self.symmetry.mode = str(sym.get("mode", self.symmetry.mode))
        self.symmetry.axis = str(sym.get("axis", self.symmetry.axis))
        self.symmetry.count = int(sym.get("count", self.symmetry.count))

    def _mesh_from_blueprint(self, blueprint: dict) -> Optional[DECMesh]:
        if not isinstance(blueprint, dict):
            return None
        obj = blueprint.get("optimized_object", {})
        if not isinstance(obj, dict):
            return None
        verts_raw = obj.get("vertices", [])
        faces_raw = obj.get("faces_ccw", [])
        if not verts_raw:
            return None

        verts = np.asarray(verts_raw, np.float64)
        faces: list[list[int]] = []

        if isinstance(faces_raw, list) and faces_raw:
            for f in faces_raw:
                if isinstance(f, list) and len(f) >= 3:
                    faces.append([int(v) for v in f])

        if not faces:
            poly = blueprint.get("polygon_primitives", [])
            if isinstance(poly, list):
                for p in poly:
                    if not isinstance(p, dict):
                        continue
                    vv = p.get("vertex_indices_ccw", [])
                    if isinstance(vv, list) and len(vv) >= 3:
                        faces.append([int(v) for v in vv])

        if len(verts) == 0 or not faces:
            return None
        return DECMesh.from_raw(verts, faces)

    def _operations_from_blueprint(self, blueprint: dict) -> list[FabricationOperation]:
        out: list[FabricationOperation] = []
        items = blueprint.get("process_itinerary", []) if isinstance(blueprint, dict) else []
        if not isinstance(items, list):
            return out
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(FabricationOperation(
                op_type=str(item.get("op_type", "")),
                tab=str(item.get("tab", self.process_tab)),
                params=dict(item.get("params", {})),
            ))
        return out

    def import_blueprint(self, file_path: str) -> bool:
        bp = self._load_blueprint_file(file_path)
        if bp is None:
            return False
        mesh = self._mesh_from_blueprint(bp)
        if mesh is None:
            return False

        self._restore_workspace_state_from_blueprint(bp)
        self.built_mesh = mesh
        self.review_mesh = None
        self.placement = None
        self._history.clear()
        self.operations = self._operations_from_blueprint(bp)
        self.mode = WorkspaceMode.IDLE
        self.last_import_path = str(file_path)
        self.last_blueprint = bp
        return True

    def import_latest_blueprint(self) -> bool:
        files = self._discover_blueprints()
        if not files:
            return False
        return self.import_blueprint(files[0])

    def replay_blueprint(self, file_path: Optional[str] = None) -> bool:
        target = file_path
        if not target:
            target = self.last_import_path
        if not target:
            files = self._discover_blueprints()
            if not files:
                return False
            target = files[0]
        if not self.import_blueprint(str(target)):
            return False
        self.review_mesh = self._current_base()
        self.mode = WorkspaceMode.REVIEW
        return True

    def apply_subtractive_sphere_cut(self, center: np.ndarray, radius: float) -> bool:
        mesh = self._current_base()
        center = np.asarray(center, np.float64).reshape(3)
        radius = max(1e-4, float(radius))
        kept_faces = []
        for face in mesh.faces:
            c = mesh.verts[np.asarray(face, np.int32)].mean(axis=0)
            if float(np.linalg.norm(c - center)) > radius:
                kept_faces.append(list(face))
        if len(kept_faces) == len(mesh.faces):
            return False
        if not kept_faces:
            return False
        self._history.append(mesh)
        self.built_mesh = DECMesh.from_raw(mesh.verts.copy(), kept_faces)
        self.mode = WorkspaceMode.IDLE
        self.placement = None
        self._record_operation("subtractive_sphere_cut", {
            "center": center.tolist(),
            "radius": radius,
        })
        return True

    def apply_subtractive_plane_cut(self, axis: str = "z", offset: float = 0.0,
                                    keep_negative: bool = True) -> bool:
        mesh = self._current_base()
        axis = str(axis).lower()
        ai = 0 if axis == "x" else (1 if axis == "y" else 2)
        kept_faces = []
        for face in mesh.faces:
            c = mesh.verts[np.asarray(face, np.int32)].mean(axis=0)
            test = float(c[ai] - offset)
            ok = (test <= 0.0) if keep_negative else (test >= 0.0)
            if ok:
                kept_faces.append(list(face))
        if len(kept_faces) == len(mesh.faces):
            return False
        if not kept_faces:
            return False
        self._history.append(mesh)
        self.built_mesh = DECMesh.from_raw(mesh.verts.copy(), kept_faces)
        self.mode = WorkspaceMode.IDLE
        self.placement = None
        self._record_operation("subtractive_plane_cut", {
            "axis": axis,
            "offset": float(offset),
            "keep_negative": bool(keep_negative),
        })
        return True

    def mark_bevel(self, amount_m: float = 0.01):
        self._record_operation("bevel_tag", {"amount_m": float(amount_m)})

    def export_scene_object(self, label: Optional[str] = None) -> dict:
        mesh = self.review_mesh if self.review_mesh is not None else self._current_base()
        tri = mesh.gl_triangles()
        verts = tri[:, :3].astype(np.float64, copy=False)
        normals = tri[:, 3:6].astype(np.float64, copy=False)
        vmin = verts.min(axis=0) if len(verts) else np.zeros(3, np.float64)
        vmax = verts.max(axis=0) if len(verts) else np.zeros(3, np.float64)
        out = {
            "id": str(label or f"fabricated_{len(self.fabricated_objects)+1}"),
            "type": "mesh",
            "label": str(label or f"Fabricated Object {len(self.fabricated_objects)+1}"),
            "verts": verts.tolist(),
            "normals": normals.tolist(),
            "bounds": {
                "min": vmin.tolist(),
                "max": vmax.tolist(),
            },
            "fabrication": {
                "tab": str(self.process_tab),
                "gimbal_deg": [float(self.gimbal[0]), float(self.gimbal[1])],
                "snap_enabled": bool(self.snap_enabled),
                "symmetry": {
                    "mode": str(self.symmetry.mode),
                    "axis": str(self.symmetry.axis),
                    "count": int(self.symmetry.count),
                },
                "operations": [
                    {"op_type": op.op_type, "tab": op.tab, "params": dict(op.params)}
                    for op in self.operations
                ],
            },
        }
        return out

    def export_placeable_entry(self, label: Optional[str] = None,
                               category: str = "fabricated_objects") -> dict:
        scene_obj = self.export_scene_object(label=label)
        bmin = np.asarray(scene_obj["bounds"]["min"], np.float64)
        bmax = np.asarray(scene_obj["bounds"]["max"], np.float64)
        ext = np.maximum(1e-6, bmax - bmin)
        footprint = [max(1, int(math.ceil(ext[0]))), max(1, int(math.ceil(ext[1])))]
        levels = max(1, int(math.ceil(ext[2])))

        obj_id = str(scene_obj.get("id", "")).lower()
        obj_label = str(scene_obj.get("label", "")).lower()
        is_duty_station = ("duty_station" in obj_id) or ("duty station" in obj_label)

        out_category = str(category)
        out_kind = "fabricated_mesh"
        out_mesh_type = "marker_box"
        if is_duty_station:
            out_category = "duty_stations"
            out_kind = "fabricated_duty_station"
            out_mesh_type = "duty_station_blueprint"

        return {
            "id": scene_obj["id"],
            "label": scene_obj["label"],
            "category": out_category,
            "kind": out_kind,
            "mesh_type": out_mesh_type,
            "surfaces": ["floor"],
            "footprint_xy": footprint,
            "level_span": levels,
            "summary": "Fabricator-exported placeable object",
            "blueprint_id": scene_obj["id"],
            "scene_object": scene_obj,
        }

    # ── Placement actions ─────────────────────────────────────────────────────

    def set_symmetry(self, mode: str, axis: str = None, count: int = None):
        self.symmetry.mode = mode
        if axis  is not None: self.symmetry.axis  = axis
        if count is not None: self.symmetry.count = count
        if self.placement is not None:
            self.placement.symmetry = SymmetryConfig(
                mode=self.symmetry.mode, axis=self.symmetry.axis,
                count=self.symmetry.count)
            self.placement.invalidate()

    def confirm_placement(self) -> bool:
        """Commit the current placement to the build mesh."""
        if self.mode != WorkspaceMode.PLACED or self.placement is None:
            return False
        self._history.append(self._current_base())
        new_mesh = self.placement.preview(self._current_base())
        self._record_operation("attach_primitive", {
            "piece_id": str(self.placement.piece_id or ""),
            "self_face": int(self.placement.self_face),
            "piece_face": int(self.placement.piece_face),
            "symmetry": {
                "mode": str(self.placement.symmetry.mode),
                "axis": str(self.placement.symmetry.axis),
                "count": int(self.placement.symmetry.count),
            },
            "gimbal_deg": [float(self.gimbal[0]), float(self.gimbal[1])],
        })
        self.built_mesh = new_mesh
        self.placement  = None
        self.mode       = WorkspaceMode.IDLE
        self.picked_id  = None
        self.picked_piece = None
        return True

    def cancel_placement(self):
        if self.mode == WorkspaceMode.PLACED:
            self.placement = None
            self.mode      = WorkspaceMode.PICKED   # keep piece in hand

    def undo(self) -> bool:
        if not self._history:
            return False
        self.built_mesh = self._history.pop()
        self.mode       = WorkspaceMode.IDLE
        self.placement  = None
        return True

    # ── Review / inventory ────────────────────────────────────────────────────

    def enter_review(self):
        self.review_mesh = self._current_base()
        self.mode        = WorkspaceMode.REVIEW

    def exit_review(self):
        self.review_mesh = None
        self.mode        = WorkspaceMode.IDLE

    def store_to_inventory(self, label: str = None):
        if self.review_mesh is None:
            self.review_mesh = self._current_base()
        label = label or f"item_{len(self.inventory)+1}"
        self.inventory.append((label, self.review_mesh))
        self.fabricated_objects.append(self.export_scene_object(label=label))
        self._reset_build()

    def clear_build(self):
        self._reset_build()

    def _reset_build(self):
        self.built_mesh  = None
        self.placement   = None
        self.review_mesh = None
        self._history.clear()
        self.mode           = WorkspaceMode.IDLE
        self.picked_id      = None
        self.picked_piece   = None
        self.picked_special = False
        self.operations.clear()

    # ── Catalog (solids + special items) ─────────────────────────────────────

    #: Special items that the fabricator can place directly into a room.
    SPECIAL_CATALOG: "list[dict]" = [
        {
            "id":    "light_point",
            "label": "Point Light",
            "kind":  "light",
            "icon":  "💡",
            "defaults": {
                "intensity": 1.0,
                "radius_m":  8.0,
                "color":     [1.0, 0.95, 0.88],
            },
        },
        {
            "id":    "light_spot",
            "label": "Spot Light",
            "kind":  "light",
            "icon":  "🔦",
            "defaults": {
                "intensity":      1.2,
                "spot_angle_deg": 35.0,
                "color":          [1.0, 0.92, 0.75],
            },
        },
        {
            "id":    "portal_frame",
            "label": "Portal Frame",
            "kind":  "portal",
            "icon":  "🚪",
            "defaults": {
                "target_room_id": "",
            },
        },
        {
            "id":    "guitar_acoustic",
            "label": "Acoustic Guitar",
            "kind":  "instrument",
            "icon":  "🎸",
            "description": (
                "6-string steel-string acoustic. "
                "FDTD Kirchhoff-plate body resonance."
            ),
            "thumbnail_color": [0.56, 0.30, 0.12],
            "preset_module":   "guitar_fabricator_preset",
            "preset_class":    "GuitarFabricatorPreset",
            "defaults": {
                "excitation":  "strum",
                "fret":        0,
                "fretless":    False,
                "amr_backend": "cpu",
            },
        },
    ]

    def catalog_items(self) -> "list[dict]":
        """Return all palette items: platonic solids then special items.

        Each entry is a dict with keys ``id``, ``label``, ``kind``, and
        (for special items) ``icon`` and ``defaults``.
        """
        solids = [
            {"id": s, "label": s.title(), "kind": "solid", "icon": "⬡"}
            for s in platonic_solids.available()
        ]
        return solids + list(self.SPECIAL_CATALOG)

    def pick_catalog_item(self, item_id: str) -> bool:
        """Pick a solid or special item from the catalog by its ``id``.

        * Solid items delegate to the existing :meth:`pick_solid` method.
        * Special items set ``picked_id`` and ``picked_special`` and move
          the workspace to ``PICKED`` mode (no mesh preview is generated).

        Returns ``True`` on success.
        """
        if item_id in platonic_solids.available():
            self.picked_special = False
            return self.pick_solid(item_id)

        if item_id in self._primitive_meshes:
            return self._pick_mesh(item_id, self._primitive_meshes[item_id])

        for entry in self.SPECIAL_CATALOG:
            if entry["id"] == item_id:
                self.picked_id      = item_id
                self.picked_piece   = None
                self.picked_special = True
                self.mode           = WorkspaceMode.PICKED
                self.placement      = None
                self.hover_face     = -1
                return True

        return False

    def to_save_dict(self) -> dict:
        return {
            "version": 1,
            "base_solid": self._cfg.get("base_solid", "icosahedron"),
            "process_tab": str(self.process_tab),
            "gimbal_deg": [float(self.gimbal[0]), float(self.gimbal[1])],
            "snap": {
                "enabled": bool(self.snap_enabled),
                "angle_deg": float(self.snap_angle_deg),
                "distance_m": float(self.snap_distance_m),
            },
            "symmetry": {
                "mode": str(self.symmetry.mode),
                "axis": str(self.symmetry.axis),
                "count": int(self.symmetry.count),
            },
            "operations": [
                {
                    "op_type": op.op_type,
                    "tab": op.tab,
                    "params": dict(op.params),
                }
                for op in self.operations
            ],
            "fabricated_objects": list(self.fabricated_objects),
            "inventory_labels": [lbl for lbl, _ in self.inventory],
        }

    @property
    def full_palette(self) -> "list[dict]":
        """Alias for :meth:`catalog_items`; kept for convenient access."""
        return self.catalog_items()
