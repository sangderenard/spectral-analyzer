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

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from dec_mesh import DECMesh
import platonic_solids


# ─────────────────────────────────────────────────────────────────────────────

class WorkspaceMode(Enum):
    IDLE    = "idle"
    PICKED  = "picked"
    PLACING = "placing"
    PLACED  = "placed"
    REVIEW  = "review"


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

        # Palette selection
        self.picked_id:   Optional[str] = None
        self.picked_piece:Optional[DECMesh] = None

        # Face under cursor (for PLACING highlight)
        self.hover_face: int = -1

        # Inventory: list of (label, mesh) committed items
        self.inventory: list[tuple[str, DECMesh]] = []

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

    def cancel_pick(self):
        self.picked_id    = None
        self.picked_piece = None
        self.placement    = None
        self.hover_face   = -1
        self.mode         = WorkspaceMode.IDLE

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

        # Pick the mating face on the piece: the one whose normal most
        # closely opposes the base-mesh face normal
        base_n = self._current_base().face_normal(face_idx)
        piece_normals = self.picked_piece.face_normals()
        dots = piece_normals @ (-base_n)
        best_piece_face = int(np.argmax(dots))

        self.placement = PlacementState(
            piece_id   = self.picked_id,
            piece      = self.picked_piece,
            self_face  = face_idx,
            piece_face = best_piece_face,
            symmetry   = SymmetryConfig(
                mode=self.symmetry.mode,
                axis=self.symmetry.axis,
                count=self.symmetry.count),
        )
        self.mode = WorkspaceMode.PLACED
        return True

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
        self._reset_build()

    def clear_build(self):
        self._reset_build()

    def _reset_build(self):
        self.built_mesh  = None
        self.placement   = None
        self.review_mesh = None
        self._history.clear()
        self.mode        = WorkspaceMode.IDLE
        self.picked_id   = None
        self.picked_piece= None
