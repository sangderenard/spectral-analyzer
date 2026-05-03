"""guitar_item.py
==================
GuitarConfig and GuitarItem — the guitar as a structured, inventory-ready
game object.

GuitarItem owns:
  • a list of GuitarPart instances (geometry + materials + visibility)
  • a GuitarConfig (all physical constants for sound simulation)
  • the outline polygon and world-space model matrix pair (M, Minv)
  • sim_sidecar dict — populated by InstrumentStation after physics build
  • inventory state (_in_inventory flag)

No GL state.  No physics process.  No pygame.
The physics process lives in InstrumentStation; this is the inert data object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from guitar_part import GuitarPart, PartVisibility
from guitar_geometry import (
    guitar_outline, guitar_model_matrix, build_guitar_parts,
    string_paths as _string_paths_fn,
    _SOUNDHOLE_CX, _SOUNDHOLE_CY, _SOUNDHOLE_R,
    _BODY_H, _SCALE_LENGTH, _STRING_CLEARANCE, _STAND_HEIGHT,
    _N_RENDER_SEGS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants (from demo_pluck_gl.py)
# ─────────────────────────────────────────────────────────────────────────────

#: Standard-tuning open-string fundamentals (Hz) for a 6-string guitar.
STRING_FUNDAMENTALS_HZ = (82.4, 110.0, 146.8, 196.0, 246.9, 329.6)  # E2 A2 D3 G3 B3 E4

#: D'Addario EJ16 light acoustic gauge in inches.
STRING_GAUGES_IN = (0.046, 0.036, 0.026, 0.017, 0.013, 0.010)

#: Saddle tensions (N) matching the above gauges at scale length 648 mm.
STRING_TENSIONS_N = (61.0, 76.8, 90.5, 112.9, 106.5, 73.0)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuitarConfig:
    """All physical constants describing a specific guitar instrument.

    These feed directly into the FDTD physics worker config dict.
    Geometry functions in guitar_geometry.py accept these as explicit params.
    """
    # --- Strings ---
    n_strings:               int   = 6
    string_fundamentals_hz:  tuple = STRING_FUNDAMENTALS_HZ
    string_gauges_in:        tuple = STRING_GAUGES_IN
    string_tensions_n:       tuple = STRING_TENSIONS_N
    scale_length_m:          float = _SCALE_LENGTH
    string_clearance_m:      float = _STRING_CLEARANCE

    # --- Body ---
    body_h:                  float = _BODY_H
    soundhole_cx:            float = _SOUNDHOLE_CX
    soundhole_cy:            float = _SOUNDHOLE_CY
    soundhole_r:             float = _SOUNDHOLE_R
    stand_height_m:          float = _STAND_HEIGHT

    # --- Articulation ---
    fret:                    int   = 0
    fretless:                bool  = False
    pluck_pos:               float = 0.20   # normalised from saddle
    pluck_amp:               float = 0.003
    bridge_force_scale:      float = 1.0

    # --- FDTD simulation ---
    dx:                      float = 0.006  # cell size metres
    pressure_margin_cells:   int   = 56     # free-air padding each side
    pressure_pml_cells:      int   = 28     # PML layer thickness
    render_segs:             int   = _N_RENDER_SEGS
    amr_backend:             str   = "cpu"
    amr_cache_grid:          bool  = True
    excitation:              str   = "strum"  # "strum" | "a-pluck" | "rest"
    diagnostic_frames:       int   = 600

    def physics_dict(self) -> dict:
        """Return config dict accepted by _PhysicsProcess / _physics_worker."""
        return {
            "n_strings":             self.n_strings,
            "dx":                    float(self.dx),
            "pressure_margin_cells": int(self.pressure_margin_cells),
            "pressure_pml_cells":    int(self.pressure_pml_cells),
            "render_segs":           int(self.render_segs),
            "bridge_force_scale":    float(self.bridge_force_scale),
            "excitation":            str(self.excitation),
            "diagnostic_frames":     int(self.diagnostic_frames),
            "fret":                  max(0, int(self.fret)),
            "fretless":              bool(self.fretless),
            "amr_backend":           str(self.amr_backend),
            "amr_cache_grid":        bool(self.amr_cache_grid),
        }


# ─────────────────────────────────────────────────────────────────────────────
# GuitarItem
# ─────────────────────────────────────────────────────────────────────────────

class GuitarItem:
    """A guitar as an inventory-ready game item.

    Instantiate via GuitarFabricatorPreset.fabricate() or directly from
    a GuitarConfig.  Owns:

      parts           — list of GuitarPart in draw order
      config          — GuitarConfig (all physical constants)
      outline         — (N, 2) float32 body outline polygon
      model_matrix    — (4, 4) float32, guitar frame → world frame
      model_matrix_inv— (4, 4) float32, world frame → guitar frame
      sim_sidecar     — populated by InstrumentStation after physics build;
                        keys mirror the ``info`` dict from _build_physics
                        (Nx, Ny, dx, gx_min, gy_min, plate_active_2d, …)

    Inventory lifecycle::

        item = preset.fabricate()
        assert item.in_inventory
        station.load_item(item)
        # … use …
        station.unload_item()
        item.release()
    """

    def __init__(self,
                 parts:   List[GuitarPart],
                 config:  GuitarConfig,
                 outline: np.ndarray):
        self._parts:    List[GuitarPart] = list(parts)
        self._config:   GuitarConfig     = config
        self._outline:  np.ndarray       = np.asarray(outline, dtype=np.float32)
        self._M, self._Minv = guitar_model_matrix(
            self._outline, config.stand_height_m
        )
        self.sim_sidecar: Dict[str, object] = {}
        self._in_inventory: bool = False

    # ── Parts access ─────────────────────────────────────────────────────────

    @property
    def parts(self) -> List[GuitarPart]:
        return self._parts

    def part(self, name: str) -> Optional[GuitarPart]:
        for p in self._parts:
            if p.name == name:
                return p
        return None

    def parts_by_tag(self, sim_tag: str) -> List[GuitarPart]:
        return [p for p in self._parts if p.sim_tag == sim_tag]

    def set_visibility(self, name: str, vis: PartVisibility) -> None:
        for i, p in enumerate(self._parts):
            if p.name == name:
                self._parts[i] = GuitarPart(
                    name=p.name, render_mat=p.render_mat, ray_mat=p.ray_mat,
                    visibility=vis,
                    verts=p.verts, norms=p.norms, indices=p.indices,
                    line_verts=p.line_verts,
                    plate_verts=p.plate_verts, plate_indices=p.plate_indices,
                    sim_tag=p.sim_tag, sim_sidecar=p.sim_sidecar,
                )
                return

    def set_all_visibility(self, vis: PartVisibility) -> None:
        self._parts = [
            GuitarPart(
                name=p.name, render_mat=p.render_mat, ray_mat=p.ray_mat,
                visibility=vis,
                verts=p.verts, norms=p.norms, indices=p.indices,
                line_verts=p.line_verts,
                plate_verts=p.plate_verts, plate_indices=p.plate_indices,
                sim_tag=p.sim_tag, sim_sidecar=p.sim_sidecar,
            )
            for p in self._parts
        ]

    # ── Physical properties ───────────────────────────────────────────────────

    @property
    def config(self) -> GuitarConfig:
        return self._config

    @property
    def outline(self) -> np.ndarray:
        return self._outline

    @property
    def model_matrix(self) -> np.ndarray:
        """Guitar-frame → world-frame transform (4×4 float32)."""
        return self._M

    @property
    def model_matrix_inv(self) -> np.ndarray:
        """World-frame → guitar-frame transform (4×4 float32)."""
        return self._Minv

    def string_paths(self, n_segs: Optional[int] = None) -> List[np.ndarray]:
        """Return per-string rest-position paths in guitar frame.

        Each element is shaped (n_segs+1, 3) float32.
        """
        segs = n_segs if n_segs is not None else self._config.render_segs
        return _string_paths_fn(
            self._outline,
            n_strings       = self._config.n_strings,
            n_segs          = segs,
            body_h          = self._config.body_h,
            scale_length    = self._config.scale_length_m,
            string_clearance= self._config.string_clearance_m,
            fret            = self._config.fret,
        )

    def physics_config(self) -> dict:
        """Config dict for the FDTD physics worker (see GuitarConfig.physics_dict)."""
        return self._config.physics_dict()

    # ── BVH export ───────────────────────────────────────────────────────────

    def bvh_triangles(self) -> List[dict]:
        """Export all opaque tri-mesh parts as flat dicts for BVH construction.

        Each dict has keys: verts (N,3), norms (N,3),
        mat_in (4,), mat_out (4,), albedo (3,).
        Only OPAQUE and TRANSPARENT parts with triangle geometry are included
        (HIDDEN parts are omitted — they have no ray-scene presence).
        """
        result = []
        for p in self._parts:
            if p.visibility is PartVisibility.HIDDEN:
                continue
            if p.verts is None and p.plate_verts is None:
                continue   # line-only parts don't enter the BVH
            v = p.triangle_verts_flat()
            n = p.triangle_norms_flat()
            if len(v) == 0:
                continue
            result.append({
                "verts":   v,
                "norms":   n,
                "mat_in":  np.array(p.ray_mat.mat_in_vec(),  np.float32),
                "mat_out": np.array(p.ray_mat.mat_out_vec(), np.float32),
                "albedo":  np.array(p.render_mat.color,       np.float32),
            })
        return result

    # ── World-space helpers ───────────────────────────────────────────────────

    def world_verts(self, part_name: str) -> Optional[np.ndarray]:
        """Transform a named part's vertex positions to world space."""
        p = self.part(part_name)
        if p is None or p.verts is None:
            return None
        v = p.verts.reshape(-1, 3)
        ones = np.ones((len(v), 1), dtype=v.dtype)
        v4 = np.hstack([v, ones])
        return (v4 @ self._M.T)[:, :3]

    # ── Inventory lifecycle ───────────────────────────────────────────────────

    @property
    def in_inventory(self) -> bool:
        return self._in_inventory

    def acquire(self) -> None:
        """Mark this item as held in the player's inventory."""
        self._in_inventory = True

    def release(self) -> None:
        """Remove this item from the player's inventory."""
        self._in_inventory = False

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        vis_counts: Dict[str, int] = {}
        for p in self._parts:
            k = p.visibility.value
            vis_counts[k] = vis_counts.get(k, 0) + 1
        inv = "in_inventory" if self._in_inventory else "not_in_inventory"
        return (f"GuitarItem({len(self._parts)} parts, "
                f"vis={vis_counts}, {inv})")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory (for quick construction without the preset system)
# ─────────────────────────────────────────────────────────────────────────────

def make_guitar_item(config: Optional[GuitarConfig] = None,
                     n_outline_pts: int = 128) -> GuitarItem:
    """Build a GuitarItem from a GuitarConfig (or defaults).

    This is the low-level constructor used by GuitarFabricatorPreset and
    by InstrumentStation during hot-reload scenarios.
    """
    cfg     = config if config is not None else GuitarConfig()
    outline = guitar_outline(n_pts=n_outline_pts)
    parts   = build_guitar_parts(
        outline,
        body_h          = cfg.body_h,
        n_strings       = cfg.n_strings,
        scale_length    = cfg.scale_length_m,
        string_clearance= cfg.string_clearance_m,
        soundhole_cx    = cfg.soundhole_cx,
        soundhole_cy    = cfg.soundhole_cy,
        soundhole_r     = cfg.soundhole_r,
        active_fret     = cfg.fret,
        fretless        = cfg.fretless,
        n_render_segs   = cfg.render_segs,
    )
    return GuitarItem(parts, cfg, outline)
