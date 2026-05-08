"""
room_surface.py — Abstract room surface objects for progressive material fill.

Each RoomSurface holds an arbitrary set of triangles and tracks a material
delivery wavefront that emanates outward from one or more seed points.
Triangles are filled in BFS (breadth-first) order starting from each seeded
triangle; calling add_seed() at any time injects a new wavefront origin.

Visual contract:
  - Unfinished triangles (material_id == -1) render with the
    construction_borosilicate material shader (faintly emissive borosilicate
    glass).
  - Finished triangles render with their assigned material color.

Seeding rules:
  - Room control station: set_station_origin() seeds from the station's world
    position.  Call this once on load; subsequent calls add additional seeds
    without clearing existing fill.
  - Player drop: add_seed(pos) finds the nearest unfinished triangle to pos and
    injects it into the wavefront.  When multiple triangles are equidistant
    (within 1 mm), one is chosen at random.
  - Disconnected deposits: same add_seed() path — if the position is far from
    any existing wavefront the new seed immediately starts its own expanding
    ring.

The dirty flag is set whenever fill state changes so the renderer can rebuild
only the VAOs that actually changed.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _tri_areas_and_centroids(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    tris: (N, 3, 3) float32 — N triangles, 3 verts, xyz each.
    Returns areas (N,) and centroids (N, 3).
    """
    a = tris[:, 0, :]
    b = tris[:, 1, :]
    c = tris[:, 2, :]
    cross = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    centroids = (a + b + c) / 3.0
    return areas.astype(np.float32), centroids.astype(np.float32)


def tris_to_pos_norm(tris: np.ndarray, flip_normals: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (N, 3, 3) triangle array to flat (N*3, 3) position and normal arrays.
    Normals are face normals (constant across each triangle).
    flip_normals=True reverses the computed normals (for outward-normal geometry
    that needs inward-facing normals for room interior rendering).
    """
    if len(tris) == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty

    a = tris[:, 0, :]
    b = tris[:, 1, :]
    c = tris[:, 2, :]
    cross = np.cross(b - a, c - a)
    norms_mag = np.linalg.norm(cross, axis=1, keepdims=True)
    norms_mag = np.where(norms_mag < 1e-10, 1.0, norms_mag)
    face_norms = cross / norms_mag  # (N, 3)
    if flip_normals:
        face_norms = -face_norms

    pos_flat  = tris.reshape(-1, 3)               # (N*3, 3)
    norm_flat = np.repeat(face_norms, 3, axis=0)  # (N*3, 3)
    return pos_flat.astype(np.float32), norm_flat.astype(np.float32)


# --------------------------------------------------------------------------- #
# RoomSurface base
# --------------------------------------------------------------------------- #

class RoomSurface:
    """
    Abstract surface holding N triangles with progressive BFS material fill.

    Parameters
    ----------
    flip_normals : bool
        True for surfaces whose raw geometry has outward-facing normals
        (walls from build_envelope_meshes) so the GL data will be corrected
        to inward-facing for interior rendering.
    """

    def __init__(self, flip_normals: bool = False):
        self._flip_normals = flip_normals
        self._tris: np.ndarray        = np.zeros((0, 3, 3), dtype=np.float32)
        self._areas: np.ndarray       = np.zeros(0, dtype=np.float32)
        self._centroids: np.ndarray   = np.zeros((0, 3), dtype=np.float32)
        self._material_id: np.ndarray = np.full(0, -1, dtype=np.int32)  # -1 = glass
        self._total_area: float       = 0.0
        self._covered_area: float     = 0.0
        self._dirty: bool             = False
        self._station_origin: np.ndarray = np.zeros(3, dtype=np.float32)

        # BFS wavefront state
        self._adj: List[List[int]]    = []     # triangle adjacency list
        self._bfs_deque: deque        = deque()
        self._bfs_queued: np.ndarray  = np.zeros(0, dtype=bool)

    # ------------------------------------------------------------------ setup

    def set_triangles(self, tris: np.ndarray,
                      station_origin: Optional[np.ndarray] = None) -> None:
        """
        Replace the triangle set.  All triangles are reset to glass state.

        tris: (N, 3, 3) float32
        station_origin: (3,) float32 — initial BFS seed position
        """
        self._tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3, 3)
        n = len(self._tris)
        self._areas, self._centroids = _tri_areas_and_centroids(self._tris)
        self._material_id  = np.full(n, -1, dtype=np.int32)
        self._bfs_queued   = np.zeros(n, dtype=bool)
        self._total_area   = float(self._areas.sum())
        self._covered_area = 0.0

        if station_origin is not None:
            self._station_origin = np.asarray(station_origin, dtype=np.float32)

        self._bfs_deque.clear()
        self._build_adjacency()

        if n > 0:
            self.add_seed(self._station_origin)

        self._dirty = True

    def set_station_origin(self, origin: np.ndarray) -> None:
        """Add a new BFS seed at origin without clearing existing fill."""
        self._station_origin = np.asarray(origin, dtype=np.float32)
        self.add_seed(self._station_origin)
        self._dirty = True

    # ------------------------------------------------------------------ adjacency

    def _build_adjacency(self) -> None:
        """Build triangle–triangle adjacency via shared edges (0.1 mm precision)."""
        n = len(self._tris)
        self._adj = [[] for _ in range(n)]
        if n == 0:
            return

        edge_to_tris: dict = defaultdict(list)
        for i in range(n):
            tri = self._tris[i]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                # Quantise to 0.1 mm to absorb float imprecision at shared edges.
                va = tuple((tri[a] * 10_000).round().astype(np.int64).tolist())
                vb = tuple((tri[b] * 10_000).round().astype(np.int64).tolist())
                key = (va, vb) if va < vb else (vb, va)
                edge_to_tris[key].append(i)

        for tris_list in edge_to_tris.values():
            if len(tris_list) < 2:
                continue
            for x in range(len(tris_list)):
                for y in range(x + 1, len(tris_list)):
                    a, b = tris_list[x], tris_list[y]
                    if b not in self._adj[a]:
                        self._adj[a].append(b)
                    if a not in self._adj[b]:
                        self._adj[b].append(a)

    # ------------------------------------------------------------------ seeding

    def _enqueue_triangle(self, idx: int) -> None:
        """Add triangle idx to the BFS front if not already queued or filled."""
        if (0 <= idx < len(self._bfs_queued)
                and not self._bfs_queued[idx]
                and self._material_id[idx] == -1):
            self._bfs_queued[idx] = True
            self._bfs_deque.appendleft(idx)  # new seeds go to front for immediate effect

    def add_seed(self, pos: np.ndarray,
                 rng: Optional[np.random.Generator] = None) -> Optional[int]:
        """
        Inject a BFS seed at the unfinished triangle nearest to pos.

        When multiple triangles are equidistant (within 1 mm of the nearest),
        one is chosen at random — satisfying the equidistant-deposit rule.

        Returns the triangle index seeded, or None if the surface is full.
        """
        n = len(self._tris)
        if n == 0:
            return None
        unfinished = np.where(self._material_id == -1)[0]
        if len(unfinished) == 0:
            return None

        p = np.asarray(pos, np.float32)
        cents = self._centroids[unfinished]  # (M, 3)
        dists = np.sqrt(np.sum((cents - p) ** 2, axis=1))
        min_d = float(dists.min())

        # Equidistant group: all unfinished triangles within 1 mm of nearest.
        near_mask = (dists - min_d) < 1e-3
        candidates = unfinished[near_mask]

        if len(candidates) == 1:
            idx = int(candidates[0])
        elif rng is not None:
            idx = int(rng.choice(candidates))
        else:
            idx = int(np.random.choice(candidates))

        self._enqueue_triangle(idx)
        return idx

    def add_seed_at_triangle(self, tri_idx: int) -> None:
        """Directly seed BFS from a specific triangle index."""
        self._enqueue_triangle(tri_idx)

    # ------------------------------------------------------------------ fill

    def fill_batch(self, material_id: int, area_m2: float) -> float:
        """
        Deliver up to area_m2 m² of material along the BFS wavefront.

        Triangles are filled in BFS order from all active seed points.
        A triangle must be fully consumed (its full area deducted) before
        the wavefront advances to its neighbours.

        Returns the area actually consumed (≤ area_m2).
        """
        if area_m2 <= 0.0 or not self._bfs_deque:
            return 0.0

        budget   = area_m2
        consumed = 0.0

        while self._bfs_deque:
            idx = self._bfs_deque[0]  # peek — pop only if we can afford it

            if self._material_id[idx] != -1:
                self._bfs_deque.popleft()  # already filled by a prior seed
                continue

            tri_area = float(self._areas[idx])
            if budget < tri_area:
                break  # insufficient budget; leave triangle at front for next call

            self._bfs_deque.popleft()
            self._material_id[idx] = material_id
            self._covered_area    += tri_area
            budget                -= tri_area
            consumed              += tri_area
            self._dirty            = True

            for nbr in self._adj[idx]:
                if not self._bfs_queued[nbr] and self._material_id[nbr] == -1:
                    self._bfs_queued[nbr] = True
                    self._bfs_deque.append(nbr)

        return consumed

    def reset(self) -> None:
        """Revert all triangles to glass state and re-seed from station origin."""
        self._material_id[:] = -1
        self._covered_area   = 0.0
        self._bfs_deque.clear()
        self._bfs_queued[:] = False
        self._dirty = True
        if len(self._tris) > 0:
            self.add_seed(self._station_origin)

    # ------------------------------------------------------------------ queries

    @property
    def is_finished(self) -> bool:
        return len(self._tris) > 0 and bool((self._material_id >= 0).all())

    @property
    def covered_fraction(self) -> float:
        if self._total_area < 1e-9:
            return 1.0
        return min(1.0, self._covered_area / self._total_area)

    @property
    def total_area_m2(self) -> float:
        return self._total_area

    @property
    def dirty(self) -> bool:
        return self._dirty

    def clear_dirty(self) -> None:
        self._dirty = False

    # ------------------------------------------------------------------ GL data

    def get_glass_tris(self) -> np.ndarray:
        """Return (M, 3, 3) array of triangles still in glass state."""
        mask = self._material_id == -1
        return self._tris[mask]

    def get_material_tris(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (tris, material_ids) for fully-filled triangles.
        tris: (M, 3, 3), material_ids: (M,) int32.
        """
        mask = self._material_id >= 0
        return self._tris[mask], self._material_id[mask]

    def get_glass_pos_norm(self) -> tuple[np.ndarray, np.ndarray]:
        """Flat (M*3, 3) position and normal arrays for glass triangles."""
        return tris_to_pos_norm(self.get_glass_tris(), self._flip_normals)

    def get_material_pos_norm_grouped(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Returns {material_id: (pos, norm)} for each distinct finished material.
        """
        tris, mids = self.get_material_tris()
        if len(tris) == 0:
            return {}
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for mid in np.unique(mids):
            subset = tris[mids == mid]
            result[int(mid)] = tris_to_pos_norm(subset, self._flip_normals)
        return result


# --------------------------------------------------------------------------- #
# Concrete surface types
# --------------------------------------------------------------------------- #

class RoomFloor(RoomSurface):
    def __init__(self):
        super().__init__(flip_normals=False)


class RoomWall(RoomSurface):
    """Wall surfaces from build_envelope_meshes have outward normals — flip them."""
    def __init__(self):
        super().__init__(flip_normals=True)


class RoomCeiling(RoomSurface):
    def __init__(self):
        super().__init__(flip_normals=False)


# --------------------------------------------------------------------------- #
# RoomSurfaceSet
# --------------------------------------------------------------------------- #

class RoomSurfaceSet:
    """
    Owns a floor, a set of walls, and a ceiling for one applied room.

    Keys used by deliver_area():
      "floor"          — the floor surface
      "ceiling"        — the ceiling surface
      "wall_<i>"       — the i-th wall (0-indexed)
      "walls_interior" — all walls at once (interior face)
      "walls_exterior" — all walls at once (exterior face, separate surface)

    Interior / exterior visibility:
      _interior_visible  bool — render interior faces of walls/ceiling/floor
      _exterior_visible  bool — render exterior faces of walls (separate RoomWall)
    """

    def __init__(self):
        self.floor:    RoomFloor   = RoomFloor()
        self.ceiling:  RoomCeiling = RoomCeiling()
        self.walls:    List[RoomWall] = []
        self.walls_ext: List[RoomWall] = []  # exterior-facing (same tris, no flip)
        self._station_origin = np.zeros(3, dtype=np.float32)
        self.interior_visible: bool = True
        self.exterior_visible: bool = False
        self.mid_to_name: Dict[int, str] = {}  # material integer ID → material name

    # ------------------------------------------------------------------ build

    def build_from_room_cfg(self, room_cfg: dict,
                             station_origin=None) -> None:
        """
        Populate surfaces from an applied room_cfg dict.

        Reads the mesh dicts produced by build_envelope_meshes and
        build_floor_material_triangulation:

          applied_room_meshes   — dict with "walls" and "ceiling" → (N,3,3) arrays
          applied_floor_meshes  — dict with "floor_tiles", "floor_fill", etc.

        All formats tolerated: (N,3,3), flat (N*9,), or lists thereof.
        """
        if station_origin is not None:
            self._station_origin = np.asarray(station_origin, dtype=np.float32)

        # ---- Floor --------------------------------------------------------
        floor_data = room_cfg.get("applied_floor_meshes")
        if floor_data is None:
            floor_data = {}
        if isinstance(floor_data, dict):
            floor_parts = []
            for key in ("floor_tiles", "floor_fill"):
                v = floor_data.get(key)
                if v is not None:
                    floor_parts.append(v)
            floor_tris = self._concat_meshes(floor_parts) if floor_parts else np.zeros((0, 3, 3), dtype=np.float32)
        else:
            floor_tris = self._concat_meshes(floor_data)
        self.floor.set_triangles(floor_tris, self._station_origin)

        # ---- Walls and ceiling from applied_room_meshes -------------------
        room_data = room_cfg.get("applied_room_meshes")
        if room_data is None:
            room_data = {}

        if isinstance(room_data, dict):
            _w = room_data.get("walls")
            _c = room_data.get("ceiling")
            wall_tris = self._parse_mesh(_w if _w is not None else [])
            ceil_tris = self._parse_mesh(_c if _c is not None else [])
        else:
            wall_parts: List[np.ndarray] = []
            ceil_parts: List[np.ndarray] = []
            for entry in room_data:
                if isinstance(entry, dict):
                    role  = entry.get("role", "wall")
                    verts = entry.get("verts")
                    if verts is None:
                        verts = entry.get("triangles")
                    if verts is None:
                        verts = []
                else:
                    role, verts = "wall", entry
                tris = self._parse_mesh(verts)
                (ceil_parts if "ceil" in str(role).lower() else wall_parts).append(tris)
            wall_tris = self._concat_meshes(wall_parts)
            ceil_tris = self._concat_meshes(ceil_parts)

        # Walls: one surface pair (interior + exterior) from the combined wall array
        self.walls = []
        self.walls_ext = []
        if len(wall_tris):
            w_int = RoomWall()
            w_int.set_triangles(wall_tris, self._station_origin)
            self.walls.append(w_int)
            w_ext = RoomSurface(flip_normals=False)
            w_ext.set_triangles(wall_tris, self._station_origin)
            self.walls_ext.append(w_ext)

        self.ceiling.set_triangles(ceil_tris, self._station_origin)

    @staticmethod
    def _parse_mesh(verts) -> np.ndarray:
        """Convert various vert formats to (N, 3, 3) float32."""
        arr = np.asarray(verts, dtype=np.float32).ravel()
        if len(arr) == 0:
            return np.zeros((0, 3, 3), dtype=np.float32)
        n = len(arr) // 9
        return arr[:n * 9].reshape(n, 3, 3)

    @staticmethod
    def _concat_meshes(meshes) -> np.ndarray:
        parts = []
        for m in meshes:
            arr = np.asarray(m, dtype=np.float32).ravel()
            if len(arr) >= 9:
                n = len(arr) // 9
                parts.append(arr[:n * 9].reshape(n, 3, 3))
        if not parts:
            return np.zeros((0, 3, 3), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    # ------------------------------------------------------------------ origin / seeding

    def set_station_origin(self, ox: float, oy: float) -> None:
        """Add a new BFS seed at (ox, oy) across all surfaces."""
        self._station_origin = np.array([ox, oy, 0.0], dtype=np.float32)
        self.floor.set_station_origin(self._station_origin)
        self.ceiling.set_station_origin(self._station_origin)
        for w in self.walls:
            w.set_station_origin(self._station_origin)
        for w in self.walls_ext:
            w.set_station_origin(self._station_origin)

    def plant_drop(self, pos: np.ndarray,
                   rng: Optional[np.random.Generator] = None
                   ) -> Optional[tuple]:
        """
        Seed BFS on the surface whose unfinished triangle is nearest to pos.

        This is the entry point for player material drops: the player aims at
        a point in 3-D space (hit point on any surface triangle, finished or
        not) and this method routes the seed to the correct surface.

        Returns (surface_key: str, surface: RoomSurface, tri_idx: int) or None
        if all surfaces are fully covered.
        """
        p = np.asarray(pos, np.float32)
        best_d = float('inf')
        best   = None

        candidates: List[tuple] = [
            ("floor", self.floor),
            ("ceiling", self.ceiling),
        ]
        for i, w in enumerate(self.walls):
            candidates.append((f"wall_{i}", w))

        for key, surf in candidates:
            unfinished = np.where(surf._material_id == -1)[0]
            if len(unfinished) == 0:
                continue
            cents = surf._centroids[unfinished]
            dists = np.sum((cents - p) ** 2, axis=1)
            d = float(dists.min())
            if d < best_d:
                best_d = d
                best   = (key, surf, unfinished[int(np.argmin(dists))])

        if best is None:
            return None

        key, surf, _ = best
        tri_idx = surf.add_seed(p, rng=rng)
        return (key, surf, tri_idx)

    # ------------------------------------------------------------------ fill

    def deliver_area(self, surface_key: str, material_id: int, area_m2: float) -> float:
        """Deliver area_m2 m² to the named surface.  Returns consumed area."""
        if surface_key == "floor":
            return self.floor.fill_batch(material_id, area_m2)
        if surface_key == "ceiling":
            return self.ceiling.fill_batch(material_id, area_m2)
        if surface_key.startswith("wall_"):
            idx = int(surface_key.split("_", 1)[1])
            if 0 <= idx < len(self.walls):
                c = self.walls[idx].fill_batch(material_id, area_m2)
                if idx < len(self.walls_ext):
                    self.walls_ext[idx].fill_batch(material_id, area_m2)
                return c
        if surface_key in ("walls_interior", "walls"):
            remaining = area_m2
            for i, w in enumerate(self.walls):
                consumed = w.fill_batch(material_id, remaining)
                if i < len(self.walls_ext):
                    self.walls_ext[i].fill_batch(material_id, consumed)
                remaining -= consumed
                if remaining <= 0.0:
                    break
            return area_m2 - remaining
        return 0.0

    def deliver_all(self, material_id: int, area_m2: float) -> float:
        """Deliver area_m2 m² across all surfaces.  Returns total consumed."""
        total = 0.0
        remaining = area_m2
        for surf in self._all_interior():
            if remaining <= 0.0:
                break
            c = surf.fill_batch(material_id, remaining)
            total += c
            remaining -= c
        return total

    def _all_interior(self):
        yield self.floor
        for w in self.walls:
            yield w
        yield self.ceiling

    # ------------------------------------------------------------------ state

    @property
    def is_finished(self) -> bool:
        return (self.floor.is_finished
                and self.ceiling.is_finished
                and all(w.is_finished for w in self.walls))

    @property
    def any_dirty(self) -> bool:
        return (self.floor.dirty
                or self.ceiling.dirty
                or any(w.dirty for w in self.walls)
                or any(w.dirty for w in self.walls_ext))

    def clear_all_dirty(self) -> None:
        self.floor.clear_dirty()
        self.ceiling.clear_dirty()
        for w in self.walls:
            w.clear_dirty()
        for w in self.walls_ext:
            w.clear_dirty()

    @property
    def covered_fraction(self) -> float:
        surfaces = list(self._all_interior())
        if not surfaces:
            return 1.0
        total_area = sum(s.total_area_m2 for s in surfaces)
        if total_area < 1e-9:
            return 1.0
        covered = sum(s.total_area_m2 * s.covered_fraction for s in surfaces)
        return min(1.0, covered / total_area)
