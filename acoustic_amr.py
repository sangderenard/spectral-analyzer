"""Adaptive multi-resolution acoustic pressure grid.

This module builds a true AMR pressure/velocity topology for the guitar body.
It is intentionally strict: construction validates the requested content
importance policy and raises on unsupported or inconsistent geometry instead
of falling back to the legacy uniform FDTD grid.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


AMR_AIR = np.uint8(0)
AMR_WALL = np.uint8(1)
AMR_PLATE = np.uint8(2)
AMR_PML = np.uint8(3)

AMR_LOW = np.uint8(0)
AMR_MESH_INTERSECTION = np.uint8(1)
AMR_GUITAR_INTERIOR = np.uint8(2)
AMR_APERTURE = np.uint8(3)

# ---------------------------------------------------------------------------
# Border condition spec
# ---------------------------------------------------------------------------

BORDER_ANECHOIC     = 0  # matches AMR_BORDER_ANECHOIC in acoustic_amr.h
BORDER_ROOM_PANEL   = 1  # matches AMR_BORDER_ROOM_PANEL
BORDER_TRANSMISSION = 2  # matches AMR_BORDER_TRANSMISSION


@dataclass
class BorderConditionSpec:
    """Parameters for AMR border absorption / transmission."""
    mode: int = BORDER_ANECHOIC
    sigma_order: float = 3.0    # polynomial grading exponent
    R_reflection: float = 0.0   # ROOM_PANEL residual reflected fraction
    Z_match: float = 0.0        # TRANSMISSION outer impedance; 0 = auto


def border_anechoic(sigma_order: float = 3.0) -> BorderConditionSpec:
    """Full anechoic PML — replicates uniform-FDTD PML absorption."""
    return BorderConditionSpec(mode=BORDER_ANECHOIC, sigma_order=sigma_order)


def border_room_panel(R_reflection: float, sigma_order: float = 2.0) -> BorderConditionSpec:
    """Partial absorber for room-panel / baffle boundary modelling."""
    return BorderConditionSpec(mode=BORDER_ROOM_PANEL, sigma_order=sigma_order,
                               R_reflection=float(R_reflection))


def border_transmission(Z_match: float = 0.0) -> BorderConditionSpec:
    """Transmission envelope — minimal reflection for nested-simulation coupling."""
    return BorderConditionSpec(mode=BORDER_TRANSMISSION, Z_match=float(Z_match))


@dataclass(frozen=True)
class AMRImportancePolicy:
    aperture: int
    guitar_interior: int
    mesh_intersections: int
    open_air: int = 0


@dataclass(frozen=True)
class AcousticAMRGrid:
    base_dx: float
    min_dx: float
    max_refinement_level: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    cell_centers: np.ndarray
    cell_half_sizes: np.ndarray
    cell_levels: np.ndarray
    cell_types: np.ndarray
    importance: np.ndarray
    cell_volumes: np.ndarray
    open_volume_fraction: np.ndarray
    face_cell_neg: np.ndarray
    face_cell_pos: np.ndarray
    face_axis: np.ndarray
    face_area: np.ndarray
    face_open_fraction: np.ndarray
    face_distance: np.ndarray
    soundhole: tuple[float, float, float]
    metadata: dict[str, Any]

    @property
    def n_cells(self) -> int:
        return int(self.cell_centers.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.face_cell_neg.shape[0])


class AcousticAMRFDTD:
    """Strict Python handle for the C++/Eigen AMR acoustic stepper."""

    def __init__(self, grid: AcousticAMRGrid, c: float = 343.0, rho_air: float = 1.21):
        try:
            from _spectral_kernels import AcousticAMR as _CAcousticAMR
        except ImportError as exc:
            raise RuntimeError(
                "_spectral_kernels.AcousticAMR is required for AMR stepping; "
                "rebuild the C extension. Python fallback stepping is forbidden."
            ) from exc
        self.grid = grid
        self.c = float(c)
        self.rho_air = float(rho_air)
        self._c = _CAcousticAMR(
            np.ascontiguousarray(grid.cell_centers, dtype=np.float64),
            np.ascontiguousarray(grid.cell_volumes, dtype=np.float64),
            np.ascontiguousarray(grid.open_volume_fraction, dtype=np.float64),
            np.ascontiguousarray(grid.cell_types, dtype=np.uint8),
            np.ascontiguousarray(grid.face_cell_neg, dtype=np.int32),
            np.ascontiguousarray(grid.face_cell_pos, dtype=np.int32),
            np.ascontiguousarray(grid.face_area, dtype=np.float64),
            np.ascontiguousarray(grid.face_open_fraction, dtype=np.float64),
            np.ascontiguousarray(grid.face_distance, dtype=np.float64),
            self.c,
            self.rho_air,
            float(grid.min_dx),
        )

    @property
    def dt(self) -> float:
        return float(self._c.dt)

    @property
    def pressure(self) -> np.ndarray:
        return self._c.get_pressure()

    @property
    def velocity(self) -> np.ndarray:
        return self._c.get_velocity()

    def reset(self) -> None:
        self._c.reset()

    def inject_pressure(self, xyz: np.ndarray, value: float) -> None:
        self._c.inject_pressure_nearest(np.asarray(xyz, dtype=np.float64).reshape(3), float(value))

    def step(self, n_steps: int = 1) -> None:
        self._c.step(int(n_steps))


def default_importance_policy(max_refinement_level: int) -> AMRImportancePolicy:
    max_level = int(max_refinement_level)
    if max_level < 1:
        raise ValueError("max_refinement_level must be >= 1")
    return AMRImportancePolicy(
        aperture=max_level,
        guitar_interior=max(1, max_level - 1),
        mesh_intersections=int(math.ceil(max_level / 2.0)),
        open_air=0,
    )


def build_guitar_amr_grid(
    outline_pts: np.ndarray,
    body_h: float,
    *,
    dx: float,
    pad_cells: int,
    n_pml: int,
    soundhole: tuple[float, float, float] | None,
    max_refinement_level: int,
    subdivision_k: int = 2,
    importance_policy: dict[str, int] | AMRImportancePolicy | None = None,
    balance_refinement: bool = True,
    aperture_band_cells: int = 2,
) -> AcousticAMRGrid:
    """Build a strict AMR grid for guitar-body acoustics.

    ``dx`` is the coarsest low-priority spacing. Refined cells use
    ``dx / subdivision_k**level``. Each base cell with level L is split into
    a uniform k×k×k grid at each of the L subdivision steps, yielding
    ``k**L`` child cells along each axis (``k**3L`` leaf cells total).
    ``subdivision_k=2`` is the classical octree; ``subdivision_k=3`` gives
    27-way splits per level; any integer >= 2 is valid.
    """
    base_dx = float(dx)
    if base_dx <= 0.0:
        raise ValueError("dx/base_dx must be positive")
    if int(pad_cells) <= int(n_pml):
        raise ValueError("pad_cells must exceed n_pml so free air exists before PML")
    if not balance_refinement:
        raise ValueError("balance_refinement=True is required for AMR construction")
    if soundhole is None:
        raise ValueError("AMR guitar grid requires explicit soundhole/aperture geometry")
    max_level = int(max_refinement_level)
    k = int(subdivision_k)
    if k < 2:
        raise ValueError("subdivision_k must be >= 2")
    policy = _coerce_policy(max_level, importance_policy)
    outline = np.asarray(outline_pts, dtype=np.float64)
    if outline.ndim != 2 or outline.shape[1] != 2 or len(outline) < 8:
        raise ValueError("outline_pts must be an (N,2) polygon with at least 8 points")

    cx, cy, hr = (float(soundhole[0]), float(soundhole[1]), float(soundhole[2]))
    if hr <= 0.0:
        raise ValueError("soundhole radius must be positive")
    body_h = float(body_h)
    if body_h <= 0.0:
        raise ValueError("body_h must be positive")

    ox = outline[:, 0]
    oy = outline[:, 1]
    pad = int(pad_cells) * base_dx
    bmin = np.array([ox.min() - pad, oy.min() - pad, -pad], dtype=np.float64)
    bmax = np.array([ox.max() + pad, oy.max() + pad, body_h + pad], dtype=np.float64)
    dims = np.maximum(4, np.ceil((bmax - bmin) / base_dx).astype(np.int32))
    bmax = bmin + dims.astype(np.float64) * base_dx

    levels = np.full(tuple(dims), policy.open_air, dtype=np.int16)
    importance = np.full(tuple(dims), AMR_LOW, dtype=np.uint8)
    base_type = np.full(tuple(dims), AMR_AIR, dtype=np.uint8)

    xs = bmin[0] + (np.arange(dims[0]) + 0.5) * base_dx
    ys = bmin[1] + (np.arange(dims[1]) + 0.5) * base_dx
    zs = bmin[2] + (np.arange(dims[2]) + 0.5) * base_dx
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    inside_xy = _pip_grid(X, Y, outline)

    soundhole_dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    in_soundhole = soundhole_dist < hr
    near_soundhole_rim = np.abs(soundhole_dist - hr) <= aperture_band_cells * base_dx

    for i in range(dims[0]):
        for j in range(dims[1]):
            inside = bool(inside_xy[i, j])
            hole = bool(in_soundhole[i, j])
            rim = bool(near_soundhole_rim[i, j])
            for k in range(dims[2]):
                z = float(zs[k])
                in_body_z = (-0.5 * base_dx <= z <= body_h + 0.5 * base_dx)
                near_top = abs(z - body_h) <= 0.75 * base_dx
                near_back = abs(z - 0.0) <= 0.75 * base_dx

                if _is_pml_base(i, j, k, dims, int(n_pml)):
                    base_type[i, j, k] = AMR_PML

                if in_body_z and not inside:
                    base_type[i, j, k] = AMR_WALL
                elif inside and near_top and not hole:
                    base_type[i, j, k] = AMR_PLATE
                elif inside and near_back:
                    base_type[i, j, k] = AMR_WALL

                if inside and 0.0 < z < body_h:
                    levels[i, j, k] = max(levels[i, j, k], policy.guitar_interior)
                    importance[i, j, k] = max(importance[i, j, k], AMR_GUITAR_INTERIOR)

                if _base_cell_crosses_outline(i, j, bmin, base_dx, outline) and in_body_z:
                    levels[i, j, k] = max(levels[i, j, k], policy.mesh_intersections)
                    importance[i, j, k] = max(importance[i, j, k], AMR_MESH_INTERSECTION)

                if inside and (near_top or near_back):
                    levels[i, j, k] = max(levels[i, j, k], policy.mesh_intersections)
                    importance[i, j, k] = max(importance[i, j, k], AMR_MESH_INTERSECTION)

                if inside and (hole or rim) and abs(z - body_h) <= (aperture_band_cells + 1) * base_dx:
                    levels[i, j, k] = max(levels[i, j, k], policy.aperture)
                    importance[i, j, k] = AMR_APERTURE
                    if hole and near_top:
                        base_type[i, j, k] = AMR_AIR

    if not np.any(importance == AMR_APERTURE):
        raise ValueError("AMR classification produced no aperture cells")
    if not np.any(importance == AMR_GUITAR_INTERIOR):
        raise ValueError("AMR classification produced no guitar interior cells")

    _balance_base_levels(levels)

    cells = _subdivide_base_cells(
        levels, importance, base_type, bmin, base_dx, outline, body_h, soundhole, int(n_pml), dims,
        subdivision_k=k,
    )
    faces = _build_faces(cells)
    grid = AcousticAMRGrid(
        base_dx=base_dx,
        min_dx=base_dx / float(k ** max_level),
        max_refinement_level=max_level,
        bounds_min=bmin,
        bounds_max=bmax,
        cell_centers=cells["centers"],
        cell_half_sizes=cells["half_sizes"],
        cell_levels=cells["levels"],
        cell_types=cells["types"],
        importance=cells["importance"],
        cell_volumes=cells["volumes"],
        open_volume_fraction=cells["open_fraction"],
        face_cell_neg=faces["neg"],
        face_cell_pos=faces["pos"],
        face_axis=faces["axis"],
        face_area=faces["area"],
        face_open_fraction=faces["open_fraction"],
        face_distance=faces["distance"],
        soundhole=(cx, cy, hr),
        metadata={
            "base_dims": dims,
            "base_levels": levels,
            "base_importance": importance,
            "importance_policy": policy,
            "pad_cells": int(pad_cells),
            "n_pml": int(n_pml),
        },
    )
    validate_amr_grid(grid, policy)
    return grid


def validate_amr_grid(grid: AcousticAMRGrid, policy: AMRImportancePolicy | None = None) -> None:
    if grid.n_cells <= 0:
        raise ValueError("AMR grid has no cells")
    if grid.n_faces <= 0:
        raise ValueError("AMR grid has no faces")
    if not np.all(grid.cell_volumes > 0.0):
        raise ValueError("AMR grid has non-positive cell volumes")
    if not np.all(grid.face_area > 0.0):
        raise ValueError("AMR grid has non-positive face areas")
    if not np.all(grid.face_distance > 0.0):
        raise ValueError("AMR grid has non-positive face distances")
    if np.any(np.abs(grid.cell_levels[grid.face_cell_neg] - grid.cell_levels[grid.face_cell_pos]) > 1):
        raise ValueError("AMR grid has refinement jumps greater than one level")
    if policy is not None:
        if np.any(grid.cell_levels[grid.importance == AMR_APERTURE] != policy.aperture):
            raise ValueError("aperture cells are not refined to max importance level")
        if np.any(grid.cell_levels[grid.importance == AMR_GUITAR_INTERIOR] < policy.guitar_interior):
            raise ValueError("guitar interior cells are below required refinement")
        if np.any(grid.cell_levels[grid.importance == AMR_MESH_INTERSECTION] < policy.mesh_intersections):
            raise ValueError("mesh intersection cells are below required refinement")
    _validate_aperture_open(grid)


def _coerce_policy(max_level: int, raw: dict[str, int] | AMRImportancePolicy | None) -> AMRImportancePolicy:
    if raw is None:
        return default_importance_policy(max_level)
    if isinstance(raw, AMRImportancePolicy):
        policy = raw
    else:
        defaults = default_importance_policy(max_level)
        policy = AMRImportancePolicy(
            aperture=int(raw.get("aperture", defaults.aperture)),
            guitar_interior=int(raw.get("guitar_interior", defaults.guitar_interior)),
            mesh_intersections=int(raw.get("mesh_intersections", defaults.mesh_intersections)),
            open_air=int(raw.get("open_air", defaults.open_air)),
        )
    vals = [policy.aperture, policy.guitar_interior, policy.mesh_intersections, policy.open_air]
    if min(vals) < 0 or max(vals) > max_level:
        raise ValueError("importance_policy levels must be within [0, max_refinement_level]")
    if policy.aperture != max_level:
        raise ValueError("importance_policy.aperture must equal max_refinement_level")
    return policy


def _is_pml_base(i: int, j: int, k: int, dims: np.ndarray, n_pml: int) -> bool:
    return (
        i < n_pml or j < n_pml or k < n_pml
        or i >= dims[0] - n_pml or j >= dims[1] - n_pml or k >= dims[2] - n_pml
    )


def _pip_grid(X: np.ndarray, Y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    inside = np.zeros(X.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        inside ^= ((yi > Y) != (yj > Y)) & (
            X < (xj - xi) * (Y - yi) / (yj - yi + 1e-15) + xi
        )
        j = i
    return inside


def _pip_point(x: float, y: float, poly: np.ndarray) -> bool:
    return bool(_pip_grid(np.array([[x]], dtype=np.float64), np.array([[y]], dtype=np.float64), poly)[0, 0])


def _base_cell_crosses_outline(i: int, j: int, bmin: np.ndarray, dx: float, outline: np.ndarray) -> bool:
    x0 = bmin[0] + i * dx
    y0 = bmin[1] + j * dx
    pts = [(x0, y0), (x0 + dx, y0), (x0, y0 + dx), (x0 + dx, y0 + dx)]
    vals = [_pip_point(x, y, outline) for x, y in pts]
    return any(vals) and not all(vals)


def _balance_base_levels(levels: np.ndarray) -> None:
    dims = levels.shape
    changed = True
    while changed:
        changed = False
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    lv = int(levels[i, j, k])
                    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < dims[0] and 0 <= nj < dims[1] and 0 <= nk < dims[2]:
                            nlv = int(levels[ni, nj, nk])
                            if lv > nlv + 1:
                                levels[ni, nj, nk] = lv - 1
                                changed = True


def _subdivide_base_cells(
    levels: np.ndarray,
    importance: np.ndarray,
    base_type: np.ndarray,
    bmin: np.ndarray,
    base_dx: float,
    outline: np.ndarray,
    body_h: float,
    soundhole: tuple[float, float, float],
    n_pml: int,
    dims: np.ndarray,
    *,
    subdivision_k: int = 2,
) -> dict[str, np.ndarray]:
    """Split every base cell into a subdivision_k×subdivision_k×subdivision_k
    uniform grid at each of its assigned refinement levels.  A base cell at
    level L produces n = subdivision_k**L leaf cells per axis.
    """
    centers = []
    half_sizes = []
    leaf_levels = []
    types = []
    imps = []
    volumes = []
    open_fracs = []
    cx, cy, hr = soundhole

    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                lv = int(levels[i, j, k])
                n = subdivision_k ** lv
                child_dx = base_dx / n
                for a in range(n):
                    for b in range(n):
                        for c in range(n):
                            lo = bmin + np.array([
                                i * base_dx + a * child_dx,
                                j * base_dx + b * child_dx,
                                k * base_dx + c * child_dx,
                            ], dtype=np.float64)
                            center = lo + 0.5 * child_dx
                            ctype = _classify_leaf_type(center, child_dx, outline, body_h, soundhole)
                            if ctype == AMR_AIR and _is_pml_base(i, j, k, dims, n_pml):
                                ctype = AMR_PML
                            centers.append(center)
                            half_sizes.append(np.array([0.5 * child_dx] * 3, dtype=np.float64))
                            leaf_levels.append(lv)
                            types.append(ctype)
                            imps.append(int(importance[i, j, k]))
                            volumes.append(child_dx ** 3)
                            open_fracs.append(0.0 if ctype in (AMR_WALL, AMR_PLATE) else 1.0)

    return {
        "centers": np.asarray(centers, dtype=np.float64),
        "half_sizes": np.asarray(half_sizes, dtype=np.float64),
        "levels": np.asarray(leaf_levels, dtype=np.int16),
        "types": np.asarray(types, dtype=np.uint8),
        "importance": np.asarray(imps, dtype=np.uint8),
        "volumes": np.asarray(volumes, dtype=np.float64),
        "open_fraction": np.asarray(open_fracs, dtype=np.float64),
    }


def _classify_leaf_type(
    center: np.ndarray,
    dx: float,
    outline: np.ndarray,
    body_h: float,
    soundhole: tuple[float, float, float],
) -> np.uint8:
    x, y, z = float(center[0]), float(center[1]), float(center[2])
    inside = _pip_point(x, y, outline)
    cx, cy, hr = soundhole
    hole = (x - cx) ** 2 + (y - cy) ** 2 < hr ** 2
    in_body_z = -0.5 * dx <= z <= body_h + 0.5 * dx
    near_top = abs(z - body_h) <= 0.5 * dx
    near_back = abs(z - 0.0) <= 0.5 * dx
    if in_body_z and not inside:
        return AMR_WALL
    if inside and near_top and not hole:
        return AMR_PLATE
    if inside and near_back:
        return AMR_WALL
    return AMR_AIR


def _build_faces(cells: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    centers = cells["centers"]
    half = cells["half_sizes"]
    types = cells["types"]
    n = len(centers)
    neg = []
    pos = []
    axes = []
    areas = []
    open_fracs = []
    distances = []
    tol = 1e-9
    lo = centers - half
    hi = centers + half

    def plane_key(value: float) -> int:
        return int(round(value / tol))

    for axis in range(3):
        other = [q for q in range(3) if q != axis]
        high_faces: dict[int, list[int]] = {}
        low_faces: dict[int, list[int]] = {}
        for idx in range(n):
            high_faces.setdefault(plane_key(float(hi[idx, axis])), []).append(idx)
            low_faces.setdefault(plane_key(float(lo[idx, axis])), []).append(idx)
        for key, left_candidates in high_faces.items():
            right_candidates = low_faces.get(key)
            if not right_candidates:
                continue
            for left in left_candidates:
                for right in right_candidates:
                    if left == right:
                        continue
                    ov0 = max(lo[left, other[0]], lo[right, other[0]])
                    ov1 = min(hi[left, other[0]], hi[right, other[0]])
                    ov2 = max(lo[left, other[1]], lo[right, other[1]])
                    ov3 = min(hi[left, other[1]], hi[right, other[1]])
                    if ov1 <= ov0 + tol or ov3 <= ov2 + tol:
                        continue
                    area = (ov1 - ov0) * (ov3 - ov2)
                    neg.append(left)
                    pos.append(right)
                    axes.append(axis)
                    areas.append(area)
                    # PLATE-AIR faces are moving boundaries: keep open so the
                    # velocity BC set by amr_apply_plate_bc reaches the divergence.
                    # Only rigid WALL faces block acoustic flux.
                    solid = types[left] == AMR_WALL or types[right] == AMR_WALL
                    open_fracs.append(0.0 if solid else 1.0)
                    distances.append(abs(centers[right, axis] - centers[left, axis]))
    if not neg:
        raise ValueError("AMR topology produced no cell faces")
    return {
        "neg": np.asarray(neg, dtype=np.int32),
        "pos": np.asarray(pos, dtype=np.int32),
        "axis": np.asarray(axes, dtype=np.uint8),
        "area": np.asarray(areas, dtype=np.float64),
        "open_fraction": np.asarray(open_fracs, dtype=np.float64),
        "distance": np.asarray(distances, dtype=np.float64),
    }


def _validate_aperture_open(grid: AcousticAMRGrid) -> None:
    cx, cy, hr = grid.soundhole
    centers = grid.cell_centers
    in_hole = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2 < hr ** 2
    open_cells = in_hole & (
        ((grid.cell_types == AMR_AIR) | (grid.cell_types == AMR_PML))
        & (grid.open_volume_fraction > 0.0)
    )
    if not np.any(open_cells):
        raise ValueError("soundhole/aperture is sealed: no open AMR cells inside aperture")
    zvals = centers[open_cells, 2]
    body_top = float(grid.bounds_max[2] - grid.metadata["pad_cells"] * grid.base_dx)
    if not (np.any(zvals < body_top) and np.any(zvals > body_top)):
        raise ValueError("soundhole/aperture does not connect interior and exterior air")


# ── Mic importance cone projection ───────────────────────────────────────────

def project_mic_importance_cone(
    grid: AcousticAMRGrid,
    mic_pos_xyz: np.ndarray,
    face_perimeter_pts: np.ndarray,
    intermediate_levels: int = 2,
) -> AcousticAMRGrid:
    """Project a conic region from the mic to the aperture face perimeter.

    Bumps cells that fall within the cone to intermediate importance levels so
    the direct acoustic path between mic and soundhole is resolved more finely
    than the surrounding body interior, but not as finely as the aperture itself.
    Multiple degrees of subdivision are produced: cells near the cone axis get
    a higher intermediate level than cells near the cone rim.

    Parameters
    ----------
    grid:
        Existing AcousticAMRGrid to annotate (importance array is updated).
    mic_pos_xyz:
        World-space mic position (3,) float array in metres.
    face_perimeter_pts:
        (M, 3) points sampling the aperture face perimeter.  Rays are traced
        from mic_pos_xyz to each of these points to define the cone boundary.
    intermediate_levels:
        Number of distinct importance levels between lowest and aperture level.
        Must be >= 1 and < grid.max_refinement_level.  Cells along the cone
        axis get level ``aperture_level - 1``; cells at the rim get level
        ``aperture_level - intermediate_levels``.

    Returns
    -------
    AcousticAMRGrid with updated importance and cell_levels arrays.  The
    soundhole aperture cells retain their original (highest) importance level.
    All other grid state is unchanged.
    """
    mic = np.asarray(mic_pos_xyz, dtype=np.float64).reshape(3)
    perimeter = np.asarray(face_perimeter_pts, dtype=np.float64)
    if perimeter.ndim != 2 or perimeter.shape[1] != 3 or len(perimeter) < 3:
        raise ValueError("face_perimeter_pts must be (M>=3, 3)")
    max_level = int(grid.max_refinement_level)
    n_levels = int(intermediate_levels)
    if n_levels < 1 or n_levels >= max_level:
        raise ValueError(
            f"intermediate_levels must be in [1, max_refinement_level-1], got {n_levels}"
        )

    # Geometric aperture centre and radius used to define the cone axis.
    aperture_center = perimeter.mean(axis=0)
    rim_vecs = perimeter - aperture_center  # (M, 3) from centre to each rim point
    max_rim_dist = float(np.linalg.norm(rim_vecs, axis=1).max())
    if max_rim_dist <= 0.0:
        raise ValueError("face_perimeter_pts are all coincident — cannot form a cone")

    # Vector from mic to aperture centre (cone axis direction).
    axis = aperture_center - mic
    axis_len = float(np.linalg.norm(axis))
    if axis_len <= 0.0:
        raise ValueError("mic position coincides with aperture centre — degenerate cone")
    axis_unit = axis / axis_len

    # For each AMR cell, compute its position relative to the mic and project
    # onto the cone axis and the transverse plane to determine whether it lies
    # within the cone and how close to the axis it is.
    centers = grid.cell_centers  # (N, 3)
    to_cell = centers - mic[np.newaxis, :]  # (N, 3)
    proj = to_cell @ axis_unit  # (N,) signed distance along cone axis

    # Only consider cells that are between the mic and the aperture (+/- one
    # cell radius buffer) along the cone axis.
    cell_r = float(grid.min_dx) * 2.0  # conservative cell radius for buffer
    in_range = (proj >= -cell_r) & (proj <= axis_len + cell_r)

    # Transverse offset from the projected cone at each cell's axial position.
    # The cone radius linearly expands from 0 at the mic to max_rim_dist at the
    # aperture centre.  We use the ratio t = proj/axis_len to interpolate.
    t = np.clip(proj / axis_len, 0.0, 1.0)
    cone_radius_at_t = t * max_rim_dist
    projected_pt = mic[np.newaxis, :] + proj[:, np.newaxis] * axis_unit[np.newaxis, :]
    transverse_dist = np.linalg.norm(centers - projected_pt, axis=1)

    # Importance levels for rim and axis:
    #   rim_level  = max_level - n_levels         (minimum intermediate)
    #   axis_level = max_level - 1                (maximum intermediate, just below aperture)
    # Cells outside the cone are unchanged.
    # Within the cone, level is linearly interpolated based on how close to the
    # axis the cell is, producing n_levels distinct integer steps.
    rim_level = max_level - n_levels
    axis_level = max_level - 1

    # Fraction: 0 at rim, 1 at axis.
    # We clip so that outside-cone cells aren't accidentally bumped.
    safe_cone_r = np.where(cone_radius_at_t > 0.0, cone_radius_at_t, 1.0)
    axis_frac = np.clip(1.0 - transverse_dist / safe_cone_r, 0.0, 1.0)
    in_cone = in_range & (transverse_dist <= cone_radius_at_t + cell_r)

    # Compute target intermediate level for each cell in the cone.
    target_level = np.floor(rim_level + axis_frac * (axis_level - rim_level + 1)).astype(np.int16)
    target_level = np.clip(target_level, rim_level, axis_level)

    # Importance value for cone cells: one notch below aperture.
    # We do NOT override aperture-importance cells.
    new_importance = grid.importance.copy()
    new_levels = grid.cell_levels.copy()
    is_aperture = grid.importance == AMR_APERTURE

    cone_and_not_aperture = in_cone & ~is_aperture
    # Only bump up, never down.
    new_levels[cone_and_not_aperture] = np.maximum(
        new_levels[cone_and_not_aperture],
        target_level[cone_and_not_aperture],
    )
    # Importance: mark as GUITAR_INTERIOR-equivalent if currently lower.
    cone_low = cone_and_not_aperture & (new_importance < AMR_GUITAR_INTERIOR)
    new_importance[cone_low] = AMR_GUITAR_INTERIOR

    # Re-balance: enforce no level jump > 1 between neighbours using a fast
    # vectorised pass on the existing face connectivity.
    new_levels = _rebalance_cell_levels(new_levels, grid.face_cell_neg, grid.face_cell_pos)

    # Return a new grid with updated arrays (all other fields identical).
    return AcousticAMRGrid(
        base_dx=grid.base_dx,
        min_dx=grid.min_dx,
        max_refinement_level=grid.max_refinement_level,
        bounds_min=grid.bounds_min,
        bounds_max=grid.bounds_max,
        cell_centers=grid.cell_centers,
        cell_half_sizes=grid.cell_half_sizes,
        cell_levels=new_levels,
        cell_types=grid.cell_types,
        importance=new_importance,
        cell_volumes=grid.cell_volumes,
        open_volume_fraction=grid.open_volume_fraction,
        face_cell_neg=grid.face_cell_neg,
        face_cell_pos=grid.face_cell_pos,
        face_axis=grid.face_axis,
        face_area=grid.face_area,
        face_open_fraction=grid.face_open_fraction,
        face_distance=grid.face_distance,
        soundhole=grid.soundhole,
        metadata=grid.metadata,
    )


def _rebalance_cell_levels(
    levels: np.ndarray,
    face_neg: np.ndarray,
    face_pos: np.ndarray,
) -> np.ndarray:
    """Enforce no refinement-level jump > 1 across each face.

    Iterates until convergence (at most max_level passes for a balanced grid).
    Mutates and returns the input array.
    """
    levels = levels.copy()
    for _ in range(int(levels.max()) + 1):
        lneg = levels[face_neg]
        lpos = levels[face_pos]
        # Cells that need bumping: neighbour is 2+ levels higher.
        bump_neg = face_neg[lpos > lneg + 1]
        bump_pos = face_pos[lneg > lpos + 1]
        if len(bump_neg) == 0 and len(bump_pos) == 0:
            break
        np.maximum.at(levels, bump_neg, levels[face_pos[lpos > lneg + 1]] - 1)
        np.maximum.at(levels, bump_pos, levels[face_neg[lneg > lpos + 1]] - 1)
    return levels


# ── Plate-to-AMR mapping builders ────────────────────────────────────────────

def build_plate_amr_mappings(
    grid: AcousticAMRGrid,
    plate_Nx: int,
    plate_Ny: int,
    plate_dx: float,
    plate_origin: np.ndarray,
    body_h: float,
) -> dict[str, np.ndarray]:
    """Build ragged CSR plate-to-AMR face and cell mappings.

    For each active plate node (those whose cell type is AMR_PLATE in the grid)
    this function identifies the AMR faces immediately above (interior cavity)
    and below (exterior/back) the plate node, plus the single AMR cell on each
    side, and encodes them as ragged CSR arrays suitable for ``amr_setup_plate``.

    Parameters
    ----------
    grid : AcousticAMRGrid
    plate_Nx, plate_Ny : int — plate grid dimensions.
    plate_dx : float — uniform plate node spacing (metres).
    plate_origin : (3,) — world position of plate node (0,0).
    body_h : float — body height above back plate (metres); plate is at z=body_h.

    Returns
    -------
    dict with keys:
        plate_active          : uint8 (plate_Nx*plate_Ny,)
        n_active_plate        : int
        plate_active_flat_idx : int32 (n_active,)
        plate_face_above_starts, plate_face_above_idx, plate_face_above_wgt
        plate_face_below_starts, plate_face_below_idx, plate_face_below_wgt
        plate_cell_above      : int32 (n_active,)
        plate_cell_below      : int32 (n_active,)
    """
    origin = np.asarray(plate_origin, dtype=np.float64).reshape(3)
    plate_Nx = int(plate_Nx)
    plate_Ny = int(plate_Ny)
    plate_dx = float(plate_dx)
    body_h = float(body_h)

    centers = grid.cell_centers  # (N_cells, 3)
    cell_types = grid.cell_types
    face_neg = grid.face_cell_neg
    face_pos = grid.face_cell_pos
    face_axis = grid.face_axis
    face_area = grid.face_area
    face_open = grid.face_open_fraction

    # Plate node world-space XY positions.
    is_x = np.arange(plate_Nx, dtype=np.float64)
    is_y = np.arange(plate_Ny, dtype=np.float64)
    node_x = origin[0] + is_x * plate_dx  # (plate_Nx,)
    node_y = origin[1] + is_y * plate_dx  # (plate_Ny,)
    plate_z = body_h

    # Identify all AMR cells at or near the plate (z ≈ body_h, cell type PLATE or AIR just above it).
    # "Above" = interior cavity side (z slightly above body_h).
    # "Below" = exterior side (z slightly below body_h).
    cell_z = centers[:, 2]
    near_plate = np.abs(cell_z - plate_z) < grid.base_dx * 2.0
    plate_cell_mask = (cell_types == AMR_PLATE) & near_plate
    air_above_mask = (cell_types == AMR_AIR) & (cell_z > plate_z) & near_plate
    air_below_mask = (cell_types == AMR_AIR) & (cell_z < plate_z) & near_plate

    # Build active node list: a node is active if at least one AMR PLATE cell
    # is within half a plate_dx of its XY position.
    plate_active_flat = np.zeros(plate_Nx * plate_Ny, dtype=np.uint8)
    plate_cell_x = centers[plate_cell_mask, 0]
    plate_cell_y = centers[plate_cell_mask, 1]
    plate_cell_indices = np.where(plate_cell_mask)[0]

    for flat in range(plate_Nx * plate_Ny):
        i = flat // plate_Ny
        j = flat % plate_Ny
        px = float(node_x[i])
        py = float(node_y[j])
        dx2 = (plate_cell_x - px) ** 2 + (plate_cell_y - py) ** 2
        if dx2.size > 0 and float(dx2.min()) < (plate_dx * 0.75) ** 2:
            plate_active_flat[flat] = 1

    active_flat_idx = np.where(plate_active_flat)[0].astype(np.int32)
    n_active = int(len(active_flat_idx))
    if n_active == 0:
        raise ValueError(
            "build_plate_amr_mappings: no active plate nodes found — "
            "check plate_origin/plate_dx against AMR grid bounds"
        )

    # For each face, determine if it is a z-axis face (axis==2) between a
    # PLATE cell and an AIR cell above or below.
    z_faces = face_axis == 2
    z_face_idx = np.where(z_faces)[0]
    neg_t = cell_types[face_neg[z_faces]]
    pos_t = cell_types[face_pos[z_faces]]
    neg_z = centers[face_neg[z_faces], 2]
    pos_z = centers[face_pos[z_faces], 2]

    # Face is "above" when: neg=PLATE cell (lower z) and pos=AIR cell (higher z).
    face_above_mask = (neg_t == AMR_PLATE) & (pos_t == AMR_AIR) & (pos_z > neg_z)
    # Face is "below" when: neg=AIR cell (lower z) and pos=PLATE cell (higher z).
    face_below_mask = (neg_t == AMR_AIR) & (pos_t == AMR_PLATE) & (neg_z < pos_z)

    above_global = z_face_idx[face_above_mask]  # global face indices for above-plate faces
    below_global = z_face_idx[face_below_mask]

    # XY centre of each above/below face = XY of the PLATE cell in that face.
    above_plate_cell = face_neg[above_global]  # PLATE cells are on neg side for above
    below_plate_cell = face_pos[below_global]  # PLATE cells are on pos side for below
    above_face_x = centers[above_plate_cell, 0]
    above_face_y = centers[above_plate_cell, 1]
    below_face_x = centers[below_plate_cell, 0]
    below_face_y = centers[below_plate_cell, 1]

    above_air_cell = face_pos[above_global]   # AIR cell above plate face
    below_air_cell = face_neg[below_global]   # AIR cell below plate face

    # Build CSR arrays per active node.
    face_above_starts = np.zeros(n_active + 1, dtype=np.int32)
    face_below_starts = np.zeros(n_active + 1, dtype=np.int32)
    face_above_idx_list: list[int] = []
    face_above_wgt_list: list[float] = []
    face_below_idx_list: list[int] = []
    face_below_wgt_list: list[float] = []
    cell_above = np.full(n_active, -1, dtype=np.int32)
    cell_below = np.full(n_active, -1, dtype=np.int32)

    for n_idx, flat in enumerate(active_flat_idx):
        i = int(flat) // plate_Ny
        j = int(flat) % plate_Ny
        px = float(node_x[i])
        py = float(node_y[j])
        r_thresh = float(plate_dx)

        # Above faces within r_thresh of this node.
        d2_above = (above_face_x - px) ** 2 + (above_face_y - py) ** 2
        ab_near = np.where(d2_above < r_thresh ** 2)[0]
        if ab_near.size > 0:
            wgt_raw = face_area[above_global[ab_near]]
            wgt_sum = float(wgt_raw.sum())
            wgt_norm = (wgt_raw / wgt_sum).astype(np.float32) if wgt_sum > 0.0 else np.ones(ab_near.size, dtype=np.float32) / ab_near.size
            face_above_idx_list.extend(above_global[ab_near].tolist())
            face_above_wgt_list.extend(wgt_norm.tolist())
            # Cell above: pick the AIR cell with the nearest face to node XY.
            best = int(ab_near[int(np.argmin(d2_above[ab_near]))])
            cell_above[n_idx] = int(above_air_cell[best])
        face_above_starts[n_idx + 1] = len(face_above_idx_list)

        # Below faces.
        d2_below = (below_face_x - px) ** 2 + (below_face_y - py) ** 2
        bl_near = np.where(d2_below < r_thresh ** 2)[0]
        if bl_near.size > 0:
            wgt_raw = face_area[below_global[bl_near]]
            wgt_sum = float(wgt_raw.sum())
            wgt_norm = (wgt_raw / wgt_sum).astype(np.float32) if wgt_sum > 0.0 else np.ones(bl_near.size, dtype=np.float32) / bl_near.size
            face_below_idx_list.extend(below_global[bl_near].tolist())
            face_below_wgt_list.extend(wgt_norm.tolist())
            best = int(bl_near[int(np.argmin(d2_below[bl_near]))])
            cell_below[n_idx] = int(below_air_cell[best])
        face_below_starts[n_idx + 1] = len(face_below_idx_list)

    return {
        "plate_active": plate_active_flat,
        "n_active_plate": n_active,
        "plate_active_flat_idx": active_flat_idx,
        "plate_face_above_starts": face_above_starts,
        "plate_face_above_idx": np.asarray(face_above_idx_list, dtype=np.int32),
        "plate_face_above_wgt": np.asarray(face_above_wgt_list, dtype=np.float32),
        "plate_face_below_starts": face_below_starts,
        "plate_face_below_idx": np.asarray(face_below_idx_list, dtype=np.int32),
        "plate_face_below_wgt": np.asarray(face_below_wgt_list, dtype=np.float32),
        "plate_cell_above": cell_above,
        "plate_cell_below": cell_below,
    }


def build_bridge_plate_mapping(
    bridge_src_xyz: np.ndarray,
    plate_Nx: int,
    plate_Ny: int,
    plate_dx: float,
    plate_origin: np.ndarray,
    sigma: float | None = None,
) -> dict[str, np.ndarray]:
    """Build Gaussian-weighted bridge-to-plate source mapping.

    Parameters
    ----------
    bridge_src_xyz : (3,) or (K,3) world positions of bridge saddle source points.
    plate_Nx, plate_Ny : plate grid dimensions.
    plate_dx : uniform plate spacing (metres).
    plate_origin : (3,) world position of plate node (0,0).
    sigma : Gaussian sigma in metres.  Defaults to 2 * plate_dx.

    Returns
    -------
    dict with keys:
        n_bridge_plate : int
        bridge_plate_idx : int32 (n_bridge_plate,) flat plate indices (j + Ny*i)
        bridge_plate_wgt : float32 (n_bridge_plate,) normalized weights
    """
    origin = np.asarray(plate_origin, dtype=np.float64).reshape(3)
    pts = np.asarray(bridge_src_xyz, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    if sigma is None:
        sigma = 2.0 * float(plate_dx)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    plate_Nx = int(plate_Nx)
    plate_Ny = int(plate_Ny)
    plate_dx = float(plate_dx)

    # Gaussian threshold: include nodes within 3 sigma.
    thresh = 3.0 * sigma
    node_flat_idx = []
    node_wgt = []

    for flat in range(plate_Nx * plate_Ny):
        i = flat // plate_Ny
        j = flat % plate_Ny
        nx = origin[0] + i * plate_dx
        ny = origin[1] + j * plate_dx
        # Sum Gaussian weights from all source points.
        w = 0.0
        for pt in pts:
            d2 = (nx - float(pt[0])) ** 2 + (ny - float(pt[1])) ** 2
            if d2 < thresh ** 2:
                w += math.exp(-0.5 * d2 / (sigma ** 2))
        if w > 1e-9:
            node_flat_idx.append(flat)
            node_wgt.append(w)

    if len(node_flat_idx) == 0:
        raise ValueError(
            "build_bridge_plate_mapping: no plate nodes within 3-sigma of bridge source — "
            "check bridge_src_xyz vs plate_origin/plate_dx"
        )

    wgt_arr = np.asarray(node_wgt, dtype=np.float32)
    wgt_arr /= float(wgt_arr.sum())
    return {
        "n_bridge_plate": len(node_flat_idx),
        "bridge_plate_idx": np.asarray(node_flat_idx, dtype=np.int32),
        "bridge_plate_wgt": wgt_arr,
    }


def build_amr_coevolver_descriptor(
    grid: AcousticAMRGrid,
    plate_Nx: int,
    plate_Ny: int,
    plate_dx: float,
    plate_origin: np.ndarray,
    body_h: float,
    plate_mass_density: float,
    plate_stiffness_D: float,
    plate_alpha_M: float,
    plate_beta_K: float,
    bridge_src_xyz: np.ndarray,
    bridge_sigma: float | None = None,
    c: float = 343.0,
    rho_air: float = 1.21,
    neck_plate_idx: np.ndarray | None = None,
    neck_plate_wgt: np.ndarray | None = None,
    border_spec: 'BorderConditionSpec | None' = None,
    n_pml: int = 0,
) -> dict[str, object]:
    """Build a complete AMRCoevolverDescriptor-compatible dict.

    All numpy arrays in the returned dict match the dtypes and shapes expected
    by the C ``AMRCoevolverDescriptor`` struct so they can be passed directly
    to a ctypes/cffi wrapper or to ``coevolver_create_amr`` via pybind11.

    Parameters
    ----------
    grid : AcousticAMRGrid
    plate_Nx, plate_Ny : plate grid dimensions.
    plate_dx : plate node spacing (metres).
    plate_origin : (3,) world position of plate corner node.
    body_h : body height / plate z-position (metres).
    plate_mass_density : rho_s * h (kg/m²).
    plate_stiffness_D : bending stiffness D = E*h³/(12*(1-nu²)) (N*m).
    plate_alpha_M : Rayleigh mass-proportional damping coefficient (s⁻¹).
    plate_beta_K : Rayleigh stiffness-proportional damping coefficient (s).
    bridge_src_xyz : (3,) or (K,3) bridge saddle source world positions.
    bridge_sigma : Gaussian sigma for bridge mapping; defaults to 2*plate_dx.
    c : speed of sound (m/s).
    rho_air : air density (kg/m³).
    neck_plate_idx : optional flat plate indices for neck coupling.
    neck_plate_wgt : optional weights for neck coupling (must match idx length).

    Returns
    -------
    dict with all fields matching AMRCoevolverDescriptor field names.
    """
    plate_origin_arr = np.asarray(plate_origin, dtype=np.float32).reshape(3)

    plate_maps = build_plate_amr_mappings(
        grid, plate_Nx, plate_Ny, plate_dx, plate_origin_arr, body_h
    )
    bridge_map = build_bridge_plate_mapping(
        bridge_src_xyz, plate_Nx, plate_Ny, plate_dx, plate_origin_arr, bridge_sigma
    )

    cx, cy, hr = grid.soundhole

    if neck_plate_idx is not None:
        neck_idx = np.asarray(neck_plate_idx, dtype=np.int32).ravel()
        neck_wgt = np.asarray(neck_plate_wgt, dtype=np.float32).ravel()
        if len(neck_idx) != len(neck_wgt):
            raise ValueError("neck_plate_idx and neck_plate_wgt must have the same length")
        n_neck = int(len(neck_idx))
    else:
        neck_idx = np.empty(0, dtype=np.int32)
        neck_wgt = np.empty(0, dtype=np.float32)
        n_neck = 0

    desc = {
        # AMR grid
        "n_cells": int(grid.n_cells),
        "cell_centers": np.ascontiguousarray(grid.cell_centers, dtype=np.float64),
        "cell_volumes": np.ascontiguousarray(grid.cell_volumes, dtype=np.float64),
        "open_volume_frac": np.ascontiguousarray(grid.open_volume_fraction, dtype=np.float64),
        "cell_types": np.ascontiguousarray(grid.cell_types, dtype=np.uint8),
        "cell_levels": np.ascontiguousarray(grid.cell_levels, dtype=np.int16),
        "n_faces": int(grid.n_faces),
        "face_cell_neg": np.ascontiguousarray(grid.face_cell_neg, dtype=np.int32),
        "face_cell_pos": np.ascontiguousarray(grid.face_cell_pos, dtype=np.int32),
        "face_area": np.ascontiguousarray(grid.face_area, dtype=np.float64),
        "face_open_frac": np.ascontiguousarray(grid.face_open_fraction, dtype=np.float64),
        "face_distance": np.ascontiguousarray(grid.face_distance, dtype=np.float64),
        "c": float(c),
        "rho_air": float(rho_air),
        "min_dx": float(grid.min_dx),
        "bounds_min": np.ascontiguousarray(grid.bounds_min, dtype=np.float64),
        "bounds_max": np.ascontiguousarray(grid.bounds_max, dtype=np.float64),
        # Plate
        "plate_Nx": int(plate_Nx),
        "plate_Ny": int(plate_Ny),
        "plate_dx": float(plate_dx),
        "plate_origin": plate_origin_arr,
        "plate_active": np.ascontiguousarray(plate_maps["plate_active"], dtype=np.uint8),
        "plate_mass_density": float(plate_mass_density),
        "plate_stiffness_D": float(plate_stiffness_D),
        "plate_alpha_M": float(plate_alpha_M),
        "plate_beta_K": float(plate_beta_K),
        # Plate-to-AMR mappings
        "n_active_plate": int(plate_maps["n_active_plate"]),
        "plate_active_flat_idx": np.ascontiguousarray(plate_maps["plate_active_flat_idx"], dtype=np.int32),
        "plate_face_above_starts": np.ascontiguousarray(plate_maps["plate_face_above_starts"], dtype=np.int32),
        "plate_face_above_idx": np.ascontiguousarray(plate_maps["plate_face_above_idx"], dtype=np.int32),
        "plate_face_above_wgt": np.ascontiguousarray(plate_maps["plate_face_above_wgt"], dtype=np.float32),
        "plate_face_below_starts": np.ascontiguousarray(plate_maps["plate_face_below_starts"], dtype=np.int32),
        "plate_face_below_idx": np.ascontiguousarray(plate_maps["plate_face_below_idx"], dtype=np.int32),
        "plate_face_below_wgt": np.ascontiguousarray(plate_maps["plate_face_below_wgt"], dtype=np.float32),
        "plate_cell_above": np.ascontiguousarray(plate_maps["plate_cell_above"], dtype=np.int32),
        "plate_cell_below": np.ascontiguousarray(plate_maps["plate_cell_below"], dtype=np.int32),
        # Bridge sources
        "n_bridge_plate": int(bridge_map["n_bridge_plate"]),
        "bridge_plate_idx": np.ascontiguousarray(bridge_map["bridge_plate_idx"], dtype=np.int32),
        "bridge_plate_wgt": np.ascontiguousarray(bridge_map["bridge_plate_wgt"], dtype=np.float32),
        # Neck sources
        "n_neck_plate": n_neck,
        "neck_plate_idx": np.ascontiguousarray(neck_idx, dtype=np.int32),
        "neck_plate_wgt": np.ascontiguousarray(neck_wgt, dtype=np.float32),
        # Aperture metadata
        "soundhole_cx": float(cx),
        "soundhole_cy": float(cy),
        "soundhole_radius": float(hr),
        # Border / PML
        "border_mode":         int(border_spec.mode         if border_spec else BORDER_ANECHOIC),
        "border_sigma_order":  float(border_spec.sigma_order  if border_spec else 3.0),
        "border_R_reflection": float(border_spec.R_reflection if border_spec else 0.0),
        "border_Z_match":      float(border_spec.Z_match      if border_spec else 0.0),
        "n_pml":               int(n_pml),
    }
    return desc


# ---------------------------------------------------------------------------
# OpenGL 4.3 Compute-Shader AMR Backend
# ---------------------------------------------------------------------------
# Requirements: PyOpenGL >= 3.1, an active OpenGL 4.3+ context (e.g. from
# opengl_widget.py or a headless context via EGL / osmesa).
# ---------------------------------------------------------------------------

_VELOCITY_UPDATE_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

/* 8th-order (STENCIL_SW=4) Fornberg gradient for velocity update.
 * Bindings match AMRGLComputeBackend.step() velocity pass.
 * face_s_cells[f*8+k]: cell index for stencil tap k, or -1 (padding).
 * face_s_coeff[f*8+k]: Fornberg 1st-deriv weight (Pa/m), precomputed.
 * face_v_damp[f]:      exp(-\u03c3_face\u00b7dt), = 1.0 outside PML.
 */
layout(std430, binding = 0) buffer PressureBuf   { float pressure[];     };
layout(std430, binding = 1) buffer VelocityBuf   { float velocity[];     };
layout(std430, binding = 2) buffer StencilCells  { int   face_s_cells[]; };
layout(std430, binding = 3) buffer StencilCoeff  { float face_s_coeff[]; };
layout(std430, binding = 4) buffer FaceVDampBuf  { float face_v_damp[];  };

uniform float dt_over_rho;
uniform int   n_faces;

void main() {
    uint f = gl_GlobalInvocationID.x;
    if (f >= uint(n_faces)) return;
    const int base = int(f) * 8;
    float grad_p = 0.0;
    for (int k = 0; k < 8; ++k) {
        int ci = face_s_cells[base + k];
        if (ci >= 0) grad_p += face_s_coeff[base + k] * pressure[ci];
    }
    velocity[f] = (velocity[f] - dt_over_rho * grad_p) * face_v_damp[f];
}
"""

_DIVERGENCE_CSR_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer VelocityBuf    { float velocity[];     };
layout(std430, binding = 1) buffer DivFluxBuf     { float div_flux[];     };
layout(std430, binding = 2) buffer CellStartsBuf  { int   cell_starts[];  };
layout(std430, binding = 3) buffer CsrIdxBuf      { int   csr_face_idx[]; };
layout(std430, binding = 4) buffer CsrSignBuf     { float csr_face_sign[];};
layout(std430, binding = 5) buffer FluxCoefBuf    { float face_flux_coef[];};

uniform int n_cells;

void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= uint(n_cells)) return;
    float d = 0.0;
    int k0 = cell_starts[c];
    int k1 = cell_starts[c + 1];
    for (int k = k0; k < k1; ++k) {
        int   f    = csr_face_idx[k];
        float sign = csr_face_sign[k];
        d += sign * velocity[f] * face_flux_coef[f];
    }
    div_flux[c] = d;
}
"""

_PRESSURE_UPDATE_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

/* Exact CPML pressure update:
 *   p_new = p * exp(-\u03c3\u00b7dt) - \u03c1c\u00b2\u00b7dt \u00b7 div/V \u00b7 (1-exp(-\u03c3\u00b7dt))/(\u03c3\u00b7dt)
 * For non-PML cells: P_damp=1, P_src_coeff=1 => standard leapfrog.
 */
layout(std430, binding = 0) buffer PressureBuf   { float pressure[];       };
layout(std430, binding = 1) buffer DivFluxBuf    { float div_flux[];       };
layout(std430, binding = 2) buffer InvDenomBuf   { float cell_inv_denom[]; };
layout(std430, binding = 3) buffer PDampBuf      { float P_damp[];         };
layout(std430, binding = 4) buffer PSrcCoeffBuf  { float P_src_coeff[];    };

uniform float bulk;
uniform int   n_cells;

void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= uint(n_cells)) return;
    pressure[c] = pressure[c] * P_damp[c]
                - bulk * div_flux[c] * cell_inv_denom[c] * P_src_coeff[c];
}
"""

_PLATE_STEP_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PlateWBuf      { float plate_w[];       };
layout(std430, binding = 1) buffer PlateWpBuf     { float plate_w_prev[];  };
layout(std430, binding = 2) buffer PlateWnBuf     { float plate_w_new[];   };
layout(std430, binding = 3) buffer ExtForceBuf    { float ext_force[];     };
layout(std430, binding = 4) buffer ActiveIdxBuf   { int   active_idx[];    };
layout(std430, binding = 5) buffer PressureBuf    { float pressure[];      };
layout(std430, binding = 6) buffer CellAboveBuf   { int   cell_above[];    };
layout(std430, binding = 7) buffer CellBelowBuf   { int   cell_below[];    };

uniform int   N_active;
uniform int   Nx;
uniform int   Ny;
uniform float dx2;       /* dx^2 */
uniform float dx4;       /* 1/dx^4 */
uniform float coeff_D0;  /* D*(1 + bK/dt) */
uniform float coeff_Dp;  /* D*(bK/dt)     */
uniform float damp_fwd;
uniform float damp_bwd;
uniform float dt2_inv_rh; /* dt^2 / (rho_h * damp_fwd) */

float W(int ii, int jj) {
    if (ii < 0 || ii >= Nx || jj < 0 || jj >= Ny) return 0.0;
    return plate_w[ii * Ny + jj];
}
float Wp(int ii, int jj) {
    if (ii < 0 || ii >= Nx || jj < 0 || jj >= Ny) return 0.0;
    return plate_w_prev[ii * Ny + jj];
}

void main() {
    uint n = gl_GlobalInvocationID.x;
    if (n >= uint(N_active)) return;

    int flat = active_idx[n];
    int i    = flat / Ny;
    int j    = flat % Ny;

    /* Acoustic load */
    float F_acou = 0.0;
    int ca = cell_above[n];
    int cb = cell_below[n];
    if (ca >= 0) F_acou += pressure[ca];
    if (cb >= 0) F_acou -= pressure[cb];
    F_acou *= dx2;

    /* 13-point biharmonic stencil */
    float L4w =
          W(i-2,j) + W(i+2,j) + W(i,j-2) + W(i,j+2)
        + 2.0*(W(i-1,j-1)+W(i-1,j+1)+W(i+1,j-1)+W(i+1,j+1))
        - 8.0*(W(i-1,j)+W(i+1,j)+W(i,j-1)+W(i,j+1))
        + 20.0*W(i,j);
    L4w *= dx4;

    float L4wp =
          Wp(i-2,j) + Wp(i+2,j) + Wp(i,j-2) + Wp(i,j+2)
        + 2.0*(Wp(i-1,j-1)+Wp(i-1,j+1)+Wp(i+1,j-1)+Wp(i+1,j+1))
        - 8.0*(Wp(i-1,j)+Wp(i+1,j)+Wp(i,j-1)+Wp(i,j+1))
        + 20.0*Wp(i,j);
    L4wp *= dx4;

    float w_c = plate_w[flat];
    float w_p = plate_w_prev[flat];
    float rhs = F_acou + ext_force[flat] - coeff_D0*L4w + coeff_Dp*L4wp;
    plate_w_new[flat] = (damp_bwd*(2.0*w_c - w_p) + dt2_inv_rh*rhs) / damp_fwd;
}
"""

_PLATE_COMMIT_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PlateWBuf      { float plate_w[];      };
layout(std430, binding = 1) buffer PlateWpBuf     { float plate_w_prev[]; };
layout(std430, binding = 2) buffer PlateWnBuf     { float plate_w_new[];  };
layout(std430, binding = 3) buffer ExtForceBuf    { float ext_force[];    };
layout(std430, binding = 4) buffer ActiveIdxBuf   { int   active_idx[];   };

uniform int N_active;

void main() {
    uint n = gl_GlobalInvocationID.x;
    if (n >= uint(N_active)) return;
    int flat = active_idx[n];
    plate_w_prev[flat] = plate_w[flat];
    plate_w[flat]      = plate_w_new[flat];
    ext_force[flat]    = 0.0;
}
"""

_PLATE_BC_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PlateWBuf      { float plate_w[];      };
layout(std430, binding = 1) buffer PlateWpBuf     { float plate_w_prev[]; };
layout(std430, binding = 2) buffer VelocityBuf    { float velocity[];     };
layout(std430, binding = 3) buffer ActiveIdxBuf   { int   active_idx[];   };
/* CSR for owned faces — each face is owned by exactly one plate node */
layout(std430, binding = 4) buffer FaceStartsBuf  { int   face_starts[];  };
layout(std430, binding = 5) buffer FaceIdxBuf     { int   face_idx[];     };
layout(std430, binding = 6) buffer FaceSignBuf    { float face_sign[];    };
layout(std430, binding = 7) buffer FaceWgtBuf     { float face_wgt[];     };

uniform int   N_active;
uniform float inv_dt;

void main() {
    uint n = gl_GlobalInvocationID.x;
    if (n >= uint(N_active)) return;
    int flat    = active_idx[n];
    float v_plt = (plate_w[flat] - plate_w_prev[flat]) * inv_dt;
    int k0 = face_starts[n];
    int k1 = face_starts[n + 1];
    for (int k = k0; k < k1; ++k)
        velocity[face_idx[k]] = face_sign[k] * v_plt * face_wgt[k];
}
"""

_MIC_SAMPLE_GLSL = """\
#version 430 core
layout(local_size_x = 1) in;   /* one invocation per mic */

layout(std430, binding = 0) buffer PressureBuf   { float pressure[];   };
layout(std430, binding = 1) buffer VelocityBuf   { float velocity[];   };
layout(std430, binding = 2) buffer MicCellIdxBuf { int   mic_cell_idx[];};
layout(std430, binding = 3) buffer MicCellWgtBuf { float mic_cell_wgt[];};
layout(std430, binding = 4) buffer MicFaceIdxBuf { int   mic_face_idx[];};
layout(std430, binding = 5) buffer MicFaceWxBuf  { float mic_face_wx[]; };
layout(std430, binding = 6) buffer MicFaceWyBuf  { float mic_face_wy[]; };
layout(std430, binding = 7) buffer MicFaceWzBuf  { float mic_face_wz[]; };
layout(std430, binding = 8) buffer MicStartsBuf  { int   mic_starts[];  };
layout(std430, binding = 9) buffer MicOutBuf     { float mic_out[];     };
/* mic_out layout: [p0, vx0, vy0, vz0, p1, vx1, vy1, vz1, ...] */

uniform int n_mics;

void main() {
    uint m = gl_GlobalInvocationID.x;
    if (m >= uint(n_mics)) return;
    int k0 = mic_starts[m];
    int k1 = mic_starts[m + 1];
    float p = 0.0, vx = 0.0, vy = 0.0, vz = 0.0;
    for (int k = k0; k < k1; ++k) {
        p  += mic_cell_wgt[k] * pressure[mic_cell_idx[k]];
        vx += mic_face_wx[k]  * velocity[mic_face_idx[k]];
        vy += mic_face_wy[k]  * velocity[mic_face_idx[k]];
        vz += mic_face_wz[k]  * velocity[mic_face_idx[k]];
    }
    int base = int(m) * 4;
    mic_out[base + 0] = p;
    mic_out[base + 1] = vx;
    mic_out[base + 2] = vy;
    mic_out[base + 3] = vz;
}
"""


def _gl_ceil_div(n: int, local: int) -> int:
    return (n + local - 1) // local


def _compile_shader(src: str):
    """Compile a GLSL compute shader; raise RuntimeError on failure."""
    from OpenGL.GL import (
        glCreateShader, glShaderSource, glCompileShader,
        glGetShaderiv, glGetShaderInfoLog,
        GL_COMPUTE_SHADER, GL_COMPILE_STATUS, GL_TRUE,
    )
    sh = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if glGetShaderiv(sh, GL_COMPILE_STATUS) != GL_TRUE:
        log = glGetShaderInfoLog(sh).decode("utf-8", errors="replace")
        raise RuntimeError(f"Compute shader compile error:\n{log}")
    return sh


def _link_program(shader):
    """Link a single compute shader into a program; raise on failure."""
    from OpenGL.GL import (
        glCreateProgram, glAttachShader, glLinkProgram,
        glGetProgramiv, glGetProgramInfoLog,
        GL_LINK_STATUS, GL_TRUE,
    )
    prog = glCreateProgram()
    glAttachShader(prog, shader)
    glLinkProgram(prog)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        log = glGetProgramInfoLog(prog).decode("utf-8", errors="replace")
        raise RuntimeError(f"Compute program link error:\n{log}")
    return prog


def _make_ssbo(data: np.ndarray, binding: int):
    """Upload a numpy array as a static SSBO and return (buffer_id, binding)."""
    from OpenGL.GL import (
        glGenBuffers, glBindBuffer, glBufferData, glBindBufferBase,
        GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW,
    )
    buf = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
    glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return buf


def _make_ssbo_zeros(nbytes: int):
    """Allocate a zeroed SSBO of the given byte size."""
    from OpenGL.GL import (
        glGenBuffers, glBindBuffer, glBufferData,
        GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW,
    )
    buf = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
    glBufferData(GL_SHADER_STORAGE_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return buf


def _update_ssbo(buf, data: np.ndarray) -> None:
    """Re-upload a numpy array into an already-allocated SSBO."""
    from OpenGL.GL import (
        glBindBuffer, glBufferSubData,
        GL_SHADER_STORAGE_BUFFER,
    )
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, data)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)


# ---------------------------------------------------------------------------
# 8th-order Fornberg stencil builder (CPU — called once at setup time)
# ---------------------------------------------------------------------------

def _fornberg_d1(x: np.ndarray, xi: float = 0.0) -> np.ndarray:
    """Fornberg (1988) first-derivative weights at xi for arbitrary node positions x.

    Returns float32 weights w such that f'(xi) ≈ Σ_k w[k] * f(x[k]).
    Accuracy order = len(x) − 1.
    """
    N = len(x)
    c = np.zeros((N, 2), dtype=np.float64)
    c[0, 0] = 1.0
    c1 = 1.0
    for n in range(1, N):
        mn = min(n, 1)
        c2 = 1.0
        for nu in range(n):
            c3 = float(x[n] - x[nu])
            c2 *= c3
            if nu == n - 1:
                for m in range(mn, 0, -1):
                    c[n, m] = c1 / c2 * (m * c[n - 1, m - 1] - (x[n] - xi) * c[n - 1, m])
                c[n, 0] = c1 / c2 * (-(x[n] - xi)) * c[n - 1, 0]
            for m in range(mn, 0, -1):
                c[nu, m] = ((x[n] - xi) * c[nu, m] - m * c[nu, m - 1]) / c3
            c[nu, 0] = (x[n] - xi) * c[nu, 0] / c3
        c1 = c2
    return c[:, 1].astype(np.float32)


def _build_face_stencil(
    grid: AcousticAMRGrid,
    sw: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-face 8th-order Fornberg gradient stencil (CPU, one-time cost).

    Mirrors ``amr_build_stencil`` in acoustic_amr.cpp.  Called once in
    ``AMRGLComputeBackend.__init__`` to populate the GPU stencil SSBOs.

    Parameters
    ----------
    grid : AcousticAMRGrid
    sw   : stencil half-width (default 4 → up to 8 cells per face).

    Returns
    -------
    face_s_cells : int32 array  (n_faces * 2*sw,)  — cell indices (−1 = unused)
    face_s_coeff : float32 array (n_faces * 2*sw,) — Fornberg weights (Pa/m)
    """
    nc, nf = grid.n_cells, grid.n_faces
    sw2 = sw * 2
    fn  = grid.face_cell_neg.astype(np.int64)
    fp  = grid.face_cell_pos.astype(np.int64)
    cc  = np.asarray(grid.cell_centers, dtype=np.float64)
    ct  = np.asarray(grid.cell_types,   dtype=np.int32)

    # Build cell→faces CSR (once)
    cfs = np.zeros(nc + 1, dtype=np.int64)
    np.add.at(cfs[1:], fn, 1)
    np.add.at(cfs[1:], fp, 1)
    np.cumsum(cfs, out=cfs)
    total = int(cfs[nc])
    cfi = np.empty(total, dtype=np.int64)   # face index
    cfg = np.empty(total, dtype=np.float64) # sign: +1 if cell is neg, -1 if pos
    fill = cfs[:-1].copy()
    for f in range(nf):
        a, b = int(fn[f]), int(fp[f])
        ka = fill[a]; fill[a] += 1; cfi[ka] = f; cfg[ka] = +1.0
        kb = fill[b]; fill[b] += 1; cfi[kb] = f; cfg[kb] = -1.0

    face_s_cells = np.full(nf * sw2, -1, dtype=np.int32)
    face_s_coeff = np.zeros(nf * sw2, dtype=np.float32)

    for f in range(nf):
        cn_i, cp_i = int(fn[f]), int(fp[f])
        d = cc[cp_i] - cc[cn_i]
        dlen = float(np.linalg.norm(d))
        if dlen < 1e-15:
            continue
        dir_ = d / dlen
        face_proj = 0.5 * (float(np.dot(cc[cn_i], dir_)) + float(np.dot(cc[cp_i], dir_)))

        def _walk(seed: int, direction: np.ndarray) -> tuple[list, list]:
            cells = [seed]
            xs    = [float(np.dot(cc[seed], direction)) - face_proj]
            cur   = seed
            for _ in range(sw - 1):
                best_nb, best_dot = -1, 0.5
                k0, k1 = int(cfs[cur]), int(cfs[cur + 1])
                for k in range(k0, k1):
                    fk = int(cfi[k])
                    sg = float(cfg[k])
                    fn_d = cc[int(fp[fk])] - cc[int(fn[fk])]
                    fn_len = float(np.linalg.norm(fn_d))
                    if fn_len < 1e-15:
                        continue
                    dot_ = sg * float(np.dot(fn_d / fn_len, direction))
                    if dot_ > best_dot:
                        nb = int(fp[fk]) if sg > 0 else int(fn[fk])
                        if nb != cur and int(ct[nb]) not in (1, 2):
                            best_dot, best_nb = dot_, nb
                if best_nb < 0:
                    break
                cells.append(best_nb)
                xs.append(float(np.dot(cc[best_nb], direction)) - face_proj)
                cur = best_nb
            return cells, xs

        neg_cells, neg_x = _walk(cn_i, -dir_)
        pos_cells, pos_x = _walk(cp_i,  dir_)

        total_pts = len(neg_cells) + len(pos_cells)
        if total_pts < 2:
            continue

        # Merge: farthest-neg → nearest-neg → nearest-pos → farthest-pos
        xs_all = np.array(neg_x[::-1] + pos_x, dtype=np.float64)
        cs_all = neg_cells[::-1] + pos_cells
        w = _fornberg_d1(xs_all, xi=0.0)

        n_neg = len(neg_cells)
        n_pos = len(pos_cells)
        base = f * sw2
        # neg slots: 0 = nearest, n_neg-1 = farthest  (reverse of xs_all order)
        for k in range(n_neg):
            face_s_cells[base + k]         = cs_all[n_neg - 1 - k]
            face_s_coeff[base + k]         = w[n_neg - 1 - k]
        # pos slots: sw+0 = nearest, sw+n_pos-1 = farthest
        for k in range(n_pos):
            face_s_cells[base + sw + k]    = cs_all[n_neg + k]
            face_s_coeff[base + sw + k]    = w[n_neg + k]

    return face_s_cells, face_s_coeff


def _uniform_i(prog, name: str, val: int):
    from OpenGL.GL import glGetUniformLocation, glUniform1i
    glUniform1i(glGetUniformLocation(prog, name), int(val))


def _uniform_f(prog, name: str, val: float):
    from OpenGL.GL import glGetUniformLocation, glUniform1f
    glUniform1f(glGetUniformLocation(prog, name), float(val))


class AMRGLComputeBackend:
    """OpenGL 4.3 compute-shader implementation of the AMR FDTD step.

    Drops-in alongside ``AcousticAMRFDTD`` for the acoustic pressure step.
    Requires an active OpenGL 4.3+ context before construction.  All heavy
    arrays (pressure, velocity, topology) live in GPU SSBOs; only mic
    samples (~4 floats × n_mics) are read back to Python per step.

    Physics
    -------
    Same staggered pressure/velocity leapfrog as the C++ backend:
      1. velocity_update  — per-face, embarrassingly parallel
      2. plate_bc         — per-active-plate-node, writes owned faces
      3. divergence_csr   — per-cell, CSR (no atomics)
      4. pressure_update  — per-cell, uses precomputed inv_denom
      5. plate_step       — per-active-plate-node biharmonic leapfrog
      6. plate_commit     — swap w/w_prev, zero ext_force
      7. mic_sample       — per-mic gather (optional, on demand)

    Parameters
    ----------
    grid  : AcousticAMRGrid topology descriptor.
    c     : Speed of sound (m/s).
    rho_air : Air density (kg/m³).
    """

    _LOCAL = 256  # local_size_x for all general shaders

    def __init__(self, grid: AcousticAMRGrid, c: float = 343.0, rho_air: float = 1.21):
        try:
            from OpenGL.GL import GL_VERSION  # noqa: F401 — presence check only
        except ImportError as exc:
            raise RuntimeError(
                "PyOpenGL is required for AMRGLComputeBackend. "
                "Install with: pip install PyOpenGL"
            ) from exc

        self.grid = grid
        self.c = float(c)
        self.rho_air = float(rho_air)

        n_cells = grid.n_cells
        n_faces = grid.n_faces

        # ── CFL timestep (matches C++ amr_create) ────────────────────────
        min_dx = float(2.0 * grid.cell_half_sizes.min())
        self._dt = 0.77 * min_dx / (c * math.sqrt(3.0))

        # ── Precomputed topology arrays ───────────────────────────────────
        face_flux_coef = (grid.face_area * grid.face_open_fraction).astype(np.float32)

        cell_inv_denom = np.zeros(n_cells, dtype=np.float32)
        acoustic_mask = ((grid.cell_types == 0) | (grid.cell_types == 3)) & (grid.open_volume_fraction > 0)
        cell_inv_denom[acoustic_mask] = (
            1.0 / (grid.cell_volumes[acoustic_mask] * np.maximum(grid.open_volume_fraction[acoustic_mask], 1e-12))
        ).astype(np.float32)
        # Wall / plate cells get inv_denom == 0, so pressure update zeroes them implicitly

        # ── CSR per-cell face lists ───────────────────────────────────────
        csr_starts = np.zeros(n_cells + 1, dtype=np.int32)
        fn = grid.face_cell_neg.astype(np.int32)
        fp = grid.face_cell_pos.astype(np.int32)
        np.add.at(csr_starts[1:], fn, 1)
        np.add.at(csr_starts[1:], fp, 1)
        np.cumsum(csr_starts, out=csr_starts)
        total = int(csr_starts[n_cells])
        csr_face_idx  = np.empty(total, dtype=np.int32)
        csr_face_sign = np.empty(total, dtype=np.float32)
        fill = csr_starts[:-1].copy()
        for f in range(n_faces):
            a, b = int(fn[f]), int(fp[f])
            ka = fill[a]; fill[a] += 1
            csr_face_idx[ka]  = f;  csr_face_sign[ka] = +1.0
            kb = fill[b]; fill[b] += 1
            csr_face_idx[kb]  = f;  csr_face_sign[kb] = -1.0

        # ── 8th-order stencil (Fornberg, built once on CPU) ──────────────
        s_cells, s_coeff = _build_face_stencil(grid, sw=4)

        # ── Upload topology SSBOs (persistent, never change) ──────────────
        self._buf_pressure      = _make_ssbo_zeros(n_cells * 4)
        self._buf_velocity      = _make_ssbo_zeros(n_faces * 4)
        self._buf_div_flux      = _make_ssbo_zeros(n_cells * 4)
        self._buf_stencil_cells = _make_ssbo(s_cells,        binding=0)
        self._buf_stencil_coeff = _make_ssbo(s_coeff,        binding=0)
        self._buf_flux_coef     = _make_ssbo(face_flux_coef, binding=0)
        self._buf_csr_starts    = _make_ssbo(csr_starts,     binding=0)
        self._buf_csr_idx       = _make_ssbo(csr_face_idx,   binding=0)
        self._buf_csr_sign      = _make_ssbo(csr_face_sign,  binding=0)
        self._buf_inv_denom     = _make_ssbo(cell_inv_denom, binding=0)
        # PML damping arrays — initialised to identity (no absorption).
        # Call setup_border() after construction to activate PML.
        self._buf_face_v_damp   = _make_ssbo(np.ones(n_faces, dtype=np.float32), binding=0)
        self._buf_p_damp        = _make_ssbo(np.ones(n_cells, dtype=np.float32), binding=0)
        self._buf_p_src_coeff   = _make_ssbo(np.ones(n_cells, dtype=np.float32), binding=0)

        self._n_cells = n_cells
        self._n_faces = n_faces

        # ── Plate state (populated in setup_plate) ────────────────────────
        self._plate_active   = False
        self._n_active_plate = 0
        self._buf_plate_w    = None
        self._buf_plate_wp   = None
        self._buf_plate_wn   = None
        self._buf_ext_force  = None
        self._buf_active_idx = None
        self._buf_cell_above = None
        self._buf_cell_below = None
        self._buf_bc_starts  = None
        self._buf_bc_idx     = None
        self._buf_bc_sign    = None
        self._buf_bc_wgt     = None
        self._plate_uniforms: dict = {}

        # ── Mic state (populated in setup_mics) ───────────────────────────
        self._n_mics = 0
        self._buf_mic_out    = None
        self._buf_mc_idx     = None
        self._buf_mc_wgt     = None
        self._buf_mf_idx     = None
        self._buf_mf_wx      = None
        self._buf_mf_wy      = None
        self._buf_mf_wz      = None
        self._buf_mic_starts = None

        # ── Compile shaders ───────────────────────────────────────────────
        self._prog_vel  = _link_program(_compile_shader(_VELOCITY_UPDATE_GLSL))
        self._prog_div  = _link_program(_compile_shader(_DIVERGENCE_CSR_GLSL))
        self._prog_pres = _link_program(_compile_shader(_PRESSURE_UPDATE_GLSL))
        self._prog_plate = _link_program(_compile_shader(_PLATE_STEP_GLSL))
        self._prog_plate_commit = _link_program(_compile_shader(_PLATE_COMMIT_GLSL))
        self._prog_plate_bc = _link_program(_compile_shader(_PLATE_BC_GLSL))
        self._prog_mic  = _link_program(_compile_shader(_MIC_SAMPLE_GLSL))

        # Precomputed step uniforms
        self._dt_over_rho = float(self._dt / rho_air)
        self._bulk = float(rho_air * c * c * self._dt)

    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def step_count(self) -> int:
        return self._step_count if hasattr(self, "_step_count") else 0

    # ------------------------------------------------------------------
    def setup_border(
        self,
        border_spec: 'BorderConditionSpec',
        n_pml: int,
        bounds_min: np.ndarray | None = None,
        bounds_max: np.ndarray | None = None,
    ) -> None:
        """Compute and upload CPML absorption arrays to the GPU.

        Call after construction (or any time the grid changes) to activate
        PML absorption.  By default the backend initialises with identity
        damping (no absorption).

        Parameters
        ----------
        border_spec : BorderConditionSpec
        n_pml       : Number of PML cells (passed to amr_set_border_condition).
        bounds_min  : (3,) min corner of the simulation domain; defaults to
                      grid bounding box.
        bounds_max  : (3,) max corner of the simulation domain.
        """
        import math as _math
        grid  = self.grid
        nc    = self._n_cells
        nf    = self._n_faces
        dt    = self._dt
        c     = self.c
        ct    = np.asarray(grid.cell_types,   dtype=np.int32)
        cc    = np.asarray(grid.cell_centers, dtype=np.float64)

        if bounds_min is None:
            bounds_min = cc.min(axis=0)
        if bounds_max is None:
            bounds_max = cc.max(axis=0)
        bounds_min = np.asarray(bounds_min, dtype=np.float64)
        bounds_max = np.asarray(bounds_max, dtype=np.float64)

        sigma_order = float(border_spec.sigma_order) if border_spec.sigma_order > 0 else 3.0
        # sigma_max calibrated for -60 dB attenuation over the PML depth
        min_dx   = dt * c * _math.sqrt(3.0) / 0.77
        pml_depth = n_pml * min_dx
        sigma_max = 3.45 / (dt * n_pml) if n_pml > 0 else 0.0
        if border_spec.mode == BORDER_ROOM_PANEL:
            sigma_max *= max(0.0, 1.0 - max(0.0, min(1.0, float(border_spec.R_reflection))))

        border_alpha = np.zeros(nc, dtype=np.float64)
        pml_mask = ct == int(AMR_PML)
        if pml_mask.any() and pml_depth > 0.0:
            cx = cc[pml_mask, 0]; cy = cc[pml_mask, 1]; cz = cc[pml_mask, 2]
            d = np.minimum.reduce([
                cx - bounds_min[0], bounds_max[0] - cx,
                cy - bounds_min[1], bounds_max[1] - cy,
                cz - bounds_min[2], bounds_max[2] - cz,
            ])
            d = np.maximum(d, 0.0)
            norm = np.clip(1.0 - d / pml_depth, 0.0, 1.0)
            border_alpha[pml_mask] = sigma_max * norm ** sigma_order

        s = border_alpha * dt
        P_damp      = np.exp(-s).astype(np.float32)
        P_src_coeff = np.where(
            s < 1e-6,
            (1.0 - 0.5 * s).astype(np.float64),
            (1.0 - np.exp(-s)) / np.maximum(s, 1e-300),
        ).astype(np.float32)

        fn = np.asarray(grid.face_cell_neg, dtype=np.int64)
        fp = np.asarray(grid.face_cell_pos, dtype=np.int64)
        ov = np.asarray(grid.open_volume_fraction, dtype=np.float64)
        vn = ov[fn]; vp = ov[fp]
        denom  = vn + vp
        s_face = np.where(
            denom > 0,
            2.0 * (vp * border_alpha[fn] + vn * border_alpha[fp]) / np.maximum(denom, 1e-300),
            0.5 * (border_alpha[fn] + border_alpha[fp]),
        )
        face_V_damp = np.exp(-s_face * dt).astype(np.float32)

        _update_ssbo(self._buf_p_damp,      P_damp)
        _update_ssbo(self._buf_p_src_coeff, P_src_coeff)
        _update_ssbo(self._buf_face_v_damp, face_V_damp)

    # ------------------------------------------------------------------
    def setup_plate(self, desc: dict) -> None:
        """Upload plate topology and initial state SSBOs.

        Parameters
        ----------
        desc : dict returned by ``build_amr_coevolver_descriptor``.
        """
        Nx = int(desc["plate_Nx"])
        Ny = int(desc["plate_Ny"])
        dx = float(desc["plate_dx"])
        N  = Nx * Ny
        rh = float(desc["plate_mass_density"])
        D  = float(desc["plate_stiffness_D"])
        aM = float(desc["plate_alpha_M"])
        bK = float(desc["plate_beta_K"])
        dt = self._dt

        n_active = int(desc["n_plate_active"])
        active_idx  = np.ascontiguousarray(desc["plate_active_idx"],   dtype=np.int32)
        cell_above  = np.ascontiguousarray(desc["plate_cell_above"],   dtype=np.int32)
        cell_below  = np.ascontiguousarray(desc["plate_cell_below"],   dtype=np.int32)

        self._buf_plate_w    = _make_ssbo_zeros(N * 4)
        self._buf_plate_wp   = _make_ssbo_zeros(N * 4)
        self._buf_plate_wn   = _make_ssbo_zeros(N * 4)
        self._buf_ext_force  = _make_ssbo_zeros(N * 4)
        self._buf_active_idx = _make_ssbo(active_idx, binding=0)
        self._buf_cell_above = _make_ssbo(cell_above, binding=0)
        self._buf_cell_below = _make_ssbo(cell_below, binding=0)

        # Per-node face BC CSR (above/below merged)
        fa_starts = np.ascontiguousarray(desc["plate_face_above_starts"], dtype=np.int32)
        fa_idx    = np.ascontiguousarray(desc["plate_face_above_idx"],    dtype=np.int32)
        fa_wgt    = np.ascontiguousarray(desc["plate_face_above_wgt"],    dtype=np.float32)
        fb_starts = np.ascontiguousarray(desc["plate_face_below_starts"], dtype=np.int32)
        fb_idx    = np.ascontiguousarray(desc["plate_face_below_idx"],    dtype=np.int32)
        fb_wgt    = np.ascontiguousarray(desc["plate_face_below_wgt"],    dtype=np.float32)

        # Merge above/below into a single CSR with sign: +1 for above, -1 for below
        bc_starts = np.zeros(n_active + 1, dtype=np.int32)
        for n in range(n_active):
            na = int(fa_starts[n + 1]) - int(fa_starts[n])
            nb = int(fb_starts[n + 1]) - int(fb_starts[n])
            bc_starts[n + 1] = bc_starts[n] + na + nb
        total_bc = int(bc_starts[n_active])
        bc_idx  = np.empty(total_bc, dtype=np.int32)
        bc_sign = np.empty(total_bc, dtype=np.float32)
        bc_wgt  = np.empty(total_bc, dtype=np.float32)
        for n in range(n_active):
            k = int(bc_starts[n])
            a0 = int(fa_starts[n]); a1 = int(fa_starts[n + 1])
            for ki in range(a0, a1):
                bc_idx[k] = fa_idx[ki]; bc_sign[k] = +1.0; bc_wgt[k] = fa_wgt[ki]; k += 1
            b0 = int(fb_starts[n]); b1 = int(fb_starts[n + 1])
            for ki in range(b0, b1):
                bc_idx[k] = fb_idx[ki]; bc_sign[k] = -1.0; bc_wgt[k] = fb_wgt[ki]; k += 1

        self._buf_bc_starts = _make_ssbo(bc_starts, binding=0)
        self._buf_bc_idx    = _make_ssbo(bc_idx,    binding=0)
        self._buf_bc_sign   = _make_ssbo(bc_sign,   binding=0)
        self._buf_bc_wgt    = _make_ssbo(bc_wgt,    binding=0)

        # Plate uniforms (precomputed, constant after setup)
        damp_fwd = 1.0 + 0.5 * aM * dt
        damp_bwd = 1.0 - 0.5 * aM * dt
        self._plate_uniforms = {
            "N_active":   n_active,
            "Nx":         Nx,
            "Ny":         Ny,
            "dx2":        float(dx * dx),
            "dx4":        float(1.0 / (dx ** 4)),
            "coeff_D0":   float(D * (1.0 + bK / dt)),
            "coeff_Dp":   float(D * (bK / dt)),
            "damp_fwd":   float(damp_fwd),
            "damp_bwd":   float(damp_bwd),
            "dt2_inv_rh": float(dt * dt / (rh * damp_fwd)),
            "inv_dt":     float(1.0 / dt),
        }
        self._n_active_plate = n_active
        self._plate_active   = True

    # ------------------------------------------------------------------
    def setup_mics(
        self,
        mic_cell_idx:  np.ndarray,
        mic_cell_wgt:  np.ndarray,
        mic_face_idx:  np.ndarray,
        mic_face_wx:   np.ndarray,
        mic_face_wy:   np.ndarray,
        mic_face_wz:   np.ndarray,
        mic_starts:    np.ndarray,
    ) -> None:
        """Upload mic sampler SSBOs.

        Each mic has a variable number of contributing cells/faces listed in
        CSR order (``mic_starts`` is the prefix-sum array, length n_mics+1).
        """
        n_mics = int(mic_starts.shape[0]) - 1
        self._n_mics      = n_mics
        self._buf_mic_out = _make_ssbo_zeros(n_mics * 4 * 4)  # 4 floats × n_mics
        self._buf_mc_idx  = _make_ssbo(mic_cell_idx.astype(np.int32),   binding=0)
        self._buf_mc_wgt  = _make_ssbo(mic_cell_wgt.astype(np.float32), binding=0)
        self._buf_mf_idx  = _make_ssbo(mic_face_idx.astype(np.int32),   binding=0)
        self._buf_mf_wx   = _make_ssbo(mic_face_wx.astype(np.float32),  binding=0)
        self._buf_mf_wy   = _make_ssbo(mic_face_wy.astype(np.float32),  binding=0)
        self._buf_mf_wz   = _make_ssbo(mic_face_wz.astype(np.float32),  binding=0)
        self._buf_mic_starts = _make_ssbo(mic_starts.astype(np.int32),  binding=0)

    # ------------------------------------------------------------------
    def inject_bridge_drive(self, forces: np.ndarray, scale: float = 1.0) -> None:
        """Accumulate bridge force into the plate ext_force SSBO.

        Parameters
        ----------
        forces : float32 array (n_bridge_plate,) — per-node force values.
        scale  : overall scale factor applied to forces.
        """
        from OpenGL.GL import (
            glBindBuffer, glMapBufferRange, glUnmapBuffer,
            GL_SHADER_STORAGE_BUFFER, GL_MAP_WRITE_BIT, GL_MAP_READ_BIT,
        )
        import ctypes
        if not self._plate_active or self._buf_ext_force is None:
            return
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_ext_force)
        n = len(forces)
        ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, n * 4,
                               GL_MAP_WRITE_BIT | GL_MAP_READ_BIT)
        arr = (ctypes.c_float * n).from_address(ptr)
        for i in range(n):
            arr[i] += float(forces[i]) * float(scale)
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # ------------------------------------------------------------------
    def inject_pressure(self, cell_idx: int, value: float) -> None:
        """Add ``value`` to the pressure of a single cell."""
        from OpenGL.GL import (
            glBindBuffer, glMapBufferRange, glUnmapBuffer,
            GL_SHADER_STORAGE_BUFFER, GL_MAP_WRITE_BIT, GL_MAP_READ_BIT,
        )
        import ctypes
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_pressure)
        ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                               cell_idx * 4, 4,
                               GL_MAP_WRITE_BIT | GL_MAP_READ_BIT)
        old_val = ctypes.c_float.from_address(ptr).value
        ctypes.c_float.from_address(ptr).value = old_val + float(value)
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # ------------------------------------------------------------------
    def _bind_base(self, buf, binding: int) -> None:
        from OpenGL.GL import glBindBufferBase, GL_SHADER_STORAGE_BUFFER
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf)

    def step(self, n_steps: int = 1) -> None:
        """Advance AMR FDTD by ``n_steps`` steps on the GPU."""
        from OpenGL.GL import (
            glUseProgram, glDispatchCompute, glMemoryBarrier,
            GL_SHADER_STORAGE_BARRIER_BIT,
        )
        n_cells  = self._n_cells
        n_faces  = self._n_faces
        L        = self._LOCAL
        n_active = self._n_active_plate

        for _ in range(n_steps):
            # ── 1. Velocity update ────────────────────────────────────
            glUseProgram(self._prog_vel)
            self._bind_base(self._buf_pressure,      0)
            self._bind_base(self._buf_velocity,      1)
            self._bind_base(self._buf_stencil_cells, 2)
            self._bind_base(self._buf_stencil_coeff, 3)
            self._bind_base(self._buf_face_v_damp,   4)
            _uniform_f(self._prog_vel, "dt_over_rho", self._dt_over_rho)
            _uniform_i(self._prog_vel, "n_faces", n_faces)
            glDispatchCompute(_gl_ceil_div(n_faces, L), 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 2. Plate BC (if active) ───────────────────────────────
            if self._plate_active:
                glUseProgram(self._prog_plate_bc)
                self._bind_base(self._buf_plate_w,  0)
                self._bind_base(self._buf_plate_wp, 1)
                self._bind_base(self._buf_velocity, 2)
                self._bind_base(self._buf_active_idx, 3)
                self._bind_base(self._buf_bc_starts,  4)
                self._bind_base(self._buf_bc_idx,     5)
                self._bind_base(self._buf_bc_sign,    6)
                self._bind_base(self._buf_bc_wgt,     7)
                _uniform_i(self._prog_plate_bc, "N_active", n_active)
                _uniform_f(self._prog_plate_bc, "inv_dt",
                           self._plate_uniforms["inv_dt"])
                glDispatchCompute(_gl_ceil_div(n_active, L), 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 3. Divergence CSR ─────────────────────────────────────
            glUseProgram(self._prog_div)
            self._bind_base(self._buf_velocity,  0)
            self._bind_base(self._buf_div_flux,  1)
            self._bind_base(self._buf_csr_starts, 2)
            self._bind_base(self._buf_csr_idx,   3)
            self._bind_base(self._buf_csr_sign,  4)
            self._bind_base(self._buf_flux_coef, 5)
            _uniform_i(self._prog_div, "n_cells", n_cells)
            glDispatchCompute(_gl_ceil_div(n_cells, L), 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 4. Pressure update (exact CPML) ──────────────────────
            glUseProgram(self._prog_pres)
            self._bind_base(self._buf_pressure,    0)
            self._bind_base(self._buf_div_flux,    1)
            self._bind_base(self._buf_inv_denom,   2)
            self._bind_base(self._buf_p_damp,      3)
            self._bind_base(self._buf_p_src_coeff, 4)
            _uniform_f(self._prog_pres, "bulk", self._bulk)
            _uniform_i(self._prog_pres, "n_cells", n_cells)
            glDispatchCompute(_gl_ceil_div(n_cells, L), 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 5. Plate step (if active) ─────────────────────────────
            if self._plate_active:
                pu = self._plate_uniforms
                glUseProgram(self._prog_plate)
                self._bind_base(self._buf_plate_w,    0)
                self._bind_base(self._buf_plate_wp,   1)
                self._bind_base(self._buf_plate_wn,   2)
                self._bind_base(self._buf_ext_force,  3)
                self._bind_base(self._buf_active_idx, 4)
                self._bind_base(self._buf_pressure,   5)
                self._bind_base(self._buf_cell_above, 6)
                self._bind_base(self._buf_cell_below, 7)
                _uniform_i(self._prog_plate, "N_active", n_active)
                _uniform_i(self._prog_plate, "Nx",       pu["Nx"])
                _uniform_i(self._prog_plate, "Ny",       pu["Ny"])
                _uniform_f(self._prog_plate, "dx2",      pu["dx2"])
                _uniform_f(self._prog_plate, "dx4",      pu["dx4"])
                _uniform_f(self._prog_plate, "coeff_D0", pu["coeff_D0"])
                _uniform_f(self._prog_plate, "coeff_Dp", pu["coeff_Dp"])
                _uniform_f(self._prog_plate, "damp_fwd", pu["damp_fwd"])
                _uniform_f(self._prog_plate, "damp_bwd", pu["damp_bwd"])
                _uniform_f(self._prog_plate, "dt2_inv_rh", pu["dt2_inv_rh"])
                glDispatchCompute(_gl_ceil_div(n_active, L), 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

                # ── 6. Plate commit (w ← w_new, clear ext_force) ─────
                glUseProgram(self._prog_plate_commit)
                self._bind_base(self._buf_plate_w,    0)
                self._bind_base(self._buf_plate_wp,   1)
                self._bind_base(self._buf_plate_wn,   2)
                self._bind_base(self._buf_ext_force,  3)
                self._bind_base(self._buf_active_idx, 4)
                _uniform_i(self._prog_plate_commit, "N_active", n_active)
                glDispatchCompute(_gl_ceil_div(n_active, L), 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += n_steps

    # ------------------------------------------------------------------
    def sample_mics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run mic sampling shader and return (p, vx, vy, vz) float32 arrays.

        Returns arrays of shape (n_mics,).  Requires ``setup_mics`` to have
        been called first.
        """
        from OpenGL.GL import (
            glUseProgram, glDispatchCompute, glMemoryBarrier,
            glBindBuffer, glGetBufferSubData,
            GL_SHADER_STORAGE_BARRIER_BIT, GL_SHADER_STORAGE_BUFFER,
        )
        n = self._n_mics
        if n == 0:
            z = np.zeros(0, dtype=np.float32)
            return z, z, z, z

        glUseProgram(self._prog_mic)
        self._bind_base(self._buf_pressure,  0)
        self._bind_base(self._buf_velocity,  1)
        self._bind_base(self._buf_mc_idx,    2)
        self._bind_base(self._buf_mc_wgt,    3)
        self._bind_base(self._buf_mf_idx,    4)
        self._bind_base(self._buf_mf_wx,     5)
        self._bind_base(self._buf_mf_wy,     6)
        self._bind_base(self._buf_mf_wz,     7)
        self._bind_base(self._buf_mic_starts, 8)
        self._bind_base(self._buf_mic_out,   9)
        _uniform_i(self._prog_mic, "n_mics", n)
        glDispatchCompute(n, 1, 1)  # one invocation per mic (local_size_x=1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        out = np.empty(n * 4, dtype=np.float32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_mic_out)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, out)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        return out[0::4].copy(), out[1::4].copy(), out[2::4].copy(), out[3::4].copy()

    # ------------------------------------------------------------------
    def get_pressure(self) -> np.ndarray:
        """Read full pressure field back to CPU (slow — debug / test only)."""
        from OpenGL.GL import (
            glBindBuffer, glGetBufferSubData, GL_SHADER_STORAGE_BUFFER,
        )
        out = np.empty(self._n_cells, dtype=np.float32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_pressure)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, out)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return out

    def get_velocity(self) -> np.ndarray:
        """Read full velocity field back to CPU (slow — debug / test only)."""
        from OpenGL.GL import (
            glBindBuffer, glGetBufferSubData, GL_SHADER_STORAGE_BUFFER,
        )
        out = np.empty(self._n_faces, dtype=np.float32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_velocity)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, out)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return out

    def reset(self) -> None:
        """Zero pressure and velocity on the GPU."""
        from OpenGL.GL import (
            glBindBuffer, glBufferData,
            GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW,
        )
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_pressure)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self._n_cells * 4, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_velocity)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self._n_faces * 4, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        if hasattr(self, "_step_count"):
            self._step_count = 0
