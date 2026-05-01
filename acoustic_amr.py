"""Adaptive multi-resolution acoustic pressure grid.

This module builds a true AMR pressure/velocity topology for the guitar body.
It is intentionally strict: construction validates the requested content
importance policy and raises on unsupported or inconsistent geometry instead
of falling back to the legacy uniform FDTD grid.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
import os
import time
from typing import Any

import numpy as np

from graph_solver import _T as _PROF


_AMR_STEP_REPORT = os.environ.get("SPECTRAL_AMR_STEP_REPORT", "").lower() in {"1", "true", "yes", "on"}
_AMR_TQDM = os.environ.get("SPECTRAL_AMR_TQDM", "1").lower() not in {"0", "false", "no", "off"}
# Headless batch mode: disables progress formatting and span timing overhead.
# Set SPECTRAL_HEADLESS_BATCH=1 (or true/yes/on) before importing this module.
_HEADLESS_BATCH = os.environ.get("SPECTRAL_HEADLESS_BATCH", "").lower() in {"1", "true", "yes", "on"}


def _tqdm_or_none(*args, **kwargs):
    if not _AMR_TQDM:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(*args, **kwargs)


def _progress_iter(iterable, *, total: int, desc: str, unit: str = "it",
                   progress_cb: Any = None, frac0: float = 0.0, frac1: float = 1.0,
                   report_every: int | None = None):
    if _HEADLESS_BATCH:
        yield from iterable
        return
    total_i = max(1, int(total))
    bar = None if progress_cb is not None else _tqdm_or_none(
        iterable, total=total_i, desc=desc, unit=unit, leave=False
    )
    if bar is not None:
        for idx, item in enumerate(bar, 1):
            if progress_cb is not None and (idx == 1 or idx == total_i or idx % max(1, total_i // 100) == 0):
                progress_cb(frac0 + (frac1 - frac0) * (idx / total_i), f"{desc} {idx}/{total_i}")
            yield item
        return

    if report_every is None:
        report_every = max(1, total_i // 20)
    for idx, item in enumerate(iterable, 1):
        if idx == 1 or idx == total_i or idx % report_every == 0:
            msg = f"{desc} {idx}/{total_i}"
            _amr_progress(progress_cb, frac0 + (frac1 - frac0) * (idx / total_i), msg)
            if not _AMR_STEP_REPORT and progress_cb is None:
                print(f"[amr] {msg}", flush=True)
        yield item


def _amr_progress(progress_cb: Any, frac: float, label: str) -> None:
    if progress_cb is not None:
        progress_cb(max(0.0, min(1.0, float(frac))), label)
    if _AMR_STEP_REPORT:
        print(f"[amr] {frac * 100.0:6.2f}%  {label}", flush=True)


@contextlib.contextmanager
def _amr_step_span(progress_cb: Any, frac: float, span: str, label: str, detail: str = ""):
    if _HEADLESS_BATCH:
        yield
        return
    msg = f"{label}: {detail}" if detail else label
    _amr_progress(progress_cb, frac, f"start {msg}")
    t0 = time.perf_counter()
    with _PROF.span(span):
        yield
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _amr_progress(progress_cb, frac, f"done {msg} ({elapsed_ms:.1f} ms)")


def _gl_wait_for_dispatch(label: str, progress_cb: Any = None, frac: float = 0.0) -> None:
    from OpenGL.GL import (
        glClientWaitSync, glDeleteSync, glFenceSync, glFlush, glMemoryBarrier,
        GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED, GL_SYNC_FLUSH_COMMANDS_BIT,
        GL_SYNC_GPU_COMMANDS_COMPLETE, GL_TIMEOUT_EXPIRED, GL_WAIT_FAILED,
        GL_SHADER_STORAGE_BARRIER_BIT, GL_BUFFER_UPDATE_BARRIER_BIT,
    )
    timeout_s = float(os.environ.get("SPECTRAL_GL_WAIT_TIMEOUT_SEC", "30"))
    poll_ns = int(float(os.environ.get("SPECTRAL_GL_WAIT_POLL_MS", "100")) * 1_000_000.0)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT)
    sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
    glFlush()
    t0 = time.monotonic()
    last_report = t0
    try:
        while True:
            status = glClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, poll_ns)
            if status in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
                elapsed = time.monotonic() - t0
                _amr_progress(progress_cb, frac, f"{label} GPU complete ({elapsed:.2f}s)")
                return
            if status == GL_WAIT_FAILED:
                raise RuntimeError(f"{label} GPU fence wait failed")
            if status != GL_TIMEOUT_EXPIRED:
                raise RuntimeError(f"{label} GPU fence returned unexpected status {int(status)}")
            now = time.monotonic()
            elapsed = now - t0
            if now - last_report >= 1.0:
                _amr_progress(progress_cb, frac, f"{label} GPU still running ({elapsed:.0f}s)")
                last_report = now
            if elapsed > timeout_s:
                raise TimeoutError(
                    f"{label} GPU dispatch did not complete within {timeout_s:.1f}s; "
                    "the lock is before CPU readback, inside the dispatched GPU work"
                )
    finally:
        glDeleteSync(sync)


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

    @classmethod
    def build_uniform_box(
        cls,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        dx: float,
    ) -> "AcousticAMRGrid":
        """Build a fully uniform 3-D box grid — for benchmarks and unit tests.

        All cells are level-0 acoustic cells (type 0). Faces are interior only
        (open domain — no PML, no boundary cells).  The face distance for every
        face equals ``dx`` (cell-centre to cell-centre for uniform spacing).
        """
        dx = float(dx)
        nx = max(1, round((x_max - x_min) / dx))
        ny = max(1, round((y_max - y_min) / dx))
        nz = max(1, round((z_max - z_min) / dx))
        n_cells = nx * ny * nz

        # ── Cell centres (column-major flattening: i fastest = C order ij) ──
        xs = x_min + dx * (np.arange(nx) + 0.5)
        ys = y_min + dx * (np.arange(ny) + 0.5)
        zs = z_min + dx * (np.arange(nz) + 0.5)
        IX, IY, IZ = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        cc = np.stack([xs[IX.ravel()], ys[IY.ravel()], zs[IZ.ravel()]], axis=1).astype(np.float64)

        # Flat cell index: i*ny*nz + j*nz + k
        stride_y = nz
        stride_x = ny * nz

        # ── X-direction faces: (i,j,k) — (i+1,j,k) for i in [0,nx-2] ──
        if nx > 1:
            Ix = np.arange(nx - 1)
            Jx = np.arange(ny)
            Kx = np.arange(nz)
            FIx, FJx, FKx = np.meshgrid(Ix, Jx, Kx, indexing="ij")
            xf_neg = (FIx * stride_x + FJx * stride_y + FKx).ravel().astype(np.int32)
            xf_pos = xf_neg + stride_x
            xf_ax  = np.zeros(len(xf_neg), dtype=np.uint8)
        else:
            xf_neg = xf_pos = xf_ax = np.empty(0, dtype=np.int32)

        # ── Y-direction faces: (i,j,k) — (i,j+1,k) for j in [0,ny-2] ──
        if ny > 1:
            Iy = np.arange(nx)
            Jy = np.arange(ny - 1)
            Ky = np.arange(nz)
            FIy, FJy, FKy = np.meshgrid(Iy, Jy, Ky, indexing="ij")
            yf_neg = (FIy * stride_x + FJy * stride_y + FKy).ravel().astype(np.int32)
            yf_pos = yf_neg + stride_y
            yf_ax  = np.ones(len(yf_neg), dtype=np.uint8)
        else:
            yf_neg = yf_pos = np.empty(0, dtype=np.int32)
            yf_ax  = np.empty(0, dtype=np.uint8)

        # ── Z-direction faces: (i,j,k) — (i,j,k+1) for k in [0,nz-2] ──
        if nz > 1:
            Iz = np.arange(nx)
            Jz = np.arange(ny)
            Kz = np.arange(nz - 1)
            FIz, FJz, FKz = np.meshgrid(Iz, Jz, Kz, indexing="ij")
            zf_neg = (FIz * stride_x + FJz * stride_y + FKz).ravel().astype(np.int32)
            zf_pos = zf_neg + 1
            zf_ax  = np.full(len(zf_neg), 2, dtype=np.uint8)
        else:
            zf_neg = zf_pos = np.empty(0, dtype=np.int32)
            zf_ax  = np.empty(0, dtype=np.uint8)

        face_neg = np.concatenate([xf_neg, yf_neg, zf_neg])
        face_pos = np.concatenate([xf_pos, yf_pos, zf_pos])
        face_ax  = np.concatenate([xf_ax,  yf_ax,  zf_ax])
        n_faces  = len(face_neg)

        return cls(
            base_dx              = dx,
            min_dx               = dx,
            max_refinement_level = 0,
            bounds_min           = np.array([x_min, y_min, z_min], dtype=np.float64),
            bounds_max           = np.array([x_max, y_max, z_max], dtype=np.float64),
            cell_centers         = cc,
            cell_half_sizes      = np.full(n_cells, dx / 2.0, dtype=np.float64),
            cell_levels          = np.zeros(n_cells, dtype=np.int16),
            cell_types           = np.zeros(n_cells, dtype=np.uint8),
            importance           = np.zeros(n_cells, dtype=np.uint8),
            cell_volumes         = np.full(n_cells, dx ** 3, dtype=np.float64),
            open_volume_fraction = np.ones(n_cells, dtype=np.float64),
            face_cell_neg        = face_neg,
            face_cell_pos        = face_pos,
            face_axis            = face_ax,
            face_area            = np.full(n_faces, dx ** 2, dtype=np.float64),
            face_open_fraction   = np.ones(n_faces, dtype=np.float64),
            face_distance        = np.full(n_faces, dx, dtype=np.float64),
            soundhole            = (0.0, 0.0, 0.0),
            metadata             = {},
        )


class AcousticAMRFDTD:
    """Strict Python handle for the C++/Eigen AMR acoustic stepper."""

    def __init__(
        self,
        grid: AcousticAMRGrid,
        c: float = 343.0,
        rho_air: float = 1.21,
        gradient_order: int = 2,
    ):
        if gradient_order not in (2, 8):
            raise ValueError(f"gradient_order must be 2 or 8, got {gradient_order}")
        with _amr_step_span(None, 0.0, "amr.cpu_backend.init",
                            "create C++ AMR backend", f"cells={grid.n_cells} faces={grid.n_faces}"):
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
                int(gradient_order),
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
        with _amr_step_span(None, 0.0, "amr.cpu_backend.reset", "reset C++ AMR backend"):
            self._c.reset()

    def inject_pressure(self, xyz: np.ndarray, value: float) -> None:
        with _amr_step_span(None, 0.0, "amr.cpu_backend.inject_pressure",
                            "inject C++ AMR pressure", f"value={float(value):.4g}"):
            self._c.inject_pressure_nearest(np.asarray(xyz, dtype=np.float64).reshape(3), float(value))

    def step(self, n_steps: int = 1) -> None:
        with _amr_step_span(None, 0.0, "amr.cpu_backend.step",
                            "step C++ AMR backend", f"n_steps={int(n_steps)}"):
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


_AMR_CACHE_VERSION = 1


def _amr_cache_dir() -> str:
    base = os.environ.get("SPECTRAL_AMR_CACHE_DIR", "").strip()
    if not base:
        base = os.path.join(os.getcwd(), ".spectral_cache", "amr_grid")
    os.makedirs(base, exist_ok=True)
    return base


def _amr_cache_policy_dict(policy: AMRImportancePolicy) -> dict[str, int]:
    return {
        "aperture": int(policy.aperture),
        "guitar_interior": int(policy.guitar_interior),
        "mesh_intersections": int(policy.mesh_intersections),
        "open_air": int(policy.open_air),
    }


def _amr_outline_hash64(outline: np.ndarray) -> int:
    digest = hashlib.blake2b(np.ascontiguousarray(outline, dtype=np.float64).tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _amr_cache_key_payload(
    *,
    dx: float,
    max_level: int,
    subdivision_k: int,
    pad_cells: int,
    n_pml: int,
    soundhole: tuple[float, float, float],
    body_h: float,
    outline_hash64: int,
    importance_policy: AMRImportancePolicy,
    balance_refinement: bool,
    aperture_band_cells: int,
    amr_backend: str,
) -> dict[str, Any]:
    return {
        "v": _AMR_CACHE_VERSION,
        "dx": float(dx),
        "max_level": int(max_level),
        "subdivision_k": int(subdivision_k),
        "pad_cells": int(pad_cells),
        "n_pml": int(n_pml),
        "soundhole": [float(soundhole[0]), float(soundhole[1]), float(soundhole[2])],
        "body_h": float(body_h),
        "outline_hash64": int(outline_hash64),
        "importance_policy": _amr_cache_policy_dict(importance_policy),
        "balance_refinement": bool(balance_refinement),
        "aperture_band_cells": int(aperture_band_cells),
        "amr_backend": str(amr_backend),
    }


def _amr_cache_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:20]


def _amr_cache_load(path: str) -> AcousticAMRGrid | None:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta_raw = str(z["cache_meta_json"].item())
            meta = json.loads(meta_raw)
            if int(meta.get("cache_version", -1)) != _AMR_CACHE_VERSION:
                return None

            policy_raw = meta.get("importance_policy", {})
            policy = AMRImportancePolicy(
                aperture=int(policy_raw.get("aperture", 0)),
                guitar_interior=int(policy_raw.get("guitar_interior", 0)),
                mesh_intersections=int(policy_raw.get("mesh_intersections", 0)),
                open_air=int(policy_raw.get("open_air", 0)),
            )

            grid = AcousticAMRGrid(
                base_dx=float(z["base_dx"].item()),
                min_dx=float(z["min_dx"].item()),
                max_refinement_level=int(z["max_refinement_level"].item()),
                bounds_min=np.ascontiguousarray(z["bounds_min"], dtype=np.float64),
                bounds_max=np.ascontiguousarray(z["bounds_max"], dtype=np.float64),
                cell_centers=np.ascontiguousarray(z["cell_centers"], dtype=np.float64),
                cell_half_sizes=np.ascontiguousarray(z["cell_half_sizes"], dtype=np.float64),
                cell_levels=np.ascontiguousarray(z["cell_levels"], dtype=np.int16),
                cell_types=np.ascontiguousarray(z["cell_types"], dtype=np.uint8),
                importance=np.ascontiguousarray(z["importance"], dtype=np.uint8),
                cell_volumes=np.ascontiguousarray(z["cell_volumes"], dtype=np.float64),
                open_volume_fraction=np.ascontiguousarray(z["open_volume_fraction"], dtype=np.float64),
                face_cell_neg=np.ascontiguousarray(z["face_cell_neg"], dtype=np.int32),
                face_cell_pos=np.ascontiguousarray(z["face_cell_pos"], dtype=np.int32),
                face_axis=np.ascontiguousarray(z["face_axis"], dtype=np.uint8),
                face_area=np.ascontiguousarray(z["face_area"], dtype=np.float64),
                face_open_fraction=np.ascontiguousarray(z["face_open_fraction"], dtype=np.float64),
                face_distance=np.ascontiguousarray(z["face_distance"], dtype=np.float64),
                soundhole=(
                    float(z["soundhole"][0]),
                    float(z["soundhole"][1]),
                    float(z["soundhole"][2]),
                ),
                metadata={
                    "base_dims": np.asarray(meta.get("base_dims", [0, 0, 0]), dtype=np.int32),
                    "importance_policy": policy,
                    "pad_cells": int(meta.get("pad_cells", 0)),
                    "n_pml": int(meta.get("n_pml", 0)),
                    "cache_hit": True,
                    "cache_key": str(meta.get("cache_key", "")),
                },
            )
            return grid
    except Exception as exc:
        print(f"[AMR grid cache] load failed ({path}): {exc}")
        return None


def _amr_cache_save(path: str, grid: AcousticAMRGrid, cache_meta: dict[str, Any]) -> None:
    policy = grid.metadata.get("importance_policy", default_importance_policy(grid.max_refinement_level))
    if not isinstance(policy, AMRImportancePolicy):
        policy = default_importance_policy(grid.max_refinement_level)

    meta = {
        "cache_version": _AMR_CACHE_VERSION,
        "cache_key": str(cache_meta.get("cache_key", "")),
        "base_dims": [int(x) for x in np.asarray(grid.metadata.get("base_dims", [0, 0, 0]), dtype=np.int32)],
        "pad_cells": int(grid.metadata.get("pad_cells", cache_meta.get("pad_cells", 0))),
        "n_pml": int(grid.metadata.get("n_pml", cache_meta.get("n_pml", 0))),
        "importance_policy": _amr_cache_policy_dict(policy),
    }

    tmp_base = path + ".tmp"
    np.savez(
        tmp_base,
        cache_meta_json=np.asarray(json.dumps(meta, sort_keys=True), dtype=np.str_),
        base_dx=np.asarray(grid.base_dx, dtype=np.float64),
        min_dx=np.asarray(grid.min_dx, dtype=np.float64),
        max_refinement_level=np.asarray(grid.max_refinement_level, dtype=np.int32),
        bounds_min=np.ascontiguousarray(grid.bounds_min, dtype=np.float64),
        bounds_max=np.ascontiguousarray(grid.bounds_max, dtype=np.float64),
        cell_centers=np.ascontiguousarray(grid.cell_centers, dtype=np.float64),
        cell_half_sizes=np.ascontiguousarray(grid.cell_half_sizes, dtype=np.float64),
        cell_levels=np.ascontiguousarray(grid.cell_levels, dtype=np.int16),
        cell_types=np.ascontiguousarray(grid.cell_types, dtype=np.uint8),
        importance=np.ascontiguousarray(grid.importance, dtype=np.uint8),
        cell_volumes=np.ascontiguousarray(grid.cell_volumes, dtype=np.float64),
        open_volume_fraction=np.ascontiguousarray(grid.open_volume_fraction, dtype=np.float64),
        face_cell_neg=np.ascontiguousarray(grid.face_cell_neg, dtype=np.int32),
        face_cell_pos=np.ascontiguousarray(grid.face_cell_pos, dtype=np.int32),
        face_axis=np.ascontiguousarray(grid.face_axis, dtype=np.uint8),
        face_area=np.ascontiguousarray(grid.face_area, dtype=np.float64),
        face_open_fraction=np.ascontiguousarray(grid.face_open_fraction, dtype=np.float64),
        face_distance=np.ascontiguousarray(grid.face_distance, dtype=np.float64),
        soundhole=np.asarray(grid.soundhole, dtype=np.float64),
    )
    os.replace(tmp_base + ".npz", path)


# ---------------------------------------------------------------------------
# Stencil disk cache helpers
# ---------------------------------------------------------------------------

def _stencil_cache_path(grid: "AcousticAMRGrid", sw: int) -> str:
    """Return the .npz path for this grid's stencil cache (sw taps per side)."""
    key = ""
    if isinstance(getattr(grid, "metadata", None), dict):
        key = str(grid.metadata.get("cache_key", ""))
    if not key:
        # Derive from face topology when the grid wasn't loaded from cache.
        data = np.concatenate([
            np.ascontiguousarray(grid.face_cell_neg, dtype=np.int32).ravel(),
            np.ascontiguousarray(grid.face_cell_pos, dtype=np.int32).ravel(),
        ])
        key = hashlib.sha256(data.tobytes()).hexdigest()[:20]
    return os.path.join(_amr_cache_dir(), f"{key}_stencil_sw{sw}.npz")


def _stencil_cache_load(path: str) -> "tuple[np.ndarray, np.ndarray] | None":
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            s_cells = np.ascontiguousarray(z["face_s_cells"], dtype=np.int32)
            s_coeff = np.ascontiguousarray(z["face_s_coeff"], dtype=np.float32)
        return s_cells, s_coeff
    except Exception as exc:
        print(f"[AMR stencil cache] load failed ({path}): {exc}")
        return None


def _stencil_cache_save(path: str, s_cells: np.ndarray, s_coeff: np.ndarray) -> None:
    try:
        tmp = path + ".tmp"
        np.savez(tmp,
                 face_s_cells=np.ascontiguousarray(s_cells, dtype=np.int32),
                 face_s_coeff=np.ascontiguousarray(s_coeff, dtype=np.float32))
        os.replace(tmp + ".npz", path)
    except Exception as exc:
        print(f"[AMR stencil cache] save failed ({path}): {exc}")


def _spread_morton3_21(v: np.ndarray) -> np.ndarray:
    """Spread 21 low bits so three coordinates can be interleaved into uint64."""
    x = np.asarray(v, dtype=np.uint64) & np.uint64(0x1fffff)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1f00000000ffff)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1f0000ff0000ff)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100f00f00f00f00f)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10c30c30c30c30c3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def _morton_reorder_cells(cells: dict[str, np.ndarray], progress_cb: Any = None) -> dict[str, np.ndarray]:
    """Sort AMR leaf-cell arrays by Z-curve order before topology/CSR build."""
    centers = np.asarray(cells["centers"])
    n = int(len(centers))
    if n <= 1:
        return cells

    t0 = time.perf_counter()
    half_sizes = np.asarray(cells["half_sizes"], dtype=np.float64)
    widths = 2.0 * half_sizes
    pos_w = widths[np.isfinite(widths) & (widths > 0.0)]
    if pos_w.size == 0:
        return cells

    key_scale = float(np.min(pos_w))
    origin = np.asarray(centers, dtype=np.float64).min(axis=0)
    ijk = np.rint((np.asarray(centers, dtype=np.float64) - origin) / key_scale).astype(np.int64)
    ijk = np.clip(ijk, 0, (1 << 21) - 1).astype(np.uint64, copy=False)
    morton = (
        _spread_morton3_21(ijk[:, 0])
        | (_spread_morton3_21(ijk[:, 1]) << np.uint64(1))
        | (_spread_morton3_21(ijk[:, 2]) << np.uint64(2))
    )
    order = np.argsort(morton, kind="stable")
    if np.all(order == np.arange(n, dtype=order.dtype)):
        return cells

    reordered = {name: np.ascontiguousarray(np.asarray(values)[order])
                 for name, values in cells.items()}
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _amr_progress(progress_cb, 0.785,
                  f"Morton reordered {n} AMR cells for cache locality ({elapsed_ms:.1f} ms)")
    return reordered


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
    progress_cb: Any = None,
    amr_backend: str = "cpu",
    cache_grid: bool = False,
    outline_hash: int | None = None,
) -> AcousticAMRGrid:
    """Build a strict AMR grid for guitar-body acoustics.

    ``dx`` is the coarsest low-priority spacing. Refined cells use
    ``dx / subdivision_k**level``. Each base cell with level L is split into
    a uniform k×k×k grid at each of the L subdivision steps, yielding
    ``k**L`` child cells along each axis (``k**3L`` leaf cells total).
    ``subdivision_k=2`` is the classical octree; ``subdivision_k=3`` gives
    27-way splits per level; any integer >= 2 is valid.

    ``cache_grid`` and ``outline_hash`` are accepted for bridge-layer
    compatibility; caching is optional and may be handled externally.
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
    sub_k = int(subdivision_k)
    if sub_k < 2:
        raise ValueError("subdivision_k must be >= 2")
    with _amr_step_span(progress_cb, 0.02, "amr.grid.validate_inputs",
                        "validate AMR inputs",
                        f"dx={base_dx:g} max_level={max_level} subdivision={sub_k}"):
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

    cache_path = ""
    cache_key = ""
    if cache_grid:
        with _amr_step_span(progress_cb, 0.05, "amr.grid.cache.lookup", "lookup AMR grid cache"):
            outline_h64 = int(outline_hash) if outline_hash is not None else _amr_outline_hash64(outline)
            payload = _amr_cache_key_payload(
                dx=base_dx,
                max_level=max_level,
                subdivision_k=sub_k,
                pad_cells=int(pad_cells),
                n_pml=int(n_pml),
                soundhole=(cx, cy, hr),
                body_h=body_h,
                outline_hash64=outline_h64,
                importance_policy=policy,
                balance_refinement=balance_refinement,
                aperture_band_cells=int(aperture_band_cells),
                amr_backend=amr_backend,
            )
            cache_key = _amr_cache_key(payload)
            cache_path = os.path.join(_amr_cache_dir(), f"{cache_key}.npz")
            grid_cached = _amr_cache_load(cache_path)
            if grid_cached is not None:
                with _amr_step_span(progress_cb, 0.98, "amr.grid.validate_grid",
                                    "validate AMR grid", f"cells={grid_cached.n_cells} faces={grid_cached.n_faces} (cache)"):
                    validate_amr_grid(grid_cached, policy)
                _amr_progress(progress_cb, 1.0,
                              f"complete AMR grid cells={grid_cached.n_cells} faces={grid_cached.n_faces} (cache hit)")
                return grid_cached

    with _amr_step_span(progress_cb, 0.08, "amr.grid.allocate_base_arrays",
                        "allocate AMR base arrays", f"dims={tuple(int(x) for x in dims)}"):
        levels = np.full(tuple(dims), policy.open_air, dtype=np.int16)
        importance = np.full(tuple(dims), AMR_LOW, dtype=np.uint8)
        base_type = np.full(tuple(dims), AMR_AIR, dtype=np.uint8)

    with _amr_step_span(progress_cb, 0.14, "amr.grid.classify_xy",
                        "classify XY outline and aperture",
                        f"outline_pts={len(outline)} soundhole=({cx:.4g},{cy:.4g},{hr:.4g})"):
        xs = bmin[0] + (np.arange(dims[0]) + 0.5) * base_dx
        ys = bmin[1] + (np.arange(dims[1]) + 0.5) * base_dx
        zs = bmin[2] + (np.arange(dims[2]) + 0.5) * base_dx
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        inside_xy = _pip_grid(X, Y, outline)

        soundhole_dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        in_soundhole = soundhole_dist < hr
        near_soundhole_rim = np.abs(soundhole_dist - hr) <= aperture_band_cells * base_dx
        crosses_outline = _base_outline_cross_mask(bmin, base_dx, dims, outline)

    with _amr_step_span(progress_cb, 0.32, "amr.grid.classify_base_cells",
                        "classify base cells",
                        f"base_cells={int(np.prod(dims))} pml={int(n_pml)}"):
        for i in _progress_iter(range(int(dims[0])), total=int(dims[0]),
                                desc="AMR classify base X", unit="x",
                                progress_cb=progress_cb, frac0=0.32, frac1=0.40):
            for j in range(dims[1]):
                inside = bool(inside_xy[i, j])
                hole = bool(in_soundhole[i, j])
                rim = bool(near_soundhole_rim[i, j])
                for iz in range(dims[2]):
                    z = float(zs[iz])
                    in_body_z = (-0.5 * base_dx <= z <= body_h + 0.5 * base_dx)
                    near_top = abs(z - body_h) <= 0.75 * base_dx
                    near_back = abs(z - 0.0) <= 0.75 * base_dx

                    if _is_pml_base(i, j, iz, dims, int(n_pml)):
                        base_type[i, j, iz] = AMR_PML

                    if in_body_z and not inside:
                        base_type[i, j, iz] = AMR_WALL
                    elif inside and near_top and not hole:
                        base_type[i, j, iz] = AMR_PLATE
                    elif inside and near_back:
                        base_type[i, j, iz] = AMR_WALL

                    if inside and 0.0 < z < body_h:
                        levels[i, j, iz] = max(levels[i, j, iz], policy.guitar_interior)
                        importance[i, j, iz] = max(importance[i, j, iz], AMR_GUITAR_INTERIOR)

                    if crosses_outline[i, j] and in_body_z:
                        levels[i, j, iz] = max(levels[i, j, iz], policy.mesh_intersections)
                        importance[i, j, iz] = max(importance[i, j, iz], AMR_MESH_INTERSECTION)

                    if inside and (near_top or near_back):
                        levels[i, j, iz] = max(levels[i, j, iz], policy.mesh_intersections)
                        importance[i, j, iz] = max(importance[i, j, iz], AMR_MESH_INTERSECTION)

                    if inside and (hole or rim) and abs(z - body_h) <= (aperture_band_cells + 1) * base_dx:
                        levels[i, j, iz] = max(levels[i, j, iz], policy.aperture)
                        importance[i, j, iz] = AMR_APERTURE
                        if hole and near_top:
                            base_type[i, j, iz] = AMR_AIR

        if not np.any(importance == AMR_APERTURE):
            raise ValueError("AMR classification produced no aperture cells")
        if not np.any(importance == AMR_GUITAR_INTERIOR):
            raise ValueError("AMR classification produced no guitar interior cells")

    with _amr_step_span(progress_cb, 0.42, "amr.grid.balance_base_levels",
                        "balance base refinement levels",
                        f"level_range={int(levels.min())}..{int(levels.max())}"):
        _balance_base_levels(levels, progress_cb=progress_cb)

    use_gl = (amr_backend == "gl")

    with _amr_step_span(progress_cb, 0.62, "amr.grid.subdivide_leaf_cells",
                        "subdivide AMR leaf cells",
                        f"subdivision={sub_k} max_level={max_level} backend={amr_backend}"):
        _subdivide_fn = _subdivide_base_cells if use_gl else _subdivide_base_cells_cpu
        cells = _subdivide_fn(
            levels, importance, base_type, bmin, base_dx, outline, body_h, soundhole, int(n_pml), dims,
            subdivision_k=sub_k,
            progress_cb=progress_cb,
        )

    with _amr_step_span(progress_cb, 0.785, "amr.grid.morton_reorder",
                        "Morton reorder AMR cells",
                        f"cells={len(cells['centers'])}"):
        cells = _morton_reorder_cells(cells, progress_cb=progress_cb)

    with _amr_step_span(progress_cb, 0.80, "amr.grid.build_faces",
                        "build AMR face topology",
                        f"cells={len(cells['centers'])} backend={amr_backend}"):
        # Optimization 5: disable GL all-pairs topology for large grids
        n_cells = len(cells['centers'])
        use_gl_for_faces = use_gl
        if use_gl and n_cells > 100_000:
            print(f"[AMR] Disabling GL all-pairs face topology for {n_cells:,d} cells; "
                  "using CPU sorted topology (O(n log n) instead of O(n²))")
            use_gl_for_faces = False
        
        _faces_fn = _build_faces if use_gl_for_faces else _build_faces_sorted
        faces = _faces_fn(cells, progress_cb=progress_cb)

    with _amr_step_span(progress_cb, 0.90, "amr.grid.package_grid",
                        "package AMR grid",
                        f"cells={len(cells['centers'])} faces={len(faces['neg'])}"):
        grid = AcousticAMRGrid(
            base_dx=base_dx,
            min_dx=base_dx / float(sub_k ** max_level),
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
                "cache_key": cache_key,
                "morton_ordered": True,
            },
        )

    with _amr_step_span(progress_cb, 0.98, "amr.grid.validate_grid",
                        "validate AMR grid", f"cells={grid.n_cells} faces={grid.n_faces}"):
        validate_amr_grid(grid, policy)
    if cache_grid and cache_path:
        with _amr_step_span(progress_cb, 0.995, "amr.grid.cache.save", "save AMR grid cache"):
            _amr_cache_save(
                cache_path,
                grid,
                {
                    "cache_key": cache_key,
                    "pad_cells": int(pad_cells),
                    "n_pml": int(n_pml),
                },
            )
    _amr_progress(progress_cb, 1.0, f"complete AMR grid cells={grid.n_cells} faces={grid.n_faces}")
    return grid


def validate_amr_grid(grid: AcousticAMRGrid, policy: AMRImportancePolicy | None = None) -> None:
    if grid.n_cells <= 0:
        raise ValueError("AMR grid has no cells")
    if grid.n_faces <= 0:
        raise ValueError("AMR grid has no faces")
    if not np.all(grid.cell_volumes > 0.0):
        raise ValueError("AMR grid has non-positive cell volumes")
    if not np.all(grid.face_area > 0.0):
        bad = ~np.isfinite(grid.face_area) | (grid.face_area <= 0.0)
        raise ValueError(
            "AMR grid has non-positive face areas "
            f"(bad={int(np.count_nonzero(bad))}/{grid.n_faces}, "
            f"min={float(np.nanmin(grid.face_area)) if grid.n_faces else float('nan'):.6g})"
        )
    if not np.all(grid.face_distance > 0.0):
        bad = ~np.isfinite(grid.face_distance) | (grid.face_distance <= 0.0)
        raise ValueError(
            "AMR grid has non-positive face distances "
            f"(bad={int(np.count_nonzero(bad))}/{grid.n_faces}, "
            f"min={float(np.nanmin(grid.face_distance)) if grid.n_faces else float('nan'):.6g})"
        )
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


def _base_outline_cross_mask(bmin: np.ndarray, dx: float, dims: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """Return a 2-D mask for base cells whose XY corners straddle the outline."""
    nx, ny = int(dims[0]), int(dims[1])
    x_edges = bmin[0] + np.arange(nx + 1, dtype=np.float64) * dx
    y_edges = bmin[1] + np.arange(ny + 1, dtype=np.float64) * dx
    Xc, Yc = np.meshgrid(x_edges, y_edges, indexing="ij")
    inside = _pip_grid(Xc, Yc, outline)
    c00 = inside[:-1, :-1]
    c10 = inside[1:, :-1]
    c01 = inside[:-1, 1:]
    c11 = inside[1:, 1:]
    any_inside = c00 | c10 | c01 | c11
    all_inside = c00 & c10 & c01 & c11
    return any_inside & ~all_inside


def _balance_base_levels(levels: np.ndarray, progress_cb: Any = None) -> None:
    dims = levels.shape
    changed = True
    pass_i = 0
    bar = None if progress_cb is not None else _tqdm_or_none(
        desc="AMR balance passes", unit="pass", leave=False
    )
    while changed:
        pass_i += 1
        _amr_progress(progress_cb, 0.42, f"start AMR balance pass {pass_i}")
        changed = False
        changes = 0
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
                                changes += 1
        if bar is not None:
            bar.set_postfix(changes=changes)
            bar.update(1)
        _amr_progress(progress_cb, 0.42, f"done AMR balance pass {pass_i} changes={changes}")
    if bar is not None:
        bar.close()


_AMR_SUBDIVIDE_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer LevelBuf { int levels[]; };
layout(std430, binding = 1) readonly buffer ImportanceBuf { int importance[]; };
layout(std430, binding = 2) readonly buffer OutlineBuf { vec2 outline[]; };
layout(std430, binding = 3) readonly buffer BaseOffsetBuf { int base_offsets[]; };

layout(std430, binding = 4) buffer CenterBuf { vec4 out_center[]; };
layout(std430, binding = 5) buffer HalfBuf { vec4 out_half[]; };
layout(std430, binding = 6) buffer MetaOutBuf { ivec4 out_meta[]; };
layout(std430, binding = 7) buffer MetricOutBuf { vec4 out_metric[]; };

uniform int dims_x;
uniform int dims_y;
uniform int dims_z;
uniform int subdivision_k;
uniform int n_outline;
uniform int n_pml;
uniform float base_dx;
uniform float body_h;
uniform vec3 bounds_min;
uniform vec3 soundhole;

int pow_i(int b, int e) {
    int v = 1;
    for (int i = 0; i < e; ++i) v *= b;
    return v;
}

bool is_pml_base(int i, int j, int k) {
    return i < n_pml || j < n_pml || k < n_pml ||
           i >= dims_x - n_pml || j >= dims_y - n_pml || k >= dims_z - n_pml;
}

bool pip_point(float x, float y) {
    bool inside = false;
    int j = n_outline - 1;
    for (int i = 0; i < n_outline; ++i) {
        vec2 pi = outline[i];
        vec2 pj = outline[j];
        bool crosses = ((pi.y > y) != (pj.y > y));
        float xint = (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y + 1e-15) + pi.x;
        if (crosses && x < xint) inside = !inside;
        j = i;
    }
    return inside;
}

void main() {
    uint base_idx = gl_GlobalInvocationID.x;
    int n_base = dims_x * dims_y * dims_z;
    if (base_idx >= uint(n_base)) return;
    int base_i = int(base_idx);

    int iz = base_i % dims_z;
    int tmp = base_i / dims_z;
    int j = tmp % dims_y;
    int i = tmp / dims_y;

    int lv = levels[base_i];
    int n_child = pow_i(subdivision_k, lv);
    int n_child3 = n_child * n_child * n_child;
    float child_dx = base_dx / float(n_child);
    int dst0 = base_offsets[base_i];

    for (int local = 0; local < n_child3; ++local) {
        int a = local / (n_child * n_child);
        int rem = local - a * n_child * n_child;
        int b = rem / n_child;
        int c = rem - b * n_child;

        vec3 center = bounds_min + vec3(
            float(i) * base_dx + (float(a) + 0.5) * child_dx,
            float(j) * base_dx + (float(b) + 0.5) * child_dx,
            float(iz) * base_dx + (float(c) + 0.5) * child_dx
        );

        bool inside = pip_point(center.x, center.y);
        float hx = center.x - soundhole.x;
        float hy = center.y - soundhole.y;
        bool hole = (hx * hx + hy * hy) < soundhole.z * soundhole.z;
        bool in_body_z = (-0.5 * child_dx <= center.z) && (center.z <= body_h + 0.5 * child_dx);
        bool near_top = abs(center.z - body_h) <= 0.5 * child_dx;
        bool near_back = abs(center.z) <= 0.5 * child_dx;

        int ctype = 0;
        if (in_body_z && !inside) ctype = 1;
        if (inside && near_top && !hole) ctype = 2;
        if (inside && near_back) ctype = 1;
        if (ctype == 0 && is_pml_base(i, j, iz)) ctype = 3;

        int dst = dst0 + local;
        out_center[dst] = vec4(center, 0.0);
        out_half[dst] = vec4(vec3(0.5 * child_dx), 0.0);
        out_meta[dst] = ivec4(lv, ctype, importance[base_i], 0);
        out_metric[dst] = vec4(child_dx * child_dx * child_dx,
                               (ctype == 1 || ctype == 2) ? 0.0 : 1.0,
                               0.0, 0.0);
    }
}
"""


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
    progress_cb: Any = None,
) -> dict[str, np.ndarray]:
    """Split every base cell into a subdivision_k×subdivision_k×subdivision_k
    uniform grid at each of its assigned refinement levels.  A base cell at
    level L produces n = subdivision_k**L leaf cells per axis.
    """
    with _amr_step_span(progress_cb, 0.621, "amr.grid.subdivide_leaf_cells.compute",
                        "compute AMR leaf subdivision on GPU",
                        f"base_cells={int(np.prod(dims))} max_level={int(levels.max())}"):
        import ctypes
        from OpenGL.GL import (
            glBindBuffer, glDeleteBuffers, glDispatchCompute, glGetBufferSubData,
            glMemoryBarrier, glUseProgram,
            GL_SHADER_STORAGE_BARRIER_BIT, GL_SHADER_STORAGE_BUFFER,
        )
        from OpenGL.GL import glBindBufferBase

        _amr_progress(progress_cb, 0.623, "subdivision prepare CPU descriptors")
        dims_i = np.asarray(dims, dtype=np.int32)
        max_level = int(levels.max())
        n_base = int(np.prod(dims_i))
        outline2 = np.asarray(outline, dtype=np.float32)
        if len(outline2) < 3:
            raise ValueError("AMR subdivision compute requires at least 3 outline points")

        levels_flat = np.ascontiguousarray(levels.ravel(), dtype=np.int32)
        importance_flat = np.ascontiguousarray(importance.ravel(), dtype=np.int32)
        child_axis = int(subdivision_k) ** levels_flat.astype(np.int64)
        child_counts64 = np.asarray(child_axis, dtype=np.int64) ** 3
        max_cells64 = int(np.sum(child_counts64, dtype=np.int64))
        if max_cells64 <= 0:
            raise ValueError("AMR subdivision compute produced no requested leaf cells")
        if max_cells64 > np.iinfo(np.int32).max:
            raise RuntimeError(
                f"AMR subdivision would produce {max_cells64} cells; "
                "split the grid or lower refinement before allocating SSBOs"
            )
        max_cells = int(max_cells64)
        base_offsets = np.empty(n_base, dtype=np.int32)
        base_offsets[:] = (
            np.cumsum(child_counts64, dtype=np.int64) - child_counts64
        ).astype(np.int32)
        vec4_bytes = max_cells * 4 * np.dtype(np.float32).itemsize
        ivec4_bytes = max_cells * 4 * np.dtype(np.int32).itemsize

        total_ssbo_mb = (vec4_bytes * 3 + ivec4_bytes) / 1048576.0
        _amr_progress(
            progress_cb, 0.628,
            f"subdivision exact_cells={max_cells} base_cells={n_base} "
            f"max_level={max_level} ssbo_mb={total_ssbo_mb:.1f}"
        )
        _amr_progress(progress_cb, 0.630, "subdivision compile/link compute shader")
        prog = _link_program(_compile_shader(_AMR_SUBDIVIDE_GLSL))
        _amr_progress(progress_cb, 0.635, "subdivision upload compact input SSBOs")
        buf_levels = _make_ssbo(levels_flat, 0)
        buf_importance = _make_ssbo(importance_flat, 1)
        buf_outline = _make_ssbo(outline2, 2)
        buf_offsets = _make_ssbo(base_offsets, 3)
        _amr_progress(progress_cb, 0.640, f"subdivision allocate center SSBO {vec4_bytes / 1048576.0:.1f} MiB")
        buf_center = _make_ssbo_zeros(vec4_bytes)
        _amr_progress(progress_cb, 0.645, f"subdivision allocate half-size SSBO {vec4_bytes / 1048576.0:.1f} MiB")
        buf_half = _make_ssbo_zeros(vec4_bytes)
        _amr_progress(progress_cb, 0.650, f"subdivision allocate meta SSBO {ivec4_bytes / 1048576.0:.1f} MiB")
        buf_meta = _make_ssbo_zeros(ivec4_bytes)
        _amr_progress(progress_cb, 0.655, f"subdivision allocate metric SSBO {vec4_bytes / 1048576.0:.1f} MiB")
        buf_metric = _make_ssbo_zeros(vec4_bytes)
        buffers = [buf_levels, buf_importance, buf_outline, buf_offsets, buf_center,
                   buf_half, buf_meta, buf_metric]
        try:
            for binding, buf in ((4, buf_center), (5, buf_half), (6, buf_meta), (7, buf_metric)):
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf)

            cx, cy, hr = soundhole
            glUseProgram(prog)
            _uniform_i(prog, "dims_x", int(dims_i[0]))
            _uniform_i(prog, "dims_y", int(dims_i[1]))
            _uniform_i(prog, "dims_z", int(dims_i[2]))
            _uniform_i(prog, "subdivision_k", int(subdivision_k))
            _uniform_i(prog, "n_outline", int(len(outline2)))
            _uniform_i(prog, "n_pml", int(n_pml))
            _uniform_f(prog, "base_dx", float(base_dx))
            _uniform_f(prog, "body_h", float(body_h))
            _uniform_vec3(prog, "bounds_min", bmin)
            _uniform_vec3(prog, "soundhole", (cx, cy, hr))

            _amr_progress(progress_cb, 0.665, f"subdivision dispatch shader groups={_gl_ceil_div(n_base, 256)}")
            bar = None if progress_cb is not None else _tqdm_or_none(
                total=1, desc="AMR subdivision shader", unit="dispatch", leave=False
            )
            glDispatchCompute(_gl_ceil_div(n_base, 256), 1, 1)
            _gl_wait_for_dispatch("AMR subdivision shader", progress_cb, 0.685)
            if bar is not None:
                bar.update(1)
            if bar is not None:
                bar.close()

            n_cells = max_cells

            def read_vec3(buf: int) -> np.ndarray:
                arr = np.empty((n_cells, 4), dtype=np.float32)
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, arr.nbytes, ctypes.c_void_p(arr.ctypes.data))
                return arr[:, :3].astype(np.float64)

            def read_ivec4(buf: int) -> np.ndarray:
                arr = np.empty((n_cells, 4), dtype=np.int32)
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, arr.nbytes, ctypes.c_void_p(arr.ctypes.data))
                return arr

            def read_vec4(buf: int) -> np.ndarray:
                arr = np.empty((n_cells, 4), dtype=np.float32)
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, arr.nbytes, ctypes.c_void_p(arr.ctypes.data))
                return arr

            read_bar = None if progress_cb is not None else _tqdm_or_none(
                total=4, desc="AMR subdivision readback", unit="buf", leave=False
            )
            _amr_progress(progress_cb, 0.690, "subdivision readback centers")
            centers = read_vec3(buf_center)
            if read_bar is not None: read_bar.update(1)
            _amr_progress(progress_cb, 0.705, "subdivision readback half sizes")
            half_sizes = read_vec3(buf_half)
            if read_bar is not None: read_bar.update(1)
            _amr_progress(progress_cb, 0.735, "subdivision readback metadata")
            meta = read_ivec4(buf_meta)
            if read_bar is not None: read_bar.update(1)
            leaf_levels = meta[:, 0].astype(np.int16)
            types = meta[:, 1].astype(np.uint8)
            imps = meta[:, 2].astype(np.uint8)
            _amr_progress(progress_cb, 0.765, "subdivision readback metrics")
            metric = read_vec4(buf_metric)
            volumes = metric[:, 0].astype(np.float64)
            open_fracs = metric[:, 1].astype(np.float64)
            if read_bar is not None:
                read_bar.update(1)
                read_bar.close()
        finally:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            glDeleteBuffers(len(buffers), buffers)

    return {
        "centers": centers,
        "half_sizes": half_sizes,
        "levels": leaf_levels,
        "types": types,
        "importance": imps,
        "volumes": volumes,
        "open_fraction": open_fracs,
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


def _subdivide_base_cells_cpu(
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
    progress_cb: Any = None,
) -> dict[str, np.ndarray]:
    """CPU vectorised version of AMR subdivision (no GL required)."""
    _t0_subdivide = time.perf_counter()
    cx_sh, cy_sh, hr_sh = float(soundhole[0]), float(soundhole[1]), float(soundhole[2])
    dims_i = np.asarray(dims, dtype=np.int32)
    n_base = int(np.prod(dims_i))
    outline_f = np.asarray(outline, dtype=np.float64)

    # Flat base-cell coordinate indices
    izs = np.arange(n_base, dtype=np.int32) % dims_i[2]
    tmp = np.arange(n_base, dtype=np.int32) // dims_i[2]
    js  = tmp % dims_i[1]
    i_vals = tmp // dims_i[1]
    is_pml = (
        (i_vals < n_pml) | (js < n_pml) | (izs < n_pml) |
        (i_vals >= dims_i[0] - n_pml) |
        (js  >= dims_i[1] - n_pml) |
        (izs >= dims_i[2] - n_pml)
    )

    base_x = bmin[0] + i_vals.astype(np.float64) * base_dx
    base_y = bmin[1] + js.astype(np.float64)    * base_dx
    base_z = bmin[2] + izs.astype(np.float64)   * base_dx

    # ── Hierarchical PIP pre-filter ──────────────────────────────────────────
    # One PIP pass on all base cell CENTERS gives inside/outside status.
    # Check all 4 XY corners; if all 4 corners agree with the center, the cell
    # is definitively inside or outside — children inherit the status and skip PIP.
    # Only boundary cells (corners disagree) need per-child PIP.
    _t_pip0 = time.perf_counter()
    _hd = base_dx * 0.5
    _bx_ctr = base_x + _hd   # XY centers (add half-cell to get center from corner origin)
    _by_ctr = base_y + _hd
    _base_inside_ctr = _pip_grid(_bx_ctr, _by_ctr, outline_f)
    _c00 = _pip_grid(base_x,        base_y,        outline_f)
    _c10 = _pip_grid(base_x + base_dx, base_y,     outline_f)
    _c01 = _pip_grid(base_x,        base_y + base_dx, outline_f)
    _c11 = _pip_grid(base_x + base_dx, base_y + base_dx, outline_f)
    # A base cell is "solid" (no outline crossing) if all corners agree with the center
    _base_solid = (_c00 == _base_inside_ctr) & (_c10 == _base_inside_ctr) \
                & (_c01 == _base_inside_ctr) & (_c11 == _base_inside_ctr)
    _pip_prefilter_ms = (time.perf_counter() - _t_pip0) * 1000.0
    _n_solid = int(_base_solid.sum())
    _n_bdr   = n_base - _n_solid
    print(f"[amr_subdiv] PIP prefilter: {_n_solid}/{n_base} solid cells, "
          f"{_n_bdr} boundary cells  ({_pip_prefilter_ms:.1f} ms)", flush=True)

    levels_flat     = levels.ravel().astype(np.int32)
    importance_flat = importance.ravel().astype(np.int32)

    child_n_per_axis = np.array([subdivision_k ** int(lv) for lv in levels_flat], dtype=np.int64)
    child_counts     = child_n_per_axis ** 3
    total_cells      = int(np.sum(child_counts))
    base_offsets     = np.zeros(n_base, dtype=np.int64)
    if n_base > 1:
        base_offsets[1:] = np.cumsum(child_counts[:-1])

    _amr_progress(
        progress_cb, 0.628,
        f"subdivision CPU exact_cells={total_cells} base_cells={n_base} "
        f"max_level={int(levels_flat.max())} "
        f"ssbo_mb={total_cells * 64 / 1048576.0:.1f}"
    )

    centers_out = np.empty((total_cells, 3), dtype=np.float32)
    half_out    = np.empty((total_cells, 3), dtype=np.float32)
    levels_out  = np.empty(total_cells,      dtype=np.int16)
    types_out   = np.empty(total_cells,      dtype=np.uint8)
    imps_out    = np.empty(total_cells,      dtype=np.uint8)

    unique_levels = np.unique(levels_flat)
    n_unique = len(unique_levels)

    for li, lv_raw in enumerate(unique_levels):
        lv = int(lv_raw)
        n_child   = subdivision_k ** lv
        child_dx  = base_dx / float(n_child)
        half_f32  = np.float32(child_dx * 0.5)

        mask    = (levels_flat == lv)
        bx      = base_x[mask]
        by      = base_y[mask]
        bz      = base_z[mask]
        imps    = importance_flat[mask]
        pml     = is_pml[mask]
        offsets = base_offsets[mask]
        M       = len(bx)

        _amr_progress(
            progress_cb,
            0.630 + 0.055 * ((li + 1) / n_unique),
            f"subdivision CPU level={lv} base_cells={M}",
        )

        n_child3 = n_child ** 3
        local    = np.arange(n_child3, dtype=np.int32)
        a_idx    = local // (n_child * n_child)
        rem      = local - a_idx * (n_child * n_child)
        b_idx    = rem // n_child
        c_idx    = rem - b_idx * n_child

        off_x = (a_idx.astype(np.float64) + 0.5) * child_dx
        off_y = (b_idx.astype(np.float64) + 0.5) * child_dx
        off_z = (c_idx.astype(np.float64) + 0.5) * child_dx

        cx_all = (bx[:, None] + off_x[None, :])  # (M, n_child3)
        cy_all = (by[:, None] + off_y[None, :])
        cz_all = (bz[:, None] + off_z[None, :])

        cx_flat = cx_all.ravel()
        cy_flat = cy_all.ravel()
        cz_flat = cz_all.ravel()

        # ── Hierarchical PIP: skip expensive ray-casting for solid parent cells ─
        # _base_solid[k] = True means all corners of base cell k agreed — children inherit.
        parent_solid = _base_solid[mask]  # (M,) bool — solid mask for selected cells
        parent_inside = _base_inside_ctr[mask]  # (M,) bool — inside status for solid cells
        if np.any(parent_solid) and not np.all(parent_solid):
            # Mixed batch: PIP only on children of boundary parents
            bdr_mask_m = ~parent_solid                         # (M,) boundary parents
            bdr_child_mask = np.repeat(bdr_mask_m, n_child3)  # (M*n_child3,) child mask
            inside = np.repeat(parent_inside, n_child3)        # default: inherit parent
            if bdr_child_mask.any():
                inside[bdr_child_mask] = _pip_grid(
                    cx_flat[bdr_child_mask], cy_flat[bdr_child_mask], outline_f)
        elif np.all(parent_solid):
            # All parents solid: inherit parent inside status for all children, skip PIP
            inside = np.repeat(parent_inside, n_child3)
        else:
            # All parents are boundary: full PIP as before
            inside = _pip_grid(cx_flat, cy_flat, outline_f)
        hole      = (cx_flat - cx_sh)**2 + (cy_flat - cy_sh)**2 < hr_sh**2
        in_body_z = (-0.5 * child_dx <= cz_flat) & (cz_flat <= body_h + 0.5 * child_dx)
        near_top  = np.abs(cz_flat - body_h) <= 0.5 * child_dx
        near_back = np.abs(cz_flat)           <= 0.5 * child_dx
        pml_flat  = np.repeat(pml, n_child3)
        imps_flat = np.repeat(imps, n_child3).astype(np.uint8)

        ctype = np.zeros(M * n_child3, dtype=np.uint8)
        ctype[in_body_z & ~inside]           = AMR_WALL
        ctype[inside & near_top & ~hole]     = AMR_PLATE
        ctype[inside & near_back]            = AMR_WALL
        ctype[(ctype == AMR_AIR) & pml_flat] = AMR_PML

        dst = (offsets[:, None] + local[None, :].astype(np.int64)).ravel()

        centers_out[dst, 0] = cx_flat.astype(np.float32)
        centers_out[dst, 1] = cy_flat.astype(np.float32)
        centers_out[dst, 2] = cz_flat.astype(np.float32)
        half_out[dst, :]    = half_f32
        levels_out[dst]     = np.int16(lv)
        types_out[dst]      = ctype
        imps_out[dst]       = imps_flat

    _subdiv_total_ms = (time.perf_counter() - _t0_subdivide) * 1000.0
    print(f"[amr_subdiv] total={_subdiv_total_ms:.0f}ms  cells={total_cells:,d}", flush=True)
    _amr_progress(progress_cb, 0.690, "subdivision CPU complete")

    centers_f64 = centers_out.astype(np.float64)
    half_f64    = half_out.astype(np.float64)
    child_dx_arr = 2.0 * half_f64[:, 0]
    volumes      = child_dx_arr ** 3
    open_fracs   = np.where((types_out == AMR_WALL) | (types_out == AMR_PLATE), 0.0, 1.0)

    return {
        "centers":       centers_f64,
        "half_sizes":    half_f64,
        "levels":        levels_out,
        "types":         types_out,
        "importance":    imps_out,
        "volumes":       volumes,
        "open_fraction": open_fracs,
    }


_AMR_FACE_TOPOLOGY_GLSL = """\
#version 430 core
layout(local_size_x = 128) in;

layout(std430, binding = 0) readonly buffer LoKeyBuf { ivec4 lo_key4[];};
layout(std430, binding = 1) readonly buffer HiKeyBuf { ivec4 hi_key4[];};
layout(std430, binding = 2) readonly buffer CenterBuf{ vec4 center4[];};
layout(std430, binding = 3) readonly buffer TypeBuf  { int  type_i[]; };

layout(std430, binding = 5) buffer FaceIBuf   { ivec4 out_i[];        };
layout(std430, binding = 6) buffer FaceFBuf   { vec4  out_f[];        };
layout(std430, binding = 7) buffer CountBuf   { uint face_counts[];   };
layout(std430, binding = 8) readonly buffer OffsetBuf { uint face_offsets[]; };

uniform int   n_cells;
uniform int   axis;
uniform int   mode;  // 0=count, 1=fill
uniform float key_scale;

bool match_face(uint left, uint right, out float area, out float dist, out float open_fraction) {
    ivec3 llo = lo_key4[left].xyz;
    ivec3 lhi = hi_key4[left].xyz;
    ivec3 rlo = lo_key4[right].xyz;
    ivec3 rhi = hi_key4[right].xyz;

    if (lhi[axis] != rlo[axis]) return false;

    int a0 = (axis + 1) % 3;
    int a1 = (axis + 2) % 3;
    int ov0 = max(llo[a0], rlo[a0]);
    int ov1 = min(lhi[a0], rhi[a0]);
    int ov2 = max(llo[a1], rlo[a1]);
    int ov3 = min(lhi[a1], rhi[a1]);
    if (ov1 <= ov0 || ov3 <= ov2) return false;
    area = float(ov1 - ov0) * float(ov3 - ov2) * key_scale * key_scale;
    dist = abs(center4[right][axis] - center4[left][axis]);
    if (!(area > 0.0) || !(dist > 0.0)) return false;

    bool solid = (type_i[left] == 1) || (type_i[right] == 1);
    open_fraction = solid ? 0.0 : 1.0;
    return true;
}

void main() {
    uint left = gl_GlobalInvocationID.x;
    if (left >= uint(n_cells)) return;

    uint slot = uint(axis) * uint(n_cells) + left;
    uint count = 0u;
    float area = 0.0;
    float dist = 0.0;
    float open_fraction = 0.0;

    if (mode == 0) {
        for (uint right = 0u; right < uint(n_cells); ++right) {
            if (right != left && match_face(left, right, area, dist, open_fraction)) {
                ++count;
            }
        }
        face_counts[slot] = count;
        return;
    }

    uint dst = face_offsets[slot];
    for (uint right = 0u; right < uint(n_cells); ++right) {
        if (right == left || !match_face(left, right, area, dist, open_fraction)) continue;
        out_i[dst] = ivec4(int(left), int(right), axis, 0);
        out_f[dst] = vec4(area, open_fraction, dist, 0.0);
        ++dst;
    }
}
"""


def _build_faces(cells: dict[str, np.ndarray], progress_cb: Any = None) -> dict[str, np.ndarray]:
    """Build AMR face topology with a required OpenGL compute shader."""
    with _amr_step_span(progress_cb, 0.801, "amr.grid.build_faces.compute",
                        "compute AMR face topology on GPU",
                        f"cells={len(cells['centers'])}"):
        import ctypes
        from OpenGL.GL import (
            glBindBuffer, glDeleteBuffers,
            glDispatchCompute, glGetBufferSubData, glMemoryBarrier, glUseProgram,
            GL_SHADER_STORAGE_BARRIER_BIT, GL_SHADER_STORAGE_BUFFER,
        )

        _amr_progress(progress_cb, 0.805, "face topology prepare lattice keys")
        centers64 = np.asarray(cells["centers"], dtype=np.float64)
        half64 = np.asarray(cells["half_sizes"], dtype=np.float64)
        types = np.asarray(cells["types"], dtype=np.int32)
        n = int(len(centers64))
        if n <= 0:
            raise ValueError("AMR topology compute requires at least one cell")

        centers = centers64.astype(np.float32)
        cell_widths = np.asarray(2.0 * half64, dtype=np.float64)
        positive_widths = cell_widths[np.isfinite(cell_widths) & (cell_widths > 0.0)]
        if positive_widths.size <= 0:
            raise ValueError("AMR topology compute requires positive cell sizes")
        key_scale = float(np.min(positive_widths))
        if not (np.isfinite(key_scale) and key_scale > 0.0):
            raise ValueError(f"AMR topology compute has invalid lattice scale {key_scale!r}")
        lo_key = np.rint((centers64 - half64) / key_scale).astype(np.int32)
        hi_key = np.rint((centers64 + half64) / key_scale).astype(np.int32)
        bad_key = np.any(hi_key <= lo_key, axis=1)
        if np.any(bad_key):
            n_bad_key = int(np.count_nonzero(bad_key))
            raise ValueError(f"AMR topology compute has {n_bad_key} non-positive lattice cells")
        lo4 = np.zeros((n, 4), dtype=np.int32)
        hi4 = np.zeros((n, 4), dtype=np.int32)
        center4 = np.zeros((n, 4), dtype=np.float32)
        lo4[:, :3] = lo_key
        hi4[:, :3] = hi_key
        center4[:, :3] = centers

        _amr_progress(progress_cb, 0.815, "face topology compile/link compute shader")
        prog = _link_program(_compile_shader(_AMR_FACE_TOPOLOGY_GLSL))
        _amr_progress(progress_cb, 0.820, "face topology upload cell key/type SSBOs")
        buf_lo = _make_ssbo(lo4, 0)
        buf_hi = _make_ssbo(hi4, 1)
        buf_center = _make_ssbo(center4, 2)
        buf_type = _make_ssbo(types, 3)
        counts = np.zeros(n * 3, dtype=np.uint32)
        buf_counts = _make_ssbo(counts, 7)
        buffers = [buf_lo, buf_hi, buf_center, buf_type, buf_counts]
        try:
            glUseProgram(prog)
            _uniform_i(prog, "n_cells", n)
            _uniform_f(prog, "key_scale", key_scale)
            groups = _gl_ceil_div(n, 128)
            count_bar = None if progress_cb is not None else _tqdm_or_none(
                total=3, desc="AMR face count shader", unit="axis", leave=False
            )
            for axis in range(3):
                _uniform_i(prog, "axis", axis)
                _uniform_i(prog, "mode", 0)
                _amr_progress(progress_cb, 0.830 + axis * 0.025, f"dispatch face count axis={axis} groups={groups}")
                glDispatchCompute(groups, 1, 1)
                _gl_wait_for_dispatch(f"AMR face count shader axis={axis}", progress_cb, 0.835 + axis * 0.025)
                if count_bar is not None:
                    count_bar.update(1)
            if count_bar is not None:
                count_bar.close()

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_counts)
            _amr_progress(progress_cb, 0.905, "face topology readback per-cell counts")
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, counts.nbytes, ctypes.c_void_p(counts.ctypes.data))
            n_faces = int(np.sum(counts, dtype=np.uint64))
            _amr_progress(
                progress_cb, 0.910,
                f"face topology counted_faces={n_faces} "
                f"cells={n} lattice={key_scale:.6g}"
            )
            if n_faces > np.iinfo(np.uint32).max:
                raise RuntimeError(
                    f"AMR topology compute produced {n_faces} faces; split the grid before readback"
                )
            if n_faces <= 0:
                raise ValueError("AMR topology compute produced no cell faces")

            offsets64 = np.cumsum(counts.astype(np.uint64), dtype=np.uint64) - counts.astype(np.uint64)
            offsets = offsets64.astype(np.uint32)
            face_i = np.empty((n_faces, 4), dtype=np.int32)
            face_f = np.empty((n_faces, 4), dtype=np.float32)
            _amr_progress(progress_cb, 0.915, f"face topology allocate compact output faces={n_faces}")
            buf_offsets = _make_ssbo(offsets, 8)
            buf_face_i = _make_ssbo_zeros(face_i.nbytes)
            buf_face_f = _make_ssbo_zeros(face_f.nbytes)
            buffers.extend([buf_offsets, buf_face_i, buf_face_f])
            from OpenGL.GL import glBindBufferBase
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, buf_face_i)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, buf_face_f)

            fill_bar = None if progress_cb is not None else _tqdm_or_none(
                total=3, desc="AMR face fill shader", unit="axis", leave=False
            )
            for axis in range(3):
                _uniform_i(prog, "axis", axis)
                _uniform_i(prog, "mode", 1)
                _amr_progress(progress_cb, 0.925 + axis * 0.015, f"dispatch face fill axis={axis} groups={groups}")
                glDispatchCompute(groups, 1, 1)
                _gl_wait_for_dispatch(f"AMR face fill shader axis={axis}", progress_cb, 0.930 + axis * 0.015)
                if fill_bar is not None:
                    fill_bar.update(1)
            if fill_bar is not None:
                fill_bar.close()

            read_bar = None if progress_cb is not None else _tqdm_or_none(
                total=2, desc="AMR face readback", unit="buf", leave=False
            )
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_face_i)
            _amr_progress(progress_cb, 0.970, "face topology readback face indices")
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, face_i.nbytes, ctypes.c_void_p(face_i.ctypes.data))
            if read_bar is not None: read_bar.update(1)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_face_f)
            _amr_progress(progress_cb, 0.975, "face topology readback face metrics")
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, face_f.nbytes, ctypes.c_void_p(face_f.ctypes.data))
            if read_bar is not None:
                read_bar.update(1)
                read_bar.close()

            neg = face_i[:, 0]
            pos = face_i[:, 1]
            axes = face_i[:, 2].astype(np.uint8)
            area = face_f[:, 0].astype(np.float64)
            open_fraction = face_f[:, 1].astype(np.float64)
            distance = face_f[:, 2].astype(np.float64)
            valid = (
                np.isfinite(area) & (area > 0.0) &
                np.isfinite(distance) & (distance > 0.0) &
                (neg >= 0) & (neg < n) & (pos >= 0) & (pos < n) & (neg != pos)
            )
            n_bad = int(valid.size - int(np.count_nonzero(valid)))
            if n_bad:
                _amr_progress(progress_cb, 0.980, f"discarded {n_bad} invalid GPU face records")
                neg = neg[valid]
                pos = pos[valid]
                axes = axes[valid]
                area = area[valid]
                open_fraction = open_fraction[valid]
                distance = distance[valid]
            if len(neg) <= 0:
                raise ValueError(
                    "AMR topology compute produced no valid cell faces "
                    f"(counted={n_faces}, lattice={key_scale:.9g})"
                )
        finally:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            glDeleteBuffers(len(buffers), buffers)

    return {
        "neg": neg,
        "pos": pos,
        "axis": axes,
        "area": area,
        "open_fraction": open_fraction,
        "distance": distance,
    }


def _build_faces_sorted(cells: dict[str, np.ndarray], progress_cb: Any = None) -> dict[str, np.ndarray]:
    """CPU O(n log n) face topology using sorted lattice boundary keys."""
    from collections import defaultdict

    centers   = np.asarray(cells["centers"],   dtype=np.float64)
    half_sizes = np.asarray(cells["half_sizes"], dtype=np.float64)
    types     = np.asarray(cells["types"],     dtype=np.uint8)
    N = len(centers)

    _amr_progress(progress_cb, 0.805, f"face topology CPU sorted-lattice n_cells={N}")
    if os.environ.get("SPECTRAL_AMR_CPP_FACES", "1").lower() not in {"0", "false", "no", "off"}:
        try:
            from _spectral_kernels import build_amr_faces_sorted as _build_faces_cpp
            t_cpp0 = time.perf_counter()
            faces = _build_faces_cpp(
                np.ascontiguousarray(centers, dtype=np.float64),
                np.ascontiguousarray(half_sizes, dtype=np.float64),
                np.ascontiguousarray(types, dtype=np.uint8),
            )
            out = {
                "neg": np.ascontiguousarray(faces["neg"], dtype=np.int32),
                "pos": np.ascontiguousarray(faces["pos"], dtype=np.int32),
                "axis": np.ascontiguousarray(faces["axis"], dtype=np.uint8),
                "area": np.ascontiguousarray(faces["area"], dtype=np.float64),
                "open_fraction": np.ascontiguousarray(faces["open_fraction"], dtype=np.float64),
                "distance": np.ascontiguousarray(faces["distance"], dtype=np.float64),
            }
            elapsed_ms = (time.perf_counter() - t_cpp0) * 1000.0
            _amr_progress(
                progress_cb, 0.895,
                f"face topology C++ complete faces={len(out['neg'])} ({elapsed_ms:.1f} ms)",
            )
            return out
        except Exception as exc:
            print(f"[AMR] C++ face topology unavailable; falling back to Python sorted sweep: {exc}",
                  flush=True)

    widths = 2.0 * half_sizes
    pos_w  = widths[np.isfinite(widths) & (widths > 0.0)]
    if pos_w.size == 0:
        raise ValueError("face topology requires positive cell sizes")
    key_scale = float(np.min(pos_w))

    lo_key = np.rint((centers - half_sizes) / key_scale).astype(np.int64)
    hi_key = np.rint((centers + half_sizes) / key_scale).astype(np.int64)

    neg_list  = []
    pos_list  = []
    ax_list   = []
    area_list = []
    open_list = []
    dist_list = []

    t_topo_start = time.perf_counter()
    total_candidate_checks = 0
    total_axis_elapsed = 0.0

    for ax in range(3):
        axis_t0 = time.perf_counter()
        ax1 = (ax + 1) % 3
        ax2 = (ax + 2) % 3
        _amr_progress(
            progress_cb,
            0.820 + ax * 0.040,
            f"face topology CPU axis={ax} n_cells={N}",
        )

        hi_k = hi_key[:, ax]
        lo_k = lo_key[:, ax]
        hi1 = hi_key[:, ax1]
        lo1 = lo_key[:, ax1]
        hi2 = hi_key[:, ax2]
        lo2 = lo_key[:, ax2]

        axis_faces_start = len(neg_list)
        axis_candidate_checks = 0
        axis_interval1_hits = 0
        axis_groups_with_matches = 0
        axis_t_build0 = time.perf_counter()

        # Build per-plane face records for a swept matcher.
        # High faces are neg-side cells, low faces are pos-side cells.
        hi_planes: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
        lo_planes: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
        for i in range(N):
            hi_planes[int(hi_k[i])].append((int(lo1[i]), int(hi1[i]), int(lo2[i]), int(hi2[i]), i))
            lo_planes[int(lo_k[i])].append((int(lo1[i]), int(hi1[i]), int(lo2[i]), int(hi2[i]), i))
        axis_t_build = time.perf_counter() - axis_t_build0

        n_planes = len(hi_planes)
        next_status_t = time.perf_counter() + 1.0
        axis_t_sweep0 = time.perf_counter()

        for plane_idx, (plane, hi_records) in enumerate(hi_planes.items(), 1):
            lo_records = lo_planes.get(plane)
            if not lo_records:
                continue
            axis_groups_with_matches += 1

            hi_sorted = sorted(hi_records, key=lambda r: (r[0], r[1], r[2], r[3], r[4]))
            lo_start_sorted = sorted(lo_records, key=lambda r: (r[0], r[1], r[2], r[3], r[4]))
            lo_end_sorted = sorted(lo_records, key=lambda r: (r[1], r[0], r[2], r[3], r[4]))

            active = set()
            start_ptr = 0
            end_ptr = 0
            n_lo = len(lo_records)

            for h_lo0, h_hi0, h_lo1, h_hi1, i in hi_sorted:
                if h_hi0 <= h_lo0 or h_hi1 <= h_lo1:
                    continue

                # Retire low records whose interval0 ended before this high starts.
                while end_ptr < n_lo and lo_end_sorted[end_ptr][1] <= h_lo0:
                    active.discard(lo_end_sorted[end_ptr])
                    end_ptr += 1

                # Activate low records whose interval0 starts before this high ends.
                while start_ptr < n_lo and lo_start_sorted[start_ptr][0] < h_hi0:
                    rec = lo_start_sorted[start_ptr]
                    if rec[1] > h_lo0:
                        active.add(rec)
                    start_ptr += 1

                # Only active lows can overlap in interval0; then test interval1.
                for l_lo0, l_hi0, l_lo1, l_hi1, j in active:
                    axis_candidate_checks += 1
                    if i == j:
                        continue
                    if l_hi0 <= h_lo0 or l_lo0 >= h_hi0:
                        continue
                    ov1_lo = max(h_lo1, l_lo1)
                    ov1_hi = min(h_hi1, l_hi1)
                    if ov1_hi <= ov1_lo:
                        continue
                    axis_interval1_hits += 1
                    ov0_lo = max(h_lo0, l_lo0)
                    ov0_hi = min(h_hi0, l_hi0)
                    if ov0_hi <= ov0_lo:
                        continue

                    area = float(ov0_hi - ov0_lo) * float(ov1_hi - ov1_lo) * key_scale * key_scale
                    dist = abs(centers[j, ax] - centers[i, ax])
                    if not (area > 0.0 and dist > 0.0):
                        continue

                    # WALL faces are closed; PLATE faces remain open for moving-boundary coupling.
                    open_frac = 0.0 if (int(types[i]) == AMR_WALL or int(types[j]) == AMR_WALL) else 1.0
                    neg_list.append(i)
                    pos_list.append(j)
                    ax_list.append(ax)
                    area_list.append(area)
                    open_list.append(open_frac)
                    dist_list.append(dist)

            now = time.perf_counter()
            if now >= next_status_t:
                axis_faces_now = len(neg_list) - axis_faces_start
                axis_elapsed_now = now - axis_t0
                _amr_progress(
                    progress_cb,
                    0.830 + ax * 0.040,
                    "still running: face topology CPU "
                    f"axis={ax} plane={plane_idx}/{n_planes} "
                    f"elapsed={axis_elapsed_now:.1f}s candidates={axis_candidate_checks} "
                    f"faces={axis_faces_now}",
                )
                next_status_t = now + 1.0

        axis_t_sweep = time.perf_counter() - axis_t_sweep0
        axis_elapsed = time.perf_counter() - axis_t0
        total_axis_elapsed += axis_elapsed
        total_candidate_checks += axis_candidate_checks
        axis_faces = len(neg_list) - axis_faces_start
        _amr_progress(
            progress_cb,
            0.835 + ax * 0.040,
            "face topology CPU axis summary "
            f"axis={ax} planes={n_planes} matched_planes={axis_groups_with_matches} "
            f"candidates={axis_candidate_checks} interval1_hits={axis_interval1_hits} "
            f"faces={axis_faces} t_build={axis_t_build:.3f}s t_sweep={axis_t_sweep:.3f}s "
            f"t_axis={axis_elapsed:.3f}s",
        )

    n_faces = len(neg_list)
    topo_elapsed = time.perf_counter() - t_topo_start
    _amr_progress(
        progress_cb,
        0.975,
        "face topology CPU summary "
        f"n_cells={N} n_faces={n_faces} candidates={total_candidate_checks} "
        f"t_total={topo_elapsed:.3f}s t_axes={total_axis_elapsed:.3f}s",
    )
    if n_faces <= 0:
        raise ValueError("face topology CPU produced no faces")

    return {
        "neg":           np.array(neg_list,  dtype=np.int32),
        "pos":           np.array(pos_list,  dtype=np.int32),
        "axis":          np.array(ax_list,   dtype=np.uint8),
        "area":          np.array(area_list, dtype=np.float64),
        "open_fraction": np.array(open_list, dtype=np.float64),
        "distance":      np.array(dist_list, dtype=np.float64),
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
    gradient_order: int = 2,
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
    gradient_order : int — 2 or 8. Order 2 uses simple 2-cell gradient (faster, lower memory).
        Order 8 builds Fornberg stencil (~168 MB for 2.6M faces). Default 8.
    neck_plate_idx : optional flat plate indices for neck coupling.
    neck_plate_wgt : optional weights for neck coupling (must match idx length).

    Returns
    -------
    dict with all fields matching AMRCoevolverDescriptor field names.
    """
    plate_origin_arr = np.asarray(plate_origin, dtype=np.float32).reshape(3)

    # Validate gradient_order (optimization 3)
    if gradient_order not in (2, 8):
        raise ValueError(f"gradient_order must be 2 or 8, got {gradient_order}")

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
        # Gradient order for velocity update (optimization 3)
        "gradient_order":      int(gradient_order),
    }

    # Build 8th-order stencil only if gradient_order==8 (optimization 3)
    # For order 2, velocity update uses only face_neg, face_pos, face_distance (already in desc)
    if gradient_order == 8:
        _sw = 4
        _stencil_path = _stencil_cache_path(grid, _sw)
        _loaded = None
        with _amr_step_span(None, 0.0, "amr.gl.stencil.cache.lookup",
                            "lookup stencil cache", f"path={_stencil_path}"):
            _loaded = _stencil_cache_load(_stencil_path)
        if _loaded is not None:
            print(f"[amr_stencil.py] cache hit: {_stencil_path}", flush=True)
            s_cells, s_coeff = _loaded
        else:
            print(f"[amr_stencil.py] cache miss: {_stencil_path}", flush=True)
            s_cells, s_coeff = _build_face_stencil(grid, sw=_sw)
            with _amr_step_span(None, 0.0, "amr.gl.stencil.cache.save",
                                "save stencil cache", f"faces={grid.n_faces} sw={_sw}"):
                _stencil_cache_save(_stencil_path, s_cells, s_coeff)
        desc["face_s_cells"] = np.ascontiguousarray(s_cells, dtype=np.int32)
        desc["face_s_coeff"] = np.ascontiguousarray(s_coeff, dtype=np.float32)
    else:
        # Order 2: add empty placeholders for C++ compatibility
        desc["face_s_cells"] = np.empty(0, dtype=np.int32)
        desc["face_s_coeff"] = np.empty(0, dtype=np.float32)

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
 * Phase 4: face_s_cells padding uses index 0 with coefficient 0 (branchless).
 * Phase 6: plate_owner[f] >= 0 triggers plate-BC early-out.
 */
layout(std430, binding = 0) buffer PressureBuf   { float pressure[];     };
layout(std430, binding = 1) buffer VelocityBuf   { float velocity[];     };
layout(std430, binding = 2) buffer StencilCells  { int   face_s_cells[]; };
layout(std430, binding = 3) buffer StencilCoeff  { float face_s_coeff[]; };
layout(std430, binding = 4) buffer FaceVDampBuf  { float face_v_damp[];  };
layout(std430, binding = 6) buffer PlateOwnerBuf { int   plate_owner[];  };
layout(std430, binding = 7) buffer PlateSignBuf  { float plate_sign[];   };
layout(std430, binding = 8) buffer PlateWgtBuf   { float plate_wgt[];    };
layout(std430, binding = 9) buffer ActiveIdxBuf  { int   active_idx[];   };
layout(std430, binding = 10) buffer PlateWBuf    { float plate_w[];      };
layout(std430, binding = 11) buffer PlateWpBuf   { float plate_w_prev[]; };

uniform float dt_over_rho;
uniform int   n_faces;
uniform float plate_inv_dt;
uniform int   plate_bc_enabled;

void main() {
    uint f = gl_GlobalInvocationID.x;
    if (f >= uint(n_faces)) return;

    if (plate_bc_enabled != 0) {
        int owner = plate_owner[f];
        if (owner >= 0) {
            int p_idx = active_idx[owner];
            float v_plt = (plate_w[p_idx] - plate_w_prev[p_idx]) * plate_inv_dt;
            velocity[f] = plate_sign[f] * v_plt * plate_wgt[f];
            return;
        }
    }

    const int base = int(f) * 8;
    float grad_p = 0.0;
    for (int k = 0; k < 8; ++k) {
        int ci = face_s_cells[base + k];
        grad_p += face_s_coeff[base + k] * pressure[ci];
    }
    velocity[f] = (velocity[f] - dt_over_rho * grad_p) * face_v_damp[f];
}
"""

_VELOCITY_UPDATE_ORDER2_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

/* 2nd-order central-difference gradient for velocity update.
 * Only two pressure cell gathers per face — no Fornberg stencil, no loop.
 * Phase 6: plate_owner[f] >= 0 triggers plate-BC early-out.
 */
layout(std430, binding = 0) buffer PressureBuf   { float pressure[];     };
layout(std430, binding = 1) buffer VelocityBuf   { float velocity[];     };
layout(std430, binding = 2) buffer FaceNegBuf    { int   face_neg[];     };
layout(std430, binding = 3) buffer FacePosBuf    { int   face_pos[];     };
layout(std430, binding = 4) buffer FaceInvDBuf   { float face_inv_d[];   };
layout(std430, binding = 5) buffer FaceVDampBuf  { float face_v_damp[];  };
layout(std430, binding = 6) buffer PlateOwnerBuf { int   plate_owner[];  };
layout(std430, binding = 7) buffer PlateSignBuf  { float plate_sign[];   };
layout(std430, binding = 8) buffer PlateWgtBuf   { float plate_wgt[];    };
layout(std430, binding = 9) buffer ActiveIdxBuf  { int   active_idx[];   };
layout(std430, binding = 10) buffer PlateWBuf    { float plate_w[];      };
layout(std430, binding = 11) buffer PlateWpBuf   { float plate_w_prev[]; };

uniform float dt_over_rho;
uniform int   n_faces;
uniform float plate_inv_dt;
uniform int   plate_bc_enabled;

void main() {
    uint f = gl_GlobalInvocationID.x;
    if (f >= uint(n_faces)) return;

    if (plate_bc_enabled != 0) {
        int owner = plate_owner[f];
        if (owner >= 0) {
            int p_idx = active_idx[owner];
            float v_plt = (plate_w[p_idx] - plate_w_prev[p_idx]) * plate_inv_dt;
            velocity[f] = plate_sign[f] * v_plt * plate_wgt[f];
            return;
        }
    }

    float grad_p = (pressure[face_pos[f]] - pressure[face_neg[f]]) * face_inv_d[f];
    velocity[f] = (velocity[f] - dt_over_rho * grad_p) * face_v_damp[f];
}
"""
_DIVERGENCE_PRESSURE_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

/* Fused divergence + pressure update (Phases 2 and 3).
 * csr_face_weight[k] = ±face_flux_coef[face] pre-baked at setup time.
 * Eliminates the intermediate div_flux buffer, one dispatch, one barrier.
 *
 * p_new = p * P_damp - bulk * div/V * P_src_coeff
 * For non-PML cells P_damp=1, P_src_coeff=1 => standard leapfrog.
 */
layout(std430, binding = 0) buffer VelocityBuf    { float velocity[];        };
layout(std430, binding = 1) buffer PressureBuf    { float pressure[];        };
layout(std430, binding = 2) buffer CellStartsBuf  { int   cell_starts[];     };
layout(std430, binding = 3) buffer CsrIdxBuf      { int   csr_face_idx[];    };
layout(std430, binding = 4) buffer CsrWeightBuf   { float csr_face_weight[]; };
layout(std430, binding = 5) buffer InvDenomBuf    { float cell_inv_denom[];  };
layout(std430, binding = 6) buffer PDampBuf       { float P_damp[];          };
layout(std430, binding = 7) buffer PSrcBuf        { float P_src_coeff[];     };

uniform float bulk;
uniform int   n_cells;

void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= uint(n_cells)) return;

    float d = 0.0;
    int k0 = cell_starts[c];
    int k1 = cell_starts[c + 1];
    for (int k = k0; k < k1; ++k) {
        d += csr_face_weight[k] * velocity[csr_face_idx[k]];
    }
    pressure[c] = pressure[c] * P_damp[c]
                - bulk * d * cell_inv_denom[c] * P_src_coeff[c];
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
layout(std430, binding = 8) buffer PlateL4PrevBuf { float plate_L4_prev[]; };
layout(std430, binding = 9) buffer PlateL4NewBuf  { float plate_L4_new[];  };

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

void main() {
    uint n = gl_GlobalInvocationID.x;
    if (n >= uint(N_active)) return;

    int p_idx = active_idx[n];
    int i    = p_idx / Ny;
    int j    = p_idx % Ny;

    /* Acoustic load */
    float F_acou = 0.0;
    int ca = cell_above[n];
    int cb = cell_below[n];
    if (ca >= 0) F_acou += pressure[ca];
    if (cb >= 0) F_acou -= pressure[cb];
    F_acou *= dx2;

    /* 13-point biharmonic of current w */
    float L4w =
          W(i-2,j) + W(i+2,j) + W(i,j-2) + W(i,j+2)
        + 2.0*(W(i-1,j-1)+W(i-1,j+1)+W(i+1,j-1)+W(i+1,j+1))
        - 8.0*(W(i-1,j)+W(i+1,j)+W(i,j-1)+W(i,j+1))
        + 20.0*W(i,j);
    L4w *= dx4;

    /* Phase 5: read cached L4(w_prev) instead of recomputing it */
    float L4wp = plate_L4_prev[p_idx];

    float w_c = plate_w[p_idx];
    float w_p = plate_w_prev[p_idx];
    float rhs = F_acou + ext_force[p_idx] - coeff_D0*L4w + coeff_Dp*L4wp;
    plate_w_new[p_idx] = (damp_bwd*(2.0*w_c - w_p) + dt2_inv_rh*rhs) / damp_fwd;

    /* Cache L4w for use as L4wp next step */
    plate_L4_new[p_idx] = L4w;
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
layout(std430, binding = 8) buffer PlateL4PrevBuf { float plate_L4_prev[];};
layout(std430, binding = 9) buffer PlateL4NewBuf  { float plate_L4_new[]; };

uniform int N_active;

void main() {
    uint n = gl_GlobalInvocationID.x;
    if (n >= uint(N_active)) return;
    int p_idx = active_idx[n];
    plate_w_prev[p_idx] = plate_w[p_idx];
    plate_w[p_idx]      = plate_w_new[p_idx];
    ext_force[p_idx]    = 0.0;
    /* Phase 5: rotate L4 cache so next step reads correct L4(w_prev) */
    plate_L4_prev[p_idx] = plate_L4_new[p_idx];
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
    int p_idx    = active_idx[n];
    float v_plt = (plate_w[p_idx] - plate_w_prev[p_idx]) * inv_dt;
    int k0 = face_starts[n];
    int k1 = face_starts[n + 1];
    for (int k = k0; k < k1; ++k)
        velocity[face_idx[k]] = face_sign[k] * v_plt * face_wgt[k];
}
"""

_MIC_SAMPLE_GLSL = """\
#version 430 core
layout(local_size_x = 64) in;   /* 64 threads per group; guard below handles n_mics < 64 */

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
/* mic_out layout: ring slots of [p0, vx0, vy0, vz0, p1, vx1, vy1, vz1, ...] */

uniform int n_mics;
uniform int mic_out_base;

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
    int base = mic_out_base + int(m) * 4;
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
    if not bool(glCreateShader):
        raise RuntimeError("OpenGL compute shader compile requested without a current GL context")
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
        glGenBuffers, glBindBuffer, glBufferData, glBindBufferBase, glGetIntegerv,
        GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,
    )
    max_bindings = int(glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS))
    if int(binding) >= max_bindings:
        raise RuntimeError(
            f"SSBO binding {binding} exceeds GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS={max_bindings}"
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
        glGenBuffers, glBindBuffer, glBufferData, glBindBufferBase,
        GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW,
    )
    zeros = np.zeros(nbytes, dtype=np.uint8)
    buf = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
    glBufferData(GL_SHADER_STORAGE_BUFFER, zeros.nbytes, zeros, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf)
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
    with _amr_step_span(None, 0.0, "amr.gl.stencil.prepare",
                        "prepare Python AMR face stencil",
                        f"cells={grid.n_cells} faces={grid.n_faces} sw={sw}"):
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

        # Phase 4: use 0 (not -1) as padding index — coefficient is 0 for padding
        # entries so pressure[0] is gathered but multiplied by 0 (branchless).
        face_s_cells = np.zeros(nf * sw2, dtype=np.int32)
        face_s_coeff = np.zeros(nf * sw2, dtype=np.float32)

    with _amr_step_span(None, 0.0, "amr.gl.stencil.build_faces",
                        "build Python AMR face stencil", f"faces={nf} taps={sw2}"):
        print(f"[amr_stencil.py] starting: n_faces={nf} sw={sw}", flush=True)
        report_every = max(1, nf // 100)
        for f in range(nf):
            if (f + 1) % report_every == 0 or (f + 1) == nf:
                pct = 100.0 * float(f + 1) / float(max(1, nf))
                print(f"[amr_stencil.py] faces {f + 1}/{nf} ({pct:.1f}%)", flush=True)

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
                visited: set[int] = {seed}   # prevent re-visiting any cell
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
                            if nb not in visited and int(ct[nb]) not in (1, 2):
                                best_dot, best_nb = dot_, nb
                    if best_nb < 0:
                        break
                    visited.add(best_nb)
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
            xs_raw = neg_x[::-1] + pos_x
            cs_raw = neg_cells[::-1] + pos_cells
            xs_arr = np.array(xs_raw, dtype=np.float64)

            # Aggregate cells that project to the same face-normal coordinate.
            # Two causes: (1) walk cycle through a previously visited cell
            # (now prevented by `visited`), (2) AMR scale mismatch — a coarse
            # and a fine cell can have centres equidistant from the face along
            # the normal.  In both cases we treat co-located projections as one
            # logical Fornberg node and distribute the resulting weight equally
            # among all physical cells in the group.
            sort_idx = np.argsort(xs_arr, kind="stable")
            sorted_x = xs_arr[sort_idx]
            tol = 1e-10   # well below any AMR cell size (min ~0.25 mm at dx=0.01 × 2 levels)
            groups: list[list[int]] = []
            grp: list[int] = [int(sort_idx[0])]
            for si in range(1, len(sort_idx)):
                if sorted_x[si] - sorted_x[si - 1] <= tol:
                    grp.append(int(sort_idx[si]))
                else:
                    groups.append(grp)
                    grp = [int(sort_idx[si])]
            groups.append(grp)

            if len(groups) < 2:
                continue

            xs_dedup = np.array([float(np.mean(xs_arr[g])) for g in groups], dtype=np.float64)
            w_dedup  = _fornberg_d1(xs_dedup, xi=0.0)

            # Build per-entry (x_rep, cell_idx, weight) list; distribute w equally within each group
            all_entries: list[tuple[float, int, float]] = []
            for gi, g in enumerate(groups):
                w_share = float(w_dedup[gi]) / len(g)
                x_rep   = float(xs_dedup[gi])
                for orig_i in g:
                    all_entries.append((x_rep, cs_raw[orig_i], w_share))

            # Split into neg/pos sides and sort nearest-to-farthest
            neg_st = [(x, c, w) for x, c, w in all_entries if x <  0.0]
            pos_st = [(x, c, w) for x, c, w in all_entries if x >= 0.0]

            if not neg_st or not pos_st:
                continue

            neg_st.sort(key=lambda e: -e[0])   # largest x (nearest) first for neg side
            pos_st.sort(key=lambda e:  e[0])   # smallest x (nearest) first for pos side

            base = f * sw2
            for k, (_, cell, wt) in enumerate(neg_st[:sw]):
                face_s_cells[base + k]      = cell
                face_s_coeff[base + k]      = wt
            for k, (_, cell, wt) in enumerate(pos_st[:sw]):
                face_s_cells[base + sw + k] = cell
                face_s_coeff[base + sw + k] = wt

        print(f"[amr_stencil.py] done: n_faces={nf}", flush=True)

    return face_s_cells, face_s_coeff


def _uniform_i(prog, name: str, val: int):
    from OpenGL.GL import glGetUniformLocation, glUniform1i
    glUniform1i(glGetUniformLocation(prog, name), int(val))


def _uniform_f(prog, name: str, val: float):
    from OpenGL.GL import glGetUniformLocation, glUniform1f
    glUniform1f(glGetUniformLocation(prog, name), float(val))


def _uniform_vec3(prog, name: str, val) -> None:
    from OpenGL.GL import glGetUniformLocation, glUniform3f
    x, y, z = val
    glUniform3f(glGetUniformLocation(prog, name), float(x), float(y), float(z))


# ---------------------------------------------------------------------------
# Module-level shader program cache (Phase 8)
# Shader programs are expensive to compile per-instance and are identical
# across backend instances for the same grid size. Cache by source name so
# catalogue rendering never recompiles.
# ---------------------------------------------------------------------------
_COMPUTE_PROGRAM_CACHE: dict[str, int] = {}


def get_or_compile_compute(name: str, src: str) -> int:
    """Return a cached or freshly compiled GL compute program for ``src``.

    ``name`` is a stable identifier; reuse is keyed on name only. If the
    source needs to change, use a different name.
    """
    prog = _COMPUTE_PROGRAM_CACHE.get(name)
    if prog is not None:
        return prog
    prog = _link_program(_compile_shader(src))
    _COMPUTE_PROGRAM_CACHE[name] = prog
    return prog


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

    def __init__(
        self,
        grid: AcousticAMRGrid,
        c: float = 343.0,
        rho_air: float = 1.21,
        gradient_order: int = 2,
    ):
        if gradient_order not in (2, 8):
            raise ValueError(f"gradient_order must be 2 or 8, got {gradient_order!r}")
        self.gradient_order = int(gradient_order)

        with _amr_step_span(None, 0.0, "amr.gl.init",
                            "create OpenGL AMR backend", f"cells={grid.n_cells} faces={grid.n_faces}"):
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

        # ── CFL timestep: matches C++ amr_create convention ──────────────────
        # order 2 → CFL 0.77;  order 8 → CFL 0.10
        cfl = 0.10 if gradient_order == 8 else 0.77
        min_dx = float(2.0 * grid.cell_half_sizes.min())
        self._dt = cfl * min_dx / (c * math.sqrt(3.0))

        # ── Precomputed topology arrays ───────────────────────────────────────
        face_flux_coef = (grid.face_area * grid.face_open_fraction).astype(np.float32)

        cell_inv_denom = np.zeros(n_cells, dtype=np.float32)
        acoustic_mask = ((grid.cell_types == 0) | (grid.cell_types == 3)) & (grid.open_volume_fraction > 0)
        cell_inv_denom[acoustic_mask] = (
            1.0 / (grid.cell_volumes[acoustic_mask] * np.maximum(grid.open_volume_fraction[acoustic_mask], 1e-12))
        ).astype(np.float32)

        # ── CSR per-cell face lists with pre-baked weights (Phase 3) ─────────
        # csr_face_weight[k] = ±face_flux_coef[face] — sign folded in at setup.
        # This eliminates csr_face_sign + face_flux_coef from the hot dispatch.
        csr_starts = np.zeros(n_cells + 1, dtype=np.int32)
        fn = grid.face_cell_neg.astype(np.int32)
        fp = grid.face_cell_pos.astype(np.int32)
        np.add.at(csr_starts[1:], fn, 1)
        np.add.at(csr_starts[1:], fp, 1)
        np.cumsum(csr_starts, out=csr_starts)
        total = int(csr_starts[n_cells])
        csr_face_idx    = np.empty(total, dtype=np.int32)
        csr_face_weight = np.empty(total, dtype=np.float32)
        fill = csr_starts[:-1].copy()
        for f in range(n_faces):
            a, b = int(fn[f]), int(fp[f])
            ka = fill[a]; fill[a] += 1
            csr_face_idx[ka]    = f;  csr_face_weight[ka] = +float(face_flux_coef[f])
            kb = fill[b]; fill[b] += 1
            csr_face_idx[kb]    = f;  csr_face_weight[kb] = -float(face_flux_coef[f])

        # ── Order-2: build per-face neg/pos index and inv-distance arrays ─────
        face_neg_i32  = grid.face_cell_neg.astype(np.int32)
        face_pos_i32  = grid.face_cell_pos.astype(np.int32)
        face_inv_dist = (1.0 / np.maximum(grid.face_distance, 1e-30)).astype(np.float32)

        # ── Order-8 Fornberg stencil: built only when gradient_order == 8 ─────
        if self.gradient_order == 8:
            with _amr_step_span(None, 0.0, "amr.gl.init.stencil",
                                "build order-8 Fornberg stencil", f"faces={n_faces}"):
                s_cells, s_coeff = _build_face_stencil(grid, sw=4)
            self._buf_stencil_cells = _make_ssbo(s_cells, binding=0)
            self._buf_stencil_coeff = _make_ssbo(s_coeff, binding=0)
        else:
            # order 2: stencil not needed; attributes stay None
            self._buf_stencil_cells = None
            self._buf_stencil_coeff = None

        # ── Upload topology SSBOs (persistent, never change) ──────────────────
        self._buf_pressure      = _make_ssbo_zeros(n_cells * 4)
        self._buf_velocity      = _make_ssbo_zeros(n_faces * 4)
        # Phase 2: no intermediate div_flux buffer; fused shader handles it inline.
        self._buf_csr_starts    = _make_ssbo(csr_starts,      binding=0)
        self._buf_csr_idx       = _make_ssbo(csr_face_idx,    binding=0)
        self._buf_csr_weight    = _make_ssbo(csr_face_weight, binding=0)
        self._buf_inv_denom     = _make_ssbo(cell_inv_denom,  binding=0)
        # Phase 1 (order 2): per-face neg/pos cell index and inverse distance
        self._buf_face_neg      = _make_ssbo(face_neg_i32,  binding=0)
        self._buf_face_pos      = _make_ssbo(face_pos_i32,  binding=0)
        self._buf_face_inv_dist = _make_ssbo(face_inv_dist, binding=0)
        # PML damping arrays — initialised to identity (no absorption).
        # Call setup_border() after construction to activate PML.
        self._buf_face_v_damp   = _make_ssbo(np.ones(n_faces, dtype=np.float32), binding=0)
        self._buf_p_damp        = _make_ssbo(np.ones(n_cells, dtype=np.float32), binding=0)
        self._buf_p_src_coeff   = _make_ssbo(np.ones(n_cells, dtype=np.float32), binding=0)

        self._n_cells = n_cells
        self._n_faces = n_faces

        # ── Plate state (populated in setup_plate) ────────────────────────────
        self._plate_active        = False
        self._n_active_plate      = 0
        self._buf_plate_w         = None
        self._buf_plate_wp        = None
        self._buf_plate_wn        = None
        self._buf_ext_force       = None
        self._buf_active_idx      = None
        self._buf_cell_above      = None
        self._buf_cell_below      = None
        self._buf_bc_starts       = None
        self._buf_bc_idx          = None
        self._buf_bc_sign         = None
        self._buf_bc_wgt          = None
        # Phase 5: L4 biharmonic cache buffers (allocated in setup_plate)
        self._buf_plate_L4_prev   = None
        self._buf_plate_L4_curr   = None
        # Phase 6: per-face plate ownership map (allocated in setup_plate)
        self._buf_plate_owner     = None
        self._buf_plate_owner_sign = None
        self._buf_plate_owner_wgt  = None
        self._plate_uniforms: dict = {}

        # ── Mic state (populated in setup_mics) ───────────────────────────────
        self._n_mics = 0
        self._buf_mic_out    = None
        self._buf_mc_idx     = None
        self._buf_mc_wgt     = None
        self._buf_mf_idx     = None
        self._buf_mf_wx      = None
        self._buf_mf_wy      = None
        self._buf_mf_wz      = None
        self._buf_mic_starts = None
        self._mic_ring_capacity = max(1, int(os.environ.get("SPECTRAL_GL_MIC_RING", "256")))
        self._mic_ring_write = 0
        self._mic_ring_pending = 0
        # Phase 7: offline mic output (allocated in setup_offline_mic_output)
        self._offline_mic_steps = 0
        self._offline_mic_write = 0

        # ── Compile / cache shaders (Phase 8) ─────────────────────────────────
        self._prog_vel_order2 = get_or_compile_compute("velocity_order2", _VELOCITY_UPDATE_ORDER2_GLSL)
        self._prog_vel_order8 = get_or_compile_compute("velocity_order8", _VELOCITY_UPDATE_GLSL)
        self._prog_vel        = self._prog_vel_order8 if gradient_order == 8 else self._prog_vel_order2
        self._prog_div_pres   = get_or_compile_compute("divergence_pressure", _DIVERGENCE_PRESSURE_GLSL)
        self._prog_plate      = get_or_compile_compute("plate_step", _PLATE_STEP_GLSL)
        self._prog_plate_commit = get_or_compile_compute("plate_commit", _PLATE_COMMIT_GLSL)
        self._prog_plate_bc   = get_or_compile_compute("plate_bc", _PLATE_BC_GLSL)
        self._prog_mic        = get_or_compile_compute("mic_sample_v64", _MIC_SAMPLE_GLSL)

        # Precomputed step uniforms
        self._dt_over_rho = float(self._dt / rho_air)
        self._bulk = float(rho_air * c * c * self._dt)

        # Optional dispatch timing (disabled by default; call enable_timing())
        self._time_dispatches: bool = False
        self._step_timers: dict[str, float] = {}
        self._step_timer_calls: int = 0

    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def step_count(self) -> int:
        return self._step_count if hasattr(self, "_step_count") else 0

    # ------------------------------------------------------------------
    def enable_timing(self, enable: bool = True) -> None:
        """Enable or disable per-kernel GL timer queries in ``step()``.

        When enabled, each dispatch in ``step()`` is wrapped with a
        ``GL_TIME_ELAPSED`` query.  Accumulated milliseconds are available via
        ``get_step_timers()``.  Reset the accumulators with
        ``reset_step_timers()``.

        Timer queries add ~1 CPU round-trip per dispatch; use only for
        profiling, not in production catalogue rendering.
        """
        self._time_dispatches = bool(enable)
        if enable:
            for k in ("velocity_ms", "div_pressure_ms",
                      "plate_step_ms", "plate_commit_ms"):
                self._step_timers.setdefault(k, 0.0)

    def reset_step_timers(self) -> None:
        """Zero all accumulated dispatch timers and call counter."""
        self._step_timers = {k: 0.0 for k in self._step_timers}
        self._step_timer_calls = 0

    def get_step_timers(self) -> dict[str, float]:
        """Return a copy of the accumulated timer dict (milliseconds per kernel).

        Keys: ``velocity_ms``, ``div_pressure_ms``, ``plate_step_ms``,
        ``plate_commit_ms``.  Values are cumulative totals since last
        ``reset_step_timers()`` or ``enable_timing()``.
        """
        return dict(self._step_timers)

    def _timed_dispatch(self, label: str, gx: int) -> None:
        """Dispatch a compute shader and, if timing is enabled, record elapsed ns."""
        from OpenGL.GL import glDispatchCompute
        if not self._time_dispatches:
            glDispatchCompute(gx, 1, 1)
            return
        from OpenGL.GL import (
            glGenQueries, glBeginQuery, glEndQuery,
            glGetQueryObjectuiv, glDeleteQueries,
            GL_TIME_ELAPSED, GL_QUERY_RESULT,
        )
        q = glGenQueries(1)[0]
        glBeginQuery(GL_TIME_ELAPSED, q)
        glDispatchCompute(gx, 1, 1)
        glEndQuery(GL_TIME_ELAPSED)
        elapsed_ns = int(glGetQueryObjectuiv(q, GL_QUERY_RESULT))
        glDeleteQueries([q])
        self._step_timers[label] = self._step_timers.get(label, 0.0) + elapsed_ns / 1e6
        self._step_timer_calls += 1

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

        # Phase 5: L4 biharmonic cache (one float per plate node, full grid size)
        self._buf_plate_L4_prev = _make_ssbo_zeros(N * 4)
        self._buf_plate_L4_curr = _make_ssbo_zeros(N * 4)

        # Phase 6: build per-face ownership map from merged BC CSR.
        # plate_owner[f] = active-node index that owns face f, or -1 if none.
        n_faces_total = self._n_faces
        plate_owner      = np.full(n_faces_total, -1, dtype=np.int32)
        plate_owner_sign = np.zeros(n_faces_total, dtype=np.float32)
        plate_owner_wgt  = np.zeros(n_faces_total, dtype=np.float32)
        for nd in range(n_active):
            for k in range(int(bc_starts[nd]), int(bc_starts[nd + 1])):
                f = int(bc_idx[k])
                if plate_owner[f] != -1:
                    raise RuntimeError(
                        f"Face {f} has duplicate plate owners: {plate_owner[f]} and {nd}"
                    )
                plate_owner[f]      = nd
                plate_owner_sign[f] = float(bc_sign[k])
                plate_owner_wgt[f]  = float(bc_wgt[k])
        self._buf_plate_owner      = _make_ssbo(plate_owner,      binding=0)
        self._buf_plate_owner_sign = _make_ssbo(plate_owner_sign, binding=0)
        self._buf_plate_owner_wgt  = _make_ssbo(plate_owner_wgt,  binding=0)

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
        self._mic_ring_write = 0
        self._mic_ring_pending = 0
        self._buf_mic_out = _make_ssbo_zeros(self._mic_ring_capacity * n_mics * 4 * 4)
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

    def _dispatch_mic_sample(self, slot: int) -> None:
        from OpenGL.GL import (
            glUseProgram, glDispatchCompute, glMemoryBarrier,
            GL_SHADER_STORAGE_BARRIER_BIT,
        )
        n = self._n_mics
        if n <= 0:
            return
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
        _uniform_i(self._prog_mic, "mic_out_base", int(slot) * n * 4)
        glDispatchCompute(_gl_ceil_div(n, 64), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def enqueue_mic_sample(self) -> bool:
        """Dispatch mic sampling into a GPU ring slot without CPU readback.

        Returns False if the ring is full and the caller should flush first.
        """
        if self._n_mics <= 0:
            return True
        if self._mic_ring_pending >= self._mic_ring_capacity:
            return False
        slot = (self._mic_ring_write + self._mic_ring_pending) % self._mic_ring_capacity
        self._dispatch_mic_sample(slot)
        self._mic_ring_pending += 1
        return True

    def flush_mic_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read queued mic ring samples in one CPU/GPU sync.

        Returns arrays shaped (n_samples, n_mics). If the ring wrapped, this
        performs two contiguous reads and concatenates on CPU.
        """
        import ctypes
        from OpenGL.GL import glBindBuffer, glGetBufferSubData, GL_SHADER_STORAGE_BUFFER
        n = self._n_mics
        count = int(self._mic_ring_pending)
        if n == 0 or count == 0:
            z = np.zeros((0, n), dtype=np.float32)
            return z, z, z, z

        floats_per_slot = n * 4

        def _read_slots(start_slot: int, n_slots: int) -> np.ndarray:
            out = np.empty(n_slots * floats_per_slot, dtype=np.float32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_mic_out)
            glGetBufferSubData(
                GL_SHADER_STORAGE_BUFFER,
                start_slot * floats_per_slot * 4,
                out.nbytes,
                ctypes.c_void_p(out.ctypes.data),
            )
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            return out.reshape(n_slots, n, 4)

        first = min(count, self._mic_ring_capacity - self._mic_ring_write)
        chunks = [_read_slots(self._mic_ring_write, first)]
        if first < count:
            chunks.append(_read_slots(0, count - first))
        arr = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
        self._mic_ring_write = (self._mic_ring_write + count) % self._mic_ring_capacity
        self._mic_ring_pending = 0
        return (
            np.ascontiguousarray(arr[:, :, 0]),
            np.ascontiguousarray(arr[:, :, 1]),
            np.ascontiguousarray(arr[:, :, 2]),
            np.ascontiguousarray(arr[:, :, 3]),
        )

    def step_with_mic_samples(self, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance and collect mic samples using the async GPU readback ring.

        This is the throughput-oriented GL path: mic samples are written into
        the SSBO ring each step, and CPU readback happens only when the ring is
        full or after the requested block completes.
        """
        chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(max(0, int(n_steps))):
            self.step(1)
            if not self.enqueue_mic_sample():
                chunks.append(self.flush_mic_samples())
                ok = self.enqueue_mic_sample()
                if not ok:
                    raise RuntimeError("AMR GL mic ring did not accept a sample after flush")
        chunks.append(self.flush_mic_samples())
        if not chunks:
            z = np.zeros((0, self._n_mics), dtype=np.float32)
            return z, z, z, z
        return tuple(
            np.concatenate([chunk[i] for chunk in chunks], axis=0)
            for i in range(4)
        )

    # ------------------------------------------------------------------
    # Phase 7: Offline (monolithic) mic output
    # ------------------------------------------------------------------
    def setup_offline_mic_output(self, n_steps: int) -> None:
        """Allocate a GPU SSBO large enough to hold all mic samples for a run.

        Must be called after ``setup_mics``.  Each step writes n_mics * 4
        floats (p, vx, vy, vz) in a flat append-buffer rather than the ring.

        Parameters
        ----------
        n_steps : total number of steps the subsequent ``run_offline_steps``
                  call will execute.  The buffer is sized exactly for this.
        """
        n = self._n_mics
        if n == 0:
            return
        n_steps = max(1, int(n_steps))
        self._offline_mic_steps = n_steps
        self._offline_mic_write = 0
        # Allocate a fresh flat buffer; oversizes by 1 slot to avoid 0-byte alloc.
        self._buf_mic_out = _make_ssbo_zeros(n_steps * n * 4 * 4)

    def run_offline_steps(self, n_steps: int, mic_every: int = 1) -> None:
        """Advance ``n_steps`` steps and append mic samples to the offline buffer.

        Must be called after ``setup_offline_mic_output``.  Calls ``step(1)``
        and ``_dispatch_mic_sample`` sequentially; no CPU readback until
        ``read_offline_mic_output`` is called.

        Parameters
        ----------
        n_steps   : number of steps to run.  Must not exceed the capacity
                    allocated by ``setup_offline_mic_output``.
        mic_every : mic sampling decimation — dispatch the mic kernel only
                    every *mic_every* FDTD substeps (default 1 = every step).
                    For offline catalogue work set this to
                    ``round(1 / (sample_rate * dt))`` to sample at audio rate
                    rather than the (much higher) FDTD rate.
        """
        remaining = self._offline_mic_steps - self._offline_mic_write
        if n_steps > remaining:
            raise RuntimeError(
                f"Offline mic buffer overflow: requested {n_steps} steps "
                f"but only {remaining} remain (capacity {self._offline_mic_steps})."
            )
        mic_every = max(1, int(mic_every))
        for step_i in range(n_steps):
            self.step(1)
            if self._n_mics > 0 and (step_i % mic_every == 0):
                self._dispatch_mic_sample(self._offline_mic_write)
                self._offline_mic_write += 1

    def read_offline_mic_output(self) -> np.ndarray:
        """Read completed offline mic output from GPU to CPU.

        Returns
        -------
        np.ndarray of shape ``(n_written, n_mics, 4)`` and dtype float32.
        Channels: [p, vx, vy, vz].
        """
        import ctypes
        from OpenGL.GL import glBindBuffer, glGetBufferSubData, GL_SHADER_STORAGE_BUFFER
        n = self._n_mics
        count = self._offline_mic_write
        if n == 0 or count == 0:
            return np.zeros((0, max(1, n), 4), dtype=np.float32)
        out = np.empty(count * n * 4, dtype=np.float32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_mic_out)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, ctypes.c_void_p(out.ctypes.data))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return out.reshape(count, n, 4)

    # ------------------------------------------------------------------
    # Phase 9: Persistent backend — field reset and in-place param update
    # ------------------------------------------------------------------
    def reset_fields(self) -> None:
        """Zero all dynamic field SSBOs without destroying topology.

        Topology SSBOs (stencil, CSR, mics, plate coupling) are preserved.
        Use this to reuse the backend for a new pluck/strike without the
        overhead of a full teardown and reinitialise.
        """
        from OpenGL.GL import (
            glBindBuffer, glBufferData,
            GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW,
        )
        import ctypes

        def _zero(buf, n_floats: int) -> None:
            if buf is None:
                return
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
            glBufferData(GL_SHADER_STORAGE_BUFFER, n_floats * 4, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        _zero(self._buf_pressure, self._n_cells)
        _zero(self._buf_velocity, self._n_faces)

        if self._plate_active:
            N = (self._plate_uniforms.get("Nx", 1)
                 * self._plate_uniforms.get("Ny", 1))
            _zero(self._buf_plate_w,      N)
            _zero(self._buf_plate_wp,     N)
            _zero(self._buf_plate_wn,     N)
            _zero(self._buf_ext_force,    N)
            _zero(self._buf_plate_L4_prev, N)
            _zero(self._buf_plate_L4_curr, N)

        if self._n_mics > 0 and self._buf_mic_out is not None:
            cap = self._mic_ring_capacity * self._n_mics * 4
            _zero(self._buf_mic_out, cap)
            self._mic_ring_write   = 0
            self._mic_ring_pending = 0
            self._offline_mic_write = 0

        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count = 0

    def update_plate_params(self, desc: dict) -> None:
        """Update plate material parameters in-place without rebuilding topology.

        Only scalar parameters (mass density, stiffness, damping) are updated.
        Plate topology (grid size, active nodes, coupling faces) must remain
        identical to the original ``setup_plate`` call.

        Parameters
        ----------
        desc : dict with the same keys as passed to ``setup_plate``.
               Only ``plate_mass_density``, ``plate_stiffness_D``,
               ``plate_alpha_M``, ``plate_beta_K`` are read.
        """
        if not self._plate_active:
            raise RuntimeError("update_plate_params called before setup_plate")
        pu = self._plate_uniforms
        dt = self._dt
        rh = float(desc["plate_mass_density"])
        D  = float(desc["plate_stiffness_D"])
        aM = float(desc["plate_alpha_M"])
        bK = float(desc["plate_beta_K"])
        damp_fwd = 1.0 + 0.5 * aM * dt
        damp_bwd = 1.0 - 0.5 * aM * dt
        pu["coeff_D0"]   = float(D * (1.0 + bK / dt))
        pu["coeff_Dp"]   = float(D * (bK / dt))
        pu["damp_fwd"]   = float(damp_fwd)
        pu["damp_bwd"]   = float(damp_bwd)
        pu["dt2_inv_rh"] = float(dt * dt / (rh * damp_fwd))
        pu["inv_dt"]     = float(1.0 / dt)

    # ------------------------------------------------------------------
    def step(self, n_steps: int = 1) -> None:
        """Advance AMR FDTD by ``n_steps`` steps on the GPU."""
        from OpenGL.GL import (
            glUseProgram, glMemoryBarrier,
            GL_SHADER_STORAGE_BARRIER_BIT,
        )
        n_cells  = self._n_cells
        n_faces  = self._n_faces
        L        = self._LOCAL
        n_active = self._n_active_plate
        plate_bc_enabled = 1 if (self._plate_active and self._buf_plate_owner is not None) else 0

        for _ in range(n_steps):
            # ── 1. Velocity update (order 2 or 8, plate BC fused in) ──────────
            glUseProgram(self._prog_vel)
            self._bind_base(self._buf_pressure,    0)
            self._bind_base(self._buf_velocity,    1)
            if self.gradient_order == 8:
                self._bind_base(self._buf_stencil_cells, 2)
                self._bind_base(self._buf_stencil_coeff, 3)
                self._bind_base(self._buf_face_v_damp,   4)
            else:
                self._bind_base(self._buf_face_neg,      2)
                self._bind_base(self._buf_face_pos,      3)
                self._bind_base(self._buf_face_inv_dist, 4)
                self._bind_base(self._buf_face_v_damp,   5)
            if plate_bc_enabled:
                self._bind_base(self._buf_plate_owner,      6)
                self._bind_base(self._buf_plate_owner_sign, 7)
                self._bind_base(self._buf_plate_owner_wgt,  8)
                self._bind_base(self._buf_active_idx,       9)
                self._bind_base(self._buf_plate_w,         10)
                self._bind_base(self._buf_plate_wp,        11)
            _uniform_f(self._prog_vel, "dt_over_rho", self._dt_over_rho)
            _uniform_i(self._prog_vel, "n_faces",     n_faces)
            _uniform_i(self._prog_vel, "plate_bc_enabled", plate_bc_enabled)
            if plate_bc_enabled:
                _uniform_f(self._prog_vel, "plate_inv_dt",
                           self._plate_uniforms["inv_dt"])
            self._timed_dispatch("velocity_ms", _gl_ceil_div(n_faces, L))
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 2. Divergence + pressure (fused, Phase 2+3) ───────────────────
            glUseProgram(self._prog_div_pres)
            self._bind_base(self._buf_velocity,    0)
            self._bind_base(self._buf_pressure,    1)
            self._bind_base(self._buf_csr_starts,  2)
            self._bind_base(self._buf_csr_idx,     3)
            self._bind_base(self._buf_csr_weight,  4)
            self._bind_base(self._buf_inv_denom,   5)
            self._bind_base(self._buf_p_damp,      6)
            self._bind_base(self._buf_p_src_coeff, 7)
            _uniform_f(self._prog_div_pres, "bulk",    self._bulk)
            _uniform_i(self._prog_div_pres, "n_cells", n_cells)
            self._timed_dispatch("div_pressure_ms", _gl_ceil_div(n_cells, L))
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # ── 3. Plate step + commit (if active) ────────────────────────────
            if self._plate_active:
                pu = self._plate_uniforms
                # 3a. Plate step
                glUseProgram(self._prog_plate)
                self._bind_base(self._buf_plate_w,      0)
                self._bind_base(self._buf_plate_wp,     1)
                self._bind_base(self._buf_plate_wn,     2)
                self._bind_base(self._buf_ext_force,    3)
                self._bind_base(self._buf_active_idx,   4)
                self._bind_base(self._buf_pressure,     5)
                self._bind_base(self._buf_cell_above,   6)
                self._bind_base(self._buf_cell_below,   7)
                self._bind_base(self._buf_plate_L4_prev, 8)
                self._bind_base(self._buf_plate_L4_curr, 9)
                _uniform_i(self._prog_plate, "N_active",   n_active)
                _uniform_i(self._prog_plate, "Nx",         pu["Nx"])
                _uniform_i(self._prog_plate, "Ny",         pu["Ny"])
                _uniform_f(self._prog_plate, "dx2",        pu["dx2"])
                _uniform_f(self._prog_plate, "dx4",        pu["dx4"])
                _uniform_f(self._prog_plate, "coeff_D0",   pu["coeff_D0"])
                _uniform_f(self._prog_plate, "coeff_Dp",   pu["coeff_Dp"])
                _uniform_f(self._prog_plate, "damp_fwd",   pu["damp_fwd"])
                _uniform_f(self._prog_plate, "damp_bwd",   pu["damp_bwd"])
                _uniform_f(self._prog_plate, "dt2_inv_rh", pu["dt2_inv_rh"])
                self._timed_dispatch("plate_step_ms", _gl_ceil_div(n_active, L))
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

                # 3b. Plate commit (w ← w_new, rotate L4 cache)
                glUseProgram(self._prog_plate_commit)
                self._bind_base(self._buf_plate_w,      0)
                self._bind_base(self._buf_plate_wp,     1)
                self._bind_base(self._buf_plate_wn,     2)
                self._bind_base(self._buf_ext_force,    3)
                self._bind_base(self._buf_active_idx,   4)
                self._bind_base(self._buf_plate_L4_prev, 8)
                self._bind_base(self._buf_plate_L4_curr, 9)
                _uniform_i(self._prog_plate_commit, "N_active", n_active)
                self._timed_dispatch("plate_commit_ms", _gl_ceil_div(n_active, L))
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
            glBindBuffer, glGetBufferSubData,
            GL_SHADER_STORAGE_BUFFER,
        )
        import ctypes
        n = self._n_mics
        if n == 0:
            z = np.zeros(0, dtype=np.float32)
            return z, z, z, z

        with _amr_step_span(None, 0.0, "amr.gl.sample_mics.dispatch",
                            "dispatch AMR GL mic sample", f"mics={n}"):
            self._dispatch_mic_sample(0)

        with _amr_step_span(None, 0.0, "amr.gl.sample_mics.readback",
                            "read back AMR GL mic sample", f"floats={n * 4}"):
            out = np.empty(n * 4, dtype=np.float32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_mic_out)
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, ctypes.c_void_p(out.ctypes.data))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        return out[0::4].copy(), out[1::4].copy(), out[2::4].copy(), out[3::4].copy()

    # ------------------------------------------------------------------
    def get_pressure(self) -> np.ndarray:
        """Read full pressure field back to CPU (slow — debug / test only)."""
        from OpenGL.GL import (
            glBindBuffer, glGetBufferSubData, GL_SHADER_STORAGE_BUFFER,
        )
        import ctypes
        with _amr_step_span(None, 0.0, "amr.gl.readback_pressure",
                            "read back AMR GL pressure", f"cells={self._n_cells}"):
            out = np.empty(self._n_cells, dtype=np.float32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_pressure)
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, ctypes.c_void_p(out.ctypes.data))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return out

    def get_velocity(self) -> np.ndarray:
        """Read full velocity field back to CPU (slow — debug / test only)."""
        from OpenGL.GL import (
            glBindBuffer, glGetBufferSubData, GL_SHADER_STORAGE_BUFFER,
        )
        import ctypes
        with _amr_step_span(None, 0.0, "amr.gl.readback_velocity",
                            "read back AMR GL velocity", f"faces={self._n_faces}"):
            out = np.empty(self._n_faces, dtype=np.float32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._buf_velocity)
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, ctypes.c_void_p(out.ctypes.data))
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
