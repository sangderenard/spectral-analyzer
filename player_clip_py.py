"""
player_clip_py.py — Python-side player clip engine with three-tier fallback.

Tier 1: pybind11 C++ extension (_spectral_kernels.PlayerClipEngine)
Tier 2: ctypes shared library (csrc_bin/player_clip.dll or .so)
Tier 3: pure-Python slab resolver (no compile step required)

All three tiers expose the same interface:
    engine = PlayerClipEngine()
    engine.set_slabs(wall_aabbs)         # numpy (N,8) float32
    new_pos, flags = engine.tick(pos, vel, floor_z, ceil_z, radius, yaw_rad, dt)
    frame = engine.read_state()          # dict with last published frame
    tensor = engine.state_tensor_view()  # owned float32 ring buffer
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Clip flag constants (match player_clip.h)
# --------------------------------------------------------------------------- #
CLIP_NONE = 0
CLIP_X    = 1
CLIP_Y    = 2
CLIP_Z    = 4

STATE_TENSOR_MIN_STRIDE = 32
STATE_TENSOR_DEFAULT_CAPACITY = 256

STATE_TENSOR_COLUMNS = {
    "px": 0, "py": 1, "pz": 2,
    "vx": 3, "vy": 4, "vz": 5,
    "pre_px": 6, "pre_py": 7, "pre_pz": 8,
    "pre_vx": 9, "pre_vy": 10, "pre_vz": 11,
    "yaw_rad": 12, "radius": 13,
    "floor_z": 14, "ceil_z": 15,
    "dt": 16, "clip_flags": 17, "generation": 18,
    "slab_count": 19,
    "clip_dx": 20, "clip_dy": 21, "clip_dz": 22,
    "speed": 23,
    "surface_hit": 24,
    "surface_t": 25,
    "surface_nx": 26, "surface_ny": 27, "surface_nz": 28,
    "surface_reject_x": 29, "surface_reject_y": 30, "surface_reject_z": 31,
}


class _StateTensorMixin:
    def _init_state_tensor(self, capacity: int = STATE_TENSOR_DEFAULT_CAPACITY,
                           stride: int = STATE_TENSOR_MIN_STRIDE) -> None:
        self._state_capacity = max(0, int(capacity))
        self._state_stride = max(STATE_TENSOR_MIN_STRIDE, int(stride))
        self._state_cursor = 0
        self._state_tensor = np.zeros((self._state_capacity, self._state_stride), dtype=np.float32)

    def configure_state_tensor(self, capacity: int,
                               stride: int = STATE_TENSOR_MIN_STRIDE) -> None:
        self._init_state_tensor(capacity, stride)

    def state_tensor_view(self) -> np.ndarray:
        return self._state_tensor

    @property
    def state_tensor_cursor(self) -> int:
        return int(self._state_cursor)

    def _write_state_tensor(self, pre_pos: np.ndarray, pre_vel: np.ndarray,
                            pos: np.ndarray, vel: np.ndarray,
                            floor_z: float, ceil_z: float,
                            radius: float, yaw_rad: float, dt: float,
                            flags: int, generation: int,
                            slab_count: int) -> None:
        if self._state_capacity <= 0:
            return
        row = self._state_tensor[self._state_cursor]
        row.fill(0.0)
        pos3 = np.asarray(pos, dtype=np.float32)[:3]
        vel3 = np.asarray(vel, dtype=np.float32)[:3]
        pre3 = np.asarray(pre_pos, dtype=np.float32)[:3]
        pvel3 = np.asarray(pre_vel, dtype=np.float32)[:3]
        row[0:3] = pos3
        row[3:6] = vel3
        row[6:9] = pre3
        row[9:12] = pvel3
        row[12] = float(yaw_rad)
        row[13] = float(radius)
        row[14] = float(floor_z)
        row[15] = float(ceil_z)
        row[16] = float(dt)
        row[17] = float(flags)
        row[18] = float(generation)
        row[19] = float(slab_count)
        row[20:23] = pos3 - pre3
        row[23] = float(np.linalg.norm(vel3))
        self._state_cursor = (self._state_cursor + 1) % self._state_capacity


def _as_triangles(raw: object) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3) or arr.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    e1 = arr[:, 1] - arr[:, 0]
    e2 = arr[:, 2] - arr[:, 0]
    area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
    return np.ascontiguousarray(arr[area2 > 1e-8], dtype=np.float32)


def _room_cfg_surface_triangles(room_cfg: dict) -> np.ndarray:
    parts: list[np.ndarray] = []
    floor = room_cfg.get("applied_floor_meshes", {})
    if isinstance(floor, dict):
        parts.append(_as_triangles(floor.get("floor_tiles", [])))
        parts.append(_as_triangles(floor.get("floor_fill", [])))
    room = room_cfg.get("applied_room_meshes", {})
    if isinstance(room, dict):
        parts.append(_as_triangles(room.get("walls", [])))
        parts.append(_as_triangles(room.get("ceiling", [])))
    parts = [p for p in parts if p.size]
    if not parts:
        return np.zeros((0, 3, 3), dtype=np.float32)
    tris = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    pts = tris.reshape(-1, 3)
    min_xy = np.min(pts[:, :2], axis=0)
    max_xy = np.max(pts[:, :2], axis=0)
    out = tris.copy()
    out[:, :, 0] -= 0.5 * float(min_xy[0] + max_xy[0])
    out[:, :, 1] -= float(min_xy[1])
    return np.ascontiguousarray(out, dtype=np.float32)

# --------------------------------------------------------------------------- #
# Tier 3: pure-Python fallback
# --------------------------------------------------------------------------- #
class _PurePythonClipEngine(_StateTensorMixin):
    def __init__(self):
        self._slabs: np.ndarray = np.zeros((0, 8), dtype=np.float32)
        self._last: dict = {"px": 0.0, "py": 0.0, "pz": 0.0,
                            "vx": 0.0, "vy": 0.0, "vz": 0.0,
                            "yaw_rad": 0.0, "radius": 0.25,
                            "floor_z": 0.0, "ceil_z": 4.0,
                            "clip_flags": 0, "generation": 0}
        self._gen = 0
        self._init_state_tensor()

    def set_slabs(self, wall_aabbs: np.ndarray) -> None:
        self._slabs = np.asarray(wall_aabbs, dtype=np.float32).reshape(-1, 8)

    def tick(self, pos: np.ndarray, vel: np.ndarray,
             floor_z: float, ceil_z: float, radius: float,
             yaw_rad: float, dt: float = 0.016) -> tuple[np.ndarray, int]:
        p = pos.astype(np.float64, copy=True)
        v = vel.astype(np.float64, copy=True)
        pre_p = p.copy()
        pre_v = v.copy()
        flags = 0

        for row in self._slabs:
            axis      = int(row[0])
            spos      = float(row[1])
            nsign     = float(row[2])
            rlo0, rlo1 = float(row[3]), float(row[4])
            rhi0, rhi1 = float(row[5]), float(row[6])

            aval = p[axis]
            perp = [p[(axis + 1) % 3], p[(axis + 2) % 3]]

            # Perpendicular range check
            if rhi0 != 0.0 or rhi1 != 0.0:
                if not (rlo0 <= perp[0] <= rhi0 and rlo1 <= perp[1] <= rhi1):
                    continue

            dist = (aval - spos) * nsign
            if dist < radius:
                p[axis] += (radius - dist) * nsign
                if v[axis] * nsign < 0.0:
                    v[axis] = 0.0
                flags |= (1 << axis)

        # Floor / ceiling
        if p[2] < floor_z + radius:
            p[2] = floor_z + radius
            if v[2] < 0.0:
                v[2] = 0.0
            flags |= CLIP_Z
        if ceil_z > floor_z and p[2] > ceil_z - radius:
            p[2] = ceil_z - radius
            if v[2] > 0.0:
                v[2] = 0.0
            flags |= CLIP_Z

        self._gen += 1
        self._last = dict(px=p[0], py=p[1], pz=p[2],
                          vx=v[0], vy=v[1], vz=v[2],
                          yaw_rad=yaw_rad, radius=radius,
                          floor_z=floor_z, ceil_z=ceil_z,
                          clip_flags=flags, generation=self._gen)
        self._write_state_tensor(pre_p, pre_v, p, v,
                                 floor_z, ceil_z, radius, yaw_rad, dt,
                                 flags, self._gen, int(self._slabs.shape[0]))
        return p.astype(np.float32), flags

    def read_state(self) -> dict:
        return dict(self._last)

    @property
    def backend(self) -> str:
        return "python"


# --------------------------------------------------------------------------- #
# Tier 2: ctypes shared-library wrapper (not yet compiled — scaffold only)
# --------------------------------------------------------------------------- #
_ctypes_engine_cls: Optional[type] = None

def _try_load_ctypes() -> Optional[type]:
    bin_dir = os.path.join(os.path.dirname(__file__), "csrc_bin")
    candidates = [
        os.path.join(bin_dir, "player_clip.dll"),
        os.path.join(bin_dir, "player_clip.so"),
        os.path.join(bin_dir, "libplayer_clip.so"),
    ]
    lib = None
    for path in candidates:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                break
            except OSError:
                continue
    if lib is None:
        return None

    # Expected C exports: pce_create, pce_set_slabs, pce_tick, pce_read, pce_destroy
    try:
        lib.pce_create.restype  = ctypes.c_void_p
        lib.pce_destroy.argtypes = [ctypes.c_void_p]
        lib.pce_set_slabs.argtypes = [ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_float),
                                       ctypes.c_int]
        lib.pce_tick.argtypes = [ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_float),  # pos (3)
                                  ctypes.POINTER(ctypes.c_float),  # vel (3)
                                  ctypes.c_float, ctypes.c_float,  # floor_z, ceil_z
                                  ctypes.c_float, ctypes.c_float]  # radius, yaw
        lib.pce_tick.restype  = ctypes.c_uint32
        # 16-float frame for read
        lib.pce_read.argtypes = [ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_float)]
    except AttributeError:
        return None

    class _CtypesClipEngine:
        def __init__(self):
            self._h  = lib.pce_create()
            self._gen = 0

        def __del__(self):
            if self._h:
                lib.pce_destroy(self._h)

        def set_slabs(self, wall_aabbs: np.ndarray) -> None:
            arr = np.asarray(wall_aabbs, dtype=np.float32).reshape(-1, 8)
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            lib.pce_set_slabs(self._h, ptr, ctypes.c_int(len(arr)))

        def tick(self, pos: np.ndarray, vel: np.ndarray,
                 floor_z: float, ceil_z: float, radius: float,
                 yaw_rad: float, dt: float = 0.016) -> tuple[np.ndarray, int]:
            p = np.asarray(pos, dtype=np.float32).copy()
            v = np.asarray(vel, dtype=np.float32).copy()
            pp = p.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            vp = v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            flags = lib.pce_tick(self._h, pp, vp,
                                 ctypes.c_float(floor_z), ctypes.c_float(ceil_z),
                                 ctypes.c_float(radius), ctypes.c_float(yaw_rad))
            self._gen += 1
            return p, int(flags)

        def read_state(self) -> dict:
            buf = (ctypes.c_float * 16)()
            lib.pce_read(self._h, buf)
            keys = ["px","py","pz","vx","vy","vz","yaw_rad","radius",
                    "floor_z","ceil_z","clip_flags_f","generation_f",
                    "_p0","_p1","_p2","_p3"]
            d = {k: float(buf[i]) for i, k in enumerate(keys)}
            d["clip_flags"] = struct.unpack("I", struct.pack("f", d.pop("clip_flags_f")))[0]
            d["generation"] = int(struct.unpack("I", struct.pack("f", d.pop("generation_f")))[0])
            for k in ["_p0","_p1","_p2","_p3"]:
                d.pop(k, None)
            return d

        @property
        def backend(self) -> str:
            return "ctypes"

    return _CtypesClipEngine


# --------------------------------------------------------------------------- #
# Tier 1: pybind11 extension
# --------------------------------------------------------------------------- #
_pybind_cls: Optional[type] = None

def _try_load_pybind() -> Optional[type]:
    try:
        from _spectral_kernels import PlayerClipEngine as _PBE  # type: ignore
        return _PBE
    except (ImportError, AttributeError):
        return None


# --------------------------------------------------------------------------- #
# Public factory — picks the best available backend
# --------------------------------------------------------------------------- #
def _build_engine() -> object:
    cls = _try_load_pybind()
    if cls is not None:
        return cls()
    cls = _try_load_ctypes()
    if cls is not None:
        return cls()
    return _PurePythonClipEngine()


class PlayerClipEngine:
    """
    Facade that delegates to the best available clip engine backend.

    Usage::
        engine = PlayerClipEngine()
        engine.set_slabs(np.array([...], dtype=np.float32))  # shape (N, 8)
        new_pos, flags = engine.tick(pos3, vel3, floor_z, ceil_z, radius, yaw)
    """

    def __init__(self):
        self._impl = _build_engine()
        self._compat_tensor = _PurePythonClipEngine()
        self._surface_clip_enabled = False
        self._surface_clip_distance = 1.0
        self._surface_clip_angle_rad = math.radians(45.0)
        self._surface_clip_rays = 9
        self._surface_tris = np.zeros((0, 3, 3), dtype=np.float32)
        if hasattr(self._impl, "configure_state_tensor"):
            try:
                self._impl.configure_state_tensor(STATE_TENSOR_DEFAULT_CAPACITY,
                                                  STATE_TENSOR_MIN_STRIDE)
            except TypeError:
                self._impl.configure_state_tensor(STATE_TENSOR_DEFAULT_CAPACITY)

    def set_slabs(self, wall_aabbs: np.ndarray) -> None:
        """
        Set collision slabs.  wall_aabbs must be shape (N, 8) float32:
          col 0: axis (0/1/2)
          col 1: slab face position on that axis
          col 2: normal_sign (+1 or -1)
          col 3,4: range_lo (perp axes); both 0 → infinite
          col 5,6: range_hi (perp axes)
          col 7: reserved (0)
        """
        arr = np.asarray(wall_aabbs, dtype=np.float32).reshape(-1, 8)
        self._impl.set_slabs(arr)
        self._compat_tensor.set_slabs(arr)

    def set_surface_field(self, triangles: object) -> None:
        """Attach world-space surface triangles used by optional cone clipping."""
        self._surface_tris = _as_triangles(triangles)

    def set_surface_field_from_room_cfg(self, room_cfg: dict) -> None:
        """Attach the current applied room geometry as the surface field."""
        self.set_surface_field(_room_cfg_surface_triangles(room_cfg))

    def enable_surface_clip(self, enabled: bool = True, *,
                            max_distance: float = 1.0,
                            cone_angle_rad: float | None = None,
                            n_rays: int = 9) -> None:
        """Use trace_cone_rays as an optional velocity-normal clip pass."""
        self._surface_clip_enabled = bool(enabled)
        self._surface_clip_distance = max(0.0, float(max_distance))
        self._surface_clip_angle_rad = (
            math.radians(45.0) if cone_angle_rad is None else max(0.0, float(cone_angle_rad))
        )
        self._surface_clip_rays = max(1, int(n_rays))

    def tick(self, pos: np.ndarray, vel: np.ndarray,
             floor_z: float = 0.0, ceil_z: float = 4.0,
             radius: float = 0.25, yaw_rad: float = 0.0,
             dt: float = 0.016) -> tuple[np.ndarray, int]:
        """
        Advance one simulation step.  Returns (corrected_pos_3f, clip_flags).
        """
        pre_pos = np.asarray(pos, dtype=np.float32).copy()
        pre_vel = np.asarray(vel, dtype=np.float32).copy()
        try:
            result = self._impl.tick(pos, vel, floor_z, ceil_z, radius, yaw_rad, dt)
        except TypeError as exc:
            msg = str(exc)
            if "incompatible function arguments" not in msg and "takes" not in msg:
                raise
            result = self._impl.tick(pos, vel, floor_z, ceil_z, radius, yaw_rad)

        if not isinstance(result, tuple):
            raise TypeError(f"PlayerClipEngine.tick backend returned {type(result).__name__}, expected tuple")
        if len(result) == 2:
            out_pos, flags = result
            out_vel = vel
        elif len(result) == 3:
            out_pos, out_vel, flags = result
        else:
            raise TypeError(f"PlayerClipEngine.tick backend returned {len(result)} values, expected 2 or 3")
        out_arr = np.asarray(out_pos, dtype=np.float32)
        out_vel_arr = np.asarray(out_vel, dtype=np.float32)
        flags_i = int(flags)
        surface_info = self._surface_cone_clip(pre_pos, out_arr, out_vel_arr)
        if surface_info is not None:
            out_arr, out_vel_arr, extra_flags, surface_meta = surface_info
            flags_i |= int(extra_flags)
        else:
            surface_meta = None
        if not hasattr(self._impl, "state_tensor_view"):
            self._compat_tensor._gen += 1
            self._compat_tensor._write_state_tensor(
                pre_pos, pre_vel,
                out_arr, out_vel_arr,
                floor_z, ceil_z, radius, yaw_rad, dt,
                flags_i, self._compat_tensor._gen,
                int(self._compat_tensor._slabs.shape[0]),
            )
            self._write_surface_tensor_fields(
                self._compat_tensor.state_tensor_view(),
                (self._compat_tensor.state_tensor_cursor - 1) % max(1, self._compat_tensor.state_tensor_view().shape[0]),
                surface_meta,
            )
        elif surface_meta is not None:
            view = self.state_tensor_view()
            if view.size:
                cursor = self.state_tensor_cursor
                self._write_surface_tensor_fields(view, (cursor - 1) % max(1, view.shape[0]), surface_meta)
        return out_arr, flags_i

    def _surface_cone_clip(self, pre_pos: np.ndarray, pos: np.ndarray,
                           vel: np.ndarray):
        if (not self._surface_clip_enabled or self._surface_tris.size == 0 or
                self._surface_clip_distance <= 0.0):
            return None
        v = np.asarray(vel, dtype=np.float32).reshape(3)
        speed = float(np.linalg.norm(v))
        if speed <= 1e-6:
            delta = np.asarray(pos, dtype=np.float32).reshape(3) - np.asarray(pre_pos, dtype=np.float32).reshape(3)
            speed = float(np.linalg.norm(delta))
            if speed <= 1e-6:
                return None
            v = delta
        try:
            from ray_tracer_bridge import trace_cone_rays
            hits = trace_cone_rays(
                None,
                np.asarray(pre_pos, dtype=np.float32).reshape(3),
                v,
                cone_angle_rad=self._surface_clip_angle_rad,
                n_rays=self._surface_clip_rays,
                max_distance=self._surface_clip_distance,
                verts=self._surface_tris,
                normals=None,
            )
        except Exception:
            return None
        if int(hits.get("n_hits", 0) or 0) <= 0:
            return None
        distances = np.asarray(hits.get("distance"), dtype=np.float32).reshape(-1)
        normals = np.asarray(hits.get("normal"), dtype=np.float32).reshape(-1, 3)
        hit_mask = np.asarray(hits.get("hit"), dtype=bool).reshape(-1)
        valid = hit_mask & np.isfinite(distances)
        if not np.any(valid):
            return None
        idx = int(np.argmin(np.where(valid, distances, np.inf)))
        n = normals[idx]
        nl = float(np.linalg.norm(n))
        if nl <= 1e-6:
            return None
        n = n / nl
        if float(np.dot(n, v)) > 0.0:
            n = -n
        vn = float(np.dot(vel, n))
        reject = np.zeros(3, dtype=np.float32)
        out_vel = np.asarray(vel, dtype=np.float32).copy()
        out_pos = np.asarray(pos, dtype=np.float32).copy()
        if vn < 0.0:
            reject = (-vn * n).astype(np.float32)
            out_vel = (out_vel + reject).astype(np.float32)
        penetration_guard = max(0.0, self._surface_clip_distance - float(distances[idx]))
        out_pos = (out_pos + n.astype(np.float32) * min(0.05, penetration_guard)).astype(np.float32)
        axis = int(np.argmax(np.abs(n)))
        meta = {
            "hit": 1.0,
            "distance": float(distances[idx]),
            "normal": n.astype(np.float32),
            "reject": reject.astype(np.float32),
        }
        return out_pos, out_vel, (1 << axis), meta

    @staticmethod
    def _write_surface_tensor_fields(tensor: np.ndarray, row_idx: int, meta) -> None:
        if meta is None or tensor.size == 0 or tensor.shape[1] < STATE_TENSOR_MIN_STRIDE:
            return
        row = tensor[int(row_idx)]
        row[24] = float(meta.get("hit", 0.0))
        row[25] = float(meta.get("distance", 0.0))
        n = np.asarray(meta.get("normal", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        r = np.asarray(meta.get("reject", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        row[26:29] = n
        row[29:32] = r

    def configure_state_tensor(self, capacity: int,
                               stride: int = STATE_TENSOR_MIN_STRIDE) -> None:
        if hasattr(self._impl, "configure_state_tensor"):
            try:
                self._impl.configure_state_tensor(int(capacity), int(stride))
            except TypeError:
                self._impl.configure_state_tensor(int(capacity))
        self._compat_tensor.configure_state_tensor(int(capacity), int(stride))

    def state_tensor_view(self) -> np.ndarray:
        if hasattr(self._impl, "state_tensor_view"):
            return np.asarray(self._impl.state_tensor_view(), dtype=np.float32)
        return self._compat_tensor.state_tensor_view()

    @property
    def state_tensor_cursor(self) -> int:
        if hasattr(self._impl, "state_tensor_cursor"):
            return int(getattr(self._impl, "state_tensor_cursor"))
        return self._compat_tensor.state_tensor_cursor

    def read_state(self) -> dict:
        """Return the last published player state as a dict."""
        return self._impl.read_state()

    @property
    def backend(self) -> str:
        return getattr(self._impl, "backend", type(self._impl).__name__)

    # ---- Convenience: build slabs from a room_cfg dict -------------------- #
    @staticmethod
    def slabs_from_room_cfg(room_cfg: dict) -> np.ndarray:
        """
        Derive axis-aligned wall slabs from an applied room_cfg dict.
        Builds four infinite slabs (±X, ±Y) from applied_room_width_m /
        applied_room_depth_m.  Returns shape (4, 8) float32.
        """
        w = float(room_cfg.get("applied_room_width_m") or 0.0)
        d = float(room_cfg.get("applied_room_depth_m") or 0.0)
        if w <= 0.0 or d <= 0.0:
            return np.zeros((0, 8), dtype=np.float32)

        hw = w / 2.0
        # axis, pos, normal_sign, rlo0, rlo1, rhi0, rhi1, reserved
        return np.array([
            [0, -hw,  1.0, 0, 0, 0, 0, 0],   # X- wall, faces +X
            [0,  hw, -1.0, 0, 0, 0, 0, 0],   # X+ wall, faces -X
            [1,  0.0,  1.0, 0, 0, 0, 0, 0],  # Y- wall, faces +Y
            [1,   d,  -1.0, 0, 0, 0, 0, 0],  # Y+ wall, faces -Y
        ], dtype=np.float32)
