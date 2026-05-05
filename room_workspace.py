"""room_workspace.py
====================
RoomWorkspace — pure state machine for the room environment.

Responsibilities
----------------
* Load and hold the room config (dimensions, physics, station position).
* Own the placed-object registry (PlacedLight, PlacedEnclosure, …).
* Provide ``physics_config()`` → dict for PlayerController initialisation.
* Provide ``apply_physics(player_ctrl)`` for live in-game edits.
* Serialise / deserialise the full scene to/from YAML-compatible dicts.
* Build the live station objects from the registry when requested.

No GL, no pygame.
"""
from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional

import numpy as np

from placed_object import (
    PlacedObject,
    PlacedLight, PlacedCamera, PlacedEnclosure,
    PlacedDutyStation, PlacedPortalFrame,
    placed_object_from_dict,
)

# YAML is optional — workspace degrades gracefully without it
try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    if not _HAS_YAML:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return _yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"[RoomWorkspace] cannot load {path}: {exc}", flush=True)
        return {}


def _dump_yaml(data: dict, path: str) -> bool:
    if not _HAS_YAML:
        return False
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            _yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
        return True
    except Exception as exc:
        print(f"[RoomWorkspace] cannot write {path}: {exc}", flush=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class RoomWorkspace:
    """Owns all scene state for the room environment.  No GL.

    Parameters
    ----------
    config_dir : str
        Directory that contains ``room.yaml``, ``physics.yaml``,
        ``station.yaml``, and ``scene.yaml``.
    """

    def __init__(self, config_dir: str = "configs/room_station"):
        self._config_dir = config_dir

        # ── load config files ────────────────────────────────────────────────
        self.room_cfg    = _load_yaml(os.path.join(config_dir, "room.yaml"))
        self.physics_cfg = _load_yaml(os.path.join(config_dir, "physics.yaml"))
        self.station_cfg = _load_yaml(os.path.join(config_dir, "station.yaml"))
        scene_cfg        = _load_yaml(os.path.join(config_dir, "scene.yaml"))

        # ── object registry ──────────────────────────────────────────────────
        self._objects: Dict[str, PlacedObject] = {}
        for obj_dict in scene_cfg.get("objects", []):
            try:
                obj = placed_object_from_dict(obj_dict)
                self._objects[obj.obj_id] = obj
            except Exception as exc:
                print(f"[RoomWorkspace] skip bad object: {exc}", flush=True)

        # ── selection state ───────────────────────────────────────────────────
        self.selected_id: Optional[str] = None

        # ── dirty flag for save ───────────────────────────────────────────────
        self._dirty = False

    # ── Class factory ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, config_dir: str = "configs/room_station") -> "RoomWorkspace":
        return cls(config_dir)

    @classmethod
    def blank(cls, config_dir: str = "configs/room_station") -> "RoomWorkspace":
        """Return a workspace with an empty object registry (ignores scene.yaml)."""
        ws = cls.__new__(cls)
        ws._config_dir = config_dir
        ws.room_cfg    = _load_yaml(os.path.join(config_dir, "room.yaml"))
        ws.physics_cfg = _load_yaml(os.path.join(config_dir, "physics.yaml"))
        ws.station_cfg = _load_yaml(os.path.join(config_dir, "station.yaml"))
        ws._objects    = {}
        ws.selected_id = None
        ws._dirty      = False
        return ws

    @classmethod
    def from_scene_file(cls, path: str,
                        config_dir: str = "configs/room_station") -> "RoomWorkspace":
        """Load a workspace from an explicit scene YAML file (not config_dir/scene.yaml)."""
        ws = cls.blank(config_dir)
        scene_data = _load_yaml(path)
        for obj_dict in scene_data.get("objects", []):
            try:
                obj = placed_object_from_dict(obj_dict)
                ws._objects[obj.obj_id] = obj
            except Exception as exc:
                print(f"[RoomWorkspace] skip bad object in {path}: {exc}", flush=True)
        return ws

    # ── Object registry ───────────────────────────────────────────────────────

    @property
    def objects(self) -> Dict[str, PlacedObject]:
        return self._objects

    def add_object(self, obj: PlacedObject) -> bool:
        if obj.obj_id in self._objects:
            return False
        self._objects[obj.obj_id] = obj
        self._dirty = True
        return True

    def remove_object(self, obj_id: str) -> bool:
        if obj_id not in self._objects:
            return False
        del self._objects[obj_id]
        if self.selected_id == obj_id:
            self.selected_id = None
        self._dirty = True
        return True

    def move_object(self, obj_id: str, new_pos: np.ndarray) -> bool:
        obj = self._objects.get(obj_id)
        if obj is None:
            return False
        obj.pos = np.asarray(new_pos, np.float64)
        self._dirty = True
        return True

    def rotate_object(self, obj_id: str, yaw_deg: float) -> bool:
        obj = self._objects.get(obj_id)
        if obj is None:
            return False
        obj.yaw_deg = float(yaw_deg)
        self._dirty = True
        return True

    def select(self, obj_id: Optional[str]) -> bool:
        if obj_id is not None and obj_id not in self._objects:
            return False
        self.selected_id = obj_id
        return True

    @property
    def selected(self) -> Optional[PlacedObject]:
        return self._objects.get(self.selected_id) if self.selected_id else None

    # ── Typed views ───────────────────────────────────────────────────────────

    def lights(self) -> List[PlacedLight]:
        return [o for o in self._objects.values() if isinstance(o, PlacedLight)]

    def enclosures(self) -> List[PlacedEnclosure]:
        return [o for o in self._objects.values() if isinstance(o, PlacedEnclosure)]

    def duty_stations(self) -> List[PlacedDutyStation]:
        return [o for o in self._objects.values() if isinstance(o, PlacedDutyStation)]

    def cameras(self) -> List[PlacedCamera]:
        return [o for o in self._objects.values() if isinstance(o, PlacedCamera)]

    def portals(self) -> List[PlacedPortalFrame]:
        return [o for o in self._objects.values() if isinstance(o, PlacedPortalFrame)]

    # ── Physics ───────────────────────────────────────────────────────────────

    def physics_config(self) -> dict:
        """Return a PlayerController-compatible config dict."""
        return dict(self.physics_cfg)

    def apply_physics(self, player_ctrl) -> None:
        """Push current physics values into a live PlayerController."""
        w = self.physics_cfg.get("walk", {})
        if hasattr(player_ctrl, "_move_speed"):
            player_ctrl._move_speed = float(w.get("move_speed", 3.0))
        if hasattr(player_ctrl, "_run_mult"):
            player_ctrl._run_mult   = float(w.get("run_multiplier", 2.2))
        if hasattr(player_ctrl, "_eye_height"):
            player_ctrl._eye_height = float(w.get("eye_height", 1.65))

    # ── Room geometry data ────────────────────────────────────────────────────

    def room_dims(self) -> dict:
        return dict(self.room_cfg.get("dimensions", {}))

    def station_pos(self) -> np.ndarray:
        p = self.station_cfg.get("position", [0.0, 0.0, 0.5])
        return np.array(p, np.float64)

    def station_yaw_deg(self) -> float:
        return float(self.station_cfg.get("yaw_deg", 180.0))

    def interact_radius(self) -> float:
        return float(self.station_cfg.get("interact_radius_m", 2.0))

    def console_camera(self) -> dict:
        return dict(self.station_cfg.get("console_camera", {}))

    def layout(self) -> dict:
        return dict(self.station_cfg.get("layout", {}))

    # ── Scene build ───────────────────────────────────────────────────────────

    def build_scene_objects(self, win_w: int = 1400, win_h: int = 900) -> list:
        """Instantiate all placeable duty-station objects from the registry.

        Returns a list of station objects that have:
          - ``init_gl()``       already called
          - ``draw(MVP,MV,lv)`` for 3-D render
          - ``handle_event(ev)``

        Enclosures that wrap a simulator are also instantiated here.
        Lights and portals are NOT instantiated as station objects.
        """
        stations = []
        for obj in self._objects.values():
            station = self._try_build_station(obj, win_w, win_h)
            if station is not None:
                stations.append(station)
        return stations

    def _try_build_station(self, obj: PlacedObject,
                           win_w: int, win_h: int):
        """Attempt to build a live station from a placed object."""
        if isinstance(obj, PlacedDutyStation):
            return self._build_duty_station(obj, win_w, win_h)

        if isinstance(obj, PlacedEnclosure) and obj.simulator is not None:
            return self._build_enclosure_simulator(obj, win_w, win_h)

        return None

    def _build_duty_station(self, obj: PlacedDutyStation,
                            win_w: int, win_h: int):
        """Build a unified DutyStation world object with an optional HUD menu.

        Tile specificity lives in ``station_type`` + ``config_dir`` on the
        placed object. The world object is always ``DutyStation``.
        """
        try:
            from duty_station import DutyStation

            cfg_dir = obj.config_dir
            st_cfg = _load_yaml(os.path.join(cfg_dir, "station.yaml"))
            st_cfg["position"] = np.asarray(obj.pos, np.float64).tolist()
            st_cfg["yaw_deg"] = float(obj.yaw_deg)
            st_cfg["interaction_radius"] = float(obj.interaction_radius)
            if "module_type" not in st_cfg:
                st_cfg["module_type"] = str(obj.station_type)

            st = DutyStation(st_cfg)

            # Unfinished-intent build state is authoritative on each object instance.
            # Keep identity stable: same station object transitions from unfinished to built.
            cons = st_cfg.get("console", {}) if isinstance(st_cfg, dict) else {}
            scr = st_cfg.get("screen", {}) if isinstance(st_cfg, dict) else {}
            cons_w = max(0.6, float(cons.get("width", 1.4)))
            cons_d = max(0.4, float(cons.get("depth", 0.62)))
            base_area = float(cons_w * cons_d)
            req_grey = max(2, int(math.ceil(base_area * 3.0)))
            req_screen = 1 if float(scr.get("height", 0.72)) > 0.0 else 0
            default_state = {
                "unfinished": bool(obj.station_type in ("room_control", "fabricator")),
                "job_order_id": f"job::{obj.obj_id}",
                "required_materials": {
                    "grey_block": int(req_grey),
                    "screen_block": int(req_screen),
                },
                "delivered_materials": {
                    "grey_block": 0,
                    "screen_block": 0,
                },
            }
            state = dict(default_state)
            state.update(dict(getattr(obj, "build_state", {}) or {}))
            st.set_unfinished_state(
                unfinished=bool(state.get("unfinished", False)),
                required=dict(state.get("required_materials", {})),
                delivered=dict(state.get("delivered_materials", {})),
                job_order_id=str(state.get("job_order_id", f"job::{obj.obj_id}")),
            )
            st._placed_ref = obj  # runtime persistence hook

            menu = None
            if obj.station_type == "fabricator":
                from fabricator_station import FabricatorStation
                ws_path = os.path.join(cfg_dir, "workspace.yaml")
                pal_path = os.path.join(cfg_dir, "palette.yaml")
                menu = FabricatorStation.from_yaml(ws_path, pal_path)
            elif obj.station_type == "simulator":
                from simulator_station import SimulatorStation
                pal_cfg = _load_yaml(os.path.join(cfg_dir, "simulator_palette.yaml"))
                coevo_cfg = _load_yaml(os.path.join(cfg_dir, "coevolution.yaml"))
                glass_cfg = _load_yaml(os.path.join(cfg_dir, "glass_room.yaml"))
                wb_min = np.array([-0.20, -0.05, 0.02])
                wb_max = np.array([0.20, 0.35, 0.38])
                menu = SimulatorStation(pal_cfg, coevo_cfg, glass_cfg,
                                       wb_min, wb_max, win_w, win_h)
                menu.init_gl()

            if menu is not None:
                st.menu = menu

            st.build_gl()

            return st
        except Exception as exc:
            print(f"[RoomWorkspace] duty station build failed: {exc}", flush=True)
            return None

    def _build_fabricator(self, obj: PlacedDutyStation,
                           win_w: int, win_h: int):
        return self._build_duty_station(obj, win_w, win_h)

    def _build_simulator(self, obj: PlacedDutyStation,
                          win_w: int, win_h: int):
        return self._build_duty_station(obj, win_w, win_h)

    def _build_enclosure_simulator(self, obj: PlacedEnclosure,
                                    win_w: int, win_h: int):
        """Build a SimulatorWorkspace (no GL) for an enclosure that has a
        ``simulator`` config block.  Returns a lightweight wrapper that just
        holds the workspace — the enclosure geometry is rendered by RoomStation."""
        try:
            from simulator_workspace import SimulatorWorkspace
            from coevolution_scheduler import CoevolutionScheduler

            sim_cfg = obj.simulator or {}
            coevo_path = "configs/duty_stations/simulator/coevolution.yaml"
            pal_path   = "configs/duty_stations/simulator/simulator_palette.yaml"
            coevo_cfg  = _load_yaml(coevo_path)
            pal_cfg    = _load_yaml(pal_path)
            n_items    = int(sim_cfg.get("n_items", 4))
            ws = SimulatorWorkspace(pal_cfg, coevo_cfg, n_items=n_items)
            plugin_id = sim_cfg.get("plugin_id")
            if plugin_id:
                ws.select_plugin(plugin_id)
            # Attach workspace to placed object so RoomStation can draw stats
            obj._simulator_ws = ws  # type: ignore[attr-defined]
            return None  # no GL station object; rendered inline by RoomStation
        except Exception as exc:
            print(f"[RoomWorkspace] enclosure sim build failed: {exc}", flush=True)
            return None

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "objects": [obj.to_dict() for obj in self._objects.values()]
        }

    def save_scene(self, path: str = None) -> bool:
        if path is None:
            path = os.path.join(self._config_dir, "scene.yaml")
        return _dump_yaml(self.to_dict(), path)

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_clean(self):
        self._dirty = False
