"""placed_object.py
==================
Data-classes for every type of object that can be placed inside a RoomWorkspace.

Hierarchy
---------
PlacedObject          base: id, label, pos (world-space), yaw_deg
  PlacedLight         point / spot light source
  PlacedCamera        named camera anchor (not the player camera)
  PlacedEnclosure     glass enclosure of any shape (rect | cyl | sphere |
                        tablet_rect | tablet_polar), optionally wrapping a
                        SimulatorWorkspace
  PlacedDutyStation   a fabricator / simulator / room-station console
  PlacedPortalFrame   doorway stub (portal logic deferred)

All classes are plain dataclasses — no GL, no pygame.
Serialisation helpers: ``to_dict()`` / ``from_dict(d)`` on each class.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arr3(v) -> np.ndarray:
    return np.array(v, dtype=np.float64)


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlacedObject:
    """Common fields for every placed object."""
    obj_id:  str
    label:   str
    pos:     np.ndarray = field(default_factory=lambda: np.zeros(3, np.float64))
    yaw_deg: float = 0.0

    # ── serialisation ─────────────────────────────────────────────────────────

    def _base_dict(self) -> dict:
        return {
            "id":      self.obj_id,
            "label":   self.label,
            "pos":     self.pos.tolist(),
            "yaw_deg": float(self.yaw_deg),
        }

    def to_dict(self) -> dict:
        return self._base_dict()

    @classmethod
    def _from_base(cls, d: dict):
        return dict(
            obj_id  = d.get("id",      _make_id("obj")),
            label   = d.get("label",   ""),
            pos     = _arr3(d.get("pos", [0, 0, 0])),
            yaw_deg = float(d.get("yaw_deg", 0.0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Light
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlacedLight(PlacedObject):
    """A point or spot light source.

    Lights are registered in RoomWorkspace and passed to the main render
    loop as scene data.  Shadow casting is deferred — only the first few
    lights are used as Phong light_v inputs for now.
    """
    kind:           str            = "point"          # "point" | "spot"
    color:          tuple          = (1.0, 0.95, 0.88)
    intensity:      float          = 1.0
    radius_m:       float          = 8.0
    spot_angle_deg: float          = 45.0             # used when kind == "spot"

    def to_dict(self) -> dict:
        d = self._base_dict()
        d.update(type="light", kind=self.kind,
                 color=list(self.color), intensity=self.intensity,
                 radius_m=self.radius_m, spot_angle_deg=self.spot_angle_deg)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlacedLight":
        c = d.get("color", [1.0, 0.95, 0.88])
        return cls(
            **cls._from_base(d),
            kind           = str(d.get("kind", "point")),
            color          = tuple(float(x) for x in c),
            intensity      = float(d.get("intensity", 1.0)),
            radius_m       = float(d.get("radius_m", 8.0)),
            spot_angle_deg = float(d.get("spot_angle_deg", 45.0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Camera anchor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlacedCamera(PlacedObject):
    """A physical camera item in the room.

    The player can walk up to it and press E to enter IN_CAMERA mode, at
    which point arrow keys pan/tilt the armature mechanically and +/- adjust
    focal_mm.  The GL camera is driven by focal_mm via Camera.fov_y_rad() so
    the viewport always reflects the actual ray-tracing lens configuration.

    ``mesh_id`` names a mesh preset (e.g. "camera_35mm", "camera_video").
    ``lens_name`` is the YAML lens file to load for min/max focal range.

    Serialisation round-trips all mutable state so a saved scene remembers
    where the camera was pointing and what focal length was set.
    """
    # Physical/optical spec (loaded from scene.yaml, not changed at runtime)
    mesh_id:            str   = "camera_35mm"
    sensor_name:        str   = "full_frame_35mm"  # SensorSpec YAML key
    lens_name:          str   = "standard_35mm"    # LensSpec YAML key
    focal_min_mm:       float = 35.0               # minimum focal length for this lens
    focal_max_mm:       float = 35.0               # maximum (> min means zoom lens)
    tilt_min_deg:       float = -80.0
    tilt_max_deg:       float =  80.0
    interaction_radius: float =  1.5               # metres — player proximity threshold

    # Live mutable armature state (saved with scene)
    pan_deg:            float = 0.0        # yaw of the camera head, world degrees
    tilt_deg:           float = 0.0        # pitch of the camera head, degrees
    focal_mm:           float = 35.0       # current focal length (driven by +/- keys)

    def to_dict(self) -> dict:
        d = self._base_dict()
        d.update(
            type               = "camera",
            mesh_id            = self.mesh_id,
            sensor_name        = self.sensor_name,
            lens_name          = self.lens_name,
            focal_min_mm       = self.focal_min_mm,
            focal_max_mm       = self.focal_max_mm,
            tilt_min_deg       = self.tilt_min_deg,
            tilt_max_deg       = self.tilt_max_deg,
            interaction_radius = self.interaction_radius,
            pan_deg            = self.pan_deg,
            tilt_deg           = self.tilt_deg,
            focal_mm           = self.focal_mm,
        )
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlacedCamera":
        return cls(
            **cls._from_base(d),
            mesh_id            = str(d.get("mesh_id",            "camera_35mm")),
            sensor_name        = str(d.get("sensor_name",        "full_frame_35mm")),
            lens_name          = str(d.get("lens_name",          "standard_35mm")),
            focal_min_mm       = float(d.get("focal_min_mm",      35.0)),
            focal_max_mm       = float(d.get("focal_max_mm",      35.0)),
            tilt_min_deg       = float(d.get("tilt_min_deg",     -80.0)),
            tilt_max_deg       = float(d.get("tilt_max_deg",      80.0)),
            interaction_radius = float(d.get("interaction_radius", 1.5)),
            pan_deg            = float(d.get("pan_deg",            0.0)),
            tilt_deg           = float(d.get("tilt_deg",           0.0)),
            focal_mm           = float(d.get("focal_mm",           35.0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Enclosure  (glass jar / tablet)
# ─────────────────────────────────────────────────────────────────────────────

#: Valid shape identifiers understood by enclosure_geometry.py
ENCLOSURE_SHAPES = frozenset({
    "rect",         # rectangular bell-jar
    "cyl",          # cylindrical shell
    "sphere",       # full UV-sphere shell (sits on pedestal)
    "tablet_rect",  # two parallel flat rectangular panes
    "tablet_polar", # two parallel circular disk panes
})


@dataclass
class PlacedEnclosure(PlacedObject):
    """A glass enclosure of configurable shape, optionally wrapping a simulator.

    ``dims`` holds shape-specific dimension keys (see enclosure_geometry.py):

    * ``rect``        — width_m, depth_m, height_m, glass_thickness_m, bevel_segs
    * ``cyl``         — radius_m, height_m, glass_thickness_m, lon_segs
    * ``sphere``      — radius_m, glass_thickness_m, lat_segs, lon_segs,
                         pedestal_radius_m, pedestal_height_m
    * ``tablet_rect`` — width_m, height_m, gap_m, glass_thickness_m
    * ``tablet_polar``— radius_m, gap_m, glass_thickness_m, segs
    """
    shape:           str        = "rect"
    dims:            Dict[str, Any] = field(default_factory=dict)
    glass_color:     tuple      = (0.15, 0.55, 0.90, 0.28)
    skirt_color:     tuple      = (0.10, 0.12, 0.16, 0.85)
    wireframe_color: tuple      = (0.20, 0.80, 1.00, 0.60)
    # optional embedded simulator config
    simulator:       Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        d = self._base_dict()
        d.update(
            type            = "enclosure",
            shape           = self.shape,
            dims            = dict(self.dims),
            glass_color     = list(self.glass_color),
            skirt_color     = list(self.skirt_color),
            wireframe_color = list(self.wireframe_color),
        )
        if self.simulator is not None:
            d["simulator"] = dict(self.simulator)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlacedEnclosure":
        def _rgba(key, default):
            v = d.get(key, default)
            while len(v) < 4:
                v = list(v) + [1.0]
            return tuple(float(x) for x in v[:4])

        return cls(
            **cls._from_base(d),
            shape           = str(d.get("shape", "rect")),
            dims            = dict(d.get("dims", {})),
            glass_color     = _rgba("glass_color",     [0.15, 0.55, 0.90, 0.28]),
            skirt_color     = _rgba("skirt_color",     [0.10, 0.12, 0.16, 0.85]),
            wireframe_color = _rgba("wireframe_color", [0.20, 0.80, 1.00, 0.60]),
            simulator       = d.get("simulator"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Duty station reference
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlacedDutyStation(PlacedObject):
    """Reference to an external duty-station type to be instantiated.

    ``station_type`` is one of "fabricator" | "simulator" | "room".
    ``config_dir`` is the directory (relative to project root) from which
    the station's YAML files are loaded.
    ``interaction_radius`` is the proximity threshold (metres) at which the
    player sees the interact hint and can press E.
    """
    station_type:       str   = "fabricator"
    config_dir:         str   = "configs/duty_stations/fabricator"
    interaction_radius: float = 2.0
    build_state:        Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self._base_dict()
        d.update(type="duty_station",
                 station_type=self.station_type,
                 config_dir=self.config_dir,
                 interaction_radius=self.interaction_radius)
        if self.build_state:
            d["build_state"] = dict(self.build_state)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlacedDutyStation":
        return cls(
            **cls._from_base(d),
            station_type       = str(d.get("station_type", "fabricator")),
            config_dir         = str(d.get("config_dir",  "")),
            interaction_radius = float(d.get("interaction_radius", 2.0)),
            build_state        = dict(d.get("build_state", {})),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Portal frame  (stub)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlacedPortalFrame(PlacedObject):
    """Visual arch/doorway stub.  Portal traversal logic is deferred.

    ``target_room_id`` names the destination room; leave blank until
    multiple rooms are implemented.
    """
    target_room_id: str = ""

    def to_dict(self) -> dict:
        d = self._base_dict()
        d.update(type="portal", target_room_id=self.target_room_id)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlacedPortalFrame":
        return cls(
            **cls._from_base(d),
            target_room_id = str(d.get("target_room_id", "")),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_MAP = {
    "light":        PlacedLight,
    "camera":       PlacedCamera,
    "enclosure":    PlacedEnclosure,
    "duty_station": PlacedDutyStation,
    "portal":       PlacedPortalFrame,
}


def placed_object_from_dict(d: dict) -> PlacedObject:
    """Deserialise a placed-object dict (as stored in scene.yaml)."""
    t = str(d.get("type", "enclosure"))
    cls = _TYPE_MAP.get(t)
    if cls is None:
        raise ValueError(f"Unknown placed-object type {t!r}")
    return cls.from_dict(d)
