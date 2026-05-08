"""
room.py — Runtime container for a deployed room.

Aggregates everything the room control duty station produces:
  - ClipConfig / RegionAuthority (collision rules)
  - GravityConfig (physics model, separate from clip)
  - RoomSurfaceSet (progressive material fill geometry)
  - Material palette (which named material fills each surface slot)
  - Surface profiles (tessellation profile per material, loaded from YAML)
  - MaterialTank dict (float m² supply buffers, one per palette slot)
  - NaiveStarDutyStation (star network supply ports)

The Room neither renders nor computes shading.  It is a data hub that the
control station populates and other systems (graph-walk renderer, physics
solver, network scheduler) read from.
"""

from __future__ import annotations

from typing import Dict, Optional

from room_clip    import ClipConfig, ClipSurface, RegionAuthority
from room_gravity import GravityConfig, GravityModel
from material_tank import MaterialTank


# ── Port layout on the star network ──────────────────────────────────────────
# LAN prong assignments for the NaiveStarDutyStation breakout.

_PORT_MATERIAL_FLOOR = 0   # inbound: floor material supply packets
_PORT_MATERIAL_WALL  = 1   # inbound: wall material supply packets
_PORT_MATERIAL_CEIL  = 2   # inbound: ceiling material supply packets
_PORT_BUILD_CMD      = 3   # inbound: build / demolish commands
_PORT_STATUS_OUT     = 4   # outbound: room status broadcasts
_BREAKOUT_COUNT      = 8   # total LAN prongs (extra ports reserved)


class Room:
    """
    Runtime container for one deployed room.

    Parameters
    ----------
    room_id
        Unique string identifier, used as RegionAuthority region_id and
        NaiveStarDutyStation owner_id.
    workspace
        The RoomWorkspace this room is bound to.  May be None during unit
        tests; surface_set and palette will be empty until
        rebuild_surfaces() is called.
    """

    def __init__(self, room_id: str, workspace=None) -> None:
        self.room_id   = str(room_id)
        self.workspace = workspace

        # Clip + gravity — separate concerns, held together for convenience
        self.authority = RegionAuthority(self.room_id)
        self.gravity   = GravityConfig()

        # Surface geometry (RoomSurfaceSet); populated by rebuild_surfaces()
        self.surface_set = None

        # Material palette: slot name → material name
        # Slots: "floor", "wall_int", "wall_ext", "ceiling"
        self.palette: Dict[str, str] = {}

        # Surface profiles: material name → profile name (from YAML)
        self.profiles: Dict[str, str] = {}

        # Material tanks: slot name → MaterialTank
        self.tanks: Dict[str, MaterialTank] = {}

        # Star network station; attached via join_star_network()
        self._network_station = None

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_workspace(
        cls,
        workspace,
        room_id: str,
        *,
        palette:        Optional[Dict[str, str]] = None,
        profiles:       Optional[Dict[str, str]] = None,
        clip_surfaces:  ClipSurface  = ClipSurface.STANDARD,
        gravity_model:  GravityModel = GravityModel.UNIFORM,
        gravity_ms2:    float        = 9.81,
        objects_clip:   bool         = False,
        clip_authority: str          = "engine",
    ) -> "Room":
        """Build a Room from an existing RoomWorkspace."""
        room = cls(room_id, workspace)
        room.authority.clip_config = ClipConfig(
            clip_surfaces   = clip_surfaces,
            objects_clip    = objects_clip,
            clip_authority  = clip_authority,
        )
        room.gravity = GravityConfig(
            model        = gravity_model,
            strength_ms2 = gravity_ms2,
        )
        if palette:
            room.palette = dict(palette)
        if profiles:
            room.profiles = dict(profiles)
        return room

    # ── Surface geometry ──────────────────────────────────────────────────────

    def rebuild_surfaces(self, station_origin: tuple = (0.0, 0.0, 0.0)) -> None:
        """Create or replace the RoomSurfaceSet from the workspace room_cfg.

        Also rebuilds the MaterialTank dict so each palette slot has a tank
        sized to the room's total surface area.
        """
        if self.workspace is None:
            return
        room_cfg = getattr(self.workspace, "room_cfg", {})

        from room_surface import RoomSurfaceSet
        ss = RoomSurfaceSet()
        ss.build_from_room_cfg(room_cfg, station_origin)
        ss.mid_to_name = {
            0: self.palette.get("floor",    "construction_glass"),
            1: self.palette.get("wall_int", "construction_glass"),
            2: self.palette.get("wall_ext", "construction_glass"),
            3: self.palette.get("ceiling",  "construction_glass"),
        }
        self.surface_set = ss
        self._rebuild_tanks()

    def _rebuild_tanks(self) -> None:
        """Create one MaterialTank per palette slot."""
        self.tanks = {}
        for slot, mat_name in self.palette.items():
            tank = MaterialTank(
                material_name = mat_name,
                capacity_m2   = 200.0,
                world_pos     = (0.0, 0.0, 0.0),
            )
            self.tanks[slot] = tank

    # ── Star network ──────────────────────────────────────────────────────────

    def join_star_network(self, star_id: str) -> None:
        """Attach this room to a NaiveStarDutyStation on the named star."""
        from naive_graph import NaiveStarDutyStation
        self._network_station = NaiveStarDutyStation(
            star_id         = star_id,
            owner_id        = self.room_id,
            breakout_count  = _BREAKOUT_COUNT,
        )

    # ── Tick: drain network inboxes ───────────────────────────────────────────

    def tick(self) -> None:
        """Process all pending messages from the star network."""
        if self._network_station is None:
            return
        try:
            from naive_graph import get_control_graph
            ctx = get_control_graph().fifo_context(self.room_id)
            for port_idx in range(_PORT_STATUS_OUT):
                ep_id = f"lan_in_{port_idx}"
                for msg in ctx.input(ep_id).drain():
                    self._handle_message(msg, port_idx)
        except Exception:
            pass

    def _handle_message(self, msg: dict, port_idx: int) -> None:
        kind = msg.get("kind", "")

        if kind == "material":
            surface = msg.get("surface", "")    # "floor", "wall_int", etc.
            mat_id  = int(msg.get("material_id", -1))
            area_m2 = float(msg.get("area_m2", 0.0))
            tank    = self.tanks.get(surface)
            if tank is not None and area_m2 > 0.0:
                tank.deposit(area_m2)
            if self.surface_set is not None and mat_id >= 0 and area_m2 > 0.0:
                # Map surface name → surface key in RoomSurfaceSet
                key_map = {
                    "floor":    ("floor",  mat_id),
                    "wall_int": ("wall_0", mat_id),
                    "wall_ext": ("wall_ext_0", mat_id),
                    "ceiling":  ("ceiling", mat_id),
                }
                sk = key_map.get(surface)
                if sk:
                    self.surface_set.deliver_area(sk[0], sk[1], area_m2)

        elif kind == "status_query":
            self._broadcast_status()

    def _broadcast_status(self) -> None:
        if self._network_station is None:
            return
        payload = self.as_dict()
        try:
            addr = self._network_station.lan_out_address(_PORT_STATUS_OUT)
            self._network_station.star.submit(payload, addr)
        except Exception:
            pass

    # ── Serialisation ─────────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        return {
            "room_id":       self.room_id,
            "palette":       dict(self.palette),
            "profiles":      dict(self.profiles),
            "clip_surfaces": self.authority.clip_config.clip_surfaces.value,
            "objects_clip":  self.authority.clip_config.objects_clip,
            "clip_authority": self.authority.clip_config.clip_authority,
            "gravity_model": self.gravity.model.value,
            "gravity_ms2":   self.gravity.strength_ms2,
            "tanks": {
                slot: tank.as_dict()
                for slot, tank in self.tanks.items()
            },
            "surface_finished": (
                self.surface_set.is_finished if self.surface_set else False
            ),
        }

    def __repr__(self) -> str:
        return (f"Room({self.room_id!r}, "
                f"palette={list(self.palette.keys())!r})")
