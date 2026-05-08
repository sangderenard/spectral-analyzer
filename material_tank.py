"""
material_tank.py — Float-valued material supply node.

A MaterialTank holds a quantity of one named material measured in surface
area (m²).  It is the buffer between star-network supply deliveries and the
RoomSurfaceSet progressive fill pipeline.

The tank is designed to be a placeable tile: it has a world position so it
can appear in the room as a visible supply terminal, and it exposes
deposit / withdraw / drain operations consumed by the Room's fill scheduler.

Material quantities are always in m² of coverage, matching the float domain
that RoomSurface.fill_batch() and deliver_area() operate in.
"""

from __future__ import annotations

from typing import Optional


class MaterialTank:
    """
    Float-valued buffer for one named material (measured in m² of coverage area).

    Parameters
    ----------
    material_name
        Registered name in configs/materials/ (e.g. "painted_plaster_wall").
    capacity_m2
        Maximum m² the tank can hold.
    world_pos
        World-space (x, y, z) placement — used for HUD display and supply routing.
    """

    def __init__(
        self,
        material_name: str,
        capacity_m2:   float = 100.0,
        world_pos:     tuple = (0.0, 0.0, 0.0),
    ) -> None:
        self.material_name: str   = material_name
        self.capacity_m2:   float = max(0.0, float(capacity_m2))
        self.world_pos:     tuple = tuple(float(x) for x in world_pos)
        self._stored_m2:    float = 0.0
        self._material_id:  int   = -1   # resolved lazily from MaterialDatabase

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def stored_m2(self) -> float:
        return self._stored_m2

    @property
    def fill_fraction(self) -> float:
        """0.0 = empty, 1.0 = full."""
        if self.capacity_m2 <= 0.0:
            return 0.0
        return min(1.0, self._stored_m2 / self.capacity_m2)

    @property
    def headroom_m2(self) -> float:
        """How much more material can be accepted."""
        return max(0.0, self.capacity_m2 - self._stored_m2)

    @property
    def is_empty(self) -> bool:
        return self._stored_m2 <= 0.0

    @property
    def is_full(self) -> bool:
        return self._stored_m2 >= self.capacity_m2

    # ── Flow operations ───────────────────────────────────────────────────────

    def deposit(self, area_m2: float) -> float:
        """
        Add material from an external supply.

        Returns the amount actually accepted (may be less than area_m2 if
        the tank is near capacity).
        """
        accepted = min(float(area_m2), self.headroom_m2)
        if accepted > 0.0:
            self._stored_m2 += accepted
        return accepted

    def withdraw(self, area_m2: float) -> float:
        """
        Remove material for consumption (e.g. surface fill).

        Returns the amount actually given (may be less than area_m2 if the
        tank doesn't have enough stored).
        """
        given = min(float(area_m2), self._stored_m2)
        if given > 0.0:
            self._stored_m2 = max(0.0, self._stored_m2 - given)
        return given

    def has_material_for(self, area_m2: float) -> bool:
        """True if the tank can supply at least area_m2 m²."""
        return self._stored_m2 >= float(area_m2)

    def drain_all(self) -> float:
        """Remove and return everything stored."""
        amount = self._stored_m2
        self._stored_m2 = 0.0
        return amount

    # ── Material ID resolution ────────────────────────────────────────────────

    def resolve_material_id(self) -> int:
        """
        Look up the integer index for this material in the global
        MaterialDatabase.  Result is cached; returns -1 if not yet registered.

        The material is registered automatically when the game loads its YAML
        via _load_material_yaml(); this method only queries, never loads.
        """
        if self._material_id >= 0:
            return self._material_id
        try:
            from material_db import MaterialDatabase
            t   = MaterialDatabase.instance().build_tensors()
            idx = t["index"].get(self.material_name)
            if idx is not None:
                self._material_id = int(idx)
        except Exception:
            pass
        return self._material_id

    # ── Serialisation ─────────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        return {
            "material_name": self.material_name,
            "capacity_m2":   self.capacity_m2,
            "stored_m2":     self._stored_m2,
            "fill_fraction": self.fill_fraction,
            "world_pos":     list(self.world_pos),
        }

    def __repr__(self) -> str:
        return (f"MaterialTank({self.material_name!r}, "
                f"{self._stored_m2:.2f}/{self.capacity_m2:.1f} m²)")
