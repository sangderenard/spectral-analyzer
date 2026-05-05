"""Recessed flush-mount port plate (advanced/custom network).

A :class:`PortPlate` is a deployable, decorative female port set into a
host object's surface (typically a duty station or a module).  It looks
like a circular cover-plate texture with a centre bore; an
:class:`AdvancedFlushCable` plugs into the bore and visually disappears
into the centre instead of clamping onto a side prong.

In contrast, naive prong endpoints stick out at the bottom of an object
and make contact by simple proximity.  The two physical forms are
intentionally incompatible — naive cables (prong-style) cannot mate
with flush bores, and advanced cables (male-only into bore) cannot mate
with prong endpoints.  The mismatch is enforced at runtime by
:class:`naive_graph.Cable`'s constructor (it consults
:func:`controls.port_class_of` on each endpoint).

This module provides the data layer only.  Mesh / texture rendering is
deferred — once the visual layer is wired up it will read the plate's
``world_pos`` / ``normal`` / ``radius`` to position the cover plate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from controls import (
    PortClass,
    register_input_endpoint,
    register_output_endpoint,
)
from naive_graph import Address


@dataclass(slots=True)
class PortPlate:
    """A recessed female bore on a host object.

    Parameters
    ----------
    plate_id:
        Unique id for the plate within its host.
    host_owner_id:
        Owner id of the host (a duty station, a module, ...).
    direction:
        ``"input"`` or ``"output"``.  A bore can be either, but the
        plate carries exactly one direction.
    endpoint_id:
        Endpoint id to register under the host.  Defaults to
        ``f"plate.{plate_id}"``.
    world_pos:
        ``(x, y, z)`` world-space centre of the bore.  Used by the
        renderer and by the cable layer's snap-to-plate logic.
    normal:
        ``(nx, ny, nz)`` surface normal at the bore.  Cables enter
        along ``-normal``.
    radius:
        Visual radius of the cover plate in world units.
    capacity:
        FIFO capacity for the registered endpoint.
    cover_open:
        Whether the cover plate is visually open (a cable is plugged in
        or the plate is in deploy mode).  False by default.
    label:
        Human-readable label for HUD display.
    metadata:
        Free-form payload merged into the registered endpoint's
        metadata dict.
    """

    plate_id: str
    host_owner_id: str
    direction: str  # "input" | "output"
    endpoint_id: str = ""
    world_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 1.0, 0.0)
    radius: float = 0.025
    capacity: int = 0
    cover_open: bool = False
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    _registered: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.direction not in ("input", "output"):
            raise ValueError(
                f"PortPlate direction must be 'input' or 'output', "
                f"got {self.direction!r}"
            )
        if not self.endpoint_id:
            self.endpoint_id = f"plate.{self.plate_id}"

    # ---- lifecycle ----

    def deploy(self) -> Address:
        """Register the underlying ADVANCED_FLUSH endpoint and return its address.

        Idempotent: a second call returns the same address without
        re-registering.
        """
        if self._registered:
            return self.address
        meta = {
            "plate_id": self.plate_id,
            "world_pos": tuple(self.world_pos),
            "normal": tuple(self.normal),
            "radius": float(self.radius),
            "host_owner_id": self.host_owner_id,
            **dict(self.metadata),
        }
        if self.direction == "input":
            register_input_endpoint(
                owner_id=self.host_owner_id,
                endpoint_id=self.endpoint_id,
                capacity=int(self.capacity),
                label=self.label or self.plate_id,
                port_class=PortClass.ADVANCED_FLUSH,
                metadata=meta,
            )
        else:
            register_output_endpoint(
                owner_id=self.host_owner_id,
                endpoint_id=self.endpoint_id,
                capacity=int(self.capacity),
                label=self.label or self.plate_id,
                port_class=PortClass.ADVANCED_FLUSH,
                metadata=meta,
            )
        self._registered = True
        return self.address

    @property
    def address(self) -> Address:
        return Address(self.host_owner_id, self.endpoint_id)

    @property
    def is_deployed(self) -> bool:
        return self._registered

    # ---- visual state hooks ----

    def open_cover(self) -> None:
        self.cover_open = True

    def close_cover(self) -> None:
        self.cover_open = False


__all__ = ["PortPlate"]
