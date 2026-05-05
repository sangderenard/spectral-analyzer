"""Cable world objects.

A :class:`CableObject` is a deployable cable in the scene.  It carries
two attachment points that the player snaps to actual ports (a naive
prong, a star's WAN port, or an advanced flush plate).  When both ends
are plugged, the cable registers a :class:`naive_graph.Cable` with the
controller; unplugging removes it.

Cables come in three variants matching the port classes they can mate
with — see :class:`naive_graph.CableKind`.  The class-class compatibility
is enforced at construction by ``Cable.__post_init__`` so it is
physically impossible to plug a naive prong cable into an advanced
flush bore (or vice versa).

This module is a data-layer stub.  The visual wire / catenary / world
mesh and the player's grab-and-snap UX live in the world-object layer
and key off ``CableObject.endpoint_world_positions()``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

from naive_graph import (
    Address,
    Cable,
    CableKind,
    get_naive_graph_controller,
)


_CABLE_ID_COUNTER = itertools.count(1)


def _next_cable_id(prefix: str = "cable") -> str:
    return f"{prefix}.{next(_CABLE_ID_COUNTER)}"


@dataclass(slots=True)
class CableObject:
    """A deployable cable whose two ends snap to ports in the world.

    Parameters
    ----------
    kind:
        :class:`naive_graph.CableKind` value.  Determines which port
        classes the ends can mate with.
    cable_id:
        Optional explicit id.  Auto-generated if empty.
    world_curve:
        Free-form list of ``(x, y, z)`` control points the renderer can
        use to draw a catenary or straight wire.  Empty by default.
    metadata:
        Free-form metadata copied into the registered :class:`Cable`.
    """

    kind: str = CableKind.NAIVE
    cable_id: str = ""
    world_curve: list[tuple[float, float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    end_a: Optional[Address] = field(default=None, init=False)
    end_b: Optional[Address] = field(default=None, init=False)
    _cable: Optional[Cable] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.cable_id:
            self.cable_id = _next_cable_id()

    # ---- snapping ----

    def attach_a(self, address_or_str: "Address | str | tuple") -> None:
        self.end_a = self._coerce(address_or_str)
        self._maybe_register()

    def attach_b(self, address_or_str: "Address | str | tuple") -> None:
        self.end_b = self._coerce(address_or_str)
        self._maybe_register()

    def unplug(self) -> None:
        """Remove the cable from the controller.  The ends remain set."""
        ctrl = get_naive_graph_controller()
        if self._cable is not None:
            ctrl.unregister_cable(self.cable_id)
            self._cable = None

    @property
    def is_plugged(self) -> bool:
        return self._cable is not None

    def endpoint_world_positions(self) -> tuple[
        Optional[tuple[float, float, float]],
        Optional[tuple[float, float, float]],
    ]:
        """Best-effort lookup of each end's world position from its host plate.

        Returns ``(pos_a, pos_b)`` where each is the plate's
        ``world_pos`` from the endpoint metadata (when present), or
        ``None`` when the host doesn't expose one.  Used by the
        renderer to draw the wire.
        """
        return (self._world_pos_for(self.end_a), self._world_pos_for(self.end_b))

    # ---- internal ----

    def _maybe_register(self) -> None:
        if self.end_a is None or self.end_b is None or self._cable is not None:
            return
        cable = Cable(
            cable_id=self.cable_id,
            kind=self.kind,
            a=self.end_a,
            b=self.end_b,
            metadata=dict(self.metadata),
        )
        get_naive_graph_controller().register_cable(cable)
        self._cable = cable

    @staticmethod
    def _coerce(value: "Address | str | tuple") -> Address:
        if isinstance(value, Address):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            return Address(str(value[0]), str(value[1]))
        s = str(value)
        owner, _, ep = s.partition("/")
        if not ep:
            raise ValueError(f"address must look like 'owner/endpoint': {value!r}")
        return Address(owner, ep)

    @staticmethod
    def _world_pos_for(addr: Optional[Address]) -> Optional[tuple[float, float, float]]:
        if addr is None:
            return None
        from controls import get_control_graph
        graph = get_control_graph()
        owner_key = (
            "global" if addr.owner_id == "global" else f"object/{addr.owner_id}"
        )
        for axis in ("inputs", "outputs"):
            node = graph.find(f"{owner_key}/{axis}/{addr.endpoint_id}")
            if node is None:
                continue
            wp = node.payload.get("world_pos")
            if wp is not None:
                return tuple(wp)  # type: ignore[return-value]
        return None


__all__ = ["CableObject"]
