"""Gateway: the singleton outermost-WAN world object.

Semantics
---------
The gateway is a small, deliberately frustrating-to-access world prop
with one naive prong on top and an antenna mesh.  It refuses to join
any star regardless of physical contact, so cables to it never extend
a star — they cross a hard boundary.

Only **one** gateway can exist program-wide.  When a payload is pushed
to its naive port, the gateway processes synchronously on whatever
thread invoked it (typically the controller tick), out-competing all
other star traffic.  This is the in-fiction representation of "we are
giving this one item thread-blocking access that overrides all
subnetworks everywhere."

Subscriber model
----------------
Exactly one subscriber callable can be installed via
:meth:`GatewayObject.set_subscriber`.  When the gateway's input port
has data, :meth:`GatewayObject.pump` drains it and calls the
subscriber once per item, blocking the caller.  ``pump`` is wired into
the controller's tick by :meth:`GatewayObject.attach_to_controller`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from controls import (
    PortClass,
    PortRole,
    get_control_graph,
    register_input_endpoint,
    register_output_endpoint,
)
from naive_graph import Address, get_naive_graph_controller


_GATEWAY_LOCK = threading.RLock()
_GATEWAY: Optional["GatewayObject"] = None


@dataclass(slots=True)
class GatewayObject:
    """Singleton gateway.  Construct via :func:`get_gateway`."""

    owner_id: str = "_gateway"
    world_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    antenna_height: float = 0.6
    capacity: int = 0  # 0 = unbounded
    _subscriber: Optional[Callable[[Any], None]] = field(default=None, init=False)
    _processed_total: int = field(default=0, init=False)
    _last_invoked_monotonic: Optional[float] = field(default=None, init=False)
    _registered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._register()

    def _register(self) -> None:
        if self._registered:
            return
        register_input_endpoint(
            owner_id=self.owner_id,
            endpoint_id="naive_in",
            capacity=self.capacity,
            port_class=PortClass.NAIVE_PRONG,
            port_role=PortRole.GATEWAY,
            metadata={
                "world_pos": tuple(self.world_pos),
                "antenna_height": float(self.antenna_height),
                "is_gateway": True,
            },
        )
        register_output_endpoint(
            owner_id=self.owner_id,
            endpoint_id="naive_out",
            capacity=self.capacity,
            port_class=PortClass.NAIVE_PRONG,
            port_role=PortRole.GATEWAY,
            metadata={
                "world_pos": tuple(self.world_pos),
                "antenna_height": float(self.antenna_height),
                "is_gateway": True,
            },
        )
        self._registered = True

    # ----- addresses -----

    @property
    def in_address(self) -> Address:
        return Address(self.owner_id, "naive_in")

    @property
    def out_address(self) -> Address:
        return Address(self.owner_id, "naive_out")

    # ----- subscriber -----

    def set_subscriber(self, fn: Optional[Callable[[Any], None]]) -> None:
        """Install or clear the (single) subscriber callable.

        Setting a new subscriber when one already exists overwrites it
        — by design there is only one slot.
        """
        self._subscriber = fn

    @property
    def has_subscriber(self) -> bool:
        return self._subscriber is not None

    @property
    def processed_total(self) -> int:
        return self._processed_total

    def pump(self) -> int:
        """Drain the gateway input synchronously, invoking the subscriber.

        Runs on the calling thread; intended to be called from the
        controller tick or from a manual driver.  Returns the number of
        items delivered.
        """
        ctx = get_control_graph().fifo_context(self.owner_id)
        items = ctx.input("naive_in").drain()
        if not items:
            return 0
        sub = self._subscriber
        import time as _t
        self._last_invoked_monotonic = _t.monotonic()
        for item in items:
            if sub is not None:
                # Unwrap Message envelopes if present.
                payload = getattr(item, "payload", item)
                sub(payload)
            self._processed_total += 1
        return len(items)

    # ----- controller wiring -----

    def attach_to_controller(self) -> None:
        """No-op placeholder.

        The controller does not own the gateway; the host application
        decides where to call :meth:`pump` (the natural place is right
        after ``controller.tick()`` so any payload that just landed on
        the gateway's input is processed before the next tick).
        """
        # Nothing to wire — kept for API symmetry / future hooks.
        return None

    def get_stats(self) -> dict:
        return {
            "owner_id": self.owner_id,
            "world_pos": tuple(self.world_pos),
            "antenna_height": float(self.antenna_height),
            "has_subscriber": self.has_subscriber,
            "processed_total": self._processed_total,
            "last_invoked_monotonic": self._last_invoked_monotonic,
        }


def get_gateway(
    *,
    owner_id: str = "_gateway",
    world_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    antenna_height: float = 0.6,
) -> GatewayObject:
    """Return the singleton gateway, constructing it on first call.

    Subsequent calls return the existing instance regardless of
    arguments (which is the in-fiction point: there is only one).
    """
    global _GATEWAY
    with _GATEWAY_LOCK:
        if _GATEWAY is None:
            _GATEWAY = GatewayObject(
                owner_id=owner_id,
                world_pos=tuple(world_pos),
                antenna_height=float(antenna_height),
            )
        return _GATEWAY


def gateway_exists() -> bool:
    return _GATEWAY is not None


__all__ = ["GatewayObject", "get_gateway", "gateway_exists"]
