"""Naive star-switch network — peer stars, no central traffic.

Architecture
------------
A :class:`StarNetwork` is a tick-isolated worker that owns:

* A set of member owner ids (the spokes).
* Its own per-tick inbox of :class:`Message` envelopes.
* Per-tick stats and a configurable priority.

There is **no** controller-level traffic relay.  Cross-star delivery
happens through ordinary member ports: each star designates one or more
member ports as ``PortRole.WAN_BOUNDARY``.  A WAN port is in every other
respect a normal naive prong — it has a Fifo, it sits on a member, and a
cable can plug into it like any other.  When a star ticks and finds an
envelope addressed to a non-member, it pushes the envelope out through
the member's own output Fifo (or, if the destination address points
*into* this star's WAN input, ingests it).  Loop protection lives on the
envelope itself: a ``visited_wan`` set records which WAN port addresses
the envelope has already crossed; any star refuses to re-emit through a
WAN port already in that set.

The :class:`NaiveGraphController` is a meta-coordinator: it tracks
stars, ticks them in priority order, and keeps a registry of cables.
Cables are pure topology — the star tick reads from / writes to member
Fifos directly, so no traffic ever passes through the controller.

Duty stations
-------------
:class:`NaiveStarDutyStation` is itself a real world object that joins
its star as a member.  It owns a row of standard breakout naive prongs
(for daisy-chaining nearby modules) plus exactly one prong tagged
``PortRole.WAN_BOUNDARY`` — that single port is what makes the duty
station a router rather than just a switch.

Merging is explicit
-------------------
Cabling two stars does NOT merge them.  Each star keeps ticking
independently; envelopes flow through cabled WAN ports.  Use
``controller.merge_stars(keep, retire)`` for explicit absorbs.
"""

from __future__ import annotations

import threading
import time as _time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from controls import (
    Fifo,
    FifoContext,
    PortClass,
    PortRole,
    get_control_graph,
    port_class_of,
    port_role_of,
    register_input_endpoint,
    register_output_endpoint,
)


# ---------------------------------------------------------------------------
# Address + Message envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Address:
    owner_id: str
    endpoint_id: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.owner_id}/{self.endpoint_id}"


def _coerce_address(value: Any) -> Address:
    if isinstance(value, Address):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return Address(str(value[0]), str(value[1]))
    if isinstance(value, str) and "/" in value:
        owner, ep = value.split("/", 1)
        return Address(owner, ep)
    raise TypeError(f"cannot coerce {value!r} to Address")


@dataclass(slots=True)
class Message:
    payload: Any
    remaining: set[Address]
    deadline_monotonic: Optional[float] = None
    src: Optional[Address] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    visited_wan: set[Address] = field(default_factory=set)

    def expired(self, now: float) -> bool:
        return self.deadline_monotonic is not None and now >= self.deadline_monotonic

    def with_remaining(self, remaining: Iterable[Address]) -> "Message":
        return Message(
            payload=self.payload,
            remaining=set(remaining),
            deadline_monotonic=self.deadline_monotonic,
            src=self.src,
            metadata=dict(self.metadata),
            visited_wan=set(self.visited_wan),
        )


def make_envelope(
    payload: Any,
    to: Any,
    *,
    timeout_s: Optional[float] = None,
    src: Any = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Message:
    if isinstance(to, (Address, tuple, str)):
        remaining = {_coerce_address(to)}
    else:
        remaining = {_coerce_address(x) for x in to}
    deadline = None if timeout_s is None else _time.monotonic() + float(timeout_s)
    return Message(
        payload=payload,
        remaining=remaining,
        deadline_monotonic=deadline,
        src=None if src is None else _coerce_address(src),
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Star membership helpers
# ---------------------------------------------------------------------------

# Some owner-id namespaces are reserved for the controller's own bookkeeping.
def _is_reserved_owner(owner_id: str) -> bool:
    return owner_id.startswith("__")


# ---------------------------------------------------------------------------
# StarNetwork
# ---------------------------------------------------------------------------

class StarNetwork:
    """Tick-isolated star.  Members are real owners with real ports."""

    def __init__(
        self,
        star_id: str,
        *,
        priority: int = 0,
        default_timeout_s: Optional[float] = None,
    ) -> None:
        self.star_id = str(star_id)
        self.priority = int(priority)
        self.default_timeout_s = default_timeout_s
        self._lock = threading.RLock()
        self._members: set[str] = set()
        # owner_id -> { endpoint_id -> Fifo }
        self._inputs: dict[str, dict[str, Fifo]] = {}
        self._outputs: dict[str, dict[str, Fifo]] = {}
        # Address -> PortRole, populated during refresh.
        self._roles: dict[Address, PortRole] = {}
        self._inbox: deque[Message] = deque()
        self._delivered_total = 0
        self._dropped_expired = 0
        self._dropped_unrouted = 0
        self._loop_blocked = 0
        self._tick_count = 0
        self._last_tick_monotonic: Optional[float] = None

    # -- membership -------------------------------------------------------

    def join(self, owner_id: str) -> None:
        if _is_reserved_owner(owner_id):
            raise ValueError(f"owner {owner_id!r} is reserved")
        with self._lock:
            self._members.add(owner_id)
            self._capture(owner_id)

    def leave(self, owner_id: str) -> None:
        with self._lock:
            self._members.discard(owner_id)
            self._inputs.pop(owner_id, None)
            self._outputs.pop(owner_id, None)
            self._roles = {a: r for a, r in self._roles.items() if a.owner_id != owner_id}

    def refresh(self, owner_id: str) -> None:
        with self._lock:
            if owner_id in self._members:
                self._capture(owner_id)

    def refresh_all(self) -> None:
        with self._lock:
            for oid in list(self._members):
                self._capture(oid)

    def has_member(self, owner_id: str) -> bool:
        return owner_id in self._members

    @property
    def member_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._members)

    def _capture(self, owner_id: str) -> None:
        ctx = get_control_graph().fifo_context(owner_id)
        self._inputs[owner_id] = dict(ctx.inputs)
        self._outputs[owner_id] = dict(ctx.outputs)
        # Refresh roles for this owner's ports.
        graph = get_control_graph()
        owner_key = "global" if owner_id == "global" else f"object/{owner_id}"
        # Drop existing entries for this owner first.
        self._roles = {a: r for a, r in self._roles.items() if a.owner_id != owner_id}
        for ep_id in self._inputs[owner_id]:
            node = graph.find(f"{owner_key}/inputs/{ep_id}")
            self._roles[Address(owner_id, ep_id)] = port_role_of(node)
        for ep_id in self._outputs[owner_id]:
            node = graph.find(f"{owner_key}/outputs/{ep_id}")
            self._roles[Address(owner_id, ep_id)] = port_role_of(node)

    # -- inbox ------------------------------------------------------------

    def submit(
        self,
        payload: Any,
        to: Any,
        *,
        timeout_s: Optional[float] = None,
        src: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        msg = make_envelope(
            payload, to,
            timeout_s=timeout_s if timeout_s is not None else self.default_timeout_s,
            src=src,
            metadata=metadata,
        )
        with self._lock:
            self._inbox.append(msg)
        return msg

    # -- tick -------------------------------------------------------------

    def tick(self, now: Optional[float] = None) -> dict:
        now = _time.monotonic() if now is None else float(now)
        ingested = delivered = expired = unrouted = egressed = loop_blocked = 0
        with self._lock:
            # 0. Drain WAN-boundary *inputs* into the inbox.  These are
            #    envelopes that arrived from another star via a cable.
            for addr, role in self._roles.items():
                if role != PortRole.WAN_BOUNDARY:
                    continue
                ins = self._inputs.get(addr.owner_id, {})
                fifo = ins.get(addr.endpoint_id)
                if fifo is None:
                    continue
                for item in fifo.drain():
                    if isinstance(item, Message):
                        item.visited_wan.add(addr)
                        self._inbox.append(item)
                        ingested += 1
                    else:
                        unrouted += 1

            # 1. Pull every member output Fifo into the inbox or relay queue.
            for oid in list(self._members):
                outs = self._outputs.get(oid, {})
                for ep_id, fifo in outs.items():
                    drained = fifo.drain()
                    for item in drained:
                        if isinstance(item, Message):
                            # Mark the egress port as visited if it's a WAN
                            # boundary, so we won't re-emit through it again.
                            src_addr = Address(oid, ep_id)
                            if self._roles.get(src_addr) == PortRole.WAN_BOUNDARY:
                                item.visited_wan.add(src_addr)
                            self._inbox.append(item)
                            ingested += 1
                        else:
                            unrouted += 1

            # 2. Process the inbox.
            new_inbox: deque[Message] = deque()
            while self._inbox:
                msg = self._inbox.popleft()
                if msg.expired(now):
                    expired += 1
                    continue
                local: set[Address] = set()
                remote: set[Address] = set()
                for addr in msg.remaining:
                    if addr.owner_id in self._members and addr.endpoint_id in self._inputs.get(addr.owner_id, {}):
                        local.add(addr)
                    else:
                        remote.add(addr)
                # Local delivery.
                for addr in local:
                    fifo = self._inputs[addr.owner_id][addr.endpoint_id]
                    fifo.push(msg.payload)
                    delivered += 1
                # Remote delivery: emit through any WAN_BOUNDARY output port
                # that isn't already in visited_wan.  We use the *first*
                # eligible WAN output we find; if none, the envelope is
                # unrouted from this star's POV.
                if remote:
                    wan_outs = [
                        addr for addr, role in self._roles.items()
                        if role == PortRole.WAN_BOUNDARY
                        and addr.endpoint_id in self._outputs.get(addr.owner_id, {})
                        and addr not in msg.visited_wan
                    ]
                    if wan_outs:
                        # Prefer deterministic ordering.
                        wan_outs.sort(key=lambda a: (a.owner_id, a.endpoint_id))
                        emit_addr = wan_outs[0]
                        out_fifo = self._outputs[emit_addr.owner_id][emit_addr.endpoint_id]
                        forwarded = msg.with_remaining(remote)
                        forwarded.visited_wan.add(emit_addr)
                        out_fifo.push(forwarded)
                        egressed += 1
                    else:
                        if remote and not local:
                            # Fully blocked (loop or no WAN at all).
                            if any(
                                r == PortRole.WAN_BOUNDARY
                                and a in msg.visited_wan
                                for a, r in self._roles.items()
                            ):
                                loop_blocked += 1
                            else:
                                unrouted += 1
            # No requeue: anything still pending was either delivered or
            # routed onto a WAN output which another star will pick up.
            self._inbox = new_inbox
            self._tick_count += 1
            self._last_tick_monotonic = now
            self._delivered_total += delivered
            self._dropped_expired += expired
            self._dropped_unrouted += unrouted
            self._loop_blocked += loop_blocked
        return {
            "ingested": ingested,
            "delivered": delivered,
            "egressed": egressed,
            "expired": expired,
            "unrouted": unrouted,
            "loop_blocked": loop_blocked,
            "pending": 0,
        }

    # -- introspection ----------------------------------------------------

    @property
    def stats(self) -> dict:
        with self._lock:
            wan_addrs = [str(a) for a, r in self._roles.items() if r == PortRole.WAN_BOUNDARY]
            return {
                "star_id": self.star_id,
                "members": len(self._members),
                "total_input_ports": sum(len(d) for d in self._inputs.values()),
                "total_output_ports": sum(len(d) for d in self._outputs.values()),
                "wan_ports": wan_addrs,
                "pending": len(self._inbox),
                "delivered_total": self._delivered_total,
                "dropped_expired_total": self._dropped_expired,
                "dropped_unrouted_total": self._dropped_unrouted,
                "loop_blocked_total": self._loop_blocked,
                "tick_count": self._tick_count,
                "priority": self.priority,
                "default_timeout_s": self.default_timeout_s,
                "last_tick_monotonic": self._last_tick_monotonic,
            }


# ---------------------------------------------------------------------------
# Cable (pure topology)
# ---------------------------------------------------------------------------

class CableKind(str):
    NAIVE = "naive"
    ADVANCED = "advanced"


_CABLE_KIND_TO_CLASSES: dict[str, tuple[PortClass, PortClass]] = {
    CableKind.NAIVE: (PortClass.NAIVE_PRONG, PortClass.NAIVE_PRONG),
    CableKind.ADVANCED: (PortClass.ADVANCED_FLUSH, PortClass.ADVANCED_FLUSH),
}


def _lookup_port_class(addr: Address) -> PortClass:
    graph = get_control_graph()
    owner_key = "global" if addr.owner_id == "global" else f"object/{addr.owner_id}"
    node = graph.find(f"{owner_key}/inputs/{addr.endpoint_id}")
    if node is None:
        node = graph.find(f"{owner_key}/outputs/{addr.endpoint_id}")
    return port_class_of(node)


@dataclass(slots=True)
class Cable:
    cable_id: str
    kind: str
    a: Address
    b: Address
    plugged: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.a = _coerce_address(self.a)
        self.b = _coerce_address(self.b)
        expected = _CABLE_KIND_TO_CLASSES.get(self.kind)
        if expected is None:
            raise ValueError(f"unknown cable kind {self.kind!r}")
        cls_a = _lookup_port_class(self.a)
        cls_b = _lookup_port_class(self.b)
        ok = (
            (cls_a == expected[0] and cls_b == expected[1])
            or (cls_a == expected[1] and cls_b == expected[0])
        )
        if not ok:
            ea, eb = expected
            raise ValueError(
                f"cable kind {self.kind!r} requires port classes "
                f"{ea.value}<->{eb.value}, got {cls_a.value}<->{cls_b.value}"
            )

    def endpoints(self) -> tuple[Address, Address]:
        return (self.a, self.b)

    def involves(self, owner_id: str) -> bool:
        return self.a.owner_id == owner_id or self.b.owner_id == owner_id


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class NaiveGraphController:
    """Meta-coordinator. Owns stars and cable registry. No traffic flows here."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stars: dict[str, StarNetwork] = {}
        self._owner_to_star: dict[str, str] = {}
        self._cables: dict[str, Cable] = {}
        self._tick_count = 0

    # -- stars ------------------------------------------------------------

    def get_or_create_star(
        self,
        star_id: str,
        *,
        priority: Optional[int] = None,
        default_timeout_s: Optional[float] = None,
    ) -> StarNetwork:
        with self._lock:
            star = self._stars.get(star_id)
            if star is None:
                star = StarNetwork(
                    star_id,
                    priority=priority if priority is not None else 0,
                    default_timeout_s=default_timeout_s,
                )
                self._stars[star_id] = star
            else:
                if priority is not None:
                    star.priority = int(priority)
                if default_timeout_s is not None:
                    star.default_timeout_s = default_timeout_s
            return star

    def join_star(self, owner_id: str, star_id: str) -> None:
        with self._lock:
            prior = self._owner_to_star.get(owner_id)
            if prior == star_id:
                return
            if prior is not None:
                self._stars[prior].leave(owner_id)
            star = self._stars[star_id]
            star.join(owner_id)
            self._owner_to_star[owner_id] = star_id

    def leave_star(self, owner_id: str) -> None:
        with self._lock:
            star_id = self._owner_to_star.pop(owner_id, None)
            if star_id is not None:
                self._stars[star_id].leave(owner_id)

    def star_of(self, owner_id: str) -> Optional[str]:
        return self._owner_to_star.get(owner_id)

    def merge_stars(self, keep_id: str, retire_id: str) -> None:
        if keep_id == retire_id:
            return
        with self._lock:
            keep = self._stars[keep_id]
            retire = self._stars[retire_id]
            for oid in retire.member_ids:
                self._owner_to_star[oid] = keep_id
                retire.leave(oid)
                keep.join(oid)
            # Move pending inbox.
            with retire._lock, keep._lock:
                while retire._inbox:
                    keep._inbox.append(retire._inbox.popleft())
            # Drop cables that referenced retired-star members only.
            del self._stars[retire_id]

    @property
    def stars(self) -> list[StarNetwork]:
        with self._lock:
            return list(self._stars.values())

    # -- cables -----------------------------------------------------------

    def register_cable(self, cable: Cable) -> Cable:
        with self._lock:
            if cable.cable_id in self._cables:
                raise ValueError(f"cable {cable.cable_id!r} already registered")
            self._cables[cable.cable_id] = cable
            # Refresh both endpoints' stars so role caches are current.
            for addr in (cable.a, cable.b):
                sid = self._owner_to_star.get(addr.owner_id)
                if sid is not None:
                    self._stars[sid].refresh(addr.owner_id)
            return cable

    def unregister_cable(self, cable_id: str) -> Optional[Cable]:
        with self._lock:
            return self._cables.pop(cable_id, None)

    def cables(self) -> list[Cable]:
        with self._lock:
            return list(self._cables.values())

    def cables_touching(self, owner_id: str) -> list[Cable]:
        with self._lock:
            return [c for c in self._cables.values() if c.involves(owner_id)]

    # -- tick -------------------------------------------------------------

    def tick(self, now: Optional[float] = None) -> dict:
        now = _time.monotonic() if now is None else float(now)
        with self._lock:
            ordered = sorted(self._stars.values(), key=lambda s: (-s.priority, s.star_id))
        results: dict[str, dict] = {}
        for star in ordered:
            results[star.star_id] = star.tick(now)
        # Cable pump pass: shuttle items across every plugged cable from
        # one endpoint's output Fifo into the peer endpoint's input Fifo.
        # Cables are pure topology + this tiny copy.  For each cable we
        # do both directions in case both endpoints happen to be outputs
        # on different stars (in practice one is an output and the peer
        # is an input; the irrelevant direction simply finds nothing).
        relayed = self._pump_cables()
        with self._lock:
            self._tick_count += 1
            return {
                "tick_index": self._tick_count,
                "stars": results,
                "relayed": relayed,
            }

    def _pump_cables(self) -> int:
        """Move items across plugged cables: each end's output -> peer end's input."""
        from controls import get_control_graph
        graph = get_control_graph()
        relayed = 0
        with self._lock:
            cables = [c for c in self._cables.values() if c.plugged]
        for cable in cables:
            relayed += self._pump_one_direction(graph, cable.a, cable.b)
            relayed += self._pump_one_direction(graph, cable.b, cable.a)
        return relayed

    @staticmethod
    def _pump_one_direction(graph, src: Address, dst: Address) -> int:
        """If ``src`` resolves to an output Fifo and ``dst`` to an input Fifo,
        drain src into dst.  Otherwise no-op.
        """
        src_owner_key = "global" if src.owner_id == "global" else f"object/{src.owner_id}"
        dst_owner_key = "global" if dst.owner_id == "global" else f"object/{dst.owner_id}"
        src_node = graph.find(f"{src_owner_key}/outputs/{src.endpoint_id}")
        dst_node = graph.find(f"{dst_owner_key}/inputs/{dst.endpoint_id}")
        if src_node is None or dst_node is None:
            return 0
        src_fifo = src_node.payload.get("fifo")
        dst_fifo = dst_node.payload.get("fifo")
        if src_fifo is None or dst_fifo is None:
            return 0
        moved = 0
        for item in src_fifo.drain():
            dst_fifo.push(item)
            moved += 1
        return moved

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "tick_count": self._tick_count,
                "cable_count": len(self._cables),
                "stars": {sid: s.stats for sid, s in self._stars.items()},
            }


# ---------------------------------------------------------------------------
# Auto-wire helper
# ---------------------------------------------------------------------------

def auto_wire_naive_network(
    star_id: str = "global",
    *,
    owner_ids: Optional[Iterable[str]] = None,
    priority: int = 0,
    default_timeout_s: Optional[float] = None,
) -> StarNetwork:
    """Sweep the control graph and join every non-reserved owner into ``star_id``."""
    ctrl = get_naive_graph_controller()
    star = ctrl.get_or_create_star(
        star_id, priority=priority, default_timeout_s=default_timeout_s,
    )
    graph = get_control_graph()
    if owner_ids is None:
        owner_ids = graph.iter_owner_ids()
    for oid in owner_ids:
        if _is_reserved_owner(oid):
            continue
        if ctrl.star_of(oid) is not None:
            continue
        ctrl.join_star(oid, star_id)
    return star


# ---------------------------------------------------------------------------
# Duty station — a real world object that joins its own star
# ---------------------------------------------------------------------------

class NaiveStarDutyStation:
    """A duty station as a real star member.

    On construction it registers itself as ``owner_id`` (default
    ``f"_duty_station.{star_id}"``), reserves N standard naive-prong
    breakout ports plus one WAN_BOUNDARY prong, and joins ``star_id``.
    """

    def __init__(
        self,
        star_id: str,
        *,
        owner_id: Optional[str] = None,
        breakout_count: int = 8,
        capacity: int = 32,
        priority: int = 0,
        default_timeout_s: Optional[float] = None,
    ) -> None:
        self.star_id = str(star_id)
        self.owner_id = owner_id or f"_duty_station.{star_id}"
        self.breakout_count = int(breakout_count)
        self._ctrl = get_naive_graph_controller()
        # Register breakout ports (standard) + WAN port (boundary).
        for i in range(self.breakout_count):
            register_input_endpoint(
                owner_id=self.owner_id,
                endpoint_id=f"lan_in_{i}",
                capacity=capacity,
                port_class=PortClass.NAIVE_PRONG,
                port_role=PortRole.STANDARD,
                metadata={"breakout_index": i},
            )
            register_output_endpoint(
                owner_id=self.owner_id,
                endpoint_id=f"lan_out_{i}",
                capacity=capacity,
                port_class=PortClass.NAIVE_PRONG,
                port_role=PortRole.STANDARD,
                metadata={"breakout_index": i},
            )
        register_input_endpoint(
            owner_id=self.owner_id,
            endpoint_id="wan_in",
            capacity=capacity,
            port_class=PortClass.NAIVE_PRONG,
            port_role=PortRole.WAN_BOUNDARY,
        )
        register_output_endpoint(
            owner_id=self.owner_id,
            endpoint_id="wan_out",
            capacity=capacity,
            port_class=PortClass.NAIVE_PRONG,
            port_role=PortRole.WAN_BOUNDARY,
        )
        self._ctrl.get_or_create_star(
            self.star_id, priority=priority, default_timeout_s=default_timeout_s,
        )
        self._ctrl.join_star(self.owner_id, self.star_id)

    @property
    def star(self) -> StarNetwork:
        return self._ctrl.get_or_create_star(self.star_id)

    @property
    def wan_in_address(self) -> Address:
        return Address(self.owner_id, "wan_in")

    @property
    def wan_out_address(self) -> Address:
        return Address(self.owner_id, "wan_out")

    def lan_in_address(self, index: int) -> Address:
        return Address(self.owner_id, f"lan_in_{index}")

    def lan_out_address(self, index: int) -> Address:
        return Address(self.owner_id, f"lan_out_{index}")

    def get_stats(self) -> dict:
        s = self.star.stats
        s["duty_station_owner"] = self.owner_id
        s["breakout_count"] = self.breakout_count
        s["cables"] = [
            {
                "cable_id": c.cable_id,
                "kind": c.kind,
                "a": str(c.a),
                "b": str(c.b),
                "plugged": c.plugged,
            }
            for c in self._ctrl.cables_touching(self.owner_id)
        ]
        return s

    def get_global_stats(self) -> dict:
        return self._ctrl.stats


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_CONTROLLER: Optional[NaiveGraphController] = None
_CONTROLLER_LOCK = threading.RLock()


def get_naive_graph_controller() -> NaiveGraphController:
    global _CONTROLLER
    with _CONTROLLER_LOCK:
        if _CONTROLLER is None:
            _CONTROLLER = NaiveGraphController()
        return _CONTROLLER


__all__ = [
    "Address",
    "Message",
    "make_envelope",
    "StarNetwork",
    "Cable",
    "CableKind",
    "NaiveGraphController",
    "NaiveStarDutyStation",
    "get_naive_graph_controller",
    "auto_wire_naive_network",
]
