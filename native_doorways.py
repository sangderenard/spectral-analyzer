"""Native walker adapter for Turing's declared AbstractUI doors.

Consumes abstract_ui_doorways records, not a second world/service schema.
The host supplies actual aperture geometry, applies returned state, cancels
navigation on crossing, and loads islands in response to the returned event.
No renderer, transport, scene creation, or authority is owned here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

Vec3 = tuple[float, float, float]


def _v(value: Sequence[float]) -> Vec3:
    if len(value) != 3 or not all(math.isfinite(float(x)) for x in value):
        raise ValueError("expected three finite coordinates")
    return tuple(float(x) for x in value)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(v):
    n = math.sqrt(_dot(v, v))
    if n < 1e-12:
        raise ValueError("zero frame vector")
    return tuple(x/n for x in v)


def _rotate_between(v, source, target):
    """Shortest-arc quaternion rotation, including antiparallel normals."""
    a, b = _unit(source), _unit(target)
    cosine = max(-1.0, min(1.0, _dot(a, b)))
    if cosine > 1 - 1e-12:
        return tuple(v)
    if cosine < -1 + 1e-12:
        axis = _unit(_cross(a, (1, 0, 0) if abs(a[0]) < .9 else (0, 1, 0)))
        return tuple(2*_dot(axis, v)*axis[i]-v[i] for i in range(3))
    axis = _unit(_cross(a, b))
    sine = math.sqrt(max(0, 1-cosine*cosine))
    cross = _cross(axis, v)
    return tuple(v[i]*cosine+cross[i]*sine+axis[i]*_dot(axis, v)*(1-cosine) for i in range(3))


@dataclass(frozen=True)
class Aperture:
    """Host aperture in metres. Normal/tangent face OUT of the host.

    Opening ``facing=inward`` reverses these two axes as in the living map.
    ``radius`` is the existing radial chart radius, normally
    max(.18, min(width, height)/2), not a portal-frame mesh bounding radius.
    """
    center: Vec3
    normal: Vec3
    tangent: Vec3
    bitangent: Vec3 = (0, 1, 0)
    radius: float = .5

    def __post_init__(self):
        for name in ("center", "normal", "tangent", "bitangent"):
            object.__setattr__(self, name, _v(getattr(self, name)))
        for name in ("normal", "tangent", "bitangent"):
            object.__setattr__(self, name, _unit(getattr(self, name)))
        if any(abs(_dot(getattr(self, a), getattr(self, b))) > 1e-6
               for a, b in (("normal", "tangent"), ("normal", "bitangent"), ("tangent", "bitangent"))):
            raise ValueError("aperture frame must be orthogonal")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("aperture radius must be positive")


@dataclass(frozen=True)
class DoorCrossing:
    source: str
    target: str | None
    position: Vec3
    velocity: Vec3
    facing: Vec3
    cooldown_until_ms: float
    event: Mapping[str, Any] | None = None
    cancel_navigation: bool = True


@dataclass
class DoorRuntime:
    """Paired immediate doors; curved probabilistic tubes remain their own runtime.

    ``geometry`` keys are opening identities or their ``coincident_with`` anchors.
    Duplicate OUT ends for a pair are rejected instead of selecting an arbitrary
    destination. Geometry is read on each step, so moving hosts may update it.
    """
    openings: Sequence[Mapping[str, Any]]
    geometry: Mapping[str, Aperture]
    cooldown_ms: float = 240.0
    dial_positions: dict[str, int] = field(default_factory=dict)
    treadmill_offsets: dict[str, int] = field(default_factory=dict)
    cooldowns: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        ids = [o["identity"] for o in self.openings]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate opening identity")
        for o in self.openings:
            if o.get("dial"):
                count = o.get("dial_positions")
                position = o.get("dial_position")
                if not isinstance(count, int) or count < 1 or not isinstance(position, int) or not 0 <= position < count:
                    raise ValueError("invalid dial declaration")
            if o.get("treadmill"):
                t = o["treadmill"]
                if (any(not isinstance(t.get(k), int) for k in ("length", "window", "offset"))
                        or not 1 <= t["window"] < t["length"] or not 0 <= t["offset"] < t["length"]):
                    raise ValueError("invalid treadmill declaration")

    def select_dial(self, dial: str, position: int) -> None:
        candidates = [o for o in self.openings if o.get("dial") == dial]
        if not candidates:
            raise KeyError(dial)
        if not isinstance(position, int) or not any(o["dial_position"] == position for o in candidates):
            raise ValueError("dial position is not declared")
        self.dial_positions[dial] = position

    def scroll_treadmill(self, opening_identity: str, offset: int) -> dict[str, int]:
        """Select a contiguous window; host materializes it, never the full sequence.

        Offsets match Turing Treadmill.address: 0 <= offset < length. Last
        windows can be shorter. No invented URL, wrapping or world identity.
        """
        o = next((o for o in self.openings if o["identity"] == opening_identity), None)
        if o is None or not o.get("treadmill"):
            raise KeyError(opening_identity)
        t = o["treadmill"]
        if not isinstance(offset, int) or not 0 <= offset < t["length"]:
            raise ValueError("offset is off the treadmill")
        # Entry and return share their declaration's anchor, not separate state.
        key = o.get("coincident_with", opening_identity)
        self.treadmill_offsets[key] = offset
        return {**t, "offset": offset}

    def _enabled(self, o):
        return not o.get("dial") or o["dial_position"] == self.dial_positions.get(o["dial"], 0)

    def _frame(self, o):
        frame = self.geometry.get(o["identity"], self.geometry.get(o.get("coincident_with")))
        if frame is None:
            raise KeyError(f"no host aperture for {o['identity']}")
        sign = -1 if o.get("facing") == "inward" else 1
        return frame, tuple(sign*x for x in frame.normal), tuple(sign*x for x in frame.tangent)

    def step(self, actor: str, previous: Sequence[float], position: Sequence[float],
             velocity: Sequence[float] = (0, 0, 0), facing: Sequence[float] = (1, 0, 0), *,
             radius: float = .001, now_ms: float, is_walker: bool = True) -> DoorCrossing | None:
        previous, position, velocity, facing = map(_v, (previous, position, velocity, facing))
        if not math.isfinite(radius) or radius < 0 or not math.isfinite(now_ms):
            raise ValueError("invalid body radius or time")
        if now_ms < self.cooldowns.get(actor, -math.inf):
            return None
        for source in self.openings:
            if source.get("transition") != "door" or source.get("port_role") != "in" or not self._enabled(source):
                continue
            frame, normal, tangent = self._frame(source)
            before = _dot(tuple(previous[i]-frame.center[i] for i in range(3)), normal)
            after = _dot(tuple(position[i]-frame.center[i] for i in range(3)), normal)
            crossed = before > 0 and after <= 0
            touched = 0 < after <= max(.12, radius+.06) and before > after+1e-5
            if not (crossed or touched):
                continue
            fraction = before/(before-after) if crossed and abs(before-after) >= 1e-8 else 1
            point = tuple(previous[i]+(position[i]-previous[i])*fraction-frame.center[i] for i in range(3))
            u, v = _dot(point, tangent), _dot(point, frame.bitangent)
            if math.hypot(u, v) > frame.radius+radius:
                continue
            targets = [o for o in self.openings if o.get("port_role") == "out"
                       and o.get("transition") == "door" and o.get("pair") == source.get("pair")
                       and source.get("pair") and self._enabled(o)
                       and o.get("aperture_class", "person") == source.get("aperture_class", "person")]
            if len(targets) > 1:
                raise ValueError(f"ambiguous door pair {source.get('pair')}")
            if targets:
                target = targets[0]
                dest, dest_normal, dest_tangent = self._frame(target)
                result_position = tuple(dest.center[i]+dest_tangent[i]*u+dest.bitangent[i]*v
                                        +dest_normal[i]*max(.09, radius+.035) for i in range(3))
                incoming = tuple(-x for x in normal)
                result = DoorCrossing(source["identity"], target["identity"], result_position,
                                      _rotate_between(velocity, incoming, dest_normal),
                                      _rotate_between(facing, incoming, dest_normal), now_ms+self.cooldown_ms)
            elif source.get("island") and is_walker:
                event = {"island": source["island"], "pair": source.get("pair"),
                         "source": source["identity"], "href": source.get("href")}
                if source.get("treadmill"):
                    key = source.get("coincident_with", source["identity"])
                    event["treadmill"] = {**source["treadmill"], "offset": self.treadmill_offsets.get(key, source["treadmill"]["offset"])}
                result = DoorCrossing(source["identity"], None, position, velocity, facing,
                                      now_ms+8*self.cooldown_ms, event)
            else:
                continue
            self.cooldowns[actor] = result.cooldown_until_ms
            return result
        return None

    def arrive(self, actor: str, pair: str, *, radius: float = .001, now_ms: float) -> DoorCrossing:
        """Complete an accepted island load at its paired OUT end, on foot."""
        targets = [o for o in self.openings if o.get("port_role") == "out" and o.get("pair") == pair and self._enabled(o)]
        if len(targets) != 1:
            raise ValueError("island arrival needs exactly one OUT end")
        target = targets[0]
        frame, normal, _ = self._frame(target)
        point = tuple(frame.center[i]+normal[i]*max(.09, radius+.035) for i in range(3))
        until = now_ms+self.cooldown_ms
        self.cooldowns[actor] = until
        return DoorCrossing("", target["identity"], point, (0, 0, 0), normal, until)
