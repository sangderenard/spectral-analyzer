"""Versioned physics-actor scene history for the thick-lens camera pipeline.

SceneVersionCache keeps N immutable SceneVersion snapshots.  Each version
records one ActorState per subject mesh group (centroid derived from the live
tri_vertices array).  Between commits the cache derives per-group linear
velocity, which drives both the velocity-epsilon rebuild scheduler and the
per-vertex motion buffer consumed by velocity-sensitive shaders.

Per-vertex motion buffer layout
--------------------------------
build_vertex_motion_buffer() returns (N_tris * 3, 3) float32.  Each block of
3 rows corresponds to one triangle; rows within a block are the triangle's
three vertices in winding order.  This mirrors the verts8 flat vertex stream
so it can be uploaded as a companion buffer or texture for shader access.

Velocity is derived as:  v = v_linear + ω × (p − centroid)

ω (angular velocity) requires orientation history — it is carried as a zero
vector until explicit quaternion tracking is wired in.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ActorState:
    """Immutable snapshot of one physics group at a specific simulation time."""

    group_id: int
    t_physics: float
    centroid: np.ndarray   # (3,) float64, world-space centroid of the group
    tri_start: int         # first triangle index in bench.tri_vertices
    tri_count: int         # number of triangles belonging to this group

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActorState):
            return NotImplemented
        return (self.group_id == other.group_id
                and self.t_physics == other.t_physics
                and np.array_equal(self.centroid, other.centroid))

    def __hash__(self) -> int:
        return hash((self.group_id, self.t_physics,
                     tuple(float(x) for x in self.centroid)))


@dataclass(frozen=True)
class SceneVersion:
    """Immutable snapshot of all tracked physics actors at one simulation time."""

    version_id: int
    t_physics: float
    actor_states: Dict[int, ActorState]   # group_id → ActorState
    label: str = ""


class PhysicsActor:
    """Tracks velocity and predicts the next rebuild time for one mesh group.

    Velocity is derived from successive ActorState centroids.  The rebuild
    predicate checks whether the predicted displacement since the last rebuild
    call has exceeded the position (or angle) epsilon.
    """

    def __init__(self,
                 group_id: int,
                 position_epsilon: float = 0.001,
                 angle_epsilon: float = 0.01) -> None:
        self.group_id = int(group_id)
        self.position_epsilon = float(position_epsilon)
        self.angle_epsilon = float(angle_epsilon)
        self._prev: Optional[ActorState] = None
        self._curr: Optional[ActorState] = None
        self.linear_velocity: np.ndarray = np.zeros(3, dtype=np.float64)
        self.angular_velocity: np.ndarray = np.zeros(3, dtype=np.float64)
        self._last_rebuild_t: float = -math.inf

    def update(self, state: ActorState) -> None:
        """Push a new ActorState; derives linear velocity from centroid delta."""
        self._prev = self._curr
        self._curr = state
        if self._prev is not None:
            dt = float(self._curr.t_physics) - float(self._prev.t_physics)
            if dt > 1e-9:
                self.linear_velocity = (
                    (self._curr.centroid - self._prev.centroid) / dt
                )

    def note_rebuild(self, at_t: float) -> None:
        """Record that a scene rebuild occurred at this sim-time."""
        self._last_rebuild_t = float(at_t)

    def time_until_rebuild(self, from_t: float) -> float:  # noqa: ARG002
        """Predicted seconds from now until any epsilon is exceeded."""
        speed = float(np.linalg.norm(self.linear_velocity))
        omega = float(np.linalg.norm(self.angular_velocity))
        t_pos = self.position_epsilon / speed if speed > 1e-9 else math.inf
        t_ang = self.angle_epsilon / omega if omega > 1e-9 else math.inf
        return min(t_pos, t_ang)

    def needs_rebuild(self, at_t: float) -> bool:
        """True when predicted displacement since last rebuild exceeds epsilon."""
        dt_since = float(at_t) - self._last_rebuild_t
        if dt_since <= 0.0:
            return False
        speed = float(np.linalg.norm(self.linear_velocity))
        omega = float(np.linalg.norm(self.angular_velocity))
        return (speed * dt_since >= self.position_epsilon
                or omega * dt_since >= self.angle_epsilon)

    def vertex_velocity(self, positions: np.ndarray) -> np.ndarray:
        """Per-vertex velocity for an (N, 3) float64 position array.

        Returns (N, 3) float32.  v = v_linear + ω × (p − centroid).
        """
        out = np.broadcast_to(self.linear_velocity, positions.shape).copy()
        omega_mag = float(np.linalg.norm(self.angular_velocity))
        if omega_mag > 1e-12 and self._curr is not None:
            dp = positions.astype(np.float64) - self._curr.centroid
            out = out + np.cross(self.angular_velocity, dp)
        return out.astype(np.float32)


class SceneVersionCache:
    """Bounded versioned history of immutable SceneVersion snapshots.

    Usage pattern
    -------------
    1. After each BVH rebuild:   cache.commit_version(t, group_tri_map, tri_vertices)
    2. After a rebuild completes: cache.note_rebuild(t)
    3. Per-frame gate check:      cache.any_actor_needs_rebuild(t_now)
    4. Before uploading to GPU:   buf = cache.build_vertex_motion_buffer(tri_vertices, group_tri_map)

    group_tri_map is {group_id: (tri_start, tri_count)} in bench.tri_vertices space.
    """

    def __init__(self,
                 max_versions: int = 8,
                 position_epsilon: float = 0.001,
                 angle_epsilon: float = 0.01) -> None:
        self.max_versions = int(max(2, max_versions))
        self.position_epsilon = float(position_epsilon)
        self.angle_epsilon = float(angle_epsilon)
        self._versions: deque[SceneVersion] = deque()
        self._actors: Dict[int, PhysicsActor] = {}
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _actor(self, group_id: int) -> PhysicsActor:
        if group_id not in self._actors:
            self._actors[group_id] = PhysicsActor(
                group_id,
                position_epsilon=self.position_epsilon,
                angle_epsilon=self.angle_epsilon,
            )
        return self._actors[group_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def commit_version(self,
                       t_physics: float,
                       group_tri_map: Dict[int, Tuple[int, int]],
                       tri_vertices: np.ndarray,
                       label: str = "") -> SceneVersion:
        """Snapshot current bench geometry into a new SceneVersion.

        Parameters
        ----------
        t_physics      Simulation time of the current scene geometry.
        group_tri_map  {group_id: (tri_start, tri_count)} in bench tri space.
        tri_vertices   bench.tri_vertices — (N_tris, 3, 3) float64.
        label          Optional human-readable tag (e.g. "rebuild:subject").
        """
        actor_states: Dict[int, ActorState] = {}
        for gid, (tri_start, tri_count) in group_tri_map.items():
            if tri_count <= 0:
                continue
            verts = tri_vertices[tri_start:tri_start + tri_count].reshape(-1, 3)
            centroid = np.mean(verts, axis=0, dtype=np.float64)
            state = ActorState(
                group_id=int(gid),
                t_physics=float(t_physics),
                centroid=centroid,
                tri_start=int(tri_start),
                tri_count=int(tri_count),
            )
            actor_states[int(gid)] = state
            self._actor(int(gid)).update(state)

        version = SceneVersion(
            version_id=self._counter,
            t_physics=float(t_physics),
            actor_states=actor_states,
            label=str(label),
        )
        self._counter += 1
        self._versions.append(version)
        while len(self._versions) > self.max_versions:
            self._versions.popleft()
        return version

    def version_for_time(self, t: float) -> Optional[SceneVersion]:
        """Return the version whose t_physics is closest to t."""
        if not self._versions:
            return None
        return min(self._versions, key=lambda v: abs(v.t_physics - t))

    def next_rebuild_time(self, from_t: float) -> float:
        """Earliest predicted sim-time at which any actor exceeds its epsilon."""
        if not self._actors:
            return math.inf
        return min(
            (float(from_t) + actor.time_until_rebuild(float(from_t))
             for actor in self._actors.values()),
            default=math.inf,
        )

    def any_actor_needs_rebuild(self, at_t: float) -> bool:
        """True when any actor's velocity-epsilon predicts stale geometry."""
        return any(a.needs_rebuild(float(at_t)) for a in self._actors.values())

    def note_rebuild(self, at_t: float) -> None:
        """Reset all actor rebuild clocks after a scene rebuild."""
        for actor in self._actors.values():
            actor.note_rebuild(float(at_t))

    def build_vertex_motion_buffer(self,
                                   tri_vertices: np.ndarray,
                                   group_tri_map: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """Build a per-vertex motion buffer for velocity-sensitive shaders.

        Returns (N_tris * 3, 3) float32 — velocity (vx, vy, vz) per vertex.
        Layout mirrors the verts8 flat stream: every 3 rows = one triangle.
        Vertices not belonging to any tracked actor receive zero velocity.
        """
        n_tris = int(tri_vertices.shape[0])
        buf = np.zeros((n_tris * 3, 3), dtype=np.float32)
        for gid, (tri_start, tri_count) in group_tri_map.items():
            if tri_count <= 0:
                continue
            actor = self._actors.get(int(gid))
            if actor is None:
                continue
            verts = tri_vertices[tri_start:tri_start + tri_count].reshape(-1, 3)
            v_start = tri_start * 3
            v_end = v_start + tri_count * 3
            buf[v_start:v_end] = actor.vertex_velocity(verts)
        return buf
