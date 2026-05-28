"""Non-committal physics backend interface for camera-driven time slicing.

The camera is the sole clock.  Before submitting rays for each exposure
slice, the camera calls PhysicsBackend.request_dt(dt_s) to declare that
simulated time is advancing by dt_s seconds.  The backend responds with
the updated scene state (positions, velocities, …) for that instant.

This stub does nothing and returns immediately.  Replace the body of
request_dt — or subclass PhysicsBackend — when real scene simulation is
wired in.  The interface is intentionally minimal so the contract can be
honoured by anything from a static scene to a full rigid-body solver.
"""
from __future__ import annotations


class PhysicsBackend:
    """Stub physics backend.  Camera calls request_dt once per exposure slice."""

    def request_dt(self, dt_s: float) -> None:
        """Advance scene state by *dt_s* seconds.

        Called by the camera thread before each slice's ray submission.
        The real implementation should update actor positions/velocities
        and signal the scene version cache so the bench can rebuild the
        BVH if displacement exceeds the configured epsilon.

        Args:
            dt_s: Duration of this exposure slice in simulated seconds.
        """
