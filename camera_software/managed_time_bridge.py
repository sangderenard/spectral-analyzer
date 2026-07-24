"""Camera-owned adapter to Turing's managed scientific-time boundary.

This module contains no timestep controller.  It translates an immutable
``CameraSliceJob`` into Turing's real ``TimeWindowRequest`` and supplies the
camera/optical completion predicate through Turing's transactional commit
gate.  The import is lazy so the optical package remains usable in standalone
calibration hosts that do not install the larger Turing checkout.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional, Protocol

from .exposure_timing import (
    CameraSliceJob,
    CausalMultireaderQueue,
)


class TimeWindowRequestFactory(Protocol):
    def __call__(
        self,
        *,
        request_id: int,
        generation: int,
        t_start: float,
        t_end: float,
        dt_initial: float,
        event_times: tuple[float, ...],
        allow_increase_mid_window: bool,
    ) -> Any: ...


class ManagedTimeRuntimeProtocol(Protocol):
    generation: int
    current_time: float

    def advance(
        self,
        request: Any,
        *,
        commit_gate: Optional[Callable[[Any, Any], bool]] = None,
    ) -> Any: ...

    def supersede(
        self, generation: int, *, at_time: Optional[float] = None
    ) -> None: ...


SliceCommitReady = Callable[[CameraSliceJob], bool]


class ManagedProcessState:
    """Ordered transaction bundle for solver state and process-runtime state."""

    def __init__(self, participants: Mapping[str, Any]) -> None:
        if not participants:
            raise ValueError("managed process state requires participants")
        self._participants = dict(participants)
        for name, participant in self._participants.items():
            if not name:
                raise ValueError("managed state participant names must be non-empty")
            if not (
                hasattr(participant, "copy_shallow")
                and callable(participant.copy_shallow)
                and hasattr(participant, "restore")
                and callable(participant.restore)
            ):
                raise TypeError(
                    f"managed state participant {name!r} lacks checkpoint methods"
                )

    def __getitem__(self, name: str) -> Any:
        return self._participants[name]

    def copy_shallow(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (name, participant.copy_shallow())
            for name, participant in self._participants.items()
        )

    def restore(self, snapshot: tuple[tuple[str, Any], ...]) -> None:
        if tuple(name for name, _value in snapshot) != tuple(self._participants):
            raise RuntimeError("managed process checkpoint participant mismatch")
        failures: list[tuple[str, Exception]] = []
        for name, value in reversed(snapshot):
            try:
                self._participants[name].restore(value)
            except Exception as exc:  # restore every participant before failing
                failures.append((name, exc))
        if failures:
            names = ", ".join(name for name, _exc in failures)
            raise RuntimeError(
                f"managed process state restore failed for: {names}"
            ) from failures[0][1]


class NativeWaveStateParticipant:
    """Managed-state adapter for the native pipeline's mutable T4 arenas.

    This participant is intentionally wave-specific. It must be composed with
    separate participants for Nodus transport and any ray-queue, sensor/BDPT,
    or GPU-resident state mutated by the same managed window.
    """

    def __init__(self, tracer: Any) -> None:
        required = (
            "in_flight_count",
            "copy_wave_transaction_state",
            "restore_wave_transaction_state",
        )
        missing = tuple(
            name for name in required
            if not hasattr(tracer, name) or not callable(getattr(tracer, name))
        )
        if missing:
            raise TypeError(
                "native tracer lacks wave transaction methods: "
                + ", ".join(missing)
            )
        self.tracer = tracer

    @property
    def quiescent(self) -> bool:
        return int(self.tracer.in_flight_count()) == 0

    def copy_shallow(self) -> Any:
        if not self.quiescent:
            raise RuntimeError(
                "native wave checkpoint requires an idle optical pipeline"
            )
        return self.tracer.copy_wave_transaction_state()

    def restore(self, snapshot: Any) -> None:
        if not self.quiescent:
            raise RuntimeError(
                "native wave restore requires an idle optical pipeline"
            )
        self.tracer.restore_wave_transaction_state(snapshot)


class NativeQueueStateParticipant:
    """Managed-state adapter for CPU T1-T5 queues and queued products.

    It retracts records emitted by a rejected window but does not cover T4
    fields, sensor/BDPT accumulator vectors, or GPU-resident state.
    """

    def __init__(self, tracer: Any) -> None:
        required = (
            "in_flight_count",
            "copy_queue_transaction_state",
            "restore_queue_transaction_state",
        )
        missing = tuple(
            name for name in required
            if not hasattr(tracer, name) or not callable(getattr(tracer, name))
        )
        if missing:
            raise TypeError(
                "native tracer lacks queue transaction methods: "
                + ", ".join(missing)
            )
        self.tracer = tracer

    @property
    def quiescent(self) -> bool:
        return int(self.tracer.in_flight_count()) == 0

    def copy_shallow(self) -> Any:
        if not self.quiescent:
            raise RuntimeError(
                "native queue checkpoint requires an idle optical pipeline"
            )
        return self.tracer.copy_queue_transaction_state()

    def restore(self, snapshot: Any) -> None:
        if not self.quiescent:
            raise RuntimeError(
                "native queue restore requires an idle optical pipeline"
            )
        self.tracer.restore_queue_transaction_state(snapshot)


def load_turing_time_window_request() -> TimeWindowRequestFactory:
    """Resolve Turing's canonical request class without copying its contract."""

    try:
        from src.common import TimeWindowRequest
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Turing's src package is unavailable; install the Turing checkout "
            "or inject its TimeWindowRequest class"
        ) from exc
    return TimeWindowRequest


class CameraManagedTimeBridge:
    """Submit camera-authored physical windows to a managed Turing runtime."""

    def __init__(
        self,
        runtime: ManagedTimeRuntimeProtocol,
        *,
        request_factory: Optional[TimeWindowRequestFactory] = None,
    ) -> None:
        if not hasattr(runtime, "advance") or not callable(runtime.advance):
            raise TypeError("managed-time runtime must provide advance()")
        self.runtime = runtime
        self._request_factory = request_factory

    @property
    def request_factory(self) -> TimeWindowRequestFactory:
        if self._request_factory is None:
            self._request_factory = load_turing_time_window_request()
        return self._request_factory

    def supersede_program(self, generation: int, *, at_time: float) -> None:
        """Explicitly align Turing with a newly authored camera program."""

        self.runtime.supersede(int(generation), at_time=float(at_time))

    def make_request(
        self,
        job: CameraSliceJob,
        *,
        dt_initial: Optional[float] = None,
        event_times: tuple[float, ...] = (),
        allow_increase_mid_window: bool = False,
    ) -> Any:
        exposure_slice = job.exposure_slice
        duration = exposure_slice.dt
        proposed_dt = duration if dt_initial is None else float(dt_initial)
        if not math.isfinite(proposed_dt) or proposed_dt <= 0.0:
            raise ValueError("camera managed-time initial dt must be positive")
        ordered_events = tuple(float(value) for value in event_times)
        if tuple(sorted(set(ordered_events))) != ordered_events:
            raise ValueError("camera event times must be unique and ordered")
        if any(
            not exposure_slice.t0 < event < exposure_slice.t1
            for event in ordered_events
        ):
            raise ValueError("camera events must lie inside their slice")

        return self.request_factory(
            request_id=int(exposure_slice.causality_id),
            generation=int(exposure_slice.program_generation),
            t_start=float(exposure_slice.t0),
            t_end=float(exposure_slice.t1),
            dt_initial=min(proposed_dt, duration),
            event_times=ordered_events,
            allow_increase_mid_window=bool(allow_increase_mid_window),
        )

    def advance(
        self,
        job: CameraSliceJob,
        *,
        commit_ready: SliceCommitReady,
        dt_initial: Optional[float] = None,
        event_times: tuple[float, ...] = (),
        allow_increase_mid_window: bool = False,
    ) -> Any:
        """Advance and commit exactly one slice if its process gate is closed."""

        if not callable(commit_ready):
            raise TypeError("camera managed-time commit predicate is required")
        request = self.make_request(
            job,
            dt_initial=dt_initial,
            event_times=event_times,
            allow_increase_mid_window=allow_increase_mid_window,
        )
        report = self.runtime.advance(
            request,
            commit_gate=lambda _request, _result: bool(commit_ready(job)),
        )
        if (
            int(report.request_id) != int(job.key.causality_id)
            or int(report.generation) != int(job.key.program_generation)
        ):
            raise RuntimeError("Turing returned a report for another camera slice")
        if not bool(report.exact_landing):
            raise RuntimeError("Turing did not land exactly on the camera boundary")
        return report

    def advance_after_release(
        self,
        job: CameraSliceJob,
        gate: CausalMultireaderQueue[Any],
        **kwargs: Any,
    ) -> Any:
        """Commit a slice only after its registered readers release it."""

        return self.advance(
            job,
            commit_ready=lambda candidate: gate.is_retired(candidate.key),
            **kwargs,
        )


__all__ = [
    "TimeWindowRequestFactory",
    "ManagedTimeRuntimeProtocol",
    "SliceCommitReady",
    "ManagedProcessState",
    "NativeWaveStateParticipant",
    "NativeQueueStateParticipant",
    "load_turing_time_window_request",
    "CameraManagedTimeBridge",
]
