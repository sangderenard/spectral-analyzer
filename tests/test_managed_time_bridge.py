from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from camera_software.exposure_timing import (
    CameraExposureScheduler,
    CameraSliceJob,
    CameraSliceResult,
    CausalMultireaderQueue,
)
from camera_software.managed_time_bridge import (
    CameraManagedTimeBridge,
    ManagedProcessState,
    NativeQueueStateParticipant,
    NativeWaveStateParticipant,
)


class _Burst:
    stages = 1
    exposure_time_s = 0.2
    duty_cycle = 1.0
    energy_scale = 1.0
    enabled = True
    profile = "steady"


class _Plate:
    shutter_mode = "open"
    shutter_open = 1.0
    shutter_center_u = 0.5
    shutter_center_v = 0.5
    shutter_softness = 0.0


class _Scene:
    camera_light_burst = _Burst()
    image_plate = _Plate()


@dataclass(frozen=True)
class _Request:
    request_id: int
    generation: int
    t_start: float
    t_end: float
    dt_initial: float
    event_times: tuple[float, ...]
    allow_increase_mid_window: bool


class _Runtime:
    def __init__(self) -> None:
        self.generation = 0
        self.current_time = 0.0
        self.requests = []

    def advance(self, request, *, commit_gate=None):
        if request.generation != self.generation:
            raise RuntimeError("stale managed-time generation")
        if request.t_start != self.current_time:
            raise RuntimeError("managed-time request is discontinuous")
        self.requests.append(request)
        result = SimpleNamespace(advanced=request.t_end - request.t_start)
        if commit_gate is not None and not commit_gate(request, result):
            raise RuntimeError("managed-time commit gate rejected window")
        self.current_time = request.t_end
        return SimpleNamespace(
            request_id=request.request_id,
            generation=request.generation,
            exact_landing=True,
            result=result,
        )

    def supersede(self, generation, *, at_time=None):
        self.generation = int(generation)
        if at_time is not None:
            self.current_time = float(at_time)


def _job(*, generation=0):
    frame = CameraExposureScheduler(
        program_generation=generation
    ).begin_frame(_Scene())
    return frame, CameraSliceJob.from_slice(frame.slices[0])


def test_bridge_maps_camera_identity_time_and_events_without_local_dt_logic():
    _frame, job = _job()
    runtime = _Runtime()
    bridge = CameraManagedTimeBridge(runtime, request_factory=_Request)

    report = bridge.advance(
        job,
        dt_initial=0.08,
        event_times=(0.05, 0.15),
        commit_ready=lambda _job: True,
    )

    request = runtime.requests[0]
    assert request.request_id == job.key.causality_id
    assert request.generation == job.key.program_generation
    assert request.t_start == pytest.approx(0.0)
    assert request.t_end == pytest.approx(0.2)
    assert request.dt_initial == pytest.approx(0.08)
    assert request.event_times == pytest.approx((0.05, 0.15))
    assert report.exact_landing


def test_bridge_refuses_camera_commit_until_multireader_prefix_retires():
    frame, job = _job()
    runtime = _Runtime()
    bridge = CameraManagedTimeBridge(runtime, request_factory=_Request)
    gate = CausalMultireaderQueue(frame)
    gate.register("wave")
    gate.register("detector")
    result = CameraSliceResult(job.key, "result-digest", None)
    gate.publish(result)

    with pytest.raises(RuntimeError, match="commit gate rejected"):
        bridge.advance_after_release(job, gate)
    assert runtime.current_time == 0.0
    assert not gate.is_retired(job.key)

    gate.acknowledge("wave", job.key)
    assert not gate.is_retired(job.key)
    gate.acknowledge("detector", job.key)
    assert gate.is_retired(job.key)

    report = bridge.advance_after_release(job, gate)
    assert report.exact_landing
    assert runtime.current_time == pytest.approx(0.2)


def test_bridge_requires_explicit_program_supersession():
    frame, job = _job(generation=1)
    runtime = _Runtime()
    bridge = CameraManagedTimeBridge(runtime, request_factory=_Request)

    with pytest.raises(RuntimeError, match="stale"):
        bridge.advance(job, commit_ready=lambda _job: True)

    bridge.supersede_program(1, at_time=frame.t0)

    assert runtime.generation == 1
    assert runtime.current_time == frame.t0
    assert bridge.advance(job, commit_ready=lambda _job: True).exact_landing


def test_managed_process_state_restores_every_participant_in_reverse_order():
    events = []

    class Participant:
        def __init__(self, name):
            self.name = name
            self.value = 0

        def copy_shallow(self):
            events.append(("snapshot", self.name))
            return self.value

        def restore(self, value):
            events.append(("restore", self.name))
            self.value = value

    solver = Participant("solver")
    fifo = Participant("fifo")
    state = ManagedProcessState({"solver": solver, "fifo": fifo})
    checkpoint = state.copy_shallow()
    solver.value = 4
    fifo.value = 9

    state.restore(checkpoint)

    assert solver.value == 0
    assert fifo.value == 0
    assert events == [
        ("snapshot", "solver"),
        ("snapshot", "fifo"),
        ("restore", "fifo"),
        ("restore", "solver"),
    ]


def test_native_wave_participant_requires_idle_and_delegates_checkpoint():
    class Tracer:
        in_flight = 1
        state = "before"

        def in_flight_count(self):
            return self.in_flight

        def copy_wave_transaction_state(self):
            return self.state

        def restore_wave_transaction_state(self, checkpoint):
            self.state = checkpoint

    tracer = Tracer()
    participant = NativeWaveStateParticipant(tracer)
    assert not participant.quiescent
    with pytest.raises(RuntimeError, match="idle"):
        participant.copy_shallow()

    tracer.in_flight = 0
    checkpoint = participant.copy_shallow()
    tracer.state = "after"
    participant.restore(checkpoint)
    assert participant.quiescent
    assert tracer.state == "before"


def test_native_queue_participant_retracts_through_tracer_contract():
    class Tracer:
        records = ["prior"]

        def in_flight_count(self):
            return 0

        def copy_queue_transaction_state(self):
            return tuple(self.records)

        def restore_queue_transaction_state(self, checkpoint):
            self.records[:] = checkpoint

    tracer = Tracer()
    participant = NativeQueueStateParticipant(tracer)
    checkpoint = participant.copy_shallow()
    tracer.records.append("rejected-window")
    participant.restore(checkpoint)
    assert tracer.records == ["prior"]
