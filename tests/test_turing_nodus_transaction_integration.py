from __future__ import annotations

import sys
from pathlib import Path

import pytest

from camera_software import (
    CameraExposureScheduler,
    CameraManagedTimeBridge,
    CameraSliceJob,
    ManagedProcessState,
    NODUS_OPTICAL_RECEPTION_TYPE_ID,
    NodusFifoQualificationBridge,
    OPTICAL_RECEPTION_TOKEN_SIZE,
    OpticalReceptionToken,
    OpticalReceptionTokenFlags,
)


NODUS_DLL = Path(r"C:\dev\Powershell\nodus\build\Release\nodus_runtime.dll")
TURING_ROOT = Path(r"C:\dev\Powershell\turing")


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


class _SolverState:
    def __init__(self):
        self.time = 0.0

    def copy_shallow(self):
        return self.time

    def restore(self, snapshot):
        self.time = float(snapshot)


def _payload() -> bytes:
    return OpticalReceptionToken(
        1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13,
        (
            OpticalReceptionTokenFlags.ACTIVE
            | OpticalReceptionTokenFlags.COHERENT
            | OpticalReceptionTokenFlags.CONTRIBUTION
        ),
    ).to_bytes()


@pytest.mark.skipif(
    not NODUS_DLL.exists() or not TURING_ROOT.exists(),
    reason="cross-repository Turing/Nodus runtime unavailable",
)
def test_camera_window_rolls_back_solver_and_nodus_then_reruns_to_commit():
    sys.path.insert(0, str(TURING_ROOT))
    try:
        from src.common import ManagedTimeRuntime
        from src.common.dt_system.dt_controller import Targets
        from src.common.dt_system.dt_scaler import Metrics
    finally:
        sys.path.remove(str(TURING_ROOT))

    frame = CameraExposureScheduler().begin_frame(_Scene())
    job = CameraSliceJob.from_slice(frame.slices[0])
    solver = _SolverState()
    drain_during_advance = False
    consumed = []

    with NodusFifoQualificationBridge(
        NODUS_DLL,
        element_size=OPTICAL_RECEPTION_TOKEN_SIZE,
        capacity=4,
        type_id=NODUS_OPTICAL_RECEPTION_TYPE_ID,
    ) as fifo:
        assert fifo.uses_extracted_runtime
        fifo.subscribe(101)
        fifo.subscribe(202)
        state = ManagedProcessState({"solver": solver, "nodus_fifo": fifo})

        def advance(managed_state, dt):
            managed_state["solver"].time += float(dt)
            assert fifo.publish(77, _payload())
            if drain_during_advance:
                consumed.append(fifo.consume(101))
                consumed.append(fifo.consume(202))
            return True, Metrics(0.0, 0.0, 0.0, 0.0)

        runtime = ManagedTimeRuntime(
            state,
            advance,
            dx=1.0,
            targets=Targets(1.0, 1.0, 1.0),
        )
        bridge = CameraManagedTimeBridge(runtime)

        with pytest.raises(RuntimeError, match="commit gate rejected"):
            bridge.advance(
                job,
                commit_ready=lambda _job: fifo.quiescent,
                dt_initial=job.exposure_slice.dt,
            )

        assert solver.time == 0.0
        assert runtime.current_time == 0.0
        assert fifo.unread(101) == 0
        assert fifo.unread(202) == 0

        drain_during_advance = True
        report = bridge.advance(
            job,
            commit_ready=lambda _job: fifo.quiescent,
            dt_initial=job.exposure_slice.dt,
        )

        assert report.exact_landing
        assert solver.time == pytest.approx(0.2)
        assert runtime.current_time == pytest.approx(0.2)
        assert consumed == [_payload(), _payload()]
        assert fifo.quiescent
