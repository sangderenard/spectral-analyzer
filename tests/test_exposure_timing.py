from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import random
from types import SimpleNamespace

import pytest

from camera_software.exposure_timing import (
    CameraExposureScheduler,
    CameraResultCollector,
    CameraSliceJob,
    CameraSliceResult,
    CameraTimeline,
    CausalMultireaderQueue,
    ExposureBarrier,
)


def _scene(
    *,
    enabled: bool = True,
    stages: int = 4,
    duty_cycle: float = 1.0,
    energy_scale: float = 2.0,
    exposure_time_s: float = 0.02,
    shutter_mode: str = "open",
    shutter_open: float = 1.0,
):
    return SimpleNamespace(
        camera_light_burst=SimpleNamespace(
            enabled=enabled,
            stages=stages,
            duty_cycle=duty_cycle,
            energy_scale=energy_scale,
            exposure_time_s=exposure_time_s,
            profile="steady",
        ),
        image_plate=SimpleNamespace(
            shutter_mode=shutter_mode,
            shutter_open=shutter_open,
            shutter_center_u=0.5,
            shutter_center_v=0.5,
            shutter_softness=0.0,
        ),
    )


def test_scheduler_authors_distinct_flash_and_sensor_budgets():
    frame = CameraExposureScheduler().begin_frame(
        _scene(stages=4, duty_cycle=0.5, energy_scale=2.0),
        dt_s=99.0,
        t0_s=3.0,
    )

    assert frame.t0 == pytest.approx(3.0)
    assert frame.t1 == pytest.approx(3.02)
    assert [item.flash_active for item in frame.slices] == [
        True, True, False, False
    ]
    assert [item.flash_weight for item in frame.slices] == pytest.approx(
        [0.5, 0.5, 0.0, 0.0]
    )
    assert [item.sensor_weight for item in frame.slices] == pytest.approx(
        [0.25, 0.25, 0.25, 0.25]
    )
    assert sum(item.flash_weight for item in frame.slices) == pytest.approx(1.0)
    assert sum(item.sensor_weight for item in frame.slices) == pytest.approx(1.0)

    timeline = CameraTimeline.from_exposure_frame(frame)
    assert timeline.flash_slices == (0, 1)
    assert timeline.sensor_slices == (0, 1, 2, 3)
    assert timeline.prerequisite_flash_slice_for_sensor == {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
    }


@pytest.mark.parametrize(
    ("enabled", "duty_cycle"),
    [(False, 1.0), (True, 0.0)],
)
def test_no_flash_program_still_integrates_sensor(
    enabled: bool, duty_cycle: float
):
    frame = CameraExposureScheduler().begin_frame(
        _scene(enabled=enabled, duty_cycle=duty_cycle)
    )
    timeline = CameraTimeline.from_exposure_frame(frame)

    assert timeline.flash_slices == ()
    assert all(not item.flash_active for item in timeline.slices)
    assert all(item.flash_weight == 0.0 for item in frame.slices)
    assert sum(item.sensor_weight for item in frame.slices) == pytest.approx(1.0)
    assert timeline.sensor_slices == (0, 1, 2, 3)
    assert all(
        prerequisite == -1
        for prerequisite in timeline.prerequisite_flash_slice_for_sensor.values()
    )
    barrier = ExposureBarrier(timeline)
    assert all(barrier.sensor_may_submit(i) for i in timeline.sensor_slices)


def test_closed_shutter_authors_no_sensor_integration():
    frame = CameraExposureScheduler().begin_frame(
        _scene(shutter_mode="closed", shutter_open=0.0)
    )
    timeline = CameraTimeline.from_exposure_frame(frame)

    assert timeline.sensor_slices == ()
    assert all(not item.sensor_integrating for item in timeline.slices)
    assert all(item.sensor_weight == 0.0 for item in frame.slices)


def test_barrier_does_not_treat_out_of_order_completion_as_a_prefix():
    frame = CameraExposureScheduler().begin_frame(_scene(stages=3))
    timeline = CameraTimeline.from_exposure_frame(frame)
    barrier = ExposureBarrier(timeline)

    barrier.record_flash_dispatched(2)
    barrier.confirm_flash_materialized(2)
    assert barrier.flash_submitted_through == -1
    assert not barrier.sensor_may_submit(0)
    assert barrier.sensor_may_submit(2)
    assert not barrier.all_flash_dispatched

    for slice_id in (0, 1):
        barrier.record_flash_dispatched(slice_id)
        barrier.confirm_flash_materialized(slice_id)
    assert barrier.flash_submitted_through == 2
    assert barrier.all_flash_dispatched

    barrier.record_sensor_submitted(2)
    assert not barrier.all_sensor_submitted
    barrier.record_sensor_submitted(0)
    barrier.record_sensor_submitted(1)
    assert barrier.all_sensor_submitted
    for slice_id in timeline.sensor_slices:
        barrier.confirm_reception_closed(slice_id)
        barrier.confirm_detector_integration_committed(slice_id)
    assert barrier.exposure_complete


def test_exposure_frame_slice_program_is_immutable():
    frame = CameraExposureScheduler().begin_frame(_scene(stages=3))
    assert isinstance(frame.slices, tuple)
    with pytest.raises(TypeError):
        frame.slices[0] = frame.slices[1]


def test_estimator_jobs_may_dispatch_concurrently_but_commit_is_causal():
    frame = CameraExposureScheduler().begin_frame(_scene(stages=1))
    barrier = ExposureBarrier(CameraTimeline.from_exposure_frame(frame))

    # Sensor-side rays are importance probes, not physical sensor emission.
    barrier.record_sensor_probe_work_dispatched(0, submitted=128)
    barrier.record_emission_work_dispatched(0, submitted=64)
    assert not barrier.exposure_complete

    with pytest.raises(RuntimeError, match="before transport"):
        barrier.confirm_reception_closed(0)
    with pytest.raises(RuntimeError, match="before reception"):
        barrier.confirm_detector_integration_committed(0)

    barrier.confirm_transport_products_materialized(0)
    assert barrier.reception_may_close(0)
    barrier.confirm_reception_closed(0)
    assert not barrier.exposure_complete
    barrier.confirm_detector_integration_committed(0)

    state = barrier.state(0)
    assert state.emission_work_dispatched
    assert state.sensor_probe_work_dispatched
    assert state.transport_products_materialized
    assert state.reception_closed
    assert state.detector_integration_committed
    assert barrier.exposure_complete


def test_no_flash_slice_still_accepts_natural_emission_work():
    frame = CameraExposureScheduler().begin_frame(
        _scene(enabled=False, stages=1)
    )
    timeline = CameraTimeline.from_exposure_frame(frame)
    barrier = ExposureBarrier(timeline)

    assert timeline.flash_slices == ()
    barrier.record_emission_work_dispatched(0, submitted=32)
    barrier.record_sensor_probe_work_dispatched(0, submitted=64)
    barrier.confirm_transport_products_materialized(0)
    barrier.confirm_reception_closed(0)
    barrier.confirm_detector_integration_committed(0)
    assert barrier.exposure_complete


def test_camera_authors_portable_monotonic_causality_keys():
    scheduler = CameraExposureScheduler(
        camera_id="thick-lens-lab/main", program_generation=7
    )
    first = scheduler.begin_frame(_scene(stages=3))
    scheduler.finish_frame()
    second = scheduler.begin_frame(_scene(stages=2))

    assert [item.causality_id for item in first.slices] == [0, 1, 2]
    assert [item.causality_id for item in second.slices] == [3, 4]
    assert all(
        item.causality_key.camera_id == "thick-lens-lab/main"
        and item.causality_key.program_generation == 7
        for item in first.slices + second.slices
    )

    scheduler.reset()
    restarted = scheduler.begin_frame(_scene(stages=1))
    assert restarted.slices[0].program_generation == 8
    assert restarted.slices[0].causality_id == 5


def test_slice_jobs_are_self_identifying_and_order_independent():
    frame = CameraExposureScheduler(camera_id="network-camera").begin_frame(
        _scene(stages=8)
    )
    jobs = [CameraSliceJob.from_slice(item) for item in frame.slices]
    random.Random(121).shuffle(jobs)
    committed: list[int] = []
    collector: CameraResultCollector[int] = CameraResultCollector(
        frame, on_commit=lambda item: committed.append(item.payload)
    )

    def finish(job: CameraSliceJob) -> tuple[int, ...]:
        value = job.exposure_slice.slice_id
        ready = collector.submit(
            CameraSliceResult(
                key=job.key,
                payload_digest=f"sha256:slice-{value}",
                payload=value,
                worker_id=f"worker-{value % 3}",
            )
        )
        return tuple(item.payload for item in ready)

    with ThreadPoolExecutor(max_workers=4) as pool:
        released = list(pool.map(finish, jobs))

    assert sorted(value for batch in released for value in batch) == list(range(8))
    assert committed == list(range(8))
    assert collector.complete
    assert collector.pending_count == 0


def test_slice_job_wire_descriptor_round_trips_and_is_content_identified():
    frame = CameraExposureScheduler(camera_id="network/camera").begin_frame(
        _scene(stages=1)
    )
    job = CameraSliceJob.from_slice(
        frame.slices[0], solver_revision="maxwell-solver:abc123"
    )
    restored = CameraSliceJob.from_wire(job.to_wire())

    assert restored == job
    assert restored.key == job.key
    assert restored.descriptor_digest == job.descriptor_digest

    changed = CameraSliceJob.from_slice(
        frame.slices[0], solver_revision="maxwell-solver:different"
    )
    assert changed.descriptor_digest != job.descriptor_digest

    collector: CameraResultCollector[str] = CameraResultCollector([job])
    with pytest.raises(ValueError, match="authored slice job"):
        collector.submit(
            CameraSliceResult(
                job.key,
                "payload-digest",
                "payload",
                job_digest=changed.descriptor_digest,
            )
        )
    accepted = CameraSliceResult(
        job.key,
        "payload-digest",
        "payload",
        job_digest=job.descriptor_digest,
    )
    assert collector.submit(accepted) == (accepted,)


def test_result_collector_deduplicates_retries_and_rejects_conflicts():
    frame = CameraExposureScheduler().begin_frame(_scene(stages=2))
    collector: CameraResultCollector[str] = CameraResultCollector(frame)
    key = frame.slices[0].causality_key
    original = CameraSliceResult(key, "digest-a", "first", attempt_id="a")
    retry = CameraSliceResult(key, "digest-a", "first", attempt_id="b")
    conflict = CameraSliceResult(key, "digest-b", "different")

    assert collector.submit(original) == (original,)
    assert collector.submit(retry) == ()
    with pytest.raises(RuntimeError, match="conflicting results"):
        collector.submit(conflict)


def test_ordered_single_worker_path_releases_each_result_immediately():
    frame = CameraExposureScheduler().begin_frame(_scene(stages=4))
    collector: CameraResultCollector[int] = CameraResultCollector(frame)

    for item in frame.slices:
        result = CameraSliceResult(
            item.causality_key,
            f"digest-{item.slice_id}",
            item.slice_id,
        )
        assert collector.submit(result) == (result,)
        assert collector.pending_count == 0
    assert collector.complete


def test_result_collector_rejects_stale_program_generation():
    scheduler = CameraExposureScheduler(camera_id="camera-a")
    old_frame = scheduler.begin_frame(_scene(stages=1))
    scheduler.reset()
    new_frame = scheduler.begin_frame(_scene(stages=1))
    collector: CameraResultCollector[None] = CameraResultCollector(new_frame)

    with pytest.raises(ValueError, match="does not belong"):
        collector.submit(
            CameraSliceResult(
                old_frame.slices[0].causality_key,
                "old-result",
                None,
            )
        )


def test_multireader_holds_gate_advance_until_every_system_acknowledges():
    frame = CameraExposureScheduler().begin_frame(_scene(stages=3))
    retired: list[int] = []
    gate: CausalMultireaderQueue[int] = CausalMultireaderQueue(
        frame, on_retire=lambda result: retired.append(result.payload)
    )
    gate.register("wave-solver")
    gate.register("detector")
    collector: CameraResultCollector[int] = CameraResultCollector(
        frame, on_commit=gate.publish
    )

    results = [
        CameraSliceResult(
            item.causality_key,
            f"digest-{item.slice_id}",
            item.slice_id,
        )
        for item in frame.slices
    ]
    for result in reversed(results):
        collector.submit(result)

    assert [r.payload for r in gate.pending_for("wave-solver")] == [0, 1, 2]
    assert [r.payload for r in gate.pending_for("detector")] == [0, 1, 2]

    # Readers and slices may finish in any order. Slice 1 still cannot retire
    # across the unacknowledged slice-0 causal boundary.
    gate.acknowledge("wave-solver", results[1].key)
    gate.acknowledge("detector", results[1].key)
    gate.acknowledge("wave-solver", results[0].key)
    assert retired == []
    released = gate.acknowledge("detector", results[0].key)
    assert [item.payload for item in released] == [0, 1]
    assert retired == [0, 1]

    gate.acknowledge("detector", results[2].key)
    assert not gate.advance_complete
    gate.acknowledge("wave-solver", results[2].key)
    assert gate.advance_complete
    assert retired == [0, 1, 2]


def test_multireader_disconnect_cannot_silently_drop_a_causal_hold():
    frame = CameraExposureScheduler().begin_frame(_scene(stages=1))
    gate: CausalMultireaderQueue[None] = CausalMultireaderQueue(frame)
    gate.register("remote-detector")
    result = CameraSliceResult(
        frame.slices[0].causality_key, "digest", None
    )
    gate.publish(result)

    with pytest.raises(RuntimeError, match="still holds"):
        gate.unregister("remote-detector")
    assert gate.retired_count == 0

    gate.unregister("remote-detector", release_holds=True)
    assert gate.advance_complete
