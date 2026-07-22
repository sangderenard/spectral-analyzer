import json

import numpy as np

from camera_software.render_work_scheduler import (
    NeedWeightedPacketScheduler,
    RenderPacketJob,
)
from render_scene_packets import (
    _display_raster_to_native_square,
    _load_or_create_scheduler,
)


def test_equal_need_jobs_advance_round_robin_without_starvation():
    scheduler = NeedWeightedPacketScheduler(
        RenderPacketJob(name, 100) for name in ("a", "b", "c")
    )
    selected = [scheduler.next_job().job_id for _ in range(9)]

    assert selected == ["a", "b", "c", "a", "b", "c", "a", "b", "c"]


def test_more_valuable_job_receives_more_packets_but_not_exclusive_ownership():
    scheduler = NeedWeightedPacketScheduler([
        RenderPacketJob("ordinary", 100, weight=1.0),
        RenderPacketJob("needed", 100, weight=3.0),
    ])
    selected = [scheduler.next_job().job_id for _ in range(40)]

    assert selected.count("needed") > selected.count("ordinary")
    assert selected.count("ordinary") >= 8


def test_underexposed_sensor_evidence_raises_scene_need(tmp_path):
    full = tmp_path / "full.npy"
    empty = tmp_path / "empty.npy"
    priority = tmp_path / "priority.npy"
    np.save(full, np.ones((8, 8), np.float32))
    np.save(empty, np.zeros((8, 8), np.float32))
    np.save(priority, np.ones((8, 8), np.float32))
    developed = RenderPacketJob(
        "developed", 10, completed_packets=2,
        exposure_weight_path=str(full), priority_map_path=str(priority),
    )
    sparse = RenderPacketJob(
        "sparse", 10, completed_packets=2,
        exposure_weight_path=str(empty), priority_map_path=str(priority),
    )

    assert sparse.evidence_need() > developed.evidence_need()
    assert sparse.effective_need() > developed.effective_need()


def test_checkpoint_round_trip_preserves_fairness_state_and_artifacts(tmp_path):
    scheduler = NeedWeightedPacketScheduler([
        RenderPacketJob("a", 4),
        RenderPacketJob("b", 4),
    ])
    first = scheduler.next_job()
    scheduler.complete_packet(
        first.job_id,
        sensor_sum_path=str(tmp_path / "sum.npy"),
        exposure_weight_path=str(tmp_path / "weight.npy"),
    )
    path = tmp_path / "scheduler.json"
    scheduler.save(str(path))
    restored = NeedWeightedPacketScheduler.load(str(path))

    assert restored.to_payload() == scheduler.to_payload()
    assert json.loads(path.read_text())["schema_version"] == 1
    assert restored.next_job().job_id == "b"


def test_native_restore_mapping_matches_square_display_inverse():
    display = np.arange(5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
    native = _display_raster_to_native_square(display)

    assert native.shape == (7, 7, 3)
    assert native.flags.c_contiguous


def test_resume_rejects_changed_job_targets(tmp_path):
    jobs = [
        {"id": "a", "token": "A", "exposure": {"sensor_sweeps": 4}},
    ]
    scheduler = NeedWeightedPacketScheduler([RenderPacketJob("a", 4)])
    state = tmp_path / "state.json"
    scheduler.save(str(state))

    try:
        _load_or_create_scheduler(str(state), jobs, 5, resume=True)
    except ValueError as exc:
        assert "target changed" in str(exc)
    else:
        raise AssertionError("changed resume target must require --no-resume")
