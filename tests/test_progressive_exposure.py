import os

import numpy as np

from camera_software.progressive_exposure import (
    ExposureProgressBroker,
    ExposureProgressEvent,
    ExposureProgressKind,
    LinearProgressArtifactWriter,
    SensorRegion,
)
from camera_software.refinement_scheduler import RecursiveSensorWorkScheduler


def _event(sequence=1, **kwargs):
    return ExposureProgressEvent(
        exposure_id=kwargs.pop("exposure_id", "camera-a"),
        sequence=sequence,
        kind=kwargs.pop("kind", ExposureProgressKind.PASS_AVAILABLE),
        region=kwargs.pop("region", SensorRegion(10, 20, 64, 48)),
        **kwargs,
    )


def test_progress_event_json_line_round_trip_preserves_region_and_layer_identity():
    event = _event(
        pass_index=7,
        zoom_level=2,
        subdivision_level=3,
        completed_work=7,
        total_work=12,
        sensor_node_id=19,
        parent_sensor_node_id=2,
        global_uv_bounds=(0.25, 0.5, 0.5, 0.75),
    )
    decoded = ExposureProgressEvent.from_line(event.to_line())
    assert decoded == event
    assert ExposureProgressEvent.from_line("ordinary renderer log") is None


def test_broker_rejects_reordered_events_and_resets_layers_for_new_exposure():
    broker = ExposureProgressBroker()
    broker.publish(_event(sequence=1, zoom_level=0, subdivision_level=0))
    broker.publish(_event(sequence=2, zoom_level=1, subdivision_level=0))
    _, layers = broker.snapshot()
    assert set(layers) == {(0, 0, None), (1, 0, None)}

    try:
        broker.publish(_event(sequence=2))
    except ValueError as exc:
        assert "increase" in str(exc)
    else:
        raise AssertionError("reordered progress event must be rejected")

    broker.publish(_event(sequence=1, exposure_id="camera-b"))
    latest, layers = broker.snapshot()
    assert latest.exposure_id == "camera-b"
    assert set(layers) == {(0, 0, None)}


def test_broker_retains_multiple_sparse_nodes_at_the_same_mip_level():
    broker = ExposureProgressBroker()
    broker.publish(_event(sequence=1, subdivision_level=2, sensor_node_id=10))
    broker.publish(_event(sequence=2, subdivision_level=2, sensor_node_id=11))
    _, layers = broker.snapshot()
    assert set(layers) == {(0, 2, 10), (0, 2, 11)}

    broker.reset()
    latest, layers = broker.snapshot()
    assert latest is None
    assert layers == {}


def test_linear_artifact_is_announced_only_after_atomic_float_array_write(tmp_path):
    lines = []
    writer = LinearProgressArtifactWriter(str(tmp_path), line_sink=lines.append)
    raw = np.arange(7 * 9 * 3, dtype=np.float32).reshape(7, 9, 3)
    announced = writer.publish(_event(pass_index=1), raw)

    assert os.path.isfile(announced.linear_accumulation_path)
    assert not os.path.exists(announced.linear_accumulation_path + ".tmp")
    assert np.array_equal(np.load(announced.linear_accumulation_path), raw)
    assert ExposureProgressEvent.from_line(lines[0]) == announced


def test_epoch_artifact_can_announce_matching_per_bin_sample_counts(tmp_path):
    lines = []
    writer = LinearProgressArtifactWriter(str(tmp_path), line_sink=lines.append)
    raw = np.ones((5, 7, 3), dtype=np.float32)
    counts = np.arange(35, dtype=np.uint32).reshape(5, 7)
    announced = writer.publish(_event(pass_index=2), raw, counts)

    assert np.array_equal(np.load(announced.sample_count_path), counts)
    assert ExposureProgressEvent.from_line(lines[0]) == announced


def test_layer_artifact_announces_raw_nn_priority_map(tmp_path):
    lines = []
    writer = LinearProgressArtifactWriter(str(tmp_path), line_sink=lines.append)
    raw = np.ones((5, 7, 3), dtype=np.float32)
    priority = np.linspace(0.0, 1.0, 35, dtype=np.float32).reshape(5, 7)
    announced = writer.publish(_event(pass_index=2), raw, priority_map=priority)

    assert np.array_equal(np.load(announced.priority_map_path), priority)
    assert ExposureProgressEvent.from_line(lines[0]) == announced


def test_epoch_artifact_rejects_misaligned_sample_counts(tmp_path):
    writer = LinearProgressArtifactWriter(str(tmp_path), line_sink=lambda line: None)
    try:
        writer.publish(
            _event(pass_index=3),
            np.ones((5, 7, 3), dtype=np.float32),
            np.ones((4, 7), dtype=np.uint32),
        )
    except ValueError as exc:
        assert "matching" in str(exc)
    else:
        raise AssertionError("misaligned sample counts must be rejected")


def test_open_ended_writer_can_retain_only_recent_live_layers(tmp_path):
    writer = LinearProgressArtifactWriter(
        str(tmp_path), line_sink=lambda line: None, retain_last=2
    )
    published = []
    for index in range(1, 5):
        published.append(writer.publish(
            _event(
                sequence=index,
                kind=ExposureProgressKind.LAYER_AVAILABLE,
                pass_index=index,
                completed_work=index,
            ),
            np.ones((5, 7, 3), dtype=np.float32),
        ))
    assert not os.path.exists(published[0].linear_accumulation_path)
    assert not os.path.exists(published[1].linear_accumulation_path)
    assert os.path.exists(published[2].linear_accumulation_path)
    assert os.path.exists(published[3].linear_accumulation_path)


def test_scheduler_always_enqueues_nine_children_after_completed_work():
    scheduler = RecursiveSensorWorkScheduler(3, maximum_depth=2)
    root = scheduler.next()
    children = scheduler.complete(root)
    assert len(children) == 9
    assert len(scheduler) == 9
    assert all(work.subdivision_level == 1 for work in children)


def test_top_k_changes_order_without_deleting_unselected_frontier_nodes():
    scheduler = RecursiveSensorWorkScheduler(3, maximum_depth=1)
    children = scheduler.complete(scheduler.next())
    scheduler.set_priorities({children[7].node_id: 10.0, children[2].node_id: 5.0})
    chosen = scheduler.top_k(2)
    assert [work.node_id for work in chosen] == [children[7].node_id, children[2].node_id]
    assert len(scheduler) == 7


def test_finest_node_is_requeued_for_an_independent_sample_epoch():
    scheduler = RecursiveSensorWorkScheduler(3, maximum_depth=0)
    first = scheduler.next()
    repeated = scheduler.complete(first)
    assert len(repeated) == 1
    assert repeated[0].node_id == first.node_id
    assert repeated[0].sample_index == 1
