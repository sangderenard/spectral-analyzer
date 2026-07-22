import pytest

from camera_software.render_eta import RenderEtaTracker, format_duration


RUNTIME = {
    "epoch_bundle_count": 16,
    "max_sensor_epochs": 1,
    "sensor_top_k": 200,
    "sensor_samples_per_node": 83_887,
    "max_bounces": 32,
}


def test_eta_tracker_exposes_hidden_t5_pages_as_whole_render_progress():
    tracker = RenderEtaTracker(RUNTIME, started_at=100.0)
    tracker.begin_bundle(1)

    snapshot, completed = tracker.observe(
        "[gpu-bounce] batch#192 intents=167774 n_hits=167774"
    )
    assert snapshot is not None
    assert snapshot.pages_per_bundle == 100
    assert completed is False

    snapshot, completed = tracker.observe(
        "[T5-gpu-native] PASS SAMPLED-ESTIMATE t5_fired->18"
    )
    assert completed is True
    assert snapshot is not None
    assert snapshot.total_units == 1600
    assert snapshot.completed_units == 18
    assert snapshot.percent == pytest.approx(1.125)


def test_eta_tracker_counts_fractional_connect_progress_inside_current_page():
    tracker = RenderEtaTracker(RUNTIME, started_at=100.0)
    tracker.observe("[gpu-bounce] intents=167774")
    tracker.observe("PASS SAMPLED-ESTIMATE t5_fired->18")
    snapshot, _ = tracker.observe(
        "connect progress: active_pairs=33554432/67108864 50.000%"
    )

    assert snapshot is not None
    assert snapshot.completed_units == pytest.approx(18.5)


def test_duration_format_is_compact_for_terminal_and_ui():
    assert format_duration(None) == "--"
    assert format_duration(65) == "1m 05s"
    assert format_duration(38_700) == "10h 45m"
