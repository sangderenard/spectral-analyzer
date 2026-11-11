from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from camera_software.exposure_timing import CameraExposureScheduler
from camera_software.trigger_bus import (
    make_camera_trigger_graph, make_camera_trigger_panel, resolve_camera_trigger_bus,
)
from routing_engine import RoutingGraph


def frame(*, shutter="open", enabled=True, duration=0.02):
    scene = SimpleNamespace(
        camera_light_burst=SimpleNamespace(stages=4, exposure_time_s=duration,
            enabled=enabled, energy_scale=2.0, duty_cycle=0.5),
        image_plate=SimpleNamespace(shutter_mode=shutter, shutter_open=1.0),
    )
    return CameraExposureScheduler(camera_id="room-camera", program_generation=3).begin_frame(scene)


def test_fanout_uses_same_camera_weights_once_and_keeps_identity():
    graph = make_camera_trigger_graph(["Key", "Fill", "Rim"], bus_keys=["Bus"])
    graph.set_weight("Flash", "Bus", 1.0)
    graph.set_weight("Bus", "Key", 1.0)
    graph.set_weight("Bus", "Fill", 0.5)
    exposure = frame()
    result = resolve_camera_trigger_bus(exposure, graph, ["Key", "Fill", "Rim"])
    assert result.frame is exposure
    assert result.frame.causality_keys == exposure.causality_keys
    assert result.signals == {"Key": (0.5, 0.5, 0, 0), "Fill": (0.25, 0.25, 0, 0), "Rim": (0, 0, 0, 0)}
    assert result.transport_contract["engine"] == "graph_solver"
    assert result.transport_contract["representation"] == "torch.complex128"


def test_existing_serialization_and_repatch_do_not_recreate_disconnected_edges():
    graph = make_camera_trigger_graph(["Key"])
    graph.set_weight("Flash", "Key", 1)
    graph = RoutingGraph.from_dict(graph.to_dict())
    assert resolve_camera_trigger_bus(frame(), graph, ["Key"]).signals["Key"] == (0.5, 0.5, 0, 0)
    graph.set_weight("Flash", "Key", 0)
    graph.prune()
    assert resolve_camera_trigger_bus(frame(), graph, ["Key"]).signals["Key"] == (0, 0, 0, 0)
    assert graph.edges == []


def test_sync_and_sensor_gate_are_independent_of_flash():
    graph = make_camera_trigger_graph(["Trigger", "Sensor", "Key"])
    for src, dst in [("Sync", "Trigger"), ("Gate", "Sensor"), ("Flash", "Key")]:
        graph.set_weight(src, dst, 1)
    outputs = resolve_camera_trigger_bus(frame(enabled=False), graph, ["Trigger", "Sensor", "Key"]).signals
    assert outputs == {"Trigger": (1, 0, 0, 0), "Sensor": (1, 1, 1, 1), "Key": (0, 0, 0, 0)}
    assert resolve_camera_trigger_bus(frame(shutter="closed"), graph, ["Sensor"]).signals["Sensor"] == (0, 0, 0, 0)


def test_camera_duration_not_compute_budget_and_loop_uses_real_solver():
    graph = make_camera_trigger_graph(["Key"], bus_keys=["A", "B"])
    graph.set_weight("Sync", "A", 1)
    graph.set_weight("A", "B", 0.5)
    graph.set_weight("B", "A", 0.25)
    graph.set_weight("B", "Key", 1)
    exposure = frame(duration=0.125)
    result = resolve_camera_trigger_bus(exposure, graph, ["Key"])
    assert result.signals["Key"] == pytest.approx((4/7, 0, 0, 0))
    assert result.frame.dt == 0.125
    assert result.transport_contract["sample_rate"] == 32


def test_phase_is_preserved_as_control_not_projected_to_photons():
    graph = make_camera_trigger_graph(["Key"])
    graph.set_weight("Flash", "Key", 1)
    graph.set_angle_rad("Flash", "Key", math.pi / 2)
    result = resolve_camera_trigger_bus(frame(), graph, ["Key"])
    assert result.signals["Key"] == pytest.approx((0.5j, 0.5j, 0, 0))


def test_nonzero_scene_time_and_single_slice_keep_original_boundaries():
    graph = make_camera_trigger_graph(["Key"])
    graph.set_weight("Sync", "Key", 1)
    scene = SimpleNamespace(camera_light_burst=SimpleNamespace(
        stages=1, exposure_time_s=1/125, enabled=False), image_plate=None)
    exposure = CameraExposureScheduler().begin_frame(scene, t0_s=3.0)
    result = resolve_camera_trigger_bus(exposure, graph, ["Key"])
    assert result.frame is exposure
    assert result.frame.slices[0].t0 == 3.0
    assert result.frame.slices[0].t1 == exposure.t1
    assert result.signals["Key"] == (1,)


@pytest.mark.parametrize("bad", ["delay", "negative_delay", "feedback", "nonfinite", "source", "transfer", "unknown"])
def test_unsupported_patch_is_not_silently_ignored(bad):
    graph = make_camera_trigger_graph(["Key"])
    graph.set_weight("Flash", "Key", 1)
    if bad == "delay": graph.edges[0].delay_s = 0.001
    if bad == "negative_delay": graph.edges[0].delay_s = -0.001
    if bad == "feedback": graph.feedback.enabled = True
    if bad == "nonfinite": graph.edges[0].weight = float("nan")
    if bad == "source": graph.set_weight("Key", "Flash", 1)
    if bad == "transfer": graph.edges[0].transfer_spec.cable_count = 2
    if bad == "unknown": graph.edges[0].dst_key = "Missing"
    with pytest.raises(ValueError): resolve_camera_trigger_bus(frame(), graph, ["Key"])


def test_repeated_resolve_does_not_accumulate_or_change_camera():
    graph = make_camera_trigger_graph(["Key"])
    graph.set_weight("Sync", "Key", 1)
    exposure = frame()
    first = resolve_camera_trigger_bus(exposure, graph, ["Key"])
    assert resolve_camera_trigger_bus(exposure, graph, ["Key"]).signals == first.signals
    with pytest.raises(ValueError, match="ordered camera frame"):
        resolve_camera_trigger_bus(replace(exposure, slices=tuple(reversed(exposure.slices))), graph, ["Key"])


def test_existing_mixer_views_same_graph_without_synthetic_audio_nodes():
    from analytic_driver import RoutingGridView
    from analytic_routing import _active_routing_graph_and_instance, _routing_grid_node_keys
    graph = make_camera_trigger_graph(["Key", "Fill"], bus_keys=["Bus"])
    patch, view = make_camera_trigger_panel(graph)
    assert type(view) is RoutingGridView
    assert _active_routing_graph_and_instance(patch, view.active_key)[0] is graph
    assert _routing_grid_node_keys(patch, view.active_key) == graph.node_keys()
    assert not patch.voices and not patch.mixers and patch.system_audio is None
