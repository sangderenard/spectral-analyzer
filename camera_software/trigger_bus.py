"""Camera exposure signals on Pluck's existing mixer and GraphSolver.

This is a bounded camera control bus, not another exposure scheduler or light
transport engine. The caller retains the original ExposureFrame. A result is a
complex control signal per named receiver and slice, not emitted photon energy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from graph_solver import TensorNode
from network_materializer import compile_nodes, routing_edge_to_tensor_edge
from routing_engine import EdgeTransferSpec, RoutingGraph, RouterInstance, TensorPortContract

from .exposure_timing import ExposureFrame
from .trigger_generators import GENERATOR_KEYS, camera_generator_signals, validate_generators


CAMERA_TRIGGER_PORTS = ("Sync", "Flash", "Gate")


def validate_flash_illumination_graph(graph: RoutingGraph, *, step_s: float = .0001) -> float:
    """Admit finite delayed routes from Flash and camera-triggered modulators."""
    active=[edge for edge in graph.edges if abs(edge.weight)>1e-12]
    keys=set(graph.node_keys())
    if any(edge.src_key not in keys or edge.dst_key not in keys for edge in active):
        raise ValueError("camera flash route names an unregistered node")
    if any(not math.isfinite(edge.delay_s) or edge.delay_s<0 for edge in active):
        raise ValueError("camera flash wire delay must be finite and nonnegative")
    if any(edge.src_key in ("Sync", "Gate") for edge in active):
        raise ValueError("light trigger receivers currently require a Flash-sourced graph")
    if any(edge.dst_key in GENERATOR_KEYS for edge in active):
        raise ValueError("camera generator nodes are authored signal sources")
    delays=[edge.delay_s for edge in active if edge.delay_s>0]
    if any(abs(round(value/step_s)*step_s-value)>1e-8 for value in delays):
        raise ValueError("camera wire delays require 0.1 ms precision")
    if delays:
        links={key:[] for key in graph.node_keys()}
        for edge in active:links[edge.src_key].append(edge.dst_key)
        visiting=set();visited=set()
        def walk(key):
            if key in visiting:raise ValueError("delayed camera flash routing cannot contain a cycle")
            if key in visited:return
            visiting.add(key)
            for other in links[key]:walk(other)
            visiting.remove(key);visited.add(key)
        for key in links:walk(key)
    return sum(delays)


def make_camera_trigger_graph(receiver_keys: Iterable[str], *, bus_keys: Iterable[str] = (), generator_keys: Iterable[str] = ()) -> RoutingGraph:
    """Register camera, wire-bus and exact receiver identities, initially unpatched.

    Sync is one at the first slice, Flash carries the scheduler's flash_weight,
    and Gate is one while the sensor integrates. Optional generator nodes receive
    camera-triggered sample arrays from the existing Spectral signal fields.
    Patch with set_weight() or RoutingGridView; save via RoutingGraph.to_dict().
    """
    keys = [*CAMERA_TRIGGER_PORTS, *bus_keys, *receiver_keys, *generator_keys]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("trigger bus node keys must be non-empty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("trigger bus node keys must be unique, including camera ports")
    graph = RoutingGraph()
    for key in keys:
        graph.add_node(key)
    return graph


def make_camera_trigger_panel(graph: RoutingGraph):
    """Return the existing AnalyticPatch and RoutingGridView on this same graph.

    No mixer UI is copied or reimplemented. The named-router view admits graph
    nodes directly, without inventing audio voices for the camera or fixtures.
    The caller uses view.render_surface(patch, w, h) / view.handle_event(...).
    """
    from analytic_driver import AnalyticPatch, RoutingGridView

    router = RouterInstance(key="camera-trigger", router_type="", label="Camera trigger bus", graph=graph)
    patch = AnalyticPatch(name=router.label, system_audio=None, mixers=[], routers=[router])
    view = RoutingGridView()
    view.active_key = "__router__:" + router.key
    return patch, view


@dataclass(frozen=True)
class CameraTriggerBusResult:
    """Signals retain the camera's original frame, slice and causality identity."""

    frame: ExposureFrame
    signals: dict[str, tuple[complex, ...]]
    transport_contract: dict


def resolve_camera_trigger_bus(
    frame: ExposureFrame, graph: RoutingGraph, receiver_keys: Iterable[str], *, flash_signal_mode="weighted", generators=None
) -> CameraTriggerBusResult:
    """Resolve a camera-authored exposure through the canonical graph runtime.

    The default Flash source retains the camera's physical flash_weight. A
    manual control preview can request a unit-level pulse without changing the
    frame or its weights. Delays are measured on the supplied frame's uniform
    time grid; variable-duration slices cannot represent them faithfully.
    A future optical-slave/peanut receiver must be driven by detected light, not
    by relabeling a direct electrical wire as optical detection.
    """
    keys = graph.node_keys()
    receivers = list(receiver_keys)
    generator_specs = validate_generators(generators)
    generator_keys = {item["key"] for item in generator_specs}
    if len(set(keys)) != len(keys) or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("trigger bus node keys must be unique non-empty strings")
    if len(set(receivers)) != len(receivers) or not receivers:
        raise ValueError("trigger bus requires unique receiver keys")
    missing = (set(CAMERA_TRIGGER_PORTS) | set(receivers)) - set(keys)
    if missing:
        raise ValueError(f"unregistered trigger bus nodes: {sorted(missing)}")
    if generator_keys - set(keys):
        raise ValueError("camera generator settings name an unregistered routing source")
    if graph.param_edges or graph.meta_edges:
        raise ValueError("trigger wire bus does not consume parameter or meta edges")
    if graph.feedback.enabled or graph.latency_compensation:
        raise ValueError("trigger wire bus does not consume feedback or latency compensation settings")
    for edge in graph.edges:
        if edge.src_key not in keys or edge.dst_key not in keys:
            raise ValueError("trigger bus edge refers to an unregistered node")
        if edge.dst_key in CAMERA_TRIGGER_PORTS:
            raise ValueError("camera-authored source ports cannot receive patched signals")
        if edge.dst_key in GENERATOR_KEYS:
            raise ValueError("camera generator nodes cannot receive patched signals")
        if not math.isfinite(edge.weight) or not math.isfinite(edge.angle_rad):
            raise ValueError("trigger bus gains and phase must be finite")
        if not math.isfinite(edge.delay_s) or edge.delay_s < 0 or getattr(edge, "delay_samples", 0) < 0:
            raise ValueError("trigger wire delay must be finite and nonnegative")
        if edge.saturation or edge.edge_kind not in ("signal", "control"):
            raise ValueError("trigger wire bus requires linear signal/control edges")
        if edge.tensor_contract != TensorPortContract() or edge.transfer_spec != EdgeTransferSpec():
            raise ValueError("trigger wire bus requires scalar identity transport")
    n = len(frame.slices)
    if not n or not math.isfinite(frame.dt) or frame.dt <= 0:
        raise ValueError("trigger bus requires a non-empty positive-duration camera frame")
    for index, item in enumerate(frame.slices):
        if (item.frame_id != frame.frame_id or item.camera_id != frame.camera_id
                or item.program_generation != frame.program_generation or item.slice_id != index):
            raise ValueError("trigger bus slices must belong to their ordered camera frame")
        if (not math.isfinite(item.t0) or not math.isfinite(item.t1) or item.t1 <= item.t0
                or item.t0 != (frame.t0 if index == 0 else frame.slices[index - 1].t1)
                or not math.isfinite(item.flash_weight)):
            raise ValueError("trigger bus requires finite, contiguous camera slices")
    if frame.slices[-1].t1 != frame.t1 or len(set(frame.causality_keys)) != n:
        raise ValueError("trigger bus requires complete, unique camera slice identities")
    if flash_signal_mode not in ("weighted", "level"):
        raise ValueError("unknown camera Flash signal mode")
    if any(edge.delay_s > 0 or getattr(edge, "delay_samples", 0) > 0 for edge in graph.edges):
        dt=frame.dt/n
        if any(abs(item.dt-dt)>max(1e-10,dt*1e-6) for item in frame.slices):
            raise ValueError("delayed camera routing requires uniform camera slices")

    # Use the very same RoutingEdges that the mixer edits, via the permanent
    # materializer entry. No parallel graph traversal or exposure weighting.
    compiled = compile_nodes(
        [TensorNode(key) for key in keys],
        [routing_edge_to_tensor_edge(edge) for edge in graph.edges],
        sample_rate=n / frame.dt,
        output_node_keys=receivers,
    )
    inputs = {
        "Sync": torch.tensor([1.0] + [0.0] * (n - 1), dtype=torch.complex128),
        "Flash": torch.tensor([s.flash_weight if flash_signal_mode=="weighted" else float(s.flash_active)
                               for s in frame.slices], dtype=torch.complex128),
        "Gate": torch.tensor([float(s.sensor_integrating) for s in frame.slices], dtype=torch.complex128),
    }
    inputs.update({key: torch.tensor(series, dtype=torch.complex128)
                   for key, series in camera_generator_signals(frame, generator_specs).items() if key in keys})
    values = compiled.process_window(inputs, n_frames=n)
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("trigger bus resolved a non-finite signal")
    return CameraTriggerBusResult(
        frame=frame,
        signals={key: tuple(values[key].reshape(n).tolist()) for key in receivers},
        transport_contract=compiled.transport_contract(),
    )
