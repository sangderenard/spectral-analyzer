"""Camera exposure signals on Pluck's existing mixer and GraphSolver.

This is the zero-delay wire-bus seam, not another exposure scheduler or light
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


CAMERA_TRIGGER_PORTS = ("Sync", "Flash", "Gate")


def make_camera_trigger_graph(receiver_keys: Iterable[str], *, bus_keys: Iterable[str] = ()) -> RoutingGraph:
    """Register camera, wire-bus and exact receiver identities, initially unpatched.

    Sync is one at the first slice, Flash carries the scheduler's flash_weight,
    and Gate is one while the sensor integrates. Nothing generates new timing or
    assumes what a receiver does with these signals. Patch with set_weight(), or
    the existing RoutingGridView. Save through RoutingGraph.to_dict().
    """
    keys = [*CAMERA_TRIGGER_PORTS, *bus_keys, *receiver_keys]
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
    frame: ExposureFrame, graph: RoutingGraph, receiver_keys: Iterable[str]
) -> CameraTriggerBusResult:
    """Resolve a camera-authored exposure through the canonical graph runtime.

    This first consumer accepts instantaneous scalar linear wiring, including
    gain, phase, fan-out, fan-in and the solver's algebraic loops. Delays,
    nonlinear receivers, feedback settings and wide transfers need a persistent
    time-domain consumer; reject them rather than silently drop their settings.
    A future optical-slave/peanut receiver must be driven by detected light, not
    by relabeling a direct electrical wire as optical detection.
    """
    keys = graph.node_keys()
    receivers = list(receiver_keys)
    if len(set(keys)) != len(keys) or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("trigger bus node keys must be unique non-empty strings")
    if len(set(receivers)) != len(receivers) or not receivers:
        raise ValueError("trigger bus requires unique receiver keys")
    missing = (set(CAMERA_TRIGGER_PORTS) | set(receivers)) - set(keys)
    if missing:
        raise ValueError(f"unregistered trigger bus nodes: {sorted(missing)}")
    if graph.param_edges or graph.meta_edges:
        raise ValueError("trigger wire bus does not consume parameter or meta edges")
    if graph.feedback.enabled or graph.latency_compensation:
        raise ValueError("trigger wire bus does not consume feedback or latency compensation settings")
    for edge in graph.edges:
        if edge.src_key not in keys or edge.dst_key not in keys:
            raise ValueError("trigger bus edge refers to an unregistered node")
        if edge.dst_key in CAMERA_TRIGGER_PORTS:
            raise ValueError("camera-authored source ports cannot receive patched signals")
        if not math.isfinite(edge.weight) or not math.isfinite(edge.angle_rad):
            raise ValueError("trigger bus gains and phase must be finite")
        if edge.delay_s != 0 or getattr(edge, "delay_samples", 0) != 0:
            raise ValueError("trigger wire bus requires zero delay; delayed receivers need persistent camera time")
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
        "Flash": torch.tensor([s.flash_weight for s in frame.slices], dtype=torch.complex128),
        "Gate": torch.tensor([float(s.sensor_integrating) for s in frame.slices], dtype=torch.complex128),
    }
    values = compiled.process_window(inputs, n_frames=n)
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("trigger bus resolved a non-finite signal")
    return CameraTriggerBusResult(
        frame=frame,
        signals={key: tuple(values[key].reshape(n).tolist()) for key in receivers},
        transport_contract=compiled.transport_contract(),
    )
