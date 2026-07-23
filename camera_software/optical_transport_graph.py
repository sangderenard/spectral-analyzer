"""Compiled graph contracts joining parametric T2 and wave-arena T4 transport.

This module is deliberately not an optical solver.  It gives the existing
``GraphSolver`` topology authority over modular optical systems, then lowers
nodes into artifacts consumed by the existing pipeline:

* fused exact-parametric payloads execute in T2;
* transverse field propagation executes in persistent T4 wave arenas;
* explicit adapter nodes cross ray/complex-ray/field representations.

The graph is never interpreted node-by-node in a ray shader hot loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OPTICAL_GRAPH_SCHEMA = "optical-transport-graph-v1"
SUPPORTED_LANE_COUNTS = (1, 3, 4, 8, 16, 32)


class OpticalExecutionDomain(str, Enum):
    PIPELINE_PORT = "pipeline-port"
    T2_PARAMETRIC = "t2-parametric"
    T4_WAVE_ARENA = "t4-wave-arena"
    MAXWELL_ARTIFACT = "maxwell-artifact"


class OpticalRepresentation(str, Enum):
    RAY = "ray"
    COMPLEX_RAY = "complex-ray"
    TRANSVERSE_FIELD = "transverse-complex-field"
    SCATTERING_OPERATOR = "polarized-scattering-operator"


class WavePropagationStyle(str, Enum):
    ANGULAR_SPECTRUM_FFT = "angular-spectrum-fft"
    SPLIT_STEP_FFT = "split-step-fft"
    MAXWELL_PATCH = "maxwell-patch"


class WaveBoundaryStyle(str, Enum):
    PADDED_ABSORBING = "padded-absorbing"
    OPEN_MODAL = "open-modal"
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"


@dataclass(frozen=True)
class OpticalNodeSpec:
    key: str
    operation: str
    domain: OpticalExecutionDomain
    input_representation: OpticalRepresentation
    output_representation: OpticalRepresentation
    lane_count: int
    polarization_components: int = 2
    directionality: str = "bidirectional"
    persistent_state: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.key.strip() or not self.operation.strip():
            raise ValueError("optical graph nodes require non-empty key and operation")
        if self.lane_count not in SUPPORTED_LANE_COUNTS:
            raise ValueError(
                f"unsupported optical lane count {self.lane_count}; "
                f"expected {SUPPORTED_LANE_COUNTS}"
            )
        if self.polarization_components not in (1, 2):
            raise ValueError("optical nodes support one or two field components")
        if self.directionality not in {"forward", "backward", "bidirectional"}:
            raise ValueError("unsupported optical node directionality")
        if (
            self.domain is OpticalExecutionDomain.T4_WAVE_ARENA
            and not self.persistent_state
        ):
            raise ValueError("T4 wave nodes require persistent state")
        if (
            self.output_representation is OpticalRepresentation.TRANSVERSE_FIELD
            and self.polarization_components != 2
        ):
            raise ValueError("transverse field nodes require two components")


@dataclass(frozen=True)
class OpticalLinkSpec:
    src_key: str
    dst_key: str
    semantic_role: str = "optical-transport"
    adapter: str = ""


@dataclass(frozen=True)
class OpticalTransportGraphSpec:
    nodes: tuple[OpticalNodeSpec, ...]
    links: tuple[OpticalLinkSpec, ...]
    entry_keys: tuple[str, ...]
    product_keys: tuple[str, ...]
    schema: str = OPTICAL_GRAPH_SCHEMA

    def validate(self) -> None:
        if self.schema != OPTICAL_GRAPH_SCHEMA:
            raise ValueError(f"unsupported optical graph schema {self.schema!r}")
        by_key: dict[str, OpticalNodeSpec] = {}
        for node in self.nodes:
            node.validate()
            if node.key in by_key:
                raise ValueError(f"duplicate optical node {node.key!r}")
            by_key[node.key] = node
        for key in self.entry_keys + self.product_keys:
            if key not in by_key:
                raise ValueError(f"optical graph references unknown endpoint {key!r}")
        for link in self.links:
            if link.src_key not in by_key or link.dst_key not in by_key:
                raise ValueError("optical link references an unknown node")
            src = by_key[link.src_key]
            dst = by_key[link.dst_key]
            if src.lane_count != dst.lane_count:
                raise ValueError(
                    "lane-count changes require an explicit rechannel module"
                )
            if src.output_representation != dst.input_representation:
                if not link.adapter:
                    raise ValueError(
                        f"{src.key}->{dst.key} changes "
                        f"{src.output_representation.value} to "
                        f"{dst.input_representation.value} without an adapter"
                    )


@dataclass
class CompiledOpticalTransportGraph:
    """Cold-compiled topology plus pipeline-owned runtime artifacts."""

    spec: OpticalTransportGraphSpec
    topology_solver: Any
    t2_payloads: dict[str, np.ndarray]
    t4_descriptors: dict[str, dict[str, Any]]

    def contract(self) -> dict[str, Any]:
        stats = self.topology_solver.graph_stats()
        return {
            "schema": self.spec.schema,
            "graph_engine": "GraphSolver",
            "execution": "compiled-not-hot-interpreted",
            "entry_keys": list(self.spec.entry_keys),
            "product_keys": list(self.spec.product_keys),
            "nodes": [{
                "key": node.key,
                "operation": node.operation,
                "domain": node.domain.value,
                "input": node.input_representation.value,
                "output": node.output_representation.value,
                "lane_count": node.lane_count,
                "polarization_components": node.polarization_components,
                "directionality": node.directionality,
                "persistent_state": node.persistent_state,
            } for node in self.spec.nodes],
            "links": [{
                "src": link.src_key,
                "dst": link.dst_key,
                "semantic_role": link.semantic_role,
                "adapter": link.adapter,
            } for link in self.spec.links],
            "t2_payload_keys": sorted(self.t2_payloads),
            "t4_descriptor_keys": sorted(self.t4_descriptors),
            "graph_stats": dict(stats),
        }


@dataclass(frozen=True)
class InstalledWaveArena:
    """One graph-declared T4 arena registered with the native pipeline."""

    node_key: str
    context_id: int
    propagation: str
    center_m: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius_m: float
    longitudinal_step_m: float
    longitudinal_steps: int


@dataclass(frozen=True)
class InstalledWaveLink:
    """A native full-field edge between compatible persistent T4 ports."""

    src_node_key: str
    dst_node_key: str
    src_context_id: int
    dst_context_id: int


@dataclass
class InstalledOpticalTransportGraph:
    """Cold installation receipt and payload-lifetime owner.

    The native scale-context ABI currently borrows payload memory.  Keeping the
    axis payload arrays here makes their lifetime visibly identical to the
    installation lifetime instead of relying on an incidental local variable.
    """

    compiled: CompiledOpticalTransportGraph
    wave_arenas: tuple[InstalledWaveArena, ...]
    wave_links: tuple[InstalledWaveLink, ...]
    borrowed_payloads: tuple[np.ndarray, ...]
    exact_t2_registration: str

    def contract(self) -> dict[str, Any]:
        return {
            "schema": self.compiled.spec.schema,
            "exact_t2_registration": self.exact_t2_registration,
            "wave_arenas": [
                {
                    "node_key": arena.node_key,
                    "context_id": arena.context_id,
                    "propagation": arena.propagation,
                    "center_m": list(arena.center_m),
                    "axis": list(arena.axis),
                    "radius_m": arena.radius_m,
                    "boundary_geometry": "oriented-plane-to-plane-patch",
                    "transverse_half_extent_m": arena.radius_m,
                    "longitudinal_extent_m": (
                        arena.longitudinal_step_m * arena.longitudinal_steps
                    ),
                    "longitudinal_step_m": arena.longitudinal_step_m,
                    "longitudinal_steps": arena.longitudinal_steps,
                }
                for arena in self.wave_arenas
            ],
            "wave_links": [
                {
                    "src_node_key": link.src_node_key,
                    "dst_node_key": link.dst_node_key,
                    "src_context_id": link.src_context_id,
                    "dst_context_id": link.dst_context_id,
                    "transfer": "persistent-full-field",
                    "resampling": "forbidden",
                }
                for link in self.wave_links
            ],
            "transition_telemetry": {
                "source": "native-wave-arena-stats",
                "available_after_pipeline_start": True,
                "field_texture_source": "persistent-t4-state-display-resolve",
                "field_texture_opt_in": True,
                "entry_adapter": "unit-l2-gaussian-with-transverse-phase",
                "exit_adapter": "power-preserving-first-moment-ray",
            },
        }


def wave_context_nodes(
    key: str,
    lane_count: int,
    *,
    propagation: WavePropagationStyle = WavePropagationStyle.ANGULAR_SPECTRUM_FFT,
    boundary: WaveBoundaryStyle = WaveBoundaryStyle.PADDED_ABSORBING,
    periodic_explicit: bool = False,
    center_m: Sequence[float] | None = None,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    radius_m: float | None = None,
    longitudinal_step_m: float | None = None,
    longitudinal_steps: int | None = None,
    medium_n_real: float = 1.0,
    medium_n_imag: float = 0.0,
) -> tuple[tuple[OpticalNodeSpec, ...], tuple[OpticalLinkSpec, ...]]:
    """Return explicit complex-ray→field→complex-ray T4 boundary modules."""

    if boundary is WaveBoundaryStyle.PERIODIC and not periodic_explicit:
        raise ValueError("periodic FFT boundaries must be explicitly authored")
    prefix = str(key)
    entry = OpticalNodeSpec(
        key=f"{prefix}.entry",
        operation="complex-ray-to-transverse-field",
        domain=OpticalExecutionDomain.PIPELINE_PORT,
        input_representation=OpticalRepresentation.COMPLEX_RAY,
        output_representation=OpticalRepresentation.TRANSVERSE_FIELD,
        lane_count=lane_count,
        polarization_components=2,
    )
    physical: dict[str, Any] = {}
    if center_m is not None:
        center = np.asarray(center_m, np.float64).reshape(-1)
        if center.size != 3 or not np.all(np.isfinite(center)):
            raise ValueError("wave arena center_m must contain three finite values")
        physical["center_m"] = tuple(float(v) for v in center)
    direction = np.asarray(axis, np.float64).reshape(-1)
    if direction.size != 3 or not np.all(np.isfinite(direction)):
        raise ValueError("wave arena axis must contain three finite values")
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("wave arena axis must be non-zero")
    physical["axis"] = tuple(float(v) for v in direction / norm)
    if radius_m is not None:
        if not np.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("wave arena radius_m must be positive")
        physical["radius_m"] = float(radius_m)
    if longitudinal_step_m is not None:
        if not np.isfinite(longitudinal_step_m) or longitudinal_step_m <= 0.0:
            raise ValueError("wave arena longitudinal_step_m must be positive")
        physical["longitudinal_step_m"] = float(longitudinal_step_m)
    if longitudinal_steps is not None:
        if int(longitudinal_steps) <= 0:
            raise ValueError("wave arena longitudinal_steps must be positive")
        physical["longitudinal_steps"] = int(longitudinal_steps)
    if not np.isfinite(medium_n_real) or medium_n_real <= 0.0:
        raise ValueError("wave arena medium_n_real must be positive")
    if not np.isfinite(medium_n_imag) or medium_n_imag < 0.0:
        raise ValueError("wave arena medium_n_imag must be non-negative")

    arena = OpticalNodeSpec(
        key=f"{prefix}.arena",
        operation="wave-propagation",
        domain=OpticalExecutionDomain.T4_WAVE_ARENA,
        input_representation=OpticalRepresentation.TRANSVERSE_FIELD,
        output_representation=OpticalRepresentation.TRANSVERSE_FIELD,
        lane_count=lane_count,
        polarization_components=2,
        persistent_state=True,
        parameters={
            "propagation": propagation.value,
            "boundary": boundary.value,
            "state_layout": "solid-contiguous-state-block",
            "allocation": "cold-only",
            "medium_n_real": float(medium_n_real),
            "medium_n_imag": float(medium_n_imag),
            **physical,
        },
    )
    exit_node = OpticalNodeSpec(
        key=f"{prefix}.exit",
        operation="transverse-field-to-complex-ray",
        domain=OpticalExecutionDomain.PIPELINE_PORT,
        input_representation=OpticalRepresentation.TRANSVERSE_FIELD,
        output_representation=OpticalRepresentation.COMPLEX_RAY,
        lane_count=lane_count,
        polarization_components=2,
    )
    return (
        (entry, arena, exit_node),
        (
            OpticalLinkSpec(entry.key, arena.key, "wave-arena-entry"),
            OpticalLinkSpec(arena.key, exit_node.key, "wave-arena-exit"),
        ),
    )


def install_optical_graph(
    tracer: Any,
    compiled: CompiledOpticalTransportGraph,
    *,
    exact_t2_registered: bool,
    clear_existing_scale_contexts: bool = True,
) -> InstalledOpticalTransportGraph:
    """Install graph-declared arenas through the existing native T4 ABI.

    This does not register a second lens transform.  The fused T2 payload must
    already have been installed by the authoritative camera builder, and the
    caller must explicitly attest to that fact.

    The native implementation owns an exact-lane, padded, bidirectional
    two-component angular-spectrum arena. Unsupported propagation styles are
    rejected rather than silently substituted.
    """

    if compiled.t2_payloads and not exact_t2_registered:
        raise RuntimeError(
            "compiled graph contains exact T2 artifacts but the camera builder "
            "did not confirm their native registration"
        )
    add_context = getattr(tracer, "add_scale_context", None)
    clear_contexts = getattr(tracer, "clear_scale_contexts", None)
    if not callable(add_context):
        raise TypeError("tracer does not expose the native scale-context API")
    if clear_existing_scale_contexts:
        if not callable(clear_contexts):
            raise TypeError("tracer cannot clear pre-graph scale contexts")
        clear_contexts()

    installed: list[InstalledWaveArena] = []
    payloads: list[np.ndarray] = []
    for node in compiled.spec.nodes:
        if node.domain is not OpticalExecutionDomain.T4_WAVE_ARENA:
            continue
        descriptor = compiled.t4_descriptors[node.key]
        propagation = str(descriptor.get("propagation", ""))
        if propagation != WavePropagationStyle.ANGULAR_SPECTRUM_FFT.value:
            raise RuntimeError(
                f"T4 node {node.key!r} requests {propagation!r}, but the live "
                "native backend implements only 'angular-spectrum-fft'; "
                "refusing a scientifically false backend substitution"
            )
        missing = [
            key for key in (
                "center_m", "radius_m", "longitudinal_step_m",
                "longitudinal_steps",
            )
            if key not in descriptor
        ]
        if missing:
            raise ValueError(
                f"T4 node {node.key!r} is not physically installable; "
                f"missing {missing}"
            )
        center = np.asarray(descriptor["center_m"], np.float64).reshape(-1)
        axis = np.asarray(descriptor.get("axis", (0.0, 0.0, 1.0)), np.float64)
        axis = axis.reshape(-1)
        if center.size != 3 or axis.size != 3:
            raise ValueError(f"T4 node {node.key!r} has invalid placement")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-12:
            raise ValueError(f"T4 node {node.key!r} has a zero propagation axis")
        axis = axis / axis_norm
        # Native WaveArena reads its propagation axis from payload[3:6].
        payload = np.ascontiguousarray(
            [0.0, 0.0, 0.0, axis[0], axis[1], axis[2]],
            np.float64,
        )
        payloads.append(payload)
        context_id = int(add_context(
            pos=np.ascontiguousarray(center, np.float64),
            radius=float(descriptor["radius_m"]),
            scale_type=1,  # RT_SCALE_WAVE
            dt_m=float(descriptor["longitudinal_step_m"]),
            n_substeps=int(descriptor["longitudinal_steps"]),
            n_real=float(descriptor.get("medium_n_real", 1.0)),
            n_imag=float(descriptor.get("medium_n_imag", 0.0)),
            context_kind=1,  # SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ
            payload=payload,
        ))
        installed.append(InstalledWaveArena(
            node_key=node.key,
            context_id=context_id,
            propagation=propagation,
            center_m=tuple(float(v) for v in center),
            axis=tuple(float(v) for v in axis),
            radius_m=float(descriptor["radius_m"]),
            longitudinal_step_m=float(descriptor["longitudinal_step_m"]),
            longitudinal_steps=int(descriptor["longitudinal_steps"]),
        ))
    installed_by_node = {arena.node_key: arena for arena in installed}
    native_links: list[InstalledWaveLink] = []
    add_wave_link = getattr(tracer, "add_wave_context_link", None)
    nodes_by_key = {node.key: node for node in compiled.spec.nodes}
    for link in compiled.spec.links:
        src = nodes_by_key[link.src_key]
        dst = nodes_by_key[link.dst_key]
        if (
            src.domain is not OpticalExecutionDomain.T4_WAVE_ARENA
            or dst.domain is not OpticalExecutionDomain.T4_WAVE_ARENA
        ):
            continue
        if not callable(add_wave_link):
            raise TypeError(
                "tracer does not expose native persistent wave-context links"
            )
        src_arena = installed_by_node[src.key]
        dst_arena = installed_by_node[dst.key]
        add_wave_link(src_arena.context_id, dst_arena.context_id)
        native_links.append(InstalledWaveLink(
            src_node_key=src.key,
            dst_node_key=dst.key,
            src_context_id=src_arena.context_id,
            dst_context_id=dst_arena.context_id,
        ))
    return InstalledOpticalTransportGraph(
        compiled=compiled,
        wave_arenas=tuple(installed),
        wave_links=tuple(native_links),
        borrowed_payloads=tuple(payloads),
        exact_t2_registration=(
            "camera-builder-confirmed" if compiled.t2_payloads else "not-required"
        ),
    )


def compile_optical_graph(
    spec: OpticalTransportGraphSpec,
    *,
    t2_payloads: Mapping[str, np.ndarray] | None = None,
) -> CompiledOpticalTransportGraph:
    """Validate with GraphSolver and lower nodes to T2/T4 artifacts."""

    spec.validate()
    from graph_solver import GraphSolver, TensorEdge, TensorNode

    graph_nodes = [
        TensorNode(
            key=node.key,
            layer="master" if node.key in spec.product_keys else node.domain.value,
            transform=None,
            subscription_ports=("optical_contract",),
            subscription_contracts={"optical_contract": {
                "operation": node.operation,
                "domain": node.domain.value,
                "input_representation": node.input_representation.value,
                "output_representation": node.output_representation.value,
                "lane_count": node.lane_count,
                "persistent_state": node.persistent_state,
            }},
        )
        for node in spec.nodes
    ]
    graph_edges = [
        TensorEdge(
            src_key=link.src_key,
            dst_key=link.dst_key,
            weight=1.0 + 0.0j,
            semantic_role=link.semantic_role,
            contract_key=OPTICAL_GRAPH_SCHEMA,
            contract_semantic_role=link.adapter or "representation-preserving",
        )
        for link in spec.links
    ]
    solver = GraphSolver(graph_nodes, graph_edges)
    payload_map = {
        str(key): np.ascontiguousarray(value, np.float32)
        for key, value in dict(t2_payloads or {}).items()
    }
    node_map = {node.key: node for node in spec.nodes}
    unknown_payloads = set(payload_map) - set(node_map)
    if unknown_payloads:
        raise ValueError(f"T2 payloads reference unknown nodes {sorted(unknown_payloads)}")
    for key, payload in payload_map.items():
        if node_map[key].domain is not OpticalExecutionDomain.T2_PARAMETRIC:
            raise ValueError(f"payload node {key!r} is not a T2 node")
        if payload.size < 8 or float(payload[0]) != 14949.0:
            raise ValueError(f"T2 node {key!r} does not contain an exact lens payload")
        if int(payload[5]) != node_map[key].lane_count:
            raise ValueError(f"T2 node {key!r} payload lane count is inconsistent")

    t4 = {
        node.key: dict(node.parameters)
        for node in spec.nodes
        if node.domain is OpticalExecutionDomain.T4_WAVE_ARENA
    }
    return CompiledOpticalTransportGraph(spec, solver, payload_map, t4)


def compile_compound_lens_graph(
    lens: Any,
    *,
    lane_count: int,
    wave_key: str | None = None,
    wave_regions: Sequence[tuple[str, Mapping[str, Any]]] = (),
    propagation: WavePropagationStyle = WavePropagationStyle.ANGULAR_SPECTRUM_FFT,
    boundary: WaveBoundaryStyle = WaveBoundaryStyle.PADDED_ABSORBING,
) -> CompiledOpticalTransportGraph:
    """Compile the existing exact lens payload as the first fused T2 module."""

    payload = np.ascontiguousarray(lens.build_gpu_payload(), np.float32)
    if int(payload[5]) != int(lane_count):
        raise ValueError("compound lens spectral payload does not match lane_count")
    scene = OpticalNodeSpec(
        "scene.complex-rays",
        "pipeline-entry",
        OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY,
        lane_count,
    )
    fused = OpticalNodeSpec(
        "camera.exact-compound-lens",
        "fused-exact-compound-lens",
        OpticalExecutionDomain.T2_PARAMETRIC,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY,
        lane_count,
        parameters={
            "payload_magic": 14949,
            "execution": "existing-t2-parametric-core",
            "hot_interpreter": False,
        },
    )
    sensor = OpticalNodeSpec(
        "camera.sensor-port",
        "pipeline-product",
        OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY,
        lane_count,
    )
    nodes: list[OpticalNodeSpec] = [scene, fused]
    links: list[OpticalLinkSpec] = [
        OpticalLinkSpec(scene.key, fused.key, "camera-entry")
    ]
    tail = fused.key
    authored_regions = list(wave_regions)
    if wave_key:
        if authored_regions:
            raise ValueError("use wave_key or wave_regions, not both")
        authored_regions.append((
            wave_key,
            {"propagation": propagation, "boundary": boundary},
        ))
    for region_key, raw_parameters in authored_regions:
        parameters = dict(raw_parameters)
        if "propagation" in parameters:
            parameters["propagation"] = WavePropagationStyle(
                parameters["propagation"]
            )
        if "boundary" in parameters:
            parameters["boundary"] = WaveBoundaryStyle(parameters["boundary"])
        wave_nodes, wave_links = wave_context_nodes(
            str(region_key),
            lane_count,
            **parameters,
        )
        nodes.extend(wave_nodes)
        links.append(OpticalLinkSpec(tail, wave_nodes[0].key, "wave-port-connect"))
        links.extend(wave_links)
        tail = wave_nodes[-1].key
    nodes.append(sensor)
    links.append(OpticalLinkSpec(tail, sensor.key, "camera-product"))
    return compile_optical_graph(
        OpticalTransportGraphSpec(
            nodes=tuple(nodes),
            links=tuple(links),
            entry_keys=(scene.key,),
            product_keys=(sensor.key,),
        ),
        t2_payloads={fused.key: payload},
    )


__all__ = [
    "OPTICAL_GRAPH_SCHEMA",
    "SUPPORTED_LANE_COUNTS",
    "OpticalExecutionDomain",
    "OpticalRepresentation",
    "WavePropagationStyle",
    "WaveBoundaryStyle",
    "OpticalNodeSpec",
    "OpticalLinkSpec",
    "OpticalTransportGraphSpec",
    "CompiledOpticalTransportGraph",
    "InstalledWaveArena",
    "InstalledOpticalTransportGraph",
    "wave_context_nodes",
    "install_optical_graph",
    "compile_optical_graph",
    "compile_compound_lens_graph",
]
