"""Cold assembly of emitter-to-sensor optical component chains.

This module connects the existing component contracts; it is not another
transport solver. Each component keeps ownership of its T1/T2/T3/T4 artifact,
while the chain compiler namespaces nodes, preserves native payloads, and
authors explicit free-space links between component ports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np

from camera_designer.camera_preset import EmitterSpec

from .optical_components import (
    CompiledOpticalComponent,
    CompoundLensComponent,
    OpticalComponent,
    OpticalEngine,
    OpticalPortDirection,
    OpticalPortSpec,
)
from .optical_transport_graph import (
    CompiledOpticalTransportGraph,
    OpticalExecutionDomain,
    OpticalLinkSpec,
    OpticalNodeSpec,
    OpticalRepresentation,
    OpticalTransportGraphSpec,
    compile_optical_graph,
)
from .sensor_back import SensorBackProfile


OPTICAL_CHAIN_SCHEMA = "optical-component-chain-v1"


def _plain(value: Any) -> Any:
    """Convert authored profile data to a stable contract mapping."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {
            item.name: _plain(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def emitter_contract(spec: EmitterSpec) -> dict[str, Any]:
    profile = spec.resolve_profile()
    if not spec.enabled:
        raise ValueError("an optical-chain emitter endpoint must be enabled")
    if not math.isfinite(float(spec.radius)) or float(spec.radius) <= 0.0:
        raise ValueError("emitter endpoint radius must be finite and positive")
    if int(spec.spatial_samples) < 1:
        raise ValueError("emitter endpoint requires positive spatial samples")
    if profile is None:
        raise ValueError("emitter endpoint has no resolvable EmitterProfile")
    return {
        "schema": "physical-emitter-endpoint-v1",
        "placement": _plain(spec.to_dict()),
        "profile": _plain(profile.to_dict()),
        "source_state": "pipeline-owned-contiguous-complex-mode-block",
        "phase_transport": "fixed-or-continuous-complex",
        "polarization_transport": "jones-complete",
    }


def sensor_contract(profile: SensorBackProfile) -> dict[str, Any]:
    exposure = float(profile.exposure_time_s)
    scale = float(profile.energy_scale)
    if not math.isfinite(exposure) or exposure <= 0.0:
        raise ValueError("sensor endpoint exposure time must be finite and positive")
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("sensor endpoint energy scale must be finite and non-negative")
    return {
        "schema": "physical-sensor-endpoint-v1",
        "label": str(profile.label),
        "geometry": _plain(profile.geometry),
        "sensor_chip": _plain(profile.sensor_chip),
        "film_emulsion": _plain(profile.film_emulsion),
        "color_science": _plain(profile.color_science),
        "exposure_time_s": exposure,
        "energy_scale": scale,
        "reception": "camera-owned-coherent-or-stochastic-closure",
        "output_pipeline": [
            "raw", "develop", "readout", "color-science", "output",
        ],
    }


def _axis(value: Sequence[float], name: str) -> tuple[float, float, float]:
    result = np.asarray(value, np.float64).reshape(3)
    length = float(np.linalg.norm(result))
    if not np.all(np.isfinite(result)) or length <= 1.0e-12:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    result /= length
    return tuple(float(item) for item in result)


def _center(value: Sequence[float], name: str) -> tuple[float, float, float]:
    result = np.asarray(value, np.float64).reshape(3)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite coordinates")
    return tuple(float(item) for item in result)


@dataclass(frozen=True)
class EmitterEndpointComponent:
    """First-class graph source backed by the existing EmitterProfile system."""

    emitter: EmitterSpec
    key: str = "emitter.endpoint"
    axis: tuple[float, float, float] | None = None

    def compile(
        self,
        lane_count: int,
        engine: OpticalEngine | str = OpticalEngine.AUTO,
    ) -> CompiledOpticalComponent:
        selected = OpticalEngine(engine)
        if selected is OpticalEngine.AUTO:
            selected = OpticalEngine.RAY
        if selected not in {
            OpticalEngine.RAY, OpticalEngine.WAVE, OpticalEngine.HYBRID,
        }:
            raise RuntimeError("emitter endpoints support ray, wave, or hybrid")
        direction = _axis(
            self.axis if self.axis is not None else self.emitter.normal,
            "emitter axis",
        )
        center = _center(self.emitter.pos, "emitter center")
        contract = emitter_contract(self.emitter)
        node = OpticalNodeSpec(
            f"{self.key}.source",
            "physical-emitter-source",
            OpticalExecutionDomain.PIPELINE_PORT,
            OpticalRepresentation.COMPLEX_RAY,
            OpticalRepresentation.COMPLEX_RAY,
            int(lane_count),
            directionality="forward",
            parameters={
                "emitter": contract,
                "source_port": "output",
            },
        )
        graph = compile_optical_graph(OpticalTransportGraphSpec(
            nodes=(node,),
            links=(),
            entry_keys=(node.key,),
            product_keys=(node.key,),
        ))
        component = CompiledOpticalComponent(
            key=self.key,
            component_kind="physical-emitter-endpoint",
            lane_count=int(lane_count),
            graph=graph,
            ports=(OpticalPortSpec(
                f"{self.key}.output",
                OpticalPortDirection.OUTPUT,
                OpticalRepresentation.COMPLEX_RAY,
                direction,
                center,
                permits_fanout=True,
            ),),
            metadata={
                "selected_engine": selected.value,
                "supported_engines": ("ray", "wave", "hybrid"),
                "emitter": contract,
            },
        )
        component.validate()
        return component


@dataclass(frozen=True)
class SensorEndpointComponent:
    """First-class graph sink backed by the existing SensorBackProfile."""

    profile: SensorBackProfile
    key: str = "sensor.endpoint"
    center_m: tuple[float, float, float] = (0.10, 0.0, 0.0)
    axis: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    resolution: tuple[int, int] = (256, 256)

    def compile(
        self,
        lane_count: int,
        engine: OpticalEngine | str = OpticalEngine.AUTO,
    ) -> CompiledOpticalComponent:
        selected = OpticalEngine(engine)
        if selected is OpticalEngine.AUTO:
            selected = OpticalEngine.RAY
        if selected not in {
            OpticalEngine.RAY, OpticalEngine.WAVE, OpticalEngine.HYBRID,
        }:
            raise RuntimeError("sensor endpoints support ray, wave, or hybrid")
        width, height = (int(value) for value in self.resolution)
        if width < 1 or height < 1:
            raise ValueError("sensor endpoint resolution must be positive")
        direction = _axis(self.axis, "sensor axis")
        center = _center(self.center_m, "sensor center")
        contract = sensor_contract(self.profile)
        node = OpticalNodeSpec(
            f"{self.key}.reception",
            "physical-sensor-reception",
            OpticalExecutionDomain.PIPELINE_PORT,
            OpticalRepresentation.COMPLEX_RAY,
            OpticalRepresentation.COMPLEX_RAY,
            int(lane_count),
            directionality="forward",
            persistent_state=False,
            parameters={
                "detector": contract,
                "resolution": [width, height],
                "reception_port": "input",
                "reception_capacity": max(64, width * height),
            },
        )
        graph = compile_optical_graph(OpticalTransportGraphSpec(
            nodes=(node,),
            links=(),
            entry_keys=(node.key,),
            product_keys=(node.key,),
        ))
        component = CompiledOpticalComponent(
            key=self.key,
            component_kind="physical-sensor-endpoint",
            lane_count=int(lane_count),
            graph=graph,
            ports=(OpticalPortSpec(
                f"{self.key}.input",
                OpticalPortDirection.INPUT,
                OpticalRepresentation.COMPLEX_RAY,
                direction,
                center,
                coherence_policy="coherent-accumulate",
                permits_fanin=True,
            ),),
            metadata={
                "selected_engine": selected.value,
                "supported_engines": ("ray", "wave", "hybrid"),
                "sensor": contract,
                "resolution": (width, height),
            },
        )
        component.validate()
        return component


@dataclass(frozen=True)
class OpticalChainElement:
    instance_key: str
    component: OpticalComponent
    engine: OpticalEngine | str = OpticalEngine.AUTO

    def validate(self) -> None:
        if not self.instance_key.strip():
            raise ValueError("optical chain instance keys must be non-empty")


@dataclass(frozen=True)
class OpticalChainSpec:
    key: str
    lane_count: int
    elements: tuple[OpticalChainElement, ...]
    schema: str = OPTICAL_CHAIN_SCHEMA

    def validate(self) -> None:
        if self.schema != OPTICAL_CHAIN_SCHEMA:
            raise ValueError(f"unsupported optical chain schema {self.schema!r}")
        if not self.key.strip():
            raise ValueError("optical chain key must be non-empty")
        if len(self.elements) < 2:
            raise ValueError("optical chains require at least source and sink")
        for element in self.elements:
            element.validate()
        keys = tuple(element.instance_key for element in self.elements)
        if len(set(keys)) != len(keys):
            raise ValueError("optical chain instance keys must be unique")

    def replace_component(
        self, instance_key: str, component: OpticalComponent
    ) -> "OpticalChainSpec":
        """Return a new chain revision with one named component replaced."""

        wanted = str(instance_key)
        found = False
        elements = []
        for element in self.elements:
            if element.instance_key == wanted:
                found = True
                elements.append(replace(element, component=component))
            else:
                elements.append(element)
        if not found:
            raise KeyError(f"optical chain has no instance {wanted!r}")
        result = replace(self, elements=tuple(elements))
        result.validate()
        return result


@dataclass(frozen=True)
class OpticalChainConnection:
    src_instance: str
    src_port: str
    dst_instance: str
    dst_port: str
    distance_m: float
    axis_alignment: float
    adapter: str


@dataclass(frozen=True)
class CompiledOpticalChain:
    spec: OpticalChainSpec
    graph: CompiledOpticalTransportGraph
    components: tuple[CompiledOpticalComponent, ...]
    connections: tuple[OpticalChainConnection, ...]

    def contract(self) -> dict[str, Any]:
        return {
            "schema": self.spec.schema,
            "key": self.spec.key,
            "lane_count": self.spec.lane_count,
            "components": [
                {
                    "instance_key": element.instance_key,
                    "component": component.contract(),
                }
                for element, component in zip(
                    self.spec.elements, self.components
                )
            ],
            "connections": [asdict(value) for value in self.connections],
            "graph": self.graph.contract(),
            "output_node": self.graph.spec.product_keys[0],
        }


def _input_port(component: CompiledOpticalComponent) -> OpticalPortSpec:
    direct = [
        port for port in component.ports
        if port.direction is OpticalPortDirection.INPUT
    ]
    if direct:
        return direct[0]
    bidirectional = [
        port for port in component.ports
        if port.direction is OpticalPortDirection.BIDIRECTIONAL
    ]
    if bidirectional:
        return bidirectional[0]
    raise ValueError(f"component {component.key!r} has no chain input port")


def _output_port(component: CompiledOpticalComponent) -> OpticalPortSpec:
    direct = [
        port for port in component.ports
        if port.direction is OpticalPortDirection.OUTPUT
    ]
    if direct:
        return direct[-1]
    bidirectional = [
        port for port in component.ports
        if port.direction is OpticalPortDirection.BIDIRECTIONAL
    ]
    if bidirectional:
        return bidirectional[-1]
    raise ValueError(f"component {component.key!r} has no chain output port")


def compile_optical_chain(spec: OpticalChainSpec) -> CompiledOpticalChain:
    """Namespace and connect independently compiled optical components."""

    spec.validate()
    compiled = tuple(
        element.component.compile(spec.lane_count, element.engine)
        for element in spec.elements
    )
    if not any(
        port.direction is OpticalPortDirection.OUTPUT
        for port in compiled[0].ports
    ):
        raise ValueError("the first optical-chain component must be a source")
    if not any(
        port.direction is OpticalPortDirection.INPUT
        for port in compiled[-1].ports
    ):
        raise ValueError("the last optical-chain component must be a sink")

    nodes: list[OpticalNodeSpec] = []
    links: list[OpticalLinkSpec] = []
    payloads: dict[str, np.ndarray] = {}
    interfaces: dict[tuple[str, str], Any] = {}
    entry_keys: list[str] = []
    product_keys: list[str] = []
    node_maps: list[dict[str, str]] = []

    for element, component in zip(spec.elements, compiled):
        prefix = f"{spec.key}.{element.instance_key}"
        mapping = {
            node.key: f"{prefix}.{node.key}"
            for node in component.graph.spec.nodes
        }
        node_maps.append(mapping)
        nodes.extend(
            replace(node, key=mapping[node.key])
            for node in component.graph.spec.nodes
        )
        links.extend(
            replace(
                link,
                src_key=mapping[link.src_key],
                dst_key=mapping[link.dst_key],
            )
            for link in component.graph.spec.links
        )
        for key, payload in component.graph.t2_payloads.items():
            payloads[mapping[key]] = payload
        for (src, dst), interface in component.graph.field_interfaces.items():
            interfaces[(mapping[src], mapping[dst])] = interface
        entry_keys.append(mapping[component.graph.spec.entry_keys[0]])
        product_keys.append(mapping[component.graph.spec.product_keys[0]])

    connections: list[OpticalChainConnection] = []
    for index in range(len(compiled) - 1):
        source = compiled[index]
        destination = compiled[index + 1]
        source_port = _output_port(source)
        destination_port = _input_port(destination)
        if source_port.representation is not destination_port.representation:
            raise ValueError(
                f"{source.key}->{destination.key} changes representation "
                "without a component adapter"
            )
        if source_port.medium != destination_port.medium:
            raise ValueError(
                f"{source.key}->{destination.key} crosses "
                f"{source_port.medium!r}->{destination_port.medium!r} "
                "without a boundary component"
            )
        if (
            source_port.spectral_mode != destination_port.spectral_mode
            and "fixed-or-continuous" not in {
                source_port.spectral_mode,
                destination_port.spectral_mode,
            }
        ):
            raise ValueError(
                f"{source.key}->{destination.key} has incompatible spectral "
                "contracts"
            )
        if (
            source_port.polarization_basis
            != destination_port.polarization_basis
        ):
            raise ValueError(
                f"{source.key}->{destination.key} requires a polarization "
                "basis adapter"
            )
        if source_port.normalization != destination_port.normalization:
            raise ValueError(
                f"{source.key}->{destination.key} requires an amplitude "
                "normalization adapter"
            )
        source_center = np.asarray(source_port.center_m, np.float64)
        destination_center = np.asarray(destination_port.center_m, np.float64)
        source_axis = np.asarray(source_port.axis, np.float64)
        destination_axis = np.asarray(destination_port.axis, np.float64)
        alignment = float(np.dot(source_axis, destination_axis))
        distance = float(np.linalg.norm(destination_center-source_center))
        adapter = "free-space-rigid-port-link"
        links.append(OpticalLinkSpec(
            product_keys[index],
            entry_keys[index + 1],
            "component-port-connect",
            adapter=adapter,
            parameters={
                "src_port": source_port.key,
                "dst_port": destination_port.key,
                "distance_m": distance,
                "axis_alignment": alignment,
                "medium": source_port.medium,
            },
        ))
        connections.append(OpticalChainConnection(
            spec.elements[index].instance_key,
            source_port.key,
            spec.elements[index + 1].instance_key,
            destination_port.key,
            distance,
            alignment,
            adapter,
        ))

    graph = compile_optical_graph(
        OpticalTransportGraphSpec(
            nodes=tuple(nodes),
            links=tuple(links),
            entry_keys=(entry_keys[0],),
            product_keys=(product_keys[-1],),
        ),
        t2_payloads=payloads,
        field_interfaces=interfaces,
    )
    return CompiledOpticalChain(
        spec=spec,
        graph=graph,
        components=compiled,
        connections=tuple(connections),
    )


def light_table_chain(
    lens: OpticalComponent,
    aperture: OpticalComponent,
    *,
    emitter: EmitterEndpointComponent | None = None,
    sensor: SensorEndpointComponent | None = None,
    lane_count: int = 1,
    key: str = "light-table",
) -> OpticalChainSpec:
    """Author a reusable emitter → aperture → lens → sensor experiment."""

    if emitter is None:
        emitter = EmitterEndpointComponent(EmitterSpec(
            pos=(-0.10, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0),
            radius=0.012,
            spatial_samples=64,
            profile_name="laser_532nm_green",
            label="light-table emitter",
        ))
    if sensor is None:
        sensor_x = 0.12
        if isinstance(lens, CompoundLensComponent):
            last_x = max(
                float(getattr(value, "x_pos", -math.inf))
                for value in lens.lens.elements
            )
            focal_distance = abs(float(lens.lens.f_eff))
            if math.isfinite(last_x) and math.isfinite(focal_distance):
                sensor_x = last_x + focal_distance
        sensor = SensorEndpointComponent(
            SensorBackProfile.fullframe_35mm(),
            center_m=(sensor_x, 0.0, 0.0),
            axis=(-1.0, 0.0, 0.0),
            resolution=(256, 256),
        )
    result = OpticalChainSpec(
        key=str(key),
        lane_count=int(lane_count),
        elements=(
            OpticalChainElement("emitter", emitter),
            OpticalChainElement("aperture", aperture),
            OpticalChainElement("lens", lens),
            OpticalChainElement("sensor", sensor),
        ),
    )
    result.validate()
    return result


__all__ = [
    "OPTICAL_CHAIN_SCHEMA",
    "EmitterEndpointComponent",
    "SensorEndpointComponent",
    "OpticalChainElement",
    "OpticalChainSpec",
    "OpticalChainConnection",
    "CompiledOpticalChain",
    "compile_optical_chain",
    "light_table_chain",
    "emitter_contract",
    "sensor_contract",
]
