"""Canonical loadable optical components for benches and component arenas.

An optical component is not a solver.  It cold-compiles authored intent into
the artifacts already owned by the optical engine:

* fused analytical schedules for T2;
* role-tagged physical geometry and canonical materials for T1/T3;
* localized persistent field regions for T4;
* representative geometry and controls for an OpenGL/UI host.

The live host may inspect these artifacts, but it must never substitute a
private tracer or an approximate transport algorithm for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from camera_designer.compound_optics import CompoundLens
from camera_designer.optical_material import MATERIAL_CATALOG

from .optical_transport_graph import (
    CompiledOpticalTransportGraph,
    OpticalExecutionDomain,
    OpticalLinkSpec,
    OpticalNodeSpec,
    OpticalRepresentation,
    OpticalTransportGraphSpec,
    WaveBoundaryStyle,
    WavePropagationStyle,
    compile_compound_lens_graph,
    compile_optical_graph,
    wave_context_nodes,
)
from .physical_aperture import LivePhysicalAperture
from .specialty_optics import OpticalTriangleAssembly, PentaprismSpec


OPTICAL_COMPONENT_SCHEMA = "optical-component-v1"


class OpticalPortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"


@dataclass(frozen=True)
class OpticalPortSpec:
    key: str
    direction: OpticalPortDirection
    representation: OpticalRepresentation
    axis: tuple[float, float, float]
    center_m: tuple[float, float, float]

    def validate(self) -> None:
        if not self.key.strip():
            raise ValueError("optical component port key must be non-empty")
        axis = np.asarray(self.axis, np.float64)
        center = np.asarray(self.center_m, np.float64)
        if axis.shape != (3,) or center.shape != (3,):
            raise ValueError("optical component ports require 3D center and axis")
        if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(center)):
            raise ValueError("optical component port coordinates must be finite")
        if float(np.linalg.norm(axis)) <= 1.0e-12:
            raise ValueError("optical component port axis must be non-zero")


@dataclass(frozen=True)
class OpticalControlSpec:
    key: str
    label: str
    value_type: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""
    rebuild_scope: str = "component"


@dataclass(frozen=True)
class OpticalMaterialRole:
    role: str
    material_name: str
    interaction: str

    def validate(self) -> None:
        if not self.role.strip() or not self.material_name.strip():
            raise ValueError("material roles require non-empty names")
        if self.material_name not in MATERIAL_CATALOG:
            raise ValueError(
                f"unknown canonical optical material {self.material_name!r}"
            )


@dataclass(frozen=True)
class CompiledOpticalComponent:
    """One immutable component revision lowered for the optical engine."""

    key: str
    component_kind: str
    lane_count: int
    graph: CompiledOpticalTransportGraph
    ports: tuple[OpticalPortSpec, ...]
    controls: tuple[OpticalControlSpec, ...] = ()
    transport_geometry: OpticalTriangleAssembly | None = None
    display_geometry: OpticalTriangleAssembly | None = None
    material_roles: tuple[OpticalMaterialRole, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = OPTICAL_COMPONENT_SCHEMA

    def validate(self) -> None:
        if self.schema != OPTICAL_COMPONENT_SCHEMA:
            raise ValueError(f"unsupported optical component schema {self.schema!r}")
        if not self.key.strip() or not self.component_kind.strip():
            raise ValueError("compiled optical components require identity")
        self.graph.spec.validate()
        if any(node.lane_count != self.lane_count for node in self.graph.spec.nodes):
            raise ValueError("component graph lane width is not uniform")
        for port in self.ports:
            port.validate()
        if len({port.key for port in self.ports}) != len(self.ports):
            raise ValueError("component port keys must be unique")
        for role in self.material_roles:
            role.validate()
        if self.transport_geometry is not None:
            declared = {role.role for role in self.material_roles}
            missing = set(self.transport_geometry.roles) - declared
            if missing:
                raise ValueError(
                    f"transport geometry has unbound material roles {sorted(missing)}"
                )

    def contract(self) -> dict[str, Any]:
        self.validate()
        geometry = self.transport_geometry
        return {
            "schema": self.schema,
            "key": self.key,
            "component_kind": self.component_kind,
            "lane_count": self.lane_count,
            "ports": [{
                "key": port.key,
                "direction": port.direction.value,
                "representation": port.representation.value,
                "axis": list(port.axis),
                "center_m": list(port.center_m),
            } for port in self.ports],
            "controls": [{
                "key": control.key,
                "label": control.label,
                "value_type": control.value_type,
                "default": control.default,
                "minimum": control.minimum,
                "maximum": control.maximum,
                "unit": control.unit,
                "rebuild_scope": control.rebuild_scope,
            } for control in self.controls],
            "material_roles": [{
                "role": role.role,
                "material": role.material_name,
                "interaction": role.interaction,
            } for role in self.material_roles],
            "transport_triangle_count": (
                0 if geometry is None else int(geometry.triangles.shape[0])
            ),
            "graph": self.graph.contract(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NativeOpticalComponentScene:
    """Material-bound T1/T3 scene arrays derived from one compiled component."""

    component_key: str
    frequencies_hz: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    material_indices: np.ndarray
    material_buffer: np.ndarray
    material_count: int
    roles: tuple[str, ...]
    role_interactions: Mapping[str, str]

    def create_tracer(self):
        import _spectral_kernels as kernels
        from mat_flags import (
            MAT_FLAG_ABSORBER,
            MAT_FLAG_APERTURE_STOP,
            MAT_FLAG_TRANSMISSIVE,
        )

        tracer = kernels.RayTracer(
            int(self.triangles.shape[0]),
            np.ascontiguousarray(self.triangles.reshape(-1, 9), np.float64),
            np.ascontiguousarray(self.normals, np.float64),
            np.ascontiguousarray(self.material_indices, np.int32),
            np.ascontiguousarray(self.material_buffer, np.float32),
            int(self.material_count),
            np.ascontiguousarray(self.frequencies_hz, np.float64),
            299_792_458.0,
            np.zeros(len(self.frequencies_hz), np.float64),
        )
        for index, role in enumerate(self.roles):
            interaction = self.role_interactions[role]
            flags = 0
            if interaction == "dielectric-interface":
                flags |= MAT_FLAG_TRANSMISSIVE
            elif interaction == "absorber":
                flags |= MAT_FLAG_ABSORBER
            elif interaction == "aperture-stop":
                flags |= MAT_FLAG_APERTURE_STOP
            tracer.set_tri_ior(index, 1, flags)
            if interaction == "dielectric-interface":
                # OpticalTriangleAssembly uses outward winding. The +normal
                # side is ambient and the -normal side is the glass volume.
                tracer.set_tri_boundary_media(
                    index, 1, -1, int(self.material_indices[index])
                )
        return tracer


class OpticalComponent(Protocol):
    key: str

    def compile(self, lane_count: int) -> CompiledOpticalComponent:
        ...


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, np.float64).reshape(3)
    length = float(np.linalg.norm(result))
    if not np.all(np.isfinite(result)) or length <= 1.0e-12:
        raise ValueError(f"{name} must be a finite non-zero vector")
    return result / length


def _port(
    key: str,
    direction: OpticalPortDirection,
    representation: OpticalRepresentation,
    center: Sequence[float],
    axis: Sequence[float],
) -> OpticalPortSpec:
    return OpticalPortSpec(
        key, direction, representation,
        tuple(float(value) for value in _unit(axis, "port axis")),
        tuple(float(value) for value in np.asarray(center, np.float64).reshape(3)),
    )


def _compile_linear_material_graph(
    *,
    prefix: str,
    lane_count: int,
    operations: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> CompiledOpticalTransportGraph:
    nodes: list[OpticalNodeSpec] = [
        OpticalNodeSpec(
            f"{prefix}.input", "pipeline-entry",
            OpticalExecutionDomain.PIPELINE_PORT,
            OpticalRepresentation.COMPLEX_RAY,
            OpticalRepresentation.COMPLEX_RAY,
            lane_count,
        )
    ]
    links: list[OpticalLinkSpec] = []
    tail = nodes[0].key
    for suffix, operation, parameters in operations:
        node = OpticalNodeSpec(
            f"{prefix}.{suffix}", operation,
            OpticalExecutionDomain.T3_MATERIAL,
            OpticalRepresentation.COMPLEX_RAY,
            OpticalRepresentation.COMPLEX_RAY,
            lane_count,
            parameters=dict(parameters),
        )
        nodes.append(node)
        links.append(OpticalLinkSpec(tail, node.key, operation))
        tail = node.key
    output = OpticalNodeSpec(
        f"{prefix}.output", "pipeline-product",
        OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY,
        lane_count,
    )
    nodes.append(output)
    links.append(OpticalLinkSpec(tail, output.key, "component-product"))
    return compile_optical_graph(OpticalTransportGraphSpec(
        nodes=tuple(nodes),
        links=tuple(links),
        entry_keys=(nodes[0].key,),
        product_keys=(output.key,),
    ))


def build_native_component_scene(
    compiled: CompiledOpticalComponent,
    frequencies_hz: Sequence[float],
) -> NativeOpticalComponentScene:
    """Lower role-tagged geometry through the canonical MatBuf adapter."""

    compiled.validate()
    geometry = compiled.transport_geometry
    if geometry is None:
        raise ValueError(
            f"component {compiled.key!r} has no T1/T3 transport geometry"
        )
    frequencies = np.ascontiguousarray(frequencies_hz, np.float64).reshape(-1)
    if frequencies.size != compiled.lane_count:
        raise ValueError("native component scene frequency count must match lanes")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("native component frequencies must be finite and positive")

    role_bindings = {binding.role: binding for binding in compiled.material_roles}
    n_tri = int(geometry.triangles.shape[0])
    n_bands = int(frequencies.size)
    reflectance = np.zeros((n_tri, n_bands), np.complex128)
    diffusion = np.zeros((n_tri, n_bands), np.float64)
    transmittance = np.zeros((n_tri, n_bands), np.float64)
    ior_real = np.ones((n_tri, n_bands), np.float64)
    ior_imag = np.zeros((n_tri, n_bands), np.float64)
    role_interactions: dict[str, str] = {}
    wavelengths_um = 299_792_458.0/frequencies*1.0e6

    for triangle_index, role in enumerate(geometry.roles):
        binding = role_bindings[role]
        material = MATERIAL_CATALOG[binding.material_name]
        role_interactions[role] = binding.interaction
        n_values = np.asarray([
            material.n_at(float(wavelength))
            for wavelength in wavelengths_um
        ])
        k_values = np.full(n_bands, float(material.k), np.float64)
        ior_real[triangle_index] = n_values
        ior_imag[triangle_index] = k_values
        diffusion[triangle_index] = float(material.scatter_albedo)
        if binding.interaction == "dielectric-interface":
            transmittance[triangle_index] = 1.0
        elif binding.interaction in {
            "analytic-specular-reflection",
            "canonical-material-conductor",
        }:
            fresnel = (
                (1.0-(n_values+1j*k_values))
                / (1.0+(n_values+1j*k_values))
            )
            reflectance[triangle_index] = fresnel

    from ray_tracer_bridge import per_tri_spectral_to_mat_buf

    mat_idx, mat_buf, mat_count = per_tri_spectral_to_mat_buf(
        reflectance.real,
        reflectance.imag,
        diffusion,
        frequencies,
        transmittance_bands=transmittance,
        ior_real_bands=ior_real,
        ior_imag_bands=ior_imag,
    )
    triangles = np.ascontiguousarray(geometry.triangles, np.float64)
    area_vectors = np.cross(
        triangles[:, 1]-triangles[:, 0],
        triangles[:, 2]-triangles[:, 0],
    )
    normals = area_vectors/np.linalg.norm(area_vectors, axis=1)[:, None]
    return NativeOpticalComponentScene(
        component_key=compiled.key,
        frequencies_hz=frequencies,
        triangles=triangles,
        normals=np.ascontiguousarray(normals, np.float64),
        material_indices=np.ascontiguousarray(mat_idx, np.int32),
        material_buffer=np.ascontiguousarray(mat_buf, np.float32),
        material_count=int(mat_count),
        roles=tuple(geometry.roles),
        role_interactions=role_interactions,
    )


@dataclass(frozen=True)
class PhysicalApertureComponent:
    aperture: LivePhysicalAperture
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    field_margin: float = 1.25
    longitudinal_steps: int = 4

    @property
    def key(self) -> str:
        return self.aperture.key

    def compile(self, lane_count: int) -> CompiledOpticalComponent:
        self.aperture.validate()
        axis = _unit(self.axis, "aperture axis")
        extent = self.aperture.assembly_radius_m*self.field_margin
        step = self.aperture.thickness_m/max(1, int(self.longitudinal_steps))
        nodes, links = wave_context_nodes(
            self.key,
            lane_count,
            propagation=WavePropagationStyle.ANGULAR_SPECTRUM_FFT,
            boundary=WaveBoundaryStyle.PADDED_ABSORBING,
            center_m=self.center_m,
            axis=axis,
            radius_m=extent,
            longitudinal_step_m=step,
            longitudinal_steps=max(1, int(self.longitudinal_steps)),
            aperture_material=self.aperture,
        )
        graph = compile_optical_graph(OpticalTransportGraphSpec(
            nodes=nodes,
            links=links,
            entry_keys=(nodes[0].key,),
            product_keys=(nodes[-1].key,),
        ))
        transport = None
        try:
            triangles, _normals = self.aperture.triangle_mesh(
                z_center_m=float(self.center_m[2])
            )
            transport = OpticalTriangleAssembly(
                triangles.reshape(-1, 3, 3),
                ("aperture_material",)*(triangles.shape[0]),
            )
        except NotImplementedError:
            pass
        component = CompiledOpticalComponent(
            key=self.key,
            component_kind="physical-aperture",
            lane_count=lane_count,
            graph=graph,
            ports=(
                _port(
                    f"{self.key}.input", OpticalPortDirection.INPUT,
                    OpticalRepresentation.COMPLEX_RAY, self.center_m, -axis,
                ),
                _port(
                    f"{self.key}.output", OpticalPortDirection.OUTPUT,
                    OpticalRepresentation.COMPLEX_RAY, self.center_m, axis,
                ),
            ),
            controls=(
                OpticalControlSpec(
                    "opening_x_m", "Opening", "float",
                    self.aperture.opening_x_m, 0.0,
                    self.aperture.assembly_radius_m, "m", "t4-state",
                ),
                OpticalControlSpec(
                    "rotation_rad", "Rotation", "float",
                    self.aperture.rotation_rad, -math.pi, math.pi,
                    "rad", "t4-state",
                ),
            ),
            transport_geometry=transport,
            display_geometry=transport,
            material_roles=(
                OpticalMaterialRole(
                    "aperture_material", self.aperture.material_name,
                    "finite-complex-index-wave-volume",
                ),
            ) if transport is not None else (),
            metadata={
                "aperture": self.aperture.graph_parameters(),
                "wave_localization": "material-bounds-only",
            },
        )
        component.validate()
        return component


@dataclass(frozen=True)
class CompoundLensComponent:
    lens: CompoundLens
    key: str = "lens.default-camera"

    def compile(self, lane_count: int) -> CompiledOpticalComponent:
        graph = compile_compound_lens_graph(self.lens, lane_count=lane_count)
        front = self.lens.side("front")
        back = self.lens.side("back")
        component = CompiledOpticalComponent(
            key=self.key,
            component_kind="exact-compound-lens",
            lane_count=lane_count,
            graph=graph,
            ports=(
                _port(
                    f"{self.key}.front", OpticalPortDirection.BIDIRECTIONAL,
                    OpticalRepresentation.COMPLEX_RAY,
                    (front.x_pos, 0.0, 0.0), (front.axis_sign, 0.0, 0.0),
                ),
                _port(
                    f"{self.key}.back", OpticalPortDirection.BIDIRECTIONAL,
                    OpticalRepresentation.COMPLEX_RAY,
                    (back.x_pos, 0.0, 0.0), (back.axis_sign, 0.0, 0.0),
                ),
            ),
            controls=(
                OpticalControlSpec(
                    "focus_extension_m", "Focus extension", "float",
                    0.0, -0.1, 0.5, "m", "component-pose",
                ),
            ),
            metadata={
                "surface_count": len(self.lens.registered_faces()),
                "execution": "existing-fused-exact-t2",
                "hot_interpreter": False,
            },
        )
        component.validate()
        return component


@dataclass(frozen=True)
class PlaneMirrorComponent:
    key: str = "mirror.plane"
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    radius_m: float = 0.025
    material_name: str = "aluminum_mirror"
    display_segments: int = 64

    def reflected_direction(self, direction: Sequence[float]) -> np.ndarray:
        incoming = _unit(direction, "mirror incoming direction")
        normal = _unit(self.normal, "mirror normal")
        return incoming - 2.0*float(np.dot(incoming, normal))*normal

    def _geometry(self) -> OpticalTriangleAssembly:
        if self.radius_m <= 0.0 or self.display_segments < 8:
            raise ValueError("mirror radius and display segment count are invalid")
        normal = _unit(self.normal, "mirror normal")
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(reference, normal))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        tangent_u = _unit(np.cross(normal, reference), "mirror tangent")
        tangent_v = _unit(np.cross(normal, tangent_u), "mirror tangent")
        center = np.asarray(self.center_m, np.float64)
        angles = np.linspace(
            0.0, 2.0*math.pi, self.display_segments, endpoint=False
        )
        rim = [
            center+self.radius_m*(math.cos(a)*tangent_u+math.sin(a)*tangent_v)
            for a in angles
        ]
        triangles = np.stack([
            np.stack((center, rim[index], rim[(index+1) % len(rim)]))
            for index in range(len(rim))
        ])
        return OpticalTriangleAssembly(
            triangles, ("reflective_surface",)*len(triangles)
        )

    def compile(self, lane_count: int) -> CompiledOpticalComponent:
        normal = _unit(self.normal, "mirror normal")
        graph = _compile_linear_material_graph(
            prefix=self.key,
            lane_count=lane_count,
            operations=((
                "surface", "analytic-plane-reflection", {
                    "center_m": tuple(float(v) for v in self.center_m),
                    "normal": tuple(float(v) for v in normal),
                    "radius_m": float(self.radius_m),
                    "material": self.material_name,
                    "jones_response": "canonical-material-conductor",
                },
            ),),
        )
        geometry = self._geometry()
        component = CompiledOpticalComponent(
            key=self.key,
            component_kind="analytic-plane-mirror",
            lane_count=lane_count,
            graph=graph,
            ports=(
                _port(
                    f"{self.key}.incident", OpticalPortDirection.INPUT,
                    OpticalRepresentation.COMPLEX_RAY, self.center_m, -normal,
                ),
                _port(
                    f"{self.key}.reflected", OpticalPortDirection.OUTPUT,
                    OpticalRepresentation.COMPLEX_RAY, self.center_m, normal,
                ),
            ),
            controls=(
                OpticalControlSpec(
                    "tilt_x_rad", "Mirror tilt X", "float",
                    0.0, -math.pi, math.pi, "rad", "component-pose",
                ),
                OpticalControlSpec(
                    "tilt_y_rad", "Mirror tilt Y", "float",
                    0.0, -math.pi, math.pi, "rad", "component-pose",
                ),
            ),
            transport_geometry=geometry,
            display_geometry=geometry,
            material_roles=(
                OpticalMaterialRole(
                    "reflective_surface", self.material_name,
                    "analytic-specular-reflection",
                ),
            ),
            metadata={"surface": "plane", "ideal_reflector": False},
        )
        component.validate()
        return component


@dataclass(frozen=True)
class PentaprismComponent:
    spec: PentaprismSpec
    key: str = "pentaprism.finder"

    def compile(self, lane_count: int) -> CompiledOpticalComponent:
        assembly = self.spec.build()
        operations = (
            ("entrance", "dielectric-interface", {
                "role": "entrance_glass", "material": "BK7",
            }),
            ("reflector-1", "analytic-plane-reflection", {
                "role": "silvered_reflector_1",
                "material": "aluminum_mirror",
            }),
            ("reflector-2", "analytic-plane-reflection", {
                "role": "silvered_reflector_2",
                "material": "aluminum_mirror",
            }),
            ("exit", "dielectric-interface", {
                "role": "exit_glass", "material": "BK7",
            }),
        )
        graph = _compile_linear_material_graph(
            prefix=self.key, lane_count=lane_count, operations=operations
        )
        roles = (
            OpticalMaterialRole(
                "entrance_glass", "BK7", "dielectric-interface"
            ),
            OpticalMaterialRole(
                "exit_glass", "BK7", "dielectric-interface"
            ),
            OpticalMaterialRole(
                "silvered_reflector_1", "aluminum_mirror",
                "analytic-specular-reflection",
            ),
            OpticalMaterialRole(
                "silvered_reflector_2", "aluminum_mirror",
                "analytic-specular-reflection",
            ),
            OpticalMaterialRole(
                "blackened_prism_face", "blackened_steel", "absorber"
            ),
            OpticalMaterialRole(
                "blackened_prism_side", "blackened_steel", "absorber"
            ),
        )
        component = CompiledOpticalComponent(
            key=self.key,
            component_kind="composite-pentaprism",
            lane_count=lane_count,
            graph=graph,
            ports=(
                _port(
                    f"{self.key}.input", OpticalPortDirection.INPUT,
                    OpticalRepresentation.COMPLEX_RAY,
                    assembly.primary_path[0], self.spec.input_axis,
                ),
                _port(
                    f"{self.key}.output", OpticalPortDirection.OUTPUT,
                    OpticalRepresentation.COMPLEX_RAY,
                    assembly.primary_path[-1], self.spec.output_axis,
                ),
            ),
            controls=(
                OpticalControlSpec(
                    "rotation_rad", "Assembly rotation", "float",
                    0.0, -math.pi, math.pi, "rad", "component-pose",
                ),
            ),
            transport_geometry=assembly,
            display_geometry=assembly,
            material_roles=roles,
            metadata={
                "composite": True,
                "chief_path": [
                    np.asarray(point, np.float64).tolist()
                    for point in assembly.primary_path
                ],
                "constant_deviation_rad": 0.5*math.pi,
            },
        )
        component.validate()
        return component


class OpticalComponentRegistry:
    """Cold component factory registry; manifests select keys, not classes."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[int], OpticalComponent]] = {}

    def register(
        self, key: str, factory: Callable[[int], OpticalComponent]
    ) -> None:
        canonical = str(key).strip()
        if not canonical:
            raise ValueError("component registry key must be non-empty")
        if canonical in self._factories:
            raise ValueError(f"optical component {canonical!r} already registered")
        self._factories[canonical] = factory

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def create(self, key: str, lane_count: int = 1) -> OpticalComponent:
        try:
            factory = self._factories[str(key)]
        except KeyError as exc:
            raise KeyError(
                f"unknown optical component {key!r}; available={self.keys()}"
            ) from exc
        return factory(int(lane_count))

    def compile(self, key: str, lane_count: int) -> CompiledOpticalComponent:
        return self.create(key, lane_count).compile(int(lane_count))


def default_optical_component_registry() -> OpticalComponentRegistry:
    registry = OpticalComponentRegistry()
    registry.register(
        "aperture.iris",
        lambda _lanes: PhysicalApertureComponent(LivePhysicalAperture.iris(
            "aperture.iris",
            blade_count=9,
            opening_radius_m=18.0e-6,
            assembly_radius_m=52.0e-6,
            thickness_m=3.0e-6,
        )),
    )

    def default_lens(lanes: int) -> CompoundLensComponent:
        from camera_designer.camera_preset import simple_doublet_preset

        return CompoundLensComponent(CompoundLens.from_preset(
            simple_doublet_preset(),
            wavelengths_um=np.linspace(0.42, 0.70, int(lanes)),
        ))

    registry.register("lens.default-camera", default_lens)
    registry.register("mirror.plane", lambda _lanes: PlaneMirrorComponent())
    registry.register(
        "pentaprism.finder",
        lambda _lanes: PentaprismComponent(
            PentaprismSpec((0.0, 0.0, 0.0))
        ),
    )
    return registry


__all__ = [
    "OPTICAL_COMPONENT_SCHEMA",
    "OpticalPortDirection",
    "OpticalPortSpec",
    "OpticalControlSpec",
    "OpticalMaterialRole",
    "CompiledOpticalComponent",
    "NativeOpticalComponentScene",
    "build_native_component_scene",
    "OpticalComponent",
    "PhysicalApertureComponent",
    "CompoundLensComponent",
    "PlaneMirrorComponent",
    "PentaprismComponent",
    "OpticalComponentRegistry",
    "default_optical_component_registry",
]
