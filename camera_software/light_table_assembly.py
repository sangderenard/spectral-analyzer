"""Deterministic mechanical assembly for emitter-to-sensor light tables.

The mechanical graph owns placement.  Holders resolve component poses from
rail/tube stations and mounting datums; the resulting ordered components are
then lowered into the optical transport graph.  This keeps visible apparatus,
collision/fit decisions, and transport coordinates on one source of truth.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any

import numpy as np

from .optical_chain import (
    EmitterEndpointComponent,
    OpticalChainElement,
    OpticalChainSpec,
)
from .optical_components import (
    CompoundLensComponent,
    OpticalComponent,
    PhysicalApertureComponent,
    PlaneMirrorComponent,
)
from .optical_chain import SensorEndpointComponent


LIGHT_TABLE_ASSEMBLY_SCHEMA = "light-table-mechanical-assembly-v1"


class MountFaceShape(str, Enum):
    CIRCULAR = "circular"
    SQUARE = "square"
    RECTANGULAR = "rectangular"


class HolderKind(str, Enum):
    POST_STAND = "post-stand"
    RAIL_CARRIAGE = "rail-carriage"
    TUBE_CELL = "tube-cell"
    CAGE_PLATE = "cage-plate"
    SENSOR_BACK = "sensor-back"
    EMITTER_PANEL = "emitter-panel"


class MountDatum(str, Enum):
    COMPONENT_CENTER = "component-center"
    FRONT_FACE = "front-face"
    BACK_FACE = "back-face"
    APERTURE_STOP = "aperture-stop"
    EMITTER_PLANE = "emitter-plane"
    SENSOR_PLANE = "sensor-plane"


def _positive_pair(
    value: tuple[float, float], name: str
) -> tuple[float, float]:
    result = tuple(float(item) for item in value)
    if (
        len(result) != 2
        or not all(math.isfinite(item) and item > 0.0 for item in result)
    ):
        raise ValueError(f"{name} must contain two finite positive dimensions")
    return result


def _unit_axis(value: tuple[float, float, float]) -> np.ndarray:
    result = np.asarray(value, np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("holder optical axis must be a finite 3-vector")
    length = float(np.linalg.norm(result))
    if length <= 1.0e-12:
        raise ValueError("holder optical axis must be non-zero")
    return result / length


@dataclass(frozen=True)
class MountInterfaceSpec:
    """Mechanical mating and clear-opening contract for one optic or holder."""

    standard: str
    shape: MountFaceShape
    outer_size_m: tuple[float, float]
    clear_size_m: tuple[float, float]
    clocking: str = "continuous"

    def validate(self) -> None:
        if not self.standard.strip():
            raise ValueError("mount interfaces require a named standard")
        outer = _positive_pair(self.outer_size_m, "mount outer_size_m")
        clear = _positive_pair(self.clear_size_m, "mount clear_size_m")
        if clear[0] > outer[0] or clear[1] > outer[1]:
            raise ValueError("mount clear opening cannot exceed its outer size")
        if not self.clocking.strip():
            raise ValueError("mount interfaces require a clocking policy")

    def accepts(self, optic: "MountInterfaceSpec") -> bool:
        self.validate()
        optic.validate()
        tolerance = 1.0e-9
        return (
            self.standard == optic.standard
            and self.shape is optic.shape
            and all(
                abs(a-b) <= tolerance
                for a, b in zip(self.outer_size_m, optic.outer_size_m)
            )
            and (
                self.clocking == "continuous"
                or self.clocking == optic.clocking
            )
        )


@dataclass(frozen=True)
class LightTableHolderSpec:
    """A physical stand, carriage, cell, or plate with one optical receiver."""

    key: str
    kind: HolderKind
    receiver: MountInterfaceSpec
    station_m: float
    transverse_m: tuple[float, float] = (0.0, 0.0)
    optical_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    support_group: str = "main-rail"
    sequence: int = 0

    def validate(self) -> None:
        if not self.key.strip() or not self.support_group.strip():
            raise ValueError("holders require keys and support groups")
        self.receiver.validate()
        if not math.isfinite(float(self.station_m)):
            raise ValueError("holder station must be finite")
        transverse = np.asarray(self.transverse_m, np.float64)
        if transverse.shape != (2,) or not np.all(np.isfinite(transverse)):
            raise ValueError("holder transverse position must be finite")
        _unit_axis(self.optical_axis)

    @property
    def center_m(self) -> tuple[float, float, float]:
        return (
            float(self.station_m),
            float(self.transverse_m[0]),
            float(self.transverse_m[1]),
        )


@dataclass(frozen=True)
class MountedOpticalElement:
    instance_key: str
    component: OpticalComponent
    holder: LightTableHolderSpec
    optic_mount: MountInterfaceSpec
    datum: MountDatum = MountDatum.COMPONENT_CENTER
    engine: str = "auto"

    def validate(self) -> None:
        if not self.instance_key.strip():
            raise ValueError("mounted optical instances require keys")
        self.holder.validate()
        self.optic_mount.validate()
        if not self.holder.receiver.accepts(self.optic_mount):
            raise ValueError(
                f"optic {self.instance_key!r} mount "
                f"{self.optic_mount.standard!r} does not fit holder "
                f"{self.holder.key!r} receiver "
                f"{self.holder.receiver.standard!r}"
            )


@dataclass(frozen=True)
class ResolvedLightTableAssembly:
    schema: str
    key: str
    chain: OpticalChainSpec
    mounted_elements: tuple[MountedOpticalElement, ...]
    apparatus_manifest: dict[str, Any]
    identity: str

    def scene_manifest(self) -> dict[str, Any]:
        """Return the authoritative apparatus input for BellJarWorkspace."""

        return {
            "schema_version": 1,
            "scene_kind": "mounted_optical_light_table",
            "units": "metres",
            "assembly_identity": self.identity,
            "mechanical_assembly": copy.deepcopy(self.apparatus_manifest),
        }


@dataclass(frozen=True)
class LightTableAssemblySpec:
    key: str
    lane_count: int
    elements: tuple[MountedOpticalElement, ...]
    schema: str = LIGHT_TABLE_ASSEMBLY_SCHEMA

    def validate(self) -> None:
        if self.schema != LIGHT_TABLE_ASSEMBLY_SCHEMA:
            raise ValueError(f"unsupported assembly schema {self.schema!r}")
        if not self.key.strip() or int(self.lane_count) < 1:
            raise ValueError("assembly requires a key and positive lane count")
        if len(self.elements) < 2:
            raise ValueError("assembly requires at least source and sink")
        for element in self.elements:
            element.validate()
        instance_keys = [value.instance_key for value in self.elements]
        holder_keys = [value.holder.key for value in self.elements]
        if len(set(instance_keys)) != len(instance_keys):
            raise ValueError("mounted optical instance keys must be unique")
        if len(set(holder_keys)) != len(holder_keys):
            raise ValueError("each mounted optic requires a distinct holder")

    def replace_mounted_component(
        self,
        instance_key: str,
        component: OpticalComponent,
        optic_mount: MountInterfaceSpec | None = None,
    ) -> "LightTableAssemblySpec":
        """Swap an optic while retaining and rechecking its physical holder."""

        found = False
        values = []
        for element in self.elements:
            if element.instance_key == instance_key:
                found = True
                values.append(replace(
                    element,
                    component=component,
                    optic_mount=(
                        element.optic_mount
                        if optic_mount is None
                        else optic_mount
                    ),
                ))
            else:
                values.append(element)
        if not found:
            raise KeyError(f"assembly has no mounted optic {instance_key!r}")
        result = replace(self, elements=tuple(values))
        result.validate()
        return result

    def resolve(self) -> ResolvedLightTableAssembly:
        """Resolve holder constraints and lower the assembly to an optical chain."""

        self.validate()
        ordered = tuple(sorted(
            self.elements,
            key=lambda value: (
                float(value.holder.station_m),
                int(value.holder.sequence),
                value.holder.key,
            ),
        ))
        resolved_components = tuple(
            _resolve_component_pose(value) for value in ordered
        )
        compiled_components = tuple(
            component.compile(int(self.lane_count), mounted.engine)
            for mounted, component in zip(ordered, resolved_components)
        )
        chain = OpticalChainSpec(
            key=self.key,
            lane_count=int(self.lane_count),
            elements=tuple(
                OpticalChainElement(
                    mounted.instance_key,
                    component,
                    mounted.engine,
                )
                for mounted, component in zip(ordered, resolved_components)
            ),
        )
        chain.validate()
        manifest = {
            "schema": self.schema,
            "key": self.key,
            "units": "metres",
            "ordering": "station-sequence-holder-key",
            "apparatus": [
                {
                    "instance_key": mounted.instance_key,
                    "holder": {
                        **asdict(mounted.holder),
                        "kind": mounted.holder.kind.value,
                        "receiver": {
                            **asdict(mounted.holder.receiver),
                            "shape": mounted.holder.receiver.shape.value,
                        },
                    },
                    "optic_mount": {
                        **asdict(mounted.optic_mount),
                        "shape": mounted.optic_mount.shape.value,
                    },
                    "datum": mounted.datum.value,
                    "resolved_component_key": str(
                        getattr(component, "key", mounted.instance_key)
                    ),
                    "component_revision": {
                        "contract": compiled.contract(),
                        "t2_payload_sha256": {
                            key: hashlib.sha256(
                                np.ascontiguousarray(payload).tobytes()
                            ).hexdigest()
                            for key, payload
                            in compiled.graph.t2_payloads.items()
                        },
                        "t4_descriptors": copy.deepcopy(
                            compiled.graph.t4_descriptors
                        ),
                    },
                }
                for mounted, component, compiled in zip(
                    ordered, resolved_components, compiled_components
                )
            ],
        }
        encoded = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return ResolvedLightTableAssembly(
            schema=self.schema,
            key=self.key,
            chain=chain,
            mounted_elements=ordered,
            apparatus_manifest=manifest,
            identity=hashlib.sha256(encoded).hexdigest(),
        )


def _resolve_component_pose(element: MountedOpticalElement) -> OpticalComponent:
    center = np.asarray(element.holder.center_m, np.float64)
    axis = _unit_axis(element.holder.optical_axis)
    component = element.component

    if isinstance(component, EmitterEndpointComponent):
        if element.datum not in {
            MountDatum.COMPONENT_CENTER, MountDatum.EMITTER_PLANE,
        }:
            raise ValueError("emitter holders resolve the emitter plane datum")
        emitter = replace(
            component.emitter,
            pos=tuple(float(value) for value in center),
            normal=tuple(float(value) for value in axis),
        )
        return replace(component, emitter=emitter, axis=tuple(axis))

    if isinstance(component, PhysicalApertureComponent):
        if element.datum not in {
            MountDatum.COMPONENT_CENTER, MountDatum.APERTURE_STOP,
        }:
            raise ValueError("aperture holders resolve center or stop datums")
        return replace(
            component,
            center_m=tuple(float(value) for value in center),
            axis=tuple(float(value) for value in axis),
        )

    if isinstance(component, PlaneMirrorComponent):
        if element.datum is not MountDatum.COMPONENT_CENTER:
            raise ValueError("mirror holders resolve the component-center datum")
        # Normal-incidence retroreflector: the incident port faces back along
        # the holder's optical axis (matching the component's own default
        # normal=(-1,0,0) for a default axis=(1,0,0) holder), so the reflected
        # beam returns along the same axis. Off-axis fold mirrors are a later
        # mounting datum, not this simple retroreflecting case.
        return replace(
            component,
            center_m=tuple(float(value) for value in center),
            normal=tuple(float(-value) for value in axis),
        )

    if isinstance(component, SensorEndpointComponent):
        if element.datum not in {
            MountDatum.COMPONENT_CENTER, MountDatum.SENSOR_PLANE,
        }:
            raise ValueError("sensor holders resolve the sensor plane datum")
        return replace(
            component,
            center_m=tuple(float(value) for value in center),
            axis=tuple(float(-value) for value in axis),
        )

    if isinstance(component, CompoundLensComponent):
        if not np.allclose(axis, (1.0, 0.0, 0.0), atol=1.0e-12):
            raise NotImplementedError(
                "exact CompoundLens mounting currently supports +X rail "
                "alignment only"
            )
        if not np.allclose(center[1:], (0.0, 0.0), atol=1.0e-12):
            raise NotImplementedError(
                "exact CompoundLens mounting currently supports the main "
                "rail centerline only"
            )
        lens = copy.deepcopy(component.lens)
        if element.datum is MountDatum.FRONT_FACE:
            datum_x = float(lens.side("front").x_pos)
        elif element.datum is MountDatum.BACK_FACE:
            datum_x = float(lens.side("back").x_pos)
        elif element.datum is MountDatum.APERTURE_STOP:
            stops = [
                value for value in lens.elements
                if value.__class__.__name__ == "ApertureStop"
            ]
            if not stops:
                raise ValueError("compound lens has no aperture-stop datum")
            datum_x = float(stops[0].x_pos)
        elif element.datum is MountDatum.COMPONENT_CENTER:
            datum_x = 0.5 * (
                float(lens.side("front").x_pos)
                + float(lens.side("back").x_pos)
            )
        else:
            raise ValueError("unsupported compound-lens mounting datum")
        lens.translate_elements(
            tuple(range(len(lens.elements))),
            float(center[0])-datum_x,
        )
        if lens.hood is not None:
            delta = float(center[0])-datum_x
            for name in ("x_tip", "x_rim"):
                if hasattr(lens.hood, name):
                    setattr(lens.hood, name, float(getattr(lens.hood, name))+delta)
        return replace(component, lens=lens)

    raise TypeError(
        f"component type {type(component).__name__} has no mechanical pose "
        "resolver"
    )


def default_circular_mount(
    standard: str,
    *,
    outer_diameter_m: float,
    clear_diameter_m: float,
    clocking: str = "continuous",
) -> MountInterfaceSpec:
    return MountInterfaceSpec(
        standard=str(standard),
        shape=MountFaceShape.CIRCULAR,
        outer_size_m=(float(outer_diameter_m),)*2,
        clear_size_m=(float(clear_diameter_m),)*2,
        clocking=str(clocking),
    )


def default_square_mount(
    standard: str,
    *,
    outer_side_m: float,
    clear_side_m: float,
    clocking: str = "fourfold",
) -> MountInterfaceSpec:
    return MountInterfaceSpec(
        standard=str(standard),
        shape=MountFaceShape.SQUARE,
        outer_size_m=(float(outer_side_m),)*2,
        clear_size_m=(float(clear_side_m),)*2,
        clocking=str(clocking),
    )


__all__ = [
    "LIGHT_TABLE_ASSEMBLY_SCHEMA",
    "MountFaceShape",
    "HolderKind",
    "MountDatum",
    "MountInterfaceSpec",
    "LightTableHolderSpec",
    "MountedOpticalElement",
    "LightTableAssemblySpec",
    "ResolvedLightTableAssembly",
    "default_circular_mount",
    "default_square_mount",
]
