"""Electrical realization and coverage for production machine graphs.

Runtime realization consumes explicit electrical declarations only.  The
audit also recognizes legacy wire-shaped records so omissions are visible,
but those heuristics never become circuit physics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from electrical_tensor_network import (
    ElectricalDeviceRegistry,
    EmpiricalPortLaw,
    GROUND,
    PortLinearization,
    impedance_port_law,
)


COPPER_RESISTIVITY_OHM_M = 1.724e-8


def _legacy_electrical_edge(edge: Mapping) -> bool:
    constraint = str(edge.get("constraint", "")).lower()
    circuit = str(edge.get("circuit_identity", "")).lower()
    fluid = str(edge.get("fluid", "")).lower()
    return (
        "wire" in constraint
        or "electrical" in circuit
        or "dc-power" in circuit
        or circuit.endswith(("12v", "24v", "48v", "hv"))
        or fluid == "electricity"
    )


@dataclass(frozen=True)
class ElectricalGraphCoverage:
    declared_conductors: tuple[str, ...]
    undeclared_legacy_edges: tuple[str, ...]
    incomplete_conductors: tuple[str, ...]
    declared_devices: tuple[str, ...]
    devices_without_laws: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not (
            self.undeclared_legacy_edges
            or self.incomplete_conductors
            or self.devices_without_laws
        )


def audit_machine_electrical_graph(
    graph: Mapping,
    *,
    device_laws: Mapping[str, object] | None = None,
) -> ElectricalGraphCoverage:
    device_laws = device_laws or {}
    declared = []
    legacy = []
    incomplete = []
    for edge in graph.get("edges", ()):
        identity = str(edge.get("identity", ""))
        if edge.get("transport_domain") == "electrical":
            declared.append(identity)
            has_direct = "resistance_ohm" in edge
            has_geometry = all(key in edge for key in (
                "conductor_area_mm2", "rest_length"))
            conductors = edge.get("conductors", ())
            has_bundle = bool(conductors) and all(
                row.get("role") and "resistance_ohm" in row
                for row in conductors)
            if not (has_direct or has_geometry or has_bundle):
                incomplete.append(identity)
        elif _legacy_electrical_edge(edge):
            legacy.append(identity)
    devices = tuple(sorted(
        str(node["identity"])
        for node in graph.get("nodes", ())
        if node.get("electrical_port_model")
    ))
    return ElectricalGraphCoverage(
        tuple(sorted(declared)),
        tuple(sorted(legacy)),
        tuple(sorted(incomplete)),
        devices,
        tuple(identity for identity in devices if identity not in device_laws),
    )


def _conductor_impedance(edge: Mapping, frequencies_hz: torch.Tensor,
                         temperature_k: float = 293.15):
    if "resistance_ohm_at_20c" in edge:
        coefficient = float(edge.get("temperature_coefficient_per_k", 0.00393))
        resistance = float(edge["resistance_ohm_at_20c"]) * (
            1.0 + coefficient * (float(temperature_k) - 293.15))
    elif "resistance_ohm" in edge:
        resistance = float(edge["resistance_ohm"])
    else:
        area_m2 = float(edge["conductor_area_mm2"]) * 1.0e-6
        length_m = float(edge["rest_length"])
        multiplier = float(edge.get("path_length_multiplier", 1.0))
        resistivity = float(edge.get(
            "conductor_resistivity_ohm_m", COPPER_RESISTIVITY_OHM_M))
        resistance = resistivity * length_m * multiplier / area_m2
    omega = 2.0 * math.pi * frequencies_hz.to(torch.float64)
    impedance = torch.full_like(
        omega, complex(resistance, 0.0), dtype=torch.complex128)
    inductance = float(edge.get("series_inductance_h", 0.0))
    if inductance:
        impedance += 1j * omega * inductance
    capacitance = float(edge.get("series_capacitance_f", 0.0))
    if capacitance:
        moving = omega != 0.0
        impedance[~moving] = complex(float("inf"), 0.0)
        impedance[moving] += 1.0 / (1j * omega[moving] * capacitance)
    return impedance


def _temperature_sensitive_law(
    registry: ElectricalDeviceRegistry,
    identity: str,
    a: str,
    b: str,
    declaration: Mapping,
    thermal_target: str,
) -> EmpiricalPortLaw:
    def linearize(_voltage_ab, frequencies_hz):
        temperature_k = registry.thermal_temperature_k.get(
            thermal_target, 293.15)
        impedance = _conductor_impedance(
            declaration, frequencies_hz, temperature_k)
        if torch.any(torch.abs(impedance) == 0.0):
            raise ValueError(f"{identity}: zero impedance")
        return PortLinearization(1.0 / impedance)
    return EmpiricalPortLaw(identity, a, b, linearize)


def register_machine_electrical_graph(
    registry: ElectricalDeviceRegistry,
    graph: Mapping,
    *,
    device_laws: Mapping[str, object],
) -> ElectricalGraphCoverage:
    coverage = audit_machine_electrical_graph(
        graph, device_laws=device_laws)
    if not coverage.complete:
        raise ValueError(f"incomplete electrical graph: {coverage}")
    for edge in graph.get("edges", ()):
        if edge.get("transport_domain") != "electrical":
            continue
        conductors = edge.get("conductors", ())
        if conductors:
            service = str(edge.get(
                "electrical_service", edge.get("circuit_identity", "electrical")))
            junction_by_endpoint = {
                str(row["endpoint"]): row for row in edge.get("junctions", ())
            }
            for conductor in conductors:
                role = str(conductor["role"])
                service_role = str(conductor.get("service_role", role))
                identity = f"{edge['identity']}::{role}"
                a_bus = f"{edge['a']}::{service}::{service_role}"
                b_bus = f"{edge['b']}::{service}::{service_role}"
                if junction_by_endpoint:
                    a_wire = f"{identity}::terminal:a"
                    b_wire = f"{identity}::terminal:b"
                else:
                    a_wire, b_wire = a_bus, b_bus
                registry.register(_temperature_sensitive_law(
                    registry, identity, a_wire, b_wire, conductor, identity))
                registry.thermal_target_by_branch[identity] = identity
                registry.thermal_fraction_by_branch[identity] = 1.0
                registry.safety_by_branch[identity] = {
                    "kind": "conductor",
                    "ampacity_a": float(conductor.get("ampacity_a", 0.0)),
                    "continuous_load_limit_a": float(edge.get(
                        "continuous_load_limit_per_conductor_a",
                        edge.get("continuous_load_limit_a",
                                 conductor.get("ampacity_a", 0.0)))),
                    "thermal_target": identity,
                }
                for endpoint, bus, wire, part in (
                    ("a", a_bus, a_wire, str(edge["a"])),
                    ("b", b_bus, b_wire, str(edge["b"])),
                ):
                    junction = junction_by_endpoint.get(endpoint)
                    if junction is None:
                        continue
                    junction_identity = f"{identity}::junction:{endpoint}"
                    registry.register(_temperature_sensitive_law(
                        registry, junction_identity, bus, wire,
                        junction, part))
                    registry.thermal_target_by_branch[junction_identity] = part
                    registry.thermal_fraction_by_branch[junction_identity] = 1.0
                    registry.safety_by_branch[junction_identity] = {
                        "kind": "junction",
                        "ampacity_a": float(conductor.get("ampacity_a", 0.0)),
                        "continuous_load_limit_a": float(edge.get(
                            "continuous_load_limit_per_conductor_a",
                            edge.get("continuous_load_limit_a",
                                     conductor.get("ampacity_a", 0.0)))),
                        "temperature_rating_c": float(junction.get(
                            "temperature_rating_c", 75.0)),
                        "torque_verified": bool(junction.get(
                            "torque_verified", False)),
                        "connector_listing": str(junction.get(
                            "connector_listing", "")),
                        "thermal_target": part,
                    }
        else:
            identity = str(edge["identity"])
            registry.register(_temperature_sensitive_law(
                registry, identity, str(edge["a"]), str(edge["b"]),
                edge, identity))
            registry.thermal_target_by_branch[identity] = identity
            registry.thermal_fraction_by_branch[identity] = 1.0
    for identity in coverage.declared_devices:
        registry.register(device_laws[identity])
    for node in graph.get("nodes", ()):
        roles = tuple(node.get("grounded_conductor_roles", ()))
        if not roles and node.get("grounded_conductor_role") is not None:
            roles = (str(node["grounded_conductor_role"]),)
        if not roles:
            continue
        service = str(node.get("electrical_service", ""))
        resistance = float(node.get("grounding_bond_resistance_ohm", 0.0))
        if not service or resistance <= 0.0:
            raise ValueError(
                f"{node.get('identity')}: a grounding bond needs an "
                "electrical service and positive resistance")
        for role in roles:
            identity = f"{node['identity']}.grounding_bond.{role}"
            terminal = f"{node['identity']}::{service}::{role}"
            registry.register(impedance_port_law(
                identity, terminal, GROUND,
                [resistance] * registry.lanes))
            registry.thermal_target_by_branch[identity] = str(node["identity"])
            registry.thermal_fraction_by_branch[identity] = 1.0
    return coverage


__all__ = [
    "ElectricalGraphCoverage",
    "audit_machine_electrical_graph",
    "register_machine_electrical_graph",
]
