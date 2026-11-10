from __future__ import annotations

from pathlib import Path
import sys

import torch

from electrical_tensor_network import (
    GROUND,
    ComplexTensorCircuit,
    ElectricalDeviceRegistry,
    EmpiricalCircuitLaw,
    ImpedanceBranch,
    MultiportLinearization,
    TheveninSource,
    bake_linear_circuit,
    battery_port_law,
    impedance_port_law,
    transmission_line_circuit_law,
)
from machine_electrical_graph import (
    audit_machine_electrical_graph,
    register_machine_electrical_graph,
)

TURING_ROOT = Path(__file__).resolve().parents[2] / "turing"
ENGINE_TOY_ROOT = Path(__file__).resolve().parents[2] / "engine_toy"
if str(TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(TURING_ROOT))
if str(ENGINE_TOY_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_TOY_ROOT))

from dc_power import DCBattery
from electrical_distribution import BuildingCable, standard_services
from thermal_domains import build_thermal_assembly
from electrical_dt_engine import ComplexElectricalEngine
from src.common.dt_system.state_table import StateTable
from src.common.dt_system.time_contracts import HOLD


def test_complex_tensor_circuit_matches_direct_nodal_solve() -> None:
    source_v = torch.tensor([48.0 + 0.0j, -24.0 - 41.569219381653056j,
                             -24.0 + 41.569219381653056j], dtype=torch.complex128)
    source_z = torch.tensor([0.08 + 0.02j] * 3, dtype=torch.complex128)
    line_z = torch.tensor([0.04 + 0.01j] * 3, dtype=torch.complex128)
    motor_z = torch.tensor([3.6 + 1.8j] * 3, dtype=torch.complex128)
    circuit = ComplexTensorCircuit(
        lanes=3,
        sources=[TheveninSource("generator", "bus", source_v, source_z)],
        branches=[
            ImpedanceBranch("feeder", "bus", "motor", line_z),
            ImpedanceBranch("motor-winding", "motor", GROUND, motor_z),
        ],
    )

    reading = circuit.solve()
    expected_i = source_v / (source_z + line_z + motor_z)
    expected_motor_v = expected_i * motor_z

    assert torch.allclose(reading.branch_current_a["feeder"], expected_i)
    assert torch.allclose(reading.voltage_v["motor"], expected_motor_v)
    assert torch.allclose(
        reading.complex_power_va("motor", expected_i),
        expected_motor_v * torch.conj(expected_i),
    )


def test_registry_can_install_compiled_llvm_nodal_solver(tmp_path) -> None:
    registry = ElectricalDeviceRegistry([0.0, 60.0])
    registry.register_source(TheveninSource(
        "source", "bus", [24.0 + 0.0j, 1.0 + 2.0j],
        [0.2 + 0.1j, 0.3 + 0.2j],
    ))
    registry.register(impedance_port_law(
        "load", "bus", GROUND, [4.0 + 1.0j, 6.0 - 2.0j],
    ))
    eager = registry.solve()

    registry.compile_llvm_solver(tmp_path / "electrical-llvm")
    native = registry.solve()

    assert registry.solve_backend == "llvm"
    assert registry.circuit().solver is None
    assert torch.allclose(
        native.voltage_v["bus"], eager.voltage_v["bus"],
        rtol=1.0e-12, atol=1.0e-12,
    )


def test_parallel_complex_loads_are_solved_simultaneously() -> None:
    circuit = ComplexTensorCircuit(
        lanes=2,
        sources=[TheveninSource("source", "bus", [24 + 0j, 1 + 2j],
                                [0.2 + 0.1j, 0.3 + 0.2j])],
        branches=[
            ImpedanceBranch("load-a", "bus", GROUND,
                            [4.0 + 1.0j, 6.0 - 2.0j]),
            ImpedanceBranch("load-b", "bus", GROUND,
                            [8.0 - 0.5j, 3.0 + 4.0j]),
        ],
    )
    reading = circuit.solve()

    for lane in range(2):
        zs = complex([0.2 + 0.1j, 0.3 + 0.2j][lane])
        za = complex([4.0 + 1.0j, 6.0 - 2.0j][lane])
        zb = complex([8.0 - 0.5j, 3.0 + 4.0j][lane])
        vs = complex([24 + 0j, 1 + 2j][lane])
        parallel = 1.0 / (1.0 / za + 1.0 / zb)
        expected = vs * parallel / (zs + parallel)
        assert torch.allclose(
            reading.voltage_v["bus"][lane],
            torch.tensor(expected, dtype=torch.complex128),
        )


def test_three_phase_generator_has_terminals_and_separate_neutral_bond() -> None:
    root_three_over_two = 3.0 ** 0.5 / 2.0
    phases = (230.0 + 0.0j,
              230.0 * complex(-0.5, -root_three_over_two),
              230.0 * complex(-0.5, root_three_over_two))
    names = ("phase-a", "phase-b", "phase-c")
    circuit = ComplexTensorCircuit(
        lanes=1,
        sources=[
            TheveninSource(f"generator-{name}", name, [emf], [0.1 + 0.03j],
                            return_node="generator-neutral")
            for name, emf in zip(names, phases)
        ],
        branches=[
            ImpedanceBranch(f"load-{name}", name, "load-neutral", [12.0 + 3.0j])
            for name in names
        ] + [
            ImpedanceBranch("neutral-conductor", "load-neutral",
                            "generator-neutral", [0.02 + 0.0j]),
            ImpedanceBranch("neutral-frame-bond", "generator-neutral",
                            GROUND, [0.01 + 0.0j]),
        ],
    )

    reading = circuit.solve()

    phase_currents = [
        reading.source_current_a[f"generator-{name}"][0] for name in names
    ]
    assert torch.allclose(
        sum(phase_currents), torch.tensor(0.0j, dtype=torch.complex128),
        atol=1e-10)
    assert torch.allclose(
        reading.branch_current_a["neutral-frame-bond"],
        torch.zeros(1, dtype=torch.complex128), atol=1e-10,
    )


def test_machine_graph_requires_explicit_conductors_and_device_laws() -> None:
    legacy = {
        "nodes": [],
        "edges": [{
            "identity": "legacy-wire", "a": "battery", "b": "motor",
            "constraint": "insulated-copper-wire",
        }],
    }
    coverage = audit_machine_electrical_graph(legacy)
    assert coverage.undeclared_legacy_edges == ("legacy-wire",)
    assert not coverage.complete

    graph = {
        "nodes": [
            {"identity": "battery", "electrical_port_model": "battery"},
            {"identity": "motor", "electrical_port_model": "motor"},
        ],
        "edges": [{
            "identity": "positive-feeder", "a": "battery", "b": "motor",
            "constraint": "insulated-copper-wire",
            "transport_domain": "electrical",
            "rest_length": 2.0,
            "conductor_area_mm2": 25.0,
        }],
    }
    registry = ElectricalDeviceRegistry([0.0])
    battery = DCBattery(
        identity="battery", chemistry="lifepo4", capacity_ah=100.0,
        series_packs=4, state_of_charge=0.75,
    )
    laws = {
        "battery": battery_port_law("battery", "battery", battery),
        "motor": impedance_port_law("motor", "motor", GROUND, 4.0 + 0.0j),
    }
    coverage = register_machine_electrical_graph(
        registry, graph, device_laws=laws)
    assert coverage.complete
    reading = registry.solve()
    assert reading.voltage_v["motor"][0].real > 0.0


def test_baked_multiport_circuit_runs_inside_the_one_electrical_solve() -> None:
    registry = ElectricalDeviceRegistry([0.0, 60.0])
    registry.register_source(TheveninSource(
        "supply", "input", [12.0 + 0.0j, 0.0j], [0.1 + 0.0j] * 2))

    def divider(_voltage, frequencies):
        lanes = int(frequencies.numel())
        g_in = 1.0 / 100.0
        g_out = 1.0 / 200.0
        matrix = torch.tensor([
            [g_in, -g_in, 0.0],
            [-g_in, g_in + g_out, -g_out],
            [0.0, -g_out, g_out],
        ], dtype=torch.complex128).unsqueeze(0).repeat(lanes, 1, 1)
        return MultiportLinearization(
            matrix, torch.zeros((lanes, 3), dtype=torch.complex128))

    registry.register_circuit(EmpiricalCircuitLaw(
        "baked-divider", ("input", "output", GROUND), divider))

    reading = registry.solve()

    assert torch.allclose(
        reading.voltage_v["output"],
        torch.tensor([8.0 + 0.0j, 0.0j], dtype=torch.complex128),
        atol=0.01,
    )
    assert reading.circuit_terminal_current_a["baked-divider"].shape == (2, 3)
    engine = ComplexElectricalEngine(registry)
    ok, metrics, _state = engine.step(0.01)
    assert ok
    assert metrics.div_inf < 1.0e-10
    assert metrics.pub_contract.tolist() == [HOLD]
    assert metrics.pub_tau_present.tolist() == [0.0]


def test_linear_circuit_bake_exactly_eliminates_internal_nodes() -> None:
    detailed = ComplexTensorCircuit(
        lanes=2,
        branches=[
            ImpedanceBranch("first", "terminal-a", "internal", [10.0, 8.0]),
            ImpedanceBranch("second", "internal", "terminal-b", [20.0, 12.0]),
        ],
    )

    baked = bake_linear_circuit(
        "series-network", detailed, ("terminal-a", "terminal-b"))
    tangent = baked.tangent(
        torch.zeros((2, 2), dtype=torch.complex128),
        torch.tensor([0.0, 60.0], dtype=torch.float64),
    )

    expected_g = torch.tensor([1.0 / 30.0, 1.0 / 20.0],
                              dtype=torch.complex128)
    assert torch.allclose(tangent.admittance_s[:, 0, 0], expected_g)
    assert torch.allclose(tangent.admittance_s[:, 0, 1], -expected_g)
    assert torch.allclose(tangent.admittance_s[:, 1, 0], -expected_g)
    assert torch.allclose(tangent.admittance_s[:, 1, 1], expected_g)
    assert torch.count_nonzero(tangent.current_offset_a) == 0


def test_uniform_transmission_line_retains_propagation_and_terminal_conservation() -> None:
    frequencies = torch.tensor([1.0e3, 1.0e6], dtype=torch.float64)
    law = transmission_line_circuit_law(
        "coaxial-run", ("send", "send-return", "receive", "receive-return"),
        length_m=20.0,
        resistance_ohm_per_m=0.01,
        inductance_h_per_m=250.0e-9,
        conductance_s_per_m=1.0e-9,
        capacitance_f_per_m=100.0e-12,
    )

    tangent = law.tangent(
        torch.zeros((2, 4), dtype=torch.complex128), frequencies)

    # A four-terminal floating line neither creates nor destroys current.
    ones = torch.ones((2, 4), dtype=torch.complex128)
    assert torch.allclose(
        torch.einsum("lij,lj->li", tangent.admittance_s, ones),
        torch.zeros_like(ones), atol=1.0e-10)
    assert torch.allclose(
        tangent.admittance_s,
        tangent.admittance_s.transpose(1, 2), atol=1.0e-12)
    # Propagation makes the high-frequency terminal law distinct from the
    # low-frequency lumped-resistance limit.
    assert not torch.allclose(
        tangent.admittance_s[0], tangent.admittance_s[1])


def test_conductor_and_junction_losses_feed_wire_thermal_mesh_state() -> None:
    service = standard_services("test")["230v-1ph-50hz"]
    cable = BuildingCable(
        "test-cable", service, awg=12, length_m=5.0,
        allowable_ampacity_a=20.0)
    graph = {
        "nodes": [
            {"identity": "source-box", "reference_position": [0.0, 0.0, 0.0],
             "thermal_domain": "volume", "material": "copper",
             "body_half_extent_m": [0.05, 0.05, 0.05]},
            {"identity": "load-box", "reference_position": [5.0, 0.0, 0.0],
             "thermal_domain": "volume", "material": "copper",
             "body_half_extent_m": [0.05, 0.05, 0.05]},
        ],
        "edges": [{
            "identity": "feeder", "a": "source-box", "b": "load-box",
            "constraint": "insulated-copper-wire",
            "circuit_identity": service.identity,
            **cable.graph_attributes(),
            # Isolate one current path for this thermal-coupling test.
            "conductors": [cable.graph_attributes()["conductors"][0]],
        }],
    }
    registry = ElectricalDeviceRegistry([50.0])
    register_machine_electrical_graph(registry, graph, device_laws={})
    source_bus = f"source-box::{service.identity}::line"
    load_bus = f"load-box::{service.identity}::line"
    registry.register_source(TheveninSource(
        "supply", source_bus, [230.0 + 0.0j], [0.05 + 0.0j]))
    registry.register(impedance_port_law(
        "load", load_bus, GROUND, [23.0 + 0.0j]))
    assembly = build_thermal_assembly(graph)
    engine = ComplexElectricalEngine(registry)
    engine.bind_thermal_assembly(assembly)

    energy_before = assembly.energy_j
    ok, metrics, _state = engine.step(1.0)
    sources = engine.thermal_source_w(assembly=assembly)
    assembly.step(1.0, source_w=sources)

    wire_identity = "feeder::line"
    assert ok and metrics.div_inf < 1.0e-10
    assert engine.branch_loss_w[wire_identity] > 0.0
    assert engine.branch_loss_w[f"{wire_identity}::junction:a"] > 0.0
    assert engine.branch_loss_w[f"{wire_identity}::junction:b"] > 0.0
    assert engine.joule_heat_w[wire_identity] == engine.branch_loss_w[wire_identity]
    assert engine.joule_heat_w["source-box"] > 0.0
    assert engine.joule_heat_w["load-box"] > 0.0
    assert assembly.energy_j > energy_before
    assert abs((assembly.energy_j - energy_before) - sum(sources.values())) < 1e-8
    assert assembly.temperature_k(wire_identity) > 293.15
    assert assembly.temperature_k("source-box") > 293.15
    assert assembly.temperature_k("load-box") > 293.15
    assert engine.safety_events == ()


def test_overloaded_or_defective_installation_is_reported_with_thermal_target() -> None:
    service = standard_services("test")['230v-1ph-50hz']
    cable = BuildingCable(
        "unsafe-cable", service, awg=12, length_m=2.0,
        allowable_ampacity_a=20.0, junction_torque_verified=False)
    attributes = cable.graph_attributes()
    attributes["conductors"] = [attributes["conductors"][0]]
    graph = {
        "nodes": [
            {"identity": "source-box", "reference_position": [0.0, 0.0, 0.0],
             "thermal_domain": "volume", "material": "copper",
             "body_half_extent_m": [0.05, 0.05, 0.05]},
            {"identity": "load-box", "reference_position": [2.0, 0.0, 0.0],
             "thermal_domain": "volume", "material": "copper",
             "body_half_extent_m": [0.05, 0.05, 0.05]},
        ],
        "edges": [{
            "identity": "unsafe-feeder", "a": "source-box", "b": "load-box",
            "constraint": "insulated-copper-wire",
            "circuit_identity": service.identity,
            "continuous_load_limit_a": 16.0,
            **attributes,
        }],
    }
    registry = ElectricalDeviceRegistry([50.0])
    register_machine_electrical_graph(registry, graph, device_laws={})
    source_bus = f"source-box::{service.identity}::line"
    load_bus = f"load-box::{service.identity}::line"
    registry.register_source(TheveninSource(
        "supply", source_bus, [230.0 + 0.0j], [0.01 + 0.0j]))
    registry.register(impedance_port_law(
        "load", load_bus, GROUND, [5.0 + 0.0j]))
    assembly = build_thermal_assembly(graph)
    assembly.states["source-box"].temperature_k.fill(400.0)
    engine = ComplexElectricalEngine(registry)
    engine.bind_thermal_assembly(assembly)

    ok, _metrics, _state = engine.step(0.1)

    conditions = {event.condition for event in engine.safety_events}
    assert ok
    assert "continuous-load-threshold-exceeded" in conditions
    assert "conductor-ampacity-exceeded" in conditions
    assert "termination-torque-unverified" in conditions
    assert "termination-temperature-exceeded" in conditions
    assert any(
        event.thermal_target == "source-box"
        for event in engine.safety_events
        if event.condition == "termination-torque-unverified"
    )


def test_registered_rlc_and_battery_keep_complex_state_under_managed_dt() -> None:
    registry = ElectricalDeviceRegistry([0.0, 60.0])
    battery_model = DCBattery(
        identity="battery", chemistry="lifepo4", capacity_ah=100.0,
        series_packs=4, state_of_charge=0.75,
    )
    registry.register(battery_port_law("battery", "bus", battery_model))
    omega = 2.0 * torch.pi * registry.frequencies_hz
    registry.register(impedance_port_law(
        "motor", "bus", GROUND,
        torch.as_tensor(4.0 + 1j * omega * 0.003, dtype=torch.complex128),
    ))
    engine = ComplexElectricalEngine(registry)
    table = StateTable()
    engine.register(
        table,
        lambda _engine: {"pos": (0.0, 0.0, 0.0), "mass": 0.0},
        (engine,),
        group_label="electrical-system",
    )
    before = engine.snapshot()

    ok, metrics, _state = engine.step_with_state(
        {}, 0.25, state_table=table)

    assert ok
    assert engine.reading is not None
    assert engine.reading.voltage_v["bus"].dtype == torch.complex128
    assert engine.reading.voltage_v["bus"].shape == (2,)
    assert metrics.div_inf < 1.0e-10
    assert battery_model.state_of_charge < 0.75
    engine.publish_to_state(table)
    assert table.get("engine", "electrical-system", "voltage_phasor_v") is not None

    engine.restore(before)
    assert battery_model.state_of_charge == 0.75
    assert engine.reading is None
