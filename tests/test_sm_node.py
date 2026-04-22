"""Tests for StateMachineNode base class and graph registration."""
from __future__ import annotations

import pytest
import torch

from routing_engine import ParamEdge, RoutingGraph, RoutingEdge, SM_LAYERS
from sm_node import StateMachineNode

_CDTYPE = torch.complex128


# ── minimal concrete subclass ──────────────────────────────────────────

class EchoSM(StateMachineNode):
    """Passes each input slot through unchanged — for testing identity preservation."""
    sm_layer = "performer"

    @property
    def input_slots(self):
        return ["driver_0", "driver_1"]

    @property
    def output_slots(self):
        return ["out_0", "out_1", "mixed"]

    def step(self, inputs):
        d0 = inputs.get("driver_0", torch.zeros((), dtype=_CDTYPE))
        d1 = inputs.get("driver_1", torch.zeros((), dtype=_CDTYPE))
        return {
            "out_0":  d0,
            "out_1":  d1,
            "mixed":  d0 + d1,   # one cross-coupled output
        }

    def reset(self):
        pass


class StatefulSM(StateMachineNode):
    """Integrates its input — carries state between steps."""
    sm_layer = "room"

    @property
    def input_slots(self):
        return ["mic_in"]

    @property
    def output_slots(self):
        return ["mic_out"]

    def __init__(self):
        self._acc = torch.zeros((), dtype=_CDTYPE)

    def step(self, inputs):
        x = inputs.get("mic_in", torch.zeros((), dtype=_CDTYPE))
        self._acc = self._acc * 0.9 + x * 0.1   # leaky integrator
        return {"mic_out": self._acc}

    def reset(self):
        self._acc = torch.zeros((), dtype=_CDTYPE)


class PerformerControlSM(StateMachineNode):
    """Performer-style SM with both signal and control outputs."""
    sm_layer = "performer"

    @property
    def input_slots(self):
        return ["driver_0"]

    @property
    def output_slots(self):
        return ["aperture_audio"]

    @property
    def control_output_slots(self):
        return ["aperture_pressure", "clan_sympathy"]

    def step(self, inputs):
        x = inputs.get("driver_0", torch.zeros((), dtype=_CDTYPE))
        return {
            "aperture_audio": x,
            "aperture_pressure": x.real.to(_CDTYPE),
            "clan_sympathy": (0.5 * x.real).to(_CDTYPE),
        }

    def reset(self):
        pass


# ══════════════════════════════════════════════════════════════════════
# Interface contract
# ══════════════════════════════════════════════════════════════════════

class TestStateMachineNodeInterface:
    def test_sm_layer_attribute(self):
        sm = EchoSM()
        assert sm.sm_layer == "performer"
        assert sm.sm_layer in SM_LAYERS

    def test_input_slots(self):
        sm = EchoSM()
        assert sm.input_slots == ["driver_0", "driver_1"]

    def test_output_slots(self):
        sm = EchoSM()
        assert sm.output_slots == ["out_0", "out_1", "mixed"]

    def test_step_identity(self):
        sm = EchoSM()
        d0 = torch.tensor(1.0 + 0j, dtype=_CDTYPE)
        d1 = torch.tensor(0.5 + 0.5j, dtype=_CDTYPE)
        out = sm.step({"driver_0": d0, "driver_1": d1})
        assert out["out_0"] is d0
        assert out["out_1"] is d1
        assert abs(out["mixed"].item() - (1.5 + 0.5j)) < 1e-12

    def test_step_missing_slot_is_zero(self):
        sm = EchoSM()
        out = sm.step({"driver_0": torch.tensor(2.0 + 0j, dtype=_CDTYPE)})
        assert out["out_1"].abs().item() < 1e-12

    def test_signals_not_aggregated(self):
        sm = EchoSM()
        d0 = torch.tensor(3.0 + 0j, dtype=_CDTYPE)
        d1 = torch.tensor(4.0 + 0j, dtype=_CDTYPE)
        out = sm.step({"driver_0": d0, "driver_1": d1})
        # out_0 must equal d0 exactly — not mixed with d1
        assert abs(out["out_0"].item() - 3.0) < 1e-12
        assert abs(out["out_1"].item() - 4.0) < 1e-12

    def test_stateful_sm_accumulates(self):
        sm = StatefulSM()
        inp = torch.tensor(1.0 + 0j, dtype=_CDTYPE)
        out1 = sm.step({"mic_in": inp})
        out2 = sm.step({"mic_in": inp})
        # second step should differ from first (state accumulated)
        assert out2["mic_out"].abs().item() > out1["mic_out"].abs().item()

    def test_reset_clears_state(self):
        sm = StatefulSM()
        for _ in range(10):
            sm.step({"mic_in": torch.tensor(1.0 + 0j, dtype=_CDTYPE)})
        sm.reset()
        out = sm.step({"mic_in": torch.zeros((), dtype=_CDTYPE)})
        assert out["mic_out"].abs().item() < 1e-12

    def test_abstract_prevents_direct_instantiation(self):
        with pytest.raises(TypeError):
            StateMachineNode()


# ══════════════════════════════════════════════════════════════════════
# Node key helpers
# ══════════════════════════════════════════════════════════════════════

class TestNodeKeyHelpers:
    def test_in_node_key_format(self):
        sm = EchoSM()
        assert sm.in_node_key("perf_0", "driver_1") == "perf_0.in.driver_1"

    def test_out_node_key_format(self):
        sm = EchoSM()
        assert sm.out_node_key("perf_0", "out_0") == "perf_0.out.out_0"

    def test_ctrl_node_key_format(self):
        sm = PerformerControlSM()
        assert sm.ctrl_node_key("perf_0", "aperture_pressure") == "perf_0.ctrl.aperture_pressure"


# ══════════════════════════════════════════════════════════════════════
# Graph registration
# ══════════════════════════════════════════════════════════════════════

class TestGraphRegistration:
    def test_register_creates_nodes(self):
        sm = EchoSM()
        g = RoutingGraph()
        sm.register_in_graph(g, "perf_0",
                             feeding_router_type="driver_router",
                             receiving_router_type="instrument_router")
        assert "perf_0.in.driver_0" in g.nodes
        assert "perf_0.in.driver_1" in g.nodes
        assert "perf_0.out.out_0" in g.nodes
        assert "perf_0.out.out_1" in g.nodes
        assert "perf_0.out.mixed" in g.nodes

    def test_register_creates_control_nodes(self):
        sm = PerformerControlSM()
        g = RoutingGraph()
        sm.register_in_graph(g, "perf_0", parameter_router_type="param")
        assert "perf_0.ctrl.aperture_pressure" in g.nodes
        assert "perf_0.ctrl.clan_sympathy" in g.nodes

    def test_control_nodes_are_param_sources_not_sinks(self):
        sm = PerformerControlSM()
        g = RoutingGraph()
        sm.register_in_graph(g, "perf_0", parameter_router_type="param")
        assert g.is_source_in_router("perf_0.ctrl.aperture_pressure", "param")
        assert not g.is_sink_in_router("perf_0.ctrl.aperture_pressure", "param")
        assert not g.is_sink_in_router("perf_0.ctrl.aperture_pressure", "driver_router")

    def test_input_nodes_are_sinks_not_sources(self):
        sm = EchoSM()
        g = RoutingGraph()
        sm.register_in_graph(g, "perf_0", feeding_router_type="driver_router")
        # Input nodes: sink in driver_router, NOT a source anywhere
        assert g.is_sink_in_router("perf_0.in.driver_0", "driver_router")
        assert not g.is_source_in_router("perf_0.in.driver_0", "driver_router")
        assert not g.is_source_in_router("perf_0.in.driver_0", "instrument_router")

    def test_output_nodes_are_sources_not_sinks(self):
        sm = EchoSM()
        g = RoutingGraph()
        sm.register_in_graph(g, "perf_0", receiving_router_type="instrument_router")
        # Output nodes: source in instrument_router, NOT a sink anywhere
        assert g.is_source_in_router("perf_0.out.out_0", "instrument_router")
        assert not g.is_sink_in_router("perf_0.out.out_0", "instrument_router")
        assert not g.is_sink_in_router("perf_0.out.out_0", "driver_router")

    def test_node_type_set_to_sm_layer(self):
        sm = EchoSM()
        g = RoutingGraph()
        sm.register_in_graph(g, "perf_0")
        assert g.get_node_type("perf_0.in.driver_0") == "performer"
        assert g.get_node_type("perf_0.out.out_0") == "performer"

    def test_control_node_type_set_to_sm_layer(self):
        sm = PerformerControlSM()
        g = RoutingGraph()
        sm.register_in_graph(g, "perf_0", parameter_router_type="param")
        assert g.get_node_type("perf_0.ctrl.aperture_pressure") == "performer"

    def test_no_router_type_means_unrestricted(self):
        sm = EchoSM()
        g = RoutingGraph()
        sm.register_in_graph(g, "perf_0")   # no router types given
        # Output nodes with no receiving_router_type → unrestricted source
        assert g.is_source_in_router("perf_0.out.out_0", "anything")

    def test_multiple_sms_no_collision(self):
        sm = EchoSM()
        g = RoutingGraph()
        sm.register_in_graph(g, "perf_0")
        sm.register_in_graph(g, "perf_1")
        assert "perf_0.in.driver_0" in g.nodes
        assert "perf_1.in.driver_0" in g.nodes
        # Keys must be distinct
        assert len(g.nodes) == len(set(g.nodes))

    def test_deregister_removes_nodes(self):
        sm = EchoSM()
        g = RoutingGraph()
        g.add_node("other")
        sm.register_in_graph(g, "perf_0")
        sm.deregister_from_graph(g, "perf_0")
        assert "perf_0.in.driver_0" not in g.nodes
        assert "perf_0.out.out_0" not in g.nodes
        assert "other" in g.nodes   # unrelated node survives

    def test_deregister_prunes_edges(self):
        sm = EchoSM()
        g = RoutingGraph()
        g.add_node("drv")
        sm.register_in_graph(g, "perf_0", feeding_router_type="dr")
        g.edges.append(RoutingEdge("drv", "perf_0.in.driver_0", 1.0,
                                   item_slot="driver_0"))
        sm.deregister_from_graph(g, "perf_0")
        assert all(e.dst_key != "perf_0.in.driver_0" for e in g.edges)

    def test_deregister_prunes_param_edges(self):
        sm = PerformerControlSM()
        g = RoutingGraph()
        g.add_node("driver_0")
        sm.register_in_graph(
            g,
            "perf_0",
            parameter_router_type="param",
            install_default_param_edges=True,
            default_param_target_map={
                "aperture_pressure": [("driver_0", "chirp.f_delta_start", 1.0, 0)],
            },
        )
        sm.deregister_from_graph(g, "perf_0")
        assert all(not pe.src_key.startswith("perf_0.ctrl.") for pe in g.param_edges)

    def test_register_can_install_default_param_edges(self):
        sm = PerformerControlSM()
        g = RoutingGraph()
        g.add_node("driver_0")
        sm.register_in_graph(
            g,
            "perf_0",
            parameter_router_type="param",
            install_default_param_edges=True,
            default_param_target_map={
                "aperture_pressure": [("driver_0", "chirp.f_delta_start", 1.25, 0)],
                "clan_sympathy": [("driver_0", "chirp.sympathy", 0.5, 0)],
            },
        )
        assert len(g.param_edges) == 2
        assert isinstance(g.param_edges[0], ParamEdge)
        assert g.param_edges[0].src_key == "perf_0.ctrl.aperture_pressure"
        assert g.param_edges[0].dst_key == "driver_0"
        assert g.param_edges[0].param_path == "chirp.f_delta_start"
        assert g.param_edges[0].delay_samples == 0


# ══════════════════════════════════════════════════════════════════════
# read_inputs / write_outputs helpers
# ══════════════════════════════════════════════════════════════════════

class TestReadWriteHelpers:
    def test_read_inputs_from_graph_state(self):
        sm = EchoSM()
        state = {
            "perf_0.in.driver_0": torch.tensor(1.0 + 0j, dtype=_CDTYPE),
            "perf_0.in.driver_1": torch.tensor(2.0 + 0j, dtype=_CDTYPE),
        }
        inputs = sm.read_inputs(state, "perf_0")
        assert abs(inputs["driver_0"].item() - (1.0 + 0j)) < 1e-12
        assert abs(inputs["driver_1"].item() - (2.0 + 0j)) < 1e-12

    def test_read_missing_slot_returns_zero(self):
        sm = EchoSM()
        inputs = sm.read_inputs({}, "perf_0")
        for slot in sm.input_slots:
            assert inputs[slot].abs().item() < 1e-12

    def test_write_outputs_to_graph_state(self):
        sm = EchoSM()
        state: dict = {}
        outputs = {
            "out_0":  torch.tensor(3.0 + 0j, dtype=_CDTYPE),
            "out_1":  torch.tensor(4.0 + 0j, dtype=_CDTYPE),
            "mixed":  torch.tensor(7.0 + 0j, dtype=_CDTYPE),
        }
        sm.write_outputs(outputs, state, "perf_0")
        assert abs(state["perf_0.out.out_0"].item() - (3.0 + 0j)) < 1e-12
        assert abs(state["perf_0.out.mixed"].item() - (7.0 + 0j)) < 1e-12

    def test_write_missing_output_slot_is_zero(self):
        sm = EchoSM()
        state: dict = {}
        sm.write_outputs({}, state, "perf_0")
        for slot in sm.output_slots:
            assert state[f"perf_0.out.{slot}"].abs().item() < 1e-12

    def test_write_control_outputs_to_graph_state(self):
        sm = PerformerControlSM()
        state: dict = {}
        outputs = {
            "aperture_pressure": torch.tensor(2.0 + 0j, dtype=_CDTYPE),
            "clan_sympathy": torch.tensor(0.25 + 0j, dtype=_CDTYPE),
        }
        sm.write_control_outputs(outputs, state, "perf_0")
        assert abs(state["perf_0.ctrl.aperture_pressure"].item() - (2.0 + 0j)) < 1e-12
        assert abs(state["perf_0.ctrl.clan_sympathy"].item() - (0.25 + 0j)) < 1e-12

    def test_write_missing_control_output_slot_is_zero(self):
        sm = PerformerControlSM()
        state: dict = {}
        sm.write_control_outputs({}, state, "perf_0")
        for slot in sm.control_output_slots:
            assert state[f"perf_0.ctrl.{slot}"].abs().item() < 1e-12

    def test_default_param_connections_builds_param_edges(self):
        sm = PerformerControlSM()
        edges = sm.default_param_connections(
            "perf_0",
            target_map={
                "aperture_pressure": [("driver_0", "chirp.f_delta_start", 1.0, 0)],
                "clan_sympathy": [("driver_1", "chirp.sympathy", 0.5, 1)],
            },
        )
        assert len(edges) == 2
        assert edges[0].src_key == "perf_0.ctrl.aperture_pressure"
        assert edges[1].src_key == "perf_0.ctrl.clan_sympathy"
        assert edges[1].delay_samples == 1

    def test_full_round_trip(self):
        """Read → step → write round-trip through graph state."""
        sm = EchoSM()
        state = {
            "perf_0.in.driver_0": torch.tensor(5.0 + 0j, dtype=_CDTYPE),
            "perf_0.in.driver_1": torch.tensor(3.0 + 0j, dtype=_CDTYPE),
        }
        inputs = sm.read_inputs(state, "perf_0")
        outputs = sm.step(inputs)
        sm.write_outputs(outputs, state, "perf_0")
        assert abs(state["perf_0.out.out_0"].item() - (5.0 + 0j)) < 1e-12
        assert abs(state["perf_0.out.mixed"].item() - (8.0 + 0j)) < 1e-12
