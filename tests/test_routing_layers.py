"""Tests for layered routing: node types, participation, router_key edges,
ParamEdge causality, saturation functions, CompiledRouter, and cycle detection."""
from __future__ import annotations

import math
import pytest
import torch

from routing_engine import (
    EdgeTransferSpec,
    FeedbackConfig,
    MetaEdge,
    ParamEdge,
    RoutingEdge,
    RoutingGraph,
    RouterInstance,
    SIGNAL_LAYERS,
    MIXER_LAYERS,
    SM_LAYERS,
    TensorPortContract,
)
from routing_solve_torch import (
    CompiledRouter,
    find_zero_delay_cycles,
    lower_meta_edges,
    _sat_tanh,
    _sat_hardclip,
    _sat_softclip,
)

_DEV = torch.device("cpu")
_SR = 48_000


# ══════════════════════════════════════════════════════════════════════
# RoutingGraph — node type and participation
# ══════════════════════════════════════════════════════════════════════

class TestNodeTypes:
    def test_add_node_with_type(self):
        g = RoutingGraph()
        g.add_node("v1", node_type="voice")
        assert g.get_node_type("v1") == "voice"

    def test_layer_alias_kept_in_sync(self):
        g = RoutingGraph()
        g.add_node("v1", layer="voice")
        assert g.get_node_type("v1") == "voice"
        assert g.node_layers["v1"] == "voice"

    def test_node_type_takes_precedence_over_layer(self):
        g = RoutingGraph()
        g.add_node("v1", node_type="driver", layer="voice")
        assert g.get_node_type("v1") == "driver"

    def test_set_node_type(self):
        g = RoutingGraph()
        g.add_node("v1")
        g.set_node_type("v1", "instrument")
        assert g.get_node_type("v1") == "instrument"

    def test_nodes_in_layer(self):
        g = RoutingGraph()
        g.add_node("v1", node_type="voice")
        g.add_node("v2", node_type="voice")
        g.add_node("d1", node_type="driver")
        assert set(g.nodes_in_layer("voice")) == {"v1", "v2"}
        assert g.nodes_in_layer("driver") == ["d1"]

    def test_source_sink_participation(self):
        g = RoutingGraph()
        g.add_node("v1",
                   source_router_types=["voice_router"],
                   sink_router_types=[])
        g.add_node("d1",
                   source_router_types=["driver_router"],
                   sink_router_types=["voice_router"])

        assert g.is_source_in_router("v1", "voice_router")
        assert not g.is_source_in_router("v1", "driver_router")
        assert g.is_sink_in_router("d1", "voice_router")
        assert not g.is_sink_in_router("d1", "driver_router")
        assert g.is_source_in_router("d1", "driver_router")

    def test_unrestricted_node_participates_everywhere(self):
        g = RoutingGraph()
        g.add_node("x")   # no participation declared
        assert g.is_source_in_router("x", "voice_router")
        assert g.is_sink_in_router("x", "master_router")

    def test_set_node_participation(self):
        g = RoutingGraph()
        g.add_node("m1")
        g.set_node_participation("m1",
                                 source_router_types=["master"],
                                 sink_router_types=["instrument"])
        assert g.get_node_source_router_types("m1") == ["master"]
        assert g.get_node_sink_router_types("m1") == ["instrument"]

    def test_nodes_for_router_source_sink(self):
        g = RoutingGraph()
        g.add_node("v1", source_router_types=["vr"], sink_router_types=[])
        g.add_node("d1", source_router_types=["dr"], sink_router_types=["vr"])
        g.add_node("free")
        srcs = g.nodes_for_router_source("vr")
        assert "v1" in srcs
        assert "free" in srcs   # unrestricted
        assert "d1" not in srcs
        sinks = g.nodes_for_router_sink("vr")
        assert "d1" in sinks
        assert "free" in sinks
        assert "v1" not in sinks


# ══════════════════════════════════════════════════════════════════════
# RoutingEdge — router_key and saturation
# ══════════════════════════════════════════════════════════════════════

class TestRoutingEdge:
    def test_router_key_default_empty(self):
        e = RoutingEdge("a", "b", 1.0)
        assert e.router_key == ""

    def test_saturation_default_empty(self):
        e = RoutingEdge("a", "b", 1.0)
        assert e.saturation == ""
        assert e.saturation_knee == 1.0

    def test_roundtrip_router_key(self):
        g = RoutingGraph()
        g.add_node("a"); g.add_node("b")
        g.edges.append(RoutingEdge("a", "b", 1.0, router_key="vr1"))
        d = g.to_dict()
        g2 = RoutingGraph.from_dict(d)
        assert g2.edges[0].router_key == "vr1"

    def test_roundtrip_saturation(self):
        g = RoutingGraph()
        g.add_node("a"); g.add_node("b")
        g.edges.append(RoutingEdge("a", "b", 1.0,
                                   saturation="tanh", saturation_knee=0.5))
        d = g.to_dict()
        g2 = RoutingGraph.from_dict(d)
        assert g2.edges[0].saturation == "tanh"
        assert g2.edges[0].saturation_knee == pytest.approx(0.5)

    def test_no_saturation_not_emitted(self):
        e = RoutingEdge("a", "b", 1.0)
        g = RoutingGraph()
        g.add_node("a"); g.add_node("b")
        g.edges.append(e)
        d = g.to_dict()
        assert "saturation" not in d["edges"][0]

    def test_tensor_contract_defaults_to_analytic_complex(self):
        e = RoutingEdge("a", "b", 1.0)
        assert e.tensor_contract.dtype == "complex128"
        assert e.tensor_contract.analytic_only is True
        assert e.transfer_spec.analytic_only is True

    def test_roundtrip_tensor_transfer_metadata(self):
        g = RoutingGraph()
        g.add_node("a"); g.add_node("b")
        g.edges.append(RoutingEdge(
            "a", "b", 1.0,
            edge_kind="control",
            src_port="room.ctrl.pressure",
            dst_port="performer.param.aperture",
            tensor_contract=TensorPortContract(
                tensor_rank=2, lane_count=0, batch_axes=2,
                parallel_group="room_feedback", group_validity="broadcast",
            ),
            transfer_spec=EdgeTransferSpec(
                transfer_policy="reduce", cable_count=8,
                batch_policy="broadcast", lane_policy="reduce",
                reduction="sum",
            ),
        ))
        d = g.to_dict()
        g2 = RoutingGraph.from_dict(d)
        e2 = g2.edges[0]
        assert e2.edge_kind == "control"
        assert e2.src_port == "room.ctrl.pressure"
        assert e2.dst_port == "performer.param.aperture"
        assert e2.tensor_contract.parallel_group == "room_feedback"
        assert e2.transfer_spec.transfer_policy == "reduce"
        assert e2.transfer_spec.cable_count == 8

    def test_meta_edge_roundtrip(self):
        g = RoutingGraph()
        g.add_node("score_sm")
        g.add_node("performer_sm")
        g.meta_edges.append(MetaEdge(
            a_key="score_sm",
            b_key="performer_sm",
            a_port="score_sm:score",
            b_port="performer_sm:score",
            semantic_role="score_bundle",
            channel_count=64,
            tensor_contract=TensorPortContract(
                tensor_rank=3,
                lane_count=0,
                batch_axes=2,
                parallel_group="score_bundle",
                group_validity="remap",
                semantic_role="score_bundle",
                channel_dims=[16, 4],
            ),
            a_to_b_transfer=EdgeTransferSpec(
                transfer_policy="scatter",
                cable_count=64,
                batch_policy="broadcast",
                lane_policy="remap",
            ),
            b_to_a_transfer=EdgeTransferSpec(
                transfer_policy="gather",
                cable_count=64,
                batch_policy="reduce",
                lane_policy="remap",
                reduction="mean",
            ),
        ))
        g2 = RoutingGraph.from_dict(g.to_dict())
        me = g2.meta_edges[0]
        assert me.semantic_role == "score_bundle"
        assert me.channel_count == 64
        assert me.tensor_contract.channel_dims == [16, 4]
        assert me.a_to_b_transfer.transfer_policy == "scatter"
        assert me.b_to_a_transfer.transfer_policy == "gather"


# ══════════════════════════════════════════════════════════════════════
# ParamEdge — instantaneous allowed, negative clamped
# ══════════════════════════════════════════════════════════════════════

class TestParamEdge:
    def test_zero_delay_allowed(self):
        pe = ParamEdge(delay_samples=0)
        assert pe.delay_samples == 0

    def test_positive_delay_kept(self):
        pe = ParamEdge(delay_samples=3)
        assert pe.delay_samples == 3

    def test_negative_clamped_to_zero(self):
        pe = ParamEdge(delay_samples=-5)
        assert pe.delay_samples == 0

    def test_param_path_roundtrip(self):
        pe = ParamEdge("a", "b", 1.0, "magnitude", 0, "chirp.f_start")
        d = pe.to_dict()
        pe2 = ParamEdge.from_dict(d)
        assert pe2.param_path == "chirp.f_start"
        assert pe2.delay_samples == 0

    def test_no_param_path_not_emitted(self):
        pe = ParamEdge("a", "b")
        d = pe.to_dict()
        assert "param_path" not in d

    def test_legacy_load_no_delay_samples(self):
        d = {"src": "a", "dst": "b", "w": 1.0, "extractor": "magnitude"}
        pe = ParamEdge.from_dict(d)
        assert pe.delay_samples == 0

    def test_param_edge_tensor_contract_defaults_analytic(self):
        pe = ParamEdge("a", "b")
        assert pe.tensor_contract.dtype == "complex128"
        assert pe.tensor_contract.analytic_only is True
        assert pe.transfer_spec.analytic_only is True

    def test_param_edge_roundtrip_transfer_metadata(self):
        pe = ParamEdge(
            "a", "b", 1.0, "magnitude", 0, "chirp.f_start",
            src_port="sm.ctrl.aperture",
            dst_port="voice_a:param:chirp.f_start",
            tensor_contract=TensorPortContract(
                tensor_rank=1, lane_count=0, batch_axes=1,
                parallel_group="performer_batch",
            ),
            transfer_spec=EdgeTransferSpec(
                transfer_policy="broadcast",
                cable_count=4,
                batch_policy="broadcast",
                lane_policy="strict",
            ),
        )
        pe2 = ParamEdge.from_dict(pe.to_dict())
        assert pe2.src_port == "sm.ctrl.aperture"
        assert pe2.dst_port == "voice_a:param:chirp.f_start"
        assert pe2.tensor_contract.parallel_group == "performer_batch"
        assert pe2.transfer_spec.cable_count == 4

    def test_param_edge_projection_policy_roundtrip(self):
        pe = ParamEdge(
            "a", "b", 1.0, "magnitude", 0, "amplitude",
            projection_policy="lane0_magnitude",
        )
        pe2 = ParamEdge.from_dict(pe.to_dict())
        assert pe2.projection_policy == "lane0_magnitude"


# ══════════════════════════════════════════════════════════════════════
# Serialisation backward compat
# ══════════════════════════════════════════════════════════════════════

class TestSerialisation:
    def test_old_node_layers_loads_as_node_types(self):
        d = {
            "nodes": ["v1", "v2"],
            "edges": [],
            "param_edges": [],
            "feedback": {},
            "node_layers": {"v1": "voice", "v2": "driver"},
        }
        g = RoutingGraph.from_dict(d)
        assert g.get_node_type("v1") == "voice"
        assert g.get_node_type("v2") == "driver"
        assert g.node_layers == g.node_types   # alias in sync

    def test_new_node_types_preferred_over_node_layers(self):
        d = {
            "nodes": ["v1"],
            "edges": [], "param_edges": [], "feedback": {},
            "node_types": {"v1": "instrument"},
            "node_layers": {"v1": "voice"},   # should be ignored
        }
        g = RoutingGraph.from_dict(d)
        assert g.get_node_type("v1") == "instrument"

    def test_source_sink_participation_roundtrip(self):
        g = RoutingGraph()
        g.add_node("x",
                   source_router_types=["voice_router"],
                   sink_router_types=["driver_router"])
        g2 = RoutingGraph.from_dict(g.to_dict())
        assert g2.node_source_router_types["x"] == ["voice_router"]
        assert g2.node_sink_router_types["x"] == ["driver_router"]

    def test_signal_layer_constants_intact(self):
        assert "voice" in SIGNAL_LAYERS
        assert "driver" in SIGNAL_LAYERS
        assert "master" in SIGNAL_LAYERS
        assert "performer" in SM_LAYERS
        assert "voice_router" in MIXER_LAYERS


# ══════════════════════════════════════════════════════════════════════
# Saturation functions
# ══════════════════════════════════════════════════════════════════════

class TestSaturationFns:
    def _z(self, mag):
        return torch.tensor(complex(mag, 0), dtype=torch.complex128)

    def test_tanh_compresses_above_knee(self):
        z = self._z(5.0)
        out = _sat_tanh(z, 1.0)
        assert out.abs().item() < 5.0
        assert out.abs().item() < 1.1   # tanh(5)·1 ≈ 0.9999

    def test_tanh_identity_near_zero(self):
        z = self._z(0.01)
        out = _sat_tanh(z, 1.0)
        assert abs(out.abs().item() - 0.01) < 1e-4

    def test_tanh_preserves_phase(self):
        z = torch.tensor(3.0 + 4.0j, dtype=torch.complex128)
        out = _sat_tanh(z, 2.0)
        assert abs(out.angle().item() - z.angle().item()) < 1e-9

    def test_hardclip_at_knee(self):
        z = self._z(2.0)
        out = _sat_hardclip(z, 1.0)
        assert abs(out.abs().item() - 1.0) < 1e-9

    def test_hardclip_passthrough_below_knee(self):
        z = self._z(0.5)
        out = _sat_hardclip(z, 1.0)
        assert abs(out.abs().item() - 0.5) < 1e-9

    def test_softclip_identity_below_knee(self):
        z = self._z(0.5)
        out = _sat_softclip(z, 1.0)
        assert abs(out.abs().item() - 0.5) < 1e-9

    def test_softclip_compresses_above_knee(self):
        z = self._z(3.0)
        out = _sat_softclip(z, 1.0)
        assert out.abs().item() < 3.0

    def test_all_fns_return_finite(self):
        z = torch.tensor(1e10 + 0j, dtype=torch.complex128)
        for fn, knee in [(_sat_tanh, 1.0), (_sat_hardclip, 1.0), (_sat_softclip, 1.0)]:
            out = fn(z, knee)
            assert torch.isfinite(out.abs())


# ══════════════════════════════════════════════════════════════════════
# find_zero_delay_cycles
# ══════════════════════════════════════════════════════════════════════

class TestCycleDetection:
    def test_dag_no_cycles(self):
        edges = [RoutingEdge("a", "b", 1.0), RoutingEdge("b", "c", 1.0)]
        assert find_zero_delay_cycles(["a", "b", "c"], edges, _SR) == []

    def test_simple_cycle(self):
        edges = [RoutingEdge("a", "b", 1.0), RoutingEdge("b", "a", 0.5)]
        sccs = find_zero_delay_cycles(["a", "b"], edges, _SR)
        assert len(sccs) == 1
        assert set(sccs[0]) == {"a", "b"}

    def test_self_loop(self):
        edges = [RoutingEdge("a", "a", 0.3)]
        sccs = find_zero_delay_cycles(["a"], edges, _SR)
        assert len(sccs) == 1

    def test_delayed_edge_not_counted_as_cycle(self):
        edges = [
            RoutingEdge("a", "b", 1.0),
            RoutingEdge("b", "a", 0.5, delay_s=0.001),  # 48-sample delay — not zero
        ]
        assert find_zero_delay_cycles(["a", "b"], edges, _SR) == []

    def test_multiple_cycles(self):
        edges = [
            RoutingEdge("a", "b", 1.0), RoutingEdge("b", "a", 0.5),
            RoutingEdge("c", "d", 1.0), RoutingEdge("d", "c", 0.5),
        ]
        sccs = find_zero_delay_cycles(["a", "b", "c", "d"], edges, _SR)
        assert len(sccs) == 2


# ══════════════════════════════════════════════════════════════════════
# CompiledRouter — linear path
# ══════════════════════════════════════════════════════════════════════

class TestCompiledRouterLinear:
    def _simple(self, weight=0.5):
        """a → b with given weight, single instance."""
        edges = [RoutingEdge("a", "b", weight)]
        return CompiledRouter(["a", "b"], edges, _SR, _DEV)

    def test_single_step_shape(self):
        cr = self._simple()
        src = torch.zeros(2, dtype=torch.complex128)
        out = cr.step(src)
        assert out.shape == (2,)

    def test_batched_step_shape(self):
        cr = CompiledRouter(["a", "b"],
                            [RoutingEdge("a", "b", 0.5)],
                            _SR, _DEV, batch_size=4)
        src = torch.zeros(4, 2, dtype=torch.complex128)
        out = cr.step(src)
        assert out.shape == (4, 2)

    def test_linear_routing_value(self):
        cr = self._simple(0.5)
        src = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
        out = cr.step(src)
        # b receives a routed through (I-W)^{-1}: b = 0.5*a / (1 - 0) = 0.5
        assert abs(out[1].item() - 0.5) < 1e-10

    def test_no_saturation_iters_zero(self):
        cr = self._simple()
        src = torch.zeros(2, dtype=torch.complex128)
        cr.step(src)
        assert cr.last_convergence_iters == 0
        assert not cr.last_saturated

    def test_delayed_edge_ring_buffer(self):
        delay_s = 2 / _SR   # exactly 2 samples
        edges = [RoutingEdge("a", "b", 1.0, delay_s=delay_s)]
        cr = CompiledRouter(["a", "b"], edges, _SR, _DEV)
        src1 = torch.tensor([1.0 + 0j, 0.0], dtype=torch.complex128)
        src0 = torch.zeros(2, dtype=torch.complex128)
        cr.step(src1)         # t=0: inject 1 at a
        cr.step(src0)         # t=1: nothing
        out = cr.step(src0)   # t=2: delayed contribution should arrive at b
        assert abs(out[1].item() - 1.0) < 1e-9

    def test_reset_clears_ring(self):
        delay_s = 1 / _SR
        edges = [RoutingEdge("a", "b", 1.0, delay_s=delay_s)]
        cr = CompiledRouter(["a", "b"], edges, _SR, _DEV)
        src = torch.tensor([1.0 + 0j, 0.0], dtype=torch.complex128)
        cr.step(src)
        cr.reset()
        out = cr.step(torch.zeros(2, dtype=torch.complex128))
        assert out[1].abs().item() < 1e-12

    def test_output_helper(self):
        cr = self._simple(0.5)
        src = torch.tensor([2.0 + 0j, 0.0], dtype=torch.complex128)
        X = cr.step(src)
        assert cr.output(X, "a").item() == X[0].item()
        assert cr.output(X, "b").item() == X[1].item()


# ══════════════════════════════════════════════════════════════════════
# CompiledRouter — saturating path
# ══════════════════════════════════════════════════════════════════════

class TestCompiledRouterSaturation:
    def _sat_cycle(self, sat="tanh", knee=0.5, gain=2.0):
        """a → b and b → a, both saturating — a feedback cycle."""
        edges = [
            RoutingEdge("a", "b", gain, saturation=sat, saturation_knee=knee),
            RoutingEdge("b", "a", gain, saturation=sat, saturation_knee=knee),
        ]
        return CompiledRouter(["a", "b"], edges, _SR, _DEV, max_iterations=128)

    def test_sat_cycle_converges(self):
        cr = self._sat_cycle()
        src = torch.tensor([0.1 + 0j, 0.0], dtype=torch.complex128)
        out = cr.step(src)
        assert cr.last_convergence_iters > 0   # iteration ran
        assert not cr.last_saturated
        assert torch.isfinite(out.abs()).all()

    def test_sat_iters_recorded(self):
        cr = self._sat_cycle()
        src = torch.tensor([1.0 + 0j, 0.0], dtype=torch.complex128)
        cr.step(src)
        assert cr.last_convergence_iters >= 1

    def test_infinity_flag_on_runaway(self):
        # Very high gain, no saturation knee to cap it — should trigger infinity flag
        edges = [
            RoutingEdge("a", "b", 100.0, saturation="tanh", saturation_knee=1e10),
            RoutingEdge("b", "a", 100.0, saturation="tanh", saturation_knee=1e10),
        ]
        cr = CompiledRouter(["a", "b"], edges, _SR, _DEV,
                            max_iterations=256, infinity_threshold=1e6)
        src = torch.tensor([1.0 + 0j, 0.0], dtype=torch.complex128)
        out = cr.step(src)
        assert cr.last_saturated
        assert torch.isfinite(out.abs()).all()  # still finite — sat fn bounded it

    def test_hardclip_cycle_bounded(self):
        cr = self._sat_cycle("hardclip", knee=1.0, gain=3.0)
        src = torch.tensor([5.0 + 0j, 5.0 + 0j], dtype=torch.complex128)
        out = cr.step(src)
        assert out.abs().max().item() < 20.0  # hard-clipped, not runaway

    def test_mixed_sat_linear_edges(self):
        """One saturating edge, one linear — only the sat path iterates."""
        edges = [
            RoutingEdge("a", "b", 1.0),                        # linear
            RoutingEdge("b", "c", 0.8, saturation="tanh", saturation_knee=1.0),
        ]
        cr = CompiledRouter(["a", "b", "c"], edges, _SR, _DEV)
        src = torch.tensor([1.0 + 0j, 0.0, 0.0], dtype=torch.complex128)
        out = cr.step(src)
        assert torch.isfinite(out.abs()).all()
        assert cr.last_convergence_iters >= 1


# ══════════════════════════════════════════════════════════════════════
# CompiledRouter.from_graph — router_key and router_type filters
# ══════════════════════════════════════════════════════════════════════

class TestCompiledRouterFromGraph:
    def _graph(self):
        g = RoutingGraph()
        g.add_node("v1", source_router_types=["voice_router"], sink_router_types=[])
        g.add_node("d1", source_router_types=["driver_router"],
                   sink_router_types=["voice_router"])
        g.edges.append(RoutingEdge("v1", "d1", 1.0, router_key="vr1"))
        g.edges.append(RoutingEdge("d1", "v1", 0.5, router_key="dr1"))
        return g

    def test_router_key_filters_edges(self):
        g = self._graph()
        cr = CompiledRouter.from_graph(g, _SR, _DEV, router_key="vr1")
        assert "v1" in cr.node_keys
        assert "d1" in cr.node_keys
        assert cr.N == 2

    def test_router_type_filters_participation(self):
        g = self._graph()
        cr = CompiledRouter.from_graph(g, _SR, _DEV, router_type="voice_router")
        # Only the edge v1→d1 passes (v1 is source in voice_router, d1 is sink)
        # The edge d1→v1 fails (d1 is NOT source in voice_router)
        assert len(cr._sat_edges) == 0
        # d1 entry in M should have no contribution from d1→v1
        src = torch.tensor([1.0 + 0j, 0.0], dtype=torch.complex128)
        out = cr.step(src)
        # d1 receives from v1; v1 does NOT receive from d1
        assert abs(out[cr.ki["d1"]].item() - 1.0) < 1e-9
        assert abs(out[cr.ki["v1"]].item() - 1.0) < 1e-9  # just its own source

    def test_empty_graph_ok(self):
        g = RoutingGraph()
        cr = CompiledRouter.from_graph(g, _SR, _DEV)
        assert cr.N == 0

    def test_from_router_instance_applies_key_and_type_filters(self):
        g = RoutingGraph()
        g.add_node("v1", source_router_types=["voice_router"], sink_router_types=[])
        g.add_node("d1", source_router_types=["driver_router"],
                   sink_router_types=["voice_router"])
        g.add_node("m1", source_router_types=["master"], sink_router_types=["instrument"])
        g.edges.append(RoutingEdge("v1", "d1", 1.0, router_key="vr1"))
        g.edges.append(RoutingEdge("d1", "m1", 0.5, router_key="master1"))
        router = RouterInstance(key="vr1", router_type="voice_router", graph=g)

        cr = CompiledRouter.from_router_instance(router, _SR, _DEV)

        assert set(cr.node_keys) == {"v1", "d1"}
        assert cr.N == 2

    def test_lower_meta_edges_builds_bidirectional_directed_edges(self):
        g = RoutingGraph()
        g.add_node("score_sm")
        g.add_node("performer_sm")
        g.meta_edges.append(MetaEdge(
            a_key="score_sm",
            b_key="performer_sm",
            a_port="score_sm:score",
            b_port="performer_sm:score",
            semantic_role="score_bundle",
            a_to_b_transfer=EdgeTransferSpec(transfer_policy="scatter"),
            b_to_a_transfer=EdgeTransferSpec(transfer_policy="gather"),
        ))

        lowered = lower_meta_edges(g)

        assert len(lowered) == 2
        assert lowered[0].src_key == "score_sm"
        assert lowered[0].dst_key == "performer_sm"
        assert lowered[0].transfer_spec.transfer_policy == "scatter"
        assert lowered[1].src_key == "performer_sm"
        assert lowered[1].dst_key == "score_sm"
        assert lowered[1].transfer_spec.transfer_policy == "gather"

    def test_from_graph_includes_lowered_meta_edges(self):
        g = RoutingGraph()
        g.add_node("score_sm")
        g.add_node("performer_sm")
        g.meta_edges.append(MetaEdge(
            a_key="score_sm",
            b_key="performer_sm",
            a_port="score_sm:score",
            b_port="performer_sm:score",
        ))

        cr = CompiledRouter.from_graph(g, _SR, _DEV)

        assert "score_sm" in cr.node_keys
        assert "performer_sm" in cr.node_keys
        assert cr.N == 2
