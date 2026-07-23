from __future__ import annotations

import torch

from graph_solver import GraphSolver, TensorEdge, TensorNode
from network_materializer import compile_nodes, routing_edge_to_tensor_edge
from routing_engine import RoutingEdge


_CDTYPE = torch.complex128


def test_zero_delay_cycle_keeps_exact_algebraic_solve() -> None:
    solver = GraphSolver(
        nodes=[TensorNode("a"), TensorNode("b")],
        edges=[
            TensorEdge("a", "b", weight=0.5),
            TensorEdge("b", "a", weight=0.25),
        ],
    )

    out = solver.step({"a": torch.tensor(1.0 + 0.0j, dtype=_CDTYPE)})

    # a = 1 + .25b, b = .5a
    assert torch.allclose(out["a"], torch.tensor(8.0 / 7.0, dtype=_CDTYPE))
    assert torch.allclose(out["b"], torch.tensor(4.0 / 7.0, dtype=_CDTYPE))
    assert not solver.delayed_edges_by_steps


def test_feed_forward_delay_stays_vectorized_over_schedule_window() -> None:
    calls: list[tuple[int, ...]] = []

    def double(value: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(value.shape))
        return value * 2.0

    solver = GraphSolver(
        nodes=[TensorNode("src"), TensorNode("dst", transform=double)],
        edges=[TensorEdge("src", "dst", weight=1.0, delay_samples=1)],
    )
    src = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2).to(_CDTYPE)

    out = solver.run_schedule({"src": src}, n_frames=3)

    assert calls == [(2, 3, 2)]
    assert torch.allclose(out["dst"][:, 0, :], torch.zeros((2, 2), dtype=_CDTYPE))
    assert torch.allclose(out["dst"][:, 1:, :], src[:, :-1, :] * 2.0)
    assert not solver._schedule_has_delayed_feedback


def test_delayed_feedback_rolls_forward_causally() -> None:
    solver = GraphSolver(
        nodes=[TensorNode("a"), TensorNode("b")],
        edges=[
            TensorEdge("a", "b", weight=1.0),
            TensorEdge("b", "a", weight=0.5, delay_samples=1),
        ],
    )
    impulse = torch.tensor([1.0, 0.0, 0.0], dtype=_CDTYPE)

    out = solver.run_schedule({"a": impulse}, n_frames=3)

    expected = torch.tensor([1.0, 0.5, 0.25], dtype=_CDTYPE).reshape(1, 3, 1)
    assert torch.allclose(out["a"], expected)
    assert torch.allclose(out["b"], expected)
    assert solver._schedule_has_delayed_feedback


def test_complex_delay_factor_and_materialized_delay_survive_boundary() -> None:
    routed = RoutingEdge(
        src_key="field_in",
        dst_key="field_out",
        weight=2.0,
        angle_rad=0.0,
        delay_s=0.25,
    )
    edge = routing_edge_to_tensor_edge(routed)
    phased = TensorEdge(
        "field_in",
        "field_out",
        weight=edge.weight,
        delay_s=edge.delay_s,
        analog_complex_delay=1.0j,
    )

    assert edge.delay_steps(8.0) == 2
    value = torch.tensor(3.0 + 0.0j, dtype=_CDTYPE)
    assert torch.allclose(phased.apply(value), torch.tensor(0.0 + 6.0j, dtype=_CDTYPE))


def test_compiled_graph_is_a_named_complex_context_surface() -> None:
    compiled = compile_nodes(
        [TensorNode("field_in"), TensorNode("field_out")],
        [TensorEdge("field_in", "field_out", weight=1.0j, delay_samples=1)],
        sample_rate=8.0,
        output_node_keys=["field_out"],
    )
    field = torch.tensor([1.0, 2.0, 3.0], dtype=_CDTYPE)

    products = compiled.process_window({"field_in": field}, n_frames=3)
    contract = compiled.transport_contract()

    expected = torch.tensor([0.0, 1.0j, 2.0j], dtype=_CDTYPE).reshape(1, 3, 1)
    assert torch.allclose(products["field_out"], expected)
    assert contract["schema"] == "complex-network-context-v1"
    assert contract["representation"] == "torch.complex128"
    assert contract["delayed_edges_by_samples"] == {"1": 1}
    assert contract["schedule"] == "vectorized_window"
