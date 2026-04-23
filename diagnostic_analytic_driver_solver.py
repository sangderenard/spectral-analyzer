"""Diagnostic harness for the isolated complex graph solver."""
from __future__ import annotations

import json
import math
from typing import Any

import torch

from graph_solver import (
    DataPort,
    EdgeGroup,
    GraphSolver,
    KnobBinding,
    MixerArchetype,
    MixerNodeSpec,
    NodeLayerPresence,
    PortShapeSpec,
    PortSliceBinding,
    PortSet,
    RouterArchetype,
    RouterNodeSpec,
    SemanticPortContract,
    TensorEdge,
    TensorNode,
    WeightShapeBinding,
)

_CDTYPE = torch.complex128


def _w(theta: float, scale: float = 1.0) -> complex:
    return scale * complex(math.cos(theta), math.sin(theta))


def _saturate(x: torch.Tensor) -> torch.Tensor:
    mag = torch.tanh(x.abs())
    return mag * torch.exp(1j * x.angle())


def _build_linear_solver() -> GraphSolver:
    nodes = [
        TensorNode(
            "lin_a",
            layer="voice",
            layer_presence=NodeLayerPresence(
                input_layers=("voice",),
                output_layers=("voice", "master"),
                parameter_inputs=True,
            ),
            archetype=RouterArchetype(
                mixer_layer_participations=("voice", "master"),
                semantic_ports=(
                    SemanticPortContract(
                        key="voice_in",
                        label="Voice In",
                        direction="in",
                        domain="signal",
                        semantic_role="layer_forward",
                        allowed_shapes=(
                            PortShapeSpec("voice_packet", dims=("batch", "channel"), batch_dims=(0,), channel_dims=(1,)),
                        ),
                        icl_meta_sim="voice_wire_in",
                    ),
                    SemanticPortContract(
                        key="voice_out",
                        label="Voice Out",
                        direction="out",
                        domain="signal",
                        semantic_role="layer_forward",
                        allowed_shapes=(
                            PortShapeSpec("voice_packet", dims=("batch", "channel"), batch_dims=(0,), channel_dims=(1,)),
                        ),
                        weight_bindings=(
                            WeightShapeBinding(
                                key="voice_gain",
                                weight_dims=("channel",),
                                src_port_key="voice_out",
                                dst_port_key="sum_in",
                                slice_bindings=(
                                    PortSliceBinding(
                                        key="channel_to_channel",
                                        src_slice="[:]",
                                        dst_slice="[:]",
                                        weight_slice="[:]",
                                    ),
                                ),
                                semantic_role="per_channel_mix",
                            ),
                        ),
                        knob_bindings=(
                            KnobBinding(
                                key="voice_gain_knob",
                                param_path="routing.voice_gain",
                                semantic_role="patch_panel_param_link",
                                patch_visible=False,
                            ),
                        ),
                        icl_meta_sim="voice_wire_out",
                        mixer_grid_capable=True,
                    ),
                ),
                router=RouterNodeSpec(
                    port_sets=(
                        PortSet(
                            "voice_ports",
                            ports=(
                                DataPort("voice.in.main", direction="in", semantic_label="voice_in", data_label="analytic_signal"),
                                DataPort("voice.out.main", direction="out", semantic_label="voice_out", data_label="analytic_signal"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        TensorNode(
            "lin_b",
            layer="voice",
            layer_presence=NodeLayerPresence(input_layers=("voice",), output_layers=("master",)),
            archetype=RouterArchetype(
                mixer_layer_participations=("voice", "master"),
                semantic_ports=(
                    SemanticPortContract(
                        key="router_in",
                        label="Router In",
                        direction="in",
                        domain="signal",
                        semantic_role="addressed_receive",
                        allowed_shapes=(
                            PortShapeSpec("voice_packet", dims=("batch", "channel"), batch_dims=(0,), channel_dims=(1,)),
                        ),
                        icl_meta_sim="router_receiver",
                    ),
                    SemanticPortContract(
                        key="router_out",
                        label="Router Out",
                        direction="out",
                        domain="signal",
                        semantic_role="addressed_send",
                        allowed_shapes=(
                            PortShapeSpec("voice_packet", dims=("batch", "channel"), batch_dims=(0,), channel_dims=(1,)),
                        ),
                        icl_meta_sim="router_sender",
                    ),
                ),
                router=RouterNodeSpec(
                    port_sets=(
                        PortSet(
                            "router_ports",
                            ports=(
                                DataPort("router.in.a", direction="in", semantic_label="router_in", data_label="voice_packet"),
                                DataPort("router.out.a", direction="out", semantic_label="router_out", data_label="voice_packet"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        TensorNode(
            "lin_c",
            layer="master",
            layer_presence=NodeLayerPresence(input_layers=("voice", "master"), output_layers=("master",)),
            archetype=MixerArchetype(
                mixer_layer_participations=("master",),
                semantic_ports=(
                    SemanticPortContract(
                        key="sum_in",
                        label="Sum In",
                        direction="in",
                        domain="signal",
                        semantic_role="mix_input",
                        allowed_shapes=(
                            PortShapeSpec("mix_packet", dims=("batch", "channel"), batch_dims=(0,), channel_dims=(1,)),
                        ),
                        weight_bindings=(
                            WeightShapeBinding(
                                key="mix_matrix",
                                weight_dims=("channel",),
                                src_port_key="voice_out",
                                dst_port_key="sum_in",
                                slice_bindings=(
                                    PortSliceBinding(
                                        key="sum_all_channels",
                                        src_slice="[:]",
                                        dst_slice="[:]",
                                        weight_slice="[:]",
                                    ),
                                ),
                                semantic_role="mix_matrix_binding",
                            ),
                        ),
                        knob_bindings=(
                            KnobBinding(
                                key="print_gain",
                                param_path="mixer.print_gain",
                                semantic_role="mixer_coefficient",
                                patch_visible=False,
                            ),
                            KnobBinding(
                                key="print_pan",
                                param_path="mixer.print_pan",
                                semantic_role="mixer_coefficient",
                                patch_visible=False,
                            ),
                        ),
                        icl_meta_sim="master_bus_sum",
                        mixer_grid_capable=True,
                    ),
                    SemanticPortContract(
                        key="mix_out",
                        label="Mix Out",
                        direction="out",
                        domain="signal",
                        semantic_role="mix_output",
                        allowed_shapes=(
                            PortShapeSpec("mix_packet", dims=("batch", "channel"), batch_dims=(0,), channel_dims=(1,)),
                        ),
                        icl_meta_sim="master_bus_out",
                    ),
                ),
                mixer=MixerNodeSpec(),
            ),
        ),
    ]
    edges = [
        TensorEdge("lin_a", "lin_b", _w(0.12, 0.35), group="voice_bus", semantic_role="feedforward"),
        TensorEdge("lin_b", "lin_c", _w(-0.08, 0.28), group="print_bus", semantic_role="feedforward"),
        TensorEdge("lin_c", "lin_a", _w(0.17, 0.18), group="feedback_bus", semantic_role="feedback"),
        TensorEdge(
            "lin_a",
            "lin_c",
            torch.tensor([_w(0.04, 0.11), _w(-0.09, 0.07)], dtype=_CDTYPE),
            group="print_bus",
            semantic_role="parallel_send",
        ),
    ]
    edge_groups = [
        EdgeGroup("voice_bus", semantic_role="voice_layer_transfer", edge_refs=(("lin_a", "lin_b"),)),
        EdgeGroup("print_bus", semantic_role="master_layer_transfer", edge_refs=(("lin_b", "lin_c"), ("lin_a", "lin_c"))),
        EdgeGroup("feedback_bus", semantic_role="cyclic_return", edge_refs=(("lin_c", "lin_a"),)),
    ]
    edges.append(
        TensorEdge(
            "lin_a",
            "lin_c",
            1.0 + 0.0j,
            group="print_meta",
            semantic_role="bundle_link",
            src_port_set="voice_ports",
            dst_port_set="mix_ports",
            src_port="voice.out.main",
            dst_port="master.in.sum",
            contract_key="voice_to_master_patch",
            contract_semantic_role="cross_layer_feedforward",
            src_mask=torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=_CDTYPE),
            dst_mask=torch.tensor([1.0 + 0.0j, 1.0 + 0.0j], dtype=_CDTYPE),
            activity_mask=torch.tensor([1.0 + 0.0j, 0.5 + 0.0j], dtype=_CDTYPE),
            activity_contract="masked_bundle",
            src_addresses=("voice.left", "voice.right"),
            dst_addresses=("master.sum_l", "master.sum_r"),
        )
    )
    return GraphSolver(nodes, edges, edge_groups=edge_groups)


def _run_linear_exact_check() -> dict[str, Any]:
    solver = _build_linear_solver()
    work_shape = (2, 3, 2)
    src_a = torch.full(work_shape, 0.0j, dtype=_CDTYPE)
    src_b = torch.full(work_shape, 0.0j, dtype=_CDTYPE)
    src_c = torch.full(work_shape, 0.0j, dtype=_CDTYPE)
    for i in range(work_shape[0]):
        for j in range(work_shape[1]):
            phase = 2.0 * math.pi * (i + 2 * j) / 7.0
            src_a[i, j] = torch.tensor([complex(math.cos(phase), math.sin(phase)), complex(0.2, -0.1)], dtype=_CDTYPE)
            src_b[i, j] = torch.tensor([complex(0.1 * (i + 1), 0.05 * (j + 1)), complex(-0.03, 0.08)], dtype=_CDTYPE)

    out = solver.step({"lin_a": src_a, "lin_b": src_b, "lin_c": src_c})

    weights = {
        ("lin_b", "lin_a"): _w(0.12, 0.35),
        ("lin_c", "lin_b"): _w(-0.08, 0.28),
        ("lin_a", "lin_c"): _w(0.17, 0.18),
        ("lin_c", "lin_a"): (
            torch.tensor([_w(0.04, 0.11), _w(-0.09, 0.07)], dtype=_CDTYPE)
            + torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=_CDTYPE)
        ),
    }
    M = torch.zeros(work_shape + (3, 3), dtype=_CDTYPE)
    M[...] = torch.eye(3, dtype=_CDTYPE)
    M[..., 1, 0] = M[..., 1, 0] - weights[("lin_b", "lin_a")]
    M[..., 2, 1] = M[..., 2, 1] - weights[("lin_c", "lin_b")]
    M[..., 0, 2] = M[..., 0, 2] - weights[("lin_a", "lin_c")]
    M[..., 2, 0] = M[..., 2, 0] - weights[("lin_c", "lin_a")]

    b = torch.stack([src_a, src_b, src_c], dim=-1)
    expected = torch.linalg.solve(M, b.unsqueeze(-1)).squeeze(-1)
    got = torch.stack([out["lin_a"], out["lin_b"], out["lin_c"]], dim=-1)
    return {
        "work_shape": list(work_shape),
        "max_abs_error": float((got - expected).abs().max().item()),
    }


def _run_metadata_check() -> dict[str, Any]:
    solver = _build_linear_solver()
    return {
        "meta_edge_count": len(solver.meta_edges),
        "edge_group_count": len(solver.edge_groups),
        "edge_group_keys": [group.key for group in solver.edge_groups],
        "node_forms": {
            node.key: {
                "is_mixer_archetype": isinstance(node.archetype, MixerArchetype),
                "is_router_archetype": isinstance(node.archetype, RouterArchetype),
            }
            for node in solver.nodes
        },
        "mixer_layer_participations": {
            node.key: list(node.archetype.mixer_layer_participations)
            for node in solver.nodes
            if node.archetype.mixer_layer_participations
        },
        "semantic_ports": {
            node.key: [
                {
                    "key": port.key,
                    "direction": port.direction,
                    "domain": port.domain,
                    "allowed_shapes": [list(shape.dims) for shape in port.allowed_shapes],
                    "batch_dims": [list(shape.batch_dims) for shape in port.allowed_shapes],
                    "channel_dims": [list(shape.channel_dims) for shape in port.allowed_shapes],
                    "weight_bindings": [
                        {
                            "key": wb.key,
                            "weight_dims": list(wb.weight_dims),
                            "src_port_key": wb.src_port_key,
                            "dst_port_key": wb.dst_port_key,
                            "slice_bindings": [
                                {
                                    "src_slice": sb.src_slice,
                                    "dst_slice": sb.dst_slice,
                                    "weight_slice": sb.weight_slice,
                                }
                                for sb in wb.slice_bindings
                            ],
                        }
                        for wb in port.weight_bindings
                    ],
                    "knob_bindings": [
                        {
                            "key": kb.key,
                            "param_path": kb.param_path,
                            "patch_visible": kb.patch_visible,
                        }
                        for kb in port.knob_bindings
                    ],
                    "icl_meta_sim": port.icl_meta_sim,
                    "mixer_grid_capable": port.mixer_grid_capable,
                }
                for port in node.archetype.semantic_ports
            ]
            for node in solver.nodes
            if node.archetype.semantic_ports
        },
        "mixer_nodes": {
            node.key: {
                "parameter_matrix_keys": list(node.archetype.mixer.parameter_matrix_keys),
                "icl_model": node.archetype.mixer.icl_model,
                "physical_presence": node.archetype.mixer.physical_presence,
            }
            for node in solver.nodes
            if isinstance(node.archetype, MixerArchetype)
        },
        "router_nodes": {
            node.key: {
                "port_sets": [port_set.key for port_set in node.archetype.router.port_sets],
                "retains_identity": bool(node.archetype.router.retains_identity),
                "addressed": bool(node.archetype.router.addressed),
            }
            for node in solver.nodes
            if isinstance(node.archetype, RouterArchetype)
        },
        "transport_sides": {
            "grid_edge_count": len([edge for edge in solver.edges if not edge.contract_key]),
            "patch_link_count": len([edge for edge in solver.edges if edge.contract_key]),
            "network_link_count": len(solver.network_links),
            "patch_contract_keys": [edge.contract_key for edge in solver.edges if edge.contract_key],
        },
        "layer_plan": {
            "signal_layers": list(solver.solve_plan.signal_layers),
            "parameter_layer": solver.solve_plan.parameter_layer.key,
            "parameter_any_in": solver.solve_plan.parameter_layer.accepts_from_any_layer,
            "parameter_any_out": solver.solve_plan.parameter_layer.emits_to_any_layer,
            "control_feedback_iterations": solver.solve_plan.control_feedback_iterations,
            "cycle_entire_layer_stack": solver.solve_plan.cycle_entire_layer_stack,
            "one_network_per_sample": solver.solve_plan.one_network_per_sample,
        },
        "node_layer_presence": {
            node.key: {
                "inputs": list(node.layer_presence.input_layers),
                "outputs": list(node.layer_presence.output_layers),
                "parameter_inputs": bool(node.layer_presence.parameter_inputs),
                "parameter_outputs": bool(node.layer_presence.parameter_outputs),
            }
            for node in solver.nodes
        },
    }


def _build_known_answer_solver() -> GraphSolver:
    nodes = [
        TensorNode("known_a", layer="voice"),
        TensorNode("known_b", layer="master"),
    ]
    edges = [
        TensorEdge("known_a", "known_b", _w(0.31, 0.23)),
        TensorEdge("known_b", "known_a", torch.tensor([_w(-0.22, 0.17), _w(0.14, 0.09)], dtype=_CDTYPE)),
    ]
    return GraphSolver(nodes, edges)


def _run_concurrent_known_answer_check() -> dict[str, Any]:
    solver = _build_known_answer_solver()
    work_shape = (3, 2, 2)
    src_a = torch.zeros(work_shape, dtype=_CDTYPE)
    src_b = torch.zeros(work_shape, dtype=_CDTYPE)
    for i in range(work_shape[0]):
        for j in range(work_shape[1]):
            phase = 2.0 * math.pi * (2 * i + j) / 11.0
            src_a[i, j] = torch.tensor(
                [complex(0.4 * math.cos(phase), 0.4 * math.sin(phase)), complex(0.05 * (i + 1), -0.04 * (j + 1))],
                dtype=_CDTYPE,
            )
            src_b[i, j] = torch.tensor(
                [complex(-0.12 * (i + 1), 0.08 * (j + 1)), complex(0.18 * math.cos(0.5 * phase), 0.18 * math.sin(0.5 * phase))],
                dtype=_CDTYPE,
            )

    out = solver.step({"known_a": src_a, "known_b": src_b})

    w_ab = torch.tensor(_w(0.31, 0.23), dtype=_CDTYPE)
    w_ba = torch.tensor([_w(-0.22, 0.17), _w(0.14, 0.09)], dtype=_CDTYPE)
    denom = 1.0 - (w_ab * w_ba)
    expected_a = (src_a + w_ba * src_b) / denom
    expected_b = src_b + (w_ab * expected_a)

    return {
        "work_shape": list(work_shape),
        "max_abs_error_a": float((out["known_a"] - expected_a).abs().max().item()),
        "max_abs_error_b": float((out["known_b"] - expected_b).abs().max().item()),
        "reference_sample": {
            "a": [str(v) for v in expected_a[0, 0].tolist()],
            "b": [str(v) for v in expected_b[0, 0].tolist()],
        },
        "solver_sample": {
            "a": [str(v) for v in out["known_a"][0, 0].tolist()],
            "b": [str(v) for v in out["known_b"][0, 0].tolist()],
        },
    }


def _build_nonlinear_solver(sample_rate: float = 48_000.0) -> GraphSolver:
    nodes = [
        TensorNode("src_a", layer="voice"),
        TensorNode("src_b", layer="voice"),
        TensorNode("cyc_a", layer="driver", transform=_saturate, natural_rate_hz=12_000.0),
        TensorNode("cyc_b", layer="instrument", transform=_saturate, natural_rate_hz=8_000.0),
        TensorNode("mix", layer="room"),
        TensorNode("out", layer="master"),
    ]
    edges = [
        TensorEdge("src_a", "cyc_a", _w(0.11, 0.75)),
        TensorEdge("src_b", "cyc_b", _w(-0.07, 0.62)),
        TensorEdge("cyc_a", "cyc_b", _w(0.24, 0.31)),
        TensorEdge("cyc_b", "cyc_a", _w(-0.19, 0.27)),
        TensorEdge("cyc_a", "mix", torch.tensor([_w(0.0, 0.7), _w(0.2, 0.35)], dtype=_CDTYPE)),
        TensorEdge("cyc_b", "mix", torch.tensor([_w(0.15, 0.3), _w(-0.1, 0.22)], dtype=_CDTYPE), delay_s=1.0 / sample_rate),
        TensorEdge("src_b", "mix", torch.tensor([_w(0.0, 0.12), _w(0.3, 0.11)], dtype=_CDTYPE)),
        TensorEdge("mix", "out", torch.tensor([_w(0.0, 0.9), _w(0.08, 0.82)], dtype=_CDTYPE)),
    ]
    return GraphSolver(
        nodes,
        edges,
        sample_rate=sample_rate,
        default_k_max=10,
        convergence_eps=1e-11,
        use_z_cache=True,
        interp_mode="polar",
        timing_offset=0.25,
    )


def _run_nonlinear_cycle(samples: int = 5) -> dict[str, Any]:
    solver = _build_nonlinear_solver()
    work_shape = (2, 3, 2)
    traces: list[dict[str, float]] = []
    for step in range(samples):
        src_a = torch.zeros(work_shape, dtype=_CDTYPE)
        src_b = torch.zeros(work_shape, dtype=_CDTYPE)
        for i in range(work_shape[0]):
            for j in range(work_shape[1]):
                phase = 2.0 * math.pi * (step + i + 2 * j) / 9.0
                src_a[i, j] = torch.tensor(
                    [complex(0.35 * math.cos(phase), 0.35 * math.sin(phase)), complex(0.18, -0.05)],
                    dtype=_CDTYPE,
                )
                src_b[i, j] = torch.tensor(
                    [complex(0.22 * math.cos(0.7 * phase), 0.22 * math.sin(0.7 * phase)), complex(-0.09, 0.14)],
                    dtype=_CDTYPE,
                )
        out = solver.step({"src_a": src_a, "src_b": src_b})
        traces.append({
            "step": step,
            "out_mean_mag": float(out["out"].abs().mean().item()),
            "mix_mean_mag": float(out["mix"].abs().mean().item()),
            "cyc_a_mean_mag": float(out["cyc_a"].abs().mean().item()),
            "cyc_b_mean_mag": float(out["cyc_b"].abs().mean().item()),
        })

    return {
        "work_shape": list(work_shape),
        "samples": samples,
        "tier": solver.condensed.tier,
        "nonlinear_sccs": len(solver.cyclic_blocks),
        "trace": traces,
    }


def run_diagnostic() -> dict[str, Any]:
    return {
        "solver_contract": {
            "dtype": str(_CDTYPE),
            "arbitrary_node_tensors": True,
            "broadcast_edge_weights": True,
            "scalar_node_assumption": False,
        },
        "metadata_check": _run_metadata_check(),
        "concurrent_known_answer_check": _run_concurrent_known_answer_check(),
        "linear_exact_check": _run_linear_exact_check(),
        "nonlinear_cycle_check": _run_nonlinear_cycle(),
    }


def main() -> None:
    print(json.dumps(run_diagnostic(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
