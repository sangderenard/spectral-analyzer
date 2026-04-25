import types

import torch

from analytic_model import (
    AnalyticModule,
    AnalyticPatch,
    AnalyticVoice,
    ControlSlider,
    ControlSurface,
    LFODefinition,
    ModRouting,
    ParamNode,
)
from graph_solver import DebugArchetype, GraphSolver, TensorEdge, TensorNode, _CDTYPE
from granular_engine import GrainPopulationSpec
from network_materializer import LFOTorchNode, compile_network, materialize_network
from parametric_curve import default_chirp, default_envelope
from routing_engine import RoutingEdge
from voice_graph_node import VoiceTorchOscillator, build_voice_mixer_network


class _ContractModule(torch.nn.Module):
    pass


def _make_voice():
    return types.SimpleNamespace(
        key="voice",
        freq_hz=440.0,
        amplitude=1.0,
        phase_origin=0.0,
        semitone_offset=0.0,
        harmonic_brightness=1.0,
        harmonic_warp_strength=0.0,
        fm=None,
        am=None,
        manifold_type="pure",
        harmonic_count=8,
        loop_enabled=False,
        loop_start=0.1,
        loop_end=0.9,
        emission_mode="single",
        pre_delay=0.0,
        envelope=None,
        chirp=None,
    )


def test_acyclic_nonlinear_singleton_bypasses_cyclic_block() -> None:
    calls = 0

    def _nonlinear(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return x + torch.tensor(1.0 + 0.0j, dtype=_CDTYPE)

    solver = GraphSolver(
        nodes=[
            TensorNode("src", layer="signal"),
            TensorNode("nonlin", layer="signal", transform=_nonlinear),
        ],
        edges=[TensorEdge("src", "nonlin", weight=1.0 + 0.0j)],
    )

    assert len(solver.cyclic_blocks) == 0

    out = solver.step({"src": torch.tensor(2.0 + 0.0j, dtype=_CDTYPE)})

    assert calls == 1
    assert torch.allclose(out["nonlin"], torch.tensor(3.0 + 0.0j, dtype=_CDTYPE))


def test_schedule_payload_preserves_batch_time_channel_axes() -> None:
    def _double(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0

    solver = GraphSolver(
        nodes=[
            TensorNode("src", layer="signal"),
            TensorNode("dst", layer="signal", transform=_double),
        ],
        edges=[TensorEdge("src", "dst", weight=1.0 + 0.0j, delay_samples=1)],
    )

    src = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2).to(_CDTYPE)
    out = solver.run_schedule({"src": src}, n_frames=3)

    assert out["src"].shape == (2, 3, 2)
    assert out["dst"].shape == (2, 3, 2)
    assert torch.allclose(out["dst"][:, 0, :], torch.zeros(2, 2, dtype=_CDTYPE))
    assert torch.allclose(out["dst"][:, 1:, :], src[:, :-1, :] * 2.0)


def test_lateral_archetype_accounting_persists_and_drives_schedule_batch() -> None:
    arch = DebugArchetype()
    solver = GraphSolver(
        nodes=[
            TensorNode("a", transform=lambda x: x, archetype_key="debug"),
            TensorNode("b", transform=lambda x: x, archetype_key="debug"),
            TensorNode("mix"),
        ],
        edges=[
            TensorEdge("a", "mix", weight=1.0 + 0.0j),
            TensorEdge("b", "mix", weight=1.0 + 0.0j),
        ],
        archetypes={"debug": arch},
    )

    assert ("debug", 0) in solver.lateral_accounting.archetype_groups
    assert len(solver.lateral_accounting.grouped_positions) == 2

    out = solver.run_schedule({}, n_frames=4)

    assert out["a"].shape == (1, 4, 1)
    assert arch.hit_counts.get("a", 0) == 1
    assert arch.hit_counts.get("b", 0) == 1


def test_subscription_contract_edges_do_not_enter_tensor_solve() -> None:
    consumer_mod = _ContractModule()
    solver = GraphSolver(
        nodes=[
            TensorNode(
                "score",
                subscription_ports=("score_out",),
                subscription_contracts={"score_out": {"kind": "score_producer"}},
            ),
            TensorNode(
                "voice",
                analytic_module=consumer_mod,
                subscription_ports=("score_in",),
                subscription_contracts={
                    "score_in": {
                        "kind": "voice_score_consumer",
                        "groups": ("lead",),
                        "aggregation": "mask",
                    }
                },
            ),
        ],
        edges=[
            TensorEdge(
                "score",
                "voice",
                weight=0.0 + 0.0j,
                semantic_role="score_bundle",
                src_port="score_out",
                dst_port="score_in",
            )
        ],
    )

    assert len(solver.contract_edges) == 1
    contract_edge = solver.contract_edges[0]
    assert contract_edge.src_key == "score"
    assert contract_edge.dst_key == "voice"
    assert contract_edge.group_mask == ("lead",)
    assert contract_edge.contract["dst"]["kind"] == "voice_score_consumer"
    assert "score_in" in consumer_mod.subscription_ports
    assert consumer_mod.subscription_ports["score_in"]["contract_edges_in"] == [contract_edge]

    out = solver.run_schedule({}, n_frames=2)

    assert out["score"].shape == (1, 2, 1)
    assert out["voice"].shape == (1, 2, 1)


def test_voice_nodes_publish_score_consumer_curve_contract() -> None:
    voice = _make_voice()
    solver, voice_nodes, _ = build_voice_mixer_network([voice], sample_rate=10.0, duration=1.0)
    vnode = voice_nodes["voice"]

    payload = solver.subscription_ports[("voice_out", "score_in")].payload
    contract = payload["contract"]

    assert contract["voice_key"] == "voice"
    assert contract["envelope_curve"] is vnode.oscillator.envelope_curve
    assert contract["chirp_curve"] is vnode.oscillator.chirp_curve
    assert contract["aggregation"] == "mask"


def test_voice_archetype_batches_same_level_curve_evaluation() -> None:
    voice = _make_voice()
    voice_b = _make_voice()
    voice_b.key = "voice_b"
    solver, voice_nodes, _ = build_voice_mixer_network(
        [voice, voice_b],
        sample_rate=10.0,
        duration=1.0,
        voice_port_occupancy={"voice": ("pitch_in",), "voice_b": ("pitch_in",)},
    )

    seen_shapes: list[tuple[int, ...]] = []
    for vnode in voice_nodes.values():
        orig_env = vnode.oscillator.envelope_curve.evaluate_normalized

        def _counted_env(t_norm: torch.Tensor, _orig=orig_env) -> torch.Tensor:
            seen_shapes.append(tuple(t_norm.shape))
            return _orig(t_norm)

        vnode.oscillator.envelope_curve.evaluate_normalized = _counted_env  # type: ignore[method-assign]

    solver.run_schedule(
        {
            "voice_pitch_in": torch.full((1, 3, 1), 220.0 + 0.0j, dtype=_CDTYPE),
            "voice_b_pitch_in": torch.full((1, 3, 1), 330.0 + 0.0j, dtype=_CDTYPE),
        },
        n_frames=3,
    )

    assert seen_shapes == [(1, 3, 1), (1, 3, 1)]


def test_voice_oscillator_reuses_scalar_curve_evaluations_within_sample() -> None:
    osc = VoiceTorchOscillator(sample_rate=48_000.0, duration=1.0, fm_depth_hz=2.0)
    osc.gate_on(0.0, 1.0)
    osc.set_sample(12)

    env_calls = 0
    chirp_calls = 0
    orig_env = osc.envelope_curve.evaluate_normalized
    orig_chirp = osc.chirp_curve.evaluate_normalized

    def _counted_env(t_norm: torch.Tensor) -> torch.Tensor:
        nonlocal env_calls
        env_calls += 1
        return orig_env(t_norm)

    def _counted_chirp(t_norm: torch.Tensor) -> torch.Tensor:
        nonlocal chirp_calls
        chirp_calls += 1
        return orig_chirp(t_norm)

    osc.envelope_curve.evaluate_normalized = _counted_env  # type: ignore[method-assign]
    osc.chirp_curve.evaluate_normalized = _counted_chirp  # type: ignore[method-assign]

    t_abs = torch.tensor([12.0 / 48_000.0], dtype=torch.float64)
    x_a = torch.tensor(0.1 + 0.0j, dtype=_CDTYPE)
    x_b = torch.tensor(0.2 + 0.0j, dtype=_CDTYPE)

    osc.forward(x_a, t_abs=t_abs)
    osc.forward(x_b, t_abs=t_abs)

    assert env_calls == 1
    assert chirp_calls == 1

    osc.set_sample(13)
    osc.forward(torch.tensor(0.3 + 0.0j, dtype=_CDTYPE), t_abs=torch.tensor([13.0 / 48_000.0], dtype=torch.float64))

    assert env_calls == 2
    assert chirp_calls == 2


def test_voice_network_solves_voice_out_through_archetype_without_feedback() -> None:
    voice = _make_voice()
    solver, voice_nodes, _ = build_voice_mixer_network(
        [voice],
        sample_rate=48_000.0,
        duration=1.0,
        voice_port_occupancy={"voice": ("pitch_in",)},
    )
    vnode = voice_nodes["voice"]
    vnode.gate_on(0.0, 1.0)
    vnode.set_sample(0)

    calls = 0
    orig_forward = vnode.oscillator.forward

    def _counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return orig_forward(*args, **kwargs)

    vnode.oscillator.forward = _counted_forward  # type: ignore[method-assign]
    solver.step({"voice_pitch_in": torch.tensor(440.0 + 0.0j, dtype=_CDTYPE)})

    assert calls == 0


def test_voice_network_omits_unoccupied_helper_ports() -> None:
    solver, _, _ = build_voice_mixer_network(
        [_make_voice()],
        sample_rate=48_000.0,
        duration=1.0,
    )

    assert set(solver.node_keys) == {"voice_out", "mix_out"}
    assert {(edge.src_key, edge.dst_key) for edge in solver.edges} == {("voice_out", "mix_out")}


def test_voice_oscillator_chirp_curve_modulates_instantaneous_frequency() -> None:
    sr = 8_000.0
    duration = 0.25
    env_curve = default_envelope("flat_env")
    for pt in env_curve.points:
        pt.v = 1.0

    chirp_curve = default_chirp("sweep")
    chirp_curve.points.clear()
    chirp_curve.add_point(0.0, 0.0)
    chirp_curve.add_point(1.0, 1.0)

    osc = VoiceTorchOscillator(
        freq_hz=440.0,
        amplitude=1.0,
        harmonic_count=1,
        manifold_type="pure",
        envelope_curve=env_curve,
        chirp_curve=chirp_curve,
        sample_rate=sr,
        duration=duration,
    )
    osc.gate_on(0.0, 1.0)

    n = int(sr * duration)
    t_abs = torch.arange(n, dtype=torch.float64) / sr
    y = osc.forward(torch.zeros(n, dtype=_CDTYPE), t_abs=t_abs)

    phase_step = torch.angle(y[1:] * torch.conj(y[:-1]))
    inst_hz = phase_step * (sr / (2.0 * torch.pi))

    start_mean = float(inst_hz[:256].mean())
    end_mean = float(inst_hz[-256:].mean())

    assert start_mean < 320.0
    assert end_mean > 560.0


def test_voice_oscillator_pre_delay_buffer_masks_early_samples() -> None:
    sr = 10.0
    env_curve = default_envelope("flat_env")
    for pt in env_curve.points:
        pt.v = 1.0

    osc = VoiceTorchOscillator(
        freq_hz=1.0,
        amplitude=1.0,
        harmonic_count=1,
        envelope_curve=env_curve,
        pre_delay=0.3,
        sample_rate=sr,
        duration=1.0,
    )
    osc.gate_on(0.0, 1.0)
    t_abs = torch.arange(6, dtype=torch.float64) / sr
    y = osc.forward(torch.zeros(6, dtype=_CDTYPE), t_abs=t_abs)

    assert torch.allclose(y[:3], torch.zeros(3, dtype=_CDTYPE))
    assert torch.any(y[3:].abs() > 0.0)
    assert torch.isclose(osc.pre_delay_s, torch.tensor(0.3, dtype=torch.float64))


def test_voice_oscillator_loop_tiling_repeats_cached_loop_region() -> None:
    sr = 10.0
    env_curve = default_envelope("ramp_env")
    env_curve.points.clear()
    env_curve.add_point(0.0, 0.0)
    env_curve.add_point(1.0, 1.0)

    osc = VoiceTorchOscillator(
        freq_hz=1.0,
        amplitude=1.0,
        harmonic_count=1,
        envelope_curve=env_curve,
        loop_enabled=True,
        loop_start=0.2,
        loop_end=0.5,
        sample_rate=sr,
        duration=1.0,
    )
    osc.gate_on(0.0, 1.0)

    samples = []
    for idx in range(6):
        osc.set_sample(idx)
        samples.append(osc.forward(torch.tensor(0.0 + 0.0j, dtype=_CDTYPE)))

    mags = torch.stack(samples).abs()
    assert torch.allclose(mags[5], mags[2])
    assert not torch.allclose(mags[5], torch.tensor(0.5, dtype=mags.dtype))


def test_voice_oscillator_am_mod_and_trigger_update_runtime_state() -> None:
    env_curve = default_envelope("flat_env")
    for pt in env_curve.points:
        pt.v = 1.0

    base = VoiceTorchOscillator(
        freq_hz=220.0,
        amplitude=1.0,
        harmonic_count=1,
        envelope_curve=env_curve,
        am_depth_amp=1.0,
        sample_rate=100.0,
        duration=1.0,
    )
    mod = VoiceTorchOscillator(
        freq_hz=220.0,
        amplitude=1.0,
        harmonic_count=1,
        envelope_curve=env_curve,
        am_depth_amp=1.0,
        sample_rate=100.0,
        duration=1.0,
    )
    base.trigger(440.0, velocity=0.8, t_abs=0.0)
    mod.trigger(440.0, velocity=0.8, t_abs=0.0)

    y_base = base.forward(torch.tensor(0.0 + 0.0j, dtype=_CDTYPE), t_abs=torch.tensor([0.0], dtype=torch.float64))
    y_mod = mod.forward(
        torch.tensor(0.0 + 0.0j, dtype=_CDTYPE),
        t_abs=torch.tensor([0.0], dtype=torch.float64),
        am_mod=torch.tensor(1.0 + 0.0j, dtype=_CDTYPE),
    )

    assert torch.isclose(torch.exp(base.log_freq), torch.tensor(440.0, dtype=torch.float64))
    assert torch.allclose(y_mod.abs(), y_base.abs() * 2.0)


def test_materializer_emits_lfo_node_and_wires_voice_fm_port() -> None:
    patch = AnalyticPatch()
    patch.preview_sr = 4
    patch.duration = 1.0
    lfo = LFODefinition(key="lfo1", rate_hz=1.0, depth=1.0, shape="Sine")
    voice = AnalyticVoice(key="v1", freq_hz=20.0, fm=ModRouting(source_key="lfo1", depth_hz=5.0))
    patch.lfos = [lfo]
    patch.voices = [voice]

    nodes, edges = materialize_network(patch, sr=4.0)
    node_keys = {node.key for node in nodes}
    edge_pairs = {(edge.src_key, edge.dst_key, edge.semantic_role) for edge in edges}

    assert {"lfo1", "v1_out", "v1_fm", "v1"}.issubset(node_keys)
    assert ("lfo1", "v1_fm", "lfo_fm_source") in edge_pairs
    assert ("v1_out", "v1", "voice_alias") in edge_pairs

    lfo_node = next(node for node in nodes if node.key == "lfo1")
    assert isinstance(lfo_node.analytic_module, LFOTorchNode)
    first = lfo_node.transform(torch.tensor(0.0 + 0.0j, dtype=_CDTYPE))
    second = lfo_node.transform(torch.tensor(0.0 + 0.0j, dtype=_CDTYPE))
    assert torch.allclose(first.real, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)
    assert second.real > 0.9


def test_materializer_wires_param_node_targets_to_voice_mod_ports() -> None:
    patch = AnalyticPatch()
    voice = AnalyticVoice(key="v1", freq_hz=20.0)
    pn = ParamNode(
        key="p1",
        default_value=0.25,
        targets=[
            {"voice_key": "v1", "attr": "chirp.f_delta_start"},
            {"voice_key": "v1", "attr": "amplitude"},
        ],
    )
    patch.voices = [voice]
    patch.param_nodes = [pn]

    nodes, edges = materialize_network(patch, sr=48_000.0)
    node_keys = {node.key for node in nodes}
    edge_pairs = {(edge.src_key, edge.dst_key, edge.semantic_role) for edge in edges}

    assert {"p1", "v1_chirp_mod", "v1_env_mod"}.issubset(node_keys)
    assert ("p1", "v1_chirp_mod", "param_series_injection") in edge_pairs
    assert ("p1", "v1_env_mod", "param_series_injection") in edge_pairs

    pnode = next(node for node in nodes if node.key == "p1")
    out = pnode.transform(torch.tensor(0.0 + 0.0j, dtype=_CDTYPE))
    assert torch.allclose(out.real, torch.tensor(0.25, dtype=torch.float64))


def test_materializer_wraps_granular_voice_as_stateful_node() -> None:
    patch = AnalyticPatch()
    patch.preview_sr = 200
    patch.duration = 0.02
    voice = AnalyticVoice(
        key="g1",
        freq_hz=120.0,
        emission_mode="granular",
        granular=GrainPopulationSpec(
            center_frequency_hz=120.0,
            grain_density_hz=50.0,
            grain_duration_s=0.01,
            editor_seed=1,
        ),
    )
    patch.voices = [voice]

    nodes, _edges = materialize_network(patch, sr=200.0)
    node = next(node for node in nodes if node.key == "g1")

    assert node.analytic_module.__class__.__name__ == "GranularVoiceNode"
    sample = node.transform(torch.tensor(0.0 + 0.0j, dtype=_CDTYPE))
    assert sample.dtype == _CDTYPE


def test_compile_network_returns_solver_outputs_and_module_registry() -> None:
    patch = AnalyticPatch()
    patch.preview_sr = 100
    patch.system_audio.output_channels = 1
    patch.voices = [AnalyticVoice(key="v1", freq_hz=20.0)]
    patch.routing.edges = [RoutingEdge(src_key="v1", dst_key="__sys_out_1__", weight=1.0)]

    compiled = compile_network(patch, sr=100.0)

    assert compiled.output_node_keys == ["__sys_out_1__"]
    assert "v1_out" in compiled.module_registry
    out = compiled.solver.step({})
    assert "__sys_out_1__" in out


def test_materializer_expands_polyphonic_voice_instances_and_sums_alias() -> None:
    patch = AnalyticPatch()
    patch.voices = [AnalyticVoice(key="vpoly", freq_hz=40.0, polyphony_count=3)]

    nodes, edges = materialize_network(patch, sr=100.0)
    node_keys = {node.key for node in nodes}
    sum_edges = [
        edge for edge in edges
        if edge.dst_key == "vpoly" and edge.semantic_role == "voice_poly_sum"
    ]

    assert {"vpoly", "vpoly__poly1_out", "vpoly__poly2_out", "vpoly__poly3_out"}.issubset(node_keys)
    assert len(sum_edges) == 3


def test_materializer_emits_module_lfo_channels_and_control_slider_sources() -> None:
    patch = AnalyticPatch()
    mod = AnalyticModule(key="mlfo", module_type="lfo")
    mod.lfo_channels = [
        {"rate_hz": 1.0, "amplitude": 0.5, "shape": "Sine", "phase_offset": 0.0},
        {"rate_hz": 2.0, "amplitude": 0.25, "shape": "Square", "phase_offset": 0.0},
    ]
    slider = ControlSlider(key="ctrl1", value=0.25, low=2.0, high=6.0)
    patch.modules = [mod]
    patch.controls = [ControlSurface(key="cs1", sliders=[slider])]

    nodes, edges = materialize_network(patch, sr=4.0)
    node_keys = {node.key for node in nodes}
    edge_pairs = {(edge.src_key, edge.dst_key, edge.semantic_role) for edge in edges}

    assert {"mlfo", "mlfo_lfoch0", "mlfo_lfoch1", "ctrl1"}.issubset(node_keys)
    assert ("mlfo_lfoch0", "mlfo", "module_lfo_channel_sum") in edge_pairs
    assert ("mlfo_lfoch1", "mlfo", "module_lfo_channel_sum") in edge_pairs
    ctrl_node = next(node for node in nodes if node.key == "ctrl1")
    assert torch.allclose(ctrl_node.transform(torch.tensor(0.0 + 0.0j, dtype=_CDTYPE)).real, torch.tensor(3.0, dtype=torch.float64))
    assert isinstance(ctrl_node.analytic_module.value, torch.nn.Parameter)


def test_materializer_fans_explicit_voice_helper_routes_to_polyphonic_instances() -> None:
    patch = AnalyticPatch()
    patch.voices = [AnalyticVoice(key="vpoly", freq_hz=40.0, polyphony_count=2)]
    patch.lfos = [LFODefinition(key="lfo1", rate_hz=2.0, depth=1.0)]
    patch.routing.edges = [RoutingEdge(src_key="lfo1", dst_key="vpoly_fm", weight=0.5)]

    _nodes, edges = materialize_network(patch, sr=100.0)
    routed = {
        (edge.src_key, edge.dst_key, float(edge.weight.real))
        for edge in edges
        if edge.src_key == "lfo1" and edge.dst_key.endswith("_fm")
    }

    assert routed == {
        ("lfo1", "vpoly__poly1_fm", 0.5),
        ("lfo1", "vpoly__poly2_fm", 0.5),
    }


def test_materializer_emits_pitch_quantizer_and_interaural_transforms() -> None:
    patch = AnalyticPatch()
    qmod = AnalyticModule(key="q1", module_type="pitch_quantizer")
    iau = AnalyticModule(key="iau1", module_type="interaural")
    iau.iau_distance = 0.5
    patch.modules = [qmod, iau]

    nodes, _edges = materialize_network(patch, sr=100.0)
    node_keys = {node.key for node in nodes}

    assert {"q1", "iau1_ch1", "iau1_ch2"}.issubset(node_keys)
    qnode = next(node for node in nodes if node.key == "q1")
    qout = qnode.transform(torch.tensor(445.0 + 2.0j, dtype=_CDTYPE))
    assert qout.dtype == _CDTYPE
    assert torch.allclose(qout.imag, torch.tensor(2.0, dtype=torch.float64))

    inode = next(node for node in nodes if node.key == "iau1_ch1")
    iout = inode.transform(torch.tensor(1.0 + 0.0j, dtype=_CDTYPE))
    assert iout.dtype == _CDTYPE
    assert iout.abs() < 1.0
