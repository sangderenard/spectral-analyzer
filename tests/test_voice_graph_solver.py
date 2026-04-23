import types

import torch

from graph_solver import GraphSolver, TensorEdge, TensorNode, _CDTYPE
from parametric_curve import default_chirp, default_envelope
from voice_graph_node import VoiceTorchOscillator, build_voice_mixer_network


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


def test_voice_network_solves_voice_out_once_per_step_without_feedback() -> None:
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

    assert calls == 1


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
