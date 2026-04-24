import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from parametric_curve import (
    ComplexSignalAggregator,
    EnvelopeRuleTree,
    GateEvent,
    ParametricCurve,
    ParametricCurveEngine,
    RegionEffect,
    TimeWarpCoordinator,
    default_chirp,
    default_envelope,
    normalize_channel_complex_signals,
    render_piecewise_audio,
)


def test_normalize_channel_complex_signals_zero_pads_shorter_inputs() -> None:
    short = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j], dtype=torch.complex128)
    long = torch.tensor([5.0 + 0.0j, 6.0 + 0.0j, 7.0 + 0.0j], dtype=torch.complex128)

    got = normalize_channel_complex_signals({"short": short, "long": long})

    assert got["short"].dtype == torch.complex128
    assert got["long"].dtype == torch.complex128
    assert got["short"].shape == got["long"].shape == (3,)
    assert torch.equal(got["short"][:2], short)
    assert torch.equal(got["long"], long)
    assert got["short"][-1] == 0


def test_normalize_channel_complex_signals_time_stretches_without_zero_padding() -> None:
    src = torch.tensor([1.0 + 0.0j, 3.0 + 0.0j], dtype=torch.complex128)
    ref = torch.tensor([0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j], dtype=torch.complex128)

    got = normalize_channel_complex_signals(
        {"src": src, "ref": ref},
        time_stretch=True,
    )

    lane = torch.view_as_real(got["src"])[:, 0]
    assert got["src"].shape == ref.shape
    assert torch.isclose(got["src"][0], torch.tensor(1.0 + 0.0j, dtype=torch.complex128))
    assert torch.isclose(got["src"][-1], torch.tensor(3.0 + 0.0j, dtype=torch.complex128))
    assert lane[1] > 1.0
    assert lane[1] < 3.0


def test_render_piecewise_audio_returns_complex128_tensors() -> None:
    amp_curve = default_envelope("amp")
    chirp_curve = default_chirp("chirp")
    gates = [GateEvent(t_on=0.0, t_off=0.03, velocity=1.0)]

    def env_fn(t_norm: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t_norm, 0.5 + 0.25j, dtype=torch.complex128)

    def chirp_fn(t_norm: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t_norm, dtype=torch.complex128)

    audio, amp_env, chirp_env, osc = render_piecewise_audio(
        env_fn=env_fn,
        chirp_fn=chirp_fn,
        gate_history=gates,
        amp_curve=amp_curve,
        chirp_curve=chirp_curve,
        freq_hz=220.0,
        gain=0.8,
        dur=0.05,
        sr=8000,
        oversample=2,
    )

    assert isinstance(audio, torch.Tensor)
    assert isinstance(amp_env, torch.Tensor)
    assert isinstance(chirp_env, torch.Tensor)
    assert isinstance(osc, torch.Tensor)
    assert audio.dtype == torch.complex128
    assert amp_env.dtype == torch.complex128
    assert chirp_env.dtype == torch.complex128
    assert osc.dtype == torch.complex128
    assert audio.numel() > 0
    assert torch.view_as_real(audio).shape[1] == 2


def test_complex_signal_aggregator_multiplies_amplitude_and_adds_frequency_phase() -> None:
    synth = ComplexSignalAggregator(8_000.0, 4)
    synth.multiply_amplitude(torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float64))
    synth.multiply_amplitude(torch.tensor([0.5, 0.25, 1.0, 0.5], dtype=torch.float64))
    synth.add_frequency_hz(torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64))
    synth.add_phase(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64))

    out = synth.emit(100.0)

    assert torch.allclose(
        synth.amplitude,
        torch.tensor([1.0, 0.5, 2.0, 1.0], dtype=torch.float64),
    )
    assert torch.allclose(
        synth.frequency_series(100.0),
        torch.tensor([110.0, 120.0, 130.0, 140.0], dtype=torch.float64),
    )
    assert out.dtype == torch.complex128


def test_parametric_curve_engine_interpret_preserves_complex_output() -> None:
    curve = default_envelope("voice")
    engine = ParametricCurveEngine(curve, rule_tree=EnvelopeRuleTree.default())
    gates = [GateEvent(t_on=0.0, t_off=0.04, velocity=1.0)]

    fn = engine.interpret(gates, cache_key="session", force_rebuild=True)
    vals = fn(torch.linspace(0.0, 1.0, 32, dtype=torch.float64))

    assert isinstance(vals, torch.Tensor)
    assert vals.dtype == torch.complex128
    assert vals.shape == (32,)


def test_parametric_curve_tensorized_coeffs_reuse_until_geometry_changes() -> None:
    curve = default_envelope("amp")
    query = torch.tensor([0.25], dtype=torch.float64)

    before = curve.evaluate_normalized(query)
    r_cached_1, _ = curve._get_tensorized_chains(dtype=torch.float64, device=query.device)
    coeff_tensor_1 = r_cached_1[0]["c0s"]

    again = curve.evaluate_normalized(query)
    r_cached_2, _ = curve._get_tensorized_chains(dtype=torch.float64, device=query.device)
    coeff_tensor_2 = r_cached_2[0]["c0s"]

    assert torch.allclose(before, again)
    assert coeff_tensor_1 is coeff_tensor_2

    curve.points[1].v *= 0.5

    after = curve.evaluate_normalized(query)
    r_cached_3, _ = curve._get_tensorized_chains(dtype=torch.float64, device=query.device)
    coeff_tensor_3 = r_cached_3[0]["c0s"]

    assert coeff_tensor_3 is not coeff_tensor_2
    assert not torch.allclose(before, after)


def test_parametric_curve_pack_batch_matches_single_curve_evaluation() -> None:
    curve_a = default_envelope("a")
    curve_b = default_envelope("b")
    curve_b.points[1].v = 0.8
    curve_b.points[2].theta = 0.35
    curve_b.activation = "tanh"
    curve_b.activation_drive = 1.4

    packed = ParametricCurve.pack_batch([curve_a, curve_b])
    query = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)

    got = ParametricCurve.evaluate_batched(packed, query)
    want = torch.stack(
        [curve_a.evaluate_normalized(query), curve_b.evaluate_normalized(query)],
        dim=0,
    )

    assert got.shape == want.shape == (2, 64)
    assert torch.allclose(got, want, atol=1e-12, rtol=0.0)


def test_parametric_curve_batched_transition_compilation_matches_engine_piecewise() -> None:
    curve = default_envelope("voice")
    packed = ParametricCurve.pack_batch([curve])
    gates = [[GateEvent(t_on=0.0, t_off=0.6, velocity=1.0)]]
    query = torch.linspace(0.0, 1.0, 96, dtype=torch.float64)

    got = ParametricCurve.evaluate_batched(
        packed,
        query,
        gate_history_batch=gates,
        clamp_r=False,
        apply_activation=False,
        apply_slew=False,
    ).squeeze(0)

    engine = ParametricCurveEngine(curve, rule_tree=EnvelopeRuleTree.default())
    want = engine.interpret(gates[0], force_rebuild=True)(query)

    assert got.shape == want.shape == (96,)
    assert torch.allclose(got, want, atol=1e-12, rtol=0.0)


def test_warp_with_curve_is_invariant_to_query_density() -> None:
    curve = default_envelope("amp")
    curve.regions[2] = RegionEffect(mode="sustain")
    coordinator = TimeWarpCoordinator(
        retrigger_mode="retrigger",
        release_mode="tail",
        loop_mode="none",
        curve_duration=1.0,
    )
    gates = [GateEvent(t_on=0.0, t_off=0.6, velocity=1.0)]

    coarse = torch.tensor(
        [0.0, 0.1, 0.2, 0.35, 0.5, 0.6, 0.8, 1.0, 1.3],
        dtype=torch.float64,
    )
    dense = torch.tensor(
        [0.0, 0.05, 0.1, 0.15, 0.2, 0.275, 0.35, 0.425, 0.5,
         0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3],
        dtype=torch.float64,
    )

    coarse_warp = coordinator.warp_with_curve(coarse, gates, curve, tau_up=0.38)
    dense_warp = coordinator.warp_with_curve(dense, gates, curve, tau_up=0.38)

    shared_dense_idx = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.long)
    assert torch.allclose(coarse_warp, dense_warp[shared_dense_idx], atol=1e-12, rtol=0.0)


def test_warp_with_curve_stays_at_zero_before_first_note_on() -> None:
    curve = default_envelope("amp")
    curve.regions[2] = RegionEffect(mode="sustain")
    coordinator = TimeWarpCoordinator(
        retrigger_mode="retrigger",
        release_mode="tail",
        loop_mode="none",
        curve_duration=1.0,
    )
    gates = [GateEvent(t_on=0.5, t_off=0.9, velocity=1.0)]
    query = torch.tensor([0.0, 0.25, 0.5, 0.65], dtype=torch.float64)

    warp = coordinator.warp_with_curve(query, gates, curve, tau_up=0.38)

    assert torch.equal(warp[:3], torch.zeros(3, dtype=torch.float64))
    assert warp[3] > 0.0
