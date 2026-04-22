import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from parametric_curve import (
    EnvelopeRuleTree,
    GateEvent,
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


def test_parametric_curve_engine_interpret_preserves_complex_output() -> None:
    curve = default_envelope("voice")
    engine = ParametricCurveEngine(curve, rule_tree=EnvelopeRuleTree.default())
    gates = [GateEvent(t_on=0.0, t_off=0.04, velocity=1.0)]

    fn = engine.interpret(gates, cache_key="session", force_rebuild=True)
    vals = fn(torch.linspace(0.0, 1.0, 32, dtype=torch.float64))

    assert isinstance(vals, torch.Tensor)
    assert vals.dtype == torch.complex128
    assert vals.shape == (32,)


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
