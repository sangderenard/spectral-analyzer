"""Smoke tests for performer_engine.py — verifies core synthesis correctness."""

import math
import torch
import numpy as np

from performer_engine import (
    DriverConfig, DriverState,
    init_driver_state, driver_synthesis_step,
    scatter_to_voices,
    multi_level_driver_step,
    fm_am_topo_levels,
    CHIRP_NONE, CHIRP_LINEAR, CHIRP_EXPONENTIAL, CHIRP_POWER,
)


def _make_pure_tone_config(
    f0: float = 440.0,
    amplitude: float = 1.0,
    duration: float = 1.0,
    sr: float = 48000.0,
    device: torch.device = torch.device("cpu"),
) -> DriverConfig:
    """Single driver, single harmonic (pure tone), flat envelope."""
    D, H, K, V = 1, 1, 2, 1
    return DriverConfig(
        D=D, H=H, K=K, V=V, device=device,
        f0=torch.tensor([f0], dtype=torch.float64, device=device),
        amplitude=torch.tensor([amplitude], dtype=torch.float64, device=device),
        phase_origin=torch.tensor([0.0], dtype=torch.float64, device=device),
        pre_delay_samples=torch.tensor([0], dtype=torch.int64, device=device),
        note_duration=torch.tensor([duration], dtype=torch.float64, device=device),
        active=torch.tensor([True], dtype=torch.bool, device=device),
        chirp_type=torch.tensor([CHIRP_NONE], dtype=torch.int64, device=device),
        chirp_f_start=torch.zeros(1, dtype=torch.float64, device=device),
        chirp_f_end=torch.zeros(1, dtype=torch.float64, device=device),
        chirp_tau=torch.ones(1, dtype=torch.float64, device=device),
        chirp_power=torch.ones(1, dtype=torch.float64, device=device),
        h_ratios=torch.tensor([[1.0]], dtype=torch.float64, device=device),
        h_amps=torch.tensor([[1.0]], dtype=torch.float64, device=device),
        n_harmonics=torch.tensor([1], dtype=torch.int64, device=device),
        env_t=torch.tensor([[0.0, 1e30]], dtype=torch.float64, device=device),
        env_v=torch.tensor([[1.0, 1.0]], dtype=torch.float64, device=device),
        env_n=torch.tensor([2], dtype=torch.int64, device=device),
        voice_idx=torch.tensor([0], dtype=torch.int64, device=device),
        instrument_idx=torch.tensor([0], dtype=torch.int64, device=device),
        fm_source_voice=torch.tensor([-1], dtype=torch.int64, device=device),
        fm_depth_hz=torch.zeros(1, dtype=torch.float64, device=device),
        am_source_voice=torch.tensor([-1], dtype=torch.int64, device=device),
        am_depth=torch.zeros(1, dtype=torch.float64, device=device),
    )


def test_pure_tone_frequency():
    """A 440 Hz pure tone should have correct instantaneous frequency."""
    sr = 48000.0
    f0 = 440.0
    T = 4800  # 0.1 s
    cfg = _make_pure_tone_config(f0=f0, sr=sr)
    state = init_driver_state(cfg)

    d_out, v_out, ns = driver_synthesis_step(cfg, state, T, sr)

    assert d_out.shape == (1, T), f"Expected (1, {T}), got {d_out.shape}"
    assert v_out.shape == (1, T), f"Expected (1, {T}), got {v_out.shape}"
    assert d_out.dtype == torch.complex128

    # Check instantaneous frequency via phase difference
    sig = d_out[0].cpu()
    phase = torch.angle(sig)
    dphi = torch.diff(phase)
    # Unwrap phase differences
    dphi = dphi - 2 * math.pi * torch.round(dphi / (2 * math.pi))
    f_inst = dphi * sr / (2 * math.pi)
    # Should be ~440 Hz everywhere (skip first few samples for edge effects)
    f_mean = f_inst[10:].mean().item()
    assert abs(f_mean - f0) < 0.5, f"Expected ~{f0} Hz, got {f_mean:.2f} Hz"
    print(f"  pure tone: mean f_inst = {f_mean:.4f} Hz (expected {f0})")


def test_phase_continuity_across_chunks():
    """Phase should be continuous across two consecutive chunks."""
    sr = 48000.0
    f0 = 1000.0
    T = 2400
    cfg = _make_pure_tone_config(f0=f0, sr=sr)
    state = init_driver_state(cfg)

    d1, _, state1 = driver_synthesis_step(cfg, state, T, sr)
    d2, _, state2 = driver_synthesis_step(cfg, state1, T, sr)

    # Last sample of chunk 1 and first sample of chunk 2 should be phase-continuous
    z1_last = d1[0, -1].cpu()
    z2_first = d2[0, 0].cpu()
    phase_jump = abs(torch.angle(z2_first * z1_last.conj()).item())
    expected_jump = 2 * math.pi * f0 / sr  # one sample's worth
    err = abs(phase_jump - expected_jump)
    assert err < 0.01, f"Phase discontinuity: jump={phase_jump:.6f}, expected={expected_jump:.6f}"
    print(f"  phase continuity: jump error = {err:.8f} rad")


def test_pre_delay():
    """Pre-delay should zero the leading samples."""
    sr = 48000.0
    delay_samples = 100
    T = 500
    cfg = _make_pure_tone_config(sr=sr)
    cfg.pre_delay_samples = torch.tensor([delay_samples], dtype=torch.int64)
    state = init_driver_state(cfg)

    d_out, _, _ = driver_synthesis_step(cfg, state, T, sr)
    sig = d_out[0].cpu()

    # First `delay_samples` should be zero
    assert (sig[:delay_samples].abs() < 1e-15).all(), "Pre-delay region not silent"
    # After delay should be nonzero
    assert sig[delay_samples:].abs().mean() > 0.1, "Signal after pre-delay is too quiet"
    print(f"  pre-delay: {delay_samples} silent samples verified")


def test_inactive_driver():
    """Inactive drivers should produce zero output."""
    cfg = _make_pure_tone_config()
    cfg.active = torch.tensor([False], dtype=torch.bool)
    state = init_driver_state(cfg)

    d_out, v_out, _ = driver_synthesis_step(cfg, state, 1000, 48000.0)
    assert (d_out.abs() < 1e-15).all(), "Inactive driver produced nonzero output"
    assert (v_out.abs() < 1e-15).all(), "Inactive driver leaked to voice output"
    print("  inactive driver: zero output verified")


def test_chirp_linear():
    """Linear chirp should sweep f0 from start to end deviation."""
    sr = 48000.0
    f0 = 440.0
    chirp_start = 100.0  # +100 Hz at start
    chirp_end = -100.0   # -100 Hz at end
    dur = 0.5
    T = int(sr * dur)
    cfg = _make_pure_tone_config(f0=f0, duration=dur, sr=sr)
    cfg.chirp_type = torch.tensor([CHIRP_LINEAR], dtype=torch.int64)
    cfg.chirp_f_start = torch.tensor([chirp_start], dtype=torch.float64)
    cfg.chirp_f_end = torch.tensor([chirp_end], dtype=torch.float64)
    state = init_driver_state(cfg)

    d_out, _, _ = driver_synthesis_step(cfg, state, T, sr)
    sig = d_out[0].cpu()

    # Check freq at start and end via phase difference
    phase = torch.angle(sig)
    dphi = torch.diff(phase)
    dphi = dphi - 2 * math.pi * torch.round(dphi / (2 * math.pi))
    f_inst = dphi * sr / (2 * math.pi)

    f_start_measured = f_inst[5:50].mean().item()
    f_end_measured = f_inst[-50:].mean().item()
    # Should be ~540 Hz at start, ~340 Hz at end
    assert abs(f_start_measured - (f0 + chirp_start)) < 5.0, \
        f"Start freq: expected ~{f0+chirp_start}, got {f_start_measured:.1f}"
    assert abs(f_end_measured - (f0 + chirp_end)) < 5.0, \
        f"End freq: expected ~{f0+chirp_end}, got {f_end_measured:.1f}"
    print(f"  linear chirp: start={f_start_measured:.1f} Hz, end={f_end_measured:.1f} Hz")


def test_harmonics():
    """Multi-harmonic voice should contain expected partials."""
    sr = 48000.0
    f0 = 200.0
    T = 4800  # 0.1 s
    H = 4
    dev = torch.device("cpu")
    cfg = _make_pure_tone_config(f0=f0, sr=sr)
    cfg.H = H
    cfg.h_ratios = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64, device=dev)
    cfg.h_amps = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float64, device=dev)
    cfg.n_harmonics = torch.tensor([4], dtype=torch.int64, device=dev)
    cfg.env_t = torch.tensor([[0.0, 1e30]], dtype=torch.float64, device=dev)
    cfg.env_v = torch.tensor([[1.0, 1.0]], dtype=torch.float64, device=dev)
    # Reinit state with correct H
    state = init_driver_state(cfg)

    d_out, _, _ = driver_synthesis_step(cfg, state, T, sr)
    sig = d_out[0].cpu().numpy()

    # FFT of analytic (complex) signal — full bilateral spectrum
    spectrum = np.abs(np.fft.fft(sig))
    freqs = np.fft.fftfreq(T, 1.0 / sr)
    for k, ratio in enumerate([1, 2, 3, 4]):
        target_f = f0 * ratio
        idx = np.argmin(np.abs(freqs - target_f))
        # Peak should be near target frequency (within 2 bins)
        local = spectrum[max(0, idx-2):idx+3]
        assert local.max() > 0.01 * spectrum.max(), \
            f"Harmonic {k+1} at {target_f} Hz not found in spectrum"
    print(f"  harmonics: 4 partials at {f0}, {2*f0}, {3*f0}, {4*f0} Hz verified")


def test_topo_levels_no_deps():
    """With no FM/AM, all voices land in one level."""
    cfg = _make_pure_tone_config()
    levels = fm_am_topo_levels(cfg)
    assert len(levels) == 1
    assert levels[0] == [0]
    print("  topo_levels (no deps): 1 level with 1 voice")


def test_multi_level_no_fm():
    """multi_level_driver_step with no FM should match single-pass."""
    sr = 48000.0
    T = 2400
    cfg = _make_pure_tone_config(sr=sr)
    state = init_driver_state(cfg)

    d_single, v_single, ns_single = driver_synthesis_step(cfg, state, T, sr)
    d_multi, v_multi, ns_multi = multi_level_driver_step(cfg, state, T, sr)

    assert torch.allclose(d_single, d_multi, atol=1e-12), "driver_out mismatch"
    assert torch.allclose(v_single, v_multi, atol=1e-12), "voice_out mismatch"
    assert torch.allclose(ns_single.phase_acc, ns_multi.phase_acc, atol=1e-12)
    print("  multi_level (no FM): matches single-pass")


def test_empty_config():
    """D=0 should not crash."""
    dev = torch.device("cpu")
    cfg = DriverConfig(
        D=0, H=1, K=2, V=0, device=dev,
        f0=torch.zeros(0, dtype=torch.float64, device=dev),
        amplitude=torch.zeros(0, dtype=torch.float64, device=dev),
        phase_origin=torch.zeros(0, dtype=torch.float64, device=dev),
        pre_delay_samples=torch.zeros(0, dtype=torch.int64, device=dev),
        note_duration=torch.zeros(0, dtype=torch.float64, device=dev),
        active=torch.zeros(0, dtype=torch.bool, device=dev),
        chirp_type=torch.zeros(0, dtype=torch.int64, device=dev),
        chirp_f_start=torch.zeros(0, dtype=torch.float64, device=dev),
        chirp_f_end=torch.zeros(0, dtype=torch.float64, device=dev),
        chirp_tau=torch.zeros(0, dtype=torch.float64, device=dev),
        chirp_power=torch.zeros(0, dtype=torch.float64, device=dev),
        h_ratios=torch.zeros(0, 1, dtype=torch.float64, device=dev),
        h_amps=torch.zeros(0, 1, dtype=torch.float64, device=dev),
        n_harmonics=torch.zeros(0, dtype=torch.int64, device=dev),
        env_t=torch.zeros(0, 2, dtype=torch.float64, device=dev),
        env_v=torch.zeros(0, 2, dtype=torch.float64, device=dev),
        env_n=torch.zeros(0, dtype=torch.int64, device=dev),
        voice_idx=torch.zeros(0, dtype=torch.int64, device=dev),
        instrument_idx=torch.zeros(0, dtype=torch.int64, device=dev),
        fm_source_voice=torch.zeros(0, dtype=torch.int64, device=dev),
        fm_depth_hz=torch.zeros(0, dtype=torch.float64, device=dev),
        am_source_voice=torch.zeros(0, dtype=torch.int64, device=dev),
        am_depth=torch.zeros(0, dtype=torch.float64, device=dev),
    )
    state = DriverState(
        phase_acc=torch.zeros(0, 1, dtype=torch.float64, device=dev),
        t_pos=torch.zeros(0, dtype=torch.float64, device=dev),
    )
    d_out, v_out, ns = driver_synthesis_step(cfg, state, 100, 48000.0)
    assert d_out.shape == (0, 100)
    print("  empty config: no crash")


if __name__ == "__main__":
    tests = [
        test_pure_tone_frequency,
        test_phase_continuity_across_chunks,
        test_pre_delay,
        test_inactive_driver,
        test_chirp_linear,
        test_harmonics,
        test_scatter_to_performers,
        test_topo_levels_no_deps,
        test_multi_level_no_fm,
        test_clan_exchange_basic,
        test_empty_config,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            print(f"Running {t.__name__}...")
            t()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        raise SystemExit(1)
