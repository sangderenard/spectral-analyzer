"""test_routing_solve_torch.py — parity tests: torch vs numpy routing solvers.

Every test constructs a routing graph, solves it with BOTH the numpy and torch
paths, and asserts bitwise-close results (atol=1e-10 for iterative paths,
exact for linear).
"""

import math
import numpy as np
import torch

from routing_engine import (
    FeedbackConfig,
    ParamEdge,
    RoutingEdge,
    RoutingGraph,
    solve_routing_complex as solve_np,
    solve_routing_with_ringdown as solve_ringdown_np,
    extract_param_series as extract_np,
    solve_param_routing as solve_param_np,
    compute_latency_compensation as compute_latency_np,
)
from routing_solve_torch import (
    solve_routing_complex as solve_torch,
    solve_routing_with_ringdown as solve_ringdown_torch,
    extract_param_series as extract_torch,
    project_tensor_to_scalar_param,
    solve_param_routing as solve_param_torch,
    compute_latency_compensation as compute_latency_torch,
    synthesize_lfo_csig,
    compute_envelope,
    _hilbert_torch,
    _eval_piecewise_poly,
    _clamped_spline_coeffs,
    _pchip_coeffs,
    _linear_coeffs,
    RoutingMixerTorch,
)


def _assert_close(np_arr, torch_arr, label, atol=1e-10):
    """Assert numpy and torch results are close."""
    t_np = torch_arr.cpu().numpy()
    diff = np.max(np.abs(np_arr - t_np))
    assert diff < atol, f"{label}: max diff = {diff} (atol={atol})"
    return diff


def test_linear_no_delay():
    """Exact linear solve: X = (I-W)^-1 @ Src — should match exactly."""
    sr = 48000.0
    T = 1000
    keys = ["a", "b", "c"]
    edges = [
        RoutingEdge("a", "b", 0.5, 0.0, 0.0),
        RoutingEdge("b", "c", 0.3, math.pi / 4, 0.0),
        RoutingEdge("c", "a", -0.2, 0.0, 0.0),
    ]
    rng = np.random.RandomState(42)
    Src_np = (rng.randn(3, T) + 1j * rng.randn(3, T)).astype(np.complex128)
    Src_t = torch.from_numpy(Src_np)

    X_np = solve_np(Src_np, edges, keys, sr)
    X_t = solve_torch(Src_t, edges, keys, sr)

    diff = _assert_close(X_np, X_t, "linear_no_delay", atol=1e-10)
    print(f"  linear no delay: max diff = {diff:.2e}")


def test_with_delay():
    """Delayed edges: causal chunk solver — should match closely."""
    sr = 48000.0
    T = 2000
    keys = ["osc1", "osc2"]
    edges = [
        RoutingEdge("osc1", "osc2", 0.5, 0.0, 0.001),   # 1ms delay
        RoutingEdge("osc2", "osc1", -0.3, 0.0, 0.002),   # 2ms delay
    ]
    rng = np.random.RandomState(123)
    Src_np = (rng.randn(2, T) + 1j * rng.randn(2, T)).astype(np.complex128)
    Src_t = torch.from_numpy(Src_np)

    X_np = solve_np(Src_np, edges, keys, sr, global_decay=0.95)
    X_t = solve_torch(Src_t, edges, keys, sr, global_decay=0.95)

    diff = _assert_close(X_np, X_t, "with_delay", atol=1e-10)
    print(f"  delayed edges: max diff = {diff:.2e}")


def test_negative_delay():
    """Pre-advance (negative delay) edges — should match."""
    sr = 48000.0
    T = 1000
    keys = ["a", "b"]
    edges = [
        RoutingEdge("a", "b", 0.4, 0.0, -0.001),  # negative = advance
    ]
    rng = np.random.RandomState(77)
    Src_np = (rng.randn(2, T) + 1j * rng.randn(2, T)).astype(np.complex128)
    Src_t = torch.from_numpy(Src_np)

    X_np = solve_np(Src_np, edges, keys, sr)
    X_t = solve_torch(Src_t, edges, keys, sr)

    diff = _assert_close(X_np, X_t, "negative_delay", atol=1e-10)
    print(f"  negative delay: max diff = {diff:.2e}")


def test_node_transform():
    """Nonlinear node transform — iterative fixed-point should converge identically."""
    sr = 48000.0
    T = 500
    keys = ["a", "b"]
    edges = [
        RoutingEdge("a", "b", 0.6, 0.0, 0.0),
        RoutingEdge("b", "a", 0.3, 0.0, 0.0),
    ]
    # Nonlinear transform on b: soft magnitude clip at 2.0
    def _clip_np(row):
        mag = np.abs(row)
        scale = np.where(mag > 2.0, 2.0 / np.maximum(mag, 1e-30), 1.0)
        return row * scale

    def _clip_torch(row):
        mag = row.abs()
        scale = torch.where(mag > 2.0, 2.0 / mag.clamp(min=1e-30), torch.ones_like(mag))
        return row * scale.to(row.dtype)

    rng = np.random.RandomState(99)
    Src_np = (rng.randn(2, T) + 1j * rng.randn(2, T)).astype(np.complex128)
    Src_t = torch.from_numpy(Src_np)

    X_np = solve_np(Src_np, edges, keys, sr, node_transforms={"b": _clip_np})
    X_t = solve_torch(Src_t, edges, keys, sr, node_transforms={"b": _clip_torch})

    diff = _assert_close(X_np, X_t, "node_transform", atol=1e-8)
    print(f"  node transform: max diff = {diff:.2e}")


def test_wcc_partitioning():
    """Disconnected components solve independently — verify partitioning."""
    sr = 48000.0
    T = 500
    keys = ["a", "b", "c", "d"]
    edges = [
        RoutingEdge("a", "b", 0.5, 0.0, 0.0),  # component {a, b}
        RoutingEdge("c", "d", 0.3, 0.0, 0.0),  # component {c, d}
    ]
    rng = np.random.RandomState(55)
    Src_np = (rng.randn(4, T) + 1j * rng.randn(4, T)).astype(np.complex128)
    Src_t = torch.from_numpy(Src_np)

    X_np = solve_np(Src_np, edges, keys, sr)
    X_t = solve_torch(Src_t, edges, keys, sr)

    diff = _assert_close(X_np, X_t, "wcc_partitioning", atol=1e-10)
    print(f"  WCC partitioning: max diff = {diff:.2e}")


def test_ringdown():
    """Ringdown extends the buffer — verify both solvers agree on shape and content."""
    sr = 48000.0
    T = 1000
    keys = ["a", "b"]
    edges = [
        RoutingEdge("a", "b", 0.8, 0.0, 0.002),
        RoutingEdge("b", "a", 0.5, 0.0, 0.002),
    ]
    fb = FeedbackConfig(
        enabled=True, decay=0.1,
        ringdown_mode="decay_to_silence",
        ringdown_max_s=1.0,
        ringdown_threshold=1e-4,
    )
    rng = np.random.RandomState(42)
    Src_np = (rng.randn(2, T) + 1j * rng.randn(2, T)).astype(np.complex128)
    Src_t = torch.from_numpy(Src_np)
    gd = max(0.0, 1.0 - fb.decay)

    X_np, nl_np = solve_ringdown_np(Src_np, edges, keys, sr, gd, fb)
    X_t, nl_t = solve_ringdown_torch(Src_t, edges, keys, sr, gd, fb)

    assert nl_np == nl_t, f"note_len mismatch: {nl_np} vs {nl_t}"
    assert X_np.shape == tuple(X_t.shape), f"shape mismatch: {X_np.shape} vs {X_t.shape}"
    diff = _assert_close(X_np, X_t, "ringdown", atol=1e-10)
    print(f"  ringdown: max diff = {diff:.2e}, shape = {X_np.shape}")


def test_latency_compensation():
    """Latency compensation is a graph algorithm — must give identical results."""
    keys = ["a", "b", "c"]
    edges = [
        RoutingEdge("a", "b", 0.5, 0.0, 0.005),  # 5ms
        RoutingEdge("b", "c", 0.3, 0.0, 0.010),  # 10ms
    ]
    sr = 48000.0

    lead_np = compute_latency_np(edges, keys, sr)
    lead_t = compute_latency_torch(edges, keys, sr)

    assert lead_np == lead_t, f"Latency compensation mismatch: {lead_np} vs {lead_t}"
    print(f"  latency compensation: {lead_t}")


def test_param_extraction():
    """All param extractors should match numpy."""
    T = 1000
    rng = np.random.RandomState(7)
    arr_np = (rng.randn(T) + 1j * rng.randn(T)).astype(np.complex128)
    arr_t = torch.from_numpy(arr_np)

    for ext in ("magnitude", "real", "imag", "phase", "energy"):
        r_np = extract_np(arr_np, ext)
        r_t = extract_torch(arr_t, ext)
        diff = np.max(np.abs(r_np - r_t.cpu().numpy()))
        assert diff < 1e-12, f"Extractor {ext}: max diff = {diff}"
        print(f"  extractor '{ext}': diff = {diff:.2e}")

    # RMS has slight numerical difference due to convolution implementation
    r_np = extract_np(arr_np, "rms")
    r_t = extract_torch(arr_t, "rms")
    diff = np.max(np.abs(r_np - r_t.cpu().numpy()))
    assert diff < 1e-6, f"Extractor rms: max diff = {diff}"
    print(f"  extractor 'rms': diff = {diff:.2e}")


def test_solve_param_routing():
    """Full param routing solve should match."""
    T = 500
    rng = np.random.RandomState(11)

    sig_np = {
        "v1": (rng.randn(T) + 1j * rng.randn(T)).astype(np.complex128),
        "v2": (rng.randn(T) + 1j * rng.randn(T)).astype(np.complex128),
    }
    sig_t = {k: torch.from_numpy(v) for k, v in sig_np.items()}

    param_edges = [
        ParamEdge("v1", "p1", 0.5, "magnitude"),
        ParamEdge("v2", "p1", -0.3, "real"),
        ParamEdge("v1", "p2", 1.0, "phase"),
    ]
    param_defaults = {"p1": 0.0, "p2": 0.5}
    param_bounds = {"p1": (-1.0, 1.0), "p2": (-3.2, 3.2)}

    r_np = solve_param_np(sig_np, param_edges, param_defaults, param_bounds)
    r_t = solve_param_torch(sig_t, param_edges, param_defaults, param_bounds)

    for key in param_defaults:
        diff = np.max(np.abs(r_np[key] - r_t[key].cpu().numpy()))
        assert diff < 1e-12, f"Param {key}: max diff = {diff}"
        print(f"  param '{key}': diff = {diff:.2e}")


def test_project_tensor_to_scalar_param_preserves_time_axis():
    x = torch.tensor(
        [
            [1.0 + 1.0j, 2.0 + 0.0j, 0.0 + 1.0j],
            [3.0 + 4.0j, 0.0 + 2.0j, 2.0 + 0.0j],
        ],
        dtype=torch.complex128,
    )
    out = project_tensor_to_scalar_param(x, "magnitude_mean", time_dim=1)
    expected = torch.tensor(
        [
            ((2.0 ** 0.5) + 5.0) / 2.0,
            (2.0 + 2.0) / 2.0,
            (1.0 + 2.0) / 2.0,
        ],
        dtype=torch.float64,
    )
    diff = (out - expected).abs().max().item()
    assert diff < 1e-12, f"scalar projection over batch dims: max diff = {diff}"


def test_solve_param_routing_uses_projection_policy_for_wide_tensor_inputs():
    sig_t = {
        "v1": torch.tensor(
            [
                [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j],
                [3.0 + 0.0j, 5.0 + 0.0j, 7.0 + 0.0j],
            ],
            dtype=torch.complex128,
        ),
    }
    param_edges = [
        ParamEdge("v1", "p1", 1.0, "magnitude", projection_policy="real_mean"),
    ]
    param_defaults = {"p1": 0.0}
    param_bounds = {"p1": (-10.0, 10.0)}

    r_t = solve_param_torch(sig_t, param_edges, param_defaults, param_bounds)
    expected = torch.tensor([2.0, 3.5, 5.0], dtype=torch.float64)
    diff = (r_t["p1"] - expected).abs().max().item()
    assert diff < 1e-12, f"wide param projection: max diff = {diff}"


def test_lfo_sine():
    """Sine LFO should be exact analytic: depth * exp(i*(wt+phi))."""
    n = 4800
    sr = 48000.0
    rate = 5.0
    depth = 0.8
    phase = 0.3

    sig = synthesize_lfo_csig(rate, "Sine", phase, depth, n, sr)
    t = torch.arange(n, dtype=torch.float64) / sr
    expected = depth * torch.exp(1j * (2 * math.pi * rate * t + phase))

    diff = (sig.cpu() - expected).abs().max().item()
    assert diff < 1e-14, f"Sine LFO: max diff = {diff}"
    print(f"  LFO Sine: diff = {diff:.2e}")


def test_hilbert_torch():
    """Hilbert transform should match scipy's output."""
    try:
        from scipy.signal import hilbert as _hilbert_scipy
    except ImportError:
        print("  Hilbert: scipy not available, skipping")
        return

    rng = np.random.RandomState(42)
    x_np = rng.randn(1024).astype(np.float64)
    x_t = torch.from_numpy(x_np)

    h_scipy = _hilbert_scipy(x_np).astype(np.complex128)
    h_torch = _hilbert_torch(x_t).cpu().numpy()

    diff = np.max(np.abs(h_scipy - h_torch))
    assert diff < 1e-10, f"Hilbert: max diff = {diff}"
    print(f"  Hilbert transform: diff = {diff:.2e}")


def test_envelope_linear():
    """Piecewise-linear envelope should match np.interp."""
    ts = np.array([0.0, 0.2, 0.5, 1.0], dtype=np.float64)
    vs = np.array([0.0, 1.0, 0.5, 0.0], dtype=np.float64)
    n = 1000
    dur = 1.0
    t_ax = np.linspace(0, dur, n, endpoint=False, dtype=np.float64)

    np_env = np.interp(t_ax, ts, vs)
    t_env = compute_envelope(
        torch.from_numpy(ts), torch.from_numpy(vs), n, dur, "adsr",
    )
    diff = np.max(np.abs(np_env - t_env.cpu().numpy()))
    assert diff < 1e-14, f"Linear envelope: max diff = {diff}"
    print(f"  envelope linear: diff = {diff:.2e}")


def test_envelope_spline():
    """Spline envelope should match scipy CubicSpline (clamped)."""
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        print("  Spline envelope: scipy not available, skipping")
        return

    ts = np.array([0.0, 0.1, 0.3, 0.6, 1.0], dtype=np.float64)
    vs = np.array([0.0, 0.8, 1.0, 0.5, 0.0], dtype=np.float64)
    n = 2000
    dur = 1.0
    t_ax = np.linspace(0, dur, n, endpoint=False, dtype=np.float64)

    scipy_env = np.clip(CubicSpline(ts, vs, bc_type="clamped")(t_ax), 0.0, None)
    torch_env = compute_envelope(
        torch.from_numpy(ts), torch.from_numpy(vs), n, dur, "spline",
    )

    diff = np.max(np.abs(scipy_env - torch_env.cpu().numpy()))
    assert diff < 1e-8, f"Spline envelope: max diff = {diff}"
    print(f"  envelope spline: diff = {diff:.2e}")


def test_envelope_monotone():
    """Monotone (PCHIP) envelope should match scipy PchipInterpolator."""
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        print("  PCHIP envelope: scipy not available, skipping")
        return

    ts = np.array([0.0, 0.15, 0.4, 0.7, 1.0], dtype=np.float64)
    vs = np.array([0.0, 1.0, 0.8, 0.3, 0.0], dtype=np.float64)
    n = 2000
    dur = 1.0
    t_ax = np.linspace(0, dur, n, endpoint=False, dtype=np.float64)

    scipy_env = np.clip(PchipInterpolator(ts, vs)(t_ax), 0.0, None)
    torch_env = compute_envelope(
        torch.from_numpy(ts), torch.from_numpy(vs), n, dur, "monotone",
    )

    diff = np.max(np.abs(scipy_env - torch_env.cpu().numpy()))
    assert diff < 1e-8, f"PCHIP envelope: max diff = {diff}"
    print(f"  envelope PCHIP: diff = {diff:.2e}")


def test_empty_graph():
    """Empty graph: X = Src — no edges, no crash."""
    sr = 48000.0
    T = 100
    keys = ["a", "b"]
    Src_np = np.ones((2, T), dtype=np.complex128)
    Src_t = torch.from_numpy(Src_np)

    X_np = solve_np(Src_np, [], keys, sr)
    X_t = solve_torch(Src_t, [], keys, sr)

    diff = _assert_close(X_np, X_t, "empty_graph", atol=1e-14)
    print(f"  empty graph: diff = {diff:.2e}")


def test_mixer_torch():
    """RoutingMixerTorch should produce same results as numpy mixer."""
    from routing_engine import RoutingMixer

    g = RoutingGraph()
    g.add_node("osc1")
    g.add_node("osc2")
    g.add_node("__mix__")
    g.edges.append(RoutingEdge("osc1", "__mix__", 1.0))
    g.edges.append(RoutingEdge("osc2", "__mix__", 0.5, math.pi / 6))

    rng = np.random.RandomState(42)
    T = 1000
    sources_np = {
        "osc1": (rng.randn(T) + 1j * rng.randn(T)).astype(np.complex128),
        "osc2": (rng.randn(T) + 1j * rng.randn(T)).astype(np.complex128),
    }
    sources_t = {k: torch.from_numpy(v) for k, v in sources_np.items()}

    mixer_np = RoutingMixer(g, sample_rate=48000.0)
    mixer_t = RoutingMixerTorch(g, sample_rate=48000.0)

    out_np = mixer_np.mix(sources_np)
    out_t = mixer_t.mix(sources_t)

    for key in out_np:
        if key in out_t:
            diff = np.max(np.abs(out_np[key] - out_t[key].cpu().numpy()))
            assert diff < 1e-10, f"Mixer key {key}: max diff = {diff}"
    print("  RoutingMixerTorch: matches numpy mixer")


if __name__ == "__main__":
    tests = [
        test_linear_no_delay,
        test_with_delay,
        test_negative_delay,
        test_node_transform,
        test_wcc_partitioning,
        test_ringdown,
        test_latency_compensation,
        test_param_extraction,
        test_solve_param_routing,
        test_lfo_sine,
        test_hilbert_torch,
        test_envelope_linear,
        test_envelope_spline,
        test_envelope_monotone,
        test_empty_graph,
        test_mixer_torch,
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
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        raise SystemExit(1)
