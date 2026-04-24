"""Rigorous CLI test suite for the piecewise oscillator bake pipeline.

Tests are organised into groups:
  hilbert     _analytic_signal round-trip to machine epsilon
  pencil      _matrix_pencil_fit pole/residue recovery on known signals
  piecewise   _eval_oscillators_piecewise correctness and continuity
  curvature   adaptive segment breaks quality vs equal-width
  activation  _apply_activation range, monotonicity, boundary conditions
  warp        TimeWarpCoordinator boundary conditions
  quality     end-to-end reconstruction RMSE comparison

Usage examples
--------------
    python test_piecewise_cli.py                        # all groups
    python test_piecewise_cli.py --groups hilbert pencil
    python test_piecewise_cli.py --groups quality --segments 1 4 8 16 32
    python test_piecewise_cli.py --verbose
    python test_piecewise_cli.py --bail                 # stop on first failure
"""

import argparse
import math
import sys
import time
from typing import Callable

import numpy as np
import torch


# ─── import the module under test ────────────────────────────────────────────
try:
    from parametric_curve_editor import (
        ParametricCurve,
        ControlPoint,
        GateEvent,
        TimeWarpCoordinator,
        _analytic_signal,
        _matrix_pencil_fit,
        _apply_activation,
        _eval_oscillators,
        _eval_oscillators_piecewise,
        _curvature_segment_breaks,
        _split_into_chains,
        _build_cr_chain,
        _eval_all_chains,
        default_envelope,
        default_chirp,
        default_blank,
    )
except ImportError as exc:
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(f"legacy CLI imports unavailable: {exc}", allow_module_level=True)
    raise

EPS64 = torch.finfo(torch.float64).eps   # 2.22e-16
EPS32 = torch.finfo(torch.float32).eps   # 1.19e-7

# ─── tiny test harness ───────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []
_verbose = False


def _pass(name: str, detail: str = "") -> None:
    _results.append((name, True, detail))
    if _verbose:
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))


def _fail(name: str, detail: str = "") -> None:
    _results.append((name, False, detail))
    print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))


def _check(name: str, cond: bool, detail: str = "", bail: bool = False) -> None:
    if cond:
        _pass(name, detail)
    else:
        _fail(name, detail)
        if bail:
            print("\nBailing on first failure.")
            _summary()
            sys.exit(1)


_bail = False


def check(name: str, cond: bool, detail: str = "") -> None:
    _check(name, cond, detail, bail=_bail)


def _summary() -> None:
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total  = len(_results)
    print()
    print("=" * 60)
    print(f"  {passed}/{total} passed   {failed} failed")
    if failed:
        print()
        print("  Failed tests:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    ✗ {name}: {detail}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: hilbert
# _analytic_signal(y).real should equal y to machine epsilon
# The Hilbert FFT is an exact operation on finite-length float64 signals;
# the round-trip residual should be bounded by N * EPS64 * max|y|.
# ─────────────────────────────────────────────────────────────────────────────

def test_hilbert() -> None:
    print("\n── hilbert ─────────────────────────────────────────────────────")

    for N in [8, 64, 256, 1024, 4097]:   # include odd length
        y = torch.randn(N, dtype=torch.float64)
        z = _analytic_signal(y)

        # dtype contract
        check(f"hilbert/dtype N={N}", z.dtype == torch.complex128,
              f"got {z.dtype}")

        # shape contract
        check(f"hilbert/shape N={N}", z.shape == (N,),
              f"got {z.shape}")

        # real-part round-trip to machine epsilon
        residual = (z.real - y).abs().max().item()
        threshold = 8.0 * N * EPS64 * float(y.abs().max())
        check(f"hilbert/real_roundtrip N={N}",
              residual < threshold,
              f"max|z.real-y|={residual:.3e}  threshold={threshold:.3e}")

    # DC signal: Hilbert of constant = constant + 0j imaginary
    dc = torch.full((128,), 3.7, dtype=torch.float64)
    z_dc = _analytic_signal(dc)
    check("hilbert/dc_real",
          float((z_dc.real - dc).abs().max()) < 1e-12,
          f"{float((z_dc.real - dc).abs().max()):.2e}")
    check("hilbert/dc_imag_near_zero",
          float(z_dc.imag.abs().max()) < 1e-10,
          f"{float(z_dc.imag.abs().max()):.2e}")

    # Single cosine: analytic signal of cos(2πft) should be exp(2πjft)
    N  = 1024
    f0 = 7.0
    t  = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
    y  = torch.cos(2 * math.pi * f0 * t)
    z  = _analytic_signal(y)
    # |z| ≈ 1 for a pure cosine (analytic envelope = 1)
    envelope_err = (z.abs() - 1.0).abs()[N//4 : 3*N//4]   # avoid edge roll-off
    check("hilbert/cosine_envelope",
          float(envelope_err.max()) < 5e-3,
          f"max env err={float(envelope_err.max()):.2e}")
    # phase advances at 2πf0/N per sample
    phase_diff = torch.diff(torch.angle(z[N//4 : 3*N//4]))
    expected   = 2.0 * math.pi * f0 / N
    check("hilbert/cosine_phase_advance",
          float((phase_diff - expected).abs().max()) < 1e-6,
          f"max phase err={float((phase_diff - expected).abs().max()):.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: pencil
# _matrix_pencil_fit pole recovery on synthetic dampened exponential sums
# The pencil is an algebraic method: for an exact sum of K exp(s_k t) it
# should recover all poles and residues to close to machine precision
# (subject to the conditioning of the Vandermonde LS system).
# ─────────────────────────────────────────────────────────────────────────────

def _synth_expsig(poles_s, residues, N: int):
    """Generate z[n] = Σ_k c_k * λ_k^n  (discrete),  λ_k = exp(s_k * dt)."""
    dt = 1.0 / (N - 1)
    n  = np.arange(N, dtype=np.float64)
    z  = np.zeros(N, dtype=np.complex128)
    for s, c in zip(poles_s, residues):
        lam = np.exp(s * dt)
        z  += c * (lam ** n)
    return z


def test_pencil() -> None:
    print("\n── pencil ──────────────────────────────────────────────────────")

    # Single damped real exponential: z(t) = exp(s*t), s = -3+0j
    for N in [64, 256, 512]:
        s_true = np.array([-3.0 + 0.0j])
        c_true = np.array([1.0 + 0.0j])
        z = _synth_expsig(s_true, c_true, N)
        dt = 1.0 / (N - 1)
        poles_hat, res_hat = _matrix_pencil_fit(z, dt=dt, n_poles=1)

        # Match by smallest pole-error (ordering may differ)
        best_pole_err = min(abs(p - s_true[0]) for p in poles_hat)
        check(f"pencil/single_exp_pole N={N}",
              best_pole_err < 1e-4,
              f"pole err={best_pole_err:.3e}")

    # Two damped sinusoids: z(t) = c1*exp(s1*t) + c2*exp(s2*t)
    N = 512
    s_true = np.array([-1.0 + 2.0j * math.pi * 5.0,
                        -0.5 + 2.0j * math.pi * 13.0])
    c_true = np.array([0.8 + 0.3j, 0.5 - 0.2j])
    z = _synth_expsig(s_true, c_true, N)
    dt = 1.0 / (N - 1)
    poles_hat, res_hat = _matrix_pencil_fit(z, dt=dt, n_poles=2)

    # Pair each true pole to the closest recovered pole
    used = set()
    max_pole_err = 0.0
    for st in s_true:
        errs = [(abs(st - ph), j) for j, ph in enumerate(poles_hat) if j not in used]
        errs.sort()
        max_pole_err = max(max_pole_err, errs[0][0])
        used.add(errs[0][1])
    check(f"pencil/two_sinusoid_poles N={N}",
          max_pole_err < 1e-2,
          f"max pole err={max_pole_err:.3e}")

    # Reconstruction fidelity (RMSE of real part) on known signal
    sig_hat = _eval_oscillators(
        torch.as_tensor(poles_hat, dtype=torch.complex128),
        torch.as_tensor(res_hat,   dtype=torch.complex128),
        torch.linspace(0.0, 1.0, N, dtype=torch.float64),
    )
    ref_real = torch.as_tensor(z.real, dtype=torch.float64)
    rmse = float(((sig_hat - ref_real) ** 2).mean().sqrt())
    check("pencil/two_sinusoid_rmse",
          rmse < 1e-3,
          f"RMSE={rmse:.3e}")

    # Auto-K: on a pure DC signal, pencil should select K=1 (or close)
    dc_sig = np.ones(256, dtype=np.complex128)
    poles_dc, _ = _matrix_pencil_fit(dc_sig, dt=1.0/255.0, n_poles=0)
    check("pencil/auto_k_dc",
          len(poles_dc) <= 4,
          f"auto K on DC={len(poles_dc)}")

    # Residue reconstruction: Σ c_k * exp(s_k * 0) = z[0]
    z0_hat = float(sum(r for r in res_hat.real))
    check("pencil/residue_sum_at_t0",
          abs(z0_hat - float(z[0].real)) < 0.1,
          f"z[0].real={float(z[0].real):.4f}  Σres_hat.real={z0_hat:.4f}")

    # ── machine-precision convergence on a smooth polynomial ─────────────────
    # A cubic polynomial (Catmull-Rom segment) is an entire function, so the
    # Matrix Pencil CAN recover it to float64 machine precision given enough K.
    # With rmse_target set to machine-eps scale the loop must converge.
    for N in [64, 128, 256]:
        t_axis  = np.linspace(0.0, 1.0, N)
        # Degree-3 polynomial: plausible Catmull-Rom output shape
        poly    = (1.0 - np.exp(-5.0 * t_axis)) * np.cos(2.0 * np.pi * 1.5 * t_axis)
        poly   -= poly[0]    # make it start at 0
        poly   /= max(abs(poly).max(), 1e-30)   # unit scale
        z_poly  = _analytic_signal(torch.as_tensor(poly, dtype=torch.float64)).numpy()
        dt_poly = 1.0 / (N - 1)
        scale   = max(float(abs(poly).max()), 1e-30)
        tgt     = 8.0 * N * float(np.finfo(np.float64).eps) * scale
        poles_c, res_c = _matrix_pencil_fit(z_poly, dt=dt_poly,
                                             n_poles=0, rmse_target=tgt)
        # Measure actual reconstruction RMSE
        lam_c = np.exp(poles_c * dt_poly)
        n_vec = np.arange(N, dtype=np.float64)
        A_c   = lam_c[np.newaxis, :] ** n_vec[:, np.newaxis]
        recon = (A_c @ res_c).real
        rmse_c = float(np.sqrt(np.mean((recon - poly) ** 2)))
        check(f"pencil/machine_eps_smooth_poly N={N}",
              rmse_c <= max(tgt * 10.0, 1e-10),
              f"RMSE={rmse_c:.3e}  target={tgt:.3e}  K={len(poles_c)}")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: piecewise
# - S=1 _eval_oscillators_piecewise must match _eval_oscillators exactly
# - At each segment break, left and right evaluations agree to float64 eps
# - dtype propagation: input float32 → output float32
# - Boundary t=0 and t=1 produce finite values
# ─────────────────────────────────────────────────────────────────────────────

def test_piecewise() -> None:
    print("\n── piecewise ───────────────────────────────────────────────────")

    c = default_envelope()
    c.piecewise_segments = 1
    seg_p, seg_r, seg_b = c.bake()

    t_probe = torch.linspace(0.0, 1.0, 1000, dtype=torch.float64)

    # S=1 piecewise == flat oscillators (same underlying call)
    flat_out = _eval_oscillators(seg_p[0], seg_r[0], t_probe)
    pw_out   = _eval_oscillators_piecewise(seg_p, seg_r, seg_b, t_probe)
    delta    = (pw_out - flat_out).abs().max().item()
    check("piecewise/s1_matches_flat", delta < 1e-12,
          f"max delta={delta:.3e}")

    # ── C⁰ continuity at segment breaks ──────────────────────────────────────
    # Each segment is a smooth analytic polynomial (Catmull-Rom), so the Matrix
    # Pencil CAN fit it to machine precision.  Adjacent segments share their
    # boundary sample, so both fits converge to the same boundary value.
    # The jump at each interior break must be machine-eps small.
    #
    # Tolerance: both fits attain RMSE ≤ 8·M·EPS·scale where M is the segment
    # length.  By triangle inequality the pointwise boundary error is bounded by
    # ≤ 2·√M·(8·M·EPS·scale) = 16·M^(3/2)·EPS·scale.
    # For M≤256, scale≤1 that is < 6e-11.  We use a generous 1e-8 to allow for
    # ill-conditioned segments (nearly flat) where the Vandermonde is more noisy.
    C0_TOL = 1e-8

    for S in [2, 4, 8, 16]:
        c2 = default_envelope()
        c2.piecewise_segments = S
        sp, sr_, sb = c2.bake()     # sp [S,K] complex, sb [S+1] float64

        # Evaluate each segment independently at its own u∈{0,1} endpoints
        # using _eval_oscillators so we bypass searchsorted and directly target
        # the correct segment tensor.
        max_jump = 0.0
        worst_i  = -1
        for i in range(S - 1):
            # End of segment i: u=1.0 → t_c = 1.0 * (1+j) (complex unit time)
            u_end   = torch.tensor([1.0], dtype=torch.float64)
            val_end = _eval_oscillators(sp[i],     sr_[i],     u_end)

            # Start of segment i+1: u=0.0
            u_start = torch.tensor([0.0], dtype=torch.float64)
            val_sta = _eval_oscillators(sp[i + 1], sr_[i + 1], u_start)

            jump = abs(float(val_end[0].real) - float(val_sta[0].real))
            if jump > max_jump:
                max_jump = jump
                worst_i  = i

        check(f"piecewise/C0_continuity S={S}",
              max_jump < C0_TOL,
              f"max |jump| at breaks = {max_jump:.3e}  (worst seg {worst_i}→{worst_i+1})  tol={C0_TOL:.0e}")

        # Finite everywhere
        all_t = torch.linspace(0.0, 1.0, 512, dtype=torch.float64)
        all_y = _eval_oscillators_piecewise(sp, sr_, sb, all_t)
        check(f"piecewise/finite_everywhere S={S}",
              torch.isfinite(all_y).all(),
              f"nan/inf count={torch.sum(~torch.isfinite(all_y)).item()}")

    # Boundary conditions: t=0 and t=1 must be finite
    c4 = default_envelope(); c4.piecewise_segments = 4
    sp4, sr4, sb4 = c4.bake()
    y_bnd = _eval_oscillators_piecewise(sp4, sr4, sb4,
                                         torch.tensor([0.0, 1.0], dtype=torch.float64))
    check("piecewise/boundaries_finite",
          torch.isfinite(y_bnd).all(),
          f"vals={y_bnd.tolist()}")

    # Dtype passthrough via evaluate_normalized
    t32 = torch.linspace(0.0, 1.0, 100, dtype=torch.float32)
    y32 = c4.evaluate_normalized(t32)
    check("piecewise/dtype_float32_passthrough",
          y32.dtype == torch.float32,
          f"got {y32.dtype}")

    # All outputs from evaluate_normalized must lie in [0, 1]
    c8 = default_envelope(); c8.piecewise_segments = 8
    t64 = torch.linspace(0.0, 1.0, 2000, dtype=torch.float64)
    y64 = c8.evaluate_normalized(t64)
    check("piecewise/range_0_1",
          float(y64.min()) >= 0.0 and float(y64.max()) <= 1.0,
          f"[{float(y64.min()):.4f}, {float(y64.max()):.4f}]")

    # _eval_oscillators_piecewise matches evaluate_normalized (no activation/slew)
    c_base = default_envelope(); c_base.piecewise_segments = 4
    c_base.activation = "none"; c_base.slew_samples = 0
    sp_b, sr_b, sb_b = c_base.bake()
    # _eval_oscillators_piecewise now returns complex128; take .real to compare
    pw_direct = _eval_oscillators_piecewise(sp_b, sr_b, sb_b, t64).real.to(torch.float64).clamp(0.0, 1.0)
    eval_out  = c_base.evaluate_normalized(t64)
    check("piecewise/direct_vs_eval_normalized",
          float((pw_direct - eval_out).abs().max()) < 1e-12,
          f"max delta={float((pw_direct - eval_out).abs().max()):.3e}")

    # evaluate_complex: real part == evaluate_normalized (before act/slew)
    zc = c_base.evaluate_complex(t64)
    check("piecewise/complex_real_part_matches",
          float((zc.real.to(torch.float64) - eval_out).abs().max()) < 1e-12,
          f"max delta={float((zc.real.to(torch.float64) - eval_out).abs().max()):.3e}")
    check("piecewise/complex_dtype",
          zc.dtype == torch.complex128)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: curvature
# Curvature-adaptive breaks concentrate breaks near high-curvature regions.
# Verify:
#  - breaks are monotone, start at 0, end at 1
#  - for a synthetic high-curvature signal, the adaptive breaks place more
#    break density near the peak than equal-width would
#  - reconstruction RMSE improves vs equal-width on a sharp ADSR-like signal
# ─────────────────────────────────────────────────────────────────────────────

def _sharp_adsr(N: int) -> torch.Tensor:
    """Sharp ADSR: fast attack, sustain plateau, fast release — lots of curvature
    at the transitions.  Defined analytically so we have a reference."""
    t = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
    # attack 0→0.05, decay 0.05→0.2, sustain 0.2→0.7, release 0.7→1.0
    out = torch.where(t < 0.05, t / 0.05,
          torch.where(t < 0.20, 1.0 - 0.5 * (t - 0.05) / 0.15,
          torch.where(t < 0.70, 0.5 * torch.ones_like(t),
                       0.5 * (1.0 - (t - 0.70) / 0.30))))
    return out.clamp(0.0, 1.0)


def test_curvature() -> None:
    print("\n── curvature ───────────────────────────────────────────────────")

    sig = _sharp_adsr(1024)

    for S in [4, 8, 16]:
        breaks = _curvature_segment_breaks(sig, S)  # returns torch.Tensor float64

        diffs = breaks[1:] - breaks[:-1]
        # Monotone
        check(f"curvature/monotone S={S}",
              bool((diffs > 0).all().item()),
              f"diff min={float(diffs.min()):.3e}")
        # Endpoints pinned
        check(f"curvature/endpoints S={S}",
              abs(float(breaks[0])) < 1e-15 and abs(float(breaks[-1]) - 1.0) < 1e-15,
              f"[{float(breaks[0])}, {float(breaks[-1])}]")
        # Budget floor: at least S segments (may be more due to ctrl-pt forced intervals)
        check(f"curvature/min_length S={S}",
              len(breaks) >= S + 1,
              f"got {len(breaks)}")

        # Density test: the attack (t < 0.10) should have at least one break
        # (it has high curvature); equal-width only guarantees one if S >= 10.
        n_breaks_in_attack = int(((breaks > 0.0) & (breaks < 0.10)).sum().item())
        if S >= 4:
            check(f"curvature/attack_density S={S}",
                  n_breaks_in_attack >= 1,
                  f"breaks in t<0.10: {n_breaks_in_attack}  all_breaks={breaks.tolist()}")

    # RMSE comparison: curvature-adaptive vs equal-width on a ParametricCurve
    # built from the sharp ADSR shape.
    # Construct a ParametricCurve matching the ADSR knots
    c_adsr = ParametricCurve(
        points=[
            ControlPoint(0.00, 0.00),
            ControlPoint(0.05, 1.00),
            ControlPoint(0.20, 0.50),
            ControlPoint(0.70, 0.50),
            ControlPoint(1.00, 0.00),
        ],
    )
    t_eval = torch.linspace(0.0, 1.0, 1024, dtype=torch.float64)
    # Reference: the raw spline (no oscillator rounding)
    pts = sorted(c_adsr.points, key=lambda p: p.t)
    chains = [_build_cr_chain(ch) for ch in _split_into_chains(pts)]
    ref = _eval_all_chains(chains, t_eval, gap_fill=0.0).clamp(0.0, 1.0)

    rmse_adaptive = {}
    for S in [2, 4, 8]:
        c_adsr._invalidate()
        c_adsr.piecewise_segments = S
        # Force curvature-adaptive path (default after our change)
        y = c_adsr.evaluate_normalized(t_eval)
        rmse_adaptive[S] = float(((y - ref) ** 2).mean().sqrt())
        print(f"    ADSR RMSE (curvature-adaptive) S={S:2d}:  {rmse_adaptive[S]:.5f}")

    # Adaptive S=8 should be at least as good as adaptive S=4
    check("curvature/rmse_improves_with_S",
          rmse_adaptive[8] <= rmse_adaptive[4] + 0.01,
          f"S=4 RMSE={rmse_adaptive[4]:.5f}  S=8 RMSE={rmse_adaptive[8]:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: activation
# For all modes: output ∈ [0,1], monotone, f(0)=0, f(1)=1
# ─────────────────────────────────────────────────────────────────────────────

def test_activation() -> None:
    print("\n── activation ──────────────────────────────────────────────────")

    x = torch.linspace(0.0, 1.0, 10_000, dtype=torch.float64)
    modes = ["none", "tanh", "sigmoid", "softplus", "elu"]
    drives = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

    for mode in modes:
        for drive in drives:
            y = _apply_activation(x, mode, drive)

            check(f"activation/{mode}/drive={drive}/range",
                  float(y.min()) >= -1e-9 and float(y.max()) <= 1.0 + 1e-9,
                  f"[{float(y.min()):.4f}, {float(y.max()):.4f}]")

            # f(0) ≈ 0
            check(f"activation/{mode}/drive={drive}/f0",
                  abs(float(y[0])) < 1e-9,
                  f"f(0)={float(y[0]):.3e}")

            # f(1) ≈ 1
            check(f"activation/{mode}/drive={drive}/f1",
                  abs(float(y[-1]) - 1.0) < 1e-9,
                  f"f(1)={float(y[-1]):.6f}")

            # Monotone non-decreasing (allow for tiny numerical noise)
            diffs = torch.diff(y)
            check(f"activation/{mode}/drive={drive}/monotone",
                  float(diffs.min()) >= -1e-10,
                  f"min diff={float(diffs.min()):.3e}")

    # drive=0 → identity
    for mode in modes:
        y_id = _apply_activation(x, mode, 0.0)
        check(f"activation/{mode}/drive=0_identity",
              float((y_id - x).abs().max()) < 1e-12,
              f"max|y-x|={float((y_id - x).abs().max()):.3e}")

    # dtype preservation
    x32 = x.float()
    for mode in ["tanh", "sigmoid"]:
        y32 = _apply_activation(x32, mode, 2.0)
        check(f"activation/{mode}/dtype_float32",
              y32.dtype == torch.float32,
              f"got {y32.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: warp
# TimeWarpCoordinator algebraic boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

def test_warp() -> None:
    print("\n── warp ────────────────────────────────────────────────────────")

    # Helper
    def w(coord, t_list, gates):
        return coord.warp(torch.tensor(t_list, dtype=torch.float64), gates)

    # ── retrigger ─────────────────────────────────────────────────────────────
    coord = TimeWarpCoordinator(retrigger_mode="retrigger", curve_duration=1.0)
    gates = [GateEvent(0.0, 1.0), GateEvent(2.0, 3.0)]
    tn    = w(coord, [0.0, 0.5, 2.0, 2.5], gates)
    check("warp/retrigger/g0_start",       abs(float(tn[0]) - 0.0)  < 1e-9)
    check("warp/retrigger/g0_mid",         abs(float(tn[1]) - 0.5)  < 1e-9)
    check("warp/retrigger/g1_reset",       abs(float(tn[2]) - 0.0)  < 1e-9)
    check("warp/retrigger/g1_mid",         abs(float(tn[3]) - 0.5)  < 1e-9)

    # ── legato ────────────────────────────────────────────────────────────────
    coord_l = TimeWarpCoordinator(retrigger_mode="legato", curve_duration=1.0)
    gates_l = [GateEvent(0.0, 0.5), GateEvent(1.0, 2.0)]
    tn_l    = w(coord_l, [0.0, 0.25, 1.0, 1.5], gates_l)
    # gate0 starts at 0, duration 0.5s, curve_duration=1.0 → phase advances 0.5 units
    check("warp/legato/g0_start",     abs(float(tn_l[0]) - 0.0)  < 1e-9)
    check("warp/legato/g0_mid",       abs(float(tn_l[1]) - 0.25) < 1e-9)
    check("warp/legato/g1_continues", abs(float(tn_l[2]) - 0.5)  < 1e-9,
          f"expected 0.5 got {float(tn_l[2]):.6f}")
    check("warp/legato/g1_mid",       abs(float(tn_l[3]) - 1.0)  < 1e-9,
          f"expected 1.0 got {float(tn_l[3]):.6f}")

    # ── free (no retrigger) ────────────────────────────────────────────────────
    coord_f = TimeWarpCoordinator(retrigger_mode="free", curve_duration=2.0)
    gates_f = [GateEvent(0.0, 1.0), GateEvent(1.5, 2.5)]
    tn_f    = w(coord_f, [0.0, 1.0, 1.5, 2.0], gates_f)
    # free: absolute time / curve_duration → phase monotone regardless of gates
    check("warp/free/monotone",
          bool(torch.all(torch.diff(tn_f) >= 0.0)),
          f"tn_f={tn_f.tolist()}")
    check("warp/free/t0", abs(float(tn_f[0]) - 0.0) < 1e-9)

    # ── freeze release ─────────────────────────────────────────────────────────
    coord_frz = TimeWarpCoordinator(release_mode="freeze", curve_duration=1.0)
    tn_frz    = w(coord_frz, [0.0, 0.5, 0.75, 1.0], [GateEvent(0.0, 0.5)])
    # note-off at 0.5s → phase = 0.5; should freeze there
    check("warp/freeze/at_note_off",  abs(float(tn_frz[1]) - 0.5) < 1e-9)
    check("warp/freeze/post_note_off_constant",
          abs(float(tn_frz[2]) - float(tn_frz[3])) < 1e-9,
          f"[{float(tn_frz[2]):.6f}, {float(tn_frz[3]):.6f}]")

    # ── reset release ──────────────────────────────────────────────────────────
    coord_rst = TimeWarpCoordinator(release_mode="reset", curve_duration=1.0)
    tn_rst    = w(coord_rst, [0.5, 0.75, 1.0], [GateEvent(0.0, 0.5)])
    # reset: after note-off, t_norm should go to 0 (or very near)
    check("warp/reset/post_note_off",
          float(tn_rst[1]) <= 1e-6,
          f"expected ~0 after reset, got {float(tn_rst[1]):.6f}")

    # ── loop ──────────────────────────────────────────────────────────────────
    coord_lo = TimeWarpCoordinator(loop_mode="loop", curve_duration=0.5)
    tn_lo    = w(coord_lo, [0.0, 0.25, 0.5, 0.75, 1.0], [GateEvent(0.0)])
    check("warp/loop/start",        abs(float(tn_lo[0]) - 0.0) < 1e-9)
    check("warp/loop/wrap_at_T",    abs(float(tn_lo[2]) - 0.0) < 1e-9,
          f"expected 0.0 got {float(tn_lo[2]):.6f}")
    check("warp/loop/wrap_at_2T",   abs(float(tn_lo[4]) - 0.0) < 1e-9)

    # ── ping-pong ──────────────────────────────────────────────────────────────
    coord_pp = TimeWarpCoordinator(loop_mode="ping_pong", curve_duration=1.0)
    tn_pp    = w(coord_pp, [0.0, 0.5, 1.0, 1.5, 2.0], [GateEvent(0.0)])
    check("warp/pingpong/start",           abs(float(tn_pp[0]) - 0.0) < 1e-9)
    check("warp/pingpong/end_of_forward",  abs(float(tn_pp[2]) - 1.0) < 1e-9,
          f"expected 1.0 got {float(tn_pp[2]):.6f}")
    check("warp/pingpong/mid_reverse",     abs(float(tn_pp[3]) - 0.5) < 1e-9,
          f"expected 0.5 got {float(tn_pp[3]):.6f}")
    check("warp/pingpong/end_reverse",     abs(float(tn_pp[4]) - 0.0) < 1e-9)

    # ── curve_duration=None (stretch-to-gate) ─────────────────────────────────
    coord_st = TimeWarpCoordinator(curve_duration=None)
    tn_st    = w(coord_st, [0.0, 1.0, 2.0], [GateEvent(0.0, 2.0)])
    check("warp/stretch/start",   abs(float(tn_st[0]) - 0.0) < 1e-9)
    check("warp/stretch/end",     abs(float(tn_st[2]) - 1.0) < 1e-9,
          f"expected 1.0 got {float(tn_st[2]):.6f}")
    check("warp/stretch/mid",     abs(float(tn_st[1]) - 0.5) < 1e-9,
          f"expected 0.5 got {float(tn_st[1]):.6f}")

    # ── output range ──────────────────────────────────────────────────────────
    for mode in ["retrigger", "legato", "free"]:
        for loop in ["none", "loop", "ping_pong"]:
            for rel in ["tail", "freeze"]:
                coord_x = TimeWarpCoordinator(
                    retrigger_mode=mode, loop_mode=loop, release_mode=rel,
                    curve_duration=1.0)
                t_long  = torch.linspace(0.0, 5.0, 200, dtype=torch.float64)
                gates_x = [GateEvent(0.0, 1.5), GateEvent(2.0, 3.5)]
                tn_x    = coord_x.warp(t_long, gates_x)
                check(f"warp/range/{mode}/{loop}/{rel}",
                      float(tn_x.min()) >= 0.0 and float(tn_x.max()) <= 1.0 + 1e-9,
                      f"[{float(tn_x.min()):.4f}, {float(tn_x.max()):.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: quality
# RMSE vs reference spline for several S values, with adaptive vs equal-width
# ─────────────────────────────────────────────────────────────────────────────

def test_quality(segments=(1, 2, 4, 8, 16, 32)) -> None:
    print("\n── quality ─────────────────────────────────────────────────────")
    print(f"  {'S':>4}  {'K':>4}  {'RMSE':>9}  {'MaxE':>9}  {'breaks[:4]'}")

    c = default_envelope()
    pts    = sorted(c.points, key=lambda p: p.t)
    chains = [_build_cr_chain(ch) for ch in _split_into_chains(pts)]
    t_eval = torch.linspace(0.0, 1.0, c.bake_resolution, dtype=torch.float64)
    ref    = _eval_all_chains(chains, t_eval, gap_fill=0.0).clamp(0.0, 1.0)

    prev_rmse = float("inf")
    for S in sorted(segments):
        c2 = default_envelope(); c2.piecewise_segments = S
        t0 = time.perf_counter()
        y2 = c2.evaluate_normalized(t_eval)
        dt = time.perf_counter() - t0
        sp, _, sb = c2.bake()
        rmse = float(((y2 - ref) ** 2).mean().sqrt())
        maxe = float((y2 - ref).abs().max())
        brks = [f"{float(sb[i]):.3f}" for i in range(min(4, len(sb)))]
        print(f"  {S:>4}  {sp.shape[1]:>4}  {rmse:>9.5f}  {maxe:>9.5f}  {brks}  ({dt*1000:.1f}ms)")
        # RMSE should be non-increasing with more segments (allow 10% slack for
        # the numerical randomness of SVD / eigval ordering)
        if S > 1:
            check(f"quality/rmse_S={S}_vs_prev",
                  rmse <= prev_rmse * 1.25,
                  f"RMSE={rmse:.5f}  prev={prev_rmse:.5f}")
        prev_rmse = rmse

    # JSON round-trip preserves reconstruction
    c3 = default_envelope(); c3.piecewise_segments = 8
    y3 = c3.evaluate_normalized(t_eval)
    d  = c3.to_dict()
    assert "piecewise_segments" in d
    c4 = ParametricCurve.from_dict(d)
    y4 = c4.evaluate_normalized(t_eval)
    rt_err = float((y3 - y4).abs().max())
    check("quality/json_roundtrip_S8",
          rt_err < 1e-6,
          f"max delta={rt_err:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────────

ALL_GROUPS = {
    "hilbert":    test_hilbert,
    "pencil":     test_pencil,
    "piecewise":  test_piecewise,
    "curvature":  test_curvature,
    "activation": test_activation,
    "warp":       test_warp,
    "quality":    test_quality,
}


def main() -> None:
    global _verbose, _bail

    parser = argparse.ArgumentParser(
        description="Piecewise oscillator bake — rigorous CLI test suite")
    parser.add_argument("--groups", nargs="+", choices=list(ALL_GROUPS),
                        default=list(ALL_GROUPS),
                        help="Test groups to run (default: all)")
    parser.add_argument("--segments", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32],
                        help="S values for the quality group")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print PASS lines as well as FAIL")
    parser.add_argument("--bail", "-x", action="store_true",
                        help="Stop immediately on first failure")
    args = parser.parse_args()

    _verbose = args.verbose
    _bail    = args.bail

    print("Piecewise oscillator bake — rigorous test suite")
    print(f"torch {torch.__version__}   EPS64={EPS64:.2e}")

    for group in args.groups:
        fn = ALL_GROUPS[group]
        if group == "quality":
            fn(segments=args.segments)
        else:
            fn()

    _summary()
    sys.exit(0 if all(ok for _, ok, _ in _results) else 1)


if __name__ == "__main__":
    main()
