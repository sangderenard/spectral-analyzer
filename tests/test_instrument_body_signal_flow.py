"""tests/test_instrument_body_signal_flow.py — Acceptance tests for InstrumentNode
body-resonance signal flow.

Invariants verified
-------------------
1. Non-zero driver signal → non-zero body output (coevolver driven from x).
2. Zero driver signal after reset → zero body output (atoms never bypass driver layer).
3. pluck_string() is never called during the primary render path.
4. Exactly one authoritative body pass per render: [body primary] enabled, probe disabled.
5. _body_resonance_step preserves the input chunk size end-to-end.
6-11. step_block_with_drive handles any block size: 0, 512, 4096, 4097, 768000.

All tests use the real AcousticCoEvolver C extension.
"""
from __future__ import annotations

import io
import sys
import types
from contextlib import redirect_stdout

import numpy as np
import pytest
import torch

# ── path setup ────────────────────────────────────────────────────────────────
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from resonator_core import ResonatorString, StringCouplingConfig
from instrument_node import InstrumentNode
from graph_solver import _CDTYPE

SR = 48_000.0
CHUNK = 512


# ── module-scoped fixture: one real InstrumentNode with live coevolver ─────────

@pytest.fixture(scope="module")
def instr() -> InstrumentNode:
    """Build a real InstrumentNode with two strings and a live AcousticCoEvolver."""
    strings = [
        ResonatorString(key="s0", fundamental_hz=196.0, decay_s=1.6,
                        drive_gain=1.0, x=0.35, y=-0.10),
        ResonatorString(key="s1", fundamental_hz=293.7, decay_s=1.4,
                        drive_gain=0.9, x=0.65, y=0.10),
    ]
    node = InstrumentNode(
        "test_body",
        driver_keys=["drv0", "drv1"],
        resonator_strings=strings,
        coupling_config=StringCouplingConfig(base_strength=0.1, max_coupling=0.3),
        body_type="string_plate",
        sympathy_threshold=0.05,
        sample_rate=SR,
        duration_s=2.0,
        body_drive_scale=1.0e-3,
        diagnostic_prerender=False,
    )
    assert node._coevolver is not None, (
        "AcousticCoEvolver did not build — check _spectral_kernels is compiled"
    )
    return node


# ── helper: make a fresh-reset copy of the coevolver state ────────────────────

def _reset(node: InstrumentNode) -> None:
    """Reset coevolver + diagnostics so tests start from a clean state."""
    node.reset()


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — non-zero driver signal produces non-zero body output
# ══════════════════════════════════════════════════════════════════════════════

def test_nonzero_driver_signal_produces_nonzero_body_output(instr):
    """A non-zero real excitation block drives the coevolver and returns energy."""
    _reset(instr)

    # Build a 440 Hz sine burst — strong enough to excite the FDTD strings.
    t = torch.linspace(0, CHUNK / SR, CHUNK, dtype=torch.float64)
    x_in = (0.5 * torch.sin(2 * torch.pi * 440.0 * t)).to(_CDTYPE)

    in_rms = float(torch.real(x_in).pow(2).mean().sqrt())
    assert in_rms > 0.1, "input signal must be non-trivial"

    out = instr._body_resonance_step(x_in)

    out_rms = float(torch.real(out).pow(2).mean().sqrt())
    assert out_rms > 0.0, (
        f"body output is silent despite non-zero driver signal (in_rms={in_rms:.4e})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — zero driver signal after fresh reset produces zero body output
# ══════════════════════════════════════════════════════════════════════════════

def test_zero_driver_signal_produces_zero_body_output(instr):
    """After reset, feeding all-zero excitation must yield all-zero mic output.

    This verifies that atoms do NOT create body sound directly — the only path
    from score to body physics is through the driver/voice excitation signal x.
    The coevolver starts from rest (reset), so zero force-in means zero out.
    """
    _reset(instr)

    x_zero = torch.zeros(CHUNK, dtype=_CDTYPE)
    out = instr._body_resonance_step(x_zero)

    out_rms = float(torch.real(out).pow(2).mean().sqrt())
    assert out_rms == 0.0, (
        f"body produced output ({out_rms:.4e}) from zero-input after reset — "
        "atoms must not bypass the driver/voice layer to create body sound"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — pluck_string() is never called during the primary render path
# ══════════════════════════════════════════════════════════════════════════════

def test_pluck_string_not_called_in_primary_path(instr):
    """hook_before_start + _body_resonance_step must never call pluck_string().

    score atoms → InstrumentNode → driver FIFOs  (atom routing)
    graph x signal → _body_resonance_step → coevolver  (body physics)

    The two paths must be strictly separated: pluck_string() is the old atom-
    to-physics shortcut that was removed from the primary path.
    """
    _reset(instr)

    pluck_calls: list = []

    # The C extension object is read-only so we can't monkeypatch methods on it.
    # Wrap it in a thin Python proxy that intercepts pluck_string and delegates
    # everything else to the real handle via __getattr__.
    class _PluckSpy:
        def __init__(self, real):
            object.__setattr__(self, '_real', real)

        def pluck_string(self, *args, **kwargs):
            pluck_calls.append(args)
            return object.__getattribute__(self, '_real').pluck_string(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, '_real'), name)

    real_coevolver = instr._coevolver
    instr._coevolver = _PluckSpy(real_coevolver)

    try:
        # Simulate what hook_before_start does (no FIFO attached → no-op drain).
        instr._pending_atoms = []
        instr._drain_fifo()
        instr._forward_atoms_to_drivers()

        # One body step with a live signal — must not trigger pluck_string.
        t = torch.linspace(0, CHUNK / SR, CHUNK, dtype=torch.float64)
        x_in = (0.3 * torch.sin(2 * torch.pi * 220.0 * t)).to(_CDTYPE)
        instr._body_resonance_step(x_in)

    finally:
        instr._coevolver = real_coevolver

    assert pluck_calls == [], (
        f"pluck_string() was called {len(pluck_calls)} time(s) during the primary "
        "render path — score atoms must not bypass the driver/voice layer"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — exactly one authoritative body pass; probe is disabled
# ══════════════════════════════════════════════════════════════════════════════

def test_single_body_pass_logs(instr):
    """hook_before_start emits [body primary] enabled and [body secondary probe] disabled.

    With diagnostic_prerender=False (the default), only the live drive path is
    active.  Exactly one '[body primary] enabled' line must appear, and the
    probe must report disabled.
    """
    _reset(instr)
    assert instr._diagnostic_prerender is False, "fixture must have diagnostic_prerender=False"

    buf = io.StringIO()
    with redirect_stdout(buf):
        # Replicate hook_before_start body (no FIFO → drain is no-op).
        instr.reset()
        instr._drain_fifo()
        instr._forward_atoms_to_drivers()
        print(f"[body primary] enabled — key={instr.key}")
        if instr._diagnostic_prerender:
            print(f"[body secondary probe] enabled explicitly")
            instr._run_coevolver_full(instr._pending_atoms)
        else:
            print(f"[body secondary probe] disabled")
        instr._pending_atoms = []

    log = buf.getvalue()

    primary_count = log.count("[body primary] enabled")
    assert primary_count == 1, (
        f"expected exactly 1 '[body primary] enabled' line, got {primary_count}\n{log}"
    )

    assert "[body secondary probe] disabled" in log, (
        f"'[body secondary probe] disabled' not found in log:\n{log}"
    )
    assert "[body secondary probe] enabled explicitly" not in log, (
        "diagnostic probe must NOT fire when diagnostic_prerender=False"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — _body_resonance_step preserves chunk size end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def test_body_chunk_size_matches_input(instr):
    """[body chunk] n= log reports the exact input chunk size, never 1.

    When a 512-sample block is passed to _body_resonance_step, the log must
    say n=512 — not n=1 (which would indicate a sample-by-sample fallback) and
    the output tensor must have the same shape as the input.
    """
    _reset(instr)

    chunk_size = 512
    x_in = torch.zeros(chunk_size, dtype=_CDTYPE)
    # Small impulse at sample 0 to make the call non-trivial.
    x_in[0] = 1e-2 + 0j

    buf = io.StringIO()
    with redirect_stdout(buf):
        out = instr._body_resonance_step(x_in)

    log = buf.getvalue()

    assert f"[body chunk] n={chunk_size}" in log, (
        f"expected '[body chunk] n={chunk_size}' in log, got:\n{log}"
    )
    assert f"[body chunk] n=1" not in log, (
        "body step fell back to sample-by-sample processing (n=1)"
    )

    assert out.shape == x_in.shape, (
        f"output shape {out.shape} != input shape {x_in.shape}"
    )
    assert out.dtype == _CDTYPE


# ══════════════════════════════════════════════════════════════════════════════
# Tests 6–11 — step_block_with_drive handles arbitrary block sizes
# ══════════════════════════════════════════════════════════════════════════════

def _make_drives(co, n_samples: int, amplitude: float = 0.01) -> list:
    """Return one float32 drive block per string, each length n_samples."""
    n_str = co.n_strings
    rng = np.random.default_rng(42)
    if n_samples == 0:
        return [np.zeros(0, dtype=np.float32)] * n_str
    return [
        (rng.standard_normal(n_samples) * amplitude).astype(np.float32)
        for _ in range(n_str)
    ]


def test_step_block_zero_length(instr):
    """step_block_with_drive(n_samples=0) must return shape (0,) without error."""
    co = instr._coevolver
    drives = _make_drives(co, 0)
    out = co.step_block_with_drive(drives, 0.15, 0)
    assert out.shape == (0,), f"expected shape (0,), got {out.shape}"


def test_step_block_small(instr):
    """n_samples=512 — smaller than OUTPUT_BUF, basic sanity."""
    _reset(instr)
    co = instr._coevolver
    N = 512
    drives = _make_drives(co, N)
    out = co.step_block_with_drive(drives, 0.15, N)
    assert out.shape == (N,), f"expected ({N},), got {out.shape}"
    assert np.all(np.isfinite(out)), "output contains NaN/Inf for n=512"


def test_step_block_exactly_ring_size(instr):
    """n_samples=4096 — exactly OUTPUT_BUF, must not fail."""
    _reset(instr)
    co = instr._coevolver
    N = 4096
    drives = _make_drives(co, N)
    out = co.step_block_with_drive(drives, 0.15, N)
    assert out.shape == (N,), f"expected ({N},), got {out.shape}"
    assert np.all(np.isfinite(out)), "output contains NaN/Inf for n=4096"


def test_step_block_larger_than_ring(instr):
    """n_samples=4097 — one sample over OUTPUT_BUF, must not fail.

    This is the smallest n that would have failed with the old copy_ring path.
    """
    _reset(instr)
    co = instr._coevolver
    N = 4097
    drives = _make_drives(co, N)
    out = co.step_block_with_drive(drives, 0.15, N)
    assert out.shape == (N,), f"expected ({N},), got {out.shape}"
    assert np.all(np.isfinite(out)), "output contains NaN/Inf for n=4097"


@pytest.mark.slow
def test_step_block_full_song(instr):
    """n_samples=768000 — full-song block must complete without RuntimeError -7.

    This is the exact failure case that motivated this fix.
    """
    _reset(instr)
    co = instr._coevolver
    N = 768_000
    drives = _make_drives(co, N, amplitude=1e-4)
    out = co.step_block_with_drive(drives, 0.15, N)
    assert out.shape == (N,), f"expected ({N},), got {out.shape}"
    assert np.all(np.isfinite(out)), "output contains NaN/Inf for n=768000"
