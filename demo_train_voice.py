"""demo_train_voice.py — Train a voice to match a randomly-parameterised target.

Workflow
--------
1. Build a random AnalyticVoice target and render it via _synthesize_voice
   (the numpy reference path — no gradients involved).
2. Build a second voice with default parameters and wire it into a GraphSolver
   via build_voice_mixer_network.
3. Run VoiceTrainer.fit() to minimise complex-L2 loss against the target,
   one solver tick per sample.
4. Render the trained voice by ticking the solver sample-by-sample.
5. Write target.wav and trained.wav — both projected to real (float32).

Usage
-----
    python demo_train_voice.py [--freq 440] [--dur 0.5] [--sr 48000]
                               [--epochs 3] [--lr 3e-3]
                               [--seed 0] [--out-dir .]
"""
from __future__ import annotations

import argparse
import math
import random
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# ---------------------------------------------------------------------------
# Minimal AnalyticVoice stub so we do not import the full pygame GUI driver.
# Only fields that _synthesize_voice and VoiceTorchOscillator.from_voice read
# are needed.
# ---------------------------------------------------------------------------

class _ADSRParams:
    def __init__(self, attack=0.005, decay=0.04, sustain=0.75, release=0.08, peak=1.0):
        self.attack  = attack
        self.decay   = decay
        self.sustain = sustain
        self.release = release
        self.peak    = peak

    def to_knots(self, dur: float):
        a   = min(self.attack,  dur)
        dk  = min(self.decay,   dur - a)
        r   = min(self.release, dur - a - dk)
        ts  = a + dk
        te  = dur - r
        return [
            [0.0,  0.0],
            [a,    self.peak],
            [ts,   self.sustain],
            [te,   self.sustain],
            [dur,  0.0],
        ]


class _ChirpSpec:
    def __init__(self, f_delta_start=0.0, f_delta_end=0.0,
                 chirp_type="none", tau=0.5, chirp_power=1.0):
        self.f_delta_start = f_delta_start
        self.f_delta_end   = f_delta_end
        self.chirp_type    = chirp_type
        self.tau           = tau
        self.chirp_power   = chirp_power


class _Voice:
    """Minimal AnalyticVoice-compatible object."""
    def __init__(
        self,
        key: str,
        freq_hz: float = 440.0,
        amplitude: float = 1.0,
        phase_origin: float = 0.0,
        adsr: _ADSRParams | None = None,
        chirp: _ChirpSpec | None = None,
        env_type: str = "adsr",
        manifold_type: str = "pure",
        harmonic_count: int = 1,
        harmonic_brightness: float = 1.0,
        harmonic_warp_strength: float = 0.0,
        semitone_offset: float = 0.0,
        fm=None,
        am=None,
        loop_enabled: bool = False,
        loop_start: float = 0.1,
        loop_end: float = 0.9,
        emission_mode: str = "single",
        pre_delay: float = 0.0,
    ):
        self.key                  = key
        self.freq_hz              = freq_hz
        self.amplitude            = amplitude
        self.phase_origin         = phase_origin
        self.adsr                 = adsr or _ADSRParams()
        self.chirp                = chirp or _ChirpSpec()
        self.env_type             = env_type
        self.manifold_type        = manifold_type
        self.harmonic_count       = harmonic_count
        self.harmonic_brightness  = harmonic_brightness
        self.harmonic_warp_strength = harmonic_warp_strength
        self.semitone_offset      = semitone_offset
        self.fm                   = fm
        self.am                   = am
        self.loop_enabled         = loop_enabled
        self.loop_start           = loop_start
        self.loop_end             = loop_end
        self.emission_mode        = emission_mode
        self.pre_delay            = pre_delay
        self.muted                = False
        self.env_knots            = self.adsr.to_knots(1.0)

    def active_knots(self):
        return self.adsr.to_knots(1.0)


# ---------------------------------------------------------------------------
# Numpy reference renderer — mirrors the core of _synthesize_voice without
# importing the full analytic_driver GUI module.
# ---------------------------------------------------------------------------

def _render_voice_numpy(voice: _Voice, sr: float, dur: float) -> np.ndarray:
    """Render one voice to a complex128 numpy array using the reference path."""
    n   = int(sr * dur)
    t   = np.arange(n, dtype=np.float64) / sr

    # Semitone offset
    freq = float(voice.freq_hz) * (2.0 ** (float(voice.semitone_offset) / 12.0))

    # Chirp
    ct = getattr(voice.chirp, "chirp_type", "none")
    f0 = float(getattr(voice.chirp, "f_delta_start", 0.0))
    f1 = float(getattr(voice.chirp, "f_delta_end",   0.0))
    if ct == "linear":
        chirp_rate = (f1 - f0) / max(dur, 1e-9)
        f_inst = freq + f0 + chirp_rate * t
    elif ct == "exponential":
        tau = max(float(getattr(voice.chirp, "tau", 0.5)), 1e-6)
        f_inst = freq + f0 * np.exp(-t / tau) + f1 * (1.0 - np.exp(-t / tau))
    else:
        f_inst = np.full(n, freq, dtype=np.float64)

    phase = np.cumsum(2.0 * np.pi * f_inst / sr) + float(voice.phase_origin)

    # Manifold
    if voice.manifold_type in ("harmonic", "harmonic_warp"):
        sig = np.zeros(n, dtype=np.complex128)
        for k in range(1, voice.harmonic_count + 1):
            amp_k = 1.0 / (k ** float(voice.harmonic_brightness))
            warp  = 1.0 + float(voice.harmonic_warp_strength) * math.log(k + 1)
            sig  += amp_k * np.exp(1j * phase * k * warp)
        # normalise to unit peak
        peak = np.abs(sig).max()
        if peak > 1e-12:
            sig /= peak
    else:
        sig = np.exp(1j * phase)

    # ADSR envelope
    knots = voice.adsr.to_knots(1.0)  # normalised [0,1] time
    t_norm = (t / max(dur, 1e-12)).clip(0.0, 1.0)
    env = np.interp(t_norm, [k[0] for k in knots], [k[1] for k in knots])
    env = env.clip(0.0, None)

    # Amplitude
    out = float(voice.amplitude) * env * sig

    # Pre-delay
    if voice.pre_delay > 0.0:
        n_silence = min(n, int(math.ceil(voice.pre_delay * sr)))
        out[:n_silence] = 0.0

    return out.astype(np.complex128)


# ---------------------------------------------------------------------------
# WAV writer (float32 mono, no scipy dependency)
# ---------------------------------------------------------------------------

def _write_wav(path: Path, signal_real: np.ndarray, sr: int) -> None:
    """Write a float32 mono WAV.  Clips to [-1, 1] then scales to int16."""
    x = np.asarray(signal_real, dtype=np.float32)
    peak = np.abs(x).max()
    if peak > 1e-12:
        x = x / peak  # normalise to ±1
    x_int16 = (x * 32767.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(x_int16.tobytes())
    print(f"  wrote {path}  ({len(x_int16)} samples, {len(x_int16)/sr:.3f} s)")


# ---------------------------------------------------------------------------
# Render a trained solver sample-by-sample (no grad)
# ---------------------------------------------------------------------------

def _render_trained(solver, vnodes, mixer_key: str, n: int) -> np.ndarray:
    """Step the solver through n ticks with no_grad and collect the mixer output."""
    solver.reset()
    for vnode in vnodes.values():
        vnode.reset()

    out = np.zeros(n, dtype=np.complex128)
    with torch.no_grad():
        for i in range(n):
            for vnode in vnodes.values():
                vnode.set_sample(i)
            results = solver.step({})
            v = results.get(mixer_key)
            if v is not None:
                out[i] = complex(v.item())
    return out


# ---------------------------------------------------------------------------
# Random voice factory
# ---------------------------------------------------------------------------

def _random_voice(key: str, freq_hz: float, rng: random.Random) -> _Voice:
    # POLICY: discrete (non-differentiable) mode flags are locked to their
    # continuous-safe defaults.  Randomising them produces targets the
    # gradient-based optimiser can never match through complex-L2 loss alone
    # because manifold_type and chirp_type are plain Python string attributes
    # on VoiceTorchOscillator — not Parameters, not relaxed categoricals.
    # Lift this restriction only if the training loop supplies a
    # Gumbel-softmax (or equivalent) relaxation over those choices.
    adsr = _ADSRParams(
        attack  = rng.uniform(0.002, 0.15),
        decay   = rng.uniform(0.01,  0.12),
        sustain = rng.uniform(0.3,   0.95),
        release = rng.uniform(0.02,  0.20),
        peak    = rng.uniform(0.8,   1.2),
    )
    chirp = _ChirpSpec(
        chirp_type    = "none",
        f_delta_start = rng.uniform(-80.0,  80.0),
        f_delta_end   = rng.uniform(-20.0,  20.0),
    )
    return _Voice(
        key               = key,
        freq_hz           = freq_hz,
        amplitude         = rng.uniform(0.5, 1.0),
        phase_origin      = rng.uniform(-math.pi, math.pi),
        adsr              = adsr,
        chirp             = chirp,
        manifold_type     = "pure",
        harmonic_count    = 1,
        harmonic_brightness = 1.0,
        semitone_offset   = rng.uniform(-0.5, 0.5),
        env_type          = "adsr",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train a voice to match a random target.")
    ap.add_argument("--freq",   type=float, default=440.0,  help="Note frequency (Hz)")
    ap.add_argument("--dur",    type=float, default=0.5,    help="Note duration (s)")
    ap.add_argument("--sr",     type=int,   default=48000,  help="Sample rate")
    ap.add_argument("--epochs", type=int,   default=3,      help="Training passes over the signal")
    ap.add_argument("--lr",     type=float, default=3e-3,   help="Adam learning rate")
    ap.add_argument("--seed",   type=int,   default=0,      help="RNG seed for target voice")
    ap.add_argument("--out-dir", type=str,  default=".",    help="Output directory for WAV files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    sr  = float(args.sr)
    dur = float(args.dur)
    N   = int(sr * dur)

    # ── 1. Build and render target ──────────────────────────────────────────
    print(f"\n=== Target voice (seed={args.seed}, freq={args.freq} Hz) ===")
    target_voice = _random_voice("target", args.freq, rng)
    print(f"  adsr  : a={target_voice.adsr.attack:.3f}  d={target_voice.adsr.decay:.3f}"
          f"  s={target_voice.adsr.sustain:.3f}  r={target_voice.adsr.release:.3f}"
          f"  pk={target_voice.adsr.peak:.3f}")
    print(f"  manifold={target_voice.manifold_type}  harmonics={target_voice.harmonic_count}"
          f"  brightness={target_voice.harmonic_brightness:.2f}")
    print(f"  chirp={target_voice.chirp.chirp_type}"
          f"  Δf_start={target_voice.chirp.f_delta_start:.1f}"
          f"  Δf_end={target_voice.chirp.f_delta_end:.1f}")

    target_np = _render_voice_numpy(target_voice, sr, dur)
    _write_wav(out_dir / "target.wav", target_np.real.astype(np.float32), args.sr)

    target_tensor = torch.from_numpy(target_np)  # complex128, shape (N,)

    # ── 2. Build trainable voice (default params, same pitch) ───────────────
    print("\n=== Building trainable voice (default params) ===")
    from voice_graph_node import build_voice_mixer_network, VoiceTrainer

    # Default voice — ADSR, pure sine, no chirp
    train_voice = _Voice(
        key     = "v1",
        freq_hz = args.freq,
        adsr    = _ADSRParams(),  # default ADSR
    )

    solver, vnodes, mixer_node = build_voice_mixer_network(
        [train_voice],
        sample_rate = sr,
        duration    = dur,
        mixer_key   = "mix_out",
    )

    osc = vnodes["v1"].oscillator
    all_params = list(osc.parameters())
    print(f"  trainable parameters: {sum(p.numel() for p in all_params)}")

    opt = optim.Adam(all_params, lr=args.lr)

    trainer = VoiceTrainer(
        solver,
        vnodes,
        opt,
        target_key = "mix_out",
    )

    # ── 3. Train ────────────────────────────────────────────────────────────
    print(f"\n=== Training ({args.epochs} epoch(s), {N} ticks/epoch) ===")
    losses = trainer.fit(target_tensor, n_steps=args.epochs, verbose=True)

    if losses:
        print(f"\n  initial loss : {losses[0]:.4e}")
        print(f"  final loss   : {losses[-1]:.4e}")
        ratio = max(losses[0], 1e-30) / max(losses[-1], 1e-30)
        print(f"  reduction    : {ratio:.0f}×")

    # ── 4. Render trained inference ─────────────────────────────────────────
    print("\n=== Rendering trained inference ===")
    trained_np = _render_trained(solver, vnodes, "mix_out", N)
    _write_wav(out_dir / "trained.wav", trained_np.real.astype(np.float32), args.sr)

    # ── 5. Quick summary ────────────────────────────────────────────────────
    print("\n=== Summary ===")
    target_rms  = float(np.sqrt(np.mean(np.abs(target_np)**2)))
    trained_rms = float(np.sqrt(np.mean(np.abs(trained_np)**2)))
    err_rms     = float(np.sqrt(np.mean(np.abs(target_np - trained_np)**2)))
    print(f"  target RMS  : {target_rms:.4f}")
    print(f"  trained RMS : {trained_rms:.4f}")
    print(f"  error RMS   : {err_rms:.4f}")
    print(f"\n  target.wav  → {out_dir / 'target.wav'}")
    print(f"  trained.wav → {out_dir / 'trained.wav'}")


if __name__ == "__main__":
    main()
