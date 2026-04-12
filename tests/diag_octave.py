"""Per-octave diagnostic for iCQT reconstruction.

Shows per-octave geometry, D[n] profile, time alignment, and
per-chunk waveform correlation — treating each octave as its
own universe.
"""
import numpy as np
import math, sys, os, time

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: F401  — headless stubs

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bass_viewer import ViewportSynthPlayer
from bass_analysis import compute_cqt, AnalysisConfig

import torch

# ── Configuration ───────────────────────────────────────────────
sr = 44100
dur = 1.0
hop = 256
bpo = 48
N = int(sr * dur)
t_arr = np.arange(N, dtype=np.float64) / sr

# Test signals
signals = {
    "440Hz_tone": np.sin(2 * np.pi * 440 * t_arr),
    "100Hz_tone": np.sin(2 * np.pi * 100 * t_arr),
    "white_noise": np.random.default_rng(42).standard_normal(N),
}

cfg = AnalysisConfig()
cfg.cqt_fmin = 27.5
cfg.cqt_fmax = sr / 2 * 0.95
cfg.bins_per_octave = bpo
cfg.hop_length = hop

Q = 1.0 / (2.0 ** (1.0 / bpo) - 1.0)


def xcorr_peak(a: np.ndarray, b: np.ndarray):
    """Normalised cross-correlation peak and its lag in samples."""
    from numpy.fft import fft, ifft
    n = len(a)
    cc = np.real(ifft(fft(a, 2*n) * np.conj(fft(b, 2*n))))
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-30
    idx = np.argmax(cc)
    lag = idx if idx <= n else idx - 2*n
    return cc[idx] / denom, lag


def run_diagnostic(name: str, signal: np.ndarray):
    print(f"\n{'='*60}")
    print(f"Signal: {name}  ({len(signal)} samples, {dur}s)")
    print(f"{'='*60}")

    # Forward CQT (reference)
    freqs, times, power, cqt = compute_cqt(signal, sr, cfg)
    n_bins, n_frames = cqt.shape
    print(f"CQT: {n_bins} bins × {n_frames} frames")

    # ── Per-octave geometry report ──────────────────────────────
    n_octaves = int(math.ceil(n_bins / bpo))
    print(f"\n--- Per-octave geometry ---")
    for octave in range(n_octaves):
        gbin_start = max(0, n_bins - (octave + 1) * bpo)
        gbin_end = n_bins - octave * bpo
        oct_sr = sr >> octave
        oct_freqs = freqs[gbin_start:gbin_end]
        if len(oct_freqs) == 0:
            continue
        klens = np.ceil(Q * oct_sr / oct_freqs).astype(int)
        max_klen = int(klens.max())
        min_klen = int(klens.min())
        center = max_klen // 2
        repeat = 1 << octave
        n_oct_frames = len(range(0, n_frames, repeat))
        print(f"  Oct {octave}: bins {gbin_start}-{gbin_end-1} "
              f"({len(oct_freqs)} bins), sr={oct_sr}, "
              f"klen=[{min_klen}..{max_klen}], center={center}, "
              f"repeat={repeat}, oct_frames={n_oct_frames}")
        print(f"         center*repeat={center*repeat} samples "
              f"({center*repeat/sr*1000:.1f}ms at full rate)")
        print(f"         freq range: {oct_freqs[0]:.1f} - {oct_freqs[-1]:.1f} Hz")

    # ── Full round-trip ─────────────────────────────────────────
    sp = ViewportSynthPlayer.__new__(ViewportSynthPlayer)
    sp.sr = sr
    sp.hop_length = hop
    sp.bins_per_octave = bpo
    sp._has_cuda = torch.cuda.is_available()

    print(f"\n--- Full round-trip ---")
    t0 = time.time()
    cqt128 = cqt.astype(np.complex128)
    recon = sp._inverse_cqt_channel(cqt128, freqs, 0, n_bins, n_frames)
    elapsed = time.time() - t0
    mn = min(N, len(recon))

    rms_orig = np.sqrt(np.mean(signal[:mn]**2))
    rms_recon = np.sqrt(np.mean(recon[:mn]**2))
    a = signal[:mn] - signal[:mn].mean()
    b = recon[:mn] - recon[:mn].mean()
    peak, lag = xcorr_peak(a, b)
    print(f"  Time: {elapsed:.2f}s")
    print(f"  RMS: orig={rms_orig:.4f}, recon={rms_recon:.4f}, ratio={rms_recon/max(rms_orig,1e-30):.4f}")
    print(f"  Xcorr: {peak:.4f} at lag={lag} samples ({lag/sr*1000:.2f}ms)")

    # Per-chunk correlation
    n_chunks = 10
    chunk_sz = mn // n_chunks
    print(f"\n  Chunk-level (n={n_chunks}):")
    for i in range(n_chunks):
        s = signal[i*chunk_sz:(i+1)*chunk_sz]
        r = recon[i*chunk_sz:(i+1)*chunk_sz]
        so = np.sqrt(np.mean(s**2))
        ro = np.sqrt(np.mean(r**2))
        if np.std(s) > 1e-12 and np.std(r) > 1e-12:
            cc = np.corrcoef(s, r)[0, 1]
        else:
            cc = 0.0
        t_ms = i * chunk_sz / sr * 1000
        print(f"    [{i:2d}] t={t_ms:6.0f}ms  "
              f"orig_rms={so:.4f}  recon_rms={ro:.4f}  "
              f"corr={cc:+.4f}")


# ── Run all ─────────────────────────────────────────────────────
for name, sig in signals.items():
    run_diagnostic(name, sig)
