"""Diagnostic: find optimal extraction offset per octave."""
import numpy as np
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bass_viewer import ViewportSynthPlayer
from bass_analysis import compute_cqt, AnalysisConfig

sr = 44100; dur = 0.5; hop = 256; bpo = 48
N = int(sr * dur)
t_arr = np.arange(N, dtype=np.float64) / sr
signal = np.sin(2 * np.pi * 440 * t_arr)

cfg = AnalysisConfig()
cfg.cqt_fmin = 27.5; cfg.cqt_fmax = sr / 2 * 0.95
cfg.bins_per_octave = bpo; cfg.hop_length = hop
freqs, _, _, cqt = compute_cqt(signal, sr, cfg)
n_bins, n_frames = cqt.shape
print(f"n_bins={n_bins}, n_frames={n_frames}, signal_len={N}")

# Quick test: what xcorr do we get with NO extraction offset?
# i.e., pure convolution output starting from sample 0
import torch

sp = ViewportSynthPlayer.__new__(ViewportSynthPlayer)
sp.sr = sr; sp.hop_length = hop; sp.bins_per_octave = bpo
sp._has_cuda = torch.cuda.is_available()

cqt128 = cqt.astype(np.complex128)
recon = sp._inverse_cqt_channel(cqt128, freqs, 0, n_bins, n_frames)

mn = min(N, len(recon))
a = signal[:mn] - signal[:mn].mean()
b = recon[:mn] - recon[:mn].mean()
from numpy.fft import fft, ifft
cc_full = np.real(ifft(fft(a, 2*mn) * np.conj(fft(b, 2*mn))))
peak_idx = np.argmax(cc_full)
peak_val = cc_full[peak_idx] / (np.sqrt(np.sum(a**2)*np.sum(b**2))+1e-30)
if peak_idx > mn:
    peak_idx -= 2*mn
print(f"\nPeak xcorr = {peak_val:.4f} at lag = {peak_idx} samples ({peak_idx/sr*1000:.1f} ms)")
print(f"RMS orig={np.sqrt(np.mean(a**2)):.4f}, recon={np.sqrt(np.mean(b**2)):.4f}")

# What's the expected center offset?
Q = 1.0 / (2.0 ** (1.0/bpo) - 1.0)
for octave in range(3):
    oct_sr = sr >> octave
    gbin_start = max(0, n_bins - (octave+1)*bpo)
    gbin_end = n_bins - octave*bpo 
    oct_freqs = freqs[gbin_start:gbin_end]
    if len(oct_freqs) == 0: continue
    klens = np.ceil(Q * oct_sr / oct_freqs).astype(int)
    max_klen = int(klens.max())
    center = max_klen // 2
    repeat = 1 << octave
    print(f"  Octave {octave}: max_klen={max_klen}, center={center}, "
          f"center*repeat={center*repeat} samples at full rate "
          f"({center*repeat/sr*1000:.1f} ms)")

# Print chunk-level correlation
chunk_sz = mn // 20
for i in range(20):
    s = signal[i*chunk_sz:(i+1)*chunk_sz]
    r = recon[i*chunk_sz:(i+1)*chunk_sz]
    sc = np.corrcoef(s, r)[0,1] if np.std(s) > 0 and np.std(r) > 0 else 0
    print(f"  chunk {i:2d}: orig_rms={np.sqrt(np.mean(s**2)):.4f} "
          f"recon_rms={np.sqrt(np.mean(r**2)):.4f} corr={sc:.4f}")
