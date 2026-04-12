"""Diagnostic script for CQT forward/inverse round-trip."""
import numpy as np
import math
import sys
import os

# Ensure conftest stubs are loaded
sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bass_viewer import ViewportSynthPlayer
from bass_analysis import compute_cqt, AnalysisConfig
import librosa

sr = 44100
dur = 0.2
hop = 256
bpo = 48
N = int(sr * dur)
t_arr = np.arange(N, dtype=np.float64) / sr
signal = np.sin(2 * np.pi * 440 * t_arr)

cfg = AnalysisConfig()
cfg.cqt_fmin = 27.5
cfg.cqt_fmax = sr / 2 * 0.95
cfg.bins_per_octave = bpo
cfg.hop_length = hop
freqs_ref, times_ref, power_ref, cqt_ref = compute_cqt(signal, sr, cfg)
n_bins = cqt_ref.shape[0]
n_frames = cqt_ref.shape[1]

sp = ViewportSynthPlayer.__new__(ViewportSynthPlayer)
sp.sr = sr
sp.hop_length = hop
sp.bins_per_octave = bpo
sp._has_cuda = True

print(f"n_bins={n_bins}, n_frames={n_frames}")

# Test 1: compute_cqt → inverse round-trip
cqt_128 = cqt_ref.astype(np.complex128)
recon = sp._inverse_cqt_channel(cqt_128, freqs_ref, 0, n_bins, n_frames)
mn = min(N, len(recon))
a = signal[:mn] - signal[:mn].mean()
b = recon[:mn] - recon[:mn].mean()
from numpy.fft import fft, ifft
cc = np.real(ifft(fft(a, 2*mn) * np.conj(fft(b, 2*mn))))
peak = cc.max() / (np.sqrt(np.sum(a**2)*np.sum(b**2))+1e-30)
print(f"\ncompute_cqt -> inverse:")
print(f"  RMS orig={np.sqrt(np.mean(signal[:mn]**2)):.4f}, recon={np.sqrt(np.mean(recon[:mn]**2)):.4f}")
print(f"  xcorr = {peak:.4f}")

# Chunk analysis
nc = 10
chunk = mn // nc
for i in range(nc):
    s = recon[i*chunk:(i+1)*chunk]
    o = signal[i*chunk:(i+1)*chunk]
    print(f"  chunk {i}: recon_rms={np.sqrt(np.mean(s**2)):.4f} orig_rms={np.sqrt(np.mean(o**2)):.4f}")

# Test 2: _forward_cqt_channel → inverse round-trip
cqt_mine = sp._forward_cqt_channel(signal, freqs_ref, 0, n_bins, n_frames)
recon2 = sp._inverse_cqt_channel(cqt_mine, freqs_ref, 0, n_bins, n_frames)
b2 = recon2[:mn] - recon2[:mn].mean()
cc2 = np.real(ifft(fft(a, 2*mn) * np.conj(fft(b2, 2*mn))))
peak2 = cc2.max() / (np.sqrt(np.sum(a**2)*np.sum(b2**2))+1e-30)
print(f"\n_forward_cqt_channel -> inverse:")
print(f"  RMS orig={np.sqrt(np.mean(signal[:mn]**2)):.4f}, recon={np.sqrt(np.mean(recon2[:mn]**2)):.4f}")
print(f"  xcorr = {peak2:.4f}")

# Test 3: noise
np.random.seed(42)
noise = np.random.randn(N)
_, _, _, cqt_noise = compute_cqt(noise, sr, cfg)
recon_n = sp._inverse_cqt_channel(cqt_noise.astype(np.complex128), freqs_ref, 0, n_bins, n_frames)
mn3 = min(N, len(recon_n))
a3 = noise[:mn3] - noise[:mn3].mean()
b3 = recon_n[:mn3] - recon_n[:mn3].mean()
cc3 = np.real(ifft(fft(a3, 2*mn3) * np.conj(fft(b3, 2*mn3))))
peak3 = cc3.max() / (np.sqrt(np.sum(a3**2)*np.sum(b3**2))+1e-30)
print(f"\nNoise round-trip:")
print(f"  RMS orig={np.sqrt(np.mean(noise[:mn3]**2)):.4f}, recon={np.sqrt(np.mean(recon_n[:mn3]**2)):.4f}")
print(f"  xcorr = {peak3:.4f}")
