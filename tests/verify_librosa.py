"""Measure the exact scaling relationship between compute_cqt and librosa.cqt,
then verify librosa.icqt round-trips correctly."""
import numpy as np
import librosa
import math, sys, os

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: headless stubs

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bass_analysis import compute_cqt, AnalysisConfig

sr = 44100
N = int(sr * 1.0)
t = np.arange(N, dtype=np.float64) / sr
signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

bpo = 48
hop = 256
fmin = 27.5
fmax = sr / 2 * 0.95
n_bins = int(math.floor(bpo * math.log2(fmax / fmin)))
Q = 1.0 / (2.0 ** (1.0 / bpo) - 1.0)

print(f"=== Config: sr={sr}, bpo={bpo}, hop={hop}, fmin={fmin}, n_bins={n_bins} ===\n")

# --- 1. Librosa round-trip ---
C_lib = librosa.cqt(signal, sr=sr, hop_length=hop, fmin=fmin,
                     n_bins=n_bins, bins_per_octave=bpo, scale=True)
y_lib = librosa.icqt(C_lib, sr=sr, hop_length=hop, fmin=fmin,
                      bins_per_octave=bpo, length=N)
cc_lib = np.corrcoef(signal[:N], y_lib[:N])[0, 1]
rms_o = np.sqrt(np.mean(signal**2))
rms_l = np.sqrt(np.mean(y_lib[:N]**2))
print(f"1. Librosa cqt→icqt round-trip: corr={cc_lib:.6f}  "
      f"rms_orig={rms_o:.4f}  rms_recon={rms_l:.4f}")

# --- 2. Our CQT ---
cfg = AnalysisConfig()
cfg.cqt_fmin = fmin
cfg.cqt_fmax = fmax
cfg.bins_per_octave = bpo
cfg.hop_length = hop
freqs, times, power, C_ours = compute_cqt(signal, sr, cfg)
print(f"\n2. Shapes: librosa={C_lib.shape}  ours={C_ours.shape}")

# --- 3. Per-octave ratio ---
n_octaves = int(math.ceil(n_bins / bpo))
freqs_lib = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bpo)

# Librosa wavelength at full sr for each bin
lengths_fullsr = np.ceil(Q * sr / freqs_lib).astype(int)

print(f"\n3. Per-octave magnitude ratio (librosa_scaled / ours):")
for octave in range(min(n_octaves, 5)):
    gbin_start = max(0, n_bins - (octave + 1) * bpo)
    gbin_end = n_bins - octave * bpo
    if gbin_end > min(C_lib.shape[0], C_ours.shape[0]):
        continue

    # Compare at the same time frames
    n_common = min(C_lib.shape[1], C_ours.shape[1])
    a_lib = np.abs(C_lib[gbin_start:gbin_end, :n_common])
    a_ours = np.abs(C_ours[gbin_start:gbin_end, :n_common])

    mask = (a_lib > a_lib.max() * 0.01) & (a_ours > a_ours.max() * 0.01)
    if not mask.any():
        print(f"  Oct {octave}: no signal")
        continue

    ratios = a_lib[mask] / a_ours[mask]
    klen_range = lengths_fullsr[gbin_start:gbin_end]
    sqrt_klen_med = np.sqrt(np.median(klen_range))
    print(f"  Oct {octave}: bins {gbin_start}-{gbin_end-1}  "
          f"ratio={np.median(ratios):.4f} ± {np.std(ratios):.4f}  "
          f"sqrt(klen_median)={sqrt_klen_med:.4f}  "
          f"klen=[{klen_range.min()}..{klen_range.max()}]")

# --- 4. Try converting our CQT → librosa format → icqt ---
# Hypothesis: C_lib_scaled = C_ours * sqrt(klen_fullsr)
print(f"\n4. Attempting conversion and round-trip...")
C_converted = C_ours[:n_bins, :C_lib.shape[1]].copy()
for k in range(n_bins):
    C_converted[k, :] *= np.sqrt(lengths_fullsr[k])

y_conv = librosa.icqt(C_converted, sr=sr, hop_length=hop, fmin=fmin,
                       bins_per_octave=bpo, length=N)
cc_conv = np.corrcoef(signal[:N], y_conv[:N])[0, 1]
rms_c = np.sqrt(np.mean(y_conv[:N]**2))
print(f"   Converted icqt:  corr={cc_conv:.6f}  rms={rms_c:.4f}")

# --- 5. Also try without conversion (raw) ---
C_raw = C_ours[:n_bins, :C_lib.shape[1]].copy()
y_raw = librosa.icqt(C_raw, sr=sr, hop_length=hop, fmin=fmin,
                      bins_per_octave=bpo, length=N, scale=False)
cc_raw = np.corrcoef(signal[:N], y_raw[:N])[0, 1]
rms_r = np.sqrt(np.mean(y_raw[:N]**2))
print(f"   Raw icqt(scale=False): corr={cc_raw:.6f}  rms={rms_r:.4f}")

# --- 6. Try other scaling: * klen ---
C_klen = C_ours[:n_bins, :C_lib.shape[1]].copy()
for k in range(n_bins):
    C_klen[k, :] *= lengths_fullsr[k]
y_klen = librosa.icqt(C_klen, sr=sr, hop_length=hop, fmin=fmin,
                       bins_per_octave=bpo, length=N, scale=False)
cc_klen = np.corrcoef(signal[:N], y_klen[:N])[0, 1]
rms_k = np.sqrt(np.mean(y_klen[:N]**2))
print(f"   *klen icqt(scale=False): corr={cc_klen:.6f}  rms={rms_k:.4f}")

print("\nDone.")
