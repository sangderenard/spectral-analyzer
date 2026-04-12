"""Focused iCQT diagnostic: use actual stored CQT data, test single bin."""
import numpy as np, math, os, glob

analysis_dirs = glob.glob("input/*_analysis")
adir = analysis_dirs[0]
npz_files = sorted(glob.glob(os.path.join(adir, "cqt_data_*.npz")),
                   key=os.path.getmtime, reverse=True)
npz_path = npz_files[0] if npz_files else os.path.join(adir, "cqt_data.npz")
npz = np.load(npz_path, mmap_mode="r")

real_l = np.asarray(npz["real_left"], dtype=np.float64)
imag_l = np.asarray(npz["imag_left"], dtype=np.float64)
freqs = np.asarray(npz["freqs"], dtype=np.float64)
sr = int(npz["sr"]); hop = int(npz["hop_length"]); bpo = int(npz["bins_per_octave"])
n_bins, n_frames = real_l.shape
Q = 1.0 / (2.0 ** (1.0 / bpo) - 1.0)
n_octaves = int(math.ceil(n_bins / bpo))
cqt = real_l + 1j * imag_l
print(f"Loaded: {n_bins} bins x {n_frames} frames, sr={sr}, hop={hop}, bpo={bpo}, Q={Q:.1f}")

# Pick a SINGLE high-frequency bin where octave 0 applies (no decimation)
# and test iCQT of just that one bin over ALL frames
bin_idx = n_bins - 1  # highest bin
f = freqs[bin_idx]
oct_sr = sr  # octave 0
klen = int(math.ceil(Q * oct_sr / f))
print(f"\nSingle bin test: bin={bin_idx}, f={f:.1f}Hz, klen={klen}, klen/hop={klen/hop:.1f}")
print(f"n_frames={n_frames}, adequate overlap: {n_frames > 2 * klen / hop}")

coeffs = cqt[bin_idx, :]
print(f"Coefficient |max|={np.abs(coeffs).max():.6e}, |mean|={np.abs(coeffs).mean():.6e}")

# iCQT for this one bin
train_len = (n_frames - 1) * hop + 1
t_k = np.arange(klen, dtype=np.float64)
win = 0.5 - 0.5 * np.cos(2.0 * np.pi * t_k / klen)
basis = win * np.exp(2j * np.pi * f * t_k / oct_sr)

train = np.zeros(train_len, dtype=np.complex128)
train[::hop] = coeffs

conv_len = train_len + klen - 1
fft_n = 1
while fft_n < conv_len:
    fft_n <<= 1
conv = np.fft.ifft(
    np.fft.fft(train, fft_n) * np.fft.fft(basis, fft_n)
).real[:conv_len]

# OLA norm (single bin)
win_sq = win ** 2
norm_train = np.zeros(train_len, dtype=np.float64)
norm_train[::hop] = 1.0
nc = np.fft.irfft(np.fft.rfft(norm_train, fft_n) *
                  np.fft.rfft(win_sq, fft_n), fft_n)[:conv_len]

print(f"\nOLA norm: min={nc.min():.4e}, max={nc.max():.4e}, median(>0)={np.median(nc[nc>0.01]):.4e}")

# Count samples where norm is < 50% of median (boundary region)
median_norm = np.median(nc[nc > 0.01])
boundary_count = np.sum(nc < median_norm * 0.5)
print(f"Boundary samples (norm < 50% median): {boundary_count}/{conv_len} "
      f"({100*boundary_count/conv_len:.1f}%)")

safe_norm = np.where(nc > 1e-10, nc, 1.0)
recon = conv / safe_norm

# Show per-chunk distribution
n_chunks = 20
chunk_sz = conv_len // n_chunks
print(f"\nPer-chunk analysis (conv_len={conv_len}):")
for i in range(n_chunks):
    s = recon[i * chunk_sz:(i + 1) * chunk_sz]
    n_region = nc[i * chunk_sz:(i + 1) * chunk_sz]
    print(f"  chunk {i:2d}: recon peak={np.abs(s).max():.4e} "
          f"norm_min={n_region.min():.2e} norm_max={n_region.max():.2e}")

# Now test with norm floor (the proposed fix)
norm_floor = max(median_norm * 0.1, 1e-10)
safe_norm_floored = np.where(nc > norm_floor, nc, norm_floor)
recon_floored = conv / safe_norm_floored

print(f"\nWith norm floor ({norm_floor:.2e}):")
for i in range(n_chunks):
    s = recon_floored[i * chunk_sz:(i + 1) * chunk_sz]
    print(f"  chunk {i:2d}: recon peak={np.abs(s).max():.4e}")

# Peak normalization comparison
for label, sig in [("raw_ola", recon), ("floored_ola", recon_floored)]:
    peak = max(np.abs(sig).max(), 1e-12)
    normed = sig / peak * 0.8
    chunk_peaks_norm = [np.abs(normed[i*chunk_sz:(i+1)*chunk_sz]).max() for i in range(n_chunks)]
    print(f"\n{label} after peak norm (peak={peak:.4e}):")
    print(f"  chunk peaks: {['%.4f' % p for p in chunk_peaks_norm]}")
