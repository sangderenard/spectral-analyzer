"""Pinpoint which filter and sample differs in the wavelet."""
import torch, numpy as np, librosa, math, scipy.signal
import torch_cqt_new as tc

SR = 22050; BPO = 48; FMIN = 27.5
freqs = librosa.cqt_frequencies(384, fmin=FMIN, bins_per_octave=BPO)
alpha = librosa.filters._relative_bandwidth(freqs=freqs)

sl = slice(-48, None)  # top octave
f_oct = freqs[sl]
a_oct = alpha[sl]

# Librosa
lib_b, lib_l = librosa.filters.wavelet(freqs=f_oct, sr=SR, filter_scale=1.0,
    pad_fft=True, norm=1, window='hann', alpha=a_oct)
lib_np = np.asarray(lib_b.todense()) if hasattr(lib_b, 'todense') else np.asarray(lib_b)

# Ours
f_t = torch.from_numpy(f_oct)
a_t = torch.from_numpy(a_oct)
our_b, our_l = tc._wavelet(f_t, SR, filter_scale=1.0, alpha=a_t)
our_np = our_b.numpy()

# Find which filter has the max diff
per_filter_diff = np.abs(lib_np.astype(np.complex128) - our_np).max(axis=1)
worst = np.argmax(per_filter_diff)
print(f"Worst filter: {worst}, diff={per_filter_diff[worst]:.6e}")
print(f"Diffs > 1e-10:")
for i in range(48):
    if per_filter_diff[i] > 1e-10:
        print(f"  filter {i}: diff={per_filter_diff[i]:.6e}")

# Zoom into worst filter
i = worst
ilen = our_l[i].item()
freq = f_oct[i]
Q = 1.0 / a_oct[i]
print(f"\nFilter {i}: freq={freq:.4f}Hz  ilen={ilen:.6f}  Q={Q:.4f}")
print(f"  floor={math.floor(ilen)}  ceil={math.ceil(ilen)}")

# Check individual samples
lib_filt = lib_np[i]
our_filt = our_np[i]
diffs = np.abs(lib_filt.astype(np.complex128) - our_filt)
nonzero = np.where(diffs > 1e-10)[0]
print(f"  Samples with diff > 1e-10: {len(nonzero)}")
if len(nonzero) > 0:
    print(f"  Sample indices: {nonzero[:10]}...")
    for idx in nonzero[:5]:
        print(f"    [{idx}] lib={lib_filt[idx]:.10e}  ours={our_filt[idx]:.10e}  diff={diffs[idx]:.6e}")

# Now reproduce what librosa does step by step and what we do
print("\n=== Step-by-step for filter", i, "===")
t_np = np.arange(-ilen // 2, ilen // 2)
t_tc = torch.arange(-ilen // 2, ilen // 2, dtype=torch.float64)
print(f"  arange len: np={len(t_np)} torch={len(t_tc)}")

# Phasor
sig_np = np.exp(1j * 2 * np.pi * freq / SR * t_np)
sig_tc = torch.exp(1j * 2.0 * math.pi * freq / SR * t_tc)
print(f"  phasor diff: {np.abs(sig_np - sig_tc.numpy()).max():.2e}")

# Window
n_min = int(math.floor(ilen))
n_max = int(math.ceil(ilen))
lib_win = scipy.signal.get_window('hann', n_min)
# __float_window does: window[n_min:] = 0, then pad to n_max
lib_win_full = np.zeros(n_max)
lib_win_full[:n_min] = lib_win
lib_win_cut = lib_win_full[:len(t_np)]

our_win = tc._hann_window(n_min, torch.device('cpu')).numpy()
our_win_full = np.zeros(n_max)
our_win_full[:n_min] = our_win
our_win_cut = our_win_full[:len(t_tc)]

print(f"  window diff: {np.abs(lib_win_cut - our_win_cut).max():.2e}")

# Apply window
sig_np_w = sig_np * lib_win_cut
sig_tc_w = sig_tc.numpy() * our_win_cut
print(f"  windowed sig diff: {np.abs(sig_np_w - sig_tc_w).max():.2e}")

# L1 normalize
sig_np_n = sig_np_w / np.abs(sig_np_w).sum()
sig_tc_n = sig_tc_w / np.abs(sig_tc_w).sum()
print(f"  L1 norm diff: {np.abs(sig_np_n - sig_tc_n).max():.2e}")

# Pad center
max_len_np = int(max(lib_l))
max_len_tc = int(our_l.max().item())
print(f"  max_len: lib={max_len_np} ours={max_len_tc}")
n_fft_np = int(2**np.ceil(np.log2(max_len_np)))
n_fft_tc = int(2**math.ceil(math.log2(max_len_tc)))
print(f"  n_fft: lib={n_fft_np} ours={n_fft_tc}")

# Pad center each
flen = len(sig_np_n)
lpad = (n_fft_np - flen) // 2
lib_padded = np.zeros(n_fft_np, dtype=np.complex128)
lib_padded[lpad:lpad+flen] = sig_np_n
our_padded = np.zeros(n_fft_tc, dtype=np.complex128)
our_padded[lpad:lpad+flen] = sig_tc_n
print(f"  padded diff: {np.abs(lib_padded - our_padded).max():.2e}")

# Check lib actual padded vs what we computed
print(f"  lib actual vs our reproduction diff: {np.abs(lib_filt.astype(np.complex128) - lib_padded).max():.2e}")
print(f"  ours actual vs our reproduction diff: {np.abs(our_filt - our_padded).max():.2e}")
