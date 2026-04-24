"""Diagnostic: understand why iCWT fails."""
import torch, math, sys
sys.path.insert(0, '.')
import torch_cqt_new as T

sr = 200; freq = 5.0; dur = 4.0
n = int(sr*dur)
t = torch.arange(n, dtype=torch.float64) / sr
y = torch.sin(2*math.pi*freq*t)

W, freqs = T.cwt(y, sr, fmin=1.0, fmax=50.0, scales_per_octave=12,
                 wavelet='morlet', dtype=torch.float64)
print(f'W shape: {W.shape}, n_frames={W.shape[1]}, n_samples={n}')

# Find the scale closest to 5 Hz
diffs = (freqs - freq).abs()
s_peak = diffs.argmin().item()
print(f'Peak scale {s_peak}, freq={freqs[s_peak]:.3f} Hz')

# Look at W at peak scale
w_peak = W[s_peak]
print(f'|W_peak| stats: mean={w_peak.abs().mean():.6f}, max={w_peak.abs().max():.6f}')

# Compare Re{W_peak} with y (trimmed)
L = min(len(y), W.shape[1])
y_trim = y[:L]
w_real = w_peak[:L].real
yn = y_trim / y_trim.norm()
wn = w_real / w_real.norm().clamp(min=1e-30)
corr_peak = torch.dot(yn, wn).item()
print(f'corr(y, Re{{W_peak}}): {corr_peak:.6f}')

# Check if there's a time offset
# Cross-correlate y with Re{W_peak}
from torch.nn.functional import conv1d
a = yn.unsqueeze(0).unsqueeze(0)  # (1,1,L)
b = wn.flip(0).unsqueeze(0).unsqueeze(0)  # (1,1,L)
xcorr = conv1d(a, b, padding=L-1).squeeze()
lag = xcorr.abs().argmax().item() - (L - 1)
print(f'Best lag = {lag} samples, peak xcorr = {xcorr.abs().max():.6f}')

# What do ALL scales contribute?
print('\nPer-scale contribution:')
for s in range(0, W.shape[0], max(1, W.shape[0]//10)):
    w_s = W[s, :L].real
    if w_s.norm() < 1e-30:
        continue
    ws_n = w_s / w_s.norm()
    c = torch.dot(yn, ws_n).item()
    print(f'  scale {s:3d} f={freqs[s]:8.3f} Hz: corr={c:+.4f} |W|_mean={W[s,:L].abs().mean():.6f}')

# Now try: just use the peak scale for reconstruction
y_hat = w_real * (y_trim.norm() / w_real.norm())
err = (y_trim - y_hat).norm() / y_trim.norm()
print(f'\nSingle-scale recon error: {err:.6f}')

# Sum over ALL scales with equal weight
total = W[:, :L].real.sum(dim=0)
tn = total / total.norm().clamp(min=1e-30)
corr_all = torch.dot(yn, tn).abs().item()
print(f'Sum all Re{{W}} corr: {corr_all:.6f}')

# Check: what if the issue is that non-peak scales ADD NOISE?
# Use only scales near the peak
for width in [1, 3, 5, 10, W.shape[0]]:
    lo = max(0, s_peak - width)
    hi = min(W.shape[0], s_peak + width + 1)
    subset = W[lo:hi, :L].real.sum(dim=0)
    sn = subset / subset.norm().clamp(min=1e-30)
    c = torch.dot(yn, sn).abs().item()
    print(f'  scales [{lo}:{hi}] ({hi-lo:2d} scales): corr={c:.6f}')
