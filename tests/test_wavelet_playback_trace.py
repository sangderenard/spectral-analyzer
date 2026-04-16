
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import json
from bass_viewer import _reconstruct_wavelet_viewport

# Simulate CWT data for a 1-second 48kHz signal
sr = 48000
n_samples = sr * 1  # 1 second
n_scales = 8
hop_length = 1

# Fake CWT data: shape (n_scales, n_frames)
W_complex = np.random.randn(n_scales, n_samples) + 1j * np.random.randn(n_scales, n_samples)
freqs = np.linspace(20, 20000, n_scales)

wv_meta = {
    "type": "cwt",
    "wavelet": "morlet",
    "sigma": 6.0,
    "epsilon": 0.01,
    "sr": sr,
    "n_samples": n_samples,
    "hop_length": hop_length,
    "n_scales": n_scales,
    "fmin": 20,
    "fmax": 20000,
}

def trace_playback(playback_rate):
    t0 = 0
    t1 = n_samples / sr
    print(f"t0={t0}, t1={t1}")
    sig = _reconstruct_wavelet_viewport(
        sr=sr,
        t0=t0,
        t1=t1,
        wv_meta=wv_meta,
        W_complex=W_complex,
        wv_freqs=freqs,
        playback_rate=playback_rate,
    )
    print(f"playback_rate={playback_rate}, output_len={len(sig)}")
    return sig

if __name__ == "__main__":
    for rate in [1.0, 32.0, 1/32.0]:
        trace_playback(rate)
