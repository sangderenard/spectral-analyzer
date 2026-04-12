"""Compare _wavelet output to librosa.filters.wavelet."""
import torch
import numpy as np
import librosa
import torch_cqt_new as tc

SR = 22050
BPO = 48
FMIN = 27.5
N_BINS = 384

freqs_lib = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BPO)
alpha_lib = librosa.filters._relative_bandwidth(freqs=freqs_lib)

# Compare each octave slice like the forward CQT does
my_sr = float(SR)
n_filters = 48

for oct_i in range(8):
    if oct_i == 0:
        sl = slice(-n_filters, None)
    else:
        sl = slice(-n_filters * (oct_i + 1), -n_filters * oct_i)

    f_oct = freqs_lib[sl]
    a_oct = alpha_lib[sl]

    # Librosa
    lib_basis, lib_len = librosa.filters.wavelet(
        freqs=f_oct, sr=my_sr,
        filter_scale=1.0, pad_fft=True, norm=1,
        window="hann", alpha=a_oct,
    )
    lib_np = np.array(lib_basis.todense()) if hasattr(lib_basis, 'todense') else np.asarray(lib_basis)

    # Ours
    f_t = torch.from_numpy(f_oct)
    a_t = torch.from_numpy(a_oct)
    our_basis, our_len = tc._wavelet(f_t, my_sr, filter_scale=1.0, alpha=a_t)
    our_np = our_basis.numpy()

    # Compare
    if lib_np.shape != our_np.shape:
        print(f"oct {oct_i} SHAPE MISMATCH: lib {lib_np.shape} vs ours {our_np.shape}")
    else:
        # Cast our to same dtype for fair comparison
        diff = np.abs(lib_np.astype(np.complex128) - our_np).max()
        # Also check if diff goes away when we truncate to float32
        diff32 = np.abs(lib_np - our_np.astype(np.complex64)).max()
        mag_lib = np.abs(lib_np).max()
        ratio = diff / mag_lib if mag_lib > 0 else 0
        print(
            f"oct {oct_i}  my_sr={my_sr:8.0f}  shape={lib_np.shape}  "
            f"max_diff={diff:.2e}  diff32={diff32:.2e}  "
            f"rel_diff={ratio:.2e}  lib_max={mag_lib:.6f}"
        )

    my_sr /= 2.0
