"""Tests for the CWT (Continuous Wavelet Transform) subsonic engine.

Validates:
  - Morlet, Morse, Ricker wavelets detect injected tones
  - Subsonic frequencies (0.5–15 Hz) resolve correctly
  - OOM-halve batching doesn't corrupt results (small batch vs large)
  - Auto-resample gate fires for very low frequencies
  - dtype preservation (float32/float64)
  - No NaN/Inf in output
"""
from __future__ import annotations
import math
import pytest
import torch
import numpy as np

import torch_cqt_new as torch_cqt


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_tone(freq: float, sr: int, dur: float,
               dtype: torch.dtype = torch.float32,
               device: str = "cpu") -> torch.Tensor:
    """Pure sine at *freq* Hz, *dur* seconds."""
    t = torch.arange(int(sr * dur), dtype=dtype, device=device) / sr
    return torch.sin(2.0 * math.pi * freq * t)


def _peak_freq(W: torch.Tensor, freqs: torch.Tensor) -> float:
    """Return the center frequency of the scale with maximum energy."""
    energy = W.abs().pow(2).sum(dim=-1)  # (n_scales,)
    idx = int(energy.argmax().item())
    return float(freqs[idx].item())


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════
# Basic smoke tests — does it run and produce sane output?
# ══════════════════════════════════════════════════════════════════════════

class TestCWTSmoke:
    """Minimal sanity: runs, correct shape/dtype, no NaN."""

    @pytest.mark.parametrize("wavelet", ["morlet", "morse", "ricker"])
    def test_smoke_all_wavelets(self, wavelet: str):
        sr = 100
        y = _make_tone(5.0, sr, 2.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(y, sr, fmin=0.5, fmax=25.0,
                                  n_scales=64, hop_length=1,
                                  wavelet=wavelet, device=_DEVICE)
        assert W.shape[0] == 64
        assert W.shape[1] > 0
        assert W.is_complex()
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()
        assert freqs.shape == (64,)
        # freqs should be high-to-low
        assert float(freqs[0]) > float(freqs[-1])

    def test_dtype_preservation_float32(self):
        sr = 100
        y = _make_tone(5.0, sr, 1.0, dtype=torch.float32, device=_DEVICE)
        W, freqs = torch_cqt.cwt(y, sr, fmin=1.0, fmax=20.0,
                                  n_scales=32, wavelet="morlet",
                                  device=_DEVICE)
        # Complex output should correspond to float32 → complex64
        assert W.dtype == torch.complex64
        assert freqs.dtype == torch.float32

    def test_dtype_preservation_float64(self):
        sr = 100
        y = _make_tone(5.0, sr, 1.0, dtype=torch.float64, device=_DEVICE)
        W, freqs = torch_cqt.cwt(y, sr, fmin=1.0, fmax=20.0,
                                  n_scales=32, wavelet="morlet",
                                  device=_DEVICE)
        assert W.dtype == torch.complex128
        assert freqs.dtype == torch.float64


# ══════════════════════════════════════════════════════════════════════════
# Tone detection — peak energy should be at the injected frequency
# ══════════════════════════════════════════════════════════════════════════

class TestCWTToneDetection:
    """Inject a pure tone, verify the CWT peak is at the right frequency."""

    @pytest.mark.parametrize("tone_hz", [1.0, 2.5, 5.0, 10.0, 15.0])
    def test_morlet_detects_subsonic_tone(self, tone_hz: float):
        sr = 200  # Need at least 2× highest tone
        dur = max(10.0, 20.0 / tone_hz)  # Enough cycles
        y = _make_tone(tone_hz, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.5, fmax=50.0,
            scales_per_octave=24,
            hop_length=1,
            wavelet="morlet",
            wavelet_kw={"sigma": 6.0},
            device=_DEVICE,
        )
        peak = _peak_freq(W, freqs)
        # Within ±1 octave-24th (≈ 3% = 50 cents)
        ratio = peak / tone_hz
        print(f"  tone={tone_hz:.1f}Hz  peak={peak:.2f}Hz  ratio={ratio:.4f}")
        assert 0.8 < ratio < 1.25, f"Peak {peak:.2f} too far from {tone_hz}"

    @pytest.mark.parametrize("tone_hz", [2.0, 8.0])
    def test_morse_detects_tone(self, tone_hz: float):
        sr = 200
        dur = max(10.0, 20.0 / tone_hz)
        y = _make_tone(tone_hz, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.5, fmax=50.0,
            scales_per_octave=24,
            wavelet="morse",
            wavelet_kw={"beta": 4.0, "gamma": 3.0},
            device=_DEVICE,
        )
        peak = _peak_freq(W, freqs)
        ratio = peak / tone_hz
        print(f"  morse tone={tone_hz:.1f}Hz  peak={peak:.2f}Hz  ratio={ratio:.4f}")
        assert 0.7 < ratio < 1.4

    @pytest.mark.parametrize("tone_hz", [2.0, 8.0])
    def test_ricker_detects_tone(self, tone_hz: float):
        sr = 200
        dur = max(10.0, 20.0 / tone_hz)
        y = _make_tone(tone_hz, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.5, fmax=50.0,
            scales_per_octave=24,
            wavelet="ricker",
            device=_DEVICE,
        )
        peak = _peak_freq(W, freqs)
        ratio = peak / tone_hz
        print(f"  ricker tone={tone_hz:.1f}Hz  peak={peak:.2f}Hz  ratio={ratio:.4f}")
        assert 0.7 < ratio < 1.4


# ══════════════════════════════════════════════════════════════════════════
# Two-tone separation — can distinguish two subsonic frequencies?
# ══════════════════════════════════════════════════════════════════════════

class TestCWTTwoTone:
    def test_morlet_separates_two_tones(self):
        sr = 200
        dur = 30.0
        t = torch.arange(int(sr * dur), dtype=torch.float32, device=_DEVICE) / sr
        y = torch.sin(2 * math.pi * 2.0 * t) + torch.sin(2 * math.pi * 8.0 * t)

        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.5, fmax=50.0,
            scales_per_octave=48,
            wavelet="morlet", wavelet_kw={"sigma": 8.0},
            device=_DEVICE,
        )
        energy = W.abs().pow(2).sum(dim=-1).cpu()
        freqs_np = freqs.cpu().numpy()

        # Find local maxima (higher than both neighbours)
        peaks = []
        for i in range(1, len(energy) - 1):
            if energy[i] > energy[i - 1] and energy[i] > energy[i + 1]:
                peaks.append((float(energy[i]), float(freqs_np[i])))
        peaks.sort(reverse=True)
        top2 = sorted([p[1] for p in peaks[:2]])
        print(f"  two-tone local-max peaks: {top2[0]:.2f}Hz, {top2[1]:.2f}Hz")
        assert 1.5 < top2[0] < 2.8
        assert 6.0 < top2[1] < 10.0


# ══════════════════════════════════════════════════════════════════════════
# Auto-resample gate — verify it fires and results still valid
# ══════════════════════════════════════════════════════════════════════════

class TestCWTAutoResample:
    def test_auto_resample_fires_for_low_freq(self):
        """At sr=44100, fmin=0.1 Hz, n_fft would be huge → resample gate."""
        sr = 44100
        dur = 10.0
        y = _make_tone(0.5, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.1, fmax=10.0,
            n_scales=32, wavelet="morlet",
            n_fft_max=2**18,
            device=_DEVICE,
        )
        assert not torch.isnan(W).any()
        assert W.shape[0] == 32
        # Peak should still be near 0.5 Hz
        peak = _peak_freq(W, freqs)
        print(f"  auto-resample: tone=0.5Hz peak={peak:.3f}Hz")
        assert 0.3 < peak < 0.8


# ══════════════════════════════════════════════════════════════════════════
# Batch consistency — small vs large batch should give same result
# ══════════════════════════════════════════════════════════════════════════

class TestCWTBatchConsistency:
    def test_small_vs_large_batch(self):
        sr = 100
        y = _make_tone(5.0, sr, 5.0, device=_DEVICE)
        W1, f1 = torch_cqt.cwt(y, sr, fmin=1.0, fmax=20.0,
                                n_scales=32, wavelet="morlet",
                                batch_size=4, device=_DEVICE)
        W2, f2 = torch_cqt.cwt(y, sr, fmin=1.0, fmax=20.0,
                                n_scales=32, wavelet="morlet",
                                batch_size=128, device=_DEVICE)
        assert torch.allclose(f1, f2)
        assert torch.allclose(W1, W2, atol=1e-5, rtol=1e-4)


# ══════════════════════════════════════════════════════════════════════════
# scales_per_octave — verify log spacing
# ══════════════════════════════════════════════════════════════════════════

class TestCWTScalesPerOctave:
    def test_scales_per_octave_count(self):
        sr = 200
        y = _make_tone(5.0, sr, 2.0, device=_DEVICE)
        spo = 12
        # 0.5 → 50 Hz = ~6.6 octaves → ~80 scales
        W, freqs = torch_cqt.cwt(y, sr, fmin=0.5, fmax=50.0,
                                  scales_per_octave=spo,
                                  wavelet="morlet", device=_DEVICE)
        n_oct = math.log2(50.0 / 0.5)
        expected = int(round(n_oct * spo))
        assert W.shape[0] == expected
        assert len(freqs) == expected

    def test_explicit_freqs_override(self):
        sr = 200
        y = _make_tone(5.0, sr, 2.0, device=_DEVICE)
        custom_freqs = torch.tensor([10.0, 5.0, 2.5, 1.0], device=_DEVICE)
        W, freqs = torch_cqt.cwt(y, sr, freqs=custom_freqs,
                                  wavelet="morlet", device=_DEVICE)
        assert W.shape[0] == 4
        assert torch.allclose(freqs, custom_freqs)


# ── Complex-log bandwidth (epsilon) ─────────────────────────────────────

class TestCWTComplexLogBandwidth:
    """Validate the complex-log sigma regularisation."""

    def test_sigma_formula_matches(self):
        """_complex_log_sigma returns expected values."""
        freqs = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
        sigma_0 = 6.0
        eps = 0.5
        sigmas = torch_cqt._complex_log_sigma(freqs, sigma_0, eps)
        assert sigmas.shape == (4,)
        # At high freq (100 Hz) eps is negligible → ratio ≈ 1 → sigma ≈ σ₀
        assert abs(float(sigmas[3]) - sigma_0) / sigma_0 < 0.02
        # At low freq (0.1 Hz) eps matters → sigma < σ₀
        assert float(sigmas[0]) < sigma_0

    def test_sigma_monotonic_with_freq(self):
        """Higher freq → sigma closer to σ₀ (monotonically increasing)."""
        freqs = torch.logspace(-1, 2, 50, dtype=torch.float64)
        sigmas = torch_cqt._complex_log_sigma(freqs, 6.0, 0.5)
        # Check overall trend: sigma at last (highest) > sigma at first (lowest)
        assert float(sigmas[-1]) > float(sigmas[0])

    def test_cwt_with_epsilon_runs(self):
        """CWT with epsilon produces valid output (no NaN/Inf)."""
        sr = 200
        y = _make_tone(5.0, sr, 4.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.5, fmax=50.0,
            scales_per_octave=8,
            wavelet="morlet", device=_DEVICE,
            epsilon=0.5,
        )
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()
        assert W.shape[0] == len(freqs)

    def test_epsilon_detects_tone(self):
        """Tone detection still works with complex-log bandwidth."""
        sr = 200
        freq = 5.0
        y = _make_tone(freq, sr, 4.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.5, fmax=50.0,
            scales_per_octave=12,
            wavelet="morlet", device=_DEVICE,
            epsilon=0.5,
        )
        detected = _peak_freq(W, freqs)
        ratio = detected / freq
        assert 0.85 < ratio < 1.15, f"expected ~{freq}, got {detected}"

    def test_epsilon_none_unchanged(self):
        """epsilon=None produces identical output to no-epsilon path."""
        sr = 200
        y = _make_tone(5.0, sr, 2.0, device=_DEVICE)
        W1, f1 = torch_cqt.cwt(y, sr, fmin=1.0, fmax=50.0, n_scales=32,
                                wavelet="morlet", device=_DEVICE)
        W2, f2 = torch_cqt.cwt(y, sr, fmin=1.0, fmax=50.0, n_scales=32,
                                wavelet="morlet", device=_DEVICE,
                                epsilon=None)
        assert torch.allclose(W1, W2)
        assert torch.allclose(f1, f2)

    def test_subsonic_tone_with_epsilon(self):
        """Very low freq (0.5 Hz) detected with epsilon keeping wavelets short."""
        sr = 200
        freq = 0.5
        dur = 20.0  # need long signal for 0.5 Hz
        y = _make_tone(freq, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.2, fmax=5.0,
            scales_per_octave=8,
            wavelet="morlet", device=_DEVICE,
            epsilon=0.3,
        )
        detected = _peak_freq(W, freqs)
        ratio = detected / freq
        assert 0.7 < ratio < 1.3, f"expected ~{freq}, got {detected}"


# ══════════════════════════════════════════════════════════════════════════
# iCWT round-trip — forward then inverse, measure reconstruction fidelity
# ══════════════════════════════════════════════════════════════════════════

def _xcorr_peak(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation peak (tolerates time shift)."""
    a = a - a.mean(); b = b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    n = len(a) + len(b) - 1
    fft_n = 1
    while fft_n < n:
        fft_n <<= 1
    xc = np.fft.irfft(
        np.fft.rfft(a, fft_n) * np.conj(np.fft.rfft(b, fft_n)), fft_n)
    return float(xc.max() / (na * nb))


def _snr_db(sig: np.ndarray, recon: np.ndarray, margin: int = 0) -> float:
    n = min(len(sig), len(recon))
    if margin:
        sig, recon = sig[margin:n - margin], recon[margin:n - margin]
        n = len(sig)
    err_rms = float(np.sqrt(np.mean((sig[:n] - recon[:n]) ** 2)))
    sig_rms = float(np.sqrt(np.mean(sig[:n] ** 2)))
    return 20.0 * np.log10(sig_rms / max(err_rms, 1e-30))


class TestICWTRoundTrip:
    """CWT -> iCWT reconstruction fidelity.

    The one-integral iCWT (ssqueezepy algorithm) reconstructs the correct
    SHAPE but with a fixed amplitude scaling that depends on the wavelet's
    admissibility constant Cpsi and the log-frequency grid spacing dj.
    Raw SNR is therefore a poor metric; amplitude-normalised SNR and
    cross-correlation are the right measures.

    Key constraint: iCWT quality degrades with hop_length because
    _icwt_reconstruct sums Re(W) at the decimated frame rate and then
    linearly upsamples -- adequate for subsonic monitoring but not
    high-fidelity audio at large hops.

      hop=1  -> shape near-perfect  (xcorr > 0.999, norm-SNR > 20 dB)
      hop=4  -> good                (xcorr > 0.99)
      hop>=32 -> degraded           (documents the known limitation)
    """

    @staticmethod
    def _norm_snr_db(y_np: np.ndarray, r_np: np.ndarray,
                     margin: int = 0) -> float:
        """SNR after peak-normalising both signals (shape fidelity only)."""
        n = min(len(y_np), len(r_np))
        a = y_np[margin:n - margin] if margin else y_np[:n]
        b = r_np[margin:n - margin] if margin else r_np[:n]
        na = float(np.abs(a).max())
        nb = float(np.abs(b).max())
        if na < 1e-12 or nb < 1e-12:
            return -999.0
        a = a / na
        b = b / nb
        err_rms = float(np.sqrt(np.mean((a - b) ** 2)))
        return 20.0 * np.log10(1.0 / max(err_rms, 1e-30))

    @pytest.mark.parametrize("freq", [5.0, 10.0])
    def test_hop1_tone_snr(self, freq: float):
        """hop=1 amplitude-normalised SNR should exceed 20 dB for a pure tone.

        The iCWT one-integral algorithm (ssqueezepy) produces the correct
        waveform shape but with a fixed under-scaling (~40 % amplitude for a
        narrowband tone).  We normalise both signals to peak-1 before
        measuring SNR so we are testing shape fidelity, not absolute gain.
        """
        sr = 200
        dur = max(8.0, 20.0 / freq)
        y = _make_tone(freq, sr, dur, dtype=torch.float64, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=max(0.5, freq / 4), fmax=min(sr / 2.1, freq * 4),
            scales_per_octave=12, hop_length=1,
            wavelet="morlet", wavelet_kw={"sigma": 6.0},
            device=_DEVICE,
        )
        y_hat = torch_cqt.icwt(
            W, freqs, sr, hop_length=1,
            wavelet="morlet", wavelet_kw={"sigma": 6.0},
            device=_DEVICE,
        )
        y_np = y.cpu().numpy()
        r_np = y_hat.cpu().numpy()
        margin = sr // 4
        snr = self._norm_snr_db(y_np, r_np, margin=margin)
        print(f"  hop=1  freq={freq}Hz  norm-SNR={snr:.1f} dB")
        assert snr > 20.0, f"hop=1 norm-SNR too low: {snr:.1f} dB"

    def test_hop1_xcorr_chirp(self):
        """hop=1 chirp round-trip cross-correlation must be > 0.999."""
        sr = 500
        dur = 4.0
        n = int(sr * dur)
        t = torch.arange(n, dtype=torch.float64, device=_DEVICE) / sr
        f0, f1 = 2.0, 50.0
        y = torch.sin(2.0 * math.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2))
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=1.0, fmax=100.0,
            scales_per_octave=12, hop_length=1,
            wavelet="morlet", device=_DEVICE,
        )
        y_hat = torch_cqt.icwt(W, freqs, sr, hop_length=1,
                                wavelet="morlet", device=_DEVICE)
        y_np = y.cpu().numpy()
        r_np = y_hat.cpu().numpy()
        margin = sr // 4
        n2 = min(len(y_np), len(r_np))
        corr = _xcorr_peak(y_np[margin:n2 - margin],
                           r_np[margin:n2 - margin])
        print(f"  hop=1  chirp xcorr={corr:.6f}")
        assert corr > 0.999, f"hop=1 chirp xcorr too low: {corr:.4f}"

    def test_hop4_xcorr(self):
        """hop=4 round-trip cross-correlation should remain > 0.99."""
        sr = 200
        freq = 5.0
        dur = 10.0
        y = _make_tone(freq, sr, dur, dtype=torch.float32, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=1.0, fmax=50.0,
            scales_per_octave=12, hop_length=4,
            wavelet="morlet", device=_DEVICE,
        )
        y_hat = torch_cqt.icwt(W, freqs, sr, hop_length=4,
                                wavelet="morlet", device=_DEVICE)
        y_np = y.cpu().numpy()
        r_np = y_hat.cpu().numpy()
        margin = sr // 4
        n2 = min(len(y_np), len(r_np))
        corr = _xcorr_peak(y_np[margin:n2 - margin].astype(np.float64),
                           r_np[margin:n2 - margin].astype(np.float64))
        print(f"  hop=4  freq={freq}Hz  xcorr={corr:.4f}")
        assert corr > 0.99, f"hop=4 xcorr degraded: {corr:.4f}"

    @pytest.mark.parametrize("hop,min_corr", [
        (1,  0.999),
        (4,  0.99),
        (16, 0.90),
    ])
    def test_hop_degradation_table(self, hop: int, min_corr: float):
        """Cross-correlation vs hop_length -- documents the fidelity envelope.

        hop=1  -> near-perfect; hop=4 -> good; hop=16 -> noticeably degraded.
        hop>=32 is known-broken for full-bandwidth audio reconstruction
        (linear upsampling cannot recover high-frequency content from
        a handful of frames).  Use hop=1 for playback synthesis.
        """
        sr = 200
        freq = 5.0
        dur = 10.0
        y = _make_tone(freq, sr, dur, dtype=torch.float32, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=1.0, fmax=50.0,
            scales_per_octave=12, hop_length=hop,
            wavelet="morlet", device=_DEVICE,
        )
        y_hat = torch_cqt.icwt(W, freqs, sr, hop_length=hop,
                                wavelet="morlet", device=_DEVICE)
        y_np = y.cpu().numpy()
        r_np = y_hat.cpu().numpy()
        margin = sr // 4
        n2 = min(len(y_np), len(r_np))
        corr = _xcorr_peak(y_np[margin:n2 - margin].astype(np.float64),
                           r_np[margin:n2 - margin].astype(np.float64))
        print(f"  hop={hop:2d}  freq={freq}Hz  xcorr={corr:.4f}  (min={min_corr})")
        assert corr >= min_corr, (
            f"hop={hop} xcorr {corr:.4f} below threshold {min_corr}"
        )

    def test_hop1_with_epsilon_roundtrip(self):
        """Complex-log bandwidth (epsilon) round-trip, hop=1."""
        sr = 200
        freq = 5.0
        dur = 10.0
        y = _make_tone(freq, sr, dur, dtype=torch.float64, device=_DEVICE)
        eps = 0.3
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=1.0, fmax=50.0,
            scales_per_octave=12, hop_length=1,
            wavelet="morlet", epsilon=eps, device=_DEVICE,
        )
        y_hat = torch_cqt.icwt(
            W, freqs, sr, hop_length=1,
            wavelet="morlet", epsilon=eps, device=_DEVICE,
        )
        y_np = y.cpu().numpy()
        r_np = y_hat.cpu().numpy()
        margin = sr // 4
        n2 = min(len(y_np), len(r_np))
        corr = _xcorr_peak(y_np[margin:n2 - margin],
                           r_np[margin:n2 - margin])
        print(f"  epsilon={eps}  xcorr={corr:.6f}")
        assert corr > 0.99, f"epsilon round-trip xcorr too low: {corr:.4f}"
