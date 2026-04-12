"""Professional infrasound / subsonic test suite.

Validates CWT analysis and audification across every major domain:
  - CTBTO/IMS atmospheric infrasound (0.02–4 Hz, sr=20)
  - Microbaroms / ocean-generated (0.1–0.5 Hz, sr=20)
  - Volcanic infrasound (0.01–10 Hz, sr=100)
  - Wind turbine monitoring (0.5–10 Hz, sr=200)
  - Earthquake precursors / tilt (0.001–0.1 Hz, sr=10)
  - Hydrophone / SOFAR channel (1–100 Hz, sr=1000)
  - Astronomical (gravitational wave proxy, millihertz, sr=4)

Each test uses *native* professional sample rates — no 44100 Hz.
"""
from __future__ import annotations
import math
import pytest
import torch
import numpy as np

import torch_cqt_new as torch_cqt


# ── Helpers ──────────────────────────────────────────────────────────────

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_tone(freq: float, sr: int, dur: float,
               dtype: torch.dtype = torch.float32,
               device: str = "cpu") -> torch.Tensor:
    t = torch.arange(int(sr * dur), dtype=dtype, device=device) / sr
    return torch.sin(2.0 * math.pi * freq * t)


def _make_multi_tone(freqs: list[float], sr: int, dur: float,
                     amps: list[float] | None = None,
                     dtype: torch.dtype = torch.float32,
                     device: str = "cpu") -> torch.Tensor:
    if amps is None:
        amps = [1.0] * len(freqs)
    t = torch.arange(int(sr * dur), dtype=dtype, device=device) / sr
    y = torch.zeros_like(t)
    for f, a in zip(freqs, amps):
        y += a * torch.sin(2.0 * math.pi * f * t)
    return y


def _peak_freq(W: torch.Tensor, freqs: torch.Tensor) -> float:
    energy = W.abs().pow(2).sum(dim=-1)
    idx = int(energy.argmax().item())
    return float(freqs[idx].item())


def _top_k_freqs(W: torch.Tensor, freqs: torch.Tensor, k: int) -> list[float]:
    """Return top-k peak frequencies using local maxima detection."""
    energy = W.abs().pow(2).sum(dim=-1).cpu()
    n = len(energy)
    peaks = []
    for i in range(n):
        left = energy[i - 1] if i > 0 else -1
        right = energy[i + 1] if i < n - 1 else -1
        if energy[i] > left and energy[i] > right:
            peaks.append((float(energy[i]), float(freqs[i].cpu())))
    peaks.sort(reverse=True)
    return [p[1] for p in peaks[:k]]


# ══════════════════════════════════════════════════════════════════════════
# CTBTO / IMS Infrasound — 0.02–4 Hz at sr=20 Hz
# Standard monitoring network for nuclear test-ban treaty
# ══════════════════════════════════════════════════════════════════════════

class TestCTBTO:
    """CTBTO/IMS infrasound station simulation (sr=20 Hz)."""

    SR = 20
    FMIN = 0.02
    FMAX = 4.0

    def test_detect_0_1hz(self):
        """Detect 0.1 Hz signal (typical microbarom)."""
        y = _make_tone(0.1, self.SR, 200.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 0.07 < detected < 0.15, f"expected ~0.1, got {detected}"

    def test_detect_1hz(self):
        """Detect 1 Hz signal (atmospheric disturbance)."""
        y = _make_tone(1.0, self.SR, 30.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 0.8 < detected < 1.25, f"expected ~1.0, got {detected}"

    def test_detect_0_02hz(self):
        """Detect 0.02 Hz — lowest CTBTO band edge."""
        dur = 500.0  # 10 full cycles at 0.02 Hz
        y = _make_tone(0.02, self.SR, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=0.01, fmax=self.FMAX,
            scales_per_octave=8, wavelet="morlet", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 0.012 < detected < 0.035, f"expected ~0.02, got {detected}"

    def test_two_tone_separation(self):
        """Separate 0.2 Hz microbarom from 2.0 Hz atmospheric event."""
        y = _make_multi_tone([0.2, 2.0], self.SR, 60.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        peaks = _top_k_freqs(W, freqs, 2)
        peaks.sort()
        assert len(peaks) >= 2
        assert 0.12 < peaks[0] < 0.35, f"low peak {peaks[0]}"
        assert 1.4 < peaks[1] < 2.8, f"high peak {peaks[1]}"

    def test_audify_ctbto(self):
        """Audify a CTBTO-band signal to audible range."""
        y = _make_multi_tone([0.1, 0.5, 2.0], self.SR, 30.0, device=_DEVICE)
        audio, out_sr, f_in, f_out = torch_cqt.audify(
            y, self.SR,
            fmin=self.FMIN, fmax=self.FMAX,
            target_sr=44100, target_hz=440.0,
            scales_per_octave=12,
            device=_DEVICE,
        )
        assert out_sr == 44100
        assert audio.shape[0] > 0
        assert not torch.isnan(audio).any()
        assert not torch.isinf(audio).any()
        # Audible freqs should be >> original
        assert float(f_out[0]) > 100.0


# ══════════════════════════════════════════════════════════════════════════
# Microbaroms — ocean-generated, 0.1–0.5 Hz at sr=20 Hz
# ══════════════════════════════════════════════════════════════════════════

class TestMicrobaroms:
    """Microbarom detection (ocean-atmosphere coupling)."""

    SR = 20
    FMIN = 0.05
    FMAX = 1.0

    def test_detect_microbarom_peak(self):
        """0.2 Hz dominant microbarom."""
        y = _make_tone(0.2, self.SR, 100.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=16, wavelet="morlet", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 0.14 < detected < 0.28, f"expected ~0.2, got {detected}"

    def test_detect_double_microbarom(self):
        """0.15 + 0.30 Hz (ocean swell doublet)."""
        y = _make_multi_tone([0.15, 0.30], self.SR, 120.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=24, wavelet="morlet", device=_DEVICE)
        peaks = _top_k_freqs(W, freqs, 2)
        peaks.sort()
        assert len(peaks) >= 2
        assert 0.10 < peaks[0] < 0.22
        assert 0.22 < peaks[1] < 0.42

    def test_morse_microbarom(self):
        """Morse wavelet on microbarom band."""
        y = _make_tone(0.2, self.SR, 100.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morse", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 0.12 < detected < 0.35, f"expected ~0.2, got {detected}"


# ══════════════════════════════════════════════════════════════════════════
# Volcanic infrasound — 0.01–10 Hz at sr=100 Hz
# ══════════════════════════════════════════════════════════════════════════

class TestVolcanic:
    """Volcanic infrasound signatures."""

    SR = 100
    FMIN = 0.01
    FMAX = 10.0

    def test_harmonic_tremor(self):
        """1.5 Hz harmonic tremor."""
        y = _make_tone(1.5, self.SR, 30.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 1.1 < detected < 2.0, f"expected ~1.5, got {detected}"

    def test_explosion_broadband(self):
        """Broadband volcanic explosion — energy across 0.5–5 Hz."""
        # Simulate with multiple tones
        y = _make_multi_tone(
            [0.5, 1.0, 2.0, 5.0], self.SR, 20.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        peaks = _top_k_freqs(W, freqs, 4)
        peaks.sort()
        assert len(peaks) >= 3  # should find at least 3 of the 4
        assert peaks[0] < 1.5

    def test_very_low_volcanic(self):
        """0.05 Hz very-long-period volcanic event."""
        dur = 300.0  # 15 cycles
        y = _make_tone(0.05, self.SR, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=8, wavelet="morlet", device=_DEVICE,
            epsilon=0.02)
        detected = _peak_freq(W, freqs)
        assert 0.03 < detected < 0.08, f"expected ~0.05, got {detected}"

    def test_audify_volcanic(self):
        """Audify volcanic tremor to audible range."""
        y = _make_tone(1.5, self.SR, 10.0, device=_DEVICE)
        audio, out_sr, f_in, f_out = torch_cqt.audify(
            y, self.SR,
            fmin=0.5, fmax=self.FMAX,
            target_sr=44100, target_hz=440.0,
            scales_per_octave=12, device=_DEVICE,
        )
        assert out_sr == 44100
        assert audio.shape[0] > 0
        assert not torch.isnan(audio).any()


# ══════════════════════════════════════════════════════════════════════════
# Earthquake precursors / tilt — 0.001–0.1 Hz at sr=10 Hz
# ══════════════════════════════════════════════════════════════════════════

class TestEarthquake:
    """Ultra-low-frequency earthquake precursor / tilt signals."""

    SR = 10
    FMIN = 0.001
    FMAX = 0.1

    def test_detect_0_01hz(self):
        """0.01 Hz tilt precursor (100-second period)."""
        dur = 2000.0  # 20 full cycles
        y = _make_tone(0.01, self.SR, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=8, wavelet="morlet", device=_DEVICE,
            epsilon=0.005)
        detected = _peak_freq(W, freqs)
        assert 0.006 < detected < 0.018, f"expected ~0.01, got {detected}"

    def test_detect_0_05hz(self):
        """0.05 Hz precursor signal."""
        dur = 400.0  # 20 cycles
        y = _make_tone(0.05, self.SR, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=8, wavelet="morlet", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 0.03 < detected < 0.08, f"expected ~0.05, got {detected}"

    def test_audify_earthquake(self):
        """Audify ultra-low earthquake data."""
        dur = 200.0
        y = _make_multi_tone([0.01, 0.05], self.SR, dur, device=_DEVICE)
        audio, out_sr, f_in, f_out = torch_cqt.audify(
            y, self.SR,
            fmin=self.FMIN, fmax=self.FMAX,
            target_sr=44100, target_hz=440.0,
            scales_per_octave=8, device=_DEVICE,
            epsilon=0.005,
        )
        assert out_sr == 44100
        assert audio.shape[0] > 0
        assert not torch.isnan(audio).any()


# ══════════════════════════════════════════════════════════════════════════
# Hydrophone / SOFAR channel — 1–100 Hz at sr=1000 Hz
# ══════════════════════════════════════════════════════════════════════════

class TestHydrophone:
    """Hydrophone / underwater acoustic monitoring."""

    SR = 1000
    FMIN = 1.0
    FMAX = 100.0

    def test_whale_20hz(self):
        """20 Hz fin whale call."""
        y = _make_tone(20.0, self.SR, 5.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 16.0 < detected < 26.0, f"expected ~20, got {detected}"

    def test_whale_plus_ship(self):
        """Separate 20 Hz whale from 60 Hz ship noise."""
        y = _make_multi_tone([20.0, 60.0], self.SR, 5.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        peaks = _top_k_freqs(W, freqs, 2)
        peaks.sort()
        assert len(peaks) >= 2
        assert 14.0 < peaks[0] < 28.0
        assert 45.0 < peaks[1] < 80.0

    def test_t_phase_5hz(self):
        """5 Hz T-phase from underwater earthquake."""
        y = _make_tone(5.0, self.SR, 10.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 3.5 < detected < 7.0, f"expected ~5, got {detected}"

    def test_audify_hydrophone(self):
        """Audify hydrophone data — already partially audible."""
        y = _make_multi_tone([5.0, 20.0, 60.0], self.SR, 5.0, device=_DEVICE)
        audio, out_sr, f_in, f_out = torch_cqt.audify(
            y, self.SR,
            fmin=self.FMIN, fmax=self.FMAX,
            target_sr=44100, target_hz=440.0,
            scales_per_octave=12, device=_DEVICE,
        )
        assert out_sr == 44100
        assert audio.shape[0] > 0
        assert not torch.isnan(audio).any()


# ══════════════════════════════════════════════════════════════════════════
# Wind turbine infrasound — 0.5–10 Hz at sr=200 Hz
# ══════════════════════════════════════════════════════════════════════════

class TestWindTurbine:
    """Wind turbine infrasound monitoring."""

    SR = 200
    FMIN = 0.5
    FMAX = 10.0

    def test_blade_pass_1hz(self):
        """1.0 Hz blade-pass frequency."""
        y = _make_tone(1.0, self.SR, 30.0, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 0.75 < detected < 1.35, f"expected ~1.0, got {detected}"

    def test_harmonics(self):
        """Blade-pass with harmonics: 1 + 2 + 3 Hz."""
        y = _make_multi_tone(
            [1.0, 2.0, 3.0], self.SR, 20.0,
            amps=[1.0, 0.5, 0.25], device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        peaks = _top_k_freqs(W, freqs, 3)
        peaks.sort()
        assert len(peaks) >= 2
        assert 0.7 < peaks[0] < 1.4
        assert 1.4 < peaks[1] < 2.6


# ══════════════════════════════════════════════════════════════════════════
# Astronomical — millihertz gravitational-wave proxy at sr=4 Hz
# ══════════════════════════════════════════════════════════════════════════

class TestAstronomical:
    """Millihertz regime — gravitational-wave analog detection."""

    SR = 4
    FMIN = 0.001
    FMAX = 0.5

    def test_detect_10mhz(self):
        """0.01 Hz (10 mHz) periodic signal."""
        dur = 2000.0  # 20 cycles
        y = _make_tone(0.01, self.SR, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=8, wavelet="morlet", device=_DEVICE,
            epsilon=0.005)
        detected = _peak_freq(W, freqs)
        assert 0.006 < detected < 0.018, f"expected ~0.01, got {detected}"

    def test_detect_100mhz(self):
        """0.1 Hz (100 mHz) — upper millihertz band."""
        dur = 200.0  # 20 cycles
        y = _make_tone(0.1, self.SR, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, self.SR, fmin=self.FMIN, fmax=self.FMAX,
            scales_per_octave=8, wavelet="morlet", device=_DEVICE)
        detected = _peak_freq(W, freqs)
        assert 0.06 < detected < 0.16, f"expected ~0.1, got {detected}"

    def test_audify_millihertz(self):
        """Audify millihertz data to concert A."""
        dur = 500.0
        y = _make_multi_tone([0.01, 0.1], self.SR, dur, device=_DEVICE)
        audio, out_sr, f_in, f_out = torch_cqt.audify(
            y, self.SR,
            fmin=self.FMIN, fmax=self.FMAX,
            target_sr=44100, target_hz=440.0,
            scales_per_octave=8, device=_DEVICE,
            epsilon=0.005,
        )
        assert out_sr == 44100
        assert audio.shape[0] > 0
        assert not torch.isnan(audio).any()
        assert not torch.isinf(audio).any()
        # Shift factor should be massive (0.01 Hz → ~440 Hz = 44000×)
        assert float(f_out.max()) > 100.0


# ══════════════════════════════════════════════════════════════════════════
# iCWT round-trip accuracy
# ══════════════════════════════════════════════════════════════════════════

class TestICWTRoundTrip:
    """Verify forward→inverse CWT reconstruction quality."""

    def test_roundtrip_5hz(self):
        """CWT→iCWT round-trip for a 5 Hz tone at sr=200."""
        sr = 200
        freq = 5.0
        dur = 4.0
        y = _make_tone(freq, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=1.0, fmax=50.0,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        y_hat = torch_cqt.icwt(
            W, freqs, sr, wavelet="morlet", device=_DEVICE,
            length=y.shape[0])
        # Normalise both to unit energy for correlation
        y_n = y / y.norm()
        y_hat_n = y_hat / y_hat.norm().clamp(min=1e-30)
        corr = float(torch.dot(y_n, y_hat_n).abs().item())
        assert corr > 0.8, f"round-trip correlation {corr:.4f} < 0.8"

    def test_roundtrip_subsonic(self):
        """CWT→iCWT round-trip for 0.5 Hz tone at sr=20."""
        sr = 20
        freq = 0.5
        dur = 30.0
        y = _make_tone(freq, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.1, fmax=5.0,
            scales_per_octave=8, wavelet="morlet", device=_DEVICE)
        y_hat = torch_cqt.icwt(
            W, freqs, sr, wavelet="morlet", device=_DEVICE,
            length=y.shape[0])
        y_n = y / y.norm()
        y_hat_n = y_hat / y_hat.norm().clamp(min=1e-30)
        corr = float(torch.dot(y_n, y_hat_n).abs().item())
        assert corr > 0.7, f"round-trip correlation {corr:.4f} < 0.7"

    def test_roundtrip_earthquake(self):
        """CWT→iCWT round-trip for 0.01 Hz earthquake at sr=10, 2000s."""
        sr = 10
        freq = 0.01
        dur = 2000.0
        y = _make_tone(freq, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.001, fmax=0.1,
            scales_per_octave=8, wavelet="morlet", device=_DEVICE,
            epsilon=0.005)
        y_hat = torch_cqt.icwt(
            W, freqs, sr, wavelet="morlet", device=_DEVICE,
            length=y.shape[0], epsilon=0.005)
        y_n = y / y.norm()
        y_hat_n = y_hat / y_hat.norm().clamp(min=1e-30)
        corr = float(torch.dot(y_n, y_hat_n).abs().item())
        print(f"  earthquake round-trip corr={corr:.4f}")
        assert corr > 0.7, f"earthquake round-trip correlation {corr:.4f} < 0.7"

    def test_roundtrip_astronomical(self):
        """CWT→iCWT round-trip for millihertz (0.01 Hz) at sr=4, 2000s."""
        sr = 4
        freq = 0.01
        dur = 2000.0
        y = _make_tone(freq, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.001, fmax=0.5,
            scales_per_octave=8, wavelet="morlet", device=_DEVICE,
            epsilon=0.005)
        y_hat = torch_cqt.icwt(
            W, freqs, sr, wavelet="morlet", device=_DEVICE,
            length=y.shape[0], epsilon=0.005)
        y_n = y / y.norm()
        y_hat_n = y_hat / y_hat.norm().clamp(min=1e-30)
        corr = float(torch.dot(y_n, y_hat_n).abs().item())
        print(f"  astronomical round-trip corr={corr:.4f}")
        assert corr > 0.7, f"astronomical round-trip correlation {corr:.4f} < 0.7"

    def test_roundtrip_ctbto_002hz(self):
        """CWT→iCWT round-trip for CTBTO 0.02 Hz at sr=20."""
        sr = 20
        freq = 0.02
        dur = 500.0
        y = _make_tone(freq, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.005, fmax=1.0,
            scales_per_octave=8, wavelet="morlet", device=_DEVICE)
        y_hat = torch_cqt.icwt(
            W, freqs, sr, wavelet="morlet", device=_DEVICE,
            length=y.shape[0])
        y_n = y / y.norm()
        y_hat_n = y_hat / y_hat.norm().clamp(min=1e-30)
        corr = float(torch.dot(y_n, y_hat_n).abs().item())
        print(f"  CTBTO 0.02Hz round-trip corr={corr:.4f}")
        assert corr > 0.7, f"CTBTO round-trip correlation {corr:.4f} < 0.7"

    def test_roundtrip_volcanic_vlp(self):
        """CWT→iCWT round-trip for volcanic VLP 0.05 Hz at sr=20."""
        sr = 20
        freq = 0.05
        dur = 200.0
        y = _make_tone(freq, sr, dur, device=_DEVICE)
        W, freqs = torch_cqt.cwt(
            y, sr, fmin=0.01, fmax=2.0,
            scales_per_octave=12, wavelet="morlet", device=_DEVICE)
        y_hat = torch_cqt.icwt(
            W, freqs, sr, wavelet="morlet", device=_DEVICE,
            length=y.shape[0])
        y_n = y / y.norm()
        y_hat_n = y_hat / y_hat.norm().clamp(min=1e-30)
        corr = float(torch.dot(y_n, y_hat_n).abs().item())
        print(f"  volcanic VLP round-trip corr={corr:.4f}")
        assert corr > 0.7, f"volcanic VLP round-trip correlation {corr:.4f} < 0.7"


# ══════════════════════════════════════════════════════════════════════════
# Audify structure tests
# ══════════════════════════════════════════════════════════════════════════

class TestAudifyStructure:
    """Verify audify() output properties and frequency mapping."""

    def test_shift_factor_auto(self):
        """Auto shift_factor maps median freq to target_hz."""
        sr = 20
        y = _make_tone(1.0, sr, 30.0, device=_DEVICE)
        _, _, f_in, f_out = torch_cqt.audify(
            y, sr, fmin=0.1, fmax=5.0,
            target_hz=440.0, target_sr=44100,
            scales_per_octave=12, device=_DEVICE,
        )
        # Median of output should be near 440 Hz
        median_out = float(f_out[len(f_out) // 2].item())
        assert 300.0 < median_out < 600.0, f"median audible = {median_out}"

    def test_shift_factor_explicit(self):
        """Explicit shift_factor is honoured."""
        sr = 20
        y = _make_tone(1.0, sr, 30.0, device=_DEVICE)
        _, _, f_in, f_out = torch_cqt.audify(
            y, sr, fmin=0.1, fmax=5.0,
            shift_factor=1000.0, target_sr=44100,
            scales_per_octave=12, device=_DEVICE,
        )
        # All output freqs should be input × 1000
        ratio = f_out / f_in
        assert torch.allclose(ratio, torch.full_like(ratio, 1000.0), rtol=1e-4)

    def test_nyquist_safety(self):
        """Shift factor auto-clamped to stay below target Nyquist."""
        sr = 20
        y = _make_tone(1.0, sr, 30.0, device=_DEVICE)
        _, out_sr, f_in, f_out = torch_cqt.audify(
            y, sr, fmin=0.1, fmax=5.0,
            target_hz=20000.0,  # would overshoot Nyquist
            target_sr=44100,
            scales_per_octave=12, device=_DEVICE,
        )
        assert float(f_out[0]) < out_sr / 2.0

    def test_output_dtype_preservation(self):
        """Output matches input dtype."""
        for dt in (torch.float32, torch.float64):
            sr = 20
            y = _make_tone(1.0, sr, 10.0, dtype=dt, device=_DEVICE)
            audio, _, _, _ = torch_cqt.audify(
                y, sr, fmin=0.5, fmax=5.0,
                target_sr=44100, scales_per_octave=8, device=_DEVICE,
            )
            assert audio.dtype == dt, f"expected {dt}, got {audio.dtype}"


# ══════════════════════════════════════════════════════════════════════════
# Data loader tests (infrasound_io)
# ══════════════════════════════════════════════════════════════════════════

class TestInfrasoundIO:
    """Test data loading utilities."""

    def test_load_numpy_npy(self, tmp_path):
        """Load a .npy file."""
        import infrasound_io

        arr = np.random.randn(10000).astype(np.float32)
        path = tmp_path / "test.npy"
        np.save(str(path), arr)

        data, sr, meta = infrasound_io.load_numpy(path, sr=20)
        assert sr == 20
        assert data.shape == (10000,)
        assert meta["format"] == "numpy"

    def test_load_numpy_npz(self, tmp_path):
        """Load a .npz file."""
        import infrasound_io

        arr = np.random.randn(5000).astype(np.float64)
        path = tmp_path / "test.npz"
        np.savez(str(path), pressure=arr)

        data, sr, meta = infrasound_io.load_numpy(path, sr=100, key="pressure")
        assert sr == 100
        assert data.shape == (5000,)

    def test_load_raw_float32(self, tmp_path):
        """Load raw float32 binary."""
        import infrasound_io

        arr = np.random.randn(2000).astype(np.float32)
        path = tmp_path / "test.bin"
        path.write_bytes(arr.tobytes())

        data, sr, meta = infrasound_io.load_raw(path, sr=20, dtype="float32")
        assert sr == 20
        assert data.shape == (2000,)
        assert torch.allclose(data, torch.from_numpy(arr), atol=1e-7)

    def test_load_auto_detect(self, tmp_path):
        """Auto-detect .npy format."""
        import infrasound_io

        arr = np.random.randn(1000).astype(np.float32)
        path = tmp_path / "data.npy"
        np.save(str(path), arr)

        data, sr, meta = infrasound_io.load(path, sr=20)
        assert sr == 20
        assert data.shape == (1000,)
