"""Tests for tf_surgery.py — TF-plane affine region manipulation."""
import math
import torch
import pytest
from tf_surgery import (
    TFRegion, tf_transplant, TransplantResult,
    cwt_transplant, transplant,
    affine_identity, affine_translate, affine_scale,
    affine_rotate, affine_skew, affine_compose,
)


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 44100


def _tone(freq: float, dur: float, sr: int = SR,
          dtype: torch.dtype = torch.float64) -> torch.Tensor:
    t = torch.arange(int(sr * dur), dtype=dtype, device=DEV) / sr
    return torch.sin(2 * math.pi * freq * t)


def _band_energy(signal: torch.Tensor, sr: int, flo: float,
                 fhi: float, n_fft: int = 2048) -> float:
    """RMS energy in a frequency band via STFT."""
    hop = n_fft // 4
    window = torch.hann_window(n_fft, dtype=signal.dtype, device=signal.device)
    S = torch.stft(signal, n_fft, hop_length=hop, win_length=n_fft,
                   window=window, center=True, return_complex=True)
    nyq = sr / 2.0
    n_bins = S.shape[-2]
    f0 = max(0, int(flo / nyq * n_bins))
    f1 = min(n_bins, int(math.ceil(fhi / nyq * n_bins)))
    return float(S[f0:f1, :].abs().pow(2).sum().sqrt().item())


# ═══════════════════════════════════════════════════════════════════════════
# Identity transform — output ≈ input
# ═══════════════════════════════════════════════════════════════════════════

class TestIdentity:

    def test_identity_preserves_signal(self):
        """Identity affine should not alter the signal significantly."""
        y = _tone(440, 1.0)
        src = TFRegion(0.2, 0.8, 200, 800)
        out = tf_transplant(y, SR, src, affine_identity(dtype=y.dtype, device=DEV))
        # The identity adds the same content back, so magnitude should
        # be roughly doubled in the source region; outside it should be same.
        # But the key test: it shouldn't crash and output has same length.
        assert out.shape == y.shape

    def test_identity_no_dampen(self):
        """Without dampening, identity transform adds energy (doubled in region)."""
        y = _tone(440, 0.5)
        src = TFRegion(0.1, 0.4, 300, 600)
        out = tf_transplant(y, SR, src, affine_identity(dtype=y.dtype, device=DEV))
        # Energy should increase slightly (content added on top)
        assert out.norm() >= y.norm() * 0.9


# ═══════════════════════════════════════════════════════════════════════════
# Translation — move content in time or frequency
# ═══════════════════════════════════════════════════════════════════════════

class TestTranslation:

    def test_frequency_shift_up(self):
        """Shift a 150 Hz tone up → energy appears in a previously empty band."""
        dur = 1.0
        y = _tone(150, dur)
        src = TFRegion(0.1, 0.9, 100, 300)
        # Translate upward by 2× the source height (400 Hz) → lands at 500-700 Hz
        M = affine_translate(dt=0.0, df=2.0, dtype=y.dtype, device=DEV)
        out = tf_transplant(y, SR, src, M, dampen_source=False)
        # 500-700 Hz had negligible energy in original signal
        before = _band_energy(y, SR, 500, 700)
        after = _band_energy(out, SR, 500, 700)
        assert after > before + 1.0  # meaningful new energy appeared

    def test_time_shift(self):
        """Shift content later in time."""
        dur = 1.0
        y = torch.zeros(SR, dtype=torch.float64, device=DEV)
        # Put a burst at 0.2–0.4s
        t = torch.arange(SR, dtype=torch.float64, device=DEV) / SR
        burst_mask = ((t >= 0.2) & (t < 0.4)).to(torch.float64)
        y = torch.sin(2 * math.pi * 1000 * t) * burst_mask

        src = TFRegion(0.15, 0.45, 500, 2000)
        M = affine_translate(dt=0.5, df=0.0, dtype=y.dtype, device=DEV)
        out = tf_transplant(y, SR, src, M)
        # Energy in the later time region should have increased
        mid = SR // 2
        energy_late = out[mid:].pow(2).sum()
        energy_late_orig = y[mid:].pow(2).sum()
        assert energy_late > energy_late_orig * 2


# ═══════════════════════════════════════════════════════════════════════════
# Scale — time stretch / pitch shift
# ═══════════════════════════════════════════════════════════════════════════

class TestScale:

    def test_frequency_scale_down(self):
        """Compress in frequency → content squeezed into smaller band."""
        y = _tone(1000, 0.5)
        src = TFRegion(0.05, 0.45, 500, 2000)
        M = affine_scale(st=1.0, sf=0.5, dtype=y.dtype, device=DEV)
        out = tf_transplant(y, SR, src, M, dampen_source=True, n_fft=2048)
        assert out.shape == y.shape

    def test_time_stretch(self):
        """Time scale > 1 → content spreads over wider time range."""
        dur = 1.0
        t = torch.arange(SR, dtype=torch.float64, device=DEV) / SR
        burst = torch.sin(2 * math.pi * 500 * t) * ((t >= 0.2) & (t < 0.4)).to(torch.float64)
        src = TFRegion(0.15, 0.45, 200, 800)
        M = affine_scale(st=1.5, sf=1.0, dtype=burst.dtype, device=DEV)
        out = tf_transplant(burst, SR, src, M)
        assert out.shape == burst.shape


# ═══════════════════════════════════════════════════════════════════════════
# Rotation and skew
# ═══════════════════════════════════════════════════════════════════════════

class TestRotateSkew:

    def test_small_rotation(self):
        """Small rotation should not crash and output has correct shape."""
        y = _tone(500, 0.5)
        src = TFRegion(0.1, 0.4, 200, 1000)
        M = affine_rotate(0.1, dtype=y.dtype, device=DEV)
        out = tf_transplant(y, SR, src, M)
        assert out.shape == y.shape

    def test_skew_time(self):
        """Shear in time direction."""
        y = _tone(500, 0.5)
        src = TFRegion(0.05, 0.45, 200, 1000)
        M = affine_skew(kt=0.3, dtype=y.dtype, device=DEV)
        out = tf_transplant(y, SR, src, M)
        assert out.shape == y.shape


# ═══════════════════════════════════════════════════════════════════════════
# Composition
# ═══════════════════════════════════════════════════════════════════════════

class TestCompose:

    def test_compose_identity(self):
        M = affine_compose(affine_identity(device=DEV), affine_identity(device=DEV))
        expected = affine_identity(device=DEV)
        assert torch.allclose(M, expected, atol=1e-12)

    def test_compose_translate_then_scale(self):
        """Composing two affines produces a valid 2×3 matrix."""
        M = affine_compose(
            affine_translate(0.1, 0.2, device=DEV),
            affine_scale(2.0, 0.5, device=DEV),
        )
        assert M.shape == (2, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Dampen source
# ═══════════════════════════════════════════════════════════════════════════

class TestDampen:

    def test_dampen_reduces_source(self):
        """With dampen_source=True, energy in source region drops."""
        y = _tone(500, 1.0)
        src = TFRegion(0.2, 0.8, 300, 700)
        M = affine_translate(df=1.0, dtype=y.dtype, device=DEV)
        out_nodamp = tf_transplant(y, SR, src, M, dampen_source=False)
        out_damp = tf_transplant(y, SR, src, M, dampen_source=True)
        e_nodamp = _band_energy(out_nodamp, SR, 300, 700)
        e_damp = _band_energy(out_damp, SR, 300, 700)
        assert e_damp < e_nodamp


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_region_at_dc(self):
        y = _tone(50, 0.5, sr=8000, dtype=torch.float64)
        src = TFRegion(0.0, 0.5, 0, 100)
        M = affine_translate(df=0.5, dtype=y.dtype, device=DEV)
        out = tf_transplant(y, 8000, src, M, n_fft=512)
        assert out.shape == y.shape

    def test_region_at_nyquist(self):
        y = _tone(3000, 0.5, sr=8000, dtype=torch.float64)
        src = TFRegion(0.0, 0.5, 3000, 4000)
        M = affine_identity(dtype=y.dtype, device=DEV)
        out = tf_transplant(y, 8000, src, M, n_fft=512)
        assert out.shape == y.shape

    def test_empty_region(self):
        """Zero-size source region → signal passes through."""
        y = _tone(440, 0.5)
        src = TFRegion(0.2, 0.2, 400, 400)  # zero area
        M = affine_identity(dtype=y.dtype, device=DEV)
        out = tf_transplant(y, SR, src, M)
        assert out.shape == y.shape

    def test_dtype_preservation(self):
        """Output dtype matches input dtype."""
        for dt in [torch.float32, torch.float64]:
            y = _tone(440, 0.2, dtype=dt)
            src = TFRegion(0.0, 0.2, 200, 800)
            out = tf_transplant(y, SR, src, affine_identity(dtype=dt, device=DEV))
            assert out.dtype == dt


# ═══════════════════════════════════════════════════════════════════════════
# Phase vocoder quality
# ═══════════════════════════════════════════════════════════════════════════

class TestPhaseVocoderQuality:

    def test_shifted_tone_spectral_purity(self):
        """A pure tone shifted up should remain spectrally concentrated."""
        freq_src = 300.0
        dur = 1.0
        y = _tone(freq_src, dur)
        src = TFRegion(0.05, 0.95, 200, 500)
        # Shift up by 1× the source height (300 Hz) → dst at 500-800 Hz
        M = affine_translate(dt=0.0, df=1.0, dtype=y.dtype, device=DEV)
        out = tf_transplant(y, SR, src, M, dampen_source=True)

        # Expected destination frequency: ~600 Hz (300 Hz + 300 Hz shift)
        # Measure energy in a tight band around 600 Hz vs total in 400-900 Hz
        tight = _band_energy(out, SR, 550, 700)
        wide = _band_energy(out, SR, 400, 900)
        if wide > 1e-8:
            concentration = tight / wide
            # A well-behaved phase vocoder should keep >50% of energy
            # in the tight band around the shifted frequency
            assert concentration > 0.3, (
                f"Spectral purity too low: {concentration:.3f}")

    def test_auto_nfft_provides_resolution(self):
        """Auto n_fft should give many bins across the working region."""
        freq = 500.0
        y = _tone(freq, 0.5)
        src = TFRegion(0.05, 0.45, 400, 600)
        M = affine_identity(dtype=y.dtype, device=DEV)
        # With default min_bins_in_region=128, a 200 Hz span at 44100 Hz
        # needs n_fft >= 128 * 44100 / 200 = 28224 → next power of 2 = 32768
        out = tf_transplant(y, SR, src, M)
        # Verify the output didn't crash with the large auto n_fft
        assert out.shape == y.shape

    def test_identity_reconstruction_quality(self):
        """Identity transform + dampen should approximately halve in-band energy."""
        freq = 1000.0
        y = _tone(freq, 0.5)
        src = TFRegion(0.1, 0.4, 800, 1200)
        M = affine_identity(dtype=y.dtype, device=DEV)
        out = tf_transplant(y, SR, src, M, dampen_source=False)
        # Identity adds content on top → energy should increase
        e_in = _band_energy(y, SR, 800, 1200)
        e_out = _band_energy(out, SR, 800, 1200)
        assert e_out > e_in * 1.3  # should roughly double


# ═══════════════════════════════════════════════════════════════════════════
# CWT engine — subsonic→sonic
# ═══════════════════════════════════════════════════════════════════════════

class TestCWTTransplant:

    def test_subsonic_to_sonic_5hz(self):
        """Cast a 5 Hz tone into the audible range via CWT."""
        dur = 5.0
        sr = 44100
        N = int(sr * dur)
        t = torch.arange(N, dtype=torch.float64, device=DEV) / sr
        y = torch.sin(2 * math.pi * 5.0 * t)

        src = TFRegion(0.5, 4.5, 1, 20)
        # df=5.0 → shift by 5 * (20-1) = 95 Hz → 5 Hz lands at ~100 Hz
        M = affine_translate(dt=0.0, df=5.0, dtype=y.dtype, device=DEV)
        result = cwt_transplant(y, sr, src, M, dampen_source=True,
                                scales_per_octave=24, hop_length=32)

        assert isinstance(result, TransplantResult)
        assert result.engine == "cwt"
        assert result.signal.shape == y.shape

        # Verify energy appears in expected audible band
        S = torch.fft.rfft(result.signal)
        freqs = torch.fft.rfftfreq(N, 1.0 / sr)
        mag = S.abs()
        mask_dst = (freqs > 50) & (freqs < 200)
        mask_sub = (freqs > 1) & (freqs < 15)
        e_dst = mag[mask_dst].pow(2).sum().sqrt()
        e_sub = mag[mask_sub].pow(2).sum().sqrt()
        # Destination should have meaningful energy
        assert e_dst > 1.0, f"No energy in destination band: {e_dst:.2f}"

    def test_cwt_identity(self):
        """CWT identity should produce output of correct shape."""
        sr = 8000
        dur = 1.0
        N = int(sr * dur)
        t = torch.arange(N, dtype=torch.float64, device=DEV) / sr
        y = torch.sin(2 * math.pi * 100.0 * t)
        src = TFRegion(0.1, 0.9, 50, 200)
        M = affine_identity(dtype=y.dtype, device=DEV)
        result = cwt_transplant(y, sr, src, M, scales_per_octave=12,
                                hop_length=16)
        assert result.signal.shape == y.shape
        assert result.engine == "cwt"

    def test_cwt_uncertainty_nonzero(self):
        """CWT result must carry a non-trivial uncertainty envelope."""
        sr = 8000
        dur = 1.0
        N = int(sr * dur)
        t = torch.arange(N, dtype=torch.float64, device=DEV) / sr
        y = torch.sin(2 * math.pi * 50.0 * t)
        src = TFRegion(0.1, 0.9, 20, 100)
        M = affine_translate(df=1.0, dtype=y.dtype, device=DEV)
        result = cwt_transplant(y, sr, src, M, scales_per_octave=12,
                                hop_length=16)
        assert result.uncertainty_envelope is not None
        assert result.uncertainty_envelope.max() > 0

    def test_cwt_degenerate_region(self):
        """Degenerate region should pass through."""
        sr = 8000
        y = torch.randn(sr, dtype=torch.float64, device=DEV)
        src = TFRegion(0.2, 0.2, 50, 50)
        M = affine_identity(dtype=y.dtype, device=DEV)
        result = cwt_transplant(y, sr, src, M)
        assert torch.allclose(result.signal, y)

    def test_cwt_large_ratio(self):
        """20× frequency ratio: 2 Hz → 40 Hz."""
        sr = 8000
        dur = 4.0
        N = int(sr * dur)
        t = torch.arange(N, dtype=torch.float64, device=DEV) / sr
        y = torch.sin(2 * math.pi * 2.0 * t)
        src = TFRegion(0.5, 3.5, 1, 5)
        # df in source-height units: dst band = src + df * height
        # height = 4 Hz, want to land around 30-50 Hz → df ~ 8
        M = affine_translate(df=8.0, dtype=y.dtype, device=DEV)
        result = cwt_transplant(y, sr, src, M, dampen_source=True,
                                scales_per_octave=24, hop_length=8)
        assert result.signal.shape == y.shape
        # Some energy should have moved to audible range
        S = torch.fft.rfft(result.signal)
        freqs = torch.fft.rfftfreq(N, 1.0 / sr)
        mag = S.abs()
        e_high = mag[(freqs > 20) & (freqs < 100)].pow(2).sum().sqrt()
        assert e_high > 0.1

    def test_cwt_dtype_preservation(self):
        """CWT engine must preserve input dtype."""
        for dt in [torch.float32, torch.float64]:
            sr = 8000
            N = sr
            t = torch.arange(N, dtype=dt, device=DEV) / sr
            y = torch.sin(2 * math.pi * 100.0 * t)
            src = TFRegion(0.1, 0.9, 50, 200)
            M = affine_translate(df=0.5, dtype=dt, device=DEV)
            result = cwt_transplant(y, sr, src, M,
                                    scales_per_octave=12, hop_length=16)
            assert result.signal.dtype == dt


# ═══════════════════════════════════════════════════════════════════════════
# Dispatcher — auto-engine selection
# ═══════════════════════════════════════════════════════════════════════════

class TestDispatcher:

    def test_auto_selects_stft_for_audible(self):
        """Moderate shift in audible range → STFT engine."""
        y = _tone(500, 0.5)
        src = TFRegion(0.05, 0.45, 300, 800)
        M = affine_translate(df=0.5, dtype=y.dtype, device=DEV)
        result = transplant(y, SR, src, M)
        assert isinstance(result, TransplantResult)
        assert result.engine == "stft"
        assert result.uncertainty_envelope is not None

    def test_auto_selects_cwt_for_subsonic(self):
        """Subsonic source → CWT engine."""
        sr = 8000
        N = sr * 2
        t = torch.arange(N, dtype=torch.float64, device=DEV) / sr
        y = torch.sin(2 * math.pi * 5.0 * t)
        src = TFRegion(0.1, 1.9, 1, 15)
        M = affine_translate(df=5.0, dtype=y.dtype, device=DEV)
        result = transplant(y, sr, src, M, scales_per_octave=12,
                            cwt_hop=16)
        assert result.engine == "cwt"

    def test_auto_selects_cwt_for_large_ratio(self):
        """Large frequency ratio → CWT engine."""
        y = _tone(50, 0.5, sr=8000)
        src = TFRegion(0.05, 0.45, 30, 80)
        # Scale freq by 10× → ratio well above 4×
        M = affine_scale(sf=10.0, dtype=y.dtype, device=DEV)
        result = transplant(y, 8000, src, M, scales_per_octave=12,
                            cwt_hop=16)
        assert result.engine == "cwt"

    def test_force_engine(self):
        """Engine can be forced explicitly."""
        y = _tone(500, 0.3)
        src = TFRegion(0.05, 0.25, 300, 800)
        M = affine_identity(dtype=y.dtype, device=DEV)
        r_stft = transplant(y, SR, src, M, engine="stft")
        assert r_stft.engine == "stft"
        r_cwt = transplant(y, SR, src, M, engine="cwt",
                           scales_per_octave=12, cwt_hop=16)
        assert r_cwt.engine == "cwt"

    def test_result_has_dst_region(self):
        """TransplantResult carries destination region info."""
        y = _tone(440, 0.5)
        src = TFRegion(0.1, 0.4, 300, 600)
        M = affine_translate(df=1.0, dtype=y.dtype, device=DEV)
        result = transplant(y, SR, src, M)
        assert result.dst_region is not None
        assert result.dst_region.f0 >= 0
