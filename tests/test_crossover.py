"""Tests for crossover.py — perfectly-reconstructing LR filter bank."""
import math
import torch
import pytest
from crossover import lr_decompose, lr_reconstruct, lr_band_response, log_crossovers


# ── Perfect reconstruction ───────────────────────────────────────────────


class TestPerfectReconstruction:
    """Sum of bands must equal original signal to machine precision."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_single_crossover(self, dtype):
        sr = 44100
        t = torch.arange(sr, dtype=dtype) / sr
        y = torch.sin(2 * math.pi * 440 * t) + 0.5 * torch.sin(2 * math.pi * 80 * t)
        bands = lr_decompose(y, [200.0], sr)
        assert bands.shape == (2, sr)
        recon = lr_reconstruct(bands)
        assert torch.allclose(recon, y, atol=1e-6 if dtype == torch.float32 else 1e-12)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_multi_crossover(self, dtype):
        sr = 48000
        N = sr * 2
        y = torch.randn(N, dtype=dtype)
        xo = [80.0, 250.0, 1000.0, 4000.0, 12000.0]
        bands = lr_decompose(y, xo, sr)
        assert bands.shape[0] == 6
        recon = lr_reconstruct(bands)
        assert torch.allclose(recon, y, atol=1e-5 if dtype == torch.float32 else 1e-11)

    def test_single_band_passthrough(self):
        """No crossovers → single band == input."""
        y = torch.randn(8000)
        bands = lr_decompose(y, [], 44100)
        assert bands.shape == (1, 8000)
        assert torch.allclose(bands[0], y)

    def test_white_noise(self):
        """Random signal: bands sum to original."""
        sr = 44100
        y = torch.randn(sr, dtype=torch.float64)
        xo = [100.0, 500.0, 2000.0, 8000.0]
        bands = lr_decompose(y, xo, sr)
        recon = lr_reconstruct(bands)
        assert torch.allclose(recon, y, atol=1e-11)

    def test_impulse(self):
        """Dirac impulse: bands sum to delta."""
        sr = 44100
        y = torch.zeros(sr, dtype=torch.float64)
        y[sr // 2] = 1.0
        bands = lr_decompose(y, [200.0, 2000.0], sr)
        recon = lr_reconstruct(bands)
        assert torch.allclose(recon, y, atol=1e-12)

    @pytest.mark.parametrize("order", [4, 8, 16, 32])
    def test_orders(self, order):
        """Perfect reconstruction holds at all orders."""
        sr = 44100
        y = torch.randn(sr, dtype=torch.float64)
        bands = lr_decompose(y, [500.0, 3000.0], sr, order=order)
        recon = lr_reconstruct(bands)
        assert torch.allclose(recon, y, atol=1e-11)


# ── Band isolation / rejection ───────────────────────────────────────────


class TestBandIsolation:
    """Verify energy lands in the correct band."""

    def test_tone_in_low_band(self):
        sr = 44100
        t = torch.arange(sr, dtype=torch.float64) / sr
        y = torch.sin(2 * math.pi * 100 * t)
        bands = lr_decompose(y, [500.0], sr, order=8)
        low_energy = bands[0].pow(2).sum()
        high_energy = bands[1].pow(2).sum()
        assert low_energy > 100 * high_energy

    def test_tone_in_high_band(self):
        sr = 44100
        t = torch.arange(sr, dtype=torch.float64) / sr
        y = torch.sin(2 * math.pi * 5000 * t)
        bands = lr_decompose(y, [500.0], sr, order=8)
        low_energy = bands[0].pow(2).sum()
        high_energy = bands[1].pow(2).sum()
        assert high_energy > 100 * low_energy

    def test_three_tones_three_bands(self):
        sr = 44100
        t = torch.arange(sr, dtype=torch.float64) / sr
        y = (torch.sin(2 * math.pi * 50 * t)
             + torch.sin(2 * math.pi * 1000 * t)
             + torch.sin(2 * math.pi * 10000 * t))
        bands = lr_decompose(y, [200.0, 4000.0], sr, order=16)
        energies = [b.pow(2).sum().item() for b in bands]
        # Each tone's energy dominates its band
        assert energies[0] > 0.4 * (sum(energies) / 3)
        assert energies[1] > 0.4 * (sum(energies) / 3)
        assert energies[2] > 0.4 * (sum(energies) / 3)

    def test_higher_order_sharper(self):
        """Higher order → more rejection in the stopband."""
        sr = 44100
        t = torch.arange(sr, dtype=torch.float64) / sr
        y = torch.sin(2 * math.pi * 100 * t)
        bands_4 = lr_decompose(y, [500.0], sr, order=4)
        bands_16 = lr_decompose(y, [500.0], sr, order=16)
        leak_4 = bands_4[1].pow(2).sum().item()
        leak_16 = bands_16[1].pow(2).sum().item()
        assert leak_16 < leak_4


# ── Band response ────────────────────────────────────────────────────────


class TestBandResponse:
    """lr_band_response must sum to unity."""

    def test_unity_sum(self):
        freqs, resp = lr_band_response([500.0, 3000.0], 44100, order=8)
        total = resp.sum(dim=0)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6)

    def test_single_crossover_shape(self):
        freqs, resp = lr_band_response([1000.0], 44100)
        assert resp.shape[0] == 2
        assert resp.shape[1] == 4096

    def test_many_bands(self):
        xo = [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        freqs, resp = lr_band_response(xo, 48000, order=16)
        assert resp.shape[0] == 8
        total = resp.sum(dim=0)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


# ── log_crossovers ───────────────────────────────────────────────────────


class TestLogCrossovers:

    def test_octave_bands(self):
        xo = log_crossovers(100.0, 10000.0, bands_per_octave=1)
        # ~6.6 octaves → expect ~6 crossovers
        assert 4 <= len(xo) <= 8
        assert all(xo[i] < xo[i + 1] for i in range(len(xo) - 1))

    def test_third_octave(self):
        xo = log_crossovers(100.0, 10000.0, bands_per_octave=3)
        assert len(xo) > 15

    def test_degenerate(self):
        assert log_crossovers(0, 100, 1) == []
        assert log_crossovers(100, 50, 1) == []


# ── Batch dimensions ─────────────────────────────────────────────────────


class TestBatch:
    """Leading batch dimensions are preserved."""

    def test_2d_batch(self):
        sr = 44100
        y = torch.randn(3, sr)
        bands = lr_decompose(y, [500.0], sr)
        assert bands.shape == (3, 2, sr)
        recon = lr_reconstruct(bands)
        assert torch.allclose(recon, y, atol=1e-5)

    def test_3d_batch(self):
        sr = 8000
        y = torch.randn(2, 4, sr)
        bands = lr_decompose(y, [1000.0, 3000.0], sr)
        assert bands.shape == (2, 4, 3, sr)
        recon = lr_reconstruct(bands)
        assert torch.allclose(recon, y, atol=1e-5)


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_crossover_outside_range_dropped(self):
        sr = 44100
        y = torch.randn(sr)
        bands = lr_decompose(y, [-10.0, 0.0, 25000.0], sr)
        assert bands.shape[0] == 1  # all invalid → single passthrough

    def test_duplicate_crossovers(self):
        sr = 44100
        y = torch.randn(sr)
        bands = lr_decompose(y, [500.0, 500.0, 500.0], sr)
        assert bands.shape[0] == 2

    def test_odd_length_signal(self):
        sr = 44100
        y = torch.randn(12345, dtype=torch.float64)
        bands = lr_decompose(y, [1000.0], sr)
        recon = lr_reconstruct(bands)
        assert recon.shape[-1] == 12345
        assert torch.allclose(recon, y, atol=1e-11)

    def test_order_validation(self):
        with pytest.raises(ValueError):
            lr_decompose(torch.randn(100), [500.0], 44100, order=3)
        with pytest.raises(ValueError):
            lr_decompose(torch.randn(100), [500.0], 44100, order=0)

    def test_very_short_signal(self):
        y = torch.randn(16, dtype=torch.float64)
        bands = lr_decompose(y, [1000.0], 44100, order=4)
        recon = lr_reconstruct(bands)
        assert torch.allclose(recon, y, atol=1e-10)
