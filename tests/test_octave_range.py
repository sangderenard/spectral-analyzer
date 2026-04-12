"""Test forward CQT across octaves -1..10 at various resolutions.

- Semitone / quarter-tone tests: compare against librosa on CPU (fast).
- Cent-resolution tests: GPU self-consistency smoke tests (no librosa).
  Validates shape, dtype, non-zero energy, tone-peak placement, and
  guardrail warnings.  Uses float32 + short signals to stay lightweight.
- Guardrail tests: verify warnings fire for misconfigurations.
- SR scaling tests: same config at different sample rates vs librosa.
"""
import math
import warnings
import numpy as np
import torch
import librosa
import pytest
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch_cqt_new as torch_cqt

# ── Device selection ─────────────────────────────────────────────────────

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ────────────────────────────────────────────────────────────

_C_MINUS1 = 8.175798915643707  # C-1 Hz

FILTER_SCALE = 1.0
HOP = 512
DURATION = 2.0  # seconds — enough for reasonable low-freq content


def _fmin_for_octave(octave: int) -> float:
    """C_{octave} in Hz.  C-1 = 8.176, C0 = 16.35, C4 = 261.6, etc."""
    return _C_MINUS1 * (2.0 ** (octave - (-1)))


def _make_signal(sr: int, duration: float, freqs: list[float]) -> np.ndarray:
    """Multi-tone test signal at native float64."""
    t = np.arange(int(sr * duration)) / sr
    y = np.zeros_like(t)
    for f in freqs:
        if f < sr / 2:
            y += np.sin(2 * np.pi * f * t)
    if np.abs(y).max() > 0:
        y = y / np.abs(y).max() * 0.8
    return y


# ── Librosa comparison helper ────────────────────────────────────────────

def _run_comparison(sr, bpo, fmin, n_bins, hop, y, label):
    """Run both librosa and torch_cqt forward, return per-octave stats."""
    n_octaves = int(math.ceil(n_bins / bpo))

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        try:
            C_lib = librosa.cqt(
                y, sr=sr, hop_length=hop, fmin=fmin,
                n_bins=n_bins, bins_per_octave=bpo,
                filter_scale=FILTER_SCALE, norm=1,
            )
        except Exception as e:
            if "Nyquist" in str(e):
                pytest.skip(f"librosa refuses this config: {e}")
            raise
        C_ours, freqs = torch_cqt.cqt(
            y, sr=sr, hop_length=hop, fmin=fmin,
            n_bins=n_bins, bins_per_octave=bpo,
            filter_scale=FILTER_SCALE, device=_DEVICE,
        )

    C_ours_np = C_ours.cpu().numpy()
    nf = min(C_lib.shape[1], C_ours_np.shape[1])
    C_lib = C_lib[:, :nf]
    C_ours_np = C_ours_np[:, :nf]

    results = {"label": label, "octaves": {}}
    for oct_i in range(n_octaves):
        sl = slice(oct_i * bpo, min((oct_i + 1) * bpo, n_bins))
        mag_lib = np.abs(C_lib[sl]).ravel()
        mag_ours = np.abs(C_ours_np[sl]).ravel()

        if mag_lib.max() == 0 and mag_ours.max() == 0:
            corr, ratio = 1.0, 1.0
        elif mag_lib.max() == 0 or mag_ours.max() == 0:
            corr, ratio = 0.0, 0.0
        else:
            corr = float(np.corrcoef(mag_lib, mag_ours)[0, 1])
            ratio = float(mag_ours.mean() / mag_lib.mean())
        results["octaves"][oct_i] = {
            "corr": corr, "ratio": ratio,
            "lib_mean": float(mag_lib.mean()),
            "ours_mean": float(mag_ours.mean()),
        }

    mag_l = np.abs(C_lib).ravel()
    mag_o = np.abs(C_ours_np).ravel()
    results["global_corr"] = (
        float(np.corrcoef(mag_l, mag_o)[0, 1])
        if mag_l.max() > 0 and mag_o.max() > 0 else 0.0
    )
    return results


# ══════════════════════════════════════════════════════════════════════════
# Semitone (BPO=12) — compare vs librosa, fast
# ══════════════════════════════════════════════════════════════════════════

class TestOctaveRangeSemitone:
    BPO = 12

    @pytest.mark.parametrize("oct_lo,oct_hi,sr", [
        (-1, 3, 44100),
        (0, 5, 44100),
        (2, 8, 44100),
        (4, 10, 44100),
        (-1, 10, 44100),
        (-1, 10, 96000),
    ])
    def test_octave_band(self, oct_lo, oct_hi, sr):
        fmin = _fmin_for_octave(oct_lo)
        n_octaves = oct_hi - oct_lo + 1
        n_bins = self.BPO * n_octaves
        nyquist = sr / 2.0
        tones = [fmin * 2 ** (i + 0.5) for i in range(n_octaves)
                 if fmin * 2 ** (i + 0.5) < nyquist]
        y = _make_signal(sr, DURATION, tones)

        r = _run_comparison(sr, self.BPO, fmin, n_bins, HOP, y,
                            f"semitone oct[{oct_lo},{oct_hi}] sr={sr}")

        print(f"\n{r['label']}  global_corr={r['global_corr']:.6f}")
        for oi, stats in sorted(r["octaves"].items()):
            print(f"  oct {oct_lo + oi:>3d}: corr={stats['corr']:.4f}  "
                  f"ratio={stats['ratio']:.4f}")

        assert r["global_corr"] > 0.95
        for oi, stats in r["octaves"].items():
            oct_top = fmin * 2 ** (oi + 1)
            if oct_top <= nyquist and stats["lib_mean"] > 1e-6:
                assert 0.5 < stats["ratio"] < 2.0


# ══════════════════════════════════════════════════════════════════════════
# Quarter-tone (BPO=48) — compare vs librosa
# ══════════════════════════════════════════════════════════════════════════

class TestOctaveRangeQuarterTone:
    BPO = 48

    @pytest.mark.parametrize("oct_lo,oct_hi,sr", [
        (-1, 6, 44100),
        (0, 8, 44100),
        (-1, 10, 44100),
        (-1, 10, 96000),
    ])
    def test_octave_band(self, oct_lo, oct_hi, sr):
        fmin = _fmin_for_octave(oct_lo)
        n_octaves = oct_hi - oct_lo + 1
        n_bins = self.BPO * n_octaves
        nyquist = sr / 2.0
        tones = [fmin * 2 ** (i + 0.5) for i in range(n_octaves)
                 if fmin * 2 ** (i + 0.5) < nyquist]
        y = _make_signal(sr, DURATION, tones)

        r = _run_comparison(sr, self.BPO, fmin, n_bins, HOP, y,
                            f"quarter-tone oct[{oct_lo},{oct_hi}] sr={sr}")

        print(f"\n{r['label']}  global_corr={r['global_corr']:.6f}")
        for oi, stats in sorted(r["octaves"].items()):
            print(f"  oct {oct_lo + oi:>3d}: corr={stats['corr']:.4f}  "
                  f"ratio={stats['ratio']:.4f}")

        assert r["global_corr"] > 0.95


# ══════════════════════════════════════════════════════════════════════════
# Cent resolution (BPO=1200) — GPU self-consistency, NO librosa
#
# Strategy: float32 (halves memory vs float64), short signals (0.5s),
# one octave at a time.  Filter bank per octave ≈ 1200 × 4097 × 8 bytes
# = ~39 MB (complex64).  GPU handles this in <1s per octave.
#
# Checks:
#   1. Output shape correct (1200 bins × expected frames)
#   2. Output dtype matches request
#   3. No NaN / Inf
#   4. Non-zero energy
#   5. Injected tone peaks near expected bin (±50 cents)
#   6. Freqs array correct
# ══════════════════════════════════════════════════════════════════════════

class TestCentResolution:
    """Cent resolution smoke tests — one octave at a time, float32, GPU."""

    BPO = 1200
    DUR = 0.5  # short signal to keep memory low

    @pytest.mark.parametrize("octave", list(range(-1, 11)))
    def test_single_octave_smoke(self, octave):
        """Cent-resolution CQT produces valid output for each octave."""
        sr = 44100
        fmin = _fmin_for_octave(octave)
        fmax = fmin * 2
        nyquist = sr / 2.0

        if fmax > nyquist:
            pytest.skip(f"fmax={fmax:.1f} > Nyquist={nyquist:.1f}")

        n_bins = self.BPO
        tone_freq = fmin * 2 ** 0.5  # geometric center

        t = np.arange(int(sr * self.DUR)) / sr
        y = (0.8 * np.sin(2 * np.pi * tone_freq * t)).astype(np.float32)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            C, freqs = torch_cqt.cqt(
                y, sr=sr, hop_length=HOP, fmin=fmin,
                n_bins=n_bins, bins_per_octave=self.BPO,
                filter_scale=FILTER_SCALE,
                device=_DEVICE, dtype=torch.float32,
            )

        # Shape
        assert C.shape[-2] == n_bins, f"Expected {n_bins} bins, got {C.shape[-2]}"
        assert C.shape[-1] > 0, "Zero frames"
        # Dtype
        assert C.dtype == torch.complex64, f"Expected complex64, got {C.dtype}"
        # Finite
        assert torch.isfinite(C).all(), "NaN or Inf in output"
        # Non-zero
        mag = C.abs()
        assert mag.sum() > 0, "All-zero output"

        # Tone peak near expected bin (600 = geometric center of 1200-bin octave)
        mean_mag = mag.mean(dim=-1)
        peak_bin = int(mean_mag.argmax().item())
        expected_bin = 600
        cent_error = abs(peak_bin - expected_bin)

        warn_summary = "; ".join(str(w.message) for w in caught)
        print(f"\n  oct={octave}  tone={tone_freq:.1f}Hz  "
              f"peak={peak_bin}  exp={expected_bin}  "
              f"err={cent_error}¢  shape={tuple(C.shape)}  "
              f"dev={C.device}  warns={len(caught)}")
        if caught:
            print(f"    warnings: {warn_summary[:200]}")

        assert cent_error < 50, (
            f"Tone peak at bin {peak_bin}, expected ~{expected_bin} (±50¢)"
        )
        # Freqs sanity
        assert freqs.shape[0] == n_bins
        assert abs(freqs[0].item() - fmin) / fmin < 0.01

        # Cleanup
        del C, freqs, mag, mean_mag
        if _DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    @pytest.mark.parametrize("octave", [-1, 4, 10])
    def test_cent_octave_sr_96k(self, octave):
        """Cent-resolution at SR=96000 for more Nyquist headroom."""
        sr = 96000
        fmin = _fmin_for_octave(octave)
        fmax = fmin * 2
        if fmax > sr / 2.0:
            pytest.skip(f"fmax={fmax:.1f} > Nyquist={sr / 2.0:.1f}")

        n_bins = self.BPO
        tone_freq = fmin * 2 ** 0.5

        t = np.arange(int(sr * self.DUR)) / sr
        y = (0.8 * np.sin(2 * np.pi * tone_freq * t)).astype(np.float32)

        C, freqs = torch_cqt.cqt(
            y, sr=sr, hop_length=HOP, fmin=fmin,
            n_bins=n_bins, bins_per_octave=self.BPO,
            filter_scale=FILTER_SCALE,
            device=_DEVICE, dtype=torch.float32,
        )

        assert C.shape[-2] == n_bins
        assert torch.isfinite(C).all()

        mean_mag = C.abs().mean(dim=-1)
        peak_bin = int(mean_mag.argmax().item())
        print(f"\n  oct={octave} sr=96k  peak={peak_bin}  "
              f"shape={tuple(C.shape)}  dev={C.device}")
        assert abs(peak_bin - 600) < 50

        del C, freqs, mean_mag
        if _DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    def test_two_adjacent_octaves(self):
        """Two octaves at cent resolution — verify stacking works."""
        sr = 44100
        oct_lo, oct_hi = 4, 5
        fmin = _fmin_for_octave(oct_lo)
        n_bins = self.BPO * 2  # 2400 bins

        tones = [fmin * 2 ** 0.5, fmin * 2 ** 1.5]
        t = np.arange(int(sr * self.DUR)) / sr
        y = np.zeros_like(t, dtype=np.float32)
        for f in tones:
            y += np.float32(0.4 * np.sin(2 * np.pi * f * t))

        C, freqs = torch_cqt.cqt(
            y, sr=sr, hop_length=HOP, fmin=fmin,
            n_bins=n_bins, bins_per_octave=self.BPO,
            filter_scale=FILTER_SCALE,
            device=_DEVICE, dtype=torch.float32,
        )

        assert C.shape[-2] == n_bins
        assert torch.isfinite(C).all()
        assert C.abs().sum() > 0

        mean_mag = C.abs().mean(dim=-1)
        peak_lo = int(mean_mag[:self.BPO].argmax().item())
        peak_hi = int(mean_mag[self.BPO:].argmax().item())
        print(f"\n  2-octave [{oct_lo},{oct_hi}]  "
              f"peak_lo={peak_lo}  peak_hi={peak_hi}  "
              f"shape={tuple(C.shape)}  dev={C.device}")

        assert abs(peak_lo - 600) < 50
        assert abs(peak_hi - 600) < 50

        del C, freqs, mean_mag
        if _DEVICE.type == "cuda":
            torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
# Guardrail tests
# ══════════════════════════════════════════════════════════════════════════

class TestGuardrails:

    def test_nyquist_warning(self):
        """fmax > Nyquist should warn but still produce output."""
        sr = 8000
        y = np.random.randn(sr * 2).astype(np.float64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            C, freqs = torch_cqt.cqt(
                y, sr=sr, n_bins=24, fmin=2000.0,
                bins_per_octave=12, device=_DEVICE,
            )

        msgs = [str(w.message) for w in caught]
        assert any("Nyquist" in m for m in msgs), f"Expected Nyquist warning: {msgs}"
        assert C.shape[-2] == 24

    def test_short_signal_warning(self):
        """Very short signal should warn about zero-padding."""
        y = np.random.randn(100).astype(np.float64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            C, freqs = torch_cqt.cqt(
                y, sr=44100, n_bins=84,
                bins_per_octave=12, hop_length=512, device=_DEVICE,
            )

        msgs = [str(w.message) for w in caught]
        assert any("zero-pad" in m or "short" in m.lower() for m in msgs), (
            f"Expected padding warning: {msgs}"
        )
        assert C.shape[-2] == 84

    def test_odd_hop_warning(self):
        """Odd hop_length should warn about no decimation."""
        y = np.random.randn(22050 * 2).astype(np.float64)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            C, freqs = torch_cqt.cqt(
                y, sr=22050, n_bins=24,
                bins_per_octave=12, hop_length=3, device=_DEVICE,
            )

        msgs = [str(w.message) for w in caught]
        assert any("odd" in m.lower() or "decimat" in m.lower() for m in msgs), (
            f"Expected hop warning: {msgs}"
        )


# ══════════════════════════════════════════════════════════════════════════
# SR scaling — same config at different sample rates vs librosa
# ══════════════════════════════════════════════════════════════════════════

class TestSampleRateScaling:

    @pytest.mark.parametrize("sr", [22050, 44100, 48000, 96000])
    def test_mid_octaves_across_sr(self, sr):
        """Octaves 2..6 should work at any standard SR."""
        fmin = _fmin_for_octave(2)
        n_bins = 12 * 5
        tones = [fmin * 2 ** (i + 0.5) for i in range(5)]
        y = _make_signal(sr, DURATION, tones)

        r = _run_comparison(sr, 12, fmin, n_bins, HOP, y, f"mid-oct sr={sr}")
        print(f"\n{r['label']}  global_corr={r['global_corr']:.6f}")
        assert r["global_corr"] > 0.98


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
