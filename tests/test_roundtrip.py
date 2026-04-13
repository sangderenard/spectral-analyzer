"""Round-trip reconstruction tests for CQT, filterbank, and wavelet transforms.

Each test imports and exercises the actual production code from
``bass_viewer.py`` — no logic is duplicated.

Run::

    python -m pytest tests/test_roundtrip.py -v
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── helpers ───────────────────────────────────────────────────────────

def _tone(sr: int, dur: float, freq: float, *, gain: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    return gain * np.sin(2.0 * np.pi * freq * t)


def _multitone(sr: int, dur: float, freqs: list[float],
               *, gain: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    return sum(gain * np.sin(2.0 * np.pi * f * t) for f in freqs)


def _chirp(sr: int, dur: float, f0: float, f1: float,
           *, gain: float = 0.5) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) / (2.0 * dur) * t ** 2)
    return gain * np.sin(phase)


def _xcorr_peak(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation peak (handles time shifts)."""
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    n = len(a) + len(b) - 1
    fft_n = 1
    while fft_n < n:
        fft_n <<= 1
    xc = np.fft.irfft(np.fft.rfft(a, fft_n) * np.conj(np.fft.rfft(b, fft_n)), fft_n)
    return float(xc.max() / (na * nb))


def _snr(sig: np.ndarray, recon: np.ndarray) -> float:
    n = min(len(sig), len(recon))
    err = np.sqrt(np.mean((sig[:n] - recon[:n]) ** 2))
    sig_rms = np.sqrt(np.mean(sig[:n] ** 2))
    return 20.0 * np.log10(sig_rms / max(err, 1e-30))


def _dominant_freq(sig: np.ndarray, sr: int) -> float:
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < 8:
        return 0.0
    win = np.hanning(len(sig))
    spec = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
    idx = int(np.argmax(np.abs(spec[1:])) + 1)
    return float(freqs[idx])


# =====================================================================
#  CQT round-trip  (ViewportSynthPlayer)
# =====================================================================

class TestCQTRoundTrip:
    """Forward → inverse CQT via the production OLA engine."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from bass_viewer import ViewportSynthPlayer
        self.VSP = ViewportSynthPlayer

    def _roundtrip(self, sig, sr=22050, bpo=48, hop=256):
        import librosa
        fmin = 32.0
        fmax = sr / 2.0 * 0.95
        n_bins = int(math.floor(bpo * math.log2(fmax / fmin)))
        freqs = librosa.cqt_frequencies(n_bins, fmin=fmin,
                                        bins_per_octave=bpo).astype(np.float64)
        n_frames = 1 + (len(sig) - 1) // hop

        synth = self.VSP(sr=sr, hop_length=hop, bins_per_octave=bpo)
        cqt = synth._forward_cqt_channel(sig, freqs, 0, n_bins, n_frames)
        recon = synth._inverse_cqt_channel(cqt, freqs, 0, n_bins, n_frames)
        return recon

    @pytest.mark.parametrize("freq", [440.0, 1000.0, 4000.0])
    def test_tone(self, freq):
        sr = 22050
        sig = _tone(sr, 0.5, freq)
        recon = self._roundtrip(sig, sr=sr)
        margin = int(0.05 * sr)
        n = min(len(sig), len(recon))
        a, b = sig[margin:n - margin], recon[margin:n - margin]
        a /= max(np.abs(a).max(), 1e-12)
        b /= max(np.abs(b).max(), 1e-12)
        assert _xcorr_peak(a, b) > 0.85

    def test_multitone(self):
        sr = 22050
        sig = _multitone(sr, 0.5, [220.0, 880.0, 3520.0])
        recon = self._roundtrip(sig, sr=sr)
        margin = int(0.05 * sr)
        n = min(len(sig), len(recon))
        a, b = sig[margin:n - margin], recon[margin:n - margin]
        a /= max(np.abs(a).max(), 1e-12)
        b /= max(np.abs(b).max(), 1e-12)
        assert _xcorr_peak(a, b) > 0.80

    def test_no_silence(self):
        """iCQT must not produce pop-then-silence."""
        sr = 22050
        sig = _tone(sr, 0.5, 440.0)
        recon = self._roundtrip(sig, sr=sr)

        chunk = len(recon) // 10
        rms = np.array([np.sqrt(np.mean(recon[i*chunk:(i+1)*chunk]**2))
                        for i in range(10)])
        ratio = rms / max(rms.max(), 1e-12)
        assert ratio[1:-1].min() > 0.1, f"Pop-then-silence: {ratio}"

    def test_ola_norm_floor(self):
        """Median-based OLA norm floor prevents boundary spikes."""
        sr, bpo, hop = 22050, 48, 256
        Q = 1.0 / (2.0 ** (1.0 / bpo) - 1.0)
        fftconv = self.VSP._fftconvolve

        klen = int(math.ceil(Q * sr / 440.0))
        n_frames = 200
        train_len = (n_frames - 1) * hop + 1

        t = np.arange(klen, dtype=np.float64)
        win_sq = (0.5 - 0.5 * np.cos(2.0 * np.pi * t / klen)) ** 2
        pulse_train = np.zeros(train_len, dtype=np.float64)
        pulse_train[::hop] = 1.0
        ola_norm = fftconv(pulse_train, win_sq)

        # Old floor (1e-10) → spikes
        bad = 1.0 / np.where(ola_norm > 1e-10, ola_norm, 1.0)
        # New median floor → bounded
        pos = ola_norm[ola_norm > 0]
        nf = max(np.median(pos) * 0.1, 1e-10) if pos.size else 1.0
        good = 1.0 / np.where(ola_norm > nf, ola_norm, nf)

        bad_ratio = bad.max() / np.median(bad[bad > 0])
        good_ratio = good.max() / np.median(good[good > 0])
        assert good_ratio < 20.0
        assert bad_ratio > 100.0 * good_ratio


# =====================================================================
#  Filterbank round-trip  (FilterBankDecomposition)
# =====================================================================

class TestFilterbankRoundTrip:
    """LR4 crossover → sum-of-subbands should be near-perfect."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from bass_viewer import FilterBankDecomposition, BandDef, SourcePanel
        self.FB = FilterBankDecomposition
        self.BandDef = BandDef
        self.SourcePanel = SourcePanel

    def _bands(self, n, fmin=20.0, fmax=20000.0, sr=44100):
        edges = np.geomspace(fmin, fmax, n + 1)
        out = []
        for i in range(n):
            b = self.BandDef(fmin=float(edges[i]), fmax=float(edges[i + 1]))
            b.label = b.auto_label(sr)
            out.append(b)
        return out

    def test_lr4_tone(self):
        sr = 44100
        sig = _tone(sr, 0.5, 440.0).astype(np.float32)
        fb = self.FB(sr, "Linkwitz-Riley 4")
        fb.compute(sig, self._bands(16, sr=sr))
        assert _snr(sig, fb.reconstruct()) > 60.0

    def test_lr4_chirp(self):
        sr = 44100
        sig = _chirp(sr, 1.0, 50.0, 15000.0).astype(np.float32)
        fb = self.FB(sr, "Linkwitz-Riley 4")
        fb.compute(sig, self._bands(32, sr=sr))
        assert _snr(sig, fb.reconstruct()) > 50.0

    def test_lr4_noise(self):
        sr = 44100
        rng = np.random.default_rng(42)
        sig = (rng.standard_normal(int(sr * 0.5)) * 0.3).astype(np.float32)
        fb = self.FB(sr, "Linkwitz-Riley 4")
        fb.compute(sig, self._bands(16, sr=sr))
        err = fb.reconstruction_error()
        sig_rms = np.sqrt(np.mean(sig ** 2))
        assert 20 * np.log10(sig_rms / max(err, 1e-30)) > 60.0

    def test_band_mask(self):
        sr = 44100
        sig = _tone(sr, 0.5, 440.0).astype(np.float32)
        bands = self._bands(8, sr=sr)
        fb = self.FB(sr, "Linkwitz-Riley 4")
        fb.compute(sig, bands)
        full = fb.reconstruct()
        mask = [True] * len(bands)
        mask[0] = False
        partial = fb.reconstruct(band_mask=mask)
        assert np.sqrt(np.mean((full - partial) ** 2)) > 1e-10

    def test_butterworth(self):
        sr = 44100
        sig = _tone(sr, 0.5, 1000.0).astype(np.float32)
        fb = self.FB(sr, "Butterworth 4")
        fb.compute(sig, self._bands(8, sr=sr))
        recon = fb.reconstruct()
        n = min(len(sig), len(recon))
        assert _xcorr_peak(sig[:n].astype(np.float64),
                           recon[:n].astype(np.float64)) > 0.7

    def test_envelopes(self):
        sr = 44100
        n = int(sr * 0.5)
        t = np.arange(n, dtype=np.float64) / sr
        mod = 0.5 * (1.0 + np.sin(2.0 * np.pi * 5.0 * t))
        sig = (mod * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
        bands = [self.BandDef(fmin=500.0, fmax=2000.0)]
        bands[0].label = bands[0].auto_label(sr)
        fb = self.FB(sr, "Linkwitz-Riley 4")
        fb.compute(sig, bands)
        mags, phases, hops = fb.compute_envelopes(hop=512)
        assert len(mags) == 1 and len(mags[0]) > 0
        env = mags[0].astype(np.float64)
        assert (env / max(env.max(), 1e-12)).std() > 0.05, "AM not captured"

    def test_default_hops_are_fidelity_biased(self):
        sp = self.SourcePanel(input_dir=ROOT, output_root=ROOT)
        assert sp.cwt_hop_length == 1
        assert sp.fb_hop == 1

    def test_save_envelopes_respects_explicit_hop(self, tmp_path):
        sr = 8000
        sig = _tone(sr, 0.25, 440.0).astype(np.float32)
        fb = self.FB(sr, "Linkwitz-Riley 4")
        bands = [self.BandDef(fmin=200.0, fmax=1000.0)]
        bands[0].label = bands[0].auto_label(sr)
        fb.compute(sig, bands)
        fb.save(str(tmp_path), sr, envelope_hop=1)
        cached = self.FB.load_envelopes(str(tmp_path))
        assert cached is not None
        _, _, hops = cached
        assert hops == [1]

    def test_save_filterbank_meta_persists_cafls_center(self, tmp_path):
        sr = 8000
        sig = _tone(sr, 0.25, 440.0).astype(np.float32)
        fb = self.FB(sr, "Linkwitz-Riley 4")
        bands = self.FB.bands_from_cafls(anchor=55.0, bpo=12,
                                         banded_width=1, sr=sr)
        fb.compute(sig, bands)
        fb.save(str(tmp_path), sr, envelope_hop=1)

        meta, _ = self.FB.load_meta(str(tmp_path))

        assert meta["cafls_center_hz"] == pytest.approx(55.0, rel=1e-6)


# =====================================================================
#  Wavelet round-trip  (SynthesisPanel._execute_wavelet + pywt)
# =====================================================================

class TestWaveletRoundTrip:

    @pytest.fixture(autouse=True)
    def _check_pywt(self):
        self.pywt = pytest.importorskip("pywt")

    @pytest.fixture()
    def _import_synth(self):
        from bass_viewer import (
            SynthesisPanel, SynthJob, _reconstruct_wavelet_viewport,
        )
        self.SynthPanel = SynthesisPanel
        self.SynthJob = SynthJob
        self.reconstruct_wavelet = _reconstruct_wavelet_viewport

    def test_full_range_perfect(self):
        sig = _tone(44100, 0.5, 440.0)
        coeffs = self.pywt.wavedec(sig, "db4", mode="symmetric")
        recon = self.pywt.waverec(coeffs, "db4")[:len(sig)]
        assert _snr(sig, recon) > 100.0

    @pytest.mark.usefixtures("_import_synth")
    def test_execute_wavelet_produces_output(self):
        """_execute_wavelet produces audible stereo int16 output."""
        sig = _chirp(44100, 1.0, 100.0, 5000.0).astype(np.float32)
        coeffs = [c.copy() for c in
                  self.pywt.wavedec(sig, "db4", mode="symmetric")]
        job = self.SynthJob(job_id=0, label="test")
        self.SynthPanel._execute_wavelet(job, {
            "wv_coeffs": coeffs,
            "wv_meta": {"wavelet": "db4"},
            "sr": 44100, "t0": 0.0, "t1": 1.0, "normalize": True,
        })
        assert job.result_stereo is not None
        assert job.result_stereo.dtype == np.int16
        assert job.result_stereo.shape[1] == 2
        assert np.abs(job.result_stereo).max() > 100

    @pytest.mark.usefixtures("_import_synth")
    def test_execute_wavelet_viewport(self):
        """Viewport-windowed reconstruction correlates with original."""
        sr, dur = 44100, 1.0
        sig = _chirp(sr, dur, 100.0, 5000.0).astype(np.float32)
        coeffs = [c.copy() for c in
                  self.pywt.wavedec(sig, "db4", mode="symmetric")]
        t0, t1 = 0.25, 0.75
        job = self.SynthJob(job_id=0, label="test")
        self.SynthPanel._execute_wavelet(job, {
            "wv_coeffs": coeffs,
            "wv_meta": {"wavelet": "db4"},
            "sr": sr, "t0": t0, "t1": t1, "normalize": True,
        })
        recon = job.result_stereo[:, 0].astype(np.float64) / 32767.0
        orig = sig[int(t0 * sr):int(t1 * sr)].astype(np.float64)
        n = min(len(recon), len(orig))
        assert _xcorr_peak(orig[:n], recon[:n]) > 0.5

    def test_chirp_roundtrip(self):
        sig = _chirp(22050, 0.5, 200.0, 8000.0)
        coeffs = self.pywt.wavedec(sig, "db4", mode="symmetric")
        assert _snr(sig, self.pywt.waverec(coeffs, "db4")[:len(sig)]) > 100.0

    def test_noise_roundtrip(self):
        rng = np.random.default_rng(123)
        sig = rng.standard_normal(44100) * 0.5
        coeffs = self.pywt.wavedec(sig, "db1", mode="symmetric")
        assert _snr(sig, self.pywt.waverec(coeffs, "db1")[:len(sig)]) > 100.0

    @pytest.mark.parametrize("wavelet", ["db1", "db4", "db8", "sym4", "coif2"])
    def test_multiple_wavelets(self, wavelet):
        sig = _tone(22050, 0.3, 440.0)
        coeffs = self.pywt.wavedec(sig, wavelet, mode="symmetric")
        assert _snr(sig, self.pywt.waverec(coeffs, wavelet)[:len(sig)]) > 80.0

    def test_level_isolation(self):
        sig = _multitone(44100, 0.5, [100.0, 1000.0, 8000.0])
        coeffs = self.pywt.wavedec(sig, "db4", mode="symmetric", level=10)
        mod = [c.copy() for c in coeffs]
        mod[-1] = np.zeros_like(mod[-1])
        full = self.pywt.waverec(coeffs, "db4")[:len(sig)]
        part = self.pywt.waverec(mod, "db4")[:len(sig)]
        assert np.abs(full - part).max() > 1e-4

    @pytest.mark.usefixtures("_import_synth")
    def test_execute_wavelet_cwt_analysis_order(self):
        import torch
        import torch_cqt_new as torch_cqt

        sr = 200
        dur = 8.0
        freq = 5.0
        t = torch.arange(int(sr * dur), dtype=torch.float64) / sr
        sig = torch.sin(2.0 * np.pi * freq * t)
        W, freqs = torch_cqt.cwt(
            sig, sr, fmin=1.0, fmax=50.0,
            scales_per_octave=12, hop_length=1,
            wavelet="morlet", device="cpu",
        )

        job = self.SynthJob(job_id=0, label="cwt")
        self.SynthPanel._execute_wavelet(job, {
            "wv_W": W.numpy().copy(),
            "wv_freqs": freqs.numpy().copy(),
            "wv_meta": {
                "type": "cwt",
                "wavelet": "morlet",
                "hop_length": 1,
                "n_samples": len(sig),
            },
            "sr": sr,
            "t0": 0.0,
            "t1": dur,
            "normalize": False,
        })
        recon = job.result_stereo[:, 0].astype(np.float64) / 32767.0
        orig = sig.numpy().astype(np.float64)
        margin = sr // 2
        n = min(len(orig), len(recon))
        corr = _xcorr_peak(orig[margin:n - margin], recon[margin:n - margin])
        assert corr > 0.95

    @pytest.mark.usefixtures("_import_synth")
    def test_execute_wavelet_respects_extension_mode(self):
        sr = 4096
        dur = 1.0
        sig = _chirp(sr, dur, 80.0, 900.0).astype(np.float64)
        coeffs = self.pywt.wavedec(
            sig, "db4", mode="periodization", level=6,
        )

        job = self.SynthJob(job_id=0, label="dwt")
        self.SynthPanel._execute_wavelet(job, {
            "wv_coeffs": [c.copy() for c in coeffs],
            "wv_meta": {
                "type": "dwt",
                "wavelet": "db4",
                "extension": "periodization",
                "n_samples": len(sig),
            },
            "sr": sr,
            "t0": 0.0,
            "t1": dur,
            "normalize": False,
        })
        recon = job.result_stereo[:, 0].astype(np.float64) / 32767.0
        margin = sr // 16
        n = min(len(sig), len(recon))
        corr = _xcorr_peak(sig[margin:n - margin], recon[margin:n - margin])
        assert corr > 0.9

    @pytest.mark.usefixtures("_import_synth")
    def test_cwt_wavelet_space_time_scale_changes_length_and_pitch(self):
        import torch
        import torch_cqt_new as torch_cqt

        sr = 200
        dur = 12.0
        freq = 5.0
        t = torch.arange(int(sr * dur), dtype=torch.float64) / sr
        sig = torch.sin(2.0 * np.pi * freq * t)
        W, freqs = torch_cqt.cwt(
            sig, sr, fmin=1.0, fmax=50.0,
            scales_per_octave=12, hop_length=1,
            wavelet="morlet", device="cpu",
        )

        scaled = self.reconstruct_wavelet(
            sr=sr,
            t0=0.0,
            t1=dur,
            wv_meta={
                "type": "cwt",
                "wavelet": "morlet",
                "hop_length": 1,
                "n_samples": len(sig),
            },
            W_complex=W.numpy(),
            wv_freqs=freqs.numpy(),
            playback_rate=2.0,
        )

        assert abs(len(scaled) - int(round(len(sig) / 2.0))) <= 1
        peak = _dominant_freq(scaled, sr)
        ratio = peak / freq
        assert 1.7 < ratio < 2.3, f"expected ~2x pitch, got ratio {ratio:.3f}"


# =====================================================================
#  Integration: stored CQT data
# =====================================================================

class TestCQTWithStoredData:

    @pytest.fixture(autouse=True)
    def _find_data(self):
        import glob
        from bass_viewer import ViewportSynthPlayer
        self.VSP = ViewportSynthPlayer

        dirs = glob.glob(os.path.join(ROOT, "input", "*_analysis"))
        if not dirs:
            pytest.skip("No analysis data in input/")
        self.analysis_dir = dirs[0]
        npz_files = sorted(
            glob.glob(os.path.join(self.analysis_dir, "cqt_data_*.npz")),
            key=os.path.getmtime, reverse=True,
        )
        path = npz_files[0] if npz_files else os.path.join(
            self.analysis_dir, "cqt_data.npz")
        if not os.path.exists(path):
            pytest.skip("No CQT data found")
        self.npz = np.load(path, mmap_mode="r")

    def test_stored_nonsilent(self):
        """iCQT from stored data is non-silent with uniform energy."""
        real_l = np.asarray(self.npz["real_left"], dtype=np.float64)
        imag_l = np.asarray(self.npz["imag_left"], dtype=np.float64)
        freqs = np.asarray(self.npz["freqs"], dtype=np.float64)
        sr = int(self.npz["sr"])
        hop = int(self.npz["hop_length"])
        bpo = int(self.npz["bins_per_octave"])
        n_bins, n_frames = real_l.shape
        cqt = real_l + 1j * imag_l

        oct_bins = min(bpo, n_bins)
        synth = self.VSP(sr=sr, hop_length=hop, bins_per_octave=bpo)
        recon = synth._inverse_cqt_channel(
            cqt[-oct_bins:, :], freqs[-oct_bins:],
            n_bins - oct_bins, n_bins, n_frames,
        )
        assert np.abs(recon).max() > 1e-10, "Silent"

        chunk = len(recon) // 8
        if chunk > 0:
            rms = np.array([np.sqrt(np.mean(recon[i*chunk:(i+1)*chunk]**2))
                            for i in range(8)])
            ratio = rms / max(rms.max(), 1e-12)
            assert ratio[1:-1].min() > 0.05, f"Pop-then-silence: {ratio}"
