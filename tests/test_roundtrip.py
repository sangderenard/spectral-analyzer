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


def _band_peak(sig: np.ndarray, sr: int, lo: float, hi: float) -> float:
    sig = np.asarray(sig, dtype=np.float64)
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(spec[mask].max())


def test_settings_hash_separates_fft_algorithms():
    from bass_analysis import AnalysisConfig, settings_token

    cfg_a = AnalysisConfig(fft_algorithm="librosa")
    cfg_b = AnalysisConfig(fft_algorithm="stft")

    assert settings_token(cfg_a) != settings_token(cfg_b)


def test_analysis_inventory_accumulates_engine_runs(tmp_path):
    from bass_viewer import _load_analysis_inventory, _record_analysis_inventory_run

    outdir = str(tmp_path)
    _record_analysis_inventory_run(
        outdir,
        engine="fft",
        run_key="hash_a",
        info={"engine": "fft", "algorithm": "librosa", "npz": "cqt_data_hash_a.npz"},
    )
    _record_analysis_inventory_run(
        outdir,
        engine="fft",
        run_key="hash_b",
        info={"engine": "fft", "algorithm": "stft", "npz": "cqt_data_hash_b.npz"},
    )
    _record_analysis_inventory_run(
        outdir,
        engine="wavelet",
        run_key="cwt_a",
        info={"engine": "wavelet", "algorithm": "cwt", "data": "wavelet/wavelet_data.npz"},
    )

    inv = _load_analysis_inventory(outdir)
    fft_runs = [row for row in inv["datasets"] if row["engine"] == "fft"]
    assert len(fft_runs) == 2
    assert {run["algorithm"] for run in fft_runs} == {"librosa", "stft"}
    assert any(row["engine"] == "wavelet" and row["algorithm"] == "cwt"
               for row in inv["datasets"])


def test_bass_analysis_writes_modern_fft_inventory(tmp_path):
    import json
    from analysis_itinerary import AnalysisInventory
    from bass_analysis import AnalysisConfig, _write_analysis_inventory

    analysis_dir = tmp_path / "demo_analysis"
    analysis_dir.mkdir()

    cfg = AnalysisConfig(fft_algorithm="stft", stft_n_fft=4096, hop_length=512)
    _write_analysis_inventory(
        str(analysis_dir),
        "abc12345",
        cfg,
        wav_path="demo.wav",
        total_duration_s=12.0,
        start_time_s=2.0,
        end_time_s=8.0,
    )

    raw = json.loads((analysis_dir / "analysis_inventory.json").read_text(encoding="utf-8"))
    inv = AnalysisInventory.from_dict(raw)
    fft_rows = [ds for ds in inv.datasets if ds.engine == "fft"]

    assert len(fft_rows) == 1
    assert fft_rows[0].dataset_key == "fft:abc12345"
    assert fft_rows[0].algorithm == "stft"
    assert fft_rows[0].settings["stft_n_fft"] == 4096
    assert fft_rows[0].time_range.start_sec == 2.0
    assert fft_rows[0].time_range.end_sec == 8.0


# =====================================================================
#  CQT round-trip  (ViewportSynthPlayer)
# =====================================================================

class TestCQTRoundTrip:
    """Forward → inverse CQT via the production OLA engine."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from bass_viewer import ViewportSynthPlayer
        self.VSP = ViewportSynthPlayer

    def _roundtrip(self, sig, sr=22050, bpo=48, hop=1):
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

    def test_compute_and_save_persists_intermediate_sidecars_before_hilbert_failure(
        self,
        tmp_path,
        monkeypatch,
    ):
        sr = 8000
        sig = _tone(sr, 0.25, 440.0).astype(np.float64)
        fb = self.FB(sr, "Linkwitz-Riley 4")
        bands = [self.BandDef(fmin=200.0, fmax=1000.0)]
        bands[0].label = bands[0].auto_label(sr)
        expected_mag = np.linspace(0.1, 0.4, 4, dtype=np.float64)
        expected_phase = np.linspace(0.0, 0.3, 4, dtype=np.float64)

        def _boom(this, hop=None, hilbert_progress=None,
                  band_result_cb=None, resume_state=None):
            if band_result_cb is not None:
                band_result_cb(0, expected_mag, expected_phase, 1)
            raise RuntimeError("hilbert boom")

        monkeypatch.setattr(self.FB, "compute_envelopes", _boom)

        with pytest.raises(RuntimeError, match="hilbert boom"):
            fb.compute_and_save(
                sig, bands, str(tmp_path), sr,
                envelope_hop=1,
                intermediate_precision="64-bit",
                save_precision="32-bit",
            )

        meta, band_meta = self.FB.load_meta(str(tmp_path))
        assert meta["band_audio_precision"] == "64-bit"
        assert meta["intermediate_precision"] == "64-bit"
        assert meta["save_precision"] == "32-bit"
        assert os.path.isfile(os.path.join(tmp_path, "filterbank",
                                           band_meta[0]["file"]))

        checkpoint = self.FB.load_envelope_checkpoint(str(tmp_path))
        assert checkpoint is not None
        assert checkpoint["completed"] == [True]
        np.testing.assert_allclose(checkpoint["mags"][0], expected_mag)
        np.testing.assert_allclose(checkpoint["phases"][0], expected_phase)

    def test_compute_and_save_finalizes_requested_save_precision(self, tmp_path):
        sr = 8000
        sig = _tone(sr, 0.25, 440.0).astype(np.float64)
        fb = self.FB(sr, "Linkwitz-Riley 4")
        bands = [self.BandDef(fmin=200.0, fmax=1000.0)]
        bands[0].label = bands[0].auto_label(sr)

        fb.compute_and_save(
            sig, bands, str(tmp_path), sr,
            envelope_hop=1,
            intermediate_precision="64-bit",
            save_precision="32-bit",
        )

        meta, _ = self.FB.load_meta(str(tmp_path))
        assert meta["band_audio_precision"] == "32-bit"
        assert meta["intermediate_precision"] == "64-bit"
        assert meta["save_precision"] == "32-bit"
        assert not os.path.exists(os.path.join(
            tmp_path, "filterbank", "fb_envelopes.partial.npz"))

        cached = self.FB.load_envelopes(str(tmp_path))
        assert cached is not None
        _, _, hops = cached
        assert hops == [1]

    def test_compute_and_save_resumes_checkpoint_without_refiltering(
        self,
        tmp_path,
        monkeypatch,
    ):
        import torch

        sr = 8000
        sig = _tone(sr, 0.25, 440.0).astype(np.float64)
        bands = [self.BandDef(fmin=200.0, fmax=1000.0)]
        bands[0].label = bands[0].auto_label(sr)
        orig_compute_envelopes = self.FB.compute_envelopes

        def _boom(this, hop=None, hilbert_progress=None,
                  band_result_cb=None, resume_state=None):
            if band_result_cb is not None:
                band_result_cb(
                    0,
                    np.linspace(0.1, 0.4, 4, dtype=np.float64),
                    np.linspace(0.0, 0.3, 4, dtype=np.float64),
                    1,
                )
            raise RuntimeError("hilbert boom")

        monkeypatch.setattr(self.FB, "compute_envelopes", _boom)

        with pytest.raises(RuntimeError, match="hilbert boom"):
            self.FB(sr, "Linkwitz-Riley 4").compute_and_save(
                sig, bands, str(tmp_path), sr,
                envelope_hop=1,
                intermediate_precision="64-bit",
                save_precision="32-bit",
            )

        monkeypatch.setattr(self.FB, "compute_envelopes", orig_compute_envelopes)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        def _refiltered(*args, **kwargs):
            raise AssertionError("unexpected refilter")

        monkeypatch.setattr(self.FB, "_filter_one_band", _refiltered)

        self.FB(sr, "Linkwitz-Riley 4").compute_and_save(
            sig, bands, str(tmp_path), sr,
            envelope_hop=1,
            intermediate_precision="64-bit",
            save_precision="32-bit",
        )

        cached = self.FB.load_envelopes(str(tmp_path))
        assert cached is not None


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

    @pytest.mark.usefixtures("_import_synth")
    def test_cwt_wavelet_uses_working_hop_metadata(self):
        n_frames = 900
        W = np.zeros((8, n_frames), dtype=np.complex128)
        freqs = np.geomspace(10.0, 1.0, 8).astype(np.float64)
        n_samples = 28800
        out = self.reconstruct_wavelet(
            sr=480,
            t0=0.0,
            t1=n_samples / 480.0,
            wv_meta={
                "type": "cwt",
                "wavelet": "morlet",
                "hop_length": 1,
                "working_hop_length": 32,
                "n_samples": n_samples,
            },
            W_complex=W,
            wv_freqs=freqs,
        )

        assert len(out) == n_frames * 32

    @pytest.mark.usefixtures("_import_synth")
    def test_cwt_wavelet_respects_visible_scale_band(self):
        from bass_viewer import ViewportBoundaryConfig

        n_frames = 96
        W = np.zeros((4, n_frames), dtype=np.complex128)
        W[0, :] = 10.0 + 0.0j   # highest-frequency source row
        W[-1, :] = 1.0 + 0.0j   # lowest-frequency source row
        freqs = np.geomspace(8.0, 1.0, 4).astype(np.float64)
        cfg = ViewportBoundaryConfig()

        low_band = self.reconstruct_wavelet(
            sr=96,
            t0=0.0,
            t1=1.0,
            wv_meta={
                "type": "cwt",
                "wavelet": "morlet",
                "hop_length": 1,
                "n_samples": n_frames,
            },
            W_complex=W,
            wv_freqs=freqs,
            view_x0=0.0,
            view_x1=float(n_frames),
            view_y0=0.0,
            view_y1=1.0,
            viewport_boundary=cfg,
        )
        high_band = self.reconstruct_wavelet(
            sr=96,
            t0=0.0,
            t1=1.0,
            wv_meta={
                "type": "cwt",
                "wavelet": "morlet",
                "hop_length": 1,
                "n_samples": n_frames,
            },
            W_complex=W,
            wv_freqs=freqs,
            view_x0=0.0,
            view_x1=float(n_frames),
            view_y0=3.0,
            view_y1=4.0,
            viewport_boundary=cfg,
        )

        lo_level = float(np.mean(np.abs(low_band)))
        hi_level = float(np.mean(np.abs(high_band)))
        assert lo_level > 0.0
        assert hi_level > lo_level * 2.0


class TestFilterbankScalerHelpers:

    def test_heterodyne_shift_moves_band_up_by_scale(self):
        from bass_viewer import _heterodyne_shift_signal

        sr = 2000
        sig = _tone(sr, 1.0, 20.0, gain=0.8)
        shifted = _heterodyne_shift_signal(
            sig, sr, freq_lo=18.0, freq_hi=22.0, scale=8.0)
        peak = _dominant_freq(shifted, sr)
        assert 145.0 < peak < 175.0

    def test_soft_fft_bandpass_prefers_requested_viewport_band(self):
        from bass_viewer import ViewportBoundaryConfig, _soft_fft_bandpass

        sr = 2000
        sig = _multitone(sr, 1.0, [10.0, 180.0], gain=0.7)
        out, lo_eff, hi_eff = _soft_fft_bandpass(
            sig, sr,
            base_lo=5.0,
            base_hi=20.0,
            cfg=ViewportBoundaryConfig(),
            filter_type="Linkwitz-Riley 4",
        )

        assert 4.0 <= lo_eff <= 6.0
        assert 19.0 <= hi_eff <= 21.0
        lo_peak = _band_peak(out, sr, 8.0, 12.0)
        hi_peak = _band_peak(out, sr, 170.0, 190.0)
        assert lo_peak > hi_peak * 8.0


class TestFFTScalerHelpers:

    def test_fft_scaler_stft_params_follow_low_frequency(self):
        from bass_viewer import _fft_scaler_stft_params

        n_fft_lo, hop_lo = _fft_scaler_stft_params(
            sr=48000, freq_lo=20.0, hop_hint=512,
            bins_per_octave=24, filter_scale=1.0, signal_len=48000,
        )
        n_fft_hi, hop_hi = _fft_scaler_stft_params(
            sr=48000, freq_lo=200.0, hop_hint=512,
            bins_per_octave=24, filter_scale=1.0, signal_len=48000,
        )

        assert n_fft_lo > n_fft_hi
        assert hop_lo >= hop_hi
        assert n_fft_lo & (n_fft_lo - 1) == 0
        assert n_fft_hi & (n_fft_hi - 1) == 0

    def test_phase_vocoder_stft_changes_time_and_pitch_independently(self):
        from bass_viewer import _phase_vocoder_stft

        sr = 4000
        sig = _tone(sr, 1.0, 110.0, gain=0.8)
        out = _phase_vocoder_stft(
            sig, sr,
            n_fft=1024, hop=256, window="hann",
            time_scale=2.0, pitch_scale=4.0,
        )

        assert abs(len(out) - int(round(len(sig) / 2.0))) <= 80
        peak = _dominant_freq(out, sr)
        assert 380.0 < peak < 500.0


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
