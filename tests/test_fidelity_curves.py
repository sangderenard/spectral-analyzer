import numpy as np
import pytest

from torch_cqt_new import fidelity_curve, fidelity_curve_dwt, fidelity_curve_fb


def test_cqt_fidelity_uses_actual_filter_support():
    sr = 48_000
    hop = 64
    fmin = 30.0
    bpo = 24
    filter_scale = 2.0

    fc = fidelity_curve(
        sr=sr,
        hop_length=hop,
        fmin=fmin,
        fmax=120.0,
        bins_per_octave=bpo,
        filter_scale=filter_scale,
    )

    alpha = (2.0 ** (2.0 / bpo) - 1.0) / (2.0 ** (2.0 / bpo) + 1.0)
    expected_support = (filter_scale / alpha) / fmin

    assert fc["freqs"][0] == pytest.approx(fmin)
    assert fc["frame_spacing"][0] == pytest.approx(hop / sr)
    assert fc["filter_support"][0] == pytest.approx(expected_support)
    assert fc["delta_t"][0] == pytest.approx(expected_support)
    assert fc["delta_t"][0] > fc["frame_spacing"][0] * 100.0


@pytest.mark.parametrize(
    ("filter_type", "expected_order"),
    [
        ("Linkwitz-Riley 4", 4),
        ("Butterworth 4", 4),
        ("Butterworth 8", 8),
    ],
)
def test_fb_fidelity_accepts_solver_filter_names(filter_type, expected_order):
    fc = fidelity_curve_fb(
        sr=48_000,
        bands_per_octave=6,
        fmin=60.0,
        fmax=6_000.0,
        hop_length=512,
        filter_type=filter_type,
    )

    assert fc["freqs"].size > 0
    assert np.all(fc["filter_order"] == expected_order)
    assert np.all(fc["delta_t"] >= fc["frame_spacing"])
    assert np.all(fc["delta_t"] >= (1.0 / (2.0 * np.maximum(fc["delta_f"], 1e-30))))


def test_dwt_fidelity_has_own_dyadic_band_curve():
    pytest.importorskip("pywt")

    level = 4
    sr = 44_100
    fc = fidelity_curve_dwt(sr=sr, wavelet="db4", level=level, extension="symmetric")

    assert fc["freqs"].shape == (level + 1,)
    assert np.array_equal(fc["band_name"], np.array(["A4", "D4", "D3", "D2", "D1"], dtype=object))
    assert bool(fc["is_approximation"][0])
    assert not bool(fc["is_approximation"][-1])
    assert np.all(np.diff(fc["freqs"]) > 0)
    assert np.all(np.diff(fc["delta_t"]) <= 0)
    assert fc["band_hi"][0] == pytest.approx((sr / 2.0) / (2.0 ** level))
    assert fc["coefficient_spacing"][0] == pytest.approx((2.0 ** level) / sr)
    assert fc["filter_support"][0] > fc["coefficient_spacing"][0]
