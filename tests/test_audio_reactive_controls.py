import numpy as np

from audio_reactive_controls import FftfreeSpectrum


def test_fftfree_spectrum_finds_a_bin_centered_tone():
    size = 2048
    expected_bin = 43
    phase = 2.0 * np.pi * expected_bin * np.arange(size) / size
    samples = np.sin(phase).astype(np.float32)
    spectrum = FftfreeSpectrum(size)
    try:
        magnitude = spectrum.magnitude(samples)
    finally:
        spectrum.close()
    assert int(np.argmax(magnitude)) == expected_bin
    assert magnitude[expected_bin] > 1000.0 * np.partition(magnitude, -2)[-2]
