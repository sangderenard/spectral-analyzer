import numpy as np

from camera_software.text_clarity import MonteCarloClarityDiscriminator


def test_rejects_black_and_noisy_layers_then_accepts_stable_signal():
    discriminator = MonteCarloClarityDiscriminator(
        min_layers=4, max_drift=0.02, max_noise_rse_p90=0.3
    )
    assert not discriminator.update(np.zeros((8, 12, 3), np.float32)).accepted
    rng = np.random.default_rng(3)
    cumulative = np.zeros((8, 12, 3), np.float64)
    report = None
    target = np.zeros_like(cumulative)
    target[2:6, 3:9] = 1.0
    for index in range(24):
        layer = np.maximum(target + rng.normal(0.0, 0.5 / (index + 1), target.shape), 0.0)
        cumulative += layer
        report = discriminator.update(cumulative)
    assert report is not None
    assert report.has_signal
    assert report.noise_rse_p90 < 0.3
    assert report.accepted
