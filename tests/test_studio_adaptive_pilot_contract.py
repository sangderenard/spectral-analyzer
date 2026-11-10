"""Independent-pilot benchmark: rare bright path, not a scene convergence claim.

Protects the statistical design boundary before adaptive sampling is wired.
Final samples never include pilot observations that chose their budget.
"""
import numpy as np


def test_data_dependent_stopping_is_biased_but_independent_pilot_is_not():
    rng = np.random.default_rng(20261109)
    trials = 200_000
    probability = 0.1
    samples = (rng.random((trials, 10)) < probability).astype(np.float64)
    # Bad policy: dark first sample stops; bright first sample continues to10.
    reused = np.where(samples[:, 0] == 0, 0, samples.mean(axis=1))
    expected_biased = probability * (1 + 9 * probability) / 10
    assert abs(reused.mean() - expected_biased) < 0.002
    assert abs(reused.mean() - probability) > 0.07
    # Independent fixed-size pilot chooses a fixed budget for fresh final work.
    pilot = rng.random(trials) < probability
    final = np.where(pilot, samples.mean(axis=1), samples[:, 0])
    assert abs(final.mean() - probability) < 0.003


def test_dark_pilot_cannot_certify_absence_of_rare_caustics():
    # Eight independent misses are still overwhelmingly likely for a1% event.
    miss_probability = (1 - 0.01) ** 8
    assert miss_probability > 0.92
