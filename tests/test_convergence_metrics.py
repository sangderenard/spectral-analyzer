from types import SimpleNamespace

import pytest

from camera_software.convergence_metrics import (
    artifact_convergence_metric,
    artifact_convergence_velocity,
    convergence_metric,
    convergence_velocity_per_pass,
)


def test_convergence_velocity_is_metric_delta_per_refinement_pass():
    assert convergence_velocity_per_pass(
        0.70, 128, 0.50, 64
    ) == pytest.approx(3.125e-3)
    assert convergence_velocity_per_pass(
        0.50, 64, 0.70, 128
    ) == 0.0


def test_artifact_uses_retained_metric_and_velocity_evidence():
    artifact = SimpleNamespace(
        samples=128,
        metadata={
            "refinement_pass": 128,
            "atlas_quality": {
                "convergence_metric": 0.8125,
                "convergence_velocity_per_pass": 4.25e-4,
            },
        },
    )

    assert artifact_convergence_metric(artifact) == 0.8125
    assert artifact_convergence_velocity(artifact) == 4.25e-4


def test_first_checkpoint_uses_evidence_depth_instead_of_fixed_45_percent():
    shallow = convergence_metric({
        "glyph_exposure_coverage": 1.0,
        "glyph_radiance_coverage": 1.0,
        "glyph_mean_weight": 0.02,
    })
    deeper = convergence_metric({
        "glyph_exposure_coverage": 1.0,
        "glyph_radiance_coverage": 1.0,
        "glyph_mean_weight": 0.20,
    })

    assert 0.0 < shallow < deeper < 0.15
    assert shallow != pytest.approx(0.45)


def test_stability_evidence_changes_convergence_by_subject():
    common = {
        "glyph_exposure_coverage": 1.0,
        "glyph_radiance_coverage": 1.0,
        "glyph_mean_weight": 1.0,
    }
    noisy = convergence_metric({
        **common,
        "relative_rmse": 0.20,
        "p95_relative_delta": 0.30,
    })
    stable = convergence_metric({
        **common,
        "relative_rmse": 0.002,
        "p95_relative_delta": 0.004,
    })

    assert noisy < stable < 1.0


def test_unversioned_retained_metric_is_recomputed_when_raw_evidence_exists():
    migrated = convergence_metric({
        "convergence_metric": 0.45,
        "glyph_exposure_coverage": 1.0,
        "glyph_radiance_coverage": 1.0,
        "glyph_mean_weight": 0.05,
        "relative_rmse": None,
        "p95_relative_delta": None,
    })

    assert 0.0 < migrated < 0.15
