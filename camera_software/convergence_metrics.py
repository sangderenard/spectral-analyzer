"""Stable display metrics derived from retained atlas quality evidence."""
from __future__ import annotations

import math
from typing import Any, Mapping

CONVERGENCE_METRIC_VERSION = 2


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0


def convergence_metric(
    quality: Mapping[str, Any] | None,
    *,
    completion_basis: str = "",
    refinement_pass: int = 0,
) -> float:
    """Convert coverage/stability evidence to the UI's [0, 1] metric."""

    evidence = dict(quality or {})
    explicit = evidence.get("convergence_metric")
    has_raw_evidence = any(
        key in evidence
        for key in (
            "glyph_exposure_coverage",
            "glyph_radiance_coverage",
            "glyph_mean_weight",
            "relative_rmse",
            "p95_relative_delta",
        )
    )
    metric_version = int(evidence.get("convergence_metric_version", 0) or 0)
    if explicit is not None and (
        metric_version >= CONVERGENCE_METRIC_VERSION or not has_raw_evidence
    ):
        return clamp01(explicit)
    if evidence.get("converged") or completion_basis:
        return 1.0
    if not evidence:
        current_pass = max(0, int(refinement_pass))
        return (
            current_pass / float(current_pass + 8)
            if current_pass else 0.0
        )
    try:
        from .render_assets import (
            INK_ATLAS_CONVERGENCE_EXPOSURE_COVERAGE,
            INK_ATLAS_CONVERGENCE_HOLD,
            INK_ATLAS_CONVERGENCE_P95_DELTA,
            INK_ATLAS_CONVERGENCE_RELATIVE_RMSE,
            MIN_INK_GLYPH_RADIANCE_COVERAGE,
        )

        coverage_target = float(INK_ATLAS_CONVERGENCE_EXPOSURE_COVERAGE)
        radiance_target = float(MIN_INK_GLYPH_RADIANCE_COVERAGE)
        hold_target = max(1, int(INK_ATLAS_CONVERGENCE_HOLD))
        rmse_target = float(INK_ATLAS_CONVERGENCE_RELATIVE_RMSE)
        delta_target = float(INK_ATLAS_CONVERGENCE_P95_DELTA)
    except ImportError:
        coverage_target, radiance_target, hold_target = 0.95, 0.95, 2
        rmse_target, delta_target = 0.01, 0.02
    coverage = min(
        clamp01(
            float(evidence.get("glyph_exposure_coverage", 0.0))
            / coverage_target
        ),
        clamp01(
            float(evidence.get("glyph_radiance_coverage", 0.0))
            / radiance_target
        ),
    )
    try:
        mean_weight = max(0.0, float(evidence.get("glyph_mean_weight", 0.0)))
    except (TypeError, ValueError):
        mean_weight = 0.0
    if not math.isfinite(mean_weight):
        mean_weight = 0.0
    # A binary "every glyph pixel was touched" result is availability, not
    # convergence. Weight it by accumulated evidence so a single shallow pass
    # cannot pin every subject to the same large score.
    evidence_depth = 1.0 - math.exp(-mean_weight)
    coverage_confidence = coverage * evidence_depth
    rmse = evidence.get("relative_rmse")
    delta = evidence.get("p95_relative_delta")
    stability = 0.0
    if rmse is not None and delta is not None:
        try:
            rmse_value = max(0.0, float(rmse))
            delta_value = max(0.0, float(delta))
        except (TypeError, ValueError):
            rmse_value = math.inf
            delta_value = math.inf
        # Smooth confidence curves retain useful motion on both sides of the
        # target instead of snapping all sufficiently stable subjects to 1.
        stability = min(
            clamp01(rmse_target / (rmse_target + rmse_value)),
            clamp01(delta_target / (delta_target + delta_value)),
        )
    hold = clamp01(float(evidence.get("stable_hold", 0)) / hold_target)
    score = clamp01(
        0.15 * coverage_confidence + 0.75 * stability + 0.10 * hold
    )
    return min(0.99, score)


def artifact_convergence_metric(artifact: Any) -> float:
    if artifact is None:
        return 0.0
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    return convergence_metric(
        metadata.get("atlas_quality", {}),
        completion_basis=str(metadata.get("completion_basis", "")),
        refinement_pass=int(
            metadata.get(
                "refinement_pass", getattr(artifact, "samples", 0)
            ) or 0
        ),
    )


def artifact_convergence_velocity(artifact: Any) -> float:
    if artifact is None:
        return 0.0
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    quality = dict(metadata.get("atlas_quality", {}) or {})
    try:
        velocity = float(quality.get(
            "convergence_velocity_per_pass",
            metadata.get("convergence_velocity_per_pass", 0.0),
        ))
    except (TypeError, ValueError):
        return 0.0
    return velocity if math.isfinite(velocity) else 0.0


def convergence_velocity_per_pass(
    current_metric: float,
    current_pass: int,
    previous_metric: float,
    previous_pass: int,
) -> float:
    """Signed convergence change per completed refinement pass."""

    pass_delta = int(current_pass) - int(previous_pass)
    if pass_delta <= 0:
        return 0.0
    velocity = (
        float(current_metric) - float(previous_metric)
    ) / float(pass_delta)
    return velocity if math.isfinite(velocity) else 0.0


__all__ = [
    "CONVERGENCE_METRIC_VERSION",
    "clamp01",
    "convergence_metric",
    "artifact_convergence_metric",
    "artifact_convergence_velocity",
    "convergence_velocity_per_pass",
]
