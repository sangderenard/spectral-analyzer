import numpy as np

from camera_software.exposure_priority import (
    LocalityAwarePriorityPolicy,
    PriorityPolicyConfig,
    TileEvidence,
)
from camera_software.progressive_exposure import SensorRegion


def tile(x, y, *, samples=4, variance=0.0, ambiguity=0.0, requested=0.0):
    return TileEvidence(
        SensorRegion(x, y, 16, 16),
        sample_count=samples,
        variance=variance,
        ambiguity=ambiguity,
        requested_priority=requested,
    )


def test_sample_count_does_not_override_authoritative_top_k_work_value():
    policy = LocalityAwarePriorityPolicy()
    ranked = policy.rank([
        tile(0, 0, samples=20, variance=100.0, ambiguity=1.0),
        tile(16, 0, samples=0),
    ])
    assert ranked[0].evidence.region.x == 0


def test_firm_requested_region_overrides_locality_and_other_scores():
    policy = LocalityAwarePriorityPolicy()
    ranked = policy.rank([
        tile(0, 0, variance=10.0, ambiguity=1.0),
        tile(512, 512, requested=0.9),
    ])
    assert ranked[0].evidence.region.x == 512


def test_locality_breaks_near_equal_priority_ties_in_morton_order():
    policy = LocalityAwarePriorityPolicy(PriorityPolicyConfig(priority_quantum=1.0))
    ranked = policy.rank([
        tile(32, 32), tile(0, 16), tile(16, 0), tile(0, 0),
    ])
    assert [(x.evidence.region.x, x.evidence.region.y) for x in ranked] == [
        (0, 0), (16, 0), (0, 16), (32, 32),
    ]


def test_learned_model_predicts_value_without_touching_evidence_statistics():
    class Model:
        def score(self, evidence):
            return np.asarray([0.1, 3.0], dtype=np.float32)

    policy = LocalityAwarePriorityPolicy(
        PriorityPolicyConfig(uncertainty_weight=0.0, ambiguity_weight=0.0),
        learned_model=Model(),
    )
    ranked = policy.rank([tile(0, 0), tile(16, 0)])
    assert ranked[0].evidence.region.x == 16
    assert ranked[0].learned_priority == 3.0


def test_invalid_learned_scores_are_rejected():
    class BadModel:
        def score(self, evidence):
            return np.asarray([np.nan] * len(evidence))

    policy = LocalityAwarePriorityPolicy(learned_model=BadModel())
    try:
        policy.rank([tile(0, 0)])
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("invalid learned priorities must fail closed")
