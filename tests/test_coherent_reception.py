from __future__ import annotations

import math
import random

import numpy as np
import pytest

from camera_software.coherent_reception import (
    CoherentReceptionPool,
    CompiledOpticalReception,
    OpticalCompletionCredit,
    OpticalFieldContribution,
    OpticalPredecessorDeclaration,
    OpticalReceptionKey,
    OpticalSpectralIdentity,
    OpticalSpectralMode,
)
from camera_software.complex_optical_operators import (
    JonesOperator,
    TransverseBasis,
)


def _basis(angle: float = 0.0) -> TransverseBasis:
    return TransverseBasis(
        s=np.asarray((math.cos(angle), math.sin(angle), 0.0)),
        p=np.asarray((-math.sin(angle), math.cos(angle), 0.0)),
        k=np.asarray((0.0, 0.0, 1.0)),
    )


def _descriptor(lanes: int = 1, *, capacity: int = 8) -> CompiledOpticalReception:
    return CompiledOpticalReception(
        pool_id=0,
        destination_node="recombiner",
        destination_port="incident",
        predecessors=(
            OpticalPredecessorDeclaration(0, 11, "path-a"),
            OpticalPredecessorDeclaration(1, 17, "path-b"),
        ),
        lane_count=lanes,
        representation="complex-ray",
        capacity=capacity,
    )


def _key(
    frequencies: tuple[float, ...] = (5.0e14,),
    *,
    coherence: int = 3,
    generation: int = 7,
    mode: OpticalSpectralMode = OpticalSpectralMode.FIXED_BANDS,
) -> OpticalReceptionKey:
    return OpticalReceptionKey(
        state_handle=5,
        state_generation=generation,
        camera_program_handle=23,
        camera_program_generation=4,
        exposure_frame_id=8,
        exposure_slice_id=2,
        destination_node="recombiner",
        destination_port="incident",
        scattering_product="recombined",
        spectral=OpticalSpectralIdentity(mode, frequencies),
        coherence_id=coherence,
        arrival_epoch=2,
        mode_id=0,
        solve_epoch=19,
    )


def _contribution(
    key: OpticalReceptionKey,
    predecessor: int,
    field: np.ndarray,
    *,
    contribution_id: int = 0,
    basis: TransverseBasis | None = None,
    grid_id: int = 0,
) -> OpticalFieldContribution:
    return OpticalFieldContribution(
        key,
        predecessor,
        contribution_id,
        np.asarray(field, np.complex128),
        basis or _basis(),
        grid_id,
    )


def _close(
    pool: CoherentReceptionPool,
    key: OpticalReceptionKey,
    order: tuple[int, int] = (11, 17),
):
    result = pool.complete(OpticalCompletionCredit(key, order[0]))
    assert result is None
    result = pool.complete(OpticalCompletionCredit(key, order[1]))
    assert result is not None
    return result


@pytest.mark.parametrize(
    ("phase", "expected_power"),
    [(0.0, 4.0), (math.pi, 0.0), (math.pi / 2.0, 2.0)],
)
def test_coherent_reception_interference(phase: float, expected_power: float):
    key = _key()
    pool = CoherentReceptionPool(
        _descriptor(), state_generation=7, destination_basis=_basis()
    )
    pool.accept(_contribution(key, 11, [[1.0, 0.0]]))
    pool.accept(_contribution(key, 17, [[np.exp(1j * phase), 0.0]]))

    closed = _close(pool, key)

    assert closed.report.reduced_field_power == pytest.approx(
        expected_power, abs=1.0e-14
    )
    assert closed.report.accepted_contributions == 2
    assert closed.report.completed_predecessors == 2


def test_reduction_is_stable_under_shuffled_arrival_and_completion():
    key = _key()
    events = [
        _contribution(key, 11, [[1.0 + 2.0j, 0.0]], contribution_id=2),
        _contribution(key, 11, [[-0.25 + 0.5j, 0.0]], contribution_id=1),
        _contribution(key, 17, [[0.75 - 1.0j, 0.0]], contribution_id=4),
    ]
    outputs = []
    for seed in range(12):
        pool = CoherentReceptionPool(
            _descriptor(), state_generation=7, destination_basis=_basis()
        )
        shuffled = list(events)
        random.Random(seed).shuffle(shuffled)
        for contribution in shuffled:
            pool.accept(contribution)
        completion_order = (11, 17) if seed % 2 else (17, 11)
        outputs.append(_close(pool, key, completion_order).output_field.copy())

    for output in outputs[1:]:
        np.testing.assert_array_equal(output, outputs[0])


def test_basis_conversion_precedes_stable_complex_sum_and_operator():
    key = _key()
    destination = _basis()
    rotated = _basis(math.pi / 2.0)
    pool = CoherentReceptionPool(
        _descriptor(),
        state_generation=7,
        destination_basis=destination,
        operator=JonesOperator.diagonal(0.5, 2.0),
    )
    pool.accept(_contribution(key, 11, [[1.0, 0.0]], basis=destination))
    # A unit s component in this rotated basis is a unit +p physical field.
    pool.accept(_contribution(key, 17, [[1.0, 0.0]], basis=rotated))

    closed = _close(pool, key)

    np.testing.assert_allclose(closed.reduced_field, [[1.0, 1.0]], atol=1e-15)
    np.testing.assert_allclose(closed.output_field, [[0.5, 2.0]], atol=1e-15)


def test_distinct_coherence_ids_close_separately_and_add_only_as_intensity():
    first_key = _key(coherence=101)
    second_key = _key(coherence=202)
    pool = CoherentReceptionPool(
        _descriptor(), state_generation=7, destination_basis=_basis()
    )
    closed = []
    for key in (first_key, second_key):
        pool.accept(_contribution(key, 11, [[1.0, 0.0]]))
        pool.accept(_contribution(key, 17, [[-1.0, 0.0]]))
        if key is second_key:
            # Make the second independent mode nonzero.
            pool.accept(
                _contribution(
                    key, 17, [[2.0, 0.0]], contribution_id=1
                )
            )
        closed.append(_close(pool, key))

    intensity = CoherentReceptionPool.incoherent_intensity(closed)
    np.testing.assert_allclose(intensity, [[4.0, 0.0]], atol=1e-15)


def test_pool_waits_for_every_credit_and_rejects_lifecycle_errors():
    key = _key()
    pool = CoherentReceptionPool(
        _descriptor(), state_generation=7, destination_basis=_basis()
    )
    contribution = _contribution(key, 11, [[1.0, 0.0]])
    pool.accept(contribution)
    assert pool.complete(OpticalCompletionCredit(key, 11)) is None
    assert pool.unresolved_predecessors(key) == (17,)
    with pytest.raises(RuntimeError, match="after predecessor completion"):
        pool.accept(
            _contribution(key, 11, [[2.0, 0.0]], contribution_id=1)
        )
    with pytest.raises(RuntimeError, match="duplicate.*completion"):
        pool.complete(OpticalCompletionCredit(key, 11))
    closed = pool.complete(OpticalCompletionCredit(key, 17))
    assert closed is not None
    with pytest.raises(RuntimeError, match="already closed"):
        pool.complete(OpticalCompletionCredit(key, 17))


def test_pool_rejects_stale_frequency_grid_duplicate_and_capacity_errors():
    pool = CoherentReceptionPool(
        _descriptor(capacity=1), state_generation=7, destination_basis=_basis()
    )
    stale = _key(generation=6)
    with pytest.raises(RuntimeError, match="stale"):
        pool.accept(_contribution(stale, 11, [[1.0, 0.0]]))

    key = _key()
    contribution = _contribution(key, 11, [[1.0, 0.0]])
    pool.accept(contribution)
    with pytest.raises(RuntimeError, match="duplicate"):
        pool.accept(contribution)
    with pytest.raises(ValueError, match="grid"):
        other_pool = CoherentReceptionPool(
            _descriptor(), state_generation=7, destination_basis=_basis()
        )
        other_pool.accept(
            _contribution(key, 11, [[1.0, 0.0]], grid_id=9)
        )
    with pytest.raises(BufferError, match="capacity"):
        pool.accept(
            _contribution(
                _key(coherence=99), 11, [[1.0, 0.0]]
            )
        )

    different_frequency = _key((5.1e14,))
    assert different_frequency != key


@pytest.mark.parametrize("lanes", [1, 3, 4, 8, 16, 32])
@pytest.mark.parametrize(
    "mode",
    [OpticalSpectralMode.FIXED_BANDS, OpticalSpectralMode.CONTINUOUS_COHORT],
)
def test_exact_lane_specializations_and_spectral_semantics_survive(
    lanes: int, mode: OpticalSpectralMode
):
    frequencies = tuple(np.linspace(4.0e14, 7.0e14, lanes))
    key = _key(frequencies, mode=mode)
    pool = CoherentReceptionPool(
        _descriptor(lanes), state_generation=7, destination_basis=_basis()
    )
    field = np.ones((lanes, 2), np.complex128)
    pool.accept(_contribution(key, 11, field))
    pool.accept(_contribution(key, 17, field))

    closed = _close(pool, key)

    assert closed.output_field.shape == (lanes, 2)
    assert key.contract()["spectral"]["mode"] == mode.value
    assert key.contract()["spectral"]["lane_count"] == lanes


def test_reference_records_round_trip_with_deterministic_equality():
    descriptor = _descriptor(3)
    key = _key((4.0e14, 5.0e14, 6.0e14))
    contribution = _contribution(
        key,
        11,
        np.asarray([
            [1.0 + 2.0j, 3.0 - 4.0j],
            [5.0 + 6.0j, 7.0 - 8.0j],
            [9.0 + 10.0j, 11.0 - 12.0j],
        ]),
        basis=_basis(0.25),
    )
    credit = OpticalCompletionCredit(key, 11)

    assert CompiledOpticalReception.from_contract(
        descriptor.contract()
    ) == descriptor
    assert OpticalReceptionKey.from_contract(key.contract()) == key
    assert OpticalFieldContribution.from_contract(
        contribution.contract()
    ) == contribution
    assert OpticalCompletionCredit.from_contract(credit.contract()) == credit


def test_pool_detaches_converted_field_and_basis_from_caller_mutation():
    key = _key()
    pool = CoherentReceptionPool(
        _descriptor(), state_generation=7, destination_basis=_basis()
    )
    first = np.asarray([[1.0, 0.0]], dtype=np.complex64)
    second = np.asarray([[1.0, 0.0]], dtype=np.complex64)
    source_basis = _basis()
    pool.accept(_contribution(key, 11, first, basis=source_basis))
    pool.accept(_contribution(key, 17, second, basis=source_basis))

    first[0, 0] = 10.0
    source_basis.s[:] = (0.0, 1.0, 0.0)

    closed = _close(pool, key)
    np.testing.assert_array_equal(closed.reduced_field, [[2.0, 0.0]])
    assert closed.report.reduced_field_power == pytest.approx(4.0)
