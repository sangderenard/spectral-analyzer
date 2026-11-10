"""Release gates for source -> field -> ray -> field energy transfer.

These exercise the existing native CPU pipeline, not a reference solver.
The separated-port case remains an explicit, strict expected failure until
the production continuation-state consumer replaces source-mode reseeding.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_wave_t4_pipeline import _tracer, _wait


def _two_arena_output_powers(*, linked: bool, source_weight: float):
    from camera_software.complex_optical_operators import (
        ComplexOpticalOperator,
        ComplexOperatorStateBlock,
        ComplexSourceMode,
        TransverseBasis,
        install_source_mode_block,
    )

    tracer = _tracer(np.asarray([550.0e-9]))
    centers = (-8.0e-6, 8.0e-6) if linked else (-100.0e-6, 100.0e-6)
    contexts = [
        tracer.add_scale_context(
            np.asarray([0.0, 0.0, z]),
            64.0e-6, 1, 2.0e-6, 8, 1.0, 0.0, 1,
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        for z in centers
    ]
    if linked:
        tracer.add_wave_context_link(*contexts)

    block = ComplexOperatorStateBlock()
    block.add_basis(TransverseBasis.from_direction((0.0, 0.0, 1.0)))
    block.add_operator(ComplexOpticalOperator())
    block.add_source_mode(ComplexSourceMode(
        jones=np.asarray([1.0, 0.0]),
        power_weight=source_weight,
        coherence_id=77,
        basis_id=0,
        operator_id=0,
    ))
    install_source_mode_block(tracer, np.asarray([1], np.uint64), block)
    tracer.ensure_pipeline(max_children=1, min_amplitude=1.0e-12)
    # Negative x/y keep the ray clear of the helper's ordinary triangle.
    tracer.submit_rays(
        np.asarray([[-10.0e-6, -10.0e-6, -1.0e-3]]),
        np.asarray([[0.0, 0.0, 1.0]]),
        np.asarray([[1.0 + 0.0j]]),
        tags=np.asarray([1], np.uint64),
        max_bounces=3,
        min_amplitude=1.0e-12,
    )
    _wait(tracer)
    arenas = tracer.wave_arena_stats()
    assert [arena["generation"] for arena in arenas] == [1, 1]
    assert all(arena["field_power"] > 0.0 for arena in arenas)
    # Compare matching unit-weight controls so ordinary ray propagation and
    # the numerical exterior cancel; this tests only source-energy accounting.
    return np.asarray([arena["field_power"] for arena in arenas])


@pytest.mark.parametrize("linked", [
    True,
    pytest.param(False, marks=pytest.mark.xfail(
        strict=True,
        reason=(
            "T4 scalar continuation does not consume its exit Jones state; "
            "the next separated arena reapplies the source power weight"
        ),
    )),
])
@pytest.mark.parametrize("source_weight", [0.25, 0.5])
def test_source_energy_is_applied_once_across_two_wave_regions(
    linked: bool, source_weight: float,
):
    unit = _two_arena_output_powers(linked=linked, source_weight=1.0)
    weighted = _two_arena_output_powers(
        linked=linked, source_weight=source_weight,
    )
    np.testing.assert_allclose(
        weighted / unit,
        [source_weight, source_weight],
        rtol=3.0e-5,
        atol=1.0e-8,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The separated T4 entry reuses the source Jones vector instead of "
        "the polarization carried by the previous field-to-ray exit"
    ),
)
def test_jones_interface_output_survives_an_ordinary_ray_gap():
    from camera_software.complex_optical_operators import (
        ComplexOpticalOperator,
        ComplexOperatorStateBlock,
        ComplexSourceMode,
        TransverseBasis,
        install_source_mode_block,
    )

    tracer = _tracer(np.asarray([550.0e-9]))
    contexts = [
        tracer.add_scale_context(
            np.asarray([0.0, 0.0, z]),
            64.0e-6, 1, 2.0e-6, 8, 1.0, 0.0, 1,
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        for z in (-108.0e-6, -92.0e-6, 100.0e-6)
    ]
    swap = np.asarray([[[0.0, 1.0], [1.0, 0.0]]], np.complex64)
    tracer.add_wave_context_interface_link(
        contexts[0], contexts[1], 0, swap.real, swap.imag,
    )
    block = ComplexOperatorStateBlock()
    block.add_basis(TransverseBasis.from_direction((0.0, 0.0, 1.0)))
    block.add_operator(ComplexOpticalOperator())
    block.add_source_mode(ComplexSourceMode(
        jones=np.asarray([1.0, 0.0]),
        power_weight=1.0,
        coherence_id=77,
        basis_id=0,
        operator_id=0,
    ))
    install_source_mode_block(tracer, np.asarray([1], np.uint64), block)
    tracer.ensure_pipeline(max_children=1, min_amplitude=1.0e-12)
    tracer.submit_rays(
        np.asarray([[-10.0e-6, -10.0e-6, -1.0e-3]]),
        np.asarray([[0.0, 0.0, 1.0]]),
        np.asarray([[1.0 + 0.0j]]),
        tags=np.asarray([1], np.uint64),
        max_bounces=3,
        min_amplitude=1.0e-12,
    )
    _wait(tracer)
    arenas = tracer.wave_arena_stats()
    assert [arena["generation"] for arena in arenas] == [1, 1, 1]
    p_fractions = []
    for arena_id in (1, 2):
        component_powers = []
        for component in (0, 1):
            field = tracer.wave_arena_field_snapshot(
                arena_id, 0, component, 0,
            )
            values = np.asarray(field["re"]) + 1j * np.asarray(field["im"])
            component_powers.append(float(np.sum(np.abs(values) ** 2)))
        assert sum(component_powers) > 0.0
        p_fractions.append(component_powers[1] / sum(component_powers))
    np.testing.assert_allclose(p_fractions, [1.0, 1.0], atol=3.0e-5)
