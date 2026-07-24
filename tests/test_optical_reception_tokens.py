from __future__ import annotations

import numpy as np
import pytest

from camera_software.coherent_reception import (
    OpticalCompletionCredit,
    OpticalFieldContribution,
    OpticalReceptionKey,
    OpticalSpectralIdentity,
    OpticalSpectralMode,
)
from camera_software.complex_optical_operators import TransverseBasis
from camera_software.optical_reception_tokens import (
    OPTICAL_RECEPTION_TOKEN_SIZE,
    OpticalReceptionKeyTable,
    OpticalReceptionToken,
    OpticalReceptionTokenFlags,
)


def _key(*, generation=7, coherence=31):
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
        spectral=OpticalSpectralIdentity(
            OpticalSpectralMode.CONTINUOUS_COHORT, (5.0e14,)
        ),
        coherence_id=coherence,
        arrival_epoch=2,
        mode_id=6,
        solve_epoch=19,
    )


def _basis():
    return TransverseBasis(
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
    )


def test_completion_token_is_exact_96_byte_round_trip():
    key = _key()
    table = OpticalReceptionKeyTable(capacity=4, generation=12)
    token = OpticalReceptionToken.from_credit(
        OpticalCompletionCredit(key, 17),
        pool_id=9,
        key_table=table,
        camera_causality_id=44,
    )

    payload = token.to_bytes()
    restored = OpticalReceptionToken.from_bytes(payload)

    assert len(payload) == OPTICAL_RECEPTION_TOKEN_SIZE == 96
    assert restored == token
    assert restored.flags & OpticalReceptionTokenFlags.COMPLETION
    assert restored.flags & OpticalReceptionTokenFlags.CONTINUOUS_FREQUENCY
    assert restored.resolve_key(table) == key


def test_contribution_and_completion_share_key_but_not_role_or_identity():
    key = _key()
    table = OpticalReceptionKeyTable(capacity=4)
    contribution = OpticalFieldContribution(
        key=key,
        predecessor_product_id=11,
        contribution_id=83,
        field=np.asarray([[1.0 + 2.0j, 0.0]], np.complex128),
        basis=_basis(),
    )
    field_token = OpticalReceptionToken.from_contribution(
        contribution,
        pool_id=9,
        key_table=table,
        camera_causality_id=44,
    )
    completion_token = OpticalReceptionToken.from_credit(
        OpticalCompletionCredit(key, 11),
        pool_id=9,
        key_table=table,
        camera_causality_id=44,
    )

    assert field_token.reception_key_handle == completion_token.reception_key_handle
    assert field_token.contribution_id == 83
    assert completion_token.contribution_id == 0
    assert field_token.flags & OpticalReceptionTokenFlags.CONTRIBUTION
    assert completion_token.flags & OpticalReceptionTokenFlags.COMPLETION


def test_key_table_rejects_stale_handles_and_token_key_disagreement():
    key = _key()
    table = OpticalReceptionKeyTable(capacity=2, generation=1)
    token = OpticalReceptionToken.from_credit(
        OpticalCompletionCredit(key, 17),
        pool_id=9,
        key_table=table,
        camera_causality_id=44,
    )
    table.supersede(2)

    with pytest.raises(RuntimeError, match="stale"):
        token.resolve_key(table)

    current = OpticalReceptionToken.from_credit(
        OpticalCompletionCredit(key, 17),
        pool_id=9,
        key_table=table,
        camera_causality_id=44,
    )
    corrupted = OpticalReceptionToken(
        **{
            **current.__dict__,
            "coherence_id": current.coherence_id + 1,
        }
    )
    with pytest.raises(RuntimeError, match="authoritative key"):
        corrupted.resolve_key(table)


def test_key_table_is_bounded_reuses_keys_and_retires_explicitly():
    table = OpticalReceptionKeyTable(capacity=1, generation=3)
    first = table.intern(_key())
    assert table.intern(_key()) == first
    with pytest.raises(BufferError, match="capacity"):
        table.intern(_key(coherence=99))

    assert table.retire(first) == _key()
    second = table.intern(_key(coherence=99))
    assert second.handle != first.handle


@pytest.mark.parametrize(
    "flags",
    [
        OpticalReceptionTokenFlags.ACTIVE
        | OpticalReceptionTokenFlags.COHERENT,
        OpticalReceptionTokenFlags.ACTIVE
        | OpticalReceptionTokenFlags.COHERENT
        | OpticalReceptionTokenFlags.CONTRIBUTION
        | OpticalReceptionTokenFlags.COMPLETION,
    ],
)
def test_token_rejects_missing_or_ambiguous_process_role(flags):
    token = OpticalReceptionToken(
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, flags,
    )
    with pytest.raises(ValueError, match="exactly one"):
        token.validate()
