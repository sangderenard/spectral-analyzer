from __future__ import annotations

import os
from pathlib import Path

import pytest

from camera_software.nodus_runtime_bridge import (
    NODUS_OPTICAL_RECEPTION_TYPE_ID,
    NodusFifoQualificationBridge,
)
from camera_software.optical_reception_tokens import (
    OPTICAL_RECEPTION_TOKEN_SIZE,
    OpticalReceptionToken,
    OpticalReceptionTokenFlags,
)


def _library_path() -> Path:
    configured = os.environ.get("NODUS_RUNTIME_LIBRARY")
    if configured:
        return Path(configured)
    return Path(r"C:\dev\Powershell\nodus\build\Release\nodus_runtime.dll")


def _legacy_library_path() -> Path:
    return Path(r"C:\dev\Powershell\nodus\build\Release\canvas_tables.dll")


def _payload(contribution_id: int) -> bytes:
    return OpticalReceptionToken(
        reception_key_handle=1,
        state_handle=2,
        camera_causality_id=3,
        coherence_id=4,
        contribution_id=contribution_id,
        arrival_epoch=5,
        solve_epoch=6,
        pool_id=7,
        predecessor_product_id=11,
        reception_key_generation=8,
        state_generation=9,
        camera_program_generation=10,
        mode_id=12,
        flags=(
            OpticalReceptionTokenFlags.ACTIVE
            | OpticalReceptionTokenFlags.COHERENT
            | OpticalReceptionTokenFlags.CONTRIBUTION
        ),
    ).to_bytes()


@pytest.mark.skipif(not _library_path().exists(), reason="Nodus DLL unavailable")
def test_reception_token_round_trip_and_slowest_reader_backpressure():
    with NodusFifoQualificationBridge(
        _library_path(),
        element_size=OPTICAL_RECEPTION_TOKEN_SIZE,
        capacity=4,
        type_id=NODUS_OPTICAL_RECEPTION_TYPE_ID,
    ) as fifo:
        assert fifo.uses_extracted_runtime
        fifo.subscribe(101)
        fifo.subscribe(202)
        payloads = tuple(_payload(index) for index in range(5))

        for payload in payloads[:4]:
            assert fifo.publish(77, payload)
        assert fifo.unread(101) == 4
        assert fifo.unread(202) == 4

        consumed_fast = tuple(fifo.consume(101) for _ in range(4))
        assert consumed_fast == payloads[:4]
        assert fifo.unread(101) == 0
        assert fifo.unread(202) == 4
        assert not fifo.publish(77, payloads[4])

        assert fifo.consume(202) == payloads[0]
        assert fifo.publish(77, payloads[4])
        remaining_slow = tuple(fifo.consume(202) for _ in range(4))
        assert remaining_slow == payloads[1:5]
        assert fifo.consume(101) == payloads[4]
        assert fifo.quiescent


@pytest.mark.skipif(not _library_path().exists(), reason="Nodus DLL unavailable")
def test_reception_fifo_transaction_snapshot_restores_reader_frontiers():
    with NodusFifoQualificationBridge(
        _library_path(),
        element_size=OPTICAL_RECEPTION_TOKEN_SIZE,
        capacity=4,
        type_id=NODUS_OPTICAL_RECEPTION_TYPE_ID,
    ) as fifo:
        assert fifo.uses_extracted_runtime
        fifo.subscribe(101)
        fifo.subscribe(202)
        payloads = tuple(_payload(index) for index in range(3))
        for payload in payloads:
            assert fifo.publish(77, payload)
        checkpoint = fifo.copy_shallow()

        assert fifo.consume(101) == payloads[0]
        assert fifo.consume(101) == payloads[1]
        assert fifo.consume(202) == payloads[0]
        fifo.restore(checkpoint)

        assert fifo.unread(101) == 3
        assert fifo.unread(202) == 3
        assert tuple(fifo.consume(101) for _ in range(3)) == payloads
        assert tuple(fifo.consume(202) for _ in range(3)) == payloads
        assert fifo.quiescent


@pytest.mark.skipif(
    not _library_path().exists() or not _legacy_library_path().exists(),
    reason="both extracted and legacy Nodus DLLs are required",
)
def test_extracted_runtime_matches_legacy_fifo_transaction_bytes():
    payloads = tuple(_payload(index) for index in range(3))
    snapshots = []
    consumed = []
    for library in (_library_path(), _legacy_library_path()):
        with NodusFifoQualificationBridge(
            library,
            element_size=OPTICAL_RECEPTION_TOKEN_SIZE,
            capacity=4,
            type_id=NODUS_OPTICAL_RECEPTION_TYPE_ID,
        ) as fifo:
            fifo.subscribe(101)
            fifo.subscribe(202)
            for payload in payloads:
                assert fifo.publish(77, payload)
            snapshots.append(fifo.copy_shallow())
            consumed.append(
                (
                    tuple(fifo.consume(101) for _ in payloads),
                    tuple(fifo.consume(202) for _ in payloads),
                )
            )
            assert fifo.quiescent

    assert snapshots[0] == snapshots[1]
    assert consumed[0] == consumed[1] == (payloads, payloads)
