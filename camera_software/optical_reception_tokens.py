"""Fixed native reception tokens backed by authoritative rich-key storage."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import struct
from typing import Optional

from .coherent_reception import (
    OpticalCompletionCredit,
    OpticalFieldContribution,
    OpticalReceptionKey,
    OpticalSpectralMode,
)


OPTICAL_RECEPTION_TOKEN_ABI_VERSION = 1
OPTICAL_RECEPTION_TOKEN_SIZE = 96
_TOKEN_STRUCT = struct.Struct("<QQQQQQQIIIIIIII8x")
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


class OpticalReceptionTokenFlags(IntFlag):
    ACTIVE = 1 << 0
    CONTRIBUTION = 1 << 1
    COMPLETION = 1 << 2
    COHERENT = 1 << 3
    CONTINUOUS_FREQUENCY = 1 << 4


@dataclass(frozen=True, order=True)
class OpticalReceptionKeyHandle:
    handle: int
    generation: int


class OpticalReceptionKeyTable:
    """Bounded local handle table for full physical reception identities."""

    def __init__(self, *, capacity: int, generation: int = 0) -> None:
        if int(capacity) < 1:
            raise ValueError("reception key table capacity must be positive")
        if not 0 <= int(generation) <= _UINT32_MAX:
            raise ValueError("reception key generation must fit uint32")
        self.capacity = int(capacity)
        self.generation = int(generation)
        self._next_handle = 1
        self._by_key: dict[OpticalReceptionKey, int] = {}
        self._by_handle: dict[int, OpticalReceptionKey] = {}

    def intern(self, key: OpticalReceptionKey) -> OpticalReceptionKeyHandle:
        key.validate()
        existing = self._by_key.get(key)
        if existing is not None:
            return OpticalReceptionKeyHandle(existing, self.generation)
        if len(self._by_handle) >= self.capacity:
            raise BufferError("optical reception key table capacity exhausted")
        if self._next_handle > _UINT64_MAX:
            raise OverflowError("optical reception key handles exhausted uint64")
        handle = self._next_handle
        self._next_handle += 1
        self._by_key[key] = handle
        self._by_handle[handle] = key
        return OpticalReceptionKeyHandle(handle, self.generation)

    def resolve(self, value: OpticalReceptionKeyHandle) -> OpticalReceptionKey:
        if int(value.generation) != self.generation:
            raise RuntimeError("stale optical reception key generation")
        try:
            return self._by_handle[int(value.handle)]
        except KeyError as exc:
            raise KeyError("unknown optical reception key handle") from exc

    def retire(self, value: OpticalReceptionKeyHandle) -> OpticalReceptionKey:
        key = self.resolve(value)
        del self._by_handle[int(value.handle)]
        del self._by_key[key]
        return key

    def supersede(self, generation: int) -> None:
        if not self.generation < int(generation) <= _UINT32_MAX:
            raise ValueError("reception key generation must increase within uint32")
        self.generation = int(generation)
        self._next_handle = 1
        self._by_key.clear()
        self._by_handle.clear()


@dataclass(frozen=True)
class OpticalReceptionToken:
    reception_key_handle: int
    state_handle: int
    camera_causality_id: int
    coherence_id: int
    contribution_id: int
    arrival_epoch: int
    solve_epoch: int
    pool_id: int
    predecessor_product_id: int
    reception_key_generation: int
    state_generation: int
    camera_program_generation: int
    mode_id: int
    flags: OpticalReceptionTokenFlags
    reserved: int = 0

    def validate(self) -> None:
        values64 = (
            self.reception_key_handle,
            self.state_handle,
            self.camera_causality_id,
            self.coherence_id,
            self.contribution_id,
            self.arrival_epoch,
            self.solve_epoch,
        )
        values32 = (
            self.pool_id,
            self.predecessor_product_id,
            self.reception_key_generation,
            self.state_generation,
            self.camera_program_generation,
            self.mode_id,
            int(self.flags),
            self.reserved,
        )
        if any(not 0 <= int(value) <= _UINT64_MAX for value in values64):
            raise ValueError("optical reception token uint64 field overflow")
        if any(not 0 <= int(value) <= _UINT32_MAX for value in values32):
            raise ValueError("optical reception token uint32 field overflow")
        roles = self.flags & (
            OpticalReceptionTokenFlags.CONTRIBUTION
            | OpticalReceptionTokenFlags.COMPLETION
        )
        if roles not in (
            OpticalReceptionTokenFlags.CONTRIBUTION,
            OpticalReceptionTokenFlags.COMPLETION,
        ):
            raise ValueError("reception token requires exactly one process role")
        required = (
            OpticalReceptionTokenFlags.ACTIVE
            | OpticalReceptionTokenFlags.COHERENT
        )
        if self.flags & required != required:
            raise ValueError("reception token must be active and coherent")
        if self.reserved != 0:
            raise ValueError("reception token reserved field must be zero")

    def to_bytes(self) -> bytes:
        self.validate()
        payload = _TOKEN_STRUCT.pack(
            self.reception_key_handle,
            self.state_handle,
            self.camera_causality_id,
            self.coherence_id,
            self.contribution_id,
            self.arrival_epoch,
            self.solve_epoch,
            self.pool_id,
            self.predecessor_product_id,
            self.reception_key_generation,
            self.state_generation,
            self.camera_program_generation,
            self.mode_id,
            int(self.flags),
            self.reserved,
        )
        if len(payload) != OPTICAL_RECEPTION_TOKEN_SIZE:
            raise AssertionError("optical reception token ABI size drift")
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> "OpticalReceptionToken":
        if len(payload) != OPTICAL_RECEPTION_TOKEN_SIZE:
            raise ValueError("optical reception token requires exactly 96 bytes")
        values = _TOKEN_STRUCT.unpack(payload)
        result = cls(*values[:-2], OpticalReceptionTokenFlags(values[-2]), values[-1])
        result.validate()
        return result

    def key_handle(self) -> OpticalReceptionKeyHandle:
        return OpticalReceptionKeyHandle(
            self.reception_key_handle, self.reception_key_generation
        )

    @classmethod
    def from_credit(
        cls,
        credit: OpticalCompletionCredit,
        *,
        pool_id: int,
        key_table: OpticalReceptionKeyTable,
        camera_causality_id: int,
    ) -> "OpticalReceptionToken":
        credit.validate()
        return cls._from_key(
            credit.key,
            pool_id=pool_id,
            predecessor_product_id=credit.predecessor_product_id,
            contribution_id=0,
            key_table=key_table,
            camera_causality_id=camera_causality_id,
            role=OpticalReceptionTokenFlags.COMPLETION,
        )

    @classmethod
    def from_contribution(
        cls,
        contribution: OpticalFieldContribution,
        *,
        pool_id: int,
        key_table: OpticalReceptionKeyTable,
        camera_causality_id: int,
    ) -> "OpticalReceptionToken":
        contribution.validated_field()
        return cls._from_key(
            contribution.key,
            pool_id=pool_id,
            predecessor_product_id=contribution.predecessor_product_id,
            contribution_id=contribution.contribution_id,
            key_table=key_table,
            camera_causality_id=camera_causality_id,
            role=OpticalReceptionTokenFlags.CONTRIBUTION,
        )

    @classmethod
    def _from_key(
        cls,
        key: OpticalReceptionKey,
        *,
        pool_id: int,
        predecessor_product_id: int,
        contribution_id: int,
        key_table: OpticalReceptionKeyTable,
        camera_causality_id: int,
        role: OpticalReceptionTokenFlags,
    ) -> "OpticalReceptionToken":
        handle = key_table.intern(key)
        flags = (
            OpticalReceptionTokenFlags.ACTIVE
            | OpticalReceptionTokenFlags.COHERENT
            | role
        )
        if key.spectral.mode is OpticalSpectralMode.CONTINUOUS_COHORT:
            flags |= OpticalReceptionTokenFlags.CONTINUOUS_FREQUENCY
        result = cls(
            reception_key_handle=handle.handle,
            state_handle=key.state_handle,
            camera_causality_id=int(camera_causality_id),
            coherence_id=key.coherence_id,
            contribution_id=int(contribution_id),
            arrival_epoch=key.arrival_epoch,
            solve_epoch=key.solve_epoch,
            pool_id=int(pool_id),
            predecessor_product_id=int(predecessor_product_id),
            reception_key_generation=handle.generation,
            state_generation=key.state_generation,
            camera_program_generation=key.camera_program_generation,
            mode_id=key.mode_id,
            flags=flags,
        )
        result.validate()
        return result

    def resolve_key(self, key_table: OpticalReceptionKeyTable) -> OpticalReceptionKey:
        key = key_table.resolve(self.key_handle())
        repeated = (
            self.state_handle == key.state_handle
            and self.coherence_id == key.coherence_id
            and self.arrival_epoch == key.arrival_epoch
            and self.solve_epoch == key.solve_epoch
            and self.state_generation == key.state_generation
            and self.camera_program_generation == key.camera_program_generation
            and self.mode_id == key.mode_id
        )
        if not repeated:
            raise RuntimeError("reception token does not match authoritative key")
        return key


assert _TOKEN_STRUCT.size == OPTICAL_RECEPTION_TOKEN_SIZE


__all__ = [
    "OPTICAL_RECEPTION_TOKEN_ABI_VERSION",
    "OPTICAL_RECEPTION_TOKEN_SIZE",
    "OpticalReceptionTokenFlags",
    "OpticalReceptionKeyHandle",
    "OpticalReceptionKeyTable",
    "OpticalReceptionToken",
]
