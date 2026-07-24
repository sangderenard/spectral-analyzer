"""Deterministic CPU reference for acyclic coherent optical reception.

The reception pool stages fields by a cold predecessor order, waits for one
completion credit from every predecessor, converts Jones coordinates into the
declared destination basis, and invokes the destination operator once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Iterable

import numpy as np

from .complex_optical_operators import (
    JonesOperator,
    TransverseBasis,
    basis_change,
)


SUPPORTED_RECEPTION_LANE_COUNTS = (1, 3, 4, 8, 16, 32)


class OpticalSpectralMode(str, Enum):
    FIXED_BANDS = "fixed-bands"
    CONTINUOUS_COHORT = "continuous-cohort"


class OpticalReceptionPolicy(str, Enum):
    DETERMINISTIC_COHERENT = "deterministic-coherent"


@dataclass(frozen=True, order=True)
class OpticalSpectralIdentity:
    """Exact lane identity; lane width never substitutes for spectral mode."""

    mode: OpticalSpectralMode
    frequencies_hz: tuple[float, ...]

    def validate(self) -> None:
        if len(self.frequencies_hz) not in SUPPORTED_RECEPTION_LANE_COUNTS:
            raise ValueError("reception spectral identity has unsupported lane width")
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in self.frequencies_hz
        ):
            raise ValueError("reception frequencies must be finite and positive")

    @property
    def lane_count(self) -> int:
        return len(self.frequencies_hz)

    def contract(self) -> dict[str, Any]:
        self.validate()
        return {
            "mode": self.mode.value,
            "frequencies_hz": list(self.frequencies_hz),
            "lane_count": self.lane_count,
        }

    @classmethod
    def from_contract(cls, value: dict[str, Any]) -> "OpticalSpectralIdentity":
        result = cls(
            OpticalSpectralMode(str(value["mode"])),
            tuple(float(item) for item in value["frequencies_hz"]),
        )
        result.validate()
        return result


@dataclass(frozen=True, order=True)
class OpticalReceptionKey:
    """Physical identity of one independently closable coherent incident state."""

    state_handle: int
    state_generation: int
    camera_program_handle: int
    camera_program_generation: int
    exposure_frame_id: int
    exposure_slice_id: int
    destination_node: str
    destination_port: str
    scattering_product: str
    spectral: OpticalSpectralIdentity
    coherence_id: int
    arrival_epoch: int
    mode_id: int
    solve_epoch: int

    def validate(self) -> None:
        identity_values = (
            self.state_handle,
            self.state_generation,
            self.camera_program_handle,
            self.camera_program_generation,
            self.exposure_frame_id,
            self.exposure_slice_id,
        )
        if min(int(value) for value in identity_values) < 0:
            raise ValueError(
                "reception state, camera program, frame, and slice identities "
                "must be non-negative"
            )
        if not self.destination_node.strip() or not self.destination_port.strip():
            raise ValueError("reception destination node and port are required")
        if not self.scattering_product.strip():
            raise ValueError("reception scattering product is required")
        if not 0 <= int(self.coherence_id) <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("reception coherence id must fit uint64")
        if min(int(self.arrival_epoch), int(self.mode_id), int(self.solve_epoch)) < 0:
            raise ValueError("reception epochs and mode id must be non-negative")
        self.spectral.validate()

    def contract(self) -> dict[str, Any]:
        self.validate()
        return {
            "state_handle": self.state_handle,
            "state_generation": self.state_generation,
            "camera_program_handle": self.camera_program_handle,
            "camera_program_generation": self.camera_program_generation,
            "exposure_frame_id": self.exposure_frame_id,
            "exposure_slice_id": self.exposure_slice_id,
            "destination_node": self.destination_node,
            "destination_port": self.destination_port,
            "scattering_product": self.scattering_product,
            "spectral": self.spectral.contract(),
            "coherence_id": self.coherence_id,
            "arrival_epoch": self.arrival_epoch,
            "mode_id": self.mode_id,
            "solve_epoch": self.solve_epoch,
        }

    @classmethod
    def from_contract(cls, value: dict[str, Any]) -> "OpticalReceptionKey":
        result = cls(
            state_handle=int(value["state_handle"]),
            state_generation=int(value["state_generation"]),
            camera_program_handle=int(value["camera_program_handle"]),
            camera_program_generation=int(value["camera_program_generation"]),
            exposure_frame_id=int(value["exposure_frame_id"]),
            exposure_slice_id=int(value["exposure_slice_id"]),
            destination_node=str(value["destination_node"]),
            destination_port=str(value["destination_port"]),
            scattering_product=str(value["scattering_product"]),
            spectral=OpticalSpectralIdentity.from_contract(value["spectral"]),
            coherence_id=int(value["coherence_id"]),
            arrival_epoch=int(value["arrival_epoch"]),
            mode_id=int(value["mode_id"]),
            solve_epoch=int(value["solve_epoch"]),
        )
        result.validate()
        return result


@dataclass(frozen=True, order=True)
class OpticalPredecessorDeclaration:
    slot: int
    product_id: int
    product_key: str

    def validate(self) -> None:
        if int(self.slot) < 0 or int(self.product_id) < 0:
            raise ValueError("reception predecessor slot and product id must be non-negative")
        if not self.product_key.strip():
            raise ValueError("reception predecessor product key is required")

    def contract(self) -> dict[str, Any]:
        self.validate()
        return {
            "slot": self.slot,
            "product_id": self.product_id,
            "product_key": self.product_key,
        }

    @classmethod
    def from_contract(
        cls, value: dict[str, Any]
    ) -> "OpticalPredecessorDeclaration":
        result = cls(
            int(value["slot"]),
            int(value["product_id"]),
            str(value["product_key"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class CompiledOpticalReception:
    """Cold descriptor for one qualified deterministic coherent graph join."""

    pool_id: int
    destination_node: str
    destination_port: str
    predecessors: tuple[OpticalPredecessorDeclaration, ...]
    lane_count: int
    representation: str
    policy: OpticalReceptionPolicy = OpticalReceptionPolicy.DETERMINISTIC_COHERENT
    grid_id: int = 0
    capacity: int = 64

    def validate(self) -> None:
        if int(self.pool_id) < 0 or not self.destination_node.strip():
            raise ValueError("compiled reception requires a stable pool and destination")
        if len(self.predecessors) < 2:
            raise ValueError("compiled reception requires at least two predecessors")
        for predecessor in self.predecessors:
            predecessor.validate()
        slots = tuple(item.slot for item in self.predecessors)
        products = tuple(item.product_id for item in self.predecessors)
        if slots != tuple(range(len(slots))):
            raise ValueError("reception predecessor slots must be dense and stable")
        if len(set(products)) != len(products):
            raise ValueError("reception predecessor product ids must be unique")
        if self.lane_count not in SUPPORTED_RECEPTION_LANE_COUNTS:
            raise ValueError("compiled reception has unsupported lane width")
        if not self.representation.strip() or int(self.grid_id) < 0:
            raise ValueError("compiled reception representation and grid are required")
        if int(self.capacity) < 1:
            raise ValueError("compiled reception capacity must be positive")

    def contract(self) -> dict[str, Any]:
        self.validate()
        return {
            "pool_id": self.pool_id,
            "destination_node": self.destination_node,
            "destination_port": self.destination_port,
            "predecessors": [item.contract() for item in self.predecessors],
            "lane_count": self.lane_count,
            "representation": self.representation,
            "policy": self.policy.value,
            "grid_id": self.grid_id,
            "capacity": self.capacity,
        }

    @classmethod
    def from_contract(cls, value: dict[str, Any]) -> "CompiledOpticalReception":
        result = cls(
            pool_id=int(value["pool_id"]),
            destination_node=str(value["destination_node"]),
            destination_port=str(value["destination_port"]),
            predecessors=tuple(
                OpticalPredecessorDeclaration.from_contract(item)
                for item in value["predecessors"]
            ),
            lane_count=int(value["lane_count"]),
            representation=str(value["representation"]),
            policy=OpticalReceptionPolicy(str(value["policy"])),
            grid_id=int(value["grid_id"]),
            capacity=int(value["capacity"]),
        )
        result.validate()
        return result


@dataclass(frozen=True, eq=False)
class OpticalFieldContribution:
    key: OpticalReceptionKey
    predecessor_product_id: int
    contribution_id: int
    field: np.ndarray
    basis: TransverseBasis
    grid_id: int = 0

    def validated_field(self) -> np.ndarray:
        """Return a validated complex128 view without changing caller storage."""

        self.key.validate()
        values = np.asarray(self.field, np.complex128)
        expected = (self.key.spectral.lane_count, 2)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(f"reception Jones contribution must have shape {expected}")
        if int(self.predecessor_product_id) < 0 or int(self.contribution_id) < 0:
            raise ValueError("reception contribution ids must be non-negative")
        if int(self.grid_id) < 0:
            raise ValueError("reception contribution grid id must be non-negative")
        return values

    def detached(self) -> "OpticalFieldContribution":
        """Copy field and basis into immutable pool-owned storage."""

        values = np.array(
            self.validated_field(), dtype=np.complex128, order="C", copy=True
        )
        basis = TransverseBasis(
            np.array(self.basis.s, dtype=np.float64, copy=True),
            np.array(self.basis.p, dtype=np.float64, copy=True),
            np.array(self.basis.k, dtype=np.float64, copy=True),
        )
        values.flags.writeable = False
        basis.s.flags.writeable = False
        basis.p.flags.writeable = False
        basis.k.flags.writeable = False
        return OpticalFieldContribution(
            key=self.key,
            predecessor_product_id=self.predecessor_product_id,
            contribution_id=self.contribution_id,
            field=values,
            basis=basis,
            grid_id=self.grid_id,
        )

    def contract(self) -> dict[str, Any]:
        values = self.validated_field()
        return {
            "key": self.key.contract(),
            "predecessor_product_id": self.predecessor_product_id,
            "contribution_id": self.contribution_id,
            "grid_id": self.grid_id,
            "field": [
                [[float(value.real), float(value.imag)] for value in lane]
                for lane in values
            ],
            "basis": {
                "s": self.basis.s.tolist(),
                "p": self.basis.p.tolist(),
                "k": self.basis.k.tolist(),
            },
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpticalFieldContribution):
            return NotImplemented
        return (
            self.key == other.key
            and self.predecessor_product_id == other.predecessor_product_id
            and self.contribution_id == other.contribution_id
            and self.grid_id == other.grid_id
            and np.array_equal(self.validated_field(), other.validated_field())
            and np.array_equal(self.basis.s, other.basis.s)
            and np.array_equal(self.basis.p, other.basis.p)
            and np.array_equal(self.basis.k, other.basis.k)
        )

    @classmethod
    def from_contract(cls, value: dict[str, Any]) -> "OpticalFieldContribution":
        complex_field = np.asarray(value["field"], np.float64)
        if complex_field.ndim != 3 or complex_field.shape[-1] != 2:
            raise ValueError("serialized reception field must contain real/imag pairs")
        basis_value = value["basis"]
        result = cls(
            key=OpticalReceptionKey.from_contract(value["key"]),
            predecessor_product_id=int(value["predecessor_product_id"]),
            contribution_id=int(value["contribution_id"]),
            field=np.ascontiguousarray(
                complex_field[..., 0] + 1j * complex_field[..., 1],
                np.complex128,
            ),
            basis=TransverseBasis(
                np.asarray(basis_value["s"], np.float64),
                np.asarray(basis_value["p"], np.float64),
                np.asarray(basis_value["k"], np.float64),
            ),
            grid_id=int(value["grid_id"]),
        )
        result.validated_field()
        return result


@dataclass(frozen=True, order=True)
class OpticalCompletionCredit:
    key: OpticalReceptionKey
    predecessor_product_id: int

    def validate(self) -> None:
        self.key.validate()
        if int(self.predecessor_product_id) < 0:
            raise ValueError("completion predecessor product id must be non-negative")

    def contract(self) -> dict[str, Any]:
        self.validate()
        return {
            "key": self.key.contract(),
            "predecessor_product_id": self.predecessor_product_id,
        }

    @classmethod
    def from_contract(cls, value: dict[str, Any]) -> "OpticalCompletionCredit":
        result = cls(
            OpticalReceptionKey.from_contract(value["key"]),
            int(value["predecessor_product_id"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ClosedOpticalReceptionReport:
    accepted_contributions: int
    expected_predecessors: int
    completed_predecessors: int
    input_power: float
    reduced_field_power: float
    output_power: float
    residual_power: float = 0.0
    rejected_incompatible: int = 0
    stale_generation: int = 0
    dropped_contributions: int = 0
    unresolved_products: int = 0
    maximum_numerical_error: float = 0.0

    def contract(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_contract(
        cls, value: dict[str, Any]
    ) -> "ClosedOpticalReceptionReport":
        return cls(**{
            item.name: value[item.name]
            for item in cls.__dataclass_fields__.values()
        })


@dataclass(frozen=True)
class ClosedOpticalReception:
    key: OpticalReceptionKey
    reduced_field: np.ndarray
    output_field: np.ndarray
    report: ClosedOpticalReceptionReport


@dataclass
class _PendingReception:
    contributions: dict[int, dict[int, OpticalFieldContribution]] = field(
        default_factory=dict
    )
    completed: set[int] = field(default_factory=set)


class CoherentReceptionPool:
    """Bounded close-once correctness oracle for deterministic acyclic joins."""

    def __init__(
        self,
        descriptor: CompiledOpticalReception,
        *,
        state_generation: int,
        destination_basis: TransverseBasis,
        operator: JonesOperator | None = None,
    ) -> None:
        descriptor.validate()
        self.descriptor = descriptor
        self.state_generation = int(state_generation)
        self.destination_basis = destination_basis
        self.operator = operator or JonesOperator.identity()
        self._expected = {
            item.product_id: item.slot for item in descriptor.predecessors
        }
        self._pending: dict[OpticalReceptionKey, _PendingReception] = {}
        self._closed: set[OpticalReceptionKey] = set()

    def _state(self, key: OpticalReceptionKey) -> _PendingReception:
        key.validate()
        if key.state_generation != self.state_generation:
            raise RuntimeError("stale optical reception state generation")
        if (
            key.destination_node != self.descriptor.destination_node
            or key.destination_port != self.descriptor.destination_port
            or key.spectral.lane_count != self.descriptor.lane_count
        ):
            raise ValueError("reception key does not match its compiled descriptor")
        if key in self._closed:
            raise RuntimeError("optical reception key is already closed")
        state = self._pending.get(key)
        if state is None:
            if len(self._pending) >= self.descriptor.capacity:
                raise BufferError("coherent reception pool capacity exhausted")
            state = self._pending[key] = _PendingReception()
        return state

    def accept(self, contribution: OpticalFieldContribution) -> None:
        owned = contribution.detached()
        state = self._state(owned.key)
        product_id = int(owned.predecessor_product_id)
        if product_id not in self._expected:
            raise ValueError("contribution came from an undeclared predecessor")
        if product_id in state.completed:
            raise RuntimeError("contribution arrived after predecessor completion")
        if int(owned.grid_id) != self.descriptor.grid_id:
            raise ValueError("incompatible reception field grid; resampling is forbidden")
        slot = state.contributions.setdefault(product_id, {})
        contribution_id = int(owned.contribution_id)
        if contribution_id in slot:
            raise RuntimeError("duplicate optical reception contribution")
        # Validate the exact basis transform before retaining the contribution.
        basis_change(owned.basis, self.destination_basis)
        slot[contribution_id] = owned

    def complete(
        self, credit: OpticalCompletionCredit
    ) -> ClosedOpticalReception | None:
        credit.validate()
        state = self._state(credit.key)
        product_id = int(credit.predecessor_product_id)
        if product_id not in self._expected:
            raise ValueError("completion came from an undeclared predecessor")
        if product_id in state.completed:
            raise RuntimeError("duplicate optical reception completion")
        state.completed.add(product_id)
        if len(state.completed) != len(self._expected):
            return None
        return self._close(credit.key, state)

    def _close(
        self, key: OpticalReceptionKey, state: _PendingReception
    ) -> ClosedOpticalReception:
        ordered_fields: list[np.ndarray] = []
        input_power = 0.0
        for predecessor in self.descriptor.predecessors:
            contributions = state.contributions.get(predecessor.product_id, {})
            for contribution_id in sorted(contributions):
                contribution = contributions[contribution_id]
                field_values = contribution.validated_field()
                converted = basis_change(
                    contribution.basis, self.destination_basis
                ).apply(field_values)
                ordered_fields.append(converted)
                input_power += float(np.sum(np.abs(converted) ** 2))
        if ordered_fields:
            reduced = np.zeros(
                (self.descriptor.lane_count, 2), dtype=np.complex128
            )
            for values in ordered_fields:
                reduced += values
        else:
            reduced = np.zeros(
                (self.descriptor.lane_count, 2), dtype=np.complex128
            )
        output = np.ascontiguousarray(self.operator.apply(reduced), np.complex128)
        reduced.flags.writeable = False
        output.flags.writeable = False
        report = ClosedOpticalReceptionReport(
            accepted_contributions=len(ordered_fields),
            expected_predecessors=len(self._expected),
            completed_predecessors=len(state.completed),
            input_power=input_power,
            reduced_field_power=float(np.sum(np.abs(reduced) ** 2)),
            output_power=float(np.sum(np.abs(output) ** 2)),
        )
        del self._pending[key]
        self._closed.add(key)
        return ClosedOpticalReception(key, reduced, output, report)

    def unresolved_predecessors(
        self, key: OpticalReceptionKey
    ) -> tuple[int, ...]:
        state = self._pending.get(key)
        if state is None:
            return tuple(sorted(self._expected))
        return tuple(
            predecessor.product_id
            for predecessor in self.descriptor.predecessors
            if predecessor.product_id not in state.completed
        )

    @staticmethod
    def incoherent_intensity(
        receptions: Iterable[ClosedOpticalReception],
    ) -> np.ndarray:
        values = tuple(receptions)
        if not values:
            return np.empty((0,), np.float64)
        shape = values[0].output_field.shape
        if any(item.output_field.shape != shape for item in values):
            raise ValueError("incoherent detector products must have matching shapes")
        return np.sum(
            [np.abs(item.output_field) ** 2 for item in values],
            axis=0,
            dtype=np.float64,
        )


__all__ = [
    "OpticalSpectralMode",
    "OpticalReceptionPolicy",
    "OpticalSpectralIdentity",
    "OpticalReceptionKey",
    "OpticalPredecessorDeclaration",
    "CompiledOpticalReception",
    "OpticalFieldContribution",
    "OpticalCompletionCredit",
    "ClosedOpticalReceptionReport",
    "ClosedOpticalReception",
    "CoherentReceptionPool",
]
