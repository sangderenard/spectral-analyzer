"""Canonical Jones and differential operators for complex optical transport.

The ordinary ray record remains compact.  Rays that enter a complex optical
context refer to persistent basis/operator records in an arena-owned state
block.  This module is the deterministic CPU/reference implementation of the
same conventions declared by ``complex_optical_operators.h`` and its GLSL
include.

Conventions
-----------
* A transverse basis is right handed: ``s × p = k``.
* Jones vectors are column vectors ``[E_s, E_p]``.
* Operator composition follows ray order: ``B.compose(A)`` means ``B @ A``.
* Differential state is canonical ``[q_s, q_p, p_s, p_p]`` with transverse
  optical momentum ``p = n * direction_transverse`` at a plane normal to k.
* The complete signed 4x4 tangent map is retained.  Its determinant is a
  diagnostic, not a replacement for the map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Sequence

import numpy as np


COMPLEX_OPTICAL_OPERATOR_SCHEMA = "complex-optical-operators-v1"
_EPS = 1.0e-12
_C_M_S = 299_792_458.0


def _unit(value: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(value, np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= _EPS:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    return vector / norm


@dataclass(frozen=True)
class TransverseBasis:
    """Right-handed local polarization basis ``(s, p, k)``."""

    s: np.ndarray
    p: np.ndarray
    k: np.ndarray

    def __post_init__(self) -> None:
        k = _unit(self.k, name="basis k")
        s_raw = np.asarray(self.s, np.float64).reshape(3)
        s = s_raw - k * float(np.dot(k, s_raw))
        s = _unit(s, name="basis s")
        p = np.cross(k, s)
        p = _unit(p, name="basis p")
        if float(np.dot(p, np.asarray(self.p, np.float64))) < 0.0:
            s = -s
            p = -p
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "k", k)

    @classmethod
    def from_direction(
        cls,
        direction: Sequence[float],
        reference: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> "TransverseBasis":
        k = _unit(direction, name="direction")
        reference_axis = _unit(reference, name="reference")
        s = np.cross(k, reference_axis)
        if float(np.linalg.norm(s)) <= 1.0e-8:
            fallback = np.array((0.0, 1.0, 0.0), np.float64)
            if abs(float(np.dot(k, fallback))) > 0.95:
                fallback = np.array((1.0, 0.0, 0.0), np.float64)
            s = np.cross(k, fallback)
        s = _unit(s, name="derived s")
        return cls(s=s, p=np.cross(k, s), k=k)

    @classmethod
    def incidence(
        cls,
        direction: Sequence[float],
        interface_normal: Sequence[float],
        *,
        shared_s: Sequence[float] | None = None,
    ) -> "TransverseBasis":
        k = _unit(direction, name="direction")
        if shared_s is None:
            normal = _unit(interface_normal, name="interface normal")
            s = np.cross(k, normal)
            if float(np.linalg.norm(s)) <= 1.0e-8:
                return cls.from_direction(k)
        else:
            s = np.asarray(shared_s, np.float64)
        s = _unit(s - k * float(np.dot(k, s)), name="incidence s")
        return cls(s=s, p=np.cross(k, s), k=k)

    def packed(self) -> np.ndarray:
        """Two std430 vec4 slots; k is reconstructed as ``cross(s,p)``."""

        return np.asarray(
            [*self.s, 0.0, *self.p, 0.0],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class JonesOperator:
    """Complex 2x2 electric-field operator in declared input/output bases."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, np.complex128)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("Jones operator must be a finite complex 2x2 matrix")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls) -> "JonesOperator":
        return cls(np.eye(2, dtype=np.complex128))

    @classmethod
    def diagonal(cls, s: complex, p: complex) -> "JonesOperator":
        return cls(np.diag(np.asarray([s, p], np.complex128)))

    def apply(self, field: Sequence[complex] | np.ndarray) -> np.ndarray:
        values = np.asarray(field, np.complex128)
        if values.shape[-1] != 2:
            raise ValueError("Jones field's last dimension must be 2")
        return np.einsum("ij,...j->...i", self.matrix, values)

    def compose(self, earlier: "JonesOperator") -> "JonesOperator":
        return JonesOperator(self.matrix @ earlier.matrix)

    def adjoint(self) -> "JonesOperator":
        return JonesOperator(self.matrix.conj().T)

    def reciprocal_reverse(self) -> "JonesOperator":
        """Reverse a reciprocal device under the same port conventions."""

        return JonesOperator(self.matrix.T)

    def packed(self) -> np.ndarray:
        """Two vec4 rows: ``(m0.re,m0.im,m1.re,m1.im)``."""

        m = self.matrix
        return np.asarray([
            m[0, 0].real, m[0, 0].imag, m[0, 1].real, m[0, 1].imag,
            m[1, 0].real, m[1, 0].imag, m[1, 1].real, m[1, 1].imag,
        ], np.float32)


def basis_change(
    source: TransverseBasis,
    destination: TransverseBasis,
) -> JonesOperator:
    """Rotate Jones coordinates without changing the physical field."""

    if float(np.dot(source.k, destination.k)) < 1.0 - 1.0e-8:
        raise ValueError("basis change requires the same propagation direction")
    return JonesOperator(np.asarray([
        [np.dot(destination.s, source.s), np.dot(destination.s, source.p)],
        [np.dot(destination.p, source.s), np.dot(destination.p, source.p)],
    ], np.complex128))


@dataclass(frozen=True)
class DielectricInterfaceResult:
    reflected_direction: np.ndarray
    transmitted_direction: np.ndarray | None
    incident_basis: TransverseBasis
    reflected_basis: TransverseBasis
    transmitted_basis: TransverseBasis | None
    reflection: JonesOperator
    transmission: JonesOperator | None
    cos_incident: float
    cos_transmitted: float
    total_internal_reflection: bool


def dielectric_interface(
    direction: Sequence[float],
    interface_normal: Sequence[float],
    n_incident: float,
    n_transmitted: float,
) -> DielectricInterfaceResult:
    """Power-normalized lossless dielectric Fresnel/Jones scattering."""

    ni, nt = float(n_incident), float(n_transmitted)
    if not (math.isfinite(ni) and math.isfinite(nt) and ni > 0.0 and nt > 0.0):
        raise ValueError("dielectric indices must be finite and positive")
    k = _unit(direction, name="incident direction")
    normal = _unit(interface_normal, name="interface normal")
    if float(np.dot(k, normal)) > 0.0:
        normal = -normal
    cos_i = min(1.0, max(0.0, -float(np.dot(k, normal))))
    reflected = _unit(k + 2.0 * cos_i * normal, name="reflected direction")
    shared_s = np.cross(k, normal)
    if float(np.linalg.norm(shared_s)) <= 1.0e-8:
        incident_basis = TransverseBasis.from_direction(k)
        shared_s = incident_basis.s
    incident_basis = TransverseBasis.incidence(k, normal, shared_s=shared_s)
    reflected_basis = TransverseBasis.incidence(
        reflected, normal, shared_s=shared_s
    )

    eta = ni / nt
    sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        cos_t_complex = 1j * math.sqrt(sin2_t - 1.0)
        rs = (ni * cos_i - nt * cos_t_complex) / (
            ni * cos_i + nt * cos_t_complex
        )
        rp = (nt * cos_i - ni * cos_t_complex) / (
            nt * cos_i + ni * cos_t_complex
        )
        return DielectricInterfaceResult(
            reflected, None, incident_basis, reflected_basis, None,
            JonesOperator.diagonal(rs, rp), None,
            cos_i, 0.0, True,
        )

    cos_t = math.sqrt(max(0.0, 1.0 - sin2_t))
    transmitted = _unit(
        eta * k + (eta * cos_i - cos_t) * normal,
        name="transmitted direction",
    )
    transmitted_basis = TransverseBasis.incidence(
        transmitted, normal, shared_s=shared_s
    )
    denom_s = ni * cos_i + nt * cos_t
    denom_p = nt * cos_i + ni * cos_t
    rs = (ni * cos_i - nt * cos_t) / denom_s
    rp = (nt * cos_i - ni * cos_t) / denom_p
    ts = 2.0 * ni * cos_i / denom_s
    tp = 2.0 * ni * cos_i / denom_p
    power_scale = math.sqrt(nt * cos_t / max(ni * cos_i, _EPS))
    return DielectricInterfaceResult(
        reflected, transmitted, incident_basis, reflected_basis,
        transmitted_basis, JonesOperator.diagonal(rs, rp),
        JonesOperator.diagonal(ts * power_scale, tp * power_scale),
        cos_i, cos_t, False,
    )


_SYMPLECTIC_FORM = np.block([
    [np.zeros((2, 2)), np.eye(2)],
    [-np.eye(2), np.zeros((2, 2))],
])


@dataclass(frozen=True)
class PhaseSpaceJacobian:
    """Signed 4x4 tangent map in canonical transverse ray coordinates."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("phase-space Jacobian must be a finite 4x4 matrix")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls) -> "PhaseSpaceJacobian":
        return cls(np.eye(4, dtype=np.float64))

    @classmethod
    def finite_difference(
        cls,
        mapping: Callable[[np.ndarray], np.ndarray],
        state: Sequence[float],
        step: float | Sequence[float] = 1.0e-6,
    ) -> "PhaseSpaceJacobian":
        x = np.asarray(state, np.float64).reshape(4)
        h = np.broadcast_to(np.asarray(step, np.float64), (4,))
        if np.any(~np.isfinite(h)) or np.any(h <= 0.0):
            raise ValueError("finite-difference steps must be finite and positive")
        columns = []
        for index in range(4):
            delta = np.zeros(4, np.float64)
            delta[index] = h[index]
            plus = np.asarray(mapping(x + delta), np.float64).reshape(4)
            minus = np.asarray(mapping(x - delta), np.float64).reshape(4)
            columns.append((plus - minus) / (2.0 * h[index]))
        return cls(np.column_stack(columns))

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.matrix))

    @property
    def symplectic_residual(self) -> float:
        error = self.matrix.T @ _SYMPLECTIC_FORM @ self.matrix - _SYMPLECTIC_FORM
        return float(np.linalg.norm(error, ord="fro"))

    def compose(self, earlier: "PhaseSpaceJacobian") -> "PhaseSpaceJacobian":
        return PhaseSpaceJacobian(self.matrix @ earlier.matrix)

    def inverse(self) -> "PhaseSpaceJacobian":
        return PhaseSpaceJacobian(np.linalg.inv(self.matrix))

    def packed(self) -> np.ndarray:
        """Column-major mat4 matching GLSL."""

        return np.asarray(self.matrix, np.float32).reshape(-1, order="F")

    def configuration_amplitude_gain(self, *, floor: float = 1.0e-12) -> float:
        """Geometric field gain for the fixed-input-momentum q→q map.

        This is ``1/sqrt(abs(det(dq_out/dq_in)))``.  It deliberately raises at
        a caustic: a wave/caustic operator must resolve that neighborhood rather
        than allowing a ray adapter to manufacture infinite amplitude.
        """

        determinant = float(np.linalg.det(self.matrix[:2, :2]))
        if not math.isfinite(determinant) or abs(determinant) <= float(floor):
            raise ValueError("configuration map is singular or at a caustic")
        return 1.0 / math.sqrt(abs(determinant))


def optical_phase(
    frequency_hz: float,
    optical_path_m: float,
    *,
    reference_optical_path_m: float = 0.0,
) -> complex:
    """Return carrier phase after removing an explicit reference OPL.

    Reducing cycles modulo one before multiplying by 2π avoids needlessly
    passing a huge visible-light phase to ``sin``/``cos``.  A coherent cohort
    must use one shared reference path.
    """

    frequency = float(frequency_hz)
    delta_opl = float(optical_path_m) - float(reference_optical_path_m)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be finite and positive")
    if not math.isfinite(delta_opl):
        raise ValueError("optical path must be finite")
    fractional_cycles = math.remainder(frequency * delta_opl / _C_M_S, 1.0)
    phase = 2.0 * math.pi * fractional_cycles
    return complex(math.cos(phase), math.sin(phase))


@dataclass(frozen=True)
class ComplexOpticalOperator:
    """One indexed persistent record used by a complex transport boundary."""

    jones: JonesOperator = field(default_factory=JonesOperator.identity)
    phase_space: PhaseSpaceJacobian = field(
        default_factory=PhaseSpaceJacobian.identity
    )

    def compose(self, earlier: "ComplexOpticalOperator") -> "ComplexOpticalOperator":
        return ComplexOpticalOperator(
            self.jones.compose(earlier.jones),
            self.phase_space.compose(earlier.phase_space),
        )

    def packed(self) -> np.ndarray:
        return np.concatenate((self.jones.packed(), self.phase_space.packed()))


@dataclass(frozen=True)
class ComplexSourceMode:
    """One coherent Jones source mode stored outside ordinary ray records."""

    jones: np.ndarray
    power_weight: float
    coherence_id: int
    basis_id: int
    operator_id: int = 0
    flags: int = 0

    def __post_init__(self) -> None:
        field = np.asarray(self.jones, np.complex128).reshape(2)
        norm = float(np.linalg.norm(field))
        if not np.all(np.isfinite(field)) or norm <= _EPS:
            raise ValueError("source Jones mode must be finite and non-zero")
        weight = float(self.power_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("source-mode power weight must be finite and non-negative")
        if not 0 <= int(self.coherence_id) <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("coherence_id must fit uint64")
        for name, value in (
            ("basis_id", self.basis_id),
            ("operator_id", self.operator_id),
            ("flags", self.flags),
        ):
            if not 0 <= int(value) <= 0xFFFFFFFF:
                raise ValueError(f"{name} must fit uint32")
        object.__setattr__(self, "jones", field / norm)
        object.__setattr__(self, "power_weight", weight)

    def packed_words(self) -> np.ndarray:
        """Three std430 vec4 slots represented as exact uint32 words."""

        words = np.zeros(12, np.uint32)
        floats = words.view(np.float32)
        floats[:5] = (
            self.jones[0].real,
            self.jones[0].imag,
            self.jones[1].real,
            self.jones[1].imag,
            self.power_weight,
        )
        coherence = int(self.coherence_id)
        words[5] = np.uint32(coherence & 0xFFFFFFFF)
        words[6] = np.uint32((coherence >> 32) & 0xFFFFFFFF)
        words[7] = np.uint32(self.basis_id)
        words[8] = np.uint32(self.operator_id)
        words[9] = np.uint32(self.flags)
        return words


class ComplexOperatorStateBlock:
    """Contiguous cold-built basis/operator tables with stable integer handles."""

    def __init__(self) -> None:
        self._bases: list[TransverseBasis] = []
        self._operators: list[ComplexOpticalOperator] = []
        self._source_modes: list[ComplexSourceMode] = []

    def add_basis(self, basis: TransverseBasis) -> int:
        self._bases.append(basis)
        return len(self._bases) - 1

    def add_operator(self, operator: ComplexOpticalOperator) -> int:
        self._operators.append(operator)
        return len(self._operators) - 1

    def add_source_mode(self, mode: ComplexSourceMode) -> int:
        self._source_modes.append(mode)
        return len(self._source_modes) - 1

    def add_polarization_modes(
        self,
        polarization: Any,
        *,
        basis_id: int,
        operator_id: int = 0,
        coherence_seed: int = 0,
    ) -> tuple[int, ...]:
        handles = []
        for mode_index, (weight, jones) in enumerate(
            polarization.coherent_mode_decomposition()
        ):
            handles.append(self.add_source_mode(ComplexSourceMode(
                jones=jones,
                power_weight=float(weight),
                coherence_id=int(coherence_seed) + mode_index,
                basis_id=int(basis_id),
                operator_id=int(operator_id),
            )))
        return tuple(handles)

    def freeze(self) -> dict[str, Any]:
        bases = np.ascontiguousarray(
            np.stack([value.packed() for value in self._bases], axis=0)
            if self._bases else np.empty((0, 8), np.float32)
        )
        operators = np.ascontiguousarray(
            np.stack([value.packed() for value in self._operators], axis=0)
            if self._operators else np.empty((0, 24), np.float32)
        )
        source_modes = np.ascontiguousarray(
            np.stack(
                [value.packed_words() for value in self._source_modes], axis=0
            )
            if self._source_modes else np.empty((0, 12), np.uint32)
        )
        return {
            "schema": COMPLEX_OPTICAL_OPERATOR_SCHEMA,
            "bases": bases,
            "operators": operators,
            "source_modes": source_modes,
            "basis_stride_bytes": 32,
            "operator_stride_bytes": 96,
            "source_mode_stride_bytes": 48,
        }


@dataclass(frozen=True)
class LensPhaseSpaceBatch:
    matrices: np.ndarray
    determinants: np.ndarray
    symplectic_residuals: np.ndarray
    valid: np.ndarray
    backend: str


def estimate_compound_lens_phase_space(
    lens: Any,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    spectral_lane: int = -1,
    n_input: float = 1.0,
    n_output: float = 1.0,
    q_step_m: float = 1.0e-6,
    p_step: float = 1.0e-6,
    n_threads: int = 0,
) -> LensPhaseSpaceBatch:
    """Evaluate full exact-lens tangent maps, preferring the native helper."""

    ray_origins = np.ascontiguousarray(origins, np.float64)
    ray_directions = np.ascontiguousarray(directions, np.float64)
    if (
        ray_origins.ndim != 2
        or ray_origins.shape[1] != 3
        or ray_directions.shape != ray_origins.shape
    ):
        raise ValueError("origins and directions must have shape (N, 3)")
    payload = np.ascontiguousarray(lens.build_gpu_payload(), np.float32)
    try:
        import _spectral_kernels as native

        estimate = getattr(native, "estimate_lens_phase_space_jacobians")
    except (ImportError, AttributeError):
        estimate = None
    if estimate is not None:
        matrices, determinants, residuals, valid = estimate(
            payload,
            ray_origins,
            ray_directions,
            spectral_lane=int(spectral_lane),
            n_input=float(n_input),
            n_output=float(n_output),
            q_step_m=float(q_step_m),
            p_step=float(p_step),
            n_threads=int(n_threads),
        )
        return LensPhaseSpaceBatch(
            np.asarray(matrices, np.float64),
            np.asarray(determinants, np.float64),
            np.asarray(residuals, np.float64),
            np.asarray(valid, np.uint8).astype(bool),
            "native-cpp",
        )

    from camera_designer.compound_optics import TerminationReason

    matrices = np.zeros((len(ray_origins), 4, 4), np.float64)
    determinants = np.zeros(len(ray_origins), np.float64)
    residuals = np.full(len(ray_origins), np.inf, np.float64)
    valid = np.zeros(len(ray_origins), bool)
    steps = np.asarray((q_step_m, q_step_m, p_step, p_step), np.float64)
    for ray_index, (origin, direction) in enumerate(
        zip(ray_origins, ray_directions)
    ):
        direction = _unit(direction, name="ray direction")
        axial_sign = 1.0 if direction[0] >= 0.0 else -1.0
        state = np.asarray((
            origin[1], origin[2],
            float(n_input) * direction[1],
            float(n_input) * direction[2],
        ))

        def mapping(value: np.ndarray) -> np.ndarray:
            transverse2 = (
                value[2] * value[2] + value[3] * value[3]
            ) / (float(n_input) ** 2)
            if transverse2 >= 1.0:
                raise ValueError("perturbed ray has no real axial direction")
            perturbed_origin = np.asarray(
                (origin[0], value[0], value[1]), np.float64
            )
            perturbed_direction = np.asarray((
                axial_sign * math.sqrt(max(0.0, 1.0 - transverse2)),
                value[2] / float(n_input),
                value[3] / float(n_input),
            ))
            result = lens.trace(perturbed_origin, perturbed_direction)
            if result.reason is not TerminationReason.PASSED:
                raise ValueError("perturbed ray terminates in compound lens")
            return np.asarray((
                result.origin[1], result.origin[2],
                float(n_output) * result.direction[1],
                float(n_output) * result.direction[2],
            ))

        try:
            jacobian = PhaseSpaceJacobian.finite_difference(
                mapping, state, steps
            )
        except ValueError:
            continue
        matrices[ray_index] = jacobian.matrix
        determinants[ray_index] = jacobian.determinant
        residuals[ray_index] = jacobian.symplectic_residual
        valid[ray_index] = True
    return LensPhaseSpaceBatch(
        matrices, determinants, residuals, valid, "python-reference"
    )


def canonical_operator_contract() -> dict[str, Any]:
    return {
        "schema": COMPLEX_OPTICAL_OPERATOR_SCHEMA,
        "jones_basis": "right-handed-local-s-p; s-cross-p=k",
        "jones_vector": "column-[E_s,E_p]",
        "fresnel": "power-normalized-electric-field",
        "phase_space_coordinates": "[q_s,q_p,n*k_s,n*k_p]",
        "phase_space_map": "signed-4x4",
        "ray_field_gain": "abs(det(dq_out/dq_in))^-1/2; caustics-forbidden",
        "carrier_phase": "reference-opl-reduced-exp(i*2pi*f*delta-opl/c)",
        "storage": "persistent-contiguous-indexed-state-block",
        "lane_references": {
            "word14": "basis_id",
            "word15": "operator_id",
        },
    }


__all__ = [
    "COMPLEX_OPTICAL_OPERATOR_SCHEMA",
    "TransverseBasis",
    "JonesOperator",
    "DielectricInterfaceResult",
    "dielectric_interface",
    "basis_change",
    "PhaseSpaceJacobian",
    "optical_phase",
    "ComplexOpticalOperator",
    "ComplexSourceMode",
    "ComplexOperatorStateBlock",
    "LensPhaseSpaceBatch",
    "estimate_compound_lens_phase_space",
    "canonical_operator_contract",
]
