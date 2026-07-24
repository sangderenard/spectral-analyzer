"""Typed contracts shared by optical graph compilers and transport engines.

These records describe physical transport.  They deliberately do not prescribe
whether a product is evaluated by a ray sampler, a parametric kernel, a wave
arena, or a cached Maxwell scattering artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from functools import reduce
import math
from typing import Any, Mapping


_C_M_S = 299_792_458.0


def commensurate_refresh_rate_hz(
    rates_hz: tuple[float, ...],
    *,
    maximum_rate_hz: float = 1.0e9,
    maximum_denominator: int = 1_000_000,
) -> float | None:
    """Return the least common refresh rate for rational component rates.

    ``None`` means no finite cadence was requested or the exact rational LCM
    would exceed the configured clock ceiling, in which case event timestamps
    remain authoritative.
    """

    positive = tuple(float(rate) for rate in rates_hz if float(rate) > 0.0)
    if not positive:
        return None
    if not all(math.isfinite(rate) for rate in positive):
        raise ValueError("component refresh rates must be finite")
    fractions = tuple(
        Fraction(str(rate)).limit_denominator(maximum_denominator)
        for rate in positive
    )
    numerator_lcm = reduce(math.lcm, (value.numerator for value in fractions))
    denominator_gcd = reduce(math.gcd, (value.denominator for value in fractions))
    result = float(Fraction(numerator_lcm, denominator_gcd))
    if result > float(maximum_rate_hz):
        return None
    return result


class OpticalProductKind(str, Enum):
    TRANSPORT = "transport"
    REFLECTED = "reflected"
    TRANSMITTED = "transmitted"
    DIFFRACTED = "diffracted"
    ABSORBED = "absorbed"
    DETECTED = "detected"
    EMITTED = "emitted"


class OpticalSolveMode(str, Enum):
    STOCHASTIC_RAY = "stochastic-ray"
    DETERMINISTIC_BRANCH = "deterministic-branch"
    HYBRID = "hybrid"


class OpticalCyclePolicy(str, Enum):
    REJECT = "reject"
    TIME_WINDOW = "time-window"
    RESIDUAL_ENERGY = "residual-energy"
    LINEAR_SCATTERING_SOLVE = "linear-scattering-solve"


@dataclass(frozen=True)
class OpticalTimingSpec:
    """Relative physical timing carried by one directed product edge.

    ``optical_path_m`` controls carrier phase. ``group_delay_s`` controls
    envelope/arrival scheduling. They are separate because dispersive systems
    generally do not permit one to be inferred from the other.
    """

    geometric_length_m: float = 0.0
    optical_path_m: float = 0.0
    group_delay_s: float = 0.0
    phase_reference_m: float = 0.0
    exact: bool = False

    def validate(self) -> None:
        values = (
            self.geometric_length_m,
            self.optical_path_m,
            self.group_delay_s,
            self.phase_reference_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("optical timing values must be finite")
        if self.geometric_length_m < 0.0:
            raise ValueError("geometric optical length must be non-negative")
        if self.optical_path_m < 0.0:
            raise ValueError("optical path length must be non-negative")
        if self.group_delay_s < 0.0:
            raise ValueError("optical group delay must be non-negative")

    @classmethod
    def homogeneous(
        cls,
        length_m: float,
        *,
        phase_index: float = 1.0,
        group_index: float | None = None,
        exact: bool = True,
    ) -> "OpticalTimingSpec":
        length = float(length_m)
        n_phase = float(phase_index)
        n_group = n_phase if group_index is None else float(group_index)
        timing = cls(
            geometric_length_m=length,
            optical_path_m=length * n_phase,
            group_delay_s=length * n_group / _C_M_S,
            exact=bool(exact),
        )
        timing.validate()
        return timing

    def contract(self) -> dict[str, Any]:
        self.validate()
        return {
            "geometric_length_m": self.geometric_length_m,
            "optical_path_m": self.optical_path_m,
            "group_delay_s": self.group_delay_s,
            "phase_reference_m": self.phase_reference_m,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class OpticalScatteringProductSpec:
    """One named physical result emitted by a scattering/propagation node."""

    key: str
    kind: OpticalProductKind = OpticalProductKind.TRANSPORT
    coherent: bool = True
    deterministic: bool = True
    selection_pdf: float | None = None
    power_upper_bound: float = 1.0
    timing: OpticalTimingSpec = field(default_factory=OpticalTimingSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.key.strip():
            raise ValueError("optical scattering products require a key")
        self.timing.validate()
        if not math.isfinite(float(self.power_upper_bound)):
            raise ValueError("product power bound must be finite")
        if not 0.0 <= float(self.power_upper_bound) <= 1.0 + 1.0e-9:
            raise ValueError("product power bound must be in [0, 1]")
        if self.selection_pdf is not None:
            pdf = float(self.selection_pdf)
            if not math.isfinite(pdf) or not 0.0 < pdf <= 1.0:
                raise ValueError("product selection PDF must be in (0, 1]")
        if self.deterministic and self.selection_pdf is not None:
            raise ValueError(
                "deterministic optical products must not carry a selection PDF"
            )
        if not self.deterministic and self.selection_pdf is None:
            raise ValueError(
                "stochastically selected optical products require a selection PDF"
            )

    def contract(self) -> dict[str, Any]:
        self.validate()
        return {
            "key": self.key,
            "kind": self.kind.value,
            "coherent": self.coherent,
            "deterministic": self.deterministic,
            "selection_pdf": self.selection_pdf,
            "power_upper_bound": self.power_upper_bound,
            "timing": self.timing.contract(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OpticalAccuracySpec:
    """Termination and accounting requirements for one optical solve."""

    mode: OpticalSolveMode = OpticalSolveMode.DETERMINISTIC_BRANCH
    relative_error: float = 1.0e-6
    residual_power: float = 1.0e-9
    maximum_branch_depth: int = 256
    arrival_window_s: float = math.inf
    cycle_policy: OpticalCyclePolicy = OpticalCyclePolicy.RESIDUAL_ENERGY
    allow_dropped_products: bool = False

    def validate(self) -> None:
        if not 0.0 < float(self.relative_error) < 1.0:
            raise ValueError("relative optical error must be in (0, 1)")
        if not 0.0 <= float(self.residual_power) < 1.0:
            raise ValueError("residual optical power must be in [0, 1)")
        if int(self.maximum_branch_depth) < 1:
            raise ValueError("maximum branch depth must be positive")
        if self.arrival_window_s <= 0.0 or math.isnan(self.arrival_window_s):
            raise ValueError("arrival window must be positive")

    def contract(self) -> dict[str, Any]:
        self.validate()
        return {
            "mode": self.mode.value,
            "relative_error": self.relative_error,
            "residual_power": self.residual_power,
            "maximum_branch_depth": self.maximum_branch_depth,
            "arrival_window_s": self.arrival_window_s,
            "cycle_policy": self.cycle_policy.value,
            "allow_dropped_products": self.allow_dropped_products,
        }


@dataclass
class OpticalSolveReport:
    """Power and convergence accounting common to every optical backend."""

    input_power: float = 0.0
    transported_power: float = 0.0
    absorbed_power: float = 0.0
    detected_power: float = 0.0
    residual_power: float = 0.0
    dropped_power: float = 0.0
    unresolved_products: int = 0
    maximum_relative_error: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, accuracy: OpticalAccuracySpec) -> None:
        accuracy.validate()
        values = (
            self.input_power,
            self.transported_power,
            self.absorbed_power,
            self.detected_power,
            self.residual_power,
            self.dropped_power,
            self.maximum_relative_error,
        )
        if not all(math.isfinite(float(value)) and value >= 0.0 for value in values):
            raise ValueError("optical solve report values must be finite and non-negative")
        if int(self.unresolved_products) < 0:
            raise ValueError("unresolved optical product count must be non-negative")
        if self.dropped_power > 0.0 and not accuracy.allow_dropped_products:
            raise RuntimeError("optical solve dropped transport products")
        if self.maximum_relative_error > accuracy.relative_error:
            raise RuntimeError("optical solve exceeded its relative-error contract")
        if self.input_power > 0.0:
            residual_fraction = self.residual_power / self.input_power
            if residual_fraction > accuracy.residual_power:
                raise RuntimeError(
                    "optical solve exceeded its residual-power contract"
                )

    def contract(self) -> dict[str, Any]:
        return {
            "input_power": self.input_power,
            "transported_power": self.transported_power,
            "absorbed_power": self.absorbed_power,
            "detected_power": self.detected_power,
            "residual_power": self.residual_power,
            "dropped_power": self.dropped_power,
            "unresolved_products": self.unresolved_products,
            "maximum_relative_error": self.maximum_relative_error,
            "metadata": dict(self.metadata),
        }
