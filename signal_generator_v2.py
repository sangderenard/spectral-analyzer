from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Protocol, Sequence, Tuple
import cmath
import math


# ============================================================
# Utility / math helpers
# ============================================================

PI2 = 2.0 * math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_expj(phase: float) -> complex:
    """Return e^(j*phase) with the standard library."""
    return cmath.rect(1.0, phase)


def integrate_trapezoid(
    fn: Callable[[float], float],
    t0: float,
    t1: float,
    steps: int = 32,
) -> float:
    """
    Basic trapezoidal integration for scalar functions.
    This is intentionally explicit and local rather than optimized.
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")

    if t0 == t1:
        return 0.0

    if t1 < t0:
        return -integrate_trapezoid(fn, t1, t0, steps=steps)

    dt = (t1 - t0) / (steps - 1)
    acc = 0.0
    for i in range(steps):
        t = t0 + i * dt
        weight = 0.5 if (i == 0 or i == steps - 1) else 1.0
        acc += weight * fn(t)
    return acc * dt


def differentiate_central(
    fn: Callable[[float], float],
    t: float,
    h: float = 1e-6,
) -> float:
    """Central difference derivative."""
    return (fn(t + h) - fn(t - h)) / (2.0 * h)


# ============================================================
# Contracts / protocols
# ============================================================

class SupportsComplexEvaluation(Protocol):
    def evaluate_complex(self, t: float) -> complex:
        ...


# ============================================================
# Parameter self-description
# ============================================================

@dataclass
class KnobSpec:
    """Self-description of one configurable parameter on a pipeline class.

    Pipeline classes advertise their configurable parameters via a ``knobs()``
    classmethod returning ``List[KnobSpec]``.  UI layers iterate the list to
    generate controls without any hard-coded knowledge of per-class parameters.

    Attributes
    ----------
    name:
        Python attribute / ``__init__`` parameter name.  Dot-path notation
        (e.g. ``"adsr.attack"``) traverses nested objects.
    label:
        Short human-readable label for the UI.
    dtype:
        ``"float"`` | ``"int"`` | ``"bool"`` | ``"choice"`` | ``"knots"``
    default:
        Default value for the parameter.
    low / high:
        Numeric bounds.
    step:
        Discrete increment (0 → UI chooses automatically).
    unit:
        Display unit string, e.g. ``"Hz"``, ``"s"``.
    choices:
        Ordered option strings for ``dtype="choice"`` knobs.
    is_log:
        Slider operates on a logarithmic scale when ``True``.
    group:
        Section label; a new section header is rendered whenever this
        field differs from the previous knob's group.
    fmt:
        Python ``format()`` spec used to render the numeric value.
    source_class:
        Name of the pipeline class this attribute maps to (informational).
    rebuild_layout:
        When ``True`` the UI should rebuild its full control layout after
        this knob changes (e.g. switching envelope type).
    visible_when:
        ``(attr_path, expected_value_str)`` — knob is hidden unless
        ``str(get_nested_attr(obj, attr_path)) == expected_value_str``.
    """
    name:           str
    label:          str
    dtype:          str                      = "float"
    default:        Any                      = None
    low:            float                    = 0.0
    high:           float                    = 1.0
    step:           float                    = 0.0
    unit:           str                      = ""
    choices:        List[str]                = field(default_factory=list)
    is_log:         bool                     = False
    group:          str                      = ""
    fmt:            str                      = ".3g"
    source_class:   str                      = ""
    rebuild_layout: bool                     = False
    visible_when:   Optional[Tuple[str, str]] = None


# ============================================================
# Core support dataclasses
# ============================================================

@dataclass(frozen=True)
class WitnessThresholds:
    """
    Controls how much accumulated phase is required to consider
    a local point in time 'witnessed' by surrounding history/future.

    Note: ``integration_steps_per_check`` was previously used by the witness
    search loop.  That loop is now incremental (one 2-point trapezoid slice per
    step), so that field no longer affects witness accuracy.  It is repurposed
    here as the tap count for any ``DriftModel`` subclass that does not supply
    a closed-form ``phase_integral_at`` and falls back to numerical integration.
    """
    backward_phase_radians: float = PI2
    forward_phase_radians: float = PI2
    max_support_seconds: float = 10.0
    support_search_step_seconds: float = 1e-3
    integration_steps_per_check: int = 64

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("backward_phase_radians",     "Backward phase",  "float", PI2, 0.1,  PI2 * 4, 0, "rad", [], False, "Witness", ".3f"),
            KnobSpec("forward_phase_radians",       "Forward phase",   "float", PI2, 0.1,  PI2 * 4, 0, "rad", [], False, "Witness", ".3f"),
            KnobSpec("max_support_seconds",         "Max support",     "float", 10.0, 0.01, 30.0,  0, "s",   [], True,  "Witness", ".3f"),
            KnobSpec("support_search_step_seconds", "Search step",     "float", 1e-3, 1e-5, 0.1,   0, "s",   [], True,  "Witness", ".5f"),
            KnobSpec("integration_steps_per_check", "Integ. steps",    "int",   64,   4,    512,   4, "",    [], False, "Witness", ".0f"),
        ]


@dataclass(frozen=True)
class DensityPolicy:
    """
    Controls adaptive internal emission density.
    """
    oversampling_factor: float = 16.0
    min_sample_rate: float = 48000.0
    max_sample_rate: float = 10_000_000.0
    derivative_weight: float = 0.5
    support_weight: float = 1.0

    @property
    def min_dt(self) -> float:
        return 1.0 / self.max_sample_rate

    @property
    def max_dt(self) -> float:
        return 1.0 / self.min_sample_rate

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("oversampling_factor", "Oversample",    "float", 16.0,    2.0,     64.0,      0, "\u00d7",  [], True,  "Density", ".1f"),
            KnobSpec("min_sample_rate",     "Min SR",        "float", 48000.0, 8000.0,  192000.0,  0, "Hz",  [], True,  "Density", ".0f"),
            KnobSpec("max_sample_rate",     "Max SR",        "float", 1e7,     96000.0, 4e6,       0, "Hz",  [], True,  "Density", ".0f"),
            KnobSpec("derivative_weight",   "Deriv. weight", "float", 0.5,     0.0,     2.0,       0, "",    [], False, "Density", ".3f"),
            KnobSpec("support_weight",      "Support wt.",   "float", 1.0,     0.0,     4.0,       0, "",    [], False, "Density", ".3f"),
        ]


@dataclass(frozen=True)
class ProjectionPolicy:
    """
    Controls projection from adaptive internal domain to uniform output.
    """
    output_sample_rate: float = 48000.0
    projection_half_support_seconds: float = 0.002
    projection_kernel_steps: int = 256

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("output_sample_rate",              "Output SR",     "float", 48000.0, 8000.0, 192000.0, 0, "Hz", [], True,  "Projection", ".0f"),
            KnobSpec("projection_half_support_seconds", "Half support",  "float", 0.002,   5e-5,   0.05,     0, "s",  [], True,  "Projection", ".5f"),
            KnobSpec("projection_kernel_steps",         "Kernel steps",  "int",   256,     8,      2048,     8, "",   [], False, "Projection", ".0f"),
        ]


@dataclass(frozen=True)
class EmissionRange:
    start_time: float
    end_time: float

    def validate(self) -> None:
        if self.end_time <= self.start_time:
            raise ValueError("EmissionRange must satisfy end_time > start_time")


@dataclass
class SupportBudget:
    backward_seconds: float
    forward_seconds: float


@dataclass
class SamplePoint:
    t: float
    value: complex


@dataclass
class AdaptiveSampleBuffer:
    """
    Holds nonuniform complex analytic samples.
    """
    samples: List[SamplePoint] = field(default_factory=list)
    _times_cache: Optional[List[float]] = field(
        default=None, init=False, repr=False, compare=False
    )

    def add(self, t: float, value: complex) -> None:
        if self.samples and t <= self.samples[-1].t:
            raise ValueError("AdaptiveSampleBuffer requires strictly increasing times")
        self.samples.append(SamplePoint(t=t, value=value))
        # Keep the cache consistent with an O(1) append instead of rebuilding.
        if self._times_cache is not None:
            self._times_cache.append(t)

    def times(self) -> List[float]:
        """Return a cached list of sample times (built once, updated incrementally)."""
        if self._times_cache is None:
            self._times_cache = [s.t for s in self.samples]
        return self._times_cache

    def values(self) -> List[complex]:
        return [s.value for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def evaluate_nearest(self, t: float) -> complex:
        if not self.samples:
            return 0j
        times = self.times()
        idx = bisect_left(times, t)
        if idx <= 0:
            return self.samples[0].value
        if idx >= len(self.samples):
            return self.samples[-1].value
        left = self.samples[idx - 1]
        right = self.samples[idx]
        if abs(left.t - t) <= abs(right.t - t):
            return left.value
        return right.value

    def evaluate_linear(self, t: float) -> complex:
        if not self.samples:
            return 0j

        times = self.times()
        idx = bisect_left(times, t)

        if idx <= 0:
            return self.samples[0].value
        if idx >= len(self.samples):
            return self.samples[-1].value

        left = self.samples[idx - 1]
        right = self.samples[idx]

        if right.t == left.t:
            return left.value

        alpha = (t - left.t) / (right.t - left.t)
        return (1.0 - alpha) * left.value + alpha * right.value


# ============================================================
# Waveform manifolds
# ============================================================

class WaveformManifold(ABC):
    """
    Defines how a voice turns phase into a complex waveform.
    """

    @abstractmethod
    def evaluate(self, phase: float, shape_state: float = 0.0) -> complex:
        raise NotImplementedError


class PureSineManifold(WaveformManifold):
    """
    Simple complex exponential manifold.
    """

    def evaluate(self, phase: float, shape_state: float = 0.0) -> complex:
        return safe_expj(phase)

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return []


class PhaseWarpedManifold(WaveformManifold):
    """
    Keeps the signal analytic but warps the internal phase geometry.
    shape_state controls the amount of warp.
    """

    def __init__(self, harmonic_warp_strength: float = 0.25) -> None:
        self._harmonic_warp_strength = harmonic_warp_strength

    def evaluate(self, phase: float, shape_state: float = 0.0) -> complex:
        warp = self._harmonic_warp_strength * shape_state
        warped_phase = (
            phase
            + warp * math.sin(phase)
            + 0.5 * warp * math.sin(2.0 * phase)
        )
        return safe_expj(warped_phase)

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("harmonic_warp_strength", "Warp strength", "float", 0.25, 0.0, 2.0, 0, "", [], False, "Manifold", ".3f"),
        ]


class HarmonicMeasureManifold(WaveformManifold):
    """
    Analytic waveform built from a controlled harmonic measure.
    This is not band-limited in any strict practical sense unless you
    choose coefficients carefully and oversample sufficiently.
    """

    def __init__(self, max_harmonics: int = 8) -> None:
        if max_harmonics < 1:
            raise ValueError("max_harmonics must be >= 1")
        self._max_harmonics = max_harmonics

    def evaluate(self, phase: float, shape_state: float = 0.0) -> complex:
        # shape_state in [0, 1] recommended
        brightness = clamp(shape_state, 0.0, 1.0)
        acc = 0j
        norm = 0.0

        for n in range(1, self._max_harmonics + 1):
            coeff = (1.0 / n) ** (1.0 - 0.75 * brightness)
            acc += coeff * safe_expj(n * phase)
            norm += coeff

        if norm == 0.0:
            return 0j
        return acc / norm

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("max_harmonics", "Max harmonics", "int", 8, 1, 32, 1, "", [], False, "Manifold", ".0f"),
        ]


# ============================================================
# Phase paths
# ============================================================

class PhasePath(ABC):
    """
    Continuous phase trajectory. Frequency is derivative of phase.
    """

    @abstractmethod
    def phase(self, t: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def frequency_hz(self, t: float) -> float:
        raise NotImplementedError

    def angular_frequency(self, t: float) -> float:
        return PI2 * self.frequency_hz(t)

    def chirp_rate_hz_per_s(self, t: float) -> float:
        return differentiate_central(self.frequency_hz, t)

    def accumulated_phase(self, t0: float, t1: float, steps: int = 32) -> float:
        return integrate_trapezoid(self.angular_frequency, t0, t1, steps=steps)


class ExponentialDecayPhasePath(PhasePath):
    """
    f(t) = f0 * exp(-t / tau)
    phase(t) = phase0 + 2π f0 tau (1 - exp(-t / tau))
    """

    def __init__(self, initial_frequency_hz: float, tau_seconds: float, phase0: float = 0.0) -> None:
        if initial_frequency_hz < 0.0:
            raise ValueError("initial_frequency_hz must be >= 0")
        if tau_seconds <= 0.0:
            raise ValueError("tau_seconds must be > 0")
        self._f0 = initial_frequency_hz
        self._tau = tau_seconds
        self._phase0 = phase0

    def frequency_hz(self, t: float) -> float:
        return self._f0 * math.exp(-t / self._tau)

    def chirp_rate_hz_per_s(self, t: float) -> float:
        # d/dt [f0 * exp(-t/tau)] = -f0/tau * exp(-t/tau)
        return -self._f0 / self._tau * math.exp(-t / self._tau)

    def phase(self, t: float) -> float:
        return self._phase0 + PI2 * self._f0 * self._tau * (1.0 - math.exp(-t / self._tau))

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("initial_frequency_hz", "f\u2080",    "float", 440.0, 1.0,  20000.0, 0, "Hz",  [], True,  "Exp Decay Path", ".2f"),
            KnobSpec("tau_seconds",          "\u03c4",      "float", 0.5,  0.01, 10.0,    0, "s",   [], True,  "Exp Decay Path", ".3f"),
            KnobSpec("phase0",               "Phase\u2080", "float", 0.0, -PI2,  PI2,     0, "rad", [], False, "Exp Decay Path", ".3f"),
        ]


class PowerLawDecayPhasePath(PhasePath):
    """
    f(t) = f0 / (1 + t / tau)^p
    """

    def __init__(
        self,
        initial_frequency_hz: float,
        tau_seconds: float,
        power: float,
        phase0: float = 0.0,
    ) -> None:
        if initial_frequency_hz < 0.0:
            raise ValueError("initial_frequency_hz must be >= 0")
        if tau_seconds <= 0.0:
            raise ValueError("tau_seconds must be > 0")
        if power <= 0.0:
            raise ValueError("power must be > 0")

        self._f0 = initial_frequency_hz
        self._tau = tau_seconds
        self._power = power
        self._phase0 = phase0

    def frequency_hz(self, t: float) -> float:
        if t < -self._tau:
            raise ValueError("time below domain for PowerLawDecayPhasePath")
        return self._f0 / ((1.0 + t / self._tau) ** self._power)

    def chirp_rate_hz_per_s(self, t: float) -> float:
        # d/dt [f0 / (1 + t/tau)^p] = -f0*p / (tau * (1 + t/tau)^(p+1))
        if t < -self._tau:
            raise ValueError("time below domain for PowerLawDecayPhasePath")
        return -self._f0 * self._power / (
            self._tau * (1.0 + t / self._tau) ** (self._power + 1.0)
        )

    def phase(self, t: float) -> float:
        if t < -self._tau:
            raise ValueError("time below domain for PowerLawDecayPhasePath")

        p = self._power
        tau = self._tau
        f0 = self._f0

        if abs(p - 1.0) < 1e-12:
            integral = tau * math.log(1.0 + t / tau)
        else:
            integral = tau * (((1.0 + t / tau) ** (1.0 - p) - 1.0) / (1.0 - p))

        return self._phase0 + PI2 * f0 * integral

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("initial_frequency_hz", "f\u2080",    "float", 440.0, 1.0,  20000.0, 0, "Hz",  [], True,  "Power Decay Path", ".2f"),
            KnobSpec("tau_seconds",          "\u03c4",      "float", 0.5,  0.01, 10.0,    0, "s",   [], True,  "Power Decay Path", ".3f"),
            KnobSpec("power",                "Power",  "float", 1.0,  0.1,  8.0,     0, "",    [], False, "Power Decay Path", ".2f"),
            KnobSpec("phase0",               "Phase\u2080", "float", 0.0, -PI2,  PI2,     0, "rad", [], False, "Power Decay Path", ".3f"),
        ]


class ConstantPhasePath(PhasePath):
    """
    Stable-pitch path: f(t) = f0 (constant).

    phase(t) = phase0 + 2π·f0·t
    chirp_rate = 0  (exact, no numerical fallback needed)
    """

    def __init__(self, frequency_hz: float, phase0: float = 0.0) -> None:
        if frequency_hz < 0.0:
            raise ValueError("frequency_hz must be >= 0")
        self._f0     = frequency_hz
        self._phase0 = phase0

    def frequency_hz(self, t: float) -> float:
        return self._f0

    def chirp_rate_hz_per_s(self, t: float) -> float:
        return 0.0

    def phase(self, t: float) -> float:
        return self._phase0 + PI2 * self._f0 * t

    def accumulated_phase(self, t0: float, t1: float, steps: int = 32) -> float:
        return PI2 * self._f0 * (t1 - t0)

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("frequency_hz", "Frequency", "float", 440.0, 1.0, 20000.0, 0, "Hz",  [], True,  "Phase Path", ".2f"),
            KnobSpec("phase0",       "Phase\u2080",    "float", 0.0, -PI2, PI2,     0, "rad", [], False, "Phase Path", ".3f"),
        ]


class LinearChirpPhasePath(PhasePath):
    """
    Linear-chirp path: f(t) = f_start + chirp_rate * (t - t_start)

    Exact closed-form phase and chirp rate — no numerical integration needed.
    This is the canonical PhasePath for a ChirpSegment from tf_ray_engine:
      phase(t) = phase_at_start + 2π * (f_start*(t-t_start) + chirp_rate/2*(t-t_start)²)

    chirp_rate_hz_per_s is the real-time df/dt (can be negative for descending chirps).
    t_start anchors the phase zero — typically the segment's t_start in real time.
    """

    def __init__(
        self,
        f_start:             float,
        chirp_rate_hz_per_s: float,
        phase_at_start:      float = 0.0,
        t_start:             float = 0.0,
    ) -> None:
        if f_start < 0.0:
            raise ValueError("f_start must be >= 0")
        self._f0   = float(f_start)
        self._cr   = float(chirp_rate_hz_per_s)
        self._phi0 = float(phase_at_start)
        self._t0   = float(t_start)

    def frequency_hz(self, t: float) -> float:
        return self._f0 + self._cr * (t - self._t0)

    def chirp_rate_hz_per_s(self, t: float) -> float:   # type: ignore[override]
        return self._cr

    def phase(self, t: float) -> float:
        dt = t - self._t0
        return self._phi0 + PI2 * (self._f0 * dt + 0.5 * self._cr * dt * dt)

    def accumulated_phase(self, t0: float, t1: float, steps: int = 32) -> float:
        # Exact closed form — steps parameter is intentionally ignored
        return self.phase(t1) - self.phase(t0)

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("f_start",             "f start",    "float", 440.0,  1.0,     20000.0, 0, "Hz",   [], True,  "Chirp Path", ".2f"),
            KnobSpec("chirp_rate_hz_per_s", "Chirp rate", "float", 0.0,   -5000.0,  5000.0,  0, "Hz/s", [], False, "Chirp Path", ".1f"),
            KnobSpec("phase_at_start",      "Phase\u2080",     "float", 0.0,   -PI2,     PI2,     0, "rad",  [], False, "Chirp Path", ".3f"),
            KnobSpec("t_start",             "t start",    "float", 0.0,    0.0,     10.0,    0, "s",    [], False, "Chirp Path", ".3f"),
        ]


# ============================================================
# Drift models
# ============================================================

class DriftModel(ABC):
    """
    Drift is additive in frequency space by default.

    ``_fallback_integration_steps`` controls how many trapezoid taps are used
    by ``phase_integral_at`` when a subclass does not supply a closed-form
    override.  Set it on the instance after construction, or pass
    ``WitnessThresholds.integration_steps_per_check`` through to it.
    """

    _fallback_integration_steps: int = 64

    @abstractmethod
    def frequency_offset_hz(self, t: float, harmonic_index: int) -> float:
        raise NotImplementedError

    def phase_integral_at(self, t: float, harmonic_index: int) -> float:
        """Integral of 2π·frequency_offset_hz from 0 to t.

        Subclasses with a closed form should override this.
        The default falls back to numerical trapezoid integration using
        ``self._fallback_integration_steps`` taps.
        """
        return PI2 * integrate_trapezoid(
            lambda tau: self.frequency_offset_hz(tau, harmonic_index),
            0.0, t, steps=self._fallback_integration_steps,
        )

    def chirp_rate_hz_per_s(self, t: float, harmonic_index: int) -> float:
        """d/dt of frequency_offset_hz.  Override with closed form where possible."""
        return differentiate_central(
            lambda tau: self.frequency_offset_hz(tau, harmonic_index), t
        )


class NullDriftModel(DriftModel):
    def frequency_offset_hz(self, t: float, harmonic_index: int) -> float:
        return 0.0

    def phase_integral_at(self, t: float, harmonic_index: int) -> float:
        return 0.0

    def chirp_rate_hz_per_s(self, t: float, harmonic_index: int) -> float:
        return 0.0

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return []


class SmoothSinusoidalDriftModel(DriftModel):
    """
    Very gentle smooth drift. Intended as a placeholder for more elaborate latent drift fields.
    """

    def __init__(
        self,
        base_rate_hz: float = 0.1,
        max_offset_hz: float = 0.5,
        harmonic_scaling_power: float = 1.0,
    ) -> None:
        self._base_rate_hz = base_rate_hz
        self._max_offset_hz = max_offset_hz
        self._harmonic_scaling_power = harmonic_scaling_power

    def frequency_offset_hz(self, t: float, harmonic_index: int) -> float:
        scale = 1.0 / (harmonic_index ** self._harmonic_scaling_power)
        return self._max_offset_hz * scale * math.sin(PI2 * self._base_rate_hz * t)

    def phase_integral_at(self, t: float, harmonic_index: int) -> float:
        """Closed-form: ∫₀ᵗ 2π·f_drift(τ,h) dτ

        = 2π · max_offset · scale · (1 − cos(2π·rate·t)) / (2π·rate)
        = max_offset · scale · (1 − cos(2π·rate·t)) / rate
        """
        rate = self._base_rate_hz
        if rate == 0.0:
            return 0.0
        scale = 1.0 / (harmonic_index ** self._harmonic_scaling_power)
        return self._max_offset_hz * scale * (1.0 - math.cos(PI2 * rate * t)) / rate

    def chirp_rate_hz_per_s(self, t: float, harmonic_index: int) -> float:
        # d/dt [max_offset * scale * sin(2π*rate*t)] = max_offset * scale * 2π*rate * cos(2π*rate*t)
        scale = 1.0 / (harmonic_index ** self._harmonic_scaling_power)
        return self._max_offset_hz * scale * PI2 * self._base_rate_hz * math.cos(
            PI2 * self._base_rate_hz * t
        )

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("base_rate_hz",           "Rate",          "float", 0.1, 0.0,  20.0,  0, "Hz", [], True,  "Drift", ".3f"),
            KnobSpec("max_offset_hz",          "Max offset",    "float", 0.5, 0.0,  500.0, 0, "Hz", [], True,  "Drift", ".2f"),
            KnobSpec("harmonic_scaling_power", "Harm. scale",   "float", 1.0, 0.0,  4.0,   0, "",   [], False, "Drift", ".2f"),
        ]


# ============================================================
# Envelope / shape fields
# ============================================================

class TimeField(ABC):
    @abstractmethod
    def value(self, t: float) -> float:
        raise NotImplementedError


class ConstantField(TimeField):
    def __init__(self, constant: float) -> None:
        self._constant = constant

    def value(self, t: float) -> float:
        return self._constant


class ExponentialEnvelope(TimeField):
    def __init__(self, initial: float = 1.0, tau_seconds: float = 1.0) -> None:
        if tau_seconds <= 0:
            raise ValueError("tau_seconds must be > 0")
        self._initial = initial
        self._tau = tau_seconds

    def value(self, t: float) -> float:
        if t < 0:
            return self._initial
        return self._initial * math.exp(-t / self._tau)


class SigmoidField(TimeField):
    def __init__(self, low: float, high: float, center: float, width: float) -> None:
        if width <= 0:
            raise ValueError("width must be > 0")
        self._low = low
        self._high = high
        self._center = center
        self._width = width

    def value(self, t: float) -> float:
        u = 1.0 / (1.0 + math.exp(-(t - self._center) / self._width))
        return self._low + (self._high - self._low) * u

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("low",    "Low",    "float", 0.0, 0.0, 1.0,  0, "",  [], False, "Sigmoid", ".3f"),
            KnobSpec("high",   "High",   "float", 1.0, 0.0, 1.0,  0, "",  [], False, "Sigmoid", ".3f"),
            KnobSpec("center", "Center", "float", 0.5, 0.0, 10.0, 0, "s", [], False, "Sigmoid", ".3f"),
            KnobSpec("width",  "Width",  "float", 0.1, 1e-4, 1.0, 0, "s", [], True,  "Sigmoid", ".4f"),
        ]


class LinearRampField(TimeField):
    """
    Linearly interpolates from *start_value* to *end_value* over [t0, t1].
    Clamps to the endpoint values outside that range.
    """

    def __init__(
        self,
        start_value: float,
        end_value:   float,
        t0:          float = 0.0,
        t1:          float = 1.0,
    ) -> None:
        if t1 <= t0:
            raise ValueError("LinearRampField requires t1 > t0")
        self._v0 = float(start_value)
        self._v1 = float(end_value)
        self._t0 = float(t0)
        self._t1 = float(t1)

    def value(self, t: float) -> float:
        if t <= self._t0:
            return self._v0
        if t >= self._t1:
            return self._v1
        alpha = (t - self._t0) / (self._t1 - self._t0)
        return self._v0 + alpha * (self._v1 - self._v0)


class PiecewiseLinearField(TimeField):
    """
    Arbitrary piecewise-linear envelope defined by a list of (time, value) knots.

    Knots must be in strictly increasing time order.
    Extrapolates flat (holds the first/last value) outside the knot range.

    Example — a simple trapezoid::

        PiecewiseLinearField([(0.0, 0.0), (0.02, 1.0), (0.8, 1.0), (1.0, 0.0)])
    """

    def __init__(self, knots: List[Tuple[float, float]]) -> None:
        if len(knots) < 2:
            raise ValueError("PiecewiseLinearField needs at least 2 knots")
        ts = [k[0] for k in knots]
        for i in range(1, len(ts)):
            if ts[i] <= ts[i - 1]:
                raise ValueError("PiecewiseLinearField knot times must be strictly increasing")
        self._knots: List[Tuple[float, float]] = list(knots)

    def value(self, t: float) -> float:
        if t <= self._knots[0][0]:
            return self._knots[0][1]
        if t >= self._knots[-1][0]:
            return self._knots[-1][1]
        # binary search for bracketing pair
        lo, hi = 0, len(self._knots) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._knots[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        t0, v0 = self._knots[lo]
        t1, v1 = self._knots[hi]
        alpha = (t - t0) / (t1 - t0)
        return v0 + alpha * (v1 - v0)


class PiecewiseSplineField(TimeField):
    """
    Smooth cubic spline envelope through a list of (time, value) knots.

    Uses a natural cubic spline (second derivative = 0 at both endpoints).
    The curve passes exactly through every knot.  Unlike piecewise-linear,
    it may overshoot between knots — useful for smooth, flowing shapes.

    Extrapolates flat (holds the first/last value) outside the knot range.

    Requires at least 3 knots (2 knots fall back to linear).

    Example — smooth bell::

        PiecewiseSplineField([(0.0, 0.0), (0.1, 0.8), (0.5, 1.0), (0.9, 0.8), (1.0, 0.0)])
    """

    def __init__(self, knots: List[Tuple[float, float]]) -> None:
        if len(knots) < 2:
            raise ValueError("PiecewiseSplineField needs at least 2 knots")
        ts = [k[0] for k in knots]
        for i in range(1, len(ts)):
            if ts[i] <= ts[i - 1]:
                raise ValueError("PiecewiseSplineField knot times must be strictly increasing")
        self._t0 = ts[0]
        self._t1 = ts[-1]
        self._v0 = knots[0][1]
        self._v1 = knots[-1][1]
        from scipy.interpolate import CubicSpline
        import numpy as np
        self._spline = CubicSpline(
            np.array(ts, dtype=float),
            np.array([k[1] for k in knots], dtype=float),
            bc_type="natural",
        )

    def value(self, t: float) -> float:
        if t <= self._t0:
            return self._v0
        if t >= self._t1:
            return self._v1
        return float(self._spline(t))


class PiecewiseMonotoneField(TimeField):
    """
    Smooth monotone-cubic (PCHIP) envelope through a list of (time, value) knots.

    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) passes exactly
    through every knot and is *guaranteed not to overshoot* between them — the
    value stays within the range of neighbouring knots.  This makes it the
    preferred choice for amplitude envelopes where a natural cubic spline would
    ring or dip below zero.

    Extrapolates flat (holds the first/last value) outside the knot range.

    Example — smooth ADSR-like curve::

        PiecewiseMonotoneField([(0.0, 0.0), (0.01, 1.0), (0.1, 0.7), (0.8, 0.7), (1.0, 0.0)])
    """

    def __init__(self, knots: List[Tuple[float, float]]) -> None:
        if len(knots) < 2:
            raise ValueError("PiecewiseMonotoneField needs at least 2 knots")
        ts = [k[0] for k in knots]
        for i in range(1, len(ts)):
            if ts[i] <= ts[i - 1]:
                raise ValueError("PiecewiseMonotoneField knot times must be strictly increasing")
        self._t0 = ts[0]
        self._t1 = ts[-1]
        self._v0 = knots[0][1]
        self._v1 = knots[-1][1]
        from scipy.interpolate import PchipInterpolator
        import numpy as np
        self._spline = PchipInterpolator(
            np.array(ts, dtype=float),
            np.array([k[1] for k in knots], dtype=float),
        )

    def value(self, t: float) -> float:
        if t <= self._t0:
            return self._v0
        if t >= self._t1:
            return self._v1
        return float(self._spline(t))


class ADSREnvelope(TimeField):
    """
    Classic 4-stage amplitude envelope (Attack → Decay → Sustain → Release).

    All time parameters are seconds measured from t=0 (local note on).

    Stage boundaries
    ────────────────
    [0,               attack_time)             — ramp  0 → peak_level
    [attack_time,     attack_time+decay_time)  — ramp  peak_level → sustain_level
    [attack+decay,    release_start)           — hold  sustain_level
    [release_start,   release_start+release)   — ramp  sustain_level → 0
    [release_start+release, …)                 — 0

    *release_start* defaults to ``note_duration - release_time`` so the tail
    ends exactly at the note boundary.  Pass ``note_duration=None`` (default)
    to omit the release stage entirely (sustain holds forever).

    If the note is shorter than attack+decay, those two stages are
    time-compressed proportionally so they still fit within the available
    pre-release window.
    """

    def __init__(
        self,
        attack_time:   float = 0.005,
        decay_time:    float = 0.04,
        sustain_level: float = 0.75,
        release_time:  float = 0.08,
        peak_level:    float = 1.0,
        note_duration: Optional[float] = None,
    ) -> None:
        if attack_time < 0.0:
            raise ValueError("attack_time must be >= 0")
        if decay_time < 0.0:
            raise ValueError("decay_time must be >= 0")
        if release_time < 0.0:
            raise ValueError("release_time must be >= 0")
        if not (0.0 <= sustain_level <= 1.0):
            raise ValueError("sustain_level must be in [0, 1]")
        if peak_level < 0.0:
            raise ValueError("peak_level must be >= 0")

        self._pk = float(peak_level)
        self._s  = float(sustain_level)
        self._r  = float(release_time)

        # Compute the window available for A+D before the release ramp starts.
        if note_duration is not None:
            avail_ad = max(0.0, float(note_duration) - float(release_time))
            self._rel_start: Optional[float] = float(note_duration) - float(release_time)
        else:
            avail_ad = None
            self._rel_start = None

        # Compress A+D proportionally if they would exceed available time.
        raw_ad = float(attack_time) + float(decay_time)
        if avail_ad is not None and raw_ad > avail_ad and raw_ad > 0.0:
            scale = avail_ad / raw_ad
            self._a = float(attack_time) * scale
            self._d = float(decay_time) * scale
        else:
            self._a = float(attack_time)
            self._d = float(decay_time)

    def value(self, t: float) -> float:
        if t < 0.0:
            return 0.0

        # --- Attack ---
        if t < self._a:
            return self._pk * (t / self._a) if self._a > 0.0 else self._pk

        # --- Decay ---
        t_post_a = t - self._a
        if t_post_a < self._d:
            frac = t_post_a / self._d if self._d > 0.0 else 1.0
            return self._pk + (self._s - self._pk) * frac

        # --- Sustain / Release ---
        if self._rel_start is None or t < self._rel_start:
            return self._s

        # --- Release ---
        t_rel = t - self._rel_start
        if self._r <= 0.0 or t_rel >= self._r:
            return 0.0
        return self._s * (1.0 - t_rel / self._r)

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            KnobSpec("attack_time",   "Attack",  "float", 0.005, 0.001, 2.0, 0, "s", [], True,  "ADSR", ".4f"),
            KnobSpec("decay_time",    "Decay",   "float", 0.04,  0.001, 2.0, 0, "s", [], True,  "ADSR", ".4f"),
            KnobSpec("sustain_level", "Sustain", "float", 0.75,  0.0,   1.0, 0, "",  [], False, "ADSR", ".3f"),
            KnobSpec("release_time",  "Release", "float", 0.08,  0.001, 4.0, 0, "s", [], True,  "ADSR", ".4f"),
            KnobSpec("peak_level",    "Peak",    "float", 1.0,   0.0,   2.0, 0, "",  [], False, "ADSR", ".3f"),
        ]


# ============================================================
# Complex-domain envelopes
# ============================================================

class ComplexEnvelope(ABC):
    """
    Post-lattice complex-domain shaping hook.

    Applied to the fully-summed instantaneous analytic value z(t) *before*
    the sample enters ``AdaptiveSampleBuffer``.  This is the earliest possible
    place to reshape energy after the harmonic voices have been combined.

    The analytic nature of z is preserved by any linear complex operation,
    which includes all three concrete forms below.

    Three canonical forms
    ─────────────────────
    AmplitudeEnvelopeShaper   z → E(t) · z            (E ∈ ℝ⁺, |z| scaled, arg unchanged)
    PhaseRotationEnvelope     z → z · e^{jΘ(t)}       (|z| unchanged, arg rotated)
    ComplexGainEnvelope       z → z · M(t) · e^{jΘ(t)} (both simultaneously)

    Any ``TimeField`` (including ``ADSREnvelope``) can be wrapped in
    ``AmplitudeEnvelopeShaper`` to become a ``ComplexEnvelope``.

    Use ``EnvelopeChain`` to compose multiple stages in sequence.
    """

    @abstractmethod
    def evaluate(self, t: float, z: complex) -> complex:
        raise NotImplementedError


class AmplitudeEnvelopeShaper(ComplexEnvelope):
    """
    Real-valued amplitude shaping in the analytic complex domain.

    Multiplies z by a real scalar E(t):

        z  →  E(t) · z

    The phase/frequency structure is untouched — only the instantaneous energy
    envelope changes.  z remains analytic.

    Any ``TimeField`` works as the amplitude field, including ``ADSREnvelope``,
    ``ExponentialEnvelope``, ``PiecewiseLinearField``, etc.
    """

    def __init__(self, amplitude_field: TimeField) -> None:
        self._field = amplitude_field

    def evaluate(self, t: float, z: complex) -> complex:
        return self._field.value(t) * z


class PhaseRotationEnvelope(ComplexEnvelope):
    """
    Time-varying phase rotation without amplitude change.

    Multiplies z by the unit phasor e^{jΘ(t)}:

        z  →  z · e^{jΘ(t)}

    |z| is preserved exactly; arg(z) gains a time-varying offset Θ(t).

    The instantaneous frequency of the resulting signal becomes
    f_orig(t) + dΘ/dt / 2π, so this is analytic-domain FM.

    Typical uses
    ────────────
    Vibrato   — ``SmoothSinusoidalDriftModel`` already handles this per-voice;
                use this envelope for a post-summing global FM layer.
    Portamento — ``LinearRampField(start_phase, end_phase)`` shifts the whole
                signal's frequency axis smoothly over a note transition.
    Phase seed — ``ConstantField(offset)`` injects a fixed global phase offset
                without touching amplitude.
    """

    def __init__(self, angle_field: TimeField) -> None:
        self._angle_field = angle_field

    def evaluate(self, t: float, z: complex) -> complex:
        return z * cmath.rect(1.0, self._angle_field.value(t))


class ComplexGainEnvelope(ComplexEnvelope):
    """
    Full complex multiplication: simultaneously shape amplitude AND rotate phase.

    The gain function is G(t) = M(t) · e^{jΘ(t)}, applied as:

        z  →  z · G(t)

    M(t) = magnitude modulation (any TimeField, e.g. ``ADSREnvelope``).
    Θ(t) = phase modulation in radians (any TimeField; defaults to zero).

    Special cases
    ─────────────
    Θ = 0    →  same as AmplitudeEnvelopeShaper(magnitude_field)
    M = 1    →  same as PhaseRotationEnvelope(phase_field)
    M = ADSR, Θ = sinusoidal → amplitude-shaped vibrato that deepens with loudness
    """

    def __init__(
        self,
        magnitude_field: TimeField,
        phase_field:     Optional[TimeField] = None,
    ) -> None:
        self._mag   = magnitude_field
        self._phase = phase_field if phase_field is not None else ConstantField(0.0)

    def evaluate(self, t: float, z: complex) -> complex:
        return z * cmath.rect(self._mag.value(t), self._phase.value(t))


class EnvelopeChain(ComplexEnvelope):
    """
    Apply a sequence of ``ComplexEnvelope`` instances in order.

    The output of each stage feeds the next:

        z  →  E₁(z)  →  E₂(z)  →  …  →  Eₙ(z)

    Example — ADSR amplitude with sinusoidal vibrato on top::

        EnvelopeChain(
            AmplitudeEnvelopeShaper(ADSREnvelope(attack_time=0.01, ...)),
            PhaseRotationEnvelope(SmoothSinusoidalDriftModel(...)),
        )
    """

    def __init__(self, *envelopes: ComplexEnvelope) -> None:
        self._chain = list(envelopes)

    def evaluate(self, t: float, z: complex) -> complex:
        for env in self._chain:
            z = env.evaluate(t, z)
        return z


# ============================================================
# Lattice voice
# ============================================================

class LatticeVoice:
    """
    A harmonic descendant of a shared master path.
    """

    def __init__(
        self,
        name: str,
        harmonic_index: int,
        master_phase_path: PhasePath,
        waveform_manifold: WaveformManifold,
        amplitude_field: TimeField,
        shape_field: Optional[TimeField] = None,
        drift_model: Optional[DriftModel] = None,
        gain: float = 1.0,
        phase_offset: float = 0.0,
    ) -> None:
        if harmonic_index < 1:
            raise ValueError("harmonic_index must be >= 1")

        self._name = name
        self._harmonic_index = harmonic_index
        self._master_phase_path = master_phase_path
        self._waveform_manifold = waveform_manifold
        self._amplitude_field = amplitude_field
        self._shape_field = shape_field or ConstantField(0.0)
        self._drift_model = drift_model or NullDriftModel()
        self._gain = gain
        self._phase_offset = phase_offset

    @property
    def name(self) -> str:
        return self._name

    @property
    def harmonic_index(self) -> int:
        return self._harmonic_index

    def base_phase(self, t: float) -> float:
        return self._harmonic_index * self._master_phase_path.phase(t) + self._phase_offset

    def base_frequency_hz(self, t: float) -> float:
        return self._harmonic_index * self._master_phase_path.frequency_hz(t)

    def effective_frequency_hz(self, t: float) -> float:
        return self.base_frequency_hz(t) + self._drift_model.frequency_offset_hz(t, self._harmonic_index)

    def effective_angular_frequency(self, t: float) -> float:
        return PI2 * self.effective_frequency_hz(t)

    def chirp_rate_hz_per_s(self, t: float) -> float:
        """Analytic instantaneous chirp rate (Hz/s) using path + drift closed forms."""
        return (
            self._harmonic_index * self._master_phase_path.chirp_rate_hz_per_s(t)
            + self._drift_model.chirp_rate_hz_per_s(t, self._harmonic_index)
        )

    def approximate_effective_phase(self, t: float) -> float:
        """
        Keeps master phase exact, then adds a drift integral.
        This is an approximation if drift is not analytically integrated elsewhere.
        """
        drift_phase = self._drift_model.phase_integral_at(t, self._harmonic_index)
        return self.base_phase(t) + drift_phase

    def amplitude(self, t: float) -> float:
        return self._gain * self._amplitude_field.value(t)

    def shape_state(self, t: float) -> float:
        return self._shape_field.value(t)

    def evaluate_complex(self, t: float) -> complex:
        amp = self.amplitude(t)
        phase = self.approximate_effective_phase(t)
        shape = self.shape_state(t)
        return amp * self._waveform_manifold.evaluate(phase, shape)


# ============================================================
# Synth graph / voice collection
# ============================================================

class HarmonicLattice:
    """
    Holds a coherent family of lattice voices sharing a master path.
    """

    def __init__(self, voices: Sequence[LatticeVoice]) -> None:
        if not voices:
            raise ValueError("HarmonicLattice requires at least one voice")
        self._voices = list(voices)
        # The voice with the lowest harmonic index always has the lowest
        # instantaneous frequency (all share the same master path; drift is tiny).
        # Cache it so DensityPlanner / WitnessPlanner can query it cheaply.
        self._min_harmonic_voice: LatticeVoice = min(
            self._voices, key=lambda v: v.harmonic_index
        )

    @property
    def voices(self) -> Sequence[LatticeVoice]:
        return self._voices

    def evaluate_complex(self, t: float) -> complex:
        acc = 0j
        for voice in self._voices:
            acc += voice.evaluate_complex(t)
        return acc

    def max_abs_frequency_hz(self, t: float) -> float:
        return max(abs(v.effective_frequency_hz(t)) for v in self._voices)

    def max_abs_chirp_like_measure(self, t: float) -> float:
        """Instantaneous chirp magnitude, using analytic derivatives where available."""
        return max(abs(v.chirp_rate_hz_per_s(t)) for v in self._voices)


# ============================================================
# Output admissibility planning
# ============================================================

@dataclass(frozen=True)
class VoiceAdmissibilityVerdict:
    """
    Describes how much a voice must be attenuated and/or warp-capped to
    remain admissible under a target output sample rate.

    gain_taper:
        Multiplicative gain attenuation in [0, 1].  1.0 = untouched.
    warp_cap:
        Maximum |warp·shape_state| product the manifold should see.
        Relevant only for PhaseWarpedManifold.
    harmonic_gain:
        Per-harmonic-index gain taper.  None means the voice passes all harmonics at full gain.
    """
    gain_taper: float
    warp_cap: float
    harmonic_gain: float  # in [0, 1]


class OutputAdmissibilityPlanner:
    """
    Estimates whether each ``LatticeVoice`` in a lattice will produce
    spectral content above the output Nyquist and derives taper values
    that attenuate the excess before projection.

    The estimate is intentionally conservative: it models sideband spray
    from drift and phase-warp as a fixed margin above the nominal voice
    frequency, then soft-clips any energy whose ceiling exceeds ``nyquist``.

    Margin conventions
    ------------------
    ``drift_margin_factor``
        Fraction of the voice's nominal frequency added as an upper-sideband
        margin to account for sinusoidal drift.  E.g. 0.05 = ±5 % drift
        sidebands.

    ``warp_margin_factor``
        Fraction of the voice's nominal frequency added per unit of warp
        strength × shape_state.  A warp of 0.25 at shape_state 1.0 with a
        margin factor of 2.0 adds 0.5× the nominal frequency.

    ``soft_knee_width``
        Width (in Hz) of the soft-knee rolloff at the Nyquist boundary.
        Energy within [nyquist − knee, nyquist] is tapered smoothly to zero.
    """

    def __init__(
        self,
        output_sample_rate: float,
        drift_margin_factor: float = 0.05,
        warp_margin_factor: float = 2.0,
        soft_knee_width: float = 500.0,
    ) -> None:
        if output_sample_rate <= 0.0:
            raise ValueError("output_sample_rate must be > 0")
        self._output_sr      = output_sample_rate
        self._nyquist        = output_sample_rate * 0.5
        self._drift_margin   = drift_margin_factor
        self._warp_margin    = warp_margin_factor
        self._soft_knee_width = max(1.0, soft_knee_width)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_spectral_ceiling(
        self, voice: LatticeVoice, t: float
    ) -> float:
        """
        Upper bound on the spectral content produced by *voice* at time *t*,
        accounting for drift sidebands and phase-warp harmonics.
        """
        f_nom   = abs(voice.effective_frequency_hz(t))
        drift_offset = abs(voice._drift_model.frequency_offset_hz(t, voice.harmonic_index))

        # Warp sideband spray: PhaseWarpedManifold with warp·shape_state = W
        # adds energy near (1 + W) × f_nom as a rough upper bound.
        warp_strength = getattr(voice._waveform_manifold,
                                '_harmonic_warp_strength', 0.0)
        shape = voice.shape_state(t)
        warp_ceiling_extra = self._warp_margin * warp_strength * abs(shape) * f_nom

        drift_ceiling_extra = (
            self._drift_margin * f_nom + drift_offset
        )

        return f_nom + max(drift_ceiling_extra, warp_ceiling_extra)

    def compute_verdict(self, voice: LatticeVoice, t: float) -> VoiceAdmissibilityVerdict:
        """
        Return a ``VoiceAdmissibilityVerdict`` for *voice* at time *t*.
        """
        ceiling = self.estimate_spectral_ceiling(voice, t)
        nyquist  = self._nyquist
        knee     = self._soft_knee_width

        # Gain taper: soft-knee rolloff from 1→0 over [nyquist-knee, nyquist].
        # Content above nyquist is zeroed; content in the knee is smoothly faded.
        if ceiling <= nyquist - knee:
            gain_taper = 1.0
        elif ceiling >= nyquist:
            gain_taper = 0.0
        else:
            # Linear knee taper (could be raised to a power for smoother rolloff)
            gain_taper = (nyquist - ceiling) / knee

        # Warp cap: if the voice is in or near the knee, reduce warp to
        # prevent further sideband spray.  The cap is proportional to headroom.
        warp_strength = getattr(voice._waveform_manifold,
                                '_harmonic_warp_strength', 0.0)
        f_nom = max(1e-12, abs(voice.effective_frequency_hz(t)))
        headroom = max(0.0, nyquist - f_nom)
        # Maximum |warp| that keeps warp ceiling < nyquist:
        # warp * shape * f_nom * warp_margin < headroom
        shape = abs(voice.shape_state(t))
        denom = self._warp_margin * max(1e-12, shape) * f_nom
        warp_cap = min(warp_strength, headroom / denom) if denom > 0 else warp_strength

        return VoiceAdmissibilityVerdict(
            gain_taper=clamp(gain_taper, 0.0, 1.0),
            warp_cap=clamp(warp_cap, 0.0, max(warp_strength, 1.0)),
            harmonic_gain=clamp(gain_taper, 0.0, 1.0),
        )


# ============================================================
# Complex pre-projection filter
# ============================================================

class ComplexPreProjectionFilter:
    """
    Applies admissibility-guided filtering to each voice contribution
    **before** samples are summed and **before** projection to a real
    uniform output grid.

    This is a two-stage filter:

    Stage 1 — parameter domain:
        Evaluates each voice under a potentially modified warp strength
        (derived from ``VoiceAdmissibilityVerdict.warp_cap``) so that phase
        spray from warp is bounded before the analytic sample is even computed.

    Stage 2 — complex sample domain:
        Multiplies the voice's complex contribution by
        ``VoiceAdmissibilityVerdict.gain_taper``, attenuating energy that
        would alias at the target output rate.

    The filter is **stateless and causal-free**: every call to
    ``evaluate_complex`` is independent.  It can therefore be inserted into
    any emit loop without changing the witness or density contracts.

    Usage
    -----
    Pass an instance to ``WitnessAwareSynthDriver`` (or directly to
    ``AdaptiveEmitter``) to activate pre-projection filtering.  If ``None``
    is supplied (the default), the pipeline runs unfiltered.
    """

    def __init__(self, planner: OutputAdmissibilityPlanner) -> None:
        self._planner = planner

    def evaluate_lattice_filtered(
        self, lattice: HarmonicLattice, t: float
    ) -> complex:
        """
        Evaluate all voices with per-voice admissibility gain taper applied.

        The warp cap is enforced by temporarily substituting the manifold's
        ``_harmonic_warp_strength`` attribute when the verdict requires it.
        The substitution is restored before returning, so the voice object
        is not permanently mutated.
        """
        acc = 0j
        for voice in lattice.voices:
            verdict = self._planner.compute_verdict(voice, t)

            # Stage 1: enforce warp cap via temporary manifold attribute override.
            manifold = voice._waveform_manifold
            original_warp = getattr(manifold, '_harmonic_warp_strength', None)
            if original_warp is not None and original_warp > verdict.warp_cap:
                manifold._harmonic_warp_strength = verdict.warp_cap

            # Stage 2: evaluate voice, then apply gain taper.
            z = voice.evaluate_complex(t)
            z *= verdict.gain_taper

            # Restore manifold warp.
            if original_warp is not None and original_warp > verdict.warp_cap:
                manifold._harmonic_warp_strength = original_warp

            acc += z
        return acc


# ============================================================
# Witness planning
# ============================================================

class WitnessPlanner:
    """
    Computes backward and forward temporal support required to
    witness the intended local phase behavior.
    """

    def __init__(self, thresholds: WitnessThresholds) -> None:
        self._thresholds = thresholds

    def _required_support_one_direction(
        self,
        voice: LatticeVoice,
        t: float,
        backward: bool,
    ) -> float:
        threshold = (
            self._thresholds.backward_phase_radians
            if backward
            else self._thresholds.forward_phase_radians
        )

        step = self._thresholds.support_search_step_seconds
        max_support = self._thresholds.max_support_seconds

        fn = voice.effective_angular_frequency

        # Incremental 2-point trapezoid: instead of re-integrating the growing
        # window from scratch each iteration (O(N²) evaluations), accumulate
        # one new slice of width `step` per iteration (O(N) total).
        #
        # The trapezoid rule is additive over adjacent intervals:
        #   ∫[a,c] = ∫[a,b] + ∫[b,c]
        # so the running sum equals the full window integral exactly.
        #
        # Each slice is only `step` wide (≤ 0.5 ms by default), so the 2-point
        # trapezoid is more than accurate enough for a smoothly varying ω(t).
        running = 0.0
        fn_prev = fn(t)   # boundary at t (right edge for backward, left for forward)
        support = step

        while support <= max_support:
            fn_curr = fn(t - support if backward else t + support)
            running += (fn_prev + fn_curr) * 0.5 * step

            if abs(running) >= threshold:
                return support

            fn_prev = fn_curr
            support += step

        return max_support

    def compute_support_budget_for_voice(self, voice: LatticeVoice, t: float) -> SupportBudget:
        backward = self._required_support_one_direction(voice, t, backward=True)
        forward = self._required_support_one_direction(voice, t, backward=False)
        return SupportBudget(backward_seconds=backward, forward_seconds=forward)

    def compute_support_budget_for_lattice(self, lattice: HarmonicLattice, t: float) -> SupportBudget:
        # The lowest-harmonic voice has the lowest instantaneous frequency and
        # therefore accumulates phase most slowly, always requiring the largest
        # temporal support window.  Checking all voices is redundant work.
        return self.compute_support_budget_for_voice(lattice._min_harmonic_voice, t)


# ============================================================
# Adaptive density planning
# ============================================================

class DensityPlanner:
    """
    Converts local phase / curvature / support requirements into
    an adaptive internal sample spacing.
    """

    def __init__(self, policy: DensityPolicy, witness_planner: WitnessPlanner) -> None:
        self._policy = policy
        self._witness_planner = witness_planner

    def local_internal_sample_rate(self, lattice: HarmonicLattice, t: float) -> float:
        max_freq = lattice.max_abs_frequency_hz(t)
        chirp_like = lattice.max_abs_chirp_like_measure(t)

        support_budget = self._witness_planner.compute_support_budget_for_lattice(lattice, t)
        support_pressure = 1.0 / max(
            1e-12,
            min(support_budget.backward_seconds, support_budget.forward_seconds, self._policy.max_dt * 1e6),
        )

        raw_sr = self._policy.oversampling_factor * (
            max_freq
            + self._policy.derivative_weight * math.sqrt(max(0.0, chirp_like))
            + self._policy.support_weight * support_pressure
        )

        sr = clamp(raw_sr, self._policy.min_sample_rate, self._policy.max_sample_rate)
        return sr

    def local_dt(self, lattice: HarmonicLattice, t: float) -> float:
        sr = self.local_internal_sample_rate(lattice, t)
        return clamp(1.0 / sr, self._policy.min_dt, self._policy.max_dt)


# ============================================================
# Adaptive emitter
# ============================================================

class AdaptiveEmitter:
    """
    Emits complex samples on an adaptive internal timeline.

    If a ``ComplexPreProjectionFilter`` is supplied, each sample is evaluated
    through the filter instead of the raw lattice sum.  This allows analytic
    admissibility filtering (gain taper + warp cap) to act in the one-sided
    complex domain before any real projection occurs.

    If a ``ComplexEnvelope`` is supplied via *complex_envelope*, it is applied
    as the final stage after the lattice sum (and after any pre-projection
    filter): each fully-formed complex sample z is passed through
    ``envelope.evaluate(t, z)`` before being stored.  This is the earliest
    possible place to reshape energy in the analytic domain — all waveform
    geometry is already in z, and the envelope acts as a complex scalar
    multiplier on that geometry.
    """

    def __init__(
        self,
        lattice: HarmonicLattice,
        density_planner: DensityPlanner,
        witness_planner: WitnessPlanner,
        pre_projection_filter: Optional[ComplexPreProjectionFilter] = None,
        complex_envelope: Optional["ComplexEnvelope"] = None,
    ) -> None:
        self._lattice = lattice
        self._density_planner = density_planner
        self._witness_planner = witness_planner
        self._filter = pre_projection_filter
        self._envelope = complex_envelope

    def _sample_lattice(self, t: float) -> complex:
        """Evaluate lattice at *t*, applying filter then envelope if set."""
        if self._filter is not None:
            z = self._filter.evaluate_lattice_filtered(self._lattice, t)
        else:
            z = self._lattice.evaluate_complex(t)
        if self._envelope is not None:
            z = self._envelope.evaluate(t, z)
        return z

    def emit(self, emission_range: EmissionRange) -> AdaptiveSampleBuffer:
        emission_range.validate()
        buffer = AdaptiveSampleBuffer()

        t = emission_range.start_time
        end = emission_range.end_time

        while t < end:
            value = self._sample_lattice(t)
            buffer.add(t, value)

            dt = self._density_planner.local_dt(self._lattice, t)
            if dt <= 0.0:
                raise RuntimeError("AdaptiveEmitter produced nonpositive dt")

            t_next = t + dt
            if t_next <= t:
                raise RuntimeError("AdaptiveEmitter failed to advance time")
            t = t_next

        if not buffer.samples or buffer.samples[-1].t < end:
            buffer.add(end, self._sample_lattice(end))

        return buffer

    def witness_budget_trace(
        self,
        emission_range: EmissionRange,
        step_seconds: float,
    ) -> List[Tuple[float, SupportBudget]]:
        emission_range.validate()
        if step_seconds <= 0.0:
            raise ValueError("step_seconds must be > 0")

        trace: List[Tuple[float, SupportBudget]] = []
        t = emission_range.start_time
        while t <= emission_range.end_time:
            trace.append((t, self._witness_planner.compute_support_budget_for_lattice(self._lattice, t)))
            t += step_seconds
        return trace


# ============================================================
# Projection / resampling
# ============================================================

class ProjectionKernel(ABC):
    @abstractmethod
    def weight(self, x: float) -> float:
        raise NotImplementedError

    def weight_array(self, x_arr):
        """Vectorised weight over a numpy array.  Override for performance."""
        import numpy as np
        return np.vectorize(self.weight)(x_arr)


class SincKernel(ProjectionKernel):
    """
    Windowless sinc kernel for demonstration. In production you would likely
    want windowing or a more sophisticated support strategy.
    """

    def weight(self, x: float) -> float:
        if abs(x) < 1e-12:
            return 1.0
        return math.sin(PI2 * 0.5 * x) / (PI2 * 0.5 * x)

    def weight_array(self, x_arr):
        import numpy as np
        px = math.pi * x_arr
        return np.where(np.abs(px) < 1e-12, 1.0, np.sin(px) / px)


class AdaptiveProjector:
    """
    Projects a nonuniform complex analytic sample buffer to a uniform output grid.
    """

    def __init__(
        self,
        policy: ProjectionPolicy,
        kernel: Optional[ProjectionKernel] = None,
    ) -> None:
        self._policy = policy
        self._kernel = kernel or SincKernel()

    def _estimate_local_nominal_dt(self, buffer: AdaptiveSampleBuffer, center_index: int) -> float:
        samples = buffer.samples
        if len(samples) < 2:
            return 1.0 / self._policy.output_sample_rate

        if center_index <= 0:
            return samples[1].t - samples[0].t
        if center_index >= len(samples) - 1:
            return samples[-1].t - samples[-2].t

        return 0.5 * ((samples[center_index].t - samples[center_index - 1].t) +
                      (samples[center_index + 1].t - samples[center_index].t))

    def _project_one(self, buffer: AdaptiveSampleBuffer, t_out: float) -> complex:
        if not buffer.samples:
            return 0j

        half_support = self._policy.projection_half_support_seconds
        start_t = t_out - half_support
        end_t = t_out + half_support

        times = buffer.times()
        left = bisect_left(times, start_t)
        right = bisect_left(times, end_t)

        if left == right:
            return buffer.evaluate_linear(t_out)

        num = 0j
        den = 0.0

        for i in range(left, right):
            sp = buffer.samples[i]
            local_dt = self._estimate_local_nominal_dt(buffer, i)
            x = (t_out - sp.t) / max(local_dt, 1e-12)
            w = self._kernel.weight(x)
            num += w * sp.value
            den += abs(w)

        if den == 0.0:
            return buffer.evaluate_linear(t_out)
        return num / den

    def project_complex(
        self,
        buffer: AdaptiveSampleBuffer,
        start_time: float,
        end_time: float,
    ) -> List[complex]:
        if end_time <= start_time:
            raise ValueError("project_complex requires end_time > start_time")

        dt = 1.0 / self._policy.output_sample_rate
        out: List[complex] = []

        t = start_time
        while t < end_time:
            out.append(self._project_one(buffer, t))
            t += dt

        out.append(self._project_one(buffer, end_time))
        return out

    def project_real(
        self,
        buffer: AdaptiveSampleBuffer,
        start_time: float,
        end_time: float,
    ) -> List[float]:
        return [z.real for z in self.project_complex(buffer, start_time, end_time)]

    # ------------------------------------------------------------------
    # Vectorised backends
    # ------------------------------------------------------------------

    def _build_projection_arrays(self, buffer: AdaptiveSampleBuffer):
        """Convert AdaptiveSampleBuffer to three numpy arrays for vectorised projection.

        Returns ``(t_arr, v_arr, dt_arr)`` where:

        * ``t_arr``  – sample times, shape ``(n,)``, dtype ``float64``
        * ``v_arr``  – complex sample values, shape ``(n,)``, dtype ``complex128``
        * ``dt_arr`` – local nominal dt per sample (centred finite difference of
          neighbours), shape ``(n,)``, dtype ``float64``
        """
        import numpy as np
        times  = buffer.times()
        values = buffer.values()
        n = len(times)
        if n == 0:
            empty_f = np.array([], dtype=np.float64)
            return empty_f, np.array([], dtype=np.complex128), empty_f
        t_arr  = np.array(times,  dtype=np.float64)
        v_arr  = np.array(values, dtype=np.complex128)
        dt_arr = np.empty(n, dtype=np.float64)
        if n == 1:
            dt_arr[0] = 1.0 / self._policy.output_sample_rate
        else:
            dt_arr[0]  = t_arr[1] - t_arr[0]
            dt_arr[-1] = t_arr[-1] - t_arr[-2]
            if n > 2:
                dt_arr[1:-1] = 0.5 * (t_arr[2:] - t_arr[:-2])
        return t_arr, v_arr, dt_arr

    def project_complex_numpy(
        self,
        buffer: AdaptiveSampleBuffer,
        start_time: float,
        end_time: float,
    ):
        """Numpy-vectorised projection to a uniform complex grid.

        The inner sinc-weight summation runs as numpy C code instead of a
        Python ``for``-loop, eliminating per-sample Python overhead in the
        kernel evaluation.  The outer loop (one Python iteration per output
        sample) remains; for a fully loop-free path use
        :meth:`project_complex_torch`.

        Returns a ``numpy.ndarray`` of ``complex128``.
        """
        import numpy as np
        if not buffer.samples:
            return np.array([], dtype=np.complex128)

        t_arr, v_arr, dt_arr = self._build_projection_arrays(buffer)

        dt_out = 1.0 / self._policy.output_sample_rate
        t_out  = np.arange(start_time, end_time + dt_out * 0.5, dt_out)
        if len(t_out) == 0 or t_out[-1] < end_time - dt_out * 1e-6:
            t_out = np.append(t_out, end_time)

        half_support = self._policy.projection_half_support_seconds
        lo_arr = np.searchsorted(t_arr, t_out - half_support, side='left')
        hi_arr = np.searchsorted(t_arr, t_out + half_support, side='right')

        out = np.zeros(len(t_out), dtype=np.complex128)
        for i in range(len(t_out)):
            lo, hi = int(lo_arr[i]), int(hi_arr[i])
            if lo >= hi:
                out[i] = buffer.evaluate_linear(float(t_out[i]))
                continue
            t_sl  = t_arr[lo:hi]
            v_sl  = v_arr[lo:hi]
            dt_sl = dt_arr[lo:hi]
            x   = (float(t_out[i]) - t_sl) / np.maximum(dt_sl, 1e-12)
            w   = self._kernel.weight_array(x)
            den = float(np.abs(w).sum())
            out[i] = (np.dot(w, v_sl) / den) if den != 0.0 else buffer.evaluate_linear(float(t_out[i]))
        return out

    def project_real_numpy(
        self,
        buffer: AdaptiveSampleBuffer,
        start_time: float,
        end_time: float,
    ):
        """Numpy-vectorised projection to a uniform real grid.

        Returns a ``numpy.ndarray`` of ``float64`` (real part of the analytic signal).
        """
        return self.project_complex_numpy(buffer, start_time, end_time).real

    def project_complex_torch(
        self,
        buffer: AdaptiveSampleBuffer,
        start_time: float,
        end_time: float,
        device: str = 'cpu',
    ):
        """Fully-vectorised torch projection — no Python loop over output samples.

        Builds a padded ``(n_out × max_window)`` gather matrix entirely in
        numpy (no Python loop), then evaluates all sinc weights as a single
        tensor operation.  Pass ``device='cuda'`` to run on GPU.

        Returns a ``numpy.ndarray`` of ``complex128``.

        Note: this path hardcodes the normalised-sinc formula; custom
        ``ProjectionKernel`` subclasses are not honoured here.
        """
        import numpy as np
        import torch

        if not buffer.samples:
            return np.array([], dtype=np.complex128)

        t_np, v_np, dt_np = self._build_projection_arrays(buffer)
        n_adaptive = len(t_np)

        dt_out   = 1.0 / self._policy.output_sample_rate
        t_out_np = np.arange(start_time, end_time + dt_out * 0.5, dt_out)
        if len(t_out_np) == 0 or t_out_np[-1] < end_time - dt_out * 1e-6:
            t_out_np = np.append(t_out_np, end_time)
        n_out = len(t_out_np)

        half_support = self._policy.projection_half_support_seconds
        lo_arr = np.searchsorted(t_np, t_out_np - half_support, side='left')
        hi_arr = np.searchsorted(t_np, t_out_np + half_support, side='right')

        window_sizes = (hi_arr - lo_arr).astype(np.int64)
        max_win = int(window_sizes.max()) if n_out > 0 and window_sizes.max() > 0 else 1

        # Build (n_out × max_win) index matrix with no Python loop.
        # src_candidates[i, j] = lo_arr[i] + j; mask out j >= window_sizes[i].
        offsets        = np.arange(max_win, dtype=np.int64)
        src_candidates = lo_arr[:, None].astype(np.int64) + offsets[None, :]
        valid_mask     = offsets[None, :] < window_sizes[:, None]
        idx_matrix     = np.where(valid_mask, src_candidates, 0)
        idx_matrix     = np.clip(idx_matrix, 0, n_adaptive - 1)

        t_gathered   = t_np[idx_matrix]
        v_r_gathered = v_np.real[idx_matrix]
        v_i_gathered = v_np.imag[idx_matrix]
        dt_gathered  = dt_np[idx_matrix]

        dtype   = torch.float64
        t_out_t = torch.tensor(t_out_np[:, None], dtype=dtype, device=device)
        t_g     = torch.tensor(t_gathered,    dtype=dtype, device=device)
        v_r     = torch.tensor(v_r_gathered,  dtype=dtype, device=device)
        v_i     = torch.tensor(v_i_gathered,  dtype=dtype, device=device)
        dt_g    = torch.tensor(dt_gathered,   dtype=dtype, device=device)
        vm      = torch.tensor(valid_mask,    dtype=torch.bool, device=device)

        x  = (t_out_t - t_g) / dt_g.clamp(min=1e-12)
        px = math.pi * x
        w  = torch.where(px.abs() < 1e-12, torch.ones_like(px), px.sin() / px)
        w  = w.masked_fill(~vm, 0.0)

        denom    = w.abs().sum(dim=1).clamp(min=1e-12)
        out_real = (w * v_r).sum(dim=1) / denom
        out_imag = (w * v_i).sum(dim=1) / denom

        return out_real.cpu().numpy() + 1j * out_imag.cpu().numpy()

    def project_real_torch(
        self,
        buffer: AdaptiveSampleBuffer,
        start_time: float,
        end_time: float,
        device: str = 'cpu',
    ):
        """Fully-vectorised torch projection to real.

        Returns a ``numpy.ndarray`` of ``float64``.
        """
        return self.project_complex_torch(buffer, start_time, end_time, device=device).real

    # ------------------------------------------------------------------
    # Quadrature (stereo) projections — no phase folding
    # ------------------------------------------------------------------

    def project_quadrature(
        self,
        buffer: AdaptiveSampleBuffer,
        start_time: float,
        end_time: float,
    ) -> Tuple[List[float], List[float]]:
        """Project to stereo quadrature pair (L, R) = (cos, sin) components.

        Unlike project_real, this preserves the full phase geometry: each
        distinct phase state maps to a unique (L, R) point in the plane.
        No folding, no projection-induced interference.

        Returns ``(L, R)`` where each is a list of float.
        """
        z_list = self.project_complex(buffer, start_time, end_time)
        return [z.real for z in z_list], [z.imag for z in z_list]

    def project_quadrature_numpy(
        self,
        buffer: AdaptiveSampleBuffer,
        start_time: float,
        end_time: float,
    ):
        """Numpy-vectorised quadrature projection.

        Returns ``(L, R)`` as a pair of ``numpy.ndarray`` of ``float64``,
        preserving the native dtype of the complex projection.
        """
        z = self.project_complex_numpy(buffer, start_time, end_time)
        return z.real.copy(), z.imag.copy()

    def project_quadrature_torch(
        self,
        buffer: AdaptiveSampleBuffer,
        start_time: float,
        end_time: float,
        device: str = 'cpu',
    ):
        """Fully-vectorised torch quadrature projection.

        Returns ``(L, R)`` as a pair of ``numpy.ndarray`` of ``float64``.
        """
        z = self.project_complex_torch(buffer, start_time, end_time, device=device)
        return z.real.copy(), z.imag.copy()

    # ------------------------------------------------------------------
    # Phase gradient projection — ray-world rendering
    # ------------------------------------------------------------------

    def project_phase_gradient_numpy(
        self,
        buffer: AdaptiveSampleBuffer,
        start_time: float,
        end_time: float,
    ):
        """Project to instantaneous frequency (phase gradient) on a uniform grid.

        Computes ``f(t) = dφ/dt / (2π)`` from consecutive complex analytic
        samples via the unbiased estimator:
            f_k = angle(z_{k+1} · conj(z_k)) / (2π · Δt_k)

        This operates on the adaptive (oversampled) internal buffer before
        any decimation — frequency accuracy exceeds what any spectrogram
        can achieve from the real projection.

        Returns ``(f_inst, t_axis)`` as ``numpy.ndarray`` of ``float64``.
        ``t_axis`` is uniformly spaced; ``f_inst`` is the phase gradient
        resampled onto it via nearest-neighbour assignment (same as
        AnalyticalTFRenderer.render_from_buffer).
        """
        import numpy as np

        times  = buffer.times()
        values = buffer.values()
        if len(times) < 2:
            dt_out = 1.0 / self._policy.output_sample_rate
            t_axis = np.arange(start_time, end_time + dt_out * 0.5, dt_out)
            return np.zeros(len(t_axis), dtype=np.float64), t_axis

        t_arr = np.array(times,  dtype=np.float64)
        z_arr = np.array(values, dtype=np.complex128)

        dt_arr = np.diff(t_arr)
        dz     = z_arr[1:] * np.conj(z_arr[:-1])
        f_inst_raw = np.angle(dz) / (PI2 * np.maximum(dt_arr, 1e-15))
        t_mid      = 0.5 * (t_arr[:-1] + t_arr[1:])

        mask   = (t_mid >= start_time) & (t_mid <= end_time)
        t_pts  = t_mid[mask]
        f_pts  = f_inst_raw[mask]

        dt_out = 1.0 / self._policy.output_sample_rate
        t_axis = np.arange(start_time, end_time + dt_out * 0.5, dt_out)
        if len(t_axis) == 0 or t_axis[-1] < end_time - dt_out * 1e-6:
            t_axis = np.append(t_axis, end_time)

        f_out = np.zeros(len(t_axis), dtype=np.float64)
        if len(t_pts) > 0:
            col_idx = np.clip(
                np.round((t_pts - start_time) / (end_time - start_time) * (len(t_axis) - 1)).astype(int),
                0, len(t_axis) - 1,
            )
            f_out[col_idx] = f_pts  # last-write wins for collisions

        return f_out, t_axis


# ============================================================
# Top-level synth driver
# ============================================================

class WitnessAwareSynthDriver:
    """
    High-level orchestration object.

    It owns:
    - the lattice (shared-path harmonic structure)
    - witness planning
    - density planning
    - adaptive emission  (optionally with a ComplexPreProjectionFilter)
    - projection to uniform output

    Passing ``pre_projection_filter`` activates pre-projection admissibility
    filtering in the analytic complex construction space.  The filter runs
    per-voice before summation, so the complex adaptive buffer is already
    band-safe before the final projection kernel runs.

    Alternatively, call ``attach_admissibility_filter(output_sample_rate)``
    after construction to auto-build a filter from the projection policy's
    sample rate.
    """

    def __init__(
        self,
        lattice: HarmonicLattice,
        witness_thresholds: WitnessThresholds,
        density_policy: DensityPolicy,
        projection_policy: ProjectionPolicy,
        pre_projection_filter: Optional[ComplexPreProjectionFilter] = None,
        complex_envelope: Optional["ComplexEnvelope"] = None,
    ) -> None:
        self._lattice = lattice
        self._witness_planner = WitnessPlanner(witness_thresholds)
        self._density_planner = DensityPlanner(density_policy, self._witness_planner)
        self._complex_envelope = complex_envelope
        self._emitter = AdaptiveEmitter(
            lattice, self._density_planner, self._witness_planner,
            pre_projection_filter=pre_projection_filter,
            complex_envelope=complex_envelope,
        )
        self._projector = AdaptiveProjector(projection_policy)
        self._projection_policy = projection_policy
        # Propagate the configurable tap count to every drift model in the lattice.
        for voice in lattice.voices:
            voice._drift_model._fallback_integration_steps = (
                witness_thresholds.integration_steps_per_check
            )

    def attach_admissibility_filter(
        self,
        output_sample_rate: Optional[float] = None,
        drift_margin_factor: float = 0.05,
        warp_margin_factor: float = 2.0,
        soft_knee_width: float = 500.0,
    ) -> "WitnessAwareSynthDriver":
        """
        Build and attach an ``OutputAdmissibilityPlanner`` + ``ComplexPreProjectionFilter``
        configured for the target output rate (defaults to the projection policy's rate).

        Returns ``self`` for chaining::

            driver = factory.build(...).attach_admissibility_filter()
        """
        sr = output_sample_rate or self._projection_policy.output_sample_rate
        planner = OutputAdmissibilityPlanner(
            output_sample_rate=sr,
            drift_margin_factor=drift_margin_factor,
            warp_margin_factor=warp_margin_factor,
            soft_knee_width=soft_knee_width,
        )
        filt = ComplexPreProjectionFilter(planner)
        self._emitter = AdaptiveEmitter(
            self._lattice, self._density_planner, self._witness_planner,
            pre_projection_filter=filt,
            complex_envelope=self._complex_envelope,
        )
        return self

    @property
    def lattice(self) -> HarmonicLattice:
        return self._lattice

    def emit_adaptive_complex(self, emission_range: EmissionRange) -> AdaptiveSampleBuffer:
        return self._emitter.emit(emission_range)

    def emit_uniform_complex(self, emission_range: EmissionRange) -> List[complex]:
        adaptive = self.emit_adaptive_complex(emission_range)
        return self._projector.project_complex(adaptive, emission_range.start_time, emission_range.end_time)

    def emit_uniform_real(self, emission_range: EmissionRange) -> List[float]:
        adaptive = self.emit_adaptive_complex(emission_range)
        return self._projector.project_real(adaptive, emission_range.start_time, emission_range.end_time)

    def emit_quadrature_stereo(
        self,
        emission_range: EmissionRange,
    ) -> Tuple[object, object]:
        """Emit the analytic signal as a quadrature stereo pair (L, R).

        L(t) = A(t)·cos(φ(t))   — in-phase (same as mono)
        R(t) = A(t)·sin(φ(t))   — quadrature

        This is a direct embedding of the complex plane into two real channels.
        Phase maps to angle in the stereo plane rather than collapsing into
        mono interference. No phase folding, no projection-induced degeneracy.

        Returns ``(L, R)`` as ``numpy.ndarray`` of the buffer's native dtype.
        """
        adaptive = self.emit_adaptive_complex(emission_range)
        return self._projector.project_quadrature_numpy(
            adaptive, emission_range.start_time, emission_range.end_time
        )

    def emit_phase_gradient(
        self,
        emission_range: EmissionRange,
    ) -> Tuple[object, object]:
        """Emit instantaneous frequency (phase gradient field) on a uniform grid.

        Computes f(t) = dφ/dt / (2π) from the oversampled analytic buffer
        before any decimation — accuracy exceeds spectrogram-based estimates.

        Returns ``(f_inst, t_axis)`` as ``numpy.ndarray`` of ``float64``.
        Use f_inst to drive a phase-accumulator synth, ray visualization, or
        frequency-domain rendering without projection artifacts.
        """
        adaptive = self.emit_adaptive_complex(emission_range)
        return self._projector.project_phase_gradient_numpy(
            adaptive, emission_range.start_time, emission_range.end_time
        )

    def witness_trace(
        self,
        emission_range: EmissionRange,
        step_seconds: float,
    ) -> List[Tuple[float, SupportBudget]]:
        return self._emitter.witness_budget_trace(emission_range, step_seconds)


# ============================================================
# Analytical time-frequency renderer
# ============================================================

class AnalyticalTFRenderer:
    """
    Time-frequency images computed directly from analytic signal geometry —
    no FFT, no windowing, no projection artifacts.

    render_from_lattice(lattice, t_start, t_end)
        Samples each voice's instantaneous frequency f(t) and amplitude A(t)
        on a uniform time grid and paints a Gaussian ridge into TF space.
        Frequency accuracy is bounded only by the analytic PhasePath
        derivatives, not by any spectral estimation step.

    render_from_buffer(buffer, t_start, t_end)
        Estimates instantaneous frequency from consecutive complex analytic
        samples via phase differences:
            f_k = ∠(z_{k+1} · z̄_k) / (2π · Δt_k)
        This is the optimal unbiased frequency estimator for an analytic signal
        — more accurate than any DFT-based spectrogram because it uses the
        one-sided complex signal before projection to real, at the adaptive
        (possibly oversampled) internal grid spacing.

    In both modes the output is a (n_freq, n_time) float64 amplitude array
    plus the corresponding t_axis and f_axis arrays for plotting.
    """

    def __init__(self, freq_sigma_hz: float = 8.0) -> None:
        """
        freq_sigma_hz
            1σ width of the Gaussian smearing kernel applied in the frequency
            dimension.  Smaller = sharper ridges; larger = smoother image.
            Has no analogue in FFT-based spectrograms — it is purely a
            display parameter, not a time-frequency resolution tradeoff.
        """
        self._sigma = max(freq_sigma_hz, 1e-3)

    # ------------------------------------------------------------------
    # Render from voice geometry
    # ------------------------------------------------------------------

    def render_from_lattice(
        self,
        lattice: "HarmonicLattice",
        t_start: float,
        t_end: float,
        n_time: int = 512,
        n_freq: int = 512,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        log_freq: bool = True,
    ) -> Tuple[object, object, object]:
        """
        Paint each voice as a frequency ridge in TF space.

        Returns ``(image, t_axis, f_axis)``.
        image shape: (n_freq, n_time), dtype float64.
        """
        import numpy as np

        t_axis = np.linspace(t_start, t_end, n_time)

        t_mid = 0.5 * (t_start + t_end)
        f_all = [abs(v.effective_frequency_hz(t_mid)) for v in lattice.voices]
        fmin  = freq_min or max(1.0, min(f_all) * 0.7)
        fmax  = freq_max or max(f_all) * 1.4
        f_axis = self._make_freq_axis(fmin, fmax, n_freq, log_freq)

        image = np.zeros((n_freq, n_time), dtype=np.float64)

        for voice in lattice.voices:
            f_ridge = np.fromiter(
                (voice.effective_frequency_hz(t) for t in t_axis),
                dtype=float, count=n_time,
            )
            A_ridge = np.fromiter(
                (voice.amplitude(t) for t in t_axis),
                dtype=float, count=n_time,
            )
            # (n_freq × n_time) vectorised outer product — one op per voice.
            diff  = f_axis[:, None] - f_ridge[None, :]
            image += np.exp(-0.5 * (diff / self._sigma) ** 2) * A_ridge[None, :]

        return image, t_axis, f_axis

    # ------------------------------------------------------------------
    # Render from complex analytic buffer
    # ------------------------------------------------------------------

    def render_from_buffer(
        self,
        buffer: "AdaptiveSampleBuffer",
        t_start: float,
        t_end: float,
        n_time: int = 512,
        n_freq: int = 512,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        log_freq: bool = True,
    ) -> Tuple[object, object, object]:
        """
        Estimate instantaneous frequency from the complex analytic buffer and
        render a TF image.

        Uses ``f_k = ∠(z_{k+1}·z̄_k) / (2π·Δt_k)`` at each pair of
        consecutive adaptive samples.  This estimator is unbiased for locally
        stationary analytic signals and uses the full oversampled internal grid
        rather than the decimated real output — accuracy far exceeds FFT.

        Returns ``(image, t_axis, f_axis)``.
        image shape: (n_freq, n_time), dtype float64.
        """
        import numpy as np

        if len(buffer) < 2:
            raise ValueError("Buffer must contain at least 2 samples")

        t_arr = np.array(buffer.times())
        z_arr = np.array(buffer.values())

        # Phase-difference instantaneous frequency — avoids unwrap issues.
        dt_arr = np.diff(t_arr)
        dz     = z_arr[1:] * np.conj(z_arr[:-1])
        f_inst = np.angle(dz) / (PI2 * np.maximum(dt_arr, 1e-15))
        t_mid  = 0.5 * (t_arr[:-1] + t_arr[1:])
        A_mid  = 0.5 * (np.abs(z_arr[:-1]) + np.abs(z_arr[1:]))

        # Restrict to the requested range; keep only positive instantaneous freqs.
        mask   = (t_mid >= t_start) & (t_mid <= t_end) & (f_inst > 0)
        t_pts  = t_mid[mask]
        f_pts  = f_inst[mask]
        A_pts  = A_mid[mask]

        t_axis = np.linspace(t_start, t_end, n_time)
        if len(t_pts) == 0:
            fmin   = freq_min or 20.0
            fmax   = freq_max or 20_000.0
            f_axis = self._make_freq_axis(fmin, fmax, n_freq, log_freq)
            return np.zeros((n_freq, n_time), dtype=np.float64), t_axis, f_axis

        fmin   = freq_min or max(1.0, float(np.percentile(f_pts, 2)) * 0.7)
        fmax   = freq_max or float(np.percentile(f_pts, 98)) * 1.4
        f_axis = self._make_freq_axis(fmin, fmax, n_freq, log_freq)
        image  = np.zeros((n_freq, n_time), dtype=np.float64)

        # Scatter buffer points into nearest time-axis columns.
        col_idx = np.clip(
            np.round((t_pts - t_start) / (t_end - t_start) * (n_time - 1)).astype(int),
            0, n_time - 1,
        )

        # Vectorised Gaussian paint: (n_freq × m) — one matrix op.
        diff  = f_axis[:, None] - f_pts[None, :]
        gauss = np.exp(-0.5 * (diff / self._sigma) ** 2) * A_pts[None, :]
        np.add.at(image, (slice(None), col_idx), gauss)

        return image, t_axis, f_axis

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_freq_axis(fmin: float, fmax: float, n: int, log: bool):
        import numpy as np
        if log:
            return np.logspace(math.log10(max(fmin, 1e-3)), math.log10(fmax), n)
        return np.linspace(fmin, fmax, n)

    @staticmethod
    def save_png(
        image,
        t_axis,
        f_axis,
        path: str,
        title: str = "Analytical TF Image",
        log_amplitude: bool = True,
        dpi: int = 150,
    ) -> None:
        """Save a TF image array to PNG via matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required — pip install matplotlib")
        import numpy as np

        img = np.log1p(image) if log_amplitude else image
        fig, ax = plt.subplots(figsize=(11, 4), dpi=dpi)
        ax.imshow(
            img,
            aspect="auto",
            origin="lower",
            extent=[float(t_axis[0]), float(t_axis[-1]),
                    float(f_axis[0]), float(f_axis[-1])],
            cmap="inferno",
            interpolation="bilinear",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"TF image saved: {path}")


# ============================================================
# Factory helpers
# ============================================================

class SynthFactory:
    """
    Convenience construction helpers to make experiment setup less noisy.
    """

    @staticmethod
    def build_exponential_dc_approach_lattice(
        initial_frequency_hz: float,
        tau_seconds: float,
        partial_count: int,
        waveform_manifold: Optional[WaveformManifold] = None,
        drift_model: Optional[DriftModel] = None,
        amplitude_decay_tau: float = 5.0,
        shape_low: float = 0.1,
        shape_high: float = 0.8,
        shape_center: float = 1.0,
        shape_width: float = 0.5,
    ) -> HarmonicLattice:
        master_path = ExponentialDecayPhasePath(
            initial_frequency_hz=initial_frequency_hz,
            tau_seconds=tau_seconds,
        )

        manifold = waveform_manifold or PhaseWarpedManifold()
        drift = drift_model or SmoothSinusoidalDriftModel(
            base_rate_hz=0.07,
            max_offset_hz=0.15,
            harmonic_scaling_power=0.8,
        )

        voices: List[LatticeVoice] = []
        for harmonic_index in range(1, partial_count + 1):
            amplitude = ExponentialEnvelope(
                initial=1.0 / harmonic_index,
                tau_seconds=amplitude_decay_tau * (1.0 + 0.1 * harmonic_index),
            )
            shape = SigmoidField(
                low=shape_low,
                high=shape_high,
                center=shape_center,
                width=shape_width,
            )

            voice = LatticeVoice(
                name=f"partial_{harmonic_index}",
                harmonic_index=harmonic_index,
                master_phase_path=master_path,
                waveform_manifold=manifold,
                amplitude_field=amplitude,
                shape_field=shape,
                drift_model=drift,
                gain=1.0,
                phase_offset=0.0,
            )
            voices.append(voice)

        return HarmonicLattice(voices)


# ============================================================
# Example usage
# ============================================================

def example_build_driver() -> WitnessAwareSynthDriver:
    lattice = SynthFactory.build_exponential_dc_approach_lattice(
        initial_frequency_hz=80.0,
        tau_seconds=3.5,
        partial_count=6,
        waveform_manifold=PhaseWarpedManifold(harmonic_warp_strength=0.35),
        drift_model=SmoothSinusoidalDriftModel(
            base_rate_hz=0.05,
            max_offset_hz=0.08,
            harmonic_scaling_power=1.0,
        ),
    )

    driver = WitnessAwareSynthDriver(
        lattice=lattice,
        witness_thresholds=WitnessThresholds(
            backward_phase_radians=PI2,
            forward_phase_radians=PI2,
            max_support_seconds=8.0,
            support_search_step_seconds=0.0005,
            integration_steps_per_check=32,
        ),
        density_policy=DensityPolicy(
            oversampling_factor=24.0,
            min_sample_rate=96_000.0,
            max_sample_rate=5_000_000.0,
            derivative_weight=0.75,
            support_weight=1.25,
        ),
        projection_policy=ProjectionPolicy(
            output_sample_rate=192_000.0,
            projection_half_support_seconds=0.0015,
            projection_kernel_steps=256,
        ),
    )
    driver.attach_admissibility_filter()
    return driver


def example_render() -> None:
    import numpy as np

    driver       = example_build_driver()
    render_range = EmissionRange(start_time=0.0, end_time=2.0)

    adaptive = driver.emit_adaptive_complex(render_range)
    real_np  = driver._projector.project_real_numpy(
        adaptive, render_range.start_time, render_range.end_time
    )
    witness = driver.witness_trace(render_range, step_seconds=0.1)

    print(f"Adaptive samples:     {len(adaptive)}")
    print(f"Uniform real samples: {len(real_np)}")
    print("First 5 adaptive samples:")
    for sp in adaptive.samples[:5]:
        print(f"  t={sp.t:.9f}, z={sp.value}")

    print("Witness budgets:")
    for t, budget in witness[:5]:
        print(
            f"  t={t:.3f}, backward={budget.backward_seconds:.6f}s, "
            f"forward={budget.forward_seconds:.6f}s"
        )

    # Write 32-bit float mono WAV.
    import scipy.io.wavfile as _wavfile
    peak        = float(np.abs(real_np).max()) or 1.0
    f32         = (real_np / peak).astype(np.float32)
    sample_rate = int(driver._projector._policy.output_sample_rate)
    output_path = "output.wav"
    _wavfile.write(output_path, sample_rate, f32)
    print(f"Written {output_path}: {len(f32)} samples @ {sample_rate} Hz, 32-bit float mono")


if __name__ == "__main__":
    example_render()