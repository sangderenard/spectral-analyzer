from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, Tuple
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
# Core support dataclasses
# ============================================================

@dataclass(frozen=True)
class WitnessThresholds:
    """
    Controls how much accumulated phase is required to consider
    a local point in time 'witnessed' by surrounding history/future.
    """
    backward_phase_radians: float = PI2
    forward_phase_radians: float = PI2
    max_support_seconds: float = 10.0
    support_search_step_seconds: float = 1e-3
    integration_steps_per_check: int = 24


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


@dataclass(frozen=True)
class ProjectionPolicy:
    """
    Controls projection from adaptive internal domain to uniform output.
    """
    output_sample_rate: float = 48000.0
    projection_half_support_seconds: float = 0.002
    projection_kernel_steps: int = 256


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

    def phase(self, t: float) -> float:
        return self._phase0 + PI2 * self._f0 * self._tau * (1.0 - math.exp(-t / self._tau))


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


# ============================================================
# Drift models
# ============================================================

class DriftModel(ABC):
    """
    Drift is additive in frequency space by default.
    """

    @abstractmethod
    def frequency_offset_hz(self, t: float, harmonic_index: int) -> float:
        raise NotImplementedError

    def phase_integral_at(self, t: float, harmonic_index: int) -> float:
        """Integral of 2π·frequency_offset_hz from 0 to t.

        Subclasses with a closed form should override this.
        The default falls back to numerical trapezoid integration.
        """
        return PI2 * integrate_trapezoid(
            lambda tau: self.frequency_offset_hz(tau, harmonic_index),
            0.0, t, steps=64,
        )


class NullDriftModel(DriftModel):
    def frequency_offset_hz(self, t: float, harmonic_index: int) -> float:
        return 0.0

    def phase_integral_at(self, t: float, harmonic_index: int) -> float:
        return 0.0


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
        """
        Approximate a curvature-related measure by differentiating
        effective frequency numerically.
        """
        max_value = 0.0
        for voice in self._voices:
            value = abs(differentiate_central(voice.effective_frequency_hz, t))
            max_value = max(max_value, value)
        return max_value


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
        max_backward = 0.0
        max_forward = 0.0
        for voice in lattice.voices:
            budget = self.compute_support_budget_for_voice(voice, t)
            max_backward = max(max_backward, budget.backward_seconds)
            max_forward = max(max_forward, budget.forward_seconds)
        return SupportBudget(backward_seconds=max_backward, forward_seconds=max_forward)


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
    """

    def __init__(
        self,
        lattice: HarmonicLattice,
        density_planner: DensityPlanner,
        witness_planner: WitnessPlanner,
    ) -> None:
        self._lattice = lattice
        self._density_planner = density_planner
        self._witness_planner = witness_planner

    def emit(self, emission_range: EmissionRange) -> AdaptiveSampleBuffer:
        emission_range.validate()
        buffer = AdaptiveSampleBuffer()

        t = emission_range.start_time
        end = emission_range.end_time

        while t < end:
            value = self._lattice.evaluate_complex(t)
            buffer.add(t, value)

            dt = self._density_planner.local_dt(self._lattice, t)
            if dt <= 0.0:
                raise RuntimeError("AdaptiveEmitter produced nonpositive dt")

            t_next = t + dt
            if t_next <= t:
                raise RuntimeError("AdaptiveEmitter failed to advance time")
            t = t_next

        if not buffer.samples or buffer.samples[-1].t < end:
            buffer.add(end, self._lattice.evaluate_complex(end))

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


class SincKernel(ProjectionKernel):
    """
    Windowless sinc kernel for demonstration. In production you would likely
    want windowing or a more sophisticated support strategy.
    """

    def weight(self, x: float) -> float:
        if abs(x) < 1e-12:
            return 1.0
        return math.sin(PI2 * 0.5 * x) / (PI2 * 0.5 * x)


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
    - adaptive emission
    - projection to uniform output
    """

    def __init__(
        self,
        lattice: HarmonicLattice,
        witness_thresholds: WitnessThresholds,
        density_policy: DensityPolicy,
        projection_policy: ProjectionPolicy,
    ) -> None:
        self._lattice = lattice
        self._witness_planner = WitnessPlanner(witness_thresholds)
        self._density_planner = DensityPlanner(density_policy, self._witness_planner)
        self._emitter = AdaptiveEmitter(lattice, self._density_planner, self._witness_planner)
        self._projector = AdaptiveProjector(projection_policy)

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

    def witness_trace(
        self,
        emission_range: EmissionRange,
        step_seconds: float,
    ) -> List[Tuple[float, SupportBudget]]:
        return self._emitter.witness_budget_trace(emission_range, step_seconds)


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
        waveform_manifold=Pha