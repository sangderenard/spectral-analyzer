"""granular_engine.py

Analytic granular population driver.

Core classes
------------
GrainSpec               Fully-specified single-grain parameters (frozen dataclass).
GrainPopulationSpec     Distribution parameters for spawning grain populations.
AnalyticGrain           Synthesises one grain into a complex128 array.
GranularClusterDriver   Spawns and sums a full grain population.

The grain model
---------------
Each grain g_k is:

    g_k(t_local) = a_k * envelope_k(t_local) * manifold_k(φ_k(t_local))

where
    φ_k(t_local) = θ_k  +  2π * (f_k * t_local  +  0.5 * chirp_k * t_local²)
    envelope_k   = raised-cosine with asymmetric attack / release fractions

All grains are complex128 (analytic).  The driver accumulates them into a
single complex128 output array that can be fed directly into RoutingMixer or
mixed with other voice sources.

Population axes (all orthogonal)
---------------------------------
1. Birth process   — density_hz, birth_jitter, burst_probability, burst_size, burst_spread_s
2. Duration field  — grain_duration_s, grain_duration_jitter (log-normal)
3. Frequency field — center_frequency_hz, grain_pitch_spread_semitones,
                     grain_harmonic_lock, grain_highband_bias
4. Chirp field     — grain_chirp_depth, grain_chirp_jitter
5. Phase coherence — grain_phase_randomness
6. Manifold field  — grain_manifold_mix (0=pure sine, 1=harmonic measure)
7. Amplitude law   — gain, grain_amp_jitter (log-normal)
8. Coherence knob  — single float that jointly governs spread, phase, duration tendency

Coherence meta-knob
-------------------
coherence=1.0 → narrow pitch spread, low phase randomness, longer grains,
                stronger harmonic lock (body-like)
coherence=0.0 → broad pitch spread, fully random phase, shorter grains,
                no harmonic lock (air-like)

Call GrainPopulationSpec.with_coherence(c) to get a new spec with all
coherence-governed parameters scaled accordingly.  Individual parameters
can still be overridden after the call.

Usage
-----
    from granular_engine import GrainPopulationSpec, GranularClusterDriver

    spec = GrainPopulationSpec(
        center_frequency_hz=440.0,
        grain_density_hz=30.0,
        grain_duration_s=0.04,
    ).with_coherence(0.8)          # body-leaning

    driver = GranularClusterDriver(spec, sr=48_000)
    z = driver.synthesize(2.0)     # complex128, shape (96000,)
    audio = z.real.astype("float32")
"""
from __future__ import annotations

import math
import random
import dataclasses
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional import of signal_generator_v2 primitives
# ---------------------------------------------------------------------------
try:
    from signal_generator_v2 import (
        PureSineManifold,
        HarmonicMeasureManifold,
        KnobSpec,
    )
    _HAS_SGV2 = True
except ImportError:
    _HAS_SGV2 = False
    # Minimal stubs so the engine works standalone
    class PureSineManifold:  # type: ignore[no-redef]
        def evaluate(self, phase: float, shape_state: float = 0.0) -> complex:
            return complex(math.cos(phase), math.sin(phase))

    class HarmonicMeasureManifold:  # type: ignore[no-redef]
        def __init__(self, max_harmonics: int = 8) -> None:
            self._n = max_harmonics
        def evaluate(self, phase: float, shape_state: float = 0.0) -> complex:
            acc = 0j
            norm = 0.0
            for n in range(1, self._n + 1):
                c = 1.0 / n
                acc += c * complex(math.cos(n * phase), math.sin(n * phase))
                norm += c
            return acc / norm if norm else 0j

    from dataclasses import dataclass as _kdc, field as _kfield
    @_kdc
    class KnobSpec:  # type: ignore[no-redef]
        name: str = ""; label: str = ""
        dtype: str = "float"; default: object = None
        low: float = 0.0; high: float = 1.0; step: float = 0.0; unit: str = ""
        choices: list = _kfield(default_factory=list)
        is_log: bool = False; group: str = ""; fmt: str = ".3g"
        source_class: str = ""; rebuild_layout: bool = False; visible_when: object = None

_PI2 = 2.0 * math.pi
_SEMITONE = 2.0 ** (1.0 / 12.0)   # ratio per semitone

_PURE_SINE = PureSineManifold()
_HARMONIC8 = HarmonicMeasureManifold(max_harmonics=8)


# ---------------------------------------------------------------------------
# GrainSpec — fully-resolved parameters for one grain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GrainSpec:
    """Fully-specified parameters for one analytic grain.

    All values are absolute (already drawn from the population distributions).
    ``birth_time`` is in seconds from the start of the output buffer.
    ``phase_origin`` is the initial phase θ_k (radians).
    """
    birth_time:      float   # seconds into the output buffer
    duration:        float   # grain duration in seconds (> 0)
    frequency_hz:    float   # instantaneous center frequency at birth
    phase_origin:    float   # θ_k — initial phase (radians)
    chirp_hz_per_s:  float   # df/dt — linear chirp rate (Hz/s)
    amplitude:       float   # peak amplitude coefficient
    manifold_mix:    float   # 0 = pure sine, 1 = harmonic measure
    attack_frac:     float   # fraction of duration used for attack ramp
    release_frac:    float   # fraction of duration used for release ramp


# ---------------------------------------------------------------------------
# GrainPopulationSpec — the user-facing distribution controls
# ---------------------------------------------------------------------------

@dataclass
class GrainPopulationSpec:
    """Distribution parameters for a grain population.

    All axis controls are orthogonal — each can be set independently.
    Use ``with_coherence(c)`` to scale several axes simultaneously.
    """

    # -- Frequency placement --------------------------------------------------
    center_frequency_hz:        float = 440.0
    grain_pitch_spread_semitones: float = 3.0  # std-dev of log-normal freq draw
    grain_harmonic_lock:        float = 0.0    # 0=free, 1=snap to integer harmonics
    grain_highband_bias:        float = 0.0    # octaves to shift distribution upward

    # -- Birth process --------------------------------------------------------
    grain_density_hz:           float = 20.0   # mean grains per second
    birth_jitter:               float = 0.5    # 0=metronomic, 1=Poisson
    burst_probability:          float = 0.0    # chance of cluster per grain slot
    burst_size:                 int   = 4      # extra grains in a burst
    burst_spread_s:             float = 0.015  # time spread within a burst (s)

    # -- Duration field -------------------------------------------------------
    grain_duration_s:           float = 0.05   # mean grain duration
    grain_duration_jitter:      float = 0.3    # std-dev / mean (coefficient of variation)

    # -- Chirp field ----------------------------------------------------------
    grain_chirp_depth:          float = 0.0    # |df| as fraction of center_frequency_hz
    grain_chirp_jitter:         float = 0.5    # variance in chirp sign / magnitude

    # -- Phase coherence ------------------------------------------------------
    grain_phase_randomness:     float = 1.0    # 0=lock to 0, 1=fully random [0,2π)

    # -- Manifold -------------------------------------------------------------
    grain_manifold_mix:         float = 0.0    # 0=pure sine, 1=harmonic measure

    # -- Amplitude law --------------------------------------------------------
    gain:                       float = 1.0    # overall amplitude scale
    grain_amp_jitter:           float = 0.2    # log-normal std-dev / mean

    # -- Grain envelope shape -------------------------------------------------
    attack_frac:                float = 0.15   # fraction of grain duration = attack
    release_frac:               float = 0.35   # fraction of grain duration = release

    # -- Coherence meta-knob --------------------------------------------------
    grain_coherence:            float = 0.5    # 0=air, 1=body (center value)

    # -- Dynamic coherence field (modulates grain_coherence over time) --------
    coherence_mode:     str   = "uniform"
    # Modes:
    #   uniform     – constant at grain_coherence (static)
    #   sine        – sinusoidal oscillation around grain_coherence
    #   random_walk – Brownian walk seeded at grain_coherence
    #   burst       – coherence spikes on a Poisson schedule then decays back
    #   gradient    – linear ramp across the full synthesis duration
    #   perlin_walk – smooth pseudo-random walk (sum of slow sine oscillators)
    coherence_rate_hz:  float = 0.5    # oscillation / walk rate for dynamic modes
    coherence_depth:    float = 0.3    # amplitude of variation around grain_coherence
    coherence_burst_tau_s: float = -1.0  # M3: burst decay time constant (s); -1 = auto (2/rate)

    # -- Seed control ---------------------------------------------------------
    editor_seed:              int   = 0     # fixed RNG seed for all editor rerenders (stable)
    seed_animate:             bool  = False # opt-in: slowly animate seed on a timer (costly!)
    seed_animate_period_s:    float = 4.0   # seconds per seed step when seed_animate=True

    # -------------------------------------------------------------------------

    def with_coherence(self, c: float) -> "GrainPopulationSpec":
        """Return a new spec with coherence-governed axes interpolated.

        c=1.0 → narrow spread, long grains, low phase randomness (body)
        c=0.0 → broad spread, short grains, fully random phase (air)

        Axis mapping
        ------------
        grain_pitch_spread_semitones : 8 → 1  (broad → tight)
        grain_phase_randomness       : 1 → 0  (random → locked)
        grain_harmonic_lock          : 0 → 0.8
        grain_duration_s             : *0.5 → *1.5  (scaled from current mean)
        grain_duration_jitter        : 0.5 → 0.1
        attack_frac                  : 0.05 → 0.2
        release_frac                 : 0.2 → 0.5
        grain_chirp_depth            : unchanged (user sets this explicitly)
        grain_highband_bias          : unchanged (user sets this explicitly)
        """
        c = max(0.0, min(1.0, c))
        ic = 1.0 - c

        def lerp(lo: float, hi: float) -> float:
            return lo + (hi - lo) * c

        # Duration: air prefers 0.5× current, body prefers 1.5× current
        new_dur = self.grain_duration_s * (0.5 + 1.0 * c)

        return replace(
            self,
            grain_coherence              = c,
            grain_pitch_spread_semitones = lerp(8.0, 1.0),
            grain_phase_randomness       = lerp(1.0, 0.0),
            grain_harmonic_lock          = lerp(0.0, 0.8),
            grain_duration_s             = new_dur,
            grain_duration_jitter        = lerp(0.5, 0.1),
            attack_frac                  = lerp(0.05, 0.20),
            release_frac                 = lerp(0.20, 0.50),
        )

    def with_coherence_relative(self, c: float) -> "GrainPopulationSpec":
        """Like with_coherence but scales current axis values rather than overriding them.

        L1 fix: ``with_coherence`` maps coherence c to absolute pole positions
        (e.g. spread always goes to lerp(8,1) semitones regardless of the user's
        explicit ``grain_pitch_spread_semitones`` setting).
        ``with_coherence_relative`` instead multiplies the *current* values by
        coherence-dependent scale factors, preserving the user's intent:

            grain_pitch_spread_semitones *= lerp(2.0, 0.125, c)   (÷16 at c=1)
            grain_phase_randomness       *= lerp(1.0, 0.0,   c)
            grain_harmonic_lock          *= lerp(1.0, 1.0,   c)   (unchanged)
            grain_duration_s             *= lerp(0.5, 1.5,   c)
            grain_duration_jitter        *= lerp(1.0, 0.2,   c)
        """
        c = max(0.0, min(1.0, c))

        def sc(lo: float, hi: float) -> float:
            return lo + (hi - lo) * c

        return replace(
            self,
            grain_coherence              = c,
            grain_pitch_spread_semitones = self.grain_pitch_spread_semitones * sc(2.0, 0.125),
            grain_phase_randomness       = self.grain_phase_randomness       * sc(1.0, 0.0),
            grain_duration_s             = self.grain_duration_s             * sc(0.5, 1.5),
            grain_duration_jitter        = self.grain_duration_jitter        * sc(1.0, 0.2),
            attack_frac                  = self.attack_frac                  * sc(0.5, 1.5),
            release_frac                 = self.release_frac                 * sc(0.5, 1.5),
        )

    @classmethod
    def air(cls, center_frequency_hz: float = 4000.0, **kwargs) -> "GrainPopulationSpec":
        """Preset: decorrelated high-frequency air texture."""
        return cls(
            center_frequency_hz=center_frequency_hz,
            grain_pitch_spread_semitones=9.0,
            grain_phase_randomness=1.0,
            grain_harmonic_lock=0.0,
            grain_highband_bias=0.5,
            grain_density_hz=40.0,
            grain_duration_s=0.025,
            grain_duration_jitter=0.4,
            grain_chirp_depth=0.05,
            grain_chirp_jitter=0.8,
            grain_amp_jitter=0.4,
            gain=0.3,
            attack_frac=0.05,
            release_frac=0.25,
            grain_coherence=0.0,
            **kwargs,
        )

    @classmethod
    def body(cls, center_frequency_hz: float = 220.0, **kwargs) -> "GrainPopulationSpec":
        """Preset: coherent harmonic body texture."""
        return cls(
            center_frequency_hz=center_frequency_hz,
            grain_pitch_spread_semitones=1.5,
            grain_phase_randomness=0.1,
            grain_harmonic_lock=0.7,
            grain_highband_bias=0.0,
            grain_density_hz=15.0,
            grain_duration_s=0.10,
            grain_duration_jitter=0.15,
            grain_chirp_depth=0.01,
            grain_chirp_jitter=0.2,
            grain_manifold_mix=0.4,
            grain_amp_jitter=0.1,
            gain=0.8,
            attack_frac=0.20,
            release_frac=0.50,
            grain_coherence=1.0,
            **kwargs,
        )

    @classmethod
    def transient(cls, center_frequency_hz: float = 800.0, **kwargs) -> "GrainPopulationSpec":
        """Preset: short clustered transient bursts."""
        return cls(
            center_frequency_hz=center_frequency_hz,
            grain_pitch_spread_semitones=4.0,
            grain_phase_randomness=0.7,
            grain_harmonic_lock=0.0,
            grain_density_hz=8.0,
            burst_probability=0.8,
            burst_size=6,
            burst_spread_s=0.008,
            grain_duration_s=0.012,
            grain_duration_jitter=0.4,
            grain_chirp_depth=0.15,
            grain_chirp_jitter=0.9,
            grain_amp_jitter=0.35,
            gain=1.0,
            attack_frac=0.04,
            release_frac=0.20,
            grain_coherence=0.3,
            **kwargs,
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GrainPopulationSpec":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return [
            # Frequency
            KnobSpec("center_frequency_hz",          "Center freq",    "float", 440.0,  20.0,   20000.0, 0, "Hz", [], True,  "Frequency",    ".1f"),
            KnobSpec("grain_pitch_spread_semitones",  "Pitch spread",   "float", 3.0,    0.0,    24.0,    0, "st", [], False, "Frequency",    ".2f"),
            KnobSpec("grain_harmonic_lock",           "Harmonic lock",  "float", 0.0,    0.0,    1.0,     0, "",   [], False, "Frequency",    ".3f"),
            KnobSpec("grain_highband_bias",           "Highband bias",  "float", 0.0,    0.0,    3.0,     0, "oct",[], False, "Frequency",    ".2f"),
            # Birth process
            KnobSpec("grain_density_hz",              "Density",        "float", 20.0,   0.5,    500.0,   0, "/s", [], True,  "Birth",        ".1f"),
            KnobSpec("birth_jitter",                  "Jitter",         "float", 0.5,    0.0,    1.0,     0, "",   [], False, "Birth",        ".2f"),
            KnobSpec("burst_probability",             "Burst prob.",    "float", 0.0,    0.0,    1.0,     0, "",   [], False, "Birth",        ".2f"),
            KnobSpec("burst_size",                    "Burst size",     "int",   4,      2,      32,      1, "",   [], False, "Birth",        ".0f"),
            KnobSpec("burst_spread_s",                "Burst spread",   "float", 0.015,  0.001,  0.2,     0, "s",  [], True,  "Birth",        ".3f"),
            # Duration
            KnobSpec("grain_duration_s",              "Duration",       "float", 0.05,   0.002,  2.0,     0, "s",  [], True,  "Duration",     ".3f"),
            KnobSpec("grain_duration_jitter",         "Dur. jitter",    "float", 0.3,    0.0,    2.0,     0, "",   [], False, "Duration",     ".2f"),
            # Chirp
            KnobSpec("grain_chirp_depth",             "Chirp depth",    "float", 0.0,    0.0,    2.0,     0, "",   [], False, "Chirp",        ".3f"),
            KnobSpec("grain_chirp_jitter",            "Chirp jitter",   "float", 0.5,    0.0,    1.0,     0, "",   [], False, "Chirp",        ".2f"),
            # Phase
            KnobSpec("grain_phase_randomness",        "Phase rand.",    "float", 1.0,    0.0,    1.0,     0, "",   [], False, "Phase",        ".2f"),
            # Manifold
            KnobSpec("grain_manifold_mix",            "Manifold mix",   "float", 0.0,    0.0,    1.0,     0, "",   [], False, "Manifold",     ".2f"),
            # Amplitude
            KnobSpec("gain",                          "Gain",           "float", 1.0,    0.0,    4.0,     0, "",   [], False, "Amplitude",    ".3f"),
            KnobSpec("grain_amp_jitter",              "Amp jitter",     "float", 0.2,    0.0,    2.0,     0, "",   [], False, "Amplitude",    ".2f"),
            # Envelope
            KnobSpec("attack_frac",                   "Attack frac.",   "float", 0.15,   0.01,   0.5,     0, "",   [], False, "Envelope",     ".2f"),
            KnobSpec("release_frac",                  "Release frac.",  "float", 0.35,   0.01,   0.8,     0, "",   [], False, "Envelope",     ".2f"),
            # Coherence meta-knob + dynamic field
            KnobSpec("grain_coherence",   "Coherence",    "float",  0.5, 0.0, 1.0,  0, "",   [],  False, "Meta", ".2f"),
            KnobSpec("coherence_mode",    "Coh. mode",    "choice", "uniform", 0, 0, 0, "",
                     ["uniform", "sine", "random_walk", "burst", "gradient", "perlin_walk"],
                     False, "Meta", ""),
            KnobSpec("coherence_rate_hz", "Coh. rate",    "float",  0.5, 0.01, 10.0, 0, "Hz", [], False, "Meta", ".2f"),
            KnobSpec("coherence_depth",   "Coh. depth",   "float",  0.3, 0.0,  1.0,  0, "",   [], False, "Meta", ".2f"),
            KnobSpec("coherence_burst_tau_s", "Burst tau", "float", -1.0, -1.0, 10.0, 0, "s", [], False, "Meta", ".2f"),
            # Seed control
            KnobSpec("editor_seed",           "Editor seed",   "int",   0,    0, 999999, 1, "",  [], False, "Seed", ".0f"),
            KnobSpec("seed_animate",          "Animate seed",  "bool",  False, 0, 1,      0, "",  [], False, "Seed", ""),
            KnobSpec("seed_animate_period_s", "Anim. period",  "float", 4.0, 0.1, 30.0,  0, "s", [], False, "Seed", ".1f"),
        ]


# ---------------------------------------------------------------------------
# AnalyticGrain — synthesises one grain from its GrainSpec
# ---------------------------------------------------------------------------

class AnalyticGrain:
    """Synthesise a single analytic grain into a complex128 array.

    The grain signal in grain-local time t ∈ [0, duration]:

        φ(t) = phase_origin + 2π (f * t + ½ * chirp * t²)
        signal(t) = amplitude * envelope(t) * manifold(φ(t))

    The envelope is a cosine-raised window with separate attack / sustain /
    release fractions:

        attack  : 0 → 1  over attack_frac * duration
        sustain : 1.0     flat
        release : 1 → 0  over release_frac * duration

    The grain is returned at sample rate ``sr``; the caller places it into
    the output buffer at the correct birth sample offset.
    """

    @staticmethod
    def synthesize(spec: GrainSpec, sr: float) -> np.ndarray:
        """Return a complex128 array of length ceil(spec.duration * sr)."""
        n = max(1, int(math.ceil(spec.duration * sr)))
        t = np.arange(n, dtype=np.float64) / sr

        # --- phase path: linear chirp ---
        phase = spec.phase_origin + _PI2 * (spec.frequency_hz * t + 0.5 * spec.chirp_hz_per_s * t * t)

        # --- manifold evaluation ---
        if spec.manifold_mix <= 0.0:
            signal = np.exp(1j * phase)
        elif spec.manifold_mix >= 1.0:
            signal = _eval_harmonic(phase)
        else:
            signal = (
                (1.0 - spec.manifold_mix) * np.exp(1j * phase)
                + spec.manifold_mix * _eval_harmonic(phase)
            )

        # --- cosine envelope ---
        env = _cosine_envelope(n, spec.attack_frac, spec.release_frac)

        return spec.amplitude * env * signal

    @staticmethod
    def accumulate(
        spec: GrainSpec,
        buf: np.ndarray,
        sr: float,
    ) -> None:
        """Add grain into ``buf`` (complex128, 1-D) at the correct offset."""
        grain = AnalyticGrain.synthesize(spec, sr)
        i0 = int(math.floor(spec.birth_time * sr))
        i1 = min(i0 + len(grain), len(buf))
        if i0 < len(buf) and i1 > 0:
            i0c = max(i0, 0)
            buf[i0c:i1] += grain[i0c - i0 : i1 - i0]


# ---------------------------------------------------------------------------
# GranularClusterDriver — spawns and sums the full population
# ---------------------------------------------------------------------------

class GranularClusterDriver:
    """Spawn and synthesise a grain population from a GrainPopulationSpec.

    Parameters
    ----------
    spec:
        The population distribution specification.
    sr:
        Output sample rate in Hz.
    parent_phase_at:
        Optional callable ``(t: float) -> float`` returning the parent voice's
        phase at time t.  When provided and ``grain_phase_randomness < 1``,
        grains inherit partial phase from the parent.
    rng_seed:
        If given, the internal RNG is seeded for reproducibility.

    Usage
    -----
        driver = GranularClusterDriver(spec, sr=48_000)
        z = driver.synthesize(2.0)         # 2-second complex output
        grains = driver.spawn_grains(2.0)  # list[GrainSpec] without synthesis
    """

    def __init__(
        self,
        spec: GrainPopulationSpec,
        sr: float = 48_000.0,
        parent_phase_at: Optional[object] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._spec = spec
        self._sr = float(sr)
        self._parent_phase = parent_phase_at
        self._rng = random.Random(rng_seed)
        # Dynamic coherence field state (re-initialised per synthesize() call)
        self._coherence_walk:   float             = spec.grain_coherence
        self._coherence_walk_t: float             = 0.0
        self._coherence_bursts: List[float]       = []
        self._perlin_params:    List[Tuple[float, float]] = []

    @property
    def spec(self) -> GrainPopulationSpec:
        return self._spec

    @spec.setter
    def spec(self, s: GrainPopulationSpec) -> None:
        self._spec = s

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def spawn_grains(self, duration_s: float) -> List[GrainSpec]:
        """Draw a grain-birth schedule and return the resolved GrainSpec list."""
        births = self._birth_schedule(duration_s)
        return [self._resolve_grain(t, duration_s) for t in births]

    def synthesize(self, duration_s: float) -> np.ndarray:
        """Synthesize and accumulate all grains into a complex128 buffer."""
        self._init_coherence_field(duration_s)
        n_samples = int(math.ceil(duration_s * self._sr))
        buf = np.zeros(n_samples, dtype=np.complex128)
        grains = self.spawn_grains(duration_s)
        for g in grains:
            AnalyticGrain.accumulate(g, buf, self._sr)
        return buf

    def synthesize_grains(self, grains: Sequence[GrainSpec]) -> np.ndarray:
        """Synthesize from a pre-computed grain list (e.g. for rendering a subset)."""
        if not grains:
            return np.zeros(0, dtype=np.complex128)
        end_t = max(g.birth_time + g.duration for g in grains)
        n_samples = int(math.ceil(end_t * self._sr)) + 1
        buf = np.zeros(n_samples, dtype=np.complex128)
        for g in grains:
            AnalyticGrain.accumulate(g, buf, self._sr)
        return buf

    # ------------------------------------------------------------------
    # Dynamic coherence field
    # ------------------------------------------------------------------

    def _init_coherence_field(self, total_duration: float) -> None:
        """Prepare per-mode state before a synthesis run."""
        spec = self._spec
        c0   = max(0.0, min(1.0, spec.grain_coherence))
        mode = spec.coherence_mode

        if mode == "random_walk":
            self._coherence_walk   = c0
            self._coherence_walk_t = 0.0

        elif mode == "burst":
            # Poisson schedule of coherence-spike events
            rate   = max(0.01, spec.coherence_rate_hz)
            bursts: List[float] = []
            t = 0.0
            while t < total_duration:
                t += self._rng.expovariate(rate)
                if t < total_duration:
                    bursts.append(t)
            self._coherence_bursts = bursts

        elif mode == "perlin_walk":
            # 4 sine oscillators at octave-spaced rates with random phases
            base = max(0.01, spec.coherence_rate_hz)
            self._perlin_params = [
                (base * (0.5 + i * 0.618), self._rng.uniform(0.0, _PI2))
                for i in range(4)
            ]

    def _coherence_at(self, t: float, total_dur: float) -> float:
        """Evaluate the coherence field at grain birth time *t*."""
        spec = self._spec
        c0   = max(0.0, min(1.0, spec.grain_coherence))
        d    = max(0.0, min(1.0, spec.coherence_depth))
        mode = spec.coherence_mode

        if mode == "sine":
            raw = c0 + d * math.sin(_PI2 * spec.coherence_rate_hz * t)

        elif mode == "random_walk":
            # State updated in _resolve_grain before this call
            raw = self._coherence_walk

        elif mode == "burst":
            # Exponential decay from each burst spike back to c0.
            # M3 fix: use coherence_burst_tau_s when positive for independent
            # control over burst height vs. burst repetition rate.
            if spec.coherence_burst_tau_s > 0:
                tau = spec.coherence_burst_tau_s
            else:
                tau = max(0.02, 2.0 / max(0.01, spec.coherence_rate_hz))
            raw = c0
            for bt in self._coherence_bursts:
                if bt <= t:
                    raw += (1.0 - c0) * d * math.exp(-(t - bt) / tau)
            raw = min(1.0, raw)

        elif mode == "gradient":
            # Linear ramp from c0−d to c0+d across total duration
            frac = t / max(total_dur, 1e-9)
            raw  = c0 - d + 2.0 * d * frac

        elif mode == "perlin_walk":
            raw = c0
            n_osc = len(self._perlin_params)
            if n_osc:
                for freq, phase in self._perlin_params:
                    raw += (d / n_osc) * math.sin(_PI2 * freq * t + phase)

        else:  # "uniform" (default)
            raw = c0

        return max(0.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Birth schedule
    # ------------------------------------------------------------------

    def _birth_schedule(self, duration_s: float) -> List[float]:
        """Generate grain birth times within [0, duration_s].

        Mix of regular and Poisson inter-grain intervals controlled by
        ``birth_jitter`` (0=metronomic, 1=pure Poisson).  Bursts are
        injected with probability ``burst_probability`` at each slot.
        """
        spec = self._spec
        if spec.grain_density_hz <= 0.0 or duration_s <= 0.0:
            return []

        mean_isi = 1.0 / spec.grain_density_hz
        births: List[float] = []
        t = 0.0

        while t < duration_s:
            # Hybrid ISI: blend metronomic and exponential
            regular_isi = mean_isi
            poisson_isi = self._rng.expovariate(spec.grain_density_hz)
            jit = spec.birth_jitter
            isi = (1.0 - jit) * regular_isi + jit * poisson_isi

            if 0.0 < t < duration_s:
                births.append(t)
                # Burst injection
                if spec.burst_probability > 0.0 and self._rng.random() < spec.burst_probability:
                    for _ in range(spec.burst_size):
                        offset = self._rng.gauss(0.0, spec.burst_spread_s)
                        bt = t + offset
                        if 0.0 <= bt < duration_s:
                            births.append(bt)

            t += max(isi, 1.0 / self._sr)

        births.sort()
        return births

    # ------------------------------------------------------------------
    # Per-grain resolver
    # ------------------------------------------------------------------

    def _resolve_grain(self, birth_time: float, total_duration: float = 1.0) -> GrainSpec:
        spec = self._spec
        rng = self._rng

        # --- advance random-walk coherence state (called in birth-time order) ---
        if spec.coherence_mode == "random_walk":
            dt = birth_time - self._coherence_walk_t
            if dt > 0:
                step = rng.gauss(0.0, spec.coherence_depth * math.sqrt(max(dt, 1e-6)))
                self._coherence_walk = max(0.0, min(1.0, self._coherence_walk + step))
                self._coherence_walk_t = birth_time

        # --- apply per-grain coherence mapping (only if field is dynamic) ---
        c = self._coherence_at(birth_time, total_duration)
        if c != spec.grain_coherence:
            spec = spec.with_coherence(c)

        # --- duration (log-normal) ---
        dur = _lognormal_sample(rng, spec.grain_duration_s, spec.grain_duration_jitter)
        dur = max(dur, 2.0 / self._sr)

        # --- frequency ---
        freq = self._draw_frequency()

        # --- chirp ---
        chirp_max = spec.grain_chirp_depth * freq
        if chirp_max > 0.0:
            sign = 1.0 if rng.random() > 0.5 else -1.0
            mag = abs(rng.gauss(chirp_max, chirp_max * spec.grain_chirp_jitter))
            chirp = sign * mag
        else:
            chirp = 0.0

        # --- phase origin ---
        if spec.grain_phase_randomness >= 1.0:
            theta = rng.uniform(0.0, _PI2)
        elif spec.grain_phase_randomness <= 0.0:
            theta = self._parent_phase_at(birth_time)
        else:
            rand_theta = rng.uniform(0.0, _PI2)
            lock_theta = self._parent_phase_at(birth_time)
            theta = (
                spec.grain_phase_randomness * rand_theta
                + (1.0 - spec.grain_phase_randomness) * lock_theta
            )

        # --- amplitude (log-normal) ---
        amp = _lognormal_sample(rng, spec.gain, spec.grain_amp_jitter)

        # --- envelope fractions (clipped so attack+release ≤ 0.95) ---
        af = max(0.01, min(spec.attack_frac, 0.47))
        rf = max(0.01, min(spec.release_frac, 0.94 - af))

        return GrainSpec(
            birth_time     = birth_time,
            duration       = dur,
            frequency_hz   = freq,
            phase_origin   = theta,
            chirp_hz_per_s = chirp,
            amplitude      = amp,
            manifold_mix   = max(0.0, min(1.0, spec.grain_manifold_mix)),
            attack_frac    = af,
            release_frac   = rf,
        )

    def _draw_frequency(self) -> float:
        spec = self._spec
        rng = self._rng

        # log-normal draw around center_frequency_hz * highband_bias shift
        f_center = spec.center_frequency_hz * (2.0 ** spec.grain_highband_bias)
        f_semitone_std = spec.grain_pitch_spread_semitones
        if f_semitone_std > 0.0:
            semitones = rng.gauss(0.0, f_semitone_std)
            f_raw = f_center * (_SEMITONE ** semitones)
        else:
            f_raw = f_center

        # optional harmonic snap: attract toward nearest integer multiple
        lock = spec.grain_harmonic_lock
        if lock > 0.0 and spec.center_frequency_hz > 0.0:
            ratio = f_raw / spec.center_frequency_hz
            nearest = max(1.0, round(ratio))
            f_harmonic = nearest * spec.center_frequency_hz
            f_raw = (1.0 - lock) * f_raw + lock * f_harmonic

        return max(f_raw, 1.0)

    def _parent_phase_at(self, t: float) -> float:
        if self._parent_phase is not None:
            try:
                return float(self._parent_phase(t))
            except Exception:
                pass
        return 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_envelope(n: int, attack_frac: float, release_frac: float) -> np.ndarray:
    """Raised-cosine envelope: attack ramp, flat sustain, release ramp."""
    env = np.ones(n, dtype=np.float64)
    n_attack  = max(1, int(round(attack_frac * n)))
    n_release = max(1, int(round(release_frac * n)))
    # ensure they fit
    n_attack  = min(n_attack,  n)
    n_release = min(n_release, n - n_attack)

    # attack: 0 → 1
    if n_attack > 0:
        env[:n_attack] = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n_attack)))
    # release: 1 → 0
    if n_release > 0:
        env[n - n_release:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, n_release)))

    return env


def _eval_harmonic(phase: np.ndarray, max_harmonics: int = 8) -> np.ndarray:
    """Vectorised HarmonicMeasureManifold evaluation for a phase array."""
    acc = np.zeros(len(phase), dtype=np.complex128)
    norm = 0.0
    for n in range(1, max_harmonics + 1):
        c = 1.0 / n
        acc += c * np.exp(1j * n * phase)
        norm += c
    return acc / norm


def _lognormal_sample(rng: random.Random, mean: float, cv: float) -> float:
    """Sample from a log-normal distribution with given mean and coefficient of variation."""
    if cv <= 0.0:
        return mean
    # log-normal params from mean and cv
    sigma2 = math.log(1.0 + cv * cv)
    mu = math.log(max(mean, 1e-12)) - 0.5 * sigma2
    return math.exp(rng.gauss(mu, math.sqrt(sigma2)))
