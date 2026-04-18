"""refraction_engine.py

Physics-informed refractive ray propagation in time-frequency space.

Each voice is a "ray" carrying state (t, f, phi, A).  When a ray hits a
RefractionBoundary the incident ray terminates at the boundary time and one
or more transmitted children are spawned that inherit the exact phase at the
split and continue with refracted frequencies and scaled amplitudes.  The
reflected component (backward in time) is discarded as physically lost.

Energy conservation at each split:
    A_transmitted² + A_reflected² = A_incident²
    A_reflected is never rendered — it leaves the scene.

The resulting ray tree rendered as a sum of phase-continuous gated sinusoids
sounds like laser light shining through fog: a single origin that fans into a
complex interference field of phase-coherent paths with frequency bending at
every scattering event.

Quick start (fixed boundaries)
-------------------------------
    from refraction_engine import RefractionBoundary, RefractivePropagator, RefractionScene

    bounds = [
        RefractionBoundary(t_seconds=0.6,  freq_ratio=0.72, transmission_fraction=0.70),
        RefractionBoundary(t_seconds=1.3,  freq_ratio=1.35, transmission_fraction=0.80),
    ]
    rays  = RefractivePropagator(bounds, scene_end=2.5).propagate(initial_freq_hz=330.0)
    RefractionScene(rays, scene_end=2.5).write_wav("refraction_fixed.wav")

Quick start (stochastic fog)
-----------------------------
    from refraction_engine import FogField, RefractionScene

    fog  = FogField(event_rate_hz=3.0, branch_prob=0.30, seed=42)
    rays = fog.propagate(initial_freq_hz=220.0, scene_end=5.0, max_rays=96)
    RefractionScene(rays, scene_end=5.0).write_wav("refraction_fog.wav")
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from signal_generator_v2 import (
    ConstantPhasePath,
    LatticeVoice,
    NullDriftModel,
    PI2,
    PureSineManifold,
    TimeField,
    WaveformManifold,
)


# ============================================================
# Windowed amplitude field
# ============================================================

class WindowedAmplitudeField(TimeField):
    """
    Amplitude nonzero only in the open interval (t_start, t_end), with
    cosine fade-in and fade-out of `fade_s` seconds to suppress boundary
    clicks.  Outside the window the value is exactly 0.0.
    """

    def __init__(
        self,
        base_amplitude: float,
        t_start: float,
        t_end: float,
        fade_s: float = 0.002,
    ) -> None:
        span = t_end - t_start
        if span <= 0.0:
            raise ValueError("t_end must be > t_start")
        self._A    = base_amplitude
        self._t0   = t_start
        self._t1   = t_end
        self._fade = min(fade_s, span * 0.25)

    def value(self, t: float) -> float:
        if t <= self._t0 or t >= self._t1:
            return 0.0
        dt0 = t - self._t0
        dt1 = self._t1 - t
        fi = 0.5 * (1.0 - math.cos(math.pi * dt0 / self._fade)) if dt0 < self._fade else 1.0
        fo = 0.5 * (1.0 - math.cos(math.pi * dt1 / self._fade)) if dt1 < self._fade else 1.0
        return self._A * fi * fo


# ============================================================
# Ray primitives
# ============================================================

@dataclass(frozen=True)
class RefractionBoundary:
    """
    A refractive interface at a fixed time.

    freq_ratio
        Ratio of transmitted to incoming frequency (n1/n2 in Snell's analogy).
        < 1 → frequency compresses (entering denser medium).
        > 1 → frequency expands  (entering faster medium).

    transmission_fraction
        Fraction of *energy* (not amplitude) that is transmitted.
        A_transmitted = A_incident * sqrt(transmission_fraction).
        Reflected fraction is discarded.  Must be in (0, 1].

    branch_count
        Number of transmitted children.  When > 1, extra branches receive
        small independent frequency perturbations around freq_ratio and share
        the transmitted energy equally (A_branch = A_t / sqrt(branch_count)).

    freq_spread
        Half-width of the uniform frequency perturbation applied to extra
        branches.  branch_count=1 ignores this.

    phase_shift_radians
        Optional phase added to all transmitted children at the boundary
        (models a half-wave shift or interface-specific phase lag).
    """
    t_seconds: float
    freq_ratio: float = 0.80
    transmission_fraction: float = 0.70
    branch_count: int = 1
    freq_spread: float = 0.06
    phase_shift_radians: float = 0.0


@dataclass
class RaySegment:
    """
    One forward-propagating ray active in the interval [t_start, t_end).

    The ray is a constant-frequency analytic sinusoid whose phase at t_start
    is phase_at_start.  Children spawned from this ray inherit that phase
    exactly at t_end (the boundary time) plus any accumulated propagation.
    """
    t_start: float
    t_end: float
    frequency_hz: float
    phase_at_start: float   # φ(t_start), exact, in radians
    amplitude: float        # amplitude within this segment


# ============================================================
# Propagator
# ============================================================

class RefractivePropagator:
    """
    Expands an initial ray into a tree of forward-propagating RaySegments.

    The algorithm is a breadth-first expansion of a work queue.  Each entry
    in the queue is one pending ray; at each boundary it encounters, the ray
    is terminated and transmitted children are enqueued.  The reflected
    component is never enqueued.

    Termination conditions (whichever comes first):
      * No further boundaries remain → ray runs to scene_end.
      * Ray amplitude drops below min_amplitude → ray is dropped.
      * Total segment count reaches max_rays → expansion stops.
    """

    def __init__(
        self,
        boundaries: List[RefractionBoundary],
        scene_end: float,
        max_rays: int = 128,
        min_amplitude: float = 0.01,
        rng: Optional[random.Random] = None,
    ) -> None:
        if scene_end <= 0.0:
            raise ValueError("scene_end must be > 0")
        self._boundaries = sorted(boundaries, key=lambda b: b.t_seconds)
        self._scene_end  = scene_end
        self._max_rays   = max_rays
        self._min_amp    = min_amplitude
        self._rng        = rng or random.Random()

    def propagate(
        self,
        initial_freq_hz: float,
        initial_amplitude: float = 1.0,
        initial_phase: float = 0.0,
        t_start: float = 0.0,
    ) -> List[RaySegment]:
        """
        Return all active ray segments, sorted by t_start.

        Phase continuity is guaranteed: every child inherits the parent's
        exact accumulated phase at the split time.
        """
        segments: List[RaySegment] = []

        # Queue entries: (t_start, freq_hz, phase_at_t_start, amplitude, next_boundary_search_idx)
        PendingRay = Tuple[float, float, float, float, int]
        queue: List[PendingRay] = [
            (t_start, initial_freq_hz, initial_phase, initial_amplitude, 0)
        ]

        while queue and len(segments) < self._max_rays:
            t0, freq, phi0, amp, b_search_from = queue.pop(0)

            if amp < self._min_amp:
                continue

            # Find the first boundary strictly after t0 (boundaries at or before t0 are skipped)
            b_idx = b_search_from
            while b_idx < len(self._boundaries) and self._boundaries[b_idx].t_seconds <= t0:
                b_idx += 1

            if b_idx >= len(self._boundaries):
                # Terminal: ray runs to scene_end
                if t0 < self._scene_end:
                    segments.append(RaySegment(
                        t_start=t0,
                        t_end=self._scene_end,
                        frequency_hz=freq,
                        phase_at_start=phi0,
                        amplitude=amp,
                    ))
                continue

            bnd   = self._boundaries[b_idx]
            t_hit = bnd.t_seconds

            if t_hit >= self._scene_end:
                # Boundary is outside the scene — treat as terminal
                segments.append(RaySegment(
                    t_start=t0,
                    t_end=self._scene_end,
                    frequency_hz=freq,
                    phase_at_start=phi0,
                    amplitude=amp,
                ))
                continue

            # Segment from t0 to the boundary
            segments.append(RaySegment(
                t_start=t0,
                t_end=t_hit,
                frequency_hz=freq,
                phase_at_start=phi0,
                amplitude=amp,
            ))

            # Phase accumulated by this ray up to the boundary (constant-freq path)
            phi_hit = phi0 + PI2 * freq * (t_hit - t0) + bnd.phase_shift_radians

            # Transmitted amplitude (reflected is discarded)
            A_t = amp * math.sqrt(max(0.0, min(1.0, bnd.transmission_fraction)))

            # Distribute equally across all branches: A_branch = A_t / sqrt(n)
            n_branches = max(1, bnd.branch_count)
            A_branch   = A_t / math.sqrt(n_branches)

            for k in range(n_branches):
                if k == 0:
                    f_child = freq * bnd.freq_ratio
                else:
                    perturb = 1.0 + self._rng.uniform(-bnd.freq_spread, bnd.freq_spread)
                    f_child = freq * bnd.freq_ratio * perturb

                f_child = max(1.0, f_child)  # keep above 1 Hz
                queue.append((t_hit, f_child, phi_hit, A_branch, b_idx + 1))

        return sorted(segments, key=lambda s: s.t_start)


# ============================================================
# Stochastic fog
# ============================================================

class FogField:
    """
    Generates RefractionBoundaries via a Poisson process (memoryless random
    arrivals) and propagates an initial ray through them.

    Parameters
    ----------
    event_rate_hz
        Average number of boundary events per second.
    n_ratio_range
        (min, max) uniform range for freq_ratio.
    transmission_range
        (min, max) uniform range for transmission_fraction.
    branch_prob
        Probability that any given boundary spawns 2 transmitted children
        (prismatic split) rather than 1.
    freq_spread
        Per-branch frequency perturbation half-width when branch_count > 1.
    seed
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        event_rate_hz: float = 2.5,
        n_ratio_range: Tuple[float, float] = (0.65, 1.45),
        transmission_range: Tuple[float, float] = (0.55, 0.85),
        branch_prob: float = 0.25,
        freq_spread: float = 0.08,
        seed: Optional[int] = None,
    ) -> None:
        self._rate    = event_rate_hz
        self._n_range = n_ratio_range
        self._t_range = transmission_range
        self._bprob   = branch_prob
        self._spread  = freq_spread
        self._rng     = random.Random(seed)

    def sample_boundaries(
        self, scene_end: float, t_start: float = 0.0
    ) -> List[RefractionBoundary]:
        """Draw boundaries from a Poisson process over [t_start, scene_end]."""
        boundaries: List[RefractionBoundary] = []
        t = t_start
        while True:
            inter = -math.log(max(1e-15, self._rng.random())) / self._rate
            t += inter
            if t >= scene_end:
                break
            boundaries.append(RefractionBoundary(
                t_seconds=t,
                freq_ratio=self._rng.uniform(*self._n_range),
                transmission_fraction=self._rng.uniform(*self._t_range),
                branch_count=2 if self._rng.random() < self._bprob else 1,
                freq_spread=self._spread,
            ))
        return boundaries

    def propagate(
        self,
        initial_freq_hz: float,
        scene_end: float,
        initial_amplitude: float = 1.0,
        initial_phase: float = 0.0,
        max_rays: int = 128,
    ) -> List[RaySegment]:
        """Sample boundaries and propagate; convenience wrapper."""
        bnd = self.sample_boundaries(scene_end)
        prop = RefractivePropagator(
            bnd, scene_end, max_rays=max_rays, rng=self._rng
        )
        return prop.propagate(initial_freq_hz, initial_amplitude, initial_phase)


# ============================================================
# Render scene
# ============================================================

def _windowed_amp_array(
    base: float,
    t_start: float,
    t_end: float,
    fade_s: float,
    t_arr: np.ndarray,
) -> np.ndarray:
    """Vectorised windowed amplitude for one ray segment."""
    span = t_end - t_start
    fade = min(fade_s, span * 0.25)
    amp  = np.zeros_like(t_arr)
    mask = (t_arr > t_start) & (t_arr < t_end)
    if not mask.any():
        return amp
    t_m  = t_arr[mask]
    dt0  = t_m - t_start
    dt1  = t_end - t_m
    fi   = np.where(dt0 < fade, 0.5 * (1.0 - np.cos(np.pi * dt0 / fade)), 1.0)
    fo   = np.where(dt1 < fade, 0.5 * (1.0 - np.cos(np.pi * dt1 / fade)), 1.0)
    amp[mask] = base * fi * fo
    return amp


class RefractionScene:
    """
    Renders a list of RaySegments to a float64 numpy array (or WAV file).

    Each segment is evaluated as:
        x(t) = A_window(t) * cos(phase_at_start + 2π·f·(t − t_start))

    All segments are summed.  The render is fully vectorised (one numpy
    array operation per segment), so even 100+ rays over 5 seconds is fast.

    Parameters
    ----------
    rays
        Output of RefractivePropagator.propagate or FogField.propagate.
    scene_end
        Total scene duration in seconds.
    output_sample_rate
        Output audio sample rate in Hz.
    fade_s
        Duration of cosine fade-in/fade-out at each segment boundary.
    """

    def __init__(
        self,
        rays: List[RaySegment],
        scene_end: float,
        output_sample_rate: float = 48_000.0,
        fade_s: float = 0.002,
    ) -> None:
        if not rays:
            raise ValueError("RefractionScene requires at least one ray segment")
        self._rays  = rays
        self._end   = scene_end
        self._sr    = output_sample_rate
        self._fade  = fade_s

    def render_numpy(self) -> np.ndarray:
        """Return a float64 array of real-valued audio samples."""
        n       = int(math.ceil(self._end * self._sr)) + 1
        t_arr   = np.linspace(0.0, self._end, n)
        out     = np.zeros(n, dtype=np.float64)

        for ray in self._rays:
            amp_arr   = _windowed_amp_array(ray.amplitude, ray.t_start, ray.t_end, self._fade, t_arr)
            # Phase: φ_start + 2π·f·(t − t_start)
            phase_arr = ray.phase_at_start + PI2 * ray.frequency_hz * (t_arr - ray.t_start)
            out      += amp_arr * np.cos(phase_arr)

        return out

    def build_lattice_voices(self) -> List[LatticeVoice]:
        """
        Return a list of LatticeVoice objects — one per ray segment — for use
        with the WitnessAwareSynthDriver pipeline if desired.

        Phase continuity is guaranteed: for each voice, ConstantPhasePath is
        configured so path.phase(t_start) == ray.phase_at_start exactly.
        """
        voices = []
        for ray in self._rays:
            # phase0 such that phase0 + 2π·f·t_start == ray.phase_at_start
            phase0 = ray.phase_at_start - PI2 * ray.frequency_hz * ray.t_start
            path   = ConstantPhasePath(frequency_hz=ray.frequency_hz, phase0=phase0)
            amp_f  = WindowedAmplitudeField(ray.amplitude, ray.t_start, ray.t_end, self._fade)
            voices.append(LatticeVoice(
                name=f"ray@{ray.t_start:.3f}_{ray.frequency_hz:.1f}Hz",
                harmonic_index=1,
                master_phase_path=path,
                waveform_manifold=PureSineManifold(),
                amplitude_field=amp_f,
                drift_model=NullDriftModel(),
                gain=1.0,
                phase_offset=0.0,
            ))
        return voices

    def write_wav(self, path: str) -> None:
        """Render and write a normalised 32-bit float mono WAV file."""
        import scipy.io.wavfile as _wav
        arr  = self.render_numpy()
        peak = float(np.abs(arr).max()) or 1.0
        f32  = (arr / peak).astype(np.float32)
        _wav.write(path, int(self._sr), f32)
        n_active = len(self._rays)
        print(
            f"Written {path}: {len(f32)} samples @ {int(self._sr)} Hz | "
            f"{n_active} ray segments | peak={peak:.4f}"
        )

    def print_tree(self) -> None:
        """Print a human-readable summary of the ray tree."""
        print(f"  {'t_start':>8}  {'t_end':>8}  {'freq_hz':>9}  {'amplitude':>10}")
        print("  " + "-" * 44)
        for r in self._rays:
            print(
                f"  {r.t_start:8.3f}  {r.t_end:8.3f}"
                f"  {r.frequency_hz:9.2f}  {r.amplitude:10.5f}"
            )

    def spectral_summary(self) -> None:
        """Print per-segment frequency and energy contribution."""
        total_energy = sum(r.amplitude ** 2 * (r.t_end - r.t_start) for r in self._rays)
        print(f"  {'freq_hz':>9}  {'energy%':>8}  {'duration':>9}")
        print("  " + "-" * 32)
        for r in sorted(self._rays, key=lambda x: x.frequency_hz):
            e = r.amplitude ** 2 * (r.t_end - r.t_start)
            print(
                f"  {r.frequency_hz:9.2f}  {100*e/total_energy:7.2f}%"
                f"  {r.t_end-r.t_start:8.3f}s"
            )

    # ------------------------------------------------------------------
    # Analytical TF image
    # ------------------------------------------------------------------

    def render_tf_image(
        self,
        n_time: int = 800,
        n_freq: int = 600,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        log_freq: bool = False,
        heisenberg_spread: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render a time-frequency amplitude image from ray geometry — no FFT.

        Each ray segment is a rectangle in TF space at its exact frequency.
        The frequency-axis spread is σ_f = heisenberg_spread / duration,
        which equals the Heisenberg uncertainty minimum at spread=1.0 and
        is a purely analytical quantity: no windowing, no leakage.

        Long-lived rays appear as sharp horizontal lines; brief rays are
        spectrally broad — exactly as the uncertainty principle requires,
        but without any estimation error.

        Parameters
        ----------
        heisenberg_spread
            Multiplier on σ_f.  1.0 = theoretical minimum.  Values of
            1.0–3.0 give progressively smoother images.
        log_freq
            Use a logarithmic frequency axis (good for wide-range scenes).

        Returns
        -------
        image  : (n_freq, n_time) float64 amplitude array.
        t_axis : (n_time,) seconds.
        f_axis : (n_freq,) Hz.
        """
        fmin = freq_min or max(1.0, min(r.frequency_hz for r in self._rays) * 0.6)
        fmax = freq_max or max(r.frequency_hz for r in self._rays) * 1.5

        t_axis = np.linspace(0.0, self._end, n_time)
        if log_freq:
            f_axis = np.logspace(np.log10(fmin), np.log10(fmax), n_freq)
        else:
            f_axis = np.linspace(fmin, fmax, n_freq)

        image = np.zeros((n_freq, n_time), dtype=np.float64)

        for ray in self._rays:
            duration = max(ray.t_end - ray.t_start, 1e-3)
            sigma_f  = heisenberg_spread / duration

            # Windowed amplitude along time axis — same window as audio render.
            amp_t   = _windowed_amp_array(ray.amplitude, ray.t_start, ray.t_end, self._fade, t_axis)

            # Gaussian in frequency — exact because ray has a single known frequency.
            f_gauss = np.exp(-0.5 * ((f_axis - ray.frequency_hz) / sigma_f) ** 2)

            # Outer product: each time column gets the Gaussian scaled by amp at that time.
            image += np.outer(f_gauss, amp_t)

        return image, t_axis, f_axis

    def save_tf_png(
        self,
        path: str,
        n_time: int = 800,
        n_freq: int = 600,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        log_freq: bool = False,
        log_amplitude: bool = True,
        heisenberg_spread: float = 1.5,
        dpi: int = 150,
    ) -> None:
        """Render the analytical TF image and save as PNG. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required — pip install matplotlib")

        image, t_axis, f_axis = self.render_tf_image(
            n_time=n_time, n_freq=n_freq,
            freq_min=freq_min, freq_max=freq_max,
            log_freq=log_freq, heisenberg_spread=heisenberg_spread,
        )

        img = np.log1p(image) if log_amplitude else image

        fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
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
        ax.set_title(
            f"Analytical TF — {len(self._rays)} ray segments | "
            f"{self._end:.1f}s | σ_f = spread/duration (Heisenberg ×{heisenberg_spread})"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"TF image saved: {path}  [{n_time}×{n_freq} px]")


# ============================================================
# Demos
# ============================================================

def demo_fixed(output_path: str = "refraction_fixed.wav") -> None:
    """
    Single 330 Hz voice passing through three hard refractive boundaries.
    The voice refracts at each crossing: frequency bends, amplitude drops,
    phase is carried exactly.  The result is a single ray chain.
    """
    boundaries = [
        RefractionBoundary(t_seconds=0.60, freq_ratio=0.72, transmission_fraction=0.70),
        RefractionBoundary(t_seconds=1.30, freq_ratio=1.40, transmission_fraction=0.80),
        RefractionBoundary(t_seconds=2.10, freq_ratio=0.85, transmission_fraction=0.75),
    ]
    rays = RefractivePropagator(boundaries, scene_end=3.0).propagate(
        initial_freq_hz=330.0, initial_amplitude=1.0
    )
    print("=== Fixed boundary ray tree ===")
    scene = RefractionScene(rays, scene_end=3.0)
    scene.print_tree()
    scene.write_wav(output_path)


def demo_fog(output_path: str = "refraction_fog.wav") -> None:
    """
    Single 220 Hz origin scattered through probabilistic Poisson fog.
    Each scattering event may branch into two transmitted paths, creating
    an exponentially growing ray tree (capped at max_rays).
    """
    fog  = FogField(
        event_rate_hz=2.5,
        n_ratio_range=(0.60, 1.50),
        transmission_range=(0.55, 0.85),
        branch_prob=0.35,
        freq_spread=0.10,
        seed=7,
    )
    rays = fog.propagate(
        initial_freq_hz=220.0,
        scene_end=5.0,
        initial_amplitude=1.0,
        max_rays=96,
    )
    print("=== Stochastic fog ray tree ===")
    scene = RefractionScene(rays, scene_end=5.0)
    scene.print_tree()
    print()
    scene.spectral_summary()
    scene.write_wav(output_path)


def demo_harmonic_fog(output_path: str = "refraction_harmonic_fog.wav") -> None:
    """
    Three harmonically related voices (f, 2f, 3f) each scattered through
    their own independent fog — then summed.  Models a harmonic tone that
    enters a dispersive medium where each partial takes a different path.
    """
    import scipy.io.wavfile as _wav
    scene_end = 5.0
    sr        = 48_000.0
    roots     = [110.0, 220.0, 330.0]
    gains     = [1.0,   0.6,   0.4]

    combined  = np.zeros(int(math.ceil(scene_end * sr)) + 1, dtype=np.float64)

    for root, gain in zip(roots, gains):
        fog  = FogField(
            event_rate_hz=2.0,
            n_ratio_range=(0.65, 1.40),
            transmission_range=(0.60, 0.85),
            branch_prob=0.30,
            freq_spread=0.08,
            seed=hash(root) & 0xFFFF,
        )
        rays = fog.propagate(root, scene_end=scene_end, initial_amplitude=gain, max_rays=64)
        combined += RefractionScene(rays, scene_end, sr).render_numpy()

    peak = float(np.abs(combined).max()) or 1.0
    f32  = (combined / peak).astype(np.float32)
    _wav.write(output_path, int(sr), f32)
    print(f"Written {output_path}: {len(f32)} samples @ {int(sr)} Hz | 3 harmonic fog streams")


if __name__ == "__main__":
    demo_fixed()
    print()
    demo_fog()
    print()
    demo_harmonic_fog()
