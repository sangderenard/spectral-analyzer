"""sequence_engine.py

Assembles sequences of analytic notes (rendered by signal_generator_v2) into
a single phase-continuous global AdaptiveSampleBuffer that can be projected
to audio with any of the numpy / torch projection paths.

Core pipeline
─────────────
ArpeggioRule ──► NoteSchedule ──► SequenceRenderer ──► AdaptiveSampleBuffer
                                          │
                                   LatticeBuilder   (one driver per note)
                                   PhaseHandoff     (terminal → seed phase)
                                   TimbrePreset     (shared timbral params)

MIDI (future)
─────────────
MidiAdapter.load(path) → NoteSchedule    (requires:  pip install mido)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from signal_generator_v2 import (
    AdaptiveSampleBuffer,
    ConstantPhasePath,
    DensityPolicy,
    DriftModel,
    EmissionRange,
    ExponentialEnvelope,
    HarmonicLattice,
    LatticeVoice,
    NullDriftModel,
    PhaseWarpedManifold,
    ProjectionPolicy,
    PureSineManifold,
    SigmoidField,
    SmoothSinusoidalDriftModel,
    WaveformManifold,
    WitnessAwareSynthDriver,
    WitnessThresholds,
    PI2,
)


# ============================================================
# Modal scale maps  (semitone intervals from root, one octave)
# ============================================================

MODAL_SCALES: Dict[str, List[int]] = {
    # Church modes
    "ionian":           [0, 2, 4, 5, 7, 9, 11],   # major
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "lydian":           [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "aeolian":          [0, 2, 3, 5, 7, 8, 10],   # natural minor
    "locrian":          [0, 1, 3, 5, 6, 8, 10],
    # Extended
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":    [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "whole_tone":       [0, 2, 4, 6, 8, 10],
    "chromatic":        list(range(12)),
}


def semitones_to_hz(root_hz: float, semitones: float) -> float:
    """Equal-temperament conversion: Hz for *semitones* above *root_hz*."""
    return root_hz * (2.0 ** (semitones / 12.0))


def scale_degrees_hz(
    root_hz: float,
    scale: str,
    octave_span: int = 2,
) -> List[float]:
    """Return the Hz values for every scale degree over *octave_span* octaves."""
    if scale not in MODAL_SCALES:
        raise ValueError(
            f"Unknown scale '{scale}'.  Available: {sorted(MODAL_SCALES)}"
        )
    intervals = MODAL_SCALES[scale]
    result: List[float] = []
    for octave in range(octave_span):
        for semitones in intervals:
            result.append(semitones_to_hz(root_hz, semitones + 12 * octave))
    return result


# ============================================================
# Note primitives
# ============================================================

@dataclass
class NoteEvent:
    """One rendered note in the global timeline."""
    fundamental_hz: float
    start_time: float           # seconds, global timeline
    duration_s: float
    velocity: float = 1.0       # 0–1, applied as amplitude scale
    partial_count: int = 6
    # Populated by PhaseHandoff before each render; do not set manually
    # unless you know exactly what you want.
    phase_hints: Dict[int, float] = field(default_factory=dict)
    # harmonic_index → phase_offset in radians at local t=0

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration_s


@dataclass
class NoteSchedule:
    """Ordered (by start_time) collection of NoteEvents."""
    events: List[NoteEvent] = field(default_factory=list)

    def add(self, event: NoteEvent) -> None:
        self.events.append(event)
        self.events.sort(key=lambda e: e.start_time)

    @property
    def total_duration(self) -> float:
        if not self.events:
            return 0.0
        return max(e.end_time for e in self.events)

    def is_monophonic(self) -> bool:
        """True when no two notes overlap in time."""
        for i in range(1, len(self.events)):
            if self.events[i].start_time < self.events[i - 1].end_time:
                return False
        return True


# ============================================================
# Arpeggiation
# ============================================================

@dataclass
class ArpeggioRule:
    """
    Generates a NoteSchedule from a root pitch, a modal scale, and a rhythm.

    Parameters
    ----------
    root_hz:
        Fundamental of the lowest note in the pattern.
    scale:
        Key into MODAL_SCALES.  Patterns index into the flattened degree list.
    pattern:
        0-based scale-degree indices.  Wraps modulo the available degrees.
    rhythm_beats:
        Duration of each rhythmic step in beats.  Cycles to fill *pattern*.
    bpm:
        Tempo in beats per minute.
    velocity_curve:
        Per-step amplitude 0–1.  Cycles to fill *pattern*.
    legato_fraction:
        Actual note duration = step_duration × legato_fraction.
        1.0 = fully legato (notes touch), <1 = gaps between notes.
    partial_count:
        Harmonic partials per note.
    octave_span:
        How many octaves of scale degrees to build the frequency table from.
    repeats:
        How many times to repeat the full pattern.
    """
    root_hz: float
    scale: str = "pentatonic_minor"
    pattern: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 3, 2, 1])
    rhythm_beats: List[float] = field(default_factory=lambda: [0.5])
    bpm: float = 120.0
    velocity_curve: List[float] = field(default_factory=lambda: [1.0])
    legato_fraction: float = 0.85
    partial_count: int = 6
    octave_span: int = 2
    repeats: int = 1

    def generate(self) -> NoteSchedule:
        degrees      = scale_degrees_hz(self.root_hz, self.scale, self.octave_span)
        beat_seconds = 60.0 / self.bpm
        schedule     = NoteSchedule()
        t            = 0.0
        n_steps      = len(self.pattern) * self.repeats

        for step in range(n_steps):
            pat_idx  = step % len(self.pattern)
            deg_idx  = self.pattern[pat_idx] % len(degrees)
            hz       = degrees[deg_idx]
            step_dur = self.rhythm_beats[step % len(self.rhythm_beats)] * beat_seconds
            note_dur = max(step_dur * self.legato_fraction, 1.0 / 48000.0)
            velocity = self.velocity_curve[step % len(self.velocity_curve)]

            schedule.add(NoteEvent(
                fundamental_hz=hz,
                start_time=t,
                duration_s=note_dur,
                velocity=velocity,
                partial_count=self.partial_count,
            ))
            t += step_dur

        return schedule


# ============================================================
# Timbre preset
# ============================================================

@dataclass
class TimbrePreset:
    """
    Timbral parameters shared across every note in a rendered sequence.

    amplitude_decay_tau:
        Exponential amplitude envelope τ in seconds.  Shorter = more percussive.
    shape_low / shape_high / shape_center / shape_width:
        Sigmoid shape-state parameters passed to the waveform manifold.
    """
    waveform_manifold: Optional[WaveformManifold] = None
    drift_model: Optional[DriftModel] = None
    amplitude_decay_tau: float = 0.25
    shape_low: float = 0.1
    shape_high: float = 0.7
    shape_center: float = 0.10
    shape_width: float = 0.06

    @classmethod
    def default(cls) -> TimbrePreset:
        """Slightly warped-phase partials with gentle slow drift."""
        return cls(
            waveform_manifold=PhaseWarpedManifold(harmonic_warp_strength=0.20),
            drift_model=SmoothSinusoidalDriftModel(
                base_rate_hz=0.07,
                max_offset_hz=0.03,
                harmonic_scaling_power=1.0,
            ),
        )

    @classmethod
    def pure_sine(cls) -> TimbrePreset:
        """Pure analytic sinusoid, no drift, no warp."""
        return cls(
            waveform_manifold=PureSineManifold(),
            drift_model=NullDriftModel(),
            amplitude_decay_tau=1.0,
            shape_low=0.0,
            shape_high=0.0,
            shape_center=0.0,
            shape_width=1.0,
        )


# ============================================================
# Phase handoff
# ============================================================

class PhaseHandoff:
    """
    Computes phase offsets that carry the end-of-note phase into the start
    of the next note, guaranteeing zero discontinuity in every harmonic.

    Monophonic continuity
    ─────────────────────
    At the end of note A (local time *dur_A*), query:

        phases = PhaseHandoff.terminal_phases(driver_A, dur_A)

    Inject into note B:

        note_B.phase_hints = phases

    Then render B on local time [0, dur_B].  Every harmonic partial picks up
    exactly where it left off — no click, no phase jump.

    Harmonic lock for polyphony
    ───────────────────────────
    When notes at different pitches start simultaneously and you want their
    harmonic structures to be constructively related, scale the terminal phases
    by the frequency ratio:

        note_B.phase_hints = PhaseHandoff.harmonic_lock_phases(
            phases, source_hz=f_A, target_hz=f_B
        )

    This places B's fundamental at the same fractional phase cycle as A's,
    scaled by f_B/f_A.
    """

    @staticmethod
    def terminal_phases(
        driver: WitnessAwareSynthDriver,
        local_duration: float,
    ) -> Dict[int, float]:
        """
        Returns ``{harmonic_index: phase_radians}`` for every voice at
        *local_duration* seconds into the note's local render window.
        """
        return {
            v.harmonic_index: v.approximate_effective_phase(local_duration)
            for v in driver.lattice.voices
        }

    @staticmethod
    def harmonic_lock_phases(
        source_phases: Dict[int, float],
        source_hz: float,
        target_hz: float,
    ) -> Dict[int, float]:
        """
        Scale terminal phases of *source_hz* to the frequency space of
        *target_hz*.  The scaling factor is the interval ratio target/source,
        so the i-th harmonic of the new note starts at the same fractional
        cycle position as if it were a continuation of the old note at its
        corresponding harmonic frequency.
        """
        if source_hz == 0.0:
            return dict(source_phases)
        ratio = target_hz / source_hz
        return {h: (phi * ratio) % PI2 for h, phi in source_phases.items()}


# ============================================================
# Lattice builder
# ============================================================

class LatticeBuilder:
    """
    Constructs a ``WitnessAwareSynthDriver`` for a single ``NoteEvent``
    using a ``ConstantPhasePath`` (stable pitch) as the master path.

    All timbral parameters come from the supplied ``TimbrePreset``; per-note
    amplitude is scaled by ``note.velocity / harmonic_index``.

    Rendering settings (witness thresholds, density, projection) are set once
    at construction and reused for every note.
    """

    def __init__(
        self,
        timbre: Optional[TimbrePreset] = None,
        witness_thresholds: Optional[WitnessThresholds] = None,
        density_policy: Optional[DensityPolicy] = None,
        projection_policy: Optional[ProjectionPolicy] = None,
    ) -> None:
        self._timbre     = timbre or TimbrePreset.default()
        self._witness    = witness_thresholds or WitnessThresholds(
            backward_phase_radians=PI2,
            forward_phase_radians=PI2,
            max_support_seconds=3.0,
            support_search_step_seconds=0.001,
            integration_steps_per_check=32,
        )
        self._density    = density_policy or DensityPolicy(
            oversampling_factor=16.0,
            min_sample_rate=48_000.0,
            max_sample_rate=2_000_000.0,
            derivative_weight=0.5,
            support_weight=1.0,
        )
        self._projection = projection_policy or ProjectionPolicy(
            output_sample_rate=48_000.0,
            projection_half_support_seconds=0.002,
            projection_kernel_steps=256,
        )

    def build(self, note: NoteEvent) -> WitnessAwareSynthDriver:
        """Return a fresh driver configured for *note*."""
        t        = self._timbre
        manifold = t.waveform_manifold or PhaseWarpedManifold()
        drift    = t.drift_model or NullDriftModel()

        master_path = ConstantPhasePath(frequency_hz=note.fundamental_hz)

        voices: List[LatticeVoice] = []
        for h in range(1, note.partial_count + 1):
            amp = ExponentialEnvelope(
                initial=note.velocity / h,
                tau_seconds=t.amplitude_decay_tau * (1.0 + 0.05 * h),
            )
            shape = SigmoidField(
                low=t.shape_low,
                high=t.shape_high,
                center=t.shape_center,
                width=t.shape_width,
            )
            voices.append(LatticeVoice(
                name=f"h{h}",
                harmonic_index=h,
                master_phase_path=master_path,
                waveform_manifold=manifold,
                amplitude_field=amp,
                shape_field=shape,
                drift_model=drift,
                gain=1.0,
                phase_offset=note.phase_hints.get(h, 0.0),
            ))

        lattice = HarmonicLattice(voices)
        return WitnessAwareSynthDriver(
            lattice=lattice,
            witness_thresholds=self._witness,
            density_policy=self._density,
            projection_policy=self._projection,
        )


# ============================================================
# Sequence renderer
# ============================================================

class SequenceRenderer:
    """
    Renders a full ``NoteSchedule`` into one phase-continuous
    ``AdaptiveSampleBuffer`` spanning the entire global timeline.

    Phase continuity
    ────────────────
    With ``carry_phase=True`` (default), ``PhaseHandoff.terminal_phases`` is
    called after every note and the resulting phases are injected into the
    next note's ``phase_hints``.  For a monophonic arpeggiation this means
    every harmonic partial crosses every note boundary without a phase jump.

    For polyphonic passages the samples from simultaneously active notes are
    merged by superposition: all active notes contribute their
    ``evaluate_linear`` value at each time point in the union of all notes'
    local sample grids, shifted to global time.

    With ``harmonic_lock=True`` the terminal-phase scaling
    (``PhaseHandoff.harmonic_lock_phases``) is applied when the next note has
    a different fundamental, so the two notes share a constructive phase
    relationship at the transition.
    """

    def __init__(
        self,
        builder: Optional[LatticeBuilder] = None,
        harmonic_lock: bool = False,
    ) -> None:
        self._builder       = builder or LatticeBuilder()
        self._harmonic_lock = harmonic_lock

    def render(
        self,
        schedule: NoteSchedule,
        carry_phase: bool = True,
    ) -> AdaptiveSampleBuffer:
        """
        Render every note in *schedule* and return the merged global buffer.

        Parameters
        ----------
        carry_phase:
            Propagate terminal phases from each note to the next.
        """
        if not schedule.events:
            return AdaptiveSampleBuffer()

        local_buffers: List[AdaptiveSampleBuffer] = []
        last_phases: Dict[int, float] = {}
        last_hz: float = 0.0

        for ev in schedule.events:
            if carry_phase and last_phases:
                if self._harmonic_lock and last_hz > 0.0:
                    ev.phase_hints = PhaseHandoff.harmonic_lock_phases(
                        last_phases, last_hz, ev.fundamental_hz
                    )
                else:
                    # Direct copy: every harmonic continues from where it left off.
                    ev.phase_hints = dict(last_phases)

            driver = self._builder.build(ev)
            local_buf = driver.emit_adaptive_complex(EmissionRange(0.0, ev.duration_s))
            local_buffers.append(local_buf)

            if carry_phase:
                last_phases = PhaseHandoff.terminal_phases(driver, ev.duration_s)
                last_hz     = ev.fundamental_hz

        return self._merge(schedule.events, local_buffers)

    @staticmethod
    def _merge(
        events: List[NoteEvent],
        local_buffers: List[AdaptiveSampleBuffer],
    ) -> AdaptiveSampleBuffer:
        """
        Merge time-shifted local buffers into a single sorted global buffer.

        The global time grid is the sorted union of every note's local sample
        times shifted to global coordinates.  At each global time point, all
        currently active notes contribute via ``evaluate_linear`` (linear
        interpolation between their adaptive grid points) and are summed.

        Near-coincident times from different notes' grids (within 1e-10 s) are
        deduplicated to one entry so the strict-increasing contract of
        ``AdaptiveSampleBuffer.add`` is satisfied.
        """
        # Collect (global_t) from every note's adaptive grid
        all_global_times: List[float] = []
        for ev, buf in zip(events, local_buffers):
            for sp in buf.samples:
                all_global_times.append(sp.t + ev.start_time)
        all_global_times.sort()

        # Deduplicate within 1e-10 s
        _EPS = 1e-10
        deduped: List[float] = []
        for t_g in all_global_times:
            if not deduped or t_g - deduped[-1] > _EPS:
                deduped.append(t_g)

        global_buf = AdaptiveSampleBuffer()
        for t_g in deduped:
            value = 0j
            for ev, buf in zip(events, local_buffers):
                local_t = t_g - ev.start_time
                if buf.samples and 0.0 <= local_t <= buf.samples[-1].t:
                    value += buf.evaluate_linear(local_t)
            global_buf.add(t_g, value)

        return global_buf


# ============================================================
# MIDI adapter  (future-proofed stub)
# ============================================================

class MidiAdapter:
    """
    Converts a Standard MIDI File into a ``NoteSchedule``.

    This class is complete for single-track melodic MIDI.  It honours
    ``note_on`` / ``note_off`` messages on all channels and respects
    ``set_tempo`` events.  Program changes, CCs, and SysEx are ignored.

    Requires ``mido``::

        pip install mido

    Usage::

        schedule = MidiAdapter.load("melody.mid", partial_count=6)
        renderer = SequenceRenderer(harmonic_lock=True)
        buf      = renderer.render(schedule)
    """

    _A4_HZ  = 440.0
    _A4_NUM = 69

    @classmethod
    def midi_note_to_hz(cls, note_number: int) -> float:
        """Equal-temperament conversion from MIDI note number to Hz."""
        return cls._A4_HZ * (2.0 ** ((note_number - cls._A4_NUM) / 12.0))

    @classmethod
    def load(
        cls,
        path: str,
        partial_count: int = 6,
        velocity_scale: float = 1.0 / 127.0,
    ) -> NoteSchedule:
        """
        Load *path* (a ``.mid`` file) and return a ``NoteSchedule``.

        Parameters
        ----------
        partial_count:
            Harmonic partials per note.
        velocity_scale:
            Multiplier applied to raw MIDI velocity (0–127) to produce the
            0–1 amplitude velocity stored on each ``NoteEvent``.
        """
        try:
            import mido
        except ImportError as exc:
            raise ImportError(
                "MidiAdapter requires mido — install it with:  pip install mido"
            ) from exc

        mid      = mido.MidiFile(path)
        tempo    = 500_000   # µs/beat = 120 bpm
        ticks_pb = mid.ticks_per_beat

        schedule: NoteSchedule            = NoteSchedule()
        active:   Dict[int, Tuple[float, float]] = {}  # note_num → (start_s, velocity)
        t_s = 0.0

        for msg in mido.merge_tracks(mid.tracks):
            t_s += mido.tick2second(msg.time, ticks_pb, tempo)

            if msg.type == "set_tempo":
                tempo = msg.tempo

            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = (t_s, msg.velocity * velocity_scale)

            elif msg.type in ("note_off", "note_on") and msg.note in active:
                start_s, vel = active.pop(msg.note)
                dur = t_s - start_s
                if dur > 0.0:
                    schedule.add(NoteEvent(
                        fundamental_hz=cls.midi_note_to_hz(msg.note),
                        start_time=start_s,
                        duration_s=dur,
                        velocity=vel,
                        partial_count=partial_count,
                    ))

        return schedule


# ============================================================
# Demo
# ============================================================

def example_render_sequence(output_path: str = "sequence.wav") -> None:
    """
    Render a two-bar pentatonic-minor arpeggiation and write a 16-bit WAV.

    The arpeggio runs over two octaves of A pentatonic minor at 100 bpm,
    stepping through a 8-note falling-then-rising pattern with slight
    velocity variation.  Phase continuity is enforced across every note
    boundary via PhaseHandoff.
    """
    import numpy as np
    from signal_generator_v2 import AdaptiveProjector, ProjectionPolicy

    rule = ArpeggioRule(
        root_hz=110.0,            # A2
        scale="pentatonic_minor",
        pattern=[0, 2, 4, 6, 7, 6, 4, 2,   # one bar up-down
                 1, 3, 5, 7, 8, 7, 5, 3],   # second bar (shifted start)
        rhythm_beats=[0.375, 0.375, 0.375, 0.375,
                      0.375, 0.375, 0.375, 0.375,
                      0.375, 0.375, 0.375, 0.375,
                      0.375, 0.375, 0.375, 0.375],
        bpm=100.0,
        velocity_curve=[1.0, 0.75, 0.85, 0.7,
                        0.9, 0.7,  0.8,  0.65,
                        1.0, 0.75, 0.85, 0.7,
                        0.9, 0.7,  0.8,  0.65],
        legato_fraction=0.82,
        partial_count=5,
        octave_span=2,
        repeats=1,
    )

    schedule = rule.generate()
    print(f"Schedule: {len(schedule.events)} notes, "
          f"{schedule.total_duration:.2f}s total")

    timbre = TimbrePreset(
        waveform_manifold=PhaseWarpedManifold(harmonic_warp_strength=0.18),
        drift_model=SmoothSinusoidalDriftModel(
            base_rate_hz=0.06,
            max_offset_hz=0.025,
            harmonic_scaling_power=1.0,
        ),
        amplitude_decay_tau=0.22,
        shape_low=0.05,
        shape_high=0.65,
        shape_center=0.08,
        shape_width=0.05,
    )
    projection_policy = ProjectionPolicy(
        output_sample_rate=192_000.0,
        projection_half_support_seconds=0.002,
        projection_kernel_steps=256,
    )

    builder  = LatticeBuilder(
        timbre=timbre,
        projection_policy=projection_policy,
    )
    renderer = SequenceRenderer(builder=builder, harmonic_lock=False)

    print("Rendering notes with phase continuity...")
    global_buf = renderer.render(schedule, carry_phase=True)
    print(f"Global adaptive samples: {len(global_buf)}")

    print("Projecting to uniform grid (numpy)...")
    projector = AdaptiveProjector(projection_policy)
    real_np   = projector.project_real_numpy(
        global_buf, 0.0, schedule.total_duration
    )
    print(f"Uniform output samples: {len(real_np)}")

    import scipy.io.wavfile as _wavfile
    peak  = float(abs(real_np).max()) or 1.0
    f32   = (real_np / peak).astype("float32")
    sr    = int(projection_policy.output_sample_rate)
    _wavfile.write(output_path, sr, f32)
    print(f"Written {output_path}: {len(f32)} samples @ {sr} Hz, 32-bit float mono")


if __name__ == "__main__":
    example_render_sequence()
