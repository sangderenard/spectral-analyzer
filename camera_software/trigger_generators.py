"""Camera-triggered control sources made from Spectral's existing signal fields."""
from __future__ import annotations

import math

from signal_generator_v2 import (ADSREnvelope, ConstantPhasePath, HarmonicMeasureManifold,
                                 LinearRampField, PhaseWarpedManifold, PiecewiseLinearField,
                                 PureSineManifold)

GENERATOR_KEYS = ("mod:1", "mod:2", "mod:3")
KINDS = ("sine", "warped", "harmonic", "ramp_up", "ramp_down", "triangle")


def validate_generators(specs):
    """Bound the three user-authored sources before allocating a control window."""
    if specs is None:
        return ()
    if not isinstance(specs, (list, tuple)) or len(specs) > len(GENERATOR_KEYS):
        raise ValueError("camera trigger generators need at most three source settings")
    clean = []
    seen = set()
    for item in specs:
        if not isinstance(item, dict) or set(item) != {
                "key", "enabled", "kind", "frequency_hz", "phase_deg", "tension",
                "attack_ms", "decay_ms", "sustain", "hold_ms", "release_ms",
                "repeat_count", "repeat_interval_ms", "level"}:
            raise ValueError("camera trigger generator has an incomplete control contract")
        key = item["key"]
        if key not in GENERATOR_KEYS or key in seen or type(item["enabled"]) is not bool:
            raise ValueError("camera trigger generator identity or enable flag is invalid")
        seen.add(key)
        if item["kind"] not in KINDS:
            raise ValueError("unknown camera trigger generator waveform")
        limits = {"frequency_hz": (0, 200), "phase_deg": (-360, 360), "tension": (0, 4),
                  "attack_ms": (0, 200), "decay_ms": (0, 200), "sustain": (0, 1),
                  "hold_ms": (0, 200), "release_ms": (0, 200),
                  "repeat_interval_ms": (1, 200), "level": (0, 2)}
        for name, (low, high) in limits.items():
            value = item[name]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"camera trigger generator {name} is outside [{low}, {high}]")
        repeat = item["repeat_count"]
        if type(repeat) is not int or not 1 <= repeat <= 16:
            raise ValueError("camera trigger generator repeat count is outside [1, 16]")
        total_ms = (repeat-1)*item["repeat_interval_ms"] + sum(item[k] for k in
            ("attack_ms", "decay_ms", "hold_ms", "release_ms"))
        if item["enabled"] and total_ms <= 0:
            raise ValueError("enabled camera trigger generator needs a positive envelope duration")
        if total_ms > 200:
            raise ValueError("camera trigger generator tail exceeds 200 ms")
        clean.append(dict(item))
    return tuple(clean)


def generator_tail_s(specs):
    return max((.001*((s["repeat_count"]-1)*s["repeat_interval_ms"] + sum(s[k] for k in
            ("attack_ms", "decay_ms", "hold_ms", "release_ms"))) for s in specs if s["enabled"]), default=0)


def _shape(spec, t):
    kind = spec["kind"]
    duration = max(.001, .001*(spec["attack_ms"]+spec["decay_ms"]+spec["hold_ms"]+spec["release_ms"]))
    if kind == "ramp_up":
        return LinearRampField(0, 1, 0, duration).value(t)
    if kind == "ramp_down":
        return LinearRampField(1, 0, 0, duration).value(t)
    if kind == "triangle":
        return PiecewiseLinearField([(0, 0), (duration/2, 1), (duration, 0)]).value(t)
    phase = ConstantPhasePath(spec["frequency_hz"], math.radians(spec["phase_deg"])).phase(t)
    if kind == "sine":
        wave = PureSineManifold().evaluate(phase)
    elif kind == "warped":
        wave = PhaseWarpedManifold(spec["tension"]).evaluate(phase, 1)
    else:
        wave = HarmonicMeasureManifold(max_harmonics=max(1, round(1+7*spec["tension"]/4))).evaluate(phase, 1)
    return (1+wave.real)/2


def camera_generator_signals(frame, specs):
    """Each Flash rising edge restarts every enabled source and its ADSR."""
    specs = validate_generators(specs)
    starts = [s.t0 for i, s in enumerate(frame.slices) if s.flash_active and
              (i == 0 or not frame.slices[i-1].flash_active)]
    outputs = {key: [0.0]*len(frame.slices) for key in GENERATOR_KEYS}
    for spec in specs:
        if not spec["enabled"]:
            continue
        a, d, h, r = (spec[k]*.001 for k in ("attack_ms", "decay_ms", "hold_ms", "release_ms"))
        duration = a+d+h+r
        envelope = ADSREnvelope(attack_time=a, decay_time=d, sustain_level=spec["sustain"],
                                release_time=r, note_duration=duration)
        triggers = [start+j*spec["repeat_interval_ms"]*.001 for start in starts
                    for j in range(spec["repeat_count"])]
        for i, sample in enumerate(frame.slices):
            t = sample.t0
            outputs[spec["key"]][i] = spec["level"]*sum(
                envelope.value(t-start)*_shape(spec, t-start)
                for start in triggers if start <= t < start+duration)
    return outputs
