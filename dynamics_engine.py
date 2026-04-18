"""dynamics_engine.py

Velocity dynamics pipeline for the rhythm sequencer.

Two layers applied together at schedule post-processing time:

  1. DynamicsCurve  — a shaped velocity envelope spanning ``scope_bars`` bars,
                      tiled/repeated to cover the full rendered phrase.
  2. AccentPattern  — per-step accent magnitudes (0.0 … 2.0), aligned to the
                      rhythm division in use; multiplied on top of the curve.

The combined velocity multiplier for each NoteEvent:
    new_vel = original_vel × curve_mult(bar_offset) × accent_level(step_index)

Usage
-----
    from dynamics_engine import DynamicsProgram, apply_dynamics
    apply_dynamics(schedule.events, patch.dynamics_program,
                   beat_s, patch.rhythm_division,
                   patch.rhythm_phrase, patch.rhythm_patterns)
"""
from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Curve shapes
# ---------------------------------------------------------------------------

CURVE_SHAPES: List[str] = [
    "flat",        # constant 1.0 — pass-through null curve
    "crescendo",   # linear ramp  quiet → loud
    "decrescendo", # linear ramp  loud → quiet
    "swell",       # arch: quiet → loud → quiet  (sin half-cycle)
    "dip",         # inverted arch: loud → quiet → loud
    "drop",        # low anticipatory build → explosive peak → decay
    "random",      # per-onset random jitter (seeded per render)
]

_SCOPE_VALUES: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
_SCOPE_LABELS: List[str]   = ["¼", "½", "1", "2", "4", "8"]

_ACCENT_STEPS: tuple = (0.0, 0.5, 1.0, 1.5, 2.0)


def _curve_at(shape: str, pos: float, intensity: float, rng: _random.Random) -> float:
    """
    Evaluate ``shape`` at normalised position *pos* ∈ [0, 1].

    Returns a raw scalar ∈ [0, 1].  *intensity* blends it with 1.0:
      output = 1 + intensity * (raw_curve * 2 - 1)   →  ∈ [0, 2]
    At intensity=0 the output is always exactly 1.0.
    """
    p = max(0.0, min(1.0, pos))
    i = max(0.0, min(1.0, intensity))
    if shape == "flat" or i == 0.0:
        return 1.0
    if shape == "crescendo":
        raw = p
    elif shape == "decrescendo":
        raw = 1.0 - p
    elif shape == "swell":
        raw = math.sin(p * math.pi)
    elif shape == "dip":
        raw = 1.0 - math.sin(p * math.pi)
    elif shape == "drop":
        # Soft anticipatory build to 40 % loudness, then explosive peak at 75 %, then fast fall
        if p < 0.75:
            raw = (p / 0.75) * 0.4
        else:
            raw = 1.0 - (p - 0.75) / 0.25
        if 0.72 <= p <= 0.78:
            raw = 1.0
    elif shape == "random":
        raw = rng.random()
    else:
        raw = 1.0
    return max(0.0, 1.0 + i * (raw * 2.0 - 1.0))


# ---------------------------------------------------------------------------
# DynamicsCurve
# ---------------------------------------------------------------------------

@dataclass
class DynamicsCurve:
    """
    A shaped velocity envelope that tiles over the phrase.

    scope_bars : how many bars one full curve cycle spans.  The curve is
                 tiled (by wrapping bar_offset modulo scope_bars) to cover
                 as many bars as the phrase contains.
    intensity  : 0.0 → no effect (always 1.0); 1.0 → full dynamic range.
    """
    shape:      str   = "flat"
    scope_bars: float = 1.0
    intensity:  float = 0.5

    def multiplier_at_bar(self, bar_offset: float, rng: _random.Random) -> float:
        """Velocity multiplier for a note whose onset is *bar_offset* bars from phrase start."""
        scope = max(1e-6, self.scope_bars)
        pos   = (bar_offset % scope) / scope
        return _curve_at(self.shape, pos, self.intensity, rng)

    def to_dict(self) -> dict:
        return {"shape": self.shape, "scope_bars": self.scope_bars,
                "intensity": self.intensity}

    @classmethod
    def from_dict(cls, d: dict) -> "DynamicsCurve":
        c = cls()
        c.shape      = str(d.get("shape", "flat"))
        c.scope_bars = float(d.get("scope_bars", 1.0))
        c.intensity  = float(d.get("intensity", 0.5))
        if c.shape not in CURVE_SHAPES:
            c.shape = "flat"
        return c


# ---------------------------------------------------------------------------
# AccentPattern
# ---------------------------------------------------------------------------

@dataclass
class AccentPattern:
    """
    Per-step accent magnitudes in [0.0, 2.0].

    Mirrors ``RhythmPattern`` but stores float magnitudes rather than bool
    on/off flags.  The grid is aligned to the same rhythm_division, so the
    accent grid overlays the step grid exactly.

    Default level for every step is 1.0 (no accent change).
    """
    name:   str  = "Dyn"
    levels: list = field(default_factory=lambda: [1.0] * 16)

    def ensure_size(self, n: int) -> None:
        while len(self.levels) < n:
            self.levels.append(1.0)

    def level_at(self, step_i: int) -> float:
        if not self.levels:
            return 1.0
        return float(self.levels[step_i % len(self.levels)])

    def cycle_level(self, step_i: int) -> None:
        """Advance step to the next discrete accent level (wraps)."""
        self.ensure_size(step_i + 1)
        cur     = self.levels[step_i]
        nearest = min(range(len(_ACCENT_STEPS)), key=lambda j: abs(_ACCENT_STEPS[j] - cur))
        self.levels[step_i] = _ACCENT_STEPS[(nearest + 1) % len(_ACCENT_STEPS)]

    def to_dict(self) -> dict:
        return {"name": self.name, "levels": list(self.levels)}

    @classmethod
    def from_dict(cls, d: dict) -> "AccentPattern":
        ap        = cls()
        ap.name   = str(d.get("name", "Dyn"))
        ap.levels = [float(x) for x in d.get("levels", [])]
        return ap


# ---------------------------------------------------------------------------
# DynamicsProgram
# ---------------------------------------------------------------------------

@dataclass
class DynamicsProgram:
    """Container: one DynamicsCurve + one AccentPattern, with an enable flag."""
    enabled: bool          = False
    curve:   DynamicsCurve = field(default_factory=DynamicsCurve)
    accent:  AccentPattern = field(default_factory=AccentPattern)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "curve":   self.curve.to_dict(),
            "accent":  self.accent.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DynamicsProgram":
        dp         = cls()
        dp.enabled = bool(d.get("enabled", False))
        if "curve" in d:
            dp.curve = DynamicsCurve.from_dict(d["curve"])
        if "accent" in d:
            dp.accent = AccentPattern.from_dict(d["accent"])
        return dp


# ---------------------------------------------------------------------------
# apply_dynamics
# ---------------------------------------------------------------------------

def apply_dynamics(
    events,
    program:          "DynamicsProgram",
    beat_s:           float,
    rhythm_division:  int,
    rng:              "_random.Random | None" = None,
) -> None:
    """
    Apply dynamics (curve × accent) to a list of NoteEvents **in place**.

    Parameters
    ----------
    events          : list[NoteEvent] — modified in place
    program         : DynamicsProgram
    beat_s          : seconds per beat (60 / bpm)
    rhythm_division : steps per bar (must match the rhythm grid)
    rng             : optional seeded Random; a fresh unseeded one is created if None
    """
    if not program.enabled or not events:
        return

    _rng  = rng if rng is not None else _random.Random()
    div   = max(1, rhythm_division)
    bar_s = beat_s * 4.0          # one bar = 4 beats
    step_s = bar_s / div

    accent = program.accent
    accent.ensure_size(div)

    for ev in events:
        t      = ev.start_time
        # Bar offset (float bars from t=0)
        bar_f  = t / bar_s
        # Which step within the bar does this onset fall on?
        step_i = int(round((t % bar_s) / max(step_s, 1e-12))) % div

        curve_mult  = program.curve.multiplier_at_bar(bar_f, _rng)
        accent_mult = accent.level_at(step_i)
        ev.velocity = max(0.0, ev.velocity * curve_mult * accent_mult)
