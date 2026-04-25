"""improv_engine.py

Stochastic ornamentation engine for the rhythm sequencer.

Overview
--------
Three flourish types are independently evaluated at each rhythm step that the
caller marks as *improv-eligible* via the per-pattern step mask.  All three
types share the same eligibility gate; each also has its own probability so
they can co-exist or be dialled in/out independently.

  Grace   — One or two micro-notes placed immediately before or after the
             main note, offset by a chromatic (±1) or modal (±3) semitone
             interval.  Pitch and timing are stolen from the front/back of
             the gate; the main note's start/duration are nudged accordingly.
             Think of this as a quick lean into the note from above or below.

  Chirp   — A burst of rapid semitone-quantised pitch steps that fills a
             fraction of the note's time slot.  The main note still sounds
             for the remainder.  Chromatic or modal step sizes, configurable
             contour (up/down/bounce/random).  Think of this as a trill or
             rapid mordent.

  Echo    — A compressed replay of the most recent ``lookback_bars`` bars of
             the main schedule, squeezed into a fraction of the current note's
             time slot.  The main note sounds first; the echo fills the tail.
             Velocity falls off geometrically per replayed note.

Architecture
------------
``ImprovProgram`` owns:
  • Three probability sliders (one per type).
  • One sub-params dataclass per type (GraceParams, ChirpParams, EchoParams).
  • A per-pattern step-eligibility mask: ``improv_steps[pattern_index][step_i]``.
    This is a list of lists of bools, aligned to the same rhythm_division and
    pattern list as the main rhythm program.  Only eligible steps are considered
    for ornament generation.

``apply_improv`` is a **post-processing** pass.  It receives the completed
NoteSchedule (same reference used by apply_dynamics), the full patch context
needed to read the rhythm grid, and returns a *new list* of additional
NoteEvents to inject.  The caller is responsible for merging them back:
    schedule.events.extend(apply_improv(...))
    schedule.events.sort(key=lambda e: e.start_time)

No existing NoteEvents are mutated; the main note stays intact.  (The one
exception is that if a pre-grace note is generated the main event's
start_time and duration are trimmed to make room — but only when
``trim_main`` is True on the GraceParams, which defaults True.)

Serialisation
-------------
All dataclasses implement ``to_dict`` / ``from_dict`` for JSON round-trips.
"""
from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from graph_solver import BlockFaculty

if TYPE_CHECKING:
    from sequence_engine import NoteEvent, NoteSchedule  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRACE_MODES:  List[str] = ["chromatic", "modal", "either"]
GRACE_POSNS:  List[str] = ["pre", "post", "both"]
CHIRP_SHAPES: List[str] = ["up", "down", "bounce", "random"]
CHIRP_MODES:  List[str] = ["chromatic", "modal"]


# ---------------------------------------------------------------------------
# GraceParams
# ---------------------------------------------------------------------------

@dataclass
class GraceParams:
    """
    Controls for grace-note ornaments.

    mode            : interval class used for the grace pitch offset.
                      "chromatic" = ±1 semitone; "modal" = ±3 semitones;
                      "either"    = choose randomly between the two.
    position        : where to place the grace note.
                      "pre"  = before the main note (steals from its start);
                      "post" = after the main note (steals from its tail);
                      "both" = one pre- and one post-grace.
    duration_frac   : fraction of step_s each grace note occupies (0.01–0.5).
    trim_main       : if True the main note's start/duration is shrunk to
                      accommodate a pre-grace without overlap.
    vel_scale       : velocity of grace notes as fraction of the main note's
                      velocity (0.0–1.5; typically softer, e.g. 0.6).
    direction       : +1 = grace above main; -1 = grace below; 0 = random.
    """
    mode:          str   = "chromatic"
    position:      str   = "pre"
    duration_frac: float = 0.10
    trim_main:     bool  = True
    vel_scale:     float = 0.6
    direction:     int   = 0   # 0 = random

    def to_dict(self) -> dict:
        return {
            "mode":          self.mode,
            "position":      self.position,
            "duration_frac": self.duration_frac,
            "trim_main":     self.trim_main,
            "vel_scale":     self.vel_scale,
            "direction":     self.direction,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraceParams":
        g = cls()
        g.mode          = str(d.get("mode",          "chromatic"))
        g.position      = str(d.get("position",      "pre"))
        g.duration_frac = float(d.get("duration_frac", 0.10))
        g.trim_main     = bool(d.get("trim_main",    True))
        g.vel_scale     = float(d.get("vel_scale",   0.6))
        g.direction     = int(d.get("direction",     0))
        if g.mode     not in GRACE_MODES:  g.mode     = "chromatic"
        if g.position not in GRACE_POSNS:  g.position = "pre"
        return g


# ---------------------------------------------------------------------------
# ChirpParams
# ---------------------------------------------------------------------------

@dataclass
class ChirpParams:
    """
    Controls for chirp (rapid semitone burst) ornaments.

    steps       : number of pitch steps in the chirp (2–16).
    shape       : contour of the burst.
                  "up"     = ascending staircase;
                  "down"   = descending;
                  "bounce" = up then back down (or v.v.);
                  "random" = random walk from the main pitch.
    mode        : "chromatic" steps = 1 semitone each;
                  "modal"    steps = 3 semitones each.
    duration_frac : fraction of step_s the entire chirp occupies.
                    Remaining time goes to the main note.
    vel_scale   : velocity of chirp notes as fraction of the main note's vel.
    """
    steps:         int   = 4
    shape:         str   = "up"
    mode:          str   = "chromatic"
    duration_frac: float = 0.25
    vel_scale:     float = 0.75

    def to_dict(self) -> dict:
        return {
            "steps":         self.steps,
            "shape":         self.shape,
            "mode":          self.mode,
            "duration_frac": self.duration_frac,
            "vel_scale":     self.vel_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChirpParams":
        c = cls()
        c.steps         = max(2, min(16, int(d.get("steps",         4))))
        c.shape         = str(d.get("shape",         "up"))
        c.mode          = str(d.get("mode",          "chromatic"))
        c.duration_frac = float(d.get("duration_frac", 0.25))
        c.vel_scale     = float(d.get("vel_scale",   0.75))
        if c.shape not in CHIRP_SHAPES: c.shape = "up"
        if c.mode  not in CHIRP_MODES:  c.mode  = "chromatic"
        return c


# ---------------------------------------------------------------------------
# EchoParams
# ---------------------------------------------------------------------------

@dataclass
class EchoParams:
    """
    Controls for compressed-replay (motif echo) ornaments.

    lookback_bars : how many bars of preceding material to pull from (1–8).
                    If fewer bars have elapsed the available history is used.
    duration_frac : fraction of step_s reserved for the echo tail.
                    The main note sounds first; the echo fills the rest.
    vel_falloff   : geometric velocity decay per replayed event (0.5–0.99).
                    Each successive echo note is multiplied by this factor.
    max_notes     : maximum number of notes replayed from history (1–16).
    """
    lookback_bars: int   = 1
    duration_frac: float = 0.5
    vel_falloff:   float = 0.7
    max_notes:     int   = 4

    def to_dict(self) -> dict:
        return {
            "lookback_bars": self.lookback_bars,
            "duration_frac": self.duration_frac,
            "vel_falloff":   self.vel_falloff,
            "max_notes":     self.max_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EchoParams":
        e = cls()
        e.lookback_bars = max(1, min(8,  int(d.get("lookback_bars",   1))))
        e.duration_frac = float(d.get("duration_frac", 0.5))
        e.vel_falloff   = max(0.1, min(0.99, float(d.get("vel_falloff", 0.7))))
        e.max_notes     = max(1, min(16, int(d.get("max_notes",        4))))
        return e


# ---------------------------------------------------------------------------
# ImprovProgram
# ---------------------------------------------------------------------------

@dataclass
class ImprovProgram:
    """
    Top-level container for the improv ornament engine.

    enabled         : master enable; when False apply_improv is a no-op.
    prob_grace      : probability of a grace ornament at each eligible step.
    prob_chirp      : probability of a chirp ornament at each eligible step.
    prob_echo       : probability of an echo ornament at each eligible step.
    grace           : GraceParams sub-config.
    chirp           : ChirpParams sub-config.
    echo            : EchoParams sub-config.
    improv_steps    : per-pattern eligibility mask, aligned to rhythm_patterns.
                      improv_steps[pat_i][step_i] → bool.
                      Grows on demand in apply_improv to match the live division.
    """
    enabled:     bool        = False
    prob_grace:  float       = 0.0
    prob_chirp:  float       = 0.0
    prob_echo:   float       = 0.0
    grace:       GraceParams = field(default_factory=GraceParams)
    chirp:       ChirpParams = field(default_factory=ChirpParams)
    echo:        EchoParams  = field(default_factory=EchoParams)
    improv_steps: List[List[bool]] = field(default_factory=lambda: [[]])

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset({"enabled", "prob_grace", "prob_chirp",
                                       "prob_echo", "grace", "chirp", "echo",
                                       "improv_steps"}),
            tracked_inputs=frozenset(),
        )

    def ensure_pattern_size(self, pat_i: int, n_steps: int) -> None:
        """Grow improv_steps so [pat_i][step_i] is always accessible."""
        while len(self.improv_steps) <= pat_i:
            self.improv_steps.append([])
        while len(self.improv_steps[pat_i]) < n_steps:
            self.improv_steps[pat_i].append(False)

    def eligible(self, pat_i: int, step_i: int) -> bool:
        if pat_i < len(self.improv_steps) and step_i < len(self.improv_steps[pat_i]):
            return bool(self.improv_steps[pat_i][step_i])
        return False

    def toggle_step(self, pat_i: int, step_i: int, n_steps: int) -> None:
        self.ensure_pattern_size(pat_i, n_steps)
        self.improv_steps[pat_i][step_i] = not self.improv_steps[pat_i][step_i]

    def to_dict(self) -> dict:
        return {
            "enabled":      self.enabled,
            "prob_grace":   self.prob_grace,
            "prob_chirp":   self.prob_chirp,
            "prob_echo":    self.prob_echo,
            "grace":        self.grace.to_dict(),
            "chirp":        self.chirp.to_dict(),
            "echo":         self.echo.to_dict(),
            "improv_steps": [list(row) for row in self.improv_steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImprovProgram":
        ip             = cls()
        ip.enabled     = bool(d.get("enabled",    False))
        ip.prob_grace  = float(d.get("prob_grace", 0.0))
        ip.prob_chirp  = float(d.get("prob_chirp", 0.0))
        ip.prob_echo   = float(d.get("prob_echo",  0.0))
        if "grace" in d:
            ip.grace  = GraceParams.from_dict(d["grace"])
        if "chirp" in d:
            ip.chirp  = ChirpParams.from_dict(d["chirp"])
        if "echo" in d:
            ip.echo   = EchoParams.from_dict(d["echo"])
        ip.improv_steps = [[bool(v) for v in row]
                           for row in d.get("improv_steps", [[]])]
        if not ip.improv_steps:
            ip.improv_steps = [[]]
        return ip


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MIN_DUR: float = 1.0 / 48000.0


def _semitone_offset(mode: str, direction: int, rng: _random.Random) -> float:
    """Return a semitone multiplier for a single pitch offset step."""
    interval = 1 if mode == "chromatic" else 3
    if direction == 0:
        sign = rng.choice([-1, 1])
    else:
        sign = 1 if direction > 0 else -1
    return 2.0 ** (sign * interval / 12.0)


def _make_grace(
    ev,
    step_s:   float,
    params:   GraceParams,
    rng:      _random.Random,
) -> List:
    """
    Generate 0, 1, or 2 grace NoteEvents around *ev*.

    Returns a list of new NoteEvents (the main event may be mutated in-place
    if trim_main is True and a pre-grace is generated).
    """
    from sequence_engine import NoteEvent  # local import avoids circular dep
    grace_dur = max(_MIN_DUR, step_s * max(0.01, min(0.5, params.duration_frac)))
    result    = []

    def _one(t_start: float, hz: float, vel: float) -> "NoteEvent":
        return NoteEvent(
            fundamental_hz = hz,
            start_time     = max(0.0, t_start),
            duration_s     = grace_dur,
            velocity       = max(0.0, vel),
            partial_count  = ev.partial_count,
        )

    effective_mode = params.mode
    if effective_mode == "either":
        effective_mode = rng.choice(["chromatic", "modal"])

    mul = _semitone_offset(effective_mode, params.direction, rng)
    grace_hz = ev.fundamental_hz * mul
    main_vel = ev.velocity
    g_vel    = main_vel * max(0.0, params.vel_scale)

    posn = params.position
    if posn in ("pre", "both"):
        t_pre = ev.start_time - grace_dur
        result.append(_one(t_pre, grace_hz, g_vel))
        if params.trim_main:
            ev.start_time = max(ev.start_time, t_pre + grace_dur)
            ev.duration_s = max(_MIN_DUR, ev.duration_s - grace_dur)

    if posn in ("post", "both"):
        t_post = ev.start_time + ev.duration_s
        result.append(_one(t_post, grace_hz, g_vel))
        # post-grace sits in the silence after the note; no trimming needed

    return result


def _make_chirp(
    ev,
    step_s:   float,
    params:   ChirpParams,
    rng:      _random.Random,
) -> List:
    """
    Generate a burst of chirp NoteEvents at the start of *ev*'s slot.

    The chirp occupies ``duration_frac * step_s`` of time from ``ev.start_time``
    and is placed *before* the main note, which is pushed forward by the same
    amount (and shortened to compensate).
    """
    from sequence_engine import NoteEvent
    n_steps   = max(2, min(16, params.steps))
    chirp_dur = max(_MIN_DUR * n_steps, step_s * max(0.01, min(0.9, params.duration_frac)))
    note_dur  = chirp_dur / n_steps
    result    = []

    # Build Hz sequence
    interval  = 1 if params.mode == "chromatic" else 3
    hz_start  = ev.fundamental_hz
    hzs       = []
    if params.shape == "up":
        for k in range(n_steps):
            hzs.append(hz_start * (2.0 ** (k * interval / 12.0)))
    elif params.shape == "down":
        for k in range(n_steps):
            hzs.append(hz_start * (2.0 ** (-(k) * interval / 12.0)))
    elif params.shape == "bounce":
        half = n_steps // 2
        for k in range(half):
            hzs.append(hz_start * (2.0 ** (k * interval / 12.0)))
        for k in range(n_steps - half):
            hzs.append(hz_start * (2.0 ** ((half - k - 1) * interval / 12.0)))
    else:  # random
        cur_offset = 0
        for _ in range(n_steps):
            cur_offset += rng.choice([-1, 1]) * interval
            hzs.append(hz_start * (2.0 ** (cur_offset / 12.0)))

    t = ev.start_time
    for k, hz in enumerate(hzs):
        vel = ev.velocity * max(0.0, params.vel_scale)
        result.append(NoteEvent(
            fundamental_hz = hz,
            start_time     = t,
            duration_s     = note_dur,
            velocity       = vel,
            partial_count  = ev.partial_count,
        ))
        t += note_dur

    # Push main note forward past the chirp
    ev.start_time = t
    ev.duration_s = max(_MIN_DUR, ev.duration_s - chirp_dur)

    return result


def _make_echo(
    ev,
    history:  List,
    beat_s:   float,
    params:   EchoParams,
    rng:      _random.Random,
    beats_per_bar: float = 4.0,
) -> List:
    """
    Replay a slice of *history* compressed into the tail of *ev*'s time slot.

    *history* is a list of NoteEvents from before *ev*.  The lookback window is
    ``params.lookback_bars * beat_s * 4`` seconds.  The selected events are
    sorted, compressed into ``params.duration_frac * gate_s`` of time starting
    at ``ev.start_time + ev.duration_s``, and replayed with fading velocity.
    """
    from sequence_engine import NoteEvent
    if not history:
        return []

    bar_s        = beat_s * max(1e-6, beats_per_bar)
    window_start = ev.start_time - params.lookback_bars * bar_s
    candidates   = [h for h in history if h.start_time >= window_start]
    if not candidates:
        candidates = history[-params.max_notes:]

    # Keep at most max_notes — take the most recent
    candidates = sorted(candidates, key=lambda e: e.start_time)[-params.max_notes:]
    if not candidates:
        return []

    # Map source time span → target time window
    src_span = max(1e-9, candidates[-1].start_time - candidates[0].start_time
                   + candidates[-1].duration_s)
    tgt_dur  = max(_MIN_DUR * len(candidates),
                   ev.duration_s * max(0.0, min(1.0, params.duration_frac)))
    scale    = tgt_dur / src_span
    t_anchor = ev.start_time + ev.duration_s
    src_t0   = candidates[0].start_time

    result = []
    vel    = ev.velocity
    for cand in candidates:
        t_off  = (cand.start_time - src_t0) * scale
        d_scal = max(_MIN_DUR, cand.duration_s * scale)
        result.append(NoteEvent(
            fundamental_hz = cand.fundamental_hz,
            start_time     = t_anchor + t_off,
            duration_s     = d_scal,
            velocity       = vel,
            partial_count  = cand.partial_count,
        ))
        vel *= params.vel_falloff

    return result


# ---------------------------------------------------------------------------
# apply_improv — public entry point
# ---------------------------------------------------------------------------

def apply_improv(
    events:            List,        # NoteEvent list from NoteSchedule — NOT mutated for ordering
    program:           ImprovProgram,
    beat_s:            float,
    rhythm_division:   int,
    rhythm_patterns:   List,        # List[RhythmPattern]
    phrase:            List[int],
    cycle_bars:        int,
    rng:               Optional[_random.Random] = None,
    beats_per_bar:     float = 4.0,
) -> List:
    """
    Evaluate improv ornaments for every eligible onset in *events*.

    Returns a *new* list of NoteEvents to inject.  The caller merges them::

        extra = apply_improv(sched.events, ...)
        sched.events.extend(extra)
        sched.events.sort(key=lambda e: e.start_time)

    Parameters
    ----------
    events           : existing NoteEvents (read + optionally mutated in-place
                       for trim_main grace notes; ordering unchanged).
    program          : ImprovProgram
    beat_s           : seconds per beat
    rhythm_division  : steps per bar (mirrors RhythmPattern granularity)
    rhythm_patterns  : list of RhythmPattern objects from the patch
    phrase           : list of pattern indices for each bar
    cycle_bars       : total bars in one render cycle (for step-identity lookup)
    rng              : seeded Random; fresh unseeded one created if None
    """
    if not program.enabled or not events:
        return []

    _rng   = rng if rng is not None else _random.Random()
    div    = max(1, rhythm_division)
    bar_s  = beat_s * max(1e-6, beats_per_bar)
    step_s = bar_s / div
    extra  = []

    for ev_i, ev in enumerate(events):
        # ── Identify which pattern + step this event came from ─────────────
        # Use the onset time to reverse-map bar and step.
        bar_f   = ev.start_time / bar_s
        bar_i   = int(bar_f) % max(1, cycle_bars)
        slot    = bar_i % max(1, len(phrase))
        pat_i   = phrase[slot] if phrase else 0
        pat_i   = min(pat_i, max(0, len(rhythm_patterns) - 1))
        step_i  = int(round((ev.start_time % bar_s) / max(step_s, 1e-12))) % div

        # Ensure mask is large enough
        program.ensure_pattern_size(pat_i, div)
        if not program.eligible(pat_i, step_i):
            continue

        # History = all events before this one
        history = events[:ev_i]

        # ── Grace ──────────────────────────────────────────────────────────
        if program.prob_grace > 0.0 and _rng.random() < program.prob_grace:
            extra.extend(_make_grace(ev, step_s, program.grace, _rng))

        # ── Chirp ──────────────────────────────────────────────────────────
        if program.prob_chirp > 0.0 and _rng.random() < program.prob_chirp:
            extra.extend(_make_chirp(ev, step_s, program.chirp, _rng))

        # ── Echo ───────────────────────────────────────────────────────────
        if program.prob_echo > 0.0 and _rng.random() < program.prob_echo:
            extra.extend(_make_echo(ev, history, beat_s, program.echo, _rng,
                                    beats_per_bar=beats_per_bar))

    return extra
