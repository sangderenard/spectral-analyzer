"""Torch-native composition and score tensor utilities.

This module is the symbolic counterpart to the audio graph.  It represents a
composition as padded event tensors:

    labels: Int64[B, E]
    params: Float64[B, E, P]
    mask:   Bool[B, E]

``B`` is composition/page/part batch, ``E`` is event slot, and ``P`` is a fixed
parameter axis.  The graph-facing form is a FIFO packet shaped
``[B, 1, pages, events, fields, params]``; sample-time data appears only after
score events have become release-complete parametric envelope/chirp jobs.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Sequence

import torch


class ScoreEventLabel(IntEnum):
    REST = 0
    NOTE = 1
    GRACE = 2
    CHIRP = 3
    ECHO = 4


class ScoreSection(IntEnum):
    UNKNOWN = 0
    SEQUENCE = 1
    RHYTHM = 2
    DYNAMICS = 3
    IMPROV = 4
    PIANO_ROLL = 5


class ScoreParam(IntEnum):
    START_TIME = 0
    DURATION_S = 1
    FUNDAMENTAL_HZ = 2
    VELOCITY = 3
    GATE = 4
    PAGE = 5
    PATTERN = 6
    STEP = 7
    BAR = 8
    DEGREE = 9
    SECTION = 10
    KIND = 11


EVENT_REST = int(ScoreEventLabel.REST)
EVENT_NOTE = int(ScoreEventLabel.NOTE)
EVENT_GRACE = int(ScoreEventLabel.GRACE)
EVENT_CHIRP = int(ScoreEventLabel.CHIRP)
EVENT_ECHO = int(ScoreEventLabel.ECHO)

EVENT_LABEL_NAMES: tuple[str, ...] = (
    "rest",
    "note",
    "grace",
    "chirp",
    "echo",
)

PARAM_START = int(ScoreParam.START_TIME)
PARAM_DURATION = int(ScoreParam.DURATION_S)
PARAM_HZ = int(ScoreParam.FUNDAMENTAL_HZ)
PARAM_VELOCITY = int(ScoreParam.VELOCITY)
PARAM_GATE = int(ScoreParam.GATE)
PARAM_PAGE = int(ScoreParam.PAGE)
PARAM_PATTERN = int(ScoreParam.PATTERN)
PARAM_STEP = int(ScoreParam.STEP)
PARAM_BAR = int(ScoreParam.BAR)
PARAM_DEGREE = int(ScoreParam.DEGREE)
PARAM_SECTION = int(ScoreParam.SECTION)
PARAM_KIND = int(ScoreParam.KIND)

SCORE_PACKET_SLOT_AXIS = 1
SCORE_PACKET_EVENT_FIELDS = 2
SCORE_FIELD_EVENT_TYPE = 0
SCORE_FIELD_PARAMS = 1

EVENT_PARAM_NAMES: tuple[str, ...] = (
    "start_time",
    "duration_s",
    "fundamental_hz",
    "velocity",
    "gate",
    "page",
    "pattern",
    "step",
    "bar",
    "degree",
    "section",
    "kind",
)

@dataclass
class ScoreTensor:
    """Batched symbolic score.

    ``labels`` and ``params`` are padded on the event axis.  ``mask`` marks
    valid event slots, which lets callers batch scores with different event
    counts without ragged Python lists.
    """

    labels: torch.Tensor
    params: torch.Tensor
    mask: torch.Tensor
    param_names: tuple[str, ...] = EVENT_PARAM_NAMES
    label_names: tuple[str, ...] = EVENT_LABEL_NAMES
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.labels.ndim != 2:
            raise ValueError("labels must have shape [B, E]")
        if self.params.ndim != 3:
            raise ValueError("params must have shape [B, E, P]")
        if self.mask.shape != self.labels.shape:
            raise ValueError("mask must match labels shape")
        if self.params.shape[:2] != self.labels.shape:
            raise ValueError("params batch/event axes must match labels")
        self.labels = self.labels.to(torch.long)
        self.params = self.params.to(torch.float64)
        self.mask = self.mask.to(torch.bool)

    @property
    def batch_size(self) -> int:
        return int(self.labels.shape[0])

    @property
    def max_events(self) -> int:
        return int(self.labels.shape[1])

    def to(self, device: torch.device | str) -> "ScoreTensor":
        return ScoreTensor(
            labels=self.labels.to(device),
            params=self.params.to(device),
            mask=self.mask.to(device),
            param_names=self.param_names,
            label_names=self.label_names,
            metadata=dict(self.metadata),
        )

    def slice_events(self, start: int | None = None, stop: int | None = None) -> "ScoreTensor":
        return ScoreTensor(
            labels=self.labels[:, start:stop],
            params=self.params[:, start:stop, :],
            mask=self.mask[:, start:stop],
            param_names=self.param_names,
            label_names=self.label_names,
            metadata=dict(self.metadata),
        )

    def slice_time(self, start_s: float, end_s: float) -> "ScoreTensor":
        start_t = float(start_s)
        end_t = float(end_s)
        ev_start = self.params[..., PARAM_START]
        ev_end = ev_start + self.params[..., PARAM_DURATION]
        keep = self.mask & (ev_end > start_t) & (ev_start < end_t)
        return ScoreTensor(
            labels=self.labels,
            params=self.params,
            mask=keep,
            param_names=self.param_names,
            label_names=self.label_names,
            metadata={**self.metadata, "time_window": (start_t, end_t)},
        )

    def max_end_time(self) -> torch.Tensor:
        end = self.params[..., PARAM_START] + self.params[..., PARAM_DURATION]
        return torch.where(self.mask, end, torch.zeros_like(end)).amax(dim=1)


@dataclass
class SparseScoreTensor:
    """Page-aware sparse score packet for graph/composer handoff.

    ``data`` has shape ``[B, 1, PAGES, EVENTS, 2, PARAMS]``:

    - field 0 stores the float event enum in param slot 0.
    - field 1 stores the event parameter vector.

    The singleton axis is deliberately not sample time. It is the graph packet
    slot used to carry composed symbolic events through KPN FIFO edges.
    """

    data: torch.Tensor
    mask: torch.Tensor
    param_names: tuple[str, ...] = EVENT_PARAM_NAMES
    label_names: tuple[str, ...] = EVENT_LABEL_NAMES
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 6:
            raise ValueError("data must have shape [B, 1, PAGES, EVENTS, 2, PARAMS]")
        if self.data.shape[1] != 1:
            raise ValueError("score packet slot axis must be singleton")
        if self.data.shape[4] != SCORE_PACKET_EVENT_FIELDS:
            raise ValueError("event field axis must have size 2")
        if self.mask.shape != self.data.shape[:4]:
            raise ValueError("mask must match [B, 1, PAGES, EVENTS]")
        self.data = self.data.to(torch.float64)
        self.mask = self.mask.to(torch.bool)

    @property
    def batch_size(self) -> int:
        return int(self.data.shape[0])

    @property
    def max_pages(self) -> int:
        return int(self.data.shape[2])

    @property
    def max_events(self) -> int:
        return int(self.data.shape[3])

    @property
    def max_params(self) -> int:
        return int(self.data.shape[5])

    def labels_float(self) -> torch.Tensor:
        return self.data[..., SCORE_FIELD_EVENT_TYPE, 0]

    def params(self) -> torch.Tensor:
        return self.data[..., SCORE_FIELD_PARAMS, :]

    def to(self, device: torch.device | str) -> "SparseScoreTensor":
        return SparseScoreTensor(
            data=self.data.to(device),
            mask=self.mask.to(device),
            param_names=self.param_names,
            label_names=self.label_names,
            metadata=dict(self.metadata),
        )


@dataclass
class ScoreEnvelopeJobs:
    """Packed note-event jobs spanning note-on through release tail."""

    batch_index: torch.Tensor
    page_index: torch.Tensor
    event_index: torch.Tensor
    start_time: torch.Tensor
    duration_s: torch.Tensor
    release_tail_s: torch.Tensor
    end_time: torch.Tensor
    fundamental_hz: torch.Tensor
    velocity: torch.Tensor
    sample_start: torch.Tensor
    sample_count: torch.Tensor
    sub_sample_offset_s: torch.Tensor
    mask: torch.Tensor

    @property
    def job_count(self) -> int:
        return int(self.mask.numel())


@dataclass
class PerformanceAtom:
    """One renderable note unit: timing context + live ParametricCurve callables.

    Carries everything a voice node needs to synthesize a single note.
    No signal is generated here; curves are bound by reference and evaluated
    lazily by the voice during synthesis.

    chirp_curve uses the same ParametricCurve rule/loop/blend system as
    envelope_curve — edited in parallel in the UI.  Its output is normalised
    [0, 1]; to_physical() converts to Hz deviation from fundamental_hz.
    Defaults to default_chirp() until chirp rules are authored.
    """
    onset_sample:   int
    sample_count:   int
    onset_time_s:   float
    duration_s:     float
    release_tail_s: float
    phase_offset:   complex   # exp(i * 2π * fundamental_hz * sub_sample_offset_s)
    fundamental_hz: float
    velocity:       float
    voice_key:      str
    batch_index:    int
    page_index:     int
    event_index:    int
    gate_history:   list      # list[GateEvent]
    envelope_curve: Any       # ParametricCurve
    chirp_curve:    Any       # ParametricCurve


def performance_atoms_from_jobs(
    jobs: "ScoreEnvelopeJobs",
    *,
    envelope_curves: "Sequence[Any]",
    chirp_curves: "Sequence[Any]",
    voice_key: str,
    sample_rate: float,
) -> "list[PerformanceAtom]":
    """Build one PerformanceAtom per job, binding live curve callables.

    ``envelope_curves`` and ``chirp_curves`` must each have length
    ``jobs.job_count`` (expand a single curve with ``[curve] * n`` before
    calling).  Curves are bound by reference — not copied.
    """
    from parametric_curve import GateEvent

    sr = max(1.0, float(sample_rate))
    atoms: list[PerformanceAtom] = []
    for i in range(jobs.job_count):
        dur_s   = float(jobs.duration_s[i])
        tail_s  = float(jobs.release_tail_s[i])
        total_s = dur_s + tail_s
        gate_off = min(1.0, dur_s / max(total_s, 1e-9))
        vel = float(jobs.velocity[i])
        hz  = float(jobs.fundamental_hz[i])
        sub = float(jobs.sub_sample_offset_s[i])
        atoms.append(PerformanceAtom(
            onset_sample   = int(jobs.sample_start[i]),
            sample_count   = int(jobs.sample_count[i]),
            onset_time_s   = float(jobs.start_time[i]),
            duration_s     = dur_s,
            release_tail_s = tail_s,
            phase_offset   = cmath.exp(1j * 2.0 * math.pi * hz * sub),
            fundamental_hz = hz,
            velocity       = vel,
            voice_key      = voice_key,
            batch_index    = int(jobs.batch_index[i]),
            page_index     = int(jobs.page_index[i]),
            event_index    = int(jobs.event_index[i]),
            gate_history   = [GateEvent(t_on=0.0, t_off=gate_off, velocity=vel)],
            envelope_curve = envelope_curves[i],
            chirp_curve    = chirp_curves[i],
        ))
    return atoms


@dataclass(frozen=True)
class ComposerBatchConfig:
    """Batch render configuration for symbolic-to-control conversion."""

    sample_rate: float
    n_samples: int
    beat_s: float
    beats_per_bar: float = 4.0
    batch_reduce: str = "amax"


def empty_score_tensor(
    batch_size: int = 1,
    max_events: int = 1,
    *,
    device: torch.device | str | None = None,
) -> ScoreTensor:
    shape = (max(1, int(batch_size)), max(1, int(max_events)))
    labels = torch.zeros(shape, dtype=torch.long, device=device)
    params = torch.zeros(shape + (len(EVENT_PARAM_NAMES),), dtype=torch.float64, device=device)
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    return ScoreTensor(labels=labels, params=params, mask=mask)


def empty_sparse_score_tensor(
    batch_size: int = 1,
    max_pages: int = 1,
    max_events: int = 1,
    *,
    max_params: int | None = None,
    device: torch.device | str | None = None,
) -> SparseScoreTensor:
    shape = (
        max(1, int(batch_size)),
        1,
        max(1, int(max_pages)),
        max(1, int(max_events)),
        SCORE_PACKET_EVENT_FIELDS,
        int(max_params or len(EVENT_PARAM_NAMES)),
    )
    data = torch.zeros(shape, dtype=torch.float64, device=device)
    mask = torch.zeros(shape[:4], dtype=torch.bool, device=device)
    return SparseScoreTensor(data=data, mask=mask)


def score_tensor_from_schedules(
    schedules: Sequence[Any],
    *,
    max_events: int | None = None,
    device: torch.device | str | None = None,
) -> ScoreTensor:
    """Convert ``NoteSchedule``-like objects into a padded ``ScoreTensor``."""
    batch = max(1, len(schedules))
    event_counts = [len(getattr(schedule, "events", []) or []) for schedule in schedules]
    e_max = max_events if max_events is not None else max(event_counts, default=1)
    score = empty_score_tensor(batch, max(1, int(e_max)), device=device)
    for bi, schedule in enumerate(schedules):
        events = list(getattr(schedule, "events", []) or [])[: score.max_events]
        for ei, event in enumerate(events):
            score.labels[bi, ei] = EVENT_NOTE
            score.mask[bi, ei] = True
            start = float(getattr(event, "start_time", 0.0))
            dur = float(getattr(event, "duration_s", 0.0))
            hz = float(getattr(event, "fundamental_hz", 0.0))
            vel = float(getattr(event, "velocity", 1.0))
            score.params[bi, ei, PARAM_START] = start
            score.params[bi, ei, PARAM_DURATION] = max(0.0, dur)
            score.params[bi, ei, PARAM_HZ] = hz
            score.params[bi, ei, PARAM_VELOCITY] = vel
            score.params[bi, ei, PARAM_GATE] = 1.0 if dur > 0.0 else 0.0
    return score


def sparse_score_tensor_from_score(
    score: ScoreTensor,
    *,
    max_pages: int | None = None,
) -> SparseScoreTensor:
    """Pack ``ScoreTensor`` into the page-aware sparse packet layout."""
    page_values = torch.where(
        score.mask,
        score.params[..., PARAM_PAGE].round().to(torch.long).clamp(min=0),
        torch.zeros_like(score.labels),
    )
    pages = int(max_pages or max(1, int(page_values.max().item()) + 1))
    events_per_page = torch.zeros((score.batch_size, pages), dtype=torch.long, device=score.params.device)
    for page in range(pages):
        events_per_page[:, page] = (score.mask & (page_values == page)).sum(dim=1)
    e_max = max(1, int(events_per_page.max().item()))
    packet = empty_sparse_score_tensor(
        score.batch_size,
        pages,
        e_max,
        max_params=score.params.shape[-1],
        device=score.params.device,
    )
    for bi in range(score.batch_size):
        write_pos = [0 for _ in range(pages)]
        for ei in range(score.max_events):
            if not bool(score.mask[bi, ei].item()):
                continue
            page = int(page_values[bi, ei].item())
            if page >= pages:
                continue
            dst = write_pos[page]
            if dst >= e_max:
                continue
            packet.mask[bi, 0, page, dst] = True
            packet.data[bi, 0, page, dst, SCORE_FIELD_EVENT_TYPE, 0] = float(score.labels[bi, ei].item())
            packet.data[bi, 0, page, dst, SCORE_FIELD_PARAMS, : score.params.shape[-1]] = score.params[bi, ei]
            write_pos[page] += 1
    return packet


def sparse_score_tensor_from_schedules(
    schedules: Sequence[Any],
    *,
    max_pages: int | None = None,
    max_events: int | None = None,
    device: torch.device | str | None = None,
) -> SparseScoreTensor:
    """Convert schedules directly into the sparse page/event score packet."""
    score = score_tensor_from_schedules(schedules, max_events=max_events, device=device)
    return sparse_score_tensor_from_score(score, max_pages=max_pages)


def envelope_jobs_from_sparse_score(
    packet: SparseScoreTensor,
    *,
    sample_rate: float,
    release_tail_s: float | torch.Tensor,
    include_labels: Sequence[int] = (EVENT_NOTE, EVENT_GRACE, EVENT_CHIRP),
) -> ScoreEnvelopeJobs:
    """Build release-complete envelope jobs from sparse score events."""
    dev = packet.data.device
    labels = packet.labels_float().round().to(torch.long)
    params = packet.params()
    include = torch.zeros_like(labels, dtype=torch.bool)
    for label in include_labels:
        include = include | (labels == int(label))
    valid = packet.mask & include
    loc = torch.nonzero(valid, as_tuple=False)
    if loc.numel() == 0:
        empty_f = torch.zeros(0, dtype=torch.float64, device=dev)
        empty_l = torch.zeros(0, dtype=torch.long, device=dev)
        empty_b = torch.zeros(0, dtype=torch.bool, device=dev)
        return ScoreEnvelopeJobs(
            batch_index=empty_l,
            page_index=empty_l,
            event_index=empty_l,
            start_time=empty_f,
            duration_s=empty_f,
            release_tail_s=empty_f,
            end_time=empty_f,
            fundamental_hz=empty_f,
            velocity=empty_f,
            sample_start=empty_l,
            sample_count=empty_l,
            sub_sample_offset_s=empty_f,
            mask=empty_b,
        )

    b = loc[:, 0]
    p = loc[:, 2]
    e = loc[:, 3]
    selected = params[b, loc[:, 1], p, e, :]
    start = selected[:, PARAM_START].to(torch.float64).clamp(min=0.0)
    duration = selected[:, PARAM_DURATION].to(torch.float64).clamp(min=0.0)
    hz = selected[:, PARAM_HZ].to(torch.float64).clamp(min=0.0)
    velocity = selected[:, PARAM_VELOCITY].to(torch.float64).clamp(min=0.0)
    tail = torch.as_tensor(release_tail_s, dtype=torch.float64, device=dev)
    tail = tail.expand_as(start).clone() if tail.ndim == 0 else tail.to(device=dev, dtype=torch.float64)
    tail = tail[: start.numel()].clamp(min=0.0) if tail.numel() != start.numel() else tail.clamp(min=0.0)
    end = start + duration + tail
    sr = max(1.0, float(sample_rate))
    start_sample_float = start * sr
    sample_start = torch.floor(start_sample_float).to(torch.long)
    sub_offset = start - sample_start.to(torch.float64) / sr
    sample_count = torch.ceil((end - sample_start.to(torch.float64) / sr) * sr).to(torch.long).clamp(min=1)
    return ScoreEnvelopeJobs(
        batch_index=b.to(torch.long),
        page_index=p.to(torch.long),
        event_index=e.to(torch.long),
        start_time=start,
        duration_s=duration,
        release_tail_s=tail,
        end_time=end,
        fundamental_hz=hz,
        velocity=velocity,
        sample_start=sample_start,
        sample_count=sample_count,
        sub_sample_offset_s=sub_offset,
        mask=torch.ones_like(start, dtype=torch.bool),
    )


def score_tensor_from_schedule_groups(
    play_groups: Sequence[tuple[Any, Sequence[Any]]],
    *,
    batch_mode: str = "groups",
    device: torch.device | str | None = None,
) -> ScoreTensor:
    """Convert demo play groups into event tensors.

    ``batch_mode='groups'`` keeps one batch row per schedule group/page/part.
    ``batch_mode='merged'`` treats the whole patch as one composition row.
    """
    schedules = [group[0] for group in play_groups]
    if batch_mode == "merged":
        events = []
        for schedule in schedules:
            events.extend(list(getattr(schedule, "events", []) or []))
        events.sort(key=lambda ev: float(getattr(ev, "start_time", 0.0)))
        merged = type("_ScheduleView", (), {"events": events})()
        return score_tensor_from_schedules([merged], device=device)
    return score_tensor_from_schedules(schedules, device=device)


def sparse_score_tensor_from_schedule_groups(
    play_groups: Sequence[tuple[Any, Sequence[Any]]],
    *,
    batch_mode: str = "groups",
    device: torch.device | str | None = None,
) -> SparseScoreTensor:
    """Convert play groups into the sparse page/event packet layout."""
    return sparse_score_tensor_from_score(
        score_tensor_from_schedule_groups(play_groups, batch_mode=batch_mode, device=device)
    )


def build_demo_sparse_score_from_patch(
    patch: Any,
    *,
    batch_mode: str = "merged",
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> SparseScoreTensor:
    """Build the sparse symbolic score packet used by torch graph consumers."""
    from analytic_score import (
        _build_sequence_pitch_context,
        _prepare_sequence_play_groups,
    )

    beat_s, degrees, pattern = _build_sequence_pitch_context(patch)
    old_metrics = getattr(patch, "_arrangement_metrics", None)
    old_parts = list(getattr(patch, "parts", []))
    try:
        groups = _prepare_sequence_play_groups(patch, beat_s, degrees, pattern)
    except Exception:
        groups = []
    finally:
        try:
            patch._arrangement_metrics = old_metrics if old_metrics is not None else {}
            patch.parts = old_parts
        except Exception:
            pass
    score = score_tensor_from_schedule_groups(groups, batch_mode=batch_mode, device=device)
    target_b = max(1, int(batch_size))
    if score.batch_size != target_b:
        if score.batch_size == 1:
            score = ScoreTensor(
                labels=score.labels.expand(target_b, -1).clone(),
                params=score.params.expand(target_b, -1, -1).clone(),
                mask=score.mask.expand(target_b, -1).clone(),
                param_names=score.param_names,
                label_names=score.label_names,
                metadata=dict(score.metadata),
            )
        elif score.batch_size > target_b:
            score = ScoreTensor(
                labels=score.labels[:target_b].clone(),
                params=score.params[:target_b].clone(),
                mask=score.mask[:target_b].clone(),
                param_names=score.param_names,
                label_names=score.label_names,
                metadata=dict(score.metadata),
            )
    return sparse_score_tensor_from_score(score)


def iter_score_events(score: ScoreTensor) -> Iterable[tuple[int, int, int, torch.Tensor]]:
    """Yield valid ``(batch, event, label, params)`` tuples for adapters/tests."""
    valid = torch.nonzero(score.mask, as_tuple=False)
    for bi, ei in valid.tolist():
        yield bi, ei, int(score.labels[bi, ei].item()), score.params[bi, ei]


# ─────────────────────────────────────────────────────────────────────────────
# Rhythmic grid: torch-native WarpCurve and BeatTree
# ─────────────────────────────────────────────────────────────────────────────


class WarpShape(IntEnum):
    """Rubato curve shape.  Matches ``WarpCurve.rubato_shape`` string values."""
    OFF     = 0
    SINE    = 1
    TROUGHS = 2
    SLOW_GO = 3
    GO_SLOW = 4


class WarpInterpolator(IntEnum):
    LINEAR = 0
    COSINE = 1


class ArticulationCode(IntEnum):
    """Per-leaf articulation.  Matches ``BeatNode.art`` int values."""
    NORMAL   = 0
    STACCATO = 1
    LEGATO   = 2
    DRONE    = 3


# Gate multipliers indexed by ArticulationCode.
# DRONE gets 1.0 here as a placeholder; schedule_beat_tree replaces it with
# the lookahead-to-next-onset value before the tensor is emitted.
_ART_GATE_MULT = torch.tensor([1.0, 0.5, 0.95, 1.0], dtype=torch.float64)


# ─────────────────────────────────────────────────────────────────────────────
# TorchWarpCurve
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TorchWarpCurve:
    """Batch parametric warp curve.

    Named parametric coefficients are ``[B]`` tensors.  The derived anchor
    table ``warped`` is ``[B, home_div+1]`` float64 and IS the passable
    coefficient form — callers that want a custom warp family can supply it
    directly via :meth:`from_coefficients`, bypassing the named params.

    The curve maps bar-fractions ``[0, 1]`` → ``[0, 1]`` through:

    1. **Swing** — odd-indexed anchors pushed forward by ``swing / home_div``
    2. **Pocket** — all anchors shifted by ``pocket / beats_per_bar``
    3. **Rubato** — one of five analytical shape families parameterised by
       ``rubato_amount``

    Evaluation is piecewise linear or cosine between anchors.
    """

    # ── Named parametric coefficients [B] ────────────────────────────────
    swing:         torch.Tensor   # ∈ [0, 0.67]
    pocket:        torch.Tensor   # beats offset ÷ beats_per_bar → bar-frac shift
    rubato_amount: torch.Tensor   # ∈ [0, 0.95]
    rubato_shape:  torch.Tensor   # int64, WarpShape
    meter_num:     torch.Tensor   # float64, time-signature numerator
    beats_per_bar: torch.Tensor   # float64

    home_div:     int
    interpolator: WarpInterpolator

    # ── Passable coefficient table [B, home_div+1] ────────────────────────
    warped: torch.Tensor

    # ── Factories ─────────────────────────────────────────────────────────

    @staticmethod
    def build(
        swing:         float | torch.Tensor = 0.0,
        pocket:        float | torch.Tensor = 0.0,
        rubato_amount: float | torch.Tensor = 0.0,
        rubato_shape:  WarpShape | int | torch.Tensor = WarpShape.OFF,
        meter_num:     float | torch.Tensor = 4.0,
        beats_per_bar: float | torch.Tensor = 4.0,
        home_div:      int = 16,
        interpolator:  WarpInterpolator = WarpInterpolator.LINEAR,
        batch_size:    int = 1,
        device:        torch.device | str | None = None,
    ) -> TorchWarpCurve:
        """Construct from named parameters; scalars broadcast to ``[B]``."""
        dev = torch.device(device) if device is not None else torch.device("cpu")

        def _t(x: Any, dtype: torch.dtype = torch.float64) -> torch.Tensor:
            t = torch.as_tensor(x, dtype=dtype, device=dev)
            return t.expand(batch_size).clone() if t.ndim == 0 else t

        sw  = _t(swing).clamp(0.0, 0.67)
        pk  = _t(pocket).clamp(-0.5, 0.5)
        ra  = _t(rubato_amount).clamp(0.0, 0.95)
        rs  = _t(rubato_shape, dtype=torch.long)
        mn  = _t(meter_num).clamp(0.125, 256.0)
        bpb = _t(beats_per_bar).clamp(1.0, 32.0)

        warped = TorchWarpCurve._compute_warped(sw, pk, ra, rs, mn, bpb, home_div, dev)
        return TorchWarpCurve(
            swing=sw, pocket=pk, rubato_amount=ra, rubato_shape=rs,
            meter_num=mn, beats_per_bar=bpb,
            home_div=home_div, interpolator=interpolator, warped=warped,
        )

    @staticmethod
    def from_coefficients(
        warped:       torch.Tensor,
        home_div:     int | None = None,
        interpolator: WarpInterpolator = WarpInterpolator.LINEAR,
    ) -> TorchWarpCurve:
        """Construct directly from a ``[B, D+1]`` coefficient table.

        Named parameters are zeroed-out / defaulted; only ``warped`` drives
        evaluation.  Use this when you want to pass arbitrary warp shapes
        produced by a model or editor rather than the standard named families.
        """
        B = warped.shape[0]
        D = (warped.shape[1] - 1) if home_div is None else home_div
        dev = warped.device
        z   = torch.zeros(B, dtype=torch.float64, device=dev)
        f   = torch.full((B,), 4.0, dtype=torch.float64, device=dev)
        return TorchWarpCurve(
            swing=z, pocket=z, rubato_amount=z,
            rubato_shape=torch.zeros(B, dtype=torch.long, device=dev),
            meter_num=f, beats_per_bar=f,
            home_div=D, interpolator=interpolator,
            warped=warped.to(torch.float64),
        )

    # ── Internal computation ───────────────────────────────────────────────

    @staticmethod
    def _compute_warped(
        swing:         torch.Tensor,
        pocket:        torch.Tensor,
        rubato_amount: torch.Tensor,
        rubato_shape:  torch.Tensor,
        meter_num:     torch.Tensor,
        beats_per_bar: torch.Tensor,
        home_div:      int,
        device:        torch.device,
    ) -> torch.Tensor:
        """Return warped anchor table ``[B, D+1]``."""
        B  = swing.shape[0]
        D  = home_div
        pi = math.pi

        # Straight anchors [B, D+1]
        k = torch.arange(D + 1, dtype=torch.float64, device=device)
        u = (k / D).unsqueeze(0).expand(B, -1).clone()

        # 1. Swing — odd anchors pushed by swing / D
        odd = (k.long() % 2 == 1).to(torch.float64)
        u   = u + swing.unsqueeze(1) * odd.unsqueeze(0) / D

        # 2. Pocket — uniform shift in bar-fraction space
        u = u + (pocket / beats_per_bar).unsqueeze(1)

        # 3. Rubato — analytical shape families
        amt = rubato_amount.unsqueeze(1)                         # [B, 1]
        rs  = rubato_shape.unsqueeze(1).expand(B, D + 1)        # [B, D+1]

        # Each shape: warped_u = u + delta
        #   SINE:    u + amt × sin(2πu) / (2π)
        #   TROUGHS: u + amt × sin(4πu) / (4π)
        #   SLOW_GO: (1-amt)×u + amt×u²     → delta = amt×(u²-u)
        #   GO_SLOW: (1-amt)×u + amt×(2u-u²) → delta = amt×u×(1-u)
        d_sine    = amt * torch.sin(2.0 * pi * u) / (2.0 * pi)
        d_troughs = amt * torch.sin(4.0 * pi * u) / (4.0 * pi)
        d_slow_go = amt * (u * u - u)
        d_go_slow = amt * u * (1.0 - u)

        delta = torch.zeros_like(u)
        delta = torch.where(rs == int(WarpShape.SINE),    d_sine,    delta)
        delta = torch.where(rs == int(WarpShape.TROUGHS), d_troughs, delta)
        delta = torch.where(rs == int(WarpShape.SLOW_GO), d_slow_go, delta)
        delta = torch.where(rs == int(WarpShape.GO_SLOW), d_go_slow, delta)
        u = u + delta

        # Pin endpoints, clamp interior
        u[:, 0]  = 0.0
        u[:, -1] = 1.0
        u = u.clamp(0.0, 1.0)
        return u

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def batch_size(self) -> int:
        return int(self.warped.shape[0])

    def warp(self, frac: torch.Tensor) -> torch.Tensor:
        """Map bar-fractions through the curve.

        ``frac`` : ``[N]`` or ``[B, N]`` float64 → ``[B, N]``
        """
        if frac.ndim == 1:
            frac = frac.unsqueeze(0).expand(self.batch_size, -1)
        D       = self.home_div
        t       = frac.clamp(0.0, 1.0).to(torch.float64) * D
        idx     = t.long().clamp(0, D - 1)
        t_local = t - idx.to(torch.float64)

        w0 = self.warped.gather(1, idx)
        w1 = self.warped.gather(1, (idx + 1).clamp(0, D))

        if self.interpolator == WarpInterpolator.COSINE:
            t_local = (1.0 - torch.cos(math.pi * t_local)) * 0.5
        return w0 + t_local * (w1 - w0)

    def warp_to_seconds(
        self, frac: torch.Tensor, bar_s: torch.Tensor
    ) -> torch.Tensor:
        """``frac`` : ``[B, N]``, ``bar_s`` : ``[B]`` → ``[B, N]`` seconds relative to bar start."""
        return self.warp(frac) * bar_s.unsqueeze(1)

    def derivative(self, frac: torch.Tensor) -> torch.Tensor:
        """Local time-stretch factor (>1 = expanded, <1 = compressed).

        ``frac`` : ``[B, N]`` → ``[B, N]``
        """
        if frac.ndim == 1:
            frac = frac.unsqueeze(0).expand(self.batch_size, -1)
        eps  = 0.5 / self.home_div
        f_hi = (frac + eps).clamp(0.0, 1.0)
        f_lo = (frac - eps).clamp(0.0, 1.0)
        return (self.warp(f_hi) - self.warp(f_lo)) / (f_hi - f_lo + 1e-15)

    def with_params(self, **kwargs: Any) -> TorchWarpCurve:
        """Return a new curve with updated named parameters, recomputing the anchor table."""
        current = dict(
            swing=self.swing, pocket=self.pocket,
            rubato_amount=self.rubato_amount, rubato_shape=self.rubato_shape,
            meter_num=self.meter_num, beats_per_bar=self.beats_per_bar,
            home_div=self.home_div, interpolator=self.interpolator,
        )
        current.update(kwargs)
        return TorchWarpCurve.build(
            batch_size=self.batch_size,
            device=self.warped.device,
            **current,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TorchBeatTree
# ─────────────────────────────────────────────────────────────────────────────

class _FlatLeaf:
    """Lightweight leaf stand-in for flat-mode RhythmPattern rows."""
    __slots__ = ("position", "duration", "on", "vel", "art", "group")

    def __init__(
        self,
        position: float,
        duration: float,
        on: bool,
        vel: float,
        art: int,
        group: int,
    ) -> None:
        self.position = position
        self.duration = duration
        self.on       = on
        self.vel      = vel
        self.art      = art
        self.group    = group


@dataclass
class TorchBeatTree:
    """Batch beat-tree in tensor form.

    **Leaf layer** ``[B, L_max]`` is the authoritative scheduling
    representation — it mirrors ``BeatTree.flat_leaves()`` and is all that
    :func:`schedule_beat_tree` requires.

    **Topology layer** ``node_*`` / ``edge_*`` (optional, ``None`` when
    omitted) encodes the parent-child structure as a padded COO edge list.
    It enables subdivision / collapse / merge operations in torch without
    round-tripping to Python objects, and allows the tree to be transported
    over ports (serialised, sent to a worker, etc.) as pure tensors.
    Pass ``with_topology=True`` to the factories to populate it.
    """

    # ── Leaf layer [B, L_max] ─────────────────────────────────────────────
    leaf_pos:   torch.Tensor   # float64  bar-fraction ∈ [0, 1)
    leaf_dur:   torch.Tensor   # float64  bar-fraction span > 0
    leaf_on:    torch.Tensor   # bool     fires a note
    leaf_vel:   torch.Tensor   # float64  velocity ∈ [0, 1]
    leaf_art:   torch.Tensor   # int64    ArticulationCode
    leaf_group: torch.Tensor   # int64    group id (0 = ungrouped)
    leaf_mask:  torch.Tensor   # bool     valid slot

    # ── Topology layer (optional) ─────────────────────────────────────────
    # Nodes: [B, N_max] — every tree node (interior + leaf)
    node_pos:  torch.Tensor | None   # float64 bar-fraction
    node_dur:  torch.Tensor | None   # float64 bar-fraction span
    node_mask: torch.Tensor | None   # bool    valid slot

    # Edges: [B, E_max] — COO parent→child index pairs
    edge_src:  torch.Tensor | None   # int64   parent node index in node array
    edge_dst:  torch.Tensor | None   # int64   child node index in node array
    edge_mask: torch.Tensor | None   # bool    valid edge slot

    home_div: int

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def batch_size(self) -> int:
        return int(self.leaf_pos.shape[0])

    @property
    def max_leaves(self) -> int:
        return int(self.leaf_pos.shape[1])

    # ── Factories ─────────────────────────────────────────────────────────

    @staticmethod
    def from_beat_trees(
        trees: Sequence[Any],
        home_div: int | None = None,
        with_topology: bool = False,
        device: torch.device | str | None = None,
    ) -> TorchBeatTree:
        """Pack a list of Python ``BeatTree`` objects into batch tensors."""
        dev = torch.device(device) if device is not None else torch.device("cpu")
        div = home_div or (getattr(trees[0], "home_div", 16) if trees else 16)
        B   = len(trees)

        all_leaves = [list(t.flat_leaves()) for t in trees]
        L = max((len(lv) for lv in all_leaves), default=1)

        lp, ld, lon, lv, la, lg, lm = _alloc_leaf_tensors(B, L, dev)
        _fill_leaf_tensors(all_leaves, lp, ld, lon, lv, la, lg, lm)

        topo = _build_topology(trees, dev) if with_topology else _null_topology()
        return TorchBeatTree(
            leaf_pos=lp, leaf_dur=ld, leaf_on=lon,
            leaf_vel=lv, leaf_art=la, leaf_group=lg, leaf_mask=lm,
            home_div=div, **topo,
        )

    @staticmethod
    def from_rhythm_pattern(
        patterns: Sequence[Any],
        home_div: int = 16,
        with_topology: bool = False,
        device: torch.device | str | None = None,
    ) -> TorchBeatTree:
        """Build from ``RhythmPattern`` objects; handles tree and flat modes."""
        dev = torch.device(device) if device is not None else torch.device("cpu")
        B   = len(patterns)

        all_leaves: list[list[Any]] = []
        trees_for_topo: list[Any]   = []

        for pat in patterns:
            if pat.is_tree_mode():
                tree = pat.get_tree(home_div)
                all_leaves.append(list(tree.flat_leaves()))
                trees_for_topo.append(tree)
            else:
                pat.ensure_size(home_div)
                all_leaves.append([
                    _FlatLeaf(
                        position = i / home_div,
                        duration = 1.0 / home_div,
                        on       = bool(pat.steps[i]) if i < len(pat.steps) else False,
                        vel      = float(pat.vel[i])  if i < len(pat.vel)   else 1.0,
                        art      = int(pat.art[i])    if i < len(pat.art)   else 0,
                        group    = 0,
                    )
                    for i in range(home_div)
                ])
                trees_for_topo.append(None)

        L = max((len(lv) for lv in all_leaves), default=1)
        lp, ld, lon, lv, la, lg, lm = _alloc_leaf_tensors(B, L, dev)
        _fill_leaf_tensors(all_leaves, lp, ld, lon, lv, la, lg, lm)

        topo = (
            _build_topology(trees_for_topo, dev)
            if with_topology
            else _null_topology()
        )
        return TorchBeatTree(
            leaf_pos=lp, leaf_dur=ld, leaf_on=lon,
            leaf_vel=lv, leaf_art=la, leaf_group=lg, leaf_mask=lm,
            home_div=home_div, **topo,
        )


# ── Leaf tensor helpers ────────────────────────────────────────────────────

def _alloc_leaf_tensors(
    B: int, L: int, dev: torch.device
) -> tuple[torch.Tensor, ...]:
    lp  = torch.zeros((B, L), dtype=torch.float64, device=dev)
    ld  = torch.zeros((B, L), dtype=torch.float64, device=dev)
    lon = torch.zeros((B, L), dtype=torch.bool,    device=dev)
    lv  = torch.ones( (B, L), dtype=torch.float64, device=dev)
    la  = torch.zeros((B, L), dtype=torch.long,    device=dev)
    lg  = torch.zeros((B, L), dtype=torch.long,    device=dev)
    lm  = torch.zeros((B, L), dtype=torch.bool,    device=dev)
    return lp, ld, lon, lv, la, lg, lm


def _fill_leaf_tensors(
    all_leaves: list[list[Any]],
    lp: torch.Tensor, ld: torch.Tensor, lon: torch.Tensor,
    lv: torch.Tensor, la: torch.Tensor, lg: torch.Tensor,
    lm: torch.Tensor,
) -> None:
    for b, leaves in enumerate(all_leaves):
        for i, leaf in enumerate(leaves):
            lp[b, i]  = float(leaf.position)
            ld[b, i]  = float(leaf.duration)
            lon[b, i] = bool(leaf.on)
            lv[b, i]  = float(getattr(leaf, "vel",   1.0))
            la[b, i]  = int(getattr(leaf,  "art",   0))
            lg[b, i]  = int(getattr(leaf,  "group", 0))
            lm[b, i]  = True


def _null_topology() -> dict[str, torch.Tensor | None]:
    return dict(
        node_pos=None, node_dur=None, node_mask=None,
        edge_src=None, edge_dst=None, edge_mask=None,
    )


def _build_topology(
    trees: Sequence[Any], dev: torch.device
) -> dict[str, torch.Tensor | None]:
    """Build COO topology tensors from a list of BeatTree objects (or None)."""
    all_nodes: list[list[tuple[float, float]]] = []
    all_edges: list[list[tuple[int, int]]]     = []

    for tree in trees:
        nodes: list[tuple[float, float]] = []
        edges: list[tuple[int, int]]     = []
        if tree is not None:
            _collect_nodes(getattr(tree, "nodes", []), nodes, edges, -1)
        all_nodes.append(nodes)
        all_edges.append(edges)

    B = len(trees)
    N = max((len(ns) for ns in all_nodes), default=1)
    E = max((len(es) for es in all_edges), default=0)

    node_pos  = torch.zeros((B, N), dtype=torch.float64, device=dev)
    node_dur  = torch.zeros((B, N), dtype=torch.float64, device=dev)
    node_mask = torch.zeros((B, N), dtype=torch.bool,    device=dev)

    if E > 0:
        edge_src  = torch.zeros((B, E), dtype=torch.long, device=dev)
        edge_dst  = torch.zeros((B, E), dtype=torch.long, device=dev)
        edge_mask = torch.zeros((B, E), dtype=torch.bool, device=dev)
    else:
        edge_src = edge_dst = edge_mask = None

    for b, (nodes, edges) in enumerate(zip(all_nodes, all_edges)):
        for i, (pos, dur) in enumerate(nodes):
            node_pos[b, i]  = pos
            node_dur[b, i]  = dur
            node_mask[b, i] = True
        if E > 0:
            for i, (src, dst) in enumerate(edges):
                edge_src[b, i]  = src   # type: ignore[index]
                edge_dst[b, i]  = dst   # type: ignore[index]
                edge_mask[b, i] = True  # type: ignore[index]

    return dict(
        node_pos=node_pos, node_dur=node_dur, node_mask=node_mask,
        edge_src=edge_src, edge_dst=edge_dst, edge_mask=edge_mask,
    )


def _collect_nodes(
    nodes: list[Any],
    out_nodes: list[tuple[float, float]],
    out_edges: list[tuple[int, int]],
    parent_idx: int,
) -> None:
    """DFS: collect (pos, dur) and (parent, child) index pairs."""
    for node in nodes:
        my_idx = len(out_nodes)
        out_nodes.append((float(node.position), float(node.duration)))
        if parent_idx >= 0:
            out_edges.append((parent_idx, my_idx))
        if getattr(node, "children", None):
            _collect_nodes(node.children, out_nodes, out_edges, my_idx)


# ─────────────────────────────────────────────────────────────────────────────
# schedule_beat_tree — core rhythmic scheduler
# ─────────────────────────────────────────────────────────────────────────────

def schedule_beat_tree(
    tree:      TorchBeatTree,
    warp:      TorchWarpCurve,
    bar_s:     torch.Tensor,         # [B] seconds per bar
    abs_bar:   torch.Tensor,         # [B] int64 absolute bar number (0-based)
    *,
    n_repeats: int = 1,
    max_events: int | None = None,
    device:    torch.device | str | None = None,
) -> ScoreTensor:
    """Render a :class:`TorchBeatTree` through a :class:`TorchWarpCurve`.

    This is the torch-native counterpart to the inner scheduling loop of
    ``analytic_score._build_rhythm_schedule``.  It preserves every knob:

    * **Warp-mapped onset times** — swing, pocket, rubato all applied via
      the warp curve coefficient table
    * **Group merging** — adjacent ON leaves with the same non-zero group id
      are fused into a single sustained event (matching ``iter_grouped_events``)
    * **Articulation gate multipliers** — STACCATO 0.5×, LEGATO 0.95×
    * **Drone gate lookahead** — DRONE leaves extend to the onset of the next
      ON leaf in the bar (matching the ``art_mul is None`` branch)
    * **Repeats** — ``abs_bar`` is incremented by ``rep`` for each pass so
      onset times land in successive bars

    Output is a padded :class:`ScoreTensor` ``[B, E_max, P]`` where ``E_max``
    equals ``n_on_leaves × n_repeats`` (before the optional ``max_events`` cap).
    ``PARAM_KIND`` carries the ``ArticulationCode`` for each event.
    """
    dev = torch.device(device) if device is not None else tree.leaf_pos.device
    B   = tree.batch_size
    L   = tree.max_leaves

    bar_s_f  = bar_s.to(device=dev, dtype=torch.float64)    # [B]
    abs_bar_i = abs_bar.to(device=dev, dtype=torch.long)     # [B]

    # Precompute the drone lookahead: for each leaf, next ON leaf's bar-fraction
    # (defaults to 1.0 if no ON leaf follows in the bar).
    # Backward scan: next_on_pos[b, i] = position of first ON leaf after i
    next_on_pos = torch.ones((B, L), dtype=torch.float64, device=dev)
    for j in range(L - 2, -1, -1):
        next_valid = tree.leaf_on[:, j + 1] & tree.leaf_mask[:, j + 1]
        next_on_pos[:, j] = torch.where(
            next_valid, tree.leaf_pos[:, j + 1], next_on_pos[:, j + 1]
        )

    # Art multiplier table [B, L], with drone handled separately
    art_idx    = tree.leaf_art.clamp(0, 3)
    art_mult_t = _ART_GATE_MULT.to(device=dev)[art_idx]          # [B, L]
    is_drone   = (tree.leaf_art == int(ArticulationCode.DRONE))   # [B, L] bool
    drone_gate = (next_on_pos - tree.leaf_pos).clamp(min=1.0 / 48000.0)  # bar-fracs

    # Identify group leaders — leaves that open a new event slot.
    # A leader is any ON valid leaf that is NOT a continuation of the previous
    # leaf's group run.
    prev_grp  = torch.zeros_like(tree.leaf_group)
    prev_on   = torch.zeros_like(tree.leaf_on)
    prev_mask = torch.zeros_like(tree.leaf_mask)
    prev_grp[:, 1:]  = tree.leaf_group[:, :-1]
    prev_on[:, 1:]   = tree.leaf_on[:, :-1]
    prev_mask[:, 1:] = tree.leaf_mask[:, :-1]

    ungrouped = (tree.leaf_group == 0)
    new_run   = ungrouped | (tree.leaf_group != prev_grp) | (~prev_on) | (~prev_mask)
    is_leader = tree.leaf_on & tree.leaf_mask & new_run            # [B, L]

    # Backward scan for group durations:
    # group_dur[b, i] = total bar-fraction span from leaf i to end of its group run.
    # Drone leaves keep their own raw duration; the lookahead gate overrides it.
    group_dur = tree.leaf_dur.clone()
    for j in range(L - 2, -1, -1):
        continues = (
            (tree.leaf_group[:, j] != 0)
            & tree.leaf_on[:, j]
            & tree.leaf_mask[:, j]
            & (tree.leaf_group[:, j + 1] == tree.leaf_group[:, j])
            & tree.leaf_on[:, j + 1]
            & tree.leaf_mask[:, j + 1]
        )
        group_dur[:, j] = torch.where(
            continues,
            group_dur[:, j] + group_dur[:, j + 1],
            group_dur[:, j],
        )

    # Per-leader gate bar-fraction:
    #   ungrouped + grouped leaders: group_dur × art_mult  (or drone_gate)
    #   non-leaders: zeroed (won't appear in output)
    gate_frac = torch.where(is_drone, drone_gate, group_dur * art_mult_t)
    gate_frac = gate_frac * is_leader.to(torch.float64)           # zero non-leaders

    # Warp leaf positions → warped bar-fractions [B, L]
    warped_pos = warp.warp(tree.leaf_pos)

    # Accumulate events across repeats
    all_onset: list[torch.Tensor] = []
    all_gate:  list[torch.Tensor] = []
    all_vel:   list[torch.Tensor] = []
    all_art:   list[torch.Tensor] = []
    all_mask:  list[torch.Tensor] = []

    for rep in range(n_repeats):
        bar_off = (abs_bar_i + rep).to(torch.float64)            # [B]
        onset_s = (
            bar_off.unsqueeze(1) * bar_s_f.unsqueeze(1)
            + warped_pos * bar_s_f.unsqueeze(1)
        )                                                         # [B, L]
        gate_s  = gate_frac * bar_s_f.unsqueeze(1)               # [B, L]

        all_onset.append(onset_s)
        all_gate.append(gate_s)
        all_vel.append(tree.leaf_vel * is_leader.to(torch.float64))
        all_art.append(tree.leaf_art)
        all_mask.append(is_leader)

    # Concatenate repeats along event axis
    onset_all = torch.cat(all_onset, dim=1)    # [B, L × n_repeats]
    gate_all  = torch.cat(all_gate,  dim=1)
    vel_all   = torch.cat(all_vel,   dim=1)
    art_all   = torch.cat(all_art,   dim=1)
    mask_all  = torch.cat(all_mask,  dim=1)

    E = onset_all.shape[1]
    if max_events is not None:
        E         = min(E, int(max_events))
        onset_all = onset_all[:, :E]
        gate_all  = gate_all[:, :E]
        vel_all   = vel_all[:, :E]
        art_all   = art_all[:, :E]
        mask_all  = mask_all[:, :E]

    P      = len(EVENT_PARAM_NAMES)
    labels = torch.where(
        mask_all,
        torch.full((B, E), EVENT_NOTE, dtype=torch.long, device=dev),
        torch.zeros((B, E), dtype=torch.long, device=dev),
    )
    params = torch.zeros((B, E, P), dtype=torch.float64, device=dev)
    params[..., PARAM_START]    = onset_all
    params[..., PARAM_DURATION] = gate_all.clamp(min=1.0 / 48000.0)
    params[..., PARAM_VELOCITY] = vel_all
    params[..., PARAM_KIND]     = art_all.to(torch.float64)

    return ScoreTensor(labels=labels, params=params, mask=mask_all)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics: TorchDynamicsCurve, TorchAccentPattern, TorchDynamicsProgram
# ─────────────────────────────────────────────────────────────────────────────


class DynamicsShape(IntEnum):
    """Curve shape.  Matches ``DynamicsCurve.shape`` string values."""
    FLAT        = 0
    CRESCENDO   = 1
    DECRESCENDO = 2
    SWELL       = 3
    DIP         = 4
    DROP        = 5
    RANDOM      = 6


DYNAMICS_SHAPE_NAMES: tuple[str, ...] = (
    "flat", "crescendo", "decrescendo", "swell", "dip", "drop", "random"
)


def _dynamics_shape_index(name: str) -> int:
    try:
        return DYNAMICS_SHAPE_NAMES.index(str(name))
    except ValueError:
        return int(DynamicsShape.FLAT)


def _eval_dynamics_curve(
    shape: torch.Tensor,     # int64  [*]
    pos:   torch.Tensor,     # float64 [*], ∈ [0, 1)
    intensity: torch.Tensor, # float64 [*]
) -> torch.Tensor:
    """Vectorised ``_curve_at`` logic; broadcast-compatible shapes."""
    pi = math.pi
    raw_flat    = torch.ones_like(pos)
    raw_cres    = pos
    raw_decres  = 1.0 - pos
    raw_swell   = torch.sin(pos * pi)
    raw_dip     = 1.0 - torch.sin(pos * pi)
    ramp        = (pos / 0.75) * 0.4
    decay       = 1.0 - (pos - 0.75) / 0.25
    raw_drop    = torch.where(pos < 0.75, ramp, decay)
    raw_drop    = torch.where((pos >= 0.72) & (pos <= 0.78), torch.ones_like(pos), raw_drop)
    raw_rnd     = torch.rand_like(pos)

    raw = raw_flat.clone()
    raw = torch.where(shape == int(DynamicsShape.CRESCENDO),   raw_cres,   raw)
    raw = torch.where(shape == int(DynamicsShape.DECRESCENDO), raw_decres, raw)
    raw = torch.where(shape == int(DynamicsShape.SWELL),       raw_swell,  raw)
    raw = torch.where(shape == int(DynamicsShape.DIP),         raw_dip,    raw)
    raw = torch.where(shape == int(DynamicsShape.DROP),        raw_drop,   raw)
    raw = torch.where(shape == int(DynamicsShape.RANDOM),      raw_rnd,    raw)

    mult = (1.0 + intensity * (raw * 2.0 - 1.0)).clamp(0.0, 2.0)
    return torch.where(
        (shape == int(DynamicsShape.FLAT)) | (intensity == 0.0),
        torch.ones_like(mult), mult,
    )


@dataclass
class TorchDynamicsCurve:
    """Batch shaped velocity envelope — torch counterpart to ``DynamicsCurve``.

    ``shape``      : ``[B]`` int64 (``DynamicsShape``)
    ``scope_bars`` : ``[B]`` float64 — bars per curve cycle
    ``intensity``  : ``[B]`` float64 ∈ [0, 1]
    """

    shape:      torch.Tensor   # int64   [B]
    scope_bars: torch.Tensor   # float64 [B]
    intensity:  torch.Tensor   # float64 [B]

    @property
    def batch_size(self) -> int:
        return int(self.shape.shape[0])

    @staticmethod
    def build(
        shape:      str | int | torch.Tensor = DynamicsShape.FLAT,
        scope_bars: float | torch.Tensor = 1.0,
        intensity:  float | torch.Tensor = 0.5,
        batch_size: int = 1,
        device:     torch.device | str | None = None,
    ) -> "TorchDynamicsCurve":
        dev = torch.device(device) if device is not None else torch.device("cpu")

        def _t(x: Any, dtype: torch.dtype) -> torch.Tensor:
            if isinstance(x, str):
                x = _dynamics_shape_index(x)
            t = torch.as_tensor(x, dtype=dtype, device=dev)
            return t.expand(batch_size).clone() if t.ndim == 0 else t.to(device=dev, dtype=dtype)

        return TorchDynamicsCurve(
            shape      = _t(shape,      torch.long),
            scope_bars = _t(scope_bars, torch.float64).clamp(1e-6, 1e6),
            intensity  = _t(intensity,  torch.float64).clamp(0.0, 1.0),
        )

    @staticmethod
    def from_dynamics_curves(
        curves: Sequence[Any],
        device: torch.device | str | None = None,
    ) -> "TorchDynamicsCurve":
        return TorchDynamicsCurve.build(
            shape      = [_dynamics_shape_index(getattr(c, "shape",      "flat")) for c in curves],
            scope_bars = [float(getattr(c, "scope_bars", 1.0)) for c in curves],
            intensity  = [float(getattr(c, "intensity",  0.5))  for c in curves],
            batch_size = len(curves),
            device     = device,
        )

    def multiplier_at_bar(self, bar_offset: torch.Tensor) -> torch.Tensor:
        """``bar_offset`` : ``[B]`` or ``[B, E]`` → multiplier same shape ∈ [0, 2]."""
        scope = self.scope_bars.clamp(min=1e-6)
        # Broadcast scope/intensity to match bar_offset shape
        if bar_offset.ndim == 2:
            scope = scope.unsqueeze(1)
            intensity = self.intensity.unsqueeze(1)
            shape = self.shape.unsqueeze(1).expand_as(bar_offset)
        else:
            intensity = self.intensity
            shape = self.shape
        pos = (bar_offset % scope) / scope
        return _eval_dynamics_curve(shape, pos, intensity)

    def with_params(self, **kwargs: Any) -> "TorchDynamicsCurve":
        return TorchDynamicsCurve.build(
            shape      = kwargs.get("shape",      self.shape),
            scope_bars = kwargs.get("scope_bars", self.scope_bars),
            intensity  = kwargs.get("intensity",  self.intensity),
            batch_size = self.batch_size,
            device     = self.shape.device,
        )


@dataclass
class TorchAccentPattern:
    """Batch per-step accent grid — torch counterpart to ``AccentPattern``.

    ``levels``  : ``[B, S_max]`` float64 ∈ [0, 2]
    ``n_steps`` : ``[B]`` int64
    """

    levels:  torch.Tensor   # float64 [B, S_max]
    n_steps: torch.Tensor   # int64   [B]

    @property
    def batch_size(self) -> int:
        return int(self.levels.shape[0])

    @property
    def max_steps(self) -> int:
        return int(self.levels.shape[1])

    @staticmethod
    def build(
        levels:     "Sequence[Sequence[float]] | torch.Tensor | None" = None,
        batch_size: int = 1,
        n_steps:    int = 16,
        device:     torch.device | str | None = None,
    ) -> "TorchAccentPattern":
        dev = torch.device(device) if device is not None else torch.device("cpu")
        if levels is None:
            lvl = torch.ones((batch_size, n_steps), dtype=torch.float64, device=dev)
            ns  = torch.full((batch_size,), n_steps, dtype=torch.long, device=dev)
        elif isinstance(levels, torch.Tensor):
            lvl = levels.to(device=dev, dtype=torch.float64)
            ns  = torch.full((lvl.shape[0],), lvl.shape[1], dtype=torch.long, device=dev)
        else:
            rows = [list(r) for r in levels]
            S    = max((len(r) for r in rows), default=1)
            lvl  = torch.ones((len(rows), S), dtype=torch.float64, device=dev)
            ns   = torch.zeros(len(rows), dtype=torch.long, device=dev)
            for b, row in enumerate(rows):
                for s, v in enumerate(row):
                    lvl[b, s] = float(v)
                ns[b] = max(1, len(row))
        return TorchAccentPattern(levels=lvl.clamp(0.0, 2.0), n_steps=ns)

    @staticmethod
    def from_accent_patterns(
        patterns: Sequence[Any],
        device:   torch.device | str | None = None,
    ) -> "TorchAccentPattern":
        return TorchAccentPattern.build(
            levels = [list(getattr(p, "levels", [1.0])) for p in patterns],
            device = device,
        )

    def level_at(self, step_i: torch.Tensor) -> torch.Tensor:
        """``step_i`` : ``[B]`` or ``[B, E]`` int64 → accent level, same shape, float64."""
        ns = self.n_steps.clamp(min=1)
        S  = self.max_steps
        if step_i.ndim == 2:
            ns = ns.unsqueeze(1)
            eff = (step_i % ns).clamp(0, S - 1)                   # [B, E]
            return self.levels.unsqueeze(1).expand(-1, step_i.shape[1], -1).gather(2, eff.unsqueeze(2)).squeeze(2)
        eff = (step_i % ns).clamp(0, S - 1)                       # [B]
        return self.levels.gather(1, eff.unsqueeze(1)).squeeze(1)  # [B]


@dataclass
class TorchDynamicsProgram:
    """Batch dynamics program — torch counterpart to ``DynamicsProgram``.

    Combines :class:`TorchDynamicsCurve` and :class:`TorchAccentPattern` with
    per-batch ``enabled`` flags.  The combined velocity multiplier is:

        ``curve_mult × accent_mult``  where enabled
        ``1.0``                        where disabled
    """

    enabled: torch.Tensor        # bool  [B]
    curve:   "TorchDynamicsCurve"
    accent:  "TorchAccentPattern"

    @property
    def batch_size(self) -> int:
        return int(self.enabled.shape[0])

    @staticmethod
    def build(
        batch_size: int = 1,
        device:     torch.device | str | None = None,
    ) -> "TorchDynamicsProgram":
        dev = torch.device(device) if device is not None else torch.device("cpu")
        return TorchDynamicsProgram(
            enabled = torch.zeros(batch_size, dtype=torch.bool,   device=dev),
            curve   = TorchDynamicsCurve.build(batch_size=batch_size, device=dev),
            accent  = TorchAccentPattern.build(batch_size=batch_size, device=dev),
        )

    @staticmethod
    def from_dynamics_programs(
        programs: Sequence[Any],
        device:   torch.device | str | None = None,
    ) -> "TorchDynamicsProgram":
        dev = torch.device(device) if device is not None else torch.device("cpu")

        def _fallback_curve(p: Any) -> Any:
            c = getattr(p, "curve", None)
            if c is None:
                c = type("_", (), {"shape": "flat", "scope_bars": 1.0, "intensity": 0.5})()
            return c

        def _fallback_accent(p: Any) -> Any:
            a = getattr(p, "accent", None)
            if a is None:
                a = type("_", (), {"levels": [1.0]})()
            return a

        return TorchDynamicsProgram(
            enabled = torch.tensor(
                [bool(getattr(p, "enabled", False)) for p in programs],
                dtype=torch.bool, device=dev,
            ),
            curve  = TorchDynamicsCurve.from_dynamics_curves(
                [_fallback_curve(p)  for p in programs], device=dev,
            ),
            accent = TorchAccentPattern.from_accent_patterns(
                [_fallback_accent(p) for p in programs], device=dev,
            ),
        )

    def velocity_multiplier(
        self,
        bar_offset: torch.Tensor,   # [B] or [B, E] float64 — bar position from phrase start
        step_i:     torch.Tensor,   # [B] or [B, E] int64   — step index
    ) -> torch.Tensor:
        """Combined multiplier ``curve_mult × accent_mult``; 1.0 where disabled."""
        curve_m  = self.curve.multiplier_at_bar(bar_offset)   # [B] or [B, E]
        accent_m = self.accent.level_at(step_i)               # [B] or [B, E]
        combined = (curve_m * accent_m).clamp(0.0, 4.0)
        enabled  = self.enabled
        if bar_offset.ndim == 2:
            enabled = enabled.unsqueeze(1)
        return torch.where(enabled, combined, torch.ones_like(combined))


def apply_dynamics_to_score(
    score:           ScoreTensor,
    program:         "TorchDynamicsProgram",
    bar_s:           torch.Tensor,    # [B] seconds per bar
    rhythm_division: int,
    *,
    accent_tree: "TorchBeatTree | None" = None,
) -> ScoreTensor:
    """Apply a :class:`TorchDynamicsProgram` to a :class:`ScoreTensor`.

    Mirrors ``dynamics_engine.apply_dynamics``:

    * **DynamicsCurve** — shapes the velocity envelope across bars
    * **AccentPattern** (or accent_tree leaf velocities) — per-step grid
    * Final velocity: ``original_vel × curve_mult × accent_mult``

    Returns a new :class:`ScoreTensor`; original is not modified.
    If no batch row has ``program.enabled == True`` the input is returned
    unchanged (fast path).
    """
    if not program.enabled.any():
        return score

    B, E, _ = score.params.shape
    dev     = score.params.device
    div     = max(1, int(rhythm_division))

    bar_s_c = bar_s.to(device=dev, dtype=torch.float64).clamp(min=1e-9)
    start   = score.params[..., PARAM_START]       # [B, E]

    # Bar position (continuous, counting from phrase start) per event
    bar_f   = start / bar_s_c.unsqueeze(1)         # [B, E]

    # ── Curve multiplier [B, E] ───────────────────────────────────────────
    curve_mult = program.curve.multiplier_at_bar(bar_f)   # [B, E] via broadcast

    # ── Accent multiplier [B, E] ──────────────────────────────────────────
    if accent_tree is not None:
        # Each leaf covers a bar-fraction span; find which leaf contains each onset
        bar_frac = (start % bar_s_c.unsqueeze(1)) / bar_s_c.unsqueeze(1)  # [B, E] ∈ [0,1)
        lf_pos  = accent_tree.leaf_pos.to(device=dev)    # [B, L]
        lf_dur  = accent_tree.leaf_dur.to(device=dev)    # [B, L]
        lf_vel  = accent_tree.leaf_vel.to(device=dev)    # [B, L]
        lf_mask = accent_tree.leaf_mask.to(device=dev)   # [B, L]
        L       = lf_pos.shape[1]

        # covers[b, e, l] = mask AND pos <= bar_frac < pos+dur
        bf_exp  = bar_frac.unsqueeze(2)          # [B, E, 1]
        lp_exp  = lf_pos.unsqueeze(1)            # [B, 1, L]
        ld_exp  = lf_dur.unsqueeze(1)            # [B, 1, L]
        lm_exp  = lf_mask.unsqueeze(1)           # [B, 1, L]

        covers  = lm_exp & (lp_exp <= bf_exp) & (bf_exp < lp_exp + ld_exp)  # [B, E, L]

        # Index of last covering leaf per (b, e)
        arange_l  = torch.arange(L, device=dev).reshape(1, 1, L)
        cover_idx = (covers.long() * (arange_l + 1)).amax(dim=2) - 1  # [B, E]
        cover_idx = cover_idx.clamp(min=0)

        # Gather leaf velocity
        lv_exp      = lf_vel.unsqueeze(1).expand(B, E, L)
        accent_mult = lv_exp.gather(2, cover_idx.unsqueeze(2)).squeeze(2)  # [B, E]
        accent_mult = torch.where(covers.any(dim=2), accent_mult, torch.ones_like(accent_mult))
    else:
        bar_frac   = (start % bar_s_c.unsqueeze(1)) / bar_s_c.unsqueeze(1)  # [B, E]
        step_i     = (bar_frac * div).round().long() % div                    # [B, E]
        accent_mult = program.accent.level_at(step_i)                         # [B, E]

    # ── Combine and write back ────────────────────────────────────────────
    combined   = (curve_mult * accent_mult).clamp(0.0, 4.0)
    enabled_be = program.enabled.unsqueeze(1).expand(B, E)
    multiplier = torch.where(enabled_be, combined, torch.ones_like(combined))

    new_vel    = (score.params[..., PARAM_VELOCITY] * multiplier).clamp(0.0, 1.0)
    new_vel    = torch.where(score.mask, new_vel, score.params[..., PARAM_VELOCITY])

    new_params = score.params.clone()
    new_params[..., PARAM_VELOCITY] = new_vel
    return ScoreTensor(
        labels=score.labels, params=new_params, mask=score.mask,
        param_names=score.param_names, label_names=score.label_names,
        metadata=dict(score.metadata),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sequence: TorchNoteStream
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TorchNoteStream:
    """Batch stateful note-progression generator — torch counterpart to ``NoteStream``.

    State tensors ``idx`` and ``direction`` are mutable and advance on each
    :meth:`next_hz` call, exactly mirroring ``NoteStream._idx`` / ``._direction``.

    ``degrees``     : ``[B, N_degs]`` float64 — Hz table, padded with last value
    ``deg_pattern`` : ``[B, N_pat]``  int64   — degree indices into ``degrees``
    ``n_degs``      : ``[B]``         int64   — actual degree count per row
    ``n_pat``       : ``[B]``         int64   — actual pattern length per row

    Probabilities (``[B]`` float64 ∈ [0, 1]):
    ``p_double_back``, ``p_subversion``, ``p_chromatic``, ``p_modal``

    Mutable state (``[B]`` int64):
    ``idx``, ``direction`` (+1 = forward, -1 = backward)
    """

    degrees:       torch.Tensor   # float64 [B, N_degs]
    deg_pattern:   torch.Tensor   # int64   [B, N_pat]
    n_degs:        torch.Tensor   # int64   [B]
    n_pat:         torch.Tensor   # int64   [B]

    p_double_back: torch.Tensor   # float64 [B]
    p_subversion:  torch.Tensor   # float64 [B]
    p_chromatic:   torch.Tensor   # float64 [B]
    p_modal:       torch.Tensor   # float64 [B]

    # Mutable traversal state
    idx:       torch.Tensor       # int64 [B]
    direction: torch.Tensor       # int64 [B], +1 or -1

    @property
    def batch_size(self) -> int:
        return int(self.idx.shape[0])

    # ── Factories ─────────────────────────────────────────────────────────

    @staticmethod
    def build(
        degrees:       "Sequence[Sequence[float]] | torch.Tensor",
        deg_pattern:   "Sequence[Sequence[int]]   | torch.Tensor",
        p_double_back: "float | Sequence[float] | torch.Tensor" = 0.0,
        p_subversion:  "float | Sequence[float] | torch.Tensor" = 0.0,
        p_chromatic:   "float | Sequence[float] | torch.Tensor" = 0.0,
        p_modal:       "float | Sequence[float] | torch.Tensor" = 0.0,
        device:        torch.device | str | None = None,
    ) -> "TorchNoteStream":
        dev = torch.device(device) if device is not None else torch.device("cpu")

        # ── degrees ───────────────────────────────────────────────────────
        if isinstance(degrees, torch.Tensor):
            deg_t    = degrees.to(device=dev, dtype=torch.float64)
            n_degs_t = torch.full((deg_t.shape[0],), deg_t.shape[1], dtype=torch.long, device=dev)
        else:
            rows_d   = [list(r) for r in degrees]
            N        = max((len(r) for r in rows_d), default=1)
            deg_t    = torch.zeros((len(rows_d), N), dtype=torch.float64, device=dev)
            n_degs_t = torch.zeros(len(rows_d), dtype=torch.long, device=dev)
            for b, row in enumerate(rows_d):
                for i, v in enumerate(row):
                    deg_t[b, i] = float(v)
                if row:
                    deg_t[b, len(row):] = float(row[-1])   # pad with last valid Hz
                n_degs_t[b] = max(1, len(row))

        # ── deg_pattern ───────────────────────────────────────────────────
        if isinstance(deg_pattern, torch.Tensor):
            pat_t   = deg_pattern.to(device=dev, dtype=torch.long)
            n_pat_t = torch.full((pat_t.shape[0],), pat_t.shape[1], dtype=torch.long, device=dev)
        else:
            rows_p  = [list(r) for r in deg_pattern]
            M       = max((len(r) for r in rows_p), default=1)
            pat_t   = torch.zeros((len(rows_p), M), dtype=torch.long, device=dev)
            n_pat_t = torch.zeros(len(rows_p), dtype=torch.long, device=dev)
            for b, row in enumerate(rows_p):
                for i, v in enumerate(row):
                    pat_t[b, i] = int(v)
                n_pat_t[b] = max(1, len(row))

        B = deg_t.shape[0]

        def _p(x: Any) -> torch.Tensor:
            t = torch.as_tensor(x, dtype=torch.float64, device=dev)
            return t.expand(B).clone() if t.ndim == 0 else t.to(device=dev, dtype=torch.float64)

        return TorchNoteStream(
            degrees       = deg_t,
            deg_pattern   = pat_t,
            n_degs        = n_degs_t,
            n_pat         = n_pat_t,
            p_double_back = _p(p_double_back).clamp(0.0, 1.0),
            p_subversion  = _p(p_subversion).clamp(0.0, 1.0),
            p_chromatic   = _p(p_chromatic).clamp(0.0, 1.0),
            p_modal       = _p(p_modal).clamp(0.0, 1.0),
            idx       = torch.zeros(B, dtype=torch.long, device=dev),
            direction = torch.ones( B, dtype=torch.long, device=dev),
        )

    @staticmethod
    def from_note_streams(
        streams: Sequence[Any],
        device:  torch.device | str | None = None,
    ) -> "TorchNoteStream":
        """Pack Python ``NoteStream`` objects into a batched ``TorchNoteStream``."""
        degrees     = [list(getattr(s, "_degrees",     [440.0])) for s in streams]
        deg_pattern = [list(getattr(s, "_deg_pattern", [0]))     for s in streams]
        probs       = [getattr(s, "_probs", None)                for s in streams]

        def _prob(p: Any, attr: str) -> float:
            return float(getattr(p, attr, 0.0)) if p is not None else 0.0

        return TorchNoteStream.build(
            degrees       = degrees,
            deg_pattern   = deg_pattern,
            p_double_back = [_prob(p, "double_back") for p in probs],
            p_subversion  = [_prob(p, "subversion")  for p in probs],
            p_chromatic   = [_prob(p, "chromatic")   for p in probs],
            p_modal       = [_prob(p, "modal")        for p in probs],
            device        = device,
        )

    # ── State management ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all batch streams to start of pattern, forward direction."""
        self.idx.fill_(0)
        self.direction.fill_(1)

    def exhausted(self) -> torch.Tensor:
        """``[B]`` bool — True where the forward traversal has consumed all notes."""
        return (self.direction == 1) & (self.idx >= self.n_pat)

    # ── Iteration ─────────────────────────────────────────────────────────

    def next_hz(self) -> torch.Tensor:
        """Advance all batch streams and return ``[B]`` Hz values.

        Exhausted streams return 0.0.  Callers should check :meth:`exhausted`
        before calling and mask those rows to rests in the score.

        Stochastic transforms use ``torch.rand``; seed with
        ``torch.manual_seed`` before calling for reproducible output.
        """
        dev = self.idx.device
        B   = self.batch_size
        ex  = self.exhausted()

        # ── Double back ──────────────────────────────────────────────────
        do_flip       = (self.p_double_back > 0.0) & (torch.rand(B, device=dev) < self.p_double_back)
        self.direction = torch.where(do_flip, -self.direction, self.direction)
        # Can't go below 0: if reversed at idx == 0, snap forward
        snap           = (self.direction == -1) & (self.idx <= 0)
        self.direction = torch.where(snap, torch.ones_like(self.direction), self.direction)

        # ── Effective index (with subversion mirror) ─────────────────────
        n_pat_c = self.n_pat.clamp(min=1)
        eff_idx = self.idx % n_pat_c                                        # [B]
        do_sub  = (self.p_subversion > 0.0) & (torch.rand(B, device=dev) < self.p_subversion)
        mirror  = (n_pat_c - 1 - eff_idx).clamp(min=0)
        eff_idx = torch.where(do_sub, mirror, eff_idx)

        # Degree lookup → Hz
        deg_idx = self.deg_pattern.gather(1, eff_idx.unsqueeze(1)).squeeze(1)   # [B]
        deg_idx = deg_idx % self.n_degs.clamp(min=1)
        hz      = self.degrees.gather(1, deg_idx.unsqueeze(1)).squeeze(1)       # [B]

        # ── Chromatic: ±1 semitone ───────────────────────────────────────
        do_chrom  = (self.p_chromatic > 0.0) & (torch.rand(B, device=dev) < self.p_chromatic)
        chrom_dir = torch.where(
            torch.rand(B, device=dev) < 0.5,
            torch.full((B,), -1.0, dtype=torch.float64, device=dev),
            torch.full((B,),  1.0, dtype=torch.float64, device=dev),
        )
        hz = torch.where(do_chrom, hz * (2.0 ** (chrom_dir / 12.0)), hz)

        # ── Modal: ±3 semitones ──────────────────────────────────────────
        do_modal  = (self.p_modal > 0.0) & (torch.rand(B, device=dev) < self.p_modal)
        modal_dir = torch.where(
            torch.rand(B, device=dev) < 0.5,
            torch.full((B,), -3.0, dtype=torch.float64, device=dev),
            torch.full((B,),  3.0, dtype=torch.float64, device=dev),
        )
        hz = torch.where(do_modal, hz * (2.0 ** (modal_dir / 12.0)), hz)

        # ── Advance index ────────────────────────────────────────────────
        self.idx = self.idx + self.direction
        # Bounce off the bottom (matches Python: idx < 0 → idx=0, direction=+1)
        at_bottom      = self.idx < 0
        self.idx       = torch.where(at_bottom, torch.zeros_like(self.idx),     self.idx)
        self.direction = torch.where(at_bottom, torch.ones_like(self.direction), self.direction)

        # Exhausted streams yield 0.0 (treat as rest)
        return torch.where(ex, torch.zeros_like(hz), hz)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence ↔ Rhythm bridge: assign_hz_to_score
# ─────────────────────────────────────────────────────────────────────────────


def assign_hz_to_score(
    score:  ScoreTensor,
    stream: TorchNoteStream,
) -> ScoreTensor:
    """Advance TorchNoteStream once per valid event and write PARAM_HZ.

    Mirrors the ``stream.next_hz()`` call per ON leaf in
    ``analytic_score._build_rhythm_schedule``.  Each batch row advances
    independently: rows where the current event slot is masked do NOT consume
    a stream step — matching the Python engine's behaviour where the stream is
    only advanced for active leaves.

    The stream's ``idx`` / ``direction`` state is mutated in-place, just as the
    Python ``NoteStream`` is.  Reset it with ``stream.reset()`` before calling
    if you need a clean starting point.
    """
    B, E = score.labels.shape
    dev  = score.params.device
    hz_buf = torch.zeros((B, E), dtype=torch.float64, device=dev)

    for e in range(E):
        active  = score.mask[:, e]              # [B] bool — rows that fire at slot e
        old_idx = stream.idx.clone()
        old_dir = stream.direction.clone()
        hz_e    = stream.next_hz()              # [B] — always advances all rows
        # Revert state for inactive rows so they don't consume a stream step
        stream.idx       = torch.where(active, stream.idx,       old_idx)
        stream.direction = torch.where(active, stream.direction,  old_dir)
        hz_buf[:, e]     = hz_e

    new_params = score.params.clone()
    new_params[..., PARAM_HZ] = torch.where(score.mask, hz_buf, new_params[..., PARAM_HZ])
    return ScoreTensor(
        labels=score.labels, params=new_params, mask=score.mask,
        param_names=score.param_names, label_names=score.label_names,
        metadata=dict(score.metadata),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Improv: enums, param dataclasses, TorchImprovProgram, apply_improv_to_score
# ─────────────────────────────────────────────────────────────────────────────


class GraceMode(IntEnum):
    CHROMATIC = 0
    MODAL     = 1
    EITHER    = 2


class GracePosition(IntEnum):
    PRE  = 0
    POST = 1
    BOTH = 2


class ChirpShape(IntEnum):
    UP     = 0
    DOWN   = 1
    BOUNCE = 2
    RANDOM = 3


class ChirpMode(IntEnum):
    CHROMATIC = 0
    MODAL     = 1


GRACE_MODE_NAMES:  tuple[str, ...] = ("chromatic", "modal", "either")
GRACE_POSN_NAMES:  tuple[str, ...] = ("pre", "post", "both")
CHIRP_SHAPE_NAMES: tuple[str, ...] = ("up", "down", "bounce", "random")
CHIRP_MODE_NAMES:  tuple[str, ...] = ("chromatic", "modal")


@dataclass
class TorchGraceParams:
    """Batch grace-note parameters — torch counterpart to ``GraceParams``."""

    mode:          torch.Tensor   # int64   [B]  GraceMode
    position:      torch.Tensor   # int64   [B]  GracePosition
    duration_frac: torch.Tensor   # float64 [B]  ∈ [0.01, 0.5]
    trim_main:     torch.Tensor   # bool    [B]
    vel_scale:     torch.Tensor   # float64 [B]  ∈ [0, 1.5]
    direction:     torch.Tensor   # int64   [B]  -1 / 0 / +1

    @property
    def batch_size(self) -> int:
        return int(self.mode.shape[0])

    @staticmethod
    def build(
        mode:          "str | int | torch.Tensor" = GraceMode.CHROMATIC,
        position:      "str | int | torch.Tensor" = GracePosition.PRE,
        duration_frac: "float | torch.Tensor" = 0.10,
        trim_main:     "bool | torch.Tensor"  = True,
        vel_scale:     "float | torch.Tensor" = 0.6,
        direction:     "int | torch.Tensor"   = 0,
        batch_size:    int = 1,
        device:        "torch.device | str | None" = None,
    ) -> "TorchGraceParams":
        dev = torch.device(device) if device is not None else torch.device("cpu")

        def _i(x: Any, names: "tuple[str,...]") -> torch.Tensor:
            if isinstance(x, str):
                try:
                    x = names.index(x)
                except ValueError:
                    x = 0
            t = torch.as_tensor(x, dtype=torch.long, device=dev)
            return t.expand(batch_size).clone() if t.ndim == 0 else t.to(device=dev, dtype=torch.long)

        def _f(x: Any) -> torch.Tensor:
            t = torch.as_tensor(x, dtype=torch.float64, device=dev)
            return t.expand(batch_size).clone() if t.ndim == 0 else t.to(device=dev, dtype=torch.float64)

        def _b(x: Any) -> torch.Tensor:
            t = torch.as_tensor(x, dtype=torch.bool, device=dev)
            return t.expand(batch_size).clone() if t.ndim == 0 else t.to(device=dev, dtype=torch.bool)

        return TorchGraceParams(
            mode          = _i(mode,     GRACE_MODE_NAMES),
            position      = _i(position, GRACE_POSN_NAMES),
            duration_frac = _f(duration_frac).clamp(0.01, 0.5),
            trim_main     = _b(trim_main),
            vel_scale     = _f(vel_scale).clamp(0.0, 1.5),
            direction     = _i(direction, ()),
        )

    @staticmethod
    def from_grace_params(
        params: "Sequence[Any]",
        device: "torch.device | str | None" = None,
    ) -> "TorchGraceParams":
        def _mode(p: Any) -> int:
            try:
                return GRACE_MODE_NAMES.index(str(getattr(p, "mode", "chromatic")))
            except ValueError:
                return 0

        def _posn(p: Any) -> int:
            try:
                return GRACE_POSN_NAMES.index(str(getattr(p, "position", "pre")))
            except ValueError:
                return 0

        return TorchGraceParams.build(
            mode          = [_mode(p)                                       for p in params],
            position      = [_posn(p)                                       for p in params],
            duration_frac = [float(getattr(p, "duration_frac", 0.10))      for p in params],
            trim_main     = [bool(getattr(p,  "trim_main",     True))       for p in params],
            vel_scale     = [float(getattr(p, "vel_scale",     0.6))        for p in params],
            direction     = [int(getattr(p,   "direction",     0))          for p in params],
            batch_size    = len(params),
            device        = device,
        )


@dataclass
class TorchChirpParams:
    """Batch chirp parameters — torch counterpart to ``ChirpParams``."""

    steps:         torch.Tensor   # int64   [B]  ∈ [2, 16]
    shape:         torch.Tensor   # int64   [B]  ChirpShape
    mode:          torch.Tensor   # int64   [B]  ChirpMode
    duration_frac: torch.Tensor   # float64 [B]  ∈ [0.01, 0.9]
    vel_scale:     torch.Tensor   # float64 [B]  ∈ [0, 1.5]

    @property
    def batch_size(self) -> int:
        return int(self.steps.shape[0])

    @staticmethod
    def build(
        steps:         "int | torch.Tensor"        = 4,
        shape:         "str | int | torch.Tensor"  = ChirpShape.UP,
        mode:          "str | int | torch.Tensor"  = ChirpMode.CHROMATIC,
        duration_frac: "float | torch.Tensor"      = 0.25,
        vel_scale:     "float | torch.Tensor"      = 0.75,
        batch_size:    int = 1,
        device:        "torch.device | str | None" = None,
    ) -> "TorchChirpParams":
        dev = torch.device(device) if device is not None else torch.device("cpu")

        def _i(x: Any, names: "tuple[str,...]") -> torch.Tensor:
            if isinstance(x, str):
                try:
                    x = names.index(x)
                except ValueError:
                    x = 0
            t = torch.as_tensor(x, dtype=torch.long, device=dev)
            return t.expand(batch_size).clone() if t.ndim == 0 else t.to(device=dev, dtype=torch.long)

        def _f(x: Any) -> torch.Tensor:
            t = torch.as_tensor(x, dtype=torch.float64, device=dev)
            return t.expand(batch_size).clone() if t.ndim == 0 else t.to(device=dev, dtype=torch.float64)

        return TorchChirpParams(
            steps         = _i(steps, ()).clamp(2, 16),
            shape         = _i(shape, CHIRP_SHAPE_NAMES),
            mode          = _i(mode,  CHIRP_MODE_NAMES),
            duration_frac = _f(duration_frac).clamp(0.01, 0.9),
            vel_scale     = _f(vel_scale).clamp(0.0, 1.5),
        )

    @staticmethod
    def from_chirp_params(
        params: "Sequence[Any]",
        device: "torch.device | str | None" = None,
    ) -> "TorchChirpParams":
        def _shape(p: Any) -> int:
            try:
                return CHIRP_SHAPE_NAMES.index(str(getattr(p, "shape", "up")))
            except ValueError:
                return 0

        def _mode(p: Any) -> int:
            try:
                return CHIRP_MODE_NAMES.index(str(getattr(p, "mode", "chromatic")))
            except ValueError:
                return 0

        return TorchChirpParams.build(
            steps         = [max(2, min(16, int(getattr(p, "steps",         4)))) for p in params],
            shape         = [_shape(p)                                             for p in params],
            mode          = [_mode(p)                                              for p in params],
            duration_frac = [float(getattr(p, "duration_frac", 0.25))             for p in params],
            vel_scale     = [float(getattr(p, "vel_scale",     0.75))             for p in params],
            batch_size    = len(params),
            device        = device,
        )


@dataclass
class TorchEchoParams:
    """Batch echo parameters — torch counterpart to ``EchoParams``."""

    lookback_bars: torch.Tensor   # int64   [B]  ∈ [1, 8]
    duration_frac: torch.Tensor   # float64 [B]  ∈ [0, 1]
    vel_falloff:   torch.Tensor   # float64 [B]  ∈ [0.1, 0.99]
    max_notes:     torch.Tensor   # int64   [B]  ∈ [1, 16]

    @property
    def batch_size(self) -> int:
        return int(self.lookback_bars.shape[0])

    @staticmethod
    def build(
        lookback_bars: "int | torch.Tensor"        = 1,
        duration_frac: "float | torch.Tensor"      = 0.5,
        vel_falloff:   "float | torch.Tensor"      = 0.7,
        max_notes:     "int | torch.Tensor"        = 4,
        batch_size:    int = 1,
        device:        "torch.device | str | None" = None,
    ) -> "TorchEchoParams":
        dev = torch.device(device) if device is not None else torch.device("cpu")

        def _i(x: Any) -> torch.Tensor:
            t = torch.as_tensor(x, dtype=torch.long, device=dev)
            return t.expand(batch_size).clone() if t.ndim == 0 else t.to(device=dev, dtype=torch.long)

        def _f(x: Any) -> torch.Tensor:
            t = torch.as_tensor(x, dtype=torch.float64, device=dev)
            return t.expand(batch_size).clone() if t.ndim == 0 else t.to(device=dev, dtype=torch.float64)

        return TorchEchoParams(
            lookback_bars = _i(lookback_bars).clamp(1, 8),
            duration_frac = _f(duration_frac).clamp(0.0, 1.0),
            vel_falloff   = _f(vel_falloff).clamp(0.1, 0.99),
            max_notes     = _i(max_notes).clamp(1, 16),
        )

    @staticmethod
    def from_echo_params(
        params: "Sequence[Any]",
        device: "torch.device | str | None" = None,
    ) -> "TorchEchoParams":
        return TorchEchoParams.build(
            lookback_bars = [max(1, min(8,   int(getattr(p, "lookback_bars", 1))))   for p in params],
            duration_frac = [float(getattr(p, "duration_frac", 0.5))                 for p in params],
            vel_falloff   = [max(0.1, min(0.99, float(getattr(p, "vel_falloff", 0.7)))) for p in params],
            max_notes     = [max(1, min(16,  int(getattr(p, "max_notes",       4))))  for p in params],
            batch_size    = len(params),
            device        = device,
        )


@dataclass
class TorchImprovProgram:
    """Batch improv program — torch counterpart to ``ImprovProgram``.

    ``step_mask`` : ``[B, P_max, S_max]`` bool — per-pattern step eligibility,
                    aligned to ``rhythm_patterns`` and ``rhythm_division``.
    ``n_pats``    : ``[B]`` int64 — actual pattern count per batch row.
    ``n_steps``   : ``[B]`` int64 — rhythm_division (steps per bar).
    """

    enabled:    torch.Tensor      # bool    [B]
    prob_grace: torch.Tensor      # float64 [B]  ∈ [0, 1]
    prob_chirp: torch.Tensor      # float64 [B]  ∈ [0, 1]
    prob_echo:  torch.Tensor      # float64 [B]  ∈ [0, 1]
    grace:      TorchGraceParams
    chirp:      TorchChirpParams
    echo:       TorchEchoParams
    step_mask:  torch.Tensor      # bool    [B, P_max, S_max]
    n_pats:     torch.Tensor      # int64   [B]
    n_steps:    torch.Tensor      # int64   [B]

    @property
    def batch_size(self) -> int:
        return int(self.enabled.shape[0])

    @staticmethod
    def build(
        batch_size: int = 1,
        device:     "torch.device | str | None" = None,
    ) -> "TorchImprovProgram":
        dev = torch.device(device) if device is not None else torch.device("cpu")
        return TorchImprovProgram(
            enabled    = torch.zeros(batch_size, dtype=torch.bool,    device=dev),
            prob_grace = torch.zeros(batch_size, dtype=torch.float64, device=dev),
            prob_chirp = torch.zeros(batch_size, dtype=torch.float64, device=dev),
            prob_echo  = torch.zeros(batch_size, dtype=torch.float64, device=dev),
            grace      = TorchGraceParams.build(batch_size=batch_size, device=dev),
            chirp      = TorchChirpParams.build(batch_size=batch_size, device=dev),
            echo       = TorchEchoParams.build( batch_size=batch_size, device=dev),
            step_mask  = torch.zeros((batch_size, 1, 1), dtype=torch.bool, device=dev),
            n_pats     = torch.ones( batch_size,          dtype=torch.long, device=dev),
            n_steps    = torch.full((batch_size,), 16,    dtype=torch.long, device=dev),
        )

    @staticmethod
    def from_improv_programs(
        programs: "Sequence[Any]",
        device:   "torch.device | str | None" = None,
    ) -> "TorchImprovProgram":
        dev = torch.device(device) if device is not None else torch.device("cpu")
        B   = len(programs)

        # Build step_mask [B, P_max, S_max] from improv_steps[pat_i][step_i]
        all_masks: list[list[list[bool]]] = []
        for prog in programs:
            raw = getattr(prog, "improv_steps", [[]])
            if not raw:
                raw = [[]]
            all_masks.append([[bool(v) for v in row] for row in raw])

        P = max((len(m) for m in all_masks), default=1)
        S = max((max((len(row) for row in m), default=1) for m in all_masks), default=1)
        step_mask = torch.zeros((B, P, S), dtype=torch.bool, device=dev)
        n_pats    = torch.zeros(B, dtype=torch.long, device=dev)
        n_steps   = torch.zeros(B, dtype=torch.long, device=dev)
        for b, m in enumerate(all_masks):
            n_pats[b] = max(1, len(m))
            for pi, row in enumerate(m):
                n_steps[b] = max(n_steps[b].item(), len(row))
                for si, v in enumerate(row):
                    if pi < P and si < S:
                        step_mask[b, pi, si] = bool(v)
        n_steps = n_steps.clamp(min=1)

        def _grace(prog: Any) -> Any:
            g = getattr(prog, "grace", None)
            return g if g is not None else type("_", (), {
                "mode": "chromatic", "position": "pre", "duration_frac": 0.10,
                "trim_main": True, "vel_scale": 0.6, "direction": 0,
            })()

        def _chirp(prog: Any) -> Any:
            c = getattr(prog, "chirp", None)
            return c if c is not None else type("_", (), {
                "steps": 4, "shape": "up", "mode": "chromatic",
                "duration_frac": 0.25, "vel_scale": 0.75,
            })()

        def _echo(prog: Any) -> Any:
            e = getattr(prog, "echo", None)
            return e if e is not None else type("_", (), {
                "lookback_bars": 1, "duration_frac": 0.5,
                "vel_falloff": 0.7, "max_notes": 4,
            })()

        return TorchImprovProgram(
            enabled    = torch.tensor([bool(getattr(p,  "enabled",    False)) for p in programs], dtype=torch.bool,    device=dev),
            prob_grace = torch.tensor([float(getattr(p, "prob_grace", 0.0))   for p in programs], dtype=torch.float64, device=dev).clamp(0.0, 1.0),
            prob_chirp = torch.tensor([float(getattr(p, "prob_chirp", 0.0))   for p in programs], dtype=torch.float64, device=dev).clamp(0.0, 1.0),
            prob_echo  = torch.tensor([float(getattr(p, "prob_echo",  0.0))   for p in programs], dtype=torch.float64, device=dev).clamp(0.0, 1.0),
            grace      = TorchGraceParams.from_grace_params([_grace(p) for p in programs], device=dev),
            chirp      = TorchChirpParams.from_chirp_params([_chirp(p) for p in programs], device=dev),
            echo       = TorchEchoParams.from_echo_params(  [_echo(p)  for p in programs], device=dev),
            step_mask  = step_mask,
            n_pats     = n_pats,
            n_steps    = n_steps,
        )


def apply_improv_to_score(
    score:           ScoreTensor,
    program:         TorchImprovProgram,
    bar_s:           torch.Tensor,      # [B] seconds per bar
    rhythm_division: int,
    phrase:          "Sequence[int]",
    cycle_bars:      int,
) -> ScoreTensor:
    """Apply stochastic improv ornaments (grace / chirp / echo) to a ScoreTensor.

    Mirrors ``improv_engine.apply_improv``.  New events with labels
    ``EVENT_GRACE``, ``EVENT_CHIRP``, or ``EVENT_ECHO`` are appended on the
    event axis.  Main event start/duration may be mutated in-place within the
    returned tensor when grace ``trim_main`` is True or a chirp pushes the note
    forward.  The returned ScoreTensor is sorted by start time per batch row.

    Returns the input unchanged when no batch row has ``program.enabled == True``.
    """
    if not program.enabled.any():
        return score

    B, E, P = score.params.shape
    dev     = score.params.device
    div     = max(1, int(rhythm_division))
    _MIN    = 1.0 / 48000.0

    # Working representation: list[B] of list[(label, [float × P])]
    # Starts as copies of the original valid events; extras are appended per row.
    rows: list[list[tuple[int, list[float]]]] = [
        [(int(score.labels[b, e].item()), score.params[b, e].tolist()) for e in range(E)]
        for b in range(B)
    ]

    for b in range(B):
        if not bool(program.enabled[b].item()):
            continue

        bar_s_b = float(bar_s[b].item())
        step_s  = bar_s_b / div

        pg = float(program.prob_grace[b].item())
        pc = float(program.prob_chirp[b].item())
        pe = float(program.prob_echo[b].item())

        g_mode   = int(program.grace.mode[b].item())
        g_posn   = int(program.grace.position[b].item())
        g_dfrac  = float(program.grace.duration_frac[b].item())
        g_trim   = bool(program.grace.trim_main[b].item())
        g_vscale = float(program.grace.vel_scale[b].item())
        g_dir    = int(program.grace.direction[b].item())

        c_steps  = int(program.chirp.steps[b].item())
        c_shape  = int(program.chirp.shape[b].item())
        c_cmode  = int(program.chirp.mode[b].item())
        c_dfrac  = float(program.chirp.duration_frac[b].item())
        c_vscale = float(program.chirp.vel_scale[b].item())

        e_lbars  = int(program.echo.lookback_bars[b].item())
        e_dfrac  = float(program.echo.duration_frac[b].item())
        e_fall   = float(program.echo.vel_falloff[b].item())
        e_max    = int(program.echo.max_notes[b].item())

        P_max = program.step_mask.shape[1]
        S_max = program.step_mask.shape[2]
        extra: list[tuple[int, list[float]]] = []

        for e in range(E):
            if not bool(score.mask[b, e].item()):
                continue

            ev_p  = rows[b][e][1]           # mutable param list for this event
            start = ev_p[PARAM_START]
            dur   = ev_p[PARAM_DURATION]
            hz    = ev_p[PARAM_HZ]
            vel   = ev_p[PARAM_VELOCITY]

            # Reverse-map onset to (pattern, step) — mirrors apply_improv
            bar_i  = int(start / max(bar_s_b, 1e-12)) % max(1, cycle_bars)
            slot   = bar_i % max(1, len(phrase))
            pat_i  = (phrase[slot] if phrase else 0) % max(1, int(program.n_pats[b].item()))
            step_i = int(round((start % max(bar_s_b, 1e-12)) / max(step_s, 1e-12))) % div
            pat_i  = min(pat_i, P_max - 1)
            step_i = min(step_i, S_max - 1)

            if not bool(program.step_mask[b, pat_i, step_i].item()):
                continue

            # ── Grace ────────────────────────────────────────────────────────
            if pg > 0.0 and float(torch.rand(1).item()) < pg:
                grace_dur = max(_MIN, step_s * g_dfrac)
                if g_mode == int(GraceMode.EITHER):
                    eff_mode = int(GraceMode.MODAL) if float(torch.rand(1).item()) < 0.5 else int(GraceMode.CHROMATIC)
                else:
                    eff_mode = g_mode
                interval  = 3 if eff_mode == int(GraceMode.MODAL) else 1
                sign      = g_dir if g_dir != 0 else (1 if float(torch.rand(1).item()) < 0.5 else -1)
                grace_hz  = hz * (2.0 ** (sign * interval / 12.0))
                g_vel     = vel * g_vscale

                if g_posn in (int(GracePosition.PRE), int(GracePosition.BOTH)):
                    ep = [0.0] * P
                    ep[PARAM_START]    = max(0.0, start - grace_dur)
                    ep[PARAM_DURATION] = grace_dur
                    ep[PARAM_HZ]       = grace_hz
                    ep[PARAM_VELOCITY] = max(0.0, g_vel)
                    ep[PARAM_GATE]     = 1.0
                    extra.append((EVENT_GRACE, ep))
                    if g_trim:
                        new_start            = max(start, ep[PARAM_START] + grace_dur)
                        trim                 = new_start - start
                        ev_p[PARAM_START]    = new_start
                        ev_p[PARAM_DURATION] = max(_MIN, dur - trim)
                        start = ev_p[PARAM_START]
                        dur   = ev_p[PARAM_DURATION]

                if g_posn in (int(GracePosition.POST), int(GracePosition.BOTH)):
                    ep = [0.0] * P
                    ep[PARAM_START]    = start + dur
                    ep[PARAM_DURATION] = grace_dur
                    ep[PARAM_HZ]       = grace_hz
                    ep[PARAM_VELOCITY] = max(0.0, g_vel)
                    ep[PARAM_GATE]     = 1.0
                    extra.append((EVENT_GRACE, ep))

            # ── Chirp ────────────────────────────────────────────────────────
            if pc > 0.0 and float(torch.rand(1).item()) < pc:
                chirp_dur  = max(_MIN * c_steps, step_s * c_dfrac)
                note_dur   = chirp_dur / c_steps
                c_interval = 3 if c_cmode == int(ChirpMode.MODAL) else 1
                hzs: list[float] = []
                if c_shape == int(ChirpShape.UP):
                    for k in range(c_steps):
                        hzs.append(hz * (2.0 ** (k * c_interval / 12.0)))
                elif c_shape == int(ChirpShape.DOWN):
                    for k in range(c_steps):
                        hzs.append(hz * (2.0 ** (-k * c_interval / 12.0)))
                elif c_shape == int(ChirpShape.BOUNCE):
                    half = c_steps // 2
                    for k in range(half):
                        hzs.append(hz * (2.0 ** (k * c_interval / 12.0)))
                    for k in range(c_steps - half):
                        hzs.append(hz * (2.0 ** ((half - k - 1) * c_interval / 12.0)))
                else:  # RANDOM
                    cur_off = 0
                    for _ in range(c_steps):
                        cur_off += c_interval * (1 if float(torch.rand(1).item()) < 0.5 else -1)
                        hzs.append(hz * (2.0 ** (cur_off / 12.0)))
                t_c = start
                for hz_c in hzs:
                    ep = [0.0] * P
                    ep[PARAM_START]    = t_c
                    ep[PARAM_DURATION] = note_dur
                    ep[PARAM_HZ]       = hz_c
                    ep[PARAM_VELOCITY] = vel * c_vscale
                    ep[PARAM_GATE]     = 1.0
                    extra.append((EVENT_CHIRP, ep))
                    t_c += note_dur
                push                 = t_c - start
                ev_p[PARAM_START]    = t_c
                ev_p[PARAM_DURATION] = max(_MIN, dur - push)
                start = ev_p[PARAM_START]
                dur   = ev_p[PARAM_DURATION]

            # ── Echo ─────────────────────────────────────────────────────────
            if pe > 0.0 and float(torch.rand(1).item()) < pe:
                window_start = start - e_lbars * bar_s_b
                cands: list[tuple[float, float, float]] = [
                    (rows[b][e2][1][PARAM_START],
                     rows[b][e2][1][PARAM_DURATION],
                     rows[b][e2][1][PARAM_HZ])
                    for e2 in range(e)
                    if bool(score.mask[b, e2].item())
                    and rows[b][e2][1][PARAM_START] >= window_start
                ]
                if not cands:
                    cands = [
                        (rows[b][e2][1][PARAM_START],
                         rows[b][e2][1][PARAM_DURATION],
                         rows[b][e2][1][PARAM_HZ])
                        for e2 in range(e)
                        if bool(score.mask[b, e2].item())
                    ]
                cands = sorted(cands, key=lambda c: c[0])[-e_max:]
                if cands:
                    src_span = max(1e-9, cands[-1][0] - cands[0][0] + cands[-1][1])
                    tgt_dur  = max(_MIN * len(cands), dur * e_dfrac)
                    scale    = tgt_dur / src_span
                    t_anch   = start + dur
                    src_t0   = cands[0][0]
                    echo_vel = vel
                    for cs, cd, ch in cands:
                        ep = [0.0] * P
                        ep[PARAM_START]    = t_anch + (cs - src_t0) * scale
                        ep[PARAM_DURATION] = max(_MIN, cd * scale)
                        ep[PARAM_HZ]       = ch
                        ep[PARAM_VELOCITY] = echo_vel
                        ep[PARAM_GATE]     = 1.0
                        extra.append((EVENT_ECHO, ep))
                        echo_vel *= e_fall

        # Append extra events to this row
        for item in extra:
            rows[b].append(item)

    # Assemble output: collect valid original events (with mutated params) + extras,
    # sorted by start time, into a new ScoreTensor.
    all_events_per_row: list[list[tuple[int, list[float]]]] = []
    for b in range(B):
        evs: list[tuple[int, list[float]]] = []
        for e in range(E):
            if bool(score.mask[b, e].item()):
                evs.append(rows[b][e])
        for item in rows[b][E:]:    # extras
            evs.append(item)
        evs.sort(key=lambda x: x[1][PARAM_START])
        all_events_per_row.append(evs)

    E_new = max((len(evs) for evs in all_events_per_row), default=1)
    labels_out = torch.zeros((B, E_new),    dtype=torch.long,    device=dev)
    params_out = torch.zeros((B, E_new, P), dtype=torch.float64, device=dev)
    mask_out   = torch.zeros((B, E_new),    dtype=torch.bool,    device=dev)

    for b, evs in enumerate(all_events_per_row):
        for ei, (lbl, ep) in enumerate(evs):
            labels_out[b, ei] = lbl
            params_out[b, ei] = torch.tensor(ep, dtype=torch.float64, device=dev)
            mask_out[b, ei]   = True

    return ScoreTensor(
        labels=labels_out, params=params_out, mask=mask_out,
        param_names=score.param_names, label_names=score.label_names,
        metadata=dict(score.metadata),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rubato: apply_rubato_to_score
# ─────────────────────────────────────────────────────────────────────────────


def apply_rubato_to_score(
    score:          ScoreTensor,
    rubato_shape:   "str | int | torch.Tensor",  # WarpShape or name string, [B] or scalar
    rubato_amount:  "float | torch.Tensor",      # [B] or scalar, ∈ [0, 0.95]
    cycle_s:        torch.Tensor,                # [B] seconds per rubato cycle
) -> ScoreTensor:
    """Remap event times through a phrase-level rubato warp.

    Mirrors ``analytic_score._apply_rubato_to_schedule``.  Both ``start_time``
    and ``end_time`` of each event are mapped through the analytical rubato
    phase function; duration is recomputed as their difference.

    The rubato shapes are the same five families used by ``TorchWarpCurve``
    (WarpShape enum): OFF, SINE, TROUGHS, SLOW_GO, GO_SLOW.

    ``cycle_s`` is the period of the rubato warp — one bar for bar-scope rubato,
    or the phrase length for phrase-scope (matching ``seq_rubato_scope``).

    Returns the input unchanged where ``rubato_shape == OFF`` or
    ``rubato_amount ≤ 0`` for a given batch row.
    """
    B, E, P = score.params.shape
    dev     = score.params.device

    # ── Normalise shape → int64 [B] ───────────────────────────────────────
    _SHAPE_MAP = {
        "off": int(WarpShape.OFF), "sine": int(WarpShape.SINE),
        "troughs": int(WarpShape.TROUGHS), "slow_go": int(WarpShape.SLOW_GO),
        "go_slow": int(WarpShape.GO_SLOW),
    }
    if isinstance(rubato_shape, str):
        shape_t = torch.full((B,), _SHAPE_MAP.get(rubato_shape.lower(), int(WarpShape.OFF)),
                             dtype=torch.long, device=dev)
    elif isinstance(rubato_shape, int):
        shape_t = torch.full((B,), rubato_shape, dtype=torch.long, device=dev)
    else:
        shape_t = torch.as_tensor(rubato_shape, dtype=torch.long, device=dev)
        if shape_t.ndim == 0:
            shape_t = shape_t.expand(B).clone()

    amt_t = torch.as_tensor(rubato_amount, dtype=torch.float64, device=dev)
    if amt_t.ndim == 0:
        amt_t = amt_t.expand(B).clone()
    amt_t   = amt_t.to(device=dev, dtype=torch.float64).clamp(0.0, 0.95)
    cycle_t = cycle_s.to(device=dev, dtype=torch.float64).clamp(min=1e-9)

    is_off = (shape_t == int(WarpShape.OFF)) | (amt_t < 1e-9)
    if is_off.all():
        return score

    def _phase_map(t: torch.Tensor) -> torch.Tensor:
        """Map absolute times through the rubato curve; shape [B, E]."""
        cycle = cycle_t.unsqueeze(1)                              # [B, 1]
        c_idx = torch.floor(t / cycle)                           # cycle number [B, E]
        u     = ((t - c_idx * cycle) / cycle).clamp(0.0, 1.0)   # phase ∈ [0, 1] [B, E]

        pi  = math.pi
        amt = amt_t.unsqueeze(1).expand_as(u)
        sh  = shape_t.unsqueeze(1).expand_as(u)

        d_sine    = amt * torch.sin(2.0 * pi * u) / (2.0 * pi)
        d_troughs = amt * torch.sin(4.0 * pi * u) / (4.0 * pi)
        d_slow_go = amt * (u * u - u)
        d_go_slow = amt * u * (1.0 - u)

        delta = torch.zeros_like(u)
        delta = torch.where(sh == int(WarpShape.SINE),    d_sine,    delta)
        delta = torch.where(sh == int(WarpShape.TROUGHS), d_troughs, delta)
        delta = torch.where(sh == int(WarpShape.SLOW_GO), d_slow_go, delta)
        delta = torch.where(sh == int(WarpShape.GO_SLOW), d_go_slow, delta)

        warped_u = (u + delta).clamp(0.0, 1.0)
        return c_idx * cycle + warped_u * cycle

    start     = score.params[..., PARAM_START]
    dur       = score.params[..., PARAM_DURATION]
    end       = start + dur

    w_start   = _phase_map(start)
    w_end     = _phase_map(end)
    w_dur     = (w_end - w_start).clamp(min=1.0 / 48000.0)

    enabled_be = (~is_off).unsqueeze(1).expand(B, E)
    new_params = score.params.clone()
    new_params[..., PARAM_START]    = torch.where(enabled_be, w_start, start)
    new_params[..., PARAM_DURATION] = torch.where(enabled_be, w_dur,   dur)

    return ScoreTensor(
        labels=score.labels, params=new_params, mask=score.mask,
        param_names=score.param_names, label_names=score.label_names,
        metadata=dict(score.metadata),
    )
