import types

import torch

from analytic_model import AnalyticPatch, AnalyticVoice
from parametric_curve import default_chirp, default_envelope
from torch_composer_engine import (
    EVENT_NOTE,
    PARAM_HZ,
    PARAM_START,
    PARAM_DURATION,
    PARAM_PAGE,
    PARAM_SECTION,
    PARAM_VELOCITY,
    ArticulationCode,
    DynamicsShape,
    ScoreEventLabel,
    ScoreParam,
    ScoreSection,
    ScoreTensor,
    envelope_jobs_from_sparse_score,
    performance_atoms_from_jobs,
    PerformanceAtom,
    TorchAccentPattern,
    TorchBeatTree,
    TorchDynamicsCurve,
    TorchDynamicsProgram,
    TorchNoteStream,
    TorchWarpCurve,
    WarpShape,
    apply_dynamics_to_score,
    build_demo_sparse_score_from_patch,
    empty_score_tensor,
    sparse_score_tensor_from_score,
    schedule_beat_tree,
    score_tensor_from_schedules,
)


def _make_schedule(events: list) -> object:
    sched = types.SimpleNamespace(events=events)
    return sched


def _make_event(hz: float, start: float, duration: float, velocity: float = 1.0) -> object:
    return types.SimpleNamespace(
        fundamental_hz=hz,
        start_time=start,
        duration_s=duration,
        velocity=velocity,
    )


def test_score_tensor_from_schedules_packs_note_events() -> None:
    events = [
        _make_event(220.0, 0.0, 0.25),
        _make_event(440.0, 0.5, 0.25),
    ]
    score = score_tensor_from_schedules([_make_schedule(events)])

    assert score.labels.shape == (1, 2)
    assert score.params.shape == (1, 2, len(score.param_names))
    assert score.mask.all()
    assert torch.all(score.labels == EVENT_NOTE)
    assert torch.isclose(score.params[0, 0, PARAM_HZ], torch.tensor(220.0, dtype=torch.float64))
    assert torch.isclose(score.params[0, 1, PARAM_START], torch.tensor(0.5, dtype=torch.float64))


def test_score_event_enum_attribution_round_trips_via_param_axis() -> None:
    score = empty_score_tensor(batch_size=1, max_events=1)
    score.labels[0, 0] = int(ScoreEventLabel.NOTE)
    score.mask[0, 0] = True
    score.params[0, 0, PARAM_SECTION] = float(ScoreSection.RHYTHM)

    assert score.labels[0, 0].item() == ScoreEventLabel.NOTE
    assert int(score.params[0, 0, PARAM_SECTION].item()) == ScoreSection.RHYTHM
    assert ScoreParam.SECTION == PARAM_SECTION


def test_sparse_score_packet_keeps_page_axis_and_float_event_enum() -> None:
    score = empty_score_tensor(batch_size=1, max_events=2)
    score.labels[:] = EVENT_NOTE
    score.mask[:] = True
    score.params[0, 0, PARAM_PAGE] = 0.0
    score.params[0, 1, PARAM_PAGE] = 1.0
    score.params[0, 1, PARAM_HZ] = 440.0

    packet = sparse_score_tensor_from_score(score)

    assert packet.data.shape == (1, 1, 2, 1, 2, len(score.param_names))
    assert packet.mask[0, 0, 0, 0]
    assert packet.mask[0, 0, 1, 0]
    assert packet.data[0, 0, 1, 0, 0, 0].item() == float(EVENT_NOTE)
    assert packet.data[0, 0, 1, 0, 1, PARAM_HZ].item() == 440.0


def test_envelope_jobs_track_note_off_through_release_tail() -> None:
    score = empty_score_tensor(batch_size=1, max_events=1)
    score.labels[0, 0] = EVENT_NOTE
    score.mask[0, 0] = True
    score.params[0, 0, PARAM_START] = 0.125
    score.params[0, 0, PARAM_DURATION] = 0.25
    score.params[0, 0, PARAM_HZ] = 220.0
    score.params[0, 0, PARAM_VELOCITY] = 0.5
    packet = sparse_score_tensor_from_score(score)

    jobs = envelope_jobs_from_sparse_score(packet, sample_rate=8.0, release_tail_s=0.25)

    assert jobs.job_count == 1
    assert jobs.sample_start.item() == 1
    assert jobs.sample_count.item() == 4
    assert torch.isclose(jobs.end_time[0], torch.tensor(0.625, dtype=torch.float64))
    assert torch.isclose(jobs.sub_sample_offset_s[0], torch.tensor(0.0, dtype=torch.float64))


def test_performance_atoms_from_jobs_two_notes() -> None:
    SR = 48000
    score = empty_score_tensor(batch_size=1, max_events=2)
    score.labels[:] = EVENT_NOTE
    score.mask[:] = True
    score.params[0, 0, PARAM_START]    = 0.0
    score.params[0, 0, PARAM_DURATION] = 0.4
    score.params[0, 0, PARAM_HZ]       = 261.63   # C4
    score.params[0, 0, PARAM_VELOCITY] = 1.0
    score.params[0, 1, PARAM_START]    = 0.5
    score.params[0, 1, PARAM_DURATION] = 0.4
    score.params[0, 1, PARAM_HZ]       = 329.63   # E4
    score.params[0, 1, PARAM_VELOCITY] = 0.8
    packet = sparse_score_tensor_from_score(score)
    jobs = envelope_jobs_from_sparse_score(packet, sample_rate=float(SR), release_tail_s=0.08)

    atoms = performance_atoms_from_jobs(
        jobs,
        envelope_curves=[default_envelope("env")],
        chirp_curves=[default_chirp("chirp")],
        voice_key="test_voice",
        sample_rate=float(SR),
    )

    assert len(atoms) == 2
    assert atoms[0].voice_key == "test_voice"
    assert abs(atoms[0].fundamental_hz - 261.63) < 0.01
    assert abs(atoms[1].fundamental_hz - 329.63) < 0.01
    assert atoms[0].onset_sample == 0
    assert atoms[1].onset_sample == int(0.5 * SR)
    assert atoms[0].envelope_curve is not None
    assert atoms[0].chirp_curve is not None
    assert atoms[1].envelope_curve is not None
    assert atoms[1].chirp_curve is not None


def test_patch_demo_sparse_score_uses_batch_dimension() -> None:
    patch = AnalyticPatch()
    patch.preview_sr = 100
    patch.duration = 0.2
    patch.voices = [AnalyticVoice()]

    packet = build_demo_sparse_score_from_patch(patch, batch_size=3)

    assert packet.batch_size == 3
    assert packet.data.shape[1] == 1


# ─────────────────────────────────────────────────────────────────────────────
# TorchWarpCurve
# ─────────────────────────────────────────────────────────────────────────────


def test_warp_curve_identity_when_no_params() -> None:
    warp = TorchWarpCurve.build(batch_size=1)
    frac = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
    warped = warp.warp(frac)
    assert warped.shape == (1, 17)
    assert torch.allclose(warped[0], frac, atol=1e-9)


def test_warp_curve_swing_delays_odd_anchors() -> None:
    warp = TorchWarpCurve.build(swing=0.5, batch_size=1)
    # Anchor 1 (odd) should be later than straight 1/16
    assert warp.warped[0, 1] > torch.tensor(1.0 / 16, dtype=torch.float64)
    # Even anchors unchanged
    assert torch.isclose(warp.warped[0, 2], torch.tensor(2.0 / 16, dtype=torch.float64), atol=1e-6)


def test_warp_curve_endpoints_always_pinned() -> None:
    for shape in WarpShape:
        warp = TorchWarpCurve.build(rubato_amount=0.8, rubato_shape=shape, batch_size=3)
        assert torch.all(warp.warped[:, 0]  == 0.0)
        assert torch.all(warp.warped[:, -1] == 1.0)


def test_warp_curve_from_coefficients_roundtrips() -> None:
    warp1  = TorchWarpCurve.build(swing=0.3, pocket=0.05, batch_size=2)
    warp2  = TorchWarpCurve.from_coefficients(warp1.warped)
    frac   = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    assert torch.allclose(warp1.warp(frac), warp2.warp(frac), atol=1e-12)


def test_warp_curve_batch_independent() -> None:
    swing  = torch.tensor([0.0, 0.5], dtype=torch.float64)
    warp   = TorchWarpCurve.build(swing=swing, batch_size=2)
    # Row 0: no swing, row 1: strong swing — anchor 1 should differ
    assert warp.warped[0, 1] < warp.warped[1, 1]


def test_warp_to_seconds_scales_by_bar() -> None:
    warp  = TorchWarpCurve.build(batch_size=2)
    frac  = torch.tensor([[0.0, 0.25, 0.5, 1.0]], dtype=torch.float64).expand(2, -1)
    bar_s = torch.tensor([2.0, 4.0], dtype=torch.float64)
    secs  = warp.warp_to_seconds(frac, bar_s)
    assert secs.shape == (2, 4)
    assert torch.isclose(secs[0, 2], torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(secs[1, 2], torch.tensor(2.0, dtype=torch.float64))


# ─────────────────────────────────────────────────────────────────────────────
# TorchBeatTree
# ─────────────────────────────────────────────────────────────────────────────

def _make_flat_leaves(div: int = 4, on_steps: list[int] | None = None) -> object:
    """Return a SimpleNamespace that looks like a RhythmPattern in flat mode."""
    on_steps = on_steps or [0, 2]
    ns = types.SimpleNamespace()
    ns.steps = [1 if i in on_steps else 0 for i in range(div)]
    ns.vel   = [1.0] * div
    ns.art   = [0]   * div

    def is_tree_mode() -> bool:
        return False
    def ensure_size(n: int) -> None:
        pass

    ns.is_tree_mode = is_tree_mode
    ns.ensure_size  = ensure_size
    return ns


def test_beat_tree_from_rhythm_pattern_flat_mode_shape() -> None:
    pats = [_make_flat_leaves(4), _make_flat_leaves(4)]
    tree = TorchBeatTree.from_rhythm_pattern(pats, home_div=4)
    assert tree.batch_size == 2
    assert tree.max_leaves == 4
    assert tree.leaf_mask.all()


def test_beat_tree_from_rhythm_pattern_on_off() -> None:
    pat  = _make_flat_leaves(4, on_steps=[0, 2])
    tree = TorchBeatTree.from_rhythm_pattern([pat], home_div=4)
    assert tree.leaf_on[0, 0].item() is True
    assert tree.leaf_on[0, 1].item() is False
    assert tree.leaf_on[0, 2].item() is True
    assert tree.leaf_on[0, 3].item() is False


def test_beat_tree_leaf_positions_uniform_flat() -> None:
    pat  = _make_flat_leaves(4)
    tree = TorchBeatTree.from_rhythm_pattern([pat], home_div=4)
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)
    assert torch.allclose(tree.leaf_pos[0], expected)


# ─────────────────────────────────────────────────────────────────────────────
# schedule_beat_tree
# ─────────────────────────────────────────────────────────────────────────────

def test_schedule_beat_tree_event_count_matches_on_leaves() -> None:
    pat  = _make_flat_leaves(4, on_steps=[0, 1, 2, 3])
    tree = TorchBeatTree.from_rhythm_pattern([pat], home_div=4)
    warp = TorchWarpCurve.build(batch_size=1)
    score = schedule_beat_tree(
        tree, warp,
        bar_s=torch.tensor([2.0], dtype=torch.float64),
        abs_bar=torch.tensor([0], dtype=torch.long),
    )
    assert score.mask[0].sum().item() == 4


def test_schedule_beat_tree_repeats_offset_bar() -> None:
    pat  = _make_flat_leaves(4, on_steps=[0])
    tree = TorchBeatTree.from_rhythm_pattern([pat], home_div=4)
    warp = TorchWarpCurve.build(batch_size=1, home_div=4)
    bar_s = torch.tensor([2.0], dtype=torch.float64)
    score = schedule_beat_tree(
        tree, warp, bar_s=bar_s,
        abs_bar=torch.tensor([0], dtype=torch.long),
        n_repeats=3,
    )
    # 1 on-leaf × 3 repeats = 3 valid events
    assert score.mask[0].sum().item() == 3
    # Extract only the valid event starts
    valid_starts = score.params[0, score.mask[0], PARAM_START]
    assert torch.isclose(valid_starts[0], torch.tensor(0.0, dtype=torch.float64))
    assert torch.isclose(valid_starts[1], torch.tensor(2.0, dtype=torch.float64))
    assert torch.isclose(valid_starts[2], torch.tensor(4.0, dtype=torch.float64))


def test_schedule_beat_tree_staccato_halves_gate() -> None:
    pat = _make_flat_leaves(2, on_steps=[0])
    pat.art = [int(ArticulationCode.STACCATO), 0]
    tree = TorchBeatTree.from_rhythm_pattern([pat], home_div=2)
    warp = TorchWarpCurve.build(batch_size=1)
    score = schedule_beat_tree(
        tree, warp,
        bar_s=torch.tensor([2.0], dtype=torch.float64),
        abs_bar=torch.tensor([0], dtype=torch.long),
    )
    # leaf_dur = 0.5 bar-frac = 1.0 s; staccato × 0.5 = 0.5 s gate
    dur = score.params[0, 0, PARAM_DURATION].item()
    assert abs(dur - 0.5) < 1e-6


def test_schedule_beat_tree_warp_applied() -> None:
    # home_div=4: leaf positions 0, 0.25, 0.5, 0.75 map to anchors 0,1,2,3.
    # Anchor 1 (position 0.25) is odd — swing should push it later.
    pat  = _make_flat_leaves(4, on_steps=[0, 1])
    tree = TorchBeatTree.from_rhythm_pattern([pat], home_div=4)
    warp_straight = TorchWarpCurve.build(swing=0.0, home_div=4, batch_size=1)
    warp_swing    = TorchWarpCurve.build(swing=0.5, home_div=4, batch_size=1)
    bar_s = torch.tensor([4.0], dtype=torch.float64)
    abs_b = torch.tensor([0], dtype=torch.long)

    s0 = schedule_beat_tree(tree, warp_straight, bar_s=bar_s, abs_bar=abs_b)
    s1 = schedule_beat_tree(tree, warp_swing,    bar_s=bar_s, abs_bar=abs_b)

    # Extract valid events (on-leaves at positions 0 and 1)
    starts0 = s0.params[0, s0.mask[0], PARAM_START]
    starts1 = s1.params[0, s1.mask[0], PARAM_START]
    # First onset (position 0.0, even anchor) unchanged by swing
    assert torch.isclose(starts0[0], starts1[0])
    # Second onset (position 0.25, odd anchor 1) should be later with swing
    assert starts1[1] > starts0[1]


# ─────────────────────────────────────────────────────────────────────────────
# TorchDynamicsCurve
# ─────────────────────────────────────────────────────────────────────────────

def test_dynamics_curve_flat_is_passthrough() -> None:
    curve = TorchDynamicsCurve.build(shape="flat", intensity=1.0, batch_size=2)
    bar_f = torch.tensor([0.0, 0.5], dtype=torch.float64)
    mult  = curve.multiplier_at_bar(bar_f)
    assert torch.allclose(mult, torch.ones(2, dtype=torch.float64))


def test_dynamics_curve_crescendo_increases() -> None:
    curve = TorchDynamicsCurve.build(shape="crescendo", intensity=1.0, batch_size=1)
    # 3 bar positions: early, mid, late — multiplier should increase
    bar_f = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64).unsqueeze(0)  # [1, 3]
    mult  = curve.multiplier_at_bar(bar_f)
    assert mult[0, 0] < mult[0, 1] < mult[0, 2]


def test_dynamics_curve_zero_intensity_always_one() -> None:
    for shape in DynamicsShape:
        curve = TorchDynamicsCurve.build(shape=shape, intensity=0.0, batch_size=1)
        bar_f = torch.tensor([0.3], dtype=torch.float64)
        mult  = curve.multiplier_at_bar(bar_f)
        assert torch.isclose(mult[0], torch.tensor(1.0, dtype=torch.float64))


def test_dynamics_curve_scope_tiles() -> None:
    # scope_bars=2: position 2.1 should behave like 0.1
    curve = TorchDynamicsCurve.build(shape="crescendo", scope_bars=2.0, intensity=1.0, batch_size=1)
    m1 = curve.multiplier_at_bar(torch.tensor([0.1], dtype=torch.float64))
    m2 = curve.multiplier_at_bar(torch.tensor([2.1], dtype=torch.float64))
    assert torch.isclose(m1[0], m2[0], atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# TorchAccentPattern
# ─────────────────────────────────────────────────────────────────────────────

def test_accent_pattern_level_at_cycles() -> None:
    levels = [[1.0, 1.5, 2.0, 0.5]]
    acc    = TorchAccentPattern.build(levels=levels)
    step   = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)  # 4 wraps to 0
    result = acc.level_at(step.unsqueeze(0).expand(1, -1))
    assert torch.isclose(result[0, 0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(result[0, 2], torch.tensor(2.0, dtype=torch.float64))
    assert torch.isclose(result[0, 4], torch.tensor(1.0, dtype=torch.float64))  # wraps


def test_accent_pattern_batch_independent() -> None:
    acc = TorchAccentPattern.build(levels=[[2.0, 1.0], [0.5, 1.5]])
    s   = torch.tensor([0], dtype=torch.long)
    r0  = acc.level_at(s.unsqueeze(0).expand(1, -1)[:1])
    r1  = acc.level_at(s.unsqueeze(0).expand(1, -1)[:1])
    assert float(acc.levels[0, 0]) == 2.0
    assert float(acc.levels[1, 0]) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# apply_dynamics_to_score
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_dynamics_disabled_is_noop() -> None:
    score = empty_score_tensor(batch_size=1, max_events=2)
    score.labels[0, :] = EVENT_NOTE
    score.mask[0, :]   = True
    score.params[0, :, PARAM_VELOCITY] = 0.7
    score.params[0, 0, PARAM_START]    = 0.0
    score.params[0, 1, PARAM_START]    = 1.0

    prog = TorchDynamicsProgram.build(batch_size=1)   # enabled=False
    out  = apply_dynamics_to_score(score, prog, bar_s=torch.tensor([2.0]), rhythm_division=4)
    assert out is score   # fast-path: same object returned


def test_apply_dynamics_crescendo_raises_late_velocities() -> None:
    # Place 4 events at fractional positions within a single bar so their
    # bar-fraction positions differ and the crescendo curve spreads across them.
    score   = empty_score_tensor(batch_size=1, max_events=4)
    bar_s_v = 4.0
    starts  = [0.1, 1.0, 2.0, 3.5]   # within one bar (scope_bars=4.0)
    for i, t in enumerate(starts):
        score.labels[0, i]                 = EVENT_NOTE
        score.mask[0, i]                   = True
        score.params[0, i, PARAM_VELOCITY] = 1.0
        score.params[0, i, PARAM_START]    = t

    prog           = TorchDynamicsProgram.build(batch_size=1)
    prog.enabled[0] = True
    prog.curve      = TorchDynamicsCurve.build(
        shape="crescendo", scope_bars=bar_s_v, intensity=1.0, batch_size=1,
    )

    out  = apply_dynamics_to_score(
        score, prog,
        bar_s=torch.tensor([bar_s_v], dtype=torch.float64),
        rhythm_division=4,
    )
    vels = out.params[0, :, PARAM_VELOCITY]
    # Crescendo: later events in the bar should be louder
    assert vels[3] > vels[0]


# ─────────────────────────────────────────────────────────────────────────────
# TorchNoteStream
# ─────────────────────────────────────────────────────────────────────────────

def test_note_stream_sequential_traversal() -> None:
    hz_table   = [[220.0, 330.0, 440.0]]
    deg_pattern = [[0, 1, 2]]
    stream = TorchNoteStream.build(degrees=hz_table, deg_pattern=deg_pattern)
    stream.reset()
    results = [stream.next_hz().item() for _ in range(3)]
    assert results == [220.0, 330.0, 440.0]


def test_note_stream_exhausted_returns_zero() -> None:
    stream = TorchNoteStream.build(degrees=[[440.0]], deg_pattern=[[0]])
    stream.next_hz()   # consume the one note
    assert stream.exhausted().item() is True
    assert stream.next_hz().item() == 0.0


def test_note_stream_reset_restarts() -> None:
    stream = TorchNoteStream.build(degrees=[[220.0, 440.0]], deg_pattern=[[0, 1]])
    stream.next_hz(); stream.next_hz()
    assert stream.exhausted().item() is True
    stream.reset()
    assert not stream.exhausted().item()
    assert stream.next_hz().item() == 220.0


def test_note_stream_batch_independent_traversal() -> None:
    stream = TorchNoteStream.build(
        degrees     = [[100.0, 200.0], [300.0, 400.0]],
        deg_pattern = [[0, 1],         [1, 0]],
    )
    h0 = stream.next_hz()
    assert torch.isclose(h0[0], torch.tensor(100.0, dtype=torch.float64))
    assert torch.isclose(h0[1], torch.tensor(400.0, dtype=torch.float64))


def test_note_stream_chromatic_shifts_hz() -> None:
    torch.manual_seed(0)
    stream = TorchNoteStream.build(
        degrees=[[440.0]], deg_pattern=[[0]],
        p_chromatic=1.0,  # always apply
    )
    hz = stream.next_hz().item()
    # ±1 semitone → 440 × 2^(±1/12)
    semitone = 2.0 ** (1.0 / 12.0)
    assert abs(hz - 440.0 * semitone) < 1e-6 or abs(hz - 440.0 / semitone) < 1e-6


def test_note_stream_from_note_streams_packs_state() -> None:
    class _FakeStream:
        _degrees     = [220.0, 440.0]
        _deg_pattern = [0, 1]
        _probs       = types.SimpleNamespace(
            double_back=0.1, subversion=0.2, chromatic=0.3, modal=0.4
        )

    stream = TorchNoteStream.from_note_streams([_FakeStream(), _FakeStream()])
    assert stream.batch_size == 2
    assert torch.isclose(stream.p_chromatic[0], torch.tensor(0.3, dtype=torch.float64))
