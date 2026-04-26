#!/usr/bin/env python3
"""Rhythm, score, and arrangement scheduling helpers."""
from __future__ import annotations

from analytic_shared import *  # noqa: F401,F403
import analytic_shared as _analytic_shared
import analytic_model as _analytic_model
import analytic_routing as _analytic_routing

def _import_all_from(module):
    globals().update({
        k: v for k, v in vars(module).items()
        if k != "_import_all_from" and not (k.startswith('__') and k.endswith('__'))
    })

_import_all_from(_analytic_model)
_import_all_from(_analytic_routing)
_import_all_from(_analytic_shared)

# Articulation → gate-fraction multiplier.
# None = drone: gate is stretched to reach the next onset.
# 0=normal  1=staccato  2=legato  3=drone
_ART_GATE: dict = {0: 1.0, 1: 0.5, 2: 0.95, 3: None}

def _build_rhythm_schedule(
        p: "AnalyticPatch",
        beat_s: float,
        degrees: list,
        deg_pattern: list,
        min_dur_s: float = 1.0 / 48000,
        page: "RhythmPage | None" = None,
        dynamics_program: "DynamicsProgram | None" = None,
        improv_program: "ImprovProgram | None" = None,
) -> "NoteSchedule":
    """Convert rhythm program + scale degrees into a NoteSchedule.

    *page* selects which RhythmPage to drive the schedule.  When None,
    the patch's 'all' page (or flat legacy fields) is used.

    One *progression cycle* = one full walk through ``deg_pattern`` via a
    ``NoteStream`` that applies ``p.seq_probabilities`` transforms per onset.
    One *phrase cycle*      = ``rhythm_prog_bars`` bars of the rhythm phrase.

    Fit modes
    ---------
    "drop"   — Run exactly ``rhythm_prog_bars`` bars.  Onsets beyond the
               progression length become rests.
    "extend" — Add whole phrase-length chunks until onset count ≥ n_pat.
               Extra onsets at the tail are rests.

    Swing:   odd-indexed steps pushed back by ``rhythm_swing × step_s``.
    Pocket:  every onset shifted by ``rhythm_pocket × beat_s`` (pos = lay-back).
    Gate:    note duration = ``rhythm_gate × step_s`` × articulation multiplier.
    Drone:   gate extends to reach the next onset in the bar (two-pass).
    """
    # Resolve the page: prefer explicit arg, then voice register page, then legacy flat.
    pg: "RhythmPage" = page if page is not None else p.page_for("all")

    sched      = NoteSchedule()
    div        = max(1, pg.rhythm_division)
    bar_s      = _bar_duration_s_for_page(p, pg, beat_s)
    step_s     = bar_s / div
    base_gate  = step_s * pg.rhythm_gate
    phrase     = pg.rhythm_phrase if pg.rhythm_phrase else [0]
    pats       = pg.rhythm_patterns if pg.rhythm_patterns else [RhythmPattern(name="Pat 1")]
    n_pat      = len(deg_pattern)
    prog_bars  = max(1, pg.rhythm_prog_bars)
    fit_mode   = pg.rhythm_fit_mode
    probs      = getattr(p, "seq_probabilities", None) or SequenceProbabilities()
    _rng       = _random.Random()   # local instance — doesn't touch global state
    stream     = NoteStream(degrees, deg_pattern, probs, _rng)

    # ── Warp curve — built once from home-grid params, shared across all bars ─
    _meter_num, _ = p.page_meter(pg)
    _warp = build_warp_curve(
        home_div      = div,
        swing         = pg.rhythm_swing,
        pocket        = pg.rhythm_pocket,
        rubato_shape  = getattr(p, "seq_rubato_shape",  "off"),
        rubato_amount = getattr(p, "seq_rubato_amount", 0.0),
        meter_num     = _meter_num,
        beats_per_bar = p.beats_per_bar(),
        interpolator  = getattr(pg, "warp_interpolator", "linear"),
        frac_beat_mode = getattr(pg, "frac_beat_mode", "warp"),
    )

    def _onsets_in_bars(start_bar: int, end_bar: int) -> int:
        total = 0
        for b in range(start_bar, end_bar):
            slot  = b % len(phrase)
            pat_i = phrase[slot]
            pat   = pats[min(pat_i, len(pats) - 1)]
            if pat.is_tree_mode():
                total += sum(1 for lf in pat.get_tree(div).flat_leaves() if lf.on)
            else:
                pat.ensure_size(div)
                total += sum(1 for s in pat.steps[:div] if s)
        return total

    # Determine cycle length in bars for this render
    if fit_mode == "extend":
        cycle_bars = prog_bars
        phrase_len = max(1, len(phrase))
        while _onsets_in_bars(0, cycle_bars) < n_pat and cycle_bars < prog_bars + phrase_len * 64:
            cycle_bars += phrase_len
    else:  # "drop"
        cycle_bars = prog_bars

    # ── First pass: collect all onset times per bar for drone gate lookahead ──
    # onset_map[abs_bar][leaf_key] = t_final for active leaves.
    # leaf_key is step_i (flat) or node_id (tree) — used only for drone lookups.
    onset_map: dict = {}
    for rep in range(max(1, p.seq_repeats)):
        for bar_i in range(cycle_bars):
            slot  = bar_i % len(phrase)
            pat_i = phrase[slot]
            pat   = pats[min(pat_i, len(pats) - 1)]
            abs_bar = rep * cycle_bars + bar_i
            if pat.is_tree_mode():
                for leaf, t_onset, _ in iter_leaf_events(pat.get_tree(div), _warp, bar_s, abs_bar):
                    onset_map.setdefault(abs_bar, {})[leaf.node_id] = t_onset
            else:
                pat.ensure_size(div)
                for step_i in range(div):
                    if not pat.steps[step_i]:
                        continue
                    t_onset = abs_bar * bar_s + _warp.warp_to_seconds(
                        step_i / div, bar_s)
                    onset_map.setdefault(abs_bar, {})[step_i] = t_onset

    def _next_onset_t(abs_bar: int, leaf_key: object, t_cur: float) -> float:
        """Return t of next onset after *leaf_key* in *abs_bar*, for drone gate."""
        max_bar = abs_bar + cycle_bars
        for b in range(abs_bar, max_bar + 1):
            bar_onsets = onset_map.get(b, {})
            for k in sorted(bar_onsets, key=lambda k: bar_onsets[k]):
                if b == abs_bar and bar_onsets[k] <= t_cur:
                    continue
                return bar_onsets[k]
        return t_cur + bar_s   # fallback: one bar ahead

    # ── Second pass: build schedule with articulation-aware gate ─────────────
    for rep in range(max(1, p.seq_repeats)):
        stream.reset()   # each repeat restarts the progression identically
        for bar_i in range(cycle_bars):
            slot  = bar_i % len(phrase)
            pat_i = phrase[slot]
            pat   = pats[min(pat_i, len(pats) - 1)]
            abs_bar = rep * cycle_bars + bar_i

            if pat.is_tree_mode():
                # ── Tree path (grouped events merge adjacent same-group leaves) ─
                for leaf, t_onset, dur_s in iter_grouped_events(
                        pat.get_tree(div), _warp, bar_s, abs_bar):
                    hz = stream.next_hz()
                    if hz is None:
                        continue
                    art_mul = _ART_GATE.get(int(leaf.art), 1.0)
                    if leaf.group != 0:
                        # Grouped: duration already merged, use it directly
                        gate_s_i = max(min_dur_s, dur_s)
                    elif art_mul is None:
                        next_t   = _next_onset_t(abs_bar, leaf.node_id, t_onset)
                        gate_s_i = max(min_dur_s, next_t - t_onset)
                    else:
                        gate_s_i = max(min_dur_s, dur_s * pg.rhythm_gate * art_mul)
                    sched.add(NoteEvent(hz, t_onset, gate_s_i, velocity=float(leaf.vel)))
            else:
                # ── Flat (legacy) path ───────────────────────────────────────
                pat.ensure_size(div)
                for step_i in range(div):
                    if not pat.steps[step_i]:
                        continue
                    hz = stream.next_hz()
                    if hz is None:
                        continue
                    t_onset = abs_bar * bar_s + _warp.warp_to_seconds(step_i / div, bar_s)
                    vel     = float(pat.vel[step_i]) if step_i < len(pat.vel) else 1.0
                    art_val = int(pat.art[step_i]) if step_i < len(pat.art) else 0
                    art_mul = _ART_GATE.get(art_val, 1.0)
                    if art_mul is None:
                        next_t   = _next_onset_t(abs_bar, step_i, t_onset)
                        gate_s_i = max(min_dur_s, next_t - t_onset)
                    else:
                        gate_s_i = max(min_dur_s, step_s * pg.rhythm_gate * art_mul)
                    sched.add(NoteEvent(hz, t_onset, gate_s_i, velocity=vel))

    # ── Post-process: apply velocity dynamics (curve + accent grid) ──────────
    dyn_prog = dynamics_program if dynamics_program is not None else getattr(p, "dynamics_program", None)
    if dyn_prog is not None and _HAS_DYN_ENG:
        # Get accent tree from the active rhythm pattern (if available)
        _acc_tree = None
        _act_i = min(pg.rhythm_active_pat, max(0, len(pats) - 1))
        _act_pat = pats[_act_i] if pats else None
        if _act_pat is not None:
            try:
                _acc_tree = _act_pat.get_accent_tree(div)
            except Exception:
                pass
        apply_dynamics(sched.events, dyn_prog, beat_s, div, _rng,
                       beats_per_bar=p.beats_per_bar(),
                       accent_tree=_acc_tree)

    # ── Post-process: apply stochastic ornaments (grace / chirp / echo) ──────
    imp_prog = improv_program if improv_program is not None else getattr(p, "improv_program", None)
    if imp_prog is not None and _HAS_IMPROV_ENG and imp_prog.enabled:
        extra = apply_improv(
            sched.events,
            imp_prog,
            beat_s,
            div,
            pats,
            phrase,
            cycle_bars,
            _rng,
            beats_per_bar=p.beats_per_bar(),
        )
        if extra:
            sched.events.extend(extra)
            sched.events.sort(key=lambda e: e.start_time)

    return sched


def _build_score_schedule_for_voice(
        p: "AnalyticPatch",
        voice: "AnalyticVoice",
        beat_s: float,
        degrees: list,
        deg_pattern: list,
        min_dur_s: float = 1.0 / 48000,
) -> "NoteSchedule":
    """Build the rendered schedule for one voice under the patch's layer mode."""
    layers = p.score_page_items_for_voice(voice)
    if not layers:
        return NoteSchedule()

    if getattr(p, "rhythm_layer_mode", "union") == "specific":
        layer_key, layer_pg = layers[-1]
        sched = _build_rhythm_schedule(
            p, beat_s, degrees, deg_pattern,
            min_dur_s=min_dur_s,
            page=layer_pg,
            dynamics_program=p.dynamics_for(layer_key),
            improv_program=p.improv_for(layer_key),
        )
        for ev in sched.events:
            ev._layer_key = layer_key
        return sched

    merged = NoteSchedule()
    for layer_key, layer_pg in layers:
        layer_sched = _build_rhythm_schedule(
            p, beat_s, degrees, deg_pattern,
            min_dur_s=min_dur_s,
            page=layer_pg,
            dynamics_program=p.dynamics_for(layer_key),
            improv_program=p.improv_for(layer_key),
        )
        if layer_sched.events:
            for ev in layer_sched.events:
                ev._layer_key = layer_key
            merged.events.extend(layer_sched.events)
    merged.events.sort(key=lambda e: e.start_time)
    return merged


def _build_sequence_pitch_context(
        p: "AnalyticPatch",
) -> "tuple[float, list[float], list[int]] | None":
    """Return (beat_s, degrees, pattern) for the patch sequence settings."""
    if not _HAS_SEQ_ENG:
        return None
    source_voices = [v for v in p.voices if not v.muted]
    template = source_voices[0] if source_voices else (p.voices[0] if p.voices else None)
    if template is None:
        return None
    scale = p.seq_scale if p.seq_scale in MODAL_SCALES else "pentatonic_minor"
    beat_s = 60.0 / max(p.seq_bpm, 1.0)
    pattern = _SEQ_PATTERN_PRESETS[
        max(0, min(p.seq_pattern_idx, len(_SEQ_PATTERN_PRESETS) - 1))
    ][1]
    if p.seq_custom_semitones.strip():
        custom_semi = [float(s) for s in p.seq_custom_semitones.split(",") if s.strip()]
        degrees: list[float] = []
        for octave in range(p.seq_octave_span):
            for st in custom_semi:
                degrees.append(semitones_to_hz(p.seq_tonic_hz, st + 12 * octave))
    else:
        base_hz = template.freq_hz if template is not None else p.seq_tonic_hz
        degrees = scale_degrees_hz(base_hz, scale, octave_span=p.seq_octave_span)
    return beat_s, degrees, pattern


def _build_legacy_sequence_schedule(
        p: "AnalyticPatch",
        beat_s: float,
        degrees: list[float],
        pattern: list[int],
) -> "NoteSchedule":
    """Return the non-rhythm schedule path used by preview/export."""
    if p.seq_custom_semitones.strip():
        schedule = NoteSchedule()
        t = 0.0
        for _rep in range(p.seq_repeats):
            for deg_i in pattern:
                hz = degrees[deg_i % len(degrees)]
                note_dur = beat_s * 0.5 * p.seq_legato
                schedule.add(NoteEvent(hz, t, max(note_dur, 1.0 / p.preview_sr)))
                t += beat_s * 0.5
        return schedule

    source_voices = [v for v in p.voices if not v.muted]
    template = source_voices[0] if source_voices else (p.voices[0] if p.voices else None)
    root_hz = template.freq_hz if template is not None else p.seq_tonic_hz
    rule = ArpeggioRule(
        root_hz=root_hz,
        scale=p.seq_scale if p.seq_scale in MODAL_SCALES else "pentatonic_minor",
        pattern=pattern,
        rhythm_beats=[0.5],
        bpm=p.seq_bpm,
        legato_fraction=p.seq_legato,
        octave_span=p.seq_octave_span,
        repeats=p.seq_repeats,
    )
    return rule.generate()


def _sync_resolved_notes(
        p: "AnalyticPatch",
        preserve_locked: bool = True,
) -> list[ResolvedNote]:
    """Refresh patch.resolved_notes from the current union solve."""
    ctx = _build_sequence_pitch_context(p)
    if ctx is None:
        return p.resolved_notes
    beat_s, degrees, pattern = ctx
    locked_map: dict[str, ResolvedNote] = {}
    locked_notes: list[ResolvedNote] = []
    if preserve_locked:
        locked_notes = [n for n in p.resolved_notes if getattr(n, "locked", False)]
        locked_map = {n.note_id: n for n in locked_notes}

    def _masked_by_locked(candidate: ResolvedNote) -> bool:
        for locked in locked_notes:
            if locked.voice_key != candidate.voice_key:
                continue
            a0 = candidate.start_time
            a1 = candidate.start_time + candidate.duration_s
            b0 = locked.start_time
            b1 = locked.start_time + locked.duration_s
            if max(a0, b0) < min(a1, b1):
                return True
        return False

    notes: list[ResolvedNote] = []
    source_voices = [v for v in p.voices if not v.muted]
    for voice in source_voices:
        if p.rhythm_enabled:
            sched = _build_score_schedule_for_voice(
                p, voice, beat_s, degrees, pattern, min_dur_s=1.0 / p.preview_sr)
        else:
            sched = _build_legacy_sequence_schedule(p, beat_s, degrees, pattern)
        for i, ev in enumerate(sched.events):
            layer_key = str(getattr(ev, "_layer_key", "legacy"))
            note_id = (
                f"{voice.key}:{layer_key}:"
                f"{round(float(ev.start_time), 6)}:"
                f"{round(float(ev.duration_s), 6)}:"
                f"{round(float(_resolved_event_hz(p, voice, ev.fundamental_hz)), 4)}"
            )
            generated = ResolvedNote(
                note_id=note_id,
                voice_key=voice.key,
                voice_label=getattr(voice, "label", voice.key[:6]),
                layer_key=layer_key,
                start_time=float(ev.start_time),
                duration_s=float(ev.duration_s),
                fundamental_hz=float(_resolved_event_hz(p, voice, ev.fundamental_hz)),
                velocity=float(ev.velocity),
                locked=False,
            )
            if note_id in locked_map:
                keep = locked_map.pop(note_id)
                keep.voice_key = generated.voice_key
                keep.voice_label = generated.voice_label
                keep.layer_key = generated.layer_key
                keep.velocity = generated.velocity
                notes.append(keep)
            elif preserve_locked and _masked_by_locked(generated):
                continue
            else:
                notes.append(generated)
    if preserve_locked and locked_notes:
        locked_ids = {n.note_id for n in notes}
        for locked in locked_notes:
            if locked.note_id not in locked_ids:
                notes.append(locked)
    notes.sort(key=lambda n: (n.start_time, n.fundamental_hz, n.voice_key, n.layer_key))
    p.resolved_notes = notes
    return notes

_REGISTER_BAND_ORDER = {"bass": 0, "mid": 1, "high": 2, "all": 3}
_REGISTER_ROW_RADIUS = {"bass": 4.3, "mid": 6.0, "high": 7.8, "all": 5.2}

# ── Stage layout: dome-backed concert stage with realistic orchestral seating ──
#
# Real orchestral layout principles encoded here:
#   • Strings (signal / body voices) occupy the front arcs in register order:
#       - High melody = 1st violins (front stage-left)
#       - High non-melody = 2nd violins (front stage-right)
#       - Mid signal/body = violas (center-left, 2nd row) or cellos (center-right)
#       - Bass signal/body = cellos (front-right) or double basses (far right, risers)
#   • Woodwinds (air voices, or mid/high non-body non-transient alternates):
#       center rear, paired in two rows behind strings
#   • Brass (body or signal voices in bass/mid with stab or root roles):
#       right rear on risers, behind woodwinds
#   • Percussion (transient voices, or stab role in any register):
#       rear center-to-left, highest risers
#   • "all" register parts default to center stage (viola territory)
#   • Body voices sit behind their timbral kin (same section, deeper row)
#   • Air voices go to woodwind section regardless of register
#   • First chairs (slot_index 0) get inside seats closest to conductor
#
# Section format: (center_angle_deg, base_radius_m, platform_z_m, arc_span_per_chair_deg)
# Angle: 0° = front-center facing audience, +deg = stage-right (audience-left),
#         −deg = stage-left (audience-right)
# Radius: distance from conductor position; deeper rows = larger radius
# Platform z: height above stage floor (risers for rear sections)
# Arc span: degrees consumed per chair for natural spacing

_STAGE_SECTIONS: dict[str, tuple[float, float, float, float]] = {
    # ── Front row: strings ────────────────────────────────────────────────
    "violin_1":       (-25.0,  3.8, 1.05, 6.0),   # front stage-left arc
    "violin_2":       (+18.0,  3.8, 1.05, 6.0),   # front stage-right arc
    # ── Second row: mid strings ──────────────────────────────────────────
    "viola":          (-12.0,  5.2, 1.08, 7.0),   # center-left, slightly deeper
    "cello":          (+26.0,  5.0, 1.08, 7.0),   # center-right
    # ── Third row: low strings, high riser ──────────────────────────────
    "bass_str":       (+52.0,  6.8, 1.22, 9.0),   # far right, standing riser
    # ── Fourth row: woodwinds center ─────────────────────────────────────
    "woodwind_1":     ( -5.0,  6.6, 1.18, 8.0),   # center-left rear (flutes/oboes)
    "woodwind_2":     ( +8.0,  7.4, 1.24, 8.0),   # center-right, deeper (clarinets/bassoons)
    # ── Fifth row: brass ─────────────────────────────────────────────────
    "brass_1":        (+30.0,  8.2, 1.35, 10.0),  # right rear (horns)
    "brass_2":        (+48.0,  8.8, 1.42, 10.0),  # far right rear (trumpets/trombones)
    # ── Sixth row: percussion ────────────────────────────────────────────
    "percussion_1":   (-38.0,  9.2, 1.52, 12.0),  # left rear, high riser (timpani, mallet)
    "percussion_2":   (  0.0,  9.6, 1.58, 12.0),  # center rear, highest (cymbals, misc)
    # ── Specialty: soloist / harp / keyboard / body-resonance ────────────
    "soloist":        (  0.0,  2.6, 1.05, 8.0),   # center-front, beside conductor
    "harp":           (-55.0,  5.8, 1.10, 10.0),  # far stage-left, behind 1st violins
    "body_front":     ( -8.0,  4.6, 1.06, 7.0),   # body voices behind front strings
    "body_rear":      (+15.0,  7.0, 1.20, 8.0),   # body voices behind mid sections
}

def _stage_section_for_part(register: str, seq_role: str, slot_index: int,
                            voice_role: str = "signal") -> str:
    """Map a Part's classification to an orchestral stage section.

    Uses register, seq_role, voice_role, and slot_index (for alternation within
    the same classification) to produce realistic orchestral seating.
    """
    # ── Transient: soloists (melody/root) go center-front near conductor,
    #    stab/bass transients go to percussion rear ────────────────────────
    if voice_role == "transient":
        if seq_role in ("melody", "root"):
            return "soloist"            # featured performer, center-front
        return "percussion_1" if slot_index % 2 == 0 else "percussion_2"

    # ── Air voices → woodwinds regardless of register ────────────────────
    if voice_role == "air":
        return "woodwind_1" if slot_index % 2 == 0 else "woodwind_2"

    # ── Body voices → behind their timbral kin ───────────────────────────
    if voice_role == "body":
        if register == "bass":
            return "body_rear"      # behind cellos/basses
        return "body_front"         # behind front-row strings

    # ── Stab (plosive/accent) placement ──────────────────────────────────
    if seq_role == "stab":
        if register == "bass":
            return "percussion_1"   # bass stabs center-back (timpani territory)
        if register == "high":
            return "brass_2"        # high stabs out wide (trumpet snaps)
        return "brass_1"            # mid stabs with horns

    # ── High register: strings front, woodwinds behind ───────────────────
    if register == "high":
        if seq_role in ("melody", ""):
            return "violin_1" if slot_index % 2 == 0 else "violin_2"
        if seq_role == "root":
            return "violin_2" if slot_index % 2 == 0 else "woodwind_1"
        return "woodwind_1" if slot_index % 2 == 0 else "woodwind_2"

    # ── Mid register: violas / cellos / woodwinds ────────────────────────
    if register == "mid":
        if seq_role == "melody":
            return "viola" if slot_index % 2 == 0 else "cello"
        if seq_role == "root":
            return "cello" if slot_index % 2 == 0 else "woodwind_2"
        if seq_role == "bass":
            return "cello"
        return "viola" if slot_index % 2 == 0 else "woodwind_1"

    # ── Bass register: cellos / basses / brass ───────────────────────────
    if register == "bass":
        if seq_role == "melody":
            return "cello" if slot_index % 2 == 0 else "bass_str"
        if seq_role == "bass":
            return "bass_str" if slot_index % 2 == 0 else "cello"
        if seq_role == "root":
            return "brass_1" if slot_index % 2 == 0 else "bass_str"
        return "cello" if slot_index % 2 == 0 else "brass_1"

    # ── "all" register → center stage (viola/cello territory) ────────────
    return "viola" if slot_index % 2 == 0 else "cello"


def _ordinal_label(index: int) -> str:
    if 10 <= (index % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(index % 10, "th")
    return f"{index}{suffix}"


def _page_specificity_score(layer_keys: list[str]) -> int:
    max_tokens = 0
    total_tokens = 0
    for key in layer_keys:
        toks = [t.strip() for t in str(key).split("+") if t.strip() and t.strip() != "all"]
        total_tokens += len(toks)
        max_tokens = max(max_tokens, len(toks))
    return max_tokens * 100 + total_tokens


def _stable_rng(seed: str) -> _random.Random:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]
    return _random.Random(int(digest, 16))


def _rotate_list(values: list[str], offset: int) -> list[str]:
    if not values:
        return []
    offset %= len(values)
    return list(values[offset:]) + list(values[:offset])


def _voice_polyphony_capacity(voice: "AnalyticVoice") -> int:
    return max(1, int(getattr(voice, "polyphony_count", 1)))


def _voice_polyphony_mode(voice: "AnalyticVoice") -> str:
    mode = str(getattr(voice, "polyphony_mode", "sympathetic"))
    return mode if mode in ("sympathetic", "unsympathetic") else "sympathetic"


def _performer_phase_mode(patch: "AnalyticPatch") -> str:
    mode = str(getattr(patch, "performer_phase_mode", "coherent"))
    return mode if mode in ("coherent", "individual") else "coherent"


def _group_event_note_key(group_key: str, voice_key: str, event: object, layer_key: str) -> str:
    start = float(getattr(event, "start_time", 0.0))
    dur = float(getattr(event, "duration_s", 0.0))
    hz = float(getattr(event, "fundamental_hz", 0.0))
    return (
        f"{group_key}:{voice_key}:{layer_key}:"
        f"{round(start, 6)}:{round(dur, 6)}:{round(hz, 4)}"
    )


def _note_demands_for_group(group_key: str, events: list, voices: list["AnalyticVoice"]) -> list[dict]:
    demands: list[dict] = []
    for voice in voices:
        for event in events:
            layer_key = str(getattr(event, "_layer_key", "all"))
            demands.append({
                "note_key": _group_event_note_key(group_key, getattr(voice, "key", ""), event, layer_key),
                "voice_key": getattr(voice, "key", ""),
                "layer_key": layer_key,
                "start_time": float(getattr(event, "start_time", 0.0)),
                "end_time": float(getattr(event, "start_time", 0.0)) + float(getattr(event, "duration_s", 0.0)),
                "duration_s": float(getattr(event, "duration_s", 0.0)),
                "fundamental_hz": float(getattr(event, "fundamental_hz", 0.0)),
            })
    demands.sort(key=lambda d: (d["start_time"], -(d["end_time"] - d["start_time"]), d["voice_key"], d["note_key"]))
    return demands


def _pack_note_demands_by_voice(demands: list[dict], voices: list["AnalyticVoice"]) -> tuple[list[dict], dict]:
    voice_map = {getattr(v, "key", ""): v for v in voices}
    bins: list[dict] = []
    summary = {
        "required_performers": 0,
        "sympathetic_bins": 0,
        "unsympathetic_bins": 0,
    }
    for voice in voices:
        voice_key = getattr(voice, "key", "")
        voice_demands = [d for d in demands if d["voice_key"] == voice_key]
        if not voice_demands:
            continue
        mode = _voice_polyphony_mode(voice)
        capacity = _voice_polyphony_capacity(voice) if mode == "sympathetic" else 1
        voice_bins: list[dict] = []
        for demand in voice_demands:
            placed = False
            for bin_info in voice_bins:
                active_ends = [et for et in bin_info["active_ends"] if et > demand["start_time"]]
                bin_info["active_ends"] = active_ends
                if len(active_ends) < capacity:
                    bin_info["active_ends"].append(demand["end_time"])
                    bin_info["note_demands"].append(demand)
                    placed = True
                    break
            if not placed:
                voice_bins.append({
                    "voice_key": voice_key,
                    "polyphony_mode": mode,
                    "capacity": capacity,
                    "active_ends": [demand["end_time"]],
                    "note_demands": [demand],
                })
        summary["required_performers"] += len(voice_bins)
        if mode == "sympathetic":
            summary["sympathetic_bins"] += len(voice_bins)
        else:
            summary["unsympathetic_bins"] += len(voice_bins)
        bins.extend(voice_bins)
    return bins, summary


def _required_chairs_for_overlap(active_events: int, voices: list["AnalyticVoice"]) -> tuple[int, dict]:
    active_events = max(0, int(active_events))
    if active_events <= 0 or not voices:
        return 0, {
            "sympathetic_sources": 0,
            "unsympathetic_sources": 0,
            "sympathetic_capacity": 0,
            "unsympathetic_capacity": 0,
            "interference_units": 0.0,
        }

    sympathetic = [v for v in voices if _voice_polyphony_mode(v) == "sympathetic"]
    unsympathetic = [v for v in voices if _voice_polyphony_mode(v) == "unsympathetic"]

    sympathetic_sources = active_events * len(sympathetic)
    sympathetic_capacity = sum(_voice_polyphony_capacity(v) for v in sympathetic)
    sympathetic_chairs = (
        math.ceil(sympathetic_sources / max(1, sympathetic_capacity))
        if sympathetic_sources > 0 else 0
    )

    unsympathetic_sources = active_events * len(unsympathetic)
    unsympathetic_capacity = sum(_voice_polyphony_capacity(v) for v in unsympathetic)
    unsympathetic_chairs = sum(
        math.ceil(active_events / _voice_polyphony_capacity(v))
        for v in unsympathetic
    )

    interference_units = float(unsympathetic_sources) + (
        float(sympathetic_sources) / max(1.0, float(sympathetic_capacity))
    )
    return sympathetic_chairs + unsympathetic_chairs, {
        "sympathetic_sources": sympathetic_sources,
        "unsympathetic_sources": unsympathetic_sources,
        "sympathetic_capacity": sympathetic_capacity,
        "unsympathetic_capacity": unsympathetic_capacity,
        "interference_units": interference_units,
    }


def _schedule_overlap_metrics(events: list, voices: list["AnalyticVoice"]) -> dict:
    if not events:
        return {
            "peak_active_events": 0,
            "peak_active_sources": 0,
            "required_chairs": max(1, len(voices)) if voices else 1,
            "interference_peak": 0.0,
        }

    times = sorted({
        float(getattr(ev, "start_time", 0.0))
        for ev in events
    } | {
        float(getattr(ev, "start_time", 0.0)) + float(getattr(ev, "duration_s", 0.0))
        for ev in events
    })
    if len(times) < 2:
        required, detail = _required_chairs_for_overlap(len(events), voices)
        return {
            "peak_active_events": len(events),
            "peak_active_sources": len(events) * len(voices),
            "required_chairs": max(1, required),
            "interference_peak": float(detail["interference_units"]),
        }

    peak_active_events = 0
    peak_active_sources = 0
    peak_required_chairs = 1
    interference_peak = 0.0
    for t0, t1 in zip(times[:-1], times[1:]):
        if t1 <= t0:
            continue
        mid_t = 0.5 * (t0 + t1)
        active_events = [
            ev for ev in events
            if float(getattr(ev, "start_time", 0.0)) <= mid_t <
            float(getattr(ev, "start_time", 0.0)) + float(getattr(ev, "duration_s", 0.0))
        ]
        if not active_events:
            continue
        active_count = len(active_events)
        required, detail = _required_chairs_for_overlap(active_count, voices)
        peak_active_events = max(peak_active_events, active_count)
        peak_active_sources = max(peak_active_sources, active_count * len(voices))
        peak_required_chairs = max(peak_required_chairs, required)
        interference_peak = max(interference_peak, float(detail["interference_units"]))

    return {
        "peak_active_events": peak_active_events,
        "peak_active_sources": peak_active_sources,
        "required_chairs": max(1, peak_required_chairs),
        "interference_peak": interference_peak,
    }


def _refresh_part_placement_layout(patch: "AnalyticPatch") -> None:
    metrics = getattr(patch, "_arrangement_metrics", {}) or {}
    groups = {
        str(gm.get("group_key", "")): gm
        for gm in metrics.get("groups", [])
        if isinstance(gm, dict) and gm.get("group_key")
    }
    voice_map = {v.key: v for v in getattr(patch, "voices", [])}

    patch.parts.sort(key=lambda pt: (
        _REGISTER_BAND_ORDER.get(pt.register, 99),
        -int(pt.solver_hints.get("page_specificity", 0)),
        -int(pt.solver_hints.get("stack_depth", 0)),
        pt.label,
    ))

    _res_cfg = getattr(patch, "placement_resonator", None)
    _layout_mode = str(getattr(_res_cfg, "layout_mode", "auto") or "auto")

    by_register: dict[str, list[Part]] = {}
    for pt in patch.parts:
        by_register.setdefault(pt.register, []).append(pt)

    for register, reg_parts in by_register.items():
        slot_count = len(reg_parts)
        for slot_index, pt in enumerate(reg_parts):
            gm = groups.get(pt.key, {})
            voice_keys = list(pt.voice_keys)
            layer_keys = list(gm.get("layer_keys", pt.solver_hints.get("layer_keys", [])))
            packed_performers = list(gm.get("packed_performers", []))
            required_chairs = max(1, len(packed_performers) or int(pt.solver_hints.get("required_chairs", 1)))
            total_players = max(required_chairs, int(pt.player_count))
            pt.solver_hints["required_chairs"] = required_chairs
            pt.solver_hints["section_slot"] = slot_index
            pt.solver_hints["section_slot_count"] = slot_count
            pt.solver_hints["layer_keys"] = list(layer_keys)

            # ── Section geometry: stage rows vs. register semicircle ──────────
            if _layout_mode == "stage":
                section_key = _stage_section_for_part(register, pt.seq_role, slot_index,
                                                     voice_role=pt.voice_role)
                sec_angle, sec_radius, sec_z, sec_arc = _STAGE_SECTIONS.get(
                    section_key, (0.0, 5.0, 1.1, 8.0))
                part_span_deg = max(sec_arc, min(sec_arc * required_chairs, 60.0))
                section_center_angle = sec_angle
                radius = sec_radius
                pt.solver_hints["section_shape"] = "semicircle"
                pt.solver_hints["stage_section"] = section_key
                pt.solver_hints["stage_z"] = sec_z
            else:
                sec_z = 1.1
                pt.solver_hints["section_shape"] = "line" if register == "all" else "semicircle"
                section_center = slot_index - 0.5 * (slot_count - 1)
                radius = _REGISTER_ROW_RADIUS.get(register, _REGISTER_ROW_RADIUS["all"])
                section_center_angle = section_center * 24.0
                part_span_deg = max(12.0, min(70.0, 10.0 * required_chairs))
                if slot_count == 1:
                    part_span_deg = min(80.0, part_span_deg + 8.0)

            chair_base = total_players // required_chairs
            chair_extra = total_players % required_chairs
            chairs: list[Chair] = []
            for chair_idx in range(required_chairs):
                chair_num = chair_idx + 1
                performer_count = chair_base + (1 if chair_idx < chair_extra else 0)
                packed = (
                    packed_performers[chair_idx]
                    if chair_idx < len(packed_performers)
                    else {}
                )
                source_voice_keys = list(packed.get("source_voice_keys", [])) or _rotate_list(voice_keys, chair_idx)
                source_layer_keys = list(packed.get("source_layer_keys", [])) or _rotate_list(layer_keys, chair_idx)
                body_types = [
                    str(getattr(voice_map.get(vk), "body_type", "direct") or "direct")
                    for vk in source_voice_keys
                    if vk
                ]
                chair_body_type = body_types[0] if body_types and len(set(body_types)) == 1 else "direct"
                if pt.solver_hints["section_shape"] == "line":
                    section_center_l = slot_index - 0.5 * (slot_count - 1)
                    if required_chairs == 1:
                        chair_x = section_center_l * 1.6
                    else:
                        chair_x = section_center_l * 1.6 + (
                            (chair_idx / (required_chairs - 1)) - 0.5
                        ) * 2.2
                    chair_y = radius
                    chair_angle_deg = 0.0
                else:
                    chair_angle_deg = (
                        section_center_angle if required_chairs == 1 else
                        section_center_angle + ((chair_idx / (required_chairs - 1)) - 0.5) * part_span_deg
                    )
                    ang = math.radians(chair_angle_deg)
                    chair_x = radius * math.sin(ang)
                    chair_y = radius * math.cos(ang)

                chair = Chair(
                    key=f"{pt.key}:chair:{chair_num}",
                    label=f"{_ordinal_label(chair_num)} chair",
                    part_key=pt.key,
                    chair_index=chair_num,
                    specificity_rank=int(pt.solver_hints.get("page_specificity", 0)),
                    source_voice_keys=source_voice_keys,
                    source_layer_keys=source_layer_keys,
                    performer_count=max(1, performer_count),
                    solver_hints={
                        "x": chair_x,
                        "y": chair_y,
                        "radius": radius,
                        "angle_deg": chair_angle_deg,
                        "section_center_angle": section_center_angle,
                    },
                )

                performers: list[PerformerPlacement] = []
                perf_center = 0.5 * (chair.performer_count - 1)
                for performer_idx in range(chair.performer_count):
                    perf_num = performer_idx + 1
                    rng = _stable_rng(f"{pt.key}:{chair.key}:{perf_num}")
                    primary_voice_key = (
                        (
                            list(packed.get("source_voice_keys", []))
                            or source_voice_keys
                        )[performer_idx % max(1, len(list(packed.get("source_voice_keys", [])) or source_voice_keys))]
                        if source_voice_keys else ""
                    )
                    primary_layer_key = (
                        (
                            list(packed.get("source_layer_keys", []))
                            or source_layer_keys
                        )[performer_idx % max(1, len(list(packed.get("source_layer_keys", [])) or source_layer_keys))]
                        if source_layer_keys else ""
                    )
                    lateral = (performer_idx - perf_center) * 0.22
                    depth = rng.uniform(-0.10, 0.10) + 0.06 * ((chair_idx % 2) - 0.5)
                    if pt.solver_hints["section_shape"] == "line":
                        perf_x = chair_x + lateral
                        perf_y = chair_y + depth
                        angle_deg = 0.0
                    else:
                        ang = math.radians(chair_angle_deg)
                        tangent_x = math.cos(ang)
                        tangent_y = -math.sin(ang)
                        radial_x = math.sin(ang)
                        radial_y = math.cos(ang)
                        perf_x = chair_x + tangent_x * lateral + radial_x * depth
                        perf_y = chair_y + tangent_y * lateral + radial_y * depth
                        angle_deg = chair_angle_deg + lateral * 6.0
                    distance = math.sqrt(perf_x * perf_x + perf_y * perf_y)
                    geometric_delay_ms = (distance * 0.85 / 343.0) * 1000.0
                    humanization_ms = rng.uniform(-4.0, 4.0) * (
                        1.0 + min(2.0, 0.01 * float(chair.specificity_rank))
                    )
                    gain_db = rng.uniform(-1.2, 1.2)
                    pan = max(-1.0, min(1.0, perf_x / 9.0))
                    phase_offset_rad = (
                        0.0 if _performer_phase_mode(patch) == "coherent"
                        else rng.uniform(0.0, 2.0 * math.pi)
                    )
                    # Aperture normal: performer faces the conductor at origin
                    _fdx, _fdy = -perf_x, -perf_y
                    _fdn = math.sqrt(_fdx * _fdx + _fdy * _fdy) or 1.0
                    _face_x, _face_y = _fdx / _fdn, _fdy / _fdn

                    performers.append(PerformerPlacement(
                        key=f"{chair.key}:performer:{perf_num}",
                        label=f"{chair.label} performer {perf_num}",
                        chair_key=chair.key,
                        chair_index=chair_num,
                        performer_index=perf_num,
                        source_voice_keys=[primary_voice_key] if primary_voice_key else [],
                        source_layer_keys=[primary_layer_key] if primary_layer_key else [],
                        assigned_note_keys=list(packed.get("assigned_note_keys", [])),
                        body_type=str(getattr(voice_map.get(primary_voice_key), "body_type", chair_body_type) or chair_body_type),
                        x=perf_x,
                        y=perf_y,
                        z=sec_z,
                        face_x=_face_x,
                        face_y=_face_y,
                        face_z=0.0,
                        radius=distance,
                        angle_deg=angle_deg,
                        geometric_delay_ms=geometric_delay_ms,
                        humanization_ms=humanization_ms,
                        phase_offset_rad=phase_offset_rad,
                        gain_db=gain_db,
                        pan=pan,
                    ))
                chair.performers = performers
                chairs.append(chair)
            pt.chairs = chairs

    # ── Auto-deploy resonator module when placement_resonator is enabled ──────
    _res_cfg2 = getattr(patch, "placement_resonator", None)
    if _res_cfg2 is not None and getattr(_res_cfg2, "enabled", False) and getattr(patch, "parts", []):
        _dep_key = str(getattr(_res_cfg2, "deployed_module_key", "") or "")
        _dep_mod = None
        if _dep_key:
            _dep_mod = next((m for m in getattr(patch, "modules", []) if m.key == _dep_key), None)
        if _dep_mod is None:
            _dep_mod = next(
                (m for m in getattr(patch, "modules", [])
                 if m.module_type == "state_machine"
                 and str(getattr(m, "sm_plugin", "")) == "orchestral_resonance"),
                None,
            )
        if _dep_mod is None:
            _dep_mod = AnalyticModule(
                key=f"placement_res_{uuid.uuid4().hex[:6]}",
                label="Orchestral Resonance",
                module_type="state_machine",
            )
            patch.modules.append(_dep_mod)
        _sync_placement_resonator_module(patch, _dep_mod)


def _performer_map_for_patch(patch: "AnalyticPatch") -> dict[str, list[PerformerPlacement]]:
    mapping: dict[str, list[PerformerPlacement]] = {}
    for pt in getattr(patch, "parts", []):
        for ch in getattr(pt, "chairs", []):
            for pf in getattr(ch, "performers", []):
                for vk in getattr(pf, "source_voice_keys", []):
                    if vk:
                        mapping.setdefault(vk, []).append(pf)
    return mapping


def _resolve_note_target(voices: list, patch: "AnalyticPatch") -> "NoteTarget":
    """Return the most holistic NoteTarget available for *voices* in *patch*.

    Dispatch hierarchy (most → least coordinated):
      1. Performers  — PerformerPlacement entries exist in chairs → apply
                       per-seat geometric delay, phase, gain, and pan.
      2. Chairs      — Chair sections exist but no performers yet → instrument-
                       level grouping without individual spatial transforms.
      3. Voice       — No Parts/Chairs resolved → synthesis-only direct path.

    The returned target always carries the resolved voice list so downstream
    synthesis is uniform regardless of which level was matched.
    """
    voice_keys = frozenset(getattr(v, "key", "") for v in voices) - {""}

    best_performers: list = []
    best_chairs: list = []
    best_part: object = None

    for pt in getattr(patch, "parts", []):
        pt_voice_keys = frozenset(vk for vk in getattr(pt, "voice_keys", []) if vk)
        if not pt_voice_keys.intersection(voice_keys):
            continue
        chairs = getattr(pt, "chairs", [])
        if not chairs:
            continue
        # Collect performers across all chairs that reference at least one of our voices
        local_performers: list = []
        local_chairs: list = []
        for ch in chairs:
            ch_voices = frozenset(vk for vk in getattr(ch, "source_voice_keys", []) if vk)
            if not ch_voices and not getattr(ch, "performers", []):
                # Chair has no voice filter — treat as matching all part voices
                ch_voices = pt_voice_keys
            if ch_voices.intersection(voice_keys) or not ch_voices:
                local_chairs.append(ch)
                local_performers.extend(getattr(ch, "performers", []))
        if local_performers:
            best_performers = local_performers
            best_chairs = local_chairs
            best_part = pt
            break  # first matching part with performers wins
        if local_chairs and best_part is None:
            best_chairs = local_chairs
            best_part = pt

    if best_performers:
        return NoteTarget(
            target_type="performer",
            voices=list(voices),
            performers=best_performers,
            chairs=best_chairs,
            part=best_part,
        )
    if best_chairs:
        return NoteTarget(
            target_type="chair",
            voices=list(voices),
            performers=[],
            chairs=best_chairs,
            part=best_part,
        )
    return NoteTarget(
        target_type="voice",
        voices=list(voices),
        performers=[],
        chairs=[],
        part=None,
    )


def _apply_performer_transforms_to_src(
    performer_parent_map: "dict[str, list[PerformerPlacement]]",
    voice_sigs: "dict[str, np.ndarray]",
    Src: "np.ndarray",
    ki: "dict[str, int]",
    n_ext: int,
    sr: int,
) -> None:
    """Inject performer-transformed voice signals into *Src* in-place.

    For every voice that has PerformerPlacement entries (those excluded from the
    normal ``Src`` population), synthesize the ensemble contribution:

      1. Take the raw synthesised voice signal.
      2. For each PerformerPlacement:
           a. Apply geometric + humanization delay (integer-sample circular shift,
              zeroing the pre-roll region so causality is preserved).
           b. Apply phase offset (complex rotation of the analytic signal).
           c. Apply gain_db (amplitude scale).
      3. Sum performer copies and average by performer count (preserves loudness
         regardless of section size).
      4. Write the result into ``Src[ki[voice_key]]``.

    This implements the "dispatch to performers" step: the solved score (NoteSchedule)
    was handed to the most holistic available target (PerformerPlacement).  When no
    performers exist, this function is a no-op and voices reach Src via the normal
    un-transformed path.
    """
    for vk, placements in performer_parent_map.items():
        if vk not in ki or vk not in voice_sigs:
            continue
        raw = np.asarray(voice_sigs[vk], dtype=np.complex128)
        if len(raw) < n_ext:
            raw = np.pad(raw, (0, n_ext - len(raw)))
        else:
            raw = raw[:n_ext]

        acc = np.zeros(n_ext, dtype=np.complex128)
        for pf in placements:
            delay_s = (float(getattr(pf, "geometric_delay_ms", 0.0))
                       + float(getattr(pf, "humanization_ms", 0.0))) * 1e-3
            delay_n = int(round(delay_s * sr))
            sig = raw.copy()
            if delay_n > 0:
                sig = np.roll(sig, delay_n)
                sig[:delay_n] = 0.0
            phase = float(getattr(pf, "phase_offset_rad", 0.0))
            if phase:
                sig = sig * complex(math.cos(phase), math.sin(phase))
            gain_db = float(getattr(pf, "gain_db", 0.0))
            if gain_db:
                sig = sig * (10.0 ** (gain_db / 20.0))
            acc += sig

        n_pl = len(placements)
        if n_pl > 1:
            acc /= n_pl
        Src[ki[vk]] = acc


def _placement_resonator_item_count(patch: "AnalyticPatch") -> int:
    performer_total = sum(
        len(getattr(ch, "performers", []))
        for pt in getattr(patch, "parts", [])
        for ch in getattr(pt, "chairs", [])
    )
    if performer_total > 0:
        return performer_total
    voice_total = len(getattr(patch, "voices", []))
    return max(1, voice_total)


def _placement_body_types_for_patch(patch: "AnalyticPatch") -> list[str]:
    types: list[str] = []
    for voice in getattr(patch, "voices", []):
        body_type = str(getattr(voice, "body_type", "direct") or "direct")
        if body_type not in types:
            types.append(body_type)
    return types or ["direct"]


def _placement_performer_geometry_json(patch: "AnalyticPatch") -> str:
    """Serialize all performer positions/directions to JSON for the resonance plugin."""
    import json as _json
    entries = []
    for pt in getattr(patch, "parts", []):
        for ch in getattr(pt, "chairs", []):
            for pf in getattr(ch, "performers", []):
                px, py, pz = float(getattr(pf, "x", 0.0)), float(getattr(pf, "y", 0.0)), float(getattr(pf, "z", 1.1))
                # Performer faces toward front-center (0, 0, pz): direction = normalize(-px, -py, 0)
                # In room coords performers face downstage (toward receiver/audience).
                dx, dy = -px, -py
                dn = math.sqrt(dx*dx + dy*dy) or 1.0
                entries.append({
                    "key": str(getattr(pf, "key", "")),
                    "x": px,
                    "y": py,
                    "z": pz,
                    "dir_x": round(float(getattr(pf, "face_x", dx / dn)), 4),
                    "dir_y": round(float(getattr(pf, "face_y", dy / dn)), 4),
                    "dir_z": round(float(getattr(pf, "face_z", 0.0)), 4),
                    "body_type": str(getattr(pf, "body_type", "direct")),
                    "part_key": str(getattr(pt, "key", "")),
                    "register": str(getattr(pt, "register", "")),
                })
    if not entries:
        return ""
    return _json.dumps(entries, separators=(",", ":"))


def _placement_resonator_module_params(patch: "AnalyticPatch") -> dict[str, object]:
    cfg = getattr(patch, "placement_resonator", PlacementResonatorConfig())
    params = {
        "room_shape": cfg.room_shape,
        "scene_path": str(cfg.scene_path),
        "room_radius": float(cfg.room_radius),
        "room_height": float(cfg.room_height),
        "feedback_iterations": int(cfg.feedback_iterations),
        "feedback_gain": float(cfg.feedback_gain),
        "passive_loss": float(cfg.passive_loss),
        "band_split_mode": cfg.band_split_mode,
        "fir_taps": int(cfg.fir_taps),
        "high_cone_deg": float(cfg.high_cone_deg),
        "diffuse_strength": float(cfg.diffuse_strength),
        "air_db_per_m": float(cfg.air_db_per_m),
        "air_highband_db_per_m": float(cfg.air_highband_db_per_m),
        "temperature_c": float(cfg.temperature_c),
        "humidity_rel": float(cfg.humidity_rel),
        "placement_owner": str(cfg.owner_module_type),
        "placement_body_types": ",".join(_placement_body_types_for_patch(patch)),
        "placement_item_count": int(_placement_resonator_item_count(patch)),
        # Receiver array preset
        "receiver_array_key": str(cfg.receiver_array_key),
        "receiver_pos_x": float(cfg.receiver_pos_x),
        "receiver_pos_y": float(cfg.receiver_pos_y),
        "receiver_pos_z": float(cfg.receiver_pos_z),
        "receiver_fwd_x": float(cfg.receiver_fwd_x),
        "receiver_fwd_y": float(cfg.receiver_fwd_y),
        "receiver_fwd_z": float(cfg.receiver_fwd_z),
        # Placement geometry for source positions
        "performer_geometry_json": _placement_performer_geometry_json(patch),
    }
    return params


def _sync_placement_resonator_module(patch: "AnalyticPatch", module: "AnalyticModule") -> None:
    cfg = getattr(patch, "placement_resonator", PlacementResonatorConfig())
    module.module_type = "state_machine"
    module.label = module.label or "Orchestral Resonance"
    module.sm_plugin = "orchestral_resonance"
    module.sm_n_items = _placement_resonator_item_count(patch)
    plug = _load_sm_plugin(module.sm_plugin)
    if plug is not None:
        module.sm_vars = _sm_plugin_output_vars(plug)
        module.sm_state_vars = _sm_plugin_state_vars(plug)
        module.sm_items = _sm_plugin_item_names(plug, module.sm_n_items)
        defaults = _sm_plugin_default_params(plug)
    else:
        defaults = {}
    module.sm_params = {
        **defaults,
        **dict(getattr(module, "sm_params", {}) or {}),
        **_placement_resonator_module_params(patch),
    }
    module.sm_use_torch = True
    module._sm_state = {}
    module._sm_out_cache = {}
    module._sm_aux_state = {}
    if not cfg.deployed_module_key:
        cfg.deployed_module_key = module.key


def _make_note_temp_patch(
    parent: "AnalyticPatch",
    note_voices: "list[AnalyticVoice]",
    duration_s: float,
    event_hz: float,
    note_keys: "list[str]",
    group_voice_keys: "list[str]",
    *,
    shared_modules: "list[AnalyticModule] | None" = None,
) -> "AnalyticPatch":
    """Build a lightweight per-note patch that shares read-only structures by reference.

    Only the module *state* needs isolation: ``_sm_state``, ``_sm_out_cache``,
    ``_sm_aux_state``, and ``_sm_log_text`` are the only fields that
    ``_synthesize_patch`` mutates on a module.  Everything else (routing, LFOs,
    controls, mixers, param_nodes, system_audio) is read-only during synthesis
    and can be shared safely.

    When *shared_modules* is provided those module objects are used directly
    (their mutable state slots are snapshotted/restored by the caller).
    Otherwise, fall back to a shallow copy with fresh state dicts.
    """
    tp = AnalyticPatch()
    tp.duration               = duration_s
    tp.preview_sr             = parent.preview_sr
    tp.voices                 = note_voices
    # Read-only — share by reference
    tp.lfos                   = parent.lfos
    tp.controls               = parent.controls
    tp.routing                = parent.routing
    tp.mixers                 = parent.mixers
    tp.param_nodes            = parent.param_nodes
    tp.system_audio           = parent.system_audio
    tp.tuning                 = parent.tuning
    tp.projection_mode        = parent.projection_mode
    tp.projection_rotation_hz = parent.projection_rotation_hz
    tp.normalize_output       = False
    tp.performer_phase_mode   = parent.performer_phase_mode
    tp.seq_tonic_hz           = parent.seq_tonic_hz
    tp._seq_note_hz           = float(event_hz)
    # Modules: shallow-copy list, reset mutable state slots so notes don't
    # cross-contaminate.  Scene caches live inside ``_sm_aux_state`` and are
    # persisted separately by the caller if desired.
    if shared_modules is not None:
        tp.modules = shared_modules
    else:
        _fresh: list[AnalyticModule] = []
        for m in parent.modules:
            mc = copy.copy(m)           # shallow — shares sm_params, sm_items etc.
            mc._sm_state     = {}
            mc._sm_out_cache = {}
            mc._sm_aux_state = {}
            mc._sm_log_text  = ""
            _fresh.append(mc)
        tp.modules = _fresh
    tp.parts = _copy_matching_parts_for_voice_keys(
        parent, group_voice_keys, note_keys)
    return tp


def _copy_matching_parts_for_voice_keys(
    source_patch: "AnalyticPatch",
    voice_keys: list[str],
    note_keys: list[str] | None = None,
) -> list[Part]:
    voice_set = frozenset(vk for vk in voice_keys if vk)
    if not voice_set:
        return []

    def _shallow_part(pt: Part) -> Part:
        """Shallow-copy a Part and its Chairs so we can reassign list fields
        without mutating the source patch.  PerformerPlacement objects are
        shared by reference (never mutated during synthesis)."""
        p2 = copy.copy(pt)
        p2.chairs = [copy.copy(ch) for ch in getattr(pt, "chairs", [])]
        return p2

    matches = [
        _shallow_part(pt)
        for pt in getattr(source_patch, "parts", [])
        if frozenset(vk for vk in getattr(pt, "voice_keys", []) if vk) == voice_set
    ]
    parts = matches if matches else [
        _shallow_part(pt)
        for pt in getattr(source_patch, "parts", [])
        if voice_set.issubset(frozenset(vk for vk in getattr(pt, "voice_keys", []) if vk))
    ]
    note_key_set = {nk for nk in (note_keys or []) if nk}
    if not note_key_set:
        return parts
    filtered_parts: list[Part] = []
    for pt in parts:
        kept_chairs: list[Chair] = []
        for ch in getattr(pt, "chairs", []):
            kept_performers = [
                pf for pf in getattr(ch, "performers", [])
                if not getattr(pf, "assigned_note_keys", [])
                or note_key_set.intersection(set(pf.assigned_note_keys))
            ]
            if not kept_performers:
                continue
            ch.performers = kept_performers
            ch.performer_count = len(kept_performers)
            kept_chairs.append(ch)
        if kept_chairs:
            pt.chairs = kept_chairs
            pt.player_count = sum(ch.performer_count for ch in kept_chairs)
            filtered_parts.append(pt)
    return filtered_parts


def _compute_arrangement_metrics(play_groups: list[tuple]) -> dict:
    """Summarize solved score groups for future placement/chair allocation."""
    all_events = []
    group_metrics = []
    voice_event_counts: dict[str, int] = {}
    for gi, (sched, voices) in enumerate(play_groups):
        events = list(getattr(sched, "events", []) or [])
        layer_keys = list(getattr(sched, "_layer_keys", []))
        group_key = str(getattr(sched, "_group_key", f"group:{gi}"))
        voice_keys = [getattr(v, "key", "") for v in voices]
        note_demands = _note_demands_for_group(group_key, events, voices)
        packed_bins, packed_summary = _pack_note_demands_by_voice(note_demands, voices)
        for vk in voice_keys:
            voice_event_counts[vk] = voice_event_counts.get(vk, 0) + len(events)
        overlap = _schedule_overlap_metrics(events, voices)
        group_metrics.append({
            "group_key": group_key,
            "layer_keys": layer_keys,
            "voice_keys": voice_keys,
            "event_count": len(events),
            "start_time": min((ev.start_time for ev in events), default=0.0),
            "end_time": max((ev.start_time + ev.duration_s for ev in events), default=0.0),
            "page_specificity": _page_specificity_score(layer_keys),
            "stack_depth": len(layer_keys),
            "note_demands": note_demands,
            "packed_performers": [
                {
                    "voice_key": pb.get("voice_key", ""),
                    "polyphony_mode": pb.get("polyphony_mode", "sympathetic"),
                    "capacity": int(pb.get("capacity", 1)),
                    "assigned_note_keys": [d["note_key"] for d in pb.get("note_demands", [])],
                    "source_voice_keys": list({
                        d["voice_key"] for d in pb.get("note_demands", []) if d.get("voice_key")
                    }),
                    "source_layer_keys": list({
                        d["layer_key"] for d in pb.get("note_demands", []) if d.get("layer_key")
                    }),
                }
                for pb in packed_bins
            ],
            "voice_profiles": [
                {
                    "voice_key": getattr(v, "key", ""),
                    "polyphony_count": _voice_polyphony_capacity(v),
                    "polyphony_mode": _voice_polyphony_mode(v),
                }
                for v in voices
            ],
            **packed_summary,
            **overlap,
        })
        all_events.extend(events)

    timeline = []
    for ev in all_events:
        t0 = float(getattr(ev, "start_time", 0.0))
        t1 = t0 + float(getattr(ev, "duration_s", 0.0))
        timeline.append((t0, 1))
        timeline.append((t1, -1))
    timeline.sort(key=lambda item: (item[0], item[1]))

    active = 0
    peak = 0
    peak_times: list[float] = []
    for t, delta in timeline:
        active += delta
        if active > peak:
            peak = active
            peak_times = [t]
        elif active == peak and peak > 0:
            peak_times.append(t)

    return {
        "group_count": len(play_groups),
        "event_count": len(all_events),
        "peak_simultaneity": peak,
        "peak_times": peak_times[:32],
        "groups": group_metrics,
        "voice_event_counts": voice_event_counts,
        "peak_required_chairs": max(
            (max(int(gm.get("required_chairs", 1)), int(gm.get("required_performers", 1))) for gm in group_metrics),
            default=1,
        ),
    }


def resolve_parts_from_patch(patch: "AnalyticPatch") -> "list[Part]":
    """Derive :class:`Part` objects from the patch's arrangement metrics.

    Each group in `_arrangement_metrics` becomes one Part. Parts with
    identical voice combinations are deduplicated. Player counts and solver
    hints are preserved from any existing parts already on the patch so that
    manually-set player counts survive a re-solve.

    Returns a fresh list ready to be stored as ``patch.parts``.
    """
    metrics = patch._arrangement_metrics
    if not metrics:
        return []
    groups = metrics.get("groups", [])
    existing_by_key = {pt.key: pt for pt in patch.parts}
    parts: list[Part] = []
    seen_voice_sets: set[frozenset] = set()
    for gm in groups:
        group_key: str = gm.get("group_key", "")
        voice_keys: list = list(gm.get("voice_keys", []))
        vset = frozenset(voice_keys)
        if vset in seen_voice_sets:
            continue
        seen_voice_sets.add(vset)
        layer_keys: list = list(gm.get("layer_keys", []))
        # Infer register and role from group / layer key naming
        combined = group_key + " ".join(layer_keys)
        combined_l = combined.lower()
        if "bass" in combined_l or "sub" in combined_l:
            register = "bass"
        elif "high" in combined_l or "treble" in combined_l or "soprano" in combined_l:
            register = "high"
        elif "mid" in combined_l or "tenor" in combined_l or "alto" in combined_l:
            register = "mid"
        else:
            register = "all"
        if "melody" in combined_l:
            seq_role = "melody"
        elif "root" in combined_l or "bass" in combined_l:
            seq_role = "bass"
        elif "stab" in combined_l or "comp" in combined_l:
            seq_role = "stab"
        else:
            seq_role = ""
        if "signal" in combined_l:
            voice_role = "signal"
        elif "air" in combined_l:
            voice_role = "air"
        elif "transient" in combined_l:
            voice_role = "transient"
        elif "body" in combined_l:
            voice_role = "body"
        else:
            voice_role = ""
        part_key = group_key or f"part-{len(parts)}"
        label_parts = [register, seq_role, voice_role]
        label = " / ".join(p for p in label_parts if p) or part_key
        existing = existing_by_key.get(part_key)
        required_chairs = max(
            1,
            int(gm.get("required_performers", gm.get("required_chairs", 1))),
        )
        player_count = max(required_chairs, existing.player_count if existing else 1)
        solver_hints: dict = {
            "event_count": gm.get("event_count", 0),
            "start_time": gm.get("start_time", 0.0),
            "end_time": gm.get("end_time", 0.0),
            "required_chairs": required_chairs,
            "required_performers": int(gm.get("required_performers", required_chairs)),
            "peak_active_events": int(gm.get("peak_active_events", 0)),
            "peak_active_sources": int(gm.get("peak_active_sources", 0)),
            "interference_peak": float(gm.get("interference_peak", 0.0)),
            "page_specificity": int(gm.get("page_specificity", 0)),
            "stack_depth": int(gm.get("stack_depth", 0)),
            "layer_keys": list(gm.get("layer_keys", [])),
            "voice_profiles": list(gm.get("voice_profiles", [])),
            "packed_performers": list(gm.get("packed_performers", [])),
        }
        if existing and existing.solver_hints:
            solver_hints.update({k: v for k, v in existing.solver_hints.items()
                                  if k not in solver_hints})
        parts.append(Part(
            key=part_key,
            label=label,
            register=register,
            seq_role=seq_role,
            voice_role=voice_role,
            voice_keys=voice_keys,
            player_count=player_count,
            solver_hints=solver_hints,
        ))
    patch.parts = parts
    _refresh_part_placement_layout(patch)
    return parts


def _build_play_groups_from_resolved_notes(
        p: "AnalyticPatch",
) -> "list[tuple[NoteSchedule, list[AnalyticVoice]]]":
    """Build per-note playback groups from locked/edited resolved notes."""
    voice_map = {v.key: v for v in p.voices if not getattr(v, "muted", False)}
    groups: list[tuple[NoteSchedule, list[AnalyticVoice]]] = []
    for note in sorted(p.resolved_notes, key=lambda n: (n.start_time, n.fundamental_hz, n.voice_key)):
        if getattr(note, "is_rest", False):
            continue
        voice = voice_map.get(note.voice_key)
        if voice is None:
            continue
        sched = NoteSchedule()
        ev = NoteEvent(
            fundamental_hz=float(note.fundamental_hz),
            start_time=float(note.start_time),
            duration_s=float(note.duration_s),
            velocity=float(note.velocity),
        )
        ev._layer_key = getattr(note, "layer_key", "roll")
        ev._exact_pitch = True
        sched.add(ev)
        sched._group_key = f"roll:{voice.key}:{note.note_id}"
        sched._layer_keys = [getattr(note, "layer_key", "roll")]
        sched._note_target = _resolve_note_target([voice], p)
        groups.append((sched, [voice]))
    return groups


def _prepare_sequence_play_groups(
        p: "AnalyticPatch",
        beat_s: float,
        degrees: list[float],
        pattern: list[int],
) -> "list[tuple[NoteSchedule, list[AnalyticVoice]]]":
    """Return the playback/export groups for the current patch state."""
    has_locked_roll = any(getattr(n, "locked", False) for n in p.resolved_notes)
    if has_locked_roll:
        _sync_resolved_notes(p, preserve_locked=True)
        groups = _build_play_groups_from_resolved_notes(p)
        p._arrangement_metrics = _compute_arrangement_metrics(groups)
        p.parts = resolve_parts_from_patch(p)
        return groups

    if p.rhythm_enabled:
        source_voices = [v for v in p.voices if not v.muted]
        groups = _group_voices_by_page(
            p, source_voices, beat_s, degrees, pattern,
            min_dur_s=1.0 / p.preview_sr)
    else:
        schedule = _build_legacy_sequence_schedule(p, beat_s, degrees, pattern)
        source_voices = [v for v in p.voices if not v.muted]
        if source_voices:
            schedule._note_target = _resolve_note_target(source_voices, p)
        groups = [(schedule, source_voices)] if source_voices else []
    if getattr(p, "seq_rubato_shape", "off") != "off" and getattr(p, "seq_rubato_amount", 0.0) > 1e-9:
        warped_groups = []
        phrase_supercycle_s = (
            _rubato_phrase_lcm_cycle_s(p, groups, beat_s)
            if getattr(p, "seq_rubato_scope", "bar") == "phrase"
            else 0.0
        )
        for sched, voices in groups:
            if getattr(p, "seq_rubato_scope", "bar") == "phrase":
                local_phrase_s = max(1e-6, _rubato_phrase_cycle_s_for_group(p, voices, beat_s))
                cycle_s = max(local_phrase_s, phrase_supercycle_s)
                repeats = max(1.0, cycle_s / local_phrase_s)
                amt_scale = 1.0 / repeats
            else:
                cycle_s = _rubato_cycle_s_for_group(p, voices, beat_s)
                amt_scale = 1.0
            warped = _apply_rubato_to_schedule(p, sched, cycle_s, amount_scale=amt_scale)
            # Preserve note-target annotation through rubato warp
            warped._note_target = getattr(sched, "_note_target",
                                          _resolve_note_target(voices, p))
            warped_groups.append((warped, voices))
        groups = warped_groups
    p._arrangement_metrics = _compute_arrangement_metrics(groups)
    p.parts = resolve_parts_from_patch(p)
    return groups


def _group_voices_by_page(
        p: "AnalyticPatch",
        source_voices: list,
        beat_s: float,
        degrees: list,
        deg_pattern: list,
        min_dur_s: float = 1.0 / 48000,
) -> "list[tuple]":
    """Return [(NoteSchedule, [voice, ...]), ...] grouped by resolved score stack."""
    pid_to_layers: dict[str, list[tuple[str, RhythmPage]]] = {}
    pid_to_voices: dict[str, list] = {}
    for v in source_voices:
        layers = p.score_page_items_for_voice(v)
        if getattr(p, "rhythm_layer_mode", "union") == "specific":
            stack_key = (layers[-1][0],) if layers else ("all",)
        else:
            stack_key = tuple(key for key, _ in layers) if layers else ("all",)
        pid = "|".join(stack_key)
        if pid not in pid_to_layers:
            pid_to_layers[pid] = layers
            pid_to_voices[pid] = []
        pid_to_voices[pid].append(v)

    result = []
    for pid, voices in pid_to_voices.items():
        sched = _build_score_schedule_for_voice(
            p, voices[0], beat_s, degrees, deg_pattern, min_dur_s=min_dur_s)
        sched._group_key = pid
        sched._layer_keys = [key for key, _ in p.score_page_items_for_voice(voices[0])]
        sched._note_target = _resolve_note_target(voices, p)
        result.append((sched, pid_to_voices[pid]))
    return result
def _voice_effectively_muted(voice: "AnalyticVoice", patch: "AnalyticPatch") -> bool:
    """True when the voice should produce silence — muted, or not soloed when a solo is active."""
    if voice.muted:
        return True
    sk = getattr(patch, "solo_key", None)
    return sk is not None and voice.key != sk


def _resolve_voice_hz(
    voice:     "AnalyticVoice",
    tuning:    "GlobalTuning",
    played_hz: "float | None" = None,
) -> float:
    """
    Resolve the actual oscillator frequency for *voice* given a *tuning* context
    and an optional *played_hz* (the note pitch from the sequencer).

    note_tracking behaviour
    -----------------------
    ``"free"``
        Ignore tuning and played pitch entirely — use ``voice.freq_hz`` as-is.
        Preserves exact backward-compatibility with pre-tuning patches.
    ``"root"``
        Voice is anchored to the tuning root.  ``semitone_offset`` shifts it up/
        down in semitones from ``tuning.root_hz``.  Sequencer notes are ignored.
    ``"note"``  (default)
        Voice tracks the played note.  ``semitone_offset`` is an additive offset
        on top of the played pitch in semitones.  When no note is playing,
        falls back to root behaviour.
    """
    tracking = getattr(voice, "note_tracking", "free")
    offset   = getattr(voice, "semitone_offset", 0.0)
    if tracking == "free":
        return voice.freq_hz
    elif tracking == "root":
        return tuning.semitone_to_hz(offset)
    else:  # "note"
        if played_hz is None or played_hz <= 0.0:
            return tuning.semitone_to_hz(offset)
        played_st = tuning.hz_to_semitones(played_hz)
        return tuning.semitone_to_hz(played_st + offset)


def _resolved_event_hz(
    patch: "AnalyticPatch",
    voice: "AnalyticVoice",
    event_hz: float,
) -> float:
    """Return the actual rendered pitch for *voice* on a scheduled event."""
    resolved_hz = _resolve_voice_hz(voice, patch.tuning, event_hz)
    seq_role = getattr(voice, "seq_role", "melody")
    if seq_role == "bass":
        resolved_hz *= (2.0 ** patch.seq_bass_octave)
    elif seq_role == "root":
        resolved_hz = patch.seq_tonic_hz * (2.0 ** patch.seq_root_octave)
    elif seq_role == "stab":
        resolved_hz *= (2.0 ** patch.seq_stab_octave)
    return resolved_hz


def _hz_to_midi(hz: float) -> float:
    if hz <= 0.0:
        return 0.0
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def _midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def _midi_note_name(midi_note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = int(midi_note)
    return f"{names[note % 12]}{(note // 12) - 1}"


def _bar_duration_s_for_patch(patch: "AnalyticPatch", beat_s: float | None = None) -> float:
    beat = (60.0 / max(float(getattr(patch, "seq_bpm", 120.0)), 1.0)
            if beat_s is None else float(beat_s))
    return beat * patch.beats_per_bar()


def _bar_duration_s_for_page(
        patch: "AnalyticPatch",
        page: "RhythmPage | None",
        beat_s: float | None = None) -> float:
    beat = (60.0 / max(float(getattr(patch, "seq_bpm", 120.0)), 1.0)
            if beat_s is None else float(beat_s))
    return beat * patch.page_beats_per_bar(page)


def _page_meter_label(patch: "AnalyticPatch", page: "RhythmPage | None") -> str:
    num, den = patch.page_meter(page)

    def _fmt(v: float) -> str:
        if abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return f"{v:.2f}".rstrip("0").rstrip(".")

    return f"{_fmt(num)}/{_fmt(den)}"


def _rubato_phase_map(shape: str, amount: float, u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    amt = max(0.0, min(0.95, float(amount)))
    if amt <= 1e-9 or shape == "off":
        return u
    if shape == "sine":
        return u + amt * math.sin(2.0 * math.pi * u) / (2.0 * math.pi)
    if shape == "troughs":
        return u + amt * math.sin(4.0 * math.pi * u) / (4.0 * math.pi)
    if shape == "slow_go":
        eased = u * u
        return (1.0 - amt) * u + amt * eased
    if shape == "go_slow":
        eased = 1.0 - (1.0 - u) * (1.0 - u)
        return (1.0 - amt) * u + amt * eased
    return u


def _rubato_cycle_s_for_group(
        patch: "AnalyticPatch",
        voices: list["AnalyticVoice"],
        beat_s: float) -> float:
    if patch.rhythm_enabled and voices:
        layers = patch.score_page_items_for_voice(voices[0])
        page = layers[-1][1] if layers else None
        bar_s = _bar_duration_s_for_page(patch, page, beat_s)
        if patch.seq_rubato_scope == "phrase":
            bars = max(1, int(getattr(page, "rhythm_prog_bars", patch.rhythm_prog_bars) if page is not None
                              else patch.rhythm_prog_bars))
            return bar_s * bars
        return bar_s
    return _bar_duration_s_for_patch(patch, beat_s)


def _rubato_phrase_cycle_s_for_group(
        patch: "AnalyticPatch",
        voices: list["AnalyticVoice"],
        beat_s: float) -> float:
    if patch.rhythm_enabled and voices:
        layers = patch.score_page_items_for_voice(voices[0])
        page = layers[-1][1] if layers else None
        bar_s = _bar_duration_s_for_page(patch, page, beat_s)
        bars = max(1, int(getattr(page, "rhythm_prog_bars", patch.rhythm_prog_bars) if page is not None
                          else patch.rhythm_prog_bars))
        return bar_s * bars
    return _bar_duration_s_for_patch(patch, beat_s)


def _rubato_phrase_lcm_cycle_s(
        patch: "AnalyticPatch",
        groups: "list[tuple[NoteSchedule, list[AnalyticVoice]]]",
        beat_s: float) -> float:
    durations: list[Fraction] = []
    for _, voices in groups:
        phrase_s = max(1e-6, _rubato_phrase_cycle_s_for_group(patch, voices, beat_s))
        qbeats = Fraction(phrase_s / max(1e-9, beat_s)).limit_denominator(768)
        durations.append(qbeats)
    if not durations:
        return _bar_duration_s_for_patch(patch, beat_s)
    lcm_num = durations[0].numerator
    gcd_den = durations[0].denominator
    for frac in durations[1:]:
        lcm_num = math.lcm(lcm_num, frac.numerator)
        gcd_den = math.gcd(gcd_den, frac.denominator)
    supercycle_qbeats = Fraction(lcm_num, gcd_den)
    return float(supercycle_qbeats) * beat_s


def _apply_rubato_to_schedule(
        patch: "AnalyticPatch",
        schedule: "NoteSchedule",
        cycle_s: float,
        amount_scale: float = 1.0) -> "NoteSchedule":
    shape = getattr(patch, "seq_rubato_shape", "off")
    amount = float(getattr(patch, "seq_rubato_amount", 0.0)) * max(0.0, float(amount_scale))
    if shape == "off" or amount <= 1e-9 or cycle_s <= 1e-9 or not getattr(schedule, "events", None):
        return schedule
    warped = NoteSchedule()
    for ev in schedule.events:
        start = float(ev.start_time)
        end = max(start + 1e-6, float(ev.start_time + ev.duration_s))
        c0 = math.floor(start / cycle_s)
        c1 = math.floor(end / cycle_s)
        if c0 != c1:
            c1 = c0
            end = min(end, (c0 + 1.0) * cycle_s)
        u0 = (start - c0 * cycle_s) / cycle_s
        u1 = (end - c0 * cycle_s) / cycle_s
        t0 = c0 * cycle_s + cycle_s * _rubato_phase_map(shape, amount, u0)
        t1 = c0 * cycle_s + cycle_s * _rubato_phase_map(shape, amount, u1)
        new_ev = NoteEvent(
            fundamental_hz=float(ev.fundamental_hz),
            start_time=float(t0),
            duration_s=max(1e-6, float(t1 - t0)),
            velocity=float(getattr(ev, "velocity", 1.0)),
        )
        for attr in ("_layer_key", "_exact_pitch"):
            if hasattr(ev, attr):
                setattr(new_ev, attr, getattr(ev, attr))
        warped.add(new_ev)
    for attr in ("_group_key", "_layer_keys"):
        if hasattr(schedule, attr):
            setattr(warped, attr, getattr(schedule, attr))
    return warped


def _meter_beat_units(meter_num: float, frac_beat_mode: str = "warp") -> list[float]:
    """Return the beat-unit list for *meter_num*.

    frac_beat_mode
        ``"grid"``  — fractional remainder is a visible beat cell.
        ``"warp"``  — remainder is absorbed into the warp curve; only full
                      integer beats are returned.
    """
    meter_num = max(0.125, float(meter_num))
    full_beats = int(math.floor(meter_num + 1e-9))
    units = [1.0] * max(0, full_beats)
    frac = meter_num - float(full_beats)
    if frac > 1e-6 and frac_beat_mode == "grid":
        units.append(frac)
    if not units:
        units = [meter_num]
    return units


def _stress_pattern_options_for_meter(meter_num: float, frac_beat_mode: str = "warp") -> list[list[int]]:
    beat_units = max(1, len(_meter_beat_units(meter_num, frac_beat_mode)))
    curated: dict[int, list[list[int]]] = {
        1: [[1]],
        2: [[2], [1, 1]],
        3: [[3], [2, 1], [1, 2]],
        4: [[2, 2], [3, 1], [1, 3]],
        5: [[3, 2], [2, 3], [2, 2, 1], [1, 2, 2]],
        6: [[3, 3], [2, 2, 2], [3, 2, 1], [1, 2, 3]],
        7: [[2, 2, 3], [3, 2, 2], [2, 3, 2]],
        8: [[3, 3, 2], [2, 3, 3], [3, 2, 3], [4, 4], [2, 2, 2, 2]],
        9: [[3, 3, 3], [2, 2, 2, 3], [3, 2, 2, 2]],
        10: [[3, 3, 2, 2], [2, 3, 3, 2], [3, 2, 3, 2], [2, 2, 3, 3]],
        11: [[3, 3, 3, 2], [2, 3, 3, 3], [3, 2, 3, 3], [3, 3, 2, 3]],
    }
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def _push(pattern: list[int]) -> None:
        if sum(pattern) != beat_units:
            return
        key = tuple(pattern)
        if key not in seen:
            seen.add(key)
            out.append(pattern)

    for pattern in curated.get(beat_units, []):
        _push(pattern)

    allowed_primary = (2, 3)
    allowed_fallback = (1, 2, 3)

    def _build(rem: int, parts: tuple[int, ...], acc: list[int]) -> None:
        if rem == 0:
            _push(list(acc))
            return
        for part in parts:
            if part <= rem:
                acc.append(part)
                _build(rem - part, parts, acc)
                acc.pop()

    _build(beat_units, allowed_primary, [])
    _build(beat_units, allowed_fallback, [])
    if not out:
        out.append([beat_units])
    return out


def _page_stress_pattern(patch: "AnalyticPatch", page: Any | None) -> list[int]:
    _fbm = getattr(page, "frac_beat_mode", "warp") if page is not None else "warp"
    options = _stress_pattern_options_for_meter(
        patch.page_meter(page if isinstance(page, RhythmPage) else None)[0]
        if page is not None else patch.meter_numerator,
        frac_beat_mode=_fbm)
    current = [int(max(1, int(x))) for x in getattr(page, "stress_pattern", [])] if page is not None else []
    if sum(current) == sum(options[0]):
        return current
    return options[0]


def _stress_step_boundaries(
        patch: "AnalyticPatch",
        page: Any,
        div: int) -> tuple[list[int], list[int], list[int]]:
    num, _ = patch.page_meter(page if isinstance(page, RhythmPage) else None)
    _fbm = getattr(page, "frac_beat_mode", "warp") if page is not None else "warp"
    beat_units = _meter_beat_units(num, _fbm)
    total_units = max(1e-9, sum(beat_units))
    beat_edges = [0]
    acc = 0.0
    for unit in beat_units:
        acc += unit
        beat_edges.append(int(round((acc / total_units) * div)))
    beat_edges[0] = 0
    beat_edges[-1] = div

    pattern = _page_stress_pattern(patch, page)
    group_edges = [0]
    group_acc = 0
    for part in pattern:
        group_acc += int(part)
        idx = min(len(beat_edges) - 1, group_acc)
        group_edges.append(beat_edges[idx])
    if group_edges[-1] != div:
        group_edges[-1] = div
    beat_starts = sorted({max(0, min(div - 1, beat_edges[i])) for i in range(len(beat_edges) - 1)})
    group_starts = sorted({max(0, min(div - 1, group_edges[i])) for i in range(len(group_edges) - 1)})
    return beat_edges, beat_starts, group_starts


def _auto_apply_rhythm_grid(patch: "AnalyticPatch", page: Any) -> None:
    div = max(1, int(page.rhythm_division))
    if not page.rhythm_patterns:
        page.rhythm_patterns = [RhythmPattern(name="Pat 1")]
    act_i = min(page.rhythm_active_pat, len(page.rhythm_patterns) - 1)
    pat = page.rhythm_patterns[act_i]
    pat.ensure_size(div)
    _, beat_starts, group_starts = _stress_step_boundaries(patch, page, div)
    pat.steps = [False] * div
    pat.art = [0] * div
    for step_i in beat_starts:
        pat.steps[step_i] = True
    for step_i in group_starts:
        pat.steps[step_i] = True
        pat.art[step_i] = 1


def _auto_apply_dynamics_accent(
        patch: "AnalyticPatch",
        page: Any,
        dyn_program: "DynamicsProgram") -> None:
    """Distribute Western conventional accent hierarchy onto the accent tree.

    Uses the meter and stress grouping to assign accent levels:
        - Group-start beats  → 2.0  (strong / downbeat)
        - Other beat starts  → 1.5  (accent)
        - Off-beat positions  → 0.5  (weak / ghosted)

    For odd meters the grouping (e.g. 7/8 = 2+2+3) is respected so that the
    '1' of each group pocket gets the strong accent.

    Operates on the accent *tree* of the active rhythm pattern so that it
    works regardless of how the tree is subdivided.
    """
    if page is None:
        return
    div = max(1, int(page.rhythm_division))
    num, _ = patch.page_meter(page if isinstance(page, RhythmPage) else None)

    # Compute beat and group boundaries in [0, div] integer space
    _fbm = getattr(page, "frac_beat_mode", "warp") if page is not None else "warp"
    beat_units  = _meter_beat_units(num, _fbm)
    total_units = max(1e-9, sum(beat_units))
    beat_frac_edges: list[float] = [0.0]
    acc = 0.0
    for unit in beat_units:
        acc += unit
        beat_frac_edges.append(acc / total_units)
    beat_frac_edges[-1] = 1.0

    pattern     = _page_stress_pattern(patch, page)
    group_frac_edges: list[float] = [0.0]
    group_acc = 0
    for part in pattern:
        group_acc += int(part)
        idx = min(len(beat_frac_edges) - 1, group_acc)
        group_frac_edges.append(beat_frac_edges[idx])
    if group_frac_edges[-1] != 1.0:
        group_frac_edges[-1] = 1.0

    beat_set  = set(beat_frac_edges[:-1])   # fractional positions that start a beat
    group_set = set(group_frac_edges[:-1])  # fractional positions that start a group

    # Get the accent tree for the active rhythm pattern
    rpats = page.rhythm_patterns
    if not rpats:
        return
    act_i   = min(page.rhythm_active_pat, max(0, len(rpats) - 1))
    act_pat = rpats[act_i]
    acc_tree = act_pat.get_accent_tree(div)

    # Walk every leaf and assign accent level by positional proximity
    eps = 0.5 / max(1, div)  # tolerance: half a grid step
    for leaf in acc_tree.flat_leaves():
        pos = float(leaf.position)
        # Check group-start first (strongest)
        if any(abs(pos - g) < eps for g in group_set):
            leaf.vel = 2.0
        elif any(abs(pos - b) < eps for b in beat_set):
            leaf.vel = 1.5
        else:
            leaf.vel = 0.5


def _auto_apply_stress_velocity(
        patch: "AnalyticPatch",
        page: Any,
        dyn_program: "DynamicsProgram") -> None:
    """Map stress rank → per-leaf velocity on the accent tree.

    Strong beats (group_starts) receive velocity 2.0 (strong),
    normal beat starts receive 1.0 (normal), all others 0.5 (weak).
    Delegates to the same tree-native logic as Auto Accent.
    """
    _auto_apply_dynamics_accent(patch, page, dyn_program)
