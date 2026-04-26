"""demo_composer_song.py — TorchComposerNode end-to-end demo.

Runs the full composition pipeline from AnalyticPatch through the KPN solver.
TorchComposerNode calls _build_rhythm_schedule per voice per epoch, firing all
existing engines: BeatTree/WarpCurve (rhythm), apply_dynamics, apply_improv.

Degree tables are derived from GlobalTuning + MODAL_SCALES so every scale,
temperament, and concert-pitch variant in the system is exercised rather than
relying on hardcoded Hz values.  Change CONCERT_A, TEMPERAMENT, or SCALE_NAME
at the top and the entire pitch world re-derives.

TorchComposerPanel (PatchPanel) and TorchComposerPianoRoll (EditorCanvas piano
roll) are bound to the node after construction.

Usage
-----
  python demo_composer_song.py           # render composer_demo.wav
  python demo_composer_song.py --ui      # pygame viewer with panel + piano roll
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
from pathlib import Path
from typing import List

import numpy as np
import torch

from analytic_model import AnalyticPatch, GlobalTuning
from edge_fifo_bank import EdgeFifoBank
from graph_solver import TensorEdge
from network_materializer import compile_nodes
from parametric_curve import ControlPoint, ParametricCurve
from sequence_engine import MODAL_SCALES, SCALE_MOODS
from audio_projector_node import AudioProjectorNode
from driver_node import DriverNode, VoiceSpread
from instrument_node import InstrumentNode
from resonator_core import ResonatorString, StringCouplingConfig
from score_persist import save_score, VoicePremixCapture
from torch_composer_node import TorchComposerNode
from voice_graph_node import MetaVoiceNode, MixerSumNode, VoiceTorchOscillator

# ── Song constants ─────────────────────────────────────────────────────────────
BPM        = 120
BEAT_S     = 60.0 / BPM
SR         = 48_000.0
RELEASE_S  = 0.08
COMPOSER_KEY = "composer"

# 4/4, 16 divisions.  seq_repeats * rhythm_prog_bars = 4*2 = 8 bars = 16 s.
N_BARS     = 8
SONG_DUR_S = N_BARS * 4 * BEAT_S          # 16.0 s
N_FRAMES   = int(SONG_DUR_S * SR)

# ── Tuning system ──────────────────────────────────────────────────────────────
# All pitch values are derived from GlobalTuning so that changing CONCERT_A,
# TEMPERAMENT, or SCALE_NAME automatically re-derives every Hz in the piece.
#
# Available temperaments: "12tet" | "just" | "pythagorean" | "custom"
# Available scales: see MODAL_SCALES (ionian, dorian, phrygian, lydian,
#   mixolydian, aeolian, locrian, harmonic_minor, melodic_minor,
#   phrygian_dominant, lydian_dominant, pentatonic_major, pentatonic_minor,
#   blues, whole_tone, octatonic_hw, octatonic_wh, chromatic,
#   double_harmonic, hungarian_minor, neapolitan_minor, enigmatic,
#   bebop_dominant, altered)
CONCERT_A   = 440.0        # standard pitch reference for A4
TEMPERAMENT = "12tet"      # equal temperament
SCALE_NAME  = "aeolian"    # D natural minor (D E F G A Bb C)
TONIC_NOTE  = "D4"         # melodic register root

# GlobalTuning instance — semitone_to_hz(n) gives Hz for n steps above root_hz
_TUNING = GlobalTuning(root_hz=CONCERT_A, temperament=TEMPERAMENT)

# Note-name lookup tables
_PC: dict = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def note_midi(name: str) -> int:
    """Parse a note name ("D4", "Bb3", "F#5") into a MIDI number (A4=69)."""
    m = re.match(r'([A-G][#b]?)(-?\d+)', name)
    if not m:
        raise ValueError(f"Cannot parse note name {name!r}")
    return _PC[m.group(1)] + (int(m.group(2)) + 1) * 12


def note_hz(name: str, concert_a: float = CONCERT_A, temperament: str = TEMPERAMENT) -> float:
    """Note name → Hz using the specified temperament and concert A pitch."""
    midi = note_midi(name)
    t = GlobalTuning(root_hz=concert_a, temperament=temperament)
    # semitones from A4 (MIDI 69)
    semis = midi - 69
    return t.semitone_to_hz(semis)


def scale_hz(
    tonic: str,
    scale_name: str = SCALE_NAME,
    *,
    octave_offsets: tuple = (0, 1),
    concert_a: float = CONCERT_A,
    temperament: str = TEMPERAMENT,
) -> List[float]:
    """Build an Hz list for a named scale across the requested octave range.

    Parameters
    ----------
    tonic:
        Root note name, e.g. "D4".  The *register* (octave) sets the lowest
        note of the first octave window.
    scale_name:
        Any key from MODAL_SCALES.  Defaults to aeolian (natural minor).
    octave_offsets:
        Tuple of integer octave shifts relative to *tonic*.  ``(0, 1)``
        generates one octave from tonic and one octave above it.
    concert_a:
        A4 reference frequency in Hz (default 440.0).
    temperament:
        GlobalTuning temperament string.

    Returns
    -------
    list[float]
        Hz values ordered lowest → highest.
    """
    semitones = MODAL_SCALES.get(scale_name, MODAL_SCALES["aeolian"])
    tonic_hz  = note_hz(tonic, concert_a, temperament)
    result: List[float] = []
    for oct_off in sorted(octave_offsets):
        for semi in semitones:
            hz = tonic_hz * 2.0 ** ((oct_off * 12 + semi) / 12.0)
            result.append(hz)
    return result


# ── Degree tables derived from the tuning system ───────────────────────────────
# Melody: D aeolian across octaves 0 and +1 from D4
#   D4 E4 F4 G4 A4 Bb4 C5 D5 E5 F5 G5 A5 Bb5 C6
MELODY_DEGREES = scale_hz(TONIC_NOTE, SCALE_NAME, octave_offsets=(0, 1))

# Bass: D aeolian, two octaves below tonic (D2 register)
#   D2 E2 F2 G2 A2 Bb2 C3 D3
BASS_DEGREES = scale_hz(TONIC_NOTE, SCALE_NAME, octave_offsets=(-2, -1))

# Pad: D aeolian D3–D5 — reference register; voice oscillators transpose per string
PAD_DEGREES = scale_hz(TONIC_NOTE, SCALE_NAME, octave_offsets=(-1, 0))

# Progression patterns
MELODY_PATTERN = [0, 2, 4, 6, 5, 3, 1, 4, 7, 5, 3, 1, 0, 4, 6, 3]
BASS_PATTERN   = [0, 4, 0, 6, 0, 5, 4, 0]
PAD_PATTERN    = [0, 2, 4, 7, 4, 2, 0, 4]

# Announce tuning choices at import time for audit trail.
_MOOD = SCALE_MOODS.get(SCALE_NAME, {})
print(
    f"[tuning]  {SCALE_NAME}  root={TONIC_NOTE}  "
    f"A={CONCERT_A}Hz  {TEMPERAMENT}  "
    f"mood={_MOOD.get('adjectives', ['?'])[:3]}"
)


# ── Envelope curves ────────────────────────────────────────────────────────────

def _melody_env() -> ParametricCurve:
    c = ParametricCurve()
    c.name = "melody_adsr"
    c.points = [
        ControlPoint(t=0.00, v=0.0),
        ControlPoint(t=0.05, v=1.0),
        ControlPoint(t=0.22, v=0.82),
        ControlPoint(t=0.80, v=0.78),
        ControlPoint(t=1.00, v=0.0),
    ]
    return c


def _bass_env() -> ParametricCurve:
    c = ParametricCurve()
    c.name = "bass_pluck"
    c.points = [
        ControlPoint(t=0.00, v=0.0),
        ControlPoint(t=0.02, v=1.0),
        ControlPoint(t=0.28, v=0.55),
        ControlPoint(t=0.75, v=0.38),
        ControlPoint(t=1.00, v=0.0),
    ]
    return c


# ── Body envelopes (slow, warm) ────────────────────────────────────────────────

def _body_env_1() -> ParametricCurve:
    c = ParametricCurve()
    c.name = "body_1_foundation"
    c.points = [
        ControlPoint(t=0.00, v=0.0),
        ControlPoint(t=0.28, v=0.75),
        ControlPoint(t=0.55, v=0.70),
        ControlPoint(t=0.82, v=0.65),
        ControlPoint(t=1.00, v=0.0),
    ]
    return c


def _body_env_2() -> ParametricCurve:
    c = ParametricCurve()
    c.name = "body_2_mid"
    c.points = [
        ControlPoint(t=0.00, v=0.0),
        ControlPoint(t=0.20, v=0.60),
        ControlPoint(t=0.45, v=0.68),
        ControlPoint(t=0.75, v=0.60),
        ControlPoint(t=1.00, v=0.0),
    ]
    return c


def _body_env_3() -> ParametricCurve:
    """Slow swell — arrives late, peaks well after the other body layers."""
    c = ParametricCurve()
    c.name = "body_3_swell"
    c.points = [
        ControlPoint(t=0.00, v=0.0),
        ControlPoint(t=0.15, v=0.35),
        ControlPoint(t=0.40, v=0.58),
        ControlPoint(t=0.60, v=0.65),
        ControlPoint(t=0.80, v=0.60),
        ControlPoint(t=1.00, v=0.0),
    ]
    return c


# ── Shimmer envelopes (fast, bright, granular) ────────────────────────────────

def _shim_env_1() -> ParametricCurve:
    c = ParametricCurve()
    c.name = "shim_1_sparkle"
    c.points = [
        ControlPoint(t=0.00, v=0.0),
        ControlPoint(t=0.06, v=0.80),
        ControlPoint(t=0.22, v=0.60),
        ControlPoint(t=0.65, v=0.50),
        ControlPoint(t=1.00, v=0.0),
    ]
    return c


def _shim_env_2() -> ParametricCurve:
    c = ParametricCurve()
    c.name = "shim_2_airy"
    c.points = [
        ControlPoint(t=0.00, v=0.0),
        ControlPoint(t=0.10, v=0.65),
        ControlPoint(t=0.32, v=0.55),
        ControlPoint(t=0.70, v=0.45),
        ControlPoint(t=1.00, v=0.0),
    ]
    return c


def _shim_env_3() -> ParametricCurve:
    """Tremolo-granular: rapid amplitude undulation throughout the note."""
    c = ParametricCurve()
    c.name = "shim_3_granular"
    c.points = [
        ControlPoint(t=0.00, v=0.0),
        ControlPoint(t=0.03, v=0.55),
        ControlPoint(t=0.10, v=0.22),
        ControlPoint(t=0.20, v=0.50),
        ControlPoint(t=0.32, v=0.18),
        ControlPoint(t=0.45, v=0.48),
        ControlPoint(t=0.58, v=0.15),
        ControlPoint(t=0.72, v=0.40),
        ControlPoint(t=0.85, v=0.12),
        ControlPoint(t=1.00, v=0.0),
    ]
    return c


# ── Sub envelopes (very slow, deep) ───────────────────────────────────────────

def _sub_env_1() -> ParametricCurve:
    c = ParametricCurve()
    c.name = "sub_1_deep"
    c.points = [
        ControlPoint(t=0.00, v=0.0),
        ControlPoint(t=0.40, v=0.55),
        ControlPoint(t=0.65, v=0.52),
        ControlPoint(t=0.88, v=0.45),
        ControlPoint(t=1.00, v=0.0),
    ]
    return c


def _sub_env_2() -> ParametricCurve:
    c = ParametricCurve()
    c.name = "sub_2_rumble"
    c.points = [
        ControlPoint(t=0.00, v=0.0),
        ControlPoint(t=0.35, v=0.50),
        ControlPoint(t=0.58, v=0.55),
        ControlPoint(t=0.82, v=0.42),
        ControlPoint(t=1.00, v=0.0),
    ]
    return c


# ── Patch ─────────────────────────────────────────────────────────────────────

PATCH_JSON = Path(__file__).with_name("symphonic_composer_song.json")


def _load_patch() -> AnalyticPatch:
    """Load the symphonic composer patch from JSON."""
    with PATCH_JSON.open(encoding="utf-8") as f:
        return AnalyticPatch.from_dict(json.load(f))


# ── Voice construction ─────────────────────────────────────────────────────────

def _voice(key: str, osc: VoiceTorchOscillator, group: str, dur_s: float, release_s: float) -> MetaVoiceNode:
    return MetaVoiceNode(key, osc, score_contract={
        "kind":           "voice_score_consumer",
        "voice_key":      key,
        "groups":         (group,),
        "pages":          (),
        "aggregation":    "mask",
        "envelope_curve": osc.envelope_curve,
        "chirp_curve":    osc.chirp_curve,
        "duration_s":     dur_s,
        "release_tail_s": release_s,
    })


def _build_voices(dur_s: float) -> tuple:
    """Return (v_mel, v_bass, body_voices, shimmer_voices, sub_voices)."""
    mel_osc = VoiceTorchOscillator(
        freq_hz=note_hz(TONIC_NOTE), sample_rate=SR, duration=dur_s,
        envelope_curve=_melody_env(),
    )
    bass_osc = VoiceTorchOscillator(
        freq_hz=note_hz(TONIC_NOTE) * 0.25, sample_rate=SR, duration=dur_s,
        envelope_curve=_bass_env(),
    )

    # Body: warm, dark harmonics (D3 register)
    body_root = note_hz(TONIC_NOTE) * 0.5
    body_oscs = [
        VoiceTorchOscillator(freq_hz=body_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=5,
                             harmonic_brightness=0.55, envelope_curve=_body_env_1()),
        VoiceTorchOscillator(freq_hz=body_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=4,
                             harmonic_brightness=0.40, envelope_curve=_body_env_2()),
        VoiceTorchOscillator(freq_hz=body_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=6,
                             harmonic_brightness=0.70, envelope_curve=_body_env_3()),
    ]

    # Shimmer: bright + inharmonic granular (D4 register)
    shim_root = note_hz(TONIC_NOTE)
    shim_oscs = [
        VoiceTorchOscillator(freq_hz=shim_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=12,
                             harmonic_brightness=1.80, envelope_curve=_shim_env_1()),
        VoiceTorchOscillator(freq_hz=shim_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic_warp", harmonic_count=10,
                             harmonic_brightness=1.40, harmonic_warp_strength=0.12,
                             envelope_curve=_shim_env_2()),
        VoiceTorchOscillator(freq_hz=shim_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic_warp", harmonic_count=16,
                             harmonic_brightness=1.20, harmonic_warp_strength=0.28,
                             envelope_curve=_shim_env_3()),
    ]

    # Sub: near-sine, very slow (D1 register)
    sub_root = note_hz(TONIC_NOTE) * 0.125
    sub_oscs = [
        VoiceTorchOscillator(freq_hz=sub_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=2,
                             harmonic_brightness=0.20, envelope_curve=_sub_env_1()),
        VoiceTorchOscillator(freq_hz=sub_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=3,
                             harmonic_brightness=0.30, envelope_curve=_sub_env_2()),
    ]

    v_mel  = _voice("v_mel",  mel_osc,  "melody", dur_s, RELEASE_S)
    v_bass = _voice("v_bass", bass_osc, "bass",   dur_s, 0.12)

    body_voices = [_voice(f"v_body_{i+1}", osc, "pad", dur_s, 0.25) for i, osc in enumerate(body_oscs)]
    shim_voices = [_voice(f"v_shim_{i+1}", osc, "pad", dur_s, 0.15) for i, osc in enumerate(shim_oscs)]
    sub_voices  = [_voice(f"v_sub_{i+1}",  osc, "pad", dur_s, 0.40) for i, osc in enumerate(sub_oscs)]

    return v_mel, v_bass, body_voices, shim_voices, sub_voices


# ── Solver wiring ──────────────────────────────────────────────────────────────

INSTR_KEY  = "instrument"
STR_1_KEY  = "str_1"   # low register — warm body voices
STR_2_KEY  = "str_2"   # mid register — bright shimmer voices
STR_3_KEY  = "str_3"   # deep register — sub voices


def _score_edge(src: str, dst_node_key: str, bank: EdgeFifoBank) -> TensorEdge:
    return TensorEdge(
        src, dst_node_key,
        weight=0.0 + 0j, semantic_role="score",
        src_port="score_out", dst_port="score_in", fifo_bank=bank,
    )


def _sig_edges(voice_keys: list[str], dst: str) -> list[TensorEdge]:
    return [TensorEdge(k, dst, weight=1.0 + 0j, semantic_role="mix_source") for k in voice_keys]


def _build_driver(key: str, dur_s: float, release_s: float, spreads: list) -> DriverNode:
    return DriverNode(key, groups=("pad",), duration_s=dur_s, release_tail_s=release_s,
                      spreads=spreads, sample_rate=SR, layer="voice")


def _build_solver(patch, v_mel, v_bass, body_voices, shim_voices, sub_voices):
    """Wire composer → instrument → 3 strings → voices → instrument body cavity → mixer."""
    bank = EdgeFifoBank(stride=1)

    composer = TorchComposerNode(
        COMPOSER_KEY, patch=patch, bpm=float(BPM),
        degrees_by_group={
            "melody": MELODY_DEGREES,
            "bass":   BASS_DEGREES,
            "pad":    PAD_DEGREES,
        },
        deg_pattern_by_group={
            "melody": MELODY_PATTERN,
            "bass":   BASS_PATTERN,
            "pad":    PAD_PATTERN,
        },
        sample_rate=SR, fifo_bank=bank, layer="score",
    )

    # Three strings of the instrument — spreads ordered to match voice consumers
    str_1 = _build_driver(STR_1_KEY, SONG_DUR_S, 0.25, spreads=[
        VoiceSpread(n_samples=6,  phase_std=0.60, amplitude_std=0.16, time_std_s=0.080),
        VoiceSpread(n_samples=6,  phase_std=0.80, amplitude_std=0.20, time_std_s=0.070),
        VoiceSpread(n_samples=4,  phase_std=0.50, amplitude_std=0.12, time_std_s=0.120),
    ])
    str_2 = _build_driver(STR_2_KEY, SONG_DUR_S, 0.15, spreads=[
        VoiceSpread(n_samples=8,  phase_std=1.20, amplitude_std=0.30, time_std_s=0.040),
        VoiceSpread(n_samples=8,  phase_std=1.00, amplitude_std=0.24, time_std_s=0.050),
        VoiceSpread(n_samples=12, phase_std=1.60, amplitude_std=0.40, time_std_s=0.030),
    ])
    str_3 = _build_driver(STR_3_KEY, SONG_DUR_S, 0.40, spreads=[
        VoiceSpread(n_samples=4,  phase_std=0.30, amplitude_std=0.10, time_std_s=0.160),
        VoiceSpread(n_samples=4,  phase_std=0.40, amplitude_std=0.12, time_std_s=0.140),
    ])

    # ONE instrument body — three strings at their natural resonant pitches
    s1_root = note_hz(TONIC_NOTE) * 0.5    # D3 ≈ 146.8 Hz
    s2_root = note_hz(TONIC_NOTE)           # D4 ≈ 293.7 Hz
    s3_root = note_hz(TONIC_NOTE) * 0.125  # D1 ≈ 36.7 Hz
    instrument = InstrumentNode(
        INSTR_KEY,
        driver_keys=[STR_1_KEY, STR_2_KEY, STR_3_KEY],
        resonator_strings=[
            ResonatorString(key="s1", fundamental_hz=s1_root, decay_s=1.8, drive_gain=0.90, x=0.30, y=-0.10),
            ResonatorString(key="s2", fundamental_hz=s2_root, decay_s=0.8, drive_gain=1.00, x=0.50, y= 0.05),
            ResonatorString(key="s3", fundamental_hz=s3_root, decay_s=3.0, drive_gain=0.85, x=0.70, y= 0.00),
        ],
        coupling_config=StringCouplingConfig(base_strength=0.22, max_coupling=0.50),
        body_type="string_plate",
        body_jitter_seed=1,
        sympathy_threshold=0.04,
        groups=("pad",),
        duration_s=SONG_DUR_S,
        release_tail_s=0.40,
        sample_rate=SR,
    )

    # Build all voice nodes
    v_mel_nodes,  v_mel_edges  = v_mel.build_nodes()
    v_bass_nodes, v_bass_edges = v_bass.build_nodes()
    all_body_nodes, all_body_edges = [], []
    all_shim_nodes, all_shim_edges = [], []
    all_sub_nodes,  all_sub_edges  = [], []
    for v in body_voices:
        ns, es = v.build_nodes();  all_body_nodes += ns;  all_body_edges += es
    for v in shim_voices:
        ns, es = v.build_nodes();  all_shim_nodes += ns;  all_shim_edges += es
    for v in sub_voices:
        ns, es = v.build_nodes();  all_sub_nodes  += ns;  all_sub_edges  += es

    mixer_node     = MixerSumNode("__mix__").build_node()
    projector      = AudioProjectorNode("audio_out", sample_rate=SR, projection="real")
    projector_node = projector.build_node()

    body_out_keys = [f"{v.key}_out" for v in body_voices]
    shim_out_keys = [f"{v.key}_out" for v in shim_voices]
    sub_out_keys  = [f"{v.key}_out" for v in sub_voices]

    # Score: composer → instrument (fans to all strings) → voices
    score_edges = [
        _score_edge(COMPOSER_KEY, "v_mel_out",  bank),
        _score_edge(COMPOSER_KEY, "v_bass_out", bank),
        _score_edge(COMPOSER_KEY, INSTR_KEY,    bank),
        _score_edge(INSTR_KEY,    STR_1_KEY,    bank),
        _score_edge(INSTR_KEY,    STR_2_KEY,    bank),
        _score_edge(INSTR_KEY,    STR_3_KEY,    bank),
        *[_score_edge(STR_1_KEY, k, bank) for k in body_out_keys],
        *[_score_edge(STR_2_KEY, k, bank) for k in shim_out_keys],
        *[_score_edge(STR_3_KEY, k, bank) for k in sub_out_keys],
    ]

    # Signal: voices → strings → instrument body cavity
    sig_edges = [
        *_sig_edges(body_out_keys, STR_1_KEY),
        *_sig_edges(shim_out_keys, STR_2_KEY),
        *_sig_edges(sub_out_keys,  STR_3_KEY),
        TensorEdge(STR_1_KEY, INSTR_KEY, weight=1.0 + 0j, semantic_role="mix_source"),
        TensorEdge(STR_2_KEY, INSTR_KEY, weight=1.0 + 0j, semantic_role="mix_source"),
        TensorEdge(STR_3_KEY, INSTR_KEY, weight=1.0 + 0j, semantic_role="mix_source"),
    ]

    mix_edges = [
        TensorEdge("v_mel_out",  "__mix__",   weight=1.00 + 0j, semantic_role="mix_source"),
        TensorEdge("v_bass_out", "__mix__",   weight=0.80 + 0j, semantic_role="mix_source"),
        TensorEdge(INSTR_KEY,    "__mix__",   weight=0.55 + 0j, semantic_role="mix_source"),
        TensorEdge("__mix__",    "audio_out", weight=1.00 + 0j, semantic_role="audio_projection_source"),
    ]

    compiled = compile_nodes(
        nodes=[
            # Score-delivery chain must be registered in dispatch_before_start
            # order: composer → instrument → str drivers → voice nodes.
            composer.build_node(),
            instrument.build_node(),
            str_1.build_node(), str_2.build_node(), str_3.build_node(),
            *v_mel_nodes, *v_bass_nodes,
            *all_body_nodes, *all_shim_nodes, *all_sub_nodes,
            mixer_node, projector_node,
        ],
        edges=[
            *v_mel_edges, *v_bass_edges,
            *all_body_edges, *all_shim_edges, *all_sub_edges,
            *score_edges, *sig_edges, *mix_edges,
        ],
        sample_rate=SR,
    )

    return compiled, bank, composer, projector, (str_1, str_2, str_3), instrument


def _extract_audio(outputs: dict) -> "np.ndarray | None":
    """Pull normalised float32 mono audio from the projector output."""
    from audio_projector_node import _time_series_1d
    raw = outputs.get("audio_out")
    if raw is None:
        return None
    audio = _time_series_1d(raw, part="real").float().numpy()
    peak = float(np.abs(audio).max())
    if peak < 1e-8:
        return None
    return (audio * (0.92 / peak)).astype(np.float32)


# ── WAV render ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f'Building "Composer Demo" — {N_BARS} bars · {BPM} BPM · D minor · 1 instrument · 3 strings · 10 voices ...')
    print(f"  {N_FRAMES:,} samples · {SONG_DUR_S:.1f} s · {int(SR)} Hz")

    patch = _load_patch()
    v_mel, v_bass, body_voices, shim_voices, sub_voices = _build_voices(SONG_DUR_S)
    compiled, bank, composer, _projector, _drivers, instrument = _build_solver(
        patch, v_mel, v_bass, body_voices, shim_voices, sub_voices
    )

    solver = compiled.solver
    print(f"  Nodes: {len(solver.nodes)} · Edges: {len(solver.edges)} "
          f"· Contract edges: {len(solver.contract_edges)}")

    solver.reset()
    solver.dispatch_before_start()

    print(f"Rendering {N_FRAMES:,} samples ...")
    with torch.no_grad():
        outputs = solver.run_schedule({}, n_frames=N_FRAMES)

    instrument.print_diagnostics()

    audio = _extract_audio(outputs)
    if audio is None:
        raise RuntimeError(f"No audio_out (silent). Keys: {list(outputs)}")

    # Bar-level RMS check
    bar_samples = int(4 * BEAT_S * SR)
    print("\nBar-level RMS:")
    for bar in range(N_BARS):
        ts  = bar * bar_samples
        te  = min(ts + bar_samples, N_FRAMES)
        rms = float(np.abs(audio[ts:te]).mean())
        label = "OK" if rms > 1e-5 else "SILENT"
        print(f"  Bar {bar+1:2d}  RMS={rms:.5f}  {label}")

    import scipy.io.wavfile

    out_path = os.path.abspath("composer_demo.wav")
    scipy.io.wavfile.write(out_path, int(SR), audio)
    print(f"\nPeak (normalised): 0.9200")
    print(f"WAV written: {out_path}")
    print("Done.")


# ── Pygame UI viewer ───────────────────────────────────────────────────────────

def main_ui() -> None:
    """Pygame window: TorchComposerPanel (left) + TorchComposerPianoRoll (right)."""
    import pygame
    pygame.init()
    pygame.mixer.init(frequency=int(SR), size=-16, channels=2, buffer=4096)

    WIN_W, WIN_H = 1400, 800
    PANEL_W      = 560    # left panel width
    ROLL_X       = PANEL_W + 4
    ROLL_W       = WIN_W - ROLL_X
    BAR_H        = 72     # progress bar height at bottom of right pane
    ROLL_H       = WIN_H - BAR_H

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("TorchComposerNode Demo")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 13)

    patch = _load_patch()
    v_mel, v_bass, body_voices, shim_voices, sub_voices = _build_voices(SONG_DUR_S)
    compiled, bank, composer, _projector, _drivers, instrument = _build_solver(
        patch, v_mel, v_bass, body_voices, shim_voices, sub_voices
    )

    solver = compiled.solver

    # Import UI panels (deferred — requires pygame.init() to be done first)
    from torch_composer_panel import TorchComposerPanel, TorchComposerPianoRoll
    from solver_progress_bar import SolverProgressBar

    panel = TorchComposerPanel(composer)
    piano = TorchComposerPianoRoll(composer)

    progress_bar = SolverProgressBar(
        w=ROLL_W,
        h=BAR_H,
        fifo_slot_labels={
            f"{COMPOSER_KEY}_v_mel_out_atoms":  "mel",
            f"{COMPOSER_KEY}_v_bass_out_atoms": "bass",
            f"{COMPOSER_KEY}_{INSTR_KEY}_atoms": "instr",
            f"{INSTR_KEY}_{STR_1_KEY}_atoms":   "str_1",
            f"{INSTR_KEY}_{STR_2_KEY}_atoms":   "str_2",
            f"{INSTR_KEY}_{STR_3_KEY}_atoms":   "str_3",
        },
    )
    progress_bar.bind(solver, fifo_bank=bank)

    # ── Solver helpers ────────────────────────────────────────────────────────

    _render_lock = threading.Lock()
    _audio_cache: dict = {"audio": None}   # cleared on recompose, reused on play/save

    def _render_audio() -> "np.ndarray | None":
        """Render the current (already composed) score.  Does NOT recompose."""
        progress_bar.start_render()
        with torch.no_grad():
            outputs = solver.run_schedule(
                {},
                n_frames=N_FRAMES,
                on_progress=progress_bar.make_progress_callback(),
            )
        instrument.print_diagnostics()
        audio = _extract_audio(outputs)
        if audio is None:
            progress_bar.finish_render(success=False, label="silent or no output")
            return None
        progress_bar.finish_render(success=True)
        _audio_cache["audio"] = audio
        return audio

    def _recompose() -> None:
        _audio_cache["audio"] = None
        solver.reset()
        solver.dispatch_before_start()
        progress_bar.set_status("recomposed")

    def _play() -> None:
        # Use cached audio if available — skip re-render entirely.
        cached = _audio_cache["audio"]
        if cached is not None:
            pcm = (cached * 32767).astype(np.int16)
            pcm_stereo = np.column_stack([pcm, pcm])
            sound = pygame.sndarray.make_sound(pcm_stereo)
            pygame.mixer.stop()
            sound.play()
            progress_bar.set_status(f"playing (cached)  {SONG_DUR_S:.1f}s")
            return

        if not _render_lock.acquire(blocking=False):
            return  # already rendering

        def _worker():
            try:
                audio = _render_audio()
                if audio is None:
                    return
                pcm = (audio * 32767).astype(np.int16)
                pcm_stereo = np.column_stack([pcm, pcm])
                sound = pygame.sndarray.make_sound(pcm_stereo)
                pygame.mixer.stop()
                sound.play()
                progress_bar.set_status(f"playing  {SONG_DUR_S:.1f}s")
            except Exception as e:
                progress_bar.set_status(f"play error: {e}")
            finally:
                _render_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _save() -> None:
        # Use cached audio if available — skip re-render entirely.
        cached = _audio_cache["audio"]
        if cached is not None:
            import scipy.io.wavfile
            out_path = os.path.abspath("composer_demo.wav")
            scipy.io.wavfile.write(out_path, int(SR), cached)
            progress_bar.set_status(f"saved (cached) \u2192 {os.path.basename(out_path)}")
            return

        if not _render_lock.acquire(blocking=False):
            return

        def _worker():
            try:
                audio = _render_audio()
                if audio is None:
                    return
                import scipy.io.wavfile
                out_path = os.path.abspath("composer_demo.wav")
                scipy.io.wavfile.write(out_path, int(SR), audio)
                progress_bar.set_status(f"saved \u2192 {os.path.basename(out_path)}")
            except Exception as e:
                progress_bar.set_status(f"save error: {e}")
            finally:
                _render_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _save_score() -> None:
        """Serialize the current arrangement to a .score.json + .sidecar.npz pair."""
        atoms = list(composer.last_atoms)
        if not atoms:
            progress_bar.set_status("score empty — recompose first")
            return

        def _worker():
            try:
                stem  = os.path.abspath("composer_demo")
                jpath = save_score(
                    atoms,
                    f"{stem}.score.json",
                    sample_rate=SR,
                    n_frames=N_FRAMES,
                    meta={"bpm": BPM, "n_bars": N_BARS, "scale": SCALE_NAME, "tonic": TONIC_NOTE},
                )
                progress_bar.set_status(f"score saved \u2192 {jpath.name}  ({len(atoms)} atoms)")

                # If we already have rendered audio, capture premix from cached audio
                # as a mono aggregate sidecar (per-voice capture requires synthesis hooks).
                cached = _audio_cache.get("audio")
                if cached is not None:
                    cap = VoicePremixCapture(sample_rate=SR, display_fps=120.0)
                    # Store the final mix as a single "mix" voice for the sidecar.
                    cap.add_voice("mix", cached.astype(np.complex64))
                    np_path = cap.save(f"{stem}.sidecar.npz")
                    progress_bar.set_status(
                        f"score + sidecar saved \u2192 {jpath.name} | {np_path.name}"
                    )
            except Exception as e:
                progress_bar.set_status(f"score save error: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    panel.on_recompose = _recompose
    panel.on_play      = _play
    panel.on_save      = _save
    panel.on_save_score = _save_score

    # Initial composition pass — populates resolved_notes for the piano roll
    _recompose()

    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                if mx < PANEL_W:
                    panel.on_mouse_down(ev.button, mx, my)
                else:
                    piano.on_mouse_down(ev.button, mx - ROLL_X, my)
            elif ev.type == pygame.MOUSEBUTTONUP:
                panel.on_mouse_up(ev.button)
                piano.on_mouse_up(ev.button)
            elif ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                if mx < PANEL_W:
                    panel.on_mouse_move(mx, my)
                else:
                    piano.on_mouse_move(mx - ROLL_X, my)
            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx < PANEL_W:
                    panel.on_scroll(ev.x, ev.y, mx, my)
                else:
                    piano.on_mouse_down(4 if ev.y > 0 else 5, mx - ROLL_X, my)
            elif ev.type == pygame.KEYDOWN:
                if not panel.on_key_down(ev.key, ev.mod):
                    piano.on_key_down(ev.key, ev.mod)

        # Render left panel
        panel_surf = panel.render()
        if panel_surf is not None:
            screen.fill((18, 18, 22))
            screen.blit(panel_surf, (0, 0))

        # Render piano roll onto right-side sub-surface (above progress bar)
        roll_surf = screen.subsurface(pygame.Rect(ROLL_X, 0, ROLL_W, ROLL_H))
        roll_surf.fill((24, 24, 30))
        piano.render(roll_surf, font)

        # Divider
        pygame.draw.line(screen, (60, 60, 80), (PANEL_W, 0), (PANEL_W, WIN_H), 2)

        # Progress bar (bottom of right pane)
        progress_bar.render(screen, x=ROLL_X, y=ROLL_H, font=font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--ui" in sys.argv:
        main_ui()
    else:
        main()
