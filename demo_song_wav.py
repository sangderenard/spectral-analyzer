"""demo_song_wav.py  —  "Tempest" scored WAV demo.

Six bars · 120 BPM · D minor · two voices · masked improv bars 3-4.

Uses compile_nodes with direct node construction — this is the target
architecture.  No AnalyticPatch, no materialize_network.

Song structure
--------------
  v1 (melody, groups=("melody",)):
      Bar 1-2 — authored opening motive (ascending / descent)
      Bar 3-4 — masked improv: authored anchors + module-filled passing tones
      Bar 5   — climax return (ff)
      Bar 6   — coda, fade (pp)
  v2 (bass, groups=("bass",)):
      All 6 bars — authored root/fifth pattern, dynamic arc pp→f→pp

Output: tempest_demo.wav  (12 s, 48 000 Hz, mono, normalised PCM-16)
"""
from __future__ import annotations

import os

import numpy as np
import scipy.io.wavfile
import torch

from edge_fifo_bank import EdgeFifoBank
from graph_solver import TensorEdge
from network_materializer import compile_nodes
from score_sequencer_node import ScoreSequencerNode
from voice_graph_node import MetaVoiceNode, MixerSumNode, VoiceTorchOscillator
from parametric_curve import ControlPoint, ParametricCurve
from torch_composer_engine import (
    EVENT_NOTE,
    PARAM_DURATION,
    PARAM_GATE,
    PARAM_HZ,
    PARAM_PATTERN,
    PARAM_START,
    PARAM_STEP,
    PARAM_VELOCITY,
    ScoreWriteMask,
    TorchNoteStream,
    assign_hz_to_score_masked,
    empty_score_tensor,
    sparse_score_tensor_from_score,
)

# ── Song parameters ───────────────────────────────────────────────────────────
BPM        = 120
BEAT_S     = 60.0 / BPM
BAR_S      = BEAT_S * 4
N_BARS     = 6
SONG_DUR_S = BAR_S * N_BARS   # 12.0 s
SR         = 48_000.0
N_FRAMES   = int(SONG_DUR_S * SR)
RELEASE_S  = 0.08

GROUP_MELODY = 0
GROUP_BASS   = 1
GROUP_KEYS   = {GROUP_MELODY: "melody", GROUP_BASS: "bass"}

D3, F3, A3 = 146.83, 174.61, 220.00
D4, F4, G4, A4 = 293.66, 349.23, 392.00, 440.00
C5, D5 = 523.25, 587.33


# ── Envelope helpers ──────────────────────────────────────────────────────────

def _adsr_curve(
    attack: float = 0.06,
    peak_decay: float = 0.18,
    sustain: float = 0.85,
    sustain_end: float = 0.80,
    name: str = "adsr",
) -> ParametricCurve:
    c = ParametricCurve()
    c.name = name
    c.points = [
        ControlPoint(t=0.0,                 v=0.0),
        ControlPoint(t=attack,              v=1.0),
        ControlPoint(t=attack + peak_decay, v=sustain),
        ControlPoint(t=sustain_end,         v=sustain),
        ControlPoint(t=1.0,                 v=0.0),
    ]
    return c


def _pluck_curve(name: str = "pluck") -> ParametricCurve:
    c = ParametricCurve()
    c.name = name
    c.points = [
        ControlPoint(t=0.0,  v=0.0),
        ControlPoint(t=0.02, v=1.0),
        ControlPoint(t=0.25, v=0.52),
        ControlPoint(t=0.70, v=0.40),
        ControlPoint(t=1.0,  v=0.0),
    ]
    return c


# ── Score builders ────────────────────────────────────────────────────────────

def _melody_events() -> list[tuple[float, float, float, float, bool]]:
    return [
        # Bar 1: ascending opening motive (mp)
        (0.00,  D4, 0.22, 0.65, True),
        (0.25,  F4, 0.22, 0.70, True),
        (0.50,  A4, 0.22, 0.75, True),
        (0.75,  D5, 0.22, 0.82, True),
        (1.00,  D5, 0.45, 0.80, True),
        (1.50,  A4, 0.22, 0.73, True),
        (1.75,  G4, 0.22, 0.68, True),
        # Bar 2: descent (mp→mf)
        (2.00,  F4, 0.45, 0.70, True),
        (2.50,  D4, 0.22, 0.65, True),
        (2.75,  A3, 0.22, 0.60, True),
        (3.00,  D4, 0.22, 0.68, True),
        (3.25,  F4, 0.45, 0.72, True),
        (3.75,  A4, 0.22, 0.78, True),
        # Bar 3: anchors + improv (mf)
        (4.00,  D4, 0.22, 0.75, True),
        (4.25,  D4, 0.22, 0.70, False),  # IMPROV
        (4.50,  A4, 0.22, 0.82, True),
        (4.75,  D4, 0.22, 0.70, False),  # IMPROV
        (5.00,  D5, 0.22, 0.88, True),
        (5.25,  D4, 0.22, 0.72, False),  # IMPROV
        (5.50,  A4, 0.22, 0.82, True),
        (5.75,  D4, 0.22, 0.70, False),  # IMPROV
        # Bar 4: development (mf→f)
        (6.00,  G4, 0.22, 0.85, True),
        (6.25,  D4, 0.22, 0.70, False),  # IMPROV
        (6.50,  F4, 0.22, 0.80, True),
        (6.75,  D4, 0.22, 0.70, False),  # IMPROV
        (7.00,  D4, 0.22, 0.75, True),
        (7.25,  D4, 0.22, 0.70, False),  # IMPROV
        (7.50,  A4, 0.22, 0.80, True),
        (7.75,  D4, 0.22, 0.70, False),  # IMPROV
        # Bar 5: climax return (ff)
        (8.00,  D4, 0.22, 0.90, True),
        (8.25,  F4, 0.22, 0.92, True),
        (8.50,  A4, 0.22, 0.95, True),
        (8.75,  D5, 0.22, 0.98, True),
        (9.00,  D5, 0.45, 0.95, True),
        (9.50,  A4, 0.22, 0.88, True),
        (9.75,  G4, 0.22, 0.83, True),
        # Bar 6: coda, fade (ff→pp)
        (10.00, F4, 0.45, 0.78, True),
        (10.50, D4, 0.22, 0.70, True),
        (10.75, A3, 0.22, 0.62, True),
        (11.00, D4, 0.22, 0.55, True),
        (11.25, F4, 0.22, 0.48, True),
        (11.50, D4, 0.45, 0.38, True),
    ]


def _bass_events() -> list[tuple[float, float, float, float]]:
    vel_per_bar = [0.48, 0.50, 0.55, 0.58, 0.68, 0.32]
    events: list[tuple[float, float, float, float]] = []
    for bar in range(N_BARS):
        t0  = bar * BAR_S
        vel = vel_per_bar[bar]
        pattern = (
            [(D3, 0.40), (F3, 0.35), (D3, 0.40), (A3, 0.35)] if bar % 2 == 0
            else [(D3, 0.40), (A3, 0.35), (D3, 0.40), (A3, 0.35)]
        )
        for beat, (hz, dur) in enumerate(pattern):
            events.append((t0 + beat * BEAT_S, hz, dur, vel))
    return events


def _build_combined_score():
    mel = _melody_events()
    bas = _bass_events()
    n_total = len(mel) + len(bas)
    score   = empty_score_tensor(1, n_total)
    authored_mel_flags: list[bool] = []
    ei = 0

    for start, hz, dur, vel, authored in mel:
        score.labels[0, ei]                 = EVENT_NOTE
        score.mask[0, ei]                   = True
        score.params[0, ei, PARAM_START]    = start
        score.params[0, ei, PARAM_DURATION] = dur
        score.params[0, ei, PARAM_HZ]       = hz
        score.params[0, ei, PARAM_VELOCITY] = vel
        score.params[0, ei, PARAM_GATE]     = 1.0
        score.params[0, ei, PARAM_STEP]     = float(ei)
        score.params[0, ei, PARAM_PATTERN]  = float(GROUP_MELODY)
        authored_mel_flags.append(authored)
        ei += 1

    mel_count    = len(mel)
    improv_slots = [i for i, a in enumerate(authored_mel_flags) if not a]

    for start, hz, dur, vel in bas:
        score.labels[0, ei]                 = EVENT_NOTE
        score.mask[0, ei]                   = True
        score.params[0, ei, PARAM_START]    = start
        score.params[0, ei, PARAM_DURATION] = dur
        score.params[0, ei, PARAM_HZ]       = hz
        score.params[0, ei, PARAM_VELOCITY] = vel
        score.params[0, ei, PARAM_GATE]     = 1.0
        score.params[0, ei, PARAM_STEP]     = float(ei)
        score.params[0, ei, PARAM_PATTERN]  = float(GROUP_BASS)
        ei += 1

    write_mask = ScoreWriteMask.like(score, module_names=("note_stream",))
    for i in improv_slots:
        write_mask.hz[0, i] = True

    stream = TorchNoteStream.build(
        degrees=[[D4, F4, G4, A4, C5]],
        deg_pattern=[[2, 4, 1, 3, 0, 4, 2, 3]],
        p_chromatic=0.0,
        p_modal=0.0,
    )
    filled = assign_hz_to_score_masked(score, stream, write_mask)
    filled.metadata.update({
        "title":         "Tempest",
        "melody_events": mel_count,
        "bass_events":   len(bas),
        "improv_slots":  improv_slots,
        "group_keys":    GROUP_KEYS,
    })
    return filled, sparse_score_tensor_from_score(filled)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print('Building "Tempest" — 6 bars · 120 BPM · D minor · 2 voices ...')
    print(f"  {N_FRAMES:,} samples · {SONG_DUR_S:.1f} s · {int(SR)} Hz")

    score, sparse = _build_combined_score()
    n_mel = int(score.metadata["melody_events"])
    n_bas = int(score.metadata["bass_events"])
    n_imp = len(score.metadata["improv_slots"])
    print(f"  Score: {int(score.mask.sum().item())} events "
          f"({n_mel} melody [{n_imp} improv] · {n_bas} bass)")

    # ── Voices — direct construction, no AnalyticPatch ────────────────────
    mel_env  = _adsr_curve(attack=0.06, peak_decay=0.18, sustain=0.85, name="melody_adsr")
    bass_env = _pluck_curve(name="bass_pluck")

    mel_osc = VoiceTorchOscillator(
        freq_hz=D4,
        sample_rate=SR,
        duration=SONG_DUR_S,
        envelope_curve=mel_env,
    )
    bass_osc = VoiceTorchOscillator(
        freq_hz=D3,
        sample_rate=SR,
        duration=SONG_DUR_S,
        envelope_curve=bass_env,
    )

    v1 = MetaVoiceNode("v1", mel_osc, score_contract={
        "kind": "voice_score_consumer",
        "voice_key": "v1",
        "groups": ("melody",),
        "pages": (),
        "aggregation": "mask",
        "envelope_curve": mel_osc.envelope_curve,
        "chirp_curve": mel_osc.chirp_curve,
        "duration_s": SONG_DUR_S,
        "release_tail_s": RELEASE_S,
    })
    v2 = MetaVoiceNode("v2", bass_osc, score_contract={
        "kind": "voice_score_consumer",
        "voice_key": "v2",
        "groups": ("bass",),
        "pages": (),
        "aggregation": "mask",
        "envelope_curve": bass_osc.envelope_curve,
        "chirp_curve": bass_osc.chirp_curve,
        "duration_s": SONG_DUR_S,
        "release_tail_s": 0.12,
    })

    v1_nodes, v1_edges = v1.build_nodes()
    v2_nodes, v2_edges = v2.build_nodes()
    mixer_node = MixerSumNode("__mix__").build_node()

    mix_edge_v1 = TensorEdge(
        src_key="v1_out", dst_key="__mix__",
        weight=1.0 + 0j,
        semantic_role="mix_source",
    )
    mix_edge_v2 = TensorEdge(
        src_key="v2_out", dst_key="__mix__",
        weight=1.0 + 0j,
        semantic_role="mix_source",
    )

    # Bank pre-created so ScoreSequencerNode and score edges share one instance.
    bank = EdgeFifoBank(stride=1)

    seq = ScoreSequencerNode(
        "seq",
        sparse,
        sample_rate=SR,
        fifo_bank=bank,
        group_keys=GROUP_KEYS,
    )
    score_edge_v1 = TensorEdge(
        "seq", "v1_out",
        weight=0.0 + 0.0j,
        semantic_role="score",
        src_port="score_out",
        dst_port="score_in",
        fifo_bank=bank,
    )
    score_edge_v2 = TensorEdge(
        "seq", "v2_out",
        weight=0.0 + 0.0j,
        semantic_role="score",
        src_port="score_out",
        dst_port="score_in",
        fifo_bank=bank,
    )

    compiled = compile_nodes(
        nodes=[seq.build_node(), *v1_nodes, *v2_nodes, mixer_node],
        edges=[*v1_edges, *v2_edges, score_edge_v1, score_edge_v2, mix_edge_v1, mix_edge_v2],
        sample_rate=SR,
    )

    solver = compiled.solver
    print(f"  Nodes: {len(solver.nodes)} · Edges: {len(solver.edges)} "
          f"· Contract edges: {len(solver.contract_edges)}")

    # ── Render ────────────────────────────────────────────────────────────────
    solver.reset()
    solver.dispatch_before_start()

    print(f"Rendering {N_FRAMES:,} samples ...")
    with torch.no_grad():
        outputs = solver.run_schedule({}, n_frames=N_FRAMES)

    mix = outputs.get("__mix__")
    if mix is None:
        raise RuntimeError(f"No '__mix__' in outputs. Keys: {list(outputs)}")
    mix = mix[0, :, 0]

    # ── Bar-level RMS verification ────────────────────────────────────────────
    bar_samples = int(BAR_S * SR)
    bar_names   = [
        "Bar 1 (opening)", "Bar 2 (descent)",
        "Bar 3 (improv)",  "Bar 4 (improv)",
        "Bar 5 (climax)",  "Bar 6 (coda)",
    ]
    print("\nBar-level RMS:")
    for bar in range(N_BARS):
        ts  = bar * bar_samples
        te  = min(ts + bar_samples, N_FRAMES)
        rms = float(mix[ts:te].abs().mean().item())
        print(f"  {bar_names[bar]:20s}  RMS={rms:.5f}  {'OK' if rms > 1e-5 else 'SILENT'}")

    assert mix.dtype == torch.complex128, f"Expected complex128, got {mix.dtype}"

    # ── WAV output ────────────────────────────────────────────────────────────
    audio = mix.real.detach().cpu().numpy().astype(np.float64)
    peak  = float(np.abs(audio).max())
    assert peak > 1e-8, "Rendered silence — check voice/sequencer wiring"
    audio = (audio * (0.92 / peak)).astype(np.float32)

    out_path = os.path.abspath("tempest_demo.wav")
    scipy.io.wavfile.write(out_path, int(SR), audio)
    print(f"\nPeak (normalised): {0.92:.4f}")
    print(f"WAV written: {out_path}")
    print('Done.  "Tempest" complete.')


if __name__ == "__main__":
    main()
