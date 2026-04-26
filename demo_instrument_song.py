"""demo_instrument_song.py — Instrument-UI demo using a pre-solved score.

Loads a .score.json written by demo_composer_song ("📋 Score" button) and
replays it through the same instrument/driver/voice network, but replaces the
TorchComposerNode with a ScoreLoaderNode so no re-composition happens.

Layout
------
  Left  (300 px)  — InstrumentPanel (top half)
                    BodyResonatorPanel (bottom half)
  Center           — PassiveCurveDisplay (envelope / chirp / pre-mix GL)
  Right (300 px)  — SympatheticCouplingPanel (full height, scrollable)
  Bottom (72 px)  — SolverProgressBar

Usage
-----
  python demo_instrument_song.py                              # looks for composer_demo.score.json
  python demo_instrument_song.py my_arrangement.score.json   # explicit path
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time as _time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

from analytic_model import AnalyticPatch
from audio_projector_node import AudioProjectorNode
from driver_node import DriverNode, VoiceSpread
from edge_fifo_bank import EdgeFifoBank
from graph_solver import TensorEdge
from instrument_node import InstrumentNode
from network_materializer import compile_nodes
from parametric_curve import ControlPoint, ParametricCurve
from body_resonator_panel import BodyResonatorPanel
from instrument_panel import InstrumentPanel
from passive_curve_display import PassiveCurveDisplay
from resonator_core import ResonatorString, StringCouplingConfig
from score_loader_node import ScoreLoaderNode
from score_persist import VoicePremixCapture
from solver_progress_bar import SolverProgressBar
from sympathetic_coupling_panel import SympatheticCouplingPanel
from voice_graph_node import MetaVoiceNode, MixerSumNode, VoiceTorchOscillator

# ── Song constants (must match the saved score) ───────────────────────────────

SR        = 48_000.0
BPM       = 120
BEAT_S    = 60.0 / BPM
N_BARS    = 8
SONG_DUR_S = N_BARS * 4 * BEAT_S   # 16.0 s
N_FRAMES  = int(SONG_DUR_S * SR)
RELEASE_S = 0.08

COMPOSER_KEY = "composer"
INSTR_KEY    = "instrument"
STR_1_KEY    = "str_1"
STR_2_KEY    = "str_2"
STR_3_KEY    = "str_3"

# ── Note helpers (mirrors demo_composer_song) ─────────────────────────────────

import re as _re
_PC: dict = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def _note_hz(name: str, concert_a: float = 440.0) -> float:
    from analytic_model import GlobalTuning
    m = _re.match(r'([A-G][#b]?)(-?\d+)', name)
    if not m:
        return 440.0
    midi  = _PC[m.group(1)] + (int(m.group(2)) + 1) * 12
    semis = midi - 69
    return GlobalTuning(root_hz=concert_a).semitone_to_hz(semis)


# ── Envelope curves (same as demo_composer_song) ──────────────────────────────

def _melody_env() -> ParametricCurve:
    c = ParametricCurve(); c.name = "melody_adsr"
    c.points = [ControlPoint(t=0.00, v=0.0), ControlPoint(t=0.05, v=1.0),
                ControlPoint(t=0.22, v=0.82), ControlPoint(t=0.80, v=0.78),
                ControlPoint(t=1.00, v=0.0)]
    return c


def _bass_env() -> ParametricCurve:
    c = ParametricCurve(); c.name = "bass_pluck"
    c.points = [ControlPoint(t=0.00, v=0.0), ControlPoint(t=0.02, v=1.0),
                ControlPoint(t=0.28, v=0.55), ControlPoint(t=0.75, v=0.38),
                ControlPoint(t=1.00, v=0.0)]
    return c


def _body_env() -> ParametricCurve:
    c = ParametricCurve(); c.name = "body"
    c.points = [ControlPoint(t=0.00, v=0.0), ControlPoint(t=0.28, v=0.75),
                ControlPoint(t=0.55, v=0.70), ControlPoint(t=0.82, v=0.65),
                ControlPoint(t=1.00, v=0.0)]
    return c


def _shim_env() -> ParametricCurve:
    c = ParametricCurve(); c.name = "shimmer"
    c.points = [ControlPoint(t=0.00, v=0.0), ControlPoint(t=0.06, v=0.80),
                ControlPoint(t=0.22, v=0.60), ControlPoint(t=0.65, v=0.50),
                ControlPoint(t=1.00, v=0.0)]
    return c


def _sub_env() -> ParametricCurve:
    c = ParametricCurve(); c.name = "sub"
    c.points = [ControlPoint(t=0.00, v=0.0), ControlPoint(t=0.40, v=0.55),
                ControlPoint(t=0.65, v=0.52), ControlPoint(t=0.88, v=0.45),
                ControlPoint(t=1.00, v=0.0)]
    return c


# ── Voice construction (mirrors demo_composer_song voice types) ────────────────

def _voice(key: str, osc: VoiceTorchOscillator, group: str,
           dur_s: float, rel_s: float) -> MetaVoiceNode:
    return MetaVoiceNode(key, osc, score_contract={
        "kind":           "voice_score_consumer",
        "voice_key":      key,
        "groups":         (group,),
        "pages":          (),
        "aggregation":    "mask",
        "envelope_curve": osc.envelope_curve,
        "chirp_curve":    osc.chirp_curve,
        "duration_s":     dur_s,
        "release_tail_s": rel_s,
    })


def _build_voices(dur_s: float) -> tuple:
    tonic = _note_hz("D4")
    mel_osc  = VoiceTorchOscillator(freq_hz=tonic, sample_rate=SR, duration=dur_s,
                                    envelope_curve=_melody_env())
    bass_osc = VoiceTorchOscillator(freq_hz=tonic * 0.25, sample_rate=SR, duration=dur_s,
                                    envelope_curve=_bass_env())
    body_root = tonic * 0.5
    body_oscs = [
        VoiceTorchOscillator(freq_hz=body_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=5,
                             harmonic_brightness=0.55, envelope_curve=_body_env()),
        VoiceTorchOscillator(freq_hz=body_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=4,
                             harmonic_brightness=0.40, envelope_curve=_body_env()),
        VoiceTorchOscillator(freq_hz=body_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=6,
                             harmonic_brightness=0.70, envelope_curve=_body_env()),
    ]
    shim_root = tonic
    shim_oscs = [
        VoiceTorchOscillator(freq_hz=shim_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=12,
                             harmonic_brightness=1.80, envelope_curve=_shim_env()),
        VoiceTorchOscillator(freq_hz=shim_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic_warp", harmonic_count=10,
                             harmonic_brightness=1.40, harmonic_warp_strength=0.12,
                             envelope_curve=_shim_env()),
        VoiceTorchOscillator(freq_hz=shim_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic_warp", harmonic_count=16,
                             harmonic_brightness=1.20, harmonic_warp_strength=0.28,
                             envelope_curve=_shim_env()),
    ]
    sub_root = tonic * 0.125
    sub_oscs = [
        VoiceTorchOscillator(freq_hz=sub_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=2,
                             harmonic_brightness=0.20, envelope_curve=_sub_env()),
        VoiceTorchOscillator(freq_hz=sub_root, sample_rate=SR, duration=dur_s,
                             manifold_type="harmonic", harmonic_count=3,
                             harmonic_brightness=0.30, envelope_curve=_sub_env()),
    ]

    v_mel  = _voice("v_mel",  mel_osc,  "melody", dur_s, RELEASE_S)
    v_bass = _voice("v_bass", bass_osc, "bass",   dur_s, 0.12)
    body_v = [_voice(f"v_body_{i+1}", o, "pad", dur_s, 0.25) for i, o in enumerate(body_oscs)]
    shim_v = [_voice(f"v_shim_{i+1}", o, "pad", dur_s, 0.15) for i, o in enumerate(shim_oscs)]
    sub_v  = [_voice(f"v_sub_{i+1}",  o, "pad", dur_s, 0.40) for i, o in enumerate(sub_oscs)]
    return v_mel, v_bass, body_v, shim_v, sub_v


# ── Solver wiring ──────────────────────────────────────────────────────────────

def _score_edge(src: str, dst: str, bank: EdgeFifoBank) -> TensorEdge:
    return TensorEdge(src, dst, weight=0j, semantic_role="score",
                      src_port="score_out", dst_port="score_in", fifo_bank=bank)


def _sig_edges(srcs: List[str], dst: str) -> List[TensorEdge]:
    return [TensorEdge(s, dst, weight=1+0j, semantic_role="mix_source") for s in srcs]


def _build_solver(score_path: Path, v_mel, v_bass, body_v, shim_v, sub_v):
    bank = EdgeFifoBank(stride=1)
    tonic = _note_hz("D4")

    loader = ScoreLoaderNode(COMPOSER_KEY, score_path, fifo_bank=bank, layer="score")

    str_1 = DriverNode(STR_1_KEY, groups=("pad",), duration_s=SONG_DUR_S, release_tail_s=0.25,
                       spreads=[
                           VoiceSpread(n_samples=6,  phase_std=0.60, amplitude_std=0.16, time_std_s=0.080),
                           VoiceSpread(n_samples=6,  phase_std=0.80, amplitude_std=0.20, time_std_s=0.070),
                           VoiceSpread(n_samples=4,  phase_std=0.50, amplitude_std=0.12, time_std_s=0.120),
                       ], sample_rate=SR, layer="voice")
    str_2 = DriverNode(STR_2_KEY, groups=("pad",), duration_s=SONG_DUR_S, release_tail_s=0.15,
                       spreads=[
                           VoiceSpread(n_samples=8,  phase_std=1.20, amplitude_std=0.30, time_std_s=0.040),
                           VoiceSpread(n_samples=8,  phase_std=1.00, amplitude_std=0.24, time_std_s=0.050),
                           VoiceSpread(n_samples=12, phase_std=1.60, amplitude_std=0.40, time_std_s=0.030),
                       ], sample_rate=SR, layer="voice")
    str_3 = DriverNode(STR_3_KEY, groups=("pad",), duration_s=SONG_DUR_S, release_tail_s=0.40,
                       spreads=[
                           VoiceSpread(n_samples=4,  phase_std=0.30, amplitude_std=0.10, time_std_s=0.160),
                           VoiceSpread(n_samples=4,  phase_std=0.40, amplitude_std=0.12, time_std_s=0.140),
                       ], sample_rate=SR, layer="voice")

    instrument = InstrumentNode(
        INSTR_KEY,
        driver_keys=[STR_1_KEY, STR_2_KEY, STR_3_KEY],
        resonator_strings=[
            ResonatorString(key="s1", fundamental_hz=tonic*0.5, decay_s=1.8,
                            drive_gain=0.90, x=0.30, y=-0.10),
            ResonatorString(key="s2", fundamental_hz=tonic,     decay_s=0.8,
                            drive_gain=1.00, x=0.50, y= 0.05),
            ResonatorString(key="s3", fundamental_hz=tonic*0.125, decay_s=3.0,
                            drive_gain=0.85, x=0.70, y= 0.00),
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

    v_mel_nodes,  v_mel_edges  = v_mel.build_nodes()
    v_bass_nodes, v_bass_edges = v_bass.build_nodes()
    all_body_nodes, all_body_edges = [], []
    all_shim_nodes, all_shim_edges = [], []
    all_sub_nodes,  all_sub_edges  = [], []
    for v in body_v:
        ns, es = v.build_nodes(); all_body_nodes += ns; all_body_edges += es
    for v in shim_v:
        ns, es = v.build_nodes(); all_shim_nodes += ns; all_shim_edges += es
    for v in sub_v:
        ns, es = v.build_nodes(); all_sub_nodes  += ns; all_sub_edges  += es

    mixer_node     = MixerSumNode("__mix__").build_node()
    projector      = AudioProjectorNode("audio_out", sample_rate=SR, projection="real")
    projector_node = projector.build_node()

    body_out_keys = [f"{v.key}_out" for v in body_v]
    shim_out_keys = [f"{v.key}_out" for v in shim_v]
    sub_out_keys  = [f"{v.key}_out" for v in sub_v]

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
    sig_edges = [
        *_sig_edges(body_out_keys, STR_1_KEY),
        *_sig_edges(shim_out_keys, STR_2_KEY),
        *_sig_edges(sub_out_keys,  STR_3_KEY),
        TensorEdge(STR_1_KEY, INSTR_KEY, weight=1+0j, semantic_role="mix_source"),
        TensorEdge(STR_2_KEY, INSTR_KEY, weight=1+0j, semantic_role="mix_source"),
        TensorEdge(STR_3_KEY, INSTR_KEY, weight=1+0j, semantic_role="mix_source"),
    ]
    mix_edges = [
        TensorEdge("v_mel_out",  "__mix__",   weight=1.00+0j, semantic_role="mix_source"),
        TensorEdge("v_bass_out", "__mix__",   weight=0.80+0j, semantic_role="mix_source"),
        TensorEdge(INSTR_KEY,    "__mix__",   weight=0.55+0j, semantic_role="mix_source"),
        TensorEdge("__mix__",    "audio_out", weight=1.00+0j, semantic_role="audio_projection_source"),
    ]

    compiled = compile_nodes(
        nodes=[
            loader.build_node(),
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
    all_voice_keys = (
        ["v_mel", "v_bass"]
        + [v.key for v in body_v]
        + [v.key for v in shim_v]
        + [v.key for v in sub_v]
    )
    return compiled, bank, loader, instrument, all_voice_keys


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _extract_audio(outputs: dict) -> Optional[np.ndarray]:
    from audio_projector_node import _time_series_1d
    raw = outputs.get("audio_out")
    if raw is None:
        return None
    audio = _time_series_1d(raw, part="real").float().numpy()
    peak = float(np.abs(audio).max())
    return None if peak < 1e-8 else (audio * (0.92 / peak)).astype(np.float32)


def _capture_premix(outputs: dict, voice_keys: List[str],
                    sr: float = SR, display_fps: float = 120.0,
                    gl_fps: float = 4000.0) -> Optional[dict]:
    """Build a sidecar dict from voice output tensors in the solver outputs."""
    cap = VoicePremixCapture(sample_rate=sr, display_fps=display_fps, gl_fps=gl_fps)
    for vk in voice_keys:
        out_key = f"{vk}_out"
        if out_key in outputs:
            t = outputs[out_key]
            arr = t.detach().cpu().numpy().squeeze()
            cap.add_voice(vk, arr)
    tmp = Path("_tmp_sidecar.npz")
    cap.save(tmp)
    try:
        return VoicePremixCapture.load(tmp)
    except Exception:
        return None
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


# ── Pygame UI ─────────────────────────────────────────────────────────────────

_C_BG = (18, 18, 22)

def main_ui(score_path: Path) -> None:
    import pygame
    pygame.init()
    pygame.mixer.init(frequency=int(SR), size=-16, channels=2, buffer=4096)

    WIN_W, WIN_H = 1400, 840
    SIDE_W = 300
    BAR_H  = 72
    CTR_X  = SIDE_W + 2
    CTR_W  = WIN_W - SIDE_W * 2 - 4
    MAIN_H = WIN_H - BAR_H
    LEFT_TOP_H = MAIN_H // 2      # InstrumentPanel height
    LEFT_BOT_H = MAIN_H - LEFT_TOP_H  # BodyResonatorPanel height

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"Instrument Demo — {score_path.name}")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 11)
    font13 = pygame.font.SysFont("monospace", 13)

    # ── Build network ─────────────────────────────────────────────────────────
    v_mel, v_bass, body_v, shim_v, sub_v = _build_voices(SONG_DUR_S)
    compiled, bank, loader, instrument, all_voice_keys = _build_solver(
        score_path, v_mel, v_bass, body_v, shim_v, sub_v
    )
    solver = compiled.solver

    # ── Panels ────────────────────────────────────────────────────────────────
    instr_panel   = InstrumentPanel(instrument)
    body_panel    = BodyResonatorPanel(instrument)
    coupling_panel = SympatheticCouplingPanel(instrument)

    # Wire rebuild callbacks so panels can clear the cached body scene
    def _on_rebuild():
        _audio_cache["audio"] = None

    instr_panel.on_rebuild   = _on_rebuild
    body_panel.on_rebuild    = _on_rebuild

    # Load sidecar if it exists alongside the score
    sidecar_path = score_path.with_suffix("").with_suffix(".sidecar.npz")
    initial_sidecar = None
    if sidecar_path.exists():
        try:
            initial_sidecar = VoicePremixCapture.load(sidecar_path)
        except Exception:
            pass

    # Load atoms from the score file for display
    from score_persist import load_score
    loaded_atoms, loaded_meta = load_score(score_path)

    curve_disp = PassiveCurveDisplay(
        atoms=loaded_atoms,
        sidecar=initial_sidecar,
        total_s=float(loaded_meta.get("n_frames", N_FRAMES)) / SR,
        prefer_voice_keys=["v_mel", "v_bass"],
    )

    fifo_labels: dict = {
        f"{COMPOSER_KEY}_v_mel_out_atoms":  "mel",
        f"{COMPOSER_KEY}_v_bass_out_atoms": "bass",
        f"{COMPOSER_KEY}_{INSTR_KEY}_atoms": "instr",
        f"{INSTR_KEY}_{STR_1_KEY}_atoms":   "str_1",
        f"{INSTR_KEY}_{STR_2_KEY}_atoms":   "str_2",
        f"{INSTR_KEY}_{STR_3_KEY}_atoms":   "str_3",
    }
    progress_bar = SolverProgressBar(w=CTR_W, h=BAR_H, fifo_slot_labels=fifo_labels)
    progress_bar.bind(solver, fifo_bank=bank)

    # ── State ─────────────────────────────────────────────────────────────────
    _render_lock   = threading.Lock()
    _audio_cache:  dict = {"audio": None}
    _sidecar_cache: dict = {"data": initial_sidecar}
    _play_start:   dict = {"t": None}   # monotonic time when playback started

    total_s = float(loaded_meta.get("n_frames", N_FRAMES)) / SR

    def _render_audio() -> Optional[np.ndarray]:
        progress_bar.start_render()
        with torch.no_grad():
            outputs = solver.run_schedule(
                {}, n_frames=N_FRAMES,
                on_progress=progress_bar.make_progress_callback(),
            )
        instrument.print_diagnostics()
        audio = _extract_audio(outputs)
        if audio is None:
            progress_bar.finish_render(success=False, label="silent or no output")
            return None
        progress_bar.finish_render(success=True)
        _audio_cache["audio"] = audio

        # Capture per-voice premix from solver outputs (at high GL rate)
        try:
            sd = _capture_premix(outputs, all_voice_keys, gl_fps=4000.0)
            if sd is not None:
                _sidecar_cache["data"] = sd
                curve_disp.set_sidecar(sd)
                # Persist alongside the score
                stem = str(score_path.with_suffix("").with_suffix(""))
                cap  = VoicePremixCapture(sample_rate=SR, display_fps=120.0, gl_fps=4000.0)
                for vk in all_voice_keys:
                    out_key = f"{vk}_out"
                    if out_key in outputs:
                        arr = outputs[out_key].detach().cpu().numpy().squeeze()
                        cap.add_voice(vk, arr)
                cap.save(Path(f"{stem}.sidecar.npz"))
        except Exception as e:
            print(f"[sidecar] capture error: {e}")
        return audio

    def _reload() -> None:
        """Reset solver with the current score (re-dispatch without recomposing)."""
        _audio_cache["audio"] = None
        solver.reset()
        solver.dispatch_before_start()
        progress_bar.set_status("reloaded")

    def _play() -> None:
        cached = _audio_cache["audio"]
        if cached is not None:
            _do_play(cached)
            return
        if not _render_lock.acquire(blocking=False):
            return
        def _worker():
            try:
                audio = _render_audio()
                if audio is not None:
                    _do_play(audio)
            finally:
                _render_lock.release()
        threading.Thread(target=_worker, daemon=True).start()

    def _do_play(audio: np.ndarray) -> None:
        pcm = (audio * 32767).astype(np.int16)
        pcm_stereo = np.column_stack([pcm, pcm])
        sound = pygame.sndarray.make_sound(pcm_stereo)
        pygame.mixer.stop()
        sound.play()
        _play_start["t"] = _time.monotonic()
        progress_bar.set_status(f"playing  {total_s:.1f}s")

    def _save_wav() -> None:
        cached = _audio_cache["audio"]
        if cached is None:
            if not _render_lock.acquire(blocking=False):
                return
            def _w():
                try:
                    audio = _render_audio()
                    if audio is not None:
                        _write_wav(audio)
                finally:
                    _render_lock.release()
            threading.Thread(target=_w, daemon=True).start()
        else:
            _write_wav(cached)

    def _write_wav(audio: np.ndarray) -> None:
        try:
            import scipy.io.wavfile
            out = os.path.abspath("instrument_demo.wav")
            scipy.io.wavfile.write(out, int(SR), audio)
            progress_bar.set_status(f"saved \u2192 {os.path.basename(out)}")
        except Exception as e:
            progress_bar.set_status(f"save error: {e}")

    # Initial load
    _reload()

    RES_X = CTR_X + CTR_W   # x origin of right (coupling) panel

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    _reload()
                elif ev.key == pygame.K_SPACE:
                    _play()
                elif ev.key == pygame.K_s:
                    _save_wav()
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                if mx < SIDE_W:
                    if my < LEFT_TOP_H:
                        instr_panel.on_mouse_down(mx, my)
                    else:
                        body_panel.on_mouse_down(mx, my)
                elif mx >= RES_X:
                    coupling_panel.on_mouse_down(mx, my)
                else:
                    curve_disp.on_mouse_down(ev.button, mx - CTR_X, my)
            elif ev.type == pygame.MOUSEBUTTONUP:
                instr_panel.on_mouse_up()
                body_panel.on_mouse_up()
                coupling_panel.on_mouse_up()
            elif ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                instr_panel.on_mouse_move(mx, my)
                body_panel.on_mouse_move(mx, my)
                coupling_panel.on_mouse_move(mx, my)
            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                coupling_panel.on_scroll(ev.y, mx, my)

        # Update playhead from playback clock
        if _play_start["t"] is not None and pygame.mixer.get_busy():
            elapsed = _time.monotonic() - _play_start["t"]
            curve_disp.playhead_t = min(1.0, elapsed / max(total_s, 1e-3))

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill(_C_BG)

        # Left column: InstrumentPanel (top) + BodyResonatorPanel (bottom)
        instr_panel.render(screen, 0, 0, SIDE_W, LEFT_TOP_H, font)
        body_panel.render(screen, 0, LEFT_TOP_H, SIDE_W, LEFT_BOT_H, font)
        pygame.draw.line(screen, (60, 60, 80), (SIDE_W, 0), (SIDE_W, MAIN_H), 2)

        # Right column: SympatheticCouplingPanel
        coupling_panel.render(screen, RES_X, 0, SIDE_W, MAIN_H, font)
        pygame.draw.line(screen, (60, 60, 80), (RES_X, 0), (RES_X, MAIN_H), 2)

        # Center: passive curve display
        ctr_surf = curve_disp.render(CTR_W, MAIN_H)
        screen.blit(ctr_surf, (CTR_X, 0))

        # Progress bar + controls
        progress_bar.render(screen, x=CTR_X, y=MAIN_H, font=font13)

        # Hotkey hint at bottom-left
        hint = "R=reload  Space=play  S=save WAV"
        htxt = font.render(hint, True, (80, 80, 100))
        screen.blit(htxt, (4, WIN_H - font.get_height() - 2))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    score_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("composer_demo.score.json")
    if not score_path.exists():
        print(f"Score file not found: {score_path}")
        print("Run demo_composer_song.py --ui and click '📋 Score' to generate it.")
        sys.exit(1)
    main_ui(score_path)


if __name__ == "__main__":
    main()
