"""demo_instrument_song.py — Instrument-UI demo using a pre-solved score.

Loads a .score.json written by demo_composer_song ("📋 Score" button) and
replays it through the same instrument/driver/voice network, but replaces the
TorchComposerNode with a ScoreLoaderNode so no re-composition happens.

Layout
------
  Left  (300 px)  — InstrumentPanel (top half)
                    BodyResonatorPanel (bottom half)
  Center           — ParametricCurveEditor (phasor_cloud / string_waveform / body_volume)
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
from parametric_curve_editor import ParametricCurveEditor
from passive_curve_display import build_cavity_pressure_volume
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

def _save_full_sidecar(sidecar: dict, path: Path) -> None:
    """Save a sidecar dict (including all extra keys) to a .npz file."""
    save_kw: dict = {}
    for k, v in sidecar.items():
        if isinstance(v, np.ndarray):
            save_kw[k] = v
        elif isinstance(v, (int, float)):
            save_kw[k] = np.array([v], dtype=np.float64)
        elif isinstance(v, list):
            save_kw[k] = np.array(v, dtype=object)
    try:
        np.savez_compressed(str(path), **save_kw)
    except Exception as e:
        print(f"[sidecar] save error: {e}")


def _load_full_sidecar(path: Path) -> Optional[dict]:
    """Load a full sidecar .npz, restoring scalars and object arrays."""
    if not path.exists():
        return None
    try:
        data = np.load(str(path), allow_pickle=True)
    except Exception as e:
        print(f"[sidecar] load error: {e}")
        return None
    out: dict = {}
    for k in data.files:
        v = data[k]
        if v.ndim == 1 and v.shape[0] == 1 and v.dtype.kind == "f":
            out[k] = float(v[0])
        elif v.dtype == object:
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _extract_audio(outputs: dict) -> Optional[np.ndarray]:
    from audio_projector_node import _time_series_1d
    raw = outputs.get("audio_out")
    if raw is None:
        return None
    audio = _time_series_1d(raw, part="real").float().numpy()
    peak = float(np.abs(audio).max())
    return None if peak < 1e-8 else (audio * (0.92 / peak)).astype(np.float32)


def _capture_premix(outputs: dict, voice_keys: List[str],
                    instrument: Any,
                    sr: float = SR, display_fps: float = 120.0,
                    gl_fps: float = 4000.0,
                    progress_cb=None) -> Optional[dict]:
    """Build a sidecar dict from voice output tensors in the solver outputs.

    progress_cb(panel_id, label, fraction) — optional, called from this thread
    to report loading state into each visualisation panel.

    Extends the base VoicePremixCapture sidecar with instrument-demo extras:
      pre_body_gl        — (1, n_gl) complex64, sum of driver outputs (phasor cloud)
      string_gl          — (3, n_gl) complex64, per-string waveforms (sympathy panel)
      ray_segments       — (N, 12) float32, ray tracer segment buffer (body volume panel)
      ray_meta_*         — scalars: max_path_length, n_sources, n_bands
      body_resonance_gl  — (1, n_gl) complex128→complex64, H(f)-filtered body output
                           (stored for future use; not routed to any panel yet)
      resonator_volume   — (n_time, n_spatial, n_spatial) float32, fallback when
                           C ray tracer extension is not built
    """
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
        sidecar = VoicePremixCapture.load(tmp)
    except Exception:
        return None
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

    step = max(1, int(sr / gl_fps))

    # ── Per-string waveforms (string_gl) ──────────────────────────────────────
    if progress_cb: progress_cb("mid", "downsampling string signals…", 0.1)
    str_arrays = []
    for key in (STR_1_KEY, STR_2_KEY, STR_3_KEY):
        if key in outputs:
            arr = outputs[key].detach().cpu().numpy().squeeze()
            if arr.ndim > 1:
                arr = arr[0]
            str_arrays.append(arr[::step].astype(np.complex64))
        else:
            str_arrays.append(None)

    if any(a is not None for a in str_arrays):
        valid = [a for a in str_arrays if a is not None]
        min_len = min(len(a) for a in valid)
        aligned = [
            (a[:min_len] if a is not None else np.zeros(min_len, dtype=np.complex64))
            for a in str_arrays
        ]
        sidecar["string_gl"] = np.array(aligned, dtype=np.complex64)

    # ── String physical positions + coupling matrix ───────────────────────────
    if progress_cb: progress_cb("mid", "reading coupling matrix…", 0.45)
    res_strings = getattr(instrument, "resonator_strings", [])
    if res_strings:
        sidecar["string_positions"] = np.array(
            [[float(s.x), float(s.y)] for s in res_strings], dtype=np.float32
        )  # (n_strings, 2)
    coupling_raw = getattr(instrument, "_coupling_matrix", None)
    if coupling_raw is not None:
        try:
            import torch as _torch
            if isinstance(coupling_raw, _torch.Tensor):
                coupling_raw = coupling_raw.detach().cpu().numpy()
        except ImportError:
            pass
        sidecar["coupling_matrix"] = np.abs(
            np.asarray(coupling_raw)
        ).astype(np.float32)  # (n_strings, n_strings) magnitudes

    if progress_cb: progress_cb("mid", "done", 1.0)

    # ── Pre-body mix (pre_body_gl) + full-rate real signal for H(f) ──────────
    # DESIGN INTENT (not yet implemented):
    # pre_body_gl should carry one row per atom that was generated, not just the
    # summed driver outputs.  Each row is the waveform that atom contributed to
    # the pre-body mix, time-aligned so that when the phase-cloud widget overlays
    # them they travel through phase space in perfect lockstep with their position
    # in the summed signal.  The purpose is to see the near-field driver phase
    # space: how individual atoms cluster, interfere, and organise before entering
    # the body resonator — i.e. the complex phase portrait of the source field
    # rather than the collapsed scalar sum.
    # Currently this only sums STR_1/STR_2/STR_3 driver outputs into a single
    # row (shape 1 × n_gl), which gives the phase cloud something to show but
    # loses all per-atom structure.  Fixing this requires collecting per-atom
    # complex waveforms from the score replay and stacking them here.
    if progress_cb: progress_cb("top", "building pre-body mix…", 0.3)
    pre_body_parts = []
    pre_body_full_parts = []
    for key in (STR_1_KEY, STR_2_KEY, STR_3_KEY):
        if key in outputs:
            arr = outputs[key].detach().cpu().numpy().squeeze()
            if arr.ndim > 1:
                arr = arr[0]
            pre_body_parts.append(arr.astype(np.complex64))
            pre_body_full_parts.append(arr.astype(np.complex128))
    if pre_body_parts:
        min_len    = min(len(a) for a in pre_body_parts)
        pre_body   = sum(a[:min_len] for a in pre_body_parts)
        sidecar["pre_body_gl"] = pre_body[::step].reshape(1, -1).astype(np.complex64)
    pre_body_full: Optional[np.ndarray] = None
    if pre_body_full_parts:
        min_len       = min(len(a) for a in pre_body_full_parts)
        pre_body_full = sum(a[:min_len] for a in pre_body_full_parts)
    if progress_cb: progress_cb("top", "done", 1.0)

    # ── Physical body acoustics: ray tracing ─────────────────────────────────
    if progress_cb: progress_cb("bot", "importing ray tracer…", 0.05)
    try:
        from ray_tracer_bridge import (
            trace_cavity_scene as _trace_scene,
            extract_scene_geometry as _extract_geo,
        )
        from ray_tracer_bridge import _HAS_C_TRACER
    except Exception as _e:
        _HAS_C_TRACER = False
        _extract_geo  = None
        print(f"[ray_tracer_bridge] import error: {_e}")

    body_scene = getattr(instrument, "_body_scene", None)

    # ── Geometry extraction (always, no C tracer required) ────────────────────
    if body_scene is not None and _extract_geo is not None:
        try:
            _geo_verts, _geo_norms = _extract_geo(body_scene)
            if len(_geo_verts):
                sidecar["ray_geo_verts_flat"] = _geo_verts
                sidecar["ray_geo_normals"]    = _geo_norms
                sidecar["ray_surface_illum"]  = np.zeros(len(_geo_verts), dtype=np.float32)
        except Exception as _e:
            print(f"[ray_tracer] geometry extract error: {_e}")

    # ── Segment buffer for RayAccumulatorWidget visualization ────────────────
    if _HAS_C_TRACER and body_scene is not None:
        if progress_cb: progress_cb("bot", "tracing acoustic rays…", 0.15)
        try:
            ray_segs, ray_meta = _trace_scene(
                body_scene,
                n_rays=256, max_bounces=8, n_bands=12,
                min_amplitude=0.005, speed_m_s=343.0, seed=42,
            )
            sidecar["ray_segments"]       = ray_segs
            sidecar["ray_meta_max_path"]  = np.array([ray_meta["max_path_length"]])
            sidecar["ray_meta_n_sources"] = np.array([ray_meta["n_sources"]], dtype=np.int32)
            sidecar["ray_meta_n_bands"]   = np.array([ray_meta["n_bands"]],   dtype=np.int32)
            if ray_meta.get("geo_verts_flat") is not None:
                # Overwrite with illuminated version from the full trace.
                sidecar["ray_geo_verts_flat"] = ray_meta["geo_verts_flat"]
                sidecar["ray_geo_normals"]    = ray_meta["geo_normals"]
                sidecar["ray_surface_illum"]  = ray_meta["surface_illum"]
        except Exception as _e:
            print(f"[ray_tracer] trace error: {_e}")

    if progress_cb: progress_cb("bot", "done", 1.0)

    return sidecar


# ── Pygame UI ─────────────────────────────────────────────────────────────────

_C_BG = (18, 18, 22)

def main_ui(score_path: Path) -> None:
    import pygame
    pygame.init()
    pygame.mixer.init(frequency=int(SR), size=-16, channels=2, buffer=4096)

    WIN_W, WIN_H = 1400, 840
    SIDE_W = 300
    BAR_H  = 72

    def _layout(ww, wh):
        ctr_x      = SIDE_W + 2
        ctr_w      = max(100, ww - SIDE_W * 2 - 4)
        main_h     = max(200, wh - BAR_H)
        left_top_h = main_h // 2
        left_bot_h = main_h - left_top_h
        res_x      = ctr_x + ctr_w
        return ctr_x, ctr_w, main_h, left_top_h, left_bot_h, res_x

    CTR_X, CTR_W, MAIN_H, LEFT_TOP_H, LEFT_BOT_H, RES_X = _layout(WIN_W, WIN_H)

    # Use OpenGL display so GL shader widgets can draw directly into the
    # framebuffer.  Falls back to a plain surface if PyOpenGL isn't installed.
    try:
        from opengl_widget import SurfaceBlitter as _SurfaceBlitter, _HAS_GL as _opengl_ok
    except Exception:
        _SurfaceBlitter = None  # type: ignore[assignment,misc]
        _opengl_ok = False

    _gl_flags = (pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE) if _opengl_ok else pygame.RESIZABLE
    screen = pygame.display.set_mode((WIN_W, WIN_H), _gl_flags)
    pygame.display.set_caption(f"Instrument Demo — {score_path.name}")
    # In OPENGL mode the screen surface cannot be drawn to directly; use a
    # separate surface for all 2-D rendering and blit it via SurfaceBlitter.
    offscreen = pygame.Surface((WIN_W, WIN_H)) if _opengl_ok else screen
    _blitter  = _SurfaceBlitter() if _opengl_ok else None

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

    # Wire rebuild callbacks — panel rebuilds don't invalidate rendered audio;
    # only R explicitly re-renders.
    def _on_rebuild():
        pass

    instr_panel.on_rebuild   = _on_rebuild
    body_panel.on_rebuild    = _on_rebuild

    # Load sidecar if it exists alongside the score
    sidecar_path = score_path.with_suffix("").with_suffix(".sidecar.npz")
    initial_sidecar = _load_full_sidecar(sidecar_path)

    # Load atoms from the score file for display
    from score_persist import load_score
    _loaded_atoms, loaded_meta = load_score(score_path)

    curve_disp = ParametricCurveEditor(
        w=CTR_W, h=MAIN_H,
        panels=[
            {"role": "phasor_cloud"},
            {"role": "string_waveform"},
            {"role": "body_volume"},
        ]
    )
    curve_disp.set_total_s(float(loaded_meta.get("n_frames", N_FRAMES)) / SR)
    if initial_sidecar is not None:
        curve_disp.set_sidecar(initial_sidecar)

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

        # Capture per-voice premix + ray-trace visualization sidecar
        try:
            progress_bar.set_status("processing…")
            sd = _capture_premix(outputs, all_voice_keys, instrument,
                                 gl_fps=4000.0)
            if sd is not None:
                _sidecar_cache["data"] = sd
                curve_disp.set_sidecar(sd)
                _save_full_sidecar(sd, sidecar_path)
            progress_bar.set_status("ready")
        except Exception as e:
            print(f"[sidecar] capture error: {e}")
            progress_bar.set_status("ready")

        return audio

    def _trigger_render() -> None:
        """Clear cache, reset solver, render in background, auto-play when done.
        This is the only path that triggers audio re-rendering (bound to R key)."""
        if not _render_lock.acquire(blocking=False):
            progress_bar.set_status("render already in progress…")
            return
        _audio_cache["audio"] = None
        solver.reset()
        solver.dispatch_before_start()
        def _worker():
            try:
                audio = _render_audio()
                if audio is not None:
                    _do_play(audio)
            except Exception as _e:
                import traceback
                print(f"[render] worker error: {_e}")
                traceback.print_exc()
            finally:
                _render_lock.release()
        threading.Thread(target=_worker, daemon=True).start()

    def _play() -> None:
        """Play cached audio only — never triggers a render."""
        cached = _audio_cache["audio"]
        if cached is not None:
            _do_play(cached)
        else:
            progress_bar.set_status("no audio — press R to render")

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
                except Exception as _e:
                    import traceback
                    print(f"[save_wav] worker error: {_e}")
                    traceback.print_exc()
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

    # Initial render: kick off background render immediately so Space works on first press
    _trigger_render()

    RES_X = CTR_X + CTR_W   # x origin of right (coupling) panel

    def _apply_resize(new_w, new_h):
        nonlocal WIN_W, WIN_H, CTR_X, CTR_W, MAIN_H, LEFT_TOP_H, LEFT_BOT_H, RES_X, offscreen
        WIN_W, WIN_H = new_w, new_h
        CTR_X, CTR_W, MAIN_H, LEFT_TOP_H, LEFT_BOT_H, RES_X = _layout(WIN_W, WIN_H)
        offscreen = pygame.Surface((WIN_W, WIN_H)) if _opengl_ok else screen
        try:
            progress_bar.w = CTR_W
        except Exception:
            pass

    running = True
    while running:
        # Autoscale: sync layout variables to actual window size each frame.
        try:
            _ww, _wh = pygame.display.get_window_size()
        except Exception:
            _ww, _wh = screen.get_size()

        if _ww != WIN_W or _wh != WIN_H:
            _apply_resize(_ww, _wh)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type in (pygame.WINDOWRESIZED, pygame.VIDEORESIZE):
                try:
                    new_w = ev.x if hasattr(ev, 'x') else ev.w
                    new_h = ev.y if hasattr(ev, 'y') else ev.h
                except Exception:
                    new_w, new_h = WIN_W, WIN_H
                _apply_resize(new_w, new_h)
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    _trigger_render()
                elif ev.key == pygame.K_SPACE:
                    _play()
                elif ev.key == pygame.K_s:
                    _save_wav()
                elif ev.key == pygame.K_n:
                    mode = curve_disp.cycle_ray_norm_mode()
                    progress_bar.set_status(f"ray norm: {mode}")
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
                    curve_disp.on_mouse_down(ev.button,
                                             float(mx - CTR_X),
                                             float(MAIN_H - my),
                                             pygame.key.get_mods())
            elif ev.type == pygame.MOUSEBUTTONUP:
                instr_panel.on_mouse_up()
                body_panel.on_mouse_up()
                coupling_panel.on_mouse_up()
                curve_disp.on_mouse_up(ev.button)
            elif ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                instr_panel.on_mouse_move(mx, my)
                body_panel.on_mouse_move(mx, my)
                coupling_panel.on_mouse_move(mx, my)
                curve_disp.on_mouse_move(float(mx - CTR_X), float(MAIN_H - my))
            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                coupling_panel.on_scroll(ev.y, mx, my)
                if CTR_X <= mx < CTR_X + CTR_W:
                    btn = 4 if ev.y > 0 else 5
                    curve_disp.on_mouse_down(btn, float(mx - CTR_X),
                                             float(MAIN_H - my), 0)

        # Update playhead from playback clock
        if _play_start["t"] is not None and pygame.mixer.get_busy():
            elapsed = _time.monotonic() - _play_start["t"]
            curve_disp.set_playhead(min(1.0, elapsed / max(total_s, 1e-3)))

        # ── Draw ─────────────────────────────────────────────────────────────
        offscreen.fill(_C_BG)

        # Left column: InstrumentPanel (top) + BodyResonatorPanel (bottom)
        instr_panel.render(offscreen, 0, 0, SIDE_W, LEFT_TOP_H, font)
        body_panel.render(offscreen, 0, LEFT_TOP_H, SIDE_W, LEFT_BOT_H, font)
        pygame.draw.line(offscreen, (60, 60, 80), (SIDE_W, 0), (SIDE_W, MAIN_H), 2)

        # Right column: SympatheticCouplingPanel
        coupling_panel.render(offscreen, RES_X, 0, SIDE_W, MAIN_H, font)
        pygame.draw.line(offscreen, (60, 60, 80), (RES_X, 0), (RES_X, MAIN_H), 2)

        # Center: parametric curve editor (three instrument panels)
        ctr_surf = curve_disp.render_to_surface(CTR_W, MAIN_H)
        offscreen.blit(ctr_surf, (CTR_X, 0))

        # Progress bar + controls
        progress_bar.render(offscreen, x=CTR_X, y=MAIN_H, font=font13)

        # Hotkey hint at bottom-left
        hint = "R=render  Space=play  S=save WAV  N=ray norm"
        htxt = font.render(hint, True, (80, 80, 100))
        offscreen.blit(htxt, (4, WIN_H - font.get_height() - 2))

        # GL composite: upload 2-D surface then draw shader panels on top
        if _blitter is not None:
            _blitter.blit(offscreen, WIN_W, WIN_H)
            curve_disp.draw_gl_overlays(CTR_X, 0, WIN_W, WIN_H)

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
