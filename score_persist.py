"""score_persist.py — Serialize / deserialize PerformanceAtom lists to JSON.

Atoms produced by TorchComposerNode (or any score producer) are captured after
dispatch_before_start and written to a compact JSON file.  ScoreLoaderNode
reads the same format and replays the atoms into the graph identically.

The format stores curves as control-point arrays so the full ParametricCurve
is reconstructable without the original patch state.  GateEvent history is
preserved for envelope warp logic.  phase_offset (complex) is stored as [re, im].

Sidecar audio
─────────────
During a render you may optionally capture per-voice pre-mix signals (the
summed voice output *before* the instrument body).  call capture_voice_premix()
inside your render loop, then save_sidecar() after run_schedule().  The sidecar
is a .npz alongside the score JSON with three arrays:

  premix          float32  (n_voices, n_display_frames) — downsampled RMS
  premix_keys     list of voice keys matching axis-0 of premix
  display_fps     scalar float
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _curve_to_points(curve: Any) -> list:
    """Return the control points of a ParametricCurve as a plain list of dicts."""
    pts = getattr(curve, "points", None) or []
    return [
        {
            "t":           float(p.t),
            "v":           float(p.v),
            "theta":       float(getattr(p, "theta", 0.0)),
            "tension":     float(getattr(p, "tension", 0.5)),
            "break_after": bool(getattr(p, "break_after", False)),
        }
        for p in pts
    ]


def _curve_from_points(pts: list) -> Any:
    """Rebuild a ParametricCurve from the saved control-point list."""
    from parametric_curve import ParametricCurve
    c = ParametricCurve(name="loaded", v_lo=0.0, v_hi=1.0)
    for p in pts:
        idx = c.add_point(float(p["t"]), float(p["v"]))
        cp = c.points[idx]
        if "theta" in p:
            cp.theta = float(p["theta"])
        if "tension" in p:
            cp.tension = float(p["tension"])
        if "break_after" in p:
            cp.break_after = bool(p["break_after"])
    return c


def _gate_history_to_list(gate_history: list) -> list:
    out = []
    for ev in gate_history:
        out.append({
            "t_on":     float(getattr(ev, "t_on", 0.0)),
            "t_off":    float(getattr(ev, "t_off", 0.0)) if getattr(ev, "t_off", None) is not None else None,
            "velocity": float(getattr(ev, "velocity", 1.0)),
        })
    return out


def _gate_history_from_list(raw: list) -> list:
    from parametric_curve import GateEvent
    return [
        GateEvent(
            t_on=float(d["t_on"]),
            t_off=float(d["t_off"]) if d.get("t_off") is not None else None,
            velocity=float(d.get("velocity", 1.0)),
        )
        for d in raw
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Atom ↔ dict
# ──────────────────────────────────────────────────────────────────────────────

def atom_to_dict(atom: Any) -> dict:
    """Serialize one PerformanceAtom to a JSON-able dict."""
    po = atom.phase_offset
    if hasattr(po, "real"):
        po_ser = [float(po.real), float(po.imag)]
    else:
        c = complex(po)
        po_ser = [float(c.real), float(c.imag)]
    return {
        "onset_sample":   int(atom.onset_sample),
        "sample_count":   int(atom.sample_count),
        "onset_time_s":   float(atom.onset_time_s),
        "duration_s":     float(atom.duration_s),
        "release_tail_s": float(atom.release_tail_s),
        "phase_offset":   po_ser,
        "fundamental_hz": float(atom.fundamental_hz),
        "velocity":       float(atom.velocity),
        "voice_key":      str(atom.voice_key),
        "batch_index":    int(atom.batch_index),
        "page_index":     int(atom.page_index),
        "event_index":    int(atom.event_index),
        "phase_delta":    float(getattr(atom, "phase_delta", 0.0)),
        "gate_history":   _gate_history_to_list(atom.gate_history),
        "envelope_points": _curve_to_points(atom.envelope_curve),
        "chirp_points":    _curve_to_points(atom.chirp_curve),
    }


def atom_from_dict(d: dict) -> Any:
    """Reconstruct a PerformanceAtom from its serialized dict."""
    from torch_composer_engine import PerformanceAtom
    po_raw = d.get("phase_offset", [1.0, 0.0])
    phase_offset = complex(float(po_raw[0]), float(po_raw[1]))
    return PerformanceAtom(
        onset_sample=int(d["onset_sample"]),
        sample_count=int(d["sample_count"]),
        onset_time_s=float(d["onset_time_s"]),
        duration_s=float(d["duration_s"]),
        release_tail_s=float(d.get("release_tail_s", 0.08)),
        phase_offset=phase_offset,
        fundamental_hz=float(d["fundamental_hz"]),
        velocity=float(d["velocity"]),
        voice_key=str(d.get("voice_key", "")),
        batch_index=int(d.get("batch_index", 0)),
        page_index=int(d.get("page_index", 0)),
        event_index=int(d.get("event_index", 0)),
        gate_history=_gate_history_from_list(d.get("gate_history", [])),
        envelope_curve=_curve_from_points(d.get("envelope_points", [])),
        chirp_curve=_curve_from_points(d.get("chirp_points", [])),
        phase_delta=float(d.get("phase_delta", 0.0)),
        note_events=[],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Score file I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_score(
    atoms: List[Any],
    path: str | Path,
    *,
    sample_rate: float = 48_000.0,
    n_frames: int = 0,
    meta: Optional[dict] = None,
) -> Path:
    """Write atom list to a .score.json file.  Returns the written Path."""
    path = Path(path)
    if path.suffix not in (".json",):
        path = path.with_suffix(".score.json")
    payload = {
        "version": 1,
        "meta": {
            "sample_rate": float(sample_rate),
            "n_frames":    int(n_frames),
            **(meta or {}),
        },
        "atoms": [atom_to_dict(a) for a in atoms],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_score(path: str | Path) -> tuple[list, dict]:
    """Load a .score.json file.  Returns (atoms, meta_dict)."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta  = dict(payload.get("meta", {}))
    atoms = [atom_from_dict(d) for d in payload.get("atoms", [])]
    return atoms, meta


# ──────────────────────────────────────────────────────────────────────────────
# Sidecar voice pre-mix capture
# ──────────────────────────────────────────────────────────────────────────────

class VoicePremixCapture:
    """Collects per-voice audio buffers during synthesis, saves downsampled sidecar.

    Usage
    -----
        cap = VoicePremixCapture(sample_rate=48000, display_fps=120)

        # inside your custom synthesize_atoms replacement, or a post-render hook:
        cap.add_voice("voice_mel_0", audio_tensor)   # complex128 (n_frames,)

        cap.save("composer_demo.sidecar.npz")
    """

    def __init__(self, sample_rate: float = 48_000.0, display_fps: float = 120.0,
                 gl_fps: float = 4000.0) -> None:
        self.sample_rate  = float(sample_rate)
        self.display_fps  = float(display_fps)
        # Higher rate for the GL phasor display — enough to preserve per-sample
        # phase rotation (120fps averages out phase entirely at audio frequencies).
        self.gl_fps = float(gl_fps)
        self._buffers: dict[str, Any] = {}   # key → numpy complex64

    def add_voice(self, key: str, audio: Any) -> None:
        """Register or accumulate a voice's audio.  audio may be a torch Tensor or numpy array."""
        try:
            import torch
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
        except ImportError:
            pass
        arr = np.asarray(audio, dtype=np.complex64).ravel()
        if key in self._buffers:
            n = min(len(self._buffers[key]), len(arr))
            self._buffers[key][:n] += arr[:n]
        else:
            self._buffers[key] = arr.copy()

    def save(self, path: str | Path) -> Path:
        """Downsample all buffers and write .sidecar.npz.

        Stored arrays
        -------------
        premix         float32   (n_voices, n_display_frames)  — RMS magnitude at display_fps
        premix_complex complex64 (n_voices, n_display_frames)  — mean complex at display_fps
        premix_gl      complex64 (n_voices, n_gl_frames)       — mean complex at gl_fps
                                  Preserved at high rate so phasors rotate visibly.
        premix_keys    object    list of voice key strings
        display_fps    float[1]
        gl_fps         float[1]
        sample_rate    float[1]
        """
        path = Path(path)
        if not self._buffers:
            return path

        def _downsample(buf: np.ndarray, hop: int) -> tuple:
            n = len(buf)
            n_out = math.ceil(n / hop)
            rms  = np.zeros(n_out, dtype=np.float32)
            cplx = np.zeros(n_out, dtype=np.complex64)
            for i in range(n_out):
                chunk = buf[i * hop: (i + 1) * hop]
                if len(chunk):
                    rms[i]  = float(np.sqrt(np.mean(np.abs(chunk) ** 2)))
                    cplx[i] = np.mean(chunk).astype(np.complex64)
            return rms, cplx

        hop_disp = max(1, int(self.sample_rate / self.display_fps))
        hop_gl   = max(1, int(self.sample_rate / self.gl_fps))

        rms_rows:  list = []
        cplx_rows: list = []
        gl_rows:   list = []
        keys:      list = []

        for key, buf in self._buffers.items():
            rms, cplx = _downsample(buf, hop_disp)
            _, gl     = _downsample(buf, hop_gl)
            rms_rows.append(rms)
            cplx_rows.append(cplx)
            gl_rows.append(gl)
            keys.append(key)

        def _pad(rows: list, dtype) -> np.ndarray:
            max_len = max(len(r) for r in rows)
            out = np.zeros((len(rows), max_len), dtype=dtype)
            for i, r in enumerate(rows):
                out[i, :len(r)] = r
            return out

        np.savez_compressed(
            str(path),
            premix=        _pad(rms_rows,  np.float32),
            premix_complex=_pad(cplx_rows, np.complex64),
            premix_gl=     _pad(gl_rows,   np.complex64),
            premix_keys=   np.array(keys, dtype=object),
            display_fps=   np.array([self.display_fps]),
            gl_fps=        np.array([self.gl_fps]),
            sample_rate=   np.array([self.sample_rate]),
        )
        return path

    @staticmethod
    def load(path: str | Path) -> dict:
        """Load a sidecar .npz.

        Returns dict with keys:
          premix          float32   (n_voices, n_display_frames)
          premix_complex  complex64 (n_voices, n_display_frames)
          premix_gl       complex64 (n_voices, n_gl_frames)       — may be absent in old files
          premix_keys     list[str]
          display_fps     float
          gl_fps          float
          sample_rate     float
        """
        data = np.load(str(path), allow_pickle=True)
        out: dict = {
            "premix":      data["premix"],
            "premix_keys": list(data["premix_keys"]),
            "display_fps": float(data["display_fps"][0]),
            "gl_fps":      float(data["gl_fps"][0]) if "gl_fps" in data else 4000.0,
            "sample_rate": float(data["sample_rate"][0]),
        }
        if "premix_complex" in data:
            out["premix_complex"] = data["premix_complex"]
        if "premix_gl" in data:
            out["premix_gl"] = data["premix_gl"]
        return out
