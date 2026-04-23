"""demo_note_playback.py — render a note schedule through GraphSolver + voice node.

Usage
-----
    python demo_note_playback.py

Inputs (edit the NOTE_SCHEDULE block near the bottom):
    Each entry is (pitch_hz: float, on_time_s: float, off_time_s: float).

Network topology
----------------
    build_voice_mixer_network([voice_obj]) → GraphSolver with:
        {key}_pitch_in → {key}_out → mix_out

Gate events are fired at quantized sample boundaries by the outer render loop.
Pitch is injected via the {key}_pitch_in ext port each tick (complex128 scalar).
Output is mix_out.real, written to a short mono WAV by default, and optionally
played through sounddevice.
"""

from pathlib import Path
import sys
import types
import wave
import torch
import sounddevice as sd

import graph_solver as _gs

_gs.PROFILE_TIMING = False
_gs.PROFILE_REPORT_INTERVAL_S = 0.0
_gs._T.stop_reporter()
_gs._T.reset()

from voice_graph_node import build_voice_mixer_network

try:
    from tqdm import tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False

_CDTYPE = torch.complex128
OUTPUT_WAV_PATH = Path("demo_note_playback.wav")
PLAY_AUDIO = False

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_voice_obj(key: str, freq_hz: float = 440.0) -> object:
    """Minimal duck-type voice object that build_voice_mixer_network reads."""
    return types.SimpleNamespace(
        key=key,
        freq_hz=freq_hz,
        amplitude=1.0,
        phase_origin=0.0,
        semitone_offset=0.0,
        harmonic_brightness=1.0,
        harmonic_warp_strength=0.0,
        fm=None,
        am=None,
        manifold_type="pure",
        harmonic_count=8,
        loop_enabled=False,
        loop_start=0.1,
        loop_end=0.9,
        emission_mode="single",
        pre_delay=0.0,
        envelope=None,
        chirp=None,
    )


def _progress(iterable, total, desc=""):
    if _has_tqdm:
        return tqdm(iterable, total=total, desc=desc, unit="samp", ncols=80)
    # Fallback: print a simple percentage to stderr every 5 %
    class _Simple:
        def __init__(self, it, total):
            self._it = iter(it)
            self._total = total
            self._n = 0
            self._last_pct = -1
        def __iter__(self):
            return self
        def __next__(self):
            val = next(self._it)
            self._n += 1
            pct = int(100 * self._n / self._total)
            if pct // 5 != self._last_pct // 5:
                self._last_pct = pct
                print(f"\r{desc}  {pct:3d}%", end="", flush=True, file=sys.stderr)
            return val
        def close(self):
            print(file=sys.stderr)
    return _Simple(iterable, total)


def _normalization_reference(waveform: torch.Tensor, top_k: int = 128) -> float:
    mags = waveform.detach().abs().reshape(-1)
    if mags.numel() == 0:
        return 1.0
    peak = float(mags.max().item())
    if peak <= 1e-12:
        return 1.0
    k = min(max(1, int(top_k)), int(mags.numel()))
    top = torch.topk(mags, k).values
    robust = float(top.median().item())
    return robust if robust > 1e-12 else peak


def _prepare_audio_for_output(
    waveform: torch.Tensor,
    *,
    target_peak: float = 0.95,
) -> tuple[torch.Tensor, dict[str, float]]:
    wav = waveform.detach().to(torch.float64).cpu()
    raw_peak = float(wav.abs().max().item()) if wav.numel() else 0.0
    rms = float(torch.sqrt(torch.mean(wav.square())).item()) if wav.numel() else 0.0
    scale_ref = _normalization_reference(wav)
    if scale_ref > 1e-12:
        wav = wav * (float(target_peak) / scale_ref)
    wav = torch.clamp(wav, -1.0, 1.0)
    clipped = int(torch.count_nonzero((wav.abs() >= 0.999999)).item()) if wav.numel() else 0
    return wav, {
        "raw_peak": raw_peak,
        "scale_ref": scale_ref,
        "rms": rms,
        "clipped_samples": float(clipped),
    }


def _write_wav_pcm16(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> Path:
    path = Path(path)
    audio = waveform.detach().to(torch.float64).cpu().clamp(-1.0, 1.0)
    pcm = torch.round(audio * 32767.0).to(torch.int16).numpy()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return path


def render_note_schedule(
    schedule: list,          # [(pitch_hz, on_s, off_s), ...]
    sample_rate: int = 48_000,
    tail_s: float = 0.5,
) -> torch.Tensor:
    """Render schedule → 1-D float64 waveform (mix_out.real)."""
    if not schedule:
        return torch.zeros(int(tail_s * sample_rate), dtype=torch.float64)

    sr = float(sample_rate)
    last_off = max(off for _, _, off in schedule)
    total_samples = int((last_off + tail_s) * sr)

    max_dur = max(off - on for _, on, off in schedule)
    max_dur = max(max_dur, 0.05)

    voice_key = "voice"
    voice_obj = _make_voice_obj(voice_key, freq_hz=schedule[0][0])

    with torch.inference_mode():
        solver, vnodes, _ = build_voice_mixer_network(
            [voice_obj],
            sample_rate=sr,
            duration=max_dur,
            device=torch.device("cpu"),
            voice_port_occupancy={voice_key: ("pitch_in",)},
        )
        vnode = vnodes[voice_key]
        pitch_node_key = f"{voice_key}_pitch_in"

        # Pre-quantize gate events.
        events: list[tuple[int, str, float]] = []
        for pitch_hz, on_s, off_s in schedule:
            events.append((int(on_s  * sr), "on",  pitch_hz))
            events.append((int(off_s * sr), "off", pitch_hz))
        events.sort(key=lambda e: e[0])

        buf: list[torch.Tensor] = []
        current_pitch: float = schedule[0][0]
        event_ptr = 0

        bar = _progress(range(total_samples), total=total_samples, desc="Rendering")
        for i in bar:
            t = i / sr

            while event_ptr < len(events) and events[event_ptr][0] == i:
                _, kind, ph = events[event_ptr]
                if kind == "on":
                    current_pitch = ph
                    vnode.gate_on(t, velocity=1.0)
                else:
                    vnode.gate_off(t)
                event_ptr += 1

            vnode.set_sample(i)

            pitch_c128 = torch.tensor(complex(current_pitch, 0.0), dtype=_CDTYPE)
            state = solver.step({pitch_node_key: pitch_c128})
            buf.append(state["mix_out"])

        if hasattr(bar, "close"):
            bar.close()

        audio = torch.stack(buf)
        return audio.real.to(torch.float64).detach()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

# fmt: off
NOTE_SCHEDULE = [
    # (pitch_hz,  on_time_s, off_time_s)
    (261.63,  0.00, 0.08),   # C4
]
# fmt: on

SAMPLE_RATE = 48_000
TAIL_S = 0.04

if __name__ == "__main__":
    print("Rendering…")
    waveform = render_note_schedule(NOTE_SCHEDULE, sample_rate=SAMPLE_RATE, tail_s=TAIL_S)
    audio_out, stats = _prepare_audio_for_output(waveform)
    wav_path = _write_wav_pcm16(OUTPUT_WAV_PATH, audio_out, SAMPLE_RATE)

    duration_s = len(audio_out) / SAMPLE_RATE
    print(
        f"Saved {wav_path}  ({duration_s:.2f}s, raw_peak={stats['raw_peak']:.4f}, "
        f"scale_ref={stats['scale_ref']:.4f}, rms={stats['rms']:.4f}, "
        f"clipped={int(stats['clipped_samples'])})"
    )

    if PLAY_AUDIO:
        audio_np = audio_out.detach().cpu().numpy().astype("float32")
        print(f"Playing {duration_s:.2f}s of audio")
        sd.play(audio_np, samplerate=SAMPLE_RATE)
        sd.wait()
    print("Done.")
