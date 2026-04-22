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
Output is mix_out.real, normalised to ±1, sent to sounddevice.
"""

import sys
import types
import torch
import sounddevice as sd
from voice_graph_node import build_voice_mixer_network

try:
    from tqdm import tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False

_CDTYPE = torch.complex128

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

    solver, vnodes, _ = build_voice_mixer_network(
        [voice_obj],
        sample_rate=sr,
        duration=max_dur,
        device=torch.device("cpu"),
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
    return audio.real.to(torch.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

# fmt: off
NOTE_SCHEDULE = [
    # (pitch_hz,  on_time_s, off_time_s)
    (261.63,  0.00, 0.30),   # C4
    (293.66,  0.35, 0.65),   # D4
    (329.63,  0.70, 1.00),   # E4
    (349.23,  1.05, 1.35),   # F4
    (392.00,  1.40, 1.70),   # G4
    (440.00,  1.75, 2.05),   # A4
    (493.88,  2.10, 2.40),   # B4
    (523.25,  2.45, 3.00),   # C5
]
# fmt: on

SAMPLE_RATE = 48_000

if __name__ == "__main__":
    print("Rendering…")
    waveform = render_note_schedule(NOTE_SCHEDULE, sample_rate=SAMPLE_RATE)

    peak = waveform.abs().max().item()
    if peak > 1e-9:
        waveform = waveform / peak

    audio_np = waveform.numpy().astype("float32")
    print(f"Playing {len(audio_np) / SAMPLE_RATE:.2f}s of audio  (peak={peak:.4f})")
    sd.play(audio_np, samplerate=SAMPLE_RATE)
    sd.wait()
    print("Done.")
