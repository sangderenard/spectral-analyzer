# Agent Prompt: LFO Design

You are designing a single **LFODefinition** instance.
LFOs modulate other parameters — typically FM depth (pitch wobble) or AM depth (amplitude tremolo) — via the voice's `fm.source_key` or `am.source_key` fields.

The instance wrapper schema is at `agent_prompts/instance_schema.md`.
Starter presets are in `presets/lfo/`.

---

## All fields

| Field | Type | Range | Default | Notes |
|---|---|---|---|---|
| `rate_hz` | float | 0.01–50 Hz (log) | 1.0 | Oscillation frequency. Below 0.5 Hz = phrase-scale. Above 10 Hz = audio-rate buzz. |
| `shape` | choice | `Sine`, `Triangle`, `Sawtooth`, `Square` | `Sine` | Waveform shape. See below. |
| `phase_offset` | float | -π to +π rad | 0.0 | Phase shift at start. Use to stagger LFOs relative to each other. |
| `depth` | float | 0.0–2.0 | 1.0 | Output amplitude. 1.0 = full-range modulation signal. |

---

## Shape guide

| Shape | Behavior | Best for |
|---|---|---|
| `Sine` | Smooth sinusoidal cycle | Vibrato, smooth tremolo, gentle sweeps |
| `Triangle` | Linear rise and fall, symmetric | Mechanical wobble, pitch modulation with clear direction |
| `Sawtooth` | Gradual linear rise, instant reset | Sweep per cycle, rhythmic ramp feel |
| `Square` | Hard binary toggle between +depth and -depth | Gating, on/off tremolo, hard switches |

---

## Rate guide

| Rate | Period | Musical meaning |
|---|---|---|
| 0.08 Hz | ~12s | Phrase-scale drift |
| 0.25 Hz | 4s | One swell per 4 bars at 60 BPM |
| 0.5 Hz | 2s | One cycle per 2 bars |
| 1.0 Hz | 1s | One cycle per bar at 60 BPM |
| 2.0 Hz | 0.5s | Quarter-note tremolo at 120 BPM |
| 6.0 Hz | 167ms | Fast vibrato (vocal-style) |
| 10 Hz+ | <100ms | Borders on audio-rate FM/AM buzz |

---

## How LFOs connect to voices

In a voice payload:
```json
"fm": {
  "source_key": "lfo_key_here",
  "depth_hz": 15.0,
  "depth_amp": 0.0
}
```
```json
"am": {
  "source_key": "lfo_key_here",
  "depth_hz": 0.0,
  "depth_amp": 0.3
}
```

The LFO's `depth` is a gain on its own output. The voice's `depth_hz` or `depth_amp` sets how strongly that output modulates the target parameter. Both multiply together.

---

## Output format

```json
{
  "instance_type": "lfo",
  "instance_id": "lfo_{slug}_v1",
  "label": "...",
  "description": "...",
  "tags": ["..."],
  "summary": {
    "rate": "X Hz (Y period)",
    "shape": "...",
    "depth": 1.0,
    "use_cases": ["..."]
  },
  "engine_version": "score-0.1",
  "created_at": "",
  "payload": {
    "key": "auto",
    "label": "...",
    "rate_hz": 1.0,
    "shape": "Sine",
    "phase_offset": 0.0,
    "depth": 1.0,
    "color": [200, 160, 60]
  }
}
```

---

## Starter presets

| Preset ID | Rate | Shape | Notes |
|---|---|---|---|
| `lfo_slow_sine_v1` | 0.15 Hz | Sine | Phrase-scale modulation |
| `lfo_medium_triangle_v1` | 2.0 Hz | Triangle | Note-scale tremolo |
| `lfo_fast_square_gate_v1` | 8.0 Hz | Square | Hard amplitude gating |
| `lfo_sawtooth_ramp_v1` | 1.0 Hz | Sawtooth | Bar-rate sweep |
