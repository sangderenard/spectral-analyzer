# Micropatches: Agent-Facing Instrument Authoring Spec

_Canonical reference for agents and tools that author, edit, or load instrument objects in this system._
_Verified against actual codebase. Field names, types, and enums here are exact._

---

## What a micropatch is

A **micropatch** is a saved instrument instance: one voice source plus optional modulation wiring.
It is the atom of instrument authoring — small enough for an agent to write safely, rich enough to define a complete sound.

Micropatches are not whole songs. They compose into larger score and arrangement structures by being added to a patch and optionally wired into the routing graph.

---

## System layers

```
1. Voice source      — AnalyticVoice: oscillator, envelope, chirp, harmonics, granular
2. Modulation        — LFODefinition: FM/AM sources attached to voice.fm / voice.am
3. Routing graph     — RoutingGraph: signal flow between voices, LFOs, and mixers
4. Score pages       — RhythmPage, DynamicsProgram, ImprovProgram: contextual control
5. Resolved output   — concrete synthesized notes after the render pass
```

A micropatch captures **layers 1 and 2**. It may reference layer 3 topology. It does not encode a whole score.

---

## Instance wrapper format

Every saved object uses the `InstanceWrapper` envelope defined in `preset_library.py`.

```json
{
  "instance_type":  "micropatch",
  "instance_id":    "lead_bright_chirp_01",
  "label":          "Bright Chirped Lead",
  "description":    "Fast-attack harmonic lead with upward chirp.",
  "tags":           ["lead", "bright", "chirped", "harmonic"],
  "summary": {
    "role":              "signal",
    "seq_role":          "melody",
    "register":          "all",
    "character":         ["bright", "chirped"],
    "pitch_behavior":    "linear upward chirp, settles in 40ms",
    "spectral_behavior": "8 harmonics, brightness 1.2",
    "envelope_behavior": "fast attack, moderate release"
  },
  "engine_version": "score-0.1",
  "created_at":     "",
  "payload": {
    "source":        { },
    "lfos":          [ ],
    "edges":         [ ]
  }
}
```

**Field rules:**

| Field | Required | Notes |
|---|---|---|
| `instance_type` | yes | `"micropatch"` for voice+modulation bundles. `"voice"` for flat single-voice presets. |
| `instance_id` | yes | Unique, machine-safe. Convention: `{type}_{slug}_{version}`. E.g. `lead_bright_01`. |
| `label` | yes | 2–5 word human name |
| `description` | yes | One sentence: what it sounds like and when to use it |
| `tags` | yes | ≥ 2. See tag vocabulary below. |
| `summary` | yes | Semantic descriptor block for agent reasoning |
| `engine_version` | yes | Always `"score-0.1"` |
| `created_at` | auto | Set by library on save; leave `""` when emitting |
| `payload` | yes | See payload structure below |

---

## Payload structure

### Minimal payload (voice only)

```json
{
  "payload": {
    "source": { ...AnalyticVoice fields... }
  }
}
```

### Full payload (voice + LFO modulation + routing)

```json
{
  "payload": {
    "source": { ...AnalyticVoice fields... },
    "lfos": [
      {
        "id":    "lfo1",
        "label": "Vibrato LFO",
        "rate_hz":      0.35,
        "shape":        "Sine",
        "phase_offset": 0.0,
        "depth":        1.0
      }
    ],
    "fm": {
      "lfo_id":   "lfo1",
      "depth_hz": 12.0
    },
    "am": null,
    "edges": [
      { "src": "source", "dst": "__mix__", "w": 1.0, "angle": 0.0, "delay": 0.0 }
    ]
  }
}
```

---

## AnalyticVoice field reference

This is what goes in `payload.source`. Every field name here is exact and round-trips through `AnalyticVoice.from_dict()`.

### Oscillator basics

| Field | Type | Default | Range / Choices | Notes |
|---|---|---|---|---|
| `key` | str | auto-UUID | — | Leave `"auto"` and the interpreter will generate. |
| `label` | str | `"Voice"` | — | Human name |
| `freq_hz` | float | 440.0 | 1–20000 Hz (log) | Base frequency. Overridden by note tracking. |
| `semitone_offset` | float | 0.0 | −48 to +48 st | Transposition on top of note pitch |
| `note_tracking` | choice | `"note"` | `"note"` `"root"` `"free"` | `"root"` for bass/pedal voices |
| `amplitude` | float | 1.0 | 0–4 | Overall level |
| `phase_origin` | float | 0.0 | −π to +π rad | Initial phase |
| `pre_delay` | float | 0.0 | 0–2 s | Silence before note onset |
| `muted` | bool | false | — | |
| `voice_role` | choice | `"signal"` | `"signal"` `"air"` `"transient"` `"body"` | |
| `seq_role` | choice | `"melody"` | `"melody"` `"bass"` `"root"` `"stab"` | |
| `register` | choice | `"all"` | `"all"` `"bass"` `"mid"` `"high"` | Which rhythm page responds |

### Envelope

| Field | Type | Default | Notes |
|---|---|---|---|
| `env_type` | choice | `"adsr"` | `"adsr"` `"spline"` `"monotone"` `"linear"` |
| `adsr.attack` | float | 0.005 s | 0.001–2 s (log). Ignored when env_type ≠ adsr. |
| `adsr.decay` | float | 0.04 s | 0.001–2 s |
| `adsr.sustain` | float | 0.75 | 0–1 |
| `adsr.release` | float | 0.08 s | 0.001–4 s |
| `adsr.peak` | float | 1.0 | 0–2. Above 1.0 adds punch on attack. |
| `env_knots` | list | `[[0,0],[0.01,1],[0.1,.75],[.85,.75],[1,0]]` | Used when `env_type` ≠ adsr. List of `[time, amp]` pairs, normalized 0–1. |
| `loop_start` | float | 0.1 | 0–0.99 |
| `loop_end` | float | 0.9 | 0.01–1 |
| `loop_enabled` | bool | false | |

### Chirp

Chirp sweeps the instantaneous frequency at note onset, then settles. **`f_delta_start` is the offset at t=0. `f_delta_end` is the offset at end of note. A typical attack chirp starts offset and settles on pitch: `f_delta_start ≠ 0, f_delta_end = 0`.**

| Field | Type | Default | Notes |
|---|---|---|---|
| `chirp.chirp_type` | choice | `"none"` | `"none"` `"linear"` `"exponential"` `"power"` |
| `chirp.f_delta_start` | float | 0.0 Hz | Frequency offset at note start. Positive = above pitch. |
| `chirp.f_delta_end` | float | 0.0 Hz | Frequency offset at note end. Usually 0.0. |
| `chirp.tau` | float | 0.5 s | Decay time constant for `exponential` type (log 0.01–10 s) |
| `chirp.chirp_power` | float | 1.0 | Exponent for `power` type (0.1–8) |

**Chirp type guide:**
- `none` — static pitch
- `linear` — linear sweep from `f_delta_start` to `f_delta_end` over note duration
- `exponential` — exponential decay from `f_delta_start` toward `f_delta_end`, time constant `tau`
- `power` — power-law decay, shape controlled by `chirp_power`

**Examples:**
- Upward attack chirp (lifts into pitch): `chirp_type=linear, f_delta_start=120.0, f_delta_end=0.0`
- Plucked string (sharp onset, exponential settle): `chirp_type=exponential, f_delta_start=-60.0, tau=0.025`
- Bell strike: `chirp_type=power, f_delta_start=300.0, tau=0.08, chirp_power=3.0`

### Harmonic manifold

| Field | Type | Default | Notes |
|---|---|---|---|
| `manifold_type` | choice | `"pure"` | `"pure"` `"harmonic"` `"harmonic_warp"` |
| `harmonic_count` | int | 8 | 1–32. Ignored for `pure`. |
| `harmonic_brightness` | float | 1.0 | Amplitude rolloff: `amp_k = 1/k^brightness`. Higher = warmer (more rolloff). |
| `harmonic_warp_strength` | float | 0.0 | 0 = integer multiples. 0.3 = subtle inharmonicity. 1.0 = strongly metallic. |

**Manifold guide:**
- `pure` — near-sinusoidal fundamental, minimal overtones
- `harmonic` — additive stack with controllable rolloff brightness
- `harmonic_warp` — stretched harmonic ratios, bell-like or metallic quality

### Granular emission

When `emission_mode = "granular"` a `granular` dict must also be present. All granular fields are ignored in `single` mode.

| Field | Type | Default | Notes |
|---|---|---|---|
| `emission_mode` | choice | `"single"` | `"single"` `"granular"` |
| `granular.center_frequency_hz` | float | 440.0 | Cloud center pitch (log, 20–20000 Hz) |
| `granular.grain_pitch_spread_semitones` | float | 3.0 | Pitch randomness per grain (0 = pitched, 12 = cloud) |
| `granular.grain_harmonic_lock` | float | 0.0 | 0 = free pitch, 1 = locked to harmonic series |
| `granular.grain_highband_bias` | float | 0.0 | Bias toward higher harmonics (octaves, 0–3) |
| `granular.grain_density_hz` | float | 20.0 | Grains per second (0.5–500, log) |
| `granular.birth_jitter` | float | 0.5 | Timing randomness per grain (0–1) |
| `granular.burst_probability` | float | 0.0 | Probability of burst emission (0–1) |
| `granular.burst_size` | int | 4 | Grains per burst (2–32) |
| `granular.burst_spread_s` | float | 0.015 | Time spread within a burst (log, 0.001–0.2 s) |
| `granular.grain_duration_s` | float | 0.05 | Per-grain duration (log, 0.002–2 s) |
| `granular.grain_duration_jitter` | float | 0.3 | Duration randomness (0–2) |
| `granular.grain_chirp_depth` | float | 0.0 | Per-grain chirp intensity (0–2) |
| `granular.grain_chirp_jitter` | float | 0.5 | Chirp direction randomness (0–1) |
| `granular.grain_phase_randomness` | float | 1.0 | 1.0 = fully random phase (washy). 0 = coherent. |
| `granular.grain_manifold_mix` | float | 0.0 | Mix of harmonic manifold into grain (0–1) |
| `granular.gain` | float | 1.0 | Overall granular gain (0–4) |
| `granular.grain_amp_jitter` | float | 0.2 | Per-grain amplitude randomness (0–2) |
| `granular.attack_frac` | float | 0.15 | Grain envelope attack as fraction of grain duration (0.01–0.5) |
| `granular.release_frac` | float | 0.35 | Grain envelope release as fraction of grain duration (0.01–0.8) |
| `granular.grain_coherence` | float | 0.5 | Coherence depth (0–1) |
| `granular.coherence_mode` | choice | `"uniform"` | `"uniform"` `"sine"` `"random_walk"` `"burst"` `"gradient"` `"perlin_walk"` |
| `granular.coherence_rate_hz` | float | 0.5 | Coherence modulation rate (0.01–10 Hz) |
| `granular.coherence_depth` | float | 0.3 | Coherence modulation depth (0–1) |

### FM and AM modulation

FM and AM are defined on the voice itself and reference LFO keys by `source_key`.

```json
"fm": {
  "source_key": "lfo_key_here",
  "depth_hz":   12.0,
  "depth_amp":  0.0
}
```

```json
"am": {
  "source_key": "lfo_key_here",
  "depth_hz":   0.0,
  "depth_amp":  0.3
}
```

- `source_key` resolves in this priority: `patch.lfos` (LFODefinition keys) → other voices → param nodes
- `depth_hz` — FM depth in Hz (how much the LFO deviates the frequency)
- `depth_amp` — AM depth (0–1: fraction of amplitude modulated)
- Set to `null` when not used

---

## LFO format (payload.lfos array)

An LFO entry in a micropatch payload defines a modulation source. On load it becomes a `LFODefinition` in `patch.lfos` and its `key` becomes the `source_key` for FM or AM.

```json
{
  "id":           "lfo1",
  "label":        "Vibrato",
  "rate_hz":      0.35,
  "shape":        "Sine",
  "phase_offset": 0.0,
  "depth":        1.0
}
```

**Exact field names — these are what the code uses:**

| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | str | required | Local ID within this micropatch (used by fm/am reference). Not the final key. |
| `label` | str | `"LFO"` | Human name |
| `rate_hz` | float | 1.0 | Oscillation rate (0.01–50 Hz, log) |
| `shape` | choice | `"Sine"` | `"Sine"` `"Triangle"` `"Sawtooth"` `"Square"` — **TitleCase, exact** |
| `phase_offset` | float | 0.0 | Initial phase (−π to +π rad). **Not `phase`.** |
| `depth` | float | 1.0 | Output amplitude (0–2). **Not `amplitude`.** |

**Unsupported in current engine (Bucket B — valid to include, ignored at load):**
- `bias` (DC offset)
- `sync` (tempo sync)
- Shapes: `noise`, `sample_hold`, `saw_up`, `saw_down`

---

## FM/AM wiring in a micropatch

Within the `payload`, use `fm` and `am` dicts to wire an LFO to the voice. The interpreter translates these to `voice.fm.source_key` using the `id`→actual key map.

```json
"fm": { "lfo_id": "lfo1", "depth_hz": 12.0 },
"am": null
```

Or reference an existing library LFO by preset ID:

```json
"fm": { "preset_id": "lfo_slow_sine_v1", "depth_hz": 8.0 },
"am": null
```

---

## Routing edges

Edges in `payload.edges` use the actual `RoutingEdge` serialization format:

```json
{ "src": "source", "dst": "__mix__", "w": 1.0, "angle": 0.0, "delay": 0.0 }
```

| Field | Notes |
|---|---|
| `src` | Local ID: `"source"` for the voice, an LFO `"id"`, or `"__mix__"` for the default mixer |
| `dst` | Same namespace |
| `w` | Amplitude weight (signed float, typically −2 to +2) |
| `angle` | Phase rotation in radians (default 0.0) |
| `delay` | Per-edge propagation delay in seconds (default 0.0) |

**Default mixer key is `"__mix__"`.** When omitted from `payload.edges`, the interpreter connects `source → __mix__` at weight 1.0 automatically.

Feedback edges must include a loop policy in `payload.feedback`:

```json
"feedback": {
  "enabled":           true,
  "delay_s":           0.05,
  "decay":             0.08,
  "ringdown_mode":     "decay_to_silence",
  "ringdown_max_s":    4.0,
  "ringdown_threshold": 1e-4
}
```

---

## Module types (actual implementation)

The `AnalyticModule` system handles signal-routing-graph modules. These are **not** per-voice insert effects — they are routing graph nodes.

**Implemented `module_type` values:**

| `module_type` | Description | Key params |
|---|---|---|
| `lfo` | Multi-channel LFO as a routing graph signal node | `rate_hz`, `shape`, `phase_offset`, `depth`, `lfo_channels` |
| `passthrough` | Identity signal pass-through | — |
| `pitch_quantizer` | Quantizes pitch signal to scale | `interpolation_mode`, `portamento_time`, `slew_rate` |
| `interaural` | Binaural spatialization (azimuth, elevation, distance) | `iau_azimuth`, `iau_elevation`, `iau_distance`, `iau_width` |
| `state_machine` | Plugin-driven physics/state simulation | `sm_plugin`, `sm_params` |

**Bucket B — aspirational, not yet in DSP (preserve, do not execute):**
- `overdrive`, `delay`, `reverb`, `resonator`, `filter`, `envelope_follower`
- `voice_source`, `voice_output` (conceptual; the actual source is `payload.source`)
- `mixer` (actual mixer is the `__mix__` node)

---

## Complete minimal micropatch example

```json
{
  "instance_type": "micropatch",
  "instance_id":   "lead_bright_chirp_01",
  "label":         "Bright Chirped Lead",
  "description":   "Fast-attack harmonic lead with upward linear chirp.",
  "tags":          ["lead", "bright", "chirped", "harmonic", "melody"],
  "summary": {
    "role":              "signal",
    "seq_role":          "melody",
    "register":          "all",
    "character":         ["bright", "chirped", "harmonic"],
    "pitch_behavior":    "linear upward chirp +120Hz onset, settles in 40ms",
    "spectral_behavior": "10 harmonics, brightness 1.2",
    "envelope_behavior": "fast attack 8ms, moderate release 100ms"
  },
  "engine_version": "score-0.1",
  "created_at":     "",
  "payload": {
    "source": {
      "key":           "auto",
      "label":         "Bright Chirped Lead",
      "freq_hz":       440.0,
      "semitone_offset": 0.0,
      "note_tracking": "note",
      "amplitude":     0.85,
      "phase_origin":  0.0,
      "chirp": {
        "chirp_type":    "linear",
        "f_delta_start": 120.0,
        "f_delta_end":   0.0,
        "tau":           0.04,
        "chirp_power":   1.0
      },
      "fm":       null,
      "am":       null,
      "env_type": "adsr",
      "adsr": {
        "attack":  0.008,
        "decay":   0.06,
        "sustain": 0.7,
        "release": 0.1,
        "peak":    1.0
      },
      "env_knots":          [[0,0],[0.01,1],[0.1,0.75],[0.85,0.75],[1,0]],
      "loop_start":         0.1,
      "loop_end":           0.9,
      "loop_enabled":       false,
      "muted":              false,
      "color":              [255, 200, 80],
      "pre_delay":          0.0,
      "manifold_type":      "harmonic",
      "harmonic_count":     10,
      "harmonic_brightness": 1.2,
      "harmonic_warp_strength": 0.0,
      "voice_role":  "signal",
      "seq_role":    "melody",
      "register":    "all",
      "emission_mode": "single",
      "granular":    null
    },
    "lfos":     [],
    "fm":       null,
    "am":       null,
    "edges":    []
  }
}
```

---

## Micropatch with LFO vibrato

```json
{
  "instance_type": "micropatch",
  "instance_id":   "lead_vibrato_01",
  "label":         "Vibrato Lead",
  "description":   "Harmonic lead with slow sine vibrato (±8Hz FM).",
  "tags":          ["lead", "harmonic", "vibrato", "melody"],
  "summary": { "role": "signal", "seq_role": "melody", "register": "all",
               "character": ["harmonic", "vibrato"] },
  "engine_version": "score-0.1",
  "created_at": "",
  "payload": {
    "source": {
      "key": "auto", "label": "Vibrato Lead",
      "freq_hz": 440.0, "semitone_offset": 0.0,
      "note_tracking": "note", "amplitude": 1.0, "phase_origin": 0.0,
      "chirp": { "chirp_type": "none", "f_delta_start": 0.0, "f_delta_end": 0.0, "tau": 0.5, "chirp_power": 1.0 },
      "fm": null, "am": null,
      "env_type": "adsr",
      "adsr": { "attack": 0.01, "decay": 0.05, "sustain": 0.75, "release": 0.1, "peak": 1.0 },
      "env_knots": [[0,0],[0.01,1],[0.1,0.75],[0.85,0.75],[1,0]],
      "loop_start": 0.1, "loop_end": 0.9, "loop_enabled": false, "muted": false,
      "color": [100, 200, 160], "pre_delay": 0.0,
      "manifold_type": "harmonic", "harmonic_count": 8, "harmonic_brightness": 1.3,
      "harmonic_warp_strength": 0.0, "voice_role": "signal", "seq_role": "melody",
      "register": "all", "emission_mode": "single", "granular": null
    },
    "lfos": [
      { "id": "lfo1", "label": "Vibrato", "rate_hz": 0.35,
        "shape": "Sine", "phase_offset": 0.0, "depth": 1.0 }
    ],
    "fm": { "lfo_id": "lfo1", "depth_hz": 8.0 },
    "am": null,
    "edges": []
  }
}
```

---

## Preset references (loading from library)

Instead of inlining a full source dict, a micropatch can reference an existing preset:

```json
{
  "payload": {
    "source": { "preset_id": "voice_harmonic_bright_lead_v1" },
    "lfos":   [ { "id": "lfo1", "preset_id": "lfo_slow_sine_v1" } ],
    "fm":     { "lfo_id": "lfo1", "depth_hz": 10.0 }
  }
}
```

The interpreter (`micropatch_interpreter.py`) resolves `preset_id` fields against the `InstanceLibrary` before constructing engine objects.

---

## Interpretation (how `micropatch_interpreter.py` loads this)

Loading a micropatch produces live engine objects and wires them into a running `AnalyticPatch`.

```python
from micropatch_interpreter import load_micropatch, apply_set_fields
from preset_library import get_library

lib = get_library()
wrapper = lib.load("micropatch", "lead_bright_chirp_01")
result = load_micropatch(wrapper, patch, library=lib)
# result = {"voice_key": "abc123", "lfo_keys": ["def456"], "key_map": {...}}
```

The interpreter:
1. Resolves any `preset_id` references from the library
2. Creates `AnalyticVoice` from `payload.source` → appends to `patch.voices`
3. Creates `LFODefinition` objects from `payload.lfos` → appends to `patch.lfos`
4. Wires `fm`/`am` by setting `voice.fm.source_key` to the resolved LFO key
5. Adds all keys to `patch.routing`, wires `voice → __mix__` unless `payload.edges` overrides

---

## Field editing (set_fields)

Agents can describe field changes as dot-path dicts:

```json
{
  "target": "source",
  "fields": {
    "chirp.chirp_type":    "exponential",
    "chirp.f_delta_start": -80.0,
    "harmonic_brightness": 0.8,
    "adsr.attack":         0.002
  }
}
```

This maps to `apply_set_fields(voice, fields)` in `micropatch_interpreter.py`, which traverses the dot path and sets each field. All dot paths correspond to the knob `name` field in `agent_prompts/manifests/voice_manifest.json`.

---

## Knob manifest reference

The machine-readable field manifest is at `agent_prompts/manifests/voice_manifest.json`.
It was generated by `python knob_manifest.py`.

Each field entry looks like:

```json
{
  "name":         "chirp.chirp_type",
  "label":        "Chirp type",
  "dtype":        "choice",
  "default":      "none",
  "low":          0.0,
  "high":         3.0,
  "step":         1.0,
  "unit":         "",
  "choices":      ["none", "linear", "exponential", "power"],
  "is_log":       false,
  "group":        "Chirp",
  "fmt":          ".0f",
  "visible_when": null
}
```

Fields with `visible_when` are only meaningful when a condition holds. For example, all `adsr.*` fields have `visible_when: ["env_type", "adsr"]` — they are ignored when `env_type != "adsr"`.

---

## Agent voice design workflow

### Step 1 — Role
Choose `voice_role`, `seq_role`, `register`. Set `note_tracking="root"` when `seq_role` is `bass` or `root`.

### Step 2 — Spectral family
Choose `manifold_type`. Add `harmonic_count` and `harmonic_brightness` for `harmonic` or `harmonic_warp`. Add `harmonic_warp_strength` for `harmonic_warp`.

### Step 3 — Envelope
Choose `env_type`. Fill `adsr.*` for `adsr`. Fill `env_knots` for others.

### Step 4 — Chirp
Choose `chirp.chirp_type`. Set `f_delta_start` (onset deviation) and `f_delta_end` (usually 0.0). Set `tau` for `exponential`. Set `chirp_power` for `power`.

### Step 5 — Granular (if needed)
Set `emission_mode="granular"` and fill the `granular.*` block. Start from: density 20/s, spread 3 semitones, duration 0.05s, phase randomness 1.0.

### Step 6 — Modulation (if needed)
Add an LFO entry in `lfos[]`, then reference it in `fm` or `am`.

### Step 7 — Emit with explanation
Produce the full micropatch JSON and a 2–4 sentence musical explanation of why the choices fit the request.

---

## Editing safety rules

**Safe edits (change freely):**
- Any scalar knob value within declared range
- Switching between enumerated choices
- `voice_role`, `seq_role`, `register`, `note_tracking`
- `emission_mode`, `manifold_type`, `chirp_type` — these change which other fields are active
- Adding or removing an LFO and its fm/am wiring

**Riskier edits (explain the tradeoff):**
- Changing `env_type` — old envelope data is not automatically migrated
- Changing `emission_mode` — granular block must be added or removed accordingly
- Routing topology changes that introduce cycles — must include a `feedback` block

**Required behaviors:**
- Preserve `key` values when editing in place (do not regenerate)
- Preserve unknown fields — do not delete fields the agent doesn't understand
- Validate enum values before writing
- If a `chirp_type` change makes `tau` or `chirp_power` irrelevant, leave them (they're ignored)

---

## Tag vocabulary

### Voice / micropatch tags
**Role:** `lead` `bass` `pad` `stab` `fx` `body` `air` `transient`  
**Spectral:** `pure` `sine` `harmonic` `inharmonic` `warped` `granular`  
**Pitch:** `chirped` `static` `root` `vibrato`  
**Character:** `bright` `warm` `soft` `sharp` `diffuse` `metallic` `plucked`  
**Arrangement:** `melody` `root` `texture` `color` `starter`

### LFO tags
**Rate:** `slow` `medium` `fast`  
**Shape:** `sine` `triangle` `sawtooth` `square`  
**Use:** `vibrato` `tremolo` `gate` `sweep` `ramp` `wobble` `AM` `FM`

---

## Relationship to flat voice presets

The `presets/voice/` directory contains `instance_type: "voice"` wrappers with flat payloads — the direct output of `AnalyticVoice.to_dict()` — no `source` wrapper, no `lfos`, no `edges`.

A `"voice"` preset is simpler and better for the library. A `"micropatch"` bundles the voice with its modulation.

Either can be loaded into a patch with `micropatch_interpreter.load_micropatch()` — it handles both formats.

---

## What is NOT yet implemented (Bucket B)

These are architecturally valid to describe in a micropatch payload. The interpreter will preserve them but not execute them:

- Per-voice DSP effects: overdrive, delay, reverb, resonator, filter
- Control edges targeting arbitrary parameter paths beyond FM/AM
- LFO `bias`, `sync`, shape `noise`, `sample_hold`
- Multi-stage control chains (LFO modulating another LFO's rate)
- Envelope control nodes (`type: "envelope"`) as routing sources
- `sample_hold` random generators
- JSON operation vocabulary (`create_micropatch`, `add_module`, etc.) as a code-level protocol
