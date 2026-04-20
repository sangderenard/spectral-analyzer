# Agent Prompt: Dynamics Program Design

You are designing a single **DynamicsProgram** instance.
A DynamicsProgram applies velocity shaping to all notes in a register.
It has two independent layers: a **curve** (phrase-scale arc) and an **accent pattern** (per-step levels).

The instance wrapper schema is at `agent_prompts/instance_schema.md`.
Starter presets are in `presets/dynamics_program/`.

---

## Structure

```json
{
  "enabled": true,
  "curve": {
    "shape": "swell",
    "scope_bars": 2.0,
    "intensity": 0.6
  },
  "accent": {
    "name": "Dyn",
    "levels": [1.5, 1.0, 1.0, 1.0, 1.2, 1.0, 1.0, 1.0,
               1.5, 1.0, 1.0, 1.0, 1.2, 1.0, 1.0, 1.0]
  }
}
```

The final velocity multiplier for any note = `curve.multiplier_at_bar(bar) × accent.level_at(step)`.

---

## Curve shapes

| `shape` | Arc over [0, 1] in time | Typical use |
|---|---|---|
| `flat` | Constant 1.0 — no effect | Neutral / disabled base |
| `crescendo` | Linear quiet → loud | Building sections |
| `decrescendo` | Linear loud → quiet | Fading out |
| `swell` | Arch: quiet → loud → quiet (sin half-cycle) | Phrase swells, pad motion |
| `dip` | Inverted arch: loud → quiet → loud | Breathing rhythm |
| `drop` | Low anticipatory build → explosive peak → decay | Drop section, climax hit |
| `random` | Per-note random jitter (seeded) | Humanization |

---

## Curve parameters

| Field | Range | Notes |
|---|---|---|
| `scope_bars` | 0.25, 0.5, 1, 2, 4, 8 | How many bars one full curve cycle spans. Tiled beyond that. |
| `intensity` | 0.0–1.0 | 0.0 = no dynamic range (everything at 1.0). 1.0 = full range (can reach near 0 or 2). |

**Intensity math:** The multiplier at any point is `1.0 + intensity × (raw_shape - 1.0)`, where `raw_shape` is in [0, 1] mapped to [-1, +1].

---

## Accent pattern

The `levels` array maps to steps in the rhythm grid (default 16 steps for 4/4 × 4 subdivisions).
Values are in [0.0, 2.0]:
- `0.0` = silence
- `0.5` = soft
- `1.0` = normal (no change)
- `1.5` = accented
- `2.0` = strongly accented

**Common patterns:**

4/4 downbeat accent (1 and 3 strong):
```json
[1.5, 1.0, 1.0, 1.0,  1.2, 1.0, 1.0, 1.0,
 1.5, 1.0, 1.0, 1.0,  1.2, 1.0, 1.0, 1.0]
```

Syncopated (off-beat emphasis):
```json
[1.0, 1.0, 1.5, 1.0,  1.0, 1.5, 1.0, 1.0,
 1.0, 1.0, 1.5, 1.0,  1.0, 1.5, 1.0, 1.0]
```

All uniform (let the curve do the work):
```json
[1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0]
```

---

## Design decisions

**When to enable:** Set `enabled=false` for a neutral preset (the curve and accent are stored but have no effect). Enable at the register/patch level when you want the shaping to fire.

**Curve vs accent:** Use the curve for macro phrase dynamics (crescendo over 4 bars). Use the accent for micro rhythmic emphasis (beat 1 louder). They combine multiplicatively — don't double-stack both at high intensity or you'll clip.

**Scope_bars and tempo:** A `scope_bars=4` swell at 120 BPM lasts 8 seconds — long and slow. At 180 BPM it lasts 5.3 seconds — still phrase-scale but more active.

---

## Output format

```json
{
  "instance_type": "dynamics_program",
  "instance_id": "dynamics_{slug}_v1",
  "label": "...",
  "description": "...",
  "tags": ["..."],
  "summary": {
    "enabled": true,
    "curve_shape": "...",
    "intensity": 0.6,
    "accent_pattern": "...",
    "use_cases": ["..."]
  },
  "engine_version": "score-0.1",
  "created_at": "",
  "payload": {
    "enabled": true,
    "curve": {
      "shape": "flat",
      "scope_bars": 1.0,
      "intensity": 0.5
    },
    "accent": {
      "name": "Dyn",
      "levels": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    }
  }
}
```

---

## Starter presets

| Preset ID | Shape | Scope | Notes |
|---|---|---|---|
| `dynamics_flat_passthrough_v1` | flat, disabled | 1 bar | Neutral baseline |
| `dynamics_2bar_swell_v1` | swell, enabled | 2 bars | Phrase swells |
| `dynamics_random_jitter_v1` | random, enabled | 1 bar | Per-note humanization |
