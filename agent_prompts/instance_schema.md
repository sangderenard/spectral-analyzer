# Instance Wrapper Schema

Every saved element in the preset library uses this JSON envelope.
The `payload` field is the raw `to_dict()` output of the native engine object.

```json
{
  "instance_type":  "voice",
  "instance_id":    "voice_bright_chirp_lead_v1",
  "label":          "Bright Chirped Lead",
  "description":    "One sentence describing what this sounds like and when to use it.",
  "tags":           ["lead", "bright", "chirped", "harmonic"],
  "summary": {
    "role":               "signal",
    "seq_role":           "melody",
    "register":           "all",
    "character":          ["bright", "chirped"],
    "pitch_behavior":     "linear upward chirp, settles in 40ms",
    "spectral_behavior":  "8 harmonics, brightness 1.5",
    "envelope_behavior":  "fast attack, moderate release"
  },
  "engine_version": "score-0.1",
  "created_at":     "2026-04-19T00:00:00",
  "payload":        { }
}
```

## Field rules

| Field | Required | Notes |
|---|---|---|
| `instance_type` | yes | One of: `voice`, `lfo`, `dynamics_program`, `improv_program`, `rhythm_page`, `routing_graph`, `grain_population`, `patch` |
| `instance_id` | yes | Unique, machine-safe string. Convention: `{type}_{descriptive_slug}_{version}`. Example: `voice_soft_bass_v1` |
| `label` | yes | Short human name, 2–5 words |
| `description` | yes | One to two sentences: what it sounds like + when to use it |
| `tags` | yes | At least 2 tags. Use existing tags when possible (see below) |
| `summary` | yes for `voice`, optional otherwise | Semantic descriptor block — lets agents reason without parsing the payload |
| `engine_version` | yes | Always `"score-0.1"` |
| `created_at` | auto | ISO timestamp; set by library on save |
| `payload` | yes | Must be a valid `{Class}.to_dict()` output for the type |

## Tag vocabulary

### Voice tags
- **Role**: `lead`, `bass`, `pad`, `stab`, `fx`, `body`, `air`, `transient`
- **Spectral**: `pure`, `sine`, `harmonic`, `inharmonic`, `warped`, `granular`
- **Pitch**: `chirped`, `static`, `root`
- **Character**: `bright`, `warm`, `soft`, `sharp`, `diffuse`, `metallic`, `plucked`
- **Arrangement**: `melody`, `root`, `texture`, `color`, `starter`

### LFO tags
- **Rate**: `slow`, `medium`, `fast`
- **Shape**: `sine`, `triangle`, `sawtooth`, `square`
- **Use**: `vibrato`, `tremolo`, `gate`, `sweep`, `ramp`, `wobble`, `AM`, `FM`

### Dynamics tags
- `flat`, `swell`, `crescendo`, `decrescendo`, `dip`, `drop`, `random`
- `humanize`, `phrase`, `bar`, `2bar`, `4bar`, `neutral`, `disabled`

## Operations

### Save
```python
from preset_library import get_library, InstanceWrapper
lib = get_library()
wrapper = InstanceWrapper(
    instance_type="voice", instance_id="voice_my_lead_v1",
    label="My Lead", description="...", tags=["lead"],
    summary={...}, payload=voice_obj.to_dict()
)
lib.save(wrapper)
```

### Load
```python
w = lib.load("voice", "voice_my_lead_v1")
voice = lib.to_engine_object(w)
```

### Clone
```python
w2 = lib.clone("voice", "voice_my_lead_v1", "voice_my_lead_v2", "My Lead v2")
```

### Search
```python
results = lib.search(instance_type="voice", tags=["lead", "bright"])
results = lib.search(query="chirp")
```

### Wrap a live object
```python
from preset_library import InstanceLibrary
wrapper = InstanceLibrary.from_engine_object(
    voice_obj,
    instance_id="voice_my_lead_v1",
    label="My Lead",
    tags=["lead"],
    summary={"role": "signal", "seq_role": "melody"}
)
lib.save(wrapper)
```
