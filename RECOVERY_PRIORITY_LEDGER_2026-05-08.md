# Recovery Priority Ledger (2026-05-08)

Purpose
- Reconstruct active technical priorities after conversation corruption.
- Record only evidence-backed items from repo history and plan docs.

Evidence Basis
- Last 7 commits: `f37e712`, `46e734f`, `c49430b`, `03b31a5`, `2c33c27`, `1811a2a`, `6ad7c53`.
- Plan docs touched/relevant:
	- `RAY_TRACER_INVENTORY.md`
	- `BLACKBODY_CALIBRATION.md`
	- `UV_EMISSION_STACK_PLAN.md`
- Implementation anchors:
	- `csrc/kernels/ray_tracer.cpp`
	- `csrc/bindings/pybind_kernels.cpp`
	- `csrc/include/ray_tracer.h`
	- `csrc/include/triangle_groups.h`
	- `bdpt_integrator.py`
	- `demo_pluck_gl.py`
	- `exposure_render_demo.py`

---

## P0 — C++/GLSL ray-tracer parity (algorithm level)

Invariant
- Parity means complete algorithm/feature parity between C++ and GLSL ray shaders.
- No unique transport, material, geometry, source, accumulation, or record semantics may exist in only one backend.
- Coupling is one requirement inside parity, not the definition of parity.

Evidence
- `RAY_TRACER_INVENTORY.md` sections 5 and 6 define parity constraints and A-E next moves.
- `demo_pluck_gl.py` has explicit parity TODO blocks above `_GPU_RAY_FIELD_CS` and `_GPU_SENSOR_CS`.

Required outcomes
1. Single geometry/material/source contract across both tracers.
2. Single transport contract (bounce rules, termination, flags, phase/amplitude behavior, refraction/attenuation behavior).
3. Single endpoint/output contract (same recordable phenomena and accumulation semantics).
4. No backend-only features. Any capability added to one tracer must be implemented in the other or explicitly removed from both.
5. Coupling behavior (when used) must not introduce backend-specific semantics.

---

## P1 — Bidirectional integrator backbone in C++ (already landed, verify and preserve)

Invariant
- Endpoint storage is complex and unreduced (`no abs()`, `no quantize`, `no band collapse`).

Evidence
- `csrc/include/bdpt_record.h` and `bdpt_integrator.py` enforce 64-byte `EndpointRecord` parity.
- `csrc/kernels/ray_tracer.cpp` implements `ray_tracer_bidirectional` plus PIXEL_CONE mode.
- `csrc/bindings/pybind_kernels.cpp` exposes tri-group registry and BDPT calls.

Required outcomes
1. Keep complex record format unchanged end-to-end.
2. Preserve tri-group role semantics (`EMISSIVE`, `SENSOR`, `BLOCKER`, `VOLUME`).
3. Preserve PIXEL_CONE mapping (`subpath_id = py*n_px + px`) and aggregator behavior.

---

## P2 — Region context dispatch framework (partially landed)

Invariant
- Context-kind enum and ABI layout remain mirrored between C++ and GLSL.

Evidence
- `csrc/include/ray_tracer.h` defines `SCALE_CONTEXT_KIND_*` and additive `RtScaleContext` fields.
- `csrc/kernels/ray_tracer.cpp` has dispatch call sites and stubs for kinds not yet implemented.
- `pybind_kernels.cpp` exports context-kind constants and `add_scale_context(context_kind, payload)`.

Required outcomes
1. Maintain enum order and struct compatibility.
2. Keep call sites present in bounce loops.
3. Promote stubs to implementations only with parity checks.

---

## P3 — GLSL BDPT parity tasks (explicit TODO, not complete)

Invariant
- GLSL side must emit equivalent endpoint/field semantics, not reduced surrogates.

Evidence
- `demo_pluck_gl.py` parity TODO blocks list missing forward/sensor pieces.
- Forward path currently accumulates layer textures; endpoint parity work is flagged as missing.

Required outcomes
1. Complex endpoint-compatible deposit path in GLSL (paired real/imag storage path).
2. Sensor-group-aware endpoint recording semantics to mirror C++ tri-group lookup behavior.
3. Preserve full spectral band handling at deposit.

---

## P4 — Blackbody calibration continuity (recent plan-doc update)

Invariant
- Emission authoring units and render-time radiometric behavior remain physically consistent.

Evidence
- `BLACKBODY_CALIBRATION.md` documents registration-time radiance derivation and runtime solid-angle intensity logic.

Required outcomes
1. No silent regressions to legacy non-physical intensity hacks.
2. Keep model-dependent calibration (`blackbody`, `parametric`, legacy default) explicit and testable.

---

## P5 — UV emission stack plan continuity (parallel track)

Invariant
- Per-texel runtime spectral solving is out of scope; prebake-only rule stays intact.

Evidence
- `UV_EMISSION_STACK_PLAN.md` states hard invariant and staged build sequence.

Required outcomes
1. Keep this track orthogonal to C++/GLSL parity work.
2. Avoid introducing runtime spectral expansion in the hot fragment path.

---

## Commit-to-Priority Mapping (last 7)

1. `f37e712`: major BDPT + tri-group + context-kind + parity TODO surfacing.
2. `46e734f`: C++ tracer/material bridge adjustments.
3. `c49430b`: bridge/scene/renderer adjustments.
4. `03b31a5`: exposure demo scaffold and calibration harness framing.
5. `2c33c27`: MAT_FLAG and ray/GLSL structural updates.
6. `1811a2a`: material DB calibration-related update.
7. `6ad7c53`: plan-doc crystallization (`RAY_TRACER_INVENTORY.md`, `BLACKBODY_CALIBRATION.md`).

---

## What to treat as canonical now

Canonical priority source order:
1. `RAY_TRACER_INVENTORY.md` section "What parity must mean".
2. `demo_pluck_gl.py` parity TODO blocks at shader declarations.
3. `csrc/*` + `bdpt_integrator.py` for landed BDPT/tri-group contracts.
4. `BLACKBODY_CALIBRATION.md` for emission/radiometry assumptions.

Parity acceptance rule:
- A feature matrix diff between C++ and GLSL must contain zero unmatched rows.
- If an unmatched row exists, parity is not complete.

This file is a continuity anchor intended to survive chat/session corruption.
