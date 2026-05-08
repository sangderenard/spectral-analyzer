# Ray-tracing faculties in spectral-analyzer

This is an untangling pass — no code changes. Goal: enumerate every
distinct ray-tracing faculty so we stop conflating them, and identify what
"parity" between the C++ tracer and the GLSL BDPT actually has to mean.

---

## 1. The four (really five) distinct ray tracers

| # | Tracer | File / entry | Lang | Quality | Direction | Output |
|---|--------|--------------|------|---------|-----------|--------|
| 1 | GLSL forward field tracer | [demo_pluck_gl.py](demo_pluck_gl.py#L3263) `_GPU_RAY_FIELD_CS` | GLSL compute | full | forward (PASS_FORWARD) | per-band R32UI 3D `_tex_bands` (lightfield), reactive PendingRayBuf |
| 2 | GLSL backward sensor tracer | [demo_pluck_gl.py](demo_pluck_gl.py#L4547) `_GPU_SENSOR_CS` | GLSL compute | full | backward (PASS_SENSOR) | per-layer RGBA32F sensor images |
| 3 | C++ acoustic/EM ray tracer | [csrc/kernels/ray_tracer.cpp](csrc/kernels/ray_tracer.cpp#L1) `_spectral_kernels.RayTracer` + [csrc/kernels/rt_field_solver.cpp](csrc/kernels/rt_field_solver.cpp#L1) `FieldSolver` | C++ (Eigen + complex) | full / scientific | forward + receiver-shadow (transfer functions) | seg buffer (N,12), per-tri `surface_flux/direct/indirect`, complex H[src,rec,band] |
| 4 | Lens-bake parametric tracer | [camera_designer/bake_worker.py](camera_designer/bake_worker.py#L74) `trace_ray` | Python float64 | physical (Sellmeier + Snell) | forward through lens elements | LensManifold noodle LUT (12-col) |
| 5 | Numpy Möller-Trumbore preview tracer | [ray_tracer_bridge.py](ray_tracer_bridge.py#L510) `_raycast_triangles` | numpy | preview / no-bounce | forward (single-shot) | nearest-hit dict (t, pos, normal, owner_tag) |

There is **no separate "camera-table preview ray tracer"** — what the user
remembers as the cheap preview is item **5**, the numpy `_raycast_triangles`
behind `trace_single_ray` / `trace_cone_rays` / `trace_depth_map`. Item
**4** is the lens-bake tracer, also Python but full-precision; it produces
a LUT, not a frame.

The "elegant CPP" the user remembers is items **3** (`ray_tracer.cpp` and
`rt_field_solver.cpp`).

---

## 2. Per-tracer details

### Tracer 1 — GLSL forward field (`_GPU_RAY_FIELD_CS`)
- Defined: [demo_pluck_gl.py](demo_pluck_gl.py#L3263), pass modes at [demo_pluck_gl.py](demo_pluck_gl.py#L3308) (`PASS_FORWARD` / `PASS_SENSOR` / `PASS_REACTIVE`).
- Geometry SSBO: 8×vec4 `Tri` (binding 0) — `v0`, `e1`, `e2`, `normal+mat_flags`, `mat_in`, `mat_out`, `albedo+emit_flag`, `emissive` (`emit_profile_idx`, `remit_profile_idx`, _, `reactive_shift_hz`).
- BVH: `NodeBuf` (binding 1), `TriIdBuf` (binding 2).
- Sources: `SourceBuf` `bdpt_sources` (binding 5) — 3×vec4 schema (pos+weight, dir, packet `freq/phase/energy/coherence`).
- Profiles: `EmissionProfileBuf` (binding 7) including `PROFILE_MANIFOLD` rows that hold pre-baked LensManifold noodles produced by tracer 4.
- Driven by [`SensorAccumulator.pump_forward()`](demo_pluck_gl.py#L7838) → [`_gpu_pump_prebuilt`](demo_pluck_gl.py#L7279).

### Tracer 2 — GLSL backward sensor (`_GPU_SENSOR_CS`)
- Defined: [demo_pluck_gl.py](demo_pluck_gl.py#L4547).
- Reads the *same* 8×vec4 `Tri` SSBO that tracer 1 writes through, plus the per-band irradiance textures tracer 1 fills (`uFwdLayer0..7`).
- Camera model: pinhole + thin-lens DOF + Scheimpflug tilt + N-blade aperture + per-channel CA + optional precomputed manifold ray dirs (`uUseManifold` reads from baked LUT slots in `EmissionProfileBuf`).
- Driven by `Renderer.tick_sensor()` ([demo_pluck_gl.py](demo_pluck_gl.py#L7838)).

### Tracer 3 — C++ scientific tracer (`_spectral_kernels.RayTracer` + `FieldSolver`)
- File: [csrc/kernels/ray_tracer.cpp](csrc/kernels/ray_tracer.cpp#L1).
- Header doc says: each ray carries a complex amplitude vector A[b] per band. Per-band complex reflectance `R[b] = refl_re + j·refl_im`. Phase accumulation `exp(j·k[b]·Δr)`, atmospheric absorption, geometric spreading. BVH accelerated.
- Field solver: [csrc/kernels/rt_field_solver.cpp](csrc/kernels/rt_field_solver.cpp#L1) — unified acoustic / EM Helmholtz, complex impedance Fresnel, modal Schroeder crossover for low frequencies.
- Python bridge: [`trace_cavity_scene`](ray_tracer_bridge.py#L1555) — produces per-tri `surface_flux`, `surface_direct`, `surface_indirect`, segment buffer.
- Consumed by [demo_pluck_gl.py L13051](demo_pluck_gl.py#L13051) `R.set_ray_lighting(meta)` → [`RayLightingState`](demo_pluck_gl.py#L2856).
- Geometry input: triangulated room (`(N,3,3)` verts + per-tri normals + scalar `[reflectivity, diffusion, absorption]` mat_props) — **NOT** the same 8×vec4 SSBO format the GLSL BDPT uses.
- Material schema: `SpectralMaterial` per-tri (when present) or fallback per-band `(refl_re, refl_im, diffusion, emission, reemission)` derived from `mat_props`. **No** PBR (albedo/IOR/opacity/emit-profile-idx/reactive_shift_hz) — those are GLSL-only fields today.

### Tracer 4 — Lens-bake parametric tracer (`bake_worker.trace_ray`)
- File: [camera_designer/bake_worker.py](camera_designer/bake_worker.py#L74).
- Operates on `ParametricSurface` 64-bit conic/polynomial geometry, *not* on triangles.
- Per-wavelength Sellmeier dispersion, vector Snell at every interface, aperture-plane test, sensor-plane test.
- Output: 12-column `noodle` LUT consumed by `LensManifold` ([camera_software/lens_manifold.py](camera_software/lens_manifold.py)) AND uploaded into `EmissionProfileBuf` slot type `PROFILE_MANIFOLD = 3` so tracer 1/2 can re-use the bake.
- Pure preview/bake utility — never touched at frame time.

### Tracer 5 — Numpy preview tracer (`_raycast_triangles`)
- File: [ray_tracer_bridge.py](ray_tracer_bridge.py#L510).
- Single-bounce Möller-Trumbore over `(N,3,3)` triangle soup, batched in numpy.
- Wrappers: [`trace_single_ray`](ray_tracer_bridge.py#L658), [`trace_cone_rays`](ray_tracer_bridge.py#L679), [`trace_depth_map`](ray_tracer_bridge.py#L719).
- Callers: [player_controller.py L64-65](player_controller.py#L64) (focus picking), [player_clip_py.py L460-461](player_clip_py.py#L460) (velocity-normal clip).
- Strictly geometric — no spectra, no materials, no bounces, no exposure. **This is the cheap preview tracer.** Already labelled an "optional velocity-normal clip pass" / "focus picking" in its callers.

### (Bonus) Tracer 6 — TF (time-frequency) ray engine
- File: [tf_ray_engine.py](tf_ray_engine.py#L1).
- Bidirectional 2-D geometric tracer in (time, frequency) space, completely separate from the 3-D scene tracers above. Has its own `TFPropagator` (CPU) and `TFRayGPU` (GPU). Out of scope for the camera/sensor calibration question — listed only so we don't conflate it.

---

## 3. How they actually fit together right now

```
Scene geometry ─┬─► verts8 + groups + MaterialDatabase
                │       │
                │       ├─► BaseGLRenderer rasterizer (preview shading) — NOT a ray tracer
                │       │
                │       ├─► Tracer 1 (GLSL fwd)  ─►  per-band 3D field
                │       │           │
                │       │           ▼
                │       └─► Tracer 2 (GLSL bwd)  ─►  per-layer sensor RGBA32F
                │                       (forward+backward = the existing BDPT)
                │
                └─► triangulated CavityScene + SpectralMaterials
                            │
                            ▼
                       Tracer 3 (C++ FieldSolver) ─► surface_flux + complex H
                            │
                            ▼
                       set_ray_lighting(meta) → RayLightingState
                            │
                            ▼  (read-only by the ILLUM layer)
                       Tracer 1 sees this only as a static prelit field
```

Lens path:
```
CameraPreset ─► Tracer 4 (bake_worker) ─► LensManifold noodle LUT
                                                │
                                                ▼ uploaded into EmissionProfileBuf
                                         Tracer 1/2 read it as PROFILE_MANIFOLD
```

Picking path:
```
mouse / focus / clip ─► Tracer 5 (_raycast_triangles) ─► nearest hit only
```

Key fact: **the C++ tracer (3) and the GLSL BDPT (1+2) are NOT coupled
inside one bidirectional solve today.** They share no SSBO. The C tracer
runs once, dumps `surface_flux` into a Python state object, and the GLSL
BDPT later reads that as a baseline lighting term — that is *not*
bidirectional path tracing across the two engines.

---

## 4. Material / geometry parity gaps (C++ ↔ GLSL BDPT)

### Geometry schema disparity
| Field | GLSL Tri (8×vec4) | C++ RayTracer | C++ FieldSolver |
|-------|-------------------|---------------|-----------------|
| v0, e1, e2 | yes | yes (`(N,9)`) | yes |
| normal | yes | yes | yes (computed) |
| mat_flags (bitfield) | yes | no — only `flags = TRANSMISSIVE/APERTURE_STOP` | no |
| albedo (rgb) | yes | no | no |
| IOR (n_in / n_out) | yes (per side) | yes (per side) | per-band complex `n_re + j n_im` |
| opacity | yes | implicit via `flags` | implicit |
| emit_profile_idx | yes (lookup into `EmissionProfileBuf`) | no — emission is a per-band scalar in `emission_bands` | same |
| remit_profile_idx | yes | no | no |
| reactive_shift_hz | yes (PendingRay queue) | no | no |
| per-band complex reflectance | no — uses `mat_in/mat_out` scalar refl + albedo | yes (`refl_re[tri][b] + j refl_im[tri][b]`) | yes |
| spectral diffusion per band | partial | yes (scalar fallback) | yes |

### Phenomena disparity
| Phenomenon | GLSL BDPT (1+2) | C++ tracer (3) |
|------------|----------------|----------------|
| BVH acceleration | yes | yes |
| Möller-Trumbore | yes | yes |
| Cosine-weighted diffuse + specular split | yes (RNG threshold = `mat_in/mat_out` diff) | yes (`diffusion[tri]` MT-RNG) |
| Snell refraction at IOR boundary | yes (per-ray) | yes (per-band; phase from complex k) |
| Per-band complex amplitude | no — RGB / per-channel only | yes |
| Phase accumulation `exp(j k Δr)` | no — incoherent splat | yes |
| Atmospheric absorption (`exp(-α Δr)`) | partial (uVolAlpha march only) | yes |
| Geometric spreading `1/(1+r…)` | no — relies on splat density | yes |
| Emission profile spectra (Planck, Gaussian, …) | yes (via `EmissionProfileBuf` headers) | partial — `SpectralMaterial.emission_bands` is per-band scalar, no profile registry shared with `EmissionProfileDatabase` |
| Re-emission / Stokes shift | yes (`MAT_FLAG_REACTIVE`, PendingRay) | no |
| Lens manifold (`PROFILE_MANIFOLD`) | yes (consumes baked noodles from tracer 4) | no |
| Receiver-shadow transfer functions H[src,rec,band] | no | yes (`FieldSolver`) |
| Modal Schroeder crossover (low-f rectangular eigenmodes) | no | yes |

### Driver / call-site parity
| | GLSL BDPT | C++ tracer |
|-|-----------|------------|
| Frame-time | yes (`tick_sensor` per frame) | no — one-shot at scene load (`set_ray_lighting`) |
| Source schema | `bdpt_sources` (N,12) from `RayOrder.bake_rays` / `pack_emissive_area_rays` | per-source `(pos, dir, directivity_power)` arrays |
| Material source of truth | `MaterialDatabase.build_tensors()['pbr']` + `EmissionProfileDatabase` | `SpectralMaterial` list on the room (separate registry) |

---

## 5. What "parity" must mean

For "C++ and GLSL mixed in a proper bidirectional solve" the following
must be true and currently is not:

1. **Single geometry contract.** Both tracers ingest `(verts8, groups, MaterialDatabase)` (the unified contract `pack_emissive_area_rays` was just rebuilt around). The C tracer's current `(verts(N,3,3), normals(N,3), mat_props(N,3))` schema must be extended to read the same 8×vec4 record (or a SoA equivalent) and the same `EmissionProfileDatabase` registry — not a parallel `SpectralMaterial` list.
2. **Single source contract.** Both tracers consume the same `bdpt_sources (N,12)` row format. C tracer's `(src_pos, src_dir, src_directivity)` must be replaced by the unified row schema. Then `pack_emissive_area_rays` feeds both.
3. **Shared PBR fidelity.** Every field the GLSL BDPT reads must have a C tracer code path: `albedo`, `IOR(n_in/n_out)`, `opacity`, `emit_profile_idx`, `remit_profile_idx`, `reactive_shift_hz`, `mat_flags`. Today the C tracer has none of these (PBR-side); GLSL has none of the C tracer's complex per-band amplitudes (physical-side). Parity = both sides hold the full union.
4. **Real coupling, not "dump once".** A "bidirectional solve" between them means the C tracer computes the *coherent* forward field (its strength) and the GLSL backward sensor pass *connects to the C-tracer surface samples* every batch — not consumes a frozen baseline. That requires either (a) C tracer streams per-batch surface flux into the GLSL `_tex_bands` SSBO each `pump_forward`, or (b) the C tracer is invoked per batch with the same `bdpt_sources` slice the GLSL forward pass consumes, and its complex H output is composed with the GLSL backward sensor signal before gain solve.
5. **No "video-game" shortcuts.** Specifically:
   - GLSL splat must not stand in for `1/r²` — use proper geometric spreading, like the C tracer does.
   - GLSL emissive splat must not stand in for spectral emission — must read `EmissionProfileDatabase` profile and sample wavelength + photon energy, like `pack_emissive_area_rays` already does.
   - Reflection must not be `albedo·NdotL` — use the per-band complex reflectance the C tracer already computes and route it back into GLSL as a per-tri × per-band texture sampled by the forward pass.

---

## 6. Recommended next moves (research, no code yet)

A. Extend the 8×vec4 `Tri` record (or add a parallel SSBO) to carry the
   per-band complex reflectance the C tracer already produces — this is
   the single biggest material-schema gap. Sourced from
   `build_complex_reflectances_spectral` in [ray_tracer_bridge.py L1610](ray_tracer_bridge.py#L1610).

B. Teach `_spectral_kernels.RayTracer` to ingest the 8×vec4 SSBO record
   (or an SoA mirror produced by `MaterialDatabase.build_tensors()`).
   Add a `RayTracer.trace_from_bdpt_sources(rows_Nx12)` entry point so
   `pack_emissive_area_rays` feeds both tracers verbatim.

C. Promote `EmissionProfileDatabase` to be the single emission registry
   on the C side too, replacing the per-tri `emission_bands` scalar
   array. Then both tracers reproduce the *same* spectrum.

D. Wire `SensorAccumulator.pump_forward()` to optionally call the C
   tracer per batch (when sensor layer 9 is on AND a "physical" toggle
   is set), splatting its per-tri complex flux into `_tex_bands` BEFORE
   the GLSL backward pass runs. This is the "C++ + GLSL mixed in one
   bidirectional solve" the user described.

E. Keep tracer 5 (`_raycast_triangles`) and tracer 4 (`bake_worker`)
   intact and clearly labelled — they are NOT in the bidirectional
   solve path. Tracer 5 is geometry-only picking; tracer 4 is a one-
   shot lens LUT bake whose output is consumed by tracers 1/2. Don't
   touch them while doing parity work on 1+2 vs 3.
