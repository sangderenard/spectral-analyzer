# BDPT Batch-Triage Work Plan + Algorithm Audit

**Author:** prepared for handoff to the rendering team
**Scope:** (1) a batch-efficiency triage layer for the T5 BDPT connection pass, delivered as a scoring prepass; (2) a deep correctness audit of the optical, material, and spectral paths as currently implemented.
**Status:** Part A is "scoring shader first" — score + read back + eyeball, no dispatch reordering yet. Part B is findings only — nothing in the audit has been changed in code.

Files referenced use the names as uploaded: `bdpt_pack_t5.comp.glsl`, `bdpt_scatter_t5.comp.glsl`, `t5_full_connect.comp.glsl`, `ray_material.comp.glsl`, `ray_material_eval.glsl.inc`, `bvh_shadow.glsl.inc`, `sensor_terminal_splat.comp.glsl`, `ray_tracer.cpp`, `thick_lens_focus_lab.py`.

---

## Part A — Batch-efficiency triage (delivered)

### A.1 Problem being solved

The T5 connection loop in `ray_tracer.cpp` (~lines 11250–11436) is a nested `tile → cbatch → lbatch` grind in which **every** camera vertex in a cam-batch is connected to **every** light vertex in a light-batch. That all-pairs structure is where the ~150B pairs come from. On a scene that is mostly empty (large `dist²`, sub-`min_geom` geometry, or no connectable endpoints), the overwhelming majority of those pairs contribute zero but still cost a dispatch slot, a geometry test, and — for survivors — a shadow ray. The triage layer scores each `(cam-batch, light-batch)` work unit cheaply *before* the real pass and lets a scheduler run high-yield units first while never discarding low-yield ones.

Design decisions locked with the requester:
- **Metric:** all three signals emitted with **tunable coefficients** (not one fixed metric).
- **Idle policy:** **drain-when-empty, never discard.** The score is a sort/route key only; nothing is ever dropped.
- **Scope this pass:** scoring shader + readback + histogram. No changes to the connection loop yet.

### A.2 Deliverables

| File | Role |
|---|---|
| `batch_score.comp.glsl` | One workgroup per `(cam-batch, light-batch)` unit. Samples `k_samples` random pairs, reduces three raw signals + a coefficient-weighted combination, writes 4 floats per unit. |
| `batch_score_host.cpp.inc` | Drop-in host harness: creates the score SSBO, sets uniforms, dispatches over the `(cbatch, lbatch)` grid per tile, reads back, appends `scores.bin`. Marks the exact insertion point. |
| `score_histogram.py` | Reads `scores.bin`, prints per-signal histograms and a run-order sweep showing how much **predicted energy** the top-k units capture. |

### A.3 The three signals (raw, un-normalised)

For each unit the shader draws `k_samples` random `(cam,light)` pairs and reduces:

- **A — predicted contribution:** mean of `beta_cam · beta_light · geom` over connectable samples. "Expected energy delivered."
- **B — connectable density:** fraction of samples that pass `vertex_connectable` *and* clear `min_geom`.
- **C — peak geom potential:** `max` geom across samples. Catches a unit that is mostly nothing but holds one strong link.
- **S — combined:** `wA·(A/normA) + wB·(B/normB) + wC·(C/normC)`.

A, B, C are written raw so the true ranges are visible before coefficients are chosen. S is purely a sort key.

The shader reuses the **identical** `vertex_connectable` predicate and geometry term from `t5_full_connect.comp.glsl`, but performs **no MIS chain walk and no shadow rays**, which is what makes it orders of magnitude cheaper than the real pass.

### A.4 Integration steps (host team)

1. **Load the program** next to `prog_t5` (~line 9536). Use a *non-shadow* compute loader — `batch_score.comp.glsl` does **not** need the BVH/shadow preamble. Cache the uniform locations listed at the top of `batch_score_host.cpp.inc`.
2. **Add SSBO binding 9** (`BatchScoreBuf`). Bindings 0 (light verts) and 1 (cam verts) are shared with T5 unchanged.
3. **Paste the per-tile block** from the harness inside the tile loop, after `n_tile_cam` is known and before the `for (uint32_t c = …)` cbatch loop (~line 11374). It dispatches `glDispatchCompute(n_cbatch, n_lbatch, 1)`.
4. **Run once, inspect:** `python score_histogram.py scores.bin`. Start with `wA=1, wB=wC=0` (pure predicted contribution).
5. **Choose coefficients + a cut point** from the histogram. The sweep table tells you what fraction of total predicted energy the top X% of units already hold — that ratio is the whole justification for the feature.

### A.5 Next step (not in this pass): the scheduler

Once the distribution looks right, the drain-when-empty scheduler is small: build an index array of the `n_units` cells, `std::stable_sort` by `scores[u*4+3]` (S) **descending**, and drive the existing `(c,b)` dispatch from that order instead of the raw nested loops. Nothing is dropped — low-S cells dispatch last, so a long render keeps refining "nothing" regions only after the high-yield ones are done. The existing TDR heartbeat (`glFinish` every 8 lbatches, ~line 11416) carries over unchanged.

### A.6 Known limitation of the score itself

Scoring **skips occlusion** (no shadow rays, by design, for cost). So B and A slightly over-estimate for unit pairs that are geometrically close but actually blocked. Because nothing is discarded, this only mis-*orders* a few cells, never drops them. If the histogram shows it mattering, one shadow sample per unit is a cheap add (the BVH preamble would then be required).

---

## Part B — Algorithm audit

Severity legend: **[S1]** corrupts output / silently loses energy · **[S2]** physically wrong but localized · **[S3]** fidelity / fragility / cleanup.

### B.0 One invariant the whole estimator rests on (verified correct, document it)

`t5_full_connect` computes `contrib = beta_cam · beta_light · brdf_cam · brdf_light · geom / denom` (lines ~681–682), with `denom = Σ prefix·suffix` over all connectable cuts (`candidate_strategy_density`). This is the balance-heuristic estimator `f/Σpᵢ`, and it is **only unbiased because the spectral betas are carried un-normalised** — T3 propagates amplitude as `amp *= reflectance` with **no division by the scatter PDF** (`ray_material.comp.glsl`, e.g. lines 902–911, 832–834). Normalization is deferred to T5's `/denom`. This is correct, but fragile: **any future change that divides betas by a sampling PDF in T3 will double-count and must be paired with removing the `/denom` normalization.** Worth a comment at both sites.

### B.1 Spectral path — three things degrading spectral fidelity

**[S1] Emission spectral records race with arrival records.**
In `ray_material.comp.glsl`, an emissive hit emits a *second* spectral record with the same `(sid, vi, band)` key carrying `amp·Le` (lines ~719–727), intended to overwrite the pre-scatter arrival beta. But `bdpt_scatter_t5.comp.glsl::scatter_spectral` writes with a **plain store** to the band slot (`t5_light[ob + LGV_BAND_BASE + band] = beta;`, lines ~74/80), and records are processed in parallel by record index with no ordering between the two same-key writers. Result: whether `Le` is applied is **last-writer-wins, nondeterministic**. Emitter colors can flicker frame-to-frame or come out as the raw arrival spectrum. Fix options: apply `Le` in T3 into a single record (no second record), or make the emission record carry a flag and resolve deterministically (e.g. emission always wins via a max/priority store, or sort records so emission is last).

**[S2] Spectral overflow silently desaturates.**
`emit_bdpt_spectral` is capped by `bdpt_max_spectral` and bumped with `atomicAdd` (lines ~418–432). When the cap is hit, the per-band records are dropped, and the affected vertex keeps the **flat luminance split** the pack pass wrote (`betas[b] = throughput / nb`, `bdpt_pack_t5.comp.glsl` lines ~117–121). That fallback is spectrally grey, so under load bright/complex regions lose chroma rather than failing loudly. At minimum, surface the overflow count; ideally size the buffer from a measured high-water mark.

**[S3] Camera `beta_r/g/b` are computed then never used.**
`bdpt_pack_t5` writes `beta_r/g/b = beta_lum/3` into `t5_cam[10..12]` (lines ~125–127, 185–187), but `t5_full_connect` derives camera colour from the per-band slots (`CGV_BAND_BASE`) via `cam_spectral_rgb`. The `[10..12]` values are dead. Harmless, but it's three wasted stores per cam vertex and a trap for the next reader — delete or document.

### B.2 Optical system — the lens

**[S1] "Dispersive" lens is non-dispersive.**
`thick_lens_focus_lab.py::_make_dispersive_lens_bands` (lines ~550–577) forces the index of refraction **flat across all wavelengths**:
```python
# Keep lens IOR spectrally constant here.  The previous Python-side Cauchy
# term injected artificial chromatic fringing in this lab path.
ior = np.full_like(..., float(base_ior), ...)
```
Every band gets the same `ior_real` and `ior_imag = 0`. Consequence: the lens cannot produce **chromatic aberration or any wavelength-dependent focus** — the entire spectral story collapses to monochrome refraction through this element. The comment says a previous Cauchy term was "artificial fringing," but the fix was to remove dispersion entirely rather than use a physically calibrated `n(λ)`. For a thick-lens focus lab this is the single biggest reason spectral effort is invisible in the lens imagery. Recommend a real `n(λ)` (Sellmeier/Cauchy with measured coefficients, or pulled from the materials database — see B.3) instead of a constant.

**[S1] BDPT connections pass straight through refractive glass.**
`bvh_shadow.glsl.inc::shadow_occluded_except` **skips transmissive triangles** (lines ~140–141, `MAT_FLAG_TRANSMISSIVE`) so lens elements don't block connections. This is reasonable for *visibility*, but it means `t5_full_connect` accepts a **straight-line connection between a camera vertex and a light vertex on opposite sides of a lens** and weights it with a vacuum geometry term — ignoring that light actually **bends** (Snell) and **attenuates** (Fresnel) at the interface. A straight segment across a refractive boundary is not a valid light-transport path, yet it is being counted. This injects geometrically invalid energy through the lens and is a primary suspect for "compromised ray tracing" in the optical path. The principled fixes are non-trivial: either forbid connections whose segment crosses a transmissive interface (treat glass as opaque to *connections* even while specular vertices already can't be endpoints), or implement manifold-next-event / specular-connection handling. Short term, the conservative correct behavior is to **reject** rather than straight-connect.

Note the related machinery is already half-present: `scatter_conn_pdf_area` returns 0 for `DELTA_SPECULAR` vertices (`t5_full_connect`, line ~207), so a vertex *on* the glass cannot be a connection endpoint. The gap is connections that *traverse* glass between two non-specular endpoints.

### B.3 Materials — bespoke synthesis where the database should drive

All scene materials are registered into a real `MaterialDatabase` (`thick_lens_focus_lab.py` imports it at line 43, builds at line ~3206, `db.register(...)` throughout ~3208–3411). The problem is **what** is registered: spectral bands are hand-synthesized inline with **flat scalar optical constants**, not measured spectral curves from the database catalog.

**[S2] Metals use a flat, ad-hoc complex index — and it barely affects the image.**
`silver_mirror` (lines ~3356–3376) is built by `_make_sidecar_spectral_bands` with `ior_real = 1.0, ior_imag = 3.0`, constant across bands. Two issues:
1. Real silver has a strongly wavelength-dependent complex index (n ≈ 0.05, k ≈ 3–4 across the visible). `n_real = 1.0` is not metallic at all, and a flat `k` removes silver's characteristic spectral rolloff. This is exactly a **bespoke material object that should be sourced from the materials database** (measured n,k), not authored as two constants.
2. More importantly, the complex index is **nearly inert** for final intensity: `ray_material.comp.glsl::mat_refl_complex` (lines ~238–257) uses `n,k` only to rotate the **phase** of the reflected amplitude — the **magnitude** is the authored `reflectance` field. Betas are reduced to magnitudes (`|amp|`) in the pack pass, so the phase rotation is discarded and `ior_imag` has essentially **no effect on the rendered brightness/colour** of these mirrors. So the metallic-index path is both physically wrong *and* doing nothing. Decide what you want: either drive reflectance magnitude from `n,k` (real Fresnel-for-conductors per band) or stop carrying `ior_imag` for metals.

**[S3] Dielectric/diffuse materials are RGB-upsampled, not spectral.**
`_synth_spectral_bands_from_rgb` (lines ~580–653) turns an `albedo_rgb` triple into per-band reflectance via fixed RGB→λ weights, and glass transmittance is a flat `R₀ = ((n−1)/(n+1))²` with a single `ior`. That's a reasonable fallback, but it is RGB metamerism dressed as spectra — it cannot represent measured pigments, selective absorbers, or dispersive transmission. Where the database has true spectral entries, prefer them; reserve the RGB synth for materials that genuinely only have an RGB albedo.

**Recommendation for B.3:** add a path that, when a registered material name matches a database catalog entry with measured spectral data (glass types, metals, common pigments), pulls the real per-band `reflectance/transmittance/n/k` instead of synthesizing flat constants. The `MaterialDatabase` is already in the loop; this is wiring, not new infrastructure.

### B.4 Numerical robustness

**[S1] Float-CAS accumulator drops energy under contention.**
`t5_full_connect.comp.glsl::atomic_add_float` (lines ~473–483) and the identical one in `sensor_terminal_splat.comp.glsl` (lines ~85–95) spin a `compSwap` loop **bounded to 64 iterations** and then silently give up. With 150B pairs splatting into a few bright pixels (the bright blobs in the reference image), per-pixel contention is extreme and the 64-try cap will be hit, **systematically under-counting the brightest pixels** — i.e. the exact regions you want to look impressive get darkened. Options: raise/remove the cap, reduce contention with the existing per-WG reduction more aggressively, or move to `GL_NV/ARB_shader_atomic_float` where available.

**[S3] `min_geom` floor also culls legitimate faint long-range transport.**
The `geom >= min_geom` gate (`t5_full_connect`, line ~639) is what makes most empty-space pairs cheap (good, and aligned with the triage motivation), but it also removes real low-G connections (distant, grazing). It's a bias/variance knob — fine, just make sure it's exposed and documented as a quality control, not a constant.

**[S3] O(group) serial backward scan in the pack pass.**
`bdpt_pack_t5.comp.glsl` recovers the packed vertex index by scanning backwards over equal-key rows (lines ~104–109). It's correct, but it's a per-thread serial loop whose cost grows with subpath length; for long chains it can dominate the pack pass. If profiling flags it, replace with a segmented-scan/prefix pass.

### B.5 Severity summary

| ID | Area | Severity | One-line |
|---|---|---|---|
| B.1a | Spectral | S1 | Emission `Le` overwrite is a data race (last-writer-wins). |
| B.1b | Spectral | S2 | Spectral-record overflow silently desaturates to grey. |
| B.2a | Optical | S1 | "Dispersive" lens has constant `n(λ)` — no chromatic aberration. |
| B.2b | Optical | S1 | Connections pass straight through glass (no Snell/Fresnel) — invalid paths counted. |
| B.4a | Numerical | S1 | 64-try CAS cap under-counts the brightest pixels. |
| B.3a | Materials | S2 | Metals use flat ad-hoc `n,k` that also barely affects intensity; should come from DB. |
| B.3b | Materials | S3 | Dielectrics are RGB-upsampled, not spectral, even when DB has real data. |
| B.0 | MIS | OK | `f/denom` is correct *iff* betas stay un-normalised — document the invariant. |
| B.1c | Spectral | S3 | Camera `beta_r/g/b` computed but unused. |
| B.4b | Numerical | S3 | `min_geom` culls faint long-range transport — expose as quality knob. |
| B.4c | Perf | S3 | Serial backward scan in pack pass scales with subpath length. |

### B.6 Suggested ordering

1. **B.4a** (CAS cap) and **B.1a** (emission race) first — both are S1, both are small, both directly change pixels you can see.
2. **B.2b** (glass-traversing connections) next — it's the deepest optical-correctness issue; even the conservative "reject" path improves correctness immediately.
3. **B.2a** + **B.3a/b** together — restoring real `n(λ)` and database-driven spectral materials is what finally makes the spectral pipeline *show up* in the lens imagery.
4. **B.1b**, then the S3 cleanups.

---

## Appendix — file placement (for the host team)

Suggested destinations (adjust to your tree):
- `batch_score.comp.glsl` → alongside `t5_full_connect.comp.glsl` in the shader directory (it's loaded the same way, minus the shadow preamble).
- `batch_score_host.cpp.inc` → next to `ray_tracer.cpp`; `#include` it at the marked insertion point, or inline the block.
- `score_histogram.py` → tools/ or scripts/.
