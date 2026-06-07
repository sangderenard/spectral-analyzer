# BDPT GPU T5 Work Report - 2026-06-07

This document records the current uncommitted renderer work around the
GPU-resident BDPT/T5 path.  The main objective was to stop treating T5 as a
CPU/readback diagnostic path and make the GPU path carry the records needed for
real BDPT/MIS: camera/sensor records, flash/light/emissive records, spectral
throughput, per-edge PDFs, optical state, material emission, and enough
profiling/batching control to survive Windows GPU timeout behavior.

## Scope

The work touches:

- GPU BDPT/T5 scheduling and record lifetime in `csrc/kernels/ray_tracer.cpp`.
- Public T5 configuration and Python bindings in `csrc/include/ray_pipeline.h`
  and `csrc/bindings/pybind_kernels.cpp`.
- T5 packing, side-record scattering, and all-pairs connection shaders in
  `csrc/shaders/bdpt_pack_t5.comp.glsl`,
  `csrc/shaders/bdpt_scatter_t5.comp.glsl`, and
  `csrc/shaders/t5_full_connect.comp.glsl`.
- Direct camera-visible emissive splatting in
  `csrc/shaders/sensor_terminal_splat.comp.glsl`.
- Material emission propagation in `csrc/shaders/ray_material.comp.glsl`.
- Shadow visibility endpoint handling in `csrc/shaders/bvh_shadow.glsl.inc`.
- Driver/profile/display support in `thick_lens_focus_lab.py`.

## GPU-Resident T5 Scheduling

Before this work, GPU mode could still route the BDPT connection latch through
the CPU/KPN sentinel path or clear records between flash and sensor families.
That made it possible for one family of records to be absent when T5 finally
ran.

The scheduling changes now do the following:

- When `gpu_skip_record_readback` is enabled and both flash/light and sensor
  families are exhausted for a BDPT cycle, `bdpt_gpu_t5_pending` is set and the
  GPU-native T5 service handles sort, pack, scatter, and connect directly.
- The CPU T5 sentinel path is bypassed in GPU-resident production mode.
- `bdpt_t5_fired` is advanced when the GPU-native service finishes its cycle,
  so the next exposure can be latched without reprocessing stale records.
- `bdpt_family_finish()` also schedules `bdpt_gpu_t5_pending` in GPU-resident
  mode instead of pushing CPU T5 work.

This keeps the BDPT record flow on GPU and makes the expected handoff explicit:
flash/light records plus sensor/camera records go into T5 together.

## BDPT Record Lifetime

The GPU-resident bounce loop now tracks a BDPT accumulation generation.  BDPT
vertex, spectral, PDF, and optical record counters are cleared only at the start
of a new BDPT cycle.  They are not cleared between separate flash and sensor
family dispatches for the same cycle.

This matters because T5 needs all streams at once.  Clearing counters between
families would make T5 see only whichever family ran last.

The GPU-resident path also avoids per-bounce UV accumulator readback and avoids
CPU requeue behavior for child rays when the GPU-side post-T3 pass already owns
the next bounce.

## T5 Pack And Scatter Split

`bdpt_pack_t5.comp.glsl` was changed to remain strictly O(vertices).  It now
packs the sorted BDPT vertices into light and camera T5 vertex buffers and
initializes fields that are later filled from side records.

The expensive side-record work was moved out into the new
`bdpt_scatter_t5.comp.glsl` shader.  That shader runs in bounded chunks and
supports three modes:

- `scatter_mode=0`: scatter spectral side records into per-band T5 beta fields.
- `scatter_mode=2`: scatter PDF side records and convert edge PDFs into
  area-measure edge PDFs for T5 MIS.
- `scatter_mode=1`: scatter optical side records and apply optical blocking and
  optical Jacobian data.

The host dispatch order is spectral, then PDF, then optical.  Optical scatter
runs after PDF scatter so optical Jacobians can adjust the edge PDFs already
written into the packed T5 rows.

This split was required because doing side-record scans inside pack caused
unbounded work in a single shader dispatch and triggered Windows timeout
behavior at realistic record counts.

## Edge PDF Area Conversion

T5 expects edge PDFs in area measure for MIS denominator construction.  The pack
shader no longer substitutes a local diffuse approximation.  PDF records are now
used by the scatter shader to fill:

- `LGV_EDGE_FWD_AREA`
- `LGV_EDGE_BWD_AREA`
- `CGV_EDGE_FWD_AREA`
- `CGV_EDGE_BWD_AREA`

The scatter shader prefers recorded area PDF data when available, otherwise it
converts solid-angle/projected-solid-angle PDF data to area measure using the
geometry between the current and next packed vertex.

This was one of the major correctness repairs.  Packing solid-angle PDFs into
fields later interpreted as area PDFs can corrupt MIS denominators and produce
extreme fireflies or underweighted contributions.

## Spectral And Material Emission Handling

The T5 vertex buffers now carry per-band beta data through the GPU path instead
of relying only on scalar throughput placeholders.

Material emission handling was added in two places:

- `ray_material.comp.glsl` now exposes `mat_emission(mat, band)` from material
  band field 5 and multiplies emitted terminal BDPT spectral records by that
  material emission.
- `sensor_terminal_splat.comp.glsl` now binds the material band buffer and
  multiplies camera-visible terminal emissive hits by the material spectral
  emission before display RGB conversion.

The terminal splat scale was restored to `1.0`, so direct visible emission is
not artificially dimmed by the previous diagnostic scale.

## Direct Visible Emissive Surfaces

Visible emissive surfaces should not require an all-pairs T5 connection to show
up on the film.  The terminal splat pass now accumulates backward sensor rays
whose terminal record is an emissive material hit.

This path is still a real sensor path contribution: it uses the terminal record
generated by the ray traversal/material path and deposits the terminal spectrum
onto the sensor film.  It is not a fake composite and does not replace T5 MIS
connections.

This is especially important for delta chains.  A sensor path that reaches an
emitter through mirror/glass/refraction events must be allowed to deposit
emission when it terminates at the emitter.

## Delta Chain Acceptance

T5 vertex validity no longer rejects vertices solely because their PDF record is
marked `BDPT_PDF_FLAG_DELTA_SPECULAR`.

This change exists in:

- GPU `vertex_connectable()` in `t5_full_connect.comp.glsl`.
- CPU reference `T5ConnContext::vertex_connectable()` in
  `csrc/include/t5_connection.h`.

The remaining delta behavior is intentional: arbitrary finite-area connections
through a delta lobe still evaluate to zero connection PDF.  That is physically
correct unless the connection direction exactly matches the deterministic
specular/refraction event.  In other words, delta vertices are accepted as part
of propagated path chains, but arbitrary pair connections are not faked through
delta lobes.

If this is still insufficient for the target renderer, the next real feature is
a deterministic delta-chain connection mode that follows already-sampled delta
edges to a non-delta endpoint or emitter, rather than treating the delta vertex
as generally connectable to any target.

## T5 Connect Shader Changes

`t5_full_connect.comp.glsl` was updated to:

- Use the packed `tri_id` field for endpoint-aware visibility.
- Avoid rejecting delta vertices at vertex-validity time.
- Build camera and light spectral RGB from per-band beta fields.
- Use scalar transport for the BDPT contribution and apply camera spectral
  color once to the final pixel contribution.
- Keep MIS denominator sane by forcing `denom >= selected_pdf` and rejecting
  non-finite or tiny denominator cases.
- Move the shadow test after connection PDFs and MIS work so `--profile` can
  isolate which stage is expensive.
- Use `shadow_occluded_except()` so a connection is not falsely blocked by the
  exact endpoint triangles it is connecting.
- Deposit T5 output with a normalized 2x2 tent splat instead of a hard integer
  pixel deposit.
- Add `t5_profile_mode` stage exits for profiling.

The current shader tile is set to `1x1` as a conservative Windows-timeout
diagnostic setting.  The host-side adaptive pair batching can grow the number
of pairs per dispatch, but the shader workgroup dimensions remain conservative.

## T5 Batching And Profiling

The host now dispatches T5 pairs through `dispatch_t5_pair_chunks()`.  It starts
with a conservative pair cap, fences each dispatch, shrinks on timeout/long
runtime, and grows toward the requested batch size when dispatches are well
under budget.

Current constants:

- Start pair cap: `1024`.
- Minimum pair cap: `256`.
- Maximum pair cap: `1048576`.
- Fence budget: `750 ms`.
- Growth target: `350 ms`.
- Maximum growth step: `2.0x`.

The profile mode exposed through Python prints staged GPU timing with a 1 second
timeout.  T5 stage labels are:

- `sort-init`
- `sort-bitonic`
- `pack-t5`
- `scatter-spectral`
- `scatter-pdf`
- `scatter-optical`
- `pre-connect-queue`
- `load`
- `spectral`
- `geometry`
- `connection-pdfs`
- `mis`
- `shadow`

When `--profile` is enabled, the Python driver also prints selected drain,
BDPT feed, visualization, in-flight wait, and `join_t5` timings.

## Shadow Visibility

`bvh_shadow.glsl.inc` now provides `shadow_occluded_except(p0, p1, skip_tri0,
skip_tri1)`.  T5 uses this to skip the camera and light endpoint triangles
during visibility testing.

Transmissive surfaces remain non-blocking for BDPT visibility, while aperture
stop triangles remain blockers.

## Python Driver And Display Changes

`thick_lens_focus_lab.py` now has:

- `--gpu-resident` to enable GPU-side bounce looping/readback skipping when
  running GPU or mixed compute mode.
- `--profile` integration with `set_t5_profile(True)`.
- Cached PIP image getters keyed by version counters, reducing repeated numpy
  normalization and texture upload when images have not changed.
- Version counter updates when forward, reverse, or camera-perspective
  accumulators change or are reset.
- A sensor coordinate sign fix for camera-perspective binning.
- PIP texture upload skips when the getter returns the same cached image object.

These changes are performance and observability support; they should not be
used as evidence that BDPT lighting is correct.

## Public API Changes

The C API now includes:

- `ray_pipeline_set_t5_profile(RayPipelineState* ps, bool v)`

The Python binding now exposes:

- `PyRayTracer.set_t5_profile(v)`

The existing batch-size controls remain:

- `set_t5_light_batch_size(n)`
- `set_t5_cam_batch_size(n)`
- `set_t5_sensor_tile_size(n)`

## New Shader Files

Two shader files are currently untracked additions intended to be committed:

- `csrc/shaders/bdpt_scatter_t5.comp.glsl`
  - Required for the current T5 pack/scatter architecture.
  - Scatters spectral, PDF, and optical side records into already-packed T5
    vertex rows.

- `csrc/shaders/t0_intent_pack.comp.glsl`
  - GPU-side raw `RayIntent` to packed intent-buffer conversion helper.
  - This is related to GPU-resident startup/packing work and is not currently
    the main T5 correctness point.

The empty untracked file named `r` is not part of the renderer work and should
not be included in the commit.

## Known Limits After This Work

This commit does not make the renderer complete.

Remaining limits include:

- Arbitrary MIS connections through delta lobes are still zero.  That is
  physically correct, but full high-quality caustics need deterministic
  delta-chain handling and/or specialized sampling, not a fake PDF.
- T5 is still an all-pairs connection system.  It can be expensive even when
  batched safely.
- The `1x1` T5 shader tile is conservative and likely leaves performance on the
  table.  It was chosen to avoid Windows timeout crashes while isolating the
  long shader stages.
- Direct visible emission is handled, but next-event estimation from camera or
  scatter vertices to sampled emissive surfaces is not yet implemented.
- Film reconstruction is currently a normalized 2x2 tent splat, not a full
  reconstruction/filtering pipeline with a separate weight buffer.
- Firefly prevention is mostly denominator hygiene; a principled temporary
  contribution clamp is not yet installed.
- The CPU T5 path is not considered the production reference for current GPU
  work.

## Validation Performed

Recent checks passed before this report was written:

- GLSL validation for `sensor_terminal_splat.comp.glsl`.
- GLSL validation for combined `t5_full_connect.comp.glsl`.
- `_spectral_kernels` build.
- Python extension import smoke.
- `git diff --check`, with only CRLF warnings reported.

## Practical Next Debug Targets

The next useful debugging pass should measure and report:

- Number of terminal emissive hits from sensor paths, split by material id and
  whether the path contained delta/specular/refraction events.
- Number of T5 candidate pairs rejected by connectability, zero geometry, zero
  connection PDF, failed MIS denominator, and shadow occlusion.
- Distribution of `selected_pdf`, `denom`, `geom`, and final contribution for
  accepted T5 pairs.
- Count of delta-flagged vertices present in packed camera and light T5 buffers.
- Whether clear/emissive objects rendering black are missing terminal emission,
  being blocked by optical flags, or carrying zero material emission bands.

Those counters should be GPU-side reductions or compact debug buffers; avoid
large per-record readback in production mode.
