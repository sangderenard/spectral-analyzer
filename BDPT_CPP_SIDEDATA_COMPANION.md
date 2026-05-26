# BDPT C++ Side-Data Companion

This document is the companion note for the current BDPT work. It separates two questions that have been getting conflated:

1. What is still missing for a proper BDPT estimator.
2. Where the C++ ray trace code should carry the extra data needed to make that estimator real.

The main rule is simple: do not put BDPT side-data into the existing GLSL render channels. The GPU kernels already have tightly packed float layouts and the current 8-channel/render payload budget is effectively occupied. BDPT needs its own buffers, record queues, and drain path.

## Current State

The current middle PIP is not yet a full BDPT implementation. It is an endpoint connector with some important foundations added around it:

- Continuous film UV is now carried instead of only a pixel id.
- Camera-side sample facts and simple PDFs exist in Python records.
- Optical transfer diagnostics and Fresnel/throughput records exist on the Python side.
- Segment records exist as a first bridge away from endpoint-only storage.
- A native C++ lens optics/Jacobian helper has been introduced, but it is not yet the central source of all ray-pipeline BDPT side-data.
- `csrc/include/bdpt_record.h` already contains a useful first C-side home: `PathVertex` and `EndpointRecord`.

The current estimator still connects selected light-side and sensor-side endpoint records. That can produce a useful diagnostic image, but it is not yet the BDPT estimator we ultimately want.

## Still Missing For Proper BDPT

Proper BDPT needs complete subpaths and per-strategy probability accounting. The missing pieces are:

- Full path vertices for both light and camera subpaths, not just endpoint rows.
- Stable `subpath_id`, `vertex_index`, `stream`, `bounce`, and `strategy_id` identities that survive CPU/GPU pipeline routing.
- Per-event forward PDFs and reverse PDFs.
- Explicit sample measure for every PDF: area, solid angle, projected solid angle, film area, aperture area, wavelength probability, shutter probability, or discrete strategy probability.
- Correct measure conversion between solid angle and area at connections.
- Geometry terms at every connection: distance squared, endpoint normals, cosines, visibility, and delta/specular eligibility.
- Accumulated throughput/beta per spectral band, including lens throughput, material response, Fresnel factors, absorption, geometric spreading, and phase.
- Correct handling of delta events through specular scene materials and lens surfaces.
- MIS over all connectable `(s,t)` strategies, not just one endpoint-connection strategy.
- A path assembly layer that consumes records and builds candidate paths before evaluating MIS.
- Overflow/lifetime policy for native side-data buffers so the estimator never silently becomes top-k history.

Until those exist, the system can be a strong optical transport diagnostic and an approximate bidirectional connector, but not a proper BDPT estimator.

## Why The Side-Data Must Stand Alone

The GLSL path currently passes ray and material data through dense flat float records. For example, `ray_material.comp.glsl` uses:

- `RefinedHit` as `26 + 2 * MAX_BANDS` floats.
- `RayIntent` as `20 + 2 * MAX_BANDS` floats.
- `TerminalRecord` as `26 + 2 * MAX_BANDS` floats.

Those layouts already carry position, direction, path length, tag, color flag, medium index, source id, bounce budget, sensor origin, and spectral complex amplitudes. Trying to squeeze BDPT PDFs, Jacobians, MIS domains, optical-event detail, and full path vertices into those same channels would force channel swapping and make the data contract fragile.

The right design is:

- Keep `RayIntent` minimal and focused on transport.
- Keep `RayRecord` useful for display/drain/debug.
- Use `tag` or a small sidecar handle only for correlation.
- Store BDPT-specific data in dedicated native side buffers.
- Give GPU kernels separate SSBO bindings and append counters for BDPT records.
- Drain BDPT records separately from `drain_records()` and `drain_records_slim()`.

In short: the render payload should not become the BDPT database.

## Existing C++ Anchors

These files are the natural integration points:

- `csrc/include/bdpt_record.h`
  - Already defines `PathVertex` and `EndpointRecord`.
  - This should become the shared ABI home for all BDPT record structs.

- `csrc/include/ray_pipeline.h`
  - `RayIntent` should only gain minimal correlation fields if needed.
  - `RayRecord` should not be expanded into a full BDPT vertex database.
  - Add native side queues to `RayPipelineState`, not broad payload fields to every ray.

- `csrc/kernels/ray_tracer.cpp`
  - T1 intersection has the geometry needed for path vertices: hit point, normal, triangle id, group id, barycentrics, incoming direction, segment start, and path length.
  - T2/refinement and parametric lens transfer are where optical events and lens Jacobian records belong.
  - T3 material scattering is where BSDF/event PDFs, Fresnel weights, delta flags, and throughput updates belong.
  - The current BDPT snap/endpoint logic should be treated as diagnostic until the side records drive the estimator.

- `csrc/kernels/lens_optics.cpp`
  - This is the right native home for heavy lens differential work.
  - Its output should feed optical/camera side records, not only Python diagnostics.

- `csrc/bindings/pybind_kernels.cpp`
  - Add separate drains such as `drain_bdpt_vertices()`, `drain_bdpt_optical_events()`, and `drain_bdpt_strategy_records()`.
  - Do not overload `drain_records_slim()` with hidden BDPT semantics.

- `csrc/shaders/*.comp.glsl`
  - If GPU kernels emit BDPT records, add dedicated SSBOs and counters.
  - Do not reuse the current ray/material float channel layout for BDPT state.

## Proposed Native BDPT Records

Extend `bdpt_record.h` with separate, fixed-layout records. Keep them std430-friendly and NumPy-mappable.

### BdptVertexRecord

One row per path vertex.

Fields:

- `uint32_t subpath_id`
- `uint16_t vertex_index`
- `uint8_t stream`
- `uint8_t sample_domain`
- `uint32_t flags`
- `uint32_t strategy_id`
- `int32_t tri_id`
- `int32_t group_id`
- `int32_t mat_idx`
- `float pos[3]`
- `float normal[3]`
- `float dir_in[3]`
- `float dir_out[3]`
- `float path_len`
- `float path_at_seg_start`
- `float pdf_fwd`
- `float pdf_rev`
- `float pdf_area`
- `float pdf_solid_angle`
- `float throughput_scalar`

This is the real replacement for endpoint guessing. It gives the estimator the whole subpath.

### BdptSpectralWeightRecord

One row per `(subpath_id, vertex_index, band)` when full complex spectral weight is needed.

Fields:

- `uint32_t subpath_id`
- `uint16_t vertex_index`
- `uint16_t band_id`
- `float beta_re`
- `float beta_im`
- `float wavelength_or_center_hz`
- `float band_pdf`
- `float sensor_rgb_weight`

This avoids bloating every vertex by `2 * MAX_BANDS` floats when a vertex only needs scalar geometry data for MIS.

### BdptPdfRecord

One row per sampling event where the measure conversion matters.

Fields:

- `uint32_t subpath_id`
- `uint16_t vertex_index`
- `uint8_t sample_domain`
- `uint8_t measure`
- `float pdf_fwd`
- `float pdf_rev`
- `float pdf_area`
- `float pdf_solid_angle`
- `float jacobian_det`
- `float geometry_term`
- `uint32_t flags`

This is the record MIS should consume first.

### BdptOpticalEventRecord

One row per lens/interface/stop event.

Fields:

- `uint32_t subpath_id`
- `uint16_t vertex_index`
- `uint16_t element_index`
- `uint8_t reason`
- `uint8_t stream`
- `uint16_t flags`
- `float pos[3]`
- `float normal[3]`
- `float dir_in[3]`
- `float dir_out[3]`
- `float cos_incident`
- `float cos_transmitted`
- `float eta_i`
- `float eta_t`
- `float fresnel_reflectance`
- `float transmittance`
- `float throughput_multiplier`
- `float opl`
- `float geom_len`
- `float aperture_radius`
- `float transverse_radius`
- `float distance_past_clear_aperture`
- `float phase_space_jacobian`

This record answers whether poor images come from sampling, lens clipping, TIR, bad aim, missing throughput, or scene connection logic.

### BdptConnectionRecord

One row per attempted connection candidate.

Fields:

- `uint32_t camera_subpath_id`
- `uint32_t light_subpath_id`
- `uint16_t camera_vertex_index`
- `uint16_t light_vertex_index`
- `uint16_t strategy_s`
- `uint16_t strategy_t`
- `uint32_t flags`
- `float p0[3]`
- `float p1[3]`
- `float dist2`
- `float cos_camera`
- `float cos_light`
- `float geometry_term`
- `float visibility`
- `float strategy_pdf`
- `float mis_weight`

This can be added after the vertex/PDF records exist. It is useful for debugging the estimator without recomputing every candidate in Python.

## Pipeline Integration Plan

### 1. Add Side Queues To RayPipelineState

In `ray_tracer.cpp`, add queues and counters next to `Q_out`, not inside `RayRecord`.

Proposed shape:

```cpp
PipelineQueue<BdptVertexRecord>       Q_bdpt_vertices;
PipelineQueue<BdptSpectralWeightRecord> Q_bdpt_spectral;
PipelineQueue<BdptPdfRecord>          Q_bdpt_pdfs;
PipelineQueue<BdptOpticalEventRecord> Q_bdpt_optical;
```

For high-rate GPU output, use preallocated vectors or SSBO-backed append buffers with atomic counters rather than one queue push per event.

### 2. Give RayIntent A Sidecar Handle, Not A Payload Dump

`RayIntent` can carry a compact identity:

```cpp
uint32_t subpath_id;
uint16_t vertex_index;
uint8_t  bdpt_stream;
uint8_t  strategy_id;
```

If binary compatibility churn is a concern, first derive these from `tag`, `src_id`, `bounce`, and `color_flag`. The important part is that the full BDPT state lives outside `RayIntent`.

### 3. Emit Camera Launch Records At Submission

At camera/backward launch, record:

- film UV in continuous coordinates
- aperture sample position
- shutter transmission
- wavelength/band probability
- sensor RGB sensitivity
- film area PDF
- aperture area PDF
- aperture solid-angle PDF after the optical transform is known
- total camera strategy PDF

The current Python-side `trace_sensor_cast()` logic is the conceptual source. The native destination should be a camera-launch or first `BdptPdfRecord`, plus a `BdptVertexRecord` for the film/aperture endpoint if it participates in path assembly.

### 4. Emit Geometry Vertices In T1

T1 already has enough data to form a vertex record:

- segment start
- hit position
- incoming direction
- hit normal
- triangle id
- group id
- barycentric coordinates
- path length

Write `BdptVertexRecord` here or immediately after refinement if the parametric normal/position replaces the mesh hit.

### 5. Emit Optical Events In T2/Lens Transfer

`apply_parametric_lens_from_f32()` and the newer `lens_optics.cpp` logic should populate `BdptOpticalEventRecord`.

Required outputs:

- element index
- termination reason
- interface hit point
- interface normal
- input/output ray direction
- Fresnel reflectance/transmittance
- eta ratio
- cosines
- OPL and geometric length
- stop/clipping geometry
- phase-space Jacobian

This keeps the lens transport parametric while giving BDPT the missing differential side-data.

### 6. Emit Scattering PDFs In T3

T3 owns the material event. It should record:

- selected lobe/domain
- discrete lobe probability
- BSDF or phase-function PDF
- reverse PDF
- throughput multiplier
- delta/specular flag
- absorption/termination reason

This is where endpoint-only BDPT currently loses correctness. A connection estimator cannot repair missing per-event PDFs after the fact.

### 7. Add Pybind Drains

Expose the side queues separately:

```cpp
drain_bdpt_vertices(max_n)
drain_bdpt_spectral(max_n)
drain_bdpt_pdfs(max_n)
drain_bdpt_optical_events(max_n)
drain_bdpt_connections(max_n)
```

Each drain should return a NumPy structured array matching `bdpt_record.h`. Keep `drain_records()` and `drain_records_slim()` as transport/debug drains.

### 8. Add GLSL SSBOs Only When The CPU Contract Is Stable

When GPU emission is needed, add separate SSBO bindings:

```glsl
layout(std430, binding = BDPT_VERTEX_BINDING) coherent buffer BdptVertexBuf { float bdpt_vertices[]; };
layout(std430, binding = BDPT_PDF_BINDING) coherent buffer BdptPdfBuf { float bdpt_pdfs[]; };
layout(std430, binding = BDPT_OPTICAL_BINDING) coherent buffer BdptOpticalBuf { float bdpt_optical[]; };
layout(std430, binding = BDPT_COUNTER_BINDING) coherent buffer BdptCounterBuf { uint counters[]; };
```

Use atomic append offsets. On overflow, increment an overflow counter and stop writing that record class for the dispatch. Do not silently wrap.

## Estimator Integration

The estimator should consume records in this order:

1. Group `BdptVertexRecord` by `(stream, subpath_id)`.
2. Join spectral weights by `(subpath_id, vertex_index, band)`.
3. Join PDF records by `(subpath_id, vertex_index, sample_domain)`.
4. Join optical events for camera-side validation and throughput.
5. Enumerate connectable `(s,t)` strategies.
6. Test visibility.
7. Evaluate contribution:
   - camera beta
   - light beta
   - geometry term
   - visibility
   - emission or sensor response
   - MIS weight
8. Accumulate to continuous film UV, then rasterize for display.

The old endpoint connector can remain as a diagnostic path while this record-driven estimator comes online. It should not be the long-term middle PIP estimator.

## Immediate Native Work Items

1. Close any process locking `_spectral_kernels.cp311-win_amd64.pyd` and rebuild the native extension.

   ```powershell
   cmake --build csrc_build --config Release --target _spectral_kernels
   ```

2. Move the C++ lens Jacobian path from optional helper status into the camera record path.

3. Extend `bdpt_record.h` with the side-data structs above.

4. Add native side queues/counters to `RayPipelineState`.

5. Add CPU emission first:
   - T1 emits vertex records.
   - T2/lens transfer emits optical event records.
   - T3 emits PDF/scatter records.

6. Add pybind drains for those queues.

7. Update Python BDPT assembly to read the new side records while keeping the endpoint connector as a fallback view.

8. Add GPU SSBO emission only after the CPU record ABI is stable.

## Non-Goals For The Next Patch

- Do not widen the current GLSL `RayIntent`, `RefinedHit`, or `TerminalRecord` layouts with full BDPT payloads.
- Do not replace continuous film UV with pixel ids.
- Do not collapse spectral complex weights into magnitude-only rows.
- Do not make the endpoint cache the source of truth for MIS.
- Do not hide buffer overflow by truncating without a counter.

## Bottom Line

The next correct step is not to stuff more values into the existing channel budget. The next correct step is a native BDPT sidecar data path:

```text
transport ray
  -> minimal identity handle
  -> standalone BDPT vertex/pdf/optical/spectral records
  -> record-driven path assembly
  -> MIS
  -> continuous film accumulation
```

That keeps the parametric optical transport intact, gives the estimator every missing probability and differential fact, and avoids turning the GLSL ray payload into an unstable overloaded channel scheme.
