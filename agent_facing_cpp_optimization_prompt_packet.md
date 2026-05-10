# Agent-Facing Optimization Prompt Packet: Ray Tracer / Field Marcher / Surface Spline

## Use case

This document is intended to be pasted into another coding agent as a dense implementation prompt. It is not a slide deck. Do not convert it into a diagram, infographic, or explanatory visual. Treat it as a technical audit plus implementation contract for optimizing a developing C++ numerical/graphics/physics codebase that currently prioritizes correctness/proof-of-life over throughput, memory discipline, and large-dataset survivability.

The agent receiving this prompt should assume the code is in active development, has valuable physics intent embedded in it, and likely contains many structures that exist because they were the shortest path to “it technically works.” Preserve physics semantics first. Optimize second. Do not remove features simply because they are expensive. Replace eager, global, allocating, or serial patterns with bounded, tiled, pooled, streaming, cache-aware, and testable equivalents.

Primary files under review:

- `ray_tracer.cpp`: complex spectral 3-D ray tracer using Eigen vectors, `std::complex<double>`, BVH triangle intersection, per-band ray amplitudes, material spectral buffer, stateful live-ray queues, camera strike capture, field-grid accumulation, and scale-context dispatch.
- `field_march.cpp`: dense complex spectral field grid and stencil marcher, regular and k-d tree grid forms, complex64 storage, reference-quality Helmholtz stepping, amplitude injection, and raw buffer exposure for Python / GLSL parity.
- `surface_spline.cpp`: Eigen quadratic POLY_BARY surface fitter using triangle 1-ring neighborhoods, per-triangle small least-squares solves, and a thread pool wrapper.

Do not summarize this as a vague “make it faster” task. Implement a disciplined performance hardening pass.

---

## Current architecture facts the implementation must respect

### Ray tracer facts

`ray_tracer.cpp` is a complex spectral ray tracer. Each ray carries a per-frequency-band complex amplitude vector. Propagation applies phase accumulation, atmospheric decay, and geometric spreading. Surface hits multiply by material complex reflectance. Direction sampling uses Fibonacci sphere directions rotated toward the source direction and optionally filtered by cosine-power directivity. Diffuse scatter is stochastic and shares the same spectral update path as specular reflection.

The file already contains an axis-aligned BVH built at scene load time and queried during tracing. Triangle records already precompute `edge1`, `edge2`, and `normal`, which is correct and should be preserved. The BVH build is median centroid split on longest axis with very small leaf size. The traversal uses a fixed stack and tests leaves with Möller-Trumbore.

The output segment path writes one record per `(source, ray, bounce, band)` into a caller-provided float buffer. Overflow is currently silently dropped for segment visualization in at least one path. That may be acceptable for visualization, but it is dangerous if reused for scientific or dataset generation output.

`RayTracerState` is large and acts as a central accumulation point: triangles, BVH, material buffer, scale contexts, live ray scheduler queues, group registries, camera visibility settings, camera field grid, camera strike rows, and Python-ingested sensor/film tensors live in this state. This is convenient but needs memory discipline and clearer ownership boundaries.

### Field marcher facts

`field_march.cpp` stores a complex spectral field as `std::complex<float>` with shape roughly `[band][cell]`. It supports regular grids and k-d/leaf chunk grids. The regular path is byte-compatible with a paired `r32f` real/imag layout for GPU/SSBO parity. This parity is important; do not break it casually.

Regular grid allocation currently computes `n_cells_total = nx * ny * nz` and then allocates `n_cells_total * n_bands` complex cells in one contiguous allocation. That is simple and correct for moderate sizes, but it is not safe for very large worlds, many bands, or high-resolution grids unless guarded by explicit size checks and/or replaced by streaming/tiled backing.

The Helmholtz step is explicitly marked reference-quality, clear but slow. It allocates one `std::vector<cd32> tmp(nx*ny*nz)` and copies each band into it each step, then loops through `z,y,x`. This is the main rewrite target for memory bandwidth, tiling, threading, and out-of-core safety.

Amplitude injection is trilinear. The regular version writes directly to `g->data`. The k-d version descends to a leaf and deposits into that leaf’s contiguous region. These direct writes are not thread-safe if multiple rays/threads inject concurrently.

### Surface spline facts

`surface_spline.cpp` builds `vertex -> triangles` adjacency, then fits one quadratic polynomial per triangle using small Eigen least-squares systems. It uses a `ThreadPool::parallel_for` over triangles. It allocates dynamic vectors and dynamic Eigen matrices inside every per-triangle fit. It also uses `unordered_set` in `one_ring`, which is a red flag for per-triangle hot loops.

The fit dimension is fixed at 6 coefficients. The number of samples is small and bounded by the local 1-ring plus vertex normal constraints. Therefore the implementation should migrate from dynamic `MatrixXd`/`VectorXd` and per-fit heap containers toward stack/local fixed-size or small-buffer structures.

---

## Global implementation directives

### Preserve semantics before optimizing

Do not collapse complex amplitudes into magnitudes. Do not remove bands. Do not quantize spectral state unless a separate explicitly named approximate mode is introduced. Do not treat visualization overflow policy as acceptable for scientific data. Do not eliminate diffuse/specular behavior, material phase, scale contexts, camera capture, or field injection just because they complicate scheduling.

### Add a measurement baseline before rewriting

Before patching large sections, add a reproducible benchmark harness for the three kernels:

1. Ray tracing benchmark:
   - fixed seed
   - fixed mesh
   - configurable sources, rays, bounces, bands
   - metrics: rays/sec, intersections/sec, bounces/sec, output records/sec, memory bandwidth estimate, time in BVH traversal, time in spectral propagation, time in material lookup, time in output writing, time in field/camera capture if enabled

2. Field marcher benchmark:
   - configurable `(nx, ny, nz, n_bands, n_steps)`
   - regular grid and k-d leaf cases
   - metrics: cell-updates/sec, bytes/sec, allocation bytes, copy bytes, time per band, time per step
   - record whether the benchmark is memory-bandwidth bound or compute-bound

3. Surface spline benchmark:
   - configurable mesh size and average valence
   - metrics: triangles/sec, allocation count, adjacency build time, fit time, solve time, percentage rank-deficient, number of skipped degenerate triangles

Add timers around kernels, not around Python wrapper overhead only. If available, use platform counters or at least allocation instrumentation. Do not trust perceived speed.

### Introduce explicit modes

Separate “debug/reference/exact” from “production/streamed/parallel” modes. The existing slow implementations should remain available as correctness or regression references where feasible.

Recommended modes:

```cpp
enum class KernelMode {
    ReferenceSingleThread,
    ProductionThreaded,
    ProductionStreaming,
    ApproximatePreview
};
```

Not all files need that exact enum, but the architecture should make it clear when a path is allowed to drop data, approximate, or stream partial results.

### Treat output and datasets as streams, not always as vectors

Any path that can generate more data than RAM must have a streaming sink option. The code should not require all ray segments, endpoints, camera strikes, or field snapshots to exist in a single contiguous vector before Python sees them.

Introduce a sink interface:

```cpp
struct RtRecordSink {
    virtual ~RtRecordSink() = default;
    virtual bool push_segments(const float* records, size_t n_records) = 0;
    virtual bool flush() = 0;
    virtual bool failed() const = 0;
};
```

Then implement:

- `MemoryRecordSink`: current behavior, bounded by caller capacity.
- `ChunkedMemorySink`: vector of fixed-size chunks, avoids giant reallocations.
- `MMapRecordSink`: writes records to a memory-mapped file.
- `FileRecordSink`: buffered binary append, ideally with metadata sidecar.
- `PythonCallbackSink`: optional, only if GIL and callback overhead are carefully isolated.
- `NullCountingSink`: for performance tests and capacity estimation.

Do not use the same overflow behavior for all sinks. For scientific output, overflow must return an error or set a truncation flag. For preview output, overflow may drop records only when explicitly requested.

---

## Large dataset / crash-avoidance requirements

This is a first-class requirement. The system appears to be heading toward huge ray, field, spectral, and camera datasets. Preventing crashes requires bounded memory, streaming, and explicit backpressure.

### Add checked multiplication everywhere sizes are computed

Every allocation size derived from dimensions must be checked before multiplication.

Examples:

```cpp
bool checked_mul_size(size_t a, size_t b, size_t& out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b;
    return true;
}
```

Use this for:

- `nx * ny * nz`
- `n_cells_total * n_bands`
- `records * stride`
- `n_tris * 6`
- `n_mats * MAX_SPECTRAL_BANDS * stride`
- `n_slots * 2`
- `ray_count * n_bands`
- `n_sources * n_rays * max_bounces * n_bands`

Also check conversion from signed integers to `size_t`. Reject negative or impossible values before casting.

### Add budgeted allocation policies

Do not let callers accidentally request a 200 GB field grid and crash the process.

Add a runtime memory budget structure:

```cpp
struct MemoryBudget {
    uint64_t max_bytes_total;
    uint64_t max_bytes_field_grid;
    uint64_t max_bytes_output_buffer;
    uint64_t max_bytes_temp_workspace;
    uint64_t max_records_per_chunk;
    bool allow_out_of_core;
};
```

Expose defaults that are safe. Let Python override them deliberately.

Any allocation request must either:

- fit the budget,
- switch to streaming/out-of-core mode,
- or fail gracefully with an error code that reports required bytes.

### Replace single giant output vectors with chunks

`camera_strike_rows` and similar unbounded vectors should become chunked append buffers.

Use fixed-size chunks such as 1–64 MB. Avoid repeated reallocation of one giant `std::vector<float>`.

Sketch:

```cpp
template<class T>
class ChunkedAppendBuffer {
public:
    explicit ChunkedAppendBuffer(size_t items_per_chunk);
    T* reserve_row(size_t n_items);
    size_t total_items() const;
    void clear_keep_capacity();
    // iteration over chunks for streaming / Python export
};
```

Benefits:

- avoids contiguous allocation failure,
- reduces copy storms,
- permits partial flushing to disk,
- permits bounded memory operation,
- makes backpressure natural.

### Add streaming output to ray tracing

The ray tracer can produce enormous records: `sources * rays * bounces * bands`. Do not require a caller to estimate `out_cap` and hope.

Recommended API additions:

```cpp
extern "C" SK_API int ray_tracer_trace_streamed(
    RayTracerState* st,
    const RayTraceConfig* cfg,
    RtRecordSinkHandle sink,
    RtTraceStats* stats);
```

Or, if C ABI constraints are strict, use function pointers:

```cpp
typedef int (*RtSegmentCallback)(
    const float* records,
    int n_records,
    void* user);

extern "C" SK_API int ray_tracer_trace_callback(
    RayTracerState* st,
    ...,
    RtSegmentCallback cb,
    void* user,
    RtTraceStats* stats);
```

Internally each worker writes to a thread-local segment buffer. When full, it flushes to the sink. The sink must either be thread-safe or receive records via a single consumer queue.

### Add backpressure

If streaming to disk or Python cannot keep up, do not let producers allocate forever.

Design:

- each worker owns a fixed local chunk buffer;
- flushed chunks go into a bounded MPSC queue;
- one writer thread drains the queue;
- producers block or yield when the queue is full;
- stats record stall time.

This avoids the classic failure mode: “parallel tracing is fast enough to generate 80 GB of output and crash before the disk writer catches up.”

### Introduce spill-to-disk for field grids and snapshots

Regular field grids currently allocate `n_bands * n_cells_total * complex64`. That is 8 bytes per complex cell, so a `512^3` grid with 32 bands is roughly 34 GB before temporaries. A copy buffer doubles per-band working memory and snapshots can multiply it again.

Add backing modes:

```cpp
enum FieldBacking {
    FIELD_BACKING_RAM,
    FIELD_BACKING_MMAP,
    FIELD_BACKING_TILED_CACHE
};
```

For large grids:

- store field data in a memory-mapped file;
- operate on slabs/tiles with halo cells;
- keep only a bounded tile cache in RAM;
- flush dirty tiles after each step or after a configurable interval;
- add metadata sidecar describing dimensions, band count, complex layout, endianness, and version.

### Tile the field marcher

The current stencil loop copies an entire band to `tmp`. Replace this with one of:

1. Ping-pong buffers:
   - maintain `src` and `dst` full-grid buffers;
   - no per-band memcpy;
   - swap pointers per step.
   - Still RAM-heavy, but faster and cleaner.

2. Slab streaming:
   - process `z` slabs with halo rows;
   - keep three or more planes in memory;
   - write output slab sequentially;
   - works with mmap or file-backed storage.

3. 3-D blocked tiles:
   - process blocks like `Bx x By x Bz`;
   - include 1-cell halo for 7-point stencil;
   - schedule blocks through thread pool;
   - ideal for cache locality and future GPU transfer.

The first production pass should likely implement ping-pong full-grid buffers for immediate speedup, then slab/tile streaming for large dataset mode.

### Thread-safe field injection

Amplitude injection currently directly mutates field storage. That is unsafe if called from parallel ray workers.

Options:

1. Thread-local sparse deposit buffers:
   - each worker accumulates `(band, cell, re, im)` deposits locally;
   - periodically sort/reduce by `(band, cell)`;
   - merge into global field under coarse locks or in a single reducer.

2. Tile-local locks:
   - partition grid into tiles;
   - each tile has a mutex/spinlock;
   - deposit locks only touched tiles.
   - Simpler but can thrash with high contention.

3. Atomic float accumulation:
   - not portable for complex float unless using `std::atomic_ref<float>` and accepting performance/ordering constraints;
   - GPU path may use CAS loop, but CPU should avoid per-deposit atomics where possible.

Recommended initial path: thread-local deposit buffers with chunked reduction. It is stable, deterministic enough if sorted, and preserves throughput.

### Add resumability

For long traces or field marches, add checkpointing:

- save config hash,
- save RNG seed/state strategy,
- save current source/ray/bounce/tile position,
- save output record offset,
- save field backing file metadata,
- save version number and ABI sizes.

Long-running jobs should be restartable after crash. This matters more than squeezing the final 5% of speed.

### Add cancellation and progress

Long kernels should accept an atomic cancellation token and progress callback/counter.

Do not wait until a 6-hour trace finishes to discover it is in the wrong mode.

Expose stats:

```cpp
struct RtTraceStats {
    uint64_t rays_started;
    uint64_t rays_completed;
    uint64_t bounces_processed;
    uint64_t records_emitted;
    uint64_t records_dropped;
    uint64_t bytes_written;
    uint64_t field_deposits;
    uint64_t sink_stall_ns;
    int truncated;
    int cancelled;
};
```

### Protect Python from raw massive buffers

Raw `float* field_grid_data_re_im(FieldGrid*)` is useful but dangerous. Keep it for advanced use, but add safe windowed access:

```cpp
extern "C" SK_API int field_grid_read_tile(
    FieldGrid* g,
    int band0, int band_count,
    int x0, int y0, int z0,
    int nx, int ny, int nz,
    float* out_re_im,
    size_t out_len);
```

Similarly provide `write_tile`, `flush`, and `prefetch`. Python should not need to map or copy the entire grid for inspection.

### Fail loudly on truncation in non-preview modes

Visualization paths may drop records when a buffer fills. Scientific generation paths must not silently drop.

Implement explicit policy:

```cpp
enum OverflowPolicy {
    OVERFLOW_DROP_PREVIEW,
    OVERFLOW_STOP_WITH_ERROR,
    OVERFLOW_STREAM_TO_DISK,
    OVERFLOW_COUNT_ONLY
};
```

Default scientific/data mode should be `STOP_WITH_ERROR` or `STREAM_TO_DISK`, never silent drop.

---

## Threading and work stealing

### Do not parallelize by blindly sharing mutable state

Ray tracing is embarrassingly parallel per ray until it writes output, deposits into field grids, or appends camera strikes. The safe architecture is:

- immutable shared scene state;
- immutable material tables;
- immutable BVH;
- thread-local RNG;
- thread-local amplitude vectors;
- thread-local output chunks;
- thread-local field-deposit chunks;
- explicit merge/flush stage.

Do not let multiple workers mutate `count`, `camera_strike_rows`, `FieldGrid::data`, or shared RNG.

### Use work stealing for irregular ray workloads

Ray workloads are irregular because rays terminate at different bounces, hit different BVH regions, enter scale contexts, or scatter. Static partitioning by source/ray can load-imbalance. Use a work-stealing pool or a dynamic task queue.

Recommended granularity:

- task = block of rays, not single ray;
- block size initially 128–4096 rays depending on per-ray cost;
- adaptive splitting if contexts make blocks too slow;
- each worker drains local deque and steals from others.

If using an existing `ThreadPool`, inspect whether it has work stealing. If not, either add it or use a known implementation pattern:

- per-worker double-ended queues;
- owner pushes/pops bottom;
- thieves steal top;
- global injector queue for initial tasks;
- no unbounded task allocation.

### Use deterministic per-ray RNG

A single `std::mt19937_64 rng` passed through trace loops blocks parallelization and makes parallel order affect results.

Replace with per-ray deterministic RNG:

```cpp
uint64_t ray_seed = hash64(global_seed, source_id, ray_id);
std::mt19937_64 rng(ray_seed);
```

Or use a faster stateless counter-based RNG if reproducibility and speed matter. This enables parallel trace order without changing stochastic distribution.

### Avoid OpenMP mixed with custom thread pools unless explicitly controlled

If the code already has some OpenMP sections and also uses a custom thread pool, avoid nested oversubscription. Choose one scheduling layer per kernel. If using Eigen internally, control Eigen’s thread count:

```cpp
Eigen::setNbThreads(1);
```

inside manually parallelized kernels, because many small Eigen solves should not spawn their own nested parallel work.

---

## Eigen-specific optimization directives

### Surface spline: replace dynamic Eigen with fixed-size accumulation

The fit solves for 6 coefficients. Do not allocate `MatrixXd A`, `VectorXd b`, and `VectorXd coeffs` per triangle unless in reference mode.

Instead accumulate normal equations directly:

```cpp
Eigen::Matrix<double, 6, 6> ATA;
Eigen::Matrix<double, 6, 1> ATb;
ATA.setZero();
ATb.setZero();

auto add_row = [&](const Eigen::Matrix<double, 1, 6>& r, double y) {
    ATA.noalias() += r.transpose() * r;
    ATb.noalias() += r.transpose() * y;
};
```

Then solve:

```cpp
Eigen::Matrix<double, 6, 1> coeffs =
    ATA.ldlt().solve(ATb);
```

If rank-deficient and no ridge is provided, either:

- add a small fallback ridge, or
- use a fixed-size ColPivHouseholderQR path, not dynamic matrices.

### Surface spline: remove `unordered_set` from hot loop

`one_ring()` currently builds an `unordered_set<int>` per triangle. That is allocation-heavy and unpredictable.

Replace with one of:

1. Precomputed triangle adjacency:
   - build `tri_neighbors` once;
   - store compact CSR arrays:
     - `tri_neighbor_offsets[n_tris + 1]`
     - `tri_neighbor_indices[...]`

2. Small vector with duplicate suppression:
   - for each triangle, gather neighbor IDs from 3 vertex lists into a small buffer;
   - sort and unique;
   - avoid hash table allocations.

CSR is preferred for large meshes and repeated fitting.

### Surface spline: precompute triangle geometry

Avoid recomputing `v0`, `edge1`, `edge2`, normal, centroid, area repeatedly in `fit_one_triangle`.

Create:

```cpp
struct TriGeom {
    Eigen::Vector3d v0;
    Eigen::Vector3d edge1;
    Eigen::Vector3d edge2;
    Eigen::Vector3d normal;
    Eigen::Vector3d centroid;
    double area2;
    uint8_t valid;
};
```

Build once. Fit reads from `TriGeom`.

### Ray tracer: do not overuse Eigen dynamic vectors in per-ray loops

`Eigen::VectorXcd amp(n_bands)` inside hot loops is convenient but can allocate. If `n_bands <= MAX_SPECTRAL_BANDS`, use fixed-capacity stack storage:

```cpp
std::array<std::complex<double>, MAX_SPECTRAL_BANDS> amp;
```

or an aligned small vector. Eigen dynamic `VectorXcd` is acceptable for setup or large band counts, but per-ray allocation must disappear.

### Ray tracer: vectorize per-band propagation where it helps

Per-band propagation is a good target for Eigen array expressions if `n_bands` is large enough. But if `MAX_SPECTRAL_BANDS` is small (e.g., 32), a hand-written loop over contiguous arrays may outperform dynamic Eigen.

Store spectral data in SoA form:

```cpp
double amp_re[MAX_BANDS];
double amp_im[MAX_BANDS];
```

This allows SIMD-friendly phase rotation:

```cpp
new_re = re * c - im * s;
new_im = re * s + im * c;
```

Do not compute `std::abs` and `std::arg` unless the output sink actually requires magnitude/phase. Keep real/imag internally and convert at boundaries.

### Field marcher: Eigen is not the main answer

Do not try to fix the 3-D stencil primarily with Eigen high-level matrices. The field marcher is a memory-layout, tiling, and threading problem. Use raw contiguous loops, restrict-like assumptions where safe, block tiling, and explicit halos. Eigen may help for small fixed transforms or vector packets, but the central improvement is data movement control.

---

## Ray tracer optimization plan

### Phase 1: make output safe and thread-local

Implement a trace worker that owns:

```cpp
struct RayWorkerLocal {
    std::vector<float> segment_chunk;
    std::vector<EndpointRecord> endpoint_chunk;
    std::vector<FieldDeposit> field_deposits;
    std::array<std::complex<double>, MAX_SPECTRAL_BANDS> amp;
    RtTraceStats stats;
};
```

The worker writes locally. It flushes chunks to a sink. No shared `count++`.

### Phase 2: deterministic parallel trace

Create a task queue over `(source range, ray range)`. Each task traces a block of rays. Each ray seed derives from `(global_seed, source_id, ray_id)`. Diffuse decisions become independent of scheduling.

The public single-threaded trace remains available for regression.

### Phase 3: split preview from full data

Currently segment records appear designed for visualization. Add separate APIs:

- `trace_preview_segments`: bounded buffer, may drop on overflow, fast.
- `trace_dataset_segments`: streaming sink, never silent drop.
- `trace_field_accumulation`: emits field deposits/reductions.
- `trace_sensor_capture`: camera/sensor strikes, chunked.

Do not overload one buffer path with all semantics.

### Phase 4: material lookup cache

Material reflectance is read per hit per band. Cache per-material/per-band derived complex reflectance and diffusion values in a compact structure during state setup:

```cpp
struct MaterialSpectralCache {
    float refl_re[MAX_BANDS];
    float refl_im[MAX_BANDS];
    float diffuse[MAX_BANDS];
    float transmittance[MAX_BANDS];
    float emission[MAX_BANDS];
    float ior_re[MAX_BANDS];
    float ior_im[MAX_BANDS];
};
```

Avoid recomputing Fresnel phase or fetching strided `mat_buf` values in the hot loop unless material edits happen live.

### Phase 5: BVH quality and traversal

The existing BVH is good enough for a baseline, but performance can improve:

- Increase leaf size experimentally; 4 may be too small.
- Use SAH or binned SAH builder for large scenes.
- Store BVH nodes in cache-friendly arrays.
- Traverse near child first based on ray direction to improve early `t_min`.
- Make stack size safe or assert; fixed stack 64 may fail for degenerate deep trees.
- Precompute triangle data in compact arrays:
  - `v0x/v0y/v0z`
  - `e1x/e1y/e1z`
  - `e2x/e2y/e2z`
  - `normal`
  - `mat_idx`
  - `flags`
- Consider packet traversal only after scalar path is clean.

### Phase 6: camera strike and field accumulation hardening

`camera_strike_rows` should become chunked. `camera_capture_max_strikes` should be enforced as a policy with stats. Field accumulation should go through thread-local deposits and reduction.

### Phase 7: scale contexts

Scale-context dispatch is a future extension point. Do not bury it. Add timing and counts:

- context entries
- context kind histogram
- average time per context kind
- number of passthrough stubs
- number of rays transformed

This will prevent future “mystery slowdowns” when wave/neural contexts become active.

---

## Field marcher optimization plan

### Phase 1: allocation safety

Before changing math, add:

- checked dimension multiplication;
- memory budget checks;
- explicit error code for allocation too large;
- byte count reporting;
- no signed overflow;
- no `calloc(size_t(n))` after unchecked `int64_t` calculation.

### Phase 2: ping-pong stepping

Replace copy-per-band-per-step with source/destination buffers.

Current conceptual pattern:

```cpp
copy base -> tmp
for each cell:
    base[i] = f(tmp)
```

Better RAM mode:

```cpp
src = data_a
dst = data_b
for step:
    for band:
        update dst from src
    swap(src, dst)
```

If API requires `g->data` to hold current data, either maintain `g->front` pointer or swap at the end.

### Phase 3: tiled threaded stencil

Tile over `z/y/x` blocks. Each task owns a tile. The source buffer is read-only for the step. The destination buffer is write-only for the same tile. That makes threading safe.

Avoid tiny tasks. Good starting blocks:

- `z` slab chunks of 4–16 planes for simple implementation;
- or 3-D blocks like `64x8x8`, `32x16x8`, tuned by cache.

### Phase 4: streaming large field mode

For grids that exceed RAM budget:

- process one band or band block at a time;
- process z-slabs with halos;
- source and destination are mmap-backed;
- only slab buffers are resident;
- flush slabs after completion.

Pseudo-structure:

```cpp
for step:
  for band_block:
    for z_chunk:
      read source halo slab [z0-1, z1+1]
      compute interior [z0, z1]
      write destination slab
  swap backing files
```

This prevents crashes on huge grids and permits jobs larger than RAM.

### Phase 5: k-d tree leaf stepping

The k-d tree grid is currently storage and injection. If marching across k-d leaves becomes necessary, define boundary exchange explicitly:

- each leaf has interior cells;
- each leaf needs ghost/halo cells from neighbor leaves;
- build neighbor relationships once;
- update leaf interiors in parallel;
- exchange halos between steps.

Do not fake k-d leaf marching without boundary policy. That will create unphysical discontinuities.

### Phase 6: field injection reductions

Add `FieldDeposit`:

```cpp
struct FieldDeposit {
    int band;
    int64_t cell;
    float re;
    float im;
};
```

Per worker, accumulate deposits. At flush time:

- sort by `(band, cell)`;
- reduce duplicates;
- apply to global field in large sequential runs;
- or route deposits to tile owners.

This avoids data races and can be streamed to disk too.

---

## Surface spline optimization plan

### Phase 1: precompute geometry and adjacency

Build once:

- `TriGeom[n_tris]`
- `vtx_to_tris` CSR
- `tri_neighbors` CSR

Replace `std::vector<std::vector<int>>` where memory overhead matters. `vector<vector<int>>` is acceptable for initial development but poor for huge meshes due to many small heap allocations.

### Phase 2: eliminate per-triangle heap allocations

Replace:

- `std::vector<Eigen::Matrix<double,1,6>> rows`
- `std::vector<double> rhs`
- `Eigen::MatrixXd A`
- `Eigen::VectorXd b`
- `unordered_set`

with:

- direct fixed-size normal equation accumulation;
- small fixed arrays for neighbor candidates;
- precomputed adjacency.

### Phase 3: thread pool reuse

Do not construct/destroy `ThreadPool` every call if this function is called repeatedly. Accept a pool from outside or use a shared pool owned by the module/runtime.

Potential API:

```cpp
surface_spline_fit_with_workspace(..., SurfaceSplineWorkspace* ws)
```

Workspace contains:

- pool reference,
- adjacency buffers,
- geometry buffers,
- per-thread temporary buffers,
- stats.

### Phase 4: work stealing or dynamic chunking

Triangle fit cost varies by valence. Use dynamic scheduling. If the thread pool only supports static `parallel_for`, modify it to chunk work dynamically or use work stealing. Target chunk size:

- enough triangles to amortize scheduling overhead;
- small enough to balance high-valence regions;
- initial chunk size: 64–512 triangles.

### Phase 5: numerical safety

If `ATA.ldlt()` fails or produces non-finite coefficients:

- retry with small ridge;
- count failure;
- zero coefficients only if fallback fails;
- report stats.

Add finite checks on input vertices, normals, and coefficients. Degenerate triangles should not poison the output.

---

## API and ABI hardening

### Add versioned structs

The code has Python ctypes / C ABI layout sensitivity. Use versioned config structs:

```cpp
struct RtTraceConfig {
    uint32_t struct_size;
    uint32_t version;
    int n_sources;
    int n_rays;
    int max_bounces;
    double min_amplitude;
    uint32_t seed;
    int overflow_policy;
};
```

At entry:

```cpp
if (!cfg || cfg->struct_size < expected_min) return SK_ERR_BAD_CONFIG;
```

This prevents silent ABI drift.

### Return useful errors

Do not return `SK_ERR_NULL_STATE` for every failure. Add:

- `SK_ERR_BAD_DIMS`
- `SK_ERR_OVERFLOW`
- `SK_ERR_ALLOCATION_TOO_LARGE`
- `SK_ERR_OUT_OF_MEMORY`
- `SK_ERR_TRUNCATED`
- `SK_ERR_CANCELLED`
- `SK_ERR_SINK_FAILED`
- `SK_ERR_NUMERICAL_FAILURE`

Expose a way to query last error details:

```cpp
extern "C" SK_API const char* sk_last_error_message();
```

or pass an error buffer into calls.

### Add stats outputs

Every expensive API should optionally fill a stats struct. Stats are not optional for development quality; they are the only way to know what changed.

---

## Memory layout recommendations

### Use SoA for hot numeric loops

Ray triangles, material spectral cache, and field deposits should be laid out for contiguous access.

For triangles, current AoS with Eigen vectors is readable. For hot traversal, create a compact acceleration copy:

```cpp
struct TriangleAccel {
    double v0[3];
    double e1[3];
    double e2[3];
    double n[3];
    int mat_idx;
    int flags;
};
```

Or SoA arrays for wider SIMD later.

### Avoid `std::complex` where real/imag SIMD matters

`std::complex<T>` is convenient but not always ideal for vectorized kernels or atomic/reduction behavior. For hot storage, consider explicit real/imag arrays:

```cpp
struct ComplexSoA {
    float* re;
    float* im;
};
```

For ABI parity with GPU and Python, expose real/imag layout directly. Keep `std::complex` in reference paths if desired.

### Align buffers

Use aligned allocators for Eigen/vectorized arrays. At minimum:

- align field buffers to 64 bytes;
- align triangle accel arrays;
- align per-thread chunks.

---

## Testing requirements

### Correctness tests

Add tests comparing:

- single-thread reference trace vs parallel trace with deterministic per-ray seeds;
- old segment buffer output vs new memory sink output for small scenes;
- field marcher reference copy mode vs ping-pong mode for small grids;
- surface spline old dynamic solve vs fixed normal-equation solve on known meshes;
- k-d injection vs regular injection where geometry overlaps.

### Stability tests

Add tests that intentionally request impossible sizes:

- huge grid dimensions that overflow `size_t`;
- huge band count;
- huge output record estimate;
- negative dimensions;
- too-small output buffers;
- null pointers;
- degenerate triangles;
- NaNs/Infs in vertices.

Expected result: graceful error, not crash, not undefined behavior.

### Streaming tests

Add tests for:

- sink flush at small chunk sizes;
- forced sink failure after N chunks;
- cancellation mid-run;
- resumable checkpoint metadata;
- output byte count equals record count * stride;
- no silent truncation in scientific mode.

### Performance regression tests

Track:

- allocations per ray;
- allocations per field step;
- allocations per triangle fit;
- throughput baseline;
- max resident memory.

Failure condition: a patch that improves one path but creates massive hidden allocation elsewhere should fail CI or at least flag.

---

## Suggested implementation order

1. Add stats structs and checked size helpers.
2. Add benchmark harnesses.
3. Add output sink abstraction and chunked buffers.
4. Convert ray tracer to deterministic per-ray RNG and thread-local chunks.
5. Add parallel ray task scheduling with work stealing/dynamic chunks.
6. Add field grid allocation budgets and safer APIs.
7. Convert field marcher to ping-pong buffers.
8. Add tiled threaded field marcher.
9. Add streaming/mmap field backing.
10. Precompute surface spline geometry and adjacency.
11. Replace dynamic Eigen fits with fixed-size normal-equation solves.
12. Reuse thread pools/workspaces.
13. Add large-data crash tests and streaming tests.
14. Tune BVH leaf size and material spectral cache.
15. Only after all this: consider SIMD packets, SAH BVH, GPU compute, or neural/wave context acceleration.

---

## Specific anti-patterns to remove

- Per-ray or per-triangle dynamic allocation.
- Shared global RNG in parallel code.
- Shared output counter mutation from workers.
- Direct concurrent writes into field grid.
- Giant `std::vector` append buffers for unbounded datasets.
- Silent overflow in scientific/data-generation paths.
- Unchecked dimension multiplication.
- Reallocating thread pools per frequent call.
- Treating Python as a safe owner of enormous raw buffers.
- Nested parallelism from Eigen/OpenMP/custom pool.
- Recomputing material spectral derived values per hit.
- Recomputing triangle geometry per local fit.
- Using hash sets in per-triangle inner loops.
- Copying full field bands every step when ping-pong or tiled halos can avoid it.

---

## Definition of done

This optimization pass is not done when the code merely runs faster on one toy scene. It is done when:

- small-scene outputs match the reference path within expected floating-point tolerance;
- parallel ray tracing is deterministic for the same seed independent of thread count;
- large allocations fail gracefully or switch to streaming mode;
- unbounded output has a streaming path;
- field marching can run in bounded memory for grids larger than RAM if out-of-core mode is enabled;
- field injection is thread-safe;
- surface spline fitting has no avoidable per-triangle heap allocation in production mode;
- benchmarks report throughput and memory;
- stats report truncation, cancellation, and sink failure;
- Python can read/write field tiles without copying entire grids;
- all new APIs preserve C ABI discipline through versioned structs.

---

## Implementation warning

The code contains real design intent: spectral complex transport, material-buffer parity, scale contexts, sensor/film tensor ingress, k-d field chunks, and per-triangle spline surfaces are not accidental clutter. Optimize by isolating costs, bounding memory, and improving scheduling. Do not “simplify” the system by deleting the hard parts. The purpose is to make the hard parts survivable at scale.
