# Agent-Facing Optimization Presentation



## Operational directive

_What the next agent must optimize for before touching code_

### Non-negotiables
- Preserve spectral complex amplitude semantics: phase, attenuation, reflectance, Stokes/re-emission, direct/indirect surface accumulation, and camera/sensor side effects cannot be collapsed to scalar intensity unless a separate mode says so.
- Treat all Python/ctypes/SSBO binary layouts as ABI contracts. Static assertions already guard SensorRecord/FilmRecord; extend that discipline to every exported C struct and packed output row.
- Every parallelization change must be deterministic enough to debug: fixed seed + fixed scheduling mode should reproduce exact ray histories where required; high-throughput nondeterministic mode must be explicitly named.
- Do not hide data-loss optimizations behind performance flags. If a path reduces bands, abs() collapses complex values, quantizes, or merges endpoint records, it must be opt-in and documented.
- Prefer changing internal storage and dispatch behind the C API boundary; minimize API churn unless a layout flaw blocks major wins.

### Immediate working posture
- Start with measurement and memory-layout inspection; avoid guessing the hot path from code size alone.
- Add isolated microbenchmarks for: BVH traversal, per-band propagation, material lookup, field_grid_step_helmholtz_regular, trilinear injection, surface_spline_fit.
- Enable compiler warnings and sanitizers on small scenes before adding threads: ASan/UBSan/TSan builds, then release -O3/-march=native builds.
- Introduce one shared thread-pool abstraction first; do not let each subsystem spawn its own pool or Eigen silently nested-parallelize under custom tasks.
- Make output accumulation strategy explicit: unique slot writes, per-thread buffers, atomic accumulation, or final reduction.


## Current-state map

_Files and dominant behavior inferred from the uploaded implementation_

### ray_tracer(13).cpp
- Large all-in-one spectral geometric/wave/camera integration unit: BVH, ray loops, material mat_buf, scale contexts, persistent ray scheduler, endpoint/sensor/image reductions, coherent projection and bidirectional packed entry points.
- Uses Eigen::Vector3d, Eigen::VectorXd, Eigen::VectorXcd, std::complex<double>, std::mt19937_64, std::vector-heavy state, and several callback-template trace loops.
- BVH build is median split over triangle AABBs; traversal is iterative stack[64]; fallback O(n_tri) path remains present and should be treated as debug only for nontrivial geometry.
- Several exported paths duplicate ray traversal ideas: trace_rays, trace_rays_v2, trace_rays_multiscale, stateful ray_tracer_spawn/step, bidirectional helpers.

### field_march + surface_spline
- field_march.cpp stores complex64 bands over either regular full grid or explicit kd-tree leaves; comments call current stencil implementation reference-quality/slow, with SIMD/threaded port deferred.
- field_grid_step_helmholtz_regular copies one full band into tmp every step, then triple loops z/y/x and writes back into base.
- field_grid_inject_amplitude performs trilinear scatter into regular grid or kd leaf; current += writes are unsafe under naive parallel injection.
- surface_spline.cpp already uses ThreadPool::parallel_for, but each triangle builds unordered_set one-ring, dynamic vectors, MatrixXd/VectorXd systems, and dynamic QR/LDLT.


## Optimization threat model

_What is likely slow, unsafe under threads, or structurally fragile_

### Likely hot costs
- Ray tracing: per-ray × per-bounce BVH traversal; per-hit per-band exp/cos/sin/abs/material access; callback write/accumulate paths; random scattering; repeated creation of temporary Eigen dynamic vectors.
- Field marching: memory bandwidth dominates; full-band memcpy per step; complex arithmetic through std::complex may block vectorization; serial z/y/x loops underutilize cores; boundary handling is implicit and unmeasured.
- Spline fitting: small fixed problem repeatedly solved with dynamic-size Eigen; unordered_set allocation and hashing in one_ring per triangle; repeated centroid/normal recomputation across adjacent triangles.
- Sensor/image reductions: many output pixels/records may receive many contributions; naïve shared += cannot be parallelized without atomics or per-thread reduction.

### Failure modes to prevent
- False sharing from many threads accumulating adjacent floats in out_direct/out_indirect/out_image/field grid.
- Nondeterministic RNG streams caused by work stealing consuming a shared generator in different orders.
- Oversubscription from Eigen internal threads + custom ThreadPool + Python caller threads.
- ABI drift from changing std::complex storage assumptions while Python/GLSL expects re/im paired float rows.
- Performance regression masked by visual correctness because the segment output silently drops when cap is hit in some paths.


## Measurement before modification

_Build hard evidence so the next agent can safely change a fragile experimental codebase_

### Instrumentation to add
- Compile-time feature flags: RT_PROFILE, RT_DETERMINISTIC_PARALLEL, RT_VALIDATE_LAYOUTS, RT_ENABLE_THREAD_POOL, RT_USE_SOA_BANDS.
- Counters inside RayTracerState or separate Profiler: rays_spawned, rays_skipped_directivity, bvh_node_tests, tri_tests, hits, misses, bounces, material_reads, band_ops, field_injections, sensor_writes.
- Timer scopes per exported entry: trace, surface trace, IR integration, image integration, multiscale trace, spawn/step, bidirectional endpoint generation, reductions, field march, spline fit.
- Memory counters: max segment rows requested/written/dropped, per-thread buffer capacity, field grid bytes, temporary bytes allocated per call, allocations inside hot loops if custom allocator hook is available.

### Benchmarks to create
- BVH microbench: fixed synthetic triangle scene + fixed rays; measure traversal only and traversal+hit payload separately.
- Band propagation microbench: n_bands sweep {4, 8, 16, 32, 64}; compare std::complex loop, Eigen::Array vectorization, sincos table/recurrence variants.
- Field march microbench: grid sizes {64³, 128³, 256³}, bands {1, 8, 32}, steps {1, 10}; report GB/s and cell-updates/s.
- Spline fit microbench: manifold meshes with varied valence; measure adjacency build, one-ring gather, solve, and total wall time.
- Regression scenes: single mirror, Lambertian box, transmissive slab, reactive material shift, camera visibility full march, kd-tree deposit boundary cases.


## Build and compiler layer

_Cheap global improvements before algorithm surgery_

### Compiler configuration
- Release profile: -O3 -DNDEBUG -march=native or target-specific AVX2/AVX512 flags where deployment allows; keep a portable baseline build too.
- Enable LTO/IPO for release artifacts; use PGO after microbenchmarks stabilize because branch-heavy BVH/ray code can benefit materially.
- Use -ffast-math only in an explicit approximate build. Default physics build should avoid silently changing NaN, signed zero, associativity, and complex behavior.
- Eigen flags: define EIGEN_NO_DEBUG for release; pin EIGEN_MAX_ALIGN_BYTES consistently across shared library boundaries; check EIGEN_DONT_PARALLELIZE when custom pool owns threading.
- Warnings: -Wall -Wextra -Wconversion -Wshadow -Wpedantic selectively; resolve narrowing and signed/unsigned conversions in exported interfaces.

### Runtime configuration
- Expose thread count and deterministic/nondeterministic scheduling as API/config, not compile-only constants.
- Disable Eigen internal threading when calling Eigen kernels inside a work-stealing pool; nested parallelism will damage throughput and reproducibility.
- Use aligned allocators for hot arrays if vectorized Eigen Maps touch them; keep ABI-facing raw float arrays unchanged at the boundary.
- Keep debug assertions for dimension consistency: n_bands <= MAX_SPECTRAL_BANDS, mat_buf row count, output strides, grid dims, kd leaf bounds, tri index validity.
- Add CI matrix: Debug sanitizer, Release scalar deterministic, Release vectorized, Release threaded.


## Thread-pool architecture

_Use one persistent pool; work stealing only at coarse enough granularity_

### Design requirements
- A single process-level or RayTracerState-owned ThreadPool should serve ray tracing, field marching, spline fitting, and reductions. Do not construct/destroy pools inside hot exported calls unless the call is explicitly one-shot.
- Work stealing should operate over chunks, not single rays/cells. Chunks amortize queue contention and preserve data locality; stealing occurs when a worker exhausts its deque.
- Provide APIs: parallel_for_chunks(begin,end,grain,fn), parallel_reduce_chunks(begin,end,grain,init,fn,join), submit_task, wait_group, current_worker_index, thread_local_scratch<T>().
- Deterministic mode: static partitioning or fixed chunk order + per-chunk RNG. Fast mode: work stealing with per-task RNG derived from stable task ID.

### Granularity recommendations
- Ray traversal: chunk by source-ray ranges, e.g. 256–4096 rays per task depending on bounce cost and output payload. Dynamic stealing handles variable bounce counts.
- Field marching: chunk by z-slab or flattened cell blocks per band; prefer slabs large enough to stream contiguous x inner loops and avoid write sharing at slab boundaries.
- Spline fitting: chunk by triangles; grain 128–2048 triangles depending on mesh valence. Each worker owns scratch vectors/matrices.
- Reductions: first produce per-thread accumulators; reduce after traversal. Atomics are acceptable only for sparse, low-collision debug paths.
- Avoid per-task allocation. Task closure should capture pointers/ranges only; scratch storage is worker-local.


## Parallel ray tracing plan

_Primary throughput win: independent source-ray paths with per-thread outputs_

### Work decomposition
- Flatten work into RayTaskRange: source index, ray_begin, ray_end. For n_rays_per_source, prefix-sum ray counts and assign contiguous global ray IDs.
- Each worker owns: RNG stream, Eigen/fixed band scratch, segment buffer, direct/indirect surface accumulators or sparse contribution list, camera strike rows, optional endpoint records.
- The immutable scene state can be shared read-only: triangles, BVH nodes, tri_ids, mat_buf, k_real, atmo_abs, scale_contexts, group metadata.
- Return ThreadTraceResult objects; final serial or parallel reduction merges segment rows and dense arrays.

### Do not share these live
- Do not share std::mt19937_64 by reference across tasks. Current trace_rays signatures pass one rng into nested loops; that blocks safe parallelization and deterministic replay.
- Do not let worker lambdas write directly to out_direct/out_indirect/out_image unless each worker owns disjoint rows or uses atomics/per-thread replicas.
- Do not increment one global segment count from many workers unless using block reservation: worker computes local count then copies, or atomic fetch_add reserves blocks.
- Do not mutate RayTracerState during stateless trace calls except explicit profiler counters behind atomics/TLS.


## RNG and deterministic work stealing

_Parallel speed without destroying reproducibility_

### Required change
- Replace shared sequential RNG consumption with counter-based or task-derived RNG. Use a stable seed hash: rng_seed = hash64(global_seed, source_id, ray_id, bounce, scatter_event_kind).
- Preferred: implement small counter-based generator for ray events (e.g. splitmix64/PCG-style) with deterministic uniform conversion. mt19937_64 is heavy and stateful; one object per task is okay, one per ray event is not.
- For current compatibility, each RayTaskRange can initialize mt19937_64 from hash(seed, task_id) and consume locally; this is faster to implement but changes exact old serial results.
- Make reproducibility policy explicit in API docs: bit-identical serial legacy vs deterministic parallel equivalence vs nondeterministic fastest.

### Agent implementation notes
- Add RayRng struct with next_u01(), next_u64(); keep it independent from std::uniform_real_distribution overhead in inner loops.
- For cosine_hemisphere, accept two uniform values rather than rng reference; this makes scattering pure and easier to vectorize/test.
- For directivity skip and Fresnel roulette, derive event-specific random values; avoid order-dependent RNG where rays with fewer bounces shift all subsequent draws.
- Store rng_state in RtRayState only if stateful stepping requires continuity; seed it deterministically at spawn and update it per step without global generator.


## Output accumulation strategy

_Parallel correctness hinges on owning writes_

### Segment records
- Current write_segment uses count by reference and silently returns if count >= cap; parallel equivalent must avoid data races and preserve predictable truncation policy.
- Option A: two-pass segments. Pass 1 counts segment needs per task; prefix-sum; Pass 2 writes exact ranges. Most correct, extra traversal cost unacceptable for expensive physics unless dry-run is cheap.
- Option B: per-thread vectors then concatenate until cap. Simple and safe; order changes unless task ranges are merged by source/ray order.
- Option C: atomic block reservation: each worker reserves K rows, fills local block, repeats. Fast but output order nondeterministic unless range ID sorting is applied.

### Dense/sparse reductions
- out_direct/out_indirect: dense n_tri*n_bands arrays; use per-thread dense buffers only when memory allows. Otherwise per-thread sparse vectors of (tri, band, value) and sort/reduce.
- out_image/rgb/sensor: per-thread image buffers are often acceptable because image size is smaller than ray count; reduce using vectorized add.
- FieldGrid injection: never naïvely parallelize += into shared g->data. Use thread-local grids or tiled atomic-free staging, then reduce; atomics only for sparse debug or GPU parity path.
- camera_strike_rows: worker-local rows then concatenate; enforce max strike cap at final merge or deterministic pre-reservation.


## Eigen policy

_Use Eigen where it matches shape; avoid dynamic Eigen in inner scalar kernels_

### Good uses
- Use Eigen::Matrix<double,3,1> / Vector3d for geometry operations already present; keep fixed-size vectors in registers and allow inlining.
- Use Eigen::Matrix<double,6,6>, Matrix<double,6,1>, Matrix<double,Dynamic,6> only where dynamic sample count truly matters. Prefer accumulating normal equations directly for the spline fitter.
- Use Eigen::Map<ArrayXcd> or explicit arrays for band vectors when n_bands is runtime but contiguous; wrap raw cd* amplitude spans without allocation.
- For field grids, Eigen::Map<ArrayXcf> can express contiguous row operations, but stencil kernels usually need hand-written loops or packet-level code for cache-aware streaming.

### Bad uses to remove
- Avoid constructing Eigen::VectorXcd amp(n_bands) inside every ray advance when max band count is small. Use small fixed-capacity stack buffer or worker scratch reused across rays.
- Avoid MatrixXd/VectorXd for every triangle in surface_spline_fit; small matrices incur heap allocation and generic solver overhead.
- Avoid Eigen temporaries in per-cell field march if they prevent compiler vectorization; std::complex arithmetic may already be opaque enough.
- Avoid implicit aliasing surprises: use .noalias() for matrix products where retained; never Map unaligned raw ABI memory as aligned unless guaranteed.


## Band-vector representation

_Complex spectral amplitudes are hot; make them contiguous, reusable, and vectorizable_

### Current pattern
- Ray loops use VXcd amp(n_bands), VXcd amp_prop(n_bands), and per-band scalar loops for propagation/reflection.
- Stateful scheduler copies ray_amp_pool[ray_id*n_bands+b] into a fresh VXcd amp each scheduler_advance_ray call, then copies back after every outcome.
- mat_refl_complex recomputes complex Fresnel-derived phase per material/band access; this is pure with respect to mat_buf and can be precomputed.
- apply_reactive_shift uses nested b×j search to find shifted destination band; precompute band_shift_dst[mat][b] or shift table per distinct shift.

### Recommended storage
- If MAX_SPECTRAL_BANDS is bounded and usually <=32: create SmallBandVec<MAX> with std::array<std::complex<double>,MAX> and active n_bands; no heap allocation.
- If n_bands can be large: worker-local std::vector<cd> scratch with reserved size, reused across all rays; use Eigen::Map only over that scratch.
- Precompute MaterialSpectralCache: refl_complex[mat][b], diffusion[mat] or diffusion[mat][b], n_real[mat], reemit_yield[mat], shift_dst[mat][b].
- Consider split re/im SoA arrays for highest throughput: amp_re[b], amp_im[b]. This makes propagation multiply explicit and SIMD-friendly.


## Material lookup cache

_The mat_buf abstraction is correct for parity but expensive inside per-hit loops_

### Problem
- mat_band_record bounds-checks and computes flat offsets every call; mat_refl_complex reads reflectance magnitude and IOR, computes complex division/arg/cos/sin-derived phase for every hit×band.
- mat_diffusion and mat_n_real repeatedly read mat_buf slot values even though material properties are stable for a trace call unless explicitly modified.
- Reactive shift currently scans all bands per source band to find nearest frequency after Stokes shift; that is O(n_bands²) when invoked.

### Implementation
- Build/rebuild RayTracerState::mat_cache whenever mat_buf changes. Cache arrays are indexed [mat_idx*n_bands+b], not MAX_SPECTRAL_BANDS unless alignment wins.
- Cache cd refl, float transmittance, float diffuse, float emission, float reemission, double n_real, double n_imag, int reactive_dst_index.
- Make mat_cache validation cheap: mat_cache_generation increments on mat_buf set; trace asserts cache_generation == mat_generation in debug.
- Keep mat_buf as source of truth for GLSL/Python parity; mat_cache is a CPU acceleration derivative only.


## BVH improvements

_Current BVH is serviceable; next wins are build quality, layout, and traversal order_

### Current state
- Median centroid split over longest axis; leaf max 4; iterative traversal with stack[64]; no near/far child ordering; Triangle stores v0/edge1/edge2/normal/material/flags.
- This is likely correct and better than O(n_tri), but not necessarily high-quality for large/uneven meshes or coherent ray bundles.
- Recursive build can remain initially; traversal is hotter than build unless scenes rebuild frequently.

### Agent tasks
- Flatten BVHNode to cache-friendly fields: bounds as float/double arrays lo[3], hi[3], child indices, tri range; consider 32-byte/64-byte alignment.
- Add near/far ordering by ray direction or computed t_near so closest hits reduce traversal. Current push order does not exploit t_min pruning optimally.
- Guard stack overflow: stack[64] assumes depth; add assert or dynamic fallback for pathological trees.
- Benchmark SAH or binned SAH builder versus median split; keep median as fast-build option.
- Precompute triangle data in SoA if intersection becomes vectorized packet traversal: v0x[], edge1x[], etc.


## Ray loop unification

_Reduce duplicate traversal kernels without flattening semantics_

### Observation
- trace_rays, trace_rays_v2, trace_rays_multiscale, stateful scheduler, and bidirectional loops repeat traversal/directivity/amplitude/surface logic with slightly different callbacks and side effects.
- Duplication increases the chance that material/refraction/reactive changes land in one path but not another.
- A full rewrite is risky; extract shared primitives first.

### Refactor sequence
- 1. Extract RaySpawnIterator: yields source_id, ray_id, src_p, initial_dir, directivity_weight for both uniform n_rays and n_rays_per_source.
- 2. Extract propagate_band_span(amp, distance, path_len, medium/context) with material cache and no segment side effects.
- 3. Extract SurfaceEvent apply_surface_event(ray, hit, material_cache, rng) used by all trace paths.
- 4. Extract OutputPolicy objects: SegmentWriter, SurfaceAccumulator, IRAccumulator, ImageAccumulator, EndpointEmitter.
- 5. Only after behavior tests pass, introduce parallel task ranges around the shared iterator.


## Stateful ray scheduler

_ray_tracer_spawn/step is close to a real scheduler but still serial and allocation-heavy_

### Current constraints
- Persistent ray_pool and ray_amp_pool already support live rays with geo_queue and ctx_queues; this is the right conceptual place for incremental simulation.
- ray_tracer_step snapshots queues by swapping vectors, advances each ray serially, and queues destinations; it creates new_ctx vector-of-vectors each call.
- scheduler_advance_ray allocates VXcd amp(n_bands), copies amp from ray_amp_pool, computes, then copies back. This is a hot per-ray per-step heap/loop pattern.

### Improvements
- Make ray_amp_pool SoA or fixed-stride aligned. Map each ray's band span directly; avoid copying to VXcd unless propagate_ms requires mutable scratch distinct from backing store.
- Use persistent per-context next queues to avoid constructing new_ctx every step; swap current/next buffers.
- Parallelize each queue snapshot by chunks; destination queues are per-worker local vectors, merged after queue processing. No shared push_back during worker loops.
- Use per-ray RNG state or event hash; current per-step RNG seed based on ray_pool size is not physically random and is order-sensitive.
- Consider compacting dead rays only at safe intervals; live_ray_count currently scans queues, not necessarily memory occupancy.


## Multiscale trace

_Context interval handling has algorithmic and allocation opportunities_

### Current behavior
- For each segment, trace_rays_multiscale scans every registered context sphere, computes ray-sphere intervals, sorts intervals by entry, then walks coarse/wave subregions.
- intervals vector is allocated once outside loops and reused, which is good; reserve is max(n_ctx,1). Complexity is O(n_ctx log n_ctx) per hit segment.
- Scale contexts are sorted smallest-radius-first for priority elsewhere, but segment interval processing may overlap contexts without explicit nesting resolution beyond sorted t0.

### Agent tasks
- If n_ctx is small, keep simple scan; if n_ctx grows, build a context BVH/grid or sphere hierarchy to avoid O(n_ctx) per segment.
- Clarify overlap policy: if contexts overlap, does finest radius win, do multiple media compose, or is first interval order enough? Lock this before optimizing.
- Precompute context centers as V3d or raw double[3] cache to avoid constructing V3d ctr in every test.
- Replace sort with insertion into small fixed vector when n_ctx <= 8; for small lists insertion sort beats general std::sort overhead.
- Make propagate_ms side-effect policy explicit: segment writing from inside propagation complicates parallel output reservation.


## Field marcher: first-order performance plan

_The reference kernel is memory-bandwidth bound and serial_

### Current kernel
- field_grid_step_helmholtz_regular allocates tmp of one band, then for each step and band memcpy()s the full cell volume into tmp and updates interior cells in base.
- The loop order x inside y inside z is good for contiguous x streaming, but std::complex<float> arithmetic and base/tmp aliasing may limit vectorization.
- Boundary cells are left unchanged. That may be correct for current behavior, but the boundary policy should be explicit.

### Immediate improvement
- Use double buffering for full grid or per-band buffer pair: src -> dst, swap after each step. This removes per-band per-step memcpy and makes reads/writes non-aliasing.
- Parallelize over bands and z-slabs. If n_bands is high, band-level tasks are easiest; if bands are few, split z slabs within a band.
- Use restrict-like assumptions where possible: cd32* __restrict src, dst or compiler-specific annotations in internal kernels.
- Separate real and imaginary arrays or use two float arrays to let compiler vectorize Laplacian and complex multiply explicitly.
- Report throughput in cell-updates/s and effective memory GB/s; optimize against roofline expectation, not only wall time.


## Field marcher: tiled threaded kernel

_Avoid atomics and cache contention by owning slabs_

### Regular grid stepping
- Partition interior z range [1,nz-1) into slabs. Each task writes dst cells for its slab; reads neighbor z±1 from src, so no write overlap.
- Use one barrier between steps after swapping src/dst. With persistent pool, the outer n_steps loop can enqueue slab tasks and wait per step.
- If n_bands*nz is large, flatten tasks as (band, z0, z1). Grain should contain enough rows to amortize scheduling overhead, e.g. 4–16 z slices per task for large nx*ny.
- Keep x loop innermost; precompute plane_stride=nx*ny; use pointer increments rather than recomputing i every x where possible.

### Trilinear injection
- Regular/kd injection is scatter-add to 8 cells; it is not thread-safe under concurrent deposits into the same grid.
- For ray-to-grid capture, prefer per-thread FieldGrid accumulators for modest grid sizes; for huge grids, use per-thread sparse deposit buffers sorted by cell ID then reduced.
- If using atomics on CPU, split real/imag float arrays and use std::atomic_ref<float> only in a debug or sparse mode; heavy collisions will be slow and nondeterministic.
- GPU parity may keep CAS float atomic emulation, but CPU path should not copy that cost model unless required.


## Kd-tree field grid

_Make AMR leaf traversal and deposit more predictable_

### Current behavior
- field_grid_create_kdtree copies nodes, assigns first_data offsets for leaves, and stores all leaf cells contiguously across bands.
- field_grid_inject_amplitude traverses from root by checking child_lo/child_hi AABBs and choosing lo if both contain the point.
- Leaf deposit repeats trilinear coordinate math and scatter-adds 8 cells.

### Improvements
- Validate kd nodes at creation: child index ranges, leaf_dims > 0, bmin < bmax, no cycles, root exists, first_data overflow checks.
- Precompute per-leaf inv_span[3] and dims_minus_one[3] to remove divisions from every deposit.
- Define overlap behavior explicitly. Current lo-preference when both children contain position is deterministic but may hide overlapping-node modeling errors.
- Build a leaf lookup cache for coherent deposits: last_leaf per worker or spatial hash if many sequential positions are local.
- For threaded deposits, stage by leaf ID so reductions are leaf-local and cache-friendly.


## Surface spline: remove dynamic work

_This file already has parallelism; the hot cost is per-triangle allocation and generic small solves_

### Current behavior
- surface_spline_fit zeroes output, builds vertex→tri adjacency, builds work_list, creates ThreadPool, then parallel_for over triangles.
- fit_one_triangle uses one_ring() which allocates unordered_set and returns vector for every triangle.
- Rows/rhs are dynamic vectors; A is MatrixXd(m,6), b VectorXd(m), ATA MatrixXd; solve uses LDLT or ColPivHouseholderQR.
- The per-triangle system is tiny and bounded by local valence; dynamic Eigen is unnecessary for the common path.

### Better structure
- Precompute triangle geometry once: vertex IDs, v0, edge1, edge2, normal, centroid, area/valid flag. Use this for all fit_one_triangle calls.
- Precompute triangle one-rings once as sorted unique vector<int> or CSR adjacency. Avoid unordered_set in the hot path entirely.
- Use fixed-size normal-equation accumulation: Matrix<double,6,6> ATA; Matrix<double,6,1> ATb. For each sample row r, ATA.noalias() += r.transpose()*r; ATb += r.transpose()*rhs.
- Use LDLT/LLT on 6x6 with ridge always applied minimally to ensure stability, or CompleteOrthogonalDecomposition only for rare debug fallback.


## Spline numerical policy

_Make the fit robust while allowing fast fixed-size kernels_

### Stability questions
- The comment says fallback gracefully when <3 neighbors, but current m>=1 still solves underdetermined systems; zero/self-anchor alone can produce arbitrary coefficients depending on solver path.
- If ridge_lambda == 0, ColPivHouseholderQR solves dynamic A directly; if ridge > 0, LDLT solves ATA. These branches can produce different behavior and stability.
- Normal-matching gradient constraints are soft linearized rows; their relative weight is fixed implicitly. That may produce scale-dependent curvature if mesh edge lengths vary.

### Agent tasks
- Always apply a tiny ridge to curvature terms and optionally all terms except constant anchor. This makes the 6x6 solve consistently SPD/semidefinite-tamed.
- Add sample weights: centroid anchor, neighbor displacement, vertex normal gradient. Make them parameters with defaults, not buried constants.
- Handle underconstrained triangles explicitly: if rank/sample count insufficient, output zero or lower-order fit; do not let arbitrary QR coefficients define geometry.
- Normalize basis or scale u/v neighborhoods to reduce conditioning issues for large out-of-range barycentric samples.
- Unit tests: planar mesh returns zero coefficients; sphere patch yields sign-consistent curvature; degenerate triangles return safe normal.


## Memory layout and ownership

_The fastest code path will be built on boring contiguous arrays_

### Ray tracer state
- Triangle AoS is readable but may be suboptimal for packet traversal. Do not convert immediately; first add a TriangleSoA acceleration cache derived from existing tris.
- BVH nodes, triangle IDs, tri areas, material cache, context cache should be immutable during trace. Enforce const-correctness internally.
- Group metadata uses vector<vector<...>>. Fine for setup, less ideal for hot traversal. Build flattened CSR-like group arrays where repeated lookup occurs.
- camera_strike_rows and sensor_film buffers are owned vectors; worker-local staging avoids shared vector growth.

### Field/spline state
- FieldGrid data is cd32* allocated by calloc. For SIMD, consider aligned_alloc and explicit row-major re/im split cache while preserving data_re_im API view.
- KdNode vector is setup-owned; add derived leaf cache: first_data, dims, inv_span, bmin, bmax, node validity.
- Spline should avoid storing Eigen objects in public ABI, but internal GeometryCache can hold Vector3d arrays or raw double arrays.
- Reserve aggressively: ray_pool reserve n_sources*n_rays after directivity estimate if possible; ray_amp_pool reserve rays*n_bands; queues reserve expected live rays.


## False sharing and reductions

_Threading performance dies if every worker writes neighboring floats_

### Places at risk
- out_direct/out_indirect: adjacent triangles and bands are contiguous; many rays can hit the same hot surfaces.
- out_image: many rays can project into same pixel; atomics create contention and nondeterministic floating order.
- FieldGrid deposits: 8-cell trilinear scatter is a collision magnet around bright caustics/sensors.
- Queue vectors: pushing to shared geo_queue/ctx_queues from parallel workers would lock or corrupt without per-worker staging.

### Policy
- Use per-worker dense buffers when size is tolerable; align each buffer start to cache line.
- Use sparse contribution buffers when dense buffer memory exceeds threshold. Store packed key=(index) and value; sort by key; reduce deterministically.
- For segment records, each worker writes its own vector<SegmentRecord> or raw float vector; final merge has no arithmetic, only copy/truncate.
- Pad per-worker counters and queues to cache lines if they are updated frequently.


## API and ABI preservation

_Make internals faster without breaking Python/GLSL callers_

### C API discipline
- Keep exported functions and struct sizes stable unless updating headers, ctypes declarations, and GLSL packing in the same patch.
- Current static_asserts on SensorRecord and FilmRecord are the right pattern; extend with runtime layout hash checks for mat_buf, endpoint records, strike rows, segment rows.
- Do not expose std::vector/Eigen/std::complex types across C ABI. Keep raw pointers + counts + strides.
- Return explicit errors for invalid dimensions/capacity; avoid silent SK_OK no-ops when user supplied impossible state unless existing API requires it.

### Versioning
- Add ray_tracer_get_build_info(): compile flags, SIMD mode, thread mode, deterministic mode, sizeof structs, MAX_SPECTRAL_BANDS.
- Add feature flags for new paths: legacy serial, threaded deterministic, threaded fast, vectorized field, cached material.
- Use golden-output tests to compare legacy serial vs new serial before enabling parallel paths.
- If output order changes, provide a canonical sort key or a comparison function tolerant to row permutation where physics permits.


## Correctness tests required before speed

_These are guardrails for an agent modifying physics-like code_

### Ray tests
- Single triangle mirror: one ray, one bounce, known reflection direction, phase propagation over known distance, reflectance cache matches mat_buf direct computation.
- Diffuse scatter deterministic: event-hash RNG produces stable hemisphere samples; samples lie on correct side of oriented normal.
- Transmissive slab: Snell path transitions entering/exiting; total internal reflection case reflects; Fresnel roulette probability can be statistically tested.
- Reactive shift: known frequency grid + shift maps energy to expected destination band; low-frequency out-of-range energy handling matches current semantics.
- BVH vs brute force: random rays compare nearest hit triangle and distance on small scenes.

### Grid/spline tests
- Field step: compare legacy memcpy/in-place output to double-buffered output for one step and multiple steps under exact boundary policy.
- Field deposit: regular and kd leaf deposits conserve complex amplitude weights summing to 1 for interior points; boundary points deterministic.
- Surface spline: planar mesh gives zero displacement and original normals; degenerate triangles produce safe fallback; subset mode leaves non-subset zeroed.
- Threaded reductions: threaded deterministic equals serial within defined floating tolerance; reruns with same seed match.


## Implementation sequence

_Do not try to optimize every subsystem in one patch_

### Phase 0: guardrails
- Add benchmark harness, profiler counters, build flags, sanitizer CI, and golden scenes. No major algorithm changes.
- Add material cache but keep serial execution; compare exact outputs where possible.
- Add worker-local scratch infrastructure without enabling parallelism.

### Phase 1–3 changes
- Phase 1: surface_spline fixed-size solve + precomputed adjacency/geometry. Low risk, already parallel, easy speed win.
- Phase 2: field_march double buffer + z-slab parallel kernel. Medium risk; tests can compare directly with legacy output.
- Phase 3: stateless ray trace parallelism with per-worker outputs/reductions. Higher risk; requires RNG and accumulation policy.
- Phase 4: stateful scheduler parallel queues and material/BVH caches. Highest risk; only after stateless trace proves infrastructure.
- Phase 5: deeper acceleration: SAH BVH, packet rays, SoA triangles, GPU field kernels, context acceleration structure.


## Patch blueprint: thread pool interface

_What another agent should implement or adapt_

### Required API
- ThreadPool should be persistent and reusable; construction cost must not happen inside every hot call unless no state object exists.
- Expose worker index for thread-local buffers; expose thread count actually used; handle n_threads<=0 as hardware_concurrency or global config.
- parallel_for_chunks(begin,end,grain,fn) should call fn(chunk_begin, chunk_end, worker_index).
- parallel_reduce_chunks should return deterministic join order in deterministic mode.

### Pseudo-interface
- class ThreadPool { size_t size() const; static ThreadPool& global(); };
- parallel_for_chunks(pool, begin, end, grain, [&](size_t a,size_t b,size_t tid){ ... });
- ThreadLocal<T> scratch(pool.size()); scratch[tid].amp.ensure(n_bands); scratch[tid].segments.clear();
- For work stealing, each worker owns deque<Chunk>; thieves steal half or one large chunk, not individual iterations.
- Do not call Eigen parallel kernels inside these tasks unless EIGEN_DONT_PARALLELIZE is false intentionally and benchmarked.


## Patch blueprint: surface_spline

_Lowest-risk concrete refactor_

### Data precompute
- Build TriGeom arrays once: int v[3], Vector3d v0, edge1, edge2, normal, centroid, valid flag.
- Build vertex→tri adjacency as CSR: offsets[n_verts+1], tri_ids[3*n_tris]. This avoids vector<vector<int>> overhead and improves locality.
- Build tri one-ring CSR once: for each triangle, collect adjacency from its three vertices into small local fixed buffer, sort, unique, append.
- Store one-ring as offsets[n_tris+1], tri_ids_flat[]. fit_one_triangle then iterates a contiguous span.

### Solve rewrite
- Replace rows/rhs/A/b dynamic allocation with direct accumulation of Matrix<double,6,6> ATA and Matrix<double,6,1> ATb.
- Function add_sample(row, rhs, weight): ATA.noalias() += weight * row.transpose()*row; ATb.noalias() += weight * row.transpose()*rhs.
- Always add ridge_lambda plus epsilon to curvature diagonal. If decomposition fails or rank is low, output zeros/lower-order fit.
- Keep public surface_spline_fit signature unchanged; use provided ThreadPool or persistent global rather than constructing when repeatedly called.


## Patch blueprint: field_march

_Replace reference loop with explicit kernel family_

### Kernel split
- Keep field_grid_step_helmholtz_regular as API wrapper; inside, dispatch to legacy or optimized kernel by flag.
- Create step_helmholtz_regular_serial_doublebuf(src,dst), step_helmholtz_regular_threaded(pool,src,dst), and optionally step_helmholtz_regular_soa.
- Allocate secondary buffer once per FieldGrid or via caller workspace, not per call if n_steps repeats frequently.
- For odd n_steps, final data may live in scratch; copy/swap ownership carefully so field_grid_data_re_im still returns current data.

### Inner loop shape
- for b in bands: src_band = src + b*cells; dst_band = dst + b*cells; parallel z slabs for interior; copy or preserve boundary according to policy.
- Use pointer offsets: i0=z*plane+y*nx+1; loop x=1..nx-2 increment i. Avoid int64 multiply per cell when possible.
- Compute real/imag explicitly if std::complex blocks vectorization: lap_re/lap_im, dst_re = c_re + ..., dst_im = c_im + ....
- Benchmark with and without vector pragmas; inspect compiler vectorization reports.


## Patch blueprint: ray tracing

_Parallel stateless trace without corrupting outputs_

### Trace task
- Define TraceTask { source_id, ray_begin, ray_end, global_ray_begin }. The task range is the unit of scheduling and deterministic output ordering.
- WorkerTraceScratch { SmallBandVec amp, amp_prop; vector<float> segments; vector<SparseContribution> direct, indirect; vector<StrikeRow> strikes; RayRng rng; counters; }.
- trace_rays_parallel calls trace_rays_range_serial for each task with worker scratch and immutable SceneCache.
- After all tasks: merge segments by task order; reduce contributions by destination arrays; copy counts/counters.

### Scene cache
- SceneCache contains const raw pointers/spans to tris/BVH/material_cache/k_real/atmo_abs/contexts and scalar config.
- No trace worker mutates RayTracerState. Camera full-march stats become per-worker counters reduced at the end.
- OutputPolicy is passed into range serial function: SegmentOnly, SurfaceAccum, IRAccum, ImageAccum, EndpointAccum.
- Keep old serial functions as wrappers until parallel version reproduces output; then internal implementation can be shared.


## Numerical and physics caveats

_Optimization must not quietly change the model_

### Potential semantic bugs to audit
- Geometric spreading uses 1/(1+path_len+t_min*0.5) in trace loops; confirm intended distance reference. This is not pure 1/r or segment-local spreading.
- mat_refl_complex derives phase from normal-incidence Fresnel using IOR but uses authored reflectance magnitude. Cache exactly this unless changing model intentionally.
- Transmissive path in multiscale uses roulette for reflection/transmission but transmission keeps amp_surf without scaling by transmittance; verify this is intended MC energy treatment.
- Stateful apply_surface multiplies amp by reflectance even on refractive direction; compare with multiscale logic and intended semantics.

### Parallel floating point
- Reduction order changes results. For deterministic mode, sort sparse contributions and reduce in stable key order, or use pairwise tree reduction with fixed order.
- SIMD/fast-math may change phase and attenuation slightly; test with relative/absolute tolerances by physical quantity, not only raw floats.
- Complex field marching stability may depend on dt, dx, k. Add stability diagnostics before optimizing into a faster diverging kernel.
- Do not remove EPS nudges until ray self-intersection tests prove an alternative.


## Profiling checklist

_What to run after every major patch_

### CPU profiling
- Linux: perf stat + perf record/report; Windows: VTune or Visual Studio Profiler; cross-platform: Tracy zones around exported calls and hot kernels.
- Collect: cycles, instructions, branch misses, cache misses, LLC loads/stores, vectorization, memory bandwidth, thread utilization, lock contention.
- Use compiler vectorization reports for field march and band loops; do not assume Eigen or std::complex vectorized.
- Record benchmark metadata: CPU model, compiler, flags, thread count, grid/ray/triangle/band counts, seed, output cap.

### Performance gates
- Surface spline: eliminate nearly all heap allocations inside per-triangle worker; expect large speedup on high-triangle meshes.
- Field march: optimized double-buffer threaded kernel should scale with memory bandwidth; single-thread should improve from no memcpy.
- Ray trace: parallel speedup should be near-linear until output reduction, memory bandwidth, or BVH cache misses dominate.
- Regression failure policy: if physics output diverges before known floating-order tolerance, stop and bisect.


## What not to do

_Avoid plausible changes that will create hidden damage_

### Rejected shortcuts
- Do not replace complex spectral amplitude with magnitude-only intensity to gain speed in the primary path.
- Do not use one global mutex around output arrays; it will serialize the workload and hide data-race design problems.
- Do not turn every += into atomic and call it done; atomics will be slow, nondeterministic for floats, and can become the hot path under caustics/sensors.
- Do not parallelize by bounce globally unless ray state compaction and output ownership are designed; independent ray paths are the natural first unit.
- Do not introduce GPU dependencies into these CPU files until CPU correctness and memory layout are stabilized.

### Subtle hazards
- Do not let a temporary Eigen::Map outlive worker scratch or point to a vector that can reallocate.
- Do not assume std::complex<T> binary layout for external ABI beyond the existing reinterpret_cast paths unless guarded by static_asserts and tests.
- Do not reuse ThreadPool from inside a task to schedule nested work unless the pool supports work-first/helping semantics; deadlock risk.
- Do not make BVH build parallel before traversal is optimized unless scene rebuild is measured hot.


## Acceptance criteria

_Definition of done for the optimization pass_

### Correctness
- All legacy serial golden tests pass or documented tolerance deltas are accepted by explicit test names.
- Threaded deterministic mode produces stable output across repeated runs with same seed/thread count; if task stealing changes order, canonical output comparison passes.
- Sanitizer builds are clean for representative small scenes: ASan, UBSan, and TSan for threaded code where practical.
- C ABI layout checks pass; Python/ctypes and GLSL mat_buf compatibility unchanged.

### Performance
- Surface spline: no per-triangle dynamic MatrixXd/VectorXd allocation in normal path; one-ring allocation removed; speedup measured.
- Field march: no per-band per-step full memcpy in optimized path; parallel z/band kernel demonstrates scaling and exact/tolerant match.
- Ray tracing: stateless trace supports per-worker output buffers and parallel task ranges; no shared RNG; output accumulations race-free.
- Profiling report attached with before/after numbers and remaining top hotspots.


## Concrete code sketch: fixed spline normal equations
```cpp
// Replace per-triangle MatrixXd assembly with fixed-size accumulation.
using Mat6 = Eigen::Matrix<double,6,6>;
using Vec6 = Eigen::Matrix<double,6,1>;
using Row6 = Eigen::Matrix<double,1,6>;

Mat6 ATA = Mat6::Zero();
Vec6 ATb = Vec6::Zero();

auto add_sample = [&](const Row6& r, double y, double w) {
    ATA.noalias() += w * r.transpose() * r;
    ATb.noalias() += w * r.transpose() * y;
};

add_sample(basis(1.0/3.0, 1.0/3.0), 0.0, anchor_w);
// add neighbor displacement rows, normal-gradient rows...
for (int j=3; j<6; ++j) ATA(j,j) += ridge_lambda + 1e-12;

Eigen::LDLT<Mat6> ldlt(ATA);
if (ldlt.info() == Eigen::Success) {
    Vec6 c = ldlt.solve(ATb);
    std::memcpy(out6, c.data(), 6*sizeof(double));
} else {
    std::memset(out6, 0, 6*sizeof(double));
}
```


## Concrete code sketch: per-worker ray trace result
```cpp
struct RayTask { int source_id; int ray_begin; int ray_end; uint64_t task_seed; };
struct WorkerTraceScratch {
    SmallBandVec amp, amp_prop;
    std::vector<float> segments;              // raw RT_FLOATS_PER_SEG rows
    std::vector<SparseContribution> direct;   // {tri_band_key, value}
    std::vector<SparseContribution> indirect;
    ProfCounters counters;
};

parallel_for_chunks(pool, 0, tasks.size(), 1, [&](size_t a, size_t b, size_t tid) {
    auto& scratch = worker_scratch[tid];
    scratch.clear_keep_capacity();
    for (size_t ti=a; ti<b; ++ti) {
        trace_task_range_serial(scene_cache, tasks[ti], scratch, output_policy);
    }
});

// Deterministic merge: task order or sorted sparse keys.
merge_segments_by_task_order(worker_scratch, out_segs, out_cap, out_count);
reduce_sparse_contribs(worker_scratch, out_direct, out_indirect);
```


## Concrete code sketch: field march slab kernel
```cpp
for (int step = 0; step < n_steps; ++step) {
    parallel_for_chunks(pool, 0, g->n_bands * nz_slab_count, 1,
      [&](size_t a, size_t b, size_t tid) {
        for (size_t job=a; job<b; ++job) {
            int band = job / nz_slab_count;
            int slab = job % nz_slab_count;
            int z0 = slab_z0(slab), z1 = slab_z1(slab);
            const cd32* __restrict src = src_grid + band*cells;
            cd32* __restrict dst = dst_grid + band*cells;
            for (int z=z0; z<z1; ++z)
              for (int y=1; y<ny-1; ++y) {
                int64_t i = (int64_t)z*plane + y*nx + 1;
                for (int x=1; x<nx-1; ++x, ++i) {
                    cd32 c = src[i];
                    cd32 lap = (src[i+1]+src[i-1]-2.f*c)*dxi2
                             + (src[i+nx]+src[i-nx]-2.f*c)*dyi2
                             + (src[i+plane]+src[i-plane]-2.f*c)*dzi2;
                    dst[i] = c + i_k2dt*c + i_dt*lap;
                }
              }
        }
    });
    std::swap(src_grid, dst_grid);
}
```


## Agent handoff checklist
- Build and run unmodified; establish baselines.
- Add instrumentation/counters without behavior changes.
- Refactor spline fixed-size and material cache first.
- Implement field double-buffer/threaded slab stepping.
- Introduce persistent thread pool/work stealing API.
- Parallelize stateless ray traces with per-worker outputs and deterministic merge.
- Optimize BVH/stateful scheduler after safer wins.
- Document semantics, determinism, benchmarks, and risk per patch.
