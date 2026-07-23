fftfree — High‑Performance FFT with Flexible Dispatch and C API

Overview

- Header‑only FFT core built on Eigen for vectorization.
- Algorithms: Cooley–Tukey (bit‑reversed, natural order output) and Stockham autosort.
- Transform modes: C2C, R2C, C2R, optional half‑spectrum packing for real FFTs.
- Output shaping: magnitude, or polar packing (real=|X|, imag=phase).
- Metadata capture: twiddles, layout maps, stage snapshots, butterfly pairs, twiddle indices, invariants, and ωk (s‑plane) per bin. Per‑batch variants available.

Threading Model (Important)

- Kernels do not own threads. Inner parallelism is disabled by default.
- A single outer WorkerPool (in the C API) owns the total thread budget and may parallelize batched windows across threads.
- Algorithms use a light `JobDispatcher` interface so the same hot loops can run inline (serial) or post jobs to the outer pool, depending on configuration.
- Both Cooley–Tukey and Stockham obtain their parallelism exclusively through `Plan::dispatcher()`; if you do not install a pool-backed dispatcher, they execute serially.

Three layers of parallelism (configurable):

1) Internal parallelism (kernel‑level)
   - Off by default. Only enabled when `allow_inner_parallel=1` and outer threads == 1.
   - Controlled by `PlanRuntimeConfig.allow_inner_parallel` and `inner_threads`.

2) Grouped plan parallelism (outer pool splitting windows)
   - Enabled when `allow_outer_parallel=1` and `threads>1` in `fft_init_full`.
   - The C API installs a dispatcher that routes algorithm jobs to the shared pool.

3) Streaming input parallelism (pipeline)
   - Supported by the outer pool design; enqueue additional work without saturating threads.

C API (CFFI‑friendly)

Key initializer (full control):

  void* fft_init_full(size_t n,
                      int threads,            // outer threads
                      int lanes,              // SIMD lane hint
                      int inverse,            // 0/1
                      int kernel,             // 1=CT, 2=Stockham, 0=auto
                      int radix,
                      const int* radix_pattern, size_t radix_pattern_len,
                      int pad_mode,           // 0=auto,1=always,2=never
                      int window, int hop, int stft_mode,   // STFT helpers
                      int transform,          // 0=C2C,1=R2C,2=C2R,3=R2R
                      int reduce_magnitude,   // 0/1
                      int store_polar,        // 0/1
                      int half_spectrum,      // 0/1
                      int allow_outer_parallel, // 0/1
                      int allow_inner_parallel, // 0/1 (only when outer threads==1)
                      int inner_threads);

Execute (batched):

  size_t fft_execute_batched(void* ctx,
                             const float* pcm, size_t pcm_len,
                             float* out_real, float* out_imag, float* out_mag,
                             int pad_mode, size_t max_frames);

Dispatcher API (internal)

- `Plan::set_dispatcher(JobDispatcher*)` installs an external dispatcher used by algorithms.
- Default is `InlineDispatcher` (serial). The C API provides a `PoolDispatcher` bound to the outer WorkerPool.

Metadata Kinds

- Twiddle, LayoutMap, StageSnapshot, StageParams, ButterflyPairs, StageInvariant, TwiddleIndexMap, Omega.
- Per‑batch variants: StageSnapshotAll, ButterflyPairsAll, TwiddleIndexMapAll.

Real FFT specifics

- Half‑spectrum packing (R2C/C2R): enable `half_spectrum` to use 0..N/2 bins. Inverse expands correctly.
- Polar packing: when `store_polar=1`, each complex slot stores (mag, phase) with the same footprint.

Notes

- No OpenMP is used in the core. All concurrency comes from the outer WorkerPool.
- Kernels are written to be safe when run under a dispatcher or inline.

