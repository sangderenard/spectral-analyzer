// fft_cffi.hpp
// C-compatible FFT function for cffi
#pragma once
#include <cstddef>
#include <complex>
// Expose core enums from the library headers rather than inventing new
// duplicated values here. Callers should include this header to get the
// authoritative kernel/radix definitions.
#include "eigen_fft.hpp"

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef FFT_CFFI_EXPORTS
#    define FFT_CFFI_API __declspec(dllexport)
#  else
#    define FFT_CFFI_API __declspec(dllimport)
#  endif
#else
#  define FFT_CFFI_API
#endif

// Preferred alias moving forward: FFTFREE_API
#ifndef FFTFREE_API
#define FFTFREE_API FFT_CFFI_API
#endif

extern "C" {
    // in_pcm: pointer to float PCM data (mono)
    // out_real, out_imag, out_mag: pointers to float arrays for output
    // n: number of samples
    // threads: desired number of threads (>=1). If <=0, implementation clamps to a safe default.
    FFT_CFFI_API void fft_pcm_to_channels(const float* in_pcm, float* out_real, float* out_imag, float* out_mag, size_t n, int threads);

    // Simple deployment-oriented API
    // Create a reusable FFT context with prebuilt plans.
    //   n: FFT size (must be power of two)
    //   threads: number of thread-local plan instances to precreate (>=1)
    //   lanes: lane capacity hint (<=0 => auto)
    //   inverse: non-zero for inverse transform
    // Returns an opaque handle or NULL on failure.
    // kernel: 0=auto, 1=cooleytukey, 2=stockham
    // radix: 0=unspecified (caller omitted). If radix==0 and kernel is specified
    //         the Plan defaults for that algorithm are used for any unspecified
    //         parameters. If radix==0 and kernel==0 (auto) the radix defaults
    //         to 2. Otherwise provide the desired radix (e.g., 2 or 4).
    // pad_mode: 0=auto (pad if required by kernel), 1=always pad up to next power-of-two, 2=never pad (error if incompatible)
    //
    // Phase-2 additions:
    // - exported integer constants for kernels/pad/radix are provided below
    // - radix_pattern: optional pointer to an array of ints describing a mixed-radix
    //   pattern (e.g., {2,2,4}). The pattern is copied by the callee; caller may
    //   pass nullptr with len=0 to indicate no pattern.

    // Do NOT invent or duplicate core enums here. Use the values defined in
    // `eigfft::KernelKind` and `eigfft::ButterflyRadix` from the library
    // headers. For convenience bind C++ constants here so C++ callers can use
    // them without repeating the enum definitions.
}

// Provide C-compatible integer constants that are derived from the authoritative
// enums in the library headers. These are computed at compile-time so they
// always reflect the upstream enum ordering/values.
enum {
    FFT_KERNEL_COOLEYTUKEY = static_cast<int>(eigfft::KernelKind::CooleyTukey),
    FFT_KERNEL_STOCKHAM = static_cast<int>(eigfft::KernelKind::Stockham),
    FFT_KERNEL_EXTERNAL = static_cast<int>(eigfft::KernelKind::External)
};

enum {
    FFT_BUTTERFLY_RADIX_2 = static_cast<int>(eigfft::ButterflyRadix::Radix2),
    FFT_BUTTERFLY_RADIX_4 = static_cast<int>(eigfft::ButterflyRadix::Radix4),
    FFT_BUTTERFLY_RADIX_8 = static_cast<int>(eigfft::ButterflyRadix::Radix8),
    FFT_BUTTERFLY_RADIX_16 = static_cast<int>(eigfft::ButterflyRadix::Radix16)
};

// Transform mode constants for C callers
enum {
    FFT_TRANSFORM_C2C = 0,
    FFT_TRANSFORM_R2C = 1,
    FFT_TRANSFORM_C2R = 2,
    FFT_TRANSFORM_R2R = 3
};

extern "C" {

    // New Phase-2 ABI: accept radix_pattern pointer and length. Caller may pass
    // nullptr/0 to indicate no pattern. The pattern values are plain ints (e.g.
    // 2,4). radix == 0 still means unspecified and probing behavior remains.
    // Backwards-compatible init: existing behavior preserved. Use fft_init_ex
    // to pass STFT/window parameters.
    FFT_CFFI_API void* fft_init(size_t n, int threads, int lanes, int inverse, int kernel, int radix, const int* radix_pattern, size_t radix_pattern_len, int pad_mode);

    // Extended initializer (STFT-aware). Parameters:
    //  - window: analysis window size (W). If 0, defaults to plan N (no extra subwindowing).
    //  - hop: hop/stride in samples between windows. If 0, defaults to window.
    //  - stft_mode: 0=disabled (legacy), 1=batched STFT helper enabled, 2=streaming mode (reserved)
    FFT_CFFI_API void* fft_init_ex(size_t n, int threads, int lanes, int inverse, int kernel, int radix, const int* radix_pattern, size_t radix_pattern_len, int pad_mode, int window, int hop, int stft_mode);

    // Full initializer (original signature, kept for ABI compatibility)
    FFT_CFFI_API void* fft_init_full(size_t n,
                                     int threads,
                                     int lanes,
                                     int inverse,
                                     int kernel,
                                     int radix,
                                     const int* radix_pattern,
                                     size_t radix_pattern_len,
                                     int pad_mode,
                                     int window,
                                     int hop,
                                     int stft_mode,
                                     int transform,
                                     int reduce_magnitude,
                                     int store_polar,
                                     int half_spectrum,
                                     int allow_outer_parallel,
                                     int allow_inner_parallel,
                                     int inner_threads,
                                     int save_crash_logs,
                                     int silent_crash_reports);

    // Full initializer v2 with window/OLA configuration appended (preferred)
    FFT_CFFI_API void* fft_init_full_v2(size_t n,
                                        int threads,
                                        int lanes,
                                        int inverse,
                                        int kernel,
                                        int radix,
                                        const int* radix_pattern,
                                        size_t radix_pattern_len,
                                        int pad_mode,
                                        int window,
                                        int hop,
                                        int stft_mode,
                                        int transform,
                                        int reduce_magnitude,
                                        int store_polar,
                                        int half_spectrum,
                                        int allow_outer_parallel,
                                        int allow_inner_parallel,
                                        int inner_threads,
                                        int save_crash_logs,
                                        int silent_crash_reports,
                                        int apply_windows,
                                        int apply_ola,
                                        int analysis_window_kind,
                                        float analysis_param1,
                                        float analysis_param2,
                                        int synthesis_window_kind,
                                        float synthesis_param1,
                                        float synthesis_param2,
                                        int window_norm_policy,
                                        int cola_mode);

    // Execute FFT using a pre-initialized context. Returns 1 on success, 0 on error.
    FFT_CFFI_API int fft_execute(void* handle,
                                 const float* in_pcm,
                                 float* out_real,
                                 float* out_imag,
                                 float* out_mag,
                                 size_t n);
    // Batched STFT helper: scatter overlapping windows from `pcm` (length pcm_len)
    // into independent N-sized columns, run a batched FFT across all frames and
    // write outputs into out_real/out_imag/out_mag arrays. Caller must provide
    // buffers with capacity at least max_frames * bins where
    // bins = ctx_N if half_spectrum is disabled, otherwise (N/2 + 1). The
    // function returns the number of frames produced (0 on error). Use pad_mode
    // to control final partial-frame padding (0=auto/refuse per init rules,
    // 1=pad last, 2=never). Layout: output is flattened frame-major with the
    // per-frame stride equal to `bins` (frame0_bin0..frame0_bin(bins-1), frame1_...)
    FFT_CFFI_API size_t fft_execute_batched(void* handle,
                                           const float* pcm,
                                           size_t pcm_len,
                                           float* out_real,
                                           float* out_imag,
                                           float* out_mag,
                                           int pad_mode,
                                           int enable_backup,
                                           size_t max_frames);

        FFT_CFFI_API int fft_griffin_lim(void* ctx_forward,
                                         void* ctx_inverse,
                                         const float* in_mag,
                                         size_t frames,
                                         int hop,
                                         int half_spectrum,
                                         int iterations,
                                         int pad_mode,
                                         unsigned int seed,
                                         float* out_real,
                                         float* out_imag);
    // Execute inverse (or complex-input) batched transforms using complex inputs.
    // in_real/in_imag: flattened frame-major arrays with `frames * bins` entries
    // bins = ctx_N if not half_spectrum, otherwise (N/2 + 1).
    // Returns number of frames produced (0 on error).
    FFT_CFFI_API size_t fft_execute_complex_batched(void* handle,
                                                   const float* in_real,
                                                   const float* in_imag,
                                                   size_t frames,
                                                   float* out_pcm,
                                                   int pad_mode,
                                                   int enable_backup,
                                                   size_t max_frames);
    // Query effective FFT size (plan size) for a context
    FFT_CFFI_API size_t fft_ctx_size(void* handle);
    // Query the number of worker threads available to the context (outer parallelism).
    FFT_CFFI_API size_t fft_ctx_worker_threads(void* handle);
    // Query the effective plan threads (after runtime/hardware limits).
  FFT_CFFI_API size_t fft_ctx_effective_threads(void* handle);

  // Test-only control: set WorkerPool debug dropout percentage [0,100].
  // This function sets the value inside the library (DLL) so worker threads
  // created by the library observe the configured probability.
  FFT_CFFI_API void fft_set_workerpool_debug_dropout(int pct);
  
  // Test-only control: configure deterministic dropout stripe pattern.
  // When period>0 and on_len>0, dropout is enabled for chunk indices i where
  // ((i + phase) mod period) < on_len. Use this to alternate ON/OFF windows
  // across the batch deterministically.
  FFT_CFFI_API void fft_set_workerpool_dropout_pattern(int period, int on_len, int phase);

  // Test-only control: configure a single chunk index to force dropout.
  // Pass -1 to disable.
  FFT_CFFI_API void fft_set_workerpool_dropout_single(long long index);

  // Test-only control: toggle persistent dropout (keep failing on retries).
  // When off (0), dropouts are one-shot per chunk index to exercise restore
  // and then allow recovery to succeed. When on (1), the same chunk will keep
  // dropping on retries to simulate persistent failure.
  FFT_CFFI_API void fft_set_workerpool_dropout_persistent(int on);
  
  // Test-only control: clear the internal one-shot history so indices can
  // drop again in subsequent phases.
  FFT_CFFI_API void fft_clear_workerpool_dropout_history();

  // Phase inference API (experimental)
  // Initialize a phase inference context. mode: 0=linear. iterations reserved for future.
  FFT_CFFI_API void* phase_init(int N, int hop, int half_spectrum, int mode, int iterations);
  // Infer complex spectra from magnitudes (frame-major). Returns frames on success or 0 on failure.
  FFT_CFFI_API size_t phase_infer_execute(void* handle,
                                          const float* in_mag,
                                          size_t frames,
                                          float* out_real,
                                          float* out_imag);
    FFT_CFFI_API void phase_free(void* handle);

    // Destroy a context created by fft_init (safe to pass NULL).
    FFT_CFFI_API void fft_free(void* handle);

    // Test-only: register a restore observer callback that will be invoked
    // whenever the library's restore path copies a column range from the
    // backup into the working buffer. This is intended for tests and
    // diagnostics only.
    typedef void (*fft_restore_observer_t)(size_t start, size_t end);
    FFT_CFFI_API void fft_register_restore_observer(fft_restore_observer_t cb);
}

// Windowing API (analysis/synthesis configuration)
// Window kinds for analysis/synthesis. Rectangular is default/backward compatible.
enum {
    FFT_WINDOW_RECT = 0,
    FFT_WINDOW_HANN = 1,
    FFT_WINDOW_HAMMING = 2,
    FFT_WINDOW_BLACKMAN = 3,
    FFT_WINDOW_TUKEY = 4,
    FFT_WINDOW_KAISER = 5
};

// Window normalization policy
enum {
    FFT_WINDOW_NORM_NONE = 0,   // raw coefficients
    FFT_WINDOW_NORM_L2 = 1,     // normalize to unit L2 energy
    FFT_WINDOW_NORM_AREA = 2    // normalize to unit sum
};

// COLA (constant overlap-add) guidance for synthesis window normalization.
// Note: the library returns per-frame PCM; OLA is performed by the caller.
enum {
    FFT_COLA_OFF = 0,
    FFT_COLA_NORMALIZE = 1
};

extern "C" {
    // Configure built-in analysis/synthesis windows for a context. Parameters:
    //  - analysis_kind, synthesis_kind: FFT_WINDOW_*
    //  - param1/param2: shape parameters (e.g., Tukey alpha, Kaiser beta)
    //  - norm_policy: FFT_WINDOW_NORM_*
    //  - cola_mode: FFT_COLA_*
    // Returns 1 on success, 0 on failure.
    FFT_CFFI_API int fft_config_window(void* handle,
                                       int analysis_kind,
                                       int synthesis_kind,
                                       float param1,
                                       float param2,
                                       int norm_policy,
                                       int cola_mode);

    // Configure custom windows by passing coefficient arrays. Either pointer may
    // be NULL to indicate unchanged. Lengths must match the context's `window`.
    FFT_CFFI_API int fft_config_window_custom(void* handle,
                                              const float* analysis_window,
                                              size_t analysis_len,
                                              const float* synthesis_window,
                                              size_t synthesis_len,
                                              int norm_policy,
                                              int cola_mode);
}
