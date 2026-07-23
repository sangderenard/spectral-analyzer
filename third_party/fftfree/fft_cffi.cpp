// fft_cffi.cpp

#include "fft_cffi.hpp"
#include "eigen_fft.hpp"
#include "plan_support.hpp"
#include "recovery_ops.hpp"
#include "crash_handler.hpp"
#include "phase_infer.hpp"
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <limits>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cctype>
#if defined(_WIN32)
#include <windows.h>
#endif

#if defined(_WIN32)
// windows.h may define macros min/max; avoid interfering with std::min/std::max
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif

namespace {
struct FftContext {
    eigfft::PlanCache<float> cache;
    eigfft::PlanRuntimeConfig cfg{};
    int N = 0;
    bool inverse = false;
    int kernel = 0;   // 0=auto,1=CT,2=Stockham
    int radix = 0;    // 0=unspecified, otherwise 2/4/8/16
    int pad_mode = 0; // 0=auto,1=always,2=never
    // Copied mixed-radix pattern from outer API (values like 2,4,...)
    std::vector<int> radix_pattern;
    // STFT/windowing settings
    int window = 0;   // analysis window (W). 0 => use plan N
    int hop = 0;      // hop/stride between windows. 0 => window
    int stft_mode = 0; // 0=disabled,1=batched helper,2=streaming (reserved)
    // Transform/output shape
    int transform = 0;        // 0=C2C,1=R2C,2=C2R,3=R2R
    int reduce_magnitude = 0; // bool-like
    int store_polar = 0;      // bool-like
    int half_spectrum = 0;    // bool-like
    int allow_outer_parallel = 1;
    int allow_inner_parallel = 0;
    int inner_threads = 0;
    // Windowing configuration (analysis/synthesis)
    int analysis_win_kind = 0;   // FFT_WINDOW_*
    int synthesis_win_kind = 0;  // FFT_WINDOW_*
    float win_param1 = 0.0f;     // e.g., Tukey alpha, Kaiser beta
    float win_param2 = 0.0f;
    int win_norm_policy = 0;     // FFT_WINDOW_NORM_*
    int cola_mode = 0;           // FFT_COLA_*
    std::vector<float> analysis_win;   // length = window
    std::vector<float> synthesis_win;  // length = window
    bool windows_enabled = false;      // Whether to apply windows internally (default off)
    int apply_ola = 0;                 // Whether to perform OLA internally in inverse
    // Persistent worker pool (outer parallelism only). Use the canonical
    // WorkerPool implementation from `eigen_fft.hpp` to avoid duplication.
    std::unique_ptr<eigfft::WorkerPool> pool;
};

// Forward declarations for window helpers used by fft_init_full
static void normalize_window(std::vector<float>& w, int policy);
static std::vector<float> build_window(int kind, int W, float p1, float p2);

static std::string trim_and_lower(std::string value) {
    auto begin = value.begin();
    while (begin != value.end() && std::isspace(static_cast<unsigned char>(*begin))) {
        ++begin;
    }
    auto end = value.end();
    while (end != begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
        --end;
    }
    std::string trimmed(begin, end);
    std::transform(trimmed.begin(), trimmed.end(), trimmed.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return trimmed;
}

static int parse_kernel_token(const std::string& token) {
    if (token.empty() || token == "0" || token == "auto" || token == "default") {
        return 0;
    }
    if (token == "cooleytukey" || token == "cooley" || token == "cooley-tukey" || token == "cooley_tukey" || token == "ct") {
        return FFT_KERNEL_COOLEYTUKEY;
    }
    if (token == "stockham" || token == "stock" || token == "sh") {
        return FFT_KERNEL_STOCKHAM;
    }
    if (token == "external" || token == "ext" || token == "custom") {
        return FFT_KERNEL_EXTERNAL;
    }
    char* end = nullptr;
    long parsed = std::strtol(token.c_str(), &end, 10);
    if (end && *end == '\0') {
        if (parsed == FFT_KERNEL_COOLEYTUKEY || parsed == FFT_KERNEL_STOCKHAM || parsed == FFT_KERNEL_EXTERNAL) {
            return static_cast<int>(parsed);
        }
    }
    return 0;
}

static int env_default_kernel_override() {
    static const int cached = []() {
        const char* env = std::getenv("FFTFREE_DEFAULT_KERNEL");
        if (!env) {
            return 0;
        }
        const std::string raw(env);
        const std::string lowered = trim_and_lower(raw);
        if (lowered.empty()) {
            return 0;
        }
        const int mapped = parse_kernel_token(lowered);
        if (mapped != 0) {
            std::fprintf(stderr, "[diag] FFTFREE_DEFAULT_KERNEL override: %s -> %d\n", raw.c_str(), mapped);
#if defined(_WIN32)
            if (!fftfree::crash_handler_silent_enabled()) {
                std::string msg = "[diag] FFTFREE_DEFAULT_KERNEL override: " + raw + " -> " + std::to_string(mapped) + "\n";
                OutputDebugStringA(msg.c_str());
            }
#endif
        } else {
            std::fprintf(stderr, "[diag] FFTFREE_DEFAULT_KERNEL ignored: %s (expected cooleytukey|stockham|external|0|1|2)\n", raw.c_str());
#if defined(_WIN32)
            if (!fftfree::crash_handler_silent_enabled()) {
                std::string msg = "[diag] FFTFREE_DEFAULT_KERNEL ignored: " + raw + " (expected cooleytukey|stockham|external|0|1|2)\n";
                OutputDebugStringA(msg.c_str());
            }
#endif
        }
        return mapped;
    }();
    return cached;
}

// Window helper implementations placed before usage to avoid forward-declare/linkage issues
static inline double kPI_constexpr() { return 3.141592653589793238462643383279502884; }
static void normalize_window(std::vector<float>& w, int policy) {
    if (w.empty()) return;
    if (policy == FFT_WINDOW_NORM_L2) {
        double s2 = 0.0; for (float v : w) s2 += double(v) * double(v);
        s2 = std::sqrt(std::max(1e-30, s2));
        if (s2 > 0) { for (auto& v : w) v = float(v / s2); }
    } else if (policy == FFT_WINDOW_NORM_AREA) {
        double s = 0.0; for (float v : w) s += double(v);
        if (std::abs(s) > 1e-30) { for (auto& v : w) v = float(v / s); }
    }
}
static std::vector<float> build_window(int kind, int W, float p1, float p2) {
    std::vector<float> w(W, 1.0f);
    if (W <= 0) return w;
    switch (kind) {
    case FFT_WINDOW_RECT:
        for (int n = 0; n < W; ++n) w[n] = 1.0f; break;
    case FFT_WINDOW_HANN:
        for (int n = 0; n < W; ++n) w[n] = float(0.5 * (1.0 - std::cos(2.0 * kPI_constexpr() * n / (W - 1)))); break;
    case FFT_WINDOW_HAMMING:
        for (int n = 0; n < W; ++n) w[n] = float(0.54 - 0.46 * std::cos(2.0 * kPI_constexpr() * n / (W - 1))); break;
    case FFT_WINDOW_BLACKMAN:
        for (int n = 0; n < W; ++n) {
            double a0 = 0.42, a1 = 0.5, a2 = 0.08;
            w[n] = float(a0 - a1 * std::cos(2.0 * kPI_constexpr() * n / (W - 1)) + a2 * std::cos(4.0 * kPI_constexpr() * n / (W - 1)));
        }
        break;
    case FFT_WINDOW_TUKEY: {
        double alpha = (p1 == 0.0f ? 0.5 : p1);
        int L = std::max(1, W - 1);
        for (int n = 0; n < W; ++n) {
            double x = double(n) / double(L);
            if (x < alpha / 2) w[n] = float(0.5 * (1 + std::cos(kPI_constexpr() * (2 * x / alpha - 1))));
            else if (x <= 1 - alpha / 2) w[n] = 1.0f;
            else w[n] = float(0.5 * (1 + std::cos(kPI_constexpr() * (2 * x / alpha - 2 / alpha + 1))));
        }
        break; }
    case FFT_WINDOW_KAISER: {
        // Placeholder: use Hann until Bessel I0 is wired
        for (int n = 0; n < W; ++n) w[n] = float(0.5 * (1.0 - std::cos(2.0 * kPI_constexpr() * n / (W - 1))));
        break; }
    default:
        break;
    }
    return w;
}

static eigfft::PlanRuntimeConfig compute_effective_runtime(const FftContext& ctx) {
    eigfft::PlanRuntimeConfig cfg = ctx.cfg;
    int outer_threads = (ctx.cfg.threads > 0) ? ctx.cfg.threads : 1;
    if ((ctx.allow_outer_parallel != 0) && ctx.pool) {
        const int pool_threads = ctx.pool->size();
        if (pool_threads > 0) {
            outer_threads = pool_threads;
        }
    }
    const bool allow_inner = (ctx.allow_inner_parallel != 0);
    int inner_budget = 0;
    if (allow_inner) {
        if (ctx.inner_threads > 0) {
            inner_budget = ctx.inner_threads;
        } else {
            int hw = static_cast<int>(std::thread::hardware_concurrency());
            inner_budget = (hw > 0) ? hw : 1;
        }
    }
    int plan_threads = outer_threads;
    if (allow_inner) {
        plan_threads = std::max(plan_threads, inner_budget);
    }
    plan_threads = std::max(plan_threads, 1);
    cfg.threads = plan_threads;
    cfg.allow_inner_parallel = allow_inner;
    cfg.inner_threads = allow_inner ? ctx.inner_threads : 0;
    if (allow_inner && outer_threads > 1) {
        static std::once_flag permute_hint_flag;
        std::call_once(permute_hint_flag, [outer_threads]() {
            std::fprintf(stderr, "[diag] compute_effective_runtime: inner parallelism active with %d outer threads. To serialize only the permute stage build with -DFFTFREE_CT_FORCE_INLINE_PERMUTE=1.\n", outer_threads);
#if defined(_WIN32)
            if (!fftfree::crash_handler_silent_enabled()) {
                std::string msg = "[diag] compute_effective_runtime: use -DFFTFREE_CT_FORCE_INLINE_PERMUTE=1 to serialize the permute stage when inner parallelism is enabled.\n";
                OutputDebugStringA(msg.c_str());
            }
#endif
        });
    }

    // Keep inner parallelism as computed above; do not force-disable.
#if EIGFFT_RUNTIME_INSTRUMENTATION
    {
        int pool_sz = ctx.pool ? ctx.pool->size() : 0;
        fprintf(stderr, "[diag] compute_effective_runtime: allow_inner_parallel=%d inner_threads=%d pool=%d outer_threads=%d effective_threads=%d\n",
                cfg.allow_inner_parallel ? 1 : 0, cfg.inner_threads, pool_sz, outer_threads, cfg.threads);
#if defined(_WIN32)
        if (!fftfree::crash_handler_silent_enabled()) {
            OutputDebugStringA("[diag] compute_effective_runtime: using computed inner/outer parallelism\n");
        }
#endif
    }
#endif
    return cfg;
}

// Dispatcher that forwards jobs to the context's outer WorkerPool, or runs inline if unavailable.
struct PoolDispatcher : public eigfft::JobDispatcher {
    eigfft::WorkerPool* pool = nullptr;
    void parallel_for(size_t total, size_t chunk, const Fn& fn) override {
        if (!pool || total == 0) {
            eigfft::InlineDispatcher::instance().parallel_for(total, chunk, fn);
            return;
        }
        pool->parallel_for(total, (chunk==0)?1:chunk, fn);
    }

    void parallel_for_with_restore(size_t total,
                                   size_t chunk,
                                   const Fn& fn,
                                   std::function<void(size_t,size_t)> restore_fn,
                                   eigfft::RecoveryOps ops,
                                   void* job_ctx) override {
        if (!pool || total == 0) {
            eigfft::InlineDispatcher::instance().parallel_for_with_restore(total, chunk, fn,
                                                                          std::move(restore_fn),
                                                                          ops, job_ctx);
            return;
        }
        pool->parallel_for_with_restore_ex(total, (chunk==0)?1:chunk, fn,
                                           std::move(restore_fn), ops, job_ctx);
    }
};
// Test-only: optional restore observer. Tests may register a callback to be
// informed when the library's restore path copies column ranges back from
// the backup. Stored here in the translation unit so it can be invoked from
// restore callbacks throughout this file.
static std::function<void(size_t,size_t)> g_restore_observer = nullptr;

struct ColumnRangeRecoveryCtx {
    using Complex = std::complex<float>;
    Complex* buffer = nullptr;
    size_t rows = 0;
    const std::vector<std::pair<size_t,size_t>>* ranges = nullptr;
    const std::vector<Complex>* backup = nullptr; // optional full-batch backup
    std::vector<ColumnSnapshot> snapshots;
};

static ColumnSlice column_range_make_slice(const ColumnRangeRecoveryCtx& ctx, size_t chunk_index) {
    ColumnSlice slice;
    if (!ctx.buffer || !ctx.ranges || chunk_index >= ctx.ranges->size()) {
        slice.rowsN = 0;
        slice.width = 0;
        return slice;
    }
    const auto& range = ctx.ranges->at(chunk_index);
    if (range.second <= range.first || ctx.rows == 0) {
        slice.rowsN = 0;
        slice.width = 0;
        return slice;
    }
    const size_t width = range.second - range.first;
    const size_t rows = ctx.rows;
    if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        width > static_cast<size_t>(std::numeric_limits<int>::max())) {
        slice.rowsN = 0;
        slice.width = 0;
        return slice;
    }
    slice.rowsN = static_cast<int>(rows);
    slice.width = static_cast<int>(width);
    slice.elem_bytes = static_cast<int>(sizeof(ColumnRangeRecoveryCtx::Complex));
    slice.axis_stride = static_cast<std::ptrdiff_t>(sizeof(ColumnRangeRecoveryCtx::Complex));
    slice.batch_stride = static_cast<std::ptrdiff_t>(rows * sizeof(ColumnRangeRecoveryCtx::Complex));
    slice.base = reinterpret_cast<std::byte*>(ctx.buffer + range.first * rows);
    return slice;
}

static void column_range_init(void* job_ctx, size_t chunk_count) noexcept {
    auto* ctx = static_cast<ColumnRangeRecoveryCtx*>(job_ctx);
    if (!ctx) return;
    try {
        ctx->snapshots.clear();
        ctx->snapshots.resize(chunk_count);
    } catch (...) {
        ctx->snapshots.clear();
    }
}

static void column_range_capture(void* job_ctx, size_t chunk_index, int /*worker_id*/) noexcept {
    auto* ctx = static_cast<ColumnRangeRecoveryCtx*>(job_ctx);
    if (!ctx || !ctx->ranges) return;
    if (chunk_index >= ctx->snapshots.size()) return;
    ColumnSlice slice = column_range_make_slice(*ctx, chunk_index);
    if (!slice.base || slice.rowsN <= 0 || slice.width <= 0 || slice.elem_bytes <= 0) return;
    try {
        capture_columns(slice, ctx->snapshots[chunk_index]);
    } catch (...) {
        // Snapshot capture is best-effort; ignore failures.
    }
}

static void column_range_restore(void* job_ctx, size_t start_chunk, size_t end_chunk) noexcept {
    auto* ctx = static_cast<ColumnRangeRecoveryCtx*>(job_ctx);
    if (!ctx || !ctx->ranges) return;
    const size_t max_chunks = ctx->ranges->size();
    if (start_chunk >= max_chunks || start_chunk >= end_chunk) return;
    end_chunk = std::min(end_chunk, max_chunks);
    for (size_t idx = start_chunk; idx < end_chunk; ++idx) {
        ColumnSlice slice = column_range_make_slice(*ctx, idx);
        if (!slice.base || slice.rowsN <= 0 || slice.width <= 0) continue;
        const bool have_snapshot = (idx < ctx->snapshots.size()) && !ctx->snapshots[idx].bytes.empty();
        try {
            if (have_snapshot) {
                restore_columns(slice, ctx->snapshots[idx]);
            } else if (ctx->backup) {
                const auto& r = ctx->ranges->at(idx);
                const size_t cols = static_cast<size_t>(r.second - r.first);
                const size_t src_off = static_cast<size_t>(r.first) * ctx->rows;
                const size_t elem_count = cols * ctx->rows;
                if (!ctx->backup->empty() && ctx->buffer) {
                    std::copy_n(ctx->backup->data() + src_off, elem_count, ctx->buffer + src_off);
                }
            }
            if (g_restore_observer) {
                const auto& r = ctx->ranges->at(idx);
                g_restore_observer(r.first, r.second);
            }
        } catch (...) {
            // keep going on restore failures
        }
    }
}

static void column_range_finalize(void* job_ctx) noexcept {
    auto* ctx = static_cast<ColumnRangeRecoveryCtx*>(job_ctx);
    if (!ctx) return;
    ctx->snapshots.clear();
    ctx->ranges = nullptr;
    ctx->buffer = nullptr;
    ctx->rows = 0;
}

static eigfft::RecoveryOps make_column_range_recovery_ops() {
    eigfft::RecoveryOps ops;
    ops.init = &column_range_init;
    ops.capture_chunk = &column_range_capture;
    ops.restore_range = &column_range_restore;
    ops.finalize = &column_range_finalize;
    return ops;
}
}
extern "C" {
void fft_pcm_to_channels(const float* in_pcm, float* out_real, float* out_imag, float* out_mag, size_t n, int threads) {
    using namespace eigfft;
    using Complex = std::complex<float>;
    if (!in_pcm || !out_real || !out_imag || !out_mag || n == 0) {
        return;
    }
    try {
        // Copy PCM to complex input (imag=0)
        // Determine plan size: auto-pad to next power-of-two so this convenience
        // function behaves consistently with the fft_init/fft_execute path.
        size_t plan_n = n;
        // Simple helper in this translation unit lives below; declare inline logic here
        auto is_pow2 = [](size_t v) {
            return v != 0 && (v & (v - 1)) == 0;
        };
        auto next_pow2 = [](size_t v) {
            if (v <= 1) return static_cast<size_t>(1);
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
#if INTPTR_MAX == INT64_MAX
            v |= v >> 32;
#endif
            v++;
            return v;
        };

        if (!is_pow2(plan_n)) plan_n = next_pow2(plan_n);

        std::vector<Complex> input(plan_n);
        // copy provided samples and zero-pad remainder
        size_t i = 0;
        for (; i < n; ++i) input[i] = Complex(in_pcm[i], 0.0f);
        for (; i < plan_n; ++i) input[i] = Complex(0.0f, 0.0f);

        // Set up FFT plan
        PlanRuntimeConfig cfg;
        cfg.threads = (threads > 0) ? threads : 1;
        cfg.lanes = 1;
        PlanEnvironment<float> env;
        env.initialize(static_cast<int>(plan_n), false, cfg);
        auto& plan = env.plan();
        // Run FFT in-place
        Eigen::Map<Eigen::Matrix<Complex, Eigen::Dynamic, 1>> data(input.data(), static_cast<Eigen::Index>(plan_n));
        fft_inplace_batched<float>(data, plan);
        // Output real, imag, mag as binary
        // Only write out the first `n` samples (match previous behavior)
        for (size_t j = 0; j < n; ++j) {
            const auto v = data(static_cast<Eigen::Index>(j));
            out_real[j] = v.real();
            out_imag[j] = v.imag();
            out_mag[j] = std::abs(v);
        }
    } catch (const std::exception& ex) {
        // Do not let exceptions escape the C boundary; zero outputs as a safe fallback.
        for (size_t i = 0; i < n; ++i) {
            out_real[i] = 0.0f;
            out_imag[i] = 0.0f;
            out_mag[i] = 0.0f;
        }
#if defined(_WIN32)
        // Best-effort signal to debugger / stderr for diagnostics.
        if (!fftfree::crash_handler_silent_enabled()) {
            OutputDebugStringA("fft_pcm_to_channels: exception caught; outputs zeroed.\n");
        }
#endif
        (void)ex; // suppress unused warning when no logging available
    } catch (...) {
        for (size_t i = 0; i < n; ++i) {
            out_real[i] = 0.0f;
            out_imag[i] = 0.0f;
            out_mag[i] = 0.0f;
        }
#if defined(_WIN32)
        if (!fftfree::crash_handler_silent_enabled()) {
            OutputDebugStringA("fft_pcm_to_channels: unknown exception; outputs zeroed.\n");
        }
#endif
    }
}

static inline bool is_power_of_two(size_t v) {
    if (v == 0) return false;
    return (v & (v - 1)) == 0;
}

static inline size_t next_power_of_two(size_t v) {
    if (v <= 1) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if INTPTR_MAX == INT64_MAX
    v |= v >> 32;
#endif
    v++;
    return v;
}

void* fft_init(size_t n, int threads, int lanes, int inverse, int kernel, int radix, const int* radix_pattern, size_t radix_pattern_len, int pad_mode) {
    // legacy init forwards to extended initializer with default window/hop/mode
    return fft_init_ex(n, threads, lanes, inverse, kernel, radix, radix_pattern, radix_pattern_len, pad_mode, 0, 0, 0);
}

void* fft_init_ex(size_t n, int threads, int lanes, int inverse, int kernel, int radix, const int* radix_pattern, size_t radix_pattern_len, int pad_mode, int window, int hop, int stft_mode) {
    try {
    // Legacy initializer: do NOT install the crash handler automatically here.
    // Prefer callers use `fft_init_full(...)` to configure runtime crash
    // handler behavior (silent/write_files) before installation. Installing
    // unconditionally here could produce noisy diagnostics before tests
    // get a chance to opt-out via the API.
    // fftfree::install_crash_handler();
        if (n == 0) return nullptr;
        auto* ctx = new FftContext();
        ctx->N = static_cast<int>(n);
        ctx->inverse = (inverse != 0);
        ctx->kernel = kernel;
        if (ctx->kernel == 0) {
            const int env_kernel = env_default_kernel_override();
            if (env_kernel != 0) {
                ctx->kernel = env_kernel;
            }
        }
        ctx->radix = radix;
        ctx->pad_mode = pad_mode;
        ctx->window = window;
        ctx->hop = hop;
        ctx->stft_mode = stft_mode;
        // copy supplied radix pattern (if any)
        if (radix_pattern && radix_pattern_len > 0) {
            ctx->radix_pattern.assign(radix_pattern, radix_pattern + radix_pattern_len);
        }
        ctx->cfg.threads = (threads > 0) ? threads : eigfft::Plan<float>::Limits::kDefaultRuntimeThreads;
        ctx->cfg.lanes = (lanes > 0) ? lanes : 0; // 0 => auto inside plan
        ctx->cfg.radix = ctx->radix;
        // Propagate any supplied pattern into runtime config so PlanEnvironment
        // can copy it into the created Plan before allocation.
        if (!ctx->radix_pattern.empty()) ctx->cfg.radix_pattern = ctx->radix_pattern;
        ctx->cfg.inverse = ctx->inverse;
    // Decide effective radix and whether the plan requires a power-of-two size.
    // Priority: explicit caller radix -> algorithm default radix (if kernel specified) -> fallback to 2.
    int effective_radix = ctx->radix;
    if (effective_radix == 0 && ctx->kernel != 0) {
        try {
            eigfft::PlanRuntimeConfig probe_cfg = ctx->cfg;
            probe_cfg.threads = 1;
            probe_cfg.lanes = 1;
            probe_cfg.radix = 0;
            eigfft::PlanEnvironment<float> probe_env;
            probe_env.initialize(2, ctx->inverse, probe_cfg);
            auto& probe_plan = probe_env.plan();
            using BR = eigfft::ButterflyRadix;
            BR br = BR::Radix2;
            if (ctx->kernel == 1) br = probe_plan.butterfly_default_cooleytukey.radix;
            else if (ctx->kernel == 2) br = probe_plan.butterfly_default_stockham.radix;
            switch (br) {
              case BR::Radix2: effective_radix = 2; break;
              case BR::Radix4: effective_radix = 4; break;
              case BR::Radix8: effective_radix = 8; break;
              case BR::Radix16: effective_radix = 16; break;
              default: effective_radix = 2; break;
            }
        } catch (...) {
            effective_radix = 2;
        }
    }

    if (effective_radix == 0) effective_radix = 2;

    ctx->cfg.radix = effective_radix;
    if (!ctx->radix_pattern.empty()) ctx->cfg.radix_pattern = ctx->radix_pattern;

    const bool requires_pow2 = (effective_radix >= 2);

    size_t plan_n = static_cast<size_t>(ctx->N);
    if (requires_pow2) {
        const bool is_pow2 = is_power_of_two(plan_n);
        if (!is_pow2) {
            if (pad_mode == 1 /*always*/) {
                plan_n = next_power_of_two(plan_n);
            } else {
                size_t attempted = next_power_of_two(plan_n);
                fprintf(stderr, "fft_init: refusing to auto-pad N=%d -> %zu because pad_mode!=ALWAYS; set pad_mode=1 to permit padding or supply power-of-two N\n", ctx->N, attempted);
#if defined(_WIN32)
                if (!fftfree::crash_handler_silent_enabled()) {
                    OutputDebugStringA("fft_init: refused to auto-pad; pad_mode not ALWAYS\n");
                }
#endif
                delete ctx;
                return nullptr;
            }
        }
    } else if (pad_mode == 1 /*always*/) {
        plan_n = next_power_of_two(plan_n);
    }
    ctx->N = static_cast<int>(plan_n);
    {
        try {
            auto tok = ctx->cache.get_plan(ctx->N, ctx->inverse, ctx->cfg);
            (void)tok;
        } catch (const std::exception& ex) {
            fprintf(stderr, "fft_init: exception while creating plan for N=%d: %s\n", ctx->N, ex.what());
#if defined(_WIN32)
            if (!fftfree::crash_handler_silent_enabled()) {
                OutputDebugStringA("fft_init: exception while creating plan (see stderr)\n");
            }
#endif
            delete ctx;
            return nullptr;
        }
    }
    return static_cast<void*>(ctx);
    } catch (...) {
        fprintf(stderr, "fft_init: unknown exception caught, returning NULL\n");
#if defined(_WIN32)
        if (!fftfree::crash_handler_silent_enabled()) {
            OutputDebugStringA("fft_init: unknown exception caught\n");
        }
#endif
        return nullptr;
    }
}

int fft_execute(void* handle,
                const float* in_pcm,
                float* out_real,
                float* out_imag,
                float* out_mag,
                size_t n) {
    if (!handle || !in_pcm || !out_real || !out_imag || !out_mag || n == 0) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    if (ctx->pad_mode == 2 /*never*/ && static_cast<int>(n) != ctx->N) return 0;
    if (n > static_cast<size_t>(ctx->N)) return 0; // cannot exceed plan size
    try {
        using Complex = std::complex<float>;
        const size_t plan_n = static_cast<size_t>(ctx->N);
        std::vector<Complex> input(plan_n);
        // Copy provided samples; zero-pad remainder when allowed
        size_t i = 0;
        for (; i < n; ++i) input[i] = Complex(in_pcm[i], 0.0f);
        for (; i < plan_n; ++i) input[i] = Complex(0.0f, 0.0f);
        // Acquire a plan instance from cache
        auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, ctx->cfg);
        auto& plan = token.plan();
        // Select kernel if requested
        if (ctx->kernel == 1) {
            plan.use_kernel(eigfft::KernelKind::CooleyTukey);
        } else if (ctx->kernel == 2) {
            plan.use_kernel(eigfft::KernelKind::Stockham);
        }
        Eigen::Map<Eigen::Matrix<Complex, Eigen::Dynamic, 1>> data(input.data(), static_cast<Eigen::Index>(plan_n));
        eigfft::fft_inplace_batched<float>(data, plan);
        for (size_t j = 0; j < plan_n; ++j) {
            const auto v = data(static_cast<Eigen::Index>(j));
            out_real[j] = v.real();
            out_imag[j] = v.imag();
            out_mag[j] = std::abs(v);
        }
        return 1;
    } catch (...) {
        return 0;
    }
}

size_t fft_execute_batched(void* handle,
                           const float* pcm,
                           size_t pcm_len,
                           float* out_real,
                           float* out_imag,
                           float* out_mag,
                           int pad_mode,
                           int enable_backup,
                           size_t max_frames) {
    if (!handle || !pcm || pcm_len == 0 || !out_real || !out_imag || !out_mag) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    const int N = ctx->N;
    if (N <= 0) return 0;
    // Determine analysis window and hop
    int W = ctx->window ? ctx->window : N;
    int H = ctx->hop ? ctx->hop : W;
    if (W <= 0 || H <= 0) return 0;
    if (W > N) {
        // Cannot analyze windows larger than plan size.
        return 0;
    }

    // Determine effective pad policy: function param 0 => defer to ctx
    int effective_pad = (pad_mode == 0) ? ctx->pad_mode : pad_mode;

    // Compute number of frames
    size_t frames = 0;
    if (pcm_len < static_cast<size_t>(W)) {
        if (effective_pad == 1) frames = 1; else return 0;
    } else {
        // at least one full frame
        frames = 1 + (pcm_len - static_cast<size_t>(W)) / static_cast<size_t>(H);
        // check for a trailing partial frame
        size_t last_start = (frames - 1) * static_cast<size_t>(H);
        size_t remaining = (pcm_len > last_start) ? (pcm_len - last_start) : 0;
        if (remaining < static_cast<size_t>(W)) {
            if (effective_pad == 1) {
                // keep partial and pad
            } else if (effective_pad == 2) {
                // drop partial
                if (frames > 0) frames -= 1;
            } else {
                // pad_mode==0 treated as refuse
                return 0;
            }
        }
    }

    if (frames == 0) return 0;
    if (max_frames != 0 && frames > max_frames) frames = max_frames;

    const size_t plan_n = static_cast<size_t>(N);

    // Chunking: avoid huge temporary allocations. Aim for <= 64MB buffer.
    const size_t max_bytes = 64ULL * 1024ULL * 1024ULL;
    const size_t bytes_per_frame = plan_n * sizeof(std::complex<float>);
    size_t batch_frames = std::max((size_t)1, static_cast<size_t>(max_bytes / (bytes_per_frame + 1)));
    if (batch_frames > frames) batch_frames = frames;

    size_t produced = 0;
    try {
        for (size_t bstart = 0; bstart < frames; bstart += batch_frames) {
            size_t bcount = std::min(batch_frames, frames - bstart);
            std::vector<std::complex<float>> buffer(plan_n * bcount);
            // Fill columns (column-major: contiguous columns of length N)
            for (size_t f = 0; f < bcount; ++f) {
                size_t frame_idx = bstart + f;
                size_t start = frame_idx * static_cast<size_t>(H);
                size_t copy_count = 0;
                if (start < pcm_len) copy_count = std::min(static_cast<size_t>(W), pcm_len - start);
                // Copy samples (apply analysis window if enabled)
                size_t col_base = f * plan_n;
                if (ctx->windows_enabled && !ctx->analysis_win.empty()) {
                    for (size_t i = 0; i < copy_count; ++i) {
                        float s = pcm[start + i] * ctx->analysis_win[i];
                        buffer[col_base + i] = std::complex<float>(s, 0.0f);
                    }
                } else {
                    for (size_t i = 0; i < copy_count; ++i) {
                        buffer[col_base + i] = std::complex<float>(pcm[start + i], 0.0f);
                    }
                }
                // Zero-pad remainder up to W
                for (size_t i = copy_count; i < static_cast<size_t>(W); ++i) {
                    buffer[col_base + i] = std::complex<float>(0.0f, 0.0f);
                }
                // Zero-pad up to plan_n
                for (size_t i = static_cast<size_t>(W); i < plan_n; ++i) {
                    buffer[col_base + i] = std::complex<float>(0.0f, 0.0f);
                }
            }

            // Build a plan runtime config for worker tokens
            eigfft::PlanRuntimeConfig local_cfg = compute_effective_runtime(*ctx);

            // Shared-plan fast path via plan dispatcher.
            auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, local_cfg);
            auto& plan = token.plan();
            if (ctx->kernel == 1) plan.use_kernel(eigfft::KernelKind::CooleyTukey);
            else if (ctx->kernel == 2) plan.use_kernel(eigfft::KernelKind::Stockham);

            PoolDispatcher dispatcher;
            dispatcher.pool = ctx->pool ? ctx->pool.get() : nullptr;
            plan.set_dispatcher(&dispatcher);

            // If recovery not requested, run shared-plan fast path
            if (!enable_backup || !ctx->pool || !ctx->allow_outer_parallel) {
                Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> sub(buffer.data(), static_cast<Eigen::Index>(plan_n), static_cast<Eigen::Index>(bcount));
                eigfft::fft_inplace_batched<float>(sub, plan);
                plan.set_dispatcher(nullptr);
                // Copy outputs after fast path
                {
                    const size_t bins_out = (ctx->cfg.half_spectrum ? static_cast<size_t>(plan_n / 2 + 1) : static_cast<size_t>(plan_n));
                    for (size_t f = 0; f < bcount; ++f) {
                        size_t frame_idx = bstart + f;
                        size_t out_base = frame_idx * bins_out;
                        size_t col_base = f * plan_n;
                        for (size_t i = 0; i < bins_out; ++i) {
                            const auto& v = buffer[col_base + i];
                            out_real[out_base + i] = v.real();
                            out_imag[out_base + i] = v.imag();
                            out_mag[out_base + i] = ctx->store_polar ? v.real() : std::abs(v);
                        }
                    }
                }
            } else {
                // Recovery path: capture column slices per chunk so retries can
                // restore only the affected frames instead of cloning the entire batch.
                // Worker function: each worker acquires its own plan token and processes a
                // contiguous range of columns [start,end). Use inline dispatcher inside the
                // worker to avoid nested pool submission.
                auto worker_fn = [ctx, local_cfg, plan_n, &buffer](size_t start, size_t end, int worker_id) {
                    try {
                        auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, local_cfg);
                        auto& local_plan = token.plan();
                        // Force inline dispatcher for per-worker execution
                        local_plan.set_dispatcher(nullptr);
                        for (size_t f = start; f < end; ++f) {
                            size_t col_base = f * plan_n;
                            Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> sub(&buffer[col_base], static_cast<Eigen::Index>(plan_n), 1);
                            eigfft::fft_inplace_batched<float>(sub, local_plan);
                        }
                        local_plan.set_dispatcher(nullptr);
                    } catch (...) {
                        // swallow - restore will be attempted by the pool
                    }
                };

                // Coordinator: process ranges with the pool and, on persistent
                // failures, split failing ranges by half (binary halving) and
                // requeue the halves. This reduces wasted work when a subset of
                // frames repeatedly fails. We reuse the pool's restore callback
                // to detect retry attempts and persistent failures.
                const int kRetryCount = 2; // keep in sync with WorkerPool (EIGFFT_JOB_RETRY_COUNT)

                // Helper that processes a vector of ranges (each range is [first, second) of frames)
                auto process_ranges = [&](std::vector<std::pair<size_t,size_t>> ranges) {
                    // Full-batch backup for second-layer recovery
                    auto backup = std::make_shared<std::vector<std::complex<float>>>(buffer);
                    // Shared state for restore callback to count restore attempts per range
                    const size_t R = ranges.size();
                    std::unique_ptr<std::atomic<int>[]> attempt_count(new std::atomic<int>[R]{});
                    for (size_t i = 0; i < R; ++i) attempt_count[i].store(0, std::memory_order_relaxed);
                    // Optional: unique-claim debug checker (set by env FFTFREE_ASSERT_UNIQUE_CLAIMS)
                    const bool assert_unique = ([](){ const char* v = std::getenv("FFTFREE_ASSERT_UNIQUE_CLAIMS"); return v && *v && v[0] != '0'; })();
                    std::unique_ptr<std::atomic<int>[]> claim_count;
                    if (assert_unique) {
                        claim_count.reset(new std::atomic<int>[R]{});
                        for (size_t i = 0; i < R; ++i) claim_count[i].store(0, std::memory_order_relaxed);
                    }
                    std::vector<std::pair<size_t,size_t>> persistent_failures;
                    std::mutex pf_m;

                    ColumnRangeRecoveryCtx recovery_ctx;
                    recovery_ctx.buffer = buffer.data();
                    recovery_ctx.rows = plan_n;
                    recovery_ctx.ranges = &ranges;
                    recovery_ctx.backup = backup.get();
                    const eigfft::RecoveryOps recovery_ops = make_column_range_recovery_ops();

                    // Worker: ranges are indexed by the start/end index into `ranges` vector
                    auto worker_for_ranges = [ctx, local_cfg, plan_n, &buffer, ranges, assert_unique, &claim_count](size_t idx, size_t idx_end, int worker_id) {
                        // Note: WorkerPool will call this with idx/idx_end referencing
                        // consecutive indices into the `ranges` vector. We treat each
                        // index as an independent subtask which itself may cover
                        // multiple frames.
                        try {
                            if (assert_unique && claim_count) {
                                for (size_t ri = idx; ri < idx_end; ++ri) {
                                    int prev = claim_count[ri].fetch_add(1, std::memory_order_acq_rel);
                                    if (prev != 0) {
                                        std::fprintf(stderr, "[assert] duplicate range claim detected: idx=%zu worker=%d\n", ri, worker_id);
                                    }
                                }
                            }
                            auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, local_cfg);
                            auto& local_plan = token.plan();
                            local_plan.set_dispatcher(nullptr);
                            for (size_t ri = idx; ri < idx_end; ++ri) {
                                const auto r = ranges[ri];
                                for (size_t f = r.first; f < r.second; ++f) {
                                    size_t col_base = f * plan_n;
                                    Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> sub(&buffer[col_base], static_cast<Eigen::Index>(plan_n), 1);
                                    eigfft::fft_inplace_batched<float>(sub, local_plan);
                                }
                            }
                            local_plan.set_dispatcher(nullptr);
                        } catch (...) {
                            // Swallow; restore will be attempted by pool's restore callback
                        }
                    };

                    // Restore callback: copy from backup and track attempts
                    auto restore_cb = [backup, plan_n, &buffer, &ranges, atp=attempt_count.get(), &persistent_failures, &pf_m, kRetryCount](size_t start_idx, size_t end_idx) noexcept {
                        try {
                            if (start_idx >= end_idx) return;
                            const size_t limit = std::min(end_idx, ranges.size());
                            for (size_t ri = start_idx; ri < limit; ++ri) {
                                const auto r = ranges[ri];
                                const size_t cols = static_cast<size_t>(r.second - r.first);
                                if (cols > 0 && backup && !backup->empty() && !buffer.empty()) {
                                    const size_t src_off = static_cast<size_t>(r.first) * plan_n;
                                    const size_t elem_count = cols * plan_n;
                                    std::copy_n(backup->data() + src_off, elem_count, buffer.data() + src_off);
                                    if (g_restore_observer) g_restore_observer(r.first, r.second);
                                }
                                const int prev = atp[ri].fetch_add(1, std::memory_order_acq_rel);
                                if (prev + 1 >= kRetryCount) {
                                    std::lock_guard<std::mutex> g(pf_m);
                                    persistent_failures.emplace_back(r);
                                }
                            }
                        } catch (...) {}
                    };

                    // Build a trivial index-to-range mapping by letting each element
                    // in the pool job correspond to one entry in `ranges`.
                    const size_t total = ranges.size();
                    if (total == 0) return persistent_failures; // nothing to do
                    // Publish to pool and wait
                    ctx->pool->parallel_for_with_restore_ex(total, 1,
                        // worker: idx -> idx+1 (one range per task)
                        [&](size_t start, size_t end, int worker_id) {
                            worker_for_ranges(start, end, worker_id);
                        },
                        // restore fn receives start/end as indices into `ranges`
                        restore_cb,
                        eigfft::RecoveryOps{},
                        nullptr);

                    return persistent_failures;
                };

                // Coordinator main loop: start with the full-batch range and iteratively
                // split persistent failures until none remain or they are single-frame.
                std::vector<std::pair<size_t,size_t>> cur_ranges;
                if (eigfft::workerpool_debug_dropout() > 0 || eigfft::workerpool_dropout_pattern_enabled() || eigfft::workerpool_dropout_single_index() >= 0) {
                    for (size_t f = 0; f < bcount; ++f) cur_ranges.emplace_back(f, f+1);
                } else {
                    cur_ranges.emplace_back(0, bcount);
                }
                bool abort_batch = false;
                while (!cur_ranges.empty()) {
                    auto persistent = process_ranges(cur_ranges);
                    if (persistent.empty()) break; // success

                    // Build next level of ranges by splitting each persistent failure
                    std::vector<std::pair<size_t,size_t>> next_ranges;
                    for (auto &r : persistent) {
                        const size_t len = r.second - r.first;
                        if (len <= 1) {
                            // Single-frame persistent failure: cannot split further.
                            abort_batch = true;
                            break;
                        }
                        const size_t mid = r.first + len / 2;
                        next_ranges.emplace_back(r.first, mid);
                        next_ranges.emplace_back(mid, r.second);
                    }
                    if (abort_batch) break;
                    cur_ranges.swap(next_ranges);
                }

                if (abort_batch) {
#if defined(EIGFFT_RESTORE_CONSOLE_LOGS)
                    std::fprintf(stderr, "[restore] coordinator aborting batch due to single-frame persistent failure\n");
#endif
                    // Let the caller see what was produced so far; fall through to copy outputs
                }

                    // Copy outputs after pool-completed processing. When half_spectrum is
                    // enabled the caller expects only the first (N/2+1) bins per frame
                    // to be written; otherwise write full N bins. Use `bins` as the
                    // per-frame output stride into the caller buffers.
                    const size_t bins = (ctx->cfg.half_spectrum ? static_cast<size_t>(plan_n / 2 + 1) : static_cast<size_t>(plan_n));
                    for (size_t f = 0; f < bcount; ++f) {
                        size_t frame_idx = bstart + f;
                        size_t out_base = frame_idx * bins;
                        size_t col_base = f * plan_n;
                        // Only write the caller-visible bins (first `bins` rows of the FFT column)
                        for (size_t i = 0; i < bins; ++i) {
                            const auto& v = buffer[col_base + i];
                            out_real[out_base + i] = v.real();
                            out_imag[out_base + i] = v.imag();
                            if (ctx->store_polar) {
                                out_mag[out_base + i] = v.real();
                            } else {
                                out_mag[out_base + i] = std::abs(v);
                            }
                        }
                    }
                // After recovery completes, copy outputs for this batch
                {
                    const size_t bins_out = (ctx->cfg.half_spectrum ? static_cast<size_t>(plan_n / 2 + 1) : static_cast<size_t>(plan_n));
                    for (size_t f = 0; f < bcount; ++f) {
                        size_t frame_idx = bstart + f;
                        size_t out_base = frame_idx * bins_out;
                        size_t col_base = f * plan_n;
                        for (size_t i = 0; i < bins_out; ++i) {
                            const auto& v = buffer[col_base + i];
                            out_real[out_base + i] = v.real();
                            out_imag[out_base + i] = v.imag();
                            out_mag[out_base + i] = ctx->store_polar ? v.real() : std::abs(v);
                        }
                    }
                }
            }
            produced += bcount;
        }
        return produced;
    } catch (...) {
        return produced;
    }
}

// Execute batched inverse from complex inputs (or complex-format inputs for
// C2R/C2C). Accepts separate real/imag arrays laid out frame-major where each
// frame contains `bins` complex entries; `bins` is implicitly determined by
// the context: bins = N (full) or (N/2 + 1) when half_spectrum is enabled.
// The function reconstructs per-frame column-major buffers and runs the plan
// with the ctx settings (ctx->inverse indicates inverse transforms).
size_t fft_execute_complex_batched(void* handle,
                                   const float* in_real,
                                   const float* in_imag,
                                   size_t frames,
                                   float* out_pcm,
                                   int pad_mode,
                                   int enable_backup,
                                   size_t max_frames) {
    if (!handle || !in_real || !in_imag || !out_pcm || frames == 0) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    const int N = ctx->N;
    if (N <= 0) return 0;

    // Determine expected bins per frame for the provided complex input
    const size_t bins = (ctx->cfg.half_spectrum ? static_cast<size_t>(N / 2 + 1) : static_cast<size_t>(N));
    if (frames == 0) return 0;
    if (max_frames != 0 && frames > max_frames) frames = max_frames;

    try {
        const size_t plan_n = static_cast<size_t>(N);
        // Build a column-major buffer (plan_n rows x frames cols)
        std::vector<std::complex<float>> buffer(plan_n * frames);
        for (size_t f = 0; f < frames; ++f) {
            const size_t in_base = f * bins;
            const size_t col_base = f * plan_n;
            // Copy provided lower half (0..N/2)
            for (size_t i = 0; i < bins; ++i) {
                buffer[col_base + i] = std::complex<float>(in_real[in_base + i], in_imag[in_base + i]);
            }
            // If half-spectrum and C2R, rebuild conjugate symmetry; else zero the remainder
            if (ctx->cfg.half_spectrum && ctx->cfg.transform == FFT_TRANSFORM_C2R) {
                // DC and Nyquist bins must be purely real for perfect symmetry
                buffer[col_base + 0].imag(0.0f);
                if (plan_n % 2 == 0) {
                    size_t ny = plan_n / 2;
                    if (ny < bins) buffer[col_base + ny].imag(0.0f);
                }
                // Fill upper half: k=1..N/2-1 -> N-k
                size_t nyq = plan_n / 2;
                size_t maxk = (bins > nyq + 1) ? nyq : (bins - 1);
                for (size_t k = 1; k < maxk; ++k) {
                    std::complex<float> v = buffer[col_base + k];
                    buffer[col_base + (plan_n - k)] = std::conj(v);
                }
                // Any remaining rows (if bins < N/2+1) set to zero
                for (size_t i = bins; i < plan_n; ++i) {
                    // Avoid overwriting conjugate-filled region
                    // Only zero those not set above
                    // Here we conservatively skip; zeroing is harmless where already set
                    // but keep for completeness
                    // buffer[col_base + i] = buffer[col_base + i];
                }
            } else {
                for (size_t i = bins; i < plan_n; ++i) buffer[col_base + i] = std::complex<float>(0.0f, 0.0f);
            }
        }

        eigfft::PlanRuntimeConfig local_cfg = compute_effective_runtime(*ctx);

        // Try shared-plan fast path first (only when recovery not requested)
        bool __fast_done = false;
        if (!enable_backup || !ctx->pool || !ctx->allow_outer_parallel) {
            try {
            auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, local_cfg);
            auto& plan = token.plan();
            if (ctx->kernel == 1) plan.use_kernel(eigfft::KernelKind::CooleyTukey);
            else if (ctx->kernel == 2) plan.use_kernel(eigfft::KernelKind::Stockham);

            PoolDispatcher dispatcher;
            dispatcher.pool = ctx->pool ? ctx->pool.get() : nullptr;
            plan.set_dispatcher(&dispatcher);

            Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> sub(buffer.data(), static_cast<Eigen::Index>(plan_n), static_cast<Eigen::Index>(frames));
            eigfft::fft_inplace_batched<float>(sub, plan);
            plan.set_dispatcher(nullptr);

            // Copy outputs: optionally apply synthesis window and OLA
            if (ctx->apply_ola) {
                const int H = (ctx->hop > 0) ? ctx->hop : ((ctx->window > 0) ? ctx->window : ctx->N);
                const size_t out_len = (frames == 0) ? 0 : ((frames - 1) * (size_t)H + (size_t)plan_n);
                // Zero destination (assume caller provided at least frames*N elements)
                for (size_t t = 0; t < frames * plan_n; ++t) out_pcm[t] = 0.0f;
                std::vector<float> norm;
                if (ctx->cola_mode == FFT_COLA_NORMALIZE) norm.assign(frames * plan_n, 0.0f);
                for (size_t f = 0; f < frames; ++f) {
                    const size_t col_base = f * plan_n;
                    const size_t start = f * (size_t)H;
                    for (size_t i = 0; i < plan_n; ++i) {
                        float s = buffer[col_base + i].real();
                        if (ctx->windows_enabled && !ctx->synthesis_win.empty()) s *= ctx->synthesis_win[i];
                        const size_t t = start + i;
                        if (t < frames * plan_n) {
                            out_pcm[t] += s;
                            // COLA normalization should accumulate squared window (or product of analysis*synthesis).
                            if (!norm.empty()) {
                                if (ctx->windows_enabled && !ctx->synthesis_win.empty()) {
                                    float w = ctx->synthesis_win[i];
                                    norm[t] += w * w;
                                } else {
                                    norm[t] += 1.0f;
                                }
                            }
                        }
                    }
                }
                if (!norm.empty()) {
                    for (size_t t = 0; t < frames * plan_n; ++t) {
                        if (norm[t] > 1e-12f) out_pcm[t] /= norm[t];
                    }
                }
            } else {
                for (size_t f = 0; f < frames; ++f) {
                    const size_t out_base = f * plan_n;
                    const size_t col_base = f * plan_n;
                    for (size_t i = 0; i < plan_n; ++i) {
                        float s = buffer[col_base + i].real();
                        if (ctx->windows_enabled && !ctx->synthesis_win.empty()) s *= ctx->synthesis_win[i];
                        out_pcm[out_base + i] = s;
                    }
                }
            }
            __fast_done = true;
            } catch (...) {
                // fall back to recovery path below when fast path fails
            }
        }
        if (__fast_done) {
            return frames;
        }

        // Recovery path when enabled
        if (enable_backup && ctx->pool && ctx->allow_outer_parallel) {
            // Recovery path: capture per-range column slices so retries only
            // rewrite the affected frames instead of cloning the full buffer.
            // Worker: get a plan token and process columns [start,end)
            auto worker_fn = [ctx, local_cfg, plan_n, &buffer](size_t start, size_t end, int worker_id) {
                try {
                    auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, local_cfg);
                    auto& local_plan = token.plan();
                    local_plan.set_dispatcher(nullptr);
                    for (size_t f = start; f < end; ++f) {
                        size_t col_base = f * plan_n;
                        Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> sub(&buffer[col_base], static_cast<Eigen::Index>(plan_n), 1);
                        eigfft::fft_inplace_batched<float>(sub, local_plan);
                    }
                    local_plan.set_dispatcher(nullptr);
                } catch (...) {
                }
            };

            // Coordinator for complex-batched path: same strategy as above.
            const int kRetryCount_complex = 2;
            auto backup = std::make_shared<std::vector<std::complex<float>>>(buffer);
            auto process_ranges_complex = [&](std::vector<std::pair<size_t,size_t>> ranges) {
                const size_t R = ranges.size();
                std::unique_ptr<std::atomic<int>[]> attempt_count(new std::atomic<int>[R]{});
                for (size_t i = 0; i < R; ++i) attempt_count[i].store(0, std::memory_order_relaxed);
                const bool assert_unique = ([](){ const char* v = std::getenv("FFTFREE_ASSERT_UNIQUE_CLAIMS"); return v && *v && v[0] != '0'; })();
                std::unique_ptr<std::atomic<int>[]> claim_count;
                if (assert_unique) {
                    claim_count.reset(new std::atomic<int>[R]{});
                    for (size_t i = 0; i < R; ++i) claim_count[i].store(0, std::memory_order_relaxed);
                }
                std::vector<std::pair<size_t,size_t>> persistent_failures;
                std::mutex pf_m;

                ColumnRangeRecoveryCtx recovery_ctx;
                recovery_ctx.buffer = buffer.data();
                recovery_ctx.rows = plan_n;
                recovery_ctx.ranges = &ranges;
                recovery_ctx.backup = backup.get();
                const eigfft::RecoveryOps recovery_ops = make_column_range_recovery_ops();

                auto worker_for_ranges = [ctx, local_cfg, plan_n, &buffer, ranges, assert_unique, &claim_count](size_t idx, size_t idx_end, int worker_id) {
                    try {
                        if (assert_unique && claim_count) {
                            for (size_t ri = idx; ri < idx_end; ++ri) {
                                int prev = claim_count[ri].fetch_add(1, std::memory_order_acq_rel);
                                if (prev != 0) {
                                    std::fprintf(stderr, "[assert] duplicate complex-range claim detected: idx=%zu worker=%d\n", ri, worker_id);
                                }
                            }
                        }
                        auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, local_cfg);
                        auto& local_plan = token.plan();
                        local_plan.set_dispatcher(nullptr);
                        for (size_t ri = idx; ri < idx_end; ++ri) {
                            const auto r = ranges[ri];
                            for (size_t f = r.first; f < r.second; ++f) {
                                size_t col_base = f * plan_n;
                                Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> sub(&buffer[col_base], static_cast<Eigen::Index>(plan_n), 1);
                                eigfft::fft_inplace_batched<float>(sub, local_plan);
                            }
                        }
                        local_plan.set_dispatcher(nullptr);
                    } catch (...) {}
                };

                auto restore_cb_complex = [backup, plan_n, &buffer, &ranges, atp=attempt_count.get(), &persistent_failures, &pf_m, kRetryCount_complex](size_t start_idx, size_t end_idx) noexcept {
                    try {
                        if (start_idx >= end_idx) return;
                        const size_t limit = std::min(end_idx, ranges.size());
                        for (size_t ri = start_idx; ri < limit; ++ri) {
                            const auto r = ranges[ri];
                            const size_t cols = static_cast<size_t>(r.second - r.first);
                            if (cols > 0 && backup && !backup->empty() && !buffer.empty()) {
                                const size_t src_off = static_cast<size_t>(r.first) * plan_n;
                                const size_t elem_count = cols * plan_n;
                                std::copy_n(backup->data() + src_off, elem_count, buffer.data() + src_off);
                                if (g_restore_observer) g_restore_observer(r.first, r.second);
                            }
                            const int prev = atp[ri].fetch_add(1, std::memory_order_acq_rel);
                            if (prev + 1 >= kRetryCount_complex) {
                                std::lock_guard<std::mutex> g(pf_m);
                                persistent_failures.emplace_back(r);
                            }
                        }
                    } catch (...) {}
                };

                const size_t total = ranges.size();
                if (total == 0) return persistent_failures;
                ctx->pool->parallel_for_with_restore_ex(total, 1,
                    [&](size_t start, size_t end, int worker_id) { worker_for_ranges(start, end, worker_id); },
                    restore_cb_complex,
                    eigfft::RecoveryOps{},
                    nullptr);
                return persistent_failures;
            };

            std::vector<std::pair<size_t,size_t>> cur_ranges;
            if (eigfft::workerpool_debug_dropout() > 0 || eigfft::workerpool_dropout_pattern_enabled() || eigfft::workerpool_dropout_single_index() >= 0) {
                for (size_t f = 0; f < frames; ++f) cur_ranges.emplace_back(f, f+1);
            } else {
                cur_ranges.emplace_back(0, frames);
            }
            bool abort_batch = false;
            while (!cur_ranges.empty()) {
                auto persistent = process_ranges_complex(cur_ranges);
                if (persistent.empty()) break;
                std::vector<std::pair<size_t,size_t>> next_ranges;
                for (auto &r : persistent) {
                    const size_t len = r.second - r.first;
                    if (len <= 1) { abort_batch = true; break; }
                    const size_t mid = r.first + len / 2;
                    next_ranges.emplace_back(r.first, mid);
                    next_ranges.emplace_back(mid, r.second);
                }
                if (abort_batch) break;
                cur_ranges.swap(next_ranges);
            }
            if (abort_batch) {
#if defined(EIGFFT_RESTORE_CONSOLE_LOGS)
                std::fprintf(stderr, "[restore] coordinator aborting complex batch due to single-frame persistent failure\n");
#endif
            }

            // Copy outputs
            if (ctx->apply_ola) {
                const int H = (ctx->hop > 0) ? ctx->hop : ((ctx->window > 0) ? ctx->window : ctx->N);
                const size_t max_len = frames * plan_n;
                for (size_t t = 0; t < max_len; ++t) out_pcm[t] = 0.0f;
                std::vector<float> norm;
                if (ctx->cola_mode == FFT_COLA_NORMALIZE) norm.assign(max_len, 0.0f);
                for (size_t f = 0; f < frames; ++f) {
                    const size_t col_base = f * plan_n;
                    const size_t start = f * static_cast<size_t>(H);
                    for (size_t i = 0; i < plan_n; ++i) {
                        float s = buffer[col_base + i].real();
                        if (ctx->windows_enabled && !ctx->synthesis_win.empty()) s *= ctx->synthesis_win[i];
                        const size_t t = start + i;
                        if (t < max_len) {
                            out_pcm[t] += s;
                            if (!norm.empty()) {
                                if (ctx->windows_enabled && !ctx->synthesis_win.empty()) {
                                    float w = ctx->synthesis_win[i];
                                    norm[t] += w * w;
                                } else {
                                    norm[t] += 1.0f;
                                }
                            }
                        }
                    }
                }
                if (!norm.empty()) {
                    for (size_t t = 0; t < max_len; ++t) {
                        if (norm[t] > 1e-12f) out_pcm[t] /= norm[t];
                    }
                }
            } else {
                for (size_t f = 0; f < frames; ++f) {
                    const size_t out_base = f * plan_n;
                    const size_t col_base = f * plan_n;
                    for (size_t i = 0; i < plan_n; ++i) {
                        float s = buffer[col_base + i].real();
                        if (ctx->windows_enabled && !ctx->synthesis_win.empty()) s *= ctx->synthesis_win[i];
                        out_pcm[out_base + i] = s;
                    }
                }
            }
            return frames;
        }

        // (removed duplicate non-recovery path)
        return 0;
    } catch (...) {
        return 0;
    }
}

size_t fft_ctx_size(void* handle) {
    if (!handle) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    return static_cast<size_t>(ctx->N);
}

size_t fft_ctx_worker_threads(void* handle) {
    if (!handle) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    if (ctx->pool) {
        const int workers = ctx->pool->size();
        return static_cast<size_t>(workers > 0 ? workers : 1);
    }
    const int requested = ctx->cfg.threads;
    return static_cast<size_t>(requested > 0 ? requested : 1);
}

size_t fft_ctx_effective_threads(void* handle) {
    if (!handle) return 0;
    auto* ctx = static_cast<FftContext*>(handle);
    try {
        eigfft::PlanRuntimeConfig runtime = ctx->cfg;
        runtime.inverse = ctx->inverse;
        auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, runtime);
        const int effective = token.plan().effective_threads(ctx->N);
        return static_cast<size_t>(effective > 0 ? effective : 1);
    } catch (...) {
        return 0;
    }
}

void fft_free(void* handle) {
    if (!handle) return;
    FftContext* ctx = static_cast<FftContext*>(handle);
    delete ctx;
}

extern "C" {
    FFT_CFFI_API void fft_register_restore_observer(void (*cb)(size_t, size_t)) {
        if (cb) {
            g_restore_observer = [cb](size_t s, size_t e) { cb(s, e); };
        } else {
            g_restore_observer = nullptr;
        }
    }
}

// Original ABI: forwards to v2 with defaults
void* fft_init_full(size_t n,
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
                    int silent_crash_reports) {
    return fft_init_full_v2(n, threads, lanes, inverse, kernel, radix,
                            radix_pattern, radix_pattern_len, pad_mode,
                            window, hop, stft_mode, transform,
                            reduce_magnitude, store_polar, half_spectrum,
                            allow_outer_parallel, allow_inner_parallel,
                            inner_threads, save_crash_logs, silent_crash_reports,
                            /*apply_windows*/0, /*apply_ola*/0,
                            /*analysis_kind*/0, /*ap1*/0.0f, /*ap2*/0.0f,
                            /*synth_kind*/0, /*sp1*/0.0f, /*sp2*/0.0f,
                            /*norm*/0, /*cola*/0);
}

void* fft_init_full_v2(size_t n,
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
                    int cola_mode) {
    try {
        // Configure crash handler behavior from explicit arguments (do not rely on env)
        // Respect caller's silent preference first so the installer may be no-op.
        fftfree::set_crash_handler_silent(silent_crash_reports != 0);
        fftfree::set_crash_handler_write_files(save_crash_logs != 0);
        // Install the crash handler now that runtime flags are set.
        fftfree::install_crash_handler();
        auto* ctx = new FftContext();
        ctx->N = static_cast<int>(n);
        ctx->inverse = (inverse != 0);
        ctx->kernel = kernel;
        ctx->radix = radix;
        ctx->pad_mode = pad_mode;
        ctx->window = window;
        ctx->hop = hop;
        ctx->stft_mode = stft_mode;
        ctx->transform = transform;
        ctx->reduce_magnitude = reduce_magnitude;
        ctx->store_polar = store_polar;
        ctx->half_spectrum = half_spectrum;
        ctx->allow_outer_parallel = allow_outer_parallel;
        ctx->allow_inner_parallel = allow_inner_parallel;
        ctx->inner_threads = inner_threads;
        // Window/OLA
        ctx->analysis_win_kind = analysis_window_kind;
        ctx->synthesis_win_kind = synthesis_window_kind;
        ctx->win_param1 = analysis_param1;
        ctx->win_param2 = analysis_param2;
        ctx->win_norm_policy = window_norm_policy;
        ctx->cola_mode = cola_mode;
        // Flags
        // Enable only if explicitly requested; otherwise preserve old behavior.
        // apply_ola handled in inverse executor.
        ctx->windows_enabled = (apply_windows != 0);
        // Build windows sized to N, applying analysis window over first W samples.
        const int useW = (window > 0) ? window : ctx->N;
        {
            auto aw = build_window(analysis_window_kind, useW, analysis_param1, analysis_param2);
            normalize_window(aw, window_norm_policy);
            ctx->analysis_win.assign(ctx->N, 1.0f);
            for (int i = 0; i < useW && i < ctx->N; ++i) ctx->analysis_win[i] = aw[i];
        }
        {
            auto sw = build_window(synthesis_window_kind, useW, synthesis_param1, synthesis_param2);
            normalize_window(sw, window_norm_policy);
            ctx->synthesis_win.assign(ctx->N, 1.0f);
            for (int i = 0; i < useW && i < ctx->N; ++i) ctx->synthesis_win[i] = sw[i];
        }
        // Store OLA flag in pad_mode slot? Keep in new context fields instead.
        ctx->apply_ola = (apply_ola != 0);
        if (radix_pattern && radix_pattern_len > 0) {
            ctx->radix_pattern.assign(radix_pattern, radix_pattern + radix_pattern_len);
        }

        ctx->cfg.threads = (threads > 0) ? threads : 1; // outer pool threads
        ctx->cfg.lanes = (lanes > 0) ? lanes : 1;
        ctx->cfg.inverse = ctx->inverse;
        ctx->cfg.radix = ctx->radix;
        ctx->cfg.radix_pattern = ctx->radix_pattern;
        ctx->cfg.transform = ctx->transform;
        ctx->cfg.reduce_magnitude = (ctx->reduce_magnitude != 0);
        ctx->cfg.store_polar = (ctx->store_polar != 0);
        ctx->cfg.half_spectrum = (ctx->half_spectrum != 0);
        ctx->cfg.allow_outer_parallel = (ctx->allow_outer_parallel != 0);
        ctx->cfg.allow_inner_parallel = (ctx->allow_inner_parallel != 0);
        ctx->cfg.inner_threads = ctx->inner_threads;

        // Always create a worker pool for the CFFI path when outer parallelism
        // is allowed, even when threads==1, so work executes on worker threads.
        if (ctx->allow_outer_parallel != 0) {
            const int pool_threads = std::max(1, ctx->cfg.threads);
            ctx->pool = std::make_unique<eigfft::WorkerPool>(pool_threads);
        }

        // Diagnostic: print config and pool size before warming so we can capture
        // any exceptions that occur during Plan allocation/warm.
        {
            int pool_sz = ctx->pool ? ctx->pool->size() : 0;
            fprintf(stderr, "[diag] fft_init_full: N=%d threads=%d lanes=%d allow_outer=%d allow_inner=%d inner_threads=%d pool_sz=%d\n",
                    ctx->N, ctx->cfg.threads, ctx->cfg.lanes, ctx->cfg.allow_outer_parallel ? 1 : 0,
                    ctx->cfg.allow_inner_parallel ? 1 : 0, ctx->cfg.inner_threads, pool_sz);
#if defined(_WIN32)
            if (!fftfree::crash_handler_silent_enabled()) {
                OutputDebugStringA("[diag] fft_init_full: warming plan (see stderr)\n");
            }
#endif
        }

        // Warm the cache with a plan instance. Wrap in a focused try/catch and
        // emit the exception message if allocation/construction fails.
        try {
            eigfft::PlanRuntimeConfig warm_cfg = compute_effective_runtime(*ctx);
            auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, warm_cfg);
            auto& plan = token.plan();
            if (ctx->kernel == 1) plan.use_kernel(eigfft::KernelKind::CooleyTukey);
            else if (ctx->kernel == 2) plan.use_kernel(eigfft::KernelKind::Stockham);
        } catch (const std::exception& ex) {
            fprintf(stderr, "[error] fft_init_full: exception while warming plan: %s\n", ex.what());
#if defined(_WIN32)
            if (!fftfree::crash_handler_silent_enabled()) {
                OutputDebugStringA("[error] fft_init_full: exception while warming plan\n");
            }
#endif
            delete ctx;
            return nullptr;
        } catch (...) {
            fprintf(stderr, "[error] fft_init_full: unknown exception while warming plan\n");
#if defined(_WIN32)
            if (!fftfree::crash_handler_silent_enabled()) {
                OutputDebugStringA("[error] fft_init_full: unknown exception while warming plan\n");
            }
#endif
            delete ctx;
            return nullptr;
        }
        return ctx;
    } catch (...) {
        return nullptr;
    }
}

// --- Windowing API --------------------------------------------

extern "C" {
int fft_config_window(void* handle,
                      int analysis_kind,
                      int synthesis_kind,
                      float param1,
                      float param2,
                      int norm_policy,
                      int cola_mode) {
    if (!handle) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    const int W = (ctx->window > 0) ? ctx->window : ctx->N;
    if (W <= 0) return 0;
    try {
        ctx->analysis_win_kind = analysis_kind;
        ctx->synthesis_win_kind = synthesis_kind;
        ctx->win_param1 = param1;
        ctx->win_param2 = param2;
        ctx->win_norm_policy = norm_policy;
        ctx->cola_mode = cola_mode;
        ctx->analysis_win = build_window(analysis_kind, W, param1, param2);
        ctx->synthesis_win = build_window(synthesis_kind, W, param1, param2);
        normalize_window(ctx->analysis_win, norm_policy);
        normalize_window(ctx->synthesis_win, norm_policy);
        // Do not enable internal application by default (preserve behavior)
        ctx->windows_enabled = false;
        return 1;
    } catch (...) { return 0; }
}

int fft_config_window_custom(void* handle,
                             const float* analysis_window,
                             size_t analysis_len,
                             const float* synthesis_window,
                             size_t synthesis_len,
                             int norm_policy,
                             int cola_mode) {
    if (!handle) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    const int W = (ctx->window > 0) ? ctx->window : ctx->N;
    if (W <= 0) return 0;
    try {
        if (analysis_window && analysis_len == (size_t)W) ctx->analysis_win.assign(analysis_window, analysis_window + analysis_len);
        if (synthesis_window && synthesis_len == (size_t)W) ctx->synthesis_win.assign(synthesis_window, synthesis_window + synthesis_len);
        ctx->win_norm_policy = norm_policy;
        ctx->cola_mode = cola_mode;
        normalize_window(ctx->analysis_win, norm_policy);
        normalize_window(ctx->synthesis_win, norm_policy);
        ctx->windows_enabled = false;
        return 1;
    } catch (...) { return 0; }
}
}
// Test-only helper: expose WorkerPool debug dropout control through the C ABI
// so callers that dynamically load this library can set the probability inside
// the library binary (not just their own TU copy of the header-only variable).
extern "C" FFT_CFFI_API void fft_set_workerpool_debug_dropout(int pct) {
    eigfft::set_workerpool_debug_dropout(pct);
}

extern "C" FFT_CFFI_API void fft_set_workerpool_dropout_pattern(int period, int on_len, int phase) {
    eigfft::set_workerpool_dropout_pattern(period, on_len, phase);
}

extern "C" FFT_CFFI_API void fft_set_workerpool_dropout_single(long long index) {
    eigfft::set_workerpool_dropout_single(index);
}

extern "C" FFT_CFFI_API void fft_set_workerpool_dropout_persistent(int on) {
    eigfft::set_workerpool_dropout_persistent(on != 0);
}

extern "C" FFT_CFFI_API void fft_clear_workerpool_dropout_history() {
    eigfft::clear_workerpool_dropout_history();
}

// ---------------------- Phase inference C API ----------------------
namespace {
struct PhaseCtx {
    phaseinfer::PhaseConfig cfg;
    eigfft::PlanCache<float> cache;
    eigfft::PlanRuntimeConfig plan_cfg{}; // single-threaded FFTs for inference
};
}
extern "C" FFT_CFFI_API void* phase_init(int N, int hop, int half_spectrum, int mode, int iterations) {
    try {
        if (N <= 0) return nullptr;
        PhaseCtx* c = new PhaseCtx();
        c->cfg.N = N;
        c->cfg.hop = (hop > 0) ? hop : N;
        c->cfg.half_spectrum = (half_spectrum != 0);
        c->cfg.mode = mode;
        c->cfg.iterations = iterations;
        // conservative runtime config
        c->plan_cfg.threads = 1;
        c->plan_cfg.lanes = 1;
        c->plan_cfg.allow_inner_parallel = false;
        c->plan_cfg.inner_threads = 0;
        return static_cast<void*>(c);
    } catch (...) { return nullptr; }
}

extern "C" FFT_CFFI_API size_t phase_infer_execute(void* handle,
                                          const float* in_mag,
                                          size_t frames,
                                          float* out_real,
                                          float* out_imag) {
    if (!handle || !in_mag || !out_real || !out_imag || frames == 0) return 0;
    PhaseCtx* c = static_cast<PhaseCtx*>(handle);
    try {
        if (c->cfg.mode == 0) {
            phaseinfer::infer_linear(in_mag, frames, c->cfg, out_real, out_imag);
            return frames;
        }
        if (c->cfg.mode == 1) {
            // Minimum-phase per-frame reconstruction using cepstrum.
            const int N = c->cfg.N;
            const int bins = c->cfg.half_spectrum ? (N/2 + 1) : N;
            if (N <= 0 || bins <= 0) return 0;
            auto inv_token = c->cache.get_plan(N, /*inverse=*/true, c->plan_cfg);
            auto fwd_token = c->cache.get_plan(N, /*inverse=*/false, c->plan_cfg);
            auto& inv_plan = inv_token.plan();
            auto& fwd_plan = fwd_token.plan();
            std::vector<std::complex<float>> spec(static_cast<size_t>(N));
            std::vector<std::complex<float>> time(static_cast<size_t>(N));
            const float eps = 1e-20f;
            for (size_t f = 0; f < frames; ++f) {
                const size_t base = f * static_cast<size_t>(bins);
                // Build real-even log-amplitude spectrum
                auto set_log = [&](int k, float magk){
                    const float v = std::log(std::max(eps, magk));
                    spec[static_cast<size_t>(k)] = std::complex<float>(v, 0.0f);
                };
                // DC
                set_log(0, in_mag[base + 0]);
                if (c->cfg.half_spectrum) {
                    for (int k = 1; k < (N/2); ++k) {
                        const float m = in_mag[base + static_cast<size_t>(k)];
                        set_log(k, m);
                        spec[static_cast<size_t>(N - k)] = spec[static_cast<size_t>(k)];
                    }
                    if ((N % 2) == 0) {
                        set_log(N/2, in_mag[base + static_cast<size_t>(N/2)]);
                    }
                } else {
                    for (int k = 1; k < N; ++k) {
                        set_log(k, in_mag[base + static_cast<size_t>(k)]);
                    }
                }
                // IFFT to real cepstrum
                {
                    Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1>> v_in(spec.data(), static_cast<Eigen::Index>(N));
                    Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1> tmp = v_in;
                    Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat(tmp.data(), static_cast<Eigen::Index>(N), 1);
                    eigfft::fft_inplace_batched<float>(mat, inv_plan);
                    for (int i = 0; i < N; ++i) time[static_cast<size_t>(i)] = tmp(i);
                }
                // Causal cepstrum shaping for minimum-phase
                if (N > 0) {
                    for (int n = 1; n < (N+1)/2; ++n) time[static_cast<size_t>(n)] *= 2.0f;
                    // keep Nyquist (if even N)
                    for (int n = (N/2)+1; n < N; ++n) time[static_cast<size_t>(n)] = std::complex<float>(0.0f, 0.0f);
                }
                // FFT to complex log spectrum
                {
                    Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1>> v_in(time.data(), static_cast<Eigen::Index>(N));
                    Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1> tmp = v_in;
                    Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat(tmp.data(), static_cast<Eigen::Index>(N), 1);
                    eigfft::fft_inplace_batched<float>(mat, fwd_plan);
                    for (int i = 0; i < N; ++i) spec[static_cast<size_t>(i)] = tmp(i);
                }
                // Exponentiate to get minimum-phase complex spectrum; write requested bins
                for (int k = 0; k < bins; ++k) {
                    const std::complex<float> L = spec[static_cast<size_t>(k)];
                    const float a = std::exp(L.real());
                    const float ph = L.imag();
                    out_real[base + static_cast<size_t>(k)] = a * std::cos(ph);
                    out_imag[base + static_cast<size_t>(k)] = a * std::sin(ph);
                }
            }
            return frames;
        }
        // Unknown mode
        return 0;
    } catch (...) {
        return 0;
    }
}

extern "C" FFT_CFFI_API void phase_free(void* handle) {
    if (!handle) return;
    PhaseCtx* c = static_cast<PhaseCtx*>(handle);
    delete c;
}

extern "C" FFT_CFFI_API int fft_griffin_lim(void* ctx_forward,
                                            void* ctx_inverse,
                                            const float* in_mag,
                                            size_t frames,
                                            int hop,
                                            int half_spectrum,
                                            int iterations,
                                            int pad_mode,
                                            unsigned int seed,
                                            float* out_real,
                                            float* out_imag) {
    if (!ctx_forward || !ctx_inverse || !in_mag || !out_real || !out_imag || frames == 0) {
        return 0;
    }
    if (in_mag == out_real || in_mag == out_imag || out_real == out_imag) {
#if EIGFFT_RUNTIME_INSTRUMENTATION
        std::fprintf(stderr,
                     "[diag] fft_griffin_lim rejected aliased buffers (mag=%p real=%p imag=%p)\n",
                     static_cast<const void*>(in_mag),
                     static_cast<const void*>(out_real),
                     static_cast<const void*>(out_imag));
#if defined(_WIN32)
        if (!fftfree::crash_handler_silent_enabled()) {
            std::string msg = "[diag] fft_griffin_lim rejected aliased buffers\n";
            OutputDebugStringA(msg.c_str());
        }
#endif
#endif
        return 0;
    }
    auto* fwd_ctx = static_cast<FftContext*>(ctx_forward);
    auto* inv_ctx = static_cast<FftContext*>(ctx_inverse);
    if (!fwd_ctx || !inv_ctx) {
        return 0;
    }

    phaseinfer::PhaseConfig cfg;
    cfg.N = (inv_ctx->N > 0) ? inv_ctx->N : fwd_ctx->N;
    if (cfg.N <= 0) {
        return 0;
    }
    cfg.half_spectrum = (half_spectrum != 0);
    if (cfg.half_spectrum != (inv_ctx->half_spectrum != 0) ||
        cfg.half_spectrum != (fwd_ctx->half_spectrum != 0)) {
        // Ensure expectations match contexts.
        return 0;
    }

    if (hop > 0) {
        cfg.hop = hop;
    } else if (inv_ctx->hop > 0) {
        cfg.hop = inv_ctx->hop;
    } else if (fwd_ctx->hop > 0) {
        cfg.hop = fwd_ctx->hop;
    } else {
        cfg.hop = cfg.N;
    }

    cfg.iterations = iterations;

    const bool ok = phaseinfer::infer_griffin_lim(
        in_mag,
        frames,
        cfg,
        ctx_forward,
        ctx_inverse,
        out_real,
        out_imag,
        iterations,
        pad_mode,
        seed);
    return ok ? 1 : 0;
}
}
