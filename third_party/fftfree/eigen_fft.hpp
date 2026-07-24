#ifndef EIGFFT_DEBUG
#define EIGFFT_DEBUG 0
#endif

#ifndef EIGFFT_TRACE_STOCKHAM
#define EIGFFT_TRACE_STOCKHAM 0
#endif

#ifndef EIGFFT_TRACE_LAYOUT
#define EIGFFT_TRACE_LAYOUT 0
#endif

#ifndef EIGFFT_TRACE_THREADS
#define EIGFFT_TRACE_THREADS 0
#endif

#ifndef FFTFREE_CT_FORCE_INLINE_PERMUTE
#define FFTFREE_CT_FORCE_INLINE_PERMUTE 0
#endif

#ifndef FFTFREE_CT_FORCE_INLINE_ALL_STAGES
#define FFTFREE_CT_FORCE_INLINE_ALL_STAGES 0
#endif

#ifndef FFTFREE_CT_FORCE_INLINE_STAGE
#define FFTFREE_CT_FORCE_INLINE_STAGE -1
#endif

#ifndef FFTFREE_CT_FORCE_INLINE_KEEP_DISPATCH
#define FFTFREE_CT_FORCE_INLINE_KEEP_DISPATCH 1
#endif


// eigen_fft.hpp (header-only)
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <complex>
#include <cstring>
#include <random>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <string>
#include <thread>
#include <vector>
#include <type_traits>
#include <chrono>
#include <unordered_set>

#include "recovery_ops.hpp"

// #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
//   #include <xmmintrin.h>  // FTZ/DAZ
// #endif
// Crash handler helpers (write minidumps / backtraces). We include this so
// we can optionally produce a minidump when a job causes an access
// violation but keep the process alive.
#include "crash_handler.hpp"
// Helper used to invoke a C++ std::function under a C-callable trampoline so
// SEH can be placed in a function without C++ object unwinding requirements.
#if defined(_WIN32) && defined(_MSC_VER)
struct SEHInvokeCtx {
  // heap-allocated copy of callable
  std::function<void(size_t, size_t, int)>* heap_fn;
  size_t start;
  size_t end;
  int worker;
};
extern "C" inline void eigfft_seh_trampoline(void* p) {
  SEHInvokeCtx* c = static_cast<SEHInvokeCtx*>(p);
  // Call the copied std::function. Any C++ exceptions are separate from SEH.
  (*(c->heap_fn))(c->start, c->end, c->worker);
}
#endif
// Sequential execution is always allowed; parallelism is coordinated by callers.

namespace eigfft {

namespace detail {
// Debug control for WorkerPool: allows tests to set a percentage chance that a
// worker will simulate a crash (debug dropout). Stored as a function-local
// atomic to avoid ODR/static initialization issues in header-only usage.
struct WorkerPoolDebug {
  static std::atomic<int>& dropout_rate() {
    static std::atomic<int> v{0};
    return v;
  }
  // Deterministic pattern controls
  static std::atomic<int>& pattern_period() {
    static std::atomic<int> v{0}; // 0 => disabled
    return v;
  }
  static std::atomic<int>& pattern_on_len() {
    static std::atomic<int> v{0}; // active window within period
    return v;
  }
  static std::atomic<int>& pattern_phase() {
    static std::atomic<int> v{0};
    return v;
  }
  static std::atomic<long long>& single_index() {
    static std::atomic<long long> v{-1}; // -1 => disabled
    return v;
  }
  static std::atomic<int>& persistent_mode() {
    static std::atomic<int> v{0}; // 0=one-shot, 1=persistent
    return v;
  }
  static std::mutex& drop_mutex() {
    static std::mutex m;
    return m;
  }
  static std::unordered_set<size_t>& dropped_once() {
    static std::unordered_set<size_t> s;
    return s;
  }
  static void init_from_env() {
    const char* env = std::getenv("FFTFREE_DEBUG_DROPOUT_PCT");
    if (!env) return;
    try {
      int p = std::stoi(env);
      if (p < 0) p = 0;
      if (p > 100) p = 100;
      dropout_rate().store(p);
    } catch (...) {}
  }
  static void set_drop_pct(int p) { dropout_rate().store(std::clamp(p, 0, 100)); }
  static int debug_dropout_rate_value() { return dropout_rate().load(); }
  static void set_pattern(int period, int on_len, int phase) {
    if (period < 0) period = 0;
    if (on_len < 0) on_len = 0;
    if (period > 0 && on_len > period) on_len = period;
    pattern_period().store(period);
    pattern_on_len().store(on_len);
    pattern_phase().store(phase);
  }
  static void set_single(long long idx) { single_index().store(idx); }
  static void clear_all() {
    dropout_rate().store(0);
    pattern_period().store(0);
    pattern_on_len().store(0);
    pattern_phase().store(0);
    single_index().store(-1);
    persistent_mode().store(0);
    std::lock_guard<std::mutex> g(drop_mutex());
    dropped_once().clear();
  }
};

struct CooleyTukeyDebugConfig {
  bool force_inline_permute = false;
  bool force_inline_all_stages = false;
  std::vector<int> stage_inline_indices;
  bool preserve_outer_dispatch = FFTFREE_CT_FORCE_INLINE_KEEP_DISPATCH != 0;

  bool stage_inline(int stage) const {
    if (stage < 0) return false;
    if (force_inline_all_stages) return true;
    return std::find(stage_inline_indices.begin(), stage_inline_indices.end(), stage) != stage_inline_indices.end();
  }
};

inline bool env_flag_enabled(const char* value) {
  if (!value) return false;
  while (*value && std::isspace(static_cast<unsigned char>(*value))) ++value;
  std::string lowered(value);
  if (lowered.empty()) return true;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lowered == "0" || lowered == "false" || lowered == "off" || lowered == "no") return false;
  return true;
}

inline void append_stage_index(std::vector<int>& indices, int value) {
  if (value < 0) return;
  if (std::find(indices.begin(), indices.end(), value) == indices.end()) {
    indices.push_back(value);
  }
}

inline CooleyTukeyDebugConfig load_cooleytukey_debug_config() {
  CooleyTukeyDebugConfig cfg;
  if (env_flag_enabled(std::getenv("FFTFREE_CT_INLINE_PERMUTE"))) {
    cfg.force_inline_permute = true;
  }

  if (env_flag_enabled(std::getenv("FFTFREE_CT_INLINE_ALL_STAGES"))) {
    cfg.force_inline_all_stages = true;
  }

  const char* preserve_env = std::getenv("FFTFREE_CT_INLINE_KEEP_DISPATCH");
  if (preserve_env) {
    cfg.preserve_outer_dispatch = env_flag_enabled(preserve_env);
  }

  const char* stage_env = std::getenv("FFTFREE_CT_INLINE_STAGE");
  if (stage_env && *stage_env) {
    std::string lowered(stage_env);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (lowered == "all" || lowered == "*" || lowered == "true" || lowered == "on") {
      cfg.force_inline_all_stages = true;
    } else {
      const char* ptr = stage_env;
      while (*ptr) {
        while (*ptr && (std::isspace(static_cast<unsigned char>(*ptr)) || *ptr == ',' || *ptr == ';' || *ptr == ':')) {
          ++ptr;
        }
        if (!*ptr) break;
        char* end = nullptr;
        long value = std::strtol(ptr, &end, 10);
        if (ptr == end) {
          ++ptr;
          continue;
        }
        if (value >= 0 && value <= std::numeric_limits<int>::max()) {
          append_stage_index(cfg.stage_inline_indices, static_cast<int>(value));
        }
        ptr = end;
      }
    }
  }

#if FFTFREE_CT_FORCE_INLINE_PERMUTE
  cfg.force_inline_permute = true;
#endif
#if FFTFREE_CT_FORCE_INLINE_ALL_STAGES
  cfg.force_inline_all_stages = true;
#endif
#if FFTFREE_CT_FORCE_INLINE_STAGE >= 0
  append_stage_index(cfg.stage_inline_indices, static_cast<int>(FFTFREE_CT_FORCE_INLINE_STAGE));
#endif

  if (cfg.force_inline_all_stages) {
    cfg.stage_inline_indices.clear();
  } else if (!cfg.stage_inline_indices.empty()) {
    std::sort(cfg.stage_inline_indices.begin(), cfg.stage_inline_indices.end());
    cfg.stage_inline_indices.erase(std::unique(cfg.stage_inline_indices.begin(), cfg.stage_inline_indices.end()),
                                   cfg.stage_inline_indices.end());
  }

  return cfg;
}

inline const CooleyTukeyDebugConfig& cooleytukey_debug_config() {
  static CooleyTukeyDebugConfig cfg = load_cooleytukey_debug_config();
  return cfg;
}
} // namespace detail


// Public API to control test-only worker pool debug dropout probability.
// Callers may set a value in [0,100] to simulate worker crashes at that
// percentage rate. This is intended for tests only.
inline void set_workerpool_debug_dropout(int pct) {
  detail::WorkerPoolDebug::set_drop_pct(pct);
}
inline int workerpool_debug_dropout() { return detail::WorkerPoolDebug::debug_dropout_rate_value(); }
inline bool workerpool_dropout_pattern_enabled() {
  return detail::WorkerPoolDebug::pattern_period().load() > 0 &&
         detail::WorkerPoolDebug::pattern_on_len().load() > 0;
}
inline long long workerpool_dropout_single_index() {
  return detail::WorkerPoolDebug::single_index().load();
}
inline void set_workerpool_dropout_persistent(bool on) {
  detail::WorkerPoolDebug::persistent_mode().store(on ? 1 : 0);
}
inline void clear_workerpool_dropout_history() {
  std::lock_guard<std::mutex> g(detail::WorkerPoolDebug::drop_mutex());
  detail::WorkerPoolDebug::dropped_once().clear();
}
// Deterministic pattern API
inline void set_workerpool_dropout_pattern(int period, int on_len, int phase) {
  detail::WorkerPoolDebug::set_pattern(period, on_len, phase);
}
inline void set_workerpool_dropout_single(long long index) {
  detail::WorkerPoolDebug::set_single(index);
}


template<class T>
struct Plan;

struct KernelContext {
  virtual ~KernelContext() = default;
};

// Generic algorithm-specific recovery hook interface carried per job.
// All callbacks are optional; null pointers are treated as no-ops.
struct RecoveryOps {
  void (*init)(void* job_ctx, size_t chunk_count) = nullptr;
  void (*capture_chunk)(void* job_ctx, size_t chunk_index, int worker_id) = nullptr;
  void (*restore_range)(void* job_ctx, size_t start_chunk, size_t end_chunk) = nullptr;
  void (*finalize)(void* job_ctx) = nullptr;
};

// Lightweight job-dispatch interface: algorithms call this to either
// run work inline or post to an external dispatcher provided by callers.
class JobDispatcher {
 public:
  using Fn = std::function<void(size_t,size_t,int)>;
  virtual ~JobDispatcher() = default;
  virtual void parallel_for(size_t total, size_t chunk, const Fn& fn) = 0;
  virtual void parallel_for_with_restore(size_t total,
                                         size_t chunk,
                                         const Fn& fn,
                                         std::function<void(size_t,size_t)> restore_fn,
                                         RecoveryOps ops,
                                         void* job_ctx) {
    // Default fallback: execute inline and invoke recovery hooks in-order.
    if (total == 0) {
      if (ops.finalize) { try { ops.finalize(job_ctx); } catch (...) {} }
      return;
    }
    if (chunk == 0) chunk = 1;
    const size_t chunk_count = (total + chunk - 1) / chunk;
    if (ops.init) { try { ops.init(job_ctx, chunk_count); } catch (...) {} }
    size_t idx = 0;
    for (size_t start = 0; start < total; start += chunk, ++idx) {
      const size_t end = std::min(total, start + chunk);
      if (ops.capture_chunk) {
        try { ops.capture_chunk(job_ctx, idx, 0); } catch (...) {}
      }
      fn(start, end, 0);
    }
    if (ops.finalize) { try { ops.finalize(job_ctx); } catch (...) {} }
    (void)restore_fn;
  }
};

namespace detail {

template <class T>
struct ColumnRecoveryJobContext {
  using Complex = std::complex<T>;
  Complex* base = nullptr;
  Eigen::Index axis_stride = 0;   // element stride between rows
  Eigen::Index batch_stride = 0;  // element stride between columns
  int rows = 0;
  int batch = 0;
  int lane_cols = 0;
  std::vector<ColumnSnapshot> snapshots;
};

template <class T>
inline ColumnSlice make_column_slice(const ColumnRecoveryJobContext<T>& ctx,
                                     size_t chunk_index) {
  ColumnSlice slice{};
  using Complex = typename ColumnRecoveryJobContext<T>::Complex;
  if (!ctx.base || ctx.rows <= 0 || ctx.batch <= 0 || ctx.lane_cols <= 0) {
    slice.rowsN = 0;
    slice.width = 0;
    return slice;
  }
  const size_t chunk_start = chunk_index * static_cast<size_t>(ctx.lane_cols);
  if (chunk_start >= static_cast<size_t>(ctx.batch)) {
    slice.rowsN = 0;
    slice.width = 0;
    return slice;
  }
  const int remaining = ctx.batch - static_cast<int>(chunk_start);
  const int width = std::min(ctx.lane_cols, remaining);
  if (width <= 0) {
    slice.rowsN = 0;
    slice.width = 0;
    return slice;
  }
  slice.rowsN = ctx.rows;
  slice.width = width;
  slice.elem_bytes = static_cast<int>(sizeof(Complex));
  slice.axis_stride = static_cast<std::ptrdiff_t>(ctx.axis_stride) * slice.elem_bytes;
  slice.batch_stride = static_cast<std::ptrdiff_t>(ctx.batch_stride) * slice.elem_bytes;
  slice.base = reinterpret_cast<std::byte*>(ctx.base +
               chunk_start * static_cast<size_t>(ctx.batch_stride));
  return slice;
}

template <class T>
inline void column_recovery_init(void* job_ctx, size_t chunk_count) {
  auto* ctx = static_cast<ColumnRecoveryJobContext<T>*>(job_ctx);
  if (!ctx) return;
  try {
    ctx->snapshots.clear();
    ctx->snapshots.resize(chunk_count);
  } catch (...) {
    ctx->snapshots.clear();
  }
}

template <class T>
inline void column_recovery_capture(void* job_ctx, size_t chunk_index, int) {
  auto* ctx = static_cast<ColumnRecoveryJobContext<T>*>(job_ctx);
  if (!ctx) return;
  if (chunk_index >= ctx->snapshots.size()) return;
  ColumnSlice slice = make_column_slice(*ctx, chunk_index);
  if (slice.rowsN <= 0 || slice.width <= 0) return;
  try {
    capture_columns(slice, ctx->snapshots[chunk_index]);
  } catch (...) {
    // best-effort capture
  }
}

template <class T>
inline void column_recovery_restore(void* job_ctx,
                                    size_t start_chunk,
                                    size_t end_chunk) {
  auto* ctx = static_cast<ColumnRecoveryJobContext<T>*>(job_ctx);
  if (!ctx) return;
  if (start_chunk >= end_chunk) return;
  end_chunk = std::min(end_chunk, ctx->snapshots.size());
  for (size_t chunk = start_chunk; chunk < end_chunk; ++chunk) {
    if (chunk >= ctx->snapshots.size()) break;
    const ColumnSnapshot& snap = ctx->snapshots[chunk];
    if (snap.bytes.empty()) continue;
    ColumnSlice slice = make_column_slice(*ctx, chunk);
    if (slice.rowsN <= 0 || slice.width <= 0) continue;
    try {
      restore_columns(slice, snap);
    } catch (...) {
      // continue restoring remaining chunks
    }
  }
}

template <class T>
inline void column_recovery_finalize(void* job_ctx) {
  auto* ctx = static_cast<ColumnRecoveryJobContext<T>*>(job_ctx);
  if (!ctx) return;
  ctx->snapshots.clear();
}

template <class T>
inline RecoveryOps make_column_recovery_ops(ColumnRecoveryJobContext<T>&) {
  RecoveryOps ops;
  ops.init = &column_recovery_init<T>;
  ops.capture_chunk = &column_recovery_capture<T>;
  ops.restore_range = &column_recovery_restore<T>;
  ops.finalize = &column_recovery_finalize<T>;
  return ops;
}

}  // namespace detail

// Lightweight, non-blocking trace logger used only when thread tracing is
// enabled. Worker threads append messages to a thread-local buffer which is
// periodically flushed to a central queue consumed by a background thread.
// This reduces contention and prevents synchronous console I/O from masking
// timing-sensitive races.
#if EIGFFT_TRACE_THREADS
class TraceLogger {
 public:
  static TraceLogger& instance() {
    static TraceLogger inst;
    return inst;
  }

  void log(std::string msg) {
    // Ensure messages end with newline for readability when flushed.
    if (msg.empty() || msg.back() != '\n') msg.push_back('\n');
    thread_local std::vector<std::string> tls_buf;
    tls_buf.push_back(std::move(msg));
    // If the thread-local buffer grows large, move it into the central queue.
    if (tls_buf.size() >= 64) {
      std::vector<std::string> to_push;
      to_push.swap(tls_buf);
      {
        std::lock_guard<std::mutex> g(queue_mutex_);
        queue_.push_back(std::move(to_push));
      }
      queue_cv_.notify_one();
    }
  }

 private:
  TraceLogger() : running_(true) {
    // Open run log before starting background writer so it can immediately
    // accept messages and overwrite previous run's contents.
    open_run_log();
    worker_ = std::thread([this]{ run(); });
  }
  ~TraceLogger() {
    // Flush any thread-local buffers by a best-effort wake (they may be lost
    // if threads exit without flushing). Stop background worker and drain.
    running_ = false;
    queue_cv_.notify_one();
    if (worker_.joinable()) worker_.join();
    drain_all();
    if (file_.is_open()) file_.close();
  }

  void run() {
    while (running_) {
      std::vector<std::vector<std::string>> work;
      {
        std::unique_lock<std::mutex> lk(queue_mutex_);
        if (queue_.empty()) queue_cv_.wait_for(lk, std::chrono::milliseconds(100));
        if (!queue_.empty()) {
          work.swap(queue_);
        }
      }
      if (!work.empty()) {
        std::lock_guard<std::mutex> out_l(out_mutex_);
        if (file_.is_open()) {
          for (auto &vec : work) {
            for (auto &m : vec) file_ << m;
          }
          file_.flush();
        } else {
          for (auto &vec : work) for (auto &m : vec) std::cout << m;
          std::cout << std::flush;
        }
      }
    }
  }

  void drain_all() {
    std::vector<std::vector<std::string>> work;
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      work.swap(queue_);
    }
    std::lock_guard<std::mutex> out_l(out_mutex_);
    if (file_.is_open()) {
      for (auto &vec : work) for (auto &m : vec) file_ << m;
      file_.flush();
    } else {
      for (auto &vec : work) for (auto &m : vec) std::cout << m;
      std::cout << std::flush;
    }
  }

  // Open the per-run log file (overwrites previous run). If the environment
  // variable FFTFREE_LOGFILE is set, use it; otherwise use default name.
  void open_run_log() {
    const char* envp = std::getenv("FFTFREE_LOGFILE");
    const std::string fname = envp ? std::string(envp) : std::string("fftfree_last_run.log");
    // Open truncating so each run overwrites the previous log
    file_.open(fname, std::ios::out | std::ios::trunc);
  }

  std::atomic<bool> running_{false};
  std::thread worker_;
  std::mutex queue_mutex_;
  static inline std::mutex out_mutex_;
  std::condition_variable queue_cv_;
  std::vector<std::vector<std::string>> queue_;
  std::ofstream file_;
};
#else
// When tracing is disabled, provide a no-op shim with the same API so call
// sites don't need to be conditionalized repeatedly.
struct TraceLogger {
  static TraceLogger& instance() { static TraceLogger i; return i; }
  void log(std::string) {}
};
#endif

// Default inline dispatcher: runs the job in the current thread.
class InlineDispatcher : public JobDispatcher {
 public:
  static InlineDispatcher& instance() { static InlineDispatcher d; return d; }
  void parallel_for(size_t total, size_t chunk, const Fn& fn) override {
    if (chunk == 0) chunk = 1;
    const size_t chunks = (total + chunk - 1) / chunk;
    for (size_t idx = 0; idx < chunks; ++idx) {
      const size_t start = idx * chunk;
      const size_t end = std::min(total, start + chunk);
      fn(start, end, 0);
    }
  }

  void parallel_for_with_restore(size_t total,
                                 size_t chunk,
                                 const Fn& fn,
                                 std::function<void(size_t,size_t)> restore_fn,
                                 RecoveryOps ops,
                                 void* job_ctx) override {
    if (chunk == 0) chunk = 1;
    const size_t chunk_count = (total + chunk - 1) / chunk;
    if (ops.init) { try { ops.init(job_ctx, chunk_count); } catch (...) {} }
    size_t idx = 0;
    for (size_t start = 0; start < total; start += chunk, ++idx) {
      const size_t end = std::min(total, start + chunk);
      if (ops.capture_chunk) {
        try { ops.capture_chunk(job_ctx, idx, 0); } catch (...) {}
      }
      fn(start, end, 0);
    }
    if (ops.finalize) { try { ops.finalize(job_ctx); } catch (...) {} }
    (void)restore_fn;
  }
};

template <class T>
struct PlanArena {
  using Complex = std::complex<T>;
  Complex* twiddles = nullptr;
  int twiddle_count = 0;
  int* bitrev = nullptr;
  int bitrev_count = 0;

  Complex* baseline_a = nullptr;
  Complex* baseline_b = nullptr;
  Complex** baseline_columns = nullptr;
  int baseline_thread_capacity = 0;
  int baseline_lane_capacity = 0;

  Complex* stockham_ping = nullptr;
  Complex* stockham_pong = nullptr;
  std::ptrdiff_t* stockham_lane_bases = nullptr;
  int stockham_thread_capacity = 0;
  int stockham_lane_capacity = 0;

  Complex* nd_transpose = nullptr;
  std::size_t nd_transpose_capacity = 0;

  // Optional algorithm-specific special buffers. Fixed-size slot array for
  // fast, indexable hot-path access. Slots may be null if unused.
  static constexpr int kMaxSpecialBuffers = 8;
  Complex* special_buf[kMaxSpecialBuffers] = {};
  std::size_t special_capacity[kMaxSpecialBuffers] = {};
  std::ptrdiff_t special_stride[kMaxSpecialBuffers] = {};
};

template <class T>
struct PlanArenaShape {
  std::size_t twiddles = 0;
  std::size_t bitrev = 0;
  std::size_t baseline_complex = 0;   // count per buffer (a/b)
  std::size_t baseline_columns = 0;   // pointer slots
  std::size_t stockham_stage = 0;     // per ping/pong buffer
  std::size_t stockham_lane_bases = 0;
};

template <class T>
inline PlanArenaShape<T> compute_plan_arena_shape(int N, int thread_capacity, int lane_capacity) {
  PlanArenaShape<T> shape;
  const int threads = std::max(1, thread_capacity);
  const int lanes = std::max(1, lane_capacity);
  shape.twiddles = static_cast<std::size_t>(std::max(1, N / 2));
  shape.bitrev = static_cast<std::size_t>(N);
  shape.baseline_complex = static_cast<std::size_t>(threads) * static_cast<std::size_t>(lanes);
  shape.baseline_columns = static_cast<std::size_t>(threads) * static_cast<std::size_t>(lanes);
  shape.stockham_stage = static_cast<std::size_t>(threads) * static_cast<std::size_t>(lanes) * static_cast<std::size_t>(N);
  shape.stockham_lane_bases = static_cast<std::size_t>(threads) * static_cast<std::size_t>(lanes);
  return shape;
}

namespace detail {

#if defined(EIGFFT_ALLOW_SEQUENTIAL)
inline constexpr bool kAllowSequentialFallback = true;
#else
inline constexpr bool kAllowSequentialFallback = false;
#endif

#if defined(EIGFFT_ENABLE_EXTERNAL_KERNEL)
inline constexpr bool kExternalKernelAvailable = true;
#else
inline constexpr bool kExternalKernelAvailable = false;
#endif

// Timing instrumentation (header-only, low-overhead when disabled).
#ifndef EIGFFT_TIMING
#define EIGFFT_TIMING 0
#endif

#if EIGFFT_TIMING
// Lightweight per-thread timing aggregator with RAII timer.
namespace timing_internal {
enum class Bin : int {
  PlanCacheLookup = 0,
  PlanNew,
  AlgorithmDispatch,
  PreButterfly,
  ButterflyStage,
  PostButterfly,
  kCount
};

struct ThreadTiming {
  double dur[static_cast<int>(Bin::kCount)];
  uint64_t cnt[static_cast<int>(Bin::kCount)];
  ThreadTiming() { for (int i = 0; i < static_cast<int>(Bin::kCount); ++i) { dur[i]=0.0; cnt[i]=0; } }
};

inline std::mutex& timing_registry_mutex() {
  static std::mutex m;
  return m;
}

inline std::vector<ThreadTiming*>& timing_registry() {
  static std::vector<ThreadTiming*> v;
  return v;
}

inline ThreadTiming& thread_timing() {
  thread_local ThreadTiming local;
  thread_local bool registered = false;
  if (!registered) {
    std::lock_guard<std::mutex> g(timing_registry_mutex());
    timing_registry().push_back(&local);
    registered = true;
  }
  return local;
}

struct ScopedTimer {
  // Use steady_clock for portability on MSVC (high_resolution_clock
  // may not be available/aliased the same on all standard library
  // implementations). Avoid the identifier `clock` which can clash
  // with platform symbols; use a clearer name.
  using clock_type = std::chrono::steady_clock;
  Bin bin;
  clock_type::time_point start;
  ScopedTimer(Bin b) : bin(b), start(clock_type::now()) {}
  ~ScopedTimer() {
    const auto end = clock_type::now();
    const double s = std::chrono::duration<double>(end - start).count();
    ThreadTiming& tt = thread_timing();
    const int idx = static_cast<int>(bin);
    tt.dur[idx] += s;
    ++tt.cnt[idx];
  }
};

inline void timing_reset_all() {
  std::lock_guard<std::mutex> g(timing_registry_mutex());
  for (ThreadTiming* p : timing_registry()) {
    for (int i = 0; i < static_cast<int>(Bin::kCount); ++i) { p->dur[i]=0.0; p->cnt[i]=0; }
  }
}

inline void timing_report(std::ostream& os) {
  std::lock_guard<std::mutex> g(timing_registry_mutex());
  ThreadTiming agg;
  for (ThreadTiming* p : timing_registry()) {
    for (int i = 0; i < static_cast<int>(Bin::kCount); ++i) {
      agg.dur[i] += p->dur[i];
      agg.cnt[i] += p->cnt[i];
    }
  }
  os << "\n=== eigfft timing report ===\n";
  auto print = [&](const char* name, int idx) {
    os << "  " << name << ": total=" << agg.dur[idx] << " s";
    if (agg.cnt[idx] > 0) os << " (calls=" << agg.cnt[idx] << ", avg=" << (agg.dur[idx] / static_cast<double>(agg.cnt[idx])) << " s)";
    os << "\n";
  };
  print("plan_cache_lookup", static_cast<int>(Bin::PlanCacheLookup));
  print("plan_new", static_cast<int>(Bin::PlanNew));
  print("algorithm_dispatch", static_cast<int>(Bin::AlgorithmDispatch));
  print("pre_butterfly", static_cast<int>(Bin::PreButterfly));
  print("butterfly_stage", static_cast<int>(Bin::ButterflyStage));
  print("post_butterfly", static_cast<int>(Bin::PostButterfly));
}

// Snapshot/delta helpers for per-test reporting.
struct TimingSnapshot {
  double dur[static_cast<int>(Bin::kCount)];
  uint64_t cnt[static_cast<int>(Bin::kCount)];
  TimingSnapshot() {
    for (int i = 0; i < static_cast<int>(Bin::kCount); ++i) { dur[i] = 0.0; cnt[i] = 0; }
  }
};

inline TimingSnapshot timing_snapshot() {
  std::lock_guard<std::mutex> g(timing_registry_mutex());
  TimingSnapshot out;
  for (ThreadTiming* p : timing_registry()) {
    for (int i = 0; i < static_cast<int>(Bin::kCount); ++i) {
      out.dur[i] += p->dur[i];
      out.cnt[i] += p->cnt[i];
    }
  }
  return out;
}

inline void timing_report_delta(const TimingSnapshot& before, const TimingSnapshot& after, std::ostream& os, const std::string& label = std::string()) {
  os << "\n=== eigfft timing delta";
  if (!label.empty()) os << " (" << label << ")";
  os << " ===\n";
  auto print = [&](const char* name, int idx) {
    const double d = after.dur[idx] - before.dur[idx];
    const uint64_t c = after.cnt[idx] - before.cnt[idx];
    os << "  " << name << ": delta=" << d << " s";
    if (c > 0) os << " (calls=" << c << ", avg=" << (d / static_cast<double>(c)) << " s)";
    os << "\n";
  };
  print("plan_cache_lookup", static_cast<int>(Bin::PlanCacheLookup));
  print("plan_new", static_cast<int>(Bin::PlanNew));
  print("algorithm_dispatch", static_cast<int>(Bin::AlgorithmDispatch));
  print("pre_butterfly", static_cast<int>(Bin::PreButterfly));
  print("butterfly_stage", static_cast<int>(Bin::ButterflyStage));
  print("post_butterfly", static_cast<int>(Bin::PostButterfly));
}
} // namespace timing_internal
#endif // EIGFFT_TIMING

class WorkerPool {
 public:
  WorkerPool() = default;

  explicit WorkerPool(int threads) { reset(threads); }

  ~WorkerPool() { shutdown(); }

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;
  WorkerPool(WorkerPool&&) = delete;
  WorkerPool& operator=(WorkerPool&&) = delete;

  void reset(int threads) {
    int requested = std::max(1, threads);
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 0) requested = std::min(requested, hw);
    if (workers_.empty()) {
      stop_ = false;
      std::atomic_store(&job_.fn, std::shared_ptr<Job::FnType>(nullptr));
      job_.total = 0;
      job_.chunk = 1;
      job_.chunk_count = 0;
      job_.next.store(0, std::memory_order_relaxed);
      job_.pending.store(0, std::memory_order_relaxed);
      job_.active = false;
      // Start with zero threads recorded so reset() will create the
      // requested number of worker threads even when requested==1.
      // This enforces a worker-thread-only execution model: work is
      // always executed by pool workers, never inline on the caller
      // thread.
      total_threads_ = 0;
      worker_count_ = 0;
    }

    if (requested <= total_threads_) {
#if EIGFFT_TRACE_THREADS
  {
    std::ostringstream _oss;
    _oss << "[pool] resize request threads=" << requested
     << " (keeping existing total_threads=" << total_threads_ << ")";
    TraceLogger::instance().log(_oss.str());
  }
#endif
      return;  // keep existing fleet alive; never shrink in the hot path
    }

    const int additional_workers = requested - total_threads_;
    if (additional_workers <= 0) return;

    workers_.reserve(static_cast<size_t>(worker_count_ + additional_workers));
    for (int i = 0; i < additional_workers; ++i) {
      // Start worker ids at 0 so the first worker is id=0. The main
      // thread is considered the waiting thread and does not participate
      // in chunk processing when workers exist.
      const int worker_id = worker_count_;
      workers_.emplace_back([this, worker_id]() { worker_loop(worker_id); });
      ++worker_count_;
      ++total_threads_;
      
#if EIGFFT_TRACE_THREADS
  {
    std::ostringstream _oss;
    _oss << "[pool] started worker id=" << worker_id
     << " total_threads=" << total_threads_;
    TraceLogger::instance().log(_oss.str());
  }
#endif
    }
  }

  int size() const { return total_threads_; }

  template <class Fn>
  void parallel_for(size_t total_items, size_t chunk, Fn&& fn) {
    if (total_items == 0) return;
    if (chunk == 0) chunk = 1;
    const size_t chunk_size = chunk;
    std::function<void(size_t, size_t, int)> wrapped = std::forward<Fn>(fn);
#if EIGFFT_TRACE_THREADS
    {
      std::ostringstream _oss;
      _oss << "[pool] parallel_for begin total=" << total_items
           << " chunk=" << chunk_size
           << " worker_count=" << worker_count_
           << " total_threads=" << total_threads_;
      TraceLogger::instance().log(_oss.str());
    }
#endif
    // Inline execution only when there are no worker threads available
    // or the work size is so small that chunking would be pointless.
    if (worker_count_ == 0 || total_items <= chunk_size) {
      size_t start = 0;
      while (start < total_items) {
        const size_t end = std::min(total_items, start + chunk_size);
        wrapped(start, end, 0);
        start = end;
      }
#if EIGFFT_TRACE_THREADS
  TraceLogger::instance().log(std::string("[pool] parallel_for completed inline"));
#endif
      return;
    }

    const size_t chunk_count = (total_items + chunk_size - 1) / chunk_size;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Store a heap-allocated callable into a shared_ptr and publish it
      // atomically so workers can safely copy the shared_ptr without
      // holding the mutex in the hot path.
  auto callable_ptr = std::make_shared<Job::FnType>(wrapped);
  std::atomic_store(&job_.fn, callable_ptr);
  // Clear any restore callback for this job.
  std::atomic_store(&job_.on_failure, std::shared_ptr<Job::RestoreType>(nullptr));
      job_.total = total_items;
      job_.chunk = chunk_size;
      job_.chunk_count = chunk_count;
      job_.next.store(0, std::memory_order_relaxed);
        // Count only worker threads; the main thread will wait and not
        // participate in chunk processing when worker threads are present.
        // This prevents the main thread (worker_id==0) from executing job
        // callbacks inline and ensures debug-dropout never targets the main
        // thread (which uses worker_id==0).
        job_.pending.store(worker_count_, std::memory_order_relaxed);
      job_.active = true;
#if EIGFFT_TRACE_THREADS
  {
    std::ostringstream _oss;
    _oss << "[pool] armed job fn=" << static_cast<const void*>(callable_ptr.get())
     << " total=" << job_.total
     << " chunks=" << job_.chunk_count
     << " pending=" << job_.pending.load();
    TraceLogger::instance().log(_oss.str());
  }
#endif
    }
    cv_job_.notify_all();

    // Don't process chunks on the main thread; wait for workers to finish.
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] { return !job_.active; });
  // At this point all workers have finished draining chunks.
  // Clear the job callback under the mutex to ensure no worker observes a
  // cleared fn while still able to enter the call site. Use atomic_store
  // on the shared_ptr so readers using atomic_load see a consistent null.
  std::atomic_store(&job_.fn, std::shared_ptr<Job::FnType>(nullptr));
  std::atomic_store(&job_.on_failure, std::shared_ptr<Job::RestoreType>(nullptr));
#if EIGFFT_TRACE_THREADS
  TraceLogger::instance().log(std::string("[pool] parallel_for finished"));
#endif
  }

  // Extended parallel_for that accepts a restore callback invoked when a
  // worker experiences a crash while executing a chunk. The restore callback
  // receives the start/end indices of the failed chunk(s).
  template <class Fn>
  void parallel_for_with_restore(size_t total_items, size_t chunk, Fn&& fn,
                                 std::function<void(size_t,size_t)> restore_fn) {
    parallel_for_with_restore_ex(total_items,
                                 chunk,
                                 std::forward<Fn>(fn),
                                 std::move(restore_fn),
                                 RecoveryOps{},
                                 nullptr);
  }

  // Extended API with algorithm-specific recovery hooks. Existing behavior is
  // preserved when hooks are null (no-ops).
  template <class Fn>
  void parallel_for_with_restore_ex(size_t total_items,
                                    size_t chunk,
                                    Fn&& fn,
                                    std::function<void(size_t,size_t)> restore_fn,
                                    RecoveryOps ops,
                                    void* job_ctx) {
    if (total_items == 0) return;
    if (chunk == 0) chunk = 1;
    const size_t chunk_size = chunk;
    std::function<void(size_t, size_t, int)> wrapped = std::forward<Fn>(fn);
#if EIGFFT_TRACE_THREADS
    {
      std::ostringstream _oss;
      _oss << "[pool] parallel_for_ex begin total=" << total_items
           << " chunk=" << chunk_size
           << " worker_count=" << worker_count_
           << " total_threads=" << total_threads_;
      TraceLogger::instance().log(_oss.str());
    }
#endif
    if (worker_count_ == 0 || total_items <= chunk_size) {
      size_t start = 0;
      const size_t chunk_count = (total_items + chunk_size - 1) / chunk_size;
      if (ops.init) { try { ops.init(job_ctx, chunk_count); } catch(...){} }
      size_t chunk_index = 0;
      while (start < total_items) {
        const size_t end = std::min(total_items, start + chunk_size);
        if (ops.capture_chunk) { try { ops.capture_chunk(job_ctx, chunk_index, 0); } catch(...){} }
        wrapped(start, end, 0);
        start = end;
        ++chunk_index;
      }
      if (ops.finalize) { try { ops.finalize(job_ctx); } catch(...){} }
#if EIGFFT_TRACE_THREADS
      TraceLogger::instance().log(std::string("[pool] parallel_for_ex completed inline"));
#endif
      return;
    }

    const size_t chunk_count = (total_items + chunk_size - 1) / chunk_size;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto callable_ptr = std::make_shared<Job::FnType>(wrapped);
      std::atomic_store(&job_.fn, callable_ptr);
      auto restore_ptr = std::make_shared<Job::RestoreType>(restore_fn);
      std::atomic_store(&job_.on_failure, restore_ptr);
      job_.total = total_items;
      job_.chunk = chunk_size;
      job_.chunk_count = chunk_count;
      job_.next.store(0, std::memory_order_relaxed);
      job_.pending.store(worker_count_, std::memory_order_relaxed);
      job_.active = true;
      job_.ops = ops;
      job_.job_ctx = job_ctx;
#if EIGFFT_TRACE_THREADS
      {
        std::ostringstream _oss;
        _oss << "[pool] armed job(ex) fn=" << static_cast<const void*>(callable_ptr.get())
             << " total=" << job_.total
             << " chunks=" << job_.chunk_count
             << " pending=" << job_.pending.load();
        TraceLogger::instance().log(_oss.str());
      }
#endif
      if (job_.ops.init) { try { job_.ops.init(job_.job_ctx, job_.chunk_count); } catch(...){} }
    }
    cv_job_.notify_all();

    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] { return !job_.active; });

    if (job_.ops.finalize) { try { job_.ops.finalize(job_.job_ctx); } catch(...){} }

    std::atomic_store(&job_.fn, std::shared_ptr<Job::FnType>(nullptr));
    std::atomic_store(&job_.on_failure, std::shared_ptr<Job::RestoreType>(nullptr));
    job_.ops = RecoveryOps{};
    job_.job_ctx = nullptr;
#if EIGFFT_TRACE_THREADS
    TraceLogger::instance().log(std::string("[pool] parallel_for_ex finished"));
#endif
  }

  private:
  struct Job {
    using FnType = std::function<void(size_t, size_t, int)>;
    using RestoreType = std::function<void(size_t, size_t)>;
    // Use shared_ptr to the callable and atomic_load/atomic_store on the
    // shared_ptr to make copies safe across threads without taking the
    // main mutex at the hot path. This avoids a tiny window where a worker
    // could observe a cleared std::function and throw std::_Xbad_function_call.
    std::shared_ptr<FnType> fn;
    size_t total = 0;
    size_t chunk = 1;
    size_t chunk_count = 0;
    std::atomic<size_t> next{0};
    std::atomic<int> pending{0};
    bool active = false;
    // Optional per-job restore callback published atomically (may be null).
    std::shared_ptr<RestoreType> on_failure;
    // Algorithm-specific recovery hooks and opaque context
    RecoveryOps ops{};
    void* job_ctx = nullptr;
  };

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (workers_.empty()) {
        stop_ = true;
        job_.active = false;
      } else {
        stop_ = true;
        job_.active = false;
      }
    }
    cv_job_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
    workers_.clear();
    total_threads_ = 1;
    worker_count_ = 0;
  }

  void worker_loop(int worker_id) {
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_job_.wait(lock, [&] { return stop_ || job_.active; });
      if (stop_) return;
#if EIGFFT_TRACE_THREADS
  {
    std::ostringstream _oss;
    _oss << "[pool] worker " << worker_id << " woke";
    TraceLogger::instance().log(_oss.str());
  }
#endif
      lock.unlock();

      drain_chunks(worker_id);

      // Worker thread finished. Decrement and signal ATOMICALLY.
      {
        std::unique_lock<std::mutex> lock2(mutex_);
        const int remaining = job_.pending.fetch_sub(1, std::memory_order_acq_rel);
        if (remaining == 1) {
          job_.active = false;
          cv_done_.notify_one();
        }
      }
    }
  }

  void drain_chunks(int worker_id) {
    for (;;) {
      const size_t index = job_.next.fetch_add(1, std::memory_order_acq_rel);
      if (index >= job_.chunk_count) break;
      const size_t start = index * job_.chunk;
      const size_t end = std::min(job_.total, start + job_.chunk);
#if EIGFFT_TRACE_THREADS
  {
    std::ostringstream _oss;
    _oss << "[pool] worker " << worker_id << " processing chunk idx=" << index
     << " start=" << start << " end=" << end;
    TraceLogger::instance().log(_oss.str());
  }
#endif
      // Make an atomic copy of the callable so it cannot be cleared out from
      // under us while we execute it. If the job has been cleared, bail.
  auto fn = std::atomic_load(&job_.fn);
  if (!fn) {
#if EIGFFT_TRACE_THREADS
  {
    std::ostringstream _oss;
    _oss << "[pool] worker " << worker_id << " saw cleared fn at idx=" << index
         << " start=" << start << " end=" << end;
    TraceLogger::instance().log(_oss.str());
  }
#endif
    break;
  }
#if EIGFFT_TRACE_THREADS
  {
    std::ostringstream _oss;
    _oss << "[pool] worker " << worker_id << " calling fn=" << static_cast<const void*>(fn.get())
         << " idx=" << index << " start=" << start << " end=" << end;
    TraceLogger::instance().log(_oss.str());
  }
#endif

  #define EIGFFT_JOB_RETRY_COUNT 2
  // Build a small "safe" wrapper that can inject a debug dropout (simulated
  // worker crash) at a configured probability. This lets tests force a
  // per-worker failure deterministically via runtime setter without touching
  // production code paths. The wrapper captures the shared callable so it
  // remains valid across the SEH trampoline and worker retry loops.
  std::function<void(size_t,size_t,int)> invoking_fn = nullptr;
  {
    // Capture the loaded shared_ptr locally to keep it alive.
    auto captured_fn = fn;
    // Capture 'this' so the dropout injection can know whether the pool
    // actually has worker threads. We allow dropout only when worker
    // threads exist; inline execution (worker_count_ == 0) remains
    // immune unless explicitly enabled via a debug flag.
    invoking_fn = [this, captured_fn](size_t s, size_t e, int w) {
  // Decide deterministic dropout first (single index / stripe pattern),
  // then fall back to probabilistic rate.
  bool do_drop = false;
  // Compute chunk index from start offset and configured chunk size
  size_t idx = 0;
  const size_t chunk_sz = (job_.chunk == 0) ? 1 : job_.chunk;
  idx = s / chunk_sz;

  long long single = eigfft::detail::WorkerPoolDebug::single_index().load();
  if (single >= 0 && static_cast<long long>(idx) == single) {
    do_drop = true;
  } else {
    int period = eigfft::detail::WorkerPoolDebug::pattern_period().load();
    int onlen = eigfft::detail::WorkerPoolDebug::pattern_on_len().load();
    int phase = eigfft::detail::WorkerPoolDebug::pattern_phase().load();
    if (period > 0 && onlen > 0) {
      long long pos = static_cast<long long>(idx) + static_cast<long long>(phase);
      int mod = static_cast<int>((pos % period + period) % period);
      if (mod < onlen) do_drop = true;
    }
  }
  if (!do_drop) {
    int rate = eigfft::detail::WorkerPoolDebug::debug_dropout_rate_value();
    if (rate > 0) {
      thread_local std::mt19937_64 tls_rng((unsigned)std::hash<std::thread::id>()(std::this_thread::get_id()) ^ 0x9e3779b97f4a7c15ULL);
      std::uniform_int_distribution<int> dist(1, 100);
      if (dist(tls_rng) <= rate) do_drop = true;
    }
  }
  if (do_drop) {
    // Enforce one-shot by default: only drop once per chunk index unless
    // persistent mode is explicitly enabled for stress testing.
    if (eigfft::detail::WorkerPoolDebug::persistent_mode().load() == 0) {
      std::lock_guard<std::mutex> g(eigfft::detail::WorkerPoolDebug::drop_mutex());
      auto &seen = eigfft::detail::WorkerPoolDebug::dropped_once();
      if (seen.find(idx) != seen.end()) {
        do_drop = false;
      } else {
        seen.insert(idx);
      }
    }
  }
  if (do_drop) {
    std::ostringstream _oss;
    _oss << "[pool] DEBUG_DROP: worker " << w << " simulating crash s=" << s << " e=" << e
         << " idx=" << idx;
    TraceLogger::instance().log(_oss.str());
    // Simulate a crash.
#if defined(_WIN32)
    volatile int* p = nullptr; *p = 0;
#else
    throw std::runtime_error("debug-dropout simulated crash");
#endif
  }
      // Invoke the original callable
      (*captured_fn)(s, e, w);
    };
  }
  // Invoke the job callable inside a platform-specific containment block so
  // an access violation or other SEH exception from the job does not unwind
  // out of the worker thread and kill the process. On Windows with MSVC we
  // use SEH (delegated to fftfree_run_with_seh) and produce a minidump; on
  // other platforms we catch C++ exceptions and log them. In both cases we
  // attempt a small number of restore+retry cycles when a failure occurs.
  #if defined(_WIN32) && defined(_MSC_VER)
  // Use fftfree_run_with_seh and a heap-copied std::function for trampoline
  // invocation so MSVC's SEH rules are satisfied.
  {
  // Use the invoking_fn wrapper so debug-dropout injection takes effect
  // Call capture hook just before invoking work for this chunk
  try { if (job_.ops.capture_chunk) job_.ops.capture_chunk(job_.job_ctx, index, worker_id); } catch (...) {}
  SEHInvokeCtx* ctx = new SEHInvokeCtx{ new std::function<void(size_t,size_t,int)>(invoking_fn), start, end, worker_id };
  int ex = fftfree::fftfree_run_with_seh(&eigfft_seh_trampoline, static_cast<void*>(ctx));
    delete ctx->heap_fn;
    delete ctx;
    if (ex) {
      std::ostringstream _oss;
      _oss << "[pool] worker " << worker_id << " job crashed at idx=" << index
           << " start=" << start << " end=" << end;
      TraceLogger::instance().log(_oss.str());
      auto restore_cb = std::atomic_load(&job_.on_failure);
      if (restore_cb) {
#if defined(EIGFFT_RESTORE_CONSOLE_LOGS)
  std::fprintf(stderr, "[restore] invoking restore worker=%d idx=%zu start=%zu end=%zu\n", worker_id, index, start, end);
#endif
#if EIGFFT_TRACE_THREADS
  {
    std::ostringstream _oss2;
    _oss2 << "[pool] invoking restore callback worker=" << worker_id << " idx=" << index
    << " start=" << start << " end=" << end;
    TraceLogger::instance().log(_oss2.str());
  }
#endif
  // Algorithm-specific restore prior to legacy callback
  try {
    if (job_.ops.restore_range) {
      const size_t start_chunk = (job_.chunk == 0) ? 0 : (start / job_.chunk);
      const size_t end_chunk = (job_.chunk == 0) ? 0 : ((end + job_.chunk - 1) / job_.chunk);
      job_.ops.restore_range(job_.job_ctx, start_chunk, end_chunk);
    }
  } catch (...) {}
  (*restore_cb)(start, end);
#if defined(EIGFFT_RESTORE_CONSOLE_LOGS)
  std::fprintf(stderr, "[restore] completed restore worker=%d idx=%zu start=%zu end=%zu\n", worker_id, index, start, end);
#endif
#if EIGFFT_TRACE_THREADS
  TraceLogger::instance().log(std::string("[pool] restore callback completed"));
#endif
      }
      // Attempt retries after restore using the SEH runner again.
      for (int attempt = 0; attempt < EIGFFT_JOB_RETRY_COUNT; ++attempt) {
        std::ostringstream _oss2;
        _oss2 << "[pool] worker " << worker_id << " retry attempt=" << (attempt+1) << " idx=" << index;
        TraceLogger::instance().log(_oss2.str());
#if defined(EIGFFT_RESTORE_CONSOLE_LOGS)
        std::fprintf(stderr, "[restore] retry attempt=%d worker=%d idx=%zu\n", (attempt+1), worker_id, index);
#endif
        SEHInvokeCtx* ctx2 = new SEHInvokeCtx{ new std::function<void(size_t,size_t,int)>(invoking_fn), start, end, worker_id };
        int ex2 = fftfree::fftfree_run_with_seh(&eigfft_seh_trampoline, static_cast<void*>(ctx2));
        delete ctx2->heap_fn;
        delete ctx2;
        if (!ex2) {
          TraceLogger::instance().log(std::string("[pool] worker retry succeeded"));
#if defined(EIGFFT_RESTORE_CONSOLE_LOGS)
          std::fprintf(stderr, "[restore] retry succeeded worker=%d idx=%zu attempt=%d\n", worker_id, index, (attempt+1));
#endif
          break;
        }
        if (attempt + 1 < EIGFFT_JOB_RETRY_COUNT && restore_cb) {
#if EIGFFT_TRACE_THREADS
          std::ostringstream _oss3;
          _oss3 << "[pool] retry invoking restore callback attempt=" << (attempt+1) << " worker=" << worker_id << " idx=" << index;
          TraceLogger::instance().log(_oss3.str());
#endif
          // Algorithm-specific restore prior to legacy callback
          try {
            if (job_.ops.restore_range) {
              const size_t start_chunk = (job_.chunk == 0) ? 0 : (start / job_.chunk);
              const size_t end_chunk = (job_.chunk == 0) ? 0 : ((end + job_.chunk - 1) / job_.chunk);
              job_.ops.restore_range(job_.job_ctx, start_chunk, end_chunk);
            }
          } catch (...) {}
          (*restore_cb)(start, end);
#if defined(EIGFFT_RESTORE_CONSOLE_LOGS)
          std::fprintf(stderr, "[restore] retry restore completed worker=%d idx=%zu attempt=%d\n", worker_id, index, (attempt+1));
#endif
#if EIGFFT_TRACE_THREADS
          TraceLogger::instance().log(std::string("[pool] retry restore completed"));
#endif
        }
        if (attempt + 1 == EIGFFT_JOB_RETRY_COUNT) {
          TraceLogger::instance().log(std::string("[pool] worker persistent failure after retries"));
#if defined(EIGFFT_RESTORE_CONSOLE_LOGS)
          std::fprintf(stderr, "[restore] persistent failure after retries worker=%d idx=%zu\n", worker_id, index);
#endif
        }
      }
    }
  }
  #else
  // Non-Windows: catch C++ exceptions and attempt restore+retry cycles.
  int attempt = 0;
  for (;;) {
    try {
      // Call the wrapper so injected debug-dropout is executed.
      // Capture just before invoking work for this chunk
      try { if (job_.ops.capture_chunk) job_.ops.capture_chunk(job_.job_ctx, index, worker_id); } catch (...) {}
      invoking_fn(start, end, worker_id);
      break; // success
    } catch (const std::exception& e) {
      std::ostringstream _oss;
      _oss << "[pool] worker " << worker_id << " job threw exception: " << e.what()
           << " idx=" << index << " start=" << start << " end=" << end;
      TraceLogger::instance().log(_oss.str());
    } catch (...) {
      std::ostringstream _oss;
      _oss << "[pool] worker " << worker_id << " job threw unknown exception at idx=" << index
           << " start=" << start << " end=" << end;
      TraceLogger::instance().log(_oss.str());
    }
    auto restore_cb = std::atomic_load(&job_.on_failure);
    if (restore_cb) {
#if defined(EIGFFT_RESTORE_CONSOLE_LOGS)
  std::fprintf(stderr, "[restore] invoking restore (non-win) worker=%d idx=%zu start=%zu end=%zu\n", worker_id, index, start, end);
#endif
#if EIGFFT_TRACE_THREADS
  std::ostringstream _oss4;
  _oss4 << "[pool] invoking restore callback (non-win) worker=" << worker_id << " idx=" << index
    << " start=" << start << " end=" << end;
  TraceLogger::instance().log(_oss4.str());
#endif
  // Algorithm-specific restore prior to legacy callback
  try {
    if (job_.ops.restore_range) {
      const size_t start_chunk = (job_.chunk == 0) ? 0 : (start / job_.chunk);
      const size_t end_chunk = (job_.chunk == 0) ? 0 : ((end + job_.chunk - 1) / job_.chunk);
      job_.ops.restore_range(job_.job_ctx, start_chunk, end_chunk);
    }
  } catch (...) {}
  (*restore_cb)(start, end);
#if defined(EIGFFT_RESTORE_CONSOLE_LOGS)
  std::fprintf(stderr, "[restore] restore completed (non-win) worker=%d idx=%zu start=%zu end=%zu\n", worker_id, index, start, end);
#endif
#if EIGFFT_TRACE_THREADS
  TraceLogger::instance().log(std::string("[pool] restore callback completed (non-win)"));
#endif
    }
    ++attempt;
    if (attempt > EIGFFT_JOB_RETRY_COUNT) {
      TraceLogger::instance().log(std::string("[pool] worker persistent failure after retries"));
      break;
    }
    std::ostringstream _oss2;
    _oss2 << "[pool] worker retry attempt=" << attempt << " idx=" << index;
    TraceLogger::instance().log(_oss2.str());
  }
#endif
    }
  }

  int total_threads_ = 1;
  int worker_count_ = 0;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_job_;
  std::condition_variable cv_done_;
  bool stop_ = false;
  Job job_{};
};

// Close the local `detail` namespace while including standalone butterfly
// headers. The butterfly headers declare their own `eigfft::detail` scope so
// include them at namespace scope to avoid nested `eigfft::detail::eigfft::detail`.
} // namespace detail
using detail::WorkerPool;

// Pool-backed JobDispatcher: forwards algorithm work to a caller-owned
// WorkerPool, falling back to inline execution when no pool is attached
// (or the pool pointer hasn't been set yet). This is the "outer WorkerPool"
// half of the threading model described in README.md: install one of these,
// bound to a single shared WorkerPool, via Plan::set_dispatcher() on every
// Plan that should share that pool's thread budget.
//
// fftfree does not reconcile pool sizing against arena sizing on its own --
// the caller must keep the WorkerPool's thread count consistent with the
// PlanRuntimeConfig::threads used to build each Plan (with
// allow_outer_parallel=true). A pool larger than the arena's thread
// capacity can hand a kernel a worker_id past the end of its per-thread
// scratch buffers.
//
// This was previously duplicated ad hoc in fft_cffi.cpp, main.cpp, and
// tests/restore_test.cpp; this is the one canonical implementation.
class PoolDispatcher : public JobDispatcher {
 public:
  PoolDispatcher() = default;
  explicit PoolDispatcher(WorkerPool* pool) : pool_(pool) {}

  void set_pool(WorkerPool* pool) { pool_ = pool; }
  WorkerPool* pool() const { return pool_; }

  void parallel_for(size_t total, size_t chunk, const Fn& fn) override {
    if (!pool_ || total == 0) {
      InlineDispatcher::instance().parallel_for(total, chunk, fn);
      return;
    }
    pool_->parallel_for(total, chunk == 0 ? 1 : chunk, fn);
  }

  void parallel_for_with_restore(size_t total,
                                 size_t chunk,
                                 const Fn& fn,
                                 std::function<void(size_t,size_t)> restore_fn,
                                 RecoveryOps ops,
                                 void* job_ctx) override {
    if (!pool_ || total == 0) {
      InlineDispatcher::instance().parallel_for_with_restore(
          total, chunk, fn, std::move(restore_fn), ops, job_ctx);
      return;
    }
    pool_->parallel_for_with_restore_ex(
        total, chunk == 0 ? 1 : chunk, fn, std::move(restore_fn), ops, job_ctx);
  }

 private:
  WorkerPool* pool_ = nullptr;
};
} // namespace eigfft

// Force the core to use the external butterfly types (compat header with
// aliases). You can set EIGFFT_USE_EXTERNAL_BFLY=0 to keep the old
// internal definitions in `eigen_fft.hpp` (not recommended).
#ifndef EIGFFT_USE_EXTERNAL_BFLY
#define EIGFFT_USE_EXTERNAL_BFLY 1
#endif

#include "butterfly_api.hpp"
#if EIGFFT_USE_EXTERNAL_BFLY
#include "butterfly_kernel.hpp" // compatibility header; defines aliases into detail
#else
// If external butterfly is disabled, optionally include internal lightweight
// implementation (kept separate to reduce file size). By default we prefer the
// external compatibility header.
#include "butterfly_impl_radix2.hpp"
#endif

namespace eigfft {
namespace detail {

template <class T>
struct StockhamState : KernelContext {
  using Complex = std::complex<T>;

  explicit StockhamState(Plan<T>& plan, PlanArena<T>& arena_ref)
      : arena(arena_ref),
        N(plan.N) {
    if (arena.stockham_lane_capacity <= 0) {
      throw std::invalid_argument("PlanArena stockham_lane_capacity must be positive");
    }
    if (!arena.stockham_ping || !arena.stockham_pong || !arena.stockham_lane_bases) {
      throw std::invalid_argument("PlanArena missing Stockham scratch buffers");
    }
  }

  void ensure_lane_capacity(int /*requested*/) {}

  void ensure_threads(int /*threads*/) {}

  size_t thread_stride() const {
    return static_cast<size_t>(N) * static_cast<size_t>(std::max(1, arena.stockham_lane_capacity));
  }

  PlanArena<T>& arena;
  int N = 0;
};

}  // namespace detail

enum class MetadataKind {
  Twiddle,
  LayoutMap,
  StageSnapshot,
  StageParams,
  ButterflyPairs,
  StageInvariant,
  TwiddleIndexMap,
  // New: Per-bin angular frequency metadata (omega_k = 2*pi*k/N)
  Omega,
  // New: Per-batch versions capture all windows (flattened B dimension)
  StageSnapshotAll,
  ButterflyPairsAll,
  TwiddleIndexMapAll
};

// Cooley–Tukey kernel state mirroring Stockham's persistent WorkerPool model.
// Cooley–Tukey uses single-threaded execution by design to avoid inner parallelism.

template <class T>
struct MetadataRequest {
  MetadataKind kind = MetadataKind::Twiddle;
  std::complex<T>* complex_buffer = nullptr;
  std::size_t element_count = 0;
};

template <class T>
struct AxisLayout {
  using Complex = std::complex<T>;
  Complex* base = nullptr;
  Eigen::Index axis_size = 0;
  Eigen::Index batch_size = 0;
  Eigen::Index axis_stride = 1;
  Eigen::Index batch_stride = 0;
  const MetadataRequest<T>* metadata_requests = nullptr;
  int metadata_request_count = 0;

  Complex* column_ptr(Eigen::Index batch) const {
    return base + batch * batch_stride;
  }

  Complex* element_ptr(Eigen::Index axis_idx, Eigen::Index batch) const {
    return base + axis_idx * axis_stride + batch * batch_stride;
  }

  bool valid() const {
    return base != nullptr && axis_size > 0;
  }
};

namespace detail {

template <class T>
struct TwiddleMetadataWriter {
  const MetadataRequest<T>* requests = nullptr;
  int count = 0;
  int stages = 0;
  std::complex<T>* twiddle_buffer = nullptr;
  bool has_twiddle = false;
  std::size_t total_twiddles = 0;

  TwiddleMetadataWriter(const AxisLayout<T>& layout, int stage_count)
      : requests(layout.metadata_requests),
        count(layout.metadata_request_count),
        stages(stage_count) {
    if (!requests || count <= 0 || stages <= 0) return;
    total_twiddles = compute_total_twiddles(stages);
    for (int i = 0; i < count; ++i) {
      const MetadataRequest<T>& req = requests[i];
      if (req.kind != MetadataKind::Twiddle) continue;
      if (total_twiddles > 0) {
        if (!req.complex_buffer) {
          throw std::invalid_argument("Twiddle metadata requires complex buffer");
        }
        if (req.element_count < total_twiddles) {
          throw std::invalid_argument("Twiddle metadata buffer too small");
        }
      }
      twiddle_buffer = req.complex_buffer;
      has_twiddle = (twiddle_buffer != nullptr);
    }
  }

  bool enabled() const { return has_twiddle; }

  void record(int stage_idx, int twiddle_idx, const std::complex<T>& value) const {
    if (!has_twiddle) return;
    if (stage_idx < 0 || stage_idx >= stages) return;
    if (twiddle_idx < 0) return;
    const std::size_t per_stage = static_cast<std::size_t>(1) << stage_idx;
    if (static_cast<std::size_t>(twiddle_idx) >= per_stage) return;
    const std::size_t stage_offset = (static_cast<std::size_t>(1) << stage_idx) - 1;
    twiddle_buffer[stage_offset + static_cast<std::size_t>(twiddle_idx)] = value;
  }

  void record_plan_twiddles(const Plan<T>& plan) const {
    if (!enabled()) return;
    const int total_stages = std::min(stages, plan.lgN);
    for (int stage = 0; stage < total_stages; ++stage) {
      const int m = 1 << stage;
      const int distance = m << 1;
      const int tw_step = plan.N / distance;
      for (int j = 0; j < m; ++j) {
        record(stage, j, plan.W[j * tw_step]);
      }
    }
  }

 private:
  static std::size_t compute_total_twiddles(int stage_count) {
    if (stage_count <= 0) return 0;
    std::size_t total = 0;
    for (int i = 0; i < stage_count; ++i) {
      total += static_cast<std::size_t>(1) << i;
    }
    return total;
  }
};

template <class T>
struct LayoutMetadataWriter {
  const MetadataRequest<T>* requests = nullptr;
  int count = 0;
  std::complex<T>* layout_buffer = nullptr;
  std::size_t capacity = 0;
  std::size_t N = 0;

  LayoutMetadataWriter(const AxisLayout<T>& layout, std::size_t n)
      : requests(layout.metadata_requests),
        count(layout.metadata_request_count),
        N(n) {
    if (!requests || count <= 0 || N == 0) return;
    for (int i = 0; i < count; ++i) {
      const MetadataRequest<T>& req = requests[i];
      if (req.kind != MetadataKind::LayoutMap) continue;
      layout_buffer = req.complex_buffer;
      capacity = req.element_count;
      break;
    }
    if (layout_buffer && capacity < N) {
      throw std::invalid_argument("LayoutMap metadata buffer too small");
    }
  }

  bool enabled() const { return layout_buffer != nullptr && N > 0; }

  void set(std::size_t position, int source_index) const {
    if (!enabled() || position >= N) return;
    layout_buffer[position] = std::complex<T>(static_cast<T>(source_index), T(0));
  }
};

template <class T>
struct StageSnapshotWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  std::size_t N = 0;
  int stages = 0;

  StageSnapshotWriter(const AxisLayout<T>& layout, std::size_t n, int s)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count),
        N(n), stages(s) {
    if (!reqs || count <= 0 || N == 0 || stages <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::StageSnapshot) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < N * (std::size_t)stages)
          throw std::invalid_argument("StageSnapshot buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && N && stages; }
  void set(int stage, int pos, const std::complex<T>& v) const {
    if (enabled() && stage >= 0 && stage < stages && pos >= 0 && (std::size_t)pos < N)
      buf[(std::size_t)stage * N + (std::size_t)pos] = v;
  }
  void copy_lane0(int stage, const std::complex<T>* src, int stride /*=1*/) const {
    if (!enabled()) return;
    for (std::size_t i = 0; i < N; ++i) set(stage, (int)i, src[i * (std::size_t)stride]);
  }
};

template <class T>
struct StageParamsWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  int stages = 0;
  StageParamsWriter(const AxisLayout<T>& layout, int s)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), stages(s) {
    if (!reqs || count <= 0 || stages <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::StageParams) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < (std::size_t)stages)
          throw std::invalid_argument("StageParams buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && stages > 0; }
  void set(int stage, int distance, int tw_step) const {
    if (!enabled() || stage < 0 || stage >= stages) return;
    buf[stage] = std::complex<T>(static_cast<T>(distance), static_cast<T>(tw_step));
  }
};

template <class T>
struct ButterflyPairsWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  std::size_t N = 0;
  int stages = 0;
  ButterflyPairsWriter(const AxisLayout<T>& layout, std::size_t n, int s)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), buf(nullptr), N(n), stages(s) {
    if (!reqs || count <= 0 || N == 0 || stages <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::ButterflyPairs) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < N * (std::size_t)stages)
          throw std::invalid_argument("ButterflyPairs buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && N && stages; }
  void set(int stage, int pos, int lhs, int rhs) const {
    if (!enabled() || stage < 0 || stage >= stages || pos < 0 || (std::size_t)pos >= N) return;
    buf[(std::size_t)stage * N + (std::size_t)pos] = std::complex<T>(static_cast<T>(lhs), static_cast<T>(rhs));
  }
};

template <class T>
struct StageInvariantWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  int stages = 0;
  StageInvariantWriter(const AxisLayout<T>& layout, int s)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), buf(nullptr), stages(s) {
    if (!reqs || count <= 0 || stages <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::StageInvariant) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < (std::size_t)stages)
          throw std::invalid_argument("StageInvariant buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && stages > 0; }
  void set(int stage, T max_r0, T max_r1) const {
    if (!enabled() || stage < 0 || stage >= stages) return;
    buf[stage] = std::complex<T>(max_r0, max_r1);
  }
};

template <class T>
struct TwiddleIndexWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  std::size_t N = 0;
  int stages = 0;
  TwiddleIndexWriter(const AxisLayout<T>& layout, std::size_t n, int s)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), buf(nullptr), N(n), stages(s) {
    if (!reqs || count <= 0 || N == 0 || stages <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::TwiddleIndexMap) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < N * (std::size_t)stages)
          throw std::invalid_argument("TwiddleIndexMap buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && N && stages; }
  void set(int stage, int pos, int tw_index) const {
    if (!enabled() || stage < 0 || stage >= stages || pos < 0 || (std::size_t)pos >= N) return;
    buf[(std::size_t)stage * N + (std::size_t)pos] = std::complex<T>(static_cast<T>(tw_index), T(0));
  }
};

// New metadata writers
template <class T>
struct OmegaMetadataWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  std::size_t capacity = 0;
  std::size_t N = 0;

  OmegaMetadataWriter(const AxisLayout<T>& layout, std::size_t n)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), N(n) {
    if (!reqs || count <= 0 || N == 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::Omega) {
        buf = reqs[i].complex_buffer;
        capacity = reqs[i].element_count;
        break;
      }
    }
    if (buf && capacity < N) {
      // Require capacity for full N; caller may ignore upper bins if half-packed
      throw std::invalid_argument("Omega metadata buffer too small");
    }
  }
  bool enabled() const { return buf && N > 0; }
  void fill(const Plan<T>& plan) const {
    if (!enabled()) return;
    const std::size_t upper = (plan.transform_mode == Plan<T>::TransformMode::R2C && plan.half_spectrum)
                                  ? static_cast<std::size_t>(plan.N/2 + 1)
                                  : static_cast<std::size_t>(plan.N);
    const T two_pi = static_cast<T>(2 * std::acos(T(-1)));
    for (std::size_t k = 0; k < upper; ++k) {
      const T omega = two_pi * static_cast<T>(k) / static_cast<T>(plan.N);
      buf[k] = std::complex<T>(omega, T(0));
    }
  }
};

template <class T>
struct StageSnapshotAllWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  std::size_t N = 0;
  int stages = 0;
  int B = 0;
  StageSnapshotAllWriter(const AxisLayout<T>& layout, std::size_t n, int s, int batches)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), buf(nullptr), N(n), stages(s), B(batches) {
    if (!reqs || count <= 0 || N == 0 || stages <= 0 || B <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::StageSnapshotAll) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < N * static_cast<std::size_t>(stages) * static_cast<std::size_t>(B))
          throw std::invalid_argument("StageSnapshotAll buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && N && stages && B > 0; }
  void set(int stage, int pos, int batch, const std::complex<T>& v) const {
    if (!enabled()) return;
    if (stage < 0 || stage >= stages || pos < 0 || (std::size_t)pos >= N || batch < 0 || batch >= B) return;
    const std::size_t per_stage = N * static_cast<std::size_t>(B);
    buf[static_cast<std::size_t>(stage) * per_stage + static_cast<std::size_t>(batch) * N + static_cast<std::size_t>(pos)] = v;
  }
};

template <class T>
struct ButterflyPairsAllWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  std::size_t N = 0;
  int stages = 0;
  int B = 0;
  ButterflyPairsAllWriter(const AxisLayout<T>& layout, std::size_t n, int s, int batches)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), buf(nullptr), N(n), stages(s), B(batches) {
    if (!reqs || count <= 0 || N == 0 || stages <= 0 || B <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::ButterflyPairsAll) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < N * static_cast<std::size_t>(stages) * static_cast<std::size_t>(B))
          throw std::invalid_argument("ButterflyPairsAll buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && N && stages && B > 0; }
  void set(int stage, int pos, int batch, int lhs, int rhs) const {
    if (!enabled()) return;
    if (stage < 0 || stage >= stages || pos < 0 || (std::size_t)pos >= N || batch < 0 || batch >= B) return;
    const std::size_t per_stage = N * static_cast<std::size_t>(B);
    buf[static_cast<std::size_t>(stage) * per_stage + static_cast<std::size_t>(batch) * N + static_cast<std::size_t>(pos)] =
        std::complex<T>(static_cast<T>(lhs), static_cast<T>(rhs));
  }
};

template <class T>
struct TwiddleIndexAllWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  std::size_t N = 0;
  int stages = 0;
  int B = 0;
  TwiddleIndexAllWriter(const AxisLayout<T>& layout, std::size_t n, int s, int batches)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), buf(nullptr), N(n), stages(s), B(batches) {
    if (!reqs || count <= 0 || N == 0 || stages <= 0 || B <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::TwiddleIndexMapAll) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < N * static_cast<std::size_t>(stages) * static_cast<std::size_t>(B))
          throw std::invalid_argument("TwiddleIndexMapAll buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && N && stages && B > 0; }
  void set(int stage, int pos, int batch, int tw_index) const {
    if (!enabled()) return;
    if (stage < 0 || stage >= stages || pos < 0 || (std::size_t)pos >= N || batch < 0 || batch >= B) return;
    const std::size_t per_stage = N * static_cast<std::size_t>(B);
    buf[static_cast<std::size_t>(stage) * per_stage + static_cast<std::size_t>(batch) * N + static_cast<std::size_t>(pos)] =
        std::complex<T>(static_cast<T>(tw_index), T(0));
  }
};

}  // namespace detail

enum class KernelKind { CooleyTukey, Stockham, External };

enum class KernelAccuracy { Default, HighPrecision, Reference };

namespace detail {

template<class T>
void cooleytukey_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx);

template<class T>
std::unique_ptr<KernelContext> cooleytukey_create_state(Plan<T>& P);

template<class T>
void cooleytukey_destroy_state(Plan<T>& P, KernelContext* ctx);

template<class T>
void stockham_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx);

template<class T>
std::unique_ptr<KernelContext> stockham_create_state(Plan<T>& P);

template<class T>
void stockham_destroy_state(Plan<T>& P, KernelContext* ctx);

template<class T>
void external_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx);

template<class T>
std::unique_ptr<KernelContext> external_create_state(Plan<T>& P);

template<class T>
void external_destroy_state(Plan<T>& P, KernelContext* ctx);

} // namespace detail

template<class T> struct Plan {
  using Complex = std::complex<T>;
  using RowBuffer = Eigen::Matrix<Complex, 1, Eigen::Dynamic, Eigen::RowMajor>;
  using ArenaResizer = void (*)(PlanArena<T>&, int N, int threads, int lanes);
  struct KernelDescriptor;
  // Transform mode and reduction hooks (hot-path, in-place)
  enum class TransformMode { C2C = 0, R2C = 1, C2R = 2, R2R = 3 };

  int N;
  bool inverse;
  Complex* W = nullptr;      // twiddle factors (size N/2)
  int* bitrev = nullptr;     // bit-reversal indices (size N)
  int lgN;
  bool use_threads;
  int requested_threads;
  int packet_cols;  // auto from Eigen packets unless overridden
  // Selected transform mode and optional magnitude reduction flag.
  // - C2C: complex-to-complex (default)
  // - R2C: real-to-complex (imag cleared pre-butterfly)
  // - C2R: complex-to-real (imag cleared post-butterfly)
  // - R2R: real-to-real (imag cleared both pre and post)
  TransformMode transform_mode = TransformMode::C2C;
  bool reduce_magnitude = false; // if true, post-butterfly reduces Complex -> |Complex| in-place (imag=0)
  bool store_polar = false;      // if true, overwrite Complex as (mag, phase) interleaved: real=|z|, imag=arg(z)
  bool half_spectrum = false;    // if true (R2C/C2R), use half-spectrum packing (first N/2+1 bins valid)

  // Per-plan default butterfly configs. These are hot-path stable pointers
  // (no allocations) that kernels will reference during execution. Users may
  // override these on the Plan before dispatch to change butterfly behavior
  // (e.g. select DIF vs DIT, set conjugation). Defaults preserve legacy
  // behavior (Radix2_DIT, no conjugation).
  ButterflyConfig<T> butterfly_default_cooleytukey;
  ButterflyConfig<T> butterfly_default_stockham;
  ButterflyConfig<T> butterfly_default_external;
  // Optional per-stage radix pattern (e.g., {2,2,4}). If empty, the Plan
  // should use its butterfly_default_* values (uniform radix) during execution.
  std::vector<ButterflyRadix> radix_pattern;

  struct Limits {
    static constexpr int kCompileTimeMaxThreads = 16;
    static constexpr int kDefaultRuntimeThreads = 4;
    static constexpr int kDefaultLaneCapacity = 2;

    static constexpr int compile_time_max_lane_capacity() {
      if constexpr (std::is_same_v<T, float>) {
        return 8;  // AVX-512 holds 8 complex<float>; AVX2 fits within this bound.
      } else {
        return 4;  // AVX-512 holds 4 complex<double>; AVX2 fits within this bound.
      }
    }
  };

  enum class ParallelDim { Auto, Columns, KBlocks };
  enum class Schedule { Auto, Static, Dynamic, Guided };
  struct Tuning {
    ParallelDim parallel_dim = ParallelDim::Auto;
    Schedule schedule = Schedule::Auto;
    int packet_step = 0;            // 0 => auto (=packet_cols), else multiple thereof
    int min_work_per_thread = 64;   // heuristics gate
    bool force_ftz_daz = true;
    // Ingestion tuning: enable background ingestion/precompute of chunk
    // pointer arrays and optional backup copies for fault recovery.
    // Enabled by default to make transforms resilient to per-chunk crashes.
    bool use_ingest = true;
    bool ingest_backup = true;
    int ingest_queue_capacity = 64;
  } tuning;

  // Butterfly execution mode for Stockham: explicit, not implicit adapter.
  // - UseScatter: Stockham will call the out-of-place ButterflyScatter implementation.
  // - UseInplaceAdapter: Stockham will use PlanArena scratch and call an
  //   in-place kernel via the Inplace-to-Scatter adapter (compatibility fallback).
  enum class ButterflyMode { UseScatter = 0, UseInplaceAdapter = 1 };
  ButterflyMode butterfly_stockham_mode = ButterflyMode::UseScatter;

  struct Workspace {
    void bind(Plan& plan_ref) {
      owner = &plan_ref;
      arena = &plan_ref.arena;
    }

    void ensure(int threadCount, int cols) {
      if (!arena) {
        throw std::logic_error("Workspace not bound to PlanArena");
      }
      if (threadCount <= 0) threadCount = 1;
      if (cols <= 0) cols = 1;
      if ((cols > arena->baseline_lane_capacity ||
           threadCount > arena->baseline_thread_capacity) && owner && owner->arena_resizer) {
        owner->arena_resizer(*arena, owner->N, threadCount, cols);
      }
      const int clamped_threads = std::min(threadCount, arena->baseline_thread_capacity);
      const int clamped_cols = std::min(cols, arena->baseline_lane_capacity);
      threads = std::max(1, clamped_threads);
      capacity = std::max(1, clamped_cols);
    }

    Complex* row_a(int thread) const {
      return arena->baseline_a + static_cast<std::size_t>(thread) * arena->baseline_lane_capacity;
    }

    Complex* row_b(int thread) const {
      return arena->baseline_b + static_cast<std::size_t>(thread) * arena->baseline_lane_capacity;
    }

    Complex** column_row(int thread) const {
      return arena->baseline_columns + static_cast<std::size_t>(thread) * arena->baseline_lane_capacity;
    }

    void ensure_nd_buffer(Eigen::Index rows, Eigen::Index cols) {
      if (!arena) {
        throw std::logic_error("Workspace not bound to PlanArena");
      }
      if (rows <= 0 || cols <= 0) {
        nd_rows = nd_cols = 0;
        return;
      }
      const std::size_t required = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
      if (required > arena->nd_transpose_capacity) {
        if (owner && owner->arena_resizer) {
          owner->pending_nd_capacity = required;
          owner->arena_resizer(*arena, owner->N, std::max(1, threads), std::max(1, capacity));
        } else {
          throw std::invalid_argument("PlanArena nd_transpose capacity insufficient");
        }
      }
      if (required > arena->nd_transpose_capacity) {
        throw std::invalid_argument("PlanArena nd_transpose capacity insufficient after resize");
      }
      nd_rows = rows;
      nd_cols = cols;
      if (owner) {
        owner->pending_nd_capacity = 0;
      }
    }

    Complex* nd_buffer() const {
      return arena ? arena->nd_transpose : nullptr;
    }

    int capacity = 0;
    int threads = 0;
    Eigen::Index nd_rows = 0;
    Eigen::Index nd_cols = 0;

   private:
    PlanArena<T>* arena = nullptr;
    Plan* owner = nullptr;
  };

  mutable Workspace workspace;
  PlanArena<T>& arena;
  ArenaResizer arena_resizer = nullptr;
  mutable std::size_t pending_nd_capacity = 0;
  const KernelDescriptor* kernel_desc_ = nullptr;
  std::unique_ptr<KernelContext> kernel_state_;
  // Non-owning pointer to a dispatcher used by algorithms for optional job posting.
  JobDispatcher* dispatcher_ = &InlineDispatcher::instance();

  struct KernelDescriptor {
    KernelKind kind = KernelKind::CooleyTukey;
    const char* name = nullptr;
    KernelAccuracy accuracy = KernelAccuracy::Default;
    bool realtime_safe = true;
    using CreateStateFn = std::unique_ptr<KernelContext> (*)(Plan&);
    using ExecuteFn = void (*)(const Plan&, const AxisLayout<T>&, KernelContext*);
    using DestroyStateFn = void (*)(Plan&, KernelContext*);
    CreateStateFn create_state = nullptr;
    ExecuteFn execute_axis = nullptr;
    DestroyStateFn destroy_state = nullptr;
  };

  // Generic, algorithm-agnostic request for extra plan workspace resources.
  struct AdvancedWorkspaceRequest {
    int min_lane_capacity = 0;              // desired per-thread lane capacity
    int min_thread_capacity = 0;            // desired thread capacity
    std::size_t min_nd_transpose_capacity = 0; // desired ND transpose capacity (elements)
    bool prefer_inplace_emulation = false;  // hint for allocator
    // Special buffer requests: preferred_slot >= 0 to request a fixed slot.
    struct SpecialRequest {
      int preferred_slot = -1; // -1 = not set; caller should prefer explicit slot for speed
      std::size_t elements = 0; // number of Complex elements requested
      bool operator==(const SpecialRequest& o) const noexcept {
        return preferred_slot == o.preferred_slot && elements == o.elements;
      }
    };
    std::vector<SpecialRequest> special_requests;

    bool operator==(const AdvancedWorkspaceRequest& o) const noexcept {
      if (min_lane_capacity != o.min_lane_capacity) return false;
      if (min_thread_capacity != o.min_thread_capacity) return false;
      if (min_nd_transpose_capacity != o.min_nd_transpose_capacity) return false;
      if (prefer_inplace_emulation != o.prefer_inplace_emulation) return false;
      if (special_requests.size() != o.special_requests.size()) return false;
      for (size_t i = 0; i < special_requests.size(); ++i) {
        if (!(special_requests[i] == o.special_requests[i])) return false;
      }
      return true;
    }
  };

  // Cached fulfilled advanced requests for this Plan. Durable for the Plan lifetime.
  mutable std::vector<AdvancedWorkspaceRequest> cached_advanced_requests;
  // Pending per-slot special capacities requested prior to calling arena_resizer.
  mutable std::array<std::size_t, PlanArena<T>::kMaxSpecialBuffers> pending_special_capacity{};
  // Mutex to protect cached requests and pending_special_capacity during reservation.
  mutable std::mutex advanced_reserve_mutex;

  // Attempt to reserve advanced workspace described by 'req'. Returns true if
  // the Plan (PlanArena) meets the request (either already satisfied or after
  // invoking arena_resizer). On success the request is cached for future
  // fast-path checks. This is conservative and durable: requests remain cached
  // for the Plan lifetime. The method may call arena_resizer if available.
  bool reserve_advanced_workspace(const AdvancedWorkspaceRequest& req) const {
    // Fast-path: already cached
    {
      std::lock_guard<std::mutex> g(advanced_reserve_mutex);
      for (const auto& r : cached_advanced_requests) {
        if (r == req) return true;
      }
    }

    // Try to satisfy ND transpose capacity first.
    if (req.min_nd_transpose_capacity > arena.nd_transpose_capacity) {
      if (arena_resizer) {
        // Set pending request and ask resizer to grow arena.
        const_cast<Plan*>(this)->pending_nd_capacity = req.min_nd_transpose_capacity;
        arena_resizer(const_cast<PlanArena<T>&>(arena), N, std::max(1, requested_threads), std::max(1, req.min_lane_capacity));
      }
      if (arena.nd_transpose_capacity < req.min_nd_transpose_capacity) return false;
    }

    // Check lane/thread capacity. If neither baseline nor stockham capacities
    // meet the requested lane count, attempt a resize.
    const bool lane_ok = (arena.baseline_lane_capacity >= req.min_lane_capacity) || (arena.stockham_lane_capacity >= req.min_lane_capacity);
    const bool thread_ok = (arena.baseline_thread_capacity >= req.min_thread_capacity) || (arena.stockham_thread_capacity >= req.min_thread_capacity);
    if (!lane_ok || !thread_ok) {
      if (arena_resizer) {
        arena_resizer(const_cast<PlanArena<T>&>(arena), N, std::max(req.min_thread_capacity, requested_threads), std::max(req.min_lane_capacity, arena.baseline_lane_capacity));
      }
      const bool lane_ok2 = (arena.baseline_lane_capacity >= req.min_lane_capacity) || (arena.stockham_lane_capacity >= req.min_lane_capacity);
      const bool thread_ok2 = (arena.baseline_thread_capacity >= req.min_thread_capacity) || (arena.stockham_thread_capacity >= req.min_thread_capacity);
      if (!lane_ok2 || !thread_ok2) return false;
    }

    // Handle special buffer requests (preferred_slot must be >= 0 for now).
    if (!req.special_requests.empty()) {
      if (!arena_resizer) return false; // cannot satisfy special requests without resizer
      // Lock while we update pending_special_capacity
      {
        std::lock_guard<std::mutex> g(advanced_reserve_mutex);
        for (const auto& s : req.special_requests) {
          if (s.preferred_slot < 0 || s.preferred_slot >= PlanArena<T>::kMaxSpecialBuffers) return false;
          const int slot = s.preferred_slot;
          // If arena already has sufficient capacity, nothing to request.
          if (arena.special_capacity[slot] >= s.elements) continue;
          // Otherwise, set pending capacity to requested size (or max of existing pending).
          pending_special_capacity[slot] = std::max(pending_special_capacity[slot], s.elements);
        }
      }
      // Call resizer to attempt to satisfy pending special capacities.
      arena_resizer(const_cast<PlanArena<T>&>(arena), N, std::max(req.min_thread_capacity, requested_threads), std::max(req.min_lane_capacity, arena.baseline_lane_capacity));
      // Verify capacities after resize.
      for (const auto& s : req.special_requests) {
        const int slot = s.preferred_slot;
        if (arena.special_capacity[slot] < s.elements) return false;
      }
    }

    // Success: cache and return true.
    {
      std::lock_guard<std::mutex> g(advanced_reserve_mutex);
      const_cast<std::vector<AdvancedWorkspaceRequest>&>(cached_advanced_requests).push_back(req);
    }
    return true;
  }

  Plan(int n, PlanArena<T>& arena_ref, bool inv=false, bool threads=true, int max_threads=0)
      : N(n), inverse(inv), lgN(0), use_threads(threads),
        requested_threads(max_threads), packet_cols(1), arena(arena_ref) {
#if EIGFFT_DEBUG
    std::cout << "Plan constructor entered, N=" << N << std::endl;
#endif
    if (requested_threads <= 0) {
      requested_threads = Limits::kDefaultRuntimeThreads;
    }
    requested_threads = std::min(requested_threads, Limits::kCompileTimeMaxThreads);

    if (!use_threads && !detail::kAllowSequentialFallback) {
      throw std::invalid_argument("Sequential execution disabled: Plan requires thread-enabled dispatcher");
    }

    if (!arena.twiddles || !arena.bitrev) {
      throw std::invalid_argument("PlanArena must provide twiddle and bit-reversal buffers");
    }
    if (arena.twiddle_count < std::max(1, N / 2) || arena.bitrev_count < N) {
      throw std::invalid_argument("PlanArena buffers smaller than required for Plan");
    }

    W = arena.twiddles;
    bitrev = arena.bitrev;

    // power-of-two check
    int t = N;
    while ((t & 1) == 0) { ++lgN; t >>= 1; }
    if ((1 << lgN) != N) throw std::invalid_argument("N must be power of two");

    // twiddles
    const T sgn = inverse ? T(+1) : T(-1);
    const T tau = sgn * T(2 * std::acos(T(-1))) / T(N);
    for (int k = 0; k < N / 2; ++k) {
      const T ang = tau * T(k);
      const T re = static_cast<T>(std::cos(ang));
      const T im = static_cast<T>(std::sin(ang));
      W[k] = Complex(re, im);
    }

    // bit-reversal
    for (int i = 0; i < N; ++i) {
      unsigned x = static_cast<unsigned>(i);
      unsigned r = 0;
      for (int b = 0; b < lgN; ++b) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
      }
      bitrev[i] = static_cast<int>(r);
    }

    if (!arena.baseline_a || !arena.baseline_b || !arena.baseline_columns) {
      throw std::invalid_argument("PlanArena missing baseline workspace buffers");
    }
    if (arena.baseline_thread_capacity <= 0 ||
        arena.baseline_thread_capacity > Limits::kCompileTimeMaxThreads) {
      throw std::invalid_argument("PlanArena baseline_thread_capacity out of range");
    }
    if (arena.baseline_lane_capacity <= 0 ||
        arena.baseline_lane_capacity > Limits::compile_time_max_lane_capacity()) {
      throw std::invalid_argument("PlanArena baseline_lane_capacity out of range");
    }

    if (!arena.stockham_ping || !arena.stockham_pong || !arena.stockham_lane_bases) {
      throw std::invalid_argument("PlanArena missing Stockham scratch buffers");
    }
    if (arena.stockham_thread_capacity <= 0 ||
        arena.stockham_thread_capacity > Limits::kCompileTimeMaxThreads) {
      throw std::invalid_argument("PlanArena stockham_thread_capacity out of range");
    }
    if (arena.stockham_lane_capacity <= 0 ||
        arena.stockham_lane_capacity > Limits::compile_time_max_lane_capacity()) {
      throw std::invalid_argument("PlanArena stockham_lane_capacity out of range");
    }

    requested_threads = std::min(requested_threads, arena.stockham_thread_capacity);
    if (arena.baseline_thread_capacity < requested_threads) {
      throw std::invalid_argument("PlanArena baseline_thread_capacity smaller than requested thread count");
    }

  int packet = static_cast<int>(Eigen::internal::packet_traits<Complex>::size);
  if (packet <= 0) packet = 1;
  const int arena_lane_cap = std::max(1, arena.baseline_lane_capacity);
  packet_cols = std::max(1, std::min(packet, arena_lane_cap));
#if EIGFFT_DEBUG
    std::cout << "Plan using packet_cols=" << packet_cols << std::endl;
#endif

  // Warm the workspace to avoid hot-path checks during the first dispatch.
  workspace.bind(*this);
  const int warm_threads = effective_threads(packet_cols);
  workspace.ensure(warm_threads, packet_cols);
    select_kernel(cooleytukey_kernel());
  // initialize butterfly defaults per algorithm (no allocations in hot path)
  butterfly_default_cooleytukey.radix = ButterflyRadix::Radix2;
  butterfly_default_cooleytukey.method = ButterflyMethod::DIT;
  butterfly_default_cooleytukey.tw_place = TwiddlePlacement::PreRHS;
  butterfly_default_cooleytukey.forward = true;
  butterfly_default_cooleytukey.conjugate_tw = false;
  butterfly_default_cooleytukey.simd_width = 1;

  butterfly_default_stockham = butterfly_default_cooleytukey;
  // Stockham execution reuses the DIT scatter math; ensure the default
  // butterfly method matches the layout expected by the Stockham index math.
  butterfly_default_stockham.method = ButterflyMethod::DIT;

  butterfly_default_external = butterfly_default_cooleytukey;
  }

  ~Plan() {
    release_kernel();
  }

  // Optional: override SIMD coarsening (e.g., 2× packet for fat batches)
  void set_packet_step(int step) {
    tuning.packet_step = step;
  }

  // Enable/disable FTZ/DAZ at call sites
  static void set_ftz_daz(bool on) {
    // #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    //   _MM_SET_FLUSH_ZERO_MODE(on ? _MM_FLUSH_ZERO_ON : _MM_FLUSH_ZERO_OFF);
    //   _MM_SET_DENORMALS_ZERO_MODE(on ? _MM_DENORMALS_ZERO_ON : _MM_DENORMALS_ZERO_OFF);
    // #else
    (void)on;
    // #endif
  }

  int effective_threads(int /*batch_cols*/) const {
    if (!use_threads) return 1;
    int limit = requested_threads;
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 1;
    const int runtime_cap = hw;
    if (limit <= 0) limit = runtime_cap; else limit = std::min(limit, runtime_cap);
    limit = std::min(limit, Limits::kCompileTimeMaxThreads);
    limit = std::min(limit, arena.stockham_thread_capacity);
    return std::max(1, limit);
  }

  void ensure_workspace(int threads, int cols) const {
    workspace.ensure(threads, cols);
  }

  void ensure_nd_workspace(Eigen::Index rows, Eigen::Index cols) const {
    workspace.ensure_nd_buffer(rows, cols);
  }

  Complex* transpose_buffer_data() const {
    return workspace.nd_buffer();
  }

  Eigen::Index transpose_rows() const { return workspace.nd_rows; }
  Eigen::Index transpose_cols() const { return workspace.nd_cols; }

  void set_arena_resizer(ArenaResizer r) { arena_resizer = r; }

  void select_kernel(const KernelDescriptor& descriptor) {
    if (kernel_desc_ == &descriptor) return;
    release_kernel();
    kernel_desc_ = &descriptor;
    if (kernel_desc_ && kernel_desc_->create_state) {
      kernel_state_ = kernel_desc_->create_state(*this);
    } else {
      kernel_state_.reset();
    }
  }

  void set_dispatcher(JobDispatcher* d) { dispatcher_ = d ? d : &InlineDispatcher::instance(); }
  JobDispatcher* dispatcher() const { return dispatcher_ ? dispatcher_ : const_cast<InlineDispatcher*>(&InlineDispatcher::instance()); }

  const KernelDescriptor& kernel() const {
    return kernel_desc_ ? *kernel_desc_ : cooleytukey_kernel();
  }

  bool use_kernel(KernelKind kind) {
    if (const KernelDescriptor* desc = find_kernel(kind)) {
      select_kernel(*desc);
      return true;
    }
    return false;
  }

  bool use_kernel(std::string_view name) {
    if (const KernelDescriptor* desc = find_kernel(name)) {
      select_kernel(*desc);
      return true;
    }
    return false;
  }

  bool kernel_realtime_safe() const { return kernel().realtime_safe; }
  KernelAccuracy kernel_accuracy() const { return kernel().accuracy; }

  KernelContext* kernel_state() const {
    return kernel_state_.get();
  }

  static const KernelDescriptor& cooleytukey_kernel();
  static const KernelDescriptor& stockham_kernel();
  static const KernelDescriptor& external_kernel();
  static const std::array<const KernelDescriptor*, 3>& builtin_kernels();

  static constexpr bool has_external_kernel() {
    return detail::kExternalKernelAvailable;
  }

 private:
  void release_kernel() {
    if (kernel_desc_ && kernel_desc_->destroy_state && kernel_state_) {
      kernel_desc_->destroy_state(*this, kernel_state_.get());
    }
    kernel_state_.reset();
    kernel_desc_ = nullptr;
  }

  const KernelDescriptor* find_kernel(KernelKind kind) const {
    switch (kind) {
      case KernelKind::CooleyTukey:
        return &cooleytukey_kernel();
      case KernelKind::Stockham:
        return &stockham_kernel();
      case KernelKind::External:
        if (detail::kExternalKernelAvailable) {
          return &external_kernel();
        }
        return nullptr;
    }
    return nullptr;
  }

  const KernelDescriptor* find_kernel(std::string_view name) const {
    const auto& all = builtin_kernels();
    for (const auto* desc : all) {
      if (desc && desc->name && name == desc->name) {
        return desc;
      }
    }
    return nullptr;
  }
};

namespace detail {

template<class T>
inline void tiled_transpose_colmajor(const std::complex<T>* src, Eigen::Index rows,
                                     Eigen::Index cols, std::complex<T>* dst,
                                     Eigen::Index tile_rows, Eigen::Index tile_cols)
{
  if (rows <= 0 || cols <= 0 || src == nullptr || dst == nullptr) return;
  if (tile_rows <= 0) tile_rows = rows;
  if (tile_cols <= 0) tile_cols = cols;
  const Eigen::Index dest_rows = cols;

  for (Eigen::Index c0 = 0; c0 < cols; c0 += tile_cols) {
    const Eigen::Index width = std::min(tile_cols, cols - c0);
    for (Eigen::Index r0 = 0; r0 < rows; r0 += tile_rows) {
      const Eigen::Index height = std::min(tile_rows, rows - r0);
      for (Eigen::Index j = 0; j < width; ++j) {
        const std::complex<T>* src_col = src + (c0 + j) * rows + r0;
        for (Eigen::Index i = 0; i < height; ++i) {
          const Eigen::Index row = c0 + j;
          const Eigen::Index col = r0 + i;
          dst[row + col * dest_rows] = src_col[i];
        }
      }
    }
  }
}

// Shared input validation for FFT axis execution (hot-path small inline)
template<class T>
inline void validate_fft_axis(const Plan<T>& P, const AxisLayout<T>& layout) {
  if (!layout.valid())
    throw std::invalid_argument("AxisLayout must reference valid data.");
  if (layout.axis_size != P.N)
    throw std::invalid_argument("AxisLayout axis_size must match Plan::N.");
}

// In-place pre/post transform hooks (hot path, no allocations)
template<class T>
inline void apply_pre_transform(const Plan<T>& P, const AxisLayout<T>& layout) {
  using Complex = std::complex<T>;
  const auto mode = P.transform_mode;
  const int N = P.N;
  const int B = static_cast<int>(layout.batch_size);
  const Eigen::Index axis_stride = layout.axis_stride;
  const Eigen::Index batch_stride = layout.batch_stride;

  if (mode == Plan<T>::TransformMode::R2C) {
    // Ensure input imaginary parts are zero; not strictly required but stabilizes results
    for (int col = 0; col < B; ++col) {
      Complex* base = layout.base + Eigen::Index(col) * batch_stride;
      for (int i = 0; i < N; ++i) {
        Complex& v = base[Eigen::Index(i) * axis_stride];
        if (v.imag() != T(0)) v.imag(T(0));
      }
    }
    return;
  }

  if (mode == Plan<T>::TransformMode::C2R) {
    // If half-spectrum packing is enabled, expand bins 0..N/2 into full Hermitian spectrum
    if (P.half_spectrum) {
      for (int col = 0; col < B; ++col) {
        Complex* base = layout.base + Eigen::Index(col) * batch_stride;
        // If stored as polar, convert back to complex
        if (P.store_polar) {
          for (int k = 0; k <= N/2; ++k) {
            const Eigen::Index k_off = Eigen::Index(k) * axis_stride;
            const Complex pv = base[k_off];
            const T mag = pv.real();
            const T ph = pv.imag();
            base[k_off] = Complex(mag * std::cos(ph), mag * std::sin(ph));
          }
        }
        // Enforce real DC and Nyquist
        base[0].imag(T(0));
        base[Eigen::Index(N/2) * axis_stride].imag(T(0));
        // Mirror to fill full spectrum
        for (int k = 1; k < N/2; ++k) {
          const Eigen::Index k_off = Eigen::Index(k) * axis_stride;
          const Eigen::Index nk_off = Eigen::Index(N - k) * axis_stride;
          base[nk_off] = std::conj(base[k_off]);
        }
      }
    }
    return;
  }
}

template<class T>
inline void apply_post_transform(const Plan<T>& P, const AxisLayout<T>& layout) {
  using Complex = std::complex<T>;
  const auto mode = P.transform_mode;
  const int N = P.N;
  const int B = static_cast<int>(layout.batch_size);
  const Eigen::Index axis_stride = layout.axis_stride;
  const Eigen::Index batch_stride = layout.batch_stride;

  if (mode == Plan<T>::TransformMode::R2C && P.half_spectrum) {
    // Ensure DC and Nyquist are real and zero the redundant half to make packing explicit
    for (int col = 0; col < B; ++col) {
      Complex* base = layout.base + Eigen::Index(col) * batch_stride;
      base[0].imag(T(0));
      base[Eigen::Index(N/2) * axis_stride].imag(T(0));
      for (int k = N/2 + 1; k < N; ++k) {
        base[Eigen::Index(k) * axis_stride] = Complex(T(0), T(0));
      }
    }
  }

  // Polar transform requested: overwrite as (mag, phase)
  if (P.store_polar) {
    for (int col = 0; col < B; ++col) {
      Complex* base = layout.base + Eigen::Index(col) * batch_stride;
      const int upper = (mode == Plan<T>::TransformMode::R2C && P.half_spectrum) ? (N/2 + 1) : N;
      for (int i = 0; i < upper; ++i) {
        Complex& v = base[Eigen::Index(i) * axis_stride];
        const T mag = std::abs(v);
        const T ph = std::arg(v);
        v.real(mag);
        v.imag(ph);
      }
    }
    return;
  }
  // Magnitude-only reduction
  if (P.reduce_magnitude) {
    for (int col = 0; col < B; ++col) {
      Complex* base = layout.base + Eigen::Index(col) * batch_stride;
      const int upper = (mode == Plan<T>::TransformMode::R2C && P.half_spectrum) ? (N/2 + 1) : N;
      for (int i = 0; i < upper; ++i) {
        Complex& v = base[Eigen::Index(i) * axis_stride];
        const T mag = std::abs(v);
        v.real(mag);
        v.imag(T(0));
      }
    }
    return;
  }

  if (mode == Plan<T>::TransformMode::C2R) {
    // Ensure small imaginary drift is cleared
    for (int col = 0; col < B; ++col) {
      Complex* base = layout.base + Eigen::Index(col) * batch_stride;
      for (int i = 0; i < N; ++i) {
        Complex& v = base[Eigen::Index(i) * axis_stride];
        if (v.imag() != T(0)) v.imag(T(0));
      }
    }
  }
}

// Unified, inline tracing helper for layout/stride information used by
// multiple algorithms. Guarded by EIGFFT_TRACE_LAYOUT so it can be
// enabled in one place to produce consistent prints for different kernels.
template<class T>
inline void trace_layout_info(const char* alg,
                              const AxisLayout<T>& layout,
                              Eigen::Index axis_stride,
                              Eigen::Index batch_stride,
                              int lane_cols,
                              int B,
                              int lane_capacity = -1) {
#if EIGFFT_TRACE_LAYOUT
  {
    std::ostringstream _oss;
    _oss << "[" << alg << "] axis_stride=" << axis_stride
         << " batch_stride=" << batch_stride;
    if (lane_capacity >= 0) _oss << " lane_capacity=" << lane_capacity;
    _oss << " lane_cols(final)=" << lane_cols;
    TraceLogger::instance().log(_oss.str());
  }
  for (int bi = 0; bi < std::min(3, B); ++bi) {
    std::ostringstream _oss2;
    const void* ptr = static_cast<const void*>(layout.base + static_cast<std::size_t>(bi) * (std::size_t)batch_stride);
    _oss2 << "[" << alg << "] base[" << bi << "]=" << ptr;
    TraceLogger::instance().log(_oss2.str());
  }
#else
  (void)alg; (void)layout; (void)axis_stride; (void)batch_stride; (void)lane_cols; (void)B; (void)lane_capacity;
#endif
}

// Prepare Plan workspace and compute lane/stride values used by execution paths.
template<class T>
inline void prepare_plan_workspace(const Plan<T>& P, const AxisLayout<T>& layout,
                                   int &out_B, int &out_threads,
                                   int &out_active_lane_cols, int &out_lane_cols,
                                   Eigen::Index &out_axis_stride, Eigen::Index &out_batch_stride,
                                   const typename Plan<T>::AdvancedWorkspaceRequest* adv_req = nullptr,
                                   bool* out_has_advanced_alloc = nullptr)
{
  out_B = static_cast<int>(layout.batch_size);
  out_threads = P.effective_threads(out_B);
  const int requested_lanes = (P.tuning.packet_step > 0) ? P.tuning.packet_step : P.packet_cols;
  out_lane_cols = std::max(1, requested_lanes);
  P.ensure_workspace(out_threads, out_lane_cols);
  out_active_lane_cols = std::max(1, std::min(out_lane_cols, P.workspace.capacity));
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(true);
  out_axis_stride = layout.axis_stride;
  out_batch_stride = layout.batch_stride;
  if (adv_req) {
    const bool ok = P.reserve_advanced_workspace(*adv_req);
    if (out_has_advanced_alloc) *out_has_advanced_alloc = ok;
  } else if (out_has_advanced_alloc) {
    *out_has_advanced_alloc = false;
  }
}

template<class T>
inline void cooleytukey_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx)
{
  (void)ctx;
  using Complex = typename Plan<T>::Complex;

  const int N = P.N;
  const int stages = P.lgN;
  validate_fft_axis(P, layout);
  const detail::TwiddleMetadataWriter<T> metadata(layout, stages);
  metadata.record_plan_twiddles(P);
  const detail::OmegaMetadataWriter<T> omega(layout, static_cast<std::size_t>(N));
  omega.fill(P);
  const detail::CooleyTukeyDebugConfig& ct_debug = detail::cooleytukey_debug_config();

  // Pre-butterfly transform hook (e.g., R2C imag clear)
  detail::apply_pre_transform(P, layout);
  const detail::LayoutMetadataWriter<T> layout_writer(layout, static_cast<std::size_t>(N));
  if (layout_writer.enabled()) {
    // Cooley–Tukey kernel produces output in natural order because it
    // bit-reverses the input before the butterfly stages. To keep the
    // metadata consistent with the actual output (and aligned with the
    // Stockham autosort kernel), record an identity mapping: each
    // output position maps to itself.
    for (int pos = 0; pos < N; ++pos) {
      layout_writer.set(static_cast<std::size_t>(pos), pos);
    }
  }
  const detail::StageSnapshotWriter<T> snap(layout, static_cast<std::size_t>(N), stages);
  const detail::StageParamsWriter<T> params(layout, stages);
  const detail::ButterflyPairsWriter<T> pairs(layout, static_cast<std::size_t>(N), stages);
  const detail::StageInvariantWriter<T> invariant(layout, stages);
  const detail::TwiddleIndexWriter<T> twmap(layout, static_cast<std::size_t>(N), stages);
  // Initialize invariant buffer to zero if enabled
  if (invariant.enabled()) {
    for (int s = 0; s < stages; ++s) invariant.set(s, T(0), T(0));
  }
  int B, threads, lane_cols, active_lane_cols;
  Eigen::Index axis_stride, batch_stride;
  prepare_plan_workspace(P, layout, B, threads, active_lane_cols, lane_cols, axis_stride, batch_stride);
  const detail::StageSnapshotAllWriter<T> snap_all(layout, static_cast<std::size_t>(N), stages, B);
  const detail::ButterflyPairsAllWriter<T> pairs_all(layout, static_cast<std::size_t>(N), stages, B);
  const detail::TwiddleIndexAllWriter<T> twmap_all(layout, static_cast<std::size_t>(N), stages, B);
  const int workspace_threads = std::max(1, P.workspace.threads);

  const int invariant_slots = workspace_threads;
  std::vector<std::complex<T>> invariant_scratch;
  if (invariant.enabled() && stages > 0) {
    invariant_scratch.assign(static_cast<size_t>(invariant_slots) * static_cast<size_t>(stages),
                             std::complex<T>(T(0), T(0)));
  }

  const int chunk_count = (B + active_lane_cols - 1) / active_lane_cols;
  const bool debug_chunks = (std::getenv("FFTFREE_DEBUG_CHUNKS") != nullptr);
  std::vector<int> permute_counts;
  std::vector<int> stage_counts;
  std::vector<int> permute_worker;
  std::vector<int> stage_worker;
  std::mutex debug_mutex;
  if (debug_chunks) {
    std::lock_guard<std::mutex> lk(debug_mutex);
    std::cerr << "[debug] workspace B=" << B
              << " threads=" << threads
              << " active_lane_cols=" << active_lane_cols
              << " chunk_count=" << chunk_count
              << " axis_stride=" << axis_stride
              << " batch_stride=" << batch_stride
              << "\n";
    permute_counts.assign(B, 0);
    stage_counts.assign(B, 0);
    permute_worker.assign(B, -1);
    stage_worker.assign(B, -1);
  }
  // Note: background ingestion/precompute removed. Column pointer arrays are
  // computed inline by worker tasks. This avoids detached threads and ensures
  // all work is scheduled via the Plan dispatcher / worker pool.
  // Unified trace for layout/stride info (no-op unless EIGFFT_TRACE_LAYOUT=1)
  detail::trace_layout_info<T>("cooleytukey", layout, axis_stride, batch_stride, static_cast<int>(active_lane_cols), B);
#if EIGFFT_TRACE_THREADS
  {
    std::ostringstream _oss;
    _oss << "[ct] workspace: B=" << B
         << " threads=" << threads
         << " lane_cols(requested)=" << lane_cols
         << " lane_cols(active)=" << active_lane_cols
         << " chunks=" << chunk_count
         << " axis_stride=" << axis_stride
         << " batch_stride=" << batch_stride;
    TraceLogger::instance().log(_oss.str());
  }
  TraceLogger::instance().log(std::string("[ct] permute begin: chunks=") + std::to_string(chunk_count) + std::string(" lane_cols=") + std::to_string(active_lane_cols));
#endif
  const int* bitrev = P.bitrev;

#if EIGFFT_TIMING
  timing_internal::ScopedTimer __t_pre_permute(timing_internal::Bin::PreButterfly);
#endif
  const bool enable_recovery = (P.dispatcher() != &InlineDispatcher::instance()) && chunk_count > 0;
  detail::ColumnRecoveryJobContext<T> permute_recovery{};
  if (enable_recovery) {
    permute_recovery.base = layout.base;
    permute_recovery.axis_stride = axis_stride;
    permute_recovery.batch_stride = batch_stride;
    permute_recovery.rows = N;
    permute_recovery.batch = B;
    permute_recovery.lane_cols = active_lane_cols;
  }
  const RecoveryOps permute_ops = enable_recovery ? make_column_recovery_ops<T>(permute_recovery)
                                                  : RecoveryOps{};
  // Optional runtime tracking for permute access conflicts and OOBs.
#if defined(EIGFFT_DEBUG_PERMUTE)
  const bool debug_permute_track = (std::getenv("FFTFREE_DEBUG_PERMUTE_TRACK") != nullptr);
  // Track map keyed by byte address -> owner worker id. Protected by mutex.
  static std::mutex permute_track_mutex;
  static std::unordered_map<std::uintptr_t, int> permute_track_map;
#else
  // When compile-time guard is disabled, keep a cheap false constant so
  // any remaining runtime checks (should be none) are dead-code.
  const bool debug_permute_track = false;
#endif

  {
    auto permute_fn = [&](size_t start, size_t end, int worker_id) {
      constexpr size_t kMaxLaneColumns =
          static_cast<size_t>(Plan<T>::Limits::compile_time_max_lane_capacity());
      std::array<Complex*, kMaxLaneColumns> columns{};
      for (size_t chunk = start; chunk < end; ++chunk) {
        const int col = static_cast<int>(chunk) * active_lane_cols;
        const int width = std::min(active_lane_cols, B - col);
#if EIGFFT_TRACE_THREADS
  {
    std::ostringstream _oss;
    _oss << "[ct] permute chunk worker=" << worker_id
         << " chunk_idx=" << chunk
         << " col0=" << col
         << " width=" << width;
    TraceLogger::instance().log(_oss.str());
  }
#endif
        if (width <= 0) continue;
        // Compute column pointers inline (no background ingestion).
        for (int lane = 0; lane < width; ++lane) {
          const Eigen::Index batch_index = Eigen::Index(col + lane);
          columns[static_cast<size_t>(lane)] = layout.base + batch_index * batch_stride;
        }
        if (debug_chunks) {
          std::lock_guard<std::mutex> lk(debug_mutex);
          for (int lane = 0; lane < width; ++lane) {
            ++permute_counts[col + lane];
            permute_worker[col + lane] = worker_id;
          }
        }
        for (int i = 0; i < N; ++i) {
          const int src = bitrev[i];
          if (src <= i) continue;
          const Eigen::Index dst_offset = Eigen::Index(i) * axis_stride;
          const Eigen::Index src_offset = Eigen::Index(src) * axis_stride;
          for (int lane = 0; lane < width; ++lane) {
            Complex* column_ptr = columns[static_cast<size_t>(lane)];
#if defined(EIGFFT_DEBUG_PERMUTE)
            // Optional conflict/OOB checks when debugging permute.
            if (debug_permute_track) {
              const char* base_ptr = reinterpret_cast<const char*>(layout.base);
              const std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(base_ptr);
              const char* dst_ptr = reinterpret_cast<const char*>(column_ptr + dst_offset);
              const char* src_ptr = reinterpret_cast<const char*>(column_ptr + src_offset);
              const std::uintptr_t dst_addr = reinterpret_cast<std::uintptr_t>(dst_ptr);
              const std::uintptr_t src_addr = reinterpret_cast<std::uintptr_t>(src_ptr);
              // Compute max valid byte address for quick OOB detection.
              const ptrdiff_t max_elem_offset = (layout.batch_size > 0 && layout.axis_size > 0)
                                                     ? (ptrdiff_t(layout.batch_size - 1) * ptrdiff_t(batch_stride) +
                                                        ptrdiff_t(layout.axis_size - 1) * ptrdiff_t(axis_stride))
                                                     : ptrdiff_t(0);
              const std::uintptr_t max_addr = base_addr + static_cast<std::uintptr_t>(max_elem_offset) * sizeof(Complex);
              if (dst_addr < base_addr || dst_addr > max_addr || src_addr < base_addr || src_addr > max_addr) {
                std::ostringstream _oss;
                _oss << "[permute-debug] OOB detected worker=" << worker_id
                     << " dst_addr=" << std::hex << dst_addr << " src_addr=" << src_addr << std::dec
                     << " base=" << reinterpret_cast<const void*>(base_ptr)
                     << " max_offset=" << max_elem_offset;
                std::cerr << _oss.str() << std::endl;
              } else {
                // Record ownership and detect if another worker already touched this address.
                {
                  std::lock_guard<std::mutex> g(permute_track_mutex);
                  auto it = permute_track_map.find(dst_addr);
                  if (it != permute_track_map.end() && it->second != worker_id) {
                    std::ostringstream _oss;
                    _oss << "[permute-debug] conflict dst addr " << std::hex << dst_addr << std::dec
                         << " prev_worker=" << it->second << " curr_worker=" << worker_id
                         << " i=" << i << " lane=" << lane << " col=" << col;
                    std::cerr << _oss.str() << std::endl;
                  }
                  permute_track_map[dst_addr] = worker_id;
                }
                {
                  std::lock_guard<std::mutex> g(permute_track_mutex);
                  auto it2 = permute_track_map.find(src_addr);
                  if (it2 != permute_track_map.end() && it2->second != worker_id) {
                    std::ostringstream _oss;
                    _oss << "[permute-debug] conflict src addr " << std::hex << src_addr << std::dec
                         << " prev_worker=" << it2->second << " curr_worker=" << worker_id
                         << " src=" << src << " i=" << i << " lane=" << lane << " col=" << col;
                    std::cerr << _oss.str() << std::endl;
                  }
                  permute_track_map[src_addr] = worker_id;
                }
              }
            }
#endif
            std::swap(column_ptr[dst_offset], column_ptr[src_offset]);
          }
        }
      }
    };
    // Dispatch permute work via either the plan dispatcher or a forced inline override.
    JobDispatcher* permute_dispatcher = P.dispatcher();
    if (ct_debug.force_inline_permute && !ct_debug.preserve_outer_dispatch) {
  permute_dispatcher = &InlineDispatcher::instance();
#if EIGFFT_TRACE_THREADS
  TraceLogger::instance().log(std::string("[ct] permute forced inline"));
#endif
    }
    if (permute_dispatcher == &InlineDispatcher::instance() && !enable_recovery) {
      // The production single-thread path is deliberately allocation-free:
      // bypass JobDispatcher's type-erased std::function ABI.
      permute_fn(0, static_cast<size_t>(chunk_count), 0);
    } else {
      permute_dispatcher->parallel_for_with_restore(static_cast<size_t>(chunk_count), 1, permute_fn,
                    std::function<void(size_t,size_t)>(),
                    permute_ops,
                    enable_recovery ? static_cast<void*>(&permute_recovery) : nullptr);
    }
  }
  
#if EIGFFT_TRACE_THREADS
  TraceLogger::instance().log(std::string("[ct] permute end"));
#endif

  
#if EIGFFT_TIMING
  timing_internal::ScopedTimer __t_butterfly(timing_internal::Bin::ButterflyStage);
#endif
  detail::ColumnRecoveryJobContext<T> stage_recovery{};
  const RecoveryOps stage_ops = [&]() {
    if (!enable_recovery) return RecoveryOps{};
    stage_recovery.base = layout.base;
    stage_recovery.axis_stride = axis_stride;
    stage_recovery.batch_stride = batch_stride;
    stage_recovery.rows = N;
    stage_recovery.batch = B;
    stage_recovery.lane_cols = active_lane_cols;
    return make_column_recovery_ops<T>(stage_recovery);
  }();

  for (int len = 2, stage_idx = 0; len <= N; len <<= 1, ++stage_idx) {
    const int half = len >> 1;
    const int step = N / len;
    const int blocks = N / len;
    if (params.enabled()) params.set(stage_idx, len, step);

    const bool dispatcher_inline = (P.dispatcher() == &InlineDispatcher::instance());
    std::complex<T>* invariant_global = (!invariant_scratch.empty() && dispatcher_inline)
                                            ? invariant_scratch.data()
                                            : nullptr;
    const bool capture_snap_all = snap_all.enabled();
    const bool capture_snap_single = snap.enabled();
    const bool capture_pairs_all = pairs_all.enabled();
    const bool capture_pairs_single = pairs.enabled();
    const bool capture_twmap_all = twmap_all.enabled();
    const bool capture_twmap_single = twmap.enabled();

    auto stage_chunk = [&](size_t start, size_t end, int worker_id) {
      constexpr size_t kMaxLaneColumns =
          static_cast<size_t>(Plan<T>::Limits::compile_time_max_lane_capacity());
      std::array<Complex*, kMaxLaneColumns> columns{};
      std::array<Complex, kMaxLaneColumns> a_buf{};
      std::array<Complex, kMaxLaneColumns> b_buf{};
      std::array<Complex, kMaxLaneColumns> a_orig{};
      std::array<Complex, kMaxLaneColumns> b_orig{};
      std::complex<T>* invariant_local = invariant_global;
      for (size_t chunk = start; chunk < end; ++chunk) {
        const int col = static_cast<int>(chunk) * active_lane_cols;
        const int width = std::min(active_lane_cols, B - col);
#if EIGFFT_TRACE_THREADS
  if (len == N) {
    std::ostringstream _oss;
    _oss << "[ct] stage chunk worker=" << worker_id
         << " len=" << len
         << " chunk_idx=" << chunk
         << " col0=" << col
         << " width=" << width;
    TraceLogger::instance().log(_oss.str());
  }
#endif
        if (width <= 0) continue;
        // Compute column pointers inline (no background ingestion).
        for (int lane = 0; lane < width; ++lane) {
          const Eigen::Index batch_index = Eigen::Index(col + lane);
          columns[static_cast<size_t>(lane)] = layout.base + batch_index * batch_stride;
        }
        if (debug_chunks) {
          std::lock_guard<std::mutex> lk(debug_mutex);
          for (int lane = 0; lane < width; ++lane) {
            ++stage_counts[col + lane];
            stage_worker[col + lane] = worker_id;
          }
        }
        for (int k = 0; k < half; ++k) {
          const Complex w = P.W[k * step];
          for (int block = 0; block < blocks; ++block) {
            const int base = block * len;
            const int a_index = base + k;
            const int b_index = a_index + half;
            const Eigen::Index a_offset = Eigen::Index(a_index) * axis_stride;
            const Eigen::Index b_offset = Eigen::Index(b_index) * axis_stride;
            for (int lane = 0; lane < width; ++lane) {
              Complex* column_ptr = columns[static_cast<size_t>(lane)];
              const Complex val_a = column_ptr[a_offset];
              const Complex val_b = column_ptr[b_offset];
              a_orig[static_cast<size_t>(lane)] = val_a;
              b_orig[static_cast<size_t>(lane)] = val_b;
              a_buf[static_cast<size_t>(lane)] = val_a;
              b_buf[static_cast<size_t>(lane)] = val_b;
            }
            detail::ButterflyKernel<T>::apply(a_buf.data(), b_buf.data(), width, w, &P.butterfly_default_cooleytukey);
            T local_max_r0 = invariant_local ? invariant_local[stage_idx].real() : T(0);
            T local_max_e = invariant_local ? invariant_local[stage_idx].imag() : T(0);
            const int tw_index = k * step;
            for (int lane = 0; lane < width; ++lane) {
              Complex* column_ptr = columns[static_cast<size_t>(lane)];
              const Complex y0 = a_buf[static_cast<size_t>(lane)];
              const Complex y1 = b_buf[static_cast<size_t>(lane)];
              column_ptr[a_offset] = y0;
              column_ptr[b_offset] = y1;
              const int batch_index = col + lane;
              if (capture_snap_all) {
                snap_all.set(stage_idx, a_index, batch_index, y0);
                snap_all.set(stage_idx, b_index, batch_index, y1);
              } else if (capture_snap_single && batch_index == 0) {
                snap.set(stage_idx, a_index, y0);
                snap.set(stage_idx, b_index, y1);
              }
              if (capture_twmap_all) {
                twmap_all.set(stage_idx, a_index, batch_index, tw_index);
                twmap_all.set(stage_idx, b_index, batch_index, tw_index);
              } else if (capture_twmap_single && batch_index == 0) {
                twmap.set(stage_idx, a_index, tw_index);
                twmap.set(stage_idx, b_index, tw_index);
              }
              if (invariant_local) {
                const Complex a_in = a_orig[static_cast<size_t>(lane)];
                const Complex b_in = b_orig[static_cast<size_t>(lane)];
                const T r0 = std::abs((y0 + y1) - T(2) * a_in);
                const T e = (std::norm(y0) + std::norm(y1)) - T(2) * (std::norm(a_in) + std::norm(b_in));
                const T abs_e = std::abs(e);
                if (r0 > local_max_r0) local_max_r0 = r0;
                if (abs_e > local_max_e) local_max_e = abs_e;
              }
              if (capture_pairs_all) {
                pairs_all.set(stage_idx, a_index, batch_index, a_index, b_index);
                pairs_all.set(stage_idx, b_index, batch_index, a_index, b_index);
              } else if (capture_pairs_single && batch_index == 0) {
                pairs.set(stage_idx, a_index, a_index, b_index);
                pairs.set(stage_idx, b_index, a_index, b_index);
              }
            }
            if (invariant_local) {
              invariant_local[stage_idx] = std::complex<T>(local_max_r0, local_max_e);
            }
          }
        }
      }
    };
#if EIGFFT_TRACE_THREADS
    {
      std::ostringstream _oss;
      _oss << "[ct] stage begin: len=" << len
           << " half=" << half
           << " blocks=" << blocks
           << " chunks=" << chunk_count;
      TraceLogger::instance().log(_oss.str());
    }
#endif
    // Dispatch stage work via the plan dispatcher, unless debugging forces inline execution.
    JobDispatcher* stage_dispatcher = P.dispatcher();
    const bool stage_inline_override = ct_debug.stage_inline(stage_idx);
    if (stage_inline_override && !ct_debug.preserve_outer_dispatch) {
      stage_dispatcher = &InlineDispatcher::instance();
#if EIGFFT_TRACE_THREADS
      {
        std::ostringstream _oss;
        _oss << "[ct] stage forced inline stage_idx=" << stage_idx;
        TraceLogger::instance().log(_oss.str());
      }
#endif
    }
    void* stage_ctx = (enable_recovery && stage_ops.init)
                          ? static_cast<void*>(&stage_recovery)
                          : nullptr;
    if (stage_dispatcher == &InlineDispatcher::instance() && !enable_recovery) {
      // As above, keep the ordinary hot path free of type-erasure allocation.
      stage_chunk(0, static_cast<size_t>(chunk_count), 0);
    } else {
      stage_dispatcher->parallel_for_with_restore(static_cast<size_t>(chunk_count), 1, stage_chunk,
                                                  std::function<void(size_t,size_t)>(),
                                                  stage_ops,
                                                  stage_ctx);
    }
#if EIGFFT_TRACE_THREADS
    {
      std::ostringstream _oss;
      _oss << "[ct] stage end: len=" << len;
      TraceLogger::instance().log(_oss.str());
    }
#endif
    // Per-stage metadata and snapshots are captured at write-time above.
  }

  if (!invariant_scratch.empty()) {
    for (int stage = 0; stage < stages; ++stage) {
      T max_r0 = T(0);
      T max_e = T(0);
      for (int slot = 0; slot < invariant_slots; ++slot) {
        const std::complex<T>& val = invariant_scratch[static_cast<size_t>(slot) * static_cast<size_t>(stages) + static_cast<size_t>(stage)];
        if (val.real() > max_r0) max_r0 = val.real();
        if (val.imag() > max_e) max_e = val.imag();
      }
      invariant.set(stage, max_r0, max_e);
    }
  }

#if EIGFFT_TIMING
  timing_internal::ScopedTimer __t_post_bfly(timing_internal::Bin::PostButterfly);
#endif

  if (P.inverse) {
    const T scale = T(1) / T(N);
    for (int col = 0; col < B; ++col) {
      const Eigen::Index batch_index = Eigen::Index(col);
      Complex* column_ptr = layout.base + batch_index * batch_stride;
      for (int i = 0; i < N; ++i) {
        column_ptr[Eigen::Index(i) * axis_stride] *= scale;
      }
    }
  }
  if (debug_chunks) {
    for (int col = 0; col < B; ++col) {
      if (permute_counts[col] != 1) {
        std::cerr << "[debug] permute column " << col << " count=" << permute_counts[col] << "\n";
      }
      if (stage_counts[col] != stages) {
        std::cerr << "[debug] stage column " << col << " count=" << stage_counts[col] << " expected=" << stages << "\n";
      }
      std::cerr << "[debug] permute worker col=" << col << " worker=" << permute_worker[col] << " stage_worker=" << stage_worker[col] << "\n";
    }
  }
  // Post-butterfly transform hook (e.g., C2R imag clear or magnitude reduction)
  detail::apply_post_transform(P, layout);
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
}

} // namespace detail

template<class T>
inline void fft_apply_axis(const Plan<T>& P, const AxisLayout<T>& layout)
{
  if (!layout.valid())
    throw std::invalid_argument("AxisLayout must reference valid data.");
  if (layout.axis_size != P.N)
    throw std::invalid_argument("AxisLayout axis_size must match Plan::N.");

  const auto& descriptor = P.kernel();
  if (descriptor.execute_axis) {
#if EIGFFT_TIMING
    detail::timing_internal::ScopedTimer __t_alg(detail::timing_internal::Bin::AlgorithmDispatch);
#endif
    descriptor.execute_axis(P, layout, P.kernel_state());
  } else {
#if EIGFFT_TIMING
    detail::timing_internal::ScopedTimer __t_alg(detail::timing_internal::Bin::AlgorithmDispatch);
#endif
    detail::cooleytukey_execute_axis(P, layout, P.kernel_state());
  }
}

template<class T>
inline void dispatch_fft(Plan<T>& P, const typename Plan<T>::KernelDescriptor& descriptor,
                         const AxisLayout<T>& layout)
{
  P.select_kernel(descriptor);
  fft_apply_axis(P, layout);
}

template<class T>
inline void dispatch_fft(Plan<T>& P, KernelKind kind, const AxisLayout<T>& layout)
{
  switch (kind) {
    case KernelKind::CooleyTukey:
      dispatch_fft(P, Plan<T>::cooleytukey_kernel(), layout);
      break;
    case KernelKind::Stockham:
      dispatch_fft(P, Plan<T>::stockham_kernel(), layout);
      break;
    case KernelKind::External:
      dispatch_fft(P, Plan<T>::external_kernel(), layout);
      break;
    default:
      throw std::invalid_argument("Unsupported KernelKind.");
  }
}

template<class T>
inline void fft_inplace_batched_with_metadata(
    Eigen::Ref<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X,
    const Plan<T>& P,
    const MetadataRequest<T>* metadata_requests,
    int metadata_request_count)
{
  AxisLayout<T> layout;
  layout.base = X.data();
  layout.axis_size = P.N;
  layout.batch_size = static_cast<Eigen::Index>(X.cols());
  layout.axis_stride = 1;
  layout.batch_stride = static_cast<Eigen::Index>(X.rows());
  layout.metadata_requests = metadata_requests;
  layout.metadata_request_count = metadata_request_count;
  fft_apply_axis(P, layout);
}

template<class T>
inline void fft_inplace_batched(Eigen::Ref<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X,
                                const Plan<T>& P)
{
  fft_inplace_batched_with_metadata<T>(X, P, nullptr, 0);
}

template<class T>
inline void fft_inplace_2d(Eigen::Ref<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X,
                           const Plan<T>& axis0_plan,
                           const Plan<T>& axis1_plan,
                           Eigen::Index tile_rows = 64,
                           Eigen::Index tile_cols = 128)
{
  using Complex = typename Plan<T>::Complex;
  const Eigen::Index rows = X.rows();
  const Eigen::Index cols = X.cols();
  if (axis0_plan.N != rows) {
    throw std::invalid_argument("axis0_plan length must match matrix rows.");
  }
  if (axis1_plan.N != cols) {
    throw std::invalid_argument("axis1_plan length must match matrix cols.");
  }

  AxisLayout<T> axis0_layout;
  axis0_layout.base = X.data();
  axis0_layout.axis_size = rows;
  axis0_layout.batch_size = cols;
  axis0_layout.axis_stride = 1;
  axis0_layout.batch_stride = rows;
  fft_apply_axis(axis0_plan, axis0_layout);

  axis0_plan.ensure_nd_workspace(rows, cols);
  Complex* scratch = axis0_plan.transpose_buffer_data();
  detail::tiled_transpose_colmajor<T>(X.data(), rows, cols, scratch, tile_rows, tile_cols);

  AxisLayout<T> axis1_layout;
  axis1_layout.base = scratch;
  axis1_layout.axis_size = cols;
  axis1_layout.batch_size = rows;
  axis1_layout.axis_stride = 1;
  axis1_layout.batch_stride = cols;
  fft_apply_axis(axis1_plan, axis1_layout);

  detail::tiled_transpose_colmajor<T>(scratch, cols, rows, X.data(), tile_rows, tile_cols);
}

namespace detail {

template<class T>
std::unique_ptr<KernelContext> cooleytukey_create_state(Plan<T>&) {
  return nullptr;
}

template<class T>
void cooleytukey_destroy_state(Plan<T>&, KernelContext*) {}

template<class T>
void stockham_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx) {
  using Complex = typename Plan<T>::Complex;
  validate_fft_axis(P, layout);
  auto* state = static_cast<StockhamState<T>*>(ctx);
  if (!state) {
    cooleytukey_execute_axis(P, layout, nullptr);
    return;
  }

  const int N = P.N;
  const int stages = P.lgN;
  const detail::LayoutMetadataWriter<T> layout_writer(layout, static_cast<std::size_t>(N));
  if (layout_writer.enabled()) {
    for (int pos = 0; pos < N; ++pos) {
      layout_writer.set(static_cast<std::size_t>(pos), pos);
    }
  }
  const detail::StageSnapshotWriter<T> snap(layout, static_cast<std::size_t>(N), stages);
  const detail::StageParamsWriter<T> params(layout, stages);
  const detail::ButterflyPairsWriter<T> pairs(layout, static_cast<std::size_t>(N), stages);
  const detail::StageInvariantWriter<T> invariant(layout, stages);
  const detail::TwiddleIndexWriter<T> twmap(layout, static_cast<std::size_t>(N), stages);
  if (invariant.enabled()) {
    for (int s = 0; s < stages; ++s) {
      invariant.set(s, T(0), T(0));
    }
  }
  const int B = static_cast<int>(layout.batch_size);
  if (B <= 0) {
    if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
    return;
  }

  const detail::TwiddleMetadataWriter<T> metadata(layout, stages);
  metadata.record_plan_twiddles(P);

#if EIGFFT_TRACE_STOCKHAM
  static std::atomic<int> dispatch_seq{0};
  const int dispatch_id = dispatch_seq.fetch_add(1, std::memory_order_relaxed);
  {
    std::ostringstream _oss;
    _oss << "[stockham] dispatch=" << dispatch_id
         << " plan=" << &P
         << " state=" << state
         << " layout.base=" << static_cast<const void*>(layout.base)
         << " N=" << N
         << " B=" << B;
    TraceLogger::instance().log(_oss.str());
  }
#endif

  const int batch_size_int = static_cast<int>(layout.batch_size);
  const int adv_threads = std::max(1, P.effective_threads(batch_size_int));
  const int adv_lane_request =
      std::max(1, (P.tuning.packet_step > 0) ? P.tuning.packet_step : P.packet_cols);
  int B2 = 0;
  int desired_threads = adv_threads;
  int desired_lanes = adv_lane_request;
  int active_lane_cols2 = 0;
  Eigen::Index axis_stride2 = layout.axis_stride;
  Eigen::Index batch_stride2 = layout.batch_stride;
  // Build an algorithm-agnostic advanced workspace request and pass it into
  // prepare_plan_workspace so the Plan can attempt to honor the reservation
  // as part of workspace preparation (may call arena_resizer).
  typename Plan<T>::AdvancedWorkspaceRequest adv_req{};
  adv_req.min_lane_capacity = adv_lane_request;
  adv_req.min_thread_capacity = adv_threads;
  bool adv_reserved = false;
  prepare_plan_workspace(P, layout, B2, desired_threads, active_lane_cols2, desired_lanes, axis_stride2, batch_stride2, &adv_req, &adv_reserved);
  const int workspace_threads = std::max(1, P.workspace.threads);
  const int workspace_lanes = std::max(1, P.workspace.capacity);

  state->ensure_threads(workspace_threads);
  state->ensure_lane_capacity(workspace_lanes);

  const int stockham_capacity = std::max(1, state->arena.stockham_lane_capacity);
  const int cooleytukey_capacity = std::max(1, state->arena.baseline_lane_capacity);
  const int lane_capacity = stockham_capacity;
  const int lane_cols = std::max(1, std::min({desired_lanes, workspace_lanes, lane_capacity, cooleytukey_capacity}));
#if EIGFFT_TRACE_STOCKHAM
  {
    std::ostringstream _oss;
    _oss << "[stockham] lanes: requested=" << requested_lanes
         << " packet_cols=" << P.packet_cols
         << " lane_cols(final)=" << lane_cols
         << " capacity(stockham/cooleytukey)=" << stockham_capacity << "/" << cooleytukey_capacity;
    TraceLogger::instance().log(_oss.str());
  }
#endif
  const Eigen::Index axis_stride = layout.axis_stride;
  const Eigen::Index batch_stride = layout.batch_stride;
#if 1
  // Unified trace for layout/stride info (no-op unless EIGFFT_TRACE_LAYOUT=1)
  detail::trace_layout_info<T>("stockham", layout, axis_stride, batch_stride, lane_cols, static_cast<int>(layout.batch_size), lane_capacity);
#endif
#if EIGFFT_TRACE_STOCKHAM
  {
    std::ostringstream _oss;
    _oss << "[stockham] dispatch=" << dispatch_id
         << " desired_threads=" << desired_threads
         << " dispatcher_threads=" << workspace_threads
         << " lane_cols=" << lane_cols
         << " lane_capacity=" << lane_capacity
         << " axis_stride=" << axis_stride
         << " batch_stride=" << batch_stride;
    TraceLogger::instance().log(_oss.str());
  }
  {
    std::ostringstream _oss2;
    _oss2 << "[stockham] arena ping=" << static_cast<const void*>(state->arena.stockham_ping)
          << " pong=" << static_cast<const void*>(state->arena.stockham_pong)
          << " lane_bases=" << static_cast<const void*>(state->arena.stockham_lane_bases)
          << " thread_capacity=" << state->arena.stockham_thread_capacity
          << " lane_capacity=" << state->arena.stockham_lane_capacity;
    TraceLogger::instance().log(_oss2.str());
  }
#endif

  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(true);

  const int thread_slots = std::max(1, state->arena.stockham_thread_capacity);
  std::vector<std::complex<T>> invariant_scratch;
  if (invariant.enabled() && stages > 0) {
    invariant_scratch.assign(static_cast<size_t>(thread_slots) * static_cast<size_t>(stages),
                             std::complex<T>(T(0), T(0)));
  }

  AxisLayout<T> layout_capture = layout;
  auto process_chunk = [&, layout_capture](size_t start, size_t end, int worker_id) {
    if (start >= static_cast<size_t>(B)) return;
    const int width = static_cast<int>(end - start);
    if (width <= 0) return;
    if (width > lane_cols) {
      std::ostringstream oss;
      oss << "Stockham chunk width exceeds lane allocation: width=" << width
          << " lane_cols=" << lane_cols
          << " start=" << start << " end=" << end
          << " batch=" << B;
      throw std::runtime_error(oss.str());
    }
    // std::cerr << "[stockham debug] chunk start=" << start << " end=" << end
    //           << " worker=" << worker_id << " width=" << width << std::endl;
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      std::ostringstream _oss;
      _oss << "[stockham] first chunk width=" << width
           << " threads=" << workspace_threads
           << " lane_capacity=" << lane_capacity;
      TraceLogger::instance().log(_oss.str());
    }
    #endif
    size_t slot = static_cast<size_t>(worker_id);
    if (slot >= static_cast<size_t>(state->arena.stockham_thread_capacity)) {
      std::ostringstream oss;
      oss << "Stockham worker slot exceeds arena thread capacity: worker_id=" << worker_id
          << " capacity=" << state->arena.stockham_thread_capacity
          << " threads=" << workspace_threads;
      throw std::runtime_error(oss.str());
    }
    const size_t lane_stride = static_cast<size_t>(lane_capacity);
    const size_t buffer_stride = state->thread_stride();
    Complex* stage_in = state->arena.stockham_ping + slot * buffer_stride;
    Complex* stage_out = state->arena.stockham_pong + slot * buffer_stride;
    std::ptrdiff_t* lane_bases = state->arena.stockham_lane_bases + slot * lane_stride;
#if EIGFFT_RUNTIME_INSTRUMENTATION
  // Print per-worker slot mapping and arena pointers to detect aliasing.
  std::cerr << "[stockham-instr] worker=" << worker_id
        << " slot=" << slot
        << " arena=" << static_cast<const void*>(&state->arena)
        << " ping=" << static_cast<const void*>(state->arena.stockham_ping)
        << " pong=" << static_cast<const void*>(state->arena.stockham_pong)
        << " lane_bases=" << static_cast<const void*>(state->arena.stockham_lane_bases)
        << " buffer_stride=" << buffer_stride
        << " lane_stride=" << lane_stride << "\n";
#endif
    const ptrdiff_t max_offset = (layout_capture.batch_size > 0 && layout_capture.axis_size > 0)
                                     ? (ptrdiff_t(layout_capture.batch_size - 1) *
                                            ptrdiff_t(layout_capture.batch_stride) +
                                        ptrdiff_t(layout_capture.axis_size - 1) *
                                            ptrdiff_t(layout_capture.axis_stride))
                                     : 0;
    for (int lane = 0; lane < width; ++lane) {
      const ptrdiff_t base_idx = ptrdiff_t(start + lane) * ptrdiff_t(layout_capture.batch_stride);
      lane_bases[static_cast<size_t>(lane)] = base_idx;
      if (base_idx < 0 || base_idx > max_offset) {
        std::ostringstream _oss;
        _oss << "[stockham debug] batch base out of bounds: lane=" << lane
             << " start=" << start << " base_idx=" << base_idx
             << " max_offset=" << max_offset;
        TraceLogger::instance().log(_oss.str());
        return;
      }
    }
    const int safe_width = std::min(width, lane_cols);
    const bool capture_metadata = (start == 0);
    const int lane0_slot = (capture_metadata && safe_width > 0) ? 0 : -1;
    std::complex<T>* invariant_local = nullptr;
    if (!invariant_scratch.empty()) {
      if (static_cast<size_t>(slot) * static_cast<size_t>(stages) >= invariant_scratch.size()) {
        throw std::runtime_error("Invariant scratch slot exceeds allocation");
      }
      invariant_local = invariant_scratch.data() + static_cast<size_t>(slot) * static_cast<size_t>(stages);
    }
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      TraceLogger::instance().log(std::string("[stockham] load phase start"));
    }
    #endif
#if EIGFFT_TIMING
    timing_internal::ScopedTimer __t_pre_load(timing_internal::Bin::PreButterfly);
#endif
    for (int i = 0; i < N; ++i) {
      const ptrdiff_t off = ptrdiff_t(i) * ptrdiff_t(layout_capture.axis_stride);
      Complex* dest = stage_in + i * lane_capacity;
      for (int lane = 0; lane < safe_width; ++lane) {
        const ptrdiff_t base_idx = lane_bases[static_cast<size_t>(lane)];
        const ptrdiff_t idx = base_idx + off;
        if (idx < 0 || idx > max_offset) {
          std::ostringstream _oss;
          _oss << "[stockham] load OOB: lane=" << lane
               << " i=" << i << " idx=" << idx
               << " max=" << max_offset;
          TraceLogger::instance().log(_oss.str());
          return;
        }
        dest[lane] = layout_capture.base[idx];
      }
    }
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      TraceLogger::instance().log(std::string("[stockham] load phase complete"));
    }
    #endif

    Complex* in = stage_in;
    Complex* out = stage_out;
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      TraceLogger::instance().log(std::string("[stockham] transform loop start"));
    }
    #endif
#if EIGFFT_TIMING
    timing_internal::ScopedTimer __t_transform(timing_internal::Bin::ButterflyStage);
#endif
    for (int stage = 0; stage < stages; ++stage) {
      const int m = 1 << stage;
      const int distance = m << 1;
      const int segments = N >> (stage + 1);  // N / (2 * m)
      const int halfN = N >> 1;
      const int tw_step = segments;  // == N / distance
      if (capture_metadata && params.enabled()) {
        params.set(stage, distance, tw_step);
      }
      for (int j = 0; j < m; ++j) {
        const int tw_index = j * tw_step;
        const Complex w = P.W[tw_index];
        for (int g = 0; g < segments; ++g) {
          const int idx0 = (2 * j) * segments + g;
          const int idx1 = idx0 + segments;
          const int out0_idx = j * segments + g;
          const int out1_idx = out0_idx + halfN;
          const Complex* src0 = in + idx0 * lane_capacity;
          const Complex* src1 = in + idx1 * lane_capacity;
          Complex* dst0 = out + out0_idx * lane_capacity;
          Complex* dst1 = out + out1_idx * lane_capacity;
          // Use the Plan-level stockham config. If the user requested
          // in-place emulation (`prefer_inplace`), we provide preallocated
          // per-worker scratch buffers from the PlanArena baseline buffers
          // (no heap). Otherwise call the native scatter path.
          ButterflyScatter<T>::apply(src0, src1, dst0, dst1, safe_width, w, &P.butterfly_default_stockham);
          if (capture_metadata && pairs.enabled()) {
            pairs.set(stage, out0_idx, idx0, idx1);
            pairs.set(stage, out1_idx, idx0, idx1);
          }
          if (capture_metadata && twmap.enabled()) {
            twmap.set(stage, out0_idx, tw_index);
            twmap.set(stage, out1_idx, tw_index);
          }
          if (invariant_local) {
            std::complex<T>& slot_value = invariant_local[stage];
            T max_r0 = slot_value.real();
            T max_e = slot_value.imag();
            for (int lane = 0; lane < safe_width; ++lane) {
              const Complex a_in = src0[lane];
              const Complex b_in = src1[lane];
              const Complex y0 = dst0[lane];
              const Complex y1 = dst1[lane];
              const T r0 = std::abs((y0 + y1) - T(2) * a_in);
              const T e = (std::norm(y0) + std::norm(y1)) -
                          T(2) * (std::norm(a_in) + std::norm(b_in));
              const T abs_e = std::abs(e);
              if (r0 > max_r0) max_r0 = r0;
              if (abs_e > max_e) max_e = abs_e;
            }
            slot_value = std::complex<T>(max_r0, max_e);
          }
      }
    }
      if (lane0_slot >= 0 && snap.enabled()) {
        for (int i = 0; i < N; ++i) {
          snap.set(stage, i, out[i * lane_capacity + lane0_slot]);
        }
      }
      std::swap(in, out);
    }
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      TraceLogger::instance().log(std::string("[stockham] transform loop complete"));
    }
    #endif

    const Complex* final_buf = (stages % 2 == 0) ? stage_in : stage_out;
    if (P.inverse) {
      const T scale = T(1) / T(N);
      #if EIGFFT_TRACE_STOCKHAM
      if (start == 0 && worker_id == 0) {
        TraceLogger::instance().log(std::string("[stockham] store inverse phase start"));
      }
      #endif
  #if EIGFFT_TIMING
      timing_internal::ScopedTimer __t_post_store(timing_internal::Bin::PostButterfly);
  #endif
      for (int i = 0; i < N; ++i) {
        const ptrdiff_t off = ptrdiff_t(i) * ptrdiff_t(layout_capture.axis_stride);
        const Complex* src = final_buf + i * lane_capacity;
        for (int lane = 0; lane < safe_width; ++lane) {
          const ptrdiff_t base_idx = lane_bases[static_cast<size_t>(lane)];
          const ptrdiff_t idx = base_idx + off;
          if (idx < 0 || idx > max_offset) {
            std::ostringstream _oss;
            _oss << "[stockham] store OOB(inv): lane=" << lane
                 << " i=" << i << " idx=" << idx
                 << " max=" << max_offset;
            TraceLogger::instance().log(_oss.str());
            return;
          }
          layout_capture.base[idx] = src[lane] * scale;
        }
      }
    } else {
  #if EIGFFT_TRACE_STOCKHAM
  if (start == 0 && worker_id == 0) {
    TraceLogger::instance().log(std::string("[stockham] store phase start"));
  }
  #endif
#if EIGFFT_TIMING
  timing_internal::ScopedTimer __t_post_store2(timing_internal::Bin::PostButterfly);
#endif
      for (int i = 0; i < N; ++i) {
        const ptrdiff_t off = ptrdiff_t(i) * ptrdiff_t(layout_capture.axis_stride);
        const Complex* src = final_buf + i * lane_capacity;
        for (int lane = 0; lane < safe_width; ++lane) {
          const ptrdiff_t base_idx = lane_bases[static_cast<size_t>(lane)];
          const ptrdiff_t idx = base_idx + off;
          if (idx < 0 || idx > max_offset) {
            std::ostringstream _oss;
            _oss << "[stockham] store OOB: lane=" << lane
                 << " i=" << i << " idx=" << idx
                 << " max=" << max_offset;
            TraceLogger::instance().log(_oss.str());
            return;
          }
          layout_capture.base[idx] = src[lane];
        }
      }
    }
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      TraceLogger::instance().log(std::string("[stockham] store phase complete"));
    }
    #endif
  };

  // Use the plan dispatcher to optionally parallelize across frames; default runs inline.
  P.dispatcher()->parallel_for(static_cast<size_t>(B), static_cast<size_t>(lane_cols), process_chunk);

#if EIGFFT_TRACE_STOCKHAM
  TraceLogger::instance().log(std::string("[stockham] dispatch=") + std::to_string(dispatch_id) + std::string(" parallel_for complete"));
#endif

  if (!invariant_scratch.empty()) {
    for (int stage = 0; stage < stages; ++stage) {
      T max_r0 = T(0);
      T max_e = T(0);
      for (int slot = 0; slot < thread_slots; ++slot) {
        const std::complex<T>& val = invariant_scratch[static_cast<size_t>(slot) * static_cast<size_t>(stages) + static_cast<size_t>(stage)];
        if (val.real() > max_r0) max_r0 = val.real();
        if (val.imag() > max_e) max_e = val.imag();
      }
      invariant.set(stage, max_r0, max_e);
    }
  }
  // Post-butterfly transform hook (e.g., C2R imag clear or magnitude reduction)
  detail::apply_post_transform(P, layout);
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
}

template<class T>
std::unique_ptr<KernelContext> stockham_create_state(Plan<T>& plan) {
  return std::unique_ptr<KernelContext>(new StockhamState<T>(plan, plan.arena));
}

template<class T>
void stockham_destroy_state(Plan<T>&, KernelContext*) {}

template<class T>
void external_execute_axis(const Plan<T>&, const AxisLayout<T>&, KernelContext*) {
  throw std::logic_error("External FFT kernel requested but no implementation supplied.");
}

template<class T>
std::unique_ptr<KernelContext> external_create_state(Plan<T>&) {
  return nullptr;
}

template<class T>
void external_destroy_state(Plan<T>&, KernelContext*) {}

} // namespace detail

template<class T>
const typename Plan<T>::KernelDescriptor& Plan<T>::cooleytukey_kernel() {
  static const KernelDescriptor desc{
      KernelKind::CooleyTukey,
      "cooleytukey",
      KernelAccuracy::Default,
      true,
      detail::cooleytukey_create_state<T>,
      detail::cooleytukey_execute_axis<T>,
      detail::cooleytukey_destroy_state<T>
  };
  return desc;
}

template<class T>
const typename Plan<T>::KernelDescriptor& Plan<T>::stockham_kernel() {
  static const KernelDescriptor desc{
      KernelKind::Stockham,
      "stockham-autosort",
      KernelAccuracy::Default,
      true,
      detail::stockham_create_state<T>,
      detail::stockham_execute_axis<T>,
      detail::stockham_destroy_state<T>
  };
  return desc;
}

template<class T>
const typename Plan<T>::KernelDescriptor& Plan<T>::external_kernel() {
  static const KernelDescriptor desc{
      KernelKind::External,
      "external-provider",
      KernelAccuracy::HighPrecision,
      false,
      detail::kExternalKernelAvailable ? detail::external_create_state<T> : nullptr,
      detail::kExternalKernelAvailable ? detail::external_execute_axis<T> : nullptr,
      detail::kExternalKernelAvailable ? detail::external_destroy_state<T> : nullptr
  };
  return desc;
}

template<class T>
const std::array<const typename Plan<T>::KernelDescriptor*, 3>& Plan<T>::builtin_kernels() {
  static const std::array<const KernelDescriptor*, 3> list{
      &cooleytukey_kernel(), &stockham_kernel(),
      detail::kExternalKernelAvailable ? &external_kernel() : nullptr};
  return list;
}

template<class T>
struct PlanProviderSelector {
  static typename detail::ButterflyRegistry<T>::ProviderInfo select(bool require_scatter, int min_simd_width, ButterflyMethod method_hint) {
    return detail::ButterflyRegistry<T>::negotiate(require_scatter, min_simd_width, method_hint);
  }
};

} // namespace eigfft
