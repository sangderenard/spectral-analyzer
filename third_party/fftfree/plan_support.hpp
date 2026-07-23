#pragma once

#include "eigen_fft.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <thread>

namespace eigfft {

#ifndef EIGFFT_RUNTIME_INSTRUMENTATION
// Enable to print lightweight runtime instrumentation for debugging
// worker <-> arena slot mappings. Set to 1 via build flags when needed.
#define EIGFFT_RUNTIME_INSTRUMENTATION 0
#endif

struct PlanRuntimeConfig {
  int threads = 4;
  int lanes = 2;
  std::size_t transpose_capacity = 0;
  bool inverse = false;
  // Outer-API requested radix. 0 means "unspecified by caller".
  // If non-zero, this value will be applied to Plan::butterfly_default_* before
  // plan allocation so the caller-specified radix drives internal configuration.
  int radix = 0;
  // Optional mixed-radix pattern supplied by the outer API. Values are raw
  // integer radices (e.g., 2,4). If empty, no pattern was provided.
  std::vector<int> radix_pattern;
  // Transform selection and optional magnitude reduction hook control.
  // transform: 0=C2C, 1=R2C, 2=C2R, 3=R2R
  int transform = 0;
  bool reduce_magnitude = false;
  bool store_polar = false; // if true and reduce_magnitude, store (mag,phase) interleaved
  bool half_spectrum = false; // half-spectrum packing for R2C/C2R (first N/2+1 bins)
  // Threading policy: outer vs inner
  bool allow_outer_parallel = true;   // if false, do not create outer worker pool
  bool allow_inner_parallel = false;  // if true and outer threads == 1, kernels may use inner threads
  int inner_threads = 0;              // desired inner thread count when allowed (<=0 => auto)
};

template <class Scalar>
class PlanEnvironment {
 public:
  using Complex = std::complex<Scalar>;

  PlanEnvironment() = default;

  void initialize(int N, bool inverse, const PlanRuntimeConfig& cfg) {
    // Enforce one-shot plan usage: if a previous plan exists, discard it.
    plan_.reset();
    initialize(N, inverse, cfg, cfg.transpose_capacity);
  }

  void initialize(int N, bool inverse, const PlanRuntimeConfig& cfg, std::size_t transpose_elements) {
    const int max_threads = Plan<Scalar>::Limits::kCompileTimeMaxThreads;
    const int clamped_threads = std::clamp(cfg.threads, 1, max_threads);
    const int max_lanes = Plan<Scalar>::Limits::compile_time_max_lane_capacity();
    const int clamped_lanes = std::clamp(cfg.lanes, 1, max_lanes);

    const bool use_inner_threads = cfg.allow_inner_parallel;
    int requested_inner = 1;
    if (use_inner_threads) {
      if (cfg.inner_threads > 0) {
        requested_inner = cfg.inner_threads;
      } else {
        int hw = static_cast<int>(std::thread::hardware_concurrency());
        if (hw <= 0) hw = 1;
        requested_inner = hw;
      }
      // Clamp inner threads to compile-time limit, not the outer threads cap.
      requested_inner =
          std::clamp(requested_inner, 1, Plan<Scalar>::Limits::kCompileTimeMaxThreads);
    }

    const int plan_thread_budget =
        std::max(clamped_threads, use_inner_threads ? requested_inner : 1);
    bool enable_plan_threads =
        (cfg.allow_outer_parallel && plan_thread_budget > 1) ||
        (use_inner_threads && plan_thread_budget > 1);
    if (!detail::kAllowSequentialFallback) {
      enable_plan_threads = plan_thread_budget > 0;
    }

    const auto shape =
        compute_plan_arena_shape<Scalar>(N, plan_thread_budget, clamped_lanes);

    twiddles_.resize(shape.twiddles);
    bitrev_.resize(shape.bitrev);
    baseline_a_.resize(shape.baseline_complex);
    baseline_b_.resize(shape.baseline_complex);
    baseline_columns_.resize(shape.baseline_columns);
    std::fill(baseline_columns_.begin(), baseline_columns_.end(), nullptr);
    stockham_ping_.resize(shape.stockham_stage);
    stockham_pong_.resize(shape.stockham_stage);
    stockham_lane_bases_.resize(shape.stockham_lane_bases);
    nd_transpose_.resize(transpose_elements);

    arena_.twiddles = twiddles_.data();
    arena_.twiddle_count = narrow_int(shape.twiddles);
    arena_.bitrev = bitrev_.data();
    arena_.bitrev_count = narrow_int(shape.bitrev);
    arena_.baseline_a = baseline_a_.data();
    arena_.baseline_b = baseline_b_.data();
    arena_.baseline_columns = baseline_columns_.data();
    arena_.baseline_thread_capacity = plan_thread_budget;
    arena_.baseline_lane_capacity = clamped_lanes;
    arena_.stockham_ping = stockham_ping_.data();
    arena_.stockham_pong = stockham_pong_.data();
    arena_.stockham_lane_bases = stockham_lane_bases_.data();
    arena_.stockham_thread_capacity = plan_thread_budget;
    arena_.stockham_lane_capacity = clamped_lanes;
    arena_.nd_transpose = nd_transpose_.empty() ? nullptr : nd_transpose_.data();
    arena_.nd_transpose_capacity = nd_transpose_.size();

    plan_ = std::make_unique<Plan<Scalar>>(N, arena_, inverse, enable_plan_threads, plan_thread_budget);
    // Apply transform mode and reduction settings
    switch (cfg.transform) {
      default:
      case 0: plan_->transform_mode = Plan<Scalar>::TransformMode::C2C; break;
      case 1: plan_->transform_mode = Plan<Scalar>::TransformMode::R2C; break;
      case 2: plan_->transform_mode = Plan<Scalar>::TransformMode::C2R; break;
      case 3: plan_->transform_mode = Plan<Scalar>::TransformMode::R2R; break;
    }
    plan_->reduce_magnitude = cfg.reduce_magnitude;
    plan_->store_polar = cfg.store_polar;
    plan_->half_spectrum = cfg.half_spectrum;
    // If the runtime configuration requested a specific radix, apply it to
    // the Plan's default butterfly configurations before the Plan is used.
    if (cfg.radix != 0) {
      // Map integer radix values to the enum used by the Plan/Butterfly code.
      using BR = ButterflyRadix;
      BR mapped = BR::Radix2;
      switch (cfg.radix) {
        case 2: mapped = BR::Radix2; break;
        case 4: mapped = BR::Radix4; break;
        case 8: mapped = BR::Radix8; break;
        case 16: mapped = BR::Radix16; break;
        default: mapped = BR::Radix2; break; // clamp unknown values to Radix2
      }
      plan_->butterfly_default_cooleytukey.radix = mapped;
      plan_->butterfly_default_stockham.radix = mapped;
      plan_->butterfly_default_external.radix = mapped;
    }
    // If the caller provided a mixed-radix pattern, copy it into the Plan so
    // kernels can consult per-stage radix values. Unknown radices are
    // clamped to Radix2.
    if (!cfg.radix_pattern.empty()) {
      plan_->radix_pattern.clear();
      using BR = ButterflyRadix;
      for (int r : cfg.radix_pattern) {
        BR mapped_p = BR::Radix2;
        switch (r) {
          case 2: mapped_p = BR::Radix2; break;
          case 4: mapped_p = BR::Radix4; break;
          case 8: mapped_p = BR::Radix8; break;
          case 16: mapped_p = BR::Radix16; break;
          default: mapped_p = BR::Radix2; break;
        }
        plan_->radix_pattern.push_back(mapped_p);
      }
    }
    plan_->set_arena_resizer(&PlanEnvironment::ResizeArena);
    plan_->workspace.bind(*plan_);
    plan_->pending_nd_capacity = 0;
    threads_ = clamped_threads;
    lanes_ = clamped_lanes;
  }

  Plan<Scalar>& plan() {
    if (!plan_) {
      throw std::logic_error("PlanEnvironment not initialized");
    }
    return *plan_;
  }

  // Explicitly discard the current plan to enforce single-use semantics.
  void discard_plan() {
    plan_.reset();
  }

  const Plan<Scalar>& plan() const {
    if (!plan_) {
      throw std::logic_error("PlanEnvironment not initialized");
    }
    return *plan_;
  }

  PlanArena<Scalar>& arena() { return arena_; }
  const PlanArena<Scalar>& arena() const { return arena_; }

  int threads() const { return threads_; }
  int lanes() const { return lanes_; }

 private:
  static void ResizeArena(PlanArena<Scalar>& arena, int N, int threads, int lanes) {
    PlanEnvironment* self = owner_from_arena(arena);
    if (!self) return;
    self->resize_arena_impl(N, threads, lanes);
  }

  static PlanEnvironment* owner_from_arena(PlanArena<Scalar>& arena) {
    return reinterpret_cast<PlanEnvironment*>(reinterpret_cast<char*>(&arena) - offsetof(PlanEnvironment, arena_));
  }

  static int narrow_int(std::size_t value) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("PlanArena size exceeds 32-bit bounds");
    }
    return static_cast<int>(value);
  }

  void resize_arena_impl(int N, int threads, int lanes) {
    const int clamped_threads = std::clamp(threads, 1, Plan<Scalar>::Limits::kCompileTimeMaxThreads);
    const int clamped_lanes = std::clamp(lanes, 1, Plan<Scalar>::Limits::compile_time_max_lane_capacity());
    const auto shape = compute_plan_arena_shape<Scalar>(N, clamped_threads, clamped_lanes);
    const std::size_t requested_nd = plan_ ? plan_->pending_nd_capacity : 0;

    twiddles_.resize(shape.twiddles);
    bitrev_.resize(shape.bitrev);
    baseline_a_.resize(shape.baseline_complex);
    baseline_b_.resize(shape.baseline_complex);
    baseline_columns_.resize(shape.baseline_columns);
    std::fill(baseline_columns_.begin(), baseline_columns_.end(), nullptr);
    stockham_ping_.resize(shape.stockham_stage);
    stockham_pong_.resize(shape.stockham_stage);
    stockham_lane_bases_.resize(shape.stockham_lane_bases);
    if (requested_nd > nd_transpose_.size()) {
      nd_transpose_.resize(requested_nd);
    }

    // Handle special per-slot buffers requested by Plan (if any). The Plan may
    // set pending_special_capacity[] prior to calling the resizer; honor those
    // requests by resizing per-slot vectors here and updating the arena
    // metadata pointers/capacities/strides.
    for (int slot = 0; slot < static_cast<int>(PlanArena<Scalar>::kMaxSpecialBuffers); ++slot) {
      std::size_t req = 0;
      if (plan_) {
        req = plan_->pending_special_capacity[static_cast<size_t>(slot)];
      }
      if (req > special_buffers_[static_cast<size_t>(slot)].size()) {
        special_buffers_[static_cast<size_t>(slot)].resize(req);
      }
    }

    arena_.twiddles = twiddles_.data();
    arena_.twiddle_count = narrow_int(shape.twiddles);
    arena_.bitrev = bitrev_.data();
    arena_.bitrev_count = narrow_int(shape.bitrev);
    arena_.baseline_a = baseline_a_.data();
    arena_.baseline_b = baseline_b_.data();
    arena_.baseline_columns = baseline_columns_.data();
    arena_.baseline_thread_capacity = clamped_threads;
    arena_.baseline_lane_capacity = clamped_lanes;
    arena_.stockham_ping = stockham_ping_.data();
    arena_.stockham_pong = stockham_pong_.data();
    arena_.stockham_lane_bases = stockham_lane_bases_.data();
    arena_.stockham_thread_capacity = clamped_threads;
    arena_.stockham_lane_capacity = clamped_lanes;
    arena_.nd_transpose = nd_transpose_.empty() ? nullptr : nd_transpose_.data();
    arena_.nd_transpose_capacity = nd_transpose_.size();

    // Publish special buffer pointers and capacities into the arena.
    for (int slot = 0; slot < static_cast<int>(PlanArena<Scalar>::kMaxSpecialBuffers); ++slot) {
      auto& vec = special_buffers_[static_cast<size_t>(slot)];
      arena_.special_buf[slot] = vec.empty() ? nullptr : vec.data();
      arena_.special_capacity[slot] = vec.size();
      arena_.special_stride[slot] = vec.empty() ? 0 : static_cast<std::ptrdiff_t>(1);
      // Clear pending request if we satisfied it.
      if (plan_) plan_->pending_special_capacity[static_cast<size_t>(slot)] = 0;
    }

    threads_ = clamped_threads;
    lanes_ = clamped_lanes;

    if (plan_) {
      plan_->W = arena_.twiddles;
      plan_->bitrev = arena_.bitrev;
      plan_->requested_threads = std::min(plan_->requested_threads, clamped_threads);
      plan_->workspace.bind(*plan_);
      plan_->workspace.threads = std::max(1, std::min(plan_->workspace.threads, clamped_threads));
      plan_->workspace.capacity = std::max(1, std::min(plan_->workspace.capacity, clamped_lanes));
      plan_->pending_nd_capacity = 0;
      // ensure any pending special capacity flags are cleared after allocation
      for (int slot = 0; slot < static_cast<int>(PlanArena<Scalar>::kMaxSpecialBuffers); ++slot) {
        plan_->pending_special_capacity[static_cast<size_t>(slot)] = 0;
      }
    }
  }

  PlanArena<Scalar> arena_{};
  std::vector<Complex> twiddles_;
  std::vector<int> bitrev_;
  std::vector<Complex> baseline_a_;
  std::vector<Complex> baseline_b_;
  std::vector<Complex*> baseline_columns_;
  std::vector<Complex> stockham_ping_;
  std::vector<Complex> stockham_pong_;
  std::vector<std::ptrdiff_t> stockham_lane_bases_;
  std::vector<Complex> nd_transpose_;
  // Per-slot special buffers for algorithm-specific scratch. These back the
  // PlanArena::special_buf[] pointers to provide fast fixed-slot access.
  std::array<std::vector<Complex>, PlanArena<Scalar>::kMaxSpecialBuffers> special_buffers_{};
  std::unique_ptr<Plan<Scalar>> plan_;
  int threads_ = 1;
  int lanes_ = 1;
};

// Thread-safe plan cache for efficient reuse
template <class Scalar>
class PlanCache {
 public:
  using Complex = std::complex<Scalar>;

  struct PlanKey {
    int N;
    bool inverse;
    int threads;
    int lanes;
    std::size_t transpose_capacity;

    bool operator==(const PlanKey& other) const {
      return N == other.N && inverse == other.inverse &&
             threads == other.threads && lanes == other.lanes &&
             transpose_capacity == other.transpose_capacity;
    }
  };

  struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const {
      std::size_t h = 0;
      h = h * 31 + std::hash<int>()(key.N);
      h = h * 31 + std::hash<bool>()(key.inverse);
      h = h * 31 + std::hash<int>()(key.threads);
      h = h * 31 + std::hash<int>()(key.lanes);
      h = h * 31 + std::hash<std::size_t>()(key.transpose_capacity);
      return h;
    }
  };

  // Token representing thread ownership of a cached plan
  class Token {
   public:
    Token() = default;
    Token(const Token&) = delete;
    Token& operator=(const Token&) = delete;
    Token(Token&& other) noexcept
        : cache_(other.cache_), key_(other.key_), index_(other.index_), plan_(other.plan_),
          valid_(other.valid_) {
      other.valid_ = false;
      other.plan_ = nullptr;
    }
    Token& operator=(Token&& other) noexcept {
      if (this != &other) {
        release();
        cache_ = other.cache_;
        key_ = other.key_;
        index_ = other.index_;
        plan_ = other.plan_;
        valid_ = other.valid_;
        other.valid_ = false;
        other.plan_ = nullptr;
      }
      return *this;
    }
    ~Token() { release(); }

    bool valid() const { return valid_; }
    Plan<Scalar>& plan() {
      if (!valid_ || !plan_) throw std::logic_error("Token is not valid");
      return *plan_;
    }
    const Plan<Scalar>& plan() const {
      if (!valid_ || !plan_) throw std::logic_error("Token is not valid");
      return *plan_;
    }

   private:
    friend class PlanCache;
    Token(PlanCache* cache, PlanKey key, std::size_t index, Plan<Scalar>* plan)
        : cache_(cache), key_(key), index_(index), plan_(plan), valid_(true) {}

    void release() {
      if (valid_ && cache_) {
        cache_->release_token(key_, index_);
      }
      valid_ = false;
      plan_ = nullptr;
    }

    PlanCache* cache_ = nullptr;
    PlanKey key_;
    std::size_t index_ = 0;
    Plan<Scalar>* plan_ = nullptr;
    bool valid_ = false;
  };

  PlanCache() = default;
  ~PlanCache() = default;

  // Get or create a plan for the given configuration. Each token owns an
  // exclusive plan instance so call sites never share scratch buffers.
  Token get_plan(int N, bool inverse, const PlanRuntimeConfig& cfg) {
#if EIGFFT_TIMING
  detail::timing_internal::ScopedTimer __t_plan_cache(detail::timing_internal::Bin::PlanCacheLookup);
#endif
    auto runtime_cfg = normalize_config(cfg);
    PlanKey key{N, inverse, runtime_cfg.threads, runtime_cfg.lanes,
                runtime_cfg.transpose_capacity};

    std::unique_lock<std::mutex> lock(mutex_);
    Bucket& bucket = ensure_bucket_locked(key, runtime_cfg);
    ensure_min_pool_locked(bucket, key, runtime_cfg,
                           static_cast<std::size_t>(runtime_cfg.threads));
    Slot slot = acquire_slot_locked(bucket, key, runtime_cfg);
    Plan<Scalar>* plan_ptr = slot.plan;
    std::size_t index = slot.index;
    lock.unlock();
    return Token(this, key, index, plan_ptr);
  }

  // Pre-populate cached plans for common audio DSP configurations so that the
  // first call is already warm. The cache keeps enough instances around to
  // cover the configured thread capacity.
  void warm_audio_profiles(const PlanRuntimeConfig& base_cfg = {}) {
    auto normalized = normalize_config(base_cfg);
    const int default_threads =
        normalized.threads > 0 ? normalized.threads : Plan<Scalar>::Limits::kDefaultRuntimeThreads;
    const int default_lanes =
        normalized.lanes > 0 ? normalized.lanes : Plan<Scalar>::Limits::kDefaultLaneCapacity;

    std::array<int, 8> fft_sizes{32, 64, 128, 256, 512, 1024, 2048, 4096};

    std::array<int, 4> lane_seed{
        1,
        default_lanes,
        Plan<Scalar>::Limits::kDefaultLaneCapacity,
        default_lanes * 2};
    std::vector<int> lane_candidates;
    lane_candidates.reserve(lane_seed.size());
    for (int candidate : lane_seed) {
      if (candidate <= 0) continue;
      int clamped = std::clamp(candidate, 1,
                               Plan<Scalar>::Limits::compile_time_max_lane_capacity());
      if (std::find(lane_candidates.begin(), lane_candidates.end(), clamped) ==
          lane_candidates.end()) {
        lane_candidates.push_back(clamped);
      }
    }

    std::array<int, 4> thread_seed{
        1,
        default_threads,
        Plan<Scalar>::Limits::kDefaultRuntimeThreads,
        Plan<Scalar>::Limits::kCompileTimeMaxThreads};
    std::vector<int> thread_candidates;
    thread_candidates.reserve(thread_seed.size());
    for (int candidate : thread_seed) {
      if (candidate <= 0) continue;
      int clamped = std::clamp(candidate, 1,
                               Plan<Scalar>::Limits::kCompileTimeMaxThreads);
      if (std::find(thread_candidates.begin(), thread_candidates.end(), clamped) ==
          thread_candidates.end()) {
        thread_candidates.push_back(clamped);
      }
    }

    std::array<bool, 2> inverse_modes{false, true};

    std::unique_lock<std::mutex> lock(mutex_);
    for (int N : fft_sizes) {
      for (int lanes : lane_candidates) {
        if (lanes <= 0) continue;
        int clamped_lanes = std::clamp(lanes, 1,
            Plan<Scalar>::Limits::compile_time_max_lane_capacity());
        for (int threads : thread_candidates) {
          if (threads <= 0) continue;
          int clamped_threads =
              std::clamp(threads, 1, Plan<Scalar>::Limits::kCompileTimeMaxThreads);
          PlanRuntimeConfig runtime = normalized;
          runtime.threads = clamped_threads;
          runtime.lanes = clamped_lanes;
          runtime.transpose_capacity = normalized.transpose_capacity;
          for (bool inverse : inverse_modes) {
            runtime.inverse = inverse;
            PlanKey key{N, inverse, runtime.threads, runtime.lanes,
                        runtime.transpose_capacity};
            Bucket& bucket = ensure_bucket_locked(key, runtime);
            ensure_min_pool_locked(bucket, key, runtime,
                                   static_cast<std::size_t>(runtime.threads));
          }
        }
      }
    }
  }

 private:
  struct CachedEntry {
    std::unique_ptr<PlanEnvironment<Scalar>> env;
    bool in_use = false;
  };

  struct Bucket {
    PlanRuntimeConfig runtime{};
    std::vector<CachedEntry> entries;
    bool runtime_set = false;
  };

  struct Slot {
    std::size_t index = 0;
    Plan<Scalar>* plan = nullptr;
    PlanEnvironment<Scalar>* env = nullptr;
  };

  static PlanRuntimeConfig normalize_config(const PlanRuntimeConfig& cfg) {
    PlanRuntimeConfig normalized = cfg;
    if (normalized.threads <= 0) {
      normalized.threads = Plan<Scalar>::Limits::kDefaultRuntimeThreads;
    }
    normalized.threads =
        std::clamp(normalized.threads, 1, Plan<Scalar>::Limits::kCompileTimeMaxThreads);

    if (normalized.lanes <= 0) {
      normalized.lanes = Plan<Scalar>::Limits::kDefaultLaneCapacity;
    }
    normalized.lanes = std::clamp(normalized.lanes, 1,
                                  Plan<Scalar>::Limits::compile_time_max_lane_capacity());
    return normalized;
  }

  Bucket& ensure_bucket_locked(const PlanKey& key,
                               const PlanRuntimeConfig& runtime_cfg) {
    auto [it, inserted] = buckets_.try_emplace(key);
    Bucket& bucket = it->second;
    if (!bucket.runtime_set) {
      bucket.runtime = runtime_cfg;
      bucket.runtime.inverse = key.inverse;
      bucket.runtime_set = true;
    }
    return bucket;
  }

  void ensure_min_pool_locked(Bucket& bucket, const PlanKey& key,
                              const PlanRuntimeConfig& runtime_cfg,
                              std::size_t copies) {
    if (copies == 0) copies = 1;
    while (bucket.entries.size() < copies) {
  auto env = std::make_unique<PlanEnvironment<Scalar>>();
  PlanRuntimeConfig cfg = runtime_cfg;
  cfg.inverse = key.inverse;
#if EIGFFT_TIMING
  {
    detail::timing_internal::ScopedTimer __t_new_plan(detail::timing_internal::Bin::PlanNew);
    env->initialize(key.N, key.inverse, cfg);
  }
#else
  env->initialize(key.N, key.inverse, cfg);
#endif
  bucket.entries.push_back({std::move(env), false});
    }
  }

  Slot acquire_slot_locked(Bucket& bucket, const PlanKey& key,
                           const PlanRuntimeConfig& runtime_cfg) {
    for (std::size_t i = 0; i < bucket.entries.size(); ++i) {
      auto& entry = bucket.entries[i];
      if (!entry.in_use) {
        entry.in_use = true;
        Slot s{i, &entry.env->plan(), entry.env.get()};
#if EIGFFT_RUNTIME_INSTRUMENTATION
        // Lightweight instrumentation: print which cache entry and arena
        // address was handed out for debugging concurrent slot aliasing.
        std::cerr << "[plancache] reuse entry index=" << s.index
                  << " env=" << static_cast<const void*>(s.env)
                  << " arena=" << static_cast<const void*>(&s.env->arena())
                  << " threads=" << s.env->threads()
                  << " lanes=" << s.env->lanes() << "\n";
#endif
        return s;
      }
    }

    auto env = std::make_unique<PlanEnvironment<Scalar>>();
    PlanRuntimeConfig cfg = runtime_cfg;
    cfg.inverse = key.inverse;
    env->initialize(key.N, key.inverse, cfg);
  bucket.entries.push_back({std::move(env), true});
  CachedEntry& entry = bucket.entries.back();
  Slot s{bucket.entries.size() - 1, &entry.env->plan(), entry.env.get()};
#if EIGFFT_RUNTIME_INSTRUMENTATION
  std::cerr << "[plancache] new entry index=" << s.index
        << " env=" << static_cast<const void*>(s.env)
        << " arena=" << static_cast<const void*>(&s.env->arena())
        << " threads=" << s.env->threads()
        << " lanes=" << s.env->lanes() << "\n";
#endif
  return s;
  }

  void release_token(const PlanKey& key, std::size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buckets_.find(key);
    if (it == buckets_.end()) return;
    Bucket& bucket = it->second;
    if (index < bucket.entries.size()) {
      // Clear any dispatcher that may have been installed on the Plan before
      // returning it to the cache. This avoids dangling pointers when
      // callers temporarily install stack-based dispatchers and then release
      // the token. Be defensive: catch any exceptions to avoid throwing while
      // holding the cache mutex.
      try {
        auto& entry = bucket.entries[index];
        if (entry.env) {
          // Reset to InlineDispatcher (null is translated to Inline inside Plan)
          entry.env->plan().set_dispatcher(nullptr);
        }
      } catch (...) {
        // swallow - best-effort cleanup only
      }
      bucket.entries[index].in_use = false;
    }
  }

  std::mutex mutex_;
  std::unordered_map<PlanKey, Bucket, PlanKeyHash> buckets_;
};

} // namespace eigfft
