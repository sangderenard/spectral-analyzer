#include "eigen_fft.hpp"
#include "plan_support.hpp"
#include "crash_handler.hpp"
#include "fft_cffi.hpp"
#include <memory>

#include <Eigen/Core>

#include <chrono>
#include <atomic>
#include <complex>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <thread>
#include <vector>
#include <utility>
#include <stdexcept>

namespace {

std::atomic<size_t> g_cffi_restore_chunks{0};

extern "C" void fft_example_restore_observer(std::size_t start, std::size_t end) {
  if (end > start) {
    g_cffi_restore_chunks.fetch_add(end - start, std::memory_order_relaxed);
  }
}

void run_cffi_recovery_demo(int thread_hint) {
  const int N = 1024;
  const int hop = N / 2;
  const int expected_frames = 6;
  const int threads = std::max(2, thread_hint);
  const int bins = N / 2 + 1;
  const std::size_t pcm_len = static_cast<std::size_t>(N + (expected_frames - 1) * hop);

  std::vector<float> pcm(pcm_len);
  constexpr float kPi = static_cast<float>(3.14159265358979323846);
  for (std::size_t i = 0; i < pcm.size(); ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(pcm.size());
    pcm[i] = 0.6f * std::sin(2.0f * kPi * 220.0f * t) +
             0.25f * std::sin(2.0f * kPi * 440.0f * t);
  }

  struct DropoutGuard {
    DropoutGuard() {
      fft_clear_workerpool_dropout_history();
      fft_set_workerpool_debug_dropout(0);
      fft_set_workerpool_dropout_pattern(2, 1, 0);
      fft_set_workerpool_dropout_single(-1);
      fft_set_workerpool_dropout_persistent(0);
    }
    ~DropoutGuard() {
      fft_set_workerpool_debug_dropout(0);
      fft_set_workerpool_dropout_pattern(0, 0, 0);
      fft_set_workerpool_dropout_single(-1);
      fft_set_workerpool_dropout_persistent(0);
      fft_clear_workerpool_dropout_history();
    }
  } dropout_guard;

  void* forward = fft_init_full(
      static_cast<std::size_t>(N),
      threads,
      1,
      0,
      FFT_KERNEL_COOLEYTUKEY,
      0,
      nullptr,
      0,
      1,
      N,
      hop,
      1,
      FFT_TRANSFORM_R2C,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      1);
  if (!forward) {
    throw std::runtime_error("fft_example: fft_init_full (forward) failed");
  }

  const std::size_t ctx_workers = fft_ctx_worker_threads(forward);
  const std::size_t ctx_effective = fft_ctx_effective_threads(forward);
  std::cout << "[cffi-demo] forward ctx workers=" << ctx_workers
            << " effective_threads=" << ctx_effective << std::endl;

  std::vector<float> real(expected_frames * bins, 0.0f);
  std::vector<float> imag(expected_frames * bins, 0.0f);
  std::vector<float> mag(expected_frames * bins, 0.0f);

  g_cffi_restore_chunks.store(0, std::memory_order_relaxed);
  fft_register_restore_observer(&fft_example_restore_observer);
  const size_t produced = fft_execute_batched(
      forward,
      pcm.data(),
      pcm.size(),
      real.data(),
      imag.data(),
      mag.data(),
      1,
      1,
      expected_frames);
  fft_register_restore_observer(nullptr);
  if (produced == 0) {
    fft_free(forward);
    throw std::runtime_error("fft_example: fft_execute_batched failed");
  }
  const int frames = static_cast<int>(produced);
  const int total_bins = frames * bins;
  real.resize(total_bins);
  imag.resize(total_bins);
  mag.resize(total_bins);

  if (g_cffi_restore_chunks.load(std::memory_order_relaxed) == 0) {
    std::cout << "[cffi-demo] warning: restore observer recorded no recoveries; "
                 "adjust dropout pattern if diagnostics required." << std::endl;
  }

  void* inverse = fft_init_full(
      static_cast<std::size_t>(N),
      threads,
      1,
      1,
      FFT_KERNEL_COOLEYTUKEY,
      0,
      nullptr,
      0,
      1,
      N,
      hop,
      1,
      FFT_TRANSFORM_C2R,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      1);
  if (!inverse) {
    fft_free(forward);
    throw std::runtime_error("fft_example: fft_init_full (inverse) failed");
  }

  std::vector<float> frames_pcm(static_cast<std::size_t>(frames) * static_cast<std::size_t>(N), 0.0f);
  const size_t recovered = fft_execute_complex_batched(
      inverse,
      real.data(),
      imag.data(),
      produced,
      frames_pcm.data(),
      1,
      1,
      produced);
  if (recovered == 0) {
    fft_free(forward);
    fft_free(inverse);
    throw std::runtime_error("fft_example: fft_execute_complex_batched failed");
  }

  const std::size_t recon_len = static_cast<std::size_t>((frames - 1) * hop + N);
  std::vector<float> recon(recon_len, 0.0f);
  std::vector<float> counts(recon_len, 0.0f);
  for (int f = 0; f < frames; ++f) {
    const std::size_t start = static_cast<std::size_t>(f * hop);
    const float* frame_pcm = frames_pcm.data() + static_cast<std::size_t>(f) * static_cast<std::size_t>(N);
    for (int i = 0; i < N; ++i) {
      recon[start + static_cast<std::size_t>(i)] += frame_pcm[i];
      counts[start + static_cast<std::size_t>(i)] += 1.0f;
    }
  }
  for (std::size_t i = 0; i < recon.size(); ++i) {
    if (counts[i] > 0.0f) {
      recon[i] /= counts[i];
    }
  }

  double max_err = 0.0;
  double max_abs = 0.0;
  const std::size_t compare = std::min<std::size_t>(recon.size(), pcm.size());
  for (std::size_t i = 0; i < compare; ++i) {
    const double a = static_cast<double>(pcm[i]);
    const double b = static_cast<double>(recon[i]);
    max_err = std::max(max_err, std::abs(a - b));
    max_abs = std::max(max_abs, std::abs(a));
  }
  const double rel_err = max_err / (max_abs + 1e-12);

  std::cout << "[cffi-demo] frames=" << frames
            << " bins/frame=" << bins
            << " restores=" << g_cffi_restore_chunks.load(std::memory_order_relaxed)
            << " rel_err=" << rel_err
            << std::endl;

  fft_free(forward);
  fft_free(inverse);
}

// Local dispatcher that forwards to an eigfft::WorkerPool when available.
struct PoolDispatcherLocal : public eigfft::JobDispatcher {
  eigfft::WorkerPool* pool = nullptr;
  void parallel_for(size_t total, size_t chunk, const Fn& fn) override {
    if (!pool || total == 0) {
      eigfft::InlineDispatcher::instance().parallel_for(total, chunk, fn);
      return;
    }
    pool->parallel_for(total, chunk, fn);
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
    pool->parallel_for_with_restore_ex(total, (chunk == 0) ? 1 : chunk, fn,
                                       std::move(restore_fn), ops, job_ctx);
  }
};


static void dump_simd_caps() {
  std::cout << "Pointer size: " << (8 * sizeof(void*)) << "-bit" << std::endl;
#if defined(EIGEN_VECTORIZE_AVX512)
  std::cout << "Eigen SIMD: AVX512" << std::endl;
#elif defined(EIGEN_VECTORIZE_AVX2)
  std::cout << "Eigen SIMD: AVX2" << std::endl;
#elif defined(EIGEN_VECTORIZE_AVX)
  std::cout << "Eigen SIMD: AVX" << std::endl;
#elif defined(EIGEN_VECTORIZE_SSE3) || defined(EIGEN_VECTORIZE_SSE2)
  std::cout << "Eigen SIMD: SSE2/SSE3" << std::endl;
#else
  std::cout << "Eigen SIMD: NONE (scalar)" << std::endl;
#endif
  std::cout << "packet<std::complex<float>>::size  = "
            << Eigen::internal::packet_traits<std::complex<float>>::size << std::endl;
  std::cout << "packet<std::complex<double>>::size = "
            << Eigen::internal::packet_traits<std::complex<double>>::size << std::endl;
}

struct BenchmarkCase {
  int N;
  int B;
  int repeats;
};

static const std::vector<BenchmarkCase> kCases{
  {256, 128, 120},
    {1024, 64, 8},
    {2048, 96, 6},
    {4096, 32, 4},
};

struct AlgorithmSpec {
  eigfft::KernelKind kind;
  const char* label;
};

static const std::vector<AlgorithmSpec> kAlgorithms{
    {eigfft::KernelKind::CooleyTukey, "cooleytukey"},
    {eigfft::KernelKind::Stockham, "stockham-autosort"},
};

struct LaneVariant {
  const char* label;
  int packet_step;  // 0 => auto, otherwise explicit lane width.
};

struct BenchmarkConfig {
  eigfft::PlanRuntimeConfig runtime;
  LaneVariant lanes;
  bool inverse = false;
};

static const std::vector<LaneVariant> kLaneVariants{
    {"lanes=1", 1},
    {"lanes=auto", 0},
};

inline std::string to_lower(std::string value) {
  for (char& ch : value) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value;
}

template <typename Scalar>
struct PrecisionTraits;

template <>
struct PrecisionTraits<double> {
  static std::string label() { return "float64"; }
  static std::string cli_token() { return "f64"; }
};

template <>
struct PrecisionTraits<float> {
  static std::string label() { return "float32"; }
  static std::string cli_token() { return "f32"; }
};

struct RealtimeOptions {
  bool enabled = false;
  double duration_seconds = 10.0;
  double sample_rate_hz = 48000.0;
  int window = 1024;
  int stride = 512;
  double delay_seconds = 0.0;
  double safety_margin = 0.85;
};

template <typename Scalar>
void run_realtime_simulation(std::mt19937_64 seed_rng, const RealtimeOptions& opts,
                             const eigfft::PlanRuntimeConfig& runtime_cfg, bool batched,
                             int window_size, int stride_size, eigfft::KernelKind kernel_kind,
                             const char* kernel_label, eigfft::JobDispatcher* dispatcher = nullptr) {
  if (!opts.enabled) return;

  using Complex = std::complex<Scalar>;
  using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  const double sample_rate = std::min(std::max(opts.sample_rate_hz, 1.0), 1'000'000.0);
  const int window = std::max(window_size, 1);
  const int stride = std::max(stride_size, 1);
  const double safety = std::clamp(opts.safety_margin, 1e-3, 0.999);
  const double duration = std::max(opts.duration_seconds, window / sample_rate);

  const std::size_t total_samples =
      static_cast<std::size_t>(std::ceil(sample_rate * duration));
  if (total_samples < static_cast<std::size_t>(window)) {
    std::cout << "Real-time probe skipped: total samples (" << total_samples
              << ") smaller than window (" << window << ")." << std::endl;
    return;
  }

  const std::size_t frames =
      1 + (total_samples - static_cast<std::size_t>(window)) / static_cast<std::size_t>(stride);
  if (frames == 0) {
    std::cout << "Real-time probe skipped: stride too large for the configured duration."
              << std::endl;
    return;
  }

  std::cout << "\n--- Real-time probe (" << PrecisionTraits<Scalar>::label()
            << ", " << (batched ? "batched" : "sequential") << ", " << kernel_label
            << ", window=" << window << ") ---" << std::endl;
  std::cout << "  sample_rate=" << sample_rate << " Hz, duration=" << duration
            << " s, frames=" << frames << std::endl;
  std::cout << "  window=" << window << ", stride=" << stride
            << ", safety=" << safety << ", delay=" << opts.delay_seconds << " s" << std::endl;

  std::vector<double> pcm(total_samples);
  std::normal_distribution<double> dist(0.0, 1.0);
  auto rng = seed_rng;
  for (std::size_t i = 0; i < total_samples; ++i) {
    pcm[i] = dist(rng);
  }

  eigfft::PlanEnvironment<Scalar> plan_env;
  eigfft::PlanRuntimeConfig env_cfg = runtime_cfg;
  const int packet_guess = std::max(1, static_cast<int>(Eigen::internal::packet_traits<std::complex<Scalar>>::size));
  env_cfg.lanes = std::max(env_cfg.lanes, packet_guess);
  plan_env.initialize(window, /*inverse=*/false, env_cfg);
  if (dispatcher) {
    try {
      plan_env.plan().set_dispatcher(dispatcher);
    } catch (...) {
      // best-effort: do not make the realtime probe fail if dispatcher cannot be installed
    }
  }
  // If an outer dispatcher was provided via runtime_cfg.threads > 1, the
  // caller may have created a WorkerPool and expects plans to dispatch to it.
  // The example creates and installs a dispatcher where appropriate in main().
  auto ensure_lane_capacity = [&](int required_lanes) {
    const int needed = std::max(1, required_lanes);
    if (needed > plan_env.lanes()) {
      env_cfg.lanes = std::max(env_cfg.lanes, needed);
      plan_env.initialize(window, /*inverse=*/false, env_cfg);
    }
  };
  ensure_lane_capacity(packet_guess);
  ensure_lane_capacity(plan_env.plan().packet_cols);
  auto& plan = plan_env.plan();
  plan.tuning.packet_step = std::min(plan.packet_cols, plan_env.lanes());
  plan.tuning.parallel_dim = eigfft::Plan<Scalar>::ParallelDim::Columns;
  plan.tuning.min_work_per_thread = 32;
  plan.tuning.force_ftz_daz = false;
  bool kernel_supported = plan.use_kernel(kernel_kind);
#ifndef NDEBUG
  if (!kernel_supported) {
    std::cout << "Warning: Requested kernel not supported, using default." << std::endl;
  }
#endif

  const double frame_period = stride / sample_rate;
  double cumulative_time = 0.0;
  double worst_frame_time = 0.0;
  double worst_compute_time = 0.0;
  double total_frame_time = 0.0;
  double total_compute_time = 0.0;
  std::size_t overruns = 0;

  const auto delay_duration = std::chrono::duration<double>(opts.delay_seconds);

  if (batched) {
    // Batched processing: create matrix with all frames
    MatrixXc all_frames(window, static_cast<Eigen::Index>(frames));
    for (std::size_t f = 0; f < frames; ++f) {
      const std::size_t start_idx = f * static_cast<std::size_t>(stride);
      for (int i = 0; i < window; ++i) {
        const Scalar real_sample = static_cast<Scalar>(pcm[start_idx + static_cast<std::size_t>(i)]);
        all_frames(i, static_cast<Eigen::Index>(f)) = Complex(real_sample, Scalar(0));
      }
    }

    const auto compute_start = std::chrono::high_resolution_clock::now();
    eigfft::fft_inplace_batched<Scalar>(all_frames, plan);
    const auto compute_end = std::chrono::high_resolution_clock::now();
    const double total_compute_time_batch = std::chrono::duration<double>(compute_end - compute_start).count();

    // For batched, distribute time equally per frame
    const double compute_time_per_frame = total_compute_time_batch / static_cast<double>(frames);
    const double frame_time_per_frame = compute_time_per_frame;  // no delay in batched

    worst_compute_time = compute_time_per_frame;
    total_compute_time = total_compute_time_batch;
    worst_frame_time = frame_time_per_frame;
    total_frame_time = total_compute_time_batch;
    cumulative_time = total_compute_time_batch;  // all at once
    overruns = 0;  // not applicable for batched
  } else {
    // Sequential processing: frame by frame
    MatrixXc frame(window, 1);
    for (std::size_t f = 0; f < frames; ++f) {
      const std::size_t start_idx = f * static_cast<std::size_t>(stride);
      for (int i = 0; i < window; ++i) {
        const Scalar real_sample = static_cast<Scalar>(pcm[start_idx + static_cast<std::size_t>(i)]);
        frame(i, 0) = Complex(real_sample, Scalar(0));
      }

      const auto compute_start = std::chrono::high_resolution_clock::now();
      eigfft::fft_inplace_batched<Scalar>(frame, plan);
      const auto compute_end = std::chrono::high_resolution_clock::now();
      const double compute_time =
          std::chrono::duration<double>(compute_end - compute_start).count();
      worst_compute_time = std::max(worst_compute_time, compute_time);
      total_compute_time += compute_time;

      if (opts.delay_seconds > 0.0) {
        std::this_thread::sleep_for(delay_duration);
      }
      const double frame_time = compute_time + opts.delay_seconds;
      worst_frame_time = std::max(worst_frame_time, frame_time);
      total_frame_time += frame_time;

      cumulative_time += frame_time;
      const double deadline = (static_cast<double>(f) + 1.0) * frame_period;
      if (cumulative_time > deadline) {
        ++overruns;
      }
    }
  }

  const double frames_d = static_cast<double>(frames);
  const double avg_frame_time = total_frame_time / frames_d;
  const double avg_compute_time = total_compute_time / frames_d;

  const double required_frame_period = worst_frame_time / safety;
  const double sustainable_frame_rate = 1.0 / required_frame_period;
  const double sustainable_sample_rate = sustainable_frame_rate * stride;

  const double allowable_time = frame_period * safety;
  const bool meets_config = worst_frame_time <= allowable_time;
  const double slack = allowable_time - worst_frame_time;

  const double avg_refresh_rate = 1.0 / avg_frame_time;

  std::cout << "  avg frame time      : " << avg_frame_time << " s" << std::endl;
  std::cout << "  avg compute time    : " << avg_compute_time << " s" << std::endl;
  std::cout << "  worst frame time    : " << worst_frame_time << " s" << std::endl;
  std::cout << "  worst compute time  : " << worst_compute_time << " s" << std::endl;
  std::cout << "  avg refresh rate    : " << avg_refresh_rate << " Hz" << std::endl;
  std::cout << "  deadline overruns   : " << overruns << std::endl;
  std::cout << "  provided frame period (stride/sample_rate): " << frame_period
            << " s" << std::endl;
  std::cout << "  allowable compute budget (safety * frame_period): " << allowable_time
            << " s" << std::endl;
  std::cout << "  meets configured rate? " << (meets_config ? "yes" : "no")
            << " (slack=" << slack << " s)" << std::endl;
  std::cout << "  max sustainable frame rate (with safety): " << sustainable_frame_rate
            << " Hz" << std::endl;
  std::cout << "  max sustainable sample rate (with safety): " << sustainable_sample_rate
            << " samples/s" << std::endl;
}

template <typename Scalar>
void run_suite_for_precision(std::mt19937_64 seed_rng, const RealtimeOptions& rt_opts,
                             const eigfft::PlanRuntimeConfig& runtime_cfg,
                             const std::vector<AlgorithmSpec>& algorithms,
                             eigfft::JobDispatcher* dispatcher = nullptr) {
  using Complex = std::complex<Scalar>;
  using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  std::cout << "\n=== Precision: " << PrecisionTraits<Scalar>::label() << " ===" << std::endl;

  std::normal_distribution<double> dist(0.0, 1.0);
  auto make_rng = [&](uint64_t seed) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    return rng;
  };

  auto fill_random = [&](MatrixXc& mat, std::mt19937& rng) {
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        const Scalar re = static_cast<Scalar>(dist(rng));
        const Scalar im = static_cast<Scalar>(dist(rng));
        mat(i, j) = Complex(re, im);
      }
    }
  };

  auto run_plan = [&](const MatrixXc& seed,
                            eigfft::PlanEnvironment<Scalar>& env,
                            const BenchmarkConfig& cfg,
                            int repeats) {
    double accum = 0.0;
    MatrixXc work(seed.rows(), seed.cols());
    for (int r = 0; r < repeats; ++r) {
      work = seed;
      const auto start = std::chrono::high_resolution_clock::now();
      // Build a fresh, single-use plan each repeat to avoid any buffer/arena reuse.
      // NOTE: For real applications, use PlanCache instead for efficient reuse!
  env.initialize(static_cast<int>(work.rows()), cfg.inverse, cfg.runtime);
  if (dispatcher) { try { env.plan().set_dispatcher(dispatcher); } catch(...) {} }
  auto& one_shot_plan = env.plan();
      eigfft::fft_inplace_batched<Scalar>(work, one_shot_plan);
      env.discard_plan();
      const auto end = std::chrono::high_resolution_clock::now();
      accum += std::chrono::duration<double>(end - start).count();
    }
    return accum / static_cast<double>(repeats);
  };

  std::cout << std::fixed << std::setprecision(6);

#if EIGFFT_TIMING
  auto __timing_start = eigfft::detail::timing_internal::timing_snapshot();
  auto __timing_prev = __timing_start;
#endif

  for (size_t idx = 0; idx < kCases.size(); ++idx) {
    const auto& task = kCases[idx];
    std::mt19937 rng = make_rng(seed_rng() + static_cast<uint64_t>(idx));

    std::cout << "\n--- Benchmark N=" << task.N << " B=" << task.B
              << " repeats=" << task.repeats << " ---" << std::endl;

    MatrixXc seed(task.N, task.B);
    fill_random(seed, rng);

    for (const auto& algo : algorithms) {
      std::cout << "  Algorithm: " << algo.label << std::endl;
      double reference_time = 0.0;
      const char* reference_label = nullptr;
      bool reference_set = false;

      for (const auto& lanes : kLaneVariants) {
        BenchmarkConfig cfg = {runtime_cfg, lanes, false};
        eigfft::PlanEnvironment<Scalar> temp_env;
        temp_env.initialize(task.N, cfg.inverse, cfg.runtime);
        if (dispatcher) {
          try { temp_env.plan().set_dispatcher(dispatcher); } catch(...) {}
        }
        auto& temp_plan = temp_env.plan();
        const int initial_packets = temp_plan.packet_cols;
        const int desired_lanes = (cfg.lanes.packet_step == 0) ? initial_packets : cfg.lanes.packet_step;
        if (desired_lanes > temp_env.lanes()) {
          cfg.runtime.lanes = std::max(cfg.runtime.lanes, desired_lanes);
          temp_env.initialize(task.N, cfg.inverse, cfg.runtime);
        }
        auto& temp_plan2 = temp_env.plan();
        const int auto_lanes = std::min(temp_plan2.packet_cols, temp_env.lanes());
        const int lane_override = (cfg.lanes.packet_step == 0)
                                      ? auto_lanes
                                      : std::min(cfg.lanes.packet_step, temp_env.lanes());
        temp_plan2.tuning.packet_step = lane_override;
        temp_plan2.tuning.parallel_dim = eigfft::Plan<Scalar>::ParallelDim::Columns;
        temp_plan2.tuning.min_work_per_thread = 32;
        temp_plan2.tuning.force_ftz_daz = false;
        if (!temp_plan2.use_kernel(algo.kind)) {
          std::cout << "    " << cfg.lanes.label << ": unavailable (kernel unsupported)" << std::endl;
          continue;
        }
  eigfft::PlanEnvironment<Scalar> plan_env;
  plan_env.discard_plan();
  // run_plan will install dispatcher on its created plan instance
  const double elapsed = run_plan(seed, plan_env, cfg, task.repeats);
        const int lanes_used = (cfg.lanes.packet_step > 0) ? std::min(cfg.lanes.packet_step, plan_env.lanes()) : plan_env.lanes();
        const int max_threads = plan_env.threads();

        std::cout << "    " << cfg.lanes.label << " (lanes=" << lanes_used
                  << ", max threads=" << max_threads << ") : " << elapsed << " s";

        if (!reference_set) {
          reference_time = elapsed;
          reference_label = cfg.lanes.label;
          reference_set = true;
          std::cout << std::endl;
        } else {
          if (elapsed < reference_time && elapsed > 0.0) {
            std::cout << "  [speedup vs " << reference_label << ": "
                      << (reference_time / elapsed) << "x]";
          } else if (elapsed > reference_time && reference_time > 0.0) {
            std::cout << "  [slowdown vs " << reference_label << ": "
                      << (elapsed / reference_time) << "x]";
          }
          std::cout << std::endl;
        }

#if EIGFFT_TIMING
        {
          std::ostringstream __timing_label_ss;
          __timing_label_ss << "N=" << task.N << " B=" << task.B
                            << " algo=" << algo.label << " lanes=" << cfg.lanes.label;
          auto __timing_now = eigfft::detail::timing_internal::timing_snapshot();
          eigfft::detail::timing_internal::timing_report_delta(__timing_prev, __timing_now, std::cout, __timing_label_ss.str());
          __timing_prev = __timing_now;
        }
#endif
      }
    }
  }

  eigfft::PlanEnvironment<Scalar> probe_env;
  probe_env.initialize(256, /*inverse=*/false, runtime_cfg);
  if (dispatcher) { try { probe_env.plan().set_dispatcher(dispatcher); } catch(...) {} }
  auto& probe_plan = probe_env.plan();
  std::cout << "\nHand-rolled Cooley-Tukey kernel uses Eigen packets (packet_cols="
            << probe_plan.packet_cols
            << ") and OpenMP for batched columns." << std::endl;

  // Run realtime simulations with different configurations
  std::vector<int> window_sizes = {1024, 64, 32};
    for (const auto& algo : algorithms) {
    for (int win : window_sizes) {
      int str = win / 2;  // stride = window / 2 for 50% overlap
      run_realtime_simulation<Scalar>(seed_rng, rt_opts, runtime_cfg, /*batched=*/false, win, str, algo.kind, algo.label, dispatcher);
      run_realtime_simulation<Scalar>(seed_rng, rt_opts, runtime_cfg, /*batched=*/true, win, str, algo.kind, algo.label, dispatcher);
    }
  }
#if EIGFFT_TIMING
  {
    auto __timing_end = eigfft::detail::timing_internal::timing_snapshot();
    eigfft::detail::timing_internal::timing_report_delta(__timing_start, __timing_end, std::cout, std::string("TOTAL"));
  }
  eigfft::detail::timing_internal::timing_report(std::cout);
  eigfft::detail::timing_internal::timing_reset_all();
#endif
}

}

// Example of how to use PlanCache for real applications (efficient reuse)
template <typename Scalar>
void example_fft_with_cache() {
  using Complex = std::complex<Scalar>;
  using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  // Create cache (can be shared across function calls)
  static eigfft::PlanCache<Scalar> cache;
  cache.warm_audio_profiles();

  // Configuration for 1024-point FFT
  eigfft::PlanRuntimeConfig cfg;
  cfg.threads = 4;
  cfg.lanes = 2;

  // Get cached plan (thread-safe, reuses existing plans)
  auto token = cache.get_plan(1024, /*inverse=*/false, cfg);
  auto& plan = token.plan();

  // Create input data
  MatrixXc data(1024, 1);
  // ... fill data ...

  // Execute FFT (plan is reused if same config requested again)
  eigfft::fft_inplace_batched<Scalar>(data, plan);

  // Token automatically releases plan when it goes out of scope
  // Plan remains cached for future use by this thread
}

int main(int argc, char** argv) {
  std::cout << "fftfree micro-benchmark" << std::endl;

  try {
    // Install crash handler early so crashes produce minidumps/backtraces.
    fftfree::install_crash_handler();
    dump_simd_caps();
    Eigen::setNbThreads(1);

  std::unordered_set<std::string> requested;
  std::vector<AlgorithmSpec> selected_algorithms;  // default set below
  RealtimeOptions realtime_opts;
  eigfft::PlanRuntimeConfig runtime_cfg;
  bool run_cffi_demo = true;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      const std::string prefix = "--precision=";
      if (arg.rfind(prefix, 0) == 0) {
        arg.erase(0, prefix.size());
        std::stringstream ss(arg);
        std::string token;
        while (std::getline(ss, token, ',')) {
          requested.insert(to_lower(token));
        }
      } else if (arg.rfind("--threads=", 0) == 0) {
        const int value = std::stoi(arg.substr(10));
        if (value <= 0) {
          std::cerr << "--threads expects a positive integer" << std::endl;
          return 1;
        }
        runtime_cfg.threads = value;
      } else if (arg.rfind("--lanes=", 0) == 0) {
        const int value = std::stoi(arg.substr(8));
        if (value <= 0) {
          std::cerr << "--lanes expects a positive integer" << std::endl;
          return 1;
        }
        runtime_cfg.lanes = value;
      } else if (arg == "--no-cffi-demo") {
        run_cffi_demo = false;
      } else if (arg == "--realtime" || arg == "--rt") {
        realtime_opts.enabled = true;
      } else if (arg.rfind("--algorithms=", 0) == 0) {
        // Comma-separated list of algorithms to run, e.g. ct,stockham
        std::string list = arg.substr(13);
        std::stringstream ss(list);
        std::string token;
        std::unordered_set<std::string> seen;
        selected_algorithms.clear();
        while (std::getline(ss, token, ',')) {
          std::string t = to_lower(token);
          if (t == "ct" || t == "cooleytukey") {
            if (!seen.count("cooleytukey")) {
              selected_algorithms.push_back({eigfft::KernelKind::CooleyTukey, "cooleytukey"});
              seen.insert("cooleytukey");
            }
          } else if (t == "stockham" || t == "stockham-autosort") {
            if (!seen.count("stockham-autosort")) {
              selected_algorithms.push_back({eigfft::KernelKind::Stockham, "stockham-autosort"});
              seen.insert("stockham-autosort");
            }
          } else if (!t.empty()) {
            std::cerr << "Unknown algorithm token '" << t
                      << "'. Supported: ct, cooleytukey, stockham." << std::endl;
            return 1;
          }
        }
      } else if (arg.rfind("--rt-sample-rate=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.sample_rate_hz = std::stod(arg.substr(17));
      } else if (arg.rfind("--rt-window=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.window = std::stoi(arg.substr(12));
      } else if (arg.rfind("--rt-stride=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.stride = std::stoi(arg.substr(12));
      } else if (arg.rfind("--rt-delay-ms=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.delay_seconds = std::stod(arg.substr(14)) / 1000.0;
      } else if (arg.rfind("--rt-safety=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.safety_margin = std::stod(arg.substr(12));
      } else if (arg.rfind("--rt-duration=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.duration_seconds = std::stod(arg.substr(14));
      } else if (arg == "--help" || arg == "-h") {
  std::cout << "Usage: fft_example [--precision=f64,f32] [--threads=N] [--lanes=M] [--algorithms=list]\n"
         "  f64 : std::complex<double>\n"
         "  f32 : std::complex<float>\n"
         "Default is to run both precisions.\n"
         "\nRuntime configuration (clamped to build limits):\n"
         "  --threads=N                      Max worker threads (default 4, max 16)\n"
         "  --lanes=M                        Stockham lane capacity (default 2)\n"
         "  --algorithms=list                Comma-separated list: ct, stockham (default: ct only)\n"
                     "  --no-cffi-demo                  Skip the fft_cffi interface smoke test\n"
                     "\nReal-time probe options (auto-enable realtime mode):\n"
                     "  --realtime | --rt                 Enable real-time simulation\n"
                     "  --rt-sample-rate=<Hz>             Input sample rate (default 48000, max 1e6)\n"
                     "  --rt-window=<samples>             FFT window size (default 1024)\n"
                     "  --rt-stride=<samples>             Hop size between frames (default 512)\n"
                     "  --rt-duration=<seconds>           PCM duration (default 10)\n"
                     "  --rt-delay-ms=<milliseconds>      Extra delay per frame (default 0)\n"
                     "  --rt-safety=<0-1)                 Fraction of frame period usable for compute (default 0.85)\n"
                     "\nFor real applications, use eigfft::PlanCache for efficient plan reuse.\n"
                     "See example_fft_with_cache() for usage pattern.\n";
        return 0;
      } else {
        std::cerr << "Unrecognized argument: " << arg << std::endl;
        return 1;
      }
    }

    if (requested.empty()) { requested = {"f64", "f32"}; }

    // Default algorithms: Cooley–Tukey only, unless overridden by --algorithms
    if (selected_algorithms.empty()) {
      selected_algorithms.push_back({eigfft::KernelKind::CooleyTukey, "cooleytukey"});
    }

    const std::vector<std::string> known = {"f64", "f32"};
    for (const auto& token : requested) {
      if (std::find(known.begin(), known.end(), token) == known.end()) {
        std::cerr << "Unknown precision token '" << token
                  << "'. Supported tokens: f64,f32." << std::endl;
        return 1;
      }
    }

    std::mt19937_64 seed_rng(1337);

    // Create a WorkerPool and dispatcher for the example to demonstrate
    // outer parallel dispatch when runtime_cfg.threads > 1.
    PoolDispatcherLocal local_dispatcher;
    std::unique_ptr<eigfft::WorkerPool> local_pool;
    if (runtime_cfg.threads > 1) {
      try {
        local_pool = std::make_unique<eigfft::WorkerPool>(runtime_cfg.threads);
        local_dispatcher.pool = local_pool.get();
      } catch (...) {
        local_pool.reset();
        local_dispatcher.pool = nullptr;
      }
    }

    if (requested.count("f64")) {
      run_suite_for_precision<double>(seed_rng, realtime_opts, runtime_cfg, selected_algorithms, (local_dispatcher.pool ? &local_dispatcher : nullptr));
    }
    if (requested.count("f32")) {
      run_suite_for_precision<float>(seed_rng, realtime_opts, runtime_cfg, selected_algorithms, (local_dispatcher.pool ? &local_dispatcher : nullptr));
    }

    if (run_cffi_demo) {
      try {
        run_cffi_recovery_demo((runtime_cfg.threads > 0) ? runtime_cfg.threads : 1);
      } catch (const std::exception& demo_err) {
        std::cerr << "[cffi-demo] " << demo_err.what() << std::endl;
      }
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
