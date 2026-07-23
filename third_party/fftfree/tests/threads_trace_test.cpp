// Restrict changes to this file only per user request.

#include "../eigen_fft.hpp"
#include "../plan_support.hpp"
#include "../fft_runtime_api.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct ThreadConfigDesc {
  eigfft::PlanRuntimeConfig cfg;
  std::string label;
};

template <typename Scalar>
static void run_thread_config_batch(const ThreadConfigDesc& desc, int N, int B) {
  using Complex = std::complex<Scalar>;
  using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  // Describe the configuration clearly.
  const auto& c = desc.cfg;
  std::cout << "\n=== Thread Configuration (batch): " << desc.label << " ===\n";
  std::cout << "  threads=" << c.threads
            << ", lanes=" << c.lanes
            << ", allow_outer_parallel=" << (c.allow_outer_parallel ? "true" : "false")
            << ", allow_inner_parallel=" << (c.allow_inner_parallel ? "true" : "false")
            << ", inner_threads=" << c.inner_threads
            << std::endl;

  // Prepare a large batch.
  MatrixXc X(N, B);
  std::mt19937 rng(123456);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int ccol = 0; ccol < B; ++ccol) {
    for (int r = 0; r < N; ++r) {
      X(r, ccol) = Complex(dist(rng), dist(rng));
    }
  }

  // Prepare transform via public runtime API (no direct dispatcher/pool wiring).
  eigfft::TransformSettings<Scalar> settings;
  settings.fft_size = N;
  settings.inverse = false;
  settings.kernel_hint = eigfft::KernelKind::Stockham;
  settings.runtime = desc.cfg;
  settings.placement = eigfft::TransformPlacement::InPlace;

  // Optionally, inspect effective threads reported by a plan from the cache for transparency.
  {
    eigfft::PlanCache<Scalar> cache;
    auto token = cache.get_plan(N, /*inverse=*/false, desc.cfg);
    int eff = token.plan().effective_threads(B);
    std::cout << "  effective_threads(report)=" << eff << std::endl;
  }

  eigfft::PreparedTransform<Scalar> prep(settings);
  auto t0 = std::chrono::high_resolution_clock::now();
  (void)prep.run(X);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "  result: ok, elapsed=" << std::fixed << std::setprecision(3) << secs << " s\n";
}

// Streaming-style exercise: construct overlapped frames from a PCM buffer and
// process in chunks of frames_per_iter to emulate realtime streaming, using only
// the public transform API.
template <typename Scalar>
static void run_thread_config_streaming(const ThreadConfigDesc& desc, int W, int H, int total_frames, int frames_per_iter) {
  using Complex = std::complex<Scalar>;
  using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  const int pcm_len = W + H * std::max(0, total_frames - 1);
  std::vector<Scalar> pcm(static_cast<std::size_t>(pcm_len));
  std::mt19937 rng(424242);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < pcm_len; ++i) pcm[i] = static_cast<Scalar>(dist(rng));

  std::cout << "\n=== Thread Configuration (streaming): " << desc.label << " ===\n";
  std::cout << "  window=" << W << ", hop=" << H
            << ", total_frames=" << total_frames
            << ", frames_per_iter=" << frames_per_iter << "\n";

  eigfft::TransformSettings<Scalar> settings;
  settings.fft_size = W;
  settings.inverse = false;
  settings.kernel_hint = eigfft::KernelKind::Stockham;
  settings.runtime = desc.cfg;
  settings.placement = eigfft::TransformPlacement::InPlace;
  eigfft::PreparedTransform<Scalar> prep(settings);

  auto t0 = std::chrono::high_resolution_clock::now();
  int produced = 0;
  for (int base = 0; base < total_frames; base += frames_per_iter) {
    const int chunk = std::min(frames_per_iter, total_frames - base);
    MatrixXc X(W, chunk);
    for (int f = 0; f < chunk; ++f) {
      const int frame_index = base + f;
      const int start = frame_index * H;
      for (int r = 0; r < W; ++r) {
        X(r, f) = Complex(pcm[start + r], Scalar(0));
      }
    }
    (void)prep.run(X);
    produced += chunk;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "  produced=" << produced << " frames, elapsed="
            << std::fixed << std::setprecision(3) << secs << " s\n";
}

int main() {
  const int N = 2048;   // length
  const int B = 4096;   // large batch to exercise chunking

  const int hw = std::max(1u, std::thread::hardware_concurrency());
  const int cap_threads = std::min(eigfft::Plan<float>::Limits::kCompileTimeMaxThreads, std::max(1, hw));
  const int packet = std::max(1, (int)Eigen::internal::packet_traits<std::complex<float>>::size);

  std::vector<ThreadConfigDesc> configs;

  // 1) Outer disabled, single-thread (baseline)
  {
    eigfft::PlanRuntimeConfig c{};
    c.threads = 1;
    c.lanes = packet;
    c.allow_outer_parallel = false;
    c.allow_inner_parallel = false;
    c.inner_threads = 0;
    configs.push_back({c, "outer=off, inner=off, threads=1"});
  }

  // 2) Outer enabled with 2 and 4 threads
  for (int t : std::vector<int>{2, 4}) {
    if (t > cap_threads) continue;
    eigfft::PlanRuntimeConfig c{};
    c.threads = t;
    c.lanes = packet;
    c.allow_outer_parallel = true;
    c.allow_inner_parallel = false;
    c.inner_threads = 0;
    std::string lbl = std::string("outer=on, inner=off, threads=") + std::to_string(t);
    configs.push_back({c, lbl});
  }

  // 3) Outer off, inner on with 2 and 4 inner threads (API-level intent)
  for (int it : std::vector<int>{2, 4}) {
    eigfft::PlanRuntimeConfig c{};
    c.threads = 1;
    c.lanes = packet;
    c.allow_outer_parallel = false;
    c.allow_inner_parallel = true;
    c.inner_threads = it;
    std::string lbl = std::string("outer=off, inner=on, inner_threads=") + std::to_string(it);
    configs.push_back({c, lbl});
  }

  // 4) Outer and inner both on (favoring outer for dispatch) with 4 threads
  if (cap_threads >= 4) {
    eigfft::PlanRuntimeConfig c{};
    c.threads = 4;
    c.lanes = packet;
    c.allow_outer_parallel = true;
    c.allow_inner_parallel = true;
    c.inner_threads = 2;
    configs.push_back({c, std::string("outer=on(4), inner=on(2)")});
  }

  // Run all configurations in batch and streaming modes.
  for (const auto& d : configs) {
    run_thread_config_batch<float>(d, N, B);
    // Streaming parameters (aligned with tools/stress_dispatch patterns)
    const int W = N;      // window size
    const int H = N / 2;  // 50% overlap
    const int total_frames = 256;
    const int frames_per_iter = 32;
    run_thread_config_streaming<float>(d, W, H, total_frames, frames_per_iter);
  }

  std::cout << "\nThread trace test complete." << std::endl;
  return 0;
}
