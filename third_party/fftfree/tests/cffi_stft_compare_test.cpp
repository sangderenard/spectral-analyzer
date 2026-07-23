#include "../fft_cffi.hpp"
#include "../eigen_fft.hpp"
#include "../plan_support.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Complex = std::complex<float>;

constexpr float kTolerance = 1e-4f;

std::vector<float> make_hann(int W) {
  constexpr double kPi = 3.14159265358979323846;
  std::vector<float> w(static_cast<std::size_t>(W));
  if (W <= 1) {
    std::fill(w.begin(), w.end(), 1.0f);
    return w;
  }
  const double denom = static_cast<double>(W - 1);
  for (int n = 0; n < W; ++n) {
    w[static_cast<std::size_t>(n)] =
        static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(n) / denom)));
  }
  return w;
}

std::vector<Complex> run_eigen_fft(const std::vector<float>& pcm, int N) {
  Eigen::Matrix<Complex, Eigen::Dynamic, 1> column(static_cast<Eigen::Index>(N));
  for (int i = 0; i < N; ++i) {
    const float sample = (i < static_cast<int>(pcm.size())) ? pcm[static_cast<std::size_t>(i)] : 0.0f;
    column(static_cast<Eigen::Index>(i)) = Complex(sample, 0.0f);
  }

  eigfft::PlanRuntimeConfig cfg;
  cfg.threads = 1;
  cfg.lanes = 1;
  cfg.allow_inner_parallel = false;
  cfg.inner_threads = 0;

  eigfft::PlanCache<float> cache;
  auto token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto& plan = token.plan();
  eigfft::fft_inplace_batched<float>(column, plan);

  std::vector<Complex> result(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    result[static_cast<std::size_t>(i)] = column(static_cast<Eigen::Index>(i));
  }
  return result;
}

std::vector<Complex> run_eigen_stft(const std::vector<float>& pcm,
                                    int N,
                                    int W,
                                    int H,
                                    const std::vector<float>& window,
                                    int frames) {
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> matrix(
      static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(frames));
  matrix.setZero();

  for (int f = 0; f < frames; ++f) {
    const std::size_t base = static_cast<std::size_t>(f) * static_cast<std::size_t>(H);
    for (int i = 0; i < W; ++i) {
      const std::size_t idx = base + static_cast<std::size_t>(i);
      float sample = 0.0f;
      if (idx < pcm.size()) {
        sample = pcm[idx];
      }
      float windowed = sample;
      if (!window.empty()) {
        windowed *= window[static_cast<std::size_t>(i)];
      }
      matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(f)) = Complex(windowed, 0.0f);
    }
  }

  eigfft::PlanRuntimeConfig cfg;
  cfg.threads = 1;
  cfg.lanes = 1;
  cfg.allow_inner_parallel = false;
  cfg.inner_threads = 0;

  eigfft::PlanCache<float> cache;
  auto token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto& plan = token.plan();
  eigfft::fft_inplace_batched<float>(matrix, plan);

  std::vector<Complex> result(static_cast<std::size_t>(frames) * static_cast<std::size_t>(N));
  for (int f = 0; f < frames; ++f) {
    for (int i = 0; i < N; ++i) {
      result[static_cast<std::size_t>(f) * static_cast<std::size_t>(N) + static_cast<std::size_t>(i)] =
          matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(f));
    }
  }
  return result;
}

double max_abs_diff(const std::vector<Complex>& a, const std::vector<Complex>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double max_err = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double err = std::abs(a[i] - b[i]);
    if (err > max_err) {
      max_err = err;
    }
  }
  return max_err;
}

void fill_random(std::vector<float>& pcm, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (float& sample : pcm) {
    sample = dist(rng);
  }
}

std::vector<Complex> collect_cffi_fft(void* ctx, const std::vector<float>& pcm) {
  const std::size_t N = fft_ctx_size(ctx);
  std::vector<float> out_real(N);
  std::vector<float> out_imag(N);
  std::vector<float> out_mag(N);
  if (!fft_execute(ctx,
                   pcm.data(),
                   out_real.data(),
                   out_imag.data(),
                   out_mag.data(),
                   pcm.size())) {
    throw std::runtime_error("fft_execute failed");
  }
  std::vector<Complex> result(N);
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = Complex(out_real[i], out_imag[i]);
  }
  return result;
}

std::vector<Complex> collect_cffi_batched(void* ctx,
                                          const std::vector<float>& pcm,
                                          int H,
                                          int& frames_out) {
  const std::size_t N = fft_ctx_size(ctx);
  const std::size_t bins = N;
  const std::size_t max_frames = (pcm.size() / static_cast<std::size_t>(std::max(1, H))) + 4;
  std::vector<float> out_real(max_frames * bins, 0.0f);
  std::vector<float> out_imag(max_frames * bins, 0.0f);
  std::vector<float> out_mag(max_frames * bins, 0.0f);

  const std::size_t produced = fft_execute_batched(ctx,
                                                   pcm.data(),
                                                   pcm.size(),
                                                   out_real.data(),
                                                   out_imag.data(),
                                                   out_mag.data(),
                                                   /*pad_mode=*/1,
                                                   /*enable_backup=*/0,
                                                   /*max_frames=*/0);
  if (produced == 0) {
    throw std::runtime_error("fft_execute_batched produced no frames");
  }
  frames_out = static_cast<int>(produced);
  std::vector<Complex> result(produced * bins);
  for (std::size_t f = 0; f < produced; ++f) {
    for (std::size_t i = 0; i < bins; ++i) {
      const std::size_t base = f * bins + i;
      result[base] = Complex(out_real[base], out_imag[base]);
    }
  }
  return result;
}

std::vector<Complex> collect_cffi_streaming(void* ctx,
                                            const std::vector<float>& pcm,
                                            int N,
                                            int W,
                                            int H,
                                            int total_frames,
                                            int chunk_frames) {
  const std::size_t bins = static_cast<std::size_t>(N);
  std::vector<Complex> aggregated(static_cast<std::size_t>(total_frames) * bins);
  std::size_t frames_done = 0;

  for (; frames_done < static_cast<std::size_t>(total_frames);) {
    const int remaining = total_frames - static_cast<int>(frames_done);
    const int frames_this = std::min(chunk_frames, remaining);
    const std::size_t start_sample = frames_done * static_cast<std::size_t>(H);
    const std::size_t needed_samples = static_cast<std::size_t>(W) + static_cast<std::size_t>(std::max(0, frames_this - 1)) * static_cast<std::size_t>(H);

    std::vector<float> chunk(needed_samples, 0.0f);
    for (std::size_t i = 0; i < needed_samples; ++i) {
      const std::size_t idx = start_sample + i;
      if (idx < pcm.size()) {
        chunk[i] = pcm[idx];
      }
    }

    const std::size_t max_chunk_frames = static_cast<std::size_t>(frames_this + 1);
    std::vector<float> out_real(max_chunk_frames * bins, 0.0f);
    std::vector<float> out_imag(max_chunk_frames * bins, 0.0f);
    std::vector<float> out_mag(max_chunk_frames * bins, 0.0f);

    const std::size_t produced = fft_execute_batched(ctx,
                                                     chunk.data(),
                                                     chunk.size(),
                                                     out_real.data(),
                                                     out_imag.data(),
                                                     out_mag.data(),
                                                     /*pad_mode=*/1,
                                                     /*enable_backup=*/0,
                                                     /*max_frames=*/0);
    if (produced != static_cast<std::size_t>(frames_this)) {
      throw std::runtime_error("Streaming STFT produced unexpected frame count");
    }

    for (std::size_t f = 0; f < produced; ++f) {
      for (std::size_t i = 0; i < bins; ++i) {
        const std::size_t src = f * bins + i;
        const std::size_t dst = (frames_done + f) * bins + i;
        aggregated[dst] = Complex(out_real[src], out_imag[src]);
      }
    }

    frames_done += produced;
  }
  return aggregated;
}

bool test_plain_fft_vs_eigen() {
  const int N = 1024;
  std::vector<float> pcm(static_cast<std::size_t>(N));
  std::mt19937 rng(1337);
  fill_random(pcm, rng);

  void* ctx = fft_init_full_v2(
      N,
      /*threads=*/1,
      /*lanes=*/1,
      /*inverse=*/0,
      /*kernel=*/0,
      /*radix=*/0,
      /*radix_pattern=*/nullptr,
      /*radix_pattern_len=*/0,
      /*pad_mode=*/1,
      /*window=*/0,
      /*hop=*/0,
      /*stft_mode=*/0,
      FFT_TRANSFORM_R2C,
      /*reduce_magnitude=*/0,
      /*store_polar=*/0,
      /*half_spectrum=*/0,
      /*allow_outer_parallel=*/0,
      /*allow_inner_parallel=*/0,
      /*inner_threads=*/0,
      /*save_crash_logs=*/0,
      /*silent_crash_reports=*/1,
      /*apply_windows=*/0,
      /*apply_ola=*/0,
      FFT_WINDOW_RECT,
      /*analysis_param1=*/0.0f,
      /*analysis_param2=*/0.0f,
      FFT_WINDOW_RECT,
      /*synthesis_param1=*/0.0f,
      /*synthesis_param2=*/0.0f,
      FFT_WINDOW_NORM_NONE,
      FFT_COLA_OFF);

  if (!ctx) {
    std::cerr << "plain FFT: failed to initialize context\n";
    return false;
  }

  bool ok = true;
  try {
    auto cffi = collect_cffi_fft(ctx, pcm);
    auto eigen = run_eigen_fft(pcm, N);
    const double err = max_abs_diff(cffi, eigen);
    std::cout << "No-window FFT max abs error=" << err << '\n';
    if (err > kTolerance) {
      std::cerr << "plain FFT mismatch (max abs error=" << err << ")\n";
      ok = false;
    }
  } catch (const std::exception& ex) {
    std::cerr << "plain FFT exception: " << ex.what() << '\n';
    ok = false;
  }

  fft_free(ctx);
  return ok;
}

bool test_batched_stft_vs_eigen(int& out_frames, std::vector<float>& pcm_out, std::vector<Complex>& eigen_out) {
  const int N = 512;
  const int W = 512;
  const int H = 128;
  const int planned_frames = 8;
  const std::size_t pcm_len = static_cast<std::size_t>(W) + static_cast<std::size_t>(planned_frames - 1) * static_cast<std::size_t>(H);

  pcm_out.assign(pcm_len, 0.0f);
  std::mt19937 rng(42);
  fill_random(pcm_out, rng);

  void* ctx = fft_init_full_v2(
      N,
      /*threads=*/1,
      /*lanes=*/1,
      /*inverse=*/0,
      /*kernel=*/0,
      /*radix=*/0,
      /*radix_pattern=*/nullptr,
      /*radix_pattern_len=*/0,
      /*pad_mode=*/1,
      /*window=*/W,
      /*hop=*/H,
      /*stft_mode=*/1,
      FFT_TRANSFORM_R2C,
      /*reduce_magnitude=*/0,
      /*store_polar=*/0,
      /*half_spectrum=*/0,
      /*allow_outer_parallel=*/0,
      /*allow_inner_parallel=*/0,
      /*inner_threads=*/0,
      /*save_crash_logs=*/0,
      /*silent_crash_reports=*/1,
      /*apply_windows=*/1,
      /*apply_ola=*/0,
      FFT_WINDOW_HANN,
      /*analysis_param1=*/0.0f,
      /*analysis_param2=*/0.0f,
      FFT_WINDOW_HANN,
      /*synthesis_param1=*/0.0f,
      /*synthesis_param2=*/0.0f,
      FFT_WINDOW_NORM_NONE,
      FFT_COLA_OFF);

  if (!ctx) {
    std::cerr << "batched STFT: failed to initialize context\n";
    return false;
  }

  bool ok = true;
  try {
    int frames = 0;
    auto cffi = collect_cffi_batched(ctx, pcm_out, H, frames);
    out_frames = frames;

    auto window = make_hann(W);
    eigen_out = run_eigen_stft(pcm_out, N, W, H, window, frames);
    const double err = max_abs_diff(cffi, eigen_out);
    std::cout << "Batched STFT max abs error=" << err << " (frames=" << frames << ")\n";
    if (err > kTolerance) {
      std::cerr << "batched STFT mismatch (max abs error=" << err << ")\n";
      ok = false;
    }
  } catch (const std::exception& ex) {
    std::cerr << "batched STFT exception: " << ex.what() << '\n';
    ok = false;
  }

  fft_free(ctx);
  return ok;
}

bool test_streaming_stft_vs_eigen(const std::vector<float>& pcm,
                                  int N,
                                  int W,
                                  int H,
                                  int total_frames,
                                  const std::vector<Complex>& eigen_reference) {
  void* ctx = fft_init_full_v2(
      N,
      /*threads=*/1,
      /*lanes=*/1,
      /*inverse=*/0,
      /*kernel=*/0,
      /*radix=*/0,
      /*radix_pattern=*/nullptr,
      /*radix_pattern_len=*/0,
      /*pad_mode=*/1,
      /*window=*/W,
      /*hop=*/H,
      /*stft_mode=*/1,
      FFT_TRANSFORM_R2C,
      /*reduce_magnitude=*/0,
      /*store_polar=*/0,
      /*half_spectrum=*/0,
      /*allow_outer_parallel=*/0,
      /*allow_inner_parallel=*/0,
      /*inner_threads=*/0,
      /*save_crash_logs=*/0,
      /*silent_crash_reports=*/1,
      /*apply_windows=*/1,
      /*apply_ola=*/0,
      FFT_WINDOW_HANN,
      /*analysis_param1=*/0.0f,
      /*analysis_param2=*/0.0f,
      FFT_WINDOW_HANN,
      /*synthesis_param1=*/0.0f,
      /*synthesis_param2=*/0.0f,
      FFT_WINDOW_NORM_NONE,
      FFT_COLA_OFF);

  if (!ctx) {
    std::cerr << "streaming STFT: failed to initialize context\n";
    return false;
  }

  bool ok = true;
  try {
    const int chunk_frames = 3;
    auto streaming = collect_cffi_streaming(ctx, pcm, N, W, H, total_frames, chunk_frames);
    const double err = max_abs_diff(streaming, eigen_reference);
    std::cout << "Streaming STFT max abs error=" << err << '\n';
    if (err > kTolerance) {
      std::cerr << "streaming STFT mismatch (max abs error=" << err << ")\n";
      ok = false;
    }
  } catch (const std::exception& ex) {
    std::cerr << "streaming STFT exception: " << ex.what() << '\n';
    ok = false;
  }

  fft_free(ctx);
  return ok;
}

}  // namespace

int main() {
  bool ok = true;

  if (!test_plain_fft_vs_eigen()) {
    ok = false;
  }

  int frames = 0;
  std::vector<float> pcm;
  std::vector<Complex> eigen_ref;
  if (!test_batched_stft_vs_eigen(frames, pcm, eigen_ref)) {
    ok = false;
  }

  if (!eigen_ref.empty() && frames > 0) {
    if (!test_streaming_stft_vs_eigen(pcm, /*N=*/512, /*W=*/512, /*H=*/128, frames, eigen_ref)) {
      ok = false;
    }
  } else {
    std::cerr << "Skipping streaming STFT test due to missing reference" << '\n';
    ok = false;
  }

  if (ok) {
    std::cout << "All CFFI vs Eigen comparisons passed." << std::endl;
  }
  return ok ? 0 : 1;
}
