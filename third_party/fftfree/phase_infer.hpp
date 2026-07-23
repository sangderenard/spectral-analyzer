// phase_infer.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "fft_cffi.hpp"

namespace phaseinfer {

struct PhaseConfig {
  int N = 0;            // plan size (power of two preferred)
  int hop = 0;          // hop size between frames
  bool half_spectrum = true; // input magnitude uses N/2+1 bins if true, else N
  int mode = 0;         // 0=linear (deterministic), reserved for future modes
  int iterations = 0;   // reserved for future iterative solvers (e.g., Griffin–Lim)
};

// Simple, non-iterative phase inference that propagates a linear phase advance
// per bin across frames: phi[f,k] = phi0[k] + f * (2*pi*k*hop/N). The initial
// phase phi0 is zero. This preserves inter-frame coherence for stationary bins
// and avoids checkerboard artifacts seen with per-frame zero phase.
//
// Inputs:
//  - mag: frame-major magnitudes of shape (frames, bins_per_frame)
//  - frames: number of frames (columns)
//  - cfg: plan/inference configuration
// Outputs:
//  - out_real,out_imag: frame-major complex arrays sized frames * bins_per_frame
//
// Bins per frame is N when cfg.half_spectrum==false, or N/2+1 otherwise.
inline void infer_linear(const float* mag,
                         size_t frames,
                         const PhaseConfig& cfg,
                         float* out_real,
                         float* out_imag) {
  const int N = cfg.N;
  const int hop = (cfg.hop > 0) ? cfg.hop : N;
  const int bins = cfg.half_spectrum ? (N/2 + 1) : N;
  if (N <= 0 || bins <= 0 || frames == 0) return;
  const double two_pi = 6.28318530717958647692;
  // Per-bin phase increment
  std::vector<double> dphi(static_cast<size_t>(bins));
  for (int k = 0; k < bins; ++k) {
    // Clamp k for half-spectrum edge cases
    const int kk = (k < N) ? k : (k % N);
    dphi[static_cast<size_t>(k)] = two_pi * static_cast<double>(kk) * static_cast<double>(hop) / static_cast<double>(N);
  }
  // Accumulated phase per bin
  std::vector<double> phi(static_cast<size_t>(bins), 0.0);
  for (size_t f = 0; f < frames; ++f) {
    const size_t base = f * static_cast<size_t>(bins);
    for (int k = 0; k < bins; ++k) {
      const double m = static_cast<double>(mag[base + static_cast<size_t>(k)]);
      const double c = std::cos(phi[static_cast<size_t>(k)]);
      const double s = std::sin(phi[static_cast<size_t>(k)]);
      out_real[base + static_cast<size_t>(k)] = static_cast<float>(m * c);
      out_imag[base + static_cast<size_t>(k)] = static_cast<float>(m * s);
      phi[static_cast<size_t>(k)] += dphi[static_cast<size_t>(k)];
    }
  }
}

// Griffin–Lim style phase reconstruction using fft_cffi batched plans.
// Returns true on success; out_real/out_imag contain the refined spectra.
inline bool infer_griffin_lim(const float* mag,
                              size_t frames,
                              const PhaseConfig& cfg,
                              void* ctx_forward,
                              void* ctx_inverse,
                              float* out_real,
                              float* out_imag,
                              int iterations_override = 0,
                              int pad_mode = 1,
                              unsigned int seed = 0x47d7bfad) {
  if (!mag || !out_real || !out_imag || frames == 0 || !ctx_forward || !ctx_inverse) {
    return false;
  }

  const size_t plan_forward = fft_ctx_size(ctx_forward);
  const size_t plan_inverse = fft_ctx_size(ctx_inverse);
  size_t plan_n = plan_forward ? plan_forward : plan_inverse;
  if (plan_n == 0 && cfg.N > 0) {
    plan_n = static_cast<size_t>(cfg.N);
  }
  if (plan_n == 0) {
    return false;
  }

  if (plan_inverse && plan_forward && plan_inverse != plan_forward) {
    // Prefer the inverse plan size if they disagree.
    plan_n = plan_inverse;
  }

  const size_t bins = cfg.half_spectrum ? (plan_n / 2 + 1) : plan_n;
  const size_t total_bins = bins * frames;
  if (total_bins == 0) {
    return false;
  }

  const size_t hop = (cfg.hop > 0) ? static_cast<size_t>(cfg.hop) : plan_n;
  const size_t ola_len = (frames > 0) ? ((frames - 1) * hop + plan_n) : plan_n;
  if (ola_len == 0) {
    return false;
  }

  std::vector<float> frame_pcm(frames * plan_n, 0.0f);
  std::vector<float> ola(ola_len, 0.0f);
  std::vector<float> ola_weight(ola_len, 0.0f);
  std::vector<float> tmp_real(total_bins, 0.0f);
  std::vector<float> tmp_imag(total_bins, 0.0f);
  std::vector<float> tmp_mag(total_bins, 0.0f);

  const int iterations_cfg = (cfg.iterations > 0) ? cfg.iterations : 0;
  const int iterations = (iterations_override > 0) ? iterations_override : (iterations_cfg > 0 ? iterations_cfg : 32);
  if (iterations <= 0) {
    return false;
  }

  constexpr float kPi = 3.14159265358979323846f;
  std::mt19937 rng(seed ? seed : 0x47d7bfad);
  std::uniform_real_distribution<float> dist(-kPi, kPi);

  for (size_t i = 0; i < total_bins; ++i) {
    const float angle = dist(rng);
    const float magnitude = mag[i];
    out_real[i] = magnitude * std::cos(angle);
    out_imag[i] = magnitude * std::sin(angle);
  }

  for (int iter = 0; iter < iterations; ++iter) {
  const size_t produced_frames = fft_execute_complex_batched(
        ctx_inverse,
        out_real,
        out_imag,
        frames,
        frame_pcm.data(),
        pad_mode,
    0,
    frames);
    if (produced_frames == 0) {
      return false;
    }

    std::fill(ola.begin(), ola.end(), 0.0f);
    std::fill(ola_weight.begin(), ola_weight.end(), 0.0f);

    const size_t usable_frames = std::min(produced_frames, frames);
    for (size_t f = 0; f < usable_frames; ++f) {
      const float* src = frame_pcm.data() + f * plan_n;
      const size_t start = f * hop;
      if (start + plan_n > ola_len) {
        break;
      }
      for (size_t n = 0; n < plan_n; ++n) {
        ola[start + n] += src[n];
        ola_weight[start + n] += 1.0f;
      }
    }
    for (size_t n = 0; n < ola_len; ++n) {
      if (ola_weight[n] > 0.0f) {
        ola[n] /= ola_weight[n];
      }
    }

  const size_t forward_frames = fft_execute_batched(
        ctx_forward,
        ola.data(),
        ola_len,
        tmp_real.data(),
        tmp_imag.data(),
        tmp_mag.data(),
        pad_mode,
    0,
    frames);
    if (forward_frames == 0) {
      return false;
    }

    const size_t update_frames = std::min(forward_frames, frames);
    const size_t update_bins = update_frames * bins;
    for (size_t i = 0; i < update_bins; ++i) {
      const float phase = std::atan2(tmp_imag[i], tmp_real[i]);
      const float magnitude = mag[i];
      out_real[i] = magnitude * std::cos(phase);
      out_imag[i] = magnitude * std::sin(phase);
    }

    if (cfg.half_spectrum) {
      for (size_t f = 0; f < update_frames; ++f) {
        const size_t base = f * bins;
        out_imag[base] = 0.0f;
        if ((plan_n & 1) == 0 && bins > 1) {
          out_imag[base + bins - 1] = 0.0f;
        }
      }
    }
  }

  return true;
}

} // namespace phaseinfer

