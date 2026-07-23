#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace fftfree::ft {

struct GridSpec {
  double time_origin = 0.0;
  double freq_origin = 0.0;
  double time_step = 1.0;
  double freq_step = 1.0;

  double time_at(int time_index) const {
    return time_origin + time_step * static_cast<double>(time_index);
  }

  double freq_at(int freq_index) const {
    return freq_origin + freq_step * static_cast<double>(freq_index);
  }
};

template <typename T>
constexpr T clamp_value(T v, T lo, T hi) {
  return std::max(lo, std::min(v, hi));
}

template <typename T>
T lerp_value(T a, T b, T t) {
  return a + (b - a) * t;
}

template <typename T>
std::complex<T> lerp_complex(const std::complex<T>& a, const std::complex<T>& b, T t) {
  return std::complex<T>(lerp_value<T>(a.real(), b.real(), t), lerp_value<T>(a.imag(), b.imag(), t));
}

template <typename T>
struct ComplexGrid {
  std::vector<std::complex<T>> values;
  int time_bins = 0;
  int freq_bins = 0;
  GridSpec grid{};

  ComplexGrid() = default;

  ComplexGrid(int time_count, int freq_count, GridSpec spec = {})
      : values(static_cast<std::size_t>(time_count) * static_cast<std::size_t>(freq_count)),
        time_bins(time_count),
        freq_bins(freq_count),
        grid(spec) {}

  bool empty() const { return values.empty(); }

  std::size_t size() const { return values.size(); }

  std::complex<T>& at(int time_index, int freq_index) {
    return values[static_cast<std::size_t>(time_index) * static_cast<std::size_t>(freq_bins) + static_cast<std::size_t>(freq_index)];
  }

  const std::complex<T>& at(int time_index, int freq_index) const {
    return values[static_cast<std::size_t>(time_index) * static_cast<std::size_t>(freq_bins) + static_cast<std::size_t>(freq_index)];
  }
};

template <typename T>
struct PolarGrid {
  std::vector<T> magnitude;
  std::vector<T> phase;
  int time_bins = 0;
  int freq_bins = 0;
  GridSpec grid{};

  PolarGrid() = default;

  PolarGrid(int time_count, int freq_count, GridSpec spec = {})
      : magnitude(static_cast<std::size_t>(time_count) * static_cast<std::size_t>(freq_count)),
        phase(static_cast<std::size_t>(time_count) * static_cast<std::size_t>(freq_count)),
        time_bins(time_count),
        freq_bins(freq_count),
        grid(spec) {}

  bool empty() const { return magnitude.empty(); }

  std::size_t size() const { return magnitude.size(); }

  T& mag_at(int time_index, int freq_index) {
    return magnitude[static_cast<std::size_t>(time_index) * static_cast<std::size_t>(freq_bins) + static_cast<std::size_t>(freq_index)];
  }

  const T& mag_at(int time_index, int freq_index) const {
    return magnitude[static_cast<std::size_t>(time_index) * static_cast<std::size_t>(freq_bins) + static_cast<std::size_t>(freq_index)];
  }

  T& phase_at(int time_index, int freq_index) {
    return phase[static_cast<std::size_t>(time_index) * static_cast<std::size_t>(freq_bins) + static_cast<std::size_t>(freq_index)];
  }

  const T& phase_at(int time_index, int freq_index) const {
    return phase[static_cast<std::size_t>(time_index) * static_cast<std::size_t>(freq_bins) + static_cast<std::size_t>(freq_index)];
  }
};

struct ResamplePlan {
  int upsample_time = 1;
  int upsample_freq = 1;
  int downsample_time = 1;
  int downsample_freq = 1;
  bool preserve_energy = true;
};

template <typename T>
std::size_t linear_index(int time_index, int freq_index, int freq_bins) {
  return static_cast<std::size_t>(time_index) * static_cast<std::size_t>(freq_bins) + static_cast<std::size_t>(freq_index);
}

template <typename T>
ComplexGrid<T> polar_to_complex(const PolarGrid<T>& src) {
  ComplexGrid<T> dst(src.time_bins, src.freq_bins, src.grid);
  const std::size_t total = src.size();
  for (std::size_t i = 0; i < total; ++i) {
    dst.values[i] = std::polar(src.magnitude[i], src.phase[i]);
  }
  return dst;
}

template <typename T>
PolarGrid<T> complex_to_polar(const ComplexGrid<T>& src) {
  PolarGrid<T> dst(src.time_bins, src.freq_bins, src.grid);
  const std::size_t total = src.size();
  for (std::size_t i = 0; i < total; ++i) {
    dst.magnitude[i] = std::abs(src.values[i]);
    dst.phase[i] = std::arg(src.values[i]);
  }
  return dst;
}

template <typename T>
ComplexGrid<T> upsample_complex(const ComplexGrid<T>& src, int time_factor, int freq_factor) {
  if (src.empty() || (time_factor <= 1 && freq_factor <= 1)) {
    return src;
  }

  const int time_factor_safe = std::max(time_factor, 1);
  const int freq_factor_safe = std::max(freq_factor, 1);

  ComplexGrid<T> dst(src.time_bins * time_factor_safe, src.freq_bins * freq_factor_safe, src.grid);
  dst.grid.time_step /= static_cast<double>(time_factor_safe);
  dst.grid.freq_step /= static_cast<double>(freq_factor_safe);

  for (int t = 0; t < dst.time_bins; ++t) {
    const double src_t = static_cast<double>(t) / static_cast<double>(time_factor_safe);
    const int t0 = static_cast<int>(std::floor(src_t));
    const int t1 = std::min(t0 + 1, src.time_bins - 1);
    const T t_weight = static_cast<T>(src_t - static_cast<double>(t0));

    for (int f = 0; f < dst.freq_bins; ++f) {
      const double src_f = static_cast<double>(f) / static_cast<double>(freq_factor_safe);
      const int f0 = static_cast<int>(std::floor(src_f));
      const int f1 = std::min(f0 + 1, src.freq_bins - 1);
      const T f_weight = static_cast<T>(src_f - static_cast<double>(f0));

      const std::complex<T>& c00 = src.at(t0, f0);
      const std::complex<T>& c10 = src.at(t1, f0);
      const std::complex<T>& c01 = src.at(t0, f1);
      const std::complex<T>& c11 = src.at(t1, f1);

      const std::complex<T> interp_t0 = lerp_complex<T>(c00, c10, t_weight);
      const std::complex<T> interp_t1 = lerp_complex<T>(c01, c11, t_weight);
      dst.at(t, f) = lerp_complex<T>(interp_t0, interp_t1, f_weight);
    }
  }

  return dst;
}

template <typename T>
ComplexGrid<T> downsample_complex(const ComplexGrid<T>& src, int time_factor, int freq_factor, bool preserve_energy) {
  if (src.empty() || (time_factor <= 1 && freq_factor <= 1)) {
    return src;
  }

  const int time_factor_safe = std::max(time_factor, 1);
  const int freq_factor_safe = std::max(freq_factor, 1);

  const int dst_time = (src.time_bins + time_factor_safe - 1) / time_factor_safe;
  const int dst_freq = (src.freq_bins + freq_factor_safe - 1) / freq_factor_safe;

  ComplexGrid<T> dst(dst_time, dst_freq, src.grid);
  dst.grid.time_step *= static_cast<double>(time_factor_safe);
  dst.grid.freq_step *= static_cast<double>(freq_factor_safe);

  for (int t = 0; t < dst.time_bins; ++t) {
    const int src_t_begin = t * time_factor_safe;
    const int src_t_end = std::min(src_t_begin + time_factor_safe, src.time_bins);

    for (int f = 0; f < dst.freq_bins; ++f) {
      const int src_f_begin = f * freq_factor_safe;
      const int src_f_end = std::min(src_f_begin + freq_factor_safe, src.freq_bins);

      std::complex<T> accum{static_cast<T>(0), static_cast<T>(0)};
      int count = 0;

      for (int st = src_t_begin; st < src_t_end; ++st) {
        for (int sf = src_f_begin; sf < src_f_end; ++sf) {
          accum += src.at(st, sf);
          ++count;
        }
      }

      if (preserve_energy && count > 0) {
        const T inv = static_cast<T>(1) / static_cast<T>(count);
        accum *= inv;
      }

      dst.at(t, f) = accum;
    }
  }

  return dst;
}

template <typename T>
PolarGrid<T> upsample_polar(const PolarGrid<T>& src, int time_factor, int freq_factor) {
  if (src.empty() || (time_factor <= 1 && freq_factor <= 1)) {
    return src;
  }
  const ComplexGrid<T> converted = polar_to_complex(src);
  return complex_to_polar(upsample_complex(converted, time_factor, freq_factor));
}

template <typename T>
PolarGrid<T> downsample_polar(const PolarGrid<T>& src, int time_factor, int freq_factor, bool preserve_energy) {
  if (src.empty() || (time_factor <= 1 && freq_factor <= 1)) {
    return src;
  }
  const ComplexGrid<T> converted = polar_to_complex(src);
  return complex_to_polar(downsample_complex(converted, time_factor, freq_factor, preserve_energy));
}

template <typename T>
ComplexGrid<T> upsample_field(const ComplexGrid<T>& src, int time_factor, int freq_factor) {
  return upsample_complex(src, time_factor, freq_factor);
}

template <typename T>
PolarGrid<T> upsample_field(const PolarGrid<T>& src, int time_factor, int freq_factor) {
  return upsample_polar(src, time_factor, freq_factor);
}

template <typename T>
ComplexGrid<T> downsample_field(const ComplexGrid<T>& src, int time_factor, int freq_factor, bool preserve_energy) {
  return downsample_complex(src, time_factor, freq_factor, preserve_energy);
}

template <typename T>
PolarGrid<T> downsample_field(const PolarGrid<T>& src, int time_factor, int freq_factor, bool preserve_energy) {
  return downsample_polar(src, time_factor, freq_factor, preserve_energy);
}

template <typename T>
class ExponentialHarmonicQuantizer {
public:
  T base_frequency = static_cast<T>(55.0);
  T magnitude_step = static_cast<T>(0.05);
  T mix = static_cast<T>(0.5);
  T distance_decay = static_cast<T>(8.0);

  void operator()(ComplexGrid<T>& field) const {
    if (field.empty() || mix <= static_cast<T>(0)) {
      return;
    }

    for (int t = 0; t < field.time_bins; ++t) {
      for (int f = 0; f < field.freq_bins; ++f) {
        auto& sample = field.at(t, f);
        apply_cell(sample, static_cast<T>(field.grid.freq_at(f)));
      }
    }
  }

  void operator()(PolarGrid<T>& field) const {
    if (field.empty() || mix <= static_cast<T>(0)) {
      return;
    }

    for (int t = 0; t < field.time_bins; ++t) {
      for (int f = 0; f < field.freq_bins; ++f) {
        const T freq = static_cast<T>(field.grid.freq_at(f));
        const std::size_t idx = linear_index<T>(t, f, field.freq_bins);
        field.magnitude[idx] = apply_magnitude(field.magnitude[idx], freq);
      }
    }
  }

private:
  void apply_cell(std::complex<T>& sample, T bin_frequency) const {
    const T original_mag = static_cast<T>(std::abs(sample));
    if (original_mag <= std::numeric_limits<T>::epsilon()) {
      return;
    }

    const T adjusted_mag = apply_magnitude(original_mag, bin_frequency);
    const T phase = static_cast<T>(std::arg(sample));
    sample = std::polar(clamp_to_non_negative(adjusted_mag), phase);
  }

  T apply_magnitude(T magnitude, T bin_frequency) const {
    const T weight = effective_weight(bin_frequency);
    if (weight <= static_cast<T>(0)) {
      return magnitude;
    }

    const T target = quantize_magnitude(magnitude);
    return lerp_value<T>(magnitude, target, weight);
  }

  T quantize_magnitude(T magnitude) const {
    if (magnitude_step <= static_cast<T>(0)) {
      return magnitude;
    }
    const T ratio = magnitude / magnitude_step;
    const T rounded = static_cast<T>(std::round(static_cast<double>(ratio)));
    return rounded * magnitude_step;
  }

  T effective_weight(T bin_frequency) const {
    if (mix <= static_cast<T>(0)) {
      return static_cast<T>(0);
    }
    if (base_frequency <= static_cast<T>(0)) {
      return clamp_value<T>(mix, static_cast<T>(0), static_cast<T>(1));
    }

    const T ratio = bin_frequency / base_frequency;
    if (!std::isfinite(static_cast<double>(ratio))) {
      return static_cast<T>(0);
    }

    const T nearest = static_cast<T>(std::round(static_cast<double>(ratio)));
    const T distance = std::abs(ratio - nearest);
    const T attraction = static_cast<T>(std::exp(-distance_decay * distance));
    const T weight = mix * attraction;
    return clamp_value<T>(weight, static_cast<T>(0), static_cast<T>(1));
  }

  T clamp_to_non_negative(T value) const {
    return std::max(value, static_cast<T>(0));
  }
};

template <typename T>
class SpectralLocalContrastBooster {
public:
  int kernel_time = 3;
  int kernel_freq = 3;
  T brightness = static_cast<T>(0);
  T contrast = static_cast<T>(0.5);
  T gamma = static_cast<T>(1);
  bool clamp_zero = true;

  void operator()(ComplexGrid<T>& field) const {
    if (field.empty()) {
      return;
    }

    std::vector<T> magnitudes(field.size());
    for (std::size_t i = 0; i < field.size(); ++i) {
      magnitudes[i] = static_cast<T>(std::abs(field.values[i]));
    }

    apply_to_magnitudes(magnitudes, field.time_bins, field.freq_bins);

    for (std::size_t i = 0; i < field.size(); ++i) {
      const T phase = static_cast<T>(std::arg(field.values[i]));
      field.values[i] = std::polar(magnitudes[i], phase);
    }
  }

  void operator()(PolarGrid<T>& field) const {
    if (field.empty()) {
      return;
    }
    apply_to_magnitudes(field.magnitude, field.time_bins, field.freq_bins);
  }

private:
  void apply_to_magnitudes(std::vector<T>& magnitudes, int time_bins, int freq_bins) const {
    std::vector<T> updated(magnitudes.size(), static_cast<T>(0));

    const bool global_time = kernel_time <= 0;
    const bool global_freq = kernel_freq <= 0;
    const int radius_time = global_time ? time_bins : std::max(kernel_time / 2, 0);
    const int radius_freq = global_freq ? freq_bins : std::max(kernel_freq / 2, 0);

    T global_mean = static_cast<T>(0);
    if (global_time && global_freq && !magnitudes.empty()) {
      for (const T value : magnitudes) {
        global_mean += value;
      }
      global_mean /= static_cast<T>(magnitudes.size());
    }

    const T gamma_safe = (gamma > static_cast<T>(0)) ? gamma : static_cast<T>(1);
    const bool use_gamma = std::abs(gamma_safe - static_cast<T>(1)) > static_cast<T>(1e-5);
    const T inv_gamma = use_gamma ? static_cast<T>(1) / gamma_safe : static_cast<T>(1);
    const T epsilon = static_cast<T>(1e-12);

    for (int t = 0; t < time_bins; ++t) {
      const int t_min = global_time ? 0 : std::max(0, t - radius_time);
      const int t_max = global_time ? time_bins - 1 : std::min(time_bins - 1, t + radius_time);

      for (int f = 0; f < freq_bins; ++f) {
        const int f_min = global_freq ? 0 : std::max(0, f - radius_freq);
        const int f_max = global_freq ? freq_bins - 1 : std::min(freq_bins - 1, f + radius_freq);

        T local_mean = static_cast<T>(0);
        int count = 0;

        if (global_time && global_freq) {
          local_mean = global_mean;
          count = time_bins * freq_bins;
        } else {
          for (int tt = t_min; tt <= t_max; ++tt) {
            for (int ff = f_min; ff <= f_max; ++ff) {
              local_mean += magnitudes[linear_index<T>(tt, ff, freq_bins)];
              ++count;
            }
          }
          if (count > 0) {
            local_mean /= static_cast<T>(count);
          }
        }

        const std::size_t idx = linear_index<T>(t, f, freq_bins);
        const T current = magnitudes[idx];
        const T centered = current - local_mean;
        const T boosted = local_mean + centered * (static_cast<T>(1) + contrast);
        T adjusted = boosted + brightness;
        if (clamp_zero) {
          adjusted = std::max(adjusted, static_cast<T>(0));
        }

        if (use_gamma) {
          const T base = std::max(adjusted, epsilon);
          adjusted = static_cast<T>(std::pow(static_cast<double>(base), static_cast<double>(inv_gamma)));
        }

        updated[idx] = adjusted;
      }
    }

    magnitudes.swap(updated);
  }
};

template <typename Field, typename... Units>
void run_pipeline(Field& field, const ResamplePlan& plan, Units&&... units) {
  if (plan.upsample_time > 1 || plan.upsample_freq > 1) {
    field = upsample_field(field, plan.upsample_time, plan.upsample_freq);
  }

  (units(field), ...);

  if (plan.downsample_time > 1 || plan.downsample_freq > 1) {
    field = downsample_field(field, plan.downsample_time, plan.downsample_freq, plan.preserve_energy);
  }
}

using ComplexGridFloat = ComplexGrid<float>;
using ComplexGridDouble = ComplexGrid<double>;
using PolarGridFloat = PolarGrid<float>;
using PolarGridDouble = PolarGrid<double>;

}  // namespace fftfree::ft
