#include "ft_tools_cffi.hpp"

#include "ft_tools.hpp"

#include <complex>
#include <limits>
#include <vector>

namespace {

using fftfree::ft::ComplexGrid;
using fftfree::ft::ExponentialHarmonicQuantizer;
using fftfree::ft::PolarGrid;
using fftfree::ft::ResamplePlan;
using fftfree::ft::SpectralLocalContrastBooster;
using fftfree::ft::downsample_field;
using fftfree::ft::upsample_field;

inline fftfree::ft::GridSpec to_grid_spec(const ft_grid_spec* spec) {
  fftfree::ft::GridSpec out;
  if (spec) {
    out.time_origin = spec->time_origin;
    out.freq_origin = spec->freq_origin;
    out.time_step = (spec->time_step != 0.0) ? spec->time_step : 1.0;
    out.freq_step = (spec->freq_step != 0.0) ? spec->freq_step : 1.0;
  }
  return out;
}

inline ResamplePlan to_plan(const ft_resample_plan_c* plan) {
  ResamplePlan out;
  if (plan) {
    out.upsample_time = (plan->upsample_time > 0) ? plan->upsample_time : 1;
    out.upsample_freq = (plan->upsample_freq > 0) ? plan->upsample_freq : 1;
    out.downsample_time = (plan->downsample_time > 0) ? plan->downsample_time : 1;
    out.downsample_freq = (plan->downsample_freq > 0) ? plan->downsample_freq : 1;
    out.preserve_energy = (plan->preserve_energy != 0);
  }
  return out;
}

inline bool configure_quantizer(const ft_quantizer_config* cfg, ExponentialHarmonicQuantizer<float>& out) {
  if (!cfg || cfg->enabled == 0) {
    return false;
  }
  out.base_frequency = cfg->base_frequency;
  out.magnitude_step = cfg->magnitude_step;
  out.mix = cfg->mix;
  out.distance_decay = cfg->distance_decay;
  return true;
}

inline bool configure_contrast(const ft_contrast_config* cfg, SpectralLocalContrastBooster<float>& out) {
  if (!cfg || cfg->enabled == 0) {
    return false;
  }
  out.kernel_time = cfg->kernel_time;
  out.kernel_freq = cfg->kernel_freq;
  out.brightness = cfg->brightness;
  out.contrast = cfg->contrast;
  out.gamma = cfg->gamma;
  out.clamp_zero = (cfg->clamp_zero != 0);
  return true;
}

inline bool validate_shape(int time_bins, int freq_bins) {
  return time_bins > 0 && freq_bins > 0 &&
         static_cast<long long>(time_bins) * static_cast<long long>(freq_bins) <
             static_cast<long long>(std::numeric_limits<int>::max());
}

}  // namespace

extern "C" {

FFTFREE_API int ft_apply_complex_pipeline(const ft_grid_spec* spec,
                                          const ft_resample_plan_c* plan,
                                          const ft_quantizer_config* quant,
                                          const ft_contrast_config* contrast,
                                          float* real,
                                          float* imag,
                                          int time_bins,
                                          int freq_bins) {
  if (!real || !imag || !validate_shape(time_bins, freq_bins)) {
    return 0;
  }
  const std::size_t total = static_cast<std::size_t>(time_bins) * static_cast<std::size_t>(freq_bins);
  ComplexGrid<float> field(time_bins, freq_bins, to_grid_spec(spec));
  for (std::size_t idx = 0; idx < total; ++idx) {
    field.values[idx] = std::complex<float>(real[idx], imag[idx]);
  }

  const ResamplePlan plan_cpp = to_plan(plan);
  if (plan_cpp.upsample_time > 1 || plan_cpp.upsample_freq > 1) {
    field = upsample_field(field, plan_cpp.upsample_time, plan_cpp.upsample_freq);
  }

  ExponentialHarmonicQuantizer<float> quantizer;
  SpectralLocalContrastBooster<float> contrast_unit;
  const bool use_quant = configure_quantizer(quant, quantizer);
  const bool use_contrast = configure_contrast(contrast, contrast_unit);

  if (use_quant) {
    quantizer(field);
  }
  if (use_contrast) {
    contrast_unit(field);
  }

  if (plan_cpp.downsample_time > 1 || plan_cpp.downsample_freq > 1) {
    field = downsample_field(field, plan_cpp.downsample_time, plan_cpp.downsample_freq, plan_cpp.preserve_energy);
  }

  if (field.time_bins != time_bins || field.freq_bins != freq_bins) {
    return 0;
  }

  for (std::size_t idx = 0; idx < total; ++idx) {
    const std::complex<float>& v = field.values[idx];
    real[idx] = v.real();
    imag[idx] = v.imag();
  }
  return 1;
}

FFTFREE_API int ft_apply_polar_pipeline(const ft_grid_spec* spec,
                                        const ft_resample_plan_c* plan,
                                        const ft_quantizer_config* quant,
                                        const ft_contrast_config* contrast,
                                        float* magnitude,
                                        float* phase,
                                        int time_bins,
                                        int freq_bins) {
  if (!magnitude || !phase || !validate_shape(time_bins, freq_bins)) {
    return 0;
  }
  const std::size_t total = static_cast<std::size_t>(time_bins) * static_cast<std::size_t>(freq_bins);
  PolarGrid<float> field(time_bins, freq_bins, to_grid_spec(spec));
  for (std::size_t idx = 0; idx < total; ++idx) {
    field.magnitude[idx] = magnitude[idx];
    field.phase[idx] = phase[idx];
  }

  const ResamplePlan plan_cpp = to_plan(plan);
  if (plan_cpp.upsample_time > 1 || plan_cpp.upsample_freq > 1) {
    field = upsample_field(field, plan_cpp.upsample_time, plan_cpp.upsample_freq);
  }

  ExponentialHarmonicQuantizer<float> quantizer;
  SpectralLocalContrastBooster<float> contrast_unit;
  const bool use_quant = configure_quantizer(quant, quantizer);
  const bool use_contrast = configure_contrast(contrast, contrast_unit);

  if (use_quant) {
    quantizer(field);
  }
  if (use_contrast) {
    contrast_unit(field);
  }

  if (plan_cpp.downsample_time > 1 || plan_cpp.downsample_freq > 1) {
    field = downsample_field(field, plan_cpp.downsample_time, plan_cpp.downsample_freq, plan_cpp.preserve_energy);
  }

  if (field.time_bins != time_bins || field.freq_bins != freq_bins) {
    return 0;
  }

  for (std::size_t idx = 0; idx < total; ++idx) {
    magnitude[idx] = field.magnitude[idx];
    phase[idx] = field.phase[idx];
  }
  return 1;
}

}  // extern "C"
