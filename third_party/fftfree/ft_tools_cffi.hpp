#pragma once

#include "fft_cffi.hpp"

#ifdef __cplusplus
extern "C" {
#endif

struct ft_grid_spec {
  double time_origin;
  double freq_origin;
  double time_step;
  double freq_step;
};

struct ft_resample_plan_c {
  int upsample_time;
  int upsample_freq;
  int downsample_time;
  int downsample_freq;
  int preserve_energy;
};

struct ft_quantizer_config {
  int enabled;
  float base_frequency;
  float magnitude_step;
  float mix;
  float distance_decay;
};

struct ft_contrast_config {
  int enabled;
  int kernel_time;
  int kernel_freq;
  float brightness;
  float contrast;
  float gamma;
  int clamp_zero;
};

FFTFREE_API int ft_apply_complex_pipeline(const ft_grid_spec* spec,
                                          const ft_resample_plan_c* plan,
                                          const ft_quantizer_config* quant,
                                          const ft_contrast_config* contrast,
                                          float* real,
                                          float* imag,
                                          int time_bins,
                                          int freq_bins);

FFTFREE_API int ft_apply_polar_pipeline(const ft_grid_spec* spec,
                                        const ft_resample_plan_c* plan,
                                        const ft_quantizer_config* quant,
                                        const ft_contrast_config* contrast,
                                        float* magnitude,
                                        float* phase,
                                        int time_bins,
                                        int freq_bins);

#ifdef __cplusplus
}
#endif
