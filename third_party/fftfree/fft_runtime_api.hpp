#pragma once

#include "eigen_fft.hpp"
#include "plan_support.hpp"

#include <Eigen/Core>
#include <optional>
#include <complex>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace eigfft {

enum class TransformPlacement {
  InPlace,
  EmulatedInPlace,
  OutOfPlace
};

template <typename Scalar>
struct TransformSettings {
  int fft_size = 0;
  bool inverse = false;
  PlanRuntimeConfig runtime{};
  std::optional<KernelKind> kernel_hint;
  bool pad_to_power_of_two = true;
  TransformPlacement placement = TransformPlacement::InPlace;
};

namespace detail {
inline bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

inline std::size_t next_power_of_two(std::size_t value) {
  if (value <= 1) return 1;
  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
#if INTPTR_MAX == INT64_MAX
  value |= value >> 32;
#endif
  value++;
  return value;
}

}  // namespace detail

template <typename Scalar>
class PreparedTransform {
 public:
  using Complex = std::complex<Scalar>;
  using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  explicit PreparedTransform(const TransformSettings<Scalar>& settings)
      : settings_(settings) {
    if (settings_.fft_size <= 0) {
      throw std::invalid_argument("TransformSettings::fft_size must be positive");
    }

    const std::size_t raw_size = static_cast<std::size_t>(settings_.fft_size);
    std::size_t planned = raw_size;
    if (settings_.pad_to_power_of_two && !detail::is_power_of_two(planned)) {
      planned = detail::next_power_of_two(planned);
    }
    if (!detail::is_power_of_two(planned)) {
      throw std::invalid_argument("PreparedTransform requires a power-of-two length");
    }
    if (planned > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("FFT size exceeds supported range");
    }
    plan_size_ = static_cast<int>(planned);

    runtime_ = settings_.runtime;
    if (runtime_.threads <= 0) {
      runtime_.threads = Plan<Scalar>::Limits::kDefaultRuntimeThreads;
    }
    runtime_.threads = std::clamp(runtime_.threads, 1, Plan<Scalar>::Limits::kCompileTimeMaxThreads);

    if (runtime_.lanes <= 0) {
      runtime_.lanes = Plan<Scalar>::Limits::kDefaultLaneCapacity;
    }
    runtime_.lanes = std::clamp(runtime_.lanes, 1, Plan<Scalar>::Limits::compile_time_max_lane_capacity());
    runtime_.inverse = settings_.inverse;

    {
      auto token = cache_.get_plan(plan_size_, settings_.inverse, runtime_);
      (void)token;
    }
  }

  int plan_size() const { return plan_size_; }
  const TransformSettings<Scalar>& settings() const { return settings_; }

  std::optional<MatrixXc> run(Eigen::Ref<MatrixXc> buffer) {
    if (buffer.rows() != plan_size_) {
      throw std::invalid_argument("Input buffer rows must match PreparedTransform::plan_size()");
    }

    auto token = cache_.get_plan(plan_size_, settings_.inverse, runtime_);
    auto& plan = token.plan();

    if (settings_.kernel_hint.has_value()) {
      const bool ok = plan.use_kernel(*settings_.kernel_hint);
      if (!ok) {
        throw std::runtime_error("Requested kernel not available for this plan");
      }
    }

    switch (settings_.placement) {
      case TransformPlacement::InPlace: {
        fft_inplace_batched<Scalar>(buffer, plan);
        return std::nullopt;
      }
      case TransformPlacement::EmulatedInPlace: {
        MatrixXc scratch(plan_size_, buffer.cols());
        scratch = buffer;
        fft_inplace_batched<Scalar>(scratch, plan);
        buffer = scratch;
        return std::nullopt;
      }
      case TransformPlacement::OutOfPlace: {
        MatrixXc scratch(plan_size_, buffer.cols());
        scratch = buffer;
        fft_inplace_batched<Scalar>(scratch, plan);
        return scratch;
      }
      default:
        throw std::logic_error("Unknown TransformPlacement mode");
    }
  }

 private:
  TransformSettings<Scalar> settings_;
  PlanRuntimeConfig runtime_{};
  PlanCache<Scalar> cache_{};
  int plan_size_ = 0;
};

}  // namespace eigfft

