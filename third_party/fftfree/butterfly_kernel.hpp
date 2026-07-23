// butterfly_kernel.hpp
#pragma once

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>
#include <utility>
#include <Eigen/Core>

namespace eigfft {

// Small policy enums and config for future extensions.

// Modular FFT butterfly configuration enums
enum class ButterflyRadix {
  Radix2,
  Radix4,
  Radix8,
  Radix16
};

// ButterflyMethod is strictly local math: DIT or DIF
enum class ButterflyMethod {
  DIT,
  DIF
};

enum class ButterflyFlow {
  InPlace,
  OutOfPlace
};

enum class ButterflySort {
  None,
  BitReversal,
  Stride,
  Custom
};

enum class TwiddlePlacement {
  PreRHS,
  PostRHS,
  PreLHS,
  PostLHS
};


template <class T>
struct ButterflyConfig {
  ButterflyRadix radix = ButterflyRadix::Radix2;
  ButterflyMethod method = ButterflyMethod::DIT;
  ButterflyFlow flow = ButterflyFlow::InPlace;
  ButterflySort sort = ButterflySort::None;
  TwiddlePlacement tw_place = TwiddlePlacement::PreRHS;
  bool forward = true;
  bool conjugate_tw = false;
  int simd_width = 1;
  int bit_rate = 0; // bits per value, for quantized/precision-aware FFTs
  std::vector<float> metrics; // e.g. [accuracy, speed, memory, ...] for negotiation
  // Snapshot hook: stage,pos,y0,y1
  std::function<void(int,int,const std::complex<T>&,const std::complex<T>&)> snapshot;
  // Invariant hook: stage, r0, energy
  std::function<void(int,T,T)> invariant;
  // Twiddle index hook: stage,pos,tw_index
  std::function<void(int,int,int)> twiddle_index;
};

// Minimal, efficient vectorized radix-2 DIT/DIF implementations.
template <class T>
struct DefaultButterflyImpl {
  using Complex = std::complex<T>;
  using Traits = Eigen::internal::packet_traits<Complex>;
  using Packet = typename Traits::type;
  static constexpr int PacketSize = Traits::size;

  // Radix-2 DIT: t = w*b; y0 = a + t; y1 = a - t
  static inline void radix2_dit(Complex* a, Complex* b, std::int64_t width, const Complex& w) {
    std::int64_t lane = 0;
    if constexpr (PacketSize > 1) {
      const Packet w_packet = Eigen::internal::pset1<Packet>(w);
      for (; lane + PacketSize <= width; lane += PacketSize) {
        const Packet a_pack = Eigen::internal::ploadu<Packet>(a + lane);
        const Packet b_pack = Eigen::internal::ploadu<Packet>(b + lane);
        const Packet b_twiddled = Eigen::internal::pmul(b_pack, w_packet);
        const Packet sum = Eigen::internal::padd(a_pack, b_twiddled);
        const Packet diff = Eigen::internal::psub(a_pack, b_twiddled);
        Eigen::internal::pstoreu<Complex, Packet>(a + lane, sum);
        Eigen::internal::pstoreu<Complex, Packet>(b + lane, diff);
      }
    }
    for (; lane < width; ++lane) {
      const Complex ai = a[static_cast<std::size_t>(lane)];
      const Complex bt = b[static_cast<std::size_t>(lane)] * w;
      a[static_cast<std::size_t>(lane)] = ai + bt;
      b[static_cast<std::size_t>(lane)] = ai - bt;
    }
  }

  // Radix-2 DIF: u = a + b; v = a - b; y0 = u; y1 = w*v
  static inline void radix2_dif(Complex* a, Complex* b, std::int64_t width, const Complex& w) {
    std::int64_t lane = 0;
    if constexpr (PacketSize > 1) {
      const Packet w_packet = Eigen::internal::pset1<Packet>(w);
      for (; lane + PacketSize <= width; lane += PacketSize) {
        const Packet a_pack = Eigen::internal::ploadu<Packet>(a + lane);
        const Packet b_pack = Eigen::internal::ploadu<Packet>(b + lane);
        const Packet sum = Eigen::internal::padd(a_pack, b_pack);
        const Packet diff = Eigen::internal::psub(a_pack, b_pack);
        const Packet diff_tw = Eigen::internal::pmul(diff, w_packet);
        Eigen::internal::pstoreu<Complex, Packet>(a + lane, sum);
        Eigen::internal::pstoreu<Complex, Packet>(b + lane, diff_tw);
      }
    }
    for (; lane < width; ++lane) {
      const Complex ai = a[static_cast<std::size_t>(lane)];
      const Complex bi = b[static_cast<std::size_t>(lane)];
      const Complex u = ai + bi;
      const Complex v = ai - bi;
      a[static_cast<std::size_t>(lane)] = u;
      b[static_cast<std::size_t>(lane)] = v * w;
    }
  }
};

// Backwards-compatible API: ButterflyKernel and ButterflyScatter wrappers.
// They call the configured algorithm; by default, Radix-2 DIT is used.
template <class T>
struct ButterflyKernel {
  using Complex = std::complex<T>;
  static void apply(Complex* a, Complex* b, std::int64_t width, const Complex& w,
                    const ButterflyConfig<T>* cfg = nullptr) {
    // Respect conjugation flag if present in config; default no-conjugation.
    const Complex w_use = (cfg && cfg->conjugate_tw) ? std::conj(w) : w;
    if (!cfg) {
      DefaultButterflyImpl<T>::radix2_dit(a, b, width, w_use);
      return;
    }
    // Use method (DIT/DIF) to select local math. Radix/flow/sort can be used
    // by more advanced kernels; for this simple default implementation we
    // dispatch only on method.
    if (cfg->method == ButterflyMethod::DIF) {
      DefaultButterflyImpl<T>::radix2_dif(a, b, width, w_use);
    } else {
      DefaultButterflyImpl<T>::radix2_dit(a, b, width, w_use);
    }
  }
};

template <class T>
struct ButterflyScatter {
  using Complex = std::complex<T>;
  // Scatter (out-of-place) apply: providers should implement this path. By
  // default we use the efficient, packetized scatter implementations below.
  static void apply(const Complex* a, const Complex* b, Complex* out0, Complex* out1,
                    std::int64_t width, const Complex& w, const ButterflyConfig<T>* cfg = nullptr) {
    using Traits = Eigen::internal::packet_traits<Complex>;
    using Packet = typename Traits::type;
    constexpr int PacketSize = Traits::size;

    // Compute twiddle use once (cfg may request conjugation)
    const Complex w_use = (cfg && cfg->conjugate_tw) ? std::conj(w) : w;
    if (!cfg) {
      // DIT scatter
      std::int64_t lane = 0;
      if constexpr (PacketSize > 1) {
        const Packet w_packet = Eigen::internal::pset1<Packet>(w_use);
        for (; lane + PacketSize <= width; lane += PacketSize) {
          const Packet a_pack = Eigen::internal::ploadu<Packet>(a + lane);
          const Packet b_pack = Eigen::internal::ploadu<Packet>(b + lane);
          const Packet b_twiddled = Eigen::internal::pmul(b_pack, w_packet);
          const Packet sum = Eigen::internal::padd(a_pack, b_twiddled);
          const Packet diff = Eigen::internal::psub(a_pack, b_twiddled);
          Eigen::internal::pstoreu<Complex, Packet>(out0 + lane, sum);
          Eigen::internal::pstoreu<Complex, Packet>(out1 + lane, diff);
        }
      }
      for (; lane < width; ++lane) {
        const Complex ai = a[static_cast<std::size_t>(lane)];
        const Complex bt = b[static_cast<std::size_t>(lane)] * w_use;
        out0[static_cast<std::size_t>(lane)] = ai + bt;
        out1[static_cast<std::size_t>(lane)] = ai - bt;
      }
      return;
    }

    if (cfg->method == ButterflyMethod::DIF) {
      // DIF scatter
      std::int64_t lane = 0;
      if constexpr (PacketSize > 1) {
        const Packet w_packet = Eigen::internal::pset1<Packet>(w_use);
        for (; lane + PacketSize <= width; lane += PacketSize) {
          const Packet a_pack = Eigen::internal::ploadu<Packet>(a + lane);
          const Packet b_pack = Eigen::internal::ploadu<Packet>(b + lane);
          const Packet sum = Eigen::internal::padd(a_pack, b_pack);
          const Packet diff = Eigen::internal::psub(a_pack, b_pack);
          const Packet diff_tw = Eigen::internal::pmul(diff, w_packet);
          Eigen::internal::pstoreu<Complex, Packet>(out0 + lane, sum);
          Eigen::internal::pstoreu<Complex, Packet>(out1 + lane, diff_tw);
        }
      }
      for (; lane < width; ++lane) {
        const Complex ai = a[static_cast<std::size_t>(lane)];
        const Complex bi = b[static_cast<std::size_t>(lane)];
        const Complex u = ai + bi;
        const Complex v = ai - bi;
        out0[static_cast<std::size_t>(lane)] = u;
        out1[static_cast<std::size_t>(lane)] = v * w_use;
      }
    } else {
      // DIT scatter
      std::int64_t lane = 0;
      if constexpr (PacketSize > 1) {
        const Packet w_packet = Eigen::internal::pset1<Packet>(w_use);
        for (; lane + PacketSize <= width; lane += PacketSize) {
          const Packet a_pack = Eigen::internal::ploadu<Packet>(a + lane);
          const Packet b_pack = Eigen::internal::ploadu<Packet>(b + lane);
          const Packet b_twiddled = Eigen::internal::pmul(b_pack, w_packet);
          const Packet sum = Eigen::internal::padd(a_pack, b_twiddled);
          const Packet diff = Eigen::internal::psub(a_pack, b_twiddled);
          Eigen::internal::pstoreu<Complex, Packet>(out0 + lane, sum);
          Eigen::internal::pstoreu<Complex, Packet>(out1 + lane, diff);
        }
      }
      for (; lane < width; ++lane) {
        const Complex ai = a[static_cast<std::size_t>(lane)];
        const Complex bt = b[static_cast<std::size_t>(lane)] * w_use;
        out0[static_cast<std::size_t>(lane)] = ai + bt;
        out1[static_cast<std::size_t>(lane)] = ai - bt;
      }
    }
  }
};

} // namespace eigfft

// Backwards compatibility: expose aliases in the detail namespace as the rest of the
// code expects 'eigfft::detail::ButterflyKernel' and 'eigfft::detail::ButterflyScatter'.
namespace eigfft { namespace detail {
template <class T> using ButterflyKernel = ::eigfft::ButterflyKernel<T>;
template <class T> using ButterflyScatter = ::eigfft::ButterflyScatter<T>;
template <class T> using ButterflyConfig_t = ::eigfft::ButterflyConfig<T>;
}} // namespace eigfft::detail
