// butterfly_api.hpp
// Small, focused API surface for butterfly primitives used by the FFT core.
// Heavy implementations live in separate implementation headers (e.g. radix-2)
// to keep `eigen_fft.hpp` compact. This file is intentionally tiny and
// defines the minimal contract callers rely on.
#pragma once

#include <complex>
#include <functional>
#include <vector>
#include "butterfly_kernel.hpp"
#include <string_view>
#include <unordered_map>
#include <mutex>

namespace eigfft {
namespace detail {

// Reuse the kernel-level enums/config shape so the API matches the concrete
// implementation in `butterfly_kernel.hpp`. This avoids a separate 'algo'
// abstraction and surfaces radix/method/flow/sort directly.
enum class TwiddlePlacement { PreRHS, PostRHS, PreLHS, PostLHS };

template <class T>
// ButterflyConfig describes algorithm and diagnostic hooks only; all buffer allocations
// must be handled by the Plan at planning time. Butterfly kernels must never allocate
// their own buffers. All scratch/emulation needs must be communicated up front so the
// Plan can allocate and supply them at execution.
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


// Primary templates' concrete definitions live in implementation headers or
// compatibility headers. Do not forward-declare the concrete `ButterflyKernel`
// or `ButterflyScatter` here to avoid ODR/aliasing conflicts with compatibility
// headers that may introduce `eigfft::ButterflyKernel` and aliases into
// `eigfft::detail`.

// Additionally allow future typed/tagged specializations if desired (declared
// only where needed by implementations).
template <class T, class AlgoTag>
struct ButterflyKernelTyped;

template <class T, class AlgoTag>
struct ButterflyScatterTyped;

// Lightweight runtime registry skeleton for future dynamic/provider use.
// For the low-effort path we provide an empty-but-usable registry API so
// runtime selection can be added later without changing call sites.
template <typename T>
struct ButterflyProviderCaps {
  bool has_scatter = false;
  ButterflyMethod preferred_method = ButterflyMethod::DIT;
  int simd_width = 1;
};

template <class T>
class ButterflyRegistry {
 public:
  using ApplyFn = void(*)(std::complex<T>*, std::complex<T>*, int, std::complex<T>, int);
  struct ProviderInfo {
    ApplyFn fn;
    ButterflyProviderCaps<T> caps;
  };

  static void register_provider(std::string_view name, ApplyFn fn, const ButterflyProviderCaps<T>& caps) {
    std::lock_guard<std::mutex> g(mu());
    map()[std::string(name)] = {fn, caps};
  }

  static ProviderInfo get(std::string_view name) {
    std::lock_guard<std::mutex> g(mu());
    auto it = map().find(std::string(name));
    if (it == map().end()) return {nullptr, {}};
    return it->second;
  }

  // Auto-negotiation: select best provider by capability
  static ProviderInfo negotiate(bool require_scatter, int min_simd_width, ButterflyMethod method_hint) {
    std::lock_guard<std::mutex> g(mu());
    ProviderInfo best = {nullptr, {}};
    for (const auto& kv : map()) {
      const auto& info = kv.second;
      if (require_scatter && !info.caps.has_scatter) continue;
      if (info.caps.simd_width < min_simd_width) continue;
      if (info.caps.preferred_method != method_hint) continue;
      // Prefer highest SIMD width
      if (!best.fn || info.caps.simd_width > best.caps.simd_width) best = info;
    }
    return best;
  }

 private:
  static std::unordered_map<std::string, ProviderInfo>& map() {
    static std::unordered_map<std::string, ProviderInfo> m;
    return m;
  }
  static std::mutex& mu() {
    static std::mutex m;
    return m;
  }
};

} // namespace detail
} // namespace eigfft
