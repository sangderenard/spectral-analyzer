#include <intrin.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdint>

namespace {

bool os_has_avx_support() {
  int info[4] = {0, 0, 0, 0};
  __cpuid(info, 1);
  const bool has_osxsave = (info[2] & (1 << 27)) != 0;
  const bool has_avx = (info[2] & (1 << 28)) != 0;
  if (!has_osxsave || !has_avx) {
    return false;
  }
#ifdef _XCR_XFEATURE_ENABLED_MASK
  const unsigned long long xcr0 = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
#else
  const unsigned long long xcr0 = _xgetbv(0);
#endif
  return (xcr0 & 0x6) == 0x6;
}

bool cpu_has_avx2() {
  int info[4] = {0, 0, 0, 0};
  __cpuid(info, 0);
  const int max_leaf = info[0];
  if (max_leaf < 7) {
    return false;
  }
  __cpuidex(info, 7, 0);
  return (info[1] & (1 << 5)) != 0;
}

bool cpu_has_avx() {
  int info[4] = {0, 0, 0, 0};
  __cpuid(info, 1);
  return (info[2] & (1 << 28)) != 0;
}

}  // namespace

int main() {
  const bool avx_supported = os_has_avx_support();
  const bool avx2_supported = avx_supported && cpu_has_avx2();
  if (avx2_supported) {
    std::puts("AVX2");
    return 0;
  }
  if (avx_supported && cpu_has_avx()) {
    std::puts("AVX");
    return 0;
  }
  std::puts("BASELINE");
  return 0;
}
