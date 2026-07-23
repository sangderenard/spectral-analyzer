/* Canonical persistent Jones/differential operator ABI.
 *
 * Ordinary rays do not carry these records. Packed complex lanes hold stable
 * basis/operator indices; an arena owns the contiguous tables below.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace complex_optical_operators {

constexpr std::uint32_t kSchemaVersion = 1u;

/* Two std430 vec4 slots. k is reconstructed as cross(s, p). */
struct alignas(16) PackedTransverseBasisGpu {
    std::array<float, 4> s{};
    std::array<float, 4> p{};
};

/* Six std430 vec4 slots.
 * jones_row*: (m0.re, m0.im, m1.re, m1.im)
 * phase_space: column-major canonical [q_s,q_p,n*k_s,n*k_p] tangent map.
 */
struct alignas(16) PackedOperatorGpu {
    std::array<float, 4> jones_row0{};
    std::array<float, 4> jones_row1{};
    std::array<float, 16> phase_space{};
};

static_assert(std::is_standard_layout<PackedTransverseBasisGpu>::value,
              "basis record must be standard layout");
static_assert(std::is_standard_layout<PackedOperatorGpu>::value,
              "operator record must be standard layout");
static_assert(sizeof(PackedTransverseBasisGpu) == 32,
              "basis record must be two vec4 slots");
static_assert(sizeof(PackedOperatorGpu) == 96,
              "operator record must be six vec4 slots");

} // namespace complex_optical_operators
