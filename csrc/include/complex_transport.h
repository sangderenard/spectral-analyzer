/* complex_transport.h -- versioned CPU/GPU complex-light sidecar ABI.
 *
 * Ordinary RayIntent storage stays unchanged.  A ray crossing a complex-field
 * port receives one RaySidecar; compatible sidecars are subsequently compacted
 * into an exact-width LaneBlock<B>.  The N-lane block is therefore paid for by
 * T4 work, never by every geometric ray in T1/T2/T3.
 *
 * PackedComplexLaneGpu is a binding-independent std430 record.  A stage may put
 * an array of these records in a tail region of an SSBO it already binds, then
 * hand that region to T4 after the producing stage releases its bindings.
 *
 * Maxwell patches do not add a volumetric solver payload to this ABI. Their
 * cold compiler produces a cache-addressed polarized scattering artifact.
 * A later schema version may add compact medium/port/artifact identity, while
 * the existing s/p amplitudes remain the field boundary representation. See
 * MAXWELL_PATCH_CONTEXT.md.
 */
#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "complex_optical_operators.h"

namespace complex_transport {

constexpr std::uint32_t kSchemaVersion = 2u;
constexpr std::uint32_t kInvalidOperatorIndex = 0xffffffffu;
constexpr std::array<int, 6> kLaneCounts = {1, 3, 4, 8, 16, 32};

enum LaneFlags : std::uint16_t {
    LaneActive       = 1u << 0,
    ContinuousSample= 1u << 1,
    Coherent         = 1u << 2,
    JonesValid       = 1u << 3,
    OpticalPathValid = 1u << 4,
    JacobianValid    = 1u << 5,
};

/* Scheduling sidecar for one ray/sample.  Amplitude remains in RayIntent so
 * CPU routing does not duplicate its dynamic spectral vector. */
struct alignas(16) RaySidecar {
    double        frequency_hz       = 0.0;  /* invariant spectral coordinate */
    double        optical_path_m     = 0.0;  /* valid only with OpticalPathValid */
    double        spectral_pdf       = 1.0;
    double        transport_jacobian = 1.0;
    std::uint64_t coherence_id       = 0u;
    std::uint32_t sample_id          = 0u;
    std::uint16_t source_lane        = 0u;
    std::uint16_t flags              = 0u;
};

/* Complete CPU lane used at the ray/field boundary. Production ordinary-wave
 * work uses both amplitudes in a declared local s/p basis. Scalar calibration
 * specializations use amplitude_s only and leave JonesValid clear. */
struct alignas(16) ComplexLane {
    RaySidecar          meta{};
    std::complex<double> amplitude_s{0.0, 0.0};
    std::complex<double> amplitude_p{0.0, 0.0};
};

/* 16 x 32-bit words, exactly four std430 vec4 slots (64 bytes).
 * Large phase-sensitive scalars use a float-float hi/lo representation so the
 * ABI does not require shader fp64 support. */
struct alignas(16) PackedComplexLaneGpu {
    float frequency_hi;
    float frequency_lo;
    float optical_path_hi;
    float optical_path_lo;

    float spectral_pdf;
    float transport_jacobian;
    float amplitude_s_re;
    float amplitude_s_im;

    float amplitude_p_re;
    float amplitude_p_im;
    std::uint32_t coherence_lo;
    std::uint32_t coherence_hi;

    std::uint32_t sample_id;
    std::uint32_t lane_flags; /* low 16 flags, high 16 source_lane */
    std::uint32_t basis_id;    /* persistent TransverseBasis table index */
    std::uint32_t operator_id; /* persistent Jones+4x4 operator table index */
};

template <int B>
struct IsSupportedLaneCount {
    static constexpr bool value =
        B == 1 || B == 3 || B == 4 || B == 8 || B == 16 || B == 32;
};

template <int B>
struct alignas(16) LaneBlock {
    static_assert(IsSupportedLaneCount<B>::value,
                  "complex transport lane count must be 1,3,4,8,16,or 32");
    std::array<ComplexLane, B> lanes{};
};

inline void split_double(double value, float& hi, float& lo) noexcept
{
    hi = static_cast<float>(value);
    lo = static_cast<float>(value - static_cast<double>(hi));
}

inline PackedComplexLaneGpu pack_gpu(
    const ComplexLane& lane,
    std::uint32_t basis_id = kInvalidOperatorIndex,
    std::uint32_t operator_id = kInvalidOperatorIndex) noexcept
{
    PackedComplexLaneGpu out{};
    split_double(lane.meta.frequency_hz, out.frequency_hi, out.frequency_lo);
    split_double(lane.meta.optical_path_m, out.optical_path_hi, out.optical_path_lo);
    out.spectral_pdf       = static_cast<float>(lane.meta.spectral_pdf);
    out.transport_jacobian = static_cast<float>(lane.meta.transport_jacobian);
    out.amplitude_s_re     = static_cast<float>(lane.amplitude_s.real());
    out.amplitude_s_im     = static_cast<float>(lane.amplitude_s.imag());
    out.amplitude_p_re     = static_cast<float>(lane.amplitude_p.real());
    out.amplitude_p_im     = static_cast<float>(lane.amplitude_p.imag());
    out.coherence_lo       = static_cast<std::uint32_t>(lane.meta.coherence_id);
    out.coherence_hi       = static_cast<std::uint32_t>(lane.meta.coherence_id >> 32u);
    out.sample_id          = lane.meta.sample_id;
    out.lane_flags         = static_cast<std::uint32_t>(lane.meta.flags)
                           | (static_cast<std::uint32_t>(lane.meta.source_lane) << 16u);
    out.basis_id           = basis_id;
    out.operator_id        = operator_id;
    return out;
}

static_assert(std::is_standard_layout<RaySidecar>::value, "sidecar must be POD-like");
static_assert(std::is_standard_layout<PackedComplexLaneGpu>::value, "GPU lane must be standard layout");
static_assert(sizeof(RaySidecar) == 48, "RaySidecar ABI changed");
static_assert(sizeof(ComplexLane) == 80, "ComplexLane ABI changed");
static_assert(sizeof(PackedComplexLaneGpu) == 64, "GPU lane must be four vec4 slots");
static_assert(sizeof(LaneBlock<1>)  == 1u  * sizeof(ComplexLane), "lane block padding changed");
static_assert(sizeof(LaneBlock<3>)  == 3u  * sizeof(ComplexLane), "lane block padding changed");
static_assert(sizeof(LaneBlock<4>)  == 4u  * sizeof(ComplexLane), "lane block padding changed");
static_assert(sizeof(LaneBlock<8>)  == 8u  * sizeof(ComplexLane), "lane block padding changed");
static_assert(sizeof(LaneBlock<16>) == 16u * sizeof(ComplexLane), "lane block padding changed");
static_assert(sizeof(LaneBlock<32>) == 32u * sizeof(ComplexLane), "lane block padding changed");

} // namespace complex_transport
