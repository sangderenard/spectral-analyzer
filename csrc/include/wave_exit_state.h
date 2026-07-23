/* wave_exit_state.h -- fixed-stride complex field-to-ray reduction records.
 *
 * These records follow representative rays emitted by T4 without widening
 * RayIntent or any GPU intent SSBO. One row describes one active spectral
 * lane. The record is side data: ray_tag + BDPT identity join it to the ray
 * lineage, while the complete Jones amplitude and explicit exit basis retain
 * the state that the legacy scalar continuation cannot carry.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "complex_transport.h"

enum WaveExitStateFlags : std::uint32_t {
    WAVE_EXIT_REPRESENTATIVE_RAY = 1u << 0,
    WAVE_EXIT_JONES_VALID        = 1u << 1,
    WAVE_EXIT_PHASE_ANCHOR_VALID = 1u << 2,
    WAVE_EXIT_CONTINUOUS_SAMPLE  = 1u << 3,
};

struct alignas(16) WaveExitStateRecord {
    std::uint64_t ray_tag = 0u;
    std::uint32_t subpath_id = 0u;
    std::uint16_t vertex_index = 0u;
    std::uint8_t stream = 0u;
    std::uint8_t direction = 0u;

    std::int32_t arena_id = -1;
    std::int32_t state_lane = -1;
    std::uint32_t band_id = 0u;
    std::uint32_t flags = 0u;

    std::array<float, 3> position_m{};
    float path_length_m = 0.0f;
    std::array<float, 3> ray_direction{};
    float phase_anchor_quality = 0.0f;

    /* Explicit standalone right-handed basis: s x p = ray_direction. */
    std::array<float, 4> basis_s{};
    std::array<float, 4> basis_p{};

    complex_transport::PackedComplexLaneGpu lane{};
};

static_assert(std::is_standard_layout<WaveExitStateRecord>::value,
              "wave exit state must be standard layout");
static_assert(sizeof(WaveExitStateRecord) == 160,
              "wave exit state stride changed");
static_assert(alignof(WaveExitStateRecord) == 16,
              "wave exit state must remain 16-byte aligned");
static_assert(offsetof(WaveExitStateRecord, position_m) == 32,
              "wave exit position offset changed");
static_assert(offsetof(WaveExitStateRecord, basis_s) == 64,
              "wave exit s-basis offset changed");
static_assert(offsetof(WaveExitStateRecord, basis_p) == 80,
              "wave exit p-basis offset changed");
static_assert(offsetof(WaveExitStateRecord, lane) == 96,
              "wave exit complex-lane offset changed");
