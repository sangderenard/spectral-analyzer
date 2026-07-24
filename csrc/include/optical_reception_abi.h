#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace optical_reception {

inline constexpr std::uint32_t kAbiVersion = 1u;

enum TokenFlags : std::uint32_t {
    TokenActive = 1u << 0,
    TokenContribution = 1u << 1,
    TokenCompletion = 1u << 2,
    TokenCoherent = 1u << 3,
    TokenContinuousFrequency = 1u << 4,
};

/* Fixed-stride process token for coherent reception.
 *
 * reception_key_handle resolves the complete physical OpticalReceptionKey in
 * pipeline-owned generation-checked storage. The token repeats only fields
 * needed for routing, duplicate contribution identity, and cheap stale-work
 * rejection. Handles are process-local and are never a network wire format.
 */
struct alignas(16) OpticalReceptionToken {
    std::uint64_t reception_key_handle = 0u;
    std::uint64_t state_handle = 0u;
    std::uint64_t camera_causality_id = 0u;
    std::uint64_t coherence_id = 0u;
    std::uint64_t contribution_id = 0u;
    std::uint64_t arrival_epoch = 0u;
    std::uint64_t solve_epoch = 0u;

    std::uint32_t pool_id = 0u;
    std::uint32_t predecessor_product_id = 0u;
    std::uint32_t reception_key_generation = 0u;
    std::uint32_t state_generation = 0u;
    std::uint32_t camera_program_generation = 0u;
    std::uint32_t mode_id = 0u;
    std::uint32_t flags = TokenActive;
    std::uint32_t reserved = 0u;
};

static_assert(std::is_standard_layout_v<OpticalReceptionToken>);
static_assert(std::is_trivially_copyable_v<OpticalReceptionToken>);
static_assert(alignof(OpticalReceptionToken) == 16u);
static_assert(sizeof(OpticalReceptionToken) == 96u);
static_assert(offsetof(OpticalReceptionToken, pool_id) == 56u);
static_assert(offsetof(OpticalReceptionToken, flags) == 80u);

} // namespace optical_reception
