#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace optical_branch {

inline constexpr std::uint32_t kAbiVersion = 1u;

enum TokenFlags : std::uint32_t {
    TokenActive = 1u << 0,
    TokenCoherent = 1u << 1,
    TokenContinuousFrequency = 1u << 2,
    TokenStochasticSelection = 1u << 3,
    TokenDeterministicProduct = 1u << 4,
    TokenOpticalPathValid = 1u << 5,
    TokenGroupDelayValid = 1u << 6,
};

/* Fixed-stride scheduling sidecar for a branch frontier.
 *
 * The token contains handles and correlation/timing metadata only. Exact
 * N-lane complex fields remain in pipeline-owned contiguous state blocks and
 * are never copied into a FIFO token. The high/low pairs preserve relative
 * carrier phase and arrival ordering over long paths.
 */
struct alignas(16) OpticalBranchToken {
    std::uint64_t state_handle = 0u;
    std::uint64_t lineage_id = 0u;
    std::uint64_t coherence_id = 0u;

    std::uint32_t node_id = 0u;
    std::uint32_t product_id = 0u;
    std::uint32_t sample_id = 0u;
    std::uint32_t source_lane = 0u;

    double arrival_time_hi_s = 0.0;
    double arrival_time_lo_s = 0.0;
    double optical_path_hi_m = 0.0;
    double optical_path_lo_m = 0.0;
    double frequency_hz = 0.0;

    float selection_pdf = 1.0f;
    float residual_power_bound = 1.0f;
    std::uint32_t flags = TokenActive;
    std::uint32_t state_generation = 0u;
};

static_assert(std::is_standard_layout_v<OpticalBranchToken>);
static_assert(std::is_trivially_copyable_v<OpticalBranchToken>);
static_assert(alignof(OpticalBranchToken) == 16u);
static_assert(sizeof(OpticalBranchToken) == 96u);

} // namespace optical_branch
