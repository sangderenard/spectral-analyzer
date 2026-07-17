#ifndef SPECTRAL_SENSOR_MIPMAP_H
#define SPECTRAL_SENSOR_MIPMAP_H

#include <cstdint>

/* Shared std430-compatible ABI for recursive sensor refinement.  These records
 * contain topology and work identity only. Spectral moment arrays remain SoA
 * buffers so the active band count does not alter record stride. */

enum SensorMipNodeFlags : uint32_t {
    SENSOR_MIP_NODE_FRONTIER    = 1u << 0,
    SENSOR_MIP_NODE_QUEUED      = 1u << 1,
    SENSOR_MIP_NODE_RUNNING     = 1u << 2,
    SENSOR_MIP_NODE_DIRECT_VALID= 1u << 3,
    SENSOR_MIP_NODE_ROLL_VALID  = 1u << 4,
    SENSOR_MIP_NODE_DIRTY       = 1u << 5,
    SENSOR_MIP_NODE_TERMINAL    = 1u << 6,
    SENSOR_MIP_NODE_COVERAGE_ANCHOR = 1u << 7,
};

static constexpr uint32_t SENSOR_MIP_NO_NODE = 0xFFFFFFFFu;
static constexpr uint32_t SENSOR_MIP_BRANCHING = 9u;
static constexpr uint32_t SENSOR_PRIORITY_NETWORK_HIDDEN = 8u;
static constexpr uint32_t SENSOR_PRIORITY_NETWORK_PARAMS = 305u;

/* Adaptive camera trace tag, shared conceptually with the GLSL terminal splat.
 * hi[31]=film tag, hi[30]=adaptive lineage, hi[24..27]=level,
 * hi[0..23]=node id; lo[0..31]=sample output index. */
static constexpr uint32_t SENSOR_MIP_TAG_FILM_BIT = 0x80000000u;
static constexpr uint32_t SENSOR_MIP_TAG_ADAPTIVE_BIT = 0x40000000u;
static constexpr uint32_t SENSOR_MIP_TAG_NODE_MASK = 0x00FFFFFFu;
static constexpr uint32_t SENSOR_MIP_TAG_LEVEL_MASK = 0x0F000000u;

constexpr uint64_t sensor_mip_pack_trace_tag(
    uint32_t node_id, uint32_t level, uint32_t output_index)
{
    const uint32_t hi = SENSOR_MIP_TAG_FILM_BIT | SENSOR_MIP_TAG_ADAPTIVE_BIT
        | ((level & 0xFu) << 24u) | (node_id & SENSOR_MIP_TAG_NODE_MASK);
    return (static_cast<uint64_t>(hi) << 32u) | output_index;
}
static_assert((sensor_mip_pack_trace_tag(0x123456u, 7u, 0x89ABCDEFu) >> 32u)
                  == 0xC7123456u,
              "adaptive sensor tag high word ABI changed");
static_assert(static_cast<uint32_t>(
                  sensor_mip_pack_trace_tag(1u, 1u, 0x89ABCDEFu)) == 0x89ABCDEFu,
              "adaptive sensor tag sample index ABI changed");

struct alignas(16) SensorMipNodeGpu {
    float uv_bounds[4];          /* global sensor u0,v0,u1,v1 */
    uint32_t parent_id;
    uint32_t first_child_id;     /* nine contiguous children or NO_NODE */
    uint32_t level;
    uint32_t flags;
    uint32_t direct_moment_offset;
    uint32_t rollup_moment_offset;
    uint32_t direct_sample_count;
    uint32_t completed_epochs;
    float priority;
    uint32_t child_slot;         /* 0..8 inside parent; root uses NO_NODE */
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(SensorMipNodeGpu) == 64, "SensorMipNodeGpu ABI must remain 64 bytes");

struct alignas(16) SensorMipWorkGpu {
    uint32_t node_id;
    uint32_t sample_begin;       /* persistent sequence index, never reset on revisit */
    uint32_t sample_count;
    uint32_t output_offset;      /* first lineage/ray slot owned by this node */
    uint32_t seed;
    float priority;
    uint32_t flags;
    uint32_t _pad0;
};
static_assert(sizeof(SensorMipWorkGpu) == 32, "SensorMipWorkGpu ABI must remain 32 bytes");

struct alignas(16) SensorMipSampleLineageGpu {
    uint32_t node_id;
    uint32_t level;
    float global_u;
    float global_v;
    float local_u;
    float local_v;
    float estimator_weight;
    uint32_t sample_index;
};
static_assert(sizeof(SensorMipSampleLineageGpu) == 32,
              "SensorMipSampleLineageGpu ABI must remain 32 bytes");

/* Replaceable scorer input. A future convolutional network writes the learned
 * channel in this buffer; the scheduler only consumes the resulting priority. */
struct alignas(16) SensorMipPriorityFeaturesGpu {
    float uncertainty;
    float ambiguity;
    float learned;
    float requested;
};
static_assert(sizeof(SensorMipPriorityFeaturesGpu) == 16,
              "SensorMipPriorityFeaturesGpu ABI must remain 16 bytes");

#endif
