#version 430 core

layout(local_size_x = 64) in;

const uint NO_NODE = 0xFFFFFFFFu;
const uint NODE_ROLL_VALID = 1u << 4;
const uint NODE_DIRTY = 1u << 5;

struct SensorMipNode {
    vec4 uv_bounds;
    uint parent_id;
    uint first_child_id;
    uint level;
    uint flags;
    uint direct_moment_offset;
    uint rollup_moment_offset;
    uint direct_sample_count;
    uint completed_epochs;
    float priority;
    uint child_slot;
    uint pad0;
    uint pad1;
};

layout(std430, binding = 0) buffer NodeBuf { SensorMipNode nodes[]; };
layout(std430, binding = 1) readonly buffer ControlBuf { uint control[]; };
layout(std430, binding = 2) readonly buffer DirectSumBuf { uint direct_sum[]; };
layout(std430, binding = 3) readonly buffer DirectWeightBuf { uint direct_weight[]; };
layout(std430, binding = 4) buffer RollupMeanBuf { uint rollup_mean[]; };
layout(std430, binding = 5) buffer RollupEvidenceBuf { uint rollup_evidence[]; };

uniform uint target_level;
uniform uint n_bands;

float child_evidence(uint child_id) {
    return uintBitsToFloat(direct_weight[child_id])
         + uintBitsToFloat(rollup_evidence[child_id]);
}

float child_mean(uint child_id, uint band) {
    SensorMipNode child = nodes[child_id];
    float direct_evidence = uintBitsToFloat(direct_weight[child_id]);
    float descendant_evidence = uintBitsToFloat(rollup_evidence[child_id]);
    float total = direct_evidence + descendant_evidence;
    if (total <= 0.0) return 0.0;
    float direct_mean = direct_evidence > 0.0
        ? uintBitsToFloat(direct_sum[child.direct_moment_offset + band]) / direct_evidence
        : 0.0;
    float descendant_mean = descendant_evidence > 0.0
        ? uintBitsToFloat(rollup_mean[child.rollup_moment_offset + band])
        : 0.0;
    return (direct_mean * direct_evidence
          + descendant_mean * descendant_evidence) / total;
}

void main() {
    uint node_id = gl_GlobalInvocationID.x;
    if (node_id >= control[0]) return;
    SensorMipNode parent = nodes[node_id];
    if (parent.level != target_level || parent.first_child_id == NO_NODE) return;
    bool even_division = (control[8] & 1u) == 0u;
    uint first = parent.first_child_id;

    float e0 = child_evidence(first + 0u);
    float e1 = child_evidence(first + 1u);
    float e2 = child_evidence(first + 2u);
    float e3 = child_evidence(first + 3u);
    float evidence = min(min(e0, e1), min(e2, e3));
    bool valid = e0 > 0.0 && e1 > 0.0 && e2 > 0.0 && e3 > 0.0;
    if (!even_division) {
        float e4 = child_evidence(first + 4u);
        float e5 = child_evidence(first + 5u);
        float e6 = child_evidence(first + 6u);
        float e7 = child_evidence(first + 7u);
        float e8 = child_evidence(first + 8u);
        valid = valid && e4 > 0.0 && e5 > 0.0 && e6 > 0.0
            && e7 > 0.0 && e8 > 0.0;
        evidence = min(evidence, min(min(e4, e5), min(min(e6, e7), e8)));
    }
    if (!valid || isnan(evidence) || isinf(evidence)) {
        rollup_evidence[node_id] = floatBitsToUint(0.0);
        atomicAnd(nodes[node_id].flags, ~NODE_ROLL_VALID);
        return;
    }
    for (uint band = 0u; band < n_bands; ++band) {
        float mean;
        if (even_division) {
            mean = (
                child_mean(first + 0u, band) + child_mean(first + 1u, band)
                + child_mean(first + 2u, band) + child_mean(first + 3u, band)
            ) * 0.25;
        } else {
            mean = (
                child_mean(first + 0u, band) + child_mean(first + 1u, band)
                + child_mean(first + 2u, band) + child_mean(first + 3u, band)
                + child_mean(first + 4u, band) + child_mean(first + 5u, band)
                + child_mean(first + 6u, band) + child_mean(first + 7u, band)
                + child_mean(first + 8u, band)
            ) * (1.0 / 9.0);
        }
        rollup_mean[parent.rollup_moment_offset + band] = floatBitsToUint(mean);
    }
    rollup_evidence[node_id] = floatBitsToUint(evidence);
    atomicOr(nodes[node_id].flags, NODE_ROLL_VALID | NODE_DIRTY);
}
