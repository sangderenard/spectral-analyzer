#version 430 core

layout(local_size_x = 64) in;

const uint NODE_DIRECT_VALID = 1u << 3;
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

struct SensorMipLineage {
    uint node_id;
    uint level;
    float global_u;
    float global_v;
    float local_u;
    float local_v;
    float estimator_weight;
    uint sample_index;
};

layout(std430, binding = 0) buffer NodeBuf { SensorMipNode nodes[]; };
layout(std430, binding = 1) readonly buffer LineageBuf { SensorMipLineage lineage[]; };
layout(std430, binding = 2) readonly buffer SpectrumBuf { float spectra[]; };
layout(std430, binding = 3) buffer DirectSumBuf { uint direct_sum[]; };
layout(std430, binding = 4) buffer DirectSumSqBuf { uint direct_sum_sq[]; };
layout(std430, binding = 5) buffer DirectWeightBuf { uint direct_weight[]; };
layout(std430, binding = 6) readonly buffer ControlBuf { uint control[]; };
layout(std430, binding = 7) readonly buffer ScheduleBuf { uint schedule[]; };

uniform uint sample_count;
uniform uint n_bands;
uniform uint use_schedule_count;

void atomic_add_sum(uint index, float value) {
    uint previous = direct_sum[index];
    while (true) {
        float current = uintBitsToFloat(previous);
        uint next = floatBitsToUint(current + value);
        uint observed = atomicCompSwap(direct_sum[index], previous, next);
        if (observed == previous) return;
        previous = observed;
    }
}

void atomic_add_sum_sq(uint index, float value) {
    uint previous = direct_sum_sq[index];
    while (true) {
        uint next = floatBitsToUint(uintBitsToFloat(previous) + value);
        uint observed = atomicCompSwap(direct_sum_sq[index], previous, next);
        if (observed == previous) return;
        previous = observed;
    }
}

void atomic_add_weight(uint index, float value) {
    uint previous = direct_weight[index];
    while (true) {
        uint next = floatBitsToUint(uintBitsToFloat(previous) + value);
        uint observed = atomicCompSwap(direct_weight[index], previous, next);
        if (observed == previous) return;
        previous = observed;
    }
}

void main() {
    uint sample_id = gl_GlobalInvocationID.x;
    uint active_sample_count = use_schedule_count != 0u
        ? schedule[0] * control[7] : sample_count;
    if (sample_id >= active_sample_count) return;
    SensorMipLineage lineage_item = lineage[sample_id];
    if (lineage_item.node_id >= control[0]) return;
    float weight = lineage_item.estimator_weight;
    if (isnan(weight) || isinf(weight) || weight <= 0.0) return;

    for (uint band = 0u; band < n_bands; ++band) {
        float value = spectra[sample_id * n_bands + band];
        if (isnan(value) || isinf(value) || value < 0.0) return;
    }

    SensorMipNode node = nodes[lineage_item.node_id];
    for (uint band = 0u; band < n_bands; ++band) {
        float value = spectra[sample_id * n_bands + band];
        uint offset = node.direct_moment_offset + band;
        atomic_add_sum(offset, weight * value);
        atomic_add_sum_sq(offset, weight * value * value);
    }
    atomic_add_weight(lineage_item.node_id, weight);
    atomicAdd(nodes[lineage_item.node_id].direct_sample_count, 1u);
    atomicOr(nodes[lineage_item.node_id].flags, NODE_DIRECT_VALID | NODE_DIRTY);
}
