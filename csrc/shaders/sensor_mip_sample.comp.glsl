#version 430 core

/* One workgroup owns one recursive sensor node. Each lane advances through
 * that node's samples, so no invocation can spill into an adjacent node. */
layout(local_size_x = 64) in;

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

struct SensorMipWork {
    uint node_id;
    uint sample_begin;
    uint sample_count;
    uint output_offset;
    uint seed;
    float priority;
    uint flags;
    uint pad0;
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

layout(std430, binding = 0) readonly buffer NodeBuf { SensorMipNode nodes[]; };
layout(std430, binding = 1) readonly buffer WorkBuf { SensorMipWork work[]; };
layout(std430, binding = 2) writeonly buffer LineageBuf { SensorMipLineage lineage[]; };
layout(std430, binding = 3) writeonly buffer SpectrumBuf { uint sample_spectra[]; };
layout(std430, binding = 4) readonly buffer ScheduleBuf { uint schedule[]; };

uniform uint work_count;
uniform uint node_count;
uniform uint output_capacity;
uniform uint n_bands;
uniform uint use_schedule_count;

uint mix_bits(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

float unit_float(uint x) {
    return float(x >> 8) * (1.0 / 16777216.0);
}

float radical_inverse_2(uint n) {
    n = bitfieldReverse(n);
    return float(n) * 2.3283064365386963e-10;
}

float radical_inverse_3(uint n) {
    float reversed = 0.0;
    float inverse = 1.0 / 3.0;
    while (n != 0u) {
        reversed += float(n % 3u) * inverse;
        n /= 3u;
        inverse /= 3.0;
    }
    return reversed;
}

void main() {
    uint work_index = gl_WorkGroupID.x;
    uint active_work_count = use_schedule_count != 0u ? schedule[0] : work_count;
    if (work_index >= active_work_count) return;
    SensorMipWork item = work[work_index];
    if (item.node_id >= node_count || item.output_offset >= output_capacity) return;
    SensorMipNode node = nodes[item.node_id];
    uint scramble = mix_bits(item.seed ^ mix_bits(item.node_id));
    vec2 rotation = vec2(unit_float(scramble), unit_float(mix_bits(scramble + 1u)));

    for (uint local_index = gl_LocalInvocationID.x;
         local_index < item.sample_count;
         local_index += gl_WorkGroupSize.x) {
        uint output_index = item.output_offset + local_index;
        if (output_index >= output_capacity) continue;
        uint sequence_index = item.sample_begin + local_index;
        vec2 local_uv = fract(vec2(
            radical_inverse_2(sequence_index + 1u),
            radical_inverse_3(sequence_index + 1u)) + rotation);
        SensorMipLineage result;
        result.node_id = item.node_id;
        result.level = node.level;
        result.global_u = mix(node.uv_bounds.x, node.uv_bounds.z, local_uv.x);
        result.global_v = mix(node.uv_bounds.y, node.uv_bounds.w, local_uv.y);
        result.local_u = local_uv.x;
        result.local_v = local_uv.y;
        result.estimator_weight = 1.0;
        result.sample_index = sequence_index;
        lineage[output_index] = result;
        for (uint band = 0u; band < n_bands; ++band)
            sample_spectra[output_index * n_bands + band] = floatBitsToUint(0.0);
    }
}
