#version 430 core

// Presentation-only reconstruction of the sparse sensor hierarchy.  One
// invocation owns one display pixel, so no atomics are needed and this buffer
// can never add energy to the scientific exposure accumulator.
layout(local_size_x = 64) in;

const uint NO_NODE = 0xFFFFFFFFu;
const uint MAX_BANDS = 32u;

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

layout(std430, binding = 0) readonly buffer NodeBuf { SensorMipNode nodes[]; };
layout(std430, binding = 1) readonly buffer DirectSumBuf { uint direct_sum[]; };
layout(std430, binding = 2) readonly buffer DirectWeightBuf { uint direct_weight[]; };
layout(std430, binding = 3) readonly buffer RollupMeanBuf { uint rollup_mean[]; };
layout(std430, binding = 4) readonly buffer RollupEvidenceBuf { uint rollup_evidence[]; };
layout(std430, binding = 5) readonly buffer ControlBuf { uint control[]; };
layout(std430, binding = 6) writeonly buffer PreviewRgbBuf { uint preview_rgb[]; };

uniform uint sensor_res;
uniform uint n_bands;
uniform vec3 rgb_w[MAX_BANDS];

vec3 resolved_rgb(uint node_id) {
    SensorMipNode node = nodes[node_id];
    float direct_e = uintBitsToFloat(direct_weight[node_id]);
    float child_e = uintBitsToFloat(rollup_evidence[node_id]);
    float total_e = direct_e + child_e;
    if (!(total_e > 0.0) || isnan(total_e) || isinf(total_e)) return vec3(0.0);
    vec3 rgb = vec3(0.0);
    uint bands = min(n_bands, MAX_BANDS);
    for (uint band = 0u; band < bands; ++band) {
        float direct_mean = direct_e > 0.0
            ? uintBitsToFloat(direct_sum[node.direct_moment_offset + band]) / direct_e
            : 0.0;
        float child_mean = child_e > 0.0
            ? uintBitsToFloat(rollup_mean[node.rollup_moment_offset + band])
            : 0.0;
        float mean = (direct_mean * direct_e + child_mean * child_e) / total_e;
        if (!isnan(mean) && !isinf(mean)) rgb += max(mean, 0.0) * rgb_w[band];
    }
    return rgb;
}

void main() {
    uint pixel = gl_GlobalInvocationID.x;
    uint pixel_count = sensor_res * sensor_res;
    if (pixel >= pixel_count || control[0] == 0u) return;
    uint x = pixel % sensor_res;
    uint y = pixel / sensor_res;
    vec2 uv = (vec2(x, y) + vec2(0.5)) / float(sensor_res);

    uint node_id = 0u;
    vec3 inherited = vec3(0.0);
    for (uint depth = 0u; depth < 64u; ++depth) {
        float evidence = uintBitsToFloat(direct_weight[node_id])
                       + uintBitsToFloat(rollup_evidence[node_id]);
        if (evidence > 0.0 && !isnan(evidence) && !isinf(evidence))
            inherited = resolved_rgb(node_id);
        SensorMipNode node = nodes[node_id];
        if (node.first_child_id == NO_NODE) break;
        vec2 extent = max(node.uv_bounds.zw - node.uv_bounds.xy, vec2(1.0e-20));
        vec2 local = clamp((uv - node.uv_bounds.xy) / extent, vec2(0.0), vec2(0.99999994));
        bool even_division = (control[8] & 1u) == 0u;
        uvec2 cell = even_division ? uvec2(local * 2.0) : uvec2(local * 3.0);
        uint slot = even_division ? cell.y * 2u + cell.x : cell.y * 3u + cell.x;
        uint child_id = node.first_child_id + slot;
        if (child_id >= control[0]) break;
        node_id = child_id;
    }

    preview_rgb[pixel] = floatBitsToUint(inherited.r);
    preview_rgb[pixel + pixel_count] = floatBitsToUint(inherited.g);
    preview_rgb[pixel + 2u * pixel_count] = floatBitsToUint(inherited.b);
}
