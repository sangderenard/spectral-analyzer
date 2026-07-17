#version 430 core

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

struct PriorityFeatures {
    float uncertainty;
    float ambiguity;
    float learned;
    float requested;
};

layout(std430, binding = 0) buffer NodeBuf { SensorMipNode nodes[]; };
layout(std430, binding = 1) readonly buffer FeatureBuf { PriorityFeatures features[]; };
layout(std430, binding = 2) readonly buffer DirectSumBuf { uint direct_sum[]; };
layout(std430, binding = 3) readonly buffer DirectSumSqBuf { uint direct_sum_sq[]; };
layout(std430, binding = 4) readonly buffer DirectWeightBuf { uint direct_weight[]; };
layout(std430, binding = 5) readonly buffer ControlBuf { uint control[]; };
/* Current GPU-resident presentation accumulation, planar RGB.  This is a
 * scorer input only; it never changes topology or spectral evidence. */
layout(std430, binding = 6) readonly buffer SensorRgbBuf { uint sensor_rgb[]; };
layout(std430, binding = 7) readonly buffer SensorWeightBuf { uint sensor_weight[]; };
layout(std430, binding = 8) readonly buffer LearnedMapBuf { float learned_map[]; };
layout(std430, binding = 9) readonly buffer RequestedMapBuf { float requested_map[]; };

uniform uint node_count;
uniform uint n_bands;
uniform uint derive_from_moments;
uniform uint sensor_res;
uniform vec4 feature_weights;
uniform uint learned_map_enabled;
uniform uint requested_map_enabled;

float valid_nonnegative(float value) {
    return (isnan(value) || isinf(value) || value < 0.0) ? 0.0 : value;
}

float sensor_luma(ivec2 p) {
    if (sensor_res == 0u) return 0.0;
    p = clamp(p, ivec2(0), ivec2(int(sensor_res) - 1));
    uint pix = uint(p.x) * sensor_res + uint(p.y);
    uint plane = sensor_res * sensor_res;
    vec3 rgb = vec3(
        uintBitsToFloat(sensor_rgb[pix]),
        uintBitsToFloat(sensor_rgb[pix + plane]),
        uintBitsToFloat(sensor_rgb[pix + 2u * plane]));
    float exposure = uintBitsToFloat(sensor_weight[pix]);
    rgb = exposure > 0.0 ? rgb / exposure : vec3(0.0);
    return max(0.0, dot(rgb, vec3(0.2126, 0.7152, 0.0722)));
}

float sensor_exposure(ivec2 p) {
    if (sensor_res == 0u) return 0.0;
    p = clamp(p, ivec2(0), ivec2(int(sensor_res) - 1));
    uint pix = uint(p.x) * sensor_res + uint(p.y);
    return valid_nonnegative(uintBitsToFloat(sensor_weight[pix]));
}

void main() {
    uint node_id = gl_GlobalInvocationID.x;
    if (node_id >= node_count || node_id >= control[0]) return;
    PriorityFeatures f;
    if (derive_from_moments != 0u) {
        SensorMipNode node = nodes[node_id];
        float weight = uintBitsToFloat(direct_weight[node_id]);
        float mean = 0.0;
        float variance = 0.0;
        if (weight > 0.0) {
            for (uint band = 0u; band < n_bands; ++band) {
                uint offset = node.direct_moment_offset + band;
                float band_mean = uintBitsToFloat(direct_sum[offset]) / weight;
                float band_second = uintBitsToFloat(direct_sum_sq[offset]) / weight;
                mean += band_mean;
                variance += max(0.0, band_second - band_mean * band_mean);
            }
        }
        float inv_bands = 1.0 / float(max(1u, n_bands));
        mean *= inv_bands;
        variance *= inv_bands;
        float stderr = sqrt(variance / max(weight, 1.0));
        float relative_uncertainty = stderr / max(abs(mean), 1.0e-6);
        float parent_mean = 0.0;
        if (node.parent_id < node_count) {
            float parent_weight = uintBitsToFloat(direct_weight[node.parent_id]);
            if (parent_weight > 0.0) {
                SensorMipNode parent = nodes[node.parent_id];
                for (uint band = 0u; band < n_bands; ++band)
                    parent_mean += uintBitsToFloat(
                        direct_sum[parent.direct_moment_offset + band]) / parent_weight;
                parent_mean *= inv_bands;
            }
        }
        float contrast = abs(mean - parent_mean) / max(abs(mean) + abs(parent_mean), 1.0e-6);
        vec2 center_uv = 0.5 * (node.uv_bounds.xy + node.uv_bounds.zw);
        vec2 radius_uv = 0.45 * (node.uv_bounds.zw - node.uv_bounds.xy);
        float image_mean = 0.0;
        float image_second = 0.0;
        float image_lit = 0.0;
        float learned_peak = 0.0;
        float requested_peak = 0.0;
        float image_exposure = 0.0;
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                vec2 uv = center_uv + radius_uv * vec2(float(ox), float(oy));
                ivec2 pixel = ivec2(
                    int(clamp(uv.y, 0.0, 0.99999994) * float(sensor_res)),
                    int(clamp(uv.x, 0.0, 0.99999994) * float(sensor_res)));
                float value = sensor_luma(pixel);
                image_exposure += sensor_exposure(pixel);
                image_mean += value;
                image_second += value * value;
                image_lit += value > 0.0 ? 1.0 : 0.0;
                if (learned_map_enabled != 0u) {
                    uint learned_pixel = uint(pixel.x) * sensor_res + uint(pixel.y);
                    learned_peak = max(learned_peak, valid_nonnegative(learned_map[learned_pixel]));
                }
                if (requested_map_enabled != 0u) {
                    uint requested_pixel = uint(pixel.x) * sensor_res + uint(pixel.y);
                    requested_peak = max(
                        requested_peak,
                        valid_nonnegative(requested_map[requested_pixel]));
                }
            }
        }
        image_mean /= 9.0;
        image_exposure /= 9.0;
        float image_variance = max(0.0, image_second / 9.0 - image_mean * image_mean);
        float image_ambiguity = sqrt(image_variance) / max(image_mean, 1.0e-7);
        float incomplete_support = image_lit > 0.0 ? (1.0 - image_lit / 9.0) : 0.0;
        /* This is the replaceable attention boundary: uncertainty estimates
         * remaining Monte-Carlo work, ambiguity emphasizes spatial evidence
         * that differs from its parent, and learned remains a separate input
         * channel for the compact discriminator network. */
        f.uncertainty = min(relative_uncertainty, 100.0);
        f.ambiguity = min(contrast, 1.0);
        f.learned = learned_map_enabled != 0u
            ? learned_peak
            : min(image_ambiguity, 10.0) + incomplete_support;
        float inherited = node.parent_id < node_count
            ? 0.1 * nodes[node.parent_id].priority : 0.0;
        /* Authored work requests identify where evidence is needed, not a
         * permanent burn-in priority. As exposure returns, their pressure
         * decays and underexposed sibling UI chunks overtake them. */
        float unresolved_request = requested_peak / sqrt(1.0 + image_exposure);
        f.requested = max(max(0.01, inherited), unresolved_request);
    } else {
        f = features[node_id];
    }
    vec4 clean = vec4(
        valid_nonnegative(f.uncertainty),
        valid_nonnegative(f.ambiguity),
        valid_nonnegative(f.learned),
        valid_nonnegative(f.requested));
    nodes[node_id].priority = max(0.0, dot(clean, max(feature_weights, vec4(0.0))));
}
