#version 430 core

layout(local_size_x = 64) in;

#define MAX_BANDS 32
#define INTENT_STRIDE (20 + 2 * MAX_BANDS)

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

layout(std430, binding = 0) readonly buffer NodeBuf { SensorMipNode nodes[]; };
layout(std430, binding = 1) readonly buffer LineageBuf { SensorMipLineage lineage[]; };
layout(std430, binding = 2) readonly buffer ControlBuf { uint control[]; };
layout(std430, binding = 3) readonly buffer ScheduleBuf { uint schedule[]; };
layout(std430, binding = 4) buffer CounterBuf { uint counters[]; };
layout(std430, binding = 5) writeonly buffer IntentBuf { float intents[]; };
/* One exposure weight per sensor bin, float bit-cast for CAS atomics. */
layout(std430, binding = 6) coherent buffer SensorWeightBuf { uint sensor_weight[]; };

uniform vec3 sensor_center;
uniform vec3 sensor_right;
uniform vec3 sensor_up;
uniform float sensor_half_w;
uniform float sensor_half_h;
uniform vec3 camera_target;
uniform vec3 aperture_right;
uniform vec3 aperture_up;
uniform float aperture_radius;
uniform uint sensor_res;
uniform uint target_mode;
uniform uint n_bands;
uniform uint max_bounces;
uniform float min_amplitude;
uniform float exposure_weight;
uniform uint epoch_seed;

uint mix_bits(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

float unit_float(uint x) { return float(x >> 8) * (1.0 / 16777216.0); }

void atomic_add_weight(uint index, float value) {
    if (value <= 0.0 || isnan(value) || isinf(value)) return;
    uint expected = sensor_weight[index];
    while (true) {
        uint desired = floatBitsToUint(uintBitsToFloat(expected) + value);
        uint actual = atomicCompSwap(sensor_weight[index], expected, desired);
        if (actual == expected) return;
        expected = actual;
    }
}

void splat_exposure_weight(float sensor_y, float sensor_z, float value) {
    float fy = (sensor_y + sensor_half_w) * float(sensor_res)
             / (2.0 * sensor_half_w) - 0.5;
    float fz = (sensor_z + sensor_half_h) * float(sensor_res)
             / (2.0 * sensor_half_h) - 0.5;
    int y0 = int(floor(fy));
    int z0 = int(floor(fz));
    float weights[4];
    int ys[4];
    int zs[4];
    float total = 0.0;
    int k = 0;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dz = 0; dz <= 1; ++dz) {
            int iy = y0 + dy;
            int iz = z0 + dz;
            float w = max(0.0, 1.0 - abs(float(iy) - fy))
                    * max(0.0, 1.0 - abs(float(iz) - fz));
            ys[k] = iy; zs[k] = iz; weights[k] = w;
            if (iy >= 0 && iy < int(sensor_res) && iz >= 0 && iz < int(sensor_res))
                total += w;
            ++k;
        }
    }
    if (total <= 0.0) return;
    for (int i = 0; i < 4; ++i) {
        int iy = ys[i], iz = zs[i];
        if (iy < 0 || iy >= int(sensor_res) || iz < 0 || iz >= int(sensor_res)) continue;
        atomic_add_weight(uint(iy) * sensor_res + uint(iz), value * weights[i] / total);
    }
}

void main() {
    uint total_samples = schedule[0] * control[7];
    uint output_index = gl_GlobalInvocationID.x;
    if (output_index == 0u) {
        counters[8] = total_samples;
        counters[9] = (total_samples + 63u) / 64u;
        counters[10] = 1u;
        counters[11] = 1u;
    }
    if (output_index >= total_samples) return;

    SensorMipLineage sample_lineage = lineage[output_index];
    SensorMipNode node = nodes[sample_lineage.node_id];
    float y = -sensor_half_w + sample_lineage.global_v * (2.0 * sensor_half_w);
    float z = -sensor_half_h + sample_lineage.global_u * (2.0 * sensor_half_h);
    vec3 origin = sensor_center + sensor_right * y + sensor_up * z;
    /* Count the camera primary once, including misses. T5 may create several
     * camera vertices for this path, but all of their MIS strategies form one
     * radiance sample and share this single normalization weight. */
    splat_exposure_weight(y, z, sample_lineage.estimator_weight);

    uint scramble = mix_bits(epoch_seed ^ sample_lineage.node_id
                             ^ mix_bits(sample_lineage.sample_index));
    float disk_u = fract(unit_float(scramble) + 0.754877666
                         * float(sample_lineage.sample_index + 1u));
    float disk_v = fract(unit_float(mix_bits(scramble + 1u)) + 0.569840296
                         * float(sample_lineage.sample_index + 1u));
    float radius = aperture_radius * sqrt(disk_u);
    float angle = 6.28318530718 * disk_v;
    vec3 target = camera_target
                + aperture_right * (radius * cos(angle))
                + aperture_up * (radius * sin(angle));
    vec3 direction = target_mode == 1u ? origin - target : target - origin;
    float direction_length = length(direction);
    direction = direction_length > 1.0e-12 ? direction / direction_length
                                            : vec3(-1.0, 0.0, 0.0);
    origin += direction * 2.56e-8;

    uint base = output_index * uint(INTENT_STRIDE);
    intents[base + 0u] = origin.x; intents[base + 1u] = origin.y; intents[base + 2u] = origin.z;
    intents[base + 3u] = direction.x; intents[base + 4u] = direction.y; intents[base + 5u] = direction.z;
    intents[base + 6u] = 0.0;
    intents[base + 7u] = intBitsToFloat(-1);
    intents[base + 8u] = uintBitsToFloat(0u);
    uint sensor_y = min(sensor_res - 1u,
                        uint(clamp(sample_lineage.global_v, 0.0, 0.99999994)
                             * float(sensor_res)));
    uint sensor_z = min(sensor_res - 1u,
                        uint(clamp(sample_lineage.global_u, 0.0, 0.99999994)
                             * float(sensor_res)));
    intents[base + 9u] = intBitsToFloat(int(sensor_y * sensor_res + sensor_z));
    intents[base + 10u] = intBitsToFloat(0);
    intents[base + 11u] = intBitsToFloat(int(max(1u, max_bounces)));
    intents[base + 12u] = min_amplitude;
    intents[base + 13u] = uintBitsToFloat(output_index);
    uint tag_hi = 0xC0000000u | ((sample_lineage.level & 0xFu) << 24u)
                | (sample_lineage.node_id & 0x00FFFFFFu);
    intents[base + 14u] = uintBitsToFloat(tag_hi);
    intents[base + 15u] = uintBitsToFloat(1u);
    intents[base + 16u] = node.priority;
    intents[base + 17u] = y;
    intents[base + 18u] = z;
    intents[base + 19u] = uintBitsToFloat(
        0x80000000u | ((mix_bits(epoch_seed) + output_index + 1u) & 0x7FFFFFFFu));
    for (uint band = 0u; band < MAX_BANDS; ++band) {
        intents[base + 20u + band] = 0.0;
        intents[base + 20u + MAX_BANDS + band] = 0.0;
    }
    uint sampled_band = sample_lineage.sample_index % max(1u, n_bands);
    intents[base + 20u + sampled_band] = exposure_weight * float(max(1u, n_bands));
}
