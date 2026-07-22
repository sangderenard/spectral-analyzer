#version 430 core

/* Deterministic first-surface preview resolver. Rays have already travelled
 * through normal T1/T2 lens transport. This stage consumes scan-tagged T3
 * terminals and writes one complete back texture; misses remain transparent. */
layout(local_size_x = 64) in;

#define MAX_BANDS 32
#define MAT_FULL_BANDS 32
#define MAT_BAND_STRIDE 12
#define INTENT_STRIDE (22 + 2 * MAX_BANDS)
#define TERMINAL_STRIDE (28 + 2 * MAX_BANDS)
#define SURFACE_SCAN_BIT 32u

layout(std430, binding = 0) readonly buffer TerminalBuf { float records[]; };
layout(std430, binding = 1) readonly buffer MetaBuf { uint meta[]; };
layout(std430, binding = 2) readonly buffer MatBandBuf { float mat_bands[]; };
layout(rgba32f, binding = 0) uniform writeonly image2D scan_out;

uniform int max_children;
uniform int n_bands;
uniform int n_mats;
uniform int sensor_res;
uniform float sensor_half_w;
uniform float sensor_half_h;
uniform vec3 rgb_w[MAX_BANDS];
uniform int clear_only;

int mat_off(int mat, int band) {
    return (clamp(mat, 0, max(n_mats - 1, 0)) * MAT_FULL_BANDS
          + clamp(band, 0, MAT_FULL_BANDS - 1)) * MAT_BAND_STRIDE;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint pixel_count = uint(sensor_res * sensor_res);
    if (clear_only != 0) {
        if (gid < pixel_count) {
            uint y = gid / uint(sensor_res);
            uint x = gid % uint(sensor_res);
            imageStore(scan_out, ivec2(int(x), int(y)), vec4(0.0));
        }
        return;
    }

    uint terminal_count = meta[1];
    if (gid >= terminal_count) return;
    uint base = uint(max_children * INTENT_STRIDE) + gid * uint(TERMINAL_STRIDE);
    uint color_flag = floatBitsToUint(records[base + 16u]);
    if ((color_flag & SURFACE_SCAN_BIT) == 0u) return;
    int mat = floatBitsToInt(records[base + 15u]);
    if (mat < 0 || mat >= n_mats) return;

    float sy = records[base + 23u];
    float sz = records[base + 24u];
    int px = int(floor((sy + sensor_half_w) * float(sensor_res)
                       / (2.0 * sensor_half_w)));
    int py = int(floor((sz + sensor_half_h) * float(sensor_res)
                       / (2.0 * sensor_half_h)));
    if (px < 0 || py < 0 || px >= sensor_res || py >= sensor_res) return;

    vec3 normal = normalize(vec3(records[base+3u], records[base+4u], records[base+5u]));
    vec3 incoming = normalize(vec3(records[base+6u], records[base+7u], records[base+8u]));
    float headlight = max(0.0, dot(normal, -incoming));
    float scan_light = 0.12 + 0.88 * headlight;
    vec3 rgb = vec3(0.0);
    float weight = 0.0;
    int bands = min(n_bands, MAX_BANDS);
    for (int band = 0; band < bands; ++band) {
        int off = mat_off(mat, band);
        float refl_amp = max(0.0, mat_bands[off + 2]);
        float emission = max(0.0, mat_bands[off + 5]);
        vec3 w = max(rgb_w[band], vec3(0.0));
        rgb += w * (scan_light * refl_amp * refl_amp + emission);
        weight += max(w.r, max(w.g, w.b));
    }
    if (weight > 0.0) rgb /= weight;
    imageStore(scan_out, ivec2(px, py), vec4(max(rgb, vec3(0.0)), 1.0));
}
