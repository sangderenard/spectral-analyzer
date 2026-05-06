#version 430 core
/**
 * emitter_angle_reduce.comp.glsl
 *
 * GPU contract for texture scrim/family-of-angles reduction.
 *
 * One workgroup reduces one texture-array layer.  This intentionally mirrors
 * emitter_angle_kernel.cpp's output record so CPU and GL compute paths can be
 * compared directly.  Dispatch with local_size_x = 256 and groups = layers.
 */

layout(local_size_x = 256) in;

layout(binding = 0) uniform sampler2DArray uEmitterTexture;

layout(std430, binding = 1) buffer MetricsOut {
    // 16 floats per layer:
    // 0 flux_scale, 1 active_fraction, 2 active_density, 3 centroid_u,
    // 4 centroid_v, 5 axis_u, 6 axis_v, 7 spread_major,
    // 8 spread_minor, 9 cone_cos, 10 cone_solid_angle, 11..15 reserved
    float metrics[];
};

uniform int uWidth;
uniform int uHeight;
uniform float uActiveThreshold;

shared float s_w[256];
shared float s_u[256];
shared float s_v[256];
shared float s_uu[256];
shared float s_uv[256];
shared float s_vv[256];
shared uint  s_a[256];

float srgb_to_linear(float c) {
    return (c <= 0.04045) ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint layer = gl_WorkGroupID.x;
    uint n = uint(max(uWidth * uHeight, 1));

    float wsum = 0.0;
    float usum = 0.0;
    float vsum = 0.0;
    float uusum = 0.0;
    float uvsum = 0.0;
    float vvsum = 0.0;
    uint active = 0u;

    for (uint i = tid; i < n; i += 256u) {
        uint x = i % uint(uWidth);
        uint y = i / uint(uWidth);
        vec2 uv = (vec2(x, y) + vec2(0.5)) / vec2(max(uWidth, 1), max(uHeight, 1));
        vec4 tex = texelFetch(uEmitterTexture, ivec3(int(x), int(y), int(layer)), 0);
        vec3 lin = vec3(srgb_to_linear(tex.r), srgb_to_linear(tex.g), srgb_to_linear(tex.b));
        float lum = dot(lin, vec3(0.2126, 0.7152, 0.0722)) * tex.a;
        if (lum > uActiveThreshold) active++;
        wsum += lum;
        usum += uv.x * lum;
        vsum += uv.y * lum;
        uusum += uv.x * uv.x * lum;
        uvsum += uv.x * uv.y * lum;
        vvsum += uv.y * uv.y * lum;
    }

    s_w[tid] = wsum;
    s_u[tid] = usum;
    s_v[tid] = vsum;
    s_uu[tid] = uusum;
    s_uv[tid] = uvsum;
    s_vv[tid] = vvsum;
    s_a[tid] = active;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_w[tid] += s_w[tid + stride];
            s_u[tid] += s_u[tid + stride];
            s_v[tid] += s_v[tid + stride];
            s_uu[tid] += s_uu[tid + stride];
            s_uv[tid] += s_uv[tid + stride];
            s_vv[tid] += s_vv[tid + stride];
            s_a[tid] += s_a[tid + stride];
        }
        barrier();
    }

    if (tid == 0u) {
        float inv_n = 1.0 / float(n);
        float w = s_w[0];
        float cu = (w > 1e-12) ? s_u[0] / w : 0.5;
        float cv = (w > 1e-12) ? s_v[0] / w : 0.5;
        float c00 = (w > 1e-12) ? s_uu[0] / w - cu * cu : 0.0;
        float c01 = (w > 1e-12) ? s_uv[0] / w - cu * cv : 0.0;
        float c11 = (w > 1e-12) ? s_vv[0] / w - cv * cv : 0.0;
        float tr = c00 + c11;
        float root = sqrt(max(0.0, (c00 - c11) * (c00 - c11) + 4.0 * c01 * c01));
        float l0 = max(0.0, 0.5 * (tr + root));
        float l1 = max(0.0, 0.5 * (tr - root));
        vec2 axis = vec2(c01, l0 - c00);
        if (dot(axis, axis) <= 1e-12) axis = vec2(1.0, 0.0);
        axis = normalize(axis);

        float active_fraction = float(s_a[0]) * inv_n;
        float flux_scale = w * inv_n;
        float active_density = (active_fraction > 1e-8) ? flux_scale / active_fraction : 0.0;
        float cone_cos = 1.0 - clamp(active_fraction, 0.0, 1.0);
        float cone_solid_angle = 6.28318530718 * clamp(active_fraction, 0.0, 1.0);

        uint b = layer * 16u;
        metrics[b + 0u] = flux_scale;
        metrics[b + 1u] = active_fraction;
        metrics[b + 2u] = active_density;
        metrics[b + 3u] = cu;
        metrics[b + 4u] = cv;
        metrics[b + 5u] = axis.x;
        metrics[b + 6u] = axis.y;
        metrics[b + 7u] = sqrt(l0);
        metrics[b + 8u] = sqrt(l1);
        metrics[b + 9u] = cone_cos;
        metrics[b + 10u] = cone_solid_angle;
        for (uint k = 11u; k < 16u; ++k) metrics[b + k] = 0.0;
    }
}
