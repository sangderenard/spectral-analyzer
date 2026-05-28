#version 430 core
/* ────────────────────────────────────────────────────────────────────────────
 * t5_full_connect.comp.glsl  —  GPU BDPT T5 full brute-force connection pass
 *
 * Each invocation = ONE flattened camera vertex.
 * Inner loop iterates every light vertex in T5LightVertBuf — O(N_cam×N_light).
 * No spatial index, no hash lookup — flat sequential reads maximise GPU memory
 * bandwidth and keep every warp reading the same light-vert index simultaneously
 * (coalesced L2 access pattern).
 *
 * Dispatch: glDispatchCompute(ceil(n_cam_verts / 64), 1, 1)
 *
 * ── SSBO binding map ──────────────────────────────────────────────────────
 *  binding  buffer          access    description
 *  -------  --------------- --------  ---------------------------------------
 *    0      T5LightVertBuf  readonly  Flat light vertices.
 *                                     Stride = T5_LGV_STRIDE = 12 floats.
 *                                       [0..2]  pos xyz
 *                                       [3..5]  normal xyz
 *                                       [6]     throughput_scalar
 *                                       [7]     uintBitsToFloat(flags)
 *                                       [8]     uintBitsToFloat(subpath_id)
 *                                       [9]     uintBitsToFloat(vert_info)
 *                                       [10]    beta_lum
 *                                       [11]    0 (pad)
 *    1      T5CamVertBuf    readonly  Flat camera vertices.
 *                                     Stride = T5_CGV_STRIDE = 14 floats.
 *                                       [0..2]  pos xyz
 *                                       [3..5]  normal xyz
 *                                       [6]     throughput_scalar
 *                                       [7]     uintBitsToFloat(flags)
 *                                       [8]     uintBitsToFloat(subpath_id)
 *                                       [9]     uintBitsToFloat(vert_index)
 *                                       [10]    beta_lum
 *                                       [11]    sensor_origin_y
 *                                       [12]    sensor_origin_z
 *                                       [13]    0 (pad)
 *    2      T5PixelBuf      coherent  Pixel accumulator.
 *                                     3 × res² uint32 bit-cast floats.
 *                                     Layout: R[res²] G[res²] B[res²].
 *                                     Written via float CAS (no ext needed).
 *    3      T5ParamsBuf     readonly  T5GpuParams std430 block (32 bytes).
 *    5      BvhBuf          readonly  BVH nodes          (shadow preamble)
 *    6      TriIdBuf        readonly  BVH triangle IDs   (shadow preamble)
 *    7      TriFullBuf      readonly  full triangle data  (shadow preamble)
 *
 * ── T5GpuParams (std430, binding 3, 32 bytes) ─────────────────────────────
 *   float  min_geom        geometry-term floor
 *   float  sensor_half_w   sensor half-width  (Y axis, metres)
 *   float  sensor_half_h   sensor half-height (Z axis, metres)
 *   float  _pad0
 *   uint   n_light_verts
 *   uint   n_cam_verts
 *   int    sensor_res      pixel grid side (res×res)
 *   uint   _pad1
 * ────────────────────────────────────────────────────────────────────────── */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define T5_LGV_STRIDE  12
#define T5_CGV_STRIDE  14

layout(std430, binding = 0) readonly buffer T5LightVertBuf {
    float light_verts[];   /* n_light_verts × T5_LGV_STRIDE */
};

layout(std430, binding = 1) readonly buffer T5CamVertBuf {
    float cam_verts[];     /* n_cam_verts × T5_CGV_STRIDE */
};

layout(std430, binding = 2) coherent buffer T5PixelBuf {
    uint pixel_accum[];    /* 3 × sensor_res² uint32 float-bits */
};

layout(std430, binding = 3) readonly buffer T5ParamsBuf {
    float  min_geom;
    float  sensor_half_w;
    float  sensor_half_h;
    float  _pad0;
    uint   n_light_verts;    /* total (bounds check only)  */
    uint   n_cam_verts;
    int    sensor_res;
    uint   light_batch_size; /* how many light verts this dispatch covers */
    uint   light_offset;     /* first light vert index for this dispatch  */
    uint   _pad1;
};

/* ── Float atomic add via CAS spin-loop ─────────────────────────────────── *
 * Avoids GL_EXT_shader_atomic_float.  Standard on all GL 4.3+ hardware.    */
void atomic_add_float(uint idx, float val) {
    uint expected = pixel_accum[idx];
    for (int i = 0; i < 64; ++i) {
        float fexp    = uintBitsToFloat(expected);
        uint  desired = floatBitsToUint(fexp + val);
        uint  actual  = atomicCompSwap(pixel_accum[idx], expected, desired);
        if (actual == expected) return;
        expected = actual;
    }
}

/* ── Main ────────────────────────────────────────────────────────────────── */
void main() {
    const uint gid = gl_GlobalInvocationID.x;
    if (gid >= n_cam_verts) return;

    /* ── Load camera vertex ─────────────────────────────────────────────── */
    const uint  cb     = gid * uint(T5_CGV_STRIDE);
    const vec3  c_pos  = vec3(cam_verts[cb+0u], cam_verts[cb+1u], cam_verts[cb+2u]);
    const vec3  c_norm = vec3(cam_verts[cb+3u], cam_verts[cb+4u], cam_verts[cb+5u]);
    const float c_beta = cam_verts[cb+10u];
    const float c_soy  = cam_verts[cb+11u];
    const float c_soz  = cam_verts[cb+12u];

    if (c_beta < 1e-15f) return;

    /* ── Pixel index ────────────────────────────────────────────────────── */
    const float inv_w = float(sensor_res) / (2.0f * sensor_half_w);
    const float inv_h = float(sensor_res) / (2.0f * sensor_half_h);
    const int iy = int((c_soy + sensor_half_w) * inv_w);
    const int iz = int((c_soz + sensor_half_h) * inv_h);
    if (iy < 0 || iy >= sensor_res || iz < 0 || iz >= sensor_res) return;
    const uint px  = uint(iy * sensor_res + iz);
    const uint pix = uint(sensor_res) * uint(sensor_res);

    /* ── Scan this batch of light vertices ──────────────────────────────── *
     * All invocations walk the same [light_offset, light_offset+batch) range*
     * in lockstep — warps coalesce on every light vert load.  C++ loops    *
     * service_t5_job through batches with a tiny params re-upload each time.*
     * This keeps each dispatch short enough to avoid Windows GPU TDR.       */
    const uint li_end = min(light_offset + light_batch_size, n_light_verts);
    float lum = 0.0f;
    for (uint li = light_offset; li < li_end; ++li) {
        const uint  lb     = li * uint(T5_LGV_STRIDE);
        const vec3  l_pos  = vec3(light_verts[lb+0u], light_verts[lb+1u], light_verts[lb+2u]);
        const vec3  l_norm = vec3(light_verts[lb+3u], light_verts[lb+4u], light_verts[lb+5u]);
        const float l_beta = light_verts[lb+10u];

        if (l_beta < 1e-15f) continue;

        const vec3  dv    = l_pos - c_pos;
        const float dist2 = dot(dv, dv);
        if (dist2 < 1e-12f) continue;
        const float dist  = sqrt(dist2);
        const vec3  wc    = dv / dist;
        const float geom  = abs(dot(c_norm, wc)) * abs(dot(l_norm, -wc)) / dist2;
        if (geom < min_geom) continue;

        if (shadow_occluded(c_pos, l_pos)) continue;

        lum += c_beta * l_beta * geom;
    }

    /* ── Write to pixel buffer (neutral 1/3 split across channels) ──────── *
     * CPU post-pass spectral recolour replaces this weight later.           */
    if (lum > 0.0f) {
        const float third = lum * (1.0f / 3.0f);
        atomic_add_float(px,          third);
        atomic_add_float(px + pix,    third);
        atomic_add_float(px + 2u*pix, third);
    }
}
