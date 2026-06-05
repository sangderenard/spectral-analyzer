#version 430 core
/* ─────────────────────────────────────────────────────────────────────────────
 * sensor_terminal_splat.comp.glsl  —  GPU-resident backward-sensor accumulation
 *
 * Reads T3 terminal records from the combined child+terminal SSBO and splats
 * spectral RGB into the global sensor accumulator.  Replaces the CPU-side
 * terminal readback + sensor_accum update path when gpu_skip_record_readback
 * is active (Pass B of GPU-residency migration).
 *
 * Dispatch: ceil(nt_term / 64) workgroups × 1 × 1
 *
 * ── SSBO binding map ─────────────────────────────────────────────────────────
 *  binding 0  ChildIntBuf   ssbo_child_int   readonly  combined children+terminals
 *  binding 1  SensorRGBBuf  ssbo_sensor_rgb  coherent  3×res² float-CAS accumulator
 *
 * ── Terminal record layout (TERMINAL_STRIDE = 26 + 2*MAX_BANDS floats) ──────
 *  [0..2]   hit_pos xyz
 *  [3..5]   hit_n xyz
 *  [6..8]   incoming_dir xyz
 *  [9..11]  seg_start xyz
 *  [12]     path_len
 *  [13]     path_at_seg_start
 *  [14]     hit_tri          (intBitsToFloat)
 *  [15]     mat_idx          (intBitsToFloat)
 *  [16]     color_flag       (uintBitsToFloat)  ← 1 = backward sensor ray
 *  [17]     bounce           (intBitsToFloat)
 *  [18]     bounces_left     (intBitsToFloat)
 *  [19]     min_amplitude
 *  [20]     src_id           (intBitsToFloat)
 *  [21]     tag_lo           (uintBitsToFloat)
 *  [22]     tag_hi           (uintBitsToFloat)
 *  [23]     sensor_origin_y
 *  [24]     sensor_origin_z
 *  [25]     is_emissive_hit  (uintBitsToFloat)  ← 1 = emissive terminal
 *  [26..26+MAX_BANDS-1]           amp_re[MAX_BANDS]
 *  [26+MAX_BANDS..26+2*MAX_BANDS-1] amp_im[MAX_BANDS]
 * ─────────────────────────────────────────────────────────────────────────────
 */
layout(local_size_x = 64) in;

#define MAX_BANDS       32
#define INTENT_STRIDE   (20 + 2 * MAX_BANDS)   /* child region stride = 84 floats */
#define TERMINAL_STRIDE (26 + 2 * MAX_BANDS)   /* terminal record stride = 90 floats */

layout(std430, binding = 0) readonly buffer ChildIntBuf {
    float child_int_buf[];
};

/* R[res²] G[res²] B[res²] as float bit-cast uint32 — accumulated with CAS atomics */
layout(std430, binding = 1) coherent buffer SensorRGBBuf {
    uint sensor_rgb[];
};

/* ssbo_t3_meta bound here so nt_term never needs a CPU readback.
 * meta[1] = terminal_count, written by T3 atomicAdd. */
layout(std430, binding = 2) readonly buffer MetaBuf {
    uint splat_meta[];
};

/* Per-band display RGB weights: n_bands × 3 packed as vec3[32].
 * Index b: r = rgb_w[b].x, g = rgb_w[b].y, b_ = rgb_w[b].z */
uniform vec3  rgb_w[MAX_BANDS];

/* nt_term is now read from splat_meta[1] (ssbo_t3_meta binding 2).
 * The uniform slot is kept as -1 so existing callers that set it are harmless. */
uniform int   term_float_base;  /* max_children * INTENT_STRIDE (float offset)     */
uniform float sensor_half_w;    /* sensor plate half-width  (Y axis, metres)       */
uniform float sensor_half_h;    /* sensor plate half-height (Z axis, metres)       */
uniform int   sensor_res;       /* pixel grid side (sensor_res × sensor_res)       */
uniform int   n_bands;          /* number of active spectral bands (≤ MAX_BANDS)   */
uniform float splat_scale;      /* SENSOR_SPLAT_SCALE = 0.10                       */

/* Float atomic-add via CAS spin-loop — no GL_EXT_shader_atomic_float needed.
 * Same pattern as t5_full_connect.comp.glsl. */
void atomic_add_float(uint idx, float val) {
    uint expected = sensor_rgb[idx];
    for (int i = 0; i < 64; ++i) {
        float fexp    = uintBitsToFloat(expected);
        uint  desired = floatBitsToUint(fexp + val);
        uint  actual  = atomicCompSwap(sensor_rgb[idx], expected, desired);
        if (actual == expected) return;
        expected = actual;
    }
}

void main() {
    const int ti = int(gl_GlobalInvocationID.x);
    if (uint(ti) >= splat_meta[1]) return;  /* nt_term from ssbo_t3_meta[1] */

    /* Base offset into ChildIntBuf for this terminal record */
    const int base = term_float_base + ti * TERMINAL_STRIDE;

    /* Only accumulate backward sensor rays that hit an emissive surface */
    const uint color_flag  = floatBitsToUint(child_int_buf[base + 16]);
    const uint is_emissive = floatBitsToUint(child_int_buf[base + 25]);
    if (is_emissive == 0u || color_flag != 1u) return;

    /* Map sensor-space origin to pixel coordinates */
    const float soy   = child_int_buf[base + 23];
    const float soz   = child_int_buf[base + 24];
    const float inv_w = float(sensor_res) / (2.0f * sensor_half_w);
    const float inv_h = float(sensor_res) / (2.0f * sensor_half_h);
    const int   iy    = int((soy + sensor_half_w) * inv_w);
    const int   iz    = int((soz + sensor_half_h) * inv_h);
    if (iy < 0 || iy >= sensor_res || iz < 0 || iz >= sensor_res) return;

    /* Spectral → display RGB (matches band_to_display_rgb + SENSOR_SPLAT_SCALE) */
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    const int nb = min(n_bands, MAX_BANDS);
    for (int b = 0; b < nb; ++b) {
        const float re  = child_int_buf[base + 26 + b];
        const float im  = child_int_buf[base + 26 + MAX_BANDS + b];
        const float amp = sqrt(re * re + im * im);
        cr += amp * rgb_w[b].x;
        cg += amp * rgb_w[b].y;
        cb += amp * rgb_w[b].z;
    }
    cr *= splat_scale;
    cg *= splat_scale;
    cb *= splat_scale;
    if (cr + cg + cb <= 0.0f) return;

    /* Accumulate into global sensor buffer: R[res²] G[res²] B[res²] */
    const uint px  = uint(iy * sensor_res + iz);
    const uint pix = uint(sensor_res) * uint(sensor_res);
    atomic_add_float(px,          cr);
    atomic_add_float(px + pix,    cg);
    atomic_add_float(px + 2u*pix, cb);
}
