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
#define INTENT_STRIDE   (22 + 2 * MAX_BANDS)
#define TERMINAL_STRIDE (28 + 2 * MAX_BANDS)
#define CAMERA_PATH_SCATTERED_BIT 16u

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

layout(std430, binding = 3) readonly buffer MatBandBuf {
    float mat_bands[];
};

/* Adaptive traces accumulate all terminal contributions by primary sample.
 * A later finalize pass commits one complete spectrum (or zero for a miss) to
 * the node moments, so squared moments are never computed per terminal. */
layout(std430, binding = 4) coherent buffer SensorMipSampleSpectrumBuf {
    uint sensor_mip_sample_spectra[];
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
uniform int   n_mats;           /* material count for emission lookup              */
uniform float splat_scale;      /* direct visible-emitter scale; normally 1.0      */
uniform uint  sensor_mip_enabled;
uniform uint  sensor_mip_sample_capacity;

float mat_emission(int mat, int b, float frequency_hz) {
    if (mat < 0 || mat >= n_mats || b < 0 || b >= MAX_BANDS) return 1.0f;
    if (frequency_hz > 0.0) {
        int last = 0;
        while (last + 1 < MAX_BANDS
               && mat_bands[(mat * MAX_BANDS + last + 1) * 12] > 0.0) last++;
        if (last > 0) {
            bool inc = mat_bands[(mat * MAX_BANDS + last) * 12]
                    >= mat_bands[(mat * MAX_BANDS) * 12];
            int lo = 0;
            while (lo + 1 < last) {
                float nf = mat_bands[(mat * MAX_BANDS + lo + 1) * 12];
                if ((inc && nf >= frequency_hz) || (!inc && nf <= frequency_hz)) break;
                lo++;
            }
            int hi = min(last, lo + 1);
            int ao = (mat * MAX_BANDS + lo) * 12, bo = (mat * MAX_BANDS + hi) * 12;
            float den = mat_bands[bo] - mat_bands[ao];
            float t = abs(den) > 1e-20 ? clamp((frequency_hz - mat_bands[ao]) / den, 0.0, 1.0) : 0.0;
            return max(0.0, mix(mat_bands[ao + 5], mat_bands[bo + 5], t));
        }
    }
    return max(0.0f, mat_bands[(mat * MAX_BANDS + b) * 12 + 5]);
}

vec3 frequency_rgb(float frequency_hz) {
    float wl = 299792458.0 / max(frequency_hz, 1.0) * 1.0e9;
    vec3 c = vec3(0.0);
    if (wl >= 380.0 && wl < 440.0) c = vec3(-(wl-440.0)/60.0, 0.0, 1.0);
    else if (wl < 490.0) c = vec3(0.0, (wl-440.0)/50.0, 1.0);
    else if (wl < 510.0) c = vec3(0.0, 1.0, -(wl-510.0)/20.0);
    else if (wl < 580.0) c = vec3((wl-510.0)/70.0, 1.0, 0.0);
    else if (wl < 645.0) c = vec3(1.0, -(wl-645.0)/65.0, 0.0);
    else if (wl <= 700.0) c = vec3(1.0, 0.0, 0.0);
    float edge = wl < 420.0 ? 0.3 + 0.7*(wl-380.0)/40.0
               : (wl > 645.0 ? 0.3 + 0.7*(700.0-wl)/55.0 : 1.0);
    return max(c * clamp(edge, 0.0, 1.0), vec3(0.0));
}

/* Float atomic-add via CAS spin-loop — no GL_EXT_shader_atomic_float needed.
 * Same pattern as t5_full_connect.comp.glsl.  Unbounded on purpose: a fixed
 * retry cap silently lost energy on the brightest, highest-contention pixels.
 * compareAndSwap guarantees global progress, so the loop always terminates. */
void atomic_add_float(uint idx, float val) {
    if (val <= 0.0f || isnan(val) || isinf(val)) return;
    uint expected = sensor_rgb[idx];
    while (true) {
        float fexp    = uintBitsToFloat(expected);
        uint  desired = floatBitsToUint(fexp + val);
        uint  actual  = atomicCompSwap(sensor_rgb[idx], expected, desired);
        if (actual == expected) return;
        expected = actual;
    }
}

void atomic_add_sample_spectrum(uint idx, float val) {
    if (val <= 0.0f || isnan(val) || isinf(val)) return;
    uint expected = sensor_mip_sample_spectra[idx];
    while (true) {
        uint desired = floatBitsToUint(uintBitsToFloat(expected) + val);
        uint actual = atomicCompSwap(sensor_mip_sample_spectra[idx], expected, desired);
        if (actual == expected) return;
        expected = actual;
    }
}

void splat_sensor_tent(float soy, float soz, float cr, float cg, float cb) {
    if (cr + cg + cb <= 0.0f) return;

    const float inv_w = float(sensor_res) / (2.0f * sensor_half_w);
    const float inv_h = float(sensor_res) / (2.0f * sensor_half_h);
    const float fy = (soy + sensor_half_w) * inv_w - 0.5f;
    const float fz = (soz + sensor_half_h) * inv_h - 0.5f;
    const int y0 = int(floor(fy));
    const int z0 = int(floor(fz));

    float wsum = 0.0f;
    float ws[4];
    int ys[4];
    int zs[4];
    int k = 0;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dz = 0; dz <= 1; ++dz) {
            const int iy = y0 + dy;
            const int iz = z0 + dz;
            const float wy = max(0.0f, 1.0f - abs(float(iy) - fy));
            const float wz = max(0.0f, 1.0f - abs(float(iz) - fz));
            const float w = wy * wz;
            ys[k] = iy;
            zs[k] = iz;
            ws[k] = w;
            if (w > 0.0f && iy >= 0 && iy < sensor_res && iz >= 0 && iz < sensor_res)
                wsum += w;
            ++k;
        }
    }
    if (wsum <= 0.0f) return;

    const uint pix = uint(sensor_res) * uint(sensor_res);
    for (int i = 0; i < 4; ++i) {
        const int iy = ys[i];
        const int iz = zs[i];
        const float w = ws[i] / wsum;
        if (w <= 0.0f || iy < 0 || iy >= sensor_res || iz < 0 || iz >= sensor_res)
            continue;
        const uint px = uint(iy * sensor_res + iz);
        atomic_add_float(px,          cr * w);
        atomic_add_float(px + pix,    cg * w);
        atomic_add_float(px + 2u*pix, cb * w);
    }
}

void main() {
    const int ti = int(gl_GlobalInvocationID.x);
    if (uint(ti) >= splat_meta[1]) return;  /* nt_term from ssbo_t3_meta[1] */

    /* Base offset into ChildIntBuf for this terminal record */
    const int base = term_float_base + ti * TERMINAL_STRIDE;

    /* This strategy represents direct camera visibility through the camera
     * optics. Emitter hits after an authored scene-material scatter remain in
     * BDPT/VCM and must not enter this unpaired terminal splat. */
    const uint color_flag  = floatBitsToUint(child_int_buf[base + 16]);
    const uint is_emissive = floatBitsToUint(child_int_buf[base + 25]);
    if (is_emissive == 0u
        || ((color_flag & 1u) == 0u)
        || ((color_flag & CAMERA_PATH_SCATTERED_BIT) != 0u)) return;
    const int mat_id = floatBitsToInt(child_int_buf[base + 15]);
    const uint tag_lo = floatBitsToUint(child_int_buf[base + 21]);
    const uint tag_hi = floatBitsToUint(child_int_buf[base + 22]);
    const bool adaptive_sample = sensor_mip_enabled != 0u
        && (tag_hi & 0x40000000u) != 0u
        && tag_lo < sensor_mip_sample_capacity;

    /* Map sensor-space origin to film coordinates */
    const float soy   = child_int_buf[base + 23];
    const float soz   = child_int_buf[base + 24];

    /* Spectral → display RGB (matches band_to_display_rgb + SENSOR_SPLAT_SCALE) */
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    const int nb = min(n_bands, MAX_BANDS);
    const float spectral_frequency_hz = child_int_buf[base + 26];
    const float spectral_pdf = max(child_int_buf[base + 27], 1.0e-30);
    const vec3 exact_rgb = spectral_frequency_hz > 0.0
        ? frequency_rgb(spectral_frequency_hz) / spectral_pdf : vec3(0.0);
    for (int b = 0; b < nb; ++b) {
        const float re  = child_int_buf[base + 28 + b];
        const float im  = child_int_buf[base + 28 + MAX_BANDS + b];
        const float amp = sqrt(re * re + im * im) * mat_emission(mat_id, b, spectral_frequency_hz);
        if (adaptive_sample)
            atomic_add_sample_spectrum(tag_lo * uint(n_bands) + uint(b), amp * splat_scale);
        vec3 weight = spectral_frequency_hz > 0.0 ? exact_rgb : rgb_w[b];
        cr += amp * weight.x;
        cg += amp * weight.y;
        cb += amp * weight.z;
    }
    cr *= splat_scale;
    cg *= splat_scale;
    cb *= splat_scale;
    if (cr + cg + cb <= 0.0f) return;

    splat_sensor_tent(soy, soz, cr, cg, cb);
}
