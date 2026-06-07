#version 430 core
/*
 * t0_intent_pack.comp.glsl — GPU-side RayIntent → SSBO intent format conversion.
 *
 * Replaces the CPU packing loop in dispatch_t1_t2_t3 for generation 0.
 * The host uploads the raw RayIntent array (straight memcpy, no field extraction),
 * then dispatches this shader to convert each ray in parallel.
 *
 * Compile-time defines injected by the C++ preamble (all uint32-index based):
 *   SIZEOF_INTENT_U32     — sizeof(RayIntent) / 4
 *   OFF_POS_U32           — offsetof(pos) / 4  (3 doubles follow at +0,+2,+4)
 *   OFF_DIR_U32           — offsetof(dir) / 4  (3 doubles follow at +0,+2,+4)
 *   OFF_PATH_LEN_U32      — offsetof(path_len) / 4
 *   OFF_MED_MAT_U32       — offsetof(medium_mat_idx) / 4
 *   OFF_IFLAGS_U32        — offsetof(interaction_flags) / 4
 *   OFF_SRC_ID_U32        — offsetof(src_id) / 4
 *   OFF_BOUNCE_U32        — offsetof(bounce) / 4
 *   OFF_BLEFT_U32         — offsetof(bounces_left) / 4
 *   OFF_MIN_AMP_U32       — offsetof(min_amplitude) / 4
 *   OFF_TAG_U32           — offsetof(tag) / 4
 *   OFF_CFLAG_U32         — offsetof(color_flag) / 4 (byte extracted via shift)
 *   OFF_CFLAG_SHIFT       — (offsetof(color_flag) % 4) * 8
 *   OFF_PRIORITY_U32      — offsetof(priority) / 4
 *   OFF_SOY_U32           — offsetof(sensor_origin_y) / 4
 *   OFF_SOZ_U32           — offsetof(sensor_origin_z) / 4
 *   OFF_SUBPATH_U32       — offsetof(bdpt_subpath_id) / 4
 *   OFF_AMP_SCALAR_U32    — offsetof(amp_scalar) / 4
 *   OFF_AMP_NBANDS_U32    — offsetof(amp_n_bands) / 4
 *   OFF_AMP_NBANDS_SHIFT  — (offsetof(amp_n_bands) % 4) * 8
 *   INTENT_STRIDE         — 84 (20 + 2*MAX_SPECTRAL_BANDS)
 *   MAX_SPECTRAL_BANDS    — 32
 */
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

/* Output: the packed float SSBO that T1 reads (same ssbo_intent, binding 0) */
layout(std430, binding = 0) writeonly buffer IntentBuf { float intents[]; };
/* Input: raw RayIntent memory, uploaded via glBufferSubData without conversion */
layout(std430, binding = 1) readonly  buffer RawBuf    { uint  raw[];     };

uniform int u_n_rays;
uniform int u_n_bands;

/*
 * Convert IEEE 754 double (little-endian: lo=bits[31:0], hi=bits[63:32]) to float.
 * No fp64 extension required — operates entirely on bit patterns.
 * Handles normal numbers correctly; denormals below float range map to ±0.
 */
float d2f(uint lo, uint hi) {
    uint sign  = hi >> 31u;
    uint exp_d = (hi >> 20u) & 0x7FFu;
    /* Top 23 mantissa bits: top 20 from hi word, top 3 from lo word */
    uint mant  = ((hi & 0xFFFFFu) << 3u) | (lo >> 29u);

    if (exp_d == 0u)     return uintBitsToFloat(sign << 31u);             /* ±0 / denormal */
    if (exp_d == 0x7FFu) return uintBitsToFloat((sign << 31u) | 0x7F800000u); /* ±Inf / NaN */

    int e = int(exp_d) - 1023;
    if (e >  127) return uintBitsToFloat((sign << 31u) | 0x7F800000u);   /* overflow → ±Inf */
    if (e < -126) return uintBitsToFloat(sign << 31u);                    /* underflow → ±0  */

    return uintBitsToFloat((sign << 31u) | (uint(e + 127) << 23u) | mant);
}

void main() {
    uint ray_id = gl_GlobalInvocationID.x;
    if (int(ray_id) >= u_n_rays) return;

    uint rb = ray_id * uint(SIZEOF_INTENT_U32);   /* raw buffer base (uint32 index) */
    uint ob = ray_id * uint(INTENT_STRIDE);        /* output buffer base (float index) */

    /* pos — three doubles */
    intents[ob +  0u] = d2f(raw[rb + OFF_POS_U32 + 0u], raw[rb + OFF_POS_U32 + 1u]);
    intents[ob +  1u] = d2f(raw[rb + OFF_POS_U32 + 2u], raw[rb + OFF_POS_U32 + 3u]);
    intents[ob +  2u] = d2f(raw[rb + OFF_POS_U32 + 4u], raw[rb + OFF_POS_U32 + 5u]);

    /* dir — three doubles */
    intents[ob +  3u] = d2f(raw[rb + OFF_DIR_U32 + 0u], raw[rb + OFF_DIR_U32 + 1u]);
    intents[ob +  4u] = d2f(raw[rb + OFF_DIR_U32 + 2u], raw[rb + OFF_DIR_U32 + 3u]);
    intents[ob +  5u] = d2f(raw[rb + OFF_DIR_U32 + 4u], raw[rb + OFF_DIR_U32 + 5u]);

    /* path_len — double */
    intents[ob +  6u] = d2f(raw[rb + OFF_PATH_LEN_U32], raw[rb + OFF_PATH_LEN_U32 + 1u]);

    /* int/uint fields — bits preserved as-is (T1 reads back via floatBitsToInt/Uint) */
    intents[ob +  7u] = uintBitsToFloat(raw[rb + OFF_MED_MAT_U32]);
    intents[ob +  8u] = uintBitsToFloat(raw[rb + OFF_IFLAGS_U32]);
    intents[ob +  9u] = uintBitsToFloat(raw[rb + OFF_SRC_ID_U32]);
    intents[ob + 10u] = uintBitsToFloat(raw[rb + OFF_BOUNCE_U32]);
    intents[ob + 11u] = uintBitsToFloat(raw[rb + OFF_BLEFT_U32]);

    /* min_amplitude — double cast to float value */
    intents[ob + 12u] = d2f(raw[rb + OFF_MIN_AMP_U32], raw[rb + OFF_MIN_AMP_U32 + 1u]);

    /* tag — two uint32 halves */
    intents[ob + 13u] = uintBitsToFloat(raw[rb + OFF_TAG_U32]);
    intents[ob + 14u] = uintBitsToFloat(raw[rb + OFF_TAG_U32 + 1u]);

    /* color_flag — uint8_t: extract byte from containing uint32 */
    uint cflag = (raw[rb + uint(OFF_CFLAG_U32)] >> uint(OFF_CFLAG_SHIFT)) & 0xFFu;
    intents[ob + 15u] = uintBitsToFloat(cflag);

    /* priority, sensor_origin_y, sensor_origin_z — float, bits preserved */
    intents[ob + 16u] = uintBitsToFloat(raw[rb + OFF_PRIORITY_U32]);
    intents[ob + 17u] = uintBitsToFloat(raw[rb + OFF_SOY_U32]);
    intents[ob + 18u] = uintBitsToFloat(raw[rb + OFF_SOZ_U32]);

    /* bdpt_subpath_id — uint, bits preserved */
    intents[ob + 19u] = uintBitsToFloat(raw[rb + OFF_SUBPATH_U32]);

    /* Spectral amplitude */
    float amp_scalar = uintBitsToFloat(raw[rb + OFF_AMP_SCALAR_U32]);
    uint  nb_raw     = (raw[rb + uint(OFF_AMP_NBANDS_U32)] >> uint(OFF_AMP_NBANDS_SHIFT)) & 0xFFu;
    /* Sign-extend int8_t */
    int amp_n_bands  = (nb_raw >= 0x80u) ? int(nb_raw) - 256 : int(nb_raw);

    int bands = min(u_n_bands, int(MAX_SPECTRAL_BANDS));

    if (amp_n_bands > 0) {
        /* Scalar fast-path: uniform amplitude, zero imaginary */
        for (int b = 0; b < bands; ++b) {
            intents[ob + 20u + uint(b)]                          = amp_scalar;
            intents[ob + 20u + uint(MAX_SPECTRAL_BANDS) + uint(b)] = 0.0;
        }
    } else {
        /* Non-scalar: amp lives on the heap — CPU patches this ray's spectral
         * region via targeted glBufferSubData after this dispatch completes. */
        for (int b = 0; b < bands; ++b) {
            intents[ob + 20u + uint(b)]                          = 0.0;
            intents[ob + 20u + uint(MAX_SPECTRAL_BANDS) + uint(b)] = 0.0;
        }
    }
}
