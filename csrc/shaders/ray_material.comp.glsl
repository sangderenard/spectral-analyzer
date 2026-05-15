#version 430 core
/**
 * ray_material.comp.glsl  —  GPU T3: per-hit material / Fresnel / scatter stage
 *
 * Direct GLSL equivalent of pipeline_material_fresnel() in ray_tracer.cpp.
 * One invocation per RefinedHit.  For each hit the shader:
 *
 *   1. Classifies the triangle (aperture-stop, emissive, budget-dead, transmissive, opaque).
 *   2. Computes Snell refraction + Fresnel split (or diffuse/specular reflection).
 *   3. Writes up to max_children child RayIntents into out_intents[].
 *   4. Writes terminal HitRecords into out_terminals[].
 *   5. Optionally accumulates backward-ray hits onto the sensor image (ch1+ch2).
 *
 * Dispatch: glDispatchCompute(ceil(n_hits / 64), 1, 1)
 *
 * ── SSBO binding map ─────────────────────────────────────────────────────
 *
 *  binding  name              access    description
 *  -------  ----------------  --------  ------------------------------------
 *    0      RefinedHitBuf     readonly  flat RefinedHit records (see layout)
 *    1      OutIntentBuf      coherent  flat RayIntent records (children out)
 *    2      OutTerminalBuf    coherent  flat TerminalRecord records
 *    3      CounterBuf        coherent  {intent_count, terminal_count} (uint)
 *    4      TriangleBuf       readonly  per-triangle {flags,mat_idx,normal[3],_pad}
 *    5      MatBandBuf        readonly  mat_buf: (n_mats*n_bands, 12) float32
 *    6      EpsilonFlagBuf    readonly  per-material uint8 epsilon flags (as uint)
 *    7      SensorAccumBuf    coherent  sensor_accum: 4*res*res float64 → float32
 *
 * ── Per-record flat float32 layouts ─────────────────────────────────────
 *
 * RefinedHit  (REFINED_HIT_STRIDE = 26 + 2*MAX_BANDS floats):
 *   [0..2]   hit_pos xyz
 *   [3..5]   hit_n xyz
 *   [6..8]   incoming_dir xyz
 *   [9..11]  seg_start xyz
 *   [12]     path_len
 *   [13]     path_at_seg_start
 *   [14]     hit_tri          (intBitsToFloat)
 *   [15]     mat_idx          (intBitsToFloat)
 *   [16]     color_flag       (uintBitsToFloat)
 *   [17]     bounce           (intBitsToFloat)
 *   [18]     bounces_left     (intBitsToFloat)
 *   [19]     min_amplitude
 *   [20]     src_id           (intBitsToFloat)
 *   [21]     tag_lo           (uintBitsToFloat)
 *   [22]     tag_hi           (uintBitsToFloat)
 *   [23]     sensor_origin_y
 *   [24]     sensor_origin_z
 *   [25]     medium_mat_idx   (intBitsToFloat)
 *   [26 .. 26+MAX_BANDS-1]          amp_re[MAX_BANDS]
 *   [26+MAX_BANDS .. 26+2*MAX_BANDS-1] amp_im[MAX_BANDS]
 *
 * RayIntent  (INTENT_STRIDE = 20 + 2*MAX_BANDS floats):
 *   [0..2]   pos xyz
 *   [3..5]   dir xyz
 *   [6]      path_len
 *   [7]      medium_mat_idx   (intBitsToFloat)
 *   [8]      interaction_flags (uintBitsToFloat)
 *   [9]      src_id           (intBitsToFloat)
 *   [10]     bounce           (intBitsToFloat)
 *   [11]     bounces_left     (intBitsToFloat)
 *   [12]     min_amplitude
 *   [13]     tag_lo           (uintBitsToFloat)
 *   [14]     tag_hi           (uintBitsToFloat)
 *   [15]     color_flag       (uintBitsToFloat)
 *   [16]     priority
 *   [17]     sensor_origin_y
 *   [18]     sensor_origin_z
 *   [19]     _pad
 *   [20 .. 20+MAX_BANDS-1]          amp_re[MAX_BANDS]
 *   [20+MAX_BANDS .. 20+2*MAX_BANDS-1] amp_im[MAX_BANDS]
 *
 * TerminalRecord  (TERMINAL_STRIDE = 26 + 2*MAX_BANDS floats):
 *   Same layout as RefinedHit plus:
 *   [25] is_emissive_hit (uintBitsToFloat)
 *   (amp fields carry the amplitude at termination)
 *
 * Triangle  (TRI_STRIDE = 8 floats):
 *   [0]     flags            (intBitsToFloat)
 *   [1]     mat_idx          (intBitsToFloat)
 *   [2..4]  normal xyz
 *   [5..7]  _pad
 *
 * MatBand  (MAT_BAND_STRIDE = 12 floats per band per material,
 *           row = mat_idx * n_bands + band_idx):
 *   [0]  center_hz     [1]  bandwidth_hz   [2]  reflectance_mag  [3]  transmittance
 *   [4]  refl_re       [5]  refl_im        [6]  ior_real         [7]  ior_imag
 *   [8]  diffusion     [9]  atmo_abs       [10] extinction_k     [11] reserved
 *
 * SensorAccumBuf: 4 * sensor_res * sensor_res float32 values.
 *   channel 0: forward plate hits      offset = 0
 *   channel 1: backward emissive hits  offset = res*res
 *   channel 2: raw amp_mag physics     offset = 2*res*res
 *   channel 3: near-miss provisional   offset = 3*res*res
 */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

/* ── Compile-time constants ─────────────────────────────────────────────── */
#define MAX_BANDS        16
#define REFINED_HIT_STRIDE  (26 + 2 * MAX_BANDS)
#define INTENT_STRIDE       (20 + 2 * MAX_BANDS)
#define TERMINAL_STRIDE     (26 + 2 * MAX_BANDS)
#define TRI_STRIDE          8
#define MAT_BAND_STRIDE     12

/* ── Material flags (must match mat_flags_generated.h) ─────────────────── */
#define MAT_FLAG_APERTURE_STOP  (1 << 0)
#define MAT_FLAG_EMISSIVE       (1 << 1)
#define MAT_FLAG_REACTIVE       (1 << 2)
#define MAT_FLAG_TRANSMISSIVE   (1 << 4)

/* ── SSBOs ──────────────────────────────────────────────────────────────── */
layout(std430, binding = 0) readonly buffer RefinedHitBuf  { float hits[];      };
layout(std430, binding = 1) coherent buffer OutIntentBuf   { float intents[];   };
layout(std430, binding = 2) coherent buffer OutTerminalBuf { float terminals[]; };
layout(std430, binding = 3) coherent buffer CounterBuf     { uint counters[];   }; /* [0]=intent_count [1]=terminal_count */
layout(std430, binding = 4) readonly buffer TriangleBuf    { float tris[];      };
layout(std430, binding = 5) readonly buffer MatBandBuf     { float mat_bands[]; };
layout(std430, binding = 6) readonly buffer EpsilonFlagBuf { uint  eps_flags[]; };
/* SensorAccumBuf removed — all sensor writes go through the image path below */

/* ── Uniforms ────────────────────────────────────────────────────────────── */
uniform int   n_hits;
uniform int   n_bands;
uniform int   n_mats;
uniform int   max_children;       /* 1 or 2 */
uniform int   sensor_res;         /* 0 = sensor disabled */
uniform float sensor_pr;          /* sensor half-radius in metres */
uniform float sensor_px;          /* sensor plate x position */
uniform uint  rng_seed;           /* per-dispatch seed */

/* ── Hash-based PRNG (xorshift32 + Weyl) ────────────────────────────────── */
/* Returns a pseudo-random float in [0, 1) given a mutable state. */
float rand_next(inout uint s) {
    s ^= s << 13u;
    s ^= s >> 17u;
    s ^= s << 5u;
    return float(s) * (1.0 / 4294967296.0);
}

uint rng_init(uint idx) {
    /* Wang hash to de-correlate across invocations */
    uint s = idx ^ rng_seed;
    s = (s ^ 61u) ^ (s >> 16u);
    s *= 9u;
    s ^= s >> 4u;
    s *= 0x27d4eb2du;
    s ^= s >> 15u;
    return (s == 0u) ? 1u : s;
}

/* ── Cosine-weighted hemisphere direction ────────────────────────────────── */
vec3 cosine_hemisphere(vec3 n, inout uint rng) {
    float u1 = rand_next(rng);
    float u2 = rand_next(rng);
    float r   = sqrt(u1);
    float phi = 6.28318530718 * u2;
    float x   = r * cos(phi);
    float y   = r * sin(phi);
    float z   = sqrt(max(0.0, 1.0 - u1));

    /* Build ONB around n */
    vec3 up  = abs(n.z) < 0.9999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 t   = normalize(cross(up, n));
    vec3 b   = cross(n, t);
    return normalize(t * x + b * y + n * z);
}

/* ── Snell refraction ────────────────────────────────────────────────────── */
/* Returns false on total internal reflection (writes no refracted dir). */
bool snell_refract(vec3 dir, vec3 n, float n1, float n2, out vec3 refracted) {
    float cos_i = max(0.0, -dot(dir, n));
    float sin2_t = (n1 / n2) * (n1 / n2) * (1.0 - cos_i * cos_i);
    if (sin2_t > 1.0) { refracted = vec3(0); return false; }
    float cos_t = sqrt(max(0.0, 1.0 - sin2_t));
    refracted = normalize((n1 / n2) * dir + ((n1 / n2) * cos_i - cos_t) * n);
    return true;
}

/* ── Fresnel reflectance (unpolarised) ───────────────────────────────────── */
float fresnel_R(float cos_i, float cos_t, float n1, float n2) {
    float rs_num = n1 * cos_i - n2 * cos_t;
    float rs_den = n1 * cos_i + n2 * cos_t;
    float rp_num = n1 * cos_t - n2 * cos_i;
    float rp_den = n1 * cos_t + n2 * cos_i;
    float Rs = (abs(rs_den) < 1e-9) ? 1.0 : (rs_num / rs_den) * (rs_num / rs_den);
    float Rp = (abs(rp_den) < 1e-9) ? 1.0 : (rp_num / rp_den) * (rp_num / rp_den);
    return clamp(0.5 * (Rs + Rp), 0.0, 1.0);
}

/* ── Material accessors ──────────────────────────────────────────────────── */
float mat_refl_re  (int mat, int b) { return mat_bands[(mat * n_bands + b) * MAT_BAND_STRIDE + 4]; }
float mat_refl_im  (int mat, int b) { return mat_bands[(mat * n_bands + b) * MAT_BAND_STRIDE + 5]; }
float mat_ior_real (int mat, int b) { return mat_bands[(mat * n_bands + b) * MAT_BAND_STRIDE + 6]; }
float mat_diffusion(int mat)        { return mat_bands[(mat * n_bands + 0) * MAT_BAND_STRIDE + 8]; }

/* ── RefinedHit field accessors ──────────────────────────────────────────── */
#define HIT(base, off)  hits[(base) + (off)]
vec3 hit_pos    (uint b) { return vec3(HIT(b,0), HIT(b,1), HIT(b,2)); }
vec3 hit_n      (uint b) { return vec3(HIT(b,3), HIT(b,4), HIT(b,5)); }
vec3 hit_dir    (uint b) { return vec3(HIT(b,6), HIT(b,7), HIT(b,8)); }
vec3 hit_segst  (uint b) { return vec3(HIT(b,9), HIT(b,10),HIT(b,11)); }
float hit_pathlen    (uint b) { return HIT(b,12); }
int   hit_tri        (uint b) { return floatBitsToInt(HIT(b,14)); }
int   hit_mat        (uint b) { return floatBitsToInt(HIT(b,15)); }
uint  hit_colorflag  (uint b) { return floatBitsToUint(HIT(b,16)); }
int   hit_bounce     (uint b) { return floatBitsToInt(HIT(b,17)); }
int   hit_bounceleft (uint b) { return floatBitsToInt(HIT(b,18)); }
float hit_minamp     (uint b) { return HIT(b,19); }
int   hit_srcid      (uint b) { return floatBitsToInt(HIT(b,20)); }
uint  hit_taglo      (uint b) { return floatBitsToUint(HIT(b,21)); }
uint  hit_taghi      (uint b) { return floatBitsToUint(HIT(b,22)); }
float hit_soy        (uint b) { return HIT(b,23); }
float hit_soz        (uint b) { return HIT(b,24); }
int   hit_medium     (uint b) { return floatBitsToInt(HIT(b,25)); }
float hit_amp_re(uint b, int band) { return HIT(b, 26 + band); }
float hit_amp_im(uint b, int band) { return HIT(b, 26 + MAX_BANDS + band); }

/* ── Triangle accessors ──────────────────────────────────────────────────── */
int   tri_flags  (int t) { return floatBitsToInt(tris[t * TRI_STRIDE + 0]); }
int   tri_mat    (int t) { return floatBitsToInt(tris[t * TRI_STRIDE + 1]); }
vec3  tri_normal (int t) { return vec3(tris[t*TRI_STRIDE+2], tris[t*TRI_STRIDE+3], tris[t*TRI_STRIDE+4]); }

/* ── Write helpers ───────────────────────────────────────────────────────── */

/* Copy RefinedHit header fields into an output terminal record slot. */
void write_terminal(uint slot, uint hbase, bool is_emissive) {
    uint tbase = slot * uint(TERMINAL_STRIDE);
    for (int i = 0; i < REFINED_HIT_STRIDE; i++)
        terminals[tbase + i] = hits[hbase + i];
    /* Overwrite field [25] with is_emissive_hit flag */
    terminals[tbase + 25] = uintBitsToFloat(is_emissive ? 1u : 0u);
}

/* Write a child RayIntent into the output intent buffer. */
void write_intent(uint slot,
                  vec3 pos, vec3 dir, float path_len,
                  int medium, uint iflags,
                  int src_id, int bounce, int bounces_left, float min_amp,
                  uint tag_lo, uint tag_hi, uint color_flag, float priority,
                  float soy, float soz,
                  float amp_re[MAX_BANDS], float amp_im[MAX_BANDS])
{
    uint ibase = slot * uint(INTENT_STRIDE);
    intents[ibase +  0] = pos.x;
    intents[ibase +  1] = pos.y;
    intents[ibase +  2] = pos.z;
    intents[ibase +  3] = dir.x;
    intents[ibase +  4] = dir.y;
    intents[ibase +  5] = dir.z;
    intents[ibase +  6] = path_len;
    intents[ibase +  7] = intBitsToFloat(medium);
    intents[ibase +  8] = uintBitsToFloat(iflags);
    intents[ibase +  9] = intBitsToFloat(src_id);
    intents[ibase + 10] = intBitsToFloat(bounce);
    intents[ibase + 11] = intBitsToFloat(bounces_left);
    intents[ibase + 12] = min_amp;
    intents[ibase + 13] = uintBitsToFloat(tag_lo);
    intents[ibase + 14] = uintBitsToFloat(tag_hi);
    intents[ibase + 15] = uintBitsToFloat(color_flag);
    intents[ibase + 16] = priority;
    intents[ibase + 17] = soy;
    intents[ibase + 18] = soz;
    intents[ibase + 19] = 0.0;  /* pad */
    for (int b = 0; b < MAX_BANDS; b++) {
        intents[ibase + 20 +           b] = amp_re[b];
        intents[ibase + 20 + MAX_BANDS + b] = amp_im[b];
    }
}

/* sensor_atomic_add(uint, float) removed — SensorAccumBuf SSBO was removed;
 * all sensor writes go through sensor_atomic_add_img() below. */

/* ── Sensor image (r32ui, 4 layers: ch0..ch3) ───────────────────────────── */
/* For correct concurrent writes the sensor is exposed as a uimage2DArray so
 * we can use imageAtomicCompSwap, matching coherent_accumulate.comp.glsl. */
layout(r32ui, binding = 7) coherent volatile uniform uimage2DArray sensor_image;
/* layer 0 = ch1 (backward emissive exposure-compensated)
 * layer 1 = ch2 (raw amp_mag physics) */

void sensor_atomic_add_img(ivec2 coord, int layer, float value) {
    ivec3 c = ivec3(coord, layer);
    uint assumed, old_val;
    old_val = imageLoad(sensor_image, c).r;
    for (int guard = 0; guard < 64; guard++) {
        assumed = old_val;
        uint new_bits = floatBitsToUint(uintBitsToFloat(assumed) + value);
        old_val = imageAtomicCompSwap(sensor_image, c, assumed, new_bits);
        if (old_val == assumed) break;
    }
}

/* ── Main ────────────────────────────────────────────────────────────────── */
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= n_hits) return;

    uint hbase = gid * uint(REFINED_HIT_STRIDE);
    uint rng   = rng_init(gid);

    /* Read hit fields */
    vec3  pos       = hit_pos(hbase);
    vec3  nrm       = normalize(hit_n(hbase));
    vec3  in_dir    = normalize(hit_dir(hbase));
    float path_len  = hit_pathlen(hbase);
    int   tri_idx   = hit_tri(hbase);
    int   mat_id    = hit_mat(hbase);
    uint  cflag     = hit_colorflag(hbase);
    int   bounce    = hit_bounce(hbase);
    int   bleft     = hit_bounceleft(hbase);
    float min_amp   = hit_minamp(hbase);
    int   src_id    = hit_srcid(hbase);
    uint  tag_lo    = hit_taglo(hbase);
    uint  tag_hi    = hit_taghi(hbase);
    float soy       = hit_soy(hbase);
    float soz       = hit_soz(hbase);
    int   medium    = hit_medium(hbase);

    /* Read amplitude for this hit */
    float amp_re[MAX_BANDS];
    float amp_im[MAX_BANDS];
    int nb = min(n_bands, MAX_BANDS);
    for (int b = 0; b < nb; b++) {
        amp_re[b] = hit_amp_re(hbase, b);
        amp_im[b] = hit_amp_im(hbase, b);
    }
    for (int b = nb; b < MAX_BANDS; b++) { amp_re[b] = 0.0; amp_im[b] = 0.0; }

    /* Read triangle flags */
    int flags   = tri_flags(tri_idx);
    bool front_face = dot(in_dir, nrm) < 0.0;

    /* ── Terminal: aperture stop ── */
    if ((flags & MAT_FLAG_APERTURE_STOP) != 0) {
        uint tslot = atomicAdd(counters[1], 1u);
        write_terminal(tslot, hbase, false);
        return;
    }

    /* ── Terminal: emissive ── */
    if ((flags & MAT_FLAG_EMISSIVE) != 0) {
        uint tslot = atomicAdd(counters[1], 1u);
        write_terminal(tslot, hbase, true);

        /* Sensor accumulation for backward rays */
        if (sensor_res > 0 && cflag == 1u) {
            float inv_r = float(sensor_res) / (2.0 * sensor_pr);
            int   iy    = int((soy + sensor_pr) * inv_r);
            int   iz    = int((soz + sensor_pr) * inv_r);
            if (iy >= 0 && iy < sensor_res && iz >= 0 && iz < sensor_res) {
                float amp_mag = 0.0;
                for (int b = 0; b < nb; b++) {
                    float re = amp_re[b], im = amp_im[b];
                    amp_mag += sqrt(re*re + im*im);
                }
                float n_bands_f = float(nb > 0 ? nb : 1);
                float transmission = amp_mag / n_bands_f;
                float weight = (transmission > 1e-12) ? (1.0 / transmission) : n_bands_f;

                sensor_atomic_add_img(ivec2(iy, iz), 0, weight);   /* ch1 */
                sensor_atomic_add_img(ivec2(iy, iz), 1, amp_mag);  /* ch2 */
            }
        }
        return;
    }

    /* ── Terminal: budget exhausted ── */
    if (bleft <= 0) {
        uint tslot = atomicAdd(counters[1], 1u);
        write_terminal(tslot, hbase, false);
        return;
    }

    /* ── Epsilon fast-path: fully absorptive opaque material ── */
    bool is_transmissive = (flags & MAT_FLAG_TRANSMISSIVE) != 0;
    if (!is_transmissive && mat_id >= 0 && mat_id < int(eps_flags.length())) {
        if (eps_flags[mat_id] != 0u) {
            uint tslot = atomicAdd(counters[1], 1u);
            write_terminal(tslot, hbase, false);
            return;
        }
    }

    int   new_bounce    = bounce + 1;
    int   new_bleft     = bleft - 1;
    uint  iflags        = 0u;

    if (is_transmissive) {
        /* ── Transmissive: compute n1, n2 from medium and material IOR ── */
        float n1 = 1.0, n2 = 1.0;
        if (medium >= 0 && medium < n_mats)
            n1 = mat_ior_real(medium, 0);
        if (front_face)
            n2 = (mat_id >= 0) ? mat_ior_real(mat_id, 0) : 1.0;
        else
            n2 = 1.0;   /* exiting into air */
        if (n1 <= 0.0) n1 = 1.0;
        if (n2 <= 0.0) n2 = 1.0;

        float cos_i  = max(0.0, -dot(in_dir, nrm));
        vec3  refracted;
        bool  can_refract = snell_refract(in_dir, nrm, n1, n2, refracted);

        if (!can_refract) {
            /* TIR — pure specular reflection, apply reflectance cache */
            vec3 rd = normalize(in_dir - 2.0 * dot(in_dir, nrm) * nrm);
            float ra_re[MAX_BANDS], ra_im[MAX_BANDS];
            for (int b = 0; b < nb; b++) {
                float rr = (mat_id >= 0) ? mat_refl_re(mat_id, b) : 1.0;
                float ri = (mat_id >= 0) ? mat_refl_im(mat_id, b) : 0.0;
                /* complex multiply amp * refl */
                ra_re[b] = amp_re[b]*rr - amp_im[b]*ri;
                ra_im[b] = amp_re[b]*ri + amp_im[b]*rr;
            }
            for (int b = nb; b < MAX_BANDS; b++) { ra_re[b]=0.0; ra_im[b]=0.0; }

            uint islot = atomicAdd(counters[0], 1u);
            int new_med = front_face ? (mat_id) : -1;
            write_intent(islot, pos, rd, path_len, new_med, iflags,
                         src_id, new_bounce, new_bleft, min_amp,
                         tag_lo, tag_hi, cflag, 1.0, soy, soz, ra_re, ra_im);
        } else {
            float sin2_t = (n1/n2)*(n1/n2)*(1.0 - cos_i*cos_i);
            float cos_t  = sqrt(max(0.0, 1.0 - sin2_t));
            float R      = fresnel_R(cos_i, cos_t, n1, n2);
            int   new_med = front_face ? mat_id : -1;

            if (max_children >= 2) {
                /* Spawn both reflected and refracted children */
                float rs = sqrt(R);
                float ts = sqrt(max(0.0, 1.0 - R));

                /* Reflected child */
                {
                    vec3  rd = normalize(in_dir - 2.0 * dot(in_dir, nrm) * nrm);
                    float ra_re[MAX_BANDS], ra_im[MAX_BANDS];
                    for (int b = 0; b < nb; b++) {
                        float rr = (mat_id >= 0) ? mat_refl_re(mat_id, b) : 1.0;
                        float ri = (mat_id >= 0) ? mat_refl_im(mat_id, b) : 0.0;
                        float scaled_re = rs * (amp_re[b]*rr - amp_im[b]*ri);
                        float scaled_im = rs * (amp_re[b]*ri + amp_im[b]*rr);
                        ra_re[b] = scaled_re;
                        ra_im[b] = scaled_im;
                    }
                    for (int b = nb; b < MAX_BANDS; b++) { ra_re[b]=0.0; ra_im[b]=0.0; }
                    uint islot = atomicAdd(counters[0], 1u);
                    write_intent(islot, pos, rd, path_len, medium, iflags,
                                 src_id, new_bounce, new_bleft, min_amp,
                                 tag_lo, tag_hi, cflag, 1.0, soy, soz, ra_re, ra_im);
                }
                /* Refracted child */
                {
                    float ta_re[MAX_BANDS], ta_im[MAX_BANDS];
                    for (int b = 0; b < nb; b++) {
                        ta_re[b] = ts * amp_re[b];
                        ta_im[b] = ts * amp_im[b];
                    }
                    for (int b = nb; b < MAX_BANDS; b++) { ta_re[b]=0.0; ta_im[b]=0.0; }
                    uint islot = atomicAdd(counters[0], 1u);
                    write_intent(islot, pos, refracted, path_len, new_med, iflags,
                                 src_id, new_bounce, new_bleft, min_amp,
                                 tag_lo, tag_hi, cflag, 1.0, soy, soz, ta_re, ta_im);
                }
            } else {
                /* Stochastic: pick reflect or refract */
                float u = rand_next(rng);
                float ca_re[MAX_BANDS], ca_im[MAX_BANDS];
                vec3  cdir;
                int   cmed;

                if (u < R) {
                    cdir = normalize(in_dir - 2.0 * dot(in_dir, nrm) * nrm);
                    cmed = medium;
                    for (int b = 0; b < nb; b++) {
                        float rr = (mat_id >= 0) ? mat_refl_re(mat_id, b) : 1.0;
                        float ri = (mat_id >= 0) ? mat_refl_im(mat_id, b) : 0.0;
                        ca_re[b] = amp_re[b]*rr - amp_im[b]*ri;
                        ca_im[b] = amp_re[b]*ri + amp_im[b]*rr;
                    }
                } else {
                    cdir = refracted;
                    cmed = new_med;
                    for (int b = 0; b < nb; b++) { ca_re[b]=amp_re[b]; ca_im[b]=amp_im[b]; }
                }
                for (int b = nb; b < MAX_BANDS; b++) { ca_re[b]=0.0; ca_im[b]=0.0; }
                uint islot = atomicAdd(counters[0], 1u);
                write_intent(islot, pos, cdir, path_len, cmed, iflags,
                             src_id, new_bounce, new_bleft, min_amp,
                             tag_lo, tag_hi, cflag, 1.0, soy, soz, ca_re, ca_im);
            }
        }
    } else {
        /* ── Opaque: diffuse or specular reflection ── */
        float diffusion = (mat_id >= 0) ? mat_diffusion(mat_id) : 0.0;
        vec3 new_dir;
        if (rand_next(rng) < diffusion) {
            new_dir = cosine_hemisphere(nrm, rng);
        } else {
            new_dir = normalize(in_dir - 2.0 * dot(in_dir, nrm) * nrm);
            if (dot(new_dir, nrm) < 0.0)
                new_dir = cosine_hemisphere(nrm, rng);
        }

        float na_re[MAX_BANDS], na_im[MAX_BANDS];
        float max_abs = 0.0;
        for (int b = 0; b < nb; b++) {
            float rr = (mat_id >= 0) ? mat_refl_re(mat_id, b) : 1.0;
            float ri = (mat_id >= 0) ? mat_refl_im(mat_id, b) : 0.0;
            na_re[b] = amp_re[b]*rr - amp_im[b]*ri;
            na_im[b] = amp_re[b]*ri + amp_im[b]*rr;
            float mag = sqrt(na_re[b]*na_re[b] + na_im[b]*na_im[b]);
            max_abs = max(max_abs, mag);
        }
        for (int b = nb; b < MAX_BANDS; b++) { na_re[b]=0.0; na_im[b]=0.0; }

        if (max_abs >= min_amp) {
            uint islot = atomicAdd(counters[0], 1u);
            write_intent(islot, pos, new_dir, path_len, medium, iflags,
                         src_id, new_bounce, new_bleft, min_amp,
                         tag_lo, tag_hi, cflag, 1.0, soy, soz, na_re, na_im);
        } else {
            /* Amplitude extinguished — terminal */
            uint tslot = atomicAdd(counters[1], 1u);
            write_terminal(tslot, hbase, false);
        }
    }
}
