#version 430 core
/**
 * ray_material.comp.glsl  —  GPU T3: per-hit material / Fresnel / scatter stage
 *
 * Direct GLSL equivalent of pipeline_material_fresnel() in ray_tracer.cpp.
 * One invocation per RefinedHit.  For each hit the shader:
 *
 *   1. Classifies the triangle (aperture-stop, emissive, budget-dead, transmissive, opaque).
 *   2. Computes Snell refraction + Fresnel split (or diffuse/specular reflection).
 *   3. Writes up to max_children child RayIntents into the first section of out_buf[].
 *   4. Writes terminal HitRecords into the tail section of out_buf[] past max_children*INTENT_STRIDE.
 *
 * Sensor accumulation is performed on the CPU from the terminal readback
 * (sensor_res kept at 0 here; the image unit at binding 7 is declared but
 * not used so the binding slot stays reserved for future use).
 *
 * ── SSBO binding map ─────────────────────────────────────────────────────
 *
 *  binding  name              access    description
 *  -------  ----------------  --------  ------------------------------------
 *    0      RefinedHitBuf     readonly  flat RefinedHit records
 *    1      OutBuf            coherent  child RayIntents [0..max_children*INTENT_STRIDE)
 *                                       + terminal records [max_children*INTENT_STRIDE..)
 *    2      MetaBuf           coherent  [0]=intent_count [1]=terminal_count
 *                                       [2..2+n_mats-1] = per-mat epsilon flags
 *    3      TriangleBuf       readonly  full triangle geometry (TRI_FULL_STRIDE=16 floats)
 *                                         [12]=flags [13]=mat_idx [9..11]=normal
 *    4      MatBandBuf        readonly  mat_buf: (n_mats*n_bands, 12) float32
 *
 * ── Per-record flat float32 layouts ─────────────────────────────────────
 *
 * RefinedHit  (REFINED_HIT_STRIDE = 27 + 2*MAX_BANDS = 59 floats):
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
 *   [58]     bdpt_subpath_id  (uintBitsToFloat; 0=untracked)
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
 *   [25] is_emissive_hit (uintBitsToFloat)   ← overwrites medium_mat_idx
 *   (amp fields carry the amplitude at termination)
 *
 * Triangle full geometry  (TRI_FULL_STRIDE = 16 floats, same as T1):
 *   [0..2]   v0 xyz
 *   [3..5]   edge1 xyz
 *   [6..8]   edge2 xyz
 *   [9..11]  normal xyz
 *   [12]     flags            (intBitsToFloat)
 *   [13]     mat_idx          (intBitsToFloat)
 *   [14]     medium_pos_mat_idx
 *   [15]     medium_neg_mat_idx
 *
 * MatBand  (MAT_BAND_STRIDE = 12 floats per band per material,
 *           row = mat_idx * MAT_FULL_BANDS + band_idx):
 *   [0]  center_hz     [1]  bandwidth_hz   [2]  reflectance_mag  [3]  transmittance
 *   [4]  diffuse_frac  [5]  emission       [6]  reemission       [7]  ior_real
 *   [8]  ior_imag      [9] reactive_shift_hz [10] ggx_alpha [11] metallic
 */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

/* ── Compile-time constants ─────────────────────────────────────────────── */
#define MAX_BANDS           32
#define REFINED_HIT_STRIDE  (27 + 2 * MAX_BANDS)   /* 59: [58]=bdpt_sid at end */
#define INTENT_STRIDE       (20 + 2 * MAX_BANDS)   /* 52 */
#define TERMINAL_STRIDE     (26 + 2 * MAX_BANDS)   /* 58: terminals don't carry bdpt_sid */
#define TRI_FULL_STRIDE     16
#define MAT_BAND_STRIDE     12
#define MAT_FULL_BANDS      32
#define RAY_ORIGIN_EPS      2.0e-4

/* ── Material flags (must match mat_flags_generated.h) ─────────────────── */
#define MAT_FLAG_EMISSIVE         1u
#define MAT_FLAG_REACTIVE         2u
#define MAT_FLAG_ABSORBER         4u
#define MAT_FLAG_NO_SHADOW        8u
#define MAT_FLAG_MANIFOLD        16u
#define MAT_FLAG_PARAMETRIC      32u
#define MAT_FLAG_TRANSMISSIVE    64u
#define MAT_FLAG_APERTURE_STOP  128u
#define MAT_FLAG_PICKING_ONLY   256u

/* ── SSBOs (8 total — max per GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS) ──────────
 *
 *  binding  name              access    description
 *  -------  ----------------  --------  ------------------------------------
 *    0      RefinedHitBuf     readonly  flat RefinedHit records (stride 59)
 *                                         hit[58] = bdpt_subpath_id (uintBitsToFloat)
 *    1      OutBuf            coherent  child intents [0..max_children*INTENT_STRIDE)
 *                                       + terminal records [max_children*INTENT_STRIDE..)
 *    2      MetaBuf           coherent  [0]=intent_count [1]=terminal_count
 *                                       [2..2+n_mats-1]=per-mat epsilon flags
 *                                       [bdpt_count_base+0..+2]=bdpt vert/spec/pdf counts
 *    3      TriangleBuf       readonly  full triangle geometry (16 floats each)
 *    4      MatBandBuf        readonly  mat_buf: (n_mats*n_bands, 12) float32
 *    5      MergedUvBuf       readonly  int32 array:
 *                                         [0..uv_group_id_base): UV coords as intBitsToFloat
 *                                         [uv_group_id_base..uv_meta_base): per-tri group IDs
 *                                         [uv_meta_base..): group meta (res, accum_offset pairs)
 *    6      UvAccumBuf        coherent  flat uint32 UV accumulator
 *    7      BdptOutputBuf     coherent  float32: verts at [0], spectral at [bdpt_spectral_base],
 *                                         pdfs at [bdpt_pdf_base]
 */
layout(std430, binding = 0) readonly buffer RefinedHitBuf { float hits[];      };
layout(std430, binding = 1) coherent buffer OutBuf        { float out_buf[];   };
layout(std430, binding = 2) coherent buffer MetaBuf       { uint  meta[];      };
layout(std430, binding = 3) readonly buffer TriangleBuf   { float tris[];      };
layout(std430, binding = 4) readonly buffer MatBandBuf    { float mat_bands[]; };
layout(std430, binding = 5) readonly buffer MergedUvBuf   { int   merged_uv[]; };
layout(std430, binding = 6) coherent buffer UvAccumBuf    { uint  uv_accum[];  };
layout(std430, binding = 7) coherent buffer BdptOutputBuf { float bdpt_out[];  };

/* image2DArray sensor — image unit binding 7 is a separate namespace from SSBO binding 7 */
layout(r32ui, binding = 7) coherent volatile uniform uimage2DArray sensor_image;

/* ── Uniforms ────────────────────────────────────────────────────────────── */
uniform int   n_hits;             /* exact hit count from T1 (replaces CounterBuf read) */
uniform int   n_bands;
uniform int   n_mats;
uniform int   max_children;
uniform int   max_children_per_hit;
uniform int   sensor_res;
uniform float sensor_pr;
uniform float sensor_px;
uniform int   n_uv_groups;
uniform int   uv_group_id_base;   /* int index in merged_uv[] where per-tri group IDs start (= n_tris*6) */
uniform int   uv_meta_base;       /* int index in merged_uv[] where group metadata starts (= n_tris*7) */
uniform uint  rng_seed;
uniform int   bdpt_max_verts;     /* cap for vertex records in BdptOutputBuf; 0 = BDPT disabled */
uniform int   bdpt_max_spectral;  /* cap for spectral records */
uniform int   bdpt_max_pdfs;      /* cap for PDF records */
uniform int   bdpt_count_base;    /* uint index in MetaBuf where BDPT counts live (= 2 + n_mats) */
uniform int   bdpt_spectral_base; /* float index in bdpt_out[] where spectral records start */
uniform int   bdpt_pdf_base;      /* float index in bdpt_out[] where PDF records start */

/* ── Hash-based PRNG (xorshift32 + Weyl) ────────────────────────────────── */
float rand_next(inout uint s) {
    s ^= s << 13u;
    s ^= s >> 17u;
    s ^= s << 5u;
    return float(s) * (1.0 / 4294967296.0);
}

uint rng_init(uint idx) {
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
    vec3 up  = abs(n.z) < 0.9999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 t   = normalize(cross(up, n));
    vec3 b   = cross(n, t);
    return normalize(t * x + b * y + n * z);
}

/* ── Snell refraction ────────────────────────────────────────────────────── */
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
int mat_band_off(int mat, int b) {
    int mm = clamp(mat, 0, max(n_mats - 1, 0));
    int bb = clamp(b, 0, MAT_FULL_BANDS - 1);
    return (mm * MAT_FULL_BANDS + bb) * MAT_BAND_STRIDE;
}

float mat_band_field(int mat, int b, int field, float fallback) {
    if (mat < 0 || mat >= n_mats) return fallback;
    return mat_bands[mat_band_off(mat, b) + field];
}

vec2 mat_refl_complex(int mat, int b) {
    float mag = max(0.0, mat_band_field(mat, b, 2, 0.0));
    float nr  = mat_band_field(mat, b, 7, 1.0);
    float ni  = mat_band_field(mat, b, 8, 0.0);

    /* CPU equivalent:
     *   r_F = (1 - (nr+i*ni)) / (1 + (nr+i*ni))
     *   reflectance = authored_mag * exp(i * arg(r_F))
     */
    float ar = 1.0 - nr;
    float ai = -ni;
    float br = 1.0 + nr;
    float bi = ni;
    float den = max(br * br + bi * bi, 1.0e-20);
    float rr = (ar * br + ai * bi) / den;
    float ri = (ai * br - ar * bi) / den;
    float len = sqrt(rr * rr + ri * ri);
    if (len <= 1.0e-12) return vec2(mag, 0.0);
    return mag * vec2(rr, ri) / len;
}

float mat_refl_re  (int mat, int b) { return mat_refl_complex(mat, b).x; }
float mat_refl_im  (int mat, int b) { return mat_refl_complex(mat, b).y; }
float mat_ior_real       (int mat, int b) { return mat_band_field(mat, b, 7, 1.0); }
float mat_transmittance  (int mat, int b) { return mat_band_field(mat, b, 3, 0.0); }
float mat_diffusion(int mat)              { return clamp(mat_band_field(mat, 0, 4, 0.0), 0.0, 1.0); }
float mat_ggx_alpha(int mat)              { return clamp(mat_band_field(mat, 0, 10, 0.0), 0.0, 1.0); }
bool  mat_is_transmissive(int mat) {
    if (mat < 0 || mat >= n_mats) return false;
    return mat_transmittance(mat, 0) > 1.0e-6 || abs(mat_ior_real(mat, 0) - 1.0) > 1.0e-6;
}
float medium_n_real(int mat) {
    if (mat < 0 || mat >= n_mats) return 1.0;
    float n = mat_ior_real(mat, 0);
    return (n > 1.0e-10) ? n : 1.0;
}

float ggx_D(float alpha, float NoH) {
    float a = max(alpha, 1.0e-4);
    float a2 = a * a;
    float d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (3.141592653589793 * d * d);
}

float ggx_G1(float alpha, float NoV) {
    float a = max(alpha, 1.0e-4);
    float a2 = a * a;
    return (NoV > 0.0) ? (2.0 * NoV) / (NoV + sqrt(a2 + (1.0 - a2) * NoV * NoV)) : 0.0;
}

float ggx_pdf_solid_angle(vec3 n, vec3 in_dir, vec3 out_dir, float alpha) {
    vec3 V = normalize(-in_dir);
    vec3 L = normalize(out_dir);
    float NoV = max(0.0, dot(n, V));
    float NoL = max(0.0, dot(n, L));
    if (NoV <= 0.0 || NoL <= 0.0) return 0.0;
    vec3 H = V + L;
    float h2 = dot(H, H);
    if (h2 <= 1.0e-20) return 0.0;
    H *= inversesqrt(h2);
    float NoH = max(0.0, dot(n, H));
    float VoH = max(0.0, dot(V, H));
    if (NoH <= 0.0 || VoH <= 1.0e-12) return 0.0;
    return ggx_D(alpha, NoH) * ggx_G1(alpha, NoV) / (4.0 * NoV);
}

vec3 ggx_sample_reflection(vec3 n, vec3 in_dir, float alpha, inout uint rng) {
    float u1 = rand_next(rng);
    float u2 = rand_next(rng);
    float a = max(alpha, 1.0e-4);
    vec3 up = abs(n.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 t = normalize(cross(up, n));
    vec3 b = cross(n, t);
    vec3 V = normalize(-in_dir);
    vec3 Vlocal = vec3(dot(V, t), dot(V, b), dot(V, n));
    if (Vlocal.z <= 1.0e-8) {
        return normalize(in_dir - 2.0 * dot(in_dir, n) * n);
    }
    vec3 Vh = normalize(vec3(a * Vlocal.x, a * Vlocal.y, Vlocal.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = (lensq > 1.0e-20) ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);
    float r = sqrt(max(0.0, u1));
    float phi = 6.28318530718 * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
    vec3 Nh = t1 * T1 + t2 * T2
            + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    vec3 Hlocal = normalize(vec3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));
    vec3 H = normalize(t * Hlocal.x + b * Hlocal.y + n * Hlocal.z);
    if (dot(H, V) < 0.0) H = -H;
    return normalize(in_dir - 2.0 * dot(in_dir, H) * H);
}

/* ── RefinedHit field accessors ──────────────────────────────────────────── */
#define HIT(base, off)  hits[(base) + (off)]
vec3 hit_pos    (uint b) { return vec3(HIT(b,0), HIT(b,1), HIT(b,2)); }
vec3 hit_n      (uint b) { return vec3(HIT(b,3), HIT(b,4), HIT(b,5)); }
vec3 hit_dir    (uint b) { return vec3(HIT(b,6), HIT(b,7), HIT(b,8)); }
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
float hit_pathatseg(uint b) { return HIT(b, 13); }
uint  hit_bdpt_sid (uint b) { return floatBitsToUint(HIT(b, 26 + 2 * MAX_BANDS)); }

/* ── BDPT emit helpers ───────────────────────────────────────────────────── */
#define BDPT_VERTEX_STRIDE   28
#define BDPT_SPECTRAL_STRIDE  8
#define BDPT_PDF_STRIDE      12
#define BDPT_DOMAIN_SOLID_ANGLE       2u
#define BDPT_DOMAIN_PROJ_SOLID_ANGLE  3u
#define BDPT_PDF_FLAG_DELTA_SPECULAR  (1u << 16)
#define BDPT_PDF_FLAG_SPLIT           (1u << 17)
#define BDPT_PDF_FLAG_DIFFUSE         (1u << 18)
#define BDPT_PDF_FLAG_GGX             (1u << 20)

void emit_bdpt_vertex(uint sid, uint packed_vi, uint tri_flags_u, int tri_id, int mat_id,
                      vec3 pos, vec3 nrm, vec3 dir_in,
                      float path_len, float path_at_seg, float throughput,
                      float soy, float soz)
{
    if (bdpt_max_verts <= 0) return;
    uint slot = atomicAdd(meta[bdpt_count_base + 0], 1u);
    if (int(slot) >= bdpt_max_verts) return;
    uint vb = slot * uint(BDPT_VERTEX_STRIDE);
    bdpt_out[vb +  0] = uintBitsToFloat(sid);
    bdpt_out[vb +  1] = uintBitsToFloat(packed_vi);
    bdpt_out[vb +  2] = uintBitsToFloat(tri_flags_u);
    bdpt_out[vb +  3] = 0.0;
    bdpt_out[vb +  4] = intBitsToFloat(tri_id);
    bdpt_out[vb +  5] = intBitsToFloat(-1);
    bdpt_out[vb +  6] = intBitsToFloat(mat_id);
    bdpt_out[vb +  7] = pos.x;   bdpt_out[vb +  8] = pos.y;   bdpt_out[vb +  9] = pos.z;
    bdpt_out[vb + 10] = nrm.x;   bdpt_out[vb + 11] = nrm.y;   bdpt_out[vb + 12] = nrm.z;
    bdpt_out[vb + 13] = dir_in.x; bdpt_out[vb + 14] = dir_in.y; bdpt_out[vb + 15] = dir_in.z;
    bdpt_out[vb + 16] = 0.0;     bdpt_out[vb + 17] = 0.0;     bdpt_out[vb + 18] = 0.0;
    bdpt_out[vb + 19] = path_len;
    bdpt_out[vb + 20] = path_at_seg;
    bdpt_out[vb + 21] = 0.0;
    bdpt_out[vb + 22] = 0.0;
    bdpt_out[vb + 23] = 0.0;
    bdpt_out[vb + 24] = 0.0;
    bdpt_out[vb + 25] = throughput;
    bdpt_out[vb + 26] = soy;
    bdpt_out[vb + 27] = soz;
}

void emit_bdpt_spectral(uint sid, uint vi_band, float re, float im, float band_pdf)
{
    if (bdpt_max_spectral <= 0) return;
    uint slot = atomicAdd(meta[bdpt_count_base + 1], 1u);
    if (int(slot) >= bdpt_max_spectral) return;
    uint sb = uint(bdpt_spectral_base) + slot * uint(BDPT_SPECTRAL_STRIDE);
    bdpt_out[sb + 0] = uintBitsToFloat(sid);
    bdpt_out[sb + 1] = uintBitsToFloat(vi_band);
    bdpt_out[sb + 2] = re;
    bdpt_out[sb + 3] = im;
    bdpt_out[sb + 4] = 0.0;
    bdpt_out[sb + 5] = band_pdf;
    bdpt_out[sb + 6] = 1.0;
    bdpt_out[sb + 7] = 0.0;
}

void emit_bdpt_pdf(uint sid, uint vi, uint domain, float pdf_fwd, float pdf_rev,
                   uint flags, vec3 nrm, vec3 dir_in, vec3 dir_out)
{
    if (sid == 0u || bdpt_max_pdfs <= 0) return;
    uint slot = atomicAdd(meta[bdpt_count_base + 2], 1u);
    if (int(slot) >= bdpt_max_pdfs) return;

    float cos_in  = max(0.0, -dot(dir_in, nrm));
    float cos_out = max(0.0,  dot(dir_out, nrm));
    uint packed_vm = ((domain & 0xFFu) << 16) | ((domain & 0xFFu) << 24) | (vi & 0xFFFFu);

    uint pb = uint(bdpt_pdf_base) + slot * uint(BDPT_PDF_STRIDE);
    bdpt_out[pb +  0] = uintBitsToFloat(sid);
    bdpt_out[pb +  1] = uintBitsToFloat(packed_vm);
    bdpt_out[pb +  2] = pdf_fwd;
    bdpt_out[pb +  3] = pdf_rev;
    bdpt_out[pb +  4] = 0.0;
    bdpt_out[pb +  5] = pdf_fwd;
    bdpt_out[pb +  6] = 1.0;
    bdpt_out[pb +  7] = cos_in * cos_out;
    bdpt_out[pb +  8] = uintBitsToFloat(flags);
    bdpt_out[pb +  9] = 0.0;
    bdpt_out[pb + 10] = 0.0;
    bdpt_out[pb + 11] = 0.0;
}

/* ── Triangle accessors (TRI_FULL_STRIDE = 16, matches T1/T2 buffer) ──────── */
int   tri_flags  (int t) { return floatBitsToInt(tris[t * TRI_FULL_STRIDE + 12]); }
int   tri_mat    (int t) { return floatBitsToInt(tris[t * TRI_FULL_STRIDE + 13]); }
int   tri_med_pos(int t) { return floatBitsToInt(tris[t * TRI_FULL_STRIDE + 14]); }
int   tri_med_neg(int t) { return floatBitsToInt(tris[t * TRI_FULL_STRIDE + 15]); }
vec3  tri_normal (int t) { return vec3(tris[t * TRI_FULL_STRIDE + 9],
                                       tris[t * TRI_FULL_STRIDE + 10],
                                       tris[t * TRI_FULL_STRIDE + 11]); }

/* ── Write helpers ───────────────────────────────────────────────────────── */

/* Copy RefinedHit header into a terminal record slot in out_buf[].
 * Terminals are packed at out_buf[max_children*INTENT_STRIDE + slot*TERMINAL_STRIDE]. */
void write_terminal(uint slot, uint hbase, bool is_emissive) {
    uint tbase = uint(max_children) * uint(INTENT_STRIDE) + slot * uint(TERMINAL_STRIDE);
    for (int i = 0; i < TERMINAL_STRIDE; i++)   /* copy 58 floats; hit[58]=bdpt_sid not needed */
        out_buf[tbase + i] = hits[hbase + i];
    out_buf[tbase + 25] = uintBitsToFloat(is_emissive ? 1u : 0u);
}

/* Write a child RayIntent into out_buf[] at slot < max_children. */
void write_intent(uint slot,
                  vec3 pos, vec3 dir, float path_len,
                  int medium, uint iflags,
                  int src_id, int bounce, int bounces_left, float min_amp,
                  uint tag_lo, uint tag_hi, uint color_flag, float priority,
                  float soy, float soz, uint bdpt_sid,
                  float amp_re[MAX_BANDS], float amp_im[MAX_BANDS])
{
    uint ibase = slot * uint(INTENT_STRIDE);
    vec3 child_pos = pos + normalize(dir) * RAY_ORIGIN_EPS;
    out_buf[ibase +  0] = child_pos.x;
    out_buf[ibase +  1] = child_pos.y;
    out_buf[ibase +  2] = child_pos.z;
    out_buf[ibase +  3] = dir.x;
    out_buf[ibase +  4] = dir.y;
    out_buf[ibase +  5] = dir.z;
    out_buf[ibase +  6] = path_len;
    out_buf[ibase +  7] = intBitsToFloat(medium);
    out_buf[ibase +  8] = uintBitsToFloat(iflags);
    out_buf[ibase +  9] = intBitsToFloat(src_id);
    out_buf[ibase + 10] = intBitsToFloat(bounce);
    out_buf[ibase + 11] = intBitsToFloat(bounces_left);
    out_buf[ibase + 12] = min_amp;
    out_buf[ibase + 13] = uintBitsToFloat(tag_lo);
    out_buf[ibase + 14] = uintBitsToFloat(tag_hi);
    out_buf[ibase + 15] = uintBitsToFloat(color_flag);
    out_buf[ibase + 16] = priority;
    out_buf[ibase + 17] = soy;
    out_buf[ibase + 18] = soz;
    out_buf[ibase + 19] = uintBitsToFloat(bdpt_sid);
    for (int b = 0; b < MAX_BANDS; b++) {
        out_buf[ibase + 20 +           b] = amp_re[b];
        out_buf[ibase + 20 + MAX_BANDS + b] = amp_im[b];
    }
}

/* ── Main ────────────────────────────────────────────────────────────────── */
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= n_hits) return;

    uint hbase = gid * uint(REFINED_HIT_STRIDE);
    uint rng   = rng_init(gid);

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

    float amp_re[MAX_BANDS];
    float amp_im[MAX_BANDS];
    int nb = min(n_bands, MAX_BANDS);
    for (int b = 0; b < nb; b++) {
        amp_re[b] = hit_amp_re(hbase, b);
        amp_im[b] = hit_amp_im(hbase, b);
    }
    for (int b = nb; b < MAX_BANDS; b++) { amp_re[b] = 0.0; amp_im[b] = 0.0; }

    /* ── BDPT vertex emission ───────────────────────────────────────────────
     * Vertex record emitted pre-scatter (position/normal/direction needed).
     * Spectral weight records are emitted POST-scatter in each scatter branch
     * so that beta_lut encodes material-weighted amplitude, not arriving flux.
     * Exception: emissive terminals emit their incoming amplitude as the
     * emission spectrum (pre-scatter IS the emission in that case). */
    uint bdpt_sid      = hit_bdpt_sid(hbase);
    uint bdpt_vi       = uint(bounce) & 0xFFFFu;
    float bdpt_band_pdf = (nb > 0) ? 1.0 / float(nb) : 1.0;
    if (bdpt_sid != 0u && bdpt_max_verts > 0) {
        uint vi         = bdpt_vi;
        uint stream_bit = (cflag == 1u) ? 1u : 0u;  /* 1=SENSOR, 0=LIGHT */
        uint packed_vi  = (vi << 16) | (stream_bit << 8);  /* sample_domain=0=UNKNOWN */
        int  tri_idx_v  = hit_tri(hbase);
        int  mat_id_v   = hit_mat(hbase);
        uint tflags_u   = (tri_idx_v >= 0) ? uint(tri_flags(tri_idx_v)) : 0u;
        float throughput = 0.0;
        for (int b = 0; b < nb; b++)
            throughput += sqrt(amp_re[b] * amp_re[b] + amp_im[b] * amp_im[b]);
        emit_bdpt_vertex(bdpt_sid, packed_vi, tflags_u, tri_idx_v, mat_id_v,
                         pos, nrm, in_dir,
                         path_len, hit_pathatseg(hbase),
                         throughput, soy, soz);
        /* spectral emit moved to post-scatter branches below */
    }

    /* Absorb: T2 rejected this ray via parametric acceptance boundary (bit 3).
     * No child intent, no terminal — ray contributes zero energy. */
    if ((cflag & 8u) != 0u) return;

    /* ── Neural passthrough: T2 teleported this hit via MLP ────────────────
     * Bit 4 of cflag is set by T2 for NEURAL_ASSEMBLY hits.  Re-emit the ray
     * directly with the updated pos/dir and current medium (air), bypassing
     * all Snell/Fresnel physics to avoid corrupting the teleported trajectory. */
    if ((cflag & 4u) != 0u) {
        float cf_re[MAX_BANDS], cf_im[MAX_BANDS];
        for (int b = 0; b < MAX_BANDS; b++) { cf_re[b] = amp_re[b]; cf_im[b] = amp_im[b]; }
        uint islot = atomicAdd(meta[0], 1u);
        write_intent(islot, pos, in_dir, path_len, medium, 0u,
                     src_id, bounce, bleft, min_amp,
                     tag_lo, tag_hi, cflag & ~4u, 1.0, soy, soz, bdpt_sid, cf_re, cf_im);
        return;
    }

    int flags   = tri_flags(tri_idx);
    vec3 geom_n = normalize(tri_normal(tri_idx));
    bool front_face = dot(in_dir, geom_n) < 0.0;

    /* ── UV integrator image splat — fires for every hit before any return ──
     *
     * Channel layout (n2 = res*res texels per channel):
     *   [0]           hit count
     *   [1]           source-ID bitfield (OR)
     *   [2..5]        bounce histogram (0 / 1 / 2 / 3+)
     *   [6]           tag_lo OR
     *   [7]           tag_hi OR
     *   [8..10]       hit-normal xyz, signed ×32768
     *   [11..11+B-1]  per-band magnitude ×65536
     *   [11+B..11+2B-1]  per-band amp_re ×32768 (signed)
     *   [11+2B..11+3B-1] per-band amp_im ×32768 (signed)
     *   [11+3B..11+4B-1] forward/emissive magnitude ×65536
     *   [11+4B..11+5B-1] sensor/reverse magnitude ×65536
     * Total channels = 11 + 5*n_bands. */
    if (n_uv_groups > 0 && tri_idx >= 0) {
        int uv_gid = merged_uv[uv_group_id_base + tri_idx];
        if (uv_gid >= 0 && uv_gid < n_uv_groups) {
            int res    = merged_uv[uv_meta_base + uv_gid * 2 + 0];
            int offset = merged_uv[uv_meta_base + uv_gid * 2 + 1];
            if (res > 0) {
                int tb = tri_idx * TRI_FULL_STRIDE;
                vec3 v0 = vec3(tris[tb + 0], tris[tb + 1], tris[tb + 2]);
                vec3 e1 = vec3(tris[tb + 3], tris[tb + 4], tris[tb + 5]);
                vec3 e2 = vec3(tris[tb + 6], tris[tb + 7], tris[tb + 8]);
                vec3 dp = pos - v0;
                float e1e1 = dot(e1, e1);
                float e1e2 = dot(e1, e2);
                float e2e2 = dot(e2, e2);
                float de1  = dot(dp, e1);
                float de2  = dot(dp, e2);
                float det  = e1e1 * e2e2 - e1e2 * e1e2;
                float bu   = (det > 1.0e-20) ? (e2e2 * de1 - e1e2 * de2) / det : 0.0;
                float bv   = (det > 1.0e-20) ? (e1e1 * de2 - e1e2 * de1) / det : 0.0;
                bu = clamp(bu, 0.0, 1.0);
                bv = clamp(bv, 0.0, 1.0 - bu);
                int uvb = tri_idx * 6;  /* into merged_uv[] UV coord section */
                vec2 uv0 = vec2(intBitsToFloat(merged_uv[uvb + 0]), intBitsToFloat(merged_uv[uvb + 1]));
                vec2 uv1 = vec2(intBitsToFloat(merged_uv[uvb + 2]), intBitsToFloat(merged_uv[uvb + 3]));
                vec2 uv2 = vec2(intBitsToFloat(merged_uv[uvb + 4]), intBitsToFloat(merged_uv[uvb + 5]));
                vec2 uv  = uv0 + bu * (uv1 - uv0) + bv * (uv2 - uv0);
                int ix    = clamp(int(uv.x * float(res)), 0, res - 1);
                int iy    = clamp(int(uv.y * float(res)), 0, res - 1);
                int texel = iy * res + ix;
                int n2    = res * res;

                /* [0] hit count */
                atomicAdd(uv_accum[offset + 0 * n2 + texel], 1u);

                /* [1] source-ID bitfield */
                atomicOr(uv_accum[offset + 1 * n2 + texel],
                         1u << uint(clamp(src_id, 0, 31)));

                /* [2-5] bounce histogram */
                atomicAdd(uv_accum[offset + (2 + clamp(bounce, 0, 3)) * n2 + texel], 1u);

                /* [6-7] tag bitfields */
                atomicOr(uv_accum[offset + 6 * n2 + texel], tag_lo);
                atomicOr(uv_accum[offset + 7 * n2 + texel], tag_hi);

                /* [8-10] hit-normal xyz, signed fixed-pt ×32768 */
                const float UV_SAFE = 1073741824.0; /* 2^30, safe clamp for int() */
                atomicAdd(uv_accum[offset + 8 * n2 + texel],
                          uint(int(clamp(nrm.x * 32768.0, -UV_SAFE, UV_SAFE))));
                atomicAdd(uv_accum[offset + 9 * n2 + texel],
                          uint(int(clamp(nrm.y * 32768.0, -UV_SAFE, UV_SAFE))));
                atomicAdd(uv_accum[offset + 10 * n2 + texel],
                          uint(int(clamp(nrm.z * 32768.0, -UV_SAFE, UV_SAFE))));

                /* [11+b] magnitude ×65536; [11+B+b] re ×32768; [11+2B+b] im ×32768 */
                for (int b = 0; b < nb; b++) {
                    float re  = amp_re[b], im = amp_im[b];
                    float mag = sqrt(re * re + im * im);
                    atomicAdd(uv_accum[offset + (11 + b) * n2 + texel],
                              uint(clamp(mag * 65536.0, 0.0, float(0xFFFFFFFFu))));
                    atomicAdd(uv_accum[offset + (11 + n_bands + b) * n2 + texel],
                              uint(int(clamp(re * 32768.0, -UV_SAFE, UV_SAFE))));
                    atomicAdd(uv_accum[offset + (11 + 2 * n_bands + b) * n2 + texel],
                              uint(int(clamp(im * 32768.0, -UV_SAFE, UV_SAFE))));
                    int split_base = 11 + ((cflag == 1u) ? 4 : 3) * n_bands;
                    atomicAdd(uv_accum[offset + (split_base + b) * n2 + texel],
                              uint(clamp(mag * 65536.0, 0.0, float(0xFFFFFFFFu))));
                }
            }
        }
    }

    /* ── Terminal: aperture stop ── */
    if ((flags & MAT_FLAG_APERTURE_STOP) != 0) {
        uint tslot = atomicAdd(meta[1], 1u);
        write_terminal(tslot, hbase, false);
        return;
    }

    /* ── Terminal: emissive ── */
    if ((flags & MAT_FLAG_EMISSIVE) != 0) {
        /* Emitter: incoming amplitude IS the emission spectrum — emit pre-scatter. */
        if (bdpt_sid != 0u && bdpt_max_verts > 0) {
            for (int b = 0; b < nb; b++) {
                uint vi_band = (bdpt_vi << 16) | uint(b);
                emit_bdpt_spectral(bdpt_sid, vi_band, amp_re[b], amp_im[b], bdpt_band_pdf);
            }
        }
        uint tslot = atomicAdd(meta[1], 1u);
        write_terminal(tslot, hbase, true);
        return;
    }

    /* ── Terminal: budget exhausted ── */
    if (bleft <= 0) {
        uint tslot = atomicAdd(meta[1], 1u);
        write_terminal(tslot, hbase, false);
        return;
    }

    /* ── Epsilon fast-path: fully absorptive opaque material ── */
    bool is_transmissive = mat_is_transmissive(mat_id);
    if (!is_transmissive && mat_id >= 0 && mat_id < n_mats) {
        if (meta[2u + uint(mat_id)] != 0u) {
            uint tslot = atomicAdd(meta[1], 1u);
            write_terminal(tslot, hbase, false);
            return;
        }
    }

    int   new_bounce    = bounce + 1;
    int   new_bleft     = bleft - 1;
    uint  iflags        = 0u;

    if (is_transmissive) {
        int med_pos = tri_med_pos(tri_idx);
        int med_neg = tri_med_neg(tri_idx);
        bool has_pair = (med_pos != med_neg);
        int medium_from = -1;
        int medium_to   = -1;
        if (has_pair) {
            if (front_face) {
                medium_from = med_pos;
                medium_to   = med_neg;
            } else {
                medium_from = med_neg;
                medium_to   = med_pos;
            }
        }

        float n1 = has_pair ? medium_n_real(medium_from) : medium_n_real(medium);
        float n2 = has_pair ? medium_n_real(medium_to)
                            : (front_face ? medium_n_real(mat_id) : 1.0);

        float cos_i  = max(0.0, -dot(in_dir, nrm));
        vec3  refracted;
        bool  can_refract = snell_refract(in_dir, nrm, n1, n2, refracted);

        if (!can_refract) {
            vec3 rd = normalize(in_dir - 2.0 * dot(in_dir, nrm) * nrm);
            float ra_re[MAX_BANDS], ra_im[MAX_BANDS];
            for (int b = 0; b < nb; b++) {
                float rr = (mat_id >= 0) ? mat_refl_re(mat_id, b) : 1.0;
                float ri = (mat_id >= 0) ? mat_refl_im(mat_id, b) : 0.0;
                ra_re[b] = amp_re[b]*rr - amp_im[b]*ri;
                ra_im[b] = amp_re[b]*ri + amp_im[b]*rr;
            }
            for (int b = nb; b < MAX_BANDS; b++) { ra_re[b]=0.0; ra_im[b]=0.0; }
            uint islot = atomicAdd(meta[0], 1u);
            int new_med = has_pair ? medium_from : (front_face ? mat_id : -1);
            emit_bdpt_pdf(bdpt_sid, uint(bounce), BDPT_DOMAIN_SOLID_ANGLE, 1.0, 1.0,
                          uint(flags) | BDPT_PDF_FLAG_DELTA_SPECULAR,
                          nrm, in_dir, rd);
            if (bdpt_sid != 0u && bdpt_max_verts > 0) {
                for (int b = 0; b < nb; b++) {
                    uint vi_band = (bdpt_vi << 16) | uint(b);
                    emit_bdpt_spectral(bdpt_sid, vi_band, ra_re[b], ra_im[b], bdpt_band_pdf);
                }
            }
            write_intent(islot, pos, rd, path_len, new_med, iflags,
                         src_id, new_bounce, new_bleft, min_amp,
                         tag_lo, tag_hi, cflag, 1.0, soy, soz, bdpt_sid, ra_re, ra_im);
        } else {
            float sin2_t = (n1/n2)*(n1/n2)*(1.0 - cos_i*cos_i);
            float cos_t  = sqrt(max(0.0, 1.0 - sin2_t));
            float R      = fresnel_R(cos_i, cos_t, n1, n2);
            float R_rev  = fresnel_R(cos_t, cos_i, n2, n1);
            int   new_med = has_pair ? medium_to : (front_face ? mat_id : -1);

            if (max_children_per_hit >= 2) {
                float rs = sqrt(R);
                float ts = sqrt(max(0.0, 1.0 - R));
                {
                    vec3  rd = normalize(in_dir - 2.0 * dot(in_dir, nrm) * nrm);
                    float ra_re[MAX_BANDS], ra_im[MAX_BANDS];
                    for (int b = 0; b < nb; b++) {
                        float rr = (mat_id >= 0) ? mat_refl_re(mat_id, b) : 1.0;
                        float ri = (mat_id >= 0) ? mat_refl_im(mat_id, b) : 0.0;
                        ra_re[b] = rs * (amp_re[b]*rr - amp_im[b]*ri);
                        ra_im[b] = rs * (amp_re[b]*ri + amp_im[b]*rr);
                    }
                    for (int b = nb; b < MAX_BANDS; b++) { ra_re[b]=0.0; ra_im[b]=0.0; }
                    uint islot = atomicAdd(meta[0], 1u);
                    emit_bdpt_pdf(bdpt_sid, uint(bounce), BDPT_DOMAIN_SOLID_ANGLE, R, R,
                                  uint(flags) | BDPT_PDF_FLAG_DELTA_SPECULAR | BDPT_PDF_FLAG_SPLIT,
                                  nrm, in_dir, rd);
                    if (bdpt_sid != 0u && bdpt_max_verts > 0) {
                        for (int b = 0; b < nb; b++) {
                            uint vi_band = (bdpt_vi << 16) | uint(b);
                            emit_bdpt_spectral(bdpt_sid, vi_band, ra_re[b], ra_im[b], bdpt_band_pdf);
                        }
                    }
                    write_intent(islot, pos, rd, path_len, medium, iflags,
                                 src_id, new_bounce, new_bleft, min_amp,
                                 tag_lo, tag_hi, cflag, 1.0, soy, soz, bdpt_sid, ra_re, ra_im);
                }
                {
                    float ta_re[MAX_BANDS], ta_im[MAX_BANDS];
                    for (int b = 0; b < nb; b++) {
                        ta_re[b] = ts * amp_re[b];
                        ta_im[b] = ts * amp_im[b];
                    }
                    for (int b = nb; b < MAX_BANDS; b++) { ta_re[b]=0.0; ta_im[b]=0.0; }
                    uint islot = atomicAdd(meta[0], 1u);
                    emit_bdpt_pdf(bdpt_sid, uint(bounce), BDPT_DOMAIN_SOLID_ANGLE,
                                  1.0 - R, 1.0 - R_rev,
                                  uint(flags) | BDPT_PDF_FLAG_DELTA_SPECULAR | BDPT_PDF_FLAG_SPLIT,
                                  nrm, in_dir, refracted);
                    if (bdpt_sid != 0u && bdpt_max_verts > 0) {
                        for (int b = 0; b < nb; b++) {
                            uint vi_band = (bdpt_vi << 16) | uint(b);
                            emit_bdpt_spectral(bdpt_sid, vi_band, ta_re[b], ta_im[b], bdpt_band_pdf);
                        }
                    }
                    write_intent(islot, pos, refracted, path_len, new_med, iflags,
                                 src_id, new_bounce, new_bleft, min_amp,
                                 tag_lo, tag_hi, cflag, 1.0, soy, soz, bdpt_sid, ta_re, ta_im);
                }
            } else {
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
                uint islot = atomicAdd(meta[0], 1u);
                uint pdf_flags = BDPT_PDF_FLAG_DELTA_SPECULAR;
                float pdf_fwd = (u < R) ? R : (1.0 - R);
                float pdf_rev = (u < R) ? R : (1.0 - R_rev);
                emit_bdpt_pdf(bdpt_sid, uint(bounce), BDPT_DOMAIN_SOLID_ANGLE, pdf_fwd, pdf_rev,
                              uint(flags) | pdf_flags, nrm, in_dir, cdir);
                if (bdpt_sid != 0u && bdpt_max_verts > 0) {
                    for (int b = 0; b < nb; b++) {
                        uint vi_band = (bdpt_vi << 16) | uint(b);
                        emit_bdpt_spectral(bdpt_sid, vi_band, ca_re[b], ca_im[b], bdpt_band_pdf);
                    }
                }
                write_intent(islot, pos, cdir, path_len, cmed, iflags,
                             src_id, new_bounce, new_bleft, min_amp,
                             tag_lo, tag_hi, cflag, 1.0, soy, soz, bdpt_sid, ca_re, ca_im);
            }
        }
    } else {
        float diffusion = (mat_id >= 0) ? mat_diffusion(mat_id) : 0.0;
        float spec_p = max(0.0, 1.0 - diffusion);
        float ggx_alpha = (mat_id >= 0) ? mat_ggx_alpha(mat_id) : 0.0;
        vec3 new_dir;
        bool chose_diffuse = false;
        bool chose_ggx = false;
        if (rand_next(rng) < diffusion) {
            new_dir = cosine_hemisphere(nrm, rng);
            chose_diffuse = true;
        } else if (spec_p > 0.0 && ggx_alpha > 1.0e-3) {
            new_dir = ggx_sample_reflection(nrm, in_dir, ggx_alpha, rng);
            chose_ggx = dot(new_dir, nrm) > 0.0;
        } else {
            new_dir = normalize(in_dir - 2.0 * dot(in_dir, nrm) * nrm);
        }
        if (dot(new_dir, nrm) <= 0.0) {
            uint tslot = atomicAdd(meta[1], 1u);
            write_terminal(tslot, hbase, false);
            return;
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
            uint islot = atomicAdd(meta[0], 1u);
            float diffuse_p = (mat_id >= 0) ? mat_diffusion(mat_id) : 0.0;
            float scatter_pdf = 0.0;
            float scatter_pdf_rev = 0.0;
            uint scatter_domain = chose_diffuse ? BDPT_DOMAIN_PROJ_SOLID_ANGLE : BDPT_DOMAIN_SOLID_ANGLE;
            uint scatter_flags = 0u;
            if (chose_diffuse) {
                scatter_pdf = diffuse_p * max(0.0, dot(new_dir, nrm)) / 3.141592653589793;
                scatter_pdf_rev = diffuse_p * max(0.0, -dot(in_dir, nrm)) / 3.141592653589793;
                scatter_flags = BDPT_PDF_FLAG_DIFFUSE;
            } else if (chose_ggx) {
                scatter_pdf = spec_p * ggx_pdf_solid_angle(nrm, in_dir, new_dir, ggx_alpha);
                scatter_pdf_rev = spec_p * ggx_pdf_solid_angle(nrm, -new_dir, -in_dir, ggx_alpha);
                scatter_flags = BDPT_PDF_FLAG_GGX;
            } else {
                scatter_pdf = spec_p;
                scatter_pdf_rev = spec_p;
                scatter_flags = BDPT_PDF_FLAG_DELTA_SPECULAR;
            }
            if (scatter_pdf <= 0.0 || scatter_pdf_rev <= 0.0) {
                uint tslot = atomicAdd(meta[1], 1u);
                write_terminal(tslot, hbase, false);
                return;
            }
            emit_bdpt_pdf(bdpt_sid, uint(bounce), scatter_domain, scatter_pdf, scatter_pdf_rev,
                          uint(flags) | scatter_flags, nrm, in_dir, new_dir);
            /* Post-scatter spectral emit: na_re/na_im = amp × mat_refl — encodes material color */
            if (bdpt_sid != 0u && bdpt_max_verts > 0) {
                for (int b = 0; b < nb; b++) {
                    uint vi_band = (bdpt_vi << 16) | uint(b);
                    emit_bdpt_spectral(bdpt_sid, vi_band, na_re[b], na_im[b], bdpt_band_pdf);
                }
            }
            write_intent(islot, pos, new_dir, path_len, medium, iflags,
                         src_id, new_bounce, new_bleft, min_amp,
                         tag_lo, tag_hi, cflag, 1.0, soy, soz, bdpt_sid, na_re, na_im);
        } else {
            uint tslot = atomicAdd(meta[1], 1u);
            write_terminal(tslot, hbase, false);
        }
    }
}
