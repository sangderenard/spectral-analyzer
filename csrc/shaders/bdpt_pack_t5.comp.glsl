#version 430 core
/* ─────────────────────────────────────────────────────────────────────────────
 * bdpt_pack_t5.comp.glsl  —  GPU-native T5: pack sorted BDPT vertices into T5 format
 *
 * After the bitonic sort, vertices are in (stream, subpath_id, vertex_index) order.
 * Light vertices occupy sorted positions [0..n_lv-1], cam vertices [n_lv..nv-1].
 * This shader reads each sorted vertex and writes it into ssbo_t5_light or
 * ssbo_t5_cam at the correct stride expected by t5_full_connect.comp.glsl.
 *
 * vinfo[0..14] = vertex_index_in_subpath (from packed_vi>>16).  The T5 shader
 * uses  flat_base = gid - li_v  to navigate the subpath chain; since vertices
 * are in subpath-consecutive order this is exactly correct.
 *
 * Approximations vs CPU path:
 *   - pdf_fwd / pdf_rev: read from vertex record [21][22] (T3 now writes them).
 *   - pdf_flags: read from vertex record [23] (T3 now writes them).
 *   - optical_block: forced 0 (conservative — treats every path as unblocked).
 *   - optical_jacobian: forced 1.0.
 *   - prefix_pdf: set to pdf_fwd of current vertex (not cumulative product).
 *   - spectral_beta_rgb: equal-weight 1/3 per channel across all bands.
 *   - edge_fwd/bwd_area: computed from consecutive vertex geometry (diffuse approx).
 *   - diffuse_p / ggx_alpha: read from mat_buf[mat_idx, band 0].
 *
 * Bindings:
 *   0  SortKeysBuf  sort_keys  readonly
 *   1  SortIdxBuf   sort_idx   readonly
 *   2  BdptVertBuf  ssbo_bdpt_output  readonly  (verts only; betas derived from throughput)
 *   3  T5LightBuf   ssbo_t5_light     writeonly
 *   4  T5CamBuf     ssbo_t5_cam       writeonly
 *   5  MatBandBuf   ssbo_mat_band     readonly  (diffuse_p, ggx_alpha lookup)
 * ─────────────────────────────────────────────────────────────────────────────
 */
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer SortKeysBuf { uint  sort_keys[];  };
layout(std430, binding = 1) readonly buffer SortIdxBuf  { uint  sort_idx[];   };
layout(std430, binding = 2) readonly buffer BdptVertBuf { float bdpt_verts[]; };
layout(std430, binding = 3)          buffer T5LightBuf  { float t5_light[];   };
layout(std430, binding = 4)          buffer T5CamBuf    { float t5_cam[];     };
layout(std430, binding = 5) readonly buffer MatBandBuf  { float mat_bands[];  };

#define BDPT_VERTEX_STRIDE 28
#define MAX_BANDS          32
#define T5_LGV_STRIDE      56
#define T5_CGV_STRIDE      72
#define MAT_FULL_BANDS     32
#define MAT_BAND_STRIDE    12

uniform int   nv;               /* actual vertex count                        */
uniform int   n_lv;             /* light vertex count (= sort_counts[0])      */
uniform int   n_bands;          /* active spectral bands                      */
uniform float sensor_half_w;    /* sensor half-width for spectral_beta_rgb    */
uniform float sensor_half_h;

float mat_diffuse_p(int mat_idx) {
    if (mat_idx < 0) return 0.5;
    return mat_bands[mat_idx * MAT_FULL_BANDS * MAT_BAND_STRIDE + 4];  /* band 0 field [4] */
}

float mat_ggx_alpha(int mat_idx) {
    if (mat_idx < 0) return 0.0;
    return mat_bands[mat_idx * MAT_FULL_BANDS * MAT_BAND_STRIDE + 10]; /* band 0 field [10] */
}

/* Diffuse area PDF for edge from v0 (pos0, nrm0) to v1 (pos1).
 * p_area(v0→v1) = cos(theta_out) / pi / dist² */
float edge_pdf_area_diffuse(vec3 pos0, vec3 nrm0, vec3 pos1) {
    vec3 d = pos1 - pos0;
    float dist2 = max(dot(d, d), 1e-12);
    float cos_out = max(0.0, dot(nrm0, normalize(d)));
    return cos_out / (3.141592653589793 * dist2);
}

void main() {
    int P = int(gl_GlobalInvocationID.x);
    if (P >= nv) return;

    int   idx  = int(sort_idx[P]);
    int   vb   = idx * BDPT_VERTEX_STRIDE;

    /* Load raw vertex record fields */
    uint sid        = floatBitsToUint(bdpt_verts[vb +  0]);
    uint packed_vi  = floatBitsToUint(bdpt_verts[vb +  1]);
    uint tflags_u   = floatBitsToUint(bdpt_verts[vb +  2]);
    int  mat_id     = floatBitsToInt (bdpt_verts[vb +  6]);
    vec3 pos        = vec3(bdpt_verts[vb + 7], bdpt_verts[vb + 8], bdpt_verts[vb + 9]);
    vec3 nrm        = vec3(bdpt_verts[vb +10], bdpt_verts[vb +11], bdpt_verts[vb +12]);
    vec3 dir_in     = vec3(bdpt_verts[vb +13], bdpt_verts[vb +14], bdpt_verts[vb +15]);
    float path_len  = bdpt_verts[vb + 19];
    float path_seg  = bdpt_verts[vb + 20];
    float pdf_fwd   = bdpt_verts[vb + 21];  /* T3 now writes this */
    float pdf_rev   = bdpt_verts[vb + 22];  /* T3 now writes this */
    uint  pdf_flags = floatBitsToUint(bdpt_verts[vb + 23]); /* T3 now writes this */
    float throughput= bdpt_verts[vb + 25];
    float soy       = bdpt_verts[vb + 26];
    float soz       = bdpt_verts[vb + 27];

    uint stream      = (packed_vi >> 8) & 1u;
    uint vi          = (packed_vi >> 16) & 0xFFFFu;   /* vertex_index in subpath */
    bool tri_valid   = (tflags_u & 0x01u) != 0u || floatBitsToInt(bdpt_verts[vb + 4]) >= 0;

    /* vinfo: same layout as CPU — li_v (flat subpath position) | stream<<16 | tri_valid<<31 */
    uint vinfo = vi | (stream << 16) | (tri_valid ? (1u << 31) : 0u);

    /* Per-band betas: throughput is sum(|amp[b]|) stored in vertex record.
     * Distribute uniformly across bands (equal-weight approximation). */
    const int nb = min(n_bands, MAX_BANDS);
    float beta_lum = throughput;
    float beta_per_band = (nb > 0) ? throughput / float(nb) : 0.0;
    float betas[MAX_BANDS];
    for (int b = 0; b < MAX_BANDS; ++b)
        betas[b] = (b < nb) ? beta_per_band : 0.0;
    float beta_r = throughput / 3.0;
    float beta_g = throughput / 3.0;
    float beta_b = throughput / 3.0;

    /* Material properties */
    float diffuse_p = mat_diffuse_p(mat_id);
    float ggx_alpha = mat_ggx_alpha(mat_id);

    /* Edge PDFs: read next sorted vertex in same subpath (P+1 if same subpath) */
    float edge_fwd = 0.0, edge_bwd = 0.0;
    if (P + 1 < nv) {
        int nxt_idx = int(sort_idx[P + 1]);
        int nxt_vb  = nxt_idx * BDPT_VERTEX_STRIDE;
        uint nxt_sid = floatBitsToUint(bdpt_verts[nxt_vb + 0]);
        uint nxt_vi  = (floatBitsToUint(bdpt_verts[nxt_vb + 1]) >> 16) & 0xFFFFu;
        if (nxt_sid == sid && nxt_vi == vi + 1u) {
            vec3 nxt_pos = vec3(bdpt_verts[nxt_vb+7], bdpt_verts[nxt_vb+8], bdpt_verts[nxt_vb+9]);
            vec3 nxt_nrm = vec3(bdpt_verts[nxt_vb+10],bdpt_verts[nxt_vb+11],bdpt_verts[nxt_vb+12]);
            edge_fwd = edge_pdf_area_diffuse(pos, nrm, nxt_pos);
            edge_bwd = edge_pdf_area_diffuse(nxt_pos, nxt_nrm, pos);
        }
    }

    /* prefix_pdf: cumulative product of pdf_fwd values along the subpath.
     * After the sort, all vertices of this subpath are at consecutive sorted positions
     * [P - vi .. P].  Walk back from P to the subpath start multiplying pdf_fwd.
     * This is O(vi) per thread (max vi ≈ 16), fully parallel, no extra pass. */
    float prefix_pdf = 1.0;
    for (int k = int(vi); k >= 0; --k) {
        int pk = P - k;
        if (pk < 0 || pk >= nv) { prefix_pdf = 0.0; break; }
        /* Verify we're still in the same subpath (same stream + subpath_id) */
        uint pk_hi = sort_keys[pk * 2 + 0];
        if (pk_hi != sort_keys[P * 2 + 0]) { prefix_pdf = 0.0; break; }
        float pf = bdpt_verts[int(sort_idx[pk]) * BDPT_VERTEX_STRIDE + 21];
        if (pf <= 0.0) { prefix_pdf = 0.0; break; }
        prefix_pdf *= pf;
    }

    if (stream == 0u) {
        /* ── Light vertex → ssbo_t5_light[P * T5_LGV_STRIDE] ── */
        int ob = P * T5_LGV_STRIDE;
        t5_light[ob +  0] = pos.x;
        t5_light[ob +  1] = pos.y;
        t5_light[ob +  2] = pos.z;
        t5_light[ob +  3] = nrm.x;
        t5_light[ob +  4] = nrm.y;
        t5_light[ob +  5] = nrm.z;
        t5_light[ob +  6] = throughput;
        t5_light[ob +  7] = uintBitsToFloat(tflags_u);
        t5_light[ob +  8] = uintBitsToFloat(sid);
        t5_light[ob +  9] = uintBitsToFloat(vinfo);
        t5_light[ob + 10] = beta_lum;
        t5_light[ob + 11] = pdf_fwd;
        t5_light[ob + 12] = pdf_rev;
        t5_light[ob + 13] = uintBitsToFloat(pdf_flags);
        t5_light[ob + 14] = 0.0;                /* optical_block = 0 */
        t5_light[ob + 15] = prefix_pdf;
        for (int b = 0; b < MAX_BANDS; ++b)
            t5_light[ob + 16 + b] = betas[b];   /* [16..47] */
        t5_light[ob + 48] = dir_in.x;
        t5_light[ob + 49] = dir_in.y;
        t5_light[ob + 50] = dir_in.z;
        t5_light[ob + 51] = diffuse_p;
        t5_light[ob + 52] = ggx_alpha;
        t5_light[ob + 53] = 1.0;                /* optical_jacobian */
        t5_light[ob + 54] = edge_fwd;
        t5_light[ob + 55] = edge_bwd;
    } else {
        /* ── Camera vertex → ssbo_t5_cam[(P - n_lv) * T5_CGV_STRIDE] ── */
        int ob = (P - n_lv) * T5_CGV_STRIDE;
        t5_cam[ob +  0] = pos.x;
        t5_cam[ob +  1] = pos.y;
        t5_cam[ob +  2] = pos.z;
        t5_cam[ob +  3] = nrm.x;
        t5_cam[ob +  4] = nrm.y;
        t5_cam[ob +  5] = nrm.z;
        t5_cam[ob +  6] = throughput;
        t5_cam[ob +  7] = uintBitsToFloat(tflags_u);
        t5_cam[ob +  8] = uintBitsToFloat(sid);
        t5_cam[ob +  9] = uintBitsToFloat(vinfo);
        t5_cam[ob + 10] = beta_r;
        t5_cam[ob + 11] = beta_g;
        t5_cam[ob + 12] = beta_b;
        t5_cam[ob + 13] = soy;
        t5_cam[ob + 14] = soz;
        t5_cam[ob + 15] = pdf_fwd;
        t5_cam[ob + 16] = pdf_rev;
        t5_cam[ob + 17] = uintBitsToFloat(pdf_flags);
        t5_cam[ob + 18] = 0.0;                  /* optical_block */
        t5_cam[ob + 19] = prefix_pdf;
        t5_cam[ob + 20] = 0.0;
        t5_cam[ob + 21] = intBitsToFloat(mat_id);
        for (int b = 0; b < MAX_BANDS; ++b)
            t5_cam[ob + 22 + b] = betas[b];     /* [22..53] */
        t5_cam[ob + 54] = dir_in.x;
        t5_cam[ob + 55] = dir_in.y;
        t5_cam[ob + 56] = dir_in.z;
        t5_cam[ob + 57] = diffuse_p;
        t5_cam[ob + 58] = ggx_alpha;
        t5_cam[ob + 59] = 1.0;                  /* optical_jacobian */
        t5_cam[ob + 60] = edge_fwd;
        t5_cam[ob + 61] = edge_bwd;
        for (int i = 62; i < T5_CGV_STRIDE; ++i)
            t5_cam[ob + i] = 0.0;
    }
}
