#version 430 core
/* ─────────────────────────────────────────────────────────────────────────────
 * bdpt_pack_t5.comp.glsl  —  GPU-native T5: pack sorted BDPT vertices into T5 format
 *
 * After the bitonic sort, vertices are in (stream, subpath_id, vertex_index) order.
 * Light vertices occupy sorted positions [0..n_lv-1], cam vertices [n_lv..nv-1].
 * This shader reads each sorted vertex and writes it into ssbo_t5_light or
 * ssbo_t5_cam at the correct stride expected by t5_full_connect.comp.glsl.
 *
 * vinfo[0..14] = packed vertex position within the sorted subpath.  The raw
 * BDPT vertex_index may have gaps from optical/lens traversal; T5 needs the
 * dense packed index so flat_base = gid - li_v navigates the actual arrays.
 *
 * This pass is intentionally O(vertices).  Per-band spectral records and
 * optical event records are scattered into the packed rows by bdpt_scatter_t5
 * after this pass.  Do not linearly scan side-record sections here: at normal
 * counts that is hundreds of billions of probes and trips Windows TDR.
 *
 * Approximations vs CPU path:
 *   - pdf_fwd / pdf_rev / pdf_flags: read from the vertex record, which T3
 *     fills from the sampling event.
 *   - prefix_pdf: initialized only; T5 builds MIS products from edge PDFs.
 *   - spectral_beta: initialized from scalar throughput here, then overwritten
 *     per band by bdpt_scatter_t5.
 *   - optical_block / optical_jacobian: initialized here, then updated by
 *     bdpt_scatter_t5 from optical records.
 *   - edge_fwd/bwd_area: initialized to zero, then filled from PDF side
 *     records by bdpt_scatter_t5.
 *   - diffuse_p / ggx_alpha: read from mat_buf[mat_idx, band 0].
 *
 * Bindings:
 *   0  SortKeysBuf  sort_keys  readonly
 *   1  SortIdxBuf   sort_idx   readonly
 *   2  BdptVertBuf  ssbo_bdpt_output  readonly  (verts + spectral/pdf/optical sections)
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
uniform int   base_idx;         /* chunk offset for split dispatches          */
uniform int   n_lv;             /* light vertex count (= sort_counts[0])      */
uniform int   n_bands;          /* active spectral bands                      */
uniform float sensor_half_w;    /* sensor half-width for spectral_beta_rgb    */
uniform float sensor_half_h;

float mat_diffuse_p(int mat_idx) {
    if (mat_idx < 0) return 0.5;
    return clamp(mat_bands[mat_idx * MAT_FULL_BANDS * MAT_BAND_STRIDE + 4], 0.0, 1.0);
}

float mat_ggx_alpha(int mat_idx) {
    if (mat_idx < 0) return 0.0;
    return clamp(mat_bands[mat_idx * MAT_FULL_BANDS * MAT_BAND_STRIDE + 10], 0.0, 1.0);
}

void main() {
    int P = base_idx + int(gl_GlobalInvocationID.x);
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

    /* vinfo: same layout as CPU — packed subpath position | stream<<16 | tri_valid<<31.
     * packed_pos = number of earlier sorted rows in the same (stream, sid)
     * group, i.e. this vertex's dense index within its subpath.
     *
     * B.4c: the bitonic sort orders rows ascending by the full key (key_hi
     * then key_lo), so all rows sharing this key_hi are CONTIGUOUS and this
     * row sits at global position P inside that run.  packed_pos is therefore
     * P minus the run's start index.  The previous code found the start with a
     * per-thread O(group) backward scan whose cost grew with subpath length;
     * replace it with an exact O(log nv) binary search (lower_bound of key_hi).
     * Result is identical — only the start index is needed. */
    uint key_hi = sort_keys[P * 2 + 0];
    int lo = 0;
    int hi = P;                       /* group start cannot exceed P          */
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (sort_keys[mid * 2 + 0] < key_hi) lo = mid + 1;
        else                                 hi = mid;
    }
    uint packed_pos = uint(P - lo);   /* == count of equal-key_hi predecessors */
    uint vinfo = packed_pos | (stream << 16) | (tri_valid ? (1u << 31) : 0u);
    if (stream == 1u && packed_pos == 0u)
        pdf_flags |= (1u << 22); /* BDPT_PDF_FLAG_SENSOR */

    /* O(vertices) initialization.  bdpt_scatter_t5 overwrites per-band values
     * from spectral side records after this pack pass. */
    const int nb = min(n_bands, MAX_BANDS);
    float betas[MAX_BANDS];
    for (int b = 0; b < MAX_BANDS; ++b)
        betas[b] = 0.0;
    if (nb > 0) {
        float beta_per_band = throughput / float(nb);
        for (int b = 0; b < nb; ++b)
            betas[b] = beta_per_band;
    }

    uint optical_block = 0u;
    float optical_jacobian = 1.0;

    /* Material properties */
    float diffuse_p = mat_diffuse_p(mat_id);
    float ggx_alpha = mat_ggx_alpha(mat_id);

    /* Edge PDFs are filled later by bdpt_scatter_t5 from BdptPdfRecord rows.
     * Do not scan the PDF side section here; this pass must remain O(vertices). */
    float edge_fwd = 0.0, edge_bwd = 0.0;

    /* T5 currently computes prefix/suffix products from edge fields directly. */
    float prefix_pdf = 1.0;

    if (stream == 0u) {
        /* ── Light vertex → ssbo_t5_light[P * T5_LGV_STRIDE] ── */
        int ob = P * T5_LGV_STRIDE;
        t5_light[ob +  0] = pos.x;
        t5_light[ob +  1] = pos.y;
        t5_light[ob +  2] = pos.z;
        t5_light[ob +  3] = nrm.x;
        t5_light[ob +  4] = nrm.y;
        t5_light[ob +  5] = nrm.z;
        t5_light[ob +  6] = intBitsToFloat(floatBitsToInt(bdpt_verts[vb + 4]));
        t5_light[ob +  7] = uintBitsToFloat(tflags_u);
        t5_light[ob +  8] = uintBitsToFloat(sid);
        t5_light[ob +  9] = uintBitsToFloat(vinfo);
        t5_light[ob + 10] = intBitsToFloat(mat_id);
        t5_light[ob + 11] = pdf_fwd;
        t5_light[ob + 12] = pdf_rev;
        t5_light[ob + 13] = uintBitsToFloat(pdf_flags);
        t5_light[ob + 14] = uintBitsToFloat(optical_block);
        t5_light[ob + 15] = prefix_pdf;
        for (int b = 0; b < MAX_BANDS; ++b)
            t5_light[ob + 16 + b] = betas[b];   /* [16..47] */
        t5_light[ob + 48] = dir_in.x;
        t5_light[ob + 49] = dir_in.y;
        t5_light[ob + 50] = dir_in.z;
        t5_light[ob + 51] = diffuse_p;
        t5_light[ob + 52] = ggx_alpha;
        t5_light[ob + 53] = optical_jacobian;
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
        t5_cam[ob +  6] = intBitsToFloat(floatBitsToInt(bdpt_verts[vb + 4]));
        t5_cam[ob +  7] = uintBitsToFloat(tflags_u);
        t5_cam[ob +  8] = uintBitsToFloat(sid);
        t5_cam[ob +  9] = uintBitsToFloat(vinfo);
        /* [10..12] RESERVED — legacy per-vertex luminance-RGB beta slots.
         * B.1c: these are DEAD.  t5_full_connect derives camera colour from the
         * per-band slots (CGV_BAND_BASE [22..53]) via cam_spectral_rgb, and
         * bdpt_scatter_t5 only overwrites those per-band slots — nothing ever
         * reads [10..12].  Zero them for deterministic buffer contents; do not
         * resurrect the flat beta_lum/3 split (it desaturated camera colour). */
        t5_cam[ob + 10] = 0.0;
        t5_cam[ob + 11] = 0.0;
        t5_cam[ob + 12] = 0.0;
        t5_cam[ob + 13] = soy;
        t5_cam[ob + 14] = soz;
        t5_cam[ob + 15] = pdf_fwd;
        t5_cam[ob + 16] = pdf_rev;
        t5_cam[ob + 17] = uintBitsToFloat(pdf_flags);
        t5_cam[ob + 18] = uintBitsToFloat(optical_block);
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
        t5_cam[ob + 59] = optical_jacobian;
        t5_cam[ob + 60] = edge_fwd;
        t5_cam[ob + 61] = edge_bwd;
        for (int i = 62; i < T5_CGV_STRIDE; ++i)
            t5_cam[ob + i] = 0.0;
    }
}
