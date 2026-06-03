#version 430 core
/* ────────────────────────────────────────────────────────────────────────────
 * t5_full_connect.comp.glsl  —  GPU BDPT T5 tiled connection pass
 *                               with full multi-strategy balance-heuristic MIS.
 *
 * 2-D tiled dispatch: X = camera tile, Y = light tile within current batch.
 *   Work-group size: TILE_C × TILE_L  (default 8×8 = 64 threads).
 *   Each invocation handles ONE (cam_vert, light_vert) pair.
 *
 * Both tiles are loaded cooperatively into shared memory so every thread in
 * the work-group reuses the data without redundant SSBO reads.
 * A per-WG shared accumulator array is reduced before the final atomic write,
 * cutting atomic-CAS pressure by TILE_L per cam vert per WG.
 *
 * ALL subpath vertices are still packed consecutively in the global SSBOs so
 * candidate_strategy_density can navigate the full chain.
 * Subpath flat base formula: flat_index − vertex_index_in_subpath.
 *
 * Dispatch (C++): glDispatchCompute(
 *     ceil(n_cam_verts / TILE_C),
 *     ceil(light_batch_size / TILE_L),
 *     1)
 *
 * ── SSBO binding map ──────────────────────────────────────────────────────
 *  binding  buffer          description
 *  -------  --------------- -----------------------------------------------
 *    0      T5LightVertBuf  Flat light vertices. Stride = T5_LGV_STRIDE = 40 floats.
 *             [0..2]   pos xyz
 *             [3..5]   normal xyz
 *             [6]      throughput_scalar
 *             [7]      uintBitsToFloat(MAT_FLAG_* material flags)
 *             [8]      uintBitsToFloat(subpath_id)
 *             [9]      uintBitsToFloat(vinfo): bits[0..14]=vertex_index,
 *                        bits[16..22]=stream, bit[31]=tri_valid (tri_id>=0)
 *             [10]     beta_lum (scalar path throughput)
 *             [11]     pdf_fwd
 *             [12]     pdf_rev
 *             [13]     uintBitsToFloat(BdptPdfRecord::flags)
 *                        bit(1<<16)=DELTA_SPECULAR  bit(1<<18)=DIFFUSE  bit(1<<20)=GGX
 *             [14]     uintBitsToFloat(optical_block)  non-zero = blocked
 *             [15]     prefix_pdf  (0 = prefix invalid)
 *             [16..31] per-band beta magnitudes [band 0..15]
 *             [32..34] dir_in xyz
 *             [35]     diffuse_p  (from mat_cache)
 *             [36]     ggx_alpha  (from surf_cache)
 *             [37]     optical_jacobian
 *             [38]     edge_fwd_area  (area-domain PDF: this vert → next in subpath)
 *             [39]     edge_bwd_area  (reverse area-domain PDF: next → this at this vert)
 *
 *    1      T5CamVertBuf    Flat camera vertices. Stride = T5_CGV_STRIDE = 56 floats.
 *             [0..2]   pos xyz
 *             [3..5]   normal xyz
 *             [6]      throughput_scalar
 *             [7]      uintBitsToFloat(MAT_FLAG_* material flags)
 *             [8]      uintBitsToFloat(subpath_id)
 *             [9]      uintBitsToFloat(vinfo): same bit layout as LGV[9]
 *             [10]     spectral_beta_r  (band_to_display_rgb weighted sum)
 *             [11]     spectral_beta_g
 *             [12]     spectral_beta_b
 *             [13]     sensor_origin_y
 *             [14]     sensor_origin_z
 *             [15]     pdf_fwd
 *             [16]     pdf_rev
 *             [17]     uintBitsToFloat(BdptPdfRecord::flags)  same layout as LGV[13]
 *             [18]     uintBitsToFloat(optical_block)         same layout as LGV[14]
 *             [19]     prefix_pdf
 *             [20]     reserved (0.0)
 *             [21]     intBitsToFloat(mat_idx)
 *             [22..37] per-band beta magnitudes [band 0..15]
 *             [38..40] dir_in xyz
 *             [41]     diffuse_p
 *             [42]     ggx_alpha
 *             [43]     optical_jacobian
 *             [44]     edge_fwd_area
 *             [45]     edge_bwd_area
 *             [46..55] reserved (0.0)
 *
 *    2      T5PixelBuf      Pixel accumulator. 3×res² uint32 float-bits.
 *                           Layout: R[res²] G[res²] B[res²].
 *                           Written via float CAS (no ext needed).
 *    3      T5ParamsBuf     T5GpuParams std430 block.
 *    5      BvhBuf          BVH nodes          (injected shadow preamble)
 *    6      TriIdBuf        BVH triangle IDs   (injected shadow preamble)
 *    7      TriFullBuf      Full triangle data  (injected shadow preamble)
 *
 * ── T5GpuParams (std430, binding 3) ───────────────────────────────────────
 *   float  min_geom        geometry-term floor
 *   float  sensor_half_w   sensor half-width  (Y axis, metres)
 *   float  sensor_half_h   sensor half-height (Z axis, metres)
 *   uint   cam_offset       first cam vert index for this cbatch
 *   uint   n_light_verts
 *   uint   n_cam_verts      upper bound: cam_offset + cbatch size
 *   int    sensor_res      pixel grid side (res×res)
 *   uint   light_batch_size
 *   uint   light_offset    first light vert index for this dispatch
 *   int    n_bands
 *   int    tile_x0         pixel column of tile left edge
 *   int    tile_y0         pixel row of tile top edge
 *   int    tile_w          tile width  in pixels (0 = full res)
 *   int    tile_h          tile height in pixels (0 = full res)
 * ────────────────────────────────────────────────────────────────────────── */

/* ── Tile dimensions (must match T5_TILE_C / T5_TILE_L in ray_pipeline.h) ── */
#define TILE_C  8
#define TILE_L  8

layout(local_size_x = TILE_C, local_size_y = TILE_L, local_size_z = 1) in;

/* ── Stride and field offsets (mirror ray_pipeline.h) ─────────────────── */
#define T5_LGV_STRIDE    56
#define T5_CGV_STRIDE    72

#define LGV_DIR_IN_X      48
#define LGV_DIFFUSE_P     51
#define LGV_GGX_ALPHA     52
#define LGV_OPT_JACOBIAN  53
#define LGV_EDGE_FWD      54
#define LGV_EDGE_BWD      55

#define CGV_DIR_IN_X      54
#define CGV_DIFFUSE_P     57
#define CGV_GGX_ALPHA     58
#define CGV_OPT_JACOBIAN  59
#define CGV_EDGE_FWD      60
#define CGV_EDGE_BWD      61

#define CGV_BAND_BASE     22    /* per-band beta magnitudes [22..53] (T5_MAX_GPU_BANDS slots) */
#define LGV_BAND_BASE     16    /* per-band beta magnitudes [16..47] (T5_MAX_GPU_BANDS slots) */
#define T5_MAX_GPU_BANDS  32

/* ── Shared memory ──────────────────────────────────────────────────────── *
 * s_cam / s_light: cooperative tile loads of all vertex fields.            *
 *   Layout: consecutive stride-sized slots, [vert * stride + field].       *
 * s_lum_*: per-invocation contributions, reduced before atomic write.      *
 *   Layout: tid = lid_l * TILE_C + lid_c.                                  */
shared float s_cam  [TILE_C * T5_CGV_STRIDE];   /* 8×72 = 576 floats (2.25 KB) */
shared float s_light[TILE_L * T5_LGV_STRIDE];   /* 8×56 = 448 floats (1.75 KB) */
shared float s_lum_r[TILE_C * TILE_L];
shared float s_lum_g[TILE_C * TILE_L];
shared float s_lum_b[TILE_C * TILE_L];

/* ── SSBOs ─────────────────────────────────────────────────────────────── */
layout(std430, binding = 0) readonly buffer T5LightVertBuf {
    float light_verts[];
};
layout(std430, binding = 1) readonly buffer T5CamVertBuf {
    float cam_verts[];
};
layout(std430, binding = 2) coherent buffer T5PixelBuf {
    uint pixel_accum[];
};
layout(std430, binding = 3) readonly buffer T5ParamsBuf {
    float  min_geom;
    float  sensor_half_w;
    float  sensor_half_h;
    uint   cam_offset;
    uint   n_light_verts;
    uint   n_cam_verts;
    int    sensor_res;
    uint   light_batch_size;
    uint   light_offset;
    int    n_bands;
    int    tile_x0;
    int    tile_y0;
    int    tile_w;
    int    tile_h;
};

/* ── Spectral colour weights (n_bands × 3): [b*3+0]=wr, [b*3+1]=wg, [b*3+2]=wb */
layout(std430, binding = 4) readonly buffer T5SpectralWeightBuf {
    float spectral_weights[];
};

/* ── Flag constants (mirror bdpt_record.h / mat_flags_generated.h) ──────── */
#define MAT_FLAG_APERTURE_STOP        128u
#define BDPT_PDF_FLAG_DELTA_SPECULAR  (1u << 16)
#define BDPT_PDF_FLAG_DIFFUSE         (1u << 18)
#define BDPT_PDF_FLAG_GGX             (1u << 20)
#define BDPT_PDF_FLAG_EMISSION        (1u << 21)

#define PI 3.14159265358979323846f

/* ── MIS chain size limit.                                                 *
 * MAX_CHAIN = 32 supports up to 15 bounces per subpath plus the two       *
 * endpoints.  Pairs exceeding this are skipped (no energy loss in practice *
 * since path depths are typically ≤ 8).                                    *
 * MAX_CHAIN_E = max edge count (N-1 for N=MAX_CHAIN vertices).            *
 * MAX_CHAIN_N = size of prefix/suffix arrays (N+1 sentinel slot).         */
#define MAX_CHAIN    32
#define MAX_CHAIN_E  31
#define MAX_CHAIN_N  33

/* ── GGX microfacet helpers ──────────────────────────────────────────────── */
float ggx_D(float alpha, float NoH) {
    float a  = max(1e-4f, alpha);
    float a2 = a * a;
    float d  = NoH * NoH * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * d * d);
}

float ggx_G1(float alpha, float NoV) {
    if (NoV <= 0.0f) return 0.0f;
    float a  = max(1e-4f, alpha);
    float a2 = a * a;
    return (2.0f * NoV) / (NoV + sqrt(a2 + (1.0f - a2) * NoV * NoV));
}

float ggx_pdf_sa(vec3 n, vec3 in_dir, vec3 out_dir, float alpha) {
    vec3  V   = normalize(-in_dir);
    vec3  L   = normalize(out_dir);
    float NoV = max(0.0f, dot(n, V));
    float NoL = max(0.0f, dot(n, L));
    if (NoV <= 0.0f || NoL <= 0.0f) return 0.0f;
    vec3  H   = normalize(V + L);
    float NoH = max(0.0f, dot(n, H));
    float VoH = max(0.0f, dot(V, H));
    if (NoH <= 0.0f || VoH < 1e-6f) return 0.0f;
    return ggx_D(alpha, NoH) * ggx_G1(alpha, NoV) / (4.0f * NoV);
}

/* ── scatter_conn_pdf_area ───────────────────────────────────────────────── *
 * Area-measure scatter PDF at vertex 'from' toward vertex 'to'.             *
 * Mirrors T5ConnContext::scatter_connection_pdf_area().                      *
 * Returns 0.0 if the vertex cannot scatter toward the target.               */
float scatter_conn_pdf_area(
    vec3  from_pos,    vec3  from_norm,  vec3  from_dir_in,
    float diffuse_p,   float ggx_alpha,
    uint  pdf_flags,   float opt_jac,
    vec3  to_pos,      vec3  to_norm)
{
    if ((pdf_flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) return 0.0f;
    if (opt_jac <= 0.0f) return 0.0f;

    vec3  d     = to_pos - from_pos;
    float dist2 = dot(d, d);
    if (dist2 < 1e-18f) return 1e-12f;
    float dist  = sqrt(dist2);
    d /= dist;

    float cos_out = max(0.0f, dot(from_norm, d));
    float cos_to  = max(0.0f, abs(dot(to_norm, -d)));
    if (cos_out <= 0.0f || cos_to <= 0.0f) return 0.0f;

    float spec_p = max(0.0f, 1.0f - diffuse_p);
    float pdf_sa = 0.0f;
    if ((pdf_flags & BDPT_PDF_FLAG_EMISSION) != 0u) {
        pdf_sa = cos_out / PI;
    } else if ((pdf_flags & BDPT_PDF_FLAG_DIFFUSE) != 0u) {
        if (diffuse_p <= 0.0f) return 0.0f;
        pdf_sa = diffuse_p * cos_out / PI;
    } else if ((pdf_flags & BDPT_PDF_FLAG_GGX) != 0u) {
        if (spec_p <= 0.0f || ggx_alpha <= 1e-3f) return 0.0f;
        pdf_sa = spec_p * ggx_pdf_sa(from_norm, from_dir_in, d, ggx_alpha);
    } else {
        return 0.0f;
    }
    if (pdf_sa <= 0.0f) return 0.0f;

    float pdf = pdf_sa * cos_to / dist2 * opt_jac;
    return max(pdf, 1e-12f);
}

/* ── vertex_connectable ──────────────────────────────────────────────────── *
 * GPU port of T5ConnContext::vertex_connectable().                           */
bool vertex_connectable(uint vinfo, uint vflags, uint pdf_flags, uint opt_block) {
    if ((vinfo     >> 31)                          == 0u) return false; /* tri_id < 0 */
    if ((vflags    & MAT_FLAG_APERTURE_STOP)       != 0u) return false;
    if ((pdf_flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) return false;
    if ( opt_block                                 != 0u) return false;
    return true;
}

/* ── candidate_strategy_density ─────────────────────────────────────────── *
 * Full multi-strategy balance-heuristic MIS.                                *
 * Direct GPU port of T5ConnContext::candidate_strategy_density().           *
 *                                                                            *
 * ci         = vertex_index of the connection cam vert in its subpath       *
 * cam_base   = flat index of subpath vertex 0 in cam_verts[]                *
 * li_v       = vertex_index of the connection light vert in its subpath     *
 * light_base = flat index of subpath vertex 0 in light_verts[]              *
 * conn_fwd   = scatter_conn_pdf_area(cam[ci] → light[li_v])                *
 * conn_bwd   = scatter_conn_pdf_area(light[li_v] → cam[ci])                *
 *                                                                            *
 * Outputs:                                                                   *
 *   out_selected_pdf = prefix_fwd[ci] × suffix_bwd[ci+1]  (selected cut)   *
 *   out_denom        = sum of prefix×suffix over all valid connectable cuts  *
 *                                                                            *
 * Returns false if chain too long, selected cut invalid, or denom is zero.  */
bool candidate_strategy_density(
    uint  ci,        uint cam_base,
    uint  li_v,      uint light_base,
    float conn_fwd,  float conn_bwd,
    out float out_selected_pdf,
    out float out_denom)
{
    uint N = ci + li_v + 2u;
    if (N > uint(MAX_CHAIN)) return false;

    /* ── Build edge PDFs for all N-1 edges in the combined chain ─────────── *
     * edge_fwd[e] > 0  ⟺  forward edge PDF is valid (non-zero).            *
     * edge_bwd[e] > 0  ⟺  backward edge PDF is valid.                      *
     * Zero serves as the "invalid/blocked" sentinel.                        *
     *                                                                        *
     * Chain:  cam[0] … cam[ci]  |  light[li_v] … light[0]                  *
     *                             ↑ connection edge at e=ci                  *
     *                                                                        *
     * Within-cam edges (e < ci):                                             *
     *   fwd = edge_fwd_area packed at cam[e]  (cam[e] → cam[e+1])          *
     *   bwd = edge_bwd_area packed at cam[e]  (cam[e+1] → cam[e] reverse)  *
     *                                                                        *
     * Connection edge (e = ci):                                              *
     *   fwd = scatter_conn_pdf_area(cam[ci] → light[li_v]) = conn_fwd      *
     *   bwd = scatter_conn_pdf_area(light[li_v] → cam[ci]) = conn_bwd      *
     *                                                                        *
     * Within-light edges reversed (e > ci):                                  *
     *   e = ci+1+j,  j = e−ci−1,  lci = li_v−j                            *
     *   parent (in orig subpath) = light[lci−1]                            *
     *   In the chain direction (reversed), fwd uses orig bwd and vice versa. *
     *   fwd = edge_bwd_area packed at light[lci−1]                         *
     *   bwd = edge_fwd_area packed at light[lci−1]                         */
    float edge_fwd[MAX_CHAIN_E];
    float edge_bwd[MAX_CHAIN_E];

    for (uint e = 0u; e < N - 1u; ++e) {
        edge_fwd[e] = 0.0f;
        edge_bwd[e] = 0.0f;

        if (e < ci) {
            uint cb_e = (cam_base + e) * uint(T5_CGV_STRIDE);
            edge_fwd[e] = cam_verts[cb_e + uint(CGV_EDGE_FWD)];
            edge_bwd[e] = cam_verts[cb_e + uint(CGV_EDGE_BWD)];

        } else if (e == ci) {
            edge_fwd[e] = conn_fwd;
            edge_bwd[e] = conn_bwd;

        } else {
            uint j   = e - ci - 1u;
            uint lci = li_v - j;
            if (lci == 0u) return false;
            uint lb_e   = (light_base + lci - 1u) * uint(T5_LGV_STRIDE);
            edge_fwd[e] = light_verts[lb_e + uint(LGV_EDGE_BWD)]; /* orig bwd → chain fwd */
            edge_bwd[e] = light_verts[lb_e + uint(LGV_EDGE_FWD)]; /* orig fwd → chain bwd */
        }
    }

    /* ── Prefix products (chain start → each vertex) ────────────────────── *
     * prefix_fwd[i] = product of edge_fwd[0 .. i-1].                       *
     * 0.0 encodes "prefix chain is broken" (invalid edge encountered).     */
    float prefix_fwd[MAX_CHAIN_N];
    prefix_fwd[0] = 1.0f;
    for (uint i = 1u; i <= N; ++i) {
        float ef = edge_fwd[i - 1u];
        prefix_fwd[i] = (prefix_fwd[i-1u] > 0.0f && ef > 0.0f)
            ? max(prefix_fwd[i-1u] * ef, 1e-30f)
            : 0.0f;
    }

    /* ── Suffix products (each vertex → chain end) ───────────────────────── *
     * suffix_bwd[i] = product of edge_bwd[i .. N-2].                       *
     * suffix_bwd[N-1] = 1.0 (no backward edges from the last vertex).      */
    float suffix_bwd[MAX_CHAIN_N];
    suffix_bwd[N - 1u] = 1.0f;
    for (int si = int(N) - 2; si >= 0; si--) {
        uint  i  = uint(si);
        float eb = edge_bwd[i];
        suffix_bwd[i] = (suffix_bwd[i + 1u] > 0.0f && eb > 0.0f)
            ? max(suffix_bwd[i + 1u] * eb, 1e-30f)
            : 0.0f;
    }

    /* ── Selected cut (our (s,t) strategy): between cam[ci] and light[li_v] */
    uint sc = ci + 1u;
    if (prefix_fwd[sc - 1u] <= 0.0f || suffix_bwd[sc] <= 0.0f) return false;
    out_selected_pdf = max(prefix_fwd[sc - 1u] * suffix_bwd[sc], 1e-30f);

    /* ── Denominator: sum prefix×suffix over all connectable cuts ─────────── */
    out_denom = 0.0f;
    for (uint cut = 1u; cut < N; ++cut) {
        float pf = prefix_fwd[cut - 1u];
        float sb = suffix_bwd[cut];
        if (pf <= 0.0f || sb <= 0.0f) continue;

        /* Load vinfo/vflags/pdf_flags/opt_block for chain[cut-1] and chain[cut]. */
        uint vi0, vf0, pf0, ob0;
        uint vi1, vf1, pf1, ob1;

        /* chain[cut-1]: cam vert if cut-1 <= ci, else light vert. */
        if (cut - 1u <= ci) {
            uint cb0 = (cam_base + (cut - 1u)) * uint(T5_CGV_STRIDE);
            vi0 = floatBitsToUint(cam_verts[cb0 +  9u]);
            vf0 = floatBitsToUint(cam_verts[cb0 +  7u]);
            pf0 = floatBitsToUint(cam_verts[cb0 + 17u]);
            ob0 = floatBitsToUint(cam_verts[cb0 + 18u]);
        } else {
            uint j0  = (cut - 1u) - ci - 1u;
            uint lb0 = (light_base + li_v - j0) * uint(T5_LGV_STRIDE);
            vi0 = floatBitsToUint(light_verts[lb0 +  9u]);
            vf0 = floatBitsToUint(light_verts[lb0 +  7u]);
            pf0 = floatBitsToUint(light_verts[lb0 + 13u]);
            ob0 = floatBitsToUint(light_verts[lb0 + 14u]);
        }

        /* chain[cut]: cam vert if cut <= ci, else light vert. */
        if (cut <= ci) {
            uint cb1 = (cam_base + cut) * uint(T5_CGV_STRIDE);
            vi1 = floatBitsToUint(cam_verts[cb1 +  9u]);
            vf1 = floatBitsToUint(cam_verts[cb1 +  7u]);
            pf1 = floatBitsToUint(cam_verts[cb1 + 17u]);
            ob1 = floatBitsToUint(cam_verts[cb1 + 18u]);
        } else {
            uint j1  = cut - ci - 1u;
            uint lb1 = (light_base + li_v - j1) * uint(T5_LGV_STRIDE);
            vi1 = floatBitsToUint(light_verts[lb1 +  9u]);
            vf1 = floatBitsToUint(light_verts[lb1 +  7u]);
            pf1 = floatBitsToUint(light_verts[lb1 + 13u]);
            ob1 = floatBitsToUint(light_verts[lb1 + 14u]);
        }

        if (!vertex_connectable(vi0, vf0, pf0, ob0)) continue;
        if (!vertex_connectable(vi1, vf1, pf1, ob1)) continue;

        float p = pf * sb;
        if (p > 0.0f && !isinf(p) && !isnan(p)) out_denom += p;
    }

    return out_selected_pdf > 0.0f && out_denom > 0.0f;
}

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
    const uint lid_c = gl_LocalInvocationID.x;   /* 0..TILE_C-1 */
    const uint lid_l = gl_LocalInvocationID.y;   /* 0..TILE_L-1 */
    const uint tid   = lid_l * uint(TILE_C) + lid_c;  /* flat thread index 0..63 */
    const uint total = uint(TILE_C) * uint(TILE_L);   /* 64 */

    /* Global tile bases.  light_tile is relative to the current batch origin,
     * so the absolute light vert index = light_offset + light_tile + lid_l.
     * cam_offset shifts the X-axis base for camera-vertex batching (cbatch). */
    const uint cam_tile   = cam_offset + gl_WorkGroupID.x * uint(TILE_C);
    const uint light_tile = gl_WorkGroupID.y * uint(TILE_L);

    /* ── Cooperative tile loads ─────────────────────────────────────────── *
     * Each of the 64 threads strides through the tile arrays.               *
     * Out-of-bounds verts are zeroed so downstream logic reads clean data.  */

    /* Camera tile: TILE_C × T5_CGV_STRIDE = 448 floats → 7 loads/thread   */
    for (uint i = tid; i < uint(TILE_C) * uint(T5_CGV_STRIDE); i += total) {
        const uint v = i / uint(T5_CGV_STRIDE);
        const uint f = i % uint(T5_CGV_STRIDE);
        const uint g = cam_tile + v;
        s_cam[i] = (g < n_cam_verts)
            ? cam_verts[g * uint(T5_CGV_STRIDE) + f]
            : 0.0f;
    }

    /* Light tile: TILE_L × T5_LGV_STRIDE = 320 floats → 5 loads/thread    */
    for (uint i = tid; i < uint(TILE_L) * uint(T5_LGV_STRIDE); i += total) {
        const uint v      = i / uint(T5_LGV_STRIDE);
        const uint f      = i % uint(T5_LGV_STRIDE);
        const uint g_glob = light_offset + light_tile + v;
        s_light[i] = (g_glob < n_light_verts)
            ? light_verts[g_glob * uint(T5_LGV_STRIDE) + f]
            : 0.0f;
    }

    /* Init per-invocation accumulators. */
    s_lum_r[tid] = 0.0f;
    s_lum_g[tid] = 0.0f;
    s_lum_b[tid] = 0.0f;

    barrier();
    memoryBarrierShared();

    /* ── Per-pair contribution ──────────────────────────────────────────── */
    const uint gid_c = cam_tile + lid_c;
    const uint gid_l = light_offset + light_tile + lid_l;

    if (gid_c < n_cam_verts && gid_l < n_light_verts) {

        /* Load camera vert from shared memory. */
        const uint sc = lid_c * uint(T5_CGV_STRIDE);
        const vec3  c_pos    = vec3(s_cam[sc+0u], s_cam[sc+1u], s_cam[sc+2u]);
        const vec3  c_norm   = vec3(s_cam[sc+3u], s_cam[sc+4u], s_cam[sc+5u]);

        /* Compute spectral camera-side beta by accumulating per-band magnitudes
         * with their pre-baked colour weights (from the stride-splitter SSBO). */
        float c_beta_r = 0.0f, c_beta_g = 0.0f, c_beta_b = 0.0f;
        {
            const int nb = clamp(n_bands, 1, T5_MAX_GPU_BANDS);
            for (int _b = 0; _b < nb; ++_b) {
                const float bm = s_cam[sc + uint(CGV_BAND_BASE + _b)];
                if (bm <= 0.0f) continue;
                c_beta_r += bm * spectral_weights[_b * 3 + 0];
                c_beta_g += bm * spectral_weights[_b * 3 + 1];
                c_beta_b += bm * spectral_weights[_b * 3 + 2];
            }
        }

        const uint c_vinfo       = floatBitsToUint(s_cam[sc +  9u]);
        const uint c_flags       = floatBitsToUint(s_cam[sc +  7u]);
        const uint c_pdf_flags   = floatBitsToUint(s_cam[sc + 17u]);
        const uint c_optical_blk = floatBitsToUint(s_cam[sc + 18u]);
        const uint ci            = c_vinfo & 0xFFFFu;
        const uint cam_flat_base = gid_c - ci;

        /* Load light vert from shared memory. */
        const uint sl = lid_l * uint(T5_LGV_STRIDE);
        const vec3  l_pos  = vec3(s_light[sl+0u], s_light[sl+1u], s_light[sl+2u]);
        const vec3  l_norm = vec3(s_light[sl+3u], s_light[sl+4u], s_light[sl+5u]);
        const float l_beta = s_light[sl+10u];
        float l_beta_r = 0.0f, l_beta_g = 0.0f, l_beta_b = 0.0f;
        {
            const int nb = clamp(n_bands, 1, T5_MAX_GPU_BANDS);
            for (int _b = 0; _b < nb; ++_b) {
                const float bm = s_light[sl + uint(LGV_BAND_BASE + _b)];
                if (bm <= 0.0f) continue;
                l_beta_r += bm * spectral_weights[_b * 3 + 0];
                l_beta_g += bm * spectral_weights[_b * 3 + 1];
                l_beta_b += bm * spectral_weights[_b * 3 + 2];
            }
        }

        const uint l_vinfo       = floatBitsToUint(s_light[sl +  9u]);
        const uint l_flags       = floatBitsToUint(s_light[sl +  7u]);
        const uint l_pdf_flags   = floatBitsToUint(s_light[sl + 13u]);
        const uint l_optical_blk = floatBitsToUint(s_light[sl + 14u]);
        const uint li_v            = l_vinfo & 0xFFFFu;
        const uint light_flat_base = gid_l - li_v;

        if (vertex_connectable(c_vinfo, c_flags, c_pdf_flags, c_optical_blk) &&
            vertex_connectable(l_vinfo, l_flags, l_pdf_flags, l_optical_blk) &&
            c_beta_r + c_beta_g + c_beta_b >= 1e-15f &&
            l_beta >= 1e-15f &&
            l_beta_r + l_beta_g + l_beta_b >= 1e-15f)
        {
            /* ── Geometry term ─────────────────────────────────────────── */
            const vec3  dv    = l_pos - c_pos;
            const float dist2 = dot(dv, dv);
            if (dist2 >= 1e-12f) {
                const float dist = sqrt(dist2);
                const vec3  wc   = dv / dist;
                const float geom = abs(dot(c_norm, wc)) * abs(dot(l_norm, -wc)) / dist2;

                if (geom >= min_geom && !shadow_occluded(c_pos, l_pos)) {

                    /* BRDF fields from shared memory. */
                    const vec3  c_dir_in    = vec3(s_cam[sc + uint(CGV_DIR_IN_X)    ],
                                                   s_cam[sc + uint(CGV_DIR_IN_X)+1u ],
                                                   s_cam[sc + uint(CGV_DIR_IN_X)+2u ]);
                    const float c_diffuse_p = s_cam[sc + uint(CGV_DIFFUSE_P)];
                    const float c_ggx_alpha = s_cam[sc + uint(CGV_GGX_ALPHA)];
                    const float c_opt_jac   = s_cam[sc + uint(CGV_OPT_JACOBIAN)];

                    const vec3  l_dir_in    = vec3(s_light[sl + uint(LGV_DIR_IN_X)    ],
                                                   s_light[sl + uint(LGV_DIR_IN_X)+1u ],
                                                   s_light[sl + uint(LGV_DIR_IN_X)+2u ]);
                    const float l_diffuse_p = s_light[sl + uint(LGV_DIFFUSE_P)];
                    const float l_ggx_alpha = s_light[sl + uint(LGV_GGX_ALPHA)];
                    const float l_opt_jac   = s_light[sl + uint(LGV_OPT_JACOBIAN)];

                    /* ── Connection PDFs ────────────────────────────────── */
                    float conn_fwd = scatter_conn_pdf_area(
                        c_pos, c_norm, c_dir_in,
                        c_diffuse_p, c_ggx_alpha, c_pdf_flags, c_opt_jac,
                        l_pos, l_norm);
                    float conn_bwd = scatter_conn_pdf_area(
                        l_pos, l_norm, l_dir_in,
                        l_diffuse_p, l_ggx_alpha, l_pdf_flags, l_opt_jac,
                        c_pos, c_norm);

                    /* ── MIS density (still reads global SSBOs for chain) ─ */
                    float selected_pdf = 0.0f, denom = 0.0f;
                    if (candidate_strategy_density(
                            ci, cam_flat_base, li_v, light_flat_base,
                            conn_fwd, conn_bwd, selected_pdf, denom))
                    {
                        /* β_cam × β_light × G / denom  (selected_pdf cancels) */
                        const float contrib = geom / denom;
                        s_lum_r[tid] = c_beta_r * l_beta_r * contrib;
                        s_lum_g[tid] = c_beta_g * l_beta_g * contrib;
                        s_lum_b[tid] = c_beta_b * l_beta_b * contrib;
                    }
                }
            }
        }
    }

    barrier();
    memoryBarrierShared();

    /* ── Per-cam-vert reduction along the light-tile dimension ─────────── *
     * Thread (lid_c, 0) sums TILE_L contributions for its cam vert and     *
     * does a single atomic write — replacing TILE_L separate CAS loops.    */
    if (lid_l == 0u && gid_c < n_cam_verts) {
        float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
        for (uint j = 0u; j < uint(TILE_L); ++j) {
            const uint idx = j * uint(TILE_C) + lid_c;
            sum_r += s_lum_r[idx];
            sum_g += s_lum_g[idx];
            sum_b += s_lum_b[idx];
        }

        if (sum_r + sum_g + sum_b > 0.0f) {
            const uint sc = lid_c * uint(T5_CGV_STRIDE);
            const float c_soy = s_cam[sc + 13u];
            const float c_soz = s_cam[sc + 14u];
            const float inv_w = float(sensor_res) / (2.0f * sensor_half_w);
            const float inv_h = float(sensor_res) / (2.0f * sensor_half_h);
            const int iy = int((c_soy + sensor_half_w) * inv_w);
            const int iz = int((c_soz + sensor_half_h) * inv_h);
            /* Tile-local pixel addressing.  When tile_w > 0 the pixel buffer
             * covers only [tile_y0..tile_y0+tile_h) × [tile_x0..tile_x0+tile_w);
             * otherwise treat the whole sensor_res×sensor_res grid as one tile. */
            const int tw  = (tile_w > 0) ? tile_w  : sensor_res;
            const int th  = (tile_h > 0) ? tile_h  : sensor_res;
            const int tx0 = (tile_w > 0) ? tile_x0 : 0;
            const int ty0 = (tile_h > 0) ? tile_y0 : 0;
            if (iy >= ty0 && iy < ty0 + th && iz >= tx0 && iz < tx0 + tw) {
                const uint px  = uint((iy - ty0) * tw + (iz - tx0));
                const uint pix = uint(tw) * uint(th);
                atomic_add_float(px,          sum_r);
                atomic_add_float(px + pix,    sum_g);
                atomic_add_float(px + 2u*pix, sum_b);
            }
        }
    }
}
