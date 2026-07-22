#version 430 core
/* ────────────────────────────────────────────────────────────────────────────
 * t5_full_connect.comp.glsl  —  GPU BDPT T5 tiled connection pass
 *                               with full multi-strategy balance-heuristic MIS.
 *
 * 2-D tiled dispatch: X = camera tile, Y = light tile within current batch.
 *   Work-group size: TILE_C × TILE_L  (diagnostic 1×1 = 1 thread).
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
 *    0      T5LightVertBuf  Flat light vertices. Stride = T5_LGV_STRIDE = 56 floats.
 *             [0..2]   pos xyz
 *             [3..5]   normal xyz
 *             [6]      intBitsToFloat(tri_id) for endpoint visibility skip
 *             [7]      uintBitsToFloat(MAT_FLAG_* material flags)
 *             [8]      uintBitsToFloat(subpath_id)
 *             [9]      uintBitsToFloat(vinfo): bits[0..14]=vertex_index,
 *                        bits[16..22]=stream, bit[31]=tri_valid (tri_id>=0)
 *             [10]     intBitsToFloat(mat_idx)
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
 *    1      T5CamVertBuf    Flat camera vertices. Stride = T5_CGV_STRIDE = 72 floats.
 *             [0..2]   pos xyz
 *             [3..5]   normal xyz
 *             [6]      intBitsToFloat(tri_id) for endpoint visibility skip
 *             [7]      uintBitsToFloat(MAT_FLAG_* material flags)
 *             [8]      uintBitsToFloat(subpath_id)
 *             [9]      uintBitsToFloat(vinfo): same bit layout as LGV[9]
 *             [10..12] RESERVED (legacy luminance-RGB beta slots, B.1c dead —
 *                      camera colour comes from per-band CGV_BAND_BASE [22..53]
 *                      via cam_spectral_rgb; these are written 0 and never read)
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
#define TILE_C  16
#define TILE_L  16

layout(local_size_x = TILE_C, local_size_y = TILE_L, local_size_z = 1) in;

/* ── Stride and field offsets (mirror ray_pipeline.h) ─────────────────── */
#define T5_LGV_STRIDE    57
#define T5_CGV_STRIDE    72

#define LGV_MAT_ID        10
#define LGV_DIR_IN_X      48
#define LGV_DIFFUSE_P     51
#define LGV_GGX_ALPHA     52
#define LGV_OPT_JACOBIAN  53
#define LGV_EDGE_FWD      54
#define LGV_EDGE_BWD      55

#define CGV_MAT_ID        21
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
shared float s_cam  [TILE_C * T5_CGV_STRIDE];   /* 16×72 = 1152 floats (4.5 KB) */
shared float s_light[TILE_L * T5_LGV_STRIDE];   /* 16×57 = 912 floats (3.56 KB) */
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
    uint   light_sample_stride;
    float  light_sample_weight;
    /* pi*r^2 of the VCM merge kernel for the vertex-merging pass paired with
     * this connection pass.  0 disables the weld terms (connection-only). */
    float  vm_area;
};

/* ── Spectral colour weights (n_bands × 3): [b*3+0]=wr, [b*3+1]=wg, [b*3+2]=wb */
layout(std430, binding = 4) readonly buffer T5SpectralWeightBuf {
    float spectral_weights[];
};
layout(std430, binding = 9) coherent buffer T5DebugBuf {
    uint debug_counts[];
};

/* Diagnostic mode, normally 0:
 * 1=load only, 2=spectral, 3=geometry, 4=connection PDFs, 5=MIS, 6=shadow. */
uniform int t5_profile_mode;
uniform int t5_debug_enabled;

/* ── Flag constants (mirror bdpt_record.h / mat_flags_generated.h) ──────── */
#define MAT_FLAG_APERTURE_STOP        128u

/* BDPT_PDF_FLAG_* and MEVAL_PI are defined by the ray_material_eval.glsl.inc
 * preamble injected by load_with_shadow_bvh in the shader loader.            */

/* ── MIS chain size limit.                                                 *
 * MAX_CHAIN = 32 supports up to 15 bounces per subpath plus the two       *
 * endpoints.  Pairs exceeding this are skipped (no energy loss in practice *
 * since path depths are typically ≤ 8).                                    *
 * MAX_CHAIN_E = max edge count (N-1 for N=MAX_CHAIN vertices).            *
 * MAX_CHAIN_N = size of prefix/suffix arrays (N+1 sentinel slot).         */
#define MAX_CHAIN    32
#define MAX_CHAIN_E  31
#define MAX_CHAIN_N  33

/* ── scatter_conn_pdf_area ───────────────────────────────────────────────── *
 * Area-measure scatter PDF at vertex 'from' toward vertex 'to'.             *
 * Mirrors T5ConnContext::scatter_connection_pdf_area().                      *
 * Returns 0.0 if the vertex cannot scatter toward the target.               *
 * Solid-angle PDF is delegated to meval_scatter_pdf_sa (ray_material_eval). */
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

    float cos_to = max(0.0f, abs(dot(to_norm, -d)));
    if (cos_to <= 0.0f) return 0.0f;

    float pdf_sa = meval_scatter_pdf_sa(from_norm, from_dir_in, d,
                                        diffuse_p, ggx_alpha, pdf_flags);
    if (pdf_sa <= 0.0f) return 0.0f;

    return max(pdf_sa * cos_to / dist2 * opt_jac, 1e-12f);
}

/* ── vertex_connectable ──────────────────────────────────────────────────── *
 * GPU port of T5ConnContext::vertex_connectable().                           */
bool vertex_connectable(uint vinfo, uint vflags, uint pdf_flags, uint opt_block) {
    if ((vinfo     >> 31)                          == 0u) return false; /* tri_id < 0 */
    if ((vflags    & MAT_FLAG_APERTURE_STOP)       != 0u) return false;
    if ( opt_block                                 != 0u) return false;
    if ((pdf_flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) return false;
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

        if (t5_debug_enabled != 0) {
            atomicAdd(debug_counts[64], 1u);
            if (((pf0 | pf1) & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u)
                atomicAdd(debug_counts[65], 1u);
        }

        float p = pf * sb;
        if (p > 0.0f && !isinf(p) && !isnan(p)) out_denom += p;
    }

    /* ── Vertex-merging (VCM weld) technique densities ────────────────── *
     * When a VCM merge pass runs alongside this connection pass, the same     *
     * physical path can also be produced by welding a light photon into a     *
     * non-delta interior vertex m.  Its density on this path is               *
     *   prefix_fwd[m] * suffix_bwd[m] * (pi r^2)                              *
     * (all N-1 edges sampled; the uniform-disk kernel supplies pi r^2 —      *
     * mirrors vcm_density() in vcm_merge.comp.glsl exactly).  Omitting these  *
     * terms would double count energy between the VC and VM estimators.      *
     * vm_area == 0 when no merge pass runs, restoring pure BDPT MIS.         */
    if (vm_area > 0.0f) {
        for (uint m = 1u; m + 1u < N; ++m) {
            float pf = prefix_fwd[m];
            float sb = suffix_bwd[m];
            if (pf <= 0.0f || sb <= 0.0f) continue;

            uint vim, vfm, pfm, obm;
            if (m <= ci) {
                uint cbm = (cam_base + m) * uint(T5_CGV_STRIDE);
                vim = floatBitsToUint(cam_verts[cbm +  9u]);
                vfm = floatBitsToUint(cam_verts[cbm +  7u]);
                pfm = floatBitsToUint(cam_verts[cbm + 17u]);
                obm = floatBitsToUint(cam_verts[cbm + 18u]);
            } else {
                uint jm  = m - ci - 1u;
                uint lbm = (light_base + li_v - jm) * uint(T5_LGV_STRIDE);
                vim = floatBitsToUint(light_verts[lbm +  9u]);
                vfm = floatBitsToUint(light_verts[lbm +  7u]);
                pfm = floatBitsToUint(light_verts[lbm + 13u]);
                obm = floatBitsToUint(light_verts[lbm + 14u]);
            }
            /* A weld requires a real, unblocked, non-delta receiving surface. */
            if (!vertex_connectable(vim, vfm, pfm, obm)) continue;
            if ((pfm & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) continue;

            float p = pf * sb * vm_area;
            if (p > 0.0f && !isinf(p) && !isnan(p)) out_denom += p;
        }
    }

    if (out_denom < out_selected_pdf)
        out_denom = out_selected_pdf;
    return out_selected_pdf > 1e-20f &&
           out_denom        > 1e-20f &&
           !isinf(out_denom) && !isnan(out_denom);
}

void cam_spectral_rgb(uint sc, out float r, out float g, out float b) {
    r = 0.0f;
    g = 0.0f;
    b = 0.0f;
    const int nb = clamp(n_bands, 1, T5_MAX_GPU_BANDS);
    for (int _b = 0; _b < nb; ++_b) {
        const float bm = s_cam[sc + uint(CGV_BAND_BASE + _b)];
        if (bm <= 0.0f) continue;
        r += bm * spectral_weights[_b * 3 + 0];
        g += bm * spectral_weights[_b * 3 + 1];
        b += bm * spectral_weights[_b * 3 + 2];
    }
}

float cam_beta_sum(uint sc) {
    float total = 0.0f;
    const int nb = clamp(n_bands, 1, T5_MAX_GPU_BANDS);
    for (int _b = 0; _b < nb; ++_b) {
        const float bm = s_cam[sc + uint(CGV_BAND_BASE + _b)];
        if (bm > 0.0f) total += bm;
    }
    return total;
}

void light_spectral_rgb(uint sl, out float r, out float g, out float b) {
    r = 0.0f;
    g = 0.0f;
    b = 0.0f;
    const int nb = clamp(n_bands, 1, T5_MAX_GPU_BANDS);
    for (int _b = 0; _b < nb; ++_b) {
        const float bm = s_light[sl + uint(LGV_BAND_BASE + _b)];
        if (bm <= 0.0f) continue;
        r += bm * spectral_weights[_b * 3 + 0];
        g += bm * spectral_weights[_b * 3 + 1];
        b += bm * spectral_weights[_b * 3 + 2];
    }
}

float light_beta_sum(uint sl) {
    float total = 0.0f;
    const int nb = clamp(n_bands, 1, T5_MAX_GPU_BANDS);
    for (int _b = 0; _b < nb; ++_b) {
        const float bm = s_light[sl + uint(LGV_BAND_BASE + _b)];
        if (bm > 0.0f) total += bm;
    }
    return total;
}

/* ── Refractive (glass) connection handling ──────────────────────────────── *
 * A BDPT connection segment that crosses one or more refractive interfaces is
 * NOT a vacuum path: at every glass surface the light is partially reflected
 * (Fresnel) and, beyond the critical angle, totally internally reflected so no
 * transmitted connection exists at all.  bvh_shadow.glsl.inc deliberately lets
 * the segment pass the *visibility* test through glass; here we recover the
 * physics it skipped and weight the connection by the real per-interface
 * Fresnel transmittance, using the identical Snell/Fresnel math and per-band
 * ior_real as the T3 forward pass (meval_ior_real(mat, band)).
 * Matching T3 exactly is what keeps the multi-strategy MIS estimator unbiased:
 * a connection that traverses the lens is attenuated by the same Fresnel factor
 * a forward-traced ray would lose at those surfaces, and a band that hits TIR
 * at any interface drops out of that band's connection (no transmitted path).
 *
 * The crossing geometry (incidence cosine, medium materials on each side) is
 * gathered ONCE per vertex pair; the Fresnel transmittance is then evaluated
 * PER band (glass_transmittance_band), since each wavelength refracts at its
 * own Snell angle and may individually total-internally-reflect.              */
#define GLASS_MAX_CROSS 8

int gather_glass_crossings(vec3 p0, vec3 p1, int skip0, int skip1,
                           out float cos_i[GLASS_MAX_CROSS],
                           out int   mat_from[GLASS_MAX_CROSS],
                           out int   mat_to[GLASS_MAX_CROSS])
{
    vec3  d    = p1 - p0;
    float dist = length(d);
    if (dist < 2.0 * SHADOW_T_SELF) return 0;
    vec3 dir     = d / dist;
    vec3 inv_dir = vec3(1.0) / dir;
    float t_max  = dist - SHADOW_T_SELF;

    int n = 0;
    int stack[SHADOW_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  /* root */

    while (sp > 0) {
        int node = stack[--sp];
        if (node < 0) continue;
        if (!_shd_aabb_hit(_shd_bvh_lo(node), _shd_bvh_hi(node),
                           p0, inv_dir, t_max)) continue;

        int left = _shd_bvh_left(node);
        if (left == -1) {
            int ts = _shd_bvh_tri_start(node);
            int te = _shd_bvh_tri_end(node);
            for (int k = ts; k < te; ++k) {
                int ti = shadow_tri_ids[k];
                if (ti == skip0 || ti == skip1) continue;
                int tb = ti * TRI_FULL_STRIDE;
                uint tf = floatBitsToUint(shadow_trifull[tb + 12]);
                /* Only refractive surfaces matter here; opaque blockers are
                 * handled by the separate shadow_occluded_except() test. */
                if ((tf & SHADOW_MAT_FLAG_TRANSMISSIVE) == 0u) continue;
                vec3 v0 = vec3(shadow_trifull[tb],   shadow_trifull[tb+1], shadow_trifull[tb+2]);
                vec3 e1 = vec3(shadow_trifull[tb+3], shadow_trifull[tb+4], shadow_trifull[tb+5]);
                vec3 e2 = vec3(shadow_trifull[tb+6], shadow_trifull[tb+7], shadow_trifull[tb+8]);
                float t = _shd_mt(v0, e1, e2, p0, dir);
                if (t <= SHADOW_T_SELF || t >= t_max) continue;
                if (n >= GLASS_MAX_CROSS) continue;
                vec3 tn = normalize(vec3(shadow_trifull[tb+9],
                                         shadow_trifull[tb+10],
                                         shadow_trifull[tb+11]));
                int med_pos = floatBitsToInt(shadow_trifull[tb + 14]);
                int med_neg = floatBitsToInt(shadow_trifull[tb + 15]);
                /* front: segment travels against the geometric normal, i.e. it
                 * enters from the +normal (med_pos) side — same convention as T3. */
                bool front = dot(dir, tn) < 0.0;
                cos_i[n]    = abs(dot(dir, tn));
                mat_from[n] = front ? med_pos : med_neg;
                mat_to[n]   = front ? med_neg : med_pos;
                n++;
            }
        } else {
            if (sp + 1 < SHADOW_STACK_SIZE) {
                stack[sp++] = left;
                stack[sp++] = _shd_bvh_right(node);
            }
        }
    }
    return n;
}

/* Per-band Fresnel amplitude-transmittance product across all gathered glass
 * interfaces, evaluated from band `band`'s ior_real.  Returns 0 if ANY interface
 * is in total internal reflection FOR THIS BAND — there is then no transmitted
 * connection for that wavelength, exactly as T3's per-band snell_refract failure
 * sends that band entirely into reflection.
 *
 * This MUST mirror the T3 forward pass to keep the BDPT estimator unbiased: T3
 * now refracts per band (each band uses its own ior_real / Snell angle) and
 * applies a per-band amplitude transmittance ts_b = sqrt(1-R_b).  The T5 betas
 * are amplitude magnitudes (the pixel reduction sums magnitude products
 * linearly), so sqrt(1-R_b) — not the power transmittance (1-R_b) — is correct. */
float glass_transmittance_band(int band, int n_cross,
                               float cos_i[GLASS_MAX_CROSS],
                               int   mat_from[GLASS_MAX_CROSS],
                               int   mat_to[GLASS_MAX_CROSS])
{
    float T = 1.0f;
    for (int i = 0; i < n_cross; ++i) {
        float n1 = meval_ior_real(mat_from[i], band);
        float n2 = meval_ior_real(mat_to[i],   band);
        float ci = clamp(cos_i[i], 0.0f, 1.0f);
        float sin2_t = (n1 / n2) * (n1 / n2) * (1.0f - ci * ci);
        if (sin2_t > 1.0f) return 0.0f;               /* TIR for this band */
        float ct = sqrt(max(0.0f, 1.0f - sin2_t));
        float R  = meval_fresnel_R(ci, ct, n1, n2);
        T *= sqrt(max(0.0f, 1.0f - R));               /* amplitude transmittance */
        if (T <= 0.0f) return 0.0f;
    }
    return T;
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

void connection_spectral_rgb(uint sc, uint sl,
                             vec3 c_norm, vec3 c_dir_in, uint c_pdf_flags,
                             int c_mat, vec3 c_to_l,
                             vec3 l_norm, vec3 l_dir_in, uint l_pdf_flags,
                             int l_mat, vec3 l_to_c,
                             int n_glass,
                             float g_cos[GLASS_MAX_CROSS],
                             int   g_from[GLASS_MAX_CROSS],
                             int   g_to[GLASS_MAX_CROSS],
                             out float r, out float g, out float b) {
    r = 0.0f;
    g = 0.0f;
    b = 0.0f;

    const int nb = clamp(n_bands, 1, T5_MAX_GPU_BANDS);
    bool any_overlap = false;
    bool any_camera_response = false;
    bool any_light_response = false;
    bool any_glass_pass = false;
    const uint spectral_sample_id = floatBitsToUint(s_cam[sc + 62u]);
    const float exact_frequency_hz = s_cam[sc + 63u];
    const float exact_spectral_pdf = max(s_cam[sc + 64u], 1.0e-30f);
    for (int _b = 0; _b < nb; ++_b) {
        const float cb = s_cam[sc + uint(CGV_BAND_BASE + _b)];
        const float lb = s_light[sl + uint(LGV_BAND_BASE + _b)];
        if (cb <= 0.0f || lb <= 0.0f) continue;
        any_overlap = true;

        /* Per-band Fresnel transmittance of any glass the straight connection
         * segment crosses (1.0 in vacuum).  Evaluated per band because each
         * wavelength refracts at its own Snell angle and may individually hit
         * total internal reflection — matching the per-band T3 forward split.
         * A band that TIRs anywhere along the segment contributes nothing. */
        const float glassT = (n_glass > 0)
            ? glass_transmittance_band(_b, n_glass, g_cos, g_from, g_to)
            : 1.0f;
        if (glassT <= 0.0f) continue;
        any_glass_pass = true;

        const float cf = meval_endpoint_response(
            c_mat, _b, c_pdf_flags, c_norm, c_dir_in, c_to_l);
        const float lf = meval_endpoint_response(
            l_mat, _b, l_pdf_flags, l_norm, l_dir_in, l_to_c);
        if (cf > 0.0f) any_camera_response = true;
        if (lf > 0.0f) any_light_response = true;
        const float v = cb * lb * cf * lf * glassT;
        if (v <= 0.0f || isnan(v) || isinf(v)) continue;
        if (spectral_sample_id != 0u && exact_frequency_hz > 0.0f) {
            const vec3 exact = frequency_rgb(exact_frequency_hz)
                             / exact_spectral_pdf;
            r += v * exact.r;
            g += v * exact.g;
            b += v * exact.b;
        } else {
            r += v * spectral_weights[_b * 3 + 0];
            g += v * spectral_weights[_b * 3 + 1];
            b += v * spectral_weights[_b * 3 + 2];
        }
    }
    if (t5_debug_enabled != 0 && any_overlap) atomicAdd(debug_counts[9], 1u);
    if (t5_debug_enabled != 0 && any_glass_pass) atomicAdd(debug_counts[10], 1u);
    if (t5_debug_enabled != 0 && any_camera_response) atomicAdd(debug_counts[11], 1u);
    if (t5_debug_enabled != 0 && any_light_response) atomicAdd(debug_counts[12], 1u);
}

/* ── Float atomic add via CAS spin-loop ─────────────────────────────────── *
 * Avoids GL_EXT_shader_atomic_float.  Standard on all GL 4.3+ hardware.
 * Unbounded on purpose: a fixed retry cap silently dropped energy on the
 * brightest, highest-contention pixels.  compareAndSwap guarantees global
 * progress, so the loop always terminates. */
void atomic_add_float(uint idx, float val) {
    if (val <= 0.0f || isnan(val) || isinf(val)) return;
    uint expected = pixel_accum[idx];
    while (true) {
        float fexp    = uintBitsToFloat(expected);
        uint  desired = floatBitsToUint(fexp + val);
        uint  actual  = atomicCompSwap(pixel_accum[idx], expected, desired);
        if (actual == expected) return;
        expected = actual;
    }
}

void splat_sensor_tent(float sensor_y, float sensor_z,
                       float val_r, float val_g, float val_b) {
    if (val_r + val_g + val_b <= 0.0f) return;
    const int tw  = (tile_w > 0) ? tile_w  : sensor_res;
    const int th  = (tile_h > 0) ? tile_h  : sensor_res;
    const int tx0 = (tile_w > 0) ? tile_x0 : 0;
    const int ty0 = (tile_h > 0) ? tile_y0 : 0;
    const uint pix = uint(tw) * uint(th);

    const float inv_w = float(sensor_res) / (2.0f * sensor_half_w);
    const float inv_h = float(sensor_res) / (2.0f * sensor_half_h);
    const float fy = (sensor_y + sensor_half_w) * inv_w - 0.5f;
    const float fz = (sensor_z + sensor_half_h) * inv_h - 0.5f;
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
            if (w > 0.0f && iy >= ty0 && iy < ty0 + th && iz >= tx0 && iz < tx0 + tw)
                wsum += w;
            ++k;
        }
    }
    if (wsum <= 0.0f) return;

    for (int i = 0; i < 4; ++i) {
        const int iy = ys[i];
        const int iz = zs[i];
        const float w = ws[i] / wsum;
        if (w <= 0.0f || iy < ty0 || iy >= ty0 + th || iz < tx0 || iz >= tx0 + tw)
            continue;
        const uint px = uint((iy - ty0) * tw + (iz - tx0));
        atomic_add_float(px,          val_r * w);
        atomic_add_float(px + pix,    val_g * w);
        atomic_add_float(px + 2u*pix, val_b * w);
    }
}

/* ── Main ────────────────────────────────────────────────────────────────── */
uint t5_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint sampled_light_vertex(uint logical, uint camera_tile) {
    if (light_sample_stride == 0u || n_light_verts == 0u)
        return logical;
    /* Each camera tile receives a randomized rotation of a stratified walk
     * through the complete packed light-vertex population.  Every logical
     * sample is marginally uniform; light_sample_weight supplies its inverse
     * probability on the host. */
    const uint rotation = t5_hash_u32(camera_tile ^ n_light_verts ^ 0x9e3779b9u);
    return (logical * light_sample_stride + rotation) % n_light_verts;
}

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

    /* Camera tile: 16 × 72 = 1152 floats → at most 5 loads/thread. */
    for (uint i = tid; i < uint(TILE_C) * uint(T5_CGV_STRIDE); i += total) {
        const uint v = i / uint(T5_CGV_STRIDE);
        const uint f = i % uint(T5_CGV_STRIDE);
        const uint g = cam_tile + v;
        s_cam[i] = (g < n_cam_verts)
            ? cam_verts[g * uint(T5_CGV_STRIDE) + f]
            : 0.0f;
    }

    /* Light tile: 16 × 57 = 912 floats → at most 4 loads/thread. */
    for (uint i = tid; i < uint(TILE_L) * uint(T5_LGV_STRIDE); i += total) {
        const uint v      = i / uint(T5_LGV_STRIDE);
        const uint f      = i % uint(T5_LGV_STRIDE);
        const uint logical = light_offset + light_tile + v;
        const uint g_glob = sampled_light_vertex(logical, cam_tile);
        const bool logical_valid = logical >= light_offset &&
                                   logical < light_offset + light_batch_size;
        s_light[i] = (logical_valid && g_glob < n_light_verts)
            ? light_verts[g_glob * uint(T5_LGV_STRIDE) + f]
            : 0.0f;
    }

    /* Init per-invocation accumulators. */
    s_lum_r[tid] = 0.0f;
    s_lum_g[tid] = 0.0f;
    s_lum_b[tid] = 0.0f;

    barrier();
    memoryBarrierShared();
    if (t5_profile_mode == 1) return;

    /* ── Per-pair contribution ──────────────────────────────────────────── */
    const uint gid_c = cam_tile + lid_c;
    const uint logical_l = light_offset + light_tile + lid_l;
    const uint gid_l = sampled_light_vertex(logical_l, cam_tile);

    if (gid_c < n_cam_verts &&
        logical_l >= light_offset && logical_l < light_offset + light_batch_size &&
        gid_l < n_light_verts) {
        if (t5_debug_enabled != 0) atomicAdd(debug_counts[0], 1u);

        /* Load camera vert from shared memory. */
        const uint sc = lid_c * uint(T5_CGV_STRIDE);
        const vec3  c_pos    = vec3(s_cam[sc+0u], s_cam[sc+1u], s_cam[sc+2u]);
        const vec3  c_norm   = vec3(s_cam[sc+3u], s_cam[sc+4u], s_cam[sc+5u]);

        /* Compute spectral camera-side beta by accumulating per-band magnitudes
         * with their pre-baked colour weights (from the stride-splitter SSBO). */
        float c_beta_r = 0.0f, c_beta_g = 0.0f, c_beta_b = 0.0f;
        cam_spectral_rgb(sc, c_beta_r, c_beta_g, c_beta_b);
        const float c_beta_sum = cam_beta_sum(sc);

        const int  c_tri_id      = floatBitsToInt(s_cam[sc +  6u]);
        const int  c_mat_id      = floatBitsToInt(s_cam[sc + uint(CGV_MAT_ID)]);
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
        const float l_beta_sum = light_beta_sum(sl);

        const int  l_tri_id      = floatBitsToInt(s_light[sl +  6u]);
        const int  l_mat_id      = floatBitsToInt(s_light[sl + uint(LGV_MAT_ID)]);
        const uint l_vinfo       = floatBitsToUint(s_light[sl +  9u]);
        const uint l_flags       = floatBitsToUint(s_light[sl +  7u]);
        const uint l_pdf_flags   = floatBitsToUint(s_light[sl + 13u]);
        const uint l_optical_blk = floatBitsToUint(s_light[sl + 14u]);
        const uint li_v            = l_vinfo & 0xFFFFu;
        const uint light_flat_base = gid_l - li_v;

        if (t5_profile_mode == 2) return;

        const bool c_ok = vertex_connectable(c_vinfo, c_flags, c_pdf_flags, c_optical_blk);
        const bool l_ok = vertex_connectable(l_vinfo, l_flags, l_pdf_flags, l_optical_blk);
        const uint c_spectral_sample = floatBitsToUint(s_cam[sc + 62u]);
        const uint l_spectral_sample = floatBitsToUint(s_light[sl + 56u]);
        const bool spectral_pair_ok =
            (c_spectral_sample == 0u && l_spectral_sample == 0u)
            || (c_spectral_sample != 0u && c_spectral_sample == l_spectral_sample);
        if (t5_debug_enabled != 0 && (c_vinfo >> 31) != 0u) {
            const uint region_bin = c_pos.x < 1.0f ? 0u : (c_pos.x > 3.0f ? 2u : 1u);
            atomicAdd(debug_counts[87u + region_bin], 1u);
            if (c_ok)
                atomicAdd(debug_counts[90u + region_bin], 1u);
            if ((c_pdf_flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u)
                atomicAdd(debug_counts[93u + region_bin], 1u);
        }
        if (t5_debug_enabled != 0 && c_ok) atomicAdd(debug_counts[1], 1u);
        if (t5_debug_enabled != 0 && l_ok) atomicAdd(debug_counts[2], 1u);
        if (t5_debug_enabled != 0 && c_ok &&
            (c_pdf_flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u)
            atomicAdd(debug_counts[66], 1u);
        if (t5_debug_enabled != 0 && l_ok &&
            (l_pdf_flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u)
            atomicAdd(debug_counts[67], 1u);
        if (c_ok && l_ok && spectral_pair_ok
            && c_beta_sum >= 1e-15f && l_beta_sum >= 1e-15f)
        {
            if (t5_debug_enabled != 0) atomicAdd(debug_counts[3], 1u);
            if (t5_debug_enabled != 0 && li_v == 0u) atomicAdd(debug_counts[13], 1u);
            /* ── Geometry term ─────────────────────────────────────────── */
            const vec3  dv    = l_pos - c_pos;
            const float dist2 = dot(dv, dv);
            if (dist2 >= 1e-12f) {
                const float dist = sqrt(dist2);
                const vec3  wc   = dv / dist;
                const float geom = abs(dot(c_norm, wc)) * abs(dot(l_norm, -wc)) / dist2;
                if (t5_profile_mode == 3) return;

                /* B.4b: `min_geom` is a host-exposed QUALITY KNOB, not a fixed
                 * constant.  It makes empty space cheap, but it also biases out
                 * legitimate faint long-range / grazing connections (low geom).
                 * Tune via RayTracer.set_t5_min_geom (cfg.t5_min_geom): lower
                 * preserves faint distant transport, higher is faster/noisier. */
                if (geom >= min_geom) {
                    if (t5_debug_enabled != 0) atomicAdd(debug_counts[4], 1u);
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
                    if (t5_debug_enabled != 0 && conn_fwd > 0.0f && conn_bwd > 0.0f)
                        atomicAdd(debug_counts[5], 1u);
                    if (t5_profile_mode == 4) return;

                    /* ── MIS density (still reads global SSBOs for chain) ─ */
                    float selected_pdf = 0.0f, denom = 0.0f;
                    bool mis_ok = candidate_strategy_density(
                            ci, cam_flat_base, li_v, light_flat_base,
                            conn_fwd, conn_bwd, selected_pdf, denom);
                    if (t5_profile_mode == 5) return;
                    if (mis_ok)
                    {
                        if (t5_debug_enabled != 0) atomicAdd(debug_counts[6], 1u);
                        /* Gather the refractive interfaces this connection segment
                         * crosses once (geometry is band-independent); the achromatic
                         * Fresnel transmittance / TIR test is applied inside
                         * connection_spectral_rgb so glass attenuates the connection
                         * exactly as T3's forward refraction does, instead of being
                         * silently ignored. */
                        float g_cos[GLASS_MAX_CROSS];
                        int   g_from[GLASS_MAX_CROSS];
                        int   g_to[GLASS_MAX_CROSS];
                        int   n_glass = gather_glass_crossings(
                            c_pos, l_pos, c_tri_id, l_tri_id, g_cos, g_from, g_to);

                        float spec_r = 0.0f, spec_g = 0.0f, spec_b = 0.0f;
                        connection_spectral_rgb(
                            sc, sl,
                            c_norm, c_dir_in, c_pdf_flags, c_mat_id, wc,
                            l_norm, l_dir_in, l_pdf_flags, l_mat_id, -wc,
                            n_glass, g_cos, g_from, g_to,
                            spec_r, spec_g, spec_b);

                        /* ── B.0 INVARIANT: betas are carried UN-NORMALISED ──
                         * `contrib = f / denom` is the balance-heuristic MIS
                         * estimator f/Σpᵢ.  It is unbiased ONLY because the
                         * spectral betas (spec_r/g/b) arrive un-normalised:
                         * T3 propagates amplitude as `amp *= reflectance` with
                         * NO division by the scatter PDF (see ray_material.comp
                         * .glsl scatter sites).  All sampling-PDF normalisation
                         * is DEFERRED to this single `/ denom` here.
                         * DO NOT divide betas by a sampling PDF in T3: doing so
                         * double-counts and MUST be paired with removing this
                         * `/ denom`.  The two sites are a matched pair. */
                        const float scale = (geom / denom) * max(light_sample_weight, 1.0f);
                        const float contrib = (spec_r + spec_g + spec_b) * scale;
                        if (t5_debug_enabled != 0 && contrib > 0.0f && !isinf(contrib) && !isnan(contrib)) {
                            int ebin = clamp((int(floor(log2(contrib))) + 64) / 4, 0, 31);
                            atomicAdd(debug_counts[16 + ebin], 1u);
                        }
                        if (t5_debug_enabled != 0 && denom > 0.0f && !isinf(denom) && !isnan(denom)) {
                            int dbin = clamp((int(floor(log2(denom))) + 64) / 8, 0, 15);
                            atomicAdd(debug_counts[48 + dbin], 1u);
                        }
                        if (t5_debug_enabled != 0 && contrib > 0.0f && !isinf(contrib) && !isnan(contrib))
                            atomicAdd(debug_counts[7], 1u);
                        if (t5_debug_enabled != 0 && contrib > 0.0f && !isinf(contrib) && !isnan(contrib)) {
                            if (c_pos.x < 1.0f) atomicAdd(debug_counts[76], 1u);
                            if (c_pos.x > 3.0f) atomicAdd(debug_counts[77], 1u);
                            if (li_v == 0u) atomicAdd(debug_counts[78], 1u);
                        }
                        if (t5_debug_enabled != 0 && li_v == 0u && contrib > 0.0f && !isinf(contrib) && !isnan(contrib))
                            atomicAdd(debug_counts[14], 1u);
                        if (contrib > 0.0f && !isinf(contrib) && !isnan(contrib)) {
                            float blocker_t = -1.0f;
                            float blocker_dist = 0.0f;
                            const int blocker_tri = shadow_first_blocker_except(
                                c_pos, l_pos, c_tri_id, l_tri_id,
                                blocker_t, blocker_dist);
                            const bool occluded = blocker_tri >= 0;
                            if (t5_debug_enabled != 0 && occluded) {
                                uint blocker_flags = floatBitsToUint(
                                    shadow_trifull[blocker_tri * TRI_FULL_STRIDE + 12]);
                                if ((blocker_flags & SHADOW_MAT_FLAG_APERTURE_STOP) != 0u)
                                    atomicAdd(debug_counts[68], 1u);
                                else if ((blocker_flags & 1u) != 0u)
                                    atomicAdd(debug_counts[69], 1u);
                                else if ((blocker_flags & SHADOW_MAT_FLAG_TRANSMISSIVE) != 0u)
                                    atomicAdd(debug_counts[70], 1u);
                                else
                                    atomicAdd(debug_counts[71], 1u);
                                if (blocker_t < 1.0e-3f)
                                    atomicAdd(debug_counts[72], 1u);
                                if (blocker_dist - blocker_t < 1.0e-3f)
                                    atomicAdd(debug_counts[73], 1u);
                                atomicMin(debug_counts[74], uint(blocker_tri));
                                atomicMax(debug_counts[75], uint(blocker_tri));
                                if (c_pos.x < 1.0f) atomicAdd(debug_counts[79], 1u);
                                if (c_pos.x < 1.0f && li_v == 0u)
                                    atomicAdd(debug_counts[80], 1u);
                                atomicAdd(debug_counts[81], uint(max(c_pos.x, 0.0f) * 1000.0f));
                                int blocker_mat = floatBitsToInt(
                                    shadow_trifull[blocker_tri * TRI_FULL_STRIDE + 13]);
                                if (blocker_mat == 0) atomicAdd(debug_counts[82], 1u);
                                else if (blocker_mat == 1) atomicAdd(debug_counts[83], 1u);
                                else if (blocker_mat == 2) atomicAdd(debug_counts[84], 1u);
                                else if (blocker_mat == 8) atomicAdd(debug_counts[85], 1u);
                                else atomicAdd(debug_counts[86], 1u);
                            }
                            if (t5_profile_mode == 6) return;
                            if (!occluded) {
                                if (t5_debug_enabled != 0) atomicAdd(debug_counts[8], 1u);
                                if (t5_debug_enabled != 0 && li_v == 0u) atomicAdd(debug_counts[15], 1u);
                                s_lum_r[tid] = spec_r * scale;
                                s_lum_g[tid] = spec_g * scale;
                                s_lum_b[tid] = spec_b * scale;
                            }
                        }
                    }
                }
            }
        }
    }

    /* Profile exits above are pair-dependent.  Keep the matching fall-through
     * exit before the reduction barrier so partially populated edge tiles do
     * not leave inactive lanes waiting at a barrier after active lanes return. */
    if (t5_profile_mode != 0) return;
    barrier();
    memoryBarrierShared();

    /* ── Per-cam-vert reduction along the light-tile dimension ─────────── *
     * Thread (lid_c, 0) sums TILE_L contributions for its cam vert and     *
     * does a single atomic write — replacing TILE_L separate CAS loops.    */
    if (lid_l == 0u && gid_c < n_cam_verts) {
        const uint sc = lid_c * uint(T5_CGV_STRIDE);
        float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
        for (uint j = 0u; j < uint(TILE_L); ++j) {
            const uint idx = j * uint(TILE_C) + lid_c;
            sum_r += s_lum_r[idx];
            sum_g += s_lum_g[idx];
            sum_b += s_lum_b[idx];
        }

        if (sum_r + sum_g + sum_b > 0.0f) {
            const float c_soy = s_cam[sc + 13u];
            const float c_soz = s_cam[sc + 14u];
            splat_sensor_tent(c_soy, c_soz, sum_r, sum_g, sum_b);
        }
    }
}
