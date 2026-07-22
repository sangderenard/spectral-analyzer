#version 430 core
/* ─────────────────────────────────────────────────────────────────────────────
 * batch_score.comp.glsl  —  cheap triage prepass for T5 BDPT connection
 *
 * Runs BEFORE t5_full_connect's all-pairs grind.  One work-group scores ONE
 * (cam-batch × light-batch) work unit — i.e. exactly one of the (c,b) dispatch
 * cells in ray_tracer.cpp's T5 loop.  Each unit corresponds to a pair of
 * Morton-coherent vertex batches, so its score is a coherent "is this region
 * pair worth grinding now?" signal.
 *
 * It does NOT do MIS chains and does NOT trace shadow rays.  It samples K random
 * pairs from the unit and reduces three cheap signals, then writes:
 *
 *   scores[(c * n_lbatch + b) * 4 + 0] = A  predicted contribution (mean energy)
 *   scores[(c * n_lbatch + b) * 4 + 1] = B  connectable density   (fraction)
 *   scores[(c * n_lbatch + b) * 4 + 2] = C  peak geom potential    (max geom)
 *   scores[(c * n_lbatch + b) * 4 + 3] = S  wA*A + wB*B + wC*C     (combined)
 *
 * A, B, C are written RAW (un-normalised) so the host / histogram tool can see
 * their true ranges and you can pick coefficients with eyes open.  The combined
 * S uses the wA/wB/wC uniforms plus optional per-signal normalisers so the cut
 * point between "run now" and the never-discarded idle pile is a simple sort.
 *
 * Granularity note: this scores the (cam-batch, light-batch) PAIR. A marginal
 * per-Morton-batch score (e.g. "is light-batch b worth anything against the
 * whole frame?") is just a reduction of column b across all c, computed host
 * side from the same buffer — no second shader needed.
 *
 * ── Bindings (share T5's vertex SSBOs; add two of our own) ──────────────────
 *   0  T5LightVertBuf  light_verts[]   readonly   stride T5_LGV_STRIDE = 57
 *   1  T5CamVertBuf    cam_verts[]     readonly   stride T5_CGV_STRIDE = 72
 *   9  BatchScoreBuf   scores[]        writeonly   4 floats per (c,b)
 * ───────────────────────────────────────────────────────────────────────────*/

layout(local_size_x = 64) in;

/* ── Strides / field offsets (mirror t5_full_connect.comp.glsl) ───────────── */
#define T5_LGV_STRIDE   57
#define T5_CGV_STRIDE   72
#define CGV_BAND_BASE   22
#define LGV_BAND_BASE   16
#define MAX_GPU_BANDS   32
#define MAT_FLAG_APERTURE_STOP  128u

layout(std430, binding = 0) readonly  buffer T5LightVertBuf { float light_verts[]; };
layout(std430, binding = 1) readonly  buffer T5CamVertBuf   { float cam_verts[];   };
layout(std430, binding = 9) writeonly buffer BatchScoreBuf  { float scores[];      };

/* ── Geometry of the batch grid (set per tile by the host) ────────────────── */
uniform uint  n_cam_verts;      /* total cam verts for this tile (filtered)     */
uniform uint  n_light_verts;    /* total light verts                            */
uniform uint  cam_batch_sz;     /* T5_CAM_BATCH   (e.g. 8192)                    */
uniform uint  light_batch_sz;   /* T5_LIGHT_BATCH (e.g. 4096)                    */
uniform int   n_bands;

/* ── Scoring controls ─────────────────────────────────────────────────────── */
uniform uint  k_samples;        /* pairs sampled per work unit (e.g. 256)       */
uniform float min_geom;         /* same floor t5_full_connect uses for B        */
uniform uint  seed;             /* per-frame jitter so successive runs decorrelate */

/* Combined-score coefficients + optional normalisers (host can leave norm=1).
 * S = wA*(A/normA) + wB*(B/normB) + wC*(C/normC).  Raw A/B/C are still emitted. */
uniform float wA, wB, wC;
uniform float normA, normB, normC;

/* ── Shared reduction scratch ─────────────────────────────────────────────── */
shared float s_sumA [64];   /* Σ beta_c*beta_l*geom over connectable samples     */
shared uint  s_cntC [64];   /* # samples that were connectable AND geom>=min_geom */
shared uint  s_cntS [64];   /* # samples actually drawn (bounds-valid)            */
shared float s_maxG [64];   /* max geom seen                                      */

/* ── PCG-ish integer hash for cheap, deterministic sampling ───────────────── */
uint hash_u32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16; return x;
}

/* vertex_connectable, identical predicate to the connection pass. */
bool connectable(uint vinfo, uint vflags, uint opt_block) {
    if ((vinfo >> 31) == 0u)                       return false; /* tri_id < 0     */
    if ((vflags & MAT_FLAG_APERTURE_STOP) != 0u)   return false; /* aperture stop  */
    if (opt_block != 0u)                           return false; /* optical block  */
    return true;
}

float spectral_beta_product(uint cb, uint lb) {
    float t = 0.0;
    int nb = clamp(n_bands, 1, MAX_GPU_BANDS);
    for (int b = 0; b < nb; ++b) {
        float c = cam_verts[cb + uint(CGV_BAND_BASE + b)];
        float l = light_verts[lb + uint(LGV_BAND_BASE + b)];
        if (c > 0.0 && l > 0.0) t += c * l;
    }
    return t;
}

void main() {
    const uint c   = gl_WorkGroupID.x;            /* cam-batch index   */
    const uint b   = gl_WorkGroupID.y;            /* light-batch index */
    const uint nlb = gl_NumWorkGroups.y;          /* total light batches */
    const uint tid = gl_LocalInvocationID.x;

    /* Resolve this unit's vertex windows. */
    const uint cam_off   = c * cam_batch_sz;
    const uint light_off = b * light_batch_sz;
    const uint cam_n   = (cam_off   < n_cam_verts)   ? min(cam_batch_sz,   n_cam_verts   - cam_off)   : 0u;
    const uint light_n = (light_off < n_light_verts) ? min(light_batch_sz, n_light_verts - light_off) : 0u;

    float locA = 0.0;
    uint  locC = 0u;
    uint  locS = 0u;
    float locG = 0.0;

    if (cam_n > 0u && light_n > 0u) {
        /* Each thread draws k_samples/64 pairs (at least the remainder split). */
        const uint per_thread = (k_samples + 63u) / 64u;
        for (uint s = 0u; s < per_thread; ++s) {
            const uint h0 = hash_u32(seed ^ (c * 73856093u) ^ (b * 19349663u)
                                          ^ ((tid * per_thread + s) * 83492791u));
            const uint h1 = hash_u32(h0 * 2654435761u + 0x9e3779b9u);

            const uint ci = cam_off   + (h0 % cam_n);
            const uint li = light_off + (h1 % light_n);
            locS += 1u;

            const uint cb = ci * uint(T5_CGV_STRIDE);
            const uint lb = li * uint(T5_LGV_STRIDE);

            /* Connectability — cheap field reads, no chain walk. */
            const uint c_vinfo = floatBitsToUint(cam_verts  [cb + 9u]);
            const uint c_flags = floatBitsToUint(cam_verts  [cb + 7u]);
            const uint c_oblk  = floatBitsToUint(cam_verts  [cb + 18u]);
            const uint l_vinfo = floatBitsToUint(light_verts[lb + 9u]);
            const uint l_flags = floatBitsToUint(light_verts[lb + 7u]);
            const uint l_oblk  = floatBitsToUint(light_verts[lb + 14u]);
            if (!connectable(c_vinfo, c_flags, c_oblk)) continue;
            if (!connectable(l_vinfo, l_flags, l_oblk)) continue;

            /* Geometry term — identical formula to t5_full_connect. */
            const vec3 c_pos  = vec3(cam_verts  [cb+0u], cam_verts  [cb+1u], cam_verts  [cb+2u]);
            const vec3 c_norm = vec3(cam_verts  [cb+3u], cam_verts  [cb+4u], cam_verts  [cb+5u]);
            const vec3 l_pos  = vec3(light_verts[lb+0u], light_verts[lb+1u], light_verts[lb+2u]);
            const vec3 l_norm = vec3(light_verts[lb+3u], light_verts[lb+4u], light_verts[lb+5u]);

            const vec3  dv    = l_pos - c_pos;
            const float dist2 = dot(dv, dv);
            if (dist2 < 1e-12) continue;
            const float dist  = sqrt(dist2);
            const vec3  wc    = dv / dist;
            const float geom  = abs(dot(c_norm, wc)) * abs(dot(l_norm, -wc)) / dist2;
            if (isnan(geom) || isinf(geom) || geom <= 0.0) continue;

            locG = max(locG, geom);                 /* C: peak geom potential */
            if (geom >= min_geom) {
                locC += 1u;                         /* B: connectable+geom count */
                /* Match T5's necessary spectral support condition.  Multiplying
                 * independent all-band sums assigned high scores to blocks
                 * whose camera and light paths occupied disjoint wavelengths. */
                const float e  = spectral_beta_product(cb, lb) * geom;
                if (e > 0.0 && !isinf(e) && !isnan(e)) locA += e;
            }
        }
    }

    s_sumA[tid] = locA;
    s_cntC[tid] = locC;
    s_cntS[tid] = locS;
    s_maxG[tid] = locG;
    barrier();

    /* Tree reduction over the 64 threads. */
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_sumA[tid] += s_sumA[tid + stride];
            s_cntC[tid] += s_cntC[tid + stride];
            s_cntS[tid] += s_cntS[tid + stride];
            s_maxG[tid]  = max(s_maxG[tid], s_maxG[tid + stride]);
        }
        barrier();
    }

    if (tid == 0u) {
        const float samp = max(1.0, float(s_cntS[0]));
        const float A = s_sumA[0] / samp;                 /* mean energy / sample  */
        const float B = float(s_cntC[0]) / samp;          /* connectable fraction  */
        const float C = s_maxG[0];                        /* peak geom             */
        const float S = wA * (A / max(normA, 1e-20))
                      + wB * (B / max(normB, 1e-20))
                      + wC * (C / max(normC, 1e-20));

        const uint o = (c * nlb + b) * 4u;
        scores[o + 0u] = A;
        scores[o + 1u] = B;
        scores[o + 2u] = C;
        scores[o + 3u] = S;
    }
}
