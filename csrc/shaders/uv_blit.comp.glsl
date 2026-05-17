#version 430 core
/**
 * uv_blit.comp.glsl — GPU-direct UV accumulator → RGBA16F texture array blit.
 *
 * Reads the flat uint32 uv_accum SSBO (binding 7, same buffer T3 writes to
 * atomically) and converts spectral magnitude channels into a displayable
 * RGBA16F image written into an image2DArray (image unit 0 = tex_uv_pages, which
 * is shared with the Python/Pygame display context when WGL context sharing is
 * active).
 *
 * When the C++ compute context and the Pygame display context are linked via
 * wglCreateContextAttribsARB(hDC, display_hglrc, attribs), all GL objects
 * (textures, buffers) live in a shared namespace:
 *   - This shader writes tex_uv_pages (compute context, gl_dispatch_thread)
 *   - Python's draw_uv_mesh binds it directly by ID (display context, main thread)
 * No CPU round-trip, no PCIe readback for UV display data.
 *
 * ── Channel layout in uv_accum (per group g, texel t = y*res+x) ──────────
 *   offset    = group_accum_offset[g]   (from ssbo_tri_uv_and_meta meta section)
 *   n2        = res * res
 *   HDR       = 11                      (UV_N_HDR_CHANNELS)
 *
 *   ch 0          : total hit count            uint, raw
 *   ch 1-7        : HDR bookkeeping            uint, raw
 *   ch 8-10       : normal xyz                 signed int / 32768
 *   ch HDR+b      (b<nb): magnitude            uint / 65536
 *   ch HDR+nb+b   (b<nb): amplitude re         signed / 32768
 *   ch HDR+2*nb+b (b<nb): amplitude im         signed / 32768
 *   ch HDR+3*nb+b (b<nb): forward magnitude    uint / 65536   ← used here
 *   ch HDR+4*nb+b (b<nb): sensor magnitude     uint / 65536   ← used here
 *
 * ── Output RGBA ───────────────────────────────────────────────────────────
 *   RGB  = spectral bands → sRGB via per-band rgb_w[] weights
 *   A    = max(R, G, B)   (luminance proxy for alpha discarding in mesh shader)
 *
 * ── Synchronization note ──────────────────────────────────────────────────
 * The caller (GlPipelineDispatch::dispatch_uv_blit) issues
 *   glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
 * before the dispatch so this shader sees the latest T3 atomic writes.
 * After the dispatch it issues
 *   glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
 * so that the display context's subsequent sampling sees the written texels.
 */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

/* ── Bindings ──────────────────────────────────────────────────────────── */

/* Flat uint32 accumulator — same buffer T3 writes with imageAtomicAdd. */
layout(std430, binding = 7) readonly buffer UvAccumBuf {
    uint uv_accum[];
};

/* Merged [n_tris group-ids] ++ [n_groups*2 meta ints].
 *   uv_meta[uv_meta_base + g*2 + 0] = group_uv_res[g]
 *   uv_meta[uv_meta_base + g*2 + 1] = group_uv_accum_offset[g]  */
layout(std430, binding = 6) readonly buffer UvMetaBuf {
    int uv_meta[];
};

/* Output: RGBA16F texture array shared with the display context.
 * One layer per UV group.  Bound as an image unit (not a sampler). */
layout(rgba16f, binding = 0) writeonly uniform image2DArray uv_out;

/* ── Uniforms ──────────────────────────────────────────────────────────── */

uniform int  n_uv_groups;   /* number of active UV groups / image layers */
uniform int  uv_meta_base;  /* index into uv_meta[] where group meta starts (= n_tris) */
uniform int  n_bands;       /* number of spectral bands (clamped to 8 in shader) */
uniform int  uv_blit_mode;  /* 0=combined (fwd+sen), 1=forward only, 2=sensor only */

/* Per-band RGB triplets: rgb_w[b].xyz maps band b's scalar magnitude to RGB.
 * Layout matches _wavelength_to_rgb_weights() in thick_lens_focus_lab.py.
 * Uploaded as a flat vec3 array (max 32 bands = MAX_SPECTRAL_BANDS). */
uniform vec3 rgb_w[32];

/* ── Main ──────────────────────────────────────────────────────────────── */

void main() {
    int gid = int(gl_GlobalInvocationID.z);
    int tx  = int(gl_GlobalInvocationID.x);
    int ty  = int(gl_GlobalInvocationID.y);

    if (gid >= n_uv_groups) return;

    /* Read group geometry from the meta section of ssbo_tri_uv_and_meta. */
    int meta_off = uv_meta_base + gid * 2;
    int res      = uv_meta[meta_off + 0];
    int acc_off  = uv_meta[meta_off + 1];

    if (res <= 0 || tx >= res || ty >= res) return;

    int n2    = res * res;
    int texel = ty * res + tx;

    /* UV_N_HDR_CHANNELS = 11 (must match the C++ constant). */
    const int HDR = 11;

    /* Accumulate spectral RGB by combining forward and/or sensor magnitudes. */
    vec3 rgb = vec3(0.0);
    int nb = min(n_bands, 32);

    for (int b = 0; b < nb; ++b) {
        /* Forward magnitude channel: HDR + 3*nb + b, unsigned ×65536. */
        float fwd = float(uv_accum[acc_off + (HDR + 3 * nb + b) * n2 + texel]) / 65536.0;
        /* Sensor magnitude channel:  HDR + 4*nb + b, unsigned ×65536. */
        float sen = float(uv_accum[acc_off + (HDR + 4 * nb + b) * n2 + texel]) / 65536.0;

        float mag;
        if      (uv_blit_mode == 1) mag = fwd;
        else if (uv_blit_mode == 2) mag = sen;
        else                        mag = fwd + sen;

        rgb += mag * rgb_w[b];
    }

    float alpha = max(max(rgb.r, rgb.g), rgb.b);
    imageStore(uv_out, ivec3(tx, ty, gid), vec4(rgb, alpha));
}
