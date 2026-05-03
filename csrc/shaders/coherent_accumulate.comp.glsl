#version 430 core
/* coherent_accumulate.comp.glsl
 *
 * GPU-parallel coherent field accumulator.
 *
 * One compute invocation per multiscale ray segment (14 floats per segment,
 * matching RT_FLOATS_PER_SEG_MS in ray_tracer.h):
 *
 *   [0..2]  seg start (x0, y0, z0)  — world-space metres
 *   [3..5]  seg end   (x1, y1, z1)
 *   [6]     src_id
 *   [7]     bounce
 *   [8]     band index  (integer stored as float)
 *   [9]     amplitude
 *   [10]    phase  (radians, cumulative optical path length × wave-number)
 *   [11]    path_len
 *   [12]    context_id
 *   [13]    scale_type  (0 = RT_SCALE_GEOMETRIC, 1 = RT_SCALE_WAVE)
 *
 * For every segment that straddles the sensor plane z = sensor_z the
 * intersection is computed and its complex amplitude accumulated:
 *
 *   E = amp · exp(i·phase)
 *   out_re[b, py, px] += amp · cos(phase)   (atomically)
 *   out_im[b, py, px] += amp · sin(phase)   (atomically)
 *
 * After accumulating many trickle batches the squared modulus
 *   I = out_re² + out_im²
 * is the coherent diffraction-correct intensity image.  Interference
 * fringes, Airy discs, speckle, and edge-diffraction rings emerge naturally.
 *
 * Pixel mapping (identical to the C fallback in ray_tracer_project_coherent):
 *   sx ∈ [−sensor_r, +sensor_r]  →  px = int((sx + sensor_r) / pixel_pitch)
 *   sy ∈ [−sensor_r, +sensor_r]  →  py = int((sy + sensor_r) / pixel_pitch)
 *
 * The images out_re / out_im are r32ui uimage2DArrays.  Float values are
 * bitcast to/from uint and accumulated via a CAS spin-loop (emulated float
 * atomicAdd, safe on all GL 4.3-capable hardware).
 *
 * Setup (Python / host side):
 *   1. Allocate GL_R32UI texture arrays of shape (n_bands, sensor_h, sensor_w).
 *   2. Upload packed segment data into an SSBO bound to binding=0.
 *   3. glDispatchCompute(ceil(n_segs / 64), 1, 1).
 *   4. glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT).
 *   5. Download / use out_re, out_im.
 */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

/* ── Segment SSBO ──────────────────────────────────────────────────────── */
layout(std430, binding = 0) readonly buffer SegBuf {
    float segs[];   /* packed n_segs × 14 floats */
};

/* ── Coherent field image arrays (r32ui, one layer per frequency band) ─── */
layout(r32ui, binding = 1) coherent volatile uniform uimage2DArray out_re;
layout(r32ui, binding = 2) coherent volatile uniform uimage2DArray out_im;

/* ── Uniforms ──────────────────────────────────────────────────────────── */
uniform int   n_segs;
uniform int   n_bands;
uniform int   sensor_w;
uniform int   sensor_h;
uniform float sensor_z;
uniform float sensor_r;
uniform float pixel_pitch;   /* = 2 * sensor_r / max(sensor_w, sensor_h) */

/* ── CAS-based float atomicAdd helpers ────────────────────────────────── */
/*
 * OpenGL 4.3 exposes imageAtomicCompSwap for r32ui images but not a native
 * float atomicAdd.  The standard workaround is a compare-and-swap spin loop.
 * This is correct but may suffer contention at high density; for production
 * use a warp-local reduction first (left as an optimisation TODO).
 */
void atomicAddFloat_re(ivec3 coord, float value) {
    uint assumed, old;
    old = imageLoad(out_re, coord).r;
    do {
        assumed = old;
        old = imageAtomicCompSwap(out_re, coord, assumed,
              floatBitsToUint(uintBitsToFloat(assumed) + value));
    } while (old != assumed);
}

void atomicAddFloat_im(ivec3 coord, float value) {
    uint assumed, old;
    old = imageLoad(out_im, coord).r;
    do {
        assumed = old;
        old = imageAtomicCompSwap(out_im, coord, assumed,
              floatBitsToUint(uintBitsToFloat(assumed) + value));
    } while (old != assumed);
}

/* ── Main ─────────────────────────────────────────────────────────────── */
void main() {
    uint seg_idx = gl_GlobalInvocationID.x;
    if (int(seg_idx) >= n_segs) return;

    uint base = seg_idx * 14u;

    float x0 = segs[base + 0u];
    float y0 = segs[base + 1u];
    float z0 = segs[base + 2u];
    float x1 = segs[base + 3u];
    float y1 = segs[base + 4u];
    float z1 = segs[base + 5u];
    /* [6] = src_id, [7] = bounce — not used here */
    int   b     = int(segs[base + 8u]);
    float amp   = segs[base + 9u];
    float phase = segs[base + 10u];
    /* [11] = path_len, [12] = context_id, [13] = scale_type — not used */

    /* Reject out-of-range band. */
    if (b < 0 || b >= n_bands) return;

    /* Parametric intersection with z = sensor_z. */
    float dz = z1 - z0;
    if (abs(dz) < 1e-9) return;          /* segment parallel to sensor plane */
    float t = (sensor_z - z0) / dz;
    if (t < 0.0 || t > 1.0) return;      /* intersection outside segment */

    float sx = x0 + t * (x1 - x0);
    float sy = y0 + t * (y1 - y0);

    /* Map world position → pixel index. */
    int px = int((sx + sensor_r) / pixel_pitch);
    int py = int((sy + sensor_r) / pixel_pitch);
    if (px < 0 || px >= sensor_w) return;
    if (py < 0 || py >= sensor_h) return;

    /* Accumulate complex amplitude. */
    ivec3 coord = ivec3(px, py, b);
    atomicAddFloat_re(coord, amp * cos(phase));
    atomicAddFloat_im(coord, amp * sin(phase));
}
