#version 430 core
/*
 * ray_wave_bpm.comp.glsl — legacy/calibration T4 backend: 2-D ADI-CN BPM.
 *
 * Computes one longitudinal step (dz) of the paraxial BPM operator for a
 * 2-D transverse wave field of shape [n_bands][ny][nx].
 *
 * One shader invocation per pixel for mode=0 (carrier phase advance).
 * One shader invocation per row  for mode=1 (horizontal ADI half-step).
 * One shader invocation per col  for mode=2 (vertical   ADI half-step).
 *
 * local_size_x = 1: each invocation is the sole thread in its workgroup and
 * handles an entire row or column sequentially.  All scratch storage is
 * private per-invocation (GLSL local variables / registers) — no shared
 * memory is used or needed.
 *
 * The C++ dispatcher calls this shader three times per z-step:
 *   (a) Carrier advance  — dispatch(n_bands * ny * nx, 1, 1)
 *   (b) glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
 *   (c) Horizontal sweep — dispatch(n_bands * ny,      1, 1)
 *   (d) glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
 *   (e) Vertical sweep   — dispatch(n_bands * nx,      1, 1)
 *   (f) glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
 *
 * Buffer layout:
 *   WaveReBuf / WaveImBuf  (n_bands * ny * nx floats each):
 *     index(band, iy, ix) = band * (ny * nx) + iy * nx + ix
 *
 *   TmpReBuf / TmpImBuf  (same size): scratch for horizontal→vertical handoff.
 *   These are allocated by the dispatcher (same size as WaveReBuf).
 *
 *   ThomasCPBuf / ThomasDPBuf  (n_bands * max(ny,nx) * max(ny,nx) floats each):
 *   Per-invocation Thomas c' and d' sweep coefficients written to global
 *   memory so each (band, row/col) invocation has its own private lane.
 *   Indexed as: lane * MAX_WAVE_DIM + i,  lane = gid.
 *
 * Uniforms:
 *   mode       : 0=carrier  1=horizontal  2=vertical
 *   nx, ny     : transverse grid dimensions
 *   n_bands    : number of active frequency bands (≤ MAX_GPU_BANDS)
 *   dx         : transverse grid spacing [m]
 *   dz         : longitudinal step [m]
 *   wavelengths: float array [MAX_GPU_BANDS], wavelength per band [m]
 */
layout(local_size_x = 1) in;

/* ── Constants ──────────────────────────────────────────────────────────── */
#ifndef WAVE_BANDS
#define WAVE_BANDS 1
#endif
#define MAX_WAVE_DIM  1024   /* max nx or ny supported */
#define TWO_PI        6.28318530717958647692

/* ── SSBOs ──────────────────────────────────────────────────────────────── */
layout(std430, binding = 0) coherent buffer WaveReBuf  { float wave_re[]; };
layout(std430, binding = 1) coherent buffer WaveImBuf  { float wave_im[]; };
layout(std430, binding = 2) coherent buffer TmpReBuf   { float tmp_re[]; };
layout(std430, binding = 3) coherent buffer TmpImBuf   { float tmp_im[]; };
/* Thomas algorithm scratch: c' and d' coefficients, one lane per invocation.
 * Layout: [lane * MAX_WAVE_DIM + i] for element i in the row/col sweep.      */
layout(std430, binding = 4) coherent buffer ThomasCPBuf { vec2 thomas_cp[]; };
layout(std430, binding = 5) coherent buffer ThomasDPBuf { vec2 thomas_dp[]; };

/* ── Uniforms ───────────────────────────────────────────────────────────── */
uniform int   mode;          /* 0=carrier  1=horizontal  2=vertical  */
uniform int   nx;
uniform int   ny;
uniform int   n_bands;
uniform float dx;
uniform float dz;
uniform float wavelengths[WAVE_BANDS];   /* exact-specialization wavelength lanes */
uniform int   absorber_cells;
uniform float absorber_step_strength;

/* ── Complex arithmetic (vec2 = {re, im}) ───────────────────────────────── */
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
vec2 cdiv(vec2 a, vec2 b) {
    float d = b.x*b.x + b.y*b.y + 1e-40;
    return vec2((a.x*b.x + a.y*b.y)/d, (a.y*b.x - a.x*b.y)/d);
}
vec2 csub(vec2 a, vec2 b) { return a - b; }
vec2 cmuls(vec2 a, float s) { return a * s; }

/* ── Thomas solver ───────────────────────────────────────────────────────
 * Uniform tridiagonal: -beta * x[i-1] + (1+2*beta) * x[i] - beta * x[i+1] = rhs[i]
 * Dirichlet: x[0] = x[n-1] = 0 (absorbing boundaries).
 *
 * rhs values are read from the global wave/tmp buffers (caller provides base
 * offset and stride), and solutions are written back to the same locations.
 * c' and d' sweep coefficients are stored in ThomasCPBuf/ThomasDPBuf at
 * lane offset `lane_base = gid * MAX_WAVE_DIM`.
 */
void thomas_row(int n, vec2 beta, uint rbase, uint stride_elem,
                bool read_tmp,   /* true: read from tmp_re/im, false: wave_re/im */
                bool write_tmp,  /* true: write to  tmp_re/im, false: wave_re/im */
                uint lane_base)
{
    vec2 diag = vec2(1.0, 0.0) + cmuls(beta, 2.0);
    vec2 off  = -beta;   /* -beta */

    /* ── Forward sweep ──────────────────────────────────────────────────── */
    /* i = 0: absorbing boundary → rhs[0] = 0 */
    {
        vec2 rhs0  = vec2(0.0);
        thomas_cp[lane_base + 0u] = cdiv(off, diag);
        thomas_dp[lane_base + 0u] = cdiv(rhs0, diag);
    }

    for (int i = 1; i < n - 1; ++i) {
        uint idx = rbase + uint(i) * stride_elem;
        float re = read_tmp ? tmp_re[idx] : wave_re[idx];
        float im = read_tmp ? tmp_im[idx] : wave_im[idx];
        vec2 u   = vec2(re, im);

        uint ip  = rbase + uint(i - 1) * stride_elem;
        float rep = read_tmp ? tmp_re[ip] : wave_re[ip];
        float imp = read_tmp ? tmp_im[ip] : wave_im[ip];
        vec2  up  = vec2(rep, imp);

        uint in_ = rbase + uint(i + 1) * stride_elem;
        float ren = read_tmp ? tmp_re[in_] : wave_re[in_];
        float imn = read_tmp ? tmp_im[in_] : wave_im[in_];
        vec2  un  = vec2(ren, imn);

        /* RHS of implicit system: rhs[i] = u + beta*(u_prev + u_next - 2u) */
        vec2 rhs_i = u + cmul(beta, (up + un - cmuls(u, 2.0)));

        vec2 denom = csub(diag, cmul(off, thomas_cp[lane_base + uint(i - 1)]));
        thomas_cp[lane_base + uint(i)] = cdiv(off, denom);
        thomas_dp[lane_base + uint(i)] = cdiv(csub(rhs_i, cmul(off, thomas_dp[lane_base + uint(i - 1)])), denom);
    }
    /* i = n-1: absorbing boundary → rhs[n-1] = 0 */
    {
        vec2 rhs_n = vec2(0.0);
        vec2 denom = csub(diag, cmul(off, thomas_cp[lane_base + uint(n - 2)]));
        thomas_dp[lane_base + uint(n - 1)] = cdiv(csub(rhs_n, cmul(off, thomas_dp[lane_base + uint(n - 2)])), denom);
    }

    /* ── Back substitution ──────────────────────────────────────────────── */
    {
        uint idx = rbase + uint(n - 1) * stride_elem;
        vec2 val = thomas_dp[lane_base + uint(n - 1)];
        if (write_tmp) { tmp_re[idx] = val.x; tmp_im[idx] = val.y; }
        else           { wave_re[idx] = val.x; wave_im[idx] = val.y; }
    }
    for (int i = n - 2; i >= 0; --i) {
        uint idxn = rbase + uint(i + 1) * stride_elem;
        float xnr = write_tmp ? tmp_re[idxn] : wave_re[idxn];
        float xni = write_tmp ? tmp_im[idxn] : wave_im[idxn];
        vec2 xn   = vec2(xnr, xni);

        vec2 val  = csub(thomas_dp[lane_base + uint(i)],
                         cmul(thomas_cp[lane_base + uint(i)], xn));
        uint idx  = rbase + uint(i) * stride_elem;
        if (write_tmp) { tmp_re[idx] = val.x; tmp_im[idx] = val.y; }
        else           { wave_re[idx] = val.x; wave_im[idx] = val.y; }
    }
}

/* ── Main ───────────────────────────────────────────────────────────────── */
void main() {
    /* ── Mode 0: carrier phase advance ─────────────────────────────────────
     * Trivially parallel — one pixel per invocation.
     * dispatch: (n_bands * ny * nx, 1, 1) */
    if (mode == 0) {
        uint gid  = gl_GlobalInvocationID.x;
        uint npix = uint(ny * nx);
        uint band = gid / npix;
        uint pix  = gid % npix;
        if (int(band) >= n_bands) return;

        float lam = wavelengths[int(band)];
        if (lam <= 0.0) return;
        float k       = float(TWO_PI) / lam;
        float cos_kdz = cos(k * dz);
        float sin_kdz = sin(k * dz);

        uint idx = band * npix + pix;
        float re = wave_re[idx];
        float im = wave_im[idx];
        int ix = int(pix % uint(nx));
        int iy = int(pix / uint(nx));
        int edge_distance = min(min(ix, nx - 1 - ix), min(iy, ny - 1 - iy));
        float attenuation = 1.0;
        if (absorber_cells > 0 && edge_distance < absorber_cells) {
            float u = float(absorber_cells - edge_distance) / float(absorber_cells);
            float u2 = u * u;
            attenuation = exp(-absorber_step_strength * u2 * u2);
        }
        wave_re[idx] = attenuation * (re * cos_kdz - im * sin_kdz);
        wave_im[idx] = attenuation * (re * sin_kdz + im * cos_kdz);
        return;
    }

    /* ── Mode 1: horizontal (x) ADI half-step ──────────────────────────────
     * One invocation per (band, row).
     * dispatch: (n_bands * ny, 1, 1)
     * Reads wave_re/im  →  writes tmp_re/im. */
    if (mode == 1) {
        uint gid  = gl_GlobalInvocationID.x;
        uint band = gid / uint(ny);
        uint row  = gid % uint(ny);
        if (int(band) >= n_bands || int(row) >= ny) return;

        float lam = wavelengths[int(band)];
        if (lam <= 0.0) return;
        float k   = float(TWO_PI) / lam;
        float dx2 = dx * dx;
        vec2 beta = vec2(0.0, dz / (4.0 * k * dx2));

        uint npix   = uint(nx * ny);
        uint rbase  = band * npix + row * uint(nx);
        uint lane_b = gid * uint(MAX_WAVE_DIM);

        /* Thomas in x: stride = 1 element, absorbing BCs at col 0 and nx-1 */
        thomas_row(nx, beta,
                   rbase, 1u,   /* base, stride */
                   false,        /* read from wave */
                   true,         /* write to tmp   */
                   lane_b);
        return;
    }

    /* ── Mode 2: vertical (y) ADI half-step ────────────────────────────────
     * One invocation per (band, col).
     * dispatch: (n_bands * nx, 1, 1)
     * Reads tmp_re/im  →  writes wave_re/im. */
    if (mode == 2) {
        uint gid  = gl_GlobalInvocationID.x;
        uint band = gid / uint(nx);
        uint col  = gid % uint(nx);
        if (int(band) >= n_bands || int(col) >= nx) return;

        float lam = wavelengths[int(band)];
        if (lam <= 0.0) return;
        float k   = float(TWO_PI) / lam;
        float dx2 = dx * dx;
        vec2 beta = vec2(0.0, dz / (4.0 * k * dx2));

        uint npix   = uint(nx * ny);
        /* First element of this column in tmp: band*npix + 0*nx + col */
        uint cbase  = band * npix + col;
        uint lane_b = gid * uint(MAX_WAVE_DIM);

        /* Thomas in y: stride = nx elements between consecutive rows */
        thomas_row(ny, beta,
                   cbase, uint(nx),   /* base, stride */
                   true,               /* read from tmp  */
                   false,              /* write to wave  */
                   lane_b);
        return;
    }
}
