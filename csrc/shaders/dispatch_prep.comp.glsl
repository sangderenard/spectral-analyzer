/*
 * dispatch_prep.comp.glsl — single-thread GPU pass between T1 and T3.
 *
 * Runs after T1 finishes (GL_SHADER_STORAGE_BARRIER_BIT), reads the hit
 * count from CounterBuf[0] and:
 *   1. Writes indirect dispatch params for T3 to CounterBuf[5..7]:
 *        {(n_hits+63)/64, 1, 1}
 *      so the host can call glDispatchComputeIndirect without a CPU readback.
 *   2. Writes n_hits to MetaBuf[bdpt_count_base+3] so T3's per-invocation
 *      guard can read the exact count from its already-bound MetaBuf.
 *
 * No other data is produced.  One workgroup of one thread is sufficient.
 */
#version 430 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) coherent buffer CounterBuf { uint counters[]; };
layout(std430, binding = 1) coherent buffer MetaBuf    { uint meta[];     };

uniform int bdpt_count_base;   /* = 2 + n_mats; matches T3's bdpt_count_base */

void main() {
    uint n_hits = counters[0];

    /* Indirect dispatch params for T3 at CounterBuf[5..7].
     * Always write at least 1 so glDispatchComputeIndirect never receives a
     * zero workgroup count (some drivers generate GL_INVALID_VALUE for that).
     * T3 guards each invocation with "if (gid >= n_hits) return;" so the
     * extra workgroup when n_hits==0 is harmless. */
    counters[5] = max(1u, (n_hits + 63u) / 64u);
    counters[6] = 1u;
    counters[7] = 1u;

    /* n_hits for T3's invocation guard at meta[bdpt_count_base+3] */
    meta[uint(bdpt_count_base) + 3u] = n_hits;
}
