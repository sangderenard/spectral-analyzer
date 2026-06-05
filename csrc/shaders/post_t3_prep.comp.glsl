/*
 * post_t3_prep.comp.glsl — single-thread GPU pass after T3.
 *
 * Mirrors dispatch_prep but runs AFTER T3 instead of between T1 and T3.
 *
 * After T3 has written:
 *   meta[0] = nc      (child intent count)
 *   meta[1] = nt_term (terminal record count)
 *
 * This shader:
 *   1. Writes T1 indirect dispatch params for the NEXT bounce into counters[8..11]:
 *        counters[8]  = nc             (T1 n_intents for next bounce)
 *        counters[9]  = ceil(nc/64)    (indirect x — always ≥ 1; T1 guards via counters[8])
 *        counters[10] = 1              (indirect y)
 *        counters[11] = 1              (indirect z)
 *
 *   2. Writes terminal splat indirect params into SplatIndirectBuf[0..2]:
 *        SplatIndirectBuf[0] = ceil(nt_term/64)  (always ≥ 1; splat shader guards via nt_term)
 *        SplatIndirectBuf[1] = 1
 *        SplatIndirectBuf[2] = 1
 *
 * One workgroup of one thread is sufficient.
 */
#version 430 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) coherent buffer CounterBuf       { uint counters[];      };
layout(std430, binding = 1) coherent buffer MetaBuf          { uint meta[];          };
layout(std430, binding = 2) coherent buffer SplatIndirectBuf { uint splat_indirect[]; };

void main() {
    uint nc     = meta[0];
    uint nt     = meta[1];

    /* T1 indirect args for next bounce */
    counters[8]  = nc;
    counters[9]  = max(1u, (nc   + 63u) / 64u);
    counters[10] = 1u;
    counters[11] = 1u;

    /* Terminal splat indirect args */
    splat_indirect[0] = max(1u, (nt + 63u) / 64u);
    splat_indirect[1] = 1u;
    splat_indirect[2] = 1u;
}
