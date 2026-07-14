#version 430 core
/* ─────────────────────────────────────────────────────────────────────────────
 * bdpt_sort_step.comp.glsl  —  GPU-native T5: one bitonic sort compare-and-swap
 *
 * Standard Batcher bitonic sort.  The C++ driver dispatches this shader in a
 * nested loop:
 *   for k = 2, 4, ..., npad:
 *     for j = k/2, k/4, ..., 1:
 *       dispatch ceil(npad/2/64) workgroups, passing k and j as uniforms
 *
 * Each invocation handles one (lo, hi) element pair.  Sort is ascending on the
 * 64-bit key (sort_keys[i*2], sort_keys[i*2+1]).
 *
 * Bindings (same as bdpt_sort_init):
 *   0  SortKeysBuf  coherent
 *   1  SortIdxBuf   coherent
 * ─────────────────────────────────────────────────────────────────────────────
 */
layout(local_size_x = 64) in;

layout(std430, binding = 0) coherent buffer SortKeysBuf { uint sort_keys[]; };
layout(std430, binding = 1) coherent buffer SortIdxBuf  { uint sort_idx[];  };

uniform int k;        /* outer loop step (power of 2) */
uniform int j;        /* inner loop stride (power of 2) */
uniform int npad;     /* padded array length             */
uniform int base_idx; /* chunk offset for split dispatches */

void main() {
    int t = base_idx + int(gl_GlobalInvocationID.x);
    if (t >= npad / 2) return;

    /* Compute the pair (lo, hi) for this thread using the standard bitonic formula. */
    int j_mask = j - 1;
    int lo = (t & ~j_mask) * 2 + (t & j_mask);
    int hi = lo + j;
    if (hi >= npad) return;

    uint loH = sort_keys[lo * 2 + 0], loL = sort_keys[lo * 2 + 1];
    uint hiH = sort_keys[hi * 2 + 0], hiL = sort_keys[hi * 2 + 1];

    /* Direction: ascending when (lo & k) == 0. */
    bool asc = ((lo & k) == 0);
    bool lo_gt_hi = (loH > hiH) || (loH == hiH && loL > hiL);
    bool do_swap  = asc ? lo_gt_hi : !lo_gt_hi;

    if (do_swap) {
        sort_keys[lo * 2 + 0] = hiH; sort_keys[lo * 2 + 1] = hiL;
        sort_keys[hi * 2 + 0] = loH; sort_keys[hi * 2 + 1] = loL;
        uint tmp = sort_idx[lo]; sort_idx[lo] = sort_idx[hi]; sort_idx[hi] = tmp;
    }
}
