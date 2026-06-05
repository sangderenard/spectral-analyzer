#version 430 core
/* ─────────────────────────────────────────────────────────────────────────────
 * bdpt_sort_init.comp.glsl  —  GPU-native T5: build sort keys for BDPT vertices
 *
 * One thread per vertex slot.  Builds a 64-bit sort key (two uint32s):
 *   key_hi = (stream:1 | subpath_id:31)
 *   key_lo = (vertex_index:16 | 0:16)
 * Ascending sort puts light vertices (stream=0) before cam vertices (stream=1),
 * then orders by subpath_id, then by vertex_index within each subpath.
 * This gives the subpath-consecutive layout that t5_full_connect needs for its
 * MIS chain walk (flat_base = gid - li_v where li_v = vertex_index_in_subpath).
 *
 * Also atomically increments sort_counts[0] (n_lv) and sort_counts[1] (n_cv).
 * Pads slots >= nv with sentinel keys (0xFFFFFFFF) so they sort to the end.
 *
 * Bindings:
 *   0  BdptVertBuf   ssbo_bdpt_output  readonly   28 floats per vertex
 *   1  SortKeysBuf   ssbo_sort_keys    coherent   2 uint32 per slot (hi, lo)
 *   2  SortIdxBuf    ssbo_sort_idx     coherent   1 uint32 per slot (original idx)
 *   3  SortCountsBuf ssbo_sort_counts  coherent   [0]=n_lv, [1]=n_cv
 * ─────────────────────────────────────────────────────────────────────────────
 */
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer BdptVertBuf   { float bdpt_verts[]; };
layout(std430, binding = 1) coherent buffer SortKeysBuf   { uint  sort_keys[];  };
layout(std430, binding = 2) coherent buffer SortIdxBuf    { uint  sort_idx[];   };
layout(std430, binding = 3) coherent buffer SortCountsBuf { uint  sort_counts[]; };

#define BDPT_VERTEX_STRIDE 28

uniform int nv;    /* actual vertex count (from ssbo_t3_meta readback) */
uniform int npad;  /* padded count = next power of 2 >= nv             */

void main() {
    int V = int(gl_GlobalInvocationID.x);
    if (V >= npad) return;

    if (V >= nv) {
        /* Pad with max-value sentinel — sorts to the end, ignored by pack shader. */
        sort_keys[V * 2 + 0] = 0xFFFFFFFFu;
        sort_keys[V * 2 + 1] = 0xFFFFFFFFu;
        sort_idx [V]          = uint(V);
        return;
    }

    /* Read subpath_id and packed_vi from vertex record. */
    uint sid       = floatBitsToUint(bdpt_verts[V * BDPT_VERTEX_STRIDE + 0]);
    uint packed_vi = floatBitsToUint(bdpt_verts[V * BDPT_VERTEX_STRIDE + 1]);
    uint stream    = (packed_vi >> 8) & 1u;        /* bit 8 = stream (0=light 1=cam) */
    uint vi        = (packed_vi >> 16) & 0xFFFFu;  /* vertex index within subpath    */

    sort_keys[V * 2 + 0] = (stream << 31) | (sid & 0x7FFFFFFFu);
    sort_keys[V * 2 + 1] = vi << 16;
    sort_idx [V]          = uint(V);

    if (stream == 0u) atomicAdd(sort_counts[0], 1u);
    else              atomicAdd(sort_counts[1], 1u);
}
