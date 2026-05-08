#version 430 core
/**
 * player_clip.comp.glsl
 *
 * GPU-side axis-aligned slab player clip pass.  Dispatched once per frame
 * (1 workgroup, 1 invocation) at the END of the main render frame.  Results
 * are read back at the START of the following frame for visual effects only —
 * C++ PlayerClipEngine holds the authoritative position.
 *
 * SSBO bindings:
 *   20 — PlayerStateIn   (read)   one PlayerStateFrame (16 floats)
 *   21 — WallSlabsIn     (read)   N * 8 floats per slab (see layout below)
 *   22 — ClipResultOut   (write)  4 floats: cx, cy, cz, clip_flags_f
 *
 * Slab layout (8 floats each):
 *   [0] axis (0/1/2 as float)
 *   [1] pos
 *   [2] normal_sign (+1 / -1)
 *   [3] range_lo[0]
 *   [4] range_lo[1]
 *   [5] range_hi[0]
 *   [6] range_hi[1]
 *   [7] (reserved / slab count in slot 0's element 7 for element 0 only)
 */

layout(local_size_x = 1) in;

/* binding 20: player state (read) */
layout(std430, binding = 20) readonly buffer PlayerStateIn {
    float state[];   /* px,py,pz, vx,vy,vz, yaw, radius, floor_z, ceil_z,
                        clip_flags_i(as float bits), generation_f, pad[4] */
};

/* binding 21: wall slabs (read) */
layout(std430, binding = 21) readonly buffer WallSlabsIn {
    float slabs[];   /* 8 floats per slab; slabs[7] of first slab = slab count */
};

/* binding 22: clip result (write) */
layout(std430, binding = 22) writeonly buffer ClipResultOut {
    float result[];  /* [0]=cx [1]=cy [2]=cz [3]=clip_flags (as uint bits) */
};

void main() {
    /* Read player state */
    float px     = state[0];
    float py     = state[1];
    float pz     = state[2];
    float vx     = state[3];
    float vy     = state[4];
    float vz     = state[5];
    float radius = state[7];
    float floor_z = state[8];
    float ceil_z  = state[9];

    /* Slab count packed into slot 0 element 7 */
    int n_slabs = int(slabs[7]);

    uint clip_flags = 0u;

    for (int i = 0; i < n_slabs; ++i) {
        int base = i * 8;
        int axis       = int(slabs[base + 0]);
        float spos     = slabs[base + 1];
        float nsign    = slabs[base + 2];
        float rlo0     = slabs[base + 3];
        float rlo1     = slabs[base + 4];
        float rhi0     = slabs[base + 5];
        float rhi1     = slabs[base + 6];

        /* Player position components */
        float aval, p0, p1;
        if (axis == 0) { aval = px; p0 = py; p1 = pz; }
        else if (axis == 1) { aval = py; p0 = px; p1 = pz; }
        else               { aval = pz; p0 = px; p1 = py; }

        /* Perpendicular range check (skip if slab is infinite: both hi == 0) */
        bool in_range = true;
        if (rhi0 != 0.0 || rhi1 != 0.0) {
            in_range = (p0 >= rlo0 && p0 <= rhi0 && p1 >= rlo1 && p1 <= rhi1);
        }
        if (!in_range) continue;

        /* Signed distance from face — positive means player is in free space */
        float dist = (aval - spos) * nsign;

        if (dist < radius) {
            float correction = radius - dist;
            if (axis == 0) { px += correction * nsign; clip_flags |= 1u; }
            else if (axis == 1) { py += correction * nsign; clip_flags |= 2u; }
            else               { pz += correction * nsign; clip_flags |= 4u; }
        }
    }

    /* Floor / ceiling clamp */
    if (pz < floor_z + radius) { pz = floor_z + radius; clip_flags |= 4u; }
    if (ceil_z > floor_z && pz > ceil_z - radius) {
        pz = ceil_z - radius; clip_flags |= 4u;
    }

    result[0] = px;
    result[1] = py;
    result[2] = pz;
    /* Pack uint flags as bit-identical float for readback */
    result[3] = uintBitsToFloat(clip_flags);
}
