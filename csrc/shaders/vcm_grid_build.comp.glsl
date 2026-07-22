#version 430 core

/* Build a linked-cell hash over merge-eligible light vertices.  A photon is
 * stored only after the light walk has reached a real, non-delta surface.
 * Specular vertices stay in the path prefix (and therefore in its throughput
 * and PDFs), but are never treated as finite-area merge receivers. */
layout(local_size_x = 128) in;

#define LGV_STRIDE 57u
#define PDF_DELTA_SPECULAR (1u << 2)
#define MAT_APERTURE_STOP 128u

layout(std430, binding=0) readonly buffer LightVertices { float lv[]; };
layout(std430, binding=10) coherent buffer GridHeads { uint heads[]; };
layout(std430, binding=11) writeonly buffer GridNext { uint next_link[]; };
layout(std430, binding=12) coherent buffer VcmDebug { uint dbg[]; };

uniform uint n_light_verts;
uniform int dispatch_base;
uniform uint hash_mask;
uniform float merge_radius;

uint hash_cell(ivec3 c) {
    uint h = uint(c.x) * 0x8da6b343u;
    h ^= uint(c.y) * 0xd8163841u;
    h ^= uint(c.z) * 0xcb1ab31fu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    return h & hash_mask;
}

void main() {
    uint i = uint(dispatch_base) + gl_GlobalInvocationID.x;
    if (i >= n_light_verts) return;
    uint b = i * LGV_STRIDE;
    uint vinfo = floatBitsToUint(lv[b + 9u]);
    uint vi = vinfo & 0xffffu;
    uint flags = floatBitsToUint(lv[b + 7u]);
    uint pdf_flags = floatBitsToUint(lv[b + 13u]);
    uint blocked = floatBitsToUint(lv[b + 14u]);
    bool eligible = vi > 0u && (vinfo >> 31) != 0u && blocked == 0u &&
                    (flags & MAT_APERTURE_STOP) == 0u &&
                    (pdf_flags & PDF_DELTA_SPECULAR) == 0u &&
                    lv[b + 10u] >= 0.0f;
    if (!eligible) {
        next_link[i] = 0xffffffffu;
        return;
    }
    vec3 p = vec3(lv[b], lv[b+1u], lv[b+2u]);
    ivec3 cell = ivec3(floor(p / merge_radius));
    uint bucket = hash_cell(cell);
    next_link[i] = atomicExchange(heads[bucket], i);
    atomicAdd(dbg[0], 1u);
}
