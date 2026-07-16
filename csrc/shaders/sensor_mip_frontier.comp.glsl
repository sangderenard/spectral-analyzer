#version 430 core

layout(local_size_x = 64) in;

const uint NODE_FRONTIER = 1u << 0;

struct SensorMipNode {
    vec4 uv_bounds;
    uint parent_id;
    uint first_child_id;
    uint level;
    uint flags;
    uint direct_moment_offset;
    uint rollup_moment_offset;
    uint direct_sample_count;
    uint completed_epochs;
    float priority;
    uint child_slot;
    uint pad0;
    uint pad1;
};

layout(std430, binding = 0) readonly buffer NodeBuf { SensorMipNode nodes[]; };
/* control[0]=node_count, control[2]=frontier_count, control[3]=error,
 * control[4]=max_nodes */
layout(std430, binding = 1) buffer ControlBuf { uint control[]; };
layout(std430, binding = 2) writeonly buffer FrontierBuf { uint frontier[]; };

uniform uint reset_only;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (reset_only != 0u) {
        if (index == 0u) control[2] = 0u;
        return;
    }
    if (index >= control[0]) return;
    if ((nodes[index].flags & NODE_FRONTIER) == 0u) return;
    uint slot = atomicAdd(control[2], 1u);
    if (slot < control[4]) frontier[slot] = index;
    else atomicOr(control[3], 16u);
}
