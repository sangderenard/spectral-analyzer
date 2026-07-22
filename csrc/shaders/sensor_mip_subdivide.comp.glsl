#version 430 core

layout(local_size_x = 64) in;

const uint NO_NODE = 0xFFFFFFFFu;
const uint NODE_FRONTIER = 1u << 0;
const uint NODE_RUNNING = 1u << 2;
const uint NODE_DIRTY = 1u << 5;
const uint NODE_TERMINAL = 1u << 6;
const uint WORK_SPLIT_REQUESTED = 1u << 1;
const uint WORK_PRESERVE_COVERAGE = 1u << 2;
const uint NODE_COVERAGE_ANCHOR = 1u << 7;

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

struct SensorMipWork {
    uint node_id; uint sample_begin; uint sample_count; uint output_offset;
    uint seed; float priority; uint flags; uint pad0;
};

layout(std430, binding = 0) buffer NodeBuf { SensorMipNode nodes[]; };
/* [0]=node_count [1]=completed_count [2]=frontier_count [3]=error
 * [4]=max_nodes [5]=max_depth [6]=n_bands [7]=samples_per_epoch
 * [8]=subdivision axis parity (even -> hard 2x2, odd -> hard 3x3) */
layout(std430, binding = 1) buffer ControlBuf { uint control[]; };
layout(std430, binding = 2) readonly buffer CompletedBuf { uint completed[]; };
layout(std430, binding = 3) buffer FrontierBuf { uint frontier[]; };
layout(std430, binding = 4) buffer DirectSumBuf { uint direct_sum[]; };
layout(std430, binding = 5) buffer DirectSumSqBuf { uint direct_sum_sq[]; };
layout(std430, binding = 6) buffer DirectWeightBuf { uint direct_weight[]; };
layout(std430, binding = 7) buffer RollupMeanBuf { uint rollup_mean[]; };
layout(std430, binding = 8) buffer RollupEvidenceBuf { uint rollup_evidence[]; };
layout(std430, binding = 9) readonly buffer WorkBuf { SensorMipWork selected_work[]; };
layout(std430, binding = 10) readonly buffer ScheduleBuf { uint schedule[]; };

uniform uint completed_from_work;

void emit_child(
    SensorMipNode parent, uint parent_id, uint child_id,
    uint child_slot, vec4 bounds)
{
    SensorMipNode child;
    child.uv_bounds = bounds;
    child.parent_id = parent_id;
    child.first_child_id = NO_NODE;
    child.level = parent.level + 1u;
    child.flags = NODE_FRONTIER;
    child.direct_moment_offset = child_id * control[6];
    child.rollup_moment_offset = child_id * control[6];
    child.direct_sample_count = 0u;
    child.completed_epochs = 0u;
    child.priority = parent.priority;
    child.child_slot = child_slot;
    child.pad0 = 0u; child.pad1 = 0u;
    nodes[child_id] = child;
    direct_weight[child_id] = floatBitsToUint(0.0);
    rollup_evidence[child_id] = floatBitsToUint(0.0);
    for (uint band = 0u; band < control[6]; ++band) {
        uint offset = child_id * control[6] + band;
        direct_sum[offset] = floatBitsToUint(0.0);
        direct_sum_sq[offset] = floatBitsToUint(0.0);
        rollup_mean[offset] = floatBitsToUint(0.0);
    }
    uint frontier_slot = atomicAdd(control[2], 1u);
    if (frontier_slot < control[4]) frontier[frontier_slot] = child_id;
    else atomicOr(control[3], 8u);
}

void main() {
    uint work_index = gl_GlobalInvocationID.x;
    uint completed_count = completed_from_work != 0u ? schedule[0] : control[1];
    if (work_index >= completed_count) return;
    uint node_id = completed_from_work != 0u
        ? selected_work[work_index].node_id : completed[work_index];
    if (node_id >= control[0]) { atomicOr(control[3], 1u); return; }
    SensorMipNode parent = nodes[node_id];
    bool split_requested = completed_from_work == 0u
        || (selected_work[work_index].flags & WORK_SPLIT_REQUESTED) != 0u;
    /* Sampling and splitting are separate decisions.  A completed leaf stays
     * schedulable unless the coverage lattice or targeted attention explicitly
     * asks to descend. */
    if (!split_requested) {
        nodes[node_id].flags = (parent.flags & ~NODE_RUNNING) | NODE_FRONTIER;
        nodes[node_id].completed_epochs = parent.completed_epochs + 1u;
        uint slot = atomicAdd(control[2], 1u);
        if (slot < control[4]) frontier[slot] = node_id; else atomicOr(control[3], 2u);
        return;
    }
    if (parent.level >= control[5]) {
        nodes[node_id].flags = (parent.flags & ~NODE_RUNNING) | NODE_FRONTIER | NODE_TERMINAL;
        nodes[node_id].completed_epochs = parent.completed_epochs + 1u;
        uint slot = atomicAdd(control[2], 1u);
        if (slot < control[4]) frontier[slot] = node_id; else atomicOr(control[3], 2u);
        return;
    }
    if (parent.first_child_id != NO_NODE) return;
    bool even_division = (control[8] & 1u) == 0u;
    uint branching = even_division ? 4u : 9u;
    /* Saturating reservation: never let node_count run beyond the pool. Once
     * storage is full, the region becomes a repeatedly sampled terminal leaf
     * instead of ending a continuous exposure with a topology error. */
    uint first = NO_NODE;
    for (;;) {
        uint observed = control[0];
        if (observed > control[4] || branching > control[4] - observed) break;
        uint previous = atomicCompSwap(control[0], observed, observed + branching);
        if (previous == observed) { first = observed; break; }
    }
    if (first == NO_NODE) {
        nodes[node_id].flags = (parent.flags & ~NODE_RUNNING) | NODE_FRONTIER | NODE_TERMINAL;
        nodes[node_id].completed_epochs = parent.completed_epochs + 1u;
        uint slot = atomicAdd(control[2], 1u);
        if (slot < control[4]) frontier[slot] = node_id; else atomicOr(control[3], 2u);
        return;
    }
    vec4 p = parent.uv_bounds;
    if (even_division) {
        float um = (p.x + p.z) * 0.5;
        float vm = (p.y + p.w) * 0.5;
        emit_child(parent, node_id, first + 0u, 0u, vec4(p.x, p.y, um, vm));
        emit_child(parent, node_id, first + 1u, 1u, vec4(um, p.y, p.z, vm));
        emit_child(parent, node_id, first + 2u, 2u, vec4(p.x, vm, um, p.w));
        emit_child(parent, node_id, first + 3u, 3u, vec4(um, vm, p.z, p.w));
    } else {
        float du = (p.z - p.x) * (1.0 / 3.0);
        float dv = (p.w - p.y) * (1.0 / 3.0);
        float u1 = p.x + du, u2 = p.x + 2.0 * du;
        float v1 = p.y + dv, v2 = p.y + 2.0 * dv;
        emit_child(parent, node_id, first + 0u, 0u, vec4(p.x, p.y, u1, v1));
        emit_child(parent, node_id, first + 1u, 1u, vec4(u1, p.y, u2, v1));
        emit_child(parent, node_id, first + 2u, 2u, vec4(u2, p.y, p.z, v1));
        emit_child(parent, node_id, first + 3u, 3u, vec4(p.x, v1, u1, v2));
        emit_child(parent, node_id, first + 4u, 4u, vec4(u1, v1, u2, v2));
        emit_child(parent, node_id, first + 5u, 5u, vec4(u2, v1, p.z, v2));
        emit_child(parent, node_id, first + 6u, 6u, vec4(p.x, v2, u1, p.w));
        emit_child(parent, node_id, first + 7u, 7u, vec4(u1, v2, u2, p.w));
        emit_child(parent, node_id, first + 8u, 8u, vec4(u2, v2, p.z, p.w));
    }
    memoryBarrierBuffer();
    nodes[node_id].first_child_id = first;
    bool preserve_coverage = completed_from_work != 0u
        && (selected_work[work_index].flags & WORK_PRESERVE_COVERAGE) != 0u;
    nodes[node_id].flags = preserve_coverage
        ? ((parent.flags & ~NODE_RUNNING) | NODE_FRONTIER | NODE_DIRTY
           | NODE_COVERAGE_ANCHOR)
        : ((parent.flags & ~(NODE_RUNNING | NODE_FRONTIER)) | NODE_DIRTY);
    nodes[node_id].completed_epochs = parent.completed_epochs + 1u;
}
