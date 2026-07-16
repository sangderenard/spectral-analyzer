#version 430 core

layout(local_size_x = 64) in;

const uint NO_NODE = 0xFFFFFFFFu;
const uint BRANCHING = 9u;
const uint NODE_FRONTIER = 1u << 0;
const uint NODE_RUNNING = 1u << 2;
const uint NODE_DIRTY = 1u << 5;
const uint NODE_TERMINAL = 1u << 6;

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
 * [4]=max_nodes [5]=max_depth [6]=n_bands [7]=samples_per_epoch */
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

void main() {
    uint work_index = gl_GlobalInvocationID.x;
    uint completed_count = completed_from_work != 0u ? schedule[0] : control[1];
    if (work_index >= completed_count) return;
    uint node_id = completed_from_work != 0u
        ? selected_work[work_index].node_id : completed[work_index];
    if (node_id >= control[0]) { atomicOr(control[3], 1u); return; }
    SensorMipNode parent = nodes[node_id];
    if (parent.level >= control[5]) {
        nodes[node_id].flags = (parent.flags & ~NODE_RUNNING) | NODE_FRONTIER | NODE_TERMINAL;
        nodes[node_id].completed_epochs = parent.completed_epochs + 1u;
        uint slot = atomicAdd(control[2], 1u);
        if (slot < control[4]) frontier[slot] = node_id; else atomicOr(control[3], 2u);
        return;
    }
    if (parent.first_child_id != NO_NODE) return;
    /* Saturating reservation: never let node_count run beyond the pool. Once
     * storage is full, the region becomes a repeatedly sampled terminal leaf
     * instead of ending a continuous exposure with a topology error. */
    uint first = NO_NODE;
    for (;;) {
        uint observed = control[0];
        if (observed > control[4] || BRANCHING > control[4] - observed) break;
        uint previous = atomicCompSwap(control[0], observed, observed + BRANCHING);
        if (previous == observed) { first = observed; break; }
    }
    if (first == NO_NODE) {
        nodes[node_id].flags = (parent.flags & ~NODE_RUNNING) | NODE_FRONTIER | NODE_TERMINAL;
        nodes[node_id].completed_epochs = parent.completed_epochs + 1u;
        uint slot = atomicAdd(control[2], 1u);
        if (slot < control[4]) frontier[slot] = node_id; else atomicOr(control[3], 2u);
        return;
    }
    float du = (parent.uv_bounds.z - parent.uv_bounds.x) / 3.0;
    float dv = (parent.uv_bounds.w - parent.uv_bounds.y) / 3.0;
    for (uint child_slot = 0u; child_slot < BRANCHING; ++child_slot) {
        uint column = child_slot % 3u;
        uint row = child_slot / 3u;
        uint child_id = first + child_slot;
        vec4 bounds = vec4(
            parent.uv_bounds.x + float(column) * du,
            parent.uv_bounds.y + float(row) * dv,
            column == 2u ? parent.uv_bounds.z : parent.uv_bounds.x + float(column + 1u) * du,
            row == 2u ? parent.uv_bounds.w : parent.uv_bounds.y + float(row + 1u) * dv);
        SensorMipNode child;
        child.uv_bounds = bounds;
        child.parent_id = node_id;
        child.first_child_id = NO_NODE;
        child.level = parent.level + 1u;
        child.flags = NODE_FRONTIER;
        child.direct_moment_offset = child_id * control[6];
        child.rollup_moment_offset = child_id * control[6];
        child.direct_sample_count = 0u;
        child.completed_epochs = 0u;
        /* Children inherit measured work value from the completed parent.
         * The scorer may change priority, never the mandatory 3x3 split. */
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
    memoryBarrierBuffer();
    nodes[node_id].first_child_id = first;
    nodes[node_id].flags = (parent.flags & ~(NODE_RUNNING | NODE_FRONTIER)) | NODE_DIRTY;
    nodes[node_id].completed_epochs = parent.completed_epochs + 1u;
}
