#version 430 core

/* Deterministic mixed GPU selection using bounded worst-first heaps.
 *
 * The previous correctness-first implementation rescanned the entire compact
 * frontier once for every selected node: O(frontier * k) work in one shader
 * invocation.  At a 960-node live batch that exceeded the Windows GPU watchdog
 * as soon as the hierarchy reached a few thousand nodes.  This pass scans the
 * frontier once per lane and maintains the best k entries in
 * O(frontier * log(k)).  A reserved coverage lane first chooses the least
 * exposed nodes in a seed-rotated Morton sweep.  Marking those nodes running
 * before the targeted scan makes the two lanes exactly disjoint.  Extraction
 * retains locality as the deterministic tie breaker when work values are equal. */
layout(local_size_x = 1) in;

const uint NODE_FRONTIER = 1u << 0;
const uint NODE_RUNNING = 1u << 2;
const uint MAX_TOP_K = 1024u;

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
    uint node_id;
    uint sample_begin;
    uint sample_count;
    uint output_offset;
    uint seed;
    float priority;
    uint flags;
    uint pad0;
};

layout(std430, binding = 0) buffer NodeBuf { SensorMipNode nodes[]; };
layout(std430, binding = 1) readonly buffer ControlBuf { uint control[]; };
layout(std430, binding = 2) readonly buffer FrontierBuf { uint frontier[]; };
layout(std430, binding = 3) writeonly buffer WorkBuf { SensorMipWork work[]; };
/* schedule[0]=selected_count, schedule[1]=requested_top_k */
layout(std430, binding = 4) buffer ScheduleBuf { uint schedule[]; };

uniform uint seed;
uniform float targeted_fraction;

shared uint heap_ids[MAX_TOP_K];
shared uint heap_locality[MAX_TOP_K];
shared uint heap_epochs[MAX_TOP_K];
shared uint heap_samples[MAX_TOP_K];
shared float heap_priority[MAX_TOP_K];

uint part1by1(uint value) {
    value &= 0x0000FFFFu;
    value = (value | (value << 8)) & 0x00FF00FFu;
    value = (value | (value << 4)) & 0x0F0F0F0Fu;
    value = (value | (value << 2)) & 0x33333333u;
    return (value | (value << 1)) & 0x55555555u;
}

uint locality_key(vec4 bounds) {
    uint u = uint(clamp((bounds.x + bounds.z) * 0.5, 0.0, 1.0) * 65535.0 + 0.5);
    uint v = uint(clamp((bounds.y + bounds.w) * 0.5, 0.0, 1.0) * 65535.0 + 0.5);
    return part1by1(u) | (part1by1(v) << 1);
}

bool better(float ap, uint al, uint ai, float bp, uint bl, uint bi) {
    return ap > bp || (ap == bp && (al < bl || (al == bl && ai < bi)));
}

uint coverage_distance(uint locality) {
    /* Unsigned subtraction intentionally wraps, rotating the full Morton ring
     * without making the same low-coordinate region lead every coating pass. */
    uint cursor = seed * 2654435761u;
    return locality - cursor;
}

bool coverage_better(uint ae, uint asamp, uint al, uint ai,
                     uint be, uint bsamp, uint bl, uint bi) {
    uint ad = coverage_distance(al);
    uint bd = coverage_distance(bl);
    return ae < be
        || (ae == be && (asamp < bsamp
        || (asamp == bsamp && (ad < bd
        || (ad == bd && ai < bi)))));
}

bool coverage_heap_entry_worse(uint a, uint b) {
    return coverage_better(
        heap_epochs[b], heap_samples[b], heap_locality[b], heap_ids[b],
        heap_epochs[a], heap_samples[a], heap_locality[a], heap_ids[a]);
}

bool heap_entry_worse(uint a, uint b) {
    return better(
        heap_priority[b], heap_locality[b], heap_ids[b],
        heap_priority[a], heap_locality[a], heap_ids[a]);
}

void swap_heap(uint a, uint b) {
    uint id = heap_ids[a]; heap_ids[a] = heap_ids[b]; heap_ids[b] = id;
    uint locality = heap_locality[a];
    heap_locality[a] = heap_locality[b]; heap_locality[b] = locality;
    float priority = heap_priority[a];
    heap_priority[a] = heap_priority[b]; heap_priority[b] = priority;
}

void sift_up(uint index) {
    while (index > 0u) {
        uint parent = (index - 1u) >> 1u;
        if (!heap_entry_worse(index, parent)) break;
        swap_heap(index, parent);
        index = parent;
    }
}

void sift_down(uint index, uint count) {
    for (;;) {
        uint left = index * 2u + 1u;
        if (left >= count) break;
        uint right = left + 1u;
        uint worst = left;
        if (right < count && heap_entry_worse(right, left)) worst = right;
        if (!heap_entry_worse(worst, index)) break;
        swap_heap(index, worst);
        index = worst;
    }
}

void swap_coverage_heap(uint a, uint b) {
    uint id = heap_ids[a]; heap_ids[a] = heap_ids[b]; heap_ids[b] = id;
    uint locality = heap_locality[a];
    heap_locality[a] = heap_locality[b]; heap_locality[b] = locality;
    uint epochs = heap_epochs[a]; heap_epochs[a] = heap_epochs[b]; heap_epochs[b] = epochs;
    uint samples = heap_samples[a]; heap_samples[a] = heap_samples[b]; heap_samples[b] = samples;
}

void coverage_sift_up(uint index) {
    while (index > 0u) {
        uint parent = (index - 1u) >> 1u;
        if (!coverage_heap_entry_worse(index, parent)) break;
        swap_coverage_heap(index, parent);
        index = parent;
    }
}

void coverage_sift_down(uint index, uint count) {
    for (;;) {
        uint left = index * 2u + 1u;
        if (left >= count) break;
        uint right = left + 1u;
        uint worst = left;
        if (right < count && coverage_heap_entry_worse(right, left)) worst = right;
        if (!coverage_heap_entry_worse(worst, index)) break;
        swap_coverage_heap(worst, index);
        index = worst;
    }
}

void write_work(uint output_index, uint node_id, float priority, uint flags) {
    SensorMipNode chosen = nodes[node_id];
    nodes[node_id].flags = (chosen.flags & ~NODE_FRONTIER) | NODE_RUNNING;
    SensorMipWork item;
    item.node_id = node_id;
    item.sample_begin = chosen.direct_sample_count;
    item.sample_count = control[7];
    item.output_offset = output_index * control[7];
    item.seed = seed;
    item.priority = priority;
    item.flags = flags;
    item.pad0 = 0u;
    work[output_index] = item;
}

void main() {
    uint limit = min(schedule[1], MAX_TOP_K);
    float fraction = clamp(targeted_fraction, 0.0, 1.0);
    uint targeted_limit = min(limit, uint(floor(float(limit) * fraction + 0.5)));
    uint coverage_limit = limit - targeted_limit;
    uint count = 0u;

    /* Lane 1: guaranteed broad coating.  Exposure count is authoritative;
     * the rotating Morton distance only orders equally exposed regions. */
    for (uint i = 0u; i < control[2]; ++i) {
        uint node_id = frontier[i];
        if (node_id >= control[0]) continue;
        SensorMipNode node = nodes[node_id];
        if ((node.flags & NODE_FRONTIER) == 0u) continue;
        uint locality = locality_key(node.uv_bounds);
        if (count < coverage_limit) {
            heap_ids[count] = node_id;
            heap_locality[count] = locality;
            heap_epochs[count] = node.completed_epochs;
            heap_samples[count] = node.direct_sample_count;
            coverage_sift_up(count);
            count += 1u;
        } else if (coverage_limit > 0u && coverage_better(
                       node.completed_epochs, node.direct_sample_count, locality, node_id,
                       heap_epochs[0], heap_samples[0], heap_locality[0], heap_ids[0])) {
            heap_ids[0] = node_id;
            heap_locality[0] = locality;
            heap_epochs[0] = node.completed_epochs;
            heap_samples[0] = node.direct_sample_count;
            coverage_sift_down(0u, count);
        }
    }
    uint coverage_selected = count;
    while (count > 0u) {
        uint output_index = count - 1u;
        uint node_id = heap_ids[0];
        float priority = max(0.0, nodes[node_id].priority);
        write_work(output_index, node_id, priority, 1u);
        count -= 1u;
        if (count > 0u) {
            heap_ids[0] = heap_ids[count];
            heap_locality[0] = heap_locality[count];
            heap_epochs[0] = heap_epochs[count];
            heap_samples[0] = heap_samples[count];
            coverage_sift_down(0u, count);
        }
    }

    /* Lane 2: learned/measured work value. Coverage nodes are already marked
     * running, so this scan cannot duplicate them. */
    count = 0u;
    for (uint i = 0u; i < control[2]; ++i) {
        uint node_id = frontier[i];
        if (node_id >= control[0]) continue;
        SensorMipNode node = nodes[node_id];
        if ((node.flags & NODE_FRONTIER) == 0u) continue;
        float priority = (isnan(node.priority) || node.priority < 0.0)
            ? 0.0 : node.priority;
        uint locality = locality_key(node.uv_bounds);
        if (count < targeted_limit) {
            heap_ids[count] = node_id;
            heap_locality[count] = locality;
            heap_priority[count] = priority;
            sift_up(count);
            count += 1u;
        } else if (targeted_limit > 0u && better(
                       priority, locality, node_id,
                       heap_priority[0], heap_locality[0], heap_ids[0])) {
            heap_ids[0] = node_id;
            heap_locality[0] = locality;
            heap_priority[0] = priority;
            sift_down(0u, count);
        }
    }

    uint targeted_selected = count;
    /* The root is the worst retained entry. Removing roots and writing from
     * the end produces deterministic best-to-worst work order. */
    while (count > 0u) {
        uint output_index = coverage_selected + count - 1u;
        write_work(output_index, heap_ids[0], heap_priority[0], 0u);
        count -= 1u;
        if (count > 0u) {
            heap_ids[0] = heap_ids[count];
            heap_locality[0] = heap_locality[count];
            heap_priority[0] = heap_priority[count];
            sift_down(0u, count);
        }
    }
    schedule[0] = coverage_selected + targeted_selected;
}
