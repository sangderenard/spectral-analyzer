/**
 * player_clip.cpp — Axis-aligned slab player collision engine.
 *
 * The resolve loop checks each slab independently.  Order of slabs doesn't
 * matter for a non-sweeping solver; corner cases are handled by the next tick.
 * Each tick the corrected state is published atomically so graphics and audio
 * threads always see a complete, consistent frame.
 */
#include "player_clip.h"

#include <algorithm>
#include <cmath>
#include <cstring>

/* -------------------------------------------------------------------------- */
PlayerClipEngine::PlayerClipEngine() {
    std::memset(_frames, 0, sizeof(_frames));
    _frames[0].radius = 0.25f;
    _frames[1].radius = 0.25f;
    _read_idx.store(0, std::memory_order_relaxed);
    configure_state_tensor(STATE_TENSOR_DEFAULT_CAPACITY,
                           STATE_TENSOR_MIN_STRIDE);
}

/* -------------------------------------------------------------------------- */
void PlayerClipEngine::set_slabs(const std::vector<PlayerWallSlab>& slabs) {
    _slabs = slabs;
}

/* -------------------------------------------------------------------------- */
uint32_t PlayerClipEngine::_resolve(float* pos, float* vel,
                                    float radius,
                                    float floor_z, float ceil_z) const {
    uint32_t flags = CLIP_NONE;

    for (const auto& s : _slabs) {
        /* Player position and perpendicular components per axis */
        float  aval, p0, p1;
        float* aptr;
        float* vptr;
        uint32_t flag_bit;

        if (s.axis == 0) {
            aval = pos[0]; p0 = pos[1]; p1 = pos[2];
            aptr = &pos[0]; vptr = &vel[0]; flag_bit = CLIP_X;
        } else if (s.axis == 1) {
            aval = pos[1]; p0 = pos[0]; p1 = pos[2];
            aptr = &pos[1]; vptr = &vel[1]; flag_bit = CLIP_Y;
        } else {
            aval = pos[2]; p0 = pos[0]; p1 = pos[1];
            aptr = &pos[2]; vptr = &vel[2]; flag_bit = CLIP_Z;
        }

        /* Perpendicular range check — skip if both hi values are zero (infinite) */
        bool finite_range = (s.range_hi[0] != 0.0f || s.range_hi[1] != 0.0f);
        if (finite_range) {
            if (p0 < s.range_lo[0] || p0 > s.range_hi[0] ||
                p1 < s.range_lo[1] || p1 > s.range_hi[1]) {
                continue;
            }
        }

        /* Signed distance from the wall face; positive = in free space side */
        float dist = (aval - s.pos) * s.normal_sign;

        if (dist < radius) {
            *aptr += (radius - dist) * s.normal_sign;
            /* Zero velocity component pushing into the wall */
            if ((*vptr) * s.normal_sign < 0.0f) *vptr = 0.0f;
            flags |= flag_bit;
        }
    }

    /* Floor / ceiling */
    if (pos[2] < floor_z + radius) {
        pos[2] = floor_z + radius;
        if (vel[2] < 0.0f) vel[2] = 0.0f;
        flags |= CLIP_Z;
    }
    if (ceil_z > floor_z && pos[2] > ceil_z - radius) {
        pos[2] = ceil_z - radius;
        if (vel[2] > 0.0f) vel[2] = 0.0f;
        flags |= CLIP_Z;
    }

    return flags;
}

/* -------------------------------------------------------------------------- */
uint32_t PlayerClipEngine::tick(float* pos_inout,
                                float* vel_inout,
                                float  floor_z,
                                float  ceil_z,
                                float  radius,
                                float  yaw_rad,
                                float  dt) {
    float pre_pos[3] = {pos_inout[0], pos_inout[1], pos_inout[2]};
    float pre_vel[3] = {vel_inout[0], vel_inout[1], vel_inout[2]};
    uint32_t flags = _resolve(pos_inout, vel_inout, radius, floor_z, ceil_z);

    /* Write to the buffer that readers are NOT currently using */
    int write_idx = 1 - _read_idx.load(std::memory_order_relaxed);
    PlayerStateFrame& w = _frames[write_idx];
    w.px         = pos_inout[0];
    w.py         = pos_inout[1];
    w.pz         = pos_inout[2];
    w.vx         = vel_inout[0];
    w.vy         = vel_inout[1];
    w.vz         = vel_inout[2];
    w.yaw_rad    = yaw_rad;
    w.radius     = radius;
    w.floor_z    = floor_z;
    w.ceil_z     = ceil_z;
    w.clip_flags = flags;
    w.generation = ++_generation;
    _write_state_tensor(pre_pos, pre_vel,
                        pos_inout, vel_inout,
                        floor_z, ceil_z, radius, yaw_rad,
                        dt, flags, _generation);

    /* Publish: flip read_idx so readers see the freshly written frame */
    _read_idx.store(write_idx, std::memory_order_release);

    return flags;
}

/* -------------------------------------------------------------------------- */
void PlayerClipEngine::read(PlayerStateFrame* out) const {
    int idx = _read_idx.load(std::memory_order_acquire);
    std::memcpy(out, &_frames[idx], sizeof(PlayerStateFrame));
}

const PlayerStateFrame* PlayerClipEngine::read_frame() const {
    int idx = _read_idx.load(std::memory_order_acquire);
    return &_frames[idx];
}

/* -------------------------------------------------------------------------- */
void PlayerClipEngine::configure_state_tensor(int capacity, int stride) {
    _state_capacity = std::max(0, capacity);
    _state_stride   = std::max(STATE_TENSOR_MIN_STRIDE, stride);
    _state_cursor   = 0;
    _state_tensor.assign((size_t)_state_capacity * (size_t)_state_stride, 0.0f);
}

const float* PlayerClipEngine::state_tensor_data() const {
    return _state_tensor.empty() ? nullptr : _state_tensor.data();
}

int PlayerClipEngine::state_tensor_capacity() const {
    return _state_capacity;
}

int PlayerClipEngine::state_tensor_stride() const {
    return _state_stride;
}

int PlayerClipEngine::state_tensor_cursor() const {
    return _state_cursor;
}

void PlayerClipEngine::_write_state_tensor(const float* pre_pos,
                                           const float* pre_vel,
                                           const float* pos,
                                           const float* vel,
                                           float floor_z,
                                           float ceil_z,
                                           float radius,
                                           float yaw_rad,
                                           float dt,
                                           uint32_t flags,
                                           uint32_t generation) {
    if (_state_capacity <= 0 || _state_stride < STATE_TENSOR_MIN_STRIDE ||
        _state_tensor.empty()) {
        return;
    }
    int row_idx = _state_cursor;
    float* row = _state_tensor.data() + (size_t)row_idx * (size_t)_state_stride;
    std::fill(row, row + _state_stride, 0.0f);

    row[0] = pos[0];      row[1] = pos[1];      row[2] = pos[2];
    row[3] = vel[0];      row[4] = vel[1];      row[5] = vel[2];
    row[6] = pre_pos[0];  row[7] = pre_pos[1];  row[8] = pre_pos[2];
    row[9] = pre_vel[0];  row[10] = pre_vel[1]; row[11] = pre_vel[2];
    row[12] = yaw_rad;
    row[13] = radius;
    row[14] = floor_z;
    row[15] = ceil_z;
    row[16] = dt;
    row[17] = static_cast<float>(flags);
    row[18] = static_cast<float>(generation);
    row[19] = static_cast<float>(_slabs.size());
    row[20] = pos[0] - pre_pos[0];
    row[21] = pos[1] - pre_pos[1];
    row[22] = pos[2] - pre_pos[2];
    row[23] = std::sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);

    _state_cursor = (_state_cursor + 1) % _state_capacity;
}
