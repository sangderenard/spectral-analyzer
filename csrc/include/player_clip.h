/**
 * player_clip.h — Axis-aligned slab player collision engine.
 *
 * Two-buffer design: the write side advances freely; the read side (graphics
 * route, Python) always sees a consistent, atomically-published frame without
 * taking a lock.  C++ owns the authoritative position correction; the GLSL
 * compute path replicates the clip test and writes a clip-flags SSBO that
 * visual shaders can use for effect blending (sparks, wall glow, etc.) with
 * one frame of latency.
 *
 * Layout contract:
 *   PlayerStateFrame is 64 bytes, aligned to 64 (one cache line).
 *   frames[2] are therefore on separate cache lines — no false sharing.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

/* ---- constants ---------------------------------------------------------- */
enum PlayerClipFlags : uint32_t {
    CLIP_NONE = 0,
    CLIP_X    = 1u << 0,
    CLIP_Y    = 1u << 1,
    CLIP_Z    = 1u << 2,
};

/* ---- PlayerWallSlab ----------------------------------------------------- */
/**
 * One infinite axis-aligned slab.
 *   axis        0=X 1=Y 2=Z  — axis the slab's normal is parallel to
 *   pos         signed position of the wall face on that axis
 *   normal_sign +1 = wall faces positive direction, -1 = negative
 *   range_lo/hi optional finite extent on the two perpendicular axes
 *               (both 0 → infinite slab)
 */
struct PlayerWallSlab {
    int   axis;
    float pos;
    float normal_sign;
    float range_lo[2];  /* [0]=first perp axis lo, [1]=second perp axis lo */
    float range_hi[2];  /* [0]=first perp axis hi, [1]=second perp axis hi */
};

/* ---- PlayerStateFrame --------------------------------------------------- */
/**
 * 64-byte, cache-line-aligned player state snapshot.
 * Written on the game tick, published atomically, read by graphics/audio.
 */
struct alignas(64) PlayerStateFrame {
    float    px, py, pz;       /* world position                            */
    float    vx, vy, vz;       /* velocity m/s                              */
    float    yaw_rad;          /* facing angle                              */
    float    radius;           /* collision sphere radius                   */
    float    floor_z;          /* current floor height                      */
    float    ceil_z;           /* current ceiling height                    */
    uint32_t clip_flags;       /* which axes were clipped this tick         */
    uint32_t generation;       /* monotonic counter, incremented each tick  */
    float    _pad[4];          /* total = 16 floats = 64 bytes              */
};
static_assert(sizeof(PlayerStateFrame) == 64, "PlayerStateFrame must be 64 bytes");

/* ---- PlayerClipEngine --------------------------------------------------- */
class PlayerClipEngine {
public:
    static constexpr int STATE_TENSOR_DEFAULT_CAPACITY = 256;
    static constexpr int STATE_TENSOR_MIN_STRIDE       = 32;

    PlayerClipEngine();

    /* Replace the full slab list.  Call whenever room geometry changes. */
    void set_slabs(const std::vector<PlayerWallSlab>& slabs);

    /**
     * tick() — advance the simulation by dt seconds.
     *
     * pos_inout[3]  in:  pre-move position (x,y,z)
     *               out: corrected position after slab resolution
     * vel_inout[3]  in:  current velocity
     *               out: velocity with clipped components zeroed
     * floor_z       floor height for this position
     * ceil_z        ceiling height
     * radius        player sphere radius
     * yaw_rad       current facing (passed through, not modified)
     *
     * Returns the clip_flags bitmask for this tick.
     */
    uint32_t tick(float* pos_inout,
                  float* vel_inout,
                  float  floor_z,
                  float  ceil_z,
                  float  radius,
                  float  yaw_rad,
                  float  dt = 0.016f);

    /* Preallocate the owned state tensor. No per-tick allocation occurs. */
    void configure_state_tensor(int capacity,
                                int stride = STATE_TENSOR_MIN_STRIDE);

    const float* state_tensor_data() const;
    int state_tensor_capacity() const;
    int state_tensor_stride() const;
    int state_tensor_cursor() const;

    /**
     * read() — copy the last published frame into *out.
     * Safe to call from any thread concurrently with tick().
     */
    void read(PlayerStateFrame* out) const;

    /* Raw access for pybind (returns pointer to current read frame). */
    const PlayerStateFrame* read_frame() const;

private:
    uint32_t _resolve(float* pos, float* vel, float radius,
                      float floor_z, float ceil_z) const;
    void _write_state_tensor(const float* pre_pos,
                             const float* pre_vel,
                             const float* pos,
                             const float* vel,
                             float floor_z,
                             float ceil_z,
                             float radius,
                             float yaw_rad,
                             float dt,
                             uint32_t flags,
                             uint32_t generation);

    PlayerStateFrame        _frames[2];
    std::atomic<int>        _read_idx{0};
    uint32_t                _generation{0};
    std::vector<PlayerWallSlab> _slabs;
    std::vector<float>      _state_tensor;
    int                     _state_capacity{0};
    int                     _state_stride{STATE_TENSOR_MIN_STRIDE};
    int                     _state_cursor{0};
};
