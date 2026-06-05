#pragma once
/*
 * ray_pipeline.h — 4-stage threaded ray transport pipeline.
 *
 * Architecture: persistent machine model.
 *   T1 intersector  — BVH query, amplitude propagation, field capture.
 *                     Routes to T4 if hit is inside a RT_SCALE_WAVE arena, else T2.
 *   T2 refiner      — parametric surface point / normal refinement.
 *   T3 material     — Fresnel / Snell / diffuse / specular physics.
 *                     Emits child RayIntents back into T1.
 *   T4 wave solver  — per-arena 2-D ADI-CN BPM march.
 *                     Exit rays re-enter T1.
 *
 * Usage:
 *   RayPipelineState* ps = ray_pipeline_create(st, &cfg);
 *   ray_pipeline_submit(ps, intents, n);          // non-blocking; call any time
 *   int n = ray_pipeline_drain(ps, records, max); // non-blocking poll
 *   int live = ray_pipeline_in_flight(ps);         // 0 means batch is done
 *   ray_pipeline_destroy(ps);                      // joins all workers
 */

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <random>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "bdpt_record.h"

struct RayTracerState;

/* ─── Pipeline data types ───────────────────────────────────────────────── */

struct RayIntent {
    Eigen::Vector3d  pos, dir;
    Eigen::VectorXcd amp;
    double           path_len          = 0.0;
    int              medium_mat_idx    = -1;
    uint32_t         interaction_flags = 0u;
    int              src_id            = 0;
    int              bounce            = 0;
    int              bounces_left      = 0;
    double           min_amplitude     = 1e-6;
    uint64_t         tag               = 0;
    uint8_t          color_flag        = 0;  /* user-supplied; carried through all stages */
    float            priority          = 1.0f; /* auxin/sugar scheduling weight; higher = sooner */
    /* Original sensor pixel position for backward rays (color_flag==1).
     * Set at submit time; propagated unchanged through all child bounces so
     * any emissive hit can splat back to the correct sensor pixel regardless
     * of how many refractions the ray traversed. */
    float            sensor_origin_y   = 0.0f;
    float            sensor_origin_z   = 0.0f;

    /* ── BDPT sidecar identity ────────────────────────────────────────────
     * Assigned at launch; propagated unchanged into every child intent so
     * that T1/T2/T3 side-data records can be correlated back to a subpath.
     * subpath_id  : unique id for the full subpath (assigned by submitter).
     * bdpt_vertex : vertex index within the subpath (incremented per bounce).
     * bdpt_stream : BDPT_SIDE_LIGHT (0) or BDPT_SIDE_SENSOR (1).
     * bdpt_strategy: which (s,t) strategy launched this subpath. */
    uint32_t         bdpt_subpath_id   = 0u;
    uint16_t         bdpt_vertex       = 0u;
    uint8_t          bdpt_stream       = 0u;
    uint8_t          bdpt_strategy     = 0u;

    /* Scalar amplitude fast-path: when amp_n_bands > 0 the amp Eigen vector
     * is NOT allocated.  All bands carry (amp_scalar, 0i).  Only valid for
     * initial submissions; GPU child intents are packed directly from SSBO.
     * The GPU dispatch pack loop checks this and skips the heap pointer. */
    float            amp_scalar        = 0.0f;
    int8_t           amp_n_bands       = 0;    /* 0 = use amp VXcd; >0 = use amp_scalar */
};

struct HitRecord {
    RayIntent        ray;
    Eigen::Vector3d  seg_start;
    Eigen::Vector3d  incoming_dir;
    double           path_at_seg_start = 0.0;
    int              hit_tri           = -1;
    uint32_t         hit_tri_flags     = 0u;
    Eigen::Vector3d  hit_pos;
    Eigen::Vector3d  hit_n;
    double           t_hit             = 0.0;
    Eigen::VectorXcd amp_propagated;
    bool             is_sensor_hit     = false;
    bool             is_emissive_hit   = false;
    int              sensor_group_id   = -1;
};

struct RefinedHit {
    HitRecord        base;
    Eigen::Vector3d  refined_pos;
    Eigen::Vector3d  refined_n;
    bool             was_parametric = false;
};

struct ChildRay {
    RayIntent intent;
};

struct WaveIntent {
    RayIntent ray;
    int       arena_id = -1;
};

/* ─── Per-stage profiling with adaptive batch control ───────────────────── */

/* Lockfree stats kept by each pipeline stage.
 *
 * CPU path: batch_sz adapts based on queue depth (doubles when queue is deep,
 * halves when thin).
 *
 * GPU path: batch_sz_gpu is the preferred GPU pop size; it adapts independently.
 * gpu_fraction (stored as IEEE-754 float bits in an atomic uint32) tracks the
 * observed fraction of items processed by the GPU vs CPU.  It is updated after
 * each batch: if GPU throughput > CPU throughput the fraction drifts toward 1,
 * otherwise toward 0.  The routing call-site reads gpu_fraction to decide
 * probabilistically whether to push a new item onto the GPU or CPU queue.
 *
 * Both CPU and GPU workers pop from the same upstream queue (competition model);
 * gpu_fraction is therefore an *observed* metric, not a directive.  Callers that
 * want a hard split can override it via set_gpu_fraction(). */
struct StageStats {
    /* CPU */
    std::atomic<uint64_t> n_processed{0};
    std::atomic<uint64_t> ns_active{0};       /* total ns spent on CPU          */
    std::atomic<int>      batch_sz{64};        /* adaptive CPU pop batch size    */
    std::atomic<int>      q_depth{0};          /* last observed upstream q len   */
    /* GPU */
    std::atomic<uint64_t> n_gpu{0};            /* items processed by GPU dispatch */
    std::atomic<uint64_t> ns_gpu{0};           /* total ns spent on GPU dispatch  */
    std::atomic<int>      batch_sz_gpu{262144}; /* adaptive GPU pop batch size (start large) */
    /* Observed GPU fraction [0,1] stored as IEEE-754 float bits */
    std::atomic<uint32_t> gpu_frac_bits{0};    /* reinterpret as float            */

    void record(int n, uint64_t elapsed_ns, int q_now) noexcept {
        n_processed.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
        ns_active  .fetch_add(elapsed_ns,               std::memory_order_relaxed);
        q_depth.store(q_now, std::memory_order_relaxed);
        /* Adapt CPU batch size */
        int cur = batch_sz.load(std::memory_order_relaxed);
        if      (q_now > cur * 4 && cur < 512) batch_sz.store(cur * 2, std::memory_order_relaxed);
        else if (q_now < 2       && cur > 4)   batch_sz.store(cur / 2, std::memory_order_relaxed);
        _update_gpu_fraction();
    }

    void record_gpu(int n, uint64_t elapsed_ns, int q_now) noexcept {
        n_gpu.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
        ns_gpu.fetch_add(elapsed_ns,              std::memory_order_relaxed);
        q_depth.store(q_now, std::memory_order_relaxed);
        /* Adapt GPU batch size as a preferred pop capacity, not as a reaction to
         * finite burst tails.  Staged exposure often drains Q_intent to zero; if
         * we shrink on low post-pop depth, the next exposure packet is forced
         * through tiny dispatches. */
        int cur = batch_sz_gpu.load(std::memory_order_relaxed);
        if (q_now > cur && cur < 1048576)
            batch_sz_gpu.store(std::min(cur * 2, 1048576), std::memory_order_relaxed);
        _update_gpu_fraction();
    }

    /* Manually pin the GPU fraction (0=all CPU, 1=all GPU). */
    void set_gpu_fraction(float f) noexcept {
        uint32_t bits;
        memcpy(&bits, &f, 4);
        gpu_frac_bits.store(bits, std::memory_order_relaxed);
    }

    float gpu_fraction() const noexcept {
        uint32_t bits = gpu_frac_bits.load(std::memory_order_relaxed);
        float f; memcpy(&f, &bits, 4); return f;
    }

    double items_per_sec() const noexcept {
        uint64_t ns = ns_active.load(std::memory_order_relaxed);
        return ns ? static_cast<double>(n_processed.load(std::memory_order_relaxed))
                    * 1e9 / static_cast<double>(ns)
                  : 0.0;
    }

    double gpu_items_per_sec() const noexcept {
        uint64_t ns = ns_gpu.load(std::memory_order_relaxed);
        return ns ? static_cast<double>(n_gpu.load(std::memory_order_relaxed))
                    * 1e9 / static_cast<double>(ns)
                  : 0.0;
    }

private:
    void _update_gpu_fraction() noexcept {
        double cpu_tp = items_per_sec();
        double gpu_tp = gpu_items_per_sec();
        double total  = cpu_tp + gpu_tp;
        float frac    = (total > 0.0) ? static_cast<float>(gpu_tp / total) : 0.0f;
        /* Exponential moving average with alpha=0.1 to smooth outliers */
        float prev = gpu_fraction();
        float next = prev * 0.9f + frac * 0.1f;
        uint32_t bits; memcpy(&bits, &next, 4);
        gpu_frac_bits.store(bits, std::memory_order_relaxed);
    }
};

/* Snapshot of all pipeline stages + output queue, returned to callers. */
struct RayPipelineStats {
    struct Stage {
        double   throughput;       /* CPU items / second */
        uint64_t processed;        /* CPU items processed */
        int      batch_size;       /* current CPU batch size */
        int      queue_depth;      /* last observed upstream queue depth */
        double   gpu_throughput;   /* GPU items / second */
        uint64_t gpu_processed;    /* GPU items processed */
        int      gpu_batch_size;   /* current GPU batch size */
        float    gpu_fraction;     /* observed fraction going to GPU [0,1] */
        double   cpu_active_ms;    /* cumulative CPU stage wall time */
        double   gpu_active_ms;    /* cumulative GPU stage wall time */
    } t1, t2, t3, t4, t5;
    int output_queue_depth;
    int intent_queue_depth;
    int intent_queue_done;
    int in_flight;
    uint64_t gpu_uv_readback_bytes;
    uint64_t gpu_uv_readback_count;
    uint64_t gpu_hit_readback_bytes;
    uint64_t gpu_hit_readback_count;
};

/* ─── Output record ─────────────────────────────────────────────────────── */

static constexpr int RAY_RECORD_MAX_BANDS = 32;

enum class RayRecordKind : uint8_t {
    STRIKE   = 0,  /* intersection event — every bounce, from T1              */
    TERMINAL = 1,  /* ray path ended in T3 (absorbed/aperture/budget/dim)     */
    MISS     = 2,  /* ray escaped all geometry — from T1                      */
    FIELD    = 3,  /* wave-solver traversal — from T4                         */
};

/* Flat POD record pushed to the output queue by all stages.
 * float32 for positions/amplitudes keeps the struct compact and cache-friendly.
 * seg_start+pos give both endpoints of a ray segment for visualisation. */
struct RayRecord {
    RayRecordKind kind              = RayRecordKind::TERMINAL;
    uint64_t      tag               = 0;
    int32_t       src_id            = 0;
    int32_t       bounce            = 0;
    float         seg_start[3]      = {};  /* ray origin entering this segment */
    float         pos[3]            = {};  /* hit position (STRIKE/TERMINAL) or last pos */
    float         dir[3]            = {};  /* incoming ray direction            */
    float         normal[3]         = {};  /* surface normal at hit (STRIKE)    */
    float         path_len          = 0.f; /* cumulative path length at hit     */
    float         path_at_seg_start = 0.f;
    int32_t       hit_tri           = -1;
    int32_t       hit_group_id      = -1;  /* tri_param_group_of_tri[hit_tri], or -1 */
    int32_t       mat_idx           = -1;
    float         bary_u            = 0.f; /* Möller–Trumbore u at hit (STRIKE) */
    float         bary_v            = 0.f; /* Möller–Trumbore v at hit (STRIKE) */
    int32_t       arena_id          = -1;  /* FIELD: which wave arena           */
    bool          is_sensor         = false;
    int32_t       sensor_group_id   = -1;
    int32_t       n_bands           = 0;
    float         amp_re[RAY_RECORD_MAX_BANDS] = {};
    float         amp_im[RAY_RECORD_MAX_BANDS] = {};
    uint8_t       color_flag        = 0;   /* propagated from submitting RayIntent */
    float         sensor_origin_y   = 0.f; /* camera-stream launch site on sensor */
    float         sensor_origin_z   = 0.f;
};

/* ─── Thread-safe pipeline queue ────────────────────────────────────────── */

template<typename T>
class PipelineQueue {
public:
    bool push(T item) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (done_) return false;
            q_.push_back(std::move(item));
        }
        cv_.notify_one();
        return true;
    }

    bool push_front(T item) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (done_) return false;
            q_.push_front(std::move(item));
            ++n_high_priority_;
        }
        cv_.notify_one();
        return true;
    }

    void push_many(std::vector<T>& items) {
        if (items.empty()) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (done_) {
                items.clear();
                return;
            }
            for (T& item : items)
                q_.push_back(std::move(item));
        }
        cv_.notify_one();
        items.clear();
    }

    template<typename KeyFn>
    void push_many_priority(std::vector<T>& items, KeyFn key_fn, float front_threshold = 1.0f) {
        if (items.empty()) return;
        std::vector<T> high;
        std::vector<T> normal;
        high.reserve(items.size());
        normal.reserve(items.size());
        for (T& item : items) {
            if (key_fn(item) > front_threshold)
                high.push_back(std::move(item));
            else
                normal.push_back(std::move(item));
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (done_) {
                items.clear();
                return;
            }
            for (auto it = high.rbegin(); it != high.rend(); ++it)
                q_.push_front(std::move(*it));
            for (T& item : normal)
                q_.push_back(std::move(item));
        }
        cv_.notify_one();
        items.clear();
    }

    /* Bounded push: blocks (spin-sleep 500 µs) until size < max_size.
     * Provides backpressure on the submit thread without busy-spinning the CPU.
     * max_size <= 0 falls through to unbounded push. */
    bool push_bounded(T item, int max_size) {
        if (max_size > 0) {
            for (;;) {
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (done_) return false;
                    if ((int)q_.size() < max_size) {
                        q_.push_back(std::move(item));
                        cv_.notify_one();
                        return true;
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        }
        return push(std::move(item));
    }

    /* Blocking pop.  Brief spin before sleeping — reduces CV overhead on hot queues.
     * Spin count kept small (8) to avoid mutex thrashing when the queue is idle.
     * 512 iterations with a lock/yield per iter caused max-CPU on multi-threaded paths. */
    bool pop(T& out) {
        for (int s = 0; s < 8; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (done_) return false;
                if (!q_.empty()) { out = std::move(q_.front()); q_.pop_front(); return true; }
            }
            std::this_thread::yield();
        }
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return !q_.empty() || done_; });
        if (done_ || q_.empty()) return false;
        out = std::move(q_.front()); q_.pop_front(); return true;
    }

    /* Blocking batch pop — waits for ≥1 item, takes up to max_n.
     * Appends to out.  Returns count taken; 0 means done+empty. */
    int pop_batch(std::vector<T>& out, int max_n) {
        for (int s = 0; s < 8; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (done_) return 0;
                if (!q_.empty()) {
                    int n = std::min(max_n, (int)q_.size());
                    for (int i = 0; i < n; ++i) { out.push_back(std::move(q_.front())); q_.pop_front(); }
                    return n;
                }
            }
            std::this_thread::yield();
        }
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return !q_.empty() || done_; });
        if (done_ || q_.empty()) return 0;
        int n = std::min(max_n, (int)q_.size());
        for (int i = 0; i < n; ++i) { out.push_back(std::move(q_.front())); q_.pop_front(); }
        return n;
    }

    /* Timed batch pop — same as pop_batch but returns -1 on timeout instead
     * of blocking indefinitely.  The caller can service side-channel work
     * (e.g. a T5 GPU job) and then loop back.  Returns 0 only when done.  */
    int pop_batch_timed(std::vector<T>& out, int max_n,
                        int timeout_ms = 5) {
        for (int s = 0; s < 8; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (done_) return 0;
                if (!q_.empty()) {
                    int n = std::min(max_n, (int)q_.size());
                    for (int i = 0; i < n; ++i) { out.push_back(std::move(q_.front())); q_.pop_front(); }
                    return n;
                }
            }
            std::this_thread::yield();
        }
        std::unique_lock<std::mutex> lk(mu_);
        const bool fired = cv_.wait_for(
            lk, std::chrono::milliseconds(timeout_ms),
            [this]{ return !q_.empty() || done_; });
        if (done_ && q_.empty()) return 0;
        if (!fired || q_.empty()) return -1;   /* timeout — nothing yet */
        int n = std::min(max_n, (int)q_.size());
        for (int i = 0; i < n; ++i) { out.push_back(std::move(q_.front())); q_.pop_front(); }
        return n;
    }

    template<typename KeyFn>
    int pop_batch_exclusive_priority(std::vector<T>& out, int max_n, KeyFn key_fn,
                                     float threshold = 1.0f) {
        if (max_n <= 0) return 0;
        for (int s = 0; s < 8; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (done_) return 0;
                if (!q_.empty()) goto take;
            }
            std::this_thread::yield();
        }
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]{ return !q_.empty() || done_; });
            if (done_ || q_.empty()) return 0;
        }
    take:
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (done_) return 0;
            /* Skip O(n) scan when no push_front items exist in queue. */
            const bool has_hi = n_high_priority_ > 0;
            int n = 0;
            if (has_hi) {
                for (auto it = q_.begin(); it != q_.end() && n < max_n; ) {
                    if (key_fn(*it) > threshold) {
                        out.push_back(std::move(*it));
                        it = q_.erase(it);
                        --n_high_priority_;
                        ++n;
                    } else {
                        ++it;
                    }
                }
            } else {
                n = std::min(max_n, (int)q_.size());
                for (int i = 0; i < n; ++i) {
                    out.push_back(std::move(q_.front()));
                    q_.pop_front();
                }
            }
            return n;
        }
    }

    template<typename KeyFn>
    int pop_batch_exclusive_priority_timed(std::vector<T>& out, int max_n, KeyFn key_fn,
                                           int timeout_ms = 5, float threshold = 1.0f) {
        if (max_n <= 0) return 0;
        for (int s = 0; s < 8; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (done_) return 0;
                if (!q_.empty()) goto take;
            }
            std::this_thread::yield();
        }
        {
            std::unique_lock<std::mutex> lk(mu_);
            const bool fired = cv_.wait_for(
                lk, std::chrono::milliseconds(timeout_ms),
                [this]{ return !q_.empty() || done_; });
            if (done_ && q_.empty()) return 0;
            if (!fired || q_.empty()) return -1;
        }
    take:
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (done_ && q_.empty()) return 0;
            if (q_.empty()) return -1;
            /* Skip O(n) scan when no push_front items exist in queue. */
            const bool has_hi = n_high_priority_ > 0;
            int n = 0;
            if (has_hi) {
                for (auto it = q_.begin(); it != q_.end() && n < max_n; ) {
                    if (key_fn(*it) > threshold) {
                        out.push_back(std::move(*it));
                        it = q_.erase(it);
                        --n_high_priority_;
                        ++n;
                    } else {
                        ++it;
                    }
                }
            } else {
                n = std::min(max_n, (int)q_.size());
                for (int i = 0; i < n; ++i) {
                    out.push_back(std::move(q_.front()));
                    q_.pop_front();
                }
            }
            return n;
        }
    }

    /* Non-blocking drain — takes up to max_n items immediately.
     * Appends to out.  Returns count taken (0 if empty). */
    int drain(std::vector<T>& out, int max_n) {
        std::lock_guard<std::mutex> lk(mu_);
        int n = std::min(max_n, (int)q_.size());
        for (int i = 0; i < n; ++i) { out.push_back(std::move(q_.front())); q_.pop_front(); }
        return n;
    }

    template<typename KeyFn>
    int drain_exclusive_priority(std::vector<T>& out, int max_n, KeyFn key_fn,
                                 float threshold = 1.0f) {
        std::lock_guard<std::mutex> lk(mu_);
        if (max_n <= 0 || q_.empty()) return 0;
        const bool has_hi = n_high_priority_ > 0;
        int n = 0;
        if (has_hi) {
            for (auto it = q_.begin(); it != q_.end() && n < max_n; ) {
                if (key_fn(*it) > threshold) {
                    out.push_back(std::move(*it));
                    it = q_.erase(it);
                    --n_high_priority_;
                    ++n;
                } else {
                    ++it;
                }
            }
        } else {
            n = std::min(max_n, (int)q_.size());
            for (int i = 0; i < n; ++i) {
                out.push_back(std::move(q_.front()));
                q_.pop_front();
            }
        }
        return n;
    }

    /* Shuffled batch pop: waits for ≥1 item, then picks a random contiguous
     * window of up to max_n elements anywhere in the queue.
     * shuffle_frac in [0,1]: 0 = pure FIFO (falls through to pop_batch),
     * 1 = window start uniformly random over whole queue.
     * Returns count taken; 0 means done+empty. */
    int pop_batch_shuffled(std::vector<T>& out, int max_n, float shuffle_frac,
                           std::mt19937& rng) {
        if (shuffle_frac <= 0.0f)
            return pop_batch(out, max_n);

        /* Spin-wait then CV-wait for at least one item (same as pop_batch). */
        for (int s = 0; s < 8; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (done_) return 0;
                if (!q_.empty()) goto take;
            }
            std::this_thread::yield();
        }
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]{ return !q_.empty() || done_; });
            if (done_ || q_.empty()) return 0;
        }
    take:
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (done_) return 0;
            int sz = (int)q_.size();
            if (sz == 0) return 0;
            int n      = std::min(max_n, sz);
            /* Slide the window start by a random fraction of the slack. */
            int slack  = sz - n;
            int offset = 0;
            if (slack > 0 && shuffle_frac > 0.0f) {
                int max_off = (int)std::ceil(slack * shuffle_frac);
                max_off = std::max(1, std::min(slack, max_off));
                offset  = (int)(std::uniform_int_distribution<int>(0, max_off - 1)(rng));
            }
            /* Move window to front by rotating offset items to the back, take
             * n items, then rotate the remainder back to preserve order. */
            for (int i = 0; i < offset; ++i) {
                q_.push_back(std::move(q_.front())); q_.pop_front();
            }
            for (int i = 0; i < n; ++i) {
                out.push_back(std::move(q_.front())); q_.pop_front();
            }
            /* Undo the rotation so the items we skipped stay at the front. */
            for (int i = 0; i < offset; ++i) {
                q_.push_front(std::move(q_.back())); q_.pop_back();
            }
            return n;
        }
    }

    /* Priority-biased batch pop: grabs up to max_n*oversample items, sorts by
     * key_fn(item) descending outside the lock, emits the top max_n to out,
     * and returns any remainder to the front of the queue in priority order.
     * Higher-priority intents are therefore processed before lower-priority
     * ones even within an otherwise-FIFO queue.
     * Returns count emitted; 0 means done+empty. */
    template<typename KeyFn>
    int pop_batch_priority(std::vector<T>& out, int max_n, KeyFn key_fn,
                           int oversample = 4) {
        if (max_n <= 0) return 0;
        for (int s = 0; s < 8; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (done_) return 0;
                if (!q_.empty()) goto priority_take;
            }
            std::this_thread::yield();
        }
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]{ return !q_.empty() || done_; });
            if (done_ || q_.empty()) return 0;
        }
    priority_take:
        {
            std::vector<T> grabbed;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (done_) return 0;
                int sz   = (int)q_.size();
                if (sz == 0) return 0;
                int grab = std::min(max_n * oversample, sz);
                grabbed.reserve(grab);
                for (int i = 0; i < grab; ++i) {
                    grabbed.push_back(std::move(q_.front())); q_.pop_front();
                }
            }
            std::sort(grabbed.begin(), grabbed.end(),
                      [&](const T& a, const T& b){ return key_fn(a) > key_fn(b); });
            int take = std::min(max_n, (int)grabbed.size());
            for (int i = 0; i < take; ++i)
                out.push_back(std::move(grabbed[i]));
            if (take < (int)grabbed.size()) {
                std::lock_guard<std::mutex> lk(mu_);
                if (done_) return take;
                for (int i = (int)grabbed.size() - 1; i >= take; --i)
                    q_.push_front(std::move(grabbed[i]));
                cv_.notify_one();
            }
            return take;
        }
    }

    int size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return (int)q_.size();
    }

    bool is_done() const {
        std::lock_guard<std::mutex> lk(mu_);
        return done_;
    }

    /* Signal that no new items will be pushed.  Unblocks blocked pops/pop_batch
     * calls so workers can drain and exit.  Does NOT clear the queue: items
     * pushed before set_done() remain available for drain() callers.  This is
     * intentional — clearing accumulated data as a side-effect of signalling
     * shutdown discards in-flight records (e.g. BDPT PDFs) before consumers
     * (e.g. T5) have had a chance to read them. */
    void set_done() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_ = true;
            /* DO NOT clear q_ here — see comment above. */
        }
        cv_.notify_all();
    }

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<T>           q_;
    bool                    done_ = false;
    int                     n_high_priority_ = 0;  /* items pushed via push_front */
};

/* ─── Wave solver arena ─────────────────────────────────────────────────── */

struct WaveArena {
    int              id      = -1;
    Eigen::Vector3d  center;
    double           radius  = 0.0;
    double           radius_sq = 0.0;
    double           n_real  = 1.0;
    Eigen::Vector3d  axis_z;
    Eigen::Vector3d  axis_x, axis_y;
    int              nx = 0, ny = 0;
    int              nz = 0;
    double           dx = 0.0;
    double           dz = 0.0;
    int              n_bands = 0;
    double           wavelengths_m[32] = {};
    std::vector<float> re_buf;
    std::vector<float> im_buf;
    std::mutex         mu;
};

/* ── T5 GPU connection pass types ────────────────────────────────────────── */
static constexpr int T5_LGV_STRIDE    = 56;  /* light vertex, floats (must match shader) */
static constexpr int T5_CGV_STRIDE    = 72;  /* camera vertex, floats (must match shader) */
static constexpr int T5_TILE_C        =  8;  /* WG X dim: camera verts per tile          */
static constexpr int T5_TILE_L        =  8;  /* WG Y dim: light verts per tile            */
static constexpr int T5_MAX_GPU_BANDS = 32;  /* per-band betas packed into GPU vert buffers */
static constexpr int LGV_BAND_BASE    = 16;  /* first per-band beta field in LGV (field index) */
static constexpr int CGV_BAND_BASE    = 22;  /* first per-band beta field in CGV (field index) */

/* ── LGV named field offsets (stride 40) ────────────────────────────────── *
 * [0..2]   pos xyz                                                           *
 * [3..5]   normal xyz                                                        *
 * [6]      throughput_scalar                                                 *
 * [7]      flags (MAT_FLAG_*, bit-cast uint)                                 *
 * [8]      subpath_id (bit-cast uint)                                        *
 * [9]      vertex_index | (stream<<16)  (bit-cast uint)                      *
 * [10]     beta_lum  (sum of per-band |beta|, scalar throughput fallback)    *
 * [11]     pdf_fwd  (BdptVertexRecord::pdf_fwd, raw)                         *
 * [12]     pdf_rev  (BdptVertexRecord::pdf_rev, raw)                         *
 * [13]     BdptPdfRecord::flags  (BDPT_PDF_FLAG_*, bit-cast uint)            *
 * [14]     optical_block  (non-zero = absorb/TIR/clip; bit-cast uint)        *
 * [15]     prefix_pdf  (cumulative subpath forward-PDF up to this vertex)    *
 * [16..47] per-band beta magnitudes [band 0..31]  (LGV_BAND_BASE)           *
 * [48..50] dir_in xyz  (incident direction at this vertex)                   *
 * [51]     diffuse_p  (mat_cache_diffusion result)                           *
 * [52]     ggx_alpha  (surf_cache_ggx_alpha result)                          *
 * [53]     optical_jacobian  (phase-space jacobian product from optical LUT) *
 * [54]     edge_fwd_area  (area-domain fwd PDF: this vertex → next in path)  *
 * [55]     edge_bwd_area  (area-domain bwd PDF: next vertex → this in path)  */
static constexpr int LGV_DIR_IN_X       = 48;
static constexpr int LGV_DIFFUSE_P      = 51;
static constexpr int LGV_GGX_ALPHA      = 52;
static constexpr int LGV_OPT_JACOBIAN   = 53;
static constexpr int LGV_EDGE_FWD_AREA  = 54;
static constexpr int LGV_EDGE_BWD_AREA  = 55;

/* ── CGV named field offsets (stride 56) ────────────────────────────────── *
 * [0..2]   pos xyz                                                           *
 * [3..5]   normal xyz                                                        *
 * [6]      throughput_scalar                                                 *
 * [7]      flags (MAT_FLAG_*, bit-cast uint)                                 *
 * [8]      subpath_id (bit-cast uint)                                        *
 * [9]      vertex_index (bit-cast uint)                                      *
 * [10]     spectral_beta_r  (display-weighted, band_to_display_rgb sum)      *
 * [11]     spectral_beta_g                                                   *
 * [12]     spectral_beta_b                                                   *
 * [13]     sensor_origin_y                                                   *
 * [14]     sensor_origin_z                                                   *
 * [15]     pdf_fwd  (BdptVertexRecord::pdf_fwd, raw)                         *
 * [16]     pdf_rev  (BdptVertexRecord::pdf_rev, raw)                         *
 * [17]     BdptPdfRecord::flags  (BDPT_PDF_FLAG_*, bit-cast uint)            *
 * [18]     optical_block  (non-zero = absorb/TIR/clip; bit-cast uint)        *
 * [19]     prefix_pdf  (cumulative subpath forward-PDF up to this vertex)    *
 * [20]     mis_denom_sum  (reserved, 0.0)                                    *
 * [21]     tri_mat_idx  (int bits of material index)                         *
 * [22..53] per-band beta magnitudes [band 0..31]  (CGV_BAND_BASE)           *
 * [54..56] dir_in xyz                                                        *
 * [57]     diffuse_p                                                         *
 * [58]     ggx_alpha                                                         *
 * [59]     optical_jacobian                                                  *
 * [60]     edge_fwd_area                                                     *
 * [61]     edge_bwd_area                                                     *
 * [62..71] pad / reserved                                                    */
static constexpr int CGV_DIR_IN_X       = 54;
static constexpr int CGV_DIFFUSE_P      = 57;
static constexpr int CGV_GGX_ALPHA      = 58;
static constexpr int CGV_OPT_JACOBIAN   = 59;
static constexpr int CGV_EDGE_FWD_AREA  = 60;
static constexpr int CGV_EDGE_BWD_AREA  = 61;

/* GPU params block uploaded to binding 3 of t5_full_connect.comp.glsl.
 * std430 layout, 56 bytes. */
struct T5GpuParams {
    float    min_geom;         /* geometry-term floor                   */
    float    sensor_half_w;    /* sensor half-width  (Y axis, metres)   */
    float    sensor_half_h;    /* sensor half-height (Z axis, metres)   */
    uint32_t cam_offset;       /* first cam vert index for this cbatch  */
    uint32_t n_light_verts;    /* total light vertices                  */
    uint32_t n_cam_verts;      /* upper bound: cam_offset + cbatch size */
    int32_t  sensor_res;       /* pixel grid side (res×res image)       */
    uint32_t light_batch_size; /* light verts per dispatch (TDR guard)  */
    uint32_t light_offset;     /* first light vert index in this batch  */
    int32_t  n_bands;          /* number of spectral bands (≥1)         */
    int32_t  tile_x0;          /* pixel column of tile left edge        */
    int32_t  tile_y0;          /* pixel row of tile top edge            */
    int32_t  tile_w;           /* tile width  in pixels (0 = full res)  */
    int32_t  tile_h;           /* tile height in pixels (0 = full res)  */
};
static_assert(sizeof(T5GpuParams) == 56, "T5GpuParams layout mismatch");

/* ── Flash light modifier ─────────────────────────────────────────────────
 * Applied per-ray inside submit_emissive_triangles.  All modes use the same
 * interaction target (target_x/y/z/r) as the reference geometry.
 *
 * NONE  : no modification; cosine-hemisphere rays all submitted.
 * SNOOT : conic tube from each emitter point to the exit disk at the target.
 *         Rays that miss the disk are culled (replaces legacy sphere test).
 *         param0 unused; set target_r to control exit-disk radius.
 * GRID  : egg-crate square grid placed at the emitter face.
 *         param0 = cell diameter in mm (default 5).
 *         param1 = grid tube depth in mm (default 25).
 *         Rays that cannot exit their cell's square aperture are culled.
 * SCRIM : translucent attenuating diffuser.
 *         param0 = transmittance in [0,1] (default 0.5).
 *         Rays are accepted with probability = transmittance. */
enum class FlashModifierType : int {
    NONE  = 0,
    SNOOT = 1,
    GRID  = 2,
    SCRIM = 3,
};

/* ─── Pipeline config ───────────────────────────────────────────────────── */

struct RayPipelineConfig {
    double wave_grid_dx_m   = 2e-6;
    int    max_children     = 2;
    int    seed             = 42;
    /* Pipeline-wide amplitude floor. */
    double min_amplitude    = 1e-6;
    /* Bounded intent queue capacity (0 = unbounded). */
    int    max_intent_queue = 0;
    /* Intent-queue shuffle lever (0 = FIFO, 1 = fully random window). */
    float  intent_queue_shuffle = 0.0f;
    /* Sensor image accumulator: world-space plate geometry.
     * plate_x: sensor X position; plate_half_w/h: rectangular half-extents (Y/Z);
     * pip_res: pixel grid side (res×res); bdpt_eps: Y/Z proximity for BDPT snap in metres. */
    float  sensor_plate_x      = 0.0f;
    float  sensor_plate_half_w = 0.0f;
    float  sensor_plate_half_h = 0.0f;
    int    sensor_pip_res   = 0;      /* 0 = disabled */
    float  sensor_bdpt_eps  = 0.008f;

    /* ── BDPT side-queue caps ─────────────────────────────────────────────
     * When a side queue reaches the cap the incoming record is dropped and
     * the corresponding overflow counter is incremented.  0 = unlimited
     * (not recommended for production — will grow unbounded for any scene).
     * Typical small-scene budget: 1–4 M entries per queue. */
    int    bdpt_max_vertices = 10000000; /* BdptVertexRecord cap */
    int    bdpt_max_spectral = 25000000; /* BdptSpectralWeightRecord cap */
    int    bdpt_max_pdfs     = 10000000; /* BdptPdfRecord cap */
    int    bdpt_max_optical  = 10000000; /* BdptOpticalEventRecord cap */
    int    bdpt_max_connections = 10000000; /* BdptConnectionRecord cap */

    /* ── T5 connection ───────────────────────────────────────────────────
     * GPU path: t5_full_connect.comp.glsl — O(N_cam × N_light) brute force.
     * CPU fallback: run_t5_allpairs() via ThreadPool.
     * t5_min_geom: geometry-term floor; pairs below this are skipped. */
    float      t5_min_geom = 1e-8f;

    /* ── Flash modifier ─────────────────────────────────────────────────── */
    FlashModifierType flash_modifier_type   = FlashModifierType::SNOOT;
    float             flash_modifier_param0 = 0.0f;  /* GRID: cell_mm  SCRIM: transmittance */
    float             flash_modifier_param1 = 0.0f;  /* GRID: depth_mm                      */

    /* ── GPU compute dispatch ────────────────────────────────────────────
     * When use_gpu_compute=true a GlPipelineDispatch thread is spawned that
     * competes with the CPU workers on the shared intent/hit/refined queues.
     * Both CPU and GPU workers remain active; adaptive batch sizing and the
     * gpu_fraction stat in StageStats track which operator handles more work.
     *
     * shader_dir: directory containing the four .comp.glsl shader files.
     *   Empty string → searches "csrc/shaders/" relative to cwd.
     *
     * gpu_batch_size_t{1,2,3,4,5}: initial GPU pop batch size per stage.
     *   0 = use the StageStats default (262144). */
    bool        use_gpu_compute    = false;
    std::string shader_dir;
    int         gpu_batch_size_t1  = 0;
    int         gpu_batch_size_t2  = 0;
    int         gpu_batch_size_t3  = 0;
    int         gpu_batch_size_t4  = 0;
    int         gpu_batch_size_t5  = 0;
    uint32_t    t5_light_batch_size  = 0;  /* 0 = use built-in default (T5_LIGHT_BATCH) */
    uint32_t    t5_cam_batch_size    = 0;  /* 0 = use built-in default (8192)           */
    uint32_t    t5_sensor_tile_size  = 0;  /* 0 = use built-in default (128)            */

    /* Fraction of work to pin to GPU per stage (0=compete freely, >0=soft target).
     * 0.0 = CPU and GPU compete naturally on the shared queue.
     * 1.0 = route all new intents/hits to the GPU sub-queues; CPU workers idle. */
    float       gpu_fraction_t1    = 0.0f;
    float       gpu_fraction_t2    = 0.0f;
    float       gpu_fraction_t3    = 0.0f;
    float       gpu_fraction_t5    = 0.0f;

    /* Optional legacy CPU field-grid updater for GPU hit segments.  When true
     * the GPU worker reads hit records back to CPU and feeds full segments into
     * accumulate_field_capture_segment.  Off by default: the GPU display path
     * accumulates from GPU hit records without a full CPU field-grid walk. */
    bool        gpu_segment_field_capture = false;

    /* When true (and use_gpu_compute=true) the CPU T1/T2/T3 workers are NOT
     * spawned; the GPU handles all intent→hit→material work exclusively.
     * This eliminates CPU/GPU competition and prevents CPU thrashing when the
     * GPU is capable of processing all work.  T4 wave-solver still runs on GPU
     * when use_gpu_compute=true.  If GPU init or required shader loading fails,
     * this mode fails pipeline creation instead of falling back to CPU. */
    bool        gpu_all_stages = false;

    /* When true, skip per-bounce terminal-record, hit-record, and T5 tile
     * readbacks to CPU.  Enabled for production GPU-resident rendering; the
     * display image goes dark until a GPU-side sensor accumulator is wired
     * (Pass B/C).  Has no effect when use_gpu_compute is false.
     * Settable at runtime via ray_pipeline_set_skip_record_readback(). */
    bool        gpu_skip_record_readback = false;

    /* Handle to the display GL context (e.g. Pygame's HGLRC on Windows).
     * When non-zero the compute context is created as a share partner of this
     * context so all GL objects (SSBOs, textures) are visible in both.
     * Must be set before the first submit_rays call.  Set via the pybind
     * method set_gl_display_hglrc() on the Python-facing RayTracer object. */
    uint64_t    gl_display_hglrc = 0;
    /* HDC of the Pygame display window.  When non-zero, the hidden compute
     * window copies this DC's pixel format so wglCreateContextAttribsARB
     * finds compatible formats and succeeds even with stricter drivers. */
    uint64_t    gl_display_hdc   = 0;
};

/* ─── Opaque pipeline state (defined in ray_tracer.cpp) ─────────────────── */

struct RayPipelineState;

/* Thin accessors so callers don't need the full RayPipelineState definition. */

/* Number of spectral bands in the associated tracer (0 if none). */
int ray_pipeline_n_bands(const RayPipelineState* ps);

/* Return the latest completed shared UV texture object ID (0 if none is ready). */
uint64_t ray_pipeline_get_uv_pages_tex_id(const RayPipelineState* ps);

/* Return the latest shared GPU field-display volume texture (GL_TEXTURE_3D), or 0. */
uint64_t ray_pipeline_get_field_display_tex_id(const RayPipelineState* ps);

/* Ask the GPU dispatch thread to clear the field-display accumulator. */
void ray_pipeline_request_field_display_clear(RayPipelineState* ps);

/* Upload per-band RGB weights for the GPU UV blit shader.
 * weights : float array of length n_bands*3 (interleaved r,g,b per band).
 * n_bands : number of bands (clamped to [1, MAX_SPECTRAL_BANDS]).
 * mode    : 0=combined, 1=forward only, 2=sensor only. */
void ray_pipeline_set_uv_blit_weights(RayPipelineState* ps,
                                       const float* weights,
                                       int n_bands,
                                       int mode);

/* Toggle the production readback-skip flag at runtime.
 * Safe to call concurrently; takes effect on the next GPU bounce. */
void ray_pipeline_set_skip_record_readback(RayPipelineState* ps, bool skip);

/* ── Pass E: explicit debug readback taps ──────────────────────────────── *
 * Returns a flat float32 buffer from the last completed GPU batch.        *
 * Blocks until the GPU thread services the request (≤ ~5 ms idle window). *
 * Never call from the normal frame loop.                                  */
std::vector<float> ray_pipeline_debug_read_hits        (RayPipelineState* ps, int max_n);
std::vector<float> ray_pipeline_debug_read_terminals   (RayPipelineState* ps, int max_n);
std::vector<float> ray_pipeline_debug_read_bdpt_vertices(RayPipelineState* ps, int max_n);

/* Feed display frame timing back into the GPU producer governor.
 * frame_ms  : most recent interactive frame time.
 * target_ms : desired frame budget, e.g. 16.667 for 60 Hz or 33.333 for 30 Hz.
 * The pipeline responds by shrinking/growing GPU batch sizes and UV update cadence.
 */
void ray_pipeline_report_display_frame_time(RayPipelineState* ps,
                                            double frame_ms,
                                            double target_ms);

/* mat_idx of triangle tri_idx (-1 if ps/st is null or index out of range). */
int ray_pipeline_tri_mat_idx(const RayPipelineState* ps, int tri_idx);

/* Non-blocking drain from Q_refined: pops up to max_n into out.
 * Returns count appended (0 if queue empty). */
int ray_pipeline_drain_refined(RayPipelineState* ps,
                                std::vector<RefinedHit>& out,
                                int max_n);

/* ── BDPT side-data drains (non-blocking, pop up to max_n each) ─────────────
 * Each function is independent of the others.  Returns count appended (0 if
 * the corresponding queue is empty).  These queues are never affected by
 * ray_pipeline_drain() or drain_records_slim() on the Python side. */
int ray_pipeline_drain_bdpt_vertices(RayPipelineState* ps,
                                      std::vector<BdptVertexRecord>& out,
                                      int max_n);
int ray_pipeline_drain_bdpt_spectral(RayPipelineState* ps,
                                      std::vector<BdptSpectralWeightRecord>& out,
                                      int max_n);
int ray_pipeline_drain_bdpt_pdfs(RayPipelineState* ps,
                                  std::vector<BdptPdfRecord>& out,
                                  int max_n);
int ray_pipeline_drain_bdpt_optical(RayPipelineState* ps,
                                     std::vector<BdptOpticalEventRecord>& out,
                                     int max_n);
int ray_pipeline_drain_bdpt_connections(RayPipelineState* ps,
                                         std::vector<BdptConnectionRecord>& out,
                                         int max_n);

/* Snapshot cumulative overflow counts for BDPT side queues.
 * Any out-pointer may be null.  Counts only increase; never wrap.
 * A non-zero count means records were silently dropped at the cap. */
void ray_pipeline_get_bdpt_overflow(const RayPipelineState* ps,
                                     uint64_t* out_vertices,
                                     uint64_t* out_spectral,
                                     uint64_t* out_pdfs,
                                     uint64_t* out_optical,
                                     uint64_t* out_connections);

/* Run BDPT connection pass with balance-heuristic MIS.
 * Drains Q_bdpt_vertices, Q_bdpt_spectral, Q_bdpt_pdfs and evaluates all
 * valid (sensor-stream vertex, light-stream vertex) strategies accumulated
 * since the last call.  Results accumulate into sensor channel 2.
 * Call once per sensor sweep after ray_pipeline_wait_idle(). */
void ray_pipeline_run_bdpt_connection(RayPipelineState* ps);

/* Signal that the camera has finished dispatching emissive-triangle (flash)
 * rays for the current exposure substage.  T5 fires once both
 * signal_flash_dispatched and signal_sensor_dispatched have been called for
 * the same substage — firing is serialised on a dedicated T5 thread. */
void ray_pipeline_signal_flash_dispatched(RayPipelineState* ps);

/* Signal that the camera has finished dispatching sensor-sweep rays for the
 * current exposure substage.  Mirrors signal_flash_dispatched. */
void ray_pipeline_signal_sensor_dispatched(RayPipelineState* ps);

/* Block until the T5 worker thread has finished.  Call after signalling both
 * flash and sensor to ensure ch2 of sensor_accum is fully written before
 * reading the sensor image. */
void ray_pipeline_join_t5(RayPipelineState* ps);

/* GPU dispatch health state (safe to call from any thread / Python):
 *   0 = disabled (use_gpu_compute=false)
 *   1 = init failed (GL context creation or shader compile error)
 *   2 = thread spawned, make_current not yet attempted
 *   3 = thread running (make_current succeeded, scene uploaded)
 *   4 = thread failed (make_current failed in worker thread)
 *   5 = thread exited normally (Q_intent exhausted) */
int ray_pipeline_gpu_dispatch_state(const RayPipelineState* ps);

/* Live-update T5 connection-pass configuration. */
void ray_pipeline_set_t5_min_geom(RayPipelineState* ps, float v);
void ray_pipeline_set_t5_light_batch_size(RayPipelineState* ps, uint32_t n);
void ray_pipeline_set_t5_cam_batch_size(RayPipelineState* ps, uint32_t n);
void ray_pipeline_set_t5_sensor_tile_size(RayPipelineState* ps, uint32_t n);
void ray_pipeline_set_force_cpu_t5(RayPipelineState* ps, bool v);

/* Live-update the flash light modifier applied in submit_emissive_triangles. */
void ray_pipeline_set_flash_modifier(RayPipelineState* ps, FlashModifierType type,
                                      float param0, float param1);

/* Submit one native sensor-frame sweep into the same T1->T2->T3 pipeline.
 * pix_offset: first tiled-Morton film/pupil schedule index.
 * max_rays:   cap on schedule entries submitted (0 = from pix_offset onward).
 * aperture_samples: number of deterministic pupil placements interleaved per film.
 * aperture_seed: 0 = center if aperture_samples == 1, otherwise schedule base.
 * Returns number of submitted intents. */
int ray_pipeline_submit_sensor_sweep(RayPipelineState* ps,
                                     int max_bounces,
                                     double min_amplitude,
                                     int max_rays,
                                     int pix_offset,
                                     int aperture_samples,
                                     uint64_t aperture_seed,
                                     int shutter_mode,
                                     double shutter_open,
                                     double shutter_center_u,
                                     double shutter_center_v,
                                     double shutter_softness,
                                     double exposure_weight);

/* Sensor-batching: call begin before multi-pass sensor sweeps on the same
 * flash, end (or frame reset) when done.  begin clears any stash; end clears
 * it again.  Between begin and end, run_bdpt_connection() automatically
 * stashes light records on the first pass and reuses them on subsequent ones. */
void ray_pipeline_begin_sensor_batching(RayPipelineState* ps);
void ray_pipeline_end_sensor_batching(RayPipelineState* ps);

RayPipelineState* ray_pipeline_create(
    RayTracerState*          st,
    const RayPipelineConfig* cfg);

void ray_pipeline_destroy(RayPipelineState* ps);

/* Non-blocking submit: push intents into the T1 input queue. */
void ray_pipeline_submit(
    RayPipelineState*  ps,
    const RayIntent*   intents,
    int                n_intents);

/* Native forward-light launcher: samples authored emissive triangles inside C++
 * and submits RayIntents directly to the persistent T1 queue.  This avoids the
 * Python path building origin/direction/amplitude arrays for every exposure
 * stage.  Returns the number of intents submitted. */
int ray_pipeline_submit_emissive_triangles(
    RayPipelineState* ps,
    const int*        tri_ids,
    int               n_tris,
    int               rays_per_tri,
    double            exposure_weight,
    double            emitter_amp_gain,
    int               max_bounces,
    double            min_amplitude,
    double            interaction_target_x,
    double            interaction_target_y,
    double            interaction_target_z,
    double            interaction_target_r,
    uint32_t          seed);

/* Non-blocking drain: pop up to max_n records from the output queue.
 * Appends to out.  Returns count appended (0 if output queue is empty). */
int ray_pipeline_drain(
    RayPipelineState*        ps,
    std::vector<RayRecord>&  out,
    int                      max_n);

/* How many ray paths are currently live inside the pipeline. */
int ray_pipeline_in_flight(const RayPipelineState* ps);

/* Snapshot throughput / batch-size / queue-depth for all four stages. */
void ray_pipeline_get_stats(const RayPipelineState* ps, RayPipelineStats* out);

/* Update the pipeline-wide amplitude floor and rebuild material epsilon flags. */
void ray_pipeline_set_min_amplitude(RayPipelineState* ps, double eps);

/* (Re)scan material reflectances and mark those whose max |refl| < min_amplitude
 * as epsilon-kill materials so T3 can skip child spawning immediately. */
void ray_pipeline_precompute_epsilon_flags(RayPipelineState* ps);

/* Set the intent-queue shuffle lever (0 = FIFO, 1 = fully random window). */
void ray_pipeline_set_shuffle(RayPipelineState* ps, float shuffle_frac);

/* Configure sensor image accumulator.  Call before submitting rays.
 * plate_x: world X of sensor plane; plate_half_w/h: rectangular half-extents (Y/Z);
 * res: pixel grid side (res×res RGB); bdpt_eps: world-space YZ proximity for BDPT snap.
 * Resets the accumulator buffer. */
void ray_pipeline_configure_sensor_image(
    RayPipelineState* ps,
    float plate_x, float plate_half_w, float plate_half_h,
    int   res,
    float bdpt_eps,
    float target_x,
    float target_r,
    float target_y,
    float target_z,
    int   target_mode);

/* Copy current sensor image into caller-owned float32 buffer.
 * buf must hold res*res*3 floats (row-major RGB).
 * Applies log tone-map: log(1 + v*9)/log(10), normalised to [0,1].
 * Sets *out_res to the configured resolution. Thread-safe. */
void ray_pipeline_get_sensor_image(
    const RayPipelineState* ps,
    float* buf,
    int*   out_res);

/* Copy the sugar-auxin priority map as a float32 (res×res) array.
 * Values ≥ 1 indicate regions of recent BDPT convergence; 1.0 = baseline.
 * buf must hold res*res floats.  Sets *out_res to the resolution. */
void ray_pipeline_get_priority_map(
    const RayPipelineState* ps,
    float* buf,
    int*   out_res);

/* Snapshot of live BDPT diagnostic stats (lock-free, updated by T3).
 * nearest_dist_m : √(min YZ d²) of any fwd/rev STRIKE pair seen so far, in metres.
 *                  -1 if no pairs have been observed yet.
 * best_collinearity : |cos θ| of best-collinear fwd/rev direction pair [0,1].
 * exact_snaps / near_miss_count : cumulative pixel accumulation counts. */
void ray_pipeline_get_bdpt_stats(
    const RayPipelineState* ps,
    float*    out_nearest_dist_m,
    float*    out_best_collinearity,
    uint64_t* out_exact_snaps,
    uint64_t* out_near_miss_count);

/* Snapshot of the four BDPT latch counters — all lock-free relaxed reads.
 * flash_dispatched: incremented once per trace_forward() call.
 * sensor_dispatched: incremented once per submit_sensor_sweep() call.
 * t5_fired: incremented at the end of each connection pass.
 * connection_running: 1 while a T5 thread is active, 0 otherwise. */
void ray_pipeline_get_bdpt_latch_state(
    const RayPipelineState* ps,
    uint32_t* out_flash_dispatched,
    uint32_t* out_sensor_dispatched,
    uint32_t* out_t5_fired,
    int*      out_connection_running);
