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
    std::atomic<int>      batch_sz{16};        /* adaptive CPU pop batch size    */
    std::atomic<int>      q_depth{0};          /* last observed upstream q len   */
    /* GPU */
    std::atomic<uint64_t> n_gpu{0};            /* items processed by GPU dispatch */
    std::atomic<uint64_t> ns_gpu{0};           /* total ns spent on GPU dispatch  */
    std::atomic<int>      batch_sz_gpu{4096};  /* adaptive GPU pop batch size (start large) */
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
        /* Adapt GPU batch size: ramp up aggressively, back off slowly.
         * Upper bound 65536 lets a GPU absorb a full frame's worth in one shot. */
        int cur = batch_sz_gpu.load(std::memory_order_relaxed);
        if      (q_now > cur     && cur < 65536) batch_sz_gpu.store(cur * 2, std::memory_order_relaxed);
        else if (q_now < 16      && cur > 256)   batch_sz_gpu.store(cur / 2, std::memory_order_relaxed);
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

/* Snapshot of all four stages + output queue, returned to callers. */
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
    } t1, t2, t3, t4;
    int output_queue_depth;
    int in_flight;
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
};

/* ─── Thread-safe pipeline queue ────────────────────────────────────────── */

template<typename T>
class PipelineQueue {
public:
    void push(T item) {
        { std::lock_guard<std::mutex> lk(mu_); q_.push_back(std::move(item)); }
        cv_.notify_one();
    }

    /* Bounded push: blocks (spin-sleep 500 µs) until size < max_size.
     * Provides backpressure on the submit thread without busy-spinning the CPU.
     * max_size <= 0 falls through to unbounded push. */
    void push_bounded(T item, int max_size) {
        if (max_size > 0) {
            for (;;) {
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if ((int)q_.size() < max_size) {
                        q_.push_back(std::move(item));
                        cv_.notify_one();
                        return;
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        }
        push(std::move(item));
    }

    /* Blocking pop.  Brief spin before sleeping — reduces CV overhead on hot queues. */
    bool pop(T& out) {
        for (int s = 0; s < 512; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!q_.empty()) { out = std::move(q_.front()); q_.pop_front(); return true; }
                if (done_) return false;
            }
            std::this_thread::yield();
        }
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return !q_.empty() || done_; });
        if (q_.empty()) return false;
        out = std::move(q_.front()); q_.pop_front(); return true;
    }

    /* Blocking batch pop — waits for ≥1 item, takes up to max_n.
     * Appends to out.  Returns count taken; 0 means done+empty. */
    int pop_batch(std::vector<T>& out, int max_n) {
        for (int s = 0; s < 512; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!q_.empty()) {
                    int n = std::min(max_n, (int)q_.size());
                    for (int i = 0; i < n; ++i) { out.push_back(std::move(q_.front())); q_.pop_front(); }
                    return n;
                }
                if (done_) return 0;
            }
            std::this_thread::yield();
        }
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return !q_.empty() || done_; });
        if (q_.empty()) return 0;
        int n = std::min(max_n, (int)q_.size());
        for (int i = 0; i < n; ++i) { out.push_back(std::move(q_.front())); q_.pop_front(); }
        return n;
    }

    /* Non-blocking drain — takes up to max_n items immediately.
     * Appends to out.  Returns count taken (0 if empty). */
    int drain(std::vector<T>& out, int max_n) {
        std::lock_guard<std::mutex> lk(mu_);
        int n = std::min(max_n, (int)q_.size());
        for (int i = 0; i < n; ++i) { out.push_back(std::move(q_.front())); q_.pop_front(); }
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
        for (int s = 0; s < 512; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!q_.empty()) goto take;
                if (done_) return 0;
            }
            std::this_thread::yield();
        }
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]{ return !q_.empty() || done_; });
            if (q_.empty()) return 0;
        }
    take:
        {
            std::lock_guard<std::mutex> lk(mu_);
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
        for (int s = 0; s < 512; ++s) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!q_.empty()) goto priority_take;
                if (done_) return 0;
            }
            std::this_thread::yield();
        }
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]{ return !q_.empty() || done_; });
            if (q_.empty()) return 0;
        }
    priority_take:
        {
            std::vector<T> grabbed;
            {
                std::lock_guard<std::mutex> lk(mu_);
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

    void set_done() {
        { std::lock_guard<std::mutex> lk(mu_); done_ = true; }
        cv_.notify_all();
    }

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<T>           q_;
    bool                    done_ = false;
};

/* ─── Wave solver arena ─────────────────────────────────────────────────── */

struct WaveArena {
    int              id      = -1;
    Eigen::Vector3d  center;
    double           radius  = 0.0;
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
     * plate_x: sensor X position; plate_r: disc radius; pip_res: pixel grid
     * side (res×res); bdpt_eps: Y/Z proximity for BDPT snap in metres. */
    float  sensor_plate_x   = 0.0f;
    float  sensor_plate_r   = 0.0f;
    int    sensor_pip_res   = 0;      /* 0 = disabled */
    float  sensor_bdpt_eps  = 0.008f;

    /* ── GPU compute dispatch ────────────────────────────────────────────
     * When use_gpu_compute=true a GlPipelineDispatch thread is spawned that
     * competes with the CPU workers on the shared intent/hit/refined queues.
     * Both CPU and GPU workers remain active; adaptive batch sizing and the
     * gpu_fraction stat in StageStats track which operator handles more work.
     *
     * shader_dir: directory containing the four .comp.glsl shader files.
     *   Empty string → searches "csrc/shaders/" relative to cwd.
     *
     * gpu_batch_size_t{1,2,3,4}: initial GPU pop batch size per stage.
     *   0 = use the StageStats default (256). */
    bool        use_gpu_compute    = false;
    std::string shader_dir;
    int         gpu_batch_size_t1  = 0;
    int         gpu_batch_size_t2  = 0;
    int         gpu_batch_size_t3  = 0;
    int         gpu_batch_size_t4  = 0;

    /* Fraction of work to pin to GPU per stage (0=compete freely, >0=soft target).
     * 0.0 = CPU and GPU compete naturally on the shared queue.
     * 1.0 = route all new intents/hits to the GPU sub-queues; CPU workers idle. */
    float       gpu_fraction_t1    = 0.0f;
    float       gpu_fraction_t2    = 0.0f;
    float       gpu_fraction_t3    = 0.0f;
};

/* ─── Opaque pipeline state (defined in ray_tracer.cpp) ─────────────────── */

struct RayPipelineState;

/* Thin accessors so callers don't need the full RayPipelineState definition. */

/* Number of spectral bands in the associated tracer (0 if none). */
int ray_pipeline_n_bands(const RayPipelineState* ps);

/* mat_idx of triangle tri_idx (-1 if ps/st is null or index out of range). */
int ray_pipeline_tri_mat_idx(const RayPipelineState* ps, int tri_idx);

/* Non-blocking drain from Q_refined: pops up to max_n into out.
 * Returns count appended (0 if queue empty). */
int ray_pipeline_drain_refined(RayPipelineState* ps,
                                std::vector<RefinedHit>& out,
                                int max_n);

RayPipelineState* ray_pipeline_create(
    RayTracerState*          st,
    const RayPipelineConfig* cfg);

void ray_pipeline_destroy(RayPipelineState* ps);

/* Non-blocking submit: push intents into the T1 input queue. */
void ray_pipeline_submit(
    RayPipelineState*  ps,
    const RayIntent*   intents,
    int                n_intents);

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
 * plate_x: world X of sensor plane; plate_r: disc radius; res: pixel grid side
 * (res×res RGB); bdpt_eps: world-space YZ proximity for BDPT snap.
 * Resets the accumulator buffer. */
void ray_pipeline_configure_sensor_image(
    RayPipelineState* ps,
    float plate_x, float plate_r,
    int   res,
    float bdpt_eps);

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
