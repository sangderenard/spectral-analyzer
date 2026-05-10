/**
 * rt_streaming.h — Streaming memory management for spectral ray tracer output.
 *
 * Provides:
 *   OverflowPolicy   — per-sink behaviour when capacity is exceeded.
 *   MemoryBudget     — runtime allocation limits for field grids and buffers.
 *   checked_mul_size — safe multiplication before allocation (size_t variant).
 *   RtRecordSink     — abstract base for segment-record consumers.
 *     MemoryRecordSink  — wraps a caller-provided flat buffer (current behaviour).
 *     NullCountingSink  — counts records; stores nothing (benchmark / dry-run).
 *     ChunkedMemorySink — fixed-size chunks; avoids one giant contiguous alloc.
 *     FileRecordSink    — buffered binary append to a file path.
 *   ChunkedAppendBuffer<T> — generic fixed-chunk append buffer for POD types.
 *
 * Thread-safety notes
 * -------------------
 * All sinks in this header are SINGLE-THREADED.  For multi-producer scenarios
 * serialise push_segments() calls externally, or give each worker thread its
 * own sink (ChunkedMemorySink) and merge after tracing completes.
 * FileRecordSink is safe to use from one dedicated writer thread.
 *
 * Design intent
 * -------------
 * The ray tracer can generate more data than RAM when sources × rays × bounces
 * × bands is large.  Callers must not be required to pre-allocate a single
 * contiguous vector and hope it fits.  These sinks decouple the producer
 * (tracer) from the consumer (Python, disk, network) and make backpressure
 * explicit rather than a silent buffer overflow.
 */
#pragma once
#include "serial_kernel.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <fstream>

/* ── OverflowPolicy ──────────────────────────────────────────────────────── */

/**
 * Controls what a sink does when it cannot accept more records.
 *
 * DROP_PREVIEW    : silently discard — acceptable for visualisation paths.
 * STOP_WITH_ERROR : push_segments() returns false; producer should stop.
 * STREAM_TO_DISK  : internal tag for FileRecordSink; not used by other sinks.
 * COUNT_ONLY      : NullCountingSink — records are counted but not stored.
 */
enum class OverflowPolicy : int {
    DROP_PREVIEW    = 0,
    STOP_WITH_ERROR = 1,
    STREAM_TO_DISK  = 2,
    COUNT_ONLY      = 3
};

/* ── MemoryBudget ────────────────────────────────────────────────────────── */

/**
 * Runtime allocation limits.  Zero in any field means uncapped.
 *
 * Pass a pointer to this struct to budget-aware allocation helpers such as
 * field_grid_create_budgeted().  The tracer does not own or free this struct.
 *
 * Typical safe defaults for a workstation with 32 GB RAM:
 *   max_bytes_field_grid    = 4 GB  (single 128³ grid, 32 bands ≈ 1 GB)
 *   max_bytes_output_buffer = 2 GB
 *   max_bytes_temp_workspace = 512 MB
 *   max_records_per_chunk    = 1 << 20  (~1 M records × 48 B ≈ 48 MB)
 */
struct MemoryBudget {
    uint64_t max_bytes_total          = 0;   /* across all subsystems        */
    uint64_t max_bytes_field_grid     = 0;   /* per FieldGrid allocation     */
    uint64_t max_bytes_output_buffer  = 0;   /* segment / camera output      */
    uint64_t max_bytes_temp_workspace = 0;   /* per-step scratch space       */
    uint64_t max_records_per_chunk    = 0;   /* ChunkedMemorySink chunk size */
    int      allow_out_of_core        = 0;   /* 1 = permit mmap/tiled path   */
};

/* ── checked_mul_size ────────────────────────────────────────────────────── */

/**
 * Multiply two size_t values, checking for overflow and optional budget cap.
 *
 * Returns false if a × b overflows size_t OR the product exceeds max_bytes
 * (when max_bytes > 0).  On success writes a × b into out.
 *
 * Use this for every allocation derived from user-supplied dimensions:
 *   nx * ny, n_cells * n_bands, n_records * stride, …
 */
static inline bool checked_mul_size(size_t a, size_t b, size_t& out,
                                    size_t max_bytes = 0)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        return false;
    out = a * b;
    return (max_bytes == 0) || (out <= max_bytes);
}

/* ── RtRecordSink ────────────────────────────────────────────────────────── */

/**
 * Abstract base for segment-record consumers.
 *
 * push_segments() is called with a batch of records; each record is
 * RT_FLOATS_PER_SEG floats (48 bytes) unless the sink is told otherwise.
 * Returns true to request more records; false to stop the producer.
 */
struct RtRecordSink {
    virtual ~RtRecordSink() = default;

    /**
     * Deliver n_records records (contiguous in memory) to the sink.
     * records points to n_records × floats_per_rec float values.
     * Returns true to continue; false to abort production.
     */
    virtual bool push_segments(const float* records, int n_records) = 0;

    /** Flush any internally buffered writes.  Called by producer on finish. */
    virtual bool flush() { return true; }

    /** True if the sink has encountered an unrecoverable error. */
    virtual bool failed() const { return false; }

    /** Total records successfully accepted by this sink so far. */
    virtual int64_t records_accepted() const = 0;

    /** Total records rejected or discarded by this sink. */
    virtual int64_t records_dropped() const { return 0; }
};

/* ── MemoryRecordSink ────────────────────────────────────────────────────── */

/**
 * Wraps a caller-provided fixed-capacity flat float buffer.
 *
 * This is a drop-in replacement for the legacy `float* out_segs, int out_cap`
 * pattern.  On overflow the policy field controls whether records are silently
 * dropped (DROP_PREVIEW) or the sink signals the producer to stop
 * (STOP_WITH_ERROR).
 *
 * floats_per_rec should be RT_FLOATS_PER_SEG (12) for standard segments or
 * RT_FLOATS_PER_SEG_MS (14) for multiscale segments.
 */
struct MemoryRecordSink final : RtRecordSink {
    float*         buf          = nullptr;
    int            cap_records  = 0;      /* capacity in whole records       */
    int            written      = 0;      /* records written so far          */
    int64_t        dropped_     = 0;
    OverflowPolicy policy       = OverflowPolicy::DROP_PREVIEW;
    bool           error_set    = false;
    int            floats_per_rec;

    MemoryRecordSink(float* out, int capacity_records,
                     int floats_per_record,
                     OverflowPolicy op = OverflowPolicy::DROP_PREVIEW)
        : buf(out), cap_records(capacity_records),
          floats_per_rec(floats_per_record), policy(op)
    {}

    bool push_segments(const float* records, int n_records) override {
        int available = cap_records - written;
        int n_fit     = (n_records <= available) ? n_records : available;
        if (n_fit > 0) {
            std::memcpy(buf + static_cast<size_t>(written) * floats_per_rec,
                        records,
                        static_cast<size_t>(n_fit) * floats_per_rec * sizeof(float));
            written += n_fit;
        }
        int n_drop = n_records - n_fit;
        if (n_drop > 0) {
            dropped_ += n_drop;
            if (policy == OverflowPolicy::STOP_WITH_ERROR) {
                error_set = true;
                return false;
            }
        }
        return true;
    }

    bool    failed()          const override { return error_set; }
    int64_t records_accepted() const override { return written; }
    int64_t records_dropped()  const override { return dropped_; }
};

/* ── NullCountingSink ────────────────────────────────────────────────────── */

/**
 * Accepts all records but writes nothing.
 *
 * Use cases:
 *   - dry-run capacity estimation before a real trace,
 *   - benchmarking the tracer without output bandwidth cost,
 *   - validating that total record counts match expectations.
 */
struct NullCountingSink final : RtRecordSink {
    int64_t accepted = 0;

    bool push_segments(const float*, int n_records) override {
        accepted += n_records;
        return true;
    }

    int64_t records_accepted() const override { return accepted; }
};

/* ── ChunkedMemorySink ───────────────────────────────────────────────────── */

/**
 * Stores records in a sequence of fixed-size heap chunks.
 *
 * Avoids one giant contiguous allocation.  Suitable for scenes where the
 * total record count is not known in advance and may exceed what a single
 * std::vector can hold contiguously.
 *
 * records_per_chunk: default 65536 (65536 × 48 B ≈ 3 MB per chunk).
 *
 * After tracing, iterate over chunks[] and copy/stream each to its
 * destination.  chunk_float_count(i) gives the number of valid floats in
 * chunk i.
 */
struct ChunkedMemorySink final : RtRecordSink {
    size_t                          records_per_chunk;
    int                             floats_per_rec;
    std::vector<std::vector<float>> chunks;
    int64_t                         accepted_count = 0;

    ChunkedMemorySink(size_t recs_per_chunk = 65536, int floats_per_record = 12)
        : records_per_chunk(recs_per_chunk), floats_per_rec(floats_per_record)
    {
        _new_chunk();
    }

    bool push_segments(const float* records, int n_records) override {
        const float* src = records;
        int remain = n_records;
        while (remain > 0) {
            auto& cur       = chunks.back();
            size_t cur_recs = cur.size() / static_cast<size_t>(floats_per_rec);
            size_t room     = (cur_recs < records_per_chunk)
                                ? (records_per_chunk - cur_recs)
                                : 0;
            if (room == 0) {
                _new_chunk();
                room = records_per_chunk;
            }
            size_t take = (static_cast<size_t>(remain) < room)
                            ? static_cast<size_t>(remain)
                            : room;
            size_t n_f = take * static_cast<size_t>(floats_per_rec);
            cur.insert(cur.end(), src, src + n_f);
            src            += n_f;
            remain         -= static_cast<int>(take);
            accepted_count += static_cast<int64_t>(take);
        }
        return true;
    }

    /** Number of valid float elements in chunk index i. */
    size_t chunk_float_count(size_t i) const {
        return (i < chunks.size()) ? chunks[i].size() : 0;
    }

    /** Total bytes stored across all chunks. */
    size_t total_bytes() const {
        size_t n = 0;
        for (const auto& ch : chunks) n += ch.size() * sizeof(float);
        return n;
    }

    int64_t records_accepted() const override { return accepted_count; }

private:
    void _new_chunk() {
        chunks.emplace_back();
        chunks.back().reserve(records_per_chunk * static_cast<size_t>(floats_per_rec));
    }
};

/* ── FileRecordSink ──────────────────────────────────────────────────────── */

/**
 * Buffered binary append to a file.
 *
 * Records are accumulated in a RAM write buffer and flushed to disk when the
 * buffer reaches write_buffer_records entries, or when flush() is called
 * explicitly.  The file is opened in binary-append mode so multiple calls to
 * the same path are non-destructive; the caller should open a fresh file name
 * per trace.
 *
 * A metadata sidecar (path + ".meta") is NOT written automatically — the
 * caller should record dimensions (n_bands, floats_per_rec, total_records)
 * after tracing completes.
 *
 * Thread safety: not thread-safe; use from one writer thread or serialise.
 */
struct FileRecordSink final : RtRecordSink {
    std::ofstream      file;
    int                floats_per_rec;
    std::vector<float> wbuf;
    size_t             wbuf_rec_cap;
    int64_t            accepted_count = 0;
    bool               error          = false;

    FileRecordSink(const char* path, int floats_per_record,
                   size_t write_buffer_records = 65536)
        : floats_per_rec(floats_per_record),
          wbuf_rec_cap(write_buffer_records)
    {
        file.open(path, std::ios::binary | std::ios::app);
        if (!file.is_open()) error = true;
        if (!error)
            wbuf.reserve(write_buffer_records * static_cast<size_t>(floats_per_rec));
    }

    bool push_segments(const float* records, int n_records) override {
        if (error) return false;
        size_t n_f = static_cast<size_t>(n_records) * floats_per_rec;
        wbuf.insert(wbuf.end(), records, records + n_f);
        accepted_count += n_records;
        if (wbuf.size() / static_cast<size_t>(floats_per_rec) >= wbuf_rec_cap)
            return flush();
        return true;
    }

    bool flush() override {
        if (error) return false;
        if (wbuf.empty()) return true;
        file.write(reinterpret_cast<const char*>(wbuf.data()),
                   static_cast<std::streamsize>(wbuf.size() * sizeof(float)));
        if (!file) { error = true; return false; }
        file.flush();
        wbuf.clear();
        return true;
    }

    bool    failed()          const override { return error; }
    int64_t records_accepted() const override { return accepted_count; }
};

/* ── ChunkedAppendBuffer<T> ──────────────────────────────────────────────── */

/**
 * Fixed-chunk append buffer for homogeneous POD items.
 *
 * Replaces unbounded std::vector<T> for buffers that can grow without bound
 * (e.g. camera_strike_rows, endpoint lists).  Benefits:
 *   - No single huge contiguous allocation.
 *   - Partial flushing: drain early chunks without copying.
 *   - Bounded growth: caller can check total_items() against a budget.
 *   - reserve_n() returns a raw pointer into one chunk; caller fills in-place.
 *
 * Items are appended in FIFO order within and across chunks.
 * NOT thread-safe; serialise push_back / reserve_n for concurrent producers.
 *
 * @tparam T  Any trivially-copyable type.  Use float for float rows.
 */
template<class T>
class ChunkedAppendBuffer {
public:
    /** items_per_chunk: tune to keep each chunk ≤ a few MB.
     *  For float: 65536 items × 4 B = 256 KB per chunk.
     *  For camera strike rows (stride ≈ 40 floats): 4096 rows × 160 B = 640 KB. */
    explicit ChunkedAppendBuffer(size_t items_per_chunk = 65536)
        : _chunk_size(items_per_chunk)
    {
        _new_chunk();
    }

    /** Append one item.  Never invalidates existing pointers. */
    void push_back(const T& val) {
        if (_chunks.back().size() == _chunk_size)
            _new_chunk();
        _chunks.back().push_back(val);
        ++_total;
    }

    /**
     * Reserve n contiguous items in a single chunk and return a pointer to the
     * first.  Returns nullptr if n > chunk_size (request cannot fit).
     * The caller must write all n items before the next call that may append.
     */
    T* reserve_n(size_t n) {
        if (n == 0 || n > _chunk_size) return nullptr;
        if (_chunks.back().size() + n > _chunk_size)
            _new_chunk();
        auto& cur  = _chunks.back();
        size_t base = cur.size();
        cur.resize(base + n);
        _total += static_cast<int64_t>(n);
        return cur.data() + base;
    }

    size_t total_items() const { return static_cast<size_t>(_total); }

    size_t n_chunks() const { return _chunks.size(); }

    /** Pointer and item count for chunk i (read-only iteration). */
    const T* chunk_data(size_t i)  const { return _chunks[i].data(); }
    size_t   chunk_items(size_t i) const { return _chunks[i].size(); }

    /** Flatten to a single contiguous vector (may be large). */
    std::vector<T> to_vector() const {
        std::vector<T> out;
        out.reserve(static_cast<size_t>(_total));
        for (const auto& ch : _chunks)
            out.insert(out.end(), ch.begin(), ch.end());
        return out;
    }

    /** Clear all data but keep the first (now-empty) chunk to avoid realloc. */
    void clear() {
        _chunks.clear();
        _total = 0;
        _new_chunk();
    }

private:
    size_t                      _chunk_size;
    std::vector<std::vector<T>> _chunks;
    int64_t                     _total = 0;

    void _new_chunk() {
        _chunks.emplace_back();
        _chunks.back().reserve(_chunk_size);
    }
};
