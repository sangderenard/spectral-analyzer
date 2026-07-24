/* nodus_optical_fifo.h -- native RAII wrapper over the extracted Nodus
 * EdgeTensorFifo C ABI (nodus_runtime_abi.h), configured for the fixed
 * 96-byte OpticalBranchToken. Real bytes through the real
 * C:\dev\Powershell\nodus\build\Release\nodus_runtime.dll, not a Python
 * ctypes bridge and not a second queue implementation.
 */
#pragma once

#include <nodus_runtime_abi.h>

#include <cstdint>
#include <stdexcept>

#include "optical_branch_abi.h"

namespace nodus_optical {

/* One lossless typed edge carrying OpticalBranchToken samples. */
class OpticalBranchFifo {
 public:
    OpticalBranchFifo(size_t slots, int32_t type_id) {
        runtime_ = nodus_edge_runtime_create();
        if (!runtime_)
            throw std::runtime_error("nodus_edge_runtime_create failed");
        const int32_t dims[1] = {1};
        const int32_t configured = nodus_edge_runtime_configure(
            runtime_, dims, 1, slots, /*top_k=*/0,
            sizeof(optical_branch::OpticalBranchToken), type_id,
            /*layout=*/2 /* opaque */, /*dtype=*/8 /* bytes */);
        if (!configured) {
            nodus_edge_runtime_destroy(runtime_);
            runtime_ = nullptr;
            throw std::runtime_error(
                "nodus_edge_runtime_configure rejected the optical branch token spec");
        }
    }

    ~OpticalBranchFifo() {
        if (runtime_) nodus_edge_runtime_destroy(runtime_);
    }

    OpticalBranchFifo(const OpticalBranchFifo&) = delete;
    OpticalBranchFifo& operator=(const OpticalBranchFifo&) = delete;

    void subscribe(std::uint64_t reader_id, bool start_at_head = true) {
        if (!nodus_edge_runtime_subscribe(runtime_, reader_id, start_at_head ? 1 : 0))
            throw std::runtime_error("nodus_edge_runtime_subscribe rejected the reader");
    }

    /* Returns false only if the runtime itself rejected the write (e.g. not
     * configured). A lossless edge with an unread slowest reader still
     * accepts the publish and applies backpressure/blocking internally per
     * NODUS_OPTICAL_KPN_INTEGRATION.md; out_dropped reports lossy drops for
     * edges explicitly configured to allow them (this one is not). */
    bool publish(std::uint64_t writer_id, const optical_branch::OpticalBranchToken& token) {
        int32_t dropped = 0;
        const int32_t accepted = nodus_edge_runtime_publish(
            runtime_, writer_id, &token, sizeof(token), &dropped);
        if (accepted && dropped)
            throw std::runtime_error("lossless optical branch FIFO dropped a token");
        return accepted != 0;
    }

    /* Returns true and fills `out` if a token was available for this
     * reader, false if the edge is caught up to the write frontier. */
    bool consume(std::uint64_t reader_id, optical_branch::OpticalBranchToken& out) {
        size_t written = 0;
        const int32_t accepted = nodus_edge_runtime_consume(
            runtime_, reader_id, &out, sizeof(out), &written);
        if (!accepted) return false;
        if (written != sizeof(out))
            throw std::runtime_error("nodus_edge_runtime_consume returned a partial token");
        return true;
    }

    bool quiescent() const {
        return nodus_edge_runtime_is_quiescent(runtime_) != 0;
    }

 private:
    NodusEdgeRuntime* runtime_ = nullptr;
};

}  // namespace nodus_optical
