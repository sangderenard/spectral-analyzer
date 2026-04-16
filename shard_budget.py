"""
shard_budget.py — central authority for VRAM → RAM → HDD memory across CQT.

Every algorithm that reads or writes CQT shard files does so exclusively
through ShardBudget.  No raw memmap handle is ever handed to a caller.
Callers hold opaque ShardHandle tokens and negotiate I/O through budget
methods.

Double-buffer write chain
-------------------------
GPU computes tile_N  →  queue.put() into slot A   (returns immediately)
Background _WriteWorker drains slot A  →  mmap shard on disk
GPU computes tile_N+1  →  queue.put() into slot B  (while A is draining)
...

queue.Queue(maxsize=N) provides all thread safety and all backpressure:
  - put() blocks when all N slots are occupied  →  GPU waits on disk rate
  - No polling, no sleep loops, no RAM floor checks in the hot path

Reading shards
--------------
iter_frame_chunks() is the only sanctioned read path.  It yields
(real_chunk, imag_chunk, t0, t1) in column-chunks; each chunk is an
independent copy so the caller cannot accidentally retain a memmap view.
No full 2-D array is ever materialised.

Three streaming operations replace full-array allocations that existed
in bass_analysis.py:
  scan_abs_max()   — replaces np.abs(cqt_L_real).max() for float16 scaling
  convert_shards() — replaces _to_save(cqt_L_real) for dtype-mismatch saves
  iter_frame_chunks() — used by auto-trim and any future consumer
"""

from __future__ import annotations

import gc as _gc
import math
import os
import queue as _queue
import shutil
import threading
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    pass  # torch imported lazily to avoid mandatory dependency at import time


# ---------------------------------------------------------------------------
# Internal double-buffer write worker
# ---------------------------------------------------------------------------

class _WriteWorker:
    """Background thread that drains a bounded tile queue to memmap shards.

    queue_depth controls the pipeline depth:
      1 — single-buffer  (submit blocks while disk write completes)
      2 — double-buffer  (submit returns; previous write overlaps; default)
      N — wider pipeline (N tiles in-flight; only useful for bursty I/O)

    queue.Queue is Python's canonical thread-safe FIFO with optional size cap.
    submit() calls put(block=True), which blocks when maxsize slots are full —
    this is the sole backpressure mechanism.  No polling, no sleep.

    Exception handling: any exception in the worker thread is captured and
    re-raised on the next submit() or flush() call on the producer side.
    """

    def __init__(self, queue_depth: int = 2) -> None:
        self._q: _queue.Queue = _queue.Queue(maxsize=queue_depth)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="shard-writer"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Producer interface (called from GPU / compute thread)
    # ------------------------------------------------------------------

    def submit(
        self,
        mm_real: np.memmap,
        mm_imag: np.memmap,
        f0: int,
        f1: int,
        t0: int,
        t1: int,
        tile: np.ndarray,  # complex numpy, shape (f1-f0, t1-t0)
    ) -> None:
        """Queue a tile write.  Blocks when all queue slots are occupied.

        tile is already a CPU numpy array (produced by .cpu().numpy() in the
        GPU pipeline).  The caller's subsequent `del tile` just decrements the
        refcount; the queue holds the reference, keeping the buffer alive until
        the worker drains it.  No copy needed.
        """
        if self._error is not None:
            raise RuntimeError(
                f"shard writer thread failed: {self._error}"
            ) from self._error
        self._q.put((mm_real, mm_imag, f0, f1, t0, t1, tile))

    def flush(self) -> None:
        """Block until every queued write has been flushed to disk."""
        self._q.join()
        if self._error is not None:
            raise RuntimeError(
                f"shard writer thread failed: {self._error}"
            ) from self._error

    def stop(self) -> None:
        """Flush all pending writes and shut down the worker thread cleanly."""
        self.flush()
        self._q.put(None)   # None sentinel — worker exits its loop
        self._thread.join()

    # ------------------------------------------------------------------
    # Worker body (runs on dedicated thread)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        # These locals are pre-declared so the del at loop end is always safe,
        # even on the first iteration or after an early return.
        mm_real = mm_imag = tile = None
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return  # clean shutdown
                mm_real, mm_imag, f0, f1, t0, t1, tile = item
                item = None  # drop queue reference; locals carry what we need
                try:
                    # real and imag written as separate slices; no extra copy.
                    mm_real[f0:f1, t0:t1] = tile.real
                    mm_real.flush()
                    mm_imag[f0:f1, t0:t1] = tile.imag
                    mm_imag.flush()
                except Exception as exc:
                    self._error = exc   # surface on next producer call
                finally:
                    # Release mmap and tile refs BEFORE the next q.get() blocks.
                    # On Windows, np.memmap file handles are held open until the
                    # last Python reference is dropped; keeping mm_real/mm_imag
                    # alive in this frame across the blocking get() prevents
                    # handle.close() + gc.collect() from actually closing the
                    # underlying file, which causes PermissionError in
                    # convert_shards when it tries to rename the shard file.
                    mm_real = mm_imag = tile = None
            finally:
                self._q.task_done()


# ---------------------------------------------------------------------------
# Opaque shard token
# ---------------------------------------------------------------------------

class ShardHandle:
    """Opaque token representing one or more mmap shard files.

    Obtained exclusively from ShardBudget.open_shards() or load_shards().
    Callers cannot directly access the underlying memmaps; all I/O is
    mediated by ShardBudget methods.

    Attributes exposed for introspection (shape, dtype, channels, stream_dir)
    are read-only and contain no live data pointers.
    """

    __slots__ = ("_mmaps", "_stream_dir", "_mode")

    def __init__(
        self,
        mmaps: dict[str, np.memmap],
        stream_dir: str,
        mode: str,
    ) -> None:
        self._mmaps: dict[str, np.memmap] = mmaps
        self._stream_dir = stream_dir
        self._mode = mode  # "w+" or "r"

    @property
    def shape(self) -> tuple[int, int]:
        """(n_bins, n_frames) — shape of each shard component."""
        return next(iter(self._mmaps.values())).shape

    @property
    def dtype(self) -> np.dtype:
        return next(iter(self._mmaps.values())).dtype

    @property
    def stream_dir(self) -> str:
        return self._stream_dir

    @property
    def channels(self) -> list[str]:
        """Channel names present, e.g. ['left'] or ['left', 'right']."""
        return sorted({k.split("_", 1)[1] for k in self._mmaps})

    def close(self) -> None:
        """Release all mmap handles.

        Required on Windows before overwriting shard files (open w+ handles
        block os.replace / os.rename on the same path).  Call flush() on the
        owning ShardBudget first so no writes are still in-flight.
        """
        self._mmaps.clear()
        _gc.collect()


# ---------------------------------------------------------------------------
# Central authority
# ---------------------------------------------------------------------------

class ShardBudget:
    """Central authority for VRAM → RAM → HDD memory across the CQT pipeline.

    One instance per analysis run.  All shard creation, tile writes, and
    shard reads are mediated through this object.  No algorithm holds a raw
    memmap reference.

    Memory tiers
    ~~~~~~~~~~~~
    VRAM  GPU compute (filter bank × frames matmul).
          Budget: configurable fraction of free VRAM, queried lazily.
    RAM   CPU tile staging (in-flight write queue).
          Budget: bounded queue depth × tile size.
    HDD   Permanent mmap shards.
          Written by background _WriteWorker; read via iter_frame_chunks().

    Lifecycle
    ~~~~~~~~~
    budget = ShardBudget(device)
    handle = budget.open_shards(...)          # pre-allocate mmap files
    sink   = budget.make_sink(handle, ch)     # pass to compute_cqt
    ...
    budget.flush()                            # wait for all writes
    handle.close()                            # release w+ mmaps (Windows)
    handle = budget.load_shards(...)          # re-open read-only
    ...
    budget.close()                            # stop worker thread
    """

    def __init__(
        self,
        device=None,               # torch.device or None (lazy detection)
        *,
        vram_fraction: float = 0.6,
        write_queue_depth: int = 2,
        write_chunk_mb: int = 64,
    ) -> None:
        self._device = device
        self._vram_fraction = vram_fraction
        self._write_chunk_mb = write_chunk_mb
        self._worker = _WriteWorker(queue_depth=write_queue_depth)

    # ------------------------------------------------------------------
    # Shard lifecycle
    # ------------------------------------------------------------------

    def open_shards(
        self,
        stream_dir: str,
        n_bins: int,
        n_frames: int,
        dtype: np.dtype,
        channels: tuple[str, ...] = ("left",),
    ) -> ShardHandle:
        """Pre-allocate writable mmap shard files and return an opaque handle.

        Creates {part}_{channel}.npy files (real + imag per channel) in
        stream_dir.  The returned handle is the sole write path; use
        make_sink() to produce a callable that feeds the write worker.
        """
        os.makedirs(stream_dir, exist_ok=True)
        mmaps: dict[str, np.memmap] = {}
        for ch in channels:
            for part in ("real", "imag"):
                key = f"{part}_{ch}"
                path = os.path.join(stream_dir, f"{key}.npy")
                try:
                    mmaps[key] = np.lib.format.open_memmap(
                        path, mode="w+", dtype=dtype, shape=(n_bins, n_frames)
                    )
                except OSError as exc:
                    try:
                        usage = shutil.disk_usage(os.path.abspath(stream_dir))
                        needed_mb = n_bins * n_frames * dtype.itemsize / 1024 / 1024
                        print(
                            f"[DiskError] Cannot allocate shard {key}: {exc}\n"
                            f"  Needed: {needed_mb:.1f} MB  "
                            f"  Free: {usage.free / 1024 / 1024 / 1024:.2f} GB"
                        )
                    except Exception:
                        pass
                    raise
        return ShardHandle(mmaps, stream_dir, mode="w+")

    def load_shards(
        self,
        stream_dir: str,
        n_frames_act: int,
        channels: tuple[str, ...] = ("left",),
    ) -> ShardHandle:
        """Open existing shard files read-only and return an opaque handle.

        Each shard is sliced to [:, :n_frames_act] so callers don't need to
        carry the actual frame count separately.
        """
        mmaps: dict[str, np.memmap] = {}
        for ch in channels:
            for part in ("real", "imag"):
                key = f"{part}_{ch}"
                path = os.path.join(stream_dir, f"{key}.npy")
                mm = np.load(path, mmap_mode="r")
                mmaps[key] = mm[:, :n_frames_act]
        return ShardHandle(mmaps, stream_dir, mode="r")

    # ------------------------------------------------------------------
    # Write path — GPU tile → async disk write
    # ------------------------------------------------------------------

    def make_sink(
        self,
        handle: ShardHandle,
        channel: str = "left",
    ):
        """Return a sink callable  sink(f0, f1, t0, t1, tile_complex_np).

        Drop-in replacement for the _sink closure in compute_cqt.  Each
        call queues the tile for async write via the double-buffer worker
        and returns immediately — the GPU pipeline is not blocked by disk.

        The worker's bounded queue provides all backpressure: if both
        write slots are occupied, the next call to the sink blocks until
        one slot is freed.  No _backpressure_wait() needed.
        """
        mm_real = handle._mmaps[f"real_{channel}"]
        mm_imag = handle._mmaps[f"imag_{channel}"]
        worker = self._worker

        def _sink(f0: int, f1: int, t0: int, t1: int, tile: np.ndarray) -> None:
            worker.submit(mm_real, mm_imag, f0, f1, t0, t1, tile)

        return _sink

    def flush(self) -> None:
        """Block until all in-flight tile writes have been flushed to disk.

        Call this before reading from a handle that was written via make_sink(),
        and before calling handle.close() on Windows.
        """
        self._worker.flush()

    # ------------------------------------------------------------------
    # Read path — chunked, never materialises a full 2-D array
    # ------------------------------------------------------------------

    def iter_frame_chunks(
        self,
        handle: ShardHandle,
        channel: str,
        chunk_cols: int = 2048,
        out_dtype: np.dtype | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, int, int]]:
        """Yield (real_chunk, imag_chunk, t0, t1) in column-chunk slices.

        Each chunk is a fresh numpy copy of shape (n_bins, cols) where
        cols ≤ chunk_cols.  Copies are returned (not memmap views) so the
        caller can freely modify or discard each chunk; at most one chunk
        per channel is live in RAM at any point.

        out_dtype — promote chunks to this dtype on read; defaults to the
                    shard's own dtype.

        Usage::
            for real_c, imag_c, t0, t1 in budget.iter_frame_chunks(handle, "left"):
                pwr = real_c ** 2 + imag_c ** 2
                ...
                del real_c, imag_c, pwr   # keep peak RAM low
        """
        mm_real = handle._mmaps[f"real_{channel}"]
        mm_imag = handle._mmaps[f"imag_{channel}"]
        dt = out_dtype if out_dtype is not None else mm_real.dtype
        _, n_frames = mm_real.shape
        for t0 in range(0, n_frames, chunk_cols):
            t1 = min(t0 + chunk_cols, n_frames)
            yield (
                mm_real[:, t0:t1].astype(dt, copy=True),
                mm_imag[:, t0:t1].astype(dt, copy=True),
                t0,
                t1,
            )

    # ------------------------------------------------------------------
    # Streaming operations (replace full-array ops in bass_analysis.py)
    # ------------------------------------------------------------------

    def scan_abs_max(
        self,
        handle: ShardHandle,
        channels: tuple[str, ...] | None = None,
        chunk_cols: int = 2048,
    ) -> float:
        """Return the absolute maximum over all channels and real+imag.

        Streams in column-chunks; never materialises the full shard.
        Replaces  float(np.abs(cqt_L_real).max())  in the float16 scale
        computation path.
        """
        if channels is None:
            channels = tuple(handle.channels)
        global_max = 0.0
        for ch in channels:
            for real_c, imag_c, _t0, _t1 in self.iter_frame_chunks(
                handle, ch, chunk_cols=chunk_cols
            ):
                chunk_max = max(
                    float(np.abs(real_c).max()),
                    float(np.abs(imag_c).max()),
                )
                if chunk_max > global_max:
                    global_max = chunk_max
                del real_c, imag_c
        return global_max

    def convert_shards(
        self,
        handle: ShardHandle,
        out_dtype: np.dtype,
        scale: float | None = None,
        chunk_cols: int = 2048,
    ) -> None:
        """Convert real+imag shards to out_dtype in-place, streaming.

        If scale is given, values are divided by scale before conversion
        (used for float16 normalisation).  Each component is written to a
        sibling .convert.tmp file first then atomically renamed, so a crash
        mid-conversion leaves the originals intact.

        The source mmaps in handle are closed before renaming (Windows
        requirement) and replaced with read-only views of the converted
        files.  Callers must call budget.flush() before this to ensure
        all write-worker jobs have completed.

        Replaces  save_dict["real_left"] = _to_save(cqt_L_real)  — that
        pattern materialized the entire shard in RAM; this streams it.
        After conversion, the shards are in final save dtype and can be
        registered as pre_sharded without any additional write pass.
        """
        stream_dir = handle._stream_dir
        target_chunk_bytes = max(4, self._write_chunk_mb) * 1024 * 1024

        for ch in handle.channels:
            for part in ("real", "imag"):
                key = f"{part}_{ch}"
                src_path = os.path.join(stream_dir, f"{key}.npy")
                tmp_path = src_path + ".convert.tmp"

                n_bins, n_frames = handle._mmaps[key].shape
                row_bytes = n_frames * np.dtype(out_dtype).itemsize
                rows_per_chunk = max(1, target_chunk_bytes // max(row_bytes, 1))

                # Pre-allocate output; write into .tmp so src stays intact on error
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                out_mm = np.lib.format.open_memmap(
                    tmp_path, mode="w+", dtype=out_dtype, shape=(n_bins, n_frames)
                )

                src_mm = handle._mmaps[key]
                for r0 in range(0, n_bins, rows_per_chunk):
                    r1 = min(r0 + rows_per_chunk, n_bins)
                    chunk = src_mm[r0:r1, :]
                    if scale is not None:
                        chunk = chunk / scale
                    out_mm[r0:r1, :] = chunk.astype(out_dtype, copy=False)
                    out_mm.flush()
                    del chunk

                del out_mm

                # Release source mmap before rename (Windows cannot overwrite
                # a file that is still mapped by the same process).
                # Must also delete the local alias 'src_mm'; the dict removal
                # alone leaves a live reference that keeps the file locked.
                del src_mm
                del handle._mmaps[key]
                _gc.collect()

                try:
                    os.replace(tmp_path, src_path)
                except PermissionError:
                    _gc.collect()
                    try:
                        os.remove(src_path)
                    except FileNotFoundError:
                        pass
                    os.rename(tmp_path, src_path)

                # Re-open converted file as read-only view
                handle._mmaps[key] = np.load(src_path, mmap_mode="r")

    # ------------------------------------------------------------------
    # VRAM tile negotiation
    # ------------------------------------------------------------------

    def vram_budget(self) -> int:
        """Available VRAM in bytes for one CQT tile.

        Queries torch.cuda.mem_get_info lazily on first call; falls back to
        512 MB for CPU or when torch is unavailable.  Device is auto-detected
        if not supplied at construction.
        """
        device = self._device
        if device is None:
            try:
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._device = device  # cache for subsequent calls
            except ImportError:
                return 512 * 1024 * 1024
        try:
            if device.type == "cuda":
                import torch
                free, _ = torch.cuda.mem_get_info(device)
                return max(64 * 1024 * 1024, int(free * self._vram_fraction))
        except Exception:
            pass
        return 512 * 1024 * 1024

    def optimal_cqt_tile(
        self,
        n_filters: int,
        n_frames: int,
        n_fft: int,
        bytes_complex: int,
    ) -> tuple[int, int]:
        """Return (K_filters, T_frames) tile that fits the current VRAM budget.

        Same AM-GM formula as torch_cqt_new._optimal_tile, but sourced from
        this authority so tile sizing stays consistent with the write pipeline.

        Working-set model:
          W = (K + T) × F × bc + K × T × bc
          where F = n_fft//2 + 1, bc = bytes_complex

        AM-GM optimum (K = T) solves:  K² + 2KF = budget/bc
        """
        budget = self.vram_budget()
        F = n_fft // 2 + 1
        bc = bytes_complex
        discriminant = F * F + budget / bc
        K = max(1, int(-F + math.sqrt(discriminant)))
        K = min(K, n_filters)
        denom = F + K
        T = max(1, int((budget / bc - K * F) // max(denom, 1)))
        T = min(T, n_frames)
        return K, T

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush all pending writes and shut down the background writer."""
        self._worker.stop()
