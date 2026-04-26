"""edge_fifo_bank.py — Preallocated FIFO channel bank for graph-solver edges.

One bank manages a named pool of circular-buffer slots.  Each slot is a
single-writer / single-reader channel with KPN backpressure semantics:

    write()  — raises FifoFull  when the slot has no room.
    read()   — raises FifoEmpty when the slot has nothing to deliver.
    try_write / try_read — non-raising variants that return False / None.

All storage is allocated once at claim() time.  No tensor allocation occurs
inside the solve loop.

Stride and capacity
-------------------
``stride``    — samples per frame (the unit of one write or read operation).
                Use 1 for per-sample solver edges, block_size for block edges.
``fifo_size`` — frames the buffer can hold before it is full.

    max_capacity = fifo_size * stride   (total complex128 samples per slot)

GraphSolver integration
-----------------------
Pass an ``EdgeFifoBank`` instance to ``GraphSolver(fifo_bank=bank)``.  The
solver propagates the bank to all node modules via ``_resolve_fifo_bank``.
Reset is forwarded automatically via ``GraphSolver.reset()``.

Object slots
------------
``claim_object(key)`` allocates a Python-object FIFO alongside the tensor
slots.  Object slots are addressed by the same key namespace.  Use
``write_batch(key, list)`` to store an entire list as one atomic item and
``try_read(key)`` to pop it.  These are used by ScoreSequencerNode to pass
``list[PerformanceAtom]`` to voice nodes without tensor overhead.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

_CDTYPE = torch.complex128


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class FifoEmpty(Exception):
    """Raised when read() is called on an empty slot."""


class FifoFull(Exception):
    """Raised when write() is called on a full slot."""


# ─────────────────────────────────────────────────────────────────────────────
# _FifoSlot — one preallocated circular buffer
# ─────────────────────────────────────────────────────────────────────────────

class _FifoSlot:
    """Preallocated circular buffer: fifo_size frames × stride samples each.

    All storage lives in ``_buf`` (shape ``(fifo_size, stride)``), allocated
    once at construction.  Heads and count are plain Python ints — no tensor
    overhead on the fast path.

    For stride=1 the public API squeezes frames to scalar tensors ``()`` so
    callers that expect scalar node outputs see the right shape.
    """

    __slots__ = ("_buf", "_fifo_size", "_stride", "_write_pos", "_read_pos", "_count")

    def __init__(
        self,
        fifo_size: int,
        stride: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._buf: Tensor = torch.zeros(fifo_size, stride, dtype=dtype, device=device)
        self._fifo_size: int = fifo_size
        self._stride: int = stride
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._count: int = 0

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return self._fifo_size

    @property
    def count(self) -> int:
        return self._count

    def is_empty(self) -> bool:
        return self._count == 0

    def is_full(self) -> bool:
        return self._count >= self._fifo_size

    # ── write ─────────────────────────────────────────────────────────────────

    def write(self, frame: Tensor) -> None:
        """Write one frame.  Raises FifoFull if no space."""
        if self._count >= self._fifo_size:
            raise FifoFull(
                f"FIFO full (capacity={self._fifo_size}, stride={self._stride})"
            )
        self._buf[self._write_pos].copy_(
            frame.to(self._buf.dtype).to(self._buf.device).reshape(self._stride)
        )
        self._write_pos = (self._write_pos + 1) % self._fifo_size
        self._count += 1

    def try_write(self, frame: Tensor) -> bool:
        """Write one frame.  Returns False (no-op) if the slot is full."""
        if self._count >= self._fifo_size:
            return False
        self.write(frame)
        return True

    # ── read ──────────────────────────────────────────────────────────────────

    def read(self) -> Tensor:
        """Consume and return the oldest frame.  Raises FifoEmpty if empty."""
        if self._count == 0:
            raise FifoEmpty(f"FIFO empty (capacity={self._fifo_size}, stride={self._stride})")
        frame = self._buf[self._read_pos].clone()
        self._read_pos = (self._read_pos + 1) % self._fifo_size
        self._count -= 1
        return frame.squeeze(0) if self._stride == 1 else frame

    def try_read(self) -> Optional[Tensor]:
        """Consume and return the oldest frame, or None if empty."""
        if self._count == 0:
            return None
        return self.read()

    def peek(self) -> Tensor:
        """Return the oldest frame without consuming it.  Raises FifoEmpty if empty."""
        if self._count == 0:
            raise FifoEmpty(f"FIFO empty (capacity={self._fifo_size}, stride={self._stride})")
        frame = self._buf[self._read_pos]
        return frame.squeeze(0) if self._stride == 1 else frame

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset heads and count without reallocating storage."""
        self._write_pos = 0
        self._read_pos = 0
        self._count = 0


# ─────────────────────────────────────────────────────────────────────────────
# ObjectFifoSlot — one Python-object circular-buffer slot
# ─────────────────────────────────────────────────────────────────────────────

class ObjectFifoSlot:
    """Single-writer/single-reader Python object FIFO with KPN backpressure.

    Storage is a fixed-size list allocated at claim time.  Writes only replace
    existing cells in the ring buffer; they do not grow a deque or allocate
    channel storage inside graph execution.
    """

    __slots__ = ("_buf", "_fifo_size", "_write_pos", "_read_pos", "_count")

    def __init__(self, fifo_size: int) -> None:
        self._fifo_size: int = int(fifo_size)
        self._buf: list[Any | None] = [None] * self._fifo_size
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._count: int = 0

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return self._fifo_size

    @property
    def count(self) -> int:
        return self._count

    def is_empty(self) -> bool:
        return self._count == 0

    def is_full(self) -> bool:
        return self._count >= self._fifo_size

    # ── write ─────────────────────────────────────────────────────────────────

    def write(self, obj: Any) -> None:
        """Write one object.  Raises FifoFull if no space."""
        if self._count >= self._fifo_size:
            raise FifoFull(f"ObjectFifo full (capacity={self._fifo_size})")
        self._buf[self._write_pos] = obj
        self._write_pos = (self._write_pos + 1) % self._fifo_size
        self._count += 1

    def try_write(self, obj: Any) -> bool:
        """Write one object.  Returns False (no-op) if the slot is full."""
        if self._count >= self._fifo_size:
            return False
        self.write(obj)
        return True

    def write_batch(self, items: list) -> None:
        """Write an entire list as ONE atomic item.  Raises FifoFull if no space."""
        self.write(items)

    # ── read ──────────────────────────────────────────────────────────────────

    def read(self) -> Any:
        """Consume and return the oldest item.  Raises FifoEmpty if empty."""
        if self._count == 0:
            raise FifoEmpty(f"ObjectFifo empty (capacity={self._fifo_size})")
        obj = self._buf[self._read_pos]
        self._buf[self._read_pos] = None
        self._read_pos = (self._read_pos + 1) % self._fifo_size
        self._count -= 1
        return obj

    def try_read(self) -> Optional[Any]:
        """Consume and return the oldest item, or None if empty."""
        if self._count == 0:
            return None
        return self.read()

    def read_batch(self) -> list:
        """Read one slot item and return as list (unwraps write_batch envelope)."""
        item = self.read()
        return item if isinstance(item, list) else [item]

    def peek(self) -> Any:
        """Return the oldest item without consuming it.  Raises FifoEmpty if empty."""
        if self._count == 0:
            raise FifoEmpty(f"ObjectFifo empty (capacity={self._fifo_size})")
        return self._buf[self._read_pos]

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset heads and count without reallocating storage."""
        for i in range(self._fifo_size):
            self._buf[i] = None
        self._write_pos = 0
        self._read_pos = 0
        self._count = 0


# ─────────────────────────────────────────────────────────────────────────────
# EdgeFifoBank — named pool of preallocated FIFO slots
# ─────────────────────────────────────────────────────────────────────────────

class EdgeFifoBank:
    """Pool of preallocated single-writer / single-reader FIFO channels.

    Supports two slot types:

    * **Tensor slots** (default) — preallocated circular buffers for complex128
      signal frames.  Claimed via ``claim(key)`` or ``claim(key, "tensor")``.
* **Object slots** — fixed-size Python-object ring buffers for non-tensor KPN
  payloads such as ``list[PerformanceAtom]``.  Claimed via
  ``claim_object(key)`` or ``claim(key, "object")``.

    All I/O methods dispatch on slot type by key.  ``has_slot`` and ``reset``
    cover both types.  The two namespaces are shared — a key must be unique
    across both slot types.

    Parameters
    ----------
    stride : int
        Samples per tensor frame.  stride=1 for per-sample graph-solver edges.
    fifo_size : int
        Default frames/items per slot.
    dtype : torch.dtype
        Element dtype for tensor slots.  Defaults to complex128.
    device : torch.device
        Allocation device for tensor slots.
    """

    def __init__(
        self,
        stride: int = 1,
        fifo_size: int = 8,
        dtype: torch.dtype = _CDTYPE,
        device: Optional[torch.device] = None,
    ) -> None:
        if stride < 1:
            raise ValueError(f"stride must be ≥ 1, got {stride}")
        if fifo_size < 1:
            raise ValueError(f"fifo_size must be ≥ 1, got {fifo_size}")
        self.stride: int = int(stride)
        self.fifo_size: int = int(fifo_size)
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device or torch.device("cpu")
        self._tensor_slots: Dict[str, _FifoSlot] = {}
        self._object_slots: Dict[str, ObjectFifoSlot] = {}

    # ── capacity ──────────────────────────────────────────────────────────────

    @property
    def max_capacity(self) -> int:
        """Total complex128 samples each tensor slot can buffer."""
        return self.fifo_size * self.stride

    # ── slot management ───────────────────────────────────────────────────────

    def claim(self, key: str, slot_type: str = "tensor", fifo_size: Optional[int] = None) -> Any:
        """Preallocate a slot for *key*.  Idempotent: returns existing slot.

        Parameters
        ----------
        key : str
            Unique slot identifier.
        slot_type : str
            ``"tensor"`` (default) or ``"object"``.
        fifo_size : int, optional
            Override the bank's default fifo_size for this slot.
        """
        if slot_type == "object":
            return self.claim_object(key, fifo_size=fifo_size or self.fifo_size)
        if key not in self._tensor_slots:
            self._tensor_slots[key] = _FifoSlot(
                fifo_size or self.fifo_size, self.stride, self.dtype, self.device
            )
        return self._tensor_slots[key]

    def claim_object(self, key: str, fifo_size: int = 8) -> ObjectFifoSlot:
        """Preallocate a Python-object FIFO slot for *key*.  Idempotent."""
        if key not in self._object_slots:
            self._object_slots[key] = ObjectFifoSlot(fifo_size)
        return self._object_slots[key]

    def has_slot(self, key: str) -> bool:
        return key in self._tensor_slots or key in self._object_slots

    def release(self, key: str) -> None:
        """Remove a slot from the bank (frees storage)."""
        self._tensor_slots.pop(key, None)
        self._object_slots.pop(key, None)

    def slot_keys(self):
        """Iterate all claimed slot keys (tensor and object)."""
        yield from self._tensor_slots.keys()
        yield from self._object_slots.keys()

    # ── I/O ───────────────────────────────────────────────────────────────────

    def write(self, key: str, frame: Tensor) -> None:
        """Write one tensor frame to slot *key*.  Raises FifoFull if no space."""
        self._tensor_slots[key].write(frame)

    def try_write(self, key: str, frame: Tensor) -> bool:
        """Write one tensor frame; returns False (no-op) if the slot is full."""
        return self._tensor_slots[key].try_write(frame)

    def read(self, key: str) -> Any:
        """Consume and return the oldest item.  Raises FifoEmpty if empty."""
        if key in self._object_slots:
            return self._object_slots[key].read()
        return self._tensor_slots[key].read()

    def try_read(self, key: str) -> Optional[Any]:
        """Consume and return the oldest item, or None if empty."""
        if key in self._object_slots:
            return self._object_slots[key].try_read()
        return self._tensor_slots[key].try_read()

    def write_batch(self, key: str, data: Any) -> None:
        """Write a batch to slot *key*.

        * Object slot: ``data`` must be a list — stored as ONE atomic item.
        * Tensor slot: ``data`` must be a Tensor — written as N sequential frames.
        """
        if key in self._object_slots:
            self._object_slots[key].write_batch(data)
        else:
            slot = self._tensor_slots[key]
            frames = data.reshape(-1, slot._stride)
            for i in range(frames.shape[0]):
                slot.write(frames[i])

    def peek(self, key: str) -> Any:
        """Return the oldest item without consuming it.  Raises FifoEmpty if empty."""
        if key in self._object_slots:
            return self._object_slots[key].peek()
        return self._tensor_slots[key].peek()

    def is_full(self, key: str) -> bool:
        if key in self._object_slots:
            return self._object_slots[key].is_full()
        return self._tensor_slots[key].is_full()

    def is_empty(self, key: str) -> bool:
        if key in self._object_slots:
            return self._object_slots[key].is_empty()
        return self._tensor_slots[key].is_empty()

    def count(self, key: str) -> int:
        if key in self._object_slots:
            return self._object_slots[key].count
        return self._tensor_slots[key].count

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all slots without reallocating any storage."""
        for slot in self._tensor_slots.values():
            slot.reset()
        for slot in self._object_slots.values():
            slot.reset()

    def reset_slot(self, key: str) -> None:
        """Reset one slot without reallocating."""
        if key in self._tensor_slots:
            self._tensor_slots[key].reset()
        elif key in self._object_slots:
            self._object_slots[key].reset()

    def __repr__(self) -> str:
        return (
            f"EdgeFifoBank(stride={self.stride}, fifo_size={self.fifo_size}, "
            f"max_capacity={self.max_capacity}, "
            f"tensor_slots={len(self._tensor_slots)}, "
            f"object_slots={len(self._object_slots)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a bank sized for a set of TensorEdges
# ─────────────────────────────────────────────────────────────────────────────

def bank_for_edges(
    edges,
    *,
    stride: int = 1,
    fifo_size: int = 8,
    dtype: torch.dtype = _CDTYPE,
    device: Optional[torch.device] = None,
    bank: Optional["EdgeFifoBank"] = None,
) -> "EdgeFifoBank":
    """Create (or return) an ``EdgeFifoBank`` for the given edge set.

    If *bank* is supplied it is returned as-is (allowing callers to pass an
    existing bank through).  Otherwise a new bank is created with the given
    parameters.

    Parameters
    ----------
    edges :
        Edge list — currently unused; reserved for future per-edge slot claims.
    stride : int
        Samples per tensor frame (passed to EdgeFifoBank).
    fifo_size : int
        Default frames/items per slot.
    dtype, device
        Forwarded to EdgeFifoBank when creating a new one.
    bank : EdgeFifoBank, optional
        Existing bank to return directly.
    """
    if bank is not None:
        return bank
    return EdgeFifoBank(stride=stride, fifo_size=fifo_size, dtype=dtype, device=device)
