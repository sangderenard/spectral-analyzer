"""Narrow qualification bridge to Nodus's runtime-only typed FIFO C ABI.

This intentionally wraps only domain-neutral byte transport. New builds use
``nodus_runtime`` directly. The historical ``canvas_tables`` ABI remains an
explicit compatibility fallback while deployed builds migrate; neither path
adds a scheduler or owns optical policy.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional


NODUS_OPTICAL_RECEPTION_TYPE_ID = 0x4F525431  # "ORT1"
_TENSOR_LAYOUT_OPAQUE = 2
_TENSOR_DTYPE_BYTES = 8


class _TensorSpec(ctypes.Structure):
    _fields_ = [
        ("dims", ctypes.c_int32 * 8),
        ("dim_count", ctypes.c_int32),
        ("slots", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("elem_size", ctypes.c_int32),
        ("type_id", ctypes.c_int32),
        ("layout", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
    ]


class NodusFifoQualificationBridge:
    """Own one lossless typed Nodus edge for fixed-size process tokens."""

    def __init__(
        self,
        library_path: str | Path,
        *,
        element_size: int,
        capacity: int,
        type_id: int,
    ) -> None:
        if int(element_size) < 1 or int(capacity) < 1:
            raise ValueError("Nodus FIFO element size and capacity must be positive")
        self.element_size = int(element_size)
        self.capacity = int(capacity)
        self.type_id = int(type_id)
        self._library = ctypes.CDLL(str(Path(library_path)))
        self._runtime_api = hasattr(self._library, "nodus_edge_runtime_create")
        self._bind()
        self._closed = False
        self._readers: set[int] = set()
        if self._runtime_api:
            self._context = self._library.nodus_edge_runtime_create()
            if not self._context:
                raise RuntimeError("Nodus failed to create an edge runtime")
            dims = (ctypes.c_int32 * 1)(1)
            if not self._library.nodus_edge_runtime_configure(
                self._context,
                dims,
                1,
                self.capacity,
                0,
                self.element_size,
                self.type_id,
                _TENSOR_LAYOUT_OPAQUE,
                _TENSOR_DTYPE_BYTES,
            ):
                self.close()
                raise RuntimeError("Nodus rejected the runtime FIFO specification")
            self.edge_index = 0
            return

        self._context = self._library.gp_table_create(None)
        if not self._context:
            raise RuntimeError("Nodus failed to create a table context")
        edge_index = int(self._library.gp_table_get_edge_count(self._context))
        if not self._library.gp_table_add_edge(self._context, 1, 2):
            self.close()
            raise RuntimeError("Nodus failed to create a typed edge")
        self.edge_index = edge_index
        spec = _TensorSpec()
        spec.dims[0] = 1
        spec.dim_count = 1
        spec.slots = self.capacity
        spec.top_k = 0
        spec.elem_size = self.element_size
        spec.type_id = self.type_id
        spec.layout = _TENSOR_LAYOUT_OPAQUE
        spec.dtype = _TENSOR_DTYPE_BYTES
        if not self._library.gp_table_edge_set_tensor_spec(
            self._context, self.edge_index, ctypes.byref(spec)
        ):
            self.close()
            raise RuntimeError("Nodus rejected the typed FIFO specification")

    def _bind(self) -> None:
        lib = self._library
        if self._runtime_api:
            lib.nodus_edge_runtime_create.restype = ctypes.c_void_p
            lib.nodus_edge_runtime_destroy.argtypes = [ctypes.c_void_p]
            lib.nodus_edge_runtime_configure.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int32,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
            ]
            lib.nodus_edge_runtime_configure.restype = ctypes.c_int32
            lib.nodus_edge_runtime_subscribe.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int32
            ]
            lib.nodus_edge_runtime_subscribe.restype = ctypes.c_int32
            lib.nodus_edge_runtime_publish.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int32),
            ]
            lib.nodus_edge_runtime_publish.restype = ctypes.c_int32
            lib.nodus_edge_runtime_consume.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            lib.nodus_edge_runtime_consume.restype = ctypes.c_int32
            lib.nodus_edge_runtime_unread.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64
            ]
            lib.nodus_edge_runtime_unread.restype = ctypes.c_uint64
            lib.nodus_edge_runtime_is_quiescent.argtypes = [ctypes.c_void_p]
            lib.nodus_edge_runtime_is_quiescent.restype = ctypes.c_int32
            lib.nodus_edge_runtime_snapshot_size.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)
            ]
            lib.nodus_edge_runtime_snapshot_size.restype = ctypes.c_int32
            lib.nodus_edge_runtime_snapshot_fill.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
            ]
            lib.nodus_edge_runtime_snapshot_fill.restype = ctypes.c_int32
            lib.nodus_edge_runtime_snapshot_restore.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
            ]
            lib.nodus_edge_runtime_snapshot_restore.restype = ctypes.c_int32
            return

        lib.gp_table_create.argtypes = [ctypes.c_void_p]
        lib.gp_table_create.restype = ctypes.c_void_p
        lib.gp_table_destroy.argtypes = [ctypes.c_void_p]
        lib.gp_table_get_edge_count.argtypes = [ctypes.c_void_p]
        lib.gp_table_get_edge_count.restype = ctypes.c_int32
        lib.gp_table_add_edge.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64
        ]
        lib.gp_table_add_edge.restype = ctypes.c_int32
        lib.gp_table_edge_set_tensor_spec.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(_TensorSpec)
        ]
        lib.gp_table_edge_set_tensor_spec.restype = ctypes.c_int32
        lib.gp_table_edge_subscribe.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint64
        ]
        lib.gp_table_edge_subscribe.restype = ctypes.c_int32
        lib.gp_table_edge_publish.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.gp_table_edge_publish.restype = ctypes.c_int32
        lib.gp_table_edge_consume.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.gp_table_edge_consume.restype = ctypes.c_int32
        lib.gp_table_edge_unread.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.gp_table_edge_unread.restype = ctypes.c_int32
        lib.gp_table_edge_transaction_snapshot_size.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.gp_table_edge_transaction_snapshot_size.restype = ctypes.c_int32
        lib.gp_table_edge_transaction_snapshot_fill.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t
        ]
        lib.gp_table_edge_transaction_snapshot_fill.restype = ctypes.c_int32
        lib.gp_table_edge_transaction_snapshot_restore.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t
        ]
        lib.gp_table_edge_transaction_snapshot_restore.restype = ctypes.c_int32

    def subscribe(self, reader_id: int) -> None:
        if self._runtime_api:
            accepted = self._library.nodus_edge_runtime_subscribe(
                self._context, int(reader_id), 1
            )
        else:
            accepted = self._library.gp_table_edge_subscribe(
                self._context, self.edge_index, int(reader_id)
            )
        if not accepted:
            raise RuntimeError("Nodus rejected the FIFO reader")
        self._readers.add(int(reader_id))

    def publish(self, writer_id: int, payload: bytes) -> bool:
        if len(payload) != self.element_size:
            raise ValueError("Nodus token payload has the wrong fixed size")
        storage = ctypes.create_string_buffer(payload, len(payload))
        dropped = ctypes.c_int32()
        if self._runtime_api:
            accepted = self._library.nodus_edge_runtime_publish(
                self._context,
                int(writer_id),
                storage,
                len(payload),
                ctypes.byref(dropped),
            )
        else:
            accepted = self._library.gp_table_edge_publish(
                self._context,
                self.edge_index,
                int(writer_id),
                storage,
                len(payload),
                ctypes.byref(dropped),
            )
        if accepted and dropped.value:
            raise RuntimeError("lossless Nodus optical FIFO dropped a token")
        return bool(accepted)

    def consume(self, reader_id: int) -> Optional[bytes]:
        storage = ctypes.create_string_buffer(self.element_size)
        if self._runtime_api:
            written = ctypes.c_size_t()
            accepted = self._library.nodus_edge_runtime_consume(
                self._context,
                int(reader_id),
                storage,
                self.element_size,
                ctypes.byref(written),
            )
        else:
            written = ctypes.c_int32()
            accepted = self._library.gp_table_edge_consume(
                self._context,
                self.edge_index,
                int(reader_id),
                storage,
                self.element_size,
                ctypes.byref(written),
            )
        if not accepted:
            return None
        if written.value != self.element_size:
            raise RuntimeError("Nodus returned a partial fixed-size token")
        return bytes(storage.raw[:written.value])

    def unread(self, reader_id: int) -> int:
        if self._runtime_api:
            return int(self._library.nodus_edge_runtime_unread(
                self._context, int(reader_id)
            ))
        count = ctypes.c_int32()
        if not self._library.gp_table_edge_unread(
            self._context,
            self.edge_index,
            int(reader_id),
            ctypes.byref(count),
        ):
            raise RuntimeError("Nodus failed to report reader progress")
        return int(count.value)

    @property
    def quiescent(self) -> bool:
        """Whether every registered reader reached the current write frontier."""

        if self._runtime_api:
            return bool(
                self._library.nodus_edge_runtime_is_quiescent(self._context)
            )
        return all(self.unread(reader_id) == 0 for reader_id in self._readers)

    @property
    def uses_extracted_runtime(self) -> bool:
        """Whether this bridge loaded the runtime-only Nodus ABI."""

        return self._runtime_api

    def copy_shallow(self) -> bytes:
        """Capture a quiescent transaction checkpoint for Turing rollback."""

        size = ctypes.c_size_t()
        if self._runtime_api:
            sized = self._library.nodus_edge_runtime_snapshot_size(
                self._context, ctypes.byref(size)
            )
        else:
            sized = self._library.gp_table_edge_transaction_snapshot_size(
                self._context, self.edge_index, ctypes.byref(size)
            )
        if not sized:
            raise RuntimeError("Nodus failed to size a FIFO transaction snapshot")
        storage = ctypes.create_string_buffer(size.value)
        if self._runtime_api:
            filled = self._library.nodus_edge_runtime_snapshot_fill(
                self._context, storage, size.value
            )
        else:
            filled = self._library.gp_table_edge_transaction_snapshot_fill(
                self._context, self.edge_index, storage, size.value
            )
        if not filled:
            raise RuntimeError("Nodus failed to capture a FIFO transaction snapshot")
        return bytes(storage.raw)

    def restore(self, snapshot: bytes) -> None:
        """Restore storage, publication tags, writer, and every reader cursor."""

        storage = ctypes.create_string_buffer(snapshot, len(snapshot))
        if self._runtime_api:
            restored = self._library.nodus_edge_runtime_snapshot_restore(
                self._context, storage, len(snapshot)
            )
        else:
            restored = self._library.gp_table_edge_transaction_snapshot_restore(
                self._context, self.edge_index, storage, len(snapshot)
            )
        if not restored:
            raise RuntimeError("Nodus rejected the FIFO transaction snapshot")

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            if self._runtime_api:
                self._library.nodus_edge_runtime_destroy(self._context)
            else:
                self._library.gp_table_destroy(self._context)
            self._closed = True

    def __enter__(self) -> "NodusFifoQualificationBridge":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "NODUS_OPTICAL_RECEPTION_TYPE_ID",
    "NodusFifoQualificationBridge",
]
