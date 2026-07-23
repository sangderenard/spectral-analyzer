"""Multi-engine rendering graph contracts for the Pluck coordinator.

The graph schedules engines and brokers products; it does not reimplement an
engine. Existing raster, document, acoustic, ray, and wave systems retain their
algorithms and persistent allocations behind pillar adapters.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol

from camera_software.gpu_preview import PreviewProductRegistry


class EngineState(str, Enum):
    READY = "ready"
    QUEUED = "queued"
    RUNNING = "running"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EngineRequest:
    request_id: str
    engine_key: str
    scene_revision: str
    payload: Mapping[str, Any]
    requested_products: tuple[str, ...] = ()
    priority: int = 100
    submitted_at_s: float = field(default_factory=time.monotonic)

    @classmethod
    def create(
        cls,
        engine_key: str,
        payload: Mapping[str, Any],
        *,
        scene_revision: str = "",
        requested_products: Iterable[str] = (),
        priority: int = 100,
    ) -> "EngineRequest":
        stable = json.dumps(
            {"engine": engine_key, "scene": scene_revision, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return cls(
            request_id=hashlib.sha256(stable).hexdigest()[:24],
            engine_key=str(engine_key),
            scene_revision=str(scene_revision),
            payload=copy.deepcopy(dict(payload)),
            requested_products=tuple(str(item) for item in requested_products),
            priority=int(priority),
        )


class EnginePillar(Protocol):
    key: str

    def submit(self, request: EngineRequest) -> bool: ...
    def tick(self, frame_index: int, dt: float) -> None: ...
    def describe(self) -> Mapping[str, Any]: ...
    def shutdown(self) -> None: ...


class ExternalEnginePillar:
    """Graph-visible adapter for a host-driven existing engine."""

    def __init__(
        self,
        key: str,
        engine: Any,
        *,
        capabilities: Iterable[str],
        products: Iterable[str],
    ) -> None:
        self.key = str(key)
        self.engine = engine
        self.capabilities = tuple(str(item) for item in capabilities)
        self.products = tuple(str(item) for item in products)

    def submit(self, request: EngineRequest) -> bool:
        submit = getattr(self.engine, "submit_engine_request", None)
        return bool(submit(request)) if callable(submit) else False

    def tick(self, frame_index: int, dt: float) -> None:
        # The Pluck host already calls this engine at its precise established
        # point.  Registering it does not add a second execution.
        return None

    def describe(self) -> Mapping[str, Any]:
        return {
            "key": self.key,
            "state": EngineState.READY.value,
            "execution": "host-driven",
            "capabilities": self.capabilities,
            "products": self.products,
        }

    def shutdown(self) -> None:
        return None


class OpticalEnginePillar:
    """Persistent gateway to the physics optical engine.

    Requests remain bounded by bench identity: a newer edit replaces an older
    unstarted request for the same bench. The actual transport backend may be
    attached later without altering graph or UI contracts.
    """

    key = "optical"

    def __init__(self, products: PreviewProductRegistry) -> None:
        self.products = products
        self._lock = threading.RLock()
        self._pending: "OrderedDict[str, EngineRequest]" = OrderedDict()
        self._active: EngineRequest | None = None
        self._active_handle: Any = None
        self._submit_backend: Callable[[Mapping[str, Any]], Any] | None = None
        self._poll_backend: Callable[[Any], Mapping[str, Any] | None] | None = None
        self._shutdown_backend: Callable[[], None] | None = None
        self._describe_backend: Callable[[], Mapping[str, Any]] | None = None
        self._state = EngineState.UNAVAILABLE
        self._last_error = "backend not attached"
        self._completed = 0

    def attach_backend(
        self,
        submit: Callable[[Mapping[str, Any]], Any],
        poll: Callable[[Any], Mapping[str, Any] | None],
        *,
        shutdown: Callable[[], None] | None = None,
        describe: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            self._submit_backend = submit
            self._poll_backend = poll
            self._shutdown_backend = shutdown
            self._describe_backend = describe
            self._state = EngineState.QUEUED if self._pending else EngineState.READY
            self._last_error = ""

    def submit(self, request: EngineRequest) -> bool:
        if request.engine_key != self.key:
            return False
        bench_id = str(request.payload.get("bench_id", request.request_id))
        with self._lock:
            self._pending[bench_id] = request
            self._pending.move_to_end(bench_id)
            if self._submit_backend is not None and self._active is None:
                self._state = EngineState.QUEUED
        return True

    def tick(self, frame_index: int, dt: float) -> None:
        del frame_index, dt
        with self._lock:
            if self._submit_backend is None or self._poll_backend is None:
                return
            if self._active is None and self._pending:
                _bench_id, request = self._pending.popitem(last=False)
                try:
                    self._active_handle = self._submit_backend(request.payload)
                    self._active = request
                    self._state = EngineState.RUNNING
                except Exception as exc:
                    self._state = EngineState.FAILED
                    self._last_error = str(exc)
                    return
            if self._active is None:
                self._state = EngineState.READY
                return
            try:
                status = self._poll_backend(self._active_handle)
            except Exception as exc:
                self._state = EngineState.FAILED
                self._last_error = str(exc)
                return
            if status and bool(status.get("complete", False)):
                self._active = None
                self._active_handle = None
                self._completed += 1
                self._state = EngineState.QUEUED if self._pending else EngineState.READY

    def describe(self) -> Mapping[str, Any]:
        with self._lock:
            backend = (
                {}
                if self._describe_backend is None
                else dict(self._describe_backend())
            )
            return {
                "key": self.key,
                "state": self._state.value,
                "execution": "graph-scheduled",
                "capabilities": ("ray", "wave", "mixed", "spectral", "camera"),
                "products": (
                    "surface_scan", "camera_geometry", "light_field",
                    "sensor_accumulation", "field_accumulation",
                    "depth", "material_ids", "transport_diagnostics",
                ),
                "queued": len(self._pending),
                "active_request": None if self._active is None else self._active.request_id,
                "completed": self._completed,
                "last_error": self._last_error,
                "backend": backend,
            }

    def shutdown(self) -> None:
        with self._lock:
            self._pending.clear()
            self._active = None
            self._active_handle = None
            shutdown = self._shutdown_backend
            self._shutdown_backend = None
        if shutdown is not None:
            shutdown()


class MultiEngineRenderGraph:
    """Small coordinator-owned registry and scheduler for engine pillars."""

    def __init__(self, products: PreviewProductRegistry | None = None) -> None:
        self.products = products or PreviewProductRegistry()
        self._pillars: "OrderedDict[str, EnginePillar]" = OrderedDict()
        self._lock = threading.RLock()

    def register(self, pillar: EnginePillar) -> None:
        key = str(pillar.key)
        with self._lock:
            if key in self._pillars:
                raise ValueError(f"render engine pillar {key!r} already registered")
            self._pillars[key] = pillar

    def get(self, key: str) -> EnginePillar | None:
        with self._lock:
            return self._pillars.get(str(key))

    def submit(self, request: EngineRequest) -> bool:
        pillar = self.get(request.engine_key)
        return False if pillar is None else bool(pillar.submit(request))

    def tick(self, frame_index: int, dt: float) -> None:
        with self._lock:
            pillars = tuple(self._pillars.values())
        for pillar in pillars:
            pillar.tick(int(frame_index), float(dt))

    def describe(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "pillars": {
                    key: dict(pillar.describe()) for key, pillar in self._pillars.items()
                },
                "product_revision": self.products.revision,
            }

    def shutdown(self) -> None:
        with self._lock:
            pillars = tuple(reversed(tuple(self._pillars.values())))
        for pillar in pillars:
            pillar.shutdown()


__all__ = [
    "EngineRequest",
    "EngineState",
    "ExternalEnginePillar",
    "MultiEngineRenderGraph",
    "OpticalEnginePillar",
]
