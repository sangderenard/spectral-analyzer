"""
globals_renderer.py
===================

Four-way global default dispatcher.

Architecture
------------
- All objects automatically become owner nodes in the control graph.
- Registered shaders (any backend, any shape) run via the
  ``ShaderFrameWalker`` regardless of the channel mode and stamp their
  outputs into the ``finalized_targets`` mask.
- After the walker completes, anything *not* finalised is passed to the
  global defaults.  There are exactly four globals — one for each
  ``(channel, backend)`` combination::

        2D-C   →  spectral_kernels.DocRenderer + composite()  → CPU RGBA8
        2D-GL  →  doc_renderer.DocRenderer (this repo)        → GL blit
        3D-C   →  spectral_kernels.BaseRasterizer             → CPU RGBA8
        3D-GL  →  Renderer.render() (existing GL pipeline)    → GL pipeline

The 2D and 3D channels are *not* lock-step.  Each channel has its own
``cadence`` (frames-per-call) and ``min_period_s`` (wall-clock floor),
exactly like a registered ``ShaderSpec``.  A channel only fires when both
gates are open AND there is leftover work for it in the current frame.

Selection
---------
``GlobalChannelDispatcher`` reads two attributes off the host renderer
each frame:

    R._mode_2d : RenderMode  (default RenderMode.C)
    R._mode_3d : RenderMode  (default RenderMode.C)

Either may be flipped at runtime; the dispatcher reconfigures lazily.

Outputs
-------
``dispatch(...)`` returns a :class:`GlobalDispatchResult` carrying:

    out_2d_rgba : Optional[np.ndarray]   (H, W, 4) uint8 if the 2D-C path ran
    out_3d_rgba : Optional[np.ndarray]   (H, W, 4) uint8 if the 3D-C path ran
    used_2d     : Optional[str]          'c' | 'gl' | None  (None = skipped)
    used_3d     : Optional[str]          'c' | 'gl' | None

The caller is responsible for blit:

    * If ``mode_3d == GL``: ``Renderer.render()`` produced the 3D frame.
      Any ``out_2d_rgba`` should be uploaded as a texture and composited
      through the existing GL final pass.
    * If ``mode_3d == C``: there is no GL final pass for this frame.
      ``out_3d_rgba`` and (whichever) 2D output are blit by pygame.

OpenGL is to become a fully optional component once this execution
stream is finalised; the dispatcher does not assume a GL context exists.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Async C-channel worker
# ──────────────────────────────────────────────────────────────────────────────
#
# Architectural rule (verbatim from spec):
#
#   "the global render, if it is c, happens asynchronously, leaving a c
#    global shader as a super low priority option, that drops if there
#    is still an asynchronous result not returned, and delivers the
#    content as it arrives finished in a no wait no block sense.  it
#    must always be remembered despite what features we enable in it
#    (and a config parameter should explain how many features to
#    process or drop)"
#
# The worker owns a single background thread, accepts at most one in-
# flight job (additional submits are dropped — never queued), and
# publishes the most recently finished result into a slot the caller
# polls with ``latest()``.  ``latest()`` never blocks and returns
# whatever was stamped last; the slot is sticky (it is NOT cleared on
# read) so a frame whose submit was dropped still presents the prior
# result, which is exactly the "no wait no block" delivery the user
# asked for.

class _AsyncCWorker:
    def __init__(self, name: str):
        self._name = str(name)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._job: Optional[Callable[[], Any]] = None
        self._busy = False
        self._latest: Any = None
        self._latest_seq = 0
        self._submitted_seq = 0
        self._completed_seq = 0
        self._dropped = 0
        self._stop = False
        self._thread = threading.Thread(
            target=self._run, name=f"GlobalC[{self._name}]", daemon=True,
        )
        self._thread.start()

    def submit(self, fn: Callable[[], Any]) -> bool:
        """Try to dispatch ``fn`` on the worker.

        Returns True if accepted, False if the worker was still busy
        with the previous job (in which case the new submission is
        DROPPED — never queued).
        """
        with self._cv:
            if self._busy or self._job is not None:
                self._dropped += 1
                return False
            self._job = fn
            self._busy = True
            self._submitted_seq += 1
            self._cv.notify()
            return True

    def latest(self) -> Any:
        """Return the most recent completed result (sticky, non-blocking)."""
        with self._lock:
            return self._latest

    def latest_seq(self) -> int:
        with self._lock:
            return self._latest_seq

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def stats(self) -> dict:
        with self._lock:
            return {
                "submitted": self._submitted_seq,
                "completed": self._completed_seq,
                "dropped":   self._dropped,
                "busy":      self._busy,
                "latest_seq": self._latest_seq,
            }

    def shutdown(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stop and self._job is None:
                    self._cv.wait()
                if self._stop:
                    return
                fn = self._job
                self._job = None
            # Run outside the lock so submit() can be called by the
            # main thread freely (it will be dropped while busy).
            try:
                out = fn()
            except Exception:
                out = None
            with self._lock:
                if out is not None:
                    self._latest = out
                    self._latest_seq += 1
                self._completed_seq += 1
                self._busy = False


# ──────────────────────────────────────────────────────────────────────────────
# Backend enum (mirrors RenderMode.{C, GL} from demo_pluck_gl)
# ──────────────────────────────────────────────────────────────────────────────

class ChannelBackend(enum.Enum):
    C  = "c"
    GL = "gl"

    @classmethod
    def from_render_mode(cls, mode: Any) -> "ChannelBackend":
        """Coerce a RenderMode (or its .value string) into a backend tag.

        HYBRID and RAYTRACE are treated as GL for the global default
        selection; explicit C means the CPU global runs.
        """
        v = getattr(mode, "value", mode)
        s = str(v).lower() if v is not None else "c"
        return cls.C if s == "c" else cls.GL


# ──────────────────────────────────────────────────────────────────────────────
# Per-channel cadence / min-period gate (mirrors ShaderSpec semantics)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _ChannelGate:
    cadence: int = 1                    # run every N frames
    min_period_s: float = 0.0           # wall-clock floor between runs
    _last_frame: int = -1
    _last_time: float = 0.0

    def due(self, frame_index: int, now: float) -> bool:
        # First call always fires; subsequent calls obey cadence/period.
        if self._last_frame < 0:
            return True
        if self.cadence > 1:
            if (frame_index - self._last_frame) < self.cadence:
                return False
        if self.min_period_s > 0.0:
            if (now - self._last_time) < self.min_period_s:
                return False
        return True

    def stamp(self, frame_index: int, now: float) -> None:
        self._last_frame = frame_index
        self._last_time = now


# ──────────────────────────────────────────────────────────────────────────────
# Result record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GlobalDispatchResult:
    out_2d_rgba: Optional[Any] = None    # numpy (H,W,4) uint8 if C 2D ran
    out_3d_rgba: Optional[Any] = None    # numpy (H,W,4) uint8 if C 3D ran
    used_2d: Optional[str] = None        # 'c' | 'gl' | None
    used_3d: Optional[str] = None        # 'c' | 'gl' | None
    skipped_2d_reason: str = ""
    skipped_3d_reason: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

class GlobalChannelDispatcher:
    """Selects and fires the correct global default per channel per frame.

    Construction does not assume any backend is available.  Backends are
    instantiated lazily on first use, and any one of them being missing
    only disables that one cell of the 2x2 matrix.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        gl_doc_renderer: Optional[Any] = None,
        gl_render_callback: Optional[Any] = None,
        c_doc_backend: Optional[Any] = None,
        geometry_packer: Optional[Any] = None,
        cadence_2d: int = 1,
        cadence_3d: int = 1,
        min_period_2d_s: float = 0.0,
        min_period_3d_s: float = 0.0,
        async_c: bool = True,
        feature_budget_2d: int = 32,
        feature_budget_3d: int = 32,
    ):
        self.width = int(width)
        self.height = int(height)

        # Pre-built GL globals (provided by host).  Either may be None.
        self._gl_doc = gl_doc_renderer        # doc_renderer.DocRenderer instance
        self._gl_render = gl_render_callback  # callable: () -> None  (Renderer.render)

        # Shared C doc state.  When the host already runs a
        # ``doc_renderer.DocRenderer`` (Python wrapper around
        # ``_spectral_kernels.DocRenderer``), the same backend instance is
        # passed in here so 2D-C composites reflect the live submissions
        # without a parallel submission stream.
        self._c_doc = c_doc_backend           # _spectral_kernels.DocRenderer or None
        if self._c_doc is None and gl_doc_renderer is not None:
            self._c_doc = getattr(gl_doc_renderer, "_backend", None)

        # Lazily-built C 3D global.
        self._c_raster = None                 # _spectral_kernels.BaseRasterizer
        self._last_2d_c_rgba = None

        # Geometry packer for 3D-C: host-supplied callable.  Returns a
        # 3-tuple (verts_view, mat_ids, proj) describing the scene.
        # Lighting is NOT a host responsibility — the C rasterizer derives
        # all illumination internally from the EMISSIVE MATERIALS attached
        # to the rendered triangles (see base_rasterizer.cpp::br_render).
        # No global "scene_rgb" / "scene_indirect" tints are accepted;
        # every photon must trace back to a real emitter cluster.
        self._geometry_packer = geometry_packer

        # Cached material bundle references for CPU/GPU sync. The material DB
        # already keeps a prebaked tensor dictionary; we only re-upload when
        # that dictionary object changes.
        self._mat_tensors_ref_c = None

        # Per-channel cadence gates.
        self.gate_2d = _ChannelGate(cadence=int(cadence_2d),
                                    min_period_s=float(min_period_2d_s))
        self.gate_3d = _ChannelGate(cadence=int(cadence_3d),
                                    min_period_s=float(min_period_3d_s))

        # Async workers for the C globals.  Submissions are dropped
        # (NOT queued) while the worker is busy; ``latest()`` returns
        # the most recent completed buffer non-blocking.  ALL leftover
        # CONTENT is always processed -- the budget below caps FEATURES
        # (optional rendering passes), never content.
        self._async_c = bool(async_c)
        if self._async_c:
            self._worker_2d_c: Optional[_AsyncCWorker] = _AsyncCWorker("2D-C")
            self._worker_3d_c: Optional[_AsyncCWorker] = _AsyncCWorker("3D-C")
        else:
            self._worker_2d_c = None
            self._worker_3d_c = None

        # Per-channel feature budget.  Distinct from leftover/content
        # count: this caps how many OPTIONAL RENDERING FEATURES the C
        # global is allowed to run on a given submission (e.g. shadows,
        # specular bounces, AO, secondary materials).  Mandatory passes
        # (geometry rasterization for 3D-C, base composite for 2D-C)
        # are always executed; only the optional feature stack is
        # dropped down to ``budget`` entries when set.  The host
        # registers features via ``register_feature_2d`` /
        # ``register_feature_3d`` in priority order.
        self.feature_budget_2d = int(feature_budget_2d)
        self.feature_budget_3d = int(feature_budget_3d)
        self._features_2d: list[tuple[int, str, Callable[[Any, Mapping[Any, Any]], None]]] = []
        self._features_3d: list[tuple[int, str, Callable[[Any, Mapping[Any, Any]], None]]] = []

    # -- Late-binding hookups -------------------------------------------------

    def attach_c_doc_backend(self, backend: Any) -> None:
        """Set / replace the shared C doc backend instance."""
        self._c_doc = backend

    def attach_geometry_packer(self, packer: Any) -> None:
        """Set / replace the 3D-C geometry packing callback."""
        self._geometry_packer = packer

    # -- C backend lazy init --------------------------------------------------

    def _ensure_c_doc(self):
        # If the host shared its backend, just use it.
        if self._c_doc is not None and self._c_doc is not False:
            return self._c_doc
        if self._c_doc is False:
            return False
        try:
            import _spectral_kernels as _sk
            self._c_doc = _sk.DocRenderer(self.width, self.height)
        except Exception:
            self._c_doc = False
        return self._c_doc

    def _ensure_c_raster(self):
        if self._c_raster is not None:
            return self._c_raster
        try:
            import _spectral_kernels as _sk
            self._c_raster = _sk.BaseRasterizer(self.width, self.height, 16)
        except Exception:
            self._c_raster = False
        return self._c_raster

    # -- Optional rendering features (NOT content) ----------------------------
    #
    # ``feature_budget_{2d,3d}`` caps how many of these are run per
    # submission, in registration order.  ALL leftover content is still
    # rendered by the mandatory base passes; only the optional feature
    # stack obeys the budget.

    def register_feature_2d(self, name: str,
                            fn: Callable[[Any, Mapping[Any, Any]], None],
                            *, priority: int = 100) -> None:
        self._features_2d.append((int(priority), str(name), fn))
        self._features_2d.sort(key=lambda t: t[0])

    def register_feature_3d(self, name: str,
                            fn: Callable[[Any, Mapping[Any, Any]], None],
                            *, priority: int = 100) -> None:
        self._features_3d.append((int(priority), str(name), fn))
        self._features_3d.sort(key=lambda t: t[0])

    def _run_features(self, features: list, budget: int,
                      backend: Any, leftovers: Mapping[Any, Any]) -> None:
        if budget <= 0 or not features:
            return
        for _prio, _name, _fn in features[:int(budget)]:
            try:
                _fn(backend, leftovers)
            except Exception:
                # An optional feature failure must never break the base
                # pass; swallow and continue.  Caller can surface via
                # logging if needed.
                pass

    def shutdown(self) -> None:
        for w in (self._worker_2d_c, self._worker_3d_c):
            if w is not None:
                try:
                    w.shutdown()
                except Exception:
                    pass

    def worker_stats(self) -> dict:
        return {
            "2d_c": self._worker_2d_c.stats() if self._worker_2d_c else None,
            "3d_c": self._worker_3d_c.stats() if self._worker_3d_c else None,
            "feature_budget_2d": self.feature_budget_2d,
            "feature_budget_3d": self.feature_budget_3d,
            "features_2d": [n for _p, n, _f in self._features_2d],
            "features_3d": [n for _p, n, _f in self._features_3d],
        }

    # -- 2D channel -----------------------------------------------------------

    def _job_2d_c(self, leftovers: Mapping[Any, Any]) -> Optional[Any]:
        """Worker-thread body: composite + optional features."""
        rdr = self._ensure_c_doc()
        if not rdr:
            return None
        rdr.flush()
        if rdr.composite_dirty or self._last_2d_c_rgba is None:
            out = rdr.composite()
            self._last_2d_c_rgba = out
        else:
            out = self._last_2d_c_rgba
        # Run optional 2D features (capped by budget).  These touch the
        # SAME backend instance and only run on this worker thread, so
        # serial use is safe.
        self._run_features(self._features_2d, self.feature_budget_2d,
                           rdr, leftovers)
        return out

    def _run_2d_gl(self, leftovers: Mapping[Any, Any], frame_index: int, dt: float) -> bool:
        if self._gl_doc is None:
            return False
        try:
            self._gl_doc._run(targets=leftovers, frame_index=int(frame_index), dt=float(dt))
            return True
        except Exception:
            return False

    # -- 3D channel -----------------------------------------------------------

    def _sync_c_material_bundle(self, rdr: Any) -> None:
        """Upload prebaked material tensors only when the DB reference changes."""
        try:
            import numpy as _np
            from material_db import MaterialDatabase as _MaterialDatabase

            _db = _MaterialDatabase.instance()
            _t = _db.build_tensors()
            if _t is self._mat_tensors_ref_c:
                return
            _pbr = _t.get("pbr", None)
            _ph = _t.get("phong_compat", None)
            _en = _t.get("enamel", None)
            _tx = _t.get("texture_stack", None)
            if _pbr is not None and len(_pbr):
                rdr.set_pbr_chunk(_np.ascontiguousarray(_pbr, dtype=_np.float32))
            if _ph is not None and len(_ph):
                rdr.set_phong_chunk(_np.ascontiguousarray(_ph, dtype=_np.float32))
            if _en is not None and len(_en):
                rdr.set_enamel_chunk(_np.ascontiguousarray(_en, dtype=_np.float32))
            if _tx is not None and len(_tx) and hasattr(rdr, "set_texture_stack_chunk"):
                rdr.set_texture_stack_chunk(_np.ascontiguousarray(_tx, dtype=_np.float32))
            self._mat_tensors_ref_c = _t
        except Exception:
            pass

    def _job_3d_c(self, packed: Any, leftovers: Mapping[Any, Any]) -> Optional[Any]:
        """Worker-thread body: clear + render + features + readback."""
        rdr = self._ensure_c_raster()
        if not rdr:
            return None
        self._sync_c_material_bundle(rdr)
        if packed is None:
            rdr.clear(0.0, 0.0, 0.0, 0.0)
            return rdr.readback_u8()

        try:
            verts_view = mat_ids = proj = None
            if isinstance(packed, (tuple, list)) and len(packed) >= 3:
                verts_view, mat_ids, proj = packed[:3]
        except Exception:
            pass

        rdr.clear(0.0, 0.0, 0.0, 0.0)
        try:
            rdr.render(verts_view, mat_ids, proj)
        except Exception:
            return None
        # Optional 3D features (capped by budget).
        self._run_features(self._features_3d, self.feature_budget_3d,
                           rdr, leftovers)
        return rdr.readback_u8()

    def _run_3d_gl(self) -> bool:
        if self._gl_render is None:
            return False
        try:
            self._gl_render()
            return True
        except Exception:
            return False

    # -- Top-level dispatch ---------------------------------------------------

    def dispatch(
        self,
        *,
        leftovers_2d: Mapping[Any, Any],
        leftovers_3d: Mapping[Any, Any],
        mode_2d: ChannelBackend,
        mode_3d: ChannelBackend,
        frame_index: int,
        dt: float,
    ) -> GlobalDispatchResult:
        """Fire each due channel default on its leftover payload.

        Channels are independent: a 2D miss does not affect 3D and vice
        versa.  Each channel obeys its own cadence / min-period gate.
        """
        result = GlobalDispatchResult()
        now = time.monotonic()

        # ── 2D channel ──────────────────────────────────────────────────
        if not leftovers_2d:
            result.skipped_2d_reason = "no leftovers"
        elif not self.gate_2d.due(frame_index, now):
            result.skipped_2d_reason = "cadence/period"
        else:
            if mode_2d is ChannelBackend.C:
                if self._worker_2d_c is not None:
                    # Async: schedule (drop if busy) and return latest.
                    _lo = leftovers_2d
                    accepted = self._worker_2d_c.submit(lambda: self._job_2d_c(_lo))
                    rgba = self._worker_2d_c.latest()
                    if rgba is not None:
                        result.out_2d_rgba = rgba
                        result.used_2d = "c"
                        self.gate_2d.stamp(frame_index, now)
                    else:
                        result.skipped_2d_reason = (
                            "c worker accepted, awaiting first result"
                            if accepted else "c worker busy, no prior result"
                        )
                else:
                    rgba = self._job_2d_c(leftovers_2d)
                    if rgba is not None:
                        result.out_2d_rgba = rgba
                        result.used_2d = "c"
                        self.gate_2d.stamp(frame_index, now)
                    else:
                        result.skipped_2d_reason = "c backend unavailable or clean"
            else:
                if self._run_2d_gl(leftovers_2d, frame_index, dt):
                    result.used_2d = "gl"
                    self.gate_2d.stamp(frame_index, now)
                else:
                    result.skipped_2d_reason = "gl backend unavailable"

        # ── 3D channel ──────────────────────────────────────────────────
        if not leftovers_3d:
            result.skipped_3d_reason = "no leftovers"
        elif not self.gate_3d.due(frame_index, now):
            result.skipped_3d_reason = "cadence/period"
        else:
            if mode_3d is ChannelBackend.C:
                # Pack geometry on the calling thread (cheap; needs the
                # host's live camera matrices), then ship the immutable
                # packed tuple to the worker.  The worker owns ALL
                # rasterizer mutation.
                packed = None
                if self._geometry_packer is not None:
                    try:
                        packed = self._geometry_packer(leftovers_3d)
                    except Exception:
                        packed = None
                if self._worker_3d_c is not None:
                    _lo = leftovers_3d
                    _pk = packed
                    accepted = self._worker_3d_c.submit(
                        lambda: self._job_3d_c(_pk, _lo)
                    )
                    rgba = self._worker_3d_c.latest()
                    if rgba is not None:
                        result.out_3d_rgba = rgba
                        result.used_3d = "c"
                        self.gate_3d.stamp(frame_index, now)
                    else:
                        result.skipped_3d_reason = (
                            "c worker accepted, awaiting first result"
                            if accepted else "c worker busy, no prior result"
                        )
                else:
                    rgba = self._job_3d_c(packed, leftovers_3d)
                    if rgba is not None:
                        result.out_3d_rgba = rgba
                        result.used_3d = "c"
                        self.gate_3d.stamp(frame_index, now)
                    else:
                        result.skipped_3d_reason = "c backend unavailable"
            else:
                if self._run_3d_gl():
                    result.used_3d = "gl"
                    self.gate_3d.stamp(frame_index, now)
                else:
                    result.skipped_3d_reason = "gl backend unavailable"

        return result

    # -- Convenience: configure cadence at runtime ----------------------------

    def set_cadence_2d(self, *, cadence: Optional[int] = None,
                       min_period_s: Optional[float] = None) -> None:
        if cadence is not None:
            self.gate_2d.cadence = max(1, int(cadence))
        if min_period_s is not None:
            self.gate_2d.min_period_s = max(0.0, float(min_period_s))

    def set_cadence_3d(self, *, cadence: Optional[int] = None,
                       min_period_s: Optional[float] = None) -> None:
        if cadence is not None:
            self.gate_3d.cadence = max(1, int(cadence))
        if min_period_s is not None:
            self.gate_3d.min_period_s = max(0.0, float(min_period_s))
