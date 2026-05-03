"""sm_plugins/instrument_sim.py
================================
InstrumentSimPlugin — SimulatorPlugin that runs the guitar FDTD/AMR physics
engine inside the SimulatorWorkspace state machine (bell jar).

The plugin is class-based and subclasses SimulatorPlugin.  It does **not**
follow the per-item "m0/m1/..." convention; the guitar is a single holistic
object so there is always exactly one item, keyed ``"guitar"``.

Routing outputs (wired into the SimulatorWorkspace graph):
  ``"mic"``    — aperture-averaged microphone signal, shape (T,), native dtype
  ``"pickup"`` — magnetic pickup signal, shape (T,), native dtype

Field data (available as properties for the bell-jar renderer):
  ``.pressure_field``    — (Nx, Ny, Nz) float32, live pressure volume
  ``.plate_displacement``— (Nx, Ny)     float32, Kirchhoff-plate w(x,y)
  ``.string_positions``  — list[(N_segs+1, 3)] float32, one array per string

Workflow::

    plugin = InstrumentSimPlugin()
    plugin.attach(guitar_item)          # bind a GuitarItem from inventory
    plugin.prime(progress_cb=cb)        # build physics (slow; call off-thread)

    # inside SimulatorWorkspace.step():
    plugin.step(inputs={}, state=plugin.make_state(), dt=512/44100)

Backend modes
-------------
``amr_backend`` in GuitarConfig (or the param override) selects how the AMR
grid is subdivided:
  ``"cpu"`` — pure Python + NumPy subdivision (default, always available)
  ``"gl"``  — OpenGL 4.3 compute-shader subdivision (faster; needs an active
               GL context in the caller thread)

For the non-AMR legacy uniform grid, pass ``dx`` larger than 0.010 m; the
AMR grid devolves to 0 refinement levels at coarse resolution.  A future
``amr_backend="none"`` shortcut may be added to skip the octree entirely.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional

import numpy as np

from sm_plugins._base import SimulatorPlugin

# ── Physics / bridge imports — guarded so the plugin loads even without C ext ──

try:
    from sm_plugins.orchestral_resonance import _build_body_scene as _build_scene_fn
    _HAS_SCENE = True
except Exception:
    _build_scene_fn = None
    _HAS_SCENE = False

try:
    from acoustic_fdtd_bridge import (
        build_acoustic_coevolver_from_scene,
        GUITAR_SCALE_LENGTH_M,
        _extract_guitar_geometry,
        _scene_soundhole,
        _pip_grid,
    )
    _HAS_BRIDGE = True
except Exception:
    build_acoustic_coevolver_from_scene = None   # type: ignore[assignment]
    GUITAR_SCALE_LENGTH_M = 0.648                # fallback — matches acoustic_fdtd_bridge
    _extract_guitar_geometry = None              # type: ignore[assignment]
    _scene_soundhole = None                      # type: ignore[assignment]
    _pip_grid = None                             # type: ignore[assignment]
    _HAS_BRIDGE = False

try:
    from _spectral_kernels import AcousticCoEvolver as _CCoEvolver  # noqa: F401
    _HAS_COEVOLVER = True
except Exception:
    _HAS_COEVOLVER = False

# ── Defaults (mirrors demo_pluck_gl.py constants) ────────────────────────────

_SAMPLE_RATE   = 44100
_BLOCK_SAMPLES = 512
_DX_DEFAULT    = 0.006       # 6 mm cells
_PAD_CELLS     = 56
_N_PML         = 28
_N_RENDER_SEGS = 240
_PLUCK_POS     = 0.20
_PLUCK_AMP     = 0.003
_BRIDGE_FORCE  = 1.0
_STRUM_OFFSETS = [0, 2205, 4410, 6615, 8820, 11025]
_A_STRING_IDX  = 1

# Vibrating string length: imported from acoustic_fdtd_bridge at load time;
# GUITAR_SCALE_LENGTH_M is defined in this module's import block above.
# Different guitar sizes (parlour=0.628, concert=0.648, baritone=0.686) vary here.
_SCALE_LENGTH_M = GUITAR_SCALE_LENGTH_M   # constant; treat as read-only

# ── Divergence / ring-down thresholds ────────────────────────────────────────

_DIAG_HISTORY_LEN     = 20     # rolling window of frame stats (matches demo_pluck_gl)
_DIVERGE_GROWTH_LIMIT = 10.0   # p_max growth factor over 1 frame → flag divergent
_RINGDOWN_THRESHOLD   = 1e-9   # p_rms below this → silence
_RINGDOWN_FRAMES      = 16     # consecutive silent frames before done=True


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic frame record
# ─────────────────────────────────────────────────────────────────────────────

class _DiagFrame(NamedTuple):
    """Per-step field statistics, exactly mirroring demo_pluck_gl diagnostics."""
    frame_index: int
    p_rms:       float
    p_max:       float
    plate_max:   float
    string_max:  float
    step_wall_us: float   # wall-clock time for ce.step(), microseconds
    divergent:   bool     # True if p_max grew by > _DIVERGE_GROWTH_LIMIT vs prev frame


# ─────────────────────────────────────────────────────────────────────────────
# Build state
# ─────────────────────────────────────────────────────────────────────────────

class _BuildState:
    """Thread-safe build progress carrier."""

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self.frac:  float = 0.0
        self.label: str   = "idle"
        self.done:  bool  = False
        self.error: Optional[str] = None

    def update(self, frac: float, label: str) -> None:
        with self._lock:
            self.frac  = max(0.0, min(1.0, float(frac)))
            self.label = str(label)

    def finish(self, error: Optional[str] = None) -> None:
        with self._lock:
            self.done  = True
            self.frac  = 1.0 if error is None else self.frac
            self.error = error

    def snapshot(self) -> tuple[float, str, bool, Optional[str]]:
        with self._lock:
            return (self.frac, self.label, self.done, self.error)


# ─────────────────────────────────────────────────────────────────────────────
# Plugin
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentSimPlugin(SimulatorPlugin):
    """FDTD guitar instrument simulation as a SimulatorWorkspace plugin.

    Lifecycle
    ---------
    1. ``attach(guitar_item)``   — bind a GuitarItem (or raw config dict).
    2. ``prime([progress_cb])``  — build physics (blocking; run off-thread
                                    if you don't want to stall the UI).
    3. ``step(inputs, state, dt, ...)`` — advance one audio block.
    4. ``restart()``             — re-schedule excitation; reset FDTD state.
    5. ``detach()``              — release physics handles; free VRAM/RAM.
    """

    # ── SimulatorPlugin interface ────────────────────────────────────────────

    state_vars  = ["frame_index", "energy_rms"]
    output_vars = ["mic", "pickup"]
    item_prefix = "guitar"

    param_specs = [
        {"name": "fret",        "min": 0,    "max": 12,   "default": 0,
         "label": "Fret",       "integer": True},
        {"name": "pluck_pos",   "min": 0.05, "max": 0.90, "default": _PLUCK_POS,
         "label": "Pluck position (fraction of scale)"},
        {"name": "pluck_amp",   "min": 0.0,  "max": 0.02, "default": _PLUCK_AMP,
         "label": "Pluck amplitude (m)"},
        {"name": "dx",          "min": 0.004,"max": 0.020,"default": _DX_DEFAULT,
         "label": "Grid cell size (m) — rebuild required"},
        {"name": "bridge_force","min": 0.1,  "max": 10.0, "default": _BRIDGE_FORCE,
         "label": "Bridge force scale — rebuild required"},
        # Enum params: SimulatorStation maps these to dropdown widgets
        {"name": "excitation",  "kind": "enum",
         "options": ["strum", "a-pluck", "rest"], "default": "strum",
         "label": "Excitation"},
        {"name": "amr_backend", "kind": "enum",
         "options": ["cpu", "gl"], "default": "cpu",
         "label": "AMR backend (cpu = pure NumPy, gl = OpenGL compute)"},
        {"name": "n_strings",   "min": 1, "max": 6, "default": 6,
         "label": "String count", "integer": True},
    ]

    def __init__(self) -> None:
        # Attached GuitarItem or config dict
        self._item:   Any                  = None
        self._config: dict                 = {}

        # Live physics handles
        self._ce:   Any                    = None   # AcousticCoEvolver C handle
        self._scene: Any                   = None   # CavityScene
        self._info:  dict                  = {}

        # Derived geometry (populated after prime())
        self._n_strings:    int            = 6
        self._render_segs:  int            = _N_RENDER_SEGS

        # Build progress
        self._build_state:  _BuildState    = _BuildState()
        self._is_primed:    bool           = False

        # Latest frame output (set by step())
        self._pressure:  Optional[np.ndarray] = None   # (Nx,Ny,Nz) float32
        self._plate:     Optional[np.ndarray] = None   # (Nx,Ny) float32
        self._strings:   list[np.ndarray]     = []     # [(segs+1,3), ...]
        self._frame_idx: int                  = 0

        # Pending excitation re-schedule flag
        self._reschedule: bool = False

        # Diagnostics (rolling window, thread-safe reads via copy)
        self._diag_history: collections.deque = collections.deque(maxlen=_DIAG_HISTORY_LEN)
        self._divergent:    bool              = False
        self._divergence_cb: Optional[Callable[["_DiagFrame"], None]] = None

        # Ring-down tracking
        self._ringdown_count: int   = 0
        self._done:           bool  = False

    # ── Attachment ───────────────────────────────────────────────────────────

    def attach(self, item: Any) -> None:
        """Bind a GuitarItem (or its ``physics_config()`` dict) to this plugin.

        Must be called before ``prime()``.  Detaches any previously primed
        physics first.
        """
        if self._is_primed:
            self.detach()
        self._item = item
        # Accept either a GuitarItem with .physics_config() or a raw dict
        if hasattr(item, "physics_config"):
            self._config = dict(item.physics_config())
        elif isinstance(item, dict):
            self._config = dict(item)
        else:
            raise TypeError(
                f"InstrumentSimPlugin.attach() expects a GuitarItem or dict, "
                f"got {type(item).__name__}"
            )

    def detach(self) -> None:
        """Release physics handles.  The attached GuitarItem is not modified."""
        self._ce    = None
        self._scene = None
        self._info  = {}
        self._pressure = None
        self._plate    = None
        self._strings  = []
        self._frame_idx = 0
        self._is_primed = False
        self._build_state = _BuildState()
        self._diag_history.clear()
        self._divergent   = False
        self._ringdown_count = 0
        self._done = False

    # ── Build ────────────────────────────────────────────────────────────────

    def prime(
        self,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        *,
        config_overrides: Optional[dict] = None,
    ) -> None:
        """Build the FDTD physics synchronously.

        Parameters
        ----------
        progress_cb:
            Optional ``(fraction: float, label: str) -> None`` callback called
            during grid construction.  Safe to call from a worker thread.
        config_overrides:
            Optional dict of config keys to override before building.
        """
        if not _HAS_SCENE or not _HAS_BRIDGE:
            self._build_state.finish(
                error="Physics unavailable: orchestral_resonance or acoustic_fdtd_bridge not importable."
            )
            return
        if not self._config:
            self._build_state.finish(error="No GuitarItem attached — call attach() first.")
            return

        cfg = dict(self._config)
        if config_overrides:
            cfg.update(config_overrides)

        # ── Resolve build parameters ─────────────────────────────────────────
        n_strings   = int(cfg.get("n_strings", 6))
        dx          = float(cfg.get("dx", _DX_DEFAULT))
        pad_cells   = _pad_cells_from_margin(
            int(cfg.get("pressure_margin_cells", _PAD_CELLS - _N_PML)),
            int(cfg.get("pressure_pml_cells",   _N_PML)),
        )
        n_pml       = int(cfg.get("pressure_pml_cells",    _N_PML))
        n_segs      = int(cfg.get("render_segs",           _N_RENDER_SEGS))
        force_scale = float(cfg.get("bridge_force_scale",  _BRIDGE_FORCE))
        fret        = int(cfg.get("fret",                  0))
        fretless    = bool(cfg.get("fretless",             False))
        amr_backend = str(cfg.get("amr_backend",           "cpu"))
        amr_cache   = bool(cfg.get("amr_cache_grid",       True))
        excitation  = str(cfg.get("excitation",            "strum"))
        scale_m     = _effective_scale_length(fret, cfg.get("scale_length_m", 0.648))

        def _cb(frac: float, label: str) -> None:
            self._build_state.update(frac, label)
            if progress_cb is not None:
                progress_cb(frac, label)

        _cb(0.0, "building scene")

        try:
            scene = _build_scene_fn("string_plate")
            if scene is None:
                raise RuntimeError("_build_body_scene returned None")

            _cb(0.02, "voxelising guitar body (AMR)" if amr_backend != "none" else "voxelising guitar body")

            ce, info = build_acoustic_coevolver_from_scene(
                scene,
                n_strings      = n_strings,
                sample_rate    = float(_SAMPLE_RATE),
                dx             = dx,
                pad_cells      = pad_cells,
                n_pml          = n_pml,
                n_segs         = n_segs,
                force_scale    = force_scale,
                scale_length_m = scale_m,
                fretless       = fretless,
                fret_number    = fret,
                amr_backend    = amr_backend,
                amr_cache_grid = amr_cache,
                amr_gradient_order = 2,
                progress_cb    = _cb,
            )

            if ce is None:
                raise RuntimeError(
                    "build_acoustic_coevolver_from_scene returned None — "
                    "C extension (_spectral_kernels) not built?"
                )

            _cb(0.98, "scheduling excitation")
            # ── plate_active_2d: soundboard mesh mask on the viz grid ──────────
            # Must be computed on the same grid as info (uses _viz_dx in AMR mode,
            # not the physics dx).  Mirrors _build_physics in demo_pluck_gl exactly.
            _info = dict(info or {})
            if _extract_guitar_geometry is not None and _pip_grid is not None:
                try:
                    outline_for_vox, _, _ = _extract_guitar_geometry(scene)
                    if outline_for_vox is None or len(outline_for_vox) < 8:
                        outline_for_vox = None
                    if outline_for_vox is not None:
                        _pdx = float(_info.get('dx', dx))
                        _pNx = int(_info['Nx'])
                        _pNy = int(_info['Ny'])
                        _pxs = _info['gx_min'] + (np.arange(_pNx) + 0.5) * _pdx
                        _pys = _info['gy_min'] + (np.arange(_pNy) + 0.5) * _pdx
                        _pX, _pY = np.meshgrid(_pxs, _pys, indexing='ij')
                        _plate_active = _pip_grid(
                            _pX, _pY,
                            np.asarray(outline_for_vox, dtype=np.float64),
                        ).astype(np.uint8)
                        # Apply soundhole cutout on the same viz grid
                        _sh = _scene_soundhole(scene) if _scene_soundhole is not None else None
                        if _sh is not None:
                            _cx, _cy, _hr = _sh
                            _mask = (_pX - _cx) ** 2 + (_pY - _cy) ** 2 < _hr ** 2
                            _plate_active[_mask] = 0
                        _info['plate_active_2d'] = _plate_active
                except Exception as _e:
                    _cb(0.985, f"plate_active_2d skipped: {_e}")

            ce.reset()
            _schedule_excitation(ce, excitation, n_strings)

            self._ce        = ce
            self._scene     = scene
            self._info      = _info
            self._n_strings = n_strings
            self._render_segs = n_segs
            self._frame_idx   = 0
            self._diag_history.clear()
            self._divergent      = False
            self._ringdown_count = 0
            self._done           = False
            self._is_primed = True
            self._build_state.finish()
            _cb(1.0, "ready")

        except Exception as exc:
            import traceback
            self._build_state.finish(error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            raise

    def prime_async(
        self,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        *,
        config_overrides: Optional[dict] = None,
        on_done: Optional[Callable[[Optional[str]], None]] = None,
    ) -> threading.Thread:
        """Build physics in a daemon background thread.

        Returns the thread immediately.  Use ``build_progress`` to poll
        status from the main thread.  ``on_done(error_or_None)`` is called
        in the worker thread when complete.
        """
        def _worker() -> None:
            err: Optional[str] = None
            try:
                self.prime(progress_cb=progress_cb, config_overrides=config_overrides)
            except Exception as exc:
                err = str(exc)
            if on_done is not None:
                try:
                    on_done(err)
                except Exception:
                    pass
        t = threading.Thread(target=_worker, daemon=True, name="instrument-sim-build")
        t.start()
        return t

    @property
    def build_progress(self) -> tuple[float, str, bool, Optional[str]]:
        """(fraction, label, done, error_or_None) — safe to poll from any thread."""
        return self._build_state.snapshot()

    # ── Excitation control ───────────────────────────────────────────────────

    def restart(self, excitation: Optional[str] = None) -> None:
        """Reset the FDTD state and re-schedule the excitation.

        Parameters
        ----------
        excitation : override excitation kind (``"strum"``, ``"a-pluck"``, ``"rest"``).
                     If None, re-uses the config value.
        """
        if self._ce is None:
            return
        kind = excitation or str(self._config.get("excitation", "strum"))
        self._ce.reset()
        _schedule_excitation(self._ce, kind, self._n_strings)
        self._frame_idx      = 0
        self._diag_history.clear()
        self._divergent      = False
        self._ringdown_count = 0
        self._done           = False

    def reschedule(
        self,
        string_index:  int,
        pluck_pos:     float,
        pluck_amp:     float,
        onset_samples: int = 0,
    ) -> None:
        """Queue a single pluck event onto the CE schedule (additive).

        Parameters
        ----------
        string_index : 0 = low E, 5 = high e
        pluck_pos    : fraction of scale length (0.0 = bridge, 1.0 = nut)
        pluck_amp    : displacement amplitude (m)
        onset_samples: sample offset from now
        """
        if self._ce is None:
            return
        self._ce.schedule_pluck(
            max(0, int(onset_samples)),
            max(0, min(int(string_index), self._n_strings - 1)),
            float(np.clip(pluck_pos, 0.0, 1.0)),
            float(pluck_amp),
        )

    # ── SimulatorPlugin.step ─────────────────────────────────────────────────

    def make_state(self, n_items: int = 1) -> dict:
        return {"guitar": {"frame_index": 0, "energy_rms": 0.0}}

    def step(
        self,
        inputs:    dict,
        state:     dict,
        dt:        float,
        n_items:   int  = 1,
        use_torch: bool = False,
        **kwargs:  Any,
    ) -> dict:
        """Advance the guitar FDTD by one audio block.

        ``dt`` should equal ``BLOCK_SAMPLES / sample_rate`` (≈ 11.6 ms for
        512 samples @ 44100 Hz).  The FDTD always advances by exactly
        ``_BLOCK_SAMPLES`` samples per call; ``dt`` is informational only.

        Returns
        -------
        ``{"guitar": {"mic": array(T,), "pickup": array(T,), "frame_index": array(1,)},
           "__diag__": _DiagFrame(...), "__done__": bool}``

        ``"__diag__"``
            Always present. Contains per-frame field stats + divergence flag.
            The SimulatorWorkspace/SimulatorStation should inspect this key and
            may trigger dt subdivision, reanalysis, or graceful exit when
            ``diag.divergent`` is True.

        ``"__done__"``
            True when ring-down is detected (energy below ``_RINGDOWN_THRESHOLD``
            for ``_RINGDOWN_FRAMES`` consecutive frames).
        """
        if self._ce is None or not self._is_primed:
            T = _BLOCK_SAMPLES
            empty = np.zeros(T, dtype=np.float32)
            null_diag = _DiagFrame(
                frame_index=self._frame_idx,
                p_rms=0.0, p_max=0.0, plate_max=0.0, string_max=0.0,
                step_wall_us=0.0, divergent=False,
            )
            return {
                "guitar":  {"mic": empty.copy(), "pickup": empty.copy(),
                            "frame_index": np.array([self._frame_idx])},
                "__diag__": null_diag,
                "__done__": self._done,
            }

        ce = self._ce
        T  = _BLOCK_SAMPLES

        # Inject any externally routed driver signal (additive).
        driver = None
        for arr in inputs.values():
            a = np.asarray(arr)
            if a.ndim == 1 and len(a) >= T:
                seg = a[:T].real if np.iscomplexobj(a) else a[:T]
                seg = seg.astype(np.float32, copy=False)
                driver = seg if driver is None else driver + seg

        if driver is not None and hasattr(ce, "inject_external_driver"):
            ce.inject_external_driver(driver)

        # ── Advance physics — timed ───────────────────────────────────────────
        _t0 = time.perf_counter()
        ce.step(T)
        step_wall_us = (time.perf_counter() - _t0) * 1e6

        # Pull field snapshots — preserve native dtype from C extension
        self._pressure = ce.get_pressure_field()
        self._plate    = ce.get_plate_displacement()
        self._strings  = [
            ce.get_string_position_n(si, self._render_segs + 1)
            for si in range(self._n_strings)
        ]

        # ── Field diagnostics (mirrors _record_diag / _frame_equilibrium_stats) ──
        P  = self._pressure
        PL = self._plate
        p_rms   = float(np.sqrt(np.mean(P.astype(np.float64, copy=False) ** 2)))
        p_max   = float(np.max(np.abs(P)))
        pl_max  = float(np.max(np.abs(PL))) if PL is not None else 0.0
        s_max   = 0.0
        for s in self._strings:
            if len(s) < 2:
                continue
            t = np.linspace(0.0, 1.0, len(s), dtype=np.float64)[:, None]
            chord = (1.0 - t) * s[0:1] + t * s[-1:]
            s_max = max(s_max, float(np.linalg.norm(
                (s.astype(np.float64, copy=False) - chord), axis=1).max()))

        # Divergence: p_max grew by more than _DIVERGE_GROWTH_LIMIT in one frame
        prev_p_max = self._diag_history[-1].p_max if self._diag_history else 0.0
        divergent  = (prev_p_max > 0.0 and p_max > prev_p_max * _DIVERGE_GROWTH_LIMIT)
        self._divergent = divergent or self._divergent

        diag = _DiagFrame(
            frame_index  = self._frame_idx,
            p_rms        = p_rms,
            p_max        = p_max,
            plate_max    = pl_max,
            string_max   = s_max,
            step_wall_us = step_wall_us,
            divergent    = divergent,
        )
        self._diag_history.append(diag)

        if divergent and self._divergence_cb is not None:
            try:
                self._divergence_cb(diag)
            except Exception:
                pass

        # ── Ring-down detection ───────────────────────────────────────────────
        if p_rms < _RINGDOWN_THRESHOLD:
            self._ringdown_count += 1
        else:
            self._ringdown_count = 0
        if self._ringdown_count >= _RINGDOWN_FRAMES:
            self._done = True

        # Routing outputs
        if hasattr(ce, "get_mic_output"):
            mic = ce.get_mic_output(0, T)
        else:
            mic = np.zeros(T, dtype=np.float32)

        if hasattr(ce, "get_pickup_output"):
            pickup = ce.get_pickup_output(0, T)
        else:
            pickup = np.zeros(T, dtype=np.float32)

        self._frame_idx += 1

        return {
            "guitar": {
                "mic":         mic,
                "pickup":      pickup,
                "frame_index": np.array([self._frame_idx]),
            },
            "__diag__": diag,
            "__done__": self._done,
        }

    # ── Field properties (for bell-jar renderer) ─────────────────────────────

    @property
    def pressure_field(self) -> Optional[np.ndarray]:
        """(Nx, Ny, Nz) float32 live pressure volume, or None before first step."""
        return self._pressure

    @property
    def plate_displacement(self) -> Optional[np.ndarray]:
        """(Nx, Ny) float32 Kirchhoff plate w(x,y), or None before first step."""
        return self._plate

    @property
    def string_positions(self) -> list[np.ndarray]:
        """List[(render_segs+1, 3) float32] — one array per string."""
        return list(self._strings)

    @property
    def grid_info(self) -> dict:
        """Grid metadata dict from build_acoustic_coevolver_from_scene."""
        return dict(self._info)

    @property
    def is_primed(self) -> bool:
        """True once prime() has completed successfully."""
        return self._is_primed

    @property
    def n_strings(self) -> int:
        return self._n_strings

    @property
    def frame_index(self) -> int:
        return self._frame_idx

    @property
    def is_done(self) -> bool:
        """True once ring-down silence has been sustained for _RINGDOWN_FRAMES frames."""
        return self._done

    @property
    def divergent(self) -> bool:
        """True if any step since last restart detected a divergence event."""
        return self._divergent

    @property
    def diag_history(self) -> list["_DiagFrame"]:
        """Snapshot of the rolling diagnostic history (up to _DIAG_HISTORY_LEN frames)."""
        return list(self._diag_history)

    def set_divergence_callback(self, cb: Optional[Callable[["_DiagFrame"], None]]) -> None:
        """Register a callback invoked synchronously inside step() on divergence.

        The callback receives the divergent _DiagFrame.  It is called in the
        stepping thread and should be fast — intended for the SimulatorWorkspace
        or SimulatorStation to trigger dt subdivision, reanalysis, or graceful
        exit rather than propagating NaN/inf through downstream routing.
        """
        self._divergence_cb = cb

    def dump_diag(self) -> str:
        """Format the diagnostic history as a multi-line string (for logging/debug)."""
        if not self._diag_history:
            return "(no diagnostic history — diverged on first step or not yet stepped)"
        lines = [
            f"  {'frame':>6}  {'p_rms':>12}  {'p_max':>12}  {'plate_max':>10}  "
            f"{'s_max':>10}  {'step_µs':>8}  {'div':>4}"
        ]
        prev_p = None
        for d in self._diag_history:
            growth = f" x{d.p_max/prev_p:.1f}" if (prev_p and prev_p > 0) else ""
            lines.append(
                f"  {d.frame_index:>6}  {d.p_rms:>12.6g}  {d.p_max:>12.6g}  "
                f"{d.plate_max:>10.6g}  {d.string_max:>10.6g}  "
                f"{d.step_wall_us:>8.1f}  {'!!!' if d.divergent else '':>4}{growth}"
            )
            prev_p = d.p_max
        return "\n".join(lines)

    def pressure_slice(self, axis: int = 2, idx: int = -1) -> Optional[np.ndarray]:
        """Return a 2-D heatmap slice of the pressure field.

        Parameters
        ----------
        axis : 0=X, 1=Y, 2=Z (default 2 → top-down view of soundboard)
        idx  : slice index along axis; -1 = mid-plane
        """
        P = self._pressure
        if P is None:
            return None
        if idx < 0:
            idx = P.shape[axis] // 2
        idx = max(0, min(P.shape[axis] - 1, int(idx)))
        if   axis == 0: return P[idx, :, :]
        elif axis == 1: return P[:, idx, :]
        else:           return P[:, :, idx]

    def soundboard_pressure(self) -> Optional[np.ndarray]:
        """Pressure on the soundboard plane (Z = plate_iz slice)."""
        iz = self._info.get("plate_iz")
        if iz is None:
            return self.pressure_slice(axis=2)
        return self.pressure_slice(axis=2, idx=int(iz))

    # ── Reconfiguration ──────────────────────────────────────────────────────

    def reconfigure(
        self,
        overrides: dict,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Update config and rebuild physics.

        Keys that require a rebuild (``dx``, ``n_strings``, ``fret``,
        ``fretless``, ``bridge_force_scale``, ``amr_backend``) trigger a full
        ``prime()``.  Keys that only affect scheduling (``excitation``,
        ``pluck_pos``, ``pluck_amp``) are applied immediately without rebuild.
        """
        _REBUILD_KEYS = frozenset({
            "dx", "n_strings", "fret", "fretless",
            "bridge_force_scale", "amr_backend", "amr_cache_grid",
            "pressure_margin_cells", "pressure_pml_cells", "render_segs",
        })
        self._config.update(overrides)
        if any(k in _REBUILD_KEYS for k in overrides):
            self.prime(progress_cb=progress_cb)
        else:
            # Soft update: re-schedule excitation if changed
            if "excitation" in overrides:
                self.restart(excitation=str(overrides["excitation"]))

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "primed" if self._is_primed else "unprimed"
        item   = getattr(self._item, "__class__", type(self._item)).__name__
        return (f"InstrumentSimPlugin({status}, item={item!r}, "
                f"strings={self._n_strings}, frame={self._frame_idx})")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants for wrap_module compat (not used — class-based only)
# ─────────────────────────────────────────────────────────────────────────────

STATE_VARS  = InstrumentSimPlugin.state_vars
OUTPUT_VARS = InstrumentSimPlugin.output_vars
PARAM_SPECS = InstrumentSimPlugin.param_specs
ITEM_PREFIX = InstrumentSimPlugin.item_prefix


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pad_cells_from_margin(margin_cells: int, pml_cells: int) -> int:
    """Total pad = free-air margin + PML, identical to demo_pluck_gl formula."""
    return max(int(pml_cells) + 4, int(margin_cells) + int(pml_cells))


def _effective_scale_length(fret: int, base_scale_m: float = _SCALE_LENGTH_M) -> float:
    """Fretted scale length (vibrating string length) in metres.

    base_scale_m defaults to _SCALE_LENGTH_M (== GUITAR_SCALE_LENGTH_M from
    acoustic_fdtd_bridge, currently 0.648 m = 25.5 inches = Fender standard).
    Different guitar bodies (parlour 0.628 m, baritone 0.686 m, etc.) vary here.
    """
    fret = max(0, int(fret))
    if fret == 0:
        return float(base_scale_m)
    return float(base_scale_m) / (2.0 ** (fret / 12.0))


def _schedule_excitation(ce: Any, kind: str, n_strings: int) -> None:
    if hasattr(ce, "clear_pluck_schedule"):
        ce.clear_pluck_schedule()
    if kind == "rest":
        return
    if kind == "a-pluck":
        si = min(_A_STRING_IDX, max(0, n_strings - 1))
        ce.schedule_pluck(0, si, _PLUCK_POS, _PLUCK_AMP)
    else:   # "strum"
        for si in range(n_strings):
            onset = _STRUM_OFFSETS[si] if si < len(_STRUM_OFFSETS) else si * 2205
            ce.schedule_pluck(onset, si, _PLUCK_POS, _PLUCK_AMP)
