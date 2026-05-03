"""camera_software/auto_computer.py
-------------------------------------
Automatic camera exposure, focus, ISO, and decay control.

``CameraComputer`` is a ``CameraSoftware`` module — attach it to any camera's
``.software`` list and it will tick every frame, driving the four auto systems
based on live diagnostic feedback from the renderer.

    from camera_software import CameraComputer
    cam.software.append(CameraComputer())

Auto modes
----------
auto_focus
    Hill-climbing depth-of-field search.  If a ``depth_map`` is available on
    the context the modal weighted depth is used directly; otherwise a
    coarse-to-fine direction search maximises ``focus_score`` [0, 1].

auto_aperture
    Automatic exposure via aperture control.  Drives the aperture radius so
    the rendered mean luminance (expressed as EV = log2(mean)) converges to
    ``target_ev``.  Uses a low-pass-filtered EV estimate for stability.

auto_iso
    Automatic ISO / sensor gain.  Works in concert with aperture AE —
    coarse gain adjustments when aperture is already at a limit; fine
    adjustment otherwise.  Targets the same ``target_ev``.

auto_decay
    Dynamically adjusts the sensor accumulator's half-life.  When the scene
    is actively receiving signal (high EV) the decay is shortened for
    temporal crispness; during silence it is lengthened so the last image
    persists.

All four modes are independent and can be enabled in any combination.
The modes write back through ``ctx.camera`` (the raw Camera object) rather
than through the ctx proxy properties, since the Camera class is authoritative
for those fields and no intermediate mapping is needed.

All arithmetic is float64 throughout.
"""
from __future__ import annotations

import math
from typing import Optional

from .base import CameraSoftware, CameraContext

__all__ = ["CameraComputer"]


class CameraComputer(CameraSoftware):
    """Automatic camera control module — attach to Camera.software.

    Parameters
    ----------
    target_ev           : float
        Desired mean log2-luminance for auto exposure / ISO (default 0.0 ≈
        18% grey for normalised HDR output).
    aperture_min        : float
        Minimum aperture radius (metres).  Prevents diffraction softening.
    aperture_max        : float
        Maximum aperture radius (metres).
    iso_min, iso_max    : float
        ISO / sensor-gain clamp range.
    focus_step_m        : float
        Initial autofocus hill-climb step in metres.
    focus_min_m         : float
        Minimum focus distance.
    focus_max_m         : float
        Maximum focus distance.
    ae_speed            : float
        Aperture-AE convergence rate (1/s).  Larger = faster.
    iso_speed           : float
        ISO-AE convergence rate (1/s).
    decay_active_s      : float
        Sensor accumulator half-life when scene is loud/bright.
    decay_quiet_s       : float
        Sensor accumulator half-life during silence.
    decay_threshold_ev  : float
        EV below which the scene is considered "quiet" for decay purposes.
    ev_lpf              : float
        Low-pass filter pole (0 → no filtering, 1 → fully frozen).
        Applied to the raw EV reading before any auto logic.
    """

    name: str = "auto_computer"

    def __init__(self,
                 target_ev:          float = 0.0,
                 aperture_min:       float = 1e-4,
                 aperture_max:       float = 0.08,
                 iso_min:            float = 0.25,
                 iso_max:            float = 12.0,
                 focus_step_m:       float = 0.04,
                 focus_min_m:        float = 0.05,
                 focus_max_m:        float = 20.0,
                 ae_speed:           float = 1.2,
                 iso_speed:          float = 0.6,
                 decay_active_s:     float = 2.0,
                 decay_quiet_s:      float = 12.0,
                 decay_threshold_ev: float = -4.0,
                 ev_lpf:             float = 0.85,
                 ) -> None:
        self.target_ev           = float(target_ev)
        self.aperture_min        = float(aperture_min)
        self.aperture_max        = float(aperture_max)
        self.iso_min             = float(iso_min)
        self.iso_max             = float(iso_max)
        self.focus_step_m        = float(focus_step_m)
        self.focus_min_m         = float(focus_min_m)
        self.focus_max_m         = float(focus_max_m)
        self.ae_speed            = float(ae_speed)
        self.iso_speed           = float(iso_speed)
        self.decay_active_s      = float(decay_active_s)
        self.decay_quiet_s       = float(decay_quiet_s)
        self.decay_threshold_ev  = float(decay_threshold_ev)
        self.ev_lpf              = float(ev_lpf)

        # ── Auto mode flags ───────────────────────────────────────────────────
        self.auto_focus:    bool = False
        self.auto_aperture: bool = False
        self.auto_iso:      bool = False
        self.auto_decay:    bool = False

        # ── Internal state ────────────────────────────────────────────────────
        # Autofocus hill-climber
        self._af_dir:       float = 1.0        # +1 or -1
        self._af_step:      float = focus_step_m
        self._af_last_score: float = 0.0
        self._af_wait:      float = 0.0        # settle timer (seconds)
        self._AF_SETTLE:    float = 0.12       # seconds to wait after nudge
        self._AF_MIN_STEP:  float = focus_step_m * 0.005
        self._AF_MAX_STEP:  float = focus_step_m * 4.0

        # EV low-pass filter state
        self._ev_filtered:  Optional[float] = None

        # AE integrator states
        self._ae_integrator: float = 0.0
        self._iso_integrator: float = 0.0

    # ── CameraSoftware interface ──────────────────────────────────────────────

    def on_attach(self, ctx: CameraContext) -> None:
        cam = ctx.camera
        if cam is not None:
            self._af_step         = self.focus_step_m
            self._af_last_score   = 0.0
            self._ev_filtered     = None
            self._ae_integrator   = 0.0
            self._iso_integrator  = 0.0

    def tick(self, dt: float, ctx: CameraContext) -> None:
        dt = max(1e-6, float(dt))
        cam = ctx.camera

        # ── EV low-pass filter ────────────────────────────────────────────────
        raw_ev: Optional[float] = ctx.last_ev
        if raw_ev is not None:
            if self._ev_filtered is None:
                self._ev_filtered = float(raw_ev)
            else:
                p = self.ev_lpf ** dt  # pole adapted to dt (geometric decay)
                self._ev_filtered = p * self._ev_filtered + (1.0 - p) * float(raw_ev)
        ev = self._ev_filtered

        # ── Auto focus ────────────────────────────────────────────────────────
        if self.auto_focus and cam is not None:
            self._tick_af(dt, ctx, cam)

        # ── Auto aperture ─────────────────────────────────────────────────────
        if self.auto_aperture and cam is not None and ev is not None:
            self._tick_ae(dt, ev, cam)

        # ── Auto ISO ──────────────────────────────────────────────────────────
        if self.auto_iso and cam is not None and ev is not None:
            self._tick_iso(dt, ev, cam)

        # ── Auto decay ────────────────────────────────────────────────────────
        if self.auto_decay and cam is not None:
            self._tick_decay(dt, ev, cam)

    # ── Auto focus ────────────────────────────────────────────────────────────

    def _tick_af(self, dt: float, ctx: CameraContext, cam) -> None:
        # ── Direct path: use depth map modal weighted focus ───────────────────
        depth_map = ctx.depth_map
        if depth_map is not None and hasattr(depth_map, '__len__') and len(depth_map) > 0:
            import numpy as np
            dm = np.asarray(depth_map, np.float64).ravel()
            valid = dm[(dm > self.focus_min_m) & (dm < self.focus_max_m)]
            if len(valid):
                # Weighted mode: histogram → weight by inverse distance variance
                hist, edges = np.histogram(valid, bins=64)
                centres = (edges[:-1] + edges[1:]) * 0.5
                if hist.max() > 0:
                    w = hist.astype(np.float64)
                    modal_depth = float(np.sum(w * centres) / np.sum(w))
                    target = float(np.clip(modal_depth, self.focus_min_m,
                                           self.focus_max_m))
                    cur = float(getattr(cam, 'focus_m', 1.6))
                    # Low-pass toward the modal depth (smooth rack)
                    cam.focus_m = float(cur + (target - cur) * min(1.0, dt * 4.0))
                    self._af_last_score = float(ctx.focus_score or 0.0)
                    return   # depth-map path done

        # ── Sharpness hill-climbing ───────────────────────────────────────────
        score = float(ctx.focus_score or 0.0)
        self._af_wait -= dt
        if self._af_wait > 0.0:
            return   # wait for lens to settle before re-evaluating

        cur = float(getattr(cam, 'focus_m', 1.6))

        if score > self._af_last_score + 1e-4:
            # Improving: keep direction, optionally enlarge step
            self._af_step = min(self._af_step * 1.3, self._AF_MAX_STEP)
        elif score < self._af_last_score - 1e-4:
            # Degrading: reverse and shrink
            self._af_dir  = -self._af_dir
            self._af_step = max(self._af_step * 0.5, self._AF_MIN_STEP)

        self._af_last_score = score

        # If step has bottomed out, do a coarse random restart
        if self._af_step < self._AF_MIN_STEP * 1.1:
            self._af_step = self.focus_step_m
            self._af_dir  = 1.0 if cur < (self.focus_min_m + self.focus_max_m) * 0.5 else -1.0

        nudge = self._af_dir * self._af_step
        cam.focus_m = float(max(self.focus_min_m,
                                min(self.focus_max_m, cur + nudge)))
        self._af_wait = self._AF_SETTLE

    # ── Auto aperture ─────────────────────────────────────────────────────────

    def _tick_ae(self, dt: float, ev: float, cam) -> None:
        """Proportional-integral aperture control toward target_ev."""
        error = self.target_ev - ev        # positive = need to open up
        self._ae_integrator += error * dt

        # Integral wind-up clamp
        max_int = 4.0
        self._ae_integrator = max(-max_int, min(max_int, self._ae_integrator))

        # Rate-limited exponential aperture adjustment
        kp = self.ae_speed * 0.35
        ki = self.ae_speed * 0.08
        delta_log = (kp * error + ki * self._ae_integrator) * dt
        delta_log = max(-1.0, min(1.0, delta_log))   # clamp per-frame change

        cur = float(getattr(cam, 'aperture', 0.0))
        if cur < 1e-6:
            # Start from a sensible default if currently zero
            if error > 0.5:
                cur = 0.001
            else:
                return  # nothing to do if pinhole and scene bright

        new_val = cur * math.exp(delta_log)
        cam.aperture = float(max(self.aperture_min, min(self.aperture_max, new_val)))

    # ── Auto ISO ──────────────────────────────────────────────────────────────

    def _tick_iso(self, dt: float, ev: float, cam) -> None:
        """Proportional ISO adjustment — acts as fine exposure trim."""
        error = self.target_ev - ev
        self._iso_integrator += error * dt
        max_int = 3.0
        self._iso_integrator = max(-max_int, min(max_int, self._iso_integrator))

        kp = self.iso_speed * 0.25
        ki = self.iso_speed * 0.06
        delta_log = (kp * error + ki * self._iso_integrator) * dt
        delta_log = max(-0.5, min(0.5, delta_log))

        cur = float(getattr(cam, 'iso',          # panel uses 'iso' attr
                    getattr(cam, 'sensor_iso', 1.4)))
        new_val = cur * math.exp(delta_log)
        new_val = float(max(self.iso_min, min(self.iso_max, new_val)))
        # Write to whichever attr the Camera uses
        if hasattr(cam, 'iso'):
            cam.iso = new_val
        if hasattr(cam, 'sensor_iso'):
            cam.sensor_iso = new_val

    # ── Auto decay ────────────────────────────────────────────────────────────

    def _tick_decay(self, dt: float, ev: Optional[float], cam) -> None:
        """Smoothly interpolate half-life between active and quiet values."""
        if ev is None:
            target_hl = self.decay_quiet_s
        else:
            # Blend: -6 EV = fully quiet, +2 EV above threshold = fully active
            t = (ev - self.decay_threshold_ev) / 6.0
            t = max(0.0, min(1.0, t))
            # Logarithmic interpolation between quiet and active
            log_active = math.log(max(self.decay_active_s, 0.01))
            log_quiet  = math.log(max(self.decay_quiet_s,  0.01))
            target_hl  = math.exp(log_quiet + t * (log_active - log_quiet))

        # Apply to Camera's decay field
        if hasattr(cam, 'decay'):
            cur = float(cam.decay)
            # First-order low-pass toward target
            alpha = min(1.0, dt * 0.8)
            cam.decay = cur + alpha * (target_hl - cur)

    # ── Convenience: sync all flags from a dict (panel integration) ───────────

    def sync_flags(self, flags: dict) -> None:
        """Apply a dict of {key: bool} auto flags from the camera panel."""
        for key in ('auto_focus', 'auto_aperture', 'auto_iso', 'auto_decay'):
            if key in flags:
                setattr(self, key, bool(flags[key]))
