"""camera_software/template.py — Annotated starter template.

Copy this file, rename the class, change `name`, and implement `tick()`.
Everything in here compiles and runs; remove what you don't need.

─────────────────────────────────────────────────────────────────────────────
QUICK REFERENCE — all writable context parameters
─────────────────────────────────────────────────────────────────────────────

  OPTICS
  ──────
  ctx.focal_mm        float   Lens focal length in mm.            [12 – 180]
  ctx.focus_m         float   Focus distance in metres.         [0.05 – 20]
  ctx.aperture        float   Aperture radius.                   [0 – 0.08]
                              0 = fully stopped down (infinite DoF).
  ctx.tilt_x          float   Tilt/shift X (Scheimpflug).       [-0.5 – 0.5]
  ctx.tilt_y          float   Tilt/shift Y.                     [-0.5 – 0.5]
  ctx.ca              float   Chromatic aberration amount.       [0 – 0.02]

  SENSOR / EXPOSURE
  ─────────────────
  ctx.sensor_iso      float   Film/sensor ISO equivalent.       [0.25 – 12]
  ctx.sensor_gain     float   Electronic gain.                  [0.5 – 32]
  ctx.decay           float   Sensor accumulation half-life s.  [0 – 30]
                              0 = single-frame (no persistence).

  RAY PIPELINE
  ────────────
  ctx.ray_density     float   Rays per pixel budget.            [0.1 – 8]
  ctx.ray_exposure    float   Ray integration exposure.         [0.25 – 6]
  ctx.ray_gamma       float   Output gamma.                     [0.2 – 1.6]
  ctx.sensor_rate     float   Rows traced per dispatch.         [1 – 512]
  ctx.sensor_spp      float   Samples per pixel per row.        [1 – 8]
  ctx.sensor_fps      float   Target sensor frame rate.         [0 – 60]
                              0 = trace as fast as possible.

  CAMERA MOUNT
  ────────────
  ctx.pan_deg         float   Head pan angle in degrees.        [0 – 360]
  ctx.tilt_deg        float   Head tilt in degrees.             [tilt_min – tilt_max]

  READ-ONLY DIAGNOSTICS  (None until the renderer has run at least one frame)
  ──────────────────────
  ctx.depth_map       np.ndarray[H,W] float32  Depth pass result.
  ctx.histogram       np.ndarray[256] float32  Normalised luminance histogram.
  ctx.last_ev         float   Measured EV of last frame (log2 of mean lum).
  ctx.focus_score     float   Sharpness metric [0, 1].

  DIRECT OBJECT ACCESS
  ────────────────────
  ctx.item            CameraItem   The owning CameraItem.
  ctx.camera          Camera|None  The GL Camera object (or None if headless).

─────────────────────────────────────────────────────────────────────────────
SUB-SYSTEM OBJECTS  (read/write, owned by the context, applied each frame)
─────────────────────────────────────────────────────────────────────────────

  ctx.sensor_surface  ParametricSurface   Shape of the photosite plane.
  ctx.film_surface    ParametricSurface   Shape of the film/chemical plane.
  ctx.plate_surface   ParametricSurface   Shape of the projection/output plane.
  ctx.exposure_dist   ExposureDistribution   Spatial sensitivity profile.
  ctx.depth_sensor    DepthSensor            Depth-texture capture config.

─────────────────────────────────────────────────────────────────────────────
MOTOR DRIVES
─────────────────────────────────────────────────────────────────────────────

  Create motors in __init__, sync them to the live parameter in on_attach,
  step them every tick, write the result back through ctx.

  MotorDrive(param_name, spline, initial_value)
    .sync(v)                   — Hard-set value and target (no transition).
    .set_target(v, speed, spline) — Begin moving toward v at speed units/s.
    .step(dt)                  — Advance; returns updated value.
    .at_target(tol)            — True when arrived within tol.
    .value                     — Current value.
    .target                    — Destination value.

  SpeedSpline presets:
    SpeedSpline.LINEAR          Constant.
    SpeedSpline.EASE_IN_OUT     Slow start + slow finish (sigmoid).
    SpeedSpline.EASE_IN         Slow start, fast finish.
    SpeedSpline.EASE_OUT        Fast start, slow finish.
    SpeedSpline.OVERSHOOT_DAMP  Fast approach then creep (AF hunting).
    SpeedSpline.SNAP            Instantaneous (no interpolation).

  Custom spline — list of (progress_t, speed_multiplier) knots:
    SpeedSpline([(0.0, 0.0), (0.1, 1.2), (0.9, 1.0), (1.0, 0.0)])

─────────────────────────────────────────────────────────────────────────────
PARAMETRIC SURFACES
─────────────────────────────────────────────────────────────────────────────

  ParametricSurface(preset, *, curvature_r, tilt_x_deg, tilt_y_deg, warp)
    preset:
      ParametricSurface.FLAT          Standard flat plane (default).
      ParametricSurface.CYLINDRICAL   Curved along U axis.
      ParametricSurface.SPHERICAL     Dome / fish-eye surface.
    tilt_x_deg / tilt_y_deg:
      Scheimpflug tilt of the focal plane.  Positive X tilts the top
      of the sensor toward the camera.  Use for architectural /
      macro depth-of-field control.
    warp:
      Callable (u, v) -> (dx, dy, dz) for arbitrary surface distortion.
      u, v are in [0, 1] sensor coordinates.
    .sample(u, v) -> (x, y, z)  — Evaluate the surface at sensor (u,v).
    .set_warp(fn)               — Attach/replace warp function.

─────────────────────────────────────────────────────────────────────────────
EXPOSURE DISTRIBUTION
─────────────────────────────────────────────────────────────────────────────

  ExposureDistribution(mode, **kwargs)
    ExposureDistribution.UNIFORM     — Flat; all pixels equal weight.
    ExposureDistribution.GAUSSIAN    — Centre-weighted.
      kwargs: sigma_x, sigma_y (normalised [0,1]), center=(cx, cy)
    ExposureDistribution.SCANLINE    — Rolling shutter.
      kwargs: direction='h'|'v', fps_override=None
    .weight(u, v) -> float          — Sensitivity at sensor coord.

─────────────────────────────────────────────────────────────────────────────
DEPTH SENSOR
─────────────────────────────────────────────────────────────────────────────

  DepthSensor(enabled, near, far, format, precision)
    format:   'linear' | 'log' | 'inverse' | 'ndc'
    precision: 'f16' | 'f32'

  After enabling, ctx.depth_map is populated the FOLLOWING tick.
  Shape: [H, W] float32.  Use it for disparity-based AF, fog, etc.

─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from camera_software.base import (
    CameraContext,
    CameraSoftware,
    DepthSensor,
    ExposureDistribution,
    MotorDrive,
    ParametricSurface,
    SpeedSpline,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Minimal example  (copy-paste starting point)
# ─────────────────────────────────────────────────────────────────────────────

class MinimalSoftware(CameraSoftware):
    """Rename this and implement tick().  Everything else is optional."""

    name = "minimal"

    def on_attach(self, ctx: CameraContext) -> None:
        # Called once when attached to a camera item.
        # Sync state here — do NOT do camera writes in __init__.
        pass

    def tick(self, dt: float, ctx: CameraContext) -> None:
        # Write camera parameters directly on ctx.
        # dt is elapsed seconds since last tick.
        pass

    def on_detach(self, ctx: CameraContext) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Motor-driven zoom  (smooth focal length transitions)
# ─────────────────────────────────────────────────────────────────────────────

class ZoomController(CameraSoftware):
    """Drives focal length and focus distance with separate motor drives.

    Usage::

        sw = ZoomController()
        camera_item.software.append(sw)
        sw.zoom_to(85.0, speed=40.0)   # 40 mm/s, ease-in-out by default
        sw.focus_to(3.0, speed=5.0)    # 5 m/s
    """

    name = "zoom_controller"

    def __init__(self):
        # Create motors with preferred splines.  Values are placeholders
        # until on_attach() synchronises them to the live camera.
        self._zoom  = MotorDrive('focal_mm', SpeedSpline.EASE_IN_OUT)
        self._focus = MotorDrive('focus_m',  SpeedSpline.EASE_OUT)

    # ── Public API ────────────────────────────────────────────────────────────

    def zoom_to(self, focal_mm: float,
                speed: float = 30.0,
                spline: SpeedSpline = None) -> None:
        """Start a zoom toward *focal_mm* at *speed* mm/s."""
        self._zoom.set_target(focal_mm, speed=speed, spline=spline)

    def focus_to(self, focus_m: float,
                 speed: float = 4.0,
                 spline: SpeedSpline = None) -> None:
        """Start a focus pull toward *focus_m* metres at *speed* m/s."""
        self._focus.set_target(focus_m, speed=speed, spline=spline)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_attach(self, ctx: CameraContext) -> None:
        self._zoom.sync(ctx.focal_mm)
        self._focus.sync(ctx.focus_m)

    def tick(self, dt: float, ctx: CameraContext) -> None:
        self._zoom.step(dt)
        self._focus.step(dt)
        ctx.focal_mm = self._zoom.value
        ctx.focus_m  = self._focus.value


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Auto-exposure  (histogram-driven ISO + gain with motor)
# ─────────────────────────────────────────────────────────────────────────────

class AutoExposure(CameraSoftware):
    """Centre-weighted auto-exposure using the frame histogram.

    Drives ``sensor_iso`` toward a target EV with a motor.  Aperture
    priority mode: aperture is locked; only ISO adjusts.

    Parameters
    ----------
    target_ev :
        Desired mean log2-luminance (0.0 = middle grey).
    iso_speed :
        Motor speed in ISO units/s.
    enabled :
        Can be toggled at runtime.
    """

    name = "auto_exposure"

    def __init__(self,
                 target_ev: float = 0.0,
                 iso_speed: float = 0.5,
                 enabled: bool = True):
        self.target_ev = float(target_ev)
        self.enabled   = enabled
        self._iso_motor = MotorDrive(
            'sensor_iso',
            SpeedSpline.EASE_OUT,   # slow deceleration avoids hunting
        )
        # Centre-weight the histogram reading
        self._dist = ExposureDistribution(
            ExposureDistribution.GAUSSIAN,
            sigma_x=0.40, sigma_y=0.40,
        )
        self._iso_speed = float(iso_speed)

    def on_attach(self, ctx: CameraContext) -> None:
        self._iso_motor.sync(ctx.sensor_iso)
        # Push the gaussian distribution to the context
        ctx.exposure_dist = self._dist

    def tick(self, dt: float, ctx: CameraContext) -> None:
        if not self.enabled:
            return

        # Measure EV from the histogram if available, else from last_ev.
        ev = ctx.last_ev
        if ev is None and ctx.histogram is not None:
            hist = ctx.histogram
            if hist.sum() > 0:
                bins = np.linspace(0.0, 1.0, len(hist))
                mean_lum = float(np.dot(hist / hist.sum(), bins))
                ev = math.log2(max(mean_lum, 1e-6))

        if ev is None:
            return  # no data yet; wait

        error = self.target_ev - ev
        # Translate EV error to ISO delta (1 EV ≈ 2× ISO).
        iso_target = ctx.sensor_iso * (2.0 ** error)
        iso_target = float(np.clip(iso_target, 0.25, 12.0))

        if not self._iso_motor.at_target(tol=0.02):
            self._iso_motor.step(dt)
        else:
            self._iso_motor.set_target(iso_target, speed=self._iso_speed)
            self._iso_motor.step(dt)

        ctx.sensor_iso = self._iso_motor.value


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Auto-focus  (disparity-map contrast AF with motor)
# ─────────────────────────────────────────────────────────────────────────────

class AutoFocus(CameraSoftware):
    """Contrast-detect auto-focus using the depth map.

    Enables the depth sensor and drives ``focus_m`` toward the scene
    depth at the AF point (default: image centre).

    Parameters
    ----------
    af_u, af_v :
        Normalised AF point [0, 1] on the sensor.  Default = centre.
    speed :
        Motor speed in m/s.
    enabled :
        Can be toggled at runtime.
    """

    name = "auto_focus"

    def __init__(self,
                 af_u: float = 0.5,
                 af_v: float = 0.5,
                 speed: float = 6.0,
                 enabled: bool = True):
        self.af_u    = float(af_u)
        self.af_v    = float(af_v)
        self.enabled = enabled
        self._focus_motor = MotorDrive(
            'focus_m',
            SpeedSpline.OVERSHOOT_DAMP,  # realistic AF hunting behaviour
        )
        self._speed = float(speed)
        self._depth_sensor = DepthSensor(
            enabled=True,
            format='inverse',   # 1/depth → disparity-space, great for AF
            near=0.05,
            far=50.0,
        )

    def on_attach(self, ctx: CameraContext) -> None:
        self._focus_motor.sync(ctx.focus_m)
        ctx.depth_sensor = self._depth_sensor

    def tick(self, dt: float, ctx: CameraContext) -> None:
        if not self.enabled:
            return

        depth_map = ctx.depth_map
        if depth_map is None:
            # Depth not ready yet; step existing motor if in motion.
            self._focus_motor.step(dt)
            ctx.focus_m = self._focus_motor.value
            return

        H, W = depth_map.shape
        # Sample a small patch around the AF point.
        patch_r = max(1, int(min(H, W) * 0.04))
        ci = int(self.af_v * H)
        cj = int(self.af_u * W)
        r0, r1 = max(0, ci - patch_r), min(H, ci + patch_r + 1)
        c0, c1 = max(0, cj - patch_r), min(W, cj + patch_r + 1)
        patch = depth_map[r0:r1, c0:c1]

        if patch.size == 0 or not np.isfinite(patch).any():
            self._focus_motor.step(dt)
            ctx.focus_m = self._focus_motor.value
            return

        # disparity (1/depth) → depth in metres
        disp_median = float(np.median(patch[np.isfinite(patch)]))
        if abs(disp_median) < 1e-6:
            self._focus_motor.step(dt)
            ctx.focus_m = self._focus_motor.value
            return
        target_m = float(np.clip(1.0 / disp_median, 0.05, 50.0))

        if self._focus_motor.at_target(tol=0.05):
            self._focus_motor.set_target(target_m, speed=self._speed)
        self._focus_motor.step(dt)
        ctx.focus_m = self._focus_motor.value


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Parametric sensor surface  (curved / tilted focal plane)
# ─────────────────────────────────────────────────────────────────────────────

class CurvedSensorMount(CameraSoftware):
    """Replaces the sensor plane with a cylindrical surface.

    Emulates panoramic film backs or IMAX-style curved sensors.
    Tilt controls add Scheimpflug correction for oblique subjects.
    """

    name = "curved_sensor"

    def __init__(self,
                 curvature_r: float = 0.8,
                 tilt_x_deg: float = 0.0,
                 tilt_y_deg: float = 0.0):
        self._surface = ParametricSurface(
            ParametricSurface.CYLINDRICAL,
            curvature_r=curvature_r,
            tilt_x_deg=tilt_x_deg,
            tilt_y_deg=tilt_y_deg,
        )
        # Optionally add a warp on top — e.g. pincushion correction:
        # self._surface.set_warp(lambda u, v: (
        #     0.02 * (u - 0.5) * ((u - 0.5)**2 + (v - 0.5)**2),
        #     0.02 * (v - 0.5) * ((u - 0.5)**2 + (v - 0.5)**2),
        #     0.0,
        # ))

    def on_attach(self, ctx: CameraContext) -> None:
        ctx.sensor_surface = self._surface

    def tick(self, dt: float, ctx: CameraContext) -> None:
        # Motor-drive the tilt for live Scheimpflug adjustments:
        # ctx.sensor_surface.tilt_x_deg = lerped_value
        pass


class ScheinpflugTilt(CameraSoftware):
    """Motor-drives the sensor tilt for live Scheimpflug correction.

    Useful for architectural shots: tilt the focal plane to follow
    a wall, floor, or oblique subject plane without changing aperture.
    """

    name = "scheimpflug_tilt"

    def __init__(self, speed_deg_s: float = 15.0):
        self._tx_motor = MotorDrive('tilt_x_deg', SpeedSpline.EASE_IN_OUT)
        self._ty_motor = MotorDrive('tilt_y_deg', SpeedSpline.EASE_IN_OUT)
        self._speed    = float(speed_deg_s)
        self._surface  = ParametricSurface(ParametricSurface.FLAT)

    def tilt_to(self, tilt_x_deg: float, tilt_y_deg: float) -> None:
        """Command a new focal-plane tilt."""
        self._tx_motor.set_target(tilt_x_deg, speed=self._speed)
        self._ty_motor.set_target(tilt_y_deg, speed=self._speed)

    def on_attach(self, ctx: CameraContext) -> None:
        self._tx_motor.sync(self._surface.tilt_x_deg)
        self._ty_motor.sync(self._surface.tilt_y_deg)
        ctx.sensor_surface = self._surface

    def tick(self, dt: float, ctx: CameraContext) -> None:
        self._tx_motor.step(dt)
        self._ty_motor.step(dt)
        ctx.sensor_surface.tilt_x_deg = self._tx_motor.value
        ctx.sensor_surface.tilt_y_deg = self._ty_motor.value


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Exposure distribution modes
# ─────────────────────────────────────────────────────────────────────────────

class ExposureProfileSwitcher(CameraSoftware):
    """Switches between UNIFORM, GAUSSIAN, and SCANLINE exposure profiles.

    Call ``set_mode()`` at runtime to transition the spatial sensitivity
    profile.  The active profile is pushed to ctx.exposure_dist each tick.
    """

    name = "exposure_profile"

    UNIFORM  = ExposureDistribution.UNIFORM
    GAUSSIAN = ExposureDistribution.GAUSSIAN
    SCANLINE = ExposureDistribution.SCANLINE

    def __init__(self, initial_mode: str = ExposureDistribution.UNIFORM):
        self._profiles = {
            ExposureDistribution.UNIFORM: ExposureDistribution(
                ExposureDistribution.UNIFORM,
            ),
            ExposureDistribution.GAUSSIAN: ExposureDistribution(
                ExposureDistribution.GAUSSIAN,
                sigma_x=0.35,
                sigma_y=0.35,
                center=(0.5, 0.5),
            ),
            ExposureDistribution.SCANLINE: ExposureDistribution(
                ExposureDistribution.SCANLINE,
                direction='h',   # top-to-bottom rolling shutter
            ),
        }
        self._mode = initial_mode

    def set_mode(self, mode: str) -> None:
        """Switch to a different exposure profile mode."""
        if mode not in self._profiles:
            raise ValueError(f"Unknown mode {mode!r}")
        self._mode = mode

    def set_gaussian(self,
                     sigma_x: float = 0.35,
                     sigma_y: float = 0.35,
                     center: tuple = (0.5, 0.5)) -> None:
        """Re-configure the gaussian profile parameters."""
        self._profiles[ExposureDistribution.GAUSSIAN] = ExposureDistribution(
            ExposureDistribution.GAUSSIAN,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            center=center,
        )

    def set_scanline(self,
                     direction: str = 'h',
                     fps_override: Optional[float] = None) -> None:
        """Re-configure the scanline profile parameters."""
        self._profiles[ExposureDistribution.SCANLINE] = ExposureDistribution(
            ExposureDistribution.SCANLINE,
            direction=direction,
            fps_override=fps_override,
        )

    def on_attach(self, ctx: CameraContext) -> None:
        ctx.exposure_dist = self._profiles[self._mode]

    def tick(self, dt: float, ctx: CameraContext) -> None:
        ctx.exposure_dist = self._profiles[self._mode]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Depth capture pass  (standalone depth-only software)
# ─────────────────────────────────────────────────────────────────────────────

class DepthCapture(CameraSoftware):
    """Enables depth texture capture and makes the map available.

    After at least one frame: ctx.depth_map contains the depth image.
    Other modules (AutoFocus, fog, etc.) can then consume it.

    Parameters
    ----------
    format :
        'linear' | 'log' | 'inverse' | 'ndc'
    near, far :
        Clip distances in scene units.
    """

    name = "depth_capture"

    def __init__(self,
                 format: str = 'linear',
                 near: float = 0.05,
                 far: float = 50.0):
        self._cfg = DepthSensor(
            enabled=True,
            near=near,
            far=far,
            format=format,
        )

    def on_attach(self, ctx: CameraContext) -> None:
        ctx.depth_sensor = self._cfg

    def tick(self, dt: float, ctx: CameraContext) -> None:
        # Keep the config live; other modules read ctx.depth_map.
        ctx.depth_sensor = self._cfg


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Full stack example  (all subsystems in one module)
# ─────────────────────────────────────────────────────────────────────────────

class FullCameraStack(CameraSoftware):
    """Comprehensive example combining every subsystem.

    Not intended to be used as-is — copy the sections you need.
    """

    name = "full_stack"

    def __init__(self):
        # ── Motors for every motorised parameter ──────────────────────────
        self._focal   = MotorDrive('focal_mm',    SpeedSpline.EASE_IN_OUT)
        self._focus   = MotorDrive('focus_m',     SpeedSpline.OVERSHOOT_DAMP)
        self._aperture = MotorDrive('aperture',   SpeedSpline.EASE_OUT)
        self._iso     = MotorDrive('sensor_iso',  SpeedSpline.EASE_OUT)
        self._gain    = MotorDrive('sensor_gain', SpeedSpline.LINEAR)
        self._pan     = MotorDrive('pan_deg',     SpeedSpline.EASE_IN_OUT)
        self._tilt    = MotorDrive('tilt_deg',    SpeedSpline.EASE_IN_OUT)
        self._tilt_x  = MotorDrive('tilt_x',      SpeedSpline.EASE_IN_OUT)
        self._tilt_y  = MotorDrive('tilt_y',      SpeedSpline.EASE_IN_OUT)

        # ── Parametric surfaces ───────────────────────────────────────────
        self.sensor_surface = ParametricSurface(
            ParametricSurface.FLAT,
            tilt_x_deg=0.0,
            tilt_y_deg=0.0,
        )
        self.film_surface = ParametricSurface(
            ParametricSurface.FLAT,
        )
        self.plate_surface = ParametricSurface(
            ParametricSurface.FLAT,
        )

        # ── Exposure distribution ─────────────────────────────────────────
        # Start uniform; call set_exposure_mode() to switch.
        self.exposure_dist = ExposureDistribution(ExposureDistribution.UNIFORM)

        # ── Depth sensor ──────────────────────────────────────────────────
        self.depth_sensor = DepthSensor(
            enabled=False,       # off by default; enable when you need depth
            format='inverse',
            near=0.05,
            far=50.0,
        )

    # ── Surface helpers ───────────────────────────────────────────────────────

    def set_sensor_surface(self, preset: str, **kwargs) -> None:
        """Replace sensor surface preset.  kwargs forwarded to ParametricSurface."""
        self.sensor_surface = ParametricSurface(preset, **kwargs)

    def set_film_surface(self, preset: str, **kwargs) -> None:
        self.film_surface = ParametricSurface(preset, **kwargs)

    def set_plate_surface(self, preset: str, **kwargs) -> None:
        self.plate_surface = ParametricSurface(preset, **kwargs)

    # ── Exposure distribution helpers ─────────────────────────────────────────

    def set_exposure_mode(self, mode: str, **kwargs) -> None:
        """Switch exposure distribution.  kwargs forwarded to ExposureDistribution."""
        self.exposure_dist = ExposureDistribution(mode, **kwargs)

    # ── Motor commands ────────────────────────────────────────────────────────

    def zoom_to(self, mm: float, speed: float = 30.0) -> None:
        self._focal.set_target(mm, speed=speed)

    def focus_to(self, metres: float, speed: float = 5.0) -> None:
        self._focus.set_target(metres, speed=speed)

    def aperture_to(self, r: float, speed: float = 0.01) -> None:
        self._aperture.set_target(r, speed=speed)

    def iso_to(self, iso: float, speed: float = 0.5) -> None:
        self._iso.set_target(iso, speed=speed)

    def pan_to(self, deg: float, speed: float = 45.0) -> None:
        self._pan.set_target(deg, speed=speed)

    def tilt_to(self, deg: float, speed: float = 30.0) -> None:
        self._tilt.set_target(deg, speed=speed)

    def scheimpflug_to(self, tx: float, ty: float, speed: float = 10.0) -> None:
        self._tilt_x.set_target(tx, speed=speed)
        self._tilt_y.set_target(ty, speed=speed)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_attach(self, ctx: CameraContext) -> None:
        # Sync every motor to the current live value.
        self._focal.sync(ctx.focal_mm)
        self._focus.sync(ctx.focus_m)
        self._aperture.sync(ctx.aperture)
        self._iso.sync(ctx.sensor_iso)
        self._gain.sync(ctx.sensor_gain)
        self._pan.sync(ctx.pan_deg)
        self._tilt.sync(ctx.tilt_deg)
        self._tilt_x.sync(ctx.tilt_x)
        self._tilt_y.sync(ctx.tilt_y)
        # Push sub-systems.
        ctx.sensor_surface = self.sensor_surface
        ctx.film_surface   = self.film_surface
        ctx.plate_surface  = self.plate_surface
        ctx.exposure_dist  = self.exposure_dist
        ctx.depth_sensor   = self.depth_sensor

    def tick(self, dt: float, ctx: CameraContext) -> None:
        # Step every motor.
        self._focal.step(dt);    ctx.focal_mm    = self._focal.value
        self._focus.step(dt);    ctx.focus_m     = self._focus.value
        self._aperture.step(dt); ctx.aperture    = self._aperture.value
        self._iso.step(dt);      ctx.sensor_iso  = self._iso.value
        self._gain.step(dt);     ctx.sensor_gain = self._gain.value
        self._pan.step(dt);      ctx.pan_deg     = self._pan.value
        self._tilt.step(dt);     ctx.tilt_deg    = self._tilt.value
        self._tilt_x.step(dt);   ctx.tilt_x      = self._tilt_x.value
        self._tilt_y.step(dt);   ctx.tilt_y      = self._tilt_y.value

        # Refresh sub-systems (allow runtime replacement).
        ctx.sensor_surface = self.sensor_surface
        ctx.film_surface   = self.film_surface
        ctx.plate_surface  = self.plate_surface
        ctx.exposure_dist  = self.exposure_dist
        ctx.depth_sensor   = self.depth_sensor

        # ── Auto-exposure from histogram ──────────────────────────────────
        # Uncomment when you want histogram-driven ISO:
        #
        # if ctx.last_ev is not None:
        #     error = 0.0 - ctx.last_ev           # 0.0 = target EV (middle grey)
        #     iso_target = ctx.sensor_iso * (2.0 ** error)
        #     iso_target = float(np.clip(iso_target, 0.25, 12.0))
        #     self._iso.set_target(iso_target, speed=0.5)

        # ── Depth-map auto-focus ──────────────────────────────────────────
        # Uncomment when depth sensor is enabled:
        #
        # if ctx.depth_map is not None:
        #     H, W = ctx.depth_map.shape
        #     centre_disp = float(ctx.depth_map[H // 2, W // 2])
        #     if abs(centre_disp) > 1e-6:
        #         self._focus.set_target(1.0 / centre_disp, speed=6.0)
