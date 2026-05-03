"""camera_software.base — Core API for camera software modules.

Every camera-software feature is expressed as a CameraSoftware subclass.
Instances are attached to a CameraItem via ``camera_item.software.append()``.
Each frame the host calls ``CameraItem.tick(dt)`` which builds a
``CameraContext`` proxy and calls ``sw.tick(dt, ctx)`` on every attached
module in order.  Modules write back to the camera through the context;
no return value is used.

Fast path
---------
For simple one-liners that just bump a parameter, write directly::

    ctx.focal_mm = 50.0

Motor path
----------
For physically-motivated transitions, create ``MotorDrive`` instances in
``__init__``, synchronise them on attach, and step them each tick::

    self._zoom = MotorDrive('focal_mm', SpeedSpline.EASE_IN_OUT)
    self._zoom.set_target(50.0, speed=20.0)   # mm/s
    # in tick:
    self._zoom.step(dt)
    ctx.focal_mm = self._zoom.value
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from camera_item import CameraItem


# ─────────────────────────────────────────────────────────────────────────────
# SpeedSpline
# ─────────────────────────────────────────────────────────────────────────────

class SpeedSpline:
    """Piecewise speed profile for motor transitions.

    A spline is a list of (t, speed_multiplier) knots where *t* is
    normalised progress [0, 1] through the transition and
    *speed_multiplier* scales the base speed at that point.  Values
    between knots are linearly interpolated.

    Preset class attributes (use as ``SpeedSpline.EASE_IN_OUT`` etc.).
    """

    #: Constant speed throughout.
    LINEAR: "SpeedSpline"
    #: Slow start, slow finish — smooth sigmoid.
    EASE_IN_OUT: "SpeedSpline"
    #: Starts slow, finishes fast.
    EASE_IN: "SpeedSpline"
    #: Starts fast, finishes slow (motor deceleration).
    EASE_OUT: "SpeedSpline"
    #: Fast then very slow approach — like an AF motor hunting.
    OVERSHOOT_DAMP: "SpeedSpline"
    #: Instant snap — teleport, no interpolation.
    SNAP: "SpeedSpline"

    def __init__(self, knots: Sequence[Tuple[float, float]]):
        """Parameters
        ----------
        knots :
            Sequence of ``(t, speed_multiplier)`` pairs, sorted by *t*.
            ``t=0`` is transition start; ``t=1`` is arrival.
        """
        self._knots: List[Tuple[float, float]] = sorted(knots, key=lambda k: k[0])

    @classmethod
    def from_knots(cls, knots: Sequence[Tuple[float, float]]) -> "SpeedSpline":
        """Alias for the constructor; use for readability."""
        return cls(knots)

    def __call__(self, t: float) -> float:
        """Evaluate speed multiplier at progress *t* ∈ [0, 1]."""
        t = float(np.clip(t, 0.0, 1.0))
        ks = self._knots
        if not ks:
            return 1.0
        if t <= ks[0][0]:
            return ks[0][1]
        if t >= ks[-1][0]:
            return ks[-1][1]
        for i in range(len(ks) - 1):
            t0, v0 = ks[i]
            t1, v1 = ks[i + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 1e-9 else 0.0
                return v0 + alpha * (v1 - v0)
        return ks[-1][1]


SpeedSpline.LINEAR        = SpeedSpline([(0.0, 1.0), (1.0, 1.0)])
SpeedSpline.EASE_IN_OUT   = SpeedSpline([(0.0, 0.05), (0.25, 1.0), (0.75, 1.0), (1.0, 0.05)])
SpeedSpline.EASE_IN       = SpeedSpline([(0.0, 0.05), (0.5, 0.6), (1.0, 1.0)])
SpeedSpline.EASE_OUT      = SpeedSpline([(0.0, 1.0), (0.5, 0.6), (1.0, 0.05)])
SpeedSpline.OVERSHOOT_DAMP = SpeedSpline([(0.0, 2.0), (0.6, 0.8), (0.9, 0.15), (1.0, 0.02)])
SpeedSpline.SNAP          = SpeedSpline([(0.0, 1e6), (1.0, 1e6)])


# ─────────────────────────────────────────────────────────────────────────────
# MotorDrive
# ─────────────────────────────────────────────────────────────────────────────

class MotorDrive:
    """Drives a single scalar parameter toward a target via a speed spline.

    The motor tracks progress as ``abs(value - original_start) / abs(target -
    original_start)`` and queries the spline for the speed multiplier at that
    progress.  This decouples distance from speed profile shape.

    Usage
    -----
    ::

        motor = MotorDrive('focal_mm', SpeedSpline.EASE_IN_OUT)
        motor.value = 35.0
        motor.set_target(85.0, speed=30.0)   # 30 mm/s

        # in tick():
        motor.step(dt)
        ctx.focal_mm = motor.value
    """

    def __init__(self,
                 param: str,
                 spline: SpeedSpline = None,
                 value: float = 0.0):
        self.param   = param
        self.spline  = spline or SpeedSpline.EASE_IN_OUT
        self.value   = float(value)
        self.target  = float(value)
        self._start  = float(value)
        self._speed  = 0.0          # units/second at spline multiplier 1.0
        self._done   = True

    # ── Public API ────────────────────────────────────────────────────────────

    def set_target(self,
                   target: float,
                   speed: float = 1.0,
                   spline: SpeedSpline = None) -> None:
        """Begin moving toward *target* at *speed* units/second (at spline == 1).

        Calling again mid-transition re-targets from the current position.
        """
        if abs(target - self.value) < 1e-9:
            self.value = target; self.target = target; self._done = True
            return
        self._start  = self.value
        self.target  = float(target)
        self._speed  = abs(float(speed))
        if spline is not None:
            self.spline = spline
        self._done   = False

    def step(self, dt: float) -> float:
        """Advance by *dt* seconds.  Returns the updated value."""
        if self._done:
            return self.value
        span = abs(self.target - self._start)
        if span < 1e-9:
            self.value = self.target; self._done = True
            return self.value
        progress = abs(self.value - self._start) / span
        mult     = self.spline(progress)
        step     = self._speed * mult * dt
        remaining = self.target - self.value
        if abs(step) >= abs(remaining):
            self.value = self.target; self._done = True
        else:
            self.value += math.copysign(step, remaining)
        return self.value

    def at_target(self, tol: float = 1e-3) -> bool:
        """True when the motor has arrived within *tol* of target."""
        return self._done or abs(self.value - self.target) <= tol

    def sync(self, value: float) -> None:
        """Hard-set value and target (no transition); use on attach."""
        self.value = float(value); self.target = float(value); self._done = True


# ─────────────────────────────────────────────────────────────────────────────
# ExposureDistribution
# ─────────────────────────────────────────────────────────────────────────────

class ExposureDistribution:
    """Spatial sensitivity profile across the sensor surface.

    Modes
    -----
    ``UNIFORM``
        Flat response: every pixel receives equal weight.  Default.

    ``GAUSSIAN``
        Centre-weighted.  Configurable via *sigma_x*, *sigma_y* (both in
        normalised sensor units [0, 1]) and *center* (u, v) offset.

    ``SCANLINE``
        Rolling-shutter style: rows accumulate exposure progressively
        over the frame period.  Direction can be ``'h'`` (horizontal
        scan, top→bottom) or ``'v'`` (vertical scan, left→right).
        ``fps_override`` forces a different line clock if needed.
    """

    UNIFORM  = "uniform"
    GAUSSIAN = "gaussian"
    SCANLINE = "scanline"

    def __init__(self,
                 mode: str = "uniform",
                 *,
                 # Gaussian params:
                 sigma_x: float = 0.35,
                 sigma_y: float = 0.35,
                 center: Tuple[float, float] = (0.5, 0.5),
                 # Scanline params:
                 direction: str = 'h',
                 fps_override: Optional[float] = None):
        self.mode         = mode
        self.sigma_x      = float(sigma_x)
        self.sigma_y      = float(sigma_y)
        self.center       = (float(center[0]), float(center[1]))
        self.direction    = direction
        self.fps_override = fps_override

    def weight(self, u: float, v: float) -> float:
        """Return normalised sensitivity weight [0, 1] at sensor coordinate (u, v).

        *u* and *v* are normalised to [0, 1] across the sensor.
        Scanline weight is returned as a time-independent positional factor;
        the ray pipeline applies the temporal offset separately.
        """
        if self.mode == self.UNIFORM:
            return 1.0
        if self.mode == self.GAUSSIAN:
            du = (u - self.center[0]) / max(self.sigma_x, 1e-6)
            dv = (v - self.center[1]) / max(self.sigma_y, 1e-6)
            return float(math.exp(-0.5 * (du * du + dv * dv)))
        if self.mode == self.SCANLINE:
            # Returns linear scanline position; pipeline uses this as phase.
            return u if self.direction == 'h' else v
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# ParametricSurface
# ─────────────────────────────────────────────────────────────────────────────

class ParametricSurface:
    """Describes the shape of the sensor, film, or plate plane.

    The pipeline samples this surface to compute where each ray lands after
    passing through the optical system.  The default is a flat plane at
    z=0 in sensor-local space.

    Built-in presets
    ----------------
    ``FLAT``
        Standard flat focal plane.  No distortion.

    ``CYLINDRICAL``
        Curved film plane — useful for panoramic or IMAX emulation.
        Configure radius via *curvature_r*.

    ``SPHERICAL``
        Dome / fish-eye sensor surface.  Configure radius via *curvature_r*.

    Scheimpflug controls
    --------------------
    *tilt_x_deg* and *tilt_y_deg* rotate the focal plane around the
    corresponding sensor axis, allowing depth-of-field to follow oblique
    planes (architectural photography, macro, etc.).

    Custom warp
    -----------
    Supply a callable ``fn(u, v) -> (dx, dy, dz)`` to displace each sample
    point arbitrarily (u, v in [0, 1] sensor coords).
    """

    FLAT        = "flat"
    CYLINDRICAL = "cylindrical"
    SPHERICAL   = "spherical"

    def __init__(self,
                 preset: str = "flat",
                 *,
                 curvature_r: float = 1.0,
                 tilt_x_deg: float = 0.0,
                 tilt_y_deg: float = 0.0,
                 warp: Optional[Callable[[float, float],
                                         Tuple[float, float, float]]] = None):
        self.preset      = preset
        self.curvature_r = float(curvature_r)
        self.tilt_x_deg  = float(tilt_x_deg)
        self.tilt_y_deg  = float(tilt_y_deg)
        self._warp_fn    = warp

    def sample(self, u: float, v: float) -> Tuple[float, float, float]:
        """Return sensor-local offset (dx, dy, dz) for normalised coords (u, v)."""
        u2, v2 = u * 2.0 - 1.0, v * 2.0 - 1.0  # [-1, 1]
        if self.preset == self.FLAT:
            x, y, z = u2, v2, 0.0
        elif self.preset == self.CYLINDRICAL:
            r = self.curvature_r
            angle = u2 * (math.pi / 4.0)
            x = r * math.sin(angle)
            y = v2
            z = r * (math.cos(angle) - 1.0)
        elif self.preset == self.SPHERICAL:
            r = self.curvature_r
            az = u2 * (math.pi / 4.0)
            el = v2 * (math.pi / 4.0)
            x  = r * math.sin(az) * math.cos(el)
            y  = r * math.sin(el)
            z  = r * (math.cos(az) * math.cos(el) - 1.0)
        else:
            x, y, z = u2, v2, 0.0

        # Scheimpflug tilt
        if self.tilt_x_deg or self.tilt_y_deg:
            tx = math.radians(self.tilt_x_deg)
            ty = math.radians(self.tilt_y_deg)
            # Tilt around X axis (affects z vs v)
            y2 = y * math.cos(tx) - z * math.sin(tx)
            z  = y * math.sin(tx) + z * math.cos(tx)
            y  = y2
            # Tilt around Y axis (affects z vs u)
            x2 = x * math.cos(ty) + z * math.sin(ty)
            z  = -x * math.sin(ty) + z * math.cos(ty)
            x  = x2

        if self._warp_fn is not None:
            wx, wy, wz = self._warp_fn(u, v)
            x += wx; y += wy; z += wz

        return (x, y, z)

    def sample_batch(self, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """Vectorised batch evaluation: (N,) u, v → (N, 3) float64 positions.

        Identical in meaning to calling ``sample(u, v)`` for each element
        but implemented entirely in NumPy — no Python loop.  Use this for
        all back-casting pipelines to avoid serial overhead.
        """
        import numpy as _np
        us = _np.asarray(us, _np.float64).ravel()
        vs = _np.asarray(vs, _np.float64).ravel()
        u2 = us * 2.0 - 1.0   # map [0,1] → [-1,1]
        v2 = vs * 2.0 - 1.0
        N  = len(us)
        if self.preset == self.FLAT:
            x = u2.copy()
            y = v2.copy()
            z = _np.zeros(N, _np.float64)
        elif self.preset == self.CYLINDRICAL:
            r     = self.curvature_r
            angle = u2 * (math.pi / 4.0)
            x = r * _np.sin(angle)
            y = v2.copy()
            z = r * (_np.cos(angle) - 1.0)
        elif self.preset == self.SPHERICAL:
            r  = self.curvature_r
            az = u2 * (math.pi / 4.0)
            el = v2 * (math.pi / 4.0)
            x  = r * _np.sin(az) * _np.cos(el)
            y  = r * _np.sin(el)
            z  = r * (_np.cos(az) * _np.cos(el) - 1.0)
        else:
            x = u2.copy()
            y = v2.copy()
            z = _np.zeros(N, _np.float64)

        # Scheimpflug tilt — same rotation order as scalar sample()
        if self.tilt_x_deg or self.tilt_y_deg:
            tx = math.radians(self.tilt_x_deg)
            ty = math.radians(self.tilt_y_deg)
            y2 =  y * math.cos(tx) - z * math.sin(tx)
            z  =  y * math.sin(tx) + z * math.cos(tx)
            y  = y2
            x2 =  x * math.cos(ty) + z * math.sin(ty)
            z  = -x * math.sin(ty) + z * math.cos(ty)
            x  = x2

        if self._warp_fn is not None:
            try:
                wx, wy, wz = self._warp_fn(us, vs)
                x = x + _np.asarray(wx, _np.float64)
                y = y + _np.asarray(wy, _np.float64)
                z = z + _np.asarray(wz, _np.float64)
            except (TypeError, ValueError):
                # Scalar warp function — fall back to map (avoids total failure)
                offsets = _np.array([self._warp_fn(float(u), float(v))
                                     for u, v in zip(us, vs)], _np.float64)
                x = x + offsets[:, 0]
                y = y + offsets[:, 1]
                z = z + offsets[:, 2]

        return _np.stack([x, y, z], axis=1)

    def set_warp(self,
                 fn: Callable[[float, float], Tuple[float, float, float]]) -> None:
        """Attach a custom warp function ``fn(u, v) -> (dx, dy, dz)``."""
        self._warp_fn = fn


# ─────────────────────────────────────────────────────────────────────────────
# DepthSensor
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SpectralChannel
# ─────────────────────────────────────────────────────────────────────────────

class SpectralChannel:
    """One colour channel of a CFA sensor with grounded spectral parameters.

    All wavelengths in nanometres.  ``qe_at(nm)`` returns the normalised
    quantum efficiency at a given wavelength modelled as a Gaussian.

    Attributes
    ----------
    name : str
        Short label: ``'R'``, ``'G'``, ``'G1'``, ``'G2'``, ``'B'``, ``'Y'``.
    peak_nm : float
        Peak wavelength of the channel's spectral response (nm).
    bandwidth_nm : float
        Full-width at half-maximum of the Gaussian QE curve (nm).
    area_fraction : float
        Fraction of sensor pixels assigned to this channel (0–1).
        Standard Bayer RGGB: R=0.25, G1=0.25, G2=0.25, B=0.25.
    sensitivity_peak : float
        Normalised peak QE (0–1).  Green usually set to 1.0 as reference.
    """

    def __init__(self,
                 name: str,
                 peak_nm: float,
                 bandwidth_nm: float,
                 area_fraction: float,
                 sensitivity_peak: float = 1.0):
        self.name             = name
        self.peak_nm          = float(peak_nm)
        self.bandwidth_nm     = float(bandwidth_nm)
        self.area_fraction    = float(area_fraction)
        self.sensitivity_peak = float(sensitivity_peak)
        # σ from FWHM: FWHM = 2√(2 ln2) σ ≈ 2.355 σ
        self._sigma_nm        = float(bandwidth_nm) / 2.3548

    def qe_at(self, nm: float) -> float:
        """Return normalised QE [0, 1] at wavelength *nm*."""
        d = (float(nm) - self.peak_nm) / max(self._sigma_nm, 1e-6)
        return self.sensitivity_peak * math.exp(-0.5 * d * d)

    @property
    def display_weight(self) -> float:
        """Area-weighted peak QE — proportional channel contribution to luminance."""
        return self.area_fraction * self.sensitivity_peak

    def __repr__(self) -> str:
        return (f"SpectralChannel({self.name!r}, peak={self.peak_nm}nm, "
                f"bw={self.bandwidth_nm}nm, area={self.area_fraction:.3f}, "
                f"QE_peak={self.sensitivity_peak:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# CFAPattern
# ─────────────────────────────────────────────────────────────────────────────

class CFAPattern:
    """Color Filter Array pattern: ordered list of SpectralChannels + layout.

    Built-in factory methods produce real-sensor patterns with accurate
    spectral parameters sourced from published CMOS sensor datasheets
    (Sony IMX-class 2×2 Bayer; Fujifilm X-Trans 6×6).

    Parameters
    ----------
    channels : list[SpectralChannel]
        All channels, including multiple green entries when applicable.
    layout : str
        Pattern label: ``'RGGB'``, ``'GRBG'``, ``'MONO'``, ``'XTRANS'``.
    """

    def __init__(self,
                 channels: List[SpectralChannel],
                 layout: str = "RGGB"):
        self.channels = list(channels)
        self.layout   = layout

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def BAYER_RGGB(cls) -> "CFAPattern":
        """Standard Bayer RGGB — Sony IMX-class spectral parameters.

        Channel  |  Peak   | FWHM   | Area  | QE_peak
        ---------|---------|--------|-------|--------
        R        | 600 nm  | 120 nm | 0.25  | 0.75
        G1       | 535 nm  |  90 nm | 0.25  | 1.00
        G2       | 545 nm  |  90 nm | 0.25  | 0.95
        B        | 455 nm  |  80 nm | 0.25  | 0.55

        Two green channels reflect the unequal G1/G2 pixel positions in the
        2×2 block.  Total green area = 0.50, matching the 2:1:1 mosaic ratio
        that mirrors the human eye's luminance sensitivity dominance.
        """
        return cls([
            SpectralChannel('R',  peak_nm=600.0, bandwidth_nm=120.0, area_fraction=0.25, sensitivity_peak=0.75),
            SpectralChannel('G1', peak_nm=535.0, bandwidth_nm= 90.0, area_fraction=0.25, sensitivity_peak=1.00),
            SpectralChannel('G2', peak_nm=545.0, bandwidth_nm= 90.0, area_fraction=0.25, sensitivity_peak=0.95),
            SpectralChannel('B',  peak_nm=455.0, bandwidth_nm= 80.0, area_fraction=0.25, sensitivity_peak=0.55),
        ], layout='RGGB')

    @classmethod
    def X_TRANS(cls) -> "CFAPattern":
        """Fujifilm X-Trans 6×6 pattern — 6 R, 6 B, 24 G pixels per 36-pixel block.

        Area fractions: R ≈ 0.167, G ≈ 0.667, B ≈ 0.167.
        Green dominance (66.7%) increases luminance resolution vs Bayer.
        Spectral params match Fujifilm EXR-series published datasheets.
        """
        return cls([
            SpectralChannel('R', peak_nm=605.0, bandwidth_nm=115.0, area_fraction=0.167, sensitivity_peak=0.72),
            SpectralChannel('G', peak_nm=538.0, bandwidth_nm= 88.0, area_fraction=0.667, sensitivity_peak=1.00),
            SpectralChannel('B', peak_nm=458.0, bandwidth_nm= 78.0, area_fraction=0.167, sensitivity_peak=0.52),
        ], layout='XTRANS')

    @classmethod
    def MONO(cls) -> "CFAPattern":
        """Monochrome sensor — single panchromatic channel (no CFA).

        Peak ~555 nm — maximum photopic luminous efficiency.
        """
        return cls([
            SpectralChannel('Y', peak_nm=555.0, bandwidth_nm=200.0,
                            area_fraction=1.0, sensitivity_peak=1.0),
        ], layout='MONO')

    # ── Spectral reconstruction ────────────────────────────────────────────────

    def _channel_rgb_weights(self) -> Tuple[float, float, float]:
        """Raw (R, G, B) display weights summed from all channels.

        Channels named 'R*' contribute to red, 'G*' to green, 'B*' to blue.
        Panchromatic/NIR channels are split equally across all three primaries.
        Weight per channel = area_fraction × sensitivity_peak.
        """
        r_w = g_w = b_w = 0.0
        for ch in self.channels:
            w = ch.display_weight
            n = ch.name.upper()
            if n.startswith('R'):
                r_w += w
            elif n.startswith('B'):
                b_w += w
            elif n.startswith('G'):
                g_w += w
            else:
                r_w += w / 3.0; g_w += w / 3.0; b_w += w / 3.0
        return (r_w, g_w, b_w)

    def rgb_matrix(self) -> np.ndarray:
        """Return normalised (R, G, B) display weights as a float64 ndarray shape (3,).

        Normalised so the maximum channel == 1.0.  Use as a colour-balance
        multiplier: ``display_rgb = lum * rgb_matrix()``.  For a spectrally
        flat (neutral grey) scene this reconstructs a perceptually correct
        white-balanced output.
        """
        r, g, b = self._channel_rgb_weights()
        peak = max(r, g, b, 1e-12)
        return np.array([r, g, b], dtype=np.float64) / peak

    def encode_lum(self, lum: float) -> Tuple[float, float, float]:
        """Convert scalar luminance to (R, G, B) using CFA spectral weights.

        Produces a plausible per-channel value for a sensor sampling a
        spectrally flat scene under the sensor's native spectral sensitivity.
        """
        w = self.rgb_matrix()
        return (float(lum * w[0]), float(lum * w[1]), float(lum * w[2]))

    def __repr__(self) -> str:
        return f"CFAPattern(layout={self.layout!r}, channels={len(self.channels)})"


# ─────────────────────────────────────────────────────────────────────────────
# DigitalSensor
# ─────────────────────────────────────────────────────────────────────────────

class DigitalSensor:
    """Configuration for digital sensor readout and display rectification.

    ``DigitalSensor`` is the digital-display complement to ``DepthSensor``.
    When ``positive`` is True the display pipeline outputs a colour-correct,
    right-way-up positive image regardless of whether the underlying
    accumulator is a film negative or has an aperture optical inversion
    (180° flip).

    Parameters
    ----------
    enabled : bool
        When False the camera behaves as a classical chemical emulsion
        (negative / enlarger model).  When True the digital readout path
        is active and ``positive`` / ``auto_orient`` take effect.
    positive : bool
        Output as a colour positive.  Negates tone inversion so bright
        scene pixels appear bright on screen.
    auto_orient : bool
        Automatically apply the full 180° UV correction to undo the
        aperture optical inversion.  Ignored if *positive* is False.
    cfa : CFAPattern | None
        Colour filter array model.  ``None`` defaults to Bayer RGGB.
    color_space : str
        Target display colour space tag: ``'sRGB'``, ``'linear_sRGB'``,
        ``'display_p3'``, ``'acescg'``.
    """

    def __init__(self,
                 enabled: bool = True,
                 *,
                 positive: bool = True,
                 auto_orient: bool = True,
                 cfa: Optional["CFAPattern"] = None,
                 color_space: str = 'sRGB'):
        self.enabled     = bool(enabled)
        self.positive    = bool(positive)
        self.auto_orient = bool(auto_orient)
        self.color_space = color_space
        self._rgb_weights: Optional[Tuple[float, float, float]] = None
        # Use property setter so None → default BAYER_RGGB
        self.cfa = cfa  # type: ignore[assignment]

    @property
    def cfa(self) -> "CFAPattern":
        return self._cfa

    @cfa.setter
    def cfa(self, value: Optional["CFAPattern"]) -> None:
        self._cfa = value if value is not None else CFAPattern.BAYER_RGGB()
        self._rgb_weights = None  # invalidate cached weights

    @property
    def rgb_weights(self) -> Tuple[float, float, float]:
        """Cached (R, G, B) display weights derived from the CFA pattern.

        Computed lazily from ``cfa.rgb_matrix()``.  Updated each tick by
        ``DigitalPositiveSensor`` when illuminant-weighted computation runs.
        """
        if self._rgb_weights is None:
            w = self._cfa.rgb_matrix()
            self._rgb_weights = (float(w[0]), float(w[1]), float(w[2]))
        return self._rgb_weights

    def encode_rgb(self, lum: float) -> Tuple[float, float, float]:
        """Encode scalar luminance to (R, G, B) using CFA spectral weights."""
        return self._cfa.encode_lum(lum)

    def __repr__(self) -> str:
        return (f"DigitalSensor(enabled={self.enabled}, positive={self.positive}, "
                f"auto_orient={self.auto_orient}, cfa={self._cfa!r})")


# ─────────────────────────────────────────────────────────────────────────────
# DepthSensor
# ─────────────────────────────────────────────────────────────────────────────

# (DigitalPositiveSensor is defined after CameraSoftware — see bottom of file)


class DepthSensor:
    """Configuration for depth-texture capture.

    When ``enabled`` is True the pipeline renders an additional depth pass
    and the result is available as ``ctx.depth_map`` (numpy array, shape
    [H, W], dtype float32) on the FOLLOWING tick.

    Depth format
    ------------
    ``'linear'``   Raw eye-space distance in scene units.
    ``'log'``      log2(depth) — compresses large ranges.
    ``'inverse'``  1/depth — useful for disparity-space auto-focus.
    ``'ndc'``      Raw NDC z [0, 1] from the depth buffer.

    Precision
    ---------
    ``'f16'``  Half float.  Faster, lower precision.
    ``'f32'``  Full float.  Slower, full precision.
    """

    def __init__(self,
                 enabled: bool = False,
                 near: float = 0.05,
                 far: float = 50.0,
                 format: str = 'linear',
                 precision: str = 'f32'):
        self.enabled   = bool(enabled)
        self.near      = float(near)
        self.far       = float(far)
        self.format    = format
        self.precision = precision


# ─────────────────────────────────────────────────────────────────────────────
# LensTransform — compiled geometric lens transform
# ─────────────────────────────────────────────────────────────────────────────

class LensTransform:
    """Compiled geometric lens transform: shift, Scheimpflug tilt, barrel
    extension, and gimbal (lens-axis rotation in camera space).

    Parameters
    ----------
    shift : (x, y) ndarray, normalised sensor units
        Lateral displacement of the principal point (tilt-shift lens
        translation). ±1.0 = ±one sensor half-width / half-height.
    lens_tilt : (x, y) ndarray, radians
        Scheimpflug tilt angles.  lens_tilt[0] rotates the focus plane
        around the camera-right axis (nod up/down); lens_tilt[1] rotates
        around the camera-up axis (pan left/right).  These modulate the
        per-ray effective focus distance in the shader.
    extension_mm : float
        Physical barrel extension in millimetres added to the nominal
        focal length. Increases effective focal length and magnification;
        the shader sees a proportionally smaller fov_tan.
    gimbal : (pan, tilt) ndarray, radians
        Rotation of the entire lens+aperture assembly in camera space.
        gimbal[0] (pan) rotates around the camera-up axis;
        gimbal[1] (tilt) rotates around the camera-right axis.
        Applied to (right, up, fwd) before dispatch so the aperture and
        ray cone point in the gimballed direction.

    Usage
    -----
    Build a temporary instance from camera fields then call compile() to
    get the shader-ready payload::

        lt = LensTransform()
        lt.shift[:]     = cam.tilt_shift          # principal-point shift
        lt.lens_tilt[:] = cam.lens_tilt           # Scheimpflug
        lt.extension_mm = cam.extension_mm        # barrel pull
        lt.gimbal[:]    = cam.gimbal              # axis angle
        payload = lt.compile(cam.focal_mm, cam.focus_m,
                             cam.sensor.height_mm,
                             right, up, fwd)
        # payload keys: right, up, fwd, fov_tan, tilt_shift, lens_tilt,
        #               effective_focal_mm
    """

    __slots__ = ('shift', 'lens_tilt', 'extension_mm', 'gimbal')

    def __init__(self):
        import numpy as _np
        self.shift        = _np.zeros(2, _np.float64)
        self.lens_tilt    = _np.zeros(2, _np.float64)
        self.extension_mm = 0.0
        self.gimbal       = _np.zeros(2, _np.float64)

    def compile(self,
                focal_mm:     float,
                focus_m:      float,
                sensor_h_mm:  float,
                right,
                up,
                fwd) -> dict:
        """Return a shader-ready payload dict.

        The returned dict contains:
          ``right``, ``up``, ``fwd``       — float32 unit vectors (gimbal applied)
          ``fov_tan``                      — tan(fov_y/2) (extension applied)
          ``tilt_shift``                   — (x, y) float tuple
          ``lens_tilt``                    — (x, y) float tuple (Scheimpflug radians)
          ``effective_focal_mm``           — focal_mm + extension_mm
        """
        import math as _math
        import numpy as _np

        # ── 1. Effective focal length ─────────────────────────────────────────
        eff_f = max(1.0, float(focal_mm) + float(self.extension_mm))
        h_mm  = max(1.0, float(sensor_h_mm))
        fov_tan = _math.tan(_math.atan(h_mm * 0.5 / eff_f))

        # ── 2. Gimbal rotation ────────────────────────────────────────────────
        # Rodrigues utility: rotate v around unit axis n by angle θ
        def _rot(v, n, theta):
            if abs(theta) < 1e-10:
                return v.copy()
            c, s = _math.cos(theta), _math.sin(theta)
            n = n / max(1e-12, float(_np.linalg.norm(n)))
            return v * c + _np.cross(n, v) * s + n * _np.dot(n, v) * (1.0 - c)

        r = _np.asarray(right, _np.float64).copy()
        u = _np.asarray(up,    _np.float64).copy()
        f = _np.asarray(fwd,   _np.float64).copy()

        pan  = float(self.gimbal[0])   # around up axis
        tilt = float(self.gimbal[1])   # around right axis

        if abs(pan) > 1e-10:
            f = _rot(f, u, pan)
            r = _rot(r, u, pan)
            f /= max(1e-12, float(_np.linalg.norm(f)))
            r /= max(1e-12, float(_np.linalg.norm(r)))
            u  = _np.cross(r, f)  # keep orthonormal
            u /= max(1e-12, float(_np.linalg.norm(u)))

        if abs(tilt) > 1e-10:
            f = _rot(f, r, tilt)
            u = _rot(u, r, tilt)
            f /= max(1e-12, float(_np.linalg.norm(f)))
            u /= max(1e-12, float(_np.linalg.norm(u)))
            r  = _np.cross(f, u)  # keep orthonormal
            r /= max(1e-12, float(_np.linalg.norm(r)))

        return {
            'right':              r.astype(_np.float32),
            'up':                 u.astype(_np.float32),
            'fwd':                f.astype(_np.float32),
            'fov_tan':            float(fov_tan),
            'tilt_shift':         (float(self.shift[0]),     float(self.shift[1])),
            'lens_tilt':          (float(self.lens_tilt[0]), float(self.lens_tilt[1])),
            'effective_focal_mm': float(eff_f),
        }

    def reset(self) -> None:
        self.shift[:]     = 0.0
        self.lens_tilt[:] = 0.0
        self.extension_mm = 0.0
        self.gimbal[:]    = 0.0

    def copy(self) -> 'LensTransform':
        lt = LensTransform()
        lt.shift[:]     = self.shift
        lt.lens_tilt[:] = self.lens_tilt
        lt.extension_mm = self.extension_mm
        lt.gimbal[:]    = self.gimbal
        return lt


# ─────────────────────────────────────────────────────────────────────────────
# CameraContext
# ─────────────────────────────────────────────────────────────────────────────

class CameraContext:
    """Rich proxy provided to CameraSoftware.tick().

    Wraps the owning ``CameraItem`` and (optionally) the underlying
    ``Camera`` so that software modules have a unified, flat interface to
    every camera parameter.  Write directly to properties; the context
    propagates changes to the right underlying object.

    Diagnostic / read-only surfaces
    --------------------------------
    These are populated by the renderer after the previous frame and are
    read-only from software.  ``None`` means not-yet-available.

    ``depth_map``
        numpy float32 [H, W] — depth pass result (if depth sensor enabled).
    ``histogram``
        numpy float32 [256] — normalised luminance histogram of last frame.
    ``last_ev``
        float — measured exposure value (log2 of mean luminance).
    ``focus_score``
        float — normalised sharpness metric [0, 1] from last frame.
    """

    def __init__(self, camera_item: "CameraItem", camera=None):
        object.__setattr__(self, '_item', camera_item)
        object.__setattr__(self, '_cam',  camera)
        # Diagnostic surfaces populated by the renderer each frame:
        object.__setattr__(self, 'depth_map',    None)  # np.ndarray | None
        object.__setattr__(self, 'histogram',    None)  # np.ndarray | None
        object.__setattr__(self, 'last_ev',      None)  # float | None
        object.__setattr__(self, 'focus_score',  None)  # float | None
        # Sub-system configs (per-context, owned by the context):
        object.__setattr__(self, 'sensor_surface',  ParametricSurface(ParametricSurface.FLAT))
        object.__setattr__(self, 'film_surface',    ParametricSurface(ParametricSurface.FLAT))
        object.__setattr__(self, 'plate_surface',   ParametricSurface(ParametricSurface.FLAT))
        object.__setattr__(self, 'exposure_dist',   ExposureDistribution())
        object.__setattr__(self, 'depth_sensor',    DepthSensor())
        # Digital display sensor — default: positive, auto-oriented, Bayer RGGB
        object.__setattr__(self, 'digital_sensor',  DigitalSensor(enabled=True, positive=True, auto_orient=True))
        # Active film back (FilmStack) chosen by camera software for this tick.
        # None means the renderer keeps using its own self.film.
        object.__setattr__(self, 'digital_back',    None)

    # ── Optics ────────────────────────────────────────────────────────────────

    @property
    def focal_mm(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(cam.focal_mm) if cam is not None else float(object.__getattribute__(self, '_item').focal_mm)

    @focal_mm.setter
    def focal_mm(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.set_lens(focal_mm=float(v))
        else: object.__getattribute__(self, '_item').focal_mm = float(v)

    @property
    def focus_m(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'focus_dist', 1.6)) if cam else 1.6

    @focus_m.setter
    def focus_m(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.focus_dist = float(v)

    @property
    def aperture(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'aperture', 0.0)) if cam else 0.0

    @aperture.setter
    def aperture(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.aperture = float(v)

    @property
    def n_blades(self) -> int:
        """Aperture blade count: 0 = circle, >=3 = regular N-gon."""
        cam = object.__getattribute__(self, '_cam')
        return int(getattr(cam, 'n_blades', 0)) if cam else 0

    @n_blades.setter
    def n_blades(self, v: int) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.n_blades = max(0, int(v))

    @property
    def aperture_rot(self) -> float:
        """First blade edge angle in radians."""
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'aperture_rot', 0.0)) if cam else 0.0

    @aperture_rot.setter
    def aperture_rot(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.aperture_rot = float(v)

    @property
    def tilt_x(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'tilt_shift', [0.0, 0.0])[0]) if cam else 0.0

    @tilt_x.setter
    def tilt_x(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.tilt_shift[0] = float(v)

    @property
    def tilt_y(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'tilt_shift', [0.0, 0.0])[1]) if cam else 0.0

    @tilt_y.setter
    def tilt_y(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.tilt_shift[1] = float(v)

    @property
    def lens_tilt_x(self) -> float:
        """Scheimpflug tilt around camera-right axis (radians, nod up/down)."""
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'lens_tilt', [0.0, 0.0])[0]) if cam else 0.0

    @lens_tilt_x.setter
    def lens_tilt_x(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.lens_tilt[0] = float(v)

    @property
    def lens_tilt_y(self) -> float:
        """Scheimpflug tilt around camera-up axis (radians, pan left/right)."""
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'lens_tilt', [0.0, 0.0])[1]) if cam else 0.0

    @lens_tilt_y.setter
    def lens_tilt_y(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.lens_tilt[1] = float(v)

    @property
    def extension_mm(self) -> float:
        """Barrel extension in mm added to focal length."""
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'extension_mm', 0.0)) if cam else 0.0

    @extension_mm.setter
    def extension_mm(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.extension_mm = float(v)

    @property
    def gimbal_pan(self) -> float:
        """Lens-axis pan (rotation around up, radians)."""
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'gimbal', [0.0, 0.0])[0]) if cam else 0.0

    @gimbal_pan.setter
    def gimbal_pan(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.gimbal[0] = float(v)

    @property
    def gimbal_tilt(self) -> float:
        """Lens-axis tilt (rotation around right, radians)."""
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'gimbal', [0.0, 0.0])[1]) if cam else 0.0

    @gimbal_tilt.setter
    def gimbal_tilt(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: cam.gimbal[1] = float(v)

    @property
    def ca(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'ca', 0.0)) if cam else 0.0

    @ca.setter
    def ca(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: setattr(cam, 'ca', float(v))

    # ── Sensor / exposure ─────────────────────────────────────────────────────

    @property
    def sensor_iso(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'iso', 1.4)) if cam else 1.4

    @sensor_iso.setter
    def sensor_iso(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: setattr(cam, 'iso', float(v))

    @property
    def sensor_gain(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'sensor_gain', 8.0)) if cam else 8.0

    @sensor_gain.setter
    def sensor_gain(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: setattr(cam, 'sensor_gain', float(v))

    @property
    def decay(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'decay', 0.0)) if cam else 0.0

    @decay.setter
    def decay(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: setattr(cam, 'decay', float(v))

    # ── Ray-camera config ─────────────────────────────────────────────────────

    @property
    def ray_density(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'ray_density', 1.0)) if cam else 1.0

    @ray_density.setter
    def ray_density(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: setattr(cam, 'ray_density', float(v))

    @property
    def ray_exposure(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'ray_exposure', 1.0)) if cam else 1.0

    @ray_exposure.setter
    def ray_exposure(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: setattr(cam, 'ray_exposure', float(v))

    @property
    def ray_gamma(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'ray_gamma', 0.45)) if cam else 0.45

    @ray_gamma.setter
    def ray_gamma(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: setattr(cam, 'ray_gamma', float(v))

    @property
    def sensor_rate(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'sensor_rate', 16.0)) if cam else 16.0

    @sensor_rate.setter
    def sensor_rate(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: setattr(cam, 'sensor_rate', float(v))

    @property
    def sensor_spp(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'sensor_spp', 1.0)) if cam else 1.0

    @sensor_spp.setter
    def sensor_spp(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: setattr(cam, 'sensor_spp', float(v))

    @property
    def sensor_fps(self) -> float:
        cam = object.__getattribute__(self, '_cam')
        return float(getattr(cam, 'sensor_fps', 0.0)) if cam else 0.0

    @sensor_fps.setter
    def sensor_fps(self, v: float) -> None:
        cam = object.__getattribute__(self, '_cam')
        if cam is not None: setattr(cam, 'sensor_fps', float(v))

    # ── Camera mount ──────────────────────────────────────────────────────────

    @property
    def pan_deg(self) -> float:
        return float(object.__getattribute__(self, '_item').pan_deg)

    @pan_deg.setter
    def pan_deg(self, v: float) -> None:
        object.__getattribute__(self, '_item').pan_deg = float(v)

    @property
    def tilt_deg(self) -> float:
        return float(object.__getattribute__(self, '_item').tilt_deg)

    @tilt_deg.setter
    def tilt_deg(self, v: float) -> None:
        object.__getattribute__(self, '_item').tilt_deg = float(v)

    # ── Raw access ────────────────────────────────────────────────────────────

    @property
    def item(self) -> "CameraItem":
        """Direct access to the underlying CameraItem."""
        return object.__getattribute__(self, '_item')

    @property
    def camera(self):
        """Direct access to the GL Camera object, or None."""
        return object.__getattribute__(self, '_cam')


# ─────────────────────────────────────────────────────────────────────────────
# CameraSoftware
# ─────────────────────────────────────────────────────────────────────────────

class CameraSoftware:
    """Base class for camera software modules.

    Subclass this, override ``tick()`` (and optionally ``on_attach`` /
    ``on_detach``), then attach to a camera::

        camera_item.software.append(MySoftware())

    The host calls ``CameraItem.tick(dt)`` once per frame, which builds a
    ``CameraContext`` and calls each software module's ``tick(dt, ctx)``.

    Film participation protocol
    ---------------------------
    Any software module — digital, chemical, acoustic, or otherwise — can
    contribute spectral layers to the renderer's 8-slot accumulator batch by
    exposing one of the following attributes (checked in priority order):

    * ``effective_layer_specs() -> list``
        Multi-back modules return a pre-flattened list of ≤ 8 spec dicts.
    * ``active_back`` (object with ``.layer_specs() -> list``)
        Single active back (e.g. a FilmStack).
    * ``film`` (object with ``.layer_specs() -> list``)
        Module with a bound film stack.
    * ``layer_specs() -> list``
        Module that is itself a film-like object.

    All layer specs share the same dict schema as ``FilmStack.layer_specs()``:
    ``dark_rgb``, ``light_rgb``, ``shadow_point``, ``highlight_point``,
    ``gain``, ``center_hz``, ``width_oct``.

    Every software module is a peer — chemical film, digital sensor, acoustic
    holography, etc.  If their layer specs are the same dimensionality they
    run in the same accumulator batch simultaneously.

    Attributes
    ----------
    name : str
        Short identifier used in logs and YAML configs.
    """

    name: str = "base"

    def on_attach(self, ctx: CameraContext) -> None:
        """Called once when the software is first connected to a camera.

        Use this to synchronise motor drives to the camera's current state::

            self._zoom_motor.sync(ctx.focal_mm)
        """

    def on_detach(self, ctx: CameraContext) -> None:
        """Called when the software is removed from the camera."""

    def tick(self, dt: float, ctx: CameraContext) -> None:
        """Called once per frame grant.

        Parameters
        ----------
        dt :
            Elapsed time in seconds since the last tick.
        ctx :
            Live camera proxy.  Read parameters, write parameters,
            inspect diagnostics, configure sub-systems.
        """

    @staticmethod
    def compute_wb_gains(
        cfa: "CFAPattern",
        illuminant_fn: "Callable[[float], float]",
    ) -> "Tuple[float, float, float]":
        """Compute normalised (R, G, B) white-balance gain vector.

        For each channel in *cfa* the illuminant SPD is sampled at the
        channel's peak wavelength, weighted by QE peak and area fraction,
        and accumulated into R / G / B primaries by channel-name prefix.
        The vector is then normalised so the largest primary equals 1.0.

        Any software module that participates in the universal film-batch
        system can call this to get a display-ready white-balance multiplier
        for the ``uDigitalRGB`` shader uniform.

        Parameters
        ----------
        cfa :
            A ``CFAPattern`` (or any object with a ``.channels`` iterable
            of ``SpectralChannel``-like objects).
        illuminant_fn :
            Callable ``(nm: float) -> float`` returning the illuminant
            spectral power at the requested wavelength in nanometres.
            Use ``_d65_at``, ``_planck_at``, or ``_illuminant_at`` as
            convenience wrappers.

        Returns
        -------
        (r_gain, g_gain, b_gain) : tuple of float
            Normalised gain vector; largest component is always 1.0.
        """
        r = g = b = 0.0
        for ch in cfa.channels:
            response = ch.sensitivity_peak * illuminant_fn(ch.peak_nm) * ch.area_fraction
            n = ch.name.upper()
            if n.startswith('R'):
                r += response
            elif n.startswith('B'):
                b += response
            elif n.startswith('G'):
                g += response
            else:
                r += response / 3.0
                g += response / 3.0
                b += response / 3.0
        peak = max(r, g, b, 1e-12)
        return (r / peak, g / peak, b / peak)


# ─────────────────────────────────────────────────────────────────────────────
# CIE D65 illuminant SPD (380–720 nm in 10 nm steps, norm. to 100 at 560 nm)
# Source: CIE 15:2004 Table 1.  Used by DigitalPositiveSensor for white balance.
# ─────────────────────────────────────────────────────────────────────────────
_D65_SPD_STEP  = 10
_D65_SPD_START = 380
_D65_SPD_VALS  = [
     49.98,  52.31,  54.65,  68.70,  82.75,  87.12,  91.49,  92.46,  93.43,
     90.06,  86.68,  95.77, 104.86, 110.94, 117.01, 117.41, 117.81, 116.34,
    114.86, 115.39, 115.92, 112.37, 108.81, 109.08, 109.35, 108.58, 107.80,
    106.30, 104.79, 106.24, 107.69, 106.05, 104.41, 104.23, 104.05,
]


def _d65_at(nm: float) -> float:
    """Linearly interpolate the D65 spectral power distribution at *nm* (nm)."""
    nm = float(nm)
    lo = int(nm // _D65_SPD_STEP) * _D65_SPD_STEP
    hi = lo + _D65_SPD_STEP
    idx_lo = (lo - _D65_SPD_START) // _D65_SPD_STEP
    idx_hi = idx_lo + 1
    if idx_lo < 0 or idx_hi >= len(_D65_SPD_VALS):
        return 0.0
    t = (nm - lo) / _D65_SPD_STEP
    return _D65_SPD_VALS[idx_lo] * (1.0 - t) + _D65_SPD_VALS[idx_hi] * t


# ─────────────────────────────────────────────────────────────────────────────
# Planckian (blackbody) SPD  —  used for manual / auto / CIE-A white balance
# ─────────────────────────────────────────────────────────────────────────────

def _planck_at(nm: float, T_K: float) -> float:
    """Relative Planckian spectral radiance at *nm* (nm) and temperature *T_K* (K).

    Normalised to 100.0 at 560 nm so values are on the same scale as the D65
    table, making mixed-illuminant arithmetic straightforward.
    """
    _H  = 6.626e-34
    _C2 = 2.998e8
    _K  = 1.381e-23
    T   = float(max(T_K, 100.0))

    def _b(lam_m: float) -> float:
        x = (_H * _C2) / (lam_m * _K * T)
        if x > 700.0:
            return 0.0
        return lam_m ** -5 / (math.exp(x) - 1.0)

    ref = _b(560e-9)
    if ref < 1e-300:
        return 0.0
    return 100.0 * _b(float(nm) * 1e-9) / ref


# Standard colour-temperature presets  (mode string → Kelvin).
# 'auto' and 'manual' use the caller-supplied color_temp_K instead.
_WB_MODE_KELVIN: dict = {
    'd65':         6504.0,
    'd50':         5003.0,
    'a':           2856.0,
    'tungsten':    2850.0,
    'fluorescent': 4150.0,
    'shade':       7500.0,
    'cloudy':      6500.0,
    'daylight':    5500.0,
    'flash':       5500.0,
}


def _illuminant_at(nm: float, mode: str, color_temp_K: float) -> float:
    """Sample the illuminant SPD at *nm* (nm).

    mode
    ~~~~
    ``'D65'``                  CIE D65 tabulated SPD (standard daylight).
    ``'D50'``                  Planckian at 5003 K (print/monitor D50).
    ``'A'`` / ``'tungsten'``   Planckian at 2856 K (incandescent).
    ``'fluorescent'``          Planckian at 4150 K (cool-white fluorescent).
    ``'shade'``                Planckian at 7500 K.
    ``'cloudy'``               Planckian at 6500 K.
    ``'daylight'`` / ``'flash'``  Planckian at 5500 K.
    ``'auto'`` / ``'manual'``  Planckian at caller-supplied *color_temp_K*.
    Any other string           Treated as ``'auto'``.
    """
    m = mode.strip().lower()
    if m == 'd65':
        return _d65_at(nm)
    T = _WB_MODE_KELVIN.get(m, float(color_temp_K))
    return _planck_at(nm, T)


# ─────────────────────────────────────────────────────────────────────────────
# DigitalPositiveSensor  (standard CameraSoftware module)
# ─────────────────────────────────────────────────────────────────────────────

class DigitalPositiveSensor(CameraSoftware):
    """Digital positive sensor module with multi-back film management.

    Manages one or more film backs (FilmStack objects) and participates in
    the camera's universal spectral-layer batching system.  On each tick it:

    * Writes the currently active back to ``ctx.digital_back``.
    * Exposes ``effective_layer_specs()`` so the camera's layer collector can
      pull all backs' layers in one call.

    Because ``CameraSoftware`` defines an open film-participation protocol,
    *any* software module — chemical film, acoustic holography, or this digital
    sensor — can run in the same 8-slot accumulator batch simultaneously.
    ``DigitalPositiveSensor`` simply adds back-cycling convenience on top.

    Multiple backs cycle with ``next_back()`` / ``prev_back()`` / ``set_back()``.
    ``effective_layer_specs()`` flattens all registered backs starting from the
    active one, up to the 8-layer GPU limit.

    Parameters
    ----------
    backs : list | None
        Film backs to register (each should be a FilmStack, or any object
        with a ``layer_specs()`` method).  ``None`` means no backs are
        pre-registered; the renderer will keep using its own ``self.film``.
    cfa : CFAPattern | None
        CFA model for the embedded DigitalSensor (spectral-physics metadata).
        ``None`` → Bayer RGGB with Sony IMX-class parameters.
    color_temp_K : float
        Informational colour temperature in Kelvin.  Default 5500 K ≈ D65.
    color_space : str
        Target display colour space tag passed to ``DigitalSensor``.
    encode_budget_frac : float
        Fraction of frame time reserved for spectral encoding.  Default 0.02.
    """

    name: str = "digital_positive"
    _MAX_LAYERS: int = 8  # GPU accumulator hard limit

    def __init__(self,
                 backs=None,
                 cfa: Optional["CFAPattern"] = None,
                 color_temp_K: float = 6504.0,
                 wb_mode: str = 'D65',
                 color_space: str = 'sRGB',
                 encode_budget_frac: float = 0.02,
                 sensor_spec=None):
        """
        Parameters
        ----------
        backs : list | None
            Film backs (FilmStack objects) to register.  None = no backs;
            renderer keeps using its own ``self.film``.
        cfa : CFAPattern | None
            CFA spectral model.  None → Bayer RGGB (Sony IMX-class).
        color_temp_K : float
            Scene colour temperature in Kelvin used for ``'manual'`` and
            ``'auto'`` white balance.  Also the starting estimate for auto
            WB adaptation.  Default 6504 K ≈ CIE D65.
        wb_mode : str
            White-balance mode.  Any of:
            ``'D65'`` (default), ``'D50'``, ``'A'``, ``'tungsten'``,
            ``'fluorescent'``, ``'shade'``, ``'cloudy'``, ``'daylight'``,
            ``'flash'``, ``'manual'``, ``'auto'``.
            ``'auto'`` adapts *color_temp_K* each tick from scene context
            (ctx.scene_color_temp_K) when available, otherwise holds the
            current value until updated externally.
        color_space : str
            Display colour-space tag passed to the inner DigitalSensor.
        encode_budget_frac : float
            Fraction of frame budget reserved for spectral encoding.
        sensor_spec : object | None
            Physical sensor metadata (e.g. SensorSpec from demo_pluck_gl).
            Stored as ``self.sensor_spec``; not used for spectral math but
            available for introspection and future extensions.
        """
        self._ds = DigitalSensor(
            enabled=True,
            positive=True,
            auto_orient=True,
            cfa=cfa,
            color_space=color_space,
        )
        self._color_temp_K       = float(color_temp_K)
        self._wb_mode: str       = str(wb_mode)
        self._encode_budget_frac = float(encode_budget_frac)
        self.sensor_spec         = sensor_spec
        # Film backs: each entry is a FilmStack (or duck-type with layer_specs())
        self._backs: list = list(backs) if backs else []
        self._active_back_idx: int = 0

    # ── Back management ───────────────────────────────────────────────────────

    @property
    def wb_mode(self) -> str:
        """White-balance mode string (e.g. ``'D65'``, ``'auto'``, ``'manual'``)."""
        return self._wb_mode

    @wb_mode.setter
    def wb_mode(self, value: str) -> None:
        self._wb_mode = str(value)

    @property
    def color_temp_K(self) -> float:
        """Scene colour temperature in Kelvin (used for manual/auto WB)."""
        return self._color_temp_K

    @color_temp_K.setter
    def color_temp_K(self, value: float) -> None:
        self._color_temp_K = float(value)

    @property
    def backs(self) -> list:
        return self._backs

    @property
    def active_back(self):
        """Currently selected back, or None if no backs registered."""
        if not self._backs:
            return None
        idx = max(0, min(self._active_back_idx, len(self._backs) - 1))
        return self._backs[idx]

    def add_back(self, back) -> "DigitalPositiveSensor":
        """Append a back and return self (fluent)."""
        self._backs.append(back)
        return self

    def remove_back(self, back) -> None:
        self._backs.remove(back)
        self._active_back_idx = max(0, min(self._active_back_idx, len(self._backs) - 1))

    def set_back(self, idx: int) -> None:
        """Select back by index (wraps around)."""
        if self._backs:
            self._active_back_idx = int(idx) % len(self._backs)

    def next_back(self) -> None:
        """Advance to the next back (wraps around)."""
        if self._backs:
            self._active_back_idx = (self._active_back_idx + 1) % len(self._backs)

    def prev_back(self) -> None:
        """Return to the previous back (wraps around)."""
        if self._backs:
            self._active_back_idx = (self._active_back_idx - 1) % len(self._backs)

    def effective_layer_specs(self) -> list:
        """Flatten all backs' layers into one ordered list of ≤ 8 specs.

        Layers are taken from the active back first, then from subsequent
        backs in registration order, until the GPU limit of 8 is reached.
        The active back always contributes its layers first so its spectral
        channels occupy the first accumulator slots.
        """
        specs: list = []
        if not self._backs:
            return specs
        # Start from active back, then wrap through remaining backs
        n = len(self._backs)
        order = [(self._active_back_idx + i) % n for i in range(n)]
        for idx in order:
            back = self._backs[idx]
            for spec in back.layer_specs():
                if len(specs) >= self._MAX_LAYERS:
                    return specs
                specs.append(spec)
        return specs

    # ── CameraSoftware interface ──────────────────────────────────────────────

    def on_attach(self, ctx: "CameraContext") -> None:
        """Synchronise digital_sensor, digital_back, and pre-compute WB."""
        object.__setattr__(ctx, 'digital_sensor', self._ds)
        object.__setattr__(ctx, 'digital_back',   self.active_back)
        self._run_encoding()

    def tick(self, dt: float, ctx: "CameraContext") -> None:
        """Refresh white-balance weights and publish active back to context."""
        object.__setattr__(ctx, 'digital_sensor', self._ds)
        object.__setattr__(ctx, 'digital_back',   self.active_back)
        # Auto WB: if the context carries a scene colour-temperature estimate,
        # adopt it so the gains track scene illuminant changes each frame.
        if self._wb_mode.lower() == 'auto':
            scene_T = getattr(ctx, 'scene_color_temp_K', None)
            if scene_T is not None:
                self._color_temp_K = float(scene_T)
        self._run_encoding()

    def _run_encoding(self) -> None:
        """Compute illuminant-weighted CFA response → white-balance weights.

        Delegates to ``CameraSoftware.compute_wb_gains()`` with an illuminant
        function selected by ``wb_mode`` and ``color_temp_K``.
        """
        illuminant_fn = lambda nm: _illuminant_at(nm, self._wb_mode, self._color_temp_K)  # noqa: E731
        self._ds._rgb_weights = CameraSoftware.compute_wb_gains(self._ds.cfa, illuminant_fn)


__all__ = [
    "SpeedSpline",
    "MotorDrive",
    "ExposureDistribution",
    "ParametricSurface",
    "SpectralChannel",
    "CFAPattern",
    "DigitalSensor",
    "DigitalPositiveSensor",
    "DepthSensor",
    "CameraContext",
    "CameraSoftware",
]

