"""tf_ray_engine.py

Bidirectional geometric ray propagation in time-frequency (TF) space.

Ray state: (t, f, v_t, v_f, phi, A)

  v_t = +1 or -1   — time direction (+1 forward, -1 backward)
  v_f = df/d_elapsed   (Hz/s)   — physical chirp rate on the arc
                                   NOT normalized; range is unlimited
  |v_t| = 1 always; |v_f| is bounded only by physical constraints.

Within each segment of elapsed arc time dt:
    t_new   = t + v_t * dt
    f_new   = f + v_f * dt
    phi_new = phi + 2pi * v_t * (f*dt + v_f/2 * dt^2)

chirp_rate in real time = df/dt_real = v_t * v_f
  (for backward ray: real-time chirp flips sign relative to arc chirp)

Snell's law at HorizontalSurface (f=const, normal in f-direction):
  angle from f-normal: sin(theta) = |v_t| / sqrt(v_t^2 + v_f^2)
                                   = 1 / sqrt(1 + v_f^2)
  Snell: n1/sqrt(1+v_f1^2) = n2/sqrt(1+v_f2^2)
       => v_f2^2 = (n2/n1)^2 * (1 + v_f1^2) - 1
  TIR when disc < 0 (n2 < n1 and ray close to normal).
  Reflection: v_f -> -v_f, v_t unchanged (frequency bounce).

Snell's law at VerticalSurface (t=const, normal in t-direction):
  angle from t-normal: sin(theta) = |v_f| / sqrt(1 + v_f^2)
  Snell: n1*|v_f1|/sqrt(1+v_f1^2) = n2*|v_f2|/sqrt(1+v_f2^2)
       => solve for v_f2 via sin-chain
  TIR when n1*sin(theta1) >= n2.
  Reflection: v_t -> -v_t (TIME REVERSAL), v_f unchanged.

Fog: Poisson scatter in elapsed-time space.
  At each scatter event: new v_t = random +/-1, new v_f = uniform(-max_chirp, +max_chirp).
  This is true isotropic scattering including complete time reversal.

Corridor boundaries:
  f_min / f_max: frequency mirrors (v_f -> -v_f)
  t = 0 / t = scene_end: temporal mirrors (v_t -> -v_t)

ChirpSegments always store t_start <= t_end (real-time order).
Backward rays are converted on storage so audio+TF rendering are uniform.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from signal_generator_v2 import PI2, LinearChirpPhasePath


# ---------------------------------------------------------------------------
# Ray state
# ---------------------------------------------------------------------------

@dataclass
class RayState2D:
    """
    Arc-length-parameterized ray state.
    v_t = +/-1 (time direction).
    v_f = physical Hz/s chirp rate on the arc (NOT unit-normalized with v_t).
    """
    t:   float   # real time (s)
    f:   float   # instantaneous frequency (Hz)
    v_t: float   # +1 (forward) or -1 (backward in time)
    v_f: float   # df/d_elapsed in Hz/s — unlimited physical scale
    phi: float   # accumulated phase (rad)
    A:   float   # amplitude

    def __post_init__(self) -> None:
        # Enforce v_t = +/-1; v_f is left completely free (physical scale)
        self.v_t = 1.0 if self.v_t >= 0.0 else -1.0

    def advance(self, dt: float) -> "RayState2D":
        """Advance by dt seconds of elapsed arc time (dt > 0)."""
        t_new   = self.t   + self.v_t * dt
        f_new   = self.f   + self.v_f * dt
        # phase integrates 2pi*f(t_real)*dt_real over the arc
        phi_new = self.phi + PI2 * self.v_t * (self.f * dt + 0.5 * self.v_f * dt * dt)
        return RayState2D(t=t_new, f=f_new, v_t=self.v_t, v_f=self.v_f,
                          phi=phi_new, A=self.A)

    @property
    def chirp_rate_hz_per_s(self) -> float:
        """df/dt_real = v_t * v_f  (signed physical chirp rate)."""
        return self.v_t * self.v_f


# ---------------------------------------------------------------------------
# Chirp segment
# ---------------------------------------------------------------------------

@dataclass
class ChirpSegment:
    """
    One straight ray segment stored in real-time order (t_start <= t_end).

    f(t)   = f_start + chirp_rate * (t - t_start)
    phi(t) = phase_at_start + 2pi * [f_start*dt + chirp_rate/2 * dt^2]
             where dt = t - t_start
    """
    t_start:        float
    t_end:          float
    f_start:        float
    chirp_rate:     float   # df/dt_real in Hz/s (can be negative)
    phase_at_start: float
    amplitude:      float


def _append_seg(segs: List[ChirpSegment], s0: RayState2D, s1: RayState2D) -> None:
    """
    Build a ChirpSegment from two consecutive ray states and append it.
    Works for both forward (s0.t < s1.t) and backward (s0.t > s1.t) rays.
    The segment is always stored in real-time order.
    """
    if abs(s1.t - s0.t) < 1e-9:
        return
    cr = s0.chirp_rate_hz_per_s
    if s0.t <= s1.t:
        # forward ray: s0 is the real-time-earlier endpoint
        segs.append(ChirpSegment(
            t_start=s0.t, t_end=s1.t,
            f_start=s0.f, chirp_rate=cr,
            phase_at_start=s0.phi, amplitude=s0.A,
        ))
    else:
        # backward ray: s1 is the real-time-earlier endpoint
        segs.append(ChirpSegment(
            t_start=s1.t, t_end=s0.t,
            f_start=s1.f, chirp_rate=cr,
            phase_at_start=s1.phi, amplitude=s0.A,
        ))


# ---------------------------------------------------------------------------
# TF surfaces
# ---------------------------------------------------------------------------

class TFSurface(ABC):
    """
    Abstract surface in TF space.

    intersect(state) -> elapsed dt to crossing (or None)
    apply(state)     -> list of (new_v_t, new_v_f, amp_scale)

    Both transmitted and reflected children are always returned.
    """

    def __init__(
        self,
        n_a: float = 1.0,
        n_b: float = 1.0,
        transmission_fraction: float = 0.80,
        reflection_fraction:   float = 0.20,
    ) -> None:
        self.n_a    = n_a
        self.n_b    = n_b
        self.t_frac = max(0.0, min(1.0, transmission_fraction))
        self.r_frac = max(0.0, min(1.0, reflection_fraction))

    @abstractmethod
    def intersect(self, state: RayState2D) -> Optional[float]:
        """Return elapsed dt > 0 to the surface, or None."""
        ...

    @abstractmethod
    def _snell(
        self, v_t: float, v_f: float, n1: float, n2: float
    ) -> Tuple[Tuple[float, float], bool]:
        """Return ((new_v_t, new_v_f), was_tir)."""
        ...

    @abstractmethod
    def _reflect(self, v_t: float, v_f: float) -> Tuple[float, float]:
        ...

    @abstractmethod
    def _side_a_to_b(self, state: RayState2D) -> bool:
        ...

    def apply(self, state: RayState2D) -> List[Tuple[float, float, float]]:
        a_to_b = self._side_a_to_b(state)
        n1 = self.n_a if a_to_b else self.n_b
        n2 = self.n_b if a_to_b else self.n_a
        (nvt, nvf), tir = self._snell(state.v_t, state.v_f, n1, n2)
        rvt, rvf = self._reflect(state.v_t, state.v_f)
        children: List[Tuple[float, float, float]] = []
        if tir:
            children.append((rvt, rvf, 1.0))
        else:
            if self.t_frac > 0.0:
                children.append((nvt, nvf, math.sqrt(self.t_frac)))
            if self.r_frac > 0.0:
                children.append((rvt, rvf, math.sqrt(self.r_frac)))
        return children


class HorizontalSurface(TFSurface):
    """
    f = const.  Normal in f-direction.
    sin(theta) = 1 / sqrt(1 + v_f^2)
    Snell: n1/sqrt(1+v_f1^2) = n2/sqrt(1+v_f2^2)
    Reflection: v_f -> -v_f, v_t unchanged.
    """
    def __init__(self, f_hz: float, n_a: float = 1.0, n_b: float = 1.0, **kw) -> None:
        super().__init__(n_a=n_a, n_b=n_b, **kw)
        self.f_hz = f_hz

    def intersect(self, state: RayState2D) -> Optional[float]:
        if abs(state.v_f) < 1e-9:
            return None
        dt = (self.f_hz - state.f) / state.v_f
        return dt if dt > 1e-9 else None

    def _side_a_to_b(self, state: RayState2D) -> bool:
        return state.v_f > 0.0

    def _snell(self, v_t, v_f, n1, n2) -> Tuple[Tuple[float, float], bool]:
        disc = (n2 / n1) ** 2 * (1.0 + v_f * v_f) - 1.0
        if disc < 0.0:
            return (self._reflect(v_t, v_f), True)
        return ((v_t, math.copysign(math.sqrt(disc), v_f)), False)

    def _reflect(self, v_t, v_f) -> Tuple[float, float]:
        return v_t, -v_f


class VerticalSurface(TFSurface):
    """
    t = const.  Normal in t-direction.
    sin(theta) = |v_f| / sqrt(1 + v_f^2)
    Snell: n1*sin = n2*sin2  =>  v_f2 via sin chain
    Reflection: v_t -> -v_t  (TIME REVERSAL), v_f unchanged.
    """
    def __init__(self, t_sec: float, n_a: float = 1.0, n_b: float = 1.0, **kw) -> None:
        super().__init__(n_a=n_a, n_b=n_b, **kw)
        self.t_sec = t_sec

    def intersect(self, state: RayState2D) -> Optional[float]:
        if abs(state.v_t) < 1e-9:
            return None
        dt = (self.t_sec - state.t) / state.v_t
        return dt if dt > 1e-9 else None

    def _side_a_to_b(self, state: RayState2D) -> bool:
        return state.v_t > 0.0

    def _snell(self, v_t, v_f, n1, n2) -> Tuple[Tuple[float, float], bool]:
        norm = math.sqrt(1.0 + v_f * v_f)
        sin1 = abs(v_f) / norm          # sin(theta_1)
        sin2 = (n1 / n2) * sin1
        if sin2 >= 1.0:
            return (self._reflect(v_t, v_f), True)
        cos2 = math.sqrt(max(0.0, 1.0 - sin2 * sin2))
        vf2  = math.copysign(sin2 / max(cos2, 1e-12), v_f)
        return ((v_t, vf2), False)

    def _reflect(self, v_t, v_f) -> Tuple[float, float]:
        return -v_t, v_f   # time reversal


class LineSurface(TFSurface):
    """
    Oblique surface: f = f0 + slope*(t - t0).
    Normal (arc-space, treating |v_t|=1 as unit t-speed): N = (-slope, 1)/||...||.
    Full vector Snell on the (1, v_f) ray direction.
    """
    def __init__(
        self, t0: float, f0: float, slope: float,
        n_a: float = 1.0, n_b: float = 1.0, **kw
    ) -> None:
        super().__init__(n_a=n_a, n_b=n_b, **kw)
        self.t0    = t0
        self.f0    = f0
        self.slope = slope
        nm         = math.sqrt(1.0 + slope * slope)
        self._nt   = -slope / nm
        self._nf   = 1.0   / nm

    def _f_surf(self, t: float) -> float:
        return self.f0 + self.slope * (t - self.t0)

    def intersect(self, state: RayState2D) -> Optional[float]:
        # f + v_f*dt = f_surf(t + v_t*dt) => dt*(v_f - slope*v_t) = f_surf(t) - f
        rel = state.v_f - self.slope * state.v_t
        if abs(rel) < 1e-9:
            return None
        dt = (self._f_surf(state.t) - state.f) / rel
        return dt if dt > 1e-9 else None

    def _side_a_to_b(self, state: RayState2D) -> bool:
        return (state.v_f - self.slope * state.v_t) > 0.0

    def _snell(self, v_t, v_f, n1, n2) -> Tuple[Tuple[float, float], bool]:
        nt, nf  = self._nt, self._nf
        spd     = math.sqrt(1.0 + v_f * v_f)           # |v_t|=1 so total arc-speed = sqrt(1+v_f^2)
        dt_hat  = v_t  / spd
        df_hat  = v_f  / spd
        cos_i   = abs(dt_hat * nt + df_hat * nf)
        sin2_t  = (n1 / n2) ** 2 * (1.0 - cos_i * cos_i)
        if sin2_t >= 1.0:
            return (self._reflect(v_t, v_f), True)
        cos_t   = math.sqrt(max(0.0, 1.0 - sin2_t))
        sign_n  = math.copysign(1.0, dt_hat * nt + df_hat * nf)
        r       = n1 / n2
        new_dt  = r * dt_hat + (r * cos_i - cos_t) * sign_n * nt
        new_df  = r * df_hat + (r * cos_i - cos_t) * sign_n * nf
        # recover v_f from (dt_hat, df_hat) back to (v_t, v_f) with |v_t|=1
        if abs(new_dt) < 1e-12:
            return (self._reflect(v_t, v_f), True)
        scale   = v_t / new_dt           # restore |v_t|=1
        new_vf  = new_df * scale
        new_vt  = 1.0 if v_t > 0 else -1.0
        return ((new_vt, new_vf), False)

    def _reflect(self, v_t, v_f) -> Tuple[float, float]:
        nt, nf = self._nt, self._nf
        spd    = math.sqrt(1.0 + v_f * v_f)
        dth, dfh = v_t / spd, v_f / spd
        dot    = dth * nt + dfh * nf
        rdth   = dth - 2.0 * dot * nt
        rdfh   = dfh - 2.0 * dot * nf
        if abs(rdth) < 1e-12:
            return v_t, -v_f
        scale  = v_t / rdth
        return (1.0 if v_t > 0 else -1.0, rdfh * scale)


# ---------------------------------------------------------------------------
# Fog medium
# ---------------------------------------------------------------------------

@dataclass
class FogMedium:
    """
    Poisson fog: scatters rays at exponential arc-time intervals.

    At each scatter event the ray is redirected to a random v_t (±1) and
    a random v_f drawn uniformly from [-max_chirp, +max_chirp].
    This is true isotropic scattering in TF space including time reversal.
    """
    scatter_rate:   float = 1.0          # scatters per elapsed second
    amplitude_loss: float = 0.30        # fractional amplitude lost per scatter
    n_scattered:    int   = 2           # child rays per scatter event
    max_chirp:      float = 1500.0      # Hz/s — max |v_f| assigned at scatter
    seed:           Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def dt_to_next(self) -> float:
        """Draw exponential inter-scatter elapsed time."""
        return float(-math.log(max(float(self._rng.random()), 1e-15)) / self.scatter_rate)

    def scatter(self, state: RayState2D) -> List[RayState2D]:
        """Return scattered child rays from a fog event."""
        A_child = state.A * (1.0 - self.amplitude_loss)
        children: List[RayState2D] = []
        for _ in range(self.n_scattered):
            vt_new = 1.0 if float(self._rng.random()) < 0.5 else -1.0
            vf_new = float(self._rng.uniform(-self.max_chirp, self.max_chirp))
            children.append(RayState2D(
                t=state.t, f=state.f,
                v_t=vt_new, v_f=vf_new,
                phi=state.phi, A=A_child,
            ))
        return children


def _draw_fog(fog: Optional[FogMedium]) -> float:
    return fog.dt_to_next() if fog is not None else math.inf


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class TFScene:
    """
    Bounded domain: t in [0, scene_end], f in [f_min, f_max].
    Frequency walls: v_f -> -v_f (frequency mirror).
    Temporal walls:  v_t -> -v_t (time reversal).
    """
    def __init__(
        self,
        f_min:            float = 40.0,
        f_max:            float = 4000.0,
        freq_mirror_loss: float = 0.0,
        time_mirror_loss: float = 0.0,
    ) -> None:
        if f_min >= f_max:
            raise ValueError("f_min must be < f_max")
        self.f_min            = f_min
        self.f_max            = f_max
        self.freq_mirror_loss = freq_mirror_loss
        self.time_mirror_loss = time_mirror_loss
        self._surfaces: List[TFSurface] = []

    def add_surface(self, surf: TFSurface) -> "TFScene":
        self._surfaces.append(surf)
        return self

    @property
    def surfaces(self) -> List[TFSurface]:
        return self._surfaces


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------

class TFPropagator:
    """
    Bidirectional TF ray tracer.

    Rays advance in elapsed-time steps.  At each step the earliest event is:
      - fog scatter: random new (v_t, v_f), creates n_scattered children
      - frequency mirror: v_f -> -v_f
      - temporal mirror:  v_t -> -v_t  (time reversal)
      - surface crossing: Snell + reflected branch

    All children are queued (BFS).  Prunes when A < min_amplitude,
    bounces > max_bounces, or total segment count exceeds max_segments.
    """

    def __init__(
        self,
        scene:         TFScene,
        scene_end:     float,
        max_segments:  int   = 5000,
        min_amplitude: float = 0.02,
        max_bounces:   int   = 200,
    ) -> None:
        self._scene   = scene
        self._end     = scene_end
        self._max_seg = max_segments
        self._min_amp = min_amplitude
        self._max_b   = max_bounces

    def propagate(
        self,
        initial: RayState2D,
        fog:     Optional[FogMedium] = None,
    ) -> List[ChirpSegment]:
        """Return all ChirpSegments sorted by t_start."""
        segments: List[ChirpSegment] = []
        # (state, bounce_count, dt_to_next_fog)
        queue: List[Tuple[RayState2D, int, float]] = [
            (initial, 0, _draw_fog(fog))
        ]

        while queue and len(segments) < self._max_seg:
            state, n_b, dt_fog = queue.pop(0)

            if (state.A < self._min_amp or n_b > self._max_b
                    or state.f < self._scene.f_min - 10.0
                    or state.f > self._scene.f_max + 10.0
                    or state.t < -1e-3 or state.t > self._end + 1e-3):
                continue

            dt_ev, surface, etype = self._next_event(state)

            # Fog scatter before next surface/mirror?
            if fog is not None and dt_fog < dt_ev:
                fog_state = state.advance(dt_fog)
                _append_seg(segments, state, fog_state)
                for child in fog.scatter(fog_state):
                    queue.append((child, n_b + 1, _draw_fog(fog)))
                continue

            hit = state.advance(dt_ev)
            _append_seg(segments, state, hit)

            fl = 1.0 - self._scene.freq_mirror_loss
            tl = 1.0 - self._scene.time_mirror_loss

            if etype == 'mirror_top':
                queue.append((
                    RayState2D(hit.t, hit.f, hit.v_t, -abs(hit.v_f), hit.phi, hit.A * fl),
                    n_b + 1, _draw_fog(fog),
                ))
            elif etype == 'mirror_bottom':
                queue.append((
                    RayState2D(hit.t, hit.f, hit.v_t, abs(hit.v_f), hit.phi, hit.A * fl),
                    n_b + 1, _draw_fog(fog),
                ))
            elif etype == 'mirror_right':
                queue.append((
                    RayState2D(hit.t, hit.f, -1.0, hit.v_f, hit.phi, hit.A * tl),
                    n_b + 1, _draw_fog(fog),
                ))
            elif etype == 'mirror_left':
                queue.append((
                    RayState2D(hit.t, hit.f, 1.0, hit.v_f, hit.phi, hit.A * tl),
                    n_b + 1, _draw_fog(fog),
                ))
            elif etype == 'surface' and surface is not None:
                for nvt, nvf, ascale in surface.apply(hit):
                    queue.append((
                        RayState2D(hit.t, hit.f, nvt, nvf, hit.phi, hit.A * ascale),
                        n_b + 1, _draw_fog(fog),
                    ))

        return sorted(segments, key=lambda s: s.t_start)

    def _next_event(
        self, state: RayState2D
    ) -> Tuple[float, Optional[TFSurface], str]:
        best = math.inf
        surf = None
        etype = 'none'

        # Temporal walls
        if state.v_t > 0.0:
            dt = (self._end - state.t)   # v_t=+1, elapsed = distance to right wall
            if 1e-9 < dt < best:
                best, surf, etype = dt, None, 'mirror_right'
        else:
            dt = state.t                 # v_t=-1, elapsed to reach t=0
            if 1e-9 < dt < best:
                best, surf, etype = dt, None, 'mirror_left'

        # Frequency walls
        if state.v_f > 1e-9:
            dt = (self._scene.f_max - state.f) / state.v_f
            if 1e-9 < dt < best:
                best, surf, etype = dt, None, 'mirror_top'
        elif state.v_f < -1e-9:
            dt = (self._scene.f_min - state.f) / state.v_f
            if 1e-9 < dt < best:
                best, surf, etype = dt, None, 'mirror_bottom'

        # Interior surfaces
        for s in self._scene.surfaces:
            dt = s.intersect(state)
            if dt is not None and 1e-9 < dt < best:
                best, surf, etype = dt, s, 'surface'

        if math.isinf(best):
            best  = 100.0
            etype = 'none'

        return best, surf, etype


# ---------------------------------------------------------------------------
# GPU-vectorized ray propagator
# ---------------------------------------------------------------------------

class TFRayGPU:
    """
    GPU-batched bidirectional TF ray tracer for fog corridor scenes.

    All active rays are stored as a single [B, 6] float64 tensor:
        col 0: t    col 1: f    col 2: v_t    col 3: v_f
        col 4: phi  col 5: A

    Each iteration:
      1. Compute per-ray dt to every event (walls, fog, surfaces) — [B, E] tensor
      2. Take the argmin per ray — each ray advances by its own dt
      3. Advance all rays simultaneously (quadratic phase)
      4. Record one ChirpSegment per live ray (accumulated as numpy on CPU)
      5. Apply events via masked index ops (zero extra GPU↔CPU syncs):
           - mirrors: in-place flip
           - fog: kill original, spawn n_scattered children (random v_t, v_f)
           - surfaces: primary ray updated in-place, reflected child appended
           - aperture jaws (mask objects): occupancy-based hard-stop or Beer-law decay
           - conductor potential fields: E = -∇phi Lorentz force on charged rays
           - magnetic arc: exact circular rotation of (v_t, v_f)
      6. Prune by amplitude / bounces / bounds
      7. When batch exceeds max_batch, queue overflow (processed next round)

    Two boundary physics modes are supported:

    Mode A — Aperture jaw (``add_mask_object`` with ``jaw_mode`` set):
      The alpha channel of the PNG is the occupancy field occ(t,f) ∈ [0,1].
      'hard'  → ray killed immediately when occ > jaw_threshold (opaque metal hole).
      'beer'  → A *= exp(-jaw_sigma * occ * dt) (Beer-law absorber with transmission hole).
      Rays that pass through the zero-occupancy hole are unaffected: true aperture.

    Mode B — Conductor potential field (``add_potential_field``):
      PNG encodes scalar potential phi(t,f).  E = -∇phi is sampled on GPU with
      grid_sample and applied as a Lorentz electric force on charged rays:
        d/dτ (v_t, v_f) += charge * charge_scale * (E_t, E_f)
      Combines with the magnetic arc (if B_field != 0) to give full 2-D
      Lorentz + electric trajectories shaped by the field geometry.

    Surface specs are plain tuples, not TFSurface objects:
      h_surfs: list of (f_hz, n_a, n_b, t_frac, r_frac)   — horizontal
      v_surfs: list of (t_sec, n_a, n_b, t_frac, r_frac)  — vertical
    """

    _EPS = 1e-9
    _INF = 1e30

    # Event indices
    _FOG    = 0
    _FMAX   = 1
    _FMIN   = 2
    _TRIGHT = 3
    _TLEFT  = 4
    # _HS_OFF = 5  (horizontal surfaces start here)
    # _VS_OFF = 5 + len(h_surfs)

    def __init__(
        self,
        scene_end:        float,
        f_min:            float,
        f_max:            float,
        h_surfs,                        # list of (f_hz, n_a, n_b, t_frac, r_frac)
        v_surfs,                        # list of (t_sec, n_a, n_b, t_frac, r_frac)
        freq_mirror_loss: float = 0.0,
        time_mirror_loss: float = 0.0,
        scatter_rate:     float = 1.2,
        amp_loss:         float = 0.30,
        n_scattered:      int   = 2,
        max_chirp:        float = 1800.0,
        max_batch:        int   = 5000,
        max_segments:     int   = 500_000,
        min_amplitude:    float = 0.03,
        max_bounces:      int   = 300,
        device:              str   = "cuda",
        seed:                int   = 42,
        phase_mode:          str   = "physical",
        density_png:         str   = None,
        density_scale:       float = 1.0,
        density_sensitivity: float = 8000.0,
        density_step:        float = None,
    ) -> None:
        import torch as _torch
        self._torch = _torch
        if device == "cpu" or not _torch.cuda.is_available():
            self.dev = _torch.device("cpu")
        else:
            self.dev = _torch.device(device)
        print(f"  TFRayGPU device={self.dev}", flush=True)

        self.scene_end        = float(scene_end)
        self.f_min            = float(f_min)
        self.f_max            = float(f_max)
        self.h_surfs          = list(h_surfs)
        self.v_surfs          = list(v_surfs)
        self.freq_mirror_loss = float(freq_mirror_loss)
        self.time_mirror_loss = float(time_mirror_loss)
        self.scatter_rate     = float(scatter_rate)
        self.amp_loss         = float(amp_loss)
        self.n_scattered      = int(n_scattered)
        self.max_chirp        = float(max_chirp)
        self.max_batch        = int(max_batch)
        self.max_segments     = int(max_segments)
        self.min_amplitude    = float(min_amplitude)
        self.max_bounces      = int(max_bounces)
        self._seed            = int(seed)
        if phase_mode not in ('coherent', 'physical'):
            raise ValueError("phase_mode must be 'coherent' or 'physical'")
        self.phase_mode           = phase_mode
        self._density_sensitivity = float(density_sensitivity)

        # Density field: PNG → greyscale → n(t,f) refractive index field.
        # Pixel brightness [0,1] maps to n in [1, 1+density_scale].
        # Gradient of n bends ray v_f continuously (eikonal ray equation).
        if density_png is not None:
            from PIL import Image as _PILImage
            img = _PILImage.open(density_png).convert('L')
            n_arr = np.array(img, dtype=np.float32) / 255.0
            n_arr = n_arr[::-1, :].copy()  # flip: row 0 = f_min, row H-1 = f_max
            n_arr = 1.0 + n_arr * float(density_scale)
            H, W  = n_arr.shape
            n_t   = _torch.tensor(n_arr)
            # Physical-unit gradients: dn/df in 1/Hz, dn/dt in 1/s
            dn_df = _torch.gradient(n_t, dim=0)[0] * (H / (f_max - f_min))
            dn_dt = _torch.gradient(n_t, dim=1)[0] * (W / scene_end)
            self._dn_df_gs       = dn_df.view(1, 1, H, W)  # for grid_sample
            self._dn_dt_gs       = dn_dt.view(1, 1, H, W)
            self._density_step   = float(density_step or (scene_end / 300))
            print(f"  Density field: {W}×{H} px, n=[1, {1+density_scale:.2f}], "
                  f"step={self._density_step:.4f}s", flush=True)
        else:
            self._dn_df_gs     = None
            self._density_step = None

        self._HS_OFF   = 5
        self._VS_OFF   = 5 + len(self.h_surfs)
        self._DENS_OFF = 5 + len(self.h_surfs) + len(self.v_surfs)

        # Aperture jaw / conductor mode:
        #   mask_objects (aperture_jaws): list of dicts, each with keys:
        #     pixels   [H,W] float32 — greyscale mask (0=air, 1=solid)
        #     grad_t   [1,1,H,W]     — spatial gradient in time direction
        #     grad_f   [1,1,H,W]     — spatial gradient in freq direction
        #     t0,t1    float         — time extent in scene (s)
        #     f0,f1    float         — freq extent in scene (Hz)
        #     reflect  float         — fraction of impact energy → reflection
        #     diffuse  float         — fraction → diffuse (charge + boost)
        #     boost    float         — amplitude multiplier on diffuse impact
        #     charge_rate float      — charge added per elapsed-second inside mask
        #     jaw_mode str           — 'none' | 'hard' | 'beer'
        #       'hard': ray killed immediately when occupancy > jaw_threshold
        #       'beer': Beer-law exponential attenuation inside solid region
        #     jaw_sigma  float       — absorption coeff for Beer-law (amplitude/s·occ)
        #     jaw_threshold float    — occupancy threshold that defines "inside solid"
        #   B_field: Lorentz rotation magnitude; d/dτ(vt,vf) = ωc×(−vf,vt), ωc=charge*B
        #   synchrotron_coeff: amplitude decay rate; dA/dτ = -A * k * charge² * f²
        #   _efields: list of conductor potential-field objects (E = -∇phi sampled on GPU)
        self._masks            = []
        self._efields          = []   # potential-field objects for conductor-like boundaries
        self._B_field          = 0.0
        self._synchrotron_coeff= 0.0
        self._arc_angle        = 0.05   # radians per magnetic arc step (≈126 steps/orbit)
        self._initial_charge   = 0.0   # charge at ray birth (enables B-field without objects)
        self._B_field_tlo      = -math.inf  # left t bound of active B-field section (s)
        self._B_field_thi      =  math.inf  # right t bound of active B-field section (s)

    def add_mask_object(
        self,
        png_path:    str,
        t0:          float,   # left edge in scene (s)
        t1:          float,   # right edge
        f0:          float,   # bottom edge (Hz)
        f1:          float,   # top edge
        # R channel = per-pixel reflectivity  (scaled by reflect)
        # G channel = per-pixel refractive index contribution (drives chirp bending)
        # B channel = per-pixel diffusivity  (scaled by diffuse)
        # A channel = occupancy field for aperture-jaw physics
        reflect:              float = 1.0,
        diffuse:              float = 1.0,
        boost:                float = 4.0,
        charge_rate:          float = 0.8,
        refract_sensitivity:  float = 2000.0,
        absorb:               float = 0.0,   # legacy: scale for alpha absorb texture
        # --- aperture-jaw mode -------------------------------------------
        # jaw_mode='hard' : ray is killed immediately when inside solid region
        # jaw_mode='beer' : Beer-law exponential attenuation inside solid region
        #   A_new = A * exp(-jaw_sigma * occ * dt)  (occ in [0,1])
        # jaw_mode='none' (default): original conductor-heuristic behaviour
        jaw_mode:             str   = 'none',
        jaw_sigma:            float = 10.0,  # absorption coefficient for Beer-law [1/(occ·s)]
        jaw_threshold:        float = 0.5,   # occupancy above which a ray is "inside"
    ) -> None:
        from PIL import Image as _PILImage
        img  = _PILImage.open(png_path).convert('RGBA')
        arr  = np.array(img, dtype=np.float32) / 255.0   # [H, W, 4]
        arr  = arr[::-1, :, :].copy()                      # row 0 = f0 (bottom)
        H, W = arr.shape[:2]
        torch = self._torch
        r_px  = torch.tensor(arr[:, :, 0])   # [H,W]  reflectivity field
        g_px  = torch.tensor(arr[:, :, 1])   # [H,W]  refractive field
        b_px  = torch.tensor(arr[:, :, 2])   # [H,W]  diffuse field
        a_px  = torch.tensor(arr[:, :, 3])   # [H,W]  absorb field (alpha channel)
        # Surface normals from G-channel gradient (physical units).
        gf = torch.gradient(g_px, dim=0)[0] * (H / max(f1 - f0, 1e-6))  # dG/df
        gt = torch.gradient(g_px, dim=1)[0] * (W / max(t1 - t0, 1e-6))  # dG/dt
        # Also need R-channel gradient for reflection when G is flat.
        rf = torch.gradient(r_px, dim=0)[0] * (H / max(f1 - f0, 1e-6))
        rt = torch.gradient(r_px, dim=1)[0] * (W / max(t1 - t0, 1e-6))
        if jaw_mode not in ('none', 'hard', 'beer'):
            raise ValueError("jaw_mode must be 'none', 'hard', or 'beer'")
        self._masks.append(dict(
            gs_r  = r_px.view(1, 1, H, W),
            gs_g  = g_px.view(1, 1, H, W),
            gs_b  = b_px.view(1, 1, H, W),
            gs_a  = a_px.view(1, 1, H, W),
            gs_gf = gf.view(1, 1, H, W),
            gs_gt = gt.view(1, 1, H, W),
            gs_rf = rf.view(1, 1, H, W),
            gs_rt = rt.view(1, 1, H, W),
            t0=float(t0), t1=float(t1), f0=float(f0), f1=float(f1),
            reflect=float(reflect), diffuse=float(diffuse),
            boost=float(boost), charge_rate=float(charge_rate),
            refract_sensitivity=float(refract_sensitivity),
            absorb=float(absorb),
            jaw_mode=jaw_mode,
            jaw_sigma=float(jaw_sigma),
            jaw_threshold=float(jaw_threshold),
        ))
        print(f"  Aperture jaw '{png_path}': {W}×{H} px  "
              f"t=[{t0:.2f},{t1:.2f}]s  f=[{f0:.0f},{f1:.0f}]Hz  "
              f"reflect_scale={reflect:.2f}  diffuse_scale={diffuse:.2f}  "
              f"boost={boost:.1f}  charge_rate={charge_rate:.2f}  "
              f"jaw_mode={jaw_mode}  jaw_sigma={jaw_sigma:.2f}  "
              f"jaw_threshold={jaw_threshold:.2f}  "
              f"refract_sensitivity={refract_sensitivity:.0f}", flush=True)

    def set_plasma(self, B_field: float = 0.001, synchrotron_coeff: float = 0.0001,
                   arc_angle: float = 0.05, initial_charge: float = 1.0,
                   plasma_tlo: float = -math.inf, plasma_thi: float = math.inf) -> None:
        """Enable charged-particle physics.
        B_field: Lorentz rotation d/dτ(vt,vf) = ωc×(-vf,vt), ωc=charge*B_field.
            Gives exact circular orbits in TF space; each step rotates velocity by
            arc_angle radians using the closed-form integral (no Euler error).
        synchrotron_coeff: amplitude drain dA/dτ = -A*k*charge²*(f/f_max)².
        arc_angle: radians of rotation per magnetic event step (smaller = more accurate
            segment representation of the spiral, default 0.05 ≈ 126 steps/orbit).
        initial_charge: charge assigned at ray birth so B_field curves trajectories
            immediately, even without mask objects to ionise rays.
        plasma_tlo / plasma_thi: t-axis window [s] in which the B-field magnet section
            is active.  Rays outside this window (e.g. at the emitter mouth) travel
            field-free — the magnetic arc event dt is set to INF there."""
        self._B_field           = float(B_field)
        self._synchrotron_coeff = float(synchrotron_coeff)
        self._arc_angle         = float(arc_angle)
        self._initial_charge    = float(initial_charge)
        self._B_field_tlo       = float(plasma_tlo)
        self._B_field_thi       = float(plasma_thi)
        tband = (f"t=[{plasma_tlo:.3f},{plasma_thi:.3f}] s"
                 if math.isfinite(plasma_tlo) or math.isfinite(plasma_thi)
                 else "(full range)")
        print(f"  Plasma: B={B_field:.6f}  synchrotron={synchrotron_coeff:.6f}"
              f"  arc_angle={arc_angle:.4f} rad ({int(round(2*math.pi/arc_angle))} steps/orbit)"
              f"  B-field section {tband}",
              flush=True)

    def add_potential_field(
        self,
        png_path:     str,
        t0:           float,
        t1:           float,
        f0:           float,
        f1:           float,
        charge_scale: float = 1.0,
        field_step:   float = None,
    ) -> None:
        """Add a conductor-like scalar potential field.

        The greyscale PNG encodes a scalar potential phi(t, f) mapped to
        pixel brightness [0, 1].  The engine computes E = -∇phi and applies
        a Lorentz+electric force during a dedicated fixed-step field event:

            d/dτ (v_t, v_f) = charge * charge_scale * (E_t, E_f)
                               + charge * B * (-v_f, v_t)   [if B_field != 0]

        Gradients are computed in physical units (1/s for d/dt, 1/Hz for d/df)
        and sampled with bilinear grid_sample on GPU — exactly the same
        infrastructure used by the density-field path.

        field_step: integration cadence in elapsed arc-seconds.  Defaults to
            scene_end / 300 (same default as density_step).
        """
        from PIL import Image as _PILImage
        img   = _PILImage.open(png_path).convert('L')
        arr   = np.array(img, dtype=np.float32) / 255.0   # [H, W]
        arr   = arr[::-1, :].copy()                        # row 0 = f0 (bottom)
        H, W  = arr.shape
        torch = self._torch
        phi_t = torch.tensor(arr)                          # [H, W]
        # Physical-unit gradients: dphi/df [1/Hz], dphi/dt [1/s]
        dphi_df = torch.gradient(phi_t, dim=0)[0] * (H / max(f1 - f0, 1.0))
        dphi_dt = torch.gradient(phi_t, dim=1)[0] * (W / max(t1 - t0, 1e-9))
        step    = float(field_step or (self.scene_end / 300))
        self._efields.append(dict(
            dphi_dt   = dphi_dt.view(1, 1, H, W),
            dphi_df   = dphi_df.view(1, 1, H, W),
            t0        = float(t0),
            t1        = float(t1),
            f0        = float(f0),
            f1        = float(f1),
            charge_scale = float(charge_scale),
            field_step   = step,
        ))
        print(f"  Potential field '{png_path}': {W}×{H} px  "
              f"t=[{t0:.2f},{t1:.2f}]s  f=[{f0:.0f},{f1:.0f}]Hz  "
              f"charge_scale={charge_scale:.3f}  field_step={step:.4f}s", flush=True)

    # ------------------------------------------------------------------
    def _uniform(self, n: int, rng) -> "Tensor":
        return self._torch.rand(n, generator=rng, device=self.dev, dtype=self._torch.float64)

    def _draw_fog_dt(self, n: int, rng) -> "Tensor":
        u = self._uniform(n, rng).clamp_(min=1e-15)
        return -self._torch.log(u) / self.scatter_rate

    def _full(self, n: int, v: float) -> "Tensor":
        return self._torch.full((n,), v, dtype=self._torch.float64, device=self.dev)

    # ------------------------------------------------------------------
    @staticmethod
    def make_emitter_states(
        n_rays:                int,
        t0:                    float,
        f0:                    float,
        vt0:                   float   = 1.0,
        f0_spread:             float   = 0.0,
        vf0_spread:            float   = 0.0,
        velocity_angle_spread: float   = 0.0,
        A0:                    float   = 1.0,
        A0_spread:             float   = 0.0,
        t0_jitter:             float   = 0.0,
        phi0:                  float   = 0.0,
        seed:                  int     = None,
    ) -> np.ndarray:
        """Build a [N, 6] init_states array with per-ray spread for use with propagate().

        All spread parameters are half-widths of a uniform distribution centred on
        the base value, so the actual range for each ray is [base - spread, base + spread].

        Parameters
        ----------
        n_rays                 : number of rays to emit
        t0                     : base launch time (s)
        f0                     : base launch frequency (Hz)
        vt0                    : base time-direction (+1 forward, -1 backward); sign is preserved
        f0_spread              : ± spread in launch frequency (Hz)
        vf0_spread             : ± spread in launch chirp-rate v_f (Hz/s)
        velocity_angle_spread  : ± spread in initial velocity angle (rad).  Rotates each
                                 ray's (vt, vf) vector by a random angle drawn from
                                 Uniform(-spread, +spread), placing rays at different
                                 positions on the cyclotron orbit circle so they dephase
                                 even when all charges are identical.
        A0                     : base amplitude
        A0_spread              : ± spread in amplitude; final A is clamped to [0, A0+A0_spread]
        t0_jitter              : ± jitter in launch time (s)
        phi0                   : base launch phase (rad); physical mode ignores this (set in propagate)
        seed                   : optional RNG seed for reproducibility

        Returns
        -------
        np.ndarray shape [N, 6] with columns [t, f, v_t, v_f, phi, A]
        """
        rng = np.random.default_rng(seed)

        def _jitter(base, spread):
            return base + rng.uniform(-spread, spread, size=n_rays) if spread > 0.0 else np.full(n_rays, base)

        t_col   = _jitter(t0,    t0_jitter)
        f_col   = _jitter(f0,    f0_spread)
        vf_col  = _jitter(0.0,   vf0_spread)
        A_col   = np.clip(_jitter(A0, A0_spread), 0.0, None)
        vt_col  = np.full(n_rays, 1.0 if vt0 >= 0.0 else -1.0)
        phi_col = np.full(n_rays, phi0)

        # Apply per-ray velocity angle jitter: rotate (vt, vf) by Δθ per ray.
        # This spreads rays around the cyclotron orbit circle at birth, breaking
        # the lockstep that causes sawtooth beam-envelope oscillations.
        if velocity_angle_spread > 0.0:
            dtheta  = rng.uniform(-velocity_angle_spread, velocity_angle_spread, size=n_rays)
            speed   = np.sqrt(vt_col ** 2 + vf_col ** 2)
            theta0  = np.arctan2(vf_col, vt_col)
            theta1  = theta0 + dtheta
            vt_col  = np.cos(theta1) * speed
            vf_col  = np.sin(theta1) * speed

        return np.stack([t_col, f_col, vt_col, vf_col, phi_col, A_col], axis=1)

    # ------------------------------------------------------------------
    def propagate(self, init_states: np.ndarray) -> List[ChirpSegment]:
        """
        init_states: np.ndarray shape [N, 6] = [t, f, v_t, v_f, phi, A]
        Returns: List[ChirpSegment] sorted by t_start.
        """
        torch = self._torch
        rng   = torch.Generator(device=self.dev)
        rng.manual_seed(self._seed)

        s = torch.tensor(init_states, dtype=torch.float64, device=self.dev)
        if self.phase_mode == 'physical':
            s[:, 4] = self._uniform(s.shape[0], rng) * PI2
        fog_dt  = self._draw_fog_dt(s.shape[0], rng)
        bounces = torch.zeros(s.shape[0], dtype=torch.int32,  device=self.dev)
        charge  = (torch.full((s.shape[0],), self._initial_charge,
                              dtype=torch.float64, device=self.dev)
                   if self._B_field != 0.0 and self._initial_charge != 0.0
                   else torch.zeros(s.shape[0], dtype=torch.float64, device=self.dev))

        # Move mask and efield tensors to device once.
        masks_dev = []
        for m in self._masks:
            masks_dev.append({k: (v.to(self.dev) if isinstance(v, self._torch.Tensor) else v)
                               for k, v in m.items()})
        efields_dev = []
        for ef in self._efields:
            efields_dev.append({k: (v.to(self.dev) if isinstance(v, self._torch.Tensor) else v)
                                 for k, v in ef.items()})
        has_plasma = bool(masks_dev) or self._B_field != 0.0

        # CPU segment buffer — pre-allocated, flushed each iteration.
        # GPU only ever holds the current working batch, never the full history.
        # Columns: [ts, te, fs, cr, phi, A].
        seg_buf   = np.empty((self.max_segments, 6), dtype=np.float64)
        seg_ptr   = 0

        # Overflow queue stores CPU tensors to avoid pinning GPU memory.
        # Items are (s_cpu, fog_dt_cpu, bounces_cpu) numpy arrays.
        overflow: list = []

        EPS = self._EPS
        INF = self._INF

        def _to_gpu(arr_cpu):
            return torch.tensor(arr_cpu, dtype=torch.float64, device=self.dev)

        def _to_gpu_i(arr_cpu):
            return torch.tensor(arr_cpu, dtype=torch.int32, device=self.dev)

        while (s.shape[0] > 0 or overflow) and seg_ptr < self.max_segments:
            if s.shape[0] == 0:
                sc, fdc, bc, cc = overflow.pop(0)
                s       = _to_gpu(sc)
                fog_dt  = _to_gpu(fdc)
                bounces = _to_gpu_i(bc)
                charge  = _to_gpu(cc)

            # Sub-batch: keep highest-amplitude rays, move rest to CPU overflow.
            if s.shape[0] > self.max_batch:
                order   = s[:, 5].argsort(descending=True)
                ov_i    = order[self.max_batch:]
                overflow.insert(0, (
                    s[ov_i].cpu().numpy(),
                    fog_dt[ov_i].cpu().numpy(),
                    bounces[ov_i].cpu().numpy(),
                    charge[ov_i].cpu().numpy(),
                ))
                s       = s[order[:self.max_batch]].contiguous()
                fog_dt  = fog_dt[order[:self.max_batch]].contiguous()
                bounces = bounces[order[:self.max_batch]].contiguous()
                charge  = charge[order[:self.max_batch]].contiguous()

            B   = s.shape[0]
            t   = s[:, 0];  f   = s[:, 1]
            vt  = s[:, 2];  vf  = s[:, 3]
            phi = s[:, 4];  A   = s[:, 5]
            INF_B = self._full(B, INF)

            # Per-ray Lorentz cyclotron frequency: ωc = charge × B.
            # Computed once per iteration; used in dt_list and event handler.
            _HAS_B   = self._B_field != 0.0
            _omega_c = (charge * self._B_field) if _HAS_B else None

            # ---- per-ray dt to every event --------------------------
            dt_list: list = []

            # 0 fog
            dt_list.append(fog_dt)

            # 1 f_max mirror
            dt_list.append(torch.where(vf > EPS, (self.f_max - f) / vf, INF_B))

            # 2 f_min mirror
            dt_list.append(torch.where(vf < -EPS, (self.f_min - f) / vf, INF_B))

            # 3 t_right  (vt=+1, hits scene_end)
            dt_list.append(torch.where(vt > 0, self.scene_end - t, INF_B))

            # 4 t_left   (vt=-1, hits t=0)
            dt_list.append(torch.where(vt < 0, t, INF_B))

            # horizontal surfaces
            for (f_hz, *_) in self.h_surfs:
                df           = f_hz - f
                approaching  = (df * vf) > EPS          # same sign → closing
                dt_list.append(torch.where(approaching, df / vf, INF_B))

            # vertical surfaces  (vt=±1 so /vt is just *vt)
            for (t_sec, *_) in self.v_surfs:
                dt_v = torch.where(vt > 0, t_sec - t, t - t_sec)
                dt_list.append(torch.where(dt_v > EPS, dt_v, INF_B))

            # density field — fixed integration step
            if self._dn_df_gs is not None:
                dt_list.append(self._full(B, self._density_step))

            # potential-field (conductor) events — one fixed-step slot per field object.
            # Rays outside the field bbox get INF so the event never fires for them.
            _EFIELD_IDX_START = len(dt_list)
            for _ef in efields_dev:
                _in_ebox = ((t >= _ef['t0']) & (t <= _ef['t1']) &
                            (f >= _ef['f0']) & (f <= _ef['f1']))
                dt_list.append(torch.where(_in_ebox, self._full(B, _ef['field_step']), INF_B))

            # mask boundary events — slab intersection to prevent stepping through thin objects.
            # Computes the distance to the nearest bbox face in the ray's travel direction.
            # This constrains dt so the step lands exactly at the mask boundary; the post-step
            # plasma block then applies physics based on spatial position.
            if self._masks:
                # vf can be zero; vt is always ±1.
                _vf_s = torch.where(vf.abs() > EPS, vf, self._full(B, EPS))
                for _m in self._masks:
                    _dt_t0 = (_m['t0'] - t) / vt          # vt is ±1, never 0
                    _dt_t1 = (_m['t1'] - t) / vt
                    _dt_f0 = (_m['f0'] - f) / _vf_s
                    _dt_f1 = (_m['f1'] - f) / _vf_s
                    _tau_t_near = torch.min(_dt_t0, _dt_t1)
                    _tau_t_far  = torch.max(_dt_t0, _dt_t1)
                    _tau_f_near = torch.min(_dt_f0, _dt_f1)
                    _tau_f_far  = torch.max(_dt_f0, _dt_f1)
                    _tau_enter  = torch.max(_tau_t_near, _tau_f_near)
                    _tau_exit   = torch.min(_tau_t_far,  _tau_f_far)
                    _valid      = (_tau_enter < _tau_exit) & (_tau_exit > EPS)
                    # Outside → step to entry face; inside (enter≤0) → step to exit face.
                    _dt_face = torch.where(
                        _valid,
                        torch.where(_tau_enter > EPS, _tau_enter, _tau_exit.clamp(min=EPS)),
                        INF_B,
                    )
                    dt_list.append(_dt_face)

            # Lorentz magnetic arc — fires every _arc_angle radians of (vt,vf) rotation.
            # dt_arc[i] = arc_angle / |ωc[i]|.  Zero-charge or zero-B rays get INF.
            _MAGNETIC_IDX = len(dt_list)   # capture index before appending
            if _HAS_B:
                _in_bfield_section = (t >= self._B_field_tlo) & (t <= self._B_field_thi)
                _dt_arc = torch.where(
                    _in_bfield_section & (_omega_c.abs() > EPS),
                    self._full(B, self._arc_angle) / _omega_c.abs().clamp(min=EPS),
                    INF_B,
                )
                dt_list.append(_dt_arc)

            dt_tensor = torch.stack(dt_list, dim=1)                  # [B, N_EVT]
            dt_tensor = dt_tensor.clamp(min=EPS)
            # zero-or-negative distances → INF (can't go backward)
            dt_tensor[dt_tensor <= EPS * 2] = INF

            dt_min, evt = dt_tensor.min(dim=1)                       # [B], [B]

            # ---- sample density gradient at current positions -------
            # Done before the advance so we use the pre-event (t, f).
            if self._dn_df_gs is not None:
                import torch.nn.functional as _F
                dn_df_g = self._dn_df_gs.to(self.dev)
                # Normalise (t, f) to [-1, 1] for grid_sample.
                # grid_sample convention: x = horizontal (W/time), y = vertical (H/freq)
                gx = (t / self.scene_end) * 2.0 - 1.0              # [B]
                gy = ((f - self.f_min) / (self.f_max - self.f_min)) * 2.0 - 1.0
                grid_xy = torch.stack([gx, gy], dim=-1).view(1, 1, B, 2).float()
                _sampled_dn_df = _F.grid_sample(
                    dn_df_g, grid_xy, mode='bilinear',
                    padding_mode='border', align_corners=True,
                ).view(B).double()

            # ---- sample potential-field gradients at current positions -----
            # Pre-sample E = -∇phi for each efield object; stored as list of (E_t, E_f).
            # These are used in the efield event handler below.
            _efield_samples: list = []
            if efields_dev:
                import torch.nn.functional as _F
                for _ef in efields_dev:
                    _gx_ef = ((t - _ef['t0']) / max(_ef['t1'] - _ef['t0'], 1e-9) * 2.0 - 1.0).float()
                    _gy_ef = ((f - _ef['f0']) / max(_ef['f1'] - _ef['f0'], 1e-9) * 2.0 - 1.0).float()
                    _grid_ef = torch.stack([_gx_ef, _gy_ef], dim=-1).view(1, 1, B, 2)
                    _E_t = -_F.grid_sample(_ef['dphi_dt'], _grid_ef, mode='bilinear',
                                           padding_mode='zeros', align_corners=True).view(B).double()
                    _E_f = -_F.grid_sample(_ef['dphi_df'], _grid_ef, mode='bilinear',
                                           padding_mode='zeros', align_corners=True).view(B).double()
                    _efield_samples.append((_E_t, _E_f))

            # ---- advance all rays -----------------------------------
            t1   = t   + vt * dt_min
            f1   = f   + vf * dt_min
            phi1 = phi + PI2 * vt * (f * dt_min + 0.5 * vf * dt_min * dt_min)

            # ---- flush current batch to CPU segment buffer immediately ----------
            # GPU never accumulates history; only the current working batch lives there.
            is_fwd    = vt > 0
            seg_ts_b  = torch.where(is_fwd, t,   t1)
            seg_te_b  = torch.where(is_fwd, t1,  t)
            seg_fs_b  = torch.where(is_fwd, f,   f1)
            seg_phi_b = torch.where(is_fwd, phi, phi1)
            seg_cr_b  = vt * vf                              # df/dt_real
            batch_np  = torch.stack(
                [seg_ts_b, seg_te_b, seg_fs_b, seg_cr_b, seg_phi_b, A], dim=1
            ).cpu().numpy()
            n_new = min(B, self.max_segments - seg_ptr)
            seg_buf[seg_ptr:seg_ptr + n_new] = batch_np[:n_new]
            seg_ptr += n_new

            if seg_ptr >= self.max_segments:
                break

            # ---- apply events — unconditional torch.where on full batch ---------
            # No .any() guards, no in-place indexed writes → zero extra GPU→CPU syncs.
            new_t   = t1
            new_f   = f1
            new_vt  = vt.clone()
            new_vf  = vf.clone()
            new_phi = phi1
            new_A   = A.clone()
            new_fog = (fog_dt - dt_min).clamp_(min=0.0)
            new_b   = bounces + 1

            spawn_s:  list = []
            spawn_fd: list = []
            spawn_b:  list = []

            # 0: fog — kill originals, create n_scattered children
            fog_m = (evt == 0)
            fi = fog_m.nonzero(as_tuple=True)[0]   # one sync; needed for spawn count
            if fi.shape[0] > 0:
                nf       = fi.shape[0]
                # Energy-conserving scatter: n_scattered children each carry
                # sqrt((1-amp_loss)/n_scattered) of parent amplitude so total
                # outgoing power = (1-amp_loss)*A_parent^2.
                child_scale = math.sqrt(max(0.0, 1.0 - self.amp_loss) / max(1, self.n_scattered))
                Ac       = new_A[fi] * child_scale
                t_c      = new_t[fi];  f_c = new_f[fi];  b_c = new_b[fi]
                phi_base = new_phi[fi]
                for _ in range(self.n_scattered):
                    vt_c = torch.where(self._uniform(nf, rng) < 0.5,
                                       self._full(nf, 1.0), self._full(nf, -1.0))
                    vf_c = (self._uniform(nf, rng) * 2.0 - 1.0) * self.max_chirp
                    phi_c = (phi_base + self._uniform(nf, rng) * PI2
                             if self.phase_mode == 'physical' else phi_base)
                    spawn_s.append(torch.stack([t_c, f_c, vt_c, vf_c, phi_c, Ac], dim=1))
                    spawn_fd.append(self._draw_fog_dt(nf, rng))
                    spawn_b.append(b_c)
            keep = ~fog_m   # GPU bool tensor, no sync

            # 1: f_max mirror — perfect reflector minus absorption loss
            m1 = (evt == 1) & keep
            new_vf = torch.where(m1, -new_vf.abs(), new_vf)
            new_A  = torch.where(m1, new_A * math.sqrt(max(0.0, 1.0 - self.freq_mirror_loss)), new_A)

            # 2: f_min mirror
            m2 = (evt == 2) & keep
            new_vf = torch.where(m2, new_vf.abs(), new_vf)
            new_A  = torch.where(m2, new_A * math.sqrt(max(0.0, 1.0 - self.freq_mirror_loss)), new_A)

            # 3: t_right mirror
            m3 = (evt == 3) & keep
            new_vt = torch.where(m3, self._full(B, -1.0), new_vt)
            new_A  = torch.where(m3, new_A * math.sqrt(max(0.0, 1.0 - self.time_mirror_loss)), new_A)

            # 4: t_left mirror
            m4 = (evt == 4) & keep
            new_vt = torch.where(m4, self._full(B,  1.0), new_vt)
            new_A  = torch.where(m4, new_A * math.sqrt(max(0.0, 1.0 - self.time_mirror_loss)), new_A)

            # horizontal surfaces — Fresnel TE amplitude splitting
            # Tuple format: (f_hz, n_a, n_b, loss)  where loss is surface absorption [0,1).
            # R and T are computed from Snell + Fresnel; they sum to 1 before loss.
            # For a horizontal surface (normal in f-direction):
            #   disc = vf_transmitted² = (n2/n1)²(1+vf²) - 1
            #   r_TE = (|vf| - sqrt(disc)) / (|vf| + sqrt(disc))
            #   A_reflected = A * |r_TE| * sqrt(1-loss)
            #   A_transmitted = A * sqrt(1-r_TE²) * sqrt(1-loss)
            for i, (f_hz, n_a, n_bv, surf_loss) in enumerate(self.h_surfs):
                is_si  = (evt == self._HS_OFF + i) & keep
                vf_pos = new_vf > 0
                n1     = torch.where(vf_pos, self._full(B, n_a),  self._full(B, n_bv))
                n2     = torch.where(vf_pos, self._full(B, n_bv), self._full(B, n_a))
                disc   = (n2 / n1) ** 2 * (1.0 + new_vf * new_vf) - 1.0
                tir_i  = disc < 0
                sqd    = disc.clamp(min=0.0).sqrt()          # |vf_t|
                vf_abs = new_vf.abs()
                denom  = (vf_abs + sqd).clamp(min=1e-12)
                r_amp  = (vf_abs - sqd) / denom              # Fresnel r_TE in [-1,1]
                R      = r_amp * r_amp                       # intensity reflectance
                T_amp  = (1.0 - R).clamp(min=0.0).sqrt()    # sqrt(1-R)
                fade   = math.sqrt(max(0.0, 1.0 - surf_loss))
                # Reflected child spawned for all non-TIR hits (Fresnel gives the fraction)
                nti_m = is_si & ~tir_i
                nti   = nti_m.nonzero(as_tuple=True)[0]
                if nti.shape[0] > 0:
                    spawn_s.append(torch.stack([
                        new_t[nti], new_f[nti], new_vt[nti], -new_vf[nti],
                        new_phi[nti], new_A[nti] * r_amp[nti].abs() * fade,
                    ], dim=1))
                    spawn_fd.append(new_fog[nti]);  spawn_b.append(new_b[nti])
                # Primary: TIR → reflect (full amplitude minus surface loss)
                #          non-TIR → transmit with Fresnel T amplitude
                ref_vf = -new_vf
                tra_vf = torch.copysign(sqd, new_vf)
                new_vf = torch.where(is_si & tir_i,  ref_vf,
                         torch.where(is_si & ~tir_i, tra_vf, new_vf))
                new_A  = torch.where(is_si & tir_i,  new_A * fade,
                         torch.where(is_si & ~tir_i, new_A * T_amp * fade, new_A))

            # vertical surfaces — Fresnel TE amplitude splitting
            # Tuple format: (t_sec, n_a, n_b, loss)
            # For a vertical surface (normal in t-direction, |vt|=1):
            #   cos(θ_i) = 1/sqrt(1+vf²),  sin(θ_i) = |vf|/sqrt(1+vf²)
            #   r_TE = (n1·cos1 - n2·cos2) / (n1·cos1 + n2·cos2)
            for i, (t_sec, n_a, n_bv, surf_loss) in enumerate(self.v_surfs):
                is_si  = (evt == self._VS_OFF + i) & keep
                vt_pos = new_vt > 0
                n1     = torch.where(vt_pos, self._full(B, n_a),  self._full(B, n_bv))
                n2     = torch.where(vt_pos, self._full(B, n_bv), self._full(B, n_a))
                inv_v  = (1.0 + new_vf * new_vf).rsqrt()    # 1/sqrt(1+vf²)
                sin1   = new_vf.abs() * inv_v                # sin(θ_i)
                cos1   = inv_v                               # cos(θ_i)
                sin2   = (n1 / n2) * sin1
                tir_i  = sin2 >= 1.0
                cos2   = (1.0 - sin2.clamp(max=1.0) ** 2).clamp(min=0.0).sqrt()
                vf_t   = torch.copysign(sin2 / cos2.clamp(min=1e-12), new_vf)
                # Fresnel r_TE = (n1·cos1 - n2·cos2) / (n1·cos1 + n2·cos2)
                n1c1   = n1 * cos1
                n2c2   = n2 * cos2
                r_amp  = (n1c1 - n2c2) / (n1c1 + n2c2).clamp(min=1e-12)
                R      = r_amp * r_amp
                T_amp  = (1.0 - R).clamp(min=0.0).sqrt()
                fade   = math.sqrt(max(0.0, 1.0 - surf_loss))
                nti_m = is_si & ~tir_i
                nti   = nti_m.nonzero(as_tuple=True)[0]
                if nti.shape[0] > 0:
                    spawn_s.append(torch.stack([
                        new_t[nti], new_f[nti], -new_vt[nti], new_vf[nti],
                        new_phi[nti], new_A[nti] * r_amp[nti].abs() * fade,
                    ], dim=1))
                    spawn_fd.append(new_fog[nti]);  spawn_b.append(new_b[nti])
                new_vt = torch.where(is_si & tir_i,  -new_vt, new_vt)
                new_vf = torch.where(is_si & tir_i,   new_vf,
                         torch.where(is_si & ~tir_i,  vf_t,   new_vf))
                new_A  = torch.where(is_si & tir_i,  new_A * fade,
                         torch.where(is_si & ~tir_i, new_A * T_amp * fade, new_A))

            # density field — bend v_f toward higher-n regions
            # dv_f/dτ = sensitivity * dn/df * (1 + v_f²) / n  (eikonal ray eq.)
            if self._dn_df_gs is not None:
                is_dens = (evt == self._DENS_OFF) & keep
                bend    = _sampled_dn_df * self._density_sensitivity * (1.0 + new_vf * new_vf)
                new_vf  = torch.where(is_dens, (new_vf + bend).clamp(-self.max_chirp, self.max_chirp), new_vf)

            # ---- conductor potential-field events ------------------------------------
            # Apply E = -∇phi Lorentz force: d/dτ(v_t, v_f) += charge * charge_scale * (E_t, E_f)
            # Combined with any magnetic arc that fires on the same ray in subsequent steps.
            # vt is allowed to drift from ±1 here (same as the magnetic arc already does);
            # its sign is used for time direction and propagates correctly.
            for _ei, _ef in enumerate(efields_dev):
                is_ef     = (evt == _EFIELD_IDX_START + _ei) & keep
                _E_t, _E_f = _efield_samples[_ei]
                _q_eff    = charge * _ef['charge_scale']
                new_vt    = torch.where(is_ef, new_vt + _q_eff * _E_t * dt_min, new_vt)
                new_vf    = torch.where(is_ef,
                                        (new_vf + _q_eff * _E_f * dt_min
                                         ).clamp(-self.max_chirp, self.max_chirp),
                                        new_vf)

            # ---- Lorentz magnetic arc event -----------------------------------
            # Exact closed-form integral of d/dτ(vt,vf) = ωc×(-vf, vt).
            # Uses pre-step (t,f,vt,vf,phi) — no other event fired for this ray.
            # Position: t(τ) = t₀ + (vt₀ sinΩ + vf₀(cosΩ−1))/ωc
            # Velocity: vt(τ) = vt₀ cosΩ − vf₀ sinΩ
            # Phase:    ∫₀^τ vt(s)f(s) ds  — analytic, derived in docstring.
            if _HAS_B:
                is_mag   = (evt == _MAGNETIC_IDX) & keep
                _Omega   = _omega_c * dt_min          # actual rotation angle this step
                _sinO    = torch.sin(_Omega)
                _cosO    = torch.cos(_Omega)
                _cosO_m1 = _cosO - 1.0
                # Safe 1/ωc (large for near-zero ωc, but is_mag is False there)
                _inv_w   = torch.where(_omega_c.abs() > EPS,
                                       1.0 / _omega_c.clamp(min=EPS),
                                       torch.zeros_like(_omega_c))
                # Exact arc endpoint positions
                _t_arc  = t + (vt * _sinO    + vf * _cosO_m1) * _inv_w
                _f_arc  = f + (vt * (-_cosO_m1) + vf * _sinO) * _inv_w  # 1−cosΩ = −cosO_m1
                # Exact Lorentz velocity rotation
                _vt_arc = vt * _cosO - vf * _sinO
                _vf_arc = vt * _sinO + vf * _cosO
                # Exact phase integral: ∫₀^τ vt(s)·f(s) ds
                # Derived from ∫ [vt cosΩs − vf sinΩs][f + (vt(1−cosΩs)+vf sinΩs)/ωc] ds
                _sin2O    = torch.sin(2.0 * _Omega)
                _cos2O_m1 = torch.cos(2.0 * _Omega) - 1.0
                _phi_int  = (
                    (f + vt * _inv_w) * (vt * _sinO + vf * _cosO_m1) * _inv_w
                    - (vt * vt + vf * vf) * dt_min * _inv_w / 2.0
                    + (vf * vf - vt * vt) * _sin2O    * (_inv_w * _inv_w) / 4.0
                    - vt * vf             * _cos2O_m1 * (_inv_w * _inv_w) / 2.0
                )
                new_t   = torch.where(is_mag, _t_arc,   new_t)
                new_f   = torch.where(is_mag, _f_arc,   new_f)
                new_vt  = torch.where(is_mag, _vt_arc,  new_vt)
                new_vf  = torch.where(is_mag,
                                      _vf_arc.clamp(-self.max_chirp, self.max_chirp),
                                      new_vf)
                new_phi = torch.where(is_mag, phi + PI2 * _phi_int, new_phi)

            # ---- plasma: mask impacts + synchrotron radiation -----------------
            new_charge = charge.clone()
            if has_plasma:
                import torch.nn.functional as _F
                step_dt = dt_min   # actual elapsed time this step

                for m in masks_dev:
                    # Map ray (t, f) into this mask's normalised grid coords [-1,1].
                    in_t = (new_t >= m['t0']) & (new_t <= m['t1'])
                    in_f = (new_f >= m['f0']) & (new_f <= m['f1'])
                    in_bbox = in_t & in_f & keep
                    if not in_bbox.any():
                        continue

                    gx = ((new_t - m['t0']) / max(m['t1'] - m['t0'], 1e-9) * 2.0 - 1.0).float()
                    gy = ((new_f - m['f0']) / max(m['f1'] - m['f0'], 1e-9) * 2.0 - 1.0).float()
                    grid_xy = torch.stack([gx, gy], dim=-1).view(1, 1, B, 2)

                    def _samp(key):
                        return _F.grid_sample(m[key], grid_xy, mode='bilinear',
                                              padding_mode='zeros',
                                              align_corners=True).view(B).double()

                    r_val = (_samp('gs_r') * m['reflect']).clamp(0.0, 1.0)
                    g_val =  _samp('gs_g')
                    b_val = (_samp('gs_b') * m['diffuse']).clamp(0.0, 1.0)
                    ngf   =  _samp('gs_gf')
                    ngt   =  _samp('gs_gt')
                    nrf   =  _samp('gs_rf')
                    nrt   =  _samp('gs_rt')

                    # Hit: any active channel is non-negligible.
                    hit = in_bbox & ((r_val > 0.05) | (b_val > 0.05))

                    # Reflection (R channel): flip velocity along dominant gradient axis.
                    # Use whichever gradient (G or R) is larger for surface normal.
                    comb_f = torch.where(ngf.abs() >= nrf.abs(), ngf, nrf)
                    comb_t = torch.where(ngt.abs() >= nrt.abs(), ngt, nrt)
                    freq_dom = hit & (comb_f.abs() >= comb_t.abs())
                    time_dom = hit & (comb_f.abs() <  comb_t.abs())
                    new_vf = torch.where(freq_dom & (torch.rand(B, device=self.dev) < r_val),
                                         -new_vf, new_vf)
                    new_vt = torch.where(time_dom & (torch.rand(B, device=self.dev) < r_val),
                                         -new_vt, new_vt)

                    # Refraction (G channel): n-gradient bends chirp rate.
                    if m['refract_sensitivity'] != 0.0:
                        new_vf = (new_vf + in_bbox.double() * ngf
                                  * m['refract_sensitivity'] * step_dt
                                  ).clamp(-self.max_chirp, self.max_chirp)

                    # Diffusion (B channel): random amplitude boost.
                    diffuse_mask = hit & (torch.rand(B, device=self.dev) < b_val)
                    new_A = torch.where(diffuse_mask,
                                        new_A * (1.0 + m['boost'] * b_val), new_A)
                    new_A = new_A.clamp(max=1e6)   # prevent float64→float32 overflow → NaN

                    # Aperture-jaw absorb logic via the alpha (occupancy) channel.
                    # jaw_mode='hard' : inside solid (occ > threshold) → A = 0 immediately.
                    # jaw_mode='beer' : Beer-law decay A *= exp(-sigma * occ * dt) inside bbox.
                    # jaw_mode='none' : legacy path — scale by absorb scalar as before.
                    _jaw = m['jaw_mode']
                    if _jaw == 'hard':
                        occ = _samp('gs_a').clamp(0.0, 1.0)
                        inside = in_bbox & (occ > m['jaw_threshold'])
                        new_A = torch.where(inside, torch.zeros_like(new_A), new_A)
                    elif _jaw == 'beer':
                        occ = _samp('gs_a').clamp(0.0, 1.0)
                        decay = torch.exp(-(m['jaw_sigma'] * occ * step_dt).clamp(max=80.0))
                        new_A = torch.where(in_bbox, new_A * decay, new_A)
                    elif m['absorb'] > 0.0:
                        a_val = (_samp('gs_a') * m['absorb']).clamp(0.0, 1.0)
                        absorbed = in_bbox & (a_val > 0.5)
                        new_A = torch.where(absorbed, torch.zeros_like(new_A), new_A)

                    # Charge accumulation proportional to combined interaction.
                    new_charge = torch.where(
                        hit,
                        new_charge + (r_val + b_val).clamp(max=1.0) * m['charge_rate'] * step_dt,
                        new_charge,
                    )

                # Synchrotron radiation: energy lost to magnetic acceleration.
                # dA/dτ = -A * k * charge² * (f/f_max)²
                # f is normalised by f_max so k is independent of the frequency range
                # and intuitive: k=1 loses ~100% amplitude in one arc-step at f_max.
                if self._synchrotron_coeff != 0.0:
                    _f_norm = new_f / max(self.f_max, 1.0)
                    decay   = (new_charge * new_charge * _f_norm * _f_norm
                                * self._synchrotron_coeff * step_dt).clamp(max=0.99)
                    new_A   = new_A * (1.0 - decay)

                # Charge decays slowly over time (radiative damping).
                new_charge = (new_charge * (1.0 - 0.05 * step_dt)).clamp(min=0.0)

            # ---- prune and merge ------------------------------------
            spawn_c: list = []

            def _prune(ss, fd, bb, cc):
                ok = ((ss[:, 5] >= self.min_amplitude) &
                      (bb <= self.max_bounces) &
                      (ss[:, 1] >= self.f_min - 10.0) & (ss[:, 1] <= self.f_max + 10.0) &
                      (ss[:, 0] >= -0.001) & (ss[:, 0] <= self.scene_end + 0.001))
                return ss[ok], fd[ok], bb[ok], cc[ok]

            live_s  = torch.stack([new_t, new_f, new_vt, new_vf, new_phi, new_A], dim=1)[keep]
            live_fd = new_fog[keep];   live_b = new_b[keep];  live_c = new_charge[keep]
            live_s, live_fd, live_b, live_c = _prune(live_s, live_fd, live_b, live_c)

            # Spawned rays inherit zero charge (fresh secondary rays).
            for i in range(len(spawn_s)):
                spawn_c.append(torch.zeros(spawn_s[i].shape[0], dtype=torch.float64, device=self.dev))

            if spawn_s:
                sp    = torch.cat(spawn_s,  dim=0)
                sp_fd = torch.cat(spawn_fd, dim=0)
                sp_b  = torch.cat(spawn_b,  dim=0)
                sp_c  = torch.cat(spawn_c,  dim=0)
                sp, sp_fd, sp_b, sp_c = _prune(sp, sp_fd, sp_b, sp_c)
                s       = torch.cat([live_s, sp],    dim=0)
                fog_dt  = torch.cat([live_fd, sp_fd], dim=0)
                bounces = torch.cat([live_b,  sp_b],  dim=0)
                charge  = torch.cat([live_c,  sp_c],  dim=0)
            else:
                s       = live_s
                fog_dt  = live_fd
                bounces = live_b
                charge  = live_c

        # ---- assemble from pre-allocated CPU buffer ------------------
        if seg_ptr == 0:
            return np.empty((0, 6), dtype=np.float32)
        buf   = seg_buf[:seg_ptr]
        valid = ((buf[:, 1] - buf[:, 0]) > EPS) & (buf[:, 5] >= self.min_amplitude)
        buf   = buf[valid]
        buf   = buf[np.isfinite(buf).all(axis=1)]   # drop NaN/inf rows from overflow
        buf   = buf[np.argsort(buf[:, 0])]
        return buf.astype(np.float32)   # [N, 6]: ts, te, fs, cr, phi, A


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _chirp_window(
    amplitude: float, t_start: float, t_end: float,
    fade_s: float, t_arr: np.ndarray,
) -> np.ndarray:
    span = t_end - t_start
    fade = min(fade_s, span * 0.25)
    amp  = np.zeros_like(t_arr)
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    if not mask.any():
        return amp
    tm   = t_arr[mask]
    dt0  = tm - t_start
    dt1  = t_end - tm
    fi   = np.where(dt0 < fade, 0.5 * (1.0 - np.cos(np.pi * dt0 / fade)), 1.0)
    fo   = np.where(dt1 < fade, 0.5 * (1.0 - np.cos(np.pi * dt1 / fade)), 1.0)
    amp[mask] = amplitude * fi * fo
    return amp


class TFRayScene:
    """
    Renders segment arrays to audio and TF image.
    Accepts either np.ndarray shape [N,6] (ts,te,fs,cr,phi,A)
    or List[ChirpSegment] for the legacy CPU propagator.
    """

    def __init__(
        self,
        segments,
        scene_end:          float,
        output_sample_rate: float = 48_000.0,
        fade_s:             float = 0.002,
    ) -> None:
        if isinstance(segments, np.ndarray):
            if segments.shape[0] == 0:
                raise ValueError("TFRayScene requires at least one segment")
            self._ts = segments[:, 0].astype(np.float32)
            self._te = segments[:, 1].astype(np.float32)
            self._fs = segments[:, 2].astype(np.float32)
            self._cr = segments[:, 3].astype(np.float32)
            self._ph = segments[:, 4].astype(np.float32)
            self._am = segments[:, 5].astype(np.float32)
        else:
            if not segments:
                raise ValueError("TFRayScene requires at least one segment")
            self._ts = np.array([s.t_start        for s in segments], dtype=np.float32)
            self._te = np.array([s.t_end          for s in segments], dtype=np.float32)
            self._fs = np.array([s.f_start        for s in segments], dtype=np.float32)
            self._cr = np.array([s.chirp_rate     for s in segments], dtype=np.float32)
            self._ph = np.array([s.phase_at_start for s in segments], dtype=np.float32)
            self._am = np.array([s.amplitude      for s in segments], dtype=np.float32)
        self._end  = float(scene_end)
        self._sr   = float(output_sample_rate)
        self._fade = float(fade_s)

    def render_audio(self) -> np.ndarray:
        """GPU-tiled audio render. Tile size [SEG_CHUNK, TIME_CHUNK] is bounded
        regardless of scene length or segment count. Returns float64 waveform."""
        import torch
        dev   = torch.device('cuda')
        n     = int(math.ceil(self._end * self._sr)) + 1
        t_g   = torch.linspace(0.0, self._end, n, dtype=torch.float64, device=dev)
        out_g = torch.zeros(n, dtype=torch.float64, device=dev)

        ts_g  = torch.tensor(self._ts, dtype=torch.float64, device=dev)
        te_g  = torch.tensor(self._te, dtype=torch.float64, device=dev)
        fs_g  = torch.tensor(self._fs, dtype=torch.float64, device=dev)
        cr_g  = torch.tensor(self._cr, dtype=torch.float64, device=dev)
        ph_g  = torch.tensor(self._ph, dtype=torch.float64, device=dev)
        am_g  = torch.tensor(self._am, dtype=torch.float64, device=dev)
        dur_g = (te_g - ts_g).clamp(min=0.0)
        fd_g  = torch.minimum(
            torch.tensor(self._fade, dtype=torch.float64, device=dev),
            dur_g * 0.25,
        ).clamp(min=1e-12)

        i0_seg = torch.searchsorted(t_g.contiguous(), ts_g.contiguous()).clamp(0, n)
        i1_seg = torch.searchsorted(t_g.contiguous(), te_g.contiguous(), right=True).clamp(0, n)

        S          = len(self._ts)
        TIME_CHUNK = 8192    # ~170 ms at 48 kHz
        SEG_CHUNK  = 1024    # worst-case tile: [1024, 8192] f64 = 67 MB

        for tc0 in range(0, n, TIME_CHUNK):
            tc1  = min(tc0 + TIME_CHUNK, n)
            t_sl = t_g[tc0:tc1]                                   # [TC]

            for sc0 in range(0, S, SEG_CHUNK):
                sc1  = min(sc0 + SEG_CHUNK, S)
                i0_c = i0_seg[sc0:sc1]
                i1_c = i1_seg[sc0:sc1]
                if int(i1_c.max()) <= tc0 or int(i0_c.min()) >= tc1:
                    continue
                act = (i0_c < tc1) & (i1_c > tc0)
                if not act.any():
                    continue

                ts_c  = ts_g[sc0:sc1][act];   fs_c = fs_g[sc0:sc1][act]
                cr_c  = cr_g[sc0:sc1][act];   ph_c = ph_g[sc0:sc1][act]
                am_c  = am_g[sc0:sc1][act];   dur_c = dur_g[sc0:sc1][act]
                fd_c  = fd_g[sc0:sc1][act]

                dt   = t_sl[None, :] - ts_c[:, None]              # [C, TC]
                mask = (dt >= 0) & (dt <= dur_c[:, None])
                dt0  = dt;  dt1 = dur_c[:, None] - dt
                fi   = torch.where(dt0 < fd_c[:, None],
                                   0.5 * (1.0 - torch.cos(math.pi * dt0 / fd_c[:, None])),
                                   torch.ones_like(dt0))
                fo   = torch.where(dt1 < fd_c[:, None],
                                   0.5 * (1.0 - torch.cos(math.pi * dt1 / fd_c[:, None])),
                                   torch.ones_like(dt1))
                env  = am_c[:, None] * fi * fo * mask.double()
                phi  = ph_c[:, None] + PI2 * (
                    fs_c[:, None] * dt + 0.5 * cr_c[:, None] * dt * dt)
                out_g[tc0:tc1] += (env * torch.cos(phi)).sum(dim=0)

        return out_g.cpu().numpy()

    def render_complex_field(
        self,
        n_time:            int   = 600,
        n_freq:            int   = 500,
        freq_min:          Optional[float] = None,
        freq_max:          Optional[float] = None,
        log_freq:          bool  = False,
        heisenberg_spread: float = 2.0,
        summation:         str   = 'complex',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Accumulate the phasor field directly from segments.

        summation='complex'  (default) — sum phasors with phase:
            Z[j, i] += A * fade * G(f_j - f_ridge) * exp(1j * phi(t_i))
            Phase is evaluated exactly from the segment's LinearChirpPhasePath at t_i.
            Crossing rays interfere constructively or destructively based on their
            accumulated phase histories.  Use this to see interference patterns.

        summation='magnitude' — sum magnitudes, no phase:
            Z[j, i] += A * fade * G(f_j - f_ridge)
            Equivalent to incoherent power accumulation.  Shows ray density
            without interference.  Result is real (returned as complex128 with
            zero imaginary part for API uniformity).

        Phase at t_i is computed exactly as:
            phi(t_i) = phase_at_start + 2π*(f_start*dt + chirp_rate/2*dt²)
            dt = t_i - t_start
        This follows directly from the segment's LinearChirpPhasePath — the same
        formula used by the ray engine during propagation.

        Returns: (Z complex128 [n_freq, n_time], t_axis, f_axis)
        """
        if summation not in ('complex', 'magnitude'):
            raise ValueError("summation must be 'complex' or 'magnitude'")

        # Arrays already stored — no list comprehensions.
        ts_np  = self._ts;  fs_np = self._fs;  cr_np = self._cr
        ph_np  = self._ph;  am_np = self._am
        dur_np = (self._te - self._ts).clip(min=1e-3)
        sg_np  = (heisenberg_spread / dur_np).astype(np.float32)
        fd_np  = np.minimum(self._fade, dur_np * 0.25).clip(min=1e-12).astype(np.float32)

        f_ends = (fs_np + cr_np * dur_np).astype(np.float64)
        fmin_v = freq_min if freq_min is not None else max(1.0, float(np.minimum(fs_np, f_ends).min()) * 0.8)
        fmax_v = freq_max if freq_max is not None else float(np.maximum(fs_np, f_ends).max()) * 1.2

        t_axis = np.linspace(0.0, self._end, n_time, dtype=np.float32)
        f_axis = (np.logspace(np.log10(max(fmin_v, 1.0)), np.log10(fmax_v), n_freq, dtype=np.float32)
                  if log_freq else np.linspace(fmin_v, fmax_v, n_freq, dtype=np.float32))

        import torch as _torch
        field = self._render_field_gpu(
            _torch, t_axis, f_axis,
            ts_np, fs_np, cr_np, ph_np, am_np, dur_np, sg_np, fd_np, summation,
        )
        return field, t_axis.astype(np.float64), f_axis.astype(np.float64)

    def _render_field_gpu(
        self, torch, t_axis, f_axis,
        ts_np, fs_np, cr_np, ph_np, am_np, dur_np, sg_np, fd_np, summation,
    ) -> np.ndarray:
        """
        GPU-tiled field accumulation.
        Processes (CHUNK_S segments) × (CHUNK_T time columns) tiles — [n_freq, C, T]
        fits comfortably in VRAM; GPU never holds more than one tile at a time.
        """
        dev    = torch.device("cuda")
        S      = len(ts_np)
        n_freq = len(f_axis)
        n_time = len(t_axis)
        CHUNK_S, CHUNK_T = 2000, 100

        t_g = torch.tensor(t_axis, device=dev)
        f_g = torch.tensor(f_axis, device=dev)

        # All segment arrays reside on GPU for the duration of the render.
        ts_g  = torch.tensor(ts_np,  device=dev)
        fs_g  = torch.tensor(fs_np,  device=dev)
        cr_g  = torch.tensor(cr_np,  device=dev)
        ph_g  = torch.tensor(ph_np,  device=dev)
        am_g  = torch.tensor(am_np,  device=dev)
        dur_g = torch.tensor(dur_np, device=dev)
        sg_g  = torch.tensor(sg_np,  device=dev)
        fd_g  = torch.tensor(fd_np,  device=dev)

        field = torch.zeros(n_freq, n_time, dtype=torch.complex64, device=dev)
        pi2   = float(PI2)

        for s0 in range(0, S, CHUNK_S):
            s1    = min(s0 + CHUNK_S, S)
            ts_c  = ts_g[s0:s1];  fs_c  = fs_g[s0:s1];  cr_c = cr_g[s0:s1]
            ph_c  = ph_g[s0:s1];  am_c  = am_g[s0:s1];  dur_c = dur_g[s0:s1]
            sg_c  = sg_g[s0:s1];  fd_c  = fd_g[s0:s1]

            for t0 in range(0, n_time, CHUNK_T):
                t1   = min(t0 + CHUNK_T, n_time)
                t_sl = t_g[t0:t1]                          # [TC]

                dt     = t_sl[None, :] - ts_c[:, None]    # [C, TC]
                active = (dt >= 0) & (dt <= dur_c[:, None])
                if not active.any():
                    continue

                fd2 = fd_c[:, None]
                dt0, dt1 = dt, dur_c[:, None] - dt
                fi  = torch.where(dt0 < fd2,
                                  0.5 * (1.0 - torch.cos(math.pi * dt0 / fd2)),
                                  torch.ones_like(dt0))
                fo  = torch.where(dt1 < fd2,
                                  0.5 * (1.0 - torch.cos(math.pi * dt1 / fd2)),
                                  torch.ones_like(dt1))
                env = am_c[:, None] * fi * fo * active.float()   # [C, TC]

                f_ridge = fs_c[:, None] + cr_c[:, None] * dt     # [C, TC]
                phi_ct  = ph_c[:, None] + pi2 * (
                    fs_c[:, None] * dt + 0.5 * cr_c[:, None] * dt * dt
                )                                                  # [C, TC]

                if summation == 'complex':
                    phasor = env * torch.exp(1j * phi_ct)         # [C, TC] c64
                else:
                    phasor = env.to(torch.complex64)

                diff  = f_g[:, None, None] - f_ridge[None, :, :] # [F, C, TC]
                gauss = torch.exp(-0.5 * (diff / sg_c[None, :, None]) ** 2)
                field[:, t0:t1] += torch.einsum('fct,ct->ft', gauss.to(torch.complex64), phasor)

        return field.cpu().numpy()


    def render_audio_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-tiled quadrature render. Returns (L, R) = (cos, sin) pair.

        L(t) = Σ A·fade·cos(φ(t))   — in-phase (identical to render_audio)
        R(t) = Σ A·fade·sin(φ(t))   — quadrature

        Each distinct phase state maps to a unique (L, R) point; no
        projection-induced interference or phase folding.  Both arrays are
        float64; caller normalises before writing.
        """
        import torch
        dev   = torch.device('cuda')
        n     = int(math.ceil(self._end * self._sr)) + 1
        t_g   = torch.linspace(0.0, self._end, n, dtype=torch.float64, device=dev)
        L_g   = torch.zeros(n, dtype=torch.float64, device=dev)
        R_g   = torch.zeros(n, dtype=torch.float64, device=dev)

        ts_g  = torch.tensor(self._ts, dtype=torch.float64, device=dev)
        te_g  = torch.tensor(self._te, dtype=torch.float64, device=dev)
        fs_g  = torch.tensor(self._fs, dtype=torch.float64, device=dev)
        cr_g  = torch.tensor(self._cr, dtype=torch.float64, device=dev)
        ph_g  = torch.tensor(self._ph, dtype=torch.float64, device=dev)
        am_g  = torch.tensor(self._am, dtype=torch.float64, device=dev)
        dur_g = (te_g - ts_g).clamp(min=0.0)
        fd_g  = torch.minimum(
            torch.tensor(self._fade, dtype=torch.float64, device=dev),
            dur_g * 0.25,
        ).clamp(min=1e-12)

        i0_seg = torch.searchsorted(t_g.contiguous(), ts_g.contiguous()).clamp(0, n)
        i1_seg = torch.searchsorted(t_g.contiguous(), te_g.contiguous(), right=True).clamp(0, n)

        S          = len(self._ts)
        TIME_CHUNK = 8192
        SEG_CHUNK  = 1024

        for tc0 in range(0, n, TIME_CHUNK):
            tc1  = min(tc0 + TIME_CHUNK, n)
            t_sl = t_g[tc0:tc1]

            for sc0 in range(0, S, SEG_CHUNK):
                sc1  = min(sc0 + SEG_CHUNK, S)
                i0_c = i0_seg[sc0:sc1]
                i1_c = i1_seg[sc0:sc1]
                if int(i1_c.max()) <= tc0 or int(i0_c.min()) >= tc1:
                    continue
                act = (i0_c < tc1) & (i1_c > tc0)
                if not act.any():
                    continue

                ts_c  = ts_g[sc0:sc1][act];   fs_c = fs_g[sc0:sc1][act]
                cr_c  = cr_g[sc0:sc1][act];   ph_c = ph_g[sc0:sc1][act]
                am_c  = am_g[sc0:sc1][act];   dur_c = dur_g[sc0:sc1][act]
                fd_c  = fd_g[sc0:sc1][act]

                dt   = t_sl[None, :] - ts_c[:, None]
                mask = (dt >= 0) & (dt <= dur_c[:, None])
                dt0  = dt;  dt1 = dur_c[:, None] - dt
                fi   = torch.where(dt0 < fd_c[:, None],
                                   0.5 * (1.0 - torch.cos(math.pi * dt0 / fd_c[:, None])),
                                   torch.ones_like(dt0))
                fo   = torch.where(dt1 < fd_c[:, None],
                                   0.5 * (1.0 - torch.cos(math.pi * dt1 / fd_c[:, None])),
                                   torch.ones_like(dt1))
                env  = am_c[:, None] * fi * fo * mask.double()
                phi  = ph_c[:, None] + PI2 * (
                    fs_c[:, None] * dt + 0.5 * cr_c[:, None] * dt * dt)

                L_g[tc0:tc1] += (env * torch.cos(phi)).sum(dim=0)
                R_g[tc0:tc1] += (env * torch.sin(phi)).sum(dim=0)

        return L_g.cpu().numpy(), R_g.cpu().numpy()

    def render_phase_gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-tiled amplitude-weighted instantaneous frequency field.

        Computes the amplitude-weighted mean instantaneous frequency at each
        output sample time.  For each active segment at time t:
            f_inst = f_start + chirp_rate * (t - t_start)

        Returns ``(f_inst, t_axis)`` as float64 arrays.  f_inst is zero where
        no segments are active.  This is the 'ray world' observable: phase
        flow without any folding onto a single real axis.

        To sonify: integrate f_inst to recover phase, then take cos.
        """
        import torch
        dev   = torch.device('cuda')
        n     = int(math.ceil(self._end * self._sr)) + 1
        t_g   = torch.linspace(0.0, self._end, n, dtype=torch.float64, device=dev)
        f_num = torch.zeros(n, dtype=torch.float64, device=dev)
        f_den = torch.zeros(n, dtype=torch.float64, device=dev)

        ts_g  = torch.tensor(self._ts, dtype=torch.float64, device=dev)
        te_g  = torch.tensor(self._te, dtype=torch.float64, device=dev)
        fs_g  = torch.tensor(self._fs, dtype=torch.float64, device=dev)
        cr_g  = torch.tensor(self._cr, dtype=torch.float64, device=dev)
        am_g  = torch.tensor(self._am, dtype=torch.float64, device=dev)
        dur_g = (te_g - ts_g).clamp(min=0.0)
        fd_g  = torch.minimum(
            torch.tensor(self._fade, dtype=torch.float64, device=dev),
            dur_g * 0.25,
        ).clamp(min=1e-12)

        i0_seg = torch.searchsorted(t_g.contiguous(), ts_g.contiguous()).clamp(0, n)
        i1_seg = torch.searchsorted(t_g.contiguous(), te_g.contiguous(), right=True).clamp(0, n)

        S          = len(self._ts)
        TIME_CHUNK = 8192
        SEG_CHUNK  = 1024

        for tc0 in range(0, n, TIME_CHUNK):
            tc1  = min(tc0 + TIME_CHUNK, n)
            t_sl = t_g[tc0:tc1]

            for sc0 in range(0, S, SEG_CHUNK):
                sc1  = min(sc0 + SEG_CHUNK, S)
                i0_c = i0_seg[sc0:sc1]
                i1_c = i1_seg[sc0:sc1]
                if int(i1_c.max()) <= tc0 or int(i0_c.min()) >= tc1:
                    continue
                act = (i0_c < tc1) & (i1_c > tc0)
                if not act.any():
                    continue

                ts_c  = ts_g[sc0:sc1][act];   fs_c = fs_g[sc0:sc1][act]
                cr_c  = cr_g[sc0:sc1][act];   am_c = am_g[sc0:sc1][act]
                dur_c = dur_g[sc0:sc1][act];  fd_c = fd_g[sc0:sc1][act]

                dt   = t_sl[None, :] - ts_c[:, None]
                mask = (dt >= 0) & (dt <= dur_c[:, None])
                dt0  = dt;  dt1 = dur_c[:, None] - dt
                fi   = torch.where(dt0 < fd_c[:, None],
                                   0.5 * (1.0 - torch.cos(math.pi * dt0 / fd_c[:, None])),
                                   torch.ones_like(dt0))
                fo   = torch.where(dt1 < fd_c[:, None],
                                   0.5 * (1.0 - torch.cos(math.pi * dt1 / fd_c[:, None])),
                                   torch.ones_like(dt1))
                env  = am_c[:, None] * fi * fo * mask.double()

                f_inst = fs_c[:, None] + cr_c[:, None] * dt  # [C, TC]
                f_num[tc0:tc1] += (env * f_inst).sum(dim=0)
                f_den[tc0:tc1] += env.sum(dim=0)

        f_out  = (f_num / f_den.clamp(min=1e-30)).cpu().numpy()
        t_axis = t_g.cpu().numpy()
        return f_out, t_axis

    def write_wav(self, path: str) -> None:
        import scipy.io.wavfile as _wav
        arr  = self.render_audio()
        peak = float(np.abs(arr).max()) or 1.0
        f32  = (arr / peak).astype(np.float32)
        _wav.write(path, int(self._sr), f32)
        print(f"Written {path}: {len(f32)} samples @ {int(self._sr)} Hz | "
              f"{len(self._ts)} segments | peak={peak:.4f}")

    def write_wav_stereo(self, path: str) -> None:
        """Write quadrature stereo WAV. L=cos, R=sin — no phase folding."""
        import scipy.io.wavfile as _wav
        L, R = self.render_audio_quadrature()
        peak = max(float(np.abs(L).max()), float(np.abs(R).max())) or 1.0
        stereo = np.stack([(L / peak).astype(np.float32),
                           (R / peak).astype(np.float32)], axis=1)
        _wav.write(path, int(self._sr), stereo)
        print(f"Written {path}: {stereo.shape[0]} samples @ {int(self._sr)} Hz | "
              f"stereo quadrature | peak={peak:.4f}")

    def write_wav_phase_gradient(self, path: str) -> None:
        """Sonify the phase gradient field: integrate f_inst → phase → cos.

        This is the 'ray world' audio: no interference, just phase flow.
        """
        import scipy.io.wavfile as _wav
        f_inst, t_axis = self.render_phase_gradient()
        dt   = float(t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 1.0 / self._sr
        # Cumulative-trapezoid integration of f_inst to recover phase.
        phase = np.zeros_like(f_inst)
        phase[1:] = np.cumsum(0.5 * (f_inst[:-1] + f_inst[1:]) * dt) * PI2
        audio = np.cos(phase).astype(np.float32)
        _wav.write(path, int(self._sr), audio)
        print(f"Written {path}: {len(audio)} samples @ {int(self._sr)} Hz | "
              f"phase-gradient world | f_mean={float(f_inst[f_inst>0].mean()):.1f} Hz"
              if f_inst.any() else f"Written {path}: {len(audio)} samples (silent)")

    def save_field_png(
        self,
        path:              str,
        n_time:            int   = 600,
        n_freq:            int   = 500,
        freq_min:          Optional[float] = None,
        freq_max:          Optional[float] = None,
        log_freq:          bool  = False,
        heisenberg_spread: float = 2.0,
        log_magnitude:     bool  = True,
        dpi:               int   = 150,
        summation:         str   = 'complex',
        amplitude_percentile: float = 100.0,
    ) -> None:
        """
        Save the complex field as a two-panel image:
          top:    magnitude (|Z|), log-scaled, colormap inferno
          bottom: phase (arg Z), colormap hsv — so you see the actual wavefronts

        Phase is only shown where magnitude exceeds 1% of peak (elsewhere black).
        amplitude_percentile: clip magnitude at this percentile before normalising.
            100 = use absolute max (default). 99 = clip top 1% so diffuse detail
            isn't drowned out by bright coherent interference spikes.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            raise ImportError("matplotlib required")

        field, t_ax, f_ax = self.render_complex_field(
            n_time, n_freq, freq_min, freq_max, log_freq, heisenberg_spread,
            summation=summation,
        )
        mag   = np.abs(field)                           # [n_freq, n_time]
        phase = np.angle(field)                         # [n_freq, n_time], range [-pi, pi]

        mag_plot = np.log1p(mag * 200.0) if log_magnitude else mag
        # Percentile clip: lets faint diffuse detail survive next to bright coherent peaks.
        if amplitude_percentile < 100.0:
            clip_val = float(np.percentile(mag_plot, amplitude_percentile))
            mag_plot = np.clip(mag_plot, 0.0, clip_val)
        pk = float(mag_plot.max())
        if pk > 0.0:
            mag_plot /= pk

        # Phase image masked where magnitude is negligible
        threshold  = float(mag.max()) * 0.01
        phase_disp = (phase + math.pi) / (2.0 * math.pi)   # [0, 1] for HSV hue
        phase_disp[mag < threshold] = 0.0

        # Build HSV image: hue=phase, saturation=1, value=mag (so dark where mag≈0)
        mag_norm   = mag / max(float(mag.max()), 1e-30)
        hsv        = np.stack([phase_disp, np.ones_like(phase_disp), mag_norm], axis=-1)
        rgb_phase  = mcolors.hsv_to_rgb(hsv)

        extent = [t_ax[0], t_ax[-1], f_ax[0], f_ax[-1]]
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=dpi, sharex=True)

        axes[0].imshow(mag_plot, aspect="auto", origin="lower", extent=extent,
                       cmap="inferno", interpolation="bilinear", vmin=0, vmax=1)
        axes[0].set_ylabel("Frequency (Hz)")
        axes[0].set_title(f"Complex field magnitude  ({len(self._ts)} segments)")

        axes[1].imshow(rgb_phase, aspect="auto", origin="lower", extent=extent,
                       interpolation="bilinear")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Frequency (Hz)")
        axes[1].set_title("Complex field phase  (hue=phase, brightness=magnitude)")

        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}  [{n_time}x{n_freq}]  "
              f"mag_peak={float(mag.max()):.4f}")

    def print_segments(self, max_print: int = 30) -> None:
        n = len(self._ts)
        print(f"  {'t_start':>8}  {'t_end':>8}  {'f_start':>9}  "
              f"{'chirp Hz/s':>11}  {'amplitude':>10}")
        print("  " + "-" * 58)
        for k in range(min(max_print, n)):
            fe = float(self._fs[k]) + float(self._cr[k]) * (float(self._te[k]) - float(self._ts[k]))
            print(f"  {self._ts[k]:8.3f}  {self._te[k]:8.3f}  {self._fs[k]:9.2f}->"
                  f"{fe:7.2f}  {self._cr[k]:11.2f}  {self._am[k]:10.5f}")
        if n > max_print:
            print(f"  ... ({n - max_print} more)")


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_fog_corridor(
    wav_path:    str   = "tf_fog_corridor.wav",
    png_path:    str   = "tf_fog_corridor.png",
    scene_end:   float = 5.0,
    f_min:       float = 80.0,
    f_max:       float = 2000.0,
    seed:        int   = 42,
    device:      str   = "cuda",
    batch_size:  int   = 5000,
    phase_mode:  str   = "physical",
    summation:   str   = "complex",
) -> None:
    """
    Dense fog corridor on GPU via TFRayGPU.
    All ray math runs on the GPU in batches of batch_size.
    """
    # Tuple format: (f_hz_or_t_sec, n_a, n_b, surface_absorption_loss)
    # Fresnel equations compute the R/T split from n1, n2, and angle of incidence.
    # surface_absorption_loss is the fraction of total energy absorbed at the interface.
    h_surfs = [
        (500.0,  1.0, 1.8, 0.05),
        (1200.0, 1.8, 1.0, 0.05),
    ]
    v_surfs = [
        (scene_end / 2.0, 1.0, 1.5, 0.05),
    ]
    tracer = TFRayGPU(
        scene_end        = scene_end,
        f_min            = f_min,
        f_max            = f_max,
        h_surfs          = h_surfs,
        v_surfs          = v_surfs,
        freq_mirror_loss = 0.05,
        time_mirror_loss = 0.05,
        scatter_rate     = 1.2,
        amp_loss         = 0.30,   # 70% of power distributed across n_scattered children
        n_scattered      = 3,
        max_chirp        = 1800.0,
        max_batch        = batch_size,
        max_segments     = 500_000,
        min_amplitude    = 0.02,
        max_bounces      = 200,
        device           = device,
        seed             = seed,
        phase_mode       = phase_mode,
    )

    freqs  = np.linspace(f_min + 50, f_max - 100, 6)
    chirps = np.linspace(-1600.0, 1600.0, 9)
    rows   = []
    for f0 in freqs:
        for vf0 in chirps:
            for vt0 in (1.0, -1.0):
                t0 = 0.0 if vt0 > 0 else scene_end
                rows.append([t0, float(f0), vt0, float(vf0), 0.0, 1.0])
    init_arr = np.array(rows, dtype=np.float64)
    print(f"  Launching {len(rows)} rays on {tracer.dev} (batch={batch_size})")
    all_segs = tracer.propagate(init_arr)
    print(f"  -> {len(all_segs)} chirp segments")

    r = TFRayScene(all_segs, scene_end=scene_end)
    r.write_wav(wav_path)
    r.save_field_png(
        png_path,
        freq_min=f_min, freq_max=f_max,
        n_time=800, n_freq=600,
        summation=summation,
    )


def demo_corridor_bounce(
    wav_path: str = "tf_corridor.wav",
    png_path: str = "tf_corridor.png",
) -> None:
    """
    No fog. Single ray with large chirp rate traverses corridor, bounces off
    frequency walls and refracts through a slab. Reflected branches kept.
    Tilted ridges in TF image show the geometric trajectory and Snell bending.
    """
    scene = TFScene(f_min=80.0, f_max=2000.0, time_mirror_loss=0.02)
    scene.add_surface(HorizontalSurface(
        f_hz=600.0, n_a=1.0, n_b=1.8,
        transmission_fraction=0.80, reflection_fraction=0.20,
    ))
    scene.add_surface(HorizontalSurface(
        f_hz=1400.0, n_a=1.8, n_b=1.0,
        transmission_fraction=0.80, reflection_fraction=0.20,
    ))

    # Ray sweeps at 500 Hz/s -- covers the 1920 Hz corridor in ~3.8 s
    initial = RayState2D(t=0.0, f=100.0, v_t=1.0, v_f=500.0, phi=0.0, A=1.0)
    segs    = TFPropagator(
        scene, scene_end=5.0, max_segments=512, max_bounces=50,
    ).propagate(initial)

    print(f"Corridor bounce: {len(segs)} segments")
    r = TFRayScene(segs, scene_end=5.0)
    r.print_segments()
    r.write_wav(wav_path)
    r.save_field_png(png_path, freq_min=80, freq_max=2000)


def demo_time_mirror(
    wav_path: str = "tf_time_mirror.wav",
    png_path: str = "tf_time_mirror.png",
) -> None:
    """
    Pure temporal mirror demo (no fog, no frequency refraction).
    One forward and one backward ray at large chirp rates bounce between
    t=0 and t=3 s.  Time-reversed ridges zigzag in real time.
    """
    scene   = TFScene(f_min=100.0, f_max=2000.0, freq_mirror_loss=0.03, time_mirror_loss=0.03)
    prop    = TFPropagator(scene, scene_end=3.0, max_segments=512, min_amplitude=0.02)
    all_segs: List[ChirpSegment] = []

    for vt0, vf0 in [(1.0, 400.0), (-1.0, -400.0), (1.0, -300.0)]:
        t0 = 0.0 if vt0 > 0 else 3.0
        initial = RayState2D(t=t0, f=400.0, v_t=vt0, v_f=vf0, phi=0.0, A=0.7)
        all_segs.extend(prop.propagate(initial))

    print(f"Time-mirror: {len(all_segs)} segments")
    r = TFRayScene(all_segs, scene_end=3.0)
    r.print_segments()
    r.write_wav(wav_path)
    r.save_field_png(png_path, freq_min=100, freq_max=2000)


def demo_dispersion(
    wav_path: str = "tf_dispersion.wav",
    png_path: str = "tf_dispersion.png",
) -> None:
    """
    5 rays at different chirp rates pass through a vertical refractive slab.
    Snell's law at the slab changes each ray's chirp rate differently
    (dispersion).  Reflected branches from the slab surfaces are also kept.
    """
    scene = TFScene(f_min=60.0, f_max=1800.0)
    scene.add_surface(VerticalSurface(
        t_sec=1.0, n_a=1.0, n_b=2.0,
        transmission_fraction=0.78, reflection_fraction=0.22,
    ))
    scene.add_surface(VerticalSurface(
        t_sec=2.5, n_a=2.0, n_b=1.0,
        transmission_fraction=0.78, reflection_fraction=0.22,
    ))

    all_segs: List[ChirpSegment] = []
    prop = TFPropagator(scene, scene_end=4.0, max_segments=256)
    for vf0 in (-600.0, -300.0, 0.0, 300.0, 600.0):
        initial = RayState2D(t=0.0, f=300.0, v_t=1.0, v_f=vf0, phi=0.0, A=0.6)
        all_segs.extend(prop.propagate(initial))

    print(f"Dispersion: {len(all_segs)} segments")
    TFRayScene(all_segs, scene_end=4.0).save_field_png(
        png_path, freq_min=60, freq_max=1800,
    )
    TFRayScene(all_segs, scene_end=4.0).write_wav(wav_path)


# ---------------------------------------------------------------------------
# Procedural mask-PNG generators (RGB: R=reflect, G=refract, B=diffuse)
# ---------------------------------------------------------------------------

def _gen_casing_potential_png(
    path:      str,
    width_px:  int   = 256,
    height_px: int   = 256,
    mode:      str   = 'sinusoidal',   # 'sinusoidal' | 'parabolic' | 'flat'
    phi_max:   float = 1.0,
) -> None:
    """Generate a scalar potential field PNG for a grounded rectangular casing.

    The casing walls are at ground (phi = 0); the interior rises to phi_max.
    E = -∇phi then points from the interior toward the walls.

    mode='sinusoidal' (default):
        phi(u, v) = phi_max * sin(π u) * sin(π v)   where u, v ∈ [0, 1]
        This is the exact fundamental Laplace eigenmode inside a rectangle
        with Dirichlet (phi=0) boundary conditions — the most physical choice.

    mode='parabolic':
        phi(u, v) = phi_max * 4*u*(1-u) * 4*v*(1-v)
        Wider flat plateau in the centre; steeper gradient near the walls.

    mode='flat':
        phi = phi_max everywhere except a 1-pixel border.
        Maximally uniform interior, sharp boundary gradient only.
    """
    from PIL import Image as _PIL
    u = np.linspace(0.0, 1.0, width_px,  dtype=np.float32)
    v = np.linspace(0.0, 1.0, height_px, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    if mode == 'sinusoidal':
        phi = np.sin(np.pi * uu) * np.sin(np.pi * vv)
    elif mode == 'parabolic':
        phi = 4.0 * uu * (1.0 - uu) * 4.0 * vv * (1.0 - vv)
    else:  # 'flat'
        phi = np.ones((height_px, width_px), dtype=np.float32)
        phi[0, :]  = 0.0;  phi[-1, :] = 0.0
        phi[:, 0]  = 0.0;  phi[:, -1] = 0.0
    phi = np.clip(phi * phi_max, 0.0, 1.0)
    arr = (phi * 255.0).astype(np.uint8)
    _PIL.fromarray(arr, 'L').save(path)


def _gen_casing_potential_png(
    path:      str,
    width_px:  int   = 256,
    height_px: int   = 256,
    mode:      str   = 'sinusoidal',   # 'sinusoidal' | 'parabolic' | 'flat'
    phi_max:   float = 1.0,
) -> None:
    """Generate a scalar potential field PNG for a grounded rectangular casing.

    The casing walls are at ground (phi = 0); the interior rises to phi_max.
    E = -∇phi then points from the interior toward the walls.

    mode='sinusoidal' (default):
        phi(u, v) = phi_max * sin(π u) * sin(π v)   where u, v ∈ [0, 1]
        This is the exact fundamental Laplace eigenmode inside a rectangle
        with Dirichlet (phi=0) boundary conditions — the most physical choice.

    mode='parabolic':
        phi(u, v) = phi_max * 4*u*(1-u) * 4*v*(1-v)
        Wider flat plateau in the centre; steeper gradient near the walls.

    mode='flat':
        phi = phi_max everywhere except a 1-pixel border.
        Maximally uniform interior, sharp boundary gradient only.
    """
    from PIL import Image as _PIL
    u = np.linspace(0.0, 1.0, width_px,  dtype=np.float32)
    v = np.linspace(0.0, 1.0, height_px, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    if mode == 'sinusoidal':
        phi = np.sin(np.pi * uu) * np.sin(np.pi * vv)
    elif mode == 'parabolic':
        phi = 4.0 * uu * (1.0 - uu) * 4.0 * vv * (1.0 - vv)
    else:  # 'flat'
        phi = np.ones((height_px, width_px), dtype=np.float32)
        phi[0, :]  = 0.0;  phi[-1, :] = 0.0
        phi[:, 0]  = 0.0;  phi[:, -1] = 0.0
    phi = np.clip(phi * phi_max, 0.0, 1.0)
    arr = (phi * 255.0).astype(np.uint8)
    _PIL.fromarray(arr, 'L').save(path)


def _gen_baffle_jaw_png(
    path: str,
    jaw_frac:  float = 0.35,  # fraction of height absorbed on EACH side (top and bottom)
    width_px:  int   = 8,
    height_px: int   = 256,
) -> None:
    """Absorbing baffle jaw wall.  Alpha=255 on top jaw_frac and bottom jaw_frac;
    alpha=0 in the center gap.  RGB=0 (no reflect/refract/diffuse contribution).
    Placed at a specific T position as a thin vertical slab; f0/f1 span full F range.
    Only rays whose F lies in the center gap pass through; all others are absorbed."""
    from PIL import Image as _PIL
    arr    = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    cut    = max(1, int(round(jaw_frac * height_px)))
    arr[:cut,  :, 3] = 255   # bottom jaw  (row 0 = f_min after load flip)
    arr[-cut:, :, 3] = 255   # top jaw
    _PIL.fromarray(arr, 'RGBA').save(path)


def _gen_slotted_wall_png(
    path: str,
    n_slots:        int   = 3,
    slot_frac:      float = 0.10,   # each slot height as fraction of total
    width_px:       int   = 32,
    height_px:      int   = 256,
    reflect:        float = 0.92,
    refract:        float = 0.0,
    diffuse:        float = 0.06,
) -> None:
    """Vertical wall, mostly reflective (R), with n_slots transparent gaps."""
    from PIL import Image as _PIL
    arr = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    arr[:, :, 0] = int(reflect * 255)
    arr[:, :, 1] = int(refract * 255)
    arr[:, :, 2] = int(diffuse * 255)
    slot_h   = max(1, int(height_px * slot_frac))
    spacing  = height_px // (n_slots + 1)
    for i in range(n_slots):
        cy = (i + 1) * spacing
        y0, y1 = max(0, cy - slot_h // 2), min(height_px, cy + slot_h // 2)
        arr[y0:y1, :, :] = 0
    _PIL.fromarray(arr, 'RGB').save(path)


def _gen_diffuse_backstop_png(
    path: str,
    width_px:   int   = 32,
    height_px:  int   = 256,
    reflect:    float = 0.05,
    refract:    float = 0.0,
    diffuse:    float = 0.90,
) -> None:
    """Flat diffusive back-wall (high B, low R)."""
    from PIL import Image as _PIL
    arr = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    arr[:, :, 0] = int(reflect * 255)
    arr[:, :, 1] = int(refract * 255)
    arr[:, :, 2] = int(diffuse * 255)
    _PIL.fromarray(arr, 'RGB').save(path)


def _gen_sphere_png(
    path: str,
    width_px:    int   = 256,
    height_px:   int   = 256,
    radius_frac: float = 0.38,
    edge_sharpness: float = 6.0,   # higher → harder edge
    reflect:     float = 0.15,
    refract:     float = 0.55,
    diffuse:     float = 0.75,
) -> None:
    """Soft-edged sphere: reflect rim, refract interior creates lensing, diffuse bulk."""
    from PIL import Image as _PIL
    yy, xx = np.mgrid[0:height_px, 0:width_px].astype(np.float32)
    cy, cx  = (height_px - 1) / 2.0, (width_px - 1) / 2.0
    r_norm  = np.sqrt(((xx - cx) / (radius_frac * width_px  / 2))**2 +
                      ((yy - cy) / (radius_frac * height_px / 2))**2)
    # Smooth interior mask (1 at centre, 0 outside radius)
    interior = np.clip(1.0 - r_norm, 0.0, 1.0) ** (1.0 / max(edge_sharpness, 0.01))
    # Rim emphasis for reflect (peaks at r_norm ≈ 1, falls inside and outside)
    rim = np.exp(-edge_sharpness * (r_norm - 0.85) ** 2)
    arr = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    arr[:, :, 0] = (np.clip(rim,      0, 1) * reflect * 255).astype(np.uint8)
    arr[:, :, 1] = (np.clip(interior, 0, 1) * refract * 255).astype(np.uint8)
    arr[:, :, 2] = (np.clip(interior, 0, 1) * diffuse * 255).astype(np.uint8)
    _PIL.fromarray(arr, 'RGB').save(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TF ray engine — bidirectional geometric ray tracer in time-frequency space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Subcommand arguments must come AFTER the subcommand name:\n"
            "  python tf_ray_engine.py fog --n-rays 1024 --duration 8\n"
            "  python tf_ray_engine.py bounce --f0 200 --vf0 800\n"
            "  python tf_ray_engine.py mirror --n-rays 5 --chirp 600\n"
            "  python tf_ray_engine.py dispersion --n-rays 11 --slab-n 2.5\n"
            "\nRun 'python tf_ray_engine.py <subcommand> --help' for all options."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_render_args(p, *, n_time=600, n_freq=500, wav=None, png=None):
        """Args shared by every subcommand's render stage."""
        if wav:
            p.add_argument("--wav", default=wav, metavar="PATH", help="Output WAV path")
        if png:
            p.add_argument("--png", default=png, metavar="PATH", help="Output PNG path")
        p.add_argument("--n-time",       type=int,   default=n_time,  metavar="N",    help="Time columns in TF image")
        p.add_argument("--n-freq",       type=int,   default=n_freq,  metavar="N",    help="Frequency rows in TF image")
        p.add_argument("--heisenberg",   type=float, default=2.0,     metavar="F",    help="TF Gaussian spread = heisenberg / segment_duration")
        p.add_argument("--log-freq",     action="store_true",                          help="Log-scale frequency axis")
        p.add_argument("--log-magnitude",action="store_true",                          help="Log-scale TF magnitude (log1p)")
        p.add_argument("--amplitude-percentile", type=float, default=100.0, metavar="PCT",
                       help="Clip TF magnitude at this percentile before normalising (e.g. 99 "
                            "prevents a few bright coherent peaks from washing out diffuse detail)")
        p.add_argument("--dpi",          type=int,   default=150,     metavar="N",    help="PNG DPI")
        p.add_argument("--sample-rate",  type=float, default=48000.0, metavar="HZ",   help="Audio output sample rate")
        p.add_argument("--fade",         type=float, default=0.002,   metavar="S",    help="Segment fade-in/out duration")
        p.add_argument("--summation",    default="complex", metavar="MODE",
                       choices=["complex", "magnitude"],
                       help="complex: sum phasors with interference; magnitude: sum amplitudes incoherently")
        p.add_argument("--no-wav",       action="store_true",                          help="Skip audio render")

    # --- fog ---
    p = sub.add_parser("fog",
        help="Dense fog corridor: rays scatter stochastically and refract at horizontal/vertical surfaces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_render_args(p, n_time=800, n_freq=600,
                     wav="tf_fog_corridor.wav", png="tf_fog_corridor.png")
    # scene geometry
    p.add_argument("--duration",         type=float, default=5.0,     metavar="S",    help="Scene length in seconds")
    p.add_argument("--fmin",             type=float, default=80.0,    metavar="HZ",   help="Lower frequency boundary (hard mirror)")
    p.add_argument("--fmax",             type=float, default=2000.0,  metavar="HZ",   help="Upper frequency boundary (hard mirror)")
    p.add_argument("--freq-mirror-loss", type=float, default=0.05,    metavar="F",    help="Amplitude² loss [0,1) at fmin/fmax mirrors")
    p.add_argument("--time-mirror-loss", type=float, default=0.05,    metavar="F",    help="Amplitude² loss [0,1) at t=0/t=end mirrors")
    # horizontal refractive surfaces (two: lo boundary and hi boundary of slab)
    p.add_argument("--h-surf-lo-f",      type=float, default=500.0,   metavar="HZ",   help="Frequency of lower horizontal refractive surface")
    p.add_argument("--h-surf-hi-f",      type=float, default=1200.0,  metavar="HZ",   help="Frequency of upper horizontal refractive surface")
    p.add_argument("--h-surf-n-slab",    type=float, default=1.8,     metavar="N",    help="Refractive index inside the horizontal slab (outside=1.0)")
    p.add_argument("--h-surf-loss",      type=float, default=0.0,     metavar="F",    help="Surface absorption loss at each horizontal interface")
    # vertical refractive surface
    p.add_argument("--v-surf-t",         type=float, default=None,    metavar="S",    help="Time position of vertical refractive surface (default: duration/2)")
    p.add_argument("--v-surf-n",         type=float, default=1.5,     metavar="N",    help="Refractive index on far side of vertical surface (near=1.0)")
    p.add_argument("--v-surf-loss",      type=float, default=0.0,     metavar="F",    help="Surface absorption loss at vertical interface")
    # fog / propagation
    p.add_argument("--scatter-rate",     type=float, default=1.2,     metavar="R",    help="Mean scatter events per elapsed second (Poisson rate)")
    p.add_argument("--amp-loss",         type=float, default=0.30,    metavar="F",    help="Fraction of amplitude² lost per scatter event")
    p.add_argument("--n-scattered",      type=int,   default=2,       metavar="N",    help="Child rays spawned per scatter event")
    p.add_argument("--max-chirp",        type=float, default=1800.0,  metavar="HZ/S", help="Max |v_f| (chirp rate) assigned to scattered children")
    p.add_argument("--min-amplitude",    type=float, default=0.03,    metavar="F",    help="Prune rays below this amplitude")
    p.add_argument("--max-bounces",      type=int,   default=300,     metavar="N",    help="Prune rays exceeding this many surface/mirror interactions")
    p.add_argument("--max-segments",     type=int,   default=500_000, metavar="N",    help="Total segment budget; propagation stops when reached")
    p.add_argument("--batch-size",       type=int,   default=5000,    metavar="N",    help="Max rays held on GPU simultaneously per step")
    # initial rays
    p.add_argument("--n-rays",           type=int,   default=108,     metavar="N",    help="Total rays launched (split evenly forward/backward, gridded across f and chirp)")
    p.add_argument("--chirp-range",      type=float, default=1600.0,  metavar="HZ/S", help="Seed chirp rates span ±chirp-range")
    # GPU / reproducibility
    p.add_argument("--device",           default="cuda",              metavar="DEV",  help="Torch device (cuda or cpu)")
    p.add_argument("--seed",             type=int,   default=42,                      help="RNG seed for fog scatter")
    p.add_argument("--phase-mode",         default="physical",          metavar="MODE",
                   choices=["physical", "coherent"],
                   help="physical: random initial+scatter phases → interference speckle; coherent: all phi=0")
    # density field
    p.add_argument("--density-png",        default=None,                metavar="PATH",
                   help="PNG loaded as greyscale; brightness → local refractive index. "
                        "Rays refract continuously through the field. "
                        "Image is centered in the scene (t axis) and spans fmin→fmax.")
    p.add_argument("--density-scale",      type=float, default=1.0,     metavar="F",
                   help="Max extra refractive index added by brightest pixel (black=n=1, white=n=1+scale)")
    p.add_argument("--density-sensitivity",type=float, default=8000.0,  metavar="F",
                   help="Scales how strongly the n gradient bends ray chirp rate per step")
    p.add_argument("--density-step",       type=float, default=None,    metavar="S",
                   help="Integration step size through density field (default: duration/300)")

    # --- bounce ---
    p = sub.add_parser("bounce",
        help="Single chirp ray bouncing between two horizontal refractive surfaces (frequency slab)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_render_args(p, wav="tf_corridor.wav", png="tf_corridor.png")
    p.add_argument("--duration",         type=float, default=5.0,    metavar="S",    help="Scene length in seconds")
    p.add_argument("--fmin",             type=float, default=80.0,   metavar="HZ",   help="Lower frequency boundary")
    p.add_argument("--fmax",             type=float, default=2000.0, metavar="HZ",   help="Upper frequency boundary")
    p.add_argument("--f0",               type=float, default=100.0,  metavar="HZ",   help="Initial ray frequency")
    p.add_argument("--vf0",              type=float, default=500.0,  metavar="HZ/S", help="Initial chirp rate (df/dt_elapsed)")
    p.add_argument("--amplitude",        type=float, default=1.0,    metavar="F",    help="Initial ray amplitude")
    p.add_argument("--slab-lo",          type=float, default=600.0,  metavar="HZ",   help="Frequency of lower slab boundary")
    p.add_argument("--slab-hi",          type=float, default=1400.0, metavar="HZ",   help="Frequency of upper slab boundary")
    p.add_argument("--slab-n",           type=float, default=1.8,    metavar="N",    help="Refractive index inside the slab (outside=1.0)")
    p.add_argument("--slab-t-frac",      type=float, default=0.80,   metavar="F",    help="Transmission amplitude fraction at each slab surface")
    p.add_argument("--slab-r-frac",      type=float, default=0.20,   metavar="F",    help="Reflection amplitude fraction at each slab surface")
    p.add_argument("--time-mirror-loss", type=float, default=0.02,   metavar="F",    help="Amplitude² loss at t=0/t=end time mirrors")
    p.add_argument("--min-amplitude",    type=float, default=0.01,   metavar="F",    help="Prune rays below this amplitude")
    p.add_argument("--max-segments",     type=int,   default=512,    metavar="N",    help="Max segments before propagation stops")
    p.add_argument("--max-bounces",      type=int,   default=50,     metavar="N",    help="Max surface interactions per ray")

    # --- mirror ---
    p = sub.add_parser("mirror",
        help="Three chirp rays undergoing time-reversal at temporal mirrors (zigzag demo)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_render_args(p, wav="tf_time_mirror.wav", png="tf_time_mirror.png")
    p.add_argument("--duration",         type=float, default=3.0,    metavar="S",    help="Scene length in seconds")
    p.add_argument("--fmin",             type=float, default=100.0,  metavar="HZ",   help="Lower frequency boundary")
    p.add_argument("--fmax",             type=float, default=2000.0, metavar="HZ",   help="Upper frequency boundary")
    p.add_argument("--f0",               type=float, default=400.0,  metavar="HZ",   help="Starting frequency for all rays")
    p.add_argument("--chirp",            type=float, default=400.0,  metavar="HZ/S", help="|v_f| seed chirp rate; rays are evenly spaced ±chirp across n-rays")
    p.add_argument("--n-rays",           type=int,   default=3,      metavar="N",    help="Number of rays, evenly spaced in chirp from -chirp to +chirp")
    p.add_argument("--amplitude",        type=float, default=0.7,    metavar="F",    help="Initial amplitude for all rays")
    p.add_argument("--freq-mirror-loss", type=float, default=0.03,   metavar="F",    help="Amplitude² loss at frequency mirrors")
    p.add_argument("--time-mirror-loss", type=float, default=0.03,   metavar="F",    help="Amplitude² loss at time mirrors")
    p.add_argument("--min-amplitude",    type=float, default=0.02,   metavar="F",    help="Prune rays below this amplitude")
    p.add_argument("--max-segments",     type=int,   default=512,    metavar="N",    help="Max segments before propagation stops")

    # --- dispersion ---
    p = sub.add_parser("dispersion",
        help="Fan of chirp rays passing through a vertical refractive slab (time-domain lens)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_render_args(p, wav="tf_dispersion.wav", png="tf_dispersion.png")
    p.add_argument("--duration",         type=float, default=4.0,    metavar="S",    help="Scene length in seconds")
    p.add_argument("--fmin",             type=float, default=60.0,   metavar="HZ",   help="Lower frequency boundary")
    p.add_argument("--fmax",             type=float, default=1800.0, metavar="HZ",   help="Upper frequency boundary")
    p.add_argument("--f0",               type=float, default=300.0,  metavar="HZ",   help="Center frequency of initial ray fan")
    p.add_argument("--chirp-range",      type=float, default=600.0,  metavar="HZ/S", help="Fan spans chirp rates from -chirp-range to +chirp-range")
    p.add_argument("--n-rays",           type=int,   default=5,      metavar="N",    help="Number of rays in the fan")
    p.add_argument("--amplitude",        type=float, default=0.6,    metavar="F",    help="Initial amplitude for each fan ray")
    p.add_argument("--slab-lo",          type=float, default=1.0,    metavar="S",    help="Time position of slab left wall")
    p.add_argument("--slab-hi",          type=float, default=2.5,    metavar="S",    help="Time position of slab right wall")
    p.add_argument("--slab-n",           type=float, default=2.0,    metavar="N",    help="Refractive index inside the slab (outside=1.0)")
    p.add_argument("--slab-t-frac",      type=float, default=0.78,   metavar="F",    help="Transmission amplitude fraction at each slab wall")
    p.add_argument("--slab-r-frac",      type=float, default=0.22,   metavar="F",    help="Reflection amplitude fraction at each slab wall")
    p.add_argument("--max-segments",     type=int,   default=256,    metavar="N",    help="Max segments before propagation stops")

    # --- plasma ---
    p = sub.add_parser("plasma",
        help="Charged-particle ray tracer: PNG mask objects cause impact ionisation, "
             "charge accumulates, magnetic field curves ray trajectories, "
             "synchrotron radiation drains amplitude into bright decaying arcs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_render_args(p, n_time=800, n_freq=600, wav="tf_plasma.wav", png="tf_plasma.png")
    p.add_argument("--duration",          type=float, default=5.0,     metavar="S",    help="Scene length in seconds")
    p.add_argument("--fmin",              type=float, default=80.0,    metavar="HZ",   help="Lower frequency boundary")
    p.add_argument("--fmax",              type=float, default=2000.0,  metavar="HZ",   help="Upper frequency boundary")
    p.add_argument("--freq-mirror-loss",  type=float, default=0.02,    metavar="F",    help="Amplitude² loss at fmin/fmax mirrors")
    p.add_argument("--time-mirror-loss",  type=float, default=0.02,    metavar="F",    help="Amplitude² loss at t=0/t=end mirrors")
    # objects: repeated --object flags, each encoding all per-object properties
    p.add_argument("--objects",           nargs='+',  default=[],      metavar="PNG",  help="One or more mask PNGs (greyscale, white=solid)")
    p.add_argument("--object-t0",         nargs='+',  type=float,      metavar="S",    help="Left time edge of each object (default: 25%% of duration)")
    p.add_argument("--object-t1",         nargs='+',  type=float,      metavar="S",    help="Right time edge of each object (default: 75%% of duration)")
    p.add_argument("--object-f0",         nargs='+',  type=float,      metavar="HZ",   help="Bottom freq edge of each object (default: fmin+10%%)")
    p.add_argument("--object-f1",         nargs='+',  type=float,      metavar="HZ",   help="Top freq edge of each object (default: fmax-10%%)")
    p.add_argument("--object-reflect",    nargs='+',  type=float,      metavar="F",    help="Reflection fraction per object (default: 0.3)")
    p.add_argument("--object-diffuse",    nargs='+',  type=float,      metavar="F",    help="Diffusion/charge fraction per object (default: 0.5)")
    p.add_argument("--object-boost",      nargs='+',  type=float,      metavar="F",    help="Amplitude multiplier on diffuse impact per object (default: 4.0)")
    p.add_argument("--object-charge-rate",nargs='+',  type=float,      metavar="F",    help="Charge added per elapsed-second inside mask per object (default: 0.8)")
    # magnetic / radiation physics
    p.add_argument("--B-field",           type=float, default=0.001,   metavar="F",    help="Magnetic field strength (Lorentz rotation: ωc=charge*B, exact circular orbits)")
    p.add_argument("--synchrotron",       type=float, default=0.0001,  metavar="F",    help="Synchrotron radiation coefficient; dA/dτ = -A * k * charge² * f²")
    p.add_argument("--arc-angle",         type=float, default=0.05,    metavar="RAD",  help="Radians of velocity rotation per magnetic arc event (smaller = more accurate spiral segments, more segments)")
    p.add_argument("--initial-charge",    type=float, default=1.0,     metavar="F",    help="Charge assigned to each ray at birth; enables B-field spiraling without mask objects (0 = no initial charge, requires objects to ionise)")
    p.add_argument("--plasma-step",       type=float, default=None,    metavar="S",    help="Continuous integration step for plasma physics (default: duration/300)")
    # focusing baffles (synchrotron-style |sin| absorbing humps on top and bottom edges)
    p.add_argument("--baffles",           action="store_true",                          help="Add sinusoidal absorbing baffles on top and bottom edges to create an emission chamber")
    p.add_argument("--baffle-amp-frac",   type=float, default=0.20,    metavar="F",    help="Baffle depth as fraction of full frequency span (default 0.20 = 20%% each edge)")
    p.add_argument("--baffle-humps",      type=int,   default=1,       metavar="N",    help="Number of enclosed field cells between the baffles (n_walls = N+1; default 1 = two walls, one cell)")
    # emitter: left-side point source with Gaussian spread
    p.add_argument("--left-emitter",      action="store_true",                          help="Replace grid rays with a left-wall point emitter (Gaussian f and chirp spread, forward only)")
    p.add_argument("--emitter-f",         type=float, default=None,    metavar="HZ",   help="Emitter centre frequency (default: midpoint of fmin/fmax)")
    p.add_argument("--emitter-f-spread",  type=float, default=150.0,   metavar="HZ",   help="1-sigma Gaussian spread in frequency (Hz)")
    p.add_argument("--emitter-chirp-spread", type=float, default=300.0, metavar="HZ/S", help="1-sigma Gaussian spread in chirp rate (Hz/s) — angular divergence of beam")
    p.add_argument("--emitter-velocity-angle-spread", type=float, default=0.0, metavar="RAD",
                   help="± uniform spread in initial velocity angle (rad) per ray; rotates each "
                        "ray's (vt,vf) randomly around the cyclotron orbit circle at birth, "
                        "breaking beam-envelope lockstep oscillations (e.g. 0.3 ≈ ±17°, π ≈ full circle)")
    # rays
    p.add_argument("--n-rays",            type=int,   default=256,     metavar="N",    help="Total initial rays (grid or Gaussian emitter count)")
    p.add_argument("--chirp-range",       type=float, default=1200.0,  metavar="HZ/S", help="Seed chirp rates span ±chirp-range (grid mode only)")
    p.add_argument("--max-segments",      type=int,   default=500_000, metavar="N",    help="Total segment budget")
    p.add_argument("--max-bounces",       type=int,   default=500,     metavar="N",    help="Max surface interactions per ray")
    p.add_argument("--min-amplitude",     type=float, default=0.01,    metavar="F",    help="Prune rays below this amplitude")
    p.add_argument("--batch-size",        type=int,   default=5000,    metavar="N",    help="Max rays per GPU batch")
    p.add_argument("--seed",              type=int,   default=42,                      help="RNG seed")
    p.add_argument("--device",            default="cuda",              metavar="DEV",  help="Torch device")
    p.add_argument("--phase-mode",        default="physical",          metavar="MODE",
                   choices=["physical", "coherent"],
                   help="physical: random phases; coherent: all phi=0")
    # particle species
    p.add_argument("--particle",          default="positron",          metavar="SPECIES",
                   choices=["electron", "positron"],
                   help="particle species: 'positron' → charge +1 (clockwise orbit), "
                        "'electron' → charge −1 (counter-clockwise orbit, reverses E-force)")
    # grounded-casing potential field
    p.add_argument("--casing",            action="store_true",
                   help="Add a grounded-casing potential field (phi=0 at walls, max at centre). "
                        "E = -∇phi is sampled on GPU and applies a Lorentz electric force on "
                        "charged rays, confining them to the interior.")
    p.add_argument("--casing-phi-mode",   default="sinusoidal",       metavar="MODE",
                   choices=["sinusoidal", "parabolic", "flat"],
                   help="Shape of the casing potential field")
    p.add_argument("--casing-charge-scale", type=float, default=500.0, metavar="F",
                   help="E-field force multiplier (charge * casing_charge_scale * E); "
                        "higher values → stronger confinement by the casing walls")

    # --- resonator ---
    p = sub.add_parser("resonator",
        help="Coherent point emitter → slotted mirror-box wall → diffuse sphere + backstop. "
             "All geometry is generated procedurally; no external PNGs required.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_render_args(p, n_time=900, n_freq=700, wav="tf_resonator.wav", png="tf_resonator.png")
    p.add_argument("--duration",         type=float, default=5.0,     metavar="S",   help="Scene length in seconds")
    p.add_argument("--fmin",             type=float, default=200.0,   metavar="HZ",  help="Lower frequency boundary")
    p.add_argument("--fmax",             type=float, default=2200.0,  metavar="HZ",  help="Upper frequency boundary")
    p.add_argument("--time-mirror-loss", type=float, default=0.005,   metavar="F",   help="Amplitude² loss at t=0/t=end time mirrors (left mirror of the box)")
    p.add_argument("--freq-mirror-loss", type=float, default=0.005,   metavar="F",   help="Amplitude² loss at fmin/fmax frequency mirrors")
    # emitter
    p.add_argument("--emitter-t",        type=float, default=0.35,    metavar="S",   help="Emitter time position (s) — should be inside the mirror box")
    p.add_argument("--emitter-f",        type=float, default=1000.0,  metavar="HZ",  help="Emitter centre frequency")
    p.add_argument("--beam-f-spread",    type=float, default=150.0,   metavar="HZ",  help="±frequency spread of beam (Hz) — width of beam in frequency")
    p.add_argument("--beam-chirp-spread",type=float, default=250.0,   metavar="HZ/S",help="±chirp-rate spread of beam (Hz/s) — angular spread")
    p.add_argument("--beam-rays",        type=int,   default=5,       metavar="N",   help="Grid points per axis (beam-rays² total forward rays)")
    # slotted wall geometry
    p.add_argument("--wall-t",           type=float, default=1.6,     metavar="S",   help="Centre time of the slotted wall")
    p.add_argument("--wall-thickness",   type=float, default=0.07,    metavar="S",   help="Wall thickness in seconds")
    p.add_argument("--wall-n-slots",     type=int,   default=3,       metavar="N",   help="Number of transmission slots in the wall")
    p.add_argument("--wall-slot-frac",   type=float, default=0.10,    metavar="F",   help="Each slot height as fraction of full frequency range")
    p.add_argument("--wall-reflect",     type=float, default=0.92,    metavar="F",   help="Peak reflectivity of the solid wall material")
    p.add_argument("--wall-diffuse",     type=float, default=0.06,    metavar="F",   help="Peak diffusivity of the solid wall material")
    # second (right) wall — catches rays that emerge and bounce back
    p.add_argument("--rwall-t",          type=float, default=3.8,     metavar="S",   help="Centre time of the right diffusive backstop wall")
    p.add_argument("--rwall-thickness",  type=float, default=0.07,    metavar="S",   help="Right wall thickness")
    p.add_argument("--rwall-reflect",    type=float, default=0.05,    metavar="F",   help="Peak reflectivity of the right wall")
    p.add_argument("--rwall-diffuse",    type=float, default=0.90,    metavar="F",   help="Peak diffusivity of the right wall")
    # sphere
    p.add_argument("--sphere-t",         type=float, default=2.6,     metavar="S",   help="Centre time of the diffusive sphere")
    p.add_argument("--sphere-f",         type=float, default=1000.0,  metavar="HZ",  help="Centre frequency of the sphere")
    p.add_argument("--sphere-dt",        type=float, default=0.6,     metavar="S",   help="Sphere half-width in time (radius in seconds)")
    p.add_argument("--sphere-df",        type=float, default=500.0,   metavar="HZ",  help="Sphere half-height in frequency (radius in Hz)")
    p.add_argument("--sphere-reflect",   type=float, default=0.18,    metavar="F",   help="Sphere rim reflectivity")
    p.add_argument("--sphere-refract",   type=float, default=0.55,    metavar="F",   help="Sphere interior refractive contribution (lensing)")
    p.add_argument("--sphere-diffuse",   type=float, default=0.80,    metavar="F",   help="Sphere bulk diffusivity")
    p.add_argument("--sphere-refract-sensitivity", type=float, default=3000.0, metavar="F",
                   help="How strongly the sphere G-gradient bends chirp rate (Hz/s per unit per step)")
    # propagation
    p.add_argument("--max-segments",     type=int,   default=500_000, metavar="N",   help="Total segment budget")
    p.add_argument("--max-bounces",      type=int,   default=600,     metavar="N",   help="Max interactions per ray")
    p.add_argument("--min-amplitude",    type=float, default=0.008,   metavar="F",   help="Prune rays below this amplitude")
    p.add_argument("--batch-size",       type=int,   default=5000,    metavar="N",   help="Max rays per GPU batch")
    p.add_argument("--seed",             type=int,   default=42,                     help="RNG seed")
    p.add_argument("--device",           default="cuda",              metavar="DEV", help="Torch device")

    args = parser.parse_args()

    # ---- fog ----
    if args.cmd == "fog":
        v_surf_t = args.v_surf_t if args.v_surf_t is not None else args.duration / 2.0
        h_surfs  = [
            (args.h_surf_lo_f, 1.0,            args.h_surf_n_slab, args.h_surf_loss),
            (args.h_surf_hi_f, args.h_surf_n_slab, 1.0,            args.h_surf_loss),
        ]
        v_surfs  = [
            (v_surf_t, 1.0, args.v_surf_n, args.v_surf_loss),
        ]
        tracer = TFRayGPU(
            scene_end        = args.duration,
            f_min            = args.fmin,
            f_max            = args.fmax,
            h_surfs          = h_surfs,
            v_surfs          = v_surfs,
            freq_mirror_loss = args.freq_mirror_loss,
            time_mirror_loss = args.time_mirror_loss,
            scatter_rate     = args.scatter_rate,
            amp_loss         = args.amp_loss,
            n_scattered      = args.n_scattered,
            max_chirp        = args.max_chirp,
            max_batch        = args.batch_size,
            max_segments     = args.max_segments,
            min_amplitude    = args.min_amplitude,
            max_bounces      = args.max_bounces,
            device               = args.device,
            seed                 = args.seed,
            phase_mode           = args.phase_mode,
            density_png          = args.density_png,
            density_scale        = args.density_scale,
            density_sensitivity  = args.density_sensitivity,
            density_step         = args.density_step,
        )
        # Distribute n_rays evenly: half forward, half backward.
        # Each half gets a grid in (f, vf); n_side = ceil(sqrt(n_rays/2)).
        n_side   = max(1, int(math.ceil(math.sqrt(args.n_rays / 2))))
        freqs    = np.linspace(args.fmin + 50, args.fmax - 100, n_side)
        chirps   = np.linspace(-args.chirp_range, args.chirp_range, n_side)
        rows     = []
        for vt0 in (1.0, -1.0):
            t0 = 0.0 if vt0 > 0 else args.duration
            for f0 in freqs:
                for vf0 in chirps:
                    rows.append([t0, float(f0), vt0, float(vf0), 0.0, 1.0])
        init_arr = np.array(rows, dtype=np.float64)
        print(f"  Launching {len(rows)} initial rays on {args.device} (batch={args.batch_size})")
        all_segs = tracer.propagate(init_arr)
        print(f"  -> {len(all_segs)} chirp segments")
        r = TFRayScene(all_segs, scene_end=args.duration,
                       output_sample_rate=args.sample_rate, fade_s=args.fade)
        if not args.no_wav:
            r.write_wav(args.wav)
        r.save_field_png(args.png, freq_min=args.fmin, freq_max=args.fmax,
                         n_time=args.n_time, n_freq=args.n_freq,
                         heisenberg_spread=args.heisenberg, log_freq=args.log_freq,
                         log_magnitude=args.log_magnitude, dpi=args.dpi,
                         summation=args.summation,
                         amplitude_percentile=args.amplitude_percentile)

    # ---- plasma ----
    elif args.cmd == "plasma":
        n_obj = len(args.objects)

        def _obj_list(attr, default_fn):
            vals = getattr(args, attr) or []
            if len(vals) < n_obj:
                vals = vals + [default_fn(i) for i in range(len(vals), n_obj)]
            return vals

        dur = args.duration
        f_span = args.fmax - args.fmin
        obj_t0  = _obj_list("object_t0",     lambda _: dur * 0.25)
        obj_t1  = _obj_list("object_t1",     lambda _: dur * 0.75)
        obj_f0  = _obj_list("object_f0",     lambda _: args.fmin + f_span * 0.10)
        obj_f1  = _obj_list("object_f1",     lambda _: args.fmax - f_span * 0.10)
        obj_ref = _obj_list("object_reflect",    lambda _: 0.3)
        obj_dif = _obj_list("object_diffuse",    lambda _: 0.5)
        obj_bst = _obj_list("object_boost",      lambda _: 4.0)
        obj_cr  = _obj_list("object_charge_rate",lambda _: 0.8)

        tracer = TFRayGPU(
            scene_end        = dur,
            f_min            = args.fmin,
            f_max            = args.fmax,
            h_surfs          = [],
            v_surfs          = [],
            freq_mirror_loss = args.freq_mirror_loss,
            time_mirror_loss = args.time_mirror_loss,
            scatter_rate     = 0.0,
            amp_loss         = 0.0,
            n_scattered      = 1,
            max_chirp        = 1800.0,
            max_batch        = args.batch_size,
            max_segments     = args.max_segments,
            min_amplitude    = args.min_amplitude,
            max_bounces      = args.max_bounces,
            device           = args.device,
            seed             = args.seed,
            phase_mode       = args.phase_mode,
            density_step     = args.plasma_step,
        )

        for i, png_path in enumerate(args.objects):
            tracer.add_mask_object(
                png_path,
                t0          = obj_t0[i],
                t1          = obj_t1[i],
                f0          = obj_f0[i],
                f1          = obj_f1[i],
                reflect     = obj_ref[i],
                diffuse     = obj_dif[i],
                boost       = obj_bst[i],
                charge_rate = obj_cr[i],
            )

        # --- procedural baffles -----------------------------------------------
        # Two vertical baffle walls bracket one synchrotron cell.
        # Each wall spans the full F range with absorbing jaws top and bottom;
        # only rays in the center gap pass through.
        # The B-field is active ONLY between the two walls; field-free outside.
        _B_tlo = 0.0
        _B_thi = dur
        if args.baffles:
            import tempfile as _tmp, os as _os
            _bdir = _tmp.mkdtemp(prefix="tf_plasma_baffles_")
            n_cells  = max(1, args.baffle_humps)  # number of field cells (default 1)
            n_walls  = n_cells + 1                # walls = cells + 1
            wall_w   = max(dur / (n_walls + 1) * 0.015, 1e-3)
            _jaw_png = _os.path.join(_bdir, "baffle_jaw.png")
            _gen_baffle_jaw_png(_jaw_png, jaw_frac=args.baffle_amp_frac,
                                width_px=8, height_px=256)
            wall_ts = [dur * (i + 1) / (n_walls + 1) for i in range(n_walls)]
            for t_center in wall_ts:
                tracer.add_mask_object(
                    _jaw_png,
                    t0=t_center - wall_w * 0.5, t1=t_center + wall_w * 0.5,
                    f0=args.fmin, f1=args.fmax,
                    reflect=0.0, diffuse=0.0, boost=0.0, charge_rate=0.0,
                    refract_sensitivity=0.0, absorb=1.0,
                )
            # B-field lives between first and last wall only
            _B_tlo = wall_ts[0]
            _B_thi = wall_ts[-1]
            gap_hz = (args.fmax - args.fmin) * (1.0 - 2.0 * args.baffle_amp_frac)
            print(f"  Baffles: {n_walls} walls  {n_cells} cell(s)  jaw_frac={args.baffle_amp_frac:.2f}"
                  f"  center gap={gap_hz:.0f} Hz"
                  f"  B-field t=[{_B_tlo:.3f},{_B_thi:.3f}] s", flush=True)

        # Particle species sets the sign of initial_charge.
        # --initial-charge magnitude is used; sign is overridden by --particle.
        _charge_sign   = -1.0 if args.particle == 'electron' else +1.0
        _charge_mag    = abs(args.initial_charge)
        _initial_charge = _charge_sign * _charge_mag
        print(f"  Particle: {args.particle}  initial_charge={_initial_charge:+.3f}", flush=True)

        # B-field is confined between the baffle walls; field-free outside.
        tracer.set_plasma(B_field=args.B_field, synchrotron_coeff=args.synchrotron,
                          arc_angle=args.arc_angle, initial_charge=_initial_charge,
                          plasma_tlo=_B_tlo, plasma_thi=_B_thi)

        # --- grounded casing potential field ------------------------------------
        # phi=0 at all four walls, phi_max at centre; E = -∇phi repels particles
        # away from the walls and toward the centre (for positrons) or vice versa
        # (for electrons, where the Lorentz electric force is reversed).
        if args.casing:
            import tempfile as _tmp_mod, os as _os_casing
            _cdir = _tmp_mod.mkdtemp(prefix="tf_plasma_casing_")
            _casing_png = _os_casing.path.join(_cdir, "casing_potential.png")
            _gen_casing_potential_png(_casing_png, width_px=256, height_px=256,
                                      mode=args.casing_phi_mode)
            tracer.add_potential_field(
                _casing_png,
                t0=0.0, t1=dur,
                f0=args.fmin, f1=args.fmax,
                charge_scale=args.casing_charge_scale,
            )
            print(f"  Casing: {args.casing_phi_mode} potential  "
                  f"charge_scale={args.casing_charge_scale:.1f}  "
                  f"({'confinement' if args.particle == 'positron' else 'wall-acceleration'})",
                  flush=True)

        # --- initial rays: emitter or uniform grid ----------------------------
        if args.left_emitter:
            emitter_f = args.emitter_f if args.emitter_f is not None else (args.fmin + args.fmax) * 0.5
            init_arr  = TFRayGPU.make_emitter_states(
                n_rays                 = args.n_rays,
                t0                     = 0.0,
                f0                     = emitter_f,
                vt0                    = 1.0,
                f0_spread              = args.emitter_f_spread,
                vf0_spread             = args.emitter_chirp_spread,
                velocity_angle_spread  = args.emitter_velocity_angle_spread,
                seed                   = args.seed,
            )
        else:
            n_side = max(1, int(math.ceil(math.sqrt(args.n_rays / 2))))
            freqs  = np.linspace(args.fmin + 50, args.fmax - 100, n_side)
            chirps = np.linspace(-args.chirp_range, args.chirp_range, n_side)
            rows   = []
            for vt0 in (1.0, -1.0):
                t0 = 0.0 if vt0 > 0 else dur
                for f0 in freqs:
                    for vf0 in chirps:
                        rows.append([t0, float(f0), vt0, float(vf0), 0.0, 1.0])
            init_arr = np.array(rows, dtype=np.float64)
        print(f"  Launching {init_arr.shape[0]} {args.particle}s on {args.device} (batch={args.batch_size})")
        all_segs = tracer.propagate(init_arr)
        print(f"  -> {len(all_segs)} chirp segments")
        r = TFRayScene(all_segs, scene_end=dur,
                       output_sample_rate=args.sample_rate, fade_s=args.fade)
        if not args.no_wav:
            r.write_wav(args.wav)
        r.save_field_png(args.png, freq_min=args.fmin, freq_max=args.fmax,
                         n_time=args.n_time, n_freq=args.n_freq,
                         heisenberg_spread=args.heisenberg, log_freq=args.log_freq,
                         log_magnitude=args.log_magnitude, dpi=args.dpi,
                         summation=args.summation,
                         amplitude_percentile=args.amplitude_percentile)

    # ---- resonator ----
    elif args.cmd == "resonator":
        import tempfile, os as _os

        dur   = args.duration
        fmin  = args.fmin
        fmax  = args.fmax

        # --- generate procedural mask PNGs into temp files ---
        _tmp = tempfile.mkdtemp(prefix="tf_resonator_")
        wall_png   = _os.path.join(_tmp, "slotted_wall.png")
        rwall_png  = _os.path.join(_tmp, "diffuse_backstop.png")
        sphere_png = _os.path.join(_tmp, "sphere.png")

        _gen_slotted_wall_png(
            wall_png,
            n_slots   = args.wall_n_slots,
            slot_frac = args.wall_slot_frac,
            reflect   = args.wall_reflect,
            diffuse   = args.wall_diffuse,
        )
        _gen_diffuse_backstop_png(
            rwall_png,
            reflect = args.rwall_reflect,
            diffuse = args.rwall_diffuse,
        )
        _gen_sphere_png(
            sphere_png,
            width_px  = 256,
            height_px = 256,
            reflect   = args.sphere_reflect,
            refract   = args.sphere_refract,
            diffuse   = args.sphere_diffuse,
        )

        tracer = TFRayGPU(
            scene_end        = dur,
            f_min            = fmin,
            f_max            = fmax,
            h_surfs          = [],
            v_surfs          = [],
            freq_mirror_loss = args.freq_mirror_loss,
            time_mirror_loss = args.time_mirror_loss,
            scatter_rate     = 0.0,   # no fog — coherent propagation only
            amp_loss         = 0.0,
            n_scattered      = 1,
            max_chirp        = 1800.0,
            max_batch        = args.batch_size,
            max_segments     = args.max_segments,
            min_amplitude    = args.min_amplitude,
            max_bounces      = args.max_bounces,
            device           = args.device,
            seed             = args.seed,
            phase_mode       = "coherent",
        )

        wt   = args.wall_t
        wthk = args.wall_thickness / 2.0
        tracer.add_mask_object(
            wall_png,
            t0 = wt - wthk, t1 = wt + wthk,
            f0 = fmin, f1 = fmax,
            refract_sensitivity = 0.0,   # mirror wall — no lensing
        )

        rwt   = args.rwall_t
        rwthk = args.rwall_thickness / 2.0
        tracer.add_mask_object(
            rwall_png,
            t0 = rwt - rwthk, t1 = rwt + rwthk,
            f0 = fmin, f1 = fmax,
            refract_sensitivity = 0.0,
        )

        st, sf  = args.sphere_t, args.sphere_f
        sdt, sdf = args.sphere_dt, args.sphere_df
        tracer.add_mask_object(
            sphere_png,
            t0 = st - sdt, t1 = st + sdt,
            f0 = sf - sdf, f1 = sf + sdf,
            boost                = 2.0,
            charge_rate          = 0.0,
            refract_sensitivity  = args.sphere_refract_sensitivity,
        )

        # Coherent beam: narrow grid of rays from emitter position, forward only.
        freqs  = np.linspace(args.emitter_f - args.beam_f_spread,
                             args.emitter_f + args.beam_f_spread,
                             args.beam_rays)
        chirps = np.linspace(-args.beam_chirp_spread,
                              args.beam_chirp_spread,
                              args.beam_rays)
        rows = []
        for f0 in freqs:
            for vf0 in chirps:
                rows.append([args.emitter_t, float(f0), 1.0, float(vf0), 0.0, 1.0])
        init_arr = np.array(rows, dtype=np.float64)
        print(f"  Resonator: {len(rows)} coherent rays on {args.device} "
              f"(wall t=[{wt-wthk:.2f},{wt+wthk:.2f}]  sphere t=[{st-sdt:.2f},{st+sdt:.2f}])")
        all_segs = tracer.propagate(init_arr)
        print(f"  -> {len(all_segs)} chirp segments")
        r = TFRayScene(all_segs, scene_end=dur,
                       output_sample_rate=args.sample_rate, fade_s=args.fade)
        if not args.no_wav:
            r.write_wav(args.wav)
        r.save_field_png(args.png, freq_min=fmin, freq_max=fmax,
                         n_time=args.n_time, n_freq=args.n_freq,
                         heisenberg_spread=args.heisenberg, log_freq=args.log_freq,
                         log_magnitude=args.log_magnitude, dpi=args.dpi,
                         summation=args.summation,
                         amplitude_percentile=args.amplitude_percentile)

        # Clean up temp PNGs.
        for _p in (wall_png, rwall_png, sphere_png):
            try: _os.remove(_p)
            except OSError: pass
        try: _os.rmdir(_tmp)
        except OSError: pass

    # ---- bounce ----
    elif args.cmd == "bounce":
        scene = TFScene(f_min=args.fmin, f_max=args.fmax,
                        time_mirror_loss=args.time_mirror_loss)
        scene.add_surface(HorizontalSurface(f_hz=args.slab_lo, n_a=1.0, n_b=args.slab_n,
                          transmission_fraction=args.slab_t_frac,
                          reflection_fraction=args.slab_r_frac))
        scene.add_surface(HorizontalSurface(f_hz=args.slab_hi, n_a=args.slab_n, n_b=1.0,
                          transmission_fraction=args.slab_t_frac,
                          reflection_fraction=args.slab_r_frac))
        segs = TFPropagator(scene, scene_end=args.duration,
                            max_segments=args.max_segments, max_bounces=args.max_bounces,
                            min_amplitude=args.min_amplitude).propagate(
            RayState2D(t=0.0, f=args.f0, v_t=1.0, v_f=args.vf0, phi=0.0, A=args.amplitude))
        print(f"Bounce: {len(segs)} segments")
        r = TFRayScene(segs, scene_end=args.duration,
                       output_sample_rate=args.sample_rate, fade_s=args.fade)
        if not args.no_wav:
            r.write_wav(args.wav)
        r.save_field_png(args.png, freq_min=args.fmin, freq_max=args.fmax,
                         n_time=args.n_time, n_freq=args.n_freq,
                         heisenberg_spread=args.heisenberg, log_freq=args.log_freq,
                         log_magnitude=args.log_magnitude, dpi=args.dpi,
                         summation=args.summation,
                         amplitude_percentile=args.amplitude_percentile)

    # ---- mirror ----
    elif args.cmd == "mirror":
        scene = TFScene(f_min=args.fmin, f_max=args.fmax,
                        freq_mirror_loss=args.freq_mirror_loss,
                        time_mirror_loss=args.time_mirror_loss)
        prop  = TFPropagator(scene, scene_end=args.duration,
                             max_segments=args.max_segments,
                             min_amplitude=args.min_amplitude)
        all_segs = []
        for i, vf0 in enumerate(np.linspace(-args.chirp, args.chirp, args.n_rays)):
            vt0 = 1.0 if i % 2 == 0 else -1.0
            t0  = 0.0 if vt0 > 0 else args.duration
            all_segs.extend(prop.propagate(
                RayState2D(t=t0, f=args.f0, v_t=vt0, v_f=float(vf0), phi=0.0, A=args.amplitude)))
        print(f"Mirror: {len(all_segs)} segments")
        r = TFRayScene(all_segs, scene_end=args.duration,
                       output_sample_rate=args.sample_rate, fade_s=args.fade)
        if not args.no_wav:
            r.write_wav(args.wav)
        r.save_field_png(args.png, freq_min=args.fmin, freq_max=args.fmax,
                         n_time=args.n_time, n_freq=args.n_freq,
                         heisenberg_spread=args.heisenberg, log_freq=args.log_freq,
                         log_magnitude=args.log_magnitude, dpi=args.dpi,
                         summation=args.summation,
                         amplitude_percentile=args.amplitude_percentile)

    # ---- dispersion ----
    elif args.cmd == "dispersion":
        scene = TFScene(f_min=args.fmin, f_max=args.fmax)
        scene.add_surface(VerticalSurface(t_sec=args.slab_lo, n_a=1.0, n_b=args.slab_n,
                          transmission_fraction=args.slab_t_frac,
                          reflection_fraction=args.slab_r_frac))
        scene.add_surface(VerticalSurface(t_sec=args.slab_hi, n_a=args.slab_n, n_b=1.0,
                          transmission_fraction=args.slab_t_frac,
                          reflection_fraction=args.slab_r_frac))
        prop = TFPropagator(scene, scene_end=args.duration, max_segments=args.max_segments)
        all_segs = []
        for vf0 in np.linspace(-args.chirp_range, args.chirp_range, args.n_rays):
            all_segs.extend(prop.propagate(
                RayState2D(t=0.0, f=args.f0, v_t=1.0, v_f=float(vf0), phi=0.0, A=args.amplitude)))
        print(f"Dispersion: {len(all_segs)} segments")
        r = TFRayScene(all_segs, scene_end=args.duration,
                       output_sample_rate=args.sample_rate, fade_s=args.fade)
        if not args.no_wav:
            r.write_wav(args.wav)
        r.save_field_png(args.png, freq_min=args.fmin, freq_max=args.fmax,
                         n_time=args.n_time, n_freq=args.n_freq,
                         heisenberg_spread=args.heisenberg, log_freq=args.log_freq,
                         log_magnitude=args.log_magnitude, dpi=args.dpi,
                         summation=args.summation,
                         amplitude_percentile=args.amplitude_percentile)
