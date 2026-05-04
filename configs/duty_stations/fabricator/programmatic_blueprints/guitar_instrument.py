"""
Guitar Instrument — Fabricator Programmatic Blueprint
=====================================================

Encapsulates the guitar as a self-contained fabricatable item:

  factory(params)         → DECMesh       static visual mesh
  plugin_factory(params)  → GuitarPlugin  string physics state machine

GuitarPlugin is a pure-Python lift of the string subsystem from the C
AcousticCoEvolver (acoustic_fdtd_bridge.py / _spectral_kernels).  It
runs the identical per-string finite-difference wave equation with:

  - Exact physical parameters from _GUITAR_STRING_PARAMS / _string_physical_params
  - CFL-safe sub-stepping (same criterion as the C coevolver)
  - Euler-Bernoulli stiffness correction (bilap term, EI)
  - Rayleigh damping (distributed along string, same alpha as string_defs)
  - Neck spring-mass-damper BC at nut end (same 65 Hz / 0.18 kg / Q=35 as string_defs)
  - Saddle end: plate-velocity coupling plug
      * If v_plate_at_saddle is None  → clamped (u[0] = 0) — standalone mode
      * If v_plate_at_saddle is supplied per sample by the pressure sim
        → u_saddle += v_plate * dt_sub  (two-way coupling, same as C coevolver)
  - Bridge force output: T*(u[1]-u[0])/dx per string per sample
    → inject at plugin.bridge_positions[si] in the pressure sim

Outline and string parameters are imported directly from acoustic_fdtd_bridge
so there is a single canonical source for all physical values.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from dec_mesh import DECMesh

# ---------------------------------------------------------------------------
# Import canonical physics from the existing bridge module.
# Only the pure-Python definitions are used here — no C extension loaded.
# ---------------------------------------------------------------------------
from acoustic_fdtd_bridge import (
    GUITAR_SCALE_LENGTH_M,
    _GUITAR_STRING_PARAMS,
    _string_physical_params,
    _default_guitar_outline,
)

_STAND_HEIGHT_M   = 0.40    # body bottom Z above stage floor
_STRING_CLEARANCE = 0.010   # m above soundboard

# Neck spring-mass-damper parameters — same as string_defs in
# build_acoustic_coevolver_from_scene:
_NECK_FREQ_HZ  = 65.0
_NECK_MASS_KG  = 0.18
_NECK_Q        = 35.0

# ---------------------------------------------------------------------------
# DECMesh geometry helpers
# ---------------------------------------------------------------------------

def _merge_raw(
    parts: list[tuple[np.ndarray, list[list[int]]]]
) -> tuple[np.ndarray, list[list[int]]]:
    v_all: list[np.ndarray] = []
    f_all: list[list[int]] = []
    off = 0
    for verts, faces in parts:
        v_all.append(verts)
        for f in faces:
            f_all.append([int(v + off) for v in f])
        off += int(len(verts))
    if not v_all:
        return np.zeros((0, 3), np.float64), []
    return np.vstack(v_all), f_all


def _box_faces(
    w: float, d: float, h: float,
    z0: float = 0.0,
    cx: float = 0.0, cy: float = 0.0,
) -> tuple[np.ndarray, list[list[int]]]:
    hx, hy, z1 = 0.5 * w, 0.5 * d, z0 + h
    v = np.array([
        [cx - hx, cy - hy, z0], [cx + hx, cy - hy, z0],
        [cx + hx, cy + hy, z0], [cx - hx, cy + hy, z0],
        [cx - hx, cy - hy, z1], [cx + hx, cy - hy, z1],
        [cx + hx, cy + hy, z1], [cx - hx, cy + hy, z1],
    ], np.float64)
    faces = [
        [0, 3, 2, 1], [4, 5, 6, 7],
        [0, 1, 5, 4], [1, 2, 6, 5],
        [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    return v, faces


def _cylinder_ring_verts(
    cx: float, cy: float, r: float, z: float, segs: int
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, segs, endpoint=False)
    return np.column_stack([
        cx + r * np.cos(angles),
        cy + r * np.sin(angles),
        np.full(segs, z, np.float64),
    ])


def _cylinder_faces(
    cx: float, cy: float, r: float, h: float,
    z0: float = 0.0, segs: int = 12,
) -> tuple[np.ndarray, list[list[int]]]:
    z1 = z0 + h
    bot = _cylinder_ring_verts(cx, cy, r, z0, segs)
    top = _cylinder_ring_verts(cx, cy, r, z1, segs)
    verts = np.vstack([bot, top])
    faces: list[list[int]] = []
    faces.append(list(range(segs - 1, -1, -1)))
    faces.append(list(range(segs, 2 * segs)))
    for i in range(segs):
        j = (i + 1) % segs
        faces.append([i, j, segs + j, segs + i])
    return verts, faces


# ---------------------------------------------------------------------------
# Guitar body DECMesh geometry
# ---------------------------------------------------------------------------

def _guitar_body_mesh(
    outline: np.ndarray,   # (N, 2)  — from _default_guitar_outline()
    body_h: float,
    stand_h: float = _STAND_HEIGHT_M,
    include_stand: bool = True,
    stand_style: str = "x_frame",
    include_neck: bool = True,
    scale_length_m: float = GUITAR_SCALE_LENGTH_M,
) -> DECMesh:
    parts: list[tuple[np.ndarray, list[list[int]]]] = []
    N = len(outline)

    # ── Body side walls (extruded outline) ───────────────────────────────────
    z_bot = float(stand_h)
    z_top = z_bot + float(body_h)
    bot_verts = np.column_stack([outline, np.full(N, z_bot, np.float64)])
    top_verts = np.column_stack([outline, np.full(N, z_top, np.float64)])
    side_verts = np.vstack([bot_verts, top_verts])
    side_faces: list[list[int]] = []
    for i in range(N):
        j = (i + 1) % N
        side_faces.append([i, j, N + j, N + i])
    side_faces.append(list(range(N - 1, -1, -1)))
    side_faces.append(list(range(N, 2 * N)))
    parts.append((side_verts, side_faces))

    # ── Neck ─────────────────────────────────────────────────────────────────
    if include_neck:
        y_body   = float(outline[:, 1].max()) * 0.85
        y_saddle = -0.070
        y_nut    = y_saddle + float(scale_length_m)
        y_head   = y_nut + 0.13
        neck_avg_w = 0.050
        head_w     = 0.092

        neck_len = y_nut - y_body
        if neck_len > 0.01:
            nv, nf = _box_faces(neck_avg_w, neck_len, 0.018,
                                 z0=z_top, cx=0.0, cy=(y_body + y_nut) * 0.5)
            parts.append((nv, nf))

        head_len = y_head - y_nut
        if head_len > 0.005:
            hv, hf = _box_faces(head_w, head_len, 0.016,
                                 z0=z_top + 0.002, cx=0.0, cy=(y_nut + y_head) * 0.5)
            parts.append((hv, hf))

    # ── Stand ─────────────────────────────────────────────────────────────────
    if include_stand and stand_h > 0.01:
        gx = 0.0
        gy = float(outline[:, 1].mean())

        if stand_style == "x_frame":
            leg_spread_x, leg_spread_y = 0.16, 0.06
            leg_r = 0.012
            for sx, sy in [
                (-leg_spread_x, gy - leg_spread_y),
                ( leg_spread_x, gy - leg_spread_y),
                (-leg_spread_x, gy + leg_spread_y),
                ( leg_spread_x, gy + leg_spread_y),
            ]:
                cv, cf = _cylinder_faces(sx, sy, leg_r, stand_h, z0=0.0, segs=8)
                parts.append((cv, cf))
        elif stand_style == "tripod":
            leg_r = 0.015
            for angle_deg in (90.0, 210.0, 330.0):
                a = math.radians(angle_deg)
                cv, cf = _cylinder_faces(
                    gx + 0.18 * math.cos(a),
                    gy + 0.18 * math.sin(a),
                    leg_r, stand_h, z0=0.0, segs=8)
                parts.append((cv, cf))

        bv, bf = _box_faces(0.36, 0.025, 0.012, z0=stand_h * 0.35, cx=gx, cy=gy)
        parts.append((bv, bf))

    verts, faces = _merge_raw(parts)
    return DECMesh.from_raw(verts, faces)


# ---------------------------------------------------------------------------
# GuitarPlugin — string wave equation lifted from AcousticCoEvolver
# ---------------------------------------------------------------------------

class GuitarPlugin:
    """Pure-Python lift of the string subsystem from _spectral_kernels.AcousticCoEvolver.

    Runs independently of any pressure sim.  When connected to a pressure
    sim, pass saddle plate velocities into step() to engage the two-way
    spring coupling at the bridge — identical to what the C co-evolver does
    sample-synchronously with its internal sub-stepping.

    All physical parameters are drawn from the canonical _GUITAR_STRING_PARAMS
    table in acoustic_fdtd_bridge — there are no separate values here.

    Per-string state machine
    ------------------------
    For each string si:

      Nut end (index N = n_segs):
        Neck spring-mass-damper (neck_freq_hz, neck_mass_kg, neck_Q).
        The neck modal coordinate x_neck follows:
          m * x'' + c * x' + k * x = T * (u[N] - u[N-1]) / dx
        The string nut node tracks the neck:
          u[N] = x_neck

      Interior (indices 1 .. N-1):
        Damped Euler-Bernoulli wave equation, n_sub CFL substeps per sample:
          u_new[i] = (2 - alpha*dt_s)*u[i] - (1 - alpha*dt_s)*u_prev[i]
                   + r^2*(u[i+1]-2u[i]+u[i-1])
                   - (EI/(mu*dx^4))*dt_s^2 * bilaplacian[i]

      Saddle end (index 0):
        If v_plate_at_saddle is None (standalone / no pressure sim):
          u[0] = 0                              (clamped)
        If v_plate_at_saddle is provided (pressure sim connected):
          u[0] += v_plate_at_saddle[si, k] * dt_sub   (per sub-step, per sample)

      Bridge force output (drives pressure sim):
        F[si, k] = T * (u[1] - u[0]) / dx
    """

    def __init__(
        self,
        n_strings:       int   = 6,
        scale_length_m:  float = GUITAR_SCALE_LENGTH_M,
        fret:            int   = 0,
        fretless:        bool  = False,
        sample_rate:     int   = 44100,
        n_segs:          int   = 240,      # same default as build_acoustic_coevolver_from_scene
        stand_height:    float = _STAND_HEIGHT_M,
        body_h:          float = 0.060,
        outline:         np.ndarray | None = None,
        neck_freq_hz:    float = _NECK_FREQ_HZ,
        neck_mass_kg:    float = _NECK_MASS_KG,
        neck_Q:          float = _NECK_Q,
    ) -> None:
        self.n_strings    = int(n_strings)
        self.sample_rate  = int(sample_rate)
        self.n_segs       = int(n_segs)
        self.stand_height = float(stand_height)
        self.body_h       = float(body_h)

        # Effective speaking length after fretting (same formula as string_defs)
        fret = max(0, int(fret))
        self.scale_length_m = float(scale_length_m) / (2.0 ** (fret / 12.0))

        if outline is None:
            outline = _default_guitar_outline()
        self.outline = np.asarray(outline, dtype=np.float32)

        # Bridge saddle positions — exact same spread as build_acoustic_coevolver_from_scene
        y_saddle = -0.070
        x_span   = 0.0088 * max(0, self.n_strings - 1)
        self.bridge_positions: list[tuple[float, float]] = []
        for si in range(self.n_strings):
            x_str = -x_span / 2.0 + si * (x_span / max(1, self.n_strings - 1))
            self.bridge_positions.append((float(x_str), float(y_saddle)))

        # String Z height above stage (same as string_defs)
        string_z = self.stand_height + self.body_h + _STRING_CLEARANCE

        y_nut = y_saddle + self.scale_length_m
        param_indices = np.linspace(0, 5, self.n_strings, dtype=int)

        dt = 1.0 / float(self.sample_rate)

        self._strings: list[dict] = []
        for si in range(self.n_strings):
            pix = int(param_indices[si])
            f0, gauge, tension, lin_mass, damping, stiffness_EI, axial_stiffness = \
                _string_physical_params(pix, self.scale_length_m)

            # Fret damping correction (identical to string_defs in bridge module)
            if fret > 0:
                damping *= 1.18 if fretless else 1.08

            x_str = self.bridge_positions[si][0]
            x_nut = x_str * 0.72

            # String path: (n_segs+1, 3) from nut to saddle
            ys_path = np.linspace(y_nut, y_saddle, n_segs + 1, dtype=np.float32)
            xs_path = np.linspace(x_nut, x_str,    n_segs + 1, dtype=np.float32)
            path = np.column_stack([
                xs_path, ys_path,
                np.full(n_segs + 1, string_z, dtype=np.float32),
            ])

            L      = self.scale_length_m
            dx     = L / float(n_segs)
            c_wave = math.sqrt(tension / max(lin_mass, 1e-12))
            r_cfl  = c_wave * dt / dx
            # Sub-step count: CFL <= 0.9 (same criterion as C coevolver)
            n_sub  = max(1, math.ceil(r_cfl / 0.9))
            dt_sub = dt / float(n_sub)
            r2     = (c_wave * dt_sub / dx) ** 2
            # EI stiffness coefficient per sub-step^2
            ei_coeff = (stiffness_EI / (lin_mass * dx ** 4)) * dt_sub ** 2 \
                       if lin_mass > 0 else 0.0
            alpha_dt = damping * dt_sub   # Rayleigh mass damping per sub-step

            # Neck spring-mass-damper at nut (index N = n_segs)
            k_neck = (2.0 * math.pi * neck_freq_hz) ** 2 * neck_mass_kg
            c_neck = (2.0 * math.pi * neck_freq_hz) * neck_mass_kg / neck_Q

            u      = np.zeros(n_segs + 1, np.float64)
            u_prev = np.zeros(n_segs + 1, np.float64)

            self._strings.append({
                "u":          u,
                "u_prev":     u_prev,
                "tension":    tension,
                "lin_mass":   lin_mass,
                "dx":         dx,
                "r2":         r2,
                "ei_coeff":   ei_coeff,
                "alpha_dt":   alpha_dt,
                "damping":    damping,
                "n_sub":      n_sub,
                "dt_sub":     dt_sub,
                "dt":         dt,
                "path":       path,
                # Neck SMD modal state [x_neck, v_neck] — mutable list
                "neck_state": [0.0, 0.0],
                "k_neck":     k_neck,
                "c_neck":     c_neck,
                "m_neck":     neck_mass_kg,
                # Metadata
                "f0":              f0,
                "gauge":           gauge,
                "stiffness_EI":    stiffness_EI,
                "axial_stiffness": axial_stiffness,
            })

        # Pluck schedule: list of (abs_onset_sample, string_idx, pos_frac, amp)
        self._pluck_schedule: list[tuple[int, int, float, float]] = []
        self._sample_count: int = 0

    # ── Pluck scheduling ──────────────────────────────────────────────────────

    def schedule_pluck(
        self,
        onset_samples: int,
        string_idx:    int,
        pos_frac:      float = 0.20,
        amp:           float = 0.003,
    ) -> None:
        """Queue a raised-cosine displacement pluck.

        onset_samples : relative to current time (0 = this frame)
        pos_frac      : fractional position along string from nut (0=nut, 1=saddle)
        amp           : peak displacement in metres
        """
        abs_onset = self._sample_count + max(0, int(onset_samples))
        self._pluck_schedule.append(
            (abs_onset, int(string_idx), float(pos_frac), float(amp))
        )

    def clear_pluck_schedule(self) -> None:
        self._pluck_schedule.clear()

    def _apply_pluck(self, si: int, pos_frac: float, amp: float) -> None:
        """Raised-cosine initial displacement on string si."""
        s = self._strings[si]
        N = self.n_segs
        i_center = int(round(pos_frac * N))
        w = max(4, N // 10)
        lo = max(1, i_center - w)
        hi = min(N - 1, i_center + w)
        idx = np.arange(lo, hi + 1)
        phase = (idx - i_center) * math.pi / float(w)
        disp = amp * (1.0 + np.cos(phase)) * 0.5
        s["u"][lo:hi + 1]      += disp
        s["u_prev"][lo:hi + 1] += disp   # zero initial velocity

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(
        self,
        n_samples:         int,
        v_plate_at_saddle: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance all strings by n_samples audio samples.

        Parameters
        ----------
        n_samples : int
            Number of audio samples to advance.
        v_plate_at_saddle : (n_strings, n_samples) array, optional
            Plate velocity at the bridge saddle position for each string and
            each sample, supplied by the pressure sim.  If None the saddle
            end is clamped (u[0] = 0) — standalone / no pressure sim mode.
            Dtype is not coerced; pass whatever the pressure sim produces.

        Returns
        -------
        bridge_forces : (n_strings, n_samples) float64
            Force in Newtons at the saddle for each string per sample.
            Sign convention: positive = away from body (tension pulls inward).
            Feed into the pressure sim at bridge_positions[si].
        """
        out = np.zeros((self.n_strings, n_samples), np.float64)
        coupled = v_plate_at_saddle is not None

        for k in range(n_samples):
            t = self._sample_count + k

            # Fire scheduled plucks at their onset sample
            fired = []
            for entry in self._pluck_schedule:
                if entry[0] <= t:
                    self._apply_pluck(entry[1], entry[2], entry[3])
                    fired.append(entry)
            for e in fired:
                self._pluck_schedule.remove(e)

            for si, s in enumerate(self._strings):
                u      = s["u"]
                u_prev = s["u_prev"]
                r2     = s["r2"]
                ei     = s["ei_coeff"]
                a_dt   = s["alpha_dt"]
                T      = s["tension"]
                dx     = s["dx"]
                n_sub  = s["n_sub"]
                dt_sub = s["dt_sub"]
                k_neck = s["k_neck"]
                c_neck = s["c_neck"]
                m_neck = s["m_neck"]
                ns     = s["neck_state"]   # [x_neck, v_neck]

                v_plate_k = float(v_plate_at_saddle[si, k]) if coupled else 0.0

                for _ in range(n_sub):
                    u_new = np.empty_like(u)

                    # ── Interior wave update (indices 1..N-1) ─────────────────
                    lap = u[2:] - 2.0 * u[1:-1] + u[:-2]   # shape (N-1,)
                    u_new[1:-1] = (
                        (2.0 - a_dt) * u[1:-1]
                        - (1.0 - a_dt) * u_prev[1:-1]
                        + r2 * lap
                    )

                    # EI stiffness correction for interior nodes 2..N-2
                    if ei > 0.0 and self.n_segs > 4:
                        bilap = (u[4:] - 4.0*u[3:-1]
                                 + 6.0*u[2:-2]
                                 - 4.0*u[1:-3] + u[:-4])
                        u_new[2:-2] -= ei * bilap

                    # ── Saddle BC (index 0) ───────────────────────────────────
                    if coupled:
                        # Two-way coupling: string saddle rides the plate
                        u_new[0] = u[0] + v_plate_k * dt_sub
                    else:
                        u_new[0] = 0.0

                    # ── Neck spring-mass-damper BC (index N = n_segs) ─────────
                    # Force string exerts on neck at nut
                    F_nut = T * (u[-2] - u[-1]) / dx
                    a_neck = (F_nut - c_neck * ns[1] - k_neck * ns[0]) / m_neck
                    ns[1] += a_neck * dt_sub      # v_neck
                    ns[0] += ns[1]  * dt_sub      # x_neck
                    u_new[-1] = ns[0]             # string nut follows neck

                    u_prev[:] = u
                    u[:]      = u_new

                # Bridge force: T * du/dx at saddle end (index 0)
                out[si, k] = T * (u[1] - u[0]) / dx

        self._sample_count += n_samples
        return out

    # ── String visualisation ──────────────────────────────────────────────────

    def get_string_positions(self, si: int, n_segs: int) -> np.ndarray:
        """World-space string positions (n_segs+1, 3) float32."""
        s    = self._strings[si]
        u    = s["u"]
        path = s["path"]   # (n_segs_orig+1, 3)

        n_orig = len(u) - 1
        if n_orig != n_segs:
            x_src = np.linspace(0.0, 1.0, n_orig + 1)
            x_dst = np.linspace(0.0, 1.0, n_segs + 1)
            u_out = np.interp(x_dst, x_src, u)
            x_src2 = np.linspace(0.0, 1.0, len(path))
            xs = np.interp(x_dst, x_src2, path[:, 0])
            ys = np.interp(x_dst, x_src2, path[:, 1])
        else:
            u_out = u
            xs = path[:, 0]
            ys = path[:, 1]

        z0 = self.stand_height + self.body_h + _STRING_CLEARANCE
        zs = z0 + u_out

        return np.column_stack([
            xs.astype(np.float32),
            ys.astype(np.float32),
            zs.astype(np.float32),
        ])

    def reset(self) -> None:
        """Zero all string displacements and neck state."""
        for s in self._strings:
            s["u"][:]      = 0.0
            s["u_prev"][:] = 0.0
            s["neck_state"][0] = 0.0
            s["neck_state"][1] = 0.0
        self._sample_count = 0
        self._pluck_schedule.clear()

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    def string_paths(self) -> list[np.ndarray]:
        """List of (n_segs+1, 3) float32 rest-position paths, one per string."""
        return [s["path"] for s in self._strings]

    @property
    def string_params(self) -> list[dict]:
        """Physical parameter dict for each string (read-only snapshot)."""
        return [
            {
                "f0":                s["f0"],
                "tension_N":         s["tension"],
                "linear_mass_kgm":   s["lin_mass"],
                "damping":           s["damping"],
                "stiffness_EI":      s["stiffness_EI"],
                "axial_stiffness_N": s["axial_stiffness"],
                "gauge_in":          s["gauge"],
                "n_sub":             s["n_sub"],
                "dt_sub":            s["dt_sub"],
            }
            for s in self._strings
        ]


# ---------------------------------------------------------------------------
# Parameter coercion helpers
# ---------------------------------------------------------------------------

def _p(params: dict, key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except Exception:
        return float(default)


def _pi(params: dict, key: str, default: int) -> int:
    try:
        return int(params.get(key, default))
    except Exception:
        return int(default)


def _pb(params: dict, key: str, default: bool) -> bool:
    try:
        return bool(params.get(key, default))
    except Exception:
        return bool(default)


# ---------------------------------------------------------------------------
# Blueprint entry points
# ---------------------------------------------------------------------------

def factory(params: dict[str, Any]) -> DECMesh:
    """Build the static guitar visual mesh."""
    n_strings      = max(1, min(6,   _pi(params, "n_strings", 6)))
    scale_length_m = max(0.50, min(0.75, _p(params, "scale_length_m", GUITAR_SCALE_LENGTH_M)))
    body_h         = max(0.04, min(0.12, _p(params, "body_h_m", 0.060)))
    stand_h        = max(0.00, min(0.80, _p(params, "stand_height_m", _STAND_HEIGHT_M)))
    stand_style    = str(params.get("stand_style", "x_frame"))
    include_neck   = _pb(params, "include_neck",  True)
    include_stand  = _pb(params, "include_stand", True)
    outline_pts    = max(16, min(128, _pi(params, "outline_resolution", 64)))
    outline        = _default_guitar_outline(n_pts=outline_pts)
    return _guitar_body_mesh(
        outline=outline,
        body_h=body_h,
        stand_h=stand_h,
        include_stand=include_stand,
        stand_style=stand_style,
        include_neck=include_neck,
        scale_length_m=scale_length_m,
    )


def plugin_factory(params: dict[str, Any]) -> GuitarPlugin:
    """Instantiate the guitar string physics state machine."""
    n_strings      = max(1, min(6,   _pi(params, "n_strings", 6)))
    scale_length_m = max(0.50, min(0.75, _p(params, "scale_length_m", GUITAR_SCALE_LENGTH_M)))
    fret           = max(0,    min(24,   _pi(params, "fret", 0)))
    fretless       = _pb(params, "fretless", False)
    sample_rate    = max(8000, min(96000, _pi(params, "sample_rate", 44100)))
    n_segs         = max(32,   min(512,  _pi(params, "physics_segments", 240)))
    stand_h        = max(0.00, min(0.80, _p(params, "stand_height_m", _STAND_HEIGHT_M)))
    body_h         = max(0.04, min(0.12, _p(params, "body_h_m", 0.060)))
    outline_pts    = max(16,   min(128,  _pi(params, "outline_resolution", 64)))
    outline        = _default_guitar_outline(n_pts=outline_pts)
    return GuitarPlugin(
        n_strings=n_strings,
        scale_length_m=scale_length_m,
        fret=fret,
        fretless=fretless,
        sample_rate=sample_rate,
        n_segs=n_segs,
        stand_height=stand_h,
        body_h=body_h,
        outline=outline,
    )


# ---------------------------------------------------------------------------
# Blueprint registration
# ---------------------------------------------------------------------------

BLUEPRINT = {
    "id":    "guitar_instrument",
    "label": "Guitar Instrument",
    "knobspec": [
        {
            "name": "n_strings", "label": "Strings", "dtype": "int",
            "default": 6, "low": 1, "high": 6, "step": 1, "fmt": "d",
        },
        {
            "name": "scale_length_m", "label": "Scale Length (m)", "dtype": "float",
            "default": GUITAR_SCALE_LENGTH_M, "low": 0.580, "high": 0.700,
            "step": 0.005, "fmt": ".3f",
        },
        {
            "name": "body_h_m", "label": "Body Depth (m)", "dtype": "float",
            "default": 0.060, "low": 0.040, "high": 0.120, "step": 0.005, "fmt": ".3f",
        },
        {
            "name": "fret", "label": "Fret (capo)", "dtype": "int",
            "default": 0, "low": 0, "high": 24, "step": 1, "fmt": "d",
        },
        {
            "name": "fretless", "label": "Fretless", "dtype": "bool",
            "default": False,
        },
        {
            "name": "stand_height_m", "label": "Stand Height (m)", "dtype": "float",
            "default": _STAND_HEIGHT_M, "low": 0.00, "high": 0.80,
            "step": 0.05, "fmt": ".2f",
        },
        {
            "name": "stand_style", "label": "Stand Style", "dtype": "choice",
            "default": "x_frame", "choices": ["x_frame", "tripod", "none"],
        },
        {
            "name": "include_neck", "label": "Show Neck", "dtype": "bool",
            "default": True,
        },
        {
            "name": "physics_segments", "label": "Physics Segments", "dtype": "int",
            "default": 240, "low": 32, "high": 512, "step": 32, "fmt": "d",
        },
        {
            "name": "outline_resolution", "label": "Body Resolution", "dtype": "int",
            "default": 64, "low": 16, "high": 128, "step": 16, "fmt": "d",
        },
        {
            "name": "sample_rate", "label": "Sample Rate (Hz)", "dtype": "int",
            "default": 44100, "low": 8000, "high": 96000, "step": 4000, "fmt": "d",
        },
    ],
    "factory":        factory,
    "plugin_factory": plugin_factory,
}
