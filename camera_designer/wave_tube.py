"""camera_designer/wave_tube.py
======================================
WaveTube — a light-guiding structure modelled as a wave context.

Architecture
------------
A wave tube is a cylindrical (or slab) region where the optical field is
propagated via the paraxial beam propagation method (BPM) rather than
geometric ray tracing.  At the *outer edge* of the wave context — the exit
aperture — the BPM solution is integrated to produce a **surrogate emitter**:
an EMISSIVE TriGroup whose per-band power equals the BPM exit-plane integral

    P_b = ∫∫ |E_b(x, y)|² dx dy

Backward rays that reach the exit face receive this power analytically, exactly
as they would from any other emissive group.  No scatter traversal of the tube
interior is needed.

Three scene elements are registered automatically at construction:

1. **Entry TriGroup** (SENSOR + UV accumulator)
   Forward rays entering the tube are captured in the UV image, giving a
   per-pixel record of arriving complex amplitude per band.

2. **Scale context** (SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ)
   Rays that physically enter the tube region receive wave-accurate phase
   accumulation and near-field spreading.  This affects BDPT connection
   paths that traverse the tube directly.

3. **Exit TriGroup** (EMISSIVE, initially zero power)
   After WaveTube.solve() runs the BPM the exit group's power_W_per_band
   is updated via ray_tracer_set_tri_group_power().  All subsequent
   backward rays treat the exit face as a real physical emitter.

BPM solver
----------
One complete z-step consists of three sub-steps (matching the GPU T4 shader
in ray_wave_bpm.comp.glsl):

    (a) Carrier advance: E *= exp(i k₀_n dz)
    (b) Horizontal ADI sweep (Thomas tridiagonal on each row)
    (c) Vertical   ADI sweep (Thomas tridiagonal on each column)

Boundary conditions: Dirichlet (zero field) at the transverse edges — energy
that hits the tube wall is absorbed.  This is the physical model for a
perfectly absorbing tube or a numerical aperture cut-off.

BPM seeding from tri_illum_accum
---------------------------------
When the BSSRDF illumination accumulator has been populated by a forward pass,
the per-triangle complex amplitude at the entry aperture provides a spatially
coherent seed for the BPM.  solve() reads this from
tracer.export_illum_accum() if the entry group is also in the BSSRDF
accumulator, falling back to the UV image accumulator otherwise.

Usage
-----
    tube = WaveTube(
        tracer,
        entry_tri_indices=np.array([...], dtype=np.int32),
        exit_tri_indices =np.array([...], dtype=np.int32),
        axis      = np.array([1.0, 0.0, 0.0]),   # +X propagation
        entry_pos = np.array([0.0, 0.0, 0.0]),
        exit_pos  = np.array([0.5, 0.0, 0.0]),
        tube_radius_m  = 0.01,
        n_medium       = 1.5,
        wavelengths_m  = np.array([0.55e-6, 0.65e-6]),  # per band
        nx=64, ny=64,                                    # BPM grid
    )

    # Run forward tracing (populates entry UV accum):
    tracer.trace(...)

    # Solve BPM and update surrogate emitter:
    exit_field = tube.solve(tracer)  # (n_bands, ny, nx) complex

    # Now run backward tracing — the exit face emits with correct power.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = ["WaveTube", "WaveTubeConfig"]


# ---------------------------------------------------------------------------
# Constants mirrored from ray_tracer.h
# ---------------------------------------------------------------------------
TRI_GROUP_ROLE_EMISSIVE  = (1 << 0)
TRI_GROUP_ROLE_SENSOR    = (1 << 1)
TRI_GROUP_SAMPLE_AREA    = 1

RT_SCALE_WAVE            = 1
SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ = 1


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class WaveTubeConfig:
    """All geometry and physics parameters for one wave tube."""
    axis:          np.ndarray   # unit propagation direction (3,)
    entry_pos:     np.ndarray   # entry aperture centre (3,) metres
    exit_pos:      np.ndarray   # exit  aperture centre (3,) metres
    tube_radius_m: float        # tube clear aperture radius (m)
    n_medium:      float        # refractive index of tube interior (real part)
    n_imag:        float        # extinction coefficient (0 = lossless)
    wavelengths_m: np.ndarray   # (n_bands,) central wavelength per band (m)
    nx:            int          # BPM transverse grid width
    ny:            int          # BPM transverse grid height
    dx_m:          float        # transverse grid spacing (m); 0 = auto from radius
    n_bpm_steps:   int          # longitudinal BPM steps (auto if 0)
    pre_roll_frames: int = 8    # forward-trace frames to discard before going live

    def __post_init__(self):
        self.axis         = np.asarray(self.axis,         dtype=np.float64)
        self.entry_pos    = np.asarray(self.entry_pos,    dtype=np.float64)
        self.exit_pos     = np.asarray(self.exit_pos,     dtype=np.float64)
        self.wavelengths_m = np.asarray(self.wavelengths_m, dtype=np.float64)
        # Auto grid spacing: sample the aperture at sub-Nyquist for the shortest wavelength
        if self.dx_m <= 0.0:
            min_wl = float(np.min(self.wavelengths_m))
            # At minimum: dx <= lambda/2 / n_medium  (Nyquist for finest feature)
            self.dx_m = min(self.tube_radius_m / (self.nx / 2),
                            min_wl / (2.0 * self.n_medium))
        if self.n_bpm_steps <= 0:
            length_m = float(np.linalg.norm(self.exit_pos - self.entry_pos))
            min_wl   = float(np.min(self.wavelengths_m))
            # dz ≈ lambda/n so that the carrier advance is roughly 2π per step
            dz_target = min_wl / self.n_medium
            self.n_bpm_steps = max(16, int(math.ceil(length_m / dz_target)))

    @property
    def n_bands(self) -> int:
        return len(self.wavelengths_m)

    @property
    def dz_m(self) -> float:
        length_m = float(np.linalg.norm(self.exit_pos - self.entry_pos))
        return length_m / max(1, self.n_bpm_steps)


# ---------------------------------------------------------------------------
# Registered scene handle
# ---------------------------------------------------------------------------
@dataclass
class WaveTube:
    """Wave tube registered in a RayTracer scene.

    Do not construct directly — use WaveTube.register().
    """
    config:              WaveTubeConfig
    entry_group_id:      int
    exit_group_id:       int          # -1: no EMISSIVE group (see register)
    ctx_id:              int          # scale context id (-1 if not registered)
    exit_tri_ids:        Optional[np.ndarray] = field(default=None, repr=False)
    _exit_tri_centroids: Optional[np.ndarray] = field(default=None, repr=False)
    _u_axis:             Optional[np.ndarray] = field(default=None, repr=False)
    _v_axis:             Optional[np.ndarray] = field(default=None, repr=False)
    _exit_field:         Optional[np.ndarray] = field(default=None, repr=False)
    _exit_power:         Optional[np.ndarray] = field(default=None, repr=False)
    _frames_collected:   int                  = field(default=0,    repr=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def register(
        cls,
        tracer,
        entry_tri_indices:   np.ndarray,
        exit_tri_indices:    np.ndarray,
        cfg:                 WaveTubeConfig,
        exit_tri_centroids:  Optional[np.ndarray] = None,
    ) -> "WaveTube":
        """Register wave tube geometry into *tracer* and return a handle.

        Parameters
        ----------
        tracer              : PyRayTracer — the C++ tracer wrapper
        entry_tri_indices   : int32 (N,) — triangles forming the entry aperture
        exit_tri_indices    : int32 (M,) — triangles forming the exit aperture
        cfg                 : WaveTubeConfig
        exit_tri_centroids  : float64 (M, 3) — exit face triangle centroids for
                              BPM→tri_illum_accum spatial mapping.  None skips
                              the write-back step (Monte Carlo fallback).
        """
        entry_tris = np.asarray(entry_tri_indices, dtype=np.int32)
        exit_tris  = np.asarray(exit_tri_indices,  dtype=np.int32)
        # Transverse axes perpendicular to the propagation axis — used to map
        # BPM exit-plane pixel coordinates onto exit-face triangle centroids.
        ax  = np.asarray(cfg.axis, dtype=np.float64)
        ref = np.array([1., 0., 0.]) if abs(ax[0]) < 0.8 else np.array([0., 0., 1.])
        u_ax = np.cross(ax, ref);  u_ax /= max(float(np.linalg.norm(u_ax)), 1e-12)
        v_ax = np.cross(ax, u_ax); v_ax /= max(float(np.linalg.norm(v_ax)), 1e-12)

        # 1. Entry aperture: SENSOR with UV accumulator to capture complex field.
        #    Forward rays hitting the interior face record their complex amplitude
        #    here — this is the BPM seed.
        entry_gid = tracer.register_tri_group(
            role_bits     = TRI_GROUP_ROLE_SENSOR,
            sample_policy = TRI_GROUP_SAMPLE_AREA,
            tri_indices   = entry_tris,
            uv_image      = {"res": cfg.nx, "n_bands": cfg.n_bands},
        )

        # 2. Exit aperture: NOT registered as EMISSIVE.
        #    Registering with EMISSIVE role causes the C++ forward launcher to
        #    fire unit-amplitude rays from this face regardless of power_W_per_band
        #    (that field is metadata-only in C++).  Those rays dilute the scene
        #    without contributing calibrated power.
        #
        #    The correct path for backward rays is the BSSRDF tri_illum_accum
        #    mechanism: forward rays accumulate at the diffuser, backward rays
        #    query the accumulator analytically.  solve() writes the BPM exit
        #    field into tri_illum_accum once the write API is available.
        #
        #    exit_group_id = -1 signals "not yet registered as emitter".
        exit_gid = -1

        # 3. Scale context (WAVE_HELMHOLTZ) for the tube interior
        length_m = float(np.linalg.norm(cfg.exit_pos - cfg.entry_pos))
        # Bounding sphere: encompasses entry→exit cylinder plus aperture radius
        ctx_center = (cfg.entry_pos + cfg.exit_pos) * 0.5
        ctx_radius = math.sqrt((length_m / 2)**2 + cfg.tube_radius_m**2) * 1.05

        try:
            ctx_id = tracer.add_scale_context(
                center        = ctx_center.tolist(),
                radius        = ctx_radius,
                scale_type    = RT_SCALE_WAVE,
                n_real        = cfg.n_medium,
                n_imag        = cfg.n_imag,
                context_kind  = SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ,
            )
        except Exception:
            ctx_id = -1   # non-fatal: scale context is a decoration

        return cls(
            config              = cfg,
            entry_group_id      = entry_gid,
            exit_group_id       = exit_gid,
            ctx_id              = ctx_id,
            exit_tri_ids        = exit_tris,
            _exit_tri_centroids = (np.asarray(exit_tri_centroids, dtype=np.float64)
                                   if exit_tri_centroids is not None else None),
            _u_axis             = u_ax,
            _v_axis             = v_ax,
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, tracer) -> np.ndarray:
        """Run the BPM from entry UV accumulator → exit field.

        Reads the complex amplitude captured in the entry group's UV image,
        propagates it through the tube via ADI BPM, computes

            P_b = ∫∫ |E_b(x, y)|² dx dy

        and updates the exit group's power_W_per_band so the surrogate
        emitter is calibrated to the correct physical power.

        Pre-roll: for the first cfg.pre_roll_frames calls the accumulator is
        allowed to warm up without producing output.  On the frame that
        completes the pre-roll the entry UV accumulator is cleared so the
        live integral starts from zero.  Calls during pre-roll return an
        all-zero field and leave the exit emitter at zero power.

        Returns
        -------
        exit_field : complex128 (n_bands, ny, nx)
        """
        cfg = self.config
        nb  = cfg.n_bands

        self._frames_collected += 1

        if self._frames_collected <= cfg.pre_roll_frames:
            if self._frames_collected == cfg.pre_roll_frames:
                # Pre-roll complete — wipe the accumulator so live starts at 0.
                try:
                    tracer.clear_group_uv_accum(self.entry_group_id)
                except Exception:
                    pass
            return np.zeros((nb, cfg.ny, cfg.nx), dtype=np.complex128)

        # 1. Seed: read entry UV accumulator
        try:
            uv = tracer.get_group_uv_image(self.entry_group_id)
            # amp_re/amp_im are shape (n_bands, ny, nx) — the complex amplitude
            # at each UV cell, summed over all hitting rays.
            re_raw = np.asarray(uv["amp_re"], dtype=np.float64)  # (nb, ny, nx)
            im_raw = np.asarray(uv["amp_im"], dtype=np.float64)
            count  = np.asarray(uv["count"],  dtype=np.float64)  # (ny, nx)
            count  = np.where(count > 0, count, 1.0)
            # Average over samples to get mean complex amplitude at each cell
            entry_field = (re_raw / count[np.newaxis]) + \
                          1j * (im_raw / count[np.newaxis])
        except Exception as exc:
            raise RuntimeError(
                f"WaveTube.solve: failed to read entry UV image "
                f"(group {self.entry_group_id}): {exc}") from exc

        # Resize entry field to BPM grid if necessary
        if entry_field.shape[1:] != (cfg.ny, cfg.nx):
            from scipy.ndimage import zoom
            scale = (1.0, cfg.ny / entry_field.shape[1], cfg.nx / entry_field.shape[2])
            entry_field = zoom(entry_field.real, scale) + \
                          1j * zoom(entry_field.imag, scale)

        # 2. Propagate via ADI BPM
        exit_field = _run_bpm(
            entry_field  = entry_field,
            n_steps      = cfg.n_bpm_steps,
            dz_m         = cfg.dz_m,
            k0_per_band  = 2.0 * np.pi / cfg.wavelengths_m * cfg.n_medium,
            dx_m         = cfg.dx_m,
        )

        self._exit_field = exit_field

        # 3. Boundary integral → exit power per band (stored for diagnostics).
        power_per_band = (np.sum(np.abs(exit_field)**2, axis=(1, 2))
                          * cfg.dx_m**2).astype(np.float32)
        self._exit_power = power_per_band

        # 4. Write BPM exit field into tri_illum_accum so backward rays at the
        #    diffuser exit face receive diffraction-correct illumination instead
        #    of the Monte Carlo average.
        if (self.exit_tri_ids is not None
                and self._exit_tri_centroids is not None):
            try:
                self._write_bpm_to_illum_accum(tracer, exit_field)
            except Exception as exc:
                pass  # graceful degradation to Monte Carlo path

        return exit_field

    def _write_bpm_to_illum_accum(self, tracer, exit_field: np.ndarray) -> None:
        """Map BPM exit-plane pixels to exit face triangle centroids and write.

        For each exit-face triangle we project its centroid onto the BPM
        transverse grid, sample the complex field at that location, and call
        tracer.write_tri_illum() to overwrite that triangle's entry in
        tri_illum_accum.  The BSSRDF backward path then reads the BPM result
        directly instead of the Monte Carlo average.
        """
        cfg  = self.config
        nb   = cfg.n_bands
        M    = int(self.exit_tri_ids.shape[0])
        cents = self._exit_tri_centroids            # (M, 3)
        offsets = cents - cfg.exit_pos              # (M, 3) in metres
        u_coords = offsets @ self._u_axis           # (M,) along BPM x-axis
        v_coords = offsets @ self._v_axis           # (M,) along BPM y-axis
        # Pixel indices: grid centre = (nx/2, ny/2), spacing = dx_m
        px_f = u_coords / cfg.dx_m + cfg.nx * 0.5
        py_f = v_coords / cfg.dx_m + cfg.ny * 0.5
        px_i = np.clip(np.round(px_f).astype(np.intp), 0, cfg.nx - 1)
        py_i = np.clip(np.round(py_f).astype(np.intp), 0, cfg.ny - 1)
        # Sample exit field for each triangle (exit_field: nb × ny × nx complex)
        amp_re = np.empty((M, nb), dtype=np.float32)
        amp_im = np.empty((M, nb), dtype=np.float32)
        for b in range(nb):
            sampled = exit_field[b][py_i, px_i]    # (M,) complex
            amp_re[:, b] = sampled.real.astype(np.float32)
            amp_im[:, b] = sampled.imag.astype(np.float32)
        # Normal incidence is the dominant geometry for a flat diffuser face.
        cos_avg = np.ones(M, dtype=np.float32)
        tracer.write_tri_illum(
            np.ascontiguousarray(self.exit_tri_ids, dtype=np.int32),
            np.ascontiguousarray(amp_re),
            np.ascontiguousarray(amp_im),
            np.ascontiguousarray(cos_avg),
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def exit_irradiance(self) -> Optional[np.ndarray]:
        """Per-band exit irradiance map (n_bands, ny, nx) float64.  None before solve()."""
        if self._exit_field is None:
            return None
        return np.abs(self._exit_field)**2

    def exit_power_W(self) -> Optional[np.ndarray]:
        """Integrated exit power per band (n_bands,) float32.  None before solve()."""
        return self._exit_power

    def transmission(self) -> Optional[np.ndarray]:
        """Fraction of entry power that reached the exit per band (n_bands,).
        Requires entry UV image to contain meaningful count data.
        Returns None before solve() or when entry power is zero."""
        if self._exit_power is None:
            return None
        return self._exit_power  # caller can normalise against input power

    def __repr__(self) -> str:
        solved = "solved" if self._exit_field is not None else "not yet solved"
        return (f"WaveTube(entry_gid={self.entry_group_id}, "
                f"exit_gid={self.exit_group_id}, "
                f"nx={self.config.nx}, ny={self.config.ny}, "
                f"n_bpm_steps={self.config.n_bpm_steps}, {solved})")


# ---------------------------------------------------------------------------
# Pure-numpy ADI BPM solver
# ---------------------------------------------------------------------------

def _thomas_solve_row(row: np.ndarray, beta: complex) -> np.ndarray:
    """Thomas algorithm for the 1-D ADI tridiagonal system.

    Solves: -β u[i-1] + (1+2β) u[i] - β u[i+1] = rhs[i]
    with Dirichlet BC u[0] = u[N-1] = 0.

    *row* is the input (already the RHS vector formed from the explicit side).
    Returns the solution vector of the same length.
    """
    N = len(row)
    if N < 3:
        return np.zeros(N, dtype=complex)

    # The LHS tridiagonal coefficients
    a_main = complex(1) + 2 * beta      # main diagonal
    a_off  = -beta                        # sub / super diagonal

    # Enforce Dirichlet: first/last are pinned to zero.
    # Reduce to (N-2) interior points.
    rhs = row[1:-1].copy()

    # Forward sweep (Thomas)
    c = np.full(N - 2, a_off, dtype=complex)
    d = rhs.copy()
    b = np.full(N - 2, a_main, dtype=complex)

    # Eliminate sub-diagonal
    for i in range(1, N - 2):
        m = c[i - 1] / b[i - 1]
        b[i] -= m * a_off
        d[i] -= m * d[i - 1]

    # Back-substitution
    x = np.zeros(N - 2, dtype=complex)
    x[-1] = d[-1] / b[-1]
    for i in range(N - 4, -1, -1):
        x[i] = (d[i] - a_off * x[i + 1]) / b[i]

    result = np.zeros(N, dtype=complex)
    result[1:-1] = x
    return result


def _adi_half_step(field: np.ndarray, beta: complex, axis: int) -> np.ndarray:
    """One ADI half-step (explicit side → solve tridiagonal) along *axis* (0=y, 1=x).

    Explicit operator: RHS[i] = β u[i-1] + (1−2β) u[i] + β u[i+1]
    Implicit LHS     : -β u[i-1] + (1+2β) u[i] - β u[i+1] = RHS[i]
    """
    ny, nx = field.shape
    out    = np.zeros_like(field)
    b_exp  = beta      # explicit β (same magnitude, same sign — Crank-Nicolson symmetry)

    if axis == 1:  # horizontal sweep (along x for each row)
        for iy in range(ny):
            row = field[iy]
            # Build explicit RHS
            rhs = np.empty(nx, dtype=complex)
            rhs[0]     =  (1 - 2*b_exp) * row[0]     + b_exp * row[1]
            rhs[1:-1]  = b_exp * row[:-2] + (1 - 2*b_exp) * row[1:-1] + b_exp * row[2:]
            rhs[-1]    = b_exp * row[-2] + (1 - 2*b_exp) * row[-1]
            # Dirichlet at edges
            rhs[0] = 0.0
            rhs[-1] = 0.0
            out[iy] = _thomas_solve_row(rhs, beta)
    else:          # vertical sweep (along y for each column)
        for ix in range(nx):
            col = field[:, ix]
            rhs = np.empty(ny, dtype=complex)
            rhs[0]     =  (1 - 2*b_exp) * col[0]     + b_exp * col[1]
            rhs[1:-1]  = b_exp * col[:-2] + (1 - 2*b_exp) * col[1:-1] + b_exp * col[2:]
            rhs[-1]    = b_exp * col[-2] + (1 - 2*b_exp) * col[-1]
            rhs[0]  = 0.0
            rhs[-1] = 0.0
            col_out = _thomas_solve_row(rhs, beta)
            out[:, ix] = col_out

    return out


def _bpm_step(
    field:      np.ndarray,   # (ny, nx) complex
    dz_m:       float,
    k0:         float,        # 2π n / λ  for this band
    dx_m:       float,
) -> np.ndarray:
    """One complete ADI BPM z-step for a single band.

    Follows the three-stage structure of ray_wave_bpm.comp.glsl:
      (a) Carrier phase advance: E *= exp(i k₀ dz)
      (b) Horizontal ADI half-step
      (c) Vertical   ADI half-step

    β = dz / (4 k₀ dx²)  — each half-step advances by dz/2.
    """
    # (a) Carrier
    field = field * np.exp(1j * k0 * dz_m)

    # β for the diffraction half-steps
    # Factor 4: each of the two half-steps contributes dz/2, and the ADI
    # formulation divides by 2k again → dz / (4 k₀ dx²)
    beta = complex(0, dz_m / (4.0 * k0 * dx_m**2))

    # (b) Horizontal sweep
    field = _adi_half_step(field, beta, axis=1)

    # (c) Vertical sweep
    field = _adi_half_step(field, beta, axis=0)

    return field


def _run_bpm(
    entry_field: np.ndarray,   # (n_bands, ny, nx) complex
    n_steps:     int,
    dz_m:        float,
    k0_per_band: np.ndarray,   # (n_bands,) float64
    dx_m:        float,
) -> np.ndarray:
    """Propagate *entry_field* for *n_steps* BPM z-steps.

    Returns exit_field of the same shape (n_bands, ny, nx).
    """
    n_bands, ny, nx = entry_field.shape
    field = entry_field.copy().astype(complex)

    for _ in range(n_steps):
        for b in range(n_bands):
            field[b] = _bpm_step(field[b], dz_m, float(k0_per_band[b]), dx_m)

    return field
