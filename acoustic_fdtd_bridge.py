"""acoustic_fdtd_bridge.py
========================
Python bridge between the guitar geometry, the C AcousticFDTD solver, and the
existing geometric ray tracer and the structural AcousticCoEvolver.

Three questions answered here
------------------------------
1.  Structural excitation
    The active guitar path uses ``AcousticCoEvolver``: string endpoint tension
    forces drive a bridge/top impedance, the equal reaction loads the plate,
    and only plate motion couples energy into the pressure field.  The older
    ``AcousticFDTD.inject_bridge(val, ddt)`` pressure shortcut is disabled in
    the C extension because it bypassed plate impedance.

2.  Smooth volumetric field
    The FDTD maintains a continuous 3-D pressure field P(x,y,z,t) sampled at
    the wave physics sample rate.  Calling ``get_pressure_field()`` returns the
    full (Nx, Ny, Nz) float32 array, which can be uploaded as a 3-D GL texture
    for trilinear-interpolated volume rendering — giving visually smooth, fluid
    animation of the acoustic field rather than sparse ray hit accumulations.

    The ``FDTDCoEvolver.get_volume_slice(axis, idx)`` helper returns a 2-D
    slice for display as a colour-mapped overlay on the existing GL viewport.

3.  Co-evolution with the geometric ray tracer
    ``FDTDCoEvolver`` runs both engines in parallel:
      • FDTD (this module)  — updated every audio sample block, provides
        wave-accurate response below ~1.5 kHz: standing body modes, plate
        resonances, near-field pressure distribution.
      • Ray tracer          — re-traced at render time, provides geometric
        reflections and late reverberation above the Schroeder crossover.
    The ``get_mixed_response()`` method blends the FDTD impulse response
    (extracted at microphone positions) with the ray-tracer transfer function
    via a smooth crossover filter at ``crossover_hz``.

Usage
-----
    from acoustic_fdtd_bridge import FDTDCoEvolver
    from sm_plugins.orchestral_resonance import _body_panels_string_plate, _build_body_scene

    scene  = _build_body_scene("string_plate")
    coev   = FDTDCoEvolver.from_body_scene(scene, dx=0.008)
    coev.prime(audio_block_size=512)

    # Per-audio-block:
    coev.push_driver_block(string_signal_sum)     # (block_size,) float32
    P_field = coev.get_pressure_field()           # (Nx, Ny, Nz) float32
    w_plate = coev.get_plate_displacement()       # (Nx, Ny)     float32

    # Visualisation slice (XY plane at body mid-depth):
    heatmap = coev.get_volume_slice(axis=2, idx=coev.fdtd.Nz // 2)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from acoustic_amr import (
    AcousticAMRGrid,
    build_guitar_amr_grid,
    build_amr_coevolver_descriptor,
    BorderConditionSpec,
    border_anechoic,
)

try:
    from ray_tracer_bridge import (
        solve_highband_mic_transfer as _solve_highband,
        apply_mic_transfer_frequency_domain as _apply_highband,
    )
    _HAS_HIGHBAND = True
except ImportError:
    _HAS_HIGHBAND = False

try:
    from _spectral_kernels import AcousticFDTD as _CAcousticFDTD
    _HAS_FDTD = True
except ImportError:
    _CAcousticFDTD = None
    _HAS_FDTD = False

try:
    from _spectral_kernels import AcousticCoEvolver as _CAcousticCoEvolver
    _HAS_COEVOLVER = True
except ImportError:
    _CAcousticCoEvolver = None
    _HAS_COEVOLVER = False

# Cell type constants matching acoustic_fdtd.h
FDTD_AIR   = np.uint8(0)
FDTD_WALL  = np.uint8(1)
FDTD_PLATE = np.uint8(2)
FDTD_PML   = np.uint8(3)


# ---------------------------------------------------------------------------
# F: Aperture-averaged microphone sampler
# ---------------------------------------------------------------------------

def _aperture_ring_positions(
    center: np.ndarray,
    axis:   np.ndarray,
    aperture_r: float,
    n: int,
) -> list[np.ndarray]:
    """Return N positions on a disc of radius aperture_r centred at center, perpendicular to axis."""
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / max(float(np.linalg.norm(ax)), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ax, ref);  u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(ax, u)
    return [
        (np.asarray(center, dtype=np.float32)
         + float(aperture_r) * (math.cos(2 * math.pi * k / n) * u
                                + math.sin(2 * math.pi * k / n) * v).astype(np.float32))
        for k in range(n)
    ]


class ApertureAveragedMicSampler:
    """Average a centre mic + ring of mics into one output block.

    Registered as info['aperture_mic'] by build_acoustic_coevolver_from_scene.
    Usage::

        mic_out = aperture_mic.get_output(co, n_samples)
    """

    def __init__(self, center_idx: int, ring_indices: list[int]) -> None:
        self._center = center_idx
        self._ring   = ring_indices

    def get_output(self, co, n_samples: int) -> np.ndarray:
        """Return aperture-averaged mic signal (float32, length n_samples)."""
        acc = co.get_mic_output(self._center, n_samples).astype(np.float64)
        for i in self._ring:
            acc += co.get_mic_output(i, n_samples)
        return (acc / (1 + len(self._ring))).astype(np.float32)


# ---------------------------------------------------------------------------
# G: Float64 accumulation for long decays and reciprocity tests
# ---------------------------------------------------------------------------

class Float64MicAccumulator:
    """Collects mic output blocks in float64 to prevent precision loss in long decays.

    Usage::

        acc = Float64MicAccumulator()
        for block in ...:
            co.step(...)
            acc.push(co.get_mic_output(0, block_size))
        signal = acc.get()   # float64 concatenated signal
    """

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    def push(self, block) -> None:
        self._chunks.append(np.asarray(block, dtype=np.float64))

    def get(self) -> np.ndarray:
        if not self._chunks:
            return np.array([], dtype=np.float64)
        return np.concatenate(self._chunks)

    def reset(self) -> None:
        self._chunks = []

    def __len__(self) -> int:
        return sum(c.size for c in self._chunks)


# ---------------------------------------------------------------------------
# 3: High-band mic transfer (ray path above crossover)
# ---------------------------------------------------------------------------

class HighbandMicPath:
    """Precomputed high-frequency source-to-mic ray transfer.

    Build once via build_acoustic_coevolver_from_scene (stored as
    info['highband_mic']), then call apply() each block to get the
    geometric high-band contribution to add to the FDTD mic output.

    Usage::

        highband = info.get('highband_mic')
        if highband:
            hb_out = highband.apply(string_drive_block)  # (1, n_samples) float32
            mic_out = fdtd_mic_out + hb_out[0]
    """

    def __init__(
        self,
        H_high:       np.ndarray,   # (n_sources, n_mics, n_bands) complex128
        freq_hz:      np.ndarray,   # (n_bands,) float64
        crossover_hz: float,
        sample_rate:  float,
    ) -> None:
        self.H_high       = np.asarray(H_high,  dtype=np.complex128)
        self.freq_hz      = np.asarray(freq_hz, dtype=np.float64)
        self.crossover_hz = float(crossover_hz)
        self.sample_rate  = float(sample_rate)

    def apply(self, source_signals, n_out: int | None = None) -> np.ndarray:
        """Apply the high-band transfer to source signals.

        Parameters
        ----------
        source_signals : (n_sources, n_samples) or (n_samples,) float32/64
        n_out          : output length; defaults to input length

        Returns
        -------
        (n_mics, n_out) float32 — add to FDTD mic output (same sign, same units)
        """
        x = np.asarray(source_signals)
        if x.ndim == 1:
            x = x[None, :]
        n   = int(n_out or x.shape[1])
        fft_n = max(n, x.shape[1])
        bins  = np.fft.rfftfreq(fft_n, d=1.0 / self.sample_rate)
        X     = np.fft.rfft(x, n=fft_n, axis=1)
        n_mics = self.H_high.shape[1]
        Y     = np.zeros((n_mics, len(bins)), dtype=np.complex128)
        for si in range(self.H_high.shape[0]):
            for mi in range(n_mics):
                h_re = np.interp(bins, self.freq_hz, self.H_high[si, mi].real,
                                 left=0.0, right=0.0)
                h_im = np.interp(bins, self.freq_hz, self.H_high[si, mi].imag,
                                 left=0.0, right=0.0)
                Y[mi] += X[si] * (h_re + 1j * h_im)
        y = np.fft.irfft(Y, n=fft_n, axis=1)[:, :n]
        return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Guitar body voxeliser
# ---------------------------------------------------------------------------

def voxelise_guitar_body(
    outline_pts:     np.ndarray,   # (N, 2) float32, XY contour (CCW)
    body_h:          float,        # rib height (Z extent) in metres
    dx:              float = 0.008,
    pad_cells:       int   = 64,   # free-air + PML cells around body each side
    n_pml:           int   = 28,   # PML cells at grid edge (innermost pad_cells-n_pml are free air)
    *,
    amr_enabled: bool = False,
    max_refinement_level: int = 3,
    importance_policy: dict[str, int] | None = None,
    balance_refinement: bool = True,
    soundhole: tuple[float, float, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict] | AcousticAMRGrid:
    """Voxelise the guitar outline into a 3-D FDTD cell-type grid.

    Returns
    -------
    cell_type    : (Nx, Ny, Nz) uint8 array, flat index k+Nz*(j+Ny*i)
    plate_active : (Nx, Ny)     uint8 array — 1 inside guitar outline
    info         : dict with grid dimensions and coordinate offsets
    """
    if amr_enabled:
        return build_guitar_amr_grid(
            outline_pts,
            body_h,
            dx=dx,
            pad_cells=pad_cells,
            n_pml=n_pml,
            soundhole=soundhole,
            max_refinement_level=max_refinement_level,
            importance_policy=importance_policy,
            balance_refinement=balance_refinement,
        )

    # Bounding box of the outline
    ox = outline_pts[:, 0]
    oy = outline_pts[:, 1]
    x_min, x_max = float(ox.min()), float(ox.max())
    y_min, y_max = float(oy.min()), float(oy.max())
    z_min, z_max = 0.0, body_h

    # Grid dimensions: body extent + padding on each side
    pad = pad_cells * dx
    gx_min = x_min - pad;  gx_max = x_max + pad
    gy_min = y_min - pad;  gy_max = y_max + pad
    gz_min = z_min - pad;  gz_max = z_max + pad

    Nx = max(4, int(math.ceil((gx_max - gx_min) / dx)))
    Ny = max(4, int(math.ceil((gy_max - gy_min) / dx)))
    Nz = max(4, int(math.ceil((gz_max - gz_min) / dx)))

    # Grid coordinate arrays (cell centres)
    xs = gx_min + (np.arange(Nx) + 0.5) * dx
    ys = gy_min + (np.arange(Ny) + 0.5) * dx
    zs = gz_min + (np.arange(Nz) + 0.5) * dx

    # Point-in-polygon: which (i, j) cells are inside the guitar outline?
    X2, Y2 = np.meshgrid(xs, ys, indexing='ij')  # (Nx, Ny)
    inside_outline = _pip_grid(X2, Y2, outline_pts)   # (Nx, Ny) bool

    # Z indices for the body walls
    iz_back  = int(np.argmin(np.abs(zs - z_min)))  # back plate
    iz_top   = int(np.argmin(np.abs(zs - z_max)))  # top plate (soundboard)

    cell_type = np.full((Nx, Ny, Nz), FDTD_AIR, dtype=np.uint8)

    for i in range(Nx):
        for j in range(Ny):
            if inside_outline[i, j]:
                # Rib walls: cells at the perimeter in Z (back & top plate)
                cell_type[i, j, iz_back] = FDTD_WALL   # back plate (rigid)
                cell_type[i, j, iz_top]  = FDTD_PLATE  # top plate (soundboard)
                # Interior body cavity: air
                # Rib side walls: mark cells outside guitar outline at body depth
            else:
                # Cells outside guitar outline inside the Z body range are walls
                for k in range(Nz):
                    z = zs[k]
                    if z_min - dx < z < z_max + dx:
                        cell_type[i, j, k] = FDTD_WALL

    # Plate active mask: cells inside the guitar outline at the top plate layer
    plate_active = inside_outline.astype(np.uint8)   # (Nx, Ny)

    # exterior_mask: True for cells OUTSIDE the guitar body cavity.
    # Interior cavity = inside XY outline + between back plate and top plate in Z.
    # Exterior encompasses the free-air buffer, PML zone, and rib walls.
    interior = np.zeros((Nx, Ny, Nz), dtype=bool)
    for k in range(iz_back + 1, iz_top):
        interior[:, :, k] = inside_outline
    exterior_mask = ~interior   # (Nx, Ny, Nz) bool

    info = {
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'dx': dx,
        'gx_min': gx_min, 'gy_min': gy_min, 'gz_min': gz_min,
        'plate_iz': iz_top,
        'iz_back':  iz_back,
        'xs': xs, 'ys': ys, 'zs': zs,
        'n_pml': n_pml,
        'exterior_mask': exterior_mask,   # (Nx, Ny, Nz) bool — True outside guitar cavity
    }
    return cell_type, plate_active, info


def voxelise_guitar_body_amr(
    outline_pts: np.ndarray,
    body_h: float,
    dx: float = 0.008,
    pad_cells: int = 64,
    n_pml: int = 28,
    *,
    soundhole: tuple[float, float, float],
    max_refinement_level: int = 3,
    importance_policy: dict[str, int] | None = None,
    balance_refinement: bool = True,
) -> AcousticAMRGrid:
    """Build the strict AMR pressure grid.

    In AMR mode ``dx`` is the base/coarsest cell size. Unsupported geometry,
    missing soundhole data, unstable refinement policy, or invalid topology
    raises during construction; this function never falls back to the dense
    uniform grid.
    """
    return build_guitar_amr_grid(
        outline_pts,
        body_h,
        dx=dx,
        pad_cells=pad_cells,
        n_pml=n_pml,
        soundhole=soundhole,
        max_refinement_level=max_refinement_level,
        importance_policy=importance_policy,
        balance_refinement=balance_refinement,
    )


def _scene_soundhole(scene):
    room = getattr(scene, 'room', None) or getattr(scene, 'geometry', None)
    for panel in getattr(room, 'baffles', []) or []:
        sh = getattr(panel, 'soundhole', None)
        if sh is not None:
            return float(sh[0]), float(sh[1]), float(sh[2])
    return None


def _apply_soundhole_cutout(cell_type: np.ndarray, plate_active: np.ndarray,
                            info: dict, soundhole) -> None:
    if soundhole is None:
        return
    cx, cy, r = soundhole
    xs = info['gx_min'] + (np.arange(info['Nx']) + 0.5) * float(info['dx'])
    ys = info['gy_min'] + (np.arange(info['Ny']) + 0.5) * float(info['dx'])
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
    plate_active[mask] = 0
    cell_type[:, :, int(info['plate_iz'])][mask] = FDTD_AIR


def _pip_grid(X: np.ndarray, Y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon for a 2-D grid."""
    inside = np.zeros(X.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        cross = ((yi > Y) != (yj > Y)) & (
            X < (xj - xi) * (Y - yi) / (yj - yi + 1e-15) + xi
        )
        inside ^= cross
        j = i
    return inside


def _pip_points(xs: np.ndarray, ys: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon for flat arrays of (x, y) pairs."""
    inside = np.zeros(len(xs), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        cross = ((yi > ys) != (yj > ys)) & (
            xs < (xj - xi) * (ys - yi) / (yj - yi + 1e-15) + xi
        )
        inside ^= cross
        j = i
    return inside


def compute_cut_cell_fractions(
    cell_type:   np.ndarray,   # (Nx, Ny, Nz) uint8
    plate_active: np.ndarray,  # (Nx, Ny) uint8
    outline_pts: np.ndarray,   # (N, 2) float32 — guitar outline polygon
    info:        dict,
) -> tuple:
    """Compute per-face open-area fractions for cut-cell body boundary modelling.

    Uses 2×2 sub-cell sampling on each boundary face (one solid, one air
    neighbour) to estimate the fraction of the face area actually inside the
    guitar body.  Interior faces stay 1.0; fully-solid faces stay 0.0.

    Returns
    -------
    vx_frac : float32 flat array, length (Nx-1)*Ny*Nz
    vy_frac : float32 flat array, length Nx*(Ny-1)*Nz
    vz_frac : float32 flat array, length Nx*Ny*(Nz-1)
    """
    Nx, Ny, Nz = info['Nx'], info['Ny'], info['Nz']
    dx        = float(info['dx'])
    gx_min    = float(info['gx_min'])
    gy_min    = float(info['gy_min'])
    gz_min    = float(info['gz_min'])
    plate_iz  = int(info['plate_iz'])
    iz_back   = int(info.get('iz_back', 0))

    # Body Z extent for the sub-point inside test
    z_body_min = gz_min + iz_back * dx
    z_body_max = gz_min + (plate_iz + 0.5) * dx   # includes plate-layer Vz faces

    solid = (cell_type == FDTD_WALL) | (cell_type == FDTD_PLATE)  # (Nx, Ny, Nz)

    # ── Vx fracs: (Nx-1, Ny, Nz) ──────────────────────────────────────────────
    solid_L = solid[:-1, :, :]   # left  cell of each Vx face
    solid_R = solid[1:,  :, :]   # right cell
    vx_frac = np.where(solid_L | solid_R, 0.0, 1.0).astype(np.float32)
    # Boundary faces: exactly one solid neighbour — compute partial fraction
    bdry = solid_L ^ solid_R     # (Nx-1, Ny, Nz) bool
    if np.any(bdry):
        iis, jjs, kks = np.where(bdry)
        x_face = gx_min + (iis + 1) * dx
        fsum = np.zeros(len(iis), dtype=np.float32)
        for dy in (0.25, 0.75):
            for dz in (0.25, 0.75):
                y_sub = gy_min + (jjs + dy) * dx
                z_sub = gz_min + (kks + dz) * dx
                in_xy = _pip_points(x_face, y_sub, outline_pts)
                in_z  = (z_sub >= z_body_min) & (z_sub <= z_body_max)
                fsum += (in_xy & in_z).astype(np.float32)
        vx_frac[iis, jjs, kks] = fsum / 4.0

    # ── Vy fracs: (Nx, Ny-1, Nz) ──────────────────────────────────────────────
    solid_L = solid[:, :-1, :]
    solid_R = solid[:, 1:,  :]
    vy_frac = np.where(solid_L | solid_R, 0.0, 1.0).astype(np.float32)
    bdry = solid_L ^ solid_R
    if np.any(bdry):
        iis, jjs, kks = np.where(bdry)
        y_face = gy_min + (jjs + 1) * dx
        fsum = np.zeros(len(iis), dtype=np.float32)
        for dx_ in (0.25, 0.75):
            for dz in (0.25, 0.75):
                x_sub = gx_min + (iis + dx_) * dx
                z_sub = gz_min + (kks + dz)  * dx
                in_xy = _pip_points(x_sub, y_face, outline_pts)
                in_z  = (z_sub >= z_body_min) & (z_sub <= z_body_max)
                fsum += (in_xy & in_z).astype(np.float32)
        vy_frac[iis, jjs, kks] = fsum / 4.0

    # ── Vz fracs: (Nx, Ny, Nz-1) ──────────────────────────────────────────────
    solid_L = solid[:, :, :-1]
    solid_R = solid[:, :, 1:]
    vz_frac = np.where(solid_L | solid_R, 0.0, 1.0).astype(np.float32)
    # Plate-layer Vz faces: plate injection drives these — always 1.0 where active
    if 0 < plate_iz < Nz - 1:
        pa = plate_active.astype(bool)
        vz_frac[:, :, plate_iz - 1][pa] = 1.0
        vz_frac[:, :, plate_iz    ][pa] = 1.0
    # Boundary faces excluding plate-layer (those are handled above)
    bdry = solid_L ^ solid_R
    if 0 < plate_iz < Nz - 1:
        bdry[:, :, plate_iz - 1] = False
        bdry[:, :, plate_iz    ] = False
    if np.any(bdry):
        iis, jjs, kks = np.where(bdry)
        z_face = gz_min + (kks + 1) * dx
        fsum = np.zeros(len(iis), dtype=np.float32)
        for dx_ in (0.25, 0.75):
            for dy in (0.25, 0.75):
                x_sub = gx_min + (iis + dx_) * dx
                y_sub = gy_min + (jjs + dy)  * dx
                in_xy = _pip_points(x_sub, y_sub, outline_pts)
                in_z  = (z_face >= z_body_min) & (z_face <= z_body_max)
                fsum += (in_xy & in_z).astype(np.float32)
        vz_frac[iis, jjs, kks] = fsum / 4.0

    return (vx_frac.ravel(), vy_frac.ravel(), vz_frac.ravel())


# ---------------------------------------------------------------------------
# Bridge source kernel builder
# ---------------------------------------------------------------------------

def build_bridge_kernel(
    bridge_positions_xy: list,   # [(x, y), ...] world-space saddle positions
    info:                dict,   # from voxelise_guitar_body
    sigma_m:             float = 0.015,  # Gaussian spread (metres)
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute flat cell indices + Gaussian weights for bridge saddle positions.

    Uses only FDTD_AIR or FDTD_PLATE cells in the layer just below the
    soundboard (iz = plate_iz − 1) so the source drives the plate from inside
    the body cavity.

    Returns
    -------
    cell_indices : (n_cells,) int32
    weights      : (n_cells,) float32
    """
    Nx    = info['Nx'];  Ny = info['Ny'];  Nz = info['Nz']
    xs    = info['xs'];  ys = info['ys']
    iz    = max(0, info['plate_iz'] - 1)   # just below top plate
    dx    = info['dx']
    sigma2 = sigma_m * sigma_m

    idx_list = []
    wgt_list = []

    for bx, by in bridge_positions_xy:
        for i in range(Nx):
            for j in range(Ny):
                r2 = (xs[i] - bx) ** 2 + (ys[j] - by) ** 2
                w  = math.exp(-r2 / (2 * sigma2))
                if w < 1e-4:
                    continue
                flat = iz + Nz * (j + Ny * i)
                idx_list.append(flat)
                wgt_list.append(float(w))

    if not idx_list:
        raise ValueError("No bridge source cells found — check bridge positions overlap the grid")

    return (np.array(idx_list, dtype=np.int32),
            np.array(wgt_list, dtype=np.float32))


# ---------------------------------------------------------------------------
# Co-evolution orchestrator
# ---------------------------------------------------------------------------

class FDTDCoEvolver:
    """Runs the FDTD acoustic solver alongside the geometric ray tracer.

    Attributes
    ----------
    fdtd           : _CAcousticFDTD  — the C FDTD handle
    info           : dict            — grid metadata from voxelise_guitar_body
    crossover_hz   : float           — FDTD / ray-tracer blend crossover (Hz)
    _sig_prev      : float           — previous summed signal value (for ddt)
    _sample_rate   : int             — audio sample rate used for ddt calc
    """

    def __init__(self,
                 fdtd:            "_CAcousticFDTD",
                 info:            dict,
                 bridge_positions: list,
                 crossover_hz:    float = 1500.0,
                 sample_rate:     int   = 44100,
                 force_scale:     float = 1.0):
        if not _HAS_FDTD:
            raise RuntimeError(
                "_spectral_kernels.AcousticFDTD not available — rebuild the C extension:\n"
                "  cmake -S csrc -B csrc_build && cmake --build csrc_build --config Release"
            )
        self.fdtd            = fdtd
        self.info            = info
        self.crossover_hz    = crossover_hz
        self._sample_rate    = sample_rate
        self._force_scale    = 1.0
        self._sig_prev       = 0.0
        self._steps_per_sample = max(1, int(math.ceil(
            (1.0 / sample_rate) / fdtd.dt
        )))
        self._bridge_positions = bridge_positions

        # Register bridge sources
        idx, wgt = build_bridge_kernel(bridge_positions, info)
        fdtd.set_bridge_sources(idx, wgt)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_body_scene(
        cls,
        scene,
        dx:             float = 0.004,
        pad_cells:      int   = 12,
        n_pml:          int   = 10,
        crossover_hz:   float = 1500.0,
        sample_rate:    int   = 44100,
        force_scale:    float = 1.0,
        c:              float = 343.0,
        rho_air:        float = 1.21,
        plate_mass:     float = 7.0,
        plate_stiff:    float = 0.45,
        plate_alpha_M:  float = 2.0,
        plate_beta_K:   float = 1e-5,
    ) -> "FDTDCoEvolver":
        """Build a FDTDCoEvolver directly from a CavityScene (string_plate body).

        Extracts the guitar outline from the scene's PolygonalRoom baffles,
        voxelises the body, and wires up the bridge saddle sources.
        """
        # Extract guitar outline from top-plate panel (baffle with material_mask)
        outline_pts, body_h, bridge_positions = _extract_guitar_geometry(scene)

        cell_type, plate_active, info = voxelise_guitar_body(
            outline_pts, body_h, dx=dx, pad_cells=pad_cells, n_pml=n_pml,
        )
        _apply_soundhole_cutout(cell_type, plate_active, info, _scene_soundhole(scene))

        Nx, Ny, Nz = info['Nx'], info['Ny'], info['Nz']
        fdtd = _CAcousticFDTD(
            Nx=Nx, Ny=Ny, Nz=Nz,
            dx=dx, c=c, rho_air=rho_air,
            cell_type=cell_type.ravel(),
            plate_iz=info['plate_iz'],
            plate_active=plate_active.ravel(),
            plate_mass_density=plate_mass,
            plate_stiffness_D=plate_stiff,
            plate_alpha_M=plate_alpha_M,
            plate_beta_K=plate_beta_K,
            n_pml=n_pml,
        )

        # Cut-cell body boundary: compute sub-cell face fractions and apply.
        vx_f, vy_f, vz_f = compute_cut_cell_fractions(
            cell_type, plate_active, outline_pts, info)
        fdtd.set_face_fractions(vx_f, vy_f, vz_f)

        return cls(fdtd, info, bridge_positions,
                   crossover_hz=crossover_hz,
                   sample_rate=sample_rate,
                   force_scale=force_scale)

    # ── Signal injection ─────────────────────────────────────────────────────

    def push_driver_sample(self, signal_sum: float) -> None:
        """Legacy direct-pressure injection is intentionally disabled."""
        raise RuntimeError(
            "FDTDCoEvolver.push_driver_sample is disabled: it used direct "
            "pressure bridge injection. Use build_acoustic_coevolver_from_scene "
            "and AcousticCoEvolver for structural string-bridge-plate coupling."
        )

    def push_driver_block(self, signal_block: np.ndarray) -> None:
        """Inject a block of audio samples and advance the FDTD.

        Parameters
        ----------
        signal_block : (T,) float32 — summed string driver signal at audio rate.
            Use np.sum(per_string_signals, axis=0) before passing.
        """
        for s in signal_block:
            self.push_driver_sample(float(s))

    # ── Field access ─────────────────────────────────────────────────────────

    def get_pressure_field(self) -> np.ndarray:
        """Return the live 3-D pressure field, shape (Nx, Ny, Nz), float32.

        Suitable for upload to a 3-D GL texture for volumetric rendering.
        The field spans the guitar body + PML padding; slice at plate_iz for
        the soundboard plane.
        """
        return self.fdtd.get_pressure_field()

    def get_plate_displacement(self) -> np.ndarray:
        """Return the Kirchhoff plate displacement w(x,y), shape (Nx, Ny), metres."""
        return self.fdtd.get_plate_displacement()

    def get_volume_slice(self, axis: int = 2, idx: int = -1) -> np.ndarray:
        """Extract a 2-D pressure slice for heatmap display.

        Parameters
        ----------
        axis : 0=X, 1=Y, 2=Z (default: Z — plan view of soundboard)
        idx  : slice index along axis.  -1 = mid-plane.

        Returns
        -------
        (A, B) float32 slice.  Axis order: for axis=2 returns (Nx, Ny).
        """
        P = self.get_pressure_field()   # (Nx, Ny, Nz)
        if idx < 0:
            idx = P.shape[axis] // 2
        idx = max(0, min(P.shape[axis] - 1, idx))
        if   axis == 0: return P[idx, :, :]
        elif axis == 1: return P[:, idx, :]
        else:           return P[:, :, idx]

    def get_soundboard_pressure(self) -> np.ndarray:
        """Return pressure on the soundboard plane (Z = plate_iz), shape (Nx, Ny)."""
        return self.get_volume_slice(axis=2, idx=self.info['plate_iz'])

    def sample_at_receivers(self, receiver_world_xyz: np.ndarray) -> np.ndarray:
        """Interpolate pressure at world-space receiver positions.

        Parameters
        ----------
        receiver_world_xyz : (n_rec, 3) float32 — positions in metres,
            same coordinate frame as the guitar geometry.

        Returns
        -------
        (n_rec,) float32 — pressure at each receiver.
        """
        info = self.info
        # Convert world coords → grid coords (fractional cell indices)
        gc = np.empty_like(receiver_world_xyz)
        gc[:, 0] = (receiver_world_xyz[:, 0] - info['gx_min']) / info['dx']
        gc[:, 1] = (receiver_world_xyz[:, 1] - info['gy_min']) / info['dx']
        gc[:, 2] = (receiver_world_xyz[:, 2] - info['gz_min']) / info['dx']
        return self.fdtd.sample_pressure(gc.astype(np.float32))

    def reset(self) -> None:
        """Zero all FDTD fields and reset derivative state."""
        self.fdtd.reset()
        self._sig_prev = 0.0

    # ── Co-evolution blend ───────────────────────────────────────────────────

    def blend_with_ray_transfer(
        self,
        H_ray:      np.ndarray,   # (n_src, n_rec, n_bands) complex128
        freq_hz:    np.ndarray,   # (n_bands,) float64
        H_fdtd:     np.ndarray,   # (n_bands,) complex128 — single src/rec pair
        crossover:  float | None = None,
    ) -> np.ndarray:
        """Blend FDTD impulse response with ray transfer function.

        Uses a smooth sigmoid crossover at ``crossover_hz`` (default:
        self.crossover_hz):
          H_blended = H_fdtd · lo_weight(f) + H_ray · hi_weight(f)
          lo_weight(f) = 1 / (1 + (f / f_cross)^4)
          hi_weight(f) = 1 − lo_weight(f)

        This allows the FDTD to dominate at low frequencies (standing body
        modes, plate resonances) while the ray tracer handles geometric
        high-frequency reflections.

        Parameters
        ----------
        H_ray   : (n_bands,) complex128 — ray transfer for one src/rec pair
        H_fdtd  : (n_bands,) complex128 — FDTD transfer for same src/rec pair
        freq_hz : (n_bands,) float64

        Returns
        -------
        H_blend : (n_bands,) complex128
        """
        fc = crossover if crossover is not None else self.crossover_hz
        ratio = (freq_hz / fc) ** 4
        lo    = 1.0 / (1.0 + ratio)
        hi    = 1.0 - lo
        return H_fdtd * lo + H_ray * hi

    # ── Info ─────────────────────────────────────────────────────────────────

    @property
    def dt(self) -> float:
        return self.fdtd.dt

    @property
    def steps_per_sample(self) -> int:
        return self._steps_per_sample

    def __repr__(self) -> str:
        info = self.info
        return (f"FDTDCoEvolver("
                f"grid={info['Nx']}×{info['Ny']}×{info['Nz']}, "
                f"dx={info['dx']*1000:.1f}mm, "
                f"dt={self.dt*1e6:.1f}µs, "
                f"steps/sample={self._steps_per_sample}, "
                f"cross={self.crossover_hz:.0f}Hz)")


# ---------------------------------------------------------------------------
# Geometry extraction helpers
# ---------------------------------------------------------------------------

def _extract_guitar_geometry(scene) -> Tuple[np.ndarray, float, list]:
    """Extract (outline_pts, body_h, bridge_positions) from a CavityScene.

    Searches the scene's PolygonalRoom baffles for the string plate panels.
    Falls back to sensible defaults if the panel structure is not recognised.
    """
    room   = getattr(scene, 'room', None) or getattr(scene, 'geometry', None)
    baffles = getattr(room, 'baffles', [])

    # Find top plate panel: normal ≈ (0, 0, -1) and has material_mask
    top_panel  = None
    body_h     = 0.06   # fallback
    for p in baffles:
        n = getattr(p, 'normal', (0, 0, 0))
        if abs(n[2] + 1.0) < 0.1 and getattr(p, 'material_mask', None) is not None:
            top_panel = p
            body_h    = float(getattr(p, 'point', (0, 0, body_h))[2])
            break

    if top_panel is not None:
        explicit_outline = getattr(top_panel, 'outline_pts', None)
        if explicit_outline is not None:
            outline_pts = np.asarray(explicit_outline, dtype=np.float32)
        else:
            # Reconstruct outline from the material_mask + half_size
            half = float(getattr(top_panel, 'half_size', 0.17))
            mask = top_panel.material_mask        # (res, res) float32
            res  = mask.shape[0]
            coords = np.linspace(-half, half, res)
            X, Y   = np.meshgrid(coords, coords)  # default 'xy': X=col=x, Y=row=y
            # Find outer boundary cells of the mask
            from scipy.ndimage import binary_erosion  # type: ignore
            solid      = mask > 0.5
            boundary   = solid & ~binary_erosion(solid)
            bx         = X[boundary].astype(np.float32)
            by         = Y[boundary].astype(np.float32)
            # Order boundary points CCW
            outline_pts = _order_boundary_ccw(bx, by)
    else:
        # Fallback: default guitar outline (nominal parameters)
        outline_pts = _default_guitar_outline()
        body_h      = 0.060

    # Bridge positions: from scene sources or default
    bridge_positions = []
    for src in getattr(scene, 'sources', []):
        pos = getattr(src, 'position', None)
        if pos is not None:
            bridge_positions.append((float(pos[0]), float(pos[1])))
    if not bridge_positions:
        # Default: three bridge saddle positions
        bridge_positions = [(-0.030, -0.070), (0.000, -0.070), (0.030, -0.070)]

    return outline_pts, body_h, bridge_positions


def _order_boundary_ccw(bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Order a cloud of 2D boundary points CCW by polar angle from centroid."""
    cx, cy = bx.mean(), by.mean()
    angles = np.arctan2(by - cy, bx - cx)
    order  = np.argsort(angles)
    pts    = np.column_stack([bx[order], by[order]]).astype(np.float32)
    return pts


def _default_guitar_outline(n_pts: int = 64) -> np.ndarray:
    """Generate a default guitar outline (used when scene geometry not available)."""
    lower_r  = 0.175; upper_r  = 0.135; waist_x  = 0.105
    lower_cy = -0.090; upper_cy = 0.100
    y_bot    = lower_cy - lower_r
    y_top    = upper_cy + upper_r
    y_span   = y_top - y_bot
    n_half   = n_pts // 2
    ys       = np.linspace(y_bot + 1e-6, y_top - 1e-6, n_half)
    xs       = np.empty(n_half)
    for k, y in enumerate(ys):
        d2_lo  = max(0.0, lower_r**2 - (y - lower_cy)**2)
        d2_hi  = max(0.0, upper_r**2 - (y - upper_cy)**2)
        x_env  = max(math.sqrt(d2_lo), math.sqrt(d2_hi))
        t      = (y - y_bot) / y_span
        wenv   = math.exp(-((t - (0.0 - y_bot) / y_span) / 0.14)**2)
        xs[k]  = x_env * (1.0 - wenv) + waist_x * wenv
    right = np.column_stack([ xs,       ys      ])
    left  = np.column_stack([-xs[::-1], ys[::-1]])
    return np.concatenate([right, left]).astype(np.float32)[:n_pts]


# ---------------------------------------------------------------------------
# Volumetric field → surface illumination adapter
# ---------------------------------------------------------------------------

def fdtd_pressure_to_surface_illum(
    P_field:     np.ndarray,   # (Nx, Ny, Nz) float32
    tri_verts:   np.ndarray,   # (N_tri, 3, 3) float64 — triangle vertices
    info:        dict,
) -> np.ndarray:
    """Map the FDTD 3-D pressure field onto triangle surface illumination.

    Samples the FDTD field at each triangle centroid (trilinear interpolation)
    and returns normalised absolute pressure values as surface illumination,
    replacing the coarse nearest-centroid ray accumulation.

    This feeds directly into ``SurfaceIllumWidget.update_geometry()`` as the
    ``illum`` argument, giving a physically accurate, smooth thermal colormap
    driven by the wave-domain pressure field rather than geometric ray hits.

    Returns
    -------
    illum : (N_tri,) float32, normalised to [0, 1]
    """
    if not _HAS_FDTD:
        # Fallback: uniform illumination
        return np.ones(len(tri_verts), dtype=np.float32) * 0.1

    n_tri    = len(tri_verts)
    centroids = tri_verts.reshape(n_tri, 3, 3).mean(axis=1).astype(np.float32)

    # Convert world coords → grid coords
    dx  = info['dx']
    gc  = np.empty_like(centroids)
    gc[:, 0] = (centroids[:, 0] - info['gx_min']) / dx
    gc[:, 1] = (centroids[:, 1] - info['gy_min']) / dx
    gc[:, 2] = (centroids[:, 2] - info['gz_min']) / dx

    # We don't have an FDTD handle here, so just sample from the pre-computed field
    # via trilinear interpolation in NumPy.
    Nx, Ny, Nz = P_field.shape
    gc[:, 0] = np.clip(gc[:, 0], 0, Nx - 1.001)
    gc[:, 1] = np.clip(gc[:, 1], 0, Ny - 1.001)
    gc[:, 2] = np.clip(gc[:, 2], 0, Nz - 1.001)

    i0 = gc[:, 0].astype(np.int32);  fx = gc[:, 0] - i0
    j0 = gc[:, 1].astype(np.int32);  fy = gc[:, 1] - j0
    k0 = gc[:, 2].astype(np.int32);  fz = gc[:, 2] - k0
    i1 = np.minimum(i0 + 1, Nx - 1)
    j1 = np.minimum(j0 + 1, Ny - 1)
    k1 = np.minimum(k0 + 1, Nz - 1)

    p = (
        (1-fx)*(1-fy)*(1-fz) * P_field[i0, j0, k0]
      + (  fx)*(1-fy)*(1-fz) * P_field[i1, j0, k0]
      + (1-fx)*(  fy)*(1-fz) * P_field[i0, j1, k0]
      + (  fx)*(  fy)*(1-fz) * P_field[i1, j1, k0]
      + (1-fx)*(1-fy)*(  fz) * P_field[i0, j0, k1]
      + (  fx)*(1-fy)*(  fz) * P_field[i1, j0, k1]
      + (1-fx)*(  fy)*(  fz) * P_field[i0, j1, k1]
      + (  fx)*(  fy)*(  fz) * P_field[i1, j1, k1]
    )

    illum = np.abs(p).astype(np.float32)
    peak  = illum.max()
    if peak > 1e-12:
        illum /= peak
    return illum


# ---------------------------------------------------------------------------
# Unified co-evolver factory
# ---------------------------------------------------------------------------

# Standard guitar string parameters (bass E → treble E, indices 0–5).
# Scaled versions used for n_strings < 6.
GUITAR_SCALE_LENGTH_M = 0.648
GUITAR_TUNING_HZ = (82.4069, 110.0, 146.832, 195.998, 246.942, 329.628)
GUITAR_GAUGE_IN = (0.046, 0.036, 0.026, 0.017, 0.013, 0.010)

_GUITAR_STRING_PARAMS = [
    # (tension_N, linear_mass_kgm, damping, stiffness_EI)   index 0 = lowest / bass
    # EI values from Cuesta et al. (2017) and Valette (1995):
    #   wound strings: high EI from stiff metal core (~1e-4 N·m²)
    #   plain steel:   lower EI, scales roughly as d^4 (~2e-5 – 6e-5 N·m²)
    (73.5, 5.35e-3, 6.0,  1.8e-4),  # E2  wound bass
    (64.5, 3.78e-3, 6.5,  1.2e-4),  # A2  wound
    (61.0, 2.50e-3, 7.0,  8.0e-5),  # D3  wound
    (71.0, 1.75e-3, 8.0,  4.5e-5),  # G3  plain (or wound depending on gauge)
    (66.0, 1.04e-3, 9.0,  2.8e-5),  # B3  plain
    (63.5, 0.40e-3, 10.0, 1.6e-5),  # E4  high treble plain
]


def _string_physical_params(index: int, scale_length_m: float) -> tuple[float, float, float, float, float, float, float]:
    """Return (frequency_hz, gauge_in, tension_N, linear_mass_kgm, damping, stiffness_EI, axial_stiffness_N)."""
    base = _GUITAR_STRING_PARAMS[index]
    _tension_ref, mu, damping, stiffness_EI = base
    f0 = GUITAR_TUNING_HZ[index]
    gauge = GUITAR_GAUGE_IN[index]
    tension = (2.0 * scale_length_m * f0) ** 2 * mu
    diameter_m = gauge * 0.0254
    area_m2 = math.pi * 0.25 * diameter_m * diameter_m
    axial_stiffness = 2.0e11 * area_m2
    return float(f0), float(gauge), float(tension), float(mu), float(damping), float(stiffness_EI), float(axial_stiffness)


def build_acoustic_coevolver_from_scene(
    scene,
    n_strings:    int   = 3,
    sample_rate:  float = 44100.0,
    dx:           float = 0.004,
    pad_cells:    int   = 10,
    n_pml:        int   = 10,
    n_segs:       int   = 240,
    modal_stride: int   = 16,
    force_scale:  float = 1.0,
    c:            float = 343.0,
    rho_air:      float = 1.21,
    plate_mass:   float = 7.0,
    plate_stiff:  float = 0.45,
    plate_alpha_M: float = 2.0,
    plate_beta_K:  float = 1e-5,
    scale_length_m: float = GUITAR_SCALE_LENGTH_M,
    fretless: bool = False,
    fret_number: int = 0,
    max_refinement_level: int = 3,
    border_spec: BorderConditionSpec | None = None,
    mic_aperture_r: float = 0.012,
    n_mic_ring:     int   = 8,
    enable_highband: bool = True,
):
    """Build an AcousticCoEvolver from a CavityScene body geometry.

    Extracts the guitar outline and bridge positions from the scene,
    voxelises the body, creates approximate guitar string/pickup/mic
    descriptors, and returns a live AcousticCoEvolver handle.

    Parameters
    ----------
    scene       : CavityScene — string_plate body scene
    n_strings   : number of simulated strings (1–6)
    sample_rate : audio sample rate (Hz)
    dx          : FDTD cell size (m).  0.010 m keeps grid under ~50^3 cells.
    n_segs      : FDTD segments per string
    force_scale : deprecated compatibility parameter; structural coupling uses
                  explicit bridge/body impedances

    Returns
    -------
    co  : _CAcousticCoEvolver handle, or None if the C extension is unavailable
    info : dict — grid metadata from voxelise_guitar_body
    """
    if not _HAS_COEVOLVER:
        print(
            "[build_acoustic_coevolver_from_scene] AcousticCoEvolver unavailable "
            "(C extension not built). Run: "
            "cmake -S csrc -B csrc_build && cmake --build csrc_build --config Release"
        )
        return None, {}

    outline_pts, body_h, bridge_positions = _extract_guitar_geometry(scene)
    soundhole = _scene_soundhole(scene)

    # ── AMR grid ──────────────────────────────────────────────────────────────
    amr_grid = voxelise_guitar_body_amr(
        outline_pts, body_h,
        dx=dx, pad_cells=pad_cells, n_pml=n_pml,
        soundhole=soundhole or (0.0, 0.0, 0.03),
        max_refinement_level=max_refinement_level,
    )
    # Compute visualization grid dimensions (must match pybind from_amr_desc formula)
    _viz_dx = max(float(amr_grid.min_dx) * 4.0, 0.005)
    _bmin = amr_grid.bounds_min
    _bmax = amr_grid.bounds_max
    def _clamp256(v: int) -> int:
        return max(1, min(v, 256))
    _viz_Nx = _clamp256(round((_bmax[0] - _bmin[0]) / _viz_dx) + 1)
    _viz_Ny = _clamp256(round((_bmax[1] - _bmin[1]) / _viz_dx) + 1)
    _viz_Nz = _clamp256(round((_bmax[2] - _bmin[2]) / _viz_dx) + 1)

    info = {
        'n_cells':   amr_grid.n_cells,
        'n_faces':   amr_grid.n_faces,
        'bounds_min': _bmin.astype(np.float32),
        'bounds_max': _bmax.astype(np.float32),
        'min_dx':    float(amr_grid.min_dx),
        # Visualization uniform grid — consumed by Renderer / GL 3-D texture.
        # 'dx' must equal the viz grid spacing so that all renderer helpers
        # (_soundhole_cutout, z-clip uniforms, plate_active_2d) use the same
        # coordinate system as Nx/Ny/Nz.
        'Nx': _viz_Nx,
        'Ny': _viz_Ny,
        'Nz': _viz_Nz,
        'dx': _viz_dx,
        'gx_min': float(_bmin[0]),
        'gy_min': float(_bmin[1]),
        'gz_min': float(_bmin[2]),
    }

    bridge_z = body_h

    # ── String descriptors ────────────────────────────────────────────────────
    # Spread n_strings bridge saddle positions across the body width.
    if bridge_positions:
        y_saddle = float(np.mean([by for _, by in bridge_positions]))
    else:
        y_saddle = -0.070  # fallback: nominal bridge y
    y_nut    = y_saddle + float(scale_length_m)
    x_span   = 0.0088 * max(0, n_strings - 1)
    string_z = body_h + 0.010

    # Select string parameters: take evenly-spaced subset from _GUITAR_STRING_PARAMS
    param_indices = np.linspace(0, 5, n_strings, dtype=int)

    string_defs = []
    for si in range(n_strings):
        x_str = -x_span / 2.0 + si * (x_span / max(1, n_strings - 1))
        x_nut = x_str * 0.72
        # Path: n_segs+1 nodes from nut to saddle with a small lateral fan.
        # The pybind binding derives n_segs from len(path_xyz) - 1.
        ys_path = np.linspace(y_nut, y_saddle, n_segs + 1, dtype=np.float32)
        xs_path = np.linspace(x_nut, x_str, n_segs + 1, dtype=np.float32)
        path = np.column_stack([
            xs_path,
            ys_path,
            np.full(n_segs + 1, string_z, dtype=np.float32),
        ])  # (n_segs+1, 3) float32
        pix = param_indices[si]
        f0, gauge, tension, lin_mass, damping, stiffness_EI, axial_stiffness = _string_physical_params(
            int(pix), float(scale_length_m))
        if int(fret_number) > 0:
            # A stopped note damps the non-speaking string length behind the
            # contact point.  The current C solver models the speaking length;
            # fold the behind-fret/finger loss into the distributed damping.
            damping *= 1.18 if fretless else 1.08
        string_defs.append({
            'path_xyz':         path,       # n_segs derived from shape
            'tension_N':        float(tension),
            'linear_mass_kgm':  float(lin_mass),
            'damping':          float(damping),
            'stiffness_EI':     float(stiffness_EI),
            'axial_stiffness_N': float(axial_stiffness),
            'fundamental_hz':    float(f0),
            'gauge_in':          float(gauge),
            'scale_length_m':    float(scale_length_m),
            # Neck bending model: replaces rigid nut BC (u[0]=0) with a
            # spring-mass-damper representing the neck's fundamental bending mode.
            # The compliance and damping at the nut from neck flex increases
            # effective string length slightly and adds frequency-dependent loss.
            'neck_freq_hz':     65.0,   # ~65 Hz fundamental neck bend (acoustic guitar)
            'neck_mass_kg':     0.18,   # ~180 g effective modal mass
            'neck_Q':           35.0,   # Q~35 for a well-finished wood neck
        })

    # ── Pickup descriptor: bridge single-coil, centred on string spread ──────
    pickup_x  = 0.0
    pickup_y  = y_saddle - 0.020   # 20 mm behind bridge
    pickup_defs = [{
        'type':         0,   # PICKUP_SINGLE_COIL
        'pos':          np.array([pickup_x, pickup_y, string_z], dtype=np.float32),
        'axis':         np.array([0.0, 0.0, 1.0], dtype=np.float32),
        'pole_sigma':   0.005,
        'coil_spacing': 0.018,
        'sensitivity':  1.0,
        'string_mask':  -1,   # all strings; C++ casts int→uint32_t: -1 = 0xFFFFFFFF
    }]

    # ── Microphone: cardioid at soundhole (~75mm from top plate, aimed down) ─
    _mic_center = np.array([0.0, 0.0, body_h + 0.075], dtype=np.float32)
    _mic_axis   = np.array([0.0, 0.0, -1.0],           dtype=np.float32)
    _mic_base   = {'axis': _mic_axis, 'polar_a': 0.5, 'polar_b': 0.5, 'gain': 1.0}
    mic_defs = [{'pos': _mic_center, **_mic_base}]

    # F: aperture disc — register N ring positions as additional mics
    _aperture_sampler = None
    if mic_aperture_r > 0 and n_mic_ring > 0:
        _ring_pos = _aperture_ring_positions(_mic_center, _mic_axis, mic_aperture_r, n_mic_ring)
        _ring_indices = list(range(1, 1 + len(_ring_pos)))
        for rp in _ring_pos:
            mic_defs.append({'pos': rp, **_mic_base})
        _aperture_sampler = ApertureAveragedMicSampler(0, _ring_indices)

    # ── Bridge saddle positions ───────────────────────────────────────────────
    bridge_xyz = np.array(
        [[bx, by, bridge_z] for bx, by in bridge_positions],
        dtype=np.float32,
    )

    # ── Plate geometry: infer from grid bounding box + body_h ─────────────────
    bmin = amr_grid.bounds_min
    bmax = amr_grid.bounds_max
    plate_origin = np.array([bmin[0], bmin[1], body_h], dtype=np.float32)
    # Use the finest AMR cell size as the plate grid spacing.
    plate_dx_   = float(amr_grid.min_dx)
    plate_Nx_   = max(1, int(round((bmax[0] - bmin[0]) / plate_dx_)))
    plate_Ny_   = max(1, int(round((bmax[1] - bmin[1]) / plate_dx_)))

    if border_spec is None:
        border_spec = border_anechoic(sigma_order=3.0)

    amr_desc = build_amr_coevolver_descriptor(
        grid               = amr_grid,
        plate_Nx           = plate_Nx_,
        plate_Ny           = plate_Ny_,
        plate_dx           = plate_dx_,
        plate_origin       = plate_origin,
        body_h             = body_h,
        plate_mass_density = plate_mass,
        plate_stiffness_D  = plate_stiff,
        plate_alpha_M      = plate_alpha_M,
        plate_beta_K       = plate_beta_K,
        bridge_src_xyz     = bridge_xyz,
        bridge_sigma       = 0.012,
        c                  = c,
        rho_air            = rho_air,
        border_spec        = border_spec,
        n_pml              = n_pml,
    )

    co = _CAcousticCoEvolver.from_amr_desc(
        amr_desc     = amr_desc,
        string_defs  = string_defs,
        pickup_defs  = pickup_defs,
        mic_defs     = mic_defs,
        sample_rate  = float(sample_rate),
        modal_stride = modal_stride,
    )

    # Upload AMR cell centres once; the renderer uses them to build a GPU TBO
    # for fragment-shader IDW interpolation (no CPU scatter grid needed).
    info['n_cells_amr']      = amr_grid.n_cells
    info['amr_cell_centers'] = co.get_amr_cell_centers()  # float32 (n_cells, 3)
    info['amr_min_dx']       = float(amr_grid.min_dx)

    # F: aperture-averaged mic sampler
    if _aperture_sampler is not None:
        info['aperture_mic'] = _aperture_sampler

    # 3: high-band mic path (ray geometric detail above crossover)
    if enable_highband and _HAS_HIGHBAND and getattr(scene, 'sources', None):
        try:
            _hb_freq = np.geomspace(20.0, float(sample_rate) / 2.0, 32)
            H_high, _hb_meta = _solve_highband(
                scene,
                mic_defs=[{
                    'pos':     _mic_center,
                    'axis':    _mic_axis,
                    'polar_a': 0.5,
                    'polar_b': 0.5,
                    'aperture_r': float(mic_aperture_r) if mic_aperture_r > 0 else 0.012,
                }],
                crossover_hz=1500.0,
                freq_hz=_hb_freq,
            )
            info['highband_mic'] = HighbandMicPath(
                H_high, _hb_meta['freq_hz'], 1500.0, sample_rate)
        except Exception as _hb_exc:
            import traceback as _tb
            print(f"[highband] build failed: {_hb_exc}", flush=True)
            _tb.print_exc()

    return co, info
