"""optical_manifold.py — Data model for ray-path manifolds through optical elements.

Concept
-------
A *parametric optic surface* is a lens surface, mirror, aperture stop, or
any optical boundary described by a ``ParametricSurface`` instance.  A ray
traverses an optical element by crossing two or more such surfaces.  The
*surface manifold* records each surface interaction as a ``ManifoldVertex``,
ordered from entry to exit.

For BDPT the aperture stop is the natural connection plane:

  Forward half-path : emitter → scene → aperture
  Backward half-path: sensor pixel → aperture          (PIXEL_CONE pass)

A ``ManifoldHalf`` wraps the manifold and pins its sub-path's crossing point
on the aperture surface using (u, v) coordinates from a ``ParametricSurface``.
An ``ApertureGrid`` bins many ``ManifoldHalf`` objects by their (u, v) so that
forward and backward halves can be matched efficiently.

Amplitude dtype
---------------
``ManifoldVertex.amp_re`` / ``amp_im`` preserve the native float32 dtype of
the C-side ``EndpointRecord``.  All arithmetic inside this module keeps
float32.  The complex contribution in ``CorrelationCandidate.contribution``
is complex64 (float32 real + float32 imaginary).

Building halves from C-side records
------------------------------------
``build_halves_from_records(records, aperture, kind, n_bands)`` converts a
(N, 16) float32 or structured ``ENDPOINT_DTYPE`` array into a list of
``ManifoldHalf``.  It:

  1. Groups records by (subpath_id, vertex_index) to aggregate per-band amps.
  2. Sorts each subpath's vertices by vertex_index.
  3. Projects the *terminal* vertex's ray backward through the aperture
     surface to determine ``aperture_uv``.
  4. Returns one ``ManifoldHalf`` per unique subpath_id.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from parametric_surface import ParametricSurface


# ═══════════════════════════════════════════════════════════════════════════════
# Interaction type constants (bitmask)
# ═══════════════════════════════════════════════════════════════════════════════

INTERACTION_PROPAGATE = 0   # straight-line propagation / free-space travel
INTERACTION_REFRACT   = 1   # refraction at a dielectric interface
INTERACTION_REFLECT   = 2   # specular reflection
INTERACTION_SCATTER   = 4   # diffuse / glossy scatter
INTERACTION_ABSORB    = 8   # partial or full absorption event


# ═══════════════════════════════════════════════════════════════════════════════
# ManifoldVertex — one surface interaction on a ray path
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ManifoldVertex:
    """One recorded surface interaction along a ray's manifold path.

    Geometry fields use float64 (world-space precision).
    Amplitude fields preserve the native float32 from ``EndpointRecord``.

    Attributes
    ----------
    surface_id : registered TriGroup / surface integer ID (-1 = unknown).
    uv         : (u, v) on the hit surface's ``ParametricSurface``
                 parameterisation.  (0.0, 0.0) when the surface geometry is
                 not yet registered.
    pos        : world-space hit position, shape (3,), float64.
    dir_in     : unit incoming ray direction at this vertex, shape (3,), float64.
    dir_out    : unit outgoing (scattered/refracted) direction, shape (3,),
                 float64.  Equal to dir_in for propagation-only vertices.
    normal     : outward unit surface normal, shape (3,), float64.
    amp_re     : per-band real amplitude, shape (n_bands,), float32.
    amp_im     : per-band imaginary amplitude, shape (n_bands,), float32.
    pdf        : sampling PDF for the outgoing direction.
    cos_in     : |dot(dir_in, normal)| — foreshortening factor.
    cos_out    : |dot(dir_out, normal)| (0.0 when unknown).
    interaction: bitmask of INTERACTION_* constants.
    pathlen_m  : cumulative optical path length from the sub-path origin (m).
    """
    surface_id:  int
    uv:          tuple[float, float]
    pos:         np.ndarray   # float64, (3,)
    dir_in:      np.ndarray   # float64, (3,)
    dir_out:     np.ndarray   # float64, (3,)
    normal:      np.ndarray   # float64, (3,)
    amp_re:      np.ndarray   # float32, (n_bands,)
    amp_im:      np.ndarray   # float32, (n_bands,)
    pdf:         float
    cos_in:      float
    cos_out:     float        = 0.0
    interaction: int          = INTERACTION_PROPAGATE
    pathlen_m:   float        = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SurfaceManifold — ordered traversal through an optical element
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SurfaceManifold:
    """Ordered sequence of ManifoldVertex connecting entry surface to exit surface.

    For a single forward-path endpoint the list has exactly one vertex (the
    sensor hit).  For a PIXEL_CONE backward path with N bounces the list has N
    vertices sorted by vertex_index ascending.

    The ``terminal_amp`` property computes the *product* of all vertex
    amplitudes along the path, preserving float32 throughout.
    """
    subpath_id:       int
    entry_surface_id: int
    exit_surface_id:  int
    vertices:         list[ManifoldVertex] = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def n_bands(self) -> int:
        return int(self.vertices[0].amp_re.shape[0]) if self.vertices else 0

    @property
    def entry_uv(self) -> tuple[float, float]:
        return self.vertices[0].uv if self.vertices else (0.0, 0.0)

    @property
    def exit_uv(self) -> tuple[float, float]:
        return self.vertices[-1].uv if self.vertices else (0.0, 0.0)

    @property
    def terminal_amp(self) -> tuple[np.ndarray, np.ndarray]:
        """Product of all vertex amplitudes — (amp_re, amp_im) float32.

        Starts from (1+0j) and multiplies each vertex's complex amplitude in
        order.  For a single-vertex path this equals that vertex's amplitude.
        """
        if not self.vertices:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        n = self.n_bands
        re = np.ones(n, dtype=np.float32)
        im = np.zeros(n, dtype=np.float32)
        for v in self.vertices:
            # (re + j*im) * (v.re + j*v.im)  — all float32
            new_re = re * v.amp_re - im * v.amp_im
            new_im = re * v.amp_im + im * v.amp_re
            re, im = new_re, new_im
        return re, im

    @property
    def terminal_vertex(self) -> ManifoldVertex | None:
        return self.vertices[-1] if self.vertices else None


# ═══════════════════════════════════════════════════════════════════════════════
# ManifoldHalf — one BDPT sub-path half, pinned to aperture (u, v)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ManifoldHalf:
    """One half of a BDPT path organised by its aperture-plane crossing.

    A forward-light ManifoldHalf carries a forward ray from the emitter that
    arrived at (or would arrive at) the aperture at ``aperture_uv``.  A
    backward-sensor ManifoldHalf (APERTURE_PUPIL / PIXEL_CONE) carries the
    backward ray from a sensor pixel that passed through the aperture at the
    same (u, v).

    Connecting a forward half and a backward half with nearby ``aperture_uv``
    is the core BDPT connection step.

    Attributes
    ----------
    kind                : RayStreamKind constant — FORWARD_LIGHT or APERTURE_PUPIL.
    aperture_surface_id : integer label for the aperture ParametricSurface.
    aperture_uv         : (u, v) on the aperture surface in normalised coords.
    manifold            : the SurfaceManifold leading to / from the aperture.
    pixel_id            : flat pixel index for APERTURE_PUPIL halves
                          (= subpath_id when using PIXEL_CONE encoding); -1 otherwise.
    source_records      : raw structured EndpointRecord rows for this subpath
                          (structured ENDPOINT_DTYPE array), or None.
    """
    kind:                int
    aperture_surface_id: int
    aperture_uv:         tuple[float, float]
    manifold:            SurfaceManifold
    pixel_id:            int           = -1
    source_records:      np.ndarray | None = None   # structured ENDPOINT_DTYPE


# ═══════════════════════════════════════════════════════════════════════════════
# ApertureGrid — spatial hash of ManifoldHalf by (u, v) bin
# ═══════════════════════════════════════════════════════════════════════════════

class ApertureGrid:
    """2-D spatial grid organising ManifoldHalf objects by aperture (u, v).

    (u, v) ∈ [-1, 1] × [-1, 1] are binned into (n_u × n_v) uniform cells.
    The grid supports O(1) insert and O(radius_bins²) neighbour query.

    Parameters
    ----------
    surface    : the ParametricSurface that defines the aperture.
    n_u, n_v   : grid resolution in u and v.
    """

    def __init__(
        self,
        surface: ParametricSurface,
        n_u: int = 16,
        n_v: int = 16,
    ) -> None:
        self.surface = surface
        self.n_u     = int(max(1, n_u))
        self.n_v     = int(max(1, n_v))
        self._bins: list[list[ManifoldHalf]] = [
            [] for _ in range(self.n_u * self.n_v)
        ]
        self._count = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _uv_to_bin(self, u: float, v: float) -> int:
        iu = int(np.clip((u + 1.0) * 0.5 * self.n_u, 0, self.n_u - 1))
        iv = int(np.clip((v + 1.0) * 0.5 * self.n_v, 0, self.n_v - 1))
        return iv * self.n_u + iu

    def _bin_to_uv_idx(self, idx: int) -> tuple[int, int]:
        return idx % self.n_u, idx // self.n_u

    # ── Public interface ──────────────────────────────────────────────────────

    def insert(self, half: ManifoldHalf) -> None:
        """Insert a ManifoldHalf into the bin corresponding to its aperture_uv."""
        u, v = half.aperture_uv
        self._bins[self._uv_to_bin(u, v)].append(half)
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def query_bin(self, u: float, v: float) -> list[ManifoldHalf]:
        """Return all ManifoldHalf in the exact bin containing (u, v)."""
        return self._bins[self._uv_to_bin(u, v)]

    def query_neighbors(
        self,
        u: float,
        v: float,
        radius_bins: int = 1,
    ) -> list[ManifoldHalf]:
        """Return all ManifoldHalf within *radius_bins* cells of (u, v).

        radius_bins=0 → exact bin only.
        radius_bins=1 → 3×3 neighbourhood (default).
        """
        iu = int(np.clip((u + 1.0) * 0.5 * self.n_u, 0, self.n_u - 1))
        iv = int(np.clip((v + 1.0) * 0.5 * self.n_v, 0, self.n_v - 1))
        result: list[ManifoldHalf] = []
        for dv in range(-radius_bins, radius_bins + 1):
            for du in range(-radius_bins, radius_bins + 1):
                niu = iu + du
                niv = iv + dv
                if 0 <= niu < self.n_u and 0 <= niv < self.n_v:
                    result.extend(self._bins[niv * self.n_u + niu])
        return result

    def iter_bins(self) -> Iterable[tuple[int, int, list[ManifoldHalf]]]:
        """Yield (iu, iv, halves) for every non-empty bin."""
        for idx, halves in enumerate(self._bins):
            if halves:
                iu, iv = self._bin_to_uv_idx(idx)
                yield iu, iv, halves

    def clear(self) -> None:
        for b in self._bins:
            b.clear()
        self._count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: project an EndpointRecord ray backward onto an aperture surface
# ═══════════════════════════════════════════════════════════════════════════════

def project_record_to_aperture(
    rec_row: np.ndarray,
    aperture: ParametricSurface,
) -> tuple[float, float] | None:
    """Find the aperture (u, v) for an EndpointRecord by tracing its ray backward.

    The ray carried by *rec_row* is traced in the direction **−dir** from
    *pos*, intersecting the aperture surface.  This works for both forward
    (sensor-hit) and backward (PIXEL_CONE scene-hit) records because in both
    cases the ray physically crossed the aperture on the way to its terminal
    vertex:

      Forward : emitter → aperture → sensor_hit   →  trace −dir from sensor_hit
      Backward: sensor → aperture → scene_hit     →  trace −dir from scene_hit

    Parameters
    ----------
    rec_row : one-element structured EndpointRecord (ENDPOINT_DTYPE) or a
              1-D float32 array of length ≥ 16.
    aperture : the ParametricSurface representing the aperture stop.

    Returns
    -------
    (u, v) on the aperture, or None if the backward ray misses the surface.
    """
    from bdpt_integrator import ENDPOINT_DTYPE

    r = rec_row
    if r.dtype == ENDPOINT_DTYPE:
        if r.ndim == 0:
            r = r.reshape(1)
        pos = np.array(
            [float(r['pos_x']), float(r['pos_y']), float(r['pos_z'])],
            dtype=np.float64,
        )
        d = np.array(
            [float(r['dir_x']), float(r['dir_y']), float(r['dir_z'])],
            dtype=np.float64,
        )
    else:
        flat = np.asarray(r, dtype=np.float64).ravel()
        pos = flat[4:7]
        d   = flat[8:11]

    d_len = math.sqrt(float(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]))
    if d_len < 1e-15:
        return None
    d_unit = d / d_len
    return aperture.project_ray_to_uv(pos, -d_unit)


# ═══════════════════════════════════════════════════════════════════════════════
# Build ManifoldHalf objects from a batch of EndpointRecord rows
# ═══════════════════════════════════════════════════════════════════════════════

def build_halves_from_records(
    records: np.ndarray,
    aperture: ParametricSurface,
    kind: int,
    n_bands: int,
    aperture_surface_id: int = 0,
    fallback_aperture_uv: tuple[float, float] = (0.0, 0.0),
) -> list[ManifoldHalf]:
    """Convert a batch of EndpointRecord rows into a list of ManifoldHalf.

    Steps
    -----
    1. Normalise *records* to structured ENDPOINT_DTYPE.
    2. Group rows by (subpath_id, vertex_index) and collect per-band amplitudes.
    3. Sort each subpath's vertices by vertex_index ascending.
    4. Project the terminal vertex's ray backward onto *aperture* to get
       ``aperture_uv``.
    5. Return one ``ManifoldHalf`` per unique subpath_id.

    Parameters
    ----------
    records              : (N, 16) float32 or structured ENDPOINT_DTYPE.
    aperture             : the aperture ParametricSurface.
    kind                 : RayStreamKind constant.
    n_bands              : number of spectral bands.
    aperture_surface_id  : integer label stored in ManifoldHalf.
    fallback_aperture_uv : (u, v) used when the backward ray misses the aperture.

    Returns
    -------
    list of ManifoldHalf, one per unique subpath_id in *records*.
    """
    from bdpt_integrator import ENDPOINT_DTYPE, RayStreamKind, endpoints_view

    if records is None or (hasattr(records, 'shape') and records.shape[0] == 0):
        return []

    recs = records if records.dtype == ENDPOINT_DTYPE else endpoints_view(records)

    # ── Step 1: collect per-band amps for each (subpath_id, vertex_index) ────
    # key: (subpath_id, vertex_index) → {band_id: record_row}
    vertex_bands: dict[tuple[int, int], dict[int, np.ndarray]] = defaultdict(dict)

    for i in range(len(recs)):
        r   = recs[i]
        sid = int(r['subpath_id'])
        vid = int(r['vertex_index'])
        bid = int(r['band_id'])
        vertex_bands[(sid, vid)][bid] = r

    # ── Step 2–3: build ManifoldVertex list per subpath ──────────────────────
    # subpath_id → [(vertex_index, ManifoldVertex)]
    subpath_verts: dict[int, list[tuple[int, ManifoldVertex]]] = defaultdict(list)

    for (sid, vid), band_map in vertex_bands.items():
        # Reference record (band 0 if available, else first)
        ref = band_map.get(0, next(iter(band_map.values())))

        # Aggregate per-band amplitudes; preserve float32
        amp_re = np.zeros(n_bands, dtype=np.float32)
        amp_im = np.zeros(n_bands, dtype=np.float32)
        for b, r in band_map.items():
            if 0 <= b < n_bands:
                amp_re[b] = float(r['amp_re'])
                amp_im[b] = float(r['amp_im'])

        pos = np.array(
            [float(ref['pos_x']), float(ref['pos_y']), float(ref['pos_z'])],
            dtype=np.float64,
        )
        d = np.array(
            [float(ref['dir_x']), float(ref['dir_y']), float(ref['dir_z'])],
            dtype=np.float64,
        )
        d_len = math.sqrt(float(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]))
        dir_unit = d / d_len if d_len > 1e-15 else d

        cos_in = float(ref['cos_theta'])
        mv = ManifoldVertex(
            surface_id=int(ref['group_id']),
            uv=(0.0, 0.0),          # surface-local UV: unavailable without surface registry
            pos=pos,
            dir_in=dir_unit,
            dir_out=dir_unit,       # forward propagation; scatter dir unknown from endpoint
            normal=-dir_unit,       # approximate: normal ≈ −dir_in (sensor/scene face)
            amp_re=amp_re,
            amp_im=amp_im,
            pdf=float(ref['pdf']),
            cos_in=cos_in,
            cos_out=0.0,
            interaction=INTERACTION_PROPAGATE,
            pathlen_m=float(ref['pathlen_m']),
        )
        subpath_verts[sid].append((vid, mv))

    # ── Step 4–5: build ManifoldHalf per subpath ──────────────────────────────
    halves: list[ManifoldHalf] = []

    for sid, vert_list in subpath_verts.items():
        vert_list.sort(key=lambda t: t[0])
        vertices = [v for _, v in vert_list]

        manifold = SurfaceManifold(
            subpath_id=sid,
            entry_surface_id=vertices[0].surface_id if vertices else -1,
            exit_surface_id=vertices[-1].surface_id if vertices else -1,
            vertices=vertices,
        )

        # Project terminal vertex ray backward onto aperture
        terminal = vertices[-1]
        ap_uv = aperture.project_ray_to_uv(terminal.pos, -terminal.dir_in)
        if ap_uv is None:
            ap_uv = fallback_aperture_uv

        # For APERTURE_PUPIL, subpath_id IS the flat pixel index (PIXEL_CONE encoding)
        pixel_id = sid if kind == RayStreamKind.APERTURE_PUPIL else -1

        # Collect source records for this subpath (all bands, all vertices)
        # so CorrelationCandidate can carry the original rows.
        src_mask = recs['subpath_id'] == np.uint32(sid)
        src_rows = recs[src_mask].copy() if np.any(src_mask) else None

        halves.append(ManifoldHalf(
            kind=kind,
            aperture_surface_id=aperture_surface_id,
            aperture_uv=ap_uv,
            manifold=manifold,
            pixel_id=pixel_id,
            source_records=src_rows,
        ))

    return halves
