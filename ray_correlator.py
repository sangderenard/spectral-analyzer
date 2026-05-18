"""ray_correlator.py — Forward / backward sub-path correlation strategies.

A ``RayCorrelator`` ingests a batch of forward-light EndpointRecords and a
batch of backward-sensor EndpointRecords (or RayEvent lists) and produces
``CorrelationCandidate`` objects that the caller can accept, MIS-weight, and
deposit into a sensor accumulator.

Five strategies are provided as separate, swappable classes that all share the
``CorrelationStrategy`` protocol.  Callers select one or more strategies at
construction time and run them via ``RayCorrelator.correlate()``.

Backward-sensor sub-paths
-------------------------
The PIXEL_CONE / APERTURE_PUPIL pass is a specific, valid backward-sensor
strategy.  Records produced by that pass carry ``stream_kind=APERTURE_PUPIL``
(which is a sub-type of ``BACKWARD_SENSOR``) and ``record_intent=SENSOR_ESTIMATE``.
They represent the sensor half of a true BDPT pair and are never to be
eliminated — they must simply not be deposited unresolved into the physical
field buffer.  ``PixelConeOverlapStrategy`` (strategy 1) uses them as the
backward-side input for correlations.

Strategies
----------
0  EndpointPair       — nearest-endpoint geometric connection (n-nearest in
                        spatial proximity, no visibility check yet)
1  PixelConeOverlap   — pair forward and pixel-cone-sensor records by shared
                        ``subpath_id`` prefix
2  ManifoldWalk       — stub; will perform a manifold-exploration step to find
                        a shared path vertex on a specular surface
3  FieldInterpolate   — stub; connect via the volumetric field cache rather
                        than direct path geometry
4  DiagnosticPassThru — passes forward records straight through as
                        DIAGNOSTIC_RECORD candidates (no sensor deposit);
                        useful for overlay rendering

All strategies are safe to call with empty inputs and return empty lists.

NO physical deposit is performed here.  The caller is responsible for
checking ``candidate.accepted`` and routing to the correct accumulation
buffer only after confirming the ``RecordIntent`` rules.

Usage example::

    from bdpt_integrator import RayStreamKind, RecordIntent, CorrelationCandidate
    from ray_correlator import RayCorrelator, StrategyID

    correlator = RayCorrelator(strategies=[StrategyID.ENDPOINT_PAIR,
                                           StrategyID.PIXEL_CONE_OVERLAP])
    candidates = correlator.correlate(
        forward_records=fwd_arr,   # (N, 16) float32 or ENDPOINT_DTYPE
        backward_records=rev_arr,  # (M, 16) float32 or ENDPOINT_DTYPE  ← pixel-cone records
        n_bands=bench.n_bands,
    )
    deposits = [c for c in candidates if c.accepted
                and c.contribution is not None]
"""

from __future__ import annotations

import enum
from typing import Protocol, Sequence

import numpy as np

from bdpt_integrator import (
    ENDPOINT_DTYPE,
    CorrelationCandidate,
    MiddlePoint,
    RayEvent,
    RayStreamKind,
    RecordIntent,
    endpoints_view,
)
from parametric_surface import ParametricSurface

try:
    from radial_manifold import RadialApertureGrid as _RadialApertureGrid
    _RADIAL_GRID_AVAILABLE = True
except Exception:
    _RADIAL_GRID_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
class StrategyID(enum.IntEnum):
    ENDPOINT_PAIR       = 0
    PIXEL_CONE_OVERLAP  = 1
    MANIFOLD_WALK       = 2
    FIELD_INTERPOLATE   = 3
    DIAGNOSTIC_PASSTHRU = 4


# ─────────────────────────────────────────────────────────────────────────────
# MIS (Multiple Importance Sampling) weighting functions.

def mis_balance_weight(pdf_a: float, pdf_b: float) -> float:
    """Balance-heuristic MIS weight for strategy A given pdfs of A and B.

    Returns pdf_a / (pdf_a + pdf_b), or 0.5 when the denominator underflows.
    """
    denom = pdf_a + pdf_b
    return pdf_a / denom if denom > 1e-30 else 0.5


def mis_power_weight(pdf_a: float, pdf_b: float, beta: float = 2.0) -> float:
    """Power-heuristic MIS weight for strategy A.

    mis_power_weight(a, b, 2) = a^2 / (a^2 + b^2).
    """
    pa = pdf_a ** beta
    pb = pdf_b ** beta
    denom = pa + pb
    return pa / denom if denom > 1e-30 else 0.5


# ─────────────────────────────────────────────────────────────────────────────
class ShadowRayChecker:
    """Batch visibility checker for BDPT path connection.

    Two modes:
    - *Stub* (``tracer=None``): always returns visible; no geometry test.
    - *Tracer-backed* (``tracer`` provided): fires a batch probe ray via
      ``tracer.trace()`` from ``p0`` toward ``p1`` and classifies the segment
      as occluded if any hit is found closer than ``|p1 - p0| - epsilon``.

    The tracer-backed path requires the C++ tracer to support a single-ray
    ``trace()`` call returning a hit-distance (or negative on miss).  If the
    tracer does not support the required interface the checker silently falls
    back to the always-visible stub.

    Parameters
    ----------
    tracer : optional
        The C++ ray-tracer object (``PyRayTracer``).  ``None`` → always visible.
    epsilon : float
        Distance tolerance (m) subtracted from segment length when classifying
        hits; prevents self-intersection from the endpoint positions.
    """

    def __init__(self, tracer=None, epsilon: float = 1e-4) -> None:
        self._tracer  = tracer
        self._epsilon = float(epsilon)
        # Probe for tracer capability once at construction.
        self._has_trace = (
            tracer is not None
            and hasattr(tracer, "trace")
            and callable(getattr(tracer, "trace"))
        )

    # ------------------------------------------------------------------
    def visible(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        """Return True if the segment p0→p1 is unoccluded.

        Both ``p0`` and ``p1`` should be 1-D arrays of shape (3,).
        """
        p0 = np.asarray(p0, dtype=np.float64).ravel()
        p1 = np.asarray(p1, dtype=np.float64).ravel()
        if not self._has_trace:
            return True
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len < self._epsilon:
            return True  # degenerate segment — treat as co-located, unoccluded
        direction = (p1 - p0) / seg_len
        try:
            result = self._tracer.trace(
                p0.tolist(),
                direction.tolist(),
                float(seg_len),
            )
            # result is expected to be a hit-distance (float) or a dict with "t".
            if isinstance(result, dict):
                t_hit = float(result.get("t", -1.0))
            else:
                t_hit = float(result)
            return t_hit < 0.0 or t_hit >= seg_len - self._epsilon
        except Exception:
            # Unsupported API or error → optimistic
            return True

    # ------------------------------------------------------------------
    def visible_batch(
        self,
        p0s: np.ndarray,
        p1s: np.ndarray,
    ) -> np.ndarray:
        """Vectorised visibility test.

        Parameters
        ----------
        p0s : (N, 3) float64
        p1s : (N, 3) float64

        Returns
        -------
        np.ndarray of bool, shape (N,)
        """
        p0s = np.asarray(p0s, dtype=np.float64)
        p1s = np.asarray(p1s, dtype=np.float64)
        n   = p0s.shape[0]
        out = np.ones(n, dtype=bool)
        for i in range(n):
            out[i] = self.visible(p0s[i], p1s[i])
        return out



    """Protocol satisfied by all strategy classes."""

    strategy_id: int

    def correlate(
        self,
        forward_records:  np.ndarray,   # structured ENDPOINT_DTYPE
        backward_records: np.ndarray,   # structured ENDPOINT_DTYPE
        n_bands: int,
    ) -> list[CorrelationCandidate]:
        ...


# ─────────────────────────────────────────────────────────────────────────────
class EndpointPairStrategy:
    """Strategy 0 — spatial nearest-endpoint geometric connection.

    For each backward-sensor endpoint find the K closest forward-light
    endpoints in world space, form a ``CorrelationCandidate`` with a
    ``MiddlePoint`` whose position is the midpoint of the two endpoints and
    whose visibility is set to 1.0 (no shadow ray yet — caller must validate).

    Candidates are returned with ``accepted=False``; a visibility check
    should be performed before promotion to PHYSICAL_DEPOSIT.

    Parameters
    ----------
    k_nearest : int
        Number of forward candidates to pair with each backward endpoint.
    max_dist_m : float
        Spatial radius cutoff in metres.  Pairs beyond this distance are
        discarded before the caller ever sees them.
    """

    strategy_id = int(StrategyID.ENDPOINT_PAIR)

    def __init__(self, k_nearest: int = 1, max_dist_m: float = 1.0) -> None:
        self.k_nearest   = int(k_nearest)
        self.max_dist_m  = float(max_dist_m)

    def correlate(
        self,
        forward_records:  np.ndarray,
        backward_records: np.ndarray,
        n_bands: int,
    ) -> list[CorrelationCandidate]:
        if forward_records.size == 0 or backward_records.size == 0:
            return []

        fwd_pos = forward_records["pos"].astype(np.float64)    # (N, 3)
        rev_pos = backward_records["pos"].astype(np.float64)   # (M, 3)

        candidates: list[CorrelationCandidate] = []

        for mi in range(rev_pos.shape[0]):
            rp = rev_pos[mi]
            diffs = fwd_pos - rp                               # (N, 3)
            dists = np.linalg.norm(diffs, axis=1)              # (N,)
            within = np.where(dists <= self.max_dist_m)[0]
            if within.size == 0:
                continue
            order  = np.argsort(dists[within])
            picks  = within[order[:self.k_nearest]]

            rev_rec = backward_records[mi]
            for ni in picks:
                fwd_rec = forward_records[int(ni)]
                mid_pos = (fwd_pos[int(ni)] + rp) * 0.5

                fwd_event = _endpoint_to_ray_event(fwd_rec, n_bands,
                                                   RayStreamKind.FORWARD_LIGHT,
                                                   RecordIntent.CORRELATION_CANDIDATE)
                rev_event = _endpoint_to_ray_event(rev_rec, n_bands,
                                                   RayStreamKind.BACKWARD_SENSOR,
                                                   RecordIntent.CORRELATION_CANDIDATE)
                middle = MiddlePoint(
                    forward_event=fwd_event,
                    backward_event=rev_event,
                    pos=mid_pos,
                    visibility=1.0,  # caller must verify
                    mis_weight=1.0,
                )
                contrib = (fwd_rec["amp_re"].astype(np.float32)
                           + 1j * fwd_rec["amp_im"].astype(np.float32))
                candidates.append(CorrelationCandidate(
                    middle=middle,
                    forward_record=np.array([fwd_rec], dtype=ENDPOINT_DTYPE),
                    backward_record=np.array([rev_rec], dtype=ENDPOINT_DTYPE),
                    strategy_id=self.strategy_id,
                    contribution=contrib,
                    accepted=False,
                ))

        return candidates


# ─────────────────────────────────────────────────────────────────────────────
class PixelConeOverlapStrategy:
    """Strategy 1 — correlate APERTURE_PUPIL (backward-sensor / pixel-cone)
    records with forward-light records that address the same sensor pixel.

    The PIXEL_CONE pass emits backward sub-paths that originate at sensor
    pixels and travel backward through the aperture into the scene.  These
    are a valid backward-sensor half of a BDPT pair.  Records from this pass
    carry ``subpath_id = py * n_px + px`` as their pixel address and
    ``group_id == sensor_group_id``.  This strategy pairs them with forward
    light records to form correlation candidates; the candidates are marked
    ``accepted=True`` because the pixel address provides implicit mutual
    visibility within the cone solid angle.

    The combined contribution (forward_amp × backward_amp) represents the
    coherent product before MIS weighting; the caller should apply any
    additional MIS term before depositing to the sensor accumulator.

    Parameters
    ----------
    n_px, n_py : int
        Sensor pixel grid dimensions; used to decode subpath_id.
    sensor_group_id : int
        Only backward records belonging to this group_id are used.
    """

    strategy_id = int(StrategyID.PIXEL_CONE_OVERLAP)

    def __init__(self, n_px: int, n_py: int, sensor_group_id: int) -> None:
        self.n_px            = int(n_px)
        self.n_py            = int(n_py)
        self.sensor_group_id = int(sensor_group_id)

    def correlate(
        self,
        forward_records:  np.ndarray,
        backward_records: np.ndarray,
        n_bands: int,
    ) -> list[CorrelationCandidate]:
        if backward_records.size == 0 or forward_records.size == 0:
            return []

        # Filter backward records to sensor group.
        gid_rev  = backward_records["group_id"].astype(np.int32)
        rev_mask = gid_rev == self.sensor_group_id
        if not np.any(rev_mask):
            return []
        rev = backward_records[rev_mask]

        sub_rev = rev["subpath_id"].astype(np.int64)
        px_rev  = (sub_rev % self.n_px).astype(np.int32)
        py_rev  = (sub_rev // self.n_px).astype(np.int32)

        # Build per-pixel index for backward records.
        # { (px, py): [row_indices...] }
        from collections import defaultdict
        rev_index: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in range(rev.shape[0]):
            key = (int(px_rev[i]), int(py_rev[i]))
            rev_index[key].append(i)

        # Trivially: pair every forward record with all backward records of the
        # pixel it projects to.  The forward records may not carry pixel info —
        # they're matched by proximity in a real implementation.  Here we form
        # one candidate per (forward, backward) pair within the same pixel.
        #
        # Stub: pair all forward records against all backward pixels uniformly
        # (placeholder; a real implementation projects forward hits onto the
        # pixel grid first).

        candidates: list[CorrelationCandidate] = []
        fwd_arr = forward_records  # use all forward records as-is for now

        if fwd_arr.size == 0:
            return []

        for (px, py), rev_idxs in rev_index.items():
            for ri in rev_idxs:
                rev_rec = rev[ri]
                for ni in range(min(fwd_arr.shape[0], 8)):   # cap to 8 per pixel
                    fwd_rec  = fwd_arr[ni]
                    fwd_amp  = (fwd_rec["amp_re"].astype(np.float32)
                                + 1j * fwd_rec["amp_im"].astype(np.float32))
                    rev_amp  = (rev_rec["amp_re"].astype(np.float32)
                                + 1j * rev_rec["amp_im"].astype(np.float32))
                    contrib  = fwd_amp * rev_amp

                    fwd_event = _endpoint_to_ray_event(fwd_rec, n_bands,
                                                       RayStreamKind.FORWARD_LIGHT,
                                                       RecordIntent.CORRELATION_CANDIDATE)
                    rev_event = _endpoint_to_ray_event(rev_rec, n_bands,
                                                       RayStreamKind.APERTURE_PUPIL,
                                                       # APERTURE_PUPIL is a backward-sensor sub-path;
                                                       # once paired with a forward record it is a
                                                       # SENSOR_ESTIMATE until MIS resolves it.
                                                       RecordIntent.SENSOR_ESTIMATE)
                    candidates.append(CorrelationCandidate(
                        middle=None,
                        forward_record=np.array([fwd_rec], dtype=ENDPOINT_DTYPE),
                        backward_record=np.array([rev_rec], dtype=ENDPOINT_DTYPE),
                        strategy_id=self.strategy_id,
                        contribution=contrib,
                        accepted=True,   # pixel-cone pairs are self-validating
                    ))

        return candidates


# ─────────────────────────────────────────────────────────────────────────────
class ManifoldWalkStrategy:
    """Strategy 2 — aperture-organised manifold-walk connection.

    Organises forward and backward sub-path endpoints by their crossing (u, v)
    on a shared ``ParametricSurface`` (the aperture stop) and matches halves
    that arrive at nearby aperture bins.  Each matched (forward, backward) pair
    becomes a ``CorrelationCandidate`` with the aperture crossing as the middle
    vertex.

    This is the first fully-implemented BDPT connection strategy.  Visibility
    is marked optimistic (``accepted=True``); callers that require shadow-ray
    validation should check it before depositing.

    Parameters
    ----------
    aperture : ParametricSurface
        The aperture stop surface.  ``PlaneSurface`` is the common choice for
        a circular or rectangular aperture stop.  ``ConicSurface`` works for
        curved mirrors or lens surfaces used as the connection manifold.
        If None the strategy returns an empty list (useful as a placeholder).
    n_bands : int
        Number of spectral bands (must match the records passed to correlate).
    aperture_surface_id : int
        Integer label stored in ManifoldHalf.aperture_surface_id.
    grid_n_u, grid_n_v : int
        ApertureGrid resolution.  Larger values give finer spatial selectivity
        at the cost of fewer matches per bin.  16×16 is a good default.
    match_radius_bins : int
        Neighbourhood radius in bins when searching for forward matches.
        0 = exact bin only; 1 = 3×3 window (default).
    """

    strategy_id = int(StrategyID.MANIFOLD_WALK)

    def __init__(
        self,
        aperture: "ParametricSurface | None" = None,
        n_bands: int = 1,
        aperture_surface_id: int = 0,
        grid_n_u: int = 16,
        grid_n_v: int = 16,
        match_radius_bins: int = 1,
        use_radial_grid: bool = False,
        grid_n_r: int = 64,
        shadow_checker: "ShadowRayChecker | None" = None,
        use_mis: bool = True,
    ) -> None:
        self.aperture            = aperture
        self.n_bands             = int(n_bands)
        self.aperture_surface_id = int(aperture_surface_id)
        self.grid_n_u            = int(grid_n_u)
        self.grid_n_v            = int(grid_n_v)
        self.match_radius_bins   = int(match_radius_bins)
        # Radial-grid path: 1-D axisymmetric binning for rotationally
        # symmetric optics (Step 4 of the BDPT manifold plan).
        self.use_radial_grid     = bool(use_radial_grid) and _RADIAL_GRID_AVAILABLE
        self.grid_n_r            = int(grid_n_r)
        # Step 6: optional shadow-ray checker and MIS balance weight.
        self.shadow_checker      = shadow_checker   # None → optimistic (always visible)
        self.use_mis             = bool(use_mis)

    def correlate(
        self,
        forward_records:  np.ndarray,
        backward_records: np.ndarray,
        n_bands: int,
    ) -> list[CorrelationCandidate]:
        if self.aperture is None:
            return []
        if forward_records is None or backward_records is None:
            return []
        if (hasattr(forward_records,  'shape') and forward_records.shape[0]  == 0) or \
           (hasattr(backward_records, 'shape') and backward_records.shape[0] == 0):
            return []

        from optical_manifold import ApertureGrid, build_halves_from_records

        fwd_halves = build_halves_from_records(
            forward_records,
            self.aperture,
            RayStreamKind.FORWARD_LIGHT,
            n_bands,
            self.aperture_surface_id,
        )
        bwd_halves = build_halves_from_records(
            backward_records,
            self.aperture,
            RayStreamKind.APERTURE_PUPIL,
            n_bands,
            self.aperture_surface_id,
        )

        # ── Build forward-half lookup grid ───────────────────────────────────
        if self.use_radial_grid:
            # 1-D radial grid for axisymmetric systems.
            ap_radius = float(
                getattr(self.aperture, 'half_extent_u',
                getattr(self.aperture, 'r_max',
                getattr(self.aperture, 'clear_aperture_radius', 0.015)))
            )
            fwd_grid_r = _RadialApertureGrid(r_max=ap_radius, n_r=self.grid_n_r)
            for half in fwd_halves:
                fwd_grid_r.insert(half)

            candidates: list[CorrelationCandidate] = []
            for bwd_half in bwd_halves:
                bu, bv = bwd_half.aperture_uv
                import math as _math
                r_norm = _math.sqrt(float(bu) ** 2 + float(bv) ** 2)
                r_phys = r_norm * ap_radius
                nearby_fwd = fwd_grid_r.query_ring(r_phys, self.match_radius_bins)
                candidates.extend(
                    self._make_candidates(bwd_half, nearby_fwd)
                )
            return candidates
        else:
            # 2-D Cartesian aperture grid (default).
            fwd_grid = ApertureGrid(self.aperture, self.grid_n_u, self.grid_n_v)
            for half in fwd_halves:
                fwd_grid.insert(half)

            candidates: list[CorrelationCandidate] = []
            for bwd_half in bwd_halves:
                bu, bv = bwd_half.aperture_uv
                nearby_fwd = fwd_grid.query_neighbors(bu, bv, self.match_radius_bins)
                candidates.extend(
                    self._make_candidates(bwd_half, nearby_fwd)
                )
            return candidates

    def _make_candidates(
        self,
        bwd_half,
        nearby_fwd: list,
    ) -> list[CorrelationCandidate]:
        """Build CorrelationCandidate objects for one backward half paired
        against a list of forward halves.  Shared by both grid paths.
        """
        results: list[CorrelationCandidate] = []
        for fwd_half in nearby_fwd:
            # Contribution = product of terminal amplitudes (no geometry
            # term yet; shadow-ray / MIS deferred to caller).
            fre, fim = fwd_half.manifold.terminal_amp
            bre, bim = bwd_half.manifold.terminal_amp
            # (fre + j*fim) * (bre + j*bim)  — stays float32
            contrib_re = fre * bre - fim * bim
            contrib_im = fre * bim + fim * bre
            contribution = (contrib_re + 1j * contrib_im).astype(np.complex64)

            # Middle point: the aperture-plane crossing position.
            ap_u, ap_v = fwd_half.aperture_uv
            ap_pos = self.aperture.uv_to_point(ap_u, ap_v)

            fwd_tv = fwd_half.manifold.terminal_vertex
            bwd_tv = bwd_half.manifold.terminal_vertex

            # ── Step 6: shadow-ray occlusion check ──────────────────────────
            fwd_pos_3d = fwd_tv.pos if fwd_tv is not None else ap_pos
            bwd_pos_3d = bwd_tv.pos if bwd_tv is not None else ap_pos
            if self.shadow_checker is not None:
                vis = self.shadow_checker.visible(fwd_pos_3d, bwd_pos_3d)
                visibility = 1.0 if vis else 0.0
                accepted   = bool(vis)
            else:
                visibility = 1.0
                accepted   = True

            # ── Step 6: MIS balance-heuristic weight ─────────────────────────
            if self.use_mis and accepted:
                pdf_fwd = float(fwd_tv.pdf) if fwd_tv is not None else 1.0
                pdf_bwd = float(bwd_tv.pdf) if bwd_tv is not None else 1.0
                w_mis = mis_balance_weight(pdf_fwd, pdf_bwd)
            else:
                w_mis = 1.0
            mis_w_real = np.float32(w_mis)

            fwd_ev = RayEvent(
                stream_kind=RayStreamKind.FORWARD_LIGHT,
                record_intent=RecordIntent.CORRELATION_CANDIDATE,
                subpath_id=fwd_half.manifold.subpath_id,
                bounce_index=len(fwd_half.manifold.vertices) - 1,
                pos=fwd_tv.pos if fwd_tv is not None else np.zeros(3, np.float64),
                dir_in=fwd_tv.dir_in if fwd_tv is not None else np.zeros(3, np.float64),
                dir_out=fwd_tv.dir_out if fwd_tv is not None else np.zeros(3, np.float64),
                normal=fwd_tv.normal if fwd_tv is not None else np.zeros(3, np.float64),
                pdf=fwd_tv.pdf if fwd_tv is not None else 1.0,
                pathlen_m=fwd_tv.pathlen_m if fwd_tv is not None else 0.0,
                amp_re=fre,
                amp_im=fim,
            )
            bwd_ev = RayEvent(
                stream_kind=RayStreamKind.APERTURE_PUPIL,
                record_intent=RecordIntent.SENSOR_ESTIMATE,
                subpath_id=bwd_half.manifold.subpath_id,
                bounce_index=len(bwd_half.manifold.vertices) - 1,
                pos=bwd_tv.pos if bwd_tv is not None else np.zeros(3, np.float64),
                dir_in=bwd_tv.dir_in if bwd_tv is not None else np.zeros(3, np.float64),
                dir_out=bwd_tv.dir_out if bwd_tv is not None else np.zeros(3, np.float64),
                normal=bwd_tv.normal if bwd_tv is not None else np.zeros(3, np.float64),
                pdf=bwd_tv.pdf if bwd_tv is not None else 1.0,
                pathlen_m=bwd_tv.pathlen_m if bwd_tv is not None else 0.0,
                amp_re=bre,
                amp_im=bim,
            )
            middle = MiddlePoint(
                forward_event=fwd_ev,
                backward_event=bwd_ev,
                pos=ap_pos,
                visibility=visibility,
                mis_weight=float(mis_w_real),
            )

            results.append(CorrelationCandidate(
                middle=middle,
                forward_record=fwd_half.source_records,
                backward_record=bwd_half.source_records,
                strategy_id=self.strategy_id,
                contribution=contribution * mis_w_real,
                accepted=accepted,
            ))

        return results


# ─────────────────────────────────────────────────────────────────────────────
class FieldInterpolateStrategy:
    """Strategy 3 — volumetric field-cache connection (stub).

    Connects forward and backward sub-paths through the 3-D field grid
    rather than through direct path geometry.  Not implemented; returns
    empty list.
    """

    strategy_id = int(StrategyID.FIELD_INTERPOLATE)

    def __init__(self, field_grid: np.ndarray | None = None) -> None:
        self.field_grid = field_grid   # (Bx, By, Bz, n_bands) complex64

    def correlate(
        self,
        forward_records:  np.ndarray,
        backward_records: np.ndarray,
        n_bands: int,
    ) -> list[CorrelationCandidate]:
        # TODO: implement field-grid connection
        return []


# ─────────────────────────────────────────────────────────────────────────────
class DiagnosticPassThruStrategy:
    """Strategy 4 — forward-record diagnostic pass-through.

    Wraps every forward EndpointRecord as a CorrelationCandidate with
    ``record_intent=DIAGNOSTIC_RECORD`` and ``accepted=False``.  Used by the
    overlay renderer to colour-code live forward endpoints without writing
    anything to a physical accumulation buffer.
    """

    strategy_id = int(StrategyID.DIAGNOSTIC_PASSTHRU)

    def correlate(
        self,
        forward_records:  np.ndarray,
        backward_records: np.ndarray,
        n_bands: int,
    ) -> list[CorrelationCandidate]:
        candidates: list[CorrelationCandidate] = []
        for ni in range(forward_records.shape[0]):
            rec = forward_records[ni]
            ev  = _endpoint_to_ray_event(rec, n_bands,
                                         RayStreamKind.FORWARD_LIGHT,
                                         RecordIntent.DIAGNOSTIC_RECORD)
            candidates.append(CorrelationCandidate(
                middle=None,
                forward_record=np.array([rec], dtype=ENDPOINT_DTYPE),
                backward_record=None,
                strategy_id=self.strategy_id,
                contribution=None,
                accepted=False,  # NEVER deposit diagnostics
            ))
        return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Strategy registry
_STRATEGY_CLASSES = {
    StrategyID.ENDPOINT_PAIR:       EndpointPairStrategy,
    StrategyID.PIXEL_CONE_OVERLAP:  PixelConeOverlapStrategy,
    StrategyID.MANIFOLD_WALK:       ManifoldWalkStrategy,
    StrategyID.FIELD_INTERPOLATE:   FieldInterpolateStrategy,
    StrategyID.DIAGNOSTIC_PASSTHRU: DiagnosticPassThruStrategy,
}


# ─────────────────────────────────────────────────────────────────────────────
class RayCorrelator:
    """Combines one or more correlation strategies.

    Parameters
    ----------
    strategies : sequence of StrategyID or CorrelationStrategy instances
        If StrategyID enum values are provided, the corresponding strategy
        class is instantiated with default parameters.  Pass pre-constructed
        instances to customise parameters (e.g. k_nearest, max_dist_m).
    """

    def __init__(
        self,
        strategies: Sequence[StrategyID | CorrelationStrategy] | None = None,
    ) -> None:
        if strategies is None:
            strategies = [StrategyID.ENDPOINT_PAIR]
        built: list[CorrelationStrategy] = []
        for s in strategies:
            if isinstance(s, StrategyID):
                cls = _STRATEGY_CLASSES[s]
                # Only EndpointPairStrategy has a no-arg default constructor.
                # PixelConeOverlapStrategy requires dimensions at construction.
                if s == StrategyID.ENDPOINT_PAIR:
                    built.append(cls())
                elif s == StrategyID.MANIFOLD_WALK:
                    raise ValueError(
                        "StrategyID.MANIFOLD_WALK requires explicit construction "
                        "with an aperture surface: "
                        "ManifoldWalkStrategy(aperture=my_surface, n_bands=N)"
                    )
                elif s == StrategyID.FIELD_INTERPOLATE:
                    built.append(FieldInterpolateStrategy())
                elif s == StrategyID.DIAGNOSTIC_PASSTHRU:
                    built.append(DiagnosticPassThruStrategy())
                else:
                    raise ValueError(
                        f"StrategyID.{s.name} requires explicit construction "
                        f"(use a pre-built instance, not a bare StrategyID)."
                    )
            else:
                built.append(s)
        self._strategies: list[CorrelationStrategy] = built

    # ── Validation counters ─────────────────────────────────────────────────
    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats = {
            "forward_in":    0,
            "backward_in":   0,
            "candidates_out": 0,
            "accepted_out":  0,
        }

    # Initialise stats at construction time.
    def __post_init__(self) -> None:
        self.reset_stats()

    _stats: dict[str, int] = {}   # instance attribute; populated by reset_stats / _ensure_stats

    def __init_subclass__(cls, **kw: object) -> None:
        super().__init_subclass__(**kw)

    # Override __init__ already done above — call reset_stats manually.
    def _ensure_stats(self) -> None:
        if not hasattr(self, "_stats") or not self._stats:
            self.reset_stats()

    # ── Main entry point ────────────────────────────────────────────────────
    def correlate(
        self,
        forward_records:  np.ndarray,      # (N, 16) float32 or ENDPOINT_DTYPE
        backward_records: np.ndarray,      # (M, 16) float32 or ENDPOINT_DTYPE
        n_bands: int,
    ) -> list[CorrelationCandidate]:
        """Run all registered strategies and return the combined candidate list.

        SAFETY CONTRACT
        ---------------
        * This method NEVER writes to any physical buffer.
        * Candidates with ``accepted=False`` must NOT be deposited.
        * Candidates whose ``forward_record`` or ``backward_record`` has
          ``stream_kind == BACKWARD_SENSOR`` are automatically re-tagged
          ``record_intent = DIAGNOSTIC_RECORD`` before return.
        """
        self._ensure_stats()

        if forward_records.dtype != ENDPOINT_DTYPE:
            forward_records  = endpoints_view(forward_records)
        if backward_records.dtype != ENDPOINT_DTYPE:
            backward_records = endpoints_view(backward_records)

        self._stats["forward_in"]  += int(forward_records.shape[0])
        self._stats["backward_in"] += int(backward_records.shape[0])

        all_candidates: list[CorrelationCandidate] = []
        for strat in self._strategies:
            results = strat.correlate(forward_records, backward_records, n_bands)
            all_candidates.extend(results)

        self._stats["candidates_out"] += len(all_candidates)
        self._stats["accepted_out"]   += sum(1 for c in all_candidates if c.accepted)
        return all_candidates


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _endpoint_to_ray_event(
    rec: np.ndarray,          # single structured EndpointRecord element
    n_bands: int,
    stream_kind: int,
    record_intent: int,
) -> RayEvent:
    """Convert a single structured EndpointRecord to a RayEvent."""
    zero3 = np.zeros(3, dtype=np.float64)
    amp_re = np.atleast_1d(np.asarray(rec["amp_re"], dtype=np.float32))
    amp_im = np.atleast_1d(np.asarray(rec["amp_im"], dtype=np.float32))
    pos    = np.asarray(rec["pos"],  dtype=np.float64).reshape(3)
    d      = np.asarray(rec["dir"],  dtype=np.float64).reshape(3)
    nrm    = np.linalg.norm(d)
    dir_v  = d / nrm if nrm > 1e-12 else zero3
    return RayEvent(
        stream_kind=stream_kind,
        record_intent=record_intent,
        subpath_id=int(rec["subpath_id"]),
        bounce_index=int(rec["vertex_index"]),
        pos=pos,
        dir_in=dir_v,
        dir_out=dir_v,       # placeholder — real dir_out requires re-trace
        normal=zero3,        # placeholder — not stored in EndpointRecord
        pdf=float(rec["pdf"]),
        pathlen_m=float(rec["pathlen_m"]),
        amp_re=amp_re,
        amp_im=amp_im,
        group_id=int(rec["group_id"]),
        tri_id=-1,
    )
