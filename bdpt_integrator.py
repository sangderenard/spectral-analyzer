"""bdpt_integrator.py — Python-side helpers for the bidirectional integrator.

Wraps the C ABI exported by ``_spectral_kernels.RayTracer.bidirectional`` and
provides:

  * ``TriangleGroup``        — descriptor for a registered tri group, mirroring
                               ``csrc/include/triangle_groups.h``.
  * ``IntegratorMetadata``   — sidecar carried alongside any rendered output;
                               records integrator kind, deposit mode, branch
                               factor, ray budgets, scene/material hashes and
                               seed sequences so a render can be replayed bit
                               for bit.
  * ``ENDPOINT_DTYPE``       — NumPy structured dtype matching
                               ``EndpointRecord`` (64 B per row).
  * ``endpoints_view(arr)``  — re-interpret a (N, 16) float32 array (which is
                               what ``RayTracer.bidirectional`` returns) as a
                               structured array for ergonomic access.
  * ``aggregate_to_image``   — reference aggregator that bins endpoint records
                               into a (n_bands, H, W) complex64 grid given a
                               sensor-plane basis.  Used by the smoke test;
                               the production display path can pick whatever
                               aggregation it likes — storage stays complex.

NO storage-side reduction.  Aggregation is opt-in and produces a separate
view; the EndpointRecord array passed in is never mutated.
"""

from __future__ import annotations

import dataclasses as _dc
import hashlib as _hashlib
import json as _json
from typing import Any

import numpy as np


# ── Role + sample-policy constants (must match triangle_groups.h) ──────────
TRI_GROUP_ROLE_EMISSIVE = 1 << 0
TRI_GROUP_ROLE_SENSOR   = 1 << 1
TRI_GROUP_ROLE_BLOCKER  = 1 << 2
TRI_GROUP_ROLE_VOLUME   = 1 << 3

TRI_GROUP_SAMPLE_UNIFORM    = 0
TRI_GROUP_SAMPLE_AREA       = 1
TRI_GROUP_SAMPLE_POWER      = 2
TRI_GROUP_SAMPLE_PIXEL_CONE = 3   # SENSOR-only: per-pixel cone scan via CameraSensor

TRI_PARAM_SURFACE_NONE      = 0
TRI_PARAM_SURFACE_POLY_BARY = 1

# Camera operating modes for CameraSensor.camera_mode.
# Tier 0 — oracle pinhole reference: 1 ray per pixel, deterministic, clean supervision
CAMERA_MODE_ORACLE_PINHOLE_REFERENCE = 0

# Tier 1 — physical pinhole: tiny aperture, photon-limited, noise-dominated
CAMERA_MODE_PHYSICAL_PINHOLE = 1

# Tier 2 — aperture cone: disk samples, no lens bending (default for simple scenes)
CAMERA_MODE_APERTURE_CONE = 2

# Tier 3 — ideal thin-lens geometric: focus-plane convergence via thin lens
CAMERA_MODE_THIN_LENS_GEOMETRIC = 3

# Tier 4 — exact analytical compound assembly. Triangle faces are BVH proxies;
# the registered parametric payload evaluates the complete conic surface chain.
CAMERA_MODE_PARAMETRIC_ASSEMBLY = 4
CAMERA_MODE_GEOMETRIC_ASSEMBLY = CAMERA_MODE_PARAMETRIC_ASSEMBLY  # legacy name

# Tier 5 — wave-patch transport: diffraction, interference, finite-element waves (STUB)
CAMERA_MODE_WAVE_ASSEMBLY = 5

# Tier 6 — baked transform function: accelerated LUT/spline/neural transport map (STUB)
CAMERA_MODE_BAKED_TRANSFORM = 6

# Legacy alias for API compatibility during transition (will be removed)
CAMERA_MODE_THICK_LENS_WAVE = 5  # maps to WAVE_ASSEMBLY

# Scale-context KIND enum (mirror of csrc/include/ray_tracer.h SCALE_CONTEXT_KIND_*).
SCALE_CONTEXT_KIND_RAY                 = 0
SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ      = 1
SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM = 2
SCALE_CONTEXT_KIND_THICK_LENS_WAVE     = 3
SCALE_CONTEXT_KIND_SPLINE_SURFACE      = 4
SCALE_CONTEXT_KIND_NEURAL_SURFACE      = 5
SCALE_CONTEXT_KIND_NEURAL_VOLUMETRIC   = 6


# ── EndpointRecord layout — must match csrc/include/bdpt_record.h ──────────
ENDPOINT_DTYPE = np.dtype([
    ("subpath_id",   np.uint32),
    ("band_id",      np.uint32),
    ("group_id",     np.int32),
    ("vertex_index", np.int32),
    ("pos",          np.float32, (3,)),
    ("pathlen_m",    np.float32),
    ("dir",          np.float32, (3,)),
    ("pdf",          np.float32),
    ("amp_re",       np.float32),
    ("amp_im",       np.float32),
    ("cos_theta",    np.float32),
    ("stream_id",    np.float32),
])
assert ENDPOINT_DTYPE.itemsize == 64, "EndpointRecord dtype size mismatch"


def endpoints_view(arr: np.ndarray) -> np.ndarray:
    """Re-view a (N, 16) float32 array (as returned by
    ``RayTracer.bidirectional``) as a structured array of EndpointRecords.

    The underlying memory is NOT copied; the result aliases ``arr``.
    """
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 16:
        raise ValueError(f"expected (N, 16) float32, got shape {a.shape} dtype {a.dtype}")
    return a.view(ENDPOINT_DTYPE).reshape(a.shape[0])


@_dc.dataclass
class CameraSensor:
    """Camera sensor descriptor for SENSOR + PIXEL_CONE TriGroups.

    Mirrors ``CameraSensorDesc`` in csrc/include/triangle_groups.h byte
    for byte semantically (not layout — pybind packs the dict).  Used by
    ``TriangleGroup`` when ``sample_policy == TRI_GROUP_SAMPLE_PIXEL_CONE``
    to drive the integrator's per-pixel cone scan.

    aperture_stop_group_id (-1 = use circular fallback) lets blade-polygon
    geometry decide aperture shape: when set, the integrator BVH-tests
    each candidate ray against the blocker group's tris before launching,
    and a real polygon stop is honored automatically.
    """
    pos:                  np.ndarray   # (3,) float64 — sensor centre
    fwd:                  np.ndarray   # (3,) float64 — unit forward
    up:                   np.ndarray   # (3,) float64 — unit up
    sensor_w_m:           float
    sensor_h_m:           float
    focal_m:              float        # image-side dist sensor→aperture (m)
    aperture_radius_m:    float
    n_px:                 int
    n_py:                 int
    n_aperture_samples:   int
    aperture_stop_group_id: int = -1
    # Optical mode extensions — optional, default = APERTURE_CONE.
    camera_mode:          int   = CAMERA_MODE_APERTURE_CONE
    effective_focal_m:    float = 0.0   # effective focal length; 0 = use focal_m
    focus_distance_m:     float = 0.0   # scene focus distance; 0 = auto
    lens_center: np.ndarray | None = None  # (3,) float64; None = use aperture centre
    lens_fwd:    np.ndarray | None = None  # (3,) float64 unit vec; None = camera fwd

    def to_dict(self) -> dict:
        d = dict(
            pos = np.asarray(self.pos, np.float64).reshape(3),
            fwd = np.asarray(self.fwd, np.float64).reshape(3),
            up  = np.asarray(self.up,  np.float64).reshape(3),
            sensor_w_m         = float(self.sensor_w_m),
            sensor_h_m         = float(self.sensor_h_m),
            focal_m            = float(self.focal_m),
            aperture_radius_m  = float(self.aperture_radius_m),
            n_px               = int(self.n_px),
            n_py               = int(self.n_py),
            n_aperture_samples = int(self.n_aperture_samples),
            aperture_stop_group_id = int(self.aperture_stop_group_id),
            camera_mode        = int(self.camera_mode),
            effective_focal_m  = float(self.effective_focal_m),
            focus_distance_m   = float(self.focus_distance_m),
        )
        if self.lens_center is not None:
            d["lens_center"] = np.asarray(self.lens_center, np.float64).reshape(3)
        if self.lens_fwd is not None:
            d["lens_fwd"] = np.asarray(self.lens_fwd, np.float64).reshape(3)
        return d


@_dc.dataclass
class TriangleGroup:
    """Descriptor passed to ``RayTracer.register_tri_group``."""
    role_bits:    int
    tri_indices:  np.ndarray                       # int32 (N,)
    sample_policy: int      = TRI_GROUP_SAMPLE_AREA
    plane_origin: np.ndarray | None = None         # float64 (3,) or None
    plane_normal: np.ndarray | None = None         # float64 (3,) or None
    custom_emit_W: float    = 0.0
    # Additive (safe to omit): per-group default material + power curve.
    default_mat_idx:  int   = -1                   # -1 = derive from majority
    power_W_per_band: np.ndarray | None = None     # float32 (n_bands,) or None
    parametric_surface: dict[str, Any] | None = None
    # SENSOR + PIXEL_CONE only; ignored otherwise.
    sensor_camera:    CameraSensor | None = None

    def register_with(self, tracer: Any) -> int:
        """Push this group into the supplied ``RayTracer`` and return its
        assigned ``group_id``.
        """
        kwargs = dict(
            role_bits      = int(self.role_bits),
            sample_policy  = int(self.sample_policy),
            tri_indices    = np.ascontiguousarray(self.tri_indices, np.int32),
            plane_origin   = (np.asarray(self.plane_origin, np.float64)
                              if self.plane_origin is not None else None),
            plane_normal   = (np.asarray(self.plane_normal, np.float64)
                              if self.plane_normal is not None else None),
            custom_emit_W  = float(self.custom_emit_W),
            default_mat_idx = int(self.default_mat_idx),
            power_W_per_band = (np.ascontiguousarray(self.power_W_per_band, np.float32)
                                if self.power_W_per_band is not None else None),
            parametric_surface = (self.parametric_surface
                                  if self.parametric_surface is not None else None),
            sensor_camera  = (self.sensor_camera.to_dict()
                              if self.sensor_camera is not None else None),
        )
        return tracer.register_tri_group(**kwargs)


@_dc.dataclass
class IntegratorMetadata:
    """Sidecar that travels with every BDPT render — enables exact replay
    and downstream auditing.  Per the integrator-rewrite directive: we do
    not aggregate or compress storage; this object records everything the
    consumer might need to reconstruct the integration.
    """
    integrator_kind:   str                       # "splat" | "bdpt"
    deposit_mode:      str                       # "all_hits" | "plane" | "field"
    n_rays_per_source: int
    max_bounces:       int
    branch_factor:     int                       # 1 = no fanout
    min_amplitude:     float
    n_bands:           int
    seed_sequence:     list[int]
    bvh_hash:          str                       # sha1 of vertex buffer
    mat_db_hash:       str                       # sha1 of mat_buf
    scale_contexts:    list[dict[str, Any]] = _dc.field(default_factory=list)
    batch_counters:    dict[str, int]      = _dc.field(default_factory=dict)
    extra:             dict[str, Any]      = _dc.field(default_factory=dict)

    def to_json(self) -> str:
        return _json.dumps(_dc.asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def hash_buffer(arr: np.ndarray) -> str:
        h = _hashlib.sha1()
        h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()


def aggregate_to_image(
        records:   np.ndarray,                 # structured (ENDPOINT_DTYPE) or (N,16) float32
        n_bands:   int,
        height:    int,
        width:     int,
        plane_origin: np.ndarray,              # (3,) — sensor centre
        plane_basis_u: np.ndarray,             # (3,) — image x axis (m/pixel × W)
        plane_basis_v: np.ndarray,             # (3,) — image y axis (m/pixel × H)
        ) -> np.ndarray:
    """Reference aggregator: bin EndpointRecords into a complex64 image.

    Returned array has shape ``(n_bands, height, width)`` and dtype
    ``complex64``.  Values are summed (NOT averaged) so each pixel carries
    the coherent sum of all contributing endpoint amplitudes — phase is
    preserved.  Magnitude / phase / RGB are produced by callers as needed.

    No reduction of the input.  Records outside the plane bounds are
    silently dropped.
    """
    if records.dtype != ENDPOINT_DTYPE:
        records = endpoints_view(records)
    out = np.zeros((n_bands, height, width), np.complex64)
    if records.size == 0:
        return out

    # Plane parameterisation: pos = origin + u·basis_u + v·basis_v.  Solve
    # the least-squares projection (basis is not assumed orthonormal).
    bu = np.asarray(plane_basis_u, np.float64)
    bv = np.asarray(plane_basis_v, np.float64)
    M = np.column_stack([bu, bv])              # (3, 2)
    Minv = np.linalg.pinv(M)                   # (2, 3)

    rel = records["pos"].astype(np.float64) - np.asarray(plane_origin, np.float64)
    uv = rel @ Minv.T                          # (N, 2), in [0, 1] inside plane
    px = np.floor(uv[:, 0] * width ).astype(np.int32)
    py = np.floor(uv[:, 1] * height).astype(np.int32)
    bid = records["band_id"].astype(np.int32)
    amp = records["amp_re"].astype(np.float32) + 1j * records["amp_im"].astype(np.float32)

    keep = (px >= 0) & (px < width) & (py >= 0) & (py < height) \
         & (bid >= 0) & (bid < n_bands)
    if not np.any(keep):
        return out
    np.add.at(out, (bid[keep], py[keep], px[keep]), amp[keep])
    return out


def aggregate_to_image_pixel_cone(
        records:   np.ndarray,                 # structured ENDPOINT_DTYPE or (N,16) float32
        n_bands:   int,
        n_px:      int,
        n_py:      int,
        sensor_group_id: int,
        sensor_camera: dict[str, Any] | None = None,
        ) -> np.ndarray:
    """PIXEL_CONE aggregator: bin EndpointRecords by decoding ``subpath_id``.

    Records emitted by the C++ PIXEL_CONE pass encode the destination pixel as
    ``subpath_id = py * n_px + px``.  Use that encoding directly; do not
    project world hit positions back onto the sensor plane.

    Returns a complex64 array of shape ``(n_bands, n_py, n_px)``, summed
    coherently (no normalisation) so the consumer can compute mean,
    variance, magnitude, etc. without precision loss.

    Records belonging to other groups, out-of-range pixels, or invalid
    bands are silently dropped — never reduced.
    """
    if records.dtype != ENDPOINT_DTYPE:
        records = endpoints_view(records)
    out = np.zeros((n_bands, n_py, n_px), np.complex64)
    if records.size == 0:
        return out

    gid = records["group_id"].astype(np.int32)
    sub = records["subpath_id"].astype(np.int64)
    bid = records["band_id"].astype(np.int32)

    keep = (gid == int(sensor_group_id)) & (bid >= 0) & (bid < n_bands)
    if not np.any(keep):
        return out

    bid = bid[keep]
    sub = sub[keep]
    re  = records["amp_re"][keep]
    im  = records["amp_im"][keep]
    amp = re.astype(np.float32) + 1j * im.astype(np.float32)

    px = (sub % int(n_px)).astype(np.int32)
    py = (sub // int(n_px)).astype(np.int32)

    in_range = (px >= 0) & (px < n_px) & (py >= 0) & (py < n_py)
    if not np.any(in_range):
        return out
    np.add.at(out, (bid[in_range], py[in_range], px[in_range]), amp[in_range])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Ray-stream semantic taxonomy
# ═══════════════════════════════════════════════════════════════════════════════
#
# These constants name the ORIGIN and TRAVEL DIRECTION of a ray sub-path.
# They are attached to EndpointRecord rows (or carried through the new
# RayEvent structs) so that downstream consumers can make semantically
# correct routing decisions — e.g. refusing to deposit a backward-sensor
# record into the forward physical field buffer.
#
# Values match the C-side enum in csrc/include/ray_stream_kind.h (to be
# added when the C kernel is updated).  Until then only the Python side
# uses them.

class RayStreamKind:
    """Integer constants naming the origin/direction of a ray sub-path."""
    FORWARD_LIGHT   = 0   # emitter → scene → sensor  (light-side subpath)
    BACKWARD_SENSOR = 1   # sensor → scene → emitter  (sensor-side subpath)
    MIDDLE_MANIFOLD = 2   # neither end is a primary emitter or sensor;
                          # placed by a manifold-walk or connection strategy
    APERTURE_PUPIL  = 3   # backward-sensor sub-path that enters the scene
                          # through the aperture / exit pupil.  This IS a
                          # BACKWARD_SENSOR path — the PIXEL_CONE pass is one
                          # concrete realisation of this stream.  It is a valid
                          # half of a true BDPT pair and must NOT be removed;
                          # it simply cannot be deposited unresolved into the
                          # physical field buffer without first being correlated
                          # with a forward light sub-path by RayCorrelator.
    DIAGNOSTIC_ONLY = 4   # carries no physical energy; display/debug only


class RecordIntent:
    """Integer constants describing HOW an EndpointRecord should be used.

    These are NOT mutually exclusive by value — a record can be promoted from
    CORRELATION_CANDIDATE to PHYSICAL_DEPOSIT only after the correlator
    resolves a valid connection.  Consumers must check intent before writing
    to any shared accumulation buffer.
    """
    PHYSICAL_DEPOSIT      = 0   # resolved contribution — safe to write to
                                #   the physical field / image accumulator
    DIAGNOSTIC_RECORD     = 1   # display-only; must NOT touch physical buffers
    CORRELATION_CANDIDATE = 2   # endpoint proposed for a connection strategy;
                                #   becomes PHYSICAL_DEPOSIT iff accepted by
                                #   RayCorrelator (visibility confirmed, MIS done)
    SENSOR_ESTIMATE       = 3   # backward sub-path sensor estimate (e.g.
                                #   PIXEL_CONE / APERTURE_PUPIL pass).  Writes
                                #   ONLY to the per-pixel sensor image buffer,
                                #   never to the physical 3-D field grid unless
                                #   first promoted via correlation.
    UV_CACHE_SAMPLE       = 4   # read from the UV lightfield cache
    UV_CACHE_WRITE        = 5   # write to the UV lightfield cache
    FIELD_CAPTURE_WRITE   = 6   # write to the 3-D volumetric field grid


# ═══════════════════════════════════════════════════════════════════════════════
# Structured data model for the forward / backward / middle correlation system
# ═══════════════════════════════════════════════════════════════════════════════

@_dc.dataclass
class RayEvent:
    """One recorded event on a ray sub-path (a surface or volume hit).

    Carries enough information to connect sub-paths without re-tracing.
    All positions and directions are in world-space metres / unit vectors.

    ``stream_kind``  — RayStreamKind constant, set at launch and propagated
                       through every bounce.
    ``record_intent`` — RecordIntent constant, set by the kernel or correlator.
    ``subpath_id``   — opaque ID linking all events on the same sub-path;
                       matches EndpointRecord.subpath_id when bridging records.
    ``bounce_index`` — 0-based depth from the sub-path origin.
    ``pos``          — hit position (world-space, metres), shape (3,).
    ``dir_in``       — incoming direction at this event (normalised), shape (3,).
    ``dir_out``      — scattered/reflected direction (normalised), shape (3,).
    ``normal``       — surface/volume normal at hit (normalised), shape (3,).
    ``pdf``          — sampling PDF for the outgoing direction.
    ``pathlen_m``    — optical path length from the sub-path origin (metres).
    ``amp_re``       — per-band amplitude real part, shape (n_bands,).
    ``amp_im``       — per-band amplitude imaginary part, shape (n_bands,).
    ``group_id``     — registered tri-group ID of the hit surface (-1 = none).
    ``tri_id``       — local triangle index within the group (-1 = none).
    """
    stream_kind:    int
    record_intent:  int
    subpath_id:     int
    bounce_index:   int
    pos:            np.ndarray          # float64, shape (3,)
    dir_in:         np.ndarray          # float64, shape (3,)
    dir_out:        np.ndarray          # float64, shape (3,)
    normal:         np.ndarray          # float64, shape (3,)
    pdf:            float
    pathlen_m:      float
    amp_re:         np.ndarray          # float32, shape (n_bands,)
    amp_im:         np.ndarray          # float32, shape (n_bands,)
    group_id:       int = -1
    tri_id:         int = -1


@_dc.dataclass
class MiddlePoint:
    """A proposed connection point on a manifold walk or a shared surface.

    Produced by RayCorrelator strategies that search for a common vertex
    between a forward light sub-path and a backward sensor sub-path.

    ``forward_event``  — the RayEvent on the light sub-path nearest to this
                         connection point (may be an extrapolation).
    ``backward_event`` — the RayEvent on the sensor sub-path nearest to this
                         connection point.
    ``pos``            — resolved connection world position, shape (3,).
    ``visibility``     — fraction in [0, 1]; 1.0 = fully unoccluded.
    ``mis_weight``     — multiple importance sampling weight for this
                         connection strategy (may be 1.0 if not yet
                         computed).
    """
    forward_event:  RayEvent
    backward_event: RayEvent
    pos:            np.ndarray          # float64, shape (3,)
    visibility:     float = 1.0
    mis_weight:     float = 1.0


@_dc.dataclass
class CorrelationCandidate:
    """A matched (forward, backward) sub-path pair proposed by RayCorrelator.

    Once a candidate is accepted (visibility confirmed, MIS weight applied)
    it can be promoted to a PHYSICAL_DEPOSIT and written into the sensor
    accumulator.

    ``middle``         — the shared / connection point (may be None if the
                         paths connect directly endpoint-to-endpoint).
    ``forward_record`` — the EndpointRecord row (structured) for the light
                         sub-path endpoint, or None if source is a RayEvent.
    ``backward_record``— the EndpointRecord row (structured) for the sensor
                         sub-path endpoint, or None if source is a RayEvent.
    ``strategy_id``    — RayCorrelator strategy that produced this candidate.
    ``contribution``   — per-band complex amplitude after MIS weighting,
                         shape (n_bands,) complex64.
    ``accepted``       — True once visibility check passes and MIS is final.
    """
    middle:           MiddlePoint | None
    forward_record:   np.ndarray | None     # single structured EndpointRecord
    backward_record:  np.ndarray | None     # single structured EndpointRecord
    strategy_id:      int = 0
    contribution:     np.ndarray | None = None   # complex64, shape (n_bands,)
    accepted:         bool = False
