"""camera_designer/ray_order.py
==============================
RayOrder — the bridge between EmitterProfile physics and the C ray tracer.

The C ray tracer (``ray_tracer_trace``) takes a flat list of point sources:

    src_pos[N, 3]       float64  — world-space positions
    src_dir[N, 3]       float64  — dominant emission direction per source
    src_directivity[N]  float64  — cosine-power exponent: rays weighted by
                                   ((cosθ+1)/2)^directivity in Fibonacci sphere

That is the complete source description the C kernel understands.  It knows
nothing about:

  • Spectral content          — amplitude weights per band are not passed
  • Phase / coherence         — all rays start with amplitude 1.0, random phase
  • Polarization              — scalar amplitude only, no Jones vector
  • Extended spatial area     — each C source is a mathematical point
  • Non-cosine directivities  — DIPOLE_INPLANE_MIXED, GAUSSIAN_BEAM, etc.
    cannot be expressed as cosine^n for any finite n; the C kernel will always
    produce an inexact distribution

A ``RayOrder`` expands one or more ``EmitterSpec`` objects (which carry
``EmitterProfile`` references, disc geometry, and orientation) into the flat
source arrays the tracer accepts.  It records every physical property that
was silently dropped or approximated in ``tracer_gaps`` so callers can decide
whether the result is fit for purpose.

Usage
-----
    from camera_designer.ray_order import RayOrder
    from camera_designer.camera_preset import EmitterSpec

    specs = preset.emitters   # List[EmitterSpec]
    order = RayOrder.from_emitter_specs(
        specs,
        wavelengths_um        = np.array([0.45, 0.55, 0.65]),
        emitter_transform     = None,      # 4×4 world transform or None = identity
        n_spatial_samples     = 4,         # disc sample points per component
        n_rays_per_source     = 512,
        max_bounces           = 6,
        min_amplitude         = 1e-4,
        seed                  = 0,
    )

    print(order.tracer_gap_report())

    # Direct call to C tracer
    n_written = ctypes.c_int(0)
    out_buf   = np.zeros(order.out_cap_estimate * 12, np.float32)
    lib.ray_tracer_trace(
        tracer_handle,
        order.n_sources,
        order.src_pos.ctypes...,
        order.src_dir.ctypes...,
        order.src_directivity.ctypes...,
        order.n_rays_per_source,
        order.max_bounces,
        order.min_amplitude,
        order.seed,
        out_buf.ctypes...,
        order.out_cap_estimate,
        ctypes.byref(n_written),
    )
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .emitter_profile import (
    AngularDistribution,
    CoherenceModel,
    DirectionalModel,
    EmitterProfile,
    PhaseState,
    PolarizationMode,
    PolarizationState,
    SpectralDistribution,
    SpectralModel,
)
from .camera_preset import EmitterSpec

__all__ = [
    "TracerGap",
    "SourceRecord",
    "RayOrder",
    "BidirectionalRayPackage",
]

# ─────────────────────────────────────────────────────────────────────────────
# TracerGap — structured record of one physics property that was dropped
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TracerGap:
    """One physical property present in an EmitterProfile that the C ray tracer
    cannot honour.

    severity
    --------
    "approximated"  — a best-effort substitute was used; result may be
                      quantitatively off but qualitatively reasonable.
    "dropped"       — the property was silently ignored; no substitute exists
                      within the current C API.
    "clamped"       — the C tracer imposes a hard limit that overrides the
                      requested value (e.g. directivity minimum of 0.5 when
                      the kernel clamps max(0.5, dirpow)).
    """
    source_label:   str   # which SourceRecord this pertains to
    property_name:  str   # e.g. "directional_model", "polarization", "phase"
    severity:       str   # "approximated" | "dropped" | "clamped"
    detail:         str   # human-readable explanation


# ─────────────────────────────────────────────────────────────────────────────
# SourceRecord — one expanded point source ready for the C tracer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceRecord:
    """Single point source row produced by expanding an EmitterSpec component.

    Fields used by the C tracer
    ---------------------------
    pos             : (3,) float64 — world-space emission point
    dir             : (3,) float64 — dominant emission direction (normalised)
    directivity     : float64      — cosine-power exponent passed to C tracer
                                     ((cosθ+1)/2)^directivity weighting;
                                     NOTE: C kernel clamps max(0.5, directivity)

    Fields that annotate what was dropped / approximated
    -----------------------------------------------------
    amplitude_weights : (n_bands,) float64 — per-band relative amplitude;
                        NOT sent to the C tracer (C has no per-band weight
                        input); callers must post-process segments by band.
    phase_state       : the original PhaseState — NOT sent to C tracer (all
                        rays initialise with amplitude=1, random phase).
    polarization      : the original PolarizationState — NOT sent to C tracer
                        (scalar amplitude, no Jones vector).
    directional_model : the original DirectionalModel — for reference; the
                        actual directivity field is the best cosine-power fit.
    profile           : the leaf EmitterProfile this row came from.
    spec_label        : label from the parent EmitterSpec.
    component_label   : name of the profile component (or "" for leaf).
    tracer_gaps       : list of TracerGap records for this source row.
    """
    pos:               np.ndarray       # (3,) float64
    dir:               np.ndarray       # (3,) float64
    directivity:       float
    amplitude_weights: np.ndarray       # (n_bands,) float64
    phase_state:       PhaseState
    polarization:      PolarizationState
    directional_model: DirectionalModel
    profile:           EmitterProfile
    spec_label:        str
    component_label:   str
    tracer_gaps:       List[TracerGap] = field(default_factory=list)


@dataclass
class BidirectionalRayPackage:
        """Packed forward/backward ray batches for bidirectional workflows.

        Layout for both arrays is the existing (N,12) float32 SourceRec schema.
        Reserved fields are populated for routing metadata:
            dir_kind.w  (col 7)  : role tag (0=forward source, 1=sensor backward)
            packet.w    (col 11) : group id (source index or sensor index)
        """
        forward_rows: np.ndarray
        backward_rows: np.ndarray
        combined_rows: np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Directional model → best-fit cosine-power exponent
# ─────────────────────────────────────────────────────────────────────────────

def _best_fit_directivity(
    angular: AngularDistribution,
    label:   str,
    gaps:    List[TracerGap],
) -> float:
    """Convert an AngularDistribution to the best cosine^n approximation.

    Returns the directivity exponent.  Appends TracerGap entries to ``gaps``
    for every approximation or hard drop made.

    Physics notes
    -------------
    LAMBERTIAN: I(θ) = cosθ → cosine^1.  Exact match.

    GAUSSIAN_BEAM: I(θ) ∝ exp(−2θ²/θ_d²).  No closed-form cosine^n match.
    We match at the half-power point: cos^n(θ_d) = 0.5 ⟹ n = ln0.5/ln(cosθ_d).
    Error is significant outside the central lobe.  Severity: approximated.

    DIPOLE_ELECTRIC (z-axis): I(θ) = sin²θ — maximum at 90°, zero on-axis.
    No cosine^n can represent this.  We use directivity=0.0 (nearest to isotropic
    the C API allows) and record a gap.  NOTE: the C kernel clamps
    max(0.5, dirpow), so the actual distribution will be a shallow
    forward hemisphere — still wrong for z-dipole.  Severity: dropped.

    DIPOLE_INPLANE_MIXED: I(θ) = (1+cos²θ)/2 — maximum on-axis, minimum 0.5
    at 90°.  Never reaches zero.  cosine^n always reaches zero at 90° for n>0.
    Best fit: minimise ∫₀^π [cos^n(θ) - (1+cos²θ)/2]² sinθ dθ over n.
    Numerical minimum is near n ≈ 0.5 but still has qualitative error at
    grazing angles.  Severity: approximated (note about C clamp).

    ETENDUE_LIMITED: flat top within NA — approximate as narrow beam with
    n = ln0.5 / ln(cos(arcsin(NA))).  Error within pass-band is zero; error
    at the sharp cutoff edge is large.  Severity: approximated.

    HENYEY_GREENSTEIN: full distribution not reducible to cosine^n.
    For g≥0 use n = (1+g)/(1-g) as a rough forward-lobe match.
    Severity: approximated for g≠0.

    PROJECTIVE: ideal hard cone — approximate as very high cosine^n.
    n = ln0.5 / ln(cos(half_angle)).  Severity: approximated (C has no hard
    cone; bright halo near the cutoff angle is unavoidable).

    MEASURED: no analytical form.  Return directivity=1.0 (Lambertian proxy)
    and record a gap.  Severity: dropped.
    """
    m = angular.model

    if m == DirectionalModel.LAMBERTIAN:
        # Exact: I(θ) = cosθ = cos^1 θ
        return 1.0

    if m == DirectionalModel.GAUSSIAN_BEAM:
        theta_d = angular.effective_divergence_rad
        if theta_d <= 0.0 or theta_d >= math.pi / 2:
            n_fit = 1.0
        else:
            cos_td = math.cos(theta_d)
            if cos_td <= 0.0 or cos_td >= 1.0:
                n_fit = 1.0
            else:
                n_fit = math.log(0.5) / math.log(cos_td)
        gaps.append(TracerGap(
            source_label  = label,
            property_name = "directional_model",
            severity      = "approximated",
            detail        = (
                f"GAUSSIAN_BEAM (θ_d={math.degrees(theta_d):.2f}°) cannot be exactly "
                f"represented as cosine^n; matched at half-power point → n={n_fit:.2f}. "
                f"Lobes outside 2×θ_d are overestimated."
            ),
        ))
        return n_fit

    if m in (DirectionalModel.DIPOLE_ELECTRIC, DirectionalModel.DIPOLE_MAGNETIC):
        ax, ay, az = angular.dipole_axis
        axis_len = math.sqrt(ax*ax + ay*ay + az*az)
        az_norm = az / axis_len if axis_len > 1e-12 else 1.0
        if abs(az_norm) > 0.99:
            # z-axis dipole: sin²θ — completely wrong for cosine^n
            gaps.append(TracerGap(
                source_label  = label,
                property_name = "directional_model",
                severity      = "dropped",
                detail        = (
                    "DIPOLE_ELECTRIC with z-axis has I(θ)=sin²θ — zero on-axis, "
                    "maximum at 90°. No cosine^n can represent this pattern. "
                    "Using directivity=0.0 (C kernel clamps to 0.5 = shallow forward "
                    "hemisphere). Result is qualitatively inverted."
                ),
            ))
        else:
            # In-plane dipole — forward-peaked ≈ Lambertian quality
            gaps.append(TracerGap(
                source_label  = label,
                property_name = "directional_model",
                severity      = "approximated",
                detail        = (
                    f"DIPOLE_ELECTRIC with non-z-axis ({ax_norm:.2f},{ay_norm:.2f},{az_norm:.2f}): "
                    f"φ-averaged pattern not cosine^n. Using directivity=0.5 as proxy "
                    f"(broader than Lambertian). Grazing emission is underestimated."
                ),
            ))
        _add_clamped_gap(label, 0.0, gaps)
        return 0.0

    if m == DirectionalModel.DIPOLE_INPLANE_MIXED:
        # I(θ) = (1+cos²θ)/2. Numerical best-fit of cos^n to this via L2 integral:
        # ∫₀^π [cos^n(θ) - (1+cos²θ)/2]² sinθ dθ minimised.
        # Integration gives: choose n that minimises:
        #   ∫ cos^(2n)θ sinθ dθ  - ∫ cos^n(θ)(1+cos²θ)/2 sinθ dθ + const
        # Using substitution u=cosθ, ∫₋₁¹ ... du:
        # ∫₋₁¹ u^(2n) du = 2/(2n+1)
        # ∫₋₁¹ u^n (1+u²)/2 du = [1/(n+1) + 1/(n+3)] for even n, else ≈ 0
        # Since the dipole pattern is symmetric and even in cosθ, we restrict to
        # even-parity component: I(θ) = 0.5 + 0.5*cos²θ.  Best matching even
        # cosine power: numerically this gives n ≈ 0.5.
        n_fit = 0.5
        gaps.append(TracerGap(
            source_label  = label,
            property_name = "directional_model",
            severity      = "approximated",
            detail        = (
                "DIPOLE_INPLANE_MIXED has I(θ)=(1+cos²θ)/2 — non-zero at θ=90° (0.5). "
                f"Best cosine^n L2 fit: n≈{n_fit}. Grazing emission (θ→90°) will be zero "
                "in the C tracer instead of 0.5; forward lobe shape is approximately correct. "
                "C kernel additionally clamps directivity to max(0.5, 0.5)=0.5."
            ),
        ))
        # No extra clamp gap since 0.5 == kernel clamp minimum
        return n_fit

    if m == DirectionalModel.ETENDUE_LIMITED:
        na = angular.numerical_aperture
        na = min(max(na, 1e-4), 1.0 - 1e-9)
        theta_max = math.asin(na)
        cos_tm = math.cos(theta_max)
        if cos_tm <= 0.0 or cos_tm >= 1.0:
            n_fit = 1.0
        else:
            n_fit = math.log(0.5) / math.log(cos_tm)
        gaps.append(TracerGap(
            source_label  = label,
            property_name = "directional_model",
            severity      = "approximated",
            detail        = (
                f"ETENDUE_LIMITED (NA={na:.3f}, θ_max={math.degrees(theta_max):.1f}°): "
                f"hard-aperture flat top replaced by cosine^{n_fit:.2f}. "
                "Sharp cutoff is softened; energy outside NA is non-zero."
            ),
        ))
        return n_fit

    if m == DirectionalModel.HENYEY_GREENSTEIN:
        g = angular.hg_g
        if abs(g) < 1e-3:
            return 1.0   # isotropic → Lambertian proxy
        if g > 0.0:
            n_fit = max(0.5, (1.0 + g) / (1.0 - g))
        else:
            # Backward-peaked: no good cosine^n match; use nearly-isotropic
            n_fit = 0.5
        gaps.append(TracerGap(
            source_label  = label,
            property_name = "directional_model",
            severity      = "approximated",
            detail        = (
                f"HENYEY_GREENSTEIN (g={g:.3f}) replaced by cosine^{n_fit:.2f}. "
                "Full HG lobe shape (ring structure) is not reproduced."
            ),
        ))
        if n_fit <= 0.5:
            _add_clamped_gap(label, n_fit, gaps)
        return n_fit

    if m == DirectionalModel.PROJECTIVE:
        ha = angular.half_angle_rad
        if ha <= 0.0 or ha >= math.pi / 2:
            n_fit = 1.0
        else:
            cos_ha = math.cos(ha)
            n_fit = math.log(0.5) / math.log(max(cos_ha, 1e-12))
        gaps.append(TracerGap(
            source_label  = label,
            property_name = "directional_model",
            severity      = "approximated",
            detail        = (
                f"PROJECTIVE hard cone (half-angle={math.degrees(ha):.1f}°) replaced by "
                f"cosine^{n_fit:.2f}. Emission outside the cone is non-zero; "
                "the sharp cutoff is not reproduced."
            ),
        ))
        return n_fit

    # MEASURED — no analytical approximation possible
    gaps.append(TracerGap(
        source_label  = label,
        property_name = "directional_model",
        severity      = "dropped",
        detail        = (
            "MEASURED angular distribution requires texture lookup; C tracer has "
            "no texture input. Replaced by LAMBERTIAN (directivity=1.0). "
            "Angular pattern is entirely wrong."
        ),
    ))
    return 1.0


def _add_clamped_gap(label: str, requested: float, gaps: List[TracerGap]) -> None:
    """Record the C tracer's directivity clamp when requested < 0.5."""
    if requested < 0.5:
        gaps.append(TracerGap(
            source_label  = label,
            property_name = "directivity_clamp",
            severity      = "clamped",
            detail        = (
                f"C kernel enforces max(0.5, directivity): requested {requested:.3f} "
                f"becomes 0.5 (shallow forward hemisphere, not full sphere). "
                "True isotropic / backward emission is not achievable."
            ),
        ))


def _phase_gaps(phase: PhaseState, label: str) -> List[TracerGap]:
    """Build TracerGap list for PhaseState properties the C tracer ignores."""
    gaps: List[TracerGap] = []
    if phase.model != CoherenceModel.INCOHERENT:
        gaps.append(TracerGap(
            source_label  = label,
            property_name = "coherence_model",
            severity      = "dropped",
            detail        = (
                f"PhaseState.model={phase.model.value}: C tracer initialises all rays "
                "with amplitude=1.0 and random phase — no coherence length, no "
                "carrier phase offset, no temporal coherence. "
                f"Requested coherence_length_um={phase.coherence_length_um:.3g}."
            ),
        ))
    if phase.phase_offset_rad != 0.0:
        gaps.append(TracerGap(
            source_label  = label,
            property_name = "phase_offset",
            severity      = "dropped",
            detail        = (
                f"phase_offset_rad={phase.phase_offset_rad:.4f} ignored; "
                "C tracer has no deterministic phase initialisation."
            ),
        ))
    return gaps


def _polarization_gaps(pol: PolarizationState, label: str) -> List[TracerGap]:
    """Build TracerGap list for polarization state the C tracer ignores."""
    if pol.mode == PolarizationMode.UNPOLARIZED:
        return []
    return [TracerGap(
        source_label  = label,
        property_name = "polarization",
        severity      = "dropped",
        detail        = (
            f"PolarizationState.mode={pol.mode.value}: C tracer tracks scalar "
            "amplitude only; no Jones vector, no Stokes parameters. "
            "Polarization is entirely lost."
        ),
    )]


# ─────────────────────────────────────────────────────────────────────────────
# Disc spatial sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_disc(
    center:   np.ndarray,   # (3,) float64
    normal:   np.ndarray,   # (3,) float64 normalised
    radius:   float,
    n_pts:    int,
    rng:      np.random.Generator,
) -> np.ndarray:
    """Return ``n_pts`` positions sampled from a disc using stratified Halton
    low-discrepancy sampling.

    For n_pts == 1 the centre of the disc is returned.
    For n_pts > 1 samples are placed on a sunflower (Fibonacci disc) grid —
    better coverage than random for small N.

    Returns (n_pts, 3) float64.
    """
    if n_pts == 1:
        return center.reshape(1, 3).copy()

    # Build orthonormal (u, v) tangent frame for the disc
    n = normal / (np.linalg.norm(normal) + 1e-30)
    # Pick a non-parallel reference vector
    ref = np.array([0., 0., 1.], np.float64)
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([1., 0., 0.], np.float64)
    u_ax = np.cross(ref, n)
    u_ax /= np.linalg.norm(u_ax) + 1e-30
    v_ax = np.cross(n, u_ax)

    # Fibonacci disc (Vogel spiral)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    idx = np.arange(n_pts, dtype=np.float64)
    r_arr = radius * np.sqrt((idx + 0.5) / n_pts)
    theta_arr = 2.0 * math.pi * idx / (golden * golden)

    x2d = r_arr * np.cos(theta_arr)
    y2d = r_arr * np.sin(theta_arr)

    pts = (center[np.newaxis, :]
           + x2d[:, np.newaxis] * u_ax[np.newaxis, :]
           + y2d[:, np.newaxis] * v_ax[np.newaxis, :])
    return pts.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# RayOrder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RayOrder:
    """Expanded, tracer-ready source description derived from EmitterSpec(s).

    This is the complete translation layer between the physics in
    ``EmitterProfile`` (composite, spectral, coherent, polarized) and the C
    ray tracer's minimal flat source API.

    Construct via ``RayOrder.from_emitter_specs()`` — do not instantiate
    directly.

    Attributes
    ----------
    sources         : expanded SourceRecord list (one per spatial sample × component)
    wavelengths_um  : (n_bands,) float64 — wavelength grid shared by all sources
    n_rays_per_source : rays spawned per source by the C tracer
    max_bounces     : max reflection depth
    min_amplitude   : ray amplitude cutoff
    seed            : RNG seed for C tracer Monte Carlo diffuse scatter

    Packed C-tracer arrays (computed once, cached as properties)
    ------------------
    src_pos          : (N, 3) float64
    src_dir          : (N, 3) float64
    src_directivity  : (N,)   float64

    Per-source metadata (NOT sent to C tracer)
    ------------------------------------------
    amplitude_weights_matrix : (N, n_bands) float64 — spectral amplitude per
                                source per band.  Use to post-weight segment
                                amplitudes by matching segment[8] (freq_band)
                                to the column index.
    all_tracer_gaps          : flat list of all TracerGap records across all sources
    """
    sources:              List[SourceRecord]
    wavelengths_um:       np.ndarray          # (n_bands,) float64
    n_rays_per_source:    int
    max_bounces:          int
    min_amplitude:        float
    seed:                 int
    _src_pos:             np.ndarray = field(init=False, repr=False)
    _src_dir:             np.ndarray = field(init=False, repr=False)
    _src_directivity:     np.ndarray = field(init=False, repr=False)
    _amp_weights:         np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        N = len(self.sources)
        B = len(self.wavelengths_um)
        self._src_pos        = np.empty((N, 3), np.float64)
        self._src_dir        = np.empty((N, 3), np.float64)
        self._src_directivity = np.empty(N,     np.float64)
        self._amp_weights    = np.empty((N, B), np.float64)
        for i, sr in enumerate(self.sources):
            self._src_pos[i]         = sr.pos
            self._src_dir[i]         = sr.dir
            self._src_directivity[i] = sr.directivity
            self._amp_weights[i]     = sr.amplitude_weights

    # ── Packed arrays ─────────────────────────────────────────────────────

    @property
    def src_pos(self) -> np.ndarray:
        """(N, 3) float64 — source positions."""
        return self._src_pos

    @property
    def src_dir(self) -> np.ndarray:
        """(N, 3) float64 — dominant emission directions."""
        return self._src_dir

    @property
    def src_directivity(self) -> np.ndarray:
        """(N,) float64 — cosine-power directivity exponents."""
        return self._src_directivity

    @property
    def amplitude_weights_matrix(self) -> np.ndarray:
        """(N, n_bands) float64 — per-source per-band spectral amplitude.

        NOT sent to the C tracer.  After tracing, multiply each segment's
        amplitude by ``amplitude_weights_matrix[source_id, band_id]`` to
        restore the spectral weighting that the C tracer discarded.
        """
        return self._amp_weights

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def n_bands(self) -> int:
        return len(self.wavelengths_um)

    @property
    def out_cap_estimate(self) -> int:
        """Conservative upper bound on segment count for buffer allocation.

        C tracer writes at most n_sources × n_rays × max_bounces × n_bands
        segments.  In practice far fewer survive the amplitude cutoff.
        """
        return self.n_sources * self.n_rays_per_source * self.max_bounces * self.n_bands

    @property
    def all_tracer_gaps(self) -> List[TracerGap]:
        """Flat list of every TracerGap recorded across all SourceRecords."""
        out: List[TracerGap] = []
        for sr in self.sources:
            out.extend(sr.tracer_gaps)
        return out

    def tracer_gap_report(self) -> str:
        """Human-readable summary of all physics properties that were dropped
        or approximated when building this RayOrder."""
        gaps = self.all_tracer_gaps
        if not gaps:
            return "RayOrder: no tracer gaps — all profile properties honoured."
        lines = [
            f"RayOrder tracer gap report — {len(gaps)} gap(s) across "
            f"{self.n_sources} source(s):",
            "",
        ]
        by_severity: Dict[str, List[TracerGap]] = {}
        for g in gaps:
            by_severity.setdefault(g.severity, []).append(g)
        for sev in ("dropped", "clamped", "approximated"):
            bucket = by_severity.get(sev, [])
            if not bucket:
                continue
            lines.append(f"  [{sev.upper()}] — {len(bucket)} item(s):")
            for g in bucket:
                lines.append(f"    source={g.source_label!r}  property={g.property_name}")
                lines.append(f"      {g.detail}")
            lines.append("")
        return "\n".join(lines)

    # ── Constructor ───────────────────────────────────────────────────────

    @classmethod
    def from_emitter_specs(
        cls,
        specs:                Sequence[EmitterSpec],
        wavelengths_um:       np.ndarray,
        emitter_transform:    Optional[np.ndarray] = None,
        n_spatial_samples:    int   = 1,
        n_rays_per_source:    int   = 512,
        max_bounces:          int   = 6,
        min_amplitude:        float = 1e-4,
        seed:                 int   = 0,
        aim_point:            Optional[np.ndarray] = None,
    ) -> "RayOrder":
        """Build a RayOrder from a list of EmitterSpec objects.

        Parameters
        ----------
        specs               : emitter placements from CameraPreset.emitters
        wavelengths_um      : (n_bands,) float64 — wavelength grid
        emitter_transform   : (4, 4) float64 world transform applied to
                              emitter positions and normals; None = identity
        n_spatial_samples   : number of disc sample points per component.
                              1 = point source at disc centre (fastest).
                              N > 1 = Fibonacci disc grid (extended source).
                              TRACER GAP: C tracer treats each sample as an
                              independent point source; spatial coherence
                              between samples is not modelled.
        n_rays_per_source   : rays per C-tracer source
        max_bounces         : max reflection depth
        min_amplitude       : ray amplitude cutoff
        seed                : RNG seed
        aim_point           : if given, all source directions are overridden
                              to aim at this world-space point.  Overrides
                              EmitterSpec.normal for every source.
                              TRACER GAP: overrides directional model entirely;
                              all sources become forward-pointing regardless of
                              DIPOLE_INPLANE_MIXED etc.
        """
        wl = np.asarray(wavelengths_um, dtype=np.float64)
        rng = np.random.default_rng(seed)
        transform = (np.asarray(emitter_transform, dtype=np.float64)
                     if emitter_transform is not None else None)

        sources: List[SourceRecord] = []

        for spec in specs:
            if not spec.enabled:
                continue

            profile = spec.resolve_profile()

            # Determine the list of (weight, leaf_profile) pairs
            if profile.components:
                component_pairs = [(w, sub) for w, sub in profile.components]
            else:
                component_pairs = [(1.0, profile)]

            # Spec geometry in local frame
            spec_pos    = np.asarray(spec.pos,    dtype=np.float64)
            spec_normal = np.asarray(spec.normal, dtype=np.float64)
            spec_radius = float(spec.radius)
            spec_label  = str(spec.label)

            # Apply world transform to position and normal
            if transform is not None:
                R3  = transform[:3, :3]
                t   = transform[:3,  3]
                spec_pos    = R3 @ spec_pos + t
                spec_normal = R3 @ spec_normal
            nl = np.linalg.norm(spec_normal)
            if nl < 1e-12:
                spec_normal = np.array([0., 0., -1.], np.float64)
            else:
                spec_normal = spec_normal / nl

            # Spatial sample positions on disc
            disc_pts = _sample_disc(spec_pos, spec_normal, spec_radius,
                                    n_spatial_samples, rng)

            for comp_weight, leaf in component_pairs:
                comp_label = leaf.name

                # Source direction
                if aim_point is not None:
                    # All disc samples aim toward the same point; direction
                    # varies per sample (correct for extended source aiming)
                    def _aim_dir(p: np.ndarray) -> np.ndarray:
                        d = np.asarray(aim_point, np.float64) - p
                        nl2 = np.linalg.norm(d)
                        return d / nl2 if nl2 > 1e-12 else spec_normal.copy()
                else:
                    def _aim_dir(p: np.ndarray) -> np.ndarray:
                        return spec_normal.copy()

                # Spectral amplitude weights at each wavelength
                raw_spec = leaf.spectral.sample_array(wl)   # preserves dtype
                # Scale by component weight and radiant_exitance
                amp_weights = (comp_weight * leaf.spectral.radiant_exitance
                               * raw_spec).astype(np.float64)

                # Best-fit directivity
                row_gaps: List[TracerGap] = []
                directivity = _best_fit_directivity(
                    leaf.directional, comp_label, row_gaps)

                # Aim-point override gap
                if aim_point is not None:
                    row_gaps.append(TracerGap(
                        source_label  = comp_label,
                        property_name = "aim_point_override",
                        severity      = "approximated",
                        detail        = (
                            "aim_point was specified: all sources directed toward "
                            f"aim={aim_point}. EmitterSpec.normal and directional "
                            "model are secondary to this geometric constraint."
                        ),
                    ))

                # Phase gaps
                row_gaps.extend(_phase_gaps(leaf.phase, comp_label))

                # Polarization gaps
                row_gaps.extend(_polarization_gaps(leaf.polarization, comp_label))

                # Spatial extension gap (if n_spatial_samples > 1)
                if n_spatial_samples > 1:
                    row_gaps.append(TracerGap(
                        source_label  = comp_label,
                        property_name = "spatial_coherence",
                        severity      = "dropped",
                        detail        = (
                            f"n_spatial_samples={n_spatial_samples}: each disc sample "
                            "is an independent C-tracer point source. Mutual spatial "
                            "coherence between samples is not modelled. Interference "
                            "fringes from extended coherent sources will not appear."
                        ),
                    ))

                # Amplitude weight gap
                row_gaps.append(TracerGap(
                    source_label  = comp_label,
                    property_name = "spectral_amplitude_weights",
                    severity      = "dropped",
                    detail        = (
                        "Per-band amplitude weights computed from EmitterProfile "
                        "spectral distribution are stored in RayOrder."
                        "amplitude_weights_matrix but are NOT passed to the C tracer. "
                        "All bands are traced with equal amplitude. Post-multiply "
                        "segment amplitudes by amplitude_weights_matrix[src_id, band_id]."
                    ),
                ))

                for pt in disc_pts:
                    sources.append(SourceRecord(
                        pos               = pt.copy(),
                        dir               = _aim_dir(pt),
                        directivity       = directivity,
                        amplitude_weights = amp_weights.copy(),
                        phase_state       = leaf.phase,
                        polarization      = leaf.polarization,
                        directional_model = leaf.directional.model,
                        profile           = leaf,
                        spec_label        = spec_label,
                        component_label   = comp_label,
                        tracer_gaps       = list(row_gaps),
                    ))

        if not sources:
            raise ValueError(
                "RayOrder.from_emitter_specs: no enabled emitters produced any "
                "sources. Enable at least one EmitterSpec."
            )

        return cls(
            sources           = sources,
            wavelengths_um    = wl,
            n_rays_per_source = n_rays_per_source,
            max_bounces       = max_bounces,
            min_amplitude     = min_amplitude,
            seed              = seed,
        )

    @classmethod
    def from_lights_list(
        cls,
        lights:             List[dict],
        wavelengths_um:     np.ndarray,
        n_rays_per_source:  int   = 512,
        max_bounces:        int   = 6,
        min_amplitude:      float = 1e-4,
        seed:               int   = 0,
        aim_point:          Optional[np.ndarray] = None,
    ) -> "RayOrder":
        """Build a RayOrder from the legacy ``lights`` list format.

        Each dict may have keys: ``"pos"`` (3-list), ``"dir"`` (3-list),
        ``"power"`` (float, used as directivity), ``"color"`` (3-list, ignored
        — not a spectral distribution), ``"label"`` (str).

        This is a thin compatibility shim for code paths that use the old
        ``lights_to_sources()`` API.  Physics gaps apply to every source:
        no spectral distribution, no phase, no polarization, directivity is
        reinterpreted as cosine^n.
        """
        if not lights:
            raise ValueError(
                "RayOrder.from_lights_list: empty lights list."
            )
        wl = np.asarray(wavelengths_um, dtype=np.float64)
        flat_weights = np.ones(len(wl), dtype=np.float64)

        sources: List[SourceRecord] = []
        for lgt in lights:
            pos  = np.asarray(lgt.get("pos",   [0., 0., 5.0]),  np.float64)
            lbl  = str(lgt.get("label", "light"))

            if aim_point is not None:
                d = np.asarray(aim_point, np.float64) - pos
            else:
                d = np.asarray(lgt.get("dir", [0., 0., -1.0]), np.float64)
            nl = np.linalg.norm(d)
            dirv = d / nl if nl > 1e-12 else np.array([0., 0., -1.], np.float64)

            raw_power = float(lgt.get("power", 1.0))
            directivity = max(0.5, raw_power)   # match legacy clamping explicitly

            gaps: List[TracerGap] = [
                TracerGap(
                    source_label  = lbl,
                    property_name = "spectral_distribution",
                    severity      = "dropped",
                    detail        = (
                        'Legacy lights dict "color" key is a display tint, not a '
                        "spectral distribution. All bands weighted equally."
                    ),
                ),
                TracerGap(
                    source_label  = lbl,
                    property_name = "phase",
                    severity      = "dropped",
                    detail        = "Legacy lights dict has no phase information.",
                ),
                TracerGap(
                    source_label  = lbl,
                    property_name = "polarization",
                    severity      = "dropped",
                    detail        = "Legacy lights dict has no polarization information.",
                ),
            ]

            sources.append(SourceRecord(
                pos               = pos,
                dir               = dirv,
                directivity       = directivity,
                amplitude_weights = flat_weights.copy(),
                phase_state       = PhaseState(),
                polarization      = PolarizationState(),
                directional_model = DirectionalModel.LAMBERTIAN,
                profile           = EmitterProfile(),
                spec_label        = lbl,
                component_label   = lbl,
                tracer_gaps       = gaps,
            ))

        return cls(
            sources           = sources,
            wavelengths_um    = wl,
            n_rays_per_source = n_rays_per_source,
            max_bounces       = max_bounces,
            min_amplitude     = min_amplitude,
            seed              = seed,
        )

    # ── Pre-baked ray batch ───────────────────────────────────────────────

    def bake_rays(self, n_rays: int, seed: int = 0) -> np.ndarray:
        """Expand sources into a flat pre-baked ray array ready for GPU SSBO.

        Each row is a complete ray sampled CPU-side from the source's
        directional model and phase_state — the GPU PASS_FORWARD reads the
        row directly and traces it without any further stochastic sampling.

        Returns
        -------
        (n_rays, 12) float32  — layout matches SourceRec in the GPU CS:
          [0:3]  pos_xyz        (pos_weight.xyz)
          [3]    amp_weight     (pos_weight.w)
          [4:7]  dir_xyz        (dir_kind.xyz)   — pre-sampled
          [7]    0.0            (dir_kind.w)     — unused (model_int removed)
          [8]    freq_hz        (packet.x)       — c / wavelength_um
          [9]    phase_rad      (packet.y)       — pre-sampled from phase_state
          [10]   energy         (packet.z)
          [11]   0.0            (packet.w)       — unused (model_param removed)

        Phase sampling per CoherenceModel
        ----------------------------------
        INCOHERENT       : uniform [0, 2π) per ray — independent random phase
        COHERENT         : phase_offset_rad (deterministic, same for every ray)
        PARTIAL          : Gaussian(phase_offset_rad, σ = π·λ_um/L_c_um) per ray
        SPATIAL_COHERENT : uniform [0, 2π) per ray (spatially random)
        """
        _C_UM_S = 2.998e14          # speed of light in µm / s
        _TWO_PI = 2.0 * math.pi

        rng = np.random.default_rng(seed)
        n_src = len(self.sources)
        rays_per_src = max(1, n_rays // n_src)
        rows: list = []

        for sr in self.sources:
            phase  = sr.phase_state
            prof   = sr.profile

            # ── Phase sigma ───────────────────────────────────────────────
            if phase.model == CoherenceModel.COHERENT:
                phase_sigma = 0.0
            elif phase.model == CoherenceModel.PARTIAL:
                amp = prof.spectral.sample_array(self.wavelengths_um).astype(np.float64)
                amp_s = amp.sum()
                if amp_s < 1e-30:
                    wl_mean = float(np.mean(self.wavelengths_um))
                else:
                    wl_mean = float(np.dot(self.wavelengths_um, amp / amp_s))
                l_c = max(float(phase.coherence_length_um), 1e-3)
                phase_sigma = math.pi * max(wl_mean, 0.3) / l_c
            else:
                # INCOHERENT or SPATIAL_COHERENT: full randomisation
                phase_sigma = math.pi

            phase_offset = float(phase.phase_offset_rad)
            amp_weight   = float(np.sum(sr.amplitude_weights))

            # ── Spectral weights for wavelength sampling ──────────────────
            amp_spec = prof.spectral.sample_array(self.wavelengths_um).astype(np.float64)
            amp_spec_sum = amp_spec.sum()
            if amp_spec_sum < 1e-30:
                amp_spec = np.ones(len(self.wavelengths_um), np.float64)
                amp_spec /= amp_spec.sum()
            else:
                amp_spec = amp_spec / amp_spec_sum

            # ── Direction frame ───────────────────────────────────────────
            axis = np.asarray(sr.dir, np.float64)
            nl = np.linalg.norm(axis)
            axis = axis / nl if nl > 1e-12 else np.array([0., 0., -1.], np.float64)
            ref = np.array([0., 0., 1.], np.float64)
            if abs(np.dot(axis, ref)) > 0.9:
                ref = np.array([1., 0., 0.], np.float64)
            tx = np.cross(ref, axis); tx /= np.linalg.norm(tx) + 1e-30
            ty = np.cross(axis, tx)

            model = sr.directional_model
            ang   = prof.directional

            for _ri in range(rays_per_src):
                u1, u2 = float(rng.random()), float(rng.random())

                # ── Direction sampling from model ─────────────────────────
                if model == DirectionalModel.LAMBERTIAN:
                    cos_t = math.sqrt(u1)
                    sin_t = math.sqrt(max(0.0, 1.0 - u1))
                    phi   = _TWO_PI * u2
                    rd = (sin_t * math.cos(phi) * tx
                          + sin_t * math.sin(phi) * ty
                          + cos_t * axis)

                elif model == DirectionalModel.GAUSSIAN_BEAM:
                    theta_d = max(float(ang.effective_divergence_rad), 1e-6)
                    r_g = abs(float(rng.standard_normal())) * theta_d * 0.4247
                    r_g = min(r_g, math.pi * 0.5)
                    sin_t, cos_t = math.sin(r_g), math.cos(r_g)
                    phi = _TWO_PI * u2
                    rd = (sin_t * math.cos(phi) * tx
                          + sin_t * math.sin(phi) * ty
                          + cos_t * axis)

                elif model in (DirectionalModel.ETENDUE_LIMITED,
                               DirectionalModel.PROJECTIVE):
                    na = min(max(float(ang.numerical_aperture), 1e-4), 1.0 - 1e-9)
                    cos_max = math.cos(math.asin(na))
                    cos_t = cos_max + (1.0 - cos_max) * u1
                    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
                    phi   = _TWO_PI * u2
                    rd = (sin_t * math.cos(phi) * tx
                          + sin_t * math.sin(phi) * ty
                          + cos_t * axis)

                elif model == DirectionalModel.DIPOLE_INPLANE_MIXED:
                    # Rejection-sample dipole sin²θ pattern
                    cosT = 0.0
                    sinT2 = 1.0
                    for _rej in range(32):
                        cosT = rng.uniform(-1.0, 1.0)
                        sinT2 = max(0.0, 1.0 - cosT * cosT)
                        if rng.random() < sinT2:
                            break
                    sin_t = math.sqrt(float(sinT2))
                    phi   = _TWO_PI * u2
                    rd = (sin_t * math.cos(phi) * tx
                          + sin_t * math.sin(phi) * ty
                          + float(cosT) * axis)

                elif model == DirectionalModel.HENYEY_GREENSTEIN:
                    g = float(ang.hg_g)
                    if abs(g) < 1e-3:
                        cos_t = 2.0 * u1 - 1.0
                    else:
                        s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1)
                        cos_t = min(1.0, max(-1.0,
                                             (1.0 + g * g - s * s) / (2.0 * g)))
                    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
                    phi   = _TWO_PI * u2
                    rd = (sin_t * math.cos(phi) * tx
                          + sin_t * math.sin(phi) * ty
                          + cos_t * axis)

                else:
                    # ISOTROPIC / MEASURED / fallback: full-sphere uniform
                    cos_t = 2.0 * u1 - 1.0
                    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
                    phi   = _TWO_PI * u2
                    rd = (sin_t * math.cos(phi) * tx
                          + sin_t * math.sin(phi) * ty
                          + cos_t * axis)

                rd_norm = np.linalg.norm(rd)
                rd = rd / rd_norm if rd_norm > 1e-12 else axis.copy()

                # ── Phase ─────────────────────────────────────────────────
                if phase.model == CoherenceModel.COHERENT:
                    ph = phase_offset
                else:
                    ph = phase_offset + float(rng.standard_normal()) * phase_sigma

                # ── Wavelength → optical frequency ────────────────────────
                wl_idx  = int(rng.choice(len(self.wavelengths_um), p=amp_spec))
                wl_um   = float(self.wavelengths_um[wl_idx])
                freq_hz = _C_UM_S / max(wl_um, 1e-6)

                rows.append([
                    float(sr.pos[0]), float(sr.pos[1]), float(sr.pos[2]),
                    amp_weight,
                    float(rd[0]), float(rd[1]), float(rd[2]),
                    0.0,        # dir_kind.w — unused
                    freq_hz,    # packet.x — optical frequency
                    ph,         # packet.y — pre-baked phase
                    1.0,        # packet.z — energy
                    0.0,        # packet.w — unused
                ])

        if not rows:
            return np.zeros((1, 12), np.float32)
        return np.array(rows, np.float32)

    def bake_backward_sensor_rays(self,
                                  sensor_bundles: Sequence[dict],
                                  *,
                                  seed: int = 0,
                                  default_freq_hz: float = 5.405e14,
                                  default_energy: float = 1.0,
                                  default_phase: float = 0.0,
                                  sensor_weight: float = 1.0) -> np.ndarray:
        """Pack backward sensor rays from one or more sensor bundles.

        Parameters
        ----------
        sensor_bundles
            Iterable of dicts each containing at least:
              origins : (N,3) float64
              dirs    : (N,3) float64
            Optional keys:
              sensor_id : int
              weights   : (N,) float64

        Returns
        -------
        (N_total, 12) float32 in SourceRec layout.
        """
        rng = np.random.default_rng(int(seed) ^ 0xA5A5A5A5)
        rows: list[list[float]] = []

        for s_idx, bundle in enumerate(sensor_bundles):
            origins = np.asarray(bundle.get("origins", np.zeros((0, 3))), np.float64)
            dirs = np.asarray(bundle.get("dirs", np.zeros((0, 3))), np.float64)
            if origins.shape[0] == 0 or dirs.shape[0] == 0:
                continue
            n = min(int(origins.shape[0]), int(dirs.shape[0]))
            sid = int(bundle.get("sensor_id", s_idx))
            w = np.asarray(bundle.get("weights", np.ones(n, np.float64)), np.float64).ravel()
            if w.size < n:
                w = np.pad(w, (0, n - w.size), mode="edge")

            for i in range(n):
                o = origins[i]
                d = dirs[i]
                dn = float(np.linalg.norm(d))
                if dn < 1e-12:
                    d = np.array([0.0, 0.0, 1.0], np.float64)
                else:
                    d = d / dn
                amp_w = float(max(0.0, sensor_weight * w[i]))
                ph = float(default_phase + (2.0 * math.pi * rng.random()))
                rows.append([
                    float(o[0]), float(o[1]), float(o[2]),
                    amp_w,
                    float(d[0]), float(d[1]), float(d[2]),
                    1.0,                         # dir_kind.w role = backward sensor ray
                    float(default_freq_hz),
                    ph,
                    float(default_energy),
                    float(sid),                  # packet.w = sensor group id
                ])

        if not rows:
            return np.zeros((0, 12), np.float32)
        return np.asarray(rows, np.float32)

    def bake_bidirectional_package(self,
                                   n_forward_rays: int,
                                   sensor_bundles: Sequence[dict],
                                   *,
                                   seed: int = 0,
                                   default_freq_hz: float = 5.405e14,
                                   default_energy: float = 1.0,
                                   default_phase: float = 0.0,
                                   sensor_weight: float = 1.0) -> BidirectionalRayPackage:
        """Build forward + backward rows for multi-emitter/multi-sensor tracing."""
        fwd = self.bake_rays(max(1, int(n_forward_rays)), seed=seed)
        if fwd.shape[0] > 0:
            fwd = np.ascontiguousarray(fwd, np.float32)
            fwd[:, 7] = 0.0   # dir_kind.w role = forward source ray
            fwd[:, 11] = 0.0  # packet.w reserved for source-group metadata

        bwd = self.bake_backward_sensor_rays(
            sensor_bundles,
            seed=seed,
            default_freq_hz=default_freq_hz,
            default_energy=default_energy,
            default_phase=default_phase,
            sensor_weight=sensor_weight,
        )

        if bwd.shape[0] > 0:
            combined = np.concatenate([fwd, bwd], axis=0).astype(np.float32, copy=False)
        else:
            combined = np.asarray(fwd, np.float32)
        return BidirectionalRayPackage(
            forward_rows=np.ascontiguousarray(fwd, np.float32),
            backward_rows=np.ascontiguousarray(bwd, np.float32),
            combined_rows=np.ascontiguousarray(combined, np.float32),
        )
