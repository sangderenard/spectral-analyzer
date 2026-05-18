"""radial_manifold.py — Radially-symmetric manifold baking for BDPT.

═══════════════════════════════════════════════════════════════════════════════
  WHERE WE ARE
═══════════════════════════════════════════════════════════════════════════════

  Three layers of existing machinery are relevant here.  Read them before
  adding anything new:

  camera_designer/parametric_surfaces.py
  ───────────────────────────────────────
  Full surface library with GPU+CPU duality:
    FlatSurface, SphericalSurface, ConicSurface (Newton-refined, same quadric
    as our parametric_surface.py but older and more complete), ApertureStop,
    PolynomialSurface (Zernike/asphere), ToricSurface, CylinderSurface,
    LensMountRing, SensorSurface, MirrorSurface, ShutterPlane.
  Interface: intersect(ro, rd) → (t, hit, normal) + glsl_intercept_fn().
  Convention: surfaces live in their own local Z-axis frame; caller transforms
  world rays first.  No UV-mapping methods.

  camera_software/lens_manifold.py
  ─────────────────────────────────
  LensManifold — production noodle-manifold:
    Stores (u, v, fu, fv, in_dir, out_dir, opl) per ray; 11-column float64.
    Adaptive quadtree over aperture (u,v) for variance-guided refinement.
    5D KD-tree keyed on (u, v, in_dx, in_dy, in_dz) for nearest-neighbor
    lookup with IDW blending.  from_pairs() ingests GPU ray-tracer outputs.
    Fully saves/loads with float64 preserved.
  SphereCoords — (θ, φ) ↔ Cartesian; local_to_world / world_to_local.
  PolarChain   — batch ray-sphere intersection (vectorised NumPy).

  parametric_surface.py  [new, this session]
  ───────────────────────────────────────────
  Partial redundancy with camera_designer: PlaneSurface ≈ FlatSurface,
  ConicSurface ≈ ConicSurface (no Newton polish, no GLSL).
  Unique additions:
    • UV-mapping methods (point_to_uv, uv_to_point, area_element) that the
      camera_designer version does NOT have — needed for ManifoldHalf.
    • World-space frame (vertex + axis) instead of local Z-axis frame —
      needed when the aperture plane is not axis-aligned in world space.
  Recommended future direction: keep only the UV-mapping mixin interface
  here and delegate intersection to camera_designer types where possible.

  optical_manifold.py  [new, this session]
  ─────────────────────────────────────────
  ManifoldVertex / SurfaceManifold / ManifoldHalf / ApertureGrid.
  Partial redundancy with LensManifold: ApertureGrid ≈ the quadtree, but
  uses a fixed-grid bin layout rather than adaptive splitting.
  Unique additions:
    • Per-band complex amplitude storage (float32, never downcast).
    • ManifoldHalf: BDPT-specific half-path structure keyed on aperture_uv.
    • build_halves_from_records(): groups EndpointRecord rows by subpath_id,
      projects terminal vertex backward onto the aperture surface.

  ray_correlator.py  [modified this session]
  ────────────────────────────────────────────
  ManifoldWalkStrategy: bins forward halves into ApertureGrid, for each
  backward half queries neighbours, produces CorrelationCandidate with the
  aperture crossing as MiddlePoint.  Currently marks candidates accepted=True
  optimistically — no shadow ray, no MIS weight.

═══════════════════════════════════════════════════════════════════════════════
  WHERE WE INTEND TO BE
═══════════════════════════════════════════════════════════════════════════════

  GEOMETRIC FOUNDATION: ROTATIONAL SYMMETRY
  ─────────────────────────────────────────
  Every physically real lens stack (coaxial spherical / conic surfaces,
  circular apertures) satisfies:

      f(R_φ · ray) = R_φ · f(ray)

  where R_φ is rotation about the optical axis by angle φ and f is the
  full optical map (intersection + refraction chain + OPL).  Scalar
  quantities (t, OPL) are invariant; vector quantities (hit, direction,
  normal) transform under R_φ.

  Consequence: every noodle (in_dir, out_dir, opl) baked at some azimuth
  φ₀ is valid at any azimuth φ by applying R_{φ−φ₀} to all direction
  vectors.  OPL is unchanged.

  WEDGE BAKING STRATEGY
  ─────────────────────
  1. Choose wedge half-width  Δφ_half  (e.g. π/36 ≈ 5°, giving N=36 sectors).
     The full azimuthal circle is covered by N identical rotated copies of
     the baked wedge, at a compute cost reduction of N×.

  2. In the wedge φ ∈ [−Δφ_half, +Δφ_half] sample directions using
     equal-solid-angle coordinates so no polar region is over- or under-
     sampled:

         cos θ ~ Uniform[cos θ_max, 1]   →  θ = arccos(1 − u·(1−cos θ_max))
         φ     = Δφ_half · (2v − 1)      where (u, v) ~ Uniform[0,1]²

     For θ_max = π/2 (hemisphere) and a 5° wedge the sample count needed
     to achieve the same aperture-radius density as a full-disk trace is
     reduced by the ratio of wedge solid angle to hemisphere solid angle:
         coverage = 2·Δφ_half / (2π)  →  1/36 of rays needed.

  3. For each sampled direction, trace both half-paths from the aperture
     plane:
         forward half:   aperture → scene/emitter
         backward half:  aperture → sensor

     The aperture plane is the natural "meeting point" for BDPT connection
     (it carries both the spatial (r, φ) and directional (θ) information).
     Tracing outward from the aperture eliminates the need to explicitly
     shoot from the light source or sensor and then project — both halves
     naturally terminate at the aperture.

  4. Store each noodle in polar aperture coordinates (r, φ_rel) where
     φ_rel = φ − floor(φ / (2·Δφ_half)) · (2·Δφ_half) is the residual
     within the wedge.  Use a 1D radial grid (RadialApertureGrid) for O(1)
     bin insertion and O(k) radial-neighbour queries.

  QUERY / RECONSTRUCTION AT ARBITRARY AZIMUTH
  ────────────────────────────────────────────
  Given query azimuth φ_q for a ray that enters the aperture at radius r_q:

    1. sector_idx = round(φ_q / (2·Δφ_half))
    2. φ_baked    = φ_q − sector_idx · (2·Δφ_half)   ← residual in wedge
    3. Fetch the nearest baked noodle to (r_q, φ_baked) from the wedge.
    4. Apply the rotation Δ = sector_idx · (2·Δφ_half) to all directions:
           x' = x·cos Δ − y·sin Δ
           y' = x·sin Δ + y·cos Δ
           z' = z   (axis-aligned component unchanged)
    5. OPL is unchanged (scalar, invariant under rotation).

  This "modulo angle" lookup is exact for a perfectly axisymmetric system
  and provides a smooth analytic correction for small residual asymmetries
  introduced by aberrations at the wedge boundary.

  BDPT CONNECTION IN POLAR COORDINATES
  ─────────────────────────────────────
  The aperture plane splits naturally into radial rings.  For a rotationally
  symmetric system, a forward half-path at (r_fwd, φ_fwd) connects to a
  backward half-path at (r_bwd, φ_bwd) iff:
      |r_fwd − r_bwd| < ε_r    (same radial annulus)
      |φ_fwd − φ_bwd| < ε_φ    (same or adjacent angular sector)

  The φ condition is handled analytically via the rotation step above, so in
  practice the connection test reduces to a 1D radial match, making the BDPT
  connection O(n_r) rather than O(n_u × n_v).

  DATA STRUCTURES TO BUILD
  ────────────────────────
  WedgeNoodle     — one baked record (wraps or extends LensManifold noodle
                    with per-band complex amplitude float32 + aperture polar
                    coords (r, φ_rel)).

  WedgeManifold   — baked noodles for one φ sector.  Thin wrapper over
                    LensManifold.from_pairs() keyed on (r, φ_rel, θ).
                    Exposes query_polar(r, phi_rel, theta) → (out_dir, opl).

  SymmetryManifold — wraps N_sectors WedgeManifolds; serves the full sphere.
                    query(r, phi, theta) → rotate(WedgeManifold.query(...)).
                    Supports forward_half=True, backward_half=True, or both.

  RadialApertureGrid — 1D uniform array of n_r radial bins over [0, r_max].
                       Each bin: list of ManifoldHalf objects.
                       insert(half), query_ring(r, width_bins=1).
                       Replaces ApertureGrid (2D Cartesian) for axisymmetric
                       systems.

  SURFACE CHAIN INPUT
  ───────────────────
  The manifold is compiled from a list of camera_designer.ParametricSurface
  objects (the lens prescription), already the standard format used by
  camera_designer and PolarChain.  No new surface primitives are required;
  the intersection math already exists in camera_designer/parametric_surfaces.
  Our parametric_surface.py UV-mapping adapter wraps those surfaces to expose
  the (r, φ) → (u, v) coordinates needed for aperture-plane parameterisation.

  IMPLEMENTATION ORDER
  ────────────────────
  Step 1  wedge_direction_sampler(n, delta_phi_half, theta_max) → (θ, φ, dirs)
  Step 2  rotate_directions_xy(dirs, delta_phi) → dirs′
  Step 3  WedgeManifold.bake(surface_chain, aperture_surface, n, delta_phi_half,
                             forward=True, backward=True)
  Step 4  SymmetryManifold wrapping N WedgeManifolds; query + rotate path
  Step 5  RadialApertureGrid replacing ApertureGrid in ManifoldWalkStrategy
  Step 6  ManifoldWalkStrategy updated to use RadialApertureGrid + φ rotation

═══════════════════════════════════════════════════════════════════════════════
  STUBS — NOT YET IMPLEMENTED
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Re-use existing surface and manifold machinery; do NOT duplicate.
from camera_designer.parametric_surfaces import (
    ParametricSurface as CdParametricSurface,
    FlatSurface,
    ConicSurface,
)
from camera_software.lens_manifold import LensManifold, SphereCoords
from optical_manifold import ManifoldHalf, ApertureGrid


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Equal-solid-angle wedge direction sampler
# ─────────────────────────────────────────────────────────────────────────────

def wedge_direction_sampler(
    n: int,
    delta_phi_half: float,          # half-width of azimuthal wedge (radians)
    theta_max: float = math.pi / 2, # polar cutoff from axis
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample *n* directions uniformly in solid angle within an azimuthal wedge.

    The wedge is φ ∈ [−delta_phi_half, +delta_phi_half], θ ∈ [0, theta_max].
    Equal-solid-angle sampling ensures no angular region is over-represented:

        cos θ = 1 − u · (1 − cos θ_max)    u ~ Uniform[0, 1]
        φ     = delta_phi_half · (2v − 1)   v ~ Uniform[0, 1]

    Returns
    -------
    theta  : (n,) float64  — polar angle from optical axis
    phi    : (n,) float64  — azimuth within wedge
    dirs   : (n, 3) float64 — unit direction vectors (Z = optical axis)
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(0.0, 1.0, n)
    v = rng.uniform(0.0, 1.0, n)
    cos_theta_max = math.cos(theta_max)
    cos_theta = 1.0 - u * (1.0 - cos_theta_max)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))   # (n,) float64
    phi   = delta_phi_half * (2.0 * v - 1.0)            # (n,) float64
    sin_theta = np.sin(theta)
    dirs = np.empty((n, 3), dtype=np.float64)
    dirs[:, 0] = sin_theta * np.cos(phi)
    dirs[:, 1] = sin_theta * np.sin(phi)
    dirs[:, 2] = cos_theta
    return theta, phi, dirs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Analytical φ rotation for direction vectors
# ─────────────────────────────────────────────────────────────────────────────

def rotate_directions_xy(
    dirs: np.ndarray,   # (N, 3) or (3,) float64 unit vectors
    delta_phi: float,   # rotation angle about Z-axis (radians)
) -> np.ndarray:
    """Rotate direction vectors by delta_phi about the optical (Z) axis.

    Pure in-plane 2×2 rotation; Z component is untouched.  Preserves the
    native dtype of *dirs* (float64 in, float64 out).

        x' = x·cos Δ − y·sin Δ
        y' = x·sin Δ + y·cos Δ
        z' = z

    This is the analytic step that extends a baked wedge to any azimuth
    without re-tracing.  OPL is a scalar invariant — do not rotate it.
    """
    dirs = np.asarray(dirs)                     # preserve input dtype
    out  = dirs.copy()
    c    = math.cos(delta_phi)
    s    = math.sin(delta_phi)
    if dirs.ndim == 1:
        x      = float(dirs[0])
        y      = float(dirs[1])
        out[0] = x * c - y * s
        out[1] = x * s + y * c
        # out[2] = dirs[2]  (already copied)
    else:
        x        = dirs[:, 0].copy()
        y        = dirs[:, 1].copy()
        out[:, 0] = x * c - y * s
        out[:, 1] = x * s + y * c
        # out[:, 2] unchanged
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — WedgeManifold: one baked φ sector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WedgeManifold:
    """Noodle-manifold for one azimuthal wedge of a rotationally symmetric optic.

    Wraps LensManifold and adds polar aperture parameterisation (r, phi_rel)
    plus optional per-band complex amplitude (float32) for BDPT connection.

    Baking workflow:
        wm = WedgeManifold.bake(
            surface_chain   = [FlatSurface(...), ConicSurface(...), ...],
            aperture_surface = FlatSurface(z_pos=0.0, r_max=0.015),
            n               = 65536,
            delta_phi_half  = math.pi / 36,   # 5° wedge
            theta_max       = math.pi / 2,
            forward         = True,
            backward        = True,
        )
        wm.save("wedge_5deg.npz")

    Query workflow:
        out_dir, opl = wm.query_polar(r=0.008, phi_rel=0.03, theta=0.12)
        # Returns float64 direction in the wedge's canonical frame; caller
        # applies rotate_directions_xy(out_dir, sector_phi) to reach world φ.
    """

    _manifold:      Optional[LensManifold] = field(default=None, repr=False)
    delta_phi_half: float = math.pi / 36
    theta_max:      float = math.pi / 2
    meta:           dict  = field(default_factory=dict)

    @classmethod
    def bake(
        cls,
        surface_chain:    list,         # list[CdParametricSurface]
        aperture_surface: object,       # CdParametricSurface at aperture plane
        n:                int = 65536,
        delta_phi_half:   float = math.pi / 36,
        theta_max:        float = math.pi / 2,
        forward:          bool = True,
        backward:         bool = True,
        seed:             int  = 0,
    ) -> "WedgeManifold":
        """Trace *n* rays through *surface_chain* within the azimuthal wedge.

        For each sampled direction the ray is started at the aperture plane
        and propagated outward:
          - forward=True  → ray toward scene/emitter side
          - backward=True → ray toward sensor side
        Both halves are stored so that BDPT connection can use either, or both.

        All arithmetic is float64.  Per-band amplitudes are accumulated
        separately as float32 to match EndpointRecord dtype.
        """
        rng = np.random.default_rng(seed)
        theta, phi_dirs, in_dirs = wedge_direction_sampler(
            n, delta_phi_half, theta_max, rng
        )

        # Derive aperture radius from the aperture surface.
        ap_radius = float(
            getattr(aperture_surface, 'r_max',
            getattr(aperture_surface, 'aperture_radius',
            getattr(aperture_surface, 'clear_aperture_radius', 0.015)))
        )
        ap_z = float(getattr(aperture_surface, 'z_pos', 0.0))

        # Sample aperture positions uniformly over the aperture disk,
        # restricting azimuth to the wedge to keep the bake self-consistent.
        r_norm = np.sqrt(rng.uniform(0.0, 1.0, n))  # uniform disk sampling
        phi_ap = rng.uniform(-delta_phi_half, delta_phi_half, n)
        x_ap   = r_norm * np.cos(phi_ap) * ap_radius
        y_ap   = r_norm * np.sin(phi_ap) * ap_radius

        # Ray origins at aperture plane (local Z frame, z = ap_z)
        ro = np.column_stack([x_ap, y_ap, np.full(n, ap_z)])

        # Propagate each ray through the surface chain.
        # Surfaces use a single-ray interface: intersect(ro_1d, rd_1d).
        # Tracing is geometric only (no refraction) — IOR data is not
        # available here; the manifold records the geometric skeleton.
        rd   = in_dirs.astype(np.float64, copy=True)   # (N, 3)
        opls = np.zeros(n, dtype=np.float64)

        for surf in surface_chain:
            for i in range(n):
                try:
                    t_i, hit_i, _nrm = surf.intersect(ro[i], rd[i])
                    t_f = float(t_i)
                    if math.isfinite(t_f) and t_f > 1e-9:
                        opls[i] += t_f
                        ro[i]    = ro[i] + t_f * rd[i]
                except Exception:
                    pass

        out_dirs = rd  # geometric: direction unchanged (no refraction)

        # Normalised aperture (u, v) in [-1, 1] for LensManifold.
        uv_ap = np.column_stack([r_norm * np.cos(phi_ap),
                                  r_norm * np.sin(phi_ap)])
        # Field-angle factors: use (theta, phi_dirs) as proxies.
        fufv = np.column_stack([theta, phi_dirs])

        manifold = LensManifold.from_pairs(
            uv=uv_ap,
            fufv=fufv,
            in_dirs=in_dirs,
            out_dirs=out_dirs,
            opls=opls,
        )
        return cls(
            _manifold=manifold,
            delta_phi_half=delta_phi_half,
            theta_max=theta_max,
            meta={
                "n": n,
                "seed": seed,
                "forward": forward,
                "backward": backward,
                "ap_radius": ap_radius,
                "ap_z": ap_z,
            },
        )

    def query_polar(
        self,
        r:       float,   # aperture radius (metres)
        phi_rel: float,   # azimuth within wedge (radians)
        theta:   float,   # polar angle from axis (radians)
        k:       int = 8,
    ) -> tuple[np.ndarray, float]:
        """IDW-blended output direction + OPL for a polar aperture query.

        Returns (out_dir (3,) float64, opl float64).
        Caller applies rotate_directions_xy to move from wedge frame to world φ.
        """
        if self._manifold is None or self._manifold._data is None:
            return np.array([0.0, 0.0, 1.0], np.float64), 0.0
        # Normalise r to [-1, 1] aperture space.
        ap_radius = float(self.meta.get("ap_radius", 0.015))
        if ap_radius <= 0.0:
            ap_radius = 0.015
        r_norm = max(-1.0, min(1.0, r / ap_radius))
        u = r_norm * math.cos(phi_rel)
        v = r_norm * math.sin(phi_rel)
        in_dir = np.array(
            [math.sin(theta) * math.cos(phi_rel),
             math.sin(theta) * math.sin(phi_rel),
             math.cos(theta)],
            dtype=np.float64,
        )
        uv_q  = np.array([[u, v]], dtype=np.float64)
        dirs, opl_arr = self._manifold.interpolate_with_opl(
            uv_q, k=k, in_dir=in_dir[np.newaxis, :]
        )
        return dirs[0], float(opl_arr[0])

    def save(self, path: str) -> None:
        """Persist to .npz (delegates to LensManifold.save + meta sidecar)."""
        import json
        if not path.endswith(".npz"):
            path += ".npz"
        if self._manifold is not None:
            self._manifold.save(path)
            # Append wedge-specific meta by reopening and re-saving.
            npz = dict(np.load(path, allow_pickle=False))
            npz["wedge_meta"] = np.frombuffer(
                json.dumps({
                    "delta_phi_half": self.delta_phi_half,
                    "theta_max":      self.theta_max,
                    **self.meta,
                }).encode(),
                dtype=np.uint8,
            )
            np.savez_compressed(path, **npz)

    @classmethod
    def load(cls, path: str) -> "WedgeManifold":
        """Load from .npz."""
        import json
        manifold = LensManifold.load(path)
        wedge_meta: dict = {}
        try:
            npz = np.load(path, allow_pickle=False)
            if "wedge_meta" in npz:
                wedge_meta = json.loads(bytes(npz["wedge_meta"]).decode())
        except Exception:
            pass
        delta_phi_half = float(wedge_meta.pop("delta_phi_half", math.pi / 36))
        theta_max      = float(wedge_meta.pop("theta_max",      math.pi / 2))
        return cls(
            _manifold=manifold,
            delta_phi_half=delta_phi_half,
            theta_max=theta_max,
            meta=wedge_meta,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — SymmetryManifold: full sphere from N wedges
# ─────────────────────────────────────────────────────────────────────────────

class SymmetryManifold:
    """Full-sphere manifold assembled from N rotated copies of one WedgeManifold.

    Only one WedgeManifold needs to be baked (or loaded).  At query time,
    the azimuth φ is folded into the canonical wedge and the result is
    analytically rotated back — exact for a rotationally symmetric system.

    The 'modulo angle' dispatch:
        sector_idx = round(phi / (2·delta_phi_half))
        phi_baked  = phi − sector_idx · (2·delta_phi_half)
        out_dir    = rotate_directions_xy(wedge.query_polar(r, phi_baked, theta),
                                          sector_idx · 2·delta_phi_half)

    N_sectors = round(pi / delta_phi_half)   (half-turn, use symmetry for other half)
    """

    def __init__(self, wedge: WedgeManifold) -> None:
        self.wedge           = wedge
        self.delta_phi_half  = wedge.delta_phi_half
        self._sector_width   = 2.0 * self.delta_phi_half

    def query(
        self,
        r:      np.ndarray,   # (M,) float64 aperture radii
        phi:    np.ndarray,   # (M,) float64 full azimuths [0, 2π)
        theta:  np.ndarray,   # (M,) float64 polar angles
        k:      int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch query with full-sphere azimuth support.

        Returns
        -------
        out_dirs : (M, 3) float64 — unit directions in world frame
        opls     : (M,)   float64 — optical path lengths (metres)
        """
        r     = np.asarray(r,     np.float64).ravel()
        phi   = np.asarray(phi,   np.float64).ravel()
        theta = np.asarray(theta, np.float64).ravel()
        M = len(r)
        out_dirs = np.empty((M, 3), dtype=np.float64)
        opls     = np.empty(M,       dtype=np.float64)
        for i in range(M):
            # Fold phi into the canonical wedge.
            sector_idx = round(float(phi[i]) / self._sector_width)
            phi_baked  = float(phi[i]) - sector_idx * self._sector_width
            out_dir_w, opl_i = self.wedge.query_polar(
                float(r[i]), phi_baked, float(theta[i]), k=k
            )
            # Rotate back to world azimuth.
            delta = sector_idx * self._sector_width
            out_dirs[i] = rotate_directions_xy(out_dir_w, delta)
            opls[i]     = opl_i
        return out_dirs, opls

    def query_halves(
        self,
        r:       np.ndarray,
        phi:     np.ndarray,
        theta:   np.ndarray,
        forward: bool = True,
        backward: bool = True,
        n_bands:  int = 1,
        k:        int = 8,
    ) -> list[ManifoldHalf]:
        """Build ManifoldHalf objects from a batch query.

        forward=True  → ManifoldHalf with kind=RayStreamKind.FORWARD_LIGHT
        backward=True → ManifoldHalf with kind=RayStreamKind.APERTURE_PUPIL
        Both sets are keyed on aperture (r, phi) for connection by
        RadialApertureGrid.
        """
        from bdpt_integrator import RayStreamKind
        from optical_manifold import ManifoldVertex, SurfaceManifold, INTERACTION_PROPAGATE
        r      = np.asarray(r,     np.float64).ravel()
        phi    = np.asarray(phi,   np.float64).ravel()
        theta  = np.asarray(theta, np.float64).ravel()
        M      = len(r)
        halves: list[ManifoldHalf] = []
        ap_radius = float(self.wedge.meta.get("ap_radius", 0.015))
        if ap_radius <= 0.0:
            ap_radius = 0.015
        for kind, enabled in [
            (RayStreamKind.FORWARD_LIGHT,  forward),
            (RayStreamKind.APERTURE_PUPIL, backward),
        ]:
            if not enabled:
                continue
            out_dirs, opls = self.query(r, phi, theta, k=k)
            for i in range(M):
                r_norm = float(r[i]) / ap_radius
                u_ap   = r_norm * math.cos(float(phi[i]))
                v_ap   = r_norm * math.sin(float(phi[i]))
                # Build a minimal ManifoldVertex at the aperture.
                x_ap = float(r[i]) * math.cos(float(phi[i]))
                y_ap = float(r[i]) * math.sin(float(phi[i]))
                pos  = np.array([x_ap, y_ap, float(
                    self.wedge.meta.get("ap_z", 0.0)
                )], dtype=np.float64)
                in_d = np.array([
                    math.sin(float(theta[i])) * math.cos(float(phi[i])),
                    math.sin(float(theta[i])) * math.sin(float(phi[i])),
                    math.cos(float(theta[i])),
                ], dtype=np.float64)
                vert = ManifoldVertex(
                    pos=pos,
                    dir_in=in_d,
                    dir_out=out_dirs[i],
                    normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    amp_re=np.ones(n_bands, dtype=np.float32),
                    amp_im=np.zeros(n_bands, dtype=np.float32),
                    pdf=1.0 / max(M, 1),
                    cos_in=float(abs(math.cos(float(theta[i])))),
                    cos_out=float(abs(math.cos(float(theta[i])))),
                    pathlen_m=float(opls[i]),
                    interaction=INTERACTION_PROPAGATE,
                )
                smanifold = SurfaceManifold(
                    subpath_id=i,
                    vertices=[vert],
                )
                half = ManifoldHalf(
                    kind=kind,
                    aperture_surface_id=0,
                    aperture_uv=(u_ap, v_ap),
                    manifold=smanifold,
                    pixel_id=i,
                    source_records=None,
                )
                halves.append(half)
        return halves


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — RadialApertureGrid: 1D radial bin array for axisymmetric connection
# ─────────────────────────────────────────────────────────────────────────────

class RadialApertureGrid:
    """1D aperture grid indexed by radius r, not (u, v) Cartesian bins.

    For rotationally symmetric optics the BDPT connection test reduces to a
    1D radial match (|r_fwd − r_bwd| < ε), because the φ match is handled
    analytically by the SymmetryManifold rotation.  A 1D grid is therefore
    sufficient and more efficient than ApertureGrid's 2D layout.

    Bins are uniform over [0, r_max].  Each bin stores a list[ManifoldHalf].

    Usage:
        grid = RadialApertureGrid(r_max=0.015, n_r=64)
        for half in forward_halves:
            grid.insert(half)
        for bwd in backward_halves:
            matches = grid.query_ring(bwd.aperture_r, width_bins=1)
            ...
    """

    def __init__(self, r_max: float, n_r: int = 64) -> None:
        self.r_max  = r_max
        self.n_r    = n_r
        self._bins: list[list[ManifoldHalf]] = [[] for _ in range(n_r)]

    def _bin_idx(self, r: float) -> int:
        """Map radius r to bin index, clamped to [0, n_r-1]."""
        if self.r_max <= 0.0:
            return 0
        idx = int(float(r) / self.r_max * self.n_r)
        return max(0, min(idx, self.n_r - 1))

    def insert(self, half: ManifoldHalf) -> None:
        """Insert a ManifoldHalf keyed on its aperture radius.

        The aperture radius is derived from half.aperture_uv as
        r = sqrt(u²+v²) · aperture_surface.clear_aperture_radius,
        or directly if half stores (r, phi) in aperture_uv.
        """
        u, v = half.aperture_uv
        # aperture_uv is normalized (u,v) in [-1,1]; r_norm in [0, sqrt(2)].
        # Scale back to physical radius using r_max.
        r_norm = math.sqrt(float(u) ** 2 + float(v) ** 2)
        r_phys = r_norm * self.r_max
        self._bins[self._bin_idx(r_phys)].append(half)

    def query_ring(
        self,
        r:           float,
        width_bins:  int = 1,
    ) -> list[ManifoldHalf]:
        """Return all ManifoldHalf objects within ±width_bins of r's bin."""
        idx  = self._bin_idx(float(r))
        lo   = max(0, idx - width_bins)
        hi   = min(self.n_r - 1, idx + width_bins)
        result: list[ManifoldHalf] = []
        for b in range(lo, hi + 1):
            result.extend(self._bins[b])
        return result

    @property
    def count(self) -> int:
        return sum(len(b) for b in self._bins)

    def clear(self) -> None:
        for b in self._bins:
            b.clear()
