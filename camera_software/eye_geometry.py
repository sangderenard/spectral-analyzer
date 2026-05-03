"""camera_software/eye_geometry.py
-----------------------------------
Geometric human-eye camera model.

Defines the eye as a layered optical system (cornea → aqueous → crystalline
lens → vitreous → hemispherical retina) with a foveally-weighted photoreceptor
density distribution.  The model bakes into a ``LensManifold`` that speaks
exactly the same (u, v, fu, fv, in_dir, out_dir) noodle language, ready for
broadcasted tensor lookups.

Physical model  (Gullstrand-Le Grand simplified schematic eye)
--------------------------------------------------------------
Surface  Location  R_curv   n_before  n_after   Name
───────  ────────  ───────  ────────  ───────   ──────────────────────
  0       z=0        7.8 mm   1.000    1.336     Cornea (single thin)
  1       z=3.6 mm  10.2 mm   1.336    1.413     Crystalline lens (ant.)
  2       z=7.2 mm  -6.0 mm   1.413    1.336     Crystalline lens (post.)

Retina: hemisphere, R=10.5 mm, centre at z=13.5 mm → fovea at z=24 mm.
Pupil: circular iris aperture at z=3.6 mm, radius 1.5–4.0 mm.

Coordinate system
-----------------
+Z   : optical axis pointing into the scene (from retina outward)
+X,Y : sensor-plane tangent vectors
Origin: corneal vertex

Scene directions are expressed as (fu, fv) = (tan(az), tan(el)) matching the
GPU shader convention, so a ``LensManifold`` baked from this model can be
queried by SensorAccumulator with no coordinate conversion.

Retinal parameterisation
------------------------
u_ret, v_ret = (az_ret / max_az, el_ret / max_az) — normalised to [-1, 1]
where (az_ret, el_ret) are the angles from the optical axis to the retinal
hit point, and max_az ≈ π/2 (90° field boundary).

These are stored in the (fu, fv) noodle columns so that foveal hits cluster
near (fu=0, fv=0).

All arithmetic is float64 throughout; no dtype coercion is ever applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .lens_manifold import (
    LensManifold, PolarChain, SphereCoords, CylinderCoords,
    _NCOLS, _C_U, _C_V, _C_FU, _C_FV, _C_IN, _C_OUT,
)

__all__ = ["EyeGeometry", "HUMAN_EYE", "bake_eye_manifold", "RetinalSensor"]

# ---------------------------------------------------------------------------
# Geometry dataclass
# ---------------------------------------------------------------------------

@dataclass
class EyeGeometry:
    """Full geometric description of a schematic eye.

    All distances in metres.  Positive radii curve away from incoming light
    (convex toward scene); negative radii curve toward it (concave).

    Surfaces are ordered front-to-back along the optical axis (+Z into scene).
    The pupil is treated as a thin circular stop at ``pupil_z``.
    """

    # ── Refractive indices ───────────────────────────────────────────────────
    n_air:      float = 1.0000   # outside the eye
    n_aqueous:  float = 1.3360   # aqueous and vitreous humour
    n_lens:     float = 1.4130   # crystalline lens (simplified uniform)

    # ── Surface geometry (z positions on optical axis, metres) ──────────────
    # Cornea (simplified: single refracting surface)
    cornea_z:        float = 0.000
    cornea_r_curv:   float = 7.8e-3     # +7.8 mm — convex toward scene

    # Crystalline lens anterior surface
    lens_ant_z:      float = 3.6e-3
    lens_ant_r_curv: float = 10.2e-3    # +10.2 mm — convex toward scene

    # Crystalline lens posterior surface
    lens_post_z:     float = 7.2e-3
    lens_post_r_curv: float = -6.0e-3   # -6.0 mm — convex toward retina

    # ── Pupil (iris aperture stop) ───────────────────────────────────────────
    pupil_z:      float = 3.6e-3        # at the anterior lens surface
    pupil_r_min:  float = 1.5e-3        # fully constricted (mm → m)
    pupil_r_max:  float = 4.0e-3        # fully dilated
    pupil_r:      float = 3.0e-3        # current radius (metres)

    # ── Retinal hemisphere ───────────────────────────────────────────────────
    retina_centre_z: float = 13.5e-3    # centre of retinal sphere from cornea
    retina_r:        float = 10.5e-3    # radius of the retinal sphere
    # fovea is at z = retina_centre_z + retina_r ≈ 24 mm

    # ── Optical axis ─────────────────────────────────────────────────────────
    optical_axis: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 1.], np.float64)
    )

    # ── Foveal sampling parameters ───────────────────────────────────────────
    foveal_sigma:           float = 0.26    # radians ≈ 15° — 1σ of foveal zone
    peripheral_fraction:    float = 0.35    # fraction of noodles in periphery
    max_field_angle:        float = math.pi / 2.0   # 90° half-angle = full retina

    @property
    def fovea_z(self) -> float:
        return self.retina_centre_z + self.retina_r

    @property
    def vitreous_depth(self) -> float:
        return self.retina_centre_z - self.lens_post_z

    def as_polar_chain(self) -> PolarChain:
        """Build a ``PolarChain`` representing the three refracting surfaces."""
        ax = self.optical_axis
        return PolarChain(
            n_media = [self.n_air,
                       self.n_aqueous,
                       self.n_lens,
                       self.n_aqueous],         # vitreous ≈ aqueous
            normals = [ax, ax, ax],
            origins = [
                ax * self.cornea_z,
                ax * self.lens_ant_z,
                ax * self.lens_post_z,
            ],
            radii   = [
                self.cornea_r_curv,
                self.lens_ant_r_curv,
                self.lens_post_r_curv,
            ],
        )

    def retina_intersect(self, ro: np.ndarray,
                         rd: np.ndarray) -> tuple:
        """Intersect rays with the retinal hemisphere.

        Parameters
        ----------
        ro : (N, 3) float64 — ray origins (after exiting the lens)
        rd : (N, 3) float64 — ray directions (unit vectors)

        Returns
        -------
        hit_pts  : (N, 3) float64 — retinal hit positions
        hit_dirs : (N, 3) float64 — ray directions at hit  (= rd, unchanged)
        valid    : (N,)   bool    — False if ray missed the retinal sphere
        az_ret   : (N,)   float64 — azimuth of hit point relative to optical axis
        el_ret   : (N,)   float64 — elevation of hit point relative to optical axis
        """
        ax     = self.optical_axis
        centre = ax * self.retina_centre_z             # (3,) float64
        R      = self.retina_r

        # Ray-sphere intersection (keep the FARTHER positive t — back of sphere)
        oc   = ro - centre[None, :]
        b    = 2.0 * np.einsum('ni,ni->n', oc, rd)
        c    = np.einsum('ni,ni->n', oc, oc) - R ** 2
        disc = b ** 2 - 4.0 * c
        eps  = 1e-9
        valid = disc >= 0.0
        sd    = np.where(valid, np.sqrt(np.maximum(disc, 0.0)), 0.0)
        t1    = (-b - sd) * 0.5
        t2    = (-b + sd) * 0.5
        # Prefer the intersection deeper into the eye (farther t > 0)
        t     = np.where(valid & (t2 > eps), t2,
                np.where(valid & (t1 > eps), t1, np.nan))
        valid &= ~np.isnan(t)
        t      = np.where(valid, t, 0.0)

        hit_pts = ro + t[:, None] * rd

        # Angular position of retinal hit relative to optical axis
        dp   = hit_pts - centre[None, :]             # vector from sphere centre
        # Project onto optical axis and perp plane
        z_ax = np.einsum('ni,i->n', dp, ax)          # along axis
        perp = dp - z_ax[:, None] * ax[None, :]
        px   = perp[:, 0]
        py   = perp[:, 1]
        az_ret = np.arctan2(px, np.maximum(z_ax, 1e-15))    # azimuth ≈ x-angle
        el_ret = np.arctan2(py, np.maximum(z_ax, 1e-15))    # elevation ≈ y-angle

        return hit_pts, rd.copy(), valid, az_ret, el_ret

    # ------------------------------------------------------------------
    # Camera back factory
    # ------------------------------------------------------------------

    def make_back(self, *,
                  res_w: int = 512, res_h: int = 512,
                  n_channels: int = 4,
                  cortical_magnification: bool = True) -> "SphericalBack":
        """Create a :class:`SphericalBack` whose geometry matches this eye's retina.

        The back is ready to receive ``accumulate()`` calls with ray directions
        in eye-local space (pointing away from the cornea, toward the retina).

        Parameters
        ----------
        res_w, res_h            : pixel resolution of the frame buffer
        n_channels              : RGBA=4, RGB=3, mono=1
        cortical_magnification  : if True, apply log-polar CMF UV warp
        """
        from .camera_back import SphericalBack
        return SphericalBack(
            res_w=res_w, res_h=res_h, n_channels=n_channels,
            radius=self.retina_r,
            max_field_angle=self.max_field_angle,
            cortical_magnification=cortical_magnification,
            cmf_k=0.065, cmf_e=15.0,
        )

    def make_manifold_back(self, manifold, *,
                           res_w: int = 512, res_h: int = 512,
                           n_channels: int = 4,
                           cortical_magnification: bool = True,
                           k: int = 8) -> "ManifoldBack":
        """Bind a baked :class:`LensManifold` to the retinal back.

        The returned :class:`ManifoldBack` routes scene-space rays through
        *manifold* and projects the output directions onto this eye's retinal
        hemisphere.

        Parameters
        ----------
        manifold                : LensManifold — baked from bake_eye_manifold()
        res_w, res_h, n_channels: passed through to the underlying SphericalBack
        cortical_magnification  : as in make_back()
        k                       : KDTree neighbour count for manifold query
        """
        from .camera_back import ManifoldBack
        spherical = self.make_back(
            res_w=res_w, res_h=res_h, n_channels=n_channels,
            cortical_magnification=cortical_magnification)
        return ManifoldBack(manifold, spherical, k=k)

    # ------------------------------------------------------------------
    # Mesh builder
    # ------------------------------------------------------------------

    def build_mesh(self, subdivisions: int = 32) -> dict:
        """Build renderable pos+norm meshes for all eye surfaces.

        Returns a ``dict`` mapping surface name →
        ``(verts (V, 6) float32 pos+norm, tris (T, 3) int32)``.

        Surfaces
        --------
        ``sclera``     — outer globe
        ``cornea_cap`` — anterior corneal surface
        ``lens_ant``   — crystalline lens anterior
        ``lens_post``  — crystalline lens posterior
        ``retina``     — inner retinal hemisphere (normals toward lens)
        ``pupil``      — iris aperture disk
        """
        from .camera_back import build_eye_mesh
        return build_eye_mesh(self, subdivisions=subdivisions)


# ---------------------------------------------------------------------------
# Default human-eye preset
# ---------------------------------------------------------------------------

HUMAN_EYE = EyeGeometry()   # all defaults — standard relaxed emmetropic eye


# ---------------------------------------------------------------------------
# Manifold baking
# ---------------------------------------------------------------------------

def bake_eye_manifold(
    eye:        EyeGeometry     = HUMAN_EYE,
    *,
    n_base:     int             = 65_536,
    n_refine:   int             = 4,
    threshold:  float           = 5e-6,
    seed:       int             = 0,
) -> LensManifold:
    """Bake a ``LensManifold`` for the eye defined by *eye*.

    Aperture sampling
    -----------------
    Rays enter through the pupil disk (CylinderCoords uniform disk sample),
    normalised to (u, v) ∈ [-1, 1] by the current pupil radius.

    Scene sampling
    --------------
    Scene directions are drawn with ``SphereCoords.foveal_sample``, which
    concentrates rays near the optical axis (foveal zone) while maintaining
    peripheral coverage.  The resulting (fu, fv) values are ``tan`` of the
    angular deviations, matching ``uFovTan`` in the GPU shader.

    Retinal parameterisation
    ------------------------
    After tracing through the three refracting surfaces and intersecting the
    retina, the retinal hit angles (az_ret, el_ret) are stored as the
    *output-side* (fu, fv) columns — so querying the manifold by input scene
    direction returns the retinal location the signal lands on.

    Returns
    -------
    LensManifold with noodles:
        u, v    — normalised pupil position
        fu, fv  — input scene tan-angles  (tan(az_scene), tan(el_scene))
        in_dir  — scene ray direction (unit vector, pointing into the eye)
        out_dir — ray direction at retina hit (unit vector)
    """
    rng = np.random.default_rng(seed)
    ax  = eye.optical_axis                        # (3,) float64

    chain      = eye.as_polar_chain()
    pupil_z    = eye.pupil_z
    pupil_r    = eye.pupil_r
    max_ang    = eye.max_field_angle

    def _bake_chunk(N: int) -> Optional[np.ndarray]:
        # ── Aperture: uniform disk at pupil plane ────────────────────────────
        xy_ap  = CylinderCoords.disk_sample(N, pupil_r, rng)          # (N, 2)
        ap_pts = np.zeros((N, 3), np.float64)
        ap_pts[:, 0] = xy_ap[:, 0]
        ap_pts[:, 1] = xy_ap[:, 1]
        ap_pts[:, 2] = pupil_z

        # Normalised (u, v) in [-1, 1]
        u = xy_ap[:, 0] / pupil_r
        v = xy_ap[:, 1] / pupil_r

        # ── Scene directions: foveal-weighted hemisphere ─────────────────────
        # Scene rays point in the +Z direction (into the eye from the world)
        scene_dirs = SphereCoords.foveal_sample(
            N, -ax,   # sample around -Z (rays coming from scene toward eye)
            rng,
            foveal_sigma=eye.foveal_sigma,
            peripheral_fraction=eye.peripheral_fraction,
            max_theta=max_ang,
        )                                                               # (N, 3)
        # Ensure they point into the eye (+Z hemisphere)
        cos_check = np.einsum('ni,i->n', scene_dirs, ax)
        scene_dirs = np.where(cos_check[:, None] < 0,
                               scene_dirs, -scene_dirs)

        # fu, fv — tan of angular deviations from optical axis
        perp_x = scene_dirs[:, 0]
        perp_y = scene_dirs[:, 1]
        cos_ax = np.clip(np.abs(np.einsum('ni,i->n', scene_dirs, ax)), 1e-9, 1.0)
        fu = perp_x / cos_ax
        fv = perp_y / cos_ax

        # ── Trace through cornea + lens surfaces ─────────────────────────────
        out_dirs, tir_valid = chain.trace(ap_pts, scene_dirs)

        # ── Retinal intersection ──────────────────────────────────────────────
        # Start tracing from the posterior lens surface position
        lens_post_pts = np.full((N, 3), [0., 0., eye.lens_post_z], np.float64)
        # Approximate: advance from pupil by lens thickness for ray starting pos
        # (precise position captured by chain; we use the final direction)
        _, _, ret_valid, _, _ = eye.retina_intersect(
            lens_post_pts, out_dirs
        )

        valid = tir_valid & ret_valid
        if not np.any(valid):
            return None

        # Filter to valid rays only
        m   = valid
        N_v = int(np.sum(m))
        data = np.empty((N_v, _NCOLS), np.float64)
        data[:, _C_U]   = u[m]
        data[:, _C_V]   = v[m]
        data[:, _C_FU]  = fu[m]
        data[:, _C_FV]  = fv[m]
        data[:, _C_IN]  = scene_dirs[m]
        data[:, _C_OUT] = out_dirs[m]
        return data

    # ── Base pass ─────────────────────────────────────────────────────────────
    oversample = max(n_base * 2, n_base + 4096)   # compensate for TIR losses
    chunk = _bake_chunk(oversample)
    data  = chunk if chunk is not None else np.empty((0, _NCOLS), np.float64)

    # Trim to n_base if we got more than requested
    if len(data) > n_base:
        idx  = rng.choice(len(data), size=n_base, replace=False)
        data = data[idx]

    # ── Build quadtree + adaptive refinement ─────────────────────────────────
    from .lens_manifold import _QuadNode, _collect_leaves, _build_tree

    root         = _QuadNode(-1., 1., -1., 1.)
    root.indices = np.arange(len(data))

    for _ in range(n_refine):
        leaves: list = []
        _collect_leaves(root, leaves)
        extras = []
        for leaf in leaves:
            if leaf.indices is None or len(leaf.indices) < 4:
                continue
            var = float(np.mean(np.var(data[leaf.indices][:, _C_OUT], axis=0)))
            if var > threshold:
                n_extra = max(128, len(leaf.indices) * 2)
                extra   = _bake_chunk(n_extra)
                if extra is not None and len(extra):
                    # Keep only the points that fall in this leaf's (u,v) box
                    mask = (
                        (extra[:, _C_U] >= leaf.u0) & (extra[:, _C_U] <= leaf.u1) &
                        (extra[:, _C_V] >= leaf.v0) & (extra[:, _C_V] <= leaf.v1)
                    )
                    if np.any(mask):
                        extras.append(extra[mask])
        if extras:
            data             = np.concatenate([data, *extras], axis=0)
            root.indices     = np.arange(len(data))
            root.children    = None
        root.split(data, threshold, max_depth=12)

    # ── Package as LensManifold ───────────────────────────────────────────────
    import json
    mf        = LensManifold()
    mf._data  = data
    mf._root  = root
    mf._tree  = _build_tree(data)
    mf.meta   = {
        "model":             "human_eye",
        "cornea_r_curv_mm":  eye.cornea_r_curv * 1e3,
        "lens_ant_r_mm":     eye.lens_ant_r_curv * 1e3,
        "lens_post_r_mm":    eye.lens_post_r_curv * 1e3,
        "pupil_r_mm":        eye.pupil_r * 1e3,
        "retina_r_mm":       eye.retina_r * 1e3,
        "axial_length_mm":   eye.fovea_z * 1e3,
        "n_noodles":         len(data),
    }
    return mf


# ---------------------------------------------------------------------------
# Sensor geometry: hemispherical retinal surface
# ---------------------------------------------------------------------------

class RetinalSensor:
    """Hemispherical sensor surface parameterised in angular coordinates.

    Maps retinal eccentricity (angle from fovea) → normalised (u, v) sensor
    coordinates, with optional foveal magnification (cortical magnification
    factor model).

    Usage::

        sensor = RetinalSensor(eye=HUMAN_EYE, cortical_magnification=True)
        u, v   = sensor.retinal_to_uv(az_ret, el_ret)   # (-1,1) sensor coords

    The resulting (u, v) values are compatible with manifold queries.
    """

    def __init__(self, eye: EyeGeometry = HUMAN_EYE,
                 cortical_magnification: bool = True) -> None:
        self.eye                    = eye
        self.cortical_magnification = cortical_magnification
        # CMF parameters (Duncan & Boynton 2003 approximation)
        self._cmf_k: float = 0.065   # deg/mm — foveal scaling constant
        self._cmf_e: float = 15.0    # deg    — half-magnification eccentricity

    def retinal_to_uv(self, az_ret: np.ndarray,
                      el_ret: np.ndarray) -> tuple:
        """Retinal angular coordinates → normalised sensor (u, v) in [-1, 1].

        az_ret, el_ret : (N,) float64 — angles in radians from optical axis
        Returns (u, v) each (N,) float64.
        """
        az_ret = np.asarray(az_ret, np.float64)
        el_ret = np.asarray(el_ret, np.float64)

        if self.cortical_magnification:
            # Log-polar magnification: compress periphery, expand fovea
            r_rad   = np.sqrt(az_ret ** 2 + el_ret ** 2)
            r_deg   = np.degrees(r_rad)
            # M^-1(r) = k * log(1 + r/e) — inverse magnification (Schwartz 1977)
            r_norm  = self._cmf_k * np.log1p(r_deg / self._cmf_e)
            r_norm  = r_norm / (self._cmf_k * math.log1p(90.0 / self._cmf_e))
            # Preserve angle
            phi     = np.arctan2(el_ret, az_ret)
            u       = r_norm * np.cos(phi)
            v       = r_norm * np.sin(phi)
        else:
            # Linear: direct tan-scale normalisation
            max_tan = math.tan(self.eye.max_field_angle * 0.5)
            u       = np.tan(az_ret) / max(max_tan, 1e-9)
            v       = np.tan(el_ret) / max(max_tan, 1e-9)

        u = np.clip(u, -1.0, 1.0)
        v = np.clip(v, -1.0, 1.0)
        return u, v

    def sample_foveal_density(self, n_pts: int,

                              rng: np.random.Generator) -> tuple:
        """Sample (az_ret, el_ret) with realistic human cone-density weighting.

        Uses an inverse-power-law approximation of Curcio et al. (1990):
            density(r_deg) ∝ 1 / (1 + (r_deg / r0)^alpha)^2

        Returns (az_ret, el_ret) each (N,) float64, radians.
        """
        r0    = 0.5    # degrees — half-density at this eccentricity
        alpha = 1.0    # power-law exponent
        max_r = math.degrees(self.eye.max_field_angle)

        # Rejection sampling on (r_deg, phi)
        collected_r   = []
        collected_phi = []
        while sum(len(x) for x in collected_r) < n_pts:
            batch  = max(n_pts * 4, 4096)
            r_try  = rng.uniform(0.0, max_r, batch)
            phi_try = rng.uniform(0.0, 2.0 * math.pi, batch)
            # density weight (normalised to 1 at r=0)
            w      = 1.0 / (1.0 + (r_try / r0) ** alpha) ** 2
            accept = rng.uniform(0.0, 1.0, batch) < w
            collected_r.append(r_try[accept])
            collected_phi.append(phi_try[accept])

        r_deg = np.concatenate(collected_r)[:n_pts]
        phi   = np.concatenate(collected_phi)[:n_pts]
        r_rad = np.radians(r_deg)
        az_ret = r_rad * np.cos(phi)
        el_ret = r_rad * np.sin(phi)
        return az_ret, el_ret
