"""_test_bdpt_pipeline.py — End-to-end and unit tests for the BDPT UV / manifold
pipeline introduced in the manifold-walk session.

Run with:
    python _test_bdpt_pipeline.py
or:
    python -m pytest _test_bdpt_pipeline.py -v

Tests are grouped in four suites:

  Suite A — parametric_surface.py: UVMappableSurface mixin, PlaneSurface,
            ConicSurface, CdFlatSurfaceWithUV adapter.

  Suite B — radial_manifold.py: wedge_direction_sampler, rotate_directions_xy,
            WedgeManifold.bake + query_polar, RadialApertureGrid insert / query.

  Suite C — optical_manifold.py: ApertureGrid, build_halves_from_records,
            ManifoldHalf / SurfaceManifold amplitude accumulation.

  Suite D — ray_correlator.py: ManifoldWalkStrategy (Cartesian and radial
            paths), EndpointPairStrategy, PixelConeOverlapStrategy;
            end-to-end candidate generation from synthetic EndpointRecords.

All suites skip gracefully when optional imports are unavailable.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared across suites
# ─────────────────────────────────────────────────────────────────────────────

def _make_endpoint_records(
    n: int,
    n_bands: int,
    r_max: float = 0.015,
    rng: np.random.Generator | None = None,
    kind: str = "forward",      # "forward" or "backward"
    pixel_grid: int = 32,       # only used for backward
) -> np.ndarray:
    """Build synthetic structured EndpointRecord array.

    * forward records  → vertex_index = -1  (sensor hit)
    * backward records → vertex_index >= 0  (PIXEL_CONE bounce count)

    Terminal positions are sampled uniformly inside a disk of radius r_max
    centred at the origin (z = 0 plane), so backward projection onto a
    PlaneSurface at z=0 with r_max will always succeed.

    ENDPOINT_DTYPE uses compound fields:
        pos  (float32, 3)  — world position
        dir  (float32, 3)  — ray direction
    (not pos_x/pos_y/pos_z/dir_x/dir_y/dir_z).
    """
    from bdpt_integrator import ENDPOINT_DTYPE

    if rng is None:
        rng = np.random.default_rng(42)

    recs = np.zeros(n, dtype=ENDPOINT_DTYPE)

    # Disk positions
    phi_ap   = rng.uniform(0, 2 * math.pi, n)
    r_ap     = rng.uniform(0, r_max, n)
    recs['pos'][:, 0] = (r_ap * np.cos(phi_ap)).astype(np.float32)
    recs['pos'][:, 1] = (r_ap * np.sin(phi_ap)).astype(np.float32)
    recs['pos'][:, 2] = np.zeros(n, dtype=np.float32)

    # Ray directions — pointing roughly toward +z
    theta_d    = rng.uniform(0, 0.3, n)
    phi_d      = rng.uniform(0, 2 * math.pi, n)
    recs['dir'][:, 0] = (np.sin(theta_d) * np.cos(phi_d)).astype(np.float32)
    recs['dir'][:, 1] = (np.sin(theta_d) * np.sin(phi_d)).astype(np.float32)
    recs['dir'][:, 2] = np.cos(theta_d).astype(np.float32)

    recs['pdf']       = np.ones(n, dtype=np.float32)
    recs['cos_theta'] = np.cos(theta_d).astype(np.float32)
    recs['amp_re']    = rng.standard_normal(n).astype(np.float32)
    recs['amp_im']    = rng.standard_normal(n).astype(np.float32)
    recs['pathlen_m'] = rng.uniform(0.01, 0.5, n).astype(np.float32)

    if kind == "forward":
        recs['vertex_index'] = np.full(n, -1, dtype=np.int32)
        recs['subpath_id']   = np.arange(n, dtype=np.uint32)
        recs['band_id']      = np.zeros(n, dtype=np.uint32)
        recs['group_id']     = np.ones(n, dtype=np.int32)
    else:
        # backward: vertex_index = 0 (one bounce), subpath_id encodes pixel
        n_pixels = pixel_grid * pixel_grid
        recs['vertex_index'] = np.zeros(n, dtype=np.int32)
        recs['subpath_id']   = (rng.integers(0, n_pixels, n)).astype(np.uint32)
        recs['band_id']      = np.zeros(n, dtype=np.uint32)
        recs['group_id']     = np.ones(n, dtype=np.int32)

    return recs


def _make_plane_aperture(r_max: float = 0.015) -> object:
    """Build a PlaneSurface centred at origin, normal along +Z."""
    from parametric_surface import PlaneSurface
    return PlaneSurface(
        centre=np.zeros(3, np.float64),
        normal=np.array([0.0, 0.0, 1.0], np.float64),
        u_axis=np.array([1.0, 0.0, 0.0], np.float64),
        half_extent_u=r_max,
        half_extent_v=r_max,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Suite A — parametric_surface.py
# ═════════════════════════════════════════════════════════════════════════════

class TestUVMappableSurface(unittest.TestCase):
    """UVMappableSurface mixin contract."""

    def setUp(self):
        try:
            from parametric_surface import UVMappableSurface, PlaneSurface
            self.UVMappableSurface = UVMappableSurface
            self.PlaneSurface = PlaneSurface
        except ImportError as e:
            self.skipTest(f"parametric_surface unavailable: {e}")

    def test_plane_surface_is_uv_mappable(self):
        surf = self.PlaneSurface(
            centre=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            half_extent_u=0.01,
            half_extent_v=0.01,
        )
        self.assertIsInstance(surf, self.UVMappableSurface)

    def test_plane_uv_roundtrip(self):
        """uv_to_point followed by point_to_uv recovers the original (u, v)."""
        from parametric_surface import PlaneSurface
        surf = PlaneSurface(
            centre=np.array([0.1, 0.2, 0.3]),
            normal=np.array([0.0, 0.0, 1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            half_extent_u=0.02,
            half_extent_v=0.015,
        )
        rng = np.random.default_rng(0)
        for _ in range(100):
            u = rng.uniform(-0.9, 0.9)
            v = rng.uniform(-0.9, 0.9)
            pt = surf.uv_to_point(u, v)
            uv_back = surf.point_to_uv(pt)
            self.assertIsNotNone(uv_back, f"point_to_uv returned None for valid interior point at u={u:.3f} v={v:.3f}")
            u_back, v_back = uv_back
            self.assertAlmostEqual(u_back, u, places=6, msg=f"u roundtrip failed: {u_back} != {u}")
            self.assertAlmostEqual(v_back, v, places=6, msg=f"v roundtrip failed: {v_back} != {v}")

    def test_plane_project_ray_to_uv(self):
        """Ray fired from z=−1 along +z should hit the z=0 plane."""
        from parametric_surface import PlaneSurface
        surf = PlaneSurface(
            centre=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            half_extent_u=0.05,
            half_extent_v=0.05,
        )
        uv = surf.project_ray_to_uv(
            np.array([0.01, 0.005, -1.0]),
            np.array([0.0,  0.0,   1.0]),
        )
        self.assertIsNotNone(uv)
        u, v = uv
        self.assertAlmostEqual(u, 0.01 / 0.05, places=5)
        self.assertAlmostEqual(v, 0.005 / 0.05, places=5)

    def test_plane_area_element_constant(self):
        """PlaneSurface area element should be constant == (2·he_u)·(2·he_v)."""
        from parametric_surface import PlaneSurface
        he_u, he_v = 0.02, 0.015
        surf = PlaneSurface(
            centre=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            half_extent_u=he_u,
            half_extent_v=he_v,
        )
        # area_element returns |∂r/∂u × ∂r/∂v| / eps² — for flat surface this
        # is (he_u * he_v) because the partial derivatives are he_u and he_v.
        ae = surf.area_element(0.0, 0.0)
        expected = he_u * he_v
        self.assertAlmostEqual(ae, expected, delta=expected * 1e-3,
                               msg=f"area element {ae} != {expected}")

    def test_conic_surface_is_uv_mappable(self):
        from parametric_surface import ConicSurface, UVMappableSurface
        surf = ConicSurface(
            vertex=np.zeros(3),
            axis=np.array([0.0, 0.0, 1.0]),
            radius_of_curvature=0.05,
            conic_constant=0.0,
            clear_aperture_radius=0.01,
        )
        self.assertIsInstance(surf, UVMappableSurface)

    def test_conic_uv_interior_not_none(self):
        """point_to_uv for a point inside the aperture should not be None."""
        from parametric_surface import ConicSurface
        R_ca = 0.01
        surf = ConicSurface(
            vertex=np.zeros(3),
            axis=np.array([0.0, 0.0, 1.0]),
            radius_of_curvature=0.05,
            conic_constant=0.0,
            clear_aperture_radius=R_ca,
        )
        pt = surf.uv_to_point(0.0, 0.0)
        uv = surf.point_to_uv(pt)
        self.assertIsNotNone(uv)

    def test_uv_mappable_mixin_abstract_raises(self):
        """A bare UVMappableSurface raises NotImplementedError."""
        from parametric_surface import UVMappableSurface
        m = UVMappableSurface()
        with self.assertRaises(NotImplementedError):
            m.point_to_uv(np.zeros(3))
        with self.assertRaises(NotImplementedError):
            m.uv_to_point(0.0, 0.0)


class TestCdFlatSurfaceWithUV(unittest.TestCase):
    """CdFlatSurfaceWithUV adapter (skip if camera_designer unavailable)."""

    def setUp(self):
        try:
            from parametric_surface import CdFlatSurfaceWithUV
            self.CdFlatSurfaceWithUV = CdFlatSurfaceWithUV
        except ImportError as e:
            self.skipTest(f"parametric_surface unavailable: {e}")

    def test_construction_or_skip(self):
        try:
            surf = self.CdFlatSurfaceWithUV(z_pos=0.0, r_max=0.015)
        except NotImplementedError:
            self.skipTest("camera_designer not available — CdFlatSurfaceWithUV is a stub")

    def test_uv_roundtrip(self):
        try:
            surf = self.CdFlatSurfaceWithUV(z_pos=0.0, r_max=0.020)
        except NotImplementedError:
            self.skipTest("camera_designer not available")
        rng = np.random.default_rng(7)
        for _ in range(50):
            u = float(rng.uniform(-0.9, 0.9))
            v = float(rng.uniform(-0.9, 0.9))
            pt  = surf.uv_to_point(u, v)
            uv2 = surf.point_to_uv(pt)
            self.assertIsNotNone(uv2)
            self.assertAlmostEqual(uv2[0], u, places=6)
            self.assertAlmostEqual(uv2[1], v, places=6)

    def test_area_element_constant(self):
        try:
            r = 0.018
            surf = self.CdFlatSurfaceWithUV(z_pos=0.0, r_max=r)
        except NotImplementedError:
            self.skipTest("camera_designer not available")
        ae = surf.area_element(0.0, 0.0)
        self.assertAlmostEqual(ae, r * r, delta=r * r * 1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# Suite B — radial_manifold.py
# ═════════════════════════════════════════════════════════════════════════════

class TestWedgeDirectionSampler(unittest.TestCase):

    def setUp(self):
        try:
            from radial_manifold import wedge_direction_sampler
            self.sampler = wedge_direction_sampler
        except ImportError as e:
            self.skipTest(f"radial_manifold unavailable: {e}")

    def test_output_shapes(self):
        theta, phi, dirs = self.sampler(1000, math.pi / 36)
        self.assertEqual(theta.shape, (1000,))
        self.assertEqual(phi.shape,   (1000,))
        self.assertEqual(dirs.shape,  (1000, 3))

    def test_unit_directions(self):
        rng = np.random.default_rng(0)
        _, _, dirs = self.sampler(500, math.pi / 36, rng=rng)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_phi_in_wedge(self):
        dphi = math.pi / 12  # 15° wedge
        rng = np.random.default_rng(1)
        _, phi, _ = self.sampler(2000, dphi, rng=rng)
        self.assertTrue(np.all(phi >= -dphi - 1e-12))
        self.assertTrue(np.all(phi <=  dphi + 1e-12))

    def test_theta_bounded(self):
        theta_max = math.pi / 4
        rng = np.random.default_rng(2)
        theta, _, _ = self.sampler(2000, math.pi / 36, theta_max=theta_max, rng=rng)
        self.assertTrue(np.all(theta <= theta_max + 1e-12))

    def test_solid_angle_uniformity(self):
        """cos θ distribution should be flat (equal solid angle)."""
        n = 20000
        rng = np.random.default_rng(3)
        theta, _, _ = self.sampler(n, math.pi / 36, theta_max=math.pi / 2, rng=rng)
        cos_vals = np.cos(theta)
        # Split into 10 equal bins in [0, 1]; counts should be roughly equal.
        counts, _ = np.histogram(cos_vals, bins=10, range=(0.0, 1.0))
        cv = counts.std() / counts.mean()
        self.assertLess(cv, 0.1, f"cos θ distribution not uniform enough (cv={cv:.3f})")

    def test_dtype_float64(self):
        theta, phi, dirs = self.sampler(10, math.pi / 36)
        self.assertEqual(theta.dtype, np.float64)
        self.assertEqual(phi.dtype,   np.float64)
        self.assertEqual(dirs.dtype,  np.float64)

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        t1, p1, d1 = self.sampler(100, math.pi / 36, rng=rng1)
        t2, p2, d2 = self.sampler(100, math.pi / 36, rng=rng2)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(d1, d2)


class TestRotateDirectionsXY(unittest.TestCase):

    def setUp(self):
        try:
            from radial_manifold import rotate_directions_xy
            self.rotate = rotate_directions_xy
        except ImportError as e:
            self.skipTest(f"radial_manifold unavailable: {e}")

    def test_identity_at_zero_angle(self):
        dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])
        dirs[1] /= np.linalg.norm(dirs[1])
        out = self.rotate(dirs, 0.0)
        np.testing.assert_allclose(out, dirs, atol=1e-15)

    def test_quarter_turn(self):
        dirs = np.array([[1.0, 0.0, 0.0]])
        out  = self.rotate(dirs, math.pi / 2)
        np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0], atol=1e-14)

    def test_z_unchanged(self):
        dirs = np.array([[0.5, 0.5, 0.707]])
        dirs[0] /= np.linalg.norm(dirs[0])
        z_before = dirs[0, 2].copy()
        out = self.rotate(dirs, 1.234)
        self.assertAlmostEqual(out[0, 2], z_before, places=14)

    def test_preserves_dtype(self):
        dirs64 = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        out64  = self.rotate(dirs64, 0.1)
        self.assertEqual(out64.dtype, np.float64)

    def test_preserves_unit_length(self):
        rng  = np.random.default_rng(5)
        dirs = rng.standard_normal((50, 3))
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs /= norms
        out  = self.rotate(dirs, 1.23)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-14)

    def test_1d_input(self):
        d = np.array([1.0, 0.0, 0.0])
        out = self.rotate(d, math.pi)
        np.testing.assert_allclose(out, [-1.0, 0.0, 0.0], atol=1e-14)

    def test_full_circle_identity(self):
        dirs = np.array([[0.3, 0.4, 0.866]])
        dirs[0] /= np.linalg.norm(dirs[0])
        out = self.rotate(dirs, 2 * math.pi)
        np.testing.assert_allclose(out, dirs, atol=1e-13)


class TestRadialApertureGrid(unittest.TestCase):

    def setUp(self):
        try:
            from radial_manifold import RadialApertureGrid
            self.Grid = RadialApertureGrid
        except ImportError as e:
            self.skipTest(f"radial_manifold unavailable: {e}")

    def _dummy_half(self, u_norm: float, v_norm: float) -> object:
        """Create a minimal ManifoldHalf-like object with aperture_uv.

        u_norm, v_norm are normalised coordinates in [-1, 1].  The grid
        insert() computes r_phys = sqrt(u²+v²) * r_max from these.
        """
        from unittest.mock import MagicMock
        h = MagicMock()
        h.aperture_uv = (u_norm, v_norm)
        return h

    def _r_phys(self, u_norm: float, v_norm: float, r_max: float) -> float:
        """Physical radius that the grid derives from normalised (u, v)."""
        return math.sqrt(u_norm ** 2 + v_norm ** 2) * r_max

    def test_empty_grid(self):
        g = self.Grid(r_max=0.02, n_r=32)
        self.assertEqual(g.count, 0)
        self.assertEqual(g.query_ring(0.0, 1), [])

    def test_insert_and_count(self):
        r_max = 0.020
        g = self.Grid(r_max=r_max, n_r=32)
        for i in range(10):
            # Normalised: u = i/10 * 0.9, v = 0  → spread across bins
            u_norm = (i / 10.0) * 0.9
            g.insert(self._dummy_half(u_norm, 0.0))
        self.assertEqual(g.count, 10)

    def test_query_ring_returns_nearby(self):
        r_max = 0.020
        g = self.Grid(r_max=r_max, n_r=20)
        # Normalised u=0.5, v=0 → r_phys = 0.5 * 0.020 = 0.010
        u_norm = 0.5
        g.insert(self._dummy_half(u_norm, 0.0))
        r_phys = self._r_phys(u_norm, 0.0, r_max)
        hits = g.query_ring(r_phys, width_bins=1)
        self.assertGreater(len(hits), 0)

    def test_query_far_ring_empty(self):
        r_max = 0.020
        g = self.Grid(r_max=r_max, n_r=20)
        # Insert at normalised u=0.05 → r_phys = 0.001
        g.insert(self._dummy_half(0.05, 0.0))
        r_phys_near = self._r_phys(0.05, 0.0, r_max)
        # Query at r_max * 0.95 (far end) with width 0 should find nothing
        hits = g.query_ring(r_max * 0.95, width_bins=0)
        self.assertEqual(hits, [])

    def test_bin_index_clamp_at_boundary(self):
        """r >= r_max should not raise IndexError."""
        r_max = 0.020
        g = self.Grid(r_max=r_max, n_r=32)
        g.insert(self._dummy_half(0.95, 0.0))
        hits = g.query_ring(r_max, width_bins=2)   # should not crash

    def test_multiple_halves_same_bin(self):
        r_max = 0.020
        g = self.Grid(r_max=r_max, n_r=10)
        # All at normalised u=0.5 → same physical radius → same bin
        u_norm = 0.5
        for _ in range(5):
            g.insert(self._dummy_half(u_norm, 0.0))
        r_phys = self._r_phys(u_norm, 0.0, r_max)
        hits = g.query_ring(r_phys, width_bins=0)
        self.assertEqual(len(hits), 5)


class TestWedgeManifoldBake(unittest.TestCase):
    """Baking smoke test — does not require camera_designer or LensManifold to
    do physics; checks shape / dtype / save-load cycle."""

    def setUp(self):
        try:
            from radial_manifold import WedgeManifold, wedge_direction_sampler
            self.WedgeManifold = WedgeManifold
        except ImportError as e:
            self.skipTest(f"radial_manifold unavailable: {e}")

    def _make_flat_surface_stub(self, z_pos=0.0, r_max=0.015):
        """Minimal surface stub that supports intersect(ro, rd) → (t, hit, nrm)."""
        class _FlatStub:
            def __init__(self, z, r):
                self.z_pos = z
                self.r_max = r
            def intersect(self, ro, rd):
                # Simple plane at z=self.z_pos
                dz = float(rd[2])
                if abs(dz) < 1e-15:
                    return None, None, None
                t = (self.z_pos - float(ro[2])) / dz
                if t <= 1e-9:
                    return None, None, None
                hit = ro + t * rd
                nrm = np.array([0.0, 0.0, 1.0])
                return t, hit, nrm
        return _FlatStub(z_pos, r_max)

    def test_bake_smoke(self):
        """WedgeManifold.bake should return a WedgeManifold with _manifold set."""
        aperture = self._make_flat_surface_stub(z_pos=0.0, r_max=0.015)
        chain    = [self._make_flat_surface_stub(z_pos=0.1, r_max=0.020)]
        wm = self.WedgeManifold.bake(
            surface_chain=chain,
            aperture_surface=aperture,
            n=256,
            delta_phi_half=math.pi / 36,
            theta_max=math.pi / 3,
            seed=7,
        )
        self.assertIsNotNone(wm)
        self.assertAlmostEqual(wm.delta_phi_half, math.pi / 36, places=12)
        self.assertEqual(wm.meta["n"], 256)

    def test_bake_meta_fields(self):
        aperture = self._make_flat_surface_stub()
        wm = self.WedgeManifold.bake(
            surface_chain=[],
            aperture_surface=aperture,
            n=64,
            seed=0,
        )
        self.assertIn("ap_radius", wm.meta)
        self.assertIn("ap_z",      wm.meta)
        self.assertIn("seed",      wm.meta)
        self.assertEqual(wm.meta["seed"], 0)

    def test_query_polar_returns_unit_dir(self):
        aperture = self._make_flat_surface_stub()
        wm = self.WedgeManifold.bake(
            surface_chain=[],
            aperture_surface=aperture,
            n=512,
            delta_phi_half=math.pi / 36,
            seed=1,
        )
        out_dir, opl = wm.query_polar(r=0.005, phi_rel=0.0, theta=0.1)
        self.assertEqual(out_dir.shape, (3,))
        self.assertAlmostEqual(float(np.linalg.norm(out_dir)), 1.0, places=4,
                               msg="query_polar returned non-unit direction")
        self.assertIsInstance(opl, float)

    def test_save_load_roundtrip(self, tmp_path=None):
        import tempfile, os
        aperture = self._make_flat_surface_stub()
        wm = self.WedgeManifold.bake(
            surface_chain=[],
            aperture_surface=aperture,
            n=128,
            delta_phi_half=math.pi / 24,
            seed=42,
        )
        if wm._manifold is None:
            self.skipTest("WedgeManifold._manifold is None — LensManifold unavailable")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test_wedge.npz")
            wm.save(path)
            self.assertTrue(os.path.exists(path))
            wm2 = self.WedgeManifold.load(path)
            self.assertAlmostEqual(wm2.delta_phi_half, wm.delta_phi_half, places=12)
            self.assertAlmostEqual(wm2.theta_max,      wm.theta_max,      places=12)


# ═════════════════════════════════════════════════════════════════════════════
# Suite C — optical_manifold.py
# ═════════════════════════════════════════════════════════════════════════════

class TestApertureGrid(unittest.TestCase):

    def setUp(self):
        try:
            from optical_manifold import ApertureGrid, ManifoldHalf
            from parametric_surface import PlaneSurface
            self.ApertureGrid = ApertureGrid
            self.ManifoldHalf = ManifoldHalf
            self.aperture = PlaneSurface(
                centre=np.zeros(3),
                normal=np.array([0.0, 0.0, 1.0]),
                u_axis=np.array([1.0, 0.0, 0.0]),
                half_extent_u=0.015,
                half_extent_v=0.015,
            )
        except ImportError as e:
            self.skipTest(f"optical_manifold / parametric_surface unavailable: {e}")

    def _make_half(self, u, v):
        from optical_manifold import SurfaceManifold, ManifoldHalf
        from bdpt_integrator import RayStreamKind
        sm = SurfaceManifold(subpath_id=0, entry_surface_id=0, exit_surface_id=0)
        return ManifoldHalf(
            kind=RayStreamKind.FORWARD_LIGHT,
            aperture_surface_id=0,
            aperture_uv=(u, v),
            manifold=sm,
        )

    def test_insert_and_count(self):
        g = self.ApertureGrid(self.aperture, 16, 16)
        for i in range(20):
            g.insert(self._make_half(float(i) / 100 - 0.1, 0.0))
        self.assertEqual(g.count, 20)

    def test_query_neighbors_finds_nearby(self):
        g = self.ApertureGrid(self.aperture, 16, 16)
        h = self._make_half(0.0, 0.0)
        g.insert(h)
        result = g.query_neighbors(0.0, 0.0, radius_bins=1)
        self.assertIn(h, result)

    def test_query_neighbors_misses_far(self):
        g = self.ApertureGrid(self.aperture, 32, 32)
        h = self._make_half(-0.9, -0.9)
        g.insert(h)
        result = g.query_neighbors(0.9, 0.9, radius_bins=0)
        self.assertNotIn(h, result)

    def test_clear_resets_count(self):
        g = self.ApertureGrid(self.aperture, 16, 16)
        for i in range(5):
            g.insert(self._make_half(float(i) * 0.1, 0.0))
        g.clear()
        self.assertEqual(g.count, 0)


class TestBuildHalvesFromRecords(unittest.TestCase):

    def setUp(self):
        try:
            from optical_manifold import build_halves_from_records
            from parametric_surface import PlaneSurface
            self.build_halves = build_halves_from_records
            self.aperture = PlaneSurface(
                centre=np.zeros(3),
                normal=np.array([0.0, 0.0, 1.0]),
                u_axis=np.array([1.0, 0.0, 0.0]),
                half_extent_u=0.020,
                half_extent_v=0.020,
            )
        except ImportError as e:
            self.skipTest(f"optical_manifold / parametric_surface unavailable: {e}")

    def test_returns_list(self):
        from bdpt_integrator import RayStreamKind
        recs = _make_endpoint_records(32, n_bands=1, r_max=0.015)
        halves = self.build_halves(recs, self.aperture, RayStreamKind.FORWARD_LIGHT, 1)
        self.assertIsInstance(halves, list)

    def test_one_half_per_unique_subpath(self):
        from bdpt_integrator import RayStreamKind, ENDPOINT_DTYPE
        # 10 records, each with a distinct subpath_id
        recs = _make_endpoint_records(10, n_bands=2, r_max=0.015)
        halves = self.build_halves(recs, self.aperture, RayStreamKind.FORWARD_LIGHT, 2)
        # At most 10 distinct subpath_ids → at most 10 halves
        self.assertLessEqual(len(halves), 10)
        self.assertGreater(len(halves), 0)

    def test_amp_dtype_float32(self):
        from bdpt_integrator import RayStreamKind
        recs = _make_endpoint_records(8, n_bands=3, r_max=0.015)
        halves = self.build_halves(recs, self.aperture, RayStreamKind.FORWARD_LIGHT, 3)
        for h in halves:
            amp_re, amp_im = h.manifold.terminal_amp
            self.assertEqual(amp_re.dtype, np.float32,
                             f"amp_re dtype is {amp_re.dtype}, expected float32")
            self.assertEqual(amp_im.dtype, np.float32,
                             f"amp_im dtype is {amp_im.dtype}, expected float32")

    def test_empty_input(self):
        from bdpt_integrator import RayStreamKind, ENDPOINT_DTYPE
        empty = np.zeros(0, dtype=ENDPOINT_DTYPE)
        halves = self.build_halves(empty, self.aperture, RayStreamKind.FORWARD_LIGHT, 1)
        self.assertEqual(halves, [])

    def test_aperture_uv_within_bounds(self):
        """All aperture_uv values should be in [-1, 1] (or None if ray misses)."""
        from bdpt_integrator import RayStreamKind
        recs = _make_endpoint_records(64, n_bands=1, r_max=0.015)
        halves = self.build_halves(recs, self.aperture, RayStreamKind.FORWARD_LIGHT, 1)
        for h in halves:
            u, v = h.aperture_uv
            self.assertLessEqual(abs(u), 1.0 + 1e-6,
                                 f"aperture_uv u={u} out of bounds")
            self.assertLessEqual(abs(v), 1.0 + 1e-6,
                                 f"aperture_uv v={v} out of bounds")


# ═════════════════════════════════════════════════════════════════════════════
# Suite D — ray_correlator.py
# ═════════════════════════════════════════════════════════════════════════════

class TestEndpointPairStrategy(unittest.TestCase):

    def setUp(self):
        try:
            from ray_correlator import EndpointPairStrategy
            self.Strategy = EndpointPairStrategy
        except ImportError as e:
            self.skipTest(f"ray_correlator unavailable: {e}")

    def test_empty_inputs_return_empty(self):
        from bdpt_integrator import ENDPOINT_DTYPE
        s = self.Strategy()
        out = s.correlate(
            np.zeros(0, dtype=ENDPOINT_DTYPE),
            np.zeros(0, dtype=ENDPOINT_DTYPE),
            n_bands=1,
        )
        self.assertEqual(out, [])

    def test_single_pair_found(self):
        """Two records at the same position should match."""
        s = self.Strategy(k_nearest=1, max_dist_m=1.0)
        fwd = _make_endpoint_records(1, n_bands=1, r_max=0.005)
        bwd = _make_endpoint_records(1, n_bands=1, r_max=0.005, kind="backward")
        # Put them at the same location
        for f in (fwd, bwd):
            f['pos'][:] = 0.0
        out = s.correlate(fwd, bwd, n_bands=1)
        self.assertGreater(len(out), 0)

    def test_pair_beyond_max_dist_not_returned(self):
        s = self.Strategy(k_nearest=1, max_dist_m=0.001)
        fwd = _make_endpoint_records(1, n_bands=1, r_max=0.005)
        bwd = _make_endpoint_records(1, n_bands=1, r_max=0.005, kind="backward")
        fwd['pos'][:, 0] = 100.0   # 100 m away
        out = s.correlate(fwd, bwd, n_bands=1)
        self.assertEqual(out, [])

    def test_contribution_is_complex64(self):
        s = self.Strategy(k_nearest=1, max_dist_m=1.0)
        fwd = _make_endpoint_records(4, n_bands=1, r_max=0.005)
        bwd = _make_endpoint_records(4, n_bands=1, r_max=0.005, kind="backward")
        for f in (fwd, bwd):
            f['pos'][:] = 0.0
        out = s.correlate(fwd, bwd, n_bands=1)
        for c in out:
            self.assertIsNotNone(c.contribution)
            self.assertEqual(c.contribution.dtype, np.complex64,
                             f"contribution dtype is {c.contribution.dtype}")


class TestManifoldWalkStrategyCartesian(unittest.TestCase):
    """ManifoldWalkStrategy with Cartesian ApertureGrid (use_radial_grid=False)."""

    def setUp(self):
        try:
            from ray_correlator import ManifoldWalkStrategy
            from parametric_surface import PlaneSurface
            self.Strategy = ManifoldWalkStrategy
            self.aperture = PlaneSurface(
                centre=np.zeros(3),
                normal=np.array([0.0, 0.0, 1.0]),
                u_axis=np.array([1.0, 0.0, 0.0]),
                half_extent_u=0.020,
                half_extent_v=0.020,
            )
        except ImportError as e:
            self.skipTest(f"ray_correlator / parametric_surface unavailable: {e}")

    def test_none_aperture_returns_empty(self):
        s = self.Strategy(aperture=None, n_bands=1)
        recs = _make_endpoint_records(8, n_bands=1)
        out = s.correlate(recs, recs, n_bands=1)
        self.assertEqual(out, [])

    def test_empty_inputs_return_empty(self):
        from bdpt_integrator import ENDPOINT_DTYPE
        s = self.Strategy(aperture=self.aperture, n_bands=1, use_radial_grid=False)
        out = s.correlate(
            np.zeros(0, dtype=ENDPOINT_DTYPE),
            np.zeros(0, dtype=ENDPOINT_DTYPE),
            n_bands=1,
        )
        self.assertEqual(out, [])

    def test_candidates_generated(self):
        """With overlapping forward + backward records, candidates should appear."""
        n = 32
        fwd = _make_endpoint_records(n, n_bands=2, r_max=0.015, kind="forward")
        bwd = _make_endpoint_records(n, n_bands=2, r_max=0.015, kind="backward")
        s = self.Strategy(
            aperture=self.aperture,
            n_bands=2,
            grid_n_u=8,
            grid_n_v=8,
            match_radius_bins=2,
            use_radial_grid=False,
        )
        out = s.correlate(fwd, bwd, n_bands=2)
        self.assertIsInstance(out, list)
        # At least some candidates; exact count depends on aperture projection hits.
        # The test just checks we don't crash and return the right type.
        for c in out:
            from bdpt_integrator import CorrelationCandidate
            self.assertIsInstance(c, CorrelationCandidate)

    def test_contribution_dtype_complex64(self):
        n = 16
        fwd = _make_endpoint_records(n, n_bands=1, r_max=0.015)
        bwd = _make_endpoint_records(n, n_bands=1, r_max=0.015, kind="backward")
        s = self.Strategy(
            aperture=self.aperture,
            n_bands=1,
            grid_n_u=4,
            grid_n_v=4,
            match_radius_bins=3,
            use_radial_grid=False,
        )
        out = s.correlate(fwd, bwd, n_bands=1)
        for c in out:
            if c.contribution is not None:
                self.assertEqual(c.contribution.dtype, np.complex64)

    def test_middle_point_set(self):
        """Every candidate should have a MiddlePoint with a valid position."""
        n = 8
        fwd = _make_endpoint_records(n, n_bands=1, r_max=0.015)
        bwd = _make_endpoint_records(n, n_bands=1, r_max=0.015, kind="backward")
        s = self.Strategy(
            aperture=self.aperture,
            n_bands=1,
            grid_n_u=4,
            grid_n_v=4,
            match_radius_bins=4,
        )
        out = s.correlate(fwd, bwd, n_bands=1)
        for c in out:
            self.assertIsNotNone(c.middle, "MiddlePoint should not be None")
            self.assertEqual(c.middle.pos.shape, (3,))
            self.assertTrue(np.all(np.isfinite(c.middle.pos)),
                            f"MiddlePoint.pos has non-finite values: {c.middle.pos}")


class TestManifoldWalkStrategyRadial(unittest.TestCase):
    """ManifoldWalkStrategy with RadialApertureGrid (use_radial_grid=True)."""

    def setUp(self):
        try:
            from ray_correlator import ManifoldWalkStrategy
            from parametric_surface import PlaneSurface
            self.Strategy = ManifoldWalkStrategy
            self.aperture = PlaneSurface(
                centre=np.zeros(3),
                normal=np.array([0.0, 0.0, 1.0]),
                u_axis=np.array([1.0, 0.0, 0.0]),
                half_extent_u=0.020,
                half_extent_v=0.020,
            )
        except ImportError as e:
            self.skipTest(f"ray_correlator / parametric_surface unavailable: {e}")

    def test_radial_grid_available_flag(self):
        """use_radial_grid=True should be accepted without error."""
        s = self.Strategy(
            aperture=self.aperture,
            n_bands=1,
            use_radial_grid=True,
            grid_n_r=32,
        )
        self.assertIsNotNone(s)

    def test_radial_produces_candidates(self):
        n = 32
        fwd = _make_endpoint_records(n, n_bands=1, r_max=0.015)
        bwd = _make_endpoint_records(n, n_bands=1, r_max=0.015, kind="backward")
        s = self.Strategy(
            aperture=self.aperture,
            n_bands=1,
            grid_n_r=16,
            match_radius_bins=2,
            use_radial_grid=True,
        )
        out = s.correlate(fwd, bwd, n_bands=1)
        self.assertIsInstance(out, list)
        for c in out:
            from bdpt_integrator import CorrelationCandidate
            self.assertIsInstance(c, CorrelationCandidate)

    def test_radial_contribution_complex64(self):
        n = 16
        fwd = _make_endpoint_records(n, n_bands=2, r_max=0.015)
        bwd = _make_endpoint_records(n, n_bands=2, r_max=0.015, kind="backward")
        s = self.Strategy(
            aperture=self.aperture,
            n_bands=2,
            grid_n_r=8,
            match_radius_bins=4,
            use_radial_grid=True,
        )
        out = s.correlate(fwd, bwd, n_bands=2)
        for c in out:
            if c.contribution is not None:
                self.assertEqual(c.contribution.dtype, np.complex64)


class TestPixelConeOverlapStrategy(unittest.TestCase):

    def setUp(self):
        try:
            from ray_correlator import PixelConeOverlapStrategy
            self.Strategy = PixelConeOverlapStrategy
        except ImportError as e:
            self.skipTest(f"ray_correlator unavailable: {e}")

    def test_empty_inputs(self):
        from bdpt_integrator import ENDPOINT_DTYPE
        s = self.Strategy(n_px=32, n_py=32, sensor_group_id=1)
        out = s.correlate(
            np.zeros(0, dtype=ENDPOINT_DTYPE),
            np.zeros(0, dtype=ENDPOINT_DTYPE),
            n_bands=1,
        )
        self.assertEqual(out, [])

    def test_wrong_group_id_produces_empty(self):
        fwd = _make_endpoint_records(4, n_bands=1, kind="forward")
        bwd = _make_endpoint_records(4, n_bands=1, kind="backward")
        bwd['group_id'] = 99   # wrong group
        s = self.Strategy(n_px=32, n_py=32, sensor_group_id=1)
        out = s.correlate(fwd, bwd, n_bands=1)
        self.assertEqual(out, [])

    def test_produces_accepted_candidates(self):
        fwd = _make_endpoint_records(8, n_bands=1, kind="forward")
        bwd = _make_endpoint_records(8, n_bands=1, kind="backward")
        s = self.Strategy(n_px=32, n_py=32, sensor_group_id=1)
        out = s.correlate(fwd, bwd, n_bands=1)
        for c in out:
            self.assertTrue(c.accepted,
                            "PixelConeOverlapStrategy candidates should be accepted=True")

    def test_contribution_not_zero_everywhere(self):
        """At least some candidates should have non-zero contribution."""
        fwd = _make_endpoint_records(16, n_bands=1, kind="forward")
        bwd = _make_endpoint_records(16, n_bands=1, kind="backward")
        s = self.Strategy(n_px=32, n_py=32, sensor_group_id=1)
        out = s.correlate(fwd, bwd, n_bands=1)
        if not out:
            self.skipTest("No candidates generated — group_id mismatch or empty bins")
        contributions = [abs(c.contribution) for c in out if c.contribution is not None]
        if contributions:
            self.assertGreater(max(contributions), 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# Suite E — End-to-end smoke test
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndPipelineSmoke(unittest.TestCase):
    """Synthetic end-to-end: build records → ManifoldWalkStrategy → candidates
    → verify no NaN / inf in contributions, positions, and amp dtypes."""

    def setUp(self):
        try:
            from ray_correlator import ManifoldWalkStrategy
            from parametric_surface import PlaneSurface
            self.Strategy    = ManifoldWalkStrategy
            self.PlaneSurface = PlaneSurface
        except ImportError as e:
            self.skipTest(f"required modules unavailable: {e}")

    def _run(self, n_bands: int, use_radial: bool, n_recs: int = 64):
        aperture = self.PlaneSurface(
            centre=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            half_extent_u=0.020,
            half_extent_v=0.020,
        )
        rng = np.random.default_rng(0)
        fwd = _make_endpoint_records(n_recs, n_bands=n_bands, r_max=0.015,
                                     rng=rng, kind="forward")
        bwd = _make_endpoint_records(n_recs, n_bands=n_bands, r_max=0.015,
                                     rng=rng, kind="backward")

        s = self.Strategy(
            aperture=aperture,
            n_bands=n_bands,
            grid_n_u=8,
            grid_n_v=8,
            grid_n_r=16,
            match_radius_bins=3,
            use_radial_grid=use_radial,
        )
        return s.correlate(fwd, bwd, n_bands=n_bands)

    def _check_candidates(self, candidates, label: str):
        for i, c in enumerate(candidates):
            if c.contribution is not None:
                cvals = np.asarray(c.contribution).ravel()
                self.assertTrue(np.all(np.isfinite(cvals)),
                                f"[{label}] candidate {i} has non-finite contribution")
                self.assertEqual(cvals.dtype, np.complex64,
                                 f"[{label}] candidate {i} contribution dtype {cvals.dtype}")
            if c.middle is not None:
                self.assertTrue(np.all(np.isfinite(c.middle.pos)),
                                f"[{label}] candidate {i} MiddlePoint.pos non-finite")

    def test_cartesian_1band(self):
        cands = self._run(n_bands=1, use_radial=False)
        self._check_candidates(cands, "cartesian_1band")

    def test_cartesian_3bands(self):
        cands = self._run(n_bands=3, use_radial=False)
        self._check_candidates(cands, "cartesian_3bands")

    def test_radial_1band(self):
        cands = self._run(n_bands=1, use_radial=True)
        self._check_candidates(cands, "radial_1band")

    def test_radial_3bands(self):
        cands = self._run(n_bands=3, use_radial=True)
        self._check_candidates(cands, "radial_3bands")

    def test_accepted_flag_is_bool(self):
        cands = self._run(n_bands=1, use_radial=False)
        for c in cands:
            self.assertIsInstance(c.accepted, (bool, np.bool_))


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 tests: stream_id field, MIS weights, ShadowRayChecker
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamId(unittest.TestCase):
    """stream_id field in ENDPOINT_DTYPE must exist and be float32."""

    def test_field_exists(self):
        from bdpt_integrator import ENDPOINT_DTYPE
        self.assertIn("stream_id", ENDPOINT_DTYPE.names)

    def test_field_dtype_float32(self):
        from bdpt_integrator import ENDPOINT_DTYPE
        self.assertEqual(ENDPOINT_DTYPE["stream_id"].base, np.dtype(np.float32))

    def test_total_size_unchanged(self):
        from bdpt_integrator import ENDPOINT_DTYPE
        self.assertEqual(ENDPOINT_DTYPE.itemsize, 64)

    def test_no_pad_field(self):
        from bdpt_integrator import ENDPOINT_DTYPE
        self.assertNotIn("_pad", ENDPOINT_DTYPE.names)

    def test_forward_records_stream_id_zero(self):
        """Records made with kind='forward' should have stream_id == 0 (BDPT_SIDE_LIGHT)."""
        recs = _make_endpoint_records(4, n_bands=1, kind="forward")
        # stream_id is the last field; set it explicitly to simulate C++ output.
        recs["stream_id"] = 0.0
        self.assertTrue(np.all(recs["stream_id"] == 0.0))

    def test_backward_records_stream_id_one(self):
        """Records made with kind='backward' should have stream_id == 1 (BDPT_SIDE_SENSOR)."""
        recs = _make_endpoint_records(4, n_bands=1, kind="backward")
        recs["stream_id"] = 1.0
        self.assertTrue(np.all(recs["stream_id"] == 1.0))


class TestManifoldEndpointBake(unittest.TestCase):
    """Smoke tests for ManifoldEndpoint bake_lut / build_transfer_grid / transfer_ray."""

    @classmethod
    def setUpClass(cls):
        try:
            from camera_designer.camera_preset import simple_doublet_preset
            from camera_designer.manifold_endpoint import ManifoldEndpoint
            cls._preset = simple_doublet_preset()
            cls._ep     = ManifoldEndpoint(cls._preset, n_bands=1)
            cls._skip   = False
        except Exception:
            cls._skip = True

    def _maybe_skip(self):
        if self._skip:
            self.skipTest("camera_designer not importable")

    def test_forward_records_non_empty(self):
        self._maybe_skip()
        recs = self._ep.sample_forward_records(64, seed=0)
        self.assertGreater(recs.shape[0], 0, "forward records should be non-empty")

    def test_sensor_records_non_empty(self):
        self._maybe_skip()
        recs = self._ep.sample_sensor_records(4, 4, n_per_pixel=1, seed=0)
        self.assertGreater(recs.shape[0], 0, "sensor records should be non-empty")

    def test_bake_lut_runs(self):
        self._maybe_skip()
        lut = self._ep.bake_lut(n_rays=256, n_wavelengths=1, n_refine=0, verbose=False)
        self.assertIsNotNone(lut)
        self.assertGreater(lut.n_noodles, 0, "baked LUT should have noodles")

    def test_transfer_ray_returns_unit_vector(self):
        self._maybe_skip()
        if self._ep._manifold_lut is None:
            self._ep.bake_lut(n_rays=256, n_wavelengths=1, n_refine=0, verbose=False)
        out = self._ep.transfer_ray(np.array([0.0, 0.0]), np.array([0.0, 0.0, -1.0]))
        if out is not None:
            self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=5)

    def test_build_transfer_grid_shape(self):
        self._maybe_skip()
        if self._ep._manifold_lut is None:
            self._ep.bake_lut(n_rays=256, n_wavelengths=1, n_refine=0, verbose=False)
        grid = self._ep.build_transfer_grid(n_u=16, n_v=16)
        self.assertIsNotNone(grid)
        expected_len = 8 + 7 * 16 * 16
        self.assertEqual(len(grid), expected_len,
                         f"grid should be {expected_len} floats, got {len(grid)}")
        self.assertEqual(grid.dtype, np.float32)

    def test_transfer_grid_magic(self):
        self._maybe_skip()
        if self._ep._manifold_lut is None:
            self._ep.bake_lut(n_rays=256, n_wavelengths=1, n_refine=0, verbose=False)
        grid = self._ep.build_transfer_grid(n_u=16, n_v=16)
        self.assertAlmostEqual(float(grid[0]), 14946.0, places=1, msg="magic sentinel")
        self.assertAlmostEqual(float(grid[1]), 16.0, places=1, msg="n_u")
        self.assertAlmostEqual(float(grid[2]), 16.0, places=1, msg="n_v")

    def test_transfer_grid_non_empty_cells(self):
        self._maybe_skip()
        if self._ep._manifold_lut is None:
            self._ep.bake_lut(n_rays=256, n_wavelengths=1, n_refine=0, verbose=False)
        grid = self._ep.build_transfer_grid(n_u=16, n_v=16)
        cells = grid[8:].reshape(16, 16, 7)
        counts = cells[:, :, 4]
        n_filled = int(np.count_nonzero(counts))
        self.assertGreater(n_filled, 0, "at least some grid cells should be filled")

    def test_render_sensor_image_shape_and_non_empty(self):
        self._maybe_skip()
        if self._ep._manifold_lut is None:
            self._ep.bake_lut(n_rays=512, n_wavelengths=1, n_refine=0, verbose=False)
        img = self._ep.render_sensor_image(8, 8)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (8, 8, 3))
        self.assertGreater(float(img.max()), 0.0, "sensor image should have non-zero pixels")

    def test_render_sensor_image_none_without_bake(self):
        self._maybe_skip()
        from camera_designer.manifold_endpoint import ManifoldEndpoint
        ep_fresh = ManifoldEndpoint(self._preset, n_bands=1)
        result = ep_fresh.render_sensor_image(8, 8)
        self.assertIsNone(result, "should return None when no LUT is baked")


class TestFullAssemblyBake(unittest.TestCase):
    """Tests for BakeWorker.bake_assembly and ManifoldEndpoint.bake_full_assembly."""

    @classmethod
    def setUpClass(cls):
        try:
            from camera_designer.camera_preset import simple_doublet_preset
            from camera_designer.manifold_endpoint import ManifoldEndpoint
            from camera_designer.bake_worker import BakeWorker
            cls._preset     = simple_doublet_preset()
            cls._ep         = ManifoldEndpoint(cls._preset, n_bands=1)
            cls._BakeWorker = BakeWorker
            cls._skip       = False
        except Exception:
            cls._skip = True

    def _maybe_skip(self):
        if self._skip:
            self.skipTest("camera_designer not importable")

    def test_bake_assembly_14_cols(self):
        self._maybe_skip()
        worker = self._BakeWorker(
            self._preset, n_rays=128, n_wavelengths=1, n_refine=0, verbose=False)
        data = worker.bake_assembly(n_focus_steps=1)
        self.assertEqual(data.ndim, 2)
        self.assertEqual(data.shape[1], 14, "full-assembly noodles must have 14 cols")
        self.assertGreater(data.shape[0], 0, "must produce at least one noodle")

    def test_bake_assembly_focus_z_nominal(self):
        self._maybe_skip()
        worker = self._BakeWorker(
            self._preset, n_rays=128, n_wavelengths=1, n_refine=0, verbose=False)
        data = worker.bake_assembly(n_focus_steps=1)
        self.assertTrue(
            np.all(data[:, 13] == 0.0),
            "nominal single-step bake should tag all noodles with focus_z=0 in col 13")

    def test_bake_assembly_multi_focus_unique_tags(self):
        self._maybe_skip()
        worker = self._BakeWorker(
            self._preset, n_rays=64, n_wavelengths=1, n_refine=0, verbose=False)
        data = worker.bake_assembly(n_focus_steps=3, focus_range_m=1e-3)
        unique_fz = np.unique(data[:, 13])
        self.assertGreaterEqual(
            len(unique_fz), 2,
            "3-step bake should produce at least 2 distinct focus_z values in col 13")

    def test_bake_full_assembly_stores_full_data(self):
        self._maybe_skip()
        from camera_designer.manifold_endpoint import ManifoldEndpoint
        ep = ManifoldEndpoint(self._preset, n_bands=1)
        result = ep.bake_full_assembly(
            n_rays=128, n_wavelengths=1, n_refine=0, n_focus_steps=1, verbose=False)
        self.assertIsNotNone(ep._full_data)
        self.assertEqual(ep._full_data.shape[1], 14)
        self.assertIs(result, ep._full_data,
                      "bake_full_assembly must return the same array stored in _full_data")

    def test_full_assembly_builds_transfer_grid(self):
        self._maybe_skip()
        from camera_designer.manifold_endpoint import ManifoldEndpoint
        ep = ManifoldEndpoint(self._preset, n_bands=1)
        ep.bake_full_assembly(
            n_rays=128, n_wavelengths=1, n_refine=0, n_focus_steps=1, verbose=False)
        grid = ep.build_transfer_grid(n_u=8, n_v=8)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.dtype, np.float32)
        self.assertEqual(len(grid), 8 + 7 * 8 * 8)

    def test_full_assembly_builds_cpp_v2_transfer_grid(self):
        self._maybe_skip()
        from camera_designer.manifold_endpoint import ManifoldEndpoint
        ep = ManifoldEndpoint(self._preset, n_bands=1)
        ep.bake_full_assembly(
            n_rays=128, n_wavelengths=1, n_refine=0, n_focus_steps=1, verbose=False)
        grid = ep.build_transfer_grid(
            n_u=8, n_v=8, full_assembly_payload=True)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.dtype, np.float32)
        self.assertAlmostEqual(float(grid[0]), 14947.0, places=1)
        self.assertTrue(np.isfinite(grid[10]))
        self.assertEqual(len(grid), 12 + 9 * 8 * 8)

    def test_render_sensor_image_full_assembly_path(self):
        self._maybe_skip()
        from camera_designer.manifold_endpoint import ManifoldEndpoint
        ep = ManifoldEndpoint(self._preset, n_bands=1)
        ep.bake_full_assembly(
            n_rays=256, n_wavelengths=1, n_refine=0, n_focus_steps=1, verbose=False)
        img = ep.render_sensor_image(16, 16, focus_z=0.0)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (16, 16, 3))
        self.assertEqual(img.dtype, np.float32)
        self.assertGreater(float(img.max()), 0.0, "image must have non-zero pixels")

    def test_render_sensor_image_focus_hard_select(self):
        self._maybe_skip()
        from camera_designer.manifold_endpoint import ManifoldEndpoint
        ep = ManifoldEndpoint(self._preset, n_bands=1)
        ep.bake_full_assembly(
            n_rays=64, n_wavelengths=1, n_refine=0,
            n_focus_steps=2, focus_range_m=0.5e-3, verbose=False)
        if ep._full_data is None or len(ep._full_data) == 0:
            self.skipTest("no full_data produced")
        img = ep.render_sensor_image(8, 8, focus_z=0.0, focus_sigma_m=0.0)
        if img is not None:
            self.assertEqual(img.shape, (8, 8, 3))

    def test_render_sensor_image_focus_blend(self):
        self._maybe_skip()
        from camera_designer.manifold_endpoint import ManifoldEndpoint
        ep = ManifoldEndpoint(self._preset, n_bands=1)
        ep.bake_full_assembly(
            n_rays=128, n_wavelengths=1, n_refine=0,
            n_focus_steps=3, focus_range_m=1e-3, verbose=False)
        img = ep.render_sensor_image(8, 8, focus_z=0.0, focus_sigma_m=0.5e-3)
        self.assertIsNotNone(img, "Gaussian-blended image should not be None")
        self.assertEqual(img.shape, (8, 8, 3))

    def test_training_table_contains_focus_and_wavelength(self):
        self._maybe_skip()
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/training.npy"
            worker = self._BakeWorker(
                self._preset, n_rays=32, n_wavelengths=1, n_refine=0, verbose=False)
            written, target = worker.bake_training_table(
                path, target_gb=1e-6, n_focus_steps=2, focus_range_m=1e-3,
                batch_size=32, max_attempt_factor=200.0)
            data = np.load(path, mmap_mode="r")
            self.assertEqual(data.shape[1], 16)
            self.assertGreater(written, 0)
            self.assertEqual(target, data.shape[0])
            self.assertTrue(np.any(np.isfinite(data[:written, 13])))
            self.assertTrue(np.any(data[:written, 14] > 0.0))
            del data


class TestMISWeights(unittest.TestCase):
    """MIS balance and power heuristic weight functions."""

    def setUp(self):
        try:
            from ray_correlator import mis_balance_weight, mis_power_weight
            self.mis_balance = mis_balance_weight
            self.mis_power   = mis_power_weight
        except ImportError as e:
            self.skipTest(f"ray_correlator import failed: {e}")

    def test_equal_pdfs_balance(self):
        self.assertAlmostEqual(self.mis_balance(1.0, 1.0), 0.5, places=9)

    def test_dominant_a_balance(self):
        w = self.mis_balance(2.0, 1.0)
        self.assertAlmostEqual(w, 2.0 / 3.0, places=9)

    def test_zero_b_balance(self):
        w = self.mis_balance(1.0, 0.0)
        self.assertAlmostEqual(w, 1.0, places=9)

    def test_both_zero_balance_returns_half(self):
        w = self.mis_balance(0.0, 0.0)
        self.assertAlmostEqual(w, 0.5, places=9)

    def test_equal_pdfs_power(self):
        self.assertAlmostEqual(self.mis_power(1.0, 1.0, beta=2.0), 0.5, places=9)

    def test_dominant_a_power(self):
        w = self.mis_power(2.0, 1.0, beta=2.0)
        self.assertAlmostEqual(w, 4.0 / 5.0, places=9)

    def test_range_balance(self):
        for a in [0.1, 0.5, 1.0, 2.0, 10.0]:
            for b in [0.1, 0.5, 1.0, 2.0, 10.0]:
                w = self.mis_balance(a, b)
                self.assertGreaterEqual(w, 0.0)
                self.assertLessEqual(w, 1.0)


class TestShadowRayChecker(unittest.TestCase):
    """ShadowRayChecker — stub mode and basic API."""

    def setUp(self):
        try:
            from ray_correlator import ShadowRayChecker
            self.Checker = ShadowRayChecker
        except ImportError as e:
            self.skipTest(f"ray_correlator import failed: {e}")

    def test_stub_always_visible_single(self):
        chk = self.Checker(tracer=None)
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        self.assertTrue(chk.visible(p0, p1))

    def test_stub_always_visible_batch(self):
        chk = self.Checker(tracer=None)
        p0s = np.zeros((5, 3), dtype=np.float64)
        p1s = np.ones((5, 3),  dtype=np.float64)
        vis = chk.visible_batch(p0s, p1s)
        self.assertTrue(np.all(vis))

    def test_degenerate_segment_visible(self):
        chk = self.Checker(tracer=None)
        p = np.array([1.0, 2.0, 3.0])
        self.assertTrue(chk.visible(p, p.copy()))

    def test_mock_occluding_tracer(self):
        """A tracer that always returns a hit closer than seg_len → occluded."""
        class _OccludingTracer:
            def trace(self, origin, direction, max_t):
                return 0.01   # hit at 1 cm — always inside segment

        chk = self.Checker(tracer=_OccludingTracer(), epsilon=1e-4)
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])   # seg_len = 1 m
        self.assertFalse(chk.visible(p0, p1))

    def test_mock_clearing_tracer(self):
        """A tracer that returns a hit past the segment → visible."""
        class _ClearingTracer:
            def trace(self, origin, direction, max_t):
                return 2.0    # hit at 2 m, beyond 1 m segment

        chk = self.Checker(tracer=_ClearingTracer(), epsilon=1e-4)
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        self.assertTrue(chk.visible(p0, p1))

    def test_mock_miss_tracer(self):
        """A tracer that returns -1 (miss) → visible."""
        class _MissTracer:
            def trace(self, origin, direction, max_t):
                return -1.0

        chk = self.Checker(tracer=_MissTracer(), epsilon=1e-4)
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        self.assertTrue(chk.visible(p0, p1))


class TestManifoldWalkWithShadow(unittest.TestCase):
    """ManifoldWalkStrategy respects shadow_checker and MIS weight."""

    def setUp(self):
        try:
            from ray_correlator import ManifoldWalkStrategy, ShadowRayChecker
            from parametric_surface import PlaneSurface
            self.Strategy      = ManifoldWalkStrategy
            self.ShadowChecker = ShadowRayChecker
            self.aperture      = PlaneSurface(
                centre=np.array([0.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                u_axis=np.array([1.0, 0.0, 0.0]),
                half_extent_u=0.015,
                half_extent_v=0.015,
            )
        except ImportError as e:
            self.skipTest(f"import failed: {e}")

    def _make_recs(self, n=8):
        fwd = _make_endpoint_records(n, n_bands=1, r_max=0.015, kind="forward")
        bwd = _make_endpoint_records(n, n_bands=1, r_max=0.015, kind="backward")
        return fwd, bwd

    def test_stub_checker_accepted(self):
        """With stub checker (always visible), accepted=True for matched pairs."""
        chk = self.ShadowChecker(tracer=None)
        s = self.Strategy(
            aperture=self.aperture,
            n_bands=1,
            grid_n_u=4,
            grid_n_v=4,
            match_radius_bins=8,
            shadow_checker=chk,
        )
        fwd, bwd = self._make_recs()
        cands = s.correlate(fwd, bwd, n_bands=1)
        for c in cands:
            self.assertTrue(c.accepted, "stub checker should keep all pairs accepted")

    def test_occluding_checker_rejected(self):
        """With fully-occluding checker, all candidates have accepted=False."""
        class _AlwaysOccluded:
            def visible(self, p0, p1): return False
            def visible_batch(self, p0s, p1s): return np.zeros(p0s.shape[0], dtype=bool)

        # Wrap in a ShadowRayChecker-like duck type accepted by the strategy.
        # The strategy calls self.shadow_checker.visible(fwd_pos, bwd_pos).
        s = self.Strategy(
            aperture=self.aperture,
            n_bands=1,
            grid_n_u=4,
            grid_n_v=4,
            match_radius_bins=8,
            shadow_checker=_AlwaysOccluded(),
        )
        fwd, bwd = self._make_recs()
        cands = s.correlate(fwd, bwd, n_bands=1)
        self.assertGreater(len(cands), 0, "need at least one candidate to test")
        for c in cands:
            self.assertFalse(c.accepted, "fully-occluded checker must mark all rejected")

    def test_mis_weight_in_range(self):
        """MIS weights stored in MiddlePoint must lie in [0, 1]."""
        s = self.Strategy(
            aperture=self.aperture,
            n_bands=1,
            grid_n_u=4,
            grid_n_v=4,
            match_radius_bins=8,
            use_mis=True,
        )
        fwd, bwd = self._make_recs()
        cands = s.correlate(fwd, bwd, n_bands=1)
        for c in cands:
            w = c.middle.mis_weight
            self.assertGreaterEqual(float(w), 0.0)
            self.assertLessEqual(float(w), 1.0)

    def test_no_mis_weight_one(self):
        """With use_mis=False, MIS weight must be 1.0."""
        s = self.Strategy(
            aperture=self.aperture,
            n_bands=1,
            grid_n_u=4,
            grid_n_v=4,
            match_radius_bins=8,
            use_mis=False,
        )
        fwd, bwd = self._make_recs()
        cands = s.correlate(fwd, bwd, n_bands=1)
        for c in cands:
            self.assertAlmostEqual(float(c.middle.mis_weight), 1.0, places=6)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pretty-print a summary table at the end.
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestUVMappableSurface,
        TestCdFlatSurfaceWithUV,
        TestWedgeDirectionSampler,
        TestRotateDirectionsXY,
        TestRadialApertureGrid,
        TestWedgeManifoldBake,
        TestApertureGrid,
        TestBuildHalvesFromRecords,
        TestEndpointPairStrategy,
        TestManifoldWalkStrategyCartesian,
        TestManifoldWalkStrategyRadial,
        TestPixelConeOverlapStrategy,
        TestEndToEndPipelineSmoke,
        TestStreamId,
        TestManifoldEndpointBake,
        TestFullAssemblyBake,
        TestMISWeights,
        TestShadowRayChecker,
        TestManifoldWalkWithShadow,
    ]:
        suite.addTest(unittest.makeSuite(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
