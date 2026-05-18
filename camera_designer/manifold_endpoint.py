"""camera_designer/manifold_endpoint.py
=======================================
Bidirectional BDPT aperture endpoint: the lens manifold as a first-class
connection vertex.

Architecture
------------
A ManifoldEndpoint wraps a CameraPreset and can emit EndpointRecord arrays
from *either* side of the aperture plane:

  sample_forward_records   — scene rays arriving at the aperture (stream_id=0)
  sample_backward_records  — sensor pixels projecting OUTWARD through the aperture
                             cone (stream_id=1).  This is the bokeh / pixel-cone
                             primitive: each pixel samples the aperture disk it
                             actually sees, so an arbitrary aperture shape
                             (hexagonal, star, transmission mask) modulates which
                             samples succeed.  Vignetting for off-axis pixels is
                             natural: fewer aperture samples reach the scene side.
  sample_sensor_records    — forward-trace variant: shoot from the scene side,
                             bin by sensor-pixel landing position.  Used when
                             full-sensor BDPT coverage matters more than per-pixel
                             cone fidelity.  Aperture is always circular.

Both backward variants produce records with stream_id=1 so the correlator
treats them identically.  Choose the variant that matches the use case:
  * Bokeh / shaped-aperture PSF  → sample_backward_records
  * Full-sensor BDPT render       → sample_sensor_records

Both forward and backward surfaces use the same aperture plane as the common
connection surface, so ManifoldWalkStrategy can match them purely by aperture
UV position.

Endpoint record layout (ENDPOINT_DTYPE, 64 bytes):
  pos         — aperture crossing point offset by ±_POS_EPSILON along +z
                so that build_halves_from_records can project it back to the
                aperture plane with a positive ray-parameter t.
  dir         — ray direction at the aperture (→sensor for fwd, →scene for bwd)
  stream_id   — 0.0 = BDPT_SIDE_LIGHT (forward), 1.0 = BDPT_SIDE_SENSOR (backward)
  subpath_id  — forward: sequential index; backward: py * n_px + px (pixel address)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .bake_worker import trace_ray, trace_ray_backward
from .camera_preset import CameraPreset

__all__ = ["ManifoldEndpoint"]

_BDPT_SIDE_LIGHT  = 0.0
_BDPT_SIDE_SENSOR = 1.0
_POS_EPSILON      = 1e-5   # metres: z-offset to ensure t > 0 in project_ray_to_uv


class ManifoldEndpoint:
    """Bidirectional BDPT aperture endpoint for a CameraPreset lens system.

    Parameters
    ----------
    preset   : CameraPreset defining the optical system.
    n_bands  : spectral band count for EndpointRecord allocation.
    fov_deg  : scene field-of-view half-angle in degrees.  Defaults to
               preset.fov_deg if present, otherwise 30°.
    """

    def __init__(
        self,
        preset:  CameraPreset,
        n_bands: int = 1,
        fov_deg: Optional[float] = None,
    ) -> None:
        self.preset   = preset
        self.n_bands  = int(n_bands)
        self._fov_deg = (fov_deg if fov_deg is not None
                         else getattr(preset, "fov_deg", 30.0))
        ap = preset.aperture_stop
        self._z_ap     = float(ap.z_pos)
        self._r_ap     = float(ap.r_outer)
        self._r_inner  = float(ap.r_inner)
        self._z_sensor = float(preset.sensor.z_pos)
        self._r_sensor = float(preset.sensor.r_max)
        self._manifold_lut = None   # set by bake_lut(); queried by transfer_ray()

    # ── Public helpers ─────────────────────────────────────────────────────

    def aperture_plane(self):
        """Return a PlaneSurface for use as the BDPT correlator aperture."""
        from parametric_surface import PlaneSurface
        return PlaneSurface(
            centre=np.array([0., 0., self._z_ap]),
            normal=np.array([0., 0., 1.]),
            u_axis=np.array([1., 0., 0.]),
            half_extent_u=self._r_ap,
            half_extent_v=self._r_ap,
        )

    def make_correlator(self, grid_n: int = 16, match_radius_bins: int = 1):
        """Return a RayCorrelator configured for this endpoint's aperture."""
        from ray_correlator import ManifoldWalkStrategy, RayCorrelator
        strategy = ManifoldWalkStrategy(
            aperture=self.aperture_plane(),
            n_bands=self.n_bands,
            grid_n_u=grid_n,
            grid_n_v=grid_n,
            match_radius_bins=match_radius_bins,
            use_mis=True,
        )
        return RayCorrelator(strategies=[strategy])

    # ── Baked transfer function ────────────────────────────────────────────

    def bake_lut(
        self,
        n_rays: int = 65_536,
        n_wavelengths: int = 3,
        n_refine: int = 2,
        verbose: bool = True,
    ):
        """Run BakeWorker to produce a dense LensManifold transfer LUT.

        The result is cached in self._manifold_lut and returned.  Subsequent
        calls to transfer_ray() and build_transfer_grid() use this cache.

        Returns
        -------
        LensManifold  (camera_software.lens_manifold.LensManifold)
        """
        from .bake_worker import BakeWorker
        worker = BakeWorker(
            self.preset,
            n_rays=n_rays,
            n_wavelengths=n_wavelengths,
            n_refine=n_refine,
            verbose=verbose,
        )
        self._manifold_lut = worker.bake()
        return self._manifold_lut

    def transfer_ray(
        self,
        ap_uv: np.ndarray,
        in_dir: np.ndarray,
        k: int = 8,
    ) -> Optional[np.ndarray]:
        """Query baked LUT: aperture UV + input direction → output direction.

        Parameters
        ----------
        ap_uv  : (2,) float64 — normalised aperture position in [-1, 1]
        in_dir : (3,) float64 — unit approach direction (scene → aperture)
        k      : KNN neighbour count for IDW blend

        Returns
        -------
        (3,) float64 unit output direction, or None if no LUT is baked.
        """
        if self._manifold_lut is None or self._manifold_lut._data is None:
            return None
        uv  = np.atleast_2d(np.asarray(ap_uv,  np.float64))
        ind = np.atleast_2d(np.asarray(in_dir, np.float64))
        dirs, weights = self._manifold_lut.query(uv, k=k, in_dir=ind)
        out = np.sum(dirs[0] * weights[0, :, np.newaxis], axis=0)
        norm = np.linalg.norm(out)
        if norm < 1e-12:
            return None
        return out / norm

    def build_transfer_grid(
        self,
        n_u: int = 32,
        n_v: int = 32,
    ) -> Optional[np.ndarray]:
        """Convert the baked LensManifold into a compact float32 payload grid.

        The grid can be passed directly to
        ``tracer.add_scale_context(..., payload=grid)`` with
        ``context_kind=SCALE_CONTEXT_KIND_NEURAL_SURFACE``.  The C++ handler
        decodes the header, projects each incoming ray to the aperture plane,
        bilinearly interpolates the baked output direction, and redirects the
        ray — skipping the detailed per-element lens geometry entirely.

        Format
        ------
        Header (8 × float32):
          [0] MAGIC = 14946.0 (sentinel)
          [1] n_u, [2] n_v  (grid dimensions, stored as float)
          [3] u_min=-1, [4] u_max=+1, [5] v_min=-1, [6] v_max=+1
          [7] r_ap  (aperture radius, metres)

        Cell data (7 × float32 per cell, row-major [iv, iu]):
          [0..2] out_dx/dy/dz — mean output direction (lens→sensor)
          [3]    opl           — mean OPL (metres)
          [4]    count         — noodle count (0 = empty cell)
          [5..6] in_dx/dy      — mean input direction x, y (diagnostics)

        Returns None if no LUT has been baked yet.
        """
        if self._manifold_lut is None or self._manifold_lut._data is None:
            return None

        data = self._manifold_lut._data   # (N, 11) float64

        MAGIC = 14946.0
        header = np.array(
            [MAGIC, float(n_u), float(n_v),
             -1.0, 1.0, -1.0, 1.0, self._r_ap],
            dtype=np.float32,
        )

        # Accumulate into (n_v, n_u, 7) grid
        cells = np.zeros((n_v, n_u, 7), dtype=np.float64)

        u_vals = data[:, 0]   # normalised u in [-1, 1]
        v_vals = data[:, 1]   # normalised v in [-1, 1]
        out_d  = data[:, 7:10]
        opl    = data[:, 10]
        in_d   = data[:, 4:6]  # x, y only

        # Map to grid indices (clamp to valid range)
        iu = np.clip(((u_vals + 1.0) * 0.5 * n_u).astype(int), 0, n_u - 1)
        iv = np.clip(((v_vals + 1.0) * 0.5 * n_v).astype(int), 0, n_v - 1)

        np.add.at(cells[:, :, 0], (iv, iu), out_d[:, 0])
        np.add.at(cells[:, :, 1], (iv, iu), out_d[:, 1])
        np.add.at(cells[:, :, 2], (iv, iu), out_d[:, 2])
        np.add.at(cells[:, :, 3], (iv, iu), opl)
        np.add.at(cells[:, :, 4], (iv, iu), 1.0)
        np.add.at(cells[:, :, 5], (iv, iu), in_d[:, 0])
        np.add.at(cells[:, :, 6], (iv, iu), in_d[:, 1])

        # Divide accumulated sums by count
        counts = cells[:, :, 4:5]
        mask   = counts[:, :, 0] > 0
        for ch in [0, 1, 2, 3, 5, 6]:
            cells[:, :, ch][mask] /= counts[:, :, 0][mask]

        payload = np.concatenate([header, cells.ravel().astype(np.float32)])
        return payload.astype(np.float32, copy=False)

    # ── Record samplers ────────────────────────────────────────────────────

    def sample_forward_records(
        self,
        n: int,
        wavelength_um: float = 0.587,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample n forward (scene → aperture) EndpointRecords.

        Rays start 1 mm in front of the frontmost lens element and point
        toward the sensor with a random field angle.  On success, trace_ray
        records the aperture crossing UV and the scene-side in_dir.

        pos in each record is offset by _POS_EPSILON in the in_dir direction
        (sensor side), so that project_ray_to_uv(pos, -in_dir) hits the
        aperture at t = epsilon > 0.

        stream_id = 0.0  (BDPT_SIDE_LIGHT)
        """
        from bdpt_integrator import ENDPOINT_DTYPE

        rng  = np.random.default_rng(seed)
        z_start = (max(el.z_vertex for el in self.preset.lens_group.elements)
                   + 0.001)
        fov_half = min(self._fov_deg * 0.5, 20.0)
        ftan = math.tan(math.radians(fov_half))

        rows = []
        attempts = 0
        while len(rows) < n and attempts < n * 10:
            attempts += 1
            ax = rng.uniform(-self._r_ap * 0.5, self._r_ap * 0.5)
            ay = rng.uniform(-self._r_ap * 0.5, self._r_ap * 0.5)
            fu  = rng.uniform(-ftan, ftan)
            fv  = rng.uniform(-ftan, ftan)

            ro = np.array([ax, ay, z_start], np.float64)
            rd = np.array([fu, fv, -1.0], np.float64)
            rd = rd / max(np.linalg.norm(rd), 1e-30)

            result = trace_ray(self.preset, ro, rd, wavelength_um)
            if result is None:
                continue

            ap_hit = result["aperture_hit"]
            in_dir = result["in_dir"]
            opl    = result["opl"]
            rows.append((ap_hit, in_dir, float(opl)))

        if not rows:
            return np.zeros(0, dtype=ENDPOINT_DTYPE)

        recs = np.zeros(len(rows), dtype=ENDPOINT_DTYPE)
        for i, (ap_hit, in_dir, opl) in enumerate(rows):
            # pos: sensor-side offset so (-in_dir) ray hits aperture at t > 0.
            # in_dir points scene→sensor (dz < 0), so ap_hit + in_dir*eps moves
            # in -z direction = toward sensor.
            recs[i]["subpath_id"]   = np.uint32(i)
            recs[i]["band_id"]      = np.uint32(0)
            recs[i]["group_id"]     = np.int32(1)
            recs[i]["vertex_index"] = np.int32(-1)
            recs[i]["pos"]          = (ap_hit + in_dir * _POS_EPSILON).astype(np.float32)
            recs[i]["dir"]          = in_dir.astype(np.float32)
            recs[i]["pathlen_m"]    = np.float32(opl)
            recs[i]["pdf"]          = np.float32(1.0)
            recs[i]["amp_re"]       = np.float32(1.0)
            recs[i]["amp_im"]       = np.float32(0.0)
            recs[i]["cos_theta"]    = np.float32(abs(float(in_dir[2])))
            recs[i]["stream_id"]    = np.float32(_BDPT_SIDE_LIGHT)

        return recs

    def sample_backward_records(
        self,
        n_px: int,
        n_py: int,
        n_per_pixel: int = 1,
        wavelength_um: float = 0.587,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample backward (sensor pixel → aperture) EndpointRecords.

        Pixel-cone / bokeh primitive.  For each pixel (px, py), traces
        n_per_pixel rays outward from the sensor through random points on the
        aperture disk using trace_ray_backward.

        This is the correct approach for arbitrary aperture shapes: sample the
        aperture disk (or any aperture transmission mask) from each pixel and
        reject samples that fall outside the open aperture region.  The
        resulting record density per pixel naturally encodes vignetting — edge
        pixels with narrow acceptance cones produce fewer valid records.

        pos in each record is offset by _POS_EPSILON in ap_dir (scene side),
        so that project_ray_to_uv(pos, -ap_dir) hits the aperture at t > 0.

        subpath_id = py * n_px + px  (flat pixel index, matched by correlator)
        stream_id  = 1.0  (BDPT_SIDE_SENSOR)
        """
        from bdpt_integrator import ENDPOINT_DTYPE

        rng = np.random.default_rng(seed)
        rows = []

        for py in range(n_py):
            for px in range(n_px):
                pixel_id = py * n_px + px
                sx = (px + 0.5) / n_px * 2.0 * self._r_sensor - self._r_sensor
                sy = (py + 0.5) / n_py * 2.0 * self._r_sensor - self._r_sensor
                sz = self._z_sensor

                for _ in range(n_per_pixel):
                    r2   = rng.random() * self._r_ap ** 2
                    ang  = rng.random() * 2.0 * math.pi
                    r    = math.sqrt(r2)
                    ap_x = r * math.cos(ang)
                    ap_y = r * math.sin(ang)

                    sensor_pos = np.array([sx, sy, sz], np.float64)
                    delta      = np.array([ap_x - sx, ap_y - sy,
                                           self._z_ap - sz], np.float64)
                    d_len      = max(np.linalg.norm(delta), 1e-30)
                    rd         = delta / d_len   # pointing sensor → aperture (+z)

                    result = trace_ray_backward(
                        self.preset, sensor_pos, rd, wavelength_um)
                    if result is None:
                        continue

                    ap_hit = result["aperture_hit"]
                    ap_dir = result["in_dir"]   # direction AT aperture, pointing +z
                    opl    = result["opl"]
                    rows.append((ap_hit, ap_dir, float(opl), pixel_id))

        if not rows:
            return np.zeros(0, dtype=ENDPOINT_DTYPE)

        recs = np.zeros(len(rows), dtype=ENDPOINT_DTYPE)
        for i, (ap_hit, ap_dir, opl, pixel_id) in enumerate(rows):
            # pos: scene-side offset so project_ray_to_uv(pos, -ap_dir) hits
            # aperture at t = _POS_EPSILON > 0.
            recs[i]["subpath_id"]   = np.uint32(pixel_id)
            recs[i]["band_id"]      = np.uint32(0)
            recs[i]["group_id"]     = np.int32(2)
            recs[i]["vertex_index"] = np.int32(0)
            recs[i]["pos"]          = (ap_hit + ap_dir * _POS_EPSILON).astype(np.float32)
            recs[i]["dir"]          = ap_dir.astype(np.float32)
            recs[i]["pathlen_m"]    = np.float32(opl)
            recs[i]["pdf"]          = np.float32(1.0)
            recs[i]["amp_re"]       = np.float32(1.0)
            recs[i]["amp_im"]       = np.float32(0.0)
            recs[i]["cos_theta"]    = np.float32(abs(float(ap_dir[2])))
            recs[i]["stream_id"]    = np.float32(_BDPT_SIDE_SENSOR)

        return recs

    def sample_sensor_records(
        self,
        n_px: int,
        n_py: int,
        n_per_pixel: int = 1,
        wavelength_um: float = 0.587,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample sensor-mapped EndpointRecords via forward traces.

        Full-sensor BDPT coverage variant.  Shoots rays from in front of the
        lens and bins each successful trace by the sensor pixel it lands on.
        The backward direction is -in_dir (time-reversal symmetry).

        Use this when you need dense coverage across all pixels for a BDPT
        render.  The aperture is always circular here; for arbitrary aperture
        shapes or per-pixel cone fidelity use sample_backward_records instead.

        stream_id = 1.0  (BDPT_SIDE_SENSOR)
        subpath_id = py * n_px + px
        """
        from bdpt_integrator import ENDPOINT_DTYPE

        rng = np.random.default_rng(seed)

        z_start  = (max(el.z_vertex for el in self.preset.lens_group.elements)
                    + 0.001)
        fov_half = self._fov_deg * 0.5
        ftan     = math.tan(math.radians(fov_half))

        pixel_bucket: dict[int, list] = {}
        target   = n_px * n_py * n_per_pixel
        attempts = 0

        while (sum(len(v) for v in pixel_bucket.values()) < target
               and attempts < target * 100):
            attempts += 1
            ax = rng.uniform(-self._r_ap * 0.5, self._r_ap * 0.5)
            ay = rng.uniform(-self._r_ap * 0.5, self._r_ap * 0.5)
            fu = rng.uniform(-ftan, ftan)
            fv = rng.uniform(-ftan, ftan)

            ro = np.array([ax, ay, z_start], np.float64)
            rd = np.array([fu, fv, -1.0],    np.float64)
            rd = rd / max(np.linalg.norm(rd), 1e-30)

            result = trace_ray(self.preset, ro, rd, wavelength_um)
            if result is None:
                continue

            sh = result["sensor_hit"]
            px = int((sh[0] + self._r_sensor) / (2.0 * self._r_sensor) * n_px)
            py = int((sh[1] + self._r_sensor) / (2.0 * self._r_sensor) * n_py)
            px = max(0, min(n_px - 1, px))
            py = max(0, min(n_py - 1, py))
            pixel_id = py * n_px + px

            bucket = pixel_bucket.setdefault(pixel_id, [])
            if len(bucket) < n_per_pixel:
                bucket.append((
                    result["aperture_hit"],
                    result["in_dir"],   # scene→sensor (dz < 0)
                    float(result["opl"]),
                ))

        if not pixel_bucket:
            return np.zeros(0, dtype=ENDPOINT_DTYPE)

        rows = []
        for pixel_id, bucket in pixel_bucket.items():
            for ap_hit, in_dir, opl in bucket:
                rows.append((ap_hit, in_dir, opl, pixel_id))

        recs = np.zeros(len(rows), dtype=ENDPOINT_DTYPE)
        for i, (ap_hit, in_dir, opl, pixel_id) in enumerate(rows):
            # Time-reversed: ap_dir points aperture → scene (dz > 0).
            ap_dir = -in_dir
            recs[i]["subpath_id"]   = np.uint32(pixel_id)
            recs[i]["band_id"]      = np.uint32(0)
            recs[i]["group_id"]     = np.int32(2)
            recs[i]["vertex_index"] = np.int32(0)
            recs[i]["pos"]          = (ap_hit + ap_dir * _POS_EPSILON).astype(np.float32)
            recs[i]["dir"]          = ap_dir.astype(np.float32)
            recs[i]["pathlen_m"]    = np.float32(opl)
            recs[i]["pdf"]          = np.float32(1.0)
            recs[i]["amp_re"]       = np.float32(1.0)
            recs[i]["amp_im"]       = np.float32(0.0)
            recs[i]["cos_theta"]    = np.float32(abs(float(ap_dir[2])))
            recs[i]["stream_id"]    = np.float32(_BDPT_SIDE_SENSOR)

        return recs
