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
        self._z_front_ref = float(
            max(el.z_vertex for el in preset.lens_group.elements) + 0.001)
        self._manifold_lut  = None   # set by bake_lut(); queried by transfer_ray()
        self._full_data: Optional[np.ndarray] = None  # (N,14) set by bake_full_assembly()

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

    def bake_full_assembly(
        self,
        n_rays: int = 65_536,
        n_wavelengths: int = 3,
        n_refine: int = 2,
        focus_offsets: "list[float] | None" = None,
        n_focus_steps: int = 1,
        focus_range_m: float = 2e-3,
        verbose: bool = True,
    ) -> np.ndarray:
        """Bake the complete front-lens → sensor path as a single (N, 14) array.

        This is the "whole assembly as one LUT" bake.  Every noodle records:
          cols 0-10 : standard schema (aperture UV, field angle, in/out dirs, OPL)
          col  11,12: sensor_x, sensor_y — the actual traced sensor landing position
          col  13   : focus_z offset (metres) — 0.0 for nominal focus

        For a single focus position use ``n_focus_steps=1`` (default).
        For a focus sweep provide either ``focus_offsets`` (explicit list of
        sensor z offsets in metres) or ``n_focus_steps`` + ``focus_range_m``
        (uniform steps over ±focus_range_m/2).

        The result is stored in ``self._full_data`` and returned.
        ``render_sensor_image()`` will use it automatically — zero ray tracing
        at query time.

        Parameters
        ----------
        n_rays        : rays per focus step
        n_wavelengths : wavelength samples for chromatic averaging
        n_refine      : adaptive refinement passes per focus step
        focus_offsets : explicit sensor z offsets (metres); overrides n_focus_steps
        n_focus_steps : number of focus steps when focus_offsets is None
        focus_range_m : total focus sweep range (metres)
        verbose       : print progress

        Returns
        -------
        (N, 14) float64 array
        """
        from .bake_worker import BakeWorker
        worker = BakeWorker(
            self.preset,
            n_rays=n_rays,
            n_wavelengths=n_wavelengths,
            n_refine=n_refine,
            verbose=verbose,
        )
        self._full_data = worker.bake_assembly(
            focus_offsets=focus_offsets,
            n_focus_steps=n_focus_steps,
            focus_range_m=focus_range_m,
        )
        return self._full_data

    def bake_cpp_transfer_grid_streaming(
        self,
        n_rays: int,
        n_u: int,
        n_v: int,
        n_wavelengths: int = 3,
        full_assembly_payload: bool = True,
        focus_z: float = 0.0,
        batch_size: int = 65_536,
        verbose: bool = True,
    ) -> np.ndarray:
        """Bake directly into the C++ transfer payload without storing noodles.

        This is the path for very large tables.  It streams traced noodle
        chunks into the final float32 grid, then normalizes cell means in place.
        """
        import copy
        from .bake_worker import BakeWorker

        stride = 9 if full_assembly_payload else 7
        header_len = 12 if full_assembly_payload else 8
        magic = 14947.0 if full_assembly_payload else 14946.0
        header = np.array(
            [magic, float(n_u), float(n_v),
             -1.0, 1.0, -1.0, 1.0, self._r_ap],
            dtype=np.float32,
        )
        if full_assembly_payload:
            header = np.concatenate([
                header,
                np.array([self._z_sensor, focus_z, self._z_front_ref, 0.0],
                         dtype=np.float32),
            ])

        payload = np.zeros(header_len + n_u * n_v * stride, dtype=np.float32)
        payload[:header_len] = header
        cells = payload[header_len:].reshape(n_v, n_u, stride)

        preset = copy.deepcopy(self.preset)
        preset.sensor.z_pos = float(preset.sensor.z_pos) + float(focus_z)
        worker = BakeWorker(
            preset,
            n_rays=batch_size,
            n_wavelengths=n_wavelengths,
            n_refine=0,
            verbose=False,
        )

        def _accumulate(data: np.ndarray) -> int:
            if data is None or len(data) == 0:
                return 0
            iu = np.clip(((data[:, 0] + 1.0) * 0.5 * n_u).astype(np.int64),
                         0, n_u - 1)
            iv = np.clip(((data[:, 1] + 1.0) * 0.5 * n_v).astype(np.int64),
                         0, n_v - 1)
            np.add.at(cells[:, :, 0], (iv, iu), data[:, 7].astype(np.float32, copy=False))
            np.add.at(cells[:, :, 1], (iv, iu), data[:, 8].astype(np.float32, copy=False))
            np.add.at(cells[:, :, 2], (iv, iu), data[:, 9].astype(np.float32, copy=False))
            np.add.at(cells[:, :, 3], (iv, iu), data[:, 10].astype(np.float32, copy=False))
            np.add.at(cells[:, :, 4], (iv, iu), 1.0)
            np.add.at(cells[:, :, 5], (iv, iu), data[:, 4].astype(np.float32, copy=False))
            np.add.at(cells[:, :, 6], (iv, iu), data[:, 5].astype(np.float32, copy=False))
            if full_assembly_payload:
                np.add.at(cells[:, :, 7], (iv, iu), data[:, 11].astype(np.float32, copy=False))
                np.add.at(cells[:, :, 8], (iv, iu), data[:, 12].astype(np.float32, copy=False))
            return int(len(data))

        traced = 0
        accepted = 0
        while traced < n_rays:
            n = min(batch_size, n_rays - traced)
            ro, rd = worker._sample_rays(n)
            chunk = worker._trace_batch(
                ro, rd, focus_z=float(focus_z), preset_override=preset)
            accepted += _accumulate(chunk)
            traced += n
            if verbose and (traced == n_rays or traced % max(batch_size * 16, 1) == 0):
                print(f"[ManifoldEndpoint] streamed {traced:,}/{n_rays:,} rays"
                      f" accepted={accepted:,}", flush=True)

        counts = cells[:, :, 4]
        mask = counts > 0.0
        for ch in ([0, 1, 2, 3, 5, 6, 7, 8]
                   if full_assembly_payload else [0, 1, 2, 3, 5, 6]):
            cells[:, :, ch][mask] /= counts[mask]
        return payload

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
        focus_z: float = 0.0,
        full_assembly_payload: bool = False,
        accumulator_dtype=np.float32,
    ) -> Optional[np.ndarray]:
        """Convert baked noodles into a compact float32 payload grid.

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

        If a standard ``LensManifold`` exists, its 11-column data is used.
        Otherwise a full-assembly bake can provide the same columns in
        ``_full_data``; for focus sweeps, the nearest focus slice is selected.

        Returns None if no LUT has been baked yet.
        """
        if self._manifold_lut is not None and self._manifold_lut._data is not None:
            data = self._manifold_lut._data   # (N, 11) float64
        elif self._full_data is not None and len(self._full_data) > 0:
            full = self._full_data
            fz = full[:, 13]
            unique_fz = np.unique(fz)
            nearest = unique_fz[np.argmin(np.abs(unique_fz - focus_z))]
            data = full[np.abs(fz - nearest) < 1e-9, :]
            if len(data) == 0:
                return None
        else:
            return None

        use_full = (
            full_assembly_payload
            and self._full_data is not None
            and len(self._full_data) > 0
        )
        MAGIC = 14947.0 if use_full else 14946.0
        header = np.array(
            [MAGIC, float(n_u), float(n_v),
             -1.0, 1.0, -1.0, 1.0, self._r_ap],
            dtype=np.float32,
        )

        if use_full:
            header = np.concatenate([
                header,
                np.array([self._z_sensor, focus_z, self._z_front_ref, 0.0], dtype=np.float32),
            ])

        stride = 9 if use_full else 7
        header_len = 12 if use_full else 8
        payload = np.zeros(header_len + n_v * n_u * stride, dtype=np.float32)
        payload[:header_len] = header.astype(np.float32, copy=False)
        cells = payload[header_len:].reshape(n_v, n_u, stride)

        u_vals = data[:, 0]   # normalised u in [-1, 1]
        v_vals = data[:, 1]   # normalised v in [-1, 1]
        out_d  = data[:, 7:10]
        opl    = data[:, 10]
        in_d   = data[:, 4:6]  # x, y only

        # Map to grid indices (clamp to valid range)
        iu = np.clip(((u_vals + 1.0) * 0.5 * n_u).astype(int), 0, n_u - 1)
        iv = np.clip(((v_vals + 1.0) * 0.5 * n_v).astype(int), 0, n_v - 1)

        np.add.at(cells[:, :, 0], (iv, iu), out_d[:, 0].astype(accumulator_dtype, copy=False))
        np.add.at(cells[:, :, 1], (iv, iu), out_d[:, 1].astype(accumulator_dtype, copy=False))
        np.add.at(cells[:, :, 2], (iv, iu), out_d[:, 2].astype(accumulator_dtype, copy=False))
        np.add.at(cells[:, :, 3], (iv, iu), opl.astype(accumulator_dtype, copy=False))
        np.add.at(cells[:, :, 4], (iv, iu), 1.0)
        np.add.at(cells[:, :, 5], (iv, iu), in_d[:, 0].astype(accumulator_dtype, copy=False))
        np.add.at(cells[:, :, 6], (iv, iu), in_d[:, 1].astype(accumulator_dtype, copy=False))
        if use_full:
            np.add.at(cells[:, :, 7], (iv, iu), data[:, 11].astype(accumulator_dtype, copy=False))
            np.add.at(cells[:, :, 8], (iv, iu), data[:, 12].astype(accumulator_dtype, copy=False))

        # Divide accumulated sums by count
        counts = cells[:, :, 4:5]
        mask   = counts[:, :, 0] > 0
        mean_channels = [0, 1, 2, 3, 5, 6]
        if use_full:
            mean_channels.extend([7, 8])
        for ch in mean_channels:
            cells[:, :, ch][mask] /= counts[:, :, 0][mask]

        return payload

    def render_sensor_image(
        self,
        n_px: int,
        n_py: int,
        focus_z: float = 0.0,
        focus_sigma_m: float = 0.5e-3,
    ) -> Optional[np.ndarray]:
        """Fast sensor image from the baked LUT — no ray tracing.

        Uses pre-baked sensor hit positions (``_full_data``) when available.
        Falls back to projecting ``out_dir`` from the standard 11-col LUT.
        Returns ``(n_py, n_px, 3)`` float32 RGB, or None if nothing is baked.

        Parameters
        ----------
        n_px, n_py    : output image dimensions
        focus_z       : desired focus offset from nominal (metres).
                        Only meaningful when ``_full_data`` has focus sweep data.
        focus_sigma_m : Gaussian width for blending across focus steps.
                        Noodles are weighted by exp(-(Δfz/focus_sigma_m)²).
                        Use 0.0 to select only the nearest focus slice.
        """
        # ── Preferred path: full-assembly data with pre-baked sensor hits ──
        if self._full_data is not None and len(self._full_data) > 0:
            data = self._full_data        # (N, 14)
            fz   = data[:, 13]           # focus_z of each noodle

            if focus_sigma_m > 0.0:
                d = (fz - focus_z) / focus_sigma_m
                w = np.exp(-(d * d))
            else:
                # Hard-select nearest focus slice
                unique_fz = np.unique(fz)
                nearest   = unique_fz[np.argmin(np.abs(unique_fz - focus_z))]
                w = (np.abs(fz - nearest) < 1e-9).astype(np.float64)

            valid_mask = w > 1e-4
            if not np.any(valid_mask):
                return None

            sx = data[valid_mask, 11]   # pre-baked sensor x (metres)
            sy = data[valid_mask, 12]   # pre-baked sensor y
            nw = w[valid_mask]

            r_s = self._r_sensor
            px  = ((sx + r_s) / (2.0 * r_s) * n_px).astype(int)
            py  = ((sy + r_s) / (2.0 * r_s) * n_py).astype(int)
            ok  = (px >= 0) & (px < n_px) & (py >= 0) & (py < n_py)
            px, py, nw = px[ok], py[ok], nw[ok]

            accum = np.zeros(n_px * n_py, dtype=np.float64)
            np.add.at(accum, py * n_px + px, nw)

            img  = np.zeros((n_py, n_px, 3), dtype=np.float32)
            peak = float(accum.max())
            if peak > 0.0:
                bright = (accum / peak).reshape(n_py, n_px).astype(np.float32)
                img[:, :, 0] = bright
                img[:, :, 1] = bright * 0.85
                img[:, :, 2] = bright * 0.65
            return img

        # ── Fallback: project out_dir from standard LUT ────────────────────
        if self._manifold_lut is None or self._manifold_lut._data is None:
            return None

        data   = self._manifold_lut._data   # (N, 11)
        u_ap   = data[:, 0] * self._r_ap
        v_ap   = data[:, 1] * self._r_ap
        out_dx = data[:, 7]
        out_dy = data[:, 8]
        out_dz = data[:, 9]

        dz_safe = np.where(np.abs(out_dz) > 1e-12, out_dz, np.nan)
        t  = (self._z_sensor - self._z_ap) / dz_safe
        sx = u_ap + out_dx * t
        sy = v_ap + out_dy * t

        r_s = self._r_sensor
        px  = ((sx + r_s) / (2.0 * r_s) * n_px).astype(int)
        py  = ((sy + r_s) / (2.0 * r_s) * n_py).astype(int)
        ok  = (px >= 0) & (px < n_px) & (py >= 0) & (py < n_py) & np.isfinite(sx)
        px, py = px[ok], py[ok]

        accum = np.zeros(n_px * n_py, dtype=np.float64)
        np.add.at(accum, py * n_px + px, 1.0)

        img  = np.zeros((n_py, n_px, 3), dtype=np.float32)
        peak = float(accum.max())
        if peak > 0.0:
            bright = (accum / peak).reshape(n_py, n_px).astype(np.float32)
            img[:, :, 0] = bright
            img[:, :, 1] = bright * 0.85
            img[:, :, 2] = bright * 0.65
        return img

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
