"""camera_designer/lens_assembly.py
=====================================
Central descriptor for a complete optical assembly.

LensAssemblySpec owns:
  - Geometry specification (lens groups, casing, baffles, aperture, sensor back)
  - Manifold representation state (NONE | LUT | MLP)
  - Registration logic (maps to tri_groups in the real GPU tracer)
  - Manifold image rendering

Replaces the scattered _neural_assembly_*, _baked_ep, _transfer_grid,
_manifold_ctx_id state on ThickLensFocusLab.

The three representation modes:

  NONE — no manifold proxy; lens surfaces registered as SDF_SPHERE (T2 exact
         parametric refinement, Fresnel physics in T3).

  LUT  — pre-baked dense transfer grid via ManifoldEndpoint.bake_*().
         TODO: replace ManifoldEndpoint.bake_* with GPU dispatch (T1→T2→T3
         read-back), so baking uses wave simulation rather than the CPU Snell
         tracer in bake_worker.py.

  MLP  — trained neural network payload registered on entrance/exit surfaces.
         Interior surfaces registered as absorbers (no payload → T2 bit-3 drop).

Every geometry element carries an ElementPose (depth, shift, tilt).
Housing tessellation (frustum, baffle, box walls) is implemented but requires
the resulting triangles to be added to the scene BVH before registration —
see tessellate_housing().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np

__all__ = [
    "ElementPose",
    "BaffleSpec", "CasingSpec", "ApertureSpec",
    "ConicFrustumSpec", "StraightBoxSpec", "FilterSpec",
    "LensAssemblySpec",
    "tessellate_frustum", "tessellate_baffle", "tessellate_box_walls",
]


# ── Pose ─────────────────────────────────────────────────────────────────────

@dataclass
class ElementPose:
    """Displacement and orientation of any assembly element.

    depth : axial offset along the optical axis (metres, positive = away from world)
    shift : (dy, dz) lateral displacement in the transverse plane (metres)
    tilt  : (ry, rz) tip-tilt rotations about the transverse axes (radians)
    """
    depth: float = 0.0
    shift: Tuple[float, float] = (0.0, 0.0)
    tilt:  Tuple[float, float] = (0.0, 0.0)


# ── Housing geometry specs ────────────────────────────────────────────────────

@dataclass
class BaffleSpec:
    """Annular disk baffle at a fixed axial position.

    Any ray hitting the baffle face (from either side) is absorbed — it
    cannot reach the sensor without passing through an approved optical surface.
    r_hole  : clear aperture radius (rays passing through the hole are free)
    r_outer : outer radius of the baffle disk (matched to barrel inner bore)
    """
    z:       float
    r_hole:  float
    r_outer: float
    n_segs:  int = 32
    pose:    ElementPose = field(default_factory=ElementPose)


@dataclass
class CasingSpec:
    """Cylindrical/frustum barrel inner surface + internal baffle rings.

    All surfaces (barrel wall and baffles) are registered as absorbers.
    r_inner / r_outer define the barrel bore and wall thickness.
    """
    r_inner: float
    r_outer: float
    z_front: float
    z_back:  float
    baffles: List[BaffleSpec] = field(default_factory=list)
    n_segs:  int = 32
    pose:    ElementPose = field(default_factory=ElementPose)


@dataclass
class ApertureSpec:
    """Aperture stop — bladed iris or simple disc.

    n_blades = 0  → circular disc (existing GPU blocker role)
    n_blades > 0  → polygonal iris blade pattern
    """
    z:        float
    r_clear:  float
    n_blades: int = 0
    pose:     ElementPose = field(default_factory=ElementPose)


@dataclass
class ConicFrustumSpec:
    """Frustum adapter — radial→radial or radial→rectangular transition.

    Used for: entrance flare adapters, exit narrows, conic lens hoods.
    kind = 'radial'          circular cross-section at both ends
    kind = 'radial_to_rect'  circle at near end, rectangle at far end
    """
    r_near:  float
    r_far:   float
    z_near:  float
    z_far:   float
    kind:    str = "radial"        # "radial" | "radial_to_rect"
    n_segs:  int = 32
    pose:    ElementPose = field(default_factory=ElementPose)


@dataclass
class StraightBoxSpec:
    """Rectangular tube of fixed depth, with a movable sensor back.

    The inner walls are absorbers.  The sensor back is a sensor-group
    plane that can travel along the optical axis within the tube depth.

    z_front        : open (lens-facing) end of the tube
    depth          : tube length along the optical axis
    sensor_z_offset: how far the sensor has been racked back from z_front
                     sensor plane is at  z_front - sensor_z_offset
                     must be in [0, depth]
    """
    half_w:           float
    half_h:           float
    z_front:          float
    depth:            float
    sensor_z_offset:  float = 0.0
    pose:             ElementPose = field(default_factory=ElementPose)


@dataclass
class FilterSpec:
    """Thin slab filter (UV cut, IR cut, polariser, etc.)."""
    z_front: float
    z_back:  float
    r:       float
    pose:    ElementPose = field(default_factory=ElementPose)


# ── Main class ────────────────────────────────────────────────────────────────

class LensAssemblySpec:
    """Central descriptor for a complete optical assembly.

    Geometry (front→back along optical axis):
      flare_adapter    straight rectangular or radial entrance adapter
      conic_flare_in   conic entrance adapter (radial→rect or radial→radial)
      uv_filter        UV/IR cut filter
      casing           barrel + baffles (wraps all optical groups)
      aperture         bladed or disc aperture
      exit_frustum     exit frustum (conic adapter at sensor end)
      straight_section rectangular tube with movable sensor back

    Manifold representation (NONE | LUT | MLP) is stored here and governs
    how register() maps lens surfaces to GPU tri-group kinds.
    """

    MODE_NONE:        str = "NONE"
    MODE_LUT:         str = "LUT"
    MODE_MLP:         str = "MLP"
    MODE_PARAMETRIC:  str = "PARAMETRIC"

    def __init__(self) -> None:
        # ── Geometry spec (all optional until tessellate_housing() is called) ─
        self.flare_adapter:    Optional[ConicFrustumSpec] = None
        self.conic_flare_in:   Optional[ConicFrustumSpec] = None
        self.uv_filter:        Optional[FilterSpec]       = None
        self.casing:           Optional[CasingSpec]       = None
        self.aperture:         Optional[ApertureSpec]     = None
        self.exit_frustum:     Optional[ConicFrustumSpec] = None
        self.straight_section: Optional[StraightBoxSpec]  = None

        # ── Exact algebraic model (authoritative source of physical geometry) ──
        # Set this to a CompoundLens instance to enable MODE_PARAMETRIC.
        # All physical positions and curvatures come from here; never from mesh.
        self.optics = None   # Optional[camera_designer.compound_optics.CompoundLens]

        # ── Manifold representation ────────────────────────────────────────────
        self.mode: str = self.MODE_NONE

        # MLP state
        self._fwd_payload:  Optional[np.ndarray] = None
        self._bwd_payload:  Optional[np.ndarray] = None

        # LUT state
        self._baked_ep                             = None   # ManifoldEndpoint | None
        self._transfer_grid: Optional[np.ndarray]  = None
        self._transfer_grid_noodles: int           = 0
        self._manifold_ctx_id: int                 = -1

        # ── Registration GIDs (set by register()) ─────────────────────────────
        self._entrance_gid: int = -1
        self._exit_gid:     int = -1

    # ── Payload management ────────────────────────────────────────────────────

    def load_payload(
        self,
        fwd: np.ndarray,
        bwd: Optional[np.ndarray] = None,
    ) -> None:
        """Load pre-trained MLP payload(s) and switch to MLP mode."""
        self._fwd_payload = fwd.astype(np.float32, copy=True)
        self._bwd_payload = bwd.astype(np.float32, copy=True) if bwd is not None else None
        self.mode = self.MODE_MLP

    def load_payload_files(self, fwd_path: str, bwd_path: Optional[str] = None) -> None:
        """Load payload(s) from .npy files, auto-detecting _bwd sibling."""
        import os
        fwd = np.load(fwd_path).astype(np.float32)
        if bwd_path is None:
            auto = fwd_path.replace(".npy", "_bwd.npy")
            if os.path.exists(auto):
                bwd_path = auto
        bwd = np.load(bwd_path).astype(np.float32) if bwd_path else None
        self.load_payload(fwd, bwd)
        print(
            f"[assembly] MLP payload {fwd_path} ({len(fwd)} floats)"
            + (f" + bwd ({len(bwd)} floats)" if bwd is not None else ""),
            flush=True,
        )

    def set_optics(self, optics, *, mode: Optional[str] = None) -> None:
        """Install the canonical compound-lens model for this assembly.

        The optics object is the physical source of truth.  Parametric transfer,
        LUT baking, MLP training, and future wave evaluators should all derive
        lens positions, apertures, curvatures, and material parameters from this
        object rather than from mesh measurements.
        """
        if not hasattr(optics, "build_gpu_payload") or not hasattr(optics, "evaluate_bundle"):
            raise TypeError("optics must provide build_gpu_payload() and evaluate_bundle()")
        self.optics = optics
        if mode is not None:
            self.mode = mode

    def require_optics(self):
        """Return the canonical compound-lens model or raise a clear error."""
        if self.optics is None:
            raise RuntimeError(
                "LensAssemblySpec.optics is not set; build a CompoundLens from "
                "the camera preset before evaluating parametric/LUT/MLP transfer."
            )
        return self.optics

    def build_parametric_payload(self) -> np.ndarray:
        """Build the shader payload from the canonical physical lens state."""
        return self.require_optics().build_gpu_payload()

    def evaluate_transfer(self, ray_bundle):
        """Evaluate the canonical parametric transfer for a vectorized ray bundle."""
        return self.require_optics().evaluate_bundle(ray_bundle)

    # ── Registration ─────────────────────────────────────────────────────────

    def register(
        self,
        tracer,
        lens_surface_groups: list,
        tri_vertices:        np.ndarray,
        tri_centroids:       np.ndarray,
        scene_lenses:        list,
    ) -> None:
        """Register all assembly surfaces with the GPU tracer.

        lens_surface_groups : [(front_ids, back_ids, r_front, r_back), ...]
                              from ThickLensFocusLab.lens_surface_groups
        tri_vertices  : (N, 3, 3) world-space triangle vertices
        tri_centroids : (N, 3)    triangle centroid positions
        scene_lenses  : _scene_lenses(scene) output — for ROC metadata

        Mode dispatch:
          NONE → SDF_SPHERE refinement on all lens surfaces
          MLP  → NEURAL_ASSEMBLY entrance/exit + interior absorbers
          LUT  → SDF_SPHERE refinement (transfer grid registered separately
                 via register_lut_ctx)
        """
        import _spectral_kernels as _sk
        sample_area     = int(getattr(_sk, "TRI_GROUP_SAMPLE_AREA",              1))
        kind_sdf        = int(getattr(_sk, "TRI_PARAM_SURFACE_SDF_SPHERE",       3))
        kind_neural     = int(getattr(_sk, "TRI_PARAM_SURFACE_NEURAL_ASSEMBLY",  4))
        kind_parametric = int(getattr(_sk, "TRI_PARAM_SURFACE_PARAMETRIC_LENS",  5))
        role_none       = 0

        if self.mode == self.MODE_PARAMETRIC:
            self._register_parametric(
                tracer, lens_surface_groups, sample_area, kind_parametric, role_none,
            )
        elif self.mode == self.MODE_MLP:
            self._register_mlp(
                tracer, lens_surface_groups, tri_centroids,
                scene_lenses, sample_area, kind_neural, role_none,
            )
        else:
            # NONE and LUT both keep exact SDF_SPHERE refinement on lens surfaces
            self._register_sdf(
                tracer, lens_surface_groups, tri_vertices,
                sample_area, kind_sdf, role_none,
            )

    def register_lut_ctx(self, tracer) -> None:
        """Register the baked LUT transfer grid as a scale context.

        Called separately from register() because the ctx requires the baked
        grid to already exist.  Safe to call multiple times — re-registers
        if grid is present.
        """
        if self._transfer_grid is None:
            return
        import _spectral_kernels as _sk
        if self._baked_ep is not None:
            ap = self._baked_ep.preset.aperture_stop
            ctx_pos = np.array([0.0, 0.0, float(ap.z_pos)])
            ctx_radius = float(ap.r_outer) * 6.0
        elif self.optics is not None:
            side = self.optics.side("front")
            # New angle-resolved parametric LUT payloads declare axis_idx=0,
            # so the runtime handler projects onto center.x.
            ctx_pos = np.array([float(side.x_pos), 0.0, 0.0])
            ctx_radius = float(side.radius) * 6.0
        else:
            return
        ctx_id = tracer.add_scale_context(
            pos          = ctx_pos,
            radius       = ctx_radius,
            scale_type   = 0,
            dt_m         = 0.0,
            n_substeps   = 0,
            n_real       = 1.0,
            n_imag       = 0.0,
            context_kind = int(_sk.SCALE_CONTEXT_KIND_NEURAL_SURFACE),
            payload      = self._transfer_grid,
        )
        self._manifold_ctx_id = ctx_id

    def _register_sdf(
        self, tracer, lsg, tri_vertices, sample_area, kind_sdf, role_none,
    ) -> None:
        n = 0
        for front_ids, back_ids, r_front, r_back in lsg:
            for ids, r_curv in [(front_ids, r_front), (back_ids, r_back)]:
                if ids.size <= 0 or abs(float(r_curv)) < 1e-6:
                    continue
                verts  = tri_vertices[ids]
                e1 = verts[:, 1] - verts[:, 0]
                e2 = verts[:, 2] - verts[:, 0]
                e3 = verts[:, 2] - verts[:, 1]
                mean_L = max(float(np.mean([
                    np.mean(np.linalg.norm(e, axis=1))
                    for e in [e1, e2, e3]
                ])), 1e-8)
                radius_param = 20.0 * abs(float(r_curv)) / (9.0 * mean_L ** 2)
                tracer.register_tri_group(
                    role_none, sample_area,
                    np.ascontiguousarray(ids, dtype=np.int32),
                    parametric_surface={
                        "kind": kind_sdf,
                        "coeffs": np.array([radius_param, 0.12], dtype=np.float64),
                    },
                )
                n += 1
        if n:
            print(f"[assembly] SDF_SPHERE groups: {n}", flush=True)

    def _register_mlp(
        self, tracer, lsg, tri_centroids, scene_lenses,
        sample_area, kind_neural, role_none,
    ) -> None:
        if not lsg or self._fwd_payload is None:
            return

        front_ids = lsg[0][0]
        back_ids  = lsg[-1][1]
        if int(front_ids.size) == 0:
            return

        x_ent  = float(np.mean(tri_centroids[front_ids, 0]))
        x_exit = float(np.mean(tri_centroids[back_ids,  0])) \
                 if int(back_ids.size) > 0 else x_ent

        # Entrance surface — forward MLP (side=0)
        roc_front = float(scene_lenses[0].radius_front) if scene_lenses else 0.0
        p = self._fwd_payload.copy()
        p[5]  = np.float32(x_ent);   p[6]  = np.float32(x_exit)
        p[9]  = np.float32(0.0);     p[11] = np.float32(roc_front)
        p[12] = np.float32(0.0);     p[13] = np.float32(0)    # axis = X
        p[14] = np.float32(0.0);     p[15] = np.float32(0)    # side = fwd

        self._entrance_gid = int(tracer.register_tri_group(
            role_none, sample_area,
            np.ascontiguousarray(front_ids, dtype=np.int32),
            parametric_surface={"kind": kind_neural, "payload_f32": p},
        ))
        print(
            f"[assembly] MLP entrance gid={self._entrance_gid}"
            f"  x_ent={x_ent*1e3:.2f}mm  x_exit={x_exit*1e3:.2f}mm"
            f"  ROC={roc_front*1e3:.1f}mm",
            flush=True,
        )

        # Exit surface — backward MLP (side=1)
        if self._bwd_payload is not None and int(back_ids.size) > 0:
            roc_back = float(scene_lenses[-1].radius_back) if scene_lenses else 0.0
            p_bwd = self._bwd_payload.copy()
            p_bwd[5]  = np.float32(x_exit);  p_bwd[6]  = np.float32(x_ent)
            p_bwd[9]  = np.float32(0.0);     p_bwd[11] = np.float32(roc_back)
            p_bwd[12] = np.float32(0.0);     p_bwd[13] = np.float32(0)
            p_bwd[14] = np.float32(0.0);     p_bwd[15] = np.float32(1)   # side = bwd

            self._exit_gid = int(tracer.register_tri_group(
                role_none, sample_area,
                np.ascontiguousarray(back_ids, dtype=np.int32),
                parametric_surface={"kind": kind_neural, "payload_f32": p_bwd},
            ))
            print(
                f"[assembly] MLP exit gid={self._exit_gid}"
                f"  x_exit={x_exit*1e3:.2f}mm  ROC={roc_back*1e3:.1f}mm",
                flush=True,
            )

        # Interior surfaces — NEURAL_ASSEMBLY absorbers (no payload → T2 bit-3 drop)
        n_interior = 0
        n_els = len(lsg)
        for i, (f_ids, b_ids, _rf, _rb) in enumerate(lsg):
            is_first = (i == 0);  is_last = (i == n_els - 1)
            for ids in ([f_ids] if not is_first else []) + ([b_ids] if not is_last else []):
                if int(ids.size) == 0:
                    continue
                tracer.register_tri_group(
                    role_none, sample_area,
                    np.ascontiguousarray(ids, dtype=np.int32),
                    parametric_surface={"kind": kind_neural},  # no payload → absorb
                )
                n_interior += int(ids.size)
        if n_interior:
            print(f"[assembly] interior absorbers: {n_interior} tris across {n_els} elements",
                  flush=True)

    def _register_parametric(
        self,
        tracer,
        lsg,
        sample_area,
        kind_parametric,
        role_none,
    ) -> None:
        """Register the PARAMETRIC_LENS payload on the entrance surface; absorb elsewhere.

        The GPU T2 handler (parametric_lens_teleport) evaluates the full assembly
        algebraically on entrance hit and teleports the ray to the exit position.
        All interior and exit mesh triangles must be registered as absorbers so
        that the teleported ray is never re-intercepted by them.
        """
        if self.optics is None:
            return

        payload = self.build_parametric_payload()

        if not lsg:
            return

        front_ids = lsg[0][0]
        back_ids  = lsg[-1][1]

        if int(front_ids.size) == 0:
            return

        self._entrance_gid = int(tracer.register_tri_group(
            role_none, sample_area,
            np.ascontiguousarray(front_ids, dtype=np.int32),
            parametric_surface={"kind": kind_parametric, "payload_f32": payload},
        ))
        print(
            f"[assembly] PARAMETRIC entrance gid={self._entrance_gid}"
            f"  n_surfs={int(payload[1])}"
            f"  payload={len(payload)} floats",
            flush=True,
        )

        # Interior and exit mesh surfaces — absorbers (no payload → T2 bit-3 drop)
        n_interior = 0
        n_els = len(lsg)
        for i, (f_ids, b_ids, _rf, _rb) in enumerate(lsg):
            is_first = (i == 0)
            is_last  = (i == n_els - 1)
            absorb_ids = []
            if not is_first:
                absorb_ids.append(f_ids)
            if not is_last:
                absorb_ids.append(b_ids)
            for ids in absorb_ids:
                if int(ids.size) == 0:
                    continue
                tracer.register_tri_group(
                    role_none, sample_area,
                    np.ascontiguousarray(ids, dtype=np.int32),
                    parametric_surface={"kind": kind_parametric},
                )
                n_interior += int(ids.size)
        # Back face of last lens group is also an absorber
        if int(back_ids.size) > 0:
            tracer.register_tri_group(
                role_none, sample_area,
                np.ascontiguousarray(back_ids, dtype=np.int32),
                parametric_surface={"kind": kind_parametric},
            )
            n_interior += int(back_ids.size)

        if n_interior:
            print(f"[assembly] PARAMETRIC absorbers: {n_interior} tris", flush=True)

    # ── Baking ────────────────────────────────────────────────────────────────

    def bake_lut(
        self,
        ep,                             # ManifoldEndpoint
        tracer=None,
        bake_full_assembly: bool  = False,
        n_rays:             int   = 65_536,
        n_wavelengths:      int   = 3,
        n_refine:           int   = 2,
        n_grid:             int   = 64,
        bake_table_gb:      float = 0.0,
        focus_steps:        int   = 1,
        focus_range_m:      float = 2e-3,
        verbose:            bool  = True,
    ) -> None:
        """Bake a LUT transfer grid and switch to LUT mode.

        ep must be a ManifoldEndpoint constructed from the camera preset.

        TODO: Replace ManifoldEndpoint.bake_* with GPU dispatch (read-back from
        T1→T2→T3 SSBO output) so that baking uses wave simulation rather than
        the CPU Snell path in bake_worker.trace_ray_batch.
        """
        stride = 9 if bake_full_assembly else 7

        if self.optics is None and ep is not None and getattr(ep, "preset", None) is not None:
            try:
                from camera_designer.compound_optics import CompoundLens
                self.optics = CompoundLens.from_preset(ep.preset)
                if verbose:
                    print("[assembly] built CompoundLens from preset for LUT bake", flush=True)
            except Exception as exc:
                if verbose:
                    print(f"[assembly] CompoundLens construction failed; falling back: {exc}",
                          flush=True)

        if self.optics is not None:
            if bake_table_gb > 0.0:
                n_cells    = max(1, int(bake_table_gb * (1024.0 ** 3) // (stride * 4)))
                n_grid_eff = max(2, int(np.sqrt(float(n_cells))))
            else:
                n_grid_eff = int(max(2, n_grid))
            n_dirs = max(4, int(math.ceil(max(1, n_rays) / float(n_grid_eff * n_grid_eff))))
            wavelengths = None
            if ep is not None and getattr(ep, "preset", None) is not None:
                wavelengths = list(getattr(ep.preset, "wavelengths", [])[:max(1, n_wavelengths)])
            if not wavelengths:
                wavelengths = [0.486, 0.587, 0.656][:max(1, n_wavelengths)]

            if verbose:
                print(
                    f"[assembly] parametric LUT bake grid={n_grid_eff}x{n_grid_eff} "
                    f"angle_samples/cell≈{n_dirs} "
                    f"wavelengths={len(wavelengths)}",
                    flush=True,
                )

            grid, n_src = self.optics.build_transfer_lut(
                side="front",
                n_u=n_grid_eff,
                n_v=n_grid_eff,
                n_directions=n_dirs,
                wavelengths_um=wavelengths,
                full_assembly_payload=bake_full_assembly,
                verbose=verbose,
            )
            self._transfer_grid         = grid
            self._transfer_grid_noodles = int(n_src)
            self._baked_ep              = None
            self.mode                   = self.MODE_LUT

            if tracer is not None:
                self.register_lut_ctx(tracer)

            if verbose:
                print(
                    f"[assembly] parametric LUT done — "
                    f"{self._transfer_grid_noodles:,} transmitted samples",
                    flush=True,
                )
            return

        if bake_table_gb > 0.0:
            n_cells    = max(1, int(bake_table_gb * (1024.0 ** 3) // (stride * 4)))
            n_grid_eff = max(2, int(np.sqrt(float(n_cells))))
            approx_gb  = ((12 if bake_full_assembly else 8)
                          + n_grid_eff * n_grid_eff * stride) * 4.0 / (1024.0 ** 3)
            if verbose:
                print(f"[assembly] streaming LUT bake {n_rays:,} rays "
                      f"into {n_grid_eff}×{n_grid_eff} (~{approx_gb:.2f} GiB)...",
                      flush=True)
            grid = ep.bake_cpp_transfer_grid_streaming(
                n_rays=n_rays, n_u=n_grid_eff, n_v=n_grid_eff,
                n_wavelengths=n_wavelengths,
                full_assembly_payload=bake_full_assembly, verbose=verbose,
            )
            n_src = n_rays
        elif bake_full_assembly:
            if verbose:
                print(f"[assembly] full-assembly bake ({n_rays:,}×{focus_steps} focus steps)...",
                      flush=True)
            ep.bake_full_assembly(
                n_rays=n_rays, n_wavelengths=n_wavelengths, n_refine=n_refine,
                n_focus_steps=focus_steps, focus_range_m=focus_range_m, verbose=verbose,
            )
            grid  = ep.build_transfer_grid(n_u=n_grid, n_v=n_grid, full_assembly_payload=True)
            n_src = len(ep._full_data) if ep._full_data is not None else 0
        else:
            if verbose:
                print(f"[assembly] LUT bake ({n_rays:,} noodles)...", flush=True)
            ep.bake_lut(n_rays=n_rays, n_wavelengths=n_wavelengths,
                        n_refine=n_refine, verbose=verbose)
            grid  = ep.build_transfer_grid(n_u=n_grid, n_v=n_grid)
            n_src = ep._manifold_lut.n_noodles if ep._manifold_lut is not None else 0

        if grid is None:
            if verbose:
                print("[assembly] LUT bake produced empty grid", flush=True)
            return

        self._transfer_grid         = grid
        self._transfer_grid_noodles = int(n_src)
        self._baked_ep              = ep
        self.mode                   = self.MODE_LUT

        if tracer is not None:
            self.register_lut_ctx(tracer)

        if verbose:
            print(f"[assembly] LUT done — {self._transfer_grid_noodles:,} noodles", flush=True)

    # ── Sensor image rendering ────────────────────────────────────────────────

    def render_sensor_image(
        self,
        preset,
        n_px:          int,
        n_py:          int,
        n_samples:     int   = 8192,
        wavelength_um: float = 0.587,
        seed:          int   = 0,
    ) -> Optional[np.ndarray]:
        """Return (n_py, n_px, 3) float32 RGB image from the current manifold.

        MLP mode  — batch-infers the network over uniform entry-surface samples.
        LUT mode  — delegates to ManifoldEndpoint.render_sensor_image().
        NONE mode — returns None (caller falls back to full BDPT correlation).
        """
        if self.mode == self.MODE_MLP and self._fwd_payload is not None:
            return self._render_from_mlp(preset, n_px, n_py, n_samples, wavelength_um, seed)
        if self.mode == self.MODE_LUT and self._baked_ep is not None:
            return self._baked_ep.render_sensor_image(n_px, n_py)
        return None

    def _render_from_mlp(
        self,
        preset,
        n_px:          int,
        n_py:          int,
        n_samples:     int,
        wavelength_um: float,
        seed:          int,
    ) -> Optional[np.ndarray]:
        """Batch MLP forward pass → scatter-accumulate onto sensor image.

        Samples (r, theta, direction) uniformly over the entry surface and
        acceptance cone, runs the stored weights, reconstructs exit Cartesian
        geometry, and projects to sensor pixel coordinates.
        """
        from camera_designer.neural_assembly import (
            MAGIC_NEURAL, N_INPUTS, N_OUTPUTS, HEADER_FLOATS, LAYER_OFFSET,
        )

        payload = self._fwd_payload
        if payload is None or float(payload[0]) != MAGIC_NEURAL:
            return None

        r_lens = float(payload[10]);  z_exit = float(payload[6])
        bnd_c0 = float(payload[7]);   bnd_c1 = float(payload[8])
        if r_lens <= 0.0:
            return None

        try:
            z_sensor = float(preset.sensor.z_pos)
            r_sensor = float(preset.sensor.r_max)
        except AttributeError:
            z_sensor = z_exit - 0.05
            r_sensor = r_lens

        rng     = np.random.default_rng(seed)
        r_in    = r_lens * np.sqrt(rng.random(n_samples).astype(np.float32))
        theta_h = rng.uniform(0.0, 2.0 * math.pi, n_samples).astype(np.float32)
        x_hit   = r_in * np.cos(theta_h);  y_hit = r_in * np.sin(theta_h)

        phi_dir   = rng.uniform(0.0, 2.0 * math.pi, n_samples).astype(np.float32)
        cos_theta = rng.random(n_samples).astype(np.float32)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
        dir_x = sin_theta * np.cos(phi_dir)
        dir_y = sin_theta * np.sin(phi_dir)
        dir_z = cos_theta

        r_norm = r_in / r_lens
        accept = dir_z >= (bnd_c0 + bnd_c1 * r_norm * r_norm).astype(np.float32)
        if not np.any(accept):
            return None

        x_hit = x_hit[accept];  y_hit = y_hit[accept]
        dir_x = dir_x[accept];  dir_y = dir_y[accept];  dir_z = dir_z[accept]
        theta_h = theta_h[accept]

        cos_t   = np.cos(theta_h);  sin_t = np.sin(theta_h)
        r_hit   = np.hypot(x_hit, y_hit)
        dir_r   =  dir_x * cos_t + dir_y * sin_t
        dir_phi = -dir_x * sin_t + dir_y * cos_t
        wl_col  = np.full(len(r_hit), wavelength_um, dtype=np.float32)
        X_in    = np.column_stack([r_hit, dir_r, dir_phi,
                                   np.abs(dir_z), wl_col]).astype(np.float32)

        # Batch MLP forward pass (mirrors apply_neural_mlp_from_f32 in C++)
        n_layers   = int(payload[1]);  input_dim  = int(payload[2])
        hidden_dim = int(payload[3]);  output_dim = int(payload[4])
        in_mean  = payload[HEADER_FLOATS:                          HEADER_FLOATS + N_INPUTS]
        in_scale = payload[HEADER_FLOATS + N_INPUTS:               HEADER_FLOATS + 2*N_INPUTS]
        out_mean = payload[HEADER_FLOATS + 2*N_INPUTS:             HEADER_FLOATS + 2*N_INPUTS + N_OUTPUTS]
        out_scl  = payload[HEADER_FLOATS + 2*N_INPUTS + N_OUTPUTS: LAYER_OFFSET]

        x   = ((X_in - in_mean) / np.maximum(np.abs(in_scale), 1e-12)).astype(np.float32)
        ptr = LAYER_OFFSET
        for lyr in range(n_layers):
            in_d  = input_dim  if lyr == 0           else hidden_dim
            out_d = output_dim if lyr == n_layers - 1 else hidden_dim
            W = payload[ptr: ptr + out_d * in_d].reshape(out_d, in_d)
            b = payload[ptr + out_d * in_d: ptr + out_d * in_d + out_d]
            ptr += out_d * in_d + out_d
            x = x @ W.T + b
            if lyr < n_layers - 1:
                x = np.maximum(x, 0.0)
        raw_out = x * out_scl + out_mean   # (N, 6)

        r_out     = raw_out[:, 0];  delta_phi   = raw_out[:, 1]
        dir_r_out = raw_out[:, 2];  dir_phi_out = raw_out[:, 3];  dir_z_out = raw_out[:, 4]

        theta_out = theta_h + delta_phi
        exit_x    = r_out * np.cos(theta_out)
        exit_y    = r_out * np.sin(theta_out)

        d_x = dir_r_out * cos_t - dir_phi_out * sin_t
        d_y = dir_r_out * sin_t + dir_phi_out * cos_t
        d_z = dir_z_out
        d_n = np.maximum(np.sqrt(d_x*d_x + d_y*d_y + d_z*d_z), 1e-12)
        d_x /= d_n;  d_y /= d_n;  d_z /= d_n

        dz_to_sensor = z_sensor - z_exit
        valid = np.abs(d_z) > 1e-6
        t_s   = np.where(valid, dz_to_sensor / d_z, 0.0)
        sx    = exit_x + t_s * d_x
        sy    = exit_y + t_s * d_y

        px = ((sx + r_sensor) / (2.0 * r_sensor) * n_px).astype(int)
        py = ((sy + r_sensor) / (2.0 * r_sensor) * n_py).astype(int)
        ok = valid & (px >= 0) & (px < n_px) & (py >= 0) & (py < n_py)

        # Fallback: if sensor geometry puts nothing on chip, scatter exit positions
        if not np.any(ok):
            r_fb = max(float(np.max(np.abs(exit_x))),
                       float(np.max(np.abs(exit_y))), r_lens, 1e-9)
            px = ((exit_x + r_fb) / (2.0 * r_fb) * n_px).astype(int)
            py = ((exit_y + r_fb) / (2.0 * r_fb) * n_py).astype(int)
            ok = (px >= 0) & (px < n_px) & (py >= 0) & (py < n_py)
            if not np.any(ok):
                return None

        accum = np.zeros(n_px * n_py, dtype=np.float64)
        np.add.at(accum, py[ok] * n_px + px[ok], 1.0)

        img  = np.zeros((n_py, n_px, 3), dtype=np.float32)
        peak = float(accum.max())
        if peak > 0.0:
            bright = (accum / peak).reshape(n_py, n_px).astype(np.float32)
            img[:, :, 0] = bright
            img[:, :, 1] = bright * 0.85
            img[:, :, 2] = bright * 0.65
        return img

    # ── Housing tessellation ─────────────────────────────────────────────────

    def tessellate_housing(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Generate triangles for all housing surfaces.

        Returns list of (role, vertices (N,3), face_indices (M,3)) tuples.
        role is 'absorber' or 'sensor'.

        These triangles must be added to the scene BVH (call scene.add_tris or
        equivalent) before registration.  Currently a structural stub pending
        the scene mesh API; the tessellation functions are complete.
        """
        result: List[Tuple[str, np.ndarray, np.ndarray]] = []

        if self.casing is not None:
            result.extend(_tessellate_casing(self.casing))

        if self.exit_frustum is not None:
            f = self.exit_frustum
            v, t = tessellate_frustum(f.r_near, f.r_far, f.z_near, f.z_far, f.n_segs)
            result.append(("absorber", v, t))

        if self.straight_section is not None:
            s = self.straight_section
            wv, wt = tessellate_box_walls(s.half_w, s.half_h, s.z_front, s.depth)
            result.append(("absorber", wv, wt))
            z_back = s.z_front - s.sensor_z_offset
            sv, st = _tessellate_rect(s.half_w, s.half_h, z_back)
            result.append(("sensor", sv, st))

        return result


# ── Tessellation utilities (module-level for standalone use) ──────────────────

def tessellate_frustum(
    r_near: float, r_far: float,
    z_near: float, z_far: float,
    n_segs: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lateral surface of a frustum (truncated cone).

    Returns (vertices (2*n_segs, 3), faces (2*n_segs, 3)) float32 / int32.
    Winding: outward normals (inner surface of a barrel points inward —
    flip faces if needed for ray intersection orientation).
    """
    angles = np.linspace(0.0, 2.0 * math.pi, n_segs, endpoint=False, dtype=np.float32)
    cos_a  = np.cos(angles);  sin_a = np.sin(angles)

    verts = np.empty((2 * n_segs, 3), dtype=np.float32)
    verts[0::2, 0] = r_near * cos_a;  verts[0::2, 1] = r_near * sin_a;  verts[0::2, 2] = z_near
    verts[1::2, 0] = r_far  * cos_a;  verts[1::2, 1] = r_far  * sin_a;  verts[1::2, 2] = z_far

    tris = []
    for i in range(n_segs):
        i0 = 2 * i;  i1 = 2 * ((i + 1) % n_segs)
        tris.append([i0, i0 + 1, i1])
        tris.append([i1, i0 + 1, i1 + 1])
    return verts, np.array(tris, dtype=np.int32)


def tessellate_baffle(
    z:       float,
    r_hole:  float,
    r_outer: float,
    n_segs:  int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Annular disk baffle.

    Returns (vertices (2*n_segs, 3), faces (2*n_segs, 3)) float32 / int32.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, n_segs, endpoint=False, dtype=np.float32)
    cos_a  = np.cos(angles);  sin_a = np.sin(angles)

    verts = np.empty((2 * n_segs, 3), dtype=np.float32)
    verts[0::2, 0] = r_hole  * cos_a;  verts[0::2, 1] = r_hole  * sin_a;  verts[0::2, 2] = z
    verts[1::2, 0] = r_outer * cos_a;  verts[1::2, 1] = r_outer * sin_a;  verts[1::2, 2] = z

    tris = []
    for i in range(n_segs):
        i0 = 2 * i;  i1 = 2 * ((i + 1) % n_segs)
        tris.append([i0, i0 + 1, i1])
        tris.append([i1, i0 + 1, i1 + 1])
    return verts, np.array(tris, dtype=np.int32)


def tessellate_box_walls(
    half_w: float, half_h: float,
    z_front: float, depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Four inner walls of a rectangular tube (open at both ends).

    Returns (vertices (16, 3), faces (8, 3)) float32 / int32.
    """
    z_back = z_front - depth
    panels = [
        ( half_w, -half_h,  half_w,  half_h),  # +x wall
        (-half_w,  half_h, -half_w, -half_h),  # -x wall
        (-half_w,  half_h,  half_w,  half_h),  # +y wall
        ( half_w, -half_h, -half_w, -half_h),  # -y wall
    ]
    verts = [];  tris = [];  base = 0
    for x0, y0, x1, y1 in panels:
        verts += [[x0, y0, z_front], [x1, y1, z_front],
                  [x0, y0, z_back],  [x1, y1, z_back]]
        tris  += [[base, base+1, base+2], [base+1, base+3, base+2]]
        base  += 4
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)


def _tessellate_rect(
    half_w: float, half_h: float, z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    verts = np.array([
        [-half_w, -half_h, z], [ half_w, -half_h, z],
        [-half_w,  half_h, z], [ half_w,  half_h, z],
    ], dtype=np.float32)
    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return verts, tris


def _tessellate_casing(
    casing: CasingSpec,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    result = []
    v, t = tessellate_frustum(
        casing.r_inner, casing.r_inner,
        casing.z_front, casing.z_back,
        casing.n_segs,
    )
    result.append(("absorber", v, t))
    for b in casing.baffles:
        v, t = tessellate_baffle(b.z, b.r_hole, b.r_outer, b.n_segs)
        result.append(("absorber", v, t))
    return result
