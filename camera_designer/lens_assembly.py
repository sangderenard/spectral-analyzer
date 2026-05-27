"""camera_designer/lens_assembly.py
=====================================
Central descriptor for a complete optical assembly.

LensAssemblySpec owns:
  - Geometry specification (lens groups, casing, baffles, aperture, sensor back)
  - Camera representation state (NONE | LUT | MLP)
  - Registration logic (maps to tri_groups in the real GPU tracer)
  - Pipeline camera registration

Replaces the scattered _neural_assembly_* and _transfer_grid state on
ThickLensFocusLab.

The three representation modes:

  NONE — no camera proxy; lens surfaces registered as SDF_SPHERE (T2 exact
         parametric refinement, Fresnel physics in T3).

  LUT  — precomputed transfer grid registered into the ray pipeline.

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

        # ── Camera representation ──────────────────────────────────────────────
        self.mode: str = self.MODE_NONE

        # MLP state
        self._fwd_payload:  Optional[np.ndarray] = None
        self._bwd_payload:  Optional[np.ndarray] = None

        # LUT state
        self._transfer_grid: Optional[np.ndarray]  = None
        self._transfer_grid_noodles: int           = 0
        self._manifold_ctx_id: int                 = -1

        # ── Registration GIDs (set by register()) ─────────────────────────────
        self._entrance_gid: int = -1
        self._exit_gid:     int = -1
        self._entrance_stats: dict = {"transmitted": 0, "absorbed": 0}
        self._exit_stats:     dict = {"transmitted": 0, "absorbed": 0}
        self._cached_acceptance_params: "dict | None" = None

        # ── Cached registration args (for re-registration without mesh geometry) ─
        self._cached_lsg:             Optional[list]       = None
        self._cached_tri_centroids:   Optional[np.ndarray] = None
        self._cached_scene_lenses:    Optional[list]       = None

        # ── Progressive refinement state (PARAMETRIC → LUT → MLP) ──────────────
        self._prog_thread    = None   # threading.Thread | None
        self._prog_stop      = None   # threading.Event  | None
        self._prog_lock      = None   # threading.Lock   | None
        self._prog_acc:      Optional[np.ndarray] = None   # (nv,nu,nb,na,7) float64
        self._prog_noodles:  Optional[np.ndarray] = None   # (mlp_min, 11) float32
        self._prog_noodle_idx: int = 0
        self._prog_header:   Optional[np.ndarray] = None   # 16-float LUT header
        self._prog_cfg:      dict = {}
        # Pending payloads posted by worker, consumed by main thread
        self._pending_lut_payload:     Optional[np.ndarray] = None
        self._pending_mlp_fwd_payload: Optional[np.ndarray] = None
        self._pending_mlp_bwd_payload: Optional[np.ndarray] = None

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
        if not hasattr(optics, "build_gpu_payload"):
            raise TypeError("optics must provide build_gpu_payload()")
        self.optics = optics
        self._cached_acceptance_params = None  # invalidate when optics changes
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

    # ── Sensor + aperture ownership ───────────────────────────────────────────

    def populate_sensor(self, scene) -> None:
        """Register the sensor plate as a part of this assembly (StraightBoxSpec).

        After this call straight_section encodes the sensor plane position and
        half-size, derived from scene.image_plate and optics.side("back").
        """
        plate = getattr(scene, "image_plate", None)
        if plate is None:
            return
        sensor_x = float(plate.x)
        sensor_r = float(getattr(plate, "radius", 0.0))
        x_exit: Optional[float] = None
        if self.optics is not None:
            try:
                x_exit = float(self.optics.side("back").x_pos)
            except Exception:
                pass
        if x_exit is None:
            x_exit = sensor_x - 0.05
        # StraightBoxSpec z_front is the lens-facing end; sensor_z_offset is how far
        # the sensor is racked back from that face (sensor_plane = z_front − offset).
        # In our scene x-coords x_exit < sensor_x, so offset must be negative of
        # (sensor_x − x_exit).  We store depth = sensor_x − x_exit and set
        # z_front = sensor_x so sensor_z_offset = 0 keeps sensor_x() correct.
        depth = max(1.0e-4, float(sensor_x) - float(x_exit))
        self.straight_section = StraightBoxSpec(
            half_w=float(sensor_r),
            half_h=float(sensor_r),
            z_front=float(sensor_x),
            depth=float(depth),
            sensor_z_offset=0.0,            # sensor plane = z_front = sensor_x
        )

    def sync_from_scene(self, scene) -> None:
        """Pull sensor plate and iris aperture from scene into the assembly.

        After this call the assembly owns all geometry needed to compute
        backward_ray_target() without consulting scene attributes again.
        """
        self.populate_sensor(scene)
        iris = getattr(scene, "iris_aperture", None)
        if iris is not None and bool(getattr(iris, "enabled", False)):
            self.aperture = ApertureSpec(
                z=float(iris.x_pos),
                r_clear=float(iris.r_inner),
            )

    def backward_ray_target(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (centroid_xyz, radius) — where sensor backward rays must aim.

        When a physical iris is present, we target the EXIT PUPIL — the image of
        the iris through whatever lens groups lie between it and the sensor.  That
        is the cone vertex that backward rays from the sensor actually converge on,
        so aiming there avoids the camera-body frustum blocking them before they
        reach the scene.

        Precedence:
          1. Exit pupil computed by optics (requires iris encoded in CompoundLens)
          2. Physical iris position (fallback when no optics or EP is degenerate)
          3. Exit face of the last lens group
          4. Front face of the straight_section
        """
        if self.optics is not None:
            try:
                x_ep, r_ep = self.optics.exit_pupil
                if r_ep > 0.0:
                    return (np.array([float(x_ep), 0.0, 0.0], dtype=np.float64),
                            float(r_ep))
            except Exception:
                pass
        if self.aperture is not None and self.aperture.r_clear > 0.0:
            return (np.array([self.aperture.z, 0.0, 0.0], dtype=np.float64),
                    float(self.aperture.r_clear))
        if self.optics is not None:
            try:
                back = self.optics.side("back")
                if back.radius > 0.0:
                    return (np.array([float(back.x_pos), 0.0, 0.0], dtype=np.float64),
                            float(back.radius))
            except Exception:
                pass
        if self.straight_section is not None:
            s = self.straight_section
            return (np.array([float(s.z_front), 0.0, 0.0], dtype=np.float64),
                    float(max(s.half_w, s.half_h)))
        return None, 0.0

    def sensor_x(self) -> Optional[float]:
        """Axial position of the sensor plane (z_front − sensor_z_offset)."""
        if self.straight_section is None:
            return None
        s = self.straight_section
        return float(s.z_front - s.sensor_z_offset)

    def sensor_radius(self) -> float:
        """Sensor half-size from straight_section."""
        if self.straight_section is None:
            return 0.0
        return float(max(self.straight_section.half_w, self.straight_section.half_h))

    def camera_body_front_x(self) -> Optional[float]:
        """Physical x where the camera barrel starts — the back face of the last lens group.

        This is the entrance to the enclosed light-tight box between the rear
        lens group and the sensor.  Used for camera-body mesh geometry; it is
        NOT the optical exit pupil (which may be virtual and in front of G1).
        """
        if self.optics is not None:
            try:
                return float(self.optics.side("back").x_pos)
            except Exception:
                pass
        if self.straight_section is not None:
            s = self.straight_section
            return float(s.z_front - s.depth)   # = z_front − depth = G4-back end
        return None

    def camera_body_entrance_radius(self) -> float:
        """Clear aperture at the barrel entrance (last lens group aperture radius).

        Determines the inner radius of the camera-body frustum at the G4-back end.
        """
        if self.optics is not None:
            try:
                return float(self.optics.side("back").radius)
            except Exception:
                pass
        if self.straight_section is not None:
            return float(max(self.straight_section.half_w, self.straight_section.half_h))
        return 0.0

    def build_parametric_payload(self) -> np.ndarray:
        """Build the forward (scene→sensor) shader payload from the canonical model."""
        return self.require_optics().build_gpu_payload()

    def build_parametric_payload_backward(self) -> np.ndarray:
        """Build the backward (sensor→scene) shader payload.

        The payload itself remains in canonical front→back physical order.
        Runtime parametric handlers already detect reverse/sensor rays and
        traverse the canonical payload in reverse with swapped media. Returning
        a pre-reversed payload here would double-reverse sensor transport and
        make back-side rays absorb at the wrong interfaces.
        """
        return self.build_parametric_payload()

    def evaluate_transfer(self, bundle):
        """Evaluate a ray bundle through the installed canonical optics."""
        return self.require_optics().evaluate_bundle(bundle)

    def evaluate_transfer_detailed(self, bundle):
        """Evaluate a ray bundle and retain per-element parametric events."""
        return self.require_optics().evaluate_bundle_detailed(bundle)

    def angular_limits_from_origin(
        self,
        origin=(0.0, 0.0, 0.0),
        *,
        side: str = "front",
        n_azimuth: int = 16,
    ):
        """Return exact-plus-verified angular limits for all registered faces."""
        return self.require_optics().angular_limits_from_origin(
            origin,
            side=side,
            n_azimuth=n_azimuth,
        )

    def vignetting_profile_from_point(
        self,
        focal_point=(0.0, 0.0, 0.0),
        *,
        side: str = "front",
        verify: bool = False,
        n_azimuth: int = 32,
    ):
        """Return per-element closed-form vignetting cones from a focal point."""
        return self.require_optics().vignetting_profile_from_point(
            focal_point,
            side=side,
            verify=verify,
            n_azimuth=n_azimuth,
        )

    def boundary_teleport_profile(
        self,
        side: str = "front",
        *,
        verify: bool = True,
        n_azimuth: int = 32,
    ):
        """Return the compound boundary-to-boundary teleport cone for one side."""
        return self.require_optics().boundary_teleport_profile(
            side,
            verify=verify,
            n_azimuth=n_azimuth,
        )

    def boundary_teleport_profiles(self, *, verify: bool = True, n_azimuth: int = 32):
        """Return front→back and back→front compound boundary profiles."""
        return self.require_optics().boundary_teleport_profiles(
            verify=verify,
            n_azimuth=n_azimuth,
        )

    def profile_field_pair(
        self,
        object_point=(0.0, 0.0, 0.0),
        image_point=None,
        *,
        verify: bool = False,
        n_azimuth: int = 32,
    ):
        """Return object-side and optional image-side field profiles."""
        return self.require_optics().profile_field_pair(
            object_point,
            image_point,
            verify=verify,
            n_azimuth=n_azimuth,
        )

    def acceptance_params(self) -> "dict | None":
        """Return acceptance cone geometry from the analytical parametric model.

        Positions and radii come from CompoundLens.side() — analytically exact
        values derived from the payload, independent of mesh geometry.  Angular
        limits combine exact face geometry with a batch parametric verification
        pass through the installed CompoundLens.

        Returns a dict with:
            entrance_gid  : registered TriGroup ID for the entrance mesh (-1 if unregistered)
            exit_gid      : registered TriGroup ID for the exit mesh (-1 if unregistered)
            entrance_x    : axial position of the entrance surface (m)
            entrance_r    : clear aperture radius at the entrance (m)
            exit_x        : axial position of the exit surface (m)
            exit_r        : clear aperture radius at the exit (m)
            angular_limits: AssemblyAngularLimits, or None on verification failure
        or None if optics is not set.
        """
        if self.optics is None:
            return None

        if self._cached_acceptance_params is not None:
            return self._cached_acceptance_params

        try:
            front = self.optics.side("front")
            back  = self.optics.side("back")
            limits = None
            try:
                limits = self.optics.angular_limits_from_origin(
                    (0.0, 0.0, 0.0),
                    side="front",
                    n_azimuth=16,
                )
                boundary_profiles = self.optics.boundary_teleport_profiles(
                    n_azimuth=16,
                    verify=True,
                )
            except Exception as exc:
                print(f"[acceptance] angular verification failed: {exc}", flush=True)
                boundary_profiles = None

            self._cached_acceptance_params = {
                "entrance_gid": self._entrance_gid,
                "exit_gid":     self._exit_gid,
                "entrance_x":   float(front.x_pos),
                "entrance_r":   float(front.radius),
                "exit_x":       float(back.x_pos),
                "exit_r":       float(back.radius),
                "spread_half_angle_rad": (
                    float(limits.spread_half_angle_rad) if limits is not None else 0.0
                ),
                "verified_spread_half_angle_rad": (
                    float(limits.verified_spread_half_angle_rad) if limits is not None else 0.0
                ),
                "convergence_half_angle_rad": (
                    float(limits.convergence_half_angle_rad) if limits is not None else 0.0
                ),
                "fwd_half_angle": (
                    float(limits.verified_spread_half_angle_rad) if limits is not None else 0.0
                ),
                "bwd_half_angle": (
                    float(self.optics.side_cone("back").half_angle_rad)
                    if self.optics is not None else 0.0
                ),
                "angular_limits": limits,
                "boundary_profiles": boundary_profiles,
            }
            print(
                f"[acceptance] entrance_x={float(front.x_pos)*1e3:.1f}mm"
                f"  exit_x={float(back.x_pos)*1e3:.1f}mm"
                f"  r_ent={float(front.radius)*1e3:.1f}mm"
                f"  r_exit={float(back.radius)*1e3:.1f}mm"
                f"  spread={self._cached_acceptance_params['spread_half_angle_rad']:.4f}rad"
                f"  verified={self._cached_acceptance_params['verified_spread_half_angle_rad']:.4f}rad"
                f"  convergence={self._cached_acceptance_params['convergence_half_angle_rad']:.4f}rad",
                flush=True,
            )
            return self._cached_acceptance_params
        except Exception as exc:
            print(f"[acceptance] failed: {exc}", flush=True)
            return None

    def refresh_teleport_stats(self, tracer) -> None:
        """Pull per-GID manifold dispatch stats from the C++ tracer.

        Updates _entrance_stats and _exit_stats dicts with keys 'magic',
        'transmitted', and 'absorbed'.  Only CPU T2 pipeline rays are counted;
        GPU-shader-dispatched T2 outcomes are not tracked here.
        """
        for attr, gid in (("_entrance_stats", self._entrance_gid),
                          ("_exit_stats",     self._exit_gid)):
            if gid >= 0:
                try:
                    setattr(self, attr, tracer.get_manifold_gid_stats(gid))
                except Exception:
                    pass

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

        # Cache so bake_lut / poll_pending_mode_switch can re-register.
        self._cached_lsg           = lens_surface_groups
        self._cached_tri_centroids = tri_centroids
        self._cached_scene_lenses  = scene_lenses

        if self.mode == self.MODE_PARAMETRIC:
            self._register_parametric(
                tracer, lens_surface_groups, sample_area, kind_parametric, role_none,
            )
        elif self.mode == self.MODE_MLP:
            self._register_mlp(
                tracer, lens_surface_groups, tri_centroids,
                scene_lenses, sample_area, kind_neural, role_none,
            )
        elif self.mode == self.MODE_LUT and self._transfer_grid is not None:
            self._register_lut(
                tracer, lens_surface_groups, sample_area, kind_neural, role_none,
            )
        else:
            # NONE and LUT-not-yet-baked: exact SDF_SPHERE refinement
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
        if self.optics is not None:
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

    def _register_lut(
        self, tracer, lsg, sample_area, kind_neural, role_none,
    ) -> None:
        """Register LUT transfer grid on entrance surface; absorb all others.

        Entrance (front face of first group) gets the full transfer-grid payload
        so the CPU T2 manifold dispatch teleports forward rays correctly.
        All interior and exit surfaces are registered as kind=4 absorbers so
        rays that reach them (e.g. backward BDPT rays) are killed rather than
        traversing lens interiors via Snell physics.
        """
        if not lsg or self._transfer_grid is None:
            return

        payload = np.ascontiguousarray(self._transfer_grid, dtype=np.float32)
        front_ids = lsg[0][0]
        back_ids  = lsg[-1][1]

        if int(front_ids.size) == 0:
            return

        self._entrance_gid = int(tracer.register_tri_group(
            role_none, sample_area,
            np.ascontiguousarray(front_ids, dtype=np.int32),
            parametric_surface={"kind": kind_neural, "payload_f32": payload},
        ))
        print(
            f"[assembly] LUT entrance gid={self._entrance_gid}"
            f"  grid={len(payload)} floats  magic={payload[0]:.0f}",
            flush=True,
        )

        # Interior and exit surfaces — NEURAL_ASSEMBLY absorbers
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
                    parametric_surface={"kind": kind_neural},
                )
                n_interior += int(ids.size)
        # Exit face (back of last group) — canonical PLENS payload. Runtime
        # handlers reverse traversal for sensor rays from color_flag/dir.
        if int(back_ids.size) > 0:
            if self.optics is not None:
                import _spectral_kernels as _sk2
                kind_p = int(getattr(_sk2, "TRI_PARAM_SURFACE_PARAMETRIC_LENS", 5))
                bwd_p = self.build_parametric_payload_backward()
                self._exit_gid = int(tracer.register_tri_group(
                    role_none, sample_area,
                    np.ascontiguousarray(back_ids, dtype=np.int32),
                    parametric_surface={"kind": kind_p, "payload_f32": bwd_p},
                ))
                print(
                    f"[assembly] LUT exit gid={self._exit_gid} (backward PLENS)",
                    flush=True,
                )
            else:
                tracer.register_tri_group(
                    role_none, sample_area,
                    np.ascontiguousarray(back_ids, dtype=np.int32),
                    parametric_surface={"kind": kind_neural},
                )
                n_interior += int(back_ids.size)
        if n_interior:
            print(f"[assembly] LUT absorbers: {n_interior} tris", flush=True)

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

        # All geometry comes from the optics model — the canonical physical source.
        # Mesh centroids and scene_lenses are only used as a fallback when optics
        # is not set (legacy load_payload_files path without a CompoundLens).
        if self.optics is not None:
            x_ent   = float(self.optics.side("front").x_pos)
            x_exit  = float(self.optics.side("back").x_pos)
            r_lens  = float(self.optics.side("front").radius)
            refr = [e for e in self.optics.elements if hasattr(e, "R_curvature")]
            roc_front = float(refr[0].R_curvature)  if refr else 0.0
            roc_back  = float(refr[-1].R_curvature) if refr else 0.0
        else:
            x_ent     = float(np.mean(tri_centroids[front_ids, 0])) \
                        if tri_centroids is not None else 0.0
            x_exit    = float(np.mean(tri_centroids[back_ids, 0])) \
                        if (tri_centroids is not None and int(back_ids.size) > 0) else x_ent
            r_lens    = 0.0
            roc_front = float(scene_lenses[0].radius_front)  if scene_lenses else 0.0
            roc_back  = float(scene_lenses[-1].radius_back)  if scene_lenses else 0.0

        # Entrance surface — forward MLP (side=0)
        p = self._fwd_payload.copy()
        p[5]  = np.float32(x_ent);   p[6]  = np.float32(x_exit)
        p[9]  = np.float32(0.0);     p[10] = np.float32(r_lens)
        p[11] = np.float32(roc_front)
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

        # Exit surface — backward MLP, or canonical PLENS fallback when no bwd MLP payload yet.
        if int(back_ids.size) > 0:
            if self._bwd_payload is not None:
                p_bwd = self._bwd_payload.copy()
                p_bwd[5]  = np.float32(x_exit);  p_bwd[6]  = np.float32(x_ent)
                p_bwd[9]  = np.float32(0.0);     p_bwd[10] = np.float32(r_lens)
                p_bwd[11] = np.float32(roc_back)
                p_bwd[12] = np.float32(0.0);     p_bwd[13] = np.float32(0)
                p_bwd[14] = np.float32(0.0);     p_bwd[15] = np.float32(1)  # side = bwd

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
            elif self.optics is not None:
                import _spectral_kernels as _sk2
                kind_p = int(getattr(_sk2, "TRI_PARAM_SURFACE_PARAMETRIC_LENS", 5))
                bwd_p = self.build_parametric_payload_backward()
                self._exit_gid = int(tracer.register_tri_group(
                    role_none, sample_area,
                    np.ascontiguousarray(back_ids, dtype=np.int32),
                    parametric_surface={"kind": kind_p, "payload_f32": bwd_p},
                ))
                print(
                    f"[assembly] MLP exit gid={self._exit_gid} (backward PLENS fallback)",
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
        # Back face of last lens group — canonical teleport payload. Runtime
        # handlers reverse traversal for sensor→scene rays.
        if int(back_ids.size) > 0:
            bwd_payload = self.build_parametric_payload_backward()
            self._exit_gid = int(tracer.register_tri_group(
                role_none, sample_area,
                np.ascontiguousarray(back_ids, dtype=np.int32),
                parametric_surface={"kind": kind_parametric, "payload_f32": bwd_payload},
            ))
            print(
                f"[assembly] PARAMETRIC exit gid={self._exit_gid}"
                f"  n_surfs={int(bwd_payload[1])} (backward)",
                flush=True,
            )

        if n_interior:
            print(f"[assembly] PARAMETRIC absorbers: {n_interior} tris", flush=True)

    # ── Progressive parametric → LUT → MLP refinement ───────────────────────

    def start_progressive_refinement(
        self,
        *,
        n_u: int   = 32,
        n_v: int   = 32,
        n_a: int   = 8,
        n_b: int   = 8,
        batch_size: int   = 4096,
        lut_min_cells_frac: float = 0.25,
        mlp_min_samples:    int   = 100_000,
        mlp_epochs:         int   = 40,
        lut_refine_interval: int  = 20,
    ) -> None:
        """Stub — Python CPU ray tracer removed; progressive baking is no-op."""
        return

    def stop_progressive_refinement(self) -> None:
        """No-op — progressive refinement was removed with the Python CPU tracer."""
        pass

    def poll_pending_mode_switch(self, tracer) -> Optional[str]:
        """Apply any pending LUT or MLP transition; must be called on the main thread.

        Returns "LUT", "MLP", or None.
        """
        import _spectral_kernels as _sk
        sa = int(getattr(_sk, "TRI_GROUP_SAMPLE_AREA",             1))
        kn = int(getattr(_sk, "TRI_PARAM_SURFACE_NEURAL_ASSEMBLY", 4))

        fwd = self._pending_mlp_fwd_payload
        if fwd is not None:
            self._pending_mlp_fwd_payload = None
            bwd = self._pending_mlp_bwd_payload
            self._pending_mlp_bwd_payload = None
            self._fwd_payload = fwd
            self._bwd_payload = bwd
            self.mode = self.MODE_MLP
            lsg = self._cached_lsg
            if lsg:
                self._register_mlp(tracer, lsg, None, None, sa, kn, 0)
            print("[assembly] → MLP mode", flush=True)
            return "MLP"

        lut = self._pending_lut_payload
        if lut is not None:
            self._pending_lut_payload = None
            self._transfer_grid = lut
            prev = self.mode
            self.mode = self.MODE_LUT
            lsg = self._cached_lsg
            if lsg:
                self._register_lut(tracer, lsg, sa, kn, 0)
            if prev != self.MODE_LUT:
                print("[assembly] → LUT mode", flush=True)
            return "LUT"

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
