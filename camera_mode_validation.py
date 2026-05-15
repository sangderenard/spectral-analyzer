"""Camera mode validation and telemetry for optical transport tiers.

This module enforces the separation of concerns across the six optical
transport tiers (oracle pinhole reference → baked transform function) and
ensures that each render mode consumes only the degrees of freedom it supports.

See CAMERA_SYSTEM_REFACTORING.md for the full architectural specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from bdpt_integrator import (
    CAMERA_MODE_ORACLE_PINHOLE_REFERENCE,
    CAMERA_MODE_PHYSICAL_PINHOLE,
    CAMERA_MODE_APERTURE_CONE,
    CAMERA_MODE_THIN_LENS_GEOMETRIC,
    CAMERA_MODE_GEOMETRIC_ASSEMBLY,
    CAMERA_MODE_WAVE_ASSEMBLY,
    CAMERA_MODE_BAKED_TRANSFORM,
)


# ─────────────────────────────────────────────────────────────────────────────
# Event and energy telemetry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraEventTelemetry:
    """Optical event counts and energy budget for one frame.

    These fields validate that active camera components are actually
    participating in the ray transport, not silently bypassed.
    """
    # Ray dispatch and termination
    rays_launched: int = 0
    rays_blocked_by_stop: int = 0
    rays_hit_lens_surface: int = 0
    rays_refracted: int = 0
    rays_reflected: int = 0
    rays_total_internal_reflection: int = 0
    rays_entered_wave_region: int = 0
    rays_exited_wave_region: int = 0
    rays_deposited_sensor: int = 0
    rays_out_of_domain: int = 0
    rays_fell_back_to_full_solve: int = 0

    # Energy accounting
    energy_in: float = 0.0
    energy_out: float = 0.0
    energy_absorbed: float = 0.0
    energy_blocked: float = 0.0

    # Optical quality metrics
    mean_phase_error: float = 0.0
    mean_focus_error: float = 0.0

    def to_dict(self) -> dict:
        """Export as JSON-serializable dict."""
        return {
            "rays_launched": int(self.rays_launched),
            "rays_blocked_by_stop": int(self.rays_blocked_by_stop),
            "rays_hit_lens_surface": int(self.rays_hit_lens_surface),
            "rays_refracted": int(self.rays_refracted),
            "rays_reflected": int(self.rays_reflected),
            "rays_total_internal_reflection": int(self.rays_total_internal_reflection),
            "rays_entered_wave_region": int(self.rays_entered_wave_region),
            "rays_exited_wave_region": int(self.rays_exited_wave_region),
            "rays_deposited_sensor": int(self.rays_deposited_sensor),
            "rays_out_of_domain": int(self.rays_out_of_domain),
            "rays_fell_back_to_full_solve": int(self.rays_fell_back_to_full_solve),
            "energy_in": float(self.energy_in),
            "energy_out": float(self.energy_out),
            "energy_absorbed": float(self.energy_absorbed),
            "energy_blocked": float(self.energy_blocked),
            "mean_phase_error": float(self.mean_phase_error),
            "mean_focus_error": float(self.mean_focus_error),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mode validation constraints
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraModeConstraints:
    """Define which solver DOFs and optical properties are valid per mode."""

    mode: int
    mode_name: str

    # Aperture constraints
    allows_nonzero_aperture_radius: bool = False
    allows_aperture_stop_geometry: bool = False
    aperture_samples_must_equal_one: bool = False

    # Lens constraints
    lens_event_count_must_be_zero: bool = False
    lens_event_count_must_be_nonzero: bool = False
    requires_effective_focal_m: bool = False
    requires_focus_distance_m: bool = False

    # Solver DOF consumption
    allows_sensor_shift: bool = False
    allows_aperture_shift: bool = False
    allows_lens_shift: bool = False
    allows_lens_tilt: bool = False
    allows_sensor_tilt: bool = False

    # Photon accounting
    applies_photon_penalty: bool = False
    produces_oracle_reference: bool = False

    def validate(
        self,
        n_aperture_samples: int,
        aperture_radius_m: float,
        has_aperture_stop: bool,
        lens_event_count: int,
        effective_focal_m: float,
        focus_distance_m: float,
        sensor_shift_xy_mm: tuple[float, float],
        aperture_shift_xy_mm: tuple[float, float],
        lens_shift_xy_mm: tuple[float, float],
        lens_tilt_xy_deg: tuple[float, float],
        sensor_tilt_mm: float,
    ) -> Optional[str]:
        """Validate frame configuration against mode constraints.

        Returns: None if valid, or an error message explaining what constraint
                 was violated.
        """
        # Aperture samples
        if self.aperture_samples_must_equal_one and n_aperture_samples != 1:
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"aperture_samples must be 1, got {n_aperture_samples}"
            )

        # Aperture radius
        if not self.allows_nonzero_aperture_radius and aperture_radius_m > 1e-6:
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"aperture_radius must be zero, got {aperture_radius_m:.6e} m"
            )

        # Aperture stop geometry
        if not self.allows_aperture_stop_geometry and has_aperture_stop:
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"aperture stop geometry not supported in this mode"
            )

        # Lens events
        if self.lens_event_count_must_be_zero and lens_event_count > 0:
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"lens_event_count must be 0, got {lens_event_count}"
            )
        if self.lens_event_count_must_be_nonzero and lens_event_count == 0:
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"lens_event_count must be > 0 (lens not participating)"
            )

        # Focal length / focus distance
        if self.requires_effective_focal_m and effective_focal_m <= 0:
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"requires positive effective_focal_m, got {effective_focal_m:.6e}"
            )
        if self.requires_focus_distance_m and focus_distance_m <= 0:
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"requires positive focus_distance_m, got {focus_distance_m:.6e}"
            )

        # Solver DOF consumption
        if not self.allows_sensor_shift and (abs(sensor_shift_xy_mm[0]) > 1e-3 or abs(sensor_shift_xy_mm[1]) > 1e-3):
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"sensor_shift not supported, got ({sensor_shift_xy_mm[0]:.6e}, {sensor_shift_xy_mm[1]:.6e}) mm"
            )
        if not self.allows_aperture_shift and (abs(aperture_shift_xy_mm[0]) > 1e-3 or abs(aperture_shift_xy_mm[1]) > 1e-3):
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"aperture_shift not supported, got ({aperture_shift_xy_mm[0]:.6e}, {aperture_shift_xy_mm[1]:.6e}) mm"
            )
        if not self.allows_lens_shift and (abs(lens_shift_xy_mm[0]) > 1e-3 or abs(lens_shift_xy_mm[1]) > 1e-3):
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"lens_shift not supported, got ({lens_shift_xy_mm[0]:.6e}, {lens_shift_xy_mm[1]:.6e}) mm"
            )
        if not self.allows_lens_tilt and (abs(lens_tilt_xy_deg[0]) > 0.01 or abs(lens_tilt_xy_deg[1]) > 0.01):
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"lens_tilt not supported, got ({lens_tilt_xy_deg[0]:.6e}, {lens_tilt_xy_deg[1]:.6e}) deg"
            )
        if not self.allows_sensor_tilt and abs(sensor_tilt_mm) > 1e-3:
            return (
                f"{self.mode_name} (mode {self.mode}): "
                f"sensor_tilt not supported, got {sensor_tilt_mm:.6e} mm"
            )

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Mode constraint definitions (one per tier)
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_MODE_CONSTRAINTS = {
    CAMERA_MODE_ORACLE_PINHOLE_REFERENCE: CameraModeConstraints(
        mode=CAMERA_MODE_ORACLE_PINHOLE_REFERENCE,
        mode_name="ORACLE_PINHOLE_REFERENCE",
        aperture_samples_must_equal_one=True,  # 1 ray per pixel
        lens_event_count_must_be_zero=True,    # No lens interaction
        produces_oracle_reference=True,
    ),

    CAMERA_MODE_PHYSICAL_PINHOLE: CameraModeConstraints(
        mode=CAMERA_MODE_PHYSICAL_PINHOLE,
        mode_name="PHYSICAL_PINHOLE",
        allows_nonzero_aperture_radius=True,
        lens_event_count_must_be_zero=True,    # No lens, just tiny aperture
        applies_photon_penalty=True,           # Photon-limited
    ),

    CAMERA_MODE_APERTURE_CONE: CameraModeConstraints(
        mode=CAMERA_MODE_APERTURE_CONE,
        mode_name="APERTURE_CONE",
        allows_nonzero_aperture_radius=True,
        allows_aperture_stop_geometry=True,
        lens_event_count_must_be_zero=True,    # No lens bending
    ),

    CAMERA_MODE_THIN_LENS_GEOMETRIC: CameraModeConstraints(
        mode=CAMERA_MODE_THIN_LENS_GEOMETRIC,
        mode_name="THIN_LENS_GEOMETRIC",
        allows_nonzero_aperture_radius=True,
        allows_aperture_stop_geometry=True,
        lens_event_count_must_be_nonzero=True,  # Lens must participate
        requires_effective_focal_m=True,
        requires_focus_distance_m=True,
    ),

    CAMERA_MODE_GEOMETRIC_ASSEMBLY: CameraModeConstraints(
        mode=CAMERA_MODE_GEOMETRIC_ASSEMBLY,
        mode_name="GEOMETRIC_ASSEMBLY",
        allows_nonzero_aperture_radius=True,
        allows_aperture_stop_geometry=True,
        lens_event_count_must_be_nonzero=True,  # Full element chain must participate
        requires_effective_focal_m=True,
        requires_focus_distance_m=True,
        allows_sensor_shift=True,
        allows_aperture_shift=True,
        allows_lens_shift=True,
        allows_lens_tilt=True,
        allows_sensor_tilt=True,
    ),

    CAMERA_MODE_WAVE_ASSEMBLY: CameraModeConstraints(
        mode=CAMERA_MODE_WAVE_ASSEMBLY,
        mode_name="WAVE_ASSEMBLY",
        allows_nonzero_aperture_radius=True,
        allows_aperture_stop_geometry=True,
        lens_event_count_must_be_nonzero=True,  # Wave handoff requires optical surfaces
        requires_effective_focal_m=True,
        requires_focus_distance_m=True,
        allows_sensor_shift=True,
        allows_aperture_shift=True,
        allows_lens_shift=True,
        allows_lens_tilt=True,
        allows_sensor_tilt=True,
    ),

    CAMERA_MODE_BAKED_TRANSFORM: CameraModeConstraints(
        mode=CAMERA_MODE_BAKED_TRANSFORM,
        mode_name="BAKED_TRANSFORM",
        allows_nonzero_aperture_radius=True,
        allows_aperture_stop_geometry=True,
        lens_event_count_must_be_nonzero=True,
        requires_effective_focal_m=True,
        requires_focus_distance_m=True,
        allows_sensor_shift=True,
        allows_aperture_shift=True,
        allows_lens_shift=True,
        allows_lens_tilt=True,
        allows_sensor_tilt=True,
    ),
}


def get_mode_constraints(camera_mode: int) -> CameraModeConstraints:
    """Fetch constraint object for the given camera mode.

    Raises ValueError if the mode is not registered.
    """
    if camera_mode not in CAMERA_MODE_CONSTRAINTS:
        raise ValueError(
            f"Unknown camera mode {camera_mode}. "
            f"Valid modes: {sorted(CAMERA_MODE_CONSTRAINTS.keys())}"
        )
    return CAMERA_MODE_CONSTRAINTS[camera_mode]


def get_mode_name(camera_mode: int) -> str:
    """Return the human-readable name for the camera mode."""
    try:
        return get_mode_constraints(camera_mode).mode_name
    except ValueError:
        return f"UNKNOWN_MODE_{camera_mode}"


def validate_frame_configuration(
    camera_mode: int,
    n_aperture_samples: int,
    aperture_radius_m: float,
    has_aperture_stop: bool,
    lens_event_count: int,
    effective_focal_m: float,
    focus_distance_m: float,
    solved_camera_package=None,  # Optional SolvedCameraPackage
) -> Optional[str]:
    """Validate the entire frame configuration against the camera mode.

    Args:
        camera_mode: One of CAMERA_MODE_* from bdpt_integrator
        n_aperture_samples: Number of aperture samples in this frame
        aperture_radius_m: Aperture radius in meters
        has_aperture_stop: True if aperture stop geometry is registered
        lens_event_count: Observed number of lens events in this frame
        effective_focal_m: Effective focal length in meters (0 if not applicable)
        focus_distance_m: Scene focus distance in meters (0 if not applicable)
        solved_camera_package: Optional SolvedCameraPackage with DOF details

    Returns:
        None if all constraints are satisfied, or an error message describing
        the first constraint violation found.
    """
    constraints = get_mode_constraints(camera_mode)

    # Extract DOF values from solved package if available
    sensor_shift_xy = (0.0, 0.0)
    aperture_shift_xy = (0.0, 0.0)
    lens_shift_xy = (0.0, 0.0)
    lens_tilt_xy = (0.0, 0.0)
    sensor_tilt = 0.0

    if solved_camera_package is not None:
        si = solved_camera_package.sanity_input
        sensor_shift_xy = (
            abs(float(si.sensor_shift_x_mm if si.sensor_shift_x_mm is not None else 0.0)),
            abs(float(si.sensor_shift_y_mm if si.sensor_shift_y_mm is not None else 0.0)),
        )
        aperture_shift_xy = (
            abs(float(si.aperture_shift_x_mm if si.aperture_shift_x_mm is not None else 0.0)),
            abs(float(si.aperture_shift_y_mm if si.aperture_shift_y_mm is not None else 0.0)),
        )
        lens_shift_xy = (
            abs(float(si.lens_front_shift_x_mm if si.lens_front_shift_x_mm is not None else 0.0)),
            abs(float(si.lens_front_shift_y_mm if si.lens_front_shift_y_mm is not None else 0.0)),
        )
        lens_tilt_xy = (
            abs(float(si.lens_front_tilt_x_deg if si.lens_front_tilt_x_deg is not None else 0.0)),
            abs(float(si.lens_front_tilt_y_deg if si.lens_front_tilt_y_deg is not None else 0.0)),
        )
        if hasattr(si, 'sensor_corner_tl_mm') and si.sensor_corner_tl_mm is not None:
            sensor_tilt = max(
                abs(float(si.sensor_corner_tl_mm)),
                abs(float(si.sensor_corner_tr_mm if si.sensor_corner_tr_mm is not None else 0.0)),
                abs(float(si.sensor_corner_bl_mm if si.sensor_corner_bl_mm is not None else 0.0)),
                abs(float(si.sensor_corner_br_mm if si.sensor_corner_br_mm is not None else 0.0)),
            )

    return constraints.validate(
        n_aperture_samples=int(n_aperture_samples),
        aperture_radius_m=float(aperture_radius_m),
        has_aperture_stop=bool(has_aperture_stop),
        lens_event_count=int(lens_event_count),
        effective_focal_m=float(effective_focal_m),
        focus_distance_m=float(focus_distance_m),
        sensor_shift_xy_mm=sensor_shift_xy,
        aperture_shift_xy_mm=aperture_shift_xy,
        lens_shift_xy_mm=lens_shift_xy,
        lens_tilt_xy_deg=lens_tilt_xy,
        sensor_tilt_mm=sensor_tilt,
    )
