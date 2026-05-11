"""Camera geometry and exposure sanity calculator.

This model supports articulated rails for sensor plate, aperture plate,
and front/rear thin-lens elements. The effective focal length emerges from
front/rear element spacing instead of being fixed as an invariant target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Literal, Optional


Status = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class CameraSanityInput:
    focal_length_mm: float
    focus_distance_m: float
    f_number: float
    exposure_time_s: float
    iso: float
    scene_ev100: float
    apparent_focus_distance_m: Optional[float] = None
    apparent_sensor_plane_z_m: Optional[float] = None

    lens_center_z_m: float = 0.0
    aperture_plane_offset_m: float = 0.0
    lens_thickness_mm: float = 8.0

    lens_front_focal_mm: float = 100.0
    lens_rear_focal_mm: float = 100.0
    lens_front_plane_z_m: Optional[float] = None
    lens_rear_plane_z_m: Optional[float] = None
    lens_front_shift_x_mm: float = 0.0
    lens_front_shift_y_mm: float = 0.0
    lens_front_tilt_x_deg: float = 0.0
    lens_front_tilt_y_deg: float = 0.0

    aperture_rail_z_m: Optional[float] = None
    aperture_shift_x_mm: float = 0.0
    aperture_shift_y_mm: float = 0.0
    aperture_radius_m: Optional[float] = None

    sensor_plane_z_m: float = -0.050
    sensor_travel_min_m: float = -0.110
    sensor_travel_max_m: float = -0.015

    sensor_corner_tl_mm: float = 0.0
    sensor_corner_tr_mm: float = 0.0
    sensor_corner_bl_mm: float = 0.0
    sensor_corner_br_mm: float = 0.0
    sensor_shift_x_mm: float = 0.0
    sensor_shift_y_mm: float = 0.0

    tilt_deg: float = 0.0
    shift_mm: float = 0.0

    acceptable_focus_margin_mm: float = 1.2
    warn_blur_coc_um: float = 18.0
    fail_blur_coc_um: float = 55.0


@dataclass(frozen=True)
class PlaneSolution:
    focal_plane_z_m: float
    image_plane_required_z_m: float
    sensor_plane_z_m: float
    aperture_plane_z_m: float
    lens_front_plane_z_m: float
    lens_rear_plane_z_m: float
    effective_focal_m: float


@dataclass(frozen=True)
class ExposureEstimate:
    required_ev100: float
    configured_ev100: float
    ev_error_stops: float


@dataclass
class CameraSanityReport:
    status: Status
    planes: PlaneSolution
    exposure: ExposureEstimate
    circle_of_confusion_um: float
    sensor_adjustment_needed_mm: float
    error_degree: "CameraErrorDegree"
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CameraErrorDegree:
    overall: float
    sensor_plane_delta_mm: float
    sensor_plane_degree: float
    focus_distance_delta_m: float
    focus_distance_degree: float
    thin_lens_target_sensor_z_m: float
    pinhole_target_sensor_z_m: float
    pinhole_comparison_mm: float
    pinhole_comparison_degree: float
    thin_lens_residual_diopter: float
    coc_degree: float
    ev_degree: float
    tilt_degree: float
    shift_degree: float


def _configured_ev100(f_number: float, exposure_time_s: float, iso: float) -> float:
    n2_over_t = (f_number * f_number) / max(exposure_time_s, 1e-9)
    return math.log2(max(n2_over_t, 1e-12)) - math.log2(max(iso / 100.0, 1e-12))


def _thin_lens_image_distance_m(focal_length_m: float, focus_distance_m: float) -> float:
    denom = (1.0 / focal_length_m) - (1.0 / focus_distance_m)
    if abs(denom) < 1e-12:
        return float("inf")
    return 1.0 / denom


def _estimate_coc_um(focal_length_m: float, f_number: float, image_required_m: float, sensor_m: float) -> float:
    pupil_d = focal_length_m / max(f_number, 1e-6)
    delta = abs(sensor_m - image_required_m)
    coc_m = pupil_d * delta / max(abs(image_required_m), 1e-9)
    return coc_m * 1e6


def _effective_focal_from_group(cfg: CameraSanityInput) -> tuple[float, float, float]:
    thickness_m = cfg.lens_thickness_mm * 1.0e-3
    zf = float(cfg.lens_front_plane_z_m) if cfg.lens_front_plane_z_m is not None else float(cfg.lens_center_z_m + 0.5 * thickness_m)
    zr = float(cfg.lens_rear_plane_z_m) if cfg.lens_rear_plane_z_m is not None else float(cfg.lens_center_z_m - 0.5 * thickness_m)
    sep = max(1.0e-6, zf - zr)
    f1 = max(float(cfg.lens_front_focal_mm) * 1.0e-3, 1.0e-6)
    f2 = max(float(cfg.lens_rear_focal_mm) * 1.0e-3, 1.0e-6)
    denom = (1.0 / f1) + (1.0 / f2) - (sep / (f1 * f2))
    if denom <= 1.0e-9:
        return max(float(cfg.focal_length_mm) * 1.0e-3, 1.0e-6), zf, zr
    return 1.0 / denom, zf, zr


def evaluate_camera_sanity(cfg: CameraSanityInput) -> CameraSanityReport:
    warnings: List[str] = []
    failures: List[str] = []

    f_nom_m = max(float(cfg.focal_length_mm) * 1.0e-3, 1.0e-6)
    if cfg.focus_distance_m <= f_nom_m * 1.001:
        failures.append("Focus distance is at/inside focal length; required image plane is non-physical for this lens geometry.")

    f_eff_m, z_front_m, z_rear_m = _effective_focal_from_group(cfg)
    if z_front_m <= z_rear_m:
        failures.append("Lens front rail is behind lens rear rail; rail stack order is invalid.")

    image_dist_m = _thin_lens_image_distance_m(f_eff_m, max(float(cfg.focus_distance_m), f_eff_m * 1.001))

    aperture_plane_z_m = float(cfg.aperture_rail_z_m) if cfg.aperture_rail_z_m is not None else float(cfg.lens_center_z_m + cfg.aperture_plane_offset_m)
    image_plane_required_z_m = z_rear_m - image_dist_m
    focal_plane_z_m = cfg.lens_center_z_m + cfg.focus_distance_m

    c_tl = float(cfg.sensor_corner_tl_mm) * 1.0e-3
    c_tr = float(cfg.sensor_corner_tr_mm) * 1.0e-3
    c_bl = float(cfg.sensor_corner_bl_mm) * 1.0e-3
    c_br = float(cfg.sensor_corner_br_mm) * 1.0e-3
    z_tl = float(cfg.sensor_plane_z_m) + c_tl
    z_tr = float(cfg.sensor_plane_z_m) + c_tr
    z_bl = float(cfg.sensor_plane_z_m) + c_bl
    z_br = float(cfg.sensor_plane_z_m) + c_br
    sensor_plane_effective_z_m = 0.25 * (z_tl + z_tr + z_bl + z_br)

    sensor_w_m = 0.036
    sensor_h_m = 0.024
    dzdx = ((z_tr + z_br) - (z_tl + z_bl)) / (2.0 * sensor_w_m)
    dzdy = ((z_bl + z_br) - (z_tl + z_tr)) / (2.0 * sensor_h_m)
    implied_tilt_deg = math.degrees(math.atan(math.sqrt(dzdx * dzdx + dzdy * dzdy)))

    if not (cfg.sensor_travel_min_m <= image_plane_required_z_m <= cfg.sensor_travel_max_m):
        failures.append("Required image plane lies outside sensor travel range; setup needs substantial geometry relocation.")

    sensor_adjust_mm = (image_plane_required_z_m - sensor_plane_effective_z_m) * 1e3
    if abs(sensor_adjust_mm) > 25.0:
        failures.append("Sensor must move more than 25 mm to focus; this is beyond generous tilt-shift/trick-photography margins.")
    elif abs(sensor_adjust_mm) > max(2.0, cfg.acceptable_focus_margin_mm):
        warnings.append("Focus requires noticeable sensor/lens rail adjustment.")

    front_tilt_abs = math.hypot(float(cfg.lens_front_tilt_x_deg), float(cfg.lens_front_tilt_y_deg))
    front_shift_abs = math.hypot(float(cfg.lens_front_shift_x_mm), float(cfg.lens_front_shift_y_mm))
    tilt_abs = max(abs(float(cfg.tilt_deg)), abs(implied_tilt_deg), front_tilt_abs)
    shift_abs = max(
        abs(float(cfg.shift_mm)),
        math.hypot(float(cfg.sensor_shift_x_mm), float(cfg.sensor_shift_y_mm)),
        front_shift_abs,
    )
    aperture_shift_abs = math.hypot(float(cfg.aperture_shift_x_mm), float(cfg.aperture_shift_y_mm))
    if tilt_abs > 20.0:
        failures.append("Tilt exceeds 20 deg; practical focus plane correction is unlikely without redesign.")
    elif tilt_abs > 8.0:
        warnings.append("Tilt is high; expect perspective effects and possible edge softness.")

    if shift_abs > 45.0:
        failures.append("Shift exceeds 45 mm; image circle coverage is unlikely for this geometry.")
    elif shift_abs > 18.0:
        warnings.append("Large shift requested; check vignetting and edge illumination.")

    if aperture_shift_abs > 45.0:
        failures.append("Aperture rail lateral shift exceeds 45 mm; lens stack decenter is likely non-viable.")
    elif aperture_shift_abs > 18.0:
        warnings.append("Large aperture rail lateral shift; verify decenter tolerances.")

    f_number_eff = float(cfg.f_number)
    if cfg.aperture_radius_m is not None and cfg.aperture_radius_m > 0.0:
        f_number_eff = float(max(f_eff_m / max(2.0 * float(cfg.aperture_radius_m), 1.0e-9), 1.0e-3))

    coc_um = _estimate_coc_um(f_eff_m, f_number_eff, image_plane_required_z_m, sensor_plane_effective_z_m)
    if coc_um >= cfg.fail_blur_coc_um:
        failures.append("Predicted blur is extreme; geometry likely needs major repositioning.")
    elif coc_um >= cfg.warn_blur_coc_um:
        warnings.append("Predicted blur is intentionally soft/aesthetic rather than critically sharp.")

    req_ev = cfg.scene_ev100
    cfg_ev = _configured_ev100(f_number_eff, cfg.exposure_time_s, cfg.iso)
    ev_err = cfg_ev - req_ev
    if abs(ev_err) > 5.0:
        failures.append("Exposure mismatch exceeds 5 stops; placement/settings are not realistically recoverable.")
    elif abs(ev_err) > 2.0:
        warnings.append("Exposure mismatch is large; consider shutter/aperture/ISO rebalance.")

    status: Status
    if failures:
        status = "fail"
    elif warnings:
        status = "warn"
    else:
        status = "pass"

    planes = PlaneSolution(
        focal_plane_z_m=float(focal_plane_z_m),
        image_plane_required_z_m=float(image_plane_required_z_m),
        sensor_plane_z_m=float(sensor_plane_effective_z_m),
        aperture_plane_z_m=float(aperture_plane_z_m),
        lens_front_plane_z_m=float(z_front_m),
        lens_rear_plane_z_m=float(z_rear_m),
        effective_focal_m=float(f_eff_m),
    )
    exposure = ExposureEstimate(
        required_ev100=float(req_ev),
        configured_ev100=float(cfg_ev),
        ev_error_stops=float(ev_err),
    )

    apparent_focus_m = float(cfg.apparent_focus_distance_m) if cfg.apparent_focus_distance_m is not None else float(cfg.focus_distance_m)
    focus_delta_m = abs(apparent_focus_m - float(cfg.focus_distance_m))

    pinhole_sensor_z_m = aperture_plane_z_m - f_eff_m
    pinhole_delta_mm = abs(sensor_plane_effective_z_m - pinhole_sensor_z_m) * 1e3

    sensor_deg = pinhole_delta_mm / max(2.0, cfg.acceptable_focus_margin_mm)
    focus_deg = 0.0
    pinhole_deg = pinhole_delta_mm / max(2.0, cfg.acceptable_focus_margin_mm)
    coc_deg = coc_um / max(1.0, cfg.fail_blur_coc_um)
    ev_deg = abs(ev_err) / 5.0
    tilt_deg = tilt_abs / 20.0
    shift_deg = shift_abs / 45.0
    residual_deg = abs((1.0 / max(f_eff_m, 1.0e-9)) - (1.0 / max(apparent_focus_m, 1.0e-9)) - (1.0 / max(abs(sensor_plane_effective_z_m), 1.0e-9)))

    overall = (
        0.52 * pinhole_deg
        + 0.20 * coc_deg
        + 0.14 * ev_deg
        + 0.08 * tilt_deg
        + 0.06 * shift_deg
    )

    err = CameraErrorDegree(
        overall=float(overall),
        sensor_plane_delta_mm=float(sensor_adjust_mm),
        sensor_plane_degree=float(sensor_deg),
        focus_distance_delta_m=float(focus_delta_m),
        focus_distance_degree=float(focus_deg),
        thin_lens_target_sensor_z_m=float(image_plane_required_z_m),
        pinhole_target_sensor_z_m=float(pinhole_sensor_z_m),
        pinhole_comparison_mm=float(pinhole_delta_mm),
        pinhole_comparison_degree=float(pinhole_deg),
        thin_lens_residual_diopter=float(residual_deg),
        coc_degree=float(coc_deg),
        ev_degree=float(ev_deg),
        tilt_degree=float(tilt_deg),
        shift_degree=float(shift_deg),
    )

    return CameraSanityReport(
        status=status,
        planes=planes,
        exposure=exposure,
        circle_of_confusion_um=float(coc_um),
        sensor_adjustment_needed_mm=float(sensor_adjust_mm),
        error_degree=err,
        warnings=warnings,
        failures=failures,
    )
