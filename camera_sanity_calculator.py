"""Camera geometry and exposure sanity calculator.

This module estimates whether a proposed camera/lens placement is feasible
without extreme geometry changes.  It computes thin-lens plane locations,
checks tilt/shift and sensor travel margins, and reports PASS/WARN/FAIL.

Design goals:
- Allow generous creative setups (tilt-shift, trick perspective, mild blur).
- Warn when the setup is likely aesthetic blur rather than critical focus.
- Fail only when geometry is far outside practical correction range.
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

    sensor_plane_z_m: float = -0.050
    sensor_travel_min_m: float = -0.110
    sensor_travel_max_m: float = -0.015

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
    thin_lens_residual_diopter: float
    coc_degree: float
    ev_degree: float
    tilt_degree: float
    shift_degree: float


def _configured_ev100(f_number: float, exposure_time_s: float, iso: float) -> float:
    n2_over_t = (f_number * f_number) / max(exposure_time_s, 1e-9)
    return math.log2(max(n2_over_t, 1e-12)) - math.log2(max(iso / 100.0, 1e-12))


def _thin_lens_image_distance_m(focal_length_m: float, focus_distance_m: float) -> float:
    # 1/f = 1/do + 1/di  => di = 1 / (1/f - 1/do)
    denom = (1.0 / focal_length_m) - (1.0 / focus_distance_m)
    if abs(denom) < 1e-12:
        return float("inf")
    return 1.0 / denom


def _estimate_coc_um(focal_length_m: float, f_number: float, image_required_m: float, sensor_m: float) -> float:
    # Thin-lens defocus approximation: c ~= D * |delta| / di
    # D = f / N, delta = sensor - ideal image plane.
    pupil_d = focal_length_m / max(f_number, 1e-6)
    delta = abs(sensor_m - image_required_m)
    coc_m = pupil_d * delta / max(abs(image_required_m), 1e-9)
    return coc_m * 1e6


def evaluate_camera_sanity(cfg: CameraSanityInput) -> CameraSanityReport:
    warnings: List[str] = []
    failures: List[str] = []

    f_m = cfg.focal_length_mm * 1e-3
    thickness_m = cfg.lens_thickness_mm * 1e-3

    if f_m <= 0.0:
        raise ValueError("focal_length_mm must be > 0")
    if cfg.focus_distance_m <= f_m * 1.001:
        failures.append(
            "Focus distance is at/inside focal length; required image plane is non-physical for this lens geometry."
        )

    image_dist_m = _thin_lens_image_distance_m(f_m, max(cfg.focus_distance_m, f_m * 1.001))

    # Convention: lens center at z=0, world/object side +z, sensor side -z.
    image_plane_required_z_m = cfg.lens_center_z_m - image_dist_m
    focal_plane_z_m = cfg.lens_center_z_m + cfg.focus_distance_m
    aperture_plane_z_m = cfg.lens_center_z_m + cfg.aperture_plane_offset_m
    lens_front_plane_z_m = cfg.lens_center_z_m + 0.5 * thickness_m
    lens_rear_plane_z_m = cfg.lens_center_z_m - 0.5 * thickness_m

    if not (cfg.sensor_travel_min_m <= image_plane_required_z_m <= cfg.sensor_travel_max_m):
        failures.append(
            "Required image plane lies outside sensor travel range; setup needs substantial geometry relocation."
        )

    sensor_adjust_mm = (image_plane_required_z_m - cfg.sensor_plane_z_m) * 1e3
    if abs(sensor_adjust_mm) > 25.0:
        failures.append(
            "Sensor must move more than 25 mm to focus; this is beyond generous tilt-shift/trick-photography margins."
        )
    elif abs(sensor_adjust_mm) > max(2.0, cfg.acceptable_focus_margin_mm):
        warnings.append("Focus requires noticeable sensor/lens rail adjustment.")

    tilt_abs = abs(cfg.tilt_deg)
    shift_abs = abs(cfg.shift_mm)
    if tilt_abs > 20.0:
        failures.append("Tilt exceeds 20 deg; practical focus plane correction is unlikely without redesign.")
    elif tilt_abs > 8.0:
        warnings.append("Tilt is high; expect perspective effects and possible edge softness.")

    if shift_abs > 45.0:
        failures.append("Shift exceeds 45 mm; image circle coverage is unlikely for this geometry.")
    elif shift_abs > 18.0:
        warnings.append("Large shift requested; check vignetting and edge illumination.")

    coc_um = _estimate_coc_um(f_m, cfg.f_number, image_plane_required_z_m, cfg.sensor_plane_z_m)
    if coc_um >= cfg.fail_blur_coc_um:
        failures.append("Predicted blur is extreme; geometry likely needs major repositioning.")
    elif coc_um >= cfg.warn_blur_coc_um:
        warnings.append("Predicted blur is intentionally soft/aesthetic rather than critically sharp.")

    req_ev = cfg.scene_ev100
    cfg_ev = _configured_ev100(cfg.f_number, cfg.exposure_time_s, cfg.iso)
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
        focal_plane_z_m=focal_plane_z_m,
        image_plane_required_z_m=image_plane_required_z_m,
        sensor_plane_z_m=cfg.sensor_plane_z_m,
        aperture_plane_z_m=aperture_plane_z_m,
        lens_front_plane_z_m=lens_front_plane_z_m,
        lens_rear_plane_z_m=lens_rear_plane_z_m,
    )
    exposure = ExposureEstimate(
        required_ev100=req_ev,
        configured_ev100=cfg_ev,
        ev_error_stops=ev_err,
    )

    apparent_focus_m = (
        float(cfg.apparent_focus_distance_m)
        if cfg.apparent_focus_distance_m is not None
        else float(cfg.focus_distance_m)
    )
    apparent_sensor_z_m = (
        float(cfg.apparent_sensor_plane_z_m)
        if cfg.apparent_sensor_plane_z_m is not None
        else float(cfg.sensor_plane_z_m)
    )

    di_apparent = max(1.0e-9, abs(cfg.lens_center_z_m - apparent_sensor_z_m))
    do_apparent = max(1.0e-9, abs(apparent_focus_m))
    thin_residual = abs((1.0 / f_m) - (1.0 / do_apparent) - (1.0 / di_apparent))

    sensor_deg = abs(sensor_adjust_mm) / max(2.0, cfg.acceptable_focus_margin_mm)
    focus_delta_m = abs(apparent_focus_m - cfg.focus_distance_m)
    focus_deg = focus_delta_m / max(0.1, cfg.focus_distance_m)
    coc_deg = coc_um / max(1.0, cfg.fail_blur_coc_um)
    ev_deg = abs(ev_err) / 5.0
    tilt_deg = abs(cfg.tilt_deg) / 20.0
    shift_deg = abs(cfg.shift_mm) / 45.0
    residual_deg = thin_residual / 2.0
    overall = (
        0.26 * sensor_deg
        + 0.16 * focus_deg
        + 0.18 * coc_deg
        + 0.16 * ev_deg
        + 0.08 * tilt_deg
        + 0.08 * shift_deg
        + 0.08 * residual_deg
    )
    err = CameraErrorDegree(
        overall=float(overall),
        sensor_plane_delta_mm=float(sensor_adjust_mm),
        sensor_plane_degree=float(sensor_deg),
        focus_distance_delta_m=float(focus_delta_m),
        focus_distance_degree=float(focus_deg),
        thin_lens_residual_diopter=float(thin_residual),
        coc_degree=float(coc_deg),
        ev_degree=float(ev_deg),
        tilt_degree=float(tilt_deg),
        shift_degree=float(shift_deg),
    )

    return CameraSanityReport(
        status=status,
        planes=planes,
        exposure=exposure,
        circle_of_confusion_um=coc_um,
        sensor_adjustment_needed_mm=sensor_adjust_mm,
        error_degree=err,
        warnings=warnings,
        failures=failures,
    )
