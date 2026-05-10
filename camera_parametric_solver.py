"""Parametric camera solver that iterates toward a sane camera configuration.

The solver builds a real camera object using the provided camera class and
optimizes thin-lens sanity metrics using ``camera_sanity_calculator``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable

import numpy as np

from camera_exposure_budget import CameraOptics, FilmExposure
from camera_sanity_calculator import CameraSanityInput, CameraSanityReport, evaluate_camera_sanity


@dataclass(frozen=True)
class SolvedCameraPackage:
    """Solved camera object and associated sanity diagnostics."""

    camera: Any
    sanity_input: CameraSanityInput
    sanity_report: CameraSanityReport
    iterations: int


def _ev100_from_settings(f_number: float, exposure_time_s: float, iso: float) -> float:
    n2_over_t = (f_number * f_number) / max(exposure_time_s, 1.0e-9)
    return math.log2(max(n2_over_t, 1.0e-12)) - math.log2(max(iso / 100.0, 1.0e-12))


def _status_rank(status: str) -> int:
    if status == "pass":
        return 0
    if status == "warn":
        return 1
    return 2


def _objective(report: CameraSanityReport) -> tuple[float, float, float, int, int]:
    # Order by status first, then by blur, sensor adjustment, EV mismatch,
    # and finally by warning/failure counts.
    return (
        float(_status_rank(report.status)),
        float(report.error_degree.overall),
        float(report.circle_of_confusion_um),
        abs(float(report.sensor_adjustment_needed_mm)),
        int(len(report.failures) * 16 + len(report.warnings)),
    )


def _build_input(
    *,
    optics: CameraOptics,
    film: FilmExposure,
    focus_distance_m: float,
    sensor_plane_z_m: float,
    scene_ev100: float,
    apparent_focus_distance_m: float,
    apparent_sensor_plane_z_m: float,
    tilt_deg: float,
    shift_mm: float,
) -> CameraSanityInput:
    return CameraSanityInput(
        focal_length_mm=float(optics.focal_mm),
        focus_distance_m=float(max(focus_distance_m, 0.05)),
        f_number=float(max(optics.f_number, 0.7)),
        exposure_time_s=float(max(film.exposure_time_s, 1.0e-6)),
        iso=float(max(film.iso, 1.0)),
        scene_ev100=float(scene_ev100),
        apparent_focus_distance_m=float(apparent_focus_distance_m),
        apparent_sensor_plane_z_m=float(apparent_sensor_plane_z_m),
        sensor_plane_z_m=float(sensor_plane_z_m),
        tilt_deg=float(tilt_deg),
        shift_mm=float(shift_mm),
    )


def _camera_from_pose(
    *,
    camera_cls: Callable[..., Any],
    width: int,
    height: int,
    eye: np.ndarray,
    target: np.ndarray,
    optics: CameraOptics,
) -> Any:
    fwd = np.asarray(target - eye, np.float64)
    fwd_norm = float(np.linalg.norm(fwd))
    if fwd_norm <= 1.0e-12:
        fwd = np.array([0.0, 0.0, -1.0], np.float64)
    else:
        fwd /= fwd_norm

    up_world = np.array([0.0, 1.0, 0.0], np.float64)
    right = np.cross(fwd, up_world)
    if float(np.linalg.norm(right)) <= 1.0e-9:
        up_world = np.array([0.0, 0.0, 1.0], np.float64)
    up = np.asarray(up_world, np.float64)
    up /= max(float(np.linalg.norm(up)), 1.0e-12)

    fov_y_rad = 2.0 * math.atan2(float(optics.sensor_h_mm) * 0.5, float(optics.focal_mm))
    return camera_cls(
        pos=np.asarray(eye, np.float64),
        fwd=fwd,
        up=up,
        fov_y_rad=float(fov_y_rad),
        width=int(width),
        height=int(height),
    )


def solve_sane_pinhole_camera(
    *,
    camera_cls: Callable[..., Any],
    width: int,
    height: int,
    optics: CameraOptics,
    film: FilmExposure,
    scene_center: np.ndarray,
    eye: np.ndarray | None = None,
    max_iters: int = 72,
    seed: int = 12345,
) -> SolvedCameraPackage:
    """Iteratively solve a camera setup toward the most sane configuration."""
    eye_vec = np.asarray([0.0, 0.0, 0.0] if eye is None else eye, np.float64)
    tgt = np.asarray(scene_center, np.float64)
    scene_dist = float(np.linalg.norm(tgt - eye_vec))
    scene_dist = max(scene_dist, 0.25)

    scene_ev100 = _ev100_from_settings(
        float(max(optics.f_number, 0.7)),
        float(max(film.exposure_time_s, 1.0e-6)),
        float(max(film.iso, 1.0)),
    )

    base_focus = scene_dist
    f_m = max(float(optics.focal_mm) * 1.0e-3, 1.0e-6)
    image_dist_m = 1.0 / max((1.0 / f_m) - (1.0 / max(base_focus, f_m * 1.001)), 1.0e-9)
    base_sensor_z = -float(image_dist_m)

    base_cfg = _build_input(
        optics=optics,
        film=film,
        focus_distance_m=base_focus,
        sensor_plane_z_m=base_sensor_z,
        scene_ev100=scene_ev100,
        apparent_focus_distance_m=scene_dist,
        apparent_sensor_plane_z_m=base_sensor_z,
        tilt_deg=0.0,
        shift_mm=0.0,
    )
    best_cfg = base_cfg
    best_report = evaluate_camera_sanity(base_cfg)
    best_obj = _objective(best_report)

    rng = random.Random(int(seed))
    focus_span = max(0.25, 0.35 * scene_dist)
    sensor_span = 0.015

    # Coarse candidate sweep first, then stochastic local refinement.
    focus_candidates = [
        scene_dist * 0.7,
        scene_dist * 0.85,
        scene_dist,
        scene_dist * 1.15,
        scene_dist * 1.3,
    ]
    sensor_candidates = [
        base_sensor_z - 0.010,
        base_sensor_z - 0.005,
        base_sensor_z,
        base_sensor_z + 0.005,
        base_sensor_z + 0.010,
    ]

    iters = 0
    for fd in focus_candidates:
        for sp in sensor_candidates:
            cfg = _build_input(
                optics=optics,
                film=film,
                focus_distance_m=fd,
                sensor_plane_z_m=sp,
                scene_ev100=scene_ev100,
                apparent_focus_distance_m=scene_dist,
                apparent_sensor_plane_z_m=base_sensor_z,
                tilt_deg=0.0,
                shift_mm=0.0,
            )
            rep = evaluate_camera_sanity(cfg)
            obj = _objective(rep)
            iters += 1
            if obj < best_obj:
                best_cfg, best_report, best_obj = cfg, rep, obj

    for _ in range(max(0, int(max_iters))):
        if best_report.status == "pass" and not best_report.warnings:
            break
        fd = best_cfg.focus_distance_m + rng.uniform(-focus_span, focus_span)
        sp = best_cfg.sensor_plane_z_m + rng.uniform(-sensor_span, sensor_span)
        tilt = best_cfg.tilt_deg + rng.uniform(-1.0, 1.0)
        shift = best_cfg.shift_mm + rng.uniform(-2.0, 2.0)
        cfg = _build_input(
            optics=optics,
            film=film,
            focus_distance_m=max(0.15, fd),
            sensor_plane_z_m=min(-0.005, max(-0.25, sp)),
            scene_ev100=scene_ev100,
            apparent_focus_distance_m=scene_dist,
            apparent_sensor_plane_z_m=base_sensor_z,
            tilt_deg=min(15.0, max(-15.0, tilt)),
            shift_mm=min(30.0, max(-30.0, shift)),
        )
        rep = evaluate_camera_sanity(cfg)
        obj = _objective(rep)
        iters += 1
        if obj < best_obj:
            best_cfg, best_report, best_obj = cfg, rep, obj
            focus_span *= 0.92
            sensor_span *= 0.92

    solved_camera = _camera_from_pose(
        camera_cls=camera_cls,
        width=width,
        height=height,
        eye=eye_vec,
        target=tgt,
        optics=optics,
    )
    return SolvedCameraPackage(
        camera=solved_camera,
        sanity_input=best_cfg,
        sanity_report=best_report,
        iterations=iters,
    )
