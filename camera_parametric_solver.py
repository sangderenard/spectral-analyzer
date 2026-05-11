"""Parametric camera solver using differentiable optimization.

This solver models a plate-camera style articulated geometry:
- Independent corner rails for sensor plate tilt.
- Independent aperture rail z/x/y placement.
- Front/rear thin-lens group on independent rails.

Effective focal length is derived from front/rear lens group spacing.
The optimization objective is pinhole-first (through aperture), with
aperture-aware blur and rail feasibility penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Optional

import numpy as np

from camera_exposure_budget import CameraOptics, FilmExposure
from camera_sanity_calculator import CameraSanityInput, CameraSanityReport, evaluate_camera_sanity

try:
    import torch
except Exception as exc:  # pragma: no cover - explicit hard requirement path
    raise RuntimeError(
        "camera_parametric_solver requires PyTorch. Install with: pip install torch"
    ) from exc


@dataclass(frozen=True)
class SolvedCameraPackage:
    """Solved camera object and associated sanity diagnostics."""

    camera: Any
    sanity_input: CameraSanityInput
    sanity_report: CameraSanityReport
    iterations: int


@dataclass(frozen=True)
class RailModelConfig:
    sensor_z_min_m: float = -0.110
    sensor_z_max_m: float = -0.015
    sensor_corner_max_mm: float = 14.0
    sensor_shift_max_mm: float = 50.0

    aperture_z_min_m: float = -0.002
    aperture_z_max_m: float = 0.002
    aperture_shift_max_mm: float = 1.0

    bellows_sep_min_m: float = 0.005
    bellows_sep_max_m: Optional[float] = None  # None => auto from focal length
    front_lens_shift_max_mm: float = 60.0
    front_lens_tilt_max_deg: float = 35.0

    aperture_radius_min_m: float = 1.0e-6
    aperture_radius_max_m: Optional[float] = None  # None => auto from focal length
    energy_deposit_epsilon: float = 1.0e-6
    energy_deposit_patience: int = 24


def _ev100_from_settings(f_number: float, exposure_time_s: float, iso: float) -> float:
    n2_over_t = (f_number * f_number) / max(exposure_time_s, 1.0e-9)
    return math.log2(max(n2_over_t, 1.0e-12)) - math.log2(max(iso / 100.0, 1.0e-12))


def _camera_from_pose(
    *,
    camera_cls: Callable[..., Any],
    width: int,
    height: int,
    eye: np.ndarray,
    target: np.ndarray,
    focal_m: float,
    sensor_h_m: float,
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

    fov_y_rad = 2.0 * math.atan2(float(sensor_h_m) * 0.5, max(float(focal_m), 1.0e-6))
    return camera_cls(
        pos=np.asarray(eye, np.float64),
        fwd=fwd,
        up=up,
        fov_y_rad=float(fov_y_rad),
        width=int(width),
        height=int(height),
    )


def _fit_corner_tilt_deg(corner_mm: np.ndarray, sensor_w_m: float, sensor_h_m: float) -> float:
    z = np.asarray(corner_mm, np.float64) * 1.0e-3
    z_tl, z_tr, z_bl, z_br = [float(v) for v in z]
    dzdx = ((z_tr + z_br) - (z_tl + z_bl)) / max(2.0 * sensor_w_m, 1.0e-9)
    dzdy = ((z_bl + z_br) - (z_tl + z_tr)) / max(2.0 * sensor_h_m, 1.0e-9)
    return float(math.degrees(math.atan(math.sqrt(dzdx * dzdx + dzdy * dzdy))))


def _rail_multiscale_activation(
    x: torch.Tensor,
    span: float,
    coarse_weight: float = 0.72,
    medium_weight: float = 0.20,
    fine_weight: float = 0.08,
    coarse_sharpness: float = 0.40,
    medium_sharpness: float = 1.25,
    fine_sharpness: float = 4.50,
) -> torch.Tensor:
    """Scaled rail response with coarse/medium/fine normalized adjustments.

    Behavior:
    - Near center: mostly coarse linear motion.
    - Mid travel: medium adjust dominates.
    - Near limits: fine trim still works while the output remains saturated.
    """
    w0 = float(coarse_weight)
    w1 = float(medium_weight)
    w2 = float(fine_weight)
    wsum = max(1.0e-9, w0 + w1 + w2)
    mix = (
        w0 * torch.tanh(float(coarse_sharpness) * x)
        + w1 * torch.tanh(float(medium_sharpness) * x)
        + w2 * torch.tanh(float(fine_sharpness) * x)
    ) / wsum
    return float(span) * mix


def _rail_multiscale_bounded(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map unconstrained rail state into [lo, hi] with multiscale saturation."""
    mid = 0.5 * (float(lo) + float(hi))
    half = 0.5 * max(1.0e-9, float(hi) - float(lo))
    return torch.tensor(mid, dtype=x.dtype, device=x.device) + _rail_multiscale_activation(x, half)


def _rail_center_linear_activation(x: torch.Tensor, span: float) -> torch.Tensor:
    """Center-normal rail with bounded travel and multiscale fine trim."""
    return _rail_multiscale_activation(x, span)


def _safe_ev100_from_nt_iso_torch(f_number: torch.Tensor, exposure_time_s: float, iso: float) -> torch.Tensor:
    n2_over_t = (f_number * f_number) / max(float(exposure_time_s), 1.0e-9)
    return torch.log2(torch.clamp(n2_over_t, min=1.0e-12)) - math.log2(max(float(iso) / 100.0, 1.0e-12))


def solve_sane_camera_rig(
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
    rail_model: Optional[RailModelConfig] = None,
) -> SolvedCameraPackage:
    """Solve a physical camera rig with a differentiable rail/lens model."""

    eye_vec = np.asarray([0.0, 0.0, 0.0] if eye is None else eye, np.float64)
    tgt = np.asarray(scene_center, np.float64)
    scene_dist = float(np.linalg.norm(tgt - eye_vec))
    scene_dist = max(scene_dist, 0.25)

    scene_ev100 = _ev100_from_settings(
        float(max(optics.f_number, 0.7)),
        float(max(film.exposure_time_s, 1.0e-6)),
        float(max(film.iso, 1.0)),
    )

    f_nom_m = max(float(optics.focal_mm) * 1.0e-3, 1.0e-6)
    f_front_nom_m = max(2.0 * f_nom_m, 1.0e-6)
    f_rear_nom_m = max(2.0 * f_nom_m, 1.0e-6)
    sensor_w_m = max(float(optics.sensor_w_mm) * 1.0e-3, 1.0e-6)
    sensor_h_m = max(float(optics.sensor_h_mm) * 1.0e-3, 1.0e-6)

    torch.manual_seed(int(seed))
    dtype = torch.float64
    device = torch.device("cpu")

    # Parameterization over articulated rail DoFs.
    # 0:sensor z, 1..4:corner rails, 5..6:sensor shift xy,
    # 7:aperture z, 8..9:aperture shift xy,
    # 10:lens back z, 11:lens sep (bellows),
    # 12:front focal scale, 13:rear focal scale,
    # 14:aperture radius,
    # 15..16:front lens lateral shift xy,
    # 17..18:front lens orientation tilt xy
    
    # Initialize to safe lens-pair defaults:
    # - Sensor plane centered (x=0 → midpoint of range)
    # - No tilt/shift on sensor (x=0 → centered)
    # - Aperture centered and nominally positioned (x=0 → centered)
    # - Lens separation mid-range (x=0 → midpoint)
    # - Focal scales nominal 1.0 (x=0 → sigmoid(0)=0.5 → 0.8+0.2=1.0)
    # - Aperture radius conservative/smaller (x=-1.2 → push toward minimum)
    # - Front lens unshifted and untilted (x=0 → no offset/rotation)
    p_init = torch.zeros((19,), dtype=dtype, device=device)
    p_init[14] = -1.2  # Aperture radius: start smaller for optical safety
    p = torch.nn.Parameter(p_init)
    opt = torch.optim.Adam([p], lr=0.045)

    softplus = torch.nn.Softplus()
    sigmoid = torch.sigmoid
    rail_cfg = rail_model if rail_model is not None else RailModelConfig()
    sensor_min = float(rail_cfg.sensor_z_min_m)
    sensor_max = float(rail_cfg.sensor_z_max_m)
    # Broad tilt/shift exploration space in 3D.
    max_corner_mm = float(max(0.1, rail_cfg.sensor_corner_max_mm))
    max_shift_mm = float(max(0.1, rail_cfg.sensor_shift_max_mm))

    # Aperture placement is body-fixed with only small micro-adjust range.
    aperture_z_min = float(rail_cfg.aperture_z_min_m)
    aperture_z_max = float(rail_cfg.aperture_z_max_m)
    aperture_shift_max_mm = float(max(0.05, rail_cfg.aperture_shift_max_mm))

    # Bellows can explore a very large range; rear lens has back-plate-like travel.
    bellows_sep_min_m = float(max(1.0e-4, rail_cfg.bellows_sep_min_m))
    bellows_sep_max_m = (
        max(bellows_sep_min_m + 1.0e-4, float(rail_cfg.bellows_sep_max_m))
        if rail_cfg.bellows_sep_max_m is not None
        else max(0.050, 2.0 * f_nom_m)
    )
    front_lens_shift_max_mm = float(max(0.1, rail_cfg.front_lens_shift_max_mm))
    front_lens_tilt_max_deg = float(max(0.1, rail_cfg.front_lens_tilt_max_deg))

    # Continuously adjustable aperture radius from near singularity to very open.
    aperture_radius_min_m = float(max(1.0e-9, rail_cfg.aperture_radius_min_m))
    aperture_radius_max_m = (
        max(aperture_radius_min_m + 1.0e-9, float(rail_cfg.aperture_radius_max_m))
        if rail_cfg.aperture_radius_max_m is not None
        else max(0.050, 1.5 * f_nom_m)
    )
    energy_deposit_epsilon = float(max(0.0, rail_cfg.energy_deposit_epsilon))
    energy_deposit_patience = int(max(1, rail_cfg.energy_deposit_patience))

    best_loss = float("inf")
    best_snapshot: dict[str, float] = {}
    low_energy_streak = 0
    steps_run = 0

    total_steps = max(120, int(max_iters) * 8)
    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)

        sensor_z = _rail_multiscale_bounded(p[0], sensor_min, sensor_max)
        corner_mm = _rail_multiscale_activation(p[1:5], max_corner_mm)
        corner_z = sensor_z + corner_mm * 1.0e-3
        sensor_z_eff = torch.mean(corner_z)

        sensor_shift_x_mm = _rail_center_linear_activation(p[5], max_shift_mm)
        sensor_shift_y_mm = _rail_center_linear_activation(p[6], max_shift_mm)

        aperture_z = _rail_multiscale_bounded(p[7], aperture_z_min, aperture_z_max)
        aperture_shift_x_mm = _rail_center_linear_activation(p[8], aperture_shift_max_mm)
        aperture_shift_y_mm = _rail_center_linear_activation(p[9], aperture_shift_max_mm)

        lens_back_z = _rail_multiscale_bounded(p[10], sensor_min, sensor_max)
        lens_sep = _rail_multiscale_bounded(p[11], bellows_sep_min_m, bellows_sep_max_m)
        lens_front_z = lens_back_z + lens_sep

        lens_front_shift_x_mm = _rail_center_linear_activation(p[15], front_lens_shift_max_mm)
        lens_front_shift_y_mm = _rail_center_linear_activation(p[16], front_lens_shift_max_mm)
        lens_front_tilt_x_deg = _rail_center_linear_activation(p[17], front_lens_tilt_max_deg)
        lens_front_tilt_y_deg = _rail_center_linear_activation(p[18], front_lens_tilt_max_deg)

        front_scale = 0.8 + 0.4 * sigmoid(p[12])
        rear_scale = 0.8 + 0.4 * sigmoid(p[13])
        f_front = torch.tensor(f_front_nom_m, dtype=dtype, device=device) * front_scale
        f_rear = torch.tensor(f_rear_nom_m, dtype=dtype, device=device) * rear_scale

        eff_denom = (1.0 / f_front) + (1.0 / f_rear) - (lens_sep / (f_front * f_rear))
        f_eff = 1.0 / torch.clamp(eff_denom, min=1.0e-6)

        aperture_radius_m = _rail_multiscale_bounded(p[14], aperture_radius_min_m, aperture_radius_max_m)
        f_number_eff = torch.clamp(f_eff / torch.clamp(2.0 * aperture_radius_m, min=1.0e-9), min=1.0e-3)

        do = torch.tensor(scene_dist, dtype=dtype, device=device)
        di = 1.0 / torch.clamp((1.0 / f_eff) - (1.0 / do), min=1.0e-6)
        image_required_z = lens_back_z - di
        pinhole_target_z = aperture_z - f_eff

        pinhole_loss = ((sensor_z_eff - pinhole_target_z) / 0.0015) ** 2

        # Outer lens orientation and decenter create a lateral optical-axis shift.
        sensor_shift_x_m = sensor_shift_x_mm * 1.0e-3
        sensor_shift_y_m = sensor_shift_y_mm * 1.0e-3
        aperture_shift_x_m = aperture_shift_x_mm * 1.0e-3
        aperture_shift_y_m = aperture_shift_y_mm * 1.0e-3
        front_shift_x_m = lens_front_shift_x_mm * 1.0e-3
        front_shift_y_m = lens_front_shift_y_mm * 1.0e-3
        front_tilt_x_rad = lens_front_tilt_x_deg * (math.pi / 180.0)
        front_tilt_y_rad = lens_front_tilt_y_deg * (math.pi / 180.0)
        front_to_sensor = torch.abs(sensor_z_eff - lens_front_z)
        axis_x_m = front_shift_x_m + torch.tan(front_tilt_x_rad) * front_to_sensor
        axis_y_m = front_shift_y_m + torch.tan(front_tilt_y_rad) * front_to_sensor
        pinhole_xy_loss = (
            ((sensor_shift_x_m - (aperture_shift_x_m + axis_x_m)) / 0.0015) ** 2
            + ((sensor_shift_y_m - (aperture_shift_y_m + axis_y_m)) / 0.0015) ** 2
        )

        pupil_d = 2.0 * aperture_radius_m
        coc_um = (torch.abs(sensor_z_eff - image_required_z) * pupil_d / torch.clamp(torch.abs(di), min=1.0e-6)) * 1.0e6
        blur_loss = (coc_um / 45.0) ** 2

        scene_ev_t = torch.tensor(scene_ev100, dtype=dtype, device=device)
        ev_est = _safe_ev100_from_nt_iso_torch(f_number_eff, film.exposure_time_s, film.iso)
        energy_deposit_rel = torch.pow(torch.tensor(2.0, dtype=dtype, device=device), scene_ev_t - ev_est)
        ev_loss = ((ev_est - scene_ev_t) / 2.5) ** 2

        z_tl, z_tr, z_bl, z_br = corner_z
        dzdx = ((z_tr + z_br) - (z_tl + z_bl)) / max(2.0 * sensor_w_m, 1.0e-9)
        dzdy = ((z_bl + z_br) - (z_tl + z_tr)) / max(2.0 * sensor_h_m, 1.0e-9)
        tilt_rad = torch.atan(torch.sqrt(dzdx * dzdx + dzdy * dzdy))
        tilt_deg = tilt_rad * (180.0 / math.pi)
        tilt_pen = softplus((tilt_deg - 20.0) / 2.0)

        shift_mag = torch.sqrt(sensor_shift_x_mm * sensor_shift_x_mm + sensor_shift_y_mm * sensor_shift_y_mm)
        shift_pen = softplus((shift_mag - max_shift_mm) / 6.0)

        ap_shift_mag = torch.sqrt(aperture_shift_x_mm * aperture_shift_x_mm + aperture_shift_y_mm * aperture_shift_y_mm)
        ap_shift_pen = softplus((ap_shift_mag - aperture_shift_max_mm) / 0.35)

        front_shift_mag = torch.sqrt(lens_front_shift_x_mm * lens_front_shift_x_mm + lens_front_shift_y_mm * lens_front_shift_y_mm)
        front_shift_pen = softplus((front_shift_mag - front_lens_shift_max_mm) / 8.0)
        front_tilt_mag = torch.sqrt(lens_front_tilt_x_deg * lens_front_tilt_x_deg + lens_front_tilt_y_deg * lens_front_tilt_y_deg)
        front_tilt_pen = softplus((front_tilt_mag - front_lens_tilt_max_deg) / 3.0)

        sensor_travel_pen = softplus((sensor_min - sensor_z_eff) / 0.001) + softplus((sensor_z_eff - sensor_max) / 0.001)
        image_travel_pen = softplus((sensor_min - image_required_z) / 0.001) + softplus((image_required_z - sensor_max) / 0.001)
        lens_stack_pen = softplus((lens_back_z - aperture_z + 0.001) / 0.0005) + softplus((aperture_z - lens_front_z + 0.001) / 0.0005)

        # Keep aperture and sensor principal axes reasonably aligned.
        axis_align_pen = (
            ((sensor_shift_x_mm - aperture_shift_x_mm) / 6.0) ** 2
            + ((sensor_shift_y_mm - aperture_shift_y_mm) / 6.0) ** 2
        )

        loss = (
            1.80 * pinhole_loss
            + 1.10 * pinhole_xy_loss
            + 0.45 * blur_loss
            + 0.30 * ev_loss
            + 0.20 * tilt_pen
            + 0.15 * shift_pen
            + 0.10 * ap_shift_pen
            + 0.10 * front_shift_pen
            + 0.10 * front_tilt_pen
            + 0.25 * sensor_travel_pen
            + 0.40 * image_travel_pen
            + 0.20 * lens_stack_pen
            + 0.15 * axis_align_pen
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p], max_norm=10.0)
        opt.step()
        steps_run = step + 1

        energy_deposit_rel_v = float(energy_deposit_rel.detach().cpu().item())
        if energy_deposit_rel_v <= energy_deposit_epsilon:
            low_energy_streak += 1
        else:
            low_energy_streak = 0

        cur_loss = float(loss.detach().cpu().item())
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_snapshot = {
                "sensor_z": float(sensor_z.detach().cpu().item()),
                "corner_tl_mm": float(corner_mm[0].detach().cpu().item()),
                "corner_tr_mm": float(corner_mm[1].detach().cpu().item()),
                "corner_bl_mm": float(corner_mm[2].detach().cpu().item()),
                "corner_br_mm": float(corner_mm[3].detach().cpu().item()),
                "sensor_shift_x_mm": float(sensor_shift_x_mm.detach().cpu().item()),
                "sensor_shift_y_mm": float(sensor_shift_y_mm.detach().cpu().item()),
                "aperture_z": float(aperture_z.detach().cpu().item()),
                "aperture_shift_x_mm": float(aperture_shift_x_mm.detach().cpu().item()),
                "aperture_shift_y_mm": float(aperture_shift_y_mm.detach().cpu().item()),
                "lens_back_z": float(lens_back_z.detach().cpu().item()),
                "lens_front_z": float(lens_front_z.detach().cpu().item()),
                "front_f_mm": float((f_front * 1.0e3).detach().cpu().item()),
                "rear_f_mm": float((f_rear * 1.0e3).detach().cpu().item()),
                "aperture_radius_m": float(aperture_radius_m.detach().cpu().item()),
                "lens_front_shift_x_mm": float(lens_front_shift_x_mm.detach().cpu().item()),
                "lens_front_shift_y_mm": float(lens_front_shift_y_mm.detach().cpu().item()),
                "lens_front_tilt_x_deg": float(lens_front_tilt_x_deg.detach().cpu().item()),
                "lens_front_tilt_y_deg": float(lens_front_tilt_y_deg.detach().cpu().item()),
            }

        if step > 36 and best_loss < 1.0e-4:
            break

        if low_energy_streak >= energy_deposit_patience:
            break

    if not best_snapshot:
        best_snapshot = {
            "sensor_z": -0.050,
            "corner_tl_mm": 0.0,
            "corner_tr_mm": 0.0,
            "corner_bl_mm": 0.0,
            "corner_br_mm": 0.0,
            "sensor_shift_x_mm": 0.0,
            "sensor_shift_y_mm": 0.0,
            "aperture_z": 0.0,
            "aperture_shift_x_mm": 0.0,
            "aperture_shift_y_mm": 0.0,
            "lens_back_z": -0.004,
            "lens_front_z": 0.004,
            "front_f_mm": float(f_front_nom_m * 1.0e3),
            "rear_f_mm": float(f_rear_nom_m * 1.0e3),
            "aperture_radius_m": float(max(aperture_radius_min_m, min(aperture_radius_max_m, 0.5 * f_nom_m))),
            "lens_front_shift_x_mm": 0.0,
            "lens_front_shift_y_mm": 0.0,
            "lens_front_tilt_x_deg": 0.0,
            "lens_front_tilt_y_deg": 0.0,
        }

    # Final hard projection: keep the solved plate center on the pinhole target
    # implied by the solved aperture and lens-group effective focal length.
    f1_m = max(float(best_snapshot["front_f_mm"]) * 1.0e-3, 1.0e-6)
    f2_m = max(float(best_snapshot["rear_f_mm"]) * 1.0e-3, 1.0e-6)
    sep_m = max(float(best_snapshot["lens_front_z"] - best_snapshot["lens_back_z"]), 1.0e-6)
    denom = (1.0 / f1_m) + (1.0 / f2_m) - (sep_m / (f1_m * f2_m))
    f_eff_m = (1.0 / max(denom, 1.0e-6))
    pinhole_target_z = float(best_snapshot["aperture_z"]) - f_eff_m
    best_snapshot["sensor_z"] = float(min(sensor_max, max(sensor_min, pinhole_target_z)))

    solved_cfg = CameraSanityInput(
        focal_length_mm=float(optics.focal_mm),
        focus_distance_m=float(scene_dist),
        f_number=float(
            max(
                (float(f_eff_m) / max(2.0 * float(best_snapshot["aperture_radius_m"]), 1.0e-9)),
                1.0e-3,
            )
        ),
        exposure_time_s=float(max(film.exposure_time_s, 1.0e-6)),
        iso=float(max(film.iso, 1.0)),
        scene_ev100=float(scene_ev100),
        apparent_focus_distance_m=float(scene_dist),
        apparent_sensor_plane_z_m=float(best_snapshot["sensor_z"]),
        lens_center_z_m=0.0,
        aperture_plane_offset_m=float(best_snapshot["aperture_z"]),
        lens_thickness_mm=float((best_snapshot["lens_front_z"] - best_snapshot["lens_back_z"]) * 1.0e3),
        lens_front_focal_mm=float(best_snapshot["front_f_mm"]),
        lens_rear_focal_mm=float(best_snapshot["rear_f_mm"]),
        lens_front_plane_z_m=float(best_snapshot["lens_front_z"]),
        lens_rear_plane_z_m=float(best_snapshot["lens_back_z"]),
        lens_front_shift_x_mm=float(best_snapshot["lens_front_shift_x_mm"]),
        lens_front_shift_y_mm=float(best_snapshot["lens_front_shift_y_mm"]),
        lens_front_tilt_x_deg=float(best_snapshot["lens_front_tilt_x_deg"]),
        lens_front_tilt_y_deg=float(best_snapshot["lens_front_tilt_y_deg"]),
        aperture_rail_z_m=float(best_snapshot["aperture_z"]),
        aperture_shift_x_mm=float(best_snapshot["aperture_shift_x_mm"]),
        aperture_shift_y_mm=float(best_snapshot["aperture_shift_y_mm"]),
        aperture_radius_m=float(best_snapshot["aperture_radius_m"]),
        sensor_plane_z_m=float(best_snapshot["sensor_z"]),
        sensor_corner_tl_mm=float(best_snapshot["corner_tl_mm"]),
        sensor_corner_tr_mm=float(best_snapshot["corner_tr_mm"]),
        sensor_corner_bl_mm=float(best_snapshot["corner_bl_mm"]),
        sensor_corner_br_mm=float(best_snapshot["corner_br_mm"]),
        sensor_shift_x_mm=float(best_snapshot["sensor_shift_x_mm"]),
        sensor_shift_y_mm=float(best_snapshot["sensor_shift_y_mm"]),
        tilt_deg=float(
            _fit_corner_tilt_deg(
                np.array(
                    [
                        best_snapshot["corner_tl_mm"],
                        best_snapshot["corner_tr_mm"],
                        best_snapshot["corner_bl_mm"],
                        best_snapshot["corner_br_mm"],
                    ],
                    dtype=np.float64,
                ),
                sensor_w_m,
                sensor_h_m,
            )
        ),
        shift_mm=float(
            math.hypot(
                best_snapshot["sensor_shift_x_mm"],
                best_snapshot["sensor_shift_y_mm"],
            )
        ),
    )

    best_report = evaluate_camera_sanity(solved_cfg)
    solved_camera = _camera_from_pose(
        camera_cls=camera_cls,
        width=width,
        height=height,
        eye=eye_vec,
        target=tgt,
        focal_m=float(best_report.planes.effective_focal_m),
        sensor_h_m=float(sensor_h_m),
    )

    return SolvedCameraPackage(
        camera=solved_camera,
        sanity_input=solved_cfg,
        sanity_report=best_report,
        iterations=int(max(1, steps_run)),
    )
