"""camera_software/optical_design.py
------------------------------------
First-order optical train design helpers.

This module deliberately stays in paraxial optics.  It answers the question
"where should the groups go for the requested camera behavior?" before the
exact conic `CompoundLens` tracer verifies the result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "OpticalDesignSpec",
    "ParaxialGroup",
    "SolvedOpticalTrain",
    "paraxial_system_matrix",
    "effective_focal_length",
    "back_focal_distance",
    "image_distance_for_object",
    "solve_four_group_zoom_surrogate",
]

_EPS = 1.0e-12


@dataclass(frozen=True)
class OpticalDesignSpec:
    """Ordinary camera desires for a solved first-order lens train."""
    form: Literal["four_group_zoom_surrogate"] = "four_group_zoom_surrogate"
    focal_length_range_m: Tuple[float, float] = (0.045, 0.070)
    zoom: float = 0.40
    focus_distance_m: float = 0.35
    f_number: float = 2.8
    entrance_x_m: float = 0.92
    sensor_x_m: float = 2.42
    sensor_clearance_m: float = 0.025
    image_radius_m: float = 0.045
    group_count: int = 4
    default_ior: float = 1.55
    min_air_gap_m: float = 0.008
    group_thickness_m: float = 0.014
    max_group_radius_m: float = 0.060
    lock_focal_length: bool = True
    lock_entrance_position: bool = True
    lock_sensor_position: bool = True
    focal_length_tolerance_m: float = 5.0e-4
    group_positions_m: Optional[Tuple[float, ...]] = None
    group_focal_lengths_m: Optional[Tuple[float, ...]] = None
    group_powers_dpt: Optional[Tuple[float, ...]] = None
    group_aperture_radii_m: Optional[Tuple[float, ...]] = None
    group_thicknesses_m: Optional[Tuple[float, ...]] = None
    group_radius_front_m: Optional[Tuple[float, ...]] = None
    group_radius_back_m: Optional[Tuple[float, ...]] = None
    group_iors: Optional[Tuple[float, ...]] = None
    lock_group_positions: bool = False
    lock_group_powers: bool = False

    @property
    def target_focal_length_m(self) -> float:
        z = float(np.clip(self.zoom, 0.0, 1.0))
        f0, f1 = self.focal_length_range_m
        return float(f0 + (f1 - f0) * z)

    @property
    def aperture_radius_m(self) -> float:
        return float(max(1.0e-5, self.target_focal_length_m / (2.0 * max(self.f_number, 0.1))))


@dataclass(frozen=True)
class ParaxialGroup:
    """One solved paraxial group."""
    name: str
    x_m: float
    focal_length_m: float
    aperture_radius_m: float
    thickness_m: float
    radius_front_m: float
    radius_back_m: float
    ior: float

    @property
    def power(self) -> float:
        return 1.0 / self.focal_length_m if abs(self.focal_length_m) > _EPS else 0.0


@dataclass(frozen=True)
class SolvedOpticalTrain:
    """Solved group positions plus first-order diagnostics."""
    spec: OpticalDesignSpec
    groups: Tuple[ParaxialGroup, ...]
    aperture_x_m: float
    aperture_radius_m: float
    assembly_front_x_m: float
    assembly_back_x_m: float
    exit_pupil_x_m: float
    exit_pupil_radius_m: float
    effective_focal_length_m: float
    image_distance_m: float
    sensor_error_m: float
    matrix: np.ndarray = field(repr=False)

    def to_lens_configs(self):
        """Convert to thick_lens_focus_lab.LensConfig objects lazily."""
        from thick_lens_focus_lab import LensConfig

        return [
            LensConfig(
                center_x=float(g.x_m),
                thickness=float(g.thickness_m),
                aperture_radius=float(g.aperture_radius_m),
                radius_front=float(g.radius_front_m),
                radius_back=float(g.radius_back_m),
                ior=float(g.ior),
            )
            for g in self.groups
        ]

    def apply_to_scene(self, scene):
        """Mutate a thick_lens_focus_lab.SceneConfig to match this solution."""
        scene.lens_stack = self.to_lens_configs()
        scene.tube_x0 = float(min(scene.tube_x0, self.assembly_front_x_m))
        scene.tube_x1 = float(max(scene.tube_x1, self.assembly_back_x_m))
        scene.exit_pupil_x = float(self.exit_pupil_x_m)
        scene.exit_pupil_radius = float(self.exit_pupil_radius_m)
        # Sync the sensor plane to the solved sensor position.  The image_plate
        # must live exactly where the solver put the focus, not at a stale default.
        plate = getattr(scene, "image_plate", None)
        if plate is not None and hasattr(plate, "x"):
            plate.x = float(self.spec.sensor_x_m)
        return scene


def _translation(d: float) -> np.ndarray:
    return np.array([[1.0, float(d)], [0.0, 1.0]], dtype=np.float64)


def _thin_lens(power: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [-float(power), 1.0]], dtype=np.float64)


def paraxial_system_matrix(groups: Sequence[ParaxialGroup]) -> np.ndarray:
    """Return ABCD matrix from first group plane to last group plane."""
    if not groups:
        return np.eye(2, dtype=np.float64)
    gs = sorted(groups, key=lambda g: g.x_m)
    m = _thin_lens(gs[0].power)
    prev_x = float(gs[0].x_m)
    for g in gs[1:]:
        m = _thin_lens(g.power) @ _translation(float(g.x_m) - prev_x) @ m
        prev_x = float(g.x_m)
    return m


def effective_focal_length(matrix: np.ndarray) -> float:
    c = float(np.asarray(matrix, dtype=np.float64)[1, 0])
    return -1.0 / c if abs(c) > _EPS else float("inf")


def back_focal_distance(matrix: np.ndarray) -> float:
    m = np.asarray(matrix, dtype=np.float64)
    a = float(m[0, 0])
    c = float(m[1, 0])
    return -a / c if abs(c) > _EPS else float("inf")


def image_distance_for_object(groups: Sequence[ParaxialGroup], object_x_m: float) -> float:
    """Distance after the last group where an on-axis object point images."""
    gs = sorted(groups, key=lambda g: g.x_m)
    if not gs:
        return float("inf")
    first_x = float(gs[0].x_m)
    s = max(first_x - float(object_x_m), 1.0e-6)
    # Ray from object point crossing first group at height h has theta=h/s.
    state = np.array([1.0, 1.0 / s], dtype=np.float64)
    state = paraxial_system_matrix(gs) @ state
    y, theta = float(state[0]), float(state[1])
    return -y / theta if abs(theta) > _EPS else float("inf")


def _group_curvature_from_power(power: float, ior: float, radius_hint: float) -> Tuple[float, float]:
    """Symmetric biconvex/biconcave curvature pair for a thin-lens power."""
    n_delta = max(float(ior) - 1.0, 1.0e-6)
    if abs(power) < _EPS:
        r = max(abs(radius_hint), 1.0)
        return r, r
    r = max(abs(2.0 * n_delta / power), abs(radius_hint), 1.0e-3)
    if power > 0.0:
        return r, r
    return -r, -r


def _min_curvature_radius_for_aperture(aperture_radius: float, thickness: float, edge_thickness: float = 0.006) -> float:
    """Minimum spherical radius that leaves the requested edge thickness."""
    ap = float(max(1.0e-6, aperture_radius))
    sag_max = 0.5 * max(float(thickness) - float(edge_thickness), 1.0e-5)
    return float((ap * ap + sag_max * sag_max) / max(2.0 * sag_max, 1.0e-9))


def _seq_value(values: Optional[Sequence[float]], index: int, default: float) -> float:
    if values is None or index >= len(values):
        return float(default)
    value = values[index]
    if value is None:
        return float(default)
    return float(value)


def _power_from_curvature(radius_front: float, radius_back: float, ior: float) -> float:
    n_delta = max(float(ior) - 1.0, 1.0e-6)
    rf = float(radius_front)
    rb = float(radius_back)
    p = 0.0
    if abs(rf) > _EPS:
        p += 1.0 / rf
    if abs(rb) > _EPS:
        p += 1.0 / rb
    return float(n_delta * p)


def _hardware_power_override(spec: OpticalDesignSpec, index: int) -> Optional[float]:
    if spec.group_powers_dpt is not None and index < len(spec.group_powers_dpt):
        return float(spec.group_powers_dpt[index])
    if spec.group_focal_lengths_m is not None and index < len(spec.group_focal_lengths_m):
        fl = float(spec.group_focal_lengths_m[index])
        if abs(fl) > _EPS:
            return float(1.0 / fl)
    if (
        spec.group_radius_front_m is not None and index < len(spec.group_radius_front_m)
        and spec.group_radius_back_m is not None and index < len(spec.group_radius_back_m)
    ):
        ior = _seq_value(spec.group_iors, index, spec.default_ior)
        return _power_from_curvature(
            float(spec.group_radius_front_m[index]),
            float(spec.group_radius_back_m[index]),
            ior,
        )
    return None


def _apply_power_overrides(spec: OpticalDesignSpec, powers: np.ndarray) -> np.ndarray:
    out = np.asarray(powers, dtype=np.float64).copy()
    for i in range(min(len(out), int(spec.group_count))):
        override = _hardware_power_override(spec, i)
        if override is not None:
            out[i] = float(override)
    return out


def _candidate_groups(spec: OpticalDesignSpec, xs: Sequence[float], powers: Sequence[float]) -> Tuple[ParaxialGroup, ...]:
    f = spec.target_focal_length_m
    image_r = float(max(1.0e-5, spec.image_radius_m))
    envelope = max(float(spec.sensor_x_m) - float(spec.entrance_x_m), 1.0e-5)
    field_tan = image_r / max(f, 1.0e-5)
    # Physical minimum front aperture: must pass both the marginal ray bundle
    # (aperture_radius_m) and the chief ray swing across the group span
    # (field coverage term).  Do NOT cap by max_group_radius_m — that is a
    # manufacturing-intent upper bound, not a physics floor.  Capping it here
    # produces glass that is geometrically too small to support the requested
    # f-number and field angle (e.g. f/1.4 on a 6×6 sensor needs ~63 mm radius
    # front element, but a 31 mm cap silently produces an undersized lens).
    front_field_r = image_r + 0.18 * envelope * field_tan
    ap = max(spec.aperture_radius_m, 0.22 * f, front_field_r)
    groups = []
    for i, (x, power) in enumerate(zip(xs, powers)):
        ior = _seq_value(spec.group_iors, i, spec.default_ior)
        thickness = _seq_value(spec.group_thicknesses_m, i, spec.group_thickness_m)
        fl = 1.0 / power if abs(power) > _EPS else 1.0e9
        frac = float(i) / float(max(1, len(xs) - 1))
        local_ap = _seq_value(
            spec.group_aperture_radii_m,
            i,
            max(spec.aperture_radius_m, ap * (1.0 - 0.18 * frac)),
        )
        rf, rb = _group_curvature_from_power(power, ior, local_ap * 1.25)
        rf = _seq_value(spec.group_radius_front_m, i, rf)
        rb = _seq_value(spec.group_radius_back_m, i, rb)
        # Keep curvature safely larger than aperture so demo mesh remains valid.
        min_r = max(
            local_ap * 1.18,
            _min_curvature_radius_for_aperture(local_ap, thickness),
        )
        rf = math.copysign(max(abs(rf), min_r), rf)
        rb = math.copysign(max(abs(rb), min_r), rb)
        groups.append(ParaxialGroup(
            name=f"G{i + 1}",
            x_m=float(x),
            focal_length_m=float(fl),
            aperture_radius_m=float(local_ap),
            thickness_m=float(thickness),
            radius_front_m=float(rf),
            radius_back_m=float(rb),
            ior=float(ior),
        ))
    return tuple(groups)


def _groups_for_scaled_powers(
    spec: OpticalDesignSpec,
    xs: Sequence[float],
    base_powers: Sequence[float],
    scale: float,
) -> Tuple[ParaxialGroup, ...]:
    return _candidate_groups(spec, xs, np.asarray(base_powers, dtype=np.float64) * float(scale))


def _fit_power_scale_for_focal_length(
    spec: OpticalDesignSpec,
    xs: Sequence[float],
    base_powers: Sequence[float],
) -> Tuple[float, Tuple[ParaxialGroup, ...], np.ndarray, float]:
    """Solve global power scale so the candidate train hits target EFL.

    This keeps focal length as an invariant while position/search parameters
    remain free.  The fallback still returns the closest sampled scale if the
    system has no clean monotonic bracket for the requested target.
    """
    target = float(max(1.0e-6, spec.target_focal_length_m))

    def _eval(scale: float):
        groups = _groups_for_scaled_powers(spec, xs, base_powers, scale)
        m = paraxial_system_matrix(groups)
        efl = effective_focal_length(m)
        return groups, m, float(efl)

    samples = np.geomspace(0.08, 16.0, 32)
    best = None
    for s in samples:
        groups, m, efl = _eval(float(s))
        if not math.isfinite(efl) or efl <= 0.0:
            continue
        metric = abs(math.log(max(efl, 1.0e-12) / target))
        if best is None or metric < best[0]:
            best = (metric, float(s), groups, m, efl)

    if best is None:
        groups, m, efl = _eval(1.0)
        return 1.0, groups, m, efl

    _, scale, groups, m, efl = best
    return scale, groups, m, efl


def _candidate_groups_with_locked_focal_length(
    spec: OpticalDesignSpec,
    xs: Sequence[float],
    base_powers: Sequence[float],
    front_power_scale: float,
) -> Tuple[np.ndarray, Tuple[ParaxialGroup, ...], np.ndarray, float]:
    """Build candidate groups with G4 power solved to force target EFL."""
    xs_arr = np.asarray(xs, dtype=np.float64)
    base = np.asarray(base_powers, dtype=np.float64)
    powers = base.copy() * float(front_power_scale)
    powers = _apply_power_overrides(spec, powers)
    if powers.shape[0] < 4:
        groups = _candidate_groups(spec, xs_arr, powers)
        m = paraxial_system_matrix(groups)
        return powers, groups, m, effective_focal_length(m)

    # Matrix from G1 through free-space propagation to the G4 plane, before
    # the G4 thin-lens power is applied.
    m_pre = _thin_lens(powers[0])
    prev_x = float(xs_arr[0])
    for i in range(1, 3):
        m_pre = _thin_lens(powers[i]) @ _translation(float(xs_arr[i]) - prev_x) @ m_pre
        prev_x = float(xs_arr[i])
    m_pre = _translation(float(xs_arr[3]) - prev_x) @ m_pre

    a_pre = float(m_pre[0, 0])
    c_pre = float(m_pre[1, 0])
    target_f = float(max(1.0e-6, spec.target_focal_length_m))
    g4_power_is_hardware = _hardware_power_override(spec, 3) is not None
    if abs(a_pre) > _EPS and not bool(spec.lock_group_powers) and not g4_power_is_hardware:
        # L(P4) @ M_pre has C = C_pre - P4*A_pre.  EFL = -1/C.
        powers[3] = (c_pre + 1.0 / target_f) / a_pre
    powers = _apply_power_overrides(spec, powers)

    groups = _candidate_groups(spec, xs_arr, powers)
    m = paraxial_system_matrix(groups)
    efl = effective_focal_length(m)
    return powers, groups, m, efl


def solve_four_group_zoom_surrogate(spec: Optional[OpticalDesignSpec] = None) -> SolvedOpticalTrain:
    """Solve a practical four-group zoom/focus surrogate.

    The solver uses a stable four-power template, then searches variator,
    compensator, and focus group positions so the paraxial system lands the
    requested focus on the fixed sensor plane.  It is a controllable first-order
    stand-in, not a full aberration-corrected photographic lens patent.
    """
    spec = spec or OpticalDesignSpec()
    f = max(1.0e-4, spec.target_focal_length_m)
    z = float(np.clip(spec.zoom, 0.0, 1.0))
    assembly_front = float(spec.entrance_x_m)
    xsensor = float(spec.sensor_x_m)
    assembly_back = xsensor
    thicknesses = [
        max(_seq_value(spec.group_thicknesses_m, i, spec.group_thickness_m), 1.0e-5)
        for i in range(max(1, int(spec.group_count)))
    ]
    thickness = max(thicknesses)
    air_gap = max(float(spec.min_air_gap_m), 1.0e-5)
    rear_limit = xsensor - max(float(spec.sensor_clearance_m), 0.5 * thickness)
    # Center-to-center group spacing must include both half-thicknesses plus
    # the requested air gap.  The previous center-spacing rule allowed glass
    # volumes to overlap even while group centers remained ordered.
    min_gap = thickness + air_gap
    usable = max(rear_limit - assembly_front, min_gap * 3.0)

    # Four-group signed-power template: positive front, negative variator,
    # positive compensator, positive rear/focus.  Zoom shifts power balance.
    base_powers = np.array([
        (0.48 + 0.10 * z) / f,
        -(0.18 + 0.12 * z) / f,
        (0.48 + 0.10 * z) / f,
        (0.34 + 0.06 * z) / f,
    ], dtype=np.float64)

    g1 = assembly_front
    compact_len = max(3.0 * min_gap, min(usable, 0.18 + 0.55 * f + 0.10 * z))
    g4_base = min(rear_limit, g1 + compact_len)
    d12_min = min_gap
    d23_min = min_gap
    d34_min = min_gap
    span = max(g4_base - g1, d12_min + d23_min + d34_min)

    object_distance = max(float(spec.focus_distance_m), 1.0e-3)
    powers = _apply_power_overrides(spec, base_powers.copy())
    best = None
    # Coarse deterministic search over normalized train length and internal
    # variator/compensator placement.  The train is never translated to meet
    # focus; residual image-plane error is reported as a real design limitation.
    explicit_positions = None
    if spec.group_positions_m is not None and len(spec.group_positions_m) >= int(spec.group_count):
        explicit_positions = np.asarray(spec.group_positions_m[:int(spec.group_count)], dtype=np.float64)
    power_scales = [None] if bool(spec.lock_focal_length) else list(np.linspace(0.45, 2.40, 16))
    for power_scale in power_scales:
        if power_scale is not None:
            powers = _apply_power_overrides(spec, base_powers * float(power_scale))
        span_fracs = [1.0] if explicit_positions is not None else np.linspace(0.18, 1.0, 10)
        for span_frac in span_fracs:
            g4 = min(rear_limit, g1 + max(3.0 * min_gap, usable * float(span_frac)))
            span = max(g4 - g1, d12_min + d23_min + d34_min)
            if g1 + span > rear_limit + 1.0e-12:
                continue
            u_values = [0.0] if explicit_positions is not None else np.linspace(0.18, 0.68, 10)
            for u in u_values:
                v_values = [0.0] if explicit_positions is not None else np.linspace(u + 0.10, 0.90, 12)
                for v in v_values:
                    if explicit_positions is not None:
                        xs = explicit_positions.copy()
                    else:
                        xs = np.array([g1, g1 + u * span, g1 + v * span, g1 + span], dtype=np.float64)
                    if xs[-1] > rear_limit or np.min(np.diff(xs)) < min_gap:
                        continue
                    if bool(spec.lock_focal_length):
                        best_locked = None
                        for front_scale in np.geomspace(0.35, 3.0, 12):
                            cand_powers, cand_groups, cand_m, cand_efl = _candidate_groups_with_locked_focal_length(
                                spec, xs, base_powers, float(front_scale)
                            )
                            if not math.isfinite(cand_efl) or cand_efl <= 0.0:
                                continue
                            if abs(cand_efl - f) > max(float(spec.focal_length_tolerance_m), 0.02 * f):
                                continue
                            max_power = float(np.max(np.abs(cand_powers)))
                            power_penalty = 1.0e-4 * max_power * max_power
                            cand_img_d = image_distance_for_object(cand_groups, float(cand_groups[0].x_m) - object_distance)
                            if not math.isfinite(cand_img_d) or cand_img_d <= 0.0:
                                continue
                            cand_sensor_d = xsensor - xs[-1]
                            cand_focus_err = cand_img_d - cand_sensor_d
                            cand_score = cand_focus_err * cand_focus_err + power_penalty
                            if best_locked is None or cand_score < best_locked[0]:
                                best_locked = (cand_score, cand_powers, cand_groups, cand_m, cand_img_d, cand_efl)
                        if best_locked is None:
                            continue
                        _, powers_for_candidate, groups, m, img_d, efl = best_locked
                    else:
                        groups = _candidate_groups(spec, xs, powers)
                        m = paraxial_system_matrix(groups)
                        efl = effective_focal_length(m)
                        powers_for_candidate = powers.copy()
                        img_d = image_distance_for_object(groups, float(groups[0].x_m) - object_distance)
                        if not math.isfinite(img_d) or img_d <= 0.0:
                            continue
                    sensor_d = xsensor - xs[-1]
                    err_focus = img_d - sensor_d
                    err_f = (efl - f) / f if math.isfinite(efl) else 1.0e6
                    compact_penalty = 0.02 * ((xs[-1] - g1) / max(usable, _EPS))
                    focal_penalty = 0.0 if bool(spec.lock_focal_length) else (0.35 * err_f) * (0.35 * err_f)
                    score = (
                        err_focus * err_focus
                        + focal_penalty
                        + compact_penalty * compact_penalty
                    )
                    if best is None or score < best[0]:
                        best = (score, xs, powers_for_candidate.copy(), groups, m, img_d, efl)

    if best is None:
        xs = explicit_positions.copy() if explicit_positions is not None else np.linspace(g1, g4_base, 4, dtype=np.float64)
        if bool(spec.lock_focal_length):
            powers, groups, m, efl = _candidate_groups_with_locked_focal_length(spec, xs, base_powers, 1.0)
            powers = np.array([g.power for g in groups], dtype=np.float64)
        else:
            groups = _candidate_groups(spec, xs, powers)
            m = paraxial_system_matrix(groups)
            efl = effective_focal_length(m)
        img_d = image_distance_for_object(groups, float(groups[0].x_m) - object_distance)
    else:
        _, xs, powers, groups, m, img_d, efl = best

    # Rack the rear focus group to close residual focus error while preserving
    # group order and the fixed camera assembly envelope.
    rack_iterations = 0 if (explicit_positions is not None or bool(spec.lock_group_positions)) else 2
    for _ in range(rack_iterations):
        sensor_d = xsensor - float(groups[-1].x_m)
        residual = img_d - sensor_d
        rack = float(np.clip(-0.55 * residual, -0.10, 0.10))
        if abs(rack) <= 1.0e-9:
            break
        xs2 = np.array([g.x_m for g in groups], dtype=np.float64)
        xs2[-1] = float(np.clip(xs2[-1] + rack, xs2[-2] + min_gap, rear_limit))
        if bool(spec.lock_focal_length):
            front_scale = float(powers[0] / base_powers[0]) if abs(float(base_powers[0])) > _EPS else 1.0
            powers, groups, m, efl = _candidate_groups_with_locked_focal_length(spec, xs2, base_powers, front_scale)
            powers = np.array([g.power for g in groups], dtype=np.float64)
        else:
            groups = _candidate_groups(spec, xs2, powers)
            m = paraxial_system_matrix(groups)
            efl = effective_focal_length(m)
        img_d = image_distance_for_object(groups, float(groups[0].x_m) - object_distance)
    sensor_d = xsensor - float(groups[-1].x_m)

    aperture_x = float(groups[1].x_m)
    exit_x = float(min(xsensor - 0.02, rear_limit, float(groups[-1].x_m) + max(0.25 * sensor_d, 0.015)))
    return SolvedOpticalTrain(
        spec=spec,
        groups=tuple(groups),
        aperture_x_m=aperture_x,
        aperture_radius_m=float(spec.aperture_radius_m),
        assembly_front_x_m=float(assembly_front),
        assembly_back_x_m=float(assembly_back),
        exit_pupil_x_m=exit_x,
        exit_pupil_radius_m=float(spec.aperture_radius_m),
        effective_focal_length_m=float(efl),
        image_distance_m=float(img_d),
        sensor_error_m=float(img_d - sensor_d),
        matrix=np.asarray(m, dtype=np.float64),
    )
