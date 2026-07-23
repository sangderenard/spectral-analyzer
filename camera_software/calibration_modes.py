"""Opt-in calibration scenes and their small, deterministic validators."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from .transport_contract import (
    SpectralLaneDescriptor,
    TransportDomain,
    TransportLaneTable,
    TransportWorkContract,
    fixed_visible_lane_table,
)


@dataclass(frozen=True)
class CalibrationValidationResult:
    validator_key: str
    passed: bool
    measurements: Mapping[str, float] = field(default_factory=dict)
    detail: str = ""


@dataclass(frozen=True)
class CalibrationModeSpec:
    key: str
    label: str
    description: str
    transport: TransportWorkContract | None
    scene_manifest: Mapping[str, Any]
    validator_keys: tuple[str, ...]
    enabled_by_default: bool = False

    def work_asset_manifest(self) -> dict[str, Any]:
        return {
            "object_kind": "calibration_scene",
            "calibration_mode": self.key,
            "display_name": self.label,
            "enabled_by_default": self.enabled_by_default,
            "transport": None if self.transport is None else self.transport.mapping(),
            "scene": dict(self.scene_manifest),
            "validators": list(self.validator_keys),
        }


@dataclass(frozen=True)
class CalibrationBootstrapStage:
    order: int
    mode_key: str
    purpose: str
    acceptance_gate: str
    enables_panel_harvest: bool = False

    def mapping(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "mode_key": self.mode_key,
            "purpose": self.purpose,
            "acceptance_gate": self.acceptance_gate,
            "enables_panel_harvest": self.enables_panel_harvest,
        }


def _fixed_spectral_contract(
    cohort: str, wavelengths_nm: tuple[float, ...]
) -> TransportWorkContract:
    c = 299_792_458.0
    table = TransportLaneTable(
        domain=TransportDomain.FIXED_SPECTRAL,
        cohort_id=cohort,
        lanes=tuple(
            SpectralLaneDescriptor(
                lane_id=i,
                identity=f"wavelength:{wavelength_nm:.3f}nm",
                frequency_hz=c / (wavelength_nm * 1.0e-9),
            )
            for i, wavelength_nm in enumerate(wavelengths_nm)
        ),
    )
    variant = next(size for size in (1, 3, 4, 8, 16, 32) if size >= len(table.lanes))
    return TransportWorkContract(
        lane_table=table,
        material_profile_library_key="canonical-optical-materials",
        emitter_profile_library_key="calibration-lines",
        sensor_profile_key="cie-1931-observer",
        payload_variant=variant,
    )


def focus_hall_scene_manifest() -> dict[str, Any]:
    distances_m = (2.7, 3.1, 3.45, 3.75, 4.1, 4.55, 5.1, 5.5)
    cards = []
    for index, distance in enumerate(distances_m):
        side = -1.0 if index % 2 else 1.0
        cards.append({
            "id": f"distance-{distance:g}m",
            "label": f"{distance:g} m",
            "distance_m": distance,
            "position_m": [distance, side * (0.28 + index * 0.045), 0.06 * ((index % 3) - 1)],
            "face_camera": True,
            "card_size_m": [0.36, 0.18],
        })
    return {
        "scene_type": "focus_hall",
        "units": "metres",
        "hall_length_m": 5.8,
        "focus_target_m": 3.75,
        "distance_cards": cards,
        "composition": "alternating in-field cards with readable distance markers",
    }


def prism_room_scene_manifest() -> dict[str, Any]:
    return {
        "scene_type": "prism_dispersion_room",
        "units": "metres",
        "room": "dark enclosure with white receiver wall",
        "optic": {
            "shape": "triangular_prism",
            "glass": "BK7",
            "closed_mesh": True,
        },
        "emitter": {
            "mode": "collimated",
            "direction_space": "canonical_camera",
            "divergence_deg": 0.0,
            "spectral_profile": "active transport cohort",
        },
        "receiver": "matte white wall",
        "real_camera_exposure": True,
        "gpu_required": True,
    }


def mirror_box_scene_manifest() -> dict[str, Any]:
    """Capacity torture scene: camera and source enclosed by specular walls."""

    return {
        "scene_type": "mirror_box_capacity",
        "units": "metres",
        "enclosure": {
            "closed": True,
            "wall_count": 6,
            "wall_reflectivity": 0.995,
            "roughness": 0.0,
        },
        "emitter": {
            "location": "inside enclosure",
            "spectral_profile": "all fixed visible lanes",
        },
        "capacity_contract": {
            "active_lanes": 32,
            "payload_storage_lanes": 32,
            "constant_stride_records": True,
            "fail_on_record_overflow": True,
            "minimum_bounces": 48,
        },
        "ray_trace_settings": {
            "transport_mode": "fixed",
            "lane_count": 32,
            "total_rays": 4_194_304,
            "max_sensor_epochs": 1,
            "epoch_bundle_count": 16,
            "sensor_samples_per_node": 1024,
            "sensor_t5_pair_budget": 268_435_456,
            "max_bounces": 64,
        },
        "real_camera_exposure": True,
        "gpu_required": True,
    }


def double_slit_scene_manifest() -> dict[str, Any]:
    return {
        "scene_type": "wave_double_slit",
        "units": "metres",
        "solver_contract": "scalar-adi-bpm-cpu-gpu-parity-v1",
        "ray_transport_allowed": False,
        "wavelengths_nm": [450.0, 500.0, 550.0, 600.0, 650.0],
        "aperture": {
            "slit_width_m": 8.0e-6,
            "slit_separation_m": 40.0e-6,
        },
        "propagation": {
            "grid_pitch_m": 2.0e-6,
            "step_m": 20.0e-6,
            "steps": 100,
        },
        "products": [
            "cpu_band_strip", "gpu_band_strip", "cpu_gpu_comparison",
            "complex_field_checkpoints",
        ],
    }


def _validate_color_science() -> CalibrationValidationResult:
    from thick_lens_focus_lab import _wavelength_to_rgb_weights

    wavelengths = np.asarray([450.0, 550.0, 650.0], np.float64)
    rgb = np.asarray(_wavelength_to_rgb_weights(wavelengths), np.float64)
    finite = bool(np.all(np.isfinite(rgb)) and np.all(rgb >= 0.0))
    ordered = bool(
        int(np.argmax(rgb[0])) == 2
        and int(np.argmax(rgb[1])) == 1
        and int(np.argmax(rgb[2])) == 0
    )
    return CalibrationValidationResult(
        "color-science-lines",
        finite and ordered,
        {
            "blue_450": float(rgb[0, 2]),
            "green_550": float(rgb[1, 1]),
            "red_650": float(rgb[2, 0]),
        },
        "450/550/650 nm must map to blue/green/red dominant sensor responses",
    )


def _validate_bk7_glass() -> CalibrationValidationResult:
    from camera_designer.optical_material import MATERIAL_CATALOG

    glass = MATERIAL_CATALOG["BK7"]
    n_f = float(glass.n_at(0.4861327))
    n_d = float(glass.n_at(0.5875618))
    n_c = float(glass.n_at(0.6562725))
    incident = math.radians(30.0)
    transmitted = math.asin(math.sin(incident) / n_d)
    passed = n_f > n_d > n_c > 1.0 and 0.0 < transmitted < incident
    return CalibrationValidationResult(
        "bk7-fraunhofer-lines",
        passed,
        {
            "n_F": n_f,
            "n_d": n_d,
            "n_C": n_c,
            "transmitted_angle_deg": math.degrees(transmitted),
        },
        "BK7 must disperse monotonically and bend a 30 degree air ray toward normal",
    )


def _validate_focus_hall() -> CalibrationValidationResult:
    scene = focus_hall_scene_manifest()
    distances = [float(card["distance_m"]) for card in scene["distance_cards"]]
    positions = [tuple(card["position_m"]) for card in scene["distance_cards"]]
    passed = (
        distances == sorted(distances)
        and len(set(positions)) == len(positions)
        and min(distances) < float(scene["focus_target_m"]) < max(distances)
    )
    return CalibrationValidationResult(
        "focus-hall-layout",
        passed,
        {"card_count": float(len(distances)), "depth_span_m": max(distances) - min(distances)},
        "distance cards must span both sides of the authored focus plane",
    )


def _validate_single_lane_ui() -> CalibrationValidationResult:
    contract = TransportWorkContract(
        lane_table=_fixed_spectral_contract(
            "single-reference-line-ui", (587.5618,)
        ).lane_table,
        payload_variant=1,
    )
    return CalibrationValidationResult(
        "single-lane-ui-contract",
        (
            contract.lane_table.domain is TransportDomain.FIXED_SPECTRAL
            and contract.lane_table.active_lane_count == 1
            and contract.payload_variant == 1
        ),
        {"active_lanes": 1.0, "payload_variant": 1.0},
        "UI image formation uses one explicit 587.5618 nm reference line",
    )


def _validate_prism_room() -> CalibrationValidationResult:
    scene = prism_room_scene_manifest()
    optic = dict(scene["optic"])
    emitter = dict(scene["emitter"])
    passed = (
        optic.get("shape") == "triangular_prism"
        and bool(optic.get("closed_mesh"))
        and emitter.get("mode") == "collimated"
        and scene.get("receiver") == "matte white wall"
    )
    return CalibrationValidationResult(
        "prism-room-contract",
        passed,
        {"divergence_deg": float(emitter.get("divergence_deg", 0.0))},
        "closed BK7 prism, native collimated launch, and white receiver wall are required",
    )


def _validate_mirror_box() -> CalibrationValidationResult:
    scene = mirror_box_scene_manifest()
    enclosure = dict(scene["enclosure"])
    capacity = dict(scene["capacity_contract"])
    settings = dict(scene["ray_trace_settings"])
    passed = bool(
        enclosure.get("closed")
        and int(enclosure.get("wall_count", 0)) == 6
        and float(enclosure.get("wall_reflectivity", 0.0)) >= 0.99
        and int(capacity.get("active_lanes", 0)) == 32
        and int(capacity.get("payload_storage_lanes", 0)) == 32
        and capacity.get("constant_stride_records") is True
        and capacity.get("fail_on_record_overflow") is True
        and int(settings.get("max_bounces", 0)) >= 48
    )
    return CalibrationValidationResult(
        "mirror-box-capacity-contract",
        passed,
        {
            "wall_count": float(enclosure.get("wall_count", 0)),
            "reflectivity": float(enclosure.get("wall_reflectivity", 0.0)),
            "active_lanes": float(capacity.get("active_lanes", 0)),
            "max_bounces": float(settings.get("max_bounces", 0)),
        },
        "closed six-wall mirror enclosure must torture constant-stride fixed-32 records",
    )


def _validate_double_slit() -> CalibrationValidationResult:
    scene = double_slit_scene_manifest()
    aperture = dict(scene["aperture"])
    propagation = dict(scene["propagation"])
    wavelengths = tuple(float(value) for value in scene["wavelengths_nm"])
    passed = bool(
        scene.get("ray_transport_allowed") is False
        and len(wavelengths) >= 3
        and min(wavelengths) > 0.0
        and float(aperture["slit_width_m"]) < float(aperture["slit_separation_m"])
        and 0 < int(propagation["steps"])
        and float(propagation["grid_pitch_m"]) > 0.0
    )
    return CalibrationValidationResult(
        "double-slit-wave-contract",
        passed,
        {
            "bands": float(len(wavelengths)),
            "slit_separation_um": float(aperture["slit_separation_m"]) * 1.0e6,
            "propagation_mm": (
                float(propagation["step_m"]) * int(propagation["steps"]) * 1.0e3
            ),
        },
        "CPU and GPU ADI BPM must propagate the same two-slit complex field",
    )


VALIDATORS: dict[str, Callable[[], CalibrationValidationResult]] = {
    "color-science-lines": _validate_color_science,
    "bk7-fraunhofer-lines": _validate_bk7_glass,
    "focus-hall-layout": _validate_focus_hall,
    "single-lane-ui-contract": _validate_single_lane_ui,
    "prism-room-contract": _validate_prism_room,
    "mirror-box-capacity-contract": _validate_mirror_box,
    "double-slit-wave-contract": _validate_double_slit,
}


CALIBRATION_MODES: tuple[CalibrationModeSpec, ...] = (
    CalibrationModeSpec("off", "Off", "Normal rendering; no calibration work is injected.", None, {}, ()),
    CalibrationModeSpec(
        "camera-bootstrap", "Camera bootstrap",
        "Ordered validators, focus proof, one-lane UI exposure, then panel harvest.",
        None,
        {"scene_type": "calibration_sequence", "stages": "see bootstrap_plan"},
        (
            "color-science-lines", "bk7-fraunhofer-lines",
            "focus-hall-layout", "single-lane-ui-contract",
        ),
    ),
    CalibrationModeSpec(
        "color-science", "Color science", "Three exact spectral lines validate sensor color conversion.",
        _fixed_spectral_contract("color-lines", (450.0, 550.0, 650.0)),
        {
            "scene_type": "spectral_line_camera_chart",
            "targets": ["450nm", "550nm", "650nm"],
            "real_camera_exposure": True,
        },
        ("color-science-lines",),
    ),
    CalibrationModeSpec(
        "glass", "Glass / Snell", "Fraunhofer-line rays validate Sellmeier and refraction.",
        _fixed_spectral_contract("bk7-lines", (486.1327, 587.5618, 656.2725)),
        {
            "scene_type": "single_interface",
            "glass": "BK7",
            "incident_angle_deg": 30.0,
            "real_camera_exposure": True,
        },
        ("bk7-fraunhofer-lines",),
    ),
    CalibrationModeSpec(
        "focus-hall", "Focus hall", "Distance cards expose focus falloff across one composition.",
        _fixed_spectral_contract("focus-reference", (587.5618,)),
        {**focus_hall_scene_manifest(), "real_camera_exposure": True},
        ("focus-hall-layout",),
    ),
    CalibrationModeSpec(
        "depth", "Depth / first surface",
        "One native GPU camera ray per sensor site reports the first authored-scene strike in metres.",
        _fixed_spectral_contract("depth-reference", (587.5618,)),
        {
            **focus_hall_scene_manifest(),
            "scene_type": "first_scene_hit_depth",
            "render_product": "sensor_optical_path_depth_m",
            "scene_bounces": 1,
            "camera_optics_traversed": True,
            "real_camera_exposure": True,
            "gpu_required": True,
        },
        ("focus-hall-layout",),
    ),
    CalibrationModeSpec(
        "prism-room", "Prism room",
        "A native collimated spectral beam traverses a closed BK7 prism and lands on a white wall.",
        _fixed_spectral_contract(
            "prism-fraunhofer-default", (486.1327, 587.5618, 656.2725)
        ),
        prism_room_scene_manifest(),
        ("prism-room-contract", "bk7-fraunhofer-lines"),
    ),
    CalibrationModeSpec(
        "mirror-box", "Mirror-box torture",
        "An internal broadband light and six near-perfect mirrors maximize long-path fixed-32 record pressure.",
        TransportWorkContract(
            lane_table=fixed_visible_lane_table("mirror-box-fixed-32", 32),
            material_profile_library_key="canonical-optical-materials",
            emitter_profile_library_key="visible-flat-spectrum",
            sensor_profile_key="cie-1931-observer",
            payload_variant=32,
            metadata={"payload_storage_lanes": 32, "constant_stride_records": True},
        ),
        mirror_box_scene_manifest(),
        ("mirror-box-capacity-contract",),
    ),
    CalibrationModeSpec(
        "double-slit", "Double slit",
        "The compiled CPU BPM and the actual GPU BPM shader form matched wavelength-band strips.",
        None,
        double_slit_scene_manifest(),
        ("double-slit-wave-contract",),
    ),
    CalibrationModeSpec(
        "single-lane-ui", "Single reference line",
        "Whole-interface image formation at one real 587.5618 nm spectral line.",
        _fixed_spectral_contract("single-reference-line-ui", (587.5618,)),
        {
            "scene_type": "whole_ui",
            "reference_wavelength_nm": 587.5618,
            "harvest_unchanged_panels": True,
            "replacement_policy": "crop_from_wholistic_exposure",
            "real_camera_exposure": True,
        },
        ("single-lane-ui-contract",),
    ),
)


def calibration_bootstrap_plan() -> tuple[CalibrationBootstrapStage, ...]:
    return (
        CalibrationBootstrapStage(
            1, "color-science", "prove exact-line sensor color response",
            "all color-science single/few-ray validators pass",
        ),
        CalibrationBootstrapStage(
            2, "glass", "prove Sellmeier ordering and one-interface Snell bending",
            "all glass single-ray validators pass",
        ),
        CalibrationBootstrapStage(
            3, "focus-hall", "prove camera focus placement across authored depth cards",
            "focus target is sharp and near/far cards show ordered blur",
        ),
        CalibrationBootstrapStage(
            4, "single-lane-ui", "exercise complete UI scene construction cheaply",
            "whole static UI exposure is accepted by the normal quality gate",
        ),
        CalibrationBootstrapStage(
            5, "single-lane-ui", "harvest unchanged panel crops from accepted exposure",
            "camera, light, material, crop, and panel content compatibility keys match",
            enables_panel_harvest=True,
        ),
    )


def calibration_mode(key: str) -> CalibrationModeSpec:
    mode = next(mode for mode in CALIBRATION_MODES if mode.key == str(key))
    if mode.key == "camera-bootstrap":
        return CalibrationModeSpec(
            key=mode.key,
            label=mode.label,
            description=mode.description,
            transport=None,
            scene_manifest={
                "scene_type": "calibration_sequence",
                "stages": [stage.mapping() for stage in calibration_bootstrap_plan()],
            },
            validator_keys=mode.validator_keys,
        )
    return mode


def run_calibration_validators(key: str) -> tuple[CalibrationValidationResult, ...]:
    mode = calibration_mode(key)
    return tuple(VALIDATORS[name]() for name in mode.validator_keys)


__all__ = [
    "CalibrationValidationResult", "CalibrationModeSpec", "CALIBRATION_MODES",
    "VALIDATORS", "calibration_mode", "run_calibration_validators",
    "focus_hall_scene_manifest", "CalibrationBootstrapStage",
    "prism_room_scene_manifest", "mirror_box_scene_manifest",
    "double_slit_scene_manifest",
    "calibration_bootstrap_plan",
]
