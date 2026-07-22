"""Canonical camera, lighting, and spectral manifests.

The manifest is deliberately ordinary JSON data.  Human-readable values are
the authority; hashes are derived indexes used to decide which prepared parts
of a camera/lighting package can be reused.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .film_format import DEFAULT_FILM_FORMAT


CAMERA_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_WAVELENGTHS_NM = tuple(700.0 - i * (300.0 / 7.0) for i in range(8))


def _plain(value: Any) -> Any:
    """Return a stable, finite, JSON-compatible representation."""

    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("camera manifest numbers must be finite")
        return 0.0 if value == 0.0 else float(value)
    if hasattr(value, "item"):
        return _plain(value.item())
    raise TypeError(f"unsupported camera manifest value {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )


def manifest_hash(value: Any, *, namespace: str = "manifest") -> str:
    digest = hashlib.sha256()
    digest.update(f"{namespace}\0".encode("utf-8"))
    digest.update(canonical_json(value).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def sample_spectral_model(
    model: Mapping[str, Any] | None,
    wavelengths_nm: Sequence[float],
) -> list[float]:
    """Sample a continuous/histogram SPD onto an explicit transport grid."""

    spec = dict(model or {})
    kind = str(spec.get("kind", "sampled_gaussian")).lower()
    wavelengths = [float(value) for value in wavelengths_nm]
    values: list[float]
    if kind in {"flat", "constant"}:
        values = [1.0 for _ in wavelengths]
    elif kind in {"blackbody", "planck"}:
        temperature = max(1.0, float(spec.get("temperature_k", 5600.0)))
        h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
        values = []
        for wavelength_nm in wavelengths:
            wavelength_m = max(1.0e-12, wavelength_nm * 1.0e-9)
            exponent = min(700.0, h * c / (wavelength_m * k * temperature))
            values.append(
                (2.0 * h * c * c) /
                (wavelength_m ** 5 * max(math.expm1(exponent), 1.0e-300))
            )
    elif kind in {"histogram", "measured", "sampled"}:
        source_wavelengths = [float(v) for v in spec.get("wavelengths_nm", ())]
        source_weights = [float(v) for v in spec.get("weights", ())]
        if len(source_wavelengths) != len(source_weights) or not source_weights:
            raise ValueError("histogram spectrum requires equal non-empty wavelengths_nm and weights")
        pairs = sorted(zip(source_wavelengths, source_weights))
        values = []
        for wavelength in wavelengths:
            if wavelength <= pairs[0][0]:
                values.append(pairs[0][1])
                continue
            if wavelength >= pairs[-1][0]:
                values.append(pairs[-1][1])
                continue
            for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
                if x0 <= wavelength <= x1:
                    alpha = (wavelength - x0) / max(x1 - x0, 1.0e-12)
                    values.append(y0 + alpha * (y1 - y0))
                    break
    else:
        center = float(spec.get("center_nm", 555.0))
        sigma = max(1.0e-9, float(spec.get("sigma_nm", 85.0)))
        values = [
            math.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
            for wavelength in wavelengths
        ]
    peak = max([max(0.0, value) for value in values], default=0.0)
    if not math.isfinite(peak) or peak <= 0.0:
        return [0.0 for _ in values]
    return [max(0.0, float(value)) / peak for value in values]


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def default_camera_manifest() -> dict[str, Any]:
    """The physical camera currently used by the thick-lens UI renderer."""

    return {
        "schema_version": CAMERA_MANIFEST_SCHEMA_VERSION,
        "asset_key": "camera/6x6/four-group/default",
        "lens": {
            "family": "four_group_zoom_surrogate",
            "profile": "55-110/4",
            "focal_length_range_mm": [55.0, 110.0],
            "zoom": 0.5,
            "focal_length_mm": 82.5,
            "f_number": 4.0,
            "aperture_diameter_mm": 20.625,
            "group_count": 4,
            "group_thickness_mm": 18.0,
            "minimum_air_gap_mm": 8.0,
            "default_ior": 1.55,
            "glass": "N-BK7",
            "surface_model": "spherical_conic_k0",
            "representation": "exact_parametric_conic",
            "design_fidelity": "first_order_surrogate_exact_surface_transport",
        },
        "focus": {
            "distance_m": 1.0,
            "mechanism": "rear_group_G4_translation",
        },
        "sensor": {
            "film_format_key": DEFAULT_FILM_FORMAT.key,
            "film_format_label": DEFAULT_FILM_FORMAT.label,
            "mount_standard": DEFAULT_FILM_FORMAT.mount_standard,
            "back_type": "flat",
            "physical_width_mm": DEFAULT_FILM_FORMAT.frame_width_mm,
            "physical_height_mm": DEFAULT_FILM_FORMAT.frame_height_mm,
            "image_circle_diameter_mm": DEFAULT_FILM_FORMAT.image_circle_diameter_mm,
            "physical_aspect_ratio": DEFAULT_FILM_FORMAT.aspect_ratio,
            "default_work_raster_px": [
                DEFAULT_FILM_FORMAT.default_work_width_px,
                DEFAULT_FILM_FORMAT.default_work_height_px,
            ],
            "default_final_raster_px": [
                DEFAULT_FILM_FORMAT.default_final_edge_px,
                DEFAULT_FILM_FORMAT.default_final_edge_px,
            ],
            "crop_origin": "top-left",
        },
        "film_stage": {
            "depth_delta_mm": 0.0,
            "tilt_right_deg": 0.0,
            "tilt_up_deg": 0.0,
            "shift_x_mm": 0.0,
            "shift_y_mm": 0.0,
        },
        "body": {
            "outer_material": "camera_body_charcoal",
            "outer_reflectance": 0.16,
            "outer_roughness": 0.28,
            "outer_metallic": 0.35,
            "interior_material": "calib_aperture_black",
            "reflection_visibility": "physical_outer_surfaces",
        },
        "spectral": {
            "authoring_domain": "continuous_functions_sampled_for_transport",
            "transport_mode": "one_discrete_wavelength_per_path",
            "transport_abi_version": 1,
            "lane_semantics": "fixed_spectral",
            "active_wavelengths_nm": list(DEFAULT_WAVELENGTHS_NM),
            "active_band_count": 8,
            "material_band_capacity": 32,
            "lens_dispersion_model": "sampled_sellmeier",
        },
        "flash": {
            "asset_key": "lighting/thick-lens/default-photographic-rig",
            "intensity_scale": 1.0,
            "emitters": [
                {
                    "key": "side_room_source",
                    "kind": "stage_light_tube",
                    "emission_scale": 12.0,
                    "declared_cct_k": 5500.0,
                    "declared_cri": 95.0,
                },
                {
                    "key": "camera_ring_light",
                    "kind": "camera_mounted_ring",
                    "emission_scale": 200.0,
                    "declared_cct_k": 5600.0,
                    "declared_cri": 98.0,
                },
            ],
            "resolved_spectral_model": {
                "kind": "sampled_gaussian",
                "center_nm": 555.0,
                "sigma_nm": 85.0,
            },
            "modifier": {"mode": "none"},
            "burst": {
                "backend_mode": "native_flash_sensor_package",
                "duration_s": 0.010,
                "stages": 1,
                "profile": "steady",
                "duty_cycle": 1.0,
                "energy_scale": 1.0,
                "exposure_weight": 1.0,
            },
        },
    }


def _compatibility_payloads(data: Mapping[str, Any]) -> dict[str, Any]:
    lens = dict(data.get("lens", {}))
    spectral = dict(data.get("spectral", {}))
    sensor = dict(data.get("sensor", {}))
    film_stage = dict(data.get("film_stage", {}))
    flash = dict(data.get("flash", {}))
    topology = {
        key: lens.get(key) for key in (
            "family", "group_count", "surface_model", "representation",
        )
    }
    geometry = {
        "topology": topology,
        "focal_length_range_mm": lens.get("focal_length_range_mm"),
        "zoom": lens.get("zoom"),
        "focal_length_mm": lens.get("focal_length_mm"),
        "f_number": lens.get("f_number"),
        "group_thickness_mm": lens.get("group_thickness_mm"),
        "minimum_air_gap_mm": lens.get("minimum_air_gap_mm"),
        "solved_groups": data.get("resolved", {}).get("groups"),
    }
    return {
        "optical_topology": topology,
        "optical_geometry": geometry,
        "spectral_grid": spectral,
        "sensor_geometry": sensor,
        "film_stage": film_stage,
        "focus_state": data.get("focus", {}),
        "flash_geometry": {
            "asset_key": flash.get("asset_key"),
            "emitters": flash.get("emitters"),
            "modifier": flash.get("modifier"),
        },
        "flash_spectrum": {
            "resolved_spectral_model": flash.get("resolved_spectral_model"),
            "spectral_grid": spectral,
        },
    }


def _with_hashes(data: Mapping[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(dict(data))
    clean.pop("identity", None)
    facets = _compatibility_payloads(clean)
    clean["identity"] = {
        "manifest_hash": manifest_hash(clean, namespace="camera-manifest-v1"),
        "compatibility": {
            name: manifest_hash(value, namespace=f"camera-{name}-v1")
            for name, value in facets.items()
        },
    }
    return clean


@dataclass(frozen=True)
class CameraManifest:
    """Immutable facade over a canonical human-readable camera mapping."""

    data: Mapping[str, Any]

    def mapping(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.data))

    @property
    def key(self) -> str:
        return str(self.data.get("asset_key", ""))

    @property
    def hash(self) -> str:
        return str(dict(self.data.get("identity", {})).get("manifest_hash", ""))

    def compatibility_key(self, facet: str) -> str:
        identity = dict(self.data.get("identity", {}))
        return str(dict(identity.get("compatibility", {})).get(str(facet), ""))


@dataclass(frozen=True)
class CameraCompatibilityDecision:
    update_class: str
    exact_match: bool
    reuse_bvh: bool
    reuse_optical_payload: bool
    update_focus: bool
    update_sensor: bool
    update_film_stage: bool
    update_flash: bool
    reasons: tuple[str, ...]

    def mapping(self) -> dict[str, Any]:
        return {
            "update_class": self.update_class,
            "exact_match": self.exact_match,
            "reuse_bvh": self.reuse_bvh,
            "reuse_optical_payload": self.reuse_optical_payload,
            "update_focus": self.update_focus,
            "update_sensor": self.update_sensor,
            "update_film_stage": self.update_film_stage,
            "update_flash": self.update_flash,
            "reasons": list(self.reasons),
        }


def camera_compatibility(
    prepared: CameraManifest, requested: CameraManifest
) -> CameraCompatibilityDecision:
    """Classify the cheapest safe transition between two camera manifests."""

    if prepared.hash == requested.hash:
        return CameraCompatibilityDecision(
            "exact_reuse", True, True, True, False, False, False, False, (),
        )
    changed = tuple(
        facet for facet in (
            "optical_topology", "optical_geometry", "spectral_grid",
            "focus_state", "sensor_geometry", "film_stage",
            "flash_geometry", "flash_spectrum",
        )
        if prepared.compatibility_key(facet) != requested.compatibility_key(facet)
    )
    topology_same = "optical_topology" not in changed
    geometry_same = "optical_geometry" not in changed
    spectral_same = "spectral_grid" not in changed
    focus_changed = "focus_state" in changed
    sensor_changed = "sensor_geometry" in changed
    stage_changed = "film_stage" in changed
    flash_changed = any(name.startswith("flash_") for name in changed)
    # Focus is a rear-group translation in the current physical manifest.
    # Until the native tracer exposes a verified BVH refit, stale focus
    # triangles are not reusable even though the lens family is unchanged.
    reuse_bvh = topology_same and geometry_same and not focus_changed
    reuse_payload = reuse_bvh and spectral_same and not focus_changed
    if not topology_same:
        update_class = "full_camera_rebuild"
    elif not spectral_same:
        update_class = "spectral_payload_rebuild"
    elif not geometry_same:
        update_class = "optical_geometry_rebuild"
    elif focus_changed:
        update_class = "rear_group_focus_geometry_rebuild"
    elif sensor_changed or stage_changed:
        update_class = "film_stage_or_sensor_update"
    elif flash_changed:
        update_class = "lighting_update"
    else:
        update_class = "metadata_only"
    return CameraCompatibilityDecision(
        update_class=update_class,
        exact_match=False,
        reuse_bvh=reuse_bvh,
        reuse_optical_payload=reuse_payload,
        update_focus=focus_changed,
        update_sensor=sensor_changed,
        update_film_stage=stage_changed,
        update_flash=flash_changed,
        reasons=changed,
    )


def resolve_camera_manifest(
    camera: Mapping[str, Any] | None = None,
    image: Mapping[str, Any] | None = None,
    flash: Mapping[str, Any] | None = None,
    *,
    wavelengths_nm: Sequence[float] | None = None,
) -> CameraManifest:
    """Resolve legacy scene-order fields and an optional nested manifest."""

    camera = dict(camera or {})
    image = dict(image or {})
    flash_order = dict(flash or {})
    authored = camera.get("manifest", {})
    if authored and not isinstance(authored, Mapping):
        raise ValueError("camera.manifest must be an object")
    data = _merge(default_camera_manifest(), dict(authored or {}))

    lens = dict(data["lens"])
    explicit_lens = bool(dict(authored or {}).get("lens"))
    legacy_focal = camera.get("focal_mm")
    legacy_aperture = camera.get("aperture_mm")
    if legacy_focal is not None and not explicit_lens:
        focal = float(legacy_focal)
        lens.update({
            "profile": f"{focal:g}mm-scene-order",
            "focal_length_range_mm": [focal, focal],
            "zoom": 0.0,
            "focal_length_mm": focal,
        })
    else:
        focal = float(lens["focal_length_mm"])
    if legacy_aperture is not None and not explicit_lens:
        aperture = float(legacy_aperture)
        lens["aperture_diameter_mm"] = aperture
        lens["f_number"] = focal / aperture
    data["lens"] = lens

    focus = dict(data["focus"])
    if "focus_distance_m" in camera:
        focus["distance_m"] = float(camera["focus_distance_m"])
        focus["distance_definition"] = "authored_explicit"
    elif isinstance(camera.get("position_m"), (list, tuple)) and isinstance(
        camera.get("focus_target_m", camera.get("target_m")), (list, tuple)
    ):
        position = [float(value) for value in camera["position_m"]]
        target = [
            float(value) for value in camera.get(
                "focus_target_m", camera.get("target_m")
            )
        ]
        if len(position) == len(target) == 3:
            focus["distance_m"] = math.sqrt(sum(
                (target[index] - position[index]) ** 2 for index in range(3)
            ))
            focus["distance_definition"] = "sensor_to_focus_target"
    data["focus"] = focus

    sensor = dict(data["sensor"])
    if "sensor_w_mm" in camera:
        sensor["physical_width_mm"] = float(camera["sensor_w_mm"])
    if "sensor_h_mm" in camera:
        sensor["physical_height_mm"] = float(camera["sensor_h_mm"])
    if "back_type" in camera:
        sensor["back_type"] = str(camera["back_type"])
    elif isinstance(camera.get("back"), Mapping) and camera["back"].get("type"):
        sensor["back_type"] = str(camera["back"]["type"])
    sensor["full_raster_px"] = [
        int(image.get("width", 0) or 0), int(image.get("height", 0) or 0),
    ]
    region = dict(image.get("region", {}) or {})
    sensor["crop_px"] = {
        "x": int(region.get("x", 0) or 0),
        "y": int(region.get("y", 0) or 0),
        "width": int(region.get("width", image.get("width", 0)) or 0),
        "height": int(region.get("height", image.get("height", 0)) or 0),
    }
    full_width, full_height = sensor["full_raster_px"]
    physical_width = float(sensor["physical_width_mm"])
    physical_height = float(sensor["physical_height_mm"])
    required_image_circle = math.hypot(physical_width, physical_height)
    authored_sensor = dict(dict(authored or {}).get("sensor", {}) or {})
    if "image_circle_diameter_mm" not in authored_sensor:
        sensor["image_circle_diameter_mm"] = max(
            float(sensor.get("image_circle_diameter_mm", 0.0)),
            required_image_circle,
        )
    elif float(sensor["image_circle_diameter_mm"]) + 1.0e-9 < required_image_circle:
        raise ValueError(
            "camera manifest image circle does not cover its recording gate: "
            f"diameter={float(sensor['image_circle_diameter_mm']):.3f}mm "
            f"required>={required_image_circle:.3f}mm"
        )
    sensor["physical_aspect_ratio"] = physical_width / physical_height
    if full_width > 0 and full_height > 0:
        crop = sensor["crop_px"]
        pitch_x = physical_width / full_width
        pitch_y = physical_height / full_height
        sensor["raster_sample_pitch_mm"] = [pitch_x, pitch_y]
        sensor["raster_pixel_aspect_ratio"] = pitch_x / pitch_y
        sensor["crop_physical_mm"] = {
            "x": float(crop["x"]) * pitch_x,
            "y": float(crop["y"]) * pitch_y,
            "width": float(crop["width"]) * pitch_x,
            "height": float(crop["height"]) * pitch_y,
        }
        sensor["crop_center_offset_mm"] = [
            (float(crop["x"]) + 0.5 * float(crop["width"])) * pitch_x
            - 0.5 * physical_width,
            0.5 * physical_height
            - (float(crop["y"]) + 0.5 * float(crop["height"])) * pitch_y,
        ]
    data["sensor"] = sensor

    spectral = dict(data["spectral"])
    if wavelengths_nm is not None:
        values = [float(item) for item in wavelengths_nm]
        spectral["active_wavelengths_nm"] = values
        spectral["active_band_count"] = len(values)
    data["spectral"] = spectral

    resolved_flash = _merge(data["flash"], dict(flash_order.get("manifest", {}) or {}))
    if "intensity_scale" in flash_order:
        resolved_flash["intensity_scale"] = float(flash_order["intensity_scale"])
    active_wavelengths = list(spectral.get("active_wavelengths_nm", ()))
    resolved_flash["resolved_spectral_samples"] = {
        "wavelengths_nm": active_wavelengths,
        "relative_power": sample_spectral_model(
            (
                resolved_flash.get("resolved_spectral_model")
                if isinstance(resolved_flash.get("resolved_spectral_model"), Mapping)
                else {}
            ),
            active_wavelengths,
        ),
        "normalization": "peak_equals_one",
    }
    data["flash"] = resolved_flash
    return CameraManifest(_with_hashes(_plain(data)))


def with_resolved_optics(
    manifest: CameraManifest,
    *,
    groups: Sequence[Mapping[str, Any]],
    effective_focal_length_mm: float,
    aperture_x_m: float,
    entrance_pupil: Sequence[float],
    exit_pupil: Sequence[float],
    sensor_x_m: float,
    sensor_error_mm: float,
    build_id: str = "",
) -> CameraManifest:
    data = manifest.mapping()
    data["resolved"] = {
        "groups": [dict(item) for item in groups],
        "effective_focal_length_mm": float(effective_focal_length_mm),
        "aperture_x_m": float(aperture_x_m),
        "entrance_pupil": [float(item) for item in entrance_pupil],
        "exit_pupil": [float(item) for item in exit_pupil],
        "exit_pupil_kind": (
            "virtual" if float(exit_pupil[0]) > float(sensor_x_m) else "real"
        ),
        "sensor_x_m": float(sensor_x_m),
        "sensor_error_mm": float(sensor_error_mm),
        "prepared_build_id": str(build_id),
    }
    return CameraManifest(_with_hashes(_plain(data)))


__all__ = [
    "CAMERA_MANIFEST_SCHEMA_VERSION",
    "DEFAULT_WAVELENGTHS_NM",
    "CameraManifest",
    "CameraCompatibilityDecision",
    "camera_compatibility",
    "canonical_json",
    "default_camera_manifest",
    "manifest_hash",
    "resolve_camera_manifest",
    "sample_spectral_model",
    "with_resolved_optics",
]
