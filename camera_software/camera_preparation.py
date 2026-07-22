"""Fast preparation seam between camera manifests and physical lab scenes."""
from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .camera_manifest import CameraManifest, manifest_hash
from .optical_design import OpticalDesignSpec


CAMERA_PREPARATION_ABI = "thick-lens-preparation-v1"


@dataclass(frozen=True)
class CameraPreparationKey:
    manifest_hash: str
    optical_geometry_key: str
    spectral_grid_key: str
    focus_state_key: str
    sensor_geometry_key: str
    backend_abi: str = CAMERA_PREPARATION_ABI

    @property
    def cache_key(self) -> str:
        """Exact reusable physical camera geometry; lighting is independent."""

        return manifest_hash({
            "optical_geometry_key": self.optical_geometry_key,
            "spectral_grid_key": self.spectral_grid_key,
            "focus_state_key": self.focus_state_key,
            "sensor_geometry_key": self.sensor_geometry_key,
            "backend_abi": self.backend_abi,
        }, namespace="prepared-camera-cache-v1")

    @property
    def optical_payload_cache_key(self) -> str:
        """Prescription/spectral payload reusable across focus racks and crops."""

        return manifest_hash({
            "optical_geometry_key": self.optical_geometry_key,
            "spectral_grid_key": self.spectral_grid_key,
            "backend_abi": self.backend_abi,
        }, namespace="prepared-optical-payload-v1")

    @classmethod
    def from_manifest(
        cls, manifest: CameraManifest, *, backend_abi: str = CAMERA_PREPARATION_ABI
    ) -> "CameraPreparationKey":
        return cls(
            manifest_hash=manifest.hash,
            optical_geometry_key=manifest.compatibility_key("optical_geometry"),
            spectral_grid_key=manifest.compatibility_key("spectral_grid"),
            focus_state_key=manifest.compatibility_key("focus_state"),
            sensor_geometry_key=manifest.compatibility_key("sensor_geometry"),
            backend_abi=str(backend_abi),
        )


def apply_camera_manifest_to_lab_scene(
    scene: Any, manifest: Mapping[str, Any]
) -> Any:
    """Make a thick-lens ``SceneConfig`` consume one canonical manifest."""

    lens = dict(manifest.get("lens", {}))
    sensor = dict(manifest.get("sensor", {}))
    flash = dict(manifest.get("flash", {}))
    focal_range = lens.get("focal_length_range_mm", [55.0, 110.0])
    if not isinstance(focal_range, (list, tuple)) or len(focal_range) != 2:
        raise ValueError("camera lens focal_length_range_mm must contain two values")
    width_mm = float(sensor.get("physical_width_mm", 56.0))
    height_mm = float(sensor.get("physical_height_mm", 56.0))
    gate_corner_radius_mm = 0.5 * math.hypot(width_mm, height_mm)
    image_circle_diameter_mm = float(sensor.get(
        "image_circle_diameter_mm", 2.0 * gate_corner_radius_mm
    ))
    if image_circle_diameter_mm + 1.0e-9 < 2.0 * gate_corner_radius_mm:
        raise ValueError(
            "camera sensor image circle does not cover the complete recording gate: "
            f"diameter={image_circle_diameter_mm:.3f}mm "
            f"required>={2.0 * gate_corner_radius_mm:.3f}mm"
        )
    scene.image_plate.sensor_half_w = 0.5e-3 * width_mm
    scene.image_plate.sensor_half_h = 0.5e-3 * height_mm
    scene.image_plate.radius = 0.5e-3 * image_circle_diameter_mm
    scene.screen_radius = float(scene.image_plate.radius)
    prior = getattr(scene, "optical_design", None)
    prior = getattr(prior, "spec", prior)
    scene.optical_design = OpticalDesignSpec(
        focal_length_range_m=(
            float(focal_range[0]) * 1.0e-3,
            float(focal_range[1]) * 1.0e-3,
        ),
        zoom=float(lens.get("zoom", 0.5)),
        focus_distance_m=float(dict(manifest.get("focus", {})).get("distance_m", 1.0)),
        f_number=float(lens.get("f_number", 4.0)),
        entrance_x_m=float(getattr(prior, "entrance_x_m", 1.08)),
        sensor_x_m=float(getattr(prior, "sensor_x_m", 1.25)),
        sensor_clearance_m=float(getattr(prior, "sensor_clearance_m", 0.030)),
        image_radius_m=float(scene.image_plate.radius),
        group_count=int(lens.get("group_count", 4)),
        default_ior=float(lens.get("default_ior", 1.55)),
        min_air_gap_m=float(lens.get("minimum_air_gap_mm", 8.0)) * 1.0e-3,
        group_thickness_m=float(lens.get("group_thickness_mm", 18.0)) * 1.0e-3,
        max_group_radius_m=float(lens.get("maximum_group_radius_mm", 120.0)) * 1.0e-3,
    )
    emitters = {
        str(item.get("key", "")): dict(item)
        for item in flash.get("emitters", ())
        if isinstance(item, Mapping)
    }
    if "side_room_source" in emitters:
        scene.side_room_source_emission = float(
            emitters["side_room_source"].get(
                "emission_scale", scene.side_room_source_emission
            )
        )
    if "camera_ring_light" in emitters:
        ring = emitters["camera_ring_light"]
        scene.ring_light_enabled = bool(ring.get("enabled", True))
        scene.ring_light_emission = float(
            ring.get("emission_scale", scene.ring_light_emission)
        )
    modifier = dict(flash.get("modifier", {}))
    if modifier:
        scene.flash_modifier = dataclasses.replace(
            scene.flash_modifier,
            mode=str(modifier.get("mode", scene.flash_modifier.mode)),
            grid_cell_mm=float(modifier.get("grid_cell_mm", scene.flash_modifier.grid_cell_mm)),
            grid_depth_mm=float(modifier.get("grid_depth_mm", scene.flash_modifier.grid_depth_mm)),
            scrim_transmittance=float(modifier.get(
                "scrim_transmittance", scene.flash_modifier.scrim_transmittance
            )),
        )
    scene.resolved_camera_manifest = copy.deepcopy(dict(manifest))
    return scene


__all__ = [
    "CAMERA_PREPARATION_ABI",
    "CameraPreparationKey",
    "apply_camera_manifest_to_lab_scene",
]
