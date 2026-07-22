"""Compact film/sensor and flash/lighting controls for the top tool row."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from camera_software.grid_knob_toolbar import GridKnobToolbar
from camera_software.film_format import DEFAULT_FILM_FORMAT
from camera_software.toolbar_manifests import (
    FILM_NAMES as FILM_NAME_OPTIONS,
    FLASH_OPTIONS,
    ISO_OPTIONS,
    LIGHTING_OPTIONS,
    SENSOR_NAMES as SENSOR_NAME_OPTIONS,
    SHUTTER_OPTIONS,
    WORK_DIM_OPTIONS,
    FINAL_EDGE_OPTIONS,
    ALLOCATION_MODE_OPTIONS,
    FOCAL_LENGTH_OPTIONS,
    F_NUMBER_OPTIONS,
    CAMERA_ANGLE_OPTIONS,
    CAMERA_OFFSET_OPTIONS,
    camera_toolbar_panel,
)


@dataclass(frozen=True)
class ExposureControlSettings:
    sensor_id: int = 0
    film_id: int = 0
    iso: int = 100
    shutter_s: float = 1.0 / 60.0
    flash_mode: str = "scene"
    lighting_ev: int = 0
    work_width_px: int = DEFAULT_FILM_FORMAT.default_work_width_px
    work_height_px: int = DEFAULT_FILM_FORMAT.default_work_height_px
    final_edge_px: int = DEFAULT_FILM_FORMAT.default_final_edge_px
    # Begin with the fastest uniformly legible image. Adaptive attention is an
    # explicit opt-in once broad sensor density is no longer the priority.
    allocation_mode: str = "even-sensor"
    focal_length_mm: float = 82.5
    f_number: float = 4.0
    yaw_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    height_offset_m: float = 0.0
    sideways_offset_m: float = 0.0

    def validated(self) -> "ExposureControlSettings":
        if int(self.sensor_id) not in (0, 1):
            raise ValueError("sensor_id must be 0 or 1")
        if int(self.film_id) not in (0, 1):
            raise ValueError("film_id must be 0 or 1")
        if int(self.iso) <= 0 or float(self.shutter_s) <= 0.0:
            raise ValueError("ISO and shutter must be positive")
        if min(
            int(self.work_width_px), int(self.work_height_px),
            int(self.final_edge_px),
        ) <= 0:
            raise ValueError("work and final raster dimensions must be positive")
        if float(self.focal_length_mm) <= 0.0 or float(self.f_number) <= 0.0:
            raise ValueError("focal length and f-number must be positive")
        camera_adjustments = (
            self.yaw_offset_deg, self.pitch_offset_deg,
            self.height_offset_m, self.sideways_offset_m,
        )
        if not all(np.isfinite(float(value)) for value in camera_adjustments):
            raise ValueError("camera pose adjustments must be finite")
        mode = str(self.flash_mode).lower()
        if mode not in {"scene", "on", "off"}:
            raise ValueError("flash_mode must be scene, on, or off")
        allocation_mode = str(self.allocation_mode).strip().lower()
        if allocation_mode not in {"focus-explore", "even-sensor", "n-tree-preview"}:
            raise ValueError(
                "allocation_mode must be focus-explore, even-sensor, or n-tree-preview"
            )
        return replace(
            self, flash_mode=mode, allocation_mode=allocation_mode
        )

    def targeted_fraction(self, focus_explore_fraction: float = 0.75) -> float:
        """Resolve the native attention/coverage split selected by the user."""

        settings = self.validated()
        if settings.allocation_mode == "even-sensor":
            return 0.0
        if settings.allocation_mode == "n-tree-preview":
            return 1.0
        return max(0.0, min(1.0, float(focus_explore_fraction)))

    def apply_to_order(self, order: dict[str, Any]) -> dict[str, Any]:
        settings = self.validated()
        exposure = order["defaults"].setdefault("exposure", {})
        exposure.update({
            "iso": float(settings.iso),
            "time_s": float(settings.shutter_s),
            "sensor_id": int(settings.sensor_id),
            "film_id": int(settings.film_id),
        })
        camera = order["defaults"].setdefault("camera", {})
        camera["focal_mm"] = float(settings.focal_length_mm)
        camera["aperture_mm"] = (
            float(settings.focal_length_mm) / float(settings.f_number)
        )
        manifest = camera.get("manifest")
        if isinstance(manifest, dict):
            lens = manifest.setdefault("lens", {})
            lens["focal_length_mm"] = float(settings.focal_length_mm)
            lens["f_number"] = float(settings.f_number)
            lens["aperture_diameter_mm"] = camera["aperture_mm"]
        if "position_m" in camera and "target_m" in camera:
            position = np.asarray(camera["position_m"], np.float64)
            target = np.asarray(camera["target_m"], np.float64)
            forward = target - position
            view_distance = float(np.linalg.norm(forward))
            if view_distance <= 1.0e-12:
                raise ValueError("camera target must differ from its position")
            forward /= view_distance
            up = np.asarray(camera.get("up", [0.0, 0.0, 1.0]), np.float64)
            up -= forward * float(np.dot(up, forward))
            up_norm = float(np.linalg.norm(up))
            if up_norm <= 1.0e-12:
                raise ValueError("camera up must not be parallel to its view direction")
            up /= up_norm
            right = np.cross(forward, up)
            right /= float(np.linalg.norm(right))

            adjusted_position = (
                position
                + right * float(settings.sideways_offset_m)
                + up * float(settings.height_offset_m)
            )
            yaw = np.deg2rad(float(settings.yaw_offset_deg))
            yaw_forward = forward * np.cos(yaw) + right * np.sin(yaw)
            yaw_right = np.cross(yaw_forward, up)
            yaw_right /= float(np.linalg.norm(yaw_right))
            pitch = np.deg2rad(float(settings.pitch_offset_deg))
            adjusted_forward = yaw_forward * np.cos(pitch) + up * np.sin(pitch)
            adjusted_forward /= float(np.linalg.norm(adjusted_forward))
            adjusted_up = np.cross(yaw_right, adjusted_forward)
            adjusted_up /= float(np.linalg.norm(adjusted_up))
            focus_distance = float(camera.get("focus_distance_m", view_distance))
            adjusted_target = adjusted_position + adjusted_forward * view_distance
            camera["position_m"] = adjusted_position.tolist()
            camera["target_m"] = adjusted_target.tolist()
            camera["up"] = adjusted_up.tolist()
            camera["focus_target_m"] = (
                adjusted_position + adjusted_forward * focus_distance
            ).tolist()
        flash = order["defaults"].setdefault("flash", {})
        flash["intensity_scale"] = float(2.0 ** int(settings.lighting_ev))
        if settings.flash_mode != "scene":
            flash["enabled"] = settings.flash_mode == "on"
        order.setdefault("runtime", {})["sensor_allocation_mode"] = (
            settings.allocation_mode
        )
        return order


class ExposureControlToolbar:
    HEIGHT = GridKnobToolbar.HEIGHT
    SENSOR_NAMES = SENSOR_NAME_OPTIONS
    FILM_NAMES = FILM_NAME_OPTIONS

    def __init__(self, settings: ExposureControlSettings | None = None) -> None:
        self.settings = (settings or ExposureControlSettings()).validated()
        self._iso = ISO_OPTIONS
        self._shutter = SHUTTER_OPTIONS
        self._flash = FLASH_OPTIONS
        self._lighting = LIGHTING_OPTIONS
        self._work_dimensions = WORK_DIM_OPTIONS
        self._final_edges = FINAL_EDGE_OPTIONS
        self._allocation_modes = tuple(
            value.lower().replace(" / ", "-").replace(" ", "-")
            for value in ALLOCATION_MODE_OPTIONS
        )
        self._focal_lengths = FOCAL_LENGTH_OPTIONS
        self._f_numbers = F_NUMBER_OPTIONS
        self._camera_angles = CAMERA_ANGLE_OPTIONS
        self._camera_offsets = CAMERA_OFFSET_OPTIONS
        # The live program composes all six canonical rows through
        # TopToolbarRows. Keep this standalone compatibility surface valid by
        # presenting the first exposure-owned row rather than a mixed row that
        # also needs RayTraceSettings.
        self.panel = camera_toolbar_panel()
        self._grid = GridKnobToolbar(self.panel, palette="amber")

    @staticmethod
    def _cycle(value: Any, options: tuple[Any, ...], delta: int) -> Any:
        index = min(range(len(options)), key=lambda i: abs(options[i] - value)) if isinstance(value, (int, float)) else options.index(value)
        return options[(index + delta) % len(options)]

    def _change(self, field: str, options: tuple[Any, ...], delta: int) -> str:
        self.settings = replace(
            self.settings,
            **{field: self._cycle(getattr(self.settings, field), options, delta)},
        ).validated()
        return "exposure-settings-changed"

    @staticmethod
    def _shutter_text(value: float) -> str:
        return f"1/{round(1 / value)}" if value < 1 else f"{value:g}s"

    def _knob_values(self) -> dict[str, int]:
        settings = self.settings
        return {
            "sensor_id": int(settings.sensor_id),
            "film_id": int(settings.film_id),
            "iso": self._iso.index(settings.iso),
            "shutter_s": min(
                range(len(self._shutter)),
                key=lambda index: abs(self._shutter[index] - settings.shutter_s),
            ),
            "flash_mode": self._flash.index(settings.flash_mode),
            "lighting_ev": self._lighting.index(settings.lighting_ev),
            "work_width_px": min(
                range(len(self._work_dimensions)),
                key=lambda index: abs(
                    self._work_dimensions[index] - settings.work_width_px
                ),
            ),
            "work_height_px": min(
                range(len(self._work_dimensions)),
                key=lambda index: abs(
                    self._work_dimensions[index] - settings.work_height_px
                ),
            ),
            "final_edge_px": min(
                range(len(self._final_edges)),
                key=lambda index: abs(
                    self._final_edges[index] - settings.final_edge_px
                ),
            ),
            "allocation_mode": self._allocation_modes.index(
                settings.allocation_mode
            ),
            "focal_length_mm": self._focal_lengths.index(
                settings.focal_length_mm
            ),
            "f_number": self._f_numbers.index(settings.f_number),
            "yaw_offset_deg": self._camera_angles.index(settings.yaw_offset_deg),
            "pitch_offset_deg": self._camera_angles.index(settings.pitch_offset_deg),
            "height_offset_m": self._camera_offsets.index(settings.height_offset_m),
            "sideways_offset_m": self._camera_offsets.index(settings.sideways_offset_m),
        }

    def render(self, destination: Any, rect: tuple[int, int, int, int]) -> None:
        self._grid.render(destination, rect, self._knob_values())

    def knob_values(self) -> dict[str, int]:
        return self._knob_values()

    def handle_routed(self, field: str, delta: int) -> str:
        options = {
            "sensor_id": (0, 1),
            "film_id": (0, 1),
            "iso": self._iso,
            "shutter_s": self._shutter,
            "flash_mode": self._flash,
            "lighting_ev": self._lighting,
            "work_width_px": self._work_dimensions,
            "work_height_px": self._work_dimensions,
            "final_edge_px": self._final_edges,
            "allocation_mode": self._allocation_modes,
            "focal_length_mm": self._focal_lengths,
            "f_number": self._f_numbers,
            "yaw_offset_deg": self._camera_angles,
            "pitch_offset_deg": self._camera_angles,
            "height_offset_m": self._camera_offsets,
            "sideways_offset_m": self._camera_offsets,
        }[field]
        return self._change(field, options, delta)

    def handle_event(self, event: Any, _rect: tuple[int, int, int, int]) -> str | None:
        routed = self._grid.route_event(event, _rect)
        if routed is None:
            return None
        field, delta = routed
        return self.handle_routed(field, delta)


__all__ = ["ExposureControlSettings", "ExposureControlToolbar"]
