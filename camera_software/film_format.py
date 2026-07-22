"""Authoritative physical film-gate and raster contract."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilmFormatSpec:
    key: str
    label: str
    mount_standard: str
    frame_width_mm: float
    frame_height_mm: float
    image_circle_diameter_mm: float
    default_work_width_px: int
    default_work_height_px: int
    default_final_edge_px: int

    @property
    def aspect_ratio(self) -> float:
        return float(self.frame_width_mm) / float(self.frame_height_mm)

    @property
    def frame_corner_radius_mm(self) -> float:
        """Radius required to cover the four corners of the recording gate."""

        return 0.5 * (
            float(self.frame_width_mm) ** 2 + float(self.frame_height_mm) ** 2
        ) ** 0.5

    @property
    def image_circle_radius_mm(self) -> float:
        return 0.5 * float(self.image_circle_diameter_mm)

    def square_sensor_raster(self, content_width: int, content_height: int) -> tuple[int, int]:
        """Return a square-pixel raster having the physical gate's aspect."""

        width = int(content_width)
        height = int(content_height)
        if width <= 0 or height <= 0:
            raise ValueError("content raster dimensions must be positive")
        scale = max(width / self.frame_width_mm, height / self.frame_height_mm)
        return (
            max(width, int(round(self.frame_width_mm * scale))),
            max(height, int(round(self.frame_height_mm * scale))),
        )

    def centered_content_region(self, content_width: int, content_height: int) -> dict[str, int]:
        full_width, full_height = self.square_sensor_raster(content_width, content_height)
        return {
            "x": (full_width - int(content_width)) // 2,
            "y": (full_height - int(content_height)) // 2,
            "width": int(content_width),
            "height": int(content_height),
        }

    def sensor_raster(self, edge_px: int) -> tuple[int, int]:
        """Resolve one live raster control without violating gate aspect."""

        edge = int(edge_px)
        if edge <= 0:
            raise ValueError("sensor raster edge must be positive")
        height = max(1, int(round(edge / self.aspect_ratio)))
        return edge, height


DEFAULT_FILM_FORMAT = FilmFormatSpec(
    key="120_6x6",
    label="120 6x6 (nominal 56 mm square gate)",
    mount_standard="120_6x6",
    frame_width_mm=56.0,
    frame_height_mm=56.0,
    # A nominal 6x6 gate has a 79.196 mm diagonal.  The 80 mm optical
    # contract leaves the complete 56 x 56 mm recording rectangle inside
    # the image circle instead of treating the 56 mm width as its diameter.
    image_circle_diameter_mm=80.0,
    default_work_width_px=256,
    default_work_height_px=256,
    # Four 256-pixel work tiles span each side of the default square image.
    # Both dimensions remain live controls; this is only the initial contract.
    default_final_edge_px=1024,
)


__all__ = ["FilmFormatSpec", "DEFAULT_FILM_FORMAT"]
