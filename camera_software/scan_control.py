"""Validated control boundary between learned attention and the next scan.

The network proposes *what to do next*; this module owns validation, metrics,
and bounded application.  Transport and scene storage do not depend on a
particular network architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .progressive_exposure import SensorPixelSlice
from .sensor_mipmap import SensorUvBounds


@dataclass(frozen=True)
class FrameDeltaMetrics:
    mean_absolute: float
    root_mean_square: float
    p95_absolute: float
    edge_delta: float
    changed_fraction: float

    @classmethod
    def between(cls, previous: np.ndarray, current: np.ndarray) -> "FrameDeltaMetrics":
        a = np.asarray(previous, dtype=np.float64)
        b = np.asarray(current, dtype=np.float64)
        if a.shape != b.shape or a.ndim not in (2, 3):
            raise ValueError("frame metrics require equally shaped 2-D or 3-D frames")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("frame metrics require finite frames")
        if a.ndim == 3:
            a = np.mean(a[..., :3], axis=2)
            b = np.mean(b[..., :3], axis=2)
        delta = b - a
        absolute = np.abs(delta)
        scale = max(float(np.percentile(np.abs(b), 95.0)), 1.0e-12)
        edge_delta = 0.0
        if min(delta.shape) > 1:
            edge_delta = float(
                np.mean(np.abs(np.diff(delta, axis=0)))
                + np.mean(np.abs(np.diff(delta, axis=1)))
            )
        return cls(
            mean_absolute=float(np.mean(absolute)),
            root_mean_square=float(np.sqrt(np.mean(delta * delta))),
            p95_absolute=float(np.percentile(absolute, 95.0)),
            edge_delta=edge_delta,
            changed_fraction=float(np.mean(absolute > scale * 0.01)),
        )


@dataclass(frozen=True)
class UvRefinementRequest:
    bounds: SensorUvBounds
    target_level: int
    work_value: float = 1.0

    def __post_init__(self) -> None:
        if self.target_level < 0:
            raise ValueError("target_level must be non-negative")
        if not np.isfinite(self.work_value) or self.work_value < 0.0:
            raise ValueError("work_value must be finite and non-negative")


@dataclass(frozen=True)
class PixelSliceRefinementRequest:
    pixel_slice: SensorPixelSlice
    target_level: int
    work_value: float = 1.0

    def __post_init__(self) -> None:
        if self.target_level < 0:
            raise ValueError("target_level must be non-negative")
        if not np.isfinite(self.work_value) or self.work_value < 0.0:
            raise ValueError("work_value must be finite and non-negative")


@dataclass(frozen=True)
class FilmPlaneAdjustment:
    """Physical next-frame film-stage command.

    Values are absolute offsets from the camera's machined-zero film pose, not
    increments from the preceding trial. ``lens_distance_delta_m`` is positive
    when the film moves farther from the lens. Tilts are rotations about the
    zero-pose horizontal/right and vertical/up axes respectively; the lens
    assembly and its optical axis remain fixed.
    """

    lens_distance_delta_m: float = 0.0
    tilt_about_right_deg: float = 0.0
    tilt_about_up_deg: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.lens_distance_delta_m,
            self.tilt_about_right_deg,
            self.tilt_about_up_deg,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("film-plane adjustment values must be finite")


@dataclass(frozen=True)
class NextSiteScan:
    """Network-agnostic command for configuring one subsequent exposure."""

    sequence: int
    uv_requests: tuple[UvRefinementRequest, ...] = ()
    pixel_slice_requests: tuple[PixelSliceRefinementRequest, ...] = ()
    targeted_fraction: float | None = None
    film_plane_adjustment: FilmPlaneAdjustment | None = None
    metrics: FrameDeltaMetrics | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("scan sequence must be non-negative")
        if self.targeted_fraction is not None and not 0.0 <= self.targeted_fraction <= 1.0:
            raise ValueError("targeted_fraction must be in [0, 1]")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "NextSiteScan":
        requests = tuple(
            UvRefinementRequest(
                SensorUvBounds(*map(float, item["uv_bounds"])),
                int(item["target_level"]),
                float(item.get("work_value", 1.0)),
            )
            for item in payload.get("uv_requests", ())
        )
        pixel_requests = tuple(
            PixelSliceRefinementRequest(
                SensorPixelSlice(
                    int(item["width"]),
                    int(item["height"]),
                    tuple(map(int, item["site_indices"])),
                ),
                int(item["target_level"]),
                float(item.get("work_value", 1.0)),
            )
            for item in payload.get("pixel_slice_requests", ())
        )
        adjustment_payload = payload.get("film_plane_adjustment")
        adjustment = None if adjustment_payload is None else FilmPlaneAdjustment(
            lens_distance_delta_m=float(
                adjustment_payload.get("lens_distance_delta_m", 0.0)
            ),
            tilt_about_right_deg=float(
                adjustment_payload.get("tilt_about_right_deg", 0.0)
            ),
            tilt_about_up_deg=float(
                adjustment_payload.get("tilt_about_up_deg", 0.0)
            ),
        )
        return cls(
            sequence=int(payload.get("sequence", 0)),
            uv_requests=requests,
            pixel_slice_requests=pixel_requests,
            targeted_fraction=(None if payload.get("targeted_fraction") is None
                               else float(payload["targeted_fraction"])),
            film_plane_adjustment=adjustment,
            metadata=dict(payload.get("metadata", {})),
        )

    def priority_map(self, height: int, width: int) -> np.ndarray:
        """Rasterize UV requests for upload/combination with learned work value."""
        if height <= 0 or width <= 0:
            raise ValueError("priority-map dimensions must be positive")
        result = np.zeros((height, width), dtype=np.float32)
        for request in self.uv_requests:
            b = request.bounds
            x0 = max(0, min(width - 1, int(np.floor(b.u0 * width))))
            x1 = max(x0 + 1, min(width, int(np.ceil(b.u1 * width))))
            y0 = max(0, min(height - 1, int(np.floor(b.v0 * height))))
            y1 = max(y0 + 1, min(height, int(np.ceil(b.v1 * height))))
            result[y0:y1, x0:x1] = np.maximum(
                result[y0:y1, x0:x1], np.float32(request.work_value)
            )
        for request in self.pixel_slice_requests:
            selected = request.pixel_slice
            index = np.asarray(selected.site_indices, dtype=np.int64)
            source_y, source_x = np.divmod(index, selected.width)
            target_x = np.minimum(
                width - 1,
                ((source_x.astype(np.float64) + 0.5) * width / selected.width).astype(np.int64),
            )
            target_y = np.minimum(
                height - 1,
                ((source_y.astype(np.float64) + 0.5) * height / selected.height).astype(np.int64),
            )
            np.maximum.at(
                result,
                (target_y, target_x),
                np.float32(request.work_value),
            )
        return result

    def active_uv_bounds(self) -> tuple[float, float, float, float] | None:
        """Union every requested region and arbitrary site slice for reporting."""

        requested: list[tuple[float, float, float, float]] = [
            (
                request.bounds.u0,
                request.bounds.v0,
                request.bounds.u1,
                request.bounds.v1,
            )
            for request in self.uv_requests
        ]
        for request in self.pixel_slice_requests:
            selected = request.pixel_slice
            bounds = selected.bounds()
            requested.append((
                bounds.x / selected.width,
                bounds.y / selected.height,
                (bounds.x + bounds.width) / selected.width,
                (bounds.y + bounds.height) / selected.height,
            ))
        if not requested:
            return None
        return (
            min(item[0] for item in requested),
            min(item[1] for item in requested),
            max(item[2] for item in requested),
            max(item[3] for item in requested),
        )


class CameraScanController:
    """Validates network proposals against physical film-stage travel."""

    def __init__(self, *, maximum_film_travel_m: float = 2.0e-3,
                 maximum_film_tilt_deg: float = 5.0) -> None:
        if not np.isfinite(maximum_film_travel_m) or maximum_film_travel_m <= 0.0:
            raise ValueError("maximum film travel must be finite and positive")
        if not np.isfinite(maximum_film_tilt_deg) or maximum_film_tilt_deg <= 0.0:
            raise ValueError("maximum film tilt must be finite and positive")
        self.maximum_film_travel_m = float(maximum_film_travel_m)
        self.maximum_film_tilt_deg = float(maximum_film_tilt_deg)
        self.last_sequence = -1

    def validate(self, command: NextSiteScan) -> FilmPlaneAdjustment | None:
        if command.sequence <= self.last_sequence:
            raise ValueError("next-scan commands must have strictly increasing sequence numbers")
        adjustment = command.film_plane_adjustment
        if adjustment is not None:
            if abs(adjustment.lens_distance_delta_m) > self.maximum_film_travel_m:
                raise ValueError(
                    f"film travel {adjustment.lens_distance_delta_m:g}m exceeds "
                    f"camera limit {self.maximum_film_travel_m:g}m"
                )
            maximum_tilt = max(
                abs(adjustment.tilt_about_right_deg),
                abs(adjustment.tilt_about_up_deg),
            )
            if maximum_tilt > self.maximum_film_tilt_deg:
                raise ValueError(
                    f"film tilt {maximum_tilt:g}deg exceeds camera limit "
                    f"{self.maximum_film_tilt_deg:g}deg"
                )
        self.last_sequence = command.sequence
        return adjustment


class CoverageBalanceController:
    """Closed-loop targeted/coverage split driven by exposure health."""

    def __init__(self, *, preferred_targeted: float = 0.75,
                 minimum_targeted: float = 0.20, response: float = 0.25) -> None:
        self.preferred_targeted = float(np.clip(preferred_targeted, 0.0, 1.0))
        self.minimum_targeted = float(np.clip(minimum_targeted, 0.0, self.preferred_targeted))
        self.response = float(np.clip(response, 0.01, 1.0))
        self.targeted_fraction = self.preferred_targeted

    def update(self, exposure_weight: np.ndarray, noise_rse_p90: float) -> float:
        weight = np.maximum(np.asarray(exposure_weight, dtype=np.float64), 0.0)
        if weight.size == 0:
            return self.targeted_fraction
        positive = weight[weight > 0.0]
        uncovered = 1.0 - float(positive.size) / float(weight.size)
        if positive.size:
            median = max(float(np.median(positive)), 1.0e-12)
            low = float(np.percentile(positive, 10.0)) / median
        else:
            low = 0.0
        noise = float(noise_rse_p90)
        noise_pressure = 1.0 if not np.isfinite(noise) else float(np.clip((noise - 0.25) / 0.75, 0.0, 1.0))
        coverage_pressure = max(uncovered, 1.0 - float(np.clip(low, 0.0, 1.0)), noise_pressure)
        desired = self.preferred_targeted - coverage_pressure * (
            self.preferred_targeted - self.minimum_targeted
        )
        self.targeted_fraction += self.response * (desired - self.targeted_fraction)
        return float(np.clip(self.targeted_fraction, self.minimum_targeted, self.preferred_targeted))


@dataclass(frozen=True)
class FocusTrialResult:
    """One focus trial measured exclusively from a ray-traced sensor frame."""

    adjustment: FilmPlaneAdjustment
    sharpness: float
    confidence: float


def raytraced_focus_score(frame: np.ndarray) -> tuple[float, float]:
    """Return normalized gradient sharpness and usable-signal confidence.

    This consumes the accumulated camera image.  It intentionally has no API
    for an orthographic reference, authored text, depth, or scene geometry.
    """
    image = np.asarray(frame, dtype=np.float64)
    if image.ndim == 3:
        if image.shape[2] < 3:
            raise ValueError("focus frames must have three color channels")
        image = (
            image[..., 0] * 0.2126
            + image[..., 1] * 0.7152
            + image[..., 2] * 0.0722
        )
    if image.ndim != 2 or min(image.shape) < 2 or not np.all(np.isfinite(image)):
        raise ValueError("focus scoring requires a finite 2-D sensor frame")
    image = np.maximum(image, 0.0)
    scale = max(float(np.percentile(image, 99.0)), 1.0e-12)
    normalized = np.clip(image / scale, 0.0, 1.0)
    dx = np.diff(normalized, axis=1)
    dy = np.diff(normalized, axis=0)
    gradient_energy = 0.5 * (
        float(np.mean(dx * dx)) + float(np.mean(dy * dy))
    )
    signal_fraction = float(np.mean(normalized > 0.01))
    confidence = float(np.clip(signal_fraction / 0.10, 0.0, 1.0))
    return gradient_energy, confidence


class FocusCalibrationController:
    """Network-facing focus trial protocol with camera-owned safety rules.

    A learned policy may propose film coordinates, but every proposal becomes
    a full-coverage ``NextSiteScan`` and every objective value comes from the
    ray-traced sensor accumulation.  The controller retains the best observed
    absolute pose; it never accumulates opaque motor deltas between trials.
    """

    def __init__(self, *, first_sequence: int = 0,
                 maximum_film_travel_m: float = 2.0e-3,
                 maximum_film_tilt_deg: float = 5.0) -> None:
        self._next_sequence = int(first_sequence)
        self._validator = CameraScanController(
            maximum_film_travel_m=maximum_film_travel_m,
            maximum_film_tilt_deg=maximum_film_tilt_deg,
        )
        self.results: list[FocusTrialResult] = []

    def request(self, adjustment: FilmPlaneAdjustment, *,
                metadata: Mapping[str, Any] | None = None) -> NextSiteScan:
        command = NextSiteScan(
            sequence=self._next_sequence,
            targeted_fraction=0.0,
            film_plane_adjustment=adjustment,
            metadata={
                "purpose": "raytraced_focus_calibration",
                "film_pose_reference": "machined_zero",
                "coverage": "full_sensor",
                **dict(metadata or {}),
            },
        )
        self._validator.validate(command)
        self._next_sequence += 1
        return command

    def observe(self, adjustment: FilmPlaneAdjustment,
                raytraced_sensor_frame: np.ndarray) -> FocusTrialResult:
        sharpness, confidence = raytraced_focus_score(raytraced_sensor_frame)
        result = FocusTrialResult(adjustment, sharpness, confidence)
        self.results.append(result)
        return result

    @property
    def best(self) -> FocusTrialResult | None:
        if not self.results:
            return None
        return max(self.results, key=lambda result: (result.sharpness, result.confidence))
