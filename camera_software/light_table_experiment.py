"""First-light emitter/aperture/exact-lens/sensor chain experiment.

The executor is deliberately narrow and truthful: it uses the production
``CompoundLens.trace`` exact parametric path and the canonical physical
aperture's geometric opening. It does not claim diffraction, finite-material
loss, or arbitrary mixed-domain graph execution; those remain native T4 chain
work. The result is useful now for projection, focus, vignetting, and bokeh
experiments while preserving the compiled chain as the authoritative design.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from camera_designer.compound_optics import ApertureStop, TerminationReason

from .optical_chain import (
    CompiledOpticalChain,
    EmitterEndpointComponent,
    SensorEndpointComponent,
)
from .optical_components import CompoundLensComponent, PhysicalApertureComponent
from .physical_aperture import LivePhysicalAperture


@dataclass(frozen=True)
class LightTableProjection:
    chain_key: str
    linear_sensor_rgb: np.ndarray
    developed_sensor_rgb: np.ndarray
    preview_rgba: np.ndarray
    sensor_plane_x_m: float
    rays_submitted: int
    rays_transmitted: int
    metadata: dict[str, Any]


def mount_aperture_at_lens_stop(
    aperture: LivePhysicalAperture,
    lens: Any,
    *,
    fill_fraction: float = 1.0,
    key: str | None = None,
) -> PhysicalApertureComponent:
    """Scale and place any canonical aperture at the exact lens stop plane."""

    stops = [
        value for value in lens.elements if isinstance(value, ApertureStop)
    ]
    if not stops:
        raise ValueError("compound lens has no authored aperture stop")
    stop = stops[0]
    fill = float(fill_fraction)
    if not math.isfinite(fill) or not 0.0 < fill <= 1.0:
        raise ValueError("fill_fraction must be in (0,1]")
    target_radius = float(stop.r_clear) * fill
    scale = target_radius / float(aperture.assembly_radius_m)
    mounted = replace(
        aperture,
        key=str(key or f"{aperture.key}.lens-mounted"),
        opening_x_m=float(aperture.opening_x_m) * scale,
        opening_y_m=float(aperture.opening_y_m) * scale,
        assembly_radius_m=target_radius,
        thickness_m=float(aperture.thickness_m) * scale,
        pitch_x_m=float(aperture.pitch_x_m) * scale,
        pitch_y_m=float(aperture.pitch_y_m) * scale,
    )
    mounted.validate()
    return PhysicalApertureComponent(
        mounted,
        center_m=(float(stop.x_pos), 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
    )


def _components(chain: CompiledOpticalChain):
    authored = [element.component for element in chain.spec.elements]
    emitters = [value for value in authored if isinstance(
        value, EmitterEndpointComponent
    )]
    apertures = [value for value in authored if isinstance(
        value, PhysicalApertureComponent
    )]
    lenses = [value for value in authored if isinstance(
        value, CompoundLensComponent
    )]
    sensors = [value for value in authored if isinstance(
        value, SensorEndpointComponent
    )]
    if not all(len(values) == 1 for values in (
        emitters, apertures, lenses, sensors,
    )):
        raise ValueError(
            "geometric light-table execution requires exactly one emitter, "
            "aperture, compound lens, and sensor"
        )
    return emitters[0], apertures[0], lenses[0], sensors[0]


def _open_aperture_samples(
    aperture: LivePhysicalAperture,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    accepted: list[np.ndarray] = []
    needed = int(count)
    for _attempt in range(32):
        batch_count = max(needed * 4, 256)
        radius = aperture.assembly_radius_m * np.sqrt(rng.random(batch_count))
        angle = rng.random(batch_count) * (2.0 * math.pi)
        points = np.stack((radius*np.cos(angle), radius*np.sin(angle)), axis=1)
        mask = aperture.open_mask(points[:, 0], points[:, 1])
        if np.any(mask):
            accepted.append(points[mask])
            needed = count-sum(len(value) for value in accepted)
            if needed <= 0:
                break
    if not accepted:
        raise RuntimeError("physical aperture contains no sampled open sites")
    return np.concatenate(accepted, axis=0)[:count]


def _prepare_emitter_content(content: np.ndarray) -> np.ndarray:
    image = np.asarray(content, np.float64)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("emitter content must be HxW, HxWx3, or HxWx4")
    image = np.maximum(image[..., :3], 0.0)
    if not np.all(np.isfinite(image)):
        raise ValueError("emitter content must be finite")
    if float(np.sum(image)) <= 0.0:
        raise ValueError("emitter content contains no emitted power")
    return image


def _develop(
    linear_rgb: np.ndarray, sensor: SensorEndpointComponent
) -> np.ndarray:
    profile = sensor.profile
    chip = profile.sensor_chip
    signal = np.asarray(linear_rgb, np.float64)
    if chip is not None:
        electrons = np.minimum(
            signal
            * float(chip.qe_peak)
            * float(profile.exposure_time_s)
            * float(profile.energy_scale),
            float(chip.full_well_e),
        )
        signal = electrons / max(float(chip.full_well_e), 1.0)
        weights = np.asarray(chip.rgb_sensitivity_weights(), np.float64)
        signal = signal * weights[None, None, :]
    peak = float(np.percentile(np.max(signal, axis=2), 99.5))
    if peak > 0.0:
        signal = signal / peak
    return np.ascontiguousarray(np.clip(signal, 0.0, 1.0), np.float32)


def run_geometric_light_table(
    chain: CompiledOpticalChain,
    emitter_content: np.ndarray,
    *,
    ray_count: int = 20_000,
    seed: int = 42,
) -> LightTableProjection:
    """Project authored content through the exact lens onto the sensor plane."""

    if int(ray_count) < 1:
        raise ValueError("light-table ray_count must be positive")
    emitter, aperture_component, lens_component, sensor = _components(chain)
    content = _prepare_emitter_content(emitter_content)
    rng = np.random.default_rng(int(seed))
    aperture = aperture_component.aperture
    aperture.validate()
    samples = _open_aperture_samples(aperture, int(ray_count), rng)

    luminance = np.sum(content, axis=2)
    probabilities = luminance.reshape(-1)
    probabilities /= float(np.sum(probabilities))
    source_indices = rng.choice(
        probabilities.size, size=int(ray_count), p=probabilities
    )
    source_y, source_x = np.unravel_index(source_indices, luminance.shape)
    height, width = luminance.shape
    radius = float(emitter.emitter.radius)
    source_transverse_y = (
        (source_x + 0.5) / width * 2.0 - 1.0
    ) * radius
    source_transverse_z = (
        1.0 - (source_y + 0.5) / height * 2.0
    ) * radius
    source_plane_x = float(emitter.emitter.pos[0])
    aperture_plane_x = float(aperture_component.center_m[0])
    if aperture_plane_x <= source_plane_x:
        raise ValueError("light-table aperture must lie after the emitter plane")

    sensor_x = float(sensor.center_m[0])
    lens_last_x = max(
        float(getattr(value, "x_pos", -math.inf))
        for value in lens_component.lens.elements
    )
    if sensor_x <= lens_last_x:
        raise ValueError("light-table sensor must lie after the compound lens")

    sensor_width_m = float(sensor.profile.geometry.frame_w_mm) * 1.0e-3
    sensor_height_m = float(sensor.profile.geometry.frame_h_mm) * 1.0e-3
    output_width, output_height = sensor.resolution
    linear = np.zeros((output_height, output_width, 3), np.float64)
    transmitted = 0

    for ray_index in range(int(ray_count)):
        origin = np.asarray((
            source_plane_x,
            source_transverse_y[ray_index],
            source_transverse_z[ray_index],
        ), np.float64)
        target = np.asarray((
            aperture_plane_x,
            samples[ray_index, 0],
            samples[ray_index, 1],
        ), np.float64)
        direction = target-origin
        direction /= np.linalg.norm(direction)
        result = lens_component.lens.trace(origin, direction)
        if result.reason is not TerminationReason.PASSED:
            continue
        if abs(float(result.direction[0])) <= 1.0e-12:
            continue
        distance = (sensor_x-float(result.origin[0]))/float(result.direction[0])
        if distance <= 0.0:
            continue
        hit = result.origin+distance*result.direction
        u = float(hit[1])/sensor_width_m+0.5
        v = 0.5-float(hit[2])/sensor_height_m
        px = int(math.floor(u*output_width))
        py = int(math.floor(v*output_height))
        if not 0 <= px < output_width or not 0 <= py < output_height:
            continue
        pixel_rgb = content[source_y[ray_index], source_x[ray_index]]
        probability = probabilities[source_indices[ray_index]]
        weight = pixel_rgb / max(
            float(probability) * float(ray_count), 1.0e-30
        )
        linear[py, px] += weight
        transmitted += 1

    developed = _develop(linear, sensor)
    preview = np.empty((*developed.shape[:2], 4), np.uint8)
    preview[..., :3] = np.clip(
        np.sqrt(developed) * 255.0, 0.0, 255.0
    ).astype(np.uint8)
    preview[..., 3] = 255
    return LightTableProjection(
        chain_key=chain.spec.key,
        linear_sensor_rgb=np.ascontiguousarray(linear, np.float32),
        developed_sensor_rgb=developed,
        preview_rgba=np.ascontiguousarray(preview),
        sensor_plane_x_m=sensor_x,
        rays_submitted=int(ray_count),
        rays_transmitted=transmitted,
        metadata={
            "transport": "exact-parametric-geometric",
            "lens": lens_component.key,
            "aperture": aperture.graph_parameters(),
            "aperture_interaction": "canonical-open-geometry",
            "diffraction": False,
            "finite_material_loss": False,
            "sensor_contract": sensor.profile.label,
            "output_node": chain.graph.spec.product_keys[0],
        },
    )


__all__ = [
    "LightTableProjection",
    "mount_aperture_at_lens_stop",
    "run_geometric_light_table",
]
