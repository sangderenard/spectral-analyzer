"""Visual confirmation for the production T4 angular-spectrum transform.

This is not a second wave solver.  It drives
``RayTracer.t4_angular_spectrum_step`` -- the same native kernel used by
persistent T4 arenas -- and only performs presentation/capture in Python.

The source is a pair of finite coherent Gaussian emitters.  No ideal aperture
or hard array mask is used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _calibration_tracer(wavelength_m: float):
    import _spectral_kernels as kernels
    from ray_tracer_bridge import per_tri_spectral_to_mat_buf

    frequencies = np.asarray([299_792_458.0 / wavelength_m], np.float64)
    reflectance = np.zeros((1, 1), np.float64)
    mat_idx, mat_buf, mat_count = per_tri_spectral_to_mat_buf(
        reflectance, reflectance, reflectance, frequencies
    )
    triangle = np.asarray(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]], np.float64
    )
    normal = np.asarray([[0.0, 0.0, 1.0]], np.float64)
    return kernels.RayTracer(
        1, triangle, normal, mat_idx, mat_buf, int(mat_count),
        frequencies, 299_792_458.0, np.zeros(1, np.float64),
    )


def _transport_calibration_tracer(wavelength_m: float):
    """Build a tiny detector bench around the native pipeline's real T4 port."""

    import _spectral_kernels as kernels
    from ray_tracer_bridge import per_tri_spectral_to_mat_buf

    frequency = np.asarray([299_792_458.0 / wavelength_m], np.float64)
    reflectance = np.zeros((2, 1), np.float64)
    mat_idx, mat_buf, mat_count = per_tri_spectral_to_mat_buf(
        reflectance, reflectance, reflectance, frequency
    )
    detector_z = 450.0e-6
    extent = 500.0e-6
    triangles = np.asarray([
        [-extent, -extent, detector_z, extent, -extent, detector_z,
         extent, extent, detector_z],
        [-extent, -extent, detector_z, extent, extent, detector_z,
         -extent, extent, detector_z],
    ], np.float64)
    normals = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], np.float64)
    tracer = kernels.RayTracer(
        2, triangles, normals, mat_idx, mat_buf, int(mat_count),
        frequency, 299_792_458.0, np.zeros(1, np.float64),
    )
    context_ids = []
    axis_payloads = []
    for center_z in (-64.0e-6, 64.0e-6):
        payload = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float64
        )
        axis_payloads.append(payload)
        context_ids.append(tracer.add_scale_context(
            np.asarray([0.0, 0.0, center_z], np.float64),
            96.0e-6,
            1,
            4.0e-6,
            32,
            1.0,
            0.0,
            1,
            payload,
        ))
    tracer.add_wave_context_link(context_ids[0], context_ids[1])
    tracer.ensure_pipeline(max_children=1, min_amplitude=1.0e-12)
    return tracer


def _source_field(size: int, pitch_m: float) -> np.ndarray:
    coordinate = (
        np.arange(size, dtype=np.float64) - (size - 1) * 0.5
    ) * pitch_m
    x, y = np.meshgrid(coordinate, coordinate)
    waist = size * pitch_m * 0.075
    separation = size * pitch_m * 0.12
    tilt = 2.0 * np.pi / (size * pitch_m * 0.28)
    left = np.exp(-((x + separation) ** 2 + y**2) / (waist**2))
    right = 0.82 * np.exp(-((x - separation) ** 2 + y**2) / (waist**2))
    return (
        left * np.exp(1j * tilt * x)
        + right * np.exp(1j * (-0.7 * tilt * y + 0.72))
    ).astype(np.complex64)


def _remove_piston_phase(field: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove one intensity-weighted global phase without changing amplitude.

    This is a presentation gauge choice only. It removes the spatially uniform
    carrier/time-of-flight rotation ("piston") while retaining every relative
    phase difference that describes wavefront shape and interference.
    """

    complex_field = np.asarray(field, np.complex128)
    amplitude = np.abs(complex_field)
    peak = float(np.max(amplitude))
    if peak <= 0.0:
        return complex_field.copy(), 0.0
    valid = amplitude > peak*1.0e-8
    phasor = np.sum(complex_field[valid]*amplitude[valid])
    if abs(phasor) <= np.finfo(np.float64).eps*np.sum(amplitude[valid]**2):
        peak_index = np.unravel_index(np.argmax(amplitude), amplitude.shape)
        piston = float(np.angle(complex_field[peak_index]))
    else:
        piston = float(np.angle(phasor))
    return complex_field*np.exp(-1j*piston), piston


def _phase_rgba(
    field: np.ndarray, *, remove_piston: bool = False,
) -> np.ndarray:
    if remove_piston:
        field, _ = _remove_piston_phase(field)
    amplitude = np.abs(field).astype(np.float64)
    peak = float(np.max(amplitude))
    value = np.zeros_like(amplitude) if peak <= 0.0 else np.power(
        amplitude / peak, 0.42
    )
    hue = (np.angle(field).astype(np.float64) + np.pi) / (2.0 * np.pi)
    saturation = np.full_like(hue, 0.88)
    scaled = hue * 6.0
    sector = np.floor(scaled).astype(np.int32) % 6
    fraction = scaled - np.floor(scaled)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    rgb = np.zeros((*field.shape, 3), np.float64)
    choices = (
        (value, t, p), (q, value, p), (p, value, t),
        (p, q, value), (t, p, value), (value, p, q),
    )
    for index, channels in enumerate(choices):
        mask = sector == index
        for channel, values in enumerate(channels):
            rgb[..., channel][mask] = values[mask]
    alpha = (amplitude > peak * 1.0e-8).astype(np.float64)
    return np.clip(
        np.concatenate((rgb, alpha[..., None]), axis=2) * 255.0,
        0.0, 255.0,
    ).astype(np.uint8)


def _scalar_rgba(values: np.ndarray, *, relative_peak: float | None = None) -> np.ndarray:
    positive = np.maximum(np.asarray(values, np.float64), 0.0)
    peak = float(np.max(positive)) if relative_peak is None else relative_peak
    normalized = np.zeros_like(positive) if peak <= 0.0 else np.clip(
        positive / peak, 0.0, 1.0
    )
    mapped = np.sqrt(normalized)
    rgb = np.stack(
        (0.16 * mapped, 0.72 * mapped, mapped), axis=2
    )
    alpha = (positive > peak * 1.0e-10).astype(np.float64)
    return np.clip(
        np.concatenate((rgb, alpha[..., None]), axis=2) * 255.0,
        0.0, 255.0,
    ).astype(np.uint8)


def _signed_scalar_rgba(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Diverging blue/black/red display for signed Stokes components."""

    signed = np.asarray(values, np.float64)
    support_value = np.maximum(np.asarray(scale, np.float64), 0.0)
    support_peak = float(np.max(support_value))
    denominator = np.maximum(support_value, 1.0e-30)
    normalized = np.clip(signed / denominator, -1.0, 1.0)
    magnitude = np.sqrt(np.abs(normalized))
    rgb = np.zeros((*signed.shape, 3), np.float64)
    positive = normalized >= 0.0
    rgb[..., 0] = np.where(positive, magnitude, 0.12 * magnitude)
    rgb[..., 1] = 0.16 * magnitude
    rgb[..., 2] = np.where(positive, 0.12 * magnitude, magnitude)
    alpha = (
        np.zeros_like(support_value)
        if support_peak <= 0.0 else
        np.power(np.clip(support_value/support_peak, 0.0, 1.0), 0.35)
    )
    return np.clip(
        np.concatenate((rgb, alpha[..., None]), axis=2) * 255.0,
        0.0, 255.0,
    ).astype(np.uint8)


def _polarization_rgba(stokes: tuple[np.ndarray, ...]) -> np.ndarray:
    """Map polarization orientation to hue and degree to saturation."""

    intensity, q, u, v = (np.asarray(value, np.float64) for value in stokes)
    peak = float(np.max(intensity))
    value = np.zeros_like(intensity) if peak <= 0.0 else np.sqrt(
        np.clip(intensity / peak, 0.0, 1.0)
    )
    degree = np.clip(
        np.sqrt(q*q + u*u + v*v) / np.maximum(intensity, 1.0e-30),
        0.0, 1.0,
    )
    orientation = 0.5 * np.arctan2(u, q)
    hue = (orientation / np.pi + 0.5) % 1.0
    # Circular handedness brightens red/blue around the orientation hue while
    # retaining a continuous orientation map for linear states.
    handed = np.clip(v / np.maximum(intensity, 1.0e-30), -1.0, 1.0)
    hue = (hue + 0.14 * handed) % 1.0
    saturation = 0.18 + 0.82 * degree
    scaled = hue * 6.0
    sector = np.floor(scaled).astype(np.int32) % 6
    fraction = scaled - np.floor(scaled)
    p = value * (1.0 - saturation)
    qv = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    rgb = np.zeros((*intensity.shape, 3), np.float64)
    choices = (
        (value, t, p), (qv, value, p), (p, value, t),
        (p, qv, value), (t, p, value), (value, p, qv),
    )
    for index, channels in enumerate(choices):
        mask = sector == index
        for channel, channel_values in enumerate(channels):
            rgb[..., channel][mask] = channel_values[mask]
    alpha = (intensity > peak * 1.0e-10).astype(np.float64)
    return np.clip(
        np.concatenate((rgb, alpha[..., None]), axis=2) * 255.0,
        0.0, 255.0,
    ).astype(np.uint8)


def _spectral_power_rgba(
    band_power: np.ndarray,
    wavelengths_m: np.ndarray,
) -> np.ndarray:
    """Integrate resolved band power using the renderer's spectral RGB map."""

    from camera_software.transport_contract import native_display_rgb_weight

    power = np.maximum(np.asarray(band_power, np.float64), 0.0)
    wavelengths = np.asarray(wavelengths_m, np.float64).reshape(-1)
    if power.ndim != 3 or power.shape[0] != len(wavelengths):
        raise ValueError("spectral power requires [band,y,x] and wavelengths")
    weights = np.asarray([
        native_display_rgb_weight(value * 1.0e9) for value in wavelengths
    ], np.float64)
    rgb = np.einsum("byx,bc->yxc", power, weights, optimize=True)
    peak = float(np.percentile(np.max(rgb, axis=2), 99.8))
    if peak > 0.0:
        rgb = np.sqrt(np.clip(rgb / peak, 0.0, 1.0))
    alpha = (
        np.max(power, axis=0)
        > max(float(np.max(power))*1.0e-10, 1.0e-30)
    ).astype(np.float64)
    return np.clip(
        np.concatenate((rgb, alpha[..., None]), axis=2) * 255.0,
        0.0, 255.0,
    ).astype(np.uint8)


_APERTURE_POLARIZATION_MODES = (
    "linear", "circular+", "circular-", "radial", "azimuthal",
    "partial", "unpolarized",
)
_APERTURE_QUALITY_MODES = ("balanced", "high", "bake")
_APERTURE_SPECTRAL_MODES = ("fixed", "continuous")
_APERTURE_LANE_COUNTS = (1, 3, 4, 8, 16, 32)


def _aperture_polarization_state(name: str, angle_deg: float = 0.0):
    from camera_designer.emitter_profile import (
        PolarizationMode,
        PolarizationState,
    )

    key = str(name).strip().lower()
    if key == "linear":
        return PolarizationState(
            mode=PolarizationMode.LINEAR, angle_deg=float(angle_deg)
        )
    if key in {"circular+", "circular-"}:
        return PolarizationState(
            mode=PolarizationMode.CIRCULAR,
            handedness=1 if key.endswith("+") else -1,
        )
    if key == "radial":
        return PolarizationState(mode=PolarizationMode.RADIAL)
    if key == "azimuthal":
        return PolarizationState(mode=PolarizationMode.AZIMUTHAL)
    if key == "partial":
        return PolarizationState(
            mode=PolarizationMode.LINEAR,
            angle_deg=float(angle_deg),
            degree_of_polarization=0.45,
        )
    if key == "unpolarized":
        return PolarizationState(mode=PolarizationMode.UNPOLARIZED)
    raise ValueError(
        f"unknown aperture polarization {name!r}; "
        f"expected {_APERTURE_POLARIZATION_MODES}"
    )


def _aperture_spectral_samples(
    mode: str,
    lane_count: int,
    *,
    wavelength_m: float,
    sample_epoch: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return wavelengths and normalized field amplitudes for an experiment."""

    key = str(mode).strip().lower()
    count = int(lane_count)
    if key not in _APERTURE_SPECTRAL_MODES:
        raise ValueError(
            f"spectral mode must be one of {_APERTURE_SPECTRAL_MODES}"
        )
    if count not in _APERTURE_LANE_COUNTS:
        raise ValueError(
            f"spectral lane count must be one of {_APERTURE_LANE_COUNTS}"
        )
    if key == "fixed":
        if count == 1:
            wavelengths = np.asarray([float(wavelength_m)], np.float64)
            weights = np.ones(1, np.float64)
        else:
            from camera_software.transport_contract import (
                perceptual_visible_bins,
            )

            bins = perceptual_visible_bins(count)
            wavelengths = np.asarray(
                [center*1.0e-9 for _lower, center, _upper in bins],
                np.float64,
            )
            weights = np.asarray(
                [upper-lower for lower, _center, upper in bins],
                np.float64,
            )
            weights /= float(np.sum(weights))
    else:
        # A deterministic shifted stratification in frequency, not wavelength:
        # lanes are arbitrary continuous samples and never become fixed bins.
        speed = 299_792_458.0
        frequency_min = speed / 700.0e-9
        frequency_max = speed / 400.0e-9
        shift = (int(sample_epoch) * 0.6180339887498949) % 1.0
        u = (
            (np.arange(count, dtype=np.float64) + 0.5) / count + shift
        ) % 1.0
        frequencies = frequency_min + u*(frequency_max-frequency_min)
        wavelengths = speed/frequencies
        weights = np.full(count, 1.0/count, np.float64)
    return (
        np.ascontiguousarray(wavelengths),
        np.ascontiguousarray(np.sqrt(weights), np.float32),
    )


def _labelled_panel(rgba: np.ndarray, label: str, scale: int) -> Image.Image:
    panel = Image.fromarray(rgba, "RGBA")
    if scale != 1:
        panel = panel.resize(
            (panel.width * scale, panel.height * scale), Image.Resampling.NEAREST
        )
    header = 26
    result = Image.new("RGBA", (panel.width, panel.height + header), (8, 11, 18, 255))
    result.paste(panel, (0, header), panel)
    ImageDraw.Draw(result).text(
        (8, 6), label, fill=(220, 230, 242, 255), font=ImageFont.load_default()
    )
    return result


def _compose(panels: list[Image.Image], footer: str) -> Image.Image:
    gap = 8
    footer_h = 30
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels) + footer_h
    image = Image.new("RGBA", (width, height), (4, 7, 12, 255))
    x = 0
    for panel in panels:
        image.paste(panel, (x, 0), panel)
        x += panel.width + gap
    ImageDraw.Draw(image).text(
        (8, height - footer_h + 8), footer,
        fill=(150, 175, 205, 255), font=ImageFont.load_default(),
    )
    return image


def _compose_grid(
    panels: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    *,
    scale: int,
    footer: str,
) -> Image.Image:
    """Compose six scientific panels into the same 3x2 shape as the live UI."""

    if len(panels) != 6 or len(labels) != 6:
        raise ValueError("aperture plates require exactly six panels")
    labelled = [
        _labelled_panel(panel, label, scale)
        for panel, label in zip(panels, labels)
    ]
    gap = 8
    footer_h = 34
    cell_width = max(panel.width for panel in labelled)
    cell_height = max(panel.height for panel in labelled)
    width = 3*cell_width + 2*gap
    height = 2*cell_height + gap + footer_h
    image = Image.new("RGBA", (width, height), (4, 7, 12, 255))
    for index, panel in enumerate(labelled):
        row, column = divmod(index, 3)
        image.paste(
            panel,
            (column*(cell_width+gap), row*(cell_height+gap)),
            panel,
        )
    ImageDraw.Draw(image).text(
        (8, height-footer_h+9), footer,
        fill=(150, 175, 205, 255), font=ImageFont.load_default(),
    )
    return image


def _transport_table_rgba(
    arena: dict[str, object] | list[dict[str, object]],
    detector_position: np.ndarray | None,
    size: int,
) -> np.ndarray:
    """Draw actual native boundary telemetry as an orthographic X/Z table."""

    image = Image.new("RGBA", (size, size), (7, 11, 18, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    arenas = arena if isinstance(arena, list) else [arena]
    first = next(
        (value for value in arenas if int(value["next_forward"]) >= 0),
        arenas[0],
    )
    terminal = next(
        (value for value in arenas if int(value["next_forward"]) < 0),
        arenas[-1],
    )
    first_boundary = dict(first["boundary"])
    terminal_boundary = dict(terminal["boundary"])
    entry = np.asarray(first_boundary["entry_world"], np.float64)
    exit_position = np.asarray(terminal_boundary["exit_world"], np.float64)
    entry_direction = np.asarray(
        first_boundary["entry_direction"], np.float64
    )
    exit_direction = np.asarray(
        terminal_boundary["exit_direction"], np.float64
    )
    z_min, z_max = -850.0e-6, 520.0e-6
    x_min, x_max = -260.0e-6, 260.0e-6

    def point(world: np.ndarray) -> tuple[int, int]:
        px = int(round(
            26 + (float(world[2]) - z_min) / (z_max-z_min) * (size-52)
        ))
        py = int(round(
            size-28 - (float(world[0]) - x_min) / (x_max-x_min) * (size-64)
        ))
        return px, py

    for index, value in enumerate(arenas):
        center = np.asarray(value["center_world"], np.float64)
        half_x = float(value["transverse_half_extent_m"])
        half_z = 0.5 * float(value["longitudinal_extent_m"])
        patch_lo = point(center + np.asarray([-half_x, 0.0, -half_z]))
        patch_hi = point(center + np.asarray([half_x, 0.0, half_z]))
        rectangle = (
            min(patch_lo[0], patch_hi[0]), min(patch_lo[1], patch_hi[1]),
            max(patch_lo[0], patch_hi[0]), max(patch_lo[1], patch_hi[1]),
        )
        color = (55, 210, 245, 255) if index == 0 else (165, 110, 255, 255)
        draw.rectangle(
            rectangle, fill=(18, 48, 72, 220), outline=color, width=2
        )
        draw.text(
            (rectangle[0]+5, rectangle[1]+5), f"T4-{index}",
            font=font, fill=color,
        )

    source = entry - entry_direction * 700.0e-6
    draw.line((point(source), point(entry)), fill=(255, 190, 72, 255), width=3)
    draw.ellipse(
        (point(source)[0]-5, point(source)[1]-5,
         point(source)[0]+5, point(source)[1]+5),
        fill=(255, 150, 45, 255),
    )
    endpoint = detector_position
    if endpoint is None:
        endpoint = exit_position + exit_direction * 380.0e-6
    draw.line(
        (point(exit_position), point(np.asarray(endpoint))),
        fill=(120, 255, 150, 255), width=3,
    )
    detector_a = point(np.asarray([-240.0e-6, 0.0, 450.0e-6]))
    detector_b = point(np.asarray([240.0e-6, 0.0, 450.0e-6]))
    draw.line((detector_a, detector_b), fill=(230, 235, 245, 255), width=3)
    if detector_position is not None:
        hit = point(detector_position)
        draw.ellipse((hit[0]-6, hit[1]-6, hit[0]+6, hit[1]+6),
                     outline=(255, 80, 190, 255), width=2)

    draw.text((8, 8), "T1 -> T4-0 ==FIELD==> T4-1 -> T1", font=font,
              fill=(225, 235, 250, 255))
    draw.text(
        (8, size-48),
        f"entry x={entry[0]*1e6:+.2f} um  exit x={exit_position[0]*1e6:+.2f} um",
        font=font, fill=(155, 180, 210, 255),
    )
    draw.text(
        (8, size-32),
        f"dir x {entry_direction[0]:+.4f} -> {exit_direction[0]:+.4f}",
        font=font, fill=(155, 180, 210, 255),
    )
    return np.asarray(image, np.uint8)


def _aperture_geometry_rgba(
    aperture, size: int, *, view_radius_m: float | None = None,
) -> np.ndarray:
    """Orthographic projection of the actual finite blade triangle mesh."""

    vertices, _ = aperture.triangle_mesh()
    image = Image.new("RGBA", (size, size), (7, 11, 18, 255))
    draw = ImageDraw.Draw(image)
    radius = (
        aperture.assembly_radius_m
        if view_radius_m is None else float(view_radius_m)
    )

    def point(value):
        return (
            int(round((0.5+0.46*float(value[0])/radius)*size)),
            int(round((0.5-0.46*float(value[1])/radius)*size)),
        )

    for triangle in vertices.reshape(-1, 3, 3):
        # Draw only the front/back faces in this top-down projection; side
        # triangles collapse to lines and are represented by the outline.
        if np.ptp(triangle[:, 2]) > 1.0e-15:
            continue
        polygon = [point(vertex) for vertex in triangle]
        draw.polygon(
            polygon, fill=(70, 76, 88, 255), outline=(155, 172, 194, 255)
        )
    return np.asarray(image, np.uint8)


def _aperture_sweep_radius(
    cycle_phase: float, size: int, pitch_m: float,
) -> tuple[float, float, float]:
    """Return (opening radius, assembly radius, normalized sweep position)."""

    field_radius = 0.5*(size-1)*pitch_m
    assembly_radius = field_radius*1.62
    pinhole_radius = pitch_m*0.76
    clear_field_radius = field_radius*1.51
    sweep = 0.5-0.5*np.cos(cycle_phase)
    opening = np.exp(
        np.log(pinhole_radius)
        + sweep*np.log(clear_field_radius/pinhole_radius)
    )
    return float(opening), float(assembly_radius), float(sweep)


def _solve_aperture_visual(
    tracer,
    *,
    size: int,
    pitch_m: float,
    wavelength_m: float,
    cycle_phase: float,
    polarization_mode: str,
    quality: str,
    spectral_mode: str,
    lane_count: int,
    spectral_epoch: int = 0,
    source_angle_deg: float = 0.0,
    analyzer_angle_deg: float = 0.0,
    vector_page: bool = True,
    piston_removed: bool = True,
    coherence_seed: int = 0,
    spectral_panel: bool = False,
) -> dict[str, object]:
    """Solve one physical-aperture frame through shared production kernels."""

    from camera_software.physical_aperture import LivePhysicalAperture
    from camera_software.vector_wave_adapter import (
        JonesFieldState,
        PaddedWaveDomain,
    )

    opening, _display_extent, sweep = _aperture_sweep_radius(
        cycle_phase, size, pitch_m,
    )
    wavelengths, spectral_amplitudes = _aperture_spectral_samples(
        spectral_mode,
        lane_count,
        wavelength_m=wavelength_m,
        sample_epoch=spectral_epoch,
    )
    domain = PaddedWaveDomain.for_quality((size, size), quality)
    solve_height, solve_width = domain.solve_shape
    aperture_extent = 0.51 * math.hypot(
        (solve_width-1)*pitch_m,
        (solve_height-1)*pitch_m,
    )
    aperture = LivePhysicalAperture.iris(
        "demo.physical-iris",
        blade_count=9,
        opening_radius_m=float(opening),
        assembly_radius_m=aperture_extent,
        thickness_m=0.10e-6,
        rotation_rad=cycle_phase*0.2,
        material_name="blackened_steel",
        material_n_real=2.9,
        material_n_imag=3.0,
    )
    payload = aperture.wave_payload()
    scalar_field = domain.uniform_scalar_field(bands=lane_count)
    scalar_field *= spectral_amplitudes[:, None, None]
    initial = JonesFieldState.from_scalar_field(
        scalar_field,
        _aperture_polarization_state(
            polarization_mode, source_angle_deg,
        ),
        coherence_seed=int(coherence_seed),
    )
    material, accounting = initial.apply_isotropic_material_native(
        tracer,
        pitch_m=pitch_m,
        distance_m=aperture.thickness_m,
        direction_sign=1,
        wavelengths_m=wavelengths,
        payload=payload,
    )
    distance = 90.0e-6
    forward, boundary = material.propagate_open_native(
        tracer,
        pitch_m=pitch_m,
        distance_m=distance,
        direction_sign=1,
        wavelengths_m=wavelengths,
        domain=domain,
    )
    forward_stokes = tuple(domain.crop(value) for value in forward.stokes())
    geometry = _aperture_geometry_rgba(
        aperture,
        size,
        view_radius_m=0.5*(size-1)*pitch_m,
    )
    spectral_power = domain.crop(np.sum(
        np.abs(forward.fields.astype(np.complex128))**2,
        axis=(0, 1),
    ))
    spectral_rgba = _spectral_power_rgba(spectral_power, wavelengths)

    if vector_page:
        if spectral_panel:
            panels = (
                geometry,
                spectral_rgba,
                _scalar_rgba(forward_stokes[0]),
                _polarization_rgba(forward_stokes),
                _signed_scalar_rgba(forward_stokes[1], forward_stokes[0]),
                _signed_scalar_rgba(forward_stokes[3], forward_stokes[0]),
            )
            labels = (
                "PHYSICAL BLADES",
                "SPECTRAL POWER / RGB",
                "STOKES I / POWER",
                "POLARIZATION",
                "STOKES Q / I",
                "STOKES V / I",
            )
        else:
            panels = (
                geometry,
                _scalar_rgba(forward_stokes[0]),
                _polarization_rgba(forward_stokes),
                _signed_scalar_rgba(forward_stokes[1], forward_stokes[0]),
                _signed_scalar_rgba(forward_stokes[3], forward_stokes[0]),
                _scalar_rgba(domain.crop(
                    forward.analyzer_intensity(analyzer_angle_deg)
                )),
            )
            labels = (
                "PHYSICAL BLADES",
                "STOKES I / POWER",
                "POLARIZATION",
                "STOKES Q / I",
                "STOKES V / I",
                f"ANALYZER {analyzer_angle_deg%180.0:.0f} DEG",
            )
    else:
        reverse_material, _reverse_accounting = (
            initial.apply_isotropic_material_native(
                tracer,
                pitch_m=pitch_m,
                distance_m=aperture.thickness_m,
                direction_sign=-1,
                wavelengths_m=wavelengths,
                payload=payload,
            )
        )
        reverse, _reverse_boundary = reverse_material.propagate_open_native(
            tracer,
            pitch_m=pitch_m,
            distance_m=distance,
            direction_sign=-1,
            wavelengths_m=wavelengths,
            domain=domain,
        )
        reverse_stokes = tuple(domain.crop(value) for value in reverse.stokes())
        material_s = domain.crop(material.fields[0, 0, 0])
        material_p = domain.crop(material.fields[0, 1, 0])
        forward_s = domain.crop(forward.fields[0, 0, 0])
        forward_p = domain.crop(forward.fields[0, 1, 0])
        panels = (
            geometry,
            _phase_rgba(material_s, remove_piston=piston_removed),
            _phase_rgba(material_p, remove_piston=piston_removed),
            _phase_rgba(forward_s, remove_piston=piston_removed),
            _phase_rgba(forward_p, remove_piston=piston_removed),
            _polarization_rgba(reverse_stokes),
        )
        labels = (
            "PHYSICAL BLADES",
            f"MODE 0 {wavelengths[0]*1e9:.1f}NM MATERIAL S",
            f"MODE 0 {wavelengths[0]*1e9:.1f}NM MATERIAL P",
            f"MODE 0 {wavelengths[0]*1e9:.1f}NM FORWARD S",
            f"MODE 0 {wavelengths[0]*1e9:.1f}NM FORWARD P",
            "REVERSE POLARIZATION",
        )

    input_power = float(accounting["input_power"])
    output_power = float(accounting["output_power"])
    return {
        "panels": panels,
        "labels": labels,
        "opening_radius_m": float(opening),
        "sweep": float(sweep),
        "wavelengths_m": wavelengths.copy(),
        "solve_shape": (int(solve_height), int(solve_width)),
        "propagation_steps": int(domain.propagation_steps),
        "retention": output_power/max(input_power, 1.0e-30),
        "border_fraction": float(boundary["border_fraction"]),
        "mode_count": int(forward.mode_count),
    }


def _boundary_power_rgba(
    arena: dict[str, object],
    size: int,
    linked_arena: dict[str, object] | None = None,
) -> np.ndarray:
    image = Image.new("RGBA", (size, size), (7, 11, 18, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    boundary = dict(arena["boundary"])
    values = [("INPUT RAY", float(boundary["input_ray_power"]),
               (255, 185, 70, 255))]
    values.append(("T4-0 EXIT FIELD", float(boundary["propagated_field_power"]),
                   (80, 205, 255, 255)))
    if linked_arena is not None:
        linked = dict(linked_arena["boundary"])
        values.extend([
            ("T4-1 LINK INPUT", float(linked["seeded_field_power"]),
             (165, 110, 255, 255)),
            ("T4-1 EXIT FIELD", float(linked["propagated_field_power"]),
             (120, 125, 255, 255)),
            ("OUTPUT RAY", float(linked["output_ray_power"]),
             (100, 255, 150, 255)),
        ])
    else:
        values.extend([
            ("SEEDED FIELD", float(boundary["seeded_field_power"]),
             (80, 205, 255, 255)),
            ("PROPAGATED", float(boundary["propagated_field_power"]),
             (120, 125, 255, 255)),
            ("OUTPUT RAY", float(boundary["output_ray_power"]),
             (100, 255, 150, 255)),
        ])
    peak = max([value for _label, value, _color in values] + [1.0e-30])
    draw.text((8, 8), "BOUNDARY POWER CONTRACT", font=font,
              fill=(225, 235, 250, 255))
    for index, (label, value, color) in enumerate(values):
        y = 42 + index * max(30, (size-75)//len(values))
        draw.text((10, y), f"{label}  {value:.6e}", font=font, fill=color)
        bar_width = int(round((size-24) * min(1.0, value/peak)))
        draw.rectangle((10, y+16, 10+bar_width, y+28), fill=color)
    seeded = float(boundary["seeded_field_power"])
    propagated = float(
        dict(linked_arena["boundary"])["propagated_field_power"]
        if linked_arena is not None
        else boundary["propagated_field_power"]
    )
    draw.text(
        (8, size-24),
        f"field retention {propagated/max(seeded, 1e-30):.8f}",
        font=font, fill=(155, 180, 210, 255),
    )
    return np.asarray(image, np.uint8)


def render_sequence(
    output_dir: str | Path,
    *,
    size: int = 128,
    frames: int = 12,
    pitch_m: float = 1.5e-6,
    step_m: float = 25.0e-6,
    wavelength_m: float = 532.0e-9,
    scale: int = 2,
) -> dict[str, object]:
    if size < 8 or size & (size - 1):
        raise ValueError("size must be a power of two and at least 8")
    if frames < 1:
        raise ValueError("frames must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tracer = _calibration_tracer(wavelength_m)
    wavelengths = np.asarray([wavelength_m], np.float64)
    initial = _source_field(size, pitch_m)
    re = np.ascontiguousarray(initial.real[None], np.float32)
    im = np.ascontiguousarray(initial.imag[None], np.float32)
    initial_power = np.abs(initial) ** 2
    frame_paths: list[str] = []

    for frame_index in range(frames + 1):
        field = re[0].astype(np.float64) + 1j * im[0].astype(np.float64)
        power = np.abs(field) ** 2
        delta = np.abs(field - initial)
        panels = [
            _labelled_panel(
                _scalar_rgba(power, relative_peak=float(np.max(initial_power))),
                "AMPLITUDE / POWER", scale,
            ),
            _labelled_panel(_phase_rgba(field), "PHASE HUE", scale),
            _labelled_panel(_scalar_rgba(delta), "CHANGE FROM INPUT", scale),
        ]
        frame_path = destination / f"frame_{frame_index:03d}.png"
        _compose(
            panels,
            f"production T4 angular spectrum  z={frame_index * step_m * 1e3:.4f} mm",
        ).save(frame_path)
        frame_paths.append(str(frame_path))
        if frame_index < frames:
            tracer.t4_angular_spectrum_step(
                1, size, size, pitch_m, step_m, 1,
                wavelengths, re, im,
            )

    propagated = re[0].astype(np.float64) + 1j * im[0].astype(np.float64)
    for _ in range(frames):
        tracer.t4_angular_spectrum_step(
            1, size, size, pitch_m, step_m, -1,
            wavelengths, re, im,
        )
    recovered = re[0].astype(np.float64) + 1j * im[0].astype(np.float64)
    error = np.abs(recovered - initial)
    summary_path = destination / "roundtrip_summary.png"
    _compose(
        [
            _labelled_panel(_phase_rgba(initial), "INPUT", scale),
            _labelled_panel(_phase_rgba(propagated), "PROPAGATED", scale),
            _labelled_panel(_phase_rgba(recovered), "REVERSED", scale),
            _labelled_panel(_scalar_rgba(error), "ROUNDTRIP ERROR", scale),
        ],
        f"max complex error={float(np.max(error)):.3e}",
    ).save(summary_path)
    return {
        "frames": tuple(frame_paths),
        "summary": str(summary_path),
        "max_roundtrip_error": float(np.max(error)),
        "rms_roundtrip_error": float(np.sqrt(np.mean(error**2))),
    }


_ULTRA_BAKE_PRESETS: dict[str, dict[str, object]] = {
    "iris-spectrum-fixed": {
        "description": "32 perceptual fixed bands through the physical iris",
        "size": 64,
        "frames": 7,
        "scale": 4,
        "quality": "bake",
        "spectral_mode": "fixed",
        "lane_count": 32,
        "polarizations": ("radial",),
        "vector_page": True,
        "spectral_panel": True,
        "phase_mode": "sweep",
    },
    "iris-spectrum-continuous": {
        "description": "resampled 32-lane continuous-frequency cohorts",
        "size": 64,
        "frames": 7,
        "scale": 4,
        "quality": "bake",
        "spectral_mode": "continuous",
        "lane_count": 32,
        "polarizations": ("azimuthal",),
        "vector_page": True,
        "spectral_panel": True,
        "phase_mode": "sweep",
    },
    "iris-polarization": {
        "description": "Jones/Stokes response across canonical source states",
        "size": 64,
        "frames": 7,
        "scale": 4,
        "quality": "bake",
        "spectral_mode": "fixed",
        "lane_count": 16,
        "polarizations": _APERTURE_POLARIZATION_MODES,
        "vector_page": True,
        "spectral_panel": True,
        "phase_mode": "hold",
    },
    "iris-coherent-phase": {
        "description": "piston-free s/p material and propagated phase",
        "size": 64,
        "frames": 7,
        "scale": 4,
        "quality": "bake",
        "spectral_mode": "fixed",
        "lane_count": 16,
        "polarizations": ("circular+",),
        "vector_page": False,
        "spectral_panel": False,
        "phase_mode": "sweep",
    },
}


def ultra_bake_presets() -> dict[str, dict[str, object]]:
    """Return a detached, JSON-friendly description of showcase recipes."""

    return {
        name: {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in recipe.items()
        }
        for name, recipe in _ULTRA_BAKE_PRESETS.items()
    }


def render_aperture_bake(
    output_dir: str | Path,
    *,
    preset: str,
    size: int | None = None,
    frames: int | None = None,
    scale: int | None = None,
    quality: str | None = None,
    lane_count: int | None = None,
    pitch_m: float = 0.75e-6,
    wavelength_m: float = 532.0e-9,
) -> dict[str, object]:
    """Bake one named aperture showcase through the production wave kernels."""

    if preset not in _ULTRA_BAKE_PRESETS:
        raise ValueError(
            f"unknown ultra bake {preset!r}; "
            f"expected {tuple(_ULTRA_BAKE_PRESETS)}"
        )
    recipe = dict(_ULTRA_BAKE_PRESETS[preset])
    active_size = int(recipe["size"] if size is None else size)
    active_frames = int(recipe["frames"] if frames is None else frames)
    active_scale = int(recipe["scale"] if scale is None else scale)
    active_quality = str(recipe["quality"] if quality is None else quality)
    active_lanes = int(
        recipe["lane_count"] if lane_count is None else lane_count
    )
    if active_size < 8 or active_size & (active_size-1):
        raise ValueError("ultra bake size must be a power of two and at least 8")
    if active_frames < 1:
        raise ValueError("ultra bake frames must be positive")
    if active_scale < 1:
        raise ValueError("ultra bake scale must be positive")
    if active_quality not in _APERTURE_QUALITY_MODES:
        raise ValueError(
            f"ultra bake quality must be one of {_APERTURE_QUALITY_MODES}"
        )
    if active_lanes not in _APERTURE_LANE_COUNTS:
        raise ValueError(
            f"ultra bake lane count must be one of {_APERTURE_LANE_COUNTS}"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tracer = _calibration_tracer(wavelength_m)
    polarizations = tuple(recipe["polarizations"])
    frame_paths: list[str] = []
    frame_records: list[dict[str, object]] = []
    if recipe["phase_mode"] == "hold":
        cycle_phases = np.full(active_frames, 0.62*np.pi, np.float64)
    elif active_frames == 1:
        cycle_phases = np.asarray([0.55*np.pi], np.float64)
    else:
        cycle_phases = np.linspace(0.0, np.pi, active_frames)

    hero_index = active_frames//2
    polarization_offset = (
        len(polarizations)//2
        if active_frames == 1 and len(polarizations) > 1 else 0
    )
    for frame_index, cycle_phase in enumerate(cycle_phases):
        polarization = polarizations[
            (frame_index+polarization_offset) % len(polarizations)
        ]
        print(
            f"[ultra-bake] start {preset} {frame_index+1}/{active_frames} "
            f"visible={active_size} quality={active_quality} "
            f"lanes={active_lanes}",
            flush=True,
        )
        solve_started = time.perf_counter()
        visual = _solve_aperture_visual(
            tracer,
            size=active_size,
            pitch_m=pitch_m,
            wavelength_m=wavelength_m,
            cycle_phase=float(cycle_phase),
            polarization_mode=polarization,
            quality=active_quality,
            spectral_mode=str(recipe["spectral_mode"]),
            lane_count=active_lanes,
            spectral_epoch=frame_index,
            vector_page=bool(recipe["vector_page"]),
            piston_removed=True,
            coherence_seed=(frame_index+1) << 16,
            spectral_panel=bool(recipe["spectral_panel"]),
        )
        solve_seconds = time.perf_counter() - solve_started
        solve_height, solve_width = visual["solve_shape"]
        footer = (
            f"{polarization} | {recipe['spectral_mode']}:{active_lanes} | "
            f"{solve_width}x{solve_height} solve | "
            f"D={2.0*visual['opening_radius_m']*1e6:.2f}um | "
            f"edge={visual['border_fraction']:.1e}"
        )
        plate = _compose_grid(
            visual["panels"],
            visual["labels"],
            scale=active_scale,
            footer=footer,
        )
        frame_path = destination / f"frame_{frame_index:03d}.png"
        plate.save(frame_path, compress_level=6)
        if frame_index == hero_index:
            plate.save(destination / "hero.png", compress_level=6)
        frame_paths.append(str(frame_path))
        frame_records.append({
            "index": frame_index,
            "path": str(frame_path),
            "polarization": polarization,
            "cycle_phase_rad": float(cycle_phase),
            "opening_radius_m": visual["opening_radius_m"],
            "sweep": visual["sweep"],
            "wavelengths_nm": [
                float(value*1.0e9) for value in visual["wavelengths_m"]
            ],
            "solve_shape": [solve_height, solve_width],
            "propagation_steps": visual["propagation_steps"],
            "solve_seconds": solve_seconds,
            "retention": visual["retention"],
            "border_fraction": visual["border_fraction"],
        })
        print(
            f"[ultra-bake] done {preset} {frame_index+1}/{active_frames} "
            f"solve={solve_width}x{solve_height} lanes={active_lanes} "
            f"seconds={solve_seconds:.2f}",
            flush=True,
        )

    manifest = {
        "schema": "spectral-aperture-ultra-bake-v1",
        "preset": preset,
        "description": recipe["description"],
        "size": active_size,
        "scale": active_scale,
        "quality": active_quality,
        "spectral_mode": recipe["spectral_mode"],
        "lane_count": active_lanes,
        "aperture": {
            "blade_count": 9,
            "thickness_m": 0.10e-6,
            "material_name": "blackened_steel",
            "material_n_real": 2.9,
            "material_n_imag": 3.0,
        },
        "frames": frame_records,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "preset": preset,
        "frames": tuple(frame_paths),
        "hero": str(destination / "hero.png"),
        "manifest": str(manifest_path),
    }


def render_ultra_bakes(
    output_dir: str | Path,
    *,
    preset: str,
    **overrides,
) -> dict[str, object]:
    """Render one named showcase or the complete showcase collection."""

    destination = Path(output_dir)
    names = (
        tuple(_ULTRA_BAKE_PRESETS)
        if preset == "all" else (preset,)
    )
    results = {
        name: render_aperture_bake(
            destination / name,
            preset=name,
            **overrides,
        )
        for name in names
    }
    index_path = destination / "ultra_bakes.json"
    destination.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps({
            "schema": "spectral-ultra-bake-collection-v1",
            "presets": {
                name: {
                    "hero": result["hero"],
                    "manifest": result["manifest"],
                    "frame_count": len(result["frames"]),
                }
                for name, result in results.items()
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"presets": results, "index": str(index_path)}


def run_live(
    *,
    size: int = 128,
    cycle_steps: int = 120,
    pitch_m: float = 1.5e-6,
    step_m: float = 25.0e-6,
    wavelength_m: float = 532.0e-9,
    fps: int = 30,
    _max_display_frames: int | None = None,
) -> None:
    """Animate production T4 transforms in OpenGL without writing captures."""

    if size < 8 or size & (size - 1):
        raise ValueError("size must be a power of two and at least 8")
    if cycle_steps < 1:
        raise ValueError("cycle_steps must be positive")
    if fps < 1:
        raise ValueError("fps must be positive")

    import pygame
    from OpenGL import GL as gl

    from camera_software.gpu_preview import (
        GLPreviewCompositor,
        PreviewProductKind,
        PreviewTextureProduct,
    )

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.set_mode(
        (1200, 460), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    )
    pygame.display.set_caption(
        "Production T4 transform — POWER | PHASE | CHANGE   "
        "[Space pause, R reset, Esc close]"
    )

    tracer = _calibration_tracer(wavelength_m)
    wavelengths = np.asarray([wavelength_m], np.float64)
    initial = _source_field(size, pitch_m)
    initial_peak = float(np.max(np.abs(initial) ** 2))
    re = np.ascontiguousarray(initial.real[None], np.float32)
    im = np.ascontiguousarray(initial.imag[None], np.float32)
    textures = [int(value) for value in gl.glGenTextures(3)]
    compositor = GLPreviewCompositor()
    compositor.init_gl()
    for texture in textures:
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
        )
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, size, size, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
        )
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    clock = pygame.time.Clock()
    running = True
    paused = False
    direction = 1
    step_index = 0
    generation = 0
    displayed_frames = 0
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        re[0] = initial.real
                        im[0] = initial.imag
                        direction = 1
                        step_index = 0

            if not paused:
                tracer.t4_angular_spectrum_step(
                    1, size, size, pitch_m, step_m, direction,
                    wavelengths, re, im,
                )
                step_index += direction
                if step_index >= cycle_steps:
                    direction = -1
                elif step_index <= 0:
                    direction = 1
                generation += 1

            field = re[0].astype(np.float64) + 1j * im[0].astype(np.float64)
            panels = (
                _scalar_rgba(np.abs(field) ** 2, relative_peak=initial_peak),
                _phase_rgba(field),
                _scalar_rgba(np.abs(field - initial)),
            )
            for texture, rgba in zip(textures, panels):
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
                gl.glTexSubImage2D(
                    gl.GL_TEXTURE_2D, 0, 0, 0, size, size,
                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                    np.ascontiguousarray(rgba),
                )
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            width, height = pygame.display.get_window_size()
            gl.glViewport(0, 0, width, height)
            gl.glClearColor(0.018, 0.025, 0.04, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gap = max(4, width // 240)
            pane_width = max(1, (width - gap * 4) // 3)
            pane_height = max(1, height - gap * 2)
            for index, texture in enumerate(textures):
                product = PreviewTextureProduct(
                    product_id=f"t4.live.{index}",
                    tab_label=("POWER", "PHASE", "CHANGE")[index],
                    texture_id=texture,
                    width=size,
                    height=size,
                    generation=generation,
                    producer="production-t4-visual",
                    internal_format=int(gl.GL_RGBA8),
                    kind=PreviewProductKind.COMPLEX_FIELD,
                    orientation="top-left",
                    alpha_mode="straight",
                )
                compositor.draw(
                    product,
                    (
                        gap + index * (pane_width + gap),
                        gap,
                        pane_width,
                        pane_height,
                    ),
                    height,
                    tone_map=False,
                )
            pygame.display.flip()
            displayed_frames += 1
            pygame.display.set_caption(
                "Production T4 transform — POWER | PHASE | CHANGE   "
                f"z={step_index * step_m * 1e3:.4f} mm "
                f"{'PAUSED' if paused else ('forward' if direction > 0 else 'reverse')}   "
                "[Space pause, R reset, Esc close]"
            )
            clock.tick(fps)
            if (
                _max_display_frames is not None
                and displayed_frames >= _max_display_frames
            ):
                running = False
    finally:
        compositor.destroy()
        gl.glDeleteTextures(len(textures), textures)
        pygame.quit()


def run_aperture_live(
    *,
    size: int = 64,
    pitch_m: float = 0.75e-6,
    wavelength_m: float = 532.0e-9,
    fps: int = 30,
    polarization_mode: str = "radial",
    quality: str = "balanced",
    spectral_mode: str = "fixed",
    lane_count: int = 1,
    _max_display_frames: int | None = None,
) -> None:
    """Animate Jones-resolved interaction with physical material iris blades."""

    if size < 8 or size & (size-1):
        raise ValueError("size must be a power of two and at least 8")
    if fps < 1:
        raise ValueError("fps must be positive")
    polarization_key = str(polarization_mode).strip().lower()
    if polarization_key not in _APERTURE_POLARIZATION_MODES:
        raise ValueError(
            f"polarization_mode must be one of {_APERTURE_POLARIZATION_MODES}"
        )
    quality_key = str(quality).strip().lower()
    if quality_key not in _APERTURE_QUALITY_MODES:
        raise ValueError(f"quality must be one of {_APERTURE_QUALITY_MODES}")
    spectral_key = str(spectral_mode).strip().lower()
    if spectral_key not in _APERTURE_SPECTRAL_MODES:
        raise ValueError(
            f"spectral_mode must be one of {_APERTURE_SPECTRAL_MODES}"
        )
    if int(lane_count) not in _APERTURE_LANE_COUNTS:
        raise ValueError(
            f"lane_count must be one of {_APERTURE_LANE_COUNTS}"
        )
    import pygame
    from OpenGL import GL as gl
    from camera_software.gpu_preview import (
        GLPreviewCompositor, PreviewProductKind, PreviewTextureProduct,
    )
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.set_mode(
        (1500, 820), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    )
    tracer = _calibration_tracer(wavelength_m)
    textures = [int(value) for value in gl.glGenTextures(6)]
    compositor = GLPreviewCompositor()
    compositor.init_gl()
    for texture in textures:
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
        )
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, size, size, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
        )
    clock = pygame.time.Clock()
    running = True
    paused = False
    piston_removed = True
    vector_page = True
    source_angle_deg = 0.0
    analyzer_angle_deg = 0.0
    polarization_index = _APERTURE_POLARIZATION_MODES.index(polarization_key)
    quality_index = _APERTURE_QUALITY_MODES.index(quality_key)
    spectral_index = _APERTURE_SPECTRAL_MODES.index(spectral_key)
    lane_index = _APERTURE_LANE_COUNTS.index(int(lane_count))
    spectral_epoch = 0
    generation = 0
    displayed_frames = 0
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_p:
                        piston_removed = not piston_removed
                    elif event.key == pygame.K_v:
                        vector_page = not vector_page
                    elif event.key == pygame.K_j:
                        polarization_index = (
                            polarization_index + 1
                        ) % len(_APERTURE_POLARIZATION_MODES)
                    elif event.key == pygame.K_k:
                        quality_index = (
                            quality_index + 1
                        ) % len(_APERTURE_QUALITY_MODES)
                    elif event.key == pygame.K_f:
                        spectral_index = (
                            spectral_index + 1
                        ) % len(_APERTURE_SPECTRAL_MODES)
                    elif event.key == pygame.K_l:
                        lane_index = (
                            lane_index + 1
                        ) % len(_APERTURE_LANE_COUNTS)
                    elif event.key == pygame.K_r:
                        spectral_epoch += 1
                    elif event.key == pygame.K_LEFT:
                        analyzer_angle_deg -= 5.0
                    elif event.key == pygame.K_RIGHT:
                        analyzer_angle_deg += 5.0
                    elif event.key == pygame.K_LEFTBRACKET:
                        source_angle_deg -= 5.0
                    elif event.key == pygame.K_RIGHTBRACKET:
                        source_angle_deg += 5.0
            if not paused:
                generation += 1
            phase = generation*0.018
            active_quality = _APERTURE_QUALITY_MODES[quality_index]
            active_spectral = _APERTURE_SPECTRAL_MODES[spectral_index]
            active_lanes = _APERTURE_LANE_COUNTS[lane_index]
            active_polarization = _APERTURE_POLARIZATION_MODES[
                polarization_index
            ]
            visual = _solve_aperture_visual(
                tracer,
                size=size,
                pitch_m=pitch_m,
                wavelength_m=wavelength_m,
                cycle_phase=phase,
                polarization_mode=active_polarization,
                quality=active_quality,
                spectral_mode=active_spectral,
                lane_count=active_lanes,
                spectral_epoch=spectral_epoch,
                source_angle_deg=source_angle_deg,
                analyzer_angle_deg=analyzer_angle_deg,
                vector_page=vector_page,
                piston_removed=piston_removed,
                coherence_seed=generation << 8,
            )
            panels = visual["panels"]
            labels = visual["labels"]
            solve_height, solve_width = visual["solve_shape"]
            for texture, rgba in zip(textures, panels):
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
                gl.glTexSubImage2D(
                    gl.GL_TEXTURE_2D, 0, 0, 0, size, size,
                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                    np.ascontiguousarray(rgba),
                )
            width, height = pygame.display.get_window_size()
            gl.glViewport(0, 0, width, height)
            gl.glClearColor(0.018, 0.025, 0.04, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gap = max(4, width//288)
            pane_width = max(1, (width-gap*4)//3)
            pane_height = max(1, (height-gap*3)//2)
            for index, texture in enumerate(textures):
                row, column = divmod(index, 3)
                compositor.draw(
                    PreviewTextureProduct(
                        product_id=f"t4.aperture.{index}",
                        tab_label=labels[index],
                        texture_id=texture,
                        width=size,
                        height=size,
                        generation=generation,
                        producer="production-t4-physical-aperture",
                        internal_format=int(gl.GL_RGBA8),
                        kind=PreviewProductKind.COMPLEX_FIELD,
                        orientation="top-left",
                        alpha_mode="straight",
                    ),
                    (
                        gap+column*(pane_width+gap),
                        gap+row*(pane_height+gap),
                        pane_width, pane_height,
                    ),
                    height,
                    tone_map=False,
                )
            pygame.display.flip()
            displayed_frames += 1
            pygame.display.set_caption(
                "Vector physical aperture — "
                f"{'STOKES' if vector_page else 'COHERENT PHASE'}  "
                f"source={active_polarization} "
                f"angle={source_angle_deg%180.0:.0f}deg "
                f"modes={visual['mode_count']} "
                f"quality={active_quality} "
                f"spectrum={active_spectral}:{active_lanes} "
                f"solve={solve_width}x{solve_height}/{size}x{size} "
                f"steps={visual['propagation_steps']} "
                f"diameter={2.0*visual['opening_radius_m']*1e6:.2f} um "
                f"range={visual['sweep']*100.0:.1f}% "
                f"retention={visual['retention']:.5f} "
                f"edge={visual['border_fraction']:.2e} "
                f"phase={'RELATIVE' if piston_removed else 'ABSOLUTE'} "
                f"{'PAUSED ' if paused else ''}"
                "[J source, K quality, F fixed/continuous, L lanes, "
                "R resample, [/] source angle, arrows analyzer, V page, "
                "P phase gauge, Space pause, Esc close]"
            )
            clock.tick(fps)
            if (
                _max_display_frames is not None
                and displayed_frames >= _max_display_frames
            ):
                running = False
    finally:
        compositor.destroy()
        gl.glDeleteTextures(len(textures), textures)
        pygame.quit()


def run_transport_live(
    *,
    wavelength_m: float = 532.0e-9,
    fps: int = 30,
    panel_size: int = 384,
    _max_display_frames: int | None = None,
) -> None:
    """Animate an actual T1→T4→T1 detector bench without writing files."""

    if fps < 1 or panel_size < 64:
        raise ValueError("fps must be positive and panel_size at least 64")
    import pygame
    from OpenGL import GL as gl
    from camera_software.gpu_preview import (
        GLPreviewCompositor,
        PreviewProductKind,
        PreviewTextureProduct,
    )

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.set_mode(
        (1600, 500), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    )
    tracer = _transport_calibration_tracer(wavelength_m)
    textures = [int(value) for value in gl.glGenTextures(4)]
    compositor = GLPreviewCompositor()
    compositor.init_gl()
    for texture in textures:
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE
        )
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
            panel_size, panel_size, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
        )
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    panels = [
        np.zeros((panel_size, panel_size, 4), np.uint8)
        for _ in range(4)
    ]
    for panel in panels:
        panel[..., 3] = 255
    clock = pygame.time.Clock()
    running = True
    paused = False
    submitted = False
    generation = 0
    displayed_frames = 0
    last_boundary: dict[str, object] | None = None
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

            if submitted and int(tracer.in_flight_count()) == 0:
                records = tracer.drain_records(64)
                arenas = [
                    dict(value) for value in tracer.wave_arena_stats()
                ]
                first = next(
                    value for value in arenas
                    if int(value["next_forward"]) >= 0
                )
                terminal = next(
                    value for value in arenas
                    if int(value["next_forward"]) < 0
                )
                boundary = dict(terminal["boundary"])
                detector_position = None
                kinds = np.asarray(records["kind"])
                strike_indices = np.flatnonzero(kinds == 0)
                if strike_indices.size:
                    detector_position = np.asarray(
                        records["pos"], np.float64
                    )[int(strike_indices[-1])]
                phases = []
                for value in (first, terminal):
                    value_boundary = dict(value["boundary"])
                    field_snapshot = tracer.wave_arena_field_snapshot(
                        int(value["arena_id"]),
                        int(value_boundary["direction"]), 0, 0,
                    )
                    field = (
                        np.asarray(field_snapshot["re"], np.float64)
                        + 1j*np.asarray(field_snapshot["im"], np.float64)
                    )
                    phases.append(np.asarray(
                        Image.fromarray(_phase_rgba(field), "RGBA").resize(
                            (panel_size, panel_size),
                            Image.Resampling.BICUBIC,
                        ),
                        np.uint8,
                    ))
                panels = [
                    _transport_table_rgba(
                        arenas, detector_position, panel_size
                    ),
                    phases[0],
                    phases[1],
                    _boundary_power_rgba(first, panel_size, terminal),
                ]
                last_boundary = boundary
                generation += 1
                submitted = False

            if not paused and not submitted and int(tracer.in_flight_count()) == 0:
                angle = 0.012 * np.sin(generation * 0.19)
                offset = 28.0e-6 * np.sin(generation * 0.11)
                direction = np.asarray(
                    [angle, 0.0, np.sqrt(max(0.0, 1.0-angle*angle))],
                    np.float64,
                )
                tracer.submit_rays(
                    np.asarray([[offset, 0.0, -800.0e-6]], np.float64),
                    direction[None],
                    np.asarray([[1.0 + 0.0j]], np.complex128),
                    max_bounces=3,
                    min_amplitude=1.0e-12,
                )
                submitted = True

            for texture, rgba in zip(textures, panels):
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
                gl.glTexSubImage2D(
                    gl.GL_TEXTURE_2D, 0, 0, 0,
                    panel_size, panel_size,
                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                    np.ascontiguousarray(rgba),
                )
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            width, height = pygame.display.get_window_size()
            gl.glViewport(0, 0, width, height)
            gl.glClearColor(0.018, 0.025, 0.04, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gap = max(4, width//250)
            pane_width = max(1, (width-gap*5)//4)
            pane_height = max(1, height-gap*2)
            for index, texture in enumerate(textures):
                compositor.draw(
                    PreviewTextureProduct(
                        product_id=f"t4.transport.{index}",
                        tab_label=(
                            "TABLE", "T4-0 EXIT", "T4-1 EXIT", "POWER"
                        )[index],
                        texture_id=texture,
                        width=panel_size,
                        height=panel_size,
                        generation=generation,
                        producer="native-t1-t4-boundary",
                        internal_format=int(gl.GL_RGBA8),
                        kind=PreviewProductKind.COMPLEX_FIELD,
                        orientation="top-left",
                        alpha_mode="opaque",
                    ),
                    (
                        gap + index*(pane_width+gap), gap,
                        pane_width, pane_height,
                    ),
                    height,
                    tone_map=False,
                )
            pygame.display.flip()
            displayed_frames += 1
            retention = ""
            if last_boundary is not None:
                seeded = float(last_boundary["seeded_field_power"])
                propagated = float(last_boundary["propagated_field_power"])
                retention = f" retention={propagated/max(seeded, 1e-30):.8f}"
            pygame.display.set_caption(
                "Native linked transport — TABLE | T4-0 | T4-1 | POWER "
                f"generation={generation}{retention} "
                f"{'PAUSED' if paused else ''}  [Space pause, Esc close]"
            )
            clock.tick(fps)
            if (
                _max_display_frames is not None
                and displayed_frames >= _max_display_frames
            ):
                running = False
    finally:
        tracer.stop_pipeline()
        compositor.destroy()
        gl.glDeleteTextures(len(textures), textures)
        pygame.quit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="exposures/wave_transform_visual"
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument(
        "--aperture-size", type=int, default=64,
        help="power-of-two field width for --aperture-live",
    )
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument(
        "--live", action="store_true",
        help="animate in OpenGL until closed; creates no output files",
    )
    parser.add_argument(
        "--transport-live", action="store_true",
        help="animate the native T1/T4 boundary and detector in OpenGL",
    )
    parser.add_argument(
        "--aperture-live", action="store_true",
        help="animate Jones-resolved finite material iris interaction in OpenGL",
    )
    parser.add_argument(
        "--ultra-bake",
        choices=("all", *_ULTRA_BAKE_PRESETS),
        help="bake a named high-investment physical-aperture showcase",
    )
    parser.add_argument(
        "--list-ultra-bakes", action="store_true",
        help="print named showcase recipes as JSON and exit",
    )
    parser.add_argument(
        "--ultra-size", type=int,
        help="override preset visible field width (power of two)",
    )
    parser.add_argument(
        "--ultra-frames", type=int,
        help="override preset frame count",
    )
    parser.add_argument(
        "--ultra-scale", type=int,
        help="override preset PNG presentation scale",
    )
    parser.add_argument(
        "--ultra-quality", choices=_APERTURE_QUALITY_MODES,
        help="override preset hidden-domain investment",
    )
    parser.add_argument(
        "--ultra-lanes", type=int, choices=_APERTURE_LANE_COUNTS,
        help="override preset exact spectral lane width",
    )
    parser.add_argument(
        "--aperture-polarization",
        choices=_APERTURE_POLARIZATION_MODES,
        default="radial",
        help="initial Jones/coherence source for --aperture-live",
    )
    parser.add_argument(
        "--aperture-quality",
        choices=_APERTURE_QUALITY_MODES,
        default="balanced",
        help="hidden padded solve investment for --aperture-live",
    )
    parser.add_argument(
        "--aperture-spectrum",
        choices=_APERTURE_SPECTRAL_MODES,
        default="fixed",
        help="fixed perceptual bands or continuous-frequency cohort",
    )
    parser.add_argument(
        "--aperture-lanes",
        type=int,
        choices=_APERTURE_LANE_COUNTS,
        default=1,
        help="exact compiled lane width for --aperture-live",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--cycle-steps", type=int, default=120,
        help="forward steps before the live animation reverses",
    )
    args = parser.parse_args()
    if args.list_ultra_bakes:
        print(json.dumps(ultra_bake_presets(), indent=2))
        return 0
    if sum((
        bool(args.live),
        bool(args.transport_live),
        bool(args.aperture_live),
        bool(args.ultra_bake),
    )) > 1:
        parser.error("choose one live or bake mode")
    if args.ultra_bake:
        result = render_ultra_bakes(
            args.output_dir,
            preset=args.ultra_bake,
            size=args.ultra_size,
            frames=args.ultra_frames,
            scale=args.ultra_scale,
            quality=args.ultra_quality,
            lane_count=args.ultra_lanes,
        )
        print(f"[ultra-bake] index={result['index']}")
        return 0
    if args.aperture_live:
        run_aperture_live(
            size=args.aperture_size,
            fps=args.fps,
            polarization_mode=args.aperture_polarization,
            quality=args.aperture_quality,
            spectral_mode=args.aperture_spectrum,
            lane_count=args.aperture_lanes,
        )
        return 0
    if args.transport_live:
        run_transport_live(fps=args.fps)
        return 0
    if args.live:
        run_live(
            size=args.size,
            cycle_steps=args.cycle_steps,
            fps=args.fps,
        )
        return 0
    result = render_sequence(
        args.output_dir, size=args.size, frames=args.frames, scale=args.scale
    )
    print(f"[wave-transform] frames={len(result['frames'])}")
    print(f"[wave-transform] summary={result['summary']}")
    print(
        "[wave-transform] roundtrip "
        f"max={result['max_roundtrip_error']:.6e} "
        f"rms={result['rms_roundtrip_error']:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
