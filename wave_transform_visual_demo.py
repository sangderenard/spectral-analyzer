"""Visual confirmation for the production T4 angular-spectrum transform.

This is not a second wave solver.  It drives
``RayTracer.t4_angular_spectrum_step`` -- the same native kernel used by
persistent T4 arenas -- and only performs presentation/capture in Python.

The source is a pair of finite coherent Gaussian emitters.  No ideal aperture
or hard array mask is used.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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
_APERTURE_PATTERNS = (
    "iris", "circular", "hole-array", "slot-array", "grating",
)


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


def _publish_square_texture(
    plate: Image.Image,
    output_size: int,
) -> Image.Image:
    """Fit a scientific plate into an exact square RGBA texture.

    Aspect ratio is retained: scientific panels are never stretched to fill
    the square.  The surrounding pixels use the compositor background so the
    result remains directly usable as an OpenGL/UI texture.
    """

    edge = int(output_size)
    if edge < 64:
        raise ValueError("square output texture must be at least 64 pixels")
    ratio = min(edge/plate.width, edge/plate.height)
    fitted_size = (
        max(1, int(round(plate.width*ratio))),
        max(1, int(round(plate.height*ratio))),
    )
    fitted = plate.resize(fitted_size, Image.Resampling.LANCZOS)
    texture = Image.new("RGBA", (edge, edge), (4, 7, 12, 255))
    texture.alpha_composite(
        fitted,
        ((edge-fitted.width)//2, (edge-fitted.height)//2),
    )
    return texture


def _select_scientific_panel(
    panels: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    selector: str,
) -> tuple[int, np.ndarray, str]:
    """Resolve a stable index or a descriptive panel-name fragment."""

    token = str(selector).strip().lower()
    if not token:
        raise ValueError("single-panel selector must be non-empty")
    if token.isdigit():
        index = int(token)
        if 0 <= index < len(panels):
            return index, panels[index], labels[index]
    aliases = {
        "physical": 0,
        "geometry": 0,
        "blades": 0,
        "spectral": 1,
        "spectrum": 1,
        "power": 2,
        "intensity": 2,
        "polarization": 3,
        "stokes-q": 4,
        "q": 4,
        "stokes-v": 5,
        "v": 5,
    }
    if token in aliases and aliases[token] < len(panels):
        index = aliases[token]
        return index, panels[index], labels[index]
    normalized = [
        label.lower().replace(" ", "-").replace("/", "-")
        for label in labels
    ]
    matches = [
        index for index, label in enumerate(normalized)
        if token in label
    ]
    if len(matches) == 1:
        index = matches[0]
        return index, panels[index], labels[index]
    raise ValueError(
        f"unknown or ambiguous panel {selector!r}; use index 0.."
        f"{len(panels)-1} or one of {tuple(labels)}"
    )


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

    image = Image.new("RGBA", (size, size), (7, 11, 18, 255))
    draw = ImageDraw.Draw(image)
    radius = (
        aperture.assembly_radius_m
        if view_radius_m is None else float(view_radius_m)
    )
    from camera_software.physical_aperture import AperturePattern

    if aperture.pattern not in (
        AperturePattern.IRIS_POLYGON,
        AperturePattern.CIRCULAR_HOLE,
    ):
        coordinate = np.linspace(-radius, radius, size, dtype=np.float64)
        x, y = np.meshgrid(coordinate, coordinate)
        inside = x*x+y*y <= aperture.assembly_radius_m**2
        material = inside & ~aperture.open_mask(x, y)
        pixels = np.zeros((size, size, 4), np.uint8)
        pixels[..., :3] = (7, 11, 18)
        pixels[..., 3] = 255
        pixels[material, :3] = (88, 96, 110)
        edge = material ^ (
            np.roll(material, 1, axis=0)
            & np.roll(material, 1, axis=1)
        )
        pixels[edge, :3] = (170, 190, 215)
        return pixels

    vertices, _ = aperture.triangle_mesh()

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


def _build_aperture_pattern(
    pattern: str,
    *,
    opening_radius_m: float,
    sweep: float,
    assembly_radius_m: float,
    pitch_m: float,
    rotation_rad: float,
):
    """Author one finite material aperture using comparable scale controls."""

    from camera_software.physical_aperture import LivePhysicalAperture

    key = str(pattern).strip().lower()
    if key not in _APERTURE_PATTERNS:
        raise ValueError(f"aperture pattern must be one of {_APERTURE_PATTERNS}")
    common = dict(
        assembly_radius_m=float(assembly_radius_m),
        thickness_m=0.10e-6,
        material_name="blackened_steel",
        material_n_real=2.9,
        material_n_imag=3.0,
    )
    if key == "iris":
        return LivePhysicalAperture.iris(
            "demo.live-iris",
            blade_count=9,
            opening_radius_m=float(opening_radius_m),
            rotation_rad=float(rotation_rad),
            **common,
        )
    if key == "circular":
        return LivePhysicalAperture.circular_hole(
            "demo.circular-bore",
            opening_radius_m=float(opening_radius_m),
            **common,
        )
    cell_pitch = 8.0*float(pitch_m)
    duty = 0.10+0.72*float(sweep)
    if key == "hole-array":
        return LivePhysicalAperture.circular_hole_array(
            "demo.circular-hole-array",
            hole_radius_m=0.5*cell_pitch*duty,
            pitch_x_m=cell_pitch,
            pitch_y_m=cell_pitch,
            rotation_rad=float(rotation_rad),
            **common,
        )
    if key == "slot-array":
        return LivePhysicalAperture.slot_array(
            "demo.rectangular-slot-array",
            slot_width_m=cell_pitch*duty,
            slot_height_m=cell_pitch*min(0.92, 1.35*duty),
            pitch_x_m=cell_pitch,
            pitch_y_m=cell_pitch,
            rotation_rad=float(rotation_rad),
            **common,
        )
    return LivePhysicalAperture.grating(
        "demo.transmission-grating",
        slit_width_m=cell_pitch*duty,
        pitch_m=cell_pitch,
        rotation_rad=float(rotation_rad),
        **common,
    )


def _solve_aperture_visual(
    tracer,
    *,
    size: int,
    pitch_m: float,
    wavelength_m: float,
    cycle_phase: float,
    aperture_pattern: str,
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
    aperture_template=None,
) -> dict[str, object]:
    """Solve one physical-aperture frame through shared production kernels."""

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
    if aperture_template is None:
        aperture = _build_aperture_pattern(
            aperture_pattern,
            opening_radius_m=float(opening),
            sweep=float(sweep),
            assembly_radius_m=aperture_extent,
            pitch_m=pitch_m,
            rotation_rad=cycle_phase*0.2,
        )
    else:
        # The arena animates the component it loaded; it must not silently
        # author a second demo-private aperture.  Preserve its blade/material
        # contract while fitting its physical extent to this sampled domain.
        scale = aperture_extent/max(
            float(aperture_template.assembly_radius_m), 1.0e-30
        )
        aperture = replace(
            aperture_template,
            opening_x_m=float(opening),
            opening_y_m=float(opening),
            assembly_radius_m=float(aperture_extent),
            thickness_m=float(aperture_template.thickness_m)*scale,
            rotation_rad=(
                float(aperture_template.rotation_rad)+cycle_phase*0.2
            ),
        )
        aperture.validate()
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
        "aperture_pattern": str(aperture_pattern),
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
        "aperture_pattern": "iris",
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
        "aperture_pattern": "iris",
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
        "aperture_pattern": "iris",
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
        "aperture_pattern": "iris",
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
    aperture_pattern: str | None = None,
    output_size: int | None = None,
    panel: str | None = None,
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
    active_pattern = str(
        recipe["aperture_pattern"]
        if aperture_pattern is None else aperture_pattern
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
    if active_pattern not in _APERTURE_PATTERNS:
        raise ValueError(
            f"ultra bake aperture pattern must be one of {_APERTURE_PATTERNS}"
        )
    if output_size is not None and int(output_size) < 64:
        raise ValueError("ultra bake output size must be at least 64 pixels")
    if panel is not None and output_size is not None:
        expected = active_size*active_scale
        if int(output_size) != expected:
            raise ValueError(
                "single-panel output is never resampled; output_size must "
                f"equal the computed panel size {expected}"
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
            aperture_pattern=active_pattern,
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
            f"{active_pattern} | {polarization} | "
            f"{recipe['spectral_mode']}:{active_lanes} | "
            f"{solve_width}x{solve_height} solve | "
            f"D={2.0*visual['opening_radius_m']*1e6:.2f}um | "
            f"edge={visual['border_fraction']:.1e}"
        )
        selected_panel_index = None
        selected_panel_label = None
        if panel is None:
            plate = _compose_grid(
                visual["panels"],
                visual["labels"],
                scale=active_scale,
                footer=footer,
            )
            native_plate_size = [int(plate.width), int(plate.height)]
            if output_size is not None:
                plate = _publish_square_texture(plate, int(output_size))
        else:
            (
                selected_panel_index,
                panel_rgba,
                selected_panel_label,
            ) = _select_scientific_panel(
                visual["panels"], visual["labels"], panel,
            )
            plate = Image.fromarray(panel_rgba, "RGBA")
            if active_scale != 1:
                plate = plate.resize(
                    (
                        plate.width*active_scale,
                        plate.height*active_scale,
                    ),
                    Image.Resampling.NEAREST,
                )
            native_plate_size = [int(plate.width), int(plate.height)]
        frame_path = destination / f"frame_{frame_index:03d}.png"
        plate.save(frame_path, compress_level=6)
        if frame_index == hero_index:
            plate.save(destination / "hero.png", compress_level=6)
        frame_paths.append(str(frame_path))
        frame_records.append({
            "index": frame_index,
            "path": str(frame_path),
            "polarization": polarization,
            "aperture_pattern": active_pattern,
            "cycle_phase_rad": float(cycle_phase),
            "opening_radius_m": visual["opening_radius_m"],
            "sweep": visual["sweep"],
            "wavelengths_nm": [
                float(value*1.0e9) for value in visual["wavelengths_m"]
            ],
            "solve_shape": [solve_height, solve_width],
            "native_plate_size": native_plate_size,
            "published_texture_size": [int(plate.width), int(plate.height)],
            "selected_panel_index": selected_panel_index,
            "selected_panel_label": selected_panel_label,
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
        "aperture_pattern": active_pattern,
        "output_size": (
            None if output_size is None else int(output_size)
        ),
        "publication_layout": (
            "single-scientific-panel"
            if panel is not None else
            (
                "native-3x2-plate"
                if output_size is None else
                "aspect-preserving-square-texture"
            )
        ),
        "selected_panel": panel,
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
    aperture_pattern: str = "iris",
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
    aperture_key = str(aperture_pattern).strip().lower()
    if aperture_key not in _APERTURE_PATTERNS:
        raise ValueError(
            f"aperture_pattern must be one of {_APERTURE_PATTERNS}"
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
    aperture_index = _APERTURE_PATTERNS.index(aperture_key)
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
                    elif event.key == pygame.K_m:
                        aperture_index = (
                            aperture_index + 1
                        ) % len(_APERTURE_PATTERNS)
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
            active_aperture = _APERTURE_PATTERNS[aperture_index]
            active_polarization = _APERTURE_POLARIZATION_MODES[
                polarization_index
            ]
            visual = _solve_aperture_visual(
                tracer,
                size=size,
                pitch_m=pitch_m,
                wavelength_m=wavelength_m,
                cycle_phase=phase,
                aperture_pattern=active_aperture,
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
                f"aperture={active_aperture} "
                f"spectrum={active_spectral}:{active_lanes} "
                f"solve={solve_width}x{solve_height}/{size}x{size} "
                f"steps={visual['propagation_steps']} "
                f"diameter={2.0*visual['opening_radius_m']*1e6:.2f} um "
                f"range={visual['sweep']*100.0:.1f}% "
                f"retention={visual['retention']:.5f} "
                f"edge={visual['border_fraction']:.2e} "
                f"phase={'RELATIVE' if piston_removed else 'ABSOLUTE'} "
                f"{'PAUSED ' if paused else ''}"
                "[J source, K quality, M aperture, F fixed/continuous, L lanes, "
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


def _arena_fit_points(
    points: np.ndarray,
    size: int,
    *,
    margin: int = 24,
    axes: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    values = np.asarray(points, np.float64).reshape(-1, 3)
    if axes is None:
        spans = np.ptp(values, axis=0)
        selected = tuple(int(value) for value in np.argsort(spans)[-2:])
        axes = (selected[0], selected[1])
    projected = values[:, axes]
    lo = np.min(projected, axis=0)
    hi = np.max(projected, axis=0)
    span = np.maximum(hi-lo, 1.0e-12)
    scale = min(
        (size-2*margin)/float(span[0]),
        (size-2*margin)/float(span[1]),
    )
    center = 0.5*(lo+hi)
    screen = (projected-center)*scale
    screen[:, 0] += 0.5*size
    screen[:, 1] = 0.5*size-screen[:, 1]
    return screen, axes


def _arena_unit(value, name: str) -> np.ndarray:
    vector = np.asarray(value, np.float64).reshape(3)
    length = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or length <= 1.0e-12:
        raise ValueError(f"{name} must be a finite non-zero vector")
    return vector/length


def _arena_font(size: int, *, title: bool = False):
    """Compact instrumentation type, about half the previous bitmap scale."""

    body_size = max(7, min(10, int(round(size/64))))
    pixel_size = min(12, body_size+2) if title else body_size
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/consola.ttf", pixel_size
        )
    except OSError:
        return ImageFont.load_default()


def _arena_geometry_panel(compiled, size: int) -> np.ndarray:
    image = Image.new("RGBA", (size, size), (6, 10, 18, 255))
    draw = ImageDraw.Draw(image)
    font = _arena_font(size)
    geometry = compiled.display_geometry
    if geometry is None:
        draw.text((12, 12), "PARAMETRIC GEOMETRY", font=font,
                  fill=(210, 225, 245, 255))
        draw.text((12, 27), "mesh-independent exact component", font=font,
                  fill=(115, 180, 240, 255))
        for index, face in enumerate(
            compiled.metadata.get("registered_faces", ())
        ):
            draw.text((12, 40+index*9), str(face), font=font,
                      fill=(155, 175, 205, 255))
        return np.asarray(image, np.uint8)

    triangles = np.asarray(geometry.triangles, np.float64)
    screen, axes = _arena_fit_points(triangles.reshape(-1, 3), size)
    screen = screen.reshape(-1, 3, 2)
    palette = {
        "reflective_surface": (95, 190, 255, 220),
        "entrance_glass": (90, 220, 235, 125),
        "exit_glass": (90, 220, 235, 125),
        "silvered_reflector_1": (225, 235, 250, 230),
        "silvered_reflector_2": (225, 235, 250, 230),
        "blackened_prism_face": (28, 35, 48, 245),
        "blackened_prism_side": (28, 35, 48, 245),
        "aperture_material": (75, 90, 115, 240),
    }
    for triangle, role in zip(screen, geometry.roles):
        color = palette.get(role, (120, 145, 185, 190))
        xy = [tuple(float(value) for value in point) for point in triangle]
        draw.polygon(xy, fill=color, outline=(175, 205, 235, 210))
    draw.rectangle((5, 5, size-6, 18), fill=(5, 9, 16, 220))
    draw.text(
        (10, 7),
        f"REPRESENTATIVE GEOMETRY  axes={axes[0]}/{axes[1]}",
        font=font, fill=(220, 232, 248, 255),
    )
    return np.asarray(image, np.uint8)


def _arena_text_panel(
    title: str,
    lines: list[str],
    size: int,
    *,
    accent: tuple[int, int, int, int] = (95, 195, 255, 255),
) -> np.ndarray:
    image = Image.new("RGBA", (size, size), (7, 11, 19, 255))
    draw = ImageDraw.Draw(image)
    font = _arena_font(size)
    title_font = _arena_font(size, title=True)
    draw.text((10, 8), title, font=title_font, fill=accent)
    y = 23
    line_step = max(7, size//48)
    for line in lines:
        words = str(line).split()
        rows: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > max(32, size//4) and current:
                rows.append(current)
                current = word
            else:
                current = candidate
        if current:
            rows.append(current)
        for row in rows:
            if y > size-line_step-5:
                draw.text((10, y), "...", font=font, fill=(130, 145, 170, 255))
                return np.asarray(image, np.uint8)
            draw.text((10, y), row, font=font, fill=(175, 195, 220, 255))
            y += line_step
        y += 2
    return np.asarray(image, np.uint8)


def _arena_graph_panel(compiled, size: int) -> np.ndarray:
    contract = compiled.graph.contract()
    nodes = contract["nodes"]
    links = contract["links"]
    image = Image.new("RGBA", (size, size), (7, 11, 19, 255))
    draw = ImageDraw.Draw(image)
    font = _arena_font(size)
    title_font = _arena_font(size, title=True)
    draw.text((10, 7), "COMPILED TRANSPORT GRAPH", font=title_font,
              fill=(160, 120, 255, 255))
    if not nodes:
        return np.asarray(image, np.uint8)
    box_h = max(11, min(36, (size-31)//len(nodes)))
    centers: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(nodes):
        y0 = 22+index*box_h
        y1 = min(size-8, y0+max(8, box_h-4))
        if y0 >= size-8:
            break
        x0, x1 = 16, size-16
        domain = node["domain"]
        color = {
            "pipeline-port": (65, 115, 165, 255),
            "t2-parametric": (65, 170, 120, 255),
            "t3-material": (205, 145, 55, 255),
            "t4-wave-arena": (135, 90, 220, 255),
        }.get(domain, (100, 110, 135, 255))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=5,
                               fill=tuple((*color[:3], 75)), outline=color)
        label = f"{node['operation']}  [{domain}]"
        draw.text((x0+6, y0+3), label[:max(32, size//4)],
                  font=font, fill=(225, 235, 248, 255))
        centers[node["key"]] = (0.5*(x0+x1), 0.5*(y0+y1))
    for link in links:
        if link["src"] not in centers or link["dst"] not in centers:
            continue
        a, b = centers[link["src"]], centers[link["dst"]]
        draw.line((a[0], a[1]+8, b[0], b[1]-8),
                  fill=(210, 220, 238, 200), width=2)
    return np.asarray(image, np.uint8)


def _arena_probe_paths(component, compiled, phase: float) -> list[np.ndarray]:
    from camera_software.optical_components import (
        CompoundLensComponent, PentaprismComponent, PlaneMirrorComponent,
    )
    paths: list[np.ndarray] = []
    if isinstance(component, CompoundLensComponent):
        front = component.lens.side("front")
        back = component.lens.side("back")
        axial_span = max(abs(back.x_pos-front.x_pos), front.radius, 1.0e-3)
        launch_x = front.x_pos-0.35*axial_span
        angle = 0.035*math.sin(phase)
        for offset in np.linspace(-0.72*front.radius, 0.72*front.radius, 9):
            result = component.lens.trace_detailed(
                (launch_x, float(offset), 0.0),
                (1.0, angle, 0.0),
            )
            points = [np.asarray((launch_x, offset, 0.0), np.float64)]
            points.extend(event.p1 for event in result.events)
            if result.events:
                tail = np.asarray(result.events[-1].p1, np.float64)
                direction = np.asarray(result.events[-1].dir_out, np.float64)
                points.append(tail+direction*0.35*axial_span)
            paths.append(np.asarray(points))
    elif isinstance(component, PlaneMirrorComponent):
        normal = _arena_unit(component.normal, "mirror normal")
        incoming = _arena_unit(
            (1.0, 0.22*math.sin(phase), 0.0), "probe"
        )
        reflected = component.reflected_direction(incoming)
        center = np.asarray(component.center_m, np.float64)
        tangent = _arena_unit(
            np.cross(normal, (0.0, 0.0, 1.0)), "mirror tangent"
        )
        for offset in np.linspace(-0.7, 0.7, 9):
            hit = center+offset*component.radius_m*tangent
            paths.append(np.stack((
                hit-incoming*2.2*component.radius_m,
                hit,
                hit+reflected*2.2*component.radius_m,
            )))
    elif isinstance(component, PentaprismComponent):
        assembly = component.spec.build()
        chief = np.asarray(assembly.primary_path, np.float64)
        extrusion = _arena_unit(
            np.cross(component.spec.input_axis, component.spec.output_axis),
            "pentaprism extrusion",
        )
        for offset in np.linspace(-0.35, 0.35, 7):
            paths.append(chief+offset*component.spec.depth_m*extrusion)
    return paths


def _arena_probe_panel(
    component,
    compiled,
    size: int,
    phase: float,
    native_paths: list[object] | None = None,
) -> np.ndarray:
    paths = (
        native_paths
        if native_paths is not None
        else _arena_probe_paths(component, compiled, phase)
    )
    image = Image.new("RGBA", (size, size), (4, 8, 14, 255))
    draw = ImageDraw.Draw(image)
    font = _arena_font(size)
    title_font = _arena_font(size, title=True)
    if not paths:
        draw.text((10, 8), "FIELD-DOMAIN PROBE", font=title_font,
                  fill=(110, 220, 255, 255))
        draw.text((10, 22), "see complex field products", font=font,
                  fill=(165, 190, 220, 255))
        return np.asarray(image, np.uint8)
    path_points = [
        np.asarray(path["points"] if isinstance(path, dict) else path)
        for path in paths
    ]
    geometry = compiled.display_geometry
    geometry_points = (
        np.asarray(geometry.triangles, np.float64).reshape(-1, 3)
        if geometry is not None else np.empty((0, 3), np.float64)
    )
    all_points = np.concatenate((*([geometry_points] if len(geometry_points) else []),
                                 *path_points))
    screen, axes = _arena_fit_points(all_points, size, margin=30)
    cursor = len(geometry_points)
    if geometry is not None:
        geometry_screen = screen[:cursor].reshape(-1, 3, 2)
        role_colors = {
            "reflective_surface": (50, 115, 160, 180),
            "entrance_glass": (30, 125, 145, 100),
            "exit_glass": (30, 125, 145, 100),
            "silvered_reflector_1": (150, 170, 200, 205),
            "silvered_reflector_2": (150, 170, 200, 205),
            "blackened_prism_face": (17, 23, 34, 230),
            "blackened_prism_side": (17, 23, 34, 230),
            "aperture_material": (45, 55, 75, 215),
        }
        for triangle, role in zip(geometry_screen, geometry.roles):
            xy = [tuple(float(value) for value in point) for point in triangle]
            draw.polygon(
                xy,
                fill=role_colors.get(role, (55, 75, 105, 150)),
                outline=(90, 140, 180, 190),
            )
    colors = (
        (80, 210, 255, 235), (115, 135, 255, 235),
        (235, 105, 255, 235), (255, 150, 95, 235),
    )
    for index, path_entry in enumerate(paths):
        path = path_points[index]
        count = len(path)
        line = screen[cursor:cursor+count]
        cursor += count
        measured = isinstance(path_entry, dict)
        if measured:
            rgb = np.asarray(path_entry["rgb"], np.float64)
            power = float(path_entry["relative_power"])
            color = tuple(
                int(value) for value in np.clip(rgb*255.0, 0.0, 255.0)
            )+(245,)
            glow = tuple(
                int(value) for value in np.clip(rgb*125.0, 0.0, 255.0)
            )+(max(20, int(120*math.sqrt(power))),)
            xy = [tuple(float(v) for v in point) for point in line]
            draw.line(xy, fill=glow, width=max(5, size//72))
        else:
            color = colors[index % len(colors)]
        draw.line(
            [tuple(float(v) for v in point) for point in line],
            fill=color, width=max(2, size//192),
        )
        for point in line[1:-1]:
            x, y = float(point[0]), float(point[1])
            draw.ellipse((x-2, y-2, x+2, y+2),
                         fill=(245, 250, 255, 255))
    if native_paths is not None:
        title = f"NATIVE LIGHT STATE axes={axes[0]}/{axes[1]}"
        legend = "COLOR=spectrum  GLOW=|E|^2"
    else:
        title = f"EXACT GEOMETRIC PATH axes={axes[0]}/{axes[1]}"
        legend = "path geometry; complex state not shown"
    draw.rectangle((5, 5, size-6, 31), fill=(4, 8, 14, 220))
    draw.text((10, 7), title, font=title_font, fill=(220, 235, 250, 255))
    draw.text((10, 19), legend, font=font, fill=(135, 175, 210, 255))
    return np.asarray(image, np.uint8)


def _component_static_panels(
    component,
    compiled,
    size: int,
    phase: float,
    *,
    native_paths: list[object] | None = None,
):
    contract = compiled.contract()
    materials = [
        f"{entry['role']}: {entry['material']} / {entry['interaction']}"
        for entry in contract["material_roles"]
    ] or ["no T3 material geometry; exact parametric artifact"]
    controls = [
        f"{entry['label']} = {entry['default']} {entry['unit']} "
        f"[{entry['rebuild_scope']}]"
        for entry in contract["controls"]
    ] or ["no authored controls"]
    ports = [
        f"{entry['key']}  {entry['direction']}  "
        f"{entry['representation']} axis={entry['axis']}"
        for entry in contract["ports"]
    ]
    status = [
        f"component: {contract['key']}",
        f"kind: {contract['component_kind']}",
        f"lanes: {contract['lane_count']}",
        f"T1/T3 triangles: {contract['transport_triangle_count']}",
        f"T2 artifacts: {len(compiled.graph.t2_payloads)}",
        f"T4 regions: {len(compiled.graph.t4_descriptors)}",
        "execution: compiled; never hot node interpretation",
    ]
    return (
        (
            _arena_geometry_panel(compiled, size),
            _arena_probe_panel(
                component, compiled, size, phase, native_paths
            ),
            _arena_graph_panel(compiled, size),
            _arena_text_panel("PORTS", ports, size),
            _arena_text_panel("CANONICAL MATERIAL ROLES", materials, size),
            _arena_text_panel("COMPONENT / CONTROLS", status+controls, size),
        ),
        ("GEOMETRY", "TRANSPORT PROBE", "GRAPH", "PORTS", "MATERIALS", "STATE"),
    )


def run_component_arena_live(
    component_key: str,
    *,
    lane_count: int = 1,
    size: int = 128,
    fps: int = 30,
    solve_hz: float = 8.0,
    wavelength_m: float = 532.0e-9,
    engine: str = "auto",
    _max_display_frames: int | None = None,
) -> None:
    """Load one canonical component into the shared OpenGL inspection arena."""

    if size < 64 or fps < 1 or not math.isfinite(solve_hz) or solve_hz <= 0.0:
        raise ValueError("component arena size/fps/solve_hz are invalid")
    import pygame
    from OpenGL import GL as gl
    from camera_software.gpu_preview import (
        GLPreviewCompositor, LatestOnlyProducer,
        PreviewProductKind, PreviewTextureProduct,
    )
    from camera_software.optical_components import (
        PentaprismComponent,
        PhysicalApertureComponent,
        PlaneMirrorComponent,
        build_native_component_scene,
        default_optical_component_registry,
    )

    registry = default_optical_component_registry()
    component = registry.create(component_key, lane_count)
    compiled = component.compile(lane_count, engine)
    compiled.validate()
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.set_mode(
        (1500, 820), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    )
    textures = [int(value) for value in gl.glGenTextures(6)]
    compositor = GLPreviewCompositor()
    compositor.init_gl()
    initial_panel = np.zeros((size, size, 4), np.uint8)
    initial_panel[..., 3] = 255
    for texture in textures:
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, size, size, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, initial_panel,
        )
    tracer = (
        _calibration_tracer(wavelength_m)
        if isinstance(component, PhysicalApertureComponent) else None
    )
    native_tracer = None
    native_submitted = False
    native_paths: list[object] | None = None
    if isinstance(component, (PlaneMirrorComponent, PentaprismComponent)):
        wavelengths = np.linspace(420.0e-9, 700.0e-9, lane_count)
        frequencies = 299_792_458.0/wavelengths
        native_scene = build_native_component_scene(compiled, frequencies)
        native_tracer = native_scene.create_tracer()
        native_tracer.ensure_pipeline(
            max_children=max(2, lane_count+1),
            min_amplitude=1.0e-12,
        )

    def produce_panels(request):
        solve_generation, solve_phase, measured_paths = request
        if isinstance(component, PhysicalApertureComponent):
            visual = _solve_aperture_visual(
                tracer,
                size=size,
                pitch_m=1.5e-6,
                wavelength_m=wavelength_m,
                cycle_phase=solve_phase,
                aperture_pattern="iris",
                polarization_mode="radial",
                quality="balanced",
                spectral_mode="fixed",
                lane_count=lane_count,
                spectral_epoch=0,
                source_angle_deg=0.0,
                analyzer_angle_deg=0.0,
                vector_page=True,
                piston_removed=True,
                coherence_seed=int(solve_generation) << 8,
                aperture_template=component.aperture,
            )
            return visual["panels"], visual["labels"]
        return _component_static_panels(
            component, compiled, size, solve_phase,
            native_paths=measured_paths,
        )

    panel_producer = LatestOnlyProducer(
        produce_panels, name=f"ComponentArena[{component_key}]"
    )
    clock = pygame.time.Clock()
    running, paused = True, False
    selected_panel = (
        1 if isinstance(component, (PlaneMirrorComponent, PentaprismComponent))
        else 0
    )
    solve_generation = displayed_frames = 0
    texture_generation = -1
    latest_result_id = -1
    latest_solve_ms = 0.0
    phase = 0.0
    previous_ui_s = time.monotonic()
    next_solve_s = previous_ui_s
    labels = ("WAITING",) * 6
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
                    elif event.key in (
                        pygame.K_TAB, pygame.K_RIGHT, pygame.K_DOWN,
                    ):
                        selected_panel = (selected_panel+1) % 6
                    elif event.key in (pygame.K_LEFT, pygame.K_UP):
                        selected_panel = (selected_panel-1) % 6
                    elif pygame.K_1 <= event.key <= pygame.K_6:
                        selected_panel = int(event.key-pygame.K_1)
            now_s = time.monotonic()
            if not paused:
                phase += max(0.0, now_s-previous_ui_s)*0.75
            previous_ui_s = now_s
            if (
                native_tracer is not None
                and native_submitted
                and int(native_tracer.in_flight_count()) == 0
            ):
                records = native_tracer.drain_records(4096)
                starts = np.asarray(records["seg_start"], np.float64)
                ends = np.asarray(records["pos"], np.float64)
                fields = (
                    np.asarray(records["amp_re"], np.float64)
                    + 1j*np.asarray(records["amp_im"], np.float64)
                )
                powers = np.sum(np.abs(fields)**2, axis=1)
                peak_power = max(float(np.max(powers, initial=0.0)), 1.0e-30)
                from camera_software.transport_contract import (
                    native_display_rgb_weight,
                )
                rgb_weights = np.asarray([
                    native_display_rgb_weight(value*1.0e9)
                    for value in wavelengths
                ], np.float64)
                native_paths = []
                for start, end, lane_power, power in zip(
                    starts, ends, np.abs(fields)**2, powers
                ):
                    if (
                        not np.all(np.isfinite(start))
                        or not np.all(np.isfinite(end))
                        or float(np.linalg.norm(end-start)) <= 1.0e-12
                    ):
                        continue
                    rgb = np.asarray(lane_power @ rgb_weights, np.float64)
                    rgb_peak = float(np.max(rgb, initial=0.0))
                    if rgb_peak > 0.0:
                        rgb /= rgb_peak
                    native_paths.append({
                        "points": np.stack((start, end)),
                        "rgb": np.sqrt(np.clip(rgb, 0.0, 1.0)),
                        "relative_power": float(power/peak_power),
                    })
                native_submitted = False
            if (
                native_tracer is not None
                and not paused
                and not native_submitted
                and int(native_tracer.in_flight_count()) == 0
            ):
                if isinstance(component, PlaneMirrorComponent):
                    direction = _arena_unit(
                        (1.0, 0.18*math.sin(phase), 0.0), "mirror probe"
                    )
                    center_origin = (
                        np.asarray(component.center_m, np.float64)
                        - direction*2.2*component.radius_m
                    )
                    tangent = _arena_unit(
                        np.cross(
                            np.asarray(component.normal, np.float64),
                            (0.0, 0.0, 1.0),
                        ),
                        "mirror bundle tangent",
                    )
                    origins = np.stack([
                        center_origin+offset*component.radius_m*tangent
                        for offset in np.linspace(-0.68, 0.68, 11)
                    ])
                else:
                    assembly = component.spec.build()
                    direction = _arena_unit(
                        component.spec.input_axis, "pentaprism probe"
                    )
                    center_origin = (
                        np.asarray(assembly.primary_path[0], np.float64)
                        - direction*0.25*component.spec.clear_size_m
                    )
                    extrusion = _arena_unit(
                        np.cross(
                            component.spec.input_axis,
                            component.spec.output_axis,
                        ),
                        "pentaprism bundle tangent",
                    )
                    cross_section = _arena_unit(
                        component.spec.output_axis,
                        "pentaprism in-plane bundle tangent",
                    )
                    origins = np.stack([
                        center_origin
                        + in_plane*component.spec.clear_size_m*cross_section
                        + depth*component.spec.depth_m*extrusion
                        for depth in np.linspace(-0.28, 0.28, 3)
                        for in_plane in np.linspace(-0.055, 0.055, 9)
                    ])
                amplitudes = np.full(
                    (len(origins), lane_count),
                    1.0/math.sqrt(max(1, lane_count*len(origins)))+0.0j,
                    np.complex128,
                )
                native_tracer.submit_rays(
                    origins,
                    np.repeat(direction[None], len(origins), axis=0),
                    amplitudes,
                    max_bounces=12,
                    min_amplitude=1.0e-12,
                    max_children=max(2, lane_count+1),
                )
                native_submitted = True
            if not paused and now_s >= next_solve_s:
                solve_generation += 1
                panel_producer.request(
                    (solve_generation, phase, native_paths)
                )
                next_solve_s = now_s + 1.0/float(solve_hz)

            result = panel_producer.poll(after_request_id=latest_result_id)
            if result is not None:
                latest_result_id = result.request_id
                if result.error is not None:
                    raise RuntimeError(
                        "component live texture producer failed"
                    ) from result.error
                panels, labels = result.payload
                for texture, rgba in zip(textures, panels):
                    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
                    gl.glTexSubImage2D(
                        gl.GL_TEXTURE_2D, 0, 0, 0, size, size,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                        np.ascontiguousarray(rgba),
                    )
                texture_generation = result.request_id
                latest_solve_ms = result.elapsed_s*1.0e3
            width, height = pygame.display.get_window_size()
            gl.glViewport(0, 0, width, height)
            gl.glClearColor(0.015, 0.022, 0.035, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gap = max(8, min(width, height)//80)
            display_size = max(1, min(width-2*gap, height-2*gap))
            display_x = max(0, (width-display_size)//2)
            display_y = max(0, (height-display_size)//2)
            index = selected_panel
            compositor.draw(
                PreviewTextureProduct(
                    product_id=f"component.{component_key}.{index}",
                    tab_label=labels[index],
                    texture_id=textures[index],
                    width=size,
                    height=size,
                    generation=max(0, texture_generation),
                    producer="canonical-optical-component-arena",
                    internal_format=int(gl.GL_RGBA8),
                    kind=(
                        PreviewProductKind.COMPLEX_FIELD
                        if isinstance(component, PhysicalApertureComponent)
                        else (
                            PreviewProductKind.CAMERA_GEOMETRY,
                            PreviewProductKind.LIGHT_FIELD,
                            PreviewProductKind.PROCESSING_GROUP,
                            PreviewProductKind.PROCESSING_GROUP,
                            PreviewProductKind.PROCESSING_GROUP,
                            PreviewProductKind.ACCUMULATION,
                        )[index]
                    ),
                    orientation="top-left",
                    alpha_mode="straight",
                ),
                (display_x, display_y, display_size, display_size),
                height,
                tone_map=False,
            )
            pygame.display.flip()
            displayed_frames += 1
            pygame.display.set_caption(
                f"Optical component arena — {component_key} "
                f"[{compiled.component_kind}] lanes={lane_count} "
                f"engine={compiled.metadata['selected_engine']} "
                f"[{selected_panel+1}/6 {labels[selected_panel]}] "
                f"texture={texture_generation} solve={latest_solve_ms:.1f}ms "
                f"producer={'BUSY' if panel_producer.busy else 'READY'} "
                f"{'PAUSED ' if paused else ''}"
                "[1-6/Tab/Arrows select, Space pause, Esc close]"
            )
            clock.tick(fps)
            if (
                _max_display_frames is not None
                and displayed_frames >= _max_display_frames
            ):
                running = False
    finally:
        panel_producer.close()
        if native_tracer is not None:
            native_tracer.stop_pipeline()
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
        "--component-live", action="store_true",
        help="load a canonical optical component into the shared OpenGL arena",
    )
    parser.add_argument(
        "--component", default="aperture.iris",
        help="component registry key for --component-live",
    )
    parser.add_argument(
        "--component-lanes", type=int, choices=_APERTURE_LANE_COUNTS, default=1,
        help="exact compiled lane width for --component-live",
    )
    parser.add_argument(
        "--component-size", type=int, default=128,
        help="square source-texture resolution for --component-live",
    )
    parser.add_argument(
        "--component-solve-hz", type=float, default=8.0,
        help="maximum asynchronous texture generations per second",
    )
    parser.add_argument(
        "--component-engine",
        choices=("auto", "ray", "parametric", "wave", "hybrid", "maxwell"),
        default="auto",
        help="required optical backend; unsupported requests never fall back",
    )
    parser.add_argument(
        "--list-components", action="store_true",
        help="print canonical optical-component registry keys and exit",
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
        "--ultra-pattern", choices=_APERTURE_PATTERNS,
        help="override preset physical aperture or scrim geometry",
    )
    parser.add_argument(
        "--ultra-output-size", type=int,
        help="publish each composite as an exact square RGBA texture",
    )
    parser.add_argument(
        "--ultra-panel",
        help=(
            "publish one raw square scientific panel by index or name "
            "(physical, spectral, power, polarization, stokes-q, stokes-v)"
        ),
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
    parser.add_argument(
        "--aperture-pattern",
        choices=_APERTURE_PATTERNS,
        default="iris",
        help="finite material aperture/scrim geometry",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--cycle-steps", type=int, default=120,
        help="forward steps before the live animation reverses",
    )
    args = parser.parse_args()
    if args.list_components:
        from camera_software.optical_components import (
            default_optical_component_registry,
        )
        print(json.dumps(
            list(default_optical_component_registry().keys()), indent=2
        ))
        return 0
    if args.list_ultra_bakes:
        print(json.dumps(ultra_bake_presets(), indent=2))
        return 0
    if sum((
        bool(args.live),
        bool(args.transport_live),
        bool(args.aperture_live),
        bool(args.component_live),
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
            aperture_pattern=args.ultra_pattern,
            output_size=args.ultra_output_size,
            panel=args.ultra_panel,
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
            aperture_pattern=args.aperture_pattern,
        )
        return 0
    if args.component_live:
        run_component_arena_live(
            args.component,
            lane_count=args.component_lanes,
            size=max(64, args.component_size),
            fps=args.fps,
            solve_hz=args.component_solve_hz,
            engine=args.component_engine,
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
