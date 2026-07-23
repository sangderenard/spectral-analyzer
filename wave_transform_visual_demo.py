"""Visual confirmation for the production T4 angular-spectrum transform.

This is not a second wave solver.  It drives
``RayTracer.t4_angular_spectrum_step`` -- the same native kernel used by
persistent T4 arenas -- and only performs presentation/capture in Python.

The source is a pair of finite coherent Gaussian emitters.  No ideal aperture
or hard array mask is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

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
    tracer.add_scale_context(
        np.asarray([0.0, 0.0, 0.0], np.float64),
        96.0e-6,
        1,
        4.0e-6,
        32,
        1.0,
        0.0,
        1,
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float64),
    )
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


def _phase_rgba(field: np.ndarray) -> np.ndarray:
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


def _transport_table_rgba(
    arena: dict[str, object],
    detector_position: np.ndarray | None,
    size: int,
) -> np.ndarray:
    """Draw actual native boundary telemetry as an orthographic X/Z table."""

    image = Image.new("RGBA", (size, size), (7, 11, 18, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    boundary = dict(arena["boundary"])
    center = np.asarray(arena["center_world"], np.float64)
    half_x = float(arena["transverse_half_extent_m"])
    half_z = 0.5 * float(arena["longitudinal_extent_m"])
    entry = np.asarray(boundary["entry_world"], np.float64)
    exit_position = np.asarray(boundary["exit_world"], np.float64)
    entry_direction = np.asarray(boundary["entry_direction"], np.float64)
    exit_direction = np.asarray(boundary["exit_direction"], np.float64)
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

    patch_lo = point(center + np.asarray([-half_x, 0.0, -half_z]))
    patch_hi = point(center + np.asarray([half_x, 0.0, half_z]))
    rectangle = (
        min(patch_lo[0], patch_hi[0]), min(patch_lo[1], patch_hi[1]),
        max(patch_lo[0], patch_hi[0]), max(patch_lo[1], patch_hi[1]),
    )
    draw.rectangle(rectangle, fill=(18, 48, 72, 220), outline=(55, 210, 245, 255), width=2)
    draw.text((rectangle[0]+5, rectangle[1]+5), "T4 FIELD PATCH", font=font,
              fill=(120, 225, 255, 255))

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

    draw.text((8, 8), "ACTUAL T1 -> T4 -> T1 -> DETECTOR", font=font,
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


def _boundary_power_rgba(arena: dict[str, object], size: int) -> np.ndarray:
    image = Image.new("RGBA", (size, size), (7, 11, 18, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    boundary = dict(arena["boundary"])
    values = [
        ("INPUT RAY", float(boundary["input_ray_power"]), (255, 185, 70, 255)),
        ("SEEDED FIELD", float(boundary["seeded_field_power"]), (80, 205, 255, 255)),
        ("PROPAGATED", float(boundary["propagated_field_power"]), (120, 125, 255, 255)),
        ("OUTPUT RAY", float(boundary["output_ray_power"]), (100, 255, 150, 255)),
    ]
    peak = max([value for _label, value, _color in values] + [1.0e-30])
    draw.text((8, 8), "BOUNDARY POWER CONTRACT", font=font,
              fill=(225, 235, 250, 255))
    for index, (label, value, color) in enumerate(values):
        y = 42 + index * max(36, (size-75)//4)
        draw.text((10, y), f"{label}  {value:.6e}", font=font, fill=color)
        bar_width = int(round((size-24) * min(1.0, value/peak)))
        draw.rectangle((10, y+16, 10+bar_width, y+28), fill=color)
    seeded = float(boundary["seeded_field_power"])
    propagated = float(boundary["propagated_field_power"])
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
        (1320, 500), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    )
    tracer = _transport_calibration_tracer(wavelength_m)
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
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
            panel_size, panel_size, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
        )
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    panels = [
        np.zeros((panel_size, panel_size, 4), np.uint8)
        for _ in range(3)
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
                arena = dict(tracer.wave_arena_stats()[0])
                boundary = dict(arena["boundary"])
                detector_position = None
                kinds = np.asarray(records["kind"])
                strike_indices = np.flatnonzero(kinds == 0)
                if strike_indices.size:
                    detector_position = np.asarray(
                        records["pos"], np.float64
                    )[int(strike_indices[-1])]
                field_snapshot = tracer.wave_arena_field_snapshot(
                    0, int(boundary["direction"]), 0, 0
                )
                field = (
                    np.asarray(field_snapshot["re"], np.float64)
                    + 1j*np.asarray(field_snapshot["im"], np.float64)
                )
                phase = Image.fromarray(_phase_rgba(field), "RGBA").resize(
                    (panel_size, panel_size), Image.Resampling.BICUBIC
                )
                panels = [
                    _transport_table_rgba(arena, detector_position, panel_size),
                    np.asarray(phase, np.uint8),
                    _boundary_power_rgba(arena, panel_size),
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
            pane_width = max(1, (width-gap*4)//3)
            pane_height = max(1, height-gap*2)
            for index, texture in enumerate(textures):
                compositor.draw(
                    PreviewTextureProduct(
                        product_id=f"t4.transport.{index}",
                        tab_label=("TABLE", "EXIT FIELD", "POWER")[index],
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
                "Native transport table — ACTUAL PATH | EXIT PHASE | POWER "
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
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--cycle-steps", type=int, default=120,
        help="forward steps before the live animation reverses",
    )
    args = parser.parse_args()
    if args.live and args.transport_live:
        parser.error("choose --live or --transport-live")
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
