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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="exposures/wave_transform_visual"
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
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
