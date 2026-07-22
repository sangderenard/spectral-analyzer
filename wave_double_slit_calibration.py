"""Durable CPU/GPU wave-only Double Slit calibration runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
SHADER_PATH = ROOT / "csrc" / "shaders" / "ray_wave_bpm.comp.glsl"
CHECKPOINT_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _wavelength_rgb(wavelength_nm: float) -> np.ndarray:
    w = float(wavelength_nm)
    if w < 440: rgb = (-(w - 440) / 60, 0.0, 1.0)
    elif w < 490: rgb = (0.0, (w - 440) / 50, 1.0)
    elif w < 510: rgb = (0.0, 1.0, -(w - 510) / 20)
    elif w < 580: rgb = ((w - 510) / 70, 1.0, 0.0)
    elif w < 645: rgb = (1.0, -(w - 645) / 65, 0.0)
    else: rgb = (1.0, 0.0, 0.0)
    return np.asarray(rgb, np.float32)


def initial_field(
    wavelengths_m: np.ndarray,
    height: int,
    width: int,
    dx_m: float,
    slit_width_m: float,
    slit_separation_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(width, dtype=np.float64) - 0.5 * (width - 1)) * dx_m
    half_width = 0.5 * slit_width_m
    centers = (-0.5 * slit_separation_m, 0.5 * slit_separation_m)
    aperture = np.logical_or.reduce(tuple(np.abs(x - center) <= half_width for center in centers))
    re = np.zeros((len(wavelengths_m), height, width), np.float32)
    re[:, 1:-1, aperture] = 1.0
    im = np.zeros_like(re)
    return re, im


def _cpu_tracer(wavelengths_m: np.ndarray):
    import _spectral_kernels as kernels
    from ray_tracer_bridge import per_tri_spectral_to_mat_buf

    frequencies = 299_792_458.0 / wavelengths_m
    bands = len(wavelengths_m)
    reflectance = np.zeros((1, bands), np.float64)
    mat_idx, mat_buf, mat_count = per_tri_spectral_to_mat_buf(
        reflectance, reflectance, reflectance, frequencies,
    )
    triangle = np.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], np.float64)
    normal = np.asarray([[0.0, 0.0, 1.0]], np.float64)
    return kernels.RayTracer(
        1, triangle, normal, mat_idx, mat_buf, int(mat_count),
        frequencies.astype(np.float64), 299_792_458.0,
        np.zeros(bands, np.float64),
    )


def run_cpu_steps(
    re: np.ndarray,
    im: np.ndarray,
    wavelengths_m: np.ndarray,
    dx_m: float,
    dz_m: float,
    start_step: int,
    total_steps: int,
    checkpoint: Callable[[int, np.ndarray, np.ndarray], None],
) -> dict[str, Any]:
    tracer = _cpu_tracer(wavelengths_m)
    bands, height, width = re.shape
    started = time.perf_counter()
    for step in range(start_step, total_steps):
        tracer.wave_bpm_step(
            bands, width, height, dx_m, dz_m,
            wavelengths_m.astype(np.float64), re, im,
        )
        if (step + 1) % 5 == 0 or step + 1 == total_steps:
            checkpoint(step + 1, re, im)
    return {
        "backend": "compiled-cpu-adi-bpm",
        "elapsed_s": time.perf_counter() - started,
    }


def run_gpu_steps(
    re: np.ndarray,
    im: np.ndarray,
    wavelengths_m: np.ndarray,
    dx_m: float,
    dz_m: float,
    start_step: int,
    total_steps: int,
    checkpoint: Callable[[int, np.ndarray, np.ndarray], None],
) -> dict[str, Any]:
    import pygame
    from OpenGL import GL
    from OpenGL.GL import shaders

    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.HIDDEN)
    shader_source = SHADER_PATH.read_text(encoding="utf-8")
    program = shaders.compileProgram(
        shaders.compileShader(shader_source, GL.GL_COMPUTE_SHADER)
    )
    bands, height, width = re.shape
    if bands > 16 or max(width, height) > 1024:
        raise ValueError("GPU BPM shader supports at most 16 bands and 1024x1024")
    arrays = [
        np.ascontiguousarray(re.reshape(-1), np.float32),
        np.ascontiguousarray(im.reshape(-1), np.float32),
        np.zeros(re.size, np.float32), np.zeros(re.size, np.float32),
    ]
    lane_count = bands * max(width, height)
    scratch_bytes = lane_count * 1024 * 2 * np.dtype(np.float32).itemsize
    buffers = GL.glGenBuffers(6)
    try:
        for binding, buffer in enumerate(buffers):
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffer)
            if binding < 4:
                GL.glBufferData(
                    GL.GL_SHADER_STORAGE_BUFFER, arrays[binding].nbytes,
                    arrays[binding], GL.GL_DYNAMIC_COPY,
                )
            else:
                GL.glBufferData(
                    GL.GL_SHADER_STORAGE_BUFFER, scratch_bytes, None,
                    GL.GL_DYNAMIC_COPY,
                )
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, buffer)
        GL.glUseProgram(program)
        uniforms = {
            name: GL.glGetUniformLocation(program, name)
            for name in ("mode", "nx", "ny", "n_bands", "dx", "dz", "wavelengths")
        }
        GL.glUniform1i(uniforms["nx"], width)
        GL.glUniform1i(uniforms["ny"], height)
        GL.glUniform1i(uniforms["n_bands"], bands)
        GL.glUniform1f(uniforms["dx"], float(dx_m))
        GL.glUniform1f(uniforms["dz"], float(dz_m))
        GL.glUniform1fv(
            uniforms["wavelengths"], bands,
            np.ascontiguousarray(wavelengths_m, np.float32),
        )
        started = time.perf_counter()
        for step in range(start_step, total_steps):
            for mode, groups in (
                (0, bands * height * width),
                (1, bands * height),
                (2, bands * width),
            ):
                GL.glUniform1i(uniforms["mode"], mode)
                GL.glDispatchCompute(groups, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            if (step + 1) % 5 == 0 or step + 1 == total_steps:
                GL.glFinish()
                for binding, target in ((0, re), (1, im)):
                    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffers[binding])
                    raw = GL.glGetBufferSubData(
                        GL.GL_SHADER_STORAGE_BUFFER, 0, target.nbytes
                    )
                    target[...] = np.frombuffer(raw, np.float32).reshape(target.shape)
                checkpoint(step + 1, re, im)
        return {
            "backend": "opengl-compute-ray_wave_bpm",
            "elapsed_s": time.perf_counter() - started,
            "gl_vendor": (GL.glGetString(GL.GL_VENDOR) or b"").decode(errors="replace"),
            "gl_renderer": (GL.glGetString(GL.GL_RENDERER) or b"").decode(errors="replace"),
            "gl_version": (GL.glGetString(GL.GL_VERSION) or b"").decode(errors="replace"),
            "shader_path": str(SHADER_PATH),
            "shader_sha256": hashlib.sha256(shader_source.encode()).hexdigest(),
        }
    finally:
        GL.glUseProgram(0)
        GL.glDeleteProgram(program)
        GL.glDeleteBuffers(6, buffers)
        pygame.display.quit()


def save_strip(
    intensity: np.ndarray, wavelengths_nm: np.ndarray, path: Path, title: str,
    scales: np.ndarray | None = None,
) -> None:
    bands, height, width = intensity.shape
    label_h = 28
    canvas = Image.new("RGB", (bands * width, height + label_h), (10, 11, 15))
    draw = ImageDraw.Draw(canvas)
    for band, wavelength in enumerate(wavelengths_nm):
        values = np.maximum(intensity[band], 0.0)
        positive = values[values > 0.0]
        scale = (
            float(scales[band]) if scales is not None else
            float(np.percentile(positive, 99.7)) if positive.size else 1.0
        )
        shown = np.sqrt(np.clip(values / max(scale, 1.0e-30), 0.0, 1.0))
        rgb = shown[..., None] * _wavelength_rgb(float(wavelength))[None, None, :]
        tile = Image.fromarray(np.asarray(rgb * 255.0, np.uint8), "RGB")
        canvas.paste(tile, (band * width, label_h))
        draw.text((band * width + 5, 5), f"{wavelength:.0f} nm", fill=(230, 232, 238))
    draw.text((max(0, canvas.width - 150), 5), title, fill=(190, 198, 218))
    canvas.save(path)


def save_band_images(
    intensity: np.ndarray, wavelengths_nm: np.ndarray, directory: Path,
    scales: np.ndarray,
) -> None:
    for band, wavelength in enumerate(wavelengths_nm):
        shown = np.sqrt(np.clip(
            np.maximum(intensity[band], 0.0) / max(float(scales[band]), 1.0e-30),
            0.0, 1.0,
        ))
        rgb = shown[..., None] * _wavelength_rgb(float(wavelength))[None, None, :]
        Image.fromarray(np.asarray(rgb * 255.0, np.uint8), "RGB").save(
            directory / f"band_{int(round(float(wavelength)))}nm.png"
        )


def _comparison(cpu_path: Path, gpu_path: Path, output: Path) -> None:
    cpu = Image.open(cpu_path).convert("RGB")
    gpu = Image.open(gpu_path).convert("RGB")
    canvas = Image.new("RGB", (max(cpu.width, gpu.width), cpu.height + gpu.height), (8, 9, 12))
    canvas.paste(cpu, (0, 0)); canvas.paste(gpu, (0, cpu.height))
    canvas.save(output)


def run_calibration(work_dir: str, width: int = 200, height: int = 200, steps: int = 100) -> dict[str, Any]:
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    wavelengths_nm = np.asarray([450.0, 500.0, 550.0, 600.0, 650.0], np.float64)
    wavelengths_m = wavelengths_nm * 1.0e-9
    dx_m, dz_m = 2.0e-6, 20.0e-6
    state_path = root / "wave_checkpoint.json"
    state = {
        "schema_version": CHECKPOINT_VERSION,
        "width": int(width), "height": int(height), "steps": int(steps),
        "wavelengths_nm": wavelengths_nm.tolist(),
        "backends": {}, "status": "working",
    }
    if state_path.is_file():
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        compatible = all(loaded.get(key) == state[key] for key in ("schema_version", "width", "height", "steps", "wavelengths_nm"))
        if compatible:
            state = loaded
    entry_re, entry_im = initial_field(
        wavelengths_m, height, width, dx_m, 8e-6, 40e-6
    )
    np.save(root / "entry_field_real.npy", entry_re, allow_pickle=False)
    np.save(root / "entry_field_imag.npy", entry_im, allow_pickle=False)
    metadata: dict[str, Any] = {}
    for backend, solver in (("cpu", run_cpu_steps), ("gpu", run_gpu_steps)):
        backend_dir = root / backend
        backend_dir.mkdir(exist_ok=True)
        re_path, im_path = backend_dir / "field_real.npy", backend_dir / "field_imag.npy"
        completed = int(dict(state.get("backends", {}).get(backend, {})).get("completed_steps", 0))
        if completed > 0 and re_path.is_file() and im_path.is_file():
            re = np.load(re_path, allow_pickle=False)
            im = np.load(im_path, allow_pickle=False)
        else:
            re, im = entry_re.copy(), entry_im.copy()
            completed = 0

        def checkpoint(done: int, current_re: np.ndarray, current_im: np.ndarray) -> None:
            np.save(re_path, current_re, allow_pickle=False)
            np.save(im_path, current_im, allow_pickle=False)
            intensity = current_re * current_re + current_im * current_im
            np.save(backend_dir / "intensity.npy", intensity, allow_pickle=False)
            save_strip(intensity, wavelengths_nm, backend_dir / "bands.png", backend.upper())
            state.setdefault("backends", {})[backend] = {
                **dict(state.get("backends", {}).get(backend, {})),
                "completed_steps": int(done), "status": "working" if done < steps else "complete",
            }
            if (root / "cpu" / "bands.png").is_file() and (root / "gpu" / "bands.png").is_file():
                _comparison(root / "cpu" / "bands.png", root / "gpu" / "bands.png", root / "comparison.png")
            _atomic_json(state_path, state)
            print("[wave-progress] " + json.dumps({
                "backend": backend, "completed": (0 if backend == "cpu" else steps) + done,
                "total": 2 * steps, "preview_path": str(root / backend / "bands.png"),
            }), flush=True)

        if completed < steps:
            metadata[backend] = solver(re, im, wavelengths_m, dx_m, dz_m, completed, steps, checkpoint)
        else:
            metadata[backend] = dict(state["backends"][backend].get("metadata", {}))
        state["backends"][backend]["metadata"] = metadata[backend]
        _atomic_json(state_path, state)
    cpu_i = np.load(root / "cpu" / "intensity.npy", allow_pickle=False)
    gpu_i = np.load(root / "gpu" / "intensity.npy", allow_pickle=False)
    difference = float(np.linalg.norm(cpu_i - gpu_i) / max(np.linalg.norm(cpu_i), 1.0e-30))
    shared_scales = np.asarray([
        max(
            float(np.percentile(cpu_i[band][cpu_i[band] > 0.0], 99.7)),
            float(np.percentile(gpu_i[band][gpu_i[band] > 0.0], 99.7)),
        )
        for band in range(len(wavelengths_nm))
    ], np.float64)
    save_strip(cpu_i, wavelengths_nm, root / "cpu" / "bands.png", "CPU", shared_scales)
    save_strip(gpu_i, wavelengths_nm, root / "gpu" / "bands.png", "GPU", shared_scales)
    save_band_images(cpu_i, wavelengths_nm, root / "cpu", shared_scales)
    save_band_images(gpu_i, wavelengths_nm, root / "gpu", shared_scales)
    _comparison(root / "cpu" / "bands.png", root / "gpu" / "bands.png", root / "comparison.png")
    propagation_m = steps * dz_m
    metrics = []
    for band, wavelength_m in enumerate(wavelengths_m):
        profile = np.mean(cpu_i[band, height // 4:3 * height // 4], axis=0)
        analytical_px = float(
            wavelength_m * propagation_m / 40.0e-6 / dx_m
        )
        candidates = np.flatnonzero(
            (profile[1:-1] > profile[:-2])
            & (profile[1:-1] >= profile[2:])
            & (profile[1:-1] > 0.08 * float(np.max(profile)))
        ) + 1
        selected: list[int] = []
        minimum_peak_distance = max(2.0, 0.55 * analytical_px)
        for candidate in sorted(candidates, key=lambda index: profile[index], reverse=True):
            if all(abs(int(candidate) - prior) >= minimum_peak_distance for prior in selected):
                selected.append(int(candidate))
        selected.sort()
        observed = float(np.median(np.diff(selected))) if len(selected) >= 2 else 0.0
        metrics.append({
            "wavelength_nm": float(wavelength_nm := wavelengths_nm[band]),
            "analytical_fringe_spacing_px": analytical_px,
            "observed_peak_spacing_px": observed,
            "cpu_symmetry_relative_l1": float(
                np.mean(np.abs(cpu_i[band] - np.flip(cpu_i[band], axis=1)))
                / max(float(np.mean(cpu_i[band])), 1.0e-30)
            ),
            "gpu_symmetry_relative_l1": float(
                np.mean(np.abs(gpu_i[band] - np.flip(gpu_i[band], axis=1)))
                / max(float(np.mean(gpu_i[band])), 1.0e-30)
            ),
        })
    state.update({
        "status": "complete", "intensity_relative_l2": difference,
        "band_metrics": metrics,
        "shared_display_scales": shared_scales.tolist(),
        "preview_path": str(root / "comparison.png"),
        "completed_at_s": time.time(),
    })
    _atomic_json(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    result = run_calibration(args.work_dir, args.width, args.height, args.steps)
    print("[wave-result] " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
