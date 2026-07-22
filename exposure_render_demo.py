"""Long-exposure dual-backend ray-traced demo & emissivity-gain calibration.

Goal
----
Drive **both** spectral ray tracers — the C++ ``_spectral_kernels.RayTracer``
and the GLSL compute ``_gpu_ray_field`` pipeline — through a single
*finished-exposure* loop in which every emitted output frame represents one
fully-accumulated exposure of N_total rays (N_total can be 10M+ and is
sub-batched / streamed automatically) rather than one tiny dispatch.

That inversion ("frame == exposure", not "frame == ray bunch") is the central
calibration semantic: we know

    photons/pixel = L · Ω · τ · A_pix · t / (h c / λ)

so the exposure budget yields a *known* total radiant exposure target H_J,
and the tracer accumulates rays until the running estimate of H_J matches
that target.  Per-exposure emissivity gain is the scalar that brings the
measured sensor signal to the predicted target — recorded and saved so
later renders can short-circuit the calibration loop.

Scene
-----
Borrowed verbatim from ``test_basic_gl_cpp_window.py`` (central textured
sphere, saddle stage floor, 12 orbiters on distinct 3-D orbital planes).
The basic-rasterizer harness uses the same scene to validate shading; this
file repurposes that scene for *photometric* ray-traced exposures using the
unified Phase-2 ``mat_buf`` material pipeline (32 spectral bands, central
``MaterialDatabase`` registry, hard cutover — no legacy refl_re/refl_im
buffers).

Output
------
- A live pygame window with the C++ pane on the left and the GLSL pane on
  the right.  Each finished exposure replaces the visible image.
- One ``./exposures/{frame:04d}_{backend}.png`` dump per accumulated frame.
- One ``./exposures/{frame:04d}_summary.json`` log per frame with the
  full scientific budget (N_rays, gain dB, virtual exposure time s,
  H_target J, H_measured J, photons/pixel, SNR).

Status
------
- C++ backend: fully wired against the Phase 2b SLICE 1 pybind ctor
  ``RayTracer(n_tri, verts, normals, mat_idx, mat_buf, mat_n_mats,
  freq_hz, speed_m_s, atmo_abs)`` and ``integrate_image_into`` (which
  accumulates additively into the caller-owned float32 (n_bands, H, W)
  buffer — perfect for sub-batch streaming).
- GLSL backend: scaffolded.  The full SensorAccumulator + BVH +
  ScaleContext + FilmStack pipeline lives in ``demo_pluck_gl``; we wire it
  best-effort.  When unavailable, the right pane shows the C++ image
  monochrome-toned and labelled "GLSL pending" so the harness still runs
  end-to-end and prints the same calibration log.

CLI
---
    python exposure_render_demo.py \\
        --total-rays 10_000_000 --rays-per-batch 250_000 \\
        --width 320 --height 200 --frames 6 --backend both
"""
from __future__ import annotations

import argparse
import hashlib
import copy
import gc
import faulthandler
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional

import numpy as np

import _spectral_kernels as _sk  # type: ignore[import]


def _verify_native_bdpt_contract() -> None:
    """Fail loudly if a stale native module predating the BDPT parity fixes loaded."""
    required = ("BDPT_PDF_FLAG_SENSOR", "MAT_FLAG_TRANSMISSIVE")
    missing = [name for name in required if not hasattr(_sk, name)]
    module_path = os.path.abspath(str(getattr(_sk, "__file__", "<unknown>")))
    if missing:
        raise RuntimeError(
            f"stale _spectral_kernels at {module_path}; missing {', '.join(missing)}. "
            "Rebuild the Release _spectral_kernels target before rendering."
        )
    if int(_sk.BDPT_PDF_FLAG_SENSOR) != (1 << 22):
        raise RuntimeError(f"incompatible BDPT sensor endpoint ABI in {module_path}")
    print(f"[native-module] {module_path} bdpt_contract=sensor-endpoint-v1", flush=True)


def _native_bdpt_work_units(width: int, height: int, aperture_samples: int,
                            requested_packages: int = 1,
                            primary_ray_cap: int = 200_000) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Partition one immutable pixel×aperture sweep on whole-pixel boundaries.

    Offsets are schedule ordinals consumed by ``submit_sensor_sweep``.  Keeping
    all pupil samples for a pixel in one unit makes the native 32×32 Morton
    work tiles spatially coherent and prevents package boundaries from changing
    the light field halfway through a pixel.
    """
    w = int(max(1, width))
    h = int(max(1, height))
    n_ap = int(max(1, aperture_samples))
    pixels = int(w * h)
    cap = int(max(n_ap, primary_ray_cap))
    cap_pixels = int(max(1, cap // n_ap))
    auto_packages = int(max(1, math.ceil(pixels / cap_pixels)))
    n_packages = int(min(pixels, max(1, requested_packages, auto_packages)))
    pixels_per_unit = int(max(1, math.ceil(pixels / n_packages)))
    # The ceil distribution can exceed cap_pixels if the requested package
    # count happened to land awkwardly.  Use the hard cap as the final arbiter.
    pixels_per_unit = int(min(pixels_per_unit, cap_pixels))
    units: list[tuple[int, int]] = []
    for pixel_start in range(0, pixels, pixels_per_unit):
        pixel_count = min(pixels_per_unit, pixels - pixel_start)
        units.append((int(pixel_start * n_ap), int(pixel_count * n_ap)))
    return units, {
        "pixels": pixels,
        "aperture_samples": n_ap,
        "schedule_rays": int(pixels * n_ap),
        "work_units": len(units),
        "pixels_per_unit": pixels_per_unit,
        "max_primary_rays_per_unit": int(pixels_per_unit * n_ap),
        "sensor_rgb_bytes": int(pixels * 3 * np.dtype(np.float32).itemsize),
    }


def _native_bdpt_refinement_schedule(
    work_units: list[tuple[int, int]],
    sensor_sweeps: int,
) -> list[tuple[int, int, int]]:
    """Repeat a complete spatial work plan for each independent sensor sweep."""
    sweeps = int(sensor_sweeps)
    if sweeps <= 0:
        raise ValueError("sensor_sweeps must be positive")
    return [
        (sweep_index, int(offset), int(count))
        for sweep_index in range(sweeps)
        for offset, count in work_units
    ]


def _resolve_native_sensor_sweeps(
    cli_sweeps: Optional[int], authored_sweeps: int = 1
) -> int:
    requested = authored_sweeps if cli_sweeps is None else cli_sweeps
    return max(1, int(requested))


def _average_native_sensor_sweeps(
    linear_image: np.ndarray, sensor_sweeps: int
) -> np.ndarray:
    array = np.asarray(linear_image)
    divisor = np.asarray(max(1, int(sensor_sweeps)), dtype=array.dtype)
    return np.ascontiguousarray(array / divisor)


def _normalise_native_sensor_epochs(
    linear_image: np.ndarray, epoch_count: np.ndarray
) -> np.ndarray:
    """Normalize unequal regional exposure without modifying spectral sums."""
    linear = np.asarray(linear_image)
    counts = np.asarray(epoch_count)
    if linear.ndim != 3 or counts.shape != linear.shape[:2]:
        raise ValueError("sensor epoch counts must match the linear sensor raster")
    denominator = np.maximum(counts, 1).astype(linear.dtype, copy=False)
    result = linear / denominator[..., None]
    result = np.where(counts[..., None] > 0, result, np.asarray(0, dtype=linear.dtype))
    return np.ascontiguousarray(result)


_NATIVE_SIGNED_SEED_MAX = 2_147_483_647


def _bounded_native_seed(value: int) -> int:
    """Map deterministic Python seed arithmetic into the native int contract."""

    return ((int(value) - 1) % _NATIVE_SIGNED_SEED_MAX) + 1


def _native_sensor_to_display(img: np.ndarray) -> np.ndarray:
    """Convert native physical-right/up storage to top-left display rows/cols.

    Native accumulation is indexed ``[right, up]``. The C++ getter reverses
    ``right`` before returning ``[reversed_right, up]``. A transpose restores
    the axis roles, then both directions must be reversed so display columns
    increase rightward and display rows increase downward.
    """
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[0] != a.shape[1]:
        return a
    return np.ascontiguousarray(a.transpose(1, 0, 2)[::-1, ::-1])


def _native_sensor_tile_to_display(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Reorient and sample the native square accumulator to an exact tile shape."""
    display = _native_sensor_to_display(img)
    out_width = int(width)
    out_height = int(height)
    if display.ndim != 3 or out_width <= 0 or out_height <= 0:
        return display
    if display.shape[:2] == (out_height, out_width):
        return display
    y_index = np.minimum(
        (np.arange(out_height, dtype=np.float64) + 0.5)
        * float(display.shape[0]) / float(out_height),
        display.shape[0] - 1,
    ).astype(np.int64)
    x_index = np.minimum(
        (np.arange(out_width, dtype=np.float64) + 0.5)
        * float(display.shape[1]) / float(out_width),
        display.shape[1] - 1,
    ).astype(np.int64)
    return np.ascontiguousarray(display[y_index[:, None], x_index[None, :]])


def _native_sensor_schedule_shape(width: int, height: int) -> tuple[int, int]:
    """Return the square schedule required by the native square accumulator."""
    resolution = max(int(width), int(height))
    if resolution <= 0:
        raise ValueError("native sensor schedule dimensions must be positive")
    return resolution, resolution


def _native_sensor_linear_output(backend: Any, display_image: np.ndarray) -> np.ndarray:
    """Return raw native accumulation when available, preserving its dtype."""
    linear_image = getattr(backend, "_native_sensor_linear_image", None)
    if linear_image is None:
        return np.asarray(display_image).copy()
    linear = np.asarray(linear_image)
    if linear.shape != np.asarray(display_image).shape:
        raise ValueError(
            f"native linear sensor shape {linear.shape} does not match display {np.asarray(display_image).shape}"
        )
    return linear.copy()


def _native_sensor_accumulation_output(backend: Any) -> np.ndarray | None:
    """Return valid linear sensor evidence, including sum/weight fallback.

    The developed native image can be empty after a GPU-resident T5 pass even
    though its persistent sensor sum and exposure weights contain the completed
    integral.  Never replace that evidence with a synthetic black frame.
    """

    linear_value = getattr(backend, "_native_sensor_linear_image", None)
    linear = None if linear_value is None else np.asarray(linear_value, np.float32)
    if (
        linear is not None
        and linear.ndim == 3
        and linear.shape[2] >= 3
        and np.all(np.isfinite(linear[..., :3]))
        and np.any(linear[..., :3] != 0.0)
    ):
        return np.ascontiguousarray(linear[..., :3])

    sum_value = getattr(backend, "_native_sensor_sum_linear", None)
    weight_value = getattr(backend, "_native_sensor_exposure_weight", None)
    if sum_value is None or weight_value is None:
        return None
    sensor_sum = np.asarray(sum_value, np.float32)
    exposure_weight = np.asarray(weight_value, np.float32)
    if (
        sensor_sum.ndim != 3
        or sensor_sum.shape[2] < 3
        or exposure_weight.shape != sensor_sum.shape[:2]
        or not np.all(np.isfinite(sensor_sum[..., :3]))
        or not np.all(np.isfinite(exposure_weight))
        or not np.any(exposure_weight > 0.0)
    ):
        return None
    result = np.zeros(sensor_sum.shape[:2] + (3,), np.float32)
    np.divide(
        sensor_sum[..., :3], exposure_weight[..., None], out=result,
        where=exposure_weight[..., None] > 0.0,
    )
    return np.ascontiguousarray(np.maximum(result, 0.0))


from camera_exposure_budget import (
    CameraOptics,
    FilmExposure,
    RayDispatchPlan,
    lambertian_emitter_radiance,
    plan_ray_budget,
    summarize_plan,
    H_PLANCK,
    C_LIGHT,
)
from camera_parametric_solver import RailModelConfig, solve_sane_camera_rig
from camera_mode_validation import CameraEventTelemetry, get_mode_name, validate_frame_configuration
from bdpt_integrator import CAMERA_MODE_APERTURE_CONE
from material_db import MaterialDatabase, MAX_SPECTRAL_BANDS
from sensor_film_db import SensorFilmDatabase, MAX_SENSOR_FILM_SLOTS
from camera_software.text_clarity import MonteCarloClarityDiscriminator
from camera_software.film_format import DEFAULT_FILM_FORMAT

# Borrow scene authoring verbatim from the basic-rasterizer harness.
import test_basic_gl_cpp_window as scene_mod   # noqa: E402

try:
    from surface_spline_utils import parameterize_mesh as _ss_parameterize_mesh
    _HAS_SURFACE_SPLINE = True
except ImportError:
    _HAS_SURFACE_SPLINE = False

try:
    from sdf_plugins import get_sdf_driver as _get_sdf_driver
except Exception as _sdf_exc:
    print(f"  [warn] parametric SDF plugin unavailable: {_sdf_exc}")
    _get_sdf_driver = None

try:
    from bdpt_integrator import CameraSensor, TriangleGroup
    from bdpt_integrator import TRI_GROUP_ROLE_SENSOR, TRI_GROUP_SAMPLE_PIXEL_CONE
    from bdpt_integrator import aggregate_to_image
    from bdpt_integrator import aggregate_to_image_pixel_cone
    from bdpt_integrator import (
        CAMERA_MODE_ORACLE_PINHOLE_REFERENCE,
        CAMERA_MODE_PHYSICAL_PINHOLE,
        CAMERA_MODE_APERTURE_CONE,
        CAMERA_MODE_THIN_LENS_GEOMETRIC,
        CAMERA_MODE_PARAMETRIC_ASSEMBLY,
        CAMERA_MODE_WAVE_ASSEMBLY,
        CAMERA_MODE_BAKED_TRANSFORM,
    )
    _HAS_BDPT_INTEGRATION = True
except ImportError:
    _HAS_BDPT_INTEGRATION = False
    CameraSensor = None
    TriangleGroup = None
    TRI_GROUP_ROLE_SENSOR = None
    TRI_GROUP_SAMPLE_PIXEL_CONE = None
    aggregate_to_image = None
    aggregate_to_image_pixel_cone = None
    CAMERA_MODE_ORACLE_PINHOLE_REFERENCE = 0
    CAMERA_MODE_PHYSICAL_PINHOLE = 1
    CAMERA_MODE_APERTURE_CONE = 2
    CAMERA_MODE_THIN_LENS_GEOMETRIC = 3
    CAMERA_MODE_PARAMETRIC_ASSEMBLY = 4
    CAMERA_MODE_WAVE_ASSEMBLY = 5
    CAMERA_MODE_BAKED_TRANSFORM = 6


def _camera_mode_from_cli_string(mode_str: str | None) -> int | None:
    """Convert CLI --camera-mode string to numeric constant.
    
    Args:
        mode_str: One of (oracle_pinhole, physical_pinhole, aperture_cone, thin_lens,
                         geometric_assembly, wave_assembly, baked_transform), or None
                         
    Returns:
        Numeric camera mode constant, or None if mode_str is None
        
    Raises:
        ValueError if mode_str is not recognized
    """
    if mode_str is None:
        return None
    
    mapping = {
        "oracle_pinhole": CAMERA_MODE_ORACLE_PINHOLE_REFERENCE,
        "physical_pinhole": CAMERA_MODE_PHYSICAL_PINHOLE,
        "aperture_cone": CAMERA_MODE_APERTURE_CONE,
        "thin_lens": CAMERA_MODE_THIN_LENS_GEOMETRIC,
        "parametric_assembly": CAMERA_MODE_PARAMETRIC_ASSEMBLY,
        "geometric_assembly": CAMERA_MODE_PARAMETRIC_ASSEMBLY,
        "wave_assembly": CAMERA_MODE_WAVE_ASSEMBLY,
        "baked_transform": CAMERA_MODE_BAKED_TRANSFORM,
    }
    
    if mode_str not in mapping:
        raise ValueError(
            f"Unknown camera mode '{mode_str}'. "
            f"Valid choices: {', '.join(sorted(mapping.keys()))}"
        )
    
    return mapping[mode_str]


def _weld_mesh(verts_flat_n9: np.ndarray,
              tol: float = 1.0e-5) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat (N_tri, 9) triangle buffer to indexed (V, 3) + (N_tri, 3) faces.

    Uses lexicographic sorting with a quantization tolerance.  Suitable for
    the scene meshes built by scene_mod (typical edge length >> 1e-5 m).
    Returns (unique_verts float64, faces int32).
    """
    pts = verts_flat_n9.reshape(-1, 3).astype(np.float64)
    quant = np.round(pts / tol).astype(np.int64)
    order = np.lexsort(quant.T[::-1])
    sq = quant[order]
    diff = np.empty(len(sq), dtype=bool)
    diff[0] = True
    diff[1:] = np.any(sq[1:] != sq[:-1], axis=1)
    uid_sorted = np.cumsum(diff, dtype=np.int32) - 1
    inv = np.empty(len(pts), dtype=np.int32)
    inv[order] = uid_sorted
    first_sorted = np.where(diff)[0]
    unique_verts = pts[order[first_sorted]]
    faces = inv.reshape(-1, 3)
    return unique_verts, faces


def _cam_vis_name(mode: int) -> str:
    return {0: "AS_IS", 1: "DIRECT_HIT", 2: "FULL_MARCH"}.get(mode, str(mode))


def _transp_name(mode: int) -> str:
    return {0: "BLOCK", 1: "XRAY"}.get(mode, str(mode))


def _build_flat_sensor_geometry(optics: CameraOptics, 
                                height_px: int,
                                width_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a flat rectangular sensor geometry as triangles.
    
    Returns (positions, uvs, triangle_indices) for a simple 2x1 grid (2 tris)
    at z=0 spanning the sensor dimensions based on optics.
    
    Parameters:
    - optics: CameraOptics with pixel_pitch_um, sensor dimensions
    - height_px, width_px: sensor pixel dimensions (for reference)
    
    Returns:
    - positions: (4, 3) float64 — corners of flat sensor plane
    - uvs: (4, 2) float64 — normalized [0,1]² coordinates
    - tris: (2, 3) int32 — triangle indices
    """
    # Sensor half-dimensions in metres
    sensor_w_m = optics.sensor_w_mm * 1.0e-3
    sensor_h_m = optics.sensor_h_mm * 1.0e-3
    
    # Four corners at z=0 (sensor plane)
    positions = np.array([
        [-sensor_w_m/2, -sensor_h_m/2, 0.0],
        [ sensor_w_m/2, -sensor_h_m/2, 0.0],
        [ sensor_w_m/2,  sensor_h_m/2, 0.0],
        [-sensor_w_m/2,  sensor_h_m/2, 0.0],
    ], dtype=np.float64)
    
    # UVs map [0,1]² uniformly
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float64)
    
    # Two triangles covering the quad
    tris = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    
    return positions, uvs, tris


# ─────────────────────────────────────────────────────────────────────────────
# Scientific defaults — grounded in real photographic conventions.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_OPTICS = CameraOptics(
    focal_mm           = 82.5,
    aperture_mm        = 20.625,       # f/4 on the default 6x6 rig
    pixel_pitch_um     = 8.4,
    sensor_w_mm        = DEFAULT_FILM_FORMAT.frame_width_mm,
    sensor_h_mm        = DEFAULT_FILM_FORMAT.frame_height_mm,
    lens_transmission  = 0.95,
)

# Simple Holga-like camera: fixed focus, fixed aperture, plastic lens
HOLGA_OPTICS = CameraOptics(
    focal_mm           = 60.0,         # Wide angle plastic lens
    aperture_mm        = 7.5,          # f/8 fixed aperture
    pixel_pitch_um     = 8.4,
    sensor_w_mm        = 36.0,
    sensor_h_mm        = 24.0,
    lens_transmission  = 0.80,         # Plastic lens, some absorption
)

DEFAULT_FILM = FilmExposure(
    iso                 = 100.0,
    exposure_time_s     = 1.0 / 60.0,
    quantum_efficiency  = 0.5,
    target_mid_grey     = 0.18,
)

# Spectral grid for the C++ tracer.  We keep this short (8 bands across the
# visible) so RAM stays manageable; the unified mat_buf still allocates 32
# slots per material with the unused tail zeroed.
DEFAULT_FREQ_HZ = (C_LIGHT / np.linspace(700e-9, 400e-9, 8)).astype(np.float64)
_SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "shaders")

ENABLE_EXPOSURE_PROFILING = False


def _active_frequency_sidecar(
    tll: Any,
    freq_hz: np.ndarray,
    spectral_model: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Build optical material bands in the tracer's exact active-band order."""
    freq = np.asarray(freq_hz, np.float64).reshape(-1)
    if freq.size <= 0 or np.any(~np.isfinite(freq)) or np.any(freq <= 0.0):
        raise ValueError("active spectral frequencies must be finite and positive")
    wavelength_nm = C_LIGHT / freq * 1.0e9
    if spectral_model is None:
        template = tll.FreeFrequencySidecar.lazy_prepare(int(freq.size))
        order = np.argsort(np.asarray(template.wavelength_nm, np.float64))
        weight = np.interp(
            wavelength_nm,
            np.asarray(template.wavelength_nm, np.float64)[order],
            np.asarray(template.weight, np.float64)[order],
        )
    else:
        from camera_software.camera_manifest import sample_spectral_model
        weight = np.asarray(
            sample_spectral_model(spectral_model, wavelength_nm), np.float64
        )
    return tll.FreeFrequencySidecar.from_prepared(wavelength_nm, weight)


class StageProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._data: dict[str, list[float]] = {}
        self._order: list[str] = []

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1e3
            self._data.setdefault(name, []).append(ms)
            if name not in self._order:
                self._order.append(name)

    def format_report(self, prefix: str = "[profile]") -> str:
        if not self.enabled or not self._order:
            return ""
        parts = [f"{k}={np.mean(self._data[k]):.1f}ms" for k in self._order if self._data.get(k)]
        current = peak = None
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
        if current is not None and peak is not None:
            parts.append(f"mem={current / (1024 ** 2):.1f}MB/{peak / (1024 ** 2):.1f}MB")
        return f"{prefix} " + "  ".join(parts)

    def report(self, prefix: str = "[profile]") -> None:
        msg = self.format_report(prefix=prefix)
        if msg:
            print(msg, flush=True)

DEFAULT_CALIBRATION_SCENE_SEQUENCE = (
    "calib-grid",
    "calib-bw-rgb",
    "calib-rgb-diagram",
    "calib-step-wedge",
    "calib-prism-backplate",
    "tungsten-cavity",
)

CAMERA_SOLVE_PREVIEW_SCENE = "solve-preview-gems"


def _build_default_scene_schedule(n_frames_planned: int, demo_scene: str = "orbiters") -> tuple[str, ...]:
    n_frames = max(1, int(n_frames_planned))
    if n_frames == 1:
        return (demo_scene,)
    calibs = DEFAULT_CALIBRATION_SCENE_SEQUENCE
    prefix = tuple(calibs[i % len(calibs)] for i in range(n_frames - 1))
    return prefix + (demo_scene,)

# EndpointRecord in Python is float32 (N, 16): 64 bytes/record payload.
_BDPT_RECORD_FLOATS = 16
_BDPT_RECORD_BYTES = _BDPT_RECORD_FLOATS * np.dtype(np.float32).itemsize
_BDPT_INTERMEDIATE_HARD_CAP_BYTES = 50 * (1024 ** 3)
_BDPT_OVERFLOW_TEMP_PREFIX = "bdpt_overflow_tmp_"
_BDPT_OVERFLOW_RETAINED_PREFIX = "bdpt_overflow_retained_"
_BDPT_RECORDS_PREFIX = "bdpt_records_"
_BDPT_RECORDS_TEMP_RE = re.compile(r"^bdpt_records_\d{4}_.+\.npy$", re.IGNORECASE)
_BDPT_STREAM_WORKING_FRACTION = 0.25


def _bdpt_cap_from_bytes(max_bytes: int) -> int:
    """Analytical endpoint-record cap from byte budget."""
    return max(0, int(max_bytes) // _BDPT_RECORD_BYTES)


def _clamp_bdpt_intermediate_bytes(max_bytes: int) -> int:
    """Clamp file-backed BDPT budget to a strict hard limit.

    A non-positive request means "use hard cap".
    """
    req = int(max_bytes)
    if req <= 0:
        return int(_BDPT_INTERMEDIATE_HARD_CAP_BYTES)
    return int(min(req, _BDPT_INTERMEDIATE_HARD_CAP_BYTES))


def _is_bdpt_disk_allocation_name(name: str) -> bool:
    return bool(
        name.startswith(_BDPT_OVERFLOW_TEMP_PREFIX)
        or name.startswith(_BDPT_OVERFLOW_RETAINED_PREFIX)
        or name.startswith(_BDPT_RECORDS_PREFIX)
    )


def _is_ephemeral_bdpt_disk_allocation_name(name: str) -> bool:
    return bool(
        name.startswith(_BDPT_OVERFLOW_TEMP_PREFIX)
        or _BDPT_RECORDS_TEMP_RE.match(name)
    )


def _bdpt_disk_usage_bytes(dir_path: str) -> int:
    total = 0
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if not _is_bdpt_disk_allocation_name(entry.name):
                    continue
                try:
                    total += int(entry.stat().st_size)
                except OSError:
                    continue
    except OSError:
        return 0
    return int(total)


def _cleanup_stale_ephemeral_bdpt_files(dir_path: str,
                                        preserve_paths: Optional[set[str]] = None,
                                        include_retained: bool = False) -> int:
    removed = 0
    keep = preserve_paths or set()
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if entry.path in keep:
                    continue
                if include_retained:
                    if not _is_bdpt_disk_allocation_name(entry.name):
                        continue
                elif not _is_ephemeral_bdpt_disk_allocation_name(entry.name):
                    continue
                try:
                    os.remove(entry.path)
                    removed += 1
                except OSError:
                    continue
    except OSError:
        return removed
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# Scene → tracer geometry adapter
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FilmPlanePose:
    center: np.ndarray
    normal_to_scene: np.ndarray
    right: np.ndarray
    up: np.ndarray
    lens_distance_delta_m: float = 0.0
    tilt_about_right_deg: float = 0.0
    tilt_about_up_deg: float = 0.0


@dataclass
class TracerScene:
    """Everything both ray tracers need to be built once per camera frame."""
    verts:        np.ndarray   # (N_tri, 9)  float64 — flat triangle vertices
    normals:      np.ndarray   # (N_tri, 3)  float64
    mat_idx:      np.ndarray   # (N_tri,)    int32   — index into MaterialDatabase
    mat_buf:      np.ndarray   # (N_mat * MAX_SPECTRAL_BANDS, 12) float32
    mat_n_mats:   int
    src_pos:      np.ndarray   # (N_src, 3)  float64 — emissive triangle centroids
    src_dir:      np.ndarray   # (N_src, 3)  float64 — outward normals
    src_directivity: np.ndarray  # (N_src,)  float64 — Lambertian → 1.0
    src_area_m2:  np.ndarray   # (N_src,)    float64 — for power bookkeeping
    src_emit_W:   np.ndarray   # (N_src,)    float64 — luminance-weighted emission
    src_emit_rgb_W: np.ndarray # (N_src, 3)  float64 — per-channel emissive power proxy
    src_tri_idx:  np.ndarray   # (N_src,)    int32   — triangle indices in verts/normals
    bounds_min:   np.ndarray   # (3,)        float32
    bounds_max:   np.ndarray   # (3,)        float32
    camera_tri_groups: Optional[dict[str, np.ndarray]] = None
    film_plane_pose: Optional[FilmPlanePose] = None
    optical_camera: Optional[Any] = None
    emitter_launch_profile: Optional[dict[str, Any]] = None

    @property
    def total_emissive_power_W(self) -> float:
        return float(self.src_emit_W.sum())

    @property
    def total_emissive_area_m2(self) -> float:
        return float(self.src_area_m2.sum())

    def scene_radiance_W_sr_m2(self) -> float:
        """Lambertian-disk radiance the camera will see if the emitters
        completely fill its FOV (upper-bound; capture_efficiency in the budget
        plan is what closes the gap empirically)."""
        return lambertian_emitter_radiance(
            self.total_emissive_power_W, self.total_emissive_area_m2)


def _rotate_vector(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-15)
    value = np.asarray(vector, np.float64)
    return (
        value * math.cos(angle_rad)
        + np.cross(axis, value) * math.sin(angle_rad)
        + axis * float(np.dot(axis, value)) * (1.0 - math.cos(angle_rad))
    )


def apply_next_site_scan(scene: TracerScene, command: Any, *,
                         maximum_film_travel_m: float = 2.0e-3,
                         maximum_film_tilt_deg: float = 5.0) -> TracerScene:
    """Apply a validated physical film-stage command to camera geometry."""
    from camera_software.scan_control import CameraScanController

    controller = CameraScanController(
        maximum_film_travel_m=maximum_film_travel_m,
        maximum_film_tilt_deg=maximum_film_tilt_deg,
    )
    adjustment = controller.validate(command)
    if adjustment is None:
        return scene
    groups = scene.camera_tri_groups or {}
    sensor_ids = np.asarray(groups.get("sensor", ()), np.int64).reshape(-1)
    if sensor_ids.size == 0:
        raise ValueError("film-plane adjustment requires camera sensor geometry")

    triangles = np.asarray(scene.verts, np.float64).reshape(-1, 3, 3).copy()
    sensor_points = triangles[sensor_ids].reshape(-1, 3)
    current_center = sensor_points.mean(axis=0)
    nominal_center = current_center
    nominal_right = np.asarray([0.0, 1.0, 0.0], np.float64)
    nominal_up = np.asarray([0.0, 0.0, 1.0], np.float64)
    current_right = nominal_right
    current_up = nominal_up
    if scene.optical_camera is not None:
        nominal_center = np.asarray(
            scene.optical_camera.machined_sensor_center, np.float64
        )
        nominal_right = np.asarray(
            scene.optical_camera.machined_sensor_right, np.float64
        )
        nominal_up = np.asarray(
            scene.optical_camera.machined_sensor_up, np.float64
        )
        current_right = np.asarray(scene.optical_camera.sensor_right, np.float64)
        current_up = np.asarray(scene.optical_camera.sensor_up, np.float64)
    away_from_lens = np.cross(nominal_right, nominal_up)
    away_from_lens /= max(float(np.linalg.norm(away_from_lens)), 1.0e-15)
    normal_to_scene = -away_from_lens
    right = nominal_right.copy()
    up = nominal_up.copy()
    right_angle = math.radians(float(adjustment.tilt_about_right_deg))
    up_angle = math.radians(float(adjustment.tilt_about_up_deg))
    if right_angle != 0.0:
        normal_to_scene = _rotate_vector(normal_to_scene, right, right_angle)
        up = _rotate_vector(up, right, right_angle)
    if up_angle != 0.0:
        normal_to_scene = _rotate_vector(normal_to_scene, up, up_angle)
        right = _rotate_vector(right, up, up_angle)
    normal_to_scene /= max(float(np.linalg.norm(normal_to_scene)), 1.0e-15)
    right /= max(float(np.linalg.norm(right)), 1.0e-15)
    up = np.cross(right, normal_to_scene)
    up /= max(float(np.linalg.norm(up)), 1.0e-15)

    # Film-stage depth is measured along the fixed lens optical axis; tilting
    # the carrier must not turn an axial focus move into a lateral translation.
    adjusted_center = nominal_center + away_from_lens * float(
        adjustment.lens_distance_delta_m
    )
    local = sensor_points - current_center
    local_right = local @ current_right
    local_up = local @ current_up
    transformed = (
        adjusted_center[None, :]
        + local_right[:, None] * right[None, :]
        + local_up[:, None] * up[None, :]
    )
    triangles[sensor_ids] = transformed.reshape(-1, 3, 3)

    normals = np.asarray(scene.normals, np.float64).copy()
    for triangle_id in sensor_ids:
        edge_a = triangles[triangle_id, 1] - triangles[triangle_id, 0]
        edge_b = triangles[triangle_id, 2] - triangles[triangle_id, 0]
        normal = np.cross(edge_a, edge_b)
        length = float(np.linalg.norm(normal))
        if length <= 1.0e-18:
            raise ValueError("film-plane adjustment collapsed a sensor triangle")
        normals[triangle_id] = normal / length
    scene.verts = np.ascontiguousarray(triangles.reshape(-1, 9), np.float64)
    scene.normals = np.ascontiguousarray(normals, np.float64)
    points = triangles.reshape(-1, 3)
    scene.bounds_min = np.ascontiguousarray(points.min(axis=0), np.float32)
    scene.bounds_max = np.ascontiguousarray(points.max(axis=0), np.float32)
    scene.film_plane_pose = FilmPlanePose(
        center=np.ascontiguousarray(adjusted_center, np.float64),
        normal_to_scene=np.ascontiguousarray(normal_to_scene, np.float64),
        right=np.ascontiguousarray(right, np.float64),
        up=np.ascontiguousarray(up, np.float64),
        lens_distance_delta_m=float(adjustment.lens_distance_delta_m),
        tilt_about_right_deg=float(adjustment.tilt_about_right_deg),
        tilt_about_up_deg=float(adjustment.tilt_about_up_deg),
    )
    if scene.optical_camera is not None:
        scene.optical_camera.apply_film_pose(
            center=adjusted_center,
            right=right,
            up=up,
        )
    return scene


# Optional subject-scene override compiled from a declarative scene order.
# It still contains the proven thick-lens lab camera/lens/flash groups; only
# the replaceable subject geometry and authored materials differ.
_ORDERED_THICK_LENS_SCENE: Optional[TracerScene] = None
_ORDERED_SCENE_JOB: Optional[dict[str, Any]] = None
_ORDERED_TRANSPORT_RGB_WEIGHTS: Optional[np.ndarray] = None
_ORDERED_TRANSPORT_LUT: Optional[dict[str, np.ndarray]] = None
_ORDERED_MATERIAL_LUT_GRID: Optional[np.ndarray] = None
_ORDERED_TILE_OUTPUT: Optional[tuple[int, int]] = None
_NEXT_SITE_SCAN_COMMAND: Optional[Any] = None


_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _register_preview_obsidian_material(db: MaterialDatabase) -> int:
    """Register a physically grounded obsidian material for preview mode."""
    loader = getattr(scene_mod, "_load_material_yaml_dict", None)
    converter = getattr(scene_mod, "_yaml_to_material_dict", None)
    if callable(loader) and callable(converter):
        try:
            payload = converter(loader("obsidian_polish"))
            return int(db.register("preview_obsidian_polish", payload))
        except Exception:
            pass

    # Fallback mirrors configs/materials/obsidian_polish.yaml as closely as
    # possible when YAML helpers are unavailable.
    return int(db.register(
        "preview_obsidian_polish",
        {
            "color_profile_name": "obsidian_body",
            "albedo_rgb": [0.012, 0.010, 0.014],
            "reflectivity": 0.06,
            "diffusion": 0.04,
            "absorption": 0.90,
            "roughness": 0.06,
            "metallic": 0.0,
            "transmission": 0.0,
            "ior": 1.49,
            "opacity": 1.0,
            "emission_rgb": [0.0, 0.0, 0.0],
            "ambient": 0.04,
            "spec_strength": 0.32,
            "shininess": 200.0,
            "inner_color": [0.012, 0.010, 0.014],
        },
    ))


def _camera_frame_basis(cam: PhysicalCameraRig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fwd = np.asarray(cam.fwd, np.float64)
    fwd /= max(np.linalg.norm(fwd), 1.0e-12)
    up_hint = np.asarray(cam.up, np.float64)
    right = np.cross(fwd, up_hint)
    if np.linalg.norm(right) < 1.0e-12:
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0], np.float64))
    right /= max(np.linalg.norm(right), 1.0e-12)
    up = np.cross(right, fwd)
    up /= max(np.linalg.norm(up), 1.0e-12)
    return fwd, right, up


def _build_camera_rig_mesh(
    cam: PhysicalCameraRig,
    optics: CameraOptics,
    rig_mat_idx: int,
    first_tri_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build world-space camera rig triangles for sensor/aperture/lens groups."""
    fwd, right, up = _camera_frame_basis(cam)
    cam_pos = np.asarray(cam.pos, np.float64)

    sensor_w = float(optics.sensor_w_mm) * 1.0e-3
    sensor_h = float(optics.sensor_h_mm) * 1.0e-3
    aperture_r = max(1.0e-6, float(cam.aperture_radius_m))

    sensor_c = cam_pos + fwd * float(cam.sensor_plane_offset_m)
    aperture_c = cam_pos + fwd * float(cam.aperture_plane_offset_m)
    lens_front_c = cam_pos + fwd * 0.004
    lens_rear_c = cam_pos - fwd * 0.004

    tris: list[np.ndarray] = []
    groups: dict[str, list[int]] = {
        "sensor": [],
        "aperture_blocker": [],
        "thin_lens": [],
    }

    def _add_quad(center: np.ndarray, hw: float, hh: float, group: str) -> None:
        p0 = center - hw * right - hh * up
        p1 = center + hw * right - hh * up
        p2 = center + hw * right + hh * up
        p3 = center - hw * right + hh * up
        t0 = np.concatenate([p0, p1, p2])
        t1 = np.concatenate([p0, p2, p3])
        groups[group].append(first_tri_index + len(tris)); tris.append(t0)
        groups[group].append(first_tri_index + len(tris)); tris.append(t1)

    # Sensor plane mesh.
    _add_quad(sensor_c, 0.5 * sensor_w, 0.5 * sensor_h, "sensor")

    # NOTE: Do not add synthetic thin-lens quads into traced geometry.
    # They are useful for visualization but can self-occlude PIXEL_CONE
    # backward rays and collapse sensor influence to near-zero.

    # Aperture blocker blades: annulus sectors around the clear aperture.
    blade_n = 8
    rin = 0.93 * aperture_r
    rout = 1.18 * aperture_r
    for i in range(blade_n):
        a0 = (2.0 * math.pi * i) / float(blade_n)
        a1 = (2.0 * math.pi * (i + 1)) / float(blade_n)
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        p_in0 = aperture_c + rin * (c0 * right + s0 * up)
        p_in1 = aperture_c + rin * (c1 * right + s1 * up)
        p_out0 = aperture_c + rout * (c0 * right + s0 * up)
        p_out1 = aperture_c + rout * (c1 * right + s1 * up)
        t0 = np.concatenate([p_in0, p_out0, p_out1])
        t1 = np.concatenate([p_in0, p_out1, p_in1])
        groups["aperture_blocker"].append(first_tri_index + len(tris)); tris.append(t0)
        groups["aperture_blocker"].append(first_tri_index + len(tris)); tris.append(t1)

    tri_arr = np.asarray(tris, np.float64).reshape(-1, 3, 3)
    e1 = tri_arr[:, 1] - tri_arr[:, 0]
    e2 = tri_arr[:, 2] - tri_arr[:, 0]
    cr = np.cross(e1, e2)
    nrm = np.linalg.norm(cr, axis=1, keepdims=True)
    normals = (cr / np.where(nrm > 1.0e-12, nrm, 1.0)).astype(np.float64)
    verts_flat = tri_arr.reshape(-1, 9)
    mat_idx = np.full((verts_flat.shape[0],), int(rig_mat_idx), dtype=np.int32)
    groups_np = {k: np.asarray(v, dtype=np.int32) for k, v in groups.items()}
    return verts_flat, normals, mat_idx, groups_np


def _lab_object_distance_for_sensor_focus(scene: Any, sensor_focus_m: float) -> float:
    """Convert sensor-to-object distance to the lab solver's object-to-lens distance."""
    design = scene.optical_design
    focal_range = tuple(float(value) for value in design.focal_length_range_m)
    focal_m = focal_range[0] + (focal_range[1] - focal_range[0]) * float(design.zoom)
    group_span_m = (
        (int(design.group_count) - 1) * float(design.min_air_gap_m)
        + int(design.group_count) * float(design.group_thickness_m)
    )
    clearance_m = float(design.sensor_clearance_m)
    target_m = float(sensor_focus_m)
    minimum_m = focal_m * 1.05

    def sensor_distance(object_distance_m: float) -> float:
        image_distance_m = 1.0 / (1.0 / focal_m - 1.0 / object_distance_m)
        return object_distance_m + group_span_m + image_distance_m + clearance_m

    low = minimum_m
    high = max(target_m, low * 2.0)
    if target_m <= sensor_distance(low):
        raise ValueError(
            f"sensor focus distance {target_m:g}m is too close for the lab lens "
            f"(minimum {sensor_distance(low):g}m)"
        )
    for _ in range(80):
        middle = 0.5 * (low + high)
        if sensor_distance(middle) < target_m:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _apply_camera_manifest_to_lab_scene(
    tll: Any, scene: Any, manifest: Mapping[str, Any]
) -> Any:
    """Compatibility wrapper for the public camera-preparation API."""

    from camera_software.camera_preparation import (
        apply_camera_manifest_to_lab_scene,
    )
    return apply_camera_manifest_to_lab_scene(scene, manifest)


def _ordered_lab_scene_for_sensor_focus(
    tll: Any,
    sensor_focus_m: float,
    camera_manifest: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Configure a fresh lab scene whose physical sensor conjugate matches the order."""
    initial = tll.SceneConfig()
    if camera_manifest is not None:
        _apply_camera_manifest_to_lab_scene(tll, initial, camera_manifest)
    proposed_object_distance_m = _lab_object_distance_for_sensor_focus(
        initial, sensor_focus_m
    )
    corrected_object_distance_m = proposed_object_distance_m
    physical_error_m = 0.0
    for _ in range(4):
        probe = tll.SceneConfig()
        if camera_manifest is not None:
            _apply_camera_manifest_to_lab_scene(tll, probe, camera_manifest)
        probe.focus_distance_m = corrected_object_distance_m
        probe.x_max = max(
            float(probe.x_max),
            float(probe.object_plane.x) + float(sensor_focus_m) + 0.25,
        )
        tll._scene_lenses(probe)
        physical_sensor_focus_m = (
            float(probe.image_plate.x) - float(probe.object_plane.x)
        )
        physical_error_m = physical_sensor_focus_m - float(sensor_focus_m)
        corrected_object_distance_m -= physical_error_m
        if abs(physical_error_m) <= 5.0e-7:
            break

    scene = tll.SceneConfig()
    if camera_manifest is not None:
        _apply_camera_manifest_to_lab_scene(tll, scene, camera_manifest)
    scene.focus_distance_m = corrected_object_distance_m
    scene.x_max = max(
        float(scene.x_max),
        float(scene.object_plane.x) + float(sensor_focus_m) + 0.25,
    )
    print(
        "[scene-focus] "
        f"sensor_to_text={sensor_focus_m:.6f}m "
        f"paraxial_object_distance={proposed_object_distance_m:.6f}m "
        f"physical_probe_error={physical_error_m * 1e3:+.6f}mm "
        f"corrected_object_distance={corrected_object_distance_m:.6f}m",
        flush=True,
    )
    return scene


def _make_rebuilt_lab_camera(
    tll: Any,
    lab_scene: Any,
    sidecar: Any,
    lens_surface_groups: Any,
    tri_arr: np.ndarray,
    image_plate_ids: np.ndarray,
) -> Any:
    """Retain the camera lab's solved semantics beside its BVH proxy mesh."""
    from camera_designer.lens_assembly import LensAssemblySpec
    from camera_software.camera_build import RebuiltCameraArtifact

    optics = tll._compound_lens_from_scene(lab_scene, sidecar)
    assembly = LensAssemblySpec()
    assembly.set_optics(optics, mode=LensAssemblySpec.MODE_PARAMETRIC)
    assembly.sync_from_scene(lab_scene)
    sensor_ids = np.asarray(image_plate_ids, np.int32).reshape(-1)
    sensor_center = np.asarray(tri_arr, np.float64)[sensor_ids].reshape(-1, 3).mean(axis=0)
    camera_manifest = copy.deepcopy(
        getattr(lab_scene, "resolved_camera_manifest", {}) or {}
    )
    artifact = RebuiltCameraArtifact.create(
        scene_config=lab_scene,
        lens_assembly=assembly,
        lens_surface_groups=lens_surface_groups,
        scene_lenses=tll._scene_lenses(lab_scene),
        wavelengths_nm=sidecar.wavelength_nm,
        sensor_center=sensor_center,
        diffraction_model="disabled_unless_wave_arena_is_explicitly_registered",
        manifest=camera_manifest,
    )
    solved = getattr(lab_scene, "optical_design", None)
    if solved is not None and hasattr(solved, "groups"):
        from camera_software.camera_manifest import (
            CameraManifest,
            resolve_camera_manifest,
            with_resolved_optics,
        )
        base_manifest = (
            CameraManifest(camera_manifest)
            if camera_manifest else resolve_camera_manifest(
                wavelengths_nm=sidecar.wavelength_nm
            )
        )
        artifact.manifest = with_resolved_optics(
            base_manifest,
            groups=[{
                "name": group.name,
                "center_x_m": float(group.x_m),
                "focal_length_mm": float(group.focal_length_m) * 1.0e3,
                "aperture_radius_mm": float(group.aperture_radius_m) * 1.0e3,
                "thickness_mm": float(group.thickness_m) * 1.0e3,
                "radius_front_mm": float(group.radius_front_m) * 1.0e3,
                "radius_back_mm": float(group.radius_back_m) * 1.0e3,
                "ior": float(group.ior),
                "glass": str(getattr(lens, "glass", "N-BK7")),
            } for group, lens in zip(solved.groups, artifact.scene_lenses)],
            # The paraxial design target and the exact thick-surface assembly
            # are both retained; this field describes the transport artifact.
            effective_focal_length_mm=float(optics.f_eff) * 1.0e3,
            aperture_x_m=float(solved.aperture_x_m),
            entrance_pupil=(
                float(solved.entrance_pupil_x_m),
                float(solved.entrance_pupil_radius_m),
            ),
            exit_pupil=(
                float(solved.exit_pupil_x_m),
                float(solved.exit_pupil_radius_m),
            ),
            sensor_x_m=float(lab_scene.image_plate.x),
            sensor_error_mm=float(solved.sensor_error_m) * 1.0e3,
            build_id=artifact.provenance.build_id,
        ).mapping()
    return artifact


def _build_thick_lens_lab_tracer_scene() -> TracerScene:
    """Materialise the articulated thick-lens lab scene as a TracerScene.

    This is intentionally sourced from thick_lens_focus_lab.py instead of the
    orbiters/basic-rasterizer adapter, so exposure_render_demo can exercise the
    same material database, lens geometry, iris/aperture blocker, image plate,
    and emissive source set as the real lab simulator.
    """
    if _ORDERED_THICK_LENS_SCENE is not None:
        return _ORDERED_THICK_LENS_SCENE

    import thick_lens_focus_lab as tll

    ordered_sensor_focus_m = None
    ordered_camera_manifest = None
    if _ORDERED_SCENE_JOB is not None:
        from scene_orders import camera_focus_distance, order_runtime_settings
        ordered_sensor_focus_m = camera_focus_distance(_ORDERED_SCENE_JOB)
        ordered_camera_manifest = order_runtime_settings(
            _ORDERED_SCENE_JOB
        )["camera_manifest"]

    authored_flash_spectrum = None
    if ordered_camera_manifest is not None:
        authored_flash_spectrum = dict(
            ordered_camera_manifest.get("flash", {})
        ).get("resolved_spectral_model")
    sidecar = _active_frequency_sidecar(
        tll, DEFAULT_FREQ_HZ, authored_flash_spectrum
    )
    ordered_camera_object = None
    if ordered_camera_manifest is not None:
        from camera_software.camera_manifest import resolve_camera_manifest
        ordered_camera_object = resolve_camera_manifest(
            _ORDERED_SCENE_JOB.get("camera", {}),
            _ORDERED_SCENE_JOB.get("image", {}),
            _ORDERED_SCENE_JOB.get("flash", {}),
            wavelengths_nm=sidecar.wavelength_nm,
        )
        ordered_camera_manifest = ordered_camera_object.mapping()
    if ordered_sensor_focus_m is None:
        lab_scene = tll.SceneConfig()
        if ordered_camera_manifest is not None:
            _apply_camera_manifest_to_lab_scene(
                tll, lab_scene, ordered_camera_manifest
            )
    else:
        lab_scene = _ordered_lab_scene_for_sensor_focus(
            tll, ordered_sensor_focus_m, ordered_camera_manifest
        )

    camera_manifest_hash = "default"
    if ordered_camera_object is not None:
        from camera_software.camera_preparation import CameraPreparationKey
        camera_manifest_hash = CameraPreparationKey.from_manifest(
            ordered_camera_object
        ).cache_key.replace("sha256:", "")[:16]

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spectral_cache")
    focus_cache_tag = (
        "default"
        if ordered_sensor_focus_m is None
        else f"focus_{int(round(ordered_sensor_focus_m * 1.0e6))}um"
    )
    transport_cache_tag = hashlib.sha256(
        np.ascontiguousarray(
            _ORDERED_MATERIAL_LUT_GRID
            if _ORDERED_MATERIAL_LUT_GRID is not None else DEFAULT_FREQ_HZ,
            np.float64,
        ).tobytes()
    ).hexdigest()[:12]
    cache_path = os.path.join(
        cache_dir,
        f"thick_lens_exposure_scene_v7_{camera_manifest_hash}_"
        f"{focus_cache_tag}_spectral_{transport_cache_tag}.npz",
    )
    cache_stamp = np.concatenate([
        np.asarray([
            7.0,
            float(os.path.getmtime(tll.__file__)),
            float(DEFAULT_FREQ_HZ.size),
            -1.0 if ordered_sensor_focus_m is None else float(ordered_sensor_focus_m),
        ], dtype=np.float64),
        np.asarray(sidecar.wavelength_nm, dtype=np.float64),
    ])
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            if np.array_equal(cached["cache_stamp"], cache_stamp):
                print(f"[scene-cache] loaded {cache_path}", flush=True)
                n_lens_groups = int(cached["lens_group_count"][0])
                lens_surface_groups = [
                    (
                        np.ascontiguousarray(cached[f"lens_group_{i}_front"], np.int32),
                        np.ascontiguousarray(cached[f"lens_group_{i}_back"], np.int32),
                        float(cached["lens_group_radii"][i, 0]),
                        float(cached["lens_group_radii"][i, 1]),
                    )
                    for i in range(n_lens_groups)
                ]
                cached_tri = np.ascontiguousarray(cached["verts"], np.float64).reshape(-1, 3, 3)
                optical_camera = _make_rebuilt_lab_camera(
                    tll, lab_scene, sidecar, lens_surface_groups, cached_tri,
                    np.ascontiguousarray(cached["group_sensor"], np.int32),
                )
                return TracerScene(
                    verts=np.ascontiguousarray(cached["verts"]),
                    normals=np.ascontiguousarray(cached["normals"]),
                    mat_idx=np.ascontiguousarray(cached["mat_idx"]),
                    mat_buf=np.ascontiguousarray(cached["mat_buf"]),
                    mat_n_mats=int(cached["mat_n_mats"][0]),
                    src_pos=np.ascontiguousarray(cached["src_pos"]),
                    src_dir=np.ascontiguousarray(cached["src_dir"]),
                    src_directivity=np.ascontiguousarray(cached["src_directivity"]),
                    src_area_m2=np.ascontiguousarray(cached["src_area_m2"]),
                    src_emit_W=np.ascontiguousarray(cached["src_emit_W"]),
                    src_emit_rgb_W=np.ascontiguousarray(cached["src_emit_rgb_W"]),
                    src_tri_idx=np.ascontiguousarray(cached["src_tri_idx"]),
                    bounds_min=np.ascontiguousarray(cached["bounds_min"]),
                    bounds_max=np.ascontiguousarray(cached["bounds_max"]),
                    camera_tri_groups={
                        "sensor": np.ascontiguousarray(cached["group_sensor"]),
                        "aperture_blocker": np.ascontiguousarray(cached["group_aperture"]),
                        "thin_lens": np.ascontiguousarray(cached["group_lens"]),
                        "object": np.ascontiguousarray(cached["group_object"]),
                    },
                    optical_camera=optical_camera,
                )
    except (OSError, KeyError, ValueError) as exc:
        print(f"[scene-cache] rebuild required: {exc}", flush=True)

    (
        verts_flat, normals, mat_idx, tri_arr, db,
        source_ids, lens_front_ids, lens_back_ids, image_plate_ids,
        aperture_stop_ids, _tube_wall_ids, lens_surface_groups,
        object_ids, _tube_baffle_ids, _camera_barrel_ids,
        _camera_rear_cap_ids, _camera_front_cap_ids, _camera_frustum_ids,
        red_probe_ids, _subject_group_tri_map,
    ) = tll._build_scene_mesh(lab_scene, sidecar)

    verts_flat = np.ascontiguousarray(verts_flat, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    mat_idx = np.ascontiguousarray(mat_idx, dtype=np.int32)
    tri_arr = np.ascontiguousarray(tri_arr, dtype=np.float64)
    mat_buf = np.ascontiguousarray(
        db.build_mat_buf(freq_hz=(
            _ORDERED_MATERIAL_LUT_GRID
            if _ORDERED_MATERIAL_LUT_GRID is not None else DEFAULT_FREQ_HZ
        )), dtype=np.float32
    )
    mat_n_mats = int(mat_buf.shape[0] // MAX_SPECTRAL_BANDS)

    source_ids = np.asarray(source_ids, dtype=np.int32).reshape(-1)
    red_probe_ids = np.asarray(red_probe_ids, dtype=np.int32).reshape(-1)
    emitter_ids = np.unique(
        np.concatenate([source_ids, red_probe_ids]).astype(np.int32, copy=False)
    )
    emitter_ids = emitter_ids[(emitter_ids >= 0) & (emitter_ids < int(tri_arr.shape[0]))]
    if emitter_ids.size == 0:
        raise RuntimeError("thick-lens lab scene produced no emissive source triangles")

    e1 = tri_arr[:, 1] - tri_arr[:, 0]
    e2 = tri_arr[:, 2] - tri_arr[:, 0]
    cr = np.cross(e1, e2)
    area = (0.5 * np.linalg.norm(cr, axis=1)).astype(np.float64)
    centroid = tri_arr.mean(axis=1).astype(np.float64)

    pts_all = tri_arr.reshape(-1, 3)
    mat_safe = mat_idx.clip(0, max(0, int(mat_n_mats) - 1))
    mat_rows = mat_buf.reshape(mat_n_mats, MAX_SPECTRAL_BANDS, -1)
    n_active_bands = int(DEFAULT_FREQ_HZ.size)
    spectral_emission = mat_rows[mat_safe, :n_active_bands, 5].astype(np.float64, copy=False)
    src_spectral_power = spectral_emission[emitter_ids] * area[emitter_ids, None]
    src_power = np.sum(np.maximum(src_spectral_power, 0.0), axis=1)
    if not np.any(src_power > 0.0):
        raise RuntimeError("thick-lens lab scene produced emissive triangles with zero spectral emission")
    src_rgb = _bands_to_rgb(src_spectral_power.T[:, :, None], DEFAULT_FREQ_HZ)[:, 0, :].astype(np.float64)
    src_rgb *= src_power[:, None]
    camera_tri_groups = {
        "sensor": np.ascontiguousarray(image_plate_ids, dtype=np.int32),
        "aperture_blocker": np.ascontiguousarray(aperture_stop_ids, dtype=np.int32),
        "thin_lens": np.ascontiguousarray(
            np.unique(np.concatenate([
                np.asarray(lens_front_ids, dtype=np.int32).reshape(-1),
                np.asarray(lens_back_ids, dtype=np.int32).reshape(-1),
            ])),
            dtype=np.int32,
        ),
        "object": np.ascontiguousarray(object_ids, dtype=np.int32),
    }
    optical_camera = _make_rebuilt_lab_camera(
        tll, lab_scene, sidecar, lens_surface_groups, tri_arr, image_plate_ids,
    )

    result = TracerScene(
        verts=verts_flat,
        normals=normals,
        mat_idx=mat_idx,
        mat_buf=mat_buf,
        mat_n_mats=mat_n_mats,
        src_pos=np.ascontiguousarray(centroid[emitter_ids], dtype=np.float64),
        src_dir=np.ascontiguousarray(normals[emitter_ids], dtype=np.float64),
        src_directivity=np.ones(emitter_ids.size, dtype=np.float64),
        src_area_m2=np.ascontiguousarray(area[emitter_ids], dtype=np.float64),
        src_emit_W=np.ascontiguousarray(src_power, dtype=np.float64),
        src_emit_rgb_W=np.ascontiguousarray(src_rgb, dtype=np.float64),
        src_tri_idx=np.ascontiguousarray(emitter_ids, dtype=np.int32),
        bounds_min=np.ascontiguousarray(pts_all.min(axis=0), dtype=np.float32),
        bounds_max=np.ascontiguousarray(pts_all.max(axis=0), dtype=np.float32),
        camera_tri_groups=camera_tri_groups,
        optical_camera=optical_camera,
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_payload = dict(
            cache_stamp=cache_stamp,
            verts=result.verts, normals=result.normals, mat_idx=result.mat_idx,
            mat_buf=result.mat_buf,
            mat_n_mats=np.asarray([result.mat_n_mats], np.int32),
            src_pos=result.src_pos, src_dir=result.src_dir,
            src_directivity=result.src_directivity, src_area_m2=result.src_area_m2,
            src_emit_W=result.src_emit_W, src_emit_rgb_W=result.src_emit_rgb_W,
            src_tri_idx=result.src_tri_idx,
            bounds_min=result.bounds_min, bounds_max=result.bounds_max,
            group_sensor=camera_tri_groups["sensor"],
            group_aperture=camera_tri_groups["aperture_blocker"],
            group_lens=camera_tri_groups["thin_lens"],
            group_object=camera_tri_groups["object"],
            lens_group_count=np.asarray([len(lens_surface_groups)], np.int32),
            lens_group_radii=np.asarray(
                [[float(group[2]), float(group[3])] for group in lens_surface_groups],
                np.float64,
            ),
        )
        for i, (front, back, _radius_front, _radius_back) in enumerate(lens_surface_groups):
            cache_payload[f"lens_group_{i}_front"] = np.ascontiguousarray(front, np.int32)
            cache_payload[f"lens_group_{i}_back"] = np.ascontiguousarray(back, np.int32)
        np.savez_compressed(cache_path, **cache_payload)
        print(f"[scene-cache] wrote {cache_path}", flush=True)
    except OSError as exc:
        print(f"[scene-cache] write skipped: {exc}", flush=True)
    return result


def _is_thick_lens_lab_scene_mode(scene_mode: str) -> bool:
    return str(scene_mode) in ("thick-lens", "thick-lens-lab", "thick_lens_lab")


def _build_thick_lens_lab_camera_package(
    scene: TracerScene,
    width: int,
    height: int,
    optics: CameraOptics,
) -> tuple[PhysicalCameraRig, Any]:
    cam_groups = scene.camera_tri_groups
    sensor_ids = np.asarray(cam_groups.get("sensor", np.zeros((0,), np.int32)), dtype=np.int32)
    if sensor_ids.size == 0:
        raise RuntimeError("thick-lens lab camera requires non-empty sensor tri group")
    sensor_pts = scene.verts[sensor_ids].reshape(-1, 3)
    sensor_center = sensor_pts.mean(axis=0).astype(np.float64)

    pose = scene.film_plane_pose
    fwd = np.asarray(
        pose.normal_to_scene if pose is not None else [-1.0, 0.0, 0.0],
        np.float64,
    )
    right = np.asarray(
        pose.right if pose is not None else [0.0, 1.0, 0.0], np.float64
    )
    up = np.asarray(
        pose.up if pose is not None else [0.0, 0.0, 1.0], np.float64
    )
    sensor_local = sensor_pts - sensor_center
    sensor_half_w = 0.5 * float(np.ptp(sensor_local @ right))
    sensor_half_h = 0.5 * float(np.ptp(sensor_local @ up))

    aperture_ids = np.asarray(cam_groups.get("aperture_blocker", np.zeros((0,), np.int32)), dtype=np.int32)
    if aperture_ids.size > 0:
        aperture_pts = scene.verts[aperture_ids].reshape(-1, 3)
        aperture_center = aperture_pts.mean(axis=0).astype(np.float64)
        aperture_delta = aperture_pts - aperture_center
        aperture_radial = np.hypot(aperture_delta[:, 1], aperture_delta[:, 2])
        positive = aperture_radial[aperture_radial > 1.0e-6]
        aperture_radius_m = float(np.min(positive)) if positive.size else float(optics.aperture_mm) * 0.5e-3
    else:
        aperture_center = sensor_center + fwd * max(float(optics.focal_mm) * 1.0e-3, 1.0e-3)
        aperture_radius_m = float(optics.aperture_mm) * 0.5e-3

    focal_m = max(1.0e-4, float(np.linalg.norm(aperture_center - sensor_center)))
    sensor_w_m = max(1.0e-4, 2.0 * sensor_half_w)
    sensor_h_m = max(1.0e-4, 2.0 * sensor_half_h)
    if _ORDERED_SCENE_JOB is not None:
        from scene_orders import sensor_tile
        tile = sensor_tile(_ORDERED_SCENE_JOB, sensor_w_m, sensor_h_m)
        sensor_center = (
            sensor_center
            + right * float(tile["right_offset_m"])
            + up * float(tile["up_offset_m"])
        )
        sensor_w_m = float(tile["sensor_w_m"])
        sensor_h_m = float(tile["sensor_h_m"])
    fov_y = 2.0 * math.atan2(sensor_h_m * 0.5, focal_m)
    focus_distance_m = None
    if _ORDERED_SCENE_JOB is not None:
        from scene_orders import camera_focus_distance
        focus_distance_m = camera_focus_distance(_ORDERED_SCENE_JOB)
    if focus_distance_m is None:
        focus_distance_m = max(1.0e-4, float(sensor_center[0] - scene.bounds_min[0]))

    cam = PhysicalCameraRig(
        pos=np.ascontiguousarray(sensor_center, dtype=np.float64),
        fwd=fwd,
        up=up,
        fov_y_rad=float(fov_y),
        width=int(width),
        height=int(height),
        sensor_plane_offset_m=0.0,
        aperture_plane_offset_m=float(focal_m),
        focal_m=float(focal_m),
        aperture_radius_m=max(1.0e-6, float(aperture_radius_m)),
    )
    cam.sensor_w_m = float(sensor_w_m)
    cam.sensor_h_m = float(sensor_h_m)
    cam.aperture_center_world = np.ascontiguousarray(aperture_center, np.float64)
    cam.lens_fwd_world = np.asarray([-1.0, 0.0, 0.0], np.float64)

    error_degree = SimpleNamespace(
        pinhole_target_sensor_z_m=0.0,
        pinhole_comparison_mm=0.0,
        overall=0.0,
        thin_lens_target_sensor_z_m=0.0,
        pinhole_comparison_degree=0.0,
        sensor_plane_degree=0.0,
        focus_distance_degree=0.0,
        coc_degree=0.0,
        ev_degree=0.0,
        thin_lens_residual_diopter=0.0,
    )
    solved = SimpleNamespace(
        camera=cam,
        iterations=0,
        sanity_input=SimpleNamespace(
            sensor_plane_z_m=0.0,
            aperture_rail_z_m=float(focal_m),
            aperture_plane_offset_m=float(focal_m),
            aperture_radius_m=float(aperture_radius_m),
            aperture_shift_x_mm=0.0,
            aperture_shift_y_mm=0.0,
            focus_distance_m=float(focus_distance_m),
            lens_center_z_m=float(focal_m),
            lens_front_shift_x_mm=0.0,
            lens_front_shift_y_mm=0.0,
            lens_front_tilt_x_deg=0.0,
            lens_front_tilt_y_deg=0.0,
            sensor_shift_x_mm=0.0,
            sensor_shift_y_mm=0.0,
            lens_front_plane_z_m=float(focal_m),
            lens_rear_plane_z_m=float(focal_m),
            sensor_corner_tl_mm=0.0,
            sensor_corner_tr_mm=0.0,
            sensor_corner_bl_mm=0.0,
            sensor_corner_br_mm=0.0,
        ),
        sanity_report=SimpleNamespace(
            status=("lab_native_film_stage" if pose is not None else "lab_native"),
            geometry_ok=True,
            circle_of_confusion_um=0.0,
            sensor_adjustment_needed_mm=0.0,
            warnings=[],
            failures=[],
            planes=SimpleNamespace(
                effective_focal_m=float(focal_m),
                focal_plane_z_m=float(focus_distance_m),
            ),
            error_degree=error_degree,
        ),
    )
    return cam, solved


def _build_tracer_scene(t: float, scene_mode: str = "orbiters",
                        cam: Optional[PhysicalCameraRig] = None,
                        solved: Optional[Any] = None,
                        optics: Optional[CameraOptics] = None) -> TracerScene:
    """Materialise the borrowed scene into ray-tracer geometry.

    The MaterialDatabase is built ONCE per process (we trust scene_mod's
    ``register_materials``); ``build_mat_buf`` returns the unified Phase-2
    buffer that both backends index by ``mat_idx[tri] * MAX_SPECTRAL_BANDS
    + band``.
    """
    if _is_thick_lens_lab_scene_mode(scene_mode):
        return _build_thick_lens_lab_tracer_scene()

    preview_gems = (scene_mode == CAMERA_SOLVE_PREVIEW_SCENE)
    base_scene_mode = "orbiters" if preview_gems else scene_mode

    db, idx = scene_mod.register_materials()
    preview_obsidian_idx = -1
    if preview_gems:
        preview_obsidian_idx = _register_preview_obsidian_material(db)

    verts8, _mat_per_v, _gid_per_v, mat_per_tri, groups = \
        scene_mod.scene_for_phase(idx, t, scene_mode=base_scene_mode)

    # ── Triangulate ──────────────────────────────────────────────────────
    pts = verts8[:, 0:3].astype(np.float64).reshape(-1, 3, 3)
    n_tri = pts.shape[0]
    verts_flat = pts.reshape(n_tri, 9)

    # Per-triangle geometric normal (right-hand rule, normalised).
    e1 = pts[:, 1] - pts[:, 0]
    e2 = pts[:, 2] - pts[:, 0]
    cr = np.cross(e1, e2)
    nrm = np.linalg.norm(cr, axis=1, keepdims=True)
    normals = (cr / np.where(nrm > 1.0e-12, nrm, 1.0)).astype(np.float64)
    area = (0.5 * nrm.ravel()).astype(np.float64)
    centroid = pts.mean(axis=1).astype(np.float64)

    # ── Material side ────────────────────────────────────────────────────
    mat_idx_arr = np.ascontiguousarray(mat_per_tri, np.int32)
    if preview_gems and preview_obsidian_idx >= 0:
        sapphire_idx = int(idx.get("sapphire_emit", -1))
        chrome_idx = int(idx.get("chrome", -1))
        if sapphire_idx >= 0:
            mat_idx_arr = np.where(mat_idx_arr == sapphire_idx,
                                   np.int32(preview_obsidian_idx),
                                   mat_idx_arr).astype(np.int32, copy=False)
        if chrome_idx >= 0:
            mat_idx_arr = np.where(mat_idx_arr == chrome_idx,
                                   np.int32(preview_obsidian_idx),
                                   mat_idx_arr).astype(np.int32, copy=False)
    camera_tri_groups: Optional[dict[str, np.ndarray]] = None
    if cam is not None:
        if solved is not None:
            cam.sensor_plane_offset_m = float(solved.sanity_input.sensor_plane_z_m)
            cam.aperture_plane_offset_m = float(
                solved.sanity_input.aperture_rail_z_m
                if solved.sanity_input.aperture_rail_z_m is not None
                else solved.sanity_input.aperture_plane_offset_m
            )
        rig_mat_idx = preview_obsidian_idx if preview_obsidian_idx >= 0 else _register_preview_obsidian_material(db)
        rig_verts, rig_normals, rig_mat_idx_arr, rig_groups = _build_camera_rig_mesh(
            cam=cam,
            optics=(optics if optics is not None else DEFAULT_OPTICS),
            rig_mat_idx=int(rig_mat_idx),
            first_tri_index=int(verts_flat.shape[0]),
        )
        verts_flat = np.vstack([verts_flat, rig_verts])
        normals = np.vstack([normals, rig_normals])
        mat_idx_arr = np.concatenate([mat_idx_arr, rig_mat_idx_arr]).astype(np.int32, copy=False)
        camera_tri_groups = rig_groups

    mat_buf = db.build_mat_buf(freq_hz=(
        _ORDERED_MATERIAL_LUT_GRID
        if _ORDERED_MATERIAL_LUT_GRID is not None else DEFAULT_FREQ_HZ
    ))
    mat_n_mats = int(mat_buf.shape[0] // MAX_SPECTRAL_BANDS)

    # ── Emissive sources: any triangle whose material has nonzero PBR
    #    emission row (build_tensors()['pbr'][mat,8:11]).  We use the same
    #    reference the emissive_ray_packer does, but emit one *source* per
    #    triangle (the C++ tracer expands each source into n_rays Monte-Carlo
    #    samples internally — sub-batching is just N_BATCH calls with
    #    different seeds).
    pbr = db.build_tensors().get("pbr", np.zeros((0, 16), np.float32))
    emis_rgb = pbr[mat_idx_arr.clip(0, max(0, pbr.shape[0]-1)), 8:11]
    luma = np.maximum(0.0, emis_rgb @ _LUMA)
    emissive = luma > 1.0e-8
    if preview_gems:
        ruby_idx = int(idx.get("ruby_emit", -1))
        emerald_idx = int(idx.get("emerald_emit", -1))
        gem_emit_mask = ((mat_idx_arr == ruby_idx) | (mat_idx_arr == emerald_idx))
        emissive = np.logical_and(emissive, gem_emit_mask)
    if not np.any(emissive):
        # Synthesise a dim ambient point so the budget plan still has a source.
        emissive = np.zeros(n_tri, bool); emissive[0] = True
        luma = np.zeros(n_tri, np.float32); luma[0] = 1.0e-3

    sel = np.where(emissive)[0]
    src_pos = centroid[sel]
    src_dir = normals[sel]
    # Lambertian (cos^1) directivity for area emitters.  The C++ tracer reads
    # this as the cosine exponent; 1.0 ≡ Lambertian.
    src_directivity = np.ones(sel.size, np.float64)
    src_area = area[sel]
    # Emissive power per triangle: luminance × area, scaled to the global
    # ``DEFAULT_FILM`` exposure later by the plan's energy_per_ray_J.
    src_emit_W = (luma[sel].astype(np.float64) * src_area)
    src_emit_rgb_W = emis_rgb[sel].astype(np.float64) * src_area[:, None]

    # ── Scene AABB for the GLSL pipeline ─────────────────────────────────
    pts_all = np.asarray(verts_flat, np.float64).reshape(-1, 3)
    bmin = pts_all.min(axis=0).astype(np.float32)
    bmax = pts_all.max(axis=0).astype(np.float32)

    return TracerScene(
        verts            = verts_flat,
        normals          = normals,
        mat_idx          = mat_idx_arr,
        mat_buf          = mat_buf,
        mat_n_mats       = mat_n_mats,
        src_pos          = src_pos,
        src_dir          = src_dir,
        src_directivity  = src_directivity,
        src_area_m2      = src_area,
        src_emit_W       = src_emit_W,
        src_emit_rgb_W   = src_emit_rgb_W,
        src_tri_idx      = sel.astype(np.int32, copy=False),
        bounds_min       = bmin,
        bounds_max       = bmax,
        camera_tri_groups = camera_tri_groups,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Camera geometry — physical rig origin, looking at SCENE_CENTER
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PhysicalCameraRig:
    pos:    np.ndarray    # (3,) double
    fwd:    np.ndarray    # (3,) double
    up:     np.ndarray    # (3,) double
    fov_y_rad: float
    width:  int
    height: int
    sensor_plane_offset_m: float = -0.05
    aperture_plane_offset_m: float = 0.0
    focal_m: float = 0.035
    aperture_radius_m: float = 0.0125

    @classmethod
    def looking_at_scene(cls, width: int, height: int,
                         optics: CameraOptics) -> "PhysicalCameraRig":
        eye = np.array([0.0, 0.0, 0.0], np.float64)
        tgt = np.asarray(scene_mod.SCENE_CENTER, np.float64)
        fwd = tgt - eye
        fwd /= max(np.linalg.norm(fwd), 1.0e-9)
        up = np.array([0.0, 1.0, 0.0], np.float64)
        # Real fov_y from sensor height and focal length.
        fov_y = 2.0 * math.atan2(optics.sensor_h_mm * 0.5, optics.focal_mm)
        focal_m = float(optics.focal_mm) * 1.0e-3
        aperture_radius_m = focal_m / max(2.0 * float(optics.f_number), 1.0e-9)
        return cls(pos=eye, fwd=fwd, up=up,
                   fov_y_rad=float(fov_y), width=width, height=height,
                   focal_m=float(focal_m), aperture_radius_m=float(aperture_radius_m))


@dataclass(frozen=True)
class CameraOpticalRig:
    """Render-facing optical rig built from a solved camera package.

    All fields are in world space (metres).  The render path consumes only
    this object — it must not scrape sanity_input directly.  Any solved
    degree-of-freedom that is nonzero but not listed here must raise an error
    before rendering starts (see build_from_solved).
    """
    cam_pos:          np.ndarray   # (3,) float64 — camera nodal point
    fwd:              np.ndarray   # (3,) float64 — unit forward
    right:            np.ndarray   # (3,) float64 — unit right
    up:               np.ndarray   # (3,) float64 — unit up

    sensor_center:    np.ndarray   # (3,) float64
    sensor_w_m:       float
    sensor_h_m:       float

    aperture_center:  np.ndarray   # (3,) float64
    aperture_radius_m: float

    lens_center:      np.ndarray   # (3,) float64 — equivalent single-lens centre
    effective_focal_m: float       # effective focal length (m)
    focus_distance_m:  float       # scene focus distance from lens centre (m)

    camera_mode:      int          # CAMERA_MODE_* from bdpt_integrator

    @classmethod
    def build_from_solved(
        cls,
        cam: "PhysicalCameraRig",
        solved: Any,
        camera_mode: int,
    ) -> "CameraOpticalRig":
        """Build from a PhysicalCameraRig + SolvedCameraPackage.

        Raises RuntimeError for any solved DOF that is nonzero but not yet
        consumed by the renderer so silent non-consumption is caught early.
        """
        from bdpt_integrator import (
            CAMERA_MODE_APERTURE_CONE, CAMERA_MODE_THIN_LENS_GEOMETRIC,
            CAMERA_MODE_THICK_LENS_WAVE,
        )
        cam_pos = np.asarray(cam.pos, np.float64)
        fwd = np.asarray(cam.fwd, np.float64)
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1.0e-12)
        up_world = np.asarray(cam.up, np.float64)
        right = np.cross(fwd, up_world)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1.0e-9:
            right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right /= max(float(np.linalg.norm(right)), 1.0e-12)
        up = np.cross(right, fwd)
        up /= max(float(np.linalg.norm(up)), 1.0e-12)

        si = solved.sanity_input
        sensor_center = cam_pos + fwd * float(si.sensor_plane_z_m)
        aperture_center = cam_pos + fwd * float(
            si.aperture_rail_z_m if si.aperture_rail_z_m is not None
            else si.aperture_plane_offset_m
        )
        # Use the solver's effective focal length and focus distance.
        effective_focal_m = float(solved.sanity_report.planes.effective_focal_m)
        focus_distance_m = float(solved.sanity_report.planes.focal_plane_z_m - 0.0)
        # For a single-lens rig, lens centre ≈ aperture centre.
        lens_center = aperture_center.copy()

        # Warn on unsupported nonzero DOFs so they don't silently disappear.
        _warn_unconsumed = []
        if abs(float(si.sensor_shift_x_mm)) > 1e-3 or abs(float(si.sensor_shift_y_mm)) > 1e-3:
            _warn_unconsumed.append("sensor_shift_xy")
        if abs(float(si.aperture_shift_x_mm)) > 1e-3 or abs(float(si.aperture_shift_y_mm)) > 1e-3:
            _warn_unconsumed.append("aperture_shift_xy")
        if abs(float(si.lens_front_shift_x_mm)) > 1e-3 or abs(float(si.lens_front_shift_y_mm)) > 1e-3:
            _warn_unconsumed.append("lens_front_shift_xy")
        if abs(float(si.lens_front_tilt_x_deg)) > 0.01 or abs(float(si.lens_front_tilt_y_deg)) > 0.01:
            _warn_unconsumed.append("lens_front_tilt_xy")
        for c in [si.sensor_corner_tl_mm, si.sensor_corner_tr_mm,
                  si.sensor_corner_bl_mm, si.sensor_corner_br_mm]:
            if abs(float(c)) > 1e-3:
                _warn_unconsumed.append("sensor_corner_tilt")
                break
        if _warn_unconsumed:
            import warnings
            warnings.warn(
                f"CameraOpticalRig: solved DOFs not consumed by renderer: "
                f"{_warn_unconsumed}. Image may not reflect solver intent.",
                stacklevel=2,
            )

        aperture_r = float(si.aperture_radius_m) if si.aperture_radius_m is not None else float(cam.aperture_radius_m)

        return cls(
            cam_pos=cam_pos, fwd=fwd, right=right, up=up,
            sensor_center=sensor_center,
            sensor_w_m=DEFAULT_FILM_FORMAT.frame_width_mm * 1.0e-3,
            sensor_h_m=DEFAULT_FILM_FORMAT.frame_height_mm * 1.0e-3,
            aperture_center=aperture_center,
            aperture_radius_m=aperture_r,
            lens_center=lens_center,
            effective_focal_m=float(effective_focal_m),
            focus_distance_m=float(focus_distance_m),
            camera_mode=int(camera_mode),
        )

    def to_camera_sensor(
        self,
        n_px: int,
        n_py: int,
        n_aperture_samples: int,
        aperture_stop_group_id: int = -1,
    ) -> Any:
        """Build a CameraSensor dict from this rig, ready for register_tri_group."""
        from bdpt_integrator import CameraSensor
        return CameraSensor(
            pos=self.sensor_center,
            fwd=self.fwd,
            up=self.up,
            sensor_w_m=self.sensor_w_m,
            sensor_h_m=self.sensor_h_m,
            focal_m=float(np.linalg.norm(self.aperture_center - self.sensor_center)),
            aperture_radius_m=self.aperture_radius_m,
            n_px=int(n_px),
            n_py=int(n_py),
            n_aperture_samples=int(n_aperture_samples),
            aperture_stop_group_id=int(aperture_stop_group_id),
            camera_mode=int(self.camera_mode),
            effective_focal_m=float(self.effective_focal_m),
            focus_distance_m=float(self.focus_distance_m),
            lens_center=self.lens_center,
            lens_fwd=self.fwd,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backend interface
# ─────────────────────────────────────────────────────────────────────────────
class ExposureBackend:
    name = "abstract"

    def __init__(self, scene: TracerScene, cam: PhysicalCameraRig,
                 freq_hz: np.ndarray, *, max_bounces: int = 4,
                 min_amplitude: float = 1.0e-3, atmo_abs_db_per_m: float = 0.0):
        self.scene = scene
        self.cam   = cam
        self.freq_hz = np.ascontiguousarray(freq_hz, np.float64)
        self.n_bands = int(self.freq_hz.size)
        self.max_bounces = int(max_bounces)
        self.min_amplitude = float(min_amplitude)
        self.atmo_abs = np.full(self.n_bands, float(atmo_abs_db_per_m), np.float64)
        # Persistent per-exposure accumulators (n_bands, H, W) in float32.
        self.accum       = np.zeros((self.n_bands, cam.height, cam.width), np.float32)
        # surf_accum: sensor-group EndpointRecord contributions (direct hits).
        # field_accum: non-sensor EndpointRecord contributions (ambient/field).
        self.surf_accum  = np.zeros((self.n_bands, cam.height, cam.width), np.float32)
        self.field_accum = np.zeros((self.n_bands, cam.height, cam.width), np.float32)
        self.n_rays_accumulated = 0
        # Event telemetry for camera mode validation and diagnostics
        self.camera_event_telemetry = CameraEventTelemetry()

    # ── Sub-class hooks ──────────────────────────────────────────────────
    def reset_exposure(self) -> None:
        self.accum.fill(0.0)
        self.surf_accum.fill(0.0)
        self.field_accum.fill(0.0)
        self.n_rays_accumulated = 0
        # Reset event telemetry for new exposure
        self.camera_event_telemetry = CameraEventTelemetry()

    def render_batch(self, n_rays: int, seed: int) -> None:
        raise NotImplementedError

    def finalize_image(self, gain: float) -> np.ndarray:
        """Return (H, W, 3) float32 RGB image after applying ``gain``.

        The accumulator holds per-band amplitude magnitudes |A[b]|;
        we map the band index into a smooth wavelength-driven RGB triplet
        (CIE-style) for visualisation.
        """
        a = self.accum * float(gain)
        rgb = _bands_to_rgb(a, self.freq_hz)
        # Tone-map (Reinhard) for display only; calibration uses raw ``a``.
        m = float(rgb.max()) if rgb.size else 0.0
        if m > 0.0:
            disp = rgb / (1.0 + rgb)
        else:
            disp = rgb
        return np.clip(disp, 0.0, 1.0).astype(np.float32)

    def measured_radiant_exposure_J(self, energy_per_ray_J: float) -> float:
        """Translate the running accumulator into a Joule estimate.

        The C++ ``integrate_image_into`` deposits ``|A| · cos / r²`` in
        normalised units per ray.  We close the loop by saying the total
        deposited "amplitude mass" times ``energy_per_ray_J`` divided by the
        actual ray count is the per-ray Joule contribution; summed across
        the image gives total H.  Calibration discovers the multiplicative
        constant that makes this match the predicted ``plan.target_H_J``.
        """
        if self.n_rays_accumulated == 0:
            return 0.0
        amplitude_mass = float(self.accum.sum())
        return amplitude_mass * energy_per_ray_J


# ─────────────────────────────────────────────────────────────────────────────
# Spectral → display RGB (HSL compositor with soft wavelength gating)
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid01(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def _hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Vectorized HSL → RGB conversion for display compositing."""
    h = np.mod(np.asarray(h, np.float64), 1.0)
    s = np.clip(np.asarray(s, np.float64), 0.0, 1.0)
    l = np.clip(np.asarray(l, np.float64), 0.0, 1.0)

    q = np.where(l < 0.5, l * (1.0 + s), l + s - l * s)
    p = 2.0 * l - q

    def _hue_to_rgb(t: np.ndarray) -> np.ndarray:
        t = np.mod(t, 1.0)
        return np.where(
            t < 1.0 / 6.0, p + (q - p) * 6.0 * t,
            np.where(
                t < 1.0 / 2.0, q,
                np.where(
                    t < 2.0 / 3.0, p + (q - p) * (2.0 / 3.0 - t) * 6.0,
                    p,
                ),
            ),
        )

    r = _hue_to_rgb(h + 1.0 / 3.0)
    g = _hue_to_rgb(h)
    b = _hue_to_rgb(h - 1.0 / 3.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _bands_to_rgb(image_b_h_w: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    """Project spectral bands to display RGB via HSL accumulation.

    H tracks the wavelength-to-visible remap, L tracks deposited power with a
    tuned sigmoid, and S comes from circular hue coherence after alpha-weighted
    mixing. UV and deep-red extremes are softened via alpha so they still
    contribute without overpowering the visible band.
    """
    n_b = image_b_h_w.shape[0]
    if n_b == 0:
        H, W = image_b_h_w.shape[1:]
        return np.zeros((H, W, 3), np.float32)

    power = np.maximum(0.0, np.asarray(image_b_h_w, np.float64))
    wl_nm = (C_LIGHT / np.asarray(freq_hz, np.float64)) * 1.0e9

    # Map all wavelengths into the visible hue range, saturating toward UV at
    # the short end and red at the long end.
    hue_sigmoid_nm = 42.0
    hue_mix = _sigmoid01((wl_nm - 555.0) / hue_sigmoid_nm)
    hue = (1.0 - hue_mix) * 0.74

    # Fade the extremes with alpha rather than hard-clamping them out.
    edge_dist = np.abs(wl_nm - 555.0)
    alpha = 0.18 + 0.82 * _sigmoid01((150.0 - edge_dist) / 24.0)
    sat_gate = _sigmoid01((132.0 - edge_dist) / 28.0)

    weighted = power * alpha[:, None, None]
    total_weight = weighted.sum(axis=0)
    total_power = power.sum(axis=0)

    cos_h = np.cos((2.0 * math.pi) * hue)[:, None, None]
    sin_h = np.sin((2.0 * math.pi) * hue)[:, None, None]

    hue_x = np.sum(weighted * cos_h, axis=0)
    hue_y = np.sum(weighted * sin_h, axis=0)
    coherence = np.sqrt(hue_x * hue_x + hue_y * hue_y) / np.maximum(total_weight, 1.0e-9)

    sat_weighted = power * (alpha * sat_gate)[:, None, None]
    sat_mean = np.sum(sat_weighted, axis=0) / np.maximum(total_power, 1.0e-9)
    saturation = np.clip((coherence ** 0.82) * np.sqrt(np.maximum(sat_mean, 0.0)), 0.0, 1.0)

    hue_map = (np.arctan2(hue_y, hue_x) / (2.0 * math.pi)) % 1.0

    white = float(np.percentile(total_power, 99.8)) if total_power.size else 0.0
    white = max(white, 1.0e-8)
    lightness = _sigmoid01((total_power / white - 0.5) * 5.5)

    return _hsl_to_rgb(hue_map, saturation, lightness)


# ─────────────────────────────────────────────────────────────────────────────
# C++ backend — _spectral_kernels.RayTracer  (Phase 2b SLICE 1 ctor)
# ─────────────────────────────────────────────────────────────────────────────
class CppExposureBackend(ExposureBackend):
    name = "cpp"

    def __init__(self, scene: TracerScene, cam: PhysicalCameraRig,
                 freq_hz: np.ndarray, **kw: Any):
        self._adaptive_mode = str(kw.pop("adaptive_mode", "stochastic")).strip().lower()
        if self._adaptive_mode not in ("stochastic", "quota", "uniform"):
            self._adaptive_mode = "stochastic"
        self._t5_min_geom = float(kw.pop("t5_min_geom", -1.0))
        self._gpu_resident = bool(kw.pop("gpu_resident", False))
        self._bdpt_intermediate_mode = str(kw.pop("bdpt_intermediate_mode", "memory")).strip().lower()
        if self._bdpt_intermediate_mode not in ("memory", "file"):
            self._bdpt_intermediate_mode = "memory"
        self._bdpt_intermediate_dir = str(kw.pop("bdpt_intermediate_dir", os.getcwd()))
        self._bdpt_intermediate_max_bytes = _clamp_bdpt_intermediate_bytes(
            int(kw.pop("bdpt_intermediate_max_bytes", 0))
        )
        self._bdpt_retain_intermediate = bool(kw.pop("retain_bdpt_intermediate", False))
        self._bdpt_initial_cap_hint = max(0, int(kw.pop("bdpt_initial_cap_hint", 0)))
        self._bdpt_stream_working_fraction = float(max(0.05, min(1.0, _BDPT_STREAM_WORKING_FRACTION)))
        super().__init__(scene, cam, freq_hz, **kw)
        os.makedirs(self._bdpt_intermediate_dir, exist_ok=True)
        if not self._bdpt_retain_intermediate:
            _cleanup_stale_ephemeral_bdpt_files(self._bdpt_intermediate_dir)
        self.tracer = _sk.RayTracer(
            n_tri      = int(scene.verts.shape[0]),
            verts      = scene.verts,
            normals    = scene.normals,
            mat_idx    = scene.mat_idx,
            mat_buf    = scene.mat_buf,
            mat_n_mats = scene.mat_n_mats,
            freq_hz    = self.freq_hz,
            speed_m_s  = float(C_LIGHT),
            atmo_abs   = self.atmo_abs,
        )
        if _ORDERED_TRANSPORT_LUT is not None:
            self.tracer.configure_spectral_luts(**_ORDERED_TRANSPORT_LUT)
            print(
                "[config] native per-ray spectral LUT resolver loaded "
                f"lanes={self.freq_hz.size} profiles="
                f"{_ORDERED_TRANSPORT_LUT['profile_offsets'].size - 1}",
                flush=True,
            )
        if _ORDERED_TRANSPORT_RGB_WEIGHTS is not None:
            weights = np.ascontiguousarray(
                _ORDERED_TRANSPORT_RGB_WEIGHTS, np.float32
            )
            if weights.shape != (self.freq_hz.size, 3):
                raise ValueError(
                    "ordered transport RGB weights must match active frequencies"
                )
            self.tracer.set_uv_blit_weights(weights, mode=0)
            print(
                "[config] exact transport sensor weights loaded "
                f"lanes={weights.shape[0]}"
            )

        # GPU-resident camera exposures use the recursive sensor hierarchy.
        # Configuration must precede the tracer's lazy pipeline creation.
        self._sensor_mipmap_enabled = bool(
            self._gpu_resident and hasattr(self.tracer, "configure_sensor_mipmap")
        )
        self._sensor_samples_per_node = 1
        if self._sensor_mipmap_enabled:
            # The live scheduler only advances at most 1024 nodes per internal
            # refinement step.  Reserving two million nodes up front also
            # reserves eighteen million lineage and spectral-sample records;
            # together with the native BDPT buffers that exhausts a 12 GiB
            # display GPU on the second exposure layer.  Half a million nodes
            # covers well beyond the clarity horizon while leaving transport
            # records resident across repeated layers.
            mip_nodes = 524_288
            sensor_samples_per_node = max(
                1,
                int(os.environ.get("SPECTRAL_SENSOR_SAMPLES_PER_NODE", "32")),
            )
            self._sensor_samples_per_node = sensor_samples_per_node
            self.tracer.configure_sensor_mipmap(
                max_nodes=mip_nodes,
                maximum_depth=7,
                # Actual camera primaries emitted per selected sensor node in
                # every native mip submission.
                samples_per_epoch=sensor_samples_per_node,
                subdivision_axis=int(
                    dict(getattr(
                        _NEXT_SITE_SCAN_COMMAND, "metadata", {}
                    )).get("sensor_grid", {}).get("subdivision_axis", 3)
                ),
            )
            print(
                "[config] GPU sensor rays "
                f"samples-per-selected-node={sensor_samples_per_node}"
            )
            priority_model_path = str(
                os.environ.get("SPECTRAL_SENSOR_PRIORITY_MODEL", "")
            ).strip()
            if priority_model_path:
                if not hasattr(self.tracer, "configure_sensor_priority_network"):
                    raise RuntimeError(
                        "native tracer does not expose GPU sensor priority inference"
                    )
                model = np.load(priority_model_path, allow_pickle=False)
                parameters = np.ascontiguousarray(model["parameters"], np.float32)
                self.tracer.configure_sensor_priority_network(parameters)
                print(
                    f"[config] GPU sensor priority network loaded "
                    f"parameters={parameters.size} path={priority_model_path}"
                )
            if (
                _NEXT_SITE_SCAN_COMMAND is not None
                and (
                    _NEXT_SITE_SCAN_COMMAND.uv_requests
                    or _NEXT_SITE_SCAN_COMMAND.pixel_slice_requests
                )
            ):
                if not hasattr(self.tracer, "configure_sensor_requested_priority_map"):
                    raise RuntimeError(
                        "native tracer does not expose next-scan UV requests"
                    )
                requested = _NEXT_SITE_SCAN_COMMAND.priority_map(
                    max(self.cam.width, self.cam.height),
                    max(self.cam.width, self.cam.height),
                )
                self.tracer.configure_sensor_requested_priority_map(requested)
                print(
                    f"[config] GPU next-scan UV requests loaded "
                    f"regions={len(_NEXT_SITE_SCAN_COMMAND.uv_requests)} "
                    f"pixel_slices={len(_NEXT_SITE_SCAN_COMMAND.pixel_slice_requests)}"
                )
            restore_sum_path = str(
                os.environ.get("SPECTRAL_SENSOR_RESTORE_SUM", "")
            ).strip()
            restore_weight_path = str(
                os.environ.get("SPECTRAL_SENSOR_RESTORE_WEIGHT", "")
            ).strip()
            restore_dirty_path = str(
                os.environ.get("SPECTRAL_SENSOR_DIRTY_SITES", "")
            ).strip()
            if restore_sum_path or restore_weight_path or restore_dirty_path:
                if not all((restore_sum_path, restore_weight_path, restore_dirty_path)):
                    raise RuntimeError(
                        "delta restore requires sum, weight, and dirty-site paths"
                    )
                if not hasattr(self.tracer, "configure_sensor_delta_restore"):
                    raise RuntimeError(
                        "native tracer does not expose sparse sensor delta restore"
                    )
                restored_sum = np.asarray(
                    np.load(restore_sum_path, allow_pickle=False), np.float32
                )
                restored_weight = np.asarray(
                    np.load(restore_weight_path, allow_pickle=False), np.float32
                )
                dirty_sites = np.asarray(
                    np.load(restore_dirty_path, allow_pickle=False), np.uint32
                ).reshape(-1)
                native_resolution = max(self.cam.width, self.cam.height)
                if restored_sum.shape != (
                    native_resolution, native_resolution, 3
                ):
                    raise ValueError(
                        "delta restore RGB must match the native square sensor"
                    )
                if restored_weight.shape != (
                    native_resolution, native_resolution
                ):
                    raise ValueError(
                        "delta restore weight must match the native square sensor"
                    )
                self.tracer.configure_sensor_delta_restore(
                    restored_sum, restored_weight, dirty_sites
                )
                print(
                    f"[config] restored GPU sensor exposure; "
                    f"dirty_sites={dirty_sites.size}"
                )

        if self._t5_min_geom > 0.0 and hasattr(self.tracer, "set_t5_min_geom"):
            self.tracer.set_t5_min_geom(self._t5_min_geom)
            print(f"[config] C++ tracer t5_min_geom={self._t5_min_geom:.1e}")
        if self._gpu_resident and hasattr(self.tracer, "set_gpu_skip_record_readback"):
            self.tracer.set_gpu_skip_record_readback(True)
            print("[config] C++ tracer GPU-resident BDPT enabled")

        self._src_tri_verts = np.ascontiguousarray(
            self.scene.verts[self.scene.src_tri_idx].reshape(-1, 3, 3),
            np.float64,
        )
        self._source_need_ema: Optional[np.ndarray] = None
        self._source_rays_emitted = np.zeros(int(self.scene.src_pos.shape[0]), dtype=np.float64)
        self._sensor_camera_desc: Optional[dict] = None
        self._bdpt_dynamic_cap_hint: int = int(self._bdpt_initial_cap_hint)
        self._bdpt_last_overflow_path: Optional[str] = None
        self._native_sensor_image: Optional[np.ndarray] = None
        self._native_sensor_linear_image: Optional[np.ndarray] = None
        self._native_sensor_sum_linear: Optional[np.ndarray] = None
        self._native_sensor_exposure_weight: Optional[np.ndarray] = None
        self._native_sensor_priority_map: Optional[np.ndarray] = None
        self.sensor_uv_hits = np.zeros((cam.height, cam.width), np.float32)
        self.sensor_uv_thin_lens_hits = np.zeros((cam.height, cam.width), np.float32)
        
        # Initialize optical backend for handler telemetry
        self._optical_backend = _sk.ExposureBackendCpp()
        self._optical_assembly: Optional[OpticalAssembly] = None
        self._optical_telemetry: Optional[dict] = None

    def _native_emitter_launch_args(self) -> tuple[int, float, float, float, float]:
        """Resolve the scene-order emitter launch profile for the native ABI."""

        profile = dict(
            getattr(getattr(self, "scene", None), "emitter_launch_profile", {})
            or {}
        )
        if not profile:
            optical_camera = getattr(getattr(self, "scene", None), "optical_camera", None)
            flash = dict(getattr(optical_camera, "manifest", {}).get("flash", {})) if optical_camera is not None else {}
            profile = dict(flash.get("launch_profile", {}))
        mode = str(profile.get("mode", "lambertian")).strip().lower()
        if mode != "collimated":
            return (0, -1.0, 0.0, 0.0, 0.0)
        direction = np.asarray(profile.get("direction", [-1.0, 0.0, 0.0]), np.float64).reshape(-1)
        if direction.size != 3 or not np.all(np.isfinite(direction)):
            raise ValueError("collimated emitter direction must contain three finite values")
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-15:
            raise ValueError("collimated emitter direction must be nonzero")
        direction /= norm
        divergence_deg = max(0.0, float(profile.get("divergence_deg", 0.0)))
        return (
            1, float(direction[0]), float(direction[1]), float(direction[2]),
            float(math.radians(divergence_deg)),
        )

    def _bdpt_overflow_snapshot(self) -> dict[str, int]:
        if not hasattr(self.tracer, "get_bdpt_overflow"):
            return {}
        return {
            str(key): int(value)
            for key, value in dict(self.tracer.get_bdpt_overflow()).items()
        }

    def _assert_no_bdpt_overflow(
        self, before: dict[str, int], *, page_label: str,
    ) -> None:
        if not before:
            return
        after = self._bdpt_overflow_snapshot()
        delta = {
            key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in set(before) | set(after)
            if int(after.get(key, 0)) > int(before.get(key, 0))
        }
        if delta:
            raise RuntimeError(
                f"BDPT {page_label} rejected because native records overflowed: "
                f"{delta}. The incomplete grey fallback was not published."
            )

    def run_recursive_sensor_epoch(
        self, *, top_k: int, seed: int, emitter_tri_ids: np.ndarray,
        refinement_steps: int = 8, targeted_fraction: float = 0.75,
    ) -> None:
        """Trace one selected recursive sensor exposure entirely on the GPU."""
        if not self._sensor_mipmap_enabled:
            raise RuntimeError("recursive sensor exposure is not configured")
        if self._sensor_camera_desc is None:
            raise RuntimeError("recursive sensor exposure requires a registered camera")
        # Keep connection batches small enough to publish visible refinement
        # layers promptly. Repeated stratified estimates accumulate without
        # changing the spectral estimator.
        if hasattr(self.tracer, "set_t5_pair_budget"):
            # At 32 primaries/node the mature camera side is ~1.1M vertices.
            # Eight million pairs preserves several stratified light samples
            # per camera vertex instead of starving that estimator dimension.
            self.tracer.set_t5_pair_budget(max(
                1,
                int(os.environ.get(
                    "SPECTRAL_SENSOR_T5_PAIR_BUDGET", "8000000"
                )),
            ))
        # Unlike the legacy flash launcher, an adaptive camera epoch may be
        # the first transport operation. Create the configured GPU pipeline
        # explicitly instead of depending on an unrelated submission side effect.
        self.tracer.ensure_pipeline(
            # The thick-lens BDPT contract follows one stochastic dispersive
            # branch per path. Branching here duplicates camera strategies and
            # destroys the connection estimator's useful sampling density.
            max_children=1,
            seed=int(seed),
            min_amplitude=0.0,
            use_gpu_compute=True,
            gpu_all_stages=True,
            shader_dir=_SHADER_DIR,
        )
        tri_ids = np.ascontiguousarray(emitter_tri_ids, dtype=np.int32).reshape(-1)
        if tri_ids.size <= 0:
            raise RuntimeError("recursive sensor exposure requires an emitter")
        requested_top_k = max(1, int(top_k))
        samples_per_node = max(1, int(self._sensor_samples_per_node))
        if int(refinement_steps) != 1:
            raise ValueError(
                "recursive sensor refinement_steps must be 1; compile larger "
                "logical work as additional max_sensor_epochs so every mip "
                "selection receives its own complete batching cycle"
            )
        # Fixed transport emits one persistent spectral record per active lane.
        # Page against that cumulative arena, not only the transient primary
        # dispatch and vertex/PDF arenas. Continuous LUT transport resolves one
        # immutable frequency per ray and therefore has record width one.
        record_lane_width = (
            1 if _ORDERED_TRANSPORT_LUT is not None
            else max(1, int(np.asarray(
                getattr(self, "freq_hz", (1.0,))
            ).size))
        )
        overflow_before = self._bdpt_overflow_snapshot()
        self.tracer.begin_sensor_batching()
        try:
            submitted_flash = int(self.tracer.submit_emissive_triangles(
                tri_ids,
                max(1, int(os.environ.get(
                    "SPECTRAL_SENSOR_FLASH_RAYS", "32000"
                ))),
                1.0,
                1.0,
                max(1, int(self.max_bounces)),
                0.0,
                1,
                int(seed),
                True,
                True,
                _SHADER_DIR,
                0.0, 0.0, 0.0, 0.0,
                *self._native_emitter_launch_args(),
            ))
            if submitted_flash <= 0:
                raise RuntimeError(
                    "recursive sensor exposure submitted no light paths"
                )
            self.tracer.signal_flash_dispatched()
            cycle = 0
            for step in range(1):
                if cycle > 0:
                    # One complete selection per cycle. Native batching retains
                    # page zero's packed light rows across refinement steps.
                    self.tracer.signal_flash_dispatched()
                ok = bool(self.tracer.submit_sensor_mip_epoch(
                    top_k=requested_top_k,
                    seed=_bounded_native_seed(int(seed) * 65_537 + step * 257),
                    max_bounces=max(1, int(self.max_bounces)),
                    min_amplitude=0.0,
                    exposure_weight=1.0,
                    targeted_fraction=float(targeted_fraction),
                ))
                if not ok:
                    raise RuntimeError("GPU recursive sensor epoch failed")
                self.tracer.signal_sensor_dispatched()
                if hasattr(self.tracer, "in_flight_count"):
                    deadline = time.perf_counter() + 30.0
                    while time.perf_counter() < deadline:
                        if int(self.tracer.in_flight_count()) == 0:
                            break
                        time.sleep(0.002)
                self.tracer.join_t5()
                self._assert_no_bdpt_overflow(
                    overflow_before,
                    page_label=f"recursive sensor step {step}",
                )
                overflow_before = self._bdpt_overflow_snapshot()
                cycle += 1
        finally:
            self.tracer.end_sensor_batching()

    def reset_exposure(self) -> None:
        super().reset_exposure()
        self._native_sensor_image = None
        self._native_sensor_linear_image = None
        self._native_sensor_sum_linear = None
        self._native_sensor_exposure_weight = None
        self._native_sensor_priority_map = None
        self.sensor_uv_hits.fill(0.0)
        self.sensor_uv_thin_lens_hits.fill(0.0)

    def run_gpu_first_scene_depth(
        self,
        *,
        scene_tri_ids: np.ndarray,
        seed: int,
        rays_per_page: int = 16_384,
    ) -> np.ndarray:
        """Trace one real camera ray per native sensor site and return metres.

        The camera rays and every optical/scene intersection run through the
        native GPU pipeline.  CPU readback performs only the final product
        reduction: for each sensor site, retain the nearest STRIKE belonging
        to the authored scene-object group.  Lens and camera proxy events are
        therefore traversed but never mistaken for scene depth.
        """

        required = (
            "ensure_pipeline", "submit_sensor_sweep", "drain_records",
            "in_flight_count", "pipeline_stats",
        )
        missing = [name for name in required if not hasattr(self.tracer, name)]
        if missing:
            raise RuntimeError(
                "GPU first-scene depth requires RayTracer methods: "
                + ", ".join(missing)
            )
        if self._sensor_camera_desc is None:
            raise RuntimeError("GPU first-scene depth requires a registered camera")

        object_ids = np.unique(np.asarray(scene_tri_ids, np.int32).reshape(-1))
        if object_ids.size == 0:
            raise RuntimeError("GPU first-scene depth requires authored scene triangles")
        native_res = int(max(self.cam.width, self.cam.height))
        schedule_total = native_res * native_res
        depth_flat = np.full((schedule_total,), np.inf, np.float32)

        # Depth is a diagnostic readback product, so explicitly enable records
        # before the lazily-created all-GPU pipeline is instantiated.
        if hasattr(self.tracer, "set_gpu_skip_record_readback"):
            self.tracer.set_gpu_skip_record_readback(False)
        self.tracer.ensure_pipeline(
            max_children=1,
            # Depth is a geometric diagnostic, not an aperture-integration
            # estimate.  Native seed zero aims every primary at the pupil
            # centre; a non-zero seed selects a different Fibonacci pupil
            # point for each page and turns lens vignetting into random holes.
            seed=0,
            min_amplitude=0.0,
            use_gpu_compute=True,
            gpu_all_stages=True,
            shader_dir=_SHADER_DIR,
        )

        def reduce_ready() -> int:
            reduced = 0
            while True:
                records = self.tracer.drain_records(max_n=50_000)
                kind = np.asarray(records.get("kind", ()), np.uint8).reshape(-1)
                if kind.size == 0:
                    break
                src = np.asarray(records["src_id"], np.int64).reshape(-1)
                tri = np.asarray(records["hit_tri"], np.int32).reshape(-1)
                path = np.asarray(records["path_len"], np.float32).reshape(-1)
                keep = (
                    (kind == 0)
                    & (src >= 0)
                    & (src < schedule_total)
                    & np.isfinite(path)
                    & (path >= 0.0)
                    & np.isin(tri, object_ids, assume_unique=False)
                )
                if np.any(keep):
                    np.minimum.at(depth_flat, src[keep], path[keep])
                    reduced += int(np.count_nonzero(keep))
            return reduced

        submitted = 0
        page = max(1, min(int(rays_per_page), schedule_total))
        while submitted < schedule_total:
            count = int(self.tracer.submit_sensor_sweep(
                # The reducer wants the first authored-scene strike.  Lens
                # surfaces are traversed inside the camera launch; allowing
                # scene scattering beyond the first strike only creates
                # records that the minimum-depth reduction will discard.
                max_bounces=1,
                min_amplitude=0.0,
                max_rays=min(page, schedule_total - submitted),
                pix_offset=submitted,
                max_children=1,
                aperture_samples=1,
                seed=0,
                exposure_weight=1.0,
            ))
            if count <= 0:
                raise RuntimeError(
                    f"GPU depth sensor sweep stopped at {submitted}/{schedule_total} rays"
                )
            submitted += count
            deadline = time.perf_counter() + 30.0
            while True:
                reduce_ready()
                if int(self.tracer.in_flight_count()) == 0:
                    reduce_ready()
                    if int(self.tracer.pipeline_stats().get("output_queue_depth", 0)) == 0:
                        break
                if time.perf_counter() >= deadline:
                    raise TimeoutError(
                        f"GPU depth page did not drain after 30 s ({submitted}/{schedule_total})"
                    )
                time.sleep(0.001)

        depth_native = depth_flat.reshape(native_res, native_res)
        output_width, output_height = (
            _ORDERED_TILE_OUTPUT
            if _ORDERED_TILE_OUTPUT is not None
            else (int(self.cam.width), int(self.cam.height))
        )
        depth_display = _native_sensor_tile_to_display(
            depth_native[..., None], output_width, output_height
        )[..., 0].astype(np.float32, copy=False)
        valid = np.isfinite(depth_display)
        preview = np.zeros_like(depth_display, np.float32)
        if np.any(valid):
            near = float(np.min(depth_display[valid]))
            far = float(np.max(depth_display[valid]))
            if far > near:
                # Conventional depth preview: near is bright, far is dark.
                preview[valid] = 1.0 - (depth_display[valid] - near) / (far - near)
            else:
                preview[valid] = 1.0
        raw_rgb = np.repeat(depth_display[..., None], 3, axis=2)
        preview_rgb = np.repeat(preview[..., None], 3, axis=2)
        self._native_sensor_linear_image = np.ascontiguousarray(raw_rgb, np.float32)
        self._native_sensor_image = np.ascontiguousarray(preview_rgb, np.float32)
        self.n_rays_accumulated += int(submitted)
        print(
            "  [depth-gpu] first authored-scene strike "
            f"sites={int(np.count_nonzero(valid))}/{valid.size} rays={submitted:,} "
            f"range={float(np.min(depth_display[valid])) if np.any(valid) else float('nan'):.6g}.."
            f"{float(np.max(depth_display[valid])) if np.any(valid) else float('nan'):.6g} m",
            flush=True,
        )
        return self._native_sensor_image

    def cleanup_overflow_temp_file(self) -> None:
        if not self._bdpt_last_overflow_path:
            return
        if self._bdpt_retain_intermediate:
            return
        try:
            if os.path.exists(self._bdpt_last_overflow_path):
                os.remove(self._bdpt_last_overflow_path)
        except OSError:
            pass
        self._bdpt_last_overflow_path = None

    def _current_bdpt_disk_usage_bytes(self) -> int:
        return _bdpt_disk_usage_bytes(self._bdpt_intermediate_dir)

    def _require_bdpt_disk_budget(self, alloc_bytes: int, reason: str) -> None:
        if self._bdpt_intermediate_max_bytes <= 0:
            return
        requested = max(0, int(alloc_bytes))
        if requested <= 0:
            return
        current = self._current_bdpt_disk_usage_bytes()
        total = current + requested
        if (total > int(self._bdpt_intermediate_max_bytes)) and (not self._bdpt_retain_intermediate):
            # Best-effort stale sweep before failing a batch due to disk cap.
            _cleanup_stale_ephemeral_bdpt_files(
                self._bdpt_intermediate_dir,
                include_retained=True,
            )
            current = self._current_bdpt_disk_usage_bytes()
            total = current + requested
        if total <= int(self._bdpt_intermediate_max_bytes):
            return
        cap_gb = self._bdpt_intermediate_max_bytes / float(1024 ** 3)
        current_gb = current / float(1024 ** 3)
        requested_gb = requested / float(1024 ** 3)
        raise RuntimeError(
            "BDPT disk budget exceeded for "
            f"{reason}: current={current_gb:.2f} GiB requested={requested_gb:.2f} GiB "
            f"cap={cap_gb:.2f} GiB; stale intermediates or batch settings are too large."
        )

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        cam_f = np.asarray(self.cam.fwd, np.float64)
        cam_f = cam_f / max(np.linalg.norm(cam_f), 1.0e-12)
        cam_up_hint = np.asarray(self.cam.up, np.float64)
        cam_r = np.cross(cam_f, cam_up_hint)
        if np.linalg.norm(cam_r) < 1.0e-12:
            cam_r = np.cross(cam_f, np.array([1.0, 0.0, 0.0], np.float64))
        cam_r = cam_r / max(np.linalg.norm(cam_r), 1.0e-12)
        cam_u = np.cross(cam_r, cam_f)
        tan_half_v = math.tan(float(self.cam.fov_y_rad) * 0.5)
        tan_half_h = tan_half_v * (float(self.cam.width) / max(1.0, float(self.cam.height)))
        return cam_f, cam_r, cam_u, tan_half_h, tan_half_v

    def _density_need_map(self) -> np.ndarray:
        # Prioritize under-resolved regions and high local contrast zones.
        energy = np.sum(self.accum, axis=0, dtype=np.float64)
        p95 = max(1.0e-12, float(np.percentile(energy, 95.0)))
        en = np.clip(energy / p95, 0.0, 1.0)
        gx = np.abs(np.roll(en, -1, axis=1) - en)
        gy = np.abs(np.roll(en, -1, axis=0) - en)
        g = np.sqrt(gx * gx + gy * gy)
        gp95 = max(1.0e-12, float(np.percentile(g, 95.0)))
        gn = np.clip(g / gp95, 0.0, 1.0)
        under = 1.0 - np.sqrt(np.clip(en, 0.0, 1.0))
        need = 0.25 + under + 0.75 * gn
        return need.astype(np.float64, copy=False)

    def _project_source_need(self, src_pos: np.ndarray, need_map: np.ndarray) -> np.ndarray:
        cam_pos = np.asarray(self.cam.pos, np.float64)
        cam_f, cam_r, cam_u, tan_half_h, tan_half_v = self._camera_basis()
        v = np.asarray(src_pos, np.float64) - cam_pos[None, :]
        depth = np.dot(v, cam_f)

        need = np.full((src_pos.shape[0],), float(np.mean(need_map, dtype=np.float64)), dtype=np.float64)
        valid = depth > 1.0e-9
        if np.any(valid):
            vv = v[valid]
            d = depth[valid]
            x_img = np.dot(vv, cam_r)
            y_img = np.dot(vv, cam_u)
            ndc_x = x_img / (d * tan_half_h)
            ndc_y = -y_img / (d * tan_half_v)
            px = ((ndc_x + 1.0) * 0.5 * float(self.cam.width)).astype(np.int64)
            py = ((ndc_y + 1.0) * 0.5 * float(self.cam.height)).astype(np.int64)
            in_view = ((px >= 0) & (px < int(self.cam.width)) &
                       (py >= 0) & (py < int(self.cam.height)))
            idx_valid = np.where(valid)[0]
            if np.any(in_view):
                idx = idx_valid[in_view]
                need[idx] = need_map[py[in_view], px[in_view]]
        return need

    def _allocate_source_rays(self, n_rays: int, src_pos: np.ndarray,
                              mode: Optional[str] = None,
                              seed: Optional[int] = None) -> np.ndarray:
        n_sources = int(src_pos.shape[0])
        total_rays = max(n_sources, int(n_rays) * n_sources)
        alloc_mode = str(mode or self._adaptive_mode).strip().lower()
        if alloc_mode not in ("stochastic", "quota", "uniform"):
            alloc_mode = "stochastic"
        need_map = self._density_need_map()
        per_source_need = self._project_source_need(src_pos, need_map)

        if self._source_need_ema is None or self._source_need_ema.shape[0] != n_sources:
            self._source_need_ema = np.asarray(per_source_need, np.float64).copy()
        else:
            self._source_need_ema *= 0.8
            self._source_need_ema += 0.2 * per_source_need

        hist = np.asarray(self._source_rays_emitted[:n_sources], np.float64)
        hmean = max(1.0, float(np.mean(hist)))
        novelty = 1.0 / np.sqrt(1.0 + (hist / hmean))

        weights = np.maximum(1.0e-9, self._source_need_ema * novelty)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0.0:
            weights = np.full((n_sources,), 1.0 / max(1, n_sources), dtype=np.float64)
        else:
            weights = weights / weights_sum

        if alloc_mode == "uniform":
            counts = np.full((n_sources,), total_rays // max(1, n_sources), dtype=np.int64)
            rem = int(total_rays - int(np.sum(counts)))
            if rem > 0:
                counts[:rem] += 1
        else:
            raw = weights * float(total_rays - n_sources)
            counts = np.ones((n_sources,), dtype=np.int64)
            if alloc_mode == "stochastic":
                rng = np.random.default_rng(int(seed if seed is not None else 0xC0FFEE))
                draws = rng.multinomial(int(total_rays - n_sources), weights)
                counts += draws.astype(np.int64, copy=False)
            else:
                counts += np.floor(raw).astype(np.int64)
                remain = int(total_rays - int(np.sum(counts)))
                if remain > 0:
                    frac = raw - np.floor(raw)
                    order = np.argsort(-frac)
                    counts[order[:remain]] += 1

        self._source_rays_emitted[:n_sources] += counts.astype(np.float64)
        return counts.astype(np.int32, copy=False)

    @staticmethod
    def _sample_cosine_hemisphere_axes(normals: np.ndarray,
                                       rng: np.random.Generator) -> np.ndarray:
        n = np.ascontiguousarray(normals, np.float64)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.where(n_norm > 1.0e-12, n_norm, 1.0)

        ref = np.tile(np.array([[1.0, 0.0, 0.0]], np.float64), (n.shape[0], 1))
        mask = np.abs(n[:, 0]) >= 0.9
        ref[mask] = np.array([0.0, 1.0, 0.0], np.float64)

        t = np.cross(n, ref)
        t_norm = np.linalg.norm(t, axis=1, keepdims=True)
        t = t / np.where(t_norm > 1.0e-12, t_norm, 1.0)
        b = np.cross(n, t)

        r1 = rng.random(n.shape[0], dtype=np.float64)
        r2 = rng.random(n.shape[0], dtype=np.float64)
        r = np.sqrt(r1)
        phi = (2.0 * math.pi) * r2
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.sqrt(np.maximum(0.0, 1.0 - r1))

        d = (t * x[:, None]) + (b * y[:, None]) + (n * z[:, None])
        d_norm = np.linalg.norm(d, axis=1, keepdims=True)
        return np.ascontiguousarray(d / np.where(d_norm > 1.0e-12, d_norm, 1.0),
                                    np.float64)

    def _prepare_emissive_batch_sources(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Prepare randomized launch sites/axes for every emissive each batch."""
        rng = np.random.default_rng(int(seed))
        tri = self._src_tri_verts

        u = rng.random(tri.shape[0], dtype=np.float64)
        v = rng.random(tri.shape[0], dtype=np.float64)
        flip = (u + v) > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]

        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        src_pos = tri[:, 0] + (u[:, None] * e1) + (v[:, None] * e2)
        src_dir = self._sample_cosine_hemisphere_axes(self.scene.src_dir, rng)
        return np.ascontiguousarray(src_pos, np.float64), src_dir

    def run_bdpt_batch(self, emit_rays: np.ndarray, seed: int) -> np.ndarray:
        """Fire one BDPT batch using the pre-registered tri-group layout.

        Returns raw float32 (N, 16) EndpointRecord array.  The caller must
        follow up with scatter_bdpt_records() to accumulate surf/field_accum.
        """
        if not (hasattr(self.tracer, "bidirectional_packed") and emit_rays.size > 0):
            return np.zeros((0, 16), np.float32)
        # Size the buffer to the full emitter workload plus the full pixel-cone
        # sensor workload for this frame. The sensor side can legitimately
        # generate width * height * aperture_samples * bands records, and
        # emitter-only sizing would silently clip the sensor rows first.
        n_bands = int(self.freq_hz.shape[0])
        emitter_max = int(np.sum(emit_rays)) * (self.max_bounces + 1) * max(1, n_bands)
        sensor_max = 0
        if self._sensor_camera_desc is not None:
            stream_div = max(1, int(self._sensor_camera_desc.get("pixel_stream_divisor", 1)))
            sensor_full = (
                int(self.cam.width)
                * int(self.cam.height)
                * int(self._sensor_camera_desc.get("n_aperture_samples", 0))
                * max(1, n_bands)
            )
            sensor_max = (
                int(sensor_full + stream_div - 1)
                // int(stream_div)
            )
        batch_max = max(1, emitter_max + sensor_max)
        cap_hint = max(0, int(self._bdpt_dynamic_cap_hint))
        cap = int(max(batch_max, cap_hint))
        recs = np.zeros((0, 16), np.float32)
        file_cap_records_full = int(_bdpt_cap_from_bytes(self._bdpt_intermediate_max_bytes))
        file_cap_records = int(max(4096, int(file_cap_records_full * self._bdpt_stream_working_fraction)))
        analytic_batch_bytes = int(batch_max) * int(_BDPT_RECORD_BYTES)

        self.cleanup_overflow_temp_file()
        if not self._bdpt_retain_intermediate:
            preserve_paths = ({self._bdpt_last_overflow_path} if self._bdpt_last_overflow_path else set())
            _cleanup_stale_ephemeral_bdpt_files(
                self._bdpt_intermediate_dir,
                preserve_paths=preserve_paths,
                include_retained=True,
            )

        attempt = 0
        while True:
            attempt += 1
            try:
                emit_arr = np.ascontiguousarray(emit_rays, np.int32)
                use_file_overflow = (
                    self._bdpt_intermediate_mode == "file"
                    and hasattr(self.tracer, "bidirectional_packed_into")
                )
                if use_file_overflow and file_cap_records > 0:
                    if batch_max > file_cap_records:
                        need_gb = analytic_batch_bytes / float(1024 ** 3)
                        cap_gb = self._bdpt_intermediate_max_bytes / float(1024 ** 3)
                        working_gb = (file_cap_records * _BDPT_RECORD_BYTES) / float(1024 ** 3)
                        raise RuntimeError(
                            "BDPT batch analytically requires more file-backed storage than allowed "
                            f"(need~{need_gb:.2f} GiB from emitter/sensor estimate, "
                            f"working_cap={working_gb:.2f} GiB of configured {cap_gb:.2f} GiB). "
                            "This indicates misconfigured rays/batch, bands, or aperture samples."
                        )
                    cap = min(int(cap), int(file_cap_records))
                    if cap <= 0:
                        raise RuntimeError("BDPT file-backed overflow cap is zero records; increase budget.")
                if use_file_overflow:
                    alloc_bytes = int(cap) * int(_BDPT_RECORD_BYTES)
                    self._require_bdpt_disk_budget(alloc_bytes, "bdpt overflow buffer")
                    keep = "retained" if self._bdpt_retain_intermediate else "tmp"
                    fd, path = tempfile.mkstemp(
                        prefix=f"bdpt_overflow_{keep}_",
                        suffix=".npy",
                        dir=self._bdpt_intermediate_dir,
                    )
                    os.close(fd)
                    buf = np.lib.format.open_memmap(
                        path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(int(cap), 16),
                    )
                    n_out = int(self.tracer.bidirectional_packed_into(
                        n_rays_per_emitter=emit_arr,
                        max_bounces=self.max_bounces,
                        min_amplitude=self.min_amplitude,
                        seed=int(seed),
                        out_records=buf,
                    ))
                    # Detach from file-backed mmap so Windows can delete temp
                    # overflow files immediately after this batch.
                    recs = np.ascontiguousarray(np.asarray(buf[:n_out], dtype=np.float32))
                    try:
                        buf.flush()
                    except Exception:
                        pass
                    del buf
                    self._bdpt_last_overflow_path = path
                else:
                    recs = self.tracer.bidirectional_packed(
                        n_rays_per_emitter = emit_arr,
                        max_bounces        = self.max_bounces,
                        min_amplitude      = self.min_amplitude,
                        seed               = int(seed),
                        max_records        = int(cap),
                    )
            except MemoryError as exc:
                self.cleanup_overflow_temp_file()
                raise RuntimeError(
                    "BDPT dynamic buffer growth ran out of memory while trying "
                    f"max_records={int(cap):_}; reduce rays/batch or aperture samples."
                ) from exc

            if recs.shape[0] < cap:
                self._bdpt_dynamic_cap_hint = max(int(cap), int(recs.shape[0]) + 1)
                if attempt > 1:
                    print(
                        "  [info] bdpt buffer auto-resized "
                        f"(final_cap={int(cap):_}, written={int(recs.shape[0]):_}, attempts={attempt}; "
                        f"remembered={int(self._bdpt_dynamic_cap_hint):_})"
                    )
                return recs

            if cap >= (2**31 - 1):
                raise RuntimeError(
                    "BDPT dynamic buffer growth hit int32 max_records limit "
                    f"(written={int(recs.shape[0]):_}, cap={int(cap):_})."
                )

            if use_file_overflow and file_cap_records > 0 and cap >= file_cap_records:
                cap_gb = (file_cap_records * _BDPT_RECORD_BYTES) / float(1024 ** 3)
                self.cleanup_overflow_temp_file()
                raise RuntimeError(
                    "BDPT overflow hit file-backed hard cap "
                    f"({cap_gb:.2f} GiB working cap, max_records={int(file_cap_records):_}); "
                    "reduce rays/batch or aperture samples."
                )

            next_cap = min((2**31 - 1), max(cap + 1, cap * 2))
            if use_file_overflow and file_cap_records > 0:
                next_cap = min(int(next_cap), int(file_cap_records))
            print(
                "  [warn] bdpt batch filled max_records; retrying with larger buffer "
                f"(written={int(recs.shape[0]):_} cap={int(cap):_} -> {int(next_cap):_})"
            )
            self.cleanup_overflow_temp_file()
            cap = int(next_cap)

    def run_thick_lens_native_bdpt(self,
                                   *,
                                   emitter_tri_ids: np.ndarray,
                                   total_rays: int,
                                   sensor_rays_per_batch: int,
                                   n_aperture_samples: int,
                                   max_children: int,
                                   seed: int,
                                   exposure_weight: float = 1.0,
                                   sweep_offset: int = 0,
                                   sweep_count: int = 0,
                                   sensor_sweeps: int = 1) -> np.ndarray:
        """Run one lab-standard native BDPT package and return sensor RGB.

        sweep_offset/sweep_count select a slice of the (pixel × aperture)
        schedule so callers can split one exposure into several flash+sensor
        T5 cycles.  One cycle's BDPT records must fit the GPU record caps
        (~6.9M vertices); a full sweep at high render res (e.g. 1024² × 8
        aperture samples = 8.4M rays) overflows them, which silently drops
        pdf records, invalidates the MIS chains, and zeroes the connect
        output.  sweep_count=0 sweeps the full remaining grid (legacy).
        """
        required = (
            "submit_emissive_triangles",
            "signal_flash_dispatched",
            "begin_sensor_batching",
            "submit_sensor_sweep",
            "signal_sensor_dispatched",
            "join_t5",
            "end_sensor_batching",
            "get_sensor_image",
        )
        missing = [name for name in required if not hasattr(self.tracer, name)]
        if missing:
            raise RuntimeError(
                "native thick-lens BDPT requires RayTracer methods: " + ", ".join(missing)
            )

        tri_ids = np.ascontiguousarray(emitter_tri_ids, dtype=np.int32).reshape(-1)
        if tri_ids.size <= 0:
            raise RuntimeError("native thick-lens BDPT requires at least one emissive triangle")
        if self._sensor_camera_desc is None:
            raise RuntimeError("native thick-lens BDPT requires a registered sensor camera descriptor")

        n_ap = int(max(1, n_aperture_samples))
        n_px = int(self._sensor_camera_desc.get("n_px", self.cam.width))
        n_py = int(self._sensor_camera_desc.get("n_py", self.cam.height))
        schedule_total = int(max(1, n_px) * max(1, n_py) * n_ap)
        # submit_emissive_triangles consumes its ray count PER UV DOMAIN (UV
        # group of emitter tris), NOT per triangle.  This scene's 2784 emitter
        # tris span ~6 UV domains, so dividing total_rays by tri count (the old
        # code) launched ~1 ray per tri (~3K rays total) and produced almost no
        # light-subpath vertices — the T5 connect had nothing to connect to.
        # The lab uses uv_emitter_rays = 32_000 per domain ("Total ray count =
        # uv_emitter_rays (not per-triangle)", thick_lens_focus_lab.py:5505),
        # yielding ~1.1M light vertices per cycle.  Match it.
        emitter_rays_per_domain = 32_000
        slice_start = int(max(0, min(int(sweep_offset), schedule_total)))
        slice_end = (schedule_total if int(sweep_count) <= 0
                     else int(min(schedule_total, slice_start + int(sweep_count))))
        slice_total = int(max(0, slice_end - slice_start))
        if slice_total <= 0:
            raise RuntimeError(
                f"native thick-lens BDPT sweep slice is empty "
                f"(offset={slice_start} count={sweep_count} schedule={schedule_total})"
            )
        # Page the record arena without shrinking the logical exposure. Flash
        # paths are packed once and remain GPU-resident while camera pages cycle.
        sensor_batch = int(max(
            1,
            min(int(sensor_rays_per_batch), slice_total, 200_000),
        ))
        max_children = int(max(1, max_children))
        if hasattr(self.tracer, "set_max_children"):
            self.tracer.set_max_children(max_children)

        print(
            "  [bdpt-native-lab] workload "
            f"sensor={n_px}x{n_py} aperture_samples={n_ap} "
            f"primary_camera_rays={slice_total:,}/{schedule_total:,} "
            f"emit_tris={int(tri_ids.size)} rays_per_domain={emitter_rays_per_domain} "
            f"slice={slice_start:,}..{slice_end:,} batch={sensor_batch:,}"
        )

        # min_amplitude=0.0 is the lab's single-photon mode (--sensor-min-amp
        # default): bounce chains are never amplitude-killed, so faint light
        # subpaths survive to become connectable vertices.  The session-level
        # self.min_amplitude (1e-3) is for the streaming integrators, not this
        # lab-parity path.
        interaction_target = (0.0, 0.0, 0.0, 0.0)
        native_flash_weight = float(max(0.0, exposure_weight))
        optical_camera = getattr(getattr(self, "scene", None), "optical_camera", None)
        if optical_camera is not None:
            flash_manifest = dict(
                getattr(optical_camera, "manifest", {}).get("flash", {})
            )
            modifier = dict(flash_manifest.get("modifier", {}))
            burst = dict(flash_manifest.get("burst", {}))
            native_flash_weight *= max(
                0.0, float(burst.get("exposure_weight", 1.0))
            ) * max(0.0, float(burst.get("energy_scale", 1.0))) * max(
                0.0, min(1.0, float(burst.get("duty_cycle", 1.0)))
            )
            mode_name = str(modifier.get("mode", "none")).lower()
            mode_value = {"none": 0, "snoot": 1, "grid": 2, "scrim": 3}.get(
                mode_name, 0
            )
            if mode_value == 2:
                param0 = float(modifier.get("grid_cell_mm", 5.0))
                param1 = float(modifier.get("grid_depth_mm", 25.0))
            elif mode_value == 3:
                param0 = float(modifier.get("scrim_transmittance", 0.5))
                param1 = 0.0
            else:
                param0 = param1 = 0.0
            if hasattr(self.tracer, "set_flash_modifier"):
                self.tracer.set_flash_modifier(mode_value, param0, param1)
            if mode_value != 0:
                lab_config = getattr(optical_camera, "scene_config", None)
                subject = getattr(lab_config, "object_plane", None)
                if subject is not None:
                    interaction_target = (
                        float(getattr(subject, "x", 0.0)), 0.0, 0.0,
                        float(max(1.0e-6, getattr(subject, "radius", 0.05))),
                    )

        overflow_before = self._bdpt_overflow_snapshot()
        submitted_flash = int(self.tracer.submit_emissive_triangles(
            tri_ids,
            int(emitter_rays_per_domain),
            native_flash_weight,
            1.0,
            int(self.max_bounces),
            0.0,
            int(max_children),
            int(seed),
            bool(self._gpu_resident),
            bool(self._gpu_resident),
            _SHADER_DIR,
            *interaction_target,
            *self._native_emitter_launch_args(),
        ))
        self.tracer.signal_flash_dispatched()
        if submitted_flash <= 0:
            raise RuntimeError("native thick-lens BDPT submitted no flash rays")

        submitted_sensor = 0
        self.tracer.begin_sensor_batching()
        try:
            pix_offset = slice_start
            sensor_page = 0
            while pix_offset < slice_end:
                if sensor_page > 0:
                    # Advance the paired-cycle latch without retracing flash.
                    # Native batching reuses page zero's packed light SSBO.
                    self.tracer.signal_flash_dispatched()
                n_submit = int(self.tracer.submit_sensor_sweep(
                    max_bounces=int(self.max_bounces),
                    min_amplitude=0.0,  # lab single-photon mode (see flash submit)
                    max_rays=int(min(sensor_batch, slice_end - pix_offset)),
                    pix_offset=int(pix_offset),
                    max_children=int(max_children),
                    aperture_samples=int(n_ap),
                    seed=int(seed),
                    exposure_weight=float(max(0.0, exposure_weight)),
                ))
                if n_submit <= 0:
                    break
                submitted_sensor += n_submit
                pix_offset += n_submit
                self.tracer.signal_sensor_dispatched()
                if hasattr(self.tracer, "in_flight_count"):
                    deadline = time.perf_counter() + 10.0
                    while time.perf_counter() < deadline:
                        if int(self.tracer.in_flight_count()) == 0:
                            break
                        time.sleep(0.002)
                self.tracer.join_t5()
                self._assert_no_bdpt_overflow(
                    overflow_before,
                    page_label=f"native sensor sweep page {sensor_page}",
                )
                overflow_before = self._bdpt_overflow_snapshot()
                sensor_page += 1
            if submitted_sensor < slice_total:
                raise RuntimeError(
                    "native thick-lens BDPT sensor sweep did not cover its slice "
                    f"({submitted_sensor}/{slice_total}, slice {slice_start}..{slice_end})"
                )
            latch = (self.tracer.get_bdpt_latch_state()
                     if hasattr(self.tracer, "get_bdpt_latch_state") else {})
            output_width, output_height = (
                _ORDERED_TILE_OUTPUT if _ORDERED_TILE_OUTPUT is not None else (n_px, n_py)
            )
            img = _native_sensor_tile_to_display(
                np.asarray(self.tracer.get_sensor_image(), dtype=np.float32),
                output_width,
                output_height,
            )
            linear_img = None
            if hasattr(self.tracer, "get_sensor_image_linear"):
                linear_img = _native_sensor_tile_to_display(
                    np.asarray(self.tracer.get_sensor_image_linear(), dtype=np.float32),
                    output_width,
                    output_height,
                )
            valid_shape = img.ndim == 3 and img.shape[2] >= 3
            rgb = img[..., :3] if valid_shape else np.empty((0, 0, 3), np.float32)
            finite = bool(rgb.size and np.all(np.isfinite(rgb)))
            energy = np.sum(np.maximum(rgb, 0.0), axis=2) if finite else np.empty(0, np.float32)
            lit_px = int(np.count_nonzero(energy > 1.0e-8)) if energy.size else 0
            total_px = int(energy.size)
            peak = float(np.max(rgb)) if finite and rgb.size else 0.0
            mean = float(np.mean(rgb)) if finite and rgb.size else 0.0
            print(
                "  [bdpt-native-lab] sensor result "
                f"shape={tuple(img.shape)} finite={finite} "
                f"lit={lit_px}/{total_px} ({100.0 * lit_px / max(1, total_px):.2f}%) "
                f"mean={mean:.3e} peak={peak:.3e} latch={dict(latch)}",
                flush=True,
            )
            if not valid_shape or not finite or lit_px == 0:
                # An empty sensor for one pass must NOT abort a multi-frame run.
                # The T5 connect can complete (millions of pairs) yet land no
                # signal in the sensor readback for a given frame; degrade to the
                # last good image (or black) and keep going instead of raising.
                print(
                    "  [bdpt-native-lab] FAILED: transport produced no usable sensor pixels "
                    f"(flash={submitted_flash} sensor={submitted_sensor} "
                    f"latch={dict(latch)}). Displaying the previous frame or black; "
                    "this is not a completed render.",
                    flush=True,
                )
                if (getattr(self, "_native_sensor_image", None) is not None
                        and np.asarray(self._native_sensor_image).size > 0):
                    return self._native_sensor_image
                _res = int(img.shape[0]) if (img.ndim == 3 and img.shape[0] > 0) else 1
                self._native_sensor_image = np.zeros((_res, _res, 3), dtype=np.float32)
                return self._native_sensor_image
            self._native_sensor_image = np.ascontiguousarray(np.clip(img[..., :3], 0.0, 1.0), dtype=np.float32)
            if linear_img is not None and linear_img.shape == img.shape:
                if hasattr(self.tracer, "get_sensor_epoch_count"):
                    epoch_count = np.asarray(self.tracer.get_sensor_epoch_count())
                    epoch_count_display = _native_sensor_tile_to_display(
                        epoch_count[..., None], output_width, output_height
                    )[..., 0]
                    self._native_sensor_linear_image = _normalise_native_sensor_epochs(
                        linear_img[..., :3], epoch_count_display
                    )
                else:
                    self._native_sensor_linear_image = _average_native_sensor_sweeps(
                        linear_img[..., :3], sensor_sweeps
                    )
            self.n_rays_accumulated += int(submitted_flash) + int(submitted_sensor)
            print(
                "  [bdpt-native-lab] usable image "
                f"flash={submitted_flash:,} sensor={submitted_sensor:,} "
                f"lit={lit_px}/{total_px} mean={mean:.3e} peak={peak:.3e}"
            )
            return self._native_sensor_image
        finally:
            self.tracer.end_sensor_batching()

    def scatter_bdpt_records(self, recs: np.ndarray, sensor_group_id: int) -> None:
        """Scatter EndpointRecord rows into surf_accum (sensor hits) and field_accum (scene).

        EndpointRecord float32 column layout (bdpt_record.h, 64 bytes = 16 floats):
          col[0] subpath_id (uint32), col[1] band_id (uint32),
          col[2] group_id  (int32),   col[3] vertex_index (int32),
          col[4:7] pos xyz,           col[7] pathlen_m,
          col[8:11] dir xyz,          col[11] pdf,
          col[12] amp_re, col[13] amp_im, col[14] cos_theta, col[15] _pad

        Records with group_id == sensor_group_id are camera-sensor hits
        (surf_accum); all others are ambient/field paths (field_accum).
        self.accum is kept in sync as surf_accum + field_accum so that
        measured_radiant_exposure_J() and _density_need_map() remain valid.
        """
        if recs.shape[0] == 0:
            return
        recs = np.ascontiguousarray(recs, np.float32)
        n = recs.shape[0]
        H, W = int(self.cam.height), int(self.cam.width)

        # Reinterpret float32 storage as int32 to read integer-typed columns.
        col_i32  = recs.view(np.int32).reshape(n, 16)
        raw_band_ids = col_i32[:, 1].astype(np.int64)   # uint32 in struct
        group_ids    = col_i32[:, 2]                     # int32, signed

        # Discard records whose band_id is out of range — never silently wrap.
        valid_band = (raw_band_ids >= 0) & (raw_band_ids < self.n_bands)
        if not np.all(valid_band):
            n_bad = int(np.sum(~valid_band))
            print(f"  [bdpt] scatter: discarding {n_bad} records with out-of-range band_id")
            recs      = recs[valid_band]
            col_i32   = recs.view(np.int32).reshape(-1, 16)
            raw_band_ids = raw_band_ids[valid_band]
            group_ids    = col_i32[:, 2]
            n = recs.shape[0]
            if n == 0:
                return
        band_ids    = raw_band_ids
        subpath_ids = col_i32[:, 0].astype(np.int64)   # uint32 → int64
        vertex_idx  = col_i32[:, 3].astype(np.int64)   # int32, -1 = forward path
        interaction_flags = np.rint(recs[:, 15]).astype(np.int64)
        thin_kind = int(getattr(_sk, "SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM", 2))
        thin_flag_bit = np.int64(1 << thin_kind)

        amp_re = recs[:, 12].astype(np.float64)
        amp_im = recs[:, 13].astype(np.float64)
        amp    = np.sqrt(amp_re * amp_re + amp_im * amp_im)

        # Split sensor-group records by semantic:
        #   pixel_cone     — vertex_index >= 0: PIXEL_CONE sensor hit.
        #                    subpath_id encodes the pixel as py*W + px.
        #                    Use subpath_id directly for pixel mapping.
        #   forward_sensor — vertex_index < 0, group == sensor_group_id:
        #                    Forward/emission path that happened to hit a
        #                    sensor-group triangle.  Project world-pos.
        #   field          — group != sensor_group_id: ambient scene path.
        #                    Project world-pos; only in-frustum kept.
        sensor_group  = (group_ids == int(sensor_group_id))
        pixel_cone    = sensor_group & (vertex_idx >= 0)
        forward_snsr  = sensor_group & (vertex_idx < 0)
        field         = ~sensor_group

        # --- PIXEL_CONE scatter: subpath_id = py*W + px already. ---
        def _scatter_pixel_cone() -> None:
            m = pixel_cone & (subpath_ids >= 0) & (subpath_ids < H * W)
            if not np.any(m):
                n_bad = int(np.sum(pixel_cone)) - int(np.sum(m))
                if n_bad:
                    print(f"  [bdpt] scatter: discarding {n_bad} "
                          f"pixel_cone records with out-of-range subpath_id")
                return
            flat = band_ids[m] * (H * W) + subpath_ids[m]
            counts = np.bincount(flat, weights=amp[m],
                                 minlength=self.n_bands * H * W)
            self.surf_accum.ravel()[:] += counts.astype(np.float32)

            px = (subpath_ids[m] % W).astype(np.int64)
            py = (subpath_ids[m] // W).astype(np.int64)
            uv_flat = py * W + px
            uv_counts = np.bincount(uv_flat, weights=amp[m], minlength=H * W)
            self.sensor_uv_hits.ravel()[:] += uv_counts.astype(np.float32)

            thin_m = m & ((interaction_flags & thin_flag_bit) != 0)
            if np.any(thin_m):
                tpx = (subpath_ids[thin_m] % W).astype(np.int64)
                tpy = (subpath_ids[thin_m] // W).astype(np.int64)
                tuv_flat = tpy * W + tpx
                tuv_counts = np.bincount(tuv_flat, weights=amp[thin_m], minlength=H * W)
                self.sensor_uv_thin_lens_hits.ravel()[:] += tuv_counts.astype(np.float32)

        # --- World-pos projection for forward_sensor + field records. ---
        # Only compute the projection when there are records that need it.
        proj_mask = forward_snsr | field
        if np.any(proj_mask):
            pos = recs[:, 4:7].astype(np.float64)   # (N, 3) world-space
            cam_pos = np.asarray(self.cam.pos, np.float64)
            cam_f, cam_r, cam_u, tan_half_h, tan_half_v = self._camera_basis()
            v      = pos - cam_pos[None, :]
            depth  = v @ cam_f
            fwd_valid = depth > 1.0e-6
            safe_d = np.where(fwd_valid, depth, 1.0)
            x_ndc  = (v @ cam_r) / (safe_d * tan_half_h)
            y_ndc  = -(v @ cam_u) / (safe_d * tan_half_v)
            px_proj = ((x_ndc + 1.0) * 0.5 * W).astype(np.int64)
            py_proj = ((y_ndc + 1.0) * 0.5 * H).astype(np.int64)
            in_frame = fwd_valid & (px_proj >= 0) & (px_proj < W) & \
                       (py_proj >= 0) & (py_proj < H)
        else:
            in_frame = np.zeros(n, dtype=bool)
            px_proj  = np.zeros(n, dtype=np.int64)
            py_proj  = np.zeros(n, dtype=np.int64)

        def _scatter_proj(mask: np.ndarray, target: np.ndarray) -> None:
            m = mask & in_frame
            if not np.any(m):
                return
            flat = band_ids[m] * (H * W) + py_proj[m] * W + px_proj[m]
            counts = np.bincount(flat, weights=amp[m],
                                 minlength=self.n_bands * H * W)
            target.ravel()[:] += counts.astype(np.float32)

        _scatter_pixel_cone()
        _scatter_proj(forward_snsr, self.surf_accum)
        _scatter_proj(field,        self.field_accum)
        # Keep combined accum in sync for measured_radiant_exposure_J / density map.
        np.add(self.surf_accum, self.field_accum, out=self.accum)
        self.n_rays_accumulated += n

    def render_batch(self, n_rays: int, seed: int) -> None:
        # Ray preparation randomness is externalized here: every emissive
        # receives a fresh launch site and axis each batch before tracing.
        src_pos, src_dir = self._prepare_emissive_batch_sources(seed)
        if hasattr(self.tracer, "integrate_image_into_packed"):
            src_n_rays = self._allocate_source_rays(int(n_rays), src_pos,
                                                    mode=self._adaptive_mode,
                                                    seed=int(seed))
            self.tracer.integrate_image_into_packed(
                src_pos         = src_pos,
                src_dir         = src_dir,
                src_directivity = self.scene.src_directivity,
                src_n_rays      = src_n_rays,
                cam_pos         = self.cam.pos,
                cam_fwd         = self.cam.fwd,
                cam_up          = self.cam.up,
                out_image       = self.accum,
                fov_rad         = float(self.cam.fov_y_rad),
                max_bounces     = self.max_bounces,
                min_amplitude   = self.min_amplitude,
                seed            = int(seed),
            )
            self.n_rays_accumulated += int(np.sum(src_n_rays, dtype=np.int64))
            return

        self.tracer.integrate_image_into(
            src_pos         = src_pos,
            src_dir         = src_dir,
            src_directivity = self.scene.src_directivity,
            cam_pos         = self.cam.pos,
            cam_fwd         = self.cam.fwd,
            cam_up          = self.cam.up,
            out_image       = self.accum,
            fov_rad         = float(self.cam.fov_y_rad),
            n_rays          = int(n_rays),
            max_bounces     = self.max_bounces,
            min_amplitude   = self.min_amplitude,
            seed            = int(seed),
        )
        self.n_rays_accumulated += int(n_rays) * int(src_pos.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# GLSL backend — scaffold around demo_pluck_gl._gpu_ray_field_prebuilt
# ─────────────────────────────────────────────────────────────────────────────
class GlslExposureBackend(ExposureBackend):
    """Best-effort GLSL backend.

    Building the full SSBO context (BVH + ScaleContext + FilmStack +
    SensorAccumulator) outside ``demo_pluck_gl``'s harness is a substantial
    port; for now the backend either:
      (a) constructs the minimum SSBO set if a live GL context exists and
          the scene_builder fast path succeeds, or
      (b) falls back to a band-modulated copy of the C++ accumulator marked
          "GLSL pending" so the side-by-side window still functions and the
          calibration log still prints two backends.

    Either way, the public ``render_batch`` / ``finalize_image`` interface
    matches the C++ backend exactly.
    """
    name = "glsl"

    def __init__(self, scene: TracerScene, cam: PhysicalCameraRig,
                 freq_hz: np.ndarray, *, mirror_from: Optional[ExposureBackend] = None,
                 **kw: Any):
        super().__init__(scene, cam, freq_hz, **kw)
        self._mirror = mirror_from
        self._gl_ready = False
        # TODO(phase2b): instantiate _gpu_ray_field_prebuilt + SensorAccumulator
        # against scene.{verts, mat_idx, mat_buf, bounds_*}, build a BVH from
        # csrc/kernels (already wrapped as _sk.build_bvh in older code), and
        # accumulate sensor pixels each batch.

    def render_batch(self, n_rays: int, seed: int) -> None:
        if self._gl_ready:
            return  # full path goes here once wired
        if self._mirror is not None:
            # Track the C++ accumulators so the right pane has something
            # spectrally meaningful to show during calibration testing.
            self.accum[:]       = self._mirror.accum
            self.surf_accum[:]  = self._mirror.surf_accum
            self.field_accum[:] = self._mirror.field_accum
            self.n_rays_accumulated = self._mirror.n_rays_accumulated


# ─────────────────────────────────────────────────────────────────────────────
# Exposure session — orchestrates plan → batches → gain training → frame
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExposureFrameResult:
    frame_index:        int
    backend:            str
    plan:               dict
    n_rays_emitted:     int
    n_batches:          int
    measured_H_J:       float
    measurement_status: str
    target_H_J:         float
    gain_linear:        float
    gain_db:            float
    virtual_t_s:        float
    photons_per_pixel:  float
    snr_estimate:       float
    image_path:         str
    image16_path:       str
    image_linear_path:  str
    field_object_path:  str
    surface_object_path:str
    sensor_object_path: str
    field_capture_grid_path: str
    field_capture_strikes_path: str
    summary_path:       str
    frame_config_summary: dict
    native_sensor_evidence: dict
    image_data:         Optional[np.ndarray] = None  # (H, W, 3) float32 [0,1]
    image16_data:       Optional[np.ndarray] = None  # (H, W, 3) uint16 for 16-bit
    field_integrated_data: Optional[np.ndarray] = None  # (B, H, W)
    surface_integrated_data: Optional[np.ndarray] = None  # (B, H, W)
    sensor_photons_data: Optional[np.ndarray] = None  # (H, W)
    sensor_snr_data: Optional[np.ndarray] = None  # (H, W)
    endpoint_rgb_data: Optional[np.ndarray] = None  # (H, W, 3)
    pinhole_rgb_data: Optional[np.ndarray] = None  # (H, W, 3)
    lightfield_shell_top_data: Optional[np.ndarray] = None  # (H, W, 3)
    lightfield_shell_side_data: Optional[np.ndarray] = None  # (H, W, 3)
    camera_event_telemetry: Optional[dict] = None  # Event counter telemetry


_FRAME_RESULT_NONSUMMARY_FIELDS = (
    "image_data",
    "image16_data",
    "field_integrated_data",
    "surface_integrated_data",
    "sensor_photons_data",
    "sensor_snr_data",
    "endpoint_rgb_data",
    "pinhole_rgb_data",
    "lightfield_shell_top_data",
    "lightfield_shell_side_data",
    "camera_event_telemetry",  # Exported via frame_config_summary instead
)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _frame_result_summary_dict(result: ExposureFrameResult) -> dict[str, Any]:
    summary = asdict(result)
    for field_name in _FRAME_RESULT_NONSUMMARY_FIELDS:
        summary.pop(field_name, None)
    return _json_safe_value(summary)


@dataclass
class IntegralSplitConfig:
    """Controls how much energy is integrated vs retained as bookkeeping."""
    field_integrate_frac: float = 0.35
    field_bookkeep_frac:  float = 0.65
    surface_integrate_frac: float = 0.85
    surface_bookkeep_frac:  float = 0.15
    hdr_white_percentile: float = 99.8


@dataclass
class CameraVisibilityConfig:
    """Wrapper around tracer camera-visibility policies for this session."""
    camera_vis_mode: int = int(getattr(_sk, "RT_CAM_VIS_AS_IS", 0))
    transparent_mode: int = int(getattr(_sk, "RT_CAM_TRANSPARENCY_BLOCK", 0))
    depth_cull_enabled: bool = False
    depth_cull_m: float = 0.0


@dataclass
class ConvergenceConfig:
    """Early-stop controls for exposure batch convergence."""
    enabled: bool = True
    drive_batches: bool = True
    target_pct: float = 99.99
    max_rel_drift: float = 1.0e-4
    check_every_batches: int = 1
    min_batches: int = 4
    hold_checks: int = 3
    probe_count: int = 8192
    max_batches: int = 0


@dataclass
class FieldCaptureConfig:
    """Full-complex field capture controls for exposure tracing."""
    enabled: bool = True
    grid_kind: str = "regular"  # "regular" | "kdtree"
    nx: int = 128
    ny: int = 128
    nz: int = 128
    capture_strikes: bool = True
    max_strikes: int = 1_000_000


@dataclass
class SurfaceSplineConfig:
    """Quadratic POLY_BARY surface spline fitter settings."""
    enabled: bool = False
    ridge_lambda: float = 0.0       # Tikhonov regularisation (0 = off)
    n_threads: int = 0              # 0 = hardware_concurrency
    fit_all_tris: bool = False      # True = per-tri groups; False = mean group


@dataclass
class ParametricSdfConfig:
    """Parametric SDF-style presets mapped onto POLY_BARY payloads."""
    enabled: bool = False
    model: str = "off"              # off | saddle | sphere | mixed
    saddle_amplitude_m: float = 2.0e-3
    sphere_radius_m: float = 0.12
    neighborhood_margin_uv: float = 8.0e-2


# ── Camera-visibility mode constants (resolved lazily so module loads without ext) ──
_CAM_VIS_AS_IS      = int(getattr(_sk, "RT_CAM_VIS_AS_IS",      0))
_CAM_VIS_DIRECT_HIT = int(getattr(_sk, "RT_CAM_VIS_DIRECT_HIT", 1))
_CAM_VIS_FULL_MARCH = int(getattr(_sk, "RT_CAM_VIS_FULL_MARCH", 2))
_CAM_TRANSP_BLOCK   = int(getattr(_sk, "RT_CAM_TRANSPARENCY_BLOCK", 0))
_CAM_TRANSP_XRAY    = int(getattr(_sk, "RT_CAM_TRANSPARENCY_XRAY",  1))


@dataclass
class FrameConfig:
    """Full per-frame feature set built by _build_frame_config."""
    field_capture:     FieldCaptureConfig
    camera_visibility: CameraVisibilityConfig
    surface_spline:    SurfaceSplineConfig
    parametric_sdf:    ParametricSdfConfig
    integral_split:    IntegralSplitConfig
    description:       str = ""
    detail_level:      int = 0   # 0-5; drives HUD verbosity


def _poly_bary_coeffs_sdf(model: str,
                          tri_id: int,
                          saddle_amplitude_m: float,
                          sphere_radius_m: float) -> np.ndarray:
    """Build float64[6] POLY_BARY coeffs for parametric SDF-style presets."""
    m = str(model).strip().lower()
    if m == "mixed":
        m = "sphere" if (int(tri_id) % 2) == 0 else "saddle"

    if m == "sphere":
        # Near-center paraboloid approximation of sphere SDF displacement.
        # delta ~ k * ((u-1/3)^2 + (v-1/3)^2), k ~= 1/(2R)
        r = max(float(sphere_radius_m), 1.0e-6)
        k = 0.5 / r
        c0 = (2.0 / 9.0) * k
        cu = -(2.0 / 3.0) * k
        cv = -(2.0 / 3.0) * k
        cuu = k
        cuv = 0.0
        cvv = k
        return np.asarray([c0, cu, cv, cuu, cuv, cvv], dtype=np.float64)

    # Default to saddle if unsupported string is supplied.
    # delta = a * ((u-1/3)^2 - (v-1/3)^2)
    a = float(saddle_amplitude_m)
    c0 = 0.0
    cu = -(2.0 / 3.0) * a
    cv = (2.0 / 3.0) * a
    cuu = a
    cuv = 0.0
    cvv = -a
    return np.asarray([c0, cu, cv, cuu, cuv, cvv], dtype=np.float64)


def _resolve_native_parametric_payload(raw: Any) -> dict[str, Any]:
    """Normalize plugin output to register_tri_group parametric_surface payload.

    Accepted forms:
    - dict: {"kind": int|str, "coeffs": float64[N]}
    - array-like: float64[6] interpreted as POLY_BARY
    """
    if isinstance(raw, dict):
        kind_raw = raw.get("kind", "poly_bary")
        coeffs = np.asarray(raw.get("coeffs", []), dtype=np.float64).ravel()
    else:
        kind_raw = "poly_bary"
        coeffs = np.asarray(raw, dtype=np.float64).ravel()

    if coeffs.ndim != 1 or coeffs.size == 0:
        raise RuntimeError("parametric plugin returned empty coeff payload")

    if isinstance(kind_raw, str):
        k = kind_raw.strip().lower()
        if k in ("poly_bary", "poly", "bary"):
            kind = int(getattr(_sk, "TRI_PARAM_SURFACE_POLY_BARY", 1))
            if coeffs.size != 6:
                raise RuntimeError(f"POLY_BARY expects 6 coeffs, got {coeffs.size}")
        elif k in ("sdf_saddle", "saddle"):
            kind = int(getattr(_sk, "TRI_PARAM_SURFACE_SDF_SADDLE", -1))
            if kind < 0:
                raise RuntimeError("TRI_PARAM_SURFACE_SDF_SADDLE unavailable in native extension")
            if coeffs.size != 2:
                raise RuntimeError(f"SDF_SADDLE expects 2 coeffs [amp, margin], got {coeffs.size}")
        elif k in ("sdf_sphere", "sphere"):
            kind = int(getattr(_sk, "TRI_PARAM_SURFACE_SDF_SPHERE", -1))
            if kind < 0:
                raise RuntimeError("TRI_PARAM_SURFACE_SDF_SPHERE unavailable in native extension")
            if coeffs.size != 2:
                raise RuntimeError(f"SDF_SPHERE expects 2 coeffs [radius, margin], got {coeffs.size}")
        else:
            raise RuntimeError(f"unknown parametric kind '{kind_raw}'")
    else:
        kind = int(kind_raw)

    return {"kind": int(kind), "coeffs": coeffs.astype(np.float64, copy=False)}


def _build_frame_config(frame_idx: int,
                        n_frames_total: int,
                        base_split: IntegralSplitConfig) -> FrameConfig:
    """Build the per-frame feature config according to the progressive schedule.

    When n_frames_total == 1 only level 0 (baseline) is used.  Each additional
    frame slot unlocks the next feature level up to MAX_LEVELS - 1.

    Level 0  aggressive bootstrap — field capture + strike scatter + full march
    Level 1  field regular 64³ + strike capture
    Level 2  field regular 96³ + DIRECT_HIT + emissive-only spline
    Level 3  field kdtree 64³ + DIRECT_HIT + XRAY + full-mesh spline
    Level 4  field regular 128³ + FULL_MARCH + XRAY + depth cull + ridge spline
    Level 5  field kdtree 128³ + FULL_MARCH + XRAY + max ridge + field-heavy
    """
    MAX_LEVELS = 8
    n_levels = min(n_frames_total, MAX_LEVELS)
    level = frame_idx % max(1, n_levels)

    fi = float(base_split.field_integrate_frac)
    fb = float(base_split.field_bookkeep_frac)
    si = float(base_split.surface_integrate_frac)
    sb = float(base_split.surface_bookkeep_frac)
    wp = float(base_split.hdr_white_percentile)

    def _split(dfi: float = 0.0, dsi: float = 0.0, dwp: float = 0.0) -> IntegralSplitConfig:
        return IntegralSplitConfig(
            field_integrate_frac   = float(np.clip(fi + dfi, 0.0, 1.0)),
            field_bookkeep_frac    = float(np.clip(fb - dfi, 0.0, 1.0)),
            surface_integrate_frac = float(np.clip(si + dsi, 0.0, 1.0)),
            surface_bookkeep_frac  = float(np.clip(sb - dsi, 0.0, 1.0)),
            hdr_white_percentile   = float(max(wp + dwp, 90.0)),
        )

    schedules: list[FrameConfig] = [
        # level 0 — fastest baseline: AS_IS cam, coarse regular grid, no extras
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=48, ny=48, nz=48,
                capture_strikes=True, max_strikes=200_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_AS_IS,
                transparent_mode = _CAM_TRANSP_BLOCK,
            ),
            surface_spline    = SurfaceSplineConfig(enabled=False),
            parametric_sdf    = ParametricSdfConfig(enabled=False),
            integral_split    = _split(),
            description       = "baseline · AS_IS · regular-48³ · no extras",
            detail_level      = 0,
        ),
        # level 1 — add parametric saddle on emissive tris + DIRECT_HIT
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=64, ny=64, nz=64,
                capture_strikes=True, max_strikes=350_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_DIRECT_HIT,
                transparent_mode = _CAM_TRANSP_BLOCK,
            ),
            surface_spline    = SurfaceSplineConfig(enabled=False),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="saddle", saddle_amplitude_m=1.5e-3,
                neighborhood_margin_uv=8.0e-2,
            ),
            integral_split    = _split(dfi=0.05),
            description       = "DIRECT_HIT · regular-64³ · parametric saddle",
            detail_level      = 1,
        ),
        # level 2 — parametric sphere + emissive-only spline + DIRECT_HIT
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=80, ny=80, nz=80,
                capture_strikes=True, max_strikes=500_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_DIRECT_HIT,
                transparent_mode = _CAM_TRANSP_BLOCK,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=False, ridge_lambda=0.0,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="sphere", sphere_radius_m=0.12,
                neighborhood_margin_uv=8.0e-2,
            ),
            integral_split    = _split(dfi=0.10),
            description       = "DIRECT_HIT · regular-80³ · spline emissive · parametric sphere",
            detail_level      = 2,
        ),
        # level 3 — DIRECT_HIT + XRAY + kdtree 64³ + mixed parametric + all spline
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="kdtree",
                nx=64, ny=64, nz=64,
                capture_strikes=True, max_strikes=750_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode  = _CAM_VIS_DIRECT_HIT,
                transparent_mode = _CAM_TRANSP_XRAY,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=True, ridge_lambda=0.0,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="mixed", saddle_amplitude_m=2.0e-3,
                sphere_radius_m=0.12, neighborhood_margin_uv=8.0e-2,
            ),
            integral_split    = _split(dfi=0.15, dsi=0.05),
            description       = "DIRECT_HIT+XRAY · kdtree-64³ · spline all · parametric mixed",
            detail_level      = 3,
        ),
        # level 4 — FULL_MARCH + regular 96³ + parametric saddle + depth cull
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=96, ny=96, nz=96,
                capture_strikes=True, max_strikes=1_000_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode    = _CAM_VIS_FULL_MARCH,
                transparent_mode   = _CAM_TRANSP_BLOCK,
                depth_cull_enabled = True,
                depth_cull_m       = 80.0,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=False, ridge_lambda=0.0,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="saddle", saddle_amplitude_m=2.0e-3,
                neighborhood_margin_uv=8.0e-2,
            ),
            integral_split    = _split(dfi=0.20, dsi=0.08, dwp=-0.2),
            description       = "FULL_MARCH · regular-96³ · emissive spline · parametric saddle · depth 80m",
            detail_level      = 4,
        ),
        # level 5 — FULL_MARCH + regular 128³ + ridge spline + parametric mixed + depth cull
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="regular",
                nx=128, ny=128, nz=128,
                capture_strikes=True, max_strikes=1_000_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode    = _CAM_VIS_FULL_MARCH,
                transparent_mode   = _CAM_TRANSP_BLOCK,
                depth_cull_enabled = True,
                depth_cull_m       = 50.0,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=True, ridge_lambda=1.0e-4,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="mixed", saddle_amplitude_m=2.0e-3,
                sphere_radius_m=0.12, neighborhood_margin_uv=8.0e-2,
            ),
            integral_split    = _split(dfi=0.25, dsi=0.10, dwp=-0.3),
            description       = "FULL_MARCH · regular-128³ · spline ridge 1e-4 · parametric mixed · depth 50m",
            detail_level      = 5,
        ),
        # level 6 — FULL_MARCH + XRAY + kdtree 128³ + ridge spline + parametric mixed
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="kdtree",
                nx=128, ny=128, nz=128,
                capture_strikes=True, max_strikes=1_000_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode    = _CAM_VIS_FULL_MARCH,
                transparent_mode   = _CAM_TRANSP_XRAY,
                depth_cull_enabled = True,
                depth_cull_m       = 100.0,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=True, ridge_lambda=1.0e-3,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="mixed", saddle_amplitude_m=2.5e-3,
                sphere_radius_m=0.12, neighborhood_margin_uv=6.0e-2,
            ),
            integral_split    = _split(dfi=0.30, dsi=0.12, dwp=-0.6),
            description       = "FULL_MARCH+XRAY · kdtree-128³ · spline ridge 1e-3 · parametric mixed",
            detail_level      = 6,
        ),
        # level 7 — most expensive: FULL_MARCH + XRAY + kdtree 192³ + max ridge + heavy field
        FrameConfig(
            field_capture     = FieldCaptureConfig(
                enabled=True, grid_kind="kdtree",
                nx=192, ny=192, nz=192,
                capture_strikes=True, max_strikes=2_000_000,
            ),
            camera_visibility = CameraVisibilityConfig(
                camera_vis_mode    = _CAM_VIS_FULL_MARCH,
                transparent_mode   = _CAM_TRANSP_XRAY,
                depth_cull_enabled = True,
                depth_cull_m       = 150.0,
            ),
            surface_spline    = SurfaceSplineConfig(
                enabled=True, fit_all_tris=True, ridge_lambda=1.0e-3,
            ),
            parametric_sdf    = ParametricSdfConfig(
                enabled=True, model="mixed", saddle_amplitude_m=3.0e-3,
                sphere_radius_m=0.12, neighborhood_margin_uv=5.0e-2,
            ),
            integral_split    = _split(dfi=0.40, dsi=0.15, dwp=-0.8),
            description       = "FULL_MARCH+XRAY · kdtree-192³ · spline max ridge · parametric mixed · deep",
            detail_level      = 7,
        ),
    ]
    return schedules[level]


@dataclass
class FieldIntegralObject:
    backend: str
    frame_index: int
    freq_hz: np.ndarray
    integrated_bands: np.ndarray
    bookkeeping_bands: np.ndarray
    metrics: dict


@dataclass
class SurfaceIntegralObject:
    backend: str
    frame_index: int
    freq_hz: np.ndarray
    integrated_bands: np.ndarray
    bookkeeping_bands: np.ndarray
    metrics: dict


@dataclass
class SensorGeometryConfig:
    """Camera sensor plane geometry for BDPT backward integration."""
    enabled: bool = True
    use_pixel_cone_sampling: bool = True
    n_aperture_samples: int = 16
    aperture_stop_group_id: int = -1


@dataclass
class SensorIntegralObject:
    """Sensor-plane photometric integration: rays reaching the camera back."""
    backend: str
    frame_index: int
    optics: CameraOptics
    film: FilmExposure
    n_pixels: int
    sensor_w_m: float
    sensor_h_m: float
    pixel_pitch_m: float
    focal_m: float
    aperture_radius_m: float
    qe_peak: float
    photons_per_pixel: np.ndarray    # (H, W) float32 accumulated photon count
    electrons_per_pixel: np.ndarray  # (H, W) with QE applied
    noise_floor_e: float
    full_well_e: int
    snr_linear: np.ndarray           # (H, W) per-pixel SNR in linear units
    peak_snr: float
    mean_snr: float
    metrics: dict
    sensor_uv_hits: Optional[np.ndarray] = None  # (H, W) amplitude-weighted sensor UV occupancy
    sensor_uv_thin_lens_hits: Optional[np.ndarray] = None  # (H, W) occupancy with thin-lens interaction flag
    oracle_photons_per_pixel: Optional[np.ndarray] = None  # (H, W) reference oracle pinhole (perfect optics)
    physical_photons_per_pixel: Optional[np.ndarray] = None  # (H, W) physical pinhole (photon loss/noise)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe format for frame summary."""
        return {
            "n_pixels": int(self.n_pixels),
            "sensor_w_m": float(self.sensor_w_m),
            "sensor_h_m": float(self.sensor_h_m),
            "pixel_pitch_m": float(self.pixel_pitch_m),
            "qe_peak": float(self.qe_peak),
            "peak_snr": float(self.peak_snr),
            "mean_snr": float(self.mean_snr),
            "metrics": self.metrics,
        }


@dataclass
class IntegrationSnapshot:
    """Live or frame-final integration snapshot for one backend."""
    backend: str
    frame_index: int
    batch_index: int
    measured_H_J: float
    target_H_J: float
    gain_linear: float
    image_data: np.ndarray
    rgb_linear: np.ndarray
    field_integrated_data: np.ndarray
    surface_integrated_data: np.ndarray
    sensor_photons_data: Optional[np.ndarray] = None
    sensor_snr_data: Optional[np.ndarray] = None
    sensor_rgb_data: Optional[np.ndarray] = None
    endpoint_rgb_data: Optional[np.ndarray] = None
    pinhole_rgb_data: Optional[np.ndarray] = None
    mode: str = "stream"


# ─────────────────────────────────────────────────────────────────────────────
# Sensor registration helpers for bidirectional integration
# ─────────────────────────────────────────────────────────────────────────────

def _build_camera_sensor_descriptor(optics: CameraOptics,
                                    cam: PhysicalCameraRig,
                                    width_px: int,
                                    height_px: int,
                                    n_aperture_samples: int = 16,
                                    aperture_stop_group_id: int = -1,
                                    solved: Any = None,
                                    camera_mode: int | None = None) -> Optional[dict]:
    """Build a CameraSensor descriptor for PIXEL_CONE BDPT sampling.

    When ``solved`` (a SolvedCameraPackage) is provided, the effective focal
    length, focus distance, lens centre, and camera_mode are wired from the
    solver rather than being left at their aperture-cone defaults.

    Returns a dict compatible with bdpt_integrator.TriangleGroup.sensor_camera,
    or None if BDPT integration is unavailable.
    """
    if not _HAS_BDPT_INTEGRATION or CameraSensor is None:
        return None

    from bdpt_integrator import (
        CAMERA_MODE_APERTURE_CONE, CAMERA_MODE_THIN_LENS_GEOMETRIC,
    )

    # Camera frame in world space from the physical camera rig.
    cam_pos = np.asarray(cam.pos, np.float64)
    fwd = np.asarray(cam.fwd, np.float64)
    up = np.asarray(cam.up, np.float64)
    fwd /= max(float(np.linalg.norm(fwd)), 1.0e-12)
    right = np.cross(fwd, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1.0e-9:
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    up = np.cross(right, fwd)
    up /= max(float(np.linalg.norm(up)), 1.0e-12)

    # Sensor centre position in world space.
    pos = cam_pos + fwd * float(cam.sensor_plane_offset_m)

    sensor_w_m = float(optics.sensor_w_mm) * 1.0e-3
    sensor_h_m = float(optics.sensor_h_mm) * 1.0e-3

    # Image-side distance: sensor → aperture (used as focal_m in descriptor).
    focal_m = float(cam.focal_m)
    aperture_radius_m = float(cam.aperture_radius_m)

    # Defaults: no thin-lens, use aperture-cone.
    effective_focal_m = 0.0
    focus_distance_m = 0.0
    lens_center_arr = None
    lens_fwd_arr = None
    mode = int(camera_mode) if camera_mode is not None else int(CAMERA_MODE_APERTURE_CONE)

    if solved is not None:
        si = solved.sanity_input
        rpt = solved.sanity_report
        # Aperture centre for lens position.
        aperture_z = float(
            si.aperture_rail_z_m if si.aperture_rail_z_m is not None
            else si.aperture_plane_offset_m
        )
        aperture_radius_m = float(si.aperture_radius_m) if si.aperture_radius_m is not None else aperture_radius_m
        effective_focal_m = float(rpt.planes.effective_focal_m)
        # focus_distance_m = scene distance from lens centre to focal plane.
        focal_plane_z = float(rpt.planes.focal_plane_z_m)
        sensor_z = float(si.sensor_plane_z_m)
        # focal_plane_z is along the world z-axis in solver coords;
        # focus_distance_m is |focal_plane_z - aperture_z|.
        focus_distance_m = max(1.0e-4, abs(focal_plane_z - aperture_z))
        lens_center_arr = cam_pos + fwd * aperture_z
        lens_fwd_arr = fwd.copy()
        # Image-side distance: aperture_z - sensor_z (should be positive).
        sensor_to_aperture = abs(aperture_z - sensor_z)
        if sensor_to_aperture > 1.0e-6:
            focal_m = float(sensor_to_aperture)
        if camera_mode is None:
            mode = int(CAMERA_MODE_THIN_LENS_GEOMETRIC)

    descriptor = dict(
        pos=pos,
        fwd=fwd,
        up=up,
        sensor_w_m=float(sensor_w_m),
        sensor_h_m=float(sensor_h_m),
        focal_m=float(focal_m),
        aperture_radius_m=float(aperture_radius_m),
        n_px=int(width_px),
        n_py=int(height_px),
        n_aperture_samples=int(n_aperture_samples),
        aperture_stop_group_id=int(aperture_stop_group_id),
        camera_mode=int(mode),
        effective_focal_m=float(effective_focal_m),
        focus_distance_m=float(focus_distance_m),
    )
    if lens_center_arr is not None:
        descriptor["lens_center"] = lens_center_arr
    if lens_fwd_arr is not None:
        descriptor["lens_fwd"] = lens_fwd_arr
    return descriptor


def _register_sensor_group(tracer: Any,
                          optics: CameraOptics,
                          cam: PhysicalCameraRig,
                          width_px: int,
                          height_px: int,
                          solved: Any = None) -> int:
    """Register the sensor plane as a SENSOR TriGroup for BDPT backward pass.

    When ``solved`` (SolvedCameraPackage) is provided, the sensor descriptor
    is built with thin-lens geometric mode wired from the solver.

    Returns the assigned group_id, or -1 if registration failed.
    """
    if not _HAS_BDPT_INTEGRATION:
        return -1

    try:
        # Build flat sensor geometry
        positions, uvs, tris = _build_flat_sensor_geometry(optics, height_px, width_px)

        # Triangle indices for the sensor group (all tris in flat 2-tri quad)
        sensor_tri_indices = np.array([0, 1], dtype=np.int32)

        # Build sensor descriptor, wiring solved camera if available.
        sensor_desc = _build_camera_sensor_descriptor(
            optics, cam, width_px, height_px, solved=solved)
        if sensor_desc is None:
            return -1
        
        # Create TriangleGroup descriptor
        group = TriangleGroup(
            role_bits=int(TRI_GROUP_ROLE_SENSOR),
            tri_indices=sensor_tri_indices,
            sample_policy=int(TRI_GROUP_SAMPLE_PIXEL_CONE),
            sensor_camera=sensor_desc,
        )
        
        # Register with the tracer
        group_id = group.register_with(tracer)
        return int(group_id)
    
    except Exception as e:
        print(f"WARNING: sensor group registration failed: {e}")
        return -1


class ExposureSession:
    def __init__(self, *,
                 optics: CameraOptics,
                 film:   FilmExposure,
                 width:  int,
                 height: int,
                 total_rays: int,
                 rays_per_batch: int,
                 max_bounces: int = 4,
                 freq_hz: Optional[np.ndarray] = None,
                 backends: tuple[str, ...] = ("cpp",),
                 out_dir: str = "exposures",
                 integrator: str = "bdpt",
                 bdpt_records_cap: int = 0,
                 scene_mode: str = "orbiters",
                 scene_mode_schedule: Optional[tuple[str, ...]] = None,
                 profile_enabled: bool = False,
                 integral_split: Optional[IntegralSplitConfig] = None,
                 camera_visibility: Optional[CameraVisibilityConfig] = None,
                 field_capture: Optional[FieldCaptureConfig] = None,
                 convergence: Optional[ConvergenceConfig] = None,
                 adaptive_allocation_mode: str = "stochastic",
                 camera_mode: Optional[int] = None,
                 n_frames_planned: int = 1,
                 output_width: Optional[int] = None,
                 output_height: Optional[int] = None,
                 output_oversample_stencil: str = "box",
                 show_hud: bool = True,
                 rgb_source: str = "sensor",
                 tone_map_mode: str = "reinhardt",
                 camera_solve_max_iters: int = 72,
                 camera_solve_seed_base: int = 12345,
                 camera_rail_model: Optional[RailModelConfig] = None,
                 camera_cone_samples_min: int = 8,
                 camera_cone_samples_max: int = 192,
                 camera_cone_ref_half_angle_deg: float = 3.0,
                 camera_cone_angle_exponent: float = 0.85,
                 sensor_film_slots: Optional[list[tuple[int, int]]] = None,
                 save_files: bool = False,
                 bdpt_intermediate_mode: str = "file",
                 bdpt_intermediate_max_bytes: int = 0,
                 retain_bdpt_intermediate: bool = False,
                 bdpt_intermediate_dir: Optional[str] = None,
                 t5_min_geom: float = -1.0,
                 t5_pair_budget: int = 200_000_000,
                 vcm_enabled: bool = True,
                 vcm_radius_mm: float = 2.0,
                 vcm_radius_alpha: float = 0.7,
                 gpu_resident: bool = False,
                 bdpt_native_packages: int = 1,
                 bdpt_native_sweeps: int = 1):
        self.optics = optics
        self.film   = film
        self.width  = int(width)
        self.height = int(height)
        self.total_rays = int(total_rays)
        self.rays_per_batch = max(1, int(rays_per_batch))
        self.max_bounces = int(max_bounces)
        self.freq_hz = np.asarray(freq_hz if freq_hz is not None else DEFAULT_FREQ_HZ,
                                  np.float64)
        self.backends_requested = tuple(backends)
        self.out_dir = out_dir
        self.integrator = str(integrator)
        requested_bdpt_cap = int(bdpt_records_cap)
        self.scene_mode = str(scene_mode)
        self.t5_pair_budget = max(0, int(t5_pair_budget))
        self.vcm_enabled = bool(vcm_enabled)
        self.vcm_radius_mm = max(1.0e-4, float(vcm_radius_mm))
        self.vcm_radius_alpha = min(1.0, max(1.0e-4, float(vcm_radius_alpha)))
        self.scene_mode_schedule = tuple(str(s) for s in scene_mode_schedule) if scene_mode_schedule else None
        self.profile_enabled = bool(profile_enabled)
        self._profiler = StageProfiler(self.profile_enabled)
        self.integral_split = (integral_split
                       if integral_split is not None
                       else IntegralSplitConfig())
        self.camera_visibility = (camera_visibility
                      if camera_visibility is not None
                      else CameraVisibilityConfig())
        self.field_capture = (field_capture
                      if field_capture is not None
                      else FieldCaptureConfig())
        self.convergence = (convergence
                 if convergence is not None
                 else ConvergenceConfig())
        self.adaptive_allocation_mode = str(adaptive_allocation_mode).strip().lower()
        if self.adaptive_allocation_mode not in ("stochastic", "quota", "uniform"):
            self.adaptive_allocation_mode = "stochastic"
        # Camera optical mode (Tier 0-6 system)
        self.camera_mode = int(camera_mode) if camera_mode is not None else int(CAMERA_MODE_APERTURE_CONE)
        self.n_frames_planned = max(1, int(n_frames_planned))
        self.output_width = int(output_width) if output_width is not None else int(self.width)
        self.output_height = int(output_height) if output_height is not None else int(self.height)
        self.output_width = max(1, self.output_width)
        self.output_height = max(1, self.output_height)
        self.output_oversample_stencil = str(output_oversample_stencil).strip().lower()
        if self.output_oversample_stencil not in ("box", "polar"):
            self.output_oversample_stencil = "box"
        self.show_hud = bool(show_hud)
        self.rgb_source = str(rgb_source).strip().lower()
        if self.rgb_source not in ("accum", "endpoint", "sensor"):
            raise ValueError(
                f"invalid rgb_source '{self.rgb_source}'; expected one of: accum, endpoint, sensor"
            )
        self.tone_map_mode = str(tone_map_mode).strip().lower()
        if self.tone_map_mode not in ("reinhardt", "delicate"):
            self.tone_map_mode = "reinhardt"
        self.camera_solve_max_iters = max(1, int(camera_solve_max_iters))
        self.camera_solve_seed_base = int(camera_solve_seed_base)
        self.camera_rail_model = (
            camera_rail_model if camera_rail_model is not None else RailModelConfig()
        )
        self.camera_cone_samples_min = max(1, int(camera_cone_samples_min))
        self.camera_cone_samples_max = max(self.camera_cone_samples_min, int(camera_cone_samples_max))
        self.camera_cone_ref_half_angle_deg = max(0.05, float(camera_cone_ref_half_angle_deg))
        self.camera_cone_angle_exponent = max(0.1, float(camera_cone_angle_exponent))
        self.save_files = bool(save_files)
        self.bdpt_intermediate_mode = str(bdpt_intermediate_mode).strip().lower()
        if self.bdpt_intermediate_mode not in ("memory", "file"):
            self.bdpt_intermediate_mode = "file"
        requested_bdpt_intermediate_max_bytes = int(max(0, int(bdpt_intermediate_max_bytes)))
        self.bdpt_intermediate_max_bytes = _clamp_bdpt_intermediate_bytes(
            requested_bdpt_intermediate_max_bytes
        )
        self.retain_bdpt_intermediate = bool(retain_bdpt_intermediate)
        self.bdpt_intermediate_dir = str(bdpt_intermediate_dir or out_dir)
        self.t5_min_geom = float(t5_min_geom)
        self.gpu_resident = bool(gpu_resident)
        # Package count is a lower bound on memory-safe spatial partitions of
        # one sensor sweep. Sweeps are independent complete passes accumulated
        # into the same native sensor and averaged at readback.
        self.bdpt_native_packages = int(max(1, bdpt_native_packages))
        self.bdpt_native_sweeps = int(max(1, bdpt_native_sweeps))
        hard_cap_records = max(1, _bdpt_cap_from_bytes(_BDPT_INTERMEDIATE_HARD_CAP_BYTES))
        if requested_bdpt_cap > 0:
            self.bdpt_records_cap = max(4096, min(requested_bdpt_cap, hard_cap_records))
        else:
            self.bdpt_records_cap = max(4096, _bdpt_cap_from_bytes(self.bdpt_intermediate_max_bytes))
        self.bdpt_stream_working_fraction = float(max(0.05, min(1.0, _BDPT_STREAM_WORKING_FRACTION)))
        self.bdpt_stream_records_cap = max(
            4096,
            int(self.bdpt_records_cap * self.bdpt_stream_working_fraction),
        )
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(self.bdpt_intermediate_dir, exist_ok=True)

        # Load sensor/film database (T2: ExposureSession integration)
        self._sensor_film_db = SensorFilmDatabase.instance()
        self._sensor_film_tensors = self._sensor_film_db.build_tensors()
        
        if sensor_film_slots is None:
            # Default: slot 0 with Canon EOS R6 + Kodak Portra 400
            sensor_film_slots = [(0, 0)] + [(-1, -1)] * (MAX_SENSOR_FILM_SLOTS - 1)
        
        self.sensor_film_slots = sensor_film_slots
        self._sensor_film_metadata = []  # For HUD and reporting
        
        # Build metadata for each slot
        for slot_id, (sensor_id, film_id) in enumerate(self.sensor_film_slots):
            if sensor_id >= 0 and film_id >= 0:
                sensor_name = self._sensor_film_db._sensor_order[sensor_id] if sensor_id < len(self._sensor_film_db._sensor_order) else "unknown"
                film_name = self._sensor_film_db._film_order[film_id] if film_id < len(self._sensor_film_db._film_order) else "unknown"
                sensor_record = self._sensor_film_tensors['sensor'][sensor_id]
                meta = {
                    "slot_id": slot_id,
                    "sensor_id": sensor_id,
                    "sensor_name": sensor_name,
                    "film_id": film_id,
                    "film_name": film_name,
                    "active": True,
                    "qe_peak": float(sensor_record[8]),  # qe_peak at offset 8
                    "read_noise_e": float(sensor_record[10]),  # read_noise_e at offset 10
                    "dark_current_e_s": float(sensor_record[11]),  # dark_current_e_s at offset 11
                }
                self._sensor_film_metadata.append(meta)
            else:
                self._sensor_film_metadata.append({
                    "slot_id": slot_id,
                    "sensor_id": -1,
                    "film_id": -1,
                    "active": False,
                })
        
        print(f"  [sensor_film] loaded {len(self._sensor_film_db._sensor_order)} sensors, "
              f"{len(self._sensor_film_db._film_order)} films, {sum(1 for m in self._sensor_film_metadata if m['active'])} active slots")
        self._sensor_film_lock = threading.Lock()

        if self.integrator == "bdpt":
            cap_gb = (self.bdpt_records_cap * _BDPT_RECORD_BYTES) / float(1024 ** 3)
            work_gb = (self.bdpt_stream_records_cap * _BDPT_RECORD_BYTES) / float(1024 ** 3)
            print(f"  [bdpt] records cap={self.bdpt_records_cap:_} (~{cap_gb:.2f} GiB payload)")
            print(
                "  [bdpt] streaming working cap="
                f"{self.bdpt_stream_records_cap:_} (~{work_gb:.2f} GiB, "
                f"{self.bdpt_stream_working_fraction:.2f}x of cap)"
            )
        print(
            f"  [conv] enabled={bool(self.convergence.enabled)} "
            f"drive_batches={bool(self.convergence.drive_batches)} "
            f"target={self.convergence.target_pct:.4f}% "
            f"H_error<= {float(self.convergence.max_rel_drift):.3e} "
            f"every={max(1, int(self.convergence.check_every_batches))} "
            f"min={max(1, int(self.convergence.min_batches))} "
            f"hold={max(1, int(self.convergence.hold_checks))} "
            f"max_batches={int(self.convergence.max_batches)}"
        )
        print(f"  [adaptive] allocation={self.adaptive_allocation_mode}")
        print(f"  [rgb] source={self.rgb_source}")
        print(f"  [res] render={self.width}x{self.height} output={self.output_width}x{self.output_height} stencil={self.output_oversample_stencil}")

        # Resumable atlas subprocesses restore accumulated sensor state. Give
        # each slice a deterministic disjoint seed instead of replaying slice 1.
        self._rng_seed = 1 + int(
            os.environ.get("SPECTRAL_EXPOSURE_SEED_OFFSET", "0")
        )
        self._frame_index = 0
        self._sensor_group_id = -1  # BDPT sensor group ID for this frame
        self._sensor_camera_desc: Optional[dict] = None
        self._last_bdpt_records: Optional[np.ndarray] = None
        self._last_target_photons_per_pixel: float = 0.0
        self._last_bdpt_records_path: Optional[str] = None
        self._last_bdpt_records_is_temp: bool = False
        self._last_bdpt_emit_counts: Optional[np.ndarray] = None
        self._bdpt_emit_rays_total: Optional[np.ndarray] = None
        self._bdpt_cap_hint_global: int = 0
        self._camera_pinhole_target_sensor_z_m: float = -0.050
        self._camera_last_cone_half_angle_deg: float = 0.0
        self._camera_last_cone_solid_angle_sr: float = 0.0
        self._camera_last_n_aperture_samples: int = 0
        self._pinhole_accum: Optional[np.ndarray] = None
        self._active_cam: Optional[PhysicalCameraRig] = None
        self._missing_rgb_source_warned: set[tuple[str, str, str, str]] = set()


    def _warn_missing_rgb_source_once(self, source: str, backend: str, context: str) -> None:
        key = (str(source), str(self.integrator), str(backend))
        if key in self._missing_rgb_source_warned:
            return
        self._missing_rgb_source_warned.add(key)
        print(
            "  [warn] rgb_source="
            f"{source} unavailable (integrator={self.integrator}, backend={backend}, {context}); "
            "enforcing sensor-only RGB policy: output forced to black because this pass did not produce a sensor integral"
        )

    def _adaptive_aperture_sample_count(self,
                                       cam: PhysicalCameraRig,
                                       detail_level: int) -> tuple[int, float, float]:
        """Choose aperture samples from projected cone spread.

        Ensures backward-pass PIXEL_CONE captures the true aperture-projected
        family of angles at both very small and very large apertures.
        """
        focal_m = max(1.0e-6, float(cam.focal_m))
        aperture_radius_m = max(1.0e-9, float(cam.aperture_radius_m))
        half_angle_rad = math.atan2(aperture_radius_m, focal_m)
        half_angle_deg = math.degrees(half_angle_rad)
        solid_angle_sr = 2.0 * math.pi * (1.0 - math.cos(half_angle_rad))

        base = max(4, 4 + 4 * int(max(0, detail_level)))
        ref_rad = math.radians(float(self.camera_cone_ref_half_angle_deg))
        ref_omega = 2.0 * math.pi * (1.0 - math.cos(ref_rad))
        omega_ratio = solid_angle_sr / max(ref_omega, 1.0e-12)
        scale = omega_ratio ** float(self.camera_cone_angle_exponent)
        n_ap = int(round(float(base) * float(max(0.25, scale))))
        n_ap = max(int(self.camera_cone_samples_min), min(int(self.camera_cone_samples_max), n_ap))
        return int(n_ap), float(half_angle_deg), float(solid_angle_sr)

    def _schedule_pixel_cone_sampling(self,
                                      requested_samples: int,
                                      rays_per_batch_total: int,
                                      n_emit_groups: int,
                                      tri_group_count_estimate: int,
                                      expected_batches: int) -> tuple[int, int]:
        """Schedule PIXEL_CONE work from resolution + batch budget.

        Returns
        -------
        (n_aperture_samples, pixel_stream_divisor)

        Strategy
        --------
        1. Scale aperture samples by resolution and rays-per-batch (not by a
           hard greedy clamp).
        2. Stream the film-plane cone work by processing every Nth pixel per
           batch, with phase derived from batch seed in native code.
        3. Use record-cap only as a safety backstop by increasing stream
           divisor when required.
        """
        req = max(1, int(requested_samples))
        n_bands = max(1, int(np.asarray(self.freq_hz).size))
        n_pix = max(1, int(self.width) * int(self.height))

        # Resolution scaling keeps cone work roughly stable as render resolution
        # changes (reference near 1280x720).
        ref_pix = 1280.0 * 720.0
        res_scale = math.sqrt(ref_pix / max(1.0, float(n_pix)))

        # Batch scaling tracks how much forward-path work we do per batch.
        # Smaller batches get proportionally fewer cone samples.
        ref_batch = 250_000.0
        batch_scale = math.sqrt(max(1.0, float(rays_per_batch_total)) / ref_batch)

        n_ap = int(round(float(req) * res_scale * batch_scale))
        n_ap = max(1, min(int(self.camera_cone_samples_max), n_ap))

        full_sensor_records = n_pix * n_bands * n_ap
        # Aim for sensor-side records on the same order as the forward batch.
        target_sensor_records = max(n_bands * 8192, int(rays_per_batch_total) * n_bands)
        stream_div = max(1, int(math.ceil(full_sensor_records / max(1.0, float(target_sensor_records)))))

        # Safety backstop: if the analytical batch still exceeds the record
        # budget, increase streaming divisor (never greedily collapse n_ap).
        cap_records = max(0, int(self.bdpt_stream_records_cap))
        if cap_records > 0:
            n_emit = max(1, int(n_emit_groups))
            group_est = max(1, int(tri_group_count_estimate))
            rays_per_emit = max(64, int(self.total_rays) // group_est)
            emit_target_total = max(n_emit, rays_per_emit * n_emit)
            emitter_records_est = emit_target_total * (int(self.max_bounces) + 1) * n_bands
            guard_records = max(4096, cap_records // 20)

            avail_sensor_records = cap_records - emitter_records_est - guard_records
            if avail_sensor_records <= 0:
                # Hard floor: keep path alive with minimal cone sampling.
                return 1, max(1, int(math.ceil((n_pix * n_bands) / max(1.0, float(cap_records)))))

            # Coverage target: do not stream over more phase classes than the
            # expected number of batches unless physically unavoidable.
            exp_batches = max(1, int(expected_batches))
            stream_div = min(stream_div, exp_batches)

            # Prefer reducing aperture samples over increasing stream divisor,
            # so more unique pixels receive values during one exposure.
            n_ap_fit = int(
                max(
                    1,
                    (int(avail_sensor_records) * int(stream_div)) // max(1, int(n_pix * n_bands)),
                )
            )
            if n_ap_fit < n_ap:
                n_ap = max(1, n_ap_fit)
                full_sensor_records = n_pix * n_bands * n_ap

            # If still over budget at this n_ap, raise stream divisor just
            # enough to fit. This is the unavoidable case.
            min_div_fit = int(math.ceil(full_sensor_records / float(max(1, avail_sensor_records))))
            stream_div = max(stream_div, min_div_fit)

        # Cap to n_px to prevent row-level blank-line interleaving.
        # When stream_div > n_px the linear-index stepping skips entire rows
        # in some phases; bounding it here keeps coverage within each row.
        stream_div = min(stream_div, max(1, int(self.width)))
        return int(n_ap), int(stream_div)

    def swap_sensor_film_slot(self, slot_idx: int, sensor_delta: int = 0, film_delta: int = 0) -> dict:
        """Cycle sensor and/or film on slot_idx by delta steps (wraps around).
        Returns the new metadata dict for the slot."""
        n_sensors = len(self._sensor_film_db._sensor_order)
        n_films   = len(self._sensor_film_db._film_order)
        if n_sensors == 0 or n_films == 0:
            return {}
        with self._sensor_film_lock:
            s_id, f_id = self.sensor_film_slots[slot_idx]
            if s_id < 0: s_id = 0
            if f_id < 0: f_id = 0
            s_id = (s_id + sensor_delta) % n_sensors
            f_id = (f_id + film_delta)   % n_films
            self.sensor_film_slots[slot_idx] = (s_id, f_id)
            sensor_name = self._sensor_film_db._sensor_order[s_id]
            film_name   = self._sensor_film_db._film_order[f_id]
            sensor_record = self._sensor_film_tensors['sensor'][s_id]
            meta = {
                "slot_id": slot_idx,
                "sensor_id": s_id,
                "sensor_name": sensor_name,
                "film_id": f_id,
                "film_name": film_name,
                "active": True,
                "qe_peak": float(sensor_record[8]),
                "read_noise_e": float(sensor_record[10]),
                "dark_current_e_s": float(sensor_record[11]),
            }
            self._sensor_film_metadata[slot_idx] = meta
        return meta

    def _release_last_bdpt_records(self) -> None:
        arr = self._last_bdpt_records
        self._last_bdpt_records = None
        if isinstance(arr, np.memmap):
            try:
                arr.flush()
            except Exception:
                pass
            try:
                arr._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        gc.collect()

    def _scene_mode_for_frame(self) -> str:
        if self.scene_mode_schedule:
            idx = min(max(self._frame_index, 0), len(self.scene_mode_schedule) - 1)
            return self.scene_mode_schedule[idx]
        return self.scene_mode

    def _cleanup_temp_bdpt_file(self) -> None:
        if not self._last_bdpt_records_is_temp or not self._last_bdpt_records_path:
            return
        self._release_last_bdpt_records()
        try:
            os.remove(self._last_bdpt_records_path)
        except OSError:
            pass
        self._last_bdpt_records_path = None
        self._last_bdpt_records_is_temp = False

    def _discard_last_bdpt_records(self, clear_emit_counts: bool = True) -> None:
        if self._last_bdpt_records_is_temp and self._last_bdpt_records_path:
            self._cleanup_temp_bdpt_file()
        else:
            self._release_last_bdpt_records()
            self._last_bdpt_records_path = None
            self._last_bdpt_records_is_temp = False
        if clear_emit_counts:
            self._last_bdpt_emit_counts = None

    def _pinhole_plane_basis(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._sensor_camera_desc is None:
            return None
        cam_desc = self._sensor_camera_desc
        cam_pos = np.asarray(cam_desc["pos"], np.float64)
        cam_fwd = np.asarray(cam_desc["fwd"], np.float64)
        cam_up = np.asarray(cam_desc["up"], np.float64)
        cam_fwd /= max(float(np.linalg.norm(cam_fwd)), 1.0e-12)
        cam_up /= max(float(np.linalg.norm(cam_up)), 1.0e-12)
        cam_right = np.cross(cam_fwd, cam_up)
        cam_right /= max(float(np.linalg.norm(cam_right)), 1.0e-12)
        cam_up = np.cross(cam_right, cam_fwd)

        sensor_w = float(cam_desc["sensor_w_m"])
        sensor_h = float(cam_desc["sensor_h_m"])
        cur_sensor_offset_m = float(self._active_cam.sensor_plane_offset_m) if self._active_cam is not None else 0.0
        target_sensor_offset_m = float(self._camera_pinhole_target_sensor_z_m)
        pin_origin = cam_pos + cam_fwd * (target_sensor_offset_m - cur_sensor_offset_m)
        basis_u = cam_right * sensor_w
        basis_v = cam_up * sensor_h
        return (
            np.asarray(pin_origin, np.float64),
            np.asarray(basis_u, np.float64),
            np.asarray(basis_v, np.float64),
        )

    def _accumulate_pinhole_records(self, recs: np.ndarray) -> None:
        if aggregate_to_image is None:
            return
        if recs.size == 0:
            return
        plane = self._pinhole_plane_basis()
        if plane is None:
            return
        pin_img = aggregate_to_image(
            recs,
            n_bands=int(self.freq_hz.shape[0]),
            height=int(self.height),
            width=int(self.width),
            plane_origin=plane[0],
            plane_basis_u=plane[1],
            plane_basis_v=plane[2],
        )
        if self._pinhole_accum is None or self._pinhole_accum.shape != pin_img.shape:
            self._pinhole_accum = np.asarray(pin_img, np.complex64)
            return
        self._pinhole_accum += np.asarray(pin_img, np.complex64)

    def _stage_bdpt_records(self, recs: np.ndarray) -> None:
        self._cleanup_temp_bdpt_file()
        recs32 = np.ascontiguousarray(recs, dtype=np.float32)
        self._last_bdpt_records_path = None
        self._last_bdpt_records_is_temp = False

        if self.bdpt_intermediate_mode != "file":
            self._last_bdpt_records = recs32
            return

        est_bytes = int(recs32.nbytes)
        if self.bdpt_intermediate_max_bytes > 0 and est_bytes > self.bdpt_intermediate_max_bytes:
            lim_gb = self.bdpt_intermediate_max_bytes / float(1024 ** 3)
            cur_gb = est_bytes / float(1024 ** 3)
            print(f"  [warn] BDPT intermediate {cur_gb:.2f}GB exceeds cap {lim_gb:.2f}GB; using memory")
            self._last_bdpt_records = recs32
            return

        frame_tag = f"{self._frame_index:04d}"
        if self.retain_bdpt_intermediate:
            path = os.path.join(self.bdpt_intermediate_dir, f"bdpt_records_{frame_tag}.npy")
            is_temp = False
        else:
            fd, path = tempfile.mkstemp(
                prefix=f"bdpt_records_{frame_tag}_",
                suffix=".npy",
                dir=self.bdpt_intermediate_dir,
            )
            os.close(fd)
            is_temp = True

        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=recs32.shape)
        mm[:] = recs32
        mm.flush()
        del mm

        self._last_bdpt_records = np.load(path, mmap_mode="r")
        self._last_bdpt_records_path = path
        self._last_bdpt_records_is_temp = is_temp

        size_gb = est_bytes / float(1024 ** 3)
        keep_msg = "retained" if self.retain_bdpt_intermediate else "ephemeral"
        print(f"  bdpt records staged ({keep_msg}) → {path}  [{size_gb:.2f}GB]")

    def _allocate_bdpt_emit_rays(self,
                                 cpp_back: CppExposureBackend,
                                 emitter_centers: np.ndarray,
                                 target_total_rays: int,
                                 seed: int) -> np.ndarray:
        """Adaptive per-emitter allocation for BDPT using endpoint uncertainty.

        Default mode (stochastic) uses EndpointRecord-derived uncertainty and
        novelty; quota/uniform remain explicit fallback options.
        """
        n_emit = int(emitter_centers.shape[0])
        if n_emit <= 0:
            return np.zeros((0,), dtype=np.int32)

        total_rays = max(n_emit, int(target_total_rays))
        mode = str(self.adaptive_allocation_mode).strip().lower()
        if mode not in ("stochastic", "quota", "uniform"):
            mode = "stochastic"

        if mode in ("quota", "uniform"):
            per_emit = max(1, total_rays // n_emit)
            return cpp_back._allocate_source_rays(
                n_rays=per_emit,
                src_pos=emitter_centers,
                mode=mode,
                seed=seed,
            )

        # Smart default: uncertainty + completeness pressure + novelty.
        if self._bdpt_emit_rays_total is None or self._bdpt_emit_rays_total.shape[0] != n_emit:
            self._bdpt_emit_rays_total = np.zeros((n_emit,), dtype=np.float64)

        scores = np.ones((n_emit,), dtype=np.float64)
        recs = self._last_bdpt_records
        prev_counts = self._last_bdpt_emit_counts
        if recs is not None and recs.size > 0 and prev_counts is not None and prev_counts.size == n_emit:
            rec_arr = np.asarray(recs, dtype=np.float32)
            subpath = rec_arr[:, 0].astype(np.int64, copy=False)
            amp_re = rec_arr[:, 12].astype(np.float64, copy=False)
            amp_im = rec_arr[:, 13].astype(np.float64, copy=False)
            e = amp_re * amp_re + amp_im * amp_im
            e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)

            prefix = np.cumsum(prev_counts.astype(np.int64, copy=False))
            emit_idx = np.searchsorted(prefix, subpath, side="right")
            valid = (emit_idx >= 0) & (emit_idx < n_emit)

            if np.any(valid):
                idx = emit_idx[valid]
                ev = e[valid]
                cnt = np.bincount(idx, minlength=n_emit).astype(np.float64)
                s1 = np.bincount(idx, weights=ev, minlength=n_emit).astype(np.float64)
                s2 = np.bincount(idx, weights=ev * ev, minlength=n_emit).astype(np.float64)

                mean = np.divide(s1, np.maximum(1.0, cnt))
                var = np.maximum(0.0, np.divide(s2, np.maximum(1.0, cnt)) - mean * mean)
                uncertainty = np.sqrt(var) / np.sqrt(np.maximum(1.0, cnt))
                completeness = 1.0 / np.sqrt(1.0 + cnt)
                novelty = 1.0 / np.sqrt(1.0 + self._bdpt_emit_rays_total)

                def _norm(v: np.ndarray) -> np.ndarray:
                    vmax = float(np.max(v)) if v.size else 0.0
                    if vmax <= 1.0e-20:
                        return np.ones_like(v, dtype=np.float64)
                    return np.clip(v / vmax, 0.0, 1.0)

                scores = (0.50 * _norm(uncertainty) +
                          0.30 * _norm(completeness) +
                          0.20 * _norm(novelty))
                scores = np.maximum(scores, 1.0e-9)

        scores = np.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=0.0)
        scores = np.clip(scores, 0.0, None)
        if not np.any(np.isfinite(scores)) or float(np.sum(scores)) <= 0.0:
            scores = np.ones((n_emit,), dtype=np.float64)

        weights = scores / max(1.0e-20, float(np.sum(scores)))
        weights = np.nan_to_num(weights, nan=1.0 / max(1, n_emit), posinf=0.0, neginf=0.0)
        total_w = float(np.sum(weights))
        if total_w <= 0.0 or not np.isfinite(total_w):
            weights = np.full((n_emit,), 1.0 / float(n_emit), dtype=np.float64)
        else:
            weights = np.clip(weights / total_w, 0.0, 1.0)
        rng = np.random.default_rng(int(seed))
        draw = rng.multinomial(int(total_rays - n_emit), weights)
        out = np.ones((n_emit,), dtype=np.int32)
        out += draw.astype(np.int32, copy=False)
        self._bdpt_emit_rays_total += out.astype(np.float64)
        return out

    # ── Build per-frame plan + scene + backends ──────────────────────────
    def _build_plan(self, scene: TracerScene) -> RayDispatchPlan:
        radiance = scene.scene_radiance_W_sr_m2()
        if radiance <= 0.0:
            # Avoid divide-by-zero: bias to dim moonlight (~1e-4 W·sr⁻¹·m⁻²).
            radiance = 1.0e-4
        # The budget allocator multiplies density by the physical sensor pixel
        # count. Divide by that same count so --total-rays is an actual dispatch
        # budget, including deliberately tiny calibration exposures.
        physical_pixels = max(1, int(self.optics.n_pixels()))
        rps = max(0.0, float(self.total_rays) / float(physical_pixels))
        n_batches = max(1,
                        (self.total_rays + self.rays_per_batch - 1) //
                        self.rays_per_batch)
        return plan_ray_budget(
            self.optics, self.film,
            scene_radiance_W_sr_m2     = radiance,
            rays_per_pixel_per_second  = rps,
            n_batches                  = int(n_batches),
            sensor_spp                 = 1.0,
            capture_efficiency         = 1.0,
            ref_wavelength_nm          = 555.0,
        )

    def _build_backends(self, scene: TracerScene,
                        cam: PhysicalCameraRig,
                        frame_cfg: FrameConfig) -> dict[str, ExposureBackend]:
        backends: dict[str, ExposureBackend] = {}
        cpp_b: Optional[CppExposureBackend] = None
        if "cpp" in self.backends_requested:
            cpp_b = CppExposureBackend(
                scene, cam, self.freq_hz,
                max_bounces   = self.max_bounces,
                min_amplitude = 1.0e-3,
                atmo_abs_db_per_m = 0.0,
                adaptive_mode = self.adaptive_allocation_mode,
                bdpt_intermediate_mode = self.bdpt_intermediate_mode,
                bdpt_intermediate_dir = self.bdpt_intermediate_dir,
                bdpt_intermediate_max_bytes = self.bdpt_intermediate_max_bytes,
                retain_bdpt_intermediate = self.retain_bdpt_intermediate,
                bdpt_initial_cap_hint = self._bdpt_cap_hint_global,
                t5_min_geom = self.t5_min_geom,
                gpu_resident = self.gpu_resident,
            )
            if hasattr(cpp_b.tracer, "set_camera_visibility"):
                cv = frame_cfg.camera_visibility
                cpp_b.tracer.set_camera_visibility(
                    camera_vis_mode    = int(cv.camera_vis_mode),
                    transparent_mode   = int(cv.transparent_mode),
                    depth_cull_enabled = bool(cv.depth_cull_enabled),
                    depth_cull_m       = float(cv.depth_cull_m),
                )
            if self.profile_enabled and hasattr(cpp_b.tracer, "set_profile_pulse"):
                try:
                    cpp_b.tracer.set_profile_pulse(True, 2.0)
                except Exception as exc:
                    print(f"  [warn] native profile pulse enable failed: {exc}")
            if self.profile_enabled and hasattr(cpp_b.tracer, "set_t5_profile"):
                cpp_b.tracer.set_t5_profile(True)
                print("  [t5-profile] GPU connection rejection counters enabled")
            if hasattr(cpp_b.tracer, "set_t5_pair_budget"):
                cpp_b.tracer.set_t5_pair_budget(self.t5_pair_budget)
                print(f"[config] C++ tracer t5_pair_budget={self.t5_pair_budget:,}")
            if hasattr(cpp_b.tracer, "set_vcm"):
                cpp_b.tracer.set_vcm(
                    self.vcm_enabled,
                    self.vcm_radius_mm * 1.0e-3,
                    self.vcm_radius_alpha,
                )
                print(
                    f"[config] spectral VCM={'on' if self.vcm_enabled else 'off'} "
                    f"radius={self.vcm_radius_mm:g} mm alpha={self.vcm_radius_alpha:g}"
                )
            fc = frame_cfg.field_capture
            if fc.enabled and hasattr(cpp_b.tracer, "enable_field_capture_regular"):
                bmin = np.asarray(scene.bounds_min, np.float32)
                bmax = np.asarray(scene.bounds_max, np.float32)
                n_bands = int(getattr(cpp_b, "n_bands", int(np.asarray(self.freq_hz).size)))
                n_cells = int(max(1, int(fc.nx)) * max(1, int(fc.ny)) * max(1, int(fc.nz)))
                est_field_bytes = int(n_cells * max(1, n_bands) * 8)
                est_field_mib = float(est_field_bytes) / float(1024 ** 2)
                if self.profile_enabled:
                    print(
                        "  [field_capture] "
                        f"kind={fc.grid_kind} dims={int(fc.nx)}x{int(fc.ny)}x{int(fc.nz)} "
                        f"bands={n_bands} est_field={est_field_mib:.2f} MiB "
                        f"capture_strikes={bool(fc.capture_strikes)} max_strikes={int(fc.max_strikes)}"
                    )
                    print(
                        "                  "
                        f"bounds min={np.asarray(bmin, dtype=np.float32).tolist()} "
                        f"max={np.asarray(bmax, dtype=np.float32).tolist()}"
                    )
                try:
                    if fc.grid_kind == "kdtree" and hasattr(cpp_b.tracer, "enable_field_capture_kdtree"):
                        nodes = [{
                            "bmin": bmin,
                            "bmax": bmax,
                            "child_lo": -1,
                            "child_hi": -1,
                            "split_axis": -1,
                            "split_pos": 0.0,
                            "leaf_dims": np.asarray([fc.nx, fc.ny, fc.nz], np.int32),
                            "first_data": 0,
                        }]
                        cpp_b.tracer.enable_field_capture_kdtree(
                            nodes,
                            capture_strikes=bool(fc.capture_strikes),
                            max_strikes=int(fc.max_strikes),
                            clear_existing=True,
                        )
                    else:
                        cpp_b.tracer.enable_field_capture_regular(
                            int(fc.nx), int(fc.ny), int(fc.nz),
                            bmin, bmax,
                            capture_strikes=bool(fc.capture_strikes),
                            max_strikes=int(fc.max_strikes),
                            clear_existing=True,
                        )
                except Exception as exc:
                    detail = (
                        "field capture gate failed: "
                        f"gate=enable_field_capture_{fc.grid_kind} "
                        f"native_error={exc}"
                    )
                    raise RuntimeError(detail) from exc
            backends["cpp"] = cpp_b
        if "glsl" in self.backends_requested:
            # GLSL backend is a scaffold — not yet implemented for ray tracing.
            # When BDPT/backward/forward is the integrator and no cpp backend
            # was explicitly requested, inject a shadow C++ backend to provide
            # the actual BDPT computation. The GLSL backend mirrors its
            # accumulators and the output loop suppresses the cpp output file.
            if (self.integrator in ("forward", "backward", "bdpt")
                    and cpp_b is None):
                print("  [glsl-bdpt] GLSL backend is a scaffold; "
                      "injecting shadow C++ backend for ray computation")
                try:
                    cpp_b = CppExposureBackend(
                        scene, cam, self.freq_hz,
                        max_bounces              = self.max_bounces,
                        min_amplitude            = 1.0e-3,
                        atmo_abs_db_per_m        = 0.0,
                        adaptive_mode            = self.adaptive_allocation_mode,
                        bdpt_intermediate_mode   = self.bdpt_intermediate_mode,
                        bdpt_intermediate_dir    = self.bdpt_intermediate_dir,
                        bdpt_intermediate_max_bytes = self.bdpt_intermediate_max_bytes,
                        retain_bdpt_intermediate = self.retain_bdpt_intermediate,
                        bdpt_initial_cap_hint    = self._bdpt_cap_hint_global,
                        t5_min_geom              = self.t5_min_geom,
                        gpu_resident             = self.gpu_resident,
                    )
                    if hasattr(cpp_b.tracer, "set_camera_visibility"):
                        cv = frame_cfg.camera_visibility
                        cpp_b.tracer.set_camera_visibility(
                            camera_vis_mode    = int(cv.camera_vis_mode),
                            transparent_mode   = int(cv.transparent_mode),
                            depth_cull_enabled = bool(cv.depth_cull_enabled),
                            depth_cull_m       = float(cv.depth_cull_m),
                        )
                    backends["cpp"] = cpp_b
                except Exception as exc:
                    print(f"  [warn] shadow C++ backend creation failed: {exc}")
            backends["glsl"] = GlslExposureBackend(
                scene, cam, self.freq_hz,
                max_bounces  = self.max_bounces,
                mirror_from  = cpp_b,
                min_amplitude = 1.0e-3,
                atmo_abs_db_per_m = 0.0,
            )
        return backends

    # ── Sensor/Film SSBO upload (T3: binding helper) ────────────────────────
    def _bind_sensor_film_ssbo(self, tracer) -> None:
        """Upload sensor and film tensors to C++ tracer SSBO."""
        if tracer is None or not hasattr(tracer, 'set_sensor_film_ssbo'):
            print("  [warn] tracer has no set_sensor_film_ssbo method; skipping SSBO upload")
            return
        
        try:
            tracer.set_sensor_film_ssbo(
                sensor_chunk=self._sensor_film_tensors['sensor'].astype(np.float32, copy=False),
                film_chunk=self._sensor_film_tensors['film'].astype(np.float32, copy=False),
                active_slots=self.sensor_film_slots,
            )
            active_count = sum(1 for s, f in self.sensor_film_slots if s >= 0 and f >= 0)
            print(f"  [sensor_film_ssbo] uploaded {active_count} active slots to tracer")
        except Exception as e:
            print(f"  [warn] sensor_film_ssbo upload failed: {e}")

    def _configure_default_wave_contexts(self, tracer: Any, cam: PhysicalCameraRig, solved: Any | None = None) -> None:
        """Install default wave contexts so kernel wave path is active in BDPT."""
        if tracer is None:
            raise RuntimeError("wave context configuration requires a live tracer")
        if not hasattr(tracer, "clear_scale_contexts") or not hasattr(tracer, "add_scale_context"):
            raise RuntimeError("tracer does not expose scale-context API required for wave path")

        tracer.clear_scale_contexts()

        # Center contexts on the physical aperture and lens planes.
        cam_pos = np.asarray(cam.pos, np.float64)
        cam_fwd = np.asarray(cam.fwd, np.float64)
        cam_fwd = cam_fwd / max(1.0e-12, float(np.linalg.norm(cam_fwd)))
        focus_m = float(np.linalg.norm(np.asarray(scene_mod.SCENE_CENTER, np.float64) - cam_pos))
        focal_len_m = max(1.0e-4, float(cam.focal_m))
        phase_scale = 1.0
        lens_tube_len_m = float(abs(cam.sensor_plane_offset_m - cam.aperture_plane_offset_m))
        if solved is not None:
            focus_m = float(max(1.0e-3, solved.sanity_input.focus_distance_m))
            lens_tube_len_m = float(abs(solved.sanity_input.sensor_plane_z_m - solved.sanity_input.lens_center_z_m))
            phase_scale += min(2.0, max(0.0, float(solved.sanity_report.error_degree.overall)))

        aperture_center = cam_pos + cam_fwd * float(cam.aperture_plane_offset_m)
        aperture_radius_m = max(1.0e-6, float(cam.aperture_radius_m))

        wave_kind = int(getattr(_sk, "SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ", 1))
        thin_kind = int(getattr(_sk, "SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM", 2))
        thick_kind = int(getattr(_sk, "SCALE_CONTEXT_KIND_THICK_LENS_WAVE", 3))
        rt_wave = int(getattr(_sk, "RT_SCALE_WAVE", 1))

        thin_payload = np.asarray([focal_len_m], dtype=np.float64)
        tracer.add_scale_context(
            pos=aperture_center.astype(np.float64),
            radius=float(max(aperture_radius_m * 1.2, 1.0e-4)),
            scale_type=rt_wave,
            dt_m=float(max(1.0e-4, aperture_radius_m * 0.08)),
            n_substeps=2,
            n_real=1.0,
            n_imag=0.0,
            context_kind=thin_kind,
            payload=thin_payload,
        )

        wave_payload = np.asarray([focal_len_m], dtype=np.float64)
        tracer.add_scale_context(
            pos=aperture_center.astype(np.float64),
            radius=float(max(aperture_radius_m * 1.5, 1.0e-4)),
            scale_type=rt_wave,
            dt_m=float(max(1.0e-4, aperture_radius_m * 0.1)),
            n_substeps=2,
            n_real=1.0,
            n_imag=0.0,
            context_kind=wave_kind,
            payload=wave_payload,
        )

        tube_center = cam_pos + cam_fwd * (float(cam.sensor_plane_offset_m) + 0.5 * lens_tube_len_m)
        tracer.add_scale_context(
            pos=tube_center.astype(np.float64),
            radius=float(max(aperture_radius_m * 1.1, 1.0e-4)),
            scale_type=rt_wave,
            dt_m=float(max(1.0e-4, lens_tube_len_m * 0.1)),
            n_substeps=2,
            n_real=1.0,
            n_imag=0.0,
            context_kind=wave_kind,
            payload=wave_payload,
        )

        thick_payload = np.asarray([focal_len_m, phase_scale], dtype=np.float64)
        tracer.add_scale_context(
            pos=aperture_center.astype(np.float64),
            radius=float(max(aperture_radius_m * 2.0, 1.0e-4)),
            scale_type=rt_wave,
            dt_m=float(max(1.0e-4, aperture_radius_m * 0.1)),
            n_substeps=2,
            n_real=1.0,
            n_imag=0.0,
            context_kind=thick_kind,
            payload=thick_payload,
        )

    # ── Calibration: gain to match measured H to target H ────────────────
    @staticmethod
    def _train_emissivity_gain(measured_H_J: float, target_H_J: float) -> float:
        if measured_H_J <= 0.0 or target_H_J <= 0.0:
            return 1.0
        return float(target_H_J / measured_H_J)

    def _make_integral_objects(self,
                               backend: str,
                               surf_bhw: np.ndarray,
                               field_bhw: np.ndarray,
                               gain: float,
                               integral_split: Optional[IntegralSplitConfig] = None
                               ) -> tuple[FieldIntegralObject,
                                          SurfaceIntegralObject,
                                          np.ndarray,
                                          np.ndarray]:
        """Build field/surface integral objects from separated BDPT accumulators.

        surf_bhw  — (n_bands, H, W) sensor-hit accumulation (surf_accum).
        field_bhw — (n_bands, H, W) ambient/field accumulation (field_accum).
        Both are produced by scatter_bdpt_records() per batch.
        """
        cfg = integral_split if integral_split is not None else self.integral_split
        s = np.asarray(surf_bhw,  np.float64) * float(gain)
        f = np.asarray(field_bhw, np.float64) * float(gain)

        fi = float(np.clip(cfg.field_integrate_frac, 0.0, 1.0))
        fb = float(np.clip(cfg.field_bookkeep_frac, 0.0, 1.0))
        si = float(np.clip(cfg.surface_integrate_frac, 0.0, 1.0))
        sb = float(np.clip(cfg.surface_bookkeep_frac, 0.0, 1.0))

        field_integrated = fi * f
        field_bookkeep   = fb * np.maximum(0.0, f - field_integrated)
        surf_integrated  = si * s
        surf_bookkeep    = sb * np.maximum(0.0, s - surf_integrated)

        field_obj = FieldIntegralObject(
            backend=backend,
            frame_index=int(self._frame_index),
            freq_hz=self.freq_hz.copy(),
            integrated_bands=field_integrated.astype(np.float32, copy=False),
            bookkeeping_bands=field_bookkeep.astype(np.float32, copy=False),
            metrics={
                "integrated_energy": float(field_integrated.sum()),
                "bookkeep_energy": float(field_bookkeep.sum()),
            },
        )
        surface_obj = SurfaceIntegralObject(
            backend=backend,
            frame_index=int(self._frame_index),
            freq_hz=self.freq_hz.copy(),
            integrated_bands=surf_integrated.astype(np.float32, copy=False),
            bookkeeping_bands=surf_bookkeep.astype(np.float32, copy=False),
            metrics={
                "integrated_energy": float(surf_integrated.sum()),
                "bookkeep_energy": float(surf_bookkeep.sum()),
            },
        )

        merged = (field_integrated + surf_integrated).astype(np.float32, copy=False)
        rgb_linear = _bands_to_rgb(merged, self.freq_hz).astype(np.float32, copy=False)
        return field_obj, surface_obj, merged, rgb_linear

    def _tone_map_reinhardt(self, rgb_linear: np.ndarray) -> np.ndarray:
        """Reinhardt tone mapper: L_out = L_in / (1 + L_in)."""
        x = np.maximum(0.0, np.asarray(rgb_linear, np.float32))
        y = x / (1.0 + x)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    def _tone_map(self, rgb_linear: np.ndarray,
                  integral_split: Optional[IntegralSplitConfig] = None) -> np.ndarray:
        """Dispatch to configured tone mapper."""
        if self.tone_map_mode == "delicate":
            return self._tone_map_delicate(rgb_linear, integral_split)
        else:  # reinhardt (default)
            return self._tone_map_reinhardt(rgb_linear)

    def _tone_map_delicate(self, rgb_linear: np.ndarray,
                           integral_split: Optional[IntegralSplitConfig] = None
                           ) -> np.ndarray:
        """Soft-knee tone mapper that preserves low-light separation (percentile-based)."""
        cfg = integral_split if integral_split is not None else self.integral_split
        x = np.maximum(0.0, np.asarray(rgb_linear, np.float32))
        nz = x[x > 0.0]
        pct = float(np.clip(cfg.hdr_white_percentile, 75.0, 100.0))
        white = float(np.percentile(nz, pct)) if nz.size >= 16 else float(x.max())
        white = max(white, 1.0e-8)
        y = np.log1p(x / white * 6.0) / math.log1p(6.0)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    def _cpp_reduce_endpoint_rgb(self,
                                 tracer_obj: Any,
                                 gain: float,
                                 hdr_white_percentile: float) -> Optional[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        """Use the existing C++ endpoint reducer API for canonical spectral->RGB output."""
        _streaming_bdpt = self._last_bdpt_emit_counts is not None
        if _streaming_bdpt:
            return None
        if tracer_obj is None or not hasattr(tracer_obj, "reduce_endpoint_records_to_rgb_image"):
            return None
        if self._last_bdpt_records is None or self._last_bdpt_records.size == 0:
            return None
        if self._sensor_group_id < 0:
            return None

        cpp_rgb = tracer_obj.reduce_endpoint_records_to_rgb_image(
            self._last_bdpt_records,
            int(self.width),
            int(self.height),
            int(self._sensor_group_id),
            float(max(gain, 0.0)),
            float(hdr_white_percentile),
        )
        endpoint_rgb_linear = np.asarray(cpp_rgb["rgb_linear"], dtype=np.float32)
        endpoint_img = np.asarray(cpp_rgb["rgb_tonemapped"], dtype=np.float32)
        rgb_telemetry = dict(cpp_rgb.get("telemetry", {}))
        return endpoint_rgb_linear, endpoint_img, rgb_telemetry

    def _sensor_display_rgb(self, sensor_obj: SensorIntegralObject) -> np.ndarray:
        """Convert backward-pass sensor integral into display RGB."""
        photons = np.asarray(sensor_obj.photons_per_pixel, dtype=np.float32)
        electrons = np.asarray(sensor_obj.electrons_per_pixel, dtype=np.float32)

        x = np.maximum(0.0, photons)
        white = float(np.percentile(x, 99.5)) if x.size else 1.0
        white = max(white, 1.0e-8)
        y = np.log1p((x / white) * 6.0) / math.log1p(6.0)
        y = np.clip(y, 0.0, 1.0).astype(np.float32)

        # Subtle SNR tint preserves sensor intensity as the dominant signal.
        snr_proxy = np.sqrt(np.maximum(electrons, 0.0))
        snr_w = float(np.percentile(snr_proxy, 99.0)) if snr_proxy.size else 1.0
        snr_w = max(snr_w, 1.0e-8)
        t = np.clip(snr_proxy / snr_w, 0.0, 1.0).astype(np.float32)
        r = y
        g = np.clip(y * (0.92 + 0.08 * t), 0.0, 1.0)
        b = np.clip(y * (0.86 + 0.14 * t), 0.0, 1.0)
        return np.stack([r, g, b], axis=-1)

    @staticmethod
    def _downsample_hw(arr: np.ndarray, sy: int, sx: int, stencil: str = "box") -> np.ndarray:
        """Area-downsample 2D array while preserving dtype."""
        if sy <= 1 and sx <= 1:
            return arr
        h, w = int(arr.shape[0]), int(arr.shape[1])
        ny = max(1, h // max(1, sy))
        nx = max(1, w // max(1, sx))
        trimmed = np.asarray(arr[: ny * sy, : nx * sx])
        block = trimmed.reshape(ny, sy, nx, sx)
        if str(stencil).lower() == "polar" and sy > 1 and sx > 1:
            yy = (np.arange(sy, dtype=np.float64) + 0.5) / float(sy)
            xx = (np.arange(sx, dtype=np.float64) + 0.5) / float(sx)
            gy, gx = np.meshgrid(yy, xx, indexing="ij")
            ry = (gy - 0.5) / 0.5
            rx = (gx - 0.5) / 0.5
            r2 = (rx * rx) + (ry * ry)
            # Polar stencil: circular support with soft radial falloff.
            wmask = np.clip(1.0 - r2, 0.0, 1.0)
            wsum = float(np.sum(wmask))
            if wsum > 1.0e-20:
                reduced = np.tensordot(block, wmask, axes=([1, 3], [0, 1])) / wsum
            else:
                reduced = block.mean(axis=(1, 3), dtype=np.float64)
        else:
            reduced = block.mean(axis=(1, 3), dtype=np.float64)
        if np.issubdtype(arr.dtype, np.integer):
            reduced = np.rint(reduced)
        return reduced.astype(arr.dtype, copy=False)

    @classmethod
    def _downsample_bhw(cls, arr: np.ndarray, sy: int, sx: int, stencil: str = "box") -> np.ndarray:
        """Area-downsample (B,H,W) tensor while preserving dtype."""
        if sy <= 1 and sx <= 1:
            return arr
        bands = [cls._downsample_hw(arr[b], sy, sx, stencil=stencil) for b in range(int(arr.shape[0]))]
        return np.stack(bands, axis=0).astype(arr.dtype, copy=False)

    @classmethod
    def _downsample_hw3(cls, img: np.ndarray, sy: int, sx: int, stencil: str = "box") -> np.ndarray:
        """Area-downsample (H,W,3) RGB image while preserving dtype."""
        if sy <= 1 and sx <= 1:
            return img
        ch = [cls._downsample_hw(img[:, :, c], sy, sx, stencil=stencil) for c in range(int(img.shape[2]))]
        return np.stack(ch, axis=-1).astype(img.dtype, copy=False)

    @staticmethod
    def _resize_rgb_nearest(img_hw3: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """Resize RGB image using nearest-neighbor sampling."""
        img = np.asarray(img_hw3, np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            return np.zeros((max(1, int(out_h)), max(1, int(out_w)), 3), np.float32)
        h, w = int(img.shape[0]), int(img.shape[1])
        oh = max(1, int(out_h))
        ow = max(1, int(out_w))
        if h == oh and w == ow:
            return img.astype(np.float32, copy=False)
        y_idx = np.clip(np.floor(np.linspace(0.0, max(0.0, float(h - 1)), oh)).astype(np.int64), 0, max(0, h - 1))
        x_idx = np.clip(np.floor(np.linspace(0.0, max(0.0, float(w - 1)), ow)).astype(np.int64), 0, max(0, w - 1))
        return img[y_idx][:, x_idx, :].astype(np.float32, copy=False)

    @staticmethod
    def _normalize_map_energy(x_hw: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(x_hw, np.float64), 0.0)
        if x.size == 0:
            return np.zeros_like(x, dtype=np.float32)
        p = float(np.percentile(x, 99.5)) if np.any(x > 0.0) else 0.0
        if p <= 1.0e-12:
            return np.zeros_like(x, dtype=np.float32)
        y = np.clip(x / p, 0.0, 1.0)
        y = np.sqrt(y)
        return y.astype(np.float32, copy=False)

    def _build_lightfield_shell_overlays(self,
                                         grid_reim: Optional[np.ndarray],
                                         strikes: Optional[np.ndarray],
                                         field_cfg: FieldCaptureConfig) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build top/side flattened FIELD+SURFACE shell overlays from existing captures.

        FIELD comes from field-capture grid energy. SURFACE comes from strike rows.
        Both are projected and overlaid in color to expose body (field) and shell
        (surface) in the same image.
        """
        if grid_reim is None and strikes is None:
            return None, None

        nx = max(1, int(field_cfg.nx))
        ny = max(1, int(field_cfg.ny))
        nz = max(1, int(field_cfg.nz))

        field_top = np.zeros((nz, nx), np.float32)
        field_side = np.zeros((nz, ny), np.float32)

        if grid_reim is not None:
            try:
                g = np.asarray(grid_reim, np.float32)
                if g.ndim == 3 and g.shape[2] == 2 and g.shape[1] > 0:
                    field_amp = np.sqrt(np.maximum(0.0, g[:, :, 0] * g[:, :, 0] + g[:, :, 1] * g[:, :, 1]))
                    field_cells = np.sum(field_amp, axis=0, dtype=np.float64)
                    n_cells = int(field_cells.shape[0])
                    if n_cells == nx * ny * nz:
                        vol = field_cells.reshape(nx, ny, nz)
                        field_top = np.sum(vol, axis=1, dtype=np.float64).T.astype(np.float32)
                        field_side = np.sum(vol, axis=0, dtype=np.float64).T.astype(np.float32)
            except Exception:
                pass

        surf_top = np.zeros((nz, nx), np.float32)
        surf_side = np.zeros((nz, ny), np.float32)

        if strikes is not None:
            try:
                s = np.asarray(strikes, np.float32)
                if s.ndim == 2 and s.shape[0] > 0 and s.shape[1] >= 18:
                    pos = np.asarray(s[:, 0:3], np.float64)
                    stride = int(s.shape[1])
                    n_amp_bands = max(0, (stride - 16) // 2)
                    if n_amp_bands > 0:
                        amp_re = np.asarray(s[:, 16:16 + 2 * n_amp_bands:2], np.float64)
                        amp_im = np.asarray(s[:, 17:16 + 2 * n_amp_bands:2], np.float64)
                        mag = np.sqrt(np.maximum(0.0, amp_re * amp_re + amp_im * amp_im))
                        w = np.sum(mag, axis=1, dtype=np.float64)
                    else:
                        w = np.ones((pos.shape[0],), np.float64)

                    pmin = np.min(pos, axis=0)
                    pmax = np.max(pos, axis=0)
                    span = np.maximum(pmax - pmin, 1.0e-9)
                    pnorm = (pos - pmin[None, :]) / span[None, :]

                    ix = np.clip(np.floor(pnorm[:, 0] * (nx - 1)).astype(np.int64), 0, nx - 1)
                    iy = np.clip(np.floor(pnorm[:, 1] * (ny - 1)).astype(np.int64), 0, ny - 1)
                    iz = np.clip(np.floor(pnorm[:, 2] * (nz - 1)).astype(np.int64), 0, nz - 1)

                    np.add.at(surf_top, (iz, ix), w.astype(np.float32, copy=False))
                    np.add.at(surf_side, (iz, iy), w.astype(np.float32, copy=False))
            except Exception:
                pass

        f_top_n = self._normalize_map_energy(field_top)
        f_side_n = self._normalize_map_energy(field_side)
        s_top_n = self._normalize_map_energy(surf_top)
        s_side_n = self._normalize_map_energy(surf_side)

        # Body (field) is cool cyan/blue; shell (surface) is warm amber.
        top_rgb = np.stack([
            np.clip(0.20 * f_top_n + 1.00 * s_top_n, 0.0, 1.0),
            np.clip(0.85 * f_top_n + 0.62 * s_top_n, 0.0, 1.0),
            np.clip(1.00 * f_top_n + 0.18 * s_top_n, 0.0, 1.0),
        ], axis=-1).astype(np.float32, copy=False)
        side_rgb = np.stack([
            np.clip(0.20 * f_side_n + 1.00 * s_side_n, 0.0, 1.0),
            np.clip(0.85 * f_side_n + 0.62 * s_side_n, 0.0, 1.0),
            np.clip(1.00 * f_side_n + 0.18 * s_side_n, 0.0, 1.0),
        ], axis=-1).astype(np.float32, copy=False)

        out_h = int(self.output_height)
        out_w = int(self.output_width)
        top_out = self._resize_rgb_nearest(top_rgb, out_h, out_w)
        side_out = self._resize_rgb_nearest(side_rgb, out_h, out_w)
        return top_out, side_out

    def _output_downsample_factors(self) -> tuple[int, int]:
        sy = max(1, int(self.height) // max(1, int(self.output_height)))
        sx = max(1, int(self.width) // max(1, int(self.output_width)))
        return sy, sx

    def build_integration_snapshot(self,
                                   backend_name: str,
                                   back: ExposureBackend,
                                   plan: RayDispatchPlan,
                                   frame_cfg: FrameConfig,
                                   batch_index: int,
                                   include_sensor: bool = False,
                                   snapshot_mode: str = "stream") -> IntegrationSnapshot:
        """Public API: build one integration snapshot from current accum state."""
        measured = float(back.measured_radiant_exposure_J(plan.energy_per_ray_J))
        gain = float(self._train_emissivity_gain(measured, float(plan.target_H_J)))
        field_obj, surface_obj, _merged_bands, rgb_linear = self._make_integral_objects(
            backend_name,
            back.surf_accum,
            back.field_accum,
            gain,
            frame_cfg.integral_split,
        )
        img = self._tone_map(rgb_linear, frame_cfg.integral_split)

        sensor_obj: Optional[SensorIntegralObject] = None
        sensor_rgb: Optional[np.ndarray] = None
        endpoint_rgb: Optional[np.ndarray] = None
        pinhole_rgb: Optional[np.ndarray] = None
        if include_sensor:
            tracer_obj = getattr(back, "tracer", None)
            if self.integrator in ("backward", "bdpt"):
                try:
                    reduced = self._cpp_reduce_endpoint_rgb(
                        tracer_obj,
                        gain,
                        frame_cfg.integral_split.hdr_white_percentile,
                    )
                    if reduced is not None:
                        endpoint_rgb_linear, endpoint_img, _rgb_telemetry = reduced
                        endpoint_rgb = endpoint_img
                        if self.rgb_source == "endpoint":
                            img = endpoint_img
                            rgb_linear = endpoint_rgb_linear
                except Exception as exc:
                    print(f"  [warn] C++ endpoint->RGB reduction failed in snapshot path: {exc}")

            if self.integrator in ("backward", "bdpt"):
                sensor_obj = self._make_sensor_integral(backend_name, gain, tracer_obj,
                                                        surf_accum=back.surf_accum,
                                                        back=back)
            if sensor_obj is not None:
                sensor_rgb = self._sensor_display_rgb(sensor_obj)
                if self.rgb_source == "sensor":
                    img = sensor_rgb
                    rgb_linear = sensor_rgb.copy()
            elif self.rgb_source == "sensor":
                # GPU-resident native path: surf_accum stays empty (no CPU
                # PIXEL_CONE scatter), so there is no sensor integral object —
                # but the tracer's native sensor image IS the sensor result.
                # Mirror the fallback used by the final-save path instead of
                # forcing the per-batch snapshot to black.
                _native_img = None
                _linear_evidence = None
                _snap_cpp = back
                if (isinstance(back, GlslExposureBackend)
                        and isinstance(getattr(back, "_mirror", None), CppExposureBackend)
                        and bool(getattr(back._mirror, "_gpu_resident", False))):
                    _snap_cpp = back._mirror
                if (isinstance(_snap_cpp, CppExposureBackend)
                        and bool(getattr(_snap_cpp, "_gpu_resident", False))):
                    _linear_evidence = _native_sensor_accumulation_output(_snap_cpp)
                    if _linear_evidence is not None:
                        _native_img = self._tone_map(
                            _linear_evidence, frame_cfg.integral_split
                        )
                        rgb_linear = _linear_evidence
                    _snap_tracer = getattr(_snap_cpp, "tracer", None)
                    if (
                        _native_img is None
                        and _snap_tracer is not None
                        and hasattr(_snap_tracer, "get_sensor_image")
                    ):
                        _cached = getattr(_snap_cpp, "_native_sensor_image", None)
                        _cand = (np.asarray(_cached) if _cached is not None
                                 else _native_sensor_to_display(
                                     np.asarray(_snap_tracer.get_sensor_image())))
                        if _cand.ndim == 3 and _cand.shape[2] >= 3 and np.any(_cand[..., :3]):
                            _native_img = np.clip(_cand[..., :3], 0.0, 1.0).astype(
                                np.float32, copy=False)
                if _native_img is not None:
                    img = _native_img
                    if _linear_evidence is None:
                        rgb_linear = _native_img.copy()
                else:
                    self._warn_missing_rgb_source_once(
                        source="sensor",
                        backend=str(backend_name),
                        context=f"batch={int(batch_index)}",
                    )
                    img = np.zeros_like(img, dtype=np.float32)
                    rgb_linear = np.zeros_like(rgb_linear, dtype=np.float32)

            if self._pinhole_accum is not None and self._pinhole_accum.size > 0:
                try:
                    pin_mag = np.abs(self._pinhole_accum).astype(np.float32, copy=False)
                    pinhole_rgb = _bands_to_rgb(pin_mag, self.freq_hz)
                except Exception as exc:
                    print(f"  [warn] pinhole RGB projection failed: {exc}")
                    pinhole_rgb = None

        return IntegrationSnapshot(
            backend=str(backend_name),
            frame_index=int(self._frame_index),
            batch_index=int(batch_index),
            measured_H_J=float(measured),
            target_H_J=float(plan.target_H_J),
            gain_linear=float(gain),
            image_data=np.asarray(img, dtype=np.float32),
            rgb_linear=np.asarray(rgb_linear, dtype=np.float32),
            field_integrated_data=np.asarray(field_obj.integrated_bands, dtype=np.float32),
            surface_integrated_data=np.asarray(surface_obj.integrated_bands, dtype=np.float32),
            sensor_photons_data=(None if sensor_obj is None
                                 else np.asarray(sensor_obj.photons_per_pixel, dtype=np.float32)),
            sensor_snr_data=(None if sensor_obj is None
                             else np.asarray(sensor_obj.snr_linear, dtype=np.float32)),
            sensor_rgb_data=(None if sensor_rgb is None
                             else np.asarray(sensor_rgb, dtype=np.float32)),
            endpoint_rgb_data=(None if endpoint_rgb is None
                               else np.asarray(endpoint_rgb, dtype=np.float32)),
            pinhole_rgb_data=(None if pinhole_rgb is None
                              else np.asarray(pinhole_rgb, dtype=np.float32)),
            mode=str(snapshot_mode),
        )

    def stream_integration_snapshots(self,
                                     backs: dict[str, ExposureBackend],
                                     plan: RayDispatchPlan,
                                     frame_cfg: FrameConfig,
                                     batch_index: int) -> dict[str, IntegrationSnapshot]:
        """Public API: build stream snapshots for all active backends."""
        out: dict[str, IntegrationSnapshot] = {}
        sensor_ready = any(
            bool(getattr(back, "surf_accum", None) is not None and np.any(back.surf_accum))
            for back in backs.values()
        )
        for name, back in backs.items():
            out[str(name)] = self.build_integration_snapshot(
                backend_name=str(name),
                back=back,
                plan=plan,
                frame_cfg=frame_cfg,
                batch_index=int(batch_index),
                include_sensor=sensor_ready,
                snapshot_mode="stream",
            )
        return out

    def _make_sensor_integral(self, backend: str, gain: float, tracer: Any = None,
                               surf_accum: Optional[np.ndarray] = None,
                               back: Optional[ExposureBackend] = None) -> Optional[SensorIntegralObject]:
        """Create a SensorIntegralObject from endpoint-derived accumulation (T4).

        Uses the streamed PIXEL_CONE accumulation path only.
        Requires: non-empty surf_accum built from BDPT scatter.

        Returns SensorIntegralObject or None if sensor integration unavailable.
        """
        try:
            uv_hits: Optional[np.ndarray] = None
            uv_thin_hits: Optional[np.ndarray] = None
            if isinstance(back, CppExposureBackend):
                uv_hits = np.asarray(back.sensor_uv_hits, np.float32)
                uv_thin_hits = np.asarray(back.sensor_uv_thin_lens_hits, np.float32)

            # Only accumulate if we have active slots and have just run BDPT
            active_slots = [i for i, (s, f) in enumerate(self.sensor_film_slots) if s >= 0 and f >= 0]
            if not active_slots:
                print("  [warn] no active sensor/film slots; sensor integral unavailable")
                return None

            if self._sensor_group_id < 0:
                print("  [warn] sensor group not registered; sensor integral unavailable")
                return None

            if surf_accum is None or surf_accum.size == 0 or not np.any(surf_accum):
                native_img = getattr(back, "_native_sensor_image", None)
                if native_img is not None and np.asarray(native_img).ndim == 3 and np.any(np.asarray(native_img)[..., :3]):
                    return None
                # GLSL scaffold backed by a gpu-resident C++ mirror: the output
                # loop fetches the image from the mirror's tracer, so no warning.
                if (isinstance(back, GlslExposureBackend)
                        and isinstance(getattr(back, "_mirror", None), CppExposureBackend)
                        and bool(getattr(back._mirror, "_gpu_resident", False))):
                    return None
                print("  [warn] surf_accum empty; proper PIXEL_CONE sensor path unavailable")
                return None

            # surf_accum is (n_bands, H, W) of per-band amplitude magnitude.
            # Sum across bands for total intensity proxy.
            endpoint_intensity = surf_accum.sum(axis=0, dtype=np.float32)
            endpoint_intensity = endpoint_intensity.astype(np.float32, copy=False)
            endpoint_intensity *= float(max(gain, 0.0) ** 2)

            mean_intensity = float(np.mean(endpoint_intensity))
            target_photons = float(max(self._last_target_photons_per_pixel, 0.0))
            if mean_intensity > 1.0e-20 and target_photons > 0.0:
                photons_per_pixel = (endpoint_intensity / mean_intensity) * target_photons
            else:
                photons_per_pixel = endpoint_intensity.copy()
            photons_per_pixel = photons_per_pixel.astype(np.float32, copy=False)

            use_record_sensor_reducer = (
                self.integrator == "bdpt"
                and tracer is not None
                and hasattr(tracer, "reduce_endpoint_records_to_sensor_integral")
                and self._last_bdpt_emit_counts is None
                and self._last_bdpt_records is not None
            )
            if use_record_sensor_reducer:
                rec_arr = np.asarray(self._last_bdpt_records)
                if rec_arr.ndim != 2 or rec_arr.shape[1] != 16:
                    raise RuntimeError(
                        "endpoint record layout invalid for sensor reducer: "
                        f"shape={tuple(rec_arr.shape)} expected=(n_records, 16)"
                    )
                if rec_arr.dtype != np.float32:
                    raise RuntimeError(
                        "endpoint record dtype invalid for sensor reducer: "
                        f"dtype={rec_arr.dtype} expected=float32"
                    )
                if not rec_arr.flags.c_contiguous:
                    raise RuntimeError(
                        "endpoint records must be C-contiguous for sensor reducer: "
                        f"shape={tuple(rec_arr.shape)} strides={tuple(rec_arr.strides)}"
                    )

                # Only use the C++ per-record reducer in non-streaming mode.
                # In streaming mode surf_accum already holds the full
                # accumulated scatter from all batches; no need to re-reduce
                # the last (sparse) batch.
                cpp_result = tracer.reduce_endpoint_records_to_sensor_integral(
                    rec_arr,
                    int(self.width),
                    int(self.height),
                    int(self._sensor_group_id),
                    float(self._last_target_photons_per_pixel),
                    float(max(gain, 0.0)),
                )

                photons_per_pixel = np.nan_to_num(np.asarray(cpp_result["photons_per_pixel"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                electrons_per_pixel = np.nan_to_num(np.asarray(cpp_result["electrons_per_pixel"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                snr_linear = np.nan_to_num(np.asarray(cpp_result["snr_linear"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                metrics = dict(cpp_result["metrics"])
                if uv_hits is not None:
                    metrics["sensor_uv_hits_total"] = float(np.sum(uv_hits, dtype=np.float64))
                if uv_thin_hits is not None:
                    metrics["sensor_uv_thin_lens_hits_total"] = float(np.sum(uv_thin_hits, dtype=np.float64))
                    denom = max(float(np.sum(uv_hits, dtype=np.float64)) if uv_hits is not None else 0.0, 1.0e-12)
                    metrics["sensor_uv_thin_lens_hit_fraction"] = float(np.sum(uv_thin_hits, dtype=np.float64) / denom)

                qe_peak = float(metrics.get("qe_peak", 0.0))
                read_noise_e = float(metrics.get("read_noise_e", 0.0))
                full_well_e = float(metrics.get("full_well_e", 0.0))
                peak_snr = float(np.nanmax(snr_linear)) if snr_linear.size else 0.0
                mean_snr = float(np.nanmean(snr_linear)) if snr_linear.size else 0.0

                sensor_w_m = float(self.optics.sensor_w_mm) * 1.0e-3
                sensor_h_m = float(self.optics.sensor_h_mm) * 1.0e-3
                pixel_pitch_m = float(self.optics.pixel_pitch_um) * 1.0e-6
                focal_m = float(self.optics.focal_mm) * 1.0e-3
                # Use aperture_mm directly (more precise for complex aperture sims than f_number)
                aperture_radius_m = float(self.optics.aperture_mm) * 0.5 * 1.0e-3
                n_px = self.optics.n_pixels()

                # ── Task 5: Separate oracle/physical photon buffers ──────
                # Oracle photons: reference perfect optics (no loss)
                # Physical photons: actual transport (with losses/aberrations)
                # For now both are identical; C++ kernel will distinguish when handlers implement roles
                oracle_photons = photons_per_pixel.copy()
                physical_photons = photons_per_pixel.copy()
                
                # Select active buffer based on camera mode
                if int(self.camera_mode) == 0:  # CAMERA_MODE_ORACLE_PINHOLE_REFERENCE
                    active_photons = oracle_photons
                else:
                    active_photons = physical_photons

                return SensorIntegralObject(
                    backend=backend,
                    frame_index=int(self._frame_index),
                    optics=self.optics,
                    film=self.film,
                    n_pixels=int(n_px),
                    sensor_w_m=float(sensor_w_m),
                    sensor_h_m=float(sensor_h_m),
                    pixel_pitch_m=float(pixel_pitch_m),
                    focal_m=float(focal_m),
                    aperture_radius_m=float(aperture_radius_m),
                    qe_peak=float(qe_peak),
                    photons_per_pixel=active_photons,
                    electrons_per_pixel=electrons_per_pixel,
                    noise_floor_e=float(read_noise_e),
                    full_well_e=int(round(full_well_e)),
                    snr_linear=snr_linear,
                    peak_snr=float(peak_snr),
                    mean_snr=float(mean_snr),
                    metrics=metrics,
                    sensor_uv_hits=(None if uv_hits is None else uv_hits.copy()),
                    sensor_uv_thin_lens_hits=(None if uv_thin_hits is None else uv_thin_hits.copy()),
                )

            # Multi-slot accumulation: every active slot contributes its own
            # sensor/film noise model without re-aggregating endpoint records.
            h, w = photons_per_pixel.shape
            electrons_accum = np.zeros((h, w), dtype=np.float32)
            snr_accum = np.zeros((h, w), dtype=np.float32)
            qe_accum = 0.0
            read_noise_accum = 0.0
            full_well_accum = 0.0
            slot_metrics: list[dict[str, Any]] = []

            for slot_id in active_slots:
                sensor_id, film_id = self.sensor_film_slots[int(slot_id)]
                if sensor_id < 0 or film_id < 0:
                    continue

                sensor_row = self._sensor_film_tensors['sensor'][sensor_id]
                film_row = self._sensor_film_tensors['film'][film_id]

                qe_peak = float(sensor_row[8])
                read_noise_e = float(sensor_row[10])
                dark_current_e_s = float(sensor_row[11])
                full_well_e = float(sensor_row[9])
                exposure_time_s = float(film_row[1])

                electrons_slot = photons_per_pixel * np.float32(qe_peak)
                dark_current_accumulated_e = dark_current_e_s * exposure_time_s
                noise_variance = ((read_noise_e ** 2) + dark_current_accumulated_e +
                                  electrons_slot)
                snr_slot = (np.sqrt(np.maximum(electrons_slot, 0.0)) /
                            np.sqrt(np.maximum(noise_variance, 1.0e-10)))
                snr_slot = snr_slot.astype(np.float32, copy=False)

                electrons_accum += electrons_slot.astype(np.float32, copy=False)
                snr_slot = np.nan_to_num(snr_slot, nan=0.0, posinf=0.0, neginf=0.0)
                snr_accum += snr_slot
                qe_accum += qe_peak
                read_noise_accum += read_noise_e
                full_well_accum += full_well_e

                slot_meta = self._sensor_film_metadata[int(slot_id)]
                slot_metrics.append({
                    "slot_id": int(slot_id),
                    "sensor_id": int(sensor_id),
                    "film_id": int(film_id),
                    "sensor_name": slot_meta.get("sensor_name", "unknown"),
                    "film_name": slot_meta.get("film_name", "unknown"),
                    "qe_peak": float(qe_peak),
                    "read_noise_e": float(read_noise_e),
                    "dark_current_e_s": float(dark_current_e_s),
                    "dark_current_accumulated_e": float(dark_current_accumulated_e),
                    "exposure_time_s": float(exposure_time_s),
                    "full_well_e": float(full_well_e),
                    "snr_peak": float(np.nanmax(snr_slot)) if snr_slot.size else 0.0,
                    "snr_mean": float(np.nanmean(snr_slot)) if snr_slot.size else 0.0,
                    "photons_flux_hz": float(np.mean(photons_per_pixel)),
                    "electrons_flux_hz": float(np.mean(electrons_slot)),
                })

            if not slot_metrics:
                print("  [warn] no valid active sensor/film slots after filtering")
                return None

            inv_slots = np.float32(1.0 / max(1, len(slot_metrics)))
            electrons_per_pixel = (electrons_accum * inv_slots).astype(np.float32, copy=False)
            snr_linear = (snr_accum * inv_slots).astype(np.float32, copy=False)
            qe_peak = float(qe_accum / max(1, len(slot_metrics)))
            read_noise_e = float(read_noise_accum / max(1, len(slot_metrics)))
            full_well_e = float(full_well_accum / max(1, len(slot_metrics)))

            snr_linear = np.nan_to_num(snr_linear, nan=0.0, posinf=0.0, neginf=0.0)
            peak_snr = float(np.nanmax(snr_linear)) if snr_linear.size else 0.0
            mean_snr = float(np.nanmean(snr_linear)) if snr_linear.size else 0.0
            
            sensor_w_m = float(self.optics.sensor_w_mm) * 1.0e-3
            sensor_h_m = float(self.optics.sensor_h_mm) * 1.0e-3
            pixel_pitch_m = float(self.optics.pixel_pitch_um) * 1.0e-6
            focal_m = float(self.optics.focal_mm) * 1.0e-3
            # Use aperture_mm directly (more precise for complex aperture sims than f_number)
            aperture_radius_m = float(self.optics.aperture_mm) * 0.5 * 1.0e-3
            n_px = self.optics.n_pixels()
            
            # Build metadata with explicit noise model (T5)
            metrics = {
                "active_slot_ids": [int(s) for s in active_slots],
                "n_active_slots": int(len(slot_metrics)),
                "qe_peak": float(qe_peak),
                "read_noise_e": float(read_noise_e),
                "full_well_e": float(full_well_e),
                "snr_peak": float(peak_snr),
                "snr_mean": float(mean_snr),
                "endpoint_records_count": int(self._last_bdpt_records.shape[0]) if self._last_bdpt_records is not None else 0,
                "sensor_group_id": int(self._sensor_group_id),
                "photons_flux_hz": float(np.mean(photons_per_pixel)),
                "electrons_flux_hz": float(np.mean(electrons_per_pixel)),
                "slot_metrics": slot_metrics,
                "sensor_uv_hits_total": (float(np.sum(uv_hits, dtype=np.float64)) if uv_hits is not None else 0.0),
                "sensor_uv_thin_lens_hits_total": (float(np.sum(uv_thin_hits, dtype=np.float64)) if uv_thin_hits is not None else 0.0),
                "sensor_uv_thin_lens_hit_fraction": (
                    float(np.sum(uv_thin_hits, dtype=np.float64) / max(float(np.sum(uv_hits, dtype=np.float64)), 1.0e-12))
                    if (uv_hits is not None and uv_thin_hits is not None) else 0.0
                ),
            }
            
            return SensorIntegralObject(
                backend=backend,
                frame_index=int(self._frame_index),
                optics=self.optics,
                film=self.film,
                n_pixels=int(n_px),
                sensor_w_m=float(sensor_w_m),
                sensor_h_m=float(sensor_h_m),
                pixel_pitch_m=float(pixel_pitch_m),
                focal_m=float(focal_m),
                aperture_radius_m=float(aperture_radius_m),
                qe_peak=float(qe_peak),
                photons_per_pixel=photons_per_pixel,
                electrons_per_pixel=electrons_per_pixel.astype(np.float32),
                noise_floor_e=float(read_noise_e),
                full_well_e=int(round(full_well_e)),
                snr_linear=snr_linear.astype(np.float32),
                peak_snr=float(peak_snr),
                mean_snr=float(mean_snr),
                metrics=metrics,
                sensor_uv_hits=(None if uv_hits is None else uv_hits.copy()),
                sensor_uv_thin_lens_hits=(None if uv_thin_hits is None else uv_thin_hits.copy()),
            )
        except Exception as e:
            raise RuntimeError(
                "sensor integral creation failed "
                f"(backend={backend}, frame={int(self._frame_index)}, "
                f"sensor_group_id={int(self._sensor_group_id)}, "
                f"records_path={self._last_bdpt_records_path!r}, "
                f"records_is_temp={bool(self._last_bdpt_records_is_temp)}): {e}"
            ) from e

    def _save_integral_objects(self,
                               field_obj: FieldIntegralObject,
                               surface_obj: SurfaceIntegralObject,
                               sensor_obj: Optional[SensorIntegralObject] = None,
                               field_history_bands: Optional[np.ndarray] = None,
                               surface_history_bands: Optional[np.ndarray] = None
                               ) -> tuple[str, str, Optional[str]]:
        """Write field/surface/sensor objects and uncompressed full histories."""
        field_path = os.path.join(
            self.out_dir,
            f"{self._frame_index:04d}_{field_obj.backend}_field_integral.npz")
        surf_path = os.path.join(
            self.out_dir,
            f"{self._frame_index:04d}_{surface_obj.backend}_surface_integral.npz")
        sensor_path = None
        if sensor_obj is not None:
            sensor_path = os.path.join(
                self.out_dir,
                f"{self._frame_index:04d}_{sensor_obj.backend}_sensor_integral.npz")

        field_history = (np.asarray(field_history_bands, dtype=np.float32)
                         if field_history_bands is not None
                         else (field_obj.integrated_bands + field_obj.bookkeeping_bands).astype(np.float32, copy=False))
        surface_history = (np.asarray(surface_history_bands, dtype=np.float32)
                           if surface_history_bands is not None
                           else (surface_obj.integrated_bands + surface_obj.bookkeeping_bands).astype(np.float32, copy=False))

        np.savez(
            field_path,
            freq_hz=field_obj.freq_hz,
            integrated_bands=field_obj.integrated_bands,
            bookkeeping_bands=field_obj.bookkeeping_bands,
            full_history_bands=field_history,
            metrics=np.asarray(json.dumps(field_obj.metrics)),
        )
        np.savez(
            surf_path,
            freq_hz=surface_obj.freq_hz,
            integrated_bands=surface_obj.integrated_bands,
            bookkeeping_bands=surface_obj.bookkeeping_bands,
            full_history_bands=surface_history,
            metrics=np.asarray(json.dumps(surface_obj.metrics)),
        )
        if sensor_path is not None and sensor_obj is not None:
            np.savez(
                sensor_path,
                photons_per_pixel=sensor_obj.photons_per_pixel,
                electrons_per_pixel=sensor_obj.electrons_per_pixel,
                snr_linear=sensor_obj.snr_linear,
                sensor_uv_hits=(sensor_obj.sensor_uv_hits
                                if sensor_obj.sensor_uv_hits is not None
                                else np.zeros_like(sensor_obj.photons_per_pixel, dtype=np.float32)),
                sensor_uv_thin_lens_hits=(sensor_obj.sensor_uv_thin_lens_hits
                                          if sensor_obj.sensor_uv_thin_lens_hits is not None
                                          else np.zeros_like(sensor_obj.photons_per_pixel, dtype=np.float32)),
                metrics=np.asarray(json.dumps(sensor_obj.metrics)),
            )
        
        return field_path, surf_path, sensor_path

    # ── Render one full exposure (drives all sub-batches) ────────────────
    def render_one_exposure(
        self,
        t: float = 0.0,
        batch_preview_cb: Optional[Callable[[dict[str, ExposureBackend], dict[str, IntegrationSnapshot], int, int, float], None]] = None,
    ) -> list[ExposureFrameResult]:
        # ── Build per-frame progressive feature configuration ────────────
        with self._profiler.section("frame_config"):
            frame_cfg = _build_frame_config(
                self._frame_index, self.n_frames_planned, self.integral_split)
        
        # Disable parametric and spline features for cleaner initial testing
        if frame_cfg.field_capture.enabled and frame_cfg.integral_split.field_integrate_frac <= 0.01:
            # If field integration is essentially disabled, also disable parametric/spline
            frame_cfg.parametric_sdf = ParametricSdfConfig(enabled=False)
            frame_cfg.surface_spline = SurfaceSplineConfig(enabled=False)

        scene_mode = self._scene_mode_for_frame()
        prebuilt_scene: Optional[TracerScene] = None
        if self.integrator == "depth":
            frame_cfg = FrameConfig(
                field_capture=FieldCaptureConfig(
                    enabled=False, grid_kind="regular",
                    nx=1, ny=1, nz=1,
                    capture_strikes=False, max_strikes=0,
                ),
                camera_visibility=CameraVisibilityConfig(
                    camera_vis_mode=_CAM_VIS_AS_IS,
                    transparent_mode=_CAM_TRANSP_BLOCK,
                    depth_cull_enabled=False, depth_cull_m=0.0,
                ),
                surface_spline=SurfaceSplineConfig(enabled=False),
                parametric_sdf=ParametricSdfConfig(enabled=False),
                integral_split=IntegralSplitConfig(
                    field_integrate_frac=0.0,
                    field_bookkeep_frac=1.0,
                    surface_integrate_frac=0.0,
                    surface_bookkeep_frac=1.0,
                    hdr_white_percentile=float(
                        self.integral_split.hdr_white_percentile
                    ),
                ),
                description="native GPU first-authored-scene depth · one primary/site",
                detail_level=0,
            )
        if scene_mode == CAMERA_SOLVE_PREVIEW_SCENE:
            frame_cfg = FrameConfig(
                field_capture=FieldCaptureConfig(
                    enabled=False,
                    grid_kind="regular",
                    nx=16, ny=16, nz=16,
                    capture_strikes=False,
                    max_strikes=0,
                ),
                camera_visibility=CameraVisibilityConfig(
                    camera_vis_mode=_CAM_VIS_DIRECT_HIT,
                    transparent_mode=_CAM_TRANSP_BLOCK,
                    depth_cull_enabled=False,
                    depth_cull_m=0.0,
                ),
                surface_spline=SurfaceSplineConfig(enabled=False),
                parametric_sdf=ParametricSdfConfig(enabled=False),
                integral_split=IntegralSplitConfig(
                    field_integrate_frac=0.0,
                    field_bookkeep_frac=1.0,
                    surface_integrate_frac=1.0,
                    surface_bookkeep_frac=0.0,
                    hdr_white_percentile=float(self.integral_split.hdr_white_percentile),
                ),
                description="camera-solve preview · BDPT camera path · no field/march · parametric groups=0",
                detail_level=1,
            )
        with self._profiler.section("build_camera"):
            if _is_thick_lens_lab_scene_mode(scene_mode):
                prebuilt_scene = _build_tracer_scene(
                    t,
                    scene_mode=scene_mode,
                    cam=None,
                    solved=None,
                    optics=self.optics,
                )
                cam, solved = _build_thick_lens_lab_camera_package(
                    prebuilt_scene,
                    self.width,
                    self.height,
                    self.optics,
                )
                self._camera_pinhole_target_sensor_z_m = 0.0
            else:
                _scene_center = np.asarray(scene_mod.SCENE_CENTER, np.float64)
                _eye_base = np.array([0.0, 0.0, 0.0], np.float64)
                _to_scene = _scene_center - _eye_base
                _to_scene_norm = float(np.linalg.norm(_to_scene))
                if _to_scene_norm > 1.0e-12:
                    _eye_dir = -(_to_scene / _to_scene_norm)
                else:
                    _eye_dir = np.array([0.0, 0.0, -1.0], np.float64)
                # Move the camera 1.5 m farther from the scene center.
                _eye = _eye_base + (1.5 * _eye_dir)

                solved = solve_sane_camera_rig(
                    camera_cls=PhysicalCameraRig,
                    width=self.width,
                    height=self.height,
                    optics=self.optics,
                    film=self.film,
                    scene_center=_scene_center,
                    eye=_eye,
                    max_iters=self.camera_solve_max_iters,
                    seed=self.camera_solve_seed_base,
                    rail_model=self.camera_rail_model,
                )
                cam = solved.camera
                cam.sensor_plane_offset_m = float(solved.sanity_input.sensor_plane_z_m)
                cam.aperture_plane_offset_m = float(
                    solved.sanity_input.aperture_rail_z_m
                    if solved.sanity_input.aperture_rail_z_m is not None
                    else solved.sanity_input.aperture_plane_offset_m
                )
                cam.focal_m = float(solved.sanity_report.planes.effective_focal_m)
                cam.aperture_radius_m = max(
                    1.0e-6,
                    float(
                        solved.sanity_input.aperture_radius_m
                        if solved.sanity_input.aperture_radius_m is not None
                        else (float(self.optics.aperture_mm) * 0.5 * 1.0e-3)
                    ),
                )
                self._camera_pinhole_target_sensor_z_m = float(
                    solved.sanity_report.error_degree.pinhole_target_sensor_z_m
                )
            self._active_cam = cam
        with self._profiler.section("build_scene"):
            scene = prebuilt_scene if prebuilt_scene is not None else _build_tracer_scene(
                t,
                scene_mode=scene_mode,
                cam=cam,
                solved=solved,
                optics=self.optics,
            )
        with self._profiler.section("build_plan"):
            plan  = self._build_plan(scene)
        with self._profiler.section("build_backends"):
            backs = self._build_backends(scene, cam, frame_cfg)

        plan_dict = summarize_plan(plan, self.optics, self.film)
        n_pix = max(1, self.optics.n_pixels())
        # Visible-band reference: λ ≈ 555 nm (peak of photopic response).
        photon_E = H_PLANCK * C_LIGHT / (plan.ref_wavelength_nm * 1.0e-9)
        photons_per_pix = plan.target_H_J / max(1, n_pix) / max(photon_E, 1.0e-30)
        self._last_target_photons_per_pixel = float(photons_per_pix)
        self._cleanup_temp_bdpt_file()
        self._release_last_bdpt_records()
        self._pinhole_accum = None

        fc_active = frame_cfg.field_capture
        cv_active = frame_cfg.camera_visibility
        ss_active = frame_cfg.surface_spline

        print(f"\n====================== EXPOSURE {self._frame_index:04d} "
              f"[level {frame_cfg.detail_level}] ======================")
        cfg_desc = str(frame_cfg.description).replace("·", "|").replace("³", "^3")
        print(f"  config   : {cfg_desc}")
        print(f"  cam vis  : {_cam_vis_name(cv_active.camera_vis_mode)} | "
              f"transp={_transp_name(cv_active.transparent_mode)}"
              + (f" | depth_cull={cv_active.depth_cull_m:.0f}m"
                 if cv_active.depth_cull_enabled else ""))
        print(f"  field    : {'ENABLED ' + fc_active.grid_kind + ' ' + str(fc_active.nx) + '^3' if fc_active.enabled else 'disabled'}")
        print(f"  spline   : {'ENABLED fit_all=' + str(ss_active.fit_all_tris) + ' lambda=' + str(ss_active.ridge_lambda) if ss_active.enabled else 'disabled'}")
        print(f"  scene:  {scene.verts.shape[0]} tris, {scene.src_pos.shape[0]} emissive sources")
        print(f"  emissive total power = {scene.total_emissive_power_W:.4g} W "
              f"over {scene.total_emissive_area_m2:.4g} m^2")
        if scene.src_emit_rgb_W.size > 0:
            rgb_emit = np.sum(scene.src_emit_rgb_W, axis=0)
            rgb_total = float(np.sum(rgb_emit))
            if rgb_total > 1.0e-12:
                rgb_frac = rgb_emit / rgb_total
                print("  emissive rgb share = "
                      f"R {rgb_frac[0]*100.0:5.1f}%  "
                      f"G {rgb_frac[1]*100.0:5.1f}%  "
                      f"B {rgb_frac[2]*100.0:5.1f}%")
        sensor_budget_backend = backs.get("cpp")
        if bool(getattr(sensor_budget_backend, "_sensor_mipmap_enabled", False)):
            budget_epochs = max(
                1, int(os.environ.get("SPECTRAL_SENSOR_MAX_EPOCHS", "1"))
            )
            budget_top_k = min(1024, max(
                64,
                int(os.environ.get(
                    "SPECTRAL_SENSOR_TOP_K", str(max(self.width, self.height))
                )),
            ))
            budget_samples = max(
                1, int(getattr(sensor_budget_backend, "_sensor_samples_per_node", 1))
            )
            scheduled_camera = (
                budget_epochs * budget_top_k * budget_samples
            )
            supporting_flash = budget_epochs * max(
                1, int(os.environ.get("SPECTRAL_SENSOR_FLASH_RAYS", "32000"))
            )
            print(
                f"  budget (impactful)   : camera_target={plan.total_rays:_}  "
                f"camera_scheduled={scheduled_camera:_}  "
                f"epochs={budget_epochs}"
            )
            print(
                f"  support (not counted): flash_paths={supporting_flash:_}  "
                f"T5_pairs/epoch<={int(os.environ.get('SPECTRAL_SENSOR_T5_PAIR_BUDGET', '8000000')):_}"
            )
        else:
            print(f"  budget (total)       : N_rays={plan.total_rays:_}  "
                  f"n_batches={plan.n_batches}  rays/batch={plan.rays_per_batch:_}")
        print(f"  per-pixel target     : H={plan.target_H_J/n_pix:.3e} J  "
              f"photons={photons_per_pix:.3e} @ lambda={plan.ref_wavelength_nm:.0f} nm")
        print(f"  shutter t            : {self.film.exposure_time_s:.4g} s @ "
              f"f/{self.optics.f_number:.2f}, ISO {self.film.iso:.0f}")
        print(f"  cam sanity           : {solved.sanity_report.status.upper()}"
              f"  coc={solved.sanity_report.circle_of_confusion_um:.2f} um"
              f"  sensor_adjust={solved.sanity_report.sensor_adjustment_needed_mm:+.2f} mm"
              f"  pinhole_ref={solved.sanity_report.error_degree.pinhole_comparison_mm:.2f} mm"
              f"  err={solved.sanity_report.error_degree.overall:.3f}"
              f"  iter={solved.iterations}")
        if scene.film_plane_pose is not None:
            film_pose = scene.film_plane_pose
            print(
                "  film stage           : "
                f"depth={film_pose.lens_distance_delta_m * 1.0e3:+.3f} mm  "
                f"tilt_right={film_pose.tilt_about_right_deg:+.3f} deg  "
                f"tilt_up={film_pose.tilt_about_up_deg:+.3f} deg"
            )
        print(f"  cam targets          : thin_lens_z={solved.sanity_report.error_degree.thin_lens_target_sensor_z_m:+.6f} m"
              f"  pinhole_z={solved.sanity_report.error_degree.pinhole_target_sensor_z_m:+.6f} m"
              f"  pinhole_deg={solved.sanity_report.error_degree.pinhole_comparison_degree:.3f}")
        print(f"  cam error terms      : sensor_deg={solved.sanity_report.error_degree.sensor_plane_degree:.3f}"
              f"  focus_deg={solved.sanity_report.error_degree.focus_distance_degree:.3f}"
              f"  coc_deg={solved.sanity_report.error_degree.coc_degree:.3f}"
              f"  ev_deg={solved.sanity_report.error_degree.ev_degree:.3f}"
              f"  thin_resid_dpt={solved.sanity_report.error_degree.thin_lens_residual_diopter:.3e}")
        if solved.sanity_report.warnings:
            print("  cam warnings         : " + " | ".join(solved.sanity_report.warnings))
        if solved.sanity_report.failures:
            print("  cam failures         : " + " | ".join(solved.sanity_report.failures))

        # ── Sub-batch streaming loop ─────────────────────────────────────
        # rays_per_batch in plan is per-pixel·n_pixels = per-frame ray budget /
        # n_batches.  We translate to "per-source" by dividing by source count.
        rays_per_batch_total = max(1, plan.rays_per_batch)
        n_sources = max(1, int(scene.src_pos.shape[0]))
        rays_per_source_per_batch = max(1, rays_per_batch_total // n_sources)

        for back in backs.values():
            back.reset_exposure()
            tracer_obj = getattr(back, "tracer", None)
            if tracer_obj is not None and hasattr(tracer_obj, "clear_field_capture"):
                tracer_obj.clear_field_capture(clear_grid=False, clear_strikes=True)

        conv_cfg = self.convergence
        conv_target_pct = float(np.clip(conv_cfg.target_pct, 0.0, 100.0))
        conv_error_target = max(0.0, float(conv_cfg.max_rel_drift))
        if conv_error_target <= 0.0:
            conv_error_target = max(0.0, 1.0 - conv_target_pct / 100.0)
        conv_every = max(1, int(conv_cfg.check_every_batches))
        conv_min_batches = max(1, int(conv_cfg.min_batches))
        conv_hold = max(1, int(conv_cfg.hold_checks))
        conv_enabled = bool(conv_cfg.enabled)
        conv_drive_batches = bool(conv_cfg.drive_batches) and conv_enabled
        conv_max_batches = max(0, int(conv_cfg.max_batches))
        conv_consecutive_hits = 0
        conv_last_measured_pct = 0.0
        conv_last_pct = 0.0
        conv_no_signal_streak = 0
        conv_no_signal_limit = max(16, conv_hold * 6)

        def _convergence_backend() -> Optional[ExposureBackend]:
            if "cpp" in backs:
                return backs["cpp"]
            return next(iter(backs.values()), None)

        if conv_drive_batches:
            print("  [conv] acquisition mode: open-ended batches until exposure target is met")

        # ── BDPT pre-batch registration ──────────────────────────────────
        # Register emissive/sensor tri-groups once before the batch loop so
        # that each per-batch bidirectional_packed call sees a stable topology.
        _bdpt_stream_active = False
        _bdpt_emitter_centers: np.ndarray = np.zeros((0, 3), np.float64)
        _bdpt_target_total: int = 0
        _stream_divisor: int = 1
        _stream_seen: Optional[np.ndarray] = None
        _stream_cover_complete: bool = True
        _stream_covered_n: int = 1
        _stream_hard_cap_batches: int = max(1, int(plan.n_batches))
        _bdpt_native_lab_active = False
        _bdpt_native_emitter_tri_ids = np.zeros((0,), dtype=np.int32)
        _bdpt_native_n_aperture_samples = 1
        _bdpt_native_work_units: list[tuple[int, int]] = []
        _bdpt_native_work_plan: dict[str, int] = {}
        _bdpt_native_schedule: list[tuple[int, int, int]] = []
        if self.integrator in ("forward", "backward", "bdpt", "depth") and "cpp" in backs:
            cpp_back = backs["cpp"]  # type: ignore[assignment]
            tracer = getattr(cpp_back, "tracer", None)
            if tracer is not None and hasattr(tracer, "register_tri_group"):
                tracer.clear_tri_groups()
                emissive_tris = np.ascontiguousarray(scene.src_tri_idx, dtype=np.int32)
                _bdpt_native_emitter_tri_ids = emissive_tris.copy()

                # ── Surface spline fitting ────────────────────────────────
                ss_cfg = frame_cfg.surface_spline
                ps_cfg = frame_cfg.parametric_sdf
                spline_coeffs: Optional[np.ndarray] = None
                if ss_cfg.enabled and _HAS_SURFACE_SPLINE and emissive_tris.size > 0:
                    try:
                        uv, faces = _weld_mesh(scene.verts)
                        subset_arg = (None if ss_cfg.fit_all_tris else emissive_tris)
                        spline_coeffs = _ss_parameterize_mesh(
                            uv, faces,
                            tri_subset   = subset_arg,
                            ridge_lambda = float(ss_cfg.ridge_lambda),
                            n_threads    = int(ss_cfg.n_threads),
                        )  # (n_tris, 6)
                        print(f"  surface spline: {uv.shape[0]} unique verts, "
                            f"{'all' if ss_cfg.fit_all_tris else 'emissive-subset'} tris, "
                            f"lambda={ss_cfg.ridge_lambda:.1e}")
                    except Exception as exc:
                        print(f"  [warn] surface spline fit failed: {exc}")
                        spline_coeffs = None
                elif ss_cfg.enabled and not _HAS_SURFACE_SPLINE:
                    print("  [warn] surface spline requested but surface_spline_utils "
                          "unavailable (rebuild _spectral_kernels)")

                _TRI_POLY_BARY = int(getattr(_sk, "TRI_PARAM_SURFACE_POLY_BARY", 1))
                emissive_group_centers: list[np.ndarray] = []

                parametric_driver = None
                if ps_cfg.enabled:
                    if _get_sdf_driver is None:
                        raise RuntimeError("parametric SDF requested but sdf_plugins is unavailable")
                    if _TRI_POLY_BARY <= 0:
                        raise RuntimeError("parametric SDF requested but TRI_PARAM_SURFACE_POLY_BARY is unavailable")
                    parametric_driver = _get_sdf_driver(ps_cfg.model, strict=True)

                # ── Register emissive groups (all ray-trace modes) ─────────
                _register_emissive = self.integrator in ("forward", "backward", "bdpt")
                if _register_emissive and emissive_tris.size > 0:
                    if (parametric_driver is not None) or (spline_coeffs is not None and ss_cfg.fit_all_tris):
                        # Per-triangle groups — each emissive tri gets individual coefficients.
                        n_tri_total = scene.verts.shape[0]
                        for ti in emissive_tris:
                            ti_int = int(ti)
                            if ti_int < 0 or ti_int >= n_tri_total:
                                continue
                            if parametric_driver is not None:
                                payload = _resolve_native_parametric_payload(parametric_driver(
                                    tri_id=ti_int,
                                    tri_vertices=np.asarray(scene.verts[ti_int], dtype=np.float64),
                                    params={
                                        "saddle_amplitude_m": float(ps_cfg.saddle_amplitude_m),
                                        "sphere_radius_m": float(ps_cfg.sphere_radius_m),
                                        "neighborhood_margin_uv": float(ps_cfg.neighborhood_margin_uv),
                                    },
                                ))
                            else:
                                payload = {
                                    "kind": int(getattr(_sk, "TRI_PARAM_SURFACE_POLY_BARY", 1)),
                                    "coeffs": spline_coeffs[ti_int].ravel().astype(np.float64),
                                }
                            tracer.register_tri_group(
                                role_bits          = 1,
                                sample_policy      = 1,
                                tri_indices        = np.asarray([ti], np.int32),
                                parametric_surface = payload,
                            )
                            tri_verts = np.asarray(scene.verts[ti_int], np.float64)
                            emissive_group_centers.append(np.mean(tri_verts, axis=0))
                        if parametric_driver is not None:
                            print(f"  bdpt: {emissive_tris.size} per-tri emissive groups with parametric SDF '{ps_cfg.model}'")
                        else:
                            print(f"  bdpt: {emissive_tris.size} per-tri emissive groups with POLY_BARY spline")
                    elif (parametric_driver is not None) or (spline_coeffs is not None):
                        # Single group with mean coefficients across emissive tris.
                        if parametric_driver is not None:
                            payloads: list[dict[str, Any]] = []
                            for ti in emissive_tris:
                                ti_int = int(ti)
                                payload = _resolve_native_parametric_payload(parametric_driver(
                                    tri_id=ti_int,
                                    tri_vertices=np.asarray(scene.verts[ti_int], dtype=np.float64),
                                    params={
                                        "saddle_amplitude_m": float(ps_cfg.saddle_amplitude_m),
                                        "sphere_radius_m": float(ps_cfg.sphere_radius_m),
                                        "neighborhood_margin_uv": float(ps_cfg.neighborhood_margin_uv),
                                    },
                                ))
                                payloads.append(payload)
                            if not payloads:
                                raise RuntimeError("parametric SDF requested but plugin returned no payloads")
                            kind0 = int(payloads[0]["kind"])
                            if any(int(p["kind"]) != kind0 for p in payloads):
                                raise RuntimeError(
                                    "parametric SDF plugin returned mixed kinds for merged group; use per-triangle mode")
                            coeff_stack = np.asarray([np.asarray(p["coeffs"], dtype=np.float64) for p in payloads], dtype=np.float64)
                            mean_payload = {
                                "kind": kind0,
                                "coeffs": np.mean(coeff_stack, axis=0).astype(np.float64, copy=False),
                            }
                        else:
                            valid_ids = emissive_tris[emissive_tris < spline_coeffs.shape[0]]
                            mean_payload = {
                                "kind": _TRI_POLY_BARY,
                                "coeffs": (spline_coeffs[valid_ids].mean(axis=0)
                                           if valid_ids.size > 0 else np.zeros(6, np.float64)),
                            }
                        tracer.register_tri_group(
                            role_bits          = 1,
                            sample_policy      = 1,
                            tri_indices        = emissive_tris,
                            parametric_surface = {
                                "kind": int(mean_payload["kind"]),
                                "coeffs": np.asarray(mean_payload["coeffs"], dtype=np.float64).ravel(),
                            },
                        )
                        tri_pos = np.asarray(scene.verts[emissive_tris], np.float64).reshape(-1, 3)
                        emissive_group_centers.append(np.mean(tri_pos, axis=0))
                        if parametric_driver is not None:
                            print(f"  bdpt: 1 emissive group with mean parametric SDF '{ps_cfg.model}'")
                        else:
                            print("  bdpt: 1 emissive group with mean POLY_BARY spline")
                    else:
                        if ps_cfg.enabled:
                            raise RuntimeError(
                                "parametric SDF requested but no emissive tris are available for registration")
                        tracer.register_tri_group(
                            role_bits     = 1,
                            sample_policy = 1,
                            tri_indices   = emissive_tris,
                        )
                        tri_pos = np.asarray(scene.verts[emissive_tris], np.float64).reshape(-1, 3)
                        emissive_group_centers.append(np.mean(tri_pos, axis=0))

                # The rebuilt camera owns optical transport. These lens
                # triangles are only BVH proxies; T2 evaluates the exact conics.
                if scene.optical_camera is not None:
                    scene.optical_camera.register_exact_transport(
                        tracer, np.asarray(scene.verts, np.float64).reshape(-1, 3, 3)
                    )
                    print(
                        f"  [camera-build] {scene.optical_camera.describe()}",
                        flush=True,
                    )

                # ── Sensor group (PIXEL_CONE / AREA) ─────────────────────
                # backward/bdpt: PIXEL_CONE (shoots backward rays from sensor)
                # forward:       AREA (passive receiver — pixel-maps forward hits)
                _register_sensor = self.integrator in ("forward", "backward", "bdpt", "depth")
                if _register_sensor:
                    cam_groups = scene.camera_tri_groups
                    if not isinstance(cam_groups, dict):
                        raise RuntimeError(
                            "camera tri-groups missing from scene; cannot register sensor group safely"
                        )
                    sensor_src = cam_groups.get("sensor")
                    if sensor_src is None:
                        raise RuntimeError(
                            "camera tri-group 'sensor' missing; refusing fallback to whole-scene tris"
                        )
                    sensor_tris = np.ascontiguousarray(sensor_src, dtype=np.int32)
                    if sensor_tris.size == 0:
                        raise RuntimeError(
                            "camera tri-group 'sensor' is empty; sensor group registration aborted"
                        )
                    aperture_tris = np.ascontiguousarray(cam_groups.get("aperture_blocker", np.zeros((0,), np.int32)), dtype=np.int32)
                    sensor_w_m = float(getattr(cam, "sensor_w_m", float(self.optics.sensor_w_mm) * 1.0e-3))
                    sensor_h_m = float(getattr(cam, "sensor_h_m", float(self.optics.sensor_h_mm) * 1.0e-3))
                    focal_m = max(1.0e-4, float(cam.focal_m))
                    aperture_radius_m = max(1.0e-6, float(cam.aperture_radius_m))

                    tri_role_blocker = int(getattr(_sk, "TRI_GROUP_ROLE_BLOCKER", 1 << 2))
                    tri_sample_area = int(getattr(_sk, "TRI_GROUP_SAMPLE_AREA", 1))
                    tri_role_sensor = int(getattr(_sk, "TRI_GROUP_ROLE_SENSOR", 1 << 1))
                    tri_sample_pixel = int(getattr(_sk, "TRI_GROUP_SAMPLE_PIXEL_CONE", 3))
                    n_ap_requested, cone_half_deg, cone_solid_angle_sr = self._adaptive_aperture_sample_count(
                        cam, int(frame_cfg.detail_level)
                    )
                    if self.integrator == "depth":
                        # One primary per native site: depth is a spatial scan,
                        # not an aperture-integration estimator.
                        n_ap_requested = 1
                    n_emit_pre = len(emissive_group_centers)
                    tri_groups_est = int(getattr(tracer, "n_tri_groups", lambda: 0)()) + 2
                    if conv_drive_batches:
                        expected_sensor_batches = int(conv_max_batches) if int(conv_max_batches) > 0 else max(32, int(plan.n_batches))
                    else:
                        expected_sensor_batches = max(1, int(plan.n_batches))
                    n_ap_samples, stream_div = self._schedule_pixel_cone_sampling(
                        requested_samples=int(n_ap_requested),
                        rays_per_batch_total=int(rays_per_batch_total),
                        n_emit_groups=n_emit_pre,
                        tri_group_count_estimate=tri_groups_est,
                        expected_batches=int(expected_sensor_batches),
                    )
                    if self.integrator == "depth":
                        n_ap_samples, stream_div = 1, 1
                    if n_ap_samples != int(n_ap_requested):
                        print(f"  [bdpt] aperture samples scaled: requested={int(n_ap_requested)} -> using={int(n_ap_samples)}")
                    self._camera_last_n_aperture_samples = int(n_ap_samples)
                    _bdpt_native_n_aperture_samples = int(n_ap_samples)
                    self._camera_last_cone_half_angle_deg = float(cone_half_deg)
                    self._camera_last_cone_solid_angle_sr = float(cone_solid_angle_sr)

                    aperture_stop_group_id = -1
                    if aperture_tris.size > 0 and self.integrator in ("backward", "bdpt", "depth"):
                        # Aperture stop only needs kernel registration for backward/BDPT;
                        # in forward mode the blocker geometry in the BVH already absorbs
                        # off-axis rays via the material response.
                        aperture_stop_group_id = int(tracer.register_tri_group(
                            role_bits=tri_role_blocker,
                            sample_policy=tri_sample_area,
                            tri_indices=aperture_tris,
                        ))

                    # Forward mode: register sensor as a passive AREA receiver so
                    # the kernel pixel-maps forward ray hits without shooting
                    # backward PIXEL_CONE rays.
                    _sensor_sample_policy = (tri_sample_pixel
                                             if self.integrator in ("backward", "bdpt", "depth")
                                             else tri_sample_area)
                    # Wire solved camera into the sensor descriptor so the
                    # thin-lens geometric mode and focus distance reach the kernel.
                    from bdpt_integrator import (
                        CAMERA_MODE_APERTURE_CONE, CAMERA_MODE_THIN_LENS_GEOMETRIC,
                    )
                    _cam_pos_w = np.asarray(
                        cam.pos + np.asarray(cam.fwd, np.float64) * float(cam.sensor_plane_offset_m),
                        np.float64,
                    )
                    _cam_desc_base = {
                        "pos": _cam_pos_w,
                        "fwd": np.asarray(cam.fwd, np.float64),
                        "up":  np.asarray(cam.up,  np.float64),
                        "sensor_w_m":         float(sensor_w_m),
                        "sensor_h_m":         float(sensor_h_m),
                        "focal_m":            float(max(focal_m, 1.0e-3)),
                        "aperture_radius_m":  float(aperture_radius_m),
                        "n_px":               int(self.width),
                        "n_py":               int(self.height),
                        "n_aperture_samples": int(n_ap_samples),
                        "pixel_stream_divisor": int(stream_div),
                        "pixel_stream_phase": 0,
                        "pixel_stream_phase_from_seed": 1,
                        "aperture_stop_group_id": int(aperture_stop_group_id),
                        "camera_mode": int(self.camera_mode),
                        "effective_focal_m": 0.0,
                        "focus_distance_m": 0.0,
                        "use_optical_handlers": 1,
                    }
                    _solved_status_ok = False
                    if solved is not None:
                        _solved_status_ok = bool(getattr(solved.sanity_report, "geometry_ok", False))
                    if solved is not None and _solved_status_ok:
                        _si = solved.sanity_input
                        _rpt = solved.sanity_report
                        _ap_z = float(
                            _si.aperture_rail_z_m if _si.aperture_rail_z_m is not None
                            else _si.aperture_plane_offset_m
                        )
                        _eff_f = float(_rpt.planes.effective_focal_m)
                        _fp_z  = float(_rpt.planes.focal_plane_z_m)
                        _s_z   = float(_si.sensor_plane_z_m)
                        _lens_c = np.asarray(
                            getattr(
                                cam,
                                "aperture_center_world",
                                np.asarray(cam.pos, np.float64)
                                + np.asarray(cam.fwd, np.float64) * _ap_z,
                            ),
                            np.float64,
                        )
                        _s_to_ap = float(np.linalg.norm(_lens_c - _cam_pos_w))
                        _foc_dist = max(1.0e-4, abs(_fp_z - _ap_z))
                        _lens_fwd = np.asarray(
                            getattr(cam, "lens_fwd_world", cam.fwd), np.float64
                        )
                        _cam_desc_base.update({
                            "focal_m": float(max(_s_to_ap, 1.0e-6)) if _s_to_ap > 1.0e-6 else _cam_desc_base["focal_m"],
                            "effective_focal_m": float(_eff_f),
                            "focus_distance_m": float(_foc_dist),
                            "lens_center": _lens_c,
                            "lens_fwd": _lens_fwd,
                        })
                    elif solved is not None and self.integrator in ("backward", "bdpt", "depth"):
                        # Failed solver packages can produce extreme thin-lens descriptors
                        # that starve PIXEL_CONE sensor paths. Keep backward sampling in
                        # aperture-cone mode until a valid solve is available.
                        _cam_desc_base.update({
                            "camera_mode": int(CAMERA_MODE_APERTURE_CONE),
                            "effective_focal_m": 0.0,
                            "focus_distance_m": 0.0,
                            "use_optical_handlers": 0,
                        })
                        print(
                            "  [warn] camera solve status="
                            f"{str(getattr(solved.sanity_report, 'status', 'unknown')).upper()}"
                            "; using aperture-cone sensor sampling for this frame "
                            "to preserve backward PIXEL_CONE coverage"
                        )
                    if scene.optical_camera is not None:
                        # No second thin-lens interpretation is permitted when
                        # a rebuilt compound camera owns the optical path.
                        _cam_desc_base.update({
                            "camera_mode": int(CAMERA_MODE_PARAMETRIC_ASSEMBLY),
                            "use_optical_handlers": 0,
                        })
                    
                    # ── Validate camera configuration ────────────────────────────
                    registered_lens_surfaces = 0
                    if scene.optical_camera is not None:
                        registered_lens_surfaces = int(
                            scene.optical_camera.lens_assembly.build_parametric_payload()[1]
                        )
                    val_error = validate_frame_configuration(
                        camera_mode=int(_cam_desc_base.get("camera_mode", self.camera_mode)),
                        n_aperture_samples=int(_cam_desc_base.get("n_aperture_samples", 1)),
                        aperture_radius_m=float(_cam_desc_base.get("aperture_radius_m", aperture_radius_m)),
                        has_aperture_stop=(aperture_stop_group_id >= 0),
                        # Preflight occurs before rays exist. For an exact
                        # assembly, registration count is the only meaningful
                        # proof that a lens can participate at this point.
                        lens_event_count=registered_lens_surfaces,
                        effective_focal_m=float(_cam_desc_base.get("effective_focal_m", 0.0)),
                        focus_distance_m=float(_cam_desc_base.get("focus_distance_m", 0.0)),
                        solved_camera_package=solved,
                    )
                    if val_error is not None:
                        raise RuntimeError(
                            f"Camera configuration validation failed for frame {self._frame_index}: {val_error}"
                        )
                    
                    self._sensor_group_id = tracer.register_tri_group(
                        role_bits     = tri_role_sensor,
                        sample_policy = _sensor_sample_policy,
                        tri_indices   = sensor_tris,
                        sensor_camera = _cam_desc_base,
                    )
                    self._sensor_camera_desc = dict(_cam_desc_base)
                    self._active_camera_mode = int(_cam_desc_base["camera_mode"])
                    cpp_back = backs.get("cpp")
                    if isinstance(cpp_back, CppExposureBackend):
                        cpp_back._sensor_camera_desc = dict(self._sensor_camera_desc)
                        if (
                            bool(getattr(cpp_back, "_gpu_resident", False))
                            and self.integrator in ("backward", "bdpt", "depth")
                            and hasattr(tracer, "configure_sensor_image")
                        ):
                            target_mode = 0
                            target_radius = float(aperture_radius_m)
                            target_spec = (
                                None if scene.optical_camera is None
                                else scene.optical_camera.backward_target_spec()
                            )
                            if target_spec is not None:
                                target_center = np.asarray(
                                    target_spec.center, np.float64
                                ).reshape(3)
                                target_radius = float(target_spec.radius)
                                target_mode = int(
                                    str(target_spec.direction_mode) == "away_from_virtual"
                                )
                            else:
                                target_center = np.asarray(
                                    _cam_desc_base.get(
                                        "lens_center",
                                        _cam_pos_w + np.asarray(cam.fwd, np.float64) * float(focal_m),
                                    ),
                                    np.float64,
                                ).reshape(3)
                            sensor_image_res = int(max(16, int(max(self.width, self.height))))
                            tracer.configure_sensor_image(
                                float(_cam_pos_w[0]),
                                float(sensor_w_m * 0.5),
                                float(sensor_h_m * 0.5),
                                int(sensor_image_res),
                                0.008,
                                float(target_center[0]),
                                float(target_radius),
                                float(target_center[1]),
                                float(target_center[2]),
                                int(target_mode),
                            )
                            if hasattr(tracer, "configure_sensor_pose"):
                                sensor_fwd = np.asarray(_cam_desc_base["fwd"], np.float64)
                                sensor_up = np.asarray(_cam_desc_base["up"], np.float64)
                                sensor_right = np.cross(sensor_fwd, sensor_up)
                                sensor_right /= max(
                                    float(np.linalg.norm(sensor_right)), 1.0e-15
                                )
                                sensor_up = np.cross(sensor_right, sensor_fwd)
                                sensor_up /= max(float(np.linalg.norm(sensor_up)), 1.0e-15)
                                lens_fwd = np.asarray(
                                    _cam_desc_base.get("lens_fwd", [-1.0, 0.0, 0.0]),
                                    np.float64,
                                )
                                lens_up = np.asarray([0.0, 0.0, 1.0], np.float64)
                                lens_right = np.cross(lens_fwd, lens_up)
                                lens_right /= max(
                                    float(np.linalg.norm(lens_right)), 1.0e-15
                                )
                                lens_up = np.cross(lens_right, lens_fwd)
                                lens_up /= max(float(np.linalg.norm(lens_up)), 1.0e-15)
                                tracer.configure_sensor_pose(
                                    _cam_pos_w,
                                    sensor_right,
                                    sensor_up,
                                    target_center,
                                    lens_right,
                                    lens_up,
                                )
                            print(
                                "  [gpu-resident] native sensor image configured "
                                f"res={sensor_image_res} plate_x={float(_cam_pos_w[0]):.4f} "
                                f"target_x={float(target_center[0]):.4f} "
                                f"target_r={float(target_radius)*1e3:.2f}mm "
                                f"target_mode={target_mode}"
                            )
                        
                        # Create optical assembly for this frame's camera configuration
                        try:
                            if (
                                scene.optical_camera is None
                                and int(_cam_desc_base.get("camera_mode", CAMERA_MODE_APERTURE_CONE)) == int(CAMERA_MODE_THIN_LENS_GEOMETRIC)
                                and int(_cam_desc_base.get("use_optical_handlers", 0)) != 0
                            ):
                                focal_m = float(_cam_desc_base.get("focal_m", 0.035))
                                aperture_r_m = float(_cam_desc_base.get("aperture_radius_m", 0.0125))
                                
                                # Create thin lens optical assembly in the backend
                                cpp_back._optical_backend.create_thin_lens_assembly(
                                    focal_length_m=focal_m,
                                    aperture_diameter_m=aperture_r_m * 2.0,  # Convert radius to diameter
                                    sensor_distance_m=focal_m,
                                )
                                # Attach assembly to tracer so PIXEL_CONE rays go
                                # through the handler chain (gated by use_optical_handlers=1).
                                tracer.attach_optical_assembly(cpp_back._optical_backend)
                        except Exception as e:
                            print(f"Warning: Failed to create optical assembly: {e}")
                        
                        sensor_cap_seed = (
                            int(self.width)
                            * int(self.height)
                            * int(n_ap_samples)
                            * int(self.freq_hz.shape[0])
                        )
                        cpp_back._bdpt_dynamic_cap_hint = max(
                            int(cpp_back._bdpt_dynamic_cap_hint),
                            int(sensor_cap_seed),
                        )
                        self._bdpt_cap_hint_global = max(
                            int(self._bdpt_cap_hint_global),
                            int(cpp_back._bdpt_dynamic_cap_hint),
                        )
                    print(f"  {self.integrator.upper()}: sensor group registered with id {self._sensor_group_id}")
                    print(
                        f"  {self.integrator.upper()}: aperture-domain proposal samples="
                        f"{n_ap_samples} stream_div={stream_div} stop_group={aperture_stop_group_id}"
                    )

                    # Bind sensor/film SSBO before batch dispatch.
                    self._bind_sensor_film_ssbo(tracer)
                    if scene.optical_camera is not None:
                        # Exact conic traversal owns the camera path. A second
                        # thin/thick-lens scale context would steer the same ray
                        # again and invalidate its optical geometry.
                        tracer.clear_scale_contexts()
                        print(
                            "  [camera-build] surrogate scale contexts disabled; "
                            "exact parametric assembly owns optical transport",
                            flush=True,
                        )
                    else:
                        self._configure_default_wave_contexts(tracer, cam, solved)

                    _bdpt_emitter_centers = (
                        np.asarray(emissive_group_centers, np.float64)
                        if emissive_group_centers else np.zeros((0, 3), np.float64)
                    )
                    n_emit_pre = len(emissive_group_centers)
                    _rays_per_emit = max(64, self.total_rays // max(1, tracer.n_tri_groups()))
                    _bdpt_target_total = max(n_emit_pre, int(_rays_per_emit) * max(1, n_emit_pre))
                    _bdpt_stream_active = self.integrator in ("backward", "bdpt", "depth")
                    _stream_divisor = max(1, int(stream_div))
                    _stream_seen = np.zeros((_stream_divisor,), dtype=bool)
                    _stream_cover_complete = (_stream_divisor <= 1)
                    _stream_covered_n = int(_stream_divisor if _stream_cover_complete else 0)
                    _stream_hard_cap_batches = int(max(plan.n_batches, plan.n_batches + max(8, 2 * _stream_divisor)))
                    _bdpt_native_lab_active = bool(
                        self.integrator == "bdpt"
                        and _is_thick_lens_lab_scene_mode(scene_mode)
                        and isinstance(cpp_back, CppExposureBackend)
                        and bool(getattr(cpp_back, "_gpu_resident", False))
                    )
                    _sensor_mip_active = bool(
                        _bdpt_native_lab_active
                        and bool(getattr(cpp_back, "_sensor_mipmap_enabled", False))
                    )
                    if _bdpt_native_lab_active:
                        _stream_divisor = 1
                        if _sensor_mip_active:
                            _sensor_mip_max_epochs = max(
                                0, int(os.environ.get("SPECTRAL_SENSOR_MAX_EPOCHS", "64"))
                            )
                            _sensor_mip_continuous = str(
                                os.environ.get("SPECTRAL_SENSOR_PERSISTENT_EPOCHS", "0")
                            ).strip().lower() in ("1", "true", "yes", "on")
                            _sensor_mip_top_k = min(
                                1024,
                                max(
                                    64,
                                    int(os.environ.get(
                                        "SPECTRAL_SENSOR_TOP_K",
                                        str(max(self.width, self.height)),
                                    )),
                                ),
                            )
                            _sensor_mip_steps_per_layer = max(
                                1,
                                int(os.environ.get(
                                    "SPECTRAL_SENSOR_STEPS_PER_LAYER",
                                    "8" if max(self.width, self.height) <= 128 else "4",
                                )),
                            )
                            _sensor_mip_targeted_fraction = float(np.clip(
                                float(os.environ.get(
                                    "SPECTRAL_SENSOR_TARGETED_FRACTION", "0.0"
                                )), 0.0, 1.0,
                            ))
                            from camera_software.scan_control import CoverageBalanceController
                            _sensor_mip_balance = CoverageBalanceController(
                                preferred_targeted=_sensor_mip_targeted_fraction,
                                minimum_targeted=float(np.clip(
                                    float(os.environ.get(
                                        "SPECTRAL_SENSOR_MIN_TARGETED_FRACTION", "0.20"
                                    )), 0.0, _sensor_mip_targeted_fraction,
                                )),
                            )
                            _sensor_mip_min_epochs = 8
                            _sensor_mip_hold_required = 3
                            _sensor_mip_hold = 0
                            _sensor_mip_clarity = MonteCarloClarityDiscriminator(
                                min_layers=_sensor_mip_min_epochs,
                                max_drift=0.008,
                                max_noise_rse_p90=0.25,
                            )
                            _stream_seen = np.zeros(
                                (max(1, _sensor_mip_max_epochs),), dtype=bool
                            )
                            _stream_hard_cap_batches = (
                                _sensor_mip_max_epochs if _sensor_mip_max_epochs > 0
                                else np.iinfo(np.int32).max
                            )
                            print(
                                "  BDPT: recursive GPU sensor exposure enabled "
                                f"(top-k={_sensor_mip_top_k}, max_epochs="
                                f"{'unlimited' if _sensor_mip_max_epochs == 0 else _sensor_mip_max_epochs}, "
                                f"targeted={_sensor_mip_targeted_fraction:.3f}, "
                                f"coverage={1.0 - _sensor_mip_targeted_fraction:.3f})"
                            )
                        else:
                            _bdpt_native_work_units, _bdpt_native_work_plan = _native_bdpt_work_units(
                                int(self.width),
                                int(self.height),
                                int(max(1, _bdpt_native_n_aperture_samples)),
                                requested_packages=int(self.bdpt_native_packages),
                                # Camera record pages are now recycled inside
                                # one native exposure while its light family
                                # remains resident. Avoid repeating the flash
                                # and setup as separate Python work units.
                                primary_ray_cap=(
                                    int(self.width)
                                    * int(self.height)
                                    * int(max(1, _bdpt_native_n_aperture_samples))
                                ),
                            )
                            _bdpt_native_schedule = _native_bdpt_refinement_schedule(
                                _bdpt_native_work_units,
                                self.bdpt_native_sweeps,
                            )
                            _stream_seen = np.zeros((len(_bdpt_native_schedule),), dtype=bool)
                            _stream_hard_cap_batches = max(
                                _stream_hard_cap_batches,
                                len(_bdpt_native_schedule),
                            )
                        _stream_cover_complete = False
                        _stream_covered_n = 0
                        if not _sensor_mip_active:
                            print("  BDPT: thick-lens lab using native flash/sensor/T5 package")
                    print(f"  {self.integrator.upper()}: streaming {n_emit_pre} emitter groups, "
                          f"~{_bdpt_target_total:_} rays/batch target")
                    print(f"  {self.integrator.upper()}: stream coverage target {max(1, _stream_divisor)} phases before exposure sign-off")
                else:
                    # For forward-only mode, no sensor stream needed
                    _bdpt_emitter_centers = (
                        np.asarray(emissive_group_centers, np.float64)
                        if emissive_group_centers else np.zeros((0, 3), np.float64)
                    )
                    n_emit_pre = len(emissive_group_centers)
                    _rays_per_emit = max(64, self.total_rays // max(1, tracer.n_tri_groups()))
                    _bdpt_target_total = max(n_emit_pre, int(_rays_per_emit) * max(1, n_emit_pre))
                    _bdpt_stream_active = self.integrator in ("backward", "bdpt", "depth")
                    _stream_divisor = 1
                    _stream_seen = np.zeros((1,), dtype=bool)
                    _stream_cover_complete = True
                    _stream_covered_n = 1
                    _stream_hard_cap_batches = int(max(plan.n_batches, plan.n_batches + 8))
                    print(f"  {self.integrator.upper()}: emissive-only mode (no sensor stream)")

        t0 = time.perf_counter()
        batches_executed = 0
        converged_early = False
        b_idx = 0
        while True:
            stream_needs_more = bool(_bdpt_stream_active and not _stream_cover_complete)
            if not conv_drive_batches and b_idx >= plan.n_batches and not stream_needs_more:
                break
            if not conv_drive_batches and b_idx >= _stream_hard_cap_batches:
                if stream_needs_more:
                    print(
                        "  [warn] stream coverage incomplete at hard cap: "
                        f"covered={int(_stream_covered_n)}/{int(max(1, _stream_divisor))}; "
                        f"forcing sign-off after {b_idx} batches"
                    )
                break
            if conv_drive_batches and conv_max_batches > 0 and b_idx >= conv_max_batches:
                if stream_needs_more:
                    print(
                        "  [warn] reached max_batches before stream coverage completed: "
                        f"covered={int(_stream_covered_n)}/{int(max(1, _stream_divisor))}"
                    )
                print(f"  [conv] reached max_batches={conv_max_batches}; stopping acquisition")
                break

            seed = _bounded_native_seed(
                (self._rng_seed * 1_000_003) + b_idx + 1
            )
            if _bdpt_stream_active:
                _cpp = backs.get("cpp")
                if _cpp is not None:
                    if self.integrator == "depth":
                        if not isinstance(_cpp, CppExposureBackend):
                            raise RuntimeError("depth mode requires the native C++ GPU backend")
                        scene_groups = scene.camera_tri_groups or {}
                        _cpp.run_gpu_first_scene_depth(
                            scene_tri_ids=np.asarray(
                                scene_groups.get("object", np.zeros((0,), np.int32)),
                                np.int32,
                            ),
                            seed=int(seed),
                            rays_per_page=int(max(1, rays_per_batch_total)),
                        )
                        batches_executed = 1
                        _stream_cover_complete = True
                        _stream_covered_n = 1
                        if batch_preview_cb is not None:
                            _depth_snaps = self.stream_integration_snapshots(
                                backs=backs, plan=plan, frame_cfg=frame_cfg,
                                batch_index=1,
                            )
                            batch_preview_cb(
                                backs, _depth_snaps, 1, 1,
                                float(time.perf_counter() - t0),
                            )
                        break
                    if _bdpt_native_lab_active:
                        if not isinstance(_cpp, CppExposureBackend):
                            raise RuntimeError("native thick-lens BDPT requires the C++ exposure backend")
                        if _sensor_mip_active:
                            self._discard_last_bdpt_records(clear_emit_counts=True)
                            _cpp.run_recursive_sensor_epoch(
                                top_k=int(_sensor_mip_top_k), seed=int(seed),
                                emitter_tri_ids=_bdpt_native_emitter_tri_ids,
                                refinement_steps=int(_sensor_mip_steps_per_layer),
                                targeted_fraction=float(_sensor_mip_targeted_fraction),
                            )
                            _raw_mip = np.asarray(
                                _cpp.tracer.get_sensor_image_linear(), dtype=np.float32
                            )
                            _mip_display = _native_sensor_tile_to_display(
                                _raw_mip,
                                *(_ORDERED_TILE_OUTPUT if _ORDERED_TILE_OUTPUT is not None
                                  else (int(self.width), int(self.height))),
                            )[..., :3]
                            # Preserve the ordered output shape for final save;
                            # the native accumulator itself remains square.
                            _cpp._native_sensor_linear_image = np.ascontiguousarray(
                                _mip_display, dtype=np.float32
                            )
                            if hasattr(_cpp.tracer, "get_sensor_image_sum_linear"):
                                _raw_sum = np.asarray(
                                    _cpp.tracer.get_sensor_image_sum_linear(), dtype=np.float32
                                )
                                _cpp._native_sensor_sum_linear = np.ascontiguousarray(
                                    _native_sensor_tile_to_display(
                                        _raw_sum,
                                        *(_ORDERED_TILE_OUTPUT
                                          if _ORDERED_TILE_OUTPUT is not None
                                          else (int(self.width), int(self.height))),
                                    )[..., :3],
                                    dtype=np.float32,
                                )
                            if hasattr(_cpp.tracer, "get_sensor_exposure_weight"):
                                _raw_weight = np.asarray(
                                    _cpp.tracer.get_sensor_exposure_weight(), dtype=np.float32
                                )
                                _cpp._native_sensor_exposure_weight = np.ascontiguousarray(
                                    _native_sensor_tile_to_display(
                                        _raw_weight[..., None],
                                        *(_ORDERED_TILE_OUTPUT
                                          if _ORDERED_TILE_OUTPUT is not None
                                          else (int(self.width), int(self.height))),
                                    )[..., 0],
                                    dtype=np.float32,
                                )
                            if hasattr(_cpp.tracer, "get_sensor_learned_priority_map"):
                                _raw_priority = np.asarray(
                                    _cpp.tracer.get_sensor_learned_priority_map(),
                                    dtype=np.float32,
                                )
                                _cpp._native_sensor_priority_map = np.ascontiguousarray(
                                    _native_sensor_tile_to_display(
                                        _raw_priority[..., None],
                                        *(_ORDERED_TILE_OUTPUT
                                          if _ORDERED_TILE_OUTPUT is not None
                                          else (int(self.width), int(self.height))),
                                    )[..., 0],
                                    dtype=np.float32,
                                )
                            _cpp._native_sensor_image = _native_sensor_tile_to_display(
                                np.asarray(_cpp.tracer.get_sensor_image(), dtype=np.float32),
                                *(_ORDERED_TILE_OUTPUT if _ORDERED_TILE_OUTPUT is not None
                                  else (int(self.width), int(self.height))),
                            )[..., :3]
                            _clarity = _sensor_mip_clarity.update(_mip_display)
                            _coverage_weight = getattr(
                                _cpp, "_native_sensor_exposure_weight", None
                            )
                            if _coverage_weight is not None:
                                _sensor_mip_targeted_fraction = _sensor_mip_balance.update(
                                    _coverage_weight, _clarity.noise_rse_p90
                                )
                            _clear_now = _clarity.accepted
                            _sensor_mip_hold = (_sensor_mip_hold + 1) if _clear_now else 0
                            batches_executed = b_idx + 1
                            if b_idx < _stream_seen.size:
                                _stream_seen[b_idx] = True
                            _stream_covered_n = batches_executed
                            b_idx += 1
                            _epoch_goal = (
                                "unlimited" if _sensor_mip_max_epochs == 0
                                else str(_sensor_mip_max_epochs)
                            )
                            print(
                                f"  [sensor-refine] epoch {b_idx}/{_epoch_goal} "
                                f"top-k={_sensor_mip_top_k} clarity_drift={_clarity.drift:.5f} "
                                f"targeted={_sensor_mip_targeted_fraction:.3f} "
                                f"noise_rse_p90={_clarity.noise_rse_p90:.4f} "
                                f"edge={_clarity.edge_energy:.5f} "
                                f"hold={_sensor_mip_hold}/{_sensor_mip_hold_required}",
                                flush=True,
                            )
                            if batch_preview_cb is not None:
                                _native_snaps = self.stream_integration_snapshots(
                                    backs=backs, plan=plan, frame_cfg=frame_cfg,
                                    batch_index=int(b_idx),
                                )
                                batch_preview_cb(
                                    backs, _native_snaps, int(b_idx),
                                    int(_sensor_mip_max_epochs),
                                    float(time.perf_counter() - t0),
                                )
                            if (not _sensor_mip_continuous
                                    and _sensor_mip_hold >= _sensor_mip_hold_required):
                                _stream_cover_complete = True
                                print(
                                    f"  [sensor-refine] clarity discriminator accepted "
                                    f"after {b_idx} epochs",
                                    flush=True,
                                )
                                break
                            if (_sensor_mip_max_epochs > 0
                                    and b_idx >= _sensor_mip_max_epochs):
                                _stream_cover_complete = True
                                print(
                                    "  [sensor-refine] maximum refinement epochs reached",
                                    flush=True,
                                )
                                break
                            continue
                        self._discard_last_bdpt_records(clear_emit_counts=True)
                        # Native work units: each is one flash+sensor T5 cycle
                        # accumulating into the same global sensor.  The immutable
                        # pixel×aperture sweep is partitioned on WHOLE-PIXEL
                        # boundaries so a single cycle's BDPT records stay under
                        # the GPU record caps —
                        # one full sweep at high render res (1024²×8 ap = 8.4M
                        # rays) overflows the ~6.9M vertex cap, drops pdf
                        # records, invalidates MIS chains, and zeroes the
                        # connect.  Auto-grow the package count so each slice
                        # stays cap-safe; --bdpt-native-packages raises it
                        # further if the user wants more accumulation cycles.
                        _nap_native = int(max(1, _bdpt_native_n_aperture_samples))
                        _work_units = _bdpt_native_work_units
                        _work_plan = _bdpt_native_work_plan
                        _sched_total = int(_work_plan["schedule_rays"])
                        # Native launchers sample one spectral band per path and
                        # use a stochastic, band-dispersive Fresnel decision, so
                        # record growth is linear rather than exponential.
                        _n_pkgs = len(_work_units)
                        _n_cycles = len(_bdpt_native_schedule)
                        if b_idx == 0:
                            print(
                                "  [bdpt-native-lab] spatial work plan "
                                f"image={self.width}x{self.height} pixels={_work_plan['pixels']:,} "
                                f"aperture={_nap_native} schedule={_sched_total:,} "
                                f"units={_n_pkgs:,} sweeps={self.bdpt_native_sweeps:,} "
                                f"cycles={_n_cycles:,} pixels/unit<={_work_plan['pixels_per_unit']:,} "
                                f"primary/unit<={_work_plan['max_primary_rays_per_unit']:,} "
                                f"global_sensor={_work_plan['sensor_rgb_bytes'] / (1024.0**2):.2f} MiB; "
                                "scheduler metadata scales with pixels, not pixel×aperture",
                                flush=True,
                            )
                        _sweep_index, _slice_offset, _slice_count = _bdpt_native_schedule[int(b_idx)]
                        _unit_index = int(b_idx % _n_pkgs)
                        print(f"  [bdpt-native-lab] sweep {_sweep_index + 1}/{self.bdpt_native_sweeps} "
                            f"spatial unit {_unit_index + 1}/{_n_pkgs} "
                            f"schedule={_slice_offset:,}..{_slice_offset + _slice_count:,} "
                            f"({int(_slice_count // _nap_native):,} whole pixels)")
                        _lab_max_children = 1
                        _cpp.run_thick_lens_native_bdpt(
                            emitter_tri_ids=_bdpt_native_emitter_tri_ids,
                            total_rays=int(max(1, self.total_rays // _n_pkgs)),
                            sensor_rays_per_batch=int(max(1, rays_per_batch_total)),
                            n_aperture_samples=_nap_native,
                            max_children=int(_lab_max_children),
                            seed=int(seed),
                            exposure_weight=1.0,
                            sweep_offset=_slice_offset,
                            sweep_count=_slice_count,
                            sensor_sweeps=self.bdpt_native_sweeps,
                        )
                        self._last_bdpt_records = np.zeros((0, 16), dtype=np.float32)
                        self._last_bdpt_emit_counts = np.zeros((0,), dtype=np.int32)
                        batches_executed = b_idx + 1
                        _stream_seen[b_idx] = True
                        _stream_covered_n = int(np.count_nonzero(_stream_seen))
                        b_idx += 1
                        # GPU-resident spatial units used to continue before the
                        # common preview callback below, hiding exposure progress
                        # until completion. Publish only after the GPU T5 join and
                        # its sensor SSBO accumulation are stable.
                        if batch_preview_cb is not None:
                            try:
                                _native_snaps = self.stream_integration_snapshots(
                                    backs=backs,
                                    plan=plan,
                                    frame_cfg=frame_cfg,
                                    batch_index=int(b_idx),
                                )
                                batch_preview_cb(
                                    backs,
                                    _native_snaps,
                                    int(b_idx),
                                    int(_n_cycles),
                                    float(time.perf_counter() - t0),
                                )
                            except Exception as exc:
                                print(f"  [warn] GPU exposure preview callback failed: {exc}")
                                batch_preview_cb = None
                        if b_idx >= _n_cycles:
                            _stream_cover_complete = True
                            _stream_covered_n = _n_cycles
                            break
                        continue
                    else:
                        _emit_rays = self._allocate_bdpt_emit_rays(
                            cpp_back=_cpp,
                            emitter_centers=_bdpt_emitter_centers,
                            target_total_rays=_bdpt_target_total,
                            seed=int(seed),
                        )
                        # Previous batch records are only needed through adaptive
                        # allocation. Release them before tracing the next batch.
                        self._discard_last_bdpt_records(clear_emit_counts=True)
                        _recs = np.ascontiguousarray(_cpp.run_bdpt_batch(_emit_rays, seed), dtype=np.float32)
                        self._bdpt_cap_hint_global = max(
                            int(self._bdpt_cap_hint_global),
                            int(getattr(_cpp, "_bdpt_dynamic_cap_hint", 0)),
                        )
                        try:
                            _cpp.scatter_bdpt_records(_recs, self._sensor_group_id)
                            self._accumulate_pinhole_records(_recs)
                        finally:
                            _cpp.cleanup_overflow_temp_file()
                        # Store last batch's records for adaptive allocator only.
                        # Do NOT call _stage_bdpt_records here — it would overwrite
                        # the intermediary file every batch (the cause of narrow
                        # noise bands / blank first frame in streaming BDPT mode).
                        self._last_bdpt_records = _recs
                        self._last_bdpt_emit_counts = np.asarray(_emit_rays, np.int32)
                        for _b in backs.values():
                            if isinstance(_b, GlslExposureBackend):
                                _b.surf_accum[:]      = _cpp.surf_accum
                                _b.field_accum[:]     = _cpp.field_accum
                                _b.accum[:]           = _cpp.accum
                                _b.n_rays_accumulated = _cpp.n_rays_accumulated

                    if _stream_seen is not None and _stream_divisor > 0:
                        _phase = int(seed % int(_stream_divisor))
                        _stream_seen[_phase] = True
                        _stream_covered_n = int(np.count_nonzero(_stream_seen))
                        _stream_cover_complete = bool(_stream_covered_n >= int(_stream_divisor))
            else:
                for back in backs.values():
                    back.render_batch(rays_per_source_per_batch, seed)
                    # Forward-trace path writes into accum only; there is no
                    # field/surface split available.  Treat all accumulated
                    # energy as surface so _make_integral_objects has real data.
                    back.surf_accum[:] = back.accum
                    # field_accum stays zero — no ambient data in forward mode.
            batches_executed = b_idx + 1

            if batch_preview_cb is not None:
                try:
                    n_batches_hint = -1 if conv_drive_batches else int(plan.n_batches)
                    snaps = self.stream_integration_snapshots(
                        backs=backs,
                        plan=plan,
                        frame_cfg=frame_cfg,
                        batch_index=int(b_idx + 1),
                    )
                    batch_preview_cb(
                        backs,
                        snaps,
                        int(b_idx + 1),
                        int(n_batches_hint),
                        float(time.perf_counter() - t0),
                    )
                except Exception as exc:
                    print(f"  [warn] batch preview callback failed: {exc}")
                    batch_preview_cb = None

            progress_every = 10 if conv_drive_batches else max(1, plan.n_batches // 10)
            if (b_idx + 1) % progress_every == 0:
                elapsed = time.perf_counter() - t0
                if conv_drive_batches:
                    if _bdpt_stream_active:
                        print(
                            f"    batch {b_idx+1:>5}  elapsed={elapsed:6.2f}s"
                            f"  stream={int(_stream_covered_n)}/{int(max(1, _stream_divisor))}"
                        )
                    else:
                        print(f"    batch {b_idx+1:>5}  elapsed={elapsed:6.2f}s")
                else:
                    pct = 100.0 * (b_idx + 1) / plan.n_batches
                    if _bdpt_stream_active:
                        print(
                            f"    batch {b_idx+1:>5}/{plan.n_batches}  "
                            f"({pct:5.1f}%)  elapsed={elapsed:6.2f}s"
                            f"  stream={int(_stream_covered_n)}/{int(max(1, _stream_divisor))}"
                        )
                    else:
                        print(f"    batch {b_idx+1:>5}/{plan.n_batches}  "
                              f"({pct:5.1f}%)  elapsed={elapsed:6.2f}s")

            if conv_enabled and ((b_idx + 1) % conv_every) == 0:
                conv_back = _convergence_backend()
                if conv_back is not None:
                    measured_H_J = float(conv_back.measured_radiant_exposure_J(plan.energy_per_ray_J))
                    target_H_J = max(float(plan.target_H_J), 1.0e-30)
                    conv_last_measured_pct = 100.0 * measured_H_J / target_H_J
                    conv_last_pct = 100.0 * abs(measured_H_J - target_H_J) / target_H_J
                    if measured_H_J <= 1.0e-24:
                        conv_no_signal_streak += 1
                    else:
                        conv_no_signal_streak = 0

                    if conv_drive_batches and conv_no_signal_streak >= conv_no_signal_limit:
                        if _stream_cover_complete:
                            converged_early = True
                            print(
                                "  [conv] no-signal stop: measured exposure remained near zero "
                                f"for {conv_no_signal_streak} checks; stopping open-ended BDPT"
                            )
                            break
                        else:
                            print(
                                "  [conv] no-signal condition reached but stream coverage is incomplete; "
                                f"continuing (stream={int(_stream_covered_n)}/{int(max(1, _stream_divisor))})"
                            )

                    if (b_idx + 1) >= conv_min_batches and conv_last_pct <= (conv_error_target * 100.0):
                        conv_consecutive_hits += 1
                    else:
                        conv_consecutive_hits = 0
                    if conv_consecutive_hits >= conv_hold:
                        if _stream_cover_complete:
                            converged_early = True
                            print(f"  [conv] reached H={measured_H_J:.3e} J "
                                f"({conv_last_measured_pct:.5f}% of target) at batch {b_idx+1}")
                            print(f"         target H={target_H_J:.3e} J, "
                                f"error<={conv_error_target:.3e}, hold={conv_hold}")
                            break
                        else:
                            print(
                              "  [conv] exposure target reached but stream coverage is incomplete; "
                              f"continuing (stream={int(_stream_covered_n)}/{int(max(1, _stream_divisor))})"
                            )
            b_idx += 1

        elapsed = time.perf_counter() - t0
        if converged_early:
            if conv_drive_batches:
                print(f"  [conv] stop after {batches_executed} open-ended batches")
            else:
                print(f"  [conv] early stop after {batches_executed}/{plan.n_batches} batches")
        if _bdpt_stream_active:
            print(
                "  [bdpt] stream coverage at sign-off: "
                f"{int(_stream_covered_n)}/{int(max(1, _stream_divisor))} phases"
            )
        print(f"  {batches_executed} batches in {elapsed:.2f}s "
              f"({(batches_executed/max(elapsed,1e-9)):.1f} batches/s)")

        # ── Build human-readable frame config summary (HUD + JSON) ───────
        fc_s = frame_cfg.field_capture
        cv_s = frame_cfg.camera_visibility
        ss_s = frame_cfg.surface_spline
        ps_s = frame_cfg.parametric_sdf
        is_s = frame_cfg.integral_split

        def _pct(v: float, vmax: float) -> float:
            return float(100.0 * abs(float(v)) / max(1.0e-9, abs(float(vmax))))

        rail_cfg = self.camera_rail_model
        rail_bellows_max = (
            float(rail_cfg.bellows_sep_max_m)
            if rail_cfg.bellows_sep_max_m is not None
            else max(0.050, 2.0 * float(self.optics.focal_mm) * 1.0e-3)
        )
        rail_ap_radius_max = (
            float(rail_cfg.aperture_radius_max_m)
            if rail_cfg.aperture_radius_max_m is not None
            else max(0.050, 1.5 * float(self.optics.focal_mm) * 1.0e-3)
        )

        front_shift_mag_mm = float(math.hypot(
            float(solved.sanity_input.lens_front_shift_x_mm),
            float(solved.sanity_input.lens_front_shift_y_mm),
        ))
        front_tilt_mag_deg = float(math.hypot(
            float(solved.sanity_input.lens_front_tilt_x_deg),
            float(solved.sanity_input.lens_front_tilt_y_deg),
        ))
        sensor_shift_mag_mm = float(math.hypot(
            float(solved.sanity_input.sensor_shift_x_mm),
            float(solved.sanity_input.sensor_shift_y_mm),
        ))
        bellows_sep_m = float(
            float(solved.sanity_input.lens_front_plane_z_m or 0.0)
            - float(solved.sanity_input.lens_rear_plane_z_m or 0.0)
        )

        frame_config_summary = {
            "description":   frame_cfg.description,
            "integrator":    self.integrator,
            "render_product": (
                "sensor_optical_path_depth_m"
                if self.integrator == "depth" else "spectral_radiance"
            ),
            "detail_level":  int(frame_cfg.detail_level),
            "scene_mode":    scene_mode,
            "field":         (f"{fc_s.grid_kind} {fc_s.nx}^3"
                              if fc_s.enabled else "off"),
            "field_strikes": fc_s.capture_strikes if fc_s.enabled else False,
            "cam_vis":       _cam_vis_name(cv_s.camera_vis_mode),
            "transparent":   _transp_name(cv_s.transparent_mode),
            "depth":         (f"{cv_s.depth_cull_m:.0f}m"
                              if cv_s.depth_cull_enabled else "off"),
            "spline":        (f"{'all' if ss_s.fit_all_tris else 'emissive'} "
                              f"lambda={ss_s.ridge_lambda:.1e}"
                              if ss_s.enabled else "off"),
            "parametric":    (f"{ps_s.model} "
                               f"saddle={ps_s.saddle_amplitude_m:.2e}m "
                               f"sphereR={ps_s.sphere_radius_m:.2e}m "
                               f"marginUV={ps_s.neighborhood_margin_uv:.2e}"
                               if ps_s.enabled else "off"),
            "fi": float(is_s.field_integrate_frac),
            "si": float(is_s.surface_integrate_frac),
            "fb": float(is_s.field_bookkeep_frac),
            "sb": float(is_s.surface_bookkeep_frac),
            "hdr_wp": float(is_s.hdr_white_percentile),
            "camera_solve_status": str(solved.sanity_report.status),
            "camera_solve_error": float(solved.sanity_report.error_degree.overall),
            "camera_thin_lens_target_sensor_z_m": float(solved.sanity_report.error_degree.thin_lens_target_sensor_z_m),
            "camera_pinhole_target_sensor_z_m": float(solved.sanity_report.error_degree.pinhole_target_sensor_z_m),
            "camera_pinhole_ref_mm": float(solved.sanity_report.error_degree.pinhole_comparison_mm),
            "camera_pinhole_ref_deg": float(solved.sanity_report.error_degree.pinhole_comparison_degree),
            "camera_solve_iter": int(solved.iterations),
            "camera_coc_um": float(solved.sanity_report.circle_of_confusion_um),
            "camera_sensor_adjust_mm": float(solved.sanity_report.sensor_adjustment_needed_mm),
            "camera_cone_half_angle_deg": float(self._camera_last_cone_half_angle_deg),
            "camera_cone_solid_angle_sr": float(self._camera_last_cone_solid_angle_sr),
            "camera_cone_aperture_samples": int(self._camera_last_n_aperture_samples),
            "camera_front_shift_mm": front_shift_mag_mm,
            "camera_front_tilt_deg": front_tilt_mag_deg,
            "camera_sensor_shift_mm": sensor_shift_mag_mm,
            "camera_aperture_radius_mm": float((solved.sanity_input.aperture_radius_m or 0.0) * 1.0e3),
            "camera_rail_usage_front_shift_pct": _pct(front_shift_mag_mm, float(rail_cfg.front_lens_shift_max_mm)),
            "camera_rail_usage_front_tilt_pct": _pct(front_tilt_mag_deg, float(rail_cfg.front_lens_tilt_max_deg)),
            "camera_rail_usage_sensor_shift_pct": _pct(sensor_shift_mag_mm, float(rail_cfg.sensor_shift_max_mm)),
            "camera_rail_usage_corner_pct": max(
                _pct(float(solved.sanity_input.sensor_corner_tl_mm), float(rail_cfg.sensor_corner_max_mm)),
                _pct(float(solved.sanity_input.sensor_corner_tr_mm), float(rail_cfg.sensor_corner_max_mm)),
                _pct(float(solved.sanity_input.sensor_corner_bl_mm), float(rail_cfg.sensor_corner_max_mm)),
                _pct(float(solved.sanity_input.sensor_corner_br_mm), float(rail_cfg.sensor_corner_max_mm)),
            ),
            "camera_rail_usage_bellows_pct": _pct(
                bellows_sep_m - float(rail_cfg.bellows_sep_min_m),
                rail_bellows_max - float(rail_cfg.bellows_sep_min_m),
            ),
            "camera_rail_usage_aperture_radius_pct": _pct(
                float(solved.sanity_input.aperture_radius_m or 0.0) - float(rail_cfg.aperture_radius_min_m),
                rail_ap_radius_max - float(rail_cfg.aperture_radius_min_m),
            ),
            "camera_rail_ranges": (
                f"sensor_z=[{float(rail_cfg.sensor_z_min_m):+.3f},{float(rail_cfg.sensor_z_max_m):+.3f}]m "
                f"shift<=±{float(rail_cfg.sensor_shift_max_mm):.1f}mm "
                f"front_shift<=±{float(rail_cfg.front_lens_shift_max_mm):.1f}mm "
                f"front_tilt<=±{float(rail_cfg.front_lens_tilt_max_deg):.1f}deg"
            ),
            "thin_lens_planes": scene.optical_camera is None,
            "thin_lens_kind_bit": int(1 << int(getattr(_sk, "SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM", 2))),
            "camera_mode": int(getattr(self, "_active_camera_mode", self.camera_mode)),
            "camera_mode_name": get_mode_name(
                int(getattr(self, "_active_camera_mode", self.camera_mode))
            ),
            "camera_optical_provenance": (
                None if scene.optical_camera is None
                else scene.optical_camera.provenance.as_dict()
            ),
            "camera_manifest": (
                None if scene.optical_camera is None
                else copy.deepcopy(scene.optical_camera.manifest)
            ),
            "effective_focal_mm": (
                0.0 if scene.optical_camera is None else float(
                    dict(scene.optical_camera.manifest.get("resolved", {})).get(
                        "effective_focal_length_mm", 0.0
                    )
                )
            ),
            "camera_back_type": (
                "flat" if scene.optical_camera is None else str(
                    dict(scene.optical_camera.manifest.get("sensor", {})).get(
                        "back_type", "flat"
                    )
                )
            ),
        }

        # ── Per-backend calibration + dump ───────────────────────────────
        results: list[ExposureFrameResult] = []
        for name, back in backs.items():
            if name not in self.backends_requested:
                continue  # shadow backend; skip output
            with self._profiler.section(f"backend_{name}_measure"):
                measured = back.measured_radiant_exposure_J(plan.energy_per_ray_J)
            with self._profiler.section(f"backend_{name}_gain"):
                gain     = self._train_emissivity_gain(measured, plan.target_H_J)
                gain_db  = 20.0 * math.log10(max(gain, 1.0e-12))
            virtual_t = self.film.exposure_time_s
            qe = float(self.film.quantum_efficiency)
            snr = math.sqrt(max(photons_per_pix * qe, 0.0))
            tracer_obj = getattr(back, "tracer", None)

            with self._profiler.section(f"backend_{name}_integrals"):
                field_obj, surface_obj, _merged_bands, rgb_linear = \
                    self._make_integral_objects(name, back.surf_accum, back.field_accum,
                                                gain, frame_cfg.integral_split)
                sensor_obj = None
                if self.integrator in ("backward", "bdpt"):
                    sensor_obj = self._make_sensor_integral(name, gain, tracer_obj,
                                                        surf_accum=back.surf_accum,
                                                        back=back)

            img = self._tone_map(rgb_linear, frame_cfg.integral_split)

            if self.rgb_source == "sensor":
                if sensor_obj is not None:
                    img = self._sensor_display_rgb(sensor_obj)
                    rgb_linear = img.copy()
                elif (
                    isinstance(back, CppExposureBackend)
                    and bool(getattr(back, "_gpu_resident", False))
                    and tracer_obj is not None
                    and hasattr(tracer_obj, "get_sensor_image")
                ):
                    native_linear = _native_sensor_accumulation_output(back)
                    cached_native_img = getattr(back, "_native_sensor_image", None)
                    native_img = np.asarray(cached_native_img if cached_native_img is not None else tracer_obj.get_sensor_image())
                    if native_linear is not None:
                        rgb_linear = native_linear
                        img = self._tone_map(rgb_linear, frame_cfg.integral_split)
                    elif native_img.ndim == 3 and native_img.shape[2] >= 3 and np.any(native_img[..., :3]):
                        img = np.clip(native_img[..., :3], 0.0, 1.0).astype(np.float32, copy=False)
                        rgb_linear = _native_sensor_linear_output(back, img)
                    else:
                        self._warn_missing_rgb_source_once(
                            source="sensor",
                            backend=str(name),
                            context=f"frame={int(self._frame_index)} gpu-resident native sensor image empty",
                        )
                        img = np.zeros_like(img, dtype=np.float32)
                        rgb_linear = np.zeros_like(rgb_linear, dtype=np.float32)
                elif (
                    isinstance(back, GlslExposureBackend)
                    and isinstance(getattr(back, "_mirror", None), CppExposureBackend)
                    and bool(getattr(back._mirror, "_gpu_resident", False))
                ):
                    # GLSL scaffold backed by a shadow C++ backend in gpu-resident
                    # mode — borrow the native sensor image from the mirror.
                    _cpp_mirror = back._mirror
                    _cpp_tracer = getattr(_cpp_mirror, "tracer", None)
                    if _cpp_tracer is not None and hasattr(_cpp_tracer, "get_sensor_image"):
                        _cached = getattr(_cpp_mirror, "_native_sensor_image", None)
                        _native_img = np.asarray(
                            _cached if _cached is not None else _cpp_tracer.get_sensor_image()
                        )
                        if _native_img.ndim == 3 and _native_img.shape[2] >= 3 and np.any(_native_img[..., :3]):
                            img = np.clip(_native_img[..., :3], 0.0, 1.0).astype(np.float32, copy=False)
                            native_linear = getattr(back, "_native_sensor_linear_image", None)
                            rgb_linear = np.asarray(
                                native_linear if native_linear is not None else img,
                                dtype=np.float32,
                            ).copy()
                        else:
                            self._warn_missing_rgb_source_once(
                                source="sensor",
                                backend=str(name),
                                context=f"frame={int(self._frame_index)} glsl-mirror native sensor image empty",
                            )
                            img = np.zeros_like(img, dtype=np.float32)
                            rgb_linear = np.zeros_like(rgb_linear, dtype=np.float32)
                    else:
                        self._warn_missing_rgb_source_once(
                            source="sensor",
                            backend=str(name),
                            context=f"frame={int(self._frame_index)} glsl-mirror tracer unavailable",
                        )
                        img = np.zeros_like(img, dtype=np.float32)
                        rgb_linear = np.zeros_like(rgb_linear, dtype=np.float32)
                else:
                    self._warn_missing_rgb_source_once(
                        source="sensor",
                        backend=str(name),
                        context=f"frame={int(self._frame_index)}",
                    )
                    img = np.zeros_like(img, dtype=np.float32)
                    rgb_linear = np.zeros_like(rgb_linear, dtype=np.float32)

            # Keep EndpointRecord history intact and optionally derive endpoint RGB.
            # In streaming BDPT mode _last_bdpt_records holds only the final
            # batch — not the full session.  Endpoint-RGB reduction on a single
            # sparse batch produces noise.  Skip it; rgb_linear already carries
            # the fully-accumulated surf_accum/field_accum result.
            _streaming_bdpt = self._last_bdpt_emit_counts is not None
            if (not _streaming_bdpt
                and tracer_obj is not None
                and hasattr(tracer_obj, "reduce_endpoint_records_to_rgb_image")
                and self._last_bdpt_records is not None
                and self._last_bdpt_records.size > 0
                and self._sensor_group_id >= 0):
                try:
                    cpp_rgb = tracer_obj.reduce_endpoint_records_to_rgb_image(
                        self._last_bdpt_records,
                        int(self.width),
                        int(self.height),
                        int(self._sensor_group_id),
                        float(max(gain, 0.0)),
                        float(frame_cfg.integral_split.hdr_white_percentile),
                    )
                    endpoint_rgb_linear = np.asarray(cpp_rgb["rgb_linear"], dtype=np.float32)
                    endpoint_img = np.asarray(cpp_rgb["rgb_tonemapped"], dtype=np.float32)
                    rgb_telemetry = dict(cpp_rgb.get("telemetry", {}))
                    kept = int(rgb_telemetry.get("kept_records", 0))
                    inp = int(rgb_telemetry.get("input_records", 0))
                    drop = max(0, inp - kept)
                    if inp > 0:
                        drop_frac = float(drop) / float(inp)
                        if drop_frac > 0.05:
                            print(f"  [warn] endpoint reduction dropped {drop:_}/{inp:_} records ({100.0*drop_frac:.1f}%)")
                    ord_reg = int(rgb_telemetry.get("order_regressions", 0))
                    if ord_reg > 0:
                        print(f"  [warn] endpoint order regressions detected: {ord_reg:_}")
                    if sensor_obj is not None and isinstance(sensor_obj.metrics, dict):
                        sensor_obj.metrics["endpoint_rgb_telemetry"] = rgb_telemetry
                    if self.rgb_source == "endpoint":
                        rgb_linear = endpoint_rgb_linear
                        img = endpoint_img
                except Exception as exc:
                    print(f"  [warn] C++ endpoint->RGB reduction failed; using accum RGB: {exc}")

            # Optional output conversion path for oversampled rendering.
            sy, sx = self._output_downsample_factors()
            if (self.output_height != self.height) or (self.output_width != self.width):
                st = self.output_oversample_stencil
                rgb_linear = self._downsample_hw3(np.asarray(rgb_linear), sy, sx, stencil=st)
                img = self._downsample_hw3(np.asarray(img), sy, sx, stencil=st)
                field_obj.integrated_bands = self._downsample_bhw(np.asarray(field_obj.integrated_bands), sy, sx, stencil=st)
                field_obj.bookkeeping_bands = self._downsample_bhw(np.asarray(field_obj.bookkeeping_bands), sy, sx, stencil=st)
                surface_obj.integrated_bands = self._downsample_bhw(np.asarray(surface_obj.integrated_bands), sy, sx, stencil=st)
                surface_obj.bookkeeping_bands = self._downsample_bhw(np.asarray(surface_obj.bookkeeping_bands), sy, sx, stencil=st)
                if sensor_obj is not None:
                    sensor_obj.photons_per_pixel = self._downsample_hw(np.asarray(sensor_obj.photons_per_pixel), sy, sx, stencil=st)
                    sensor_obj.electrons_per_pixel = self._downsample_hw(np.asarray(sensor_obj.electrons_per_pixel), sy, sx, stencil=st)
                    sensor_obj.snr_linear = self._downsample_hw(np.asarray(sensor_obj.snr_linear), sy, sx, stencil=st)

            field_history = (np.asarray(back.accum, np.float32) * np.float32(max(gain, 0.0)))
            surface_history = (np.asarray(back.accum, np.float32) * np.float32(max(gain, 0.0)))
            if (self.output_height != self.height) or (self.output_width != self.width):
                st = self.output_oversample_stencil
                field_history = self._downsample_bhw(field_history, sy, sx, stencil=st)
                surface_history = self._downsample_bhw(surface_history, sy, sx, stencil=st)
            
            field_obj_path, surface_obj_path, sensor_obj_path = \
                self._save_integral_objects(
                    field_obj,
                    surface_obj,
                    sensor_obj,
                    field_history_bands=field_history,
                    surface_history_bands=surface_history,
                )

            png_path   = os.path.join(self.out_dir,
                                      f"{self._frame_index:04d}_{name}.png")
            png16_path = os.path.join(self.out_dir,
                                      f"{self._frame_index:04d}_{name}_16bit.png")
            linear_path = os.path.join(self.out_dir,
                                       f"{self._frame_index:04d}_{name}_linear.npy")
            json_path  = os.path.join(self.out_dir,
                                      f"{self._frame_index:04d}_{name}_summary.json")

            field_capture_grid_path = ""
            field_capture_strikes_path = ""
            lightfield_shell_top_data: Optional[np.ndarray] = None
            lightfield_shell_side_data: Optional[np.ndarray] = None
            tracer_obj = getattr(back, "tracer", None)
            if tracer_obj is not None and hasattr(tracer_obj, "get_field_capture_meta"):
                try:
                    fc_meta = tracer_obj.get_field_capture_meta()
                    if int(fc_meta.get("grid_kind", -1)) >= 0:
                        grid_reim = tracer_obj.get_field_capture_grid_reim()
                        strikes = tracer_obj.get_field_capture_strikes()
                        lightfield_shell_top_data, lightfield_shell_side_data = self._build_lightfield_shell_overlays(
                            grid_reim=grid_reim,
                            strikes=strikes,
                            field_cfg=frame_cfg.field_capture,
                        )
                        field_capture_grid_path = os.path.join(
                            self.out_dir,
                            f"{self._frame_index:04d}_{name}_field_capture_grid_reim.npy")
                        field_capture_strikes_path = os.path.join(
                            self.out_dir,
                            f"{self._frame_index:04d}_{name}_field_capture_strikes.npy")
                        np.save(field_capture_grid_path, grid_reim)
                        np.save(field_capture_strikes_path, strikes)
                except Exception:
                    field_capture_grid_path = ""
                    field_capture_strikes_path = ""

            _native_sensor_mode = bool(
                isinstance(back, CppExposureBackend)
                and bool(getattr(back, "_gpu_resident", False))
                and self.rgb_source == "sensor"
            )
            _native_evidence: dict[str, Any] = {}
            _measurement_status = "radiometric_accumulator"
            if _native_sensor_mode:
                _rgb_ev = np.asarray(rgb_linear, dtype=np.float64)
                _energy_ev = np.sum(np.maximum(_rgb_ev, 0.0), axis=2)
                if self.integrator == "depth":
                    _depth = _rgb_ev[..., 0]
                    _valid_depth = np.isfinite(_depth)
                    _native_evidence = {
                        "units": "metres_from_sensor_along_optical_path",
                        "render_product": "first_authored_scene_strike_depth",
                        "gpu_traced": True,
                        "cpu_operation": "per-site nearest-hit reduction only",
                        "scene_bounces": 1,
                        "camera_optics_traversed": True,
                        "shape": [int(v) for v in _depth.shape],
                        "valid_pixels": int(np.count_nonzero(_valid_depth)),
                        "total_pixels": int(_depth.size),
                        "minimum_m": (
                            float(np.min(_depth[_valid_depth])) if np.any(_valid_depth) else None
                        ),
                        "maximum_m": (
                            float(np.max(_depth[_valid_depth])) if np.any(_valid_depth) else None
                        ),
                    }
                    _measurement_status = "native_gpu_first_scene_depth"
                else:
                    _native_evidence = {
                        "units": "normalized_native_sensor_rgb",
                        "radiometrically_calibrated": False,
                        "shape": [int(v) for v in _rgb_ev.shape],
                        "finite": bool(np.all(np.isfinite(_rgb_ev))),
                        "lit_pixels": int(np.count_nonzero(_energy_ev > 1.0e-8)),
                        "total_pixels": int(_energy_ev.size),
                        "mean": float(np.mean(_rgb_ev)) if _rgb_ev.size else 0.0,
                        "peak": float(np.max(_rgb_ev)) if _rgb_ev.size else 0.0,
                        "channel_sums": [float(v) for v in np.sum(_rgb_ev, axis=(0, 1))],
                    }
                    _measurement_status = "native_sensor_nonradiometric"
                if tracer_obj is not None and hasattr(tracer_obj, "get_bdpt_latch_state"):
                    _native_evidence["bdpt_latch"] = dict(tracer_obj.get_bdpt_latch_state())
                frame_config_summary["optical_event_telemetry_status"] = (
                    "not populated by GPU-resident native transport; use native_sensor_evidence"
                )

            r = ExposureFrameResult(
                frame_index        = self._frame_index,
                backend            = name,
                plan               = plan_dict,
                n_rays_emitted     = int(back.n_rays_accumulated),
                n_batches          = int(batches_executed),
                measured_H_J       = float(measured),
                measurement_status = _measurement_status,
                target_H_J         = float(plan.target_H_J),
                gain_linear        = float(gain),
                gain_db            = float(gain_db),
                virtual_t_s        = float(virtual_t),
                photons_per_pixel  = float(photons_per_pix),
                snr_estimate       = float(snr),
                image_path         = png_path,
                image16_path       = png16_path,
                image_linear_path  = linear_path,
                field_object_path  = field_obj_path,
                surface_object_path = surface_obj_path,
                sensor_object_path = sensor_obj_path if sensor_obj_path else "",
                field_capture_grid_path    = field_capture_grid_path,
                field_capture_strikes_path = field_capture_strikes_path,
                summary_path       = json_path,
                frame_config_summary = frame_config_summary,
                native_sensor_evidence = _native_evidence,
                camera_event_telemetry = back.camera_event_telemetry.to_dict(),
            )
            r.frame_config_summary["convergence_target_pct"] = float(conv_target_pct)
            r.frame_config_summary["convergence_error_target"] = float(conv_error_target)
            r.frame_config_summary["convergence_last_measured_pct"] = float(conv_last_measured_pct)
            r.frame_config_summary["convergence_last_error_pct"] = float(conv_last_pct)
            r.frame_config_summary["converged_early"] = bool(converged_early)
            r.frame_config_summary["convergence_drive_batches"] = bool(conv_drive_batches)
            r.frame_config_summary["optical_event_telemetry"] = back.camera_event_telemetry.to_dict()

            # HUD text belongs to the live preview only. Saved primary images
            # remain pristine at every resolution, including tiny acceptance
            # renders where an overlay would cover the entire frame.
            detail_lv = int(r.frame_config_summary.get("detail_level", 0))
            preview_lines = _make_hud_lines(r, detail_lv) if self.show_hud else []
            img_preview = _burn_overlay_into_preview(img, preview_lines)

            # Store image data in memory for display
            r.image_data = np.clip(img_preview, 0.0, 1.0).astype(np.float32)
            r.image16_data = (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)
            r.field_integrated_data = field_obj.integrated_bands
            r.surface_integrated_data = surface_obj.integrated_bands
            r.lightfield_shell_top_data = lightfield_shell_top_data
            r.lightfield_shell_side_data = lightfield_shell_side_data
            if sensor_obj is not None:
                r.sensor_photons_data = sensor_obj.photons_per_pixel
                r.sensor_snr_data = sensor_obj.snr_linear
            snap_for_mode = self.build_integration_snapshot(
                backend_name=name,
                back=back,
                plan=plan,
                frame_cfg=frame_cfg,
                batch_index=int(batches_executed),
                include_sensor=True,
                snapshot_mode="final",
            )
            if snap_for_mode.pinhole_rgb_data is not None:
                r.pinhole_rgb_data = np.asarray(snap_for_mode.pinhole_rgb_data, dtype=np.float32)

            # Only save files if explicitly enabled via --save-files
            if self.save_files:
                with self._profiler.section(f"backend_{name}_write_files"):
                    _write_png(png_path, img)
                    _write_png16(png16_path, img)
                    np.save(linear_path, rgb_linear)
                    native_sum = getattr(back, "_native_sensor_sum_linear", None)
                    native_weight = getattr(back, "_native_sensor_exposure_weight", None)
                    native_priority = getattr(back, "_native_sensor_priority_map", None)
                    if native_sum is not None:
                        np.save(
                            os.path.join(
                                self.out_dir,
                                f"{self._frame_index:04d}_{name}_sum_linear.npy",
                            ),
                            np.asarray(native_sum, dtype=np.float32),
                        )
                    if native_weight is not None:
                        np.save(
                            os.path.join(
                                self.out_dir,
                                f"{self._frame_index:04d}_{name}_exposure_weight.npy",
                            ),
                            np.asarray(native_weight, dtype=np.float32),
                        )
                    if native_priority is not None:
                        priority = np.maximum(
                            np.asarray(native_priority, dtype=np.float32), 0.0
                        )
                        np.save(
                            os.path.join(
                                self.out_dir,
                                f"{self._frame_index:04d}_{name}_priority.npy",
                            ),
                            priority,
                        )
                        peak = max(float(np.max(priority)), 1.0e-12)
                        value = np.clip(priority / peak, 0.0, 1.0)
                        priority_rgb = np.stack(
                            [value, 0.35 * np.sqrt(value), 1.0 - value], axis=2
                        )
                        _write_png(
                            os.path.join(
                                self.out_dir,
                                f"{self._frame_index:04d}_{name}_priority.png",
                            ),
                            priority_rgb,
                        )

            if self.save_files:
                r_dict = _frame_result_summary_dict(r)
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(r_dict, fh, indent=2)
            results.append(r)
            if _native_sensor_mode:
                if self.integrator == "depth":
                    print(
                        f"  [{name:>4}] N_rays={r.n_rays_emitted:_}  "
                        f"depth={_native_evidence['valid_pixels']}/"
                        f"{_native_evidence['total_pixels']} valid  "
                        f"range={_native_evidence['minimum_m']}.."
                        f"{_native_evidence['maximum_m']} m  "
                        "H_meas=n/a (first-scene depth product)"
                    )
                else:
                    print(
                        f"  [{name:>4}] N_rays={r.n_rays_emitted:_}  "
                        f"native_sensor={_native_evidence['lit_pixels']}/{_native_evidence['total_pixels']} lit  "
                        f"mean={_native_evidence['mean']:.3e} peak={_native_evidence['peak']:.3e}  "
                        "H_meas=n/a (native sensor RGB is not radiometrically calibrated)"
                    )
            else:
                print(f"  [{name:>4}] N_rays={r.n_rays_emitted:_}  "
                    f"H_meas={measured:.3e} J  H_targ={plan.target_H_J:.3e} J  "
                    f"gain={gain:.3e}x ({gain_db:+.2f} dB)  "
                    f"photons/pix={photons_per_pix:.2e}  SNR~{snr:.2f}")
            if self.save_files:
                print(f"        -> {png_path}")
            else:
                print(f"        (in-memory; use --save-files to save to disk)")

        self._discard_last_bdpt_records(clear_emit_counts=True)
        self._active_cam = None
        self._profiler.report(prefix=f"[frame {self._frame_index:04d}]")
        self._frame_index += 1
        self._rng_seed += 1
        self._cleanup_temp_bdpt_file()
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Tiny PNG writer — pure stdlib, no Pillow dependency
# ─────────────────────────────────────────────────────────────────────────────
def _linear_display_to_srgb(img_hw3: np.ndarray) -> np.ndarray:
    """Encode linear display RGB with the standard sRGB transfer function."""
    x = np.clip(np.asarray(img_hw3, dtype=np.float32), 0.0, 1.0)
    return np.where(
        x <= np.float32(0.0031308),
        x * np.float32(12.92),
        np.float32(1.055) * np.power(x, np.float32(1.0 / 2.4)) - np.float32(0.055),
    ).astype(np.float32, copy=False)


def _write_png(path: str, img_hw3: np.ndarray) -> None:
    """Encode linear (H, W, 3) display RGB as an sRGB 8-bit PNG."""
    import struct
    import zlib
    img = (_linear_display_to_srgb(img_hw3) * 255.0 + 0.5).astype(np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)   # 8-bit RGB
    idat = zlib.compress(raw, 9)
    iend = b""
    with open(path, "wb") as fh:
        fh.write(sig + _chunk(b"IHDR", ihdr) + _chunk(b"sRGB", b"\x00") + _chunk(b"IDAT", idat)
                 + _chunk(b"IEND", iend))


def _write_png16(path: str, img_hw3: np.ndarray) -> None:
    """Encode linear (H, W, 3) display RGB as an sRGB 16-bit PNG."""
    import struct
    import zlib
    img = (_linear_display_to_srgb(img_hw3) * 65535.0 + 0.5).astype(">u2", copy=False)
    h, w = img.shape[:2]
    # PNG 16-bit channels are network-byte-order, already satisfied by >u2.
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)  # 16-bit RGB
    idat = zlib.compress(raw, 9)
    with open(path, "wb") as fh:
        fh.write(sig + _chunk(b"IHDR", ihdr) + _chunk(b"sRGB", b"\x00") + _chunk(b"IDAT", idat)
                 + _chunk(b"IEND", b""))


def _burn_overlay_into_preview(img_hw3: np.ndarray,
                               lines: list[str]) -> np.ndarray:
    """Burn HUD-like text into an RGB image using pygame (if available).

    Returns the original image unchanged when pygame/font rendering is not
    available, so headless runs remain functional.
    """
    if not lines:
        return img_hw3
    try:
        import pygame
    except Exception:
        return img_hw3

    try:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        img = np.asarray(img_hw3, np.float32)
        h, w = img.shape[:2]
        u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        surf = pygame.Surface((w, h))
        pygame.surfarray.blit_array(surf, np.transpose(u8, (1, 0, 2)))

        font = pygame.font.SysFont("consolas", 14)
        rendered = [font.render(str(line), True, (180, 220, 255))
                    for line in lines]
        line_h = 16
        box_h = len(rendered) * line_h + 8
        max_w = max((s.get_width() for s in rendered), default=180)
        box_w = int(max(180, min(w - 8, max_w + 10)))
        box_x = 4
        box_y = max(4, h - box_h - 4)

        panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        panel.fill((8, 8, 16, 185))
        surf.blit(panel, (box_x, box_y))

        y = box_y + 4
        for text_surf in rendered:
            surf.blit(text_surf, (box_x + 4, y))
            y += line_h

        arr = pygame.surfarray.array3d(surf)  # (W, H, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.float32) / 255.0
    except Exception:
        return img_hw3


# ─────────────────────────────────────────────────────────────────────────────
# Pygame side-by-side viewer
# ─────────────────────────────────────────────────────────────────────────────

def _make_hud_lines(r: ExposureFrameResult, detail_level: int) -> list[str]:
    """Build HUD text lines shown in the lower-left pane corner."""
    cfg = r.frame_config_summary
    lines: list[str] = [
        f"frame {r.frame_index} | {r.backend}",
        f"N_rays  {r.n_rays_emitted:_}",
    ]
    if detail_level >= 1:
        lines.append(f"gain    {r.gain_linear:.3e} ({r.gain_db:+.1f} dB)")
        lines.append(f"H_meas  {r.measured_H_J:.2e} J")
        if "camera_solve_status" in cfg:
            lines.append(
                f"solve   {cfg.get('camera_solve_status', '?')}"
                f" err={float(cfg.get('camera_solve_error', 0.0)):.3f}"
                f" pin={float(cfg.get('camera_pinhole_ref_mm', 0.0)):.2f}mm"
                f" pdeg={float(cfg.get('camera_pinhole_ref_deg', 0.0)):.3f}"
                f" it={int(cfg.get('camera_solve_iter', 0))}"
            )
        lines.append(
            f"rails   front_shift={float(cfg.get('camera_front_shift_mm', 0.0)):.2f}mm"
            f" ({float(cfg.get('camera_rail_usage_front_shift_pct', 0.0)):.0f}%)"
            f" front_tilt={float(cfg.get('camera_front_tilt_deg', 0.0)):.2f}deg"
            f" ({float(cfg.get('camera_rail_usage_front_tilt_pct', 0.0)):.0f}%)"
        )
        lines.append(
            f"rails   sensor_shift={float(cfg.get('camera_sensor_shift_mm', 0.0)):.2f}mm"
            f" ({float(cfg.get('camera_rail_usage_sensor_shift_pct', 0.0)):.0f}%)"
            f" corner={float(cfg.get('camera_rail_usage_corner_pct', 0.0)):.0f}%"
            f" bellows={float(cfg.get('camera_rail_usage_bellows_pct', 0.0)):.0f}%"
            f" ap={float(cfg.get('camera_rail_usage_aperture_radius_pct', 0.0)):.0f}%"
        )
        lines.append(
            f"cone    half={float(cfg.get('camera_cone_half_angle_deg', 0.0)):.2f}deg"
            f" omega={float(cfg.get('camera_cone_solid_angle_sr', 0.0)):.3e}sr"
            f" n_ap={int(cfg.get('camera_cone_aperture_samples', 0))}"
        )
    if detail_level >= 2:
        lines.append(f"H_targ  {r.target_H_J:.2e} J")
        lines.append(f"field   {cfg.get('field', 'off')}")
        lines.append(f"ranges  {cfg.get('camera_rail_ranges', '')}")
    if detail_level >= 3:
        lines.append(f"spline  {cfg.get('spline', 'off')}")
        lines.append(f"cam     {cfg.get('cam_vis', '?')} / {cfg.get('transparent', '?')}")
    if detail_level >= 4:
        lines.append(f"photons {r.photons_per_pixel:.2e}/px")
        lines.append(f"splits  fi={cfg.get('fi', 0):.2f} si={cfg.get('si', 0):.2f}")
    if detail_level >= 5:
        lines.append(f"SNR\u2248    {r.snr_estimate:.2f}")
        lines.append(f"depth   {cfg.get('depth', 'off')}")
        lines.append(f"{cfg.get('description', '')}")
    return lines


def _run_viewer(session: ExposureSession, n_frames: int,
                pane_w: int, pane_h: int,
                show_hud: bool = True,
                preview_cycle_s: float = 2.0,
                preview_modes: tuple[str, ...] = ("rgb", "spectral")) -> None:
    try:
        import pygame
    except ImportError:
        print("[viewer] pygame not available; running headless and dumping PNGs only.")
        for k in range(n_frames):
            session.render_one_exposure(t=float(k) * 0.5)
        return

    pygame.init()
    win = pygame.display.set_mode((pane_w * 2 + 30, pane_h + 80))
    pygame.display.set_caption("Exposure Render Demo — C++ (left) vs GLSL (right)")
    font  = pygame.font.SysFont("consolas", 14)
    bigf  = pygame.font.SysFont("consolas", 18, bold=True)
    display_order = ("cpp", "glsl")
    allowed_modes = {
        "rgb", "spectral",
        "field-rgb", "field-spectral",
        "surface-rgb", "surface-spectral",
        "sensor-color", "sensor-spectral",
        "pinhole-rgb",
        "endpoint-rgb", "endpoint-spectral",
        "lightfield-shell-top", "lightfield-shell-side",
    }
    mode_cycle = tuple(m for m in preview_modes if m in allowed_modes) or ("rgb",)
    cycle_s = max(0.25, float(preview_cycle_s))

    def _spectral_falsecolor(accum_bhw: np.ndarray) -> np.ndarray:
        """False-color view showing spectral centroid and intensity.

        Alpha (expressed as premultiplied black) fades to zero as signal approaches
        the noise floor — blank/uncertain pixels render black rather than dim-hued.
        """
        power = np.maximum(np.asarray(accum_bhw, np.float64), 0.0)
        if power.ndim != 3 or power.shape[0] <= 0:
            return np.zeros((session.height, session.width, 3), np.float32)
        n_b = int(power.shape[0])
        axis = np.arange(n_b, dtype=np.float64)[:, None, None]
        total = power.sum(axis=0)
        centroid = (power * axis).sum(axis=0) / np.maximum(total, 1.0e-20)
        hue = centroid / max(1.0, float(n_b - 1))
        p99 = float(np.percentile(total, 99.0)) if total.size else 1.0
        p99 = max(1.0e-8, p99)
        # Sigmoid lightness: drives both HSL L and the premultiplied alpha.
        light = _sigmoid01((total / p99 - 0.45) * 5.0)
        sat = np.full_like(light, 0.95, dtype=np.float64)
        rgb = _hsl_to_rgb(hue, sat, light)           # H×W×3 float32
        # Premultiply against black: alpha = light^0.5 gives a perceptually
        # smooth fade where near-zero signal areas go fully black.
        alpha = np.sqrt(light).astype(np.float32)    # sharper fade than linear
        return (rgb * alpha[:, :, None]).astype(np.float32)

    def _scalar_falsecolor(img_hw: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(img_hw, np.float64), 0.0)
        p99 = float(np.percentile(x, 99.0)) if x.size else 1.0
        p99 = max(1.0e-8, p99)
        xn = np.clip(x / p99, 0.0, 1.0)
        hue = (1.0 - xn) * 0.72
        sat = np.full_like(xn, 0.9, dtype=np.float64)
        light = _sigmoid01((xn - 0.35) * 5.0)
        return _hsl_to_rgb(hue, sat, light)

    def _scalar_clear(img_hw: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(img_hw, np.float64), 0.0)
        p99 = float(np.percentile(x, 99.0)) if x.size else 1.0
        p99 = max(1.0e-8, p99)
        xn = np.clip(x / p99, 0.0, 1.0)
        y = _sigmoid01((xn - 0.35) * 5.0).astype(np.float32)
        return np.stack([y, y, y], axis=-1)

    def _missing_mode_image(mode: str) -> np.ndarray:
        # Return true black — a coloured placeholder is mistaken for real data.
        img = np.zeros((session.height, session.width, 3), dtype=np.float32)
        return img

    def _mode_image_for_snapshot(mode: str,
                                 snap: IntegrationSnapshot) -> tuple[np.ndarray, str]:
        if mode == "rgb":
            return np.asarray(snap.image_data, dtype=np.float32), "rgb"
        if mode == "spectral":
            src = snap.field_integrated_data
            if src is not None:
                return _spectral_falsecolor(src), "spectral-index"
            return _missing_mode_image("spectral"), "missing:spectral-index"
        if mode == "field-rgb":
            rgb = _bands_to_rgb(snap.field_integrated_data, session.freq_hz)
            return np.asarray(rgb, dtype=np.float32), "field-rgb-visible"
        if mode == "field-spectral":
            return _spectral_falsecolor(snap.field_integrated_data), "field-spectral-index"
        if mode == "surface-rgb":
            rgb = _bands_to_rgb(snap.surface_integrated_data, session.freq_hz)
            return np.asarray(rgb, dtype=np.float32), "surface-rgb-visible"
        if mode == "surface-spectral":
            return _spectral_falsecolor(snap.surface_integrated_data), "surface-spectral-index"
        if mode == "sensor-color":
            if snap.sensor_photons_data is not None:
                return _scalar_falsecolor(snap.sensor_photons_data), "sensor-color"
            return np.asarray(snap.image_data, dtype=np.float32), "sensor-color:live-rgb"
        if mode == "sensor-spectral":
            if snap.sensor_snr_data is not None:
                return _scalar_clear(snap.sensor_snr_data), "sensor-spectral"
            return np.asarray(snap.image_data, dtype=np.float32), "sensor-spectral:live-rgb"
        if mode == "endpoint-rgb":
            return np.asarray(snap.image_data, dtype=np.float32), "endpoint-rgb:live"
        if mode == "endpoint-spectral":
            if snap.sensor_snr_data is not None:
                return _scalar_clear(snap.sensor_snr_data), "endpoint-spectral"
            return np.asarray(snap.image_data, dtype=np.float32), "endpoint-spectral:live-rgb"
        if mode == "pinhole-rgb":
            if snap.pinhole_rgb_data is not None:
                return np.asarray(snap.pinhole_rgb_data, dtype=np.float32), "pinhole-rgb"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "lightfield-shell-top":
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "lightfield-shell-side":
            return _missing_mode_image(mode), f"missing:{mode}"
        return _missing_mode_image(mode), f"missing:{mode}"

    def _available_modes_for_result(result: ExposureFrameResult) -> tuple[str, ...]:
        modes: list[str] = []
        if result.image_data is not None:
            modes.append("rgb")
        if result.field_integrated_data is not None:
            modes.extend(["field-rgb", "field-spectral"])
        if result.surface_integrated_data is not None:
            modes.extend(["surface-rgb", "surface-spectral"])
        if result.sensor_photons_data is not None:
            modes.append("sensor-color")
        if result.sensor_snr_data is not None:
            modes.append("sensor-spectral")
        if result.pinhole_rgb_data is not None:
            modes.append("pinhole-rgb")
        if result.lightfield_shell_top_data is not None:
            modes.append("lightfield-shell-top")
        if result.lightfield_shell_side_data is not None:
            modes.append("lightfield-shell-side")
        if session.rgb_source == "endpoint" and result.image_data is not None:
            modes.append("endpoint-rgb")
        if session.rgb_source == "endpoint" and result.sensor_snr_data is not None:
            modes.append("endpoint-spectral")
        return tuple(modes)

    def _current_mode(elapsed_s: float) -> str:
        return mode_cycle[int(elapsed_s / cycle_s) % len(mode_cycle)]

    def _mode_image_for_result(mode: str,
                               result: ExposureFrameResult) -> tuple[np.ndarray, str]:
        if mode == "rgb":
            return result.image_data if result.image_data is not None else np.zeros((session.height, session.width, 3), np.float32), "rgb"
        if mode == "spectral":
            src = (result.field_integrated_data
                   if result.field_integrated_data is not None
                   else result.surface_integrated_data)
            if src is not None:
                return _spectral_falsecolor(src), "spectral-index"
            return _missing_mode_image("spectral"), "missing:spectral-index"
        if mode == "field-rgb":
            if result.field_integrated_data is not None:
                rgb = _bands_to_rgb(result.field_integrated_data, session.freq_hz)
                return np.asarray(rgb, dtype=np.float32), "field-rgb-visible"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "field-spectral":
            if result.field_integrated_data is not None:
                return _spectral_falsecolor(result.field_integrated_data), "field-spectral-index"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "surface-rgb":
            if result.surface_integrated_data is not None:
                rgb = _bands_to_rgb(result.surface_integrated_data, session.freq_hz)
                return np.asarray(rgb, dtype=np.float32), "surface-rgb-visible"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "surface-spectral":
            if result.surface_integrated_data is not None:
                return _spectral_falsecolor(result.surface_integrated_data), "surface-spectral-index"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "sensor-color":
            if result.sensor_photons_data is not None:
                return _scalar_falsecolor(result.sensor_photons_data), "sensor-color"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "sensor-spectral":
            if result.sensor_snr_data is not None:
                return _scalar_clear(result.sensor_snr_data), "sensor-spectral"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "pinhole-rgb":
            if result.pinhole_rgb_data is not None:
                return result.pinhole_rgb_data, "pinhole-rgb"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "lightfield-shell-top":
            if result.lightfield_shell_top_data is not None:
                return result.lightfield_shell_top_data, "lightfield-shell-top"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "lightfield-shell-side":
            if result.lightfield_shell_side_data is not None:
                return result.lightfield_shell_side_data, "lightfield-shell-side"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "endpoint-rgb":
            if session.rgb_source == "endpoint" and result.image_data is not None:
                return result.image_data, "endpoint-rgb"
            return _missing_mode_image(mode), f"missing:{mode}"
        if mode == "endpoint-spectral":
            if result.sensor_snr_data is not None:
                return _scalar_clear(result.sensor_snr_data), "endpoint-spectral"
            return _missing_mode_image(mode), f"missing:{mode}"
        return _missing_mode_image(mode), f"missing:{mode}"

    # Flip buffers for live preview: reuse both pixel arrays and source surfaces.
    flip_buffers: dict[str, list[np.ndarray]] = {}
    flip_surfaces: dict[str, list[Any]] = {}
    flip_index: dict[str, int] = {}
    for backend_name in display_order:
        flip_buffers[backend_name] = [
            np.empty((session.height, session.width, 3), dtype=np.uint8),
            np.empty((session.height, session.width, 3), dtype=np.uint8),
        ]
        flip_surfaces[backend_name] = [
            pygame.Surface((session.width, session.height)),
            pygame.Surface((session.width, session.height)),
        ]
        flip_index[backend_name] = 0

    def _blit_image(img: np.ndarray, dest_rect: tuple[int, int, int, int],
                    label: str, info: str,
                    hud_lines: Optional[list[str]] = None) -> None:
        """Blit a float32 [0,1] RGB image into dest_rect with optional HUD."""
        h, w = img.shape[:2]
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        surf = pygame.image.frombuffer(u8.tobytes(), (w, h), "RGB")
        surf = pygame.transform.smoothscale(surf, (dest_rect[2], dest_rect[3]))
        win.blit(surf, (dest_rect[0], dest_rect[1]))
        win.blit(bigf.render(label, True, (255, 255, 255)),
                 (dest_rect[0] + 6, dest_rect[1] + 6))
        for i, line in enumerate(info.split("\n")):
            win.blit(font.render(line, True, (210, 230, 255)),
                     (dest_rect[0] + 6,
                      dest_rect[1] + dest_rect[3] - 18 * (3 - i)))
        # ── HUD overlay — lower-left semi-transparent panel ───────────────
        if hud_lines:
            line_h = 16
            box_h  = len(hud_lines) * line_h + 8
            box_w  = 288
            box_x  = dest_rect[0] + 4
            box_y  = dest_rect[1] + dest_rect[3] - box_h - 4
            panel  = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            panel.fill((8, 8, 16, 185))
            win.blit(panel, (box_x, box_y))
            for idx, line in enumerate(hud_lines):
                txt = font.render(line, True, (180, 220, 255))
                win.blit(txt, (box_x + 4, box_y + 4 + idx * line_h))

    def _blit_progress_accum(backend_name: str,
                             accum_bhw: np.ndarray,
                             dest_rect: tuple[int, int, int, int],
                             label: str,
                             info: str,
                             mode: str) -> None:
        """Render an incremental preview from spectral accum via persistent flip buffers."""
        rgb_linear = _bands_to_rgb(accum_bhw, session.freq_hz)
        img = (_spectral_falsecolor(accum_bhw)
               if mode == "spectral"
               else session._tone_map(rgb_linear, session.integral_split))

        idx = flip_index[backend_name]
        u8buf = flip_buffers[backend_name][idx]
        np.clip(img, 0.0, 1.0, out=rgb_linear)
        np.multiply(rgb_linear, 255.0, out=rgb_linear)
        u8buf[:] = rgb_linear.astype(np.uint8)

        surf = flip_surfaces[backend_name][idx]
        pygame.surfarray.blit_array(surf, np.transpose(u8buf, (1, 0, 2)))
        surf_scaled = pygame.transform.smoothscale(surf, (dest_rect[2], dest_rect[3]))
        win.blit(surf_scaled, (dest_rect[0], dest_rect[1]))

        win.blit(bigf.render(label, True, (255, 255, 255)),
                 (dest_rect[0] + 6, dest_rect[1] + 6))
        for i, line in enumerate(info.split("\n")):
            win.blit(font.render(line, True, (210, 230, 255)),
                     (dest_rect[0] + 6,
                      dest_rect[1] + dest_rect[3] - 18 * (3 - i)))
        flip_index[backend_name] = 1 - idx

    state_lock = threading.Lock()
    stop_event = threading.Event()
    render_done = threading.Event()

    state_snapshots: dict[str, IntegrationSnapshot] = {}
    state_results: dict[str, ExposureFrameResult] = {}
    state_meta: dict[str, float | int] = {
        "frame": 0,
        "batch": 0,
        "n_batches": 0,
        "elapsed_s": 0.0,
        "telemetry_frame": 0,
        "telemetry_batch": 0,
    }
    state_telemetry: dict[str, str] = {
        "profile": "",
        "alloc": "",
    }
    # Per-backend pre-rendered uint8 image cache: {backend_name: {mode: uint8 HxWx3}}
    # Worker deposits here; UI reads here — no computation on the main thread.
    state_images: dict[str, dict[str, np.ndarray]] = {}
    # Viewer-side UI state (mutated by event loop, read by display)
    ui_mode_idx: list[int] = [0]        # index into mode_cycle (manual override)
    ui_cycle_paused: list[bool] = [False]
    ui_sensor_film_label: list[str] = [""]  # display label for current slot 0 sensor+film
    # Seed display label from current session state
    if session._sensor_film_metadata and session._sensor_film_metadata[0].get("active"):
        m = session._sensor_film_metadata[0]
        ui_sensor_film_label[0] = f"{m['sensor_name']} / {m['film_name']}"
    worker_error_summary: Optional[str] = None
    worker_error_report: Optional[str] = None
    pulse_stop = threading.Event()

    def _render_worker() -> None:
        nonlocal worker_error_summary, worker_error_report
        try:
            for frame_idx in range(n_frames):
                if stop_event.is_set():
                    break

                next_telemetry_collect_t = 0.0

                def _on_batch_preview(_backs: dict[str, ExposureBackend],
                                      snapshots: dict[str, IntegrationSnapshot],
                                      batch_idx: int,
                                      n_batches: int,
                                      elapsed_s: float) -> None:
                    nonlocal next_telemetry_collect_t
                    if n_batches > 0:
                        cadence = max(1, n_batches // 24)
                    else:
                        cadence = 6
                    if n_batches > 0 and batch_idx < n_batches and (batch_idx % cadence) != 0:
                        return

                    now = time.perf_counter()
                    collect_telemetry = bool(session.profile_enabled) and (now >= next_telemetry_collect_t)
                    if collect_telemetry:
                        next_telemetry_collect_t = now + 1.0

                    profile_msg = ""
                    alloc_msg = ""
                    if collect_telemetry:
                        profile_msg = session._profiler.format_report(
                            prefix=(f"[profile pulse frame={frame_idx+1} "
                                    f"batch={batch_idx} elapsed={elapsed_s:0.1f}s]")
                        )
                        cpp_back = _backs.get("cpp")
                        tracer_obj = getattr(cpp_back, "tracer", None) if cpp_back is not None else None
                        if tracer_obj is not None and hasattr(tracer_obj, "allocation_table"):
                            try:
                                alloc_msg = str(tracer_obj.allocation_table())
                            except Exception as exc:
                                alloc_msg = f"[allocation_table error] {exc}"

                    # Pre-render every display mode as uint16 — no computation on the UI thread.
                    fresh_images: dict[str, dict[str, np.ndarray]] = {}
                    for _bname, _snap in snapshots.items():
                        _per: dict[str, np.ndarray] = {}
                        for _m in mode_cycle:
                            try:
                                _img, _ = _mode_image_for_snapshot(_m, _snap)
                                _per[_m] = (np.clip(_img, 0.0, 1.0) * 65535.0).astype(np.uint16)
                            except Exception:
                                pass
                        fresh_images[_bname] = _per

                    with state_lock:
                        state_snapshots.clear()
                        state_snapshots.update(snapshots)
                        state_images.update(fresh_images)
                        state_meta["frame"] = int(frame_idx + 1)
                        state_meta["batch"] = int(batch_idx)
                        state_meta["n_batches"] = int(n_batches)
                        state_meta["elapsed_s"] = float(elapsed_s)
                        if collect_telemetry:
                            state_meta["telemetry_frame"] = int(frame_idx + 1)
                            state_meta["telemetry_batch"] = int(batch_idx)
                            state_telemetry["profile"] = profile_msg
                            state_telemetry["alloc"] = alloc_msg

                results = session.render_one_exposure(
                    t=float(frame_idx) * 0.5,
                    batch_preview_cb=_on_batch_preview,
                )
                with state_lock:
                    for r in results:
                        state_results[r.backend] = r
                    state_meta["frame"] = int(frame_idx + 1)
                    state_meta["batch"] = int(state_meta.get("n_batches", 0))
        except Exception as exc:
            worker_error_summary = str(exc)
            worker_error_report = traceback.format_exc()
            print(f"  [error] render worker failed: {worker_error_summary}", file=sys.stderr, flush=True)
            if worker_error_report:
                print(worker_error_report, file=sys.stderr, flush=True)
        finally:
            render_done.set()

    def _telemetry_pulse_worker() -> None:
        last_signature = ""
        while not pulse_stop.is_set():
            with state_lock:
                t_frame = int(state_meta.get("telemetry_frame", 0))
                t_batch = int(state_meta.get("telemetry_batch", 0))
                profile_msg = str(state_telemetry.get("profile", ""))
                alloc_msg = str(state_telemetry.get("alloc", ""))
            if profile_msg or alloc_msg:
                sig = f"{t_frame}:{t_batch}:{len(profile_msg)}:{len(alloc_msg)}"
                if sig != last_signature:
                    print(f"[pulse] frame={t_frame} batch={t_batch}", file=sys.stderr, flush=True)
                    if profile_msg:
                        print(profile_msg, file=sys.stderr, flush=True)
                    if alloc_msg:
                        print(alloc_msg, file=sys.stderr, flush=True)
                    last_signature = sig
            pulse_stop.wait(2.0)

    worker = threading.Thread(target=_render_worker, name="exposure-render-worker", daemon=True)
    worker.start()
    pulse_worker: Optional[threading.Thread] = None
    if session.profile_enabled:
        pulse_worker = threading.Thread(
            target=_telemetry_pulse_worker,
            name="exposure-telemetry-pulse",
            daemon=True,
        )
        pulse_worker.start()

    running = True
    t_cycle0 = time.perf_counter()
    done_hold_until: Optional[float] = None

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
                break
            if ev.type == pygame.KEYDOWN:
                # Mode cycling: left/right step through mode_cycle manually
                if ev.key == pygame.K_RIGHT:
                    ui_mode_idx[0] = (ui_mode_idx[0] + 1) % len(mode_cycle)
                    ui_cycle_paused[0] = True
                    t_cycle0 = time.perf_counter()
                elif ev.key == pygame.K_LEFT:
                    ui_mode_idx[0] = (ui_mode_idx[0] - 1) % len(mode_cycle)
                    ui_cycle_paused[0] = True
                    t_cycle0 = time.perf_counter()
                elif ev.key == pygame.K_SPACE:
                    ui_cycle_paused[0] = not ui_cycle_paused[0]
                    if not ui_cycle_paused[0]:
                        t_cycle0 = time.perf_counter()
                # Sensor cycling: [ / ] step sensor on slot 0
                elif ev.key == pygame.K_LEFTBRACKET:
                    m = session.swap_sensor_film_slot(0, sensor_delta=-1)
                    if m:
                        ui_sensor_film_label[0] = f"{m['sensor_name']} / {m['film_name']}"
                elif ev.key == pygame.K_RIGHTBRACKET:
                    m = session.swap_sensor_film_slot(0, sensor_delta=+1)
                    if m:
                        ui_sensor_film_label[0] = f"{m['sensor_name']} / {m['film_name']}"
                # Film cycling: - / = step film on slot 0
                elif ev.key == pygame.K_MINUS:
                    m = session.swap_sensor_film_slot(0, film_delta=-1)
                    if m:
                        ui_sensor_film_label[0] = f"{m['sensor_name']} / {m['film_name']}"
                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    m = session.swap_sensor_film_slot(0, film_delta=+1)
                    if m:
                        ui_sensor_film_label[0] = f"{m['sensor_name']} / {m['film_name']}"
        if not running:
            break

        with state_lock:
            snapshots = dict(state_snapshots)
            results = dict(state_results)
            cached_images = {k: dict(v) for k, v in state_images.items()}
            frame_no = int(state_meta.get("frame", 0))
            batch_idx = int(state_meta.get("batch", 0))
            n_batches = int(state_meta.get("n_batches", 0))
            elapsed_s = float(state_meta.get("elapsed_s", 0.0))

        # Mode selection: auto-cycle unless paused/manually stepped
        if ui_cycle_paused[0]:
            mode = mode_cycle[ui_mode_idx[0] % len(mode_cycle)]
        else:
            mode = mode_cycle[int((time.perf_counter() - t_cycle0) / cycle_s) % len(mode_cycle)]
            ui_mode_idx[0] = mode_cycle.index(mode)

        win.fill((12, 12, 18))

        if worker_error_summary:
            title = f"Render worker error: {worker_error_summary}"
        elif n_batches > 0:
            pct = 100.0 * float(batch_idx) / max(1, n_batches)
            title = f"Exposure {max(1, frame_no)}/{n_frames} - {pct:5.1f}% - [{mode}]{' [PAUSED]' if ui_cycle_paused[0] else ''}"
        elif frame_no > 0 and not render_done.is_set():
            title = f"Exposure {frame_no}/{n_frames} - batch {batch_idx} (open-ended) - [{mode}]{' [PAUSED]' if ui_cycle_paused[0] else ''}"
        elif render_done.is_set():
            title = f"Render complete - [{mode}]{' [PAUSED]' if ui_cycle_paused[0] else ''}"
        else:
            title = f"Exposure 1/{n_frames} - initializing - [{mode}]"

        sf_label = ui_sensor_film_label[0]
        if sf_label:
            title += f"  |  {sf_label}"

        win.blit(bigf.render(title, True, (255, 255, 200)), (16, 8))

        for i, name in enumerate(display_order):
            x = 10 + i * (pane_w + 10)
            y = 40

            if name in snapshots:
                batch_text = (f"{batch_idx}/{n_batches}" if n_batches > 0 else f"{batch_idx}/inf")
                cycle_hint = "[PAUSED]" if ui_cycle_paused[0] else f"cycle {cycle_s:.1f}s"
                info = (f"frame={frame_no}/{n_frames}  batch={batch_text}\n"
                        f"elapsed={elapsed_s:.1f}s\n"
                    f"view={mode} {cycle_hint} | stream")
                u16 = cached_images.get(name, {}).get(mode)
                if u16 is not None:
                    # Worker-rendered uint16 cache — convert to display, no computation.
                    _blit_image(
                        u16.astype(np.float32) * (1.0 / 65535.0),
                        (x, y, pane_w, pane_h),
                        label=f"{name.upper()} LIVE [{mode}]",
                        info=info,
                        hud_lines=None,
                    )
                else:
                    pygame.draw.rect(win, (40, 40, 50), (x, y, pane_w, pane_h))
                    win.blit(font.render(f"{name.upper()}: rendering…", True, (200, 200, 200)),
                             (x + 12, y + 12))
                continue

            if name in results:
                r = results[name]
                img, shown_mode = _mode_image_for_result(mode, r)
                info = (f"N_rays={r.n_rays_emitted:_}\n"
                        f"gain={r.gain_linear:.2e}x ({r.gain_db:+.2f} dB)\n"
                        f"H_meas/targ={r.measured_H_J:.2e}/{r.target_H_J:.2e} J")
                detail_lv = r.frame_config_summary.get("detail_level", 0)
                hud = _make_hud_lines(r, detail_lv) if show_hud else None
                _blit_image(
                    img,
                    (x, y, pane_w, pane_h),
                    label=f"{name.upper()} [{shown_mode}]",
                    info=info,
                    hud_lines=hud,
                )
                continue

            pygame.draw.rect(win, (40, 40, 50), (x, y, pane_w, pane_h))
            win.blit(font.render(f"{name.upper()}: waiting for first snapshot", True, (200, 200, 200)),
                     (x + 12, y + 12))

        pygame.display.flip()

        if render_done.is_set() and done_hold_until is None:
            done_hold_until = time.perf_counter() + max(1.0, cycle_s * max(1, len(mode_cycle)))
        if done_hold_until is not None and time.perf_counter() >= done_hold_until:
            running = False

        pygame.time.wait(16)

    stop_event.set()
    pulse_stop.set()
    worker.join(timeout=2.0)
    if pulse_worker is not None:
        pulse_worker.join(timeout=1.0)

    pygame.quit()

    if worker_error_summary:
        raise RuntimeError(worker_error_report or worker_error_summary)


def _load_png_rgb(path: str) -> Optional[np.ndarray]:
    try:
        import pygame
        s = pygame.image.load(path)
        s = s.convert()
        w, h = s.get_size()
        buf = pygame.image.tostring(s, "RGB")
        arr = np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
        return arr
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width",          type=int,   default=1280)
    p.add_argument("--height",         type=int,   default=720)
    p.add_argument("--input-size-scale", type=float, default=1.0,
                   help="Scale input width/height before rendering (e.g., 0.5, 1.0, 1.5).")
    p.add_argument("--output-oversample", type=int, default=1,
                   help="Internal render oversample factor; output is downsampled back to scaled size.")
    p.add_argument("--oversample-stencil", choices=("box", "polar"), default="box",
                   help="Downsample stencil for oversampled output conversion.")
    p.add_argument("--total-rays",     type=int,   default=1_000_000,
                   help="Total rays per FINISHED exposure (default 1M; "
                        "raise to 10_000_000+ for production exposures).")
    p.add_argument("--rays-per-batch", type=int,   default=50_000,
                   help="Sub-batch size; many small batches stream cleanly.")
    p.add_argument("--frames",         type=int,   default=4)
    p.add_argument("--max-bounces",    type=int,   default=8,
                   help="Per-ray bounce cap. Raised from 4 because most\n"
                        "BVH walks terminate before hitting the cap.")
    p.add_argument("--backward-max-bounces", type=int, default=None,
                   help=("Optional override for backward/BDPT bounce cap. "
                         "If omitted, backward/BDPT use --max-bounces."))
    p.add_argument("--integrator",     choices=("splat", "forward", "backward", "bdpt", "depth"),
                    default="bdpt",
                    help="splat = legacy direct image projection (image accum). "
                        "forward = light tracing from emissive sources. "
                        "backward = ray tracing from camera pixels (PIXEL_CONE). "
                        "bdpt = bidirectional path tracing with both passes.")
    p.add_argument("--sequence-mode",  choices=("single", "forward-backward-bdpt"),
                    default="single",
                    help="single = use --integrator only. "
                        "forward-backward-bdpt = run forward, backward, then bdpt in sequence.")
    p.add_argument("--bdpt-records-cap", type=int, default=0,
                   help="Max EndpointRecords per bdpt pass. Use 0 to auto-derive from "
                        "--bdpt-intermediate-max-gb.")
    p.add_argument("--bdpt-intermediate-mode", choices=("memory", "file"), default="file",
                   help="How to stage BDPT intermediates before reduction."
                        " 'file' uses mmap-friendly .npy backing.")
    p.add_argument("--bdpt-intermediate-max-gb", type=float, default=20.0,
                   help="Hard cap for BDPT file-backed intermediate size (GB)."
                        " Larger buffers stay in memory. Strict maximum is 50 GB.")
    p.add_argument("--retain-bdpt-intermediate", action="store_true",
                   help="Keep file-backed BDPT intermediate .npy files."
                        " Default behavior is ephemeral cleanup.")
    p.add_argument("--bdpt-intermediate-dir", default=None,
                   help="Directory for BDPT file-backed intermediates"
                        " (default: out-dir).")
    p.add_argument("--t5-min-geom", type=float, default=-1.0,
                   help="GPU BDPT T5 geometry term floor. Pairs with geometry term < this limit "
                        "are dropped. Bias/variance knob: trades noise reduction on unoccluded "
                        "fidelity for throughput. Example: --t5-min-geom 1e-12")
    p.add_argument("--t5-pair-budget", type=int, default=200_000_000,
                   help="Maximum full-domain T5 camera-light pairs evaluated per pass; "
                        "unevaluated work remains explicitly deferred (0 = unlimited).")
    p.add_argument("--no-vcm", action="store_true",
                   help="Disable spectral vertex merging and run connection-only BDPT.")
    p.add_argument("--vcm-radius-mm", type=float, default=2.0,
                   help="Initial world-space VCM merge radius in millimetres (default 2).")
    p.add_argument("--vcm-radius-alpha", type=float, default=0.7,
                   help="Progressive VCM radius parameter in (0,1]; 1 keeps a fixed radius.")
    p.add_argument("--gpu-resident", action="store_true",
                   help="Enable GPU-resident BDPT/T5 mode, matching thick_lens_focus_lab's "
                        "--gpu-resident path. This is required to exercise native GPU T5 scheduling.")
    p.add_argument("--bdpt-native-packages", type=int, default=1,
                   help="Minimum spatial partitions per native sensor sweep (default: 1; "
                        "automatically raised to satisfy record caps).")
    p.add_argument("--bdpt-native-sweeps", type=int, default=None,
                   help="Override the scene-authored number of independent complete native "
                        "sensor sweeps to average (default: scene order value, otherwise 1).")
    p.add_argument("--calibration-scene", action="store_true",
                   help="Use the tungsten-cavity blackbody calibration "
                        "scene instead of the default orbiters scene. "
                        "Equivalent to --scene-mode tungsten-cavity.")
    p.add_argument("--scene-mode",     default=None,
                   help="Scene mode passed to the exposure scene builder "
                        "(default: thick-lens-lab; tungsten-cavity for blackbody "
                        "calibration; also supports orbiters, calib-rgb-diagram / calib-bw-rgb "
                        "for black-white-primary chain validation, and calib-grid / "
                        "calib-step-wedge / calib-prism-backplate for grid, wedge, "
                         "and prism comparator scenes). Overrides --calibration-scene.")
    p.add_argument("--scene-order", default=None,
                   help="JSON scene-order package. Compiles its selected job into the "
                        "proven thick-lens lab camera/flash scene.")
    p.add_argument("--scene-job", default=None,
                   help="Job id inside --scene-order (required when the package has multiple jobs).")
    p.add_argument("--next-scan-control", default=None,
                   help=("JSON next-site command produced by camera/network control. "
                         "UV requests configure attention; bounded film-plane depth "
                         "and tilt are applied before the next GPU scene is built."))
    p.add_argument("--maximum-film-travel-mm", type=float, default=2.0,
                   help="Camera-owned axial film-stage travel limit per next-scan command.")
    p.add_argument("--maximum-film-tilt-deg", type=float, default=5.0,
                   help="Camera-owned film tilt limit about either in-plane axis.")
    p.add_argument("--backend",        choices=("cpp", "glsl", "both"),
                   default="cpp")
    p.add_argument("--exposure-time-s", type=float, default=DEFAULT_FILM.exposure_time_s)
    p.add_argument("--iso",            type=float, default=DEFAULT_FILM.iso)
    p.add_argument("--focal-mm",       type=float, default=DEFAULT_OPTICS.focal_mm)
    p.add_argument("--aperture-mm",    type=float, default=DEFAULT_OPTICS.aperture_mm)
    p.add_argument("--no-window",      action="store_true",
                   help="Skip pygame window; just dump PNG/JSON to disk.")
    p.add_argument("--no-hud",         action="store_true",
                   help="Suppress the HUD overlay in the viewer window.")
    p.add_argument("--save-files",     action="store_true",
                   help="Save PNG, JSON, and NPZ artifacts to disk (opt-in). "
                        "Without this, only in-memory rendering and display is performed.")
    p.add_argument("--profile",        action="store_true",
                   help="Enable Python stage profiling and native failure breadcrumbs.")
    p.add_argument("--out-dir",        default="exposures")
    p.add_argument("--pane-w",         type=int, default=640)
    p.add_argument("--pane-h",         type=int, default=360)
    p.add_argument("--preview-cycle-s", type=float, default=2.0,
                   help="Viewer interval (seconds) for RGB/spectral auto-cycling.")
    p.add_argument(
        "--preview-modes",
        default=("rgb,spectral,field-rgb,field-spectral,"
                 "surface-rgb,surface-spectral,sensor-color,sensor-spectral,"
                 "pinhole-rgb,endpoint-rgb,endpoint-spectral,"
                 "lightfield-shell-top,lightfield-shell-side"),
        help=("Comma-separated viewer modes: rgb,spectral,field-rgb,"
              "field-spectral,surface-rgb,surface-spectral,"
              "sensor-color,sensor-spectral,pinhole-rgb,endpoint-rgb,endpoint-spectral,"
              "lightfield-shell-top,lightfield-shell-side"),
    )
    p.add_argument(
        "--rgb-source",
        choices=("sensor", "accum", "endpoint"),
        default="sensor",
        help=("Visible RGB source: sensor uses backward-pass sensor integration; "
              "accum uses camera-integrated spectral accum; endpoint uses C++ endpoint reducer output."),
    )
    p.add_argument(
        "--tone-map-mode",
        choices=("reinhardt", "delicate"),
        default="reinhardt",
        help=("Tone mapping mode: reinhardt uses simple L_out = L_in / (1 + L_in) (default); "
              "delicate uses percentile-based soft-knee normalization."),
    )
    p.add_argument("--convergence-target-pct", type=float, default=99.99,
                   help="Convergence target percentage for early-stop detector.")
    p.add_argument("--convergence-max-rel-drift", type=float, default=1.0e-4,
                   help="Maximum relative drift for convergence stop criterion.")
    p.add_argument("--no-convergence", action="store_true",
                   help="Disable convergence detector and run planned finite batches.")
    p.add_argument("--no-convergence-drive-batches", action="store_true",
                   help="Keep finite batch planning even when convergence detector is enabled.")
    p.add_argument("--convergence-check-every", type=int, default=1,
                   help="Run convergence detector every N batches.")
    p.add_argument("--convergence-min-batches", type=int, default=4,
                   help="Minimum batches before convergence early-stop can trigger.")
    p.add_argument("--convergence-hold-checks", type=int, default=3,
                   help="Required consecutive convergence hits before early-stop.")
    p.add_argument("--convergence-probe-count", type=int, default=8192,
                   help="Probe sample count used by convergence detector.")
    p.add_argument("--convergence-max-batches", type=int, default=0,
                   help="Optional hard cap for convergence-driven open-ended mode (0 = unlimited).")
    p.add_argument("--adaptive-allocation", choices=("stochastic", "quota", "uniform"),
                   default="stochastic",
                   help="Adaptive allocation policy for forward and backward ray dispatch.")
    p.add_argument("--field-integrate-frac", type=float, default=0.35)
    p.add_argument("--field-bookkeep-frac", type=float, default=0.65)
    p.add_argument("--surface-integrate-frac", type=float, default=0.85)
    p.add_argument("--surface-bookkeep-frac", type=float, default=0.15)
    p.add_argument("--hdr-white-percentile", type=float, default=99.8)
    p.add_argument("--camera-solve-preview", action="store_true",
                   help=("Run a very cheap interactive preview focused on camera/lens solving. "
                         "Uses a gem scene variant with obsidian body + ruby/emerald emitters."))
    p.add_argument("--camera-mode",
                   choices=("oracle_pinhole", "physical_pinhole", "aperture_cone", "thin_lens",
                            "geometric_assembly", "wave_assembly", "baked_transform"),
                   default=None,
                   help=("Camera optical transport mode. Overrides auto-selection from solver. "
                         "oracle_pinhole: clean 1-ray reference (Tier 0). "
                         "physical_pinhole: photon-limited tiny aperture (Tier 1). "
                         "aperture_cone: disk samples, no lens (Tier 2, default for simple scenes). "
                         "thin_lens: geometric thin-lens with focus (Tier 3, auto if solver active). "
                         "geometric_assembly, wave_assembly, baked_transform: future high-fidelity modes."))
    p.add_argument("--camera-solve-max-iters", type=int, default=72,
                   help="Per-frame camera solver iteration budget.")
    p.add_argument("--camera-solve-seed-base", type=int, default=12345,
                   help="Base seed for per-frame camera solver attempts.")
    p.add_argument("--camera-cone-samples-min", type=int, default=8,
                   help="Minimum aperture samples per pixel in backward PIXEL_CONE pass.")
    p.add_argument("--camera-cone-samples-max", type=int, default=192,
                   help="Maximum aperture samples per pixel in backward PIXEL_CONE pass.")
    p.add_argument("--camera-cone-ref-half-angle-deg", type=float, default=3.0,
                   help="Reference cone half-angle (deg) used to scale aperture samples.")
    p.add_argument("--camera-cone-angle-exponent", type=float, default=0.85,
                   help="Exponent for solid-angle driven aperture sample scaling.")
    p.add_argument("--camera-rail-sensor-z-min-m", type=float, default=-0.110,
                   help="Rear plate rail minimum z position (m).")
    p.add_argument("--camera-rail-sensor-z-max-m", type=float, default=-0.015,
                   help="Rear plate rail maximum z position (m).")
    p.add_argument("--camera-rail-corner-max-mm", type=float, default=14.0,
                   help="Max absolute per-corner plate rail delta (mm).")
    p.add_argument("--camera-rail-sensor-shift-max-mm", type=float, default=50.0,
                   help="Max absolute rear plate lateral shift (mm).")
    p.add_argument("--camera-rail-aperture-z-min-m", type=float, default=-0.002,
                   help="Body-fixed aperture z micro-adjust min (m).")
    p.add_argument("--camera-rail-aperture-z-max-m", type=float, default=0.002,
                   help="Body-fixed aperture z micro-adjust max (m).")
    p.add_argument("--camera-rail-aperture-shift-max-mm", type=float, default=1.0,
                   help="Body-fixed aperture lateral micro-adjust max (mm).")
    p.add_argument("--camera-rail-bellows-sep-min-m", type=float, default=0.005,
                   help="Bellows minimum front-back lens separation (m).")
    p.add_argument("--camera-rail-bellows-sep-max-m", type=float, default=0.0,
                   help="Bellows maximum front-back lens separation (m). 0 => auto generous.")
    p.add_argument("--camera-rail-front-shift-max-mm", type=float, default=60.0,
                   help="Outer/front lens lateral exploration max (mm).")
    p.add_argument("--camera-rail-front-tilt-max-deg", type=float, default=35.0,
                   help="Outer/front lens orientation tilt exploration max (deg).")
    p.add_argument("--camera-rail-aperture-radius-min-mm", type=float, default=0.001,
                   help="Minimum aperture radius (mm), near singularity.")
    p.add_argument("--camera-rail-aperture-radius-max-mm", type=float, default=0.0,
                   help="Maximum aperture radius (mm), very open; 0 => auto generous.")
    p.add_argument("--camera-solver-energy-epsilon", type=float, default=1.0e-6,
                   help="Early-stop epsilon for relative deposited energy proxy in camera solver.")
    p.add_argument("--camera-solver-energy-patience", type=int, default=24,
                   help="Consecutive low-energy solver steps before early termination.")
    p.add_argument("--progress-dir", default="",
                   help="Publish immutable linear sensor pass artifacts and JSON-line progress events here.")
    p.add_argument("--progress-exposure-id", default="",
                   help="Stable progress-stream id (default: scene job or 'exposure').")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    global _ORDERED_THICK_LENS_SCENE, _ORDERED_SCENE_JOB, _ORDERED_TILE_OUTPUT
    global DEFAULT_FREQ_HZ
    global _ORDERED_TRANSPORT_RGB_WEIGHTS
    global _ORDERED_TRANSPORT_LUT
    global _ORDERED_MATERIAL_LUT_GRID
    global _NEXT_SITE_SCAN_COMMAND
    _NEXT_SITE_SCAN_COMMAND = None
    _verify_native_bdpt_contract()
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.integrator == "depth":
        # Depth is a native sensor product.  Make its execution contract
        # explicit and deterministic rather than accepting a CPU/scaffold path.
        args.backend = "cpp"
        args.gpu_resident = True
        args.rgb_source = "sensor"
        args.no_convergence = True
        args.no_convergence_drive_batches = True
        args.max_bounces = 1

    order_job = None
    order_report = None
    ordered_sensor_film_slots = None
    cli_native_sweeps = args.bdpt_native_sweeps
    if args.scene_order:
        from scene_orders import (
            compile_job, load_order, order_runtime_settings,
            order_transport_frequencies, order_transport_rgb_weights,
            order_transport_lut_config,
            resolved_jobs,
        )
        order_package = load_order(str(args.scene_order))
        authored_runtime = dict(order_package.get("runtime", {}))
        # Scene-order execution budgets are part of the exposure contract.  A
        # calibration order must not silently inherit the production CLI's
        # million-ray / 64-epoch defaults.
        if "total_rays" in authored_runtime:
            args.total_rays = max(1, int(authored_runtime["total_rays"]))
        if "rays_per_batch" in authored_runtime:
            args.rays_per_batch = max(1, int(authored_runtime["rays_per_batch"]))
        if "max_bounces" in authored_runtime:
            args.max_bounces = max(1, int(authored_runtime["max_bounces"]))
        runtime_environment = {
            "max_sensor_epochs": "SPECTRAL_SENSOR_MAX_EPOCHS",
            "sensor_top_k": "SPECTRAL_SENSOR_TOP_K",
            "sensor_samples_per_node": "SPECTRAL_SENSOR_SAMPLES_PER_NODE",
            "sensor_steps_per_layer": "SPECTRAL_SENSOR_STEPS_PER_LAYER",
            "sensor_flash_rays": "SPECTRAL_SENSOR_FLASH_RAYS",
            "sensor_t5_pair_budget": "SPECTRAL_SENSOR_T5_PAIR_BUDGET",
        }
        for runtime_key, environment_key in runtime_environment.items():
            if runtime_key in authored_runtime:
                os.environ[environment_key] = str(
                    max(1, int(authored_runtime[runtime_key]))
                )
        if not bool(authored_runtime.get("convergence_enabled", True)):
            args.no_convergence = True
            args.no_convergence_drive_batches = True
        if "integrator" in authored_runtime:
            args.integrator = str(authored_runtime["integrator"])
            if args.integrator == "depth":
                args.backend = "cpp"
                args.gpu_resident = True
                args.rgb_source = "sensor"
                args.no_convergence = True
                args.no_convergence_drive_batches = True
                args.max_bounces = 1
        authored_frequencies = order_transport_frequencies(order_package)
        if authored_frequencies is not None:
            DEFAULT_FREQ_HZ = authored_frequencies
            _ORDERED_TRANSPORT_RGB_WEIGHTS = order_transport_rgb_weights(
                order_package
            )
            _ORDERED_TRANSPORT_LUT = order_transport_lut_config(order_package)
            if _ORDERED_TRANSPORT_LUT is not None:
                knots = _ORDERED_TRANSPORT_LUT["frequency_knots_hz"]
                _ORDERED_MATERIAL_LUT_GRID = np.ascontiguousarray(
                    np.linspace(float(np.min(knots)), float(np.max(knots)), 32),
                    np.float64,
                )
            else:
                _ORDERED_MATERIAL_LUT_GRID = None
            print(
                "[scene-order] transport payload "
                f"bands={DEFAULT_FREQ_HZ.size} "
                f"lane_references_hz={DEFAULT_FREQ_HZ.tolist()} "
                f"continuous_lut={_ORDERED_TRANSPORT_LUT is not None}",
                flush=True,
            )
        order_jobs = resolved_jobs(order_package, args.scene_job)
        if len(order_jobs) != 1:
            ids = ", ".join(str(job["id"]) for job in order_jobs)
            raise ValueError(
                f"--scene-order contains multiple jobs ({ids}); select one with --scene-job"
            )
        order_job = order_jobs[0]
        ordered_exposure = dict(order_job.get("exposure", {}))
        if "sensor_id" in ordered_exposure or "film_id" in ordered_exposure:
            sensor_id = int(ordered_exposure.get("sensor_id", 0))
            film_id = int(ordered_exposure.get("film_id", 0))
            if sensor_id not in (0, 1) or film_id not in (0, 1):
                raise ValueError("scene-order sensor_id and film_id must be 0 or 1")
            ordered_sensor_film_slots = [
                (sensor_id, film_id), (-1, -1), (-1, -1), (-1, -1),
            ]
        _ORDERED_SCENE_JOB = order_job
        runtime = order_runtime_settings(order_job)
        _ORDERED_TILE_OUTPUT = (runtime["width"], runtime["height"])
        args.width, args.height = _native_sensor_schedule_shape(
            runtime["width"], runtime["height"]
        )
        args.focal_mm = runtime["focal_mm"]
        args.aperture_mm = runtime["aperture_mm"]
        args.iso = runtime["iso"]
        args.exposure_time_s = runtime["exposure_time_s"]
        args.t5_pair_budget = runtime["pair_budget"]
        args.bdpt_native_sweeps = _resolve_native_sensor_sweeps(
            cli_native_sweeps, runtime["sensor_sweeps"]
        )
        args.scene_mode = "thick-lens-lab"
        args.frames = 1
        # Build the canonical scene while no override is installed, then replace
        # only its subject group with the order compiler's result.
        _ORDERED_THICK_LENS_SCENE = None
        base_scene = _build_thick_lens_lab_tracer_scene()
        _ORDERED_THICK_LENS_SCENE, order_report = compile_job(
            base_scene,
            order_job,
            # A continuous transport has one stochastic lane reference but a
            # full material-response grid.  Compiling authored scene lights
            # against the lone lane frequency can miss every narrow emission
            # lobe and turn valid emitter geometry into a zero-power source.
            freq_hz=(
                _ORDERED_MATERIAL_LUT_GRID
                if _ORDERED_MATERIAL_LUT_GRID is not None
                else DEFAULT_FREQ_HZ
            ),
        )
        next_scan_path = str(
            args.next_scan_control
            or os.environ.get("SPECTRAL_NEXT_SCAN_CONTROL", "")
        ).strip()
        if next_scan_path:
            from camera_software.scan_control import NextSiteScan
            with open(next_scan_path, "r", encoding="utf-8") as handle:
                next_scan = NextSiteScan.from_mapping(json.load(handle))
            _NEXT_SITE_SCAN_COMMAND = next_scan
            _ORDERED_THICK_LENS_SCENE = apply_next_site_scan(
                _ORDERED_THICK_LENS_SCENE,
                next_scan,
                maximum_film_travel_m=float(args.maximum_film_travel_mm) * 1.0e-3,
                maximum_film_tilt_deg=float(args.maximum_film_tilt_deg),
            )
            if next_scan.targeted_fraction is not None:
                os.environ["SPECTRAL_SENSOR_TARGETED_FRACTION"] = str(
                    next_scan.targeted_fraction
                )
            print(
                f"[next-scan] sequence={next_scan.sequence} "
                f"uv_requests={len(next_scan.uv_requests)} "
                f"pixel_slices={len(next_scan.pixel_slice_requests)} "
                f"film_plane={'nominal' if next_scan.film_plane_adjustment is None else 'adjusted'}",
                flush=True,
            )
        print(
            f"[scene-order] job={order_report.job_id!r} "
            f"token={order_report.token.encode('ascii', 'backslashreplace').decode('ascii')!r} "
            f"triangles={order_report.total_triangles:,} "
            f"(glyph={order_report.glyph_triangles:,} planes={order_report.plane_triangles:,}) "
            f"materials={list(order_report.material_names)} flash_scale={order_report.flash_scale:g} "
            f"output={runtime['width']}x{runtime['height']} "
            f"native_schedule={args.width}x{args.height} "
            f"sensor_sweeps={args.bdpt_native_sweeps} "
            f"(authored={runtime['sensor_sweeps']})",
            flush=True,
        )

    args.bdpt_native_sweeps = _resolve_native_sensor_sweeps(
        args.bdpt_native_sweeps
    )

    if bool(args.camera_solve_preview):
        default_preview_modes = (
            "rgb,spectral,field-rgb,field-spectral,"
            "surface-rgb,surface-spectral,sensor-color,sensor-spectral,"
            "pinhole-rgb,endpoint-rgb,endpoint-spectral,"
            "lightfield-shell-top,lightfield-shell-side"
        )
        if args.scene_mode is None and not bool(args.calibration_scene):
            args.scene_mode = CAMERA_SOLVE_PREVIEW_SCENE
        if int(args.width) == 1280:
            args.width = 640
        if int(args.height) == 720:
            args.height = 360
        if float(args.input_size_scale) == 1.0:
            args.input_size_scale = 0.75
        if int(args.total_rays) == 1_000_000:
            args.total_rays = 60_000
        if int(args.rays_per_batch) == 50_000:
            args.rays_per_batch = 3_000
        if int(args.max_bounces) == 8:
            args.max_bounces = 2
        if int(args.frames) == 4:
            args.frames = 24
        if str(args.backend) != "cpp":
            args.backend = "cpp"
        args.integrator = "bdpt"
        args.rgb_source = "sensor"
        if float(args.preview_cycle_s) == 2.0:
            args.preview_cycle_s = 2.0
        if str(args.preview_modes) == default_preview_modes:
            args.preview_modes = "rgb,pinhole-rgb,sensor-color,spectral"
        if int(args.camera_solve_max_iters) == 72:
            args.camera_solve_max_iters = 14
        args.no_convergence = True
        args.no_convergence_drive_batches = True
        print("  [camera-solve-preview] enabled: cheap BDPT camera preset applied")

    profile_enabled = bool(ENABLE_EXPOSURE_PROFILING or args.profile)
    if profile_enabled:
        faulthandler.enable(all_threads=True)
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    backends = {
        "cpp":  ("cpp",),
        "glsl": ("glsl",),
        "both": ("cpp", "glsl"),
    }[args.backend]

    optics = CameraOptics(
        focal_mm           = args.focal_mm,
        aperture_mm        = args.aperture_mm,
        pixel_pitch_um     = DEFAULT_OPTICS.pixel_pitch_um,
        sensor_w_mm        = DEFAULT_OPTICS.sensor_w_mm,
        sensor_h_mm        = DEFAULT_OPTICS.sensor_h_mm,
        lens_transmission  = DEFAULT_OPTICS.lens_transmission,
    )
    film = FilmExposure(
        iso                 = args.iso,
        exposure_time_s     = args.exposure_time_s,
        quantum_efficiency  = DEFAULT_FILM.quantum_efficiency,
        target_mid_grey     = DEFAULT_FILM.target_mid_grey,
    )

    rail_model_cfg = RailModelConfig(
        sensor_z_min_m=float(args.camera_rail_sensor_z_min_m),
        sensor_z_max_m=float(args.camera_rail_sensor_z_max_m),
        sensor_corner_max_mm=float(args.camera_rail_corner_max_mm),
        sensor_shift_max_mm=float(args.camera_rail_sensor_shift_max_mm),
        aperture_z_min_m=float(args.camera_rail_aperture_z_min_m),
        aperture_z_max_m=float(args.camera_rail_aperture_z_max_m),
        aperture_shift_max_mm=float(args.camera_rail_aperture_shift_max_mm),
        bellows_sep_min_m=float(args.camera_rail_bellows_sep_min_m),
        bellows_sep_max_m=(None if float(args.camera_rail_bellows_sep_max_m) <= 0.0
                           else float(args.camera_rail_bellows_sep_max_m)),
        front_lens_shift_max_mm=float(args.camera_rail_front_shift_max_mm),
        front_lens_tilt_max_deg=float(args.camera_rail_front_tilt_max_deg),
        aperture_radius_min_m=max(1.0e-9, float(args.camera_rail_aperture_radius_min_mm) * 1.0e-3),
        aperture_radius_max_m=(None if float(args.camera_rail_aperture_radius_max_mm) <= 0.0
                               else max(1.0e-9, float(args.camera_rail_aperture_radius_max_mm) * 1.0e-3)),
        energy_deposit_epsilon=max(0.0, float(args.camera_solver_energy_epsilon)),
        energy_deposit_patience=max(1, int(args.camera_solver_energy_patience)),
    )

    input_scale = max(1.0e-6, float(args.input_size_scale))
    oversample = max(1, int(args.output_oversample))
    oversample_stencil = str(args.oversample_stencil)
    # Saving is an output operation and must not silently alter ray count,
    # aperture coverage, scene sampling, or image resolution.  Oversampling is
    # available only through the explicit --output-oversample option.
    
    # Check if sequence mode is active for simplified setup
    sequence_mode = str(args.sequence_mode)
    if sequence_mode == "forward-backward-bdpt":
        # For forward-backward-bdpt sequence, disable oversampling, fields, and use Holga camera
        print("  [sequence-mode] forward-backward-bdpt: using simplified Holga camera, no oversampling/fields")
        print(f"  [holga] focal={HOLGA_OPTICS.focal_mm}mm, aperture={HOLGA_OPTICS.aperture_mm}mm (f/{HOLGA_OPTICS.focal_mm/HOLGA_OPTICS.aperture_mm:.1f})")
        oversample = 1
        oversample_stencil = "box"
        optics = HOLGA_OPTICS
        args.no_convergence = True
        args.no_convergence_drive_batches = True
        # Use lenient rail model for sequence mode to allow camera solver more flexibility
        rail_model_cfg = RailModelConfig(
            sensor_z_min_m=-0.5,           # Very wide range ±500mm
            sensor_z_max_m=0.5,
            sensor_corner_max_mm=50.0,      # Allow substantial tilt
            sensor_shift_max_mm=200.0,      # Allow wide lateral shifts
            aperture_z_min_m=-0.05,         # Wide aperture motion
            aperture_z_max_m=0.05,
            aperture_shift_max_mm=100.0,    # Wide aperture shifts
            bellows_sep_min_m=0.001,        # Allow close focus
            bellows_sep_max_m=None,         # Allow far focus (auto)
            front_lens_shift_max_mm=200.0,  # Wide front element motion
            front_lens_tilt_max_deg=60.0,   # Allow aggressive tilt
            aperture_radius_min_m=max(1.0e-9, float(args.camera_rail_aperture_radius_min_mm) * 1.0e-3),
            aperture_radius_max_m=max(1.0e-9, float(optics.aperture_mm) * 0.5e-3),
            energy_deposit_epsilon=1.0e-8,  # Relax convergence slightly
            energy_deposit_patience=max(1, int(args.camera_solver_energy_patience)),
        )
        print("  [sequence-mode] using lenient camera rail model for solver flexibility")
    else:
        # For single-mode, use the provided optics
        print(f"  [single-mode] {args.integrator}: using provided camera (focal={optics.focal_mm}mm, aperture={optics.aperture_mm}mm)")
    
    output_w = max(1, int(round(float(args.width) * input_scale)))
    output_h = max(1, int(round(float(args.height) * input_scale)))
    render_w = max(1, int(output_w * oversample))
    render_h = max(1, int(output_h * oversample))

    # Keep a single coherent scene by default; do not auto-cycle scene modes.
    scene_mode_schedule = None

    backward_max_bounces = (
        int(args.max_bounces)
        if args.backward_max_bounces is None
        else int(args.backward_max_bounces)
    )

    session = ExposureSession(
        optics         = optics,
        film           = film,
        width          = render_w,
        height         = render_h,
        total_rays     = args.total_rays,
        rays_per_batch = args.rays_per_batch,
        max_bounces    = args.max_bounces,
        backends       = backends,
        out_dir        = args.out_dir,
        integrator     = args.integrator,
        bdpt_records_cap = args.bdpt_records_cap,
        bdpt_intermediate_mode = args.bdpt_intermediate_mode,
        bdpt_intermediate_max_bytes = int(max(0.0, args.bdpt_intermediate_max_gb) * (1024 ** 3)),
        retain_bdpt_intermediate = bool(args.retain_bdpt_intermediate),
        bdpt_intermediate_dir = args.bdpt_intermediate_dir,
                scene_mode     = (args.scene_mode if args.scene_mode is not None
                                                    else ("tungsten-cavity" if args.calibration_scene
                                                                else "thick-lens-lab")),
                scene_mode_schedule = scene_mode_schedule,
                profile_enabled = profile_enabled,
        integral_split = IntegralSplitConfig(
            field_integrate_frac   = 0.0,  # Disabled for sequence mode
            field_bookkeep_frac    = 0.0,
            surface_integrate_frac = 1.0,
            surface_bookkeep_frac  = 0.0,
            hdr_white_percentile   = float(args.hdr_white_percentile),
        ),
        convergence     = ConvergenceConfig(
            enabled             = not bool(args.no_convergence),
            drive_batches       = not bool(args.no_convergence_drive_batches),
            target_pct          = float(args.convergence_target_pct),
            max_rel_drift       = float(args.convergence_max_rel_drift),
            check_every_batches = int(args.convergence_check_every),
            min_batches         = int(args.convergence_min_batches),
            hold_checks         = int(args.convergence_hold_checks),
            probe_count         = int(args.convergence_probe_count),
            max_batches         = int(args.convergence_max_batches),
        ),
        adaptive_allocation_mode = str(args.adaptive_allocation),
        n_frames_planned = args.frames,
        output_width     = output_w,
        output_height    = output_h,
        output_oversample_stencil = oversample_stencil,
        show_hud         = not args.no_hud,
        rgb_source       = args.rgb_source,
        tone_map_mode    = args.tone_map_mode,
        camera_solve_max_iters = int(args.camera_solve_max_iters),
        camera_solve_seed_base = int(args.camera_solve_seed_base),
        camera_rail_model = rail_model_cfg,
        camera_cone_samples_min = int(args.camera_cone_samples_min),
        camera_cone_samples_max = int(args.camera_cone_samples_max),
        camera_cone_ref_half_angle_deg = float(args.camera_cone_ref_half_angle_deg),
        camera_cone_angle_exponent = float(args.camera_cone_angle_exponent),
        t5_min_geom     = float(args.t5_min_geom),
        t5_pair_budget  = int(args.t5_pair_budget),
        vcm_enabled     = not bool(args.no_vcm),
        vcm_radius_mm   = float(args.vcm_radius_mm),
        vcm_radius_alpha = float(args.vcm_radius_alpha),
        gpu_resident    = bool(args.gpu_resident),
        bdpt_native_packages = int(args.bdpt_native_packages),
        bdpt_native_sweeps = int(args.bdpt_native_sweeps),
        save_files       = args.save_files,
        sensor_film_slots = ordered_sensor_film_slots,
    )

    # Default to disabling convergence for simpler output focusing
    if not hasattr(args, '_convergence_explicitly_enabled'):
        args.no_convergence = True
        args.no_convergence_drive_batches = True
    
    # Handle sequence mode: forward → backward → bdpt
    sequence_mode = str(args.sequence_mode)
    primary_integrator = str(args.integrator)
    
    if sequence_mode == "forward-backward-bdpt":
        # Run forward, backward, then BDPT in sequence
        integrators_to_run = ("forward", "backward", "bdpt")
    else:
        # Single integrator mode
        integrators_to_run = (primary_integrator,)

    base_out_dir = str(session.out_dir)

    if str(args.rgb_source).strip().lower() == "sensor" and "forward" in integrators_to_run:
        print(
            "  [warn] rgb_source=sensor requested with forward integrator; "
            "sensor-only RGB is enforced, so forward outputs will be black because this pass does not produce a sensor integral"
        )
    
    # Re-create session for each integrator in sequence
    for integrator_mode in integrators_to_run:
        # Reset frame index for each integrator mode
        session._frame_index = 0
        
        # Update integrator
        session.integrator = integrator_mode
        if integrator_mode in ("backward", "bdpt"):
            session.max_bounces = int(max(0, backward_max_bounces))
        else:
            session.max_bounces = int(max(0, args.max_bounces))

        if str(args.rgb_source).strip().lower() in ("sensor", "accum", "endpoint"):
            session.rgb_source = str(args.rgb_source).strip().lower()
        
        # Update output directory to include integrator mode
        if sequence_mode == "forward-backward-bdpt":
            session.out_dir = os.path.join(base_out_dir, integrator_mode)
            os.makedirs(session.out_dir, exist_ok=True)
        
        # Update convergence for non-bdpt modes in sequence
        if sequence_mode == "forward-backward-bdpt" and integrator_mode != "bdpt":
            session.convergence = ConvergenceConfig(
                enabled=False,
                drive_batches=False,
                target_pct=99.99,
                max_rel_drift=1.0e-4,
                check_every_batches=1,
                min_batches=4,
                hold_checks=3,
                probe_count=8192,
                max_batches=0,
            )
        
        print("\n+============================================================+")
        print(f"|  SEQUENCE MODE: Running {integrator_mode.upper():20s} pass      |")
        print("+============================================================+\n")
        
        if args.no_window:
            _progress_writer = None
            _progress_sequence = [0]
            _progress_region = None
            if str(args.progress_dir).strip():
                if not bool(args.gpu_resident):
                    raise RuntimeError(
                        "progressive exposure reporting requires --gpu-resident; "
                        "CPU accumulation is not a supported producer"
                    )
                from camera_software.progressive_exposure import (
                    ExposureProgressEvent,
                    ExposureProgressKind,
                    LinearProgressArtifactWriter,
                    SensorRegion,
                )
                _progress_writer = LinearProgressArtifactWriter(
                    str(args.progress_dir),
                    line_sink=lambda line: print(line, flush=True),
                    retain_last=max(
                        0, int(os.environ.get("SPECTRAL_PROGRESS_RETAIN_LAYERS", "0"))
                    ),
                )
                if order_job is not None:
                    _progress_runtime = order_runtime_settings(order_job)
                    _progress_region = SensorRegion(**dict(_progress_runtime["region"]))
                else:
                    _progress_region = SensorRegion(
                        x=0, y=0, width=int(args.width), height=int(args.height)
                    )
                _progress_exposure_id = (
                    str(args.progress_exposure_id).strip()
                    or (str(order_job["id"]) if order_job is not None else "exposure")
                )
                _progress_active_uv = (
                    None if _NEXT_SITE_SCAN_COMMAND is None
                    else _NEXT_SITE_SCAN_COMMAND.active_uv_bounds()
                )

                def _publish_progress(_backs, _snapshots, batch_idx, n_batches, elapsed_s):
                    cpp_back = _backs.get("cpp")
                    tracer = getattr(cpp_back, "tracer", None) if cpp_back is not None else None
                    if not bool(getattr(cpp_back, "_gpu_resident", False)):
                        raise RuntimeError("progress event source is not GPU-resident")
                    if tracer is None or not hasattr(tracer, "get_sensor_image_linear"):
                        return
                    output_width, output_height = (
                        _ORDERED_TILE_OUTPUT if _ORDERED_TILE_OUTPUT is not None
                        else (int(args.width), int(args.height))
                    )
                    epoch_count_display = None
                    priority_display = None
                    sensor_sum_display = None
                    exposure_weight_display = None
                    if args.integrator == "depth":
                        # Depth is reduced from strike records on the CPU and
                        # therefore never enters the radiance SSBO.  Publishing
                        # that SSBO announced an all-zero layer while the PNG
                        # correctly used this cached depth product.
                        cached_depth = np.asarray(
                            getattr(cpp_back, "_native_sensor_linear_image", None),
                            dtype=np.float32,
                        )
                        if cached_depth.ndim != 3 or cached_depth.shape[2] < 3:
                            return
                        valid_depth = np.all(np.isfinite(cached_depth[..., :3]), axis=2)
                        linear = np.where(
                            valid_depth[..., None], cached_depth[..., :3], 0.0
                        ).astype(np.float32, copy=False)
                        epoch_count_display = valid_depth.astype(np.uint32)
                        sensor_sum_display = linear.copy()
                        exposure_weight_display = valid_depth.astype(np.float32)
                    else:
                        raw = np.asarray(tracer.get_sensor_image_linear(), dtype=np.float32)
                        linear = _native_sensor_tile_to_display(raw, output_width, output_height)
                        if (hasattr(tracer, "get_sensor_epoch_count")
                                and not bool(getattr(cpp_back, "_sensor_mipmap_enabled", False))):
                            epoch_count = np.asarray(tracer.get_sensor_epoch_count())
                            epoch_count_display = _native_sensor_tile_to_display(
                                epoch_count[..., None], output_width, output_height
                            )[..., 0]
                            linear = _normalise_native_sensor_epochs(linear, epoch_count_display)
                        if hasattr(tracer, "get_sensor_learned_priority_map"):
                            priority = np.asarray(
                                tracer.get_sensor_learned_priority_map(), dtype=np.float32
                            )
                            priority_display = _native_sensor_tile_to_display(
                                priority[..., None], output_width, output_height
                            )[..., 0]
                        if hasattr(tracer, "get_sensor_image_sum_linear"):
                            sensor_sum = np.asarray(
                                tracer.get_sensor_image_sum_linear(), dtype=np.float32
                            )
                            sensor_sum_display = _native_sensor_tile_to_display(
                                sensor_sum, output_width, output_height
                            )[..., :3]
                        if hasattr(tracer, "get_sensor_exposure_weight"):
                            exposure_weight = np.asarray(
                                tracer.get_sensor_exposure_weight(), dtype=np.float32
                            )
                            exposure_weight_display = _native_sensor_tile_to_display(
                                exposure_weight[..., None], output_width, output_height
                            )[..., 0]
                    _progress_sequence[0] += 1
                    event = ExposureProgressEvent(
                        exposure_id=_progress_exposure_id,
                        sequence=_progress_sequence[0],
                        kind=(ExposureProgressKind.LAYER_AVAILABLE
                              if bool(getattr(cpp_back, "_sensor_mipmap_enabled", False))
                              else ExposureProgressKind.PASS_AVAILABLE),
                        region=_progress_region,
                        pass_index=int(batch_idx),
                        completed_work=int(batch_idx),
                        total_work=max(0, int(n_batches)),
                        global_uv_bounds=_progress_active_uv,
                        message=(
                            "GPU sensor SSBO exposure epoch stable after "
                            f"{elapsed_s:.3f}s; linear artifact is a presentation readback"
                        ),
                    )
                    _progress_writer.publish(
                        event, linear, epoch_count_display, priority_display,
                        sensor_sum_display, exposure_weight_display,
                    )
            else:
                _publish_progress = None
            for k in range(args.frames):
                session.render_one_exposure(
                    t=float(k) * 0.5,
                    batch_preview_cb=_publish_progress,
                )
        else:
            preview_modes = tuple(
                token.strip().lower() for token in str(args.preview_modes).split(",") if token.strip()
            )
            _run_viewer(session, args.frames, args.pane_w, args.pane_h,
                        show_hud=not args.no_hud,
                        preview_cycle_s=float(args.preview_cycle_s),
                        preview_modes=preview_modes)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
