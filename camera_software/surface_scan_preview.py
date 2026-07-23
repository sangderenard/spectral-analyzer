"""Interactive one-ray-per-pixel camera preview orchestration.

The native ray pipeline owns transport and the double-buffered OpenGL output.
This module only owns invalidation and submission policy.  In particular it
does not read pixels back, write image files, or turn the scan into a reduced
path-tracing exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SurfaceScanPose:
    sensor_center: tuple[float, float, float]
    aperture_center: tuple[float, float, float]
    sensor_half_width: float
    sensor_half_height: float
    resolution: int = 256
    sensor_right: tuple[float, float, float] = (0.0, 1.0, 0.0)
    sensor_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    aperture_right: tuple[float, float, float] = (0.0, 1.0, 0.0)
    aperture_up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def validated(self) -> "SurfaceScanPose":
        if self.resolution < 1:
            raise ValueError("surface-scan resolution must be positive")
        if self.sensor_half_width <= 0.0 or self.sensor_half_height <= 0.0:
            raise ValueError("surface-scan sensor dimensions must be positive")
        return self


def current_wgl_handles() -> tuple[int, int]:
    """Return the current display (HGLRC, HDC), or zeros off Windows."""

    if os.name != "nt":
        return 0, 0
    try:
        import ctypes

        get_context = ctypes.windll.opengl32.wglGetCurrentContext
        get_dc = ctypes.windll.opengl32.wglGetCurrentDC
        get_context.restype = ctypes.c_void_p
        get_dc.restype = ctypes.c_void_p
        return int(get_context() or 0), int(get_dc() or 0)
    except Exception:
        return 0, 0


def restore_wgl_context(hglrc: int, hdc: int) -> bool:
    """Restore a display context after native shared-context initialization."""

    if os.name != "nt" or not hglrc or not hdc:
        return False
    try:
        import ctypes

        make_current = ctypes.windll.opengl32.wglMakeCurrent
        make_current.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        make_current.restype = ctypes.c_int
        return bool(make_current(ctypes.c_void_p(hdc), ctypes.c_void_p(hglrc)))
    except Exception:
        return False


class SurfaceScanPreview:
    """Drive native deterministic scans without touching the display hot path."""

    def __init__(
        self,
        tracer: Any,
        pose: SurfaceScanPose,
        *,
        shader_dir: str,
        display_hglrc: int = 0,
        display_hdc: int = 0,
    ) -> None:
        required = (
            "configure_sensor_image", "configure_sensor_pose", "ensure_pipeline",
            "submit_surface_scan", "surface_scan_texture_info", "in_flight_count",
        )
        missing = [name for name in required if not callable(getattr(tracer, name, None))]
        if missing:
            raise RuntimeError(
                "loaded _spectral_kernels needs rebuilding for surface scan: "
                + ", ".join(missing)
            )
        self.tracer = tracer
        self.shader_dir = os.path.abspath(shader_dir)
        self._display_hglrc = int(display_hglrc)
        self._display_hdc = int(display_hdc)
        self._pose: SurfaceScanPose | None = None
        self._requested = False
        self._submitted = False
        self._last_generation = 0
        self._last_info: dict[str, Any] = {}

        if display_hdc and callable(getattr(tracer, "set_gl_display_hdc", None)):
            tracer.set_gl_display_hdc(int(display_hdc))
        if display_hglrc and callable(getattr(tracer, "set_gl_display_hglrc", None)):
            tracer.set_gl_display_hglrc(int(display_hglrc))

        # Surface scan is deliberately geometric.  Authored wave arenas remain
        # part of the full engine but must not make an editing preview stochastic
        # or iterative.
        clear_contexts = getattr(tracer, "clear_scale_contexts", None)
        if callable(clear_contexts):
            clear_contexts()
        skip_readback = getattr(tracer, "set_gpu_skip_record_readback", None)
        if callable(skip_readback):
            skip_readback(True)
        self.configure(pose)
        # Pixel-ray raster mode remains one geometric ray per pixel. T3 uses
        # the central spectral lane for deterministic glass refraction while
        # retaining all material bands for the terminal surface colour.
        tracer.ensure_pipeline(
            max_children=1,
            seed=42,
            min_amplitude=1.0e-8,
            use_gpu_compute=True,
            gpu_all_stages=True,
            shader_dir=self.shader_dir,
        )
        restore_wgl_context(self._display_hglrc, self._display_hdc)
        self.invalidate()

    @property
    def texture_info(self) -> dict[str, Any]:
        return dict(self._last_info)

    @property
    def busy(self) -> bool:
        return bool(int(self.tracer.in_flight_count()))

    def configure(self, pose: SurfaceScanPose) -> None:
        pose = pose.validated()
        self._pose = pose
        sensor = np.asarray(pose.sensor_center, np.float64)
        aperture = np.asarray(pose.aperture_center, np.float64)
        sensor_right = np.asarray(pose.sensor_right, np.float64)
        sensor_up = np.asarray(pose.sensor_up, np.float64)
        aperture_right = np.asarray(pose.aperture_right, np.float64)
        aperture_up = np.asarray(pose.aperture_up, np.float64)
        self.tracer.configure_sensor_image(
            float(sensor[0]),
            float(pose.sensor_half_width),
            float(pose.sensor_half_height),
            int(pose.resolution),
            1.0e-5,
            float(aperture[0]),
            0.0,
            float(aperture[1]),
            float(aperture[2]),
            0,
        )
        self.tracer.configure_sensor_pose(
            sensor, sensor_right, sensor_up,
            aperture, aperture_right, aperture_up,
        )
        self.invalidate()

    def invalidate(self) -> None:
        """Request one new complete generation when the pipeline becomes idle."""

        self._requested = True

    def poll(self) -> dict[str, Any]:
        """Advance submission/publication without blocking the UI thread."""

        info = dict(self.tracer.surface_scan_texture_info() or {})
        restore_wgl_context(self._display_hglrc, self._display_hdc)
        generation = int(info.get("generation", 0))
        if generation > self._last_generation and int(info.get("texture_id", 0)) > 0:
            self._last_generation = generation
            self._last_info = info
            self._submitted = False

        if self._requested and not self.busy and not self._submitted:
            self.tracer.submit_surface_scan(min_amplitude=1.0e-8, seed=42)
            restore_wgl_context(self._display_hglrc, self._display_hdc)
            self._requested = False
            self._submitted = True
        return self.texture_info

    def close(self) -> None:
        # ray_pipeline_destroy historically released whichever WGL context was
        # current on the calling thread while deleting its hidden context. Keep
        # the host context stable across shutdown even when an older native
        # module is loaded; newer native modules avoid disturbing it as well.
        host_hglrc, host_hdc = current_wgl_handles()
        stop = getattr(self.tracer, "stop_pipeline", None)
        try:
            if callable(stop):
                stop()
        finally:
            restore_wgl_context(
                host_hglrc or self._display_hglrc,
                host_hdc or self._display_hdc,
            )
        self._requested = False
        self._submitted = False


__all__ = [
    "SurfaceScanPose", "SurfaceScanPreview", "current_wgl_handles",
    "restore_wgl_context",
]
