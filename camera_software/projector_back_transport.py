"""Physical projector-back launch through the common optical pipeline.

This module owns no tracer and performs no transport.  It lowers the camera
designer's large projector source plane into the existing native
``RayTracer.submit_rays`` ABI, including fixed-lane complex phase, then leaves
the exact reciprocal lens and any authored T4 arenas to the normal pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from camera_designer.ray_order import RayOrder
from camera_designer.compound_optics import CompoundLens
from .surface_scan_preview import current_wgl_handles, restore_wgl_context


@dataclass(frozen=True)
class ProjectorBackLaunch:
    origins: np.ndarray
    directions: np.ndarray
    amplitudes: np.ndarray
    profile_contract: dict[str, Any]

    @property
    def ray_count(self) -> int:
        return int(self.origins.shape[0])


def prepare_projector_back_launch(
    preset: Any,
    *,
    n_rays: int = 65_536,
    seed: int = 42,
    canonical_optical_x: bool = True,
) -> ProjectorBackLaunch:
    """Prepare a distributed complex launch from an enabled projector back."""
    projector = getattr(preset, "projector_back", None)
    if projector is None or not bool(projector.enabled):
        raise ValueError("camera preset has no enabled projector back")
    wavelengths = np.asarray(preset.wavelengths, np.float64)
    spec = projector.as_emitter_spec(
        sensor_z_pos=float(preset.sensor.z_pos),
        sensor_radius=float(preset.sensor.r_max),
        wavelengths_um=wavelengths,
    )
    profile = spec.resolve_profile()
    lens = CompoundLens.from_preset(
        preset,
        wavelengths_um=wavelengths,
        axial_scale=1.0,
    )
    pupil_z, pupil_radius = lens.exit_pupil
    pupil_center = np.asarray([0.0, 0.0, float(pupil_z)], np.float64)
    order = RayOrder.from_emitter_specs(
        [spec],
        wavelengths_um=wavelengths,
        aim_point=pupil_center,
    )
    origins, directions, amplitudes = order.bake_pipeline_submission(
        max(1, int(n_rays)),
        seed=int(seed),
    )

    # Leave the emitting triangle robustly in its authored forward direction.
    origins = origins + directions * 2.0e-4
    if canonical_optical_x:
        # Designer (x,y,z) -> native exact-camera (-z,x,y).
        origins = np.ascontiguousarray(
            np.column_stack((-origins[:, 2], origins[:, 0], origins[:, 1])),
            np.float64,
        )
        directions = np.ascontiguousarray(
            np.column_stack(
                (-directions[:, 2], directions[:, 0], directions[:, 1])
            ),
            np.float64,
        )

    return ProjectorBackLaunch(
        origins=np.ascontiguousarray(origins, np.float64),
        directions=np.ascontiguousarray(directions, np.float64),
        amplitudes=np.ascontiguousarray(amplitudes, np.complex128),
        profile_contract={
            "source": "camera.projector-back-port",
            "profile": profile.to_dict(),
            "spatial_samples": int(spec.spatial_samples),
            "plane_radius_m": float(spec.radius),
            "exit_pupil_center_m": pupil_center.tolist(),
            "exit_pupil_radius_m": float(pupil_radius),
            "pupil_sampling": "center-site-first-light-source-optics-boundary",
            "lane_count": int(len(wavelengths)),
            "phase_transport": "fixed-lane-complex",
            "continuous_frequency": "requires-continuous-source-sidecar",
        },
    )


def submit_projector_back(
    tracer: Any,
    launch: ProjectorBackLaunch,
    *,
    shader_dir: str = "",
    max_bounces: int = 8,
    min_amplitude: float = 1.0e-8,
    seed: int = 42,
) -> int:
    """Submit the launch without acquiring tracer or pipeline ownership."""
    submit = getattr(tracer, "submit_rays", None)
    if not callable(submit):
        raise TypeError("optical tracer does not expose submit_rays")
    submit(
        launch.origins,
        launch.directions,
        launch.amplitudes,
        color_flags=np.full(
            launch.ray_count, 1 << 6, dtype=np.uint8
        ),
        max_bounces=int(max_bounces),
        min_amplitude=float(min_amplitude),
        max_children=2,
        seed=int(seed),
        use_gpu_compute=True,
        gpu_all_stages=True,
        shader_dir=str(shader_dir),
    )
    return launch.ray_count


class ProjectorBackPreview:
    """Schedule reciprocal projection on a backend-owned native tracer.

    This owns invalidation and submission policy only. Ray, exact-lens and
    wave-region transport remain entirely in the canonical pipeline.
    """

    def __init__(
        self,
        tracer: Any,
        preset: Any,
        *,
        shader_dir: str,
        ray_count: int = 65_536,
        seed: int = 42,
        display_hglrc: int = 0,
        display_hdc: int = 0,
    ) -> None:
        required = ("submit_rays", "in_flight_count")
        missing = [
            name for name in required
            if not callable(getattr(tracer, name, None))
        ]
        if missing:
            raise RuntimeError(
                "loaded _spectral_kernels needs rebuilding for projector back: "
                + ", ".join(missing)
            )
        self.tracer = tracer
        self.shader_dir = os.path.abspath(shader_dir)
        self._display_hglrc = int(display_hglrc)
        self._display_hdc = int(display_hdc)
        self._seed = int(seed)
        self._launch = prepare_projector_back_launch(
            preset,
            n_rays=max(1, int(ray_count)),
            seed=self._seed,
        )
        self._requested = True
        self._submitted = False
        self._generation = 0

        if display_hdc and callable(getattr(tracer, "set_gl_display_hdc", None)):
            tracer.set_gl_display_hdc(int(display_hdc))
        if display_hglrc and callable(
            getattr(tracer, "set_gl_display_hglrc", None)
        ):
            tracer.set_gl_display_hglrc(int(display_hglrc))
        skip_readback = getattr(tracer, "set_gpu_skip_record_readback", None)
        if callable(skip_readback):
            skip_readback(True)

    @property
    def profile_contract(self) -> dict[str, Any]:
        return dict(self._launch.profile_contract)

    @property
    def busy(self) -> bool:
        return bool(int(self.tracer.in_flight_count()))

    def invalidate(self) -> None:
        self._requested = True

    def poll(self) -> dict[str, Any]:
        busy = self.busy
        if self._submitted and not busy:
            self._generation += 1
            self._submitted = False
        if self._requested and not busy and not self._submitted:
            submit_projector_back(
                self.tracer,
                self._launch,
                shader_dir=self.shader_dir,
                seed=self._seed,
            )
            restore_wgl_context(self._display_hglrc, self._display_hdc)
            self._requested = False
            self._submitted = True
            busy = self.busy
        return {
            "generation": int(self._generation),
            "ray_count": int(self._launch.ray_count),
            "profile_contract": self.profile_contract,
        }

    def close(self) -> None:
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
    "ProjectorBackLaunch",
    "ProjectorBackPreview",
    "prepare_projector_back_launch",
    "submit_projector_back",
]
