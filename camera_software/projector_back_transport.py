"""Physical projector-back launch through the common optical pipeline.

This module owns no tracer and performs no transport.  It lowers the camera
designer's large projector source plane into the existing native
``RayTracer.submit_rays`` ABI, including fixed-lane complex phase, then leaves
the exact reciprocal lens and any authored T4 arenas to the normal pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

import numpy as np

from camera_designer.ray_order import RayOrder
from camera_designer.compound_optics import CompoundLens
from camera_designer.emitter_profile import CoherenceModel
from .complex_optical_operators import (
    ComplexOpticalOperator,
    ComplexOperatorStateBlock,
    ComplexSourceMode,
    TransverseBasis,
    install_source_mode_block,
)
from .surface_scan_preview import current_wgl_handles, restore_wgl_context


@dataclass(frozen=True)
class ProjectorBackLaunch:
    origins: np.ndarray
    directions: np.ndarray
    amplitudes: np.ndarray
    ray_tags: np.ndarray
    source_mode_indices: np.ndarray
    source_state: dict[str, Any]
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
    source_indices = order.ray_source_indices(max(1, int(n_rays)))

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

    # Expand partially polarized source records into independent coherent
    # modes and attach their Jones/basis state through stable ray tags. Scalar
    # spectral phase remains in amplitudes; relative s/p phase lives here.
    expanded_origins: list[np.ndarray] = []
    expanded_directions: list[np.ndarray] = []
    expanded_amplitudes: list[np.ndarray] = []
    state = ComplexOperatorStateBlock()
    operator_id = state.add_operator(ComplexOpticalOperator())
    tag_values: list[int] = []
    mode_indices: list[int] = []
    coherent_ids: dict[tuple[str, str, int], int] = {}
    tag_prefix = (
        0x50524A0000000000
        | ((int(seed) & 0xFFFF) << 32)
    )
    for ray_index, source_index in enumerate(source_indices):
        source = order.sources[int(source_index)]
        azimuth = math.atan2(
            float(source.uv[1])-0.5,
            float(source.uv[0])-0.5,
        )
        modes = source.polarization.coherent_mode_decomposition(azimuth)
        direction = directions[ray_index]
        basis_id = state.add_basis(TransverseBasis.from_direction(
            direction,
            reference=(0.0, 0.0, 1.0),
        ))
        for component_index, (power_weight, jones) in enumerate(modes):
            expanded_index = len(expanded_origins)
            tag = (tag_prefix+expanded_index+1) & 0xFFFFFFFFFFFFFFFF
            if source.phase_state.model is CoherenceModel.COHERENT:
                coherence_key = (
                    str(source.spec_label),
                    str(source.component_label),
                    component_index,
                )
                if coherence_key not in coherent_ids:
                    coherent_ids[coherence_key] = (
                        0x5052430000000000+len(coherent_ids)+1
                    )
                coherence = coherent_ids[coherence_key]
            else:
                coherence = (
                    0x5052490000000000+expanded_index+1
                ) & 0xFFFFFFFFFFFFFFFF
            mode_index = state.add_source_mode(ComplexSourceMode(
                jones=np.asarray(jones, np.complex128),
                power_weight=1.0,
                coherence_id=coherence,
                basis_id=basis_id,
                operator_id=operator_id,
            ))
            expanded_origins.append(origins[ray_index])
            expanded_directions.append(direction)
            expanded_amplitudes.append(
                amplitudes[ray_index]*math.sqrt(max(0.0, power_weight))
            )
            tag_values.append(tag)
            mode_indices.append(mode_index)
    origins = np.ascontiguousarray(expanded_origins, np.float64)
    directions = np.ascontiguousarray(expanded_directions, np.float64)
    amplitudes = np.ascontiguousarray(expanded_amplitudes, np.complex128)
    ray_tags = np.ascontiguousarray(tag_values, np.uint64)
    source_mode_indices = np.ascontiguousarray(mode_indices, np.uint32)
    source_state = state.freeze()

    return ProjectorBackLaunch(
        origins=origins,
        directions=directions,
        amplitudes=amplitudes,
        ray_tags=ray_tags,
        source_mode_indices=source_mode_indices,
        source_state=source_state,
        profile_contract={
            "source": "camera.projector-back-port",
            "emissive_back": projector.source_contract(
                wavelengths,
                plane_radius_m=float(spec.radius),
            ),
            "spatial_samples": int(spec.spatial_samples),
            "plane_radius_m": float(spec.radius),
            "exit_pupil_center_m": pupil_center.tolist(),
            "exit_pupil_radius_m": float(pupil_radius),
            "pupil_sampling": "center-site-first-light-source-optics-boundary",
            "lane_count": int(len(wavelengths)),
            "phase_transport": "fixed-lane-complex",
            "jones_transport": "pipeline-source-mode-block",
            "source_mode_count": int(len(source_state["source_modes"])),
            "source_binding_count": int(len(ray_tags)),
            "texture_shape": (
                list(profile.texture.data.shape)
                if profile.texture is not None else None
            ),
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
    install_source_mode_block(
        tracer,
        launch.ray_tags,
        launch.source_state,
        mode_indices=launch.source_mode_indices,
    )
    submit(
        launch.origins,
        launch.directions,
        launch.amplitudes,
        color_flags=np.full(
            launch.ray_count, 1 << 6, dtype=np.uint8
        ),
        tags=launch.ray_tags,
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
        required = (
            "submit_rays", "in_flight_count",
            "configure_complex_source_modes",
        )
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
