"""Canonical optical-engine execution service for interactive and final clients.

UI stations submit ordinary engine requests.  This service alone owns native
tracers, transport modes, GL sharing, pipeline lifetime and preview publication.
It is attached to :class:`pluck_render_graph.OpticalEnginePillar`; it is not a
camera-station helper.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import threading
from typing import Any, Mapping

import numpy as np

from .camera_designer_bridge import camera_manifest_to_designer_preset
from .gpu_preview import (
    OpenGLShareGroup,
    PreviewProductRegistry,
    RayPipelinePreviewBridge,
)
from .optical_bench import resolve_bench_camera_manifest
from .surface_scan_preview import SurfaceScanPose, SurfaceScanPreview


@dataclass
class OpticalEngineJob:
    request_revision: int
    backend_mode: str
    preview: SurfaceScanPreview
    publisher: RayPipelinePreviewBridge
    transport_installation: Any | None = None
    completed_generation: int = 0
    closed: bool = False


class OpticalEngineBackend:
    """Persistent execution boundary for the repository's optical backend."""

    SUPPORTED_MODES = frozenset({"fast_preview"})

    def __init__(
        self,
        products: PreviewProductRegistry,
        share_group: OpenGLShareGroup,
        *,
        shader_dir: str | None = None,
    ) -> None:
        if share_group.context_handle <= 0 or share_group.device_context_handle <= 0:
            raise RuntimeError("optical backend requires a live shared OpenGL context")
        self.products = products
        self.share_group = share_group
        self.shader_dir = os.path.abspath(
            shader_dir
            or os.path.join(os.path.dirname(os.path.dirname(__file__)), "csrc", "shaders")
        )
        self._lock = threading.RLock()
        self._active: OpticalEngineJob | None = None
        self._submitted = 0

    @staticmethod
    def _camera_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        scene = dict(payload.get("scene_manifest", {}))
        apparatus = list(scene.get("apparatus", ()))
        if not apparatus:
            raise ValueError("optical request has no canonical camera apparatus")
        camera = dict(apparatus[0]).get("camera_manifest")
        if not isinstance(camera, Mapping):
            raise ValueError("optical apparatus has no camera_manifest")
        return camera

    @staticmethod
    def _confinement_box(preset: Any, scene_object: Mapping[str, Any] | None):
        z_values = [0.0]
        z_values.extend(
            float(element.z_vertex) for element in preset.lens_group.elements
        )
        z_values.append(float(preset.sensor.z_pos))
        z_values.append(float(preset.aperture_stop.z_pos))
        if scene_object:
            position = scene_object.get("position", scene_object.get("pos", (0, 0, 0)))
            if len(position) >= 3:
                z_values.append(float(position[2]))
        radius = max(
            [0.025]
            + [
                float(getattr(element.surface, "r_max", 0.025))
                for element in preset.lens_group.elements
            ]
        )
        half_xy = max(0.15, radius * 4.0)
        return (
            np.asarray(
                [-half_xy, -half_xy, min(z_values) - 0.10], np.float32
            ),
            np.asarray(
                [half_xy, half_xy, max(z_values) + 0.30], np.float32
            ),
        )

    @staticmethod
    def _pose(payload: Mapping[str, Any], preset: Any) -> SurfaceScanPose:
        value = dict(payload.get("sensor_pose", {}))
        sensor_z = tuple(value.get(
            "sensor_center", (0.0, 0.0, float(preset.sensor.z_pos))
        ))
        aperture_z = tuple(value.get(
            "aperture_center", (0.0, 0.0, float(preset.aperture_stop.z_pos))
        ))
        return SurfaceScanPose(
            sensor_center=(-sensor_z[2], sensor_z[0], sensor_z[1]),
            aperture_center=(-aperture_z[2], aperture_z[0], aperture_z[1]),
            sensor_half_width=float(
                value.get("sensor_half_width", preset.sensor.r_max)
            ),
            sensor_half_height=float(
                value.get("sensor_half_height", preset.sensor.r_max)
            ),
            resolution=int(value.get("resolution", 256)),
            sensor_right=(0.0, 1.0, 0.0),
            sensor_up=(0.0, 0.0, 1.0),
            aperture_right=(0.0, 1.0, 0.0),
            aperture_up=(0.0, 0.0, 1.0),
        ).validated()

    def submit(self, payload: Mapping[str, Any]) -> OpticalEngineJob:
        """Create or replace one backend-owned transport job."""
        mode = str(payload.get("backend_mode", "")).strip()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"unsupported optical backend mode {mode!r}; "
                f"expected one of {sorted(self.SUPPORTED_MODES)}"
            )
        manifest = resolve_bench_camera_manifest(self._camera_mapping(payload))
        preset = camera_manifest_to_designer_preset(manifest)
        scene_object = dict(payload.get("scene_object", {})) or None
        lights = [dict(value) for value in payload.get("lights", ())]
        revision = int(payload.get("revision", 0))
        print(
            "[optical-backend] accepted mode=fast_preview "
            f"owner=optical-engine revision={revision}",
            flush=True,
        )

        # Import at the engine boundary so UI modules cannot accidentally gain
        # construction authority by importing this service.
        from camera_designer.scene_builder import build_tracer

        tracer, _contexts = build_tracer(
            preset,
            lights,
            scene_object,
            wavelengths_um=preset.wavelengths,
            confinement_box=self._confinement_box(preset, scene_object),
        )
        installation = None
        transport_mode = str(payload.get("transport_mode", "ray")).strip().lower()
        authored_contexts = [
            dict(value) for value in payload.get("contexts", ())
            if str(value.get("transport", "ray")).strip().lower() == "wave"
        ]
        if transport_mode in {"wave", "mixed"} and authored_contexts:
            from camera_designer.compound_optics import CompoundLens
            from .optical_transport_graph import (
                compile_compound_lens_graph,
                install_optical_graph,
            )

            wave_regions = []
            for index, context in enumerate(authored_contexts):
                key = str(context.pop("key", f"bench.wave-{index}"))
                context.pop("transport", None)
                wave_regions.append((key, context))
            graph = compile_compound_lens_graph(
                CompoundLens.from_preset(
                    preset,
                    wavelengths_um=preset.wavelengths,
                    axial_scale=-1.0,
                ),
                lane_count=len(preset.wavelengths),
                wave_regions=tuple(wave_regions),
            )
            installation = install_optical_graph(
                tracer,
                graph,
                exact_t2_registered=True,
            )
            print(
                "[optical-backend] installed graph transport "
                f"t2=exact t4={len(installation.wave_arenas)} "
                f"backend=angular-spectrum-fft",
                flush=True,
            )
        preview = SurfaceScanPreview(
            tracer,
            self._pose(payload, preset),
            shader_dir=self.shader_dir,
            display_hglrc=self.share_group.context_handle,
            display_hdc=self.share_group.device_context_handle,
        )
        job = OpticalEngineJob(
            request_revision=revision,
            backend_mode=mode,
            preview=preview,
            publisher=RayPipelinePreviewBridge(self.products, tracer),
            transport_installation=installation,
        )
        with self._lock:
            previous = self._active
            self._active = job
            self._submitted += 1
        if previous is not None:
            previous.closed = True
            previous.preview.close()
        print(
            "[optical-backend] mode=fast_preview owner=optical-engine "
            f"revision={job.request_revision} resolution={preview._pose.resolution}",
            flush=True,
        )
        return job

    def poll(self, job: OpticalEngineJob) -> dict[str, Any]:
        if job.closed:
            return {"complete": True, "superseded": True}
        info = job.preview.poll()
        job.publisher.poll()
        generation = int(info.get("generation", 0))
        if generation > 0:
            job.completed_generation = generation
        return {
            "complete": generation > 0,
            "generation": generation,
            "backend_mode": job.backend_mode,
            "owner": "optical-engine",
            "transport_graph": (
                None if job.transport_installation is None
                else job.transport_installation.contract()
            ),
        }

    def shutdown(self) -> None:
        with self._lock:
            active = self._active
            self._active = None
        if active is not None and not active.closed:
            active.closed = True
            active.preview.close()

    def describe(self) -> dict[str, Any]:
        with self._lock:
            active = self._active
            return {
                "owner": "optical-engine",
                "supported_modes": tuple(sorted(self.SUPPORTED_MODES)),
                "submitted": self._submitted,
                "active_revision": (
                    None if active is None else active.request_revision
                ),
            }


__all__ = ["OpticalEngineBackend", "OpticalEngineJob"]
