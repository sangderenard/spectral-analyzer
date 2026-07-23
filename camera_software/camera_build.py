"""Authoritative camera-build artifact shared by camera software and renderers.

The artifact deliberately keeps physical camera semantics beside the triangle
proxy used by the BVH.  Render clients must not reconstruct a thin-lens camera
from triangle bounds after the camera software has already solved and built the
compound assembly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CameraOpticalProvenance:
    build_id: str
    optical_model: str
    intersection_model: str
    dispersion_model: str
    wavelength_count: int
    diffraction_model: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "build_id": self.build_id,
            "optical_model": self.optical_model,
            "intersection_model": self.intersection_model,
            "dispersion_model": self.dispersion_model,
            "wavelength_count": self.wavelength_count,
            "diffraction_model": self.diffraction_model,
        }


@dataclass
class RebuiltCameraArtifact:
    """One complete camera rebuild consumed without optical reinterpretation."""

    scene_config: Any
    lens_assembly: Any
    lens_surface_groups: tuple[tuple[np.ndarray, np.ndarray, float, float], ...]
    scene_lenses: tuple[Any, ...]
    wavelengths_nm: np.ndarray
    sensor_center: np.ndarray
    sensor_right: np.ndarray
    sensor_up: np.ndarray
    machined_sensor_center: np.ndarray
    machined_sensor_right: np.ndarray
    machined_sensor_up: np.ndarray
    manifest: dict[str, Any]
    provenance: CameraOpticalProvenance

    @classmethod
    def create(
        cls,
        *,
        scene_config: Any,
        lens_assembly: Any,
        lens_surface_groups: Iterable[tuple[np.ndarray, np.ndarray, float, float]],
        scene_lenses: Sequence[Any],
        wavelengths_nm: Sequence[float],
        sensor_center: Sequence[float],
        sensor_right: Sequence[float] = (0.0, 1.0, 0.0),
        sensor_up: Sequence[float] = (0.0, 0.0, 1.0),
        diffraction_model: str = "disabled",
        manifest: Mapping[str, Any] | None = None,
    ) -> "RebuiltCameraArtifact":
        groups = tuple(
            (
                np.ascontiguousarray(front, np.int32).reshape(-1),
                np.ascontiguousarray(back, np.int32).reshape(-1),
                float(radius_front),
                float(radius_back),
            )
            for front, back, radius_front, radius_back in lens_surface_groups
        )
        if not groups or not any(front.size or back.size for front, back, _, _ in groups):
            raise ValueError("rebuilt camera requires parametric lens surface proxy groups")
        wavelengths = np.ascontiguousarray(wavelengths_nm, np.float64).reshape(-1)
        if wavelengths.size == 0:
            raise ValueError("rebuilt camera requires at least one wavelength")
        payload = np.ascontiguousarray(lens_assembly.build_parametric_payload(), np.float32)
        if payload.size < 8 or float(payload[0]) != 14949.0:
            raise ValueError("rebuilt camera lens assembly is not exact-parametric")
        spectral_count = int(payload[5])
        if spectral_count != wavelengths.size:
            raise ValueError(
                "parametric lens spectral table does not match the camera wavelength table"
            )
        digest = hashlib.sha256()
        digest.update(payload.tobytes())
        digest.update(wavelengths.tobytes())
        digest.update(np.asarray(sensor_center, np.float64).reshape(3).tobytes())
        build_id = digest.hexdigest()[:16]
        provenance = CameraOpticalProvenance(
            build_id=build_id,
            optical_model="compound_lens",
            intersection_model="exact_parametric_conic",
            dispersion_model="sampled_wavelength_sellmeier",
            wavelength_count=int(wavelengths.size),
            diffraction_model=str(diffraction_model),
        )
        center = np.ascontiguousarray(sensor_center, np.float64).reshape(3)
        right = np.ascontiguousarray(sensor_right, np.float64).reshape(3)
        up = np.ascontiguousarray(sensor_up, np.float64).reshape(3)
        return cls(
            scene_config=scene_config,
            lens_assembly=lens_assembly,
            lens_surface_groups=groups,
            scene_lenses=tuple(scene_lenses),
            wavelengths_nm=wavelengths,
            sensor_center=center.copy(),
            sensor_right=right.copy(),
            sensor_up=up.copy(),
            machined_sensor_center=center.copy(),
            machined_sensor_right=right.copy(),
            machined_sensor_up=up.copy(),
            manifest=dict(manifest or {}),
            provenance=provenance,
        )

    def register_exact_transport(self, tracer: Any, tri_vertices: np.ndarray) -> None:
        """Register the exact compound transform on its BVH proxy surfaces."""
        vertices = np.ascontiguousarray(tri_vertices, np.float64).reshape(-1, 3, 3)
        centroids = np.ascontiguousarray(vertices.mean(axis=1), np.float64)
        self.lens_assembly.register(
            tracer,
            list(self.lens_surface_groups),
            vertices,
            centroids,
            list(self.scene_lenses),
        )

    def remap_triangles(self, old_to_new: np.ndarray) -> "RebuiltCameraArtifact":
        """Carry lens proxy identities through subject-geometry replacement."""
        remap = np.asarray(old_to_new, np.int64).reshape(-1)

        def mapped(ids: np.ndarray) -> np.ndarray:
            source = np.asarray(ids, np.int64).reshape(-1)
            valid = (source >= 0) & (source < remap.size)
            result = remap[source[valid]]
            return np.ascontiguousarray(result[result >= 0], np.int32)

        return replace(
            self,
            lens_surface_groups=tuple(
                (mapped(front), mapped(back), radius_front, radius_back)
                for front, back, radius_front, radius_back in self.lens_surface_groups
            ),
        )

    def apply_film_pose(
        self,
        *,
        center: Sequence[float],
        right: Sequence[float],
        up: Sequence[float],
    ) -> None:
        """Update the camera-owned film stage after a validated adjustment."""
        self.sensor_center = np.ascontiguousarray(center, np.float64).reshape(3)
        self.sensor_right = np.ascontiguousarray(right, np.float64).reshape(3)
        self.sensor_up = np.ascontiguousarray(up, np.float64).reshape(3)
        straight = getattr(self.lens_assembly, "straight_section", None)
        if straight is not None:
            self.lens_assembly.straight_section = replace(
                straight,
                z_front=float(self.sensor_center[0]) + float(straight.sensor_z_offset),
            )

    def backward_target_spec(self) -> Any:
        return self.lens_assembly.backward_ray_target_spec()

    def compile_transport_graph(self, *, wave_key: str | None = None) -> Any:
        """Cold-compile this exact camera into the shared optical graph ABI.

        This is opt-in until the pipeline consumes compiled graph schedules
        directly; ordinary camera construction therefore pays no Torch/graph
        import or compilation cost.
        """
        from .optical_transport_graph import compile_compound_lens_graph

        return compile_compound_lens_graph(
            self.lens_assembly.require_optics(),
            lane_count=int(self.wavelengths_nm.size),
            wave_key=wave_key,
        )

    def describe(self) -> str:
        p = self.provenance
        return (
            f"build={p.build_id} optics={p.optical_model} "
            f"intersection={p.intersection_model} dispersion={p.dispersion_model} "
            f"wavelengths={p.wavelength_count} diffraction={p.diffraction_model}"
        )
