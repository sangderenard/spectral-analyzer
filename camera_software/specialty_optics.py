"""Physical specialty optics for the thick-lens camera.

The meshes in this module are transport geometry, not display effects.  The
pentaprism assembly provides explicit beam-splitter, glass, silvered, housing,
and instrumentation-receiver triangle roles.  The fisheye prescription is a
real stack of rotational conics consumed by the same exact CompoundLens path
as the ordinary camera.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 1.0e-12:
        raise ValueError(f"{name} must be a finite nonzero vector")
    return vector / length


@dataclass(frozen=True)
class OpticalTriangleAssembly:
    triangles: np.ndarray
    roles: tuple[str, ...]
    primary_path: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        triangles = np.ascontiguousarray(self.triangles, np.float64).reshape(-1, 3, 3)
        if triangles.shape[0] != len(self.roles):
            raise ValueError("one optical role is required for every triangle")
        if not np.all(np.isfinite(triangles)):
            raise ValueError("optical triangles must be finite")
        area_vectors = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        if np.any(np.linalg.norm(area_vectors, axis=1) <= 1.0e-12):
            raise ValueError("optical assembly contains a degenerate triangle")
        object.__setattr__(self, "triangles", triangles)

    def material_indices(self, role_materials: Mapping[str, int]) -> np.ndarray:
        missing = sorted(set(self.roles) - set(role_materials))
        if missing:
            raise KeyError(f"missing materials for optical roles {missing}")
        return np.ascontiguousarray(
            [int(role_materials[role]) for role in self.roles], np.int32
        )

    def role_triangles(self, role: str) -> np.ndarray:
        selected = [index for index, value in enumerate(self.roles) if value == role]
        return np.ascontiguousarray(self.triangles[selected], np.float64)


@dataclass(frozen=True)
class PentaprismSpec:
    """Closed constant-deviation pentaprism around an authored chief ray.

    ``entrance_center_m`` is the point where the diagnostic chief ray enters.
    ``input_axis`` and ``output_axis`` must be perpendicular.  The two silvered
    faces differ by 45 degrees and produce a net 90-degree turn.
    """

    entrance_center_m: tuple[float, float, float]
    input_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    output_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    clear_size_m: float = 0.040
    depth_m: float = 0.032
    glass: str = "N-BK7"

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_axis = _unit(self.input_axis, "input_axis")
        output_axis = _unit(self.output_axis, "output_axis")
        if abs(float(np.dot(input_axis, output_axis))) > 1.0e-7:
            raise ValueError("pentaprism input and output axes must be perpendicular")
        extrusion = _unit(np.cross(input_axis, output_axis), "extrusion_axis")
        return input_axis, output_axis, extrusion

    def _cross_section(self) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
        size = float(self.clear_size_m)
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("clear_size_m must be positive")
        diagonal = math.sqrt(0.5)
        p1 = np.array([0.42 * size, 0.0], np.float64)
        p2 = p1 + 0.40 * size * np.array([diagonal, diagonal])
        d0 = np.array([1.0, 0.0])
        d1 = np.array([diagonal, diagonal])
        d2 = np.array([0.0, 1.0])
        n1 = (d0 - d1) / np.linalg.norm(d0 - d1)
        n2 = (d1 - d2) / np.linalg.norm(d1 - d2)

        constraints = {
            "entrance_glass": (np.array([-1.0, 0.0]), 0.0),
            "exit_glass": (np.array([0.0, 1.0]), size),
            "silvered_reflector_1": (n1, float(n1 @ p1)),
            "silvered_reflector_2": (n2, float(n2 @ p2)),
            "blackened_prism_face": (
                np.array([-1.0, 1.0]) / math.sqrt(2.0),
                0.80 * size / math.sqrt(2.0),
            ),
        }

        def intersect(first: str, second: str) -> np.ndarray:
            n_a, c_a = constraints[first]
            n_b, c_b = constraints[second]
            return np.linalg.solve(np.stack([n_a, n_b]), np.array([c_a, c_b]))

        edge_roles = (
            "silvered_reflector_1",
            "silvered_reflector_2",
            "exit_glass",
            "blackened_prism_face",
            "entrance_glass",
        )
        vertices = np.stack([
            intersect("entrance_glass", "silvered_reflector_1"),
            intersect("silvered_reflector_1", "silvered_reflector_2"),
            intersect("silvered_reflector_2", "exit_glass"),
            intersect("exit_glass", "blackened_prism_face"),
            intersect("blackened_prism_face", "entrance_glass"),
        ])
        chief_path = np.stack([
            np.array([0.0, 0.0]), p1, p2,
            np.array([p2[0], size]),
        ])
        return vertices, edge_roles, chief_path

    def build(self) -> OpticalTriangleAssembly:
        input_axis, output_axis, extrusion = self._basis()
        origin = np.asarray(self.entrance_center_m, np.float64).reshape(3)
        polygon, edge_roles, chief_path_2d = self._cross_section()
        half_depth = 0.5 * float(self.depth_m)
        if not math.isfinite(half_depth) or half_depth <= 0.0:
            raise ValueError("depth_m must be positive")

        def world(point: np.ndarray, depth: float) -> np.ndarray:
            return (
                origin + point[0] * input_axis + point[1] * output_axis
                + depth * extrusion
            )

        lower = [world(point, -half_depth) for point in polygon]
        upper = [world(point, +half_depth) for point in polygon]
        triangles: list[np.ndarray] = []
        roles: list[str] = []
        for index, role in enumerate(edge_roles):
            nxt = (index + 1) % len(polygon)
            triangles.extend([
                np.stack([lower[index], lower[nxt], upper[nxt]]),
                np.stack([lower[index], upper[nxt], upper[index]]),
            ])
            roles.extend([role, role])
        for index in range(1, len(polygon) - 1):
            triangles.append(np.stack([lower[0], lower[index + 1], lower[index]]))
            triangles.append(np.stack([upper[0], upper[index], upper[index + 1]]))
            roles.extend(["blackened_prism_side", "blackened_prism_side"])
        chief_path = tuple(world(point, 0.0) for point in chief_path_2d)
        return OpticalTriangleAssembly(np.stack(triangles), tuple(roles), chief_path)


@dataclass(frozen=True)
class DiagnosticPickoffSpec:
    center_m: tuple[float, float, float]
    main_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    diverted_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    half_size_m: float = 0.024
    reflected_fraction: float = 0.30

    def build(self) -> OpticalTriangleAssembly:
        main = _unit(self.main_axis, "main_axis")
        diverted = _unit(self.diverted_axis, "diverted_axis")
        if abs(float(np.dot(main, diverted))) > 1.0e-7:
            raise ValueError("pickoff axes must be perpendicular")
        normal = _unit(main - diverted, "beam_splitter_normal")
        tangent_a = _unit(np.cross(main, diverted), "beam_splitter_tangent")
        tangent_b = _unit(np.cross(normal, tangent_a), "beam_splitter_tangent")
        center = np.asarray(self.center_m, np.float64).reshape(3)
        half = float(self.half_size_m)
        corners = [
            center - half * tangent_a - half * tangent_b,
            center + half * tangent_a - half * tangent_b,
            center + half * tangent_a + half * tangent_b,
            center - half * tangent_a + half * tangent_b,
        ]
        triangles = np.stack([
            np.stack([corners[0], corners[1], corners[2]]),
            np.stack([corners[0], corners[2], corners[3]]),
        ])
        return OpticalTriangleAssembly(
            triangles, ("diagnostic_beam_splitter",) * 2,
            (center - main * half, center, center + diverted * half),
        )


def build_pentaprism_diagnostic(
    *,
    pickoff_center_m: Sequence[float],
    camera_axis: Sequence[float] = (1.0, 0.0, 0.0),
    finder_axis: Sequence[float] = (0.0, 1.0, 0.0),
    instrument_axis: Sequence[float] = (0.0, 0.0, 1.0),
    prism_gap_m: float = 0.010,
    prism_size_m: float = 0.040,
    prism_depth_m: float = 0.032,
    receiver_gap_m: float = 0.012,
) -> OpticalTriangleAssembly:
    """Build a main-path pickoff, pentaprism, and focusing/meter screen."""

    center = np.asarray(pickoff_center_m, np.float64).reshape(3)
    main = _unit(camera_axis, "camera_axis")
    finder = _unit(finder_axis, "finder_axis")
    instrument = _unit(instrument_axis, "instrument_axis")
    pickoff = DiagnosticPickoffSpec(
        tuple(center), tuple(main), tuple(finder),
        half_size_m=0.45 * float(prism_size_m),
    ).build()
    prism = PentaprismSpec(
        tuple(center + finder * float(prism_gap_m)),
        tuple(finder), tuple(instrument),
        clear_size_m=float(prism_size_m), depth_m=float(prism_depth_m),
    ).build()
    exit_point = prism.primary_path[-1]
    receiver_center = exit_point + instrument * float(receiver_gap_m)
    receiver_half = 0.34 * float(prism_size_m)
    receiver_u = _unit(np.cross(instrument, finder), "receiver_u")
    receiver_v = _unit(np.cross(instrument, receiver_u), "receiver_v")
    corners = [
        receiver_center - receiver_half * receiver_u - receiver_half * receiver_v,
        receiver_center + receiver_half * receiver_u - receiver_half * receiver_v,
        receiver_center + receiver_half * receiver_u + receiver_half * receiver_v,
        receiver_center - receiver_half * receiver_u + receiver_half * receiver_v,
    ]
    receiver = np.stack([
        np.stack([corners[0], corners[2], corners[1]]),
        np.stack([corners[0], corners[3], corners[2]]),
    ])
    return OpticalTriangleAssembly(
        np.concatenate([pickoff.triangles, prism.triangles, receiver]),
        pickoff.roles + prism.roles + ("instrument_receiver",) * 2,
        pickoff.primary_path[:2] + prism.primary_path + (receiver_center,),
    )


@dataclass(frozen=True)
class DramaticFisheyeSpec:
    """A physical retrofocus fisheye prescription for the exact conic tracer."""

    front_vertex_x_m: float = 0.92
    focal_length_m: float = 0.020
    f_number: float = 2.8
    image_radius_m: float = 0.028

    def lens_stack(self):
        from thick_lens_focus_lab import LensConfig

        f = float(self.focal_length_m)
        if f <= 0.0 or float(self.f_number) <= 0.0:
            raise ValueError("fisheye focal length and f-number must be positive")
        x_envelope = float(self.front_vertex_x_m)
        # Negative meniscus front groups admit the extreme chief-ray angle;
        # the positive rear triplet restores power and flattens the field.
        prescription = (
            # thickness, gap, aperture/f, Rf/f, Rb/f, n, glass, kf, kb
            (0.70, 0.18, 3.20, -3.45, -5.20, 1.5168, "N-BK7", -0.82, -0.35),
            (0.38, 0.12, 2.65, -4.60, -2.75, 1.6200, "N-SF10", -0.40, -0.75),
            (0.44, 0.10, 2.15,  2.25,  3.80, 1.5891, "N-BAK4", -0.65, -0.20),
            (0.30, 0.16, 1.65, -3.60, -2.85, 1.6200, "N-SF10", -0.25, -0.55),
            (0.46, 0.08, 1.55,  2.10,  2.75, 1.5891, "N-BAK4", -0.70, -0.35),
            (0.36, 0.07, 1.40,  2.55,  5.20, 1.5168, "N-BK7", -0.45,  0.00),
            (0.32, 0.00, 1.30,  3.20,  2.40, 1.6200, "N-SF10", -0.25, -0.60),
        )
        result = []
        for thickness_f, gap_f, aperture_f, rf_f, rb_f, ior, glass, kf, kb in prescription:
            aperture = aperture_f * f
            radius_front = rf_f * f
            radius_back = rb_f * f

            def edge_sag(radius: float, conic: float) -> float:
                radicand = 1.0 - (1.0 + conic) * (aperture / radius) ** 2
                if radicand < 0.0:
                    raise ValueError("fisheye conic has no real edge at its aperture")
                return aperture * aperture / (
                    radius * (1.0 + math.sqrt(radicand))
                )

            front_sag = edge_sag(radius_front, kf)
            back_sag = edge_sag(-radius_back, kb)
            # Maintain a real positive glass edge thickness even on the strong
            # menisci; this also prevents adjacent BVH proxy volumes crossing.
            thickness = max(
                thickness_f * f,
                front_sag - back_sag + 0.25 * f,
            )
            local_min = min(0.0, front_sag, thickness, thickness + back_sag)
            local_max = max(0.0, front_sag, thickness, thickness + back_sag)
            front_vertex = x_envelope - local_min
            result.append(LensConfig(
                center_x=front_vertex + 0.5 * thickness,
                thickness=thickness,
                aperture_radius=aperture,
                radius_front=radius_front,
                radius_back=radius_back,
                ior=ior,
                glass=glass,
                conic_front=kf,
                conic_back=kb,
            ))
            x_envelope = front_vertex + local_max + gap_f * f
        return result

    def apply_to_scene(self, scene):
        stack = self.lens_stack()
        scene.optical_design = None
        scene.lens_stack = list(stack)
        scene.lens = stack[0]
        scene.image_plate.radius = max(
            float(scene.image_plate.radius), float(self.image_radius_m)
        )
        scene.image_plate.sensor_half_w = float(self.image_radius_m) / math.sqrt(2.0)
        scene.image_plate.sensor_half_h = float(self.image_radius_m) / math.sqrt(2.0)
        scene.image_plate.x = float(stack[-1].x_back + 1.10 * self.focal_length_m)
        scene.focus_distance_m = 1.0e6
        scene.lens_hood_min_half_field_deg = 60.0
        scene.ring_light_enabled = False
        return scene


__all__ = [
    "OpticalTriangleAssembly",
    "PentaprismSpec",
    "DiagnosticPickoffSpec",
    "build_pentaprism_diagnostic",
    "DramaticFisheyeSpec",
]
