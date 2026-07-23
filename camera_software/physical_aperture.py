"""Reusable live physical-aperture authoring and T4 ABI.

One descriptor is authoritative for ray/GL geometry and localized complex
field interaction.  It never represents an ideal binary aperture: opaque
regions have finite thickness and a named complex-index material.  Baking is
deliberately outside this module; ``wave_payload`` is the live ABI consumed by
the persistent T4 arena.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Sequence

import numpy as np


APERTURE_PAYLOAD_MAGIC = 41_505_434.0  # "APTR"
APERTURE_PAYLOAD_VERSION = 1.0
APERTURE_PAYLOAD_VALUES = 19


class AperturePattern(IntEnum):
    IRIS_POLYGON = 1
    SHADOW_MASK = 2
    SLOT_MASK = 3
    APERTURE_GRILLE = 4


@dataclass(frozen=True)
class LivePhysicalAperture:
    """Finite material assembly shared by ray, GL, and wave contexts."""

    key: str
    pattern: AperturePattern
    opening_x_m: float
    opening_y_m: float
    assembly_radius_m: float
    thickness_m: float
    material_name: str = "blackened_steel"
    material_n_real: float = 2.9
    material_n_imag: float = 3.0
    element_count: int = 0
    pitch_x_m: float = 0.0
    pitch_y_m: float = 0.0
    rotation_rad: float = 0.0

    def validate(self) -> None:
        if not self.key.strip():
            raise ValueError("physical aperture key must be non-empty")
        finite = (
            self.opening_x_m, self.opening_y_m, self.assembly_radius_m,
            self.thickness_m, self.material_n_real, self.material_n_imag,
            self.pitch_x_m, self.pitch_y_m, self.rotation_rad,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("physical aperture parameters must be finite")
        if self.opening_x_m <= 0.0 or self.assembly_radius_m <= 0.0:
            raise ValueError("opening and assembly radii must be positive")
        if self.thickness_m <= 0.0:
            raise ValueError("physical aperture thickness must be positive")
        if self.material_n_real <= 0.0 or self.material_n_imag < 0.0:
            raise ValueError("material complex index must be physical")
        if self.pattern is AperturePattern.IRIS_POLYGON:
            if self.element_count < 3:
                raise ValueError("an iris requires at least three blades")
            if self.opening_x_m >= self.assembly_radius_m:
                raise ValueError("iris opening must fit inside its assembly")
        else:
            if self.pitch_x_m <= 0.0:
                raise ValueError("a repeated aperture requires positive x pitch")
            if (
                self.pattern is not AperturePattern.APERTURE_GRILLE
                and self.pitch_y_m <= 0.0
            ):
                raise ValueError("a two-axis aperture requires positive y pitch")
            if self.opening_y_m <= 0.0:
                raise ValueError("repeated aperture y opening must be positive")

    @classmethod
    def iris(
        cls,
        key: str,
        *,
        blade_count: int,
        opening_radius_m: float,
        assembly_radius_m: float,
        thickness_m: float,
        rotation_rad: float = 0.0,
        material_name: str = "blackened_steel",
        material_n_real: float | None = None,
        material_n_imag: float | None = None,
    ) -> "LivePhysicalAperture":
        if material_n_real is None or material_n_imag is None:
            from camera_designer.optical_material import MATERIAL_CATALOG

            try:
                canonical = MATERIAL_CATALOG[str(material_name)]
            except KeyError as exc:
                raise ValueError(
                    f"unknown optical aperture material {material_name!r}"
                ) from exc
            if material_n_real is None:
                material_n_real = float(canonical.n_d)
            if material_n_imag is None:
                material_n_imag = float(canonical.k)
        result = cls(
            key=key,
            pattern=AperturePattern.IRIS_POLYGON,
            element_count=int(blade_count),
            opening_x_m=float(opening_radius_m),
            opening_y_m=float(opening_radius_m),
            assembly_radius_m=float(assembly_radius_m),
            thickness_m=float(thickness_m),
            rotation_rad=float(rotation_rad),
            material_name=str(material_name),
            material_n_real=float(material_n_real),
            material_n_imag=float(material_n_imag),
        )
        result.validate()
        return result

    def wave_payload(
        self, axis: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> np.ndarray:
        """Return the fixed 19-double live T4 payload.

        Values 0:3 remain reserved for scale-context transforms and 3:6 carry
        the established propagation axis.  The aperture extension begins at 6.
        """

        self.validate()
        direction = np.asarray(axis, np.float64).reshape(-1)
        if direction.size != 3 or not np.all(np.isfinite(direction)):
            raise ValueError("aperture wave axis must contain three finite values")
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            raise ValueError("aperture wave axis must be non-zero")
        direction /= norm
        payload = np.zeros(APERTURE_PAYLOAD_VALUES, np.float64)
        payload[3:6] = direction
        payload[6:] = (
            APERTURE_PAYLOAD_MAGIC,
            APERTURE_PAYLOAD_VERSION,
            float(self.pattern),
            float(self.element_count),
            self.opening_x_m,
            self.opening_y_m,
            self.pitch_x_m,
            self.pitch_y_m,
            self.rotation_rad,
            self.assembly_radius_m,
            self.thickness_m,
            self.material_n_real,
            self.material_n_imag,
        )
        return np.ascontiguousarray(payload)

    def graph_parameters(self) -> dict[str, object]:
        """Self-description retained in the cold optical graph contract."""

        self.validate()
        return {
            "schema": "live-physical-aperture-v1",
            "key": self.key,
            "pattern": self.pattern.name.lower().replace("_", "-"),
            "element_count": self.element_count,
            "opening_x_m": self.opening_x_m,
            "opening_y_m": self.opening_y_m,
            "pitch_x_m": self.pitch_x_m,
            "pitch_y_m": self.pitch_y_m,
            "rotation_rad": self.rotation_rad,
            "assembly_radius_m": self.assembly_radius_m,
            "thickness_m": self.thickness_m,
            "material_name": self.material_name,
            "material_n_real": self.material_n_real,
            "material_n_imag": self.material_n_imag,
            "interaction": "finite-complex-index-split-step",
            "ideal_mask": False,
        }

    def triangle_mesh(self, *, z_center_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Build finite-thickness material geometry for the live iris.

        The returned arrays match the camera scene builder's triangle ABI:
        flattened vertices ``(T, 9)`` and one normal ``(T, 3)``. Repeated CRT
        sheets remain parametric/instanced and are intentionally not exploded
        into a giant CPU triangle soup by this method.
        """

        self.validate()
        if self.pattern is not AperturePattern.IRIS_POLYGON:
            raise NotImplementedError(
                "repeated masks/grilles require the instanced parametric path"
            )
        n = self.element_count
        angles = (
            np.arange(n, dtype=np.float64)*(2.0*math.pi/n)
            + self.rotation_rad
        )
        inner = self.opening_x_m*np.stack(
            (np.cos(angles), np.sin(angles)), axis=1
        )
        outer = self.assembly_radius_m*np.stack(
            (np.cos(angles), np.sin(angles)), axis=1
        )
        z0 = float(z_center_m)-0.5*self.thickness_m
        z1 = float(z_center_m)+0.5*self.thickness_m
        triangles: list[np.ndarray] = []
        normals: list[np.ndarray] = []

        def add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
            normal = np.cross(b-a, c-a)
            length = float(np.linalg.norm(normal))
            if length <= 1.0e-30:
                return
            triangles.append(np.concatenate((a, b, c)))
            normals.append(normal/length)

        for blade in range(n):
            next_blade = (blade+1) % n
            polygon_xy = (inner[blade], outer[blade],
                          outer[next_blade], inner[next_blade])
            bottom = [np.array([p[0], p[1], z0]) for p in polygon_xy]
            top = [np.array([p[0], p[1], z1]) for p in polygon_xy]
            add(bottom[0], bottom[2], bottom[1])
            add(bottom[0], bottom[3], bottom[2])
            add(top[0], top[1], top[2])
            add(top[0], top[2], top[3])
            for edge in range(4):
                nxt = (edge+1) % 4
                add(bottom[edge], bottom[nxt], top[nxt])
                add(bottom[edge], top[nxt], top[edge])
        return (
            np.ascontiguousarray(triangles, np.float64),
            np.ascontiguousarray(normals, np.float64),
        )


__all__ = [
    "APERTURE_PAYLOAD_MAGIC",
    "APERTURE_PAYLOAD_VERSION",
    "APERTURE_PAYLOAD_VALUES",
    "AperturePattern",
    "LivePhysicalAperture",
]
