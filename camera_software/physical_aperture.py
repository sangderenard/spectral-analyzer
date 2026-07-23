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
    CIRCULAR_HOLE = 5


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
        elif self.pattern is AperturePattern.CIRCULAR_HOLE:
            if self.opening_x_m >= self.assembly_radius_m:
                raise ValueError("circular opening must fit inside its assembly")
            if self.element_count < 12:
                raise ValueError(
                    "circular-bore display geometry requires at least 12 segments"
                )
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
            if 2.0*self.opening_x_m >= self.pitch_x_m:
                raise ValueError("x openings must leave finite material between cells")
            if (
                self.pattern is not AperturePattern.APERTURE_GRILLE
                and 2.0*self.opening_y_m >= self.pitch_y_m
            ):
                raise ValueError("y openings must leave finite material between cells")

    @staticmethod
    def _material_index(
        material_name: str,
        material_n_real: float | None,
        material_n_imag: float | None,
    ) -> tuple[float, float]:
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
        return float(material_n_real), float(material_n_imag)

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
        material_n_real, material_n_imag = cls._material_index(
            material_name, material_n_real, material_n_imag,
        )
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

    @classmethod
    def circular_hole(
        cls,
        key: str,
        *,
        opening_radius_m: float,
        assembly_radius_m: float,
        thickness_m: float,
        display_segments: int = 96,
        material_name: str = "blackened_steel",
        material_n_real: float | None = None,
        material_n_imag: float | None = None,
    ) -> "LivePhysicalAperture":
        """A true circular bore through a finite material plate."""

        material_n_real, material_n_imag = cls._material_index(
            material_name, material_n_real, material_n_imag,
        )
        result = cls(
            key=key,
            pattern=AperturePattern.CIRCULAR_HOLE,
            element_count=int(display_segments),
            opening_x_m=float(opening_radius_m),
            opening_y_m=float(opening_radius_m),
            assembly_radius_m=float(assembly_radius_m),
            thickness_m=float(thickness_m),
            material_name=str(material_name),
            material_n_real=material_n_real,
            material_n_imag=material_n_imag,
        )
        result.validate()
        return result

    @classmethod
    def circular_hole_array(
        cls,
        key: str,
        *,
        hole_radius_m: float,
        pitch_x_m: float,
        pitch_y_m: float,
        assembly_radius_m: float,
        thickness_m: float,
        rotation_rad: float = 0.0,
        material_name: str = "blackened_steel",
        material_n_real: float | None = None,
        material_n_imag: float | None = None,
    ) -> "LivePhysicalAperture":
        """A two-axis elliptical/circular perforated scientific scrim."""

        material_n_real, material_n_imag = cls._material_index(
            material_name, material_n_real, material_n_imag,
        )
        result = cls(
            key=key,
            pattern=AperturePattern.SHADOW_MASK,
            opening_x_m=float(hole_radius_m),
            opening_y_m=float(hole_radius_m),
            pitch_x_m=float(pitch_x_m),
            pitch_y_m=float(pitch_y_m),
            assembly_radius_m=float(assembly_radius_m),
            thickness_m=float(thickness_m),
            rotation_rad=float(rotation_rad),
            material_name=str(material_name),
            material_n_real=material_n_real,
            material_n_imag=material_n_imag,
        )
        result.validate()
        return result

    @classmethod
    def slot_array(
        cls,
        key: str,
        *,
        slot_width_m: float,
        slot_height_m: float,
        pitch_x_m: float,
        pitch_y_m: float,
        assembly_radius_m: float,
        thickness_m: float,
        rotation_rad: float = 0.0,
        material_name: str = "blackened_steel",
        material_n_real: float | None = None,
        material_n_imag: float | None = None,
    ) -> "LivePhysicalAperture":
        """A rectangular two-axis slot array with finite bars."""

        material_n_real, material_n_imag = cls._material_index(
            material_name, material_n_real, material_n_imag,
        )
        result = cls(
            key=key,
            pattern=AperturePattern.SLOT_MASK,
            opening_x_m=0.5*float(slot_width_m),
            opening_y_m=0.5*float(slot_height_m),
            pitch_x_m=float(pitch_x_m),
            pitch_y_m=float(pitch_y_m),
            assembly_radius_m=float(assembly_radius_m),
            thickness_m=float(thickness_m),
            rotation_rad=float(rotation_rad),
            material_name=str(material_name),
            material_n_real=material_n_real,
            material_n_imag=material_n_imag,
        )
        result.validate()
        return result

    @classmethod
    def grating(
        cls,
        key: str,
        *,
        slit_width_m: float,
        pitch_m: float,
        assembly_radius_m: float,
        thickness_m: float,
        rotation_rad: float = 0.0,
        material_name: str = "blackened_steel",
        material_n_real: float | None = None,
        material_n_imag: float | None = None,
    ) -> "LivePhysicalAperture":
        """A one-dimensional transmission grating made from finite bars."""

        material_n_real, material_n_imag = cls._material_index(
            material_name, material_n_real, material_n_imag,
        )
        result = cls(
            key=key,
            pattern=AperturePattern.APERTURE_GRILLE,
            opening_x_m=0.5*float(slit_width_m),
            opening_y_m=float(assembly_radius_m),
            pitch_x_m=float(pitch_m),
            pitch_y_m=0.0,
            assembly_radius_m=float(assembly_radius_m),
            thickness_m=float(thickness_m),
            rotation_rad=float(rotation_rad),
            material_name=str(material_name),
            material_n_real=material_n_real,
            material_n_imag=material_n_imag,
        )
        result.validate()
        return result

    def open_mask(
        self,
        x_m: np.ndarray | float,
        y_m: np.ndarray | float,
    ) -> np.ndarray:
        """Vectorized geometric occupancy matching the native material kernel."""

        self.validate()
        x = np.asarray(x_m, np.float64)
        y = np.asarray(y_m, np.float64)
        c, s = math.cos(self.rotation_rad), math.sin(self.rotation_rad)
        u, v = c*x+s*y, -s*x+c*y
        if self.pattern is AperturePattern.CIRCULAR_HOLE:
            opened = u*u+v*v <= self.opening_x_m*self.opening_x_m
        elif self.pattern is AperturePattern.IRIS_POLYGON:
            opened = np.ones(np.broadcast_shapes(u.shape, v.shape), bool)
            n = self.element_count
            for edge in range(n):
                a0, a1 = 2.0*math.pi*edge/n, 2.0*math.pi*(edge+1)/n
                x0, y0 = self.opening_x_m*math.cos(a0), self.opening_x_m*math.sin(a0)
                x1, y1 = self.opening_x_m*math.cos(a1), self.opening_x_m*math.sin(a1)
                opened &= (x1-x0)*(v-y0)-(y1-y0)*(u-x0) >= 0.0
        elif self.pattern is AperturePattern.SHADOW_MASK:
            cell_x = u-np.round(u/self.pitch_x_m)*self.pitch_x_m
            cell_y = v-np.round(v/self.pitch_y_m)*self.pitch_y_m
            opened = (
                (cell_x/self.opening_x_m)**2
                + (cell_y/self.opening_y_m)**2 <= 1.0
            )
        elif self.pattern is AperturePattern.SLOT_MASK:
            cell_x = u-np.round(u/self.pitch_x_m)*self.pitch_x_m
            cell_y = v-np.round(v/self.pitch_y_m)*self.pitch_y_m
            opened = (
                np.abs(cell_x) <= self.opening_x_m
            ) & (np.abs(cell_y) <= self.opening_y_m)
        else:
            cell_x = u-np.round(u/self.pitch_x_m)*self.pitch_x_m
            opened = np.abs(cell_x) <= self.opening_x_m
        outside = x*x+y*y > self.assembly_radius_m*self.assembly_radius_m
        return np.asarray(opened | outside, bool)

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
        if self.pattern not in (
            AperturePattern.IRIS_POLYGON,
            AperturePattern.CIRCULAR_HOLE,
        ):
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
