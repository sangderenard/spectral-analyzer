"""Jones-complete reference boundary adapter for production T4 kernels.

This module is not a wave solver. It organizes coherent Jones modes and calls
the native T4 material/propagation kernels once per active transverse
component. It is the qualification path for the eventual pipeline-side
handle consumer and is also usable by diagnostics such as the physical
aperture demo.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from camera_designer.emitter_profile import PolarizationMode, PolarizationState
from .complex_optical_operators import TransverseBasis


_OPEN_DOMAIN_QUALITY = {
    # solve-width multiplier, propagation substeps, absorber width divisor
    "balanced": (2, 4, 8),
    "high": (4, 8, 8),
    "bake": (8, 16, 8),
}


@dataclass(frozen=True)
class PaddedWaveDomain:
    """Hidden open-boundary solve domain surrounding a visible field crop."""

    visible_shape: tuple[int, int]
    solve_shape: tuple[int, int]
    crop_y: slice
    crop_x: slice
    absorber_cells: int
    absorber_strength: float
    propagation_steps: int
    quality: str

    @classmethod
    def for_quality(
        cls,
        visible_shape: tuple[int, int],
        quality: str = "balanced",
        *,
        absorber_strength: float = 8.0,
    ) -> "PaddedWaveDomain":
        height, width = (int(value) for value in visible_shape)
        if height < 1 or width < 1:
            raise ValueError("visible wave-domain dimensions must be positive")
        key = str(quality).strip().lower()
        try:
            scale, steps, absorber_divisor = _OPEN_DOMAIN_QUALITY[key]
        except KeyError as exc:
            raise ValueError(
                f"wave-domain quality must be one of "
                f"{tuple(_OPEN_DOMAIN_QUALITY)}"
            ) from exc

        def power_of_two_ceiling(value: int) -> int:
            return 1 << (max(1, value) - 1).bit_length()

        solve_height = power_of_two_ceiling(height * scale)
        solve_width = power_of_two_ceiling(width * scale)
        pad_y = (solve_height - height) // 2
        pad_x = (solve_width - width) // 2
        exterior_margin = min(pad_y, pad_x)
        absorber_cells = min(
            max(12, min(solve_height, solve_width) // absorber_divisor),
            max(1, exterior_margin - 4),
        )
        if not math.isfinite(absorber_strength) or absorber_strength < 0.0:
            raise ValueError("absorber strength must be finite and non-negative")
        return cls(
            visible_shape=(height, width),
            solve_shape=(solve_height, solve_width),
            crop_y=slice(pad_y, pad_y + height),
            crop_x=slice(pad_x, pad_x + width),
            absorber_cells=absorber_cells,
            absorber_strength=float(absorber_strength),
            propagation_steps=steps,
            quality=key,
        )

    @property
    def sample_count(self) -> int:
        return int(self.solve_shape[0] * self.solve_shape[1])

    @property
    def investment_ratio(self) -> float:
        visible = self.visible_shape[0] * self.visible_shape[1]
        return float(self.sample_count / visible)

    def uniform_scalar_field(
        self, *, bands: int = 1, value: complex = 1.0 + 0.0j,
    ) -> np.ndarray:
        if int(bands) < 1:
            raise ValueError("wave-domain band count must be positive")
        return np.full(
            (int(bands), *self.solve_shape), value, dtype=np.complex64
        )

    def crop(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-2:] != self.solve_shape:
            raise ValueError("field does not match padded solve dimensions")
        return array[..., self.crop_y, self.crop_x]


@dataclass(frozen=True)
class JonesFieldState:
    """Independent coherent modes over s/p, spectral band, y, and x."""

    fields: np.ndarray  # complex64 [mode, component, band, y, x]
    coherence_ids: np.ndarray
    basis: TransverseBasis
    basis_id: int = 0
    operator_id: int = 0

    def __post_init__(self) -> None:
        fields = np.ascontiguousarray(self.fields, np.complex64)
        coherence = np.ascontiguousarray(self.coherence_ids, np.uint64)
        if fields.ndim != 5 or fields.shape[1] != 2:
            raise ValueError(
                "Jones fields require shape [mode,2,band,y,x]"
            )
        if coherence.shape != (fields.shape[0],):
            raise ValueError("one coherence id is required per Jones mode")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "coherence_ids", coherence)

    @property
    def mode_count(self) -> int:
        return int(self.fields.shape[0])

    @property
    def band_count(self) -> int:
        return int(self.fields.shape[2])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.fields.shape[3]), int(self.fields.shape[4])

    @classmethod
    def from_scalar_field(
        cls,
        scalar_field: np.ndarray,
        polarization: PolarizationState,
        *,
        basis: TransverseBasis | None = None,
        basis_id: int = 0,
        operator_id: int = 0,
        coherence_seed: int = 0,
    ) -> "JonesFieldState":
        scalar = np.asarray(scalar_field, np.complex64)
        if scalar.ndim == 2:
            scalar = scalar[None, :, :]
        if scalar.ndim != 3:
            raise ValueError("scalar field must have shape [band,y,x] or [y,x]")
        height, width = scalar.shape[-2:]
        y, x = np.mgrid[:height, :width]
        azimuth = np.arctan2(
            y - 0.5 * (height - 1),
            x - 0.5 * (width - 1),
        )
        mode = polarization.mode
        if mode is PolarizationMode.UNPOLARIZED:
            degree = 0.0
            primary_s = np.ones((height, width), np.complex64)
            primary_p = np.zeros((height, width), np.complex64)
        elif mode is PolarizationMode.RADIAL:
            degree = min(
                1.0, max(0.0, float(polarization.degree_of_polarization))
            )
            primary_s = np.cos(azimuth).astype(np.complex64)
            primary_p = np.sin(azimuth).astype(np.complex64)
        elif mode is PolarizationMode.AZIMUTHAL:
            degree = min(
                1.0, max(0.0, float(polarization.degree_of_polarization))
            )
            primary_s = (-np.sin(azimuth)).astype(np.complex64)
            primary_p = np.cos(azimuth).astype(np.complex64)
        else:
            degree = min(
                1.0, max(0.0, float(polarization.degree_of_polarization))
            )
            vector = polarization.jones_vector.astype(np.complex64)
            primary_s = np.full((height, width), vector[0], np.complex64)
            primary_p = np.full((height, width), vector[1], np.complex64)

        orthogonal_s = -np.conj(primary_p)
        orthogonal_p = np.conj(primary_s)
        weights = (0.5 * (1.0 + degree), 0.5 * (1.0 - degree))
        modes = []
        coherence_ids = []
        for mode_index, (weight, component_s, component_p) in enumerate((
            (weights[0], primary_s, primary_p),
            (weights[1], orthogonal_s, orthogonal_p),
        )):
            if weight <= 1.0e-15:
                continue
            amplitude = math.sqrt(weight)
            modes.append(np.stack((
                scalar * (amplitude * component_s)[None, :, :],
                scalar * (amplitude * component_p)[None, :, :],
            ), axis=0))
            coherence_ids.append(int(coherence_seed) + mode_index)
        return cls(
            np.stack(modes, axis=0),
            np.asarray(coherence_ids, np.uint64),
            basis or TransverseBasis.from_direction((0.0, 0.0, 1.0)),
            basis_id=int(basis_id),
            operator_id=int(operator_id),
        )

    def total_power(self) -> float:
        return float(np.sum(np.abs(self.fields.astype(np.complex128)) ** 2))

    def stokes(self, *, sum_bands: bool = True) -> tuple[np.ndarray, ...]:
        """Return incoherently accumulated I,Q,U,V."""

        s = self.fields[:, 0].astype(np.complex128)
        p = self.fields[:, 1].astype(np.complex128)
        cross = s * np.conj(p)
        axis = (0, 1) if sum_bands else 0
        intensity = np.sum(np.abs(s) ** 2 + np.abs(p) ** 2, axis=axis)
        q = np.sum(np.abs(s) ** 2 - np.abs(p) ** 2, axis=axis)
        u = np.sum(2.0 * np.real(cross), axis=axis)
        # Positive V corresponds to the repository's +handedness [1,+i]/sqrt2.
        v = np.sum(-2.0 * np.imag(cross), axis=axis)
        return intensity, q, u, v

    def analyzer_intensity(self, angle_deg: float) -> np.ndarray:
        angle = math.radians(float(angle_deg))
        projected = (
            math.cos(angle) * self.fields[:, 0]
            + math.sin(angle) * self.fields[:, 1]
        )
        return np.sum(np.abs(projected.astype(np.complex128)) ** 2, axis=(0, 1))

    def apply_isotropic_material_native(
        self,
        tracer: Any,
        *,
        pitch_m: float,
        distance_m: float,
        direction_sign: int,
        wavelengths_m: np.ndarray,
        payload: np.ndarray,
    ) -> tuple["JonesFieldState", dict[str, float]]:
        """Apply the existing scalar material kernel to every Jones component."""

        output = self.fields.copy()
        absorbed = 0.0
        for mode_index in range(self.mode_count):
            for component in range(2):
                values = output[mode_index, component]
                re = np.ascontiguousarray(values.real, np.float32)
                im = np.ascontiguousarray(values.imag, np.float32)
                result = tracer.t4_apply_aperture_material(
                    self.band_count,
                    self.shape[1],
                    self.shape[0],
                    float(pitch_m),
                    float(distance_m),
                    int(direction_sign),
                    np.ascontiguousarray(wavelengths_m, np.float64),
                    np.ascontiguousarray(payload, np.float64),
                    re,
                    im,
                )
                output[mode_index, component] = re + 1j * im
                absorbed += float(result.get("absorbed_power", 0.0))
        state = JonesFieldState(
            output,
            self.coherence_ids,
            self.basis,
            self.basis_id,
            self.operator_id,
        )
        return state, {
            "input_power": self.total_power(),
            "output_power": state.total_power(),
            "absorbed_power": absorbed,
            "ideal_mask": False,
        }

    def propagate_native(
        self,
        tracer: Any,
        *,
        pitch_m: float,
        distance_m: float,
        direction_sign: int,
        wavelengths_m: np.ndarray,
    ) -> "JonesFieldState":
        output = self.fields.copy()
        for mode_index in range(self.mode_count):
            for component in range(2):
                values = output[mode_index, component]
                re = np.ascontiguousarray(values.real, np.float32)
                im = np.ascontiguousarray(values.imag, np.float32)
                tracer.t4_angular_spectrum_step(
                    self.band_count,
                    self.shape[1],
                    self.shape[0],
                    float(pitch_m),
                    float(distance_m),
                    int(direction_sign),
                    np.ascontiguousarray(wavelengths_m, np.float64),
                    re,
                    im,
                )
                output[mode_index, component] = re + 1j * im
        return JonesFieldState(
            output,
            self.coherence_ids,
            self.basis,
            self.basis_id,
            self.operator_id,
        )

    def propagate_open_native(
        self,
        tracer: Any,
        *,
        pitch_m: float,
        distance_m: float,
        direction_sign: int,
        wavelengths_m: np.ndarray,
        domain: PaddedWaveDomain,
    ) -> tuple["JonesFieldState", dict[str, float]]:
        """Substep propagation through the production absorbing exterior."""

        if self.shape != domain.solve_shape:
            raise ValueError("Jones field must occupy the padded solve domain")
        output = self.fields.copy()
        steps = int(domain.propagation_steps)
        step_distance = float(distance_m) / steps
        absorbed = 0.0
        border_power = 0.0
        for _step in range(steps):
            step_border_power = 0.0
            for mode_index in range(self.mode_count):
                for component in range(2):
                    values = output[mode_index, component]
                    re = np.ascontiguousarray(values.real, np.float32)
                    im = np.ascontiguousarray(values.imag, np.float32)
                    tracer.t4_angular_spectrum_step(
                        self.band_count,
                        self.shape[1],
                        self.shape[0],
                        float(pitch_m),
                        step_distance,
                        int(direction_sign),
                        np.ascontiguousarray(wavelengths_m, np.float64),
                        re,
                        im,
                    )
                    result = tracer.t4_apply_absorbing_border(
                        self.band_count,
                        self.shape[1],
                        self.shape[0],
                        int(domain.absorber_cells),
                        float(domain.absorber_strength),
                        1.0 / steps,
                        re,
                        im,
                    )
                    output[mode_index, component] = re + 1j * im
                    absorbed += float(result.get("absorbed_power", 0.0))
                    step_border_power += float(
                        result.get("border_power", 0.0)
                    )
            border_power = step_border_power
        state = JonesFieldState(
            output,
            self.coherence_ids,
            self.basis,
            self.basis_id,
            self.operator_id,
        )
        return state, {
            "input_power": self.total_power(),
            "output_power": state.total_power(),
            "absorbed_power": absorbed,
            "border_power": border_power,
            "border_fraction": border_power / max(state.total_power(), 1.0e-30),
            "steps": float(steps),
            "solve_samples": float(domain.sample_count),
            "investment_ratio": domain.investment_ratio,
        }


__all__ = ["JonesFieldState", "PaddedWaveDomain"]
