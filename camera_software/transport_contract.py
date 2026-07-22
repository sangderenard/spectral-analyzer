"""Renderer-neutral spectral transport contracts.

Fixed spectral lanes contain frequencies.  Continuous spectral lanes do not:
they contain stable indices into a spectral lookup library.  At ray launch the
lane selects a LUT, a single frequency/PDF pair is resolved from it, and that
pair remains immutable for the complete ray path.
"""
from __future__ import annotations

import bisect
import functools
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Mapping


TRANSPORT_ABI_VERSION = 2
MAX_TRANSPORT_LANES = 32
VISIBLE_WAVELENGTH_MIN_NM = 380.0
VISIBLE_WAVELENGTH_MAX_NM = 700.0


def native_display_rgb_weight(wavelength_nm: float) -> tuple[float, float, float]:
    """Return the renderer's non-negative linear display weight for one line.

    This deliberately matches ``band_to_display_rgb`` in the native renderer.
    Keeping the transport manifest and GPU conversion on one transfer prevents
    a fixed spectral lane from silently falling back to equal-weight grey.
    """

    wl = min(VISIBLE_WAVELENGTH_MAX_NM, max(VISIBLE_WAVELENGTH_MIN_NM,
                                             float(wavelength_nm)))
    r = g = b = 0.0
    if wl < 440.0:
        r = -(wl - 440.0) / 60.0
        b = 1.0
    elif wl < 490.0:
        g = (wl - 440.0) / 50.0
        b = 1.0
    elif wl < 510.0:
        g = 1.0
        b = -(wl - 510.0) / 20.0
    elif wl < 580.0:
        r = (wl - 510.0) / 70.0
        g = 1.0
    elif wl < 645.0:
        r = 1.0
        g = -(wl - 645.0) / 65.0
    else:
        r = 1.0
    edge = 1.0
    if wl < 420.0:
        edge = 0.3 + 0.7 * (wl - 380.0) / 40.0
    elif wl > 645.0:
        edge = 0.3 + 0.7 * (700.0 - wl) / 55.0
    return r * edge, g * edge, b * edge


def _cie_1931_xyz(wavelength_nm: float) -> tuple[float, float, float]:
    """Smooth CIE 1931 2-degree CMF approximation used for lane placement."""

    wl = float(wavelength_nm)

    def gaussian(mu: float, sigma: float, amplitude: float = 1.0) -> float:
        return amplitude * math.exp(-0.5 * ((wl - mu) / sigma) ** 2)

    return (
        gaussian(599.8, 37.9, 1.056)
        + gaussian(442.0, 16.0, 0.362)
        + gaussian(501.1, 20.4, -0.065),
        gaussian(568.8, 46.9, 0.821)
        + gaussian(530.9, 16.3, 0.286),
        gaussian(437.0, 20.0, 1.217)
        + gaussian(459.0, 11.18, 0.681),
    )


@functools.lru_cache(maxsize=5)
def perceptual_visible_bins(
    lane_count: int,
) -> tuple[tuple[float, float, float], ...]:
    """Return ``(lower_nm, centre_nm, upper_nm)`` perceptual bins.

    Boundaries divide CIE 1976 u'v' locus arc length equally. Their unequal
    nanometre widths are retained for quadrature rather than pretending that
    every perceptual bin integrates the same amount of a flat spectrum.
    """

    count = int(lane_count)
    if count not in (1, 3, 8, 16, 32):
        raise ValueError("fixed visible transport requires 1, 3, 8, 16, or 32 lanes")
    step_nm = 0.25
    sample_count = int(round(
        (VISIBLE_WAVELENGTH_MAX_NM - VISIBLE_WAVELENGTH_MIN_NM) / step_nm
    )) + 1
    wavelengths = [VISIBLE_WAVELENGTH_MIN_NM + i * step_nm
                   for i in range(sample_count)]
    uv: list[tuple[float, float]] = []
    for wl in wavelengths:
        x, y, z = _cie_1931_xyz(wl)
        denom = max(1.0e-30, x + 15.0 * y + 3.0 * z)
        uv.append((4.0 * x / denom, 9.0 * y / denom))
    cumulative = [0.0]
    for previous, current in zip(uv[:-1], uv[1:]):
        cumulative.append(cumulative[-1] + math.hypot(
            current[0] - previous[0], current[1] - previous[1]
        ))
    total = cumulative[-1]
    def wavelength_at(target: float) -> float:
        cursor = min(len(cumulative) - 1, max(
            1, bisect.bisect_left(cumulative, target)
        ))
        lo_s, hi_s = cumulative[cursor - 1], cumulative[cursor]
        fraction = 0.0 if hi_s <= lo_s else (target - lo_s) / (hi_s - lo_s)
        return wavelengths[cursor - 1] + fraction * step_nm

    boundaries = [wavelength_at(total * lane / count)
                  for lane in range(count + 1)]
    centres = [wavelength_at(total * (lane + 0.5) / count)
               for lane in range(count)]
    return tuple((boundaries[i], centres[i], boundaries[i + 1])
                 for i in range(count))


def perceptual_visible_wavelengths(lane_count: int) -> tuple[float, ...]:
    """Return the centre wavelength of every perceptually equal visible bin."""

    return tuple(center for _lower, center, _upper
                 in perceptual_visible_bins(int(lane_count)))


class TransportDomain(str, Enum):
    ACHROMATIC_SCALAR = "achromatic_scalar"
    DISPLAY_RGB = "display_rgb"
    FIXED_SPECTRAL = "fixed_spectral"
    CONTINUOUS_SPECTRAL_LUT = "continuous_spectral_lut"



@dataclass(frozen=True)
class SpectralLookupTable:
    """Piecewise-linear probability distribution stored in the scene ABI."""

    key: str
    frequency_knots_hz: tuple[float, ...]
    density: tuple[float, ...]

    def __post_init__(self) -> None:
        f = tuple(float(v) for v in self.frequency_knots_hz)
        d = tuple(float(v) for v in self.density)
        if not self.key:
            raise ValueError("spectral LUT requires a stable key")
        if len(f) != len(d) or len(f) < 2:
            raise ValueError("spectral LUT knots/density require equal length >= 2")
        if any(not math.isfinite(v) or v <= 0.0 for v in f):
            raise ValueError("spectral LUT frequencies must be finite and positive")
        if any(b <= a for a, b in zip(f, f[1:])):
            raise ValueError("spectral LUT frequencies must increase")
        if any(not math.isfinite(v) or v < 0.0 for v in d):
            raise ValueError("spectral LUT density must be finite and non-negative")
        if self.integral <= 0.0 or not math.isfinite(self.integral):
            raise ValueError("spectral LUT density must have positive finite area")
        object.__setattr__(self, "frequency_knots_hz", f)
        object.__setattr__(self, "density", d)

    @property
    def integral(self) -> float:
        return math.fsum(
            0.5 * (d0 + d1) * (f1 - f0)
            for f0, f1, d0, d1 in zip(
                self.frequency_knots_hz, self.frequency_knots_hz[1:],
                self.density, self.density[1:]
            )
        )

    def mapping(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "frequency_knots_hz": list(self.frequency_knots_hz),
            "density": list(self.density),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpectralLookupTable":
        return cls(
            key=str(value["key"]),
            frequency_knots_hz=tuple(float(v) for v in value["frequency_knots_hz"]),
            density=tuple(float(v) for v in value["density"]),
        )


@dataclass(frozen=True)
class ResolvedRayFrequency:
    lut_index: int
    sample_u: float
    frequency_hz: float
    sampling_pdf: float


class SpectralLookupCache:
    """Validated LUT registry plus deterministic resolved-value cache.

    Cache keys are `(lut_index, sample_u.hex())`; this avoids rounding an exact
    Monte-Carlo coordinate and guarantees that resolving the same launch state
    cannot silently produce a different frequency later in a path.
    """

    def __init__(self, tables: tuple[SpectralLookupTable, ...]):
        if not tables:
            raise ValueError("continuous transport requires at least one LUT")
        keys = [table.key for table in tables]
        if len(keys) != len(set(keys)):
            raise ValueError("spectral LUT keys must be unique")
        self.tables = tuple(tables)
        self._resolved: dict[tuple[int, str], ResolvedRayFrequency] = {}

    def resolve(self, lut_index: int, sample_u: float) -> ResolvedRayFrequency:
        index = int(lut_index)
        u = float(sample_u)
        if not 0 <= index < len(self.tables):
            raise IndexError("spectral LUT index is out of range")
        if not math.isfinite(u) or not 0.0 <= u < 1.0:
            raise ValueError("spectral sample coordinate must lie in [0, 1)")
        cache_key = (index, u.hex())
        cached = self._resolved.get(cache_key)
        if cached is not None:
            return cached

        table = self.tables[index]
        total = table.integral
        areas = [
            0.5 * (d0 + d1) * (f1 - f0)
            for f0, f1, d0, d1 in zip(
                table.frequency_knots_hz, table.frequency_knots_hz[1:],
                table.density, table.density[1:]
            )
        ]
        cumulative = [0.0]
        for area in areas:
            cumulative.append(cumulative[-1] + area / total)
        cumulative[-1] = 1.0
        interval = min(len(areas) - 1, max(0, bisect.bisect_right(cumulative, u) - 1))
        while areas[interval] <= 0.0:
            interval += 1
        f0, f1 = table.frequency_knots_hz[interval:interval + 2]
        d0, d1 = table.density[interval:interval + 2]
        width = f1 - f0
        target_area = (u - cumulative[interval]) * total
        slope = (d1 - d0) / width
        if abs(slope) <= 1.0e-30:
            offset = target_area / max(d0, 1.0e-300)
        else:
            disc = max(0.0, d0 * d0 + 2.0 * slope * target_area)
            roots = ((-d0 + math.sqrt(disc)) / slope,
                     (-d0 - math.sqrt(disc)) / slope)
            offset = next(v for v in roots if -1.0e-9 <= v <= width + 1.0e-9)
        offset = min(width, max(0.0, offset))
        frequency = f0 + offset
        resolved = ResolvedRayFrequency(
            lut_index=index,
            sample_u=u,
            frequency_hz=frequency,
            sampling_pdf=(d0 + slope * offset) / total,
        )
        self._resolved[cache_key] = resolved
        return resolved

    @property
    def cached_resolution_count(self) -> int:
        return len(self._resolved)


@dataclass(frozen=True)
class SpectralLaneDescriptor:
    lane_id: int
    identity: str
    frequency_hz: float | None = None
    lut_index: int | None = None
    # Historical ABI name: native consumers treat this as linear display RGB.
    # None means derive the equipment default from the lane/domain.
    sensor_weight_xyz: tuple[float, float, float] | None = None
    wavelength_bin_nm: tuple[float, float] | None = None
    quadrature_weight: float = 1.0

    def validate(self, domain: TransportDomain, lut_count: int) -> None:
        if not 0 <= int(self.lane_id) < MAX_TRANSPORT_LANES:
            raise ValueError("lane_id lies outside the native 32-lane ABI")
        if not self.identity:
            raise ValueError("every transport lane requires a stable identity")
        if domain is TransportDomain.FIXED_SPECTRAL:
            if self.frequency_hz is None or not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
                raise ValueError("fixed spectral lanes require a positive frequency")
            if self.lut_index is not None:
                raise ValueError("fixed spectral lanes cannot reference a LUT")
        elif domain is TransportDomain.CONTINUOUS_SPECTRAL_LUT:
            if self.frequency_hz is not None:
                raise ValueError("continuous LUT lanes must not contain a preselected frequency")
            if self.lut_index is None or not 0 <= int(self.lut_index) < lut_count:
                raise ValueError("continuous spectral lane requires a valid LUT index")
        if self.sensor_weight_xyz is None or len(self.sensor_weight_xyz) != 3:
            raise ValueError("transport lane requires three resolved sensor weights")
        if any(not math.isfinite(float(value)) or float(value) < 0.0
               for value in self.sensor_weight_xyz):
            raise ValueError("transport lane sensor weights must be finite and non-negative")
        if not math.isfinite(float(self.quadrature_weight)) or float(self.quadrature_weight) <= 0.0:
            raise ValueError("transport lane quadrature weight must be finite and positive")
        if self.wavelength_bin_nm is not None:
            lo, hi = (float(value) for value in self.wavelength_bin_nm)
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError("transport wavelength bin requires finite increasing bounds")


@dataclass(frozen=True)
class TransportLaneTable:
    domain: TransportDomain
    cohort_id: str
    lanes: tuple[SpectralLaneDescriptor, ...]
    lookup_tables: tuple[SpectralLookupTable, ...] = ()
    abi_version: int = TRANSPORT_ABI_VERSION

    def __post_init__(self) -> None:
        if int(self.abi_version) != TRANSPORT_ABI_VERSION:
            raise ValueError("unsupported transport-lane ABI version")
        if not self.cohort_id or not 1 <= len(self.lanes) <= MAX_TRANSPORT_LANES:
            raise ValueError("transport lane table requires a cohort and 1..32 lanes")
        if len({lane.lane_id for lane in self.lanes}) != len(self.lanes):
            raise ValueError("transport lane ids must be unique")
        if self.domain is TransportDomain.CONTINUOUS_SPECTRAL_LUT and not self.lookup_tables:
            raise ValueError("continuous transport requires embedded LUT definitions")
        resolved_lanes: list[SpectralLaneDescriptor] = []
        for lane in self.lanes:
            weights = lane.sensor_weight_xyz
            if weights is None:
                if self.domain is TransportDomain.FIXED_SPECTRAL:
                    wavelength_nm = 299_792_458.0 / float(lane.frequency_hz) * 1.0e9
                    weights = native_display_rgb_weight(wavelength_nm)
                elif self.domain is TransportDomain.DISPLAY_RGB:
                    weights = tuple(1.0 if channel == int(lane.lane_id) else 0.0
                                    for channel in range(3))
                else:
                    weights = (1.0, 1.0, 1.0)
            lane = replace(lane, sensor_weight_xyz=tuple(float(v) for v in weights))
            resolved_lanes.append(lane)
            lane.validate(self.domain, len(self.lookup_tables))
        object.__setattr__(self, "lanes", tuple(resolved_lanes))

    @property
    def active_lane_count(self) -> int:
        return len(self.lanes)

    @property
    def compatibility_key(self) -> str:
        payload = json.dumps(self.mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def mapping(self) -> dict[str, Any]:
        return {
            "abi_version": self.abi_version, "domain": self.domain.value,
            "cohort_id": self.cohort_id,
            "lanes": [asdict(lane) for lane in self.lanes],
            "lookup_tables": [table.mapping() for table in self.lookup_tables],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TransportLaneTable":
        raw_domain = str(value["domain"])
        if raw_domain == "sampled_continuous_spectral":
            raise ValueError("ABI v1 sampled-continuous orders are not continuous and must be rebuilt")
        domain = TransportDomain(raw_domain)
        lane_values = []
        for raw_lane in value.get("lanes", ()):
            lane_value = dict(raw_lane)
            if lane_value.get("sensor_weight_xyz") is not None:
                lane_value["sensor_weight_xyz"] = tuple(
                    lane_value["sensor_weight_xyz"]
                )
            if lane_value.get("wavelength_bin_nm") is not None:
                lane_value["wavelength_bin_nm"] = tuple(
                    lane_value["wavelength_bin_nm"]
                )
            lane_values.append(SpectralLaneDescriptor(**lane_value))
        return cls(
            domain=domain, cohort_id=str(value["cohort_id"]),
            lanes=tuple(lane_values),
            lookup_tables=tuple(SpectralLookupTable.from_mapping(v) for v in value.get("lookup_tables", ())),
            abi_version=int(value.get("abi_version", TRANSPORT_ABI_VERSION)),
        )


@dataclass(frozen=True)
class TransportWorkContract:
    lane_table: TransportLaneTable
    material_profile_library_key: str = ""
    emitter_profile_library_key: str = ""
    sensor_profile_key: str = ""
    payload_variant: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        variant = int(self.payload_variant or self.lane_table.active_lane_count)
        if variant not in (1, 3, 8, 16, 32) or variant < self.lane_table.active_lane_count:
            raise ValueError("payload_variant must be 1,3,8,16,32 and cannot hold fewer slots than active lanes")
        object.__setattr__(self, "payload_variant", variant)

    def mapping(self) -> dict[str, Any]:
        return {"lane_table": self.lane_table.mapping(),
                "material_profile_library_key": self.material_profile_library_key,
                "emitter_profile_library_key": self.emitter_profile_library_key,
                "sensor_profile_key": self.sensor_profile_key,
                "payload_variant": self.payload_variant, "metadata": dict(self.metadata)}


def achromatic_lane_table(cohort_id: str = "achromatic") -> TransportLaneTable:
    return TransportLaneTable(TransportDomain.ACHROMATIC_SCALAR, cohort_id,
                              (SpectralLaneDescriptor(0, "luminance"),))


def rgb_lane_table(cohort_id: str = "display-rgb") -> TransportLaneTable:
    return TransportLaneTable(TransportDomain.DISPLAY_RGB, cohort_id,
                              tuple(SpectralLaneDescriptor(i, v) for i, v in enumerate(("red", "green", "blue"))))


def continuous_lut_lane_table(
    cohort_id: str, frequency_knots_hz: tuple[float, ...],
    proposal_density: tuple[float, ...], lane_count: int, *, seed: int = 0,
) -> TransportLaneTable:
    """Create lane signals for a continuous LUT; no frequency is sampled here."""
    del seed  # Sampling belongs to ray launch, never scene construction.
    payload = json.dumps({"frequency_knots_hz": frequency_knots_hz,
                          "density": proposal_density}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    key = "sha256:" + hashlib.sha256(payload).hexdigest()
    table = SpectralLookupTable(key, tuple(frequency_knots_hz), tuple(proposal_density))
    count = int(lane_count)
    if not 1 <= count <= MAX_TRANSPORT_LANES:
        raise ValueError("continuous lane count must be in [1, 32]")
    return TransportLaneTable(
        TransportDomain.CONTINUOUS_SPECTRAL_LUT, str(cohort_id),
        tuple(SpectralLaneDescriptor(i, f"spectral-lut:{key}:lane:{i}", lut_index=0)
              for i in range(count)),
        lookup_tables=(table,),
    )


def fixed_visible_lane_table(
    cohort_id: str, lane_count: int,
) -> TransportLaneTable:
    """Build a perceptually partitioned, display-colour-aware visible table."""

    bins = perceptual_visible_bins(int(lane_count))
    mean_width_nm = (
        VISIBLE_WAVELENGTH_MAX_NM - VISIBLE_WAVELENGTH_MIN_NM
    ) / len(bins)
    c = 299_792_458.0
    return TransportLaneTable(
        TransportDomain.FIXED_SPECTRAL,
        str(cohort_id),
        tuple(
            SpectralLaneDescriptor(
                lane_id=index,
                identity=f"visible-perceptual:{wavelength_nm:.3f}nm",
                frequency_hz=c / (wavelength_nm * 1.0e-9),
                sensor_weight_xyz=tuple(
                    value * ((upper_nm - lower_nm) / mean_width_nm)
                    for value in native_display_rgb_weight(wavelength_nm)
                ),
                wavelength_bin_nm=(lower_nm, upper_nm),
                quadrature_weight=(upper_nm - lower_nm) / mean_width_nm,
            )
            for index, (lower_nm, wavelength_nm, upper_nm) in enumerate(bins)
        ),
    )


__all__ = [
    "TRANSPORT_ABI_VERSION", "MAX_TRANSPORT_LANES", "TransportDomain",
    "SpectralLookupTable", "ResolvedRayFrequency", "SpectralLookupCache",
    "SpectralLaneDescriptor", "TransportLaneTable", "TransportWorkContract",
    "achromatic_lane_table", "rgb_lane_table", "continuous_lut_lane_table",
    "fixed_visible_lane_table", "perceptual_visible_bins",
    "perceptual_visible_wavelengths",
    "native_display_rgb_weight",
]
