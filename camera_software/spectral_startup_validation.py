"""Bounded native spectral checks run before ordinary render work starts."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping

import numpy as np


STARTUP_LANE_CONFIGURATIONS = (1, 3, 8, 16, 32)


@dataclass(frozen=True)
class SpectralStartupCase:
    lane_count: int
    passed: bool
    cpu_status: str
    gpu_status: str
    elapsed_ms: float
    measurements: Mapping[str, float] = field(default_factory=dict)
    detail: str = ""

    def mapping(self) -> dict[str, Any]:
        return {
            "lane_count": self.lane_count,
            "passed": self.passed,
            "cpu_status": self.cpu_status,
            "gpu_status": self.gpu_status,
            "elapsed_ms": self.elapsed_ms,
            "measurements": dict(self.measurements),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SpectralStartupReport:
    cases: tuple[SpectralStartupCase, ...]
    started_at_s: float
    elapsed_ms: float
    native_module: str

    @property
    def passed(self) -> bool:
        return bool(self.cases) and all(case.passed for case in self.cases)

    def mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "spectral_startup_validation",
            "passed": self.passed,
            "started_at_s": self.started_at_s,
            "elapsed_ms": self.elapsed_ms,
            "native_module": self.native_module,
            "cases": [case.mapping() for case in self.cases],
        }


def _enclosing_cube_triangles(half_extent: float = 10.0) -> np.ndarray:
    s = float(half_extent)
    points = np.asarray([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], np.float64)
    faces = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
    )
    return np.asarray(
        [[points[a], points[b], points[c]] for a, b, c in faces],
        np.float64,
    )


def _run_native_cpu_lane_case(lane_count: int) -> tuple[bool, dict[str, float], str]:
    """Trace exactly one native ray for every lane through a closed fixture."""

    import _spectral_kernels as native
    from ray_tracer_bridge import per_tri_spectral_to_mat_buf

    c = 299_792_458.0
    wavelengths_nm = np.linspace(700.0, 400.0, int(lane_count), dtype=np.float64)
    frequencies_hz = c / (wavelengths_nm * 1.0e-9)
    triangles = _enclosing_cube_triangles()
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    shape = (len(triangles), int(lane_count))
    reflectance = np.full(shape, 0.5, np.float64)
    zeros = np.zeros(shape, np.float64)
    mat_idx, mat_buf, mat_n_mats = per_tri_spectral_to_mat_buf(
        reflectance, zeros, zeros, frequencies_hz
    )
    tracer = native.RayTracer(
        len(triangles),
        np.ascontiguousarray(triangles.reshape(-1, 9)),
        np.ascontiguousarray(normals),
        np.ascontiguousarray(mat_idx),
        np.ascontiguousarray(mat_buf),
        int(mat_n_mats),
        np.ascontiguousarray(frequencies_hz),
        c,
        np.zeros(int(lane_count), np.float64),
    )
    traced = tracer.trace_with_frequency_sidecar(
        np.asarray([[0.0, 0.0, 0.0]], np.float64),
        np.asarray([[1.0, 0.0, 0.0]], np.float64),
        np.asarray([0.0], np.float64),
        n_rays=int(lane_count),
        max_bounces=1,
        min_amplitude=0.0,
        seed=1,
        out_cap=int(lane_count) + 4,
    )
    segments = np.asarray(traced["segs"], np.float32)
    sidecar = traced["freq_sidecar"]
    observed_ids = np.asarray(sidecar["band_id"], np.int32)
    observed_hz = np.asarray(sidecar["frequency_hz"], np.float64)
    observed_nm = np.asarray(sidecar["wavelength_nm"], np.float64)
    expected_ids = np.arange(int(lane_count), dtype=np.int32)
    finite = bool(
        np.all(np.isfinite(segments))
        and np.all(np.isfinite(observed_hz))
        and np.all(np.isfinite(observed_nm))
    )
    ids_exact = bool(np.array_equal(observed_ids, expected_ids))
    frequencies_exact = bool(np.array_equal(observed_hz, frequencies_hz))
    amplitudes_positive = bool(
        len(segments) == int(lane_count) and np.all(segments[:, 9] > 0.0)
    )
    passed = finite and ids_exact and frequencies_exact and amplitudes_positive
    measurements = {
        "rays_requested": float(lane_count),
        "segments_observed": float(len(segments)),
        "unique_lanes_observed": float(len(np.unique(observed_ids))),
        "minimum_amplitude": (
            float(np.min(segments[:, 9])) if len(segments) else 0.0
        ),
        "maximum_frequency_error_hz": (
            float(np.max(np.abs(observed_hz - frequencies_hz)))
            if len(observed_hz) == len(frequencies_hz) else float("inf")
        ),
    }
    detail = (
        "native CPU traced one ray per lane; IDs and frequency sidecar are exact"
        if passed else
        "native CPU lane identity, sidecar, finiteness, or amplitude check failed"
    )
    return passed, measurements, detail


def run_startup_spectral_sanity(
    lane_configurations: tuple[int, ...] = STARTUP_LANE_CONFIGURATIONS,
    *,
    gpu_probe: Callable[[int], bool] | None = None,
) -> SpectralStartupReport:
    """Run the small startup matrix before any ordinary work is scheduled.

    ``gpu_probe`` is deliberately injected by a host that owns a valid GPU
    context. Without one, GPU status is reported as ``not-run`` and is never
    inferred from the CPU result.
    """

    started_at_s = time.time()
    started = time.perf_counter()
    cases: list[SpectralStartupCase] = []
    for lane_count in lane_configurations:
        case_started = time.perf_counter()
        cpu_status = "failed"
        gpu_status = "not-run"
        measurements: dict[str, float] = {}
        try:
            cpu_passed, measurements, detail = _run_native_cpu_lane_case(
                int(lane_count)
            )
            cpu_status = "passed" if cpu_passed else "failed"
        except Exception as exc:  # retain a visible validation row on bad builds
            cpu_passed = False
            detail = f"native CPU probe raised {type(exc).__name__}: {exc}"
        if gpu_probe is not None:
            try:
                gpu_status = "passed" if bool(gpu_probe(int(lane_count))) else "failed"
            except Exception as exc:
                gpu_status = f"unavailable:{type(exc).__name__}"
        elapsed_ms = (time.perf_counter() - case_started) * 1000.0
        cases.append(SpectralStartupCase(
            lane_count=int(lane_count),
            passed=bool(cpu_passed) and gpu_status not in {"failed"},
            cpu_status=cpu_status,
            gpu_status=gpu_status,
            elapsed_ms=elapsed_ms,
            measurements=measurements,
            detail=detail,
        ))
    return SpectralStartupReport(
        cases=tuple(cases),
        started_at_s=started_at_s,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        native_module="_spectral_kernels.RayTracer.trace_with_frequency_sidecar",
    )


__all__ = [
    "STARTUP_LANE_CONFIGURATIONS", "SpectralStartupCase",
    "SpectralStartupReport", "run_startup_spectral_sanity",
]
