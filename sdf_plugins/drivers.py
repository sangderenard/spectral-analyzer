from __future__ import annotations

from typing import Callable, Dict, Any

import numpy as np

SdfDriver = Callable[..., Dict[str, Any]]


def _sphere_driver(*, tri_id: int, tri_vertices: np.ndarray, params: dict) -> Dict[str, Any]:
    """Native SDF sphere driver payload (radius in metres)."""
    radius = max(float(params.get("sphere_radius_m", 0.12)), 1.0e-6)
    margin = max(0.0, float(params.get("neighborhood_margin_uv", 8.0e-2)))
    return {"kind": "sdf_sphere", "coeffs": np.asarray([radius, margin], dtype=np.float64)}


def _saddle_driver(*, tri_id: int, tri_vertices: np.ndarray, params: dict) -> Dict[str, Any]:
    """Native SDF saddle driver payload (amplitude in metres)."""
    amp = float(params.get("saddle_amplitude_m", 2.0e-3))
    margin = max(0.0, float(params.get("neighborhood_margin_uv", 8.0e-2)))
    return {"kind": "sdf_saddle", "coeffs": np.asarray([amp, margin], dtype=np.float64)}


def _mixed_driver(*, tri_id: int, tri_vertices: np.ndarray, params: dict) -> Dict[str, Any]:
    """Alternates sphere/saddle per triangle for richer first-frame topology."""
    if (int(tri_id) % 2) == 0:
        return _sphere_driver(tri_id=tri_id, tri_vertices=tri_vertices, params=params)
    return _saddle_driver(tri_id=tri_id, tri_vertices=tri_vertices, params=params)


BUILTIN_DRIVERS: dict[str, SdfDriver] = {
    "sphere": _sphere_driver,
    "saddle": _saddle_driver,
    "mixed": _mixed_driver,
}
