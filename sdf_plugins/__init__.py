from __future__ import annotations

from .drivers import BUILTIN_DRIVERS, SdfDriver


def list_sdf_drivers() -> tuple[str, ...]:
    return tuple(sorted(BUILTIN_DRIVERS.keys()))


def get_sdf_driver(name: str, strict: bool = True) -> SdfDriver:
    key = str(name).strip().lower()
    if key in BUILTIN_DRIVERS:
        return BUILTIN_DRIVERS[key]
    if strict:
        raise RuntimeError(
            f"Unknown SDF driver '{name}'. Available: {', '.join(list_sdf_drivers())}")
    return BUILTIN_DRIVERS["mixed"]
