"""Versioned, per-exposure RGB film response for the marble photo worker.

These defaults preserve the existing ``em_rgb_fixed3`` preset; they are not
a calibrated film stock or standard color profile. Width is the existing
peak-normalized Gaussian's sigma in log2-frequency octaves, not nanometres.
This module deliberately has no renderer, NumPy, or OpenGL dependencies.
"""
from __future__ import annotations

from copy import deepcopy
import math


FILM_RESPONSE_VERSION = 1
FILM_CENTER_BOUNDS_NM = (380.0, 780.0)
FILM_SIGMA_BOUNDS_OCT = (0.01, 1.0)
DEFAULT_FILM_CENTERS_NM = (650.0, 550.0, 450.0)
DEFAULT_FILM_SIGMAS_OCT = (0.06, 0.06, 0.06)
_OMITTED = object()


def normalize_film_response(value=_OMITTED) -> dict:
    """Validate a complete v1 JSON object and return independent R/G/B lists.

    Only omission requests defaults. Explicit null, unknown fields, bools,
    strings, nonfinite numbers, unsupported versions and invalid lengths fail.
    Channels are independent; crossing or overlapping centers is permitted.
    """
    if value is _OMITTED:
        return {
            "version": FILM_RESPONSE_VERSION,
            "centers_nm": list(DEFAULT_FILM_CENTERS_NM),
            "sigmas_oct": list(DEFAULT_FILM_SIGMAS_OCT),
        }
    if not isinstance(value, dict):
        raise ValueError("film_response must be a v1 object")
    if set(value) != {"version", "centers_nm", "sigmas_oct"}:
        raise ValueError("film_response requires exactly version, centers_nm, and sigmas_oct")
    if type(value["version"]) is not int or value["version"] != FILM_RESPONSE_VERSION:
        raise ValueError("film_response.version must be integer 1")
    result = {"version": FILM_RESPONSE_VERSION}
    for key, bounds, unit in (
        ("centers_nm", FILM_CENTER_BOUNDS_NM, "nm"),
        ("sigmas_oct", FILM_SIGMA_BOUNDS_OCT, "octaves"),
    ):
        values = value[key]
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"film_response.{key} must contain exactly three R/G/B numbers")
        normalized = []
        for item in values:
            if type(item) not in (int, float) or not bounds[0] <= item <= bounds[1]:
                raise ValueError(
                    f"film_response.{key} must contain finite numbers in "
                    f"[{bounds[0]:g}, {bounds[1]:g}] {unit}"
                )
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(f"film_response.{key} must contain finite numbers")
            normalized.append(number)
        result[key] = normalized
    return result


def build_film_stack(preset, response: dict):
    """Copy the existing ordered RGB stack and change only its sensitivity.

    Both transport and camera must receive this same per-job object. Its
    fixed wavelength basis keeps material optics independent of the custom
    response, applied once at the sensor. Retain the renderer's existing
    2.998e14 Hz*um conversion so default sensitivity parameters are exact.
    """
    response = normalize_film_response(response)
    if list(preset.layers) != ["R", "G", "B"]:
        raise ValueError("marble RGB film preset must have ordered R/G/B layers")
    film = deepcopy(preset)
    film.transport_wavelengths_nm = (650.0, 550.0, 450.0)
    for index, name in enumerate(("R", "G", "B")):
        layer = film.layers[name]
        layer.response_center_hz = 2.998e14 / (response["centers_nm"][index] / 1000.0)
        layer.response_width_oct = response["sigmas_oct"][index]
    return film
