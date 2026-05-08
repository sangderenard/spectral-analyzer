from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShaderCalibrationProfile:
    name: str
    temperature_k: float
    cat_ccm_matrix: np.ndarray
    gl_intensity_gain: float
    c_intensity_gain: float
    emitter_material: str
    receiver_material: str


def _kelvin_to_srgb_approx(temperature_k: float) -> np.ndarray:
    """Approximate illuminant sRGB for a Planckian source.

    This is a compact calibration approximation used only to derive
    diagonal white-balance gains, not for physically exact rendering.
    """
    t = max(1000.0, min(40000.0, float(temperature_k))) / 100.0

    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
        if t <= 19.0:
            b = 0.0
        else:
            b = 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * np.power(t - 60.0, -0.1332047592)
        g = 288.1221695283 * np.power(t - 60.0, -0.0755148492)
        b = 255.0

    srgb = np.array([
        np.clip(r, 0.0, 255.0),
        np.clip(g, 0.0, 255.0),
        np.clip(b, 0.0, 255.0),
    ], dtype=np.float32) / 255.0
    return srgb


def _diagonal_cat_gain_from_temperature(temperature_k: float) -> np.ndarray:
    """Return RGB diagonal CAT gain from illuminant temperature."""
    illum = _kelvin_to_srgb_approx(float(temperature_k))
    eps = 1e-6
    g = float(max(illum[1], eps))
    return np.array([
        g / float(max(illum[0], eps)),
        1.0,
        g / float(max(illum[2], eps)),
    ], dtype=np.float32)


def cat_ccm_matrix_from_temperature(temperature_k: float) -> np.ndarray:
    """Compose a default CAT/CCM matrix from illuminant temperature.

    Default policy:
        CAT = diag([G/R, 1, G/B])
        CCM = I
        M = CCM @ CAT

    Shader compensation equation:
        C_out = M * C_in
    """
    gain = _diagonal_cat_gain_from_temperature(temperature_k)
    mat = np.eye(3, dtype=np.float32)
    mat[0, 0] = gain[0]
    mat[1, 1] = gain[1]
    mat[2, 2] = gain[2]
    return mat


def _as_mat3(raw: Any, field_name: str) -> np.ndarray:
    m = np.asarray(raw, dtype=np.float32)
    if m.shape == (3, 3):
        return np.ascontiguousarray(m, dtype=np.float32)
    if m.ndim == 1 and m.shape[0] == 9:
        return np.ascontiguousarray(m.reshape(3, 3), dtype=np.float32)
    raise ValueError(f"{field_name} must be shape (3,3) or flat 9 values")


def _as_profile(name: str, raw: dict[str, Any], temperature_override: float | None = None) -> ShaderCalibrationProfile:
    temperature_k = float(raw.get("temperature_K", 6500.0) if temperature_override is None else temperature_override)
    if "cat_ccm_matrix" in raw:
        cat_ccm = _as_mat3(raw["cat_ccm_matrix"], f"profile {name}: cat_ccm_matrix")
    else:
        if "cat_matrix" in raw:
            cat = _as_mat3(raw["cat_matrix"], f"profile {name}: cat_matrix")
        else:
            cat = cat_ccm_matrix_from_temperature(temperature_k)
        if "ccm_matrix" in raw:
            ccm = _as_mat3(raw["ccm_matrix"], f"profile {name}: ccm_matrix")
        else:
            ccm = np.eye(3, dtype=np.float32)
        cat_ccm = np.ascontiguousarray(ccm @ cat, dtype=np.float32)

    return ShaderCalibrationProfile(
        name=name,
        temperature_k=temperature_k,
        cat_ccm_matrix=cat_ccm,
        gl_intensity_gain=float(raw.get("gl_intensity_gain", 1.0)),
        c_intensity_gain=float(raw.get("c_intensity_gain", 1.0)),
        emitter_material=str(raw.get("emitter_material", "tungsten_filament")),
        receiver_material=str(raw.get("receiver_material", "pearl_white_tile")),
    )


def load_shader_calibration_profile(profile_name: str, file_path: str,
                                    temperature_override: float | None = None) -> ShaderCalibrationProfile:
    with open(file_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    profiles = payload.get("profiles", {})
    if profile_name not in profiles:
        known = ", ".join(sorted(profiles.keys()))
        raise KeyError(f"unknown calibration profile '{profile_name}'. Known: {known}")
    return _as_profile(profile_name, profiles[profile_name], temperature_override)


def save_shader_calibration_gains(profile_name: str,
                                  file_path: str,
                                  gl_intensity_gain: float,
                                  c_intensity_gain: float) -> None:
    """Persist calibrated GL/C intensity gains to a profile JSON file.

    The write is atomic (`os.replace`) to avoid partial/corrupted files.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError(f"invalid calibration payload in {file_path}: expected object")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"invalid calibration payload in {file_path}: missing 'profiles' object")
    if profile_name not in profiles or not isinstance(profiles[profile_name], dict):
        known = ", ".join(sorted(profiles.keys()))
        raise KeyError(f"unknown calibration profile '{profile_name}'. Known: {known}")

    prof = profiles[profile_name]
    prof["gl_intensity_gain"] = float(gl_intensity_gain)
    prof["c_intensity_gain"] = float(c_intensity_gain)

    out_dir = os.path.dirname(os.path.abspath(file_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".calib.", suffix=".json", dir=out_dir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(payload, tmp, indent=2)
            tmp.write("\n")
        os.replace(tmp_path, file_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
