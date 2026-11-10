"""Reference colorimetry for Marble Studio's three discrete transport lines.

This is not a measured-camera ICC profile or a broadband spectral model.
Sensor RGB is unmixed before CIE XYZ conversion. A fixed Bradford adaptation
maps the Studio's equal-power three-line reference white to sRGB D65. Neither
image statistics nor artistic retouch settings participate in this matrix.
"""
from __future__ import annotations

import numpy as np

from camera_software.sensor_back import ColorScienceProfile
from camera_designer.film_response import normalize_film_response
from camera_designer.spectral_sensor_basis import fixed3_response_matrix

PROFILE_ID = "marble-fixed3-cie1931-d65-v1"
PREVIEW_PATCH_NAMES = ("White", "Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Gray 20%")
PREVIEW_SENSOR_RGB = ((1., 1., 1.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                      (1., 1., 0.), (1., 0., 1.), (0., 1., 1.), (.2, .2, .2))
MAX_RESPONSE_CONDITION = 1000.0
MIN_RESPONSE_SINGULAR_VALUE = 1e-6

# Three unmodified rows from CIE 018:2019, Table 6, ordered 650/550/450nm.
# CIE 2019, DOI 10.25039/CIE.DS.xvudnb9b, CC BY-SA 4.0 (data excerpt).
# https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv
# Full source SHA256 verified: fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1
CIE_FIXED3_XYZ = np.array([
    [.283500000000, .433449900000, .336200000000],
    [.107000000000, .994950100000, .038000000000],
    [.000000000000, .008749999000, 1.772110000000],
], dtype=np.float64)

# IEC sRGB primaries and D65 chromaticity, as published by the ICC registry:
# https://registry.color.org/rgb-registry/srgb
D65_XYZ = np.array([.3127 / .3290, 1.0, (1.0 - .3127 - .3290) / .3290])
_PRIMARY_XY = np.array([[.64, .33], [.30, .60], [.15, .06]])
_PRIMARY_XYZ = np.stack((_PRIMARY_XY[:, 0] / _PRIMARY_XY[:, 1],
                         np.ones(3),
                         (1 - _PRIMARY_XY.sum(axis=1)) / _PRIMARY_XY[:, 1]))
SRGB_TO_XYZ = _PRIMARY_XYZ * np.linalg.solve(_PRIMARY_XYZ, D65_XYZ)
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)
SRGB_LUMA = SRGB_TO_XYZ[1]
_BRADFORD = np.array([[.8951, .2664, -.1614],
                      [-.7502, 1.7135, .0367],
                      [.0389, -.0685, 1.0296]])
_REFERENCE_Y = CIE_FIXED3_XYZ[1].sum()
_REFERENCE_WHITE = CIE_FIXED3_XYZ.sum(axis=1) / _REFERENCE_Y
_ADAPTATION = np.linalg.solve(
    _BRADFORD, np.diag((_BRADFORD @ D65_XYZ) / (_BRADFORD @ _REFERENCE_WHITE)) @ _BRADFORD)
_LINE_TO_XYZ = _ADAPTATION @ (CIE_FIXED3_XYZ / _REFERENCE_Y)


def response_matrix(response=None):
    response = normalize_film_response() if response is None else normalize_film_response(response)
    return fixed3_response_matrix([
        {"center_hz": 2.998e17 / center, "width_oct": sigma, "gain": 1.0}
        for center, sigma in zip(response["centers_nm"], response["sigmas_oct"])
    ])


def build_output_profile(response=None):
    """Return the existing ColorScienceProfile and serializable provenance.

    Rank-deficient/unstable sensitivities cannot reconstruct three colors.
    The explicit fallback estimates only neutral-white intensity, with no
    invented hue or chromatic matrix inversion. Its scalar normalization can
    amplify weak responses; this is not recovered spectral luminance.
    Original sensor measurements remain intact.
    """
    sensor = response_matrix(response)
    singular = np.linalg.svd(sensor, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    fallback = condition > MAX_RESPONSE_CONDITION or singular[-1] < MIN_RESPONSE_SINGULAR_VALUE
    if fallback:
        white_response = sensor @ np.ones(3)
        denominator = float(white_response @ white_response)
        neutral = white_response / denominator if denominator > 1e-12 else np.zeros(3)
        sensor_to_xyz = D65_XYZ[:, None] * neutral[None, :]
    else:
        sensor_to_xyz = np.linalg.solve(sensor.T, _LINE_TO_XYZ.T).T
    matrix = XYZ_TO_SRGB @ sensor_to_xyz
    profile = ColorScienceProfile.identity()
    profile.color_matrix = matrix
    metadata = {
        "id": PROFILE_ID,
        "observer": "CIE 1931 2 degree",
        "source": "https://doi.org/10.25039/CIE.DS.xvudnb9b",
        "transport_wavelengths_nm": [650.0, 550.0, 450.0],
        "reference_white": "equal-power three-line reference, Bradford adapted to D65",
        "output_space": "linear sRGB D65",
        "condition_number": condition if np.isfinite(condition) else None,
        "condition_limit": MAX_RESPONSE_CONDITION,
        "minimum_singular_value": float(singular[-1]),
        "neutral_fallback": bool(fallback),
        "fallback_intensity": ("least-squares neutral-white intensity, not spectral luminance"
                               if fallback else None),
        "warning": ("Film response cannot stably recover color; neutral fallback is shown."
                    if fallback else None),
        "sensor_to_xyz": sensor_to_xyz.tolist(),
        "sensor_to_linear_srgb": matrix.tolist(),
        "display_transform": "separate selectable luminance tone and normalization; neutral-axis gamut compression; sRGB transfer once",
        "exposure_reference": "default: 99th percentile of positive profiled luminance; manual white is a relative display reference, not radiometric calibration",
        "limits": "Three discrete lines only; reference profile, not measured camera calibration.",
    }
    return profile, metadata


def sensor_to_linear_srgb(sensor, response=None):
    raw = np.asarray(sensor)
    if raw.shape[-1] != 3 or not np.issubdtype(raw.dtype, np.floating):
        raise ValueError("sensor exposure must be a floating array with three channels")
    profile, metadata = build_output_profile(response)
    return profile.apply_matrix(raw), metadata


def normalize_display_params(params=None):
    """Validate the display-only controls, independent of sensor calibration."""
    params = params or {}
    normalization = params.get("normalization", "auto99")
    tone_curve = params.get("tone_curve", "exponential")
    if normalization not in ("auto99", "manual", "percentile", "specular"):
        raise ValueError("normalization must be auto99, manual, percentile or specular")
    if tone_curve not in ("exponential", "linear"):
        raise ValueError("tone_curve must be exponential or linear")
    display_look = params.get("display_look", "reference")
    if display_look not in ("reference", "color_priority"):
        raise ValueError("display_look must be reference or color_priority")
    if normalization == "specular":
        display_look = "reference"  # the explicitly requested white-clipping preset
        tone_curve = "linear"
    if isinstance(params.get("normalization_white"), bool):
        raise ValueError("normalization_white must be a number, not a boolean")
    try:
        white = float(params.get("normalization_white", 1.0))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("normalization_white must be a finite number in [1e-6,1e12]") from None
    if not np.isfinite(white) or not 1e-6 <= white <= 1e12:
        raise ValueError("normalization_white must be a finite number in [1e-6,1e12]")
    result = {"normalization": normalization, "normalization_white": white, "tone_curve": tone_curve, "display_look": display_look}
    enabled = params.get("firefly_filter", False)
    if type(enabled) is not bool:
        raise ValueError("firefly_filter must be a boolean")
    result["firefly_filter"] = enabled
    for name, low, high, default in (("saturation", 0., 2., 1.),
                                     ("vibrance", -1., 1., 0.),
                                     ("black_point", 0., .5, 0.),
                                     ("firefly_threshold", 2., 64., 8.),
                                     ("firefly_strength", 0., 1., 1.),
                                     ("normalization_percentile", 1., 100.,
                                      95. if normalization == "specular" else 99.)):
        value = params.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a number, not a boolean")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a finite number in [{low},{high}]") from None
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be a finite number in [{low},{high}]")
        result[name] = value
    return result


def display_transform_metadata(params=None, effective_white=None):
    result = normalize_display_params(params)
    result.update({
        "effective_white": effective_white,
        "tone_formula": ("1-exp(-2*Y/white)" if result["tone_curve"] == "exponential"
                         else "clip(Y/white,0,1)"),
        "transfer": "standard sRGB once, followed by optional artistic gamma/intensity",
        "gamut_mapping": "neutral-axis compression at mapped luminance",
        "white_units": "relative profiled linear luminance before artistic gains",
        "black_point_formula": "x=max((Y/white-black_point)/(1-black_point),0); tone uses x",
        "chroma_formula": "chroma *= saturation*(1+vibrance*(1-C)); C=clip((maxRGB-minRGB)/maxAbsRGB,0,1)",
        "creative_controls": "Artistic linear-sRGB adjustments; not calibration or perceptually uniform saturation",
        "firefly_method": "if interior luminance exceeds threshold times the maximum of 8 neighbors, blend toward that maximum by strength; RGB ratios preserved; after gains, before tone",
        "firefly_warning": "May reduce genuine isolated glints or sharp highlight peaks. Brightest-neighbor comparison protects connected detail; borders, raw HDR and white reference stay unchanged.",
        "effective_percentile": (None if result["normalization"] == "manual" else
                                  99. if result["normalization"] == "auto99" else result["normalization_percentile"]),
        "specular_clip_mode": result["normalization"] == "specular",
        "highlight_policy": ("Linear normalized luminance at or above white is clipped to display white; lower values remain graded."
                             if result["tone_curve"] == "linear" else "Exponential luminance shoulder approaches display white."),
    })
    if result["black_point"] != 0:
        result["tone_formula"] = ("1-exp(-2*x)" if result["tone_curve"] == "exponential"
                                  else "clip(x,0,1)")
    if result["display_look"] == "color_priority":
        result["gamut_mapping"] = "scene-linear negative-channel neutral compression; ratio-preserving maximum-channel shoulder"
        result["tone_formula"] = "map maxRGB/white with selected curve, scale all channels together after black point"
        result["highlight_policy"] = "Color-priority custom look preserves positive RGB ratios through the shoulder; displayed luminance may be lower. Not Canon or ACES."
    return result


def suppress_fireflies(rgb, controls, luma_weights=SRGB_LUMA):
    """Conservative opt-in spatial cap, not proof that a pixel is noise.

    A real one-pixel light and a noisy sample can be indistinguishable. Use
    the brightest of all eight neighbors rather than a median to preserve
    connected small highlights. Never filter borders or mutate HDR input.
    """
    if (not controls["firefly_filter"] or controls["firefly_strength"] == 0
            or rgb.ndim != 3 or min(rgb.shape[:2]) < 3):
        return rgb
    y = np.maximum(rgb @ luma_weights, 0.0)
    center = y[1:-1, 1:-1]
    neighbor_max = np.zeros_like(center)
    h, w = center.shape
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                np.maximum(neighbor_max, y[dy:dy+h, dx:dx+w], out=neighbor_max)
    isolated = center > neighbor_max * controls["firefly_threshold"]
    ratio = np.divide(neighbor_max, center, out=np.ones_like(center),
                      where=isolated & (center > 0))
    scale = (1.0 - controls["firefly_strength"]) + controls["firefly_strength"] * ratio
    filtered = rgb.copy()
    filtered[1:-1, 1:-1] *= scale[..., None]
    return filtered


def display_white_reference(linear, params=None):
    """Relative display white from unfiltered profiled HDR, before gains."""
    controls = normalize_display_params(params)
    if controls["normalization"] == "manual":
        return controls["normalization_white"]
    y = np.maximum(np.asarray(linear, dtype=np.float64) @ SRGB_LUMA, 0.0)
    positive = y[y > 0]
    percentile = 99. if controls["normalization"] == "auto99" else controls["normalization_percentile"]
    return max(float(np.percentile(positive, percentile)) if len(positive) else 0.0, 1e-6)


def color_priority_map(rgb, controls, white, luma_weights):
    """Optional custom display look; never changes sensor/profile measurements.

    The old fixed-Y gamut projection makes bright green pale and cancels a
    saturation increase once a channel reaches the display boundary. Here
    negative channels are first compressed toward neutral in scene-linear RGB;
    one scalar shoulder then maps the maximum channel. Positive RGB ratios
    survive that shoulder. This trades displayed luminance for color, not a
    colorimetric/Canon/ACES rendering intent. Neutral ramps remain unchanged.
    """
    y = np.maximum(rgb @ luma_weights, 0.0)
    chroma = rgb - y[..., None]
    spread = rgb.max(axis=-1) - rgb.min(axis=-1)
    magnitude = np.abs(rgb).max(axis=-1)
    colorfulness = np.clip(np.divide(spread, magnitude,
        out=np.zeros_like(spread), where=magnitude > 0), 0.0, 1.0)
    factor = controls["saturation"] * (1.0 + controls["vibrance"] * (1.0 - colorfulness))
    chroma = chroma * factor[..., None]
    low = chroma.min(axis=-1)
    compression = np.minimum(1.0, np.divide(y, -low,
        out=np.ones_like(y), where=low < 0))
    positive = np.maximum(y[..., None] + compression[..., None] * chroma, 0.0)
    normalized_y = y / white
    black_adjusted = np.maximum((normalized_y - controls["black_point"]) /
                                (1.0 - controls["black_point"]), 0.0)
    positive *= np.divide(black_adjusted, normalized_y,
        out=np.zeros_like(y), where=normalized_y > 0)[..., None] / white
    peak = positive.max(axis=-1)
    mapped_peak = (-np.expm1(-2.0 * peak) if controls["tone_curve"] == "exponential"
                   else np.clip(peak, 0.0, 1.0))
    return np.clip(positive * np.divide(mapped_peak, peak,
        out=np.zeros_like(peak), where=peak > 0)[..., None], 0.0, 1.0)


def render_linear_srgb(linear, params=None):
    """One display transform, shared by preview and live retouch.

    Default automatic exposure retains the existing 99th-percentile convention. Tone
    mapping scales by luminance, not three independent nonlinear curves.
    Out-of-gamut colors are compressed toward neutral at the same luminance;
    this is a documented display choice, not an ICC perceptual rendering intent.
    """
    params = params or {}
    controls = normalize_display_params(params)
    rgb = np.asarray(linear, dtype=np.float64)
    white = display_white_reference(rgb, controls)
    exposure = 2.0 ** float(params.get("exposure", 0.0))
    wb = 2.0 ** float(params.get("whitebalance", 0.0))
    tint = 2.0 ** float(params.get("tint", 0.0))
    gain = np.array([float(params.get("red", 1.0)) * wb,
                     float(params.get("green", 1.0)) / tint,
                     float(params.get("blue", 1.0)) / wb]) * exposure
    rgb = rgb * gain
    # Keep the original white reference: a local cleanup must not silently
    # re-expose the rest of the image when automatic normalization is used.
    rgb = suppress_fireflies(rgb, controls)
    luminance = np.maximum(rgb @ SRGB_LUMA, 0.0)
    target_y = (-np.expm1(-luminance * (2.0 / white))
                if controls["tone_curve"] == "exponential"
                else np.clip(luminance / white, 0.0, 1.0))
    if controls["black_point"] != 0:
        normalized = np.maximum((luminance / white - controls["black_point"]) /
                                (1.0 - controls["black_point"]), 0.0)
        target_y = (-np.expm1(-2.0 * normalized) if controls["tone_curve"] == "exponential"
                    else np.clip(normalized, 0.0, 1.0))
    scaled = rgb * np.divide(target_y, luminance, out=np.zeros_like(luminance),
                            where=luminance > 0)[..., None]
    chroma = scaled - target_y[..., None]
    if controls["saturation"] != 1.0 or controls["vibrance"] != 0.0:
        # An explicitly artistic rule in linear sRGB, not a calibrated
        # perceptual metric. Chroma scaling preserves mapped luminance.
        spread = scaled.max(axis=-1) - scaled.min(axis=-1)
        magnitude = np.abs(scaled).max(axis=-1)
        colorfulness = np.clip(np.divide(spread, magnitude,
            out=np.zeros_like(spread), where=magnitude > 0), 0.0, 1.0)
        factor = controls["saturation"] * (1.0 + controls["vibrance"] * (1.0 - colorfulness))
        chroma = chroma * factor[..., None]
    low, high = chroma.min(axis=-1), chroma.max(axis=-1)
    saturation = np.ones_like(target_y)
    saturation = np.minimum(saturation, np.divide(target_y, -low,
        out=np.ones_like(target_y), where=low < 0))
    saturation = np.minimum(saturation, np.divide(1.0 - target_y, high,
        out=np.ones_like(target_y), where=high > 0))
    mapped = np.clip(target_y[..., None] + saturation[..., None] * chroma, 0.0, 1.0)
    if controls["display_look"] == "color_priority":
        mapped = color_priority_map(rgb, controls, white, SRGB_LUMA)
    display = ColorScienceProfile.srgb_standard().apply_gamma(mapped)
    display = np.clip(display, 0.0, 1.0) ** (1.0 / max(float(params.get("gamma", 1.0)), 1e-3))
    display = np.clip(display * max(float(params.get("intensity", 1.0)), 0.0), 0.0, 1.0)
    return np.rint(display * 255).astype(np.uint8), white


def preview_sensor_samples(samples=PREVIEW_SENSOR_RGB, response=None, params=None):
    """Direct CPU-only sensor-to-output test; no scene, exposure, or GPU.

    The input swatches are declared linear sensor values. The left-panel
    reference uses only sRGB encoding; the right uses the exact photo path.
    All swatches share one display exposure reference, not one per color.
    """
    raw = np.asarray(samples, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 or not 1 <= len(raw) <= 4096:
        raise ValueError("sensor_samples must contain 1..4096 RGB triplets")
    if not np.all(np.isfinite(raw)) or np.any(raw < 0) or np.any(raw > 16):
        raise ValueError("sensor_samples must be finite linear values in [0,16]")
    output, metadata = sensor_to_linear_srgb(raw[None, ...], response)
    display, white = render_linear_srgb(output, params)
    reference = ColorScienceProfile.srgb_standard().apply_gamma(np.clip(raw, 0, 1))
    return {"ok": True, "input_space": "linear sensor RGB", "sensor_rgb": raw.tolist(),
            "patch_names": list(PREVIEW_PATCH_NAMES) if np.array_equal(raw, PREVIEW_SENSOR_RGB)
                           else [str(i + 1) for i in range(len(raw))],
            "input_display_rgb8": np.rint(np.clip(reference, 0, 1) * 255).astype(np.uint8).tolist(),
            "output_display_rgb8": display[0].tolist(),
            "linear_srgb": output[0].tolist(),
            "display_white": white, "profile": metadata,
            "display_transform": display_transform_metadata(params, white)}


def main():
    """Bounded JSON stdin/stdout adapter for the photo bridge's test panel."""
    import json
    import sys
    request = json.load(sys.stdin)
    result = preview_sensor_samples(request.get("sensor_samples", PREVIEW_SENSOR_RGB),
                                    request.get("film_response"), request.get("params"))
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
