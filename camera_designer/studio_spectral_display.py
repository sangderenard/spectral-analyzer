"""Display-only projections of the Studio fixed:8 sensor planes.

These are explicitly false-colour views. The transport wavelengths and saved
linear_spectral_exposure are never resampled or modified.
"""
from __future__ import annotations

import numpy as np

from thick_lens_focus_lab import _wavelength_to_rgb_weights

WAVELENGTHS_NM = np.array([650., 550., 450., 590., 365., 395., 850., 940.])
LOOKS = ("compressed", "nir_red", "uv_violet", "visible_reference")


def project_spectral_exposure(bands, look="compressed"):
    """Return linear RGB display data and a truthful projection receipt."""
    raw = np.asarray(bands)
    if raw.ndim < 2 or raw.shape[-1] != 8 or not np.issubdtype(raw.dtype, np.floating):
        raise ValueError("fixed:8 display requires eight floating sensor planes")
    if look not in LOOKS:
        raise ValueError("unknown fixed:8 spectral display look")
    if look == "compressed":
        # Preserve wavelength order while mapping the full 365–940 nm span
        # onto the existing camera software's 380–700 nm display palette.
        display_nm = 380. + (WAVELENGTHS_NM - 365.) * (320. / 575.)
        weights = _wavelength_to_rgb_weights(display_nm)
    else:
        weights = np.zeros((8, 3), dtype=np.float64)
        # Artist's channel-substitution views, with the visible transport
        # lines retained as a spatial base when an invisible band is promoted.
        weights[0, 0] = weights[1, 1] = weights[2, 2] = 1.
        if look == "nir_red":
            weights[6, 0] = 1.
            weights[7, 0] = .5
        elif look == "uv_violet":
            weights[4, 2] = 1.
            weights[5, 0] = weights[5, 2] = .5
    # Equal energy in the selected planes maps to neutral white; individual
    # band hues remain distinct. This is display white balance, not calibration.
    weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
    rgb = raw.astype(np.float64) @ weights
    receipt = {"id": "studio-fixed8-falsecolor-v1", "look": look,
               "transport_wavelengths_nm": WAVELENGTHS_NM.tolist(),
               "display_wavelengths_nm": display_nm.tolist() if look == "compressed" else None,
               "sensor_to_linear_display_rgb": weights.T.tolist(),
               "sensor_to_linear_srgb": weights.T.tolist(),
               "output_space": "linear display RGB, artistic false colour",
               "reference_white": "equal selected-band sensor exposure, display balanced to neutral",
               "display_transform": "selected false-colour projection, then ordinary Studio tone and sRGB transfer",
               "exposure_reference": "relative display white from selected projected luminance",
               "limits": "Wavelength compression and channel substitution are visualizations, not physical colorimetry."}
    return rgb, receipt
