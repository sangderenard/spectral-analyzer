"""Discrete Studio transport basis, independent of the camera's film filters.

These are the existing fixed:3 line wavelengths, not sRGB primaries.  Keep
material transport in this basis and integrate the sensor response afterward.
The RGB-to-spectrum functions are the existing authored-material approximation,
not measured absorption spectra or a colorimetric profile.
"""

import numpy as np

FIXED3_WAVELENGTHS_NM = (650.0, 550.0, 450.0)


def uses_fixed3_transport(film):
    return tuple(getattr(film, "transport_wavelengths_nm", ())) == FIXED3_WAVELENGTHS_NM


def fixed3_response_matrix(layer_specs):
    """Rows are sensor filters; columns are fixed transport wavelengths."""
    frequency = 2.998e17 / np.asarray(FIXED3_WAVELENGTHS_NM)
    return np.asarray([
        max(0.0, spec["gain"]) * np.exp(-0.5 * (
            np.log2(frequency / spec["center_hz"]) / spec["width_oct"]
        ) ** 2)
        for spec in layer_specs
    ], dtype=np.float64)


def fixed3_nee_records(source_records):
    """Condition the existing camera NEE packet estimator on exact position.

    Its geometry/visibility depends only on position, not baked direction or
    wavelength. Replace the random single-band weight by its conditional mean
    in all three bands. Original row count/order and source-selection PDF stay
    unchanged. This removes wavelength-selection variance, not geometric noise
    or bias in the original baked spectrum/point-source approximation.

    Camera-only SourceRec ABI: xyz remains position; dir_kind.xyz contains
    mean packet-energy * amplitude-weight in each 650/550/450 nm band.
    Never submit this derived buffer to the forward ray tracer.
    """
    rows = np.asarray(source_records, np.float32)
    if rows.ndim != 2 or rows.shape[1] != 12 or not np.isfinite(rows).all():
        raise ValueError("NEE sources must be finite (N,12) records")
    out = rows.copy()
    if not len(rows):
        return out
    if np.any(rows[:,8] <= 0) or np.any(rows[:,10] < 0):
        raise ValueError("NEE source frequencies must be positive and energies nonnegative")
    _, group, counts = np.unique(rows[:,:3], axis=0, return_inverse=True, return_counts=True)
    frequencies = 2.998e17 / np.asarray(FIXED3_WAVELENGTHS_NM)
    band = np.argmin(np.abs(np.log(rows[:,8,None].astype(np.float64)/frequencies)),axis=1)
    sums = np.zeros((len(counts),3),np.float64)
    np.add.at(sums,(group,band),rows[:,10].astype(np.float64)*np.maximum(rows[:,3],1e-9))
    means = sums/counts[:,None]
    if np.any(means > np.finfo(np.float32).max):
        raise ValueError("NEE spectral mean exceeds float32 range")
    out[:,4:7] = means[group]
    return np.ascontiguousarray(out)


FIXED3_SPECTRAL_GLSL = r"""
uniform int uFixed3Transport;

float studio_band_frequency(int i) {
    return 2.998e17 / ((i == 0) ? 650.0 : ((i == 1) ? 550.0 : 450.0));
}

int studio_band_index(float frequency) {
    int result = 0;
    float distance = abs(log(frequency / studio_band_frequency(0)));
    for (int i = 1; i < 3; ++i) {
        float candidate = abs(log(frequency / studio_band_frequency(i)));
        if (candidate < distance) { distance = candidate; result = i; }
    }
    return result;
}

// The existing forward-material RGB spectral approximation, shared verbatim
// with the camera.  Film response and display swatches cannot change it.
float studio_material_weight(vec3 albedo, float freq_hz) {
    const float C_LIGHT = 299792458.0;
    float freq_r = C_LIGHT / 650.0e-9;
    float freq_g = C_LIGHT / 550.0e-9;
    float freq_b = C_LIGHT / 450.0e-9;
    const float wid_oct = 0.12;
    float lf = log(max(freq_hz, 1e-3)) / log(2.0);
    float wr = exp(-0.5 * pow((lf - log(freq_r) / log(2.0)) / wid_oct, 2.0));
    float wg = exp(-0.5 * pow((lf - log(freq_g) / log(2.0)) / wid_oct, 2.0));
    float wb = exp(-0.5 * pow((lf - log(freq_b) / log(2.0)) / wid_oct, 2.0));
    float den = wr + wg + wb;
    return (den > 1e-6) ? clamp(dot(vec3(wr, wg, wb), albedo) / den, 0.0, 1.0) : 1.0;
}

// Same additive q=12 Gaussian emission lines as RayOrder.from_lights_list
// and the saved Studio EmissionProfile.  Input amplitudes are linear.
float studio_emission_weight(vec3 amplitudes, float freq_hz) {
    float wavelength = 2.998e17 / max(freq_hz, 1e-3);
    vec3 centers = vec3(650.0, 550.0, 450.0);
    vec3 sigma = centers / (12.0 * 2.3548);
    vec3 z = (vec3(wavelength) - centers) / sigma;
    return dot(max(amplitudes, vec3(0.0)), exp(-0.5 * z * z));
}
"""
