"""
Calibrate all emission profiles to real visible radiance values.

This script computes the equivalent visible radiance for each builtin emission profile,
accounting for their spectral distributions and power scalings. The radiance values
are then used in the renderer with the solid-angle model for accurate illumination.
"""

import numpy as np
import math

# Physical constants
PLANCK_H = 6.62607015e-34      # J·s
LIGHT_C = 299792458.0          # m/s
BOLTZMANN_K = 1.380649e-23     # J/K

# Import CIE CMF and wavelengths (from material_db.py)
from material_db import _WL_NM, _CMF_Y, _CMF_Y_NORM, ParametricSpread, _spd_term_to_array

def planck_radiance_from_temperature(T_K, emissivity=0.95):
    """Compute visible radiance from Planck's law at temperature T."""
    wl_m = _WL_NM * 1e-9
    numerator = 2.0 * PLANCK_H * LIGHT_C**2 / (wl_m**5)
    exponent = (PLANCK_H * LIGHT_C) / (wl_m * BOLTZMANN_K * T_K)
    exponent = np.clip(exponent, 0.0, 200.0)
    denominator = np.expm1(exponent)
    denominator = np.maximum(denominator, 1e-30)
    L_lambda = numerator / denominator
    
    dl = 5.0  # nm
    L_vis = float(np.sum(L_lambda * _CMF_Y) * dl) * emissivity
    return L_vis


def spd_visible_radiance(spd_array, power_scale=1.0):
    """Compute visible radiance from SPD and power scale."""
    dl = 5.0
    spd = np.asarray(spd_array, np.float64)
    weighted = spd * _CMF_Y
    L_vis_integrated = float(np.sum(weighted) * dl)
    if _CMF_Y_NORM > 0.0:
        L_vis = power_scale * L_vis_integrated / _CMF_Y_NORM
    else:
        L_vis = 0.0
    return L_vis


print("=" * 80)
print("EMISSION PROFILE CALIBRATION")
print("=" * 80)

# Profile 0: basic_led_display
print("\n--- Profile 0: basic_led_display ---")
print("Original: total_power_W = 0.35 W")
print("Spectrum: blue peak 450nm + cyan shoulder 510nm")
spd_led = np.zeros(len(_WL_NM))
for term in [
    ParametricSpread(amp=1.0, center=450.0, q=18.0),
    ParametricSpread(amp=0.30, center=510.0, q=5.0),
]:
    spd_led += _spd_term_to_array(term)
L_vis_led = spd_visible_radiance(spd_led, power_scale=0.35)
print(f"Computed visible radiance: {L_vis_led:.6e} W/(m²·sr)")
print(f"Recommend: radiance_W_sr_m2={L_vis_led:.6e}")

# Profile 1: missing_material_hazard
print("\n--- Profile 1: missing_material_hazard ---")
print("Original: total_power_W = 2.6 W")
print("Spectrum: warm amber, peak 590nm + secondary 620nm")
spd_hazard = np.zeros(len(_WL_NM))
for term in [
    ParametricSpread(amp=1.0, center=590.0, q=12.0),
    ParametricSpread(amp=0.25, center=620.0, q=10.0),
]:
    spd_hazard += _spd_term_to_array(term)
L_vis_hazard = spd_visible_radiance(spd_hazard, power_scale=2.6)
print(f"Computed visible radiance: {L_vis_hazard:.6e} W/(m²·sr)")
print(f"Recommend: radiance_W_sr_m2={L_vis_hazard:.6e}")
# Alternatively, approximate as thermal source
T_equivalent = 2100  # amber is warm
L_vis_hazard_thermal = planck_radiance_from_temperature(T_equivalent, emissivity=0.95)
print(f"Or as equivalent {T_equivalent}K thermal source: {L_vis_hazard_thermal:.6e} W/(m²·sr)")

# Profile 2: material_database_failure
print("\n--- Profile 2: material_database_failure ---")
print("Original: total_power_W = 3.2 W")
print("Spectrum: neon green, peak 532nm + shoulder 505nm")
spd_failure = np.zeros(len(_WL_NM))
for term in [
    ParametricSpread(amp=1.0, center=532.0, q=18.0),
    ParametricSpread(amp=0.30, center=505.0, q=10.0),
]:
    spd_failure += _spd_term_to_array(term)
L_vis_failure = spd_visible_radiance(spd_failure, power_scale=3.2)
print(f"Computed visible radiance: {L_vis_failure:.6e} W/(m²·sr)")
print(f"Recommend: radiance_W_sr_m2={L_vis_failure:.6e}")

# Profile 3: construction_borosilicate
print("\n--- Profile 3: construction_borosilicate ---")
print("Original: total_power_W = 0.06 W")
print("Spectrum: cool blue-white, peak 480nm + shoulder 540nm")
spd_boro = np.zeros(len(_WL_NM))
for term in [
    ParametricSpread(amp=1.0,  center=480.0, q=10.0),
    ParametricSpread(amp=0.45, center=540.0, q=6.0),
]:
    spd_boro += _spd_term_to_array(term)
L_vis_boro = spd_visible_radiance(spd_boro, power_scale=0.06)
print(f"Computed visible radiance: {L_vis_boro:.6e} W/(m²·sr)")
print(f"Recommend: radiance_W_sr_m2={L_vis_boro:.6e}")
# Alternatively, as ~4500K thermal source
T_equivalent_boro = 4500
L_vis_boro_thermal = planck_radiance_from_temperature(T_equivalent_boro, emissivity=0.95)
print(f"Or as equivalent {T_equivalent_boro}K thermal source: {L_vis_boro_thermal:.6e} W/(m²·sr)")

# Profile 4: tungsten_2800k (now physical)
print("\n--- Profile 4: tungsten_2800k ---")
print("Original: total_power_W = 1500.0 (fake-scale)")
print("Now: temperature_K = 2800.0, emissivity = 0.95 (physical)")
L_vis_tungsten = planck_radiance_from_temperature(2800.0, emissivity=0.95)
print(f"Computed visible radiance (Planck): {L_vis_tungsten:.6e} W/(m²·sr)")
print(f"Note: This is baked at registration time from temperature_K")

print("\n" + "=" * 80)
print("SUMMARY: Update builtin profiles in material_db.py")
print("=" * 80)

print(f"""
Replace the profiles with these values:

Index 0: basic_led_display
    radiance_W_sr_m2 = {L_vis_led:.6e}

Index 1: missing_material_hazard
    Option A (explicit): radiance_W_sr_m2 = {L_vis_hazard:.6e}
    Option B (thermal):  temperature_K = 2100, emissivity = 0.95

Index 2: material_database_failure
    radiance_W_sr_m2 = {L_vis_failure:.6e}

Index 3: construction_borosilicate
    Option A (explicit): radiance_W_sr_m2 = {L_vis_boro:.6e}
    Option B (thermal):  temperature_K = 4500, emissivity = 0.95

Index 4: tungsten_2800k
    temperature_K = 2800
    emissivity = 0.95
    (radiance computed from Planck law at registration time)

All profiles will now be physically calibrated and use the solid-angle model
for accurate inverse-square falloff without arbitrary power scaling.
""")

print("\n" + "=" * 80)
print("RENDERED INTENSITY MODEL")
print("=" * 80)

print("""
In _derive_gl_lights(), the intensity is now computed as:

    intensity = L_vis × Ω(r)

Where:
    L_vis = visible radiance from profile_to_radiance() [W/(m²·sr)]
    Ω(r)  = solid angle subtended by emitter at distance r [sr]
    
For a sphere of radius R at distance r:
    Ω(r) ≈ π(R/r)² for r >> R (far field)

This replaces the fake-scale model:
    OLD: intensity = emitter_area  (arbitrary)
    NEW: intensity = L_vis × Ω     (physical)

Result: consistent inverse-square falloff and correct brightness across all emitters.
""")
