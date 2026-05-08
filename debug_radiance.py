"""
Debug: what should the Planck radiance actually be?
"""
import numpy as np

# Physical constants
PLANCK_H = 6.62607015e-34      # J·s
LIGHT_C = 299792458.0          # m/s
BOLTZMANN_K = 1.380649e-23     # J/K

# Test wavelengths and CIE Y
WL_NM = np.linspace(380, 2500, 433)
wl_m = WL_NM * 1e-9

# Approximate CIE Y
def cie_y_approx(wl_nm):
    x = (wl_nm - 555.0) / 100.0
    return np.exp(-0.5 * x**2) * (wl_nm < 780) * (wl_nm > 360)

CIE_Y = cie_y_approx(WL_NM)
CIE_Y_NORM = np.sum(CIE_Y) * 5.0  # normalize by step size

print(f"CIE_Y range: {CIE_Y.min():.6f} to {CIE_Y.max():.6f}")
print(f"CIE_Y_NORM: {CIE_Y_NORM:.6f}")
print(f"CIE_Y sum × 5nm: {np.sum(CIE_Y) * 5.0:.6f}")

# Planck at 2800K
T = 2800.0
numerator = 2.0 * PLANCK_H * LIGHT_C**2 / (wl_m**5)
exponent = (PLANCK_H * LIGHT_C) / (wl_m * BOLTZMANN_K * T)
exponent = np.clip(exponent, 0.0, 200.0)
denominator = np.expm1(exponent)
denominator = np.maximum(denominator, 1e-30)
L_lambda = numerator / denominator

print(f"\nPlanck spectral radiance at {T}K:")
print(f"  L_λ range: {L_lambda.min():.6e} to {L_lambda.max():.6e} W/(m³·sr)")
print(f"  Peak at ~760nm: {L_lambda[np.argmax(L_lambda)]:.6e} W/(m³·sr)")

# Integrate
dl = 5.0
L_vis_raw = np.sum(L_lambda * CIE_Y) * dl
print(f"\nRaw integration:")
print(f"  ∫ L_λ × y(λ) dλ = {L_vis_raw:.6e} W/(m²·sr)")

# Normalized like profile_to_rgb
L_vis_normalized = L_vis_raw / CIE_Y_NORM if CIE_Y_NORM > 0 else 0
print(f"\nNormalized by CIE_Y_NORM:")
print(f"  L_vis = {L_vis_normalized:.6e} W/(m²·sr)")

# With emissivity
epsilon = 0.95
L_vis_final = L_vis_normalized * epsilon
print(f"\nWith emissivity {epsilon}:")
print(f"  L_vis × ε = {L_vis_final:.6e} W/(m²·sr)")

# What about without normalizing?
L_vis_unnormalized = L_vis_raw * epsilon
print(f"\nWithout normalizing by CIE_Y_NORM:")
print(f"  L_vis_raw × ε = {L_vis_unnormalized:.6e} W/(m²·sr)")

print("\n" + "="*70)
print("Physical check: Stefan-Boltzmann radiance")
print("="*70)

# Total radiance from Stefan-Boltzmann: M = σT⁴ (total power per unit area)
# Radiance (per steradian): L = M/π
sigma = 5.67e-8  # W/(m²·K⁴)
M = sigma * (T ** 4)
L_total = M / np.pi
print(f"Stefan-Boltzmann total radiance:")
print(f"  M = σT⁴ = {M:.6e} W/m²")
print(f"  L = M/π ≈ {L_total:.6e} W/(m²·sr)")
print(f"  (This is order-of-magnitude check for visible radiance)")

print("\n" + "="*70)
print("What intensity do we need in the shader?")
print("="*70)

print("""
Shader formula:
  Lcol = rgb × (intensity / r²)
  
Old system:
  intensity = emitter_area (in world units, typically meters)
  For tiny bulb: area ~ 1e-5 to 1e-4 m²
  At r=1m: Lcol = [0.9, 0.6, 0.2] × 1e-5 = [9e-6, 6e-6, 2e-6]
  
That's a tiny multiplier on the color! This seems wrong.

Unless the shader expects intensity in different units...
OR the geometry is in millimeters, not meters?
  For bulb in mm: area ~ 10-100 mm² = 0.01-0.1
  At r=1000mm: Lcol = [0.9, 0.6, 0.2] × 0.1 = [0.09, 0.06, 0.02]
  That's much larger!

So if the scene is in millimeters internally, then:
  intensity ~ 0.01-0.1 (for small emitters)
  
But Planck radiance in W/(m²·sr) is ~0.06 for 2800K visible only.
That's in the right ballpark if everything is in consistent units.
""")

# Check the actual code to see what _MAT_DB.build_radiance_tensor() returns
print("\n" + "="*70)
print("Current implementation check")
print("="*70)

from material_db import EmissionProfileDatabase, _WL_NM, _CMF_Y, _CMF_Y_NORM

ep_db = EmissionProfileDatabase.instance()
rad_tensor = ep_db.build_radiance_tensor()

print(f"Radiance tensor: {rad_tensor}")
print(f"Tungsten (profile 4): {rad_tensor[4]:.6e}")

# Check what profile_to_radiance actually does
from material_db import profile_to_radiance, tungsten_2800k_profile

# Get the tungsten profile
tungsten_profile = ep_db.get(4)
print(f"\nTungsten profile:")
print(f"  temperature_K: {tungsten_profile.temperature_K}")
print(f"  emissivity: {tungsten_profile.emissivity}")
print(f"  emission_model: {tungsten_profile.emission_model}")
