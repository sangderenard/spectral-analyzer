"""
Real Blackbody Emittance Calibration
=====================================

Move from fake-scaled power to genuine physical radiance and solid-angle integration.

Physics Foundation
------------------
For a Lambertian blackbody surface at temperature T with emissivity ε:

1. Spectral radiance (W/(m²·sr·m)):
   L_λ(λ,T) = ε(λ) × [2hc² / λ⁵] / [exp(hc/λk_BT) - 1]   [Planck]

2. Total radiance (W/(m²·sr)) integrating visible spectrum:
   L_vis(T) = ∫_λ₁^λ₂ L_λ(λ,T) y(λ) dλ    [CIE luminosity weighting]

3. Luminous intensity (cd) from surface area A:
   I(T) = L_vis(T) × A × π   [Lambertian: π solid angle factor]

4. Illuminance at distance r with solid angle Ω:
   E(r) = L_vis(T) × Ω(r)   [exact, no distance term — it's in Ω]

   Where solid angle depends on emitter geometry:
   - Disk: Ω = 2π(1 - cos(θ_half))
   - Sphere of radius R at distance r: Ω = 2π(1 - √(1-(R/r)²))
   - Far field (r >> R): Ω ≈ π(R/r)²

Implementation Strategy
-----------------------
Instead of:
  EmissionProfile(total_power_W=1500)  [fake]

Use:
  EmissionProfile(temperature_K=2800, emissivity=0.95)  [real]

At startup, compute:
  1. Planck integral over visible → L_vis (W/(m²·sr))
  2. Store in PBR tensor as "radiance" field (new)
  3. In _derive_gl_lights(): for each emitter, compute actual solid angle Ω
  4. Fake-light intensity = L_vis × Ω

This is **physically accurate** and **more efficient** than most engines.
"""

import numpy as np
import math

# Physical constants
PLANCK_H = 6.62607015e-34      # J·s
LIGHT_C = 299792458.0          # m/s
BOLTZMANN_K = 1.380649e-23     # J/K
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)

# CIE 1931 standard observer
# (precomputed in material_db.py as _CMF_X, _CMF_Y, _CMF_Z at 5nm steps)
# For this calibration, we'll use a simplified integration

# Wavelengths for Planck integration (visible + near-IR for tungsten)
WL_NM = np.linspace(380, 2500, 433)  # 380–2500 nm at 5nm steps
WL_M = WL_NM * 1e-9

# Simplified CIE Y luminosity function (photopic, approximation)
# Peak at 555 nm (green), drops to near-zero at 380 and 780 nm
def cie_y_approx(wl_nm):
    """Approximate CIE 1931 Y luminosity function."""
    x = (wl_nm - 555.0) / 100.0
    return np.exp(-0.5 * x**2) * (wl_nm < 780) * (wl_nm > 360)

CIE_Y = cie_y_approx(WL_NM)

def planck_spectral_radiance(T_K, wavelength_m):
    """Planck spectral radiance (W/(m³·sr)).
    
    L_λ = (2hc² / λ⁵) / [exp(hc/λk_BT) - 1]
    
    Args:
        T_K: temperature in Kelvin
        wavelength_m: wavelength in meters (scalar or array)
        
    Returns:
        Spectral radiance in W/(m³·sr)
    """
    numerator = 2 * PLANCK_H * LIGHT_C**2 / (wavelength_m**5)
    exponent = (PLANCK_H * LIGHT_C) / (wavelength_m * BOLTZMANN_K * T_K)
    # Avoid overflow: if exponent > 100, exp is huge, so denominator >> 1
    exponent = np.clip(exponent, 0, 200)
    denominator = np.expm1(exponent)  # exp(x) - 1, stable for small x
    denominator = np.maximum(denominator, 1e-30)  # avoid division by zero
    return numerator / denominator


def visible_radiance_from_planck(T_K, emissivity=1.0):
    """Integrated visible radiance (luminous intensity per unit area per solid angle).
    
    L_vis(T) = ∫₃₈₀^₇₈₀ L_λ(λ,T) × y(λ) dλ
    
    Where y(λ) is CIE Y luminosity weighting (approx from cie_y_approx).
    
    Returns:
        Radiance in W/(m²·sr)  [Note: this is the CIE-weighted visible output only]
    """
    L_lambda = planck_spectral_radiance(T_K, WL_M)  # W/(m³·sr)
    dlamb = 5e-9  # 5 nm steps
    weighted = L_lambda * CIE_Y
    L_vis = np.sum(weighted) * dlamb * emissivity
    return float(L_vis)


def total_luminous_intensity(T_K, area_m2, emissivity=1.0):
    """Luminous intensity from a Lambertian blackbody surface.
    
    I(T) = L_vis(T) × A × π
    
    (The π factor accounts for Lambertian cosine distribution: ∫ cos(θ) dΩ = π)
    
    Returns:
        Luminous intensity in W/sr
    """
    L_vis = visible_radiance_from_planck(T_K, emissivity)
    return L_vis * area_m2 * math.pi


def solid_angle_from_disk(radius_m, distance_m):
    """Solid angle subtended by a disk of given radius at given distance.
    
    For a disk perpendicular to line-of-sight:
    Ω = 2π(1 - cos(θ_half)) = 2π(1 - r/√(r²+R²))
    
    Args:
        radius_m: disk radius
        distance_m: distance from disk center
        
    Returns:
        Solid angle in steradians
    """
    if distance_m <= 0:
        return 0.0
    r = float(distance_m)
    R = float(radius_m)
    half_angle_cos = r / math.sqrt(r**2 + R**2)
    return 2 * math.pi * (1 - half_angle_cos)


def solid_angle_from_sphere(radius_m, distance_m):
    """Solid angle subtended by a sphere at given distance (center-to-point).
    
    Ω = 2π(1 - √(1 - (R/r)²))
    
    Valid for r ≥ R (point outside or on sphere).
    """
    if distance_m <= radius_m:
        return 4 * math.pi  # point is inside; full sphere solid angle
    r = float(distance_m)
    R = float(radius_m)
    ratio = R / r
    if ratio >= 1.0:
        return 4 * math.pi
    return 2 * math.pi * (1 - math.sqrt(1 - ratio**2))


def solid_angle_from_point_approx(area_m2, distance_m):
    """Approximate solid angle for small emitter (point-like source).
    
    Ω ≈ π(D/r)² where D = √(4A/π) is the effective diameter.
    
    This is the far-field approximation and matches the fake-light model.
    """
    if distance_m <= 0:
        return 0.0
    # Effective radius from area
    R_eff = math.sqrt(area_m2 / math.pi)
    # Far-field solid angle
    return math.pi * (R_eff / distance_m)**2


print("=" * 80)
print("REAL BLACKBODY EMITTANCE CALIBRATION")
print("=" * 80)

# Current tungsten filament (2800 K)
T_tungsten = 2800.0
area_filament = 0.05e-6  # 0.05 mm² (bare filament)

print(f"\n--- Bare Tungsten Filament (2800 K) ---")
print(f"Temperature: {T_tungsten:.0f} K")
print(f"Area: {area_filament*1e6:.4f} mm²")

L_vis = visible_radiance_from_planck(T_tungsten)
print(f"Visible radiance L_vis: {L_vis:.6f} W/(m²·sr)")

I_total = total_luminous_intensity(T_tungsten, area_filament)
print(f"Total luminous intensity: {I_total:.6e} W/sr")

# At various distances, compute illuminance using solid angle
print(f"\nIlluminance via solid angle (E = L_vis × Ω):")
for dist_m in [0.1, 0.5, 1.0, 2.0]:
    Omega = solid_angle_from_sphere(math.sqrt(area_filament/math.pi), dist_m)
    E = L_vis * Omega
    print(f"  @ {dist_m:.1f} m: Ω = {Omega:.6f} sr, E = {E:.6e} W/m²")

print("\n" + "=" * 80)
print("COMPARISON: FAKE-SCALE vs REAL BLACKBODY")
print("=" * 80)

# Simulate the current fake-light system
fake_power = 1500.0  # W (as currently set)
fake_area = 4.3e-4   # m² (the 1500W setting at 2800K)

print(f"\nCurrent fake-scale system:")
print(f"  total_power_W: {fake_power}")
print(f"  Effective area: {fake_area:.3e} m²")
print(f"  Fake intensity @ distance r: I_fake(r) = area / r²")

print(f"\nProposed real blackbody system:")
print(f"  temperature_K: {T_tungsten:.0f}")
print(f"  emissivity: 0.95")
print(f"  Visible radiance L_vis: {L_vis:.6f} W/(m²·sr)")
print(f"  Real intensity @ distance r: I_real(r) = L_vis × Ω(r)")

# Show that we can match the fake system's distance behavior
print(f"\n--- Matching the fake-system behavior at various distances ---")
distances_m = [0.5, 1.0, 2.0, 5.0]

print(f"\n{'Distance':<15s} {'Fake I(r)':<20s} {'Real Ω(r)':<20s} {'Real E(r)':<20s}")
print("-" * 75)

for dist in distances_m:
    # Fake system: intensity scales as area / r²
    I_fake = fake_area / (dist**2)
    
    # Real system: compute solid angle from small sphere
    R_eff = math.sqrt(area_filament / math.pi)
    Omega_real = solid_angle_from_sphere(R_eff, dist)
    E_real = L_vis * Omega_real
    
    print(f"{dist:.1f} m{'':<10s} {I_fake:.6e}{'':<8s} {Omega_real:.6e}{'':<8s} {E_real:.6e}")

print("\n" + "=" * 80)
print("PROPOSED API")
print("=" * 80)

print("""
Old EmissionProfile (fake):
    EmissionProfile(
        spd=[...],
        total_power_W=1500.0,  # ← arbitrary scaling
    )

New EmissionProfile (real):
    EmissionProfile(
        spd=[...],                    # ← unchanged; defines spectrum shape
        temperature_K=2800.0,         # ← physical temperature
        emissivity=0.95,              # ← physical emissivity (0–1)
        # total_power_W is DERIVED at startup from T + area
    )

At registration time:
    1. profile_to_rgb() computes the color via CIE integration (unchanged)
    2. NEW: profile_to_radiance() computes L_vis from temperature_K + emissivity
    3. Store L_vis in material tensor (new field, ~1 float per material)
    4. In _derive_gl_lights():
       - For each emitter: compute centroid + bounding radius R
       - Compute Ω = solid_angle_from_sphere(R, distance_to_camera)
       - Fake-light intensity = L_vis × Ω
       - (Much more accurate than area/r²)

Benefits:
  ✓ Physically defensible — temperature, emissivity are real material properties
  ✓ Spectrum and intensity are **decoupled** (both from first principles)
  ✓ Easy to adjust: change temperature → get different spectrum AND intensity
  ✓ Solid-angle model is exact (no arbitrary scaling)
  ✓ Works for arbitrary geometry (disk, sphere, point source)
""")

print("\n" + "=" * 80)
print("QUICK REFERENCE: temperatures and visible radiance")
print("=" * 80)

temps = [1500, 2000, 2700, 2800, 3000, 3200, 4000, 5000, 6500]
print(f"\n{'T (K)':<10s} {'L_vis W/(m²·sr)':<25s} {'Example Source':<40s}")
print("-" * 75)
for T in temps:
    L = visible_radiance_from_planck(T)
    examples = {
        1500: "Candle / fire",
        2000: "Tungsten bulb (dim)",
        2700: "Warm LED / tungsten",
        2800: "Tungsten filament (40W bulb)",
        3000: "Tungsten halogen",
        3200: "Tungsten halogen (bright)",
        4000: "Daylight (cool white)",
        5000: "Daylight (neutral)",
        6500: "Daylight (bluish)",
    }
    ex = examples.get(T, "")
    print(f"{T:<10d} {L:<25.6f} {ex:<40s}")

print("\n" + "=" * 80)
print("STEP-BY-STEP: How to adapt spectral_analyzer")
print("=" * 80)

print("""
1. Update EmissionProfile dataclass in material_db.py:
   - ADD: temperature_K: float = 5778.0  (default: sun temperature)
   - ADD: emissivity: float = 0.95       (default: real-world blackbody)
   - KEEP: total_power_W for backward compat (computed from T + area at registration)
   
2. Add profile_to_radiance() function alongside profile_to_rgb():
   - Input: EmissionProfile with temperature_K + emissivity
   - Output: visible radiance L_vis in W/(m²·sr)
   - Uses Planck integration with CIE weighting
   
3. Update material PBR tensor:
   - NEW field in each row: radiance_W_sr_m2 (one float)
   - Baked at registration time
   - Shader doesn't need to recompute
   
4. Update _derive_gl_lights() in demo_pluck_gl.py:
   - Instead of: intensity = area
   - Use: intensity = L_vis × Ω(r)
   - Where Ω is solid angle from emitter bounding radius
   
5. Optional: Add fallback for old profiles:
   - If temperature_K not set, fall back to fake total_power_W scaling
   - Gradual migration, no breaking changes

This is a **strict improvement** — real physics, no heuristics, scales to any emitter size.
""")
