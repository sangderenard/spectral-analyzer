"""
Blackbody Calibration for Emissive Materials
==============================================

Architecture
~~~~~~~~~~~~

Materials are authored in their natural units (spectrum + power scale).
At render time, the blackbody calibration converts these to physical radiance.

Step 1: Registration Time (material_db.py)
------------------------------------------
Each EmissionProfile has:
  - spd: spectral distribution (defines color)
  - total_power_W: intensity scaling (in natural units)
  - emission_model: calibration method ("blackbody", "parametric", or default)
  - temperature_K, emissivity: physical parameters (for blackbody model)

At registration, profile_to_radiance(profile) computes visible radiance L_vis:

  Model "blackbody":
    L_vis = Planck(λ,T) integrated over visible + weighted by emissivity
    Result: W/(m²·sr) independent of total_power_W
    
  Model "parametric":
    L_vis = total_power_W × (∫ SPD(λ)·y(λ) dλ) / (∫ y(λ) dλ)
    Result: total_power_W is used as a scaling factor
    
  Default (no model):
    L_vis = total_power_W
    Result: direct mapping (legacy)

The L_vis value is baked into the radiance tensor at startup.

Step 2: Render Time (_derive_gl_lights in demo_pluck_gl.py)
-----------------------------------------------------------
For each emissive material group:
  1. Compute centroid and bounding radius R
  2. Fetch L_vis from radiance_tensor[mat_id]
  3. Compute solid angle: Ω(r) ≈ π(R/r)² where r = distance to camera
  4. Intensity = L_vis × Ω(r)

This gives physically-correct inverse-square falloff via solid angle,
replacing the old fake area/r² model.

Benefits
~~~~~~~~
✓ Materials stay in natural units (spectrum + power scale)
✓ Blackbody calibration is applied uniformly at render time
✓ Consistent inverse-square falloff for all emitters
✓ Easy to add new materials: just set emission_model + natural units
✓ Backward compatible: default model = legacy behavior

Example: Tungsten Filament
~~~~~~~~~~~~~~~~~~~~~~~~~~
Material authored as:
  EmissionProfile(
    spd=[...2800K spectrum...],
    temperature_K=2800,
    emissivity=0.95,
    emission_model="blackbody",
  )

At registration:
  L_vis = profile_to_radiance(profile)
        = Planck(2800K) × 0.95
        ≈ 0.065 W/(m²·sr)

At render time (in _derive_gl_lights):
  For emitter at distance r with radius R:
    Ω = π(R/r)²
    intensity = 0.065 × Ω
  
  Shader applies:
    light_color = rgb × (intensity / r²)
    
  Result: physically correct illumination

Migration Path
~~~~~~~~~~~~~~
Existing profiles can be migrated incrementally:

1. Tungsten_2800k: add emission_model="blackbody", temperature_K=2800
   ✓ Immediately uses physical Planck law

2. LEDs/phosphors: add emission_model="parametric"
   ✓ total_power_W now means "scale factor for SPD shape"
   ✓ Better than before: intensity now respects spectrum

3. Legacy profiles: leave as default (no emission_model)
   ✓ Still work; use total_power_W as-is
   ✓ Can be updated later

Calibration Values
~~~~~~~~~~~~~~~~~~~
Builtin profiles:

  basic_led_display (0.35 W):
    emission_model="parametric"
    L_vis = 0.35 × (SPD_shape integrated) ≈ 0.001 W/(m²·sr)
  
  missing_material_hazard (2.6 W):
    emission_model="parametric"
    L_vis ≈ 0.001 W/(m²·sr)
  
  material_database_failure (3.2 W):
    emission_model="parametric"
    L_vis ≈ 0.001 W/(m²·sr)
  
  construction_borosilicate (0.06 W):
    emission_model="parametric"
    L_vis ≈ 0.00001 W/(m²·sr)
  
  tungsten_2800k:
    emission_model="blackbody", temperature_K=2800, emissivity=0.95
    L_vis = Planck(2800K) × 0.95 ≈ 0.065 W/(m²·sr)

Adding a New Thermal Emitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example: warm LED at 3200K with 80% emissivity

  self.register("warm_led_3200k", EmissionProfile(
    spd=[...your spectrum...],
    total_power_W=1.0,
    emission_model="blackbody",
    temperature_K=3200,
    emissivity=0.80,
  ))

At registration, L_vis is computed from Planck law.
At render time, solid-angle model uses L_vis for correct falloff.
No artificial tweaking needed — pure physics.

Adding a Non-Thermal Emitter (LED/Phosphor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example: typical blue LED

  self.register("blue_led", EmissionProfile(
    spd=[...measured spectrum...],
    total_power_W=1.0,  # ← scaling factor for this SPD
    emission_model="parametric",
    # (or default for legacy behavior)
  ))

At registration, L_vis = total_power_W × (∫ SPD·y dλ).
Spectrum defines color; total_power_W defines intensity scale.

No manual radiance computation needed — the framework handles it.
"""

print(__doc__)
