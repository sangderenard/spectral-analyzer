# Physical apertures and emissive backs

This contract joins finite aperture materials and authored light-source backs
to the optical engine without creating a second tracer or widening ordinary
ray records.

## Aperture and scrim geometry

`camera_software.physical_aperture.LivePhysicalAperture` is authoritative for
both the displayed geometry and the native T4 material payload. Available
factories are:

- `iris`: finite-thickness polygonal bladed aperture;
- `circular_hole`: a singular circular bore through a finite plate;
- `circular_hole_array`: a two-axis perforated scientific scrim;
- `slot_array`: a two-axis rectangular slot screen;
- `grating`: a one-axis transmission grating with finite bars.

Repeated openings must leave a nonzero material web. The native kernel applies
the named material's complex refractive index and physical thickness; these are
not ideal binary masks.

Live inspection:

```powershell
python wave_transform_visual_demo.py --aperture-live --aperture-pattern circular
python wave_transform_visual_demo.py --aperture-live --aperture-pattern hole-array
python wave_transform_visual_demo.py --aperture-live --aperture-pattern slot-array
python wave_transform_visual_demo.py --aperture-live --aperture-pattern grating
```

Press `M` in the live view to cycle the same choices. Fixed-band and continuous
spectral cohorts remain independently selectable.

## Baked emissive backs

`camera_designer.camera_preset.ProjectorBackSpec` is also the reusable physical
emissive-back authoring object. It can reference a catalog emitter or carry an
inline `EmissiveTexture`. A home-baked complex transverse field becomes a
source as follows:

```python
back = ProjectorBackSpec.from_jones_field(
    jones_field,                         # complex (H, W, 2): Ex, Ey
    asset_key="bench.projector.prototype-01",
    base_profile_name="laser_532nm_green",
    power=2.0,
)
preset.projector_back = back
```

The texture can carry local intensity, spectral shift and width, phase,
coherence length, polarization angle, or a complete Jones vector. Source-plane
quadrature now samples these channels at each physical UV site. The launch
keeps scalar spectral phase in fixed complex lanes and attaches Jones,
transverse-basis, coherence, and operator state through stable ray tags into
one pipeline-owned contiguous source block.

`ProjectorBackSpec.source_contract()` produces the cacheable
`physical-emissive-back-v1` manifest. When projector transport is compiled,
its compact reference/shape contract is attached to the graph's physical
source-plane entry; the large inline field is not duplicated into graph state.
The station remains a client; the optical backend owns the tracer and
transport.

## Flashlights and other luminaires

`PlayerFlashlight` uses the same emissive-back contract at its real bulb
position inside the physical parabolic reflector. Its default is a catalog
2856 K tungsten source. Passing a `ProjectorBackSpec` as `emissive_back`
installs a home-baked spatial or Jones source without replacing the reflector
geometry:

```python
light = PlayerFlashlight(emissive_back=back)
light.enable()
emitter = light.optical_emitter_spec(wavelengths_um)
manifest = light.optical_source_contract(wavelengths_um)
```

The returned `EmitterSpec` is flashlight-local so the scene coordinator can
apply the same held/world transform to the source and reflector. This is the
common luminaire boundary intended for future condenser, reflector, diffuser,
and beam-shaping modules.

## Current boundary

The projector path is fully texture-aware at source lowering and enters the
existing reciprocal exact-lens/pipeline path. The flashlight now authors the
same physical source contract, but dynamic scene publication still needs a
general hot luminaire-source update hook before a held flashlight can replace
its live source buffer without rebuilding scene-source state.
