# Opt-in native scene and volume-domain contract

WITHDRAWN SCOPE: The user subsequently required copying evaluated native
responses before integration instead of exporting geometry for reconstruction.
This module and document are preserved unused pending careful cleanup of
concurrent work. They are not the response package contract. See
`STUDIO_RESPONSE_LINEAGE.md` for the minimal identity-only retention path.

`camera_designer/studio_field_domains.py` serializes the actual packed native
scene. It does not create intersections, infer geometry from PBR material IDs,
approximate a solid with a sphere, modify canonical transport, or reduce the
scene to the field recording ROI. It is CPU-only. The caller invokes it only
for an explicit `light_field_information='spectral-arrivals-v1'` request.

## Producer API

`capture_scene_contract(packed_geom, packed_shade, mat_buf, bvh_nodes, bvh_ids,
context_buf, native_pbr, sphere_data=None, region_data=None, *, bounds,
surface_transport=None, emission_profiles=None, transport_source=None)`

Pass the buffers already assembled by `_gpu_ray_field_prebuilt`, including
the actual `_ep_tensor` emission profile upload. Pass the complete transport
bounds, never the bounded recording ROI. Capture again when native geometry
or material buffers change. `transport_source`, when supplied, is the actual
assembled program string, rather than the current contents of its source file.

The result detaches the inputs into byte strings. `domain_sections(contract,
offset_bytes, payload_limit=256*1024*1024)` returns bytes and the descriptor to
append as the field artifact's `scene_contract`. Offsets are absolute from
the beginning of that same artifact. Allocation limits reject the whole
publication, not a tail of geometry, material rows, or regions.

The scene section itself is capped at 256 MiB. `payload_limit` separately
checks the end offset against the enclosing package's admitted total; a
512 MiB package must pass that total explicitly. Chunking does not relax
either admission check or omit any native records.

## Exact buffers

Every section contains `offset_bytes`, `byte_length`, `count`, `stride_bytes`,
`shape`, `format`, `binding`, `layout`, and a SHA-256 of its bytes.

| Name | Native layout | Bytes per row |
| --- | --- | --- |
| geometry | TriGeom: v0, e1, e2, normal; normal.w is uint bits | 64 |
| shade | TriShade: mat_in, mat_out, albedo, emissive; emissive.z is uint material bits | 64 |
| bvh_nodes | Node: lo_left, hi_right, start_count | 48 |
| bvh_ids | Existing BVH permutation of packed triangle IDs | 4 |
| spectral_materials | MatBuf: 32 spectral records per native material | 48 |
| scale_contexts | Actual ScaleContext center/radius and dt/n/k/type | 32 |
| native_pbr | Actual native PBR16 at the same native material ID | 64 |
| spheres | Native exact sphere center/radius and owner triangle | 32 |
| regions | Native modes/materials, plus every axial bundle affine/divider row | 16 |
| emission_profiles | Complete flat native profile header/noodle buffer | 4 |
| triangle_domains | Eight signed integer identity/ownership fields | 32 |
| medium_materials | Four signed integer identity fields | 16 |

Float sections are little-endian float32. Packed flag and material-ID bits
are preserved unchanged, even when that float32 bit pattern is nonfinite.
Integer sections are little-endian int32. Empty sections remain explicit.
Geometry and shading use original packed scene IDs, not BVH permutation order.

`triangle_domains` fields are triangle ID, native material ID, ownership kind,
positive medium, negative medium, region ID, region mode, sphere index.
Ownership kinds are 0 surface-only, 1 canonical explicit pair, 2 dynamic region,
3 unresolved transparency. Medium -1 is air; -2 is invalid/dynamic/unknown.
Dynamic sides are deliberately not fabricated from an assumed nesting stack.
Absent region/mode/sphere indexes are -1. A standalone scene region retains
both its exact explicit medium pair and native mode-4 region ID.

`medium_materials` fields are encoded medium, signed native medium, spectral
material ID, native PBR ID. Encoded medium equals signed medium + 2. Encoded 0
is invalid; 1 is air. For every nonnegative medium, its MatBuf and native PBR
row IDs are exactly that medium ID. This is not an authored-to-native guess.

## Domain identity and additive dye

The canonical identity is the signed medium plus the complete active region
bitset, not the medium alone. `canonical_domain_key` encodes an int32 medium
followed by `ceil(region_count/32)` uint32 words. It accepts the native full
32-word array only when all bits beyond the region count are zero. It rejects
missing words and references to nonexistent regions. It never substitutes a
hash for the actual key. The arrival recorder may retain all 32 words; its
encoded medium must be converted by subtracting 2 to use this helper.

The existing `studio_region_gpu.REGION_WRAPPERS` remain authoritative:

- Host, glass rod, injected dye, air bubble and standalone region modes are
  0, 1, 2, 3 and 4.
- Active dyes add their actual MatBuf k to the base medium only while the
  host and dye regions are active. Host-only and host-plus-dye arrivals can
  have identical medium IDs but different extinction domains.
- Air, invalid overlap, multiple rods and axial bundle partitions retain
  native resolver behavior. Full bundle rows and owner triangle IDs survive.
- Exact Fresnel, Snell and TIR remain in `studio_boundary_media`; exact sphere
  and axial divider adapters remain the native geometry authority.

Different keys require separate arrival accumulators. An unsplit mixed bin
is unresolved, never a known zero. No count, positive energy, occupied-cell
percentage, or complete scene buffer is a convergence certificate.

## Optical coefficients and authority

The MatBuf record fields are center frequency, bandwidth, reflectance,
transmittance, diffuse fraction, emission, reemission, real IOR, imaginary
IOR, reactive shift, GGX control, metallic control. The native medium lookup
chooses the positive-frequency row nearest in absolute log-frequency ratio.
Its real IOR and k clamps remain those of `studio_medium_optics`. The export
copies coefficients; it does not regenerate a spectrum from PBR color.

The descriptor carries a hash of every native source module that defines
these layouts/laws and a separate optional actual assembled transport-program
hash. Source-file hashes do not prove which already-imported program was
compiled. `scene_buffers_sha256` identifies all buffers plus transport bounds
and the selected surface transport; the enclosing producer must also bind
the source-population and current scene revision. Camera interval products
must bind the camera/dispatch/seed revision separately.

## Supported and unresolved representations

The export preserves canonical explicit triangle medium pairs, tagged
host/rod/dye/bubble regions, standalone scene volumes, exact native spheres,
and axial bundle metadata. It preserves ordinary opaque triangles outside
the subject ROI as well, since those still participate in transport.

The audit names every untagged transparent triangle, unsupported parametric
descriptor-only triangle, referenced material without any positive-frequency
spectral row, and missing/out-of-range profile reference. These cannot become
an authoritative solid merely by exporting its PBR16 or surface class mask.
ExtinctionGrid combined with interior arrivals is an explicitly unsupported
joint attenuation integration in the existing producer.

`volume_domains_complete` and `optical_buffers_complete` are narrow buffer
audits, not consumer implementation, topological/watertightness proofs, ray
coverage, scattering laws, or permission to render. The source currently has
no new proof of arbitrary overlapping geometry or float32 grazing cases.
The screen consumer must separately reject unsupported/missing contracts and
fall back to its primary shading for unknown samples.

## CPU verification

`python -m pytest tests/test_studio_field_domains.py tests/test_studio_sphere_regions.py tests/test_studio_region_gpu.py tests/test_studio_boundary_media.py -k 'not actual_gpu_interfaces' -q`

The focused run passed 36 tests, with the actual GPU interface test explicitly
deselected. Cases cover byte-for-byte round trips,
whole-scene remote geometry retention, exact remapped material/region IDs,
same-medium distinct dye identities, native exact sphere and bundle records,
capture detachment, section relocation, unsupported transparency/parametrics,
missing optical data, invalid IDs/BVH coverage, bit-packed flags, the actual
canonical scene packer's n/k/material IDs, and strict resource overflow.
No GL context, render, exposure, screenshot or deployment
was performed.
