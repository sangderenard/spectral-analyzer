# Studio directional forward bundle v3

This is an opt-in track-length recording estimator over existing forward transport.
It adds no transport rays and does not change the primary transport/scalar field.
`receive_light_field=true` (fixed:3 only) enables recording; all other callers
retain their existing shader and v1 export. Do not use this as calibrated
irradiance, a complete angular radiance distribution, or a mirror image solver.

## One immutable binary

Little-endian; all sections are contiguous with no padding. SHA256 covers every
byte. Top-level metadata retains the v1 dimensions, bounds, profile, sequence,
revision, sweep and normalization. `byte_length` describes the entire artifact.

1. Offset zero: existing scalar 96 cubed RGB float32, z/y/x/channel, unchanged.
   Other producer dimensions remain supported. Its length is X*Y*Z*12.
2. `directional`: 32 cubed, configurable 6 or 12 bins, six float32 per bin,
   z/y/x/bin/channel. Configuration `light_field_stencil` accepts integer6 or12
   and defaults to12. Atlas bytes:4,718,592 for6;9,437,184 for12.
   `stencil`, `bin_count`, and `bin_directions` explicitly identify the basis.
   Nearest dot product assigns the actual ray to one bin; lowest index wins ties. Channels are linear
   sRGB throughput-density R,G,B then track-length-weighted mean world propagation x,y,z. Values retain
   signed out-of-gamut RGB. The browser may deinterleave into two RGB textures
   with explicitly packed bin slabs. Twelve simple z slabs exceed WebGL2
   minimum depth guarantees; consumers must choose a supported x/z packing.
   Incident direction for shading is the opposite propagation.
3. `triangles`: uint32 rows [original_triangle_id, native_material_id,
   native_flags, class_mask], 16 bytes per row. IDs refer to the **assembled server
   scene before BVH reordering**, not browser mesh indices. Class mask bits are
   volume1 (all), reflectance2 (nonabsorber), glass4 (transmissive flag, opacity,
   or native PBR transmission). These are conservative eligibility masks, not
   evidence that a triangle is visible or was sampled. Visibility must be ANDed
   with the browser's actual currently visible geometry.
4. `sources`: actual float32 Nx12 baked source records, 48 bytes per row:
   position_xyz_m, amplitude_weight, direction_xyz, kind, frequency_hz,
   phase_radians, energy, packet_aux. These are sampled rays, not emitter IDs.
5. `materials`: native float32 PBR16 rows (64 bytes), indexed by native material
   ID: albedo_rgb, roughness, metallic, transmission, IOR, opacity,
   emission_rgb, pad, reserved4. Do not equate compact browser IDs with these IDs.
   Match the explicit authored lineage below, not derived PBR when it differs.
   Require verified identity/mapping; matching ambiguous records cannot establish
   triangle lineage.

6. `source_pbr16`: original submitted authored float32 Nx16 rows (64 bytes),
   with metadata `material_ids` linking each row to the resulting native ID.
   Recorded at the existing append/registration seam, before engine-derived
   material changes; degenerate omitted triangles do not create lineage. Multiple
   authored rows may map to one native ID. Byte-identical pairs are deduplicated.
   This is explicit material lineage, not a claim that browser and server triangle
   indices match. Camera-created materials have no fabricated authored row.

Each named top-level section has offset_bytes/byte_length and format/layout or
stride/count. `geometry_sha256` is the input subject identity supplied in config;
`scene_sha256` hashes actual packed geometry, shade, native PBR, baked sources,
spectral material buffer, contexts, bounds, spheres and optional region records
with float32 array shapes. It is not a browser mesh hash.

## What is actually recorded

The directional section requires `estimator: track-length-throughput-v1`. Previous
v2 volume-scatter-deposit bundles are not interchangeable with this estimator.
Consumers must reject unsupported/missing estimator identities rather than
interpreting legacy fog deposits as incident throughput.

At each actual forward/reactive **exterior** traced segment (ro to hp), use its
existing entry packet throughput and world propagation direction. Clip that
segment to the field bounds. Traverse at most 99 cells with a 32-cubed voxel DDA
and deposit packet.z times the exact segment length inside each crossed voxel.
The traversal does not repeat endpoint samples or depend on the legacy volume
step size. A half-open box excludes parallel segments on its maximum faces;
minimum faces are included. Zero-length and missed segments deposit nothing.

GPU fixed-point line counts use 65535 counts per transport-energy*meter, rounded
to nearest integer once per segment/voxel. Direction moments use the same counts.
CPU conversion applies the existing fixed3 linear-sRGB matrix and divides by the
world voxel volume. The result is a cumulative track-length throughput density,
with units transport-energy/m^2. It is not divided by ray jobs or solid angle,
and is not calibrated irradiance, a complete angular radiance distribution, or
a reconstructed scene image. Mean direction uses pre-RGB nonnegative line weights.
Direction rounding is at most 0.5 count per component per segment/voxel;
unit-ball clamping corrects rounding only, never substitutes a bin axis.

No legacy fog extinction/scattering factor or phase-coherence display weighting
is applied to this record. In particular, the optical-frequency exp(-4*d) kernel
of the **unchanged** scalar fog field must not erase actual travelling throughput
in this separate record. This is entry throughput before legacy wave-context
application, not a reconstructed spatial wave field. Material interiors remain
excluded; emission splats lacking a traced outgoing direction remain excluded.
There are no new ray directions, surface emission samples, interior energy, or
atmospheric beams. Atmospheric scattering/particles are a separate future model.

Recording allocates atlas bytes plus4 for its overflow flag (6bins:4,718,596;
12bins:9,437,188), uses SSBO binding
19, and requires 20 binding slots. Shader injection verifies exactly one forward
and one reactive entry; changed/missing entries fail explicitly. Signed atomic
or individual deposition overflow makes directional publication unavailable until
an ordinary field clear. The publisher continues exact scalar v1 exports with
`directional_unavailable` metadata, never clipped/wrapped moments, an invented
direction, a hidden clear, or an aborted primary exposure.

Recording clears with the field, rebinds on every forward pump and is destroyed
with its owning field. Bounds changes refresh its voxel volume. Export retains
the existing one-second rate limit and four-artifact retention. Readback does not
reset transport. No source ancestry, full angular distribution, or triangle-specific
transport is retained in the atlas.

## Verification

`python -m pytest tests/test_studio_directional_field.py tests/test_studio_forward_field.py`

With no active exposure, run `python tests/studio_directional_field_gpu_check.py`.
This isolated 128-ray fixture checks both shader variants compile, scalar
noninterference, actual oblique propagation in the correct signed bin, exact
binary layout, immutable readback and reset. Optional destination argument keeps
the artifact and metadata for API/receiver integration checks.


The world propagation convention, signed bin order, scene bounds and authored
metadata are explicit future alignment points for geodesic texture-pack baking.
No geodesic texture-pack implementation is included in this change.

The GPU fixture also forces actual signed-atomic overflow and verifies scalar
fallback plus recovery on the next ordinary clear. A second argument after the
output directory sets the bounded ray count (128 through 131072); this is a
same-trajectory stress case, not a representative full-scene performance claim.


`python tests/studio_directional_tracklength_gpu_check.py` independently compares
the production GLSL DDA against plane-intersection integration for axis signs,
oblique rays, bounds clipping, parallel boundary ownership, zero/missed segments,
and maximum-crossing termination. It also verifies density scales with the
actual voxel volume (doubling all lengths gives one-quarter density).


## Angular stencil and version compatibility

New exports use `studio.forward-field.v3`, even when6 bins are selected. Previous
`studio.forward-field.v2` stays six-bin-only in compatible consumers. The binary
per-bin layout is unchanged, but consumers must not infer bin count from a v3
schema or reinterpret an unsupported basis. The track-length estimator identity
remains `track-length-throughput-v1`.

`axes6-v1` keeps the old +X,-X,+Y,-Y,+Z,-Z order. `icosahedron12-v1` uses labels
I0 through I11 and the following normalized vectors, in exact order (phi is the
golden ratio): (0,1,phi), (0,-1,phi), (0,1,-phi), (0,-1,-phi), (1,phi,0),
(-1,phi,0), (1,-phi,0), (-1,-phi,0), (phi,0,1), (phi,0,-1), (-phi,0,1),
(-phi,0,-1). Nonzero unit components are approximately0.5257311121191336 and
0.85065080835204. Metadata publishes the full unit vectors. They classify actual
traced directions; output direction moments remain measured, not bin centers.

`weight_from_linear_rgb` is derived from the exact inverse of the existing fixed3
RGB conversion matrix: [1.4400948563066378,0.7787163904320042,0.781188753261357].
Dotting the **signed, unclipped** RGB density with this vector recovers the sum of
the nonnegative spectral throughput densities, up to float32 export roundoff.
For spatial interpolation, form direction moments as mean_direction*weight,
interpolate those moments and the weight, then divide by positive weight. Never
interpolate normalized directions with equal weight across empty and lit cells.
This reconstruction introduces no rays, energy or brightness calibration.

Changing angular bins does not increase spatial resolution or remove nearest-cell
blockiness by itself. Spatial interpolation is a separate consumer operation.
GPU validation must not compete with an active exposure; obtain its explicit
maintenance window first. GPU fixtures accept an explicit stencil argument;
run both6 and12 in that window.


## Explicit v4 root-sweep estimator and denser grid

Existing v3 behavior is unchanged. Only an explicit
`light_field_estimator: track-length-root-sweep-v1` selects
`studio.forward-field.v4`. For that estimator `light_field_grid_size` accepts32
or64 and defaults64. Scalar texture dimensions/bytes remain unchanged. The
12-bin directional64-cubed atlas contains75,497,472 bytes; its SSBO adds4 bytes
for the error flag. DDA termination is bounded by3*grid_size+3 (195 for64).

Source beta already includes native emitter area/count/wavelength compensation.
For an actual completed pump of M jobs and N source roots, root r is launched
c_r=floor(M/N)+(r<M%N) times. The auxiliary recorder multiplies every contribution
from that root by1/c_r, without changing the source ray, packet, or primary
scalar estimator. It then averages completed full-coverage pump estimates and
divides by voxel volume. One such corrected pump is one equivalent root sweep,
regardless of its within-pump replay count. There is no division by total ray
jobs or extra source-area factor.

Pumps with M<N are wholly excluded from the auxiliary record and counted
explicitly, including an insufficient128-ray warmup. They are not treated as
zero-valued contributions from unvisited roots. Until at least one eligible pump
completes, directional data is explicitly unavailable and canonical scalar v1
may still be published. A failed/uncertain/incomplete dispatch invalidates the
auxiliary record until an ordinary clear; partially deposited data is never
claimed to be a complete estimate. Counts commit only after every planned
chunk is submitted. Geometry updates and clear reset auxiliary data and counts.
No extra primary rays are launched to complete coverage.

Required `directional.root_sampling` metadata contains:
- version: per-complete-pump-inverse-replay-v1
- source_root_count: N, equal to sources.count
- completed_root_sweeps: S>0
- recorded_root_jobs: J>=N*S, eligible pumps only
- normalization_divisor: exactly S
- excluded_partial_root_jobs: nonnegative count of ineligible pumps' jobs
- incomplete_dispatch: false for every publishable v4 atlas

Inverse replay correction precedes fixed-point quantization. Different replay
multiplicities agree within the stated rounding bound, not necessarily bit for
bit; tiny corrected contributions can round to zero. The estimator is still a
finite-sample throughput-density proxy, not calibrated irradiance. Supplying an
extinction grid continues to integrate that supplied transmittance over each
actual segment; no atmosphere or scattering is manufactured.

A64 grid halves cell widths but increases density along an isolated track by
about4 and reduces occupancy. It does not supply missing wavelength samples.
Native roots retain their actual wavelength/direction combinations; repeated
roots cannot fix direct-path chromatic sampling noise. No spectral smoothing,
RGB matrix alteration, or arbitrary gain correction is applied.

CPU gate: tests/test_studio_root_sweep_field.py. Bounded GPU verification:
`python tests/studio_root_sweep_field_gpu_check.py <optional-output-directory>`.
The proof checks32/64 grids, scalar equality, repeated root sweeps, uneven root
replays, excluded partial coverage, geometry reset, and supplied extinction.


## V5 bounded subject recording region

Explicit `light_field_region="subject"` retains the full transport scene and
scalar bounds, but clips only the auxiliary recorded segments to independent
`uStudioRecordMin/Max` bounds. It retains v4 root-sweep normalization. The CPU
`subject_field_region(vertices, config)` receipt is computed before GPU allocation.
World subject vertices use the canonical float32 scale/translation; declared
analytic subject sphere extents are included. It never substitutes fixture bounds.

`light_field_cell_size_m` accepts 0.00005–0.01 metres (requested 0.0005).
`light_field_margin_m` accepts 0–0.032 metres, default 0.001. Effective physical
padding is max(requested margin, one cell), so the outer subject surface is
strictly inside the half-open recording domain. Padded bounds snap outward to
integer cell coordinates, then receive symmetric integer extra padding to the
smallest supported32/64 cube. A request exceeding64 cells on any axis is rejected;
no coarsening or sparse approximation occurs.

V5 directional metadata adds independent bounds, cell_size_m, margin_m and region
(kind subject, source geometry SHA256, transformed world geometry SHA256, actual
bounds and effective_margin_m). The scalar prefix retains global bounds. Empty
cells indicate no recorded nonzero quantized throughput, not proof of no light.
Fixed source paths may entirely miss a small subject ROI; no added rays or gain
are used to conceal that limitation. Live geometry changes rederive the ROI and
clear accumulated moments; a changed grid size requires a new candidate.


## V6 progressive native angular populations

`emitter_sampling` exposes the existing native generator only: version1,
ray_budget128..65536, seed0..2147483647, refresh_each_sweep boolean. The actual
rounded spectral-cycle count must fit the live ray batch. Refresh requires live
fixed3 fixture transport with a root-sweep field. It does not enable the native
interaction-target option: that proposal currently lacks matching PDF weights.

At a complete live sweep boundary, the worker generates seed(base+index) modulo
2^31 through native_emissive_sources, installs actual SourceRec12 rows into the
original SSBO5, refreshes the original camera area-NEE view, and traces the full
scene. Seed changes angular disk samples, not emitter position strata or cyclic
wavelength selection. These are refreshed angular populations, not IID sampling
of every path dimension. No overlay is required to receive the field; the
existing sensor loop still runs.

The scalar is cleared each sweep. Auxiliary raw six-channel moments are merged
only after a complete correctly weighted population, then exported as their
mean. Float64 aggregate storage costs150,994,944 bytes for64^3×12×6, separately
from the75,497,476-byte GPU atlas and export buffers. Integer-exact float64 sum
capacity is checked before merging. A scene/camera history reset discards this
aggregate; ordinary sweep clear preserves it. Root count changes require a new
candidate. Dynamic subject updates with refresh are currently unsupported.

V6 directional fixedpoint divisor is2^24, version fixed24-track-length-v1, with
matched decoding; scalar and old v3-v5 recording remain65535. Each deposit has
at most0.5/divisor energy-length rounding error. This is precision, not gain;
very short or weak tracks can still round to zero. Overflow invalidates the
population. Actual .5mm twice-replayed axial deposits from the captured scene's
predicted32768-budget roots produce roughly80–158 counts rather than0–1.

`source_populations` records the generator, requested budget, base seed, index
range, count, latest seed/source hash and rolling hash. Each chain entry hashes
previous32-byte chain + uint64LE population index +32-byte source-buffer hash;
the initial chain is32 zero bytes. The source section contains only the latest
complete population. Population provenance is reproducible, not individual
path ancestry. Coverage reports actual nonzero quantized cells; unvisited cells
are never filled. Unchanged completed populations reuse their existing export.


### Opt-in canonical internal-medium observer

`light_field_internal_media=true` requires refreshed v6 populations. Defaultfalse
and all older jobs retain exterior-only recording. The existing tested helper
moves only observer calls before canonical Beer endpoint attenuation, integrating
the installed material/region/dye attenuation over each recorded cell from the
actual entry packet. Primary/scalar transport and its air guards are unchanged.
Invalid medium ownership is excluded. `directional.internal_media` explicitly
identifies canonical-beer-segment-integral-v1. This records only actual interior
ray segments, not a complete reconstructed interior or invented glass fog.
Combining it with supplied external ExtinctionGrid is rejected before allocation
until joint integration is implemented. False still permits the existing grid.

## Explicit opaque surface transport

`surface_transport: studio-pbr-mixture-v1` selects the shared opaque dielectric
Lambert/GGX mixture in the existing Studio forward, reactive and camera paths.
It requires fixed:3 transport and fixture geometry. Authored albedo, roughness,
IOR and compatibility diffusion determine the mixture; glass and conductor
optical laws remain unchanged. Both sampling and evaluation use the full
mixture PDF. Opaque surface residuals are not interpreted as attenuation over
the preceding air segment. Positive finite forward packets survive until the
existing finite bounce limit rather than an absolute packet-energy cutoff.
The corrected camera path uses the same BSDF for direct and indirect lighting,
with a disjoint direct-emitter estimator partition, and does not add the old
uncalibrated scalar-volume surface term. Signed Gaussian spectral arguments
are squared by multiplication, avoiding undefined GLSL pow on negative inputs.

The top-level field receipt repeats the exact surface_transport string. Absent
configuration retains the previous shader. This explicit correction changes
transport results; scalar bit parity is not claimed for the corrected mode.
Neither finer cells nor this correction guarantees that a finite source
population visits the subject ROI. Coverage reports only actual recorded data.

## Validity and darkness are not established by occupancy

Every directional export carries validity version partial-track-estimate-v1.
This metadata describes data quality and does not disable the field path. A nonzero quantized value
is a recorded estimate, not a completeness certificate; a zero is unknown.
Neither completed root sweeps nor occupied-cell percentages establish spatial,
angular, spectral or material completeness. Finite Monte Carlo samples cannot
prove a region has exactly zero transport.

Current six-float bins discard pre-quantization visit counts, per-spectral-line
direction moments, angular second moments, per-cell medium/bounce ancestry,
and per-cell exclusion/rounded-to-zero counts. They carry no known-occlusion
or error/convergence certificate. Scene material tables identify geometry but
do not identify which medium contributed each cell. Internal Beer observer
metadata describes the integrated law, not per-cell interior coverage. The
three fixed spectral lines are not a continuous spectrum. Mean directions can
cancel and cannot represent every incident direction within a bin. Signed
RGB values and exact transform metadata must not be confused with negative
physical line energy.

A future measured sampling contract needs bounded per-cell/per-band arrival
and exclusion counters, spectral moments, medium attribution or an explicit
mixed/unknown state, complete reset/population accounting, counter-overflow
handling and producer-to-screen validity propagation. Even those observations
must distinguish measured visits from certified visibility and must not turn
sample thresholds into proof of darkness. Sparse field availability must alpha-blend with Phong for the unknown portion;
energy zero must not be used as availability.
