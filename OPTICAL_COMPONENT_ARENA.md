# Canonical optical components and inspection arena

Status: first complete component-to-engine vertical slice  
Schema: `optical-component-v1`

## What is now authoritative

`camera_software.optical_components` is the loadable component boundary for
benches, bell jars, editors, manifests, and tests. A component is authored
once and cold-compiles into artifacts owned by the existing optical engine:

- T1/T3 role-tagged physical geometry and canonical materials;
- T2 fused parametric schedules;
- T4 localized persistent complex-field regions;
- named complex-ray/field ports, controls, and representative geometry.

The component layer is deliberately not a solver. The OpenGL arena is also
not a solver: it presents compiled contracts and invokes the production
native transports selected by those contracts.

The initial registry contains:

| Key | Authoritative transport |
| --- | --- |
| `aperture.iris` | finite bladed material in a localized vector T4 region |
| `lens.default-camera` | the existing fused exact-conic T2 payload |
| `mirror.plane` | T1 hit plus canonical conductor interaction in T3 |
| `pentaprism.finder` | closed role-tagged T1 mesh, dielectric and conductor T3 interactions |

All four compile at exact lane widths `1, 3, 4, 8, 16, 32`. Fixed lane
variants remain fixed; the graph/component layer does not turn them into a
maximum-width dynamic allocation.

## Live inspection

List the installed components:

```powershell
python wave_transform_visual_demo.py --list-components
```

Open one without retaining frames:

```powershell
python wave_transform_visual_demo.py --component-live `
  --component aperture.iris

python wave_transform_visual_demo.py --component-live `
  --component lens.default-camera

python wave_transform_visual_demo.py --component-live `
  --component mirror.plane

python wave_transform_visual_demo.py --component-live `
  --component pentaprism.finder --component-lanes 16 `
  --component-engine ray --component-size 384
```

`--component-size`, rather than the unrelated sequence-render `--size`, controls
the arena's source texture and wave crop. Live defaults are one spectral lane,
a 128-pixel crop, and at most eight asynchronously produced texture generations
per second. Override them explicitly with `--component-lanes`,
`--component-size`, and `--component-solve-hz`.

The OpenGL event/control loop is independent of texture production. It keeps
presenting the last completed immutable generation while the solver builds the
next one, and obsolete queued control states collapse to the newest request.
Only a completed generation uploads its six panels.

This distinction matters most for `aperture.iris`: balanced open-boundary
quality pads each visible dimension by two and performs four propagation
substeps for every Jones component and spectral lane. The old implicit
512-pixel/four-lane live request therefore solved a 1024-by-1024 hidden field
on every display frame. That was not the named `bake` quality mode, but it was
an inappropriate interactive workload. High-investment lane counts and sizes
remain available explicitly, and named ultra-bakes remain the retained-output
path.

One square product owns the viewport; this is not a six-up contact sheet.
Use `1` through `6`, arrow keys, or Tab to select products. The six products
have stable meanings:

1. representative/transport geometry;
2. the authoritative transport probe;
3. the compiled execution graph;
4. typed component ports;
5. canonical material-role bindings;
6. component identity, exact lane width, and controls.

The compact type is intentionally instrumentation-scale so the data does not
consume the visualization. A pane names the provenance of what it shows:

- `NATIVE LIGHT STATE` means segments drained from T1/T3. Their hue is
  computed from the actual per-lane spectral power and their glow follows
  transported field power `|E|^2`.
- `EXACT GEOMETRIC PATH` means the authoritative parametric path is shown but
  the complex sidecar is not yet available at that observation point.
- `COMPILED TRANSPORT GRAPH`, ports, materials, and state are contracts rather
  than light measurements.
- aperture Stokes, polarization, phase, analyzer, and spectral-power panes are
  resolved directly from the production vector complex field.

This provenance labeling is a rule for future arena products: a beautiful
diagnostic must state which physical quantity supplies color, opacity, and
brightness. Decorative rays must never masquerade as measured light state.

`--component-engine` accepts `auto`, `ray`, `parametric`, `wave`, `hybrid`,
and `maxwell`. Selection is a requirement, not a preference: unsupported
component/backend pairs fail with their available engines and never substitute
a sparse ray view for requested wave transport. At this revision the physical
aperture supports wave/hybrid, the exact lens supports parametric, and the
mirror and pentaprism support ray. The rotated complex-field port chain needed
for a truthful wave pentaprism remains an explicit acceptance gate.

For the aperture, the panes are the production vector-field products. Its
animated opening is derived from the loaded `LivePhysicalAperture`, preserving
its physical blade count, named material, finite thickness, and rotation. No
ideal mask or viewer-private iris is substituted.

The default lens probe calls the same `CompoundLens.trace_detailed` exact
surface model whose byte-identical fused payload is registered for T2.
Mirrors and pentaprisms are assembled through `build_native_component_scene`
and run through the native T1/T3 tracer. Their pictures are therefore
transport observations, not illustrative ray-line guesses.

## Loading and embedding

Hosts select a registry key and lane count:

```python
registry = default_optical_component_registry()
component = registry.create("pentaprism.finder", lane_count=8)
compiled = component.compile(8)
compiled.validate()
```

For components with role-tagged T1/T3 geometry:

```python
scene = build_native_component_scene(compiled, frequencies_hz)
tracer = scene.create_tracer()
```

The lowering uses the shared material dictionary and the canonical MatBuf
adapter. Dielectric roles get explicit inside/outside media, conductor roles
use the material's complex index, and absorber/stop roles receive native
material flags. Manifests should store component keys and authored control
values, not Python class names or flattened demo meshes.

Composite optics are ordinary components whose compiled graph contains
multiple material/parametric operations and branches. `PentaprismComponent`
is the first composite. This is the intended route for pentaprisms, reflex
mirrors, beam splitters, projector backs, lens-plus-baffle groups, and later
user-authored assemblies.

## Boundaries that remain intentionally open

The component contract does not claim that downstream GPU ray stages yet
retain every complex field datum after a T4 exit. The current compact GPU
`RayIntent` has no spare full-width word, and T1 does not retain
`interaction_flags` in its surface-hit record. Encoding a state index into
that field would alias branches and lose the index at the first hit.

The correct continuation is a compiled complex-intent record variant, sharing
the common pipeline implementation while adding one explicit state handle.
That handle addresses a cold-allocated, solid contiguous complex-path state
block. T1 copies it, T2 preserves/transforms it, T3 copy-on-write clones it
when a path branches, and T4 consumes/publishes it. The ordinary compact ray
variant remains unchanged. Existing SSBO rechanneling must carry the extended
variant over the same hardware-limited bindings; it must not add an assumed
ninth channel.

That work is the next mixed ray/wave transport slice. It is separate from,
and now cleanly enabled by, the component API completed here.

## Emitter-to-sensor light tables

Emitters and sensors are first-class components, not viewer decorations.
`EmitterEndpointComponent` retains the complete `EmitterProfile` contract
(spectrum, phase, coherence, polarization, directionality, and optional
texture), while `SensorEndpointComponent` retains physical geometry, quantum
efficiency, CFA/readout, film, color-science, exposure, and output stages.
The default registry exposes these as `emitter.laser-532nm` and
`sensor.fullframe`.

`compile_optical_chain` compiles each component independently, namespaces its
nodes, preserves its native T2 payloads and T4 descriptors, and authors
checked free-space links between typed ports. Named instances make a mounted
part replaceable without rebuilding or flattening neighboring component
contracts:

```python
import numpy as np

from camera_software import (
    LivePhysicalAperture,
    compile_optical_chain,
    default_optical_component_registry,
    light_table_chain,
    mount_aperture_at_lens_stop,
    run_geometric_light_table,
)

registry = default_optical_component_registry()
lens = registry.create("lens.default-camera", 1)
iris = LivePhysicalAperture.iris(
    "demo.iris",
    blade_count=7,
    opening_radius_m=30e-6,
    assembly_radius_m=52e-6,
    thickness_m=3e-6,
)
mounted = mount_aperture_at_lens_stop(iris, lens.lens)
spec = light_table_chain(lens, mounted, lane_count=1)
chain = compile_optical_chain(spec)

content = np.zeros((32, 32, 3), np.float32)
content[8:24, 14:18] = (1.0, 0.2, 0.05)
projection = run_geometric_light_table(chain, content, ray_count=20_000)
rgba = projection.preview_rgba
```

Replace the named `aperture` element with another mounted iris, grating, slit,
or custom canonical physical aperture via `spec.replace_component(...)`. The
exact compound-lens T2 payload does not change.

The compiled chain is the durable arbitrary emitter-to-sensor description.
The first executable output path is deliberately narrower: it currently
supports one textured emitter, one canonical physical aperture, one exact
`CompoundLens`, and one sensor. It uses `CompoundLens.trace`, applies the real
aperture opening geometrically, and produces linear, developed, and preview
sensor arrays. It is suitable for projection, focus, vignetting, and geometric
bokeh experiments. Its metadata explicitly reports that diffraction and
finite-material loss are not yet included.

Universal mixed T1/T2/T3/T4 dispatch, branch/fan-out execution, and coherent
sensor reduction remain follow-on executor work. The component graph now
retains the endpoint physics and native artifacts required for that work
instead of hiding them in a demo-specific scene.

### Mechanical assembly is authoritative

Inside the light-table environment, component coordinates should not be typed
directly into the transport graph. `LightTableAssemblySpec` is the mechanical
source of truth. It contains physical holders (post stands, rail carriages,
tube cells, cage plates, sensor backs, and emitter panels), their rail/tube
station, transverse placement, optical axis, and receiver interface. Each
mounted optic declares its own interface and datum.

Mount interfaces have a named standard, circular/square/rectangular face
shape, outer size, clear opening, and clocking policy. A swap is accepted only
when the optic mates with the holder. The clear opening remains explicit
apparatus geometry. Lowering holder-edge geometry into T1/T3 so that it
physically vignettes transport is a remaining step; the contract no longer
loses the dimensions needed to do it.

Resolution is deterministic:

1. validate holder/optic mechanical compatibility;
2. sort mounts by axial station, insertion sequence, and holder key;
3. resolve emitter, aperture, lens, and sensor poses from mounting datums;
4. lower that ordered physical assembly to `OpticalChainSpec`;
5. compile the typed optical transport graph.

An aperture tube cell and its lens holder may share a station; insertion
sequence gives their unambiguous optical order. `ResolvedLightTableAssembly`
also supplies `scene_manifest()` for `BellJarWorkspace`, so displayed stands
and holders and the optical job share the same assembly identity.

The first resolver supports the common collinear +X light-table rail. Emitters,
apertures, and sensor planes are positioned directly. Exact compound lenses
can be translated by front, back, center, or aperture-stop datum. An off-axis
or rotated exact lens currently fails explicitly because that requires a
rigidly transformed compound-lens parametric contract, not merely transformed
display geometry.
