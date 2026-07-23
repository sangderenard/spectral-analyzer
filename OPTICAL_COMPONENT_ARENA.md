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
  --component aperture.iris --component-lanes 16 --size 384

python wave_transform_visual_demo.py --component-live `
  --component lens.default-camera --component-lanes 16 --size 384

python wave_transform_visual_demo.py --component-live `
  --component mirror.plane --component-lanes 16 --size 384

python wave_transform_visual_demo.py --component-live `
  --component pentaprism.finder --component-lanes 16 --size 384
```

The six panes have stable meanings:

1. representative/transport geometry;
2. the authoritative transport probe;
3. the compiled execution graph;
4. typed component ports;
5. canonical material-role bindings;
6. component identity, exact lane width, and controls.

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
