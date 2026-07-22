# Maxwell patch context and material contract

## Scope

`MaxwellPatchContext` is the intended full-complex electromagnetic engine for
localized material structure. Its job is to turn microscopic authored geometry
and constitutive laws into a reusable scattering artifact. Structural colour,
metasurfaces, diffraction gratings, sub-wavelength coatings, resonant cells,
and unusual anisotropic patches are its primary use cases.

It is deliberately not:

- a hidden branch of ordinary T4 propagation;
- a camera-room-scale visible-light voxel solver;
- an RGB texture generator;
- a replacement for bulk complex refractive index where bulk optics suffices;
- permission to call an expensive solver from a ray or field hot loop.

## Authored material declaration

The unified `spectral_material.Material` may round-trip a cold declaration:

```yaml
material:
  domain: em_optical
  ior: 1.52
  transmission: 0.9
  maxwell_patch:
    schema: 1
    asset: materials/structural/morpho_cell.usda
    prim: /Cell
    parameters:
      ridge_pitch_m: 6.4e-7
      ridge_height_m: 2.1e-7
    requested_product: polarized_scattering
    artifact: null
    fallback: bulk
```

The declaration is authoring metadata, not a hot tensor. `asset` and `prim`
select localized USD microgeometry. `parameters` bind named, unit-bearing
values declared by that asset. `requested_product` names a physical output
contract, not a solver algorithm. `artifact` may identify a compatible cached
compile. `fallback` must be explicit (`bulk`, `error`, or a named material).

The declaration must not contain GLSL bindings, SSBO offsets, grid strides,
backend-private tolerances, or assumptions about RCWA/FEM/FDTD. Those belong
to a compile manifest selected by the Maxwell engine.

## MaxwellPatchContext input

A resolved context contains:

- localized geometry and a defined local coordinate frame;
- frequency range and sampling policy;
- excitation and observation ports with normalized modes;
- complex, frequency-dependent permittivity and permeability tensors;
- optional conductivity, dispersion, loss, and nonlinear declarations;
- physical interface laws and explicit periodic/open/closed boundaries;
- incidence-angle, azimuth, and polarization domain;
- differentiable parameters and allowed bounds;
- requested accuracy, conservation, reciprocity, and convergence gates;
- backend selection and complete numerical compile manifest.

A metric tensor or Laplace--Beltrami operator is useful discretization
infrastructure, especially for curved coordinates and differentiability, but
is not itself this context. Full Maxwell qualification requires compatible
vector or differential-form unknowns, curl operators, the divergence
constraint, constitutive tensors, interface continuity, ports, and absorbing
boundaries.

## Compiled artifact

The durable product is a versioned `MaxwellPatchArtifact`, conceptually:

```text
identity + source/parameter hash
frequency/angle/azimuth validity domain
input and output port/modal bases
complex bidirectional polarized scattering operator S
optional reduced-order interpolation/surrogate data
absorption and unresolved/diffuse energy channels
normalization and coordinate/basis conventions
error bounds and convergence history
passivity, energy, and reciprocity test evidence
backend/build/hardware provenance
```

The canonical product is complex and polarized. A Jones/modal S-matrix is
preferred for coherent deterministic channels. Mueller or sampled stochastic
lobes may be derived for incoherent ray transport, but cannot replace the
coherent artifact. RGB previews are derived displays only.

## Runtime use

`MaterialDatabase.build_tensors()` continues to build its compact PBR,
spectral, enamel, and compatibility chunks unchanged. A later cold scene-
compile pass resolves `maxwell_patch` declarations, validates or builds
artifacts, and creates a separate compact artifact table. Only materials that
use a patch pay for a table entry.

At runtime:

- geometric transport samples the artifact's polarized directional channels;
- ordinary T4 applies its coherent bidirectional modal/Jones operator;
- coherence identity, spectral PDF, transport Jacobian, basis, and amplitude
  cross the existing complex sidecar boundary;
- out-of-domain queries follow the explicit fallback and are diagnosed;
- artifacts are immutable and cache-addressed during an exposure.

This is the coding direction: implement the artifact compiler and table beside
the material database, not inside its existing per-material hot rows, and make
both ray and wave consumers use the same versioned physical contract.

