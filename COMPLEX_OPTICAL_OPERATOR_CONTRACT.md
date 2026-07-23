# Complex optical operator contract

This is the mathematical contract shared by parametric ray transport, complex
ray records, wave-arena boundaries, and future compiled Maxwell patches. It is
not a second optical solver.

## Why this layer exists

Two complex numbers are not a complete polarization contract unless their
basis is known. A scalar Jacobian determinant is not a differential optical
map. Those shortcuts lose information precisely where ray, field, and
parametric representations meet.

The canonical state is therefore split into:

- a compact per-lane complex amplitude and metadata record;
- a persistent indexed transverse-basis table;
- a persistent indexed operator table containing one complex 2x2 Jones
  operator and one signed canonical 4x4 tangent map per record.

Ordinary geometric rays pay for none of the tables. A complex lane remains
exactly 64 bytes. Words 14 and 15, previously reserved, are now `basis_id` and
`operator_id`. The full records live in an arena-owned contiguous block and
are compiled or updated outside hot shader dispatch.

## Polarization convention

Every local basis is right handed:

```text
s × p = k
```

`k` is the propagation direction. Jones fields are columns:

```text
[E_s, E_p]^T
```

Basis changes are real orthogonal Jones operators obtained by projecting the
old basis onto the new basis. Interface scattering constructs one shared
s direction from the incidence plane, then constructs incident, reflected,
and transmitted p directions independently so every port remains
right-handed.

Lossless dielectric transmission coefficients are power-normalized:

```text
t_power = t_E sqrt(n_t cos(theta_t) / (n_i cos(theta_i)))
```

Consequently `|r|² + |t_power|² = 1` separately for s and p away from loss.
Total internal reflection retains its complex phase and unit magnitude.

Unpolarized and partially polarized light is not represented by pretending it
has one Jones vector. `PolarizationState.coherent_mode_decomposition()` emits
orthogonal Jones modes with power weights. Those modes require distinct
coherence identities and combine in intensity. Radial and azimuthal sources
resolve their Jones vector at the source-plane azimuth.

## Differential convention

At a port plane, the canonical transverse state is:

```text
[q_s, q_p, p_s, p_p]
p_s = n k_s
p_p = n k_p
```

The differential operator is the complete signed matrix:

```text
J = d(output canonical state) / d(input canonical state)
```

It is never replaced internally by `abs(det(J))`. A lossless smooth optical
map should be symplectic:

```text
J^T Omega J = Omega
det(J) = +1
```

The residual of that identity is reported. Sign, off-diagonal coupling, and
the complete matrix are required for:

- forward/backward reciprocity;
- differential ray propagation;
- pupil and sensor PDF conversion;
- field-amplitude conversion;
- caustic detection;
- sensitivity and optimization;
- composing modular graph operators.

The ray-to-field geometric amplitude factor for a nonsingular fixed-momentum
configuration map is:

```text
1 / sqrt(abs(det(dq_out/dq_in)))
```

A singular configuration map is a caustic. The adapter rejects it; a wave or
uniform-caustic operator must resolve that neighborhood. It must not clamp an
infinite ray amplitude and call the result physical.

## Carrier phase

Carrier phase uses optical path length:

```text
exp(i 2 pi f (OPL - OPL_reference) / c)
```

The reference path is shared by a coherent cohort. Cycles are reduced modulo
one before evaluating sine and cosine, avoiding avoidable loss of precision at
visible-light carrier counts. Relative phase is physical; an unrelated
per-ray reference is not allowed.

## Implemented surfaces

- `camera_software/complex_optical_operators.py`
  - deterministic CPU/reference bases, Jones algebra, Fresnel scattering,
    coherent-mode decomposition support, carrier phase, canonical tangent
    maps, composition/inversion, symplectic diagnostics, caustic-safe
    ray-to-field gain, and contiguous state-block packing, including fixed
    48-byte coherent source-mode records.
- `csrc/include/complex_optical_operators.h`
  - binding-independent 32-byte basis, 96-byte combined operator, and 48-byte
    source-mode records.
- `csrc/shaders/complex_optical_operators.glsl.inc`
  - matching binding-independent GLSL records and Jones/basis/determinant
    operations.
- `camera_software/vector_wave_adapter.py`
  - a reusable reference boundary adapter that keeps coherent modes separate,
    drives both native T4 transverse components through the production
    aperture-material and angular-spectrum kernels, and combines modes only
    into Stokes/analyzer intensity.
- `wave_transform_visual_demo.py --aperture-live`
  - qualifies linear, circular, radial, azimuthal, partial, and unpolarized
    source states through finite material blades with vector/Stokes and
    coherent-component diagnostic pages.
- `lens_optics_estimate_phase_space_jacobians`
  - threaded native exact-lens central finite differences returning the full
    signed 4x4 matrix, determinant, symplectic residual, and validity per ray.
    It selects the same per-spectral-lane refractive-index table as exact T2.
- optical graph contracts
  - advertise the operator schema and persistent indexed storage expected by
    exact T2 and ray/field boundary modules.

## Deliberately not claimed complete

The current native T4 arena already owns forward/backward s/p field planes.
The reusable reference adapter now seeds and propagates both components, so
the material and propagation kernels can be qualified without a demo-private
solver. The production ray-pipeline boundary still seeds only s and reduces
the exit field to one scalar ray amplitude. The reference adapter does not
disguise that old hot-path adapter as Jones-complete.

The next implementation gate is:

1. install the implemented fixed-stride source-mode block in
   `RayPipelineState`;
2. carry only its stable handle through ordinary ray stages;
3. apply interface Jones operators at physical material boundaries;
4. seed both T4 transverse components in the declared entry basis;
5. extract a Jones complex ray or retain the full field when reduction is
   scientifically invalid;
6. propagate and compose the canonical 4x4 map and OPL through exact T2;
7. qualify CPU/GLSL parity, Fresnel power, reciprocity, and caustic behavior.

No new per-ray N-lane payload and no shader-hot object construction are
authorized by this contract.

## Physical-aperture vector qualification

Run:

```powershell
python wave_transform_visual_demo.py --aperture-live --aperture-polarization radial
```

The live view uses the real finite-thickness blade material and production
native T4 kernels. `J` cycles source states; `[` and `]` rotate the source
orientation; Left/Right rotate the analyzer; `V` switches between Stokes and
coherent-component phase pages; `P` switches relative and absolute phase;
Space pauses.

The Stokes page shows the material aperture, total intensity, spatial
polarization, normalized Q and V, and a rotatable analyzer. The coherent page
shows s/p phase before and after propagation for one explicitly selected
coherence mode. Partial and unpolarized sources are never summed as complex
amplitudes. Because the current blade material is isotropic, it correctly
does not manufacture polarization conversion; radial and azimuthal inputs
still expose spatially varying vector diffraction.
