# fftfree vendored snapshot

- Source: `C:\dev\Powershell\fftfree`
- Source Git commit: `642dcf4`
- Snapshot date: 2026-07-23
- Imported by explicit project-owner authorization.

This directory contains source, public headers, focused project tests, build
metadata, and implementation documentation. The source repository's `.git`,
build products, editor state, generated logs, media experiments, images,
trained weights, and binaries were intentionally excluded.

## Integration boundary

`fftfree` is a transform executor candidate, not a wave solver. T4 retains
authority over persistent field ownership, angular-spectrum transfer
functions, boundary conditions, polarization, spectral lanes, and pipeline
scheduling.

Initial adoption is limited to the complex C2C Cooley–Tukey path. Stockham,
mixed-radix, real transforms, magnitude/polar packing, STFT helpers, recovery
operators, and metadata capture remain opt-in until isolated parity and
allocation tests demonstrate their suitability.

The useful T4-facing capabilities are:

- in-place complex batched transforms;
- explicit axis and batch strides;
- preplanned twiddles and work arenas;
- preallocated 2D transpose workspace;
- caller-owned dispatch;
- optional stage/twiddle/layout telemetry.

The vendored CMake project is not added as a subdirectory of the production
build yet. Its current test registration also exposes dependency tests and one
focused test executable is absent from the old build tree. Integration should
use a narrow adapter target and a clean, isolated test configuration.

## Licensing

On 2026-11-11 the project owner selected MIT for owner-authored material in
this repository, including the first-party parts of this snapshot. See the
root LICENSE and LICENSING.md. The recorded source commit and import history
remain unchanged. This replaces the earlier deferral of license selection.

The source history attributes commits to Albert and sangderenard; that is
provenance evidence, not a claim of ownership of every dependency. Eigen and
other third-party material retain their own terms and notices. This declaration
does not modify the separate fftfree source repository.
