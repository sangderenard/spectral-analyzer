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

The project owner has not finalized repository licensing. This code may
ultimately be covered by more than one license, and the source and containing
repositories may select or combine different compatible terms. This snapshot
does not invent or narrow those terms; it preserves provenance and defers to
the licenses the owner later declares.
