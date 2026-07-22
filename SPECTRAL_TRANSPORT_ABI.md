# Spectral transport ABI

The renderer has two distinct spectral contracts.

## Fixed transport

A lane contains an authored physical frequency. The frequency is constant for
that lane and is appropriate for line sources, controlled laser-table work,
RGB-like experiments, depth, and other intentionally discrete exposures.

## Continuous LUT transport

A lane is an indirection signal, not a frequency bin and not a frequency
sample. It contains an integer index into the scene's spectral lookup library.
The selected LUT stores a continuous piecewise-linear probability density over
frequency.

At root-ray launch:

1. the active lane selects its LUT;
2. the root selects a reusable cached-sample identity within that LUT;
3. that identity resolves one continuous coordinate, exact `frequency_hz`, and sampling `pdf`;
4. those resolved values are written to the ray;
5. every reflection, refraction, medium segment, and child ray retains them.

Frequency is never resampled at a bounce or material. A one-lane continuous
payload is therefore valid: one lane can identify a distribution while each
root path receives a value from that distribution.

The cached-sample identity is part of BDPT side data. Camera and light
subpaths may connect only when both are fixed-band paths or when both carry
the same nonzero continuous sample identity. This prevents T5 from multiplying
independently sampled frequencies merely because they occupied the same lane.
The CPU and GPU launchers use the same lane-plus-sample key and LUT quantile;
the frequency then remains immutable for the complete subpath.

The native tracer caches validated, normalized LUT CDFs. Material response is
also prepared as a 32-knot cache and interpolated at the ray's exact frequency;
the lane count does not set the material cache resolution. CPU and GPU T1/T2/T3
carry the resolved frequency and PDF as path state. GPU adaptive sensor raygen
uses the same LUT inversion, and terminal sensor conversion evaluates color at
the resolved frequency with PDF correction.

The GPU upload contract currently permits up to 32 LUT profiles and 256 total
knots per scene. This is a storage ceiling, not spectral quantization: lookup
inversion returns values between knots.

## Scene-order representation

`continuous_spectral_lut` orders embed:

- `lookup_tables`: stable keys, increasing frequency knots, and density values;
- `lanes[*].lut_index`: the lookup signal;
- no lane-level `frequency_hz`.

The former `sampled_continuous_spectral` scene contract was not continuous
transport: it sampled once while building the scene and then treated those
values as fixed bands. ABI version 2 rejects those old orders instead of
silently giving them the new name.

## Calibration execution

Calibration selection remains explicit (`AUTO: OFF`). Fixed prism modes use
fixed frequencies. Continuous prism modes use the LUT contract at payload
widths 1, 3, 8, 16, or 32. Every calibration result is a real exposure of its
real scene through the physical camera path.
