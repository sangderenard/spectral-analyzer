# Studio wave transition handoff (2026-11-14)

## Executable boundary

`ray_tracer_resolve_studio_wave_interface` now returns the authenticated hit
point and normalized incident vector with flags for an authored region surface,
base material change, real-index change, imaginary-index change, and an exact
sphere. It retains the two-sided per-frequency optical indices and the
wavelength-derived local margin. A dye interface may have identical base
materials but a changed imaginary index; it must remain a transition. A
zero-contrast authored surface remains identified as geometry, but does not
claim an optical contrast. These flags describe a **pre-scatter crossing**;
they do not claim that T4 has solved it.

## Procedure to complete the photo path

1. T1/T2 must emit each real crossing as a correlated vector record at the
   stage boundary, preserving root tag, subpath/vertex, launch time, world
   hit point, incident direction, outward normal, triangle/region identity,
   frequency lane and complex amplitude. The existing classifier supplies
   the optical flags and side indices; the record needs explicit job and
   generation provenance in the Nodus envelope. A stop/aperture edge needs
   its own geometry flag, even if no material index changes.
2. Gather contiguous wavelength-scale interface neighborhoods from the
   *flagged triangle lists* and exact curved surfaces. The margin is local
   to every boundary: host exterior, bubbles, rods, dye transitions and lens
   aperture edges. Do not wrap the entire marble in one coarse volume. A
   context is admitted only when its spatial pitch and extent resolve its
   requested wavelengths. Keep noncontiguous neighborhoods separate.
3. Feed each admitted crossing's complex field, Jones/coherence lineage and
   medium state to the appropriate T4 arena through Nodus tensor-state
   addresses. The present 2D angular-spectrum T4 is a transverse propagation
   operator. The SymPy 3D scalar stencil and its Nodus 64³ GPU trial are a
   separate volume operator; production use needs two time levels, neighbor
   mapping, boundaries and a Courant gate. Neither operator can silently
   substitute for the other. The exact conic T2 aperture handoff already
   exists for CPU traversal; GPU interior-stop support is still absent.
4. On exit, form continuation vectors with position, direction, phase,
   polarization, spectral weights and medium ownership. Match the wave exits
   to their input lineage before returning them to T1/T3. The known
   separated-arena power and Jones-reset failures must be corrected before
   claiming a physically weighted image. T5 must receive both estimator
   families and their PDFs before film integration.
5. Expose Nodus taps after the stage that produced them: crossing vectors
   after T1/T2, path records after T3/T4, linear endpoint values before
   integration, and film values afterward. Each tap needs an application
   envelope (job, generation, stage, layout, count, completion) over the
   existing TCP byte-stream row. The current aggregate pipeline receipt does
   not deliver intermediate products. Completion must be based on finished
   stage work, including overflow or refusal, not on a socket being open.

## Qualification gates

Use an authored small subject with a host surface, embedded bubble, dye
boundary, and a thick-lens aperture edge. Verify each true crossing is flagged
once with correct two-sided index; reversed traversal swaps sides. Verify
local pitch and Courant/admission refusals, field energy and Jones continuity
across separated contexts, and compare the final linear sensor against the
non-wave transport with a declared tolerance. A rendered image is evidence
only after those gates and a native Nodus T1–T5 job pass.
