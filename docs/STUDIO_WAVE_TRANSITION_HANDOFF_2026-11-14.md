# Studio wave transition handoff (2026-11-14)

## Executable boundary

`ray_tracer_resolve_studio_transition_vector_v1` now returns the authenticated hit
point and normalized incident vector with flags for an authored region surface,
base material change, real-index change, imaginary-index change, and an exact
sphere. It retains the two-sided per-frequency optical indices and the
wavelength-derived local margin. The older C result ABI remains unchanged.
A dye interface may have identical base
materials but a changed imaginary index; it must remain a transition. A
zero-contrast authored surface remains identified as geometry, but does not
claim an optical contrast. These flags describe a **pre-scatter crossing**;
they do not claim that T4 has solved it.

## Procedure to complete the photo path

1. T2 now emits authored Studio triangle and exact-sphere hits into a separate
   native `RayTransitionRecordV1` queue before T3. Each row carries root tag,
   subpath/vertex, launch time, hit point, incident direction, outward normal,
   triangle/region identity, resolved frequency lanes, complex amplitude,
   two-sided optical indices, classifier flags, and an explicit status. The
   Python binding drains versioned raw rows without consuming ordinary ray
   output; pipeline checkpoints include the queue. This is an in-process
   off-ramp, not yet a bounded or TCP-delivered Nodus stream. The Nodus
   envelope still needs job and generation provenance. Exact lens conic and
   stop events remain on the separate optical-event path; a stop/aperture
   edge needs its own geometry flag even when no material index changes.
2. Gather contiguous wavelength-scale interface neighborhoods from the
   *flagged triangle lists* and exact curved surfaces. The margin is local
   to every boundary: host exterior, bubbles, rods, dye transitions and lens
   aperture edges. Do not wrap the entire marble in one coarse volume. A
   context is admitted only when its spatial pitch and extent resolve its
   requested wavelengths. Keep noncontiguous neighborhoods separate.
   `ray_tracer_collect_studio_wave_triangles` now supplies a bounded local
   candidate list from the installed Studio region BVHs and exact sphere
   shells. It tests actual triangle distance after BVH pruning, returns sorted
   triangle IDs, and refuses insufficient output capacity without writing a
   partial list. The caller sets a search radius explicitly. This identifies
   nearby authored surfaces; it does not yet partition them into connected
   3D arenas or admit a numerical grid.
3. Feed each admitted crossing's complex field, Jones/coherence lineage and
   medium state to the appropriate T4 arena through Nodus tensor-state
   addresses. The present 2D angular-spectrum T4 is a transverse propagation
   operator. The SymPy 3D scalar stencil is a real compiled kernel running
   through the Nodus GPU row and tensor state library on three GPUs. It is a
   separate volume operator; arena service use needs two time levels, neighbor
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

## Exact parametric lens elements

The graph contract now enumerates every surface in the canonical exact T2
payload as an indexed element. Conic interfaces carry their authored geometry
and per-lane before/after indices. A physical aperture wave context must bind
to exactly one indexed stop; a missing or ambiguous stop is refused. The
native T2 traversal emits an optical event at the exact stop's T4 entry with
that same element index and an arena ID, before it yields to the wave arena. This identifies
the internal lens transition even though the lens still executes as one fused
T2 program. It does not turn the scalar 3D volume kernel into the optical T4
operator or make all lens paths available as a Nodus crossing-vector stream.
