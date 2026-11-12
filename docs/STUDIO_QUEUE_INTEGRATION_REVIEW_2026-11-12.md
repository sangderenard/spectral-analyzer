# Pinned CPU integration and remaining coverage review

## Integration state

The owned Spectral branch merged canonical `009178d94686143ac3ba0a8e82924a8d3a8c7a02`
as `9a7039e4d5d8ab60fb55fb17fbb76d1350d712b5`. The only overlapping production
file was `marble_preview_worker.py`; camera-trigger timing/generator changes and
our response/queue parameters merged without conflicts. No canonical file,
running service, deployment, user setting or hardware hold was changed.

The owned Turing receiver merged pinned canonical
`d030e18d7224df39f3499645bde850f79c44ed77` as `a4102c19`; receipt commit
`50f5b2c4dc8f8de38c701dec21e15a7b1136ec8d` documents its verification.
The new flash/trigger contract coexists with response/camera/ancestry options.
Canonical dirty Stage Programmer, lighting, UI/export and test changes remain
separate. They were read only, not copied into the integration branch.

CPU results:229 Spectral tests passed, including16 offline shader variants and
the new trigger/preview regressions. The original trigger-grid software check
also passed. Its generated grid/solver diagnostics are not optical renders.
Turing:54 backend tests passed, plus18 bounded-index cases, v1/v2/v3 receiver,
lifecycle, gem-cutter and actual-store roundtrip checks. Its committed retouch
test has a mock missing `classList.toggle`; the canonical dirty test contains
the pending adaptation. That unrelated uncommitted bundle was not imported or
claimed verified. Backend tests used an already-installed Python311 interpreter
with PyYAML; no package installation was needed.

## Concrete remaining path/volume gaps

1. Corrected opaque PBR currently bypasses native edge diffraction. The branch
   inserted by `studio_opaque_pbr.with_opaque_pbr` executes before the three
   native edge enqueue calls. Its successful path continues; rejected sampling
   or throughput breaks. Those opaque hits therefore never generate their
   native diffraction children. This predates the queue owner update.
   Fluorescent, emissive and absorber flags are excluded from that PBR branch,
   so this does not suppress fluorescence through the same mechanism.
   Restoring diffraction there requires a deliberate optical/RNG order and
   accounting decision, not a record-delivery patch.
2. `studio_directional_media.with_internal_medium_recording` explicitly refuses
   an ExtinctionGrid combined with canonical material/region Beer recording.
   The product needs integration over both partitions; it is unimplemented.
   Material/region interiors alone are supported by the current contract.
3. Forward/reactive miss records retain termination5 and `hp=ro`, not a finite
   escaped interval. Camera miss word11 is explicitly NaN (`0x7fc00000u`), even
   though the capture call passes zero before `sci_begin` specializes it. Never
   project either record to invented bounds or call missing support zero energy.
4. Invalid ownership (`medium < -1`) is excluded from directional integration,
   not treated as air or darkness. Camera observation does not retain NEE sample
   lineage or certify untraced paths/field response evaluation.
5. Bounded FIFO eviction is explicit unavailable evidence. A complete retained
   snapshot is not exhaustive volume classification, path coverage, deposited
   energy, convergence, or Phong readiness.

The new queue guarantee is narrower: all native children actually generated
within the admitted parent bounds are consumed before completion, or the field
fails explicitly. It cannot restore optical paths never generated upstream.

## Non-render hardware-test readiness

Using the existing CPU-only `tests/studio_forward_source_control.py` composer,
plus the same shared scalar/response and queued-ancestry composers, the actual
region/sphere/PBR/internal-medium/root-sweep source assembles to:

- 118,053 UTF-8 bytes
- SHA256 `32f1f492ff61aefd18fbbc8ebdd078cdc43c83374799575df68a375b8796734b`
- 16 declared compute storage blocks; at least24 binding slots
- 10 exact copied RHS seams across forward/reactive paths
- No renderer import, GL context, shader compiler invocation or GPU execution
  in this source-capture check

This is not the historical bespoke-splitter GLSL artifact. It is the current
whole native composer output and contains both forward/reactive entry paths.
Resource counts are source-side requirements, not a device/link success claim.

A link-only check can use the existing renderer `_prog` on this whole assembled
source after the required notice. Do not mistake `_gpu_ray_field_prebuilt` for
a link-only call: even `initial_rays=0` dispatches128 warmup rays when sources
exist. The existing small dielectric fixture in
`tests/studio_internal_field_integrated_gpu_check.py` is the appropriate base
for later bounded compute validation, but does not yet test the new opted-in
owner with a genuinely nonempty child queue. An empty-queue pass would not
validate reactive draining. Camera-pair readback requires its own actual test.

No GPU link, compute dispatch, optical render, live request, push or deployment
was performed. Complete-data/render readiness remains unestablished. Announce
hardware activity beforehand; show any eventual actual render only once the
complete-data constraint has been met.

## Resumed joint-extinction lane, 2026-11-12

The retained joint observer now integrates the installed material extinction and
ExtinctionGrid over the grid's existing traversal, including material-only
intervals outside that grid. Primary transport helpers and RNG calls remain
unchanged. Invalid or incomplete traversal is unknown, not a vacuum remainder.
The production prebuilt assembly previously installed its internal observer
before replacing the source with the volume variant, losing the observer. The
observer now attaches after final source selection. A test executes that actual
AST statement without importing the renderer or creating GL state.

Verification on the retained worktree:
- 52 tests passed: joint observer, directional media, forward response copying.
- 64 tests passed: volume extinction, internal config, native source control,
  queue ancestry/transport and opaque PBR.
- Offline GLSL checks in those tests are syntax compilation, not driver linking.

The worker's joint-volume admission remains closed pending real device validation;
no GPU link, dispatch, render, server change or deployment was done. Source-side
success is not screen completeness or application readiness.

Diffraction remains pending a real allocation choice: native per-edge offers
can each be half the parent while leaving it untouched. The user authorized
parent debit. Sequential debits change later edge weights by vertex order;
same-parent offers normalized to one budget preserve ratios but introduce an
allocation rule. Neither is an existing cascade parameter. Do not silently choose
one. Fluorescence is independently additive and not fixed by a diffraction patch.

## Approved proportional diffraction budget

The owner subsequently approved computing all three edge offers from the same
incident energy and scaling proportionally only when total offers exceed that
energy. This supersedes the pending allocation choice above.

The explicit corrected-Studio PBR composer now preserves native Keller offers,
eligibility, cone sampling, phase, bounce bounds and single-generation queue.
Successful queue writes alone debit the parent; rejected writes cannot consume
parent energy. The PBR branch follows edge processing, so its break/continue no
longer bypasses diffraction. Ordinary unspecialized Pluck source is unchanged.
Fluorescence remains separately additive and is not claimed energy-conserving.

Actual debit is copied into an explicit kind2 response row before subsequent
BSDF factor reuse. Response admission and the shared buffer tail both reserve
two rows per ray-bounce, rejecting insufficient budgets rather than shrinking
work or silently dropping this record. Metadata identifies intermediate debit
rows; they are not terminal samples. Unknown/malformed packets retain existing
validity checks. The additional response data remains opt-in.

CPU/offline checks: 91 tests passed across budget, PBR, joint observer, queue and
response copies; 81 additional factory/retention/config/volume tests passed.
Float32 oracle added with bounded rounding tolerance, not bit-exact conservation.
Offline shader compilation is not GPU driver link or execution evidence.
No GPU, render, live deployment, or server restart performed.
