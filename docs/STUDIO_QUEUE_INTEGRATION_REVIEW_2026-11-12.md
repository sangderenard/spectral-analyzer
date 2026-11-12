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
