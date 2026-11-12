# Reactive queue ownership and dispatch gaps

Read-only source assessment at Spectral `c68d27deaf29a6b06058ec97ffce455f852a0a6b`.
No queue scheduler, optical law, shader entry, GPU dispatch or permission was
changed by this assessment. The evaluated-response hardware hold remains.

## Existing implementation, not a missing optical model

`demo_pluck_gl.py:7486-7506` contains a real one-shot reactive pass in the ordinary
renderer: read counter word 4, bound the work by the actual queue capacity,
select PASS_REACTIVE, use the original seed 31415926, dispatch the full stored
queue with zero batch offset, issue the existing barriers, then restore forward
mode. `reactive_main` at 4550 consumes these already-produced rays and uses the
canonical intersection, medium, material and continuation calculations.

The same queue receives edge-diffraction rays (4148-4149) and fluorescent/
re-emission rays (4425-4426). The current 64-byte `PendingRay` ABI (3366-3370)
contains four vec4 fields: origin/flags, direction/remaining bounces, packet,
and spawning normal/triangle. The triangle is not a baked-source root. Neither
the parent source root nor population/replay identity is retained. Both enqueue
paths use PRAY_NO_REACTIVE; that flag alone does not distinguish their origin.

## Why Studio cannot simply call that pass

1. The opted-in prebuilt program is deliberately specialized to forward-only
   (`studio_forward_responses.py:304-327`), and the owner records that contract.
   It must not claim that selecting mode 2 enables a removed entry.
2. `main` passes local `gid` to `reactive_main` (`demo_pluck_gl.py:4321-4324`).
   `uBatchOffset` affects RNG ray identity, but not that queue index. Multiple
   chunks would reprocess the queue prefix unless the actual native entry
   becomes explicitly offset-aware. A metadata offset is not execution support.
3. Queue capacity is up to four forward batches (7782), while transient response
   storage is admitted for one forward batch times its bounce bound (8013).
   Dispatching the full queue is not covered by that response allocation.
4. The prebuilt owner has no drain cursor/reset transaction. `_clear_prebuilt`
   clears scalar/directional products, not queue counter/storage (8323-8346).
   `record_queue_observation` correctly treats its count as allocation-lifetime
   state and keeps missing work sticky. No current path may report a reset as
   completed secondary work.
5. Source populations and scene buffers may change independently of this queue.
   A lifetime backlog cannot be stamped with the new source/scene identity and
   treated as if those were its enqueue conditions.
6. The current root-sweep recorder initializes replay weight to zero and assigns
   it only in the forward root branch (`studio_directional_field.py:266-268,
   295-297`). The missing queue root identity cannot be guessed from the spawning
   triangle or current source buffer. The existing arrival export explicitly
   declines normalized reactive contributions (`studio_field_arrivals.py:262`).

The separate C++ exposure backend has its own in-path reactive spectral handling;
it does not supply an interchangeable owner for this GLSL queue and its buffers.
No compatible existing prebuilt drain/ancestry implementation was found.

## Supported next renderer work

For the requested all-path normalized field, extend the existing queue contract
to copy real parent-root/replay/population provenance at the actual enqueue
sites. Preserve distinct enqueue origin when needed; never infer diffraction
versus fluorescence from packet values. Then make the native reactive entry and
existing prebuilt owner support bounded, offset-aware consumption under a queue
lifecycle that cannot cross scene/source mutation unnoticed. Use the original
reactive body and actual remaining bounce budgets, not another transport model.

Only completed real execution may advance dispatched counts. Overflow,
unprocessed work and uncertain completion must remain unavailable; retrying an
uncertain chunk must not duplicate its scalar/response accumulation. An explicit
queue ABI and ownership change is needed before this can be implemented as a
correct integration, followed by CPU ownership/contract tests and separately
announced hardware validation. A detached scheduler or permission flag alone
would not close these gaps.

## Independent camera-index work

Camera v2 pairs retain actual query origin (8-10), parameter distance (11), and
incoming direction (12-14). Finite positive hit queries can support separately
labeled *derived query-parametric geometry* using `origin + direction * distance`
without direction normalization. Words 116-118 are the retained first-hit point,
not every current endpoint. Misses store NaN distance and must not be extended
to guessed grid bounds. Status 4 can have a genuine intersection query but no
surface evaluation. Invalid medium -2, partial band paths and untouched post-
region bits must not become valid energy, coverage or empty-region statements.

This bounded indexing step can proceed without changing queue scheduling,
optics, raw records, outer attachment, Phong application or hardware readiness.

## Implemented and verified independently

Turing receiver branch `codex/camera-pair-receiver-20261112` contains the existing
bounded record index as `b99d7f9d` and the explicit camera query extension as
`72f91419`, with conservative clipping correction `2d605524`.
`cameraQuerySegments: true` requires the v2 pair envelope and a
positive `recordIndexBytes` budget; both remain opt-in. Support bit 4 means only
derived query-parametric geometry. Copied endpoint/segment support bits 1/2
retain their prior meaning. Unresolved intervals publish no guessed traversal.

The producer-side cross-language fixture now accepts `--camera-query-segments`
(which implies `--camera-pairs`). Its two native pairs have non-unit direction,
both endpoints outside the ROI, statuses 4 and 7, and a deliberately unrelated
retained first-hit point. The real producer serializer and actual Turing decoder
round-tripped all 512 words exactly and indexed 64 query references, without
changing outer unlocated attachment or claiming energy/application readiness.

CPU verification from this Spectral worktree, with the canonical Spectral root
on `PYTHONPATH` for its existing installed CPU module:

```powershell
$env:PYTHONPATH='C:\dev\Powershell\spectral-analyzer'
python tests/studio_scene_response_transport_check.py C:\Users\alber\turing-camera-pair-receiver-20261112 --camera-query-segments
python tests/studio_scene_response_transport_check.py C:\Users\alber\turing-camera-pair-receiver-20261112 --camera-pairs
python tests/studio_scene_response_transport_check.py C:\Users\alber\turing-camera-pair-receiver-20261112 --require-record-index
python tests/studio_scene_response_transport_check.py C:\Users\alber\turing-vector-index-20261112 --require-record-index
```

All four passed: exact v2 pairs with and without the new query index, and exact
v1 forward/reactive segments through both the updated and original index.
Review caught an extreme-coordinate case where float64 entry/exit parameters
collapse to equality. The shared walker now marks that ambiguous geometry
unavailable, without tolerance or invented cells; strictly disjoint intervals
stay empty. Both query and copied-segment regressions passed, including exact
word preservation. The receiver's index suite now has 18 passing cases.
No live job, GPU compilation/link, render, push, or deployment was performed.
The reactive gaps above are not permission denials or hardware failures; the
necessary execution/provenance contracts are not yet implemented. Nothing here
unlocks the hardware hold or completes the requested all-path deposition field.
