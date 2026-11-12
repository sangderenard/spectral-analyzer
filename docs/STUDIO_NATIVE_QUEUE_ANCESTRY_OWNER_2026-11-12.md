# Opt-in native queued ancestry and bounded owner

This updates the existing prebuilt Studio renderer, not the Pluck application
workflow, the compiler, Nodus, a stage splitter or a second transport model.
The ordinary `_gpu_ray_field` and canonical shader literals are unchanged.

## Explicit admission and exact records

`light_field_reactive_queue='native-ancestry-v1'` requires evaluated responses
and the existing v6 internal-medium/root-sweep observer. Absent leaves the
forward-only prebuilt response contract and legacy64-byte queue unchanged.
The preview worker still refuses evaluated-response hardware execution before
GL creation. No UI default, permissions, deployment or hardware hold changed.

The existing PendingRay's first64 bytes remain unchanged. An explicit96-byte
version appends two uvec4s:

- Actual parent dispatch token, parent ray ordinal, selected baked-source row,
  and parent bounce.
- Copied root replay-weight bits, observed enqueue cause (edge diffraction or
  fluorescence), and two reserved zero words.

Original origin/direction/packet/remaining budget/spawn normal/triangle are
unchanged. Parent tokens resolve through immutable dispatch metadata to actual
source population/hash, scene, seed and root-sampling context. No root is
inferred from triangle, child frequency, queue ordinal or current population.
The exact current replay weight is copied, not recomputed for secondary jobs.
Weight zero retains unavailable normalization; it does not certify darkness.

The existing512-byte response rows carry ancestry via explicitly versioned
forward-v2 records and scene-v3 envelopes. See
`STUDIO_FORWARD_ANCESTRY_TRANSPORT_V3.md`. Legacy and new rows are not relabeled
or mixed. Completed camera samples and camera-pair contracts remain separate.

## Existing owner, bounded consumption

After each actual forward chunk the same prebuilt owner:

1. Reads the existing pending counter and consumes the forward response copy.
2. Dispatches its original PASS_REACTIVE body in chunks no larger than the
   already admitted response ray capacity, with the native seed31415926.
3. Reads and consumes each actual response before advancing its queue cursor.
4. Verifies the single-generation queue counter did not change, resets only
   counter word4, and restores forward mode/seed.

Reactive queue indexing now uses the real batch offset. A logical enqueue base
retains the existing seed/index convention across local counter resets; it is
bounded to uint32 and cannot silently wrap. Child optical expressions, random
calls, enqueue order and remaining bounce budgets are preserved. Reactive
directional observations inherit their actual parent replay weight. They are
not counted as new source roots. Root completion and scalar publication happen
only after successful forward-plus-reactive consumption.

The queue cannot cross parent dispatch, source population, geometry or clear
boundaries. Guards precede source/material/geometry metadata and buffer updates.
An exception after submission, readback, consumption, reset or restore leaves
the additive field unusable for retry. Recovery is destruction/recreation, not
rerunning the same offset or clearing away evidence of uncertain work.

## Resource admission

The native source has three edge opportunities and one fluorescence opportunity
per forward bounce; the final bounce has no remaining child budget. The new
queue bound is `4 * dispatch_batch * (max_bounces - 1)`. Composer seam checks
reject a changed native enqueue/bounce contract rather than assuming this bound
still holds. One-bounce jobs validly have zero children and a minimum96-byte
allocation; they never dispatch reactive work.

Per-SSBO admission occurs before allocation. The larger queue allocation is
deferred until the existing measured field allocation/publication/external
reservations and response plan admit it within the512MiB working-memory ceiling.
No ray/bounce schedule is reduced to fit. Queue overflow is an explicit failure,
not an accepted partial result. Validation uses512-row chunks and reserves64KiB
of scratch through allocation and rebind.

## Verification and limits

CPU fixtures exercise actual shader composition, actual owner/begin metadata,
fake-GL dispatch offsets, source lineage, real store serialization, empty queues,
overflow and uncertain submission/consumer/reset failures. Offline
glslangValidator checks cover legacy/new ancestry, terminal/all-response,
region/nonregion and sphere/nonsphere combinations. These are CPU syntax/type
checks, not GPU linking or executed optical evidence.

Final focused/regression run:208 passed, including16 offline shader variants.
Turing request/lease regressions:27 passed. Existing decoder/lifecycle programs
and18 bounded-index cases passed. Actual-store fixture roundtrips preserve256
ancestry words and512 camera-pair words (64 query references) exactly.

The broad run exposed an existing source-population test fixture that ignored
`glGetBufferSubData` offsets and returned the entire atlas when asked for its
overflow word. The same failure was reproduced in the unchanged canonical
checkout. Only the owned test mock was corrected to honor offset/length; no
directional accumulation or readback production code changed for that issue.

No GPU link, compute dispatch, render, live request, push or deployment ran.
The queue ownership/provenance gap is now implemented and CPU-checked. Actual
GPU correctness, throughput, complete interior/path observations and usable
energy/Phong interpretation remain unproven. Query/response presence is not
coverage, energy deposition or proof of darkness. Do not render until complete
data is established; notify the owner before any GPU link and show any eventual
actual render for accumulation/debug review.
