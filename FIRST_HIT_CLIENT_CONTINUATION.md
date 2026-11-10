# Composed sensor first-hit continuation (experimental, 2026-11-09)

This is an actual producer/consumer seam, not the local distance diagnostic.
`camera_designer/studio_first_hit_loan.py` captures the original sensor shader's
camera launch and resumes the same composed transport after a triangle result.
`SensorAccumulator._dispatch_once` rejects ordinary work while a strip is on
loan. Capture does not advance the row, pass, frame, sensor image or history.

The owner GL thread creates `FirstHitLoan`, calls `issue`, retains the strip,
and eventually calls `complete` or explicitly cancels with `close`. HTTP
threads must only transport messages; they must never call these GL methods.
Expired work remains pending. Reissue changes the capability token but keeps
the exact rays. Duplicate, expired, changed-scene and malformed results are
rejected. An uncertain GPU commit stays blocked rather than being retried.

`issue` returns the source hash, geometry hash, scene revision, sample prefix,
capability token, exact stride-eight float32 rays and the actual BVH triangle
candidate set. The browser-law hash is a separate identity: it is not equal
to the complete composed-shader hash. The browser must use Studio's current
`hit_tri` law. The native `ray_bvh_intersect` thresholds are different and
cannot silently substitute for this law on miniature geometry.

The current admission is deliberately narrow: at most 4096 triangles/rays,
unique row strips, one sample per pixel, no explicit material regions or
sample-evidence side channel. Canonical sphere competition stays on the server.
Regions require their clipping/ownership and position-dependent precision
contract. Later bounces, shadow queries, volume transport and sensor deposition
also still execute server-side. This is not an entire browser renderer and
cannot render without server computation.

The scene, forward field and accumulator must be frozen on their owning thread
until the strip resolves. Accumulator settings, geometry/shading/material/source
buffers, history and forward/output texture bytes are checked before and after
validation. These deliberately expensive readbacks have a 64 MiB admission cap;
production needs exclusive versioned resource ownership instead of this audit
mechanism. They do not support a performance-savings claim.
No live worker or public endpoint is enabled by this module. No billing or
credit service is invented. Budget exhaustion must pause retained work when
no allowed compute provider is available, not skip or normalize missing rays.

The completion API requires an authoritative validator. Finite numbers and
valid triangle IDs alone cannot prove nearest hits or misses. Current isolated
tests recompute the reference, so they demonstrate parity, not time savings.
Do not claim avoided server computation until an explicit result-trust model
and an actual renderer-work replacement are measured.

## Verification

`python tests/studio_first_hit_loan_gpu_check.py` executes real GL, captures 256
aperture-jittered rays, returns authoritative first hits, and compares the full
sensor image bitwise with uninterrupted tracing. It checks no capture deposit,
blocked pending dispatch, duplicate rejection, expired work retention, reissue
of identical rays and explicit cancellation/server fallback. It contacts no
production API. An optional JSON argument supplies browser-computed hits;
the receipt records the external result path, distinguishing browser-produced
WebGPU output from Node WASM execution. Both returned results passed sensor
resume parity in separate tests.

`python tests/studio_first_hit_loan_gpu_check.py --relay` runs the real
origin-bound mailbox and HTTP handoff to a Node WASM worker, takes the result
once on the GL owner thread, validates it and commits. Set
`STUDIO_LOAN_MULTI=1` for three triangles including near/far hits, misses and
an exact tie: 256 rays, 82 hits, bitwise full-image parity. Mutation gates reject
same-handle material updates and validator callbacks that change camera state.

`--nodus` uses the actual native Nodus two-table contract graph and declared
result FIFO, with its socket ServicePeer running pure compiled WASM queries.
Opaque contract payload references resolve only within the authorized engine
fixture. Native routing callbacks defer; the GL owner commits outside those
callbacks and only then acknowledges commitment. `--nodus-recovery` injects
one false result, authoritatively rejects it, receives a fresh native offer,
and commits the correct returned intersections without double deposition.
Both actual Nodus tests passed on the three-triangle fixture. Source APIs enforce
the loan's GL owner thread. These are not live service activation or a result
trust model for arbitrary remote participants.

Receipts and the serialized exact-ray request live under
`build/dot-first-hit-loan-20261109/`. Initial GPU gate passed; production
deployment and default-on participation remain unfinished.

## 2026-11-09 resident scene and sphere competition gate

`scene_descriptor()` separates immutable packed query geometry from camera/sample
leases. `issue(include_scene=False)` omits heavy geometry only for a recipient
that has acknowledged the exact geometry hash and compiled query identity. This
is an integration API, not an enabled cache or permission to trust arbitrary
page meshes. Browser Room geometry and native Exposure camera/stage packing can
differ. Default packets remain backward-compatible with complete geometry.

The isolated 256-ray, three-triangle native Nodus rejection/reoffer test now also
passes with an analytic sphere competing on the server. Full sensor output is
bitwise equal to uninterrupted rendering. Client sphere computation is not
claimed. Compiled host-BVH WASM results also passed this same sensor gate.

`tests/export_stage_candidate_query.py` exports a saved 95,217-triangle,
65,535-node stage and authoritative unchanged GPU nearest-triangle results for
256 camera-origin probes. This is explicitly plain-law triangle-candidate
validation, not the region-aware scene render: material-region thresholds,
filters and partition boundaries remain excluded from loan admission.

The native gate now performs fresh authoritative GPU verification only inside
completion after a structurally valid client result arrives. Receipt
`returned_batches` counts those validator-admitted batches, not malformed
transport messages; `service_calls` counts client attempts. `verified_rays`
reports the actual 1:1 checking workload separately from committed samples.
Both false-but-well-formed and missing-ray returns are tested: malformed batches
are rejected without GPU work, reoffered through native Nodus, and never cause a
sample to be skipped. Initial test reference dispatches remain separate baseline
expectation work, not the operational completion validator.

## Region primary candidate continuation (isolated, not production enabled)

Explicit `query_variant='region-bvh'` now admits original region shaders up to
200,000 triangles; the default plain contract remains capped at4,096. Shader
variant and region presence must agree. Capture exports the original region
position-dependent minimum parameter. Region SSBO bytes are included in the
resource freeze. Origin classification runs normally before the substituted
primary query, restores region filter=-1, and the server still competes analytic
spheres and computes rod partition boundaries and every later transport step.
This does not transfer region classification, sphere queries or full paths to
clients. The client must use the separately hashed region law.

The actual saved stage95,217triangles,3regions and analytic spheres passes
16x16 full sensor bitwise reference-resume parity and retention/mutation gates.
`stage-region-request.json` contains exact captured camera rays and original GPU
candidate expectations for independent compiled-client parity. It is not a
claim that this isolated test is serving the production job loop.

The actual region stage subsequently passed real compiled WASM BVH results and
then the native Nodus socket/FIFO round trip. The first complete returned batch
was deliberately corrupted, rejected after authoritative checking, and reoffered
with a fresh native generation. The second committed exactly once. The full
sensor remains bitwise equal to the uninterrupted reference. Two completed
256-ray returns cost512 server verification rays. This is correct cooperative
execution, not a net compute-savings measurement. See
`build/dot-first-hit-loan-20261109/stage-region-gpu-receipt.json`.

The isolated participant is `tests/first_hit_bvh_client.mjs`; it invokes the same
compiled BVH runtime and chooses the region-specific artifacts. The operational
authenticated browser job entry point and worker-loop hook remain unconnected.

Reusable completion validator is now `studio_completed_hit_verifier.py`.
Construction compiles/allocates but does not issue a query. Actual GPU checking
occurs only when called by `loan.complete` after a complete return. Exact hit IDs
and validity are required; distances allow at most8float32 ULPs, with no fixed
world-space tolerance. Actual saved region-stage WASM output passed this tighter
check and full-sensor parity. The module reports checked rays/batches/time.

## Opt-in worker and HTTPS integration checkpoint

`studio_cooperative_sensor.py` now connects the GL owner's retained strip to the
native Nodus host and service peer, then transports scene/work/results through
the owner-authenticated process mailbox. The still worker enables this path only
for `client_intersections: true`, fixed:3, one frame, at most 256 longest-edge
pixels. Each strip contains at most 256 rays. Default jobs are unchanged.
Readiness credit precedes issuing a lease, so a throttled or hidden client does
not consume a fresh lease while resting. Expired/bad returns reoffer the same
sample. A transport error fails explicitly; it never substitutes server work.

Completion waits for at least one complete image sweep, even if the requested
wall time has elapsed. Status remains exposing, with separate client_work_state
and client_work_profile fields. Scene registration is reused by exact packed
geometry hash. The current implementation still creates a native host for each
strip; measured overhead is substantial and is not represented as savings.

Real isolated HTTPS test uses the actual server handler, session cookie, browser
credential, distinct process secret, local CA verification, scene acknowledgement,
compiled Node WASM participant, native FIFO and the actual GL sensor. It does not
claim to be a Chrome browser test or an entire production photograph. For the
same 95,217-triangle, three-region scene and 256 camera rays, sensor output is
bitwise identical. Latest receipt gives server-only sensor/readback 0.01195 s,
cooperative total 13.775 s, client-reported computation 6.089 s, HTTP mailbox
4.591 s, native steps 5.180 s, and authoritative verification 0.00485 s. Components
run concurrently and must not be summed. Both paths used identical rays/seeds.

Tests: `--https-pump` real GL integration passed; 11 malformed optional telemetry
tests and 16 existing Live batch tests pass. No production server restart or
activation has been performed by this worker.

## Full-image validation and numeric boundary corrections

The initial 256-ray acceptance missed two float32 boundary cases in later image
rows. One same-owner distance differed by 12 ULPs, another by 34 ULPs. GPU FMA
contraction versus separate Wasm rounding explains these; a fixed ULP threshold
was inappropriate. The validator now computes a geometry-scaled determinant
forward-error window with gamma(32), explicit origin-subtraction error, and a
denominator interval check. Nonfinite or unresolved intervals fail closed.

A neighboring-triangle disagreement also occurred at a coplanar shared edge.
The original barycentric slack admits both triangles; one-ULP distance ordering
selects different owners. Such a result is accepted only after the original GPU
hit_tri law independently proves the claimed candidate intersects this ray and
is within the computed near-tie window. Miss/validity disagreements still fail.
The continuation always consumes canonical GPU records from the mandatory 1:1
check, preserving the original material/region owner and output. This does not
infer that neighboring faces share a material. Three failed completed returns
stop the job explicitly rather than retry forever.

A complete 128x128 sensor sweep now passes: 16,384 distinct camera rays in four
4,096-ray batches, 16,384 verified and committed, zero reoffers, two proved
near-ties, full output bitwise equal to its uninterrupted server baseline.
The actual owner-authenticated HTTPS/Nodus/compiled-Wasm route took 19.26 s;
client-reported query time totaled 2.585 s and verification 0.0453 s. Query
scheduling optimizations and larger batches were both present, so this alone
must not be described as a batch-size-only comparison. Browser Chrome acceptance
and the full forward-field PhotoJobs process are still separate tests.

## Row-dispatch and benchmark corrections

The larger sweep uncovered a separate real small-strip bug: row-mode dispatch
rounded to 16-row workgroups but did not mask padded invocations. A requested
128x2 strip could execute 16 rows, overlap pixels and exceed retained-ray buffer
bounds. The original sensor row branch now checks uDispatchPixelCount and width;
the existing accumulator already sets the count, and the legacy strip renderer
now sets it too. A real GPU/HTTPS 128x2 test verifies exactly256 covered pixels,
all other pixels clear, and bitwise matching output. Larger aligned32-row strips
were unaffected. Full small-strip acceptance is being rerun.

Freezing the dispatch frame alone does not make camera rays identical across
batch sizes: the original RNG also includes the local invocation index. Therefore
cross-batch throughput comparisons must replay the same captured ray array;
separate full sensor runs compare only against their own matched server baseline.
Earlier fixed-frame-only runs must not be used as identical-workload comparisons.

## 2026-11-10: authorized participant mode and honest accounting

The user's latest instruction removes duplicate server intersection verification from normal cooperative work. `CooperativeSensorPump` now defaults to `verification_mode="participant"`. It validates authenticated immutable identities and finite/ranged primitive records, retains ownership and expiry checks, then consumes the authorized client's result. It does not trace those intersections again. `gpu_reference` remains an explicit QA option, not a production requirement. Structural acceptance is not independent proof of geometric correctness.

`trusted-native-4096-receipt.json` records an isolated real HTTPS/native Nodus full 128x128 sweep with 16,384 committed client queries, zero verification rays/time, no reoffers, and complete image coverage. No baseline render was run. Its ~19.99 s includes conservative frozen-resource readback/hash checks, scene registration, repeated mailbox/native callbacks, and a Node surrogate that reserializes scene data locally. It is not an actual browser performance result. Geometry-only resident cache identity is now separate from frozen transport identity; exact canonical component hashes bind owners, triangle float32 bytes, BVH node bytes and traversal index order. This does not yet adopt arbitrary Room geometry as Exposure geometry.

`RetainedRayBatch` provides disjoint child ownership, generation/participant checks, retained assembly and exactly-once parent commit. Its tests pass. It is not a scheduler and is not yet connected to production parallel participant channels. Native Nodus multi-host scheduling has independent tests; the production Studio pump remains serial until the integration gate passes. No concurrency or cost-saving claim should be inferred from these component tests.
