# Opt-in evaluated scene response copies

Request: `light_field_information="evaluated-responses-v1"`. The current worker requires an explicit fixed:3 field, manual fixed bounce budget, and untiled native sensor. Ordinary requests do not construct these owners or allocate their auxiliary storage. This is additional data from the existing tracer, not a separate renderer.

## Native dispatch

Prebuilt initial warmup and forward pump keep their original launch sizes, offsets, seeds and bounce budget. A synchronous consumer receives terminal outgoing packet/disposition rows after each actual dispatch. The original 128-ray warmup is labeled; partial source coverage is not a complete source-population estimator. A native queue counter reports attempted, stored, dropped and unprocessed reactive entries. The prebuilt schedule does not dispatch reactive work. A shader instrumentation seam for reactive bodies does not imply those bodies ran.

Each terminal row is 128 little-endian uint32 words. Float data retains its native float32 bits. The complete packet is frequency, phase, energy, auxiliary; no packet field is relabeled calibrated radiance.

- 0: semantic kind (5 terminal packet, 6 disposition); 1: native path family; 2/3: native ray/root; 4/5/6: bounce, hit, native material.
- 7/8: medium before/after last evaluated response; 9: copied factor attribute.
- 10–13/14–17: before/after packet; 18–20/21–23: response origin/hit position; 24–26/27–29: evaluated-response direction/outgoing direction.
- 30/31: validity/termination. Termination includes evaluated zero, nonfinite, old absolute cutoff, rejected boundary, escape, rejected sample, absorber and finite bounce-budget exhaustion. Escape has no hit position.
- 32–63/64–95: exact native active-region words before/after. These are not converted into guessed material identities.
- 96: intermediate-stage presence; 97–100: existing factors; 104–107/108–111: existing Beer before/after packets. Factors are attributes, not independent energy deposits.
- 112/113: exact selected primitive kind/owner; 114–116: original segment-entry direction; 117–119: entry origin; 120–123: entry packet; 124: entry medium.

Native recipient lineage is copied at the original mesh packing and sphere compaction events. No material-ID or positional approximation substitutes for missing lineage.

## One auxiliary allocation

Only the new forward auxiliary data share SSBO binding 23. At the current 96 cubed scalar grid and three bands:

- Visit/loss prefix: offset 0, length 7,077,888 bytes.
- Native scalar columns: offset 7,077,888, length 42,467,328 bytes.
- Transient terminal rows: offset 49,545,216, length `8 + dispatch_batch * bounce_limit * 512` bytes.

Offsets are recorded in dispatch metadata. Each view clears only its own range. The native field owns and releases the allocation once. A terminal owner borrows it; the sensor query reads the unchanged visit prefix. Original native buffers remain separate. Whole-buffer initialization happens only upon allocation.

The full corrected, region/sphere, internal-recording variant declares 16 compute storage blocks. Runtime admission checks compute-block, binding and per-buffer limits; offline compilation is not hardware execution proof.

## Original scalar integration

Four uint32 columns per original cell/band are copied at the existing native integration:

1. Finite nonnegative evaluated attempts, including evaluated zeros, before the positive-deposit filter.
2. Accepted original atomic additions, not independent rays.
3. Exact original uint `val` sum, without a new scattering formula or normalization.
4. Sticky loss flags: positive rounding 1, clipping 2, invalid value 4, sum overflow 8, attempt-count overflow 16, accepted-count overflow 32.

Scalar source availability additionally reports both count overflows; any loss leaves source availability unknown. These cells preserve the original global scalar grid and bounds. They are not the separately requested directional subject ROI and are not labeled 0.5 mm. Original scalar deposition may mix air-volume, emission and other native inputs; it does not provide a fabricated per-medium or angular partition or complete in-solid coverage.

Readback uses exact-index z-slabs. The bounded store keeps sparse original cell/band rows, retaining evaluated zeros and losses. Only entirely untouched zero rows are omitted. The scalar snapshot policy is latest native cumulative snapshot, not progressive averaging across source epochs. Superseded, discarded and evicted observations remain explicit metadata. Forward/camera scene records have their own bounded retention.

## Identity and application

Retained-scene identity hashes authoritative saved target arrays and fixed-scene configuration. Camera pose and sampling-population changes do not erase target history. Every observation separately retains exact assembled `transport_scene_sha256`, including real camera housing. Different housing contexts cannot be combined or applied as compatible.

`source_sha256` has explicit source kind: forward rows identify baked source records; camera rows identify the installed camera program. Camera records separately retain `source_population_sha256`. Original camera/NEE contributions remain view/prefix-conditioned native responses, never summed with forward packet energy.

The producer application stage uses the existing native accumulation kernel and the same normalized film profile as the original sensor. The receiver applies copied results only under the exact current query/visibility contract. Missing, evicted, loss-limited and unsupported data remain unavailable, not black. Count or visit presence is not convergence or a proof of physical darkness.

## Publication and limits

The existing authenticated field route serves `light-scene-response-<id>.bin`, with immutable `.scene.json` ownership receipts. A snapshot is at most 64 MiB; at most three retained artifacts are protected by shared locks and transfer leases. Backpressure preserves active readers. There is no unbounded raw-history spool.

The overall working-memory ceiling is 512 MiB, including actual native allocations, retained arrays, shared auxiliary storage, source/history/sensor allocations, readbacks, conversion temporaries and bounded retained/application data. Admission never reduces the requested native dispatch schedule to fit. Current actual Live settings fit the known portions with a 512-ray worker chunk; an 8192-ray chunk exceeds this ceiling before scene buffers and must be rejected unless memory use is reduced without changing transport.

`application_ready` remains false until ownership, budget and current-query review closes. Current validation is CPU/source/offline compilation and separately authorized link-only checks, not a rendered acceptance result.

### Native queue availability and execution hold

Prebuilt execution is forward-only. Existing native reactive/diffraction enqueue
counts are copied, but pending work is not dispatched by this owner. Camera
context records carry the actual queue observation; pending entries, dropped
entries, counter regression, or missing observations cannot establish complete
response availability. Incompleteness is sticky for that native queue allocation:
clearing scalar images or refreshing source populations does not drain the queue.
Computed response values remain retained separately from this unresolved work.

Evaluated-response worker execution is currently held before context creation:
hardware compile resource validation is pending. Source-only and CPU ownership
checks do not certify hardware execution. The specialized prebuilt entry retains
the exact forward body and rejects non-forward modes; generic shaders and the
separate native camera program are unchanged.
