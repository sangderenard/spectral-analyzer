# Versioned camera interval/response delivery

## Contract and scope

The existing observer produces two aligned 128-word buffers. V2 transports
exactly one interval followed by its corresponding response row: 256 little-
endian uint32 words / 1024 bytes. Float32 payload bits, including evaluated zero,
signed zero and nonfinite response attributes, remain unchanged. No third row
is invented to fill the old unused 384-word reservation.

The envelope is `studio-scene-responses-v2`. Family 3 uses
`native-camera-response-pairs-v2`; `camera_rhs_rows` has stride 1024. Other
families retain their existing layouts. The exact `camera_pair_contract` is:

```json
{
  "version": "native-camera-response-pairs-v2",
  "row_words": 256,
  "interval_schema": "native-camera-intervals-v1",
  "interval_offset_words": 0,
  "interval_words": 128,
  "response_schema": "native-camera-response-copies-v1",
  "response_offset_words": 128,
  "response_words": 128,
  "alignment": "same-dispatch-row-ordinal",
  "attachment": "native-only-unlocated",
  "application_ready": false
}
```

The outer index says an observation exists, not that a response is available or
physically complete. It carries the actual entry direction and current primitive
kind. Recipient is unknown, position is unlocated, and spatial cells are sentinel
values. The raw interval retains entry origin/distance, medium, active-region
bits, current hit/primitive/owner and first-hit fields. Neither the current hit
nor the first hit is guessed to be a valid raster recipient. No interval is
reconstructed from a completed camera total. Additions and multipliers are not
summed together, divided by a guessed prefix, or converted to Phong color.

## Existing path, explicit opt-in

An existing evaluated-response request may additionally specify:

```json
{
  "light_field_information": "evaluated-responses-v1",
  "light_field_camera_records": "interval-responses-v2"
}
```

All original evaluated-response prerequisites still apply. The new mode selects
the existing interval/response capture instead of the completed-sample capture.
It is not a simultaneous second sensor run. Absent this additional selection,
the existing completed-sample mode remains unchanged and publishes v1. No UI
default, automatic request, shader, optics, ray count, seed or transport changes.

Flow: existing `StudioResponseEvidence.finish` supplies the synchronous two-
buffer callback; `SceneResponseConsumer.consume_camera_pairs` retains each pair;
the existing scene publisher writes the immutable artifact and ownership receipt;
Turing's existing authenticated/leased artifact route serves the same bytes; the
existing browser decoder validates v2 and returns exact typed words. The response
owner now supplies its real stream ID and actual dispatch program hash to this
callback, as it already did for completed samples. Short paired readback fails.

V2 remains selected for that store's lifetime after accepting actual pairs, even
after eviction or scene invalidation. Legacy v1 remains separately supported;
mixed layouts, codecs, ownership envelopes or pair contracts fail validation.
The API requires the additional mode before serving a v2 artifact.

## Memory, coverage and execution

Pairs use the existing bounded native-only partition, preventing raw intervals
from evicting surface samples. The existing allocation is not enlarged; staging
adds only one 256-word scratch row. Metadata/retention and eviction stay bounded.
Reservations include both GPU buffers and both readbacks at the original bounce
bound. A dispatch that cannot fit is rejected rather than silently reducing its
rays, row schedule or bounce budget.

Current Phong application is deliberately not invoked for this raw-only mode.
`application_ready` and physical completeness remain false. Missing, discarded,
evicted, invalid or untraced data are not black. Native sample identity and
denominator metadata survive; response rows are not independent camera samples.
The existing prebuilt reactive queue is still not dispatched. This change does
not supply all-path/interior coverage or camera-interval spatial indexing.

The evaluated-response hardware execution hold remains before GL initialization.
No GPU compile/link/dispatch, exposure or render was performed for this update.
The code is retained in isolated branches, not merged, published or deployed.

## Verification

CPU checks include exact two-buffer readback through the real callback and store,
unknown/zero preservation, malformed row/primitive/mask rejection before cache
mutation, v1 default compatibility, bounded eviction, publication/version closure,
short-read and overflow refusal, and the unchanged pre-GL execution hold.

```
python -m pytest tests/test_studio_camera_pair_delivery.py tests/test_studio_scene_responses.py tests/test_studio_response_segment_retention.py tests/test_studio_response_factory.py tests/test_studio_response_lineage.py tests/test_studio_reactive_application.py -q
python -m pytest tests/test_studio_camera_intervals.py -q -k response_evidence_lifecycle_with_cpu_fake_gl
python tests/studio_scene_response_transport_check.py <turing-camera-pair-receiver-worktree> --camera-pairs
python tests/studio_scene_response_transport_check.py <turing-camera-pair-receiver-worktree>
python tests/studio_scene_response_transport_check.py <turing-vector-index-worktree> --require-record-index
```

The focused suite passed 116 tests, plus the existing fake-GL lifecycle test.
The existing native CPU module was supplied on process-local `PYTHONPATH`, as in
the preceding retention receipt. Two real store pairs round-tripped as all 512
unchanged words. The old forward/reactive fixture still round-tripped as 256
unchanged words, and the separate earlier record index still produced 64 cell
references. No optical correctness or live throughput claim follows from these
CPU checks.
