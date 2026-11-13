# Finite-segment retention update, 2026-11-12

The existing bounded scene store now keeps a forward/reactive terminal or
disposition record when its actual finite segment overlaps the requested ROI,
even if both endpoints lie outside. It uses the existing copied entry origin
(words 117-119), entry direction (114-116), actual hit (21-23), and hit identity.
There is no new ray trace, optical evaluation, row codec, GPU allocation or
energy sum. Exact finite IEEE coordinates are compared with three rational slab
intervals so a short real crossing is not lost to rounded float64 parameters.
ROI bounds must be finite. Lower faces are included; upper faces are excluded;
a point-only touch is not segment support.

The endpoint stays where the original tracer put it. Its index cell remains
unavailable when outside the ROI; the downstream optional finite-segment index
can refer to this same observation from crossed cells. Such references are not
new independent observations or deposits. Category restrictions, bounded
surface/native-only partitions and eviction remain in force. Missing hit or
entry evidence is not turned into an infinite ray. Camera completed-sample
point retention is unchanged. This update cannot recover a path the producer
never ran, supply missing camera intervals, or complete shader application.

## Verification

Base: Spectral `917183f4c5e20ba2a2ffbce7add504061949c0cc`.

```
python -m pytest tests/test_studio_scene_responses.py tests/test_studio_response_segment_retention.py tests/test_studio_response_factory.py tests/test_studio_response_lineage.py tests/test_studio_reactive_application.py -q
python tests/studio_scene_response_transport_check.py <turing-root>
python tests/studio_scene_response_transport_check.py <turing-record-index-worktree> --require-record-index
```

The focused suite passed 84 tests. The initial isolated-worktree run passed 83
and failed one geometry fixture because its existing `_spectral_kernels` native
CPU extension was not present in the worktree. Supplying the existing canonical
repository on process-local `PYTHONPATH` resolved that missing dependency; no
package or binary was installed or rebuilt. Current worktree Python sources
remain first on the import path.

The existing Turing decoder at `7813a6a37f4916a1c9d095d3203571f1e45a4b43`
accepted a real store snapshot with both forward/reactive crossing records and
all 256 native words unchanged. The isolated record-index implementation at
`8a75596b7e4e076f267ebbd13abd59c2b5f6268e` also returned 64 cell references to
those same two observations. Both checks retained `applicationReady:false` and
used zero GPU bytes. These were CPU fixtures, not a linked producer or render.

The CPU cases include both path families, terminal/disposition semantics,
native-only interior primitives, unchanged zero/NaN payload attributes and
region identity, half-open faces, corner touches, extreme finite crossings,
explicit category scope, missing entry evidence, bounded partition eviction,
and 250 deterministic cases against an independent plane-event oracle.

Review caught a float64 interval-collapse case before finalization. The fixture
now requires genuine extreme crossings to survive, using exact arithmetic
rather than a tolerance or an incorrect out-of-scope classification.

## Still incomplete

The camera-pair delivery gap is separate: the existing observer emits one
128-word interval plus one 128-word response row, but the unused family 3 scene
section reserves 384 words without a defined third row or writer. This update
does not fabricate padding or reinterpret that incomplete contract. The
hardware execution hold, queued-reactive-work limit, and current Phong
application limits remain unchanged. No core tracer source, transport law,
compiler, Nodus permission, shader, live opt-in default, or GPU call changed.

## Prompt history

The owner directed: "you are then taking on the job of updating a new version of it".
Earlier requirements retained here: classify regions from actual native
observations, preserve non-opaque interiors and direction, do not rederive
transport in Phong, and distinguish absent evidence from evaluated zero.
