# Queued ancestry transport v3

Explicit request `light_field_reactive_queue: native-ancestry-v1` requires
`light_field_information: evaluated-responses-v1`. The JS packer option is
`lightFieldReactiveQueue`; absent leaves the ordinary request unchanged.
No UI activation or executable-readiness claim is introduced.

The first accepted evaluated-forward-responses-v2 batch selects persistent
studio-scene-responses-v3. Families1/2 keep128 words with the exact exported
FORWARD_ANCESTRY_CONTRACT. Family3 uses the established256-word camera pair ABI,
even when empty; family4 completed samples remains unchanged. Retained forward
v1 or384-word camera rows cannot be relabeled. No implicit eviction/padding.

Path2 keeps queue ordinal2 and actual selected source root3. New fields101–103
are parent dispatch/ray/bounce;125 is copied inverse-replay float bits;126 is
actual enqueue cause;127 is spawn triangle. Forward path1 spare fields stay zero.
Contract is exact and recursively type-checked in Python/strict-valued in JS.
Parent context preserves dispatch.native_queue and the immutable scene/source/
population provenance. Rows are checked against generation, parent ray range,
actual source-root modulo, bounce budget, finite weight0..1, cause1/2 and signed
nonnegative spawn triangle. Weight0 is retained, not interpreted as coverage.

The existing store, immutable file publication, lease route and decoder are
reused. V3 artifact access requires queue opt-in. Nonempty camera-pair sections
also require camera-record opt-in. No native arithmetic, energy normalization,
position, application availability or Phong behavior changes here.

CPU reproduction from Turing receiver worktree:

    $env:STUDIO_SPECTRAL_TEST_ROOT='C:\Users\alber\spectral-response-retention-20261112'
    & C:\dev\Powershell\turing\.venv\Scripts\python.exe tests/studio_forward_ancestry_roundtrip.py

This serializes actual SceneResponseStore rows and sends them to the JS receiver,
verifies exact native bytes, negative parent/codec/contract cases and packer opt-in.
Backend route tests: tests/test_meowdoku_forward_ancestry.py, using the same
SPEAKTOME_REPO/PYTHONPATH overrides as documented for camera-pair tests.
Spectral tests: tests/test_studio_forward_ancestry_transport.py.

Depends on the matching producer's FORWARD_ANCESTRY_CONTRACT. CPU synthetic
record fixtures prove serialization/admission, not actual shader execution,
complete sampling, deposited energy or usable shading. Hardware execution hold
and the no-render constraint remain unchanged.
