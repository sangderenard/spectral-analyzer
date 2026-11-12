# Ink atlas through Studio REST

`live_spectral_text_demo.py` retains its glyph → token → complete-string planner,
catalog and object library. The `--studio-atlas-url` option selects an account
owned Studio photo job for each scheduled asset; omitting it keeps the original
native BDPT renderer.

The Studio adapter takes the camera, exposure, backplate and emitting ring
from the saved `configs/studio_ink_atlas_photo.json` contract. It uploads the
requested glyph or token mesh itself as a private OBJ subject in the signed-in
account, then submits the existing MRP1 photo contract. The completed
Studio HDR's `linear_srgb_exposure` is decomposed by the existing sprite
extractor and entered into the same atlas catalog and object library. The
remote photo stays in the account's Studio photo assets.

Launch the existing UI with an authenticated Studio session cookie supplied in
the process environment:

```powershell
$env:SPECTRAL_STUDIO_COOKIE = '<account session cookie>'
python live_spectral_text_demo.py --text 'AXE' --studio-atlas-url 'https://stichka.duckdns.org:8766' --studio-atlas-contract configs/studio_ink_atlas_photo.json
```

The saved contract has a fixed Studio scene and default 0.1 second shutter,
one substep, ISO 100, and RGB transport. A different saved contract may set
the camera, full fixture mesh and material rows, film, transport and exposure.
The request still owns the OBJ subject and square capture resolution. CLI
shutter/substep/ISO flags override only those three contract settings.

The Studio option submits one static timed shutter frame. Its model shutter
duration and substeps are separate from wall clock GPU runtime and from the
BDPT order's 1/60 second exposure setting. The current Studio acceptance path
requires a square capture of at least 128 pixels. The Studio result records a
completed shutter, not BDPT convergence; exposure calibration has not yet
selected a sensor sampling budget. No live server or GPU render is implied by
the contract tests.

`tests/test_studio_atlas_rest.py` covers the MRP1 subject/scene/shutter payload,
linear HDR extraction, and the catalog round trip with a fake REST service.
The constructed payload was also passed through Turing's real
`meowdoku.marble_photo.decode_upload`, then through Spectral's
`prepare_photo_geometry`. This checks admission and scene construction,
not rendered image quality.

The atlas planner still schedules text glyphs and tokens. Multi-view object
studies, game map packaging, HTML page rendering and an account-facing bake
endpoint are not implemented by this adapter. The existing general
`plan_bake_requests` API models page/part/sequence work; it is separate from
the current production `plan_ink_atlas_bake` path.

Native T5 has not been connected to Studio REST by this adapter. T5 currently
runs through `exposure_render_demo.CppExposureBackend` and the native
flash/sensor/T5 latch. Studio photos run through
`camera_designer.marble_preview_worker` and `SensorAccumulator`. A genuine
algorithm choice requires both workers to receive the same Studio scene and
camera contract, separate sampling controls, and comparable HDR/status
artifacts. A flag that still executes Studio's sensor shader would not be a T5
integration.
