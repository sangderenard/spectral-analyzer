# Marble photo film response

The RGB photo worker accepts an optional per-job `film_response` object:

```json
{"version":1,"centers_nm":[650,550,450],"sigmas_oct":[0.06,0.06,0.06]}
```

Arrays are ordered R/G/B. Centers are independent wavelengths in nanometres
(380–780 inclusive). Widths are independent Gaussian standard deviations in
log2-frequency octaves (0.01–1.0 inclusive), **not nanometres or FWHM**.
The existing peak-normalized response is
`exp(-0.5 * (log2(frequency_hz / center_hz) / sigma_oct)^2)`.
Changing sigma changes integrated sensitivity as well as overlap; it is not
area-normalized, an exposure correction, or a display-gamma adjustment.

These defaults preserve the existing `em_rgb_fixed3` response exactly. They
are a custom sensor setting, not a calibrated film stock, ICC profile, or
standard observer. The transport still samples only 450, 550, and 650 nm.
Moving a narrow response away from those samples can substantially reduce
its exposure. A new exposure is required; existing accumulated data is not
reweighted when controls change.

## Validation and job records

- `camera_designer/film_response.py` is the dependency-free contract module.
  `normalize_film_response()` returns fresh defaults; an explicit argument
  must be the complete object with exactly the three keys above.
- Version must be integer 1. Triplets must be JSON arrays of three finite
  numbers. Null, strings, booleans, unknown keys, missing keys, unsupported
  versions, wrong lengths and out-of-range values fail. Crossing/overlapping
  centers is allowed.
- `fixed:1` keeps its previous behavior and rejects an explicit RGB response.
- The input config and its existing digest algorithm are not rewritten.
  `effective_config.json` records the normalized effective RGB response even
  for old configs that omitted it. Response values remain fixed for the job.
- Preparing, exposing, encoding, completed, and same-job error statuses retain
  `film_response`. RGB statuses also name `effective_config.json`.
- Finished HDR NPZs contain `film_response_version`, `film_centers_nm`, and
  `film_sigmas_oct`, alongside the existing exposure and wavelength fields.
  Legacy `wavelengths_nm` labels the discrete transport bands; it does not
  describe shifted sensitivity centers or claim spectrally pure channels.

## Actual transport paths

`build_film_stack(preset, response)` deep-copies the existing ordered R/G/B
stack and changes only its layers' `response_center_hz` and
`response_width_oct`. It sets `transport_wavelengths_nm=(650,550,450)` on
the copied stack, opting into the separately implemented fixed transport
basis. The conversion preserves the renderer's `2.998e14 Hz·µm` constant.
The same per-job stack goes to both paths:

1. The worker passes `film_stack=film` to `_gpu_ray_field_prebuilt`.
   That function sets `uFixed3Transport` from the opt-in and also uploads
   `film.layer_specs()` centers, widths and gains. Subsequent
   `_gpu_pump_prebuilt` dispatches retain that uniform state. In fixed3 mode,
   forward `splat_spectral` stores the raw discrete transport-band energy,
   rather than mixing it into custom film-filter channels. Material optics
   remain in the fixed wavelength basis, independent of film settings.
2. Every worker `SensorAccumulator` receives `accumulator.film = film`
   before its first tick, including each orbit frame. Each camera
   `_dispatch_once` sets `uFixed3Transport`, reads `self.film.layer_specs()`
   and uploads center, width and gain uniforms. Direct/NEE, emitted and
   gathered light stay in the same fixed transport basis. Only the final
   sensor accumulation applies the custom Gaussian response once.

Both saved marble and fixture calls to
`append_saved_studio_mesh_to_gpu_scene` set `spectral_emission_rgb=True` for
fixed3, so the separate color-path implementation can preserve authored
emission-band amplitudes rather than display-integrated colors. Monochrome
passes false and does not opt into the fixed3 path.

The response implementation itself does not change layer colors, display
tonemap or gamma. Its integration activates the separately implemented
material/color-path correction; default **sensitivity parameters** remain
exact, but the resulting pixels can change because the former coupling of
material transport and film filters is corrected. Combined GPU verification
belongs to that correction's receipt, not the CPU tests below.

## Verification

`python -m pytest tests/test_marble_film_response.py tests/test_marble_preview_worker_status.py -q`

47 passed in 3.63 seconds after the color-path opt-in integration (run inside
a 90-second subprocess timeout).
Combined CPU check with `tests/test_studio_spectral_sensor_basis.py`: 51
passed in 3.26 seconds.
Coverage includes strict malformed/boundary inputs, the actual renderer
preset's exact defaults and other-field preservation in a clean bounded
import, same-stack worker routing, per-job status/config/HDR metadata,
unchanged monochrome behavior, all eight orbit sensors and encoding status,
retained response on an error, fixed3 opt-in only on the cloned film, and
spectral-emission mode on both mesh imports. Worker routing uses CPU/GL test doubles;
this is not a live GPU render or full-suite claim. No server, active exposure,
deployment or compiler build was started or changed.

The pre-existing dirty worker was backed up before editing at
`repo-cache/film-response-before-20261109-065138/marble_preview_worker.py`,
with its existing Git diff alongside it. Unrelated dirty files were not edited.
