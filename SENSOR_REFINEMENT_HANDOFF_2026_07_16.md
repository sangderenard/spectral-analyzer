# Recursive Spectral Sensor and Learned Priority Handoff - 2026-07-16

## Objective

The scene renderer must place authored objects and a fixed camera in world
space, then expose only the requested camera sensor region through the existing
spectral thick-lens BDPT/VCM transport. Sensor refinement is continuous within
UV regions; hierarchy nodes are not point samples or pixel-center substitutes.

The live text demonstration is the integration target. While it remains open,
it must keep accumulating independent Monte Carlo evidence, recursively refine
the sensor, preserve broad coverage, prioritize useful regions, and show visible
progress without replacing the physical renderer with a raster or denoising
shortcut.

## Non-Negotiable Invariants

- The camera, lens assembly, text plane, text mesh, lights, and materials remain
  scene geometry. Adaptive scheduling does not move or reinterpret them. An
  explicit bounded film-stage command may move/tilt only the sensor plane.
- Each selected sensor node launches continuous stochastic UV samples through
  the genuine GPU thick-lens spectral BDPT path.
- Every completed nonterminal node subdivides into a 3 by 3 child stencil.
  Subdivision is not conditional on priority.
- Terminal nodes are requeued for new independent exposure epochs.
- Direct evidence remains stored at the level where it was measured.
  Descendant rollups are stored separately and are area weighted, so coarse UV
  estimates can improve without discarding their own observations.
- Learned priority changes work order only. It does not alter radiance,
  throughput, material response, path weights, or the estimator.
- A configurable fraction of work remains broad coverage. The live default is
  75 percent targeted and 25 percent coverage.
- GPU execution is required for ray-traced training, priority inference, sensor
  sampling, hierarchy maintenance, and spectral transport.

## Implemented Architecture

### Sparse sensor hierarchy

`camera_software/sensor_mipmap.py` is the executable reference model for UV
bounds, weighted spectral moments, direct observations, 3 by 3 subdivision,
and coarse rollup invariants. `camera_software/refinement_scheduler.py` defines
top-k ordering separately from unconditional subdivision.

The native ABI and GPU implementation are in:

- `csrc/include/sensor_mipmap.h`
- `csrc/include/ray_pipeline.h`
- `csrc/kernels/ray_tracer.cpp`
- `csrc/bindings/pybind_kernels.cpp`
- `csrc/shaders/sensor_mip_frontier.comp.glsl`
- `csrc/shaders/sensor_mip_score.comp.glsl`
- `csrc/shaders/sensor_mip_select.comp.glsl`
- `csrc/shaders/sensor_mip_sample.comp.glsl`
- `csrc/shaders/sensor_mip_raygen.comp.glsl`
- `csrc/shaders/sensor_mip_accumulate.comp.glsl`
- `csrc/shaders/sensor_mip_subdivide.comp.glsl`
- `csrc/shaders/sensor_mip_rollup.comp.glsl`

The live native configuration currently reserves 524,288 nodes, uses maximum
depth 7, traces 32 continuous samples per selected node epoch, and advances up
to 1,024 selected nodes per internal refinement step. These limits were chosen
to remain viable on a 12 GiB RTX 3060 while retaining the existing BDPT record
buffers.

The native selection shader mixes top-k targeted work and spatially broad work.
The coverage side uses locality-aware ordering so the neutral case remains
memory coherent. `camera_software/exposure_priority.py` contains the matching
policy contracts and Morton ordering reference.

### Learned work-value network

`camera_software/sensor_priority_network.py` contains:

- A presentation-only CUDA orthographic renderer of the exact scene triangles
  using flat authored colors, without lighting or camera optics.
- Four runtime input channels: compressed luminance, red chromatic share,
  green chromatic share, and logarithmic exposure confidence.
- A fixed 4-to-8 3x3 convolution, ReLU, and 8-to-1 1x1 softplus head.
- A stable 305-float GLSL parameter ABI.
- Model, flat-reference, heat-overlay, and JSON metadata export.

`csrc/shaders/sensor_priority_infer.comp.glsl` runs the same network directly
over the GPU sensor RGB sums and exposure weights. Its output SSBO feeds the
native sensor-node scorer. Python loads the exported parameter vector before
the ray pipeline is created. The inferred map is read back only for inspection;
the scheduling decision itself remains GPU resident.

`camera_software/raytraced_priority_training.py` trains only from progressive
spectral sensor sums and exposure weights. Its target is the positive reduction
in camera-image error caused by the next real exposure layer, measured against
the latest available exposure. The orthographic preview has no import or data
path into training. A future discriminator objective can replace this measured
target without changing the 4-channel runtime transport or scheduler interface.

### Progressive communication

`camera_software/progressive_exposure.py` defines transport-neutral immutable
events and atomic artifacts. Each available layer can announce:

- Linear camera accumulation.
- Per-bin sample counts when applicable.
- Raw NN priority map.
- Flat orthographic reference.
- Region, pass, zoom, subdivision, and sensor-node identity.

The writer retains only the configured number of recent live layers. These
files are a presentation/readback boundary and never feed the physical
accumulator or scheduler.

### Physical film-stage control

`camera_software/scan_control.py` defines the network-agnostic `NextSiteScan`
contract. Its optional `FilmPlaneAdjustment` commands axial film-to-lens depth
and tilt about the film's right/up axes; it does not expose arbitrary mesh
vertices. Values are absolute offsets from machined zero, so trials cannot
accumulate motor-history drift. Camera-owned travel and tilt limits are
validated before scene build. `FocusCalibrationController` provides a
network-facing trial protocol which requests full-sensor scans and computes its
objective strictly from the ray-traced sensor image; it has no orthographic,
text, depth, or authored-scene input.

`exposure_render_demo.py` applies that command as a rigid transform to the
sensor triangle group only. The aperture, thick-lens geometry, subject, and
lights remain fixed. The same resulting pose is supplied to the camera
descriptor and native pipeline. `sensor_mip_raygen.comp.glsl` converts local
continuous film UV samples to world origins with the tilted film basis, while
the aperture remains in the fixed lens basis. Forward sensor hits are projected
back into the same local film coordinates. Thus depth/tilt changes physical
focus geometry without changing recursive UV node identity or radiance weights.

The command is accepted through `--next-scan-control` (or
`SPECTRAL_NEXT_SCAN_CONTROL`) using:

```json
{"sequence":1,"film_plane_adjustment":{"lens_distance_delta_m":0.0002,"tilt_about_right_deg":0.75,"tilt_about_up_deg":-0.5}}
```

The priority CNN still emits only a work-value map. A future trained focus head
may produce the already-bounded film commands without changing the camera,
transport, or recursive sensor interfaces.

### Authoritative physical camera transport

`camera_software/camera_build.py` retains the complete rebuilt camera artifact:
the source camera configuration, exact compound-lens assembly, proxy-surface
identities, wavelength table, sensor basis, machined-zero pose, and explicit
optical provenance. Subject replacement remaps proxy triangle IDs without
reconstructing or downgrading the optics.

The text exposure registers exact parametric conic/plane intersections on the
GPU. Proxy triangles remain only BVH entry surfaces. The exact Snell/Fresnel
step now selects refractive indices from the active wavelength's Sellmeier
table; it no longer applies one scalar index to every spectral band. Recursive
sensor rays target the assembly's real or virtual exit pupil and no toy
thin-lens handler is attached in this mode. Diffraction is explicitly reported
as disabled; the separate wave arena is not mislabeled as camera diffraction.

### Live text demonstration

`live_spectral_text_demo.py` now performs the complete path automatically for
every submitted text revision:

1. Resolve the same scene job used by the spectral renderer.
2. Render the CUDA flat orthographic reference for display only.
3. Load a reusable ray-trained model when `--priority-model` is supplied;
   otherwise use measured heuristic/exploration scheduling without training.
4. Pass reusable weights through `SPECTRAL_SENSOR_PRIORITY_MODEL` when present.
5. Start the native GPU thick-lens BDPT exposure with no authored epoch cap.
6. Stream linear camera accumulation and the current GPU NN priority map after
   each stable exposure layer.
7. Display three panels: `SPECTRAL CAMERA`, `FLAT SCENE`, and
   `NN WORK VALUE`.
8. Continue until the window closes or a newer text revision supersedes it.

No special command-line switch is required. The normal entry point is:

```powershell
python live_spectral_text_demo.py --display-width 100 --display-height 100
```

The default targeted fraction is exposed as `--targeted-fraction`. The live
worker sets continuous exposure, unlimited epochs, and three retained
presentation layers. It never trains from the orthographic image.

### Text mesh correction

The live scene now uses formatter-relative extrusion through
`extrusion_depth_ratio=0.08`, rather than a fixed depth that becomes unsuitable
when paragraph fitting changes glyph size. It uses four outline subdivisions,
a 64-cell cap grid, contour cleanup, and degenerate-cap rejection. The plane and
camera placement were not changed by this correction.

## Artifacts

Normal saved native output now includes, when available:

- `0000_cpp_linear.npy`: normalized linear camera image for presentation.
- `0000_cpp_sum_linear.npy`: unnormalized accumulated RGB sums.
- `0000_cpp_exposure_weight.npy`: regional exposure weight.
- `0000_cpp_priority.npy`: latest raw learned GPU work-value map.
- `0000_cpp_priority.png`: normalized priority visualization.

Per-revision attention artifacts include:

- `attention/orthographic_flat.png`

Bounded randomized ray-traced training writes a reusable model and metadata
under the trainer output's `model/` directory.

Generated evidence under `exposures_test_temp/` is intentionally untracked and
must not be committed.

## Validation Completed

- Native Release extension rebuilt successfully with the new shaders and ABI.
- The combined focused suite covering exact camera construction, spectral
  indices, scan control, priority policy/training, progressive transport,
  sensor hierarchy, live text, and scene orders passed: 79 tests (2 obsolete
  legacy expectations deselected).
- `python tests/t5_cpu_gpu_parity.py` passed both plain and glass cases. The
  latest stochastic run reported relative L2 error `1.67e-7` and `9.03e-3`
  respectively, with zero record overflows.
- A 32 by 32 recursive GPU text exposure completed with a +0.2 mm axial film
  move and nonzero two-axis tilt. It produced 998/1024 lit sensor bins and a
  finite spectral output through native GPU T5/VCM.
- A real bounded 100 by 100 GPU exposure loaded the 305 model parameters,
  ran recursive spectral BDPT, and announced both a finite linear accumulation
  and a finite 100 by 100 priority array in the same layer event.
- The bounded validation layer took about 42.3 seconds of exposure work and
  73.2 seconds wall time on the available RTX 3060.
- `git diff --check` and Python compilation passed before handoff.
- A fresh native Release extension build completed after the parametric-camera
  ABI and naming changes.

The interactive three-panel window was not left running for a long manual
acceptance session. Its component paths compile and its worker/progress tests
pass; the underlying live GPU transport was exercised independently as above.

## Known Follow-Up Work

1. Replace or augment measured next-layer improvement with the intended discriminator
   objective: predict expected improvement in exact-string discrimination
   against character-scrambled/noisy negatives.
2. Train the reusable model across many randomized bounded scenes and compare
   it against heuristic-only scheduling on held-out text.
4. Measure priority quality over multiple exposure layers. The first sparse
   layer can correctly appear nearly uniform because sensor evidence is mostly
   absent; spatial structure develops as observations accumulate.
5. Profile the T5 connection pass. It remains the dominant cost, not NN
   inference or priority-map transport.
6. Add a numerical PyTorch-versus-GLSL inference parity fixture if the model ABI
   or architecture changes. The present ABI is runtime-smoked but not compared
   element by element in an automated test.
7. Keep the broad-coverage fraction nonzero unless an unbiased alternative is
   proven. Learned targeting must not create permanent sampling holes.

## Useful Environment Controls

```text
SPECTRAL_SENSOR_PRIORITY_MODEL        exported .npz model path
SPECTRAL_SENSOR_TARGETED_FRACTION     default 0.75
SPECTRAL_SENSOR_STEPS_PER_LAYER       default 8 at <=128 resolution, else 4
SPECTRAL_SENSOR_MAX_EPOCHS            0 means unlimited
SPECTRAL_SENSOR_CONTINUOUS            true for live exposure
SPECTRAL_PROGRESS_RETAIN_LAYERS       live default 3
```

## Recommended Takeover Checks

```powershell
python -m pytest tests/test_progressive_exposure.py tests/test_live_spectral_text_demo.py tests/test_sensor_priority_network.py -q
python tests/t5_cpu_gpu_parity.py
python live_spectral_text_demo.py --display-width 100 --display-height 100
```

Do not evaluate the final physical image from the flat reference or priority
overlay. The flat reference is presentation-only and never training data. The
`SPECTRAL CAMERA` panel remains the genuine spectral thick-lens result.
