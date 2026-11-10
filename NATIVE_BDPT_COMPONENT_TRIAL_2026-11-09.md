# Native BDPT without camera simulation: current development state

Date: 2026-11-09. Scope: original Spectral native renderer, tested in isolated processes. Studio, its saved scenes, and the deployed native binary were not replaced.

## Finding

Yes. The original transport can run without a compound lens, CameraSoftware rig, physical sensor group, sensor electronics, noise, or T4 wave simulation. A virtual film remains necessary to assign image samples to pixels. The working entry is RayTracer + configure_sensor_image/pose + submit_emissive_triangles + submit_sensor_sweep. For the resident GPU path, use set_gpu_skip_record_readback(True), both native completion markers, and join_t5. This actually completed; it is not inferred from a mode enum.

The native pipeline is a useful component source and reference candidate, not a validated drop-in replacement for Studio. Keep the composed Studio shader and its geometry/material ownership.

## Reproducible bounded trial

The driver in exposures/dot_bdpt_transport_20261109/trial.py reuses quad_x from the repository's tests/t5_cpu_gpu_parity.py. Six triangles: a diffuse receiver, an off-camera area emitter, and a nonparticipating back plane. Four fixed wavelengths: 680, 600, 520, 440 nm. Four film samples per pixel cover all four lanes. The projection uses a virtual film looking away from its projection centre, with a negligible positive pupil required by the existing API. No sensor-camera group is registered. Eight-triangle glass variation was a continuation smoke only, not a physically calibrated glass test.

The before-T5 image is black, so the shown receiver is genuinely supplied by the original T5 connector. All recorded trials finish with finite output and zero vertex, spectral, PDF, optical, or connection overflow.

Measured after the emitter correction:

| Mode | Film | To saved readback | T5 phase | Lit pixels | Raw RGB sum |
|---|---:|---:|---:|---:|---:|
| CPU transport + CPU T5 | 16x16 | 1.29 s | 1.013 s | 256/256 | 52.7104 |
| CPU transport + GPU T5, host records | 16x16 | 6.37 s | 0.0366 s | 256/256 | 52.7104 |
| GPU T1-T3 + host-packed GPU T5 | 16x16 | 1.20 s | 0.1610 s | 256/256 | 66.5295 |
| GPU-resident T1-T3, native sort/scatter/T5 | 16x16 | 0.515 s | 0.0202 s | 256/256 | 18.2853 |
| GPU-resident T1-T3, native sort/scatter/T5 | 64x64 | 1.344 s | 0.0688 s | 4096/4096 | 292.3737 |

Timings are individual runs, not stable benchmark distributions. Cold GL compilation/allocation varies substantially. The CPU all-pairs progress thread sleeps for one second and then joins, imposing a large artificial floor on tiny CPU runs. Resident runs use the existing 16-million-float record-budget option (about 60 MiB), not the default multi-GiB allocation. The native T5 stage counter does not record resident time reliably; the table uses explicit wall-clock boundaries and detailed native logs. The 64x64 resident run evaluated about 9.64 million candidate pairs. These are small-scene results, not performance claims for the full Studio marble scene.

## Repaired native wavelength omission

The native emitter advanced seq once when selecting sampled_band and again when creating the ordering key. For even band counts this emitted only every other wavelength. Actual root spectral records proved 4-band launch counts [128,0,128,0]; the sensor side did cover all four lanes. In a 32-band regression, every odd lane was likewise absent.

The source repair removes the second increment in csrc/kernels/ray_tracer.cpp around line 18875. Geometry samples, Morton ordering, authored throughput, and the existing material database are retained. Five native coverage regressions exercise 1,3,4,8,32 bands and check area-weighted authored throughput. The old binary fails 4/8/32; the isolated rebuilt binary passes all five. Combined with the native work-plan suite: 19 passed in 7.25 s. Independent source review found no introduced blocker.

At the same settings, CPU raw RGB sums change from [15.2602,24.2688,0] to [22.1161,22.6764,7.9179]. This is restored missing transport, not a cosmetic tint. The repaired native target is built at build/dot-native-bdpt-isolated/module. The repository-root production binary is unchanged (SHA256 72390416ce34bca3e870a577e800d5a995debb0c90433043ea725683f6f317e6). The isolated binary is 5fe18913f4892c94b2967ae3778f75bf2775766208f2bbabe45533625890796b.

## What the checks establish, and what they do not

- With CPU paths held in the same native lane, CPU T5 and GPU T5 agree at relative L2 2.20e-7.
- Doubling emitted ray count from 256 to 512 changes post-fix total energy by +0.335%, not 2x. Doubling authored emitter strength gives exactly 2x. The native emitter has sample-budget normalization absent in the currently tested Studio field contribution.
- Doubling film samples gives about 2x raw image sum. Legacy uniform film readback is a sum, not a normalized radiometric exposure. Divide by the intended sample/exposure measure before an A/B; do not copy values into Studio by visual brightness matching.
- Native all-GPU host-readback is NOT equivalent to resident transport. Halving receiver reflectance from 0.8 to 0.4 gives exactly 0.5x on CPU, 0.9938x in host-packed all-GPU, and 0.4691x resident in this low-sample pilot. This is a real integration warning, not acceptable parity.
- Source explains a relevant mismatch: CPU T3 records post-scatter beta (ray_tracer.cpp ~17308); GPU T3 explicitly records pre-scatter beta (ray_material.comp.glsl ~614). CPU/host packing marks the first camera surface as a SENSOR endpoint, whose material response is bypassed (t5_connection.h ~415; ray_tracer.cpp ~19822). Resident PDF scatter overwrites that flag with the actual event flags (bdpt_scatter_t5.comp.glsl ~137), so it evaluates a material response. These representations must be reconciled before a shared quality claim.
- Sensor-sweep wavelengths are ap_idx % n_bands. Eight pupil samples in 32-band transport only cover lanes 0-7, even over repeated seeds. Use a complete multiple of the band count for current fixed-band reference runs. This was not changed in this patch.
- Continuous LUT sample-ID pairing was not validated. Emitter and sensor begin at different cached sample IDs; T5 requires compatible nonzero IDs. The fixed-band regression does not cover that contract.
- T5 uses real path records, per-band overlap, endpoint BSDF evaluation, edge-PDF measure conversion, delta-event handling and MIS. Yet its straight connection through glass applies Fresnel/TIR attenuation along the straight segment; it does not solve a refracted two-point connection. A marble's refractive caustics need dedicated reference tests.
- Native band_to_display_rgb is a piecewise spectral-locus approximation, not Studio's newer CIE/D65 profile. The native getter's legacy docstring is stale. Raw arrays, a common display transform, and a documented observer must be used for color comparison.
- CPU/GPU agreement is implementation agreement, not proof of physical correctness. No claim of converged full-scene image quality, calibrated watts/joules, wave-coupling validation, or correct glass interior autofocus is made.

## Comparison with Studio and the component path forward

The separate lighting-isolation evidence at repo-cache/lighting-isolation-20261109 proves that doubling Studio forward samples nearly doubles its field contribution (99.3% extra field in that pilot). Its forward voxel field is nearest-cell sampled and is not the native all-pairs BDPT connector. The tests are different scenes and cannot rank renderer image quality or speed directly.

Keep Studio's responsive camera, scene ownership, exact sphere/rod-region geometry, shared MaterialDatabase, and display profile. The first reusable pieces are the original normalized emitter-domain sampler, path records with wavelength and PDF metadata, common material endpoint evaluation, and a bounded native connection pass. Use the resident pipeline as the next offline reference candidate, with an explicit pixel-film boundary, rather than inserting the whole camera/sensor suite.

Next acceptance sequence:
1. Keep the small wavelength repair and its regression; decide separately when to replace the deployed native binary.
2. Reconcile pre/post-scatter beta, true film endpoint versus first scene surface, PDF flags, and continuous spectral sample pairing. Require reflectance, emission, ray-budget, sample-budget and CPU/resident checks.
3. Use the SAME authored material records, wavelengths, lights, camera, and observer for one small diffuse/occlusion comparison. Compare raw means across seeds and variance, not separately auto-exposed previews.
4. Exercise absorption slabs and dielectric interfaces, then a simple sphere/caustic, with measured conservation and accepted approximation limits.
5. Bridge the actual Studio saved scene without losing analytic spheres, nested rod/injection ownership, or directional boundary media. A flat triangle/material dump is insufficient: prepare_photo_geometry can remove mesh spheres in favor of separate analytic records.
6. Integrate only accepted components into Studio, with a selectable diagnostic/reference path and bounded GPU work. Keep exposure/substep/flash orchestration outside transport and preserve its existing scheduling interfaces.

## Legacy color versus glass rod

Both paths already use MaterialDatabase.instance() and build_mat_buf(freq_hz) through scene_builder.append_saved_studio_mesh_to_gpu_scene. Explicit optical records supply n/k from internal transmittance at a reference thickness via optical_extinction; legacy PBR supplies albedo/interface parameters and generally no authored bulk absorption. The missing support is the current explicit rejection of canonical material_authoring together with optical/region media, which needs a composition adapter. Replacing the observer with native band_to_display_rgb does not solve it. The material worker owns this adapter; no second material database was added here.

## Files and reproduction

- Driver, per-run arguments, raw arrays, images, timing/overflow logs: exposures/dot_bdpt_transport_20261109.
- Regression: tests/test_native_emitter_spectral_coverage.py.
- Native API: csrc/bindings/pybind_kernels.cpp, csrc/kernels/ray_tracer.cpp.
- Connector: csrc/include/t5_connection.h; csrc/shaders/t5_full_connect.comp.glsl; bdpt_pack_t5.comp.glsl; bdpt_scatter_t5.comp.glsl; ray_material_eval.glsl.inc.
- Isolated build uses the existing csrc CMake target; its local parent project only redirects output away from the production binary.

Example resident run (from repository root, using Python 3.11): set process environment SPECTRAL_TRIAL_MODULE_DIR to the isolated module folder, then run trial.py --gpu --all-stages --resident --res 64 --label your_new_label. Check current jobs before GPU use. Existing output labels should not be reused.
