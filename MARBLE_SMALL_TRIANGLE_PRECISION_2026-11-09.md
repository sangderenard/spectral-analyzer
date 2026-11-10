# Miniature triangle intersection correction, 2026-11-09

## Verified fault
Both base forward and sensor GLSL `hit_tri` functions rejected `abs(det) < 1e-8`. This determinant is proportional to triangle area in world metres and ray direction length, so the fixed constant deletes valid small triangles rather than just rejecting near-parallel rays.

The 14:11 live payload `f379a0425edc4f44653f4000f3abb56402699272dee5a7f1e1ebb6b922ecb2c9` contains 72,960 full uploaded triangles, 24,320 each for shell material 14 and ribbons 16/17, with scale_m 0.01. Each ribbon has 16,368 triangles whose twice-area is below 1e-8: 67.3% cannot intersect any normalized ray under the old test. At a head-on angle the relative test rejects zero of these actual triangles. 512 shell triangles also meet the old always-miss bound, but this job replaces the outer shell with one exact sphere.

The saved job reports 119,536 input scene triangles, 95,216 effective triangles plus one exact sphere, with a 250,000 budget. This path rejects over-budget uploads; it does not subsample the mesh. No minimized marble image is used as its intersection geometry.

## Change
Use the existing opt-in region path's float32-relative determinant threshold in BOTH legacy shader definitions: 2 * float32 epsilon * |e1| * |e2| * |ray direction|. Degenerate and sufficiently parallel triangles still reject. No material, field resolution, scene geometry or display transformation changes.

## Checks
- `python -m pytest tests/test_studio_small_triangle_precision.py tests/test_studio_region_gpu.py -q`: 10 passed.
- Offline glslangValidator compilation of full base forward, base sensor, region forward, region sensor shaders: all passed.
- Actual payload CPU area bound reproduced the miss counts above.
- `tests/studio_small_triangle_gpu_check.py` added for isolated before/after GPU intersections at every uploaded triangle centroid. NOT RUN yet: user GPU was actively rendering, 100% utilization. Run only when idle; optional argument is payload.npz.

## Limits / separate findings
This proves a geometry correctness bug, not that it accounts for every visible dark/color/checker patch. No before/after image has yet been rendered. Existing active workers retain already compiled old shaders until a coordinated new worker launch. No active worker was stopped.

Phong uploads smooth per-vertex normals; the saved ray mesh transmits positions/materials and reconstructs geometric face normals. Also the legacy camera samples a nearest-cell 96-cubed whole-room forward light field. These can produce distinct visual differences and need separate controlled investigation; neither is silently changed here. The fixed 1e-6 hit-distance/continuation thresholds remain in legacy and may miss very thin features. The optional region variant has a different coordinated position-relative policy.

## Coarse-field measurement
CPU reconstruction of that exact payload/config gives padded field bounds (-0.14,-0.0602,0.047) to (0.14,0.1304,0.34809146)m. At 96 cubed, cells are 2.9167 x 1.9854 x 3.1364 mm. The marble radius is 10mm (diameter20mm), giving only about6-10 cells across it. `sample_layer_energy` floor-selects one cell, and opaque ribbon shading adds this term. This is a separate plausible source of rectangular illumination patches; causal image isolation remains pending. Trilinear filtering would hide discontinuities but does not recover missing spatial resolution and can leak across material boundaries, so it is not silently substituted as a correctness fix.

## Actual GPU gate completed
After the user explicitly allowed interrupting their exploratory render, the isolated gate ran on NVIDIA GeForce RTX3060 without stopping the existing worker. Both actual legacy GLSL functions, using the uploaded72,960-triangle payload, hit72,960/72,960 centroid-normal rays with the fix versus39,712/72,960 using the old cutoff. Whole command completed in3.9s. This verifies intersections, not a full rendered before/after image. Coordinated production worker reload remains owned by the release worker.
