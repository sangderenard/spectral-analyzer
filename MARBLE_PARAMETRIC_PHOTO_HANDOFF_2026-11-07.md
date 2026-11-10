# Continuous marble surface in the Spectral photo worker — handoff

**Current source follow-up (2026-11-09):** Read
[`MARBLE_PARAMETRIC_REGIONS_HANDOFF_2026-11-09.md`](MARBLE_PARAMETRIC_REGIONS_HANDOFF_2026-11-09.md)
before implementing this proposal. In particular, the native kind named
`SDF_SPHERE` is triangle-local barycentric post-hit displacement, not an
exact global sphere intersection. The follow-up distinguishes that original
contract, exact compound-lens transport, and scale-context regions, and
records the missing photo data path and observed source-level parity gaps.

**Status (2026-11-07):** The deployed photo path is triangulated. A continuous-surface toggle is requested but is not operative yet. Do not label a mesh subdivision or smooth normal interpolation as a parametric ray intersection.

## Current boundary

`camera_designer/marble_preview_worker.py` builds the scene with `build_gpu_scene`, then `append_saved_studio_mesh_to_gpu_scene`. The adapter in `camera_designer/scene_builder.py` consumes the exact saved Studio face positions and per-face PBR material IDs and appends them as `TriGeom` and `TriShade` rows before the triangle BVH is built. Both relevant shaders in `demo_pluck_gl.py` use Möller–Trumbore `hit_tri`: the forward light field near line 3621 and the `SensorAccumulator` thin-lens sensor near line 4752. Their BVH traversal calls `hit_tri` and returns a triangle ID. The photo worker has no analytic object primitive or per-hit material field. Its analytic thin lens is the **camera**, not a continuous marble shell.

The native C++ tracer does implement `TRI_PARAM_SURFACE_SDF_SPHERE` and `register_tri_group` (`csrc/kernels/ray_tracer.cpp`, `camera_designer/lens_assembly.py`), but that registered-group contract is separate from the GPU photo worker's packed scene and sensor shaders. The native BDPT route was previously too slow even for bounded 16×16 probes; replacing the current GPU worker with it would miss the 10-minute, 512×512 photo target.

## Smallest truthful first mode

1. Add an **analytic sphere hit** to both GPU paths, with a shared center/radius/material record and a nearest-hit choice between sphere and the existing interior/checker triangle BVH. Test first and second intersections, inside/outside normal orientation, refraction, Fresnel handling, and shadowing. Exclude the triangulated outer shell from both GPU scenes in this mode; keep dyed ribbons, confetti and background as triangles. Merely refining the normal after a triangle hit leaves faceted silhouette and boundary artifacts.
2. Carry the saved object's verified generator and design into the photo job. The MRP1 upload presently verifies every triangle against the saved geometry SHA-256; keep that integrity check, then derive the analytic shell center/radius from the authenticated spherical recipe. Reject dice and any design whose outer geometry is not a sphere. Include surface mode in the job digest and status.
3. Start with a uniform shell material. Partial enamel and painted bands need a continuous hit-point material field derived from the saved recipe, including its UV/phase/band semantics; per-triangle material IDs cannot define a continuous shell. Keep these designs in triangle mode or reject parametric mode explicitly until material mapping is verified.
4. Compare identical saved glass spheres in both modes with an oblique checker, inside/outside rays and multi-bounce RGB exposure. Require a visibly round silhouette and continuous refraction at low triangle count before enabling the Studio toggle. Preserve the triangle option for its optical artifacts, as requested.

This is a renderer-contract change, not a UI-only switch. It should be developed and measured in Spectral first, then exposed by Meowdoku's photo contract and Studio controls. No Wasm build is involved.
