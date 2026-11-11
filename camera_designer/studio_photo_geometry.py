"""CPU-only scene preparation shared by preflight and actual photo jobs."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from .camera_preset import thin_lens_housing_preset
from .camera_frame import thin_lens_housing_frame
from .scene_builder import append_saved_studio_mesh_to_gpu_scene, build_gpu_scene
from .studio_sphere_regions import validate_sphere_regions, pack_sphere_regions, geometry_statistics


CAMERA_GEOMETRY_VERSION = "native-housing-behind-thin-lens-v1"


def camera_geometry_key(camera, target_default):
    """Only changes that physically move or resize the authored pupil."""
    return (tuple(np.asarray(camera.get("eye_m", [0, 0, .07]), np.float32)),
            tuple(np.asarray(camera.get("target_m", target_default), np.float32)),
            float(camera.get("aperture_radius_m", .005)),
            int(camera.get("aperture_blades", 0)),
            float(camera.get("aperture_rotation_radians", 0)))


def prepare_photo_geometry(config, vertices, material_ids, pbr, *, material_lineage_out=None):
    transport = config.get("transport", "fixed:1")
    wavelengths = [.5875618] if transport == "fixed:1" else [.450, .550, .650]
    policy = config.get("sphere_coverage", "mesh")
    budget = config.get("triangle_budget", 250000)
    regions = None
    if policy == "exact":
        regions = validate_sphere_regions(config.get("analytic_spheres"), vertices, material_ids, pbr)
    camera = config.get("camera", {})
    preset = thin_lens_housing_preset(camera)
    checker = config.get("checker", {})
    checker_pos = checker.get("center_m", [0, 0, .30])
    checker_width = float(checker.get("width_m", .15))
    eye = camera.get("eye_m", [0, 0, .07])
    target = camera.get("target_m", checker_pos)
    camera_frame = thin_lens_housing_frame(preset, eye, target)
    lights = config.get("lights", [
        {"pos": [.14, .10, .17], "dir": [-.14, -.10, .13],
         "power": 1.0, "label": "preview-stage"},
    ])
    fixtures = config.get("fixtures")
    if fixtures is not None:
        # Studio sources come only from the completed authored geometry.
        # Empty/dark fixtures remain dark; no fallback or centroid lights.
        lights = []
    if fixtures is not None:
        # "fixtures" is the Phong preview's entire non-marble scene at the
        # moment of capture -- floor, plaster wall, backdrop roll, and
        # light fixtures, combined client-side into one buffer and
        # appended below exactly like the saved marble mesh. No separate
        # placeholder backdrop plane: that would be a second, independent
        # piece of geometry standing in for a room that's already present.
        scene_object = None
    else:
        scene_object = {"type": "checker_target", "checks": int(checker.get("checks", 4)),
                        "width": checker_width, "pos": checker_pos, "emissive": True,
                        "normal_sign": int(checker.get("normal_sign", -1))}
    scene = build_gpu_scene(
        preset, lights,
        scene_object=scene_object,
        wavelengths_um=wavelengths,
        camera_frame=camera_frame,
    )
    object_offset = len(scene[0])
    scene, original_indices = append_saved_studio_mesh_to_gpu_scene(
        scene, vertices, material_ids, pbr,
        scale_m=float(config.get("scale_m", .025)),
        center_m=config.get("object_center_m", [0, 0, .18]),
        wavelengths_um=wavelengths,
        spectral_emission_rgb=(transport == "fixed:3"),
        boundary_media=config.get("boundary_media"),
        optical_media=config.get("optical_media"),
        region_media=config.get("region_media"),
        material_authoring=config.get("material_authoring"),
        canonical_volume_optics=config.get("canonical_volume_optics"),
        return_index_map=True,
        material_lineage_out=material_lineage_out,
    )
    # Real emissive fixture geometry (the Studio light housings/panels),
    # appended through the exact same function used for the saved marble
    # mesh -- any triangle in here whose PBR row carries emission becomes a
    # genuine area light via Spectral's own MAT_FLAG_EMISSIVE detection,
    # not a synthetic point-light stand-in. Already baked to real photo-
    # metre coordinates client-side, so scale/center here are the identity.
    if fixtures is not None:
        fx_verts = np.asarray(fixtures["vertices"], np.float32).reshape(-1, 3, 3)
        from .studio_scene_engines import project_scene_engines, advanced_surface_triangles, ADVANCED_CONDUCTOR_RUNTIME_ENABLED
        fx_ids, fx_pbr, fx_boundary, fx_optics = project_scene_engines(fixtures)
        scene = append_saved_studio_mesh_to_gpu_scene(
            scene, fx_verts, fx_ids, fx_pbr,
            scale_m=1.0, center_m=[0.0, 0.0, 0.0],
            material_authoring=fixtures.get("material_authoring"),
            emitter_profiles=fixtures.get("emitter_profiles"),
            boundary_media=fx_boundary, optical_media=fx_optics,
            advanced_surface_mask=advanced_surface_triangles(fixtures),
            wavelengths_um=wavelengths,
            spectral_emission_rgb=(transport == "fixed:3"),
            material_lineage_out=material_lineage_out,
            lineage_source=(fixtures["material_ids"], np.asarray(fixtures["pbr"],np.float32).reshape(-1,16)),
        )
    input_count = len(scene[0])
    spheres = np.zeros((0, 8), np.float32)
    if regions is not None:
        scene, spheres = pack_sphere_regions(scene, original_indices, object_offset, regions,
            scale_m=float(config.get("scale_m", .025)), center_m=config.get("object_center_m", [0,0,.18]))
    if fixtures is not None:
        from .studio_native_emitters import native_emissive_sources, sampling_settings
        sampling=sampling_settings(config)
        sources = native_emissive_sources(scene, wavelengths,ray_budget=sampling['ray_budget'],seed=sampling['seed'])
        if sampling['refresh_each_sweep'] and len(sources)>config['live'].get('ray_batch_size',8192):raise ValueError(f'live ray_batch_size must cover all {len(sources)} rounded native source roots')
        scene = (*scene[:7], sources)
    stats = geometry_statistics(input_triangles=input_count, packed_triangles=len(scene[0]),
        sphere_count=len(spheres), policy=policy, budget=budget)
    if fixtures is not None:
        from .studio_native_emitters import VERSION
        stats['emissive_sampling'] = {'version': VERSION, 'root_samples': len(scene[7]),
            'geometry': 'authored emissive triangles', 'independent_lights': False,
            'camera_estimator': 'area NEE plus camera paths and existing forward field',
            'voxel_field_in_image': True}
    if fixtures is not None:
        engines=fixtures.get("material_engines")
        stats["scene_material_engines"] = {
            "stage": next((g["engine"] for g in engines["groups"] if g["name"]=="stage"), "legacy") if engines else "legacy",
            "room": next((g["engine"] for g in engines["groups"] if g["name"]=="room"), "legacy") if engines else "legacy",
            "closed_optical_volumes": len(engines["volumes"]) if engines else 0,
            "optical_law": "shared real-index Fresnel/Snell and spectral Beer absorption; no volume scattering",
            "advanced_conductor_enabled": ADVANCED_CONDUCTOR_RUNTIME_ENABLED,
            "opaque_metal_law": "single-scatter GGX with canonical roughness; explicit native n/k or artistic spectral F0" if ADVANCED_CONDUCTOR_RUNTIME_ENABLED else "existing surface compatibility law; new conductor path pending release",
        }
    stats["camera_geometry"] = {
        "version": CAMERA_GEOMETRY_VERSION,
        "body_dimensions_m": [preset.body.width, preset.body.height, preset.body.depth],
        "eye_m": list(map(float, eye)), "target_m": list(map(float, target)),
        "pupil_radius_m": float(preset.aperture_stop.r_outer),
        "front_clearance_m": float(preset.aperture_stop.thickness_m),
        "optics": "analytical thin lens; native nonrefractive housing behind pupil",
    }
    return scene, spheres, stats, checker_pos


def main():
    # No renderer import, GL context, service access, or exposure side effect.
    payload, config_path = map(Path, sys.argv[1:3])
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with np.load(payload, allow_pickle=False) as saved:
            _, _, stats, _ = prepare_photo_geometry(config, saved["vertices"], saved["material_ids"], saved["pbr"])
        print(json.dumps({"ok": True, "geometry_stats": stats}), flush=True)
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}), flush=True)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
