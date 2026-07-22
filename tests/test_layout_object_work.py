import json

import numpy as np
from PIL import Image

from camera_software import (
    LayoutWorkProgressCache,
    RenderObjectLibrary,
    PanelPatchAsset,
    PanelPatchRole,
    build_layout_object_work_manifest,
    build_window_element_scene_manifest,
    compose_usd_stage,
    layout_program_ui,
    program_ui_manifest,
    rack_focus_pose_animation,
    write_layout_usd_package,
)


def _layout():
    return layout_program_ui(
        program_ui_manifest(), 360, 180, work_width=100, work_height=100
    )


def test_bespoke_layout_requests_become_design_object_subtype_hierarchy(tmp_path):
    layout = _layout()
    manifest = build_layout_object_work_manifest(
        layout,
        str(tmp_path),
        rack_frames=5,
        hold_frames=3,
        focus_location_count=4,
    )

    assert {item.object_key for item in manifest.objects} == {
        "layout-panel-style:bakery-slate",
        "layout-control-style:bakery-slate",
    }
    assert all(len(item.subtypes) == 1 for item in manifest.objects)
    discovered_consumers = sum(
        len(subtype.consumers)
        for item in manifest.objects for subtype in item.subtypes
    )
    authored_requests = sum(
        len(requests) for requests in layout.panel_object_requests.values()
    )
    assert discovered_consumers == authored_requests
    subtype = manifest.objects[0].subtypes[0]
    assert subtype.subtype_key == "representative-square"
    assert len(subtype.pose_animations) == 1
    assert len(subtype.pose_animations[0].frames) == 13
    assert subtype.parameter_spec["scene_family"] == (
        "parametric-nine-slice-panel-v1"
    )
    assert set(subtype.parameter_spec["patches"]) == {
        role.value for role in PanelPatchRole
    }
    assert "target_width_px" in subtype.parameter_spec["instance_parameters"]
    assert subtype.aspect_variants
    assert all(
        variant.sampling_policy
        == "repeat_square_tiles_clip_partial_terminal_tile"
        for variant in subtype.aspect_variants
    )
    assert all(
        variant.physical_pixel_aspect == 1.0
        for variant in subtype.aspect_variants
    )
    by_owner = {}
    for item in manifest.objects:
        for child in item.subtypes:
            for consumer in child.consumers:
                by_owner.setdefault(consumer.owner_id, []).append(consumer)
    for owner_id, consumers in by_owner.items():
        owner = layout.region(owner_id)
        assert sum(
            consumer.target_rect_px[2] * consumer.target_rect_px[3]
            for consumer in consumers
        ) == owner[2] * owner[3]


def test_rack_focus_animation_has_phases_and_quantized_percentage_locations():
    animation = rack_focus_pose_animation(
        "object", "subtype",
        rack_frames=6,
        hold_frames=2,
        focus_location_count=5,
    )

    assert [frame.phase for frame in animation.frames[:6]] == ["rack_in"] * 6
    assert [frame.phase for frame in animation.frames[6:8]] == ["hold"] * 2
    assert [frame.phase for frame in animation.frames[8:]] == ["rack_out"] * 6
    assert animation.frames[0].focus_location_percent == 100.0
    assert animation.frames[5].focus_location_percent == 0.0
    assert animation.frames[-1].focus_location_percent == 100.0
    assert {
        frame.focus_location_percent for frame in animation.frames
    } <= {0.0, 25.0, 50.0, 75.0, 100.0}
    assert len({
        frame.progress_cache_key for frame in animation.frames
    }) == len(animation.frames)


def test_progress_cache_prefers_accepted_hold_pose_as_panel_replacement(tmp_path):
    manifest = build_layout_object_work_manifest(_layout(), str(tmp_path))
    subtype = manifest.objects[0].subtypes[0]
    cache = LayoutWorkProgressCache(manifest.progress_cache_path)
    still_path = tmp_path / "still.png"
    hold_path = tmp_path / "hold.png"
    Image.new("RGBA", (9, 9), (20, 30, 40, 255)).save(still_path)
    Image.new("RGBA", (9, 9), (90, 80, 70, 255)).save(hold_path)
    cache.update(
        subtype.still_cache_key,
        status="completed",
        convergence=1.0,
        artifact_path=str(still_path),
    )
    assert cache.replacement_path(subtype) == str(still_path.resolve())
    cache.update(
        subtype.pose_animations[0].replacement_cache_key,
        status="accepted",
        convergence=1.0,
        artifact_path=str(hold_path),
    )
    assert cache.replacement_path(subtype) == str(hold_path.resolve())

    patch = PanelPatchAsset(
        PanelPatchRole.CENTER,
        object_key=manifest.objects[0].object_key,
        subtype_key=subtype.subtype_key,
    )
    loaded = cache.patch_loader(manifest)(patch)
    assert loaded is not None
    assert np.array_equal(loaded[4, 4], np.array([90, 80, 70, 255]))


def test_progress_cache_live_adopts_and_refreshes_developing_artifacts(tmp_path):
    manifest = build_layout_object_work_manifest(_layout(), str(tmp_path))
    subtype = manifest.objects[0].subtypes[0]
    observer = LayoutWorkProgressCache(manifest.progress_cache_path)
    writer = LayoutWorkProgressCache(manifest.progress_cache_path)
    artifact = tmp_path / "developing.png"
    Image.new("RGBA", (5, 5), (7, 11, 13, 255)).save(artifact)

    writer.update(
        subtype.still_cache_key,
        status="developing",
        convergence=0.0001,
        samples=1,
        artifact_path=str(artifact),
    )

    signature = observer.revision_signature()
    assert observer.replacement_path(subtype) == str(artifact.resolve())
    patch = PanelPatchAsset(
        PanelPatchRole.CENTER,
        object_key=manifest.objects[0].object_key,
        subtype_key=subtype.subtype_key,
    )
    loaded = observer.patch_loader(manifest)(patch)
    assert np.array_equal(loaded[0, 0], np.array([7, 11, 13, 255]))

    Image.new("RGBA", (6, 5), (17, 19, 23, 255)).save(artifact)
    assert observer.revision_signature() != signature


def test_work_manifest_serializes_cache_and_replacement_contract(tmp_path):
    manifest = build_layout_object_work_manifest(_layout(), str(tmp_path))
    path = tmp_path / "layout_work.json"

    manifest.save(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["source_layout"] == "program-backdrop"
    assert payload["replacement_policy"].startswith(
        "first_valid_developing_hold_pose"
    )
    assert payload["objects"][0]["subtypes"][0]["pose_animations"]


def test_openusd_package_composes_subtypes_animations_and_full_layout_stage(tmp_path):
    manifest = build_layout_object_work_manifest(_layout(), str(tmp_path))
    cache = LayoutWorkProgressCache(manifest.progress_cache_path)
    library = RenderObjectLibrary(str(tmp_path / "library.json"))
    library.register_parametric_layout_manifest(manifest)

    package = write_layout_usd_package(
        manifest, cache, str(tmp_path / "usd"), render_library=library,
        window_element_manifest=build_window_element_scene_manifest(_layout()),
    )

    assert len(package.object_layers) == 2
    assert len(package.subtype_layers) == 2
    assert len(package.animation_layers) == 2
    stage = (tmp_path / "usd" / "stages" / "program_backdrop.usda").read_text(
        encoding="utf-8"
    )
    animation = next(iter(package.animation_layers.values()))
    animation_text = open(animation, encoding="utf-8").read()
    subtype = open(next(iter(package.subtype_layers.values())), encoding="utf-8").read()
    object_layer = open(next(iter(package.object_layers.values())), encoding="utf-8").read()
    assert '#usda 1.0' in stage
    assert 'kind = "assembly"' in stage
    assert "references = @../objects/" in stage
    assert "xformOp:translate" in stage
    assert "custom int4 targetRectPx" in stage
    assert "sourceScene = @" in subtype
    assembly = json.loads(
        (tmp_path / "usd" / "assembly_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert assembly["aspect_library"]
    assert assembly["holistic_exposure"]["start_policy"] == (
        "content_static_and_render_work_idle"
    )
    assert assembly["recrop_reuse"][
        "requires_same_reflection_boundary_signature"
    ]
    assert "references = @../subtypes/" in object_layer
    assert 'def Scope "PoseAnimations"' in subtype
    assert "focusDistance.timeSamples" in animation_text
    assert 'def Camera "DemonstrationCamera"' in animation_text
    assert package.window_element_manifest_path
    assert package.window_element_layer_path
    assert "retained_2d_layout_before_rasterization" in stage


def test_generic_stage_builder_adds_retained_scene_layers_by_reference(tmp_path):
    asset = tmp_path / "asset.usda"
    asset.write_text(
        '#usda 1.0\n(defaultPrim = "Asset")\ndef Xform "Asset" {}\n',
        encoding="utf-8",
    )
    stage_path = tmp_path / "composite.usda"

    compose_usd_stage(
        str(stage_path),
        [{
            "name": "PanelInstance",
            "asset_path": str(asset),
            "prim_path": "/Asset",
            "metadata": {"role": "panel"},
        }],
    )

    content = stage_path.read_text(encoding="utf-8")
    assert "references = @asset.usda@</Asset>" in content
    assert 'custom string role = "panel"' in content
