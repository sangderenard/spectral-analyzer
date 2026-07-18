from dataclasses import replace
from pathlib import Path

import numpy as np

from camera_software import (
    BakeTargetKind,
    AtlasCaptureSpec,
    CharacterAtlas,
    DEFAULT_INK_CONDITION,
    DisplayPrimitive,
    DisplayPrimitiveKind,
    DisplayProductKind,
    ExtrudedTokenAsset,
    ExtrusionAssetSpec,
    FontAssetSpec,
    LightFieldCondition,
    RenderAssetCatalog,
    RenderedAssetRecord,
    RotatingStageSpec,
    SENSOR_DISPLAY_ORIENTATION,
    build_ink_on_slate_order,
    ink_order_for_request,
    plan_ink_atlas_bake,
    plan_display_scene_bake,
)
import live_spectral_text_demo as demo


def _token(token: str = "AXE") -> ExtrudedTokenAsset:
    return ExtrudedTokenAsset(
        token=token,
        font=FontAssetSpec("DejaVu Sans", "bold", "normal"),
        extrusion=ExtrusionAssetSpec(
            height_m=0.05,
            line_height_m=0.05,
            text_box_m=(0.3, 0.1),
        ),
        material="text_surface",
    )


def test_token_asset_identity_covers_font_geometry_material_and_content():
    asset = _token()

    assert asset.asset_key == _token().asset_key
    assert asset.asset_key != asset.with_token("A").asset_key
    assert asset.asset_key != replace(
        asset, font=replace(asset.font, weight="normal")
    ).asset_key
    assert asset.asset_key != replace(
        asset, extrusion=replace(asset.extrusion, depth_ratio=0.12)
    ).asset_key
    assert asset.asset_key != replace(asset, material_revision="2").asset_key


def test_token_asset_is_the_scene_order_adapter():
    raw = _token("fresh").scene_order_object("status", "status-plane")

    assert raw["token"] == "fresh"
    assert raw["font"]["family"] == "DejaVu Sans"
    assert raw["geometry"]["embed_plane"] == "status-plane"
    assert raw["geometry"]["extrusion_depth_ratio"] == 0.08
    assert raw["geometry"]["text_box_m"] == [0.3, 0.1]


def test_character_atlas_prefers_sequence_then_falls_back_to_unique_chars(tmp_path):
    catalog = RenderAssetCatalog(str(tmp_path / "assets.json"))
    condition = LightFieldCondition(azimuth_deg=45.0)
    asset = _token("AREA")
    glyph_a = asset.with_token("A")
    record_a = RenderedAssetRecord(
        glyph_a.asset_key,
        condition.condition_key,
        DisplayProductKind.LIGHT_FIELD,
        preview_path="a.png",
    )
    catalog.record(record_a)

    fallback = CharacterAtlas(catalog).resolve(asset, condition)
    assert fallback.exact is None
    assert fallback.uses_character_fallback
    assert fallback.character_artifacts == (record_a,)
    assert {item.token for item in fallback.missing_characters} == {"R", "E"}

    exact = RenderedAssetRecord(
        asset.asset_key,
        condition.condition_key,
        DisplayProductKind.LIGHT_FIELD,
        preview_path="area.png",
    )
    catalog.record(exact)
    resolved = CharacterAtlas(
        RenderAssetCatalog(str(tmp_path / "assets.json"))
    ).resolve(asset, condition)
    assert resolved.exact == exact
    assert resolved.character_artifacts == ()
    assert resolved.missing_characters == ()


def test_scene_bake_plans_pages_parts_sequences_and_atlas_fallbacks():
    scene = demo.build_self_rendering_program_scene(
        "editable", display_width=80, display_height=48
    )
    stage = RotatingStageSpec(
        azimuth_views=2,
        elevations_deg=(0.0,),
        light_rigs=("warm",),
        material_variants=("default",),
    )
    plan = plan_display_scene_bake(scene, stage, RenderAssetCatalog())

    kinds = {request.target_kind for request in plan.requests}
    assert kinds == {
        BakeTargetKind.PAGE,
        BakeTargetKind.PART,
        BakeTargetKind.TOKEN_SEQUENCE,
        BakeTargetKind.ATLAS_GLYPH,
    }
    pages = [
        request for request in plan.requests
        if request.target_kind is BakeTargetKind.PAGE
    ]
    parts = [
        request for request in plan.requests
        if request.target_kind is BakeTargetKind.PART
    ]
    assert len(pages) == 2
    assert len(parts) == 2 * len(scene.objects)
    # Repeated letters and labels share one atlas request per condition/style.
    glyph_keys = [
        request.request_key for request in plan.requests
        if request.target_kind is BakeTargetKind.ATLAS_GLYPH
    ]
    assert len(glyph_keys) == len(set(glyph_keys))
    assert plan.cached == ()

    completed = RenderAssetCatalog()
    first = plan.requests[0]
    record = completed.complete(
        first, preview_path="first.png", width=80, height=48, samples=4
    )
    assert completed.find(
        first.target_key, first.condition, first.product_kind
    ) == record
    assert record.metadata["target_kind"] == first.target_kind.value


def test_live_order_uses_asset_contract_without_changing_scene_order_shape():
    scene = demo.build_program_display_scene("asset-backed", display_width=80, display_height=48)
    primitive = DisplayPrimitive(DisplayPrimitiveKind.TEXT, content="updated")
    scene = replace(
        scene,
        objects=(replace(scene.objects[0], primitive=primitive),),
    )
    order = demo.build_display_scene_order(scene)
    raw = order["defaults"]["objects"][0]

    assert raw["token"] == "updated"
    assert raw["geometry"]["profile"] == "straight"
    assert raw["geometry"]["material"] == "text_surface"
    assert RotatingStageSpec.knobs()[0].name == "azimuth_views"


def test_production_defaults_to_one_head_on_ring_light_frame():
    stage = RotatingStageSpec()
    assert stage.conditions() == (DEFAULT_INK_CONDITION,)
    assert RotatingStageSpec.knobs()[0].default == 1


def test_ink_order_preserves_accepted_scene_and_adds_padded_capture():
    import scene_orders

    capture = AtlasCaptureSpec()
    package = build_ink_on_slate_order(("A", "AXE"), capture=capture)
    glyph, token = scene_orders.resolved_jobs(package)

    assert glyph["image"] == {"width": 128, "height": 128}
    assert capture.content_region == (24, 24, 80, 80)
    assert "position_m" not in glyph["camera"]
    assert glyph["planes"][0]["size_m"] == [0.52, 0.42]
    assert glyph["materials"]["matte_black"]["roughness"] == 0.9
    assert glyph["materials"]["glossy_red"]["roughness"] == 0.065
    assert glyph["materials"]["glossy_red"]["ior"] == 1.52
    assert glyph["geometry"]["profile"] == "circular"
    assert glyph["geometry"]["depth_m"] == 0.032
    assert glyph["geometry"]["embed_fraction"] == 0.5
    assert token["geometry"]["depth_m"] == 0.028
    assert token["geometry"]["height_m"] == 0.13


def test_ink_atlas_accepts_trickled_tokens_and_only_queues_missing_work():
    catalog = RenderAssetCatalog()
    capture = AtlasCaptureSpec(width=96, height=96, content_width=64, content_height=64)
    first = plan_ink_atlas_bake(("AXE",), catalog, capture=capture)

    assert {request.product_kind for request in first.requests} == {
        DisplayProductKind.IMAGE
    }
    assert {request.condition for request in first.requests} == {
        DEFAULT_INK_CONDITION
    }
    assert {
        request.token_asset.token
        for request in first.requests
        if request.target_kind is BakeTargetKind.ATLAS_GLYPH
    } == {"A", "X", "E"}
    assert not any(
        request.target_kind is BakeTargetKind.TOKEN_SEQUENCE
        for request in first.requests
    )

    glyph_a = next(
        request for request in first.requests
        if request.target_kind is BakeTargetKind.ATLAS_GLYPH
        and request.token_asset.token == "A"
    )
    catalog.complete(
        glyph_a,
        preview_path="A.png",
        samples=8,
        metadata={
            "bounded_background_render": True,
            "sensor_display_orientation": SENSOR_DISPLAY_ORIENTATION,
            "refinement_state": "converged",
            "atlas_quality": {
                "composable": True,
                "converged": True,
                "glyph_exposure_coverage": 1.0,
                "glyph_radiance_coverage": 1.0,
            },
        },
    )
    second = plan_ink_atlas_bake(("AXE", "AREA"), catalog, capture=capture)
    missing_glyphs = {
        request.token_asset.token
        for request in second.requests
        if request.target_kind is BakeTargetKind.ATLAS_GLYPH
    }
    assert "A" not in missing_glyphs
    assert {"X", "E", "R"} <= missing_glyphs
    assert second.cached
    assert not any(
        request.target_kind is BakeTargetKind.TOKEN_SEQUENCE
        for request in second.requests
    )


def test_token_sequence_work_is_blocked_until_every_glyph_converges():
    catalog = RenderAssetCatalog()
    initial = plan_ink_atlas_bake(("AXE",), catalog)
    for request in initial.requests:
        assert request.target_kind is BakeTargetKind.ATLAS_GLYPH
        catalog.complete(
            request,
            samples=4,
            metadata={
                "bounded_background_render": True,
                "sensor_display_orientation": SENSOR_DISPLAY_ORIENTATION,
                "refinement_state": "converged",
                "refinement_pass": 4,
                "atlas_quality": {
                    "composable": True,
                    "converged": True,
                    "glyph_exposure_coverage": 1.0,
                    "glyph_radiance_coverage": 1.0,
                },
            },
        )

    sequence_plan = plan_ink_atlas_bake(("AXE",), catalog)

    assert len(sequence_plan.requests) == 1
    sequence = sequence_plan.next_request
    assert sequence is not None
    assert sequence.target_kind is BakeTargetKind.TOKEN_SEQUENCE
    assert sequence.token_asset.token == "AXE"
    assert ink_order_for_request(sequence)["jobs"][0]["token"] == "AXE"


def test_token_string_work_is_blocked_until_all_tokens_converge():
    catalog = RenderAssetCatalog()

    glyph_plan = plan_ink_atlas_bake(
        ("AB", "CD"), catalog, token_strings=("AB CD",)
    )
    assert {
        request.target_kind for request in glyph_plan.requests
    } == {BakeTargetKind.ATLAS_GLYPH}
    for request in glyph_plan.requests:
        catalog.complete(
            request,
            metadata={
                "bounded_background_render": True,
                "sensor_display_orientation": SENSOR_DISPLAY_ORIENTATION,
                "refinement_state": "converged",
                "atlas_quality": {
                    "composable": True,
                    "converged": True,
                    "glyph_exposure_coverage": 1.0,
                    "glyph_radiance_coverage": 1.0,
                },
            },
        )

    token_plan = plan_ink_atlas_bake(
        ("AB", "CD"), catalog, token_strings=("AB CD",)
    )
    assert {
        request.target_kind for request in token_plan.requests
    } == {BakeTargetKind.TOKEN_SEQUENCE}
    for request in token_plan.requests:
        catalog.complete(
            request,
            metadata={
                "bounded_background_render": True,
                "sensor_display_orientation": SENSOR_DISPLAY_ORIENTATION,
                "refinement_state": "converged",
                "atlas_quality": {
                    "composable": True,
                    "converged": True,
                    "glyph_exposure_coverage": 1.0,
                    "glyph_radiance_coverage": 1.0,
                },
            },
        )

    string_plan = plan_ink_atlas_bake(
        ("AB", "CD"), catalog, token_strings=("AB CD",)
    )
    assert len(string_plan.requests) == 1
    assert string_plan.next_request.target_kind is BakeTargetKind.TOKEN_STRING
    assert string_plan.next_request.token_asset.token == "AB CD"
    catalog.complete(
        string_plan.next_request,
        metadata={
            "bounded_background_render": True,
            "sensor_display_orientation": SENSOR_DISPLAY_ORIENTATION,
            "refinement_state": "converged",
            "atlas_quality": {
                "composable": True,
                "converged": True,
                "glyph_exposure_coverage": 1.0,
                "glyph_radiance_coverage": 1.0,
            },
        },
    )
    assert plan_ink_atlas_bake(
        ("AB", "CD"), catalog, token_strings=("AB CD",)
    ).requests == ()


def test_ink_atlas_requeues_legacy_one_epoch_record_instead_of_composing_it():
    catalog = RenderAssetCatalog()
    initial = plan_ink_atlas_bake(("A",), catalog)
    request = initial.next_request
    assert request is not None
    catalog.complete(
        request,
        linear_path="nearly-empty.npy",
        samples=1,
        metadata={"bounded_background_render": True},
    )

    replanned = plan_ink_atlas_bake(("A",), catalog)
    assert replanned.next_request is not None
    assert replanned.next_request.target_key == request.target_key
    assert replanned.next_request.refinement_pass == 0
    assert replanned.cached == ()


def test_atlas_restarts_incompatible_mirrored_accumulation_contract():
    catalog = RenderAssetCatalog()
    request = plan_ink_atlas_bake(("F",), catalog).next_request
    assert request is not None
    catalog.complete(
        request,
        samples=33,
        metadata={
            "bounded_background_render": True,
            "sensor_display_orientation": "native-transpose-v2",
            "refinement_state": "developing",
            "refinement_pass": 33,
            "atlas_quality": {
                "composable": True,
                "converged": False,
                "glyph_exposure_coverage": 1.0,
                "glyph_radiance_coverage": 1.0,
            },
        },
    )

    replanned = plan_ink_atlas_bake(("F",), catalog)

    assert replanned.next_request is not None
    assert replanned.next_request.refinement_pass == 0
    assert replanned.cached == ()


def test_atlas_executor_restores_and_overwrites_until_image_converges(
    tmp_path, monkeypatch
):
    catalog = RenderAssetCatalog(str(tmp_path / "catalog.json"))
    request = plan_ink_atlas_bake(("A",), catalog).next_request
    assert request is not None
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            launched.append((command, dict(kwargs["env"])))
            epochs = int(kwargs["env"]["SPECTRAL_SENSOR_MAX_EPOCHS"])
            self.stdout = [
                f"  [sensor-refine] epoch {index}/{epochs}\n"
                for index in range(1, epochs + 1)
            ]
            out_dir = command[command.index("--out-dir") + 1]
            np.save(
                str(Path(out_dir) / "0000_cpp_linear.npy"),
                np.ones((128, 128, 3), np.float32),
            )
            np.save(
                str(Path(out_dir) / "0000_cpp_sum_linear.npy"),
                np.ones((128, 128, 3), np.float32),
            )
            np.save(
                str(Path(out_dir) / "0000_cpp_exposure_weight.npy"),
                np.ones((128, 128), np.float32),
            )

        def wait(self):
            return 0

        def poll(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr(demo.subprocess, "Popen", Process)
    renderer = demo.InkAtlasSubprocessRenderer(str(tmp_path), catalog)
    records = []
    for pass_index in range(4):
        record = renderer(replace(request, refinement_pass=pass_index))
        records.append(record)

    first_command, first_environment = launched[0]
    assert "exposure_render_demo.py" in " ".join(first_command)
    assert "--scene-order" in first_command
    assert first_environment["SPECTRAL_SENSOR_MAX_EPOCHS"] == "1"
    assert first_environment["SPECTRAL_SENSOR_CONTINUOUS"] == "1"
    assert first_environment["SPECTRAL_SENSOR_TOP_K"] == "1024"
    assert first_environment["SPECTRAL_SENSOR_STEPS_PER_LAYER"] == "64"
    assert first_environment["SPECTRAL_SENSOR_SAMPLES_PER_NODE"] == "1024"
    assert "SPECTRAL_SENSOR_RESTORE_SUM" not in first_environment
    for _command, environment in launched[1:]:
        assert environment["SPECTRAL_SENSOR_RESTORE_SUM"].endswith(
            "restore_sensor_sum_native.npy"
        )
        assert environment["SPECTRAL_SENSOR_RESTORE_WEIGHT"].endswith(
            "restore_sensor_weight_native.npy"
        )
    assert len({
        environment["SPECTRAL_EXPOSURE_SEED_OFFSET"]
        for _command, environment in launched
    }) == 4
    assert record.metadata["bounded_background_render"] is True
    assert record.metadata["refinement_state"] == "converged"
    assert record.metadata["atlas_quality"]["converged"] is True
    assert record.samples == 4
    assert [item.linear_path for item in records] == [record.linear_path] * 4
    assert record.metadata["sprite_path"].endswith("raytraced_sprite.npz")
    assert record.metadata["capture"]["content_region"] == [24, 24, 80, 80]
    work_revision, work_path, work_token, work_pass, active = (
        renderer.work_snapshot()
    )
    assert work_revision > 0
    assert work_path == record.preview_path
    assert work_token == "A"
    assert work_pass == 4
    assert not active
    assert catalog.find(
        request.target_key, request.condition, request.product_kind
    ) == record


def test_atlas_plan_refines_the_least_developed_glyph_first():
    catalog = RenderAssetCatalog()
    initial = plan_ink_atlas_bake(("AB",), catalog)
    glyphs = {
        item.token_asset.token: item
        for item in initial.requests
        if item.target_kind is BakeTargetKind.ATLAS_GLYPH
    }
    for token, refinement_pass in (("A", 5), ("B", 2)):
        request = glyphs[token]
        catalog.complete(
            request,
            samples=refinement_pass,
            metadata={
                "bounded_background_render": True,
                "sensor_display_orientation": SENSOR_DISPLAY_ORIENTATION,
                "refinement_state": "developing",
                "refinement_pass": refinement_pass,
                "atlas_quality": {
                    "composable": False,
                    "converged": False,
                },
            },
        )

    next_request = plan_ink_atlas_bake(("AB",), catalog).next_request
    assert next_request is not None
    assert next_request.token_asset.token == "B"
    assert next_request.refinement_pass == 2
