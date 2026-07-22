import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

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
    render_style_key,
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
    fit = token["single_shot_fit"]
    assert fit["policy"] == "uniform_shrink_to_fit"
    assert fit["single_shot"] is True
    assert fit["panoramic"] is False
    assert fit["collage"] is False
    assert 0.0 < fit["scale"] <= 1.0
    assert token["geometry"]["height_m"] == pytest.approx(
        0.13 * fit["scale"]
    )
    assert token["geometry"]["depth_m"] == pytest.approx(
        0.028 * fit["scale"]
    )


def test_whole_word_is_uniformly_shrunk_inside_camera_aspect_frame():
    import scene_orders

    package = build_ink_on_slate_order(("actually",))
    (job,) = scene_orders.resolved_jobs(package)
    fit = job["single_shot_fit"]

    fitted_width = fit["outline_aspect"] * job["geometry"]["height_m"]
    assert fit["scale"] < 1.0
    assert fitted_width <= fit["frame_size_m"][0] + 1.0e-12
    assert job["geometry"]["height_m"] <= fit["frame_size_m"][1]
    assert job["planes"][0]["size_m"] == [0.52, 0.42]
    assert job["geometry"]["depth_m"] / fit["original_depth_m"] == pytest.approx(
        fit["scale"]
    )


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


def test_atlas_plan_exposes_the_same_normalized_need_used_for_dispatch():
    catalog = RenderAssetCatalog()
    plan = plan_ink_atlas_bake(("ABC",), catalog)

    assert plan.next_request is not None
    assert plan.next_request.priority_need == max(
        request.priority_need for request in plan.requests
    )
    assert sum(plan.normalized_priorities.values()) == pytest.approx(1.0)
    assert plan.normalized_priorities[plan.next_request.request_key] == pytest.approx(
        plan.next_request.priority_need
        / sum(request.priority_need for request in plan.requests)
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
            out_dir = command[command.index("--out-dir") + 1]
            progress_dir = Path(
                command[command.index("--progress-dir") + 1]
            )
            progress_dir.mkdir(parents=True, exist_ok=True)
            progress_path = progress_dir / "active.npy"
            progress_sum_path = progress_dir / "active_sum.npy"
            progress_weight_path = progress_dir / "active_weight.npy"
            np.save(progress_path, np.ones((128, 128, 3), np.float32))
            np.save(progress_sum_path, np.ones((128, 128, 3), np.float32))
            np.save(progress_weight_path, np.ones((128, 128), np.float32))
            self.stdout = []
            for index in range(1, epochs + 1):
                progress = demo.ExposureProgressEvent(
                    exposure_id=command[
                        command.index("--progress-exposure-id") + 1
                    ],
                    sequence=index,
                    kind=demo.ExposureProgressKind.LAYER_AVAILABLE,
                    region=demo.SensorRegion(0, 0, 128, 128),
                    pass_index=index,
                    completed_work=index,
                    total_work=epochs,
                    linear_accumulation_path=str(progress_path),
                    sensor_sum_path=str(progress_sum_path),
                    exposure_weight_path=str(progress_weight_path),
                )
                self.stdout.extend([
                    f"  [sensor-refine] epoch {index}/{epochs}\n",
                    progress.to_line() + "\n",
                ])
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
    assert "--progress-dir" in first_command
    assert "--progress-exposure-id" in first_command
    assert first_environment["SPECTRAL_PROGRESS_RETAIN_LAYERS"] == "2"
    assert first_environment["SPECTRAL_SENSOR_MAX_EPOCHS"] == "1"
    assert first_environment["SPECTRAL_SENSOR_PERSISTENT_EPOCHS"] == "1"
    assert first_environment["SPECTRAL_SENSOR_TOP_K"] == "64"
    assert first_environment["SPECTRAL_SENSOR_STEPS_PER_LAYER"] == "1"
    assert first_environment["SPECTRAL_SENSOR_SAMPLES_PER_NODE"] == "64"
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
    assert "convergence_metric" in record.metadata["atlas_quality"]
    assert (
        "convergence_velocity_per_pass"
        in record.metadata["atlas_quality"]
    )
    assert record.metadata["atlas_quality"]["converged"] is True
    assert record.samples == 4
    assert [item.linear_path for item in records] == [record.linear_path] * 4
    assert record.metadata["sprite_path"].endswith("raytraced_sprite.npz")
    assert record.metadata["capture"]["content_region"] == [24, 24, 80, 80]
    work_revision, work_path, work_token, work_pass, active = (
        renderer.work_snapshot()
    )
    assert work_revision >= 12
    assert work_path == record.preview_path
    assert work_token == "A"
    assert work_pass == 4
    assert not active
    (
        object_revision,
        work_object_key,
        object_token,
        object_pass,
        object_active,
    ) = renderer.work_object_snapshot()
    assert object_revision == work_revision
    style_key = render_style_key(request.token_asset)
    assert work_object_key == style_key
    assert object_token == "A"
    assert object_pass == 4
    assert not object_active
    bundle = renderer.object_library.find(style_key)
    assert bundle is not None
    subtype = renderer.object_library.find_subtype(style_key, "A", kind="glyph")
    assert subtype is not None
    assert subtype.scenes[0].renderer_context_path.endswith(
        "render_context.json"
    )
    assert Path(bundle.manifest_path).is_file()
    assert catalog.find(
        request.target_key, request.condition, request.product_kind
    ) == record


def test_atlas_executor_resumes_from_each_published_epoch(tmp_path, monkeypatch):
    catalog = RenderAssetCatalog(str(tmp_path / "catalog.json"))
    request = plan_ink_atlas_bake(("A",), catalog).next_request
    assert request is not None
    launches = []

    class Process:
        def __init__(self, command, **kwargs):
            environment = dict(kwargs["env"])
            launches.append((command, environment))
            attempt = len(launches)
            epoch_goal = int(environment["SPECTRAL_SENSOR_MAX_EPOCHS"])
            completed = 3 if attempt == 1 else epoch_goal
            out_dir = Path(command[command.index("--out-dir") + 1])
            progress_dir = Path(command[command.index("--progress-dir") + 1])
            progress_dir.mkdir(parents=True, exist_ok=True)
            linear = progress_dir / f"active_{attempt}.npy"
            sensor_sum = progress_dir / f"sum_{attempt}.npy"
            weight = progress_dir / f"weight_{attempt}.npy"
            np.save(linear, np.ones((128, 128, 3), np.float32))
            np.save(sensor_sum, np.full((128, 128, 3), attempt, np.float32))
            np.save(weight, np.full((128, 128), attempt, np.float32))
            event = demo.ExposureProgressEvent(
                exposure_id=command[
                    command.index("--progress-exposure-id") + 1
                ],
                sequence=1,
                kind=demo.ExposureProgressKind.LAYER_AVAILABLE,
                region=demo.SensorRegion(0, 0, 128, 128),
                pass_index=completed,
                completed_work=completed,
                total_work=epoch_goal,
                linear_accumulation_path=str(linear),
                sensor_sum_path=str(sensor_sum),
                exposure_weight_path=str(weight),
            )
            self.stdout = [
                f"  [sensor-refine] epoch {index}/{epoch_goal}\n"
                for index in range(1, completed + 1)
            ] + [event.to_line() + "\n"]
            self.return_code = 1 if attempt == 1 else 0
            if self.return_code == 0:
                np.save(
                    out_dir / "0000_cpp_linear.npy",
                    np.ones((128, 128, 3), np.float32),
                )
                np.save(
                    out_dir / "0000_cpp_sum_linear.npy",
                    np.ones((128, 128, 3), np.float32),
                )
                np.save(
                    out_dir / "0000_cpp_exposure_weight.npy",
                    np.ones((128, 128), np.float32),
                )

        def wait(self):
            return self.return_code

        def poll(self):
            return self.return_code

        def terminate(self):
            pass

    monkeypatch.setattr(demo.subprocess, "Popen", Process)
    renderer = demo.InkAtlasSubprocessRenderer(
        str(tmp_path), catalog, steps_per_epoch=64
    )

    with pytest.raises(demo.subprocess.CalledProcessError):
        renderer(request)

    work_revision, work_path, work_token, work_pass, active = (
        renderer.work_snapshot()
    )
    assert work_revision >= 3
    assert work_path.endswith("active_1.npy")
    assert work_token == "A"
    assert work_pass == 3
    assert not active

    request_dir = Path(launches[0][0][launches[0][0].index("--out-dir") + 1])
    checkpoint_path = request_dir / "active_epoch_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["completed_epochs"] == 3
    assert checkpoint["target_refinement_pass"] == 64

    record = renderer(request)

    second_environment = launches[1][1]
    assert second_environment["SPECTRAL_SENSOR_MAX_EPOCHS"] == "61"
    assert second_environment["SPECTRAL_SENSOR_STEPS_PER_LAYER"] == "1"
    assert second_environment["SPECTRAL_SENSOR_RESTORE_SUM"].endswith(
        "restore_sensor_sum_native.npy"
    )
    assert second_environment["SPECTRAL_SENSOR_RESTORE_WEIGHT"].endswith(
        "restore_sensor_weight_native.npy"
    )
    assert record.samples == 64
    assert not checkpoint_path.exists()


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


def test_one_completed_packet_replans_to_an_unresolved_peer():
    catalog = RenderAssetCatalog()
    initial = plan_ink_atlas_bake(("AB",), catalog)
    first = initial.next_request
    assert first is not None
    assert first.token_asset.token == "A"
    catalog.complete(
        first,
        samples=1,
        metadata={
            "bounded_background_render": True,
            "sensor_display_orientation": SENSOR_DISPLAY_ORIENTATION,
            "refinement_state": "developing",
            "refinement_pass": 1,
            "atlas_quality": {
                "composable": False,
                "converged": False,
            },
        },
    )

    replanned = plan_ink_atlas_bake(("AB",), catalog).next_request

    assert replanned is not None
    assert replanned.token_asset.token == "B"
    assert replanned.refinement_pass == 0


def test_atlas_plan_can_resume_a_converged_asset_to_a_requested_pass():
    catalog = RenderAssetCatalog()
    request = plan_ink_atlas_bake(("A",), catalog).next_request
    assert request is not None
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

    assert plan_ink_atlas_bake(("A",), catalog).next_request is None
    resumed = plan_ink_atlas_bake(
        ("A",),
        catalog,
        refinement_targets={request.target_key: 6},
    ).next_request

    assert resumed is not None
    assert resumed.target_key == request.target_key
    assert resumed.refinement_pass == 4


def test_human_visual_pass_finishes_queue_but_allows_later_requested_passes(tmp_path):
    catalog = RenderAssetCatalog(str(tmp_path / "catalog.json"))
    plan = plan_ink_atlas_bake(("A",), catalog)
    request = plan.next_request
    assert request is not None
    linear_path = tmp_path / "A_linear.npy"
    np.save(linear_path, np.ones((4, 4, 3), np.float32))
    record = catalog.complete(
        request,
        linear_path=str(linear_path),
        samples=3,
        metadata={
            "bounded_background_render": True,
            "sensor_display_orientation": SENSOR_DISPLAY_ORIENTATION,
            "refinement_state": "developing",
            "refinement_pass": 3,
            "atlas_quality": {
                "composable": True,
                "converged": False,
                "glyph_exposure_coverage": 1.0,
                "glyph_radiance_coverage": 1.0,
            },
        },
    )
    assert plan_ink_atlas_bake(("A",), catalog).next_request is not None

    accepted = catalog.accept_visual_pass(record)

    assert accepted.metadata["completion_basis"] == "human_visual_pass"
    assert plan_ink_atlas_bake(("A",), catalog).requests == ()
    later = plan_ink_atlas_bake(
        ("A",),
        catalog,
        refinement_targets={request.token_asset.asset_key: 4},
    )
    assert later.next_request is not None
    assert later.next_request.refinement_pass == 3
