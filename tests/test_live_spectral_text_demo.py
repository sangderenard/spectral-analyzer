import copy
import json
import threading
import time
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import live_spectral_text_demo as demo
import scene_orders as orders
from camera_software.progressive_exposure import (
    ExposureProgressEvent,
    ExposureProgressKind,
    SensorPixelSlice,
    SensorRegion,
)


def _resolved(text: str) -> dict:
    package = demo.build_paragraph_order(text, display_width=160, display_height=96)
    return orders.resolved_jobs(package, demo.JOB_ID)[0]


def test_live_program_defaults_to_four_complete_sensor_sweeps():
    args = demo._args([])
    assert args.sensor_sweeps == 4
    package = demo.build_paragraph_order("default sweep count")
    job = orders.resolved_jobs(package, demo.JOB_ID)[0]
    assert orders.order_runtime_settings(job)["sensor_sweeps"] == 4


def test_live_work_product_defaults_to_256_square_without_reflowing_ui():
    args = demo._args([])
    assert (args.work_width, args.work_height) == (256, 256)

    manifest = demo.program_ui_manifest()
    layout_reference = demo._layout_reference_size(manifest)
    assert layout_reference == (960, 600)
    assert demo._layout_reference_size(manifest) == layout_reference
    assert demo._layout_work_viewport_size(manifest) == (256, 256)


def test_calibration_whole_sensor_is_fifo_partitioned_into_256_tiles():
    order = demo.build_calibration_render_order(
        "prism-room", display_width=256, display_height=256
    )
    image = order["defaults"]["image"]
    tiles = demo._sensor_work_tiles(
        image["region"],
        order["runtime"]["sensor_work_tile_width"],
        order["runtime"]["sensor_work_tile_height"],
    )

    assert image == {
        "width": 1024,
        "height": 1024,
        "region": {"x": 0, "y": 0, "width": 1024, "height": 1024},
    }
    assert len(tiles) == 16
    assert [(tile["x"], tile["y"]) for tile in tiles] == [
        (x, y)
        for y in (0, 256, 512, 768)
        for x in (0, 256, 512, 768)
    ]
    assert all(
        (tile["width"], tile["height"]) == (256, 256) for tile in tiles
    )
    physical = []
    for tile in (tiles[0], tiles[-1]):
        child = copy.deepcopy(order)
        child["defaults"]["image"]["region"] = tile
        job = orders.resolved_jobs(child, demo.JOB_ID)[0]
        physical.append(orders.sensor_tile(job, 0.056, 0.056))
    assert np.isclose(physical[0]["sensor_w_m"], 0.014)
    assert np.isclose(physical[0]["sensor_h_m"], 0.014)
    assert np.allclose(
        [physical[0]["right_offset_m"], physical[0]["up_offset_m"]],
        [-0.021, 0.021],
    )
    assert np.allclose(
        [physical[1]["right_offset_m"], physical[1]["up_offset_m"]],
        [0.021, -0.021],
    )


def test_rectangular_sensor_tile_composes_y_rows_then_x_columns():
    full_linear = np.zeros((6, 11, 3), np.float32)
    full_sum = np.zeros_like(full_linear)
    full_weight = np.zeros((6, 11), np.float32)
    pattern = np.asarray([[1, 2, 3], [4, 5, 6]], np.float32)
    tile_linear = np.repeat(pattern[..., None], 3, axis=2)
    tile_sum = tile_linear * 10.0
    tile_weight = pattern * 100.0

    demo._compose_sensor_work_tile(
        full_linear, full_sum, full_weight,
        tile_linear, tile_sum, tile_weight,
        {"x": 4, "y": 1, "width": 3, "height": 2},
    )

    assert np.array_equal(full_linear[1:3, 4:7, 0], pattern)
    assert np.array_equal(full_sum[1:3, 4:7, 0], pattern * 10.0)
    assert np.array_equal(full_weight[1:3, 4:7], pattern * 100.0)
    assert np.count_nonzero(full_weight) == 6


def test_fixed_ui_layout_keeps_right_panel_and_square_preview_when_resized():
    manifest = demo.program_ui_manifest()
    scene_width, scene_height = demo._layout_reference_size(manifest)
    work_width, work_height = demo._layout_work_viewport_size(manifest)
    photographed = demo.program_display_region(scene_width, scene_height)
    layout = demo.resolved_program_ui_layout(
        scene_width,
        scene_height,
        work_width=work_width,
        work_height=work_height,
        manifest=manifest,
    )
    work_rect = demo._sensor_product_window_rect(
        demo.SensorRegion(*layout.regions["work-panel"]),
        photographed,
        scene_width,
        scene_height,
    )
    browser_rect = demo._sensor_product_window_rect(
        demo.SensorRegion(*layout.regions["asset-browser"]),
        photographed,
        scene_width,
        scene_height,
    )

    assert work_rect[2:] == (256, 256)
    assert browser_rect[2] > 0
    assert browser_rect[0] + browser_rect[2] == scene_width
    previous_height = 0
    for physical in ((960, 600), (1200, 700), (1200, 900)):
        presentation = demo._presentation_rect(
            scene_width, scene_height, *physical
        )
        displayed_work = demo._presentation_subrect(
            work_rect, presentation, scene_width, scene_height
        )
        assert abs(displayed_work[2] - displayed_work[3]) <= 1
        assert presentation[3] >= previous_height
        previous_height = presentation[3]


def test_live_work_product_accepts_free_size_and_legacy_cli_aliases():
    current = demo._args(["--work-width", "317", "--work-height", "149"])
    legacy = demo._args(["--display-width", "83", "--display-height", "271"])

    assert (current.work_width, current.work_height) == (317, 149)
    assert (legacy.work_width, legacy.work_height) == (83, 271)


def test_live_final_raster_is_adjustable_but_remains_square_and_reports_ui_crop():
    scene = demo.build_self_rendering_program_scene(
        "adjustable square gate",
        display_width=960,
        display_height=600,
        sensor_edge_px=1200,
    )
    job = orders.resolved_jobs(
        demo.build_display_scene_order(scene), demo.JOB_ID
    )[0]
    runtime = orders.order_runtime_settings(job)

    assert (scene.sensor_width, scene.sensor_height) == (1200, 1200)
    assert runtime["region"] == {
        "x": 0, "y": 0, "width": 1200, "height": 1200,
    }
    assert runtime["camera_manifest"]["sensor"]["ui_content_region_px"] == {
        "x": 120, "y": 300, "width": 960, "height": 600,
    }
    assert runtime["camera_manifest"]["sensor"][
        "raster_pixel_aspect_ratio"
    ] == 1.0


def test_locked_grid_partitions_global_uv_without_gaps_or_overlaps():
    scene = demo.build_program_display_scene(
        "locked grid", display_width=120, display_height=80
    )
    control = demo.build_ui_next_scan_control(
        scene,
        demo.program_display_region(120, 80),
        sequence=1,
        targeted_fraction=0.0,
        grid_mode="locked",
        locked_grid_columns=5,
        locked_grid_rows=3,
    )

    requests = control["uv_requests"]
    assert len(requests) == 15
    assert sum(
        (request["uv_bounds"][2] - request["uv_bounds"][0])
        * (request["uv_bounds"][3] - request["uv_bounds"][1])
        for request in requests
    ) == pytest.approx(1.0)
    assert requests[0]["uv_bounds"] == [0.0, 0.0, 0.2, 1 / 3]
    assert requests[-1]["uv_bounds"] == [0.8, 2 / 3, 1.0, 1.0]
    assert control["metadata"]["sensor_grid"]["children_per_split"] == 9


def test_atlas_preview_and_convergence_checkpoint_cadence_is_explicit():
    cadence = demo.atlas_update_cadence()

    assert cadence == {
        "preview_update_passes": 1,
        "convergence_update_passes": 1,
        "primary_rays_per_pass": 4_096,
        "primary_rays_per_convergence_update": 4_096,
    }
    custom = demo.atlas_update_cadence(
        epochs_per_exposure=2,
        steps_per_epoch=8,
        sensor_top_k=128,
        samples_per_node=256,
    )
    assert custom["preview_update_passes"] == 1
    assert custom["convergence_update_passes"] == 16
    assert custom["primary_rays_per_convergence_update"] == 524_288


def test_live_production_starts_with_fixed_width_alphabet_cells():
    catalog = demo.RenderAssetCatalog()
    plan = demo._ink_production_plan("Actual light", catalog)

    assert len(plan.requests) >= len(demo.FIXED_IMAGE_ALPHABET)
    assert {
        request.target_kind for request in plan.requests
    } == {demo.BakeTargetKind.ATLAS_GLYPH}
    assert {
        request.token_asset.font.family for request in plan.requests
    } == {"DejaVu Sans Mono"}
    assert not demo._total_scene_is_ready("Actual light", catalog)
    assert not demo._fixed_image_alphabet_is_ready(catalog)


def test_custom_manifest_pipeline_drives_font_glyphs_tokens_and_scene_text():
    manifest = demo.program_ui_manifest()
    manifest.payload["render_pipeline"] = {
        **manifest.payload["render_pipeline"],
        "font": {
            "mode": "monofont",
            "family": "DejaVu Sans Mono",
            "weight": "normal",
            "style": "normal",
        },
        "glyph_alphabet": "AZ",
        "static_tokens": ["CAMERA", "WORK"],
    }
    catalog = demo.RenderAssetCatalog()
    plan = demo._ink_production_plan("ZAP", catalog, manifest)
    scene = demo.build_self_rendering_program_scene(
        "ZAP", display_width=320, display_height=180,
        work_width=80, work_height=80, program_manifest=manifest,
    )

    planned_glyphs = {request.token_asset.token for request in plan.requests}
    assert {"A", "Z"} <= planned_glyphs
    assert {
        request.target_kind for request in plan.requests
    } == {demo.BakeTargetKind.ATLAS_GLYPH}
    assert {
        request.token_asset.font.weight for request in plan.requests
    } == {"normal"}
    text_objects = [
        item for item in scene.objects
        if item.primitive.kind is demo.DisplayPrimitiveKind.TEXT
    ]
    assert {item.font_weight for item in text_objects} == {"normal"}


def test_live_cli_accepts_reusable_raytrained_priority_model():
    args = demo._args(["--priority-model", "learned.npz"])
    assert args.priority_model == "learned.npz"


def test_live_cli_exposes_scoped_camera_clear_modes():
    clear_then_start = demo._args(["--clear-camera"])
    clear_only = demo._args(["--clear-camera-only"])

    assert clear_then_start.clear_camera
    assert not clear_then_start.clear_camera_only
    assert clear_only.clear_camera_only


def test_live_cli_defaults_to_one_bounded_interleaving_packet():
    args = demo._args([])

    assert args.atlas_epochs_per_exposure == 1
    assert args.atlas_sensor_top_k == 64
    assert args.atlas_steps_per_epoch == 1
    assert args.atlas_samples_per_node == 64
    assert args.atlas_character_horizontal_spacing_px == -10
    assert args.atlas_character_vertical_spacing_px == -10
    assert (
        args.atlas_sensor_top_k
        * args.atlas_steps_per_epoch
        * args.atlas_samples_per_node
    ) == 4_096


def test_atlas_cache_uses_style_subtype_condition_object_layout(tmp_path):
    from camera_software import (
        BakeRequest,
        BakeTargetKind,
        DEFAULT_INK_CONDITION,
        DisplayProductKind,
        ink_token_asset,
        render_style_key,
    )

    glyph = ink_token_asset("A")
    default_request = BakeRequest(
        glyph.asset_key,
        BakeTargetKind.ATLAS_GLYPH,
        DEFAULT_INK_CONDITION,
        DisplayProductKind.IMAGE,
        "atlas",
        ("A",),
        glyph,
    )
    other_condition = replace(DEFAULT_INK_CONDITION, azimuth_deg=90.0)
    other_request = replace(default_request, condition=other_condition)

    default_path = Path(demo._atlas_request_directory(str(tmp_path), default_request))
    other_path = Path(demo._atlas_request_directory(str(tmp_path), other_request))

    assert "render_objects" in default_path.parts
    assert render_style_key(glyph).rsplit(":", 1)[-1] in default_path.parts
    assert "subtypes" in default_path.parts
    assert "glyph" in default_path.parts
    assert default_path.parent == other_path.parent
    assert default_path.name == DEFAULT_INK_CONDITION.condition_key.rsplit(":", 1)[-1]
    assert other_path.name == other_condition.condition_key.rsplit(":", 1)[-1]
    assert other_path != default_path


def test_clear_camera_storage_removes_only_live_camera_artifacts(tmp_path):
    root = tmp_path / "live"
    (root / "render_objects" / "style").mkdir(parents=True)
    (root / "render_interfaces" / "assembly").mkdir(parents=True)
    (root / "render_asset_catalog.json").write_text("{}", encoding="utf-8")
    (root / "render_object_library.json").write_text("{}", encoding="utf-8")
    (root / "revision_0001" / "progress").mkdir(parents=True)
    (root / "revision_42").mkdir()
    (root / "display_inventory.json").write_text("{}", encoding="utf-8")
    (root / "display_inventory.json.tmp").write_text("{}", encoding="utf-8")
    (root / "notes.txt").write_text("keep", encoding="utf-8")
    (root / "revision_draft").mkdir()

    removed = demo.clear_camera_storage(str(root))

    assert len(removed) == 8
    assert not (root / "render_objects").exists()
    assert not (root / "render_interfaces").exists()
    assert not (root / "render_asset_catalog.json").exists()
    assert not (root / "render_object_library.json").exists()
    assert not (root / "revision_0001").exists()
    assert not (root / "revision_42").exists()
    assert not (root / "display_inventory.json").exists()
    assert (root / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert (root / "revision_draft").is_dir()


def test_repeated_order_finds_latest_retained_accumulation(tmp_path):
    order = demo.build_paragraph_order(
        "repeat me", display_width=20, display_height=12
    )
    for sequence in (1, 3):
        revision = tmp_path / f"revision_{sequence:04d}"
        job = revision / demo.JOB_ID
        job.mkdir(parents=True)
        prior = copy.deepcopy(order)
        prior["cohort_seed"] = sequence
        (revision / "scene_order.json").write_text(
            json.dumps(prior), encoding="utf-8"
        )
        sum_path = job / "0000_cpp_sum_linear.npy"
        np.save(sum_path, np.full((12, 20, 3), sequence))
        np.save(job / "0000_cpp_exposure_weight.npy", np.full((12, 20), sequence))
        demo._mark_sensor_display_orientation(str(sum_path))

    current = copy.deepcopy(order)
    current["cohort_seed"] = 99
    found = demo._find_repeat_accumulation(
        str(tmp_path), current, before_sequence=4
    )
    assert found is not None
    assert "revision_0003" in found[0]
    assert demo._latest_revision_sequence(str(tmp_path)) == 3


def test_repeat_signature_ignores_sampling_budget_and_continuous_cohort():
    budget = {
        "total_rays": 1_024,
        "rays_per_batch": 256,
        "max_sensor_epochs": 4,
        "sensor_top_k": 20,
        "sensor_samples_per_node": 13,
        "sensor_steps_per_layer": 1,
        "sensor_flash_rays": 256,
        "sensor_t5_pair_budget": 32_768,
        "max_bounces": 8,
    }
    first = demo.build_calibration_render_order(
        "prism-room",
        display_width=20,
        display_height=12,
        cohort_seed=1,
        transport_option="continuous:8",
        render_budget=budget,
    )
    larger = demo.build_calibration_render_order(
        "prism-room",
        display_width=20,
        display_height=12,
        cohort_seed=2,
        transport_option="continuous:8",
        render_budget={**budget, "total_rays": 4_096, "rays_per_batch": 1_024},
    )

    assert demo._repeat_order_signature(first) == demo._repeat_order_signature(larger)


def test_epoch_bundles_restore_each_completed_bundle(tmp_path, monkeypatch):
    launches = []

    class Process:
        def __init__(self, command, **kwargs):
            environment = dict(kwargs["env"])
            launches.append(environment)
            out_dir = Path(command[command.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            if "SPECTRAL_SENSOR_RESTORE_SUM" in environment:
                sensor_sum = np.load(
                    environment["SPECTRAL_SENSOR_RESTORE_SUM"],
                    allow_pickle=False,
                )
                weight = np.load(
                    environment["SPECTRAL_SENSOR_RESTORE_WEIGHT"],
                    allow_pickle=False,
                )
                # Native restore storage is square; the real subprocess maps
                # it back to this tile's exact display raster on readback.
                sensor_sum = np.full(
                    (12, 20, 3), float(sensor_sum.reshape(-1)[0]), np.float32
                )
                weight = np.full(
                    (12, 20), float(weight.reshape(-1)[0]), np.float32
                )
            else:
                sensor_sum = np.zeros((12, 20, 3), np.float32)
                weight = np.zeros((12, 20), np.float32)
            sensor_sum = sensor_sum + 1.0
            weight = weight + 1.0
            np.save(out_dir / "0000_cpp_sum_linear.npy", sensor_sum)
            np.save(out_dir / "0000_cpp_exposure_weight.npy", weight)
            np.save(
                out_dir / "0000_cpp_linear.npy",
                sensor_sum / weight[..., None],
            )
            event = demo.ExposureProgressEvent(
                exposure_id="revision-0001",
                sequence=1,
                kind=demo.ExposureProgressKind.PASS_AVAILABLE,
                region=demo.SensorRegion(0, 0, 20, 12),
                pass_index=1,
                linear_accumulation_path=str(
                    out_dir / "0000_cpp_linear.npy"
                ),
                sensor_sum_path=str(out_dir / "0000_cpp_sum_linear.npy"),
                exposure_weight_path=str(
                    out_dir / "0000_cpp_exposure_weight.npy"
                ),
            )
            self.stdout = [event.to_line() + "\n"]

        def wait(self):
            return 0

        def poll(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr(demo.subprocess, "Popen", Process)
    order = demo.build_calibration_render_order(
        "prism-room", display_width=20, display_height=12
    )
    order["defaults"]["image"] = {
        "width": 20,
        "height": 12,
        "region": {"x": 0, "y": 0, "width": 20, "height": 12},
    }
    order["runtime"]["epoch_bundle_count"] = 2
    progress = []
    renderer = demo.make_subprocess_renderer(
        str(tmp_path), demo.ColorScienceProfile.identity()
    )

    renderer(1, "bundle test", order, None, progress.append)

    assert len(launches) == 2
    assert "SPECTRAL_SENSOR_RESTORE_SUM" not in launches[0]
    assert "SPECTRAL_SENSOR_RESTORE_SUM" in launches[1]
    assert launches[0]["SPECTRAL_EXPOSURE_SEED_OFFSET"] == str(
        demo.iching_coin_seed(1, 0)
    )
    assert launches[1]["SPECTRAL_EXPOSURE_SEED_OFFSET"] == str(
        demo.iching_coin_seed(1, 1)
    )
    assert (
        launches[0]["SPECTRAL_EXPOSURE_SEED_OFFSET"]
        != launches[1]["SPECTRAL_EXPOSURE_SEED_OFFSET"]
    )
    final_sum = np.load(
        tmp_path / "revision_0001" / demo.JOB_ID
        / "0000_cpp_sum_linear.npy",
        allow_pickle=False,
    )
    assert np.all(final_sum == 2.0)
    assert [event.sequence for event in progress] == list(range(7))
    eta_events = [event for event in progress if event.total_work]
    assert eta_events[-1].progress_fraction == 1.0
    assert "overall 100.00%" in eta_events[-1].message
    assert eta_events[-1].linear_accumulation_path.endswith(
        "0000_cpp_linear.npy"
    )


def test_i_ching_coin_seed_fills_reproducible_native_words():
    seeds = [demo.iching_coin_seed(19, bundle) for bundle in range(256)]

    assert seeds == [
        demo.iching_coin_seed(19, bundle) for bundle in range(256)
    ]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= seed < (1 << demo.NATIVE_SEED_BITS) for seed in seeds)
    assert demo.iching_coin_seed(20, 0) != seeds[0]


def test_toolbar_depth_mode_selects_depth_product_without_replacing_transport():
    order = demo.build_calibration_render_order(
        "prism-room", transport_option="continuous:1"
    )
    demo.apply_toolbar_render_mode(order, "depth")

    assert order["transport"]["domain"] == "continuous_spectral_lut"
    assert order["runtime"]["integrator"] == "depth"
    assert order["runtime"]["render_product"] == "sensor_optical_path_depth_m"
    assert order["runtime"]["convergence_enabled"] is False


def test_live_renderer_never_trains_from_orthographic_preview():
    source = inspect.getsource(demo.make_subprocess_renderer)
    assert "train_scene_priority_network" not in source
    assert "SPECTRAL_PRIORITY_TRAINING_STEPS" not in source


def test_display_raster_and_text_face_focus_are_explicit():
    job = _resolved("intricate spectral letterforms")
    runtime = orders.order_runtime_settings(job)
    metadata = orders.composition_metadata(job)

    assert (runtime["width"], runtime["height"]) == (160, 96)
    assert runtime["region"] == {"x": 432, "y": 464, "width": 160, "height": 96}
    assert metadata["full_frame"] == {"width": 1024, "height": 1024}
    assert metadata["film_format"] == {
        "key": "120_6x6",
        "mount_standard": "120_6x6",
        "physical_width_mm": 56.0,
        "physical_height_mm": 56.0,
        "physical_aspect_ratio": 1.0,
    }
    assert np.isclose(metadata["physical_sensor_tile"]["sensor_w_m"], 0.00875)
    assert np.isclose(metadata["physical_sensor_tile"]["sensor_h_m"], 0.00525)
    expected = (
        np.asarray(job["planes"][0]["normal"])
        * orders.resolved_glyph_depth(job)
        * (1.0 - float(job["geometry"]["embed_fraction"]))
    )
    assert np.allclose(job["camera"]["focus_target_m"], expected, atol=1.0e-12)
    assert np.isclose(
        metadata["camera"]["focus_distance_m"],
        orders.camera_focus_distance(job),
    )
    assert metadata["camera"]["focus_distance_m"] < np.linalg.norm(
        np.asarray(job["camera"]["target_m"]) - np.asarray(job["camera"]["position_m"])
    )


def test_live_font_depth_tracks_formatter_resolved_height_and_mesh_is_detailed():
    short = _resolved("Hi")
    long = _resolved(
        "A substantially longer paragraph forces the formatter to reduce its "
        "physical line height while preserving the authored extrusion ratio."
    )

    ratio = demo.DEFAULT_EXTRUSION_DEPTH_RATIO
    for job in (short, long):
        assert np.isclose(
            orders.resolved_glyph_depth(job) / orders.resolved_glyph_height(job),
            ratio,
        )
        assert job["geometry"]["outline_subdivisions"] == 4
        assert job["geometry"]["cap_grid"] == 64
    assert orders.resolved_glyph_height(long) < orders.resolved_glyph_height(short)
    assert orders.resolved_glyph_depth(long) < orders.resolved_glyph_depth(short)


def test_render_contract_reports_native_raster_and_coordinate_only_frame():
    summary = demo.render_contract_summary(
        120, 100, sensor_sweeps=4, scene_width=960, scene_height=600
    )
    assert "ui_scene=960x600" in summary
    assert "scan_region<=120x100" in summary
    assert "film=120_6x6" in summary
    assert "gate=56x56mm" in summary
    assert "composition_frame=1024x1024" in summary
    assert "authored_sensor_sweeps=4" in summary
    assert "foreground_epochs<=64" in summary
    assert "production=alphabet->tokens->token-string->total-scene" in summary


def test_progress_preview_uses_presentation_only_auto_exposure_source():
    source = Path(demo.__file__).read_text(encoding="utf-8")
    assert "preview_white" in source
    assert "linear_progress[..., :3]" in source
    assert "np.save" not in source[source.index("preview_white") - 500:source.index("preview_white") + 500]


def test_live_order_exposes_independent_sensor_refinement_sweeps():
    package = demo.build_paragraph_order(
        "refine",
        display_width=120,
        display_height=100,
        sensor_sweeps=4,
    )
    job = orders.resolved_jobs(package, demo.JOB_ID)[0]
    assert orders.order_runtime_settings(job)["sensor_sweeps"] == 4


def test_self_rendering_scene_places_editor_and_controls_on_distinct_sensor_regions():
    scene = demo.build_self_rendering_program_scene(
        "editable words", display_width=200, display_height=120, revision=3
    )
    expected = demo.program_ui_sensor_regions(200, 120)

    assert {item.object_id for item in scene.objects} == set(expected)
    assert len({item.placement.center_m for item in scene.objects}) == len(scene.objects)
    for item in scene.objects:
        assert item.products[0].sensor_region == expected[item.object_id]
        assert item.revision == (
            3 if item.object_id in {"editor-text", "status-text"}
            else demo.PROGRAM_STATIC_GEOMETRY_REVISION
        )
    # Perspective is shared: no object contract contains a camera.
    assert all(not hasattr(item, "camera") for item in scene.objects)


def test_static_geometry_change_invalidates_retained_sensor_evidence():
    previous = demo.build_self_rendering_program_scene(
        "old editor", display_width=200, display_height=120, revision=1
    )
    current = demo.build_self_rendering_program_scene(
        "new editor", display_width=200, display_height=120, revision=2
    )
    assert demo._static_scene_is_reusable(previous, current)

    previous_by_id = {item.object_id: item for item in previous.objects}
    stale_panel = replace(
        previous_by_id["camera-panel"],
        primitive=demo.DisplayPrimitive(
            demo.DisplayPrimitiveKind.BOX, label="CAMERA"
        ),
        revision=1,
    )
    stale = replace(
        previous,
        objects=tuple(
            stale_panel if item.object_id == "camera-panel" else item
            for item in previous.objects
        ),
    )
    assert not demo._static_scene_is_reusable(stale, current)


def test_retained_sensor_evidence_requires_matching_orientation_marker(tmp_path):
    sensor_sum = tmp_path / "sum.npy"
    sensor_sum.write_bytes(b"kept evidence")

    assert not demo._sensor_sum_has_current_orientation(str(sensor_sum))

    demo._mark_sensor_display_orientation(str(sensor_sum))
    assert demo._sensor_sum_has_current_orientation(str(sensor_sum))

    marker = Path(demo._sensor_orientation_marker_path(str(sensor_sum)))
    marker.write_text("legacy-mirrored", encoding="utf-8")
    assert not demo._sensor_sum_has_current_orientation(str(sensor_sum))


def test_arrival_shimmer_uses_real_delta_and_settles_high_sample_regions():
    previous = np.zeros((2, 2, 3), np.float32)
    current = previous.copy()
    current[0, 0] = (2.0, 0.5, 0.1)
    current[0, 1] = (1.0, 1.0, 1.0)
    counts = np.asarray([[1, 10_000], [1, 1]], np.int32)

    shimmer = demo._arrival_shimmer_rgb(
        previous, current, sample_count=counts
    )

    assert shimmer.shape == (2, 2, 3)
    assert np.all(np.isfinite(shimmer))
    assert float(np.max(shimmer[0, 0])) > float(np.max(shimmer[0, 1]))
    assert np.all(shimmer[1] == 0.0)


def test_arrival_shimmer_marks_negative_revision_as_cool_glint():
    previous = np.ones((1, 1, 3), np.float32)
    current = np.zeros_like(previous)

    shimmer = demo._arrival_shimmer_rgb(previous, current)

    assert shimmer[0, 0, 2] > shimmer[0, 0, 0]


def test_unexposed_whole_work_preview_is_marked_pending_without_touching_samples():
    display = np.full((4, 4, 3), 0.6, np.float32)
    weight = np.zeros((4, 4), np.float32)
    weight[:2, :2] = 1.0

    preview = demo._mark_unexposed_preview(display, weight, checker_size=1)

    assert preview.shape == (4, 4, 4)
    assert np.allclose(preview[:2, :2, :3], 0.6)
    assert np.allclose(preview[:2, :2, 3], 1.0)
    assert np.all(preview[2:, 2:, :3] > 0.0)
    assert not np.allclose(preview[2, 2], preview[2, 3])
    assert 0.0 < preview[2, 2, 3] < 1.0
    assert np.all(display == 0.6)

def test_ui_products_become_gpu_uv_requests_inside_photographed_region():
    scene = demo.build_self_rendering_program_scene(
        "work allocation", display_width=200, display_height=120
    )
    control = demo.build_ui_next_scan_control(
        scene,
        demo.program_display_region(200, 120),
        sequence=7,
        targeted_fraction=0.75,
    )

    assert control["sequence"] == 7
    assert control["targeted_fraction"] == 0.75
    assert len(control["uv_requests"]) == len(scene.objects)
    assert len(control["pixel_slice_requests"]) == 2
    assert set(control["metadata"]["object_ids"]) == {
        item.object_id for item in scene.objects
    }
    for request in control["uv_requests"]:
        u0, v0, u1, v1 = request["uv_bounds"]
        assert 0.0 <= u0 < u1 <= 1.0
        assert 0.0 <= v0 < v1 <= 1.0
        assert request["target_level"] >= 2


def test_ui_regions_are_split_into_bounded_scan_chunks_without_resizing_scene():
    scene = demo.build_self_rendering_program_scene(
        "chunked work", display_width=960, display_height=600
    )
    photographed = demo.program_display_region(960, 600)
    control = demo.build_ui_next_scan_control(
        scene,
        photographed,
        sequence=1,
        targeted_fraction=0.75,
        scan_width=100,
        scan_height=100,
    )

    assert scene.sensor_width == 1024
    assert scene.sensor_height == 1024
    assert len(control["uv_requests"]) > len(scene.objects)
    for request in control["uv_requests"]:
        u0, v0, u1, v1 = request["uv_bounds"]
        assert (u1 - u0) * photographed.width <= 100.0 + 1.0e-9
        assert (v1 - v0) * photographed.height <= 100.0 + 1.0e-9


def test_delta_blend_changes_only_arbitrary_dirty_sites():
    current = np.full((3, 4, 3), 0.8, np.float32)
    current_weight = np.full((3, 4), 2.0, np.float32)
    previous_sum = np.full((3, 4, 3), 0.2, np.float32)
    previous_weight = np.ones((3, 4), np.float32)
    selected = SensorPixelSlice(4, 3, (0, 6, 11))

    blended = demo._blend_delta_exposure(
        current, current_weight, previous_sum, previous_weight, selected
    )

    assert np.allclose(blended, current)
    current_weight[:] = 0.0
    blended = demo._blend_delta_exposure(
        current, current_weight, previous_sum, previous_weight, selected
    )
    assert np.allclose(blended.reshape(-1, 3)[[0, 6, 11]], 0.2)
    assert np.allclose(blended.reshape(-1, 3)[[1, 2, 3]], 0.8)


def test_self_rendering_order_has_one_camera_and_all_ui_text_geometry():
    scene = demo.build_self_rendering_program_scene(
        "same camera", display_width=160, display_height=96
    )
    job = orders.resolved_jobs(
        demo.build_display_scene_order(scene), demo.JOB_ID
    )[0]

    assert [item["id"] for item in job["objects"]] == [
        "editor-text",
        "camera-label",
        "work-label",
        "asset-browser-label",
        "status-text",
        "work-visual-pass",
        "queue-pause-auto",
    ]
    assert {
        item["id"]: item["token"] for item in job["objects"]
    } == {
        "editor-text": "same camera",
        "camera-label": "CAMERA",
        "work-label": "WORK",
        "asset-browser-label": "ASSETS / JOBS",
        "status-text": "SPECTRAL EXPOSURE ACTIVE",
        "work-visual-pass": "VISUAL PASS",
        "queue-pause-auto": "TOGGLE WORK",
    }
    plane_ids = {item["id"] for item in job["planes"]}
    assert "display-surface-editor-text" in plane_ids
    assert "display-surface-camera-label" in plane_ids
    assert "display-surface-work-label" in plane_ids
    assert "display-surface-asset-browser-label" in plane_ids
    assert "display-surface-asset-browser" in plane_ids
    assert "display-surface-status-text" in plane_ids
    assert "display-surface-window-minimize" in plane_ids
    assert "display-icon-window-minimize-bar" in plane_ids
    assert "display-icon-window-close-forward" in plane_ids
    assert "display-icon-window-close-backward" in plane_ids
    planes = {item["id"]: item for item in job["planes"]}
    assert planes["display-surface-camera-panel"]["library_object_key"] == (
        "layout-panel-style:bakery-slate"
    )
    assert planes["display-surface-work-visual-pass"][
        "library_object_key"
    ] == "layout-control-style:bakery-slate"
    assert len(planes["display-surface-work-visual-pass"][
        "library_parameters"
    ]["patch_instances"]) == 9
    assert job["camera"]["position_m"] == list(scene.camera.position_m)
    assert job["camera"]["target_m"] == list(scene.camera.target_m)
    assert orders.order_runtime_settings(job)["region"] == {
        "x": 0, "y": 0, "width": 1024, "height": 1024,
    }
    assert orders.order_runtime_settings(job)["camera_manifest"]["sensor"][
        "ui_content_region_px"
    ] == {"x": 432, "y": 464, "width": 160, "height": 96}


def test_display_preview_never_enlarges_traced_pixels():
    assert demo._texture_display_rect(960, 600, 960, 600) == (0, 0, 960, 600)
    assert demo._texture_display_rect(960, 600, 1920, 1200) == (480, 300, 960, 600)
    assert demo._texture_display_rect(960, 600, 480, 300) == (0, 0, 480, 300)


def test_completed_work_preview_is_scaled_and_centered_to_use_panel():
    assert demo._texture_panel_rect(10, 5, 100, 100) == (0, 25, 100, 50)


def test_active_calibration_bundle_exposure_owns_the_center_work_panel():
    exposure = object()
    browser = object()
    atlas = object()
    priority = object()

    assert demo._work_panel_preview_texture(
        exposure, browser, atlas, priority, calibration_active=True
    ) is exposure
    assert demo._work_panel_preview_texture(
        exposure, browser, atlas, priority, calibration_active=False
    ) is browser
    assert demo._work_panel_preview_texture(
        exposure, None, atlas, priority, calibration_active=False
    ) is atlas


def test_program_layout_owns_six_tool_rows_above_preview_panels():
    layout = demo.resolved_program_ui_layout(
        800, 600, work_width=200, work_height=200
    )
    row_keys = (
        "camera-toolbar", "lens-toolbar", "light-toolbar", "film-toolbar",
        "integrator-toolbar", "exposure-toolbar",
    )
    rows = [layout.region(key) for key in row_keys]
    first_panel_y = min(
        layout.region(key)[1]
        for key in ("camera-panel", "work-panel", "asset-browser")
    )

    assert all(row[0] == layout.region(layout.manifest.name)[0] for row in rows)
    assert all(row[2] == layout.width for row in rows)
    action_bottom = max(
        layout.region(key)[1] + layout.region(key)[3]
        for key in (
            "work-visual-pass", "queue-pause-auto", "window-minimize",
            "window-maximize", "window-close",
        )
    )
    assert action_bottom <= rows[0][1]
    assert all(row[3] <= 13 for row in rows)
    assert all(
        first[1] + first[3] <= second[1]
        for first, second in zip(rows, rows[1:])
    )
    assert rows[-1][1] + rows[-1][3] <= first_panel_y


def test_sensor_product_maps_into_shared_progressive_texture():
    photographed = demo.program_display_region(200, 120)
    products = demo.program_ui_sensor_regions(200, 120)

    editor_rect = demo._sensor_product_texture_rect(
        products["editor-text"], photographed, 200, 120
    )
    close_rect = demo._sensor_product_texture_rect(
        products["window-close"], photographed, 200, 120
    )
    assert editor_rect[0] == 0
    assert editor_rect[2] == (
        products["camera-panel"].width + products["work-panel"].width
    )
    assert editor_rect[1] > products["camera-panel"].y - photographed.y
    assert editor_rect[1] + editor_rect[3] <= photographed.height
    assert close_rect[0] + close_rect[2] <= photographed.width
    assert close_rect[1] < products["camera-label"].y - photographed.y


def test_sensor_product_maps_to_exact_ui_view_destination():
    photographed = demo.program_display_region(200, 120)
    product = demo.SensorRegion(
        photographed.x + 100, photographed.y + 10, 50, 20
    )

    assert demo._sensor_product_window_rect(
        product, photographed, 400, 240
    ) == (200, 20, 100, 40)


def test_display_to_native_square_is_inverse_of_readback_orientation():
    import exposure_render_demo as exposure

    native = np.arange(5 * 5 * 3, dtype=np.float32).reshape(5, 5, 3)
    cpp_getter_output = native[::-1, :, :]
    display = exposure._native_sensor_to_display(cpp_getter_output)

    restored = demo._display_raster_to_native_square(display)

    assert np.array_equal(restored, native)


def test_asymmetric_sensor_restore_does_not_create_a_mirrored_copy():
    import exposure_render_demo as exposure

    native = np.zeros((7, 7, 3), np.float32)
    native[1:6, 1, 0] = 1.0
    native[1, 1:5, 0] = 1.0
    native[3, 1:4, 0] = 1.0
    display = exposure._native_sensor_to_display(native[::-1])

    restored_native = demo._display_raster_to_native_square(display)
    resumed_display = exposure._native_sensor_to_display(restored_native[::-1])

    assert np.array_equal(resumed_display, display)
    assert not np.array_equal(resumed_display, display[:, ::-1])


def test_hud_extends_canvas_and_scales_for_small_displays():
    large_hud = demo._hud_layout(960)
    small_hud = demo._hud_layout(80)
    assert small_hud["height"] < large_hud["height"]
    assert small_hud["editor_font_px"] < large_hud["editor_font_px"]
    assert small_hud["status_font_px"] < large_hud["status_font_px"]

    canvas, hud = demo._window_regions(80, 48 + small_hud["height"])
    assert canvas == (0, 0, 80, 48)
    assert hud == (0, 48, 80, small_hud["height"])
    assert canvas[1] + canvas[3] == hud[1]


@pytest.mark.parametrize(
    ("work_width", "work_height"),
    [(24, 24), (64, 48), (100, 100), (320, 180), (960, 600)],
)
def test_work_size_derives_scaling_program_frame(work_width, work_height):
    width, height = demo._initial_window_size(work_width, work_height)
    crop = demo.program_display_region(width, height)
    regions = demo.program_ui_sensor_regions(
        width, height, work_width=work_width, work_height=work_height
    )

    assert width >= 2 * work_width
    assert height > work_height
    assert regions["program-backdrop"] == crop
    assert regions["camera-panel"].width == round(
        regions["work-panel"].width * 1.20
    )
    assert regions["camera-panel"].height <= work_height
    assert regions["work-panel"].width <= work_width
    assert regions["work-panel"].height <= work_height
    assert regions["work-panel"].x == (
        regions["camera-panel"].x + regions["camera-panel"].width
    )
    assert regions["camera-label"].y < regions["camera-panel"].y
    assert regions["work-label"].y < regions["work-panel"].y
    assert (
        regions["camera-panel"].y
        >= regions["camera-label"].y
        + regions["camera-label"].height
        + 3
    )
    assert regions["editor-text"].y == (
        regions["camera-panel"].y + regions["camera-panel"].height
    )
    assert regions["status-text"].y == (
        regions["editor-text"].y + regions["editor-text"].height
    )
    assert abs(
        regions["status-text"].y + regions["status-text"].height
        - (crop.y + crop.height)
    ) <= 1
    for region in regions.values():
        assert region.x >= crop.x
        assert region.y >= crop.y
        assert region.x + region.width <= crop.x + crop.width
        assert region.y + region.height <= crop.y + crop.height


def test_program_world_layout_matches_square_native_camera_frame():
    scene_width, scene_height = demo._initial_window_size(100, 100)
    crop = demo.program_display_region(scene_width, scene_height)
    full = demo._placement_for_sensor_region(
        crop, crop, thickness_m=0.01
    )

    assert np.isclose(full.size_m[0], demo.PROGRAM_SCENE_EXTENT_M)
    assert np.isclose(full.size_m[1], demo.PROGRAM_SCENE_EXTENT_M)


def test_work_sized_scene_has_exact_camera_and_work_sensor_products():
    scene_width, scene_height = demo._initial_window_size(100, 100)
    crop = demo.program_display_region(scene_width, scene_height)
    regions = demo.program_ui_sensor_regions(
        scene_width,
        scene_height,
        work_width=100,
        work_height=100,
    )
    scene = demo.build_self_rendering_program_scene(
        "one camera renders this whole UI",
        display_width=scene_width,
        display_height=scene_height,
        work_width=100,
        work_height=100,
    )

    assert crop.width == scene_width
    assert crop.height == scene_height
    assert regions["program-backdrop"] == crop
    assert regions["camera-panel"].x == crop.x
    assert regions["camera-panel"].width == 120
    assert regions["camera-panel"].height == 100
    assert regions["work-panel"].x == (
        regions["camera-panel"].x + regions["camera-panel"].width
    )
    assert regions["work-panel"].width == 100
    assert regions["work-panel"].height == 100
    assert regions["camera-panel"].y == regions["work-panel"].y
    assert regions["camera-label"].y < regions["camera-panel"].y
    assert regions["work-label"].y < regions["work-panel"].y
    assert regions["window-close"].y < regions["camera-label"].y
    assert regions["editor-text"].x == crop.x
    assert regions["editor-text"].y == (
        regions["camera-panel"].y + regions["camera-panel"].height
    )
    assert regions["editor-text"].width == (
        regions["camera-panel"].width + regions["work-panel"].width
    )
    assert regions["asset-browser"].x == (
        regions["work-panel"].x + regions["work-panel"].width
    )
    assert regions["asset-browser"].width == (
        crop.width - regions["editor-text"].width
    )
    assert regions["asset-browser"].y == regions["camera-panel"].y
    assert regions["asset-browser"].y + regions["asset-browser"].height == (
        crop.y + crop.height
    )
    assert regions["status-text"].x == crop.x
    assert regions["status-text"].y == (
        regions["editor-text"].y + regions["editor-text"].height
    )
    assert regions["status-text"].width == regions["editor-text"].width
    assert (
        regions["status-text"].y + regions["status-text"].height
        == crop.y + crop.height
    )
    for region in regions.values():
        assert region.x >= crop.x
        assert region.y >= crop.y
        assert region.x + region.width <= crop.x + crop.width
        assert region.y + region.height <= crop.y + crop.height
    expected_sensor_extent = demo.DEFAULT_FILM_FORMAT.default_final_edge_px
    assert scene.sensor_width == expected_sensor_extent
    assert scene.sensor_height == expected_sensor_extent
    work_label = regions["work-label"]
    for object_id in (
        "window-minimize", "window-maximize", "window-close",
    ):
        control_region = regions[object_id]
        assert (
            work_label.x + work_label.width <= control_region.x
            or control_region.x + control_region.width <= work_label.x
            or work_label.y + work_label.height <= control_region.y
            or control_region.y + control_region.height <= work_label.y
        )
    by_id = {item.object_id: item for item in scene.objects}
    assert by_id["editor-text"].horizontal_align == "left"
    assert by_id["editor-text"].vertical_align == "top"
    assert by_id["status-text"].horizontal_align == "left"
    assert by_id["status-text"].vertical_align == "bottom"
    assert by_id["camera-panel"].placement.thickness_m <= 0.002
    assert by_id["camera-label"].placement.thickness_m <= 0.001
    assert by_id["program-backdrop"].products[0].sensor_region == crop
    assert by_id["program-backdrop"].placement.size_m == (
        demo.PROGRAM_SCENE_EXTENT_M,
        demo.PROGRAM_SCENE_EXTENT_M,
    )
    job = orders.resolved_jobs(
        demo.build_display_scene_order(scene, sensor_sweeps=1), demo.JOB_ID
    )[0]
    assert orders.order_runtime_settings(job)["region"] == {
        "x": 0, "y": 0,
        "width": expected_sensor_extent, "height": expected_sensor_extent,
    }
    jobs_by_id = {
        item["id"]: item for item in job["objects"]
    }
    assert jobs_by_id["editor-text"]["geometry"]["horizontal_align"] == "left"
    assert jobs_by_id["editor-text"]["geometry"]["vertical_align"] == "top"
    assert jobs_by_id["editor-text"]["font"]["family"] == "DejaVu Sans Mono"
    assert jobs_by_id["status-text"]["geometry"]["horizontal_align"] == "left"
    assert jobs_by_id["status-text"]["geometry"]["vertical_align"] == "bottom"
    assert jobs_by_id["status-text"]["font"]["family"] == "DejaVu Sans Mono"
    control = demo.build_ui_next_scan_control(
        scene,
        crop,
        sequence=1,
        targeted_fraction=0.75,
        scan_width=100,
        scan_height=100,
    )
    for request in control["uv_requests"]:
        u0, v0, u1, v1 = request["uv_bounds"]
        assert (u1 - u0) * crop.width <= 100.0 + 1.0e-9
        assert (v1 - v0) * crop.height <= 100.0 + 1.0e-9
    assert all(
        len(request["site_indices"]) <= 100 * 100
        for request in control["pixel_slice_requests"]
    )
    assert {obj.object_id for obj in scene.objects} == {
        "camera-panel",
        "program-backdrop",
        "work-panel",
        "asset-browser",
        "camera-toolbar",
        "lens-toolbar",
        "light-toolbar",
        "film-toolbar",
        "integrator-toolbar",
        "exposure-toolbar",
        "camera-label",
        "work-label",
        "asset-browser-label",
        "editor-text",
        "status-text",
        "work-visual-pass",
        "queue-pause-auto",
        "window-minimize",
        "window-maximize",
        "window-close",
    }
    assert {
        item["id"]: item["token"] for item in job["objects"]
    } == {
        "editor-text": "one camera renders this whole UI",
        "camera-label": "CAMERA",
        "work-label": "WORK",
        "asset-browser-label": "ASSETS / JOBS",
        "status-text": "SPECTRAL EXPOSURE ACTIVE",
        "work-visual-pass": "VISUAL PASS",
        "queue-pause-auto": "TOGGLE WORK",
    }


def test_authored_focus_reaches_native_camera_and_bdpt_descriptor():
    import exposure_render_demo as exposure

    job = _resolved("focused text")
    base = exposure._build_thick_lens_lab_tracer_scene()
    scene, _ = orders.compile_job(base, job)
    previous_job = exposure._ORDERED_SCENE_JOB
    try:
        exposure._ORDERED_SCENE_JOB = job
        cam, solved = exposure._build_thick_lens_lab_camera_package(
            scene, 160, 160, exposure.DEFAULT_OPTICS
        )
        expected_sensor_distance = orders.camera_focus_distance(job)
        assert np.isclose(solved.sanity_input.focus_distance_m, expected_sensor_distance)
        assert np.isclose(solved.sanity_report.planes.focal_plane_z_m, expected_sensor_distance)

        descriptor = exposure._build_camera_sensor_descriptor(
            exposure.DEFAULT_OPTICS, cam, 160, 160, solved=solved
        )
        assert descriptor is not None
        expected_lens_distance = expected_sensor_distance - cam.focal_m
        assert np.isclose(descriptor["focus_distance_m"], expected_lens_distance)
        assert descriptor["focus_distance_m"] > 0.0
    finally:
        exposure._ORDERED_SCENE_JOB = previous_job


def test_authored_sensor_focus_is_converted_to_physical_lab_conjugate():
    import exposure_render_demo as exposure
    import thick_lens_focus_lab as lab

    scene = lab.SceneConfig()
    sensor_focus_m = 3.75
    object_distance_m = exposure._lab_object_distance_for_sensor_focus(scene, sensor_focus_m)
    design = scene.optical_design
    focal_m = design.focal_length_range_m[0] + (
        design.focal_length_range_m[1] - design.focal_length_range_m[0]
    ) * design.zoom
    group_span_m = (
        (design.group_count - 1) * design.min_air_gap_m
        + design.group_count * design.group_thickness_m
    )
    image_distance_m = 1.0 / (1.0 / focal_m - 1.0 / object_distance_m)
    reconstructed = (
        object_distance_m + group_span_m + image_distance_m + design.sensor_clearance_m
    )

    assert object_distance_m < sensor_focus_m
    assert np.isclose(reconstructed, sensor_focus_m, atol=1.0e-12)


def test_physical_lab_focus_probe_corrects_finite_lens_error():
    import exposure_render_demo as exposure
    import thick_lens_focus_lab as lab

    sensor_focus_m = 3.75
    scene = exposure._ordered_lab_scene_for_sensor_focus(lab, sensor_focus_m)
    lab._scene_lenses(scene)
    physical_sensor_focus_m = scene.image_plate.x - scene.object_plane.x

    assert np.isclose(physical_sensor_focus_m, sensor_focus_m, atol=2.0e-6)


def test_paragraph_wraps_and_centers_inside_physical_text_box():
    job = _resolved(
        "A physically rendered paragraph wraps its words and remains centered "
        "inside the authored surface without touching its boundary."
    )
    layout = orders.layout_paragraph(job)
    contours = orders._paragraph_contours(job)
    points = np.concatenate([contour[:-1] for contour in contours], axis=0)
    box = np.asarray(layout["text_box_m"], np.float64)

    assert len(layout["lines"]) >= 2
    assert layout["block_height_m"] <= box[1] + 1.0e-10
    assert points[:, 0].min() >= -0.5 * box[0] - 1.0e-10
    assert points[:, 0].max() <= +0.5 * box[0] + 1.0e-10
    assert points[:, 1].min() >= -0.5 * box[1] - 1.0e-10
    assert points[:, 1].max() <= +0.5 * box[1] + 1.0e-10
    assert abs(float(points[:, 1].min() + points[:, 1].max())) < 0.02


def test_long_unbroken_word_is_hard_wrapped_and_fitted():
    job = _resolved("spectrophotometrically" * 5)
    layout = orders.layout_paragraph(job)

    assert len(layout["lines"]) > 1
    assert "".join(layout["lines"]) == job["token"]
    assert layout["block_height_m"] <= layout["text_box_m"][1] + 1.0e-10


def test_render_worker_keeps_inflight_revision_and_coalesces_pending_edits():
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[tuple[int, str]] = []

    def render(sequence, text, order, next_scan_control, progress_sink):
        calls.append((sequence, text))
        if sequence == 1:
            first_started.set()
            assert release_first.wait(timeout=2.0)
        return demo.RenderedTextRevision(
            sequence=sequence,
            text=text,
            image_path=f"{sequence}.png",
            linear_path=f"{sequence}.npy",
            manifest_path=f"{sequence}.json",
            elapsed_s=0.01,
        )

    worker = demo.SpectralTextRenderWorker(render)
    try:
        worker.submit("one", {"value": 1})
        assert first_started.wait(timeout=2.0)
        worker.submit("two", {"value": 2})
        worker.submit("three", {"value": 3})
        release_first.set()

        deadline = time.monotonic() + 2.0
        latest = None
        while time.monotonic() < deadline:
            latest, busy, error, submitted = worker.snapshot()
            if latest is not None and latest.text == "three" and not busy:
                break
            time.sleep(0.01)
        assert error == ""
        assert submitted == 3
        assert latest is not None and latest.sequence == 3
        assert calls == [(1, "one"), (3, "three")]
    finally:
        release_first.set()
        worker.close()


def test_render_worker_drops_superseded_progress_and_resets_between_submissions():
    first_started = threading.Event()
    release_first = threading.Event()

    def event(sequence: int, event_sequence: int) -> ExposureProgressEvent:
        return ExposureProgressEvent(
            exposure_id=f"revision-{sequence:04d}",
            sequence=event_sequence,
            kind=ExposureProgressKind.PASS_AVAILABLE,
            region=SensorRegion(x=0, y=0, width=8, height=8),
            pass_index=event_sequence,
            completed_work=event_sequence,
            total_work=2,
        )

    def render(sequence, text, order, next_scan_control, progress_sink):
        if sequence == 1:
            progress_sink(event(1, 1))
            first_started.set()
            assert release_first.wait(timeout=2.0)
            progress_sink(event(1, 2))
        else:
            progress_sink(event(sequence, 1))
        return demo.RenderedTextRevision(
            sequence=sequence,
            text=text,
            image_path=f"{sequence}.png",
            linear_path=f"{sequence}.npy",
            manifest_path=f"{sequence}.json",
            elapsed_s=0.01,
        )

    worker = demo.SpectralTextRenderWorker(render)
    try:
        worker.submit("one", {})
        assert first_started.wait(timeout=2.0)
        latest, _ = worker.progress_snapshot()
        assert latest is not None and latest.exposure_id == "revision-0001"

        worker.submit("two", {})
        latest, layers = worker.progress_snapshot()
        assert latest is None
        assert layers == {}
        release_first.set()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            latest, _ = worker.progress_snapshot()
            rendered, busy, error, _ = worker.snapshot()
            if latest is not None and rendered is not None and not busy:
                break
            time.sleep(0.01)
        assert error == ""
        assert latest is not None and latest.exposure_id == "revision-0002"
        assert latest.sequence == 1
    finally:
        release_first.set()
        worker.close()


def test_render_worker_forwards_current_progress_to_inventory_observer():
    observed = []

    def render(sequence, text, order, next_scan_control, progress_sink):
        progress_sink(ExposureProgressEvent(
            exposure_id="shared-scene",
            sequence=1,
            kind=ExposureProgressKind.LAYER_AVAILABLE,
            region=SensorRegion(0, 0, 8, 8),
            pass_index=1,
        ))
        return demo.RenderedTextRevision(
            sequence=sequence,
            text=text,
            image_path="image.png",
            linear_path="linear.npy",
            manifest_path="manifest.json",
            elapsed_s=0.01,
        )

    worker = demo.SpectralTextRenderWorker(render, progress_observer=observed.append)
    try:
        worker.submit("one", {})
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            latest, busy, error, _ = worker.snapshot()
            if latest is not None and not busy:
                break
            time.sleep(0.01)
        assert error == ""
        assert len(observed) == 1
        assert observed[0].exposure_id == "shared-scene"
    finally:
        worker.close()


def test_render_owner_preempts_background_atlas_work_for_foreground_revision():
    from camera_software import (
        BakeRequest,
        BakeTargetKind,
        DEFAULT_INK_CONDITION,
        DisplayProductKind,
        ink_token_asset,
    )

    glyph = ink_token_asset("A")
    request = BakeRequest(
        glyph.asset_key,
        BakeTargetKind.ATLAS_GLYPH,
        DEFAULT_INK_CONDITION,
        DisplayProductKind.IMAGE,
        "atlas",
        ("A",),
        glyph,
    )
    background_started = threading.Event()
    background_retried = threading.Event()
    release_first = threading.Event()
    calls = []

    class Background:
        def __init__(self):
            self.attempt = 0

        def cancel_current(self):
            release_first.set()

        def __call__(self, _request):
            self.attempt += 1
            calls.append(f"background-{self.attempt}")
            if self.attempt == 1:
                background_started.set()
                assert release_first.wait(timeout=2.0)
                raise RuntimeError("preempted")
            background_retried.set()

    background = Background()

    def source():
        return None if background.attempt >= 2 else request

    def render(sequence, text, order, next_scan_control, progress_sink):
        calls.append(f"foreground-{sequence}")
        return demo.RenderedTextRevision(
            sequence=sequence,
            text=text,
            image_path="image.png",
            linear_path="linear.npy",
            manifest_path="manifest.json",
            elapsed_s=0.01,
        )

    worker = demo.SpectralTextRenderWorker(
        render,
        background_source=source,
        background_render_function=background,
    )
    try:
        assert background_started.wait(timeout=2.0)
        worker.submit("urgent", {})
        assert background_retried.wait(timeout=2.0)
        assert calls[:3] == ["background-1", "foreground-1", "background-2"]
        assert worker.background_snapshot()[1] == ""
    finally:
        release_first.set()
        worker.close()


def test_background_queue_pause_is_explicit_and_resumable():
    def render(sequence, text, order, next_scan_control, progress_sink):
        return demo.RenderedTextRevision(
            sequence, text, "image.png", "linear.npy", "manifest.json", 0.01
        )

    worker = demo.SpectralTextRenderWorker(render)
    try:
        assert not worker.background_is_paused()
        worker.set_background_paused(True)
        assert worker.background_is_paused()
        worker.wake_background()
        assert worker.background_is_paused()
        assert worker.toggle_background_paused() is False
        assert not worker.background_is_paused()
    finally:
        worker.close()


def test_background_queue_can_be_constructed_startup_paused():
    def render(sequence, text, order, next_scan_control, progress_sink):
        return demo.RenderedTextRevision(
            sequence, text, "image.png", "linear.npy", "manifest.json", 0.01
        )

    worker = demo.SpectralTextRenderWorker(render, background_paused=True)
    try:
        assert worker.background_is_paused()
        worker.wake_background()
        assert worker.background_is_paused()
        assert worker.toggle_background_paused() is False
    finally:
        worker.close()


@pytest.mark.parametrize(
    "mode_key",
    (
        "color-science", "glass", "focus-hall", "depth", "prism-room",
        "single-lane-ui",
    ),
)
def test_calibration_modes_build_valid_image_producing_scene_orders(mode_key):
    order = demo.build_calibration_render_order(
        mode_key, display_width=96, display_height=64, sensor_sweeps=1
    )

    orders.validate_order(order)
    assert order["runtime"]["work_kind"] == "calibration"
    assert order["runtime"]["max_sensor_epochs"] == 1
    assert order["runtime"]["ordinary_work"] is False
    assert order["runtime"]["total_rays"] > 0
    assert order["runtime"]["rays_per_batch"] > 0
    transport = order["transport"]
    expected_lanes = (
        3 if mode_key in {"color-science", "glass", "prism-room"} else 1
    )
    assert len(orders.order_transport_frequencies(order)) == expected_lanes
    assert orders.order_transport_rgb_weights(order).shape == (expected_lanes, 3)
    assert transport["domain"] == "fixed_spectral"
    job = orders.resolved_jobs(order, demo.JOB_ID)[0]
    assert job["planes"]
    assert job["image"]["region"] == {
        "x": 0, "y": 0, "width": 1024, "height": 1024,
    }
    assert order["runtime"]["sensor_work_tile_width"] == 96
    assert order["runtime"]["sensor_work_tile_height"] == 64
    if mode_key in {"focus-hall", "depth"}:
        assert len(job["planes"]) == 12
        assert len(job["objects"]) == 7
    if mode_key == "depth":
        assert order["runtime"]["integrator"] == "depth"
        assert order["runtime"]["render_product"] == "sensor_optical_path_depth_m"
    if mode_key == "glass":
        material = job["materials"]["bk7_calibration"]
        assert len(material["bands"]) == 3
        assert material["bands"][0]["ior_real"] > material["bands"][-1]["ior_real"]
    if mode_key == "prism-room":
        prism = next(plane for plane in job["planes"] if plane["id"] == "prism")
        assert prism["shape"] == "triangular_prism"
        assert job["flash"]["enabled"] is False
        assert job["emitter"]["mode"] == "collimated"


@pytest.mark.parametrize("lane_count", (1, 3, 8, 16, 32))
def test_same_camera_scene_accepts_continuous_lane_table_at_any_lane_count(lane_count):
    from camera_software.transport_contract import continuous_lut_lane_table

    c = 299_792_458.0
    table = continuous_lut_lane_table(
        f"test-cohort-{lane_count}",
        (c / 700.0e-9, c / 550.0e-9, c / 400.0e-9),
        (0.35, 1.0, 0.35),
        lane_count,
        seed=lane_count,
    )
    order = demo.build_calibration_render_order(
        "transport-sanity",
        display_width=32,
        display_height=24,
        lane_table=table,
        startup_validation_key=f"spectral:{lane_count}",
    )

    orders.validate_order(order)
    assert order["transport"]["domain"] == "continuous_spectral_lut"
    assert len(orders.order_transport_frequencies(order)) == lane_count
    assert orders.order_transport_rgb_weights(order) is None
    lut = orders.order_transport_lut_config(order)
    assert lut is not None
    assert lut["lane_lut_index"].shape == (lane_count,)
    assert order["runtime"]["startup_validation_key"] == f"spectral:{lane_count}"
    assert order["runtime"]["total_rays"] == 256


@pytest.mark.parametrize("lane_count", (16, 32))
def test_prism_room_accepts_large_fixed_perceptual_lane_payloads(lane_count):
    order = demo.build_calibration_render_order(
        "prism-room",
        display_width=32,
        display_height=24,
        transport_option=f"fixed:{lane_count}",
    )

    orders.validate_order(order)
    frequencies = orders.order_transport_frequencies(order)
    weights = orders.order_transport_rgb_weights(order)
    lanes = order["transport"]["lane_table"]["lanes"]
    assert frequencies.shape == (lane_count,)
    assert weights.shape == (lane_count, 3)
    assert np.allclose(weights, np.asarray([
        lane["sensor_weight_xyz"] for lane in lanes
    ], np.float32))
    assert all(tuple(lane["sensor_weight_xyz"]) != (1.0, 1.0, 1.0)
               for lane in lanes)


def test_mirror_box_is_closed_fixed32_capacity_torture_scene():
    order = demo.build_calibration_render_order(
        "mirror-box", display_width=256, display_height=256,
    )

    orders.validate_order(order)
    planes = {plane["id"]: plane for plane in order["defaults"]["planes"]}
    assert {
        "mirror-floor", "mirror-ceiling", "mirror-left", "mirror-right",
        "mirror-front", "mirror-back", "internal-light", "diffuse-witness",
    } == set(planes)
    assert planes["internal-light"]["emitter"] is True
    assert len(order["transport"]["frequencies_hz"]) == 32
    assert order["runtime"]["transport_option"] == "fixed:32"
    assert order["runtime"]["sensor_flash_page_count"] == 64
    assert order["runtime"]["sensor_flash_total_rays"] == 1_048_576
    assert order["runtime"]["max_bounces"] == 64
