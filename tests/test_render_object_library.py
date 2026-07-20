import json
from pathlib import Path

import numpy as np

from camera_software import (
    DEFAULT_INK_CONDITION,
    DisplayProductKind,
    FontAssetSpec,
    LightFieldCondition,
    ObjectViewMode,
    RayTracedSprite,
    RenderAssetCatalog,
    RenderObjectLibrary,
    RenderedAssetRecord,
    ink_token_asset,
    render_style_key,
)

FONT = FontAssetSpec(family="DejaVu Sans Mono", weight="bold", style="normal")


def _scene(path: Path, token: str, *, focal_mm: float = 35.0, character: bool | None = None) -> tuple[Path, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    asset = ink_token_asset(token, font=FONT, character=character)
    path.write_text(json.dumps({
        "schema_version": 1,
        "defaults": {
            "camera": {"focal_mm": focal_mm, "aperture_mm": 25.0},
            "exposure": {"time_s": 1.0 / 60.0, "iso": 100.0},
            "flash": {"intensity_scale": 1.0},
            "font": FONT.scene_order_mapping(),
            "materials": {"ink": {"roughness": 0.065}, "slate": {"roughness": 0.9}},
            "geometry": asset.extrusion.scene_order_mapping("backplate", "ink"),
        },
        "jobs": [asset.scene_order_object("text-subtype", "backplate")],
    }), encoding="utf-8")
    return path, asset


def _record(directory: Path, asset, *, condition=DEFAULT_INK_CONDITION, value: float = 1.0, target_kind: str = "atlas_glyph") -> RenderedAssetRecord:
    linear = directory / "0000_cpp_linear.npy"
    preview = directory / "0000_cpp.png"
    sprite_path = directory / "raytraced_sprite.npz"
    sensor_sum = directory / "0000_cpp_sum_linear.npy"
    weight = directory / "0000_cpp_exposure_weight.npy"
    np.save(linear, np.full((8, 8, 3), value, np.float32))
    np.save(sensor_sum, np.full((8, 8, 3), value, np.float32))
    np.save(weight, np.ones((8, 8), np.float32))
    preview.write_bytes(b"png")
    alpha = np.zeros((8, 8), np.float32)
    alpha[2:6, 2:6] = 1.0
    RayTracedSprite(
        token=asset.token,
        asset_key=asset.asset_key,
        premultiplied_rgb=np.full((8, 8, 3), value, np.float32) * alpha[..., None],
        alpha=alpha,
        additive_rgb=np.zeros((8, 8, 3), np.float32),
        source_background_rgb=np.zeros((8, 8, 3), np.float32),
        content_region=(0, 0, 8, 8),
    ).save(str(sprite_path))
    (directory / "0000_cpp_summary.json").write_text('{"resolved": true}', encoding="utf-8")
    (directory / "render_context.json").write_text('{"command": ["renderer"]}', encoding="utf-8")
    (directory / "render.log").write_text("rendered", encoding="utf-8")
    return RenderedAssetRecord(
        asset.asset_key, condition.condition_key, DisplayProductKind.IMAGE,
        linear_path=str(linear), preview_path=str(preview), width=8, height=8, samples=4,
        metadata={
            "target_kind": target_kind,
            "sprite_path": str(sprite_path),
            "sum_linear_path": str(sensor_sum),
            "exposure_weight_path": str(weight),
            "completion_basis": "test-complete",
        },
    )


def test_style_object_archives_glyph_as_canonical_subtype(tmp_path):
    legacy = tmp_path / "legacy_glyph_a"
    scene_path, asset = _scene(legacy / "scene_order.json", "A", character=True)
    record = _record(legacy, asset)
    library_path = tmp_path / "object_library.json"
    library = RenderObjectLibrary(str(library_path))

    bundle = library.adopt_record(record, scene_path=str(scene_path), tags=("font", "alphabet"))

    assert bundle.object_key == render_style_key(asset)
    assert bundle.object_kind == "render_style"
    assert len(bundle.subtypes) == 1
    subtype = bundle.subtypes[0]
    assert subtype.kind == "glyph"
    assert subtype.text == "A"
    assert subtype.complete
    assert bundle.subtype_sets["glyph"] == (subtype.subtype_key,)
    assert bundle.subtype_sets["token"] == ()
    assert set(bundle.available_view_modes) == set(ObjectViewMode)
    assert Path(subtype.scenes[0].scene_path).is_relative_to(tmp_path / "render_objects")
    assert not Path(subtype.scenes[0].scene_path).is_relative_to(legacy)
    assert subtype.scenes[0].camera_spec["focal_mm"] == 35.0
    assert Path(bundle.manifest_path).is_file()

    scene_view = library.select_view(bundle.object_key, ObjectViewMode.SCENE, text="A")
    image_view = library.select_view(bundle.object_key, ObjectViewMode.IMAGE, text="A")
    assert scene_view.subtype_key == subtype.subtype_key
    assert image_view.primary_path == subtype.artifacts[0].preview_path
    assert RenderObjectLibrary(str(library_path)).find(bundle.object_key) is not None


def test_style_object_grows_condition_collection_without_overwrite(tmp_path):
    legacy_a = tmp_path / "view_000"
    scene_a, asset = _scene(legacy_a / "scene_order.json", "A", character=True)
    second = LightFieldCondition(azimuth_deg=90.0, elevation_deg=10.0, light_rig="fixed_key", material_variant="red_ink_on_black_slate", frame=1)
    legacy_b = tmp_path / "view_001"
    scene_b, _ = _scene(legacy_b / "scene_order.json", "A", focal_mm=50.0, character=True)
    library = RenderObjectLibrary(str(tmp_path / "library.json"))
    library.adopt_record(_record(legacy_a, asset), scene_path=str(scene_a))
    bundle = library.adopt_record(_record(legacy_b, asset, condition=second, value=2.0), scene_path=str(scene_b), condition=second)

    subtype = bundle.subtypes[0]
    assert len(subtype.scenes) == 2
    assert len(subtype.artifacts) == 2
    assert len({Path(scene.scene_path).parent for scene in subtype.scenes}) == 2
    textured = library.select_view(bundle.object_key, ObjectViewMode.LIGHT_FIELD, text="A")
    assert len(textured.texture_paths) == 2


def test_one_style_contains_partial_glyph_and_whole_token_subtype_sets_and_composes(tmp_path):
    library = RenderObjectLibrary(str(tmp_path / "library.json"))
    assets = []
    for token, character, target_kind in (("A", True, "atlas_glyph"), ("B", True, "atlas_glyph"), ("AB", False, "token")):
        legacy = tmp_path / f"legacy_{token}"
        scene, asset = _scene(legacy / "scene_order.json", token, character=character)
        library.adopt_record(_record(legacy, asset, target_kind=target_kind), scene_path=str(scene))
        assets.append(asset)

    assert len(library.snapshot()) == 1
    bundle = library.snapshot()[0]
    assert len(bundle.subtype_sets["glyph"]) == 2
    assert len(bundle.subtype_sets["token"]) == 1
    assert library.find_subtype(bundle.object_key, "AB", kind="token") is not None
    exact_scene = library.compose_scene(bundle.object_key, "AB")
    glyph_scene = library.compose_scene(bundle.object_key, "AB", prefer_whole_tokens=False)
    assert len(exact_scene.scene_objects) == 1
    assert exact_scene.scene_objects[0]["token"] == "AB"
    assert len(glyph_scene.scene_objects) == 2
    assert [obj["token"] for obj in glyph_scene.scene_objects] == ["A", "B"]
    sprite = library.compose_sprite(bundle.object_key, "AB", 96, 48)
    assert sprite.used_tokens == ("AB",)
    assert not sprite.missing_characters


def test_catalog_migration_repoints_records_to_canonical_object_storage(tmp_path):
    legacy = tmp_path / "old_atlas" / "glyph_a"
    scene, asset = _scene(legacy / "scene_order.json", "A", character=True)
    record = _record(legacy, asset)
    catalog = RenderAssetCatalog(str(tmp_path / "catalog.json"))
    catalog.record(record)
    original_mtime = Path(record.preview_path).stat().st_mtime_ns
    library = RenderObjectLibrary(str(tmp_path / "object_library.json"))

    bundles = library.migrate_catalog(catalog)
    migrated = catalog.find(asset.asset_key, DEFAULT_INK_CONDITION, DisplayProductKind.IMAGE)

    assert len(bundles) == 1
    assert migrated is not None
    assert Path(migrated.preview_path).is_relative_to(tmp_path / "render_objects")
    assert not Path(migrated.preview_path).is_relative_to(legacy)
    assert Path(record.preview_path).stat().st_mtime_ns == original_mtime
    assert Path(migrated.preview_path).read_bytes() == Path(record.preview_path).read_bytes()

def test_final_interface_after_render_is_canonical_and_traces_component_subtypes(tmp_path):
    library = RenderObjectLibrary(str(tmp_path / "object_library.json"))
    for token in ("A", "B"):
        legacy = tmp_path / f"legacy_{token}"
        scene, asset = _scene(legacy / "scene_order.json", token, character=True)
        library.adopt_record(_record(legacy, asset), scene_path=str(scene))

    revision = tmp_path / "revision_0001"
    job = revision / "spectral-program-scene"
    job.mkdir(parents=True)
    scene_path = revision / "scene_order.json"
    scene_path.write_text(json.dumps({
        "schema_version": 1,
        "defaults": {
            "objects": [
                {"id": "title", "token": "AB"},
                {"id": "button", "token": "A"},
            ],
        },
        "jobs": [{"id": "live_paragraph"}],
    }), encoding="utf-8")
    image = job / "0000_cpp_camera.png"
    image.write_bytes(b"camera")
    linear = job / "0000_cpp_linear.npy"
    np.save(linear, np.ones((4, 4, 3), np.float32))
    manifest = job / "composition_manifest.json"
    manifest.write_text('{"complete": true}', encoding="utf-8")
    diagnostic = job / "0000_cpp.png"
    diagnostic.write_bytes(b"diagnostic")
    log = revision / "render.log"
    log.write_text("complete", encoding="utf-8")

    final = library.register_interface_after_render(
        "complete-ui",
        1,
        scene_path=str(scene_path),
        image_path=str(image),
        linear_path=str(linear),
        manifest_path=str(manifest),
        text="AB",
        diagnostic_image_path=str(diagnostic),
        render_log_path=str(log),
        metadata={"all_ui_elements_in_place": True},
    )

    assert Path(final.image_path).is_relative_to(tmp_path / "render_interfaces")
    assert final.metadata["all_ui_elements_in_place"]
    assert final.scene_object_ids == ("title", "button")
    assert [(ref.scene_object_id, ref.text) for ref in final.component_references] == [
        ("title", "A"), ("title", "B"), ("button", "A"),
    ]
    selected = library.select_interface_after_render("complete-ui")
    assert selected.render_key == final.render_key
    assembly = library.find_interface(final.assembly_key)
    assert assembly is not None
    assert assembly.latest == final
    assert Path(assembly.manifest_path).is_file()
    restored = RenderObjectLibrary(str(tmp_path / "object_library.json"))
    assert restored.select_interface_after_render("complete-ui").render_key == final.render_key