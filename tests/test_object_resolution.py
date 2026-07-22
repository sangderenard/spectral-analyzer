from types import SimpleNamespace

from PIL import Image

from camera_software import (
    HierarchicalObjectResolver,
    LayoutWorkProgressCache,
    build_layout_object_work_manifest,
    layout_program_ui,
    program_ui_manifest,
    write_layout_usd_package,
)
from render_work_browser import RenderWorkBrowser


def _resolver(tmp_path):
    layout = layout_program_ui(
        program_ui_manifest(), 360, 180, work_width=100, work_height=100
    )
    manifest = build_layout_object_work_manifest(layout, str(tmp_path))
    cache = LayoutWorkProgressCache(manifest.progress_cache_path)
    scene = tmp_path / "camera_scene.json"
    scene.write_text('{"schema_version": 1}', encoding="utf-8")
    preview = tmp_path / "camera.png"
    Image.new("RGBA", (32, 16), (220, 190, 120, 255)).save(preview)

    def subtype(text, kind):
        artifact = SimpleNamespace(
            samples=8,
            created_at_s=1.0,
            preview_path=str(preview),
            sprite_path="",
            linear_path="",
            metadata={
                "atlas_quality": {
                    "glyph_exposure_coverage": 0.8,
                    "glyph_radiance_coverage": 0.75,
                    "converged": False,
                }
            },
        )
        archived = SimpleNamespace(scene_path=str(scene))
        return SimpleNamespace(
            subtype_key=f"{kind}:{text}",
            kind=kind,
            text=text,
            scenes=(archived,),
            artifacts=(artifact,),
        )

    subtypes = (
        subtype("CAMERA", "token"),
        *(subtype(character, "glyph") for character in "WORK"),
    )
    bundle = SimpleNamespace(
        object_key="render-style:monofont",
        style_spec={"font": {"family": "DejaVu Sans Mono"}},
        subtype_sets={"token": ("token:CAMERA",), "glyph": tuple("WORK")},
        subtypes=subtypes,
        updated_at_s=1.0,
    )
    library = SimpleNamespace(snapshot=lambda: (bundle,))
    return HierarchicalObjectResolver(
        layout, library, manifest, cache
    ), manifest, cache


def test_layout_text_uses_exact_token_then_glyph_fallback_in_same_hierarchy(tmp_path):
    resolver, _manifest, _cache = _resolver(tmp_path)

    camera = resolver.resolve_text_image("camera-label", "CAMERA")
    work = resolver.resolve_text_image("work-label", "WORK")

    assert camera.resolution == "exact_token"
    assert [part.subtype_kind for part in camera.parts] == ["token"]
    assert work.resolution == "glyph_fallback"
    assert [part.text for part in work.parts] == list("WORK")
    assert all(part.scene_path.endswith("camera_scene.json") for part in work.parts)
    assert all(part.artifact_path.endswith("camera.png") for part in work.parts)


def test_resolution_tree_organizes_design_and_monofont_objects_under_scene(tmp_path):
    resolver, _manifest, _cache = _resolver(tmp_path)

    tree = resolver.tree()

    assert tree.kind == "scene"
    assert [child.key for child in tree.children] == [
        "layout-design-objects",
        "monofont-text-images",
    ]
    text_nodes = tree.children[1].children
    camera = next(node for node in text_nodes if "camera-label" in node.key)
    assert camera.status == "exact_token"
    assert camera.children[0].kind == "token"
    design_subtype = tree.children[0].children[0].children[0]
    assert any(
        child.kind == "aspect_variant" for child in design_subtype.children
    )


def test_assembly_readiness_reports_independent_provider_signoffs(tmp_path):
    resolver, _manifest, _cache = _resolver(tmp_path)

    readiness = resolver.assembly_readiness()

    providers = {item.provider_key: item for item in readiness.providers}
    assert not providers["monofont-render-library"].ready
    assert providers["monofont-render-library"].missing
    assert not providers["parametric-panel-library"].ready
    assert providers["parametric-panel-library"].fallback_permitted


def test_resolved_text_images_are_usd_children_and_camera_selectable(tmp_path):
    resolver, manifest, cache = _resolver(tmp_path)
    images = resolver.text_images()

    package = write_layout_usd_package(
        manifest,
        cache,
        str(tmp_path / "usd"),
        resolved_text_images=images,
    )
    stage = open(package.stage_path, encoding="utf-8").read()
    camera_layer = open(
        package.text_image_layers["camera-label"], encoding="utf-8"
    ).read()

    assert len(package.text_image_layers) == len(images)
    assert "text_image_camera_label" in stage
    assert 'custom token resolution = "exact_token"' in camera_layer
    assert 'custom string subtypeKey = "token:CAMERA"' in camera_layer
    assert "sourceScene = @" in camera_layer

    browser = RenderWorkBrowser()
    browser.sync([], [], resolved_text_images=images)
    camera_row = next(
        spec for spec in browser.widget.subpanels
        if spec.key == "text-image:camera-label"
    )
    assert camera_row.title == "TEXT EXACT  camera-label"
    browser.widget.selected_key = camera_row.key
    assert browser.selected_payload["subtype"].text == "CAMERA"
