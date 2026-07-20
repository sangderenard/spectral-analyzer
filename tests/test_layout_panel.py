import numpy as np

from camera_software import (
    LayoutPanelPrimitive,
    PanelPatchAsset,
    PanelPatchFill,
    PanelPatchRole,
    compose_layout_panel_rgba,
    layout_panel_primitive_from_mapping,
)


def _colored_primitive() -> LayoutPanelPrimitive:
    colors = {
        role: (index + 1, index + 11, index + 21, 255)
        for index, role in enumerate(PanelPatchRole)
    }
    return LayoutPanelPrimitive(
        "test-panel",
        tuple(
            PanelPatchAsset(role=role, color_rgba=colors[role])
            for role in PanelPatchRole
        ),
        border_px=(3, 2, 4, 5),
    )


def test_nine_slice_places_four_corners_four_sides_and_center():
    primitive = _colored_primitive()
    image = compose_layout_panel_rgba(primitive, 20, 16)
    colors = {patch.role: patch.color_rgba for patch in primitive.patches}

    assert image.shape == (16, 20, 4)
    assert tuple(image[0, 0]) == colors[PanelPatchRole.TOP_LEFT]
    assert tuple(image[0, 5]) == colors[PanelPatchRole.TOP]
    assert tuple(image[0, 19]) == colors[PanelPatchRole.TOP_RIGHT]
    assert tuple(image[4, 0]) == colors[PanelPatchRole.LEFT]
    assert tuple(image[4, 5]) == colors[PanelPatchRole.CENTER]
    assert tuple(image[4, 19]) == colors[PanelPatchRole.RIGHT]
    assert tuple(image[15, 0]) == colors[PanelPatchRole.BOTTOM_LEFT]
    assert tuple(image[15, 5]) == colors[PanelPatchRole.BOTTOM]
    assert tuple(image[15, 19]) == colors[PanelPatchRole.BOTTOM_RIGHT]


def test_panel_contract_supports_independent_render_object_patch_assets():
    primitive = layout_panel_primitive_from_mapping(
        {
            "border_px": [8, 9, 10, 11],
            "patches": [
                {
                    "role": role.value,
                    "object_key": "ornate-window",
                    "subtype_key": role.value,
                    "asset_key": f"ornate-window/{role.value}",
                    "fill": "tile" if role in {
                        PanelPatchRole.TOP, PanelPatchRole.RIGHT,
                        PanelPatchRole.BOTTOM, PanelPatchRole.LEFT,
                    } else "stretch",
                }
                for role in PanelPatchRole
            ],
        },
        primitive_id="ornate-window-panel",
    )
    contract = primitive.mapping()

    assert contract["kind"] == "nine_slice"
    assert contract["border_px"] == [8, 9, 10, 11]
    assert {patch["role"] for patch in contract["patches"]} == {
        role.value for role in PanelPatchRole
    }
    assert all(patch["object_key"] == "ornate-window" for patch in contract["patches"])


def test_tiled_side_uses_loaded_patch_image_and_tiny_sizes_contract_safely():
    primitive = LayoutPanelPrimitive(
        "tiled-panel",
        (
            PanelPatchAsset(
                role=PanelPatchRole.TOP,
                asset_key="top-pattern",
                fill=PanelPatchFill.TILE,
            ),
            PanelPatchAsset(
                role=PanelPatchRole.CENTER,
                color_rgba=(9, 8, 7, 255),
            ),
        ),
        border_px=(4, 1, 4, 1),
    )
    pattern = np.array(
        [[[255, 0, 0, 255], [0, 255, 0, 255]]],
        dtype=np.uint8,
    )
    composed = compose_layout_panel_rgba(
        primitive,
        10,
        4,
        image_loader=lambda patch: pattern if patch.asset_key == "top-pattern" else None,
    )
    tiny = compose_layout_panel_rgba(primitive, 3, 2)

    assert tuple(composed[0, 4]) == (255, 0, 0, 255)
    assert tuple(composed[0, 5]) == (0, 255, 0, 255)
    assert tiny.shape == (2, 3, 4)

def test_representative_square_crop_vectors_feed_all_nine_object_requests():
    primitive = layout_panel_primitive_from_mapping(
        {
            "object_key": "layout-panel-style:test",
            "subtype_key": "representative-square",
            "border_px": [3, 3, 3, 3],
        },
        primitive_id="crop-addressed-panel",
    )
    source = np.zeros((9, 9, 4), np.uint8)
    source[..., 3] = 255
    role_cells = {
        PanelPatchRole.TOP_LEFT: (0, 0),
        PanelPatchRole.TOP: (1, 0),
        PanelPatchRole.TOP_RIGHT: (2, 0),
        PanelPatchRole.LEFT: (0, 1),
        PanelPatchRole.CENTER: (1, 1),
        PanelPatchRole.RIGHT: (2, 1),
        PanelPatchRole.BOTTOM_LEFT: (0, 2),
        PanelPatchRole.BOTTOM: (1, 2),
        PanelPatchRole.BOTTOM_RIGHT: (2, 2),
    }
    colors = {}
    for index, (role, (column, row)) in enumerate(role_cells.items()):
        color = (20 + index, 50 + index, 80 + index, 255)
        colors[role] = color
        source[row * 3:(row + 1) * 3, column * 3:(column + 1) * 3] = color

    composed = compose_layout_panel_rgba(
        primitive, 12, 12, image_loader=lambda _patch: source
    )
    contract = primitive.mapping()

    assert len(primitive.object_requests) == 9
    assert len(contract["object_requests"]) == 9
    assert all(
        request["object_key"] == "layout-panel-style:test"
        and request["subtype_key"] == "representative-square"
        and request["source_capture"] == "representative_square"
        for request in contract["object_requests"]
    )
    assert tuple(composed[0, 0]) == colors[PanelPatchRole.TOP_LEFT]
    assert tuple(composed[0, 5]) == colors[PanelPatchRole.TOP]
    assert tuple(composed[0, 11]) == colors[PanelPatchRole.TOP_RIGHT]
    assert tuple(composed[5, 0]) == colors[PanelPatchRole.LEFT]
    assert tuple(composed[5, 5]) == colors[PanelPatchRole.CENTER]
    assert tuple(composed[5, 11]) == colors[PanelPatchRole.RIGHT]
    assert tuple(composed[11, 0]) == colors[PanelPatchRole.BOTTOM_LEFT]
    assert tuple(composed[11, 5]) == colors[PanelPatchRole.BOTTOM]
    assert tuple(composed[11, 11]) == colors[PanelPatchRole.BOTTOM_RIGHT]