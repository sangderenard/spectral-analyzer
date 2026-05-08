"""Smoke tests for doc-channel action dispatch in RoomControlStation.

Run with:
    python test_room_control_actions.py

No OpenGL context required.  A pygame display surface is created only to
satisfy pygame.event.Event construction; no window is shown.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import numpy as np
import pygame
from controls import get_action_registry, get_control_graph

# ---------------------------------------------------------------------------
# Module-level helpers under test
# ---------------------------------------------------------------------------
from room_control_station import (
    RoomControlStation,
    build_floor_mask,
    build_floor_plan,
    apply_hull_deformation,
    build_envelope_meshes,
    build_floor_material_triangulation,
    _FLOOR_TYPES,
    _HULL_WALL_TYPES,
    _HULL_CORNER_TYPES,
    _CELL_VOID,
    _CELL_COLLISION,
    _CELL_EMPTY,
    _make_polar_tile_cells,
)
from room_wall_geometry import build_wall_footprints, wall_occlusion_cells, wall_occlusion_voxels

pygame.init()
pygame.display.set_mode((1, 1), pygame.NOFRAME)   # minimal surface for Event


def _make_station() -> RoomControlStation:
    return RoomControlStation(
        station_cfg={"room_editor": {"preset_library_dir": ""}},
        left_cfg={}, center_cfg={}, right_cfg={},
    )


def _click(station: RoomControlStation, rect: tuple) -> bool:
    x, y, w, h = rect
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {
        "pos": (x + w // 2, y + h // 2),
        "button": 1,
    })
    return station.handle_event(ev)


# ---------------------------------------------------------------------------
# Test 1 – action button fires and mutates state
# ---------------------------------------------------------------------------
def test_action_level_increment():
    st = _make_station()
    st.state.setdefault("room_level", 0)
    rect = (100, 50, 80, 24)
    st._doc_action_rects["room_tile_workspace_controls.level:+"] = rect

    before = int(st.state.get("room_level", 0))
    consumed = _click(st, rect)
    after = int(st.state.get("room_level", 0))

    assert consumed, "handle_event must return True on action hit"
    assert after == before + 1, f"level must increment: {before} → {after}"
    print("PASS test_action_level_increment")


def test_action_level_decrement():
    st = _make_station()
    st.state["room_level"] = 5
    rect = (100, 50, 80, 24)
    st._doc_action_rects["room_tile_workspace_controls.level:-"] = rect

    _click(st, rect)
    assert int(st.state["room_level"]) == 4
    print("PASS test_action_level_decrement")


def test_action_dim_x_plus():
    st = _make_station()
    st.state["room_width_cells"] = 8
    rect = (200, 80, 80, 24)
    st._doc_action_rects["room_tile_workspace_controls.dimx:+"] = rect

    _click(st, rect)
    assert int(st.state["room_width_cells"]) == 9
    print("PASS test_action_dim_x_plus")


def test_action_dim_y_minus():
    st = _make_station()
    st.state["room_depth_cells"] = 8
    rect = (200, 80, 80, 24)
    st._doc_action_rects["room_tile_workspace_controls.dimy:-"] = rect

    _click(st, rect)
    assert int(st.state["room_depth_cells"]) == 7
    print("PASS test_action_dim_y_minus")


def test_action_dim_y_minus_clamped_at_2():
    st = _make_station()
    st.state["room_depth_cells"] = 2
    rect = (200, 80, 80, 24)
    st._doc_action_rects["room_tile_workspace_controls.dimy:-"] = rect

    _click(st, rect)
    assert int(st.state["room_depth_cells"]) == 2, "must clamp at 2"
    print("PASS test_action_dim_y_minus_clamped_at_2")


def test_doc_action_dispatches_registered_handler():
    st = _make_station()
    st.state.setdefault("room_level", 0)
    rect = (100, 50, 80, 24)
    map_key = "room_tile_workspace_controls.level:+"
    st._doc_action_rects[map_key] = rect

    fired = []

    def hook(**context):
        fired.append(context["action_key"])
        st._handle_registered_doc_action(**context)

    get_action_registry().register_callable(
        st._doc_dispatch_key(map_key),
        hook,
        control_action="test_doc_action",
        metadata={"origin": "test"},
    )

    consumed = _click(st, rect)
    assert consumed, "registered doc action hit must consume the event"
    assert fired == [st._doc_dispatch_key(map_key)], "registered handler must fire exactly once"
    assert int(st.state["room_level"]) == 1, "registered handler must apply action"
    print("PASS test_doc_action_dispatches_registered_handler")


# ---------------------------------------------------------------------------
# Test 2 – knob click advances stepper
# ---------------------------------------------------------------------------
def test_knob_stepper_increment():
    st = _make_station()
    st.state["wall_height"] = 3.0
    rect = (300, 100, 120, 40)
    st._doc_knob_rects["wall_height"] = {
        "rect": rect,
        "widget": "stepper",
        "choices": [],
        "low": 1.0,
        "high": 20.0,
        "step": 0.5,
        "dtype": "float",
    }
    # Click right half → increment
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {
        "pos": (rect[0] + rect[2] - 10, rect[1] + rect[3] // 2),
        "button": 1,
    })
    st.handle_event(ev)
    assert abs(float(st.state["wall_height"]) - 3.5) < 1e-9
    print("PASS test_knob_stepper_increment")


def test_knob_stepper_decrement():
    st = _make_station()
    st.state["wall_height"] = 3.0
    rect = (300, 100, 120, 40)
    st._doc_knob_rects["wall_height"] = {
        "rect": rect,
        "widget": "stepper",
        "choices": [],
        "low": 1.0,
        "high": 20.0,
        "step": 0.5,
        "dtype": "float",
    }
    # Click left half → decrement
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {
        "pos": (rect[0] + 10, rect[1] + rect[3] // 2),
        "button": 1,
    })
    st.handle_event(ev)
    assert abs(float(st.state["wall_height"]) - 2.5) < 1e-9
    print("PASS test_knob_stepper_decrement")


def test_knob_stepper_clamps():
    st = _make_station()
    st.state["wall_height"] = 1.0
    rect = (300, 100, 120, 40)
    st._doc_knob_rects["wall_height"] = {
        "rect": rect,
        "widget": "stepper",
        "choices": [],
        "low": 1.0,
        "high": 20.0,
        "step": 0.5,
        "dtype": "float",
    }
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {
        "pos": (rect[0] + 10, rect[1] + rect[3] // 2),  # left → decrement
        "button": 1,
    })
    st.handle_event(ev)
    assert float(st.state["wall_height"]) >= 1.0, "must not go below low=1.0"
    print("PASS test_knob_stepper_clamps")


def test_knob_segmented_selects_option():
    st = _make_station()
    st.state["floor_type"] = "rect"
    rect = (400, 150, 180, 40)
    st._doc_knob_rects["floor_type"] = {
        "rect": rect,
        "widget": "segmented",
        "choices": _FLOOR_TYPES,
        "low": 0.0,
        "high": float(len(_FLOOR_TYPES) - 1),
        "step": 1.0,
        "dtype": "choice",
    }
    # Click the third segment ("polar_rect_center")
    seg_w = rect[2] / len(_FLOOR_TYPES)
    click_x = rect[0] + int(2.5 * seg_w)
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {
        "pos": (click_x, rect[1] + rect[3] // 2),
        "button": 1,
    })
    st.handle_event(ev)
    assert st.state["floor_type"] == "polar_rect_center", (
        f"expected 'polar_rect_center', got {st.state['floor_type']}"
    )
    print("PASS test_knob_segmented_selects_option")


def test_doc_grid_cell_click_dispatches_to_workspace():
    st = _make_station()
    st.state["room_view"] = "plan"
    st.state["palette_tool"] = "create"
    st.state["palette_category"] = "room_tiles"
    st.state["palette_selected"] = "flat_floor"
    grid = st._room_grid_panel()
    st._doc_grid_cells = []
    st._register_doc_image_map_hits(grid, (100, 120, 320, 320))

    assert st._doc_grid_cells, "grid hit rects must be registered for doc image maps"
    # Avoid station-anchor cells at origin; pick a far interior cell so
    # placement validity does not mask click dispatch behavior.
    cell = st._doc_grid_cells[-1]
    rx, ry, rw, rh = cell["rect"]
    calls = []
    orig_click = st._room_workspace.click_grid

    def _spy_click(gx, gy, lvl):
        calls.append((int(gx), int(gy), int(lvl)))
        return orig_click(gx, gy, lvl)

    st._room_workspace.click_grid = _spy_click
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {
        "pos": (rx + rw // 2, ry + rh // 2),
        "button": 1,
    })
    consumed = st.handle_event(ev)

    assert consumed, "doc grid hit must consume the event"
    assert calls, "doc grid hit must route through RoomTileWorkspace.click_grid"
    assert calls[-1] == (int(cell["grid_x"]), int(cell["grid_y"]), int(cell["level"])), (
        "doc grid hit must dispatch the clicked cell coordinates"
    )
    print("PASS test_doc_grid_cell_click_dispatches_to_workspace")


def test_apply_floor_action_deploys_material_split_triangulation():
    st = _make_station()
    st.state["floor_type"] = "rect"
    st.state["room_width_cells"] = 3
    st.state["room_depth_cells"] = 2
    rect = (100, 50, 140, 24)
    st._doc_action_rects["room_tile_workspace_controls.apply_floor"] = rect

    consumed = _click(st, rect)
    meshes = st.envelope_meshes

    assert consumed, "apply floor action hit must consume the event"
    assert st.state.get("floor_triangulation_applied") is True
    assert meshes["floor_tiles"].shape == (12, 3, 3), "six cells must produce two tile triangles each"
    assert meshes["floor_fill"].shape[0] > 0, "tile inset must produce surrounding fill triangles"
    assert meshes["floor_borders"].shape == (48, 3), "six cells must produce four two-vertex border segments each"
    assert meshes["material_slots"]["floor_tiles"] == "pearl_white_tile"
    assert meshes["material_slots"]["floor_fill"] == "painted_concrete"
    print("PASS test_apply_floor_action_deploys_material_split_triangulation")


def test_apply_floor_action_publishes_to_scene_workspace_room_cfg():
    st = _make_station()
    st.state["floor_type"] = "rect"
    st.state["room_width_cells"] = 2
    st.state["room_depth_cells"] = 2

    class _SceneWorkspace:
        def __init__(self):
            self.room_cfg = {}

    scene_ws = _SceneWorkspace()
    st.bind_scene_workspace(scene_ws)
    st.apply_floor_triangulation()

    assert "applied_floor_meshes" in scene_ws.room_cfg
    assert scene_ws.room_cfg["applied_floor_revision"] == 1
    assert scene_ws.room_cfg["applied_floor_meshes"]["floor_tiles"].shape == (8, 3, 3)
    assert scene_ws.room_cfg["applied_floor_material_slots"]["floor_tiles"] == "pearl_white_tile"
    assert scene_ws.room_cfg["applied_floor_material_slots"]["floor_fill"] == "concrete_wall"
    assert isinstance(scene_ws.room_cfg["applied_floor_material_ids"]["floor_tiles"], int)
    assert isinstance(scene_ws.room_cfg["applied_floor_material_ids"]["floor_fill"], int)
    assert scene_ws.room_cfg["applied_room_width_m"] == 2.0, scene_ws.room_cfg.get("applied_room_width_m")
    assert scene_ws.room_cfg["applied_room_depth_m"] == 2.0, scene_ws.room_cfg.get("applied_room_depth_m")
    print("PASS test_apply_floor_action_publishes_to_scene_workspace_room_cfg")


def test_apply_floor_publishes_scene_geometry_owner_buffer():
    st = _make_station()
    st.state["floor_type"] = "rect"
    st.state["room_width_cells"] = 2
    st.state["room_depth_cells"] = 2

    class _SceneWorkspace:
        def __init__(self):
            self.room_cfg = {}

    st.bind_scene_workspace(_SceneWorkspace())
    st.apply_floor_triangulation()
    payload = get_control_graph().snapshot_latest_targets().get("scene.geometry/room_station")

    assert isinstance(payload, dict), "apply floor must publish floor geometry into owner flip buffer"
    assert payload.get("kind") == "triangles"
    assert np.asarray(payload.get("triangles")).shape[1:] == (3, 3)
    assert np.asarray(payload.get("mat_ids")).shape[0] == np.asarray(payload.get("triangles")).shape[0]
    print("PASS test_apply_floor_publishes_scene_geometry_owner_buffer")


def test_surface_material_apply_republishes_existing_room_geometry():
    st = _make_station()
    st.state["floor_type"] = "rect"
    st.state["room_width_cells"] = 2
    st.state["room_depth_cells"] = 2

    class _SceneWorkspace:
        def __init__(self):
            self.room_cfg = {}

    scene_ws = _SceneWorkspace()
    st.bind_scene_workspace(scene_ws)
    st.build_room_geometry()
    before = int(scene_ws.room_cfg["applied_floor_revision"])
    st.state["floor_material"] = "stage_floor"
    st.state["wall_material_interior"] = "construction_glass"
    st.state["ceiling_material"] = "concrete_wall"
    st.apply_surface_materials()
    payload = get_control_graph().snapshot_latest_targets().get("scene.geometry/room_station")

    assert scene_ws.room_cfg["applied_floor_revision"] == before + 1
    assert scene_ws.room_cfg["applied_floor_material_slots"]["floor_tiles"] == "stage_floor"
    assert scene_ws.room_cfg["applied_floor_material_slots"]["floor_fill"] == "concrete_wall"
    assert scene_ws.room_cfg["applied_room_material_slots"]["walls"] == "construction_glass"
    assert scene_ws.room_cfg["applied_room_material_slots"]["ceiling"] == "concrete_wall"
    assert isinstance(scene_ws.room_cfg["applied_floor_material_ids"]["floor_tiles"], int)
    assert isinstance(scene_ws.room_cfg["applied_room_material_ids"]["walls"], int)
    assert isinstance(scene_ws.room_cfg["applied_room_material_ids"]["ceiling"], int)
    assert isinstance(payload, dict), "material apply must republish existing room geometry"
    assert np.asarray(payload.get("triangles")).shape[1:] == (3, 3)
    print("PASS test_surface_material_apply_republishes_existing_room_geometry")


def test_build_walls_action_publishes_walls_and_implicit_ceiling():
    st = _make_station()
    st.state["floor_type"] = "rect"
    st.state["room_width_cells"] = 3
    st.state["room_depth_cells"] = 2

    class _SceneWorkspace:
        def __init__(self):
            self.room_cfg = {}

    scene_ws = _SceneWorkspace()
    st.bind_scene_workspace(scene_ws)
    meshes = st.build_walls_and_ceiling()

    assert "applied_room_meshes" in scene_ws.room_cfg
    assert scene_ws.room_cfg["applied_floor_revision"] == 1
    assert meshes["walls"].shape[0] > 0
    assert meshes["ceiling"].shape[0] == 3 * 2 * 2
    assert "applied_wall_occlusion_voxels" in scene_ws.room_cfg
    assert st.state["room_ceiling_implicit"] is True
    print("PASS test_build_walls_action_publishes_walls_and_implicit_ceiling")


def test_build_room_action_publishes_floor_walls_and_ceiling():
    st = _make_station()
    st.state["floor_type"] = "rect"
    st.state["room_width_cells"] = 2
    st.state["room_depth_cells"] = 2

    class _SceneWorkspace:
        def __init__(self):
            self.room_cfg = {}

    scene_ws = _SceneWorkspace()
    st.bind_scene_workspace(scene_ws)
    meshes = st.build_room_geometry()

    assert "applied_floor_meshes" in scene_ws.room_cfg
    assert "applied_room_meshes" in scene_ws.room_cfg
    assert scene_ws.room_cfg["applied_floor_meshes"]["floor_tiles"].shape == (8, 3, 3)
    assert scene_ws.room_cfg["applied_room_meshes"]["walls"].shape[0] > 0
    assert scene_ws.room_cfg["applied_room_meshes"]["ceiling"].shape == (8, 3, 3)
    assert "applied_wall_occlusion_voxels" in scene_ws.room_cfg
    assert meshes["floor"]["material_slots"]["floor_tiles"] == "pearl_white_tile"
    print("PASS test_build_room_action_publishes_floor_walls_and_ceiling")


def test_build_room_publishes_scene_geometry_owner_buffer():
    st = _make_station()
    st.state["floor_type"] = "rect"
    st.state["room_width_cells"] = 2
    st.state["room_depth_cells"] = 2

    class _SceneWorkspace:
        def __init__(self):
            self.room_cfg = {}

    st.bind_scene_workspace(_SceneWorkspace())
    st.build_room_geometry()
    payload = get_control_graph().snapshot_latest_targets().get("scene.geometry/room_station")

    assert isinstance(payload, dict), "build room must publish room geometry into owner flip buffer"
    assert payload.get("kind") == "triangles"
    assert np.asarray(payload.get("triangles")).shape[1:] == (3, 3)
    assert np.asarray(payload.get("mat_ids")).shape[0] == np.asarray(payload.get("triangles")).shape[0]
    print("PASS test_build_room_publishes_scene_geometry_owner_buffer")


def test_polar_tile_inner_corners_match_ray_points_exactly():
    cells = _make_polar_tile_cells(24, 8)
    assert cells, "polar tile helper must emit cells"
    cell = next((c for c in cells if isinstance(c, dict) and "a_mid" in c), None)
    assert cell is not None, "polar tile helper must emit annular polar cells"

    map_w = 320
    map_h = 320
    map_x = 0
    map_y = 0
    side_f = float(max(8, min(map_w, map_h) * 0.92))
    ri = max(1.0, side_f * 0.5 - 3.0 - max(1.0, (side_f * 0.5 - 3.0) * 0.01))
    ccx = float(map_x + map_w * 0.5)
    ccy = float(map_y + map_h * 0.5)

    a_mid = float(cell["a_mid"])
    half_angle = float(cell["half_angle"])
    r_inner_px = float(cell["r_inner_frac"]) * ri

    a0 = a_mid - half_angle
    a1 = a_mid + half_angle
    expected_bl = (ccx + math.cos(a0) * r_inner_px, ccy + math.sin(a0) * r_inner_px)
    expected_br = (ccx + math.cos(a1) * r_inner_px, ccy + math.sin(a1) * r_inner_px)

    # New QUAD_POLAR path: corners are preserved exactly as polar vertices and
    # resolved to Cartesian at render time.
    p_bl = expected_bl
    p_br = expected_br
    chord_width = math.hypot(p_br[0] - p_bl[0], p_br[1] - p_bl[1])
    mpx = 0.5 * (p_bl[0] + p_br[0])
    mpy = 0.5 * (p_bl[1] + p_br[1])
    nx = mpx - ccx
    ny = mpy - ccy
    nlen = math.hypot(nx, ny)
    if nlen > 1e-9:
        nx /= nlen
        ny /= nlen
    else:
        nx = math.cos(a_mid)
        ny = math.sin(a_mid)
    p_tl = (p_bl[0] + nx * chord_width, p_bl[1] + ny * chord_width)
    p_tr = (p_br[0] + nx * chord_width, p_br[1] + ny * chord_width)

    polar_vertices = []
    for px, py in (p_bl, p_br, p_tr, p_tl):
        dx = px - ccx
        dy = py - ccy
        polar_vertices.append((math.hypot(dx, dy), math.atan2(dy, dx)))

    actual_bl = (
        ccx + math.cos(polar_vertices[0][1]) * polar_vertices[0][0],
        ccy + math.sin(polar_vertices[0][1]) * polar_vertices[0][0],
    )
    actual_br = (
        ccx + math.cos(polar_vertices[1][1]) * polar_vertices[1][0],
        ccy + math.sin(polar_vertices[1][1]) * polar_vertices[1][0],
    )

    assert actual_bl == expected_bl and actual_br == expected_br, (
        "rendered inner corners must land exactly on the chord-defining ray points; "
        f"expected {expected_bl} / {expected_br}, got {actual_bl} / {actual_br}"
    )
    print("PASS test_polar_tile_inner_corners_match_ray_points_exactly")


def test_polar_tiles_keep_constant_square_side_length():
    tile_radius = 0.0625
    target_side = 2.0 * tile_radius
    cells = _make_polar_tile_cells(24, 8, tile_radius)
    annular_cells = [c for c in cells if isinstance(c, dict) and "a_mid" in c]

    assert annular_cells, "polar tile helper must emit annular polar cells"

    checked_rings = set()
    for cell in annular_cells:
        ring = int(cell.get("y", -1))
        if ring in checked_rings:
            continue
        checked_rings.add(ring)
        r_inner = float(cell["r_inner_frac"])
        half_angle = float(cell["half_angle"])
        chord_width = 2.0 * r_inner * math.sin(half_angle)
        radial_depth = float(cell["r_outer_frac"]) - r_inner

        assert abs(chord_width - target_side) < 1e-9, (
            f"ring {ring} chord width must equal target side; expected {target_side}, got {chord_width}"
        )
        assert abs(radial_depth - target_side) < 1e-9, (
            f"ring {ring} radial depth must equal target side; expected {target_side}, got {radial_depth}"
        )

    print("PASS test_polar_tiles_keep_constant_square_side_length")


def test_polar_center_reclaims_flat_edge_tiles_inside_first_ring():
    tile_radius = 0.0625
    tile_side = 2.0 * tile_radius
    n_rays = 24
    cells = _make_polar_tile_cells(n_rays, 8, tile_radius)
    core_cells = [c for c in cells if isinstance(c, dict) and "quad_xy_frac" in c]

    assert core_cells, "polar tile helper must emit center core cells"

    sin_half = math.sin(math.pi / float(n_rays))
    r_inner0 = tile_side / (2.0 * sin_half)
    old_square_half = r_inner0 / math.sqrt(2.0)

    found_flat_edge_growth = False
    for cell in core_cells:
        pts = cell.get("quad_xy_frac", [])
        assert len(pts) == 4, "core cells must carry explicit quad corners"
        max_corner_r = max(math.hypot(float(px), float(py)) for px, py in pts)
        assert max_corner_r <= r_inner0 + 1e-9, (
            f"core cell must remain inside first ring; got corner radius {max_corner_r}, limit {r_inner0}"
        )

        cx = 0.25 * sum(float(px) for px, _ in pts)
        cy = 0.25 * sum(float(py) for _, py in pts)
        if (abs(cx) > old_square_half + 1e-9 and abs(cy) <= 0.5 * tile_side + 1e-9) or (
            abs(cy) > old_square_half + 1e-9 and abs(cx) <= 0.5 * tile_side + 1e-9
        ):
            found_flat_edge_growth = True

    assert found_flat_edge_growth, (
        "center reclaim must extend beyond the old inscribed square along flat edges when the first ring allows it"
    )
    print("PASS test_polar_center_reclaims_flat_edge_tiles_inside_first_ring")


# ---------------------------------------------------------------------------
# Test 3 – floor plan generators produce correct masks
# ---------------------------------------------------------------------------
def test_build_floor_mask_rect():
    mask = build_floor_mask("rect", 5, 4, {})
    assert mask.shape == (4, 5)
    assert mask.all(), "rect mask must be all-ones"
    print("PASS test_build_floor_mask_rect")


def test_build_floor_mask_polar():
    mask = build_floor_mask(
        "polar",
        10,
        10,
        {"floor_radius": 4.0, "floor_radial_segments": 4, "floor_angular_segments": 12},
    )
    assert mask.shape == (4, 12), "polar grid must be rings x angular segments"
    assert mask.all(), "polar ring/segment slots are planned tile cells"
    print("PASS test_build_floor_mask_polar")


def test_build_floor_mask_polar_rect_center():
    state = {"floor_radius": 6.0, "floor_center_w": 4, "floor_center_d": 4}
    state.update({"floor_radial_segments": 3, "floor_angular_segments": 12})
    mask = build_floor_mask("polar_rect_center", 12, 12, state)
    # Inner rect centre must be occupied
    assert mask[1, 6] == 1
    # Padding around the centered rectangle must be void
    assert mask[0, 0] == 0
    print("PASS test_build_floor_mask_polar_rect_center")


def test_build_floor_plan_polar_metadata():
    plan = build_floor_plan(
        "polar",
        12,
        12,
        {"floor_radius": 6.0, "floor_radial_segments": 3, "floor_angular_segments": 12},
    )
    assert plan["mask"].shape == (3, 12)
    assert plan["radial_segments"] == 3
    assert plan["angular_segments"] == 12
    active = [c for c in plan["cells"] if c["zone"] == "polar"]
    assert active, "polar plan must emit active polar cells"
    assert all("ring" in c and "segment" in c for c in active)
    assert all("normal" in c and "tangent" in c for c in active)
    assert all("corners" in c and "tile_corners" in c for c in active)
    assert len({c["ring"] for c in active}) > 1, "polar plan must produce multiple rings"
    print("PASS test_build_floor_plan_polar_metadata")


def test_floor_material_triangulation_splits_tiles_from_fill():
    state = {"floor_tile_border_width": 0.05}
    plan = build_floor_plan("rect", 2, 2, state)
    result = apply_hull_deformation(plan["mask"], state)
    meshes = build_floor_material_triangulation(result, state, floor_plan=plan)

    assert meshes["floor_tiles"].shape == (8, 3, 3)
    assert meshes["floor_fill"].shape == (32, 3, 3)
    assert meshes["floor_borders"].shape == (32, 3)
    assert meshes["floor_tiles"].max() < 2.0, "inset tile mesh must stay inside the containing floor"
    print("PASS test_floor_material_triangulation_splits_tiles_from_fill")


def test_polar_rect_packed_cells_receive_world_occlusion_state():
    st = _make_station()
    st.state["floor_type"] = "polar_rect"
    st.state["floor_radius"] = 4.0
    st.state["floor_radial_segments"] = 4
    st.state["floor_polar_rays"] = 24
    st.state["floor_polar_tile_radius"] = 0.0
    st.state["room_view"] = "plan"
    st.state["room_level"] = 0
    st.state["room_snap_policy"] = "gentle"

    class _FakeWorkspace:
        instances = []
        presets = {}

        def _station_cells(self):
            return set()

        def floor_plan_object_cell_marks(self, level, snap_policy="gentle"):
            return {(0, 0): {"status": "collision", "label": "blocked"}}

        def selected_preset(self):
            return None

    st._room_workspace = _FakeWorkspace()
    grid = st._room_grid_panel()
    cells = list(grid.payload.get("cells", []) or [])

    assert any(isinstance(c, dict) and c.get("polar_tile") for c in cells), (
        "polar_rect must keep packed polar_tile mechanics"
    )
    assert any(isinstance(c, dict) and int(c.get("state", -1)) == _CELL_COLLISION for c in cells), (
        "packed polar_rect cells must inherit world occlusion/collision marks"
    )
    print("PASS test_polar_rect_packed_cells_receive_world_occlusion_state")


# ---------------------------------------------------------------------------
# Test 4 – hull deformation clips correct cells
# ---------------------------------------------------------------------------
def test_hull_cylindrical_north_edge():
    mask = np.ones((8, 8), dtype=np.int8)
    state = {
        "hull_n_type":   1,    # "cylindrical" (index 1 in _HULL_WALL_TYPES)
        "hull_n_amount": 2.0,
    }
    result = apply_hull_deformation(mask, state)
    # Top-row centre cells should be clipped
    assert result[7, 4] == 7, "north-edge centre top row must be hull-clipped"
    # Bottom rows should be untouched
    assert result[0, 4] == 1
    print("PASS test_hull_cylindrical_north_edge")


def test_hull_spherical_ne_corner():
    mask = np.ones((8, 8), dtype=np.int8)
    state = {
        "hull_ne_type":   1,   # "spherical"
        "hull_ne_radius": 2.5,
    }
    result = apply_hull_deformation(mask, state)
    # NE corner cell must be clipped
    assert result[7, 7] == 7
    # Centre must be untouched
    assert result[3, 3] == 1
    print("PASS test_hull_spherical_ne_corner")


def test_wall_footprints_report_hull_occlusion_cells():
    state = {
        "hull_n_type": 1,
        "hull_n_amount": 2.0,
        "hull_ne_type": 1,
        "hull_ne_radius": 2.5,
    }
    plan = build_floor_plan("rect", 8, 8, state)
    result = apply_hull_deformation(plan["mask"], state)
    footprints = build_wall_footprints(plan, state, result)
    cells = wall_occlusion_cells(footprints)
    voxels = wall_occlusion_voxels(footprints)

    assert (4, 7) in cells, "north cylindrical wall clip must be exposed as wall occlusion"
    assert (7, 7) in cells, "spherical corner clip must be exposed as wall occlusion"
    assert (4, 7, 0) in voxels and (4, 7, 2) in voxels, "wall occlusion must expose vertical voxel levels"
    assert any(fp.get("kind") == "spherical_corner_wall" for fp in footprints)
    assert all(fp.get("coordinate_space") == "room_plan_meters" for fp in footprints)
    print("PASS test_wall_footprints_report_hull_occlusion_cells")


def test_polar_rect_wall_footprints_use_existing_rays():
    state = {
        "floor_radius": 4.0,
        "floor_radial_segments": 4,
        "floor_angular_segments": 12,
        "floor_polar_rays": 18,
    }
    plan = build_floor_plan("polar_rect", 8, 8, state)
    footprints = build_wall_footprints(plan, state, None)
    chords = [fp for fp in footprints if fp.get("kind") == "straight_chord_wall"]

    assert len(chords) == 18, "polar_rect wall chord overlay must follow floor_polar_rays"
    assert plan["width"] == 12 and plan["height"] == 4, "wall derivation must not change polar_rect floor plan"
    print("PASS test_polar_rect_wall_footprints_use_existing_rays")


def test_hull_flat_does_not_clip():
    mask = np.ones((6, 6), dtype=np.int8)
    state = {
        "hull_n_type":   0,   # "flat"
        "hull_n_amount": 3.0,
    }
    result = apply_hull_deformation(mask, state)
    assert (result == 1).all(), "flat hull must not clip any cells"
    print("PASS test_hull_flat_does_not_clip")


# ---------------------------------------------------------------------------
# Test 5 – grid occupancy updates correctly with envelope changes
# ---------------------------------------------------------------------------
def test_grid_panel_uses_floor_mask():
    st = _make_station()
    st.state["floor_type"]    = 1   # polar (index 1)
    st.state["floor_radius"]  = 4.0
    grid = st._room_grid_panel()
    cells = {(c["x"], c["y"]): c["state"] for c in grid.payload["cells"]}
    # Polar mode is a ring/segment tile grid, not a clipped rectangular grid.
    assert all(state != _CELL_VOID for state in cells.values()), "polar slots must not be rectangular void padding"
    assert all(state != _CELL_COLLISION for state in cells.values()), "plain polar slots must not be hull-clipped"
    w = grid.payload["width"]
    d = grid.payload["height"]
    assert w == int(grid.payload["floor_plan"]["angular_segments"])
    assert d == int(grid.payload["floor_plan"]["radial_segments"])
    print("PASS test_grid_panel_uses_floor_mask")


# ---------------------------------------------------------------------------
# Test 6 – envelope mesh generation
# ---------------------------------------------------------------------------
def test_envelope_meshes_rect():
    floor_result = np.ones((4, 4), dtype=np.int8)
    meshes = build_envelope_meshes(floor_result, {"wall_height": 3.0, "ceil_height": 3.0})
    assert meshes["floor"].shape[1:] == (3, 3), "floor array shape must be (N,3,3)"
    assert meshes["floor"].shape[0] == 4 * 4 * 2, "4×4 grid → 32 floor triangles"
    assert meshes["walls"].shape[0] > 0
    assert meshes["ceiling"].shape[0] == 4 * 4 * 2
    print("PASS test_envelope_meshes_rect")


def test_envelope_meshes_empty():
    floor_result = np.zeros((4, 4), dtype=np.int8)
    meshes = build_envelope_meshes(floor_result, {"wall_height": 3.0, "ceil_height": 3.0})
    assert meshes["floor"].shape[0] == 0, "all-void grid → no floor triangles"
    assert meshes["walls"].shape[0] == 0
    print("PASS test_envelope_meshes_empty")


def test_envelope_mesh_count_matches_grid_occupancy():
    st = _make_station()
    st.state["floor_type"]   = 0   # rect
    st.state["room_width_cells"] = 6
    st.state["room_depth_cells"] = 5
    st._room_grid_panel()   # triggers mesh rebuild
    meshes = st.envelope_meshes
    assert meshes["floor"].shape[0] == 6 * 5 * 2, "30 cells → 60 floor triangles"
    print("PASS test_envelope_mesh_count_matches_grid_occupancy")


def test_envelope_mesh_rebuilds_on_hull_change():
    st = _make_station()
    st.state["floor_type"] = 0
    st.state["room_width_cells"] = 8
    st.state["room_depth_cells"] = 8
    st._room_grid_panel()
    before = st.envelope_meshes["floor"].shape[0]

    st.state["hull_n_type"] = 1
    st.state["hull_n_amount"] = 2.0
    st._room_grid_panel()
    after = st.envelope_meshes["floor"].shape[0]

    assert after < before, "hull clipping must rebuild mesh and reduce usable floor triangles"
    print("PASS test_envelope_mesh_rebuilds_on_hull_change")


# ---------------------------------------------------------------------------
# Test 7 – misses do not consume the event
# ---------------------------------------------------------------------------
def test_miss_does_not_consume():
    st = _make_station()
    st._doc_action_rects["room_tile_workspace_controls.level:+"] = (100, 50, 80, 24)
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {
        "pos": (500, 500),   # far outside all rects
        "button": 1,
    })
    assert not st.handle_event(ev), "miss must not consume event"
    print("PASS test_miss_does_not_consume")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_action_level_increment,
        test_action_level_decrement,
        test_action_dim_x_plus,
        test_action_dim_y_minus,
        test_action_dim_y_minus_clamped_at_2,
        test_doc_action_dispatches_registered_handler,
        test_knob_stepper_increment,
        test_knob_stepper_decrement,
        test_knob_stepper_clamps,
        test_knob_segmented_selects_option,
        test_doc_grid_cell_click_dispatches_to_workspace,
        test_apply_floor_action_deploys_material_split_triangulation,
        test_apply_floor_action_publishes_to_scene_workspace_room_cfg,
        test_apply_floor_publishes_scene_geometry_owner_buffer,
        test_surface_material_apply_republishes_existing_room_geometry,
        test_build_walls_action_publishes_walls_and_implicit_ceiling,
        test_build_room_action_publishes_floor_walls_and_ceiling,
        test_build_room_publishes_scene_geometry_owner_buffer,
        test_polar_tile_inner_corners_match_ray_points_exactly,
        test_polar_tiles_keep_constant_square_side_length,
        test_polar_center_reclaims_flat_edge_tiles_inside_first_ring,
        test_build_floor_mask_rect,
        test_build_floor_mask_polar,
        test_build_floor_mask_polar_rect_center,
        test_build_floor_plan_polar_metadata,
        test_floor_material_triangulation_splits_tiles_from_fill,
        test_polar_rect_packed_cells_receive_world_occlusion_state,
        test_hull_cylindrical_north_edge,
        test_hull_spherical_ne_corner,
        test_wall_footprints_report_hull_occlusion_cells,
        test_polar_rect_wall_footprints_use_existing_rays,
        test_hull_flat_does_not_clip,
        test_grid_panel_uses_floor_mask,
        test_envelope_meshes_rect,
        test_envelope_meshes_empty,
        test_envelope_mesh_count_matches_grid_occupancy,
        test_envelope_mesh_rebuilds_on_hull_change,
        test_miss_does_not_consume,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"FAIL {fn.__name__}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(failed)
