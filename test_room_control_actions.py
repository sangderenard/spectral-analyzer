"""Smoke tests for doc-channel action dispatch in RoomControlStation.

Run with:
    python test_room_control_actions.py

No OpenGL context required.  A pygame display surface is created only to
satisfy pygame.event.Event construction; no window is shown.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pygame
from controls import get_action_registry

# ---------------------------------------------------------------------------
# Module-level helpers under test
# ---------------------------------------------------------------------------
from room_control_station import (
    RoomControlStation,
    build_floor_mask,
    build_floor_plan,
    apply_hull_deformation,
    build_envelope_meshes,
    _FLOOR_TYPES,
    _HULL_WALL_TYPES,
    _HULL_CORNER_TYPES,
    _CELL_VOID,
    _CELL_COLLISION,
    _CELL_EMPTY,
)

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
    st.state["floor_type"] = 0  # "rect"
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
    assert int(st.state["floor_type"]) == 2, f"expected index 2, got {st.state['floor_type']}"
    print("PASS test_knob_segmented_selects_option")


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
        test_build_floor_mask_rect,
        test_build_floor_mask_polar,
        test_build_floor_mask_polar_rect_center,
        test_build_floor_plan_polar_metadata,
        test_hull_cylindrical_north_edge,
        test_hull_spherical_ne_corner,
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
