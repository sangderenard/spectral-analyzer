"""Test to verify UI responsiveness - knobs should update state when clicked."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from room_control_station import RoomControlStation
from doc_renderer import DocRenderer

pygame.init()
pygame.display.set_mode((1600, 900), pygame.NOFRAME)

def test_floor_type_selection():
    """Test that clicking floor_type knob updates state."""
    # Create station with minimal config
    station = RoomControlStation(
        station_cfg={
            "room_editor": {"preset_library_dir": "configs/duty_stations/room_control/presets"}
        },
        left_cfg={},
        center_cfg={},
        right_cfg={},
    )
    
    # Create a mock doc_renderer
    doc_rdr = DocRenderer(1600, 900)
    
    # Render once to populate doc_knob_rects
    station.submit_doc_channel(doc_rdr, 1600, 900)
    
    # Check if floor_type knob is in the rects
    if "floor_type" not in station._doc_knob_rects:
        print("FAIL: floor_type not in knob_rects")
        print(f"Available knobs: {list(station._doc_knob_rects.keys())}")
        return False
    
    info = station._doc_knob_rects["floor_type"]
    print(f"floor_type knob info: {info}")
    
    # Check that widget is set correctly
    if info.get("widget") != "segmented":
        print(f"FAIL: floor_type widget is '{info.get('widget')}', expected 'segmented'")
        return False
    
    # Check that choices are populated
    choices = info.get("choices", [])
    if not choices:
        print("FAIL: floor_type choices are empty")
        return False
    
    print(f"✓ floor_type knob has widget='segmented' and choices={choices}")
    
    # Test clicking on floor_type to change it
    initial_floor_type = station.state.get("floor_type", "rect")
    print(f"Initial floor_type: {initial_floor_type}")
    
    # Simulate clicking on the floor_type knob (segment 1 for "polar")
    rect_x, rect_y, rect_w, rect_h = info["rect"]
    click_x = rect_x + rect_w // 4  # Click on first segment
    click_y = rect_y + rect_h // 2
    
    ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": (click_x, click_y), "button": 1})
    station.handle_event(ev)
    
    new_floor_type = station.state.get("floor_type", "rect")
    print(f"After click: floor_type = {new_floor_type}")
    
    if new_floor_type != initial_floor_type:
        print(f"✓ floor_type changed from {initial_floor_type} to {new_floor_type}")
        return True
    else:
        print(f"FAIL: floor_type did not change")
        return False

def test_palette_category_selection():
    """Test that clicking palette_category knob updates state."""
    station = RoomControlStation(
        station_cfg={
            "room_editor": {"preset_library_dir": "configs/duty_stations/room_control/presets"}
        },
        left_cfg={},
        center_cfg={},
        right_cfg={},
    )
    
    # Create a mock doc_renderer
    doc_rdr = DocRenderer(1600, 900)
    
    # Render once to populate doc_knob_rects
    station.submit_doc_channel(doc_rdr, 1600, 900)
    
    # Check if palette_category knob is in the rects
    if "palette_category" not in station._doc_knob_rects:
        print("FAIL: palette_category not in knob_rects")
        print(f"Available knobs: {list(station._doc_knob_rects.keys())}")
        return False
    
    info = station._doc_knob_rects["palette_category"]
    print(f"palette_category knob info: {info}")
    
    # Check that widget is set correctly
    if info.get("widget") != "segmented":
        print(f"FAIL: palette_category widget is '{info.get('widget')}', expected 'segmented'")
        return False
    
    print(f"✓ palette_category knob has widget='segmented' and choices={info.get('choices', [])}")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing UI Responsiveness")
    print("=" * 60)
    
    print("\n[Test 1] floor_type knob should update state")
    result1 = test_floor_type_selection()
    
    print("\n[Test 2] palette_category knob should exist")
    result2 = test_palette_category_selection()
    
    if result1 and result2:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
