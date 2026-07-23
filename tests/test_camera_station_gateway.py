from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

from camera_station import CameraStationMenu


def test_default_pluck_room_places_camera_station_jar_and_specimen():
    # demo_pluck_gl is the simulation coordinator, not a library module.  Probe
    # its authored default world in a clean interpreter so pytest's pygame/GL
    # test doubles cannot leak into (or redefine) the coordinator environment.
    probe = r'''
import json
from demo_pluck_gl import _build_default_scene

workspace = _build_default_scene()
specimen = workspace.cameras()[0]
print(json.dumps({
    "station_types": [item.station_type for item in workspace.duty_stations()],
    "enclosure_count": len(workspace.enclosures()),
    "glass_alpha": workspace.enclosures()[0].glass_color[3],
    "camera_count": len(workspace.cameras()),
    "mesh_id": specimen.mesh_id,
    "focal_mm": specimen.focal_mm,
    "interaction_radius": specimen.interaction_radius,
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload == {
        "station_types": ["room_control", "fabricator", "camera_designer"],
        "enclosure_count": 1,
        "glass_alpha": 0.42,
        "camera_count": 1,
        "mesh_id": "camera_box",
        "focal_mm": 82.5,
        "interaction_radius": 0.0,
    }


def test_camera_station_menu_publishes_separate_navigation_requests(monkeypatch):
    import pygame

    # Some neighboring headless tests install a deliberately tiny pygame
    # module.  Supply only the event constant this menu contract needs.
    monkeypatch.setattr(pygame, "MOUSEBUTTONDOWN", 1025, raising=False)
    menu = CameraStationMenu()
    menu.show_hud(True)
    menu._doc_action_rects = {
        "camera_station_gateway.open_optics": (10, 10, 100, 24),
        "camera_station_gateway.open_engine": (10, 40, 100, 24),
    }

    optics = SimpleNamespace(
        type=pygame.MOUSEBUTTONDOWN, button=1, pos=(20, 20)
    )
    assert menu.handle_event(optics)
    assert menu.consume_open_optics_request()
    assert not menu.consume_open_optics_request()

    engine = SimpleNamespace(
        type=pygame.MOUSEBUTTONDOWN, button=1, pos=(20, 50)
    )
    assert menu.handle_event(engine)
    assert menu.consume_open_engine_request()
    assert not menu.consume_open_engine_request()
