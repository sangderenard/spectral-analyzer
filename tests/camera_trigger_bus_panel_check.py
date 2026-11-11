"""Actual CPU Pygame mixer render/events, outside conftest's GUI stubs.

Run directly: python tests/camera_trigger_bus_panel_check.py
No OpenGL context, audio initialization, service changes or visible window.
"""
from pathlib import Path
import os
import sys
from types import SimpleNamespace

os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from camera_software.exposure_timing import CameraExposureScheduler
from camera_software.trigger_bus import (
    make_camera_trigger_graph, make_camera_trigger_panel, resolve_camera_trigger_bus,
)


def main():
    pygame.display.init()
    pygame.font.init()
    try:
        pygame.display.set_mode((1, 1))
        graph = make_camera_trigger_graph(["Key", "Fill", "Rim"], bus_keys=["Bus"])
        graph.set_weight("Flash", "Bus", 1)
        graph.set_weight("Bus", "Key", 1)
        graph.set_weight("Bus", "Fill", 0.5)
        patch, view = make_camera_trigger_panel(graph)
        scene = SimpleNamespace(
            camera_light_burst=SimpleNamespace(stages=4, exposure_time_s=0.02,
                enabled=True, energy_scale=2.0, duty_cycle=0.5),
            image_plate=SimpleNamespace(shutter_mode="open", shutter_open=1.0),
        )
        frame = CameraExposureScheduler(camera_id="room-camera").begin_frame(scene)

        def signal():
            return resolve_camera_trigger_bus(frame, graph, ["Key"]).signals["Key"]

        surface = view.render_surface(patch, 1000, 620)
        assert signal() == (0.5, 0.5, 0, 0)
        keys = graph.node_keys()
        col, row = keys.index("Bus"), keys.index("Key")
        position = (view._vp_x + col * view._cell_size + view._cell_size // 2,
                    view._vp_y + row * view._cell_size + view._cell_size // 2)
        right_click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=position)
        assert view.handle_event(right_click, patch, 0, 0)
        assert signal() == (0, 0, 0, 0)
        view.render_surface(patch, 1000, 620)
        assert signal() == (0, 0, 0, 0), "redraw must not reinstall a deleted wire"

        click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position)
        release = pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=position)
        assert not view.handle_event(click, patch, 0, 0)
        view.handle_event(release, patch, 0, 0)
        assert view.handle_event(click, patch, 0, 0)
        view.handle_event(release, patch, 0, 0)
        assert signal() == (0.5, 0.5, 0, 0)
        surface = view.render_surface(patch, 1000, 620)
        output = Path("build/camera-trigger-bus-panel.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surface, str(output))
        print("PASS: real RoutingGridView rendering, right-click disconnect, redraw persistence, double-click reconnect, GraphSolver signal equality")
        print(output.resolve())
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
