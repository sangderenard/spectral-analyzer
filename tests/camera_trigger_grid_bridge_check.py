"""Real headless Pygame grid, outside pytest's GUI stubs."""
from pathlib import Path
import os
import sys
import subprocess
import json

video_driver_before = os.environ.get("SDL_VIDEODRIVER")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_software.trigger_grid_bridge import CameraTriggerGrid
from routing_engine import RoutingGraph
import pygame


def main():
    assert os.environ.get("SDL_VIDEODRIVER") == video_driver_before
    assert not pygame.display.get_init()
    grid = CameraTriggerGrid(["shop:17"], ["rack:center"])
    first = grid.frame()
    assert first["image"].startswith("data:image/png;base64,")
    assert first["graph"]["nodes"] == ["Sync", "Flash", "Gate", "rack:center", "shop:17", "mod:1", "mod:2", "mod:3"]
    keys = grid.graph.node_keys()
    x = grid.view._vp_x + keys.index("Flash") * grid.view._cell_size + grid.view._cell_size // 2
    y = grid.view._vp_y + keys.index("rack:center") * grid.view._cell_size + grid.view._cell_size // 2
    grid.event("down", x, y)
    grid.event("up", x, y)
    grid.event("down", x, y)
    result = grid.event("up", x, y)
    assert RoutingGraph.from_dict(result["graph"]).get_weight("Flash", "rack:center") == 1
    grid.graph.set_weight("rack:center", "shop:17", 0.5)
    grid.graph.set_delay_s("Flash", "rack:center", .005)
    pulse=grid.pulse()
    assert pulse["signals"]["shop:17"][:6] == [0]*6
    assert pulse["signals"]["shop:17"][6:16] == [0.5]*10
    mod={"key":"mod:1","enabled":True,"kind":"ramp_down","frequency_hz":0,
         "phase_deg":0,"tension":1,"attack_ms":0,"decay_ms":0,"sustain":1,
         "hold_ms":10,"release_ms":0,"repeat_count":2,
         "repeat_interval_ms":10,"level":1}
    mod_grid=CameraTriggerGrid(["shop:17"], generators=[mod])
    mod_grid.graph.set_weight("mod:1", "shop:17", 1)
    mod_pulse=mod_grid.pulse()["signals"]["shop:17"]
    assert mod_pulse[1] == 1 and mod_pulse[2] > 0 and mod_pulse[11] == 1
    wide=CameraTriggerGrid([f"studio:{i}" for i in range(20)])
    wide.frame()
    wide.event("wheel", wheel_y=-1, shift=True)
    assert wide.view._scroll_col == 1
    grid.event("down", x, y, button=3)
    assert grid.graph.get_weight("Flash", "rack:center") == 0
    saved = CameraTriggerGrid(["shop:17"]).frame()["graph"]
    assert CameraTriggerGrid(["shop:17"], saved=saved).frame()["graph"] == saved
    colour=CameraTriggerGrid(["shop:17", "filter:studio:0:r"], saved=saved)
    assert colour.graph.node_keys()[-1] == "filter:studio:0:r"
    colour.graph.set_weight("mod:1", "filter:studio:0:r", 1)
    assert "filter:studio:0:r" in colour.pulse()["signals"]
    intensity = {**mod, "key":"mod:2", "kind":"sine", "frequency_hz":30}
    colour.generators = (mod, intensity)
    colour.graph.set_weight("mod:2", "shop:17", 1)
    routed = colour.pulse()["signals"]
    assert routed["filter:studio:0:r"][1:11] != routed["shop:17"][1:11]
    try:
        CameraTriggerGrid(["shop:18"], saved=saved)
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("stale material receiver was accepted")
    assert os.environ.get("SDL_VIDEODRIVER") == video_driver_before
    assert not pygame.display.get_init()
    assert pygame.display.get_surface() is None
    # Render workers inherit the service environment after the grid is opened.
    inherited = subprocess.check_output([sys.executable, "-c",
        "import json, os; print(json.dumps(os.environ.get('SDL_VIDEODRIVER')))"])
    assert json.loads(inherited) == video_driver_before
    print("PASS: original N×N grid image, event editing, serialization and receiver identity")


if __name__ == "__main__":
    main()
