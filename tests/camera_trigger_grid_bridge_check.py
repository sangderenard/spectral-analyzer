"""Real headless Pygame grid, outside pytest's GUI stubs."""
from pathlib import Path
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_software.trigger_grid_bridge import CameraTriggerGrid
from routing_engine import RoutingGraph


def main():
    grid = CameraTriggerGrid(["shop:17"], ["rack:center"])
    first = grid.frame()
    assert first["image"].startswith("data:image/png;base64,")
    assert first["graph"]["nodes"] == ["Sync", "Flash", "Gate", "rack:center", "shop:17"]
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
    grid.event("down", x, y, button=3)
    assert grid.graph.get_weight("Flash", "rack:center") == 0
    saved = CameraTriggerGrid(["shop:17"]).frame()["graph"]
    assert CameraTriggerGrid(["shop:17"], saved=saved).frame()["graph"] == saved
    try:
        CameraTriggerGrid(["shop:18"], saved=saved)
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("stale material receiver was accepted")
    print("PASS: original N×N grid image, event editing, serialization and receiver identity")


if __name__ == "__main__":
    main()
