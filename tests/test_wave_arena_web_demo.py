"""The browser host must advance and reset the production native field."""
import json
import threading
from urllib.request import Request, urlopen

import numpy as np

from wave_arena_web_demo import ArenaServer, WaveArena


def test_wave_arena_page_and_native_step():
    with ArenaServer(("127.0.0.1", 0)) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def get(path):
            with urlopen(base + path) as response:
                return response.read()

        def post(path, value):
            request = Request(
                base + path,
                json.dumps(value).encode(),
                {"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request) as response:
                return json.loads(response.read())

        try:
            assert b"T4 Wave Arena" in get("/")
            initial = post("/api/wave_arena", {"op": "configure", "size": 64, "relative_phase": 1.2})
            advanced = post("/api/wave_arena", {"op": "step", "count": 2})
            assert advanced["generation"] == 2
            assert advanced["position"] == 2
            assert advanced["panels"]["power"] != initial["panels"]["power"]
            assert advanced["field_power"] > 0
            reset = post("/api/wave_arena", {"op": "configure", "size": 64, "relative_phase": 1.2})
            assert reset["generation"] == 0
            assert reset["panels"]["power"] == initial["panels"]["power"]
        finally:
            server.shutdown()
            worker.join(timeout=5)


def test_larger_arena_uses_native_absorbing_border():
    periodic = WaveArena({"size": 512, "boundary": "periodic", "absorber_cells": 24})
    absorbing = WaveArena({"size": 512, "boundary": "absorbing", "absorber_cells": 24})
    for arena in (periodic, absorbing):
        arena.re.fill(0)
        arena.im.fill(0)
        arena.re[0, 2, 256] = 1.0
    periodic.step()
    absorbing.step()
    periodic_power = float(np.sum(periodic.re**2 + periodic.im**2))
    absorbing_power = float(np.sum(absorbing.re**2 + absorbing.im**2))
    assert periodic_power > 0.99
    assert absorbing_power < periodic_power


def test_finite_aperture_material_changes_field_before_propagation():
    open_field = WaveArena({"size": 128, "aperture_shape": "none"})
    iris = WaveArena({"size": 128, "aperture_shape": "iris", "aperture_opening_fraction": 0.12})
    bore = WaveArena({"size": 128, "aperture_shape": "circular", "aperture_opening_fraction": 0.12})
    powers = [arena.snapshot()["field_power"] for arena in (open_field, iris, bore)]
    assert powers[1] < powers[0]
    assert powers[2] < powers[0]
    assert not np.array_equal(iris.re, bore.re)
    iris.step()
    assert iris.generation == 1
    fine = WaveArena({"size": 512, "pitch_um": 0.2, "aperture_shape": "circular"})
    fine.step()
    assert fine.snapshot()["field_power"] > 0
