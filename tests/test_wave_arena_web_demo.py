"""The browser host must advance and reset the production native field."""
import json
import threading
from urllib.request import Request, urlopen

from wave_arena_web_demo import ArenaServer


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
            initial = post("/api/configure", {"size": 64, "relative_phase": 1.2})
            advanced = post("/api/step", {"count": 2})
            assert advanced["generation"] == 2
            assert advanced["position"] == 2
            assert advanced["panels"]["power"] != initial["panels"]["power"]
            assert advanced["field_power"] > 0
            reset = post("/api/configure", {"size": 64, "relative_phase": 1.2})
            assert reset["generation"] == 0
            assert reset["panels"]["power"] == initial["panels"]["power"]
        finally:
            server.shutdown()
            worker.join(timeout=5)
