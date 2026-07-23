from __future__ import annotations

from camera_software.gpu_preview import PreviewProductRegistry
from pluck_render_graph import (
    EngineRequest,
    EngineState,
    MultiEngineRenderGraph,
    OpticalEnginePillar,
)


def _request(revision: int) -> EngineRequest:
    return EngineRequest.create(
        "optical",
        {
            "bench_id": "camera-bench",
            "revision": revision,
            "transport_mode": "mixed",
        },
        scene_revision=f"scene-{revision}",
        requested_products=("surface_scan", "sensor_accumulation"),
    )


def test_optical_pillar_retains_latest_bench_request_until_backend_attaches():
    registry = PreviewProductRegistry()
    graph = MultiEngineRenderGraph(registry)
    optical = OpticalEnginePillar(registry)
    graph.register(optical)

    assert graph.submit(_request(1))
    assert graph.submit(_request(2))
    before = optical.describe()
    assert before["state"] == EngineState.UNAVAILABLE.value
    assert before["queued"] == 1

    submitted = []
    polled = []

    def submit(payload):
        submitted.append(dict(payload))
        return "handle"

    def poll(handle):
        polled.append(handle)
        return {"complete": True}

    optical.attach_backend(submit, poll)
    graph.tick(1, 1.0 / 60.0)

    assert submitted == [{
        "bench_id": "camera-bench",
        "revision": 2,
        "transport_mode": "mixed",
    }]
    assert polled == ["handle"]
    after = optical.describe()
    assert after["state"] == EngineState.READY.value
    assert after["completed"] == 1
    assert graph.products is registry


def test_engine_request_identity_is_stable_for_equal_work():
    assert _request(4).request_id == _request(4).request_id
    assert _request(4).request_id != _request(5).request_id

