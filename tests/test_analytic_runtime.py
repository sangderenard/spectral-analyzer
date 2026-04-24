import numpy as np
import torch

from analytic_model import AnalyticPatch, AnalyticVoice
from analytic_runtime import PatchRenderResult, RenderProduct, render_patch_graph
from routing_engine import RoutingEdge


def test_render_result_treats_stereo_as_compat_view_not_contract() -> None:
    products = {
        "node:a": RenderProduct(
            key="node:a",
            tensor=torch.tensor([1.0 + 2.0j, 3.0 + 4.0j], dtype=torch.complex128),
            semantic="arbitrary_complex_product",
        )
    }
    result = PatchRenderResult(
        engine="test",
        sample_rate=2,
        n_samples=2,
        products=products,
        output_product_keys=["node:a"],
    )

    bus = result.output_bus()
    left, right = result.stereo_pair()

    assert bus.shape == (2, 1)
    assert np.allclose(bus[:, 0], [1.0, 3.0])
    assert np.allclose(left, right)


def test_graph_render_returns_named_complex128_products() -> None:
    patch = AnalyticPatch()
    patch.preview_sr = 8
    patch.duration = 0.25
    patch.system_audio.output_channels = 1
    patch.voices = [AnalyticVoice(key="v1", freq_hz=4.0)]
    patch.routing.edges = [RoutingEdge(src_key="v1", dst_key="__sys_out_1__", weight=1.0)]

    result = render_patch_graph(
        patch,
        sample_rate=8,
        n_samples=2,
        product_keys=["v1", "__sys_out_1__"],
        use_cache=False,
    )

    assert result.engine == "graph"
    assert set(result.products) == {"v1", "__sys_out_1__"}
    assert result.products["v1"].tensor.dtype == torch.complex128
    assert result.products["__sys_out_1__"].tensor.shape == (2,)
