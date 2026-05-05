import numpy as np
import pytest


pytest.importorskip("_spectral_kernels")


def test_doc_renderer_c_sibling_order_ignores_submission_order():
    from doc_renderer import DR_NODE_PRIM_RECT, DocRenderer

    rdr = DocRenderer(32, 24)
    # Submit the higher-order node first. It should still paint on top because
    # draw order is explicit hierarchy/sibling order, not call order.
    rdr.submit_raw(
        2,
        (10, 8, 16, 12),
        DR_NODE_PRIM_RECT,
        bg=(0.0, 0.0, 0.0, 0.0),
        accent=(0.0, 0.0, 1.0, 1.0),
        border_px=0,
        sibling_order=0,
    )
    rdr.submit_raw(
        1,
        (4, 4, 16, 12),
        DR_NODE_PRIM_RECT,
        bg=(0.0, 0.0, 0.0, 0.0),
        accent=(1.0, 0.0, 0.0, 1.0),
        border_px=0,
        sibling_order=10,
    )

    rdr._backend.flush()
    out = rdr._backend.composite()

    # Primitive rectangles draw with a small inset. This pixel is in both
    # inner fills, so the higher sibling order should win.
    assert out[11, 13].tolist() == [255, 0, 0, 255]


def test_global_dispatcher_reuses_clean_2d_c_composite():
    from doc_renderer import DR_NODE_PRIM_RECT, DocRenderer
    from globals_renderer import ChannelBackend, GlobalChannelDispatcher

    rdr = DocRenderer(32, 24)
    rdr.submit_raw(
        1,
        (4, 4, 16, 12),
        DR_NODE_PRIM_RECT,
        bg=(0.0, 0.0, 0.0, 0.0),
        accent=(1.0, 0.0, 0.0, 1.0),
        border_px=0,
    )

    dispatcher = GlobalChannelDispatcher(
        width=32,
        height=24,
        gl_doc_renderer=None,
        c_doc_backend=rdr._backend,
        async_c=False,
    )
    leftovers = {"doc.layer/test": {"kind": "test"}}

    first = dispatcher.dispatch(
        leftovers_2d=leftovers,
        leftovers_3d={},
        mode_2d=ChannelBackend.C,
        mode_3d=ChannelBackend.C,
        frame_index=1,
        dt=0.016,
    )
    second = dispatcher.dispatch(
        leftovers_2d=leftovers,
        leftovers_3d={},
        mode_2d=ChannelBackend.C,
        mode_3d=ChannelBackend.C,
        frame_index=2,
        dt=0.016,
    )

    assert first.used_2d == "c"
    assert second.used_2d == "c"
    assert first.out_2d_rgba is not None
    assert second.out_2d_rgba is not None
    np.testing.assert_array_equal(first.out_2d_rgba, second.out_2d_rgba)


def test_doc_renderer_c_hierarchy_paints_child_after_parent():
    from doc_renderer import DR_NODE_PRIM_RECT, DocRenderer

    rdr = DocRenderer(32, 24)

    # Submit the child first.  Hierarchical order should still paint it after
    # the parent because parent_id carries invocation structure.
    rdr.submit_raw(
        2,
        (10, 8, 16, 12),
        DR_NODE_PRIM_RECT,
        bg=(0.0, 0.0, 0.0, 0.0),
        accent=(0.0, 0.0, 1.0, 1.0),
        border_px=0,
        parent_id=1,
        sibling_order=0,
    )
    rdr.submit_raw(
        1,
        (4, 4, 20, 16),
        DR_NODE_PRIM_RECT,
        bg=(0.0, 0.0, 0.0, 0.0),
        accent=(1.0, 0.0, 0.0, 1.0),
        border_px=0,
        sibling_order=0,
    )

    rdr._backend.flush()
    out = rdr._backend.composite()

    assert out[11, 13].tolist() == [0, 0, 255, 255]
