"""Archived direct camera draw fallback from demo_pluck_gl.py.

This path bypassed the scene.geometry owner-buffer flow and drew camera
objects directly after the global dispatcher. Cameras already publish their
geometry through the owner flip buffers, so the live frame should use the
same leftover/global renderer path as other scene geometry.
"""


def legacy_direct_camera_draw_block() -> None:
    """Reference copy only; do not call from the live frame loop."""
    # _rs_lv = np.array([0.5, 1.0, 0.6], np.float32)
    # _rs_lv /= np.linalg.norm(_rs_lv)
    # if cameras and _cam_pure_matrices is not None:
    #     _P, _V = _cam_pure_matrices(R.cam)
    #     _MVP = (_P @ _V).astype(np.float32)
    #     _MV  = _V.astype(np.float32)
    #     _lv  = _rs_lv
    #     for _ci in cameras:
    #         _ci.draw(_MVP, _MV, _lv)
    return None
