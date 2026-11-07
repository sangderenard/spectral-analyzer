"""Saved Studio geometry enters the existing spectral GPU scene and material ABI."""

import numpy as np

from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
from mat_flags import MAT_FLAG_TRANSMISSIVE


def test_saved_mesh_keeps_distinct_optical_and_opaque_materials():
    scene = (
        np.zeros((0, 16), np.float32), np.zeros((0, 16), np.float32),
        np.zeros((0, 12), np.float32), np.zeros((0, 3, 3), np.float32),
        np.zeros((0, 8), np.float32), {},
        (np.array([-.1, -.1, -.1], np.float32),
         np.array([.1, .1, .1], np.float32)),
        np.zeros((0, 12), np.float32),
    )
    vertices = np.array([
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0]],
        [[-1, -1, 1], [1, -1, 1], [0, 1, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # tessellated pole cap
    ], np.float32)
    pbr = np.zeros((2, 16), np.float32)
    pbr[:, 0:3] = .8
    pbr[:, 6] = 1.52
    pbr[0, 3], pbr[0, 5], pbr[0, 7] = .05, 1.0, 0.0
    pbr[1, 3], pbr[1, 5], pbr[1, 7] = .8, 0.0, 1.0

    geom, shade, mat_buf, tris, *_ = append_saved_studio_mesh_to_gpu_scene(
        scene, vertices, np.array([0, 1, 0]), pbr,
        scale_m=.025, center_m=(0, 0, .18), wavelengths_um=[.5875618],
    )

    assert len(tris) == len(geom) == len(shade) == 2
    np.testing.assert_allclose(tris[0, 0], [-.025, -.025, .18])
    np.testing.assert_allclose(shade[:, 3], [1.52, 1.52], rtol=1e-6)
    np.testing.assert_allclose(shade[:, 7], [0.0, 1.0])
    assert geom[0, 15:16].view(np.uint32)[0] & MAT_FLAG_TRANSMISSIVE
    assert not (geom[1, 15:16].view(np.uint32)[0] & MAT_FLAG_TRANSMISSIVE)
    assert shade[0, 14:15].view(np.uint32)[0] != shade[1, 14:15].view(np.uint32)[0]
    assert len(mat_buf) >= 2 * 32  # fixed-stride spectral rows remain present
