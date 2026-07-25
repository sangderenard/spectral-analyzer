import numpy as np

from ordinary_gl_mesh_viewer import triangle_mesh_vertex_rows


def test_triangle_mesh_vertex_rows_builds_flat_unit_normals():
    triangle = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    rows = triangle_mesh_vertex_rows(triangle)
    assert rows.shape == (3, 8)
    assert rows.dtype == np.float32
    assert np.allclose(rows[:, 3:6], (0.0, 0.0, 1.0))
