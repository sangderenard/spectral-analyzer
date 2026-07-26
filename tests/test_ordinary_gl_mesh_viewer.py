import numpy as np

from ordinary_gl_mesh_viewer import (
    render_triangle_mesh_image,
    rolling_profile_lines,
    scalar_triangle_bins,
    triangle_mesh_vertex_rows,
)


def test_triangle_mesh_vertex_rows_builds_flat_unit_normals():
    triangle = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    rows = triangle_mesh_vertex_rows(triangle)
    assert rows.shape == (3, 8)
    assert rows.dtype == np.float32
    assert np.allclose(rows[:, 3:6], (0.0, 0.0, 1.0))


def test_scalar_triangle_bins_are_signed_symmetric_and_clipped():
    bins, limit = scalar_triangle_bins(
        np.asarray((-10.0, -2.0, 0.0, 2.0, 10.0)), bin_count=5, limit=2.0
    )
    assert limit == 2.0
    assert np.array_equal(bins, (0, 0, 2, 4, 4))


def test_headless_renderer_writes_png(tmp_path):
    triangle = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    output = render_triangle_mesh_image(
        triangle, tmp_path / "mesh.png", triangle_values=np.asarray((0.5,))
    )
    assert output.is_file()
    assert output.read_bytes().startswith(b"\x89PNG")


def test_rolling_profile_lines_include_all_runs_and_p95():
    history = ({"solve": 1.0}, {"solve": 3.0})
    lines = rolling_profile_lines(history[-1], history, time_value=0.25)
    text = "\n".join(lines)
    assert "simulation t" in text
    assert "2000.0" in text
    assert "runs included             2" in text


def test_headless_renderer_accepts_profile_side_panel(tmp_path):
    triangle = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    output = render_triangle_mesh_image(
        triangle,
        tmp_path / "profiled.png",
        side_panel_lines=("honest profile", "solve 12.3 ms"),
    )
    assert output.is_file()
