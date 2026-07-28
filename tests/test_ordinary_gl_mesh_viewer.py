import numpy as np

from ordinary_gl_mesh_viewer import (
    _heatmap_rgba,
    render_triangle_mesh_image,
    rolling_profile_lines,
    summarize_video_profile,
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


def test_heatmap_texture_conversion_is_rgba_and_monotonic():
    rgba = _heatmap_rgba(np.asarray([[0.0, 0.5, 1.0]]), 0.0, 1.0)
    assert rgba.shape == (1, 3, 4)
    assert rgba.dtype == np.uint8
    assert np.all(rgba[..., 3] == 255)
    assert len({tuple(pixel) for pixel in rgba[0]}) == 3


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


def test_video_summary_keeps_total_and_unforced_gpu_semantics():
    summary = summarize_video_profile(
        {"frame": (0.010, 0.020), "swap": (0.001, 0.003)},
        rendered_frames=2,
        session_elapsed_sec=0.04,
    )
    assert summary["session_elapsed_sec"] == 0.04
    assert summary["stages"]["frame"]["mean_sec"] == 0.015
    assert summary["stages"]["frame"]["max_sec"] == 0.020
    assert summary["gpu_completion_forced"] is False
