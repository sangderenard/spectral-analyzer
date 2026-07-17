from pathlib import Path

import numpy as np

import exposure_render_demo as demo
import thick_lens_focus_lab as tll


ROOT = Path(__file__).resolve().parents[1]


def test_exact_lens_teleport_advances_gpu_and_cpu_path_vertices() -> None:
    shader = (ROOT / "csrc/shaders/ray_material.comp.glsl").read_text(
        encoding="utf-8"
    )
    native = (ROOT / "csrc/kernels/ray_tracer.cpp").read_text(encoding="utf-8")

    assert "src_id, bounce + 1, max(0, bleft - 1), min_amp" in shader
    assert "fill_bdpt_vertex_pdf(g_bdpt_slot, 1.0, 1.0, optical_flags)" in shader
    assert "ri.bounce        += 1;" in native
    assert "ri.bdpt_vertex += 1u;" in native
    assert "pr.flags = st.tris[static_cast<size_t>(hr.hit_tri)].flags" in native


def test_emissive_terminals_are_front_face_only() -> None:
    shader = (ROOT / "csrc/shaders/ray_material.comp.glsl").read_text(
        encoding="utf-8"
    )
    native = (ROOT / "csrc/kernels/ray_tracer.cpp").read_text(encoding="utf-8")

    assert "if (!front_face)" in shader
    assert "eh.is_emissive_hit = emissive_front;" in native


def test_delta_optical_vertices_are_not_arbitrary_t5_endpoints() -> None:
    shader = (ROOT / "csrc/shaders/t5_full_connect.comp.glsl").read_text(
        encoding="utf-8"
    )

    assert (
        "if ((pdf_flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) return false;"
        in shader
    )


def test_scattered_sensor_paths_do_not_use_direct_emitter_splat() -> None:
    material_shader = (ROOT / "csrc/shaders/ray_material.comp.glsl").read_text(
        encoding="utf-8"
    )
    splat_shader = (
        ROOT / "csrc/shaders/sensor_terminal_splat.comp.glsl"
    ).read_text(encoding="utf-8")
    native = (ROOT / "csrc/kernels/ray_tracer.cpp").read_text(encoding="utf-8")

    assert "cflag |= CAMERA_PATH_SCATTERED_BIT;" in material_shader
    assert "color_flag & CAMERA_PATH_SCATTERED_BIT" in splat_shader
    assert "ri.color_flag |= CAMERA_PATH_SCATTERED_BIT;" in native
    assert (
        "hr.ray.color_flag & CAMERA_PATH_SCATTERED_BIT) == 0u"
        in native
    )


def test_dielectric_ior_does_not_make_opaque_subject_transmissive() -> None:
    material_shader = (ROOT / "csrc/shaders/ray_material.comp.glsl").read_text(
        encoding="utf-8"
    )
    native = (ROOT / "csrc/kernels/ray_tracer.cpp").read_text(encoding="utf-8")

    assert "return mat_transmittance(mat, 0) > 1.0e-6;" in material_shader
    assert (
        "return mat_transmittance(st, tri.mat_idx, band) > 1e-6;"
        in native
    )


def test_optical_sidecar_matches_active_tracer_band_order() -> None:
    sidecar = demo._active_frequency_sidecar(tll, demo.DEFAULT_FREQ_HZ)
    expected_nm = demo.C_LIGHT / demo.DEFAULT_FREQ_HZ * 1.0e9

    np.testing.assert_allclose(sidecar.wavelength_nm, expected_nm)
    np.testing.assert_allclose(sidecar.freq_hz, demo.DEFAULT_FREQ_HZ)
    assert np.all(np.diff(sidecar.wavelength_nm) < 0.0)
