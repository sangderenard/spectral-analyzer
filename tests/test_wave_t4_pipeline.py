import time

import numpy as np
import pytest


kernels = pytest.importorskip("_spectral_kernels")


def _tracer(wavelengths_m: np.ndarray):
    from ray_tracer_bridge import per_tri_spectral_to_mat_buf

    frequencies = 299_792_458.0 / wavelengths_m
    bands = len(wavelengths_m)
    reflectance = np.zeros((1, bands), np.float64)
    mat_idx, mat_buf, mat_count = per_tri_spectral_to_mat_buf(
        reflectance, reflectance, reflectance, frequencies
    )
    triangle = np.asarray(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]], np.float64
    )
    normal = np.asarray([[0.0, 0.0, 1.0]], np.float64)
    return kernels.RayTracer(
        1,
        triangle,
        normal,
        mat_idx,
        mat_buf,
        int(mat_count),
        frequencies.astype(np.float64),
        299_792_458.0,
        np.zeros(bands, np.float64),
    )


def _wait(tracer, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while tracer.in_flight_count() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert tracer.in_flight_count() == 0


def _add_arena(tracer):
    tracer.add_scale_context(
        np.array([0.0, 0.0, 0.0]),
        0.02,
        1,
        0.005,
        3,
        1.0,
        0.0,
        0,
        None,
    )


def test_empty_space_crossing_routes_through_t4():
    tracer = _tracer(np.array([550e-9]))
    _add_arena(tracer)
    tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12)
    tracer.submit_rays(
        np.array([[0.0, 0.0, -0.05]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[1.0 + 0.0j]]),
        max_bounces=1,
        min_amplitude=1e-12,
    )
    _wait(tracer)
    records = tracer.drain_records(16)
    assert np.asarray(records["kind"]).tolist() == [3]
    assert tracer.wave_arena_stats()[0]["generation"] == 1


def test_compiled_optical_graph_installs_and_drives_native_t4():
    from camera_software.optical_transport_graph import (
        OpticalTransportGraphSpec,
        WavePropagationStyle,
        compile_optical_graph,
        install_optical_graph,
        wave_context_nodes,
    )

    tracer = _tracer(np.array([550e-9]))
    nodes, links = wave_context_nodes(
        "bench.wave",
        1,
        propagation=WavePropagationStyle.ADI_REFERENCE,
        center_m=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        radius_m=0.02,
        longitudinal_step_m=0.005,
        longitudinal_steps=3,
    )
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        nodes=nodes,
        links=links,
        entry_keys=(nodes[0].key,),
        product_keys=(nodes[-1].key,),
    ))
    receipt = install_optical_graph(
        tracer,
        compiled,
        exact_t2_registered=False,
    )
    tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12)
    tracer.submit_rays(
        np.array([[0.0, 0.0, -0.05]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[1.0 + 0.0j]]),
        max_bounces=1,
        min_amplitude=1e-12,
    )
    _wait(tracer)

    arena = tracer.wave_arena_stats()[0]
    assert receipt.wave_arenas[0].node_key == "bench.wave.arena"
    assert arena["generation"] == 1
    assert arena["bands"] == 1
    assert arena["backend"] == 0  # LegacyAdiCalibration


def test_continuous_paths_share_one_exact_width_complex_dispatch():
    tracer = _tracer(np.array([450e-9, 500e-9, 600e-9, 700e-9]))
    tracer.configure_spectral_luts(
        np.zeros(4, np.int32),
        np.array([0, 2], np.int32),
        np.array([4.0e14, 7.5e14]),
        np.ones(2),
    )
    _add_arena(tracer)
    tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12)
    origins = np.array([[x, 0.0, -0.05] for x in (-0.003, -0.001, 0.001, 0.003)])
    tracer.submit_rays(
        origins,
        np.tile([0.0, 0.0, 1.0], (4, 1)),
        np.ones((4, 4), np.complex128),
        tags=np.array([11, 22, 33, 44], np.uint64),
        max_bounces=1,
        min_amplitude=1e-12,
    )
    _wait(tracer)
    arena = tracer.wave_arena_stats()[0]
    assert arena["spectral_mode"] == 1
    assert arena["generation"] == 1
    assert sum(lane["active"] for lane in arena["lanes"]) == 4
    assert len({lane["frequency_hz"] for lane in arena["lanes"] if lane["active"]}) > 1
    assert np.asarray(tracer.drain_records(16)["kind"]).tolist() == [3, 3, 3, 3]


def test_gpu_t1_routes_empty_space_continuous_cohort_to_t4():
    """Exercise GPU T1 itself: gpu_all_stages leaves no CPU T1 worker alive."""
    tracer = _tracer(np.array([450e-9, 500e-9, 600e-9, 700e-9]))
    tracer.configure_spectral_luts(
        np.zeros(4, np.int32),
        np.array([0, 2], np.int32),
        np.array([4.0e14, 7.5e14]),
        np.ones(2),
    )
    _add_arena(tracer)
    tracer.ensure_pipeline(
        max_children=1,
        min_amplitude=1e-12,
        use_gpu_compute=True,
        gpu_all_stages=True,
        shader_dir="csrc/shaders",
    )
    origins = np.array([[x, 0.0, -0.05] for x in (-0.003, -0.001, 0.001, 0.003)])
    tags = np.array([11, 22, 33, 44], np.uint64)
    tracer.submit_rays(
        origins,
        np.tile([0.0, 0.0, 1.0], (4, 1)),
        np.ones((4, 4), np.complex128),
        tags=tags,
        max_bounces=1,
        min_amplitude=1e-12,
    )
    _wait(tracer)

    arena = tracer.wave_arena_stats()[0]
    active = [lane for lane in arena["lanes"] if lane["active"]]
    assert arena["generation"] == 1
    assert arena["spectral_mode"] == 1
    assert len(active) == 4
    assert [lane["coherence_id"] for lane in active] == tags.tolist()
    assert np.asarray(tracer.drain_records(16)["kind"]).tolist() == [3, 3, 3, 3]
