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
    field = tracer.wave_arena_field_snapshot(0, 0, 0, 0)
    assert field["generation"] == 1
    assert np.asarray(field["re"]).shape == (
        tracer.wave_arena_stats()[0]["ny"],
        tracer.wave_arena_stats()[0]["nx"],
    )
    assert np.max(np.abs(
        np.asarray(field["re"]) + 1j * np.asarray(field["im"])
    )) > 0.0
    p_field = tracer.wave_arena_field_snapshot(0, 0, 1, 0)
    assert np.count_nonzero(p_field["re"]) == 0
    assert np.count_nonzero(p_field["im"]) == 0


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
        propagation=WavePropagationStyle.ANGULAR_SPECTRUM_FFT,
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
    assert arena["backend"] == 0  # AngularSpectrum
    assert arena["field_count"] == 4
    assert arena["state_float_count"] == (
        arena["bands"] * arena["fft_nx"] * arena["fft_ny"] * 8
    )
    assert sum(arena["field_active"]) == 1


def test_wave_boundary_preserves_power_tilt_and_authored_axial_extent():
    tracer = _tracer(np.array([550e-9]))
    radius = 64.0e-6
    dz = 2.0e-6
    steps = 8
    tracer.add_scale_context(
        np.array([0.0, 0.0, 0.0]),
        radius,
        1,
        dz,
        steps,
        1.0,
        0.0,
        1,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12)
    tilt = 0.01
    direction = np.array([tilt, 0.0, np.sqrt(1.0 - tilt * tilt)])
    tracer.submit_rays(
        np.array([[0.0, 0.0, -0.001]]),
        direction[None],
        np.array([[1.0 + 0.0j]]),
        max_bounces=2,
        min_amplitude=1e-12,
    )
    _wait(tracer)

    arena = tracer.wave_arena_stats()[0]
    boundary = arena["boundary"]
    assert arena["longitudinal_extent_m"] == pytest.approx(dz * steps)
    assert boundary["entry_local"][2] == pytest.approx(
        -0.5 * dz * steps, abs=2.0e-8
    )
    assert boundary["exit_local"][2] == pytest.approx(
        0.5 * dz * steps, abs=2.0e-8
    )
    assert boundary["seeded_field_power"] == pytest.approx(
        boundary["input_ray_power"], rel=2.0e-5
    )
    assert boundary["output_ray_power"] == pytest.approx(
        boundary["propagated_field_power"], rel=2.0e-5
    )
    assert boundary["exit_direction"][0] == pytest.approx(tilt, abs=2.5e-3)

    records = tracer.drain_records(16)
    field_index = int(np.flatnonzero(np.asarray(records["kind"]) == 3)[0])
    assert np.asarray(records["seg_start"])[field_index, 2] < 0.0
    assert np.asarray(records["pos"])[field_index, 2] > 0.0


def test_compatible_wave_contexts_chain_without_intermediate_ray_collapse():
    tracer = _tracer(np.array([550e-9]))
    radius = 64.0e-6
    dz = 2.0e-6
    steps = 8
    axis_payloads = []
    context_ids = []
    for center_z in (-8.0e-6, 8.0e-6):
        payload = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float64
        )
        axis_payloads.append(payload)
        context_ids.append(tracer.add_scale_context(
            np.array([0.0, 0.0, center_z]),
            radius, 1, dz, steps, 1.0, 0.0, 1, payload,
        ))
    tracer.add_wave_context_link(context_ids[0], context_ids[1])
    tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12)
    tracer.submit_rays(
        np.array([[12.0e-6, -7.0e-6, -0.001]]),
        np.array([[0.006, -0.004, np.sqrt(1.0 - 0.006**2 - 0.004**2)]]),
        np.array([[1.0 + 0.0j]]),
        max_bounces=2,
        min_amplitude=1e-12,
    )
    _wait(tracer)

    arenas = tracer.wave_arena_stats()
    by_context = {arena["context_id"]: arena for arena in arenas}
    first = by_context[context_ids[0]]
    second = by_context[context_ids[1]]
    assert first["generation"] == 1
    assert second["generation"] == 1
    assert first["linked_transfers"] == 1
    assert first["next_forward"] == second["arena_id"]
    assert second["next_backward"] == first["arena_id"]
    assert first["boundary"]["linked_to_arena"] == second["arena_id"]
    assert second["boundary"]["linked_from_arena"] == first["arena_id"]
    assert second["boundary"]["seeded_field_power"] == pytest.approx(
        first["field_power"], rel=2.0e-6
    )

    records = tracer.drain_records(16)
    field_records = np.flatnonzero(np.asarray(records["kind"]) == 3)
    assert len(field_records) == 1
    assert int(np.asarray(records["arena_id"])[field_records[0]]) == (
        second["arena_id"]
    )


def test_production_angular_spectrum_plane_wave_phase_and_reverse():
    wavelength = 550.0e-9
    tracer = _tracer(np.array([wavelength]))
    width = height = 8
    dz = 1.25e-6
    re = np.ones((1, height, width), np.float32)
    im = np.zeros_like(re)
    original = re.astype(np.complex64) + 1j * im.astype(np.complex64)

    tracer.t4_angular_spectrum_step(
        1, width, height, 1.0e-6, dz, 1,
        np.array([wavelength], np.float64), re, im,
    )
    expected = np.exp(1j * (2.0 * np.pi / wavelength) * dz)
    propagated = re.astype(np.complex64) + 1j * im.astype(np.complex64)
    np.testing.assert_allclose(
        propagated, original * expected, rtol=2.0e-5, atol=2.0e-5
    )

    tracer.t4_angular_spectrum_step(
        1, width, height, 1.0e-6, dz, -1,
        np.array([wavelength], np.float64), re, im,
    )
    recovered = re.astype(np.complex64) + 1j * im.astype(np.complex64)
    np.testing.assert_allclose(
        recovered, original, rtol=3.0e-5, atol=3.0e-5
    )


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
    assert arena["boundary"]["seeded_field_power"] == pytest.approx(
        arena["boundary"]["input_ray_power"], rel=2.0e-5
    )
    assert arena["boundary"]["output_ray_power"] == pytest.approx(
        arena["boundary"]["propagated_field_power"], rel=2.0e-5
    )
    assert np.asarray(tracer.drain_records(16)["kind"]).tolist() == [3, 3, 3, 3]


def test_continuous_cohort_marches_linked_field_chain_only_once():
    tracer = _tracer(np.array([450e-9, 500e-9, 600e-9, 700e-9]))
    tracer.configure_spectral_luts(
        np.zeros(4, np.int32),
        np.array([0, 2], np.int32),
        np.array([4.0e14, 7.5e14]),
        np.ones(2),
    )
    context_ids = []
    payloads = []
    for center_z in (-8.0e-6, 8.0e-6):
        payload = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float64
        )
        payloads.append(payload)
        context_ids.append(tracer.add_scale_context(
            np.array([0.0, 0.0, center_z]),
            64.0e-6, 1, 2.0e-6, 8, 1.0, 0.0, 1, payload,
        ))
    tracer.add_wave_context_link(context_ids[0], context_ids[1])
    tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12)
    origins = np.array([
        [x, 0.0, -0.001] for x in (-12e-6, -4e-6, 4e-6, 12e-6)
    ])
    tracer.submit_rays(
        origins,
        np.tile([0.0, 0.0, 1.0], (4, 1)),
        np.ones((4, 4), np.complex128),
        tags=np.array([11, 22, 33, 44], np.uint64),
        max_bounces=2,
        min_amplitude=1e-12,
    )
    _wait(tracer)

    arenas = tracer.wave_arena_stats()
    by_context = {arena["context_id"]: arena for arena in arenas}
    first = by_context[context_ids[0]]
    second = by_context[context_ids[1]]
    assert first["spectral_mode"] == 1
    assert second["spectral_mode"] == 1
    assert first["generation"] == 1
    assert second["generation"] == 1
    assert first["linked_transfers"] == 1
    records = tracer.drain_records(16)
    field_mask = np.asarray(records["kind"]) == 3
    assert int(np.count_nonzero(field_mask)) == 4
    assert set(np.asarray(records["arena_id"])[field_mask].tolist()) == {
        second["arena_id"]
    }


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
    assert arena["boundary"]["output_ray_power"] == pytest.approx(
        arena["boundary"]["propagated_field_power"], rel=3.0e-5
    )
    assert np.asarray(tracer.drain_records(16)["kind"]).tolist() == [3, 3, 3, 3]
