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
    assert tracer.drain_wave_exit_states(16).shape == (0, 160)


def test_wave_transaction_checkpoint_restores_and_replays_native_field_state():
    tracer = _tracer(np.array([550e-9]))
    _add_arena(tracer)
    tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12)
    wave_checkpoint = tracer.copy_wave_transaction_state()
    queue_checkpoint = tracer.copy_queue_transaction_state()

    def run_once():
        tracer.submit_rays(
            np.array([[0.0, 0.0, -0.05]]),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[1.0 + 0.0j]]),
            max_bounces=1,
            min_amplitude=1e-12,
        )
        _wait(tracer)
        field = tracer.wave_arena_field_snapshot(0, 0, 0, 0)
        return (
            tracer.wave_arena_stats()[0],
            np.asarray(field["re"]).copy(),
            np.asarray(field["im"]).copy(),
        )

    first_stats, first_re, first_im = run_once()
    assert first_stats["generation"] == 1
    assert tracer.pipeline_stats()["output_queue_depth"] == 1
    tracer.restore_queue_transaction_state(queue_checkpoint)
    tracer.restore_wave_transaction_state(wave_checkpoint)
    restored = tracer.wave_arena_stats()[0]
    assert restored["generation"] == 0
    assert restored["completed_steps"] == 0
    assert tracer.pipeline_stats()["output_queue_depth"] == 0

    second_stats, second_re, second_im = run_once()
    assert second_stats["generation"] == 1
    assert second_stats["completed_steps"] == first_stats["completed_steps"]
    np.testing.assert_array_equal(second_re, first_re)
    np.testing.assert_array_equal(second_im, first_im)
    assert np.asarray(tracer.drain_records(16)["kind"]).tolist() == [3]


def test_fixed_band_source_mode_seeds_both_components_and_full_coherence():
    from camera_software.complex_optical_operators import (
        ComplexOpticalOperator,
        ComplexOperatorStateBlock,
        ComplexSourceMode,
        JonesOperator,
        TransverseBasis,
        install_source_mode_block,
        parse_wave_exit_states,
    )

    tracer = _tracer(np.array([550e-9]))
    _add_arena(tracer)
    tag = 0x12345678ABCDEF01
    coherence = 0xFFEEDDCC87654321
    block = ComplexOperatorStateBlock()
    block.add_basis(TransverseBasis.from_direction((0.0, 0.0, 1.0)))
    block.add_operator(ComplexOpticalOperator(jones=JonesOperator(
        np.asarray([[1.0, 0.0], [0.0, 1.0j]], np.complex128)
    )))
    block.add_source_mode(ComplexSourceMode(
        jones=np.asarray([1.0, 1.0]),
        power_weight=1.0,
        coherence_id=coherence,
        basis_id=0,
    ))
    install_source_mode_block(
        tracer,
        np.asarray([tag, tag + 1], np.uint64),
        block,
        mode_indices=np.zeros(2, np.uint32),
    )
    tracer.ensure_pipeline(
        max_children=1,
        min_amplitude=1e-12,
        capture_wave_exit_states=True,
    )
    tracer.submit_rays(
        np.array([[0.0, 0.0, -0.05]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[1.0 + 0.0j]]),
        tags=np.asarray([tag], np.uint64),
        max_bounces=1,
        min_amplitude=1e-12,
    )
    _wait(tracer)

    s_field = tracer.wave_arena_field_snapshot(0, 0, 0, 0)
    p_field = tracer.wave_arena_field_snapshot(0, 0, 1, 0)
    s = np.asarray(s_field["re"]) + 1j*np.asarray(s_field["im"])
    p = np.asarray(p_field["re"]) + 1j*np.asarray(p_field["im"])
    illuminated = np.abs(s) > float(np.max(np.abs(s))) * 1.0e-6
    np.testing.assert_allclose(p[illuminated], 1j*s[illuminated], rtol=3e-5)
    arena = tracer.wave_arena_stats()[0]
    assert arena["lanes"][0]["coherence_id"] == coherence
    assert arena["field_active"][:2] == [1, 1]

    exits = parse_wave_exit_states(tracer.drain_wave_exit_states(16))
    assert exits.shape == (1,)
    exit_state = exits[0]
    assert int(exit_state["ray_tag"]) == tag
    assert int(exit_state["arena_id"]) == 0
    assert int(exit_state["state_lane"]) == -1
    assert int(exit_state["band_id"]) == 0
    assert int(exit_state["record_flags"]) & 0x3 == 0x3
    lane_flags = int(exit_state["lane_flags"]) & 0xFFFF
    assert lane_flags & (1 << 0)  # active
    assert lane_flags & (1 << 3)  # Jones valid
    assert lane_flags & (1 << 4)  # optical path valid (lineage path_len accumulator)
    assert lane_flags & (1 << 6)  # field reduction
    assert lane_flags & (1 << 7)  # wave exit
    assert int(exit_state["coherence_lo"]) | (
        int(exit_state["coherence_hi"]) << 32
    ) == coherence
    frequency_hz = float(exit_state["frequency_hi"]) + float(
        exit_state["frequency_lo"]
    )
    assert frequency_hz == pytest.approx(299_792_458.0 / 550e-9, rel=2e-7)
    optical_path_m = float(exit_state["optical_path_hi"]) + float(
        exit_state["optical_path_lo"]
    )
    # Real lineage OPL now (OpticalPathValid is set): ray launched from
    # z=-0.05 into the arena, so this should be a small positive distance,
    # not the old fabrication-avoidance stub of exactly 0.
    assert 0.0 < optical_path_m < 1.0

    direction = np.asarray(exit_state["ray_direction"], np.float64)
    basis_s = np.asarray(exit_state["basis_s"][:3], np.float64)
    basis_p = np.asarray(exit_state["basis_p"][:3], np.float64)
    np.testing.assert_allclose(np.cross(basis_s, basis_p), direction, atol=2e-6)
    assert np.dot(basis_s, direction) == pytest.approx(0.0, abs=2e-6)
    assert np.dot(basis_p, direction) == pytest.approx(0.0, abs=2e-6)

    amplitude_s = complex(
        float(exit_state["amplitude_s_re"]),
        float(exit_state["amplitude_s_im"]),
    )
    amplitude_p = complex(
        float(exit_state["amplitude_p_re"]),
        float(exit_state["amplitude_p_im"]),
    )
    assert amplitude_p / amplitude_s == pytest.approx(1j, rel=4e-5, abs=4e-5)
    retained_power = abs(amplitude_s) ** 2 + abs(amplitude_p) ** 2
    assert retained_power == pytest.approx(
        arena["boundary"]["propagated_field_power"], rel=3e-5
    )

    field_records = tracer.drain_records(16)
    assert np.asarray(field_records["kind"]).tolist() == [3]
    assert int(np.asarray(field_records["tag"])[0]) == tag


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
    # The state block is cohort-sized: a whole number of n_bands-wide ray
    # slices share one batched march, so lane capacity, not bands alone,
    # sets the allocation.
    assert arena["cohort_lanes"] % arena["bands"] == 0
    assert arena["cohort_lanes"] >= arena["bands"]
    assert arena["state_float_count"] == (
        arena["cohort_lanes"] * arena["fft_nx"] * arena["fft_ny"] * 8
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


def test_native_wave_fanout_conserves_power_and_is_link_order_invariant():
    def run(link_order):
        tracer = _tracer(np.array([550e-9]))
        radius = 64.0e-6
        dz = 2.0e-6
        steps = 8
        payload = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float64
        )
        context_ids = [
            tracer.add_scale_context(
                np.asarray([0.0, 0.0, center_z]),
                radius, 1, dz, steps, 1.0, 0.0, 1, payload.copy(),
            )
            for center_z in (-8.0e-6, 8.0e-6, 8.0e-6)
        ]
        amplitudes = (0.6, 0.8)
        for destination_index in link_order:
            amplitude = amplitudes[destination_index]
            jones = np.asarray(
                [[[amplitude, 0.0], [0.0, amplitude]]], np.complex64
            )
            tracer.add_wave_context_interface_link(
                context_ids[0],
                context_ids[destination_index + 1],
                0,
                jones.real,
                jones.imag,
            )
        tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12)
        tracer.submit_rays(
            np.asarray([[7.0e-6, -5.0e-6, -0.001]]),
            np.asarray([[0.0, 0.0, 1.0]]),
            np.asarray([[1.0 + 0.0j]]),
            max_bounces=1,
            min_amplitude=1e-12,
        )
        _wait(tracer)

        by_context = {
            arena["context_id"]: arena for arena in tracer.wave_arena_stats()
        }
        source = by_context[context_ids[0]]
        destinations = [
            by_context[context_ids[index]] for index in (1, 2)
        ]
        records = tracer.drain_records(16)
        field_mask = np.asarray(records["kind"]) == 3
        return source, destinations, records, field_mask

    forward = run((0, 1))
    reverse = run((1, 0))
    for source, destinations, records, field_mask in (forward, reverse):
        assert source["forward_link_count"] == 2
        assert source["linked_transfers"] == 2
        assert [item["backward_link_count"] for item in destinations] == [1, 1]
        assert [item["generation"] for item in destinations] == [1, 1]
        branch_powers = [
            item["boundary"]["seeded_field_power"] for item in destinations
        ]
        assert branch_powers[0] == pytest.approx(
            0.6**2 * source["field_power"], rel=4.0e-6
        )
        assert branch_powers[1] == pytest.approx(
            0.8**2 * source["field_power"], rel=4.0e-6
        )
        assert sum(branch_powers) == pytest.approx(
            source["field_power"], rel=4.0e-6
        )
        assert int(np.count_nonzero(field_mask)) == 2
        assert set(np.asarray(records["arena_id"])[field_mask].tolist()) == {
            item["arena_id"] for item in destinations
        }

    np.testing.assert_allclose(
        [
            item["boundary"]["seeded_field_power"]
            for item in forward[1]
        ],
        [
            item["boundary"]["seeded_field_power"]
            for item in reverse[1]
        ],
        rtol=4.0e-6,
    )


def test_native_wave_link_api_rejects_fanin_and_cycles_before_pipeline_build():
    tracer = _tracer(np.array([550e-9]))
    payload = np.asarray(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float64
    )
    contexts = [
        tracer.add_scale_context(
            np.asarray([0.0, 0.0, center_z]),
            64.0e-6, 1, 2.0e-6, 8, 1.0, 0.0, 1, payload.copy(),
        )
        for center_z in (-16.0e-6, 0.0, 16.0e-6)
    ]

    tracer.add_wave_context_link(contexts[0], contexts[2])
    with pytest.raises(ValueError, match="single-producer"):
        tracer.add_wave_context_link(contexts[1], contexts[2])
    with pytest.raises(ValueError, match="acyclic"):
        tracer.add_wave_context_link(contexts[2], contexts[0])


def test_rigid_interface_link_turns_and_attenuates_persistent_field():
    tracer = _tracer(np.array([550e-9]))
    radius = 64.0e-6
    dz = 2.0e-6
    steps = 8
    half = 0.5*dz*steps
    incoming = np.asarray((1.0, 0.0, 0.0))
    outgoing = np.asarray((1.0, 1.0, 0.0))/np.sqrt(2.0)
    payloads = [
        np.asarray((0.0, 0.0, 0.0, *incoming), np.float64),
        np.asarray((0.0, 0.0, 0.0, *outgoing), np.float64),
    ]
    context_ids = [
        tracer.add_scale_context(
            -incoming*half, radius, 1, dz, steps, 1.0, 0.0, 1,
            payloads[0],
        ),
        tracer.add_scale_context(
            outgoing*half, radius, 1, dz, steps, 1.0, 0.0, 1,
            payloads[1],
        ),
    ]
    jones = np.asarray([[[0.5, 0.0], [0.0, 0.5]]], np.complex64)
    tracer.add_wave_context_interface_link(
        context_ids[0], context_ids[1], 1, jones.real, jones.imag
    )
    tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12)
    tracer.submit_rays(
        np.asarray([[-0.001, 7.0e-6, 0.0]]),
        incoming[None],
        np.asarray([[1.0+0.0j]]),
        max_bounces=2,
        min_amplitude=1e-12,
    )
    _wait(tracer)

    arenas = tracer.wave_arena_stats()
    first, second = arenas
    assert first["next_forward"] == second["arena_id"]
    assert second["next_backward"] == first["arena_id"]
    assert second["boundary"]["seeded_field_power"] == pytest.approx(
        0.25*first["field_power"], rel=3.0e-6
    )
    records = tracer.drain_records(16)
    field_index = int(np.flatnonzero(np.asarray(records["kind"]) == 3)[0])
    np.testing.assert_allclose(
        np.asarray(records["dir"])[field_index], outgoing, atol=2.0e-3
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


@pytest.mark.parametrize("bands", [1, 3, 4, 8, 16, 32])
def test_production_absorbing_border_is_exact_lane_and_keeps_interior(bands):
    tracer = _tracer(np.linspace(450.0e-9, 650.0e-9, bands))
    re = np.ones((bands, 16, 16), np.float32)
    im = np.zeros_like(re)
    result = tracer.t4_apply_absorbing_border(
        bands, 16, 16, 4, 8.0, 1.0, re, im,
    )

    assert result["absorbed_power"] > 0.0
    assert result["border_power"] < float(bands * 16 * 16)
    assert result["field_power"] == pytest.approx(
        float(np.sum(re*re + im*im)), rel=1.0e-6
    )
    assert re[0, 8, 8] == pytest.approx(1.0)
    assert re[0, 0, 0] < 0.001


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


def test_continuous_cohort_uses_same_jones_source_block():
    from camera_software.complex_optical_operators import (
        ComplexOpticalOperator,
        ComplexOperatorStateBlock,
        ComplexSourceMode,
        TransverseBasis,
        install_source_mode_block,
        parse_wave_exit_states,
    )

    tracer = _tracer(np.array([450e-9, 500e-9, 600e-9, 700e-9]))
    tracer.configure_spectral_luts(
        np.zeros(4, np.int32),
        np.array([0, 2], np.int32),
        np.array([4.0e14, 7.5e14]),
        np.ones(2),
    )
    _add_arena(tracer)
    tags = np.asarray([
        0x100000001, 0x200000002, 0x300000003, 0x400000004,
    ], np.uint64)
    coherences = [
        0xA000000010, 0xB000000020, 0xC000000030, 0xD000000040,
    ]
    block = ComplexOperatorStateBlock()
    block.add_basis(TransverseBasis.from_direction((0.0, 0.0, 1.0)))
    block.add_operator(ComplexOpticalOperator())
    for coherence in coherences:
        block.add_source_mode(ComplexSourceMode(
            jones=np.asarray([1.0, 1.0]),
            power_weight=1.0,
            coherence_id=coherence,
            basis_id=0,
        ))
    install_source_mode_block(tracer, tags, block)
    tracer.ensure_pipeline(
        max_children=1,
        min_amplitude=1e-12,
        capture_wave_exit_states=True,
    )
    tracer.submit_rays(
        np.array([
            [x, 0.0, -0.05] for x in (-0.003, -0.001, 0.001, 0.003)
        ]),
        np.tile([0.0, 0.0, 1.0], (4, 1)),
        np.ones((4, 4), np.complex128),
        tags=tags,
        max_bounces=1,
        min_amplitude=1e-12,
    )
    _wait(tracer)

    arena = tracer.wave_arena_stats()[0]
    assert arena["spectral_mode"] == 1
    active = [lane for lane in arena["lanes"] if lane["active"]]
    assert [lane["coherence_id"] for lane in active] == coherences
    for lane in range(4):
        s_field = tracer.wave_arena_field_snapshot(0, 0, 0, lane)
        p_field = tracer.wave_arena_field_snapshot(0, 0, 1, lane)
        s = np.asarray(s_field["re"]) + 1j*np.asarray(s_field["im"])
        p = np.asarray(p_field["re"]) + 1j*np.asarray(p_field["im"])
        np.testing.assert_allclose(p, s, rtol=4e-5, atol=2e-7)

    exits = parse_wave_exit_states(tracer.drain_wave_exit_states(16))
    assert exits.shape == (4,)
    by_tag = {int(row["ray_tag"]): row for row in exits}
    assert set(by_tag) == set(map(int, tags))
    exit_frequencies = []
    for tag, coherence in zip(tags, coherences):
        row = by_tag[int(tag)]
        assert int(row["record_flags"]) & (1 << 3)
        lane_flags = int(row["lane_flags"]) & 0xFFFF
        assert lane_flags & (1 << 1)  # continuous sample
        assert lane_flags & (1 << 3)  # Jones valid
        assert lane_flags & (1 << 7)  # wave exit
        recovered_coherence = int(row["coherence_lo"]) | (
            int(row["coherence_hi"]) << 32
        )
        assert recovered_coherence == coherence
        exit_frequencies.append(
            float(row["frequency_hi"]) + float(row["frequency_lo"])
        )
    assert len(set(exit_frequencies)) == 4


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
    from camera_software.complex_optical_operators import (
        ComplexOpticalOperator,
        ComplexOperatorStateBlock,
        ComplexSourceMode,
        TransverseBasis,
        install_source_mode_block,
    )

    tracer = _tracer(np.array([450e-9, 500e-9, 600e-9, 700e-9]))
    tracer.configure_spectral_luts(
        np.zeros(4, np.int32),
        np.array([0, 2], np.int32),
        np.array([4.0e14, 7.5e14]),
        np.ones(2),
    )
    _add_arena(tracer)
    tags = np.array([11, 22, 33, 44], np.uint64)
    coherences = [
        0x1000000000B, 0x20000000016, 0x30000000021, 0x4000000002C,
    ]
    block = ComplexOperatorStateBlock()
    block.add_basis(TransverseBasis.from_direction((0.0, 0.0, 1.0)))
    block.add_operator(ComplexOpticalOperator())
    for coherence in coherences:
        block.add_source_mode(ComplexSourceMode(
            jones=np.asarray([1.0, 1.0j]),
            power_weight=1.0,
            coherence_id=coherence,
            basis_id=0,
            operator_id=0,
        ))
    install_source_mode_block(tracer, tags, block)
    tracer.ensure_pipeline(
        max_children=1,
        min_amplitude=1e-12,
        use_gpu_compute=True,
        gpu_all_stages=True,
        shader_dir="csrc/shaders",
    )
    origins = np.array([[x, 0.0, -0.05] for x in (-0.003, -0.001, 0.001, 0.003)])
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
    assert [lane["coherence_id"] for lane in active] == coherences
    for lane in range(4):
        s_field = tracer.wave_arena_field_snapshot(0, 0, 0, lane)
        p_field = tracer.wave_arena_field_snapshot(0, 0, 1, lane)
        s = np.asarray(s_field["re"]) + 1j*np.asarray(s_field["im"])
        p = np.asarray(p_field["re"]) + 1j*np.asarray(p_field["im"])
        illuminated = np.abs(s) > float(np.max(np.abs(s))) * 1.0e-6
        np.testing.assert_allclose(
            p[illuminated], 1j*s[illuminated], rtol=5e-5
        )
    assert arena["boundary"]["output_ray_power"] == pytest.approx(
        arena["boundary"]["propagated_field_power"], rel=3.0e-5
    )
    assert np.asarray(tracer.drain_records(16)["kind"]).tolist() == [3, 3, 3, 3]
