from __future__ import annotations

import numpy as np
import pytest

from camera_designer.camera_preset import simple_doublet_preset
from camera_designer.compound_optics import CompoundLens
from camera_software.complex_optical_operators import (
    TransverseBasis,
    compile_planar_reflection_interface,
)
from camera_software.optical_transport_graph import (
    OPTICAL_GRAPH_SCHEMA,
    OpticalExecutionDomain,
    OpticalLinkSpec,
    OpticalNodeSpec,
    OpticalRepresentation,
    OpticalTransportGraphSpec,
    WaveBoundaryStyle,
    WavePropagationStyle,
    compile_compound_lens_graph,
    compile_optical_graph,
    compile_projector_back_graph,
    install_optical_graph,
    wave_context_nodes,
)
from camera_software.optical_transport_contracts import (
    OpticalAccuracySpec,
    OpticalProductKind,
    OpticalScatteringProductSpec,
    OpticalSolveReport,
    OpticalTimingSpec,
)


def _lens(lane_count: int) -> CompoundLens:
    preset = simple_doublet_preset()
    wavelengths = np.linspace(0.42, 0.70, lane_count)
    return CompoundLens.from_preset(
        preset,
        wavelengths_um=wavelengths,
    )


def test_graph_cold_compiles_rigid_field_interface_without_resampler():
    incoming = np.asarray((1.0, 0.0, 0.0))
    outgoing = np.asarray((1.0, 1.0, 0.0))/np.sqrt(2.0)
    normal = (incoming-outgoing)/np.linalg.norm(incoming-outgoing)
    interface = compile_planar_reflection_interface(
        TransverseBasis.from_direction(incoming),
        TransverseBasis.from_direction(outgoing),
        normal,
        np.linspace(450e-9, 650e-9, 4),
    )
    first = OpticalNodeSpec(
        "prism.leg-1", "wave-propagation",
        OpticalExecutionDomain.T4_WAVE_ARENA,
        OpticalRepresentation.TRANSVERSE_FIELD,
        OpticalRepresentation.TRANSVERSE_FIELD,
        4, persistent_state=True,
    )
    second = OpticalNodeSpec(
        "prism.leg-2", "wave-propagation",
        OpticalExecutionDomain.T4_WAVE_ARENA,
        OpticalRepresentation.TRANSVERSE_FIELD,
        OpticalRepresentation.TRANSVERSE_FIELD,
        4, persistent_state=True,
    )
    link = OpticalLinkSpec(
        first.key, second.key, "silvered-reflection",
        "rigid-complex-field-interface",
        interface.graph_parameters(),
    )
    compiled = compile_optical_graph(
        OpticalTransportGraphSpec(
            (first, second), (link,), (first.key,), (second.key,)
        ),
        field_interfaces={(first.key, second.key): interface},
    )

    assert compiled.field_interfaces[(first.key, second.key)] is interface
    assert compiled.contract()["field_interface_keys"] == [
        "prism.leg-1->prism.leg-2"
    ]
    assert compiled.operator_state_block["operators"].shape[0] == 5


@pytest.mark.parametrize("lane_count", [1, 3, 4, 8, 16, 32])
def test_exact_compound_payload_is_the_fused_t2_graph_artifact(lane_count):
    lens = _lens(lane_count)
    source_payload = lens.build_gpu_payload()

    compiled = compile_compound_lens_graph(lens, lane_count=lane_count)
    payload = compiled.t2_payloads["camera.exact-compound-lens"]
    contract = compiled.contract()

    assert np.array_equal(payload, source_payload)
    assert contract["schema"] == OPTICAL_GRAPH_SCHEMA
    assert contract["graph_engine"] == "GraphSolver"
    assert contract["execution"] == "compiled-not-hot-interpreted"
    assert contract["t2_payload_keys"] == ["camera.exact-compound-lens"]
    assert contract["t4_descriptor_keys"] == []


def test_projector_back_uses_reciprocal_exact_lens_graph():
    compiled = compile_projector_back_graph(_lens(4), lane_count=4)
    contract = compiled.contract()
    nodes = {node["key"]: node for node in contract["nodes"]}

    assert contract["entry_keys"] == ["camera.projector-back-port"]
    assert contract["product_keys"] == ["scene.projected-field"]
    assert nodes["camera.projector-back-port"]["directionality"] == "backward"
    assert nodes["camera.exact-compound-lens"]["directionality"] == "backward"
    assert np.array_equal(
        compiled.t2_payloads["camera.exact-compound-lens"],
        _lens(4).build_gpu_payload(),
    )
    assert [link["semantic_role"] for link in contract["links"]] == [
        "projector-back-entry",
        "projected-scene-product",
    ]


def test_projector_back_graph_carries_authored_source_contract_at_entry():
    source = {
        "schema": "physical-emissive-back-v1",
        "asset_key": "bench.home-baked",
    }
    compiled = compile_projector_back_graph(
        _lens(4),
        lane_count=4,
        source_contract=source,
    )
    entry = next(
        node for node in compiled.spec.nodes
        if node.key == "camera.projector-back-port"
    )

    assert entry.parameters["emissive_source"] == source
    assert entry.parameters["source_state"] == (
        "pipeline-owned-contiguous-block"
    )


def test_wave_context_defaults_to_vector_fft_and_absorbing_padding():
    compiled = compile_compound_lens_graph(
        _lens(8),
        lane_count=8,
        wave_key="camera.diffraction",
    )

    descriptor = compiled.t4_descriptors["camera.diffraction.arena"]

    assert descriptor["propagation"] == "angular-spectrum-fft"
    assert descriptor["boundary"] == "padded-absorbing"
    assert descriptor["state_layout"] == "solid-contiguous-state-block"
    assert descriptor["allocation"] == "cold-only"
    assert compiled.contract()["graph_stats"]["edges"] == 5


def test_periodic_fft_boundary_requires_explicit_authoring():
    with pytest.raises(ValueError, match="explicitly authored"):
        wave_context_nodes(
            "periodic-test",
            8,
            boundary=WaveBoundaryStyle.PERIODIC,
        )


def test_representation_change_without_adapter_or_adapter_node_is_rejected():
    ray = OpticalNodeSpec(
        "ray",
        "source",
        OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY,
        8,
    )
    field = OpticalNodeSpec(
        "field",
        "wave-propagation",
        OpticalExecutionDomain.T4_WAVE_ARENA,
        OpticalRepresentation.TRANSVERSE_FIELD,
        OpticalRepresentation.TRANSVERSE_FIELD,
        8,
        persistent_state=True,
    )
    spec = OpticalTransportGraphSpec(
        nodes=(ray, field),
        links=(OpticalLinkSpec("ray", "field"),),
        entry_keys=("ray",),
        product_keys=("field",),
    )

    with pytest.raises(ValueError, match="without an adapter"):
        compile_optical_graph(spec)


def test_graph_solver_preserves_branching_optical_topology():
    splitter = OpticalNodeSpec(
        "splitter",
        "polarized-beam-splitter",
        OpticalExecutionDomain.T2_PARAMETRIC,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY,
        4,
    )
    a = OpticalNodeSpec(
        "product.a",
        "pipeline-product",
        OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY,
        4,
    )
    b = OpticalNodeSpec(
        "product.b",
        "pipeline-product",
        OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY,
        4,
    )
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        nodes=(splitter, a, b),
        links=(
            OpticalLinkSpec("splitter", "product.a", "transmitted"),
            OpticalLinkSpec("splitter", "product.b", "reflected"),
        ),
        entry_keys=("splitter",),
        product_keys=("product.a", "product.b"),
    ))

    contract = compiled.contract()
    assert contract["graph_stats"]["nodes"] == 3
    assert contract["graph_stats"]["edges"] == 2
    assert {link["semantic_role"] for link in contract["links"]} == {
        "transmitted", "reflected",
    }
    assert contract["schedule"]["branch_node_keys"] == ["splitter"]
    assert contract["schedule"]["maximum_fanout"] == 2
    assert contract["schedule"]["requires_branch_frontier"] is True
    assert contract["schedule"]["requires_timed_worklist"] is False


def test_optical_schedule_preserves_continuous_product_timing():
    source = OpticalNodeSpec(
        "source", "source", OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY, 4,
    )
    product = OpticalNodeSpec(
        "product", "product", OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY, 4,
    )
    timing = OpticalTimingSpec.homogeneous(
        0.125, phase_index=1.51, group_index=1.53,
    )
    link = OpticalLinkSpec(
        source.key, product.key, "transmitted", product=
        OpticalScatteringProductSpec(
            "plate.transmitted",
            OpticalProductKind.TRANSMITTED,
            timing=timing,
            power_upper_bound=0.96,
        ),
    )
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        (source, product), (link,), (source.key,), (product.key,),
    ))
    contract = compiled.contract()

    assert contract["links"][0]["product"]["key"] == "plate.transmitted"
    assert contract["links"][0]["product"]["timing"]["optical_path_m"] == pytest.approx(
        0.125 * 1.51
    )
    assert contract["schedule"]["has_timestamped_edges"] is True
    assert contract["schedule"]["requires_timed_worklist"] is False
    assert contract["schedule"]["clock_policy"] == (
        "lcm-commensurate-event-timestamp-otherwise"
    )


def test_optical_schedule_detects_delayed_cycles_without_quantizing_them():
    a = OpticalNodeSpec(
        "a", "field", OpticalExecutionDomain.T4_WAVE_ARENA,
        OpticalRepresentation.TRANSVERSE_FIELD,
        OpticalRepresentation.TRANSVERSE_FIELD, 4,
        persistent_state=True,
    )
    b = OpticalNodeSpec(
        "b", "field", OpticalExecutionDomain.T4_WAVE_ARENA,
        OpticalRepresentation.TRANSVERSE_FIELD,
        OpticalRepresentation.TRANSVERSE_FIELD, 4,
        persistent_state=True,
    )
    sink = OpticalNodeSpec(
        "sink", "detector", OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.TRANSVERSE_FIELD,
        OpticalRepresentation.COMPLEX_RAY, 4,
    )
    delay = OpticalTimingSpec.homogeneous(0.01)
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        (a, b, sink),
        (
            OpticalLinkSpec(
                "a", "b", product=OpticalScatteringProductSpec(
                    "a-to-b", timing=delay,
                ),
            ),
            OpticalLinkSpec(
                "b", "a", product=OpticalScatteringProductSpec(
                    "b-to-a", timing=delay,
                ),
            ),
            # This zero-delay edge leaves the SCC and must not contaminate its
            # internal-delay telemetry.
            OpticalLinkSpec("b", "sink"),
        ),
        ("a",), ("sink",),
    ))

    assert len(compiled.schedule.cyclic_region_ids) == 1
    region = compiled.schedule.regions[compiled.schedule.cyclic_region_ids[0]]
    assert set(region.node_keys) == {"a", "b"}
    assert region.minimum_internal_group_delay_s == pytest.approx(
        delay.group_delay_s
    )
    assert region.has_positive_delay_edge is True
    assert region.contains_zero_delay_cycle is False
    assert compiled.schedule.requires_cycle_solver is True
    assert compiled.schedule.requires_timed_worklist is True


def test_optical_schedule_separates_zero_delay_scc_from_timed_work():
    node = OpticalNodeSpec(
        "feedback", "field", OpticalExecutionDomain.T4_WAVE_ARENA,
        OpticalRepresentation.TRANSVERSE_FIELD,
        OpticalRepresentation.TRANSVERSE_FIELD, 4,
        persistent_state=True,
    )
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        (node,), (OpticalLinkSpec(node.key, node.key),),
        (node.key,), (node.key,),
    ))

    region = compiled.schedule.regions[0]
    assert region.cyclic is True
    assert region.contains_zero_delay_cycle is True
    assert compiled.schedule.zero_delay_cycle_region_ids == (0,)
    assert compiled.schedule.requires_cycle_solver is True
    assert compiled.schedule.requires_timed_worklist is False


def test_stochastic_product_and_final_report_contracts_fail_loudly():
    with pytest.raises(ValueError, match="require a selection PDF"):
        OpticalScatteringProductSpec(
            "sampled.reflection", deterministic=False,
        ).validate()
    with pytest.raises(ValueError, match="must not carry"):
        OpticalScatteringProductSpec(
            "deterministic.reflection", selection_pdf=0.5,
        ).validate()

    accuracy = OpticalAccuracySpec(residual_power=1.0e-3)
    with pytest.raises(RuntimeError, match="residual-power"):
        OpticalSolveReport(
            input_power=1.0, residual_power=2.0e-3,
        ).validate(accuracy)
    with pytest.raises(RuntimeError, match="dropped"):
        OpticalSolveReport(
            input_power=1.0, dropped_power=1.0e-6,
        ).validate(accuracy)


def test_optical_graph_uses_lcm_clock_for_commensurate_component_rates():
    a = OpticalNodeSpec(
        "a", "component", OpticalExecutionDomain.T2_PARAMETRIC,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY, 4,
        parameters={"refresh_rate_hz": 120.0},
    )
    b = OpticalNodeSpec(
        "b", "component", OpticalExecutionDomain.T2_PARAMETRIC,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY, 4,
        parameters={"refresh_rate_hz": 90.0},
    )
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        (a, b), (OpticalLinkSpec("a", "b"),), ("a",), ("b",),
    ))

    assert compiled.schedule.clock_rate_hz == pytest.approx(360.0)
    assert compiled.topology_solver.sample_rate == pytest.approx(360.0)
    assert compiled.topology_solver.network_clock.dt == pytest.approx(1.0/360.0)


class _RecordingTracer:
    def __init__(self):
        self.clears = 0
        self.contexts = []
        self.wave_links = []

    def clear_scale_contexts(self):
        self.clears += 1

    def add_scale_context(self, **values):
        self.contexts.append(values)
        return len(self.contexts) - 1

    def add_wave_context_link(self, src_context_id, dst_context_id):
        self.wave_links.append((src_context_id, dst_context_id))


def test_installer_places_real_angular_arena_without_registering_surrogate_lens():
    nodes, links = wave_context_nodes(
        "bench.wave",
        8,
        propagation=WavePropagationStyle.ANGULAR_SPECTRUM_FFT,
        center_m=(0.1, 0.2, 0.3),
        axis=(2.0, 0.0, 0.0),
        radius_m=0.012,
        longitudinal_step_m=2.0e-5,
        longitudinal_steps=48,
        medium_n_real=1.33,
    )
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        nodes=nodes,
        links=links,
        entry_keys=(nodes[0].key,),
        product_keys=(nodes[-1].key,),
    ))
    tracer = _RecordingTracer()

    receipt = install_optical_graph(
        tracer,
        compiled,
        exact_t2_registered=False,
    )

    assert tracer.clears == 1
    assert len(tracer.contexts) == 1
    context = tracer.contexts[0]
    assert context["context_kind"] == 1
    assert context["scale_type"] == 1
    assert context["radius"] == pytest.approx(0.012)
    assert context["n_real"] == pytest.approx(1.33)
    assert np.array_equal(context["payload"][3:6], [1.0, 0.0, 0.0])
    assert receipt.borrowed_payloads[0] is context["payload"]
    assert receipt.contract()["transition_telemetry"][
        "field_texture_source"
    ] == "persistent-t4-state-display-resolve"
    assert receipt.contract()["transition_telemetry"]["field_texture_opt_in"] is True
    assert receipt.contract()["transition_telemetry"][
        "entry_adapter"
    ] == "unit-l2-gaussian-with-transverse-phase"
    arena_contract = receipt.contract()["wave_arenas"][0]
    assert arena_contract["boundary_geometry"] == "oriented-plane-to-plane-patch"
    assert arena_contract["longitudinal_extent_m"] == pytest.approx(9.6e-4)


def test_installer_lowers_field_to_field_edge_to_native_persistent_link():
    first_nodes, first_links = wave_context_nodes(
        "bench.first", 4, center_m=(0.0, 0.0, -8.0e-6),
        radius_m=64.0e-6, longitudinal_step_m=2.0e-6,
        longitudinal_steps=8,
    )
    second_nodes, second_links = wave_context_nodes(
        "bench.second", 4, center_m=(0.0, 0.0, 8.0e-6),
        radius_m=64.0e-6, longitudinal_step_m=2.0e-6,
        longitudinal_steps=8,
    )
    field_link = OpticalLinkSpec(
        first_nodes[1].key, second_nodes[1].key, "persistent-field-port"
    )
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        nodes=first_nodes + second_nodes,
        links=first_links + second_links + (field_link,),
        entry_keys=(first_nodes[0].key,),
        product_keys=(second_nodes[-1].key,),
    ))
    tracer = _RecordingTracer()

    receipt = install_optical_graph(
        tracer, compiled, exact_t2_registered=False
    )

    assert tracer.wave_links == [(0, 1)]
    assert len(receipt.wave_links) == 1
    assert receipt.contract()["wave_links"][0]["transfer"] == (
        "persistent-full-field"
    )
    assert receipt.contract()["wave_links"][0]["resampling"] == "forbidden"


def test_installer_lowers_wave_fanout_as_two_cold_native_products():
    def arena(key: str, center_z: float) -> OpticalNodeSpec:
        return OpticalNodeSpec(
            key, "wave-propagation", OpticalExecutionDomain.T4_WAVE_ARENA,
            OpticalRepresentation.TRANSVERSE_FIELD,
            OpticalRepresentation.TRANSVERSE_FIELD, 4,
            persistent_state=True,
            parameters={
                "propagation": WavePropagationStyle.ANGULAR_SPECTRUM_FFT.value,
                "boundary": WaveBoundaryStyle.PADDED_ABSORBING.value,
                "center_m": (0.0, 0.0, center_z),
                "axis": (0.0, 0.0, 1.0),
                "radius_m": 64.0e-6,
                "longitudinal_step_m": 2.0e-6,
                "longitudinal_steps": 8,
            },
        )

    source = arena("split.source", 0.0)
    reflected = arena("split.reflected", 16.0e-6)
    transmitted = arena("split.transmitted", 16.0e-6)
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        (source, reflected, transmitted),
        (
            OpticalLinkSpec(
                source.key, reflected.key, "reflected",
                product=OpticalScatteringProductSpec(
                    "split.reflected", OpticalProductKind.REFLECTED,
                ),
            ),
            OpticalLinkSpec(
                source.key, transmitted.key, "transmitted",
                product=OpticalScatteringProductSpec(
                    "split.transmitted", OpticalProductKind.TRANSMITTED,
                ),
            ),
        ),
        (source.key,), (reflected.key, transmitted.key),
    ))
    tracer = _RecordingTracer()

    receipt = install_optical_graph(
        tracer, compiled, exact_t2_registered=False,
    )

    assert compiled.schedule.branch_node_keys == (source.key,)
    assert tracer.wave_links == [(0, 1), (0, 2)]
    assert len(receipt.wave_links) == 2


def test_installer_refuses_wave_fanin_until_coherent_accumulator_exists():
    def arena(key: str) -> OpticalNodeSpec:
        return OpticalNodeSpec(
            key, "wave-propagation", OpticalExecutionDomain.T4_WAVE_ARENA,
            OpticalRepresentation.TRANSVERSE_FIELD,
            OpticalRepresentation.TRANSVERSE_FIELD, 4,
            persistent_state=True,
            parameters={
                "propagation": WavePropagationStyle.ANGULAR_SPECTRUM_FFT.value,
                "center_m": (0.0, 0.0, 0.0),
                "radius_m": 1.0e-3,
                "longitudinal_step_m": 1.0e-5,
                "longitudinal_steps": 1,
            },
        )

    a, b, joined = arena("a"), arena("b"), arena("joined")
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        (a, b, joined),
        (OpticalLinkSpec("a", "joined"), OpticalLinkSpec("b", "joined")),
        ("a", "b"), ("joined",),
    ))

    reception = compiled.schedule.receptions[0]
    assert reception.pool_id == 0
    assert reception.destination_node == "joined"
    assert tuple(
        predecessor.product_id for predecessor in reception.predecessors
    ) == (0, 1)
    assert compiled.contract()["schedule"]["receptions"][0]["policy"] == (
        "deterministic-coherent"
    )

    with pytest.raises(RuntimeError, match="coherent fan-in"):
        install_optical_graph(
            _RecordingTracer(), compiled, exact_t2_registered=False,
        )


def test_join_compiler_rejects_mixed_coherence_policy():
    a = OpticalNodeSpec(
        "a", "source", OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY, 1,
    )
    b = OpticalNodeSpec(
        "b", "source", OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY, 1,
    )
    joined = OpticalNodeSpec(
        "joined", "join", OpticalExecutionDomain.PIPELINE_PORT,
        OpticalRepresentation.COMPLEX_RAY,
        OpticalRepresentation.COMPLEX_RAY, 1,
    )
    spec = OpticalTransportGraphSpec(
        (a, b, joined),
        (
            OpticalLinkSpec(
                "a", "joined", product=OpticalScatteringProductSpec(
                    "a-product", coherent=True,
                ),
            ),
            OpticalLinkSpec(
                "b", "joined", product=OpticalScatteringProductSpec(
                    "b-product", coherent=False,
                ),
            ),
        ),
        ("a", "b"), ("joined",),
    )

    with pytest.raises(ValueError, match="mixes unsupported coherence"):
        compile_optical_graph(spec)


def test_installer_refuses_unsupported_split_step_substitution():
    nodes, links = wave_context_nodes(
        "bench.wave",
        4,
        propagation=WavePropagationStyle.SPLIT_STEP_FFT,
        center_m=(0.0, 0.0, 0.0),
        radius_m=0.01,
        longitudinal_step_m=1.0e-5,
        longitudinal_steps=16,
    )
    compiled = compile_optical_graph(OpticalTransportGraphSpec(
        nodes=nodes,
        links=links,
        entry_keys=(nodes[0].key,),
        product_keys=(nodes[-1].key,),
    ))

    with pytest.raises(RuntimeError, match="scientifically false"):
        install_optical_graph(
            _RecordingTracer(),
            compiled,
            exact_t2_registered=False,
        )


def test_installer_requires_camera_builder_confirmation_for_fused_t2():
    compiled = compile_compound_lens_graph(_lens(3), lane_count=3)

    with pytest.raises(RuntimeError, match="did not confirm"):
        install_optical_graph(
            _RecordingTracer(),
            compiled,
            exact_t2_registered=False,
        )


def test_compound_graph_can_chain_multiple_authored_wave_regions():
    physical = {
        "propagation": WavePropagationStyle.ANGULAR_SPECTRUM_FFT,
        "center_m": (0.0, 0.0, 0.0),
        "radius_m": 0.01,
        "longitudinal_step_m": 1.0e-5,
        "longitudinal_steps": 8,
    }
    compiled = compile_compound_lens_graph(
        _lens(4),
        lane_count=4,
        wave_regions=(
            ("bench.entry-wave", physical),
            ("bench.exit-wave", {
                **physical,
                "center_m": (0.0, 0.0, 0.04),
            }),
        ),
    )
    tracer = _RecordingTracer()

    receipt = install_optical_graph(
        tracer,
        compiled,
        exact_t2_registered=True,
    )

    assert [item.node_key for item in receipt.wave_arenas] == [
        "bench.entry-wave.arena",
        "bench.exit-wave.arena",
    ]
    assert len(tracer.contexts) == 2
