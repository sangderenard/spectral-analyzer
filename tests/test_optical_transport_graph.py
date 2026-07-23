from __future__ import annotations

import numpy as np
import pytest

from camera_designer.camera_preset import simple_doublet_preset
from camera_designer.compound_optics import CompoundLens
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
    install_optical_graph,
    wave_context_nodes,
)


def _lens(lane_count: int) -> CompoundLens:
    preset = simple_doublet_preset()
    wavelengths = np.linspace(0.42, 0.70, lane_count)
    return CompoundLens.from_preset(
        preset,
        wavelengths_um=wavelengths,
    )


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


class _RecordingTracer:
    def __init__(self):
        self.clears = 0
        self.contexts = []

    def clear_scale_contexts(self):
        self.clears += 1

    def add_scale_context(self, **values):
        self.contexts.append(values)
        return len(self.contexts) - 1


def test_installer_places_real_adi_arena_without_registering_a_surrogate_lens():
    nodes, links = wave_context_nodes(
        "bench.wave",
        8,
        propagation=WavePropagationStyle.ADI_REFERENCE,
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
    ] == "not-yet-published-from-t4-state"


def test_installer_refuses_to_pass_planned_fft_off_as_current_adi_backend():
    nodes, links = wave_context_nodes(
        "bench.wave",
        4,
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
        "propagation": WavePropagationStyle.ADI_REFERENCE,
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
