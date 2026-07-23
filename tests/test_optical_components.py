from __future__ import annotations

import time

import numpy as np
import pytest

from camera_designer.camera_preset import simple_doublet_preset
from camera_designer.compound_optics import CompoundLens
from camera_software.optical_components import (
    OPTICAL_COMPONENT_SCHEMA,
    CompoundLensComponent,
    OpticalEngine,
    PentaprismComponent,
    PhysicalApertureComponent,
    PlaneMirrorComponent,
    build_native_component_scene,
    default_optical_component_registry,
)
from camera_software.optical_transport_graph import OpticalExecutionDomain
from camera_software.physical_aperture import LivePhysicalAperture
from camera_software.specialty_optics import PentaprismSpec


@pytest.mark.parametrize("lane_count", [1, 3, 4, 8, 16, 32])
def test_default_registry_compiles_every_component_at_exact_lane_width(
    lane_count: int,
) -> None:
    registry = default_optical_component_registry()
    assert registry.keys() == (
        "aperture.iris",
        "lens.default-camera",
        "mirror.plane",
        "pentaprism.finder",
    )
    for key in registry.keys():
        compiled = registry.compile(key, lane_count)
        contract = compiled.contract()
        assert contract["schema"] == OPTICAL_COMPONENT_SCHEMA
        assert contract["lane_count"] == lane_count
        assert all(
            node["lane_count"] == lane_count
            for node in contract["graph"]["nodes"]
        )


def test_aperture_component_keeps_wave_interaction_local_and_physical() -> None:
    aperture = LivePhysicalAperture.iris(
        "bench.iris",
        blade_count=11,
        opening_radius_m=15e-6,
        assembly_radius_m=50e-6,
        thickness_m=4e-6,
    )
    compiled = PhysicalApertureComponent(aperture).compile(8)
    contract = compiled.contract()
    arena = next(
        node for node in contract["graph"]["nodes"]
        if node["domain"] == OpticalExecutionDomain.T4_WAVE_ARENA.value
    )

    assert arena["parameters"]["aperture_material"]["ideal_mask"] is False
    assert arena["parameters"]["state_layout"] == "solid-contiguous-state-block"
    assert contract["metadata"]["wave_localization"] == "material-bounds-only"
    assert contract["transport_triangle_count"] == 11*12


def test_lens_component_is_the_existing_fused_exact_t2_artifact() -> None:
    lens = CompoundLens.from_preset(
        simple_doublet_preset(),
        wavelengths_um=np.linspace(0.42, 0.70, 4),
    )
    compiled = CompoundLensComponent(lens).compile(4)
    contract = compiled.contract()

    assert np.array_equal(
        compiled.graph.t2_payloads["camera.exact-compound-lens"],
        lens.build_gpu_payload(),
    )
    assert contract["metadata"]["execution"] == "existing-fused-exact-t2"
    assert contract["metadata"]["hot_interpreter"] is False


def test_plane_mirror_uses_canonical_material_and_exact_reflection() -> None:
    mirror = PlaneMirrorComponent(
        normal=(-1.0, 1.0, 0.0),
        material_name="aluminum_mirror",
    )
    reflected = mirror.reflected_direction((1.0, 0.0, 0.0))
    np.testing.assert_allclose(reflected, (0.0, 1.0, 0.0), atol=1e-12)

    compiled = mirror.compile(4)
    node = next(
        node for node in compiled.graph.spec.nodes
        if node.operation == "analytic-plane-reflection"
    )
    assert node.domain is OpticalExecutionDomain.T3_MATERIAL
    assert node.parameters["jones_response"] == "canonical-material-conductor"
    assert compiled.material_roles[0].material_name == "aluminum_mirror"


def test_pentaprism_is_a_material_bound_composite_graph() -> None:
    component = PentaprismComponent(
        PentaprismSpec((0.0, 0.0, 0.0))
    ).compile(4)
    contract = component.contract()
    operations = [node["operation"] for node in contract["graph"]["nodes"]]

    assert contract["component_kind"] == "composite-pentaprism"
    assert operations.count("analytic-plane-reflection") == 2
    assert operations.count("dielectric-interface") == 2
    assert contract["transport_triangle_count"] == 16
    assert set(component.transport_geometry.roles) == {
        role.role for role in component.material_roles
    }
    assert contract["metadata"]["constant_deviation_rad"] == pytest.approx(
        0.5*np.pi
    )


def test_registry_rejects_unknown_component_without_fallback() -> None:
    with pytest.raises(KeyError, match="unknown optical component"):
        default_optical_component_registry().compile("not-a-component", 4)


def test_engine_selection_is_explicit_and_never_falls_back() -> None:
    registry = default_optical_component_registry()

    aperture = registry.compile("aperture.iris", 4, OpticalEngine.WAVE)
    assert aperture.metadata["selected_engine"] == "wave"
    lens = registry.compile("lens.default-camera", 4, "parametric")
    assert lens.metadata["selected_engine"] == "parametric"
    with pytest.raises(RuntimeError, match="Refusing fallback"):
        registry.compile("pentaprism.finder", 4, "wave")
    with pytest.raises(RuntimeError, match="Refusing fallback"):
        registry.compile("lens.default-camera", 4, "ray")


def _wait(tracer, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic()+timeout_s
    while tracer.in_flight_count() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert tracer.in_flight_count() == 0


def test_mirror_component_runs_through_native_t1_t3_pipeline() -> None:
    compiled = PlaneMirrorComponent(
        center_m=(0.0, 0.0, 0.0),
        normal=(-1.0, 0.0, 0.0),
    ).compile(1)
    scene = build_native_component_scene(
        compiled, np.asarray([299_792_458.0/550e-9])
    )
    tracer = scene.create_tracer()
    tracer.ensure_pipeline(max_children=1, min_amplitude=1e-12)
    tracer.submit_rays(
        np.asarray([[-0.05, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([[1.0+0.0j]]),
        max_bounces=2,
        min_amplitude=1e-12,
    )
    _wait(tracer)
    records = tracer.drain_records(16)
    kinds = np.asarray(records["kind"])
    directions = np.asarray(records["dir"])

    assert kinds.tolist() == [0, 2]
    np.testing.assert_allclose(directions[-1], (-1.0, 0.0, 0.0), atol=1e-7)
    tracer.stop_pipeline()


def test_pentaprism_native_scene_retains_glass_mirror_and_absorber_roles() -> None:
    compiled = PentaprismComponent(
        PentaprismSpec((0.0, 0.0, 0.0))
    ).compile(1)
    scene = build_native_component_scene(
        compiled, np.asarray([299_792_458.0/550e-9])
    )

    assert scene.role_interactions["entrance_glass"] == "dielectric-interface"
    assert scene.role_interactions["silvered_reflector_1"] == (
        "analytic-specular-reflection"
    )
    assert scene.role_interactions["blackened_prism_side"] == "absorber"
