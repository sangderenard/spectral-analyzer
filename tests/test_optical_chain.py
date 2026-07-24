from __future__ import annotations

import numpy as np
import pytest

from camera_software.light_table_experiment import (
    mount_aperture_at_lens_stop,
    run_geometric_light_table,
)
from camera_software.light_table_assembly import (
    HolderKind,
    LightTableAssemblySpec,
    LightTableHolderSpec,
    MountDatum,
    MountedOpticalElement,
    default_circular_mount,
    default_square_mount,
)
from camera_software.optical_chain import (
    EmitterEndpointComponent,
    SensorEndpointComponent,
    compile_optical_chain,
    light_table_chain,
)
from camera_software.optical_components import (
    default_optical_component_registry,
)
from camera_software.physical_aperture import LivePhysicalAperture
from camera_software.sensor_back import SensorBackProfile


def _lens_and_aperture(pattern: str = "iris"):
    registry = default_optical_component_registry()
    lens = registry.create("lens.default-camera", 1)
    if pattern == "iris":
        aperture = LivePhysicalAperture.iris(
            "table.iris",
            blade_count=7,
            opening_radius_m=30.0e-6,
            assembly_radius_m=52.0e-6,
            thickness_m=3.0e-6,
        )
    else:
        aperture = LivePhysicalAperture.grating(
            "table.grating",
            slit_width_m=7.0e-6,
            pitch_m=15.0e-6,
            assembly_radius_m=52.0e-6,
            thickness_m=3.0e-6,
        )
    return lens, mount_aperture_at_lens_stop(aperture, lens.lens)


def test_registry_exposes_emitter_and_sensor_as_components():
    registry = default_optical_component_registry()
    assert "emitter.laser-532nm" in registry.keys()
    assert "sensor.fullframe" in registry.keys()

    emitter = registry.compile("emitter.laser-532nm", 3)
    sensor = registry.compile("sensor.fullframe", 3)
    assert emitter.component_kind == "physical-emitter-endpoint"
    assert sensor.component_kind == "physical-sensor-endpoint"
    assert emitter.metadata["emitter"]["profile"]["phase"]
    assert sensor.metadata["sensor"]["sensor_chip"]["qe_peak"] > 0.0


def test_light_table_chain_preserves_component_artifacts_and_endpoints():
    lens, aperture = _lens_and_aperture()
    chain = compile_optical_chain(
        light_table_chain(lens, aperture, lane_count=1)
    )
    contract = chain.contract()

    assert [value.component_kind for value in chain.components] == [
        "physical-emitter-endpoint",
        "physical-aperture",
        "exact-compound-lens",
        "physical-sensor-endpoint",
    ]
    assert len(chain.connections) == 3
    assert len(chain.graph.t2_payloads) == 1
    assert any(
        value.get("aperture_material") is not None
        for value in chain.graph.t4_descriptors.values()
    )
    assert contract["graph"]["entry_keys"][0].endswith(".source")
    assert contract["output_node"].endswith(".reception")
    assert contract["components"][0]["component"]["metadata"]["emitter"]
    assert contract["components"][-1]["component"]["metadata"]["sensor"]


def test_named_aperture_instance_can_swap_without_changing_exact_lens_payload():
    lens, iris = _lens_and_aperture("iris")
    first_spec = light_table_chain(lens, iris, lane_count=1)
    first = compile_optical_chain(first_spec)
    _lens_again, grating = _lens_and_aperture("grating")
    second = compile_optical_chain(
        first_spec.replace_component("aperture", grating)
    )

    first_payload = next(iter(first.graph.t2_payloads.values()))
    second_payload = next(iter(second.graph.t2_payloads.values()))
    assert np.array_equal(first_payload, second_payload)
    first_aperture = next(
        value["aperture_material"]
        for value in first.graph.t4_descriptors.values()
        if value.get("aperture_material") is not None
    )
    second_aperture = next(
        value["aperture_material"]
        for value in second.graph.t4_descriptors.values()
        if value.get("aperture_material") is not None
    )
    assert first_aperture["pattern"] == "iris-polygon"
    assert second_aperture["pattern"] == "aperture-grille"


def test_geometric_light_table_projects_content_to_sensor_output():
    lens, aperture = _lens_and_aperture()
    sensor_x = max(
        value.x_pos for value in lens.lens.elements
    ) + abs(lens.lens.f_eff)
    sensor = SensorEndpointComponent(
        SensorBackProfile.fullframe_35mm(),
        center_m=(sensor_x, 0.0, 0.0),
        resolution=(48, 32),
    )
    chain = compile_optical_chain(
        light_table_chain(
            lens, aperture, sensor=sensor, lane_count=1,
        )
    )
    content = np.zeros((9, 9, 3), np.float32)
    content[2:7, 4] = (1.0, 0.2, 0.1)
    content[4, 2:7] = (0.1, 0.6, 1.0)

    output = run_geometric_light_table(
        chain, content, ray_count=1_200, seed=7
    )

    assert output.rays_transmitted > 0
    assert output.linear_sensor_rgb.shape == (32, 48, 3)
    assert output.developed_sensor_rgb.shape == (32, 48, 3)
    assert output.preview_rgba.shape == (32, 48, 4)
    assert np.count_nonzero(output.preview_rgba[..., :3]) > 0
    assert output.metadata["transport"] == "exact-parametric-geometric"
    assert output.metadata["diffraction"] is False


def test_mechanical_holders_deterministically_place_and_order_optical_chain():
    registry = default_optical_component_registry()
    emitter = registry.create("emitter.laser-532nm", 1)
    lens, aperture = _lens_and_aperture()
    sensor = registry.create("sensor.fullframe", 1)
    tube_mount = default_circular_mount(
        "M30-light-cell",
        outer_diameter_m=0.030,
        clear_diameter_m=0.024,
    )
    panel_mount = default_square_mount(
        "40mm-panel",
        outer_side_m=0.040,
        clear_side_m=0.032,
    )

    def mounted(
        key, component, station, sequence, mount, kind, datum
    ):
        holder = LightTableHolderSpec(
            f"{key}.holder",
            kind,
            mount,
            station,
            sequence=sequence,
        )
        return MountedOpticalElement(
            key, component, holder, mount, datum,
        )

    values = (
        mounted(
            "sensor", sensor, 0.140, 0, panel_mount,
            HolderKind.SENSOR_BACK, MountDatum.SENSOR_PLANE,
        ),
        mounted(
            "lens", lens, 0.050, 1, tube_mount,
            HolderKind.RAIL_CARRIAGE, MountDatum.FRONT_FACE,
        ),
        mounted(
            "emitter", emitter, -0.100, 0, panel_mount,
            HolderKind.EMITTER_PANEL, MountDatum.EMITTER_PLANE,
        ),
        mounted(
            "aperture", aperture, 0.050, 0, tube_mount,
            HolderKind.TUBE_CELL, MountDatum.APERTURE_STOP,
        ),
    )
    assembly = LightTableAssemblySpec("mounted-table", 1, values)
    resolved = assembly.resolve()
    chain = compile_optical_chain(resolved.chain)

    assert [
        value.instance_key for value in resolved.chain.elements
    ] == ["emitter", "aperture", "lens", "sensor"]
    assert resolved.apparatus_manifest["ordering"] == (
        "station-sequence-holder-key"
    )
    assert len(resolved.identity) == 64
    assert resolved.scene_manifest()["scene_kind"] == (
        "mounted_optical_light_table"
    )
    assert resolved.scene_manifest()["assembly_identity"] == resolved.identity
    assert chain.connections[0].distance_m == pytest.approx(0.150)
    assert chain.connections[1].distance_m == 0.0
    assert chain.connections[2].distance_m > 0.0
    aperture_geometry = chain.components[1].transport_geometry
    assert aperture_geometry is not None
    assert np.mean(aperture_geometry.triangles[..., 0]) == pytest.approx(0.050)
    assert np.ptp(aperture_geometry.triangles[..., 0]) > 0.0

    _lens_again, grating = _lens_and_aperture("grating")
    swapped = assembly.replace_mounted_component(
        "aperture", grating,
    ).resolve()
    assert swapped.identity != resolved.identity
    assert (
        swapped.mounted_elements[1].holder.key
        == resolved.mounted_elements[1].holder.key
    )


def test_mechanical_mount_shape_mismatch_rejects_optic_swap():
    registry = default_optical_component_registry()
    emitter = registry.create("emitter.laser-532nm", 1)
    circular = default_circular_mount(
        "small-cell",
        outer_diameter_m=0.030,
        clear_diameter_m=0.024,
    )
    square = default_square_mount(
        "small-cell",
        outer_side_m=0.030,
        clear_side_m=0.024,
    )
    holder = LightTableHolderSpec(
        "source.holder",
        HolderKind.EMITTER_PANEL,
        square,
        -0.1,
    )
    mounted = MountedOpticalElement(
        "emitter",
        emitter,
        holder,
        circular,
        MountDatum.EMITTER_PLANE,
    )

    with np.testing.assert_raises_regex(ValueError, "does not fit"):
        mounted.validate()
