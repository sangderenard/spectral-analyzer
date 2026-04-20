from __future__ import annotations

import torch

from cavity_engine import (
    AtmosphericSpec,
    CavityAperture,
    CavityPanel,
    CavityReceiver,
    CavityScene,
    CavitySource,
    CavitySourceBand,
    CircularRoom,
    DiffuseTailSpec,
    MeshRoom,
    PolygonalRoom,
    SceneMaterial,
    SceneTriangle,
    atmospheric_attenuation,
    build_binaural_receivers,
    build_room_panels,
    build_test_scene_array,
    flush_cavity_stream_state,
    init_cavity_stream_state,
    load_obj_mesh_room,
    render_cavity_scene,
    render_cavity_scene_step,
)


def test_polygonal_room_builds_walls_floor_and_roof() -> None:
    room = PolygonalRoom(
        vertices_xy=[(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)],
        height=3.0,
    )
    panels = build_room_panels(room)

    assert len(panels) == 6
    assert {p.key for p in panels} >= {"floor", "roof", "wall_0", "wall_1", "wall_2", "wall_3"}


def test_circular_room_builds_segmented_walls() -> None:
    room = CircularRoom(radius=2.5, height=3.0, n_segments=8)
    panels = build_room_panels(room)

    assert len(panels) == 10
    assert panels[0].key == "wall_0"
    assert panels[-2].key == "floor"
    assert panels[-1].key == "roof"


def test_mesh_room_builds_two_sided_panels_with_materials() -> None:
    room = MeshRoom(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        triangles=[
            SceneTriangle(
                key="tri0",
                vertices=(0, 1, 2),
                material_key="front",
                back_material_key="back",
                winding="ccw",
            )
        ],
        materials={
            "front": SceneMaterial(key="front", reflectivity=0.8, absorption=0.1),
            "back": SceneMaterial(key="back", reflectivity=0.2, absorption=0.4),
        },
    )

    panels = build_room_panels(room)

    assert len(panels) == 2
    assert panels[0].reflectivity == 0.8
    assert panels[1].reflectivity == 0.2
    assert panels[0].normal == (0.0, 0.0, 1.0)
    assert panels[1].normal == (-0.0, -0.0, -1.0)


def test_atmospheric_attenuation_falls_with_distance_and_highband_weight() -> None:
    distances = torch.tensor([1.0, 5.0], dtype=torch.float64)
    atmosphere = AtmosphericSpec(attenuation_db_per_m=0.02, high_band_extra_db_per_m=0.08)

    low = atmospheric_attenuation(atmosphere, distances, high_band_weight=0.0)
    high = atmospheric_attenuation(atmosphere, distances, high_band_weight=1.0)

    assert low[0] > low[1]
    assert high[1] < low[1]


def test_build_binaural_receivers_offsets_ears_laterally() -> None:
    receivers = build_binaural_receivers((0.0, 0.0, 1.5), (0.0, -1.0, 0.0), ear_spacing_m=0.2)

    assert len(receivers) == 2
    assert receivers[0].position[0] > receivers[1].position[0]
    assert receivers[0].position[2] == receivers[1].position[2] == 1.5


def test_cavity_render_includes_direct_specular_and_diffuse_components() -> None:
    scene = CavityScene(
        sources=[CavitySource(key="src", position=(0.0, 0.0, 1.0))],
        receivers=[CavityReceiver(key="rx", position=(1.0, 0.0, 1.0))],
        geometry=PolygonalRoom(
            vertices_xy=[(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)],
            height=3.0,
        ),
        diffuse_tail=DiffuseTailSpec(taps_per_panel=4, decay_s=0.8, delay_spread_s=0.04, strength=0.4),
    )
    sig = torch.zeros((1, 256), dtype=torch.complex128)
    sig[0, 0] = 1.0 + 0.0j

    result = render_cavity_scene(scene, sig, sample_rate=256.0)

    assert result.receiver_signals.shape == (1, 256)
    assert torch.max(torch.abs(result.direct_signals)) > 0
    assert torch.max(torch.abs(result.specular_signals)) > 0
    assert torch.max(torch.abs(result.diffuse_signals)) > 0
    assert result.aperture_pressures.shape == (0, 256)
    assert result.panel_count == 6


def test_baffle_panels_are_first_class_geometry() -> None:
    room = PolygonalRoom(
        vertices_xy=[(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)],
        height=3.0,
        baffles=[
            CavityPanel(
                key="baffle_0",
                point=(0.0, 0.0, 1.5),
                normal=(1.0, 0.0, 0.0),
                reflectivity=0.9,
                diffusion=0.6,
                absorption=0.05,
                is_baffle=True,
            )
        ],
    )
    scene = CavityScene(
        sources=[CavitySource(key="src", position=(-1.0, 0.0, 1.5))],
        receivers=[CavityReceiver(key="rx", position=(1.0, 0.0, 1.5))],
        geometry=room,
    )
    sig = torch.zeros((1, 128), dtype=torch.complex128)
    sig[0, 0] = 1.0 + 0.0j

    result = render_cavity_scene(scene, sig, sample_rate=128.0)

    assert "baffle_0" in result.metadata["panel_keys"]
    assert result.panel_count == 7


def test_streaming_step_matches_full_render_for_main_signal_window() -> None:
    scene = build_test_scene_array("polygon")
    scene.band_split_mode = "fft"
    sig = torch.zeros((len(scene.sources), 192), dtype=torch.complex128)
    sig[0, 0] = 1.0 + 0.0j
    sig[1, 16] = 0.75 + 0.25j
    sig[2, 40] = 0.5 - 0.1j

    full = render_cavity_scene(scene, sig, sample_rate=256.0)

    state = init_cavity_stream_state(scene, 256.0)
    out_chunks = []
    for start in range(0, sig.shape[1], 48):
        step = render_cavity_scene_step(
            scene,
            sig[:, start:start + 48],
            sample_rate=256.0,
            state=state,
        )
        state = step.state
        out_chunks.append(step.chunk_output)
    streamed = torch.cat(out_chunks, dim=1)

    assert streamed.shape == full.receiver_signals[:, :sig.shape[1]].shape
    assert torch.allclose(streamed, full.receiver_signals[:, :sig.shape[1]], atol=1e-8, rtol=1e-6)
    assert flush_cavity_stream_state(state).shape[1] == state.tail_samples


def test_fir_streaming_path_runs_with_history_state() -> None:
    scene = build_test_scene_array("polygon")
    scene.band_split_mode = "fir"
    scene.band_split_fir_taps = 65
    sig = torch.zeros((len(scene.sources), 96), dtype=torch.complex128)
    sig[0, 0] = 1.0 + 0.0j
    state = init_cavity_stream_state(scene, 256.0)

    step = render_cavity_scene_step(
        scene,
        sig[:, :48],
        sample_rate=256.0,
        state=state,
    )

    assert step.chunk_output.shape == (len(scene.receivers), 48)
    assert step.state.history_samples == 64
    assert step.state.source_history.shape == (len(scene.sources), 64)


def test_build_test_scene_array_has_multiple_sources_and_receivers() -> None:
    scene = build_test_scene_array("circular")

    assert len(scene.sources) >= 3
    assert len(scene.receivers) >= 3
    assert len(scene.apertures) >= 3
    assert isinstance(scene.geometry, CircularRoom)
    assert scene.band_split_mode == "fir"


def test_aperture_pressures_are_reported_and_feedback_changes_field() -> None:
    base_scene = build_test_scene_array("polygon")
    base_scene.aperture_feedback_iterations = 0
    coupled_scene = build_test_scene_array("polygon")
    coupled_scene.aperture_feedback_iterations = 2
    sig = torch.zeros((len(base_scene.sources), 192), dtype=torch.complex128)
    sig[0, 0] = 1.0 + 0.0j
    sig[1, 32] = 0.7 + 0.1j

    base = render_cavity_scene(base_scene, sig, sample_rate=256.0)
    coupled = render_cavity_scene(coupled_scene, sig, sample_rate=256.0)

    assert base.aperture_pressures.shape == (len(base_scene.apertures), 192)
    assert torch.max(torch.abs(base.aperture_pressures)) > 0
    assert coupled.metadata["aperture_feedback_iterations"] == 2
    assert not torch.allclose(base.receiver_signals, coupled.receiver_signals)


def test_aperture_pressure_excludes_owned_direct_self_field() -> None:
    scene = CavityScene(
        sources=[CavitySource(key="src", position=(0.0, 0.0, 1.0), direction=(1.0, 0.0, 0.0))],
        receivers=[CavityReceiver(key="rx", position=(1.0, 0.0, 1.0))],
        geometry=PolygonalRoom(
            vertices_xy=[(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)],
            height=3.0,
        ),
        apertures=[CavityAperture(key="ap", source_key="src", position=(0.05, 0.0, 1.0), direction=(1.0, 0.0, 0.0))],
    )
    sig = torch.zeros((1, 128), dtype=torch.complex128)
    sig[0, 0] = 1.0 + 0.0j

    result = render_cavity_scene(scene, sig, sample_rate=256.0)

    assert torch.max(torch.abs(result.aperture_pressures)) == 0


def test_band_limited_directionality_keeps_bass_broad_and_treble_forward() -> None:
    scene = CavityScene(
        sources=[
            CavitySource(
                key="src",
                position=(0.0, 0.0, 1.0),
                direction=(1.0, 0.0, 0.0),
                gain=1.0,
                band_profiles=[
                    CavitySourceBand(low_hz=0.0, high_hz=180.0, cone_angle_deg=180.0, directivity_power=0.0, gain=1.0),
                    CavitySourceBand(low_hz=180.0, high_hz=None, cone_angle_deg=20.0, directivity_power=8.0, gain=1.0),
                ],
            )
        ],
        receivers=[
            CavityReceiver(key="front", position=(1.0, 0.0, 1.0)),
            CavityReceiver(key="side", position=(0.0, 1.0, 1.0)),
        ],
        geometry=PolygonalRoom(
            vertices_xy=[(-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)],
            height=3.0,
        ),
        diffuse_tail=DiffuseTailSpec(taps_per_panel=1, decay_s=0.3, delay_spread_s=0.01, strength=0.05),
    )
    sample_rate = 2048.0
    n = 256
    t = torch.arange(n, dtype=torch.float64) / sample_rate
    low = torch.exp(1j * (2.0 * torch.pi * 80.0 * t)).to(torch.complex128)
    high = torch.exp(1j * (2.0 * torch.pi * 896.0 * t)).to(torch.complex128)

    low_result = render_cavity_scene(scene, low[None, :], sample_rate=sample_rate)
    high_result = render_cavity_scene(scene, high[None, :], sample_rate=sample_rate)

    low_front = torch.max(torch.abs(low_result.direct_signals[0])).item()
    low_side = torch.max(torch.abs(low_result.direct_signals[1])).item()
    high_front = torch.max(torch.abs(high_result.direct_signals[0])).item()
    high_side = torch.max(torch.abs(high_result.direct_signals[1])).item()

    assert low_side > 0.4 * low_front
    assert high_side < 0.15 * high_front
    assert high_front > high_side


def test_reflection_incidence_changes_complex_phase_and_magnitude() -> None:
    panel = CavityPanel(
        key="wall_custom",
        point=(0.0, 0.0, 1.5),
        normal=(1.0, 0.0, 0.0),
        reflectivity=0.7,
        normal_reflectivity=0.95,
        grazing_reflectivity=0.35,
        normal_phase_rad=0.1,
        grazing_phase_rad=1.1,
        diffusion=0.3,
        absorption=0.05,
        is_baffle=True,
    )
    scene = CavityScene(
        sources=[CavitySource(key="src", position=(-0.8, 0.0, 1.5), direction=(1.0, 0.0, 0.0))],
        receivers=[
            CavityReceiver(key="near_normal", position=(-1.2, 0.0, 1.5)),
            CavityReceiver(key="grazing", position=(0.8, 2.0, 1.5)),
        ],
        geometry=PolygonalRoom(
            vertices_xy=[(-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)],
            height=3.0,
            baffles=[panel],
        ),
        diffuse_tail=DiffuseTailSpec(taps_per_panel=1, decay_s=0.3, delay_spread_s=0.01, strength=0.0),
    )
    sig = torch.zeros((1, 128), dtype=torch.complex128)
    sig[0, 0] = 1.0 + 0.0j

    result = render_cavity_scene(scene, sig, sample_rate=512.0)
    spec = result.specular_signals
    normal_peak_idx = int(torch.argmax(torch.abs(spec[0])).item())
    grazing_peak_idx = int(torch.argmax(torch.abs(spec[1])).item())
    normal_val = spec[0, normal_peak_idx]
    grazing_val = spec[1, grazing_peak_idx]

    assert torch.abs(normal_val) > torch.abs(grazing_val)
    assert abs(torch.angle(normal_val).item() - torch.angle(grazing_val).item()) > 0.2


def test_fir_and_fft_band_split_modes_are_consistent_for_projection() -> None:
    scene = CavityScene(
        sources=[
            CavitySource(
                key="src",
                position=(0.0, 0.0, 1.0),
                direction=(1.0, 0.0, 0.0),
                band_profiles=[
                    CavitySourceBand(low_hz=0.0, high_hz=256.0, cone_angle_deg=180.0, directivity_power=0.0, gain=1.0),
                    CavitySourceBand(low_hz=256.0, high_hz=None, cone_angle_deg=60.0, directivity_power=3.0, gain=1.0),
                ],
            )
        ],
        receivers=[
            CavityReceiver(key="front", position=(1.0, 0.0, 1.0)),
            CavityReceiver(key="side", position=(0.0, 1.0, 1.0)),
        ],
        geometry=PolygonalRoom(
            vertices_xy=[(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)],
            height=3.0,
        ),
        diffuse_tail=DiffuseTailSpec(taps_per_panel=1, decay_s=0.3, delay_spread_s=0.01, strength=0.0),
        band_split_mode="fir",
        band_split_fir_taps=65,
    )
    sample_rate = 2048.0
    n = 256
    t = torch.arange(n, dtype=torch.float64) / sample_rate
    sig = (
        0.7 * torch.exp(1j * (2.0 * torch.pi * 128.0 * t))
        + 0.4 * torch.exp(1j * (2.0 * torch.pi * 512.0 * t))
    ).to(torch.complex128)

    fir_result = render_cavity_scene(scene, sig[None, :], sample_rate=sample_rate)
    scene.band_split_mode = "fft"
    fft_result = render_cavity_scene(scene, sig[None, :], sample_rate=sample_rate)

    fir_peak = torch.max(torch.abs(fir_result.direct_signals)).item()
    fft_peak = torch.max(torch.abs(fft_result.direct_signals)).item()

    assert fir_result.metadata["band_split_mode"] == "fir"
    assert fft_result.metadata["band_split_mode"] == "fft"
    assert abs(fir_peak - fft_peak) / max(fir_peak, fft_peak, 1e-9) < 0.2


def test_load_obj_mesh_room_parses_vertices_and_faces(tmp_path) -> None:
    obj_path = tmp_path / "room.obj"
    obj_path.write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n",
        encoding="utf-8",
    )

    room = load_obj_mesh_room(str(obj_path))

    assert len(room.vertices) == 3
    assert len(room.triangles) == 1
    assert room.triangles[0].vertices == (0, 1, 2)
