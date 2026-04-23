from __future__ import annotations

import sys
import threading
import numpy as np


def _stub_analytic_driver_imports() -> None:
    """Augment the lightweight test stubs so analytic_driver can import."""
    locals_mod = sys.modules["pygame.locals"]
    for name in ["K_LCTRL", "K_RCTRL", "K_z", "K_DELETE", "K_n", "K_o", "K_d", "K_m", "K_l", "K_t", "K_RETURN", "K_BACKSPACE", "K_a", "K_EQUALS", "K_MINUS"]:
        setattr(locals_mod, name, 0)

    gl_mod = sys.modules["OpenGL.GL"]
    for name in [
        "GL_BLEND", "GL_CLAMP_TO_EDGE", "GL_COLOR_BUFFER_BIT", "GL_LINEAR",
        "GL_LINE_LOOP", "GL_LINE_STRIP", "GL_LINES", "GL_NEAREST",
        "GL_ONE_MINUS_SRC_ALPHA", "GL_QUADS", "GL_RGBA", "GL_SRC_ALPHA",
        "GL_TEXTURE_2D", "GL_TEXTURE_MAG_FILTER", "GL_TEXTURE_MIN_FILTER",
        "GL_TEXTURE_WRAP_S", "GL_TEXTURE_WRAP_T", "GL_TRIANGLES",
        "GL_UNSIGNED_BYTE", "GL_PROJECTION", "GL_MODELVIEW",
        "GL_SCISSOR_TEST",
    ]:
        setattr(gl_mod, name, 0)
    for name in [
        "glBegin", "glBindTexture", "glBlendFunc", "glClear", "glClearColor",
        "glColor4f", "glDeleteTextures", "glDisable", "glEnable", "glEnd",
        "glGenTextures", "glLineWidth", "glTexCoord2f", "glTexImage2D",
        "glTexParameteri", "glVertex2f", "glViewport", "glDrawPixels",
        "glLoadIdentity", "glMatrixMode", "glOrtho", "glScissor",
    ]:
        setattr(gl_mod, name, lambda *args, **kwargs: 0)


_stub_analytic_driver_imports()

from analytic_driver import (  # noqa: E402
    AnalyticMixer,
    AnalyticModule,
    AnalyticPatch,
    EditorCanvas,
    EditorMode,
    AnalyticVoice,
    PiecewiseVoiceEnvelope,
    Chair,
    GlobalTuning,
    NoteEvent,
    NoteSchedule,
    Part,
    PerformerPlacement,
    PlacementModule,
    QuantizerHandle,
    ParamNode,
    SidecarBus,
    _compute_arrangement_metrics,
    _load_sm_plugin,
    _sm_plugin_default_params,
    _sm_plugin_item_names,
    _sm_plugin_output_vars,
    _sm_plugin_state_vars,
    _sm_unwrap,
    _param_target_attr_specs,
    _param_target_node_specs,
    _patch_node_keys,
    _prepare_output_bus_for_device,
    _refresh_system_audio_report,
    _detected_envelope_artifact_paths,
    resolve_parts_from_patch,
    _synthesize_patch,
    _working_routing_graph_for_synthesis,
)
from routing_engine import RoutingEdge, solve_routing_complex  # noqa: E402


def test_working_graph_keeps_deleted_mix_edges_deleted() -> None:
    patch = AnalyticPatch()
    patch.voices = [
        AnalyticVoice(key="v1", label="V1"),
        AnalyticVoice(key="v2", label="V2"),
    ]
    patch.mixers = [AnalyticMixer(key="__mix__", label="Mix")]
    patch.routing.edges = [RoutingEdge(src_key="v1", dst_key="__mix__", weight=1.0)]

    working = _working_routing_graph_for_synthesis(patch)

    assert [(e.src_key, e.dst_key) for e in patch.routing.edges] == [("v1", "__mix__")]
    assert [(e.src_key, e.dst_key) for e in working.edges] == [("v1", "__mix__")]


def test_working_graph_adds_legacy_defaults_only_for_empty_graph() -> None:
    patch = AnalyticPatch()
    patch.voices = [AnalyticVoice(key="v1", label="V1")]
    patch.mixers = [AnalyticMixer(key="__mix__", label="Mix")]
    patch.routing.edges = []

    working = _working_routing_graph_for_synthesis(patch)

    assert patch.routing.edges == []
    assert [(e.src_key, e.dst_key, e.weight) for e in working.edges] == [("v1", "__mix__", 1.0)]


def test_quantizer_process_series_matches_scalar_discrete_mode() -> None:
    tuning = GlobalTuning()
    handle_vec = QuantizerHandle(tuning, [0, 2, 4, 5, 7, 9, 11], mode="discrete")
    handle_sca = QuantizerHandle(tuning, [0, 2, 4, 5, 7, 9, 11], mode="discrete")
    hz = np.array([220.0, 233.08, 246.94, 261.63, 277.18], dtype=np.float64)

    got = handle_vec.process_series(hz, hz, domain="hz", dt=1.0 / 48_000.0)
    expected = np.array(
        [handle_sca(v, v, domain="hz", dt=1.0 / 48_000.0) for v in hz],
        dtype=np.float64,
    )

    assert np.allclose(got, expected)


def test_reduced_nonlinear_solve_matches_linear_solution_for_identity_transform() -> None:
    src = np.array(
        [[1.0 + 0.0j, 2.0 + 0.0j, 0.5 + 0.0j],
         [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
         [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]],
        dtype=np.complex128,
    )
    edges = [
        RoutingEdge(src_key="a", dst_key="b", weight=0.2),
        RoutingEdge(src_key="b", dst_key="c", weight=0.15),
        RoutingEdge(src_key="c", dst_key="b", weight=-0.05),
    ]
    keys = ["a", "b", "c"]

    expected = solve_routing_complex(src, edges, keys, 48_000.0)
    got = solve_routing_complex(
        src,
        edges,
        keys,
        48_000.0,
        node_transforms={"b": lambda row: row},
    )

    assert np.allclose(got, expected)


def test_param_nodes_expose_wave_tabs_and_use_routed_series(monkeypatch) -> None:
    patch = AnalyticPatch()
    pn = ParamNode()
    pn.key = "p1"
    pn.label = "Param 1"
    patch.param_nodes = [pn]

    canvas = EditorCanvas()
    canvas.active_key = pn.key
    modes, _labels = canvas._visible_modes(patch)
    assert EditorMode.WAVEFORM in modes

    bus = SidecarBus()
    routed = np.array([1.0 + 0.5j, -0.25 + 0.0j, 0.75 - 0.25j], dtype=np.complex128)
    bus.put(pn.key, "routed", routed)

    def _fake_synth(*args, **kwargs):
        return (
            np.zeros(len(routed), dtype=np.float32),
            np.zeros(len(routed), dtype=np.float32),
            bus,
        )

    monkeypatch.setattr("analytic_driver._synthesize_patch", _fake_synth)

    canvas._rebuild_work(
        patch,
        pn.key,
        EditorMode.WAVEFORM,
        False,
        0,
        threading.Event(),
        "fp",
    )

    assert np.allclose(canvas._complex_sig, routed)
    assert np.allclose(canvas._signal, routed.real.astype(np.float32))


def test_param_target_specs_include_mixer_routing_knobs() -> None:
    patch = AnalyticPatch()
    patch.voices = [AnalyticVoice(key="v1", label="Voice 1")]
    patch.param_nodes = [ParamNode(key="p1", label="Param 1")]
    patch.mixers = [AnalyticMixer(key="mx", label="Main Mix")]

    node_specs = _param_target_node_specs(patch)
    assert ("mx", "≡ Main Mix") in node_specs

    attr_specs = _param_target_attr_specs(patch, "mx")
    assert ("routing.mix::v1::mx", "Mix / Voice 1 -> Main Mix") in attr_specs
    assert ("routing.angle::v1::mx", "Angle / Voice 1 -> Main Mix") in attr_specs
    assert ("routing.delay::v1::mx", "Delay / Voice 1 -> Main Mix") in attr_specs


def test_interaural_modules_expose_only_two_signal_channels() -> None:
    patch = AnalyticPatch()
    mod = AnalyticModule(key="iau1", label="IAU", module_type="interaural")
    patch.modules = [mod]

    keys = _patch_node_keys(patch)

    assert mod.key not in keys
    assert mod.ch1_key() in keys
    assert mod.ch2_key() in keys


def test_system_audio_nodes_appear_in_patch_node_keys() -> None:
    patch = AnalyticPatch()
    patch.system_audio.output_channels = 3
    patch.system_audio.input_channels = 2

    keys = _patch_node_keys(patch)

    assert "__sys_in_1__" in keys
    assert "__sys_in_2__" in keys
    assert "__sys_out_1__" in keys
    assert "__sys_out_2__" in keys
    assert "__sys_out_3__" in keys


def test_param_node_can_drive_mixer_routing_mix_knob() -> None:
    patch = AnalyticPatch()
    patch.duration = 0.1
    patch.preview_sr = 1024
    patch.normalize_output = False
    patch.voices = [AnalyticVoice(key="v1", label="Voice 1", freq_hz=32.0)]
    patch.mixers = [AnalyticMixer(key="mx", label="Main Mix")]
    patch.param_nodes = [
        ParamNode(
            key="p1",
            label="Param 1",
            targets=[{"voice_key": "mx", "attr": "routing.mix::v1::mx"}],
        )
    ]
    patch.routing.nodes = ["v1", "mx", "p1"]
    patch.routing.edges = [RoutingEdge(src_key="v1", dst_key="mx", weight=0.0)]
    patch._param_series_cache = {
        "p1": np.ones(int(patch.duration * patch.preview_sr), dtype=np.float64)
    }

    left, right = _synthesize_patch(patch)

    assert np.max(np.abs(left)) > 1e-6
    assert np.max(np.abs(right)) > 1e-6


def test_system_output_routing_supersedes_legacy_mixer_pcm() -> None:
    patch = AnalyticPatch()
    patch.duration = 0.05
    patch.preview_sr = 256
    patch.normalize_output = False
    patch.system_audio.output_channels = 2
    patch.voices = [AnalyticVoice(key="v1", label="Voice 1", freq_hz=32.0)]
    patch.mixers = [AnalyticMixer(key="mx", label="Main Mix", projection_active=False)]
    patch.routing.edges = [
        RoutingEdge(src_key="v1", dst_key="__sys_out_1__", weight=1.0),
        RoutingEdge(src_key="v1", dst_key="__sys_out_2__", weight=0.5),
    ]

    left, right, out_bus = _synthesize_patch(patch, _return_output_channels=True)

    assert out_bus.shape[1] == 2
    assert np.max(np.abs(left)) > 1e-6
    assert np.max(np.abs(right)) > 1e-6
    assert np.max(np.abs(right)) < np.max(np.abs(left))


def test_prepare_output_bus_for_device_resamples_and_maps_channels() -> None:
    src = np.column_stack([
        np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        np.linspace(1.0, -1.0, 8, dtype=np.float32),
    ])

    got = _prepare_output_bus_for_device(src, src_sr=8, dst_sr=16, dst_channels=4)

    assert got.shape == (16, 4)
    assert np.allclose(got[:, 0], got[:, 2])
    assert np.allclose(got[:, 1], got[:, 3])


def test_system_audio_report_does_not_mutate_configured_channel_counts(monkeypatch) -> None:
    from analytic_driver import SystemAudioDevice

    sysdev = SystemAudioDevice(output_channels=6, input_channels=5)

    monkeypatch.setattr("analytic_driver._list_audio_devices",
                        lambda iscapture: ["default-in"] if iscapture else ["default-out"])
    monkeypatch.setattr(
        "analytic_driver._probe_audio_device",
        lambda name, iscapture, requested_channels, requested_rate=48000: (
            "default-in" if iscapture else "default-out",
            2 if iscapture else 4,
            44100 if iscapture else 48000,
        ),
    )

    _refresh_system_audio_report(sysdev)

    assert sysdev.output_channels == 6
    assert sysdev.input_channels == 5
    assert sysdev._reported_output_hw_channels == 4
    assert sysdev._reported_input_hw_channels == 2


def test_waveform_plot_autoscales_to_signal_amplitude(monkeypatch) -> None:
    patch = AnalyticPatch()
    patch.duration = 0.1
    patch.preview_sr = 64
    voice = AnalyticVoice(key="v1", label="Voice 1", freq_hz=32.0)
    patch.voices = [voice]
    canvas = EditorCanvas()

    csig = np.array([0.0 + 0.0j, 3.0 + 0.0j, -2.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    monkeypatch.setattr("analytic_driver._synthesize_voice", lambda *args, **kwargs: csig)
    monkeypatch.setattr("analytic_driver._compute_envelope",
                        lambda *args, **kwargs: np.zeros(len(csig), dtype=np.float32))

    canvas._rebuild_work(patch, "v1", EditorMode.WAVEFORM, False, 0, threading.Event(), "fp_wave")

    assert canvas.v_hi > 3.0
    assert canvas.v_lo < -2.0


def test_state_machine_plugin_helpers_support_custom_item_names() -> None:
    class Plugin:
        STATE_VARS = ["x", "y", "x"]
        @staticmethod
        def item_names(n_items):
            return [f"mass_{i}" for i in range(n_items)]

    assert _sm_plugin_state_vars(Plugin) == ["x", "y"]
    assert _sm_plugin_item_names(Plugin, 3) == ["mass_0", "mass_1", "mass_2"]


def test_bound_particle_plugin_is_discoverable_and_declares_split_api() -> None:
    plug = _load_sm_plugin("bound_particle_circle")
    assert plug is not None
    assert _sm_plugin_output_vars(plug) == ["pos_x", "pos_y"]
    assert _sm_plugin_state_vars(plug) == ["pos_x", "pos_y", "vel_x", "vel_y"]
    defaults = _sm_plugin_default_params(plug)
    assert defaults["spring_k1"] == 35.0
    assert defaults["n_bindings"] == 12.0


def test_state_machine_plugin_helpers_load_orchestral_resonance() -> None:
    plug = _load_sm_plugin("orchestral_resonance")

    assert plug is not None
    assert _sm_plugin_output_vars(plug) == [
        "drive",
        "aperture_pressure",
        "feedback_pressure",
        "mic_left",
        "mic_right",
    ]
    assert _sm_plugin_state_vars(plug) == [
        "drive_rms",
        "aperture_peak",
        "feedback_peak",
        "mic_peak",
    ]
    defaults = _sm_plugin_default_params(plug)
    assert defaults["band_split_mode"] == "fir"
    assert defaults["feedback_iterations"] == 1
    assert defaults["room_shape"] == "polygon"
    assert defaults["temperature_c"] == 20.0
    assert _sm_plugin_item_names(plug, 2) == ["inst0", "inst1"]


def test_sm_unwrap_preserves_complex_outputs() -> None:
    arr = np.array([1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex128)

    got = _sm_unwrap(arr)

    assert got.dtype == np.complex128
    assert np.allclose(got, arr)


def test_state_machine_module_from_dict_populates_plugin_metadata() -> None:
    mod = AnalyticModule.from_dict({
        "module_type": "state_machine",
        "sm_plugin": "bound_particle_circle",
        "sm_n_items": 1,
    })
    assert mod.sm_vars == ["pos_x", "pos_y"]
    assert mod.sm_state_vars == ["pos_x", "pos_y", "vel_x", "vel_y"]
    assert mod.sm_items == ["particle"]
    assert mod.sm_params["mass"] == 1.0


def test_voice_body_type_round_trips() -> None:
    voice = AnalyticVoice(key="v1", label="Voice 1", body_type="string_plate")

    restored = AnalyticVoice.from_dict(voice.to_dict())

    assert restored.body_type == "string_plate"


def test_placement_module_syncs_owned_resonator_module() -> None:
    patch = AnalyticPatch()
    patch.voices = [
        AnalyticVoice(key="v1", label="V1", body_type="string_plate"),
        AnalyticVoice(key="v2", label="V2", body_type="reed_box"),
    ]
    patch.placement_resonator.room_shape = "circular"
    patch.placement_resonator.scene_path = "C:/rooms/test.obj"
    patch.placement_resonator.feedback_iterations = 2
    patch.placement_resonator.high_cone_deg = 55.0
    patch.placement_resonator.temperature_c = 24.0
    placement = PlacementModule(patch)

    mod = placement.ensure_resonator_module()

    assert mod.sm_plugin == "orchestral_resonance"
    assert mod.sm_n_items == 2
    assert mod.sm_use_torch is True
    assert mod.sm_params["room_shape"] == "circular"
    assert mod.sm_params["scene_path"] == "C:/rooms/test.obj"
    assert mod.sm_params["feedback_iterations"] == 2
    assert mod.sm_params["high_cone_deg"] == 55.0
    assert mod.sm_params["temperature_c"] == 24.0
    assert mod.sm_params["placement_body_types"] == "string_plate,reed_box"
    assert patch.placement_resonator.deployed_module_key == mod.key


def test_state_machine_nodes_expose_log_tab_and_capture_plugin_output(monkeypatch) -> None:
    patch = AnalyticPatch()
    mod = AnalyticModule(key="sm1", label="State Machine", module_type="state_machine")
    patch.modules = [mod]

    canvas = EditorCanvas()
    canvas.active_key = mod.key
    modes, _labels = canvas._visible_modes(patch)
    assert EditorMode.SM_LOG in modes

    def _fake_synth(fake_patch, *args, **kwargs):
        fake_patch.modules[0]._sm_log_text = "stdout line\nstderr line\nplugin line"
        return (
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            SidecarBus(),
        )

    monkeypatch.setattr("analytic_driver._synthesize_patch", _fake_synth)

    canvas._rebuild_work(
        patch,
        mod.key,
        EditorMode.SM_LOG,
        False,
        0,
        threading.Event(),
        "fp_sm_log",
    )

    assert canvas._sm_log_text == "stdout line\nstderr line\nplugin line"


def test_analytic_voice_polyphony_roundtrip() -> None:
    voice = AnalyticVoice(
        key="vpoly",
        label="Poly Voice",
        polyphony_count=3,
        polyphony_mode="unsympathetic",
    )

    restored = AnalyticVoice.from_dict(voice.to_dict())

    assert restored.polyphony_count == 3
    assert restored.polyphony_mode == "unsympathetic"


def test_piecewise_voice_payload_roundtrips() -> None:
    voice = AnalyticVoice(
        key="vpw",
        label="Piecewise Voice",
        env_type="piecewise",
        piecewise_env=PiecewiseVoiceEnvelope(),
    )
    voice.piecewise_env.curve.name = "amp_curve"
    voice.piecewise_env.curve.markers[0].label = "attack_end"
    voice.piecewise_env.source_path = "C:/tmp/fb_envelopes.npz"

    restored = AnalyticVoice.from_dict(voice.to_dict())

    assert restored.env_type == "piecewise"
    assert restored.piecewise_env is not None
    assert restored.piecewise_env.curve.name == "amp_curve"
    assert restored.piecewise_env.curve.markers[0].label == "attack_end"
    assert restored.piecewise_env.source_path.endswith("fb_envelopes.npz")


def test_detected_envelope_artifacts_include_inventory_paths(tmp_path) -> None:
    analysis_dir = tmp_path / "analysis_run"
    filterbank_dir = analysis_dir / "filterbank"
    filterbank_dir.mkdir(parents=True)
    env_path = filterbank_dir / "fb_envelopes.npz"
    env_path.write_bytes(b"placeholder")
    inv_path = analysis_dir / "analysis_inventory.json"
    inv_path.write_text(
        """
        {
          "schema": "analysis_inventory_v2",
          "datasets": [
            {
              "dataset_key": "fb:test",
              "engine": "fb",
              "artifacts": [
                {"kind": "envelopes", "path": "filterbank/fb_envelopes.npz"}
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    found = _detected_envelope_artifact_paths(str(tmp_path))

    assert str(env_path.resolve()) in found


def test_arrangement_solver_creates_multiple_chairs_from_polyphony() -> None:
    patch = AnalyticPatch()
    voice = AnalyticVoice(
        key="v1",
        label="Lead",
        register="mid",
        seq_role="melody",
        voice_role="signal",
        polyphony_count=2,
        polyphony_mode="sympathetic",
    )
    patch.voices = [voice]

    sched = NoteSchedule()
    sched.add(NoteEvent(220.0, 0.0, 1.0))
    sched.add(NoteEvent(330.0, 0.1, 0.9))
    sched.add(NoteEvent(440.0, 0.2, 0.8))
    sched._group_key = "mid+melody+signal"
    sched._layer_keys = ["mid", "mid+melody", "mid+melody+signal"]

    patch._arrangement_metrics = _compute_arrangement_metrics([(sched, [voice])])
    patch.parts = resolve_parts_from_patch(patch)
    part = patch.parts[0]

    assert part.solver_hints["required_chairs"] == 2
    assert part.player_count == 2
    assert len(part.chairs) == 2
    assert [ch.label for ch in part.chairs] == ["1st chair", "2nd chair"]
    assert part.solver_hints["page_specificity"] >= 100


def test_unsympathetic_sections_expand_to_multiple_performer_placements() -> None:
    patch = AnalyticPatch()
    patch.voices = [
        AnalyticVoice(
            key="v1",
            label="Voice 1",
            register="mid",
            seq_role="melody",
            voice_role="signal",
            polyphony_count=2,
            polyphony_mode="unsympathetic",
        ),
        AnalyticVoice(
            key="v2",
            label="Voice 2",
            register="mid",
            seq_role="melody",
            voice_role="signal",
            polyphony_count=2,
            polyphony_mode="unsympathetic",
        ),
    ]

    sched = NoteSchedule()
    sched.add(NoteEvent(220.0, 0.0, 1.0))
    sched.add(NoteEvent(330.0, 0.05, 0.95))
    sched.add(NoteEvent(440.0, 0.1, 0.9))
    sched._group_key = "mid+melody+signal"
    sched._layer_keys = ["mid+melody+signal"]

    patch._arrangement_metrics = _compute_arrangement_metrics([(sched, patch.voices)])
    patch.parts = resolve_parts_from_patch(patch)
    part = patch.parts[0]

    assert part.solver_hints["required_chairs"] == 6
    assert len(part.chairs) == 6

    placement = PlacementModule(patch)
    placement.set_player_count(part.key, 8)
    part = patch.parts[0]

    performers = [pf for ch in part.chairs for pf in ch.performers]
    assert sum(ch.performer_count for ch in part.chairs) == 8
    assert len(performers) == 8
    assert any(pf.assigned_note_keys for pf in performers)
    assert any(abs(pf.humanization_ms) > 0.0 for pf in performers)
    assert all(pf.geometric_delay_ms >= 0.0 for pf in performers)
    assert len(placement.placement_summary()[0]["chairs"]) == 6


def test_performer_phase_mode_roundtrip_and_coherent_default() -> None:
    patch = AnalyticPatch()
    assert patch.performer_phase_mode == "coherent"
    patch.performer_phase_mode = "individual"

    restored = AnalyticPatch.from_dict(patch.to_dict())

    assert restored.performer_phase_mode == "individual"


def test_synthesize_patch_uses_explicit_performer_phase_and_delay() -> None:
    patch = AnalyticPatch()
    patch.duration = 0.05
    patch.preview_sr = 256
    patch.normalize_output = False
    patch.performer_phase_mode = "individual"
    patch.voices = [AnalyticVoice(key="v1", label="Voice 1", freq_hz=8.0, amplitude=1.0)]
    patch.parts = [
        Part(
            key="p1",
            label="Part 1",
            register="mid",
            seq_role="melody",
            voice_role="signal",
            voice_keys=["v1"],
            player_count=1,
            chairs=[
                Chair(
                    key="p1:chair:1",
                    label="1st chair",
                    part_key="p1",
                    chair_index=1,
                    performer_count=1,
                    performers=[
                        PerformerPlacement(
                            key="p1:chair:1:performer:1",
                            label="Performer 1",
                            chair_key="p1:chair:1",
                            source_voice_keys=["v1"],
                            geometric_delay_ms=10.0,
                            humanization_ms=0.0,
                            phase_offset_rad=np.pi,
                        )
                    ],
                )
            ],
        )
    ]

    left, right = _synthesize_patch(patch)

    assert np.allclose(left[:2], 0.0)
    assert np.allclose(right[:2], 0.0)
    assert np.min(left[3:]) < 0.0
