from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys

from camera_designer.camera_preset import (
    EmitterSpec,
    ProjectorBackSpec,
    simple_doublet_preset,
)
from camera_designer.emitter_profile import (
    CoherenceModel,
    EMITTER_CATALOG,
)
from camera_designer.ray_order import RayOrder
from camera_software.projector_back_transport import (
    ProjectorBackPreview,
    prepare_projector_back_launch,
    submit_projector_back,
)
from camera_designer.scene_builder import (
    build_tracer,
    build_gpu_scene,
    _collect_optical_geometry,
    _apply_element_surface_frame,
)
from camera_designer.compound_optics import CompoundLens
from material_db import MAX_SPECTRAL_BANDS


@pytest.fixture(scope="module")
def gpu_scene():
    preset = simple_doublet_preset()
    return preset, build_gpu_scene(
        preset,
        [],
        wavelengths_um=preset.wavelengths,
    )


def test_camera_bench_builds_shared_split_gpu_contract(gpu_scene):
    _preset, scene = gpu_scene
    (geom, shade, mat_buf, bvh_tris, contexts, ctx_map,
     bounds, sources) = scene

    assert geom.ndim == 2 and geom.shape[1] == 16
    assert shade.shape == geom.shape
    assert bvh_tris.shape == (len(geom), 3, 3)
    assert mat_buf.ndim == 2 and mat_buf.shape[1] == 12
    assert mat_buf.shape[0] % MAX_SPECTRAL_BANDS == 0
    assert contexts.shape[1] == 8
    assert len(contexts) == len(ctx_map)
    assert sources.ndim == 2 and sources.shape[1] == 12
    assert geom.flags.c_contiguous
    assert shade.flags.c_contiguous
    assert mat_buf.flags.c_contiguous
    assert len(bounds) == 2


def test_projector_back_lowers_to_common_coherent_emitter_without_catalog_mutation():
    preset = simple_doublet_preset()
    before = EMITTER_CATALOG["laser_532nm_green"].spectral.radiant_exitance
    preset.projector_back = ProjectorBackSpec(
        enabled=True,
        power=3.0,
        profile_name="laser_532nm_green",
        spatial_samples=17,
    )

    spec = preset.projector_back.as_emitter_spec(
        sensor_z_pos=preset.sensor.z_pos,
        sensor_radius=preset.sensor.r_max,
        wavelengths_um=[0.532],
    )
    profile = spec.resolve_profile()

    assert isinstance(spec, EmitterSpec)
    assert spec.radius == pytest.approx(preset.sensor.r_max)
    assert spec.spatial_samples == 17
    assert profile.phase.model is CoherenceModel.COHERENT
    assert profile.spectral.radiant_exitance == pytest.approx(3.0 * before)
    assert EMITTER_CATALOG["laser_532nm_green"].spectral.radiant_exitance == before


def test_home_baked_projector_back_is_enabled_serializable_and_self_describing():
    field = np.zeros((2, 3, 2), np.complex128)
    field[..., 0] = 1.0
    field[:, 1:, 1] = 0.5j
    back = ProjectorBackSpec.from_jones_field(
        field,
        asset_key="bench.baked.test",
        power=2.0,
    )

    restored = ProjectorBackSpec.from_dict(back.to_dict())
    contract = restored.source_contract([0.532], plane_radius_m=0.028)
    compact = restored.source_contract(
        [0.532], plane_radius_m=0.028, include_profile=False,
    )

    assert restored.enabled is True
    assert restored.spatial_samples == 6
    assert contract["schema"] == "physical-emissive-back-v1"
    assert contract["asset_key"] == "bench.baked.test"
    assert contract["texture_shape"] == [2, 3, 10]
    assert contract["transport"]["polarization"] == (
        "jones-source-mode-block"
    )
    assert "profile" in contract
    assert "profile" not in compact
    assert compact["texture_shape"] == [2, 3, 10]


def test_projector_back_gpu_launch_covers_large_source_plane_with_fixed_phase():
    preset = simple_doublet_preset()
    preset.projector_back = ProjectorBackSpec(
        enabled=True,
        power=1.0,
        profile_name="laser_532nm_green",
        spatial_samples=32,
    )
    *_, source_buf = build_gpu_scene(
        preset,
        [],
        wavelengths_um=[0.532],
    )

    radial = np.linalg.norm(source_buf[:, :2], axis=1)
    assert np.count_nonzero(radial > 0.25 * preset.sensor.r_max) > 0
    assert np.max(radial) <= preset.sensor.r_max * (1.0 + 1.0e-6)
    assert np.ptp(source_buf[:, 9]) == pytest.approx(0.0, abs=1.0e-7)


def test_emitter_exitance_is_linear_not_quadratic():
    profile = EMITTER_CATALOG["laser_532nm_green"]
    inline = profile.to_dict()
    inline["spectral"]["radiant_exitance"] = 3.0
    order = RayOrder.from_emitter_specs(
        [EmitterSpec(profile_name="", profile_inline=inline)],
        wavelengths_um=np.array([0.532]),
    )

    assert order.sources[0].amplitude_weights[0] == pytest.approx(3.0)


def test_projection_geometry_keeps_transmissive_sensor_scrim_without_black_back_wall():
    preset = simple_doublet_preset()
    preset.projector_back = ProjectorBackSpec(
        enabled=True,
        scrim_transmission=[0.2, 0.5, 0.8],
        scrim_diffusion=0.07,
    )
    result = _collect_optical_geometry(
        preset, None, preset.wavelengths, None,
    )
    verts = result[0]
    ior_specs = result[5]
    mat_groups = result[7]
    points = verts.reshape(-1, 3)
    sensor_plane_vertices = np.isclose(
        points[:, 2], preset.sensor.z_pos, atol=1.0e-10
    )

    assert np.any(sensor_plane_vertices)
    assert any(spec[-1] == "sensor_scrim" for spec in ior_specs)
    assert any(group[-1] == "sensor_scrim" for group in mat_groups)
    assert not any(
        group[-1] == "aperture_stop"
        and np.allclose(
            verts[group[0]:group[0] + group[1], (2, 5, 8)],
            preset.sensor.z_pos - 0.002,
        )
        for group in mat_groups
    )
    assert np.any(np.isclose(
        points[:, 2],
        preset.sensor.z_pos - preset.projector_back.z_offset,
        atol=1.0e-10,
    ))


def test_projector_launch_enters_native_exact_camera_frame_as_complex_lanes():
    preset = simple_doublet_preset()
    preset.wavelengths = [0.532]
    preset.projector_back = ProjectorBackSpec(
        enabled=True,
        profile_name="laser_532nm_green",
        spatial_samples=8,
    )
    launch = prepare_projector_back_launch(preset, n_rays=64, seed=7)

    assert launch.origins.shape == (64, 3)
    assert launch.directions.shape == (64, 3)
    assert launch.amplitudes.shape == (64, 1)
    assert np.all(launch.directions[:, 0] < 0.0)
    assert np.ptp(np.angle(launch.amplitudes[:, 0])) == pytest.approx(
        0.0, abs=1.0e-7
    )
    assert launch.profile_contract["source"] == "camera.projector-back-port"

    class RecordingTracer:
        def configure_complex_source_modes(
            self, tags, indices, words, bases, operators,
        ):
            self.source_block = (
                tags, indices, words, bases, operators,
            )

        def submit_rays(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    tracer = RecordingTracer()
    submitted = submit_projector_back(tracer, launch, shader_dir="shaders")
    assert submitted == 64
    assert tracer.args[2] is launch.amplitudes
    assert np.all(tracer.kwargs["color_flags"] == 64)
    assert tracer.kwargs["use_gpu_compute"] is True
    assert tracer.kwargs["gpu_all_stages"] is True
    assert tracer.kwargs["tags"] is launch.ray_tags
    assert np.array_equal(tracer.source_block[0], launch.ray_tags)
    assert launch.profile_contract["emissive_back"]["schema"] == (
        "physical-emissive-back-v1"
    )


def test_baked_jones_projector_field_reaches_pipeline_source_modes():
    preset = simple_doublet_preset()
    preset.wavelengths = [0.532]
    field = np.zeros((2, 2, 2), np.complex128)
    field[0, 0] = (1.0 + 0.0j, 0.0 + 0.0j)
    field[0, 1] = (0.0 + 0.0j, 1.0j)
    field[1, 0] = (0.25 + 0.25j, 0.5 + 0.0j)
    field[1, 1] = (1.0j, 0.5 + 0.5j)
    preset.projector_back = ProjectorBackSpec.from_jones_field(
        field,
        spatial_samples=4,
        asset_key="bench.jones.quadrants",
    )

    launch = prepare_projector_back_launch(preset, n_rays=16, seed=3)
    mode_floats = launch.source_state["source_modes"].view(np.float32)

    assert launch.ray_count == 16
    assert launch.source_state["source_modes"].shape == (16, 12)
    assert np.unique(np.round(mode_floats[:, :4], 5), axis=0).shape[0] > 1
    assert np.ptp(np.abs(launch.amplitudes[:, 0])) > 0.0
    assert launch.profile_contract["texture_shape"] == [2, 2, 10]


def test_projector_preview_only_schedules_the_existing_native_pipeline(tmp_path):
    preset = simple_doublet_preset()
    preset.wavelengths = [0.532]
    preset.projector_back = ProjectorBackSpec(
        enabled=True,
        profile_name="laser_532nm_green",
    )

    class PipelineTracer:
        def __init__(self):
            self.busy = 0
            self.submissions = []
            self.stopped = False

        def in_flight_count(self):
            return self.busy

        def configure_complex_source_modes(
            self, tags, indices, words, bases, operators,
        ):
            self.source_block = (
                tags, indices, words, bases, operators,
            )

        def submit_rays(self, *args, **kwargs):
            self.submissions.append((args, kwargs))
            self.busy = 1

        def stop_pipeline(self):
            self.stopped = True

    tracer = PipelineTracer()
    preview = ProjectorBackPreview(
        tracer,
        preset,
        shader_dir=str(tmp_path),
        ray_count=32,
    )
    first = preview.poll()
    assert first["generation"] == 0
    assert len(tracer.submissions) == 1
    assert tracer.submissions[0][0][2].shape == (32, 1)
    assert np.all(tracer.submissions[0][1]["color_flags"] == 64)

    tracer.busy = 0
    complete = preview.poll()
    assert complete["generation"] == 1
    assert complete["profile_contract"]["source"] == (
        "camera.projector-back-port"
    )
    preview.close()
    assert tracer.stopped is True


def test_camera_glass_keeps_dispersion_in_central_matbuf(gpu_scene):
    preset, scene = gpu_scene
    _geom, shade, mat_buf, *_rest = scene

    # The first geometry group is a lens surface.  TriShade slot 14 stores its
    # central MatBuf index as uint bits in a float, matching the GLSL contract.
    lens_mat_idx = int(shade[:1, 14].copy().view(np.uint32)[0])
    rows = mat_buf.reshape(-1, MAX_SPECTRAL_BANDS, 12)[lens_mat_idx]
    active = rows[:len(preset.wavelengths)]

    assert np.all(active[:, 0] > 0.0)
    assert np.ptp(active[:, 7]) > 1.0e-4
    # Wavelengths are authored blue -> red; ordinary crown glass index falls.
    assert np.all(np.diff(active[:, 7]) < 0.0)


def test_tessellated_lenses_retain_authored_camera_space_stations():
    preset = simple_doublet_preset()
    verts, _normals, *_tail = _collect_optical_geometry(
        preset, None, preset.wavelengths, None,
    )
    ior_specs = _tail[-4]
    transmissive = [spec for spec in ior_specs if spec[-1] == "transmissive"]
    assert len(transmissive) == 3
    for expected, (tri_start, n_tris, *_) in zip(
        (0.060, 0.057, 0.054), transmissive,
    ):
        z = verts[tri_start:tri_start + n_tris, (2, 5, 8)]
        assert float(np.min(np.abs(z - expected))) < 2.0e-6


def test_surface_frame_applies_decenter_and_tilt_to_preview_geometry():
    element = simple_doublet_preset().lens_group.elements[0]
    element.shift_xy_m = (0.002, -0.003)
    element.tilt_xy_deg = (5.0, -7.0)
    verts = np.array([[0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0]])
    transformed, transformed_normals = _apply_element_surface_frame(
        verts, normals, element,
    )
    points = transformed.reshape(-1, 3)
    assert points[0] == pytest.approx([0.002, -0.003, element.z_vertex])
    assert not np.allclose(transformed_normals[0], normals[0])
    assert np.linalg.norm(transformed_normals[0]) == pytest.approx(1.0)


def test_default_doublet_iris_is_at_the_lens_not_near_the_sensor():
    preset = simple_doublet_preset()
    assert preset.aperture_stop.z_pos == pytest.approx(0.050)
    assert preset.lens_group.elements[-1].z_vertex - preset.aperture_stop.z_pos \
        == pytest.approx(0.004)
    assert preset.aperture_stop.z_pos - preset.sensor.z_pos > 0.050


def test_mesh_only_camera_transport_requires_diagnostic_escape_hatch():
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        build_tracer(
            simple_doublet_preset(),
            [],
            exact_lens_transport=False,
        )


def test_canonical_axis_adapter_keeps_exact_spectral_payload_tight():
    preset = simple_doublet_preset()
    lens = CompoundLens.from_preset(
        preset,
        wavelengths_um=preset.wavelengths,
        axial_scale=-1.0,
    )
    payload = lens.build_gpu_payload()
    axial_positions = [
        float(element.x_pos) for element in lens._elements
    ]
    assert axial_positions == sorted(axial_positions)
    assert int(payload[5]) == len(preset.wavelengths)
    assert len(payload) == (
        8 + int(payload[1]) * 8
        + int(payload[1]) * 2 * len(preset.wavelengths)
    )


def test_camera_station_has_no_optical_transport_builder():
    # Import in a clean process because demo_pluck_gl and the duty-station
    # registry intentionally import each other during application bootstrap.
    check = subprocess.run(
        [
            sys.executable,
            "-c",
                (
                    "import camera_designer_station as s; "
                    "assert s._HAS_CAMERA_DESIGNER; "
                    "assert not hasattr(s, 'build_gpu_scene'); "
                    "assert not hasattr(s, 'build_tracer'); "
                    "assert s.BakeWorker is None"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr


def test_camera_designer_exposes_pluck_render_contract():
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import camera_designer_station as s; "
                "o=s.CameraDesignerStation.__new__(s.CameraDesignerStation); "
                "o.win_w=1400; o.win_h=900; calls=[]; "
                "o.draw=lambda w,h: calls.append((w,h)); o.render(); "
                "assert calls == [(1400,900)]"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr


def test_camera_designer_accepts_authored_transport_without_owning_a_solver():
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import camera_designer_station as s; "
                "o=s.CameraDesignerStation.__new__(s.CameraDesignerStation); "
                "calls=[]; o._request_optical_preview=lambda:calls.append(1); "
                "o.configure_transport('mixed', ({'transport':'wave','key':'w'},)); "
                "assert o._transport_mode=='mixed'; "
                "assert o._transport_contexts[0]['key']=='w'; "
                "assert calls==[1]"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr


def test_sensor_drag_invalidates_the_optical_backend():
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import types, pygame; "
                "import camera_designer_station as s; "
                "p=s._ComponentTreePanel(s.simple_doublet_preset()); p.render(230,900); "
                "p._drag_knob='sensor_z'; p._drag_start=(0,0); p._drag_val0=0.0; "
                "used,key=p.handle_event(pygame.event.Event(pygame.MOUSEMOTION,{'pos':(10,0)})); "
                "assert used and key=='sensor_z'; "
                "o=s.CameraDesignerStation.__new__(s.CameraDesignerStation); "
                "o.win_w=1400; o.win_h=900; o._left_panel=types.SimpleNamespace(" 
                "handle_event=lambda *a,**k:(True,'sensor_z')); "
                "o._right_panel=types.SimpleNamespace(); o._panels_dirty=False; "
                "o._rebuild_lines=lambda:None; calls=[]; "
                "o._request_optical_preview=lambda:calls.append('backend-request'); "
                "assert o.handle_event(types.SimpleNamespace()) is True; "
                "assert calls==['backend-request']"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr
