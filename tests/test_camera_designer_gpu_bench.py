from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys

from camera_designer.camera_preset import simple_doublet_preset
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
