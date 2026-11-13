"""Studio atlas bridge contract, without a GPU or a live account."""

import io
import json
import struct
from pathlib import Path

import numpy as np

import scene_orders
from camera_designer.obj_loader import load_obj
from camera_designer.studio_photo_geometry import prepare_photo_geometry
from camera_designer.studio_t5_scene import native_t5_scene_inputs, create_native_t5_tracer
from camera_software.render_assets import (
    RenderAssetCatalog, ink_order_for_request, plan_ink_atlas_bake,
)
from camera_software.studio_atlas_rest import (
    _obj_text, _packet, _snapshot, _studio_points, build_studio_atlas_upload,
    load_studio_atlas_contract, studio_linear_image,
)
from live_spectral_text_demo import InkAtlasStudioRestRenderer


def test_atlas_uses_timed_studio_shutter_and_original_authored_geometry():
    request = plan_ink_atlas_bake(("A",), RenderAssetCatalog()).next_request
    job = scene_orders.resolved_jobs(ink_order_for_request(request))[0]
    glyph = _studio_points(scene_orders._glyph_triangles(job, job["planes"][0]))
    vertices, _ids, _pbr, _names = load_obj(_obj_text((glyph - [0, 0, .18]) / .1), None)
    asset = {"digest": "a" * 64, "geometry": {"vertices": vertices.reshape(-1).tolist()}}
    packed = _packet(asset)
    payload = build_studio_atlas_upload(request, asset, shutter_seconds=.2, substeps=2)
    size, count, reserved = struct.unpack_from("<III", payload, 4)
    config = json.loads(payload[16:16 + size])
    assert payload[:4] == b"MRP1" and count == len(packed) and reserved == 0
    assert config["object_snapshot"] == _snapshot(asset, packed)
    assert config["exposure_state_frames"] == [{
        "t0": 0.0, "t1": .2, "pbr": config["fixtures"]["pbr"],
        "emitter_profiles": None,
    }]
    assert config["exposure_seconds"] == .2
    assert config["exposure_seconds"] != job["exposure"]["time_s"]
    assert len(config["fixtures"]["material_ids"]) > 0
    assert set(config["fixtures"]["material_ids"]) == {0, 1}
    assert config["object_snapshot"]["materials"]["index"] == {"ink": 0}
    np.testing.assert_allclose(packed["positions"].reshape(-1, 3, 3), vertices, rtol=0, atol=1e-6)
    # The same admitted geometry/material scene is a valid native T5 input.
    scene, *_ = prepare_photo_geometry(config, packed["positions"].reshape(-1, 3, 3),
                                       packed["id"],
                                       np.asarray(config["object_snapshot"]["materials"]["tables"]["pbr"]["data"],
                                                  np.float32).reshape(-1, 16))
    native = native_t5_scene_inputs(scene, [.45, .55, .65])
    np.testing.assert_array_equal(native["verts"], scene[3].reshape(-1, 9))
    np.testing.assert_array_equal(native["mat_buf"], scene[2])
    assert len(native["emitter_tri_ids"]) > 0
    tracer = create_native_t5_tracer(scene, [.45, .55, .65])
    try:
        assert tracer is not None
    finally:
        tracer.stop_pipeline()


def test_studio_hdr_keeps_linear_profiled_frame():
    image = np.full((128, 128, 3), .25, np.float32)
    stream = io.BytesIO()
    np.savez_compressed(stream, linear_srgb_exposure=image)
    np.testing.assert_array_equal(studio_linear_image(stream.getvalue()), image)


def test_saved_studio_photo_contract_supplies_camera_and_exposure(tmp_path):
    request = plan_ink_atlas_bake(("A",), RenderAssetCatalog()).next_request
    job = scene_orders.resolved_jobs(ink_order_for_request(request))[0]
    slate = _studio_points(scene_orders._box_plane_triangles(job["planes"][0]), center_z=0)
    vertices, *_ = load_obj(_obj_text(slate / .1), None)
    asset = {"digest": "a" * 64, "geometry": {"vertices": vertices.reshape(-1).tolist()}}
    saved = {"schema": "studio-atlas-photo-profile-v1", "photo": {
        "camera": {"eye_m": [0, 0, -.55], "target_m": [0, 0, .18],
                   "fov_degrees": 42, "aperture_radius_m": .002,
                   "focus_distance_m": .73},
        "transport": "fixed:8", "exposure_seconds": .3,
        "exposure_substeps": 3, "exposure_iso": 200,
        "object_center_m": [0, 0, .18], "camera_visible": True,
        "fixtures": load_studio_atlas_contract()["fixtures"],
    }}
    path = tmp_path / "studio-photo.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    contract = load_studio_atlas_contract(path)
    payload = build_studio_atlas_upload(request, asset, photo_contract=contract)
    size = struct.unpack_from("<I", payload, 4)[0]
    config = json.loads(payload[16:16 + size])
    assert config["camera"]["eye_m"] == [0, 0, -.55]
    assert config["camera"]["resolution"] == request.capture.width
    assert config["transport"] == "fixed:8"
    assert config["exposure_seconds"] == .3
    assert config["exposure_state_frames"][0]["t1"] == .3
    assert config["camera_visible"] is True
    saved["photo"]["fixtures"] = {"vertices": []}
    path.write_text(json.dumps(saved), encoding="utf-8")
    try:
        load_studio_atlas_contract(path)
    except ValueError as error:
        assert "fixture" in str(error)
    else:
        raise AssertionError("a profile must provide a valid saved scene")


def test_studio_rest_bake_enters_existing_atlas_catalog(tmp_path):
    class Studio:
        base_url = "https://studio.invalid"

        def upload_subject(self, subject):
            vertices, _ids, _pbr, _names = load_obj(_obj_text((subject - [0, 0, .18]) / .1), None)
            return {"digest": "a" * 64, "geometry": {"vertices": vertices.reshape(-1).tolist()}}

        def submit(self, payload):
            assert payload[:4] == b"MRP1"
            return {"digest": "b" * 64}

        def await_complete(self, _digest):
            return {"state": "complete", "hdr": "exposure-000001.npz",
                    "image": "preview-000001.png"}

        def artifact(self, _digest, _name, *, hdr=False):
            if not hdr:
                return b"png-from-studio"
            stream = io.BytesIO()
            np.savez_compressed(stream, linear_srgb_exposure=np.ones((128, 128, 3), np.float32))
            return stream.getvalue()

    catalog = RenderAssetCatalog(str(tmp_path / "catalog.json"))
    request = plan_ink_atlas_bake(("A",), catalog).next_request
    renderer = InkAtlasStudioRestRenderer(str(tmp_path), catalog,
                                           studio_url="https://studio.invalid",
                                           studio_cookie="session=fake")
    renderer.studio_client = Studio()
    record = renderer(request)
    assert record.metadata["backend"] == "studio-rest-timed"
    assert record.metadata["completion_basis"] == "studio_completed_shutter"
    assert plan_ink_atlas_bake(("A",), catalog).next_request is None
    manifest = json.loads(Path(record.manifest_path).read_text())
    assert manifest["clock_relation"] == "independent camera shutter and BDPT order settings"
