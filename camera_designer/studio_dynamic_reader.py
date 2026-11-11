"""Bounded subject-only scene revisions at the incumbent live sweep boundary.

The constructor is called with the actual worker config and installed CPU
scene, never with a caller's assertion that an Exposure scene matches Studio.
No network endpoint, timer, ray scheduler or replacement scene packer lives
here. Authored vertex normals are retained in the receipt; current Spectral
TriGeom consumes geometric face normals. Unsupported optics fail explicitly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading

import numpy as np

from .studio_photo_geometry import prepare_photo_geometry


def _f32(value, shape, name):
    result = np.asarray(value, dtype=np.float32).reshape(shape).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite dynamic {name}")
    result.flags.writeable = False
    return result


def _equal(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


class StudioDynamicReader:
    """One immutable pending packet, installed only by the owning GL thread.

    A failed installation poisons this reader. The owning loop must stop,
    never issue another ray or acknowledge a partially installed revision.
    This first adapter admits moving, nonemissive triangle subjects only.
    Room/light/material/camera changes require a new initialized candidate.
    """

    def __init__(self, renderer, field, config, installed_scene, material_ids, pbr):
        if config.get("transport") != "fixed:3" or "fixtures" not in config:
            raise ValueError("dynamic reader needs an actual full Studio fixed:3 scene")
        if config.get("sphere_coverage", "mesh") != "mesh" or any(
                config.get(k) for k in ("analytic_spheres", "boundary_media", "optical_media", "region_media")):
            raise ValueError("dynamic analytic/optical-region ownership is not supported; retain the original reader")
        if field.get("sphere_count", 0) or field.get("region_count", 0):
            raise ValueError("installed analytic/region reader cannot accept this triangle adapter")
        if not _equal(field["source_buf"], installed_scene[7]):
            raise ValueError("installed forward sources do not match the worker scene")
        self.renderer, self.field = renderer, field
        self.config = copy.deepcopy(config)
        static_vertices = _f32(config["fixtures"]["vertices"], (-1,3,3), "static vertices")
        static_areas = np.linalg.norm(np.cross(static_vertices[:,1]-static_vertices[:,0],
                                              static_vertices[:,2]-static_vertices[:,0]), axis=1)
        if np.any(static_areas <= 1e-12):
            raise ValueError("canonical ray packer would drop a static scene triangle")
        self.ids = np.asarray(material_ids, np.int32).copy()
        self.pbr = _f32(pbr, (-1, 16), "PBR")
        if np.any(self.pbr[self.ids, 8:11] != 0):
            raise ValueError("moving emissive subjects require a source-buffer update")
        self.sources = np.array(installed_scene[7], copy=True)
        self.triangle_count = len(installed_scene[3])
        self.lock = threading.Lock()
        self.pending = None
        self.latest = (-1, -1)
        self.ack = None
        self.failed = False
        self.owner = threading.get_ident()

    def _camera_matches(self, packet):
        # The capture carries the displayed column-major Studio view, not a
        # separately chosen photographic camera. Prove it matches this reader.
        camera = packet["camera"]
        matrix = _f32(camera["view"], (4, 4), "view").T.astype(np.float64)
        if not np.array_equal(matrix[3], [0, 0, 0, 1]):
            raise ValueError("dynamic camera needs an affine displayed view")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=4e-6, rtol=0):
            raise ValueError("dynamic view is not a rigid camera")
        unit = packet["nativeFrame"]["unitM"]
        if not isinstance(unit, (int, float)) or isinstance(unit, bool) or not np.isfinite(unit) or unit <= 0:
            raise ValueError("dynamic camera needs the shared metre scale")
        origin = _f32(packet["rayOriginM"], (3,), "ray origin").astype(np.float64)
        eye = (-rotation.T @ matrix[:3, 3]) * unit + origin
        from .camera_frame import sensor_camera_basis
        actual = self.config["camera"]
        reader_eye, right, up, forward = sensor_camera_basis(actual["eye_m"], actual["target_m"])
        # Compare float32-composed camera matrices; never move/reframe either.
        tolerance = 32 * np.finfo(np.float32).eps
        if not np.allclose(eye, reader_eye, rtol=tolerance, atol=tolerance * unit) or not all(
                np.allclose(a, b, rtol=0, atol=tolerance)
                for a, b in ((rotation[0], right), (rotation[1], up), (-rotation[2], forward))):
            raise ValueError("current Studio camera differs from initialized ray reader")
        if camera["fovDegrees"] != actual["fov_degrees"] or camera["aspect"] != self.config["live"]["aspect"]:
            raise ValueError("current Studio projection differs from initialized ray reader")

    def stage(self, packet):
        # Immutable bytes sever every producer reference before publication.
        raw = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > 64 * 1024 * 1024:
            raise ValueError("dynamic scene packet exceeds 64 MiB")
        value = json.loads(raw)
        if value.get("schema") != "turing.studio-dynamic-geometry.v1":
            raise ValueError("unsupported dynamic scene packet")
        revision, step = value.get("revision"), value.get("step")
        if any(type(v) is not int or not 0 <= v <= 2**53 - 1 for v in (revision, step)):
            raise ValueError("dynamic revision/step must be exact nonnegative integers")
        seconds = value.get("seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds < 0:
            raise ValueError("dynamic scene needs finite native seconds")
        self._camera_matches(value)
        static = value["staticRay"]
        fixtures = self.config["fixtures"]
        for key in ("vertices", "material_ids", "pbr"):
            if static[key] != fixtures[key]:
                raise ValueError("room/light/material change needs a new initialized ray candidate")
        if "normals" in static:
            _f32(static["normals"], (-1,3,3), "static authored normals")
            if len(static["normals"]) != len(static["vertices"]):
                raise ValueError("static authored normals must match every vertex")
        if "normals" in fixtures and static.get("normals") != fixtures["normals"]:
            raise ValueError("static authored normals differ from initialized scene receipt")
        if not _equal(_f32(static["pbr"], (-1,16), "PBR"), self.pbr):
            raise ValueError("subject materials differ from installed ray materials")
        ray = value["subject"]["ray"]
        vertices = _f32(ray["vertices"], (-1,3,3), "vertices")
        normals = _f32(ray["normals"], vertices.shape, "authored normals")
        ids = ray["material_ids"]
        if any(type(v) is not int for v in ids) or not _equal(ids, self.ids) or len(vertices) != len(self.ids):
            raise ValueError("subject topology/material identity changed")
        if np.any(np.linalg.norm(normals, axis=2) == 0):
            raise ValueError("authored subject normals must remain nonzero")
        # Match the real packer's identity-scale admission threshold. Do not
        # silently drop a pole/closure triangle to obtain a smaller scene.
        areas = np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0], vertices[:,2]-vertices[:,0]), axis=1)
        if np.any(areas <= 1e-12):
            raise ValueError("canonical ray packer would drop a dynamic triangle")
        with self.lock:
            if self.failed:
                raise RuntimeError("dynamic reader failed; initialize a new candidate")
            if (revision, step) <= self.latest:
                raise ValueError("stale or repeated dynamic revision/step")
            self.latest = (revision, step)
            self.pending = (raw, vertices, normals)
        return {"staged": True, "revision": revision, "step": step}

    def commit_boundary(self, sensor):
        if threading.get_ident() != self.owner:
            raise RuntimeError("only the owning GL thread may install a scene")
        if sensor._row_off != 0:
            raise RuntimeError("dynamic scene installation requires the existing sweep boundary")
        with self.lock:
            if self.failed:
                raise RuntimeError("dynamic reader failed; no further rays may be issued")
            pending, self.pending = self.pending, None
        if pending is None:
            return None
        raw, vertices, _normals = pending
        value = json.loads(raw)
        config = {**self.config, "scale_m": 1.0, "object_center_m": [0,0,0]}
        try:
            record=self.field.get('directional_record')
            refreshed_roi=None
            if record and record.get('recording_region'):
                from camera_designer.studio_directional_field import subject_field_region
                refreshed_roi=subject_field_region(vertices,config)
                if refreshed_roi['dimensions'][0]!=record['grid_size']:raise ValueError('subject ROI grid changed; new candidate required')
            lineage=[] if self.field.get("directional_record") is not None else None
            scene, spheres, stats, _ = prepare_photo_geometry(config, vertices, self.ids, self.pbr, **({"material_lineage_out": lineage} if lineage is not None else {}))
            geom, shade, mat, triangles, contexts, _, bounds, sources = scene
            if len(triangles) != self.triangle_count or len(spheres):
                raise ValueError("canonical packing changed full triangle/medium ownership")
            if not _equal(sources, self.sources):
                raise ValueError("dynamic scene changed emissive sources; new candidate required")
            nodes, ids = self.renderer._build_gpu_bvh(triangles)
            margin = np.asarray([.02,.02,.04],np.float32)
            bounds = (bounds[0]-margin,bounds[1]+margin)
            # Existing upload waits glFinish before mutating its shared SSBOs.
            # Both readers retain these handles. No ray is dispatched here.
            from material_db import MaterialDatabase
            if refreshed_roi is not None:record['recording_region']=refreshed_roi
            self.renderer._update_prebuilt_geometry(self.field,geom,shade,mat,nodes,ids,contexts,bounds,spheres,
                native_pbr=MaterialDatabase.instance().build_tensors()["pbr"])
            if lineage is not None:self.field["directional_record"]["material_lineage"]=lineage
            sensor.set_scene_geometry(n_tris=len(triangles),n_bvh_nodes=len(nodes),
                bmin=bounds[0],bmax=bounds[1],n_spheres=0)
            sensor._geometry_stats = stats
            seed = self.field["_pump_seed"]
            self.renderer._clear_prebuilt(self.field)
            self.field["_pump_seed"] = seed
            sensor.clear()
            from OpenGL.GL import glFinish
            glFinish()
            ack = {"revision":value["revision"],"step":value["step"],"seconds":value["seconds"],
                "packet_sha256":hashlib.sha256(raw).hexdigest(),"subject_triangles":len(vertices),
                "reader_triangles":len(triangles),"normal_mode":"geometric-face",
                "authored_normals_retained":True,"authored_vertex_normals_consumed":False,
                "scope":"nonemissive-subject-only",
                "both_readers_installed":True,"histories_cleared":True}
            self.active_packet = raw
            self.ack = ack
            return dict(ack)
        except Exception:
            with self.lock:
                self.failed = True
                self.pending = None
            # Let the incumbent worker's finally release the reader. Never
            # continue with partially replaced geometry or falsely acknowledge.
            raise
