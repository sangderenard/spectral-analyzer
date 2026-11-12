"""Submit the existing ink atlas geometry to an account-owned Studio photo job.

Studio's timed exposure is a model shutter interval. It does not use elapsed
GPU time as the exposure clock. The BDPT scene-order exposure remains separate.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import time
from pathlib import Path
from typing import Mapping
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

from .render_assets import BakeRequest, GLOSSY_RED_INK


DEFAULT_STUDIO_ATLAS_CONTRACT = Path(__file__).resolve().parents[1] / "configs" / "studio_ink_atlas_photo.json"
_PHOTO_PROFILE_FIELDS = frozenset({
    "camera", "transport", "exposure_seconds", "exposure_substeps",
    "exposure_iso", "highlight_clip_percent", "film_response", "gamma",
    "max_bounces", "bounce_mode", "triangle_budget", "sphere_coverage",
    "camera_visible", "object_center_m", "surface_transport", "oversampling",
    "samples_per_output_pixel", "tiled", "fixtures", "exposure_state_frames",
})


def load_studio_atlas_contract(path: str | Path = DEFAULT_STUDIO_ATLAS_CONTRACT) -> dict:
    """Read photo settings from a saved Studio MRP1 contract profile.

    The saved scene owns the fixtures and lighting. The bake request supplies
    the OBJ subject and capture resolution.
    """
    root = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(root, dict) or root.get("schema") != "studio-atlas-photo-profile-v1":
        raise ValueError("expected a Studio atlas photo profile v1")
    photo = root.get("photo")
    if not isinstance(photo, dict) or set(photo) - _PHOTO_PROFILE_FIELDS:
        raise ValueError("Studio atlas profile contains unsupported photo fields")
    camera = photo.get("camera")
    if not isinstance(camera, dict) or "resolution" in camera or not camera:
        raise ValueError("Studio atlas profile needs a camera without resolution")
    if photo.get("transport", "fixed:3") not in ("fixed:1", "fixed:3", "fixed:4", "fixed:8"):
        raise ValueError("unsupported Studio atlas profile transport")
    fixtures = photo.get("fixtures")
    if not isinstance(fixtures, dict) or not all(k in fixtures for k in
        ("vertices", "material_ids", "pbr")):
        raise ValueError("Studio atlas profile needs saved scene fixtures")
    if len(fixtures["vertices"]) != len(fixtures["material_ids"]) * 9 or len(fixtures["pbr"]) % 16:
        raise ValueError("Studio atlas profile has inconsistent fixture rows")
    # JSON round-trip rejects NaN and gives callers their own mutable copy.
    return json.loads(json.dumps(photo, allow_nan=False))


def _studio_points(triangles: np.ndarray, *, center_z: float = 0.18) -> np.ndarray:
    """Map the authored +X optical axis and +Z up into a readable Studio view."""
    tri = np.asarray(triangles, np.float64).reshape(-1, 3, 3)
    mapped = np.stack((-tri[..., 1], tri[..., 2], tri[..., 0] + center_z), axis=-1)
    return np.ascontiguousarray(mapped[:, (0, 2, 1), :], np.float32)


def _pbr(material: dict, *, emission=(0.0, 0.0, 0.0)) -> list[float]:
    row = np.zeros(16, np.float32)
    row[:3] = material["albedo_rgb"]
    row[3] = material["roughness"]
    row[4] = material["metallic"]
    row[6] = material.get("ior", 1.5)
    row[7] = 1.0
    row[8:11] = emission
    return row.tolist()


def _obj_text(triangles: np.ndarray) -> str:
    lines = ["# ink-on-slate atlas backplate"]
    for tri in np.asarray(triangles).reshape(-1, 3):
        lines.append("v " + " ".join(format(float(v), ".9g") for v in tri))
    lines.extend(f"f {i} {i+1} {i+2}" for i in range(1, len(triangles) * 3 + 1, 3))
    return "\n".join(lines) + "\n"


def _packet(asset: dict) -> np.ndarray:
    positions = np.asarray(asset["geometry"]["vertices"], dtype="<f4").reshape(-1, 9)
    tri = positions.astype(np.float64).reshape(-1, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if not np.all(lengths > 0):
        raise ValueError("Studio OBJ source contains a degenerate triangle")
    normals = np.repeat((cross / lengths[:, None])[:, None, :], 3, axis=1).astype("<f4").reshape(-1, 9)
    packed = np.empty(len(positions), dtype=[("positions", "<f4", (9,)),
        ("normals", "<f4", (9,)), ("id", "<u4")])
    packed["positions"], packed["normals"], packed["id"] = positions, normals, 0
    return packed


def _snapshot(asset: dict, packed: np.ndarray) -> dict:
    # Imported-object snapshot schema and its exact mechanics receipt are the
    # existing Studio object contract, not a second asset repository.
    tables = {name: {"stride": stride, "data": data} for name, stride, data in (
        ("pbr", 16, _pbr(GLOSSY_RED_INK)),
        ("phong", 1, [0.0]), ("enamel", 1, [0.0]), ("texstack", 1, [0.0]))}
    return {
        "v": 1, "name": "Ink atlas subject", "generator": "meowdoku-imported-object-v1",
        "design": {"base-shape": "obj", "source_asset": asset["digest"],
                   "material_mode": "studio-shell", "shell": "ink"},
        "materials": {"index": {"ink": 0}, "extra": {}, "tables": tables},
        "physical": {"radius-mm": 100},
        "derived": {"massKg": None, "massExact": False, "validForRoll": False,
                    "centerM": None, "inertiaKgM2": None, "components": [],
                    "principalMoments": [],
                    "mechanicsNote": "Imported mesh mechanics have not been certified"},
        "geometry_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
    }


def build_studio_atlas_upload(request: BakeRequest, asset: dict, *, shutter_seconds: float | None = None,
                              substeps: int | None = None, iso: float | None = None,
                              photo_contract: Mapping | None = None) -> bytes:
    """Use the baker's exact geometry with saved Studio photo settings."""
    profile = dict(load_studio_atlas_contract() if photo_contract is None else photo_contract)
    if set(profile) - _PHOTO_PROFILE_FIELDS or not isinstance(profile.get("camera"), dict) or not isinstance(profile.get("fixtures"), dict):
        raise ValueError("invalid Studio atlas photo contract")
    camera = dict(profile["camera"])
    if "resolution" in camera:
        raise ValueError("the atlas capture owns camera resolution")
    shutter_seconds = float(profile.get("exposure_seconds", .1) if shutter_seconds is None else shutter_seconds)
    substeps = profile.get("exposure_substeps", 1) if substeps is None else substeps
    iso = float(profile.get("exposure_iso", 100.0) if iso is None else iso)
    if not .1 <= shutter_seconds <= 3600 or not math.ceil(shutter_seconds * 10 - 1e-9) <= substeps <= 216000:
        raise ValueError("Studio shutter/substeps are outside the REST contract")
    if not 25 <= iso <= 6400:
        raise ValueError("Studio ISO is outside the REST contract")
    packed = _packet(asset)
    # The uploaded OBJ is the glyph, not a stand-in backplate. A saved Studio
    # scene supplies the slate and all lighting fixtures around that subject.
    fixture = json.loads(json.dumps(profile["fixtures"], allow_nan=False))
    timed_frames = profile.get("exposure_state_frames")
    if timed_frames is None:
        timed_frames = [{"t0": 0.0, "t1": shutter_seconds,
                         "pbr": fixture["pbr"], "emitter_profiles": None}]
    elif (not isinstance(timed_frames, list) or not timed_frames
          or float(timed_frames[0]["t0"]) != 0.0
          or float(timed_frames[-1]["t1"]) != shutter_seconds
          or any(float(a["t1"]) != float(b["t0"])
                 for a, b in zip(timed_frames, timed_frames[1:]))):
        raise ValueError("saved Studio state frames must cover the selected shutter")
    config = {
        **profile,
        "object_snapshot": _snapshot(asset, packed),
        "camera": {**camera, "resolution": request.capture.width},
        "exposure_seconds": shutter_seconds,
        "exposure_substeps": substeps, "exposure_iso": iso,
        "fixtures": fixture,
        "exposure_state_frames": timed_frames,
    }
    data = json.dumps(config, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Studio fixture config exceeds the REST limit")
    count = len(packed)
    return (b"MRP1" + struct.pack("<III", len(data), count, 0) + data
            + packed["positions"].tobytes() + packed["normals"].tobytes()
            + packed["id"].tobytes())


class StudioAtlasRestClient:
    """Authenticated client for the existing account-scoped Studio REST paths."""

    def __init__(self, base_url: str, cookie: str, *, timeout: float = 120):
        if not base_url.startswith(("http://", "https://")) or not cookie:
            raise ValueError("Studio URL and authenticated account cookie are required")
        self.base_url, self.cookie, self.timeout = base_url.rstrip("/"), cookie, timeout
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, request, fp, code, msg, headers, newurl):
                raise RuntimeError("Studio REST redirect refused for account cookie")
        self._opener = build_opener(NoRedirect)

    def request(self, path: str, *, body: bytes | None = None, content_type: str = "application/json"):
        request = Request(self.base_url + path, data=body,
                          headers={"Cookie": self.cookie, "Content-Type": content_type})
        with self._opener.open(request, timeout=self.timeout) as response:
            return response.read()

    def json(self, path: str, *, value: dict | None = None) -> dict:
        body = None if value is None else json.dumps(value).encode("utf-8")
        return json.loads(self.request(path, body=body))

    def upload_subject(self, subject: np.ndarray) -> dict:
        # Studio imports OBJ in radius units, then positions it by object_center_m.
        obj = _obj_text((subject - np.asarray((0, 0, .18), np.float32)) / .1)
        receipt = self.json("/api/obj_assets", value={"name": "Ink atlas subject", "obj": obj})
        return self.json("/api/obj_assets/" + receipt["digest"])

    def submit(self, payload: bytes) -> dict:
        return json.loads(self.request("/api/marble_photo", body=payload,
                                       content_type="application/octet-stream"))

    def await_complete(self, digest: str, *, deadline_seconds: float = 600) -> dict:
        deadline = time.monotonic() + deadline_seconds
        while True:
            status = self.json("/api/marble_photo?" + urlencode({"digest": digest}))
            if status.get("state") == "complete":
                return status
            if status.get("state") in ("error", "canceled"):
                raise RuntimeError(status.get("error") or "Studio photo did not complete")
            if time.monotonic() >= deadline:
                raise TimeoutError("Studio photo is still running; job " + digest)
            time.sleep(1)

    def artifact(self, digest: str, name: str, *, hdr: bool = False) -> bytes:
        path = "/api/marble_photo/hdr" if hdr else "/api/marble_photo/image"
        return self.request(path + "?" + urlencode({"digest": digest, "name": name}))


def studio_linear_image(hdr: bytes) -> np.ndarray:
    with np.load(io.BytesIO(hdr), allow_pickle=False) as data:
        linear = np.asarray(data["linear_srgb_exposure"], np.float32)
    if linear.ndim != 3 or linear.shape[2] != 3 or not np.isfinite(linear).all():
        raise ValueError("Studio HDR has no finite linear sRGB frame")
    return linear
