"""Common scene-file coordinator for canonical and advanced transport data.

This module provides a single I/O surface for opening and saving scene files
in a conventional JSON layout that mirrors common DCC/game pipelines
(nodes/meshes/materials/cameras/lights), while preserving advanced data in a
safe metadata namespace that conventional tools can ignore.

Design goals
------------
1. Keep conventional fields easy to consume:
   - perspective camera with vertical FOV and clip ranges
   - RGB base color, metallic, roughness, and emissive factors
   - node transforms, mesh references, and scene roots
2. Preserve full spectral-analyzer depth in namespaced metadata:
   - advanced camera software stacks, manifold/lens data, nonplanar backs
   - emission/remission profile references and spectral payloads
   - custom object classes and tracing-only controls
3. Use metadata carriers that are safe for external tools to skip:
   - top-level and item-level `extras["spectral_analyzer"]`
   - optional `extensions["SPECTRAL_ANALYZER_ray_tracer"]`

This coordinator intentionally uses JSON dictionaries rather than binding to
any one third-party scene SDK so it can be reused across glTF-like, Blender
export JSON, and engine-side interchange pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .unified import CameraSpec, CanonicalScene, FilmSpec, MaterialTable, SourceTable


SCENE_METADATA_NS = "spectral_analyzer"
SCENE_EXTENSION_KEY = "SPECTRAL_ANALYZER_ray_tracer"


@dataclass(slots=True)
class SceneDocument:
    """Structured wrapper around a conventional scene dictionary."""

    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneDecodeResult:
    """Decoded canonical scene and optional camera/film envelopes."""

    scene: CanonicalScene
    camera: CameraSpec | None = None
    cameras: list[CameraSpec] = field(default_factory=list)
    film: FilmSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SceneCoordinator:
    """Coordinator for loading/saving common scene files and advanced metadata."""

    def load_scene(self, path: str | Path) -> SceneDocument:
        """Open a JSON scene file from disk and return a scene document wrapper."""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Scene file must decode to a JSON object.")
        return SceneDocument(payload=payload)

    def save_scene(
        self,
        path: str | Path,
        document: SceneDocument,
        *,
        indent: int = 2,
        sort_keys: bool = True,
    ) -> None:
        """Save a scene document to disk in deterministic JSON form."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(document.payload, handle, indent=indent, sort_keys=sort_keys)
            handle.write("\n")

    def decode(self, document: SceneDocument) -> SceneDecodeResult:
        """Decode a common scene document into canonical scene structures."""
        payload = document.payload
        metadata = self._read_metadata(payload)
        metadata.setdefault("_passthrough_top_level", self._read_top_level_passthrough(payload))

        materials_payload = {
            "materials": [self._decode_material(item) for item in payload.get("materials", [])]
        }
        cameras_payload = self._decode_all_cameras(payload)
        camera_payload = cameras_payload[0] if cameras_payload else None
        meshes = payload.get("meshes", []) if isinstance(payload.get("meshes"), list) else []

        scene = CanonicalScene(
            geometry={
                "asset": payload.get("asset", {}),
                "scenes": payload.get("scenes", []),
                "scene": payload.get("scene"),
                "nodes": [
                    self._decode_node(item, cameras_payload, meshes)
                    for item in payload.get("nodes", [])
                ],
                "meshes": payload.get("meshes", []),
                "lights": payload.get("lights", []),
                "objects": metadata.get("objects", []),
                "cameras": cameras_payload,
            },
            materials=MaterialTable(payload=materials_payload),
            sources=SourceTable(payload=metadata.get("sources", {})),
            media=metadata.get("media", {}),
            scale_contexts=metadata.get("scale_contexts", {}),
        )

        camera = CameraSpec(payload=camera_payload) if camera_payload else None
        cameras = [CameraSpec(payload=item) for item in cameras_payload]
        film_payload = metadata.get("film")
        film = FilmSpec(payload=film_payload) if isinstance(film_payload, dict) else None

        return SceneDecodeResult(
            scene=scene,
            camera=camera,
            cameras=cameras,
            film=film,
            metadata=metadata,
        )

    def encode(
        self,
        scene: CanonicalScene,
        *,
        camera: CameraSpec | None = None,
        film: FilmSpec | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SceneDocument:
        """Encode canonical scene structures into a common scene document."""
        geometry = scene.geometry
        out: dict[str, Any] = {
            "asset": geometry.get("asset") or {"version": "2.0", "generator": "spectral-analyzer"},
            "scene": geometry.get("scene", 0),
            "scenes": geometry.get("scenes") or [{"name": "MainScene", "nodes": []}],
            "nodes": [self._encode_node(item) for item in geometry.get("nodes", [])],
            "meshes": list(geometry.get("meshes", [])),
            "materials": [
                self._encode_material(item)
                for item in scene.materials.payload.get("materials", [])
            ],
        }

        if geometry.get("lights"):
            out["lights"] = list(geometry.get("lights", []))

        cameras: list[dict[str, Any]] = []
        if camera is not None and camera.payload:
            cameras.append(self._encode_camera(camera.payload))
        elif geometry.get("cameras"):
            cameras = [self._encode_camera(item) for item in geometry.get("cameras", [])]
        if cameras:
            out["cameras"] = cameras

        merged_metadata: dict[str, Any] = {}
        merged_metadata.update(metadata or {})
        merged_metadata.setdefault("sources", scene.sources.payload)
        merged_metadata.setdefault("media", scene.media)
        merged_metadata.setdefault("scale_contexts", scene.scale_contexts)
        if film is not None and film.payload:
            merged_metadata.setdefault("film", film.payload)
        if geometry.get("objects"):
            merged_metadata.setdefault("objects", geometry.get("objects"))

        self._write_metadata(out, merged_metadata)
        passthrough = merged_metadata.get("_passthrough_top_level")
        if isinstance(passthrough, dict):
            for key, value in passthrough.items():
                if key not in out:
                    out[key] = value
        return SceneDocument(payload=out)

    def decode_gltf_document(self, payload: dict[str, Any]) -> SceneDecodeResult:
        """Decode a glTF-like JSON object into canonical structures."""
        if not isinstance(payload, dict):
            raise ValueError("glTF payload must be a JSON object.")
        return self.decode(SceneDocument(payload=payload))

    def encode_gltf_document(
        self,
        scene: CanonicalScene,
        *,
        camera: CameraSpec | None = None,
        film: FilmSpec | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Encode canonical structures into a glTF-like JSON object."""
        return self.encode(scene, camera=camera, film=film, metadata=metadata).payload

    def decode_gltf_file(self, path: str | Path) -> SceneDecodeResult:
        """Load and decode a glTF-like JSON file in one call."""
        return self.decode_file(path)

    def encode_gltf_file(
        self,
        path: str | Path,
        scene: CanonicalScene,
        *,
        camera: CameraSpec | None = None,
        film: FilmSpec | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Encode and write a glTF-like JSON file; return the written payload."""
        return self.encode_file(path, scene, camera=camera, film=film, metadata=metadata).payload

    def decode_file(self, path: str | Path) -> SceneDecodeResult:
        """Convenience: load and decode in one call."""
        return self.decode(self.load_scene(path))

    def encode_file(
        self,
        path: str | Path,
        scene: CanonicalScene,
        *,
        camera: CameraSpec | None = None,
        film: FilmSpec | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SceneDocument:
        """Convenience: encode and save in one call."""
        document = self.encode(scene, camera=camera, film=film, metadata=metadata)
        self.save_scene(path, document)
        return document

    @staticmethod
    def _read_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        extras = payload.get("extras")
        if isinstance(extras, dict):
            scoped = extras.get(SCENE_METADATA_NS)
            if isinstance(scoped, dict):
                return dict(scoped)

        ext = payload.get("extensions")
        if isinstance(ext, dict):
            scoped = ext.get(SCENE_EXTENSION_KEY)
            if isinstance(scoped, dict):
                return dict(scoped)

        return {}

    @staticmethod
    def _write_metadata(payload: dict[str, Any], metadata: dict[str, Any]) -> None:
        if not metadata:
            return
        extras = payload.setdefault("extras", {})
        if isinstance(extras, dict):
            extras[SCENE_METADATA_NS] = dict(metadata)

        extensions = payload.setdefault("extensions", {})
        if isinstance(extensions, dict):
            extensions[SCENE_EXTENSION_KEY] = dict(metadata)

    @staticmethod
    def _decode_node(node: Any, cameras: list[dict[str, Any]], meshes: list[Any]) -> dict[str, Any]:
        if not isinstance(node, dict):
            return {}
        out = dict(node)
        extras = node.get("extras")
        if isinstance(extras, dict) and isinstance(extras.get(SCENE_METADATA_NS), dict):
            out["advanced"] = dict(extras[SCENE_METADATA_NS])

        camera_index = node.get("camera")
        if isinstance(camera_index, int) and 0 <= camera_index < len(cameras):
            out["camera_name_resolved"] = cameras[camera_index].get("name")

        mesh_index = node.get("mesh")
        if isinstance(mesh_index, int) and 0 <= mesh_index < len(meshes):
            mesh_record = meshes[mesh_index]
            if isinstance(mesh_record, dict):
                out["mesh_name_resolved"] = mesh_record.get("name")

        return out

    @staticmethod
    def _encode_node(node: Any) -> dict[str, Any]:
        if not isinstance(node, dict):
            return {}
        out = dict(node)
        advanced = out.pop("advanced", None)
        if isinstance(advanced, dict):
            extras = out.setdefault("extras", {})
            if isinstance(extras, dict):
                extras[SCENE_METADATA_NS] = dict(advanced)
        return out

    @staticmethod
    def _decode_material(material: Any) -> dict[str, Any]:
        if not isinstance(material, dict):
            return {}

        out: dict[str, Any] = {
            "name": material.get("name"),
            "base_color_rgb": None,
            "metallic": None,
            "roughness": None,
            "emissive_rgb": None,
            "alpha_mode": material.get("alphaMode", "OPAQUE"),
        }

        pbr = material.get("pbrMetallicRoughness")
        if isinstance(pbr, dict):
            base_color = pbr.get("baseColorFactor")
            if isinstance(base_color, list) and len(base_color) >= 3:
                out["base_color_rgb"] = base_color[:3]
            out["metallic"] = pbr.get("metallicFactor")
            out["roughness"] = pbr.get("roughnessFactor")

        emissive = material.get("emissiveFactor")
        if isinstance(emissive, list) and len(emissive) >= 3:
            out["emissive_rgb"] = emissive[:3]

        extras = material.get("extras")
        if isinstance(extras, dict):
            advanced = extras.get(SCENE_METADATA_NS)
            if isinstance(advanced, dict):
                out["advanced"] = dict(advanced)
            out["passthrough_extras"] = {
                key: value for key, value in extras.items() if key != SCENE_METADATA_NS
            }

        extensions = material.get("extensions")
        if isinstance(extensions, dict):
            out["passthrough_extensions"] = dict(extensions)

        return out

    @staticmethod
    def _encode_material(material: Any) -> dict[str, Any]:
        if not isinstance(material, dict):
            return {}

        out: dict[str, Any] = {
            "name": material.get("name", "Material"),
            "alphaMode": material.get("alpha_mode", "OPAQUE"),
            "pbrMetallicRoughness": {
                "baseColorFactor": [
                    *(material.get("base_color_rgb") or [1.0, 1.0, 1.0]),
                    float(material.get("alpha", 1.0)),
                ],
                "metallicFactor": float(material.get("metallic", 0.0)),
                "roughnessFactor": float(material.get("roughness", 1.0)),
            },
            "emissiveFactor": list(material.get("emissive_rgb") or [0.0, 0.0, 0.0]),
        }

        advanced = material.get("advanced")
        if isinstance(advanced, dict):
            extras = out.setdefault("extras", {})
            if isinstance(extras, dict):
                extras[SCENE_METADATA_NS] = dict(advanced)

        passthrough_extras = material.get("passthrough_extras")
        if isinstance(passthrough_extras, dict):
            extras = out.setdefault("extras", {})
            if isinstance(extras, dict):
                for key, value in passthrough_extras.items():
                    if key not in extras:
                        extras[key] = value

        passthrough_extensions = material.get("passthrough_extensions")
        if isinstance(passthrough_extensions, dict):
            out["extensions"] = dict(passthrough_extensions)

        return out

    @staticmethod
    def _decode_all_cameras(payload: dict[str, Any]) -> list[dict[str, Any]]:
        cameras = payload.get("cameras")
        if not isinstance(cameras, list) or not cameras:
            return []

        decoded_list: list[dict[str, Any]] = []
        for raw in cameras:
            if not isinstance(raw, dict):
                continue

            perspective = raw.get("perspective")
            decoded: dict[str, Any] = {
                "name": raw.get("name", "Camera"),
                "kind": raw.get("type", "perspective"),
            }
            if isinstance(perspective, dict):
                decoded["yfov"] = perspective.get("yfov")
                decoded["znear"] = perspective.get("znear")
                decoded["zfar"] = perspective.get("zfar")
                decoded["aspect_ratio"] = perspective.get("aspectRatio")

            extras = raw.get("extras")
            if isinstance(extras, dict):
                advanced = extras.get(SCENE_METADATA_NS)
                if isinstance(advanced, dict):
                    decoded["advanced"] = dict(advanced)
                decoded["passthrough_extras"] = {
                    key: value for key, value in extras.items() if key != SCENE_METADATA_NS
                }

            extensions = raw.get("extensions")
            if isinstance(extensions, dict):
                decoded["passthrough_extensions"] = dict(extensions)

            decoded_list.append(decoded)

        return decoded_list

    @staticmethod
    def _decode_primary_camera(payload: dict[str, Any]) -> dict[str, Any] | None:
        cameras = SceneCoordinator._decode_all_cameras(payload)
        if not cameras:
            return None
        return cameras[0]

    @staticmethod
    def _encode_camera(camera: dict[str, Any]) -> dict[str, Any]:
        perspective = {
            "yfov": camera.get("yfov", 0.7853981633974483),
            "znear": camera.get("znear", 0.01),
            "zfar": camera.get("zfar", 10000.0),
        }
        if camera.get("aspect_ratio") is not None:
            perspective["aspectRatio"] = camera.get("aspect_ratio")

        out: dict[str, Any] = {
            "name": camera.get("name", "Camera"),
            "type": camera.get("kind", "perspective"),
            "perspective": perspective,
        }

        advanced = camera.get("advanced")
        if isinstance(advanced, dict):
            extras = out.setdefault("extras", {})
            if isinstance(extras, dict):
                extras[SCENE_METADATA_NS] = dict(advanced)

        passthrough_extras = camera.get("passthrough_extras")
        if isinstance(passthrough_extras, dict):
            extras = out.setdefault("extras", {})
            if isinstance(extras, dict):
                for key, value in passthrough_extras.items():
                    if key not in extras:
                        extras[key] = value

        passthrough_extensions = camera.get("passthrough_extensions")
        if isinstance(passthrough_extensions, dict):
            out["extensions"] = dict(passthrough_extensions)

        return out

    @staticmethod
    def _read_top_level_passthrough(payload: dict[str, Any]) -> dict[str, Any]:
        known = {
            "asset",
            "scene",
            "scenes",
            "nodes",
            "meshes",
            "materials",
            "cameras",
            "lights",
            "extras",
            "extensions",
        }
        return {key: value for key, value in payload.items() if key not in known}
