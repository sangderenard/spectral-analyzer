"""JSON-safe AbstractUI view of Pluck's authoritative material database.

This is a cold compiler bridge.  It loads material authoring YAML, registers
each material with :mod:`material_db`, and exports the same packed rows used by
the native renderer and ``base_material.frag.glsl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from material_db import MaterialDatabase


MATERIAL_DIRECTORY = Path(__file__).resolve().parent / "configs" / "materials"
MATTE_MINIMUM_ROUGHNESS = 0.72
MATTE_MAXIMUM_SPECULAR = 0.18
MATTE_MAXIMUM_SHININESS = 24.0


def _material_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize supported YAML spellings without deriving packed records."""

    pbr = payload.get("pbr") if isinstance(payload.get("pbr"), dict) else {}
    result = dict(payload)
    albedo = pbr.get(
        "albedo", pbr.get("albedo_rgb", payload.get("albedo_rgb", [0.5, 0.5, 0.5]))
    )
    result["albedo_rgb"] = [float(value) for value in albedo[:3]]
    for key in ("roughness", "metallic", "transmission", "ior", "opacity"):
        if key in pbr:
            result[key] = float(pbr[key])
    if "roughness" not in result and "smoothness" in result:
        result["roughness"] = max(0.0, min(1.0, 1.0 - float(result["smoothness"])))
    emission = pbr.get("emission_rgb", payload.get("emission_rgb", [0.0, 0.0, 0.0]))
    result["emission_rgb"] = [float(value) for value in emission[:3]]
    return result


def build_material_catalog(material_directory: Path = MATERIAL_DIRECTORY) -> dict[str, Any]:
    """Return named materials plus the exact four chunks consumed by Phong."""

    database = MaterialDatabase()
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(material_directory.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or path.stem)
        database.register(name, _material_payload(payload))
        metadata[name] = {
            "description": str(payload.get("description") or ""),
            "source": path.relative_to(material_directory.parent.parent).as_posix(),
        }
        medium = database.medium_profile(name)
        if medium is not None:
            metadata[name]['medium'] = medium
            metadata[name]['surface_role'] = 'editor-placeholder-only'

    tensors = database.build_tensors()
    records = []
    for index, name in enumerate(database.names()):
        pbr = tensors["pbr"][index].tolist()
        phong = tensors["phong_compat"][index].tolist()
        enamel = tensors["enamel"][index].tolist()
        # The living-world base catalogue is intentionally matte.  Preserve
        # authored colour/optical identity while preventing every wall and
        # floor from behaving like polished display plastic.
        pbr[3] = max(float(pbr[3]), MATTE_MINIMUM_ROUGHNESS)
        pbr[4] = min(float(pbr[4]), 0.08)
        pbr[5] = 0.0
        pbr[8:11] = [0.0, 0.0, 0.0]
        phong[0] = max(float(phong[0]), 0.06)
        phong[1] = min(float(phong[1]), MATTE_MAXIMUM_SPECULAR)
        phong[2] = min(float(phong[2]), MATTE_MAXIMUM_SHININESS)
        enamel[0] = 0.0
        enamel[3] = max(float(enamel[3]), MATTE_MINIMUM_ROUGHNESS)
        records.append({
            "identity": name,
            "label": name.replace("_", " "),
            "index": index,
            **metadata[name],
            "pbr": pbr,
            "phong": phong,
            "enamel": enamel,
            "texture_stack": tensors["texture_stack"][index].tolist(),
        })
    return {
        "schema": "abstract-ui-material-catalog-v0",
        "identity": "pluck-material-database",
        "label": "Pluck material database",
        "source": str(material_directory),
        "record_count": len(records),
        "chunk_strides": {"pbr": 16, "phong": 8, "enamel": 8, "texture_stack": 16},
        "surface_policy": {
            "profile": "matte-world-base",
            "minimum_roughness": MATTE_MINIMUM_ROUGHNESS,
            "maximum_specular_strength": MATTE_MAXIMUM_SPECULAR,
            "maximum_shininess": MATTE_MAXIMUM_SHININESS,
            "minimum_indirect_ambient": 0.06,
            "maximum_metallic": 0.08,
            "transmission": 0.0,
            "emission": [0.0, 0.0, 0.0],
            "enamel_thickness_nm": 0.0,
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-directory", type=Path, default=MATERIAL_DIRECTORY)
    arguments = parser.parse_args()
    print(json.dumps(build_material_catalog(arguments.material_directory), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
