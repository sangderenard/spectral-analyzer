"""Wavefront OBJ/MTL intake for the camera scene pipeline.

Converts an industry-standard OBJ mesh (optionally with a companion MTL
material library) into exactly the shapes
``append_saved_studio_mesh_to_gpu_scene`` already accepts for a saved Studio
marble: a face-ordered ``(N, 3, 3)`` triangle array in Studio radius units,
a per-face ``(N,)`` material-id array, and an ``(M, 16)`` PBR16 table using
the same column layout as ``examples/meowdoku/material_catalog.py``:
``albedo[0:3] roughness[3] metallic[4] transmission[5] ior[6] opacity[7]
emission[8:11]``.

This is intentionally a new *object source*, not a new rendering path: once
parsed, an OBJ import is indistinguishable from a hand-designed Studio
marble to every downstream consumer (scene builder, GLSL forward/sensor
shaders, marble_preview_worker.py's timed photo contract).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

_V_RE = re.compile(r"^v\s+(\S+)\s+(\S+)\s+(\S+)")
_F_TOKEN_RE = re.compile(r"^(-?\d+)(?:/(-?\d*)(?:/(-?\d*))?)?$")

# PBR16 column layout, matching examples/meowdoku/material_catalog.py and
# camera_designer/scene_builder.py's reading of a saved object's table.
_DEFAULT_PBR_ROW = (0.70, 0.70, 0.70,  # albedo (neutral mid-gray)
                    0.55, 0.0,         # roughness, metallic
                    0.0, 1.5, 1.0,     # transmission, ior, opacity (opaque)
                    0.0, 0.0, 0.0,     # emission
                    0.0, 0.0, 0.0, 0.0, 0.0)  # reserved


class ObjImportError(ValueError):
    """Raised for malformed OBJ/MTL input, mirroring marble_photo.PhotoError
    in spirit (reject clearly, never guess past a point of ambiguity)."""


def _parse_mtl(text: str) -> dict[str, list[float]]:
    """Return {material_name: PBR16 row} for every ``newmtl`` block found."""
    materials: dict[str, list[float]] = {}
    name: str | None = None
    row: list[float] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].lower()
        if key == "newmtl":
            if name is not None:
                materials[name] = row  # type: ignore[assignment]
            name = " ".join(parts[1:]) or f"material_{len(materials)}"
            row = list(_DEFAULT_PBR_ROW)
            continue
        if row is None:
            continue  # stray directive before the first newmtl
        try:
            if key == "kd" and len(parts) >= 4:
                row[0:3] = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif key == "ke" and len(parts) >= 4:
                row[8:11] = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif key == "ns" and len(parts) >= 2:
                # MTL shininess is 0..1000 (Phong exponent); roughness is its
                # inverse sense on a 0..1 scale. sqrt keeps the common
                # low/mid shininess range from crushing to near-zero roughness.
                shininess = max(0.0, float(parts[1]))
                row[3] = float(np.clip((1000.0 - shininess) / 1000.0, 0.02, 1.0) ** 0.5)
            elif key in ("ni",) and len(parts) >= 2:
                row[6] = max(1.0, float(parts[1]))
            elif key in ("d",) and len(parts) >= 2:
                row[7] = float(np.clip(float(parts[1]), 0.0, 1.0))
            elif key == "tr" and len(parts) >= 2:
                row[7] = float(np.clip(1.0 - float(parts[1]), 0.0, 1.0))
            elif key == "illum" and len(parts) >= 2:
                # illum 7 is "refraction, Fresnel" in the MTL spec -- the
                # closest standard signal that a material should transmit.
                if int(float(parts[1])) == 7:
                    row[5] = max(row[5], 0.9)
        except ValueError:
            continue  # malformed numeric token; keep the default for that field
    if name is not None:
        materials[name] = row  # type: ignore[assignment]
    return materials


def _resolve_face_index(token: str, count: int) -> int:
    match = _F_TOKEN_RE.match(token)
    if not match:
        raise ObjImportError(f"unrecognized OBJ face token {token!r}")
    raw = int(match.group(1))
    # OBJ indices are 1-based; negative indices are relative to the current
    # end of that attribute's list (the OBJ spec's own convention).
    index = raw - 1 if raw > 0 else count + raw
    if not 0 <= index < count:
        raise ObjImportError(f"OBJ face index {raw} is out of range for {count} vertices")
    return index


def load_obj(obj_source: str | bytes | Path, mtl_source: str | bytes | Path | None = None,
            *, default_material: str = "obj_import_default") -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Parse OBJ (+ optional MTL) text into Studio-compatible mesh arrays.

    Returns ``(vertices, material_ids, pbr, material_names)`` where
    ``vertices`` is ``(N, 3, 3)`` float32 in Studio radius units (recentered
    on the mesh's bounding-box midpoint and rescaled so its circumscribing
    radius is 1.0 -- the same convention a hand-designed Studio marble uses
    before ``scale_m``/``object_center_m`` places it in the real scene),
    ``material_ids`` is ``(N,)`` int32, ``pbr`` is ``(M, 16)`` float32, and
    ``material_names`` gives the row order for diagnostics.
    """
    def _text(source) -> str | None:
        if source is None:
            return None
        if isinstance(source, Path):
            return source.read_text(encoding="utf-8", errors="replace")
        if isinstance(source, bytes):
            return source.decode("utf-8", errors="replace")
        return source

    obj_text = _text(obj_source)
    if obj_text is None:
        raise ObjImportError("no OBJ source provided")
    mtl_text = _text(mtl_source)

    materials: dict[str, list[float]] = {}
    if mtl_text is not None:
        materials = _parse_mtl(mtl_text)

    positions: list[tuple[float, float, float]] = []
    faces: list[tuple[list[int], str]] = []  # (vertex indices, material name)
    current_material = default_material
    unresolved_mtllib: list[str] = []

    for raw_line in obj_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line[0] == "v" and line[1:2] in (" ", "\t"):
            m = _V_RE.match(line)
            if not m:
                raise ObjImportError(f"malformed vertex line: {raw_line!r}")
            positions.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            continue
        parts = line.split()
        key = parts[0].lower()
        if key == "f":
            if len(parts) < 4:
                raise ObjImportError(f"face needs at least 3 vertices: {raw_line!r}")
            idx = [_resolve_face_index(tok.split("/")[0], len(positions)) for tok in parts[1:]]
            # Fan-triangulate polygons (quads and beyond), the conventional
            # OBJ-consumer behavior for convex/planar faces.
            for i in range(1, len(idx) - 1):
                faces.append(([idx[0], idx[i], idx[i + 1]], current_material))
        elif key == "usemtl":
            current_material = " ".join(parts[1:]) or default_material
        elif key == "mtllib":
            unresolved_mtllib.extend(parts[1:])
        # vt/vn/o/g/s and anything else: no effect on this engine's geometric
        # face normals or (currently unsupported) UV-mapped albedo textures.

    if not positions:
        raise ObjImportError("OBJ contains no vertices")
    if not faces:
        raise ObjImportError("OBJ contains no faces")
    if mtl_text is None and unresolved_mtllib:
        raise ObjImportError(
            f"OBJ references mtllib {unresolved_mtllib!r} but no MTL source was supplied; "
            "pass its contents as mtl_source or drop the mtllib line"
        )

    verts = np.asarray(positions, np.float64)
    center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    recentered = verts - center
    radius = float(np.max(np.linalg.norm(recentered, axis=1)))
    if radius < 1e-12:
        raise ObjImportError("OBJ mesh has zero extent")
    recentered /= radius

    used_names = sorted({name for _, name in faces})
    if not materials:
        for name in used_names:
            materials[name] = list(_DEFAULT_PBR_ROW)
    missing = [name for name in used_names if name not in materials]
    if missing:
        raise ObjImportError(
            f"OBJ uses material(s) {missing!r} with no matching MTL newmtl block"
        )
    name_to_id = {name: i for i, name in enumerate(used_names)}
    pbr = np.asarray([materials[name] for name in used_names], np.float32).reshape(-1, 16)

    tri_vertices = np.empty((len(faces), 3, 3), np.float32)
    material_ids = np.empty(len(faces), np.int32)
    for i, (idx, name) in enumerate(faces):
        tri_vertices[i] = recentered[idx]
        material_ids[i] = name_to_id[name]

    return tri_vertices, material_ids, pbr, tuple(used_names)
