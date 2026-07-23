"""Physical QR witness used to validate the optical preview path.

The target is deliberately represented as material geometry, never as a UI
overlay or an image copied into the result.  A camera must therefore preserve
orientation, contrast, coverage and enough focus for the code to remain useful.

``SPECTRAL-BENCH-V1`` is a version-2, error-correction-H QR symbol.  Keeping the
small canonical matrix here avoids making the camera station depend on a QR
package at runtime.  The matrix was generated without its four-module quiet
zone; :func:`triangulate_qr_target` authors that quiet zone as white material.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


QR_PAYLOAD = "SPECTRAL-BENCH-V1"
QR_MATRIX_ROWS = (
    "1111111011111111101111111",
    "1000001001001110101000001",
    "1011101001101000101011101",
    "1011101011111101001011101",
    "1011101001000011101011101",
    "1000001001111100001000001",
    "1111111010101010101111111",
    "0000000001100010100000000",
    "0010111010111001010001001",
    "0011010011011001101110010",
    "0111001010101011011011101",
    "0111000010101101101110000",
    "1111101101100011000000111",
    "0010010110100111011110100",
    "1010111011010000000111111",
    "0101100011110111011101010",
    "1011111010101100111110101",
    "0000000011000011100010100",
    "1111111000100000101010010",
    "1000001011010000100011110",
    "1011101011000101111111000",
    "1011101001111011110000110",
    "1011101011100011001111001",
    "1000001001001111010100011",
    "1111111001000100100010101",
)


def qr_target_spec(*, z: float = 0.30, width: float = 0.15) -> dict:
    """Return the canonical scene description for the optical QR witness."""

    return {
        "type": "qr_target",
        "payload": QR_PAYLOAD,
        "pos": [0.0, 0.0, float(z)],
        "width": float(width),
        "quiet_zone": 4,
    }


def _module_triangles(x0: float, y0: float, x1: float, y1: float, z: float):
    # Viewed by the camera from -Z, rows read top-to-bottom (+Y to -Y).
    p00 = np.array([x0, y0, z], np.float64)
    p10 = np.array([x1, y0, z], np.float64)
    p11 = np.array([x1, y1, z], np.float64)
    p01 = np.array([x0, y1, z], np.float64)
    # Winding gives -Z normals, toward the camera.
    return (
        np.concatenate((p00, p11, p10)),
        np.concatenate((p00, p01, p11)),
    )


def triangulate_qr_target(scene_object: dict) -> list[tuple[bool, np.ndarray, np.ndarray]]:
    """Return contiguous white and black triangle groups for a QR target.

    Each tuple is ``(is_black, vertices[N,9], normals[N,3])``.  The backing and
    quiet zone are one white quad; only black modules need individual geometry.
    This is both tighter and avoids coplanar white/black module overlaps.
    """

    rows: Iterable[str] = QR_MATRIX_ROWS
    matrix = tuple(rows)
    if len(matrix) != 25 or any(len(row) != 25 for row in matrix):
        raise RuntimeError("canonical optical QR matrix is not 25x25")
    pos = np.asarray(scene_object.get("pos", (0.0, 0.0, 0.30)), np.float64)
    width = float(scene_object.get("width", 0.15))
    quiet = int(scene_object.get("quiet_zone", 4))
    if width <= 0.0 or quiet < 4:
        raise ValueError("QR target needs positive width and a >=4 module quiet zone")

    total_modules = len(matrix) + quiet * 2
    module = width / total_modules
    left = float(pos[0]) - width * 0.5
    top = float(pos[1]) + width * 0.5
    z = float(pos[2])

    white_verts = list(_module_triangles(left, top, left + width, top - width, z))
    black_verts: list[np.ndarray] = []
    for row_i, row in enumerate(matrix):
        for col_i, value in enumerate(row):
            if value != "1":
                continue
            x0 = left + (quiet + col_i) * module
            x1 = x0 + module
            y0 = top - (quiet + row_i) * module
            y1 = y0 - module
            # Offset black ink toward the camera by one micron: it remains a
            # physical coating while avoiding coplanar hit ambiguity.
            black_verts.extend(_module_triangles(x0, y0, x1, y1, z - 1.0e-6))

    normal = np.array([0.0, 0.0, -1.0], np.float64)
    white = np.asarray(white_verts, np.float64).reshape(-1, 9)
    black = np.asarray(black_verts, np.float64).reshape(-1, 9)
    return [
        (False, white, np.tile(normal, (len(white), 1))),
        (True, black, np.tile(normal, (len(black), 1))),
    ]


__all__ = [
    "QR_PAYLOAD", "QR_MATRIX_ROWS", "qr_target_spec", "triangulate_qr_target",
]
