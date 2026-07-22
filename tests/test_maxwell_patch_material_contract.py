from __future__ import annotations

import numpy as np
import pytest

from material_db import MaterialDatabase
from spectral_material import Material


def _declaration() -> dict:
    return {
        "schema": 1,
        "asset": "materials/structural/morpho_cell.usda",
        "prim": "/Cell",
        "parameters": {"ridge_pitch_m": 6.4e-7},
        "requested_product": "polarized_scattering",
        "artifact": None,
        "fallback": "bulk",
    }


def test_maxwell_patch_round_trips_as_cold_authoring_metadata() -> None:
    declaration = _declaration()
    material = Material.from_dict(
        {"domain": "em_optical", "maxwell_patch": declaration},
        name="structural_blue",
    )

    assert material.maxwell_patch == declaration
    assert material.to_dict()["maxwell_patch"] == declaration
    assert material.maxwell_patch is not declaration


def test_maxwell_patch_does_not_change_hot_material_tensor_layouts() -> None:
    db = MaterialDatabase()
    db.register(
        "structural_blue",
        Material.from_dict(
            {"domain": "em_optical", "maxwell_patch": _declaration()},
            name="structural_blue",
        ),
    )

    tensors = db.build_tensors()
    assert tensors["pbr"].shape == (1, 16)
    assert tensors["spectral"].ndim == 3
    assert tensors["enamel"].shape == (1, 8)
    assert "maxwell_patch" not in tensors
    assert all(array.dtype in (np.dtype("float32"), np.dtype("int32"))
               for name, array in tensors.items()
               if name != "index")


def test_maxwell_patch_rejects_non_mapping_authoring_data() -> None:
    with pytest.raises(ValueError, match="maxwell_patch must be a mapping"):
        Material.from_dict({"maxwell_patch": ["not", "a", "mapping"]})

