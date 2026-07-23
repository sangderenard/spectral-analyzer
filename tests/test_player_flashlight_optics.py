from __future__ import annotations

import numpy as np

from camera_designer.camera_preset import ProjectorBackSpec
from player_flashlight import PlayerFlashlight


def test_flashlight_uses_shared_physical_emissive_back_contract():
    light = PlayerFlashlight()
    light.enable()

    spec = light.optical_emitter_spec([0.45, 0.55, 0.65])
    contract = light.optical_source_contract([0.45, 0.55, 0.65])

    assert spec.enabled is True
    assert spec.profile_name == ""
    assert np.allclose(spec.pos, light.bulb_local_pos)
    assert contract["schema"] == "physical-luminaire-source-v1"
    assert contract["emissive_back"]["schema"] == (
        "physical-emissive-back-v1"
    )
    assert contract["emissive_back"]["source_role"] == "flashlight"
    assert contract["emissive_back"]["asset_key"] == (
        "flashlight.default.tungsten"
    )


def test_flashlight_accepts_a_home_baked_complex_source():
    field = np.zeros((2, 2, 2), np.complex128)
    field[..., 0] = 1.0
    field[1, :, 1] = 0.5j
    baked = ProjectorBackSpec.from_jones_field(
        field,
        asset_key="flashlight.baked.prototype",
        source_role="flashlight",
    )
    light = PlayerFlashlight(emissive_back=baked)
    light.enable()

    contract = light.optical_source_contract([0.532])

    assert contract["enabled"] is True
    assert contract["emissive_back"]["enabled"] is True
    assert contract["emissive_back"]["texture_shape"] == [2, 2, 10]
    assert contract["emissive_back"]["asset_key"] == (
        "flashlight.baked.prototype"
    )
