from __future__ import annotations

import numpy as np

from resonator_core import (
    ResonatorString,
    StringCouplingConfig,
    build_string_coupling_matrix,
    solve_coupled_strings,
)


def test_coupling_matrix_prefers_harmonic_and_spatial_proximity() -> None:
    strings = [
        ResonatorString(key="a", fundamental_hz=110.0, x=0.0, y=0.0),
        ResonatorString(key="b", fundamental_hz=220.0, x=0.2, y=0.0),
        ResonatorString(key="c", fundamental_hz=311.0, x=4.0, y=0.0),
    ]
    cfg = StringCouplingConfig(
        harmonic_sigma=0.1,
        distance_sigma=1.0,
        harmonic_strength=0.8,
        distance_strength=0.2,
        base_strength=0.2,
    )

    coupling = build_string_coupling_matrix(strings, cfg)

    assert coupling.shape == (3, 3)
    assert np.allclose(np.diag(coupling), 0.0)
    assert abs(coupling[0, 1]) > abs(coupling[0, 2])
    assert abs(coupling[1, 0]) > abs(coupling[2, 0])


def test_coupled_string_solver_spreads_energy_to_neighboring_strings() -> None:
    strings = [
        ResonatorString(key="driver", fundamental_hz=110.0, decay_s=0.25, x=0.0, y=0.0),
        ResonatorString(key="near", fundamental_hz=220.0, decay_s=0.25, x=0.15, y=0.0),
        ResonatorString(key="far", fundamental_hz=330.0, decay_s=0.25, x=3.5, y=0.0),
    ]
    drive = np.zeros((3, 128), dtype=np.complex128)
    drive[0, 0] = 1.0 + 0.0j

    result = solve_coupled_strings(
        drive,
        strings,
        sample_rate=48_000.0,
        coupling_config=StringCouplingConfig(base_strength=0.35),
    )

    near_energy = float(np.sum(np.abs(result.output[1])))
    far_energy = float(np.sum(np.abs(result.output[2])))

    assert near_energy > 0.0
    assert far_energy >= 0.0
    assert near_energy > far_energy


def test_resonator_solver_preserves_complex_domain() -> None:
    strings = [
        ResonatorString(key="s1", fundamental_hz=110.0, phase_offset_rad=0.25),
        ResonatorString(key="s2", fundamental_hz=220.0, phase_offset_rad=0.5),
    ]
    t = np.linspace(0.0, 1.0, 64, endpoint=False)
    drive = np.vstack([
        np.exp(1j * 2.0 * np.pi * 2.0 * t),
        np.zeros_like(t, dtype=np.complex128),
    ])

    result = solve_coupled_strings(drive, strings, sample_rate=64.0)

    assert np.iscomplexobj(result.output)
    assert np.iscomplexobj(result.coupling_matrix)
    assert np.max(np.abs(result.output.imag)) > 0.0
