from pathlib import Path
import re

import numpy as np
import pytest

from camera_software.complex_optical_operators import (
    WAVE_EXIT_STATE_DTYPE,
    parse_wave_exit_states,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_gpu_complex_schema_is_versioned_and_exact_width():
    cpu = (ROOT / "csrc/include/complex_transport.h").read_text(encoding="utf-8")
    gpu = (ROOT / "csrc/shaders/complex_transport.glsl.inc").read_text(encoding="utf-8")

    assert "kSchemaVersion = 2u" in cpu
    assert "kLaneCounts = {1, 3, 4, 8, 16, 32}" in cpu
    assert "sizeof(PackedComplexLaneGpu) == 64" in cpu
    assert "COMPLEX_SCHEMA_VERSION       2u" in gpu
    assert "COMPLEX_LANE_WORDS          16u" in gpu
    assert "COMPLEX_LANE_BYTES          64u" in gpu
    assert "basis_id" in cpu and "operator_id" in cpu
    assert "COMPLEX_BASIS_ID" in gpu and "COMPLEX_OPERATOR_ID" in gpu


def test_hot_schema_does_not_declare_an_ssbo_binding():
    gpu = (ROOT / "csrc/shaders/complex_transport.glsl.inc").read_text(encoding="utf-8")
    assert "layout(std430" not in gpu
    assert "binding =" not in gpu


def test_gpu_t1_wave_routing_stays_within_existing_eight_ssbo_channels():
    shader = (
        ROOT / "csrc/shaders/ray_bvh_intersect.comp.glsl"
    ).read_text(encoding="utf-8")
    bindings = {
        int(value) for value in re.findall(r"binding\s*=\s*(\d+)", shader)
    }

    assert bindings == set(range(8))
    assert "WaveIntent tail" in shader
    assert "8 × n_arenas" in shader
    assert "WaveArenaBuf" not in shader


def test_wave_exit_side_record_is_fixed_stride_and_does_not_widen_ray_intent():
    header = (
        ROOT / "csrc/include/wave_exit_state.h"
    ).read_text(encoding="utf-8")
    ray_header = (
        ROOT / "csrc/include/ray_pipeline.h"
    ).read_text(encoding="utf-8")

    assert WAVE_EXIT_STATE_DTYPE.itemsize == 160
    assert "sizeof(WaveExitStateRecord) == 160" in header
    ray_intent_body = ray_header.split(
        "struct RayIntent {", 1
    )[1].split("};", 1)[0]
    assert "WaveExitStateRecord" not in ray_intent_body
    assert "wave_exit" not in ray_intent_body.lower()

    raw = np.zeros((2, 160), np.uint8)
    records = parse_wave_exit_states(raw)
    assert records.shape == (2,)
    assert not records.flags.writeable
    with pytest.raises(ValueError):
        parse_wave_exit_states(np.zeros((1, 159), np.uint8))
