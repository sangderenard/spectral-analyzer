from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_gpu_complex_schema_is_versioned_and_exact_width():
    cpu = (ROOT / "csrc/include/complex_transport.h").read_text(encoding="utf-8")
    gpu = (ROOT / "csrc/shaders/complex_transport.glsl.inc").read_text(encoding="utf-8")

    assert "kSchemaVersion = 1u" in cpu
    assert "kLaneCounts = {1, 3, 4, 8, 16, 32}" in cpu
    assert "sizeof(PackedComplexLaneGpu) == 64" in cpu
    assert "COMPLEX_SCHEMA_VERSION       1u" in gpu
    assert "COMPLEX_LANE_WORDS          16u" in gpu
    assert "COMPLEX_LANE_BYTES          64u" in gpu


def test_hot_schema_does_not_declare_an_ssbo_binding():
    gpu = (ROOT / "csrc/shaders/complex_transport.glsl.inc").read_text(encoding="utf-8")
    assert "layout(std430" not in gpu
    assert "binding =" not in gpu
