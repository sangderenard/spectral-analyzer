"""Canonical secondary field export and real hidden-GL regression gates."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from compile_base_material_webgl import compile_base_material_webgl_chain


def test_secondary_field_hidden_gpu_pixels():
    probe = Path(__file__).with_name("studio_secondary_field_probe.py")
    result = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["status"] == "PASS"
    assert receipt["checks"] >= 5
    assert receipt["legacy_bit_identical"]
    assert receipt["current_response_fallback"]


def test_secondary_field_project_ir_export_and_es_link(tmp_path):
    vertex, fragment = compile_base_material_webgl_chain()
    assert vertex.complete and fragment.complete
    assert "invariant gl_Position;" in vertex.source
    assert fragment.project_ir.schema == "turing.glsl-project-ir.v1"
    source = fragment.source
    for name in ("uLinearOutput", "uResponseRows", "uResponseCount", "uResponseQueryContext",
                 "uResponseDimensions", "uResponseWorldFromView", "uResponseStencil",
                 "uFieldResolvedColor", "uFieldResolvedVolume", "uResponseBinCount", "PLUCK_SECONDARY_FIELD"):
        assert name in source
    assert "vResponseRecipient" in source and "aResponseRecipient" in vertex.source
    assert "field_material_mask" not in source
    assert "direction_moment" not in source
    assert "float(b.w) / float(c.x)" in source
    assert "turing_ssbo_load_pbr" in source
    assert "turing_ssbo_load_phong" in source
    assert "turing_ssbo_load_texstack" in source
    assert "layout(std430" not in source
    # Actual exported ES syntax/type/link check, including shadow specialization.
    compiler = shutil.which("glslangValidator")
    assert compiler, "Official Vulkan SDK glslangValidator is required for this gate"
    vert = tmp_path / "secondary.vert"
    frag = tmp_path / "secondary.frag"
    vert.write_text(vertex.source, encoding="utf-8")
    for shadow, primary in ((False, False), (True, False), (True, True)):
        defines = "#define PLUCK_TRANSMISSION 1\n#define PLUCK_PREMULTIPLIED_GLASS 1\n#define PLUCK_SECONDARY_FIELD 1\n"
        if primary:
            defines = defines.replace("#define PLUCK_SECONDARY_FIELD 1", "#define PLUCK_FIELD_PRIMARY 1")
        if shadow:
            defines += "#define PLUCK_SHADOW_MAP 1\n"
        frag.write_text(source.replace("#version 300 es\n", "#version 300 es\n" + defines, 1), encoding="utf-8")
        run = subprocess.run([compiler, "-l", str(vert), str(frag)], capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stdout + run.stderr
