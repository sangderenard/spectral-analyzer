"""Observer composition checks; GPU transport proof is a separate opt-in gate."""
import ast
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from camera_designer.studio_boundary_media import STUDIO_BOUNDARY_GLSL
from camera_designer.studio_directional_field import recording_shader, ROOT_ESTIMATOR
from camera_designer.studio_directional_media import (
    INTEGRAL_GLSL, medium_rate_glsl, with_internal_medium_recording,
)
from camera_designer.studio_region_gpu import region_boundary_helpers, with_region_media


def canonical_source():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "demo_pluck_gl.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_GPU_RAY_FIELD_CS" for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value.replace("__STUDIO_BOUNDARY_HELPERS__", STUDIO_BOUNDARY_GLSL)
    raise AssertionError("canonical forward shader literal missing")


def test_rate_is_derived_from_current_canonical_optics_without_endpoint_log():
    source = medium_rate_glsl(STUDIO_BOUNDARY_GLSL)
    assert "float k=studio_medium_optics(medium,frequency).y;" in source
    assert "return 4.0*3.14159265358979323846*k*frequency/299792458.0;" in source
    assert "if(medium<0) return 0.0;" in source
    assert "log(" not in source and "distance" not in source


def test_region_rate_preserves_installed_additive_dye_and_legacy_delegate():
    source = medium_rate_glsl(region_boundary_helpers())
    assert source.count("float studio_record_medium_sigma(") == 1
    assert source.count("float studio_record_legacy_medium_sigma(") == 1
    assert "if(medium==-2)return -1.0;" in source
    assert "studio_record_legacy_medium_sigma(medium,frequency)" in source
    assert "k+=studio_medium_optics(int(studio_region_record(i).y),frequency).y;" in source
    assert "studio_is_active(0) && studio_is_active(i) && int(studio_region_record(i).x)==2" in source


@pytest.mark.parametrize("regions", [False, True])
@pytest.mark.parametrize("root_sweep", [False, True])
def test_observer_receives_entry_packet_and_scalar_and_beer_are_unchanged(regions, root_sweep):
    original = canonical_source()
    if regions:
        original = with_region_media(original)
    kwargs = dict(estimator=ROOT_ESTIMATOR, grid_size=64, independent_bounds=True) if root_sweep else {}
    recorded = recording_shader(original, **kwargs)
    result = with_internal_medium_recording(recorded)
    medium_source = region_boundary_helpers() if regions else STUDIO_BOUNDARY_GLSL
    assert medium_source in result  # original authority, not rewritten attenuation
    hook = "studio_record_segment(ro,hp,packet,currentMedium);\n        packet.z*=studio_medium_transmittance(currentMedium,packet.x,best);"
    assert result.count(hook) == 2
    guard = "==-1" if regions else "<0"
    for call in ("splat_segment_multiscale(ro, hp, packet, path_len);", "splat_segment_spectral(ro, hp, packet);"):
        assert f"if(currentMedium{guard}) {{ {call} }}" in result
    assert "if(medium < -1)return;" in result
    assert "studio_record_medium_integral(studio_record_sigma,cursor,next)" in result
    assert "studio_record_track(q,d,packet,next-cursor);" not in result
    assert result.count("studio_record_bin=0;") == recorded.count("studio_record_bin=0;")
    if root_sweep:
        assert "studio_record_root_weight" in result and "uStudioRecordMin" in result


def test_incomplete_grid_composition_and_changed_laws_fail_closed():
    recorded = recording_shader(canonical_source())
    with pytest.raises(ValueError, match="joint attenuation"):
        with_internal_medium_recording(recorded, volume_extinction=True)
    with pytest.raises(ValueError, match="joint attenuation"):
        with_internal_medium_recording(recorded + "\nvec2 studio_volume_path(vec3 a,vec3 b);")
    with pytest.raises(ValueError, match="already installed"):
        with_internal_medium_recording(with_internal_medium_recording(recorded))
    with pytest.raises(ValueError, match="Beer-rate"):
        medium_rate_glsl(STUDIO_BOUNDARY_GLSL.replace("return exp(-", "return 0.5*exp(-"))
    with pytest.raises(ValueError, match="air identity"):
        medium_rate_glsl(STUDIO_BOUNDARY_GLSL.replace("if(medium<0) return 1.0;", "if(medium<0) return 0.5;"))


def test_stable_integral_equation_for_clear_and_authored_millimetre_rods():
    # Independent analytic checks of the intended integral, not a GPU claim.
    # The real subject's millimetre transmission samples span these optical depths.
    for transmission in (1., .91, .73, .43, .39, .1, .08, np.exp(-100)):
        sigma = -np.log(transmission) / .001
        for first, last in ((0., .0005), (.0005, .001), (.1, .1005)):
            span = last-first
            tau = sigma*span
            stable = span*(1-tau*.5+tau*tau/6) if tau < .001 else -np.expm1(-tau)/sigma
            actual = np.exp(-sigma*first)*stable
            expected = span if sigma == 0 else np.exp(-sigma*first)*(-np.expm1(-sigma*span))/sigma
            assert actual == pytest.approx(expected, rel=5e-11, abs=1e-30)
    assert "exp(-sigma*max(first,0.0))*within" in INTEGRAL_GLSL


@pytest.mark.parametrize("regions", [False, True])
def test_derived_helpers_compile_with_real_glsl_compiler(tmp_path, regions):
    compiler = shutil.which("glslangValidator")
    assert compiler, "official glslangValidator is required for the helper gate"
    material = region_boundary_helpers() if regions else STUDIO_BOUNDARY_GLSL
    helpers = medium_rate_glsl(material)
    prefix = """#version 430 core
layout(local_size_x=1) in;
layout(std430,binding=0) buffer Output { float result[]; };
int uStudioRegionCount=0;
bool studio_is_active(int id){return false;}
vec4 studio_region_record(int id){return vec4(0);}
vec2 studio_medium_optics(int medium,float frequency){return vec2(1.5,0.000001);}
"""
    path = tmp_path / "medium_integral.comp"
    path.write_text(prefix + helpers + INTEGRAL_GLSL + "\nvoid main(){result[0]=studio_record_medium_integral(studio_record_medium_sigma(0,5e14),0.0,0.001);}", encoding="utf-8")
    run = subprocess.run([compiler, str(path)], capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr
