from compile_base_material_webgl import (
    compile_base_material_webgl,
    compile_base_material_webgl_chain,
)


def test_real_base_material_shader_compiles_through_project_ir_to_webgl():
    translated = compile_base_material_webgl()

    assert translated.complete, [
        item.format() for item in translated.diagnostics
    ]
    project = translated.project_ir
    assert project.schema == "turing.glsl-project-ir.v1"
    assert len(project.instructions) >= 350
    assert {item.member_name for item in project.storage_buffers} == {
        "pbr", "phong", "enamel", "texstack", "thermal_k",
    }
    assert translated.execution_plan.compute_sections == ()
    assert all(
        section.stage in {"fragment", "inactive"}
        for section in translated.execution_plan.sections
    )
    assert translated.source.startswith("#version 300 es")
    assert "layout(std430" not in translated.source
    assert "turing_ssbo_load_phong(int(id*PHONG_STRIDE+1))" in translated.source
    assert "bool front_facing = gl_FrontFacing;" in translated.source
    assert "FragColor = vec4(srgb, alpha);" in translated.source
    assert "uniform float uExposure" in translated.source
    assert "ACES approximation" in translated.source
    assert "PLUCK_SHADOW_MAP" in translated.source
    assert "uniform sampler2DArray uShadowMaps" in translated.source
    assert "light_shadow_visibility(i, N, L)" in translated.source


def test_real_base_material_vertex_and_fragment_stages_form_a_webgl_chain():
    vertex, fragment = compile_base_material_webgl_chain()

    assert vertex.complete and fragment.complete
    assert vertex.project_ir.stage_hint == "vertex"
    assert fragment.project_ir.stage_hint == "fragment"
    assert vertex.execution_plan.sections[-1].stage == "vertex"
    assert fragment.execution_plan.sections[-1].stage == "fragment"
    assert "gl_Position = uMVP * vec4(aPos, 1.0);" in vertex.source
    assert "FragColor = vec4(srgb, alpha);" in fragment.source
