from abstract_ui_material_catalog import build_material_catalog


def test_abstract_ui_catalog_uses_material_database_shader_chunks():
    catalog = build_material_catalog()

    assert catalog["schema"] == "abstract-ui-material-catalog-v0"
    assert catalog["record_count"] == len(catalog["records"])
    assert catalog["record_count"] > 0
    assert catalog["chunk_strides"] == {
        "pbr": 16,
        "phong": 8,
        "enamel": 8,
        "texture_stack": 16,
    }
    assert catalog["surface_policy"] == {
        "profile": "matte-world-base",
        "minimum_roughness": 0.72,
        "maximum_specular_strength": 0.18,
        "maximum_shininess": 24.0,
        "minimum_indirect_ambient": 0.06,
        "maximum_metallic": 0.08,
        "transmission": 0.0,
        "emission": [0.0, 0.0, 0.0],
        "enamel_thickness_nm": 0.0,
    }
    for record in catalog["records"]:
        assert len(record["pbr"]) == 16
        assert len(record["phong"]) == 8
        assert len(record["enamel"]) == 8
        assert len(record["texture_stack"]) == 16
        assert record["pbr"][3] >= 0.72
        assert record["phong"][1] <= 0.18
        assert record["phong"][0] >= 0.06
        assert record["phong"][2] <= 24.0
        assert record["pbr"][4] <= 0.08
        assert record["pbr"][5] == 0.0
        assert record["pbr"][8:11] == [0.0, 0.0, 0.0]
        assert record["enamel"][0] == 0.0
