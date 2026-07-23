from camera_software.gpu_preview import (
    PreviewProductKind,
    PreviewProductRegistry,
    PreviewProductPublisher,
    PreviewTextureProduct,
    RayPipelinePreviewBridge,
)


def product(product_id="camera.surface", generation=1, **kwargs):
    values = dict(
        product_id=product_id,
        tab_label="SURFACE",
        texture_id=7,
        width=256,
        height=256,
        generation=generation,
        producer="camera",
    )
    values.update(kwargs)
    return PreviewTextureProduct(**values)


def test_registry_retains_latest_generation_in_stable_tab_order():
    registry = PreviewProductRegistry()
    assert registry.publish(product(generation=1))
    assert not registry.publish(product(generation=1))
    assert not registry.publish(product(generation=0))
    assert registry.publish(product(generation=2))
    assert registry.get("camera.surface").generation == 2
    assert [item.product_id for item in registry.snapshot()] == ["camera.surface"]


def test_registry_supports_accumulation_and_processing_group_products():
    registry = PreviewProductRegistry()
    registry.publish(product(
        "wave.accumulation", kind=PreviewProductKind.ACCUMULATION,
        group_id="wave-arena",
    ))
    registry.publish(product(
        "pipeline.t3", kind=PreviewProductKind.PROCESSING_GROUP,
        group_id="ray-pipeline",
    ))
    assert [item.product_id for item in registry.products_for_group("wave-arena")] == [
        "wave.accumulation"
    ]
    assert registry.withdraw("pipeline.t3")
    assert registry.get("pipeline.t3") is None


def test_product_rejects_invalid_texture_contract():
    import pytest
    with pytest.raises(ValueError):
        product(texture_id=0)


def test_publisher_expresses_wave_and_pipeline_live_views():
    registry = PreviewProductRegistry()
    pipeline = PreviewProductPublisher(
        registry, "ray-pipeline", group_id="pipeline"
    )
    wave = PreviewProductPublisher(registry, "wave-arena", group_id="wave")
    pipeline.publish_processing_group("pipeline.t3", "T3 HITS", 3, 32, 32)
    wave.publish_accumulation("wave.energy", "WAVE ENERGY", 4, 64, 64)
    wave.publish_complex_field("wave.phase", "WAVE PHASE", 5, 64, 64)
    assert registry.get("pipeline.t3").kind is PreviewProductKind.PROCESSING_GROUP
    assert registry.get("wave.energy").kind is PreviewProductKind.ACCUMULATION
    assert registry.get("wave.phase").kind is PreviewProductKind.COMPLEX_FIELD


def test_native_pipeline_bridge_publishes_surface_and_field_generations():
    class Tracer:
        def surface_scan_texture_info(self):
            return {"texture_id": 9, "width": 32, "height": 32,
                    "depth": 1, "generation": 4}

        def get_field_display_texture_info(self):
            return {"texture_id": 10, "width": 16, "height": 8,
                    "depth": 12, "generation": 7, "texture_target": 0x806F}

        def get_wave_arena_texture_info(self):
            return {"texture_id": 13, "width": 64, "height": 64,
                    "depth": 1, "generation": 5, "arena_id": 2,
                    "band": 1, "direction": 0}

        def get_camera_geometry_texture_info(self):
            return {"texture_id": 11, "width": 320, "height": 180,
                    "depth": 1, "generation": 2}

        def get_light_field_texture_info(self):
            return {"texture_id": 12, "width": 128, "height": 128,
                    "depth": 6, "generation": 3, "texture_target": 0x8C1A}

    registry = PreviewProductRegistry()
    products = RayPipelinePreviewBridge(registry, Tracer()).poll()
    assert [item.product_id for item in products] == [
        "camera.surface-scan", "transport.complex-accumulation",
        "wave.arena-state",
        "camera.geometry", "camera.light-field",
    ]
    field = registry.get("transport.complex-accumulation")
    assert field.depth == 12
    assert field.tab_label == "COMPLEX TRANSPORT"
    assert field.metadata["representation"] == "ray-carried-complex-amplitude"
    assert field.metadata["wave_solver"] is False
    wave = registry.get("wave.arena-state")
    assert wave.tab_label == "WAVE ARENA"
    assert wave.metadata["wave_solver"] is True
    assert wave.metadata["arena_id"] == 2
    assert registry.get("camera.geometry").kind is PreviewProductKind.CAMERA_GEOMETRY
    assert registry.get("camera.light-field").kind is PreviewProductKind.LIGHT_FIELD
