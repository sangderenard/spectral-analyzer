import json

import exposure_render_demo as exposure
import live_spectral_text_demo as demo
import scene_orders
from camera_software import (
    DisplayPrimitive,
    DisplayPrimitiveKind,
    DisplayProductKind,
    DisplayProductRequest,
    DisplaySceneInventory,
    DisplaySceneWorkScheduler,
    ExposureProgressEvent,
    ExposureProgressKind,
    SensorRegion,
)


def _two_object_scene(revision: int = 1):
    management = demo.management_display_object(
        "menu-control",
        DisplayPrimitiveKind.ICON,
        center_m=(0.05, 0.20, 0.12),
        size_m=(0.28, 0.18),
        icon_name="menu",
        label="tools",
        revision=revision,
    )
    return demo.build_program_display_scene(
        "one physical camera",
        display_width=80,
        display_height=60,
        revision=revision,
        extra_objects=(management,),
    )


def test_management_primitives_produce_authored_geometry():
    assert DisplayPrimitive(DisplayPrimitiveKind.BOX, label="panel").authored_text() == "panel"
    assert DisplayPrimitive(DisplayPrimitiveKind.BOX).authored_text() == ""
    assert DisplayPrimitive(
        DisplayPrimitiveKind.ICON, icon_name="close", label="dismiss"
    ).authored_text() == "× dismiss"


def test_display_scene_order_places_all_objects_under_one_fixed_camera():
    scene = _two_object_scene()
    order = demo.build_display_scene_order(scene, sensor_sweeps=4)
    job = scene_orders.resolved_jobs(order, demo.JOB_ID)[0]

    assert job["camera"]["position_m"] == list(scene.camera.position_m)
    assert job["camera"]["target_m"] == list(scene.camera.target_m)
    assert [item["id"] for item in job["objects"]] == ["main-text"]
    assert {
        "display-surface-main-text",
        "display-surface-menu-control",
        "display-icon-menu-control-top",
        "display-icon-menu-control-middle",
        "display-icon-menu-control-bottom",
    }.issubset({plane["id"] for plane in job["planes"]})


def test_multi_object_compiler_keeps_shared_camera_and_builds_every_object():
    scene = _two_object_scene()
    job = scene_orders.resolved_jobs(
        demo.build_display_scene_order(scene), demo.JOB_ID
    )[0]
    base = exposure._build_thick_lens_lab_tracer_scene()
    compiled, report = scene_orders.compile_job(base, job)

    assert report.camera_position_m == scene.camera.position_m
    assert report.plane_triangles == 12 * len(job["planes"])
    assert report.glyph_triangles > 0
    assert compiled.camera_tri_groups["sensor"].size == base.camera_tri_groups["sensor"].size
    assert compiled.camera_tri_groups["thin_lens"].size == base.camera_tri_groups["thin_lens"].size


def test_inventory_persists_authored_objects_and_shared_progress(tmp_path):
    path = tmp_path / "inventory.json"
    scene = _two_object_scene()
    inventory = DisplaySceneInventory(str(path))
    inventory.put_scene(scene)
    event = ExposureProgressEvent(
        exposure_id="shared-exposure",
        sequence=2,
        kind=ExposureProgressKind.LAYER_AVAILABLE,
        region=SensorRegion(160, 120, 80, 60),
        pass_index=2,
        linear_accumulation_path="shared.npy",
    )
    inventory.record_progress(
        scene.scene_id,
        (item.object_id for item in scene.objects),
        event,
    )

    restored = DisplaySceneInventory(str(path)).snapshot()[0]
    assert restored.scene.camera == scene.camera
    assert [item.object_id for item in restored.scene.objects] == [
        item.object_id for item in scene.objects
    ]
    assert all(state.products[0].linear_path == "shared.npy" for state in restored.objects)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_scheduler_round_robins_scenes_without_changing_their_cameras(tmp_path):
    inventory = DisplaySceneInventory(str(tmp_path / "inventory.json"))
    first = _two_object_scene()
    second_base = _two_object_scene()
    second = type(second_base)(
        scene_id="second-scene",
        camera=type(second_base.camera)(
            position_m=(4.0, -1.0, 1.5),
            target_m=(0.0, 0.0, 0.0),
            focus_target_m=(0.0, 0.0, 0.0),
        ),
        sensor_width=second_base.sensor_width,
        sensor_height=second_base.sensor_height,
        objects=second_base.objects,
    )
    inventory.put_scene(first)
    inventory.put_scene(second)
    scheduler = DisplaySceneWorkScheduler(inventory)

    leases = [scheduler.next_lease(8), scheduler.next_lease(8)]
    assert {lease.scene_id for lease in leases} == {first.scene_id, second.scene_id}
    assert leases[0].scene_id != leases[1].scene_id
    assert all(lease.work_units == 8 for lease in leases)
    cameras = {state.scene.scene_id: state.scene.camera for state in inventory.snapshot()}
    assert cameras[first.scene_id] == first.camera
    assert cameras[second.scene_id] == second.camera


def test_product_region_must_fit_fixed_scene_sensor():
    scene = _two_object_scene()
    bad_main = type(scene.objects[0])(
        **{
            **scene.objects[0].__dict__,
            "products": (
                DisplayProductRequest(
                    DisplayProductKind.IMAGE,
                    SensorRegion(scene.sensor_width, 0, 1, 1),
                ),
            ),
        }
    )
    try:
        type(scene)(
            **{**scene.__dict__, "objects": (bad_main, *scene.objects[1:])}
        )
    except ValueError as exc:
        assert "fit within the scene sensor" in str(exc)
    else:
        raise AssertionError("out-of-bounds product regions must be rejected")
