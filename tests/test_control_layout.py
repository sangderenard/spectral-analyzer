from controls import Panel, choice_knob, stepper_knob

from camera_software import (
    SensorRegion,
    layout_control_panel,
    photography_control_manifest,
)
import live_spectral_text_demo as demo


def test_panel_projection_preserves_knob_action_and_nested_click_routes():
    child = Panel(
        "child",
        "Child",
        knobs=[stepper_knob("passes", "Passes", "int", 1, 0, 8, 1)],
        payload={"actions": [{"key": "resume", "label": "Resume"}]},
    )
    panel = Panel(
        "root",
        "Root",
        knobs=[choice_knob("view", "View", ("scene", "image"), default=1)],
        panels=[child],
        payload={"actions": [{"key": "shoot", "label": "Shoot"}]},
    )

    design = layout_control_panel(panel, 220, 220)

    assert design.knob_routes["view"]["choices"] == ["scene", "image"]
    assert "root.shoot" in design.action_routes
    assert "child.resume" in design.action_routes
    rect = design.knob_routes["view"]["rect"]
    assert design.route_at(rect[0] + 1, rect[1] + 1) == ("knob", "view")
    action = design.action_routes["root.shoot"]
    assert design.route_at(action[0] + 1, action[1] + 1) == (
        "action", "root.shoot"
    )


def test_photography_manifest_becomes_objects_in_one_raytraced_scene():
    manifest = photography_control_manifest()
    design = layout_control_panel(
        manifest,
        230,
        260,
        rect=(10, 10, 230, 260),
        knob_values={"additional_passes": 3},
    )
    scene = demo.build_self_rendering_program_scene(
        "layout",
        display_width=320,
        display_height=320,
        control_layout=design,
    )

    layout_objects = [
        item for item in scene.objects
        if item.object_id.startswith("control-layout-")
    ]
    assert len(layout_objects) == len(design.elements)
    assert any(
        item.primitive.content == "Additional passes  3"
        for item in layout_objects
        if item.primitive.kind.value == "text"
    )
    assert design.route_at(13, 235) is not None
    assert all(
        request.sensor_region is not None
        for item in layout_objects
        for request in item.products
    )

def test_program_manifest_owns_three_hosts_and_context_requests():
    from camera_software import (
        layout_program_ui,
        program_frame_metrics,
        program_ui_manifest,
    )

    manifest = program_ui_manifest()
    metrics = program_frame_metrics(100, 100, manifest)
    layout = layout_program_ui(
        manifest,
        metrics["frame_width"],
        metrics["frame_height"],
        work_width=100,
        work_height=100,
    )

    camera = layout.region("camera-panel")
    work = layout.region("work-panel")
    browser = layout.region("asset-browser")
    assert work[0] == camera[0] + camera[2]
    assert browser[0] == work[0] + work[2]
    assert browser[2] == metrics["browser_width"] == 160
    assert browser[2] >= 2 * 80
    assert layout.roles["asset-browser"] == "widget_host"
    pipeline = layout.render_pipeline
    assert pipeline.font_family == "DejaVu Sans Mono"
    assert [tier.value for tier in pipeline.tiers] == [
        "monofont_glyphs",
        "whole_token_single_shots",
        "interface_after_render",
    ]
    assert pipeline.whole_token_policy == "single_shot_shrink_to_fit"
    assert pipeline.final_policy == "shared_camera_interface_after_render"
    assert set(layout.context_requests) == {"camera-panel", "work-panel"}
    browser_panel = next(
        panel for panel in manifest.panels if panel.name == "asset-browser"
    )
    assert browser_panel.payload["widget"] == "ScrollableSubpanelList"

def test_program_actions_are_manifest_owned_resolved_and_serialized():
    from camera_software import layout_program_ui, program_ui_manifest

    manifest = program_ui_manifest()
    layout = layout_program_ui(
        manifest, 360, 180, work_width=100, work_height=100
    )

    assert tuple(layout.actions) == (
        "work-visual-pass",
        "queue-pause-auto",
        "window-minimize",
        "window-maximize",
        "window-close",
    )
    assert layout.actions["work-visual-pass"].label == "VISUAL PASS"
    assert layout.actions["queue-pause-auto"].label == "PAUSE AUTO"
    assert layout.authored_text["work-visual-pass"] == "VISUAL PASS"
    assert layout.authored_text["queue-pause-auto"] == "PAUSE AUTO"
    assert set(layout.action_primitives) == set(layout.actions)
    assert all(
        primitive.mapping()["kind"] == "nine_slice"
        for primitive in layout.action_primitives.values()
    )
    first = layout.region("work-visual-pass")
    second = layout.region("queue-pause-auto")
    assert first[0] + first[2] <= second[0]
    contract = layout.mapping()
    assert contract["actions"]["work-visual-pass"]["align"] == "start"
    assert contract["authored_text"]["work-visual-pass"] == "VISUAL PASS"
    visual_primitive = contract["action_primitives"]["work-visual-pass"]
    assert visual_primitive["kind"] == "nine_slice"
    assert len(visual_primitive["object_requests"]) == 9
    assert {
        request["subtype_key"]
        for request in visual_primitive["object_requests"]
    } == {"representative-square"}
    assert len(
        contract["panel_primitives"]["camera-panel"]["object_requests"]
    ) == 9
    assert len(contract["panel_object_requests"]["camera-panel"]) == 9
    assert len(contract["panel_object_requests"]["work-visual-pass"]) == 9
    assert set(layout.panel_object_requests) >= {
        "camera-panel", "work-panel", "work-visual-pass"
    }
    assert "VISUAL PASS" in layout.render_pipeline.static_tokens
    assert "PAUSE AUTO" in layout.render_pipeline.static_tokens