"""Canonical Panel/KnobSpec manifests for the program's tool grids."""
from __future__ import annotations

from controls import Panel, button_knob, choice_knob, readonly_knob


RAY_OPTIONS = (
    1_024, 4_096, 16_384, 65_536, 204_800, 262_144, 1_048_576,
    4_194_304, 16_777_216, 67_108_864,
)
EPOCH_OPTIONS = (1, 2, 4, 8, 16, 32, 64, 128)
BUNDLE_OPTIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SAMPLE_OPTIONS = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096,
)
T5_OPTIONS = (
    32_768, 65_536, 262_144, 1_048_576, 4_194_304,
    16_777_216, 67_108_864, 268_435_456,
)
BOUNCE_OPTIONS = (1, 2, 4, 8, 16, 32, 64, 128)
GRID_MODE_OPTIONS = ("N-TREE", "LOCKED")
ALLOCATION_MODE_OPTIONS = ("FOCUS / EXPLORE", "EVEN SENSOR", "N-TREE PREVIEW")
SUBDIVISION_AXIS_OPTIONS = (2, 3)
LOCKED_GRID_OPTIONS = (1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 24, 32, 48, 64)

ISO_OPTIONS = (50, 100, 200, 400, 800, 1600, 3200, 6400)
SHUTTER_OPTIONS = (
    1/1000, 1/500, 1/250, 1/125, 1/60, 1/30,
    1/15, 1/8, 1/4, 1/2, 1.0,
)
FLASH_OPTIONS = ("scene", "on", "off")
LIGHTING_OPTIONS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
SENSOR_NAMES = ("CANON R6", "NIKON D850")
FILM_NAMES = ("PORTRA 400", "SUPERIA 200")
WORK_DIM_OPTIONS = (64, 100, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1024)
FINAL_EDGE_OPTIONS = (
    512, 768, 1000, 1024, 1200, 1600, 2000, 2400, 3200, 4000, 4800, 6000, 8000,
)
FOCAL_LENGTH_OPTIONS = (24.0, 35.0, 50.0, 55.0, 70.0, 82.5, 100.0, 110.0)
F_NUMBER_OPTIONS = (1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0)
CAMERA_ANGLE_OPTIONS = (
    -30.0, -20.0, -15.0, -10.0, -5.0, -2.0, 0.0,
    2.0, 5.0, 10.0, 15.0, 20.0, 30.0,
)
CAMERA_OFFSET_OPTIONS = (
    -1.0, -0.5, -0.25, -0.10, -0.05, 0.0,
    0.05, 0.10, 0.25, 0.5, 1.0,
)


def _grid_payload(columns: int, palette: str) -> dict:
    return {
        "role": "control_grid",
        "widget": "GridKnobToolbar",
        "palette": palette,
        "grid": {
            "mode": "grid",
            "columns": int(columns),
            "title_width": 112,
            "title_overlay": True,
            "padding": 1,
            "padding_y": 0,
            "column_gap": 2,
            # Logical height cap. The program layout may shrink it further.
            "row_height": 13,
        },
        "geometry": {"kind": "plane"},
    }


def integrator_toolbar_panel() -> Panel:
    return Panel(
        "integrator-toolbar",
        "INTEGRATOR",
        knobs=[
            choice_knob("transport_mode", "MODE", ("continuous", "fixed", "depth"), widget="stepper"),
            choice_knob("lane_count", "LANES", ("1", "3", "8", "16", "32"), widget="stepper"),
            choice_knob("total_rays", "RAYS", tuple(f"{value:,}" for value in RAY_OPTIONS), widget="stepper"),
            choice_knob("max_sensor_epochs", "EPOCHS", EPOCH_OPTIONS, widget="stepper"),
            choice_knob("epoch_bundle_count", "BUNDLES", BUNDLE_OPTIONS, widget="stepper"),
            choice_knob("sensor_samples_per_node", "SPP/NODE", SAMPLE_OPTIONS, widget="stepper"),
            choice_knob("max_bounces", "BOUNCES", BOUNCE_OPTIONS, widget="stepper"),
            choice_knob("sensor_t5_pair_budget", "T5 PAIRS", tuple(f"{value:.1e}" for value in T5_OPTIONS), widget="stepper"),
        ],
        payload=_grid_payload(8, "blue"),
    )


def camera_toolbar_panel() -> Panel:
    return Panel(
        "camera-toolbar", "CAMERA",
        knobs=[
            choice_knob("sensor_id", "SENSOR", SENSOR_NAMES, widget="stepper"),
            choice_knob(
                "shutter_s", "SHUTTER",
                tuple(f"1/{round(1/value)}" if value < 1 else f"{value:g}s" for value in SHUTTER_OPTIONS),
                widget="stepper",
            ),
            choice_knob("yaw_offset_deg", "YAW", CAMERA_ANGLE_OPTIONS, widget="stepper"),
            choice_knob("pitch_offset_deg", "PITCH", CAMERA_ANGLE_OPTIONS, widget="stepper"),
            choice_knob("height_offset_m", "HEIGHT", CAMERA_OFFSET_OPTIONS, widget="stepper"),
            choice_knob("sideways_offset_m", "SIDE", CAMERA_OFFSET_OPTIONS, widget="stepper"),
        ],
        payload=_grid_payload(6, "amber"),
    )


def lens_toolbar_panel() -> Panel:
    return Panel(
        "lens-toolbar", "LENS",
        knobs=[
            choice_knob("focal_length_mm", "FOCAL", FOCAL_LENGTH_OPTIONS, widget="stepper"),
            choice_knob("f_number", "F/", F_NUMBER_OPTIONS, widget="stepper"),
        ],
        payload=_grid_payload(2, "amber"),
    )


def light_toolbar_panel() -> Panel:
    return Panel(
        "light-toolbar", "LIGHT",
        knobs=[
            choice_knob("flash_mode", "FLASH", tuple(value.upper() for value in FLASH_OPTIONS), widget="stepper"),
            choice_knob("lighting_ev", "LIGHT EV", tuple(f"{value:+d}" for value in LIGHTING_OPTIONS), widget="stepper"),
        ],
        payload=_grid_payload(2, "amber"),
    )


def film_toolbar_panel() -> Panel:
    return Panel(
        "film-toolbar", "FILM",
        knobs=[
            choice_knob("film_id", "FILM", FILM_NAMES, widget="stepper"),
            choice_knob("iso", "ISO", ISO_OPTIONS, widget="stepper"),
        ],
        payload=_grid_payload(2, "amber"),
    )


def exposure_toolbar_panel() -> Panel:
    return Panel(
        "exposure-toolbar", "EXPOSURE",
        knobs=[
            choice_knob(
                "allocation_mode", "SAMPLING", ALLOCATION_MODE_OPTIONS,
                widget="stepper",
            ),
            choice_knob("grid_mode", "TREE", GRID_MODE_OPTIONS, widget="stepper"),
            choice_knob("subdivision_axis", "N", SUBDIVISION_AXIS_OPTIONS, widget="stepper"),
            choice_knob("locked_grid_columns", "GRID X", LOCKED_GRID_OPTIONS, widget="stepper"),
            choice_knob("locked_grid_rows", "GRID Y", LOCKED_GRID_OPTIONS, widget="stepper"),
            choice_knob("work_width_px", "WORK W", WORK_DIM_OPTIONS, widget="stepper"),
            choice_knob("work_height_px", "WORK H", WORK_DIM_OPTIONS, widget="stepper"),
            choice_knob("final_edge_px", "FINAL □", FINAL_EDGE_OPTIONS, widget="stepper"),
        ],
        payload=_grid_payload(8, "amber"),
    )


def ray_trace_toolbar_panel() -> Panel:
    """Compatibility name for the now explicitly named integrator row."""
    return integrator_toolbar_panel()


def render_progress_panel() -> Panel:
    return Panel(
        "render-progress-toolbar",
        "RENDER",
        knobs=[
            readonly_knob("phase", "PHASE"),
            readonly_knob("overall", "OVERALL"),
            readonly_knob("elapsed", "ELAPSED"),
            readonly_knob("eta", "ETA"),
            choice_knob(
                "pause_render", "PAUSE / RESUME", ("PAUSE", "RESUME"),
                widget="button",
            ),
            button_knob("cancel_render", "CANCEL"),
        ],
        payload={
            **_grid_payload(6, "status"),
            "role": "status",
        },
    )


def work_preview_tabs_panel() -> Panel:
    """Canonical overlay tabs for choosing the work-panel presentation."""

    return Panel(
        "work-preview-tabs",
        "VIEW",
        knobs=[
            button_knob("whole-work", "WHOLE WORK"),
            button_knob("work-piece", "WORK PIECE"),
        ],
        payload={
            **_grid_payload(2, "status"),
            "role": "view_tabs",
            "grid": {
                "mode": "grid",
                "columns": 2,
                "title_width": 42,
                "padding": 1,
                "column_gap": 2,
                "row_height": 24,
            },
        },
    )


__all__ = [
    "RAY_OPTIONS", "EPOCH_OPTIONS", "BUNDLE_OPTIONS", "SAMPLE_OPTIONS",
    "T5_OPTIONS", "BOUNCE_OPTIONS", "ISO_OPTIONS", "SHUTTER_OPTIONS",
    "GRID_MODE_OPTIONS", "SUBDIVISION_AXIS_OPTIONS", "ALLOCATION_MODE_OPTIONS",
    "LOCKED_GRID_OPTIONS",
    "FLASH_OPTIONS", "LIGHTING_OPTIONS", "SENSOR_NAMES", "FILM_NAMES",
    "WORK_DIM_OPTIONS", "FINAL_EDGE_OPTIONS", "FOCAL_LENGTH_OPTIONS",
    "F_NUMBER_OPTIONS", "CAMERA_ANGLE_OPTIONS", "CAMERA_OFFSET_OPTIONS",
    "camera_toolbar_panel", "lens_toolbar_panel",
    "light_toolbar_panel", "film_toolbar_panel", "integrator_toolbar_panel",
    "ray_trace_toolbar_panel", "exposure_toolbar_panel",
    "render_progress_panel", "work_preview_tabs_panel",
]
