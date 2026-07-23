"""World-station menu that opens the optical camera workbench.

The world object and its display jar are deliberately lightweight.  This menu
only publishes navigation requests; camera physics remains owned by the
canonical camera manifest and the optical engine opened by the host.
"""
from __future__ import annotations

from typing import Any

from controls import Panel, readonly_knob
from camera_software.camera_manifest import resolve_camera_manifest


class CameraStationMenu:
    """Doc-rendered gateway from the Pluck room to camera/optics tools."""

    def __init__(self) -> None:
        camera = resolve_camera_manifest()
        mapping = camera.mapping()
        lens = dict(mapping["lens"])
        sensor = dict(mapping["sensor"])
        self._hud_visible = False
        self._host_station = None
        self._scene_workspace = None
        self._doc_ids: dict[str, int] = {}
        self._doc_action_rects: dict[str, tuple[int, int, int, int]] = {}
        self._open_optics_requested = False
        self._open_engine_requested = False
        self.status = "Camera specimen ready"
        self.camera_manifest_hash = camera.hash
        self.panel_spec = Panel(
            "camera_station_gateway",
            "CAMERA / OPTICS STATION",
            knobs=[
                readonly_knob("specimen", "Specimen", default="canonical 6x6 camera"),
                readonly_knob(
                    "lens", "Lens",
                    default=f"{float(lens['focal_length_mm']):g} mm  f/{float(lens['f_number']):g}",
                ),
                readonly_knob(
                    "gate", "Gate",
                    default=(
                        f"{float(sensor['physical_width_mm']):g} x "
                        f"{float(sensor['physical_height_mm']):g} mm"
                    ),
                ),
                readonly_knob("jar", "Display jar", default="dark optical glass"),
                readonly_knob("status", "Status", default=self.status),
            ],
            payload={
                "action_first": True,
                "actions": [
                    {"key": "open_optics", "label": "OPEN OPTICS BENCH"},
                    {"key": "open_engine", "label": "OPEN OPTICAL ENGINE"},
                ],
            },
        )

    def bind_host_station(self, station: Any) -> None:
        self._host_station = station

    def bind_scene_workspace(self, workspace: Any) -> None:
        self._scene_workspace = workspace

    def show_hud(self, visible: bool) -> None:
        self._hud_visible = bool(visible)

    @property
    def knob_values(self) -> dict[str, Any]:
        values = {
            knob.name: knob.default for knob in self.panel_spec.knobs
        }
        values["status"] = self.status
        return values

    def submit_doc_channel(self, doc_rdr: Any, win_w: int, win_h: int) -> None:
        if not self._hud_visible:
            return
        width = min(520, max(320, int(win_w) - 40))
        height = min(430, max(260, int(win_h) - 40))
        self._doc_action_rects.clear()
        doc_rdr.submit_panel(
            self.panel_spec,
            ((int(win_w) - width) // 2, (int(win_h) - height) // 2, width, height),
            node_id_map=self._doc_ids,
            knob_values=self.knob_values,
            action_rects=self._doc_action_rects,
        )

    def handle_event(self, event: Any) -> bool:
        if not self._hud_visible:
            return False
        try:
            import pygame
            if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
                return False
        except Exception:
            return False
        point = tuple(getattr(event, "pos", (-1, -1)))
        for route, rect in self._doc_action_rects.items():
            x, y, width, height = map(int, rect)
            if x <= point[0] < x + width and y <= point[1] < y + height:
                action = route.rsplit(".", 1)[-1]
                if action == "open_optics":
                    self._open_optics_requested = True
                    self.status = "Opening optics bench"
                elif action == "open_engine":
                    self._open_engine_requested = True
                    self.status = "Optical engine interface requested"
                else:
                    return False
                return True
        return False

    def consume_open_optics_request(self) -> bool:
        requested = self._open_optics_requested
        self._open_optics_requested = False
        return requested

    def consume_open_engine_request(self) -> bool:
        requested = self._open_engine_requested
        self._open_engine_requested = False
        return requested


__all__ = ["CameraStationMenu"]
