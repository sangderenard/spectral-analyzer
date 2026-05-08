"""player_tool.py — Tool item protocol and concrete implementations.

ToolItem
    Minimal base class for hotbar-equippable tools.

FlashlightTool
    Wraps a PlayerFlashlight.  Placed in a hotbar slot; selecting the slot
    equips it.  G toggles on/off; [ / ] adjust bulb position.

FlashlightPickup
    World object that floats in place (on, glowing) until the player presses
    E to collect it into the first open hotbar slot.  Follows the
    interaction_triangles_world / try_pickup_* protocol used by duty_stations.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class ToolItem:
    """Base class for hotbar-equippable tools."""
    tool_type: str = "generic"
    label: str     = "Tool"


# ─────────────────────────────────────────────────────────────────────────────
# Flashlight tool (lives in a hotbar slot)
# ─────────────────────────────────────────────────────────────────────────────

class FlashlightTool(ToolItem):
    """Flashlight tool — select its hotbar slot to equip; G to toggle."""
    tool_type = "flashlight"
    label     = "Flashlight"

    def __init__(self, flashlight=None):
        try:
            from player_flashlight import PlayerFlashlight
            self.flashlight = (flashlight
                               if flashlight is not None
                               else PlayerFlashlight())
        except Exception:
            self.flashlight = flashlight
        if self.flashlight is not None:
            self.flashlight.enable()


# ─────────────────────────────────────────────────────────────────────────────
# Flashlight world pickup
# ─────────────────────────────────────────────────────────────────────────────

class FlashlightPickup:
    """Floating world pickup — press E to collect into first open hotbar slot.

    The pickup displays the full flashlight geometry (already on) at
    ``world_position``, cup facing up (+Z).  Once collected, all geometry
    methods return empty arrays.

    Interaction protocol (compatible with duty_stations):
      interaction_triangles_world()  → (N, 3, 3) float64  for ray-picking
      unfinished_tooltip_lines()     → ["Flashlight", "[E] pick up"]
      try_pickup_tool(tool_slots)    → bool; places FlashlightTool in first slot
      handle_menu_event(ev)          → False (no menu)
      world_tris_by_material()       → [(mat_name, tris), ...] for rendering
    """

    label: str = "Flashlight"
    tool_type: str = "flashlight"
    pickup_kind: str = "tool"

    def __init__(self, pos):
        self.pos            = np.array(pos, np.float64)
        self.world_position = self.pos
        self.interaction_radius: float = 2.2
        self._collected: bool = False
        self._world_visible: bool = True
        self._published_world: bool = False
        self._last_pick_slot: int = -1
        self.obj_id: str = "flashlight_pickup"

        try:
            from player_flashlight import PlayerFlashlight
            self._flashlight = PlayerFlashlight()
            self._flashlight.enable()
        except Exception:
            self._flashlight = None

    # ── Pickup logic ──────────────────────────────────────────────────────────

    @property
    def quantity(self) -> int:
        return 0 if self._collected else 1

    def try_pickup_tool(self, tool_slots: list, tool_inventory: Optional[list] = None) -> bool:
        """Place a FlashlightTool in the first empty slot. Returns True on success."""
        if self._collected or self._flashlight is None:
            return False
        for i, slot in enumerate(tool_slots):
            if slot is None:
                tool = FlashlightTool(self._flashlight)
                tool_slots[i] = tool
                self._last_pick_slot = int(i)
                if isinstance(tool_inventory, list) and tool not in tool_inventory:
                    tool_inventory.append(tool)
                self._collected = True
                self._world_visible = False
                self.interaction_radius = 0.0
                return True
        return False

    @property
    def last_pick_slot(self) -> int:
        return int(self._last_pick_slot)

    def handle_menu_event(self, ev) -> bool:
        return False

    def unfinished_tooltip_lines(self) -> list:
        if self._collected:
            return []
        return ["Flashlight", "[E] pick up → hotbar"]

    # ── Geometry ──────────────────────────────────────────────────────────────

    def pickup_transform(self) -> np.ndarray:
        """4×4 column-major transform: cup faces +Z, centred at world_position."""
        M = np.eye(4, dtype=np.float32)
        if self._flashlight is None:
            M[:3, 3] = self.pos.astype(np.float32)
            return M
        # right=X  up=Y  forward(cup opening)=+Z
        M[:3, 0] = [1.0, 0.0, 0.0]
        M[:3, 1] = [0.0, 1.0, 0.0]
        M[:3, 2] = [0.0, 0.0, 1.0]
        # Shift back so the cup midpoint sits at world_position
        half_depth = float(self._flashlight.cup_depth) * 0.5
        M[:3, 3] = (self.pos + np.array([0.0, 0.0, -half_depth])).astype(np.float32)
        return M

    def interaction_triangles_world(self) -> np.ndarray:
        """All triangles merged — used for E-key ray-pick detection."""
        if self._collected or self._flashlight is None:
            return np.zeros((0, 3, 3), np.float64)
        M = self.pickup_transform()
        parts = list(self._flashlight.world_tris(M).values())
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            return np.zeros((0, 3, 3), np.float64)
        return np.concatenate(parts, axis=0).astype(np.float64)

    _SECTION_MAT = {
        "inner_reflector": "flashlight_reflector",
        "outer_shell":     "flashlight_shell",
        "handle":          "flashlight_shell",
        "back_cap":        "flashlight_shell",
        "front_rim":       "flashlight_shell",
        "front_glass":     "borosilicate_glass",
        "bulb":            "tungsten_filament",
    }

    def publish_geometry(self, resolve_mat_id) -> None:
        """Publish pickup geometry into the shader walker's geometry leftovers.

        resolve_mat_id: callable(name: str) -> int  — resolves material name to
        integer SSBO index.  Call once per frame from the scene-node publish loop.
        """
        from controls import get_shader_walker as _get_shader_walker
        _walker = _get_shader_walker()

        if self._flashlight is None:
            return

        if self._collected or not self._world_visible:
            # Clear stale geometry from the scene target exactly once.
            if self._published_world:
                _walker.publish_owner_target(
                    owner_id=self.obj_id,
                    target_id=f"scene.geometry/{self.obj_id}",
                    payload={
                        "owner_id": self.obj_id,
                        "kind": "triangles",
                        "triangles": np.zeros((0, 3, 3), np.float64),
                        "mat_ids": np.zeros((0,), np.int32),
                        "mat_id": 0,
                    },
                    flip_slots=2,
                    change_key=("clear", int(self._collected)),
                )
                self._published_world = False
            return

        M = self.pickup_transform()
        parts = self._flashlight.world_tris(M)
        tri_list, id_list = [], []
        for section, tris in parts.items():
            if len(tris) == 0:
                continue
            mat_id = resolve_mat_id(self._SECTION_MAT.get(section, "flashlight_shell"))
            tri_list.append(np.asarray(tris, dtype=np.float64))
            id_list.append(np.full((len(tris),), mat_id, dtype=np.int32))
        if not tri_list:
            return
        triangles = np.concatenate(tri_list, axis=0)
        mat_ids   = np.concatenate(id_list,  axis=0)
        cull_immune = np.ones((int(triangles.shape[0]),), dtype=np.uint8)
        _walker.publish_owner_target(
            owner_id=self.obj_id,
            target_id=f"scene.geometry/{self.obj_id}",
            payload={
                "owner_id": self.obj_id,
                "kind":     "triangles",
                "triangles": triangles,
                "mat_ids":   mat_ids,
                "mat_id":    int(id_list[0][0]),
                "cull_immune": True,
                "cull_immune_tri_mask": cull_immune,
            },
            flip_slots=2,
            change_key=(
                bool(self._collected),
                float(self.pos[0]), float(self.pos[1]), float(self.pos[2]),
                int(triangles.shape[0]),
            ),
        )
        self._published_world = True


def publish_equipped_flashlight_geometry(player_ctrl, resolve_mat_id) -> None:
    """Publish the equipped flashlight mesh in player hands.

    Uses the exact same flashlight object held in the active hotbar slot and
    places it via PlayerController.flashlight_transform().
    """
    if player_ctrl is None:
        return
    from controls import get_shader_walker as _get_shader_walker

    _walker = _get_shader_walker()
    _held_owner = "player_flashlight_held"

    def _clear_held_once() -> None:
        if not bool(getattr(player_ctrl, "_held_flashlight_published", False)):
            return
        _walker.publish_owner_target(
            owner_id=_held_owner,
            target_id=f"scene.geometry/{_held_owner}",
            payload={
                "owner_id": _held_owner,
                "kind": "triangles",
                "triangles": np.zeros((0, 3, 3), np.float64),
                "mat_ids": np.zeros((0,), np.int32),
                "mat_id": 0,
            },
            flip_slots=2,
            change_key=("clear", int(getattr(player_ctrl, "_tool_index", -1))),
        )
        setattr(player_ctrl, "_held_flashlight_published", False)

    state = getattr(player_ctrl, "state", None)
    if getattr(state, "name", "") != "WALK":
        _clear_held_once()
        return
    tool = getattr(player_ctrl, "equipped_tool", None)
    if tool is None or str(getattr(tool, "tool_type", "")) != "flashlight":
        _clear_held_once()
        return
    flashlight = getattr(tool, "flashlight", None)
    if flashlight is None:
        _clear_held_once()
        return
    xf_fn = getattr(player_ctrl, "flashlight_transform", None)
    if not callable(xf_fn):
        _clear_held_once()
        return

    M = np.asarray(xf_fn(), dtype=np.float32)
    parts = flashlight.world_tris(M)
    tri_list, id_list = [], []
    section_mat = FlashlightPickup._SECTION_MAT
    for section, tris in parts.items():
        if len(tris) == 0:
            continue
        mat_id = int(resolve_mat_id(section_mat.get(section, "flashlight_shell")))
        tri_list.append(np.asarray(tris, dtype=np.float64))
        id_list.append(np.full((len(tris),), mat_id, dtype=np.int32))
    if not tri_list:
        return

    triangles = np.concatenate(tri_list, axis=0)
    mat_ids = np.concatenate(id_list, axis=0)
    cull_immune = np.ones((int(triangles.shape[0]),), dtype=np.uint8)
    _mk = tuple(np.round(M.reshape(-1), 4).tolist())
    _walker.publish_owner_target(
        owner_id=_held_owner,
        target_id=f"scene.geometry/{_held_owner}",
        payload={
            "owner_id": _held_owner,
            "kind": "triangles",
            "triangles": triangles,
            "mat_ids": mat_ids,
            "mat_id": int(mat_ids[0]) if len(mat_ids) else 0,
            "cull_immune": True,
            "cull_immune_tri_mask": cull_immune,
        },
        flip_slots=2,
        change_key=(
            int(getattr(player_ctrl, "_tool_index", -1)),
            bool(getattr(flashlight, "enabled", False)),
            round(float(getattr(flashlight, "bulb_z", 0.0)), 6),
            int(triangles.shape[0]),
            _mk,
        ),
    )
    setattr(player_ctrl, "_held_flashlight_published", True)
