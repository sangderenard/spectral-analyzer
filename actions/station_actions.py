from __future__ import annotations

from typing import Any


def on_tile_object_action(
    *,
    action_key: str,
    control_action: str,
    metadata: dict[str, Any],
    **context: Any,
) -> dict[str, Any]:
    """Default station object action hook used by room-control presets."""
    return {
        "handled": True,
        "action_key": action_key,
        "control_action": control_action,
        "metadata": dict(metadata or {}),
        "context": dict(context),
    }
