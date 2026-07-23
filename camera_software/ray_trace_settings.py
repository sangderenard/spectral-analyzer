"""Shared, explicit ray-trace settings and manifest precedence.

The interactive toolbar supplies a user layer.  A scene may deliberately
override individual fields by publishing ``ray_trace_settings``; omitted
fields continue to inherit from the toolbar.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class RayTraceSettings:
    transport_mode: str = "continuous"
    lane_count: int = 1
    wave_mode: bool = False
    # Keep the requested impactful-ray target near 200k. The default 256-wide
    # work tile and 1,024 sample/node floor may schedule 262,144 physical
    # camera samples when every tile column is selected in one epoch.
    total_rays: int = 204_800
    max_sensor_epochs: int = 1
    epoch_bundle_count: int = 16
    sensor_samples_per_node: int = 1_024
    sensor_t5_pair_budget: int = 67_108_864
    max_bounces: int = 16
    grid_mode: str = "n-tree"
    subdivision_axis: int = 3
    locked_grid_columns: int = 1
    locked_grid_rows: int = 1

    def validated(self) -> "RayTraceSettings":
        mode = str(self.transport_mode).strip().lower()
        if mode not in {"fixed", "continuous", "depth"}:
            raise ValueError(f"unsupported transport mode {mode!r}")
        lanes = int(self.lane_count)
        allowed = (1, 3, 4, 8, 16, 32)
        if lanes not in allowed:
            raise ValueError(f"{mode} transport does not support {lanes} lanes")
        grid_mode = str(self.grid_mode).strip().lower()
        if grid_mode not in {"n-tree", "locked"}:
            raise ValueError("grid_mode must be n-tree or locked")
        if int(self.subdivision_axis) not in {2, 3}:
            raise ValueError("subdivision_axis must be 2 (even) or 3 (odd)")
        if not 1 <= int(self.locked_grid_columns) <= 64:
            raise ValueError("locked_grid_columns must be in [1, 64]")
        if not 1 <= int(self.locked_grid_rows) <= 64:
            raise ValueError("locked_grid_rows must be in [1, 64]")
        positive = (
            "total_rays", "max_sensor_epochs", "epoch_bundle_count",
            "sensor_samples_per_node",
            "sensor_t5_pair_budget", "max_bounces",
        )
        for field_name in positive:
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be positive")
        return replace(
            self,
            transport_mode=mode,
            lane_count=lanes,
            wave_mode=bool(self.wave_mode),
            grid_mode=grid_mode,
        )

    @property
    def transport_option(self) -> str:
        # Depth is an output/integrator mode; it retains continuous spectral
        # transport so switching products does not degrade the scene contract.
        mode = "continuous" if self.transport_mode == "depth" else self.transport_mode
        return f"{mode}:{self.lane_count}"

    def render_budget(self, *, sensor_top_k: int = 64) -> dict[str, int]:
        """Budget camera-impacting samples; flash traversal is support work."""
        epochs = max(1, int(self.max_sensor_epochs))
        top_k = max(1, int(sensor_top_k))
        rays_per_epoch = max(
            1, (int(self.total_rays) + epochs - 1) // epochs
        )
        requested_samples_per_node = max(
            int(self.sensor_samples_per_node),
            (rays_per_epoch + top_k - 1) // top_k,
        )
        # The native lineage/spectrum arena is deliberately fixed at one
        # million samples. Preserve larger logical epochs as several complete
        # mip selections; never split one mutable frontier selection into fake
        # top-k pages.
        record_lane_width = int(self.lane_count) if self.transport_mode == "fixed" else 1
        camera_primary_capacity = max(1, min(
            1_048_576,
            4_194_304 // max(1, int(self.max_bounces)),
            12_582_912 // max(
                1, int(self.max_bounces) * record_lane_width
            ),
        ))
        lineage_samples_per_step = max(1, camera_primary_capacity // top_k)
        epoch_split = max(
            1,
            (requested_samples_per_node + lineage_samples_per_step - 1)
            // lineage_samples_per_step,
        )
        samples_per_node = (
            requested_samples_per_node + epoch_split - 1
        ) // epoch_split
        effective_epochs = epochs * epoch_split
        # Flash paths are support work retained for the camera pages.  Sending
        # the entire high-end camera budget as one flash submission can create
        # tens of millions of BDPT records before T5 gets a chance to drain
        # them. Keep one bundle inside the native record arena; outer bundles
        # provide the additional independent light-path samples.
        # Bound one flash page by the smallest native vertex/PDF arena, not by
        # the larger spectral arena. A path may author one vertex and one PDF
        # per bounce even in continuous single-lane transport.
        flash_page_rays = max(
            1,
            min(1_048_576, 4_194_304 // max(
                1, int(self.max_bounces) * record_lane_width
            )),
        )
        flash_total_rays = max(
            1,
            (min(rays_per_epoch, 1_048_576) + epoch_split - 1) // epoch_split,
        )
        flash_page_count = max(
            1, (flash_total_rays + flash_page_rays - 1) // flash_page_rays
        )
        # A flash page is a complete independent exposure bundle.  Multiplying
        # the outer bundle count preserves every requested light-path sample
        # without requiring variable-size ray records or crossing partially
        # retained light and camera arenas inside T5.
        flash_rays_this_bundle = (
            flash_total_rays + flash_page_count - 1
        ) // flash_page_count
        t5_pairs_this_epoch = max(
            max(1, (rays_per_epoch + epoch_split - 1) // epoch_split),
            max(1, (
                int(self.sensor_t5_pair_budget) + epoch_split - 1
            ) // epoch_split),
        )
        return {
            "total_rays": int(self.total_rays),
            "rays_per_batch": int(max(
                1, (int(self.total_rays) + effective_epochs - 1)
                // effective_epochs,
            )),
            "max_sensor_epochs": effective_epochs,
            "epoch_bundle_count": int(self.epoch_bundle_count) * flash_page_count,
            "sensor_flash_page_count": flash_page_count,
            "sensor_flash_total_rays": flash_total_rays,
            "sensor_top_k": top_k,
            "sensor_samples_per_node": samples_per_node,
            "sensor_steps_per_layer": 1,
            # Light paths make the camera connections possible, but they do
            # not consume the user-facing impactful-ray count.
            "sensor_flash_rays": int(flash_rays_this_bundle),
            "sensor_t5_pair_budget": t5_pairs_this_epoch,
            "max_bounces": int(self.max_bounces),
        }

    def mapping(self) -> dict[str, Any]:
        return asdict(self.validated())


RAY_TRACE_DEFAULTS = RayTraceSettings()
_FIELDS = frozenset(asdict(RAY_TRACE_DEFAULTS))


def manifest_ray_trace_overrides(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only deliberate scene overrides, never inferred scene metadata."""

    if not manifest:
        return {}
    overrides: dict[str, Any] = {}
    for container in (manifest, manifest.get("scene", {})):
        if not isinstance(container, Mapping):
            continue
        authored = container.get("ray_trace_settings", {})
        if not isinstance(authored, Mapping):
            raise ValueError("ray_trace_settings must be a mapping")
        unknown = sorted(set(authored) - _FIELDS)
        if unknown:
            raise ValueError(f"unknown ray-trace settings: {unknown}")
        overrides.update(authored)
    return overrides


def resolve_ray_trace_settings(
    user: RayTraceSettings | Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> RayTraceSettings:
    """Resolve defaults < user toolbar < explicitly authored manifest values."""

    values = RAY_TRACE_DEFAULTS.mapping()
    if isinstance(user, RayTraceSettings):
        values.update(user.mapping())
    elif user is not None:
        unknown = sorted(set(user) - _FIELDS)
        if unknown:
            raise ValueError(f"unknown user ray-trace settings: {unknown}")
        values.update(user)
    values.update(manifest_ray_trace_overrides(manifest))
    return RayTraceSettings(**values).validated()
