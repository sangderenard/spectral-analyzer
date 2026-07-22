"""Background-independent sprites derived from padded ray-traced captures.

The artifact uses two layers because alpha alone cannot represent light that a
glossy object spills onto neighboring pixels:

    output = premultiplied_rgb + (1 - alpha) * destination + additive_rgb

``additive_rgb`` is signed, so it can preserve highlights, colored spill, and
shadows. Alpha is derived from the authored font geometry rather than guessed
from foreground/background color.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import ndimage

from .display_scene import DisplayProductKind
from .render_assets import (
    AtlasCaptureSpec,
    DEFAULT_INK_CONDITION,
    MIN_INK_GLYPH_EXPOSURE_COVERAGE,
    MIN_INK_GLYPH_RADIANCE_COVERAGE,
    LIVE_INK_GLYPH_EXPOSURE_COVERAGE,
    LIVE_INK_GLYPH_RADIANCE_COVERAGE,
    INK_ATLAS_CONVERGENCE_EXPOSURE_COVERAGE,
    INK_ATLAS_CONVERGENCE_RELATIVE_RMSE,
    INK_ATLAS_CONVERGENCE_P95_DELTA,
    INK_ATLAS_CONVERGENCE_HOLD,
    ExtrudedTokenAsset,
    FontAssetSpec,
    RenderAssetCatalog,
    RenderedAssetRecord,
    ink_record_is_converged,
    ink_record_is_usable,
    ink_token_asset,
)


SPRITE_SCHEMA_VERSION = 1
DEFAULT_MONOFONT_HORIZONTAL_SPACING_PX = -10
DEFAULT_MONOFONT_VERTICAL_SPACING_PX = -10


def _inside_even_odd(points: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    for raw in contours:
        polygon = np.asarray(raw, np.float64)
        if polygon.shape[0] < 3:
            continue
        if not np.allclose(polygon[0], polygon[-1]):
            polygon = np.vstack([polygon, polygon[0]])
        hit = np.zeros(points.shape[0], dtype=bool)
        for a, b in zip(polygon[:-1], polygon[1:]):
            crosses = (a[1] > y) != (b[1] > y)
            x_at_y = (
                (b[0] - a[0]) * (y - a[1])
                / (b[1] - a[1] + 1.0e-300)
                + a[0]
            )
            hit ^= crosses & (x < x_at_y)
        inside ^= hit
    return inside


def token_alpha_mask(
    asset: ExtrudedTokenAsset,
    capture: AtlasCaptureSpec,
    *,
    supersample: int = 3,
) -> np.ndarray:
    """Rasterize authored token geometry into its padded capture frame."""

    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    kwargs: dict[str, Any] = {
        "family": asset.font.family,
        "weight": asset.font.weight,
        "style": asset.font.style,
    }
    if asset.font.file:
        kwargs["fname"] = asset.font.file
    path = TextPath(
        (0.0, 0.0),
        asset.token,
        size=1.0,
        prop=FontProperties(**kwargs),
    )
    contours = [
        np.asarray(polygon, np.float64)
        for polygon in path.to_polygons(closed_only=True)
        if len(polygon) >= 3
    ]
    if not contours:
        raise ValueError(f"font produced no mask for {asset.token!r}")
    points = np.concatenate(contours, axis=0)
    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = np.maximum(hi - lo, 1.0e-12)
    scale_x = 0.88 * capture.content_width / extent[0]
    scale_y = 0.88 * capture.content_height / extent[1]
    # Match the circular extrusion's widest intermediate ring. Side-wall
    # transport outside this geometric body remains in the signed spill layer.
    silhouette_scale = 1.0 + max(0.0, asset.extrusion.profile_bulge)
    scale = min(scale_x, scale_y) * silhouette_scale
    center = 0.5 * (lo + hi)
    cx = 0.5 * capture.width
    cy = 0.5 * capture.height
    mapped = []
    for contour in contours:
        transformed = np.empty_like(contour)
        transformed[:, 0] = cx + (contour[:, 0] - center[0]) * scale
        transformed[:, 1] = cy - (contour[:, 1] - center[1]) * scale
        mapped.append(transformed)

    ss = max(1, int(supersample))
    yy, xx = np.meshgrid(
        (np.arange(capture.height * ss) + 0.5) / ss,
        (np.arange(capture.width * ss) + 0.5) / ss,
        indexing="ij",
    )
    samples = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    high = _inside_even_odd(samples, mapped).reshape(
        capture.height * ss, capture.width * ss
    )
    alpha = high.reshape(
        capture.height, ss, capture.width, ss
    ).mean(axis=(1, 3))
    return np.ascontiguousarray(alpha, dtype=np.float32)


def _fit_background(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Fit a smooth slate field outside a generously dilated glyph."""

    height, width = alpha.shape
    dilation = max(4, int(round(min(height, width) * 0.06)))
    excluded = ndimage.binary_dilation(alpha > 0.005, iterations=dilation)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    design = np.stack(
        [np.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy], axis=-1
    )
    keep = ~excluded
    if np.count_nonzero(keep) < 12:
        median = np.median(image.reshape(-1, 3), axis=0)
        return np.broadcast_to(median, image.shape).copy()
    model = np.empty_like(image, dtype=np.float64)
    x_fit = design[keep]
    for channel in range(3):
        coefficients, *_ = np.linalg.lstsq(
            x_fit, image[..., channel][keep], rcond=None
        )
        model[..., channel] = design @ coefficients
    return np.maximum(model, 0.0)


@dataclass(frozen=True)
class SpriteExposureQuality:
    refinement_pass: int
    glyph_exposure_coverage: float
    glyph_radiance_coverage: float
    sensor_exposure_coverage: float
    glyph_mean_weight: float
    relative_rmse: float | None
    p95_relative_delta: float | None
    stable_hold: int
    finite: bool
    live_usable: bool
    composable: bool
    converged: bool

    def as_metadata(self) -> dict[str, Any]:
        return {
            "refinement_pass": self.refinement_pass,
            "glyph_exposure_coverage": self.glyph_exposure_coverage,
            "glyph_radiance_coverage": self.glyph_radiance_coverage,
            "sensor_exposure_coverage": self.sensor_exposure_coverage,
            "glyph_mean_weight": self.glyph_mean_weight,
            "relative_rmse": self.relative_rmse,
            "p95_relative_delta": self.p95_relative_delta,
            "stable_hold": self.stable_hold,
            "finite": self.finite,
            "live_usable": self.live_usable,
            "composable": self.composable,
            "converged": self.converged,
        }


def measure_sprite_exposure_quality(
    linear_image: np.ndarray,
    exposure_weight: np.ndarray,
    alpha: np.ndarray,
    *,
    refinement_pass: int,
    previous_image: np.ndarray | None = None,
    previous_stable_hold: int = 0,
) -> SpriteExposureQuality:
    image = np.asarray(linear_image, np.float64)
    weight = np.asarray(exposure_weight, np.float64)
    matte = np.asarray(alpha)
    if image.shape[:2] != weight.shape or weight.shape != matte.shape:
        raise ValueError("quality inputs must share capture dimensions")
    glyph = matte > 0.05
    if not np.any(glyph):
        raise ValueError("quality measurement requires a visible glyph matte")
    glyph_coverage = float(np.mean(weight[glyph] > 0.0))
    sensor_coverage = float(np.mean(weight > 0.0))
    mean_weight = float(np.mean(weight[glyph]))
    glyph_luminance = np.max(np.maximum(image[glyph], 0.0), axis=1)
    radiance_scale = float(np.percentile(glyph_luminance, 95.0))
    radiance_floor = max(radiance_scale * 1.0e-4, 1.0e-12)
    glyph_radiance_coverage = float(np.mean(glyph_luminance > radiance_floor))
    finite = bool(
        np.all(np.isfinite(image))
        and np.all(np.isfinite(weight))
        and np.all(weight >= 0.0)
    )
    live_usable = bool(
        finite
        and mean_weight > 0.0
        and glyph_coverage >= LIVE_INK_GLYPH_EXPOSURE_COVERAGE
        and glyph_radiance_coverage >= LIVE_INK_GLYPH_RADIANCE_COVERAGE
    )
    composable = bool(
        finite
        and glyph_coverage >= MIN_INK_GLYPH_EXPOSURE_COVERAGE
        and glyph_radiance_coverage >= MIN_INK_GLYPH_RADIANCE_COVERAGE
    )
    relative_rmse = None
    p95_relative_delta = None
    if previous_image is not None:
        previous = np.asarray(previous_image, np.float64)
        if previous.shape != image.shape:
            raise ValueError("previous image must match the current capture")
        # Include the body and a generous halo so neighboring light and
        # shadows must stabilize too, without letting untouched slate dominate.
        region = ndimage.binary_dilation(glyph, iterations=8)
        current_region = image[region]
        previous_region = previous[region]
        scale = max(
            float(np.percentile(np.linalg.norm(current_region, axis=1), 95.0)),
            float(np.percentile(np.linalg.norm(previous_region, axis=1), 95.0)),
            1.0e-12,
        )
        delta_norm = np.linalg.norm(current_region - previous_region, axis=1)
        relative_rmse = float(
            np.sqrt(np.mean(np.square(current_region - previous_region))) / scale
        )
        p95_relative_delta = float(np.percentile(delta_norm, 95.0) / scale)
    stable_now = bool(
        finite
        and glyph_coverage >= INK_ATLAS_CONVERGENCE_EXPOSURE_COVERAGE
        and relative_rmse is not None
        and p95_relative_delta is not None
        and relative_rmse <= INK_ATLAS_CONVERGENCE_RELATIVE_RMSE
        and p95_relative_delta <= INK_ATLAS_CONVERGENCE_P95_DELTA
    )
    stable_hold = int(previous_stable_hold) + 1 if stable_now else 0
    converged = bool(
        composable and stable_hold >= INK_ATLAS_CONVERGENCE_HOLD
    )
    return SpriteExposureQuality(
        refinement_pass=int(refinement_pass),
        glyph_exposure_coverage=glyph_coverage,
        glyph_radiance_coverage=glyph_radiance_coverage,
        sensor_exposure_coverage=sensor_coverage,
        glyph_mean_weight=mean_weight,
        relative_rmse=relative_rmse,
        p95_relative_delta=p95_relative_delta,
        stable_hold=stable_hold,
        finite=finite,
        live_usable=live_usable,
        composable=composable,
        converged=converged,
    )


@dataclass(frozen=True)
class RayTracedSprite:
    token: str
    asset_key: str
    premultiplied_rgb: np.ndarray
    alpha: np.ndarray
    additive_rgb: np.ndarray
    source_background_rgb: np.ndarray
    content_region: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        shape = np.asarray(self.alpha).shape
        for value in (
            self.premultiplied_rgb,
            self.additive_rgb,
            self.source_background_rgb,
        ):
            if np.asarray(value).shape != (*shape, 3):
                raise ValueError("sprite RGB layers must match alpha dimensions")

    def compose_over(self, destination: np.ndarray) -> np.ndarray:
        dst = np.asarray(destination, np.float64)
        if dst.shape != self.premultiplied_rgb.shape:
            raise ValueError("destination must match the sprite capture")
        return (
            np.asarray(self.premultiplied_rgb, np.float64)
            + (1.0 - np.asarray(self.alpha, np.float64)[..., None]) * dst
            + np.asarray(self.additive_rgb, np.float64)
        )

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray([SPRITE_SCHEMA_VERSION], np.int32),
            token=np.asarray([self.token]),
            asset_key=np.asarray([self.asset_key]),
            premultiplied_rgb=np.asarray(self.premultiplied_rgb, np.float32),
            alpha=np.asarray(self.alpha, np.float32),
            additive_rgb=np.asarray(self.additive_rgb, np.float32),
            source_background_rgb=np.asarray(
                self.source_background_rgb, np.float32
            ),
            content_region=np.asarray(self.content_region, np.int32),
        )
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> "RayTracedSprite":
        with np.load(path, allow_pickle=False) as raw:
            if int(raw["schema_version"][0]) != SPRITE_SCHEMA_VERSION:
                raise ValueError("unsupported ray-traced sprite schema")
            return cls(
                token=str(raw["token"][0]),
                asset_key=str(raw["asset_key"][0]),
                premultiplied_rgb=np.asarray(
                    raw["premultiplied_rgb"], np.float32
                ),
                alpha=np.asarray(raw["alpha"], np.float32),
                additive_rgb=np.asarray(raw["additive_rgb"], np.float32),
                source_background_rgb=np.asarray(
                    raw["source_background_rgb"], np.float32
                ),
                content_region=tuple(
                    int(value) for value in raw["content_region"]
                ),
            )


def extract_raytraced_sprite(
    linear_image: np.ndarray,
    asset: ExtrudedTokenAsset,
    capture: AtlasCaptureSpec,
) -> RayTracedSprite:
    """Separate a ray-traced frame into body alpha and neighboring light."""

    image = np.asarray(linear_image, np.float64)
    if image.shape != (capture.height, capture.width, 3):
        raise ValueError("linear atlas frame does not match its capture contract")
    alpha = token_alpha_mask(asset, capture)
    background = _fit_background(image, alpha)
    premultiplied = alpha[..., None] * image
    additive = image - premultiplied - (1.0 - alpha[..., None]) * background

    # Keep the signed residual lossless. Monte Carlo denoising may be applied
    # to this layer later, but thresholding here would destroy exact
    # reconstruction and can erase faint material-independent spill.
    return RayTracedSprite(
        token=asset.token,
        asset_key=asset.asset_key,
        premultiplied_rgb=np.ascontiguousarray(premultiplied, np.float32),
        alpha=alpha,
        additive_rgb=np.ascontiguousarray(additive, np.float32),
        source_background_rgb=np.ascontiguousarray(background, np.float32),
        content_region=capture.content_region,
    )


@dataclass(frozen=True)
class CachedStringComposition:
    linear_rgb: np.ndarray
    used_tokens: tuple[str, ...]
    missing_tokens: tuple[str, ...]
    missing_characters: tuple[str, ...]


class CachedTokenStringComposer:
    """Compose exact tokens when present and fall back to cached glyph sprites."""

    def __init__(
        self,
        catalog: RenderAssetCatalog,
        *,
        font: FontAssetSpec | None = None,
        horizontal_spacing_px: int = DEFAULT_MONOFONT_HORIZONTAL_SPACING_PX,
        vertical_spacing_px: int = DEFAULT_MONOFONT_VERTICAL_SPACING_PX,
    ) -> None:
        self.catalog = catalog
        self.font = font or FontAssetSpec()
        self.horizontal_spacing_px = int(horizontal_spacing_px)
        self.vertical_spacing_px = int(vertical_spacing_px)

    def _monofont_layout_metrics(
        self, height: int, font_scale: float = 1.0
    ) -> tuple[int, int, int]:
        """Return glyph height, character advance, and line advance in pixels."""

        base_height = max(8, int(round(min(48.0, height * 0.42))))
        glyph_line_height = max(
            1,
            min(int(height), int(round(base_height * max(0.1, float(font_scale))))),
        )
        worst_case_width = max(3, int(round(glyph_line_height * 0.72)))
        # Tracking controls are authored at the 48 px atlas design size. A
        # literal -10 px adjustment at an 8 px UI label used to collapse both
        # advances to one pixel and clip an otherwise valid rendered glyph to
        # a single row. Scale the authored tracking with the resolved size.
        spacing_scale = glyph_line_height / 48.0
        horizontal_spacing = int(round(
            self.horizontal_spacing_px * spacing_scale
        ))
        vertical_spacing = int(round(
            self.vertical_spacing_px * spacing_scale
        ))
        character_advance = max(
            max(2, glyph_line_height // 3),
            worst_case_width + horizontal_spacing,
        )
        line_advance = max(
            max(2, int(round(glyph_line_height * 0.72))),
            glyph_line_height + vertical_spacing,
        )
        return glyph_line_height, character_advance, line_advance

    def _record(
        self,
        token: str,
        *,
        character: bool,
        require_converged: bool = False,
    ) -> RenderedAssetRecord | None:
        if not str(token).strip():
            return None
        asset = ink_token_asset(token, font=self.font, character=character)
        record = self.catalog.find(
            asset.asset_key, DEFAULT_INK_CONDITION, DisplayProductKind.IMAGE
        )
        if record is None:
            return None
        if not ink_record_is_usable(record):
            return None
        if require_converged and not ink_record_is_converged(record):
            return None
        sprite_path = str(record.metadata.get("sprite_path", ""))
        if (
            (not sprite_path or not os.path.isfile(sprite_path))
            and record.linear_path
            and os.path.isfile(record.linear_path)
        ):
            raw_capture = dict(record.metadata.get("capture", {}))
            capture = AtlasCaptureSpec(
                width=int(raw_capture.get("width", record.width or 128)),
                height=int(raw_capture.get("height", record.height or 128)),
                content_width=int(
                    raw_capture.get("content_region", [24, 24, 80, 80])[2]
                ),
                content_height=int(
                    raw_capture.get("content_region", [24, 24, 80, 80])[3]
                ),
            )
            sprite = extract_raytraced_sprite(
                np.asarray(
                    np.load(record.linear_path, allow_pickle=False),
                    np.float32,
                )[..., :3],
                asset,
                capture,
            )
            sprite_path = sprite.save(
                os.path.join(
                    os.path.dirname(os.path.abspath(record.linear_path)),
                    "raytraced_sprite.npz",
                )
            )
            record = replace(
                record,
                metadata={**dict(record.metadata), "sprite_path": sprite_path},
            )
            self.catalog.record(record)
        return record if sprite_path and os.path.isfile(sprite_path) else None

    def _pieces(
        self,
        text: str,
        *,
        character_tiles_only: bool = False,
    ) -> tuple[list[tuple[str, RenderedAssetRecord | None]], set[str], set[str]]:
        if not text:
            return [], set(), set()
        visible_characters = tuple(dict.fromkeys(
            character for character in text if not character.isspace()
        ))
        individual_glyphs_ready = all(
            self._record(
                character,
                character=True,
                require_converged=True,
            ) is not None
            for character in visible_characters
        )
        if (
            not character_tiles_only
            and text.strip()
            and individual_glyphs_ready
        ):
            exact = self._record(text, character=(len(text) == 1))
            if exact is not None:
                return [(text, exact)], set(), set()
        pieces: list[tuple[str, RenderedAssetRecord | None]] = []
        missing_tokens: set[str] = set()
        missing_characters: set[str] = set()
        for segment in re.findall(r"\s+|\S+", text):
            if segment.isspace():
                pieces.append((segment, None))
                continue
            segment_glyphs_ready = all(
                self._record(
                    character,
                    character=True,
                    require_converged=True,
                ) is not None
                for character in dict.fromkeys(segment)
            )
            token_record = (
                self._record(segment, character=(len(segment) == 1))
                if segment_glyphs_ready and not character_tiles_only
                else None
            )
            if token_record is not None:
                pieces.append((segment, token_record))
                continue
            if len(segment) > 1:
                missing_tokens.add(segment)
            for character in segment:
                record = self._record(character, character=True)
                if record is None:
                    missing_characters.add(character)
                pieces.append((character, record))
        return pieces, missing_tokens, missing_characters

    def compose(
        self,
        text: str,
        width: int,
        height: int,
        *,
        background_rgb: tuple[float, float, float] = (0.002, 0.002, 0.002),
        character_tiles_only: bool = False,
        font_scale: float = 1.0,
    ) -> CachedStringComposition:
        if width <= 0 or height <= 0:
            raise ValueError("composition dimensions must be positive")
        canvas = np.broadcast_to(
            np.asarray(background_rgb, np.float64), (height, width, 3)
        ).copy()
        pieces, missing_tokens, missing_characters = self._pieces(
            str(text),
            character_tiles_only=character_tiles_only,
        )
        used: list[str] = []
        line_height, cell_advance, line_advance = (
            self._monofont_layout_metrics(height, font_scale)
        )
        x = max(2, line_height // 6)
        y = max(1, line_height // 5)
        line_start = x
        for token, record in pieces:
            if token.isspace():
                for character in token:
                    if character == "\n":
                        x = line_start
                        y += line_advance
                    else:
                        if x + cell_advance > width and x > line_start:
                            x = line_start
                            y += line_advance
                        x += cell_advance
                continue
            cell_span = max(1, len(token)) * cell_advance
            # Reserve and wrap the tile span before looking for image evidence.
            # Missing/developing glyphs therefore occupy exactly the same cells
            # they will use after a later catalog refresh.
            if x + cell_span > width and x > line_start:
                x = line_start
                y += line_advance
            if record is None:
                x += cell_span
                continue
            sprite = RayTracedSprite.load(str(record.metadata["sprite_path"]))
            alpha_points = np.argwhere(sprite.alpha > 0.01)
            if not alpha_points.size:
                x += cell_span
                continue
            y0, x0 = alpha_points.min(axis=0)
            y1, x1 = alpha_points.max(axis=0) + 1
            pad = max(2, int(round(0.08 * max(y1 - y0, x1 - x0))))
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1 = min(sprite.alpha.shape[1], x1 + pad)
            y1 = min(sprite.alpha.shape[0], y1 + pad)
            target_height = max(1, int(round(line_height * 0.78)))
            scale = target_height / max(1, y1 - y0)
            target_width = max(1, int(round((x1 - x0) * scale)))
            visible_width = min(target_width, cell_span)
            visible_height = min(target_height, line_advance)
            if y + visible_height > height:
                break
            source_x = max(0, (target_width - visible_width) // 2)
            source_y = max(0, (target_height - visible_height) // 2)
            draw_x = x + max(0, (cell_span - visible_width) // 2)
            zoom = (target_height / (y1 - y0), target_width / (x1 - x0))
            premul = ndimage.zoom(
                sprite.premultiplied_rgb[y0:y1, x0:x1],
                (*zoom, 1.0),
                order=1,
            )[:target_height, :target_width][
                source_y:source_y + visible_height,
                source_x:source_x + visible_width,
            ]
            alpha = ndimage.zoom(
                sprite.alpha[y0:y1, x0:x1], zoom, order=1
            )[:target_height, :target_width][
                source_y:source_y + visible_height,
                source_x:source_x + visible_width,
            ]
            additive = ndimage.zoom(
                sprite.additive_rgb[y0:y1, x0:x1],
                (*zoom, 1.0),
                order=1,
            )[:target_height, :target_width][
                source_y:source_y + visible_height,
                source_x:source_x + visible_width,
            ]
            # Clip source and destination as one rectangle. A very narrow
            # fitted UI can place the centered sprite wholly beyond an edge;
            # slicing only the canvas then produces a zero-width destination
            # against an unclipped alpha layer.
            dst_x0 = max(0, draw_x)
            dst_y0 = max(0, y)
            dst_x1 = min(width, draw_x + visible_width)
            dst_y1 = min(height, y + visible_height)
            if dst_x1 > dst_x0 and dst_y1 > dst_y0:
                src_x0 = dst_x0 - draw_x
                src_y0 = dst_y0 - y
                src_x1 = src_x0 + (dst_x1 - dst_x0)
                src_y1 = src_y0 + (dst_y1 - dst_y0)
                clipped_premul = premul[src_y0:src_y1, src_x0:src_x1]
                clipped_alpha = alpha[src_y0:src_y1, src_x0:src_x1]
                clipped_additive = additive[src_y0:src_y1, src_x0:src_x1]
                destination = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
                canvas[dst_y0:dst_y1, dst_x0:dst_x1] = (
                    clipped_premul
                    + (1.0 - clipped_alpha[..., None]) * destination
                    + clipped_additive
                )
                used.append(token)
            x += cell_span
        return CachedStringComposition(
            linear_rgb=np.ascontiguousarray(canvas, np.float32),
            used_tokens=tuple(used),
            missing_tokens=tuple(sorted(missing_tokens)),
            missing_characters=tuple(sorted(missing_characters)),
        )


__all__ = [
    "SPRITE_SCHEMA_VERSION",
    "DEFAULT_MONOFONT_HORIZONTAL_SPACING_PX",
    "DEFAULT_MONOFONT_VERTICAL_SPACING_PX",
    "token_alpha_mask",
    "SpriteExposureQuality",
    "measure_sprite_exposure_quality",
    "RayTracedSprite",
    "extract_raytraced_sprite",
    "CachedStringComposition",
    "CachedTokenStringComposer",
]
