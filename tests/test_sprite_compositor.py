import numpy as np
from scipy import ndimage

from camera_software import (
    AtlasCaptureSpec,
    CachedTokenStringComposer,
    DEFAULT_INK_CONDITION,
    DisplayProductKind,
    RayTracedSprite,
    RenderAssetCatalog,
    RenderedAssetRecord,
    extract_raytraced_sprite,
    ink_token_asset,
    measure_sprite_exposure_quality,
    token_alpha_mask,
)


def _synthetic_sprite(token: str, tmp_path):
    capture = AtlasCaptureSpec(
        width=64, height=64, content_width=42, content_height=42
    )
    asset = ink_token_asset(token, character=(len(token) == 1))
    alpha = token_alpha_mask(asset, capture)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, capture.height),
        np.linspace(-1.0, 1.0, capture.width),
        indexing="ij",
    )
    background = np.stack([
        0.03 + 0.006 * xx,
        0.04 + 0.004 * yy,
        0.05 + 0.003 * xx * yy,
    ], axis=-1)
    # Deliberately blue foreground proves extraction is not keyed to red ink.
    foreground = np.asarray([0.08, 0.22, 0.75])
    outer = ndimage.binary_dilation(alpha > 0.01, iterations=5)
    spill_band = outer & ~(alpha > 0.01)
    spill = np.zeros_like(background)
    spill[spill_band] = np.asarray([0.015, 0.025, 0.06])
    image = alpha[..., None] * foreground + (1.0 - alpha[..., None]) * background
    image += spill
    sprite = extract_raytraced_sprite(image, asset, capture)
    path = sprite.save(str(tmp_path / f"{token}.npz"))
    return asset, sprite, path, image


def test_sprite_layers_reconstruct_arbitrary_material_and_neighboring_light(tmp_path):
    _asset, sprite, path, image = _synthetic_sprite("A", tmp_path)
    restored = RayTracedSprite.load(path)
    reconstructed = restored.compose_over(restored.source_background_rgb)

    assert np.max(np.abs(reconstructed - image)) < 2.0e-3
    assert np.any(restored.additive_rgb[restored.alpha < 0.01] > 0.01)
    assert np.allclose(restored.alpha, sprite.alpha)

    green_destination = np.full_like(image, [0.02, 0.3, 0.04])
    composed = restored.compose_over(green_destination)
    body = restored.alpha > 0.9
    outside_spill = (
        (restored.alpha < 0.01)
        & (np.linalg.norm(restored.additive_rgb, axis=2) > 0.01)
    )
    assert np.mean(composed[body, 2]) > np.mean(green_destination[body, 2])
    assert np.any(composed[outside_spill] != green_destination[outside_spill])


def test_sprite_quality_uses_coverage_and_image_stability_not_a_ray_count():
    capture = AtlasCaptureSpec(
        width=64, height=64, content_width=42, content_height=42
    )
    asset = ink_token_asset("A")
    alpha = token_alpha_mask(asset, capture)
    image = np.ones((64, 64, 3), np.float32)
    partial = np.zeros((64, 64), np.float32)
    partial[:, :32] = 1.0

    covered = measure_sprite_exposure_quality(
        image, np.ones_like(partial), alpha, refinement_pass=1
    )
    incomplete = measure_sprite_exposure_quality(
        image, partial, alpha, refinement_pass=100
    )
    converged = measure_sprite_exposure_quality(
        image,
        np.ones_like(partial),
        alpha,
        refinement_pass=2,
        previous_image=image.copy(),
    )

    assert covered.composable
    assert not covered.converged
    assert not incomplete.composable
    assert incomplete.glyph_exposure_coverage < 0.95
    assert converged.composable
    assert converged.converged
    assert converged.relative_rmse == 0.0


def test_first_sparse_finite_evidence_is_live_usable_before_it_is_clear():
    capture = AtlasCaptureSpec(
        width=64, height=64, content_width=42, content_height=42
    )
    alpha = token_alpha_mask(ink_token_asset("A"), capture)
    image = np.zeros((64, 64, 3), np.float32)
    weight = np.zeros((64, 64), np.float32)
    first_glyph_pixel = tuple(np.argwhere(alpha > 0.05)[0])
    image[first_glyph_pixel] = (0.1, 0.02, 0.01)
    weight[first_glyph_pixel] = 1.0

    quality = measure_sprite_exposure_quality(
        image, weight, alpha, refinement_pass=1
    )

    assert quality.live_usable
    assert not quality.composable
    assert not quality.converged
    assert quality.as_metadata()["live_usable"] is True


def test_negative_design_tracking_does_not_collapse_small_ui_text_to_one_pixel(
    tmp_path,
):
    composer = CachedTokenStringComposer(
        RenderAssetCatalog(),
        horizontal_spacing_px=-10,
        vertical_spacing_px=-10,
    )

    line_height, character_advance, line_advance = (
        composer._monofont_layout_metrics(14)
    )

    assert line_height == 8
    assert character_advance >= 4
    assert line_advance >= 6

    asset, _sprite, path, _image = _synthetic_sprite("A", tmp_path)
    catalog = RenderAssetCatalog()
    catalog.record(RenderedAssetRecord(
        asset.asset_key,
        DEFAULT_INK_CONDITION.condition_key,
        DisplayProductKind.IMAGE,
        metadata={"sprite_path": path},
    ))
    rendered = CachedTokenStringComposer(
        catalog,
        horizontal_spacing_px=-10,
        vertical_spacing_px=-10,
    ).compose("A", 80, 14, character_tiles_only=True).linear_rgb
    changed_rows = np.flatnonzero(np.any(
        np.abs(rendered - np.asarray((0.002, 0.002, 0.002))) > 1.0e-6,
        axis=(1, 2),
    ))
    assert len(changed_rows) >= 4


def test_cached_string_composer_prefers_available_glyphs_and_reports_pressure(
    tmp_path,
):
    catalog = RenderAssetCatalog()
    for token in ("A", "B"):
        asset, _sprite, path, _image = _synthetic_sprite(token, tmp_path)
        catalog.record(RenderedAssetRecord(
            asset.asset_key,
            DEFAULT_INK_CONDITION.condition_key,
            DisplayProductKind.IMAGE,
            metadata={"sprite_path": path},
        ))

    composition = CachedTokenStringComposer(catalog).compose(
        "AB C", 180, 64
    )

    assert composition.used_tokens == ("A", "B")
    assert composition.missing_tokens == ("AB",)
    assert composition.missing_characters == ("C",)
    assert composition.linear_rgb.shape == (64, 180, 3)
    assert float(np.max(composition.linear_rgb)) > 0.05


def test_cached_string_composer_accepts_empty_and_whitespace_editor_states():
    composer = CachedTokenStringComposer(RenderAssetCatalog())

    empty = composer.compose("", 80, 32)
    whitespace = composer.compose(" \n\t", 80, 32)

    for composition in (empty, whitespace):
        assert composition.used_tokens == ()
        assert composition.missing_tokens == ()
        assert composition.missing_characters == ()
        assert composition.linear_rgb.shape == (32, 80, 3)


def test_cached_string_composer_clips_sprite_when_fitted_panel_is_too_narrow(
    tmp_path,
):
    catalog = RenderAssetCatalog()
    asset, _sprite, path, _image = _synthetic_sprite("A", tmp_path)
    catalog.record(RenderedAssetRecord(
        asset.asset_key,
        DEFAULT_INK_CONDITION.condition_key,
        DisplayProductKind.IMAGE,
        metadata={"sprite_path": path},
    ))

    composition = CachedTokenStringComposer(catalog).compose(
        "A", 1, 80, character_tiles_only=True
    )

    assert composition.linear_rgb.shape == (80, 1, 3)
    assert np.all(np.isfinite(composition.linear_rgb))


def test_monofont_spacing_defaults_to_negative_ten_and_allows_cropping(tmp_path):
    default = CachedTokenStringComposer(RenderAssetCatalog())
    expanded = CachedTokenStringComposer(
        RenderAssetCatalog(),
        horizontal_spacing_px=3,
        vertical_spacing_px=4,
    )
    cropped = CachedTokenStringComposer(
        RenderAssetCatalog(),
        horizontal_spacing_px=-1000,
        vertical_spacing_px=-1000,
    )

    glyph_height, default_x, default_y = default._monofont_layout_metrics(64)
    _, expanded_x, expanded_y = expanded._monofont_layout_metrics(64)
    _, cropped_x, cropped_y = cropped._monofont_layout_metrics(64)

    assert default.horizontal_spacing_px == -10
    assert default.vertical_spacing_px == -10
    scale = glyph_height / 48.0
    assert expanded_x == default_x + (
        round(3 * scale) - round(-10 * scale)
    )
    assert expanded_y == default_y + (
        round(4 * scale) - round(-10 * scale)
    )
    assert cropped_x >= glyph_height // 3
    assert cropped_y >= round(glyph_height * 0.72)
    assert glyph_height > cropped_y

    catalog = RenderAssetCatalog()
    asset, _sprite, path, _image = _synthetic_sprite("A", tmp_path)
    catalog.record(RenderedAssetRecord(
        asset.asset_key,
        DEFAULT_INK_CONDITION.condition_key,
        DisplayProductKind.IMAGE,
        metadata={"sprite_path": path},
    ))
    clipped = CachedTokenStringComposer(
        catalog,
        horizontal_spacing_px=-1000,
        vertical_spacing_px=-1000,
    ).compose("A", 64, 64, character_tiles_only=True)

    assert clipped.used_tokens == ("A",)
    assert np.all(np.isfinite(clipped.linear_rgb))


def test_cached_string_composer_prefers_exact_token_over_character_fallback(
    tmp_path,
):
    catalog = RenderAssetCatalog()
    for token in ("A", "B", "AB"):
        asset, _sprite, path, _image = _synthetic_sprite(token, tmp_path)
        catalog.record(RenderedAssetRecord(
            asset.asset_key,
            DEFAULT_INK_CONDITION.condition_key,
            DisplayProductKind.IMAGE,
            metadata={"sprite_path": path},
        ))

    composition = CachedTokenStringComposer(catalog).compose("AB", 180, 64)

    assert composition.used_tokens == ("AB",)
    assert composition.missing_tokens == ()
    assert composition.missing_characters == ()


def test_cached_gestalt_is_hidden_until_its_individual_glyphs_exist(tmp_path):
    catalog = RenderAssetCatalog()
    asset, _sprite, path, _image = _synthetic_sprite("AB", tmp_path)
    catalog.record(RenderedAssetRecord(
        asset.asset_key,
        DEFAULT_INK_CONDITION.condition_key,
        DisplayProductKind.IMAGE,
        metadata={"sprite_path": path},
    ))

    composition = CachedTokenStringComposer(catalog).compose("AB", 180, 64)

    assert composition.used_tokens == ()
    assert composition.missing_tokens == ("AB",)
    assert composition.missing_characters == ("A", "B")


def test_cached_string_composer_can_hold_fixed_character_tiles(tmp_path):
    catalog = RenderAssetCatalog()
    for token in ("A", "B", "AB"):
        asset, _sprite, path, _image = _synthetic_sprite(token, tmp_path)
        catalog.record(RenderedAssetRecord(
            asset.asset_key,
            DEFAULT_INK_CONDITION.condition_key,
            DisplayProductKind.IMAGE,
            metadata={"sprite_path": path},
        ))

    composition = CachedTokenStringComposer(catalog).compose(
        "AB", 180, 64, character_tiles_only=True
    )

    assert composition.used_tokens == ("A", "B")
