import numpy as np

from camera_software import (
    ColorScienceProfile,
    process_linear_sensor_image,
    save_linear_sensor_image,
)


def _profile(**kwargs):
    return ColorScienceProfile.spectral_sensor_srgb(
        tone_curve_mode="linear",
        **kwargs,
    )


def test_fixed_reference_does_not_renormalize_for_hot_pixels():
    raw = np.full((2, 2, 3), 0.18, dtype=np.float32)
    with_hot_pixel = raw.copy()
    with_hot_pixel[0, 0] = 100.0

    baseline = process_linear_sensor_image(raw, _profile())
    changed = process_linear_sensor_image(with_hot_pixel, _profile())

    np.testing.assert_array_equal(changed[1, 1], baseline[1, 1])
    assert changed[0, 0, 0] > baseline[0, 0, 0]


def test_exposure_and_white_balance_are_explicit_camera_controls():
    raw = np.full((1, 1, 3), 0.01, dtype=np.float64)
    baseline = process_linear_sensor_image(raw, _profile())
    adjusted = process_linear_sensor_image(
        raw,
        _profile(
            exposure_compensation_ev=1.0,
            white_balance=np.asarray([2.0, 1.0, 0.5]),
        ),
    )

    assert adjusted[0, 0, 0] > adjusted[0, 0, 1] > adjusted[0, 0, 2]
    assert np.all(adjusted > baseline * 0.9)


def test_processing_preserves_float_dtype_and_raw_values():
    raw = np.asarray([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    original = raw.copy()

    output = process_linear_sensor_image(
        raw,
        _profile(sensor_white_level=0.5, exposure_compensation_ev=-1.0),
    )

    assert output.dtype == raw.dtype
    np.testing.assert_array_equal(raw, original)


def test_camera_package_writes_display_png_without_changing_raw(tmp_path):
    raw = np.full((3, 4, 3), 0.25, dtype=np.float32)
    original = raw.copy()
    path = tmp_path / "camera.png"

    output = save_linear_sensor_image(raw, str(path), _profile())

    assert path.is_file()
    assert path.stat().st_size > 0
    assert output.shape == raw.shape
    np.testing.assert_array_equal(raw, original)
