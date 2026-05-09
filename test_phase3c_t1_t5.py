#!/usr/bin/env python3
"""
Integration test for Phase 3C T1-T5 implementation.

Validates:
- T2: ExposureSession loads SensorFilmDatabase and default slot metadata
- T3: _bind_sensor_film_ssbo method exists and has correct signature
- T4: _make_sensor_integral produces valid SensorIntegralObject with noise model
- T5: Metadata includes explicit noise terms (read_noise_e, dark_current_e_s, etc.)
"""

import numpy as np
import sys
import os

# Add spectral-analyzer to path
sys.path.insert(0, os.path.dirname(__file__))

from exposure_render_demo import ExposureSession, CameraOptics, FilmExposure
from sensor_film_db import SensorFilmDatabase


def test_t2_sensor_film_loading():
    """Test T2: ExposureSession loads sensor/film database on init."""
    print("\n[T2] Testing sensor/film slot loading...")
    
    # Create minimal optics and film
    optics = CameraOptics(
        focal_mm=50.0,
        aperture_mm=25.0,  # f/2 (50mm / 25mm)
        pixel_pitch_um=5.0,
        sensor_w_mm=24.0,
        sensor_h_mm=16.0,
    )
    film = FilmExposure(
        quantum_efficiency=0.4,
        exposure_time_s=0.02,
    )
    
    # Create session with default slots
    session = ExposureSession(
        optics=optics,
        film=film,
        width=1024,
        height=768,
        total_rays=10_000,
        rays_per_batch=1000,
        show_hud=False,
    )
    
    # Verify database loaded
    assert session._sensor_film_db is not None, "Database not loaded"
    assert session._sensor_film_tensors is not None, "Tensors not built"
    assert 'sensor' in session._sensor_film_tensors, "Sensor tensor missing"
    assert 'film' in session._sensor_film_tensors, "Film tensor missing"
    
    sensor_tensor = session._sensor_film_tensors['sensor']
    film_tensor = session._sensor_film_tensors['film']
    assert sensor_tensor.shape == (8, 48), f"Sensor shape wrong: {sensor_tensor.shape}"
    assert film_tensor.shape == (8, 64), f"Film shape wrong: {film_tensor.shape}"
    
    # Verify metadata built for default slots
    assert len(session._sensor_film_metadata) == 8, "Metadata not built for 8 slots"
    assert session._sensor_film_metadata[0]['active'], "Slot 0 should be active"
    assert session._sensor_film_metadata[1]['active'] == False, "Slot 1 should be inactive"
    
    # Verify metadata has required keys
    meta0 = session._sensor_film_metadata[0]
    assert 'qe_peak' in meta0, "qe_peak missing"
    assert 'read_noise_e' in meta0, "read_noise_e missing"
    assert 'dark_current_e_s' in meta0, "dark_current_e_s missing"
    
    print(f"  ✅ Database loaded: {len(session._sensor_film_db._sensor_order)} sensors, "
          f"{len(session._sensor_film_db._film_order)} films")
    print(f"  ✅ Slot 0: {meta0['sensor_name']} + {meta0['film_name']}")
    print(f"  ✅ QE={meta0['qe_peak']:.3f}, Read Noise={meta0['read_noise_e']:.2f} e-, "
          f"Dark Current={meta0['dark_current_e_s']:.2e} e-/s")


def test_t3_binding_method():
    """Test T3: _bind_sensor_film_ssbo method exists and callable."""
    print("\n[T3] Testing binding helper method...")
    
    optics = CameraOptics(
        focal_mm=50.0,
        aperture_mm=25.0,
        pixel_pitch_um=5.0,
        sensor_w_mm=24.0,
        sensor_h_mm=16.0,
    )
    film = FilmExposure(
        quantum_efficiency=0.4,
        exposure_time_s=0.02,
    )
    
    session = ExposureSession(
        optics=optics,
        film=film,
        width=1024,
        height=768,
        total_rays=10_000,
        rays_per_batch=1000,
        show_hud=False,
    )
    
    # Verify method exists
    assert hasattr(session, '_bind_sensor_film_ssbo'), "Method not found"
    
    # Test with None tracer (should handle gracefully)
    try:
        session._bind_sensor_film_ssbo(None)
        print("  ✅ Method handles None tracer gracefully")
    except Exception as e:
        print(f"  ❌ Method failed with None tracer: {e}")
        raise


def test_t4_sensor_integral():
    """Test T4: _make_sensor_integral produces valid output with noise model."""
    print("\n[T4] Testing sensor integral with noise model...")
    
    optics = CameraOptics(
        focal_mm=50.0,
        aperture_mm=25.0,
        pixel_pitch_um=5.0,
        sensor_w_mm=24.0,
        sensor_h_mm=16.0,
    )
    film = FilmExposure(
        quantum_efficiency=0.4,
        exposure_time_s=0.02,
    )
    
    session = ExposureSession(
        optics=optics,
        film=film,
        width=256,
        height=192,
        total_rays=10_000,
        rays_per_batch=1000,
        show_hud=False,
    )
    
    # Create sensor integral
    sensor_obj = session._make_sensor_integral(backend="cpp", gain=1.0)
    
    assert sensor_obj is not None, "Sensor integral object is None"
    assert sensor_obj.photons_per_pixel.shape == (192, 256), "Photons shape wrong"
    assert sensor_obj.electrons_per_pixel.shape == (192, 256), "Electrons shape wrong"
    assert sensor_obj.snr_linear.shape == (192, 256), "SNR shape wrong"
    
    # Verify metrics has noise terms (T5)
    metrics = sensor_obj.metrics
    required_keys = [
        'read_noise_e',
        'dark_current_e_s',
        'dark_current_accumulated_e',
        'exposure_time_s',
        'full_well_e',
    ]
    for key in required_keys:
        assert key in metrics, f"Noise term '{key}' missing from metrics"
        print(f"  ✅ {key}: {metrics[key]}")
    
    # Verify SNR formula logic: SNR = sqrt(e) / sqrt(read_noise^2 + dark*t + e)
    # For slot 0, should have non-zero values
    peak_snr = float(np.nanmax(sensor_obj.snr_linear))
    assert peak_snr > 0.0, "Peak SNR is zero or NaN"
    print(f"  ✅ SNR computed: peak={peak_snr:.2f}, mean={sensor_obj.mean_snr:.2f}")
    
    # Verify photons and electrons non-trivial
    mean_photons = float(np.mean(sensor_obj.photons_per_pixel))
    mean_electrons = float(np.mean(sensor_obj.electrons_per_pixel))
    print(f"  ✅ Photons: mean={mean_photons:.1f}, electrons: mean={mean_electrons:.1f}")


def test_t5_metadata_completeness():
    """Test T5: Saved metadata includes all noise model terms."""
    print("\n[T5] Testing metadata completeness...")
    
    optics = CameraOptics(
        focal_mm=50.0,
        aperture_mm=25.0,
        pixel_pitch_um=5.0,
        sensor_w_mm=24.0,
        sensor_h_mm=16.0,
    )
    film = FilmExposure(
        quantum_efficiency=0.4,
        exposure_time_s=0.02,
    )
    
    session = ExposureSession(
        optics=optics,
        film=film,
        width=128,
        height=96,
        total_rays=1000,
        rays_per_batch=100,
        show_hud=False,
    )
    
    sensor_obj = session._make_sensor_integral(backend="cpp", gain=1.0)
    metrics = sensor_obj.metrics
    
    # List all noise-related keys
    noise_keys = {
        'read_noise_e': float,
        'dark_current_e_s': float,
        'dark_current_accumulated_e': float,
        'exposure_time_s': float,
        'full_well_e': float,
        'qe_peak': float,
    }
    
    for key, expected_type in noise_keys.items():
        assert key in metrics, f"Missing: {key}"
        val = metrics[key]
        assert isinstance(val, (int, float, np.floating)), f"Wrong type for {key}: {type(val)}"
        print(f"  ✅ {key:30s} = {val:15.6e}")


if __name__ == '__main__':
    print("=" * 70)
    print("Phase 3C Integration Tests (T1-T5)")
    print("=" * 70)
    
    try:
        test_t2_sensor_film_loading()
        test_t3_binding_method()
        test_t4_sensor_integral()
        test_t5_metadata_completeness()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
