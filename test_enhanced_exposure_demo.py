#!/usr/bin/env python3
"""
Integration test for enhanced exposure demo with opt-in file output and features.

Validates:
- In-memory rendering without requiring file I/O
- Sensor/film integration fully active
- Image data stored in results for direct display
- Optional file output via save_files flag
- Full exposure session with BDPT, field capture, and integral splitting
"""

import numpy as np
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from exposure_render_demo import (
    ExposureSession, CameraOptics, FilmExposure, IntegralSplitConfig,
    FieldCaptureConfig, CameraVisibilityConfig
)


def test_in_memory_rendering():
    """Test rendering without file output."""
    print("\n[Feature] In-memory rendering (no --save-files)...")
    
    optics = CameraOptics(
        focal_mm=50.0,
        aperture_mm=25.0,
        pixel_pitch_um=5.0,
        sensor_w_mm=24.0,
        sensor_h_mm=16.0,
    )
    film = FilmExposure(
        iso=100,
        exposure_time_s=0.01,
        quantum_efficiency=0.4,
    )
    
    # Create session WITHOUT save_files
    session = ExposureSession(
        optics=optics,
        film=film,
        width=512,
        height=384,
        total_rays=100_000,
        rays_per_batch=10_000,
        max_bounces=4,
        backends=("cpp",),
        out_dir="exposures_test_temp",
        integrator="bdpt",
        n_frames_planned=1,
        show_hud=True,
        save_files=False,  # KEY: No file output
    )
    
    results = session.render_one_exposure(t=0.0)
    
    # Verify results
    assert len(results) >= 1, "No results returned"
    r = results[0]
    
    # Verify image data is in memory
    assert r.image_data is not None, "image_data is None"
    assert r.image_data.shape == (384, 512, 3), f"Wrong shape: {r.image_data.shape}"
    assert r.image_data.dtype == np.float32, f"Wrong dtype: {r.image_data.dtype}"
    assert r.image_data.min() >= 0.0 and r.image_data.max() <= 1.0, "Data outside [0,1]"
    
    # Verify files were NOT created (save_files=False)
    assert not os.path.exists(r.image_path), f"PNG created when save_files=False: {r.image_path}"
    
    print(f"  ✅ In-memory image shape: {r.image_data.shape}")
    print(f"  ✅ Image value range: [{r.image_data.min():.3f}, {r.image_data.max():.3f}]")
    print(f"  ✅ No files created (save_files=False)")
    print(f"  ✅ N_rays: {r.n_rays_emitted:_}, gain: {r.gain_linear:.3e}×")
    
    # Cleanup
    if os.path.exists("exposures_test_temp"):
        shutil.rmtree("exposures_test_temp")


def test_optional_file_output():
    """Test file output with --save-files equivalent."""
    print("\n[Feature] Optional file output (save_files=True)...")
    
    optics = CameraOptics(
        focal_mm=50.0,
        aperture_mm=25.0,
        pixel_pitch_um=5.0,
        sensor_w_mm=24.0,
        sensor_h_mm=16.0,
    )
    film = FilmExposure(
        iso=100,
        exposure_time_s=0.01,
        quantum_efficiency=0.4,
    )
    
    tmpdir = tempfile.mkdtemp(prefix="exposure_test_")
    
    try:
        # Create session WITH save_files
        session = ExposureSession(
            optics=optics,
            film=film,
            width=256,
            height=192,
            total_rays=50_000,
            rays_per_batch=5_000,
            max_bounces=4,
            backends=("cpp",),
            out_dir=tmpdir,
            integrator="bdpt",
            n_frames_planned=1,
            show_hud=True,
            save_files=True,  # KEY: Enable file output
        )
        
        results = session.render_one_exposure(t=0.0)
        
        # Verify results
        assert len(results) >= 1, "No results returned"
        r = results[0]
        
        # Verify image data is in memory
        assert r.image_data is not None, "image_data is None"
        
        # Verify files WERE created (save_files=True)
        assert os.path.exists(r.image_path), f"PNG not created: {r.image_path}"
        
        # Verify PNG can be loaded
        try:
            import pygame
            surf = pygame.image.load(r.image_path)
            w, h = surf.get_size()
            assert w == 256 and h == 192, f"Wrong PNG size: {w}×{h}"
            print(f"  ✅ PNG created and readable: {os.path.basename(r.image_path)}")
        except ImportError:
            print(f"  ✅ PNG created (pygame not available for verification)")
        
        print(f"  ✅ File output enabled, artifacts saved to disk")
        print(f"  ✅ Image path: {r.image_path}")
        
    finally:
        # Cleanup
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


def test_sensor_film_integration():
    """Test that sensor/film features are fully integrated."""
    print("\n[Feature] Sensor/film integration in rendering...")
    
    optics = CameraOptics(
        focal_mm=50.0,
        aperture_mm=25.0,
        pixel_pitch_um=5.0,
        sensor_w_mm=24.0,
        sensor_h_mm=16.0,
    )
    film = FilmExposure(
        iso=100,
        exposure_time_s=0.01,
        quantum_efficiency=0.4,
    )
    
    session = ExposureSession(
        optics=optics,
        film=film,
        width=256,
        height=192,
        total_rays=50_000,
        rays_per_batch=5_000,
        max_bounces=4,
        backends=("cpp",),
        out_dir="exposures_test_temp",
        integrator="bdpt",
        n_frames_planned=1,
        show_hud=True,
        save_files=False,
        sensor_film_slots=[(0, 0)] + [(-1, -1)] * 7,  # Canon + Kodak as default
    )
    
    # Verify database is loaded
    assert session._sensor_film_db is not None, "Database not loaded"
    assert session._sensor_film_tensors is not None, "Tensors not built"
    
    # Verify metadata
    assert len(session._sensor_film_metadata) == 8, "Metadata length wrong"
    assert session._sensor_film_metadata[0]['active'], "Slot 0 should be active"
    
    results = session.render_one_exposure(t=0.0)
    
    # Verify results have sensor/film context
    r = results[0]
    assert r.image_data is not None, "image_data is None"
    
    # The sensor integral should have been created
    print(f"  ✅ Sensor/film database loaded: {len(session._sensor_film_db._sensor_order)} sensors")
    print(f"  ✅ Active slot: {session._sensor_film_metadata[0]['sensor_name']} + {session._sensor_film_metadata[0]['film_name']}")
    print(f"  ✅ Rendered with sensor integration")
    
    # Cleanup
    if os.path.exists("exposures_test_temp"):
        shutil.rmtree("exposures_test_temp")


def test_full_feature_session():
    """Test with all major features enabled."""
    print("\n[Feature] Full session with all features enabled...")
    
    optics = CameraOptics(
        focal_mm=50.0,
        aperture_mm=25.0,
        pixel_pitch_um=5.0,
        sensor_w_mm=24.0,
        sensor_h_mm=16.0,
    )
    film = FilmExposure(
        iso=100,
        exposure_time_s=0.01,
        quantum_efficiency=0.4,
    )
    
    session = ExposureSession(
        optics=optics,
        film=film,
        width=256,
        height=192,
        total_rays=50_000,
        rays_per_batch=5_000,
        max_bounces=4,
        backends=("cpp",),
        out_dir="exposures_test_temp",
        integrator="bdpt",
        n_frames_planned=1,
        show_hud=True,
        save_files=False,
        # Add field capture
        field_capture=FieldCaptureConfig(
            enabled=True,
            grid_kind="uniform",
            nx=4,
            ny=4,
            nz=4,
        ),
        # Add integral split config
        integral_split=IntegralSplitConfig(
            field_integrate_frac=0.35,
            field_bookkeep_frac=0.65,
            surface_integrate_frac=0.85,
            surface_bookkeep_frac=0.15,
            hdr_white_percentile=99.8,
        ),
        # Add camera visibility
        camera_visibility=CameraVisibilityConfig(),
    )
    
    results = session.render_one_exposure(t=0.0)
    
    r = results[0]
    assert r.image_data is not None, "image_data is None"
    assert r.frame_config_summary is not None, "frame_config_summary is None"
    
    # Verify config was applied
    config = r.frame_config_summary
    assert config.get('detail_level', -1) >= 0, "detail_level not set"
    
    print(f"  ✅ Field capture: {config.get('field', 'off')}")
    print(f"  ✅ Camera visibility: {config.get('cam_vis', 'unknown')}")
    print(f"  ✅ Detail level: {config.get('detail_level', 0)}")
    print(f"  ✅ Image produced: {r.image_data.shape}")
    
    # Cleanup
    if os.path.exists("exposures_test_temp"):
        shutil.rmtree("exposures_test_temp")


if __name__ == '__main__':
    print("=" * 70)
    print("Enhanced Exposure Demo Integration Tests")
    print("=" * 70)
    
    try:
        test_in_memory_rendering()
        test_optional_file_output()
        test_sensor_film_integration()
        test_full_feature_session()
        
        print("\n" + "=" * 70)
        print("✅ ALL FEATURE TESTS PASSED")
        print("=" * 70)
        print("\nSummary:")
        print("  ✅ In-memory rendering works without file I/O")
        print("  ✅ Optional file output via save_files flag")
        print("  ✅ Sensor/film features fully integrated")
        print("  ✅ All major features (field capture, integral split, camera visibility)")
        print("\nUsage:")
        print("  python exposure_render_demo.py              # In-memory, display only")
        print("  python exposure_render_demo.py --save-files # Save PNG/JSON to disk")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
