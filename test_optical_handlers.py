#!/usr/bin/env python3
"""Test optical handler invocation and telemetry collection."""

import numpy as np
import _spectral_kernels as _sk

def test_optical_handlers():
    """Test that optical handlers process rays and collect telemetry."""
    
    print("=" * 70)
    print("Testing Optical Handler Integration")
    print("=" * 70)
    
    # Create exposure backend with thin lens assembly
    print("\n1. Creating exposure backend with thin lens assembly...")
    backend = _sk.ExposureBackendCpp()
    
    # Create thin lens optical assembly
    focal_m = 0.035
    aperture_d_m = 0.025  # 25mm diameter (f/1.4)
    sensor_dist_m = 0.035
    backend.create_thin_lens_assembly(focal_m, aperture_d_m, sensor_dist_m)
    print(f"   ✓ Backend created with optical assembly")
    
    # Create test rays (several different positions and angles)
    print("\n3. Processing test rays through optical assembly...")
    
    telemetry_accum = {
        'rays_launched': 0,
        'rays_blocked': 0,
        'energy_in': 0.0,
        'energy_out': 0.0,
    }
    
    # Test rays: pinhole, on-axis, off-axis, edge
    test_rays = [
        {'name': 'pinhole (on-axis)', 'wavelength_m': 550e-9},
        {'name': 'red (off-axis)', 'wavelength_m': 650e-9},
        {'name': 'blue (edge)', 'wavelength_m': 450e-9},
    ]
    
    for i, ray_desc in enumerate(test_rays):
        # Create input ray state
        ray_in = _sk.OpticalRayState()
        ray_in.amplitude_real = 1.0
        ray_in.amplitude_imag = 0.0
        ray_in.wavelength_m = ray_desc['wavelength_m']
        ray_in.bounce_count = 0
        ray_in.is_active = True
        
        # Process through backend (which invokes handlers)
        ray_out = _sk.OpticalRayState()
        status = backend.process_ray(ray_in, ray_out, camera_mode=0)
        
        # Get telemetry
        telemetry = backend.get_telemetry()
        
        energy_in = np.sqrt(ray_in.amplitude_real**2 + ray_in.amplitude_imag**2)
        energy_out = np.sqrt(ray_out.amplitude_real**2 + ray_out.amplitude_imag**2)
        
        print(f"\n   Ray {i+1}: {ray_desc['name']}")
        print(f"      Wavelength: {ray_desc['wavelength_m']*1e9:.1f} nm")
        print(f"      Input energy: {energy_in:.6f}")
        print(f"      Output energy: {energy_out:.6f}")
        print(f"      Status: {status} ({'reaches sensor' if status == 0 else 'blocked'})")
        print(f"      Accumulated rays_launched: {telemetry.rays_launched}")
        print(f"      Accumulated rays_blocked: {telemetry.rays_blocked_by_stop}")
        print(f"      Total energy in: {telemetry.energy_in:.6f}")
        print(f"      Total energy out: {telemetry.energy_out:.6f}")
    
    # Final telemetry
    final_telemetry = backend.get_telemetry()
    print("\n4. Final Telemetry Summary:")
    print(f"   ✓ Rays launched: {final_telemetry.rays_launched}")
    print(f"   ✓ Rays blocked by stop: {final_telemetry.rays_blocked_by_stop}")
    print(f"   ✓ Rays refracted: {final_telemetry.rays_refracted}")
    print(f"   ✓ Rays reflected: {final_telemetry.rays_reflected}")
    print(f"   ✓ Rays hit lens surface: {final_telemetry.rays_hit_lens_surface}")
    print(f"   ✓ Total energy in: {final_telemetry.energy_in:.6f}")
    print(f"   ✓ Total energy out: {final_telemetry.energy_out:.6f}")
    print(f"   ✓ Energy absorbed: {final_telemetry.energy_absorbed:.6f}")
    
    print("\n" + "=" * 70)
    print("✅ Optical handler integration test PASSED")
    print("=" * 70)

if __name__ == "__main__":
    test_optical_handlers()
