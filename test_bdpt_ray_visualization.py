#!/usr/bin/env python3
"""Test script to visualize BDPT ray paths.

This script runs a BDPT simulation and exports all backward ray paths
to a text file so you can see exactly where each ray goes.

Usage:
    python test_bdpt_ray_visualization.py
    
Output:
    - bdpt_rays_overlay.txt: Detailed ray path data
    - Console output showing statistics
"""

from thick_lens_focus_lab import (
    SceneConfig,
    ForwardCppLensBench,
    DEFAULT_FREQ_HZ,
)
import numpy as np

def main():
    # Create a scene config
    scene_cfg = SceneConfig()
    
    # Create BDPT bench
    print("[setup] Initializing BDPT bench...")
    bench = ForwardCppLensBench(
        scene_cfg,
        DEFAULT_FREQ_HZ,
        480,
        640,
    )
    
    # Run BDPT simulation
    print("[bdpt] Running BDPT with 64 bounces...")
    rgb = bench.capture_plate_bdpt_rgb(
        pixels=96,
        aperture_samples=4,
        seed=42,
        max_bounces=64,
        leak=1.0,
        n_rays_bdpt=512,
        camera_mode=0,
    )
    
    print(f"[result] RGB shape: {rgb.shape}")
    print(f"[result] RGB max: {float(np.max(rgb)):.6f}")
    
    # Export ray visualization
    print("\n[export] Exporting ray paths...")
    stats = bench.export_bdpt_ray_visualization(
        output_file="bdpt_rays_overlay.txt"
    )
    
    print("\n[summary] Ray Statistics:")
    print(f"  Rays traced: {stats['ray_count']}")
    print(f"  Total endpoints: {stats['endpoint_count']}")
    print(f"  Total path length: {stats['total_path_length']:.3f}m")
    print(f"  Max path length: {stats['max_path_length']:.3f}m")
    print(f"  Bounce distribution:")
    for bounces, count in sorted(stats['bounce_distribution'].items()):
        print(f"    {bounces} bounces: {count} rays")
    
    # Now read and analyze the file
    print("\n[analysis] Reading ray file to analyze patterns...")
    try:
        with open("bdpt_rays_overlay.txt", "r") as f:
            lines = f.readlines()
            
        # Count rays by path length
        ray_data = {}
        current_ray = None
        for line in lines:
            if line.startswith("ray "):
                parts = line.split()
                current_ray = {"bounces": 0, "max_pathlen": 0.0}
            elif line.strip().startswith("bounce ") and current_ray is not None:
                # Extract pathlen value
                if "pathlen=" in line:
                    start = line.find("pathlen=") + 8
                    end = line.find(" amplitude=")
                    try:
                        pathlen = float(line[start:end])
                        current_ray["max_pathlen"] = max(current_ray["max_pathlen"], pathlen)
                        current_ray["bounces"] += 1
                    except ValueError:
                        pass
        
        print("\n[patterns] Ray behavior analysis:")
        print("  Rays are exported in order of subpath ID")
        print("  Each ray shows all bounce points from sensor to termination")
        print("  Look for:")
        print("    - Rays terminating early (few bounces) → hitting low-reflectance materials")
        print("    - Rays with long paths → finding complex reflections")
        print("    - Amplitude decay per bounce → should be ~0.97 per reflection")
        print("\n  Check the file 'bdpt_rays_overlay.txt' for full details")
        
    except Exception as e:
        print(f"[error] Could not analyze file: {e}")

if __name__ == "__main__":
    main()
