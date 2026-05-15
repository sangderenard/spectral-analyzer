#!/usr/bin/env python3
"""Patch exposure_render_demo.py to add oracle/physical photon fields."""

import re

with open('exposure_render_demo.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the SensorIntegralObject creation and add oracle/physical fields
old_pattern = r"(sensor_uv_thin_lens_hits=\(None if uv_thin_hits is None else uv_thin_hits\.copy\(\)\),)\s*(\n\s+\))"
new_pattern = r"\1\n                    oracle_photons_per_pixel=oracle_photons.copy() if oracle_photons is not None else None,\n                    physical_photons_per_pixel=physical_photons.copy() if physical_photons is not None else None,\2"

new_content = re.sub(old_pattern, new_pattern, content)

if new_content != content:
    with open('exposure_render_demo.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("✅ Added oracle/physical photon fields to SensorIntegralObject")
else:
    print("⚠️  Pattern not found - checking alternate pattern...")
