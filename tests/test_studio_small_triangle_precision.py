"""Legacy miniature-mesh determinant regression, without a graphics context."""
from pathlib import Path
import re
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'demo_pluck_gl.py'
RELATIVE = 'if (abs(a) <= 2.384185791015625e-7 * length(t.e1.xyz) * length(t.e2.xyz) * length(rd)) return false;'

def hit_functions():
    return re.findall(r'bool hit_tri\(vec3 ro, vec3 rd, TriGeom t, out float hit_t\) \{.*?\n\}', SOURCE.read_text(encoding='utf-8'), re.S)

def test_both_legacy_paths_use_relative_parallel_test():
    bodies = hit_functions()
    assert len(bodies) == 2
    assert all(RELATIVE in b for b in bodies)
    assert all('abs(a) < 1e-8' not in b for b in bodies)

def test_miniature_triangle_survives_relative_test():
    # 10 um edges: valid triangle, area 5e-11 m^2, viewed normally at 1 mm.
    e1 = np.array([1e-5, 0, 0], np.float32)
    e2 = np.array([0, 1e-5, 0], np.float32)
    for scale in (1e-2, 1, 1e2):
        a, b = e1 * scale, e2 * scale
        rd = np.array([0, 0, -1], np.float32)
        det = np.dot(a, np.cross(rd, b))
        limit = 2 * np.finfo(np.float32).eps * np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(rd)
        assert abs(det) > limit
    assert abs(np.dot(e1, np.cross(rd, e2))) < 1e-8  # old code misses it
    parallel = np.array([1, 0, 0], np.float32)
    assert np.dot(e1, np.cross(parallel, e2)) == 0
