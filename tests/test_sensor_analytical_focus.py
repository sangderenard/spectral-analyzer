"""CPU-only contracts for the actual composed sensor's analytical focus math.

The small GLSL expression block is read from production, not copied into a
second implementation. Its results are checked against independent ray/plane
intersection geometry. No renderer import or OpenGL context is required.
"""
import ast
from functools import lru_cache
import math
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def renderer_tree():
    return ast.parse((ROOT / "demo_pluck_gl.py").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def analytical_block():
    node = next(n for n in renderer_tree().body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_GPU_SENSOR_CS" for t in n.targets)
                and isinstance(n.value, ast.Constant))
    shader = node.value.value
    return shader.split("// Analytical thin-lens path.", 1)[1].split("float ap =", 1)[0]


def shader_focus(*, focus, pixel=(.83, .16), tilt=(0., 0.), shift=(0., 0.),
                 fov=52., aspect=1.5, eye=(0., 0., 0.), basis=None):
    if basis is None:
        basis = np.eye(3)
    right, up, fwd = np.asarray(basis, dtype=float)
    ns = dict(uCamFwd=fwd, uCamRight=right, uCamUp=up, ro=np.asarray(eye, float),
              uFocusDist=focus, pu=pixel[0], pv=pixel[1],
              uCamFovTan=math.tan(math.radians(fov) / 2), uCamAspect=aspect,
              uLensTilt=SimpleNamespace(x=tilt[0], y=tilt[1]),
              uTiltShift=SimpleNamespace(x=shift[0], y=shift[1]),
              tan=math.tan, max=max)
    source = re.sub(r"//[^\n]*", "", analytical_block())
    statements = re.findall(r"(?:float|vec3)\s+(\w+)\s*=\s*([^;]+);", source)
    assert [name for name, _ in statements] == ["fu", "fv", "focal_ray", "_lt_denom", "fd_eff", "focus_pt"]
    for name, expression in statements:
        ns[name] = eval(" ".join(expression.split()), {"__builtins__": {}}, ns)
    return ns


def sensor_method(name):
    cls = next(n for n in renderer_tree().body if isinstance(n, ast.ClassDef)
               and n.name == "SensorAccumulator")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(np=np, math=math)
    if name == "__init__":
        # Execute the constructor's actual focus assignment without allocating
        # textures, importing the full renderer, or duplicating the clamp.
        node = next(n for n in node.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and t.attr == "_focus_dist" for t in n.targets))
        def constructor_focus(self, *, focus_dist):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "sensor-focus-init", "exec"),
                 dict(self=self, focus_dist=focus_dist))
        return constructor_focus
    exec(compile(ast.Module(body=[node], type_ignores=[]), "sensor-focus-method", "exec"), ns)
    return ns[name]


@pytest.mark.parametrize("method", ["__init__", "set_camera", "set_lens"])
@pytest.mark.parametrize("focus", [-1., 0., .0005, .001, .005, .009, .019, .02, 1., 10., 100.])
def test_all_cpu_entry_points_use_one_mm_numerical_floor(method, focus):
    obj = SimpleNamespace()
    kwargs = dict(focus_dist=focus)
    if method == "set_camera":
        kwargs.update(camera_eye=[0, 0, 0], camera_target=[0, 0, 1], fov_deg=52)
    sensor_method(method)(obj, **kwargs)
    assert obj._focus_dist == max(.001, focus)
    assert math.isfinite(obj._focus_dist)


@pytest.mark.parametrize("method", ["set_camera", "set_lens"])
def test_omitted_focus_does_not_reset_existing_lens(method):
    obj = SimpleNamespace(_focus_dist=.005)
    kwargs = dict(camera_eye=[0, 0, 0], camera_target=[0, 1, 0], fov_deg=52) if method == "set_camera" else {}
    sensor_method(method)(obj, **kwargs)
    assert obj._focus_dist == .005


@pytest.mark.parametrize("focus", [0., .0005, .001, .005, .009, .019, .02, 1., 10.])
@pytest.mark.parametrize("tilt", [(0., 0.), (.11, 0.), (0., -.09), (.11, -.09)])
@pytest.mark.parametrize("pixel", [(.5, .5), (0., 0.), (1., 1.), (.1, .9)])
def test_actual_shader_intersects_axial_tilted_plane(focus, tilt, pixel):
    # A translated, rotated camera proves that this is not world-Z focus.
    angle = .83
    basis = np.array([[math.cos(angle), 0, -math.sin(angle)], [0, 1, 0],
                      [math.sin(angle), 0, math.cos(angle)]])
    eye = np.array([.7, -.2, 1.1])
    result = shader_focus(focus=focus, pixel=pixel, tilt=tilt, shift=(.13, -.07),
                          fov=78., aspect=1.77, eye=eye, basis=basis)
    normal = basis[2] - math.tan(tilt[1]) * basis[0] - math.tan(tilt[0]) * basis[1]
    intercept = max(.001, focus)
    unit_ray = result["focal_ray"] / np.linalg.norm(result["focal_ray"])
    assert result["_lt_denom"] > .005  # exact plane, not the guarded horizon
    expected = eye + unit_ray * (intercept / np.dot(normal, unit_ray))
    np.testing.assert_allclose(result["focus_pt"], expected, rtol=2e-13, atol=3e-15)
    assert np.dot(normal, result["focus_pt"] - eye) == pytest.approx(intercept, rel=2e-12, abs=3e-15)
    # A disk or polygon pupil changes origin, never the selected focal point.
    for aperture in (0., .0001, .002, .02):
        for theta in (0., .7, 2.1, 4.2):
            origin = eye + aperture * (math.cos(theta) * basis[0] + math.sin(theta) * basis[1])
            direction = result["focus_pt"] - origin
            direction /= np.linalg.norm(direction)
            distance = (intercept - np.dot(normal, origin - eye)) / np.dot(normal, direction)
            assert np.isfinite(direction).all() and distance > 0
            np.testing.assert_allclose(origin + distance * direction, expected, rtol=2e-12, atol=3e-14)


def test_off_axis_focus_is_not_the_old_sphere_and_pinhole_projection_is_preserved():
    result = shader_focus(focus=.005, pixel=(1., 0.), fov=90., aspect=1.)
    point = result["focus_pt"]
    assert point[2] == pytest.approx(.005)
    old = result["focal_ray"] / np.linalg.norm(result["focal_ray"]) * .005
    assert old[2] == pytest.approx(.005 / math.sqrt(3))
    for focus in (.001, .005, 10.):
        point = shader_focus(focus=focus, pixel=(1., 0.), fov=90., aspect=1.)["focus_pt"]
        np.testing.assert_allclose(point / np.linalg.norm(point), old / np.linalg.norm(old))


def test_minimum_is_axial_intercept_not_an_off_axis_tilt_clip():
    result = shader_focus(focus=.001, pixel=(0., .5), tilt=(0., .25), fov=70.)
    assert result["_lt_denom"] > 1
    assert 0 < result["fd_eff"] < .001
    normal = np.array([-math.tan(.25), 0., 1.])
    assert normal @ result["focus_pt"] == pytest.approx(.001)


@pytest.mark.parametrize("denominator", [.006, .005, .004, 0., -.1])
def test_existing_tilt_horizon_guard_is_finite_and_unchanged(denominator):
    # pixel.x=1, fov=90, aspect=1 gives fu=1.
    tilt = (0., math.atan(1 - denominator))
    result = shader_focus(focus=.001, pixel=(1., .5), fov=90., aspect=1., tilt=tilt)
    assert result["_lt_denom"] == pytest.approx(denominator)
    assert result["fd_eff"] == pytest.approx(.001 / max(.005, denominator))
    assert np.isfinite(result["focus_pt"]).all()
    assert 0 < result["focus_pt"][2] <= .2 + 1e-14
