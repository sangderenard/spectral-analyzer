"""Native T1/T3 must retain wavelength-scale authored interfaces."""

import time

import numpy as np
import pytest

from ray_tracer_bridge import per_tri_spectral_to_mat_buf


kernels = pytest.importorskip("_spectral_kernels")


def test_native_hit_survives_micrometre_triangle_determinant():
    frequency = np.array([299_792_458.0 / 550e-9], np.float64)
    zero = np.zeros((1, 1), np.float64)
    material_ids, materials, count = per_tri_spectral_to_mat_buf(
        zero, zero, zero, frequency
    )
    half_width = 0.5e-6
    triangle = np.array([[
        -half_width, -half_width, 0.18,
         half_width, -half_width, 0.18,
         0.0,         half_width, 0.18,
    ]], np.float64)
    tracer = kernels.RayTracer(
        1, triangle, np.array([[0., 0., 1.]], np.float64),
        material_ids, materials, count, frequency, 299_792_458.0,
        np.zeros(1, np.float64),
    )
    try:
        tracer.submit_rays(
            np.array([[0., 0., 0.18001]], np.float64),
            np.array([[0., 0., -1.]], np.float64),
            np.ones((1, 1), np.complex128),
            max_bounces=1, max_children=1, min_amplitude=0.,
        )
        deadline = time.monotonic() + 5
        while tracer.in_flight_count() and time.monotonic() < deadline:
            time.sleep(.001)
        assert tracer.in_flight_count() == 0
        records = tracer.drain_records(16)
        strikes = np.flatnonzero(np.asarray(records["kind"]) == 0)
        assert len(strikes) == 1
        assert int(records["hit_tri"][strikes[0]]) == 0
        assert records["pos"][strikes[0], 2] == pytest.approx(.18, abs=1e-7)
    finally:
        tracer.stop_pipeline()


def _cube(center, half_width):
    triangles, normals = [], []
    center = np.asarray(center, np.float64)
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for sign in (-1, 1):
            middle = center.copy(); middle[axis] += sign * half_width
            u = np.zeros(3); u[others[0]] = half_width
            v = np.zeros(3); v[others[1]] = half_width
            corners = [middle-u-v, middle+u-v, middle+u+v, middle-u+v]
            normal = np.zeros(3); normal[axis] = sign
            for ids in ((0, 1, 2), (0, 2, 3)):
                tri = np.asarray([corners[i] for i in ids])
                if np.dot(np.cross(tri[1]-tri[0], tri[2]-tri[0]), normal) < 0:
                    tri = tri[[0, 2, 1]]
                triangles.append(tri)
                normals.append(normal)
    return np.asarray(triangles), np.asarray(normals)


def test_native_t3_resolves_host_dye_and_bubble_from_authored_surfaces():
    frequency = np.array([299_792_458.0 / 550e-9], np.float64)
    host, host_normals = _cube((0, 0, 0), .002)
    dye, dye_normals = _cube((.001, 0, 0), .0003)
    bubble, bubble_normals = _cube((-.001, 0, 0), .0003)
    triangles = np.concatenate((host, dye, bubble))
    normals = np.concatenate((host_normals, dye_normals, bubble_normals))
    n = len(triangles)
    zero = np.zeros((n, 1), np.float64)
    index = np.full((n, 1), 1.5, np.float64)
    extinction = zero.copy(); extinction[len(host):len(host)+len(dye)] = 1e-7
    ids, materials, count = per_tri_spectral_to_mat_buf(
        zero, zero, zero, frequency,
        transmittance_bands=np.ones_like(zero),
        ior_real_bands=index, ior_imag_bands=extinction,
    )
    tracer = kernels.RayTracer(
        n, triangles.reshape(n, 9), normals, ids, materials, count,
        frequency, 299_792_458.0, np.zeros(1, np.float64),
    )
    try:
        tracer.set_studio_regions(
            np.array([[0, ids[0]], [2, ids[len(host)]], [3, -1]], np.int32),
            np.repeat(np.arange(3, dtype=np.int32), 12),
        )
        for point, expected_n, expected_k in (
            ((0., 0., .003), 1., 0.),
            ((0., 0., 0.), 1.5, 0.),
            ((0., 0., .002 + 1e-12), 1., 0.),
            ((0., 0., .002 - 1e-12), 1.5, 0.),
            ((.001, 0., 0.), 1.5, 1e-7),
            ((-.001, 0., 0.), 1., 0.),
        ):
            medium = tracer.resolve_studio_medium(
                np.asarray(point, np.float64),
                np.array([0., 0., 1.], np.float64), frequency,
            )
            assert medium['n_real'][0] == pytest.approx(expected_n)
            assert medium['n_imag'][0] == pytest.approx(expected_k)
    finally:
        tracer.stop_pipeline()


def test_native_exact_sphere_competes_with_host_mesh_and_resolves_bubble():
    frequency = np.array([299_792_458.0 / 550e-9], np.float64)
    host, normals = _cube((0, 0, 0), .002)
    owner = np.zeros((1, 3, 3), np.float64)
    triangles = np.concatenate((host, owner))
    normals = np.concatenate((normals, np.array([[0., 0., 1.]])))
    count = len(triangles)
    zero = np.zeros((count, 1), np.float64)
    index = np.full((count, 1), 1.5, np.float64)
    ids, materials, material_count = per_tri_spectral_to_mat_buf(
        zero, zero, zero, frequency,
        transmittance_bands=np.ones_like(zero), ior_real_bands=index,
    )
    tracer = kernels.RayTracer(
        count, triangles.reshape(count, 9), normals, ids, materials,
        material_count, frequency, 299_792_458.0, np.zeros(1, np.float64),
    )
    try:
        tracer.set_studio_regions(
            np.array([[0, ids[0]], [3, -1]], np.int32),
            np.array([0] * 12 + [1], np.int32),
        )
        tracer.set_studio_spheres(
            np.array([[0., 0., 0., .0003, 12.]], np.float64))
        for point, expected in (((0., 0., 0.), 1.),
                                ((.0005, 0., 0.), 1.5),
                                ((.0003 - 1e-12, 0., 0.), 1.),
                                ((.0003 + 1e-12, 0., 0.), 1.5)):
            medium = tracer.resolve_studio_medium(
                np.asarray(point, np.float64),
                np.array([1., 0., 0.], np.float64), frequency)
            assert medium['n_real'][0] == pytest.approx(expected)
        tracer.submit_rays(
            np.array([[0., 0., 0.]], np.float64),
            np.array([[1., 0., 0.]], np.float64),
            np.ones((1, 1), np.complex128),
            max_bounces=3, max_children=2, min_amplitude=0.,
        )
        deadline = time.monotonic() + 5
        while tracer.in_flight_count() and time.monotonic() < deadline:
            time.sleep(.001)
        assert tracer.in_flight_count() == 0
        records = tracer.drain_records(16)
        strikes = np.flatnonzero(np.asarray(records['kind']) == 0)
        assert len(strikes) >= 1
        assert int(records['hit_tri'][strikes[0]]) == 12
        assert records['pos'][strikes[0], 0] == pytest.approx(.0003, abs=1e-8)
        assert 12 in np.asarray(records['hit_tri'])[strikes]
        assert np.any(np.isclose(records['pos'][strikes, 0], .002, atol=1e-8))
    finally:
        tracer.stop_pipeline()
