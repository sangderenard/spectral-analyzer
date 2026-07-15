from types import SimpleNamespace

import numpy as np

from exposure_render_demo import (
    CppExposureBackend,
    _average_native_sensor_sweeps,
    _native_bdpt_refinement_schedule,
    _native_bdpt_work_units,
    _resolve_native_sensor_sweeps,
)


def test_work_units_exactly_partition_large_sweep_on_pixel_boundaries():
    n_ap = 192
    units, plan = _native_bdpt_work_units(1024, 1024, n_ap)

    assert plan["schedule_rays"] == 1024 * 1024 * n_ap
    assert plan["sensor_rgb_bytes"] == 1024 * 1024 * 3 * 4
    assert max(count for _, count in units) <= 200_000
    assert all(offset % n_ap == 0 and count % n_ap == 0
               for offset, count in units)

    cursor = 0
    for offset, count in units:
        assert offset == cursor
        cursor += count
    assert cursor == plan["schedule_rays"]


def test_requested_extra_units_remain_complete_and_nonempty():
    n_ap = 7
    units, plan = _native_bdpt_work_units(
        65, 63, n_ap, requested_packages=19, primary_ray_cap=10_000,
    )

    assert len(units) >= 19
    assert all(count > 0 and count <= 10_000 for _, count in units)
    assert sum(count for _, count in units) == plan["schedule_rays"]
    assert all(offset % n_ap == 0 and count % n_ap == 0
               for offset, count in units)


def test_refinement_sweeps_repeat_complete_spatial_plan():
    units, _ = _native_bdpt_work_units(120, 120, 32, primary_ray_cap=200_000)
    schedule = _native_bdpt_refinement_schedule(units, sensor_sweeps=3)

    assert len(units) == 3
    assert len(schedule) == 9
    for sweep_index in range(3):
        cycle = schedule[sweep_index * len(units):(sweep_index + 1) * len(units)]
        assert cycle == [(sweep_index, offset, count) for offset, count in units]


def test_explicit_sensor_sweeps_override_authored_default():
    assert _resolve_native_sensor_sweeps(None, 4) == 4
    assert _resolve_native_sensor_sweeps(2, 4) == 2
    assert _resolve_native_sensor_sweeps(None) == 1


def test_native_sensor_sweep_average_preserves_dtype():
    accumulated = np.asarray([4.0, 8.0, 12.0], dtype=np.float32)
    averaged = _average_native_sensor_sweeps(accumulated, 4)

    assert averaged.dtype == accumulated.dtype
    np.testing.assert_array_equal(averaged, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_native_backend_readback_uses_passed_session_sweep_count():
    class FakeTracer:
        def submit_emissive_triangles(self, *args):
            return 1

        def signal_flash_dispatched(self):
            pass

        def begin_sensor_batching(self):
            pass

        def submit_sensor_sweep(self, **kwargs):
            return kwargs["max_rays"]

        def signal_sensor_dispatched(self):
            pass

        def join_t5(self):
            pass

        def end_sensor_batching(self):
            pass

        def get_sensor_image(self):
            return np.ones((1, 1, 3), dtype=np.float32)

        def get_sensor_image_linear(self):
            return np.full((1, 1, 3), 4.0, dtype=np.float32)

    backend = CppExposureBackend.__new__(CppExposureBackend)
    backend.tracer = FakeTracer()
    backend._sensor_camera_desc = {"n_px": 1, "n_py": 1}
    backend.cam = SimpleNamespace(width=1, height=1)
    backend.max_bounces = 1
    backend._gpu_resident = False
    backend._native_sensor_image = None
    backend._native_sensor_linear_image = None
    backend.n_rays_accumulated = 0

    image = backend.run_thick_lens_native_bdpt(
        emitter_tri_ids=np.asarray([0], dtype=np.int32),
        total_rays=1,
        sensor_rays_per_batch=1,
        n_aperture_samples=1,
        max_children=1,
        seed=1,
        sensor_sweeps=4,
    )

    np.testing.assert_array_equal(image, np.ones((1, 1, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        backend._native_sensor_linear_image,
        np.ones((1, 1, 3), dtype=np.float32),
    )


def test_small_sensor_can_require_more_spatial_units_than_forward_batches():
    units, plan = _native_bdpt_work_units(
        120, 120, 32, requested_packages=1, primary_ray_cap=200_000,
    )

    assert plan["schedule_rays"] == 120 * 120 * 32
    assert len(units) == 3
    assert sum(count for _, count in units) == plan["schedule_rays"]
