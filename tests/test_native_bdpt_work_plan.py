from exposure_render_demo import _native_bdpt_work_units


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
