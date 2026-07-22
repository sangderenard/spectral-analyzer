from camera_software.spectral_startup_validation import (
    STARTUP_LANE_CONFIGURATIONS,
    run_startup_spectral_sanity,
)


def test_startup_matrix_traces_one_native_ray_per_lane():
    report = run_startup_spectral_sanity()

    assert report.passed
    assert tuple(case.lane_count for case in report.cases) == (
        STARTUP_LANE_CONFIGURATIONS
    )
    for case in report.cases:
        assert case.cpu_status == "passed"
        assert case.gpu_status == "not-run"
        assert case.measurements["rays_requested"] == case.lane_count
        assert case.measurements["segments_observed"] == case.lane_count
        assert case.measurements["unique_lanes_observed"] == case.lane_count
        assert case.measurements["maximum_frequency_error_hz"] == 0.0


def test_gpu_probe_failure_is_not_misreported_as_native_success():
    report = run_startup_spectral_sanity((3,), gpu_probe=lambda _lanes: False)

    assert not report.passed
    assert report.cases[0].cpu_status == "passed"
    assert report.cases[0].gpu_status == "failed"
