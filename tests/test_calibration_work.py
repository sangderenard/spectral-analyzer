from pathlib import Path

from camera_software.calibration_work import CalibrationWorkCatalog


def test_calibration_work_is_configuration_addressed_and_resumable(tmp_path: Path):
    catalog = CalibrationWorkCatalog(str(tmp_path / "calibration_assets"))
    manifest = {
        "parameters": {"focus_distance_m": 3.75},
        "render_product": {"width": 200, "height": 200},
        "resolved_ray_trace_settings": {"total_rays": 200_000},
        "exposure_control_settings": {"epoch_bundle_count": 8},
    }
    first = catalog.begin("focus-hall", "Focus hall", manifest, 11)
    catalog.checkpoint(first.work_key, 11)
    second = catalog.begin("focus-hall", "Focus hall", manifest, 12)

    assert second.work_key == first.work_key
    assert second.launch_count == 2
    assert Path(second.manifest_path).is_file()
    assert second.status == "working"

    preview = tmp_path / "preview.png"
    linear = tmp_path / "linear.npy"
    result = tmp_path / "result.json"
    for path in (preview, linear, result):
        path.write_bytes(b"evidence")
    completed = catalog.complete(
        second.work_key, 12,
        preview_path=str(preview), linear_path=str(linear),
        result_manifest_path=str(result), elapsed_s=2.5,
    )
    assert completed.status == "complete"
    assert completed.completed_runs == 1
    assert CalibrationWorkCatalog(str(tmp_path / "calibration_assets")).snapshot()[0].status == "complete"
