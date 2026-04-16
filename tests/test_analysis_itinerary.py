from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _DummyTarget:
    pass


def test_analysis_itinerary_json_roundtrip():
    from analysis_itinerary import (
        AnalysisGlobalSettings,
        AnalysisItinerary,
        AnalysisRun,
        FFTAnalysisSettings,
        FilterbankAnalysisSettings,
    )

    itinerary = AnalysisItinerary(
        globals=AnalysisGlobalSettings(region_start=1.25, region_end=3.5),
        runs=[
            AnalysisRun(
                key="fft_1",
                engine="fft",
                label="FFT 1",
                settings=FFTAnalysisSettings(
                    fft_algorithm="stft",
                    hop_length=256,
                    bins_per_octave=48,
                ),
            ),
            AnalysisRun(
                key="fb_1",
                engine="fb",
                label="FB 1",
                settings=FilterbankAnalysisSettings(
                    fb_hop=4,
                    fb_crossovers=[120.0, 480.0],
                ),
            ),
        ],
        metadata={"name": "demo"},
    )

    restored = AnalysisItinerary.from_json(itinerary.to_json())

    assert restored.globals.region_start == 1.25
    assert restored.runs[0].settings.fft_algorithm == "stft"
    assert restored.runs[1].settings.fb_crossovers == [120.0, 480.0]


def test_analysis_itinerary_apply_run_to_stamps_globals_and_run_settings():
    from analysis_itinerary import (
        AnalysisGlobalSettings,
        AnalysisItinerary,
        AnalysisRun,
        FFTAnalysisSettings,
    )

    itinerary = AnalysisItinerary(
        globals=AnalysisGlobalSettings(region_start=2.0, resample_taps_idx=5),
        runs=[
            AnalysisRun(
                key="fft_1",
                engine="fft",
                settings=FFTAnalysisSettings(
                    fft_algorithm="nsgt",
                    hop_length=1024,
                    cqt_filter_scale=1.25,
                ),
            )
        ],
    )
    target = _DummyTarget()

    run = itinerary.apply_run_to(target, "fft_1")

    assert run.key == "fft_1"
    assert target.region_start == 2.0
    assert target.resample_taps_idx == 5
    assert target.cqt_algorithm_idx == 1
    assert target.hop_length == 1024
    assert target.cqt_filter_scale == 1.25


def test_source_panel_builds_typed_analysis_itinerary():
    from bass_viewer import SourcePanel

    panel = SourcePanel(input_dir=ROOT, output_root=ROOT)
    panel._add_analysis_module("fft")
    itinerary = panel.build_analysis_itinerary()

    assert len(itinerary.runs) == len(panel._analysis_itinerary)
    assert [run.engine for run in itinerary.runs].count("fft") == 2
    assert itinerary.metadata["source"] == "SourcePanel"


def test_analysis_inventory_roundtrip_and_backcompat_shape():
    from analysis_itinerary import AnalysisArtifact, AnalysisDatasetRecord, AnalysisInventory

    inv = AnalysisInventory(datasets=[
        AnalysisDatasetRecord(
            dataset_key="fft:abcd1234",
            engine="fft",
            algorithm="stft",
            run_key="abcd1234",
            settings_hash="abcd1234",
            folder="demo_analysis",
            artifacts=[AnalysisArtifact(kind="npz", path="cqt_data_abcd1234.npz")],
            settings={"hop_length": 256},
        )
    ])
    restored = AnalysisInventory.from_json(inv.to_json())
    assert restored.datasets[0].dataset_key == "fft:abcd1234"
    assert restored.datasets[0].artifacts[0].path == "cqt_data_abcd1234.npz"

    legacy = {
        "schema": "analysis_inventory_v1",
        "engines": {
            "fft": {
                "runs": [
                    {"run_key": "abc", "algorithm": "librosa", "npz": "cqt_data_abc.npz"}
                ]
            }
        },
    }
    restored_legacy = AnalysisInventory.from_dict(legacy)
    assert restored_legacy.datasets[0].engine == "fft"
    assert restored_legacy.datasets[0].artifacts[0].path == "cqt_data_abc.npz"


def test_analysis_time_range_fraction_span_handles_full_and_partial():
    from analysis_itinerary import AnalysisTimeRange

    partial = AnalysisTimeRange(start_sec=5.0, end_sec=15.0, total_sec=20.0)
    full = AnalysisTimeRange(start_sec=0.0, end_sec=0.0, total_sec=20.0)

    assert partial.as_fraction_span() == (0.25, 0.75)
    assert full.as_fraction_span() == (0.0, 1.0)


def test_source_panel_projects_itinerary_demands():
    from bass_viewer import SourcePanel

    panel = SourcePanel(input_dir=ROOT, output_root=ROOT)
    panel._wav_sr = 48000
    panel._wav_n_samples = 48000 * 20
    panel._wav_duration = 20.0
    fft_key = next(item["key"] for item in panel._analysis_itinerary if item["engine"] == "fft")
    panel._module_item(fft_key)["start_sec"] = 2.0
    panel._module_item(fft_key)["end_sec"] = 6.0

    demand = panel._project_itinerary_demands()

    assert len(demand["rows"]) == len(panel._analysis_itinerary)
    assert demand["total_bytes"] > 0
    fft_row = next(row for row in demand["rows"] if row["engine"] == "fft")
    assert fft_row["duration_sec"] == 4.0
    assert "x" in fft_row["detail"]


def test_source_panel_analysis_run_output_dir_is_shared_for_fft_only():
    from analysis_itinerary import AnalysisRun, FFTAnalysisSettings, FilterbankAnalysisSettings
    from bass_viewer import SourcePanel

    panel = SourcePanel(input_dir=ROOT, output_root=ROOT)
    fft_run = AnalysisRun(key="fft_1", engine="fft", settings=FFTAnalysisSettings())
    fb_run = AnalysisRun(key="fb_1", engine="fb", settings=FilterbankAnalysisSettings())

    assert panel._analysis_run_output_dir("demo_analysis", fft_run) == "demo_analysis"
    assert panel._analysis_run_output_dir("demo_analysis", fb_run).endswith("fb_fb_1")
