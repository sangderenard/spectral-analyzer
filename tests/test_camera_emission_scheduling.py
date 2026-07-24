from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from thick_lens_focus_lab import ForwardCppLensBench


class _RecordingTracer:
    def __init__(self) -> None:
        self.submissions: list[tuple] = []
        self.light_family_markers = 0

    def submit_emissive_triangles(self, *args):
        self.submissions.append(args)
        return 1 if float(args[2]) > 0.0 else 0

    def signal_flash_dispatched(self) -> None:
        self.light_family_markers += 1


def _bench() -> ForwardCppLensBench:
    bench = ForwardCppLensBench.__new__(ForwardCppLensBench)
    bench.compute_mode = "cpu"
    bench.scene = SimpleNamespace(flash_modifier=None)
    bench.tracer = _RecordingTracer()
    bench._flash_emitter_tri_ids_i32 = np.asarray([3], np.int32)
    bench._natural_emitter_tri_ids_i32 = np.asarray([7], np.int32)
    bench._emitter_tri_ids_i32 = bench._flash_emitter_tri_ids_i32
    bench.emitter_amp_gain = 1.0
    bench._min_amplitude = 0.0
    bench._uv_emitter_ray_count = lambda: 0
    bench._emitter_interaction_target = lambda: (
        0.0, 0.0, 0.0, 0.0, 0, -1.0, 0.0, 0.0, 0.0
    )
    bench._ensure_drain_loop = lambda: None
    return bench


def test_disabled_camera_flash_does_not_suppress_natural_light_family():
    bench = _bench()

    submitted = bench.trace_forward(
        8,
        11,
        max_bounces=2,
        exposure_weight=0.0,
        flash_exposure_weight=0.0,
        natural_exposure_weight=0.25,
    )

    assert submitted == 1
    assert len(bench.tracer.submissions) == 2
    assert bench.tracer.submissions[0][2] == 0.0
    assert bench.tracer.submissions[1][2] == pytest.approx(0.25)
    assert bench.tracer.light_family_markers == 1


def test_legacy_exposure_weight_still_applies_to_both_emitter_families():
    bench = _bench()

    submitted = bench.trace_forward(
        8, 13, max_bounces=2, exposure_weight=0.4
    )

    assert submitted == 2
    assert [args[2] for args in bench.tracer.submissions] == pytest.approx(
        [0.4, 0.4]
    )
    assert bench.tracer.light_family_markers == 1
