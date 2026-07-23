from __future__ import annotations

import camera_software.surface_scan_preview as surface_scan_preview
from camera_software.surface_scan_preview import SurfaceScanPose, SurfaceScanPreview


class _FakeTracer:
    n_bands = 3
    def __init__(self):
        self.calls = []
        self.in_flight = 0
        self.generation = 0

    def set_gl_display_hdc(self, value): self.calls.append(("hdc", value))
    def set_gl_display_hglrc(self, value): self.calls.append(("hglrc", value))
    def clear_scale_contexts(self): self.calls.append(("clear-contexts",))
    def set_gpu_skip_record_readback(self, value): self.calls.append(("skip-readback", value))
    def configure_sensor_image(self, *args): self.calls.append(("image", args))
    def configure_sensor_pose(self, *args): self.calls.append(("pose", args))
    def ensure_pipeline(self, **kwargs): self.calls.append(("pipeline", kwargs))
    def in_flight_count(self): return self.in_flight
    def submit_surface_scan(self, **kwargs):
        self.calls.append(("submit", kwargs))
        self.in_flight = 1
    def surface_scan_texture_info(self):
        if not self.generation:
            return {}
        return {
            "texture_id": 17, "width": 256, "height": 256,
            "depth": 1, "generation": self.generation,
        }
    def stop_pipeline(self): self.calls.append(("stop",))


def _pose(x=0.0):
    return SurfaceScanPose(
        sensor_center=(x, 0.0, -0.0024),
        aperture_center=(0.0, 0.0, 0.006),
        sensor_half_width=0.028,
        sensor_half_height=0.028,
        resolution=256,
    )


def test_surface_scan_is_one_idle_native_submission_and_no_wave_contexts(tmp_path):
    tracer = _FakeTracer()
    preview = SurfaceScanPreview(
        tracer, _pose(), shader_dir=str(tmp_path),
        display_hglrc=11, display_hdc=12,
    )

    preview.poll()
    preview.poll()
    assert [call[0] for call in tracer.calls].count("submit") == 1
    assert ("clear-contexts",) in tracer.calls
    assert ("skip-readback", True) in tracer.calls
    pipeline = next(call[1] for call in tracer.calls if call[0] == "pipeline")
    assert pipeline["use_gpu_compute"] is True
    assert pipeline["gpu_all_stages"] is True
    assert pipeline["max_children"] == 1

    tracer.in_flight = 0
    tracer.generation = 1
    assert preview.poll()["texture_id"] == 17


def test_pose_change_invalidates_cached_center_site_rays(tmp_path):
    tracer = _FakeTracer()
    preview = SurfaceScanPreview(tracer, _pose(), shader_dir=str(tmp_path))
    preview.poll()
    tracer.in_flight = 0
    tracer.generation = 1
    preview.poll()

    preview.configure(_pose(x=0.001))
    preview.poll()
    assert [call[0] for call in tracer.calls].count("submit") == 2
    assert [call[0] for call in tracer.calls].count("pose") == 2


def test_close_restores_the_callers_display_context(tmp_path, monkeypatch):
    tracer = _FakeTracer()
    restored = []
    monkeypatch.setattr(
        surface_scan_preview, "current_wgl_handles", lambda: (101, 202),
    )
    monkeypatch.setattr(
        surface_scan_preview, "restore_wgl_context",
        lambda hglrc, hdc: restored.append((hglrc, hdc)) or True,
    )
    preview = SurfaceScanPreview(tracer, _pose(), shader_dir=str(tmp_path))
    restored.clear()

    preview.close()

    assert ("stop",) in tracer.calls
    assert restored == [(101, 202)]
