"""Opt-in tiny GPU numerical/allocation smoke. Run only in an idle GPU slot.

This creates no scene and never calls the production server. It allocates at
most about 13 MiB history, not a near-cap stress allocation.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    test = runpy.run_path(str(ROOT / "tests/test_sensor_history.py"))
    test["test_actual_gpu_ring_wrap_expiry_reset_and_layer_independence"]()
    from camera_designer.sensor_history import SensorHistoryLayout, replace_history_buffers
    from OpenGL import GL
    owner = SimpleNamespace(_history_buffers=[], _history_samples=0,
                            _history_bytes=0, _history_checkpoint=False)
    allocations = []
    try:
        for samples, checkpoint in ((1024, True), (4096, True), (16, False)):
            layout = SensorHistoryLayout(16, 16, 3, samples, checkpoint)
            replace_history_buffers(owner, layout)
            assert owner._history_bytes == layout.nbytes
            assert owner._history_samples == samples
            assert owner._history_checkpoint is checkpoint
            allocations.append(dict(samples=samples, checkpoint=checkpoint,
                                    allocated_bytes=owner._history_bytes,
                                    buffer_limit=owner._history_block_limit))
    finally:
        replace_history_buffers(owner, None)
    assert not owner._history_buffers and owner._history_bytes == 0
    receipt = dict(numeric_ring_checkpoint_reset_resize="PASS", allocations=allocations,
                   cleanup="PASS", renderer=GL.glGetString(GL.GL_RENDERER).decode())
    target = ROOT / "build/dot-live-history-20261109/gpu-check.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
