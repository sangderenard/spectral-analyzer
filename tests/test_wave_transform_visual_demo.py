from pathlib import Path

from PIL import Image
import pytest

from wave_transform_visual_demo import (
    render_sequence,
    run_live,
    run_transport_live,
)


def test_visual_demo_uses_production_transform_and_writes_sequence(tmp_path):
    result = render_sequence(tmp_path, size=16, frames=2, scale=1)
    assert len(result["frames"]) == 3
    assert result["max_roundtrip_error"] < 2.0e-5
    for value in (*result["frames"], result["summary"]):
        path = Path(value)
        assert path.is_file()
        with Image.open(path) as image:
            assert image.width > 0
            assert image.height > 0


def test_live_mode_validates_before_opening_a_context():
    with pytest.raises(ValueError, match="power of two"):
        run_live(size=15)
    with pytest.raises(ValueError, match="cycle_steps"):
        run_live(cycle_steps=0)
    with pytest.raises(ValueError, match="fps"):
        run_live(fps=0)
    with pytest.raises(ValueError, match="panel_size"):
        run_transport_live(panel_size=32)
