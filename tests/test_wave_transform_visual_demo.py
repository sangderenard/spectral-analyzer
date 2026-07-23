from pathlib import Path

from PIL import Image

from wave_transform_visual_demo import render_sequence


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
