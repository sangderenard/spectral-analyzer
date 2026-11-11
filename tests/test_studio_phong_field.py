"""Run the real field shader outside pytest GUI import stubs."""
import json
import subprocess
import sys
from pathlib import Path


def test_hidden_gpu_field_mapping_preserves_legacy_and_signed_rgb():
    probe = Path(__file__).with_name("studio_phong_field_probe.py")
    result = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["status"] == "PASS"
    assert receipt["checks"] == 45
    assert receipt["signed_rgb_preserved"]
