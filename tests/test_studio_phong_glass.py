"""Exercise the real shader variants without pytest's GUI import stubs."""
import json
import subprocess
import sys
from pathlib import Path


def test_hidden_gpu_glass_fresnel_compositing_preserves_opaque_pixels():
    probe=Path(__file__).with_name("studio_phong_glass_probe.py")
    result=subprocess.run([sys.executable,str(probe)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["status"]=="PASS"
    assert len(receipt["comparisons"])==4
