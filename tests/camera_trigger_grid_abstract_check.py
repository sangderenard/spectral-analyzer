"""Run the original grid fixture with no available PyTorch dependency.

Requires Turing on PYTHONPATH; runs headlessly via the original fixture.
"""
import importlib.abc
import os
from pathlib import Path
import runpy
import sys


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("Torch deliberately unavailable")


if __name__ == "__main__":
    os.environ["SPECTRAL_TENSOR_BACKEND"] = "abstract"
    sys.meta_path.insert(0, NoTorch())
    runpy.run_path(str(Path(__file__).with_name("camera_trigger_grid_bridge_check.py")), run_name="__main__")
    assert "torch" not in sys.modules
    print("PASS: original grid and radio graph without Torch")
