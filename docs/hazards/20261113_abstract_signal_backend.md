# Optional AbstractTensor signal graph backend

Set SPECTRAL_TENSOR_BACKEND=abstract before importing signal modules, with Turing
on PYTHONPATH. The default remains torch. signal_tensor_backend only selects
existing numeric/NN implementations and their dtype/device vocabulary; it does
not implement arithmetic, solver rules, a new mixer or a replacement graph.
Turing commit 284ff7f8 provides the required AbstractNN named state facilities.

Import closure from the real CameraTriggerGrid reaches analytic_driver,
analytic_shared, parametric editors, performer engine and legacy routing module.
Their imports and dtype spellings now use the same selection. Their algorithms
are unchanged. Do not interpret this as validation of every UI synthesis feature
under the abstract backend; the tested surface is the camera grid/radio graph.

Verification:
- python tests/camera_trigger_grid_abstract_check.py (Turing on PYTHONPATH): PASS,
  original grid image, events, restore, radio gains, stale reset and delay refusal.
  The test blocks all torch imports and asserts torch is absent from sys.modules.
- SPECTRAL_TENSOR_BACKEND=torch; python tests/camera_trigger_grid_bridge_check.py:
  PASS, unchanged original grid behavior.
- SPECTRAL_TENSOR_BACKEND=torch; python -m pytest tests/test_camera_trigger_bus.py
  tests/test_graph_solver_delays.py tests/test_voice_graph_solver.py -q:
  43 passed in 1.89 s.
- Isolated actual GraphSolver/materializer abstract tests passed chain,
  linear feedback, one-sample delay and direct nonlinear square transform using
  native complex128 NumPy tensors (not ComplexPrecision expansion).

Not deployed. No shader render/link or dependency installation. The service must
still receive these source changes, Turing import availability and opt-in backend
setting through its supported deployment configuration. LLVM compilation remains
blocked by a compiler SSA KeyError under separate investigation; this commit is
an interpreted real-graph path, not a compiled product.

## Actual service interpreter verification

C:/dev/Powershell/srv/python/python.exe (Python 3.11.18, Torch absent) ran
`tests/camera_trigger_grid_abstract_check.py` successfully with Turing on
PYTHONPATH and SPECTRAL_TENSOR_BACKEND=abstract. NumPy, Pygame, SciPy and Pillow
were already installed. Eight actual GraphSolver schedules ran. This removes
the prior Torch-free-process blocker for the tested grid path, without a new
worker or solver. Deployment environment/import wiring and source publication
remain; no live server was started or restarted by this test.
