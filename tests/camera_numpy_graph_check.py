"""Exercise the public materializer boundary using the emitted NumPy graph."""
import os
import sys
from pathlib import Path

os.environ["SPECTRAL_TENSOR_BACKEND"] = "abstract"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.tensors import AbstractTensor
from graph_solver import TensorNode, TensorEdge
from network_materializer import compile_nodes

nodes = [TensorNode(key="radio"), TensorNode(key="light")]
edges = [TensorEdge(src_key="radio", dst_key="light", weight=0.5)]
graph = compile_nodes(nodes, edges, 1.0, output_node_keys=["light"], execution="numpy")
result = graph.process_window({"radio": AbstractTensor.tensor([0.4 + 0.2j], dtype="complex128")}, n_frames=1)
assert isinstance(result["light"], AbstractTensor)
assert abs(result["light"].item() - (0.2 + 0.1j)) < 1e-12
contract = graph.transport_contract()
assert contract["execution"] == "numpy" and contract["source_digest"]
try:
    graph.process_sample({})
except ValueError as error:
    assert "process_window" in str(error)
else:
    raise AssertionError("training/sample operation silently accepted")
try:
    compile_nodes(nodes, [TensorEdge(src_key="radio", dst_key="light", weight=1.0, delay_samples=1)], 1.0, execution="numpy")
except ValueError as error:
    assert "zero-delay" in str(error)
else:
    raise AssertionError("unsupported deployment path silently accepted")
assert "torch" not in sys.modules
print("PASS emitted graph public tensor boundary, digest and capability refusal")

