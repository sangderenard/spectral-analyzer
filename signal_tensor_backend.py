"""Select the existing signal tensor substrate once per process.

Default remains PyTorch. Set SPECTRAL_TENSOR_BACKEND=abstract before importing
signal modules for the AbstractTensor/AbstractNN path. Never switch a live graph.
No numerical operations or solver algorithms are implemented here.
"""
import os

BACKEND = os.environ.get("SPECTRAL_TENSOR_BACKEND", "torch")
if BACKEND == "abstract":
    from src.common.tensors import AbstractTensor as tensor
    from src.common.tensors import AbstractTensor as Tensor
    from src.common.tensors import abstract_nn as nn
    COMPLEX = "complex128"
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    INT64 = "int64"
    BOOL = "bool"
    CPU = "cpu"
elif BACKEND == "torch":
    import torch as tensor
    from torch import Tensor
    from torch import nn
    COMPLEX = tensor.complex128
    FLOAT64 = tensor.float64
    FLOAT32 = tensor.float32
    INT64 = tensor.int64
    BOOL = tensor.bool
    CPU = tensor.device("cpu")
else:
    raise ValueError("SPECTRAL_TENSOR_BACKEND must be torch or abstract")
