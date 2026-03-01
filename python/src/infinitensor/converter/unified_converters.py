import torch.nn as nn
from .registry import registry

#https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml

@registry.register("matmul","default")
def convert_matmul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.gemm(a, b, None)

@registry.register("add","Tensor")
def convert_add(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.add(a, b, None)

@registry.register("mul","Tensor")
def convert_add(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.add(a, b, None)

@registry.register("sub","Tensor")
def convert_add(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.add(a, b, None)

@registry.register("clamp", "default")
def convert_clamp_default(translator, node):
    """Handle torch.clamp(x, min=scalar, max=scalar) with scalar bounds."""
    import numpy as np
    import ctypes
    from pyinfinitensor import ShapeExpr, dtype_from_string
    x = translator.tensors[node.args[0]]
    # min/max may be positional args or keyword args
    min_v = node.args[1] if len(node.args) > 1 else node.kwargs.get('min', float('-inf'))
    max_v = node.args[2] if len(node.args) > 2 else node.kwargs.get('max', float('inf'))
    if min_v is None:
        min_v = float('-inf')
    if max_v is None:
        max_v = float('inf')
    # Get dtype from FX node output metadata
    dtype = dtype_from_string(str(node.meta["val"].dtype))
    # Create single-element constant tensors for min and max
    min_arr = np.array([float(min_v)], dtype=np.float32)
    max_arr = np.array([float(max_v)], dtype=np.float32)
    min_t = translator.builder.tensor(ShapeExpr([1]), dtype)
    max_t = translator.builder.tensor(ShapeExpr([1]), dtype)
    min_t.set_data(min_arr.ctypes.data_as(ctypes.c_void_p).value, translator.runtime)
    max_t.set_data(max_arr.ctypes.data_as(ctypes.c_void_p).value, translator.runtime)
    translator.tensors[node] = translator.builder.clip(x, min_t, max_t, None)

@registry.register("clamp", "Tensor")
def convert_clamp_tensor(translator, node):
    """Handle torch.clamp(x, min=tensor, max=tensor) with tensor bounds."""
    x       = translator.tensors[node.args[0]]
    min_val = translator.tensors[node.args[1]]
    max_val = translator.tensors[node.args[2]]
    translator.tensors[node] = translator.builder.clip(x, min_val, max_val, None)