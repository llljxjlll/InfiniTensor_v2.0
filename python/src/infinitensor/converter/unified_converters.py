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


# ---------------------------------------------------------------------------
# Conv
# ---------------------------------------------------------------------------

@registry.register("convolution", "default")
def convert_convolution(translator, node):
    """Handle torch.nn.functional.conv2d / torch._C._nn.convolution."""
    # args: input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    x      = translator.tensors[node.args[0]]
    w      = translator.tensors[node.args[1]]
    b_node = node.args[2]
    b      = translator.tensors[b_node] if b_node is not None else None

    strides   = list(node.args[3])   # e.g. [1, 1]
    padding   = list(node.args[4])   # e.g. [0, 0]
    dilations = list(node.args[5])   # e.g. [1, 1]
    # args[6] = transposed (bool), args[7] = output_padding, args[8] = groups
    # Only non-transposed conv is supported here.
    translator.tensors[node] = translator.builder.conv(
        x, w, b, padding, strides, dilations, None
    )

@registry.register("conv2d", "default")
def convert_conv2d(translator, node):
    """Handle torch.nn.functional.conv2d."""
    x      = translator.tensors[node.args[0]]
    w      = translator.tensors[node.args[1]]
    b_node = node.args[2] if len(node.args) > 2 else node.kwargs.get('bias', None)
    b      = translator.tensors[b_node] if b_node is not None else None

    stride    = list(node.kwargs.get('stride',   node.args[3] if len(node.args) > 3 else [1, 1]))
    padding   = list(node.kwargs.get('padding',  node.args[4] if len(node.args) > 4 else [0, 0]))
    dilation  = list(node.kwargs.get('dilation', node.args[5] if len(node.args) > 5 else [1, 1]))
    translator.tensors[node] = translator.builder.conv(
        x, w, b, padding, stride, dilation, None
    )


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

@registry.register("layer_norm", "default")
def convert_layer_norm(translator, node):
    """Handle torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)."""
    x      = translator.tensors[node.args[0]]
    # args[1] = normalized_shape (list), args[2] = weight, args[3] = bias, args[4] = eps
    w_node = node.args[2] if len(node.args) > 2 else node.kwargs.get('weight', None)
    b_node = node.args[3] if len(node.args) > 3 else node.kwargs.get('bias',   None)
    eps    = node.args[4] if len(node.args) > 4 else node.kwargs.get('eps',    1e-5)
    weight = translator.tensors[w_node]
    bias   = translator.tensors[b_node]
    translator.tensors[node] = translator.builder.layernorm(x, weight, bias, float(eps), None)


# ---------------------------------------------------------------------------
# RmsNorm
# ---------------------------------------------------------------------------

@registry.register("rms_norm", "default")
def convert_rms_norm(translator, node):
    """Handle torch.nn.functional.rms_norm(x, normalized_shape, weight, eps)."""
    x      = translator.tensors[node.args[0]]
    w_node = node.args[2] if len(node.args) > 2 else node.kwargs.get('weight', None)
    eps    = node.args[3] if len(node.args) > 3 else node.kwargs.get('eps',    1e-6)
    weight = translator.tensors[w_node]
    translator.tensors[node] = translator.builder.rmsnorm(x, weight, float(eps), None)


# ---------------------------------------------------------------------------
# Softmax / LogSoftmax
# ---------------------------------------------------------------------------

@registry.register("softmax", "int")
def convert_softmax(translator, node):
    """Handle torch.nn.functional.softmax(x, dim=..., ...)."""
    x   = translator.tensors[node.args[0]]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', -1)
    if dim is None:
        dim = -1
    translator.tensors[node] = translator.builder.softmax(x, int(dim), None)

@registry.register("log_softmax", "int")
def convert_log_softmax(translator, node):
    """Handle torch.nn.functional.log_softmax(x, dim=..., ...)."""
    x = translator.tensors[node.args[0]]
    translator.tensors[node] = translator.builder.logsoftmax(x, None)


# ---------------------------------------------------------------------------
# LpNorm  (torch.nn.functional.normalize maps to lp_norm internally)
# ---------------------------------------------------------------------------

@registry.register("lp_pool2d", "default")
def convert_lp_pool2d(translator, node):
    """Handle torch.nn.functional.lp_pool2d — maps to LpNorm on last dim."""
    x    = translator.tensors[node.args[0]]
    p    = int(node.args[1]) if len(node.args) > 1 else int(node.kwargs.get('norm_type', 2))
    axis = -1
    translator.tensors[node] = translator.builder.lpnorm(x, axis, p, 1e-12, None)