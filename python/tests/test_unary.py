"""
Tests for Unary activation operators (relu, sigmoid, silu, gelu, softplus, tanh)
integration in InfiniTensor.

Covers three levels:
1. Direct GraphBuilder Python API (gb.{op}())
2. PyTorch FX translator
3. Numerical correctness against PyTorch reference
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from infinitensor import TorchFXTranslator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(shape=(3, 4), dtype="float32"):
    """Return a random float32 numpy array and the corresponding torch tensor."""
    arr = np.random.randn(*shape).astype(dtype)
    return arr, torch.as_tensor(arr)


def _run_fx(runtime, model, inputs):
    """FX-translate a model, run it, and return numpy outputs."""
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inputs)
    try:
        translator.run(inputs)
    except RuntimeError as e:
        if "): 5" in str(e):
            pytest.skip(f"Op not supported on current device: {e}")
        raise
    return [o.numpy() for o in translator.get_outputs()]


# ---------------------------------------------------------------------------
# Relu
# ---------------------------------------------------------------------------

def test_relu_direct_api(runtime, torch_rng_seed):
    """gb.relu(x) returns a non-None tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([3, 4]), pit.dtype_from_string("float32"))
    y = gb.relu(x)
    assert y is not None, "gb.relu() should return an output tensor"


def test_relu_direct_api_with_output(runtime, torch_rng_seed):
    """gb.relu(x, Y=y_pre) returns the pre-allocated tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y = gb.relu(x, Y=y_pre)
    assert y is not None


def test_relu_fx_basic(runtime, torch_rng_seed):
    """FX translation of torch.relu matches PyTorch reference."""
    class ReluModel(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    arr, inp = _make_input()
    outputs = _run_fx(runtime, ReluModel(), [inp])

    assert len(outputs) == 1
    ref = torch.relu(inp).numpy()
    np.testing.assert_allclose(outputs[0], ref, atol=1e-5,
                               err_msg="relu FX result differs from PyTorch")


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def test_sigmoid_direct_api(runtime, torch_rng_seed):
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([3, 4]), pit.dtype_from_string("float32"))
    y = gb.sigmoid(x)
    assert y is not None


def test_sigmoid_direct_api_with_output(runtime, torch_rng_seed):
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y = gb.sigmoid(x, Y=y_pre)
    assert y is not None


def test_sigmoid_fx_basic(runtime, torch_rng_seed):
    class SigmoidModel(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    arr, inp = _make_input()
    outputs = _run_fx(runtime, SigmoidModel(), [inp])

    assert len(outputs) == 1
    ref = torch.sigmoid(inp).numpy()
    np.testing.assert_allclose(outputs[0], ref, atol=1e-5,
                               err_msg="sigmoid FX result differs from PyTorch")


# ---------------------------------------------------------------------------
# SiLU
# ---------------------------------------------------------------------------

def test_silu_direct_api(runtime, torch_rng_seed):
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([3, 4]), pit.dtype_from_string("float32"))
    y = gb.silu(x)
    assert y is not None


def test_silu_direct_api_with_output(runtime, torch_rng_seed):
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y = gb.silu(x, Y=y_pre)
    assert y is not None


def test_silu_fx_basic(runtime, torch_rng_seed):
    class SiluModel(torch.nn.Module):
        def forward(self, x):
            return F.silu(x)

    arr, inp = _make_input()
    outputs = _run_fx(runtime, SiluModel(), [inp])

    assert len(outputs) == 1
    ref = F.silu(inp).numpy()
    np.testing.assert_allclose(outputs[0], ref, atol=1e-5,
                               err_msg="silu FX result differs from PyTorch")


# ---------------------------------------------------------------------------
# GELU
# ---------------------------------------------------------------------------

def test_gelu_direct_api(runtime, torch_rng_seed):
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([3, 4]), pit.dtype_from_string("float32"))
    y = gb.gelu(x)
    assert y is not None


def test_gelu_direct_api_with_output(runtime, torch_rng_seed):
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y = gb.gelu(x, Y=y_pre)
    assert y is not None


def test_gelu_fx_basic(runtime, torch_rng_seed):
    class GeluModel(torch.nn.Module):
        def forward(self, x):
            return F.gelu(x)

    arr, inp = _make_input()
    outputs = _run_fx(runtime, GeluModel(), [inp])

    assert len(outputs) == 1
    ref = F.gelu(inp).numpy()
    np.testing.assert_allclose(outputs[0], ref, atol=1e-5,
                               err_msg="gelu FX result differs from PyTorch")


# ---------------------------------------------------------------------------
# Softplus
# ---------------------------------------------------------------------------

def test_softplus_direct_api(runtime, torch_rng_seed):
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([3, 4]), pit.dtype_from_string("float32"))
    y = gb.softplus(x)
    assert y is not None


def test_softplus_direct_api_with_output(runtime, torch_rng_seed):
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y = gb.softplus(x, Y=y_pre)
    assert y is not None


def test_softplus_fx_basic(runtime, torch_rng_seed):
    class SoftplusModel(torch.nn.Module):
        def forward(self, x):
            return F.softplus(x)

    arr, inp = _make_input()
    outputs = _run_fx(runtime, SoftplusModel(), [inp])

    assert len(outputs) == 1
    ref = F.softplus(inp).numpy()
    np.testing.assert_allclose(outputs[0], ref, atol=1e-5,
                               err_msg="softplus FX result differs from PyTorch")


# ---------------------------------------------------------------------------
# Tanh
# ---------------------------------------------------------------------------

def test_tanh_direct_api(runtime, torch_rng_seed):
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([3, 4]), pit.dtype_from_string("float32"))
    y = gb.tanh(x)
    assert y is not None


def test_tanh_direct_api_with_output(runtime, torch_rng_seed):
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y = gb.tanh(x, Y=y_pre)
    assert y is not None


def test_tanh_fx_basic(runtime, torch_rng_seed):
    class TanhModel(torch.nn.Module):
        def forward(self, x):
            return torch.tanh(x)

    arr, inp = _make_input()
    outputs = _run_fx(runtime, TanhModel(), [inp])

    assert len(outputs) == 1
    ref = torch.tanh(inp).numpy()
    np.testing.assert_allclose(outputs[0], ref, atol=1e-5,
                               err_msg="tanh FX result differs from PyTorch")


if __name__ == "__main__":
    import sys

    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(0 if exit_code == 0 else 1)
