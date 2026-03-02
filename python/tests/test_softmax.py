"""
Tests for the Softmax and LogSoftmax operator integration in InfiniTensor.

Covers three levels:
1. Direct GraphBuilder Python API (gb.softmax(), gb.logsoftmax())
2. PyTorch FX translator (F.softmax / F.log_softmax → SoftmaxObj)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from infinitensor import TorchFXTranslator


# ---------------------------------------------------------------------------
# Task 1: Direct Python API tests — Softmax
# ---------------------------------------------------------------------------


def test_softmax_direct_api(runtime, torch_rng_seed):
    """Verify that GraphBuilder.softmax() binding is usable and returns a tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    y = gb.softmax(x, axis=-1)
    assert y is not None, "gb.softmax() should return an output tensor"
    print("✅ Direct softmax API test passed!")


def test_softmax_direct_api_with_output(runtime, torch_rng_seed):
    """Verify that gb.softmax() accepts a pre-allocated output tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    y = gb.softmax(x, axis=-1, Y=y_pre)
    assert y is not None, "gb.softmax() with explicit Y should return that tensor"
    print("✅ Direct softmax API with explicit output test passed!")


# ---------------------------------------------------------------------------
# Task 1: Direct Python API tests — LogSoftmax
# ---------------------------------------------------------------------------


def test_logsoftmax_direct_api(runtime, torch_rng_seed):
    """Verify that GraphBuilder.logsoftmax() binding is usable and returns a tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    y = gb.logsoftmax(x)
    assert y is not None, "gb.logsoftmax() should return an output tensor"
    print("✅ Direct logsoftmax API test passed!")


def test_logsoftmax_direct_api_with_output(runtime, torch_rng_seed):
    """Verify that gb.logsoftmax() accepts a pre-allocated output tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    y = gb.logsoftmax(x, Y=y_pre)
    assert y is not None, "gb.logsoftmax() with explicit Y should return that tensor"
    print("✅ Direct logsoftmax API with explicit output test passed!")


# ---------------------------------------------------------------------------
# Task 2: PyTorch FX translator tests — Softmax
# ---------------------------------------------------------------------------


def test_softmax_fx_basic(runtime, device_type, torch_rng_seed):
    """Verify FX translation of F.softmax."""
    from infinitensor import DeviceType as DT
    if device_type == DT.CPU:
        pytest.skip("InfiniCore Softmax has no CPU backend; run with --device=cuda")

    class SoftmaxModel(nn.Module):
        def forward(self, x):
            return F.softmax(x, dim=-1)

    model = SoftmaxModel()
    inp = [torch.as_tensor(np.random.randn(2, 4).astype("float32"))]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inp)
    translator.run(inp)
    outputs = translator.get_outputs()

    ref = F.softmax(inp[0], dim=-1).numpy()
    assert len(outputs) == 1
    assert outputs[0].shape == (2, 4), f"Unexpected shape: {outputs[0].shape}"
    np.testing.assert_allclose(
        outputs[0].numpy(),
        ref,
        atol=1e-5,
        err_msg="Softmax FX result differs from PyTorch",
    )
    print("✅ FX Softmax test passed!")


def test_softmax_fx_axis0(runtime, device_type, torch_rng_seed):
    """Verify FX translation of F.softmax along axis 0."""
    from infinitensor import DeviceType as DT
    if device_type == DT.CPU:
        pytest.skip("InfiniCore Softmax has no CPU backend; run with --device=cuda")

    class SoftmaxAxisModel(nn.Module):
        def forward(self, x):
            return F.softmax(x, dim=0)

    model = SoftmaxAxisModel()
    inp = [torch.as_tensor(np.random.randn(4, 3).astype("float32"))]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inp)
    translator.run(inp)
    outputs = translator.get_outputs()

    ref = F.softmax(inp[0], dim=0).numpy()
    assert len(outputs) == 1
    np.testing.assert_allclose(
        outputs[0].numpy(),
        ref,
        atol=1e-5,
        err_msg="Softmax axis=0 FX result differs from PyTorch",
    )
    print("✅ FX Softmax axis=0 test passed!")


# ---------------------------------------------------------------------------
# Task 2: PyTorch FX translator tests — LogSoftmax
# ---------------------------------------------------------------------------


def test_logsoftmax_fx_basic(runtime, torch_rng_seed):
    """Verify FX translation of F.log_softmax."""

    class LogSoftmaxModel(nn.Module):
        def forward(self, x):
            return F.log_softmax(x, dim=-1)

    model = LogSoftmaxModel()
    inp = [torch.as_tensor(np.random.randn(2, 4).astype("float32"))]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inp)
    translator.run(inp)
    outputs = translator.get_outputs()

    ref = F.log_softmax(inp[0], dim=-1).numpy()
    assert len(outputs) == 1
    assert outputs[0].shape == (2, 4), f"Unexpected shape: {outputs[0].shape}"
    np.testing.assert_allclose(
        outputs[0].numpy(),
        ref,
        atol=1e-5,
        err_msg="LogSoftmax FX result differs from PyTorch",
    )
    print("✅ FX LogSoftmax test passed!")


if __name__ == "__main__":
    import sys
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(0 if exit_code == 0 else 1)
