"""
Tests for the Clip operator integration in InfiniTensor.

Covers three levels:
1. Direct GraphBuilder Python API (gb.clip())
2. PyTorch FX translator with scalar min/max (torch.clamp → ClipObj)
3. PyTorch FX translator with tensor min/max (torch.clamp.Tensor → ClipObj)
"""

import ctypes

import numpy as np
import pytest
import torch

from infinitensor import TorchFXTranslator

# ---------------------------------------------------------------------------
# Task 1: Direct Python API tests
# ---------------------------------------------------------------------------


def test_clip_direct_api(runtime, torch_rng_seed):
    """Verify that GraphBuilder.clip() binding is usable and returns a tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([3, 4]), pit.dtype_from_string("float32"))
    mn = gb.tensor(pit.ShapeExpr([1]), pit.dtype_from_string("float32"))
    mx = gb.tensor(pit.ShapeExpr([1]), pit.dtype_from_string("float32"))
    y = gb.clip(x, mn, mx)
    assert y is not None, "gb.clip() should return an output tensor"
    print("✅ Direct clip API test passed!")


def test_clip_direct_api_with_output(runtime, torch_rng_seed):
    """Verify that gb.clip() accepts a pre-allocated output tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    mn = gb.tensor(pit.ShapeExpr([1]), pit.dtype_from_string("float32"))
    mx = gb.tensor(pit.ShapeExpr([1]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 3]), pit.dtype_from_string("float32"))
    y = gb.clip(x, mn, mx, Y=y_pre)
    assert y is not None, "gb.clip() with explicit Y should return that tensor"
    print("✅ Direct clip API with explicit output test passed!")


# ---------------------------------------------------------------------------
# Task 2: PyTorch FX translator tests
# ---------------------------------------------------------------------------


def test_clip_fx_scalar_minmax(runtime, torch_rng_seed):
    """
    Verify FX translation of torch.clamp(x, min=scalar, max=scalar).
    Also validates numerical correctness against PyTorch reference.
    """

    class ClampModel(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=-1.0, max=1.0)

    model = ClampModel()
    inp = [torch.as_tensor(np.random.randn(3, 4).astype("float32"))]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inp)
    translator.run(inp)
    outputs = translator.get_outputs()

    assert len(outputs) == 1, "Should produce exactly one output"
    assert outputs[0].shape == (3, 4), f"Unexpected shape: {outputs[0].shape}"

    ref = torch.clamp(inp[0], min=-1.0, max=1.0).numpy()
    np.testing.assert_allclose(
        outputs[0].numpy(),
        ref,
        atol=1e-5,
        err_msg="Clip FX scalar result differs from PyTorch",
    )
    print("✅ FX scalar clamp test passed!")


def test_clip_fx_scalar_one_sided_min(runtime, torch_rng_seed):
    """Verify FX translation of torch.clamp(x, min=scalar) (no upper bound)."""

    class ClampMinModel(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=0.0)

    model = ClampMinModel()
    inp = [torch.as_tensor(np.random.randn(4, 4).astype("float32"))]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inp)
    translator.run(inp)
    outputs = translator.get_outputs()

    assert len(outputs) == 1
    ref = torch.clamp(inp[0], min=0.0).numpy()
    np.testing.assert_allclose(
        outputs[0].numpy(),
        ref,
        atol=1e-5,
        err_msg="One-sided min clamp differs from PyTorch",
    )
    print("✅ FX one-sided min clamp test passed!")


def test_clip_fx_tensor_minmax(runtime, torch_rng_seed):
    """
    Verify FX translation of torch.clamp(x, min=tensor, max=tensor).
    """

    class ClampTensorModel(torch.nn.Module):
        def forward(self, x, mn, mx):
            return torch.clamp(x, min=mn, max=mx)

    model = ClampTensorModel()
    x = torch.as_tensor(np.random.randn(3, 4).astype("float32"))
    mn = torch.full((3, 4), -1.0)
    mx = torch.full((3, 4), 1.0)

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, [x, mn, mx])
    translator.run([x, mn, mx])
    outputs = translator.get_outputs()

    assert len(outputs) == 1
    assert outputs[0].shape == (3, 4), f"Unexpected shape: {outputs[0].shape}"
    ref = torch.clamp(x, min=mn, max=mx).numpy()
    np.testing.assert_allclose(
        outputs[0].numpy(),
        ref,
        atol=1e-5,
        err_msg="Clip FX tensor result differs from PyTorch",
    )
    print("✅ FX tensor clamp test passed!")


if __name__ == "__main__":
    import sys

    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(0 if exit_code == 0 else 1)
