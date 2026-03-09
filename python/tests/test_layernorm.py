"""
Tests for the LayerNorm operator integration in InfiniTensor.

Covers three levels:
1. Direct GraphBuilder Python API (gb.layernorm())
2. PyTorch FX translator (torch.nn.LayerNorm → LayerNormObj)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from infinitensor import TorchFXTranslator


# ---------------------------------------------------------------------------
# Task 1: Direct Python API tests
# ---------------------------------------------------------------------------


def test_layernorm_direct_api(runtime, torch_rng_seed):
    """Verify that GraphBuilder.layernorm() binding is usable and returns a tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    w = gb.tensor(pit.ShapeExpr([4]), pit.dtype_from_string("float32"))
    b = gb.tensor(pit.ShapeExpr([4]), pit.dtype_from_string("float32"))
    y = gb.layernorm(x, w, b)
    assert y is not None, "gb.layernorm() should return an output tensor"
    print("✅ Direct layernorm API test passed!")


def test_layernorm_direct_api_with_output(runtime, torch_rng_seed):
    """Verify that gb.layernorm() accepts a pre-allocated output tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    w = gb.tensor(pit.ShapeExpr([4]), pit.dtype_from_string("float32"))
    b = gb.tensor(pit.ShapeExpr([4]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    y = gb.layernorm(x, w, b, Y=y_pre)
    assert y is not None, "gb.layernorm() with explicit Y should return that tensor"
    print("✅ Direct layernorm API with explicit output test passed!")


# ---------------------------------------------------------------------------
# Task 2: PyTorch FX translator tests
# ---------------------------------------------------------------------------


def test_layernorm_fx_basic(runtime, torch_rng_seed):
    """Verify FX translation of torch.nn.LayerNorm."""

    class LayerNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(4)

        def forward(self, x):
            return self.ln(x)

    model = LayerNormModel().eval()
    inp = [torch.as_tensor(np.random.randn(2, 4).astype("float32"))]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inp)
    translator.run(inp)
    outputs = translator.get_outputs()

    ref = model(inp[0]).detach().numpy()
    assert len(outputs) == 1
    assert outputs[0].shape == (2, 4), f"Unexpected shape: {outputs[0].shape}"
    np.testing.assert_allclose(
        outputs[0].numpy(),
        ref,
        atol=1e-5,
        err_msg="LayerNorm FX result differs from PyTorch",
    )
    print("✅ FX LayerNorm test passed!")


def test_layernorm_fx_3d(runtime, torch_rng_seed):
    """Verify FX translation of torch.nn.LayerNorm on 3D input."""

    class LayerNormModel3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(8)

        def forward(self, x):
            return self.ln(x)

    model = LayerNormModel3D().eval()
    inp = [torch.as_tensor(np.random.randn(1, 4, 8).astype("float32"))]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inp)
    translator.run(inp)
    outputs = translator.get_outputs()

    ref = model(inp[0]).detach().numpy()
    assert len(outputs) == 1
    assert outputs[0].shape == ref.shape, f"Shape mismatch: {outputs[0].shape} vs {ref.shape}"
    np.testing.assert_allclose(
        outputs[0].numpy(),
        ref,
        atol=1e-5,
        err_msg="LayerNorm 3D FX result differs from PyTorch",
    )
    print("✅ FX LayerNorm 3D test passed!")


if __name__ == "__main__":
    import sys
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(0 if exit_code == 0 else 1)
