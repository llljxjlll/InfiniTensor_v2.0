"""
Tests for the Conv operator integration in InfiniTensor.

Covers three levels:
1. Direct GraphBuilder Python API (gb.conv())
2. PyTorch FX translator (torch.nn.Conv2d → ConvObj)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from infinitensor import TorchFXTranslator


# ---------------------------------------------------------------------------
# Task 1: Direct Python API tests
# ---------------------------------------------------------------------------


def test_conv_direct_api(runtime, torch_rng_seed):
    """Verify that GraphBuilder.conv() binding is usable and returns a tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([1, 1, 5, 5]), pit.dtype_from_string("float32"))
    w = gb.tensor(pit.ShapeExpr([1, 1, 3, 3]), pit.dtype_from_string("float32"))
    y = gb.conv(x, w, None, [0, 0], [1, 1], [1, 1])
    assert y is not None, "gb.conv() should return an output tensor"
    print("✅ Direct conv API test passed!")


def test_conv_direct_api_with_output(runtime, torch_rng_seed):
    """Verify that gb.conv() accepts a pre-allocated output tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([1, 1, 5, 5]), pit.dtype_from_string("float32"))
    w = gb.tensor(pit.ShapeExpr([1, 1, 3, 3]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([1, 1, 3, 3]), pit.dtype_from_string("float32"))
    y = gb.conv(x, w, None, [0, 0], [1, 1], [1, 1], Y=y_pre)
    assert y is not None, "gb.conv() with explicit Y should return that tensor"
    print("✅ Direct conv API with explicit output test passed!")


# ---------------------------------------------------------------------------
# Task 2: PyTorch FX translator tests
# ---------------------------------------------------------------------------


def test_conv_fx_basic(runtime, torch_rng_seed):
    """Verify FX translation of torch.nn.Conv2d."""

    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
            nn.init.ones_(self.conv.weight)

        def forward(self, x):
            return self.conv(x)

    model = ConvModel().eval()
    inp = [torch.ones(1, 1, 5, 5)]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inp)
    translator.run(inp)
    outputs = translator.get_outputs()

    assert len(outputs) == 1
    assert outputs[0].shape == (1, 1, 3, 3), f"Unexpected shape: {outputs[0].shape}"

    ref = model(inp[0]).detach().numpy()
    np.testing.assert_allclose(
        outputs[0].numpy(),
        ref,
        atol=1e-4,
        err_msg="Conv FX result differs from PyTorch",
    )
    print("✅ FX Conv2d test passed!")


def test_conv_fx_with_bias(runtime, torch_rng_seed):
    """Verify FX translation of torch.nn.Conv2d with bias."""

    class ConvBiasModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=True)

        def forward(self, x):
            return self.conv(x)

    model = ConvBiasModel().eval()
    inp = [torch.as_tensor(np.random.randn(1, 1, 4, 4).astype("float32"))]

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
        atol=1e-4,
        err_msg="Conv FX with bias result differs from PyTorch",
    )
    print("✅ FX Conv2d with bias test passed!")


if __name__ == "__main__":
    import sys
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(0 if exit_code == 0 else 1)
