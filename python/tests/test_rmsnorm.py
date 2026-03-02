"""
Tests for the RmsNorm operator integration in InfiniTensor.

Covers three levels:
1. Direct GraphBuilder Python API (gb.rmsnorm())
2. PyTorch FX translator (torch.nn.RMSNorm → RmsNormObj)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from infinitensor import TorchFXTranslator


# ---------------------------------------------------------------------------
# Task 1: Direct Python API tests
# ---------------------------------------------------------------------------


def test_rmsnorm_direct_api(runtime, torch_rng_seed):
    """Verify that GraphBuilder.rmsnorm() binding is usable and returns a tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    w = gb.tensor(pit.ShapeExpr([4]), pit.dtype_from_string("float32"))
    y = gb.rmsnorm(x, w)
    assert y is not None, "gb.rmsnorm() should return an output tensor"
    print("✅ Direct rmsnorm API test passed!")


def test_rmsnorm_direct_api_with_output(runtime, torch_rng_seed):
    """Verify that gb.rmsnorm() accepts a pre-allocated output tensor."""
    import pyinfinitensor as pit

    gb = pit.GraphBuilder(runtime)
    x = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    w = gb.tensor(pit.ShapeExpr([4]), pit.dtype_from_string("float32"))
    y_pre = gb.tensor(pit.ShapeExpr([2, 4]), pit.dtype_from_string("float32"))
    y = gb.rmsnorm(x, w, Y=y_pre)
    assert y is not None, "gb.rmsnorm() with explicit Y should return that tensor"
    print("✅ Direct rmsnorm API with explicit output test passed!")


# ---------------------------------------------------------------------------
# Task 2: PyTorch FX translator tests
# ---------------------------------------------------------------------------


def _rms_norm_ref(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Reference RMS normalization (applied over the last dimension)."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def test_rmsnorm_fx_basic(runtime, torch_rng_seed):
    """Verify FX translation of torch.nn.RMSNorm (PyTorch >= 2.4)."""
    if not hasattr(nn, "RMSNorm"):
        pytest.skip("torch.nn.RMSNorm requires PyTorch >= 2.4")

    class RMSNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rms = nn.RMSNorm(4, eps=1e-6)

        def forward(self, x):
            return self.rms(x)

    model = RMSNormModel().eval()
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
        err_msg="RmsNorm FX result differs from PyTorch",
    )
    print("✅ FX RMSNorm test passed!")


def test_rmsnorm_direct_correctness(runtime, torch_rng_seed):
    """Verify numerical correctness of RmsNorm via FX translator (end-to-end)."""
    if not hasattr(torch.nn, "RMSNorm"):
        pytest.skip("torch.nn.RMSNorm requires PyTorch >= 2.4")

    class RMSNormModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rms = torch.nn.RMSNorm(4, eps=1e-6)

        def forward(self, x):
            return self.rms(x)

    x_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    w_np = np.ones(4, dtype=np.float32)
    eps = 1e-6
    rms = np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + eps)
    expected = (x_np / rms * w_np).flatten()

    model = RMSNormModel().eval()
    # Override the weight to ones for deterministic reference
    with torch.no_grad():
        model.rms.weight.fill_(1.0)
    inp = [torch.as_tensor(x_np)]

    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, inp)
    translator.run(inp)
    outputs = translator.get_outputs()

    assert len(outputs) == 1
    np.testing.assert_allclose(outputs[0].numpy().flatten(), expected, atol=1e-4)
    print("✅ RmsNorm numerical correctness test passed!")


if __name__ == "__main__":
    import sys
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])
    sys.exit(0 if exit_code == 0 else 1)
