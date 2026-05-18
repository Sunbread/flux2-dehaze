"""Tests for flow_matching_loss."""

import torch
import pytest
from dehaze_lora.loss import flow_matching_loss


def test_loss_zero_when_pred_equals_target():
    """Loss is zero when model_pred == noise - clean_latent."""
    clean = torch.randn(2, 128, 32, 32, dtype=torch.float32)
    noise = torch.randn(2, 128, 32, 32, dtype=torch.float32)
    target = noise - clean
    loss = flow_matching_loss(target, clean, noise)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_loss_positive_for_random_pred():
    """Loss > 0 when prediction is random (not equal to target)."""
    clean = torch.randn(2, 128, 32, 32, dtype=torch.float32)
    noise = torch.randn(2, 128, 32, 32, dtype=torch.float32)
    random_pred = torch.randn(2, 128, 32, 32, dtype=torch.float32)
    loss = flow_matching_loss(random_pred, clean, noise)
    assert loss.item() > 0.0


def test_loss_scalar_output():
    """Output is a scalar tensor."""
    clean = torch.randn(2, 16, 8, 8)
    noise = torch.randn(2, 16, 8, 8)
    pred = torch.randn(2, 16, 8, 8)
    loss = flow_matching_loss(pred, clean, noise)
    assert loss.ndim == 0


def test_loss_gradient_flow():
    """Loss backward propagates gradients to prediction tensor."""
    pred = torch.randn(2, 16, 8, 8, requires_grad=True)
    clean = torch.randn(2, 16, 8, 8)
    noise = torch.randn(2, 16, 8, 8)
    loss = flow_matching_loss(pred, clean, noise)
    loss.backward()
    assert pred.grad is not None
    assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))


def test_loss_bf16_computation():
    """Loss works with bf16 inputs, cast internally to float32."""
    clean = torch.randn(2, 16, 8, 8, dtype=torch.bfloat16)
    noise = torch.randn(2, 16, 8, 8, dtype=torch.bfloat16)
    pred = torch.randn(2, 16, 8, 8, dtype=torch.bfloat16)
    loss = flow_matching_loss(pred, clean, noise)
    assert loss.dtype == torch.float32


def test_loss_latent_shaped_inputs():
    """Loss handles typical latent shapes (B, 128, H, W)."""
    for shape in [(1, 128, 16, 16), (2, 128, 32, 32), (4, 128, 8, 8)]:
        clean = torch.randn(*shape)
        noise = torch.randn(*shape)
        pred = torch.randn(*shape)
        loss = flow_matching_loss(pred, clean, noise)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
