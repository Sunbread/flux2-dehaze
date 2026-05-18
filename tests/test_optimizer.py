"""Tests for Muon optimizer creation and validation."""

import torch
import torch.nn as nn
import pytest
from dehaze_lora.optimizer import create_optimizer


class Simple2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(16, 8))
        self.lora_B = nn.Parameter(torch.randn(8, 16))
        self.frozen = nn.Parameter(torch.randn(16, 8), requires_grad=False)

    def forward(self, x):
        return x @ self.lora_A @ self.lora_B


class Non2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_1d = nn.Parameter(torch.randn(16))
        self.param_2d = nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        return x


def test_creates_muon_optimizer():
    model = Simple2DModel()
    opt = create_optimizer(model, lr=1e-3, weight_decay=0.01)
    assert isinstance(opt, torch.optim.Muon)


def test_only_trainable_params_included():
    model = Simple2DModel()
    opt = create_optimizer(model)
    param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert id(model.lora_A) in param_ids
    assert id(model.lora_B) in param_ids
    assert id(model.frozen) not in param_ids


def test_muon_rejects_non_2d():
    model = Non2DModel()
    with pytest.raises(ValueError, match="only supports 2D"):
        create_optimizer(model)


def test_default_hyperparameters():
    model = Simple2DModel()
    opt = create_optimizer(model)
    group = opt.param_groups[0]
    assert group["lr"] == 1e-3
    assert group["weight_decay"] == 0.01
    assert group["momentum"] == 0.95


def test_muon_step_changes_params():
    model = Simple2DModel()
    opt = create_optimizer(model)
    initial_A = model.lora_A.clone()
    initial_B = model.lora_B.clone()

    x = torch.randn(4, 16)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    assert not torch.allclose(model.lora_A, initial_A)
    assert not torch.allclose(model.lora_B, initial_B)


class AllFrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16, 16),
                                         requires_grad=False)

    def forward(self, x):
        return x @ self.weight


def test_no_trainable_params_raises():
    """create_optimizer with all-frozen model raises clear error."""
    model = AllFrozenModel()
    with pytest.raises(ValueError, match="No trainable parameters"):
        create_optimizer(model)


def test_custom_lr_and_wd():
    model = Simple2DModel()
    opt = create_optimizer(model, lr=5e-4, weight_decay=0.05)
    group = opt.param_groups[0]
    assert group["lr"] == 5e-4
    assert group["weight_decay"] == 0.05
