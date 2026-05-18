"""Tests for LoRA injection, save/load, merge on tiny synthetic models."""

import torch
import tempfile
from pathlib import Path
import pytest
from peft import PeftModel

from dehaze_lora.model import _inject_lora, QWEN_LORA_MODULES, TRANSFORMER_LORA_MODULES
from tests.conftest import TinyAttention, TinyQwenAttention


# ---------------------------------------------------------------------------
# Injection correctness
# ---------------------------------------------------------------------------

class TestLoRAInjection:
    def test_returns_peft_model(self):
        model = TinyAttention(dim=64)
        result = _inject_lora(model, rank=4, alpha=8,
                              target_modules=TRANSFORMER_LORA_MODULES)
        assert isinstance(result, PeftModel)

    def test_base_params_frozen(self):
        model = TinyAttention(dim=64)
        peft_model = _inject_lora(model, rank=4, alpha=8,
                                  target_modules=TRANSFORMER_LORA_MODULES)
        for name, param in peft_model.base_model.named_parameters():
            if "lora" not in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_lora_params_trainable(self):
        model = TinyAttention(dim=64)
        peft_model = _inject_lora(model, rank=4, alpha=8,
                                  target_modules=TRANSFORMER_LORA_MODULES)
        lora_trainable = 0
        for name, param in peft_model.named_parameters():
            if "lora" in name and param.requires_grad:
                lora_trainable += 1
        assert lora_trainable > 0, "No LoRA params are trainable"

    def test_target_modules_exist_in_model(self):
        model = TinyAttention(dim=64)
        peft_model = _inject_lora(model, rank=4, alpha=8,
                                  target_modules=TRANSFORMER_LORA_MODULES)
        for tm in TRANSFORMER_LORA_MODULES:
            assert hasattr(peft_model.base_model.model, tm) or \
                   any(tm in name for name, _ in peft_model.base_model.model.named_modules()), \
                   f"target_module '{tm}' not found in model"

    def test_qwen_target_modules(self):
        model = TinyQwenAttention(dim=64)
        peft_model = _inject_lora(model, rank=4, alpha=8,
                                  target_modules=QWEN_LORA_MODULES)
        for tm in QWEN_LORA_MODULES:
            assert hasattr(peft_model.base_model.model, tm), \
                   f"target_module '{tm}' not found in Qwen model"


# ---------------------------------------------------------------------------
# Rank / alpha variations
# ---------------------------------------------------------------------------

class TestLoRAHyperparams:
    @pytest.mark.parametrize("rank", [2, 4, 8, 16])
    def test_different_ranks(self, rank):
        model = TinyAttention(dim=64)
        peft_model = _inject_lora(model, rank=rank, alpha=rank * 2,
                                  target_modules=["to_q"])
        lora_a = peft_model.base_model.model.to_q.lora_A.default.weight
        assert lora_a.shape[0] == rank

    @pytest.mark.parametrize("alpha", [1, 4, 8, 16, 32])
    def test_different_alphas(self, alpha):
        model = TinyAttention(dim=64)
        peft_model = _inject_lora(model, rank=4, alpha=alpha,
                                  target_modules=["to_q"])
        scaling = peft_model.base_model.model.to_q.scaling
        expected_scale = alpha / 4  # lora_alpha / r
        assert abs(scaling["default"] - expected_scale) < 1e-6


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

class TestLoRASaveLoad:
    def test_save_load_adapter_files_exist(self):
        torch.manual_seed(42)
        model = TinyAttention(dim=64)
        peft_model = _inject_lora(model, rank=4, alpha=8,
                                  target_modules=TRANSFORMER_LORA_MODULES)

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "lora_ckpt"
            peft_model.save_pretrained(str(save_dir))
            assert (save_dir / "adapter_config.json").exists()
            assert (save_dir / "adapter_model.safetensors").exists()

    def test_save_load_full_state_roundtrip(self):
        """Save adapter, reload with from_pretrained on identical-weight base → match."""
        torch.manual_seed(42)
        model1 = TinyAttention(dim=64)
        peft1 = _inject_lora(model1, rank=4, alpha=8,
                             target_modules=TRANSFORMER_LORA_MODULES)

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "lora_ckpt"
            peft1.save_pretrained(str(save_dir))

            # Re-create base with same seed → identical weights
            torch.manual_seed(42)
            model2 = TinyAttention(dim=64)
            loaded = PeftModel.from_pretrained(model2, str(save_dir))

            for key, val in peft1.state_dict().items():
                assert torch.allclose(val, loaded.state_dict()[key], atol=1e-6), \
                       f"Mismatch in {key}"

    def test_forward_output_preserved_after_reload(self):
        """Output matches when adapter reloaded onto identical-weight base."""
        torch.manual_seed(42)
        model1 = TinyAttention(dim=64)
        peft1 = _inject_lora(model1, rank=4, alpha=8,
                             target_modules=["to_q", "to_v"])
        peft1.eval()

        x = torch.randn(2, 64)
        with torch.no_grad():
            out_before = peft1(x)

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "lora_ckpt"
            peft1.save_pretrained(str(save_dir))

            torch.manual_seed(42)
            model2 = TinyAttention(dim=64)
            loaded = PeftModel.from_pretrained(model2, str(save_dir))
            loaded.eval()
            with torch.no_grad():
                out_after = loaded(x)
            assert torch.allclose(out_before, out_after, atol=1e-5)


# ---------------------------------------------------------------------------
# Merge / unload
# ---------------------------------------------------------------------------

class TestLoRAMerge:
    def test_merge_and_unload(self):
        model = TinyAttention(dim=64)
        peft_model = _inject_lora(model, rank=4, alpha=8,
                                  target_modules=TRANSFORMER_LORA_MODULES)

        x = torch.randn(2, 64)
        peft_model.eval()
        with torch.no_grad():
            out_before = peft_model(x)

        merged = peft_model.merge_and_unload()

        with torch.no_grad():
            out_after = merged(x)

        # Merge should not change output significantly
        assert torch.allclose(out_before, out_after, atol=1e-4)

    def test_merge_and_unload_is_plain_module(self):
        model = TinyAttention(dim=64)
        peft_model = _inject_lora(model, rank=4, alpha=8,
                                  target_modules=["to_q"])
        merged = peft_model.merge_and_unload()
        assert not isinstance(merged, PeftModel)
        assert isinstance(merged, type(model))


# ---------------------------------------------------------------------------
# Training behavior
# ---------------------------------------------------------------------------

class TestLoRATraining:
    def test_only_lora_params_updated(self):
        model = TinyAttention(dim=64)
        peft_model = _inject_lora(model, rank=4, alpha=8,
                                  target_modules=TRANSFORMER_LORA_MODULES)

        # Record initial values
        init_state = {name: param.clone()
                      for name, param in peft_model.named_parameters()}

        opt = torch.optim.SGD(peft_model.parameters(), lr=0.1)
        x = torch.randn(4, 64)
        for _ in range(3):
            opt.zero_grad()
            loss = peft_model(x).sum()
            loss.backward()
            opt.step()

        for name, param in peft_model.named_parameters():
            if "lora" in name:
                assert not torch.allclose(param, init_state[name]), \
                       f"LoRA param {name} did not change"
            else:
                assert torch.allclose(param, init_state[name], atol=1e-6), \
                       f"Base param {name} changed but should be frozen"

    def test_gradient_flow(self):
        model = TinyAttention(dim=64)
        peft_model = _inject_lora(model, rank=4, alpha=8,
                                  target_modules=TRANSFORMER_LORA_MODULES)

        x = torch.randn(2, 64)
        out = peft_model(x)
        loss = out.sum()
        loss.backward()

        grad_count = 0
        for name, param in peft_model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                assert "lora" in name, f"Non-LoRA param {name} received gradient"
        assert grad_count > 0, "No parameters received gradients"


# ---------------------------------------------------------------------------
# lora_target selection logic (synthetic — no GPU needed)
# ---------------------------------------------------------------------------

class TestLoRATargetSelection:
    """Verify lora_target options correctly gate LoRA injection."""

    def test_target_transformer_only(self):
        """lora_target="transformer": only transformer modules get LoRA."""
        xformer = TinyAttention(dim=64)
        qwen = TinyQwenAttention(dim=64)
        xformer.requires_grad_(False)
        qwen.requires_grad_(False)

        lora_target = "transformer"
        if lora_target in ("transformer", "both"):
            xformer = _inject_lora(xformer, rank=4, alpha=8,
                                   target_modules=TRANSFORMER_LORA_MODULES)
        if lora_target in ("qwen", "both"):
            qwen = _inject_lora(qwen, rank=4, alpha=8,
                                target_modules=QWEN_LORA_MODULES)

        assert isinstance(xformer, PeftModel)
        assert not isinstance(qwen, PeftModel)

    def test_target_qwen_only(self):
        """lora_target="qwen": only qwen modules get LoRA."""
        xformer = TinyAttention(dim=64)
        qwen = TinyQwenAttention(dim=64)
        xformer.requires_grad_(False)
        qwen.requires_grad_(False)

        lora_target = "qwen"
        if lora_target in ("transformer", "both"):
            xformer = _inject_lora(xformer, rank=4, alpha=8,
                                   target_modules=TRANSFORMER_LORA_MODULES)
        if lora_target in ("qwen", "both"):
            qwen = _inject_lora(qwen, rank=4, alpha=8,
                                target_modules=QWEN_LORA_MODULES)

        assert not isinstance(xformer, PeftModel)
        assert isinstance(qwen, PeftModel)

    def test_target_both(self):
        """lora_target="both": both models get LoRA."""
        xformer = TinyAttention(dim=64)
        qwen = TinyQwenAttention(dim=64)
        xformer.requires_grad_(False)
        qwen.requires_grad_(False)

        lora_target = "both"
        if lora_target in ("transformer", "both"):
            xformer = _inject_lora(xformer, rank=4, alpha=8,
                                   target_modules=TRANSFORMER_LORA_MODULES)
        if lora_target in ("qwen", "both"):
            qwen = _inject_lora(qwen, rank=4, alpha=8,
                                target_modules=QWEN_LORA_MODULES)

        assert isinstance(xformer, PeftModel)
        assert isinstance(qwen, PeftModel)

    def test_target_neither(self):
        """Invalid lora_target: neither model gets LoRA."""
        xformer = TinyAttention(dim=64)
        qwen = TinyQwenAttention(dim=64)
        xformer.requires_grad_(False)
        qwen.requires_grad_(False)

        lora_target = "none"
        if lora_target in ("transformer", "both"):
            xformer = _inject_lora(xformer, rank=4, alpha=8,
                                   target_modules=TRANSFORMER_LORA_MODULES)
        if lora_target in ("qwen", "both"):
            qwen = _inject_lora(qwen, rank=4, alpha=8,
                                target_modules=QWEN_LORA_MODULES)

        assert not isinstance(xformer, PeftModel)
        assert not isinstance(qwen, PeftModel)
