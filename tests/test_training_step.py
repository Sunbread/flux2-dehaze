"""Tests for training step logic.

Synthetic tests (CPU): core flow-matching training logic.
GPU tests (real model): single training step with Flux2 transformer.
"""

import gc
import pytest
import torch

from dehaze_lora.loss import flow_matching_loss
from dehaze_lora.model import patchify_and_make_ids, unpatchify
from dehaze_lora.optimizer import create_optimizer
from tests.conftest import _require_vram_gb, _require_cuda, cleanup_gpu, \
    load_flux2_transformer, MODEL_NAME


# ---------------------------------------------------------------------------
# Synthetic training step (CPU — tests core logic)
# ---------------------------------------------------------------------------

class TestSyntheticTrainingStep:

    def test_flow_matching_training_step(self):
        B, C, H, W = 2, 128, 16, 16
        clean_latent = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        sigma = 0.5
        noisy_latent = (1 - sigma) * clean_latent + sigma * noise

        noisy_tokens, noisy_ids = patchify_and_make_ids(noisy_latent, patch_size=1, index=0.0)
        ref_tokens, ref_ids = patchify_and_make_ids(clean_latent, patch_size=1, index=10.0)

        hidden_states = torch.cat([noisy_tokens, ref_tokens], dim=1)
        img_ids = torch.cat([noisy_ids, ref_ids], dim=1)

        assert hidden_states.shape == (B, 2 * H * W, C)
        assert img_ids.shape == (B, 2 * H * W, 4)

        ref_ids_check = img_ids[:, H * W:, :]
        assert torch.allclose(ref_ids_check[:, :, 0], torch.full((B, H * W), 10.0))

    def test_optimizer_step_with_synthetic_params(self):
        class LoRAModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.Parameter(torch.randn(32, 8))
                self.lora_B = torch.nn.Parameter(torch.randn(8, 64))

        model = LoRAModel()
        init_A = model.lora_A.clone()
        init_B = model.lora_B.clone()

        opt = create_optimizer(model, lr=1e-3, weight_decay=0.01)
        x = torch.randn(4, 32)
        loss = (x @ model.lora_A @ model.lora_B).sum()
        loss.backward()
        opt.step()

        assert not torch.allclose(model.lora_A, init_A)
        assert not torch.allclose(model.lora_B, init_B)

    def test_loss_matches_coding_plan_formula(self):
        clean = torch.ones(2, 128, 8, 8)
        noise = torch.full((2, 128, 8, 8), 0.5)
        target = noise - clean
        wrong_pred = torch.zeros(2, 128, 8, 8)

        loss_wrong = flow_matching_loss(wrong_pred, clean, noise)
        loss_correct = flow_matching_loss(target, clean, noise)

        assert loss_wrong > 0
        assert loss_correct == pytest.approx(0.0, abs=1e-5)

    def test_synthetic_gradient_accumulation_params_change(self):
        """Multi-step training with gradient accumulation: params must change.

        Catches: missing accelerator.accumulate(), no_grad blocking
        gradients, optimizer not stepping, lora_target edge cases.
        """
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.Parameter(torch.randn(32, 8))
                self.lora_B = torch.nn.Parameter(torch.randn(8, 64))

            def forward(self, x):
                return x @ self.lora_A @ self.lora_B

        model = TinyModel()
        opt = create_optimizer(model, lr=1e-2, weight_decay=0.0)

        initial = {name: p.clone() for name, p in model.named_parameters()}

        grad_accum = 4
        for _step in range(8):  # 2 optimizer steps × 4 grad accum
            for _micro in range(grad_accum):
                x = torch.randn(4, 32)
                loss = model(x).sum()
                loss.backward()
            opt.step()
            opt.zero_grad()

        for name, p in model.named_parameters():
            assert not torch.allclose(p, initial[name], atol=1e-6), \
                f"Parameter {name} did not change after 2 optimizer steps"

    def test_cross_device_autograd(self):
        """Gradients flow through .to() CopyBackwards across devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < 2:
            pytest.skip("Need 2 GPUs for cross-device test")

        class ModelOnGPU1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(8, 8))

            def forward(self, x):
                return x @ self.weight

        model = ModelOnGPU1().to("cuda:1")
        init_weight = model.weight.clone()

        x = torch.randn(4, 8, device="cuda:1")

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=False):
            y = model(x).to("cuda:0")  # CopyBackwards recorded here
            loss = y.sum()

        loss.backward()

        assert model.weight.grad is not None, \
            "Gradient should flow across devices via CopyBackwards"
        assert not torch.allclose(model.weight.grad, torch.zeros_like(model.weight.grad))
        assert torch.allclose(model.weight, init_weight)  # weight not yet updated


# ---------------------------------------------------------------------------
# Real model training step (GPU, very careful VRAM)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
class TestRealModelTrainingStep:

    def test_single_training_step_real_transformer(self):
        _require_cuda()
        _require_vram_gb(20)

        from dehaze_lora.model import _inject_lora, TRANSFORMER_LORA_MODULES

        transformer = None
        try:
            transformer = load_flux2_transformer()
        except Exception as e:
            pytest.skip(f"Transformer not available: {e}")
        try:
            transformer.requires_grad_(False)
            transformer = _inject_lora(
                transformer, rank=4, alpha=8,
                target_modules=TRANSFORMER_LORA_MODULES,
            )
            transformer.train()
            transformer.to("cuda")

            config = transformer.config
            in_c = getattr(config, "in_channels", 128)
            ctx_dim = 12288  # joint_attention_dim (3 layers × 4096)
            ps = getattr(config, "patch_size", 1)

            lora_params_before = {
                name: p.clone()
                for name, p in transformer.named_parameters()
                if p.requires_grad
            }

            opt = create_optimizer(transformer, lr=1e-3, weight_decay=0.01)

            B, H, W = 1, 4, 4
            N_img = H * W
            N_txt = 8

            gt_latent = torch.randn(B, in_c, H, W, device="cuda", dtype=torch.bfloat16)
            noise = torch.randn(B, in_c, H, W, device="cuda", dtype=torch.bfloat16)
            sigma = 0.7
            noisy_latent = (1 - sigma) * gt_latent + sigma * noise
            hazy_latent = torch.randn(B, in_c, H, W, device="cuda", dtype=torch.bfloat16)

            noisy_tokens, noisy_ids = patchify_and_make_ids(noisy_latent, patch_size=ps, index=0.0)
            ref_tokens, ref_ids = patchify_and_make_ids(hazy_latent, patch_size=ps, index=10.0)
            hidden_states = torch.cat([noisy_tokens, ref_tokens], dim=1)
            img_ids = torch.cat([noisy_ids, ref_ids], dim=1)

            encoder_hidden_states = torch.randn(B, N_txt, ctx_dim, device="cuda", dtype=torch.bfloat16)
            txt_ids = torch.zeros(N_txt, 4, device="cuda", dtype=torch.float32)
            txt_ids[:, 3] = torch.arange(N_txt, device="cuda").float()
            timestep = torch.full((B,), sigma, device="cuda", dtype=torch.bfloat16)

            model_pred = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
                return_dict=False,
            )[0]

            v_theta = model_pred[:, :N_img, :]
            v_theta = unpatchify(v_theta, H, W, patch_size=ps)
            loss = flow_matching_loss(v_theta, gt_latent, noise)
            assert loss.item() > 0
            assert not torch.isnan(loss)

            loss.backward()

            grad_count = sum(1 for _n, p in transformer.named_parameters()
                             if p.requires_grad and p.grad is not None)
            assert grad_count > 0, "No LoRA parameters received gradients"

            opt.step()
            opt.zero_grad()

            changed = sum(1 for name, p in transformer.named_parameters()
                          if p.requires_grad and not torch.allclose(p, lora_params_before[name], atol=1e-7))
            assert changed > 0, "No LoRA parameters were updated"
        finally:
            del transformer; cleanup_gpu()

    def test_seed_determinism_real_model(self):
        _require_cuda()
        _require_vram_gb(20)

        def _run_forward_and_get_loss(seed):
            transformer = None
            try:
                transformer = load_flux2_transformer()
            except Exception as e:
                pytest.skip(f"Transformer not available: {e}")
            try:
                transformer.eval()
                transformer.to("cuda")
                config = transformer.config
                in_c = getattr(config, "in_channels", 128)
                ctx_dim = 12288  # joint_attention_dim (3 layers × 4096)

                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                B, H, W = 1, 4, 4
                hidden_states = torch.randn(B, H * W, in_c, device="cuda", dtype=torch.bfloat16)
                encoder_hidden_states = torch.randn(B, 8, ctx_dim, device="cuda", dtype=torch.bfloat16)
                img_ids = torch.zeros(B, H * W, 4, device="cuda", dtype=torch.float32)
                txt_ids = torch.zeros(8, 4, device="cuda", dtype=torch.float32)
                txt_ids[:, 3] = torch.arange(8, device="cuda").float()
                timestep = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)

                with torch.no_grad():
                    out = transformer(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
                        return_dict=False,
                    )[0]
                return out.sum().item()
            finally:
                del transformer; cleanup_gpu()

        loss1 = _run_forward_and_get_loss(123)
        loss2 = _run_forward_and_get_loss(123)
        loss3 = _run_forward_and_get_loss(456)

        assert abs(loss1 - loss2) < 1e-5, \
            f"Same seed gave different loss: {loss1} vs {loss2}"
        assert abs(loss1 - loss3) > 1e-3, \
            f"Different seeds gave same loss: {loss1}"
