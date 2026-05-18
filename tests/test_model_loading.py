"""GPU tests for real model loading with LoRA injection.

Each test loads one large model at a time with strict VRAM limits.
Models are loaded via HF hub (cached or downloaded); skipped only on
load failure or insufficient VRAM.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from peft import PeftModel

from dehaze_lora.model import (
    load_models,
    _inject_lora,
    QWEN_LORA_MODULES,
    TRANSFORMER_LORA_MODULES,
)
from tests.conftest import _require_vram_gb, _require_cuda, cleanup_gpu, \
    load_flux2_transformer, MODEL_NAME


def _load_qwen():
    from transformers import Qwen3ForCausalLM
    return Qwen3ForCausalLM.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------------
# Transformer LoRA
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
class TestTransformerLoRA:

    def test_transformer_lora_injection(self):
        _require_cuda()
        _require_vram_gb(20)
        transformer = None
        try:
            transformer = load_flux2_transformer()
        except Exception as e:
            pytest.skip(f"Transformer not available: {e}")
        try:
            transformer.requires_grad_(False)
            peft_model = _inject_lora(
                transformer, rank=4, alpha=8,
                target_modules=TRANSFORMER_LORA_MODULES,
            )
            assert isinstance(peft_model, PeftModel)
            lora_param_count = sum(
                p.numel() for p in peft_model.parameters() if p.requires_grad
            )
            assert lora_param_count > 0
        finally:
            del transformer; cleanup_gpu()

    def test_transformer_forward_tiny_input(self):
        _require_cuda()
        _require_vram_gb(20)
        transformer = None
        try:
            transformer = load_flux2_transformer()
        except Exception as e:
            pytest.skip(f"Transformer not available: {e}")
        try:
            transformer.eval()
            transformer.to("cuda")
            config = transformer.config
            in_channels = getattr(config, "in_channels", 128)
            ctx_dim = 12288  # joint_attention_dim (3 layers × 4096)

            B, H, W = 1, 4, 4
            hidden_states = torch.randn(
                B, H * W, in_channels, device="cuda", dtype=torch.bfloat16,
            )
            encoder_hidden_states = torch.randn(
                B, 32, ctx_dim, device="cuda", dtype=torch.bfloat16,
            )
            img_ids = torch.zeros(B, H * W, 4, device="cuda", dtype=torch.float32)
            img_ids[:, :, 1] = torch.arange(H, device="cuda").unsqueeze(1).repeat(1, W).reshape(-1).float()
            img_ids[:, :, 2] = torch.arange(W, device="cuda").unsqueeze(0).repeat(H, 1).reshape(-1).float()
            txt_ids = torch.zeros(32, 4, device="cuda", dtype=torch.float32)
            txt_ids[:, 3] = torch.arange(32, device="cuda").float()
            timestep = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)

            with torch.no_grad():
                output = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
                    return_dict=False,
                )[0]
            assert output.shape == hidden_states.shape
        finally:
            del transformer; cleanup_gpu()

    def test_lora_save_load_roundtrip_real_model(self):
        _require_cuda()
        _require_vram_gb(20)
        transformer = None
        try:
            transformer = load_flux2_transformer()
        except Exception as e:
            pytest.skip(f"Transformer not available: {e}")
        save_tmp = tempfile.TemporaryDirectory()
        try:
            transformer.requires_grad_(False)
            peft_model = _inject_lora(
                transformer, rank=4, alpha=8,
                target_modules=TRANSFORMER_LORA_MODULES,
            )
            save_dir = Path(save_tmp.name) / "transformer_lora"
            peft_model.save_pretrained(str(save_dir))
            assert (save_dir / "adapter_config.json").exists()
            assert (save_dir / "adapter_model.safetensors").exists()

            transformer = None
            peft_model = None
            cleanup_gpu()

            transformer2 = load_flux2_transformer()
            loaded = PeftModel.from_pretrained(transformer2, str(save_dir))
            assert isinstance(loaded, PeftModel)
            lora_params = sum(1 for name, _ in loaded.named_parameters() if "lora" in name)
            assert lora_params > 0, "No LoRA params after reload"
            transformer2 = None
            loaded = None
            cleanup_gpu()
        finally:
            save_tmp.cleanup()
            transformer = None
            cleanup_gpu()


# ---------------------------------------------------------------------------
# LoRA target selection
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
class TestLoRATargetSelection:

    def test_lora_target_transformer_only(self):
        _require_cuda()
        _require_vram_gb(20)
        models = None
        try:
            models = load_models(
                model_name=MODEL_NAME, lora_target="transformer",
                lora_rank=4, lora_alpha=8, device="cuda",
            )
        except Exception as e:
            pytest.skip(f"Models not available: {e}")
        try:
            assert isinstance(models["transformer"], PeftModel)
            assert not isinstance(models["text_encoder"], PeftModel)
        finally:
            if models:
                del models["vae"], models["transformer"], models["text_encoder"]; cleanup_gpu()



# ---------------------------------------------------------------------------
# Text encoder LoRA
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
class TestQwenLoRA:

    def test_qwen_lora_injection(self):
        _require_cuda()
        _require_vram_gb(18)
        text_encoder = None
        try:
            text_encoder = _load_qwen()
        except Exception as e:
            pytest.skip(f"Qwen3 not available: {e}")
        try:
            text_encoder.requires_grad_(False)
            peft_model = _inject_lora(
                text_encoder, rank=4, alpha=8,
                target_modules=QWEN_LORA_MODULES,
            )
            assert isinstance(peft_model, PeftModel)
            lora_params = sum(
                p.numel() for p in peft_model.parameters() if p.requires_grad
            )
            assert lora_params > 0
        finally:
            del text_encoder; cleanup_gpu()
