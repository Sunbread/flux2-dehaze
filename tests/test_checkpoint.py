"""GPU tests for checkpoint save/load with real Flux2 transformer."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from peft import PeftModel

from dehaze_lora.model import _inject_lora, TRANSFORMER_LORA_MODULES
from tests.conftest import _require_vram_gb, _require_cuda, cleanup_gpu, load_flux2_transformer


@pytest.mark.gpu
@pytest.mark.slow
class TestCheckpointSaveLoad:

    def test_checkpoint_file_structure(self):
        _require_cuda()
        _require_vram_gb(20)
        transformer = None
        try:
            transformer = load_flux2_transformer()
        except Exception as e:
            pytest.skip(f"Transformer not available: {e}")
        try:
            torch.manual_seed(42)
            transformer.requires_grad_(False)
            peft_model = _inject_lora(
                transformer, rank=4, alpha=8,
                target_modules=TRANSFORMER_LORA_MODULES,
            )
            with tempfile.TemporaryDirectory() as tmp:
                ckpt_dir = Path(tmp) / "checkpoint-100" / "transformer_lora"
                ckpt_dir.mkdir(parents=True)
                peft_model.save_pretrained(str(ckpt_dir))
                assert (ckpt_dir / "adapter_config.json").exists()
                assert (ckpt_dir / "adapter_model.safetensors").exists()
                config = json.loads((ckpt_dir / "adapter_config.json").read_text())
                assert config["r"] == 4
                assert config["lora_alpha"] == 8
                assert "target_modules" in config
        finally:
            transformer = None
            cleanup_gpu()

    def test_checkpoint_roundtrip_weights(self):
        _require_cuda()
        _require_vram_gb(20)
        transformer = None
        try:
            transformer = load_flux2_transformer()
        except Exception as e:
            pytest.skip(f"Transformer not available: {e}")
        save_dir = None
        try:
            torch.manual_seed(42)
            transformer.requires_grad_(False)
            peft_model = _inject_lora(
                transformer, rank=4, alpha=8,
                target_modules=TRANSFORMER_LORA_MODULES,
            )
            save_tmp = tempfile.TemporaryDirectory()
            save_dir = Path(save_tmp.name) / "transformer_lora"
            peft_model.save_pretrained(str(save_dir))

            # Free first transformer before loading second
            transformer = None
            peft_model = None
            save_tmp2 = save_tmp  # keep alive
            cleanup_gpu()

            torch.manual_seed(42)
            transformer2 = load_flux2_transformer()
            loaded = PeftModel.from_pretrained(transformer2, str(save_dir))
            lora_params = sum(1 for name, _ in loaded.named_parameters() if "lora" in name)
            assert lora_params > 0, "No LoRA params after reload"
            transformer2 = None
            loaded = None
            save_tmp.cleanup()
            cleanup_gpu()
        finally:
            transformer = None
            cleanup_gpu()

    def test_checkpoint_forward_consistency(self):
        _require_cuda()
        _require_vram_gb(20)
        transformer = None
        try:
            transformer = load_flux2_transformer()
        except Exception as e:
            pytest.skip(f"Transformer not available: {e}")
        save_dir = None
        try:
            torch.manual_seed(42)
            transformer.requires_grad_(False)
            peft_model = _inject_lora(
                transformer, rank=4, alpha=8,
                target_modules=TRANSFORMER_LORA_MODULES,
            )
            peft_model.eval()
            peft_model.to("cuda")

            base_config = peft_model.get_base_model().config
            in_c = getattr(base_config, "in_channels", 128)
            ctx_dim = 12288

            torch.manual_seed(123)
            B, H, W = 1, 4, 4
            hidden_states = torch.randn(B, H * W, in_c, device="cuda", dtype=torch.bfloat16)
            encoder_hidden_states = torch.randn(B, 8, ctx_dim, device="cuda", dtype=torch.bfloat16)
            img_ids = torch.zeros(B, H * W, 4, device="cuda", dtype=torch.float32)
            img_ids[:, :, 1] = torch.arange(H, device="cuda").unsqueeze(1).expand(-1, W).reshape(-1).float()
            img_ids[:, :, 2] = torch.arange(W, device="cuda").unsqueeze(0).expand(H, -1).reshape(-1).float()
            txt_ids = torch.zeros(8, 4, device="cuda", dtype=torch.float32)
            txt_ids[:, 3] = torch.arange(8, device="cuda").float()
            timestep = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)

            with torch.no_grad():
                out_before = peft_model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
                    return_dict=False,
                )[0]

            save_tmp = tempfile.TemporaryDirectory()
            save_dir = Path(save_tmp.name) / "transformer_lora"
            peft_model.save_pretrained(str(save_dir))

            # Free first transformer
            transformer = None
            peft_model = None
            cleanup_gpu()

            torch.manual_seed(42)
            transformer2 = load_flux2_transformer()
            loaded = PeftModel.from_pretrained(transformer2, str(save_dir))
            loaded.eval()
            loaded.to("cuda")

            with torch.no_grad():
                out_after = loaded(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
                    return_dict=False,
                )[0]

            assert torch.allclose(out_before, out_after, atol=1e-3), \
                "Forward output differs after checkpoint reload"
            transformer2 = None
            loaded = None
            save_tmp.cleanup()
            cleanup_gpu()
        finally:
            transformer = None
            cleanup_gpu()
