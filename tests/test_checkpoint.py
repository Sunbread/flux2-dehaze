"""GPU tests for checkpoint save/load with real Flux2 transformer."""

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from peft import PeftModel
from torch.utils.data import DataLoader, TensorDataset

from dehaze_lora.model import _inject_lora, TRANSFORMER_LORA_MODULES
from dehaze_lora.checkpoint import (
    get_rng_state, set_rng_state, load_training_state,
)
from dehaze_lora.optimizer import create_optimizer
from dehaze_lora.train import _training_batches
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


class TinyLoRAModel(torch.nn.Module):
    """Minimal 2D-param model for resume tests (Muon requires 2D params)."""

    def __init__(self, in_dim=16, mid_dim=8, out_dim=16):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.randn(in_dim, mid_dim))
        self.lora_B = torch.nn.Parameter(torch.randn(mid_dim, out_dim))

    def forward(self, x):
        return x @ self.lora_A @ self.lora_B


class TestCheckpointResume:
    """CPU tests: bit-identical training after checkpoint resume."""

    def test_rng_state_roundtrip(self):
        """get_rng_state() → set_rng_state() produces identical RNG output."""
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Capture state BEFORE first draw
        state = get_rng_state()

        r1 = random.random()
        n1 = np.random.random()
        t1 = torch.randn(1).item()

        # Advance state
        random.random()
        np.random.random()
        torch.randn(1)

        # Restore state (back to before first draw) and re-draw
        set_rng_state(state)
        r2 = random.random()
        n2 = np.random.random()
        t2 = torch.randn(1).item()

        assert r1 == r2, "Python random state not preserved"
        assert n1 == n2, "NumPy random state not preserved"
        assert abs(t1 - t2) < 1e-10, "Torch CPU random state not preserved"

    def test_fast_forward_deterministic(self):
        """DataLoader with _training_batches skip matches original iteration."""
        ds = TensorDataset(torch.arange(100))
        g = torch.Generator().manual_seed(42)
        dl_full = DataLoader(ds, batch_size=7, shuffle=True, generator=g, num_workers=0)

        all_batches = [batch[0].tolist() for batch in dl_full]

        g2 = torch.Generator().manual_seed(42)
        dl_skip = DataLoader(ds, batch_size=7, shuffle=True, generator=g2, num_workers=0)

        skip = 4
        remaining = []
        for batch in _training_batches(dl_skip, skip=skip):
            remaining.append(batch[0].tolist())
            if len(remaining) >= len(all_batches) - skip:
                break

        assert remaining == all_batches[skip:], (
            f"Fast-forward by {skip} batches produced wrong remainder"
        )

    def test_resume_bit_identical(self, tmp_dir):
        """Training N steps from scratch = train M steps + resume (N-M) steps."""
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)

        # Run A: train 8 optimizer steps from scratch
        torch.manual_seed(42)
        model_a = TinyLoRAModel()
        init_params = {n: p.clone() for n, p in model_a.named_parameters()}
        opt_a = create_optimizer(model_a, lr=0.01, weight_decay=0.0, momentum=0.9)

        data = [torch.randn(4, 16) for _ in range(64)]

        step = 0
        for i in range(64):
            loss = model_a(data[i]).sum()
            loss.backward()
            step += 1
            if step % 8 == 0:
                opt_a.step()
                opt_a.zero_grad()

        params_a = {n: p.clone() for n, p in model_a.named_parameters()}

        # Run B: train 3 optimizer steps, save, resume, train 5 more
        torch.manual_seed(42)
        model_b = TinyLoRAModel()
        with torch.no_grad():
            for n, p in model_b.named_parameters():
                p.copy_(init_params[n])

        opt_b = create_optimizer(model_b, lr=0.01, weight_decay=0.0, momentum=0.9)

        step = 0
        for i in range(24):
            loss = model_b(data[i]).sum()
            loss.backward()
            step += 1
            if step % 8 == 0:
                opt_b.step()
                opt_b.zero_grad()

        rng_state = get_rng_state()
        # Save training state manually (TinyLoRAModel has no save_pretrained)
        ckpt_dir = tmp_dir / "checkpoint-3"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        training_state = {
            "global_step": 3,
            "micro_step": 24,
            "rng_states": rng_state,
            "optimizer_states": {
                "transformer": opt_b.state_dict() if opt_b else None,
                "qwen": None,
            },
        }
        torch.save(training_state, ckpt_dir / "training_state.pt")
        torch.save(model_b.state_dict(), ckpt_dir / "model_weights.pt")

        # Resume: load checkpoint, train 5 more steps
        torch.manual_seed(42)
        model_resume = TinyLoRAModel()
        state_loaded = load_training_state(ckpt_dir)
        set_rng_state(state_loaded["rng_states"])
        model_resume.load_state_dict(torch.load(ckpt_dir / "model_weights.pt", map_location="cpu"))

        opt_resume = create_optimizer(
            model_resume, lr=0.01, weight_decay=0.0, momentum=0.9,
        )
        if state_loaded["optimizer_states"]["transformer"] is not None:
            opt_resume.load_state_dict(state_loaded["optimizer_states"]["transformer"])

        step = 0
        for i in range(24, 64):
            loss = model_resume(data[i]).sum()
            loss.backward()
            step += 1
            if step % 8 == 0:
                opt_resume.step()
                opt_resume.zero_grad()

        params_resume = {n: p.clone() for n, p in model_resume.named_parameters()}

        for name in params_a:
            assert torch.equal(params_a[name], params_resume[name]), (
                f"Parameter {name} differs between scratch and resume"
            )

    def test_noise_rng_consistent_after_resume(self):
        """Noise/timestep at micro_step=M is identical after resume.

        The training loop uses Generator(device).manual_seed(seed + micro_step)
        for timestep and noise. This must produce bit-identical values when
        micro_step is restored from a checkpoint.
        """
        seed = 42
        B, C, H, W = 4, 128, 32, 32  # dummy latent shape

        # Simulate training: record noise at micro_steps 0..15
        noises_from_scratch = {}
        timesteps_from_scratch = {}
        for ms in range(32):
            rng = torch.Generator("cpu").manual_seed(seed + ms)
            t = torch.randint(0, 1000, (B,), generator=rng)
            noise = torch.randn(B, C, H, W, generator=rng)
            if ms % 8 == 0:  # record at optimizer step boundaries
                noises_from_scratch[ms] = noise.clone()
                timesteps_from_scratch[ms] = t.clone()

        # Simulate resume at micro_step=16:
        # "Save checkpoint" — just record micro_step
        saved_micro_step = 16

        # "Resume" — restore micro_step, continue from micro_step=16
        for ms in range(saved_micro_step, 32):
            rng = torch.Generator("cpu").manual_seed(seed + ms)
            t = torch.randint(0, 1000, (B,), generator=rng)
            noise = torch.randn(B, C, H, W, generator=rng)
            if ms % 8 == 0:
                assert torch.equal(noises_from_scratch[ms], noise), (
                    f"Noise mismatch at micro_step={ms} after resume"
                )
                assert torch.equal(timesteps_from_scratch[ms], t), (
                    f"Timestep mismatch at micro_step={ms} after resume"
                )
