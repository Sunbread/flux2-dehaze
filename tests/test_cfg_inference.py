"""Tests for CFG two-pass inference orchestration logic.

Tests production helpers (_apply_cfg, _concat_image_and_ref_tokens) and
verifies _denoise_all_modes orchestration with fake models.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from dehaze_lora.model import patchify_and_make_ids, unpatchify
from dehaze_lora.validate import _apply_cfg, _concat_image_and_ref_tokens


# ---------------------------------------------------------------------------
# _apply_cfg production helper tests
# ---------------------------------------------------------------------------

class TestApplyCFG:
    """v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)."""

    def test_no_guidance_equals_uncond(self):
        v_cond = torch.tensor([3.0, 5.0])
        v_uncond = torch.tensor([1.0, 2.0])
        v_cfg = _apply_cfg(v_cond, v_uncond, 0.0)
        assert torch.allclose(v_cfg, v_uncond)

    def test_guidance_1_equals_cond(self):
        v_cond = torch.tensor([3.0, 5.0])
        v_uncond = torch.tensor([1.0, 2.0])
        v_cfg = _apply_cfg(v_cond, v_uncond, 1.0)
        assert torch.allclose(v_cfg, v_cond)

    def test_guidance_3_5_extrapolates(self):
        v_cond = torch.tensor([2.0, 4.0])
        v_uncond = torch.tensor([0.0, 2.0])
        v_cfg = _apply_cfg(v_cond, v_uncond, 3.5)
        expected = torch.tensor([7.0, 9.0])
        assert torch.allclose(v_cfg, expected)

    def test_cfg_shape_preserved(self):
        B, C, H, W = 2, 128, 16, 16
        v_cond = torch.randn(B, C, H, W)
        v_uncond = torch.randn(B, C, H, W)
        v_cfg = _apply_cfg(v_cond, v_uncond, 3.5)
        assert v_cfg.shape == (B, C, H, W)

    def test_int_guidance_scale(self):
        """guidance_scale as int (3, not 3.0) should still work."""
        v_cond = torch.ones(2, 4)
        v_uncond = torch.zeros(2, 4)
        v_cfg = _apply_cfg(v_cond, v_uncond, 3)
        assert torch.allclose(v_cfg, torch.full((2, 4), 3.0))

    def test_float32_guidance_scale(self):
        """guidance_scale=np.float32(3.5) should still work."""
        import numpy as np
        v_cond = torch.ones(2, 4) * 2.0
        v_uncond = torch.ones(2, 4)
        v_cfg = _apply_cfg(v_cond, v_uncond, np.float32(3.5))
        assert torch.allclose(v_cfg, torch.full((2, 4), 4.5))


# ---------------------------------------------------------------------------
# _concat_image_and_ref_tokens production helper tests
# ---------------------------------------------------------------------------

class TestConcatImageAndRefTokens:
    """[_concat_image_and_ref_tokens] verifies [noisy | ref] order + IDs."""

    def test_noisy_before_ref(self):
        B, C, H, W = 1, 16, 4, 4
        noisy = torch.randn(B, C, H, W)
        ref = torch.randn(B, C, H, W) + 100  # easily distinguishable
        ref_tokens, ref_ids = patchify_and_make_ids(ref, patch_size=1, index=10.0)

        hidden, ids, N_noisy = _concat_image_and_ref_tokens(noisy, ref_tokens, ref_ids, 1)
        N_ref = ref_tokens.shape[1]

        assert hidden.shape == (B, N_noisy + N_ref, C)
        # First N_noisy tokens should be the noisy ones
        noisy_tokens, _ = patchify_and_make_ids(noisy, patch_size=1, index=0.0)
        assert torch.allclose(hidden[:, :N_noisy, :], noisy_tokens, atol=1e-5)
        # Last N_ref tokens should be the ref ones
        assert torch.allclose(hidden[:, N_noisy:, :], ref_tokens, atol=1e-5)

    def test_output_slicing_extracts_only_noisy(self):
        """After transformer forward, output[:, :N_noisy, :] is kept."""
        B, C, H, W = 1, 16, 4, 4
        noisy = torch.randn(B, C, H, W)
        ref = torch.randn(B, C, H, W)
        ref_tokens, ref_ids = patchify_and_make_ids(ref, patch_size=1, index=10.0)

        _hidden, _ids, N_noisy = _concat_image_and_ref_tokens(noisy, ref_tokens, ref_ids, 1)
        N_total = _hidden.shape[1]
        model_output = torch.randn(B, N_total, C)
        extracted = model_output[:, :N_noisy, :]
        assert extracted.shape == (B, N_noisy, C)

    def test_ref_ids_have_index_10(self):
        B, C, H, W = 1, 16, 4, 4
        noisy = torch.randn(B, C, H, W)
        ref = torch.randn(B, C, H, W)
        ref_tokens, ref_ids = patchify_and_make_ids(ref, patch_size=1, index=10.0)

        _hidden, combined_ids, N_noisy = _concat_image_and_ref_tokens(noisy, ref_tokens, ref_ids, 1)

        assert torch.all(combined_ids[:, :N_noisy, 0] == 0.0)
        assert torch.all(combined_ids[:, N_noisy:, 0] == 10.0)

    def test_different_patch_size(self):
        """patch_size=2 produces fewer tokens."""
        B, C, H, W = 1, 32, 8, 8
        noisy = torch.randn(B, C, H, W)
        ref = torch.randn(B, C, H, W)
        ref_tokens, ref_ids = patchify_and_make_ids(ref, patch_size=2, index=10.0)

        hidden, _ids, N_noisy = _concat_image_and_ref_tokens(noisy, ref_tokens, ref_ids, 2)
        assert N_noisy == (8 // 2) * (8 // 2)  # 16 tokens for 8x8 with patch_size=2


# ---------------------------------------------------------------------------
# Scheduler integration (exercises real FlowMatchEulerDiscreteScheduler)
# ---------------------------------------------------------------------------

class TestSchedulerIntegration:
    """Test scheduler.step integration with synthetic data."""

    def test_scheduler_step_changes_latent(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(28)

        z = torch.randn(1, 128, 8, 8)
        v_pred = torch.randn(1, 128, 8, 8)
        t = scheduler.timesteps[0]

        out = scheduler.step(v_pred, t, z)
        assert hasattr(out, "prev_sample")
        assert out.prev_sample.shape == z.shape
        assert not torch.allclose(out.prev_sample, z)

    def test_full_denoising_loop_shape(self):
        """Run a full synthetic denoising loop with zero-velocity predictions."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(28)

        z = torch.randn(1, 128, 8, 8)
        initial = z.clone()

        for t in scheduler.timesteps:
            v_pred = torch.zeros_like(z)
            out = scheduler.step(v_pred, t, z)
            z = out.prev_sample

        assert z.shape == initial.shape


# ---------------------------------------------------------------------------
# Mocked end-to-end _denoise_all_modes orchestration
# ---------------------------------------------------------------------------

class MockTransformerConfig:
    patch_size = 1


class TestDenoiseAllModesOrchestration:
    """Test _denoise_all_modes orchestration with fake models."""

    @staticmethod
    def _make_fake_scheduler(num_steps=1):
        """Create a scheduler mock with real config from FlowMatchEulerDiscreteScheduler."""
        from diffusers import FlowMatchEulerDiscreteScheduler
        real = FlowMatchEulerDiscreteScheduler()
        real.set_timesteps(num_steps)
        return real

    def test_denoise_all_modes_calls_transformer_correctly(self):
        """For 1-step denoising, transformer is called 4 times: cond, uncond, CFG-cond, CFG-uncond."""
        import numpy as np
        from dehaze_lora.validate import _denoise_all_modes

        B, C, H, W = 1, 128, 4, 4
        N_txt = 8

        hazy_latent = torch.randn(B, C, H, W)
        cond_embeds = torch.randn(B, N_txt, 4096)
        uncond_embeds = torch.randn(B, N_txt, 4096)
        cond_text_ids = torch.randint(0, 10, (B, N_txt, 4), dtype=torch.int64)
        uncond_text_ids = cond_text_ids.clone()

        # Fake transformer: records calls, returns distinguishable values
        call_records = []

        class FakeTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockTransformerConfig()

            def forward(self, hidden_states, encoder_hidden_states, timestep,
                        img_ids, txt_ids, return_dict=False):
                call_records.append({
                    "hidden_states_shape": hidden_states.shape,
                    "img_ids": img_ids.clone(),
                    "enc_shape": encoder_hidden_states.shape,
                })
                B_in, total, C_in = hidden_states.shape
                # Return distinguishable velocities: cond=2.0, uncond=1.0
                if encoder_hidden_states[:, 0, 0].mean() == cond_embeds[:, 0, 0].mean():
                    val = torch.ones(B_in, total, C_in) * 2.0
                else:
                    val = torch.ones(B_in, total, C_in) * 1.0
                return (val,)

        fake_transformer = FakeTransformer()

        # Fake VAE decode: return identity
        def fake_decode(vae, latents):
            return latents[:, :3, :, :]  # (B, 3, H, W) — dummy image

        # Fake scheduler
        fake_scheduler = self._make_fake_scheduler(num_steps=1)

        with patch("dehaze_lora.validate.decode_vae_image", fake_decode), \
             patch("dehaze_lora.validate.compute_empirical_mu", return_value=1.0):
            result = _denoise_all_modes(
                transformer=fake_transformer,
                scheduler=fake_scheduler,
                vae=None,
                hazy_latent=hazy_latent,
                cond_embeds=cond_embeds,
                cond_text_ids=cond_text_ids,
                uncond_embeds=uncond_embeds,
                uncond_text_ids=uncond_text_ids,
                guidance_scale=3.5,
                num_inference_steps=1,
                device="cpu",
                noise_seed=42,
            )

        # 4 calls: cond, uncond, CFG-cond, CFG-uncond
        assert len(call_records) == 4, f"Expected 4 calls, got {len(call_records)}"

        # Each call should have noisy tokens + ref tokens
        for rec in call_records:
            hs = rec["hidden_states_shape"]
            # 4x4=16 noisy + 4x4=16 ref = 32 tokens
            assert hs[1] == 32  # total tokens

        # img_ids should have index 0.0 for first half, 10.0 for second half
        for rec in call_records:
            ids = rec["img_ids"]
            N = ids.shape[1] // 2
            assert torch.all(ids[:, :N, 0] == 0.0), "Noisy token index should be 0.0"
            assert torch.all(ids[:, N:, 0] == 10.0), "Ref token index should be 10.0"

        # CFG output should differ from cond/uncond (guidance_scale != 1)
        assert not torch.allclose(result["cfg"], result["cond"])

    def test_denoise_all_modes_guidance_scale_effect(self):
        """Changing guidance_scale changes CFG output when cond != uncond."""
        import numpy as np
        from dehaze_lora.validate import _denoise_all_modes

        B, C, H, W = 1, 128, 4, 4
        N_txt = 8

        hazy_latent = torch.randn(B, C, H, W)
        cond_embeds = torch.randn(B, N_txt, 4096)
        uncond_embeds = torch.randn(B, N_txt, 4096) + 100  # different!
        cond_text_ids = torch.zeros(B, N_txt, 4, dtype=torch.int64)
        cond_text_ids[:, :, 3] = torch.arange(N_txt)
        uncond_text_ids = cond_text_ids.clone()

        class FakeTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockTransformerConfig()

            def forward(self, hidden_states, encoder_hidden_states, timestep,
                        img_ids, txt_ids, return_dict=False):
                B_in, total, C_in = hidden_states.shape
                # Distinguishable by encoder_hidden_states
                if torch.allclose(encoder_hidden_states[:1, 0, :1], cond_embeds[:1, 0, :1]):
                    val = torch.ones(B_in, total, C_in) * 2.0  # cond
                else:
                    val = torch.ones(B_in, total, C_in) * 1.0  # uncond
                return (val,)

        fake_scheduler = self._make_fake_scheduler(num_steps=1)

        def fake_decode(vae, latents):
            return latents[:, :3, :, :]

        with patch("dehaze_lora.validate.decode_vae_image", fake_decode), \
             patch("dehaze_lora.validate.compute_empirical_mu", return_value=1.0):
            result_low = _denoise_all_modes(
                transformer=FakeTransformer(),
                scheduler=fake_scheduler,
                vae=None,
                hazy_latent=hazy_latent,
                cond_embeds=cond_embeds,
                cond_text_ids=cond_text_ids,
                uncond_embeds=uncond_embeds,
                uncond_text_ids=uncond_text_ids,
                guidance_scale=1.0,
                num_inference_steps=1,
                device="cpu",
                noise_seed=42,
            )

            result_high = _denoise_all_modes(
                transformer=FakeTransformer(),
                scheduler=fake_scheduler,
                vae=None,
                hazy_latent=hazy_latent,
                cond_embeds=cond_embeds,
                cond_text_ids=cond_text_ids,
                uncond_embeds=uncond_embeds,
                uncond_text_ids=uncond_text_ids,
                guidance_scale=7.0,
                num_inference_steps=1,
                device="cpu",
                noise_seed=42,
            )

        # At guidance=1, CFG = cond (both should be 2.0)
        # At guidance=7, CFG = 1 + 7*(2-1) = 8.0 (simplified for first step)
        assert not torch.allclose(result_low["cfg"], result_high["cfg"]), \
            "Different guidance scales should produce different CFG outputs"

    def test_denoise_all_modes_guidance_1_equals_cond(self):
        """When guidance_scale=1, CFG output should match cond output."""
        import numpy as np
        from dehaze_lora.validate import _denoise_all_modes

        B, C, H, W = 1, 128, 4, 4
        N_txt = 8

        hazy_latent = torch.randn(B, C, H, W)
        cond_embeds = torch.randn(B, N_txt, 4096)
        uncond_embeds = torch.randn(B, N_txt, 4096) + 100
        cond_text_ids = torch.zeros(B, N_txt, 4, dtype=torch.int64)
        cond_text_ids[:, :, 3] = torch.arange(N_txt)
        uncond_text_ids = cond_text_ids.clone()

        class FakeTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockTransformerConfig()

            def forward(self, hidden_states, encoder_hidden_states, timestep,
                        img_ids, txt_ids, return_dict=False):
                B_in, total, C_in = hidden_states.shape
                if torch.allclose(encoder_hidden_states[:1, 0, :1], cond_embeds[:1, 0, :1]):
                    val = torch.full((B_in, total, C_in), 2.0)
                else:
                    val = torch.full((B_in, total, C_in), 0.0)
                return (val,)

        fake_scheduler = self._make_fake_scheduler(num_steps=1)

        def fake_decode(vae, latents):
            return latents[:, :3, :, :]

        with patch("dehaze_lora.validate.decode_vae_image", fake_decode), \
             patch("dehaze_lora.validate.compute_empirical_mu", return_value=1.0):
            result = _denoise_all_modes(
                transformer=FakeTransformer(),
                scheduler=fake_scheduler,
                vae=None,
                hazy_latent=hazy_latent,
                cond_embeds=cond_embeds,
                cond_text_ids=cond_text_ids,
                uncond_embeds=uncond_embeds,
                uncond_text_ids=uncond_text_ids,
                guidance_scale=1.0,
                num_inference_steps=1,
                device="cpu",
                noise_seed=42,
            )

        # CFG = uncond + 1.0*(cond - uncond) = cond
        assert torch.allclose(result["cfg"], result["cond"]), \
            "At guidance_scale=1, CFG should equal cond"
