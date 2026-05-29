"""Tests for CFG two-pass inference orchestration logic.

Verifies token concatenation, CFG formula, output slicing,
and scheduler integration with mocked models.
"""

import pytest
import torch
from unittest.mock import MagicMock

from dehaze_lora.model import patchify_and_make_ids, unpatchify


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockTransformerConfig:
    patch_size = 1


# ---------------------------------------------------------------------------
# Token concatenation order
# ---------------------------------------------------------------------------

class TestTokenConcat:
    """Verify [noisy_tokens | ref_tokens] concatenation order."""

    def test_noisy_before_ref(self):
        B, C, H, W = 1, 16, 4, 4
        noisy = torch.randn(B, C, H, W)
        ref = torch.randn(B, C, H, W) + 100  # easily distinguishable

        noisy_tokens, noisy_ids = patchify_and_make_ids(noisy, patch_size=1, index=0.0)
        ref_tokens, ref_ids = patchify_and_make_ids(ref, patch_size=1, index=10.0)

        N_noisy = noisy_tokens.shape[1]
        N_ref = ref_tokens.shape[1]

        combined = torch.cat([noisy_tokens, ref_tokens], dim=1)
        assert combined.shape == (B, N_noisy + N_ref, C)

        # First N_noisy tokens should be the noisy ones
        assert torch.allclose(combined[:, :N_noisy, :], noisy_tokens, atol=1e-5)
        # Last N_ref tokens should be the ref ones
        assert torch.allclose(combined[:, N_noisy:, :], ref_tokens, atol=1e-5)

    def test_output_slicing_extracts_only_noisy(self):
        """After transformer forward, output[:, :N_noisy, :] is kept."""
        B, C = 1, 16
        N_noisy, N_ref = 16, 16

        model_output = torch.randn(B, N_noisy + N_ref, C)
        extracted = model_output[:, :N_noisy, :]
        assert extracted.shape == (B, N_noisy, C)

    def test_ref_ids_have_index_10(self):
        B, C, H, W = 1, 16, 4, 4
        noisy = torch.randn(B, C, H, W)
        ref = torch.randn(B, C, H, W)

        noisy_tokens, noisy_ids = patchify_and_make_ids(noisy, patch_size=1, index=0.0)
        ref_tokens, ref_ids = patchify_and_make_ids(ref, patch_size=1, index=10.0)

        combined_ids = torch.cat([noisy_ids, ref_ids], dim=1)
        N_noisy = noisy_ids.shape[1]

        assert torch.all(combined_ids[:, :N_noisy, 0] == 0.0)
        assert torch.all(combined_ids[:, N_noisy:, 0] == 10.0)


# ---------------------------------------------------------------------------
# CFG formula
# ---------------------------------------------------------------------------

class TestCFGFormula:
    """v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)."""

    def test_no_guidance_equals_uncond(self):
        v_cond = torch.tensor([3.0, 5.0])
        v_uncond = torch.tensor([1.0, 2.0])
        guidance_scale = 0.0
        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
        assert torch.allclose(v_cfg, v_uncond)

    def test_guidance_1_equals_cond(self):
        v_cond = torch.tensor([3.0, 5.0])
        v_uncond = torch.tensor([1.0, 2.0])
        guidance_scale = 1.0
        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
        assert torch.allclose(v_cfg, v_cond)

    def test_guidance_3_5_extrapolates(self):
        v_cond = torch.tensor([2.0, 4.0])
        v_uncond = torch.tensor([0.0, 2.0])
        guidance_scale = 3.5
        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
        expected = torch.tensor([7.0, 9.0])
        assert torch.allclose(v_cfg, expected)

    def test_cfg_shape_preserved(self):
        B, C, H, W = 2, 128, 16, 16
        v_cond = torch.randn(B, C, H, W)
        v_uncond = torch.randn(B, C, H, W)
        guidance_scale = 3.5
        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
        assert v_cfg.shape == (B, C, H, W)


# ---------------------------------------------------------------------------
# Scheduler integration (synthetic)
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
        # After one step with random v_pred, latent should change
        assert not torch.allclose(out.prev_sample, z)

    def test_full_denoising_loop_shape(self):
        """Run a full synthetic denoising loop with zero-velocity predictions."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(28)

        z = torch.randn(1, 128, 8, 8)
        initial = z.clone()

        for t in scheduler.timesteps:
            v_pred = torch.zeros_like(z)  # predict zero velocity
            out = scheduler.step(v_pred, t, z)
            z = out.prev_sample

        assert z.shape == initial.shape
        # Shape preserved through full 28-step denoising loop.
        # (zero-velocity prediction exercises scheduler dynamics;
        #  output differs from input but we only assert shape here)


# ---------------------------------------------------------------------------
# Mocked end-to-end orchestration
# ---------------------------------------------------------------------------

class TestDehazeOrchestration:
    """Test the CFG denoising orchestration logic with a mocked transformer."""

    def test_cfg_loop_orchestration(self):
        """Simulate one denoising step with mocked models."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        B, C, H, W = 1, 128, 8, 8
        N_txt = 16

        # Synthetic inputs (mimicking after VAE encode + text encode)
        z = torch.randn(B, C, H, W)
        cond_embeds = torch.randn(B, N_txt, 4096)
        uncond_embeds = torch.randn(B, N_txt, 4096)
        cond_text_ids = torch.zeros(N_txt, 4)
        cond_text_ids[:, 3] = torch.arange(N_txt).float()
        uncond_text_ids = cond_text_ids.clone()
        hazy_latent = torch.randn(B, C, H, W)

        # Patchify ref (once)
        ref_tokens, ref_ids = patchify_and_make_ids(
            hazy_latent, patch_size=1, index=10.0,
        )

        # Patchify noisy
        noisy_tokens, noisy_ids = patchify_and_make_ids(
            z, patch_size=1, index=0.0,
        )

        N_noisy = noisy_tokens.shape[1]

        # Concat [noisy | ref]
        img_hidden = torch.cat([noisy_tokens, ref_tokens], dim=1)
        img_ids_combined = torch.cat([noisy_ids, ref_ids], dim=1)

        # Mock transformer: return zeros for cond, ones for uncond
        total_tokens = img_hidden.shape[1]

        call_count = [0]

        def mock_transformer(hidden_states, encoder_hidden_states, timestep,
                             img_ids, txt_ids, return_dict=False):
            call_count[0] += 1
            B_in, total, C_in = hidden_states.shape
            if call_count[0] == 1:  # cond
                val = torch.ones(B_in, total, C_in) * 2.0
            else:  # uncond
                val = torch.ones(B_in, total, C_in) * 1.0
            return (val,)

        trans = MagicMock()
        trans.config = MockTransformerConfig()
        trans.side_effect = mock_transformer

        # Cond forward
        v_cond = trans(
            hidden_states=img_hidden,
            encoder_hidden_states=cond_embeds,
            timestep=None,
            img_ids=img_ids_combined,
            txt_ids=cond_text_ids,
            return_dict=False,
        )[0]
        v_cond = v_cond[:, :N_noisy, :]
        v_cond = unpatchify(v_cond, z.shape[2], z.shape[3], patch_size=1)

        # Uncond forward
        v_uncond = trans(
            hidden_states=img_hidden,
            encoder_hidden_states=uncond_embeds,
            timestep=None,
            img_ids=img_ids_combined,
            txt_ids=uncond_text_ids,
            return_dict=False,
        )[0]
        v_uncond = v_uncond[:, :N_noisy, :]
        v_uncond = unpatchify(v_uncond, z.shape[2], z.shape[3], patch_size=1)

        # CFG
        guidance_scale = 3.5
        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

        assert v_cfg.shape == (B, C, H, W)
        # v_cond=2, v_uncond=1, guidance=3.5 → v_cfg = 1 + 3.5*(2-1) = 4.5
        assert torch.allclose(v_cfg, torch.full_like(v_cfg, 4.5))

    def test_guidance_scale_effect(self):
        """Higher guidance → larger difference from uncond."""
        shapes = (1, 128, 8, 8)
        v_cond = torch.full(shapes, 5.0)
        v_uncond = torch.full(shapes, 1.0)

        cfg_low = v_uncond + 1.0 * (v_cond - v_uncond)
        cfg_high = v_uncond + 7.0 * (v_cond - v_uncond)

        assert torch.allclose(cfg_low, torch.full(shapes, 5.0))
        assert torch.allclose(cfg_high, torch.full(shapes, 29.0))
        assert (cfg_high - cfg_low).abs().sum() > 0
