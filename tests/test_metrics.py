"""Tests for PSNR / SSIM metric computation via production helpers."""

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from dehaze_lora.validate import _compute_image_metrics


class TestComputeImageMetrics:
    """Tests for _compute_image_metrics (production helper)."""

    def test_identical_images(self):
        img = torch.rand(3, 64, 64)
        p, s = _compute_image_metrics(img, img)
        assert p > 100 or np.isinf(p)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_different_images(self):
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        p, s = _compute_image_metrics(img1, img2)
        assert p < 100
        assert s < 1.0

    def test_returns_python_floats(self):
        """_compute_image_metrics returns Python float, not numpy scalar."""
        img = torch.rand(3, 64, 64)
        p, s = _compute_image_metrics(img, img)
        assert isinstance(p, float)
        assert isinstance(s, float)
        assert not isinstance(p, np.floating)
        assert not isinstance(s, np.floating)

    def test_clamps_prediction_outside_0_1(self):
        """Values outside [0, 1] are clamped before metric computation."""
        gt = torch.ones(3, 64, 64) * 0.5
        pred = torch.randn(3, 64, 64) * 0.3 + 1.5  # some values > 1
        p, s = _compute_image_metrics(gt, pred)
        assert isinstance(p, float)
        assert isinstance(s, float)
        # Should not crash despite out-of-range pred values

    def test_accepts_3HW_tensors(self):
        """Inputs are (3, H, W) channel-first tensors."""
        gt = torch.rand(3, 128, 256)
        pred = torch.rand(3, 128, 256)
        p, s = _compute_image_metrics(gt, pred)
        assert isinstance(p, float)
        assert isinstance(s, float)

    def test_clamp_actually_changes_values(self):
        """Verify that clamping actually modifies out-of-range values."""
        gt = torch.ones(3, 32, 32)
        pred = torch.ones(3, 32, 32) * 2.0  # all values > 1

        # Clamp manually: pred_np should all be 1.0 after clamp
        pred_np = pred.permute(1, 2, 0).cpu().float().clamp(0, 1).numpy()
        assert pred_np.max() == 1.0  # was 2.0 before clamp

        # With clamping, pred == gt (both are all 1.0 after clamp), so perfect metrics
        p, s = _compute_image_metrics(gt, pred)
        assert p > 100 or np.isinf(p)  # identical after clamp


class TestRunValidationBatchEmpty:
    """run_validation_batch with empty val_subset returns empty lists."""

    def test_empty_val_subset(self):
        from dehaze_lora.validate import run_validation_batch

        result = run_validation_batch(
            vae=None, transformer=None, scheduler=None,
            text_encoder=None, tokenizer=None,
            val_subset=[],
        )
        assert result == {"images": [], "psnr": [], "ssim": []}


class TestRunValidationBatchMocked:
    """Mocked run_validation_batch produces one output per input row."""

    def test_mocked_path_returns_one_psnr_ssim_per_row(self, tmp_path):
        """With mocked encode/denoise, each row produces one PSNR/SSIM."""
        import json
        from PIL import Image

        # Create temp images for metadata
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        hazy_path = img_dir / "hazy.png"
        gt_path = img_dir / "gt.png"
        Image.new("RGB", (64, 64), color=(128, 64, 32)).save(hazy_path)
        Image.new("RGB", (64, 64), color=(100, 150, 200)).save(gt_path)

        val_subset = [
            {"image": str(hazy_path), "gt": str(gt_path)},
            {"image": str(hazy_path), "gt": str(gt_path)},
        ]

        # Mock denoising to return known tensors
        fake_cfg = torch.rand(2, 3, 512, 512)

        with patch("dehaze_lora.validate.encode_vae_image", return_value=torch.randn(2, 128, 32, 32)), \
             patch("dehaze_lora.validate.encode_prompts",
                   side_effect=[(torch.randn(2, 16, 12288), torch.zeros(2, 16, 4, dtype=torch.int64)),
                                (torch.randn(2, 16, 12288), torch.zeros(2, 16, 4, dtype=torch.int64))]), \
             patch("dehaze_lora.validate._denoise_all_modes", return_value={
                 "cond": torch.rand(2, 3, 512, 512),
                 "uncond": torch.rand(2, 3, 512, 512),
                 "cfg": fake_cfg,
             }):
            from dehaze_lora.validate import run_validation_batch

            result = run_validation_batch(
                vae=MagicMock(),
                transformer=MagicMock(),
                scheduler=MagicMock(),
                text_encoder=MagicMock(),
                tokenizer=MagicMock(),
                val_subset=val_subset,
                transformer_device="cpu",
                qwen_device="cpu",
            )

        assert len(result["images"]) == 2
        assert len(result["psnr"]) == 2
        assert len(result["ssim"]) == 2
        for p in result["psnr"]:
            assert isinstance(p, float)
        for s in result["ssim"]:
            assert isinstance(s, float)

    def test_cfg_out_of_range_values_are_clamped(self, tmp_path):
        """run_validation_batch clamps CFG values outside [0, 1]."""
        import json
        from PIL import Image

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        hazy_path = img_dir / "hazy.png"
        gt_path = img_dir / "gt.png"
        Image.new("RGB", (64, 64), color=(128, 64, 32)).save(hazy_path)
        Image.new("RGB", (64, 64), color=(100, 150, 200)).save(gt_path)

        val_subset = [{"image": str(hazy_path), "gt": str(gt_path)}]

        # CFG output with values outside [0, 1]
        fake_cfg = torch.ones(1, 3, 512, 512) * 3.0

        with patch("dehaze_lora.validate.encode_vae_image", return_value=torch.randn(1, 128, 32, 32)), \
             patch("dehaze_lora.validate.encode_prompts",
                   side_effect=[(torch.randn(1, 16, 12288), torch.zeros(1, 16, 4, dtype=torch.int64)),
                                (torch.randn(1, 16, 12288), torch.zeros(1, 16, 4, dtype=torch.int64))]), \
             patch("dehaze_lora.validate._denoise_all_modes", return_value={
                 "cond": torch.rand(1, 3, 512, 512),
                 "uncond": torch.rand(1, 3, 512, 512),
                 "cfg": fake_cfg,
             }):
            from dehaze_lora.validate import run_validation_batch

            result = run_validation_batch(
                vae=MagicMock(),
                transformer=MagicMock(),
                scheduler=MagicMock(),
                text_encoder=MagicMock(),
                tokenizer=MagicMock(),
                val_subset=val_subset,
                transformer_device="cpu",
                qwen_device="cpu",
            )

        # Should not crash; clamping handles out-of-range values
        assert len(result["psnr"]) == 1
