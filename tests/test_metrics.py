"""Tests for PSNR / SSIM metric computation (pure numpy, CPU)."""

import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class TestPSNR:
    def test_identical_images(self):
        img = np.random.rand(256, 256, 3).astype(np.float32)
        result = psnr(img, img, data_range=1.0)
        assert np.isinf(result) or result > 100

    def test_maximum_difference(self):
        black = np.zeros((64, 64, 3), dtype=np.float64)
        white = np.ones((64, 64, 3), dtype=np.float64)
        result = psnr(black, white, data_range=1.0)
        assert result < 1.0

    def test_known_noise_level(self):
        gt = np.ones((128, 128, 3), dtype=np.float64) * 0.5
        noisy = gt + np.random.randn(*gt.shape).astype(np.float64) * 0.01
        result = psnr(gt, noisy, data_range=1.0)
        assert 30 < result < 50

    def test_data_range_255(self):
        gt = np.full((32, 32, 3), 128, dtype=np.float64)
        noisy = gt + 1.0
        result = psnr(gt, noisy, data_range=255.0)
        assert 40 < result < 50


class TestSSIM:
    def test_identical_images(self):
        img = np.random.rand(128, 128, 3).astype(np.float64)
        result = ssim(img, img, data_range=1.0, channel_axis=2)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_different_images(self):
        img1 = np.random.rand(64, 64, 3).astype(np.float64)
        img2 = np.random.rand(64, 64, 3).astype(np.float64)
        result = ssim(img1, img2, data_range=1.0, channel_axis=2)
        assert result < 1.0
        assert result > -1.0  # SSIM can be negative for very different images

    def test_blurred_vs_sharp(self):
        from scipy.ndimage import gaussian_filter

        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 1, 128)
        xx, yy = np.meshgrid(x, y)
        sharp = np.stack([
            np.sin(xx * 20) * 0.5 + 0.5,
            np.cos(yy * 20) * 0.5 + 0.5,
            np.sin((xx + yy) * 10) * 0.5 + 0.5,
        ], axis=-1).astype(np.float64)

        blurred = np.stack([gaussian_filter(sharp[:, :, c], sigma=3.0)
                            for c in range(3)], axis=-1)

        result = ssim(sharp, blurred, data_range=1.0, channel_axis=2)
        assert 0.3 < result < 0.99, f"SSIM = {result}"

    def test_monochannel(self):
        img1 = np.random.rand(64, 64).astype(np.float64)
        img2 = img1 + 0.001
        result = ssim(img1, img2, data_range=1.0)
        assert result > 0.9


class TestMetricsEdgeCases:
    def test_clamp_output_range(self):
        """Output from VAE decode is clamped to [0, 1] before metrics."""
        raw = np.array([-0.1, 0.5, 1.2], dtype=np.float64)
        clamped = np.clip(raw, 0, 1)
        assert clamped[0] == 0.0
        assert clamped[2] == 1.0
        assert clamped[1] == 0.5

    def test_validate_output_format(self):
        """Simulate validate.py's tensor → numpy conversion path."""
        # Pretend this is VAE decode output: (1, 3, H, W), bf16 on GPU
        fake_output = np.random.rand(1, 3, 64, 64).astype(np.float32)

        # Simulate: squeeze(0).permute(1,2,0).cpu().float().clamp(0,1).numpy()
        clear_np = fake_output.squeeze(0).transpose(1, 2, 0).clip(0, 1)
        assert clear_np.shape == (64, 64, 3)
        assert clear_np.min() >= 0.0
        assert clear_np.max() <= 1.0

    def test_batch_psnr_aggregation(self):
        psnr_vals = [25.0, 26.0, 27.0, 24.0, 28.0]
        assert np.mean(psnr_vals) == pytest.approx(26.0)
        assert np.std(psnr_vals) > 0
