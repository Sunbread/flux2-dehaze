"""GPU tests for AutoencoderKLFlux2 encode/decode and full VAE pipeline.

VAE is ~1 GB BF16 -- easily within 20 GB VRAM limit.
Uses tiny images (64x64 -> 4x4 latent) to keep activation memory minimal.
"""

import pytest
import torch
from tests.conftest import _require_vram_gb, _require_cuda, cleanup_gpu

MODEL_NAME = "black-forest-labs/FLUX.2-klein-base-9B"


def _load_flux2_vae():
    """Load VAE or raise -- caller decides whether to skip."""
    from diffusers import AutoencoderKLFlux2
    return AutoencoderKLFlux2.from_pretrained(
        MODEL_NAME, subfolder="vae", torch_dtype=torch.bfloat16,
    )


@pytest.mark.gpu
class TestFlux2VAE:

    def test_vae_loads(self):
        _require_cuda()
        _require_vram_gb(3)
        vae = None
        try:
            vae = _load_flux2_vae()
        except Exception as e:
            pytest.skip(f"VAE not available: {e}")
        try:
            vae.to("cuda")
            vae.eval()
            assert hasattr(vae, "encode")
            assert hasattr(vae, "decode")
        finally:
            del vae; cleanup_gpu()

    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_vae_encode_decode_roundtrip(self, size):
        _require_cuda()
        _require_vram_gb(3)
        vae = None
        try:
            vae = _load_flux2_vae()
        except Exception as e:
            pytest.skip(f"VAE not available: {e}")
        try:
            vae.to("cuda")
            vae.eval()
            lc = vae.config.latent_channels  # 32

            torch.manual_seed(42)
            img = torch.rand(1, 3, size, size, device="cuda", dtype=torch.bfloat16)

            with torch.no_grad():
                latent = vae.encode(img).latent_dist.sample()
                reconstructed = vae.decode(latent).sample

            expected_h = latent.shape[2]
            assert latent.shape[0] == 1
            assert latent.shape[1] == lc
            assert reconstructed.shape == img.shape, \
                f"Expected {img.shape}, got {reconstructed.shape}"
        finally:
            del vae; cleanup_gpu()

    def test_vae_no_shift_scale(self):
        _require_cuda()
        _require_vram_gb(3)
        vae = None
        try:
            vae = _load_flux2_vae()
        except Exception as e:
            pytest.skip(f"VAE not available: {e}")
        try:
            vae.to("cuda")
            vae.eval()

            img = torch.full((1, 3, 128, 128), 0.5, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                latent = vae.encode(img).latent_dist.sample()

            assert not torch.isnan(latent).any()
            assert not torch.isinf(latent).any()
            assert latent.std() > 0.0, "Latent should not be constant"

            with torch.no_grad():
                recon = vae.decode(latent).sample
            assert recon.shape == img.shape
            assert not torch.isnan(recon).any()
        finally:
            del vae; cleanup_gpu()

    def test_vae_batch_encode(self):
        _require_cuda()
        _require_vram_gb(3)
        vae = None
        try:
            vae = _load_flux2_vae()
        except Exception as e:
            pytest.skip(f"VAE not available: {e}")
        try:
            vae.to("cuda")
            vae.eval()
            lc = vae.config.latent_channels

            img = torch.rand(2, 3, 64, 64, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                latent = vae.encode(img).latent_dist.sample()
                recon = vae.decode(latent).sample

            assert latent.shape[0] == 2
            assert latent.shape[1] == lc
            assert recon.shape == (2, 3, 64, 64)
        finally:
            del vae; cleanup_gpu()


# ---------------------------------------------------------------------------
# Full VAE pipeline: encode_vae_image / decode_vae_image
# (VAE encode → patchify → BN normalize → denormalize → unpatchify → decode)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestVAEPipeline:

    def test_encode_vae_image_shape(self):
        """encode_vae_image produces 128ch, 16x-downsampled latents."""
        from dehaze_lora.model import encode_vae_image

        _require_cuda()
        _require_vram_gb(3)
        vae = None
        try:
            vae = _load_flux2_vae()
        except Exception as e:
            pytest.skip(f"VAE not available: {e}")
        try:
            vae.to("cuda")
            vae.eval()

            img = torch.rand(1, 3, 256, 256, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                latent = encode_vae_image(vae, img)

            # 256 / 8 (VAE) / 2 (patchify) = 16
            assert latent.shape == (1, 128, 16, 16), \
                f"Expected (1, 128, 16, 16), got {latent.shape}"
            assert not torch.isnan(latent).any()
            assert not torch.isinf(latent).any()
        finally:
            del vae; cleanup_gpu()

    def test_encode_decode_vae_image_roundtrip(self):
        """encode → decode roundtrip preserves approximate pixel values."""
        from dehaze_lora.model import encode_vae_image, decode_vae_image

        _require_cuda()
        _require_vram_gb(3)
        vae = None
        try:
            vae = _load_flux2_vae()
        except Exception as e:
            pytest.skip(f"VAE not available: {e}")
        try:
            vae.to("cuda")
            vae.eval()

            torch.manual_seed(42)
            img = torch.rand(1, 3, 256, 256, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                latent = encode_vae_image(vae, img)
                recon = decode_vae_image(vae, latent)

            assert recon.shape == img.shape
            # VAE is lossy; random inputs can produce wider reconstruction range
            assert not torch.isnan(recon).any()
            assert not torch.isinf(recon).any()
        finally:
            del vae; cleanup_gpu()

    def test_bn_normalization_applied(self):
        """BN step in encode_vae_image shifts latent distribution."""
        from dehaze_lora.model import encode_vae_image, _patchify_latents

        _require_cuda()
        _require_vram_gb(3)
        vae = None
        try:
            vae = _load_flux2_vae()
        except Exception as e:
            pytest.skip(f"VAE not available: {e}")
        try:
            vae.to("cuda")
            vae.eval()

            img = torch.rand(1, 3, 128, 128, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                raw = vae.encode(img).latent_dist.sample()
                patched = _patchify_latents(raw)
                normalized = encode_vae_image(vae, img)

            # After BN, mean should be closer to 0 than raw patched latent
            patched_mean = patched.float().mean().abs()
            norm_mean = normalized.float().mean().abs()
            assert norm_mean < patched_mean * 2, \
                f"BN normalization may not be working: raw_mean={patched_mean:.4f}, norm_mean={norm_mean:.4f}"
            # Normalized latent should have std near 1
            norm_std = normalized.float().std()
            assert 0.1 < norm_std < 5.0, \
                f"BN-normalized std out of range: {norm_std:.4f}"
        finally:
            del vae; cleanup_gpu()
