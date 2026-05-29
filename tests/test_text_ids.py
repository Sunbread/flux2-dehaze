"""Tests for text_ids generation format.

Flux2 uses 4D rope axes [32, 32, 32, 32]. Text only populates dim 3.
All tests use the production _prepare_text_ids function.
"""

import torch
import pytest

from dehaze_lora.model import _prepare_text_ids


class TestPrepareTextIDs:
    """Test the production _prepare_text_ids function."""

    def test_basic_shape(self):
        for L in [1, 32, 256, 512]:
            prompt_embeds = torch.randn(1, L, 12288)
            ids = _prepare_text_ids(prompt_embeds)
            assert ids.shape == (1, L, 4)

    def test_seq_len_1(self):
        prompt_embeds = torch.randn(1, 1, 12288)
        ids = _prepare_text_ids(prompt_embeds)
        assert ids.shape == (1, 1, 4)
        assert ids[0, 0, 3].item() == 0
        assert ids[0, 0, 0].item() == 0

    def test_dims_0_1_2_are_zero(self):
        prompt_embeds = torch.randn(1, 50, 12288)
        ids = _prepare_text_ids(prompt_embeds)
        assert torch.all(ids[0, :, 0] == 0)
        assert torch.all(ids[0, :, 1] == 0)
        assert torch.all(ids[0, :, 2] == 0)

    def test_dim_3_is_arange(self):
        prompt_embeds = torch.randn(1, 10, 12288)
        ids = _prepare_text_ids(prompt_embeds)
        expected = torch.arange(10, dtype=torch.int64)
        assert torch.equal(ids[0, :, 3], expected)

    def test_dtype_is_int64(self):
        prompt_embeds = torch.randn(1, 16, 12288)
        ids = _prepare_text_ids(prompt_embeds)
        assert ids.dtype == torch.int64

    def test_batch_dimension(self):
        B, L = 3, 64
        prompt_embeds = torch.randn(B, L, 12288)
        ids = _prepare_text_ids(prompt_embeds)
        assert ids.shape == (B, L, 4)
        for b in range(B):
            assert torch.equal(ids[b, :, 3], torch.arange(L))

    def test_device_cpu(self):
        prompt_embeds = torch.randn(1, 8, 12288)
        ids = _prepare_text_ids(prompt_embeds)
        assert ids.device.type == "cpu"

    @pytest.mark.gpu
    def test_device_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        prompt_embeds = torch.randn(1, 8, 12288, device="cuda")
        ids = _prepare_text_ids(prompt_embeds)
        assert ids.device.type == "cuda"
