"""Tests for text_ids generation format.

Flux2 uses 4D rope axes [32, 32, 32, 32]. Text only populates dim 3.
"""

import torch
import pytest


def _make_text_ids(seq_len, device="cpu", dtype=torch.float32):
    """Replicate encode_prompts text_ids logic for isolated testing."""
    text_ids = torch.zeros(seq_len, 4, device=device, dtype=dtype)
    text_ids[:, 3] = torch.linspace(
        0, seq_len - 1, steps=seq_len, device=device, dtype=dtype
    )
    return text_ids


class TestTextIDs:
    def test_shape(self):
        for sl in [1, 32, 256, 512]:
            ids = _make_text_ids(sl)
            assert ids.shape == (sl, 4)

    def test_dim3_is_linspace(self):
        ids = _make_text_ids(10)
        expected = torch.arange(10, dtype=torch.float32)
        assert torch.allclose(ids[:, 3], expected)

    def test_dims_0_1_2_are_zero(self):
        ids = _make_text_ids(50)
        for dim in [0, 1, 2]:
            assert torch.all(ids[:, dim] == 0.0)

    def test_seq_len_1(self):
        ids = _make_text_ids(1)
        assert ids[0, 3].item() == 0.0
        assert ids[0, 0].item() == 0.0

    def test_dtype_preserved(self):
        for dtype in [torch.float32, torch.bfloat16]:
            ids = _make_text_ids(8, dtype=dtype)
            assert ids.dtype == dtype

    def test_device(self):
        ids = _make_text_ids(4, device="cpu")
        assert ids.device.type == "cpu"

    @pytest.mark.gpu
    def test_device_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        ids = _make_text_ids(4, device="cuda")
        assert ids.device.type == "cuda"


class TestPrepareTextIDs:
    """Test the actual _prepare_text_ids production function."""

    def test_matches_reference_impl(self):
        from dehaze_lora.model import _prepare_text_ids

        for L in [1, 32, 256, 512]:
            # _prepare_text_ids takes (B, L, D) input, returns (B, L, 4)
            prompt_embeds = torch.randn(1, L, 12288)
            ids = _prepare_text_ids(prompt_embeds)

            assert ids.shape == (1, L, 4)
            assert ids.dtype == torch.int64  # cartesian_prod preserves int64 input dtype
            # dims 0,1,2 are zero
            assert torch.all(ids[0, :, 0] == 0)
            assert torch.all(ids[0, :, 1] == 0)
            assert torch.all(ids[0, :, 2] == 0)
            # dim 3 is 0..L-1
            assert torch.equal(ids[0, :, 3], torch.arange(L))

    def test_batch_dimension(self):
        from dehaze_lora.model import _prepare_text_ids

        B, L = 3, 64
        prompt_embeds = torch.randn(B, L, 12288)
        ids = _prepare_text_ids(prompt_embeds)

        assert ids.shape == (B, L, 4)
        # Each batch element has same pattern
        for b in range(B):
            assert torch.equal(ids[b, :, 3], torch.arange(L))
