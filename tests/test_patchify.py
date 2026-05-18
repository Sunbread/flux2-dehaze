"""Tests for patchify / unpatchify low-level tensor ops."""

import torch
import pytest
from dehaze_lora.model import patchify_and_make_ids, unpatchify


class TestPatchifyRoundtrip:
    """unpatchify(patchify(x)) == x for various shapes."""

    @pytest.mark.parametrize("shape", [
        (1, 128, 32, 32),
        (2, 128, 16, 16),
        (1, 128, 64, 64),
        (4, 128, 8, 8),
    ])
    def test_roundtrip(self, shape):
        x = torch.randn(*shape)
        tokens, _ids = patchify_and_make_ids(x, patch_size=1)
        recovered = unpatchify(tokens, shape[2], shape[3], patch_size=1)
        assert torch.allclose(x, recovered, atol=1e-5)

    @pytest.mark.parametrize("h,w", [(31, 31), (33, 33), (31, 64), (64, 33)])
    def test_non_divisible_sizes_padded(self, h, w):
        """Non-divisible sizes are padded then cropped back."""
        x = torch.randn(1, 128, h, w)
        tokens, _ids = patchify_and_make_ids(x, patch_size=1)
        recovered = unpatchify(tokens, h, w, patch_size=1)
        assert recovered.shape == (1, 128, h, w)
        assert torch.allclose(x, recovered, atol=1e-5)


class TestPositionIDs:
    """Flux2 position ID format: (B, N, 4) with axes [index, h, w, 0]."""

    def test_ids_shape(self):
        x = torch.randn(2, 128, 16, 32)
        _tokens, ids = patchify_and_make_ids(x, patch_size=1, index=0.0)
        assert ids.shape == (2, 16 * 32, 4)

    def test_dim0_is_index(self):
        x = torch.randn(1, 128, 8, 8)
        for idx_val in [0.0, 10.0, -5.0]:
            _tokens, ids = patchify_and_make_ids(x, patch_size=1, index=idx_val)
            assert torch.allclose(ids[0, :, 0], torch.full((64,), idx_val))

    def test_dim1_is_h_linspace(self):
        x = torch.randn(1, 128, 4, 8)
        _tokens, ids = patchify_and_make_ids(x, patch_size=1, index=0.0)
        h_coords = ids[0, :, 1].reshape(4, 8)
        for row in range(4):
            assert torch.allclose(h_coords[row, :], torch.full((8,), float(row)))

    def test_dim2_is_w_linspace(self):
        x = torch.randn(1, 128, 4, 8)
        _tokens, ids = patchify_and_make_ids(x, patch_size=1, index=0.0)
        w_coords = ids[0, :, 2].reshape(4, 8)
        expected = torch.arange(8, dtype=torch.float32).unsqueeze(0).expand(4, -1)
        assert torch.allclose(w_coords, expected)

    def test_dim3_is_zero(self):
        x = torch.randn(1, 128, 8, 8)
        _tokens, ids = patchify_and_make_ids(x, patch_size=1, index=0.0)
        assert torch.all(ids[:, :, 3] == 0.0)

    def test_batch_expansion(self):
        x = torch.randn(3, 128, 4, 4)
        _tokens, ids = patchify_and_make_ids(x, patch_size=1, index=0.0)
        assert ids.shape[0] == 3
        assert torch.allclose(ids[0], ids[1])
        assert torch.allclose(ids[1], ids[2])

    def test_ids_device(self):
        x = torch.randn(1, 128, 4, 4)
        _tokens, ids = patchify_and_make_ids(x, patch_size=1)
        assert ids.device == x.device


class TestPatchifyEdgeCases:
    def test_single_pixel(self):
        """1x1 latent should work."""
        x = torch.randn(1, 128, 1, 1)
        tokens, ids = patchify_and_make_ids(x, patch_size=1)
        assert tokens.shape == (1, 1, 128)
        assert ids.shape == (1, 1, 4)
        recovered = unpatchify(tokens, 1, 1, patch_size=1)
        assert torch.allclose(x, recovered, atol=1e-5)
