"""Type boundary tests: verifying explicit conversions preserve correctness.

These tests validate that type coercions (int(), float(), str(), etc.)
applied at boundaries do not change behavior. They run on CPU only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ─── model.py boundary tests ───────────────────────────────────────────

from src.dehaze_lora.model import (
    patchify_and_make_ids,
    unpatchify,
    _prepare_text_ids,
    encode_prompts,
)


class TestPatchifyExplicitTypes:
    """Verify patchify_and_make_ids works with values that need explicit conversion."""

    def test_patchify_accepts_float_index(self):
        """index should be explicitly converted from numpy float/int."""
        x = torch.randn(2, 128, 4, 4, dtype=torch.float32)
        tokens, ids = patchify_and_make_ids(
            x, patch_size=1, index=np.float64(10.0), axes_dim=4,
        )
        assert tokens.shape == (2, 16, 128)
        assert ids.shape == (2, 16, 4)
        assert ids.dtype == torch.float32
        # index value preserved as float 10.0 in the 0th dimension
        assert torch.allclose(ids[:, :, 0], torch.tensor(10.0))

    def test_patchify_works_with_numpy_int_patch_size(self):
        """patch_size explicitly converted from numpy int64."""
        x = torch.randn(1, 32, 4, 4, dtype=torch.float32)
        tokens, ids = patchify_and_make_ids(
            x, patch_size=int(np.int64(1)), index=0.0, axes_dim=4,
        )
        assert tokens.shape == (1, 16, 32)
        assert ids.shape == (1, 16, 4)

    def test_patchify_zero_index(self):
        """index=0 (int) should produce 0.0 in ids."""
        x = torch.randn(1, 128, 4, 4, dtype=torch.float32)
        tokens, ids = patchify_and_make_ids(
            x, patch_size=1, index=0, axes_dim=4,
        )
        assert torch.allclose(ids[:, :, 0], torch.tensor(0.0))

    def test_patchify_ids_always_float32(self):
        """ids must always be float32 regardless of input types."""
        x = torch.randn(1, 128, 4, 4, dtype=torch.float32)
        _, ids = patchify_and_make_ids(x, patch_size=1, index=10.0, axes_dim=4)
        assert ids.dtype == torch.float32

    def test_patchify_ids_index_preserved(self):
        """Index 10.0 for reference tokens is preserved as float."""
        x = torch.randn(1, 128, 4, 4, dtype=torch.float32)
        _, ids = patchify_and_make_ids(
            x, patch_size=1, index=np.float32(10.0), axes_dim=4,
        )
        # Each position ID has [index, h, w, 0] along last dim
        assert ids[:, :, 0].unique().item() == pytest.approx(10.0)
        assert ids[:, :, 3].unique().item() == pytest.approx(0.0)


class TestUnpatchifyExplicitInt:
    """Verify unpatchify works with numpy int boundaries."""

    def test_unpatchify_with_numpy_int64_h_orig(self):
        """h_orig/w_orig explicitly converted from numpy int64."""
        # unpatchify expects B(HW)C 3D tokens, h_orig=w_orig=1, patch_size=1
        # For a 1x1 grid: C=128 → after unpatchify: (B, 128, 1, 1)
        x = torch.randn(1, 1, 128, dtype=torch.float32)
        result = unpatchify(x, np.int64(1), np.int64(1), patch_size=1)
        assert result.shape == (1, 128, 1, 1)

    def test_unpatchify_roundtrip_with_numpy_boundaries(self):
        """Roundtrip through patchify → unpatchify preserves shape with numpy ints."""
        original = torch.randn(2, 128, 4, 8, dtype=torch.float32)
        tokens, ids = patchify_and_make_ids(original, patch_size=1, index=0.0)
        recovered = unpatchify(
            tokens,
            h_orig=np.int32(4),
            w_orig=np.int32(8),
            patch_size=np.int32(1),
        )
        assert recovered.shape == original.shape

    def test_unpatchify_from_tensor_shape_boundaries(self):
        """h_orig/w_orig derived from tensor shape attrs (from_shape as int)."""
        original = torch.randn(2, 128, 4, 8, dtype=torch.float32)
        tokens, ids = patchify_and_make_ids(original, patch_size=1, index=0.0)
        # Use tensor shape attributes (which are already Python int)
        h = int(original.shape[2])
        w = int(original.shape[3])
        recovered = unpatchify(tokens, h, w, patch_size=1)
        assert recovered.shape == original.shape


class TestTextIdsInt64Boundary:
    """_prepare_text_ids must return int64 — verified by existing tests."""

    def test_text_ids_returns_int64(self):
        """Explicitly assert _prepare_text_ids returns torch.int64."""
        prompt_embeds = torch.randn(2, 16, 12288)
        ids = _prepare_text_ids(prompt_embeds)
        assert ids.dtype == torch.int64
        assert ids.shape == (2, 16, 4)

    def test_text_ids_sequence_dimension_is_arange(self):
        """The last dimension (L coordinate) is arange(seq_len) in int64."""
        prompt_embeds = torch.randn(1, 8, 12288)
        ids = _prepare_text_ids(prompt_embeds)
        expected_l = torch.arange(8, dtype=torch.int64)
        assert torch.equal(ids[0, :, 3], expected_l)

    def test_text_ids_t_h_w_zero_for_text(self):
        """T, H, W coordinates are all zero for text tokens."""
        prompt_embeds = torch.randn(1, 8, 12288)
        ids = _prepare_text_ids(prompt_embeds)
        assert torch.all(ids[:, :, 0] == 0)  # T
        assert torch.all(ids[:, :, 1] == 0)  # H
        assert torch.all(ids[:, :, 2] == 0)  # W


# ─── Metric boundary tests ─────────────────────────────────────────────

class TestMetricFloatConversions:
    """PSNR/SSIM return Python float, not numpy scalar."""

    def test_psnr_returns_python_float(self):
        """float(psnr(...)) should be Python float, not np.float64."""
        gt = np.random.rand(64, 64, 3).astype(np.float64)
        noisy = gt + np.random.randn(*gt.shape).astype(np.float64) * 0.01
        result = float(psnr(gt, noisy, data_range=1.0))
        assert isinstance(result, float)
        assert not isinstance(result, np.float64)
        assert not isinstance(result, np.floating)

    def test_ssim_returns_python_float(self):
        """float(ssim(...)) should be Python float."""
        gt = np.random.rand(64, 64, 3).astype(np.float64)
        noisy = gt + np.random.randn(*gt.shape).astype(np.float64) * 0.01
        result = float(ssim(gt, noisy, data_range=1.0, channel_axis=2))
        assert isinstance(result, float)
        assert not isinstance(result, np.float64)

    def test_np_mean_produces_numpy_scalar(self):
        """np.mean returns numpy scalar — verify before wrapping."""
        values = [1.0, 2.0, 3.0]
        m = np.mean(values)
        assert isinstance(m, np.floating)

    def test_float_np_mean_produces_python_float(self):
        """float(np.mean(...)) should be Python float."""
        values = [1.0, 2.0, 3.0]
        result = float(np.mean(values))
        assert isinstance(result, float)
        assert not isinstance(result, np.floating)

    def test_tensor_item_float_chain(self):
        """tensor.item() produces Python number; float() ensures float."""
        t = torch.tensor(3.14159)
        result = float(t.item())
        assert isinstance(result, float)
        assert result == pytest.approx(3.14159)

    def test_tensor_int_item(self):
        """tensor long .item() returns Python int."""
        t = torch.tensor(42, dtype=torch.long)
        result = int(t.item())
        assert isinstance(result, int)
        assert result == 42


# ─── Config coercion tests ─────────────────────────────────────────────

class TestConfigCoercions:
    """Config values from YAML may be str/np int/np float — coerce explicitly."""

    def test_int_coercion_accepts_numeric_string(self):
        """int("4") should work."""
        assert int("4") == 4
        assert isinstance(int("4"), int)

    def test_int_coercion_accepts_numpy_int(self):
        """int(np.int64(4)) should be Python int."""
        assert int(np.int64(4)) == 4
        assert isinstance(int(np.int64(4)), int)

    def test_float_coercion_accepts_numeric_string(self):
        """float("0.05") should work."""
        result = float("0.05")
        assert result == pytest.approx(0.05)
        assert isinstance(result, float)

    def test_float_coercion_accepts_numpy_float(self):
        """float(np.float32(0.05)) should be Python float."""
        result = float(np.float32(0.05))
        assert isinstance(result, float)
        assert result == pytest.approx(0.05)

    def test_val_split_truncation_preserved(self):
        """int(0.059 * 100) == 5 (truncation, not rounding)."""
        val_split = 0.059
        threshold = int(val_split * 100)
        assert threshold == 5  # not 6

    def test_bool_coercion_accepts_int(self):
        """bool(1) → True, bool(0) → False."""
        assert bool(1) is True
        assert bool(0) is False

    def test_str_coercion_is_idempotent(self):
        """str() on a string is identity."""
        assert str("cuda:0") == "cuda:0"
        assert str("both") == "both"


# ─── Dataset boundary tests ────────────────────────────────────────────

class TestDatasetBoundaries:
    """Dataset type boundaries: metadata, sizes, caption dropout."""

    def test_dataset_item_types(self, dummy_metadata_jsonl):
        """DatasetItem has hazy/gt as tensors and caption as str."""
        from src.dehaze_lora.dataset import DehazeDataset
        ds = DehazeDataset(
            dummy_metadata_jsonl,
            caption_dropout_rate=0.1,
            target_size=512,
            dropout_seed=42,
        )
        item = ds[0]
        assert isinstance(item["hazy"], torch.Tensor)
        assert isinstance(item["gt"], torch.Tensor)
        assert isinstance(item["caption"], str)
        assert item["hazy"].shape == (3, 512, 512)
        assert item["gt"].shape == (3, 512, 512)

    def test_dataset_target_size_numpy_int(self):
        """target_size=np.int64(64) should produce (3, 64, 64) outputs."""
        from src.dehaze_lora.dataset import DehazeDataset

        # Create a tiny metadata file with a real image
        import tempfile
        from PIL import Image

        tmpdir = Path(tempfile.mkdtemp())
        img = Image.new("RGB", (128, 128), color=(128, 128, 128))
        img_path = tmpdir / "img.png"
        img.save(img_path)
        gt_path = tmpdir / "gt.png"
        img.save(gt_path)

        meta_path = tmpdir / "meta.jsonl"
        with open(meta_path, "w") as f:
            json.dump({"image": str(img_path), "gt": str(gt_path)}, f)
            f.write("\n")

        ds = DehazeDataset(
            str(meta_path),
            caption_dropout_rate=0.0,
            target_size=int(np.int64(64)),
        )
        item = ds[0]
        assert item["hazy"].shape == (3, 64, 64)
        assert item["gt"].shape == (3, 64, 64)

    def test_caption_dropout_full_rate(self):
        """caption_dropout_rate=1.0 should make ALL captions empty."""
        from src.dehaze_lora.dataset import DehazeDataset

        import tempfile
        from PIL import Image

        tmpdir = Path(tempfile.mkdtemp())
        img = Image.new("RGB", (128, 128), color=(128, 128, 128))
        img_path = tmpdir / "img.png"
        img.save(img_path)

        meta_path = tmpdir / "meta.jsonl"
        with open(meta_path, "w") as f:
            for i in range(10):
                json.dump({"image": str(img_path), "gt": str(img_path)}, f)
                f.write("\n")

        ds = DehazeDataset(
            str(meta_path),
            caption_dropout_rate=float(np.float32(1.0)),
            target_size=64,
            dropout_seed=42,
        )
        empty_count = sum(1 for i in range(len(ds)) if ds[i]["caption"] == "")
        assert empty_count == 10

    def test_val_dataset_fallback_to_prompt(self):
        """DehazeValDataset falls back to DEHAZE_PROMPT when no caption."""
        from src.dehaze_lora.dataset import DehazeValDataset, DEHAZE_PROMPT

        import tempfile
        from PIL import Image

        tmpdir = Path(tempfile.mkdtemp())
        img = Image.new("RGB", (128, 128), color=(128, 128, 128))
        img_path = tmpdir / "img.png"
        img.save(img_path)
        gt_path = tmpdir / "gt.png"
        img.save(gt_path)

        meta_path = tmpdir / "meta.jsonl"
        with open(meta_path, "w") as f:
            json.dump({"image": str(img_path), "gt": str(gt_path)}, f)
            f.write("\n")

        ds = DehazeValDataset(str(meta_path), target_size=64)
        item = ds[0]
        assert item["caption"] == DEHAZE_PROMPT


# ─── PathLike acceptance tests ─────────────────────────────────────────

class TestPathLikeAcceptance:
    """Functions accepting PathInput should accept str, Path, and os.PathLike."""

    def test_load_config_accepts_str(self, tmp_path):
        """load_config accepts a str path."""
        from src.dehaze_lora.utils import load_config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("key: value\n")
        result = load_config(str(config_path))
        assert result == {"key": "value"}

    def test_load_config_accepts_path(self, tmp_path):
        """load_config accepts a Path object."""
        from src.dehaze_lora.utils import load_config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("key: value\n")
        result = load_config(config_path)
        assert result == {"key": "value"}

    def test_save_config_accepts_path(self, tmp_path):
        """save_config accepts a Path object."""
        from src.dehaze_lora.utils import save_config
        path = tmp_path / "out.yaml"
        save_config({"key": "value"}, path)
        assert path.exists()

    def test_load_training_state_accepts_str(self, tmp_path):
        """load_training_state accepts str path to checkpoint dir."""
        from src.dehaze_lora.checkpoint import load_training_state
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        torch.save(
            {"global_step": 0, "micro_step": 0, "rng_states": {}, "optimizer_states": {}},
            ckpt / "training_state.pt",
        )
        state = load_training_state(str(ckpt))
        assert state["global_step"] == 0

    def test_load_training_state_accepts_path(self, tmp_path):
        """load_training_state accepts Path object."""
        from src.dehaze_lora.checkpoint import load_training_state
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        torch.save(
            {"global_step": 5, "micro_step": 3, "rng_states": {}, "optimizer_states": {}},
            ckpt / "training_state.pt",
        )
        state = load_training_state(ckpt)
        assert state["global_step"] == 5
        assert state["micro_step"] == 3


# ─── RNG state type tests ──────────────────────────────────────────────

class TestRNGStateTypes:
    """get_rng_state / set_rng_state work correctly."""

    def test_get_rng_state_keys(self):
        """get_rng_state returns expected keys."""
        from src.dehaze_lora.checkpoint import get_rng_state
        state = get_rng_state()
        assert "python_random" in state
        assert "numpy" in state
        assert "torch_cpu" in state
        assert "torch_cuda" in state

    def test_set_rng_state_roundtrip(self):
        """Capture → set → capture is idempotent for python random state."""
        import random
        from src.dehaze_lora.checkpoint import get_rng_state, set_rng_state

        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        state_before = get_rng_state()
        # Consume some entropy
        for _ in range(10):
            random.random()
            torch.randn(1)
        set_rng_state(state_before)

        # Verify Python state was restored exactly
        state_after = get_rng_state()
        assert state_before["python_random"] == state_after["python_random"]


# ─── Types module smoke tests ──────────────────────────────────────────

class TestTypesModule:
    """Verify the types module exports are importable and structurally correct."""

    def test_types_import(self):
        """All types can be imported."""
        from src.dehaze_lora.types import (
            MetadataItem,
            DatasetItem,
            ValidationImageOutput,
            ValidationBatchResult,
            MetricsDict,
            DenoiseOutput,
            PathInput,
            LoraTarget,
            ModelDict,
        )
        # Just verify they import successfully

    def test_typeddict_structural_conformance(self):
        """MetadataItem and ValidationBatchResult satisfy their structural constraints."""
        from src.dehaze_lora.types import MetadataItem, ValidationBatchResult

        # MetadataItem with minimal fields
        item: MetadataItem = {"image": "a.png", "gt": "b.png"}
        assert item["image"] == "a.png"

        # MetadataItem with optional caption
        with_caption: MetadataItem = {
            "image": "a.png", "gt": "b.png", "caption": "Dehaze",
        }
        assert with_caption["caption"] == "Dehaze"

    def test_tensor_item_float_types(self):
        """float(tensor.item()) chain works for 0-d tensor."""
        t = torch.tensor(0.5)
        result = float(t.item())
        assert isinstance(result, float)
        assert result == 0.5

    def test_tensor_0d_to_python_int(self):
        """0-d int tensor.item() → Python int."""
        t = torch.tensor(100)
        result = int(t.item())
        assert isinstance(result, int)
        assert result == 100
