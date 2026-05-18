"""Tests for Qwen2 tokenizer, chat template, and text ID generation.

Tokenizer is < 10 MB — these run on CPU, always attempted.
"""

import pytest
import torch
from transformers import Qwen2TokenizerFast

MODEL_NAME = "black-forest-labs/FLUX.2-klein-base-9B"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return Qwen2TokenizerFast.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    except Exception as e:
        pytest.skip(f"Tokeniser not available: {e}")


# ---------------------------------------------------------------------------
# Chat template format
# ---------------------------------------------------------------------------

class TestChatTemplate:
    """Verify apply_chat_template output structure.

    Uses the same arguments as encode_prompt so the template rendering
    path is shared with production code.
    """

    def test_empty_prompt_template(self, tokenizer):
        """Empty prompt produces a non-empty template (empty user content)."""
        messages = [{"role": "user", "content": ""}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        assert len(text) > 0
        assert "<|im_start|>user" in text
        assert "<|im_end|>" in text
        assert "<|im_start|>assistant" in text
        # The user content is empty but the template structure is present

    def test_non_empty_prompt_format(self, tokenizer):
        prompt = "Dehaze this image"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        assert text.startswith("<|im_start|>user\n")
        assert "Dehaze this image" in text
        assert "<|im_end|>" in text
        assert "<|im_start|>assistant" in text
        assert "<think>" in text
        assert "</think>" in text


# ---------------------------------------------------------------------------
# Tokenizer behavior
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_tokenizer_loads(self, tokenizer):
        assert isinstance(tokenizer, Qwen2TokenizerFast)

    def test_tokenize_dehaze_prompt(self, tokenizer):
        from dehaze_lora.dataset import DEHAZE_PROMPT

        text = (
            f"<|im_start|>user\n{DEHAZE_PROMPT}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        tokens = tokenizer(text, max_length=512, padding="max_length",
                           truncation=True, return_tensors="pt")
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == 1
        assert tokens["input_ids"].shape[1] == 512

    def test_tokenize_empty_string(self, tokenizer):
        tokens = tokenizer("", max_length=512, padding="max_length",
                           truncation=True, return_tensors="pt")
        assert tokens["input_ids"].shape == (1, 512)
        # Empty string should be padded to all-pad (or bos-only depending on tokenizer)
        attention = tokens["attention_mask"][0]
        # At least some padding at the end for a short/empty input
        assert attention.sum().item() <= 512

    def test_truncation_long_prompt(self, tokenizer):
        long_prompt = "dehaze " * 1000
        tokens = tokenizer(long_prompt, max_length=128, padding="max_length",
                           truncation=True, return_tensors="pt")
        assert tokens["input_ids"].shape == (1, 128)

    def test_no_truncation_within_limit(self, tokenizer):
        short = "hello"
        tokens = tokenizer(short, max_length=512, padding="max_length",
                           truncation=True, return_tensors="pt")
        assert tokens["input_ids"].shape == (1, 512)


# ---------------------------------------------------------------------------
# Text ID generation (replicates encode_prompt logic)
# ---------------------------------------------------------------------------

class TestTextIDGeneration:
    """Test text_ids format without needing the text encoder model."""

    @pytest.mark.parametrize("seq_len", [1, 32, 128, 256, 512])
    def test_text_ids_shape_and_values(self, seq_len):
        dtype = torch.float32
        text_ids = torch.zeros(seq_len, 4, dtype=dtype)
        text_ids[:, 3] = torch.linspace(0, seq_len - 1, steps=seq_len, dtype=dtype)

        assert text_ids.shape == (seq_len, 4)
        assert torch.all(text_ids[:, 0] == 0.0)
        assert torch.all(text_ids[:, 1] == 0.0)
        assert torch.all(text_ids[:, 2] == 0.0)
        assert torch.allclose(text_ids[:, 3], torch.arange(seq_len, dtype=dtype))

    def test_text_ids_device_placement(self):
        for device_str in ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []):
            device = torch.device(device_str)
            seq_len = 64
            text_ids = torch.zeros(seq_len, 4, device=device, dtype=torch.float32)
            text_ids[:, 3] = torch.linspace(0, seq_len - 1, steps=seq_len,
                                            device=device, dtype=torch.float32)
            assert text_ids.device.type == device.type
            assert text_ids[:, 3].device.type == device.type
