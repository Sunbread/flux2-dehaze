"""Tests for image preprocessing utilities."""

from PIL import Image
from dehaze_lora.preprocess import resize_and_save, DEHAZE_PROMPT


def test_resize_and_save_creates_file(tmp_dir, rgb_image_256):
    dst = tmp_dir / "output.png"
    resize_and_save(rgb_image_256, dst, target_size=128)
    assert dst.exists()


def test_resize_and_save_target_size(tmp_dir, rgb_image_256):
    dst = tmp_dir / "resized.png"
    resize_and_save(rgb_image_256, dst, target_size=128)
    result = Image.open(dst)
    assert result.size == (128, 128)


def test_resize_and_save_rgb_mode(tmp_dir, rgb_image_256):
    dst = tmp_dir / "rgb.png"
    resize_and_save(rgb_image_256, dst)
    result = Image.open(dst)
    assert result.mode == "RGB"


def test_resize_and_save_default_size(tmp_dir, rgb_image_small):
    """Default target_size=512 produces 512x512 output."""
    dst = tmp_dir / "default.png"
    resize_and_save(rgb_image_small, dst)
    result = Image.open(dst)
    assert result.size == (512, 512)


def test_dehaze_prompt_content():
    assert "Dehaze" in DEHAZE_PROMPT
    assert "fog" in DEHAZE_PROMPT.lower()
    assert "photorealistic" in DEHAZE_PROMPT.lower()
