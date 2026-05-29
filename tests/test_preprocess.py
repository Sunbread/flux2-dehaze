"""Tests for image preprocessing utilities."""

import json
from pathlib import Path

from PIL import Image
from dehaze_lora.preprocess import resize_and_save, _parse_path_list, _process_pair, DEHAZE_PROMPT


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


# ── _parse_path_list tests ──

def test_parse_single_path():
    result = _parse_path_list("hazy/1.png")
    assert result == ["hazy/1.png"]


def test_parse_bracketed_list():
    result = _parse_path_list("['hazy/1.png', 'hazy/2.png']")
    assert result == ["hazy/1.png", "hazy/2.png"]


def test_parse_bracketed_list_with_double_quotes():
    result = _parse_path_list('["hazy/1.png", "hazy/2.png"]')
    assert result == ["hazy/1.png", "hazy/2.png"]


# ── _process_pair test ──

def test_process_pair_creates_outputs_and_metadata(tmp_dir, rgb_image_256):
    """_process_pair creates output files and returns correct metadata."""
    hazy_dst = tmp_dir / "hazy_out.png"
    clear_dst = tmp_dir / "clear_out.png"

    # Need a source file for the "hazy" image
    src = tmp_dir / "src.png"
    rgb_image_256.save(src)

    args = (src, src, hazy_dst, clear_dst, 128, "test_pair")
    result = _process_pair(args)

    assert hazy_dst.exists()
    assert clear_dst.exists()
    assert result["image"] == str(hazy_dst)
    assert result["gt"] == str(clear_dst)
    assert result["caption"] == DEHAZE_PROMPT
