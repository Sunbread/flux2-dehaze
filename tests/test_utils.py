"""Tests for config utilities."""

import pytest
from pathlib import Path
from dehaze_lora.utils import load_config, save_config


def test_load_config(tmp_dir):
    config_path = tmp_dir / "config.yaml"
    config_path.write_text("key1: value1\nkey2: 42\n")
    result = load_config(config_path)
    assert result == {"key1": "value1", "key2": 42}


def test_load_config_accepts_str(tmp_dir):
    """load_config accepts a str path (boundary: str, not just Path)."""
    config_path = tmp_dir / "config.yaml"
    config_path.write_text("key: value\n")
    result = load_config(str(config_path))
    assert result == {"key": "value"}


def test_load_config_accepts_path(tmp_dir):
    """load_config accepts a Path object."""
    config_path = tmp_dir / "config.yaml"
    config_path.write_text("key: value\n")
    result = load_config(config_path)
    assert result == {"key": "value"}


def test_save_config(tmp_dir):
    config = {"a": 1, "b": [2, 3], "c": {"nested": "yes"}}
    path = tmp_dir / "out.yaml"
    save_config(config, path)
    assert path.exists()
    reloaded = load_config(path)
    assert reloaded == config


def test_save_config_accepts_path(tmp_dir):
    """save_config accepts a Path object."""
    path = tmp_dir / "out.yaml"
    save_config({"key": "value"}, path)
    assert path.exists()


def test_config_roundtrip_preserves_types(tmp_dir):
    config = {
        "lr": 1e-3,
        "seed": 42,
        "name": "test",
        "nested": {"flag": True, "none_val": None},
    }
    path = tmp_dir / "roundtrip.yaml"
    save_config(config, path)
    reloaded = load_config(path)
    assert reloaded == config


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")
