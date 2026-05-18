"""Tests for DehazeDataset and DehazeValDataset."""

import torch
from dehaze_lora.dataset import DehazeDataset, DehazeValDataset, DEHAZE_PROMPT


def test_dataset_returns_correct_keys(dummy_metadata_jsonl):
    ds = DehazeDataset(str(dummy_metadata_jsonl))
    item = ds[0]
    assert set(item.keys()) == {"hazy", "gt", "caption"}


def test_dataset_tensor_shapes(dummy_metadata_jsonl):
    ds = DehazeDataset(str(dummy_metadata_jsonl))
    item = ds[0]
    assert item["hazy"].shape == (3, 512, 512)
    assert item["gt"].shape == (3, 512, 512)
    assert item["hazy"].dtype == torch.float32
    assert item["gt"].dtype == torch.float32


def test_dataset_tensor_range(dummy_metadata_jsonl):
    ds = DehazeDataset(str(dummy_metadata_jsonl))
    for i in range(min(5, len(ds))):
        item = ds[i]
        assert item["hazy"].min() >= 0.0
        assert item["hazy"].max() <= 1.0
        assert item["gt"].min() >= 0.0
        assert item["gt"].max() <= 1.0


def test_dataset_length(dummy_metadata_jsonl):
    ds = DehazeDataset(str(dummy_metadata_jsonl))
    assert len(ds) == 20


def test_caption_is_string(dummy_metadata_jsonl):
    ds = DehazeDataset(str(dummy_metadata_jsonl), caption_dropout_rate=0.0)
    item = ds[0]
    assert isinstance(item["caption"], str)
    assert len(item["caption"]) > 0


def test_caption_dropout_produces_empty_strings(dummy_metadata_jsonl):
    """With dropout_rate=1.0, ALL captions should be empty."""
    ds = DehazeDataset(str(dummy_metadata_jsonl), caption_dropout_rate=1.0)
    for i in range(len(ds)):
        assert ds[i]["caption"] == ""


def test_caption_dropout_zero_never_empty(dummy_metadata_jsonl):
    """With dropout_rate=0.0, NO captions should be empty."""
    ds = DehazeDataset(str(dummy_metadata_jsonl), caption_dropout_rate=0.0)
    for i in range(len(ds)):
        assert ds[i]["caption"] != ""


def test_caption_dropout_statistical(dummy_metadata_jsonl):
    """With dropout_rate=0.5, roughly half should be empty over many samples."""
    ds = DehazeDataset(str(dummy_metadata_jsonl), caption_dropout_rate=0.5)
    empty_count = sum(1 for i in range(len(ds)) if ds[i]["caption"] == "")
    # With 20 samples at rate 0.5: binomial CI ~ [3, 17]
    assert 2 <= empty_count <= 18


def test_val_dataset_always_has_caption(dummy_metadata_jsonl):
    ds = DehazeValDataset(str(dummy_metadata_jsonl))
    for i in range(len(ds)):
        assert ds[i]["caption"] != ""


def test_val_dataset_returns_correct_keys(dummy_metadata_jsonl):
    ds = DehazeValDataset(str(dummy_metadata_jsonl))
    item = ds[0]
    assert set(item.keys()) == {"hazy", "gt", "caption"}


def test_val_dataset_tensor_shapes(dummy_metadata_jsonl):
    ds = DehazeValDataset(str(dummy_metadata_jsonl))
    item = ds[0]
    assert item["hazy"].shape == (3, 512, 512)
    assert item["gt"].shape == (3, 512, 512)


def test_target_size_actually_works(dummy_metadata_jsonl):
    """target_size parameter changes output resolution (not silently ignored)."""
    ds = DehazeDataset(str(dummy_metadata_jsonl), caption_dropout_rate=0.0,
                       target_size=256)
    item = ds[0]
    assert item["hazy"].shape == (3, 256, 256)
    assert item["gt"].shape == (3, 256, 256)


def test_val_target_size_actually_works(dummy_metadata_jsonl):
    ds = DehazeValDataset(str(dummy_metadata_jsonl), target_size=128)
    item = ds[0]
    assert item["hazy"].shape == (3, 128, 128)
    assert item["gt"].shape == (3, 128, 128)


def test_dehaze_prompt_constant():
    assert len(DEHAZE_PROMPT) > 50
    assert "Dehaze" in DEHAZE_PROMPT or "dehaze" in DEHAZE_PROMPT.lower()
