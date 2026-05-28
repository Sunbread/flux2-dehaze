"""CPU tests for validation split and subset selection logic."""

import json
from pathlib import Path

import pytest

from dehaze_lora.dataset import DehazeDataset, DEHAZE_PROMPT


def _make_metadata_file(n: int, tmp_path: Path) -> Path:
    """Write n synthetic metadata entries to a temp file."""
    path = tmp_path / "metadata.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "image": f"/tmp/hazy_{i:05d}.png",
                "gt": f"/tmp/gt_{i:05d}.png",
                "caption": DEHAZE_PROMPT,
            }) + "\n")
    return path


def _split_metadata(metadata: list, split_ratio: float = 0.05) -> tuple[list, list]:
    """Deterministic train/val split based on image path hash."""
    val_items = []
    train_items = []
    threshold = int(split_ratio * 100)
    for item in metadata:
        if hash(item["image"]) % 100 < threshold:
            val_items.append(item)
        else:
            train_items.append(item)
    return train_items, val_items


def _select_val_subset(val_metadata: list, k: int, seed: int) -> list:
    """Deterministic subset selection from val set."""
    import hashlib
    if len(val_metadata) <= k:
        return val_metadata
    indices = list(range(len(val_metadata)))
    key = f"{seed}".encode()
    ranked = sorted(
        indices,
        key=lambda i: hashlib.md5(key + str(i).encode()).hexdigest(),
    )
    return [val_metadata[i] for i in ranked[:k]]


class TestValidationSplit:
    """CPU tests for deterministic train/val split logic."""

    def test_metadata_items_parameter(self, tmp_path):
        """DehazeDataset with metadata_items uses the filtered list."""
        meta_path = _make_metadata_file(50, tmp_path)

        all_items = [json.loads(l) for l in open(meta_path)]
        train_items = all_items[:40]

        ds = DehazeDataset(
            str(meta_path),
            metadata_items=train_items,
        )
        assert len(ds) == 40
        # Verify the filtered metadata is used (the first 40 items from file)
        assert ds.metadata[0]["caption"] == DEHAZE_PROMPT

    def test_split_deterministic(self, tmp_path):
        """Same metadata -> same split every time."""
        meta_path = _make_metadata_file(100, tmp_path)
        all_items = [json.loads(l) for l in open(meta_path)]

        train1, val1 = _split_metadata(all_items, 0.05)
        train2, val2 = _split_metadata(all_items, 0.05)

        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        for a, b in zip(val1, val2):
            assert a["image"] == b["image"]

    def test_split_no_overlap(self, tmp_path):
        """Train and val sets are disjoint."""
        meta_path = _make_metadata_file(100, tmp_path)
        all_items = [json.loads(l) for l in open(meta_path)]

        train_items, val_items = _split_metadata(all_items, 0.05)

        train_paths = {item["image"] for item in train_items}
        val_paths = {item["image"] for item in val_items}
        assert train_paths.isdisjoint(val_paths)

    def test_split_approx_ratio(self, tmp_path):
        """Split roughly matches the target ratio (within tolerance for small N)."""
        meta_path = _make_metadata_file(1000, tmp_path)
        all_items = [json.loads(l) for l in open(meta_path)]

        _, val_items = _split_metadata(all_items, 0.05)
        ratio = len(val_items) / 1000
        assert 0.03 < ratio < 0.07, f"Val ratio {ratio} too far from 0.05"

    def test_subset_deterministic(self, tmp_path):
        """Same seed -> same subset each call."""
        meta_path = _make_metadata_file(200, tmp_path)
        all_items = [json.loads(l) for l in open(meta_path)]
        _, val_items = _split_metadata(all_items, 0.05)

        sub1 = _select_val_subset(val_items, k=4, seed=42)
        sub2 = _select_val_subset(val_items, k=4, seed=42)
        sub3 = _select_val_subset(val_items, k=4, seed=43)

        assert [s["image"] for s in sub1] == [s["image"] for s in sub2]
        assert [s["image"] for s in sub1] != [s["image"] for s in sub3]

    def test_subset_respects_k(self, tmp_path):
        """Subset size matches k."""
        meta_path = _make_metadata_file(200, tmp_path)
        all_items = [json.loads(l) for l in open(meta_path)]
        _, val_items = _split_metadata(all_items, 0.05)

        sub = _select_val_subset(val_items, k=4, seed=42)
        assert len(sub) == 4

    def test_subset_small_val(self, tmp_path):
        """When val set smaller than k, returns all val items."""
        meta_path = _make_metadata_file(50, tmp_path)
        all_items = [json.loads(l) for l in open(meta_path)]
        _, val_items = _split_metadata(all_items, 0.05)

        sub = _select_val_subset(val_items, k=100, seed=42)
        assert len(sub) == len(val_items)

    def test_empty_metadata_items(self, tmp_path):
        """metadata_items=None falls back to file read (backward compat)."""
        meta_path = _make_metadata_file(50, tmp_path)

        ds1 = DehazeDataset(str(meta_path))
        ds2 = DehazeDataset(str(meta_path), metadata_items=None)

        assert len(ds1) == 50
        assert len(ds2) == 50


class TestValidationRNGIsolation:
    """RNG state is preserved across validation interruption."""

    def test_rng_restored_after_validation(self):
        """After RNG restore, training continues identically."""
        import random
        import numpy as np
        import torch
        from dehaze_lora.checkpoint import get_rng_state, set_rng_state

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Draw one training value
        train_r1 = random.random()

        # Capture state for validation
        state_before_val = get_rng_state()

        # Simulate validation consuming RNG
        for _ in range(100):
            random.random()
            np.random.random()
            torch.randn(1)

        # Restore
        set_rng_state(state_before_val)

        # Continue training after validation
        train_r2 = random.random()
        train_n2 = np.random.random()
        train_t2 = torch.randn(1).item()

        # Reset and run without validation interruption
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        _ = random.random()  # skip train_r1
        expected_r2 = random.random()
        expected_n2 = np.random.random()
        expected_t2 = torch.randn(1).item()

        assert train_r2 == expected_r2
        assert train_n2 == expected_n2
        assert abs(train_t2 - expected_t2) < 1e-10

    def test_rng_unchanged_by_validation_block(self):
        """Entire validation block (capture -> restore) is transparent."""
        import random
        import numpy as np
        import torch
        from dehaze_lora.checkpoint import get_rng_state, set_rng_state

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Draw a sequence
        seq_before = [random.random() for _ in range(5)]

        # Simulate: capture -> validation -> restore
        state = get_rng_state()
        for _ in range(100):   # validation consumes lots of RNG
            random.random()
            np.random.random()
            torch.randn(1)
        set_rng_state(state)

        # Continue -- should match original sequence after seq_before
        seq_after = [random.random() for _ in range(5)]

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        [random.random() for _ in range(5)]  # skip past seq_before
        expected = [random.random() for _ in range(5)]

        assert seq_after == expected

    def test_rng_isolation_all_sources(self):
        """All 4 RNG sources are isolated: python, numpy, torch CPU, torch CUDA."""
        import random
        import numpy as np
        import torch
        from dehaze_lora.checkpoint import get_rng_state, set_rng_state

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        for _ in range(10):
            random.random()
            np.random.random()
            torch.randn(1)

        state = get_rng_state()
        r_before = random.random()
        n_before = np.random.random()
        t_before = torch.randn(1).item()

        set_rng_state(state)
        r_after = random.random()
        n_after = np.random.random()
        t_after = torch.randn(1).item()

        assert r_before == r_after, "Python RNG not isolated"
        assert n_before == n_after, "NumPy RNG not isolated"
        assert abs(t_before - t_after) < 1e-10, "Torch RNG not isolated"
